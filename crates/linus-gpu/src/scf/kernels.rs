//! The SCF loop's own kernels: element-wise matrix updates (Fock assembly, the level shift,
//! the DIIS extrapolation, the orthogonalizer's columns) and batched reductions (dot products,
//! largest magnitudes, non-finite counts).
//!
//! Rounding follows linus's CPU loop operation for operation: products and sums are written
//! with `__dmul_rn`/`__dadd_rn`/`__dsub_rn`, which the compiler never contracts into an FMA,
//! so an element-wise update of the same inputs gives the CPU's bits.
//!
//! Reductions are deterministic and compensated. Every task of a batch is reduced by
//! [`RED_BLOCKS`] blocks of [`RED_THREADS`] threads, whatever the batch and the device: each
//! thread sums a fixed strided set of products as a double-double (TwoSum, as the CPU's
//! Neumaier sum `util::linalg::dot`), the block combines its threads in a fixed tree in shared
//! memory, and one block per task combines the partial sums in a fixed tree. The result is
//! the sum of the rounded products to about 1 ulp, like the CPU's, and bitwise the same run
//! to run on any device. No shuffles and no warp-size assumption.

/// Blocks per reduction task (fixed: the partition never depends on the batch or the device).
pub const RED_BLOCKS: u32 = 128;
/// Threads per reduction block.
pub const RED_THREADS: u32 = 256;
/// Threads per block of the element-wise kernels.
pub const EW_THREADS: u32 = 256;

/// Reduction modes of a task.
pub const MODE_DOT: u64 = 0;
pub const MODE_MAXABS: u64 = 1;
pub const MODE_NONFINITE: u64 = 2;

/// Words per task in the reduction table: x, y, n, mode.
pub const TASK_WORDS: usize = 4;

/// The CUDA C source (the device layer pastes the portability header in front).
pub const SOURCE: &str = include_str!("../../device/scf/kernels.cu");

#[cfg(feature = "gpu")]
pub use device::*;

#[cfg(feature = "gpu")]
mod device {
    use super::*;
    use crate::error::GpuError;
    use crate::kernels::registry::KernelModule;
    use crate::linalg::{Blas1, Operand, Scale};
    use crate::{Arg, DeviceBuf, Gpu, Kernel, Launch};
    use std::borrow::Cow;

    /// The device loop's module ([`SOURCE`]).
    pub(crate) const MODULE: KernelModule = KernelModule::new(
        "scf",
        || Cow::Borrowed(SOURCE),
        &[
            "scf_reduce_partial",
            "scf_reduce_finish",
            "scf_add",
            "scf_fock",
            "scf_antisym",
            "scf_level_shift",
            "scf_combine",
            "scf_orth_columns",
            "scf_factor_columns",
        ],
    );

    /// A sub-array of a device buffer: (buffer, first element).
    pub type At<'a> = (&'a DeviceBuf<f64>, usize);

    fn addr(b: At<'_>, n: usize) -> Result<u64, GpuError> {
        match b.1.checked_add(n) {
            Some(end) if end <= b.0.len() => Ok(b.0.at(b.1).addr()),
            _ => Err(GpuError::Contract(format!(
                "elements {}..{} outside a buffer of {}",
                b.1,
                b.1.saturating_add(n),
                b.0.len()
            ))),
        }
    }

    /// One reduction of a batch: operands (device addresses), length and mode.
    #[derive(Clone, Copy, Debug)]
    pub struct Task {
        x: u64,
        y: u64,
        n: usize,
        mode: u64,
    }

    impl Task {
        /// sum x_i y_i over n elements.
        pub fn dot(x: At<'_>, y: At<'_>, n: usize) -> Result<Task, GpuError> {
            Ok(Task {
                x: addr(x, n)?,
                y: addr(y, n)?,
                n,
                mode: MODE_DOT,
            })
        }

        /// max |x_i| over n elements (NaN is ignored: count it with [`Task::non_finite`]).
        pub fn max_abs(x: At<'_>, n: usize) -> Result<Task, GpuError> {
            Ok(Task {
                x: addr(x, n)?,
                y: 0,
                n,
                mode: MODE_MAXABS,
            })
        }

        /// The number of non-finite values among n elements.
        pub fn non_finite(x: At<'_>, n: usize) -> Result<Task, GpuError> {
            Ok(Task {
                x: addr(x, n)?,
                y: 0,
                n,
                mode: MODE_NONFINITE,
            })
        }
    }

    /// The loop's compiled kernels and the reduction workspace.
    pub struct Ops {
        reduce_partial: Kernel,
        reduce_finish: Kernel,
        add: Kernel,
        blas1: Blas1,
        fock: Kernel,
        antisym: Kernel,
        level_shift: Kernel,
        combine: Kernel,
        orth_columns: Kernel,
        factor_columns: Kernel,
        /// the task table (TASK_WORDS per task), the partial results, the combination table
        tab: DeviceBuf<u64>,
        part: DeviceBuf<f64>,
        ctab: DeviceBuf<u64>,
        max_tasks: usize,
    }

    fn ew(n: usize) -> Launch {
        Launch::linear(n, EW_THREADS)
    }

    fn len(n: usize) -> Arg {
        Arg::I64(n as i64)
    }

    fn opt(b: Option<&DeviceBuf<f64>>, n: usize) -> Result<Arg, GpuError> {
        Ok(Arg::Ptr(match b {
            Some(b) => addr((b, 0), n)?,
            None => 0,
        }))
    }

    impl Ops {
        /// Compiles (or loads from the kernel cache) the kernels; batches of up to `max_tasks`
        /// reductions and combinations of up to `max_terms` matrices.
        pub fn new(gpu: &Gpu, max_tasks: usize, max_terms: usize) -> Result<Ops, GpuError> {
            let src = MODULE.source();
            let k = |entry: &str| gpu.kernel(MODULE.name, &src, entry);
            Ok(Ops {
                reduce_partial: k("scf_reduce_partial")?,
                reduce_finish: k("scf_reduce_finish")?,
                add: k("scf_add")?,
                blas1: Blas1::new(gpu)?,
                fock: k("scf_fock")?,
                antisym: k("scf_antisym")?,
                level_shift: k("scf_level_shift")?,
                combine: k("scf_combine")?,
                orth_columns: k("scf_orth_columns")?,
                factor_columns: k("scf_factor_columns")?,
                tab: gpu.alloc_zeros(TASK_WORDS * max_tasks.max(1))?,
                part: gpu.alloc_zeros(2 * RED_BLOCKS as usize * max_tasks.max(1))?,
                ctab: gpu.alloc_zeros(2 * max_terms.max(1))?,
                max_tasks: max_tasks.max(1),
            })
        }

        /// Device bytes of the workspace.
        pub fn bytes(&self) -> usize {
            8 * (self.tab.len() + self.part.len() + self.ctab.len())
        }

        /// Queues the reductions `tasks`; the result of task t lands in `out[t]`.
        pub fn reduce(
            &mut self,
            gpu: &Gpu,
            tasks: &[Task],
            out: &mut DeviceBuf<f64>,
        ) -> Result<(), GpuError> {
            if tasks.is_empty() {
                return Ok(());
            }
            if tasks.len() > self.max_tasks || out.len() < tasks.len() {
                return Err(GpuError::Contract(format!(
                    "reduction batch of {} tasks (at most {}, {} result slots)",
                    tasks.len(),
                    self.max_tasks,
                    out.len()
                )));
            }
            let mut words = Vec::with_capacity(TASK_WORDS * tasks.len());
            for t in tasks {
                words.extend_from_slice(&[t.x, t.y, t.n as u64, t.mode]);
            }
            gpu.upload_into(&mut self.tab, 0, &words)?;
            let nt = tasks.len() as u32;
            // SAFETY: the table holds nt tasks whose operands lie in live device buffers (Task's
            // constructors check the ranges); part holds 2 RED_BLOCKS doubles per task, out at
            // least nt.
            unsafe {
                gpu.launch(
                    &self.reduce_partial,
                    Launch {
                        grid: (RED_BLOCKS, nt, 1),
                        block: (RED_THREADS, 1, 1),
                        shared_bytes: 0,
                    },
                    &[Arg::buf(&self.tab), Arg::buf_mut(&mut self.part)],
                )?;
                gpu.launch(
                    &self.reduce_finish,
                    Launch {
                        grid: (nt, 1, 1),
                        block: (RED_BLOCKS, 1, 1),
                        shared_bytes: 0,
                    },
                    &[Arg::buf(&self.tab), Arg::buf(&self.part), Arg::buf_mut(out)],
                )
            }
        }

        /// out = a + beta b over n elements.
        pub fn add(
            &self,
            gpu: &Gpu,
            n: usize,
            a: At<'_>,
            b: At<'_>,
            beta: f64,
            out: (&mut DeviceBuf<f64>, usize),
        ) -> Result<(), GpuError> {
            let (pa, pb) = (addr(a, n)?, addr(b, n)?);
            let po = addr((&*out.0, out.1), n)?;
            // SAFETY: n elements of each operand (checked).
            unsafe {
                gpu.launch(
                    &self.add,
                    ew(n),
                    &[
                        len(n),
                        Arg::Ptr(pa),
                        Arg::Ptr(pb),
                        Arg::F64(beta),
                        Arg::Ptr(po),
                    ],
                )
            }
        }

        /// x = alpha x over the first n elements.
        pub fn scale(
            &self,
            gpu: &Gpu,
            n: usize,
            x: &mut DeviceBuf<f64>,
            alpha: f64,
        ) -> Result<(), GpuError> {
            self.blas1
                .scale(gpu, n, Scale::Scalar(alpha), Operand::Out, x)
        }

        /// F_s = ((h + j) - k_s) + (vxc_s + venv) for `nsp` spins back to back (h, j, venv:
        /// one n2 matrix; k, vxc: nsp of them; k, vxc and venv optional).
        #[allow(clippy::too_many_arguments)]
        pub fn fock(
            &self,
            gpu: &Gpu,
            nsp: usize,
            n2: usize,
            h: &DeviceBuf<f64>,
            j: &DeviceBuf<f64>,
            k: Option<&DeviceBuf<f64>>,
            vxc: Option<&DeviceBuf<f64>>,
            venv: Option<&DeviceBuf<f64>>,
            f: &mut DeviceBuf<f64>,
        ) -> Result<(), GpuError> {
            let nt = nsp * n2;
            let args = [
                len(nt),
                len(n2),
                Arg::Ptr(addr((h, 0), n2)?),
                Arg::Ptr(addr((j, 0), n2)?),
                opt(k, nt)?,
                opt(vxc, nt)?,
                opt(venv, n2)?,
                Arg::Ptr(addr((&*f, 0), nt)?),
            ];
            // SAFETY: every operand checked to hold its elements; null pointers are skipped.
            unsafe { gpu.launch(&self.fock, ew(nt), &args) }
        }

        /// e = a - a^T for the n x n matrix a.
        pub fn antisym(
            &self,
            gpu: &Gpu,
            n: usize,
            a: &DeviceBuf<f64>,
            e: &mut DeviceBuf<f64>,
        ) -> Result<(), GpuError> {
            let args = [
                len(n),
                Arg::Ptr(addr((a, 0), n * n)?),
                Arg::Ptr(addr((&*e, 0), n * n)?),
            ];
            // SAFETY: n^2 elements of both (checked).
            unsafe { gpu.launch(&self.antisym, ew(n * n), &args) }
        }

        /// f_s += shift (s - sds_s) in place, for `nsp` spins back to back (s one n2 matrix).
        #[allow(clippy::too_many_arguments)]
        pub fn level_shift(
            &self,
            gpu: &Gpu,
            nsp: usize,
            n2: usize,
            s: &DeviceBuf<f64>,
            sds: &DeviceBuf<f64>,
            shift: f64,
            f: &mut DeviceBuf<f64>,
        ) -> Result<(), GpuError> {
            let nt = nsp * n2;
            let pf = addr((&*f, 0), nt)?;
            let args = [
                len(nt),
                len(n2),
                Arg::Ptr(pf),
                Arg::Ptr(addr((s, 0), n2)?),
                Arg::Ptr(addr((sds, 0), nt)?),
                Arg::F64(shift),
                Arg::Ptr(pf),
            ];
            // SAFETY: every operand checked to hold its elements; each thread reads and writes
            // its own element of f.
            unsafe { gpu.launch(&self.level_shift, ew(nt), &args) }
        }

        /// out = sum_k c_k F_k over n elements, summed in the order of `terms`.
        pub fn combine(
            &mut self,
            gpu: &Gpu,
            n: usize,
            terms: &[(&DeviceBuf<f64>, f64)],
            out: &mut DeviceBuf<f64>,
        ) -> Result<(), GpuError> {
            let m = terms.len();
            if 2 * m > self.ctab.len() {
                return Err(GpuError::Contract(format!(
                    "a combination of {m} matrices (at most {})",
                    self.ctab.len() / 2
                )));
            }
            let mut words = Vec::with_capacity(2 * m);
            for t in terms {
                words.push(addr((t.0, 0), n)?);
            }
            words.extend(terms.iter().map(|t| t.1.to_bits()));
            gpu.upload_into(&mut self.ctab, 0, &words)?;
            let po = addr((&*out, 0), n)?;
            // SAFETY: the table holds m operands of n elements (checked) and their coefficients.
            unsafe {
                gpu.launch(
                    &self.combine,
                    ew(n),
                    &[
                        len(n),
                        Arg::buf(&self.ctab),
                        Arg::I32(m as i32),
                        Arg::Ptr(po),
                    ],
                )
            }
        }

        /// x[:, c] = u[:, first + c] / sqrt(w[first + c]) for c < ncol (u n x n, x n x ncol).
        #[allow(clippy::too_many_arguments)]
        pub fn orth_columns(
            &self,
            gpu: &Gpu,
            n: usize,
            ncol: usize,
            first: usize,
            u: &DeviceBuf<f64>,
            w: &DeviceBuf<f64>,
            x: &mut DeviceBuf<f64>,
        ) -> Result<(), GpuError> {
            if first + ncol > n {
                return Err(GpuError::Contract(
                    "orthogonalizer: columns out of range".into(),
                ));
            }
            let args = [
                len(n),
                len(ncol),
                len(first),
                Arg::Ptr(addr((u, 0), n * n)?),
                Arg::Ptr(addr((w, 0), n)?),
                Arg::Ptr(addr((&*x, 0), n * ncol)?),
            ];
            // SAFETY: the columns read and written lie in the buffers (checked).
            unsafe { gpu.launch(&self.orth_columns, ew(n * ncol), &args) }
        }

        /// x\[:, c\] = u\[:, first + c\] sqrt(sign w\[first + c\]) for c < ncol into x at element
        /// `xoff` (u n x n; sign +1 for the positive part of a factorization, -1 for the
        /// negative one).
        #[allow(clippy::too_many_arguments)]
        pub fn factor_columns(
            &self,
            gpu: &Gpu,
            n: usize,
            ncol: usize,
            first: usize,
            u: &DeviceBuf<f64>,
            w: &DeviceBuf<f64>,
            sign: f64,
            x: (&mut DeviceBuf<f64>, usize),
        ) -> Result<(), GpuError> {
            if first + ncol > n {
                return Err(GpuError::Contract("factor: columns out of range".into()));
            }
            if ncol == 0 {
                return Ok(());
            }
            let args = [
                len(n),
                len(ncol),
                len(first),
                Arg::Ptr(addr((u, 0), n * n)?),
                Arg::Ptr(addr((w, 0), n)?),
                Arg::F64(sign),
                Arg::Ptr(addr((&*x.0, x.1), n * ncol)?),
            ];
            // SAFETY: the columns read and written lie in the buffers (checked).
            unsafe { gpu.launch(&self.factor_columns, ew(n * ncol), &args) }
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn source_is_portable() {
        let src = super::SOURCE;
        assert!(!src.contains("asm"), "no inline PTX");
        assert!(
            !src.contains("__shfl"),
            "no shuffles: the trees are in shared memory"
        );
        assert!(!src.contains("atomic"), "no atomics: fixed reduction trees");
        assert!(!src.contains("32"), "no warp-size literal");
        // the reduction constants agree with the host's
        assert!(src.contains(&format!("#define RED_THREADS {}", super::RED_THREADS)));
        assert!(src.contains(&format!("#define RED_BLOCKS {}", super::RED_BLOCKS)));
    }
}
