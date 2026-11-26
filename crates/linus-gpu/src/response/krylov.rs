//! The device's algebra of the shared-subspace solver of linear response (CPHF/CPKS): the
//! subspace V, its products A V, the right-hand sides B, the residuals R and the new directions
//! resident on the device, so the long GEMMs of every iteration (projections against V, the new
//! rows of V^T A V, the residuals B - A V c) run on cuBLAS instead of the host. The iteration
//! itself (which right-hand sides stay active, the projected system and its solve, convergence)
//! is linus-scf's one driver (`linus_scf::response::solver::subspace_solve_with`), which the
//! facade hands this algebra: `KrylovSpace`'s methods (feature `gpu`) are its steps.
//!
//! Reductions (norms, max-abs) are the device loop's deterministic compensated ones
//! (`crate::scf::kernels`), GEMMs cuBLAS's: a solve is bitwise reproducible run to run on a
//! device. Its results agree with the host's algebra to round-off (other GEMM orders), not bit
//! for bit.
//!
//! Memory ([`device_bytes`], exactly what it allocates): V and A V for `cap` columns, B, R and
//! the new directions (len x nr each), the coefficients and the projections (cap x nr each), the
//! preconditioner, the reductions' workspace; at the end, with the iterations' buffers freed,
//! the combined extras and one block of them. All of it from the caller's grant. A subspace that
//! outgrows `cap` grows by reallocation outside the grant, which then gives the rest of its
//! bytes back to the device's pool and is not drawn from again.

#[cfg(feature = "gpu")]
pub use device::*;

/// The largest Gram-Schmidt block the reductions' workspace is sized for (linus-scf's
/// `GS_BLOCK`; a larger block is a contract error).
pub const GS_BLOCK_MAX: usize = 16;

/// Device bytes of a `KrylovSpace` with `len` unknowns, `nr` right-hand sides, a subspace of
/// `cap` columns and extras of `elen` elements combined in blocks of `extras_block` columns: the
/// larger of the iterations' allocations and the final combination's, beside what both keep.
pub fn device_bytes(len: usize, nr: usize, cap: usize, elen: usize, extras_block: usize) -> usize {
    let cap = cap.max(nr).max(1);
    let (n, l) = (nr.max(1), len.max(1));
    let tasks = reduction_tasks(nr);
    // kept to the end: the reductions' table and workspace, the column indices, the
    // coefficients
    let kept = 8 * (tasks + n + cap * n) + crate::scf::kernels::ops_bytes(tasks, 1);
    // the iterations': B, R, the new directions, V, A V, the preconditioner, the projections
    let iterations = 8 * (3 * l * n + 2 * l * cap + l + cap * n);
    // the end's: the combined extras and one block of them
    let e = elen.max(1);
    let extras = 8 * (e * n + e * extras_block.min(cap).max(1));
    kept + iterations.max(extras)
}

/// Reductions per batch: a max-abs and a non-finite count per residual column, or the norms of
/// a Gram-Schmidt block's columns.
fn reduction_tasks(nr: usize) -> usize {
    2 * nr.max(GS_BLOCK_MAX + 1)
}

#[cfg(feature = "gpu")]
mod device {
    use super::{reduction_tasks, GS_BLOCK_MAX};
    use crate::blas::{GemmDims, MatMut, MatRef, Op};
    use crate::error::GpuError;
    use crate::kernels::registry::KernelModule;
    use crate::linalg::{Arith, Gemm};
    use crate::pool::Grant;
    use crate::scf::kernels::{ops_bytes, Ops, Task};
    use crate::{Arg, DeviceBuf, Gpu, Kernel, Launch};
    use std::borrow::Cow;

    const SOURCE: &str = include_str!("../../device/response/krylov.cu");

    /// The solver's module.
    pub(crate) const MODULE: KernelModule = KernelModule::new(
        "krylov",
        || Cow::Borrowed(SOURCE),
        &["krylov_gather_div", "krylov_div"],
    );

    const THREADS: u32 = 256;

    /// The buffers of the iterations, freed before the extras are combined.
    struct Iterations {
        /// B, R, the new directions (len x nr each)
        bd: DeviceBuf<f64>,
        r: DeviceBuf<f64>,
        zm: DeviceBuf<f64>,
        /// V and A V (len x cap)
        vm: DeviceBuf<f64>,
        avm: DeviceBuf<f64>,
        /// the preconditioner's diagonal (len)
        dd: DeviceBuf<f64>,
        /// projections against V (m x na) and within a block (k x nb) (cap x nr)
        p: DeviceBuf<f64>,
    }

    /// The device's algebra of one shared-subspace solve (module docs): its steps in the
    /// driver's order. Columns are column-major (leading dimension len); `m0` is the
    /// subspace's size before an iteration's new directions.
    pub struct KrylovSpace<'g> {
        gpu: &'g Gpu,
        /// Drawn from until a subspace outgrows it.
        grant: Option<&'g Grant>,
        gemm: Gemm,
        ops: Ops,
        gather_div: Kernel,
        div: Kernel,
        scal: DeviceBuf<f64>,
        idx: DeviceBuf<i64>,
        /// the coefficients c (cap x nr)
        cd: DeviceBuf<f64>,
        /// None after the extras' combination
        it: Option<Iterations>,
        len: usize,
        nr: usize,
        cap: usize,
        /// the iteration's new directions (active right-hand sides)
        na: usize,
        /// the subspace's size of the last coefficients
        m: usize,
    }

    /// A step after [`KrylovSpace::combine_extras`], the last one.
    fn after_end() -> GpuError {
        GpuError::Contract("KrylovSpace: a step after the extras' combination".into())
    }

    /// A device buffer of `n` doubles, from `grant` while it lasts.
    fn alloc(gpu: &Gpu, grant: Option<&Grant>, n: usize) -> Result<DeviceBuf<f64>, GpuError> {
        gpu.alloc_zeros_granted(grant, n.max(1))
    }

    impl<'g> KrylovSpace<'g> {
        /// The algebra of A X = B for the `nr` right-hand sides `b` (len x nr, column-major)
        /// with the Jacobi preconditioner `diag` (len), its subspace for up to `cap` columns
        /// (more by reallocation), its memory ([`super::device_bytes`]) drawn from `grant`.
        pub fn new(
            gpu: &'g Gpu,
            b: &[f64],
            nr: usize,
            diag: &[f64],
            cap: usize,
            grant: Option<&'g Grant>,
        ) -> Result<Self, GpuError> {
            let len = diag.len();
            if b.len() != len * nr {
                return Err(GpuError::Contract(format!(
                    "KrylovSpace: B has {} elements, not {len} x {nr}",
                    b.len()
                )));
            }
            let cap = cap.max(nr).max(1);
            let tasks = reduction_tasks(nr);
            // the reductions' workspace is the device loop's (Ops allocates it): drawn from the
            // grant before it is made
            if let Some(g) = grant {
                g.draw(ops_bytes(tasks, 1));
            }
            let src = MODULE.source();
            let bd = gpu.upload_granted(grant, b)?;
            let mut r = alloc(gpu, grant, len * nr)?;
            gpu.copy(&bd, &mut r)?;
            let it = Iterations {
                dd: gpu.upload_granted(grant, diag)?,
                zm: alloc(gpu, grant, len * nr)?,
                vm: alloc(gpu, grant, len * cap)?,
                avm: alloc(gpu, grant, len * cap)?,
                p: alloc(gpu, grant, cap * nr)?,
                bd,
                r,
            };
            Ok(KrylovSpace {
                gpu,
                grant,
                gemm: Gemm::new(gpu)?,
                ops: Ops::new(gpu, tasks, 1)?,
                gather_div: gpu.kernel(MODULE.name, &src, "krylov_gather_div")?,
                div: gpu.kernel(MODULE.name, &src, "krylov_div")?,
                scal: alloc(gpu, grant, tasks)?,
                idx: gpu.alloc_zeros_granted(grant, nr.max(1))?,
                cd: alloc(gpu, grant, cap * nr)?,
                it: Some(it),
                len,
                nr,
                cap,
                na: 0,
                m: 0,
            })
        }

        /// The max-abs of each of the `nr` residual columns, NaN where a column holds a
        /// non-finite value.
        pub fn residual_max(&mut self) -> Result<Vec<f64>, GpuError> {
            let (len, nr) = (self.len, self.nr);
            let mut out = Vec::with_capacity(nr);
            // batches of at most half the table (a max and a count per column)
            let per = (self.scal.len() / 2).max(1);
            for k0 in (0..nr).step_by(per) {
                let k1 = (k0 + per).min(nr);
                let r = &self.it.as_ref().ok_or_else(after_end)?.r;
                let mut tasks = Vec::with_capacity(2 * (k1 - k0));
                for k in k0..k1 {
                    tasks.push(Task::max_abs((r, k * len), len)?);
                    tasks.push(Task::non_finite((r, k * len), len)?);
                }
                let v = reduce(self.gpu, &mut self.ops, &mut self.scal, &tasks)?;
                for p in v.chunks(2) {
                    out.push(if p[1] != 0.0 { f64::NAN } else { p[0] });
                }
            }
            Ok(out)
        }

        /// The new directions: the residuals of the `active` columns divided by the
        /// preconditioner's diagonal; their norms.
        pub fn precondition(&mut self, active: &[usize]) -> Result<Vec<f64>, GpuError> {
            let (gpu, len, na) = (self.gpu, self.len, active.len());
            let ia: Vec<i64> = active.iter().map(|&k| k as i64).collect();
            gpu.upload_into(&mut self.idx, 0, &ia)?;
            self.na = na;
            let it = self.it.as_mut().ok_or_else(after_end)?;
            // SAFETY: matches krylov_gather_div; idx holds na column indices < nr of r
            // (len x nr), dd has len elements, zm len x nr >= na columns.
            unsafe {
                gpu.launch(
                    &self.gather_div,
                    Launch::linear(len * na, THREADS),
                    &[
                        Arg::I64(len as i64),
                        Arg::I64(na as i64),
                        Arg::buf(&self.idx),
                        Arg::buf(&it.r),
                        Arg::buf(&it.dd),
                        Arg::buf_mut(&mut it.zm),
                    ],
                )?;
            }
            self.col_norms(0, na)
        }

        /// Projects the subspace's `m0` columns out of the new directions, twice.
        pub fn project_out(&mut self, m0: usize) -> Result<(), GpuError> {
            let (len, na) = (self.len, self.na);
            if m0 == 0 {
                return Ok(());
            }
            let it = self.it.as_mut().ok_or_else(after_end)?;
            for _ in 0..2 {
                gemm(
                    &self.gemm,
                    Op::T,
                    Op::N,
                    (m0, na, len),
                    (1.0, 0.0),
                    (&it.vm, 0, len),
                    (&it.zm, 0, len),
                    (&mut it.p, 0, m0),
                )?;
                gemm(
                    &self.gemm,
                    Op::N,
                    Op::N,
                    (len, na, m0),
                    (-1.0, 1.0),
                    (&it.vm, 0, len),
                    (&it.p, 0, m0),
                    (&mut it.zm, 0, len),
                )?;
            }
            Ok(())
        }

        /// Orthonormalizes the new directions among themselves into V's columns from `m0` on
        /// (block classical Gram-Schmidt in blocks of `block` columns, a direction dropped when
        /// its norm falls to `drop_ratio` of `z0` or 0) and returns how many it kept; V grows
        /// first when they would not fit.
        pub fn orthonormalize(
            &mut self,
            m0: usize,
            z0: &[f64],
            block: usize,
            drop_ratio: f64,
        ) -> Result<usize, GpuError> {
            if block == 0 || block > GS_BLOCK_MAX {
                return Err(GpuError::Contract(format!(
                    "KrylovSpace: Gram-Schmidt blocks of {block} columns (1 to {GS_BLOCK_MAX})"
                )));
            }
            let (gpu, len, na) = (self.gpu, self.len, self.na);
            if m0 + na > self.cap {
                self.grow(m0, (2 * self.cap).max(m0 + na))?;
            }
            let mut k = 0usize;
            for b0 in (0..na).step_by(block) {
                let nb = block.min(na - b0);
                if k > 0 {
                    let it = self.it.as_mut().ok_or_else(after_end)?;
                    for _ in 0..2 {
                        gemm(
                            &self.gemm,
                            Op::T,
                            Op::N,
                            (k, nb, len),
                            (1.0, 0.0),
                            (&it.vm, m0 * len, len),
                            (&it.zm, b0 * len, len),
                            (&mut it.p, 0, k),
                        )?;
                        gemm(
                            &self.gemm,
                            Op::N,
                            Op::N,
                            (len, nb, k),
                            (-1.0, 1.0),
                            (&it.vm, m0 * len, len),
                            (&it.p, 0, k),
                            (&mut it.zm, b0 * len, len),
                        )?;
                    }
                }
                let kb = k;
                for j in 0..nb {
                    let zoff = (b0 + j) * len;
                    if k > kb {
                        let it = self.it.as_mut().ok_or_else(after_end)?;
                        for _ in 0..2 {
                            gemm(
                                &self.gemm,
                                Op::T,
                                Op::N,
                                (k - kb, 1, len),
                                (1.0, 0.0),
                                (&it.vm, (m0 + kb) * len, len),
                                (&it.zm, zoff, len),
                                (&mut it.p, 0, k - kb),
                            )?;
                            gemm(
                                &self.gemm,
                                Op::N,
                                Op::N,
                                (len, 1, k - kb),
                                (-1.0, 1.0),
                                (&it.vm, (m0 + kb) * len, len),
                                (&it.p, 0, k - kb),
                                (&mut it.zm, zoff, len),
                            )?;
                        }
                    }
                    let nz = self.col_norms(b0 + j, 1)?[0];
                    if nz > drop_ratio * z0[b0 + j] && nz > 0.0 {
                        let it = self.it.as_mut().ok_or_else(after_end)?;
                        let zcol = it.zm.at(zoff).addr();
                        let vcol = it.vm.at((m0 + k) * len).addr();
                        // SAFETY: matches krylov_div; len elements of zm from zoff and of vm
                        // from column m0 + k < cap.
                        unsafe {
                            gpu.launch(
                                &self.div,
                                Launch::linear(len, THREADS),
                                &[
                                    Arg::I64(len as i64),
                                    Arg::Ptr(zcol),
                                    Arg::F64(nz),
                                    Arg::Ptr(vcol),
                                ],
                            )?;
                        }
                        k += 1;
                    }
                }
            }
            Ok(k)
        }

        /// V, A V, the projections and the coefficients for `ncap` columns, V's and A V's first
        /// `m0` kept: outside the grant, which gives the rest of its bytes back to the pool
        /// (the bytes of the buffers freed here among them) and is not drawn from again.
        fn grow(&mut self, m0: usize, ncap: usize) -> Result<(), GpuError> {
            let (gpu, len, nr) = (self.gpu, self.len, self.nr);
            let mut v2 = gpu.alloc_zeros::<f64>(len * ncap)?;
            let mut a2 = gpu.alloc_zeros::<f64>(len * ncap)?;
            let p2 = gpu.alloc_zeros::<f64>(ncap * nr)?;
            let c2 = gpu.alloc_zeros::<f64>(ncap * nr)?;
            let it = self.it.as_mut().ok_or_else(after_end)?;
            gpu.copy_range(&it.vm, 0, &mut v2, 0, len * m0)?;
            gpu.copy_range(&it.avm, 0, &mut a2, 0, len * m0)?;
            (it.vm, it.avm, it.p) = (v2, a2, p2);
            self.cd = c2;
            self.cap = ncap;
            if let Some(g) = self.grant.take() {
                g.draw(g.remaining());
            }
            Ok(())
        }

        /// The `nn` kept new directions (V's columns m0..m0 + nn), downloaded.
        pub fn new_directions(&mut self, m0: usize, nn: usize) -> Result<Vec<Vec<f64>>, GpuError> {
            let len = self.len;
            let it = self.it.as_ref().ok_or_else(after_end)?;
            let qn = self.gpu.download_range(&it.vm, m0 * len, nn * len)?;
            Ok(qn.chunks(len).map(|c| c.to_vec()).collect())
        }

        /// A z of the new directions (`az`: their vectors one after another) into A V's columns
        /// from `m0` on.
        pub fn append_products(&mut self, m0: usize, az: &[f64]) -> Result<(), GpuError> {
            let len = self.len;
            let it = self.it.as_mut().ok_or_else(after_end)?;
            self.gpu.upload_into(&mut it.avm, m0 * len, az)
        }

        /// The projected system's new parts, column-major: V^T A V_new (m x nn), V_new^T A
        /// V_old (nn x m0) and V_new^T B (nn x nr), V_new the columns m0..m0 + nn, m = m0 + nn.
        #[allow(
            clippy::type_complexity,
            reason = "three named blocks of one projection"
        )]
        pub fn new_rows(
            &mut self,
            m0: usize,
            nn: usize,
        ) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>), GpuError> {
            let (gpu, len, nr) = (self.gpu, self.len, self.nr);
            let m = m0 + nn;
            let it = self.it.as_mut().ok_or_else(after_end)?;
            gemm(
                &self.gemm,
                Op::T,
                Op::N,
                (m, nn, len),
                (1.0, 0.0),
                (&it.vm, 0, len),
                (&it.avm, m0 * len, len),
                (&mut it.p, 0, m),
            )?;
            let x = gpu.download_range(&it.p, 0, m * nn)?;
            let y = if m0 > 0 {
                gemm(
                    &self.gemm,
                    Op::T,
                    Op::N,
                    (nn, m0, len),
                    (1.0, 0.0),
                    (&it.vm, m0 * len, len),
                    (&it.avm, 0, len),
                    (&mut it.p, 0, nn),
                )?;
                gpu.download_range(&it.p, 0, nn * m0)?
            } else {
                Vec::new()
            };
            gemm(
                &self.gemm,
                Op::T,
                Op::N,
                (nn, nr, len),
                (1.0, 0.0),
                (&it.vm, m0 * len, len),
                (&it.bd, 0, len),
                (&mut it.p, 0, nn),
            )?;
            let vbn = gpu.download_range(&it.p, 0, nn * nr)?;
            Ok((x, y, vbn))
        }

        /// R = B - A V c for the coefficients `c` (m x nr, column-major).
        pub fn residuals(&mut self, c: &[f64], m: usize) -> Result<(), GpuError> {
            let (gpu, len, nr) = (self.gpu, self.len, self.nr);
            gpu.upload_into(&mut self.cd, 0, c)?;
            self.m = m;
            let it = self.it.as_mut().ok_or_else(after_end)?;
            gpu.copy(&it.bd, &mut it.r)?;
            gemm(
                &self.gemm,
                Op::N,
                Op::N,
                (len, nr, m),
                (-1.0, 1.0),
                (&it.avm, 0, len),
                (&self.cd, 0, m),
                (&mut it.r, 0, len),
            )
        }

        /// X = V c of the last coefficients (len x nr, column-major; zeros for an empty
        /// subspace).
        pub fn solution(&mut self) -> Result<Vec<f64>, GpuError> {
            let (gpu, len, nr, m) = (self.gpu, self.len, self.nr, self.m);
            if m == 0 {
                return Ok(vec![0.0; len * nr]);
            }
            let it = self.it.as_mut().ok_or_else(after_end)?;
            gemm(
                &self.gemm,
                Op::N,
                Op::N,
                (len, nr, m),
                (1.0, 0.0),
                (&it.vm, 0, len),
                (&self.cd, 0, m),
                (&mut it.zm, 0, len),
            )?;
            gpu.download_range(&it.zm, 0, len * nr)
        }

        /// sum_j c_jk e_j of the extras `ex` (`elen` each, one per subspace column) with the
        /// last coefficients (elen x nr, column-major), in blocks of `block` columns; the
        /// iterations' buffers are freed first (the last step).
        pub fn combine_extras(
            &mut self,
            ex: &[Vec<f64>],
            elen: usize,
            block: usize,
        ) -> Result<Vec<f64>, GpuError> {
            let (gpu, nr, m) = (self.gpu, self.nr, self.m);
            drop(self.it.take());
            if elen == 0 || m == 0 {
                return Ok(vec![0.0; elen * nr]);
            }
            let mut fd = alloc(gpu, self.grant, elen * nr)?;
            let bs = block.min(m).max(1);
            let mut eb = alloc(gpu, self.grant, elen * bs)?;
            let mut stage = vec![0.0; elen * bs];
            for j0 in (0..m).step_by(bs) {
                let j1 = (j0 + bs).min(m);
                for (j, e) in ex[j0..j1].iter().enumerate() {
                    stage[j * elen..(j + 1) * elen].copy_from_slice(e);
                }
                gpu.upload_into(&mut eb, 0, &stage[..(j1 - j0) * elen])?;
                gemm(
                    &self.gemm,
                    Op::N,
                    Op::N,
                    (elen, nr, j1 - j0),
                    (1.0, 1.0),
                    (&eb, 0, elen),
                    (&self.cd, j0, m),
                    (&mut fd, 0, elen),
                )?;
            }
            gpu.download_range(&fd, 0, elen * nr)
        }

        /// The norms of the `nc` new-direction columns from column `j0`.
        fn col_norms(&mut self, j0: usize, nc: usize) -> Result<Vec<f64>, GpuError> {
            let len = self.len;
            let mut out = Vec::with_capacity(nc);
            let per = self.scal.len().max(1);
            for a in (0..nc).step_by(per) {
                let b = (a + per).min(nc);
                let z = &self.it.as_ref().ok_or_else(after_end)?.zm;
                let tasks = (a..b)
                    .map(|j| {
                        let off = (j0 + j) * len;
                        Task::dot((z, off), (z, off), len)
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                let v = reduce(self.gpu, &mut self.ops, &mut self.scal, &tasks)?;
                out.extend(v.into_iter().map(f64::sqrt));
            }
            Ok(out)
        }
    }

    /// The reductions `tasks` into `scal`, downloaded.
    fn reduce(
        gpu: &Gpu,
        ops: &mut Ops,
        scal: &mut DeviceBuf<f64>,
        tasks: &[Task],
    ) -> Result<Vec<f64>, GpuError> {
        if tasks.is_empty() {
            return Ok(Vec::new());
        }
        ops.reduce(gpu, tasks, scal)?;
        gpu.download_range(scal, 0, tasks.len())
    }

    /// C = alpha op(A) op(B) + beta C on column-major operands (buffer, first element, leading
    /// dimension), in FP64.
    #[allow(
        clippy::too_many_arguments,
        reason = "a GEMM's operands, each with its offset and leading dimension"
    )]
    fn gemm(
        gemm: &Gemm,
        ta: Op,
        tb: Op,
        (m, n, k): (usize, usize, usize),
        (alpha, beta): (f64, f64),
        a: (&DeviceBuf<f64>, usize, usize),
        b: (&DeviceBuf<f64>, usize, usize),
        c: (&mut DeviceBuf<f64>, usize, usize),
    ) -> Result<(), GpuError> {
        if m == 0 || n == 0 {
            return Ok(());
        }
        gemm.gemm(
            Arith::Fp64,
            GemmDims::new(ta, tb, m, n, k),
            (alpha, beta),
            MatRef::new(a.0, a.1, a.2.max(1)),
            MatRef::new(b.0, b.1, b.2.max(1)),
            MatMut::new(c.0, c.1, c.2.max(1)),
        )
    }
}
