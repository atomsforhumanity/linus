//! The second-order assembly's stacked GEMMs on the device (linus-scf's [`Gram`]: hr += f A^T B
//! over the matrices vectorized as columns, n nocc long), for a Hessian whose products ran on
//! the device: the stacks uploaded, one cuBLAS GEMM, the 3N x 3N product added on the host
//! (its sums in cuBLAS's order: the host's to round-off). Where the device runs out of memory
//! the host's GEMM runs; `response[].assembly` records where they ran, and why on the CPU.

use crate::backend::gpu::device_error_in;
use crate::driver::backend::ResponseRecords;
use crate::driver::report::PlaceRecord;
use faer::Mat;
use linus_base::error::LinusError;
use linus_gpu::blas::{GemmDims, MatMut, MatRef, Op};
use linus_gpu::linalg::{Arith, Gemm};
use linus_gpu::{Gpu, GpuError};
use linus_scf::response::second_order::Gram;
use std::sync::{Arc, Mutex, OnceLock};

/// The name of this device work in errors.
const WHAT: &str = "the response's assembly";

/// The device's stacked GEMMs of a response's assembly, with the response's record.
pub(crate) struct DeviceGram {
    gpu: Arc<Gpu>,
    /// The GEMM service (one cuBLAS handle), made at the first GEMM.
    gemm: OnceLock<Mutex<Gemm>>,
    records: (ResponseRecords, &'static str),
}

impl DeviceGram {
    /// The assembly's GEMMs on `gpu`, recorded in `records` (the responses' records and the
    /// response's purpose).
    pub(crate) fn new(gpu: Arc<Gpu>, records: (ResponseRecords, &'static str)) -> Self {
        DeviceGram {
            gpu,
            gemm: OnceLock::new(),
            records,
        }
    }

    fn run(
        &self,
        hr: &mut Mat<f64>,
        f: f64,
        a: &[&Mat<f64>],
        b: &[&Mat<f64>],
    ) -> Result<(), GpuError> {
        let len = a.first().map_or(0, |m| m.nrows() * m.ncols());
        let (na, nb) = (a.len(), b.len());
        if len == 0 || na == 0 || nb == 0 {
            return Ok(());
        }
        // column c: the columns of v[c] one after another (as stacked_gram's stacks)
        let stack = |v: &[&Mat<f64>]| -> Vec<f64> {
            let mut out = Vec::with_capacity(len * v.len());
            for m in v {
                for j in 0..m.ncols() {
                    out.extend_from_slice(m.col_as_slice(j));
                }
            }
            out
        };
        let gpu = &self.gpu;
        let gemm = match self.gemm.get() {
            Some(g) => g,
            None => {
                let g = Gemm::new(gpu)?;
                self.gemm.get_or_init(|| Mutex::new(g))
            }
        };
        let da = gpu.upload(&stack(a))?;
        let db = gpu.upload(&stack(b))?;
        let mut g = gpu.alloc_zeros::<f64>(na * nb)?;
        gemm.lock().unwrap_or_else(|e| e.into_inner()).gemm(
            Arith::Fp64,
            GemmDims::new(Op::T, Op::N, na, nb, len),
            (f, 0.0),
            MatRef::new(&da, 0, len),
            MatRef::new(&db, 0, len),
            MatMut::new(&mut g, 0, na),
        )?;
        let gh = gpu.download(&g)?;
        for y in 0..nb {
            for x in 0..na {
                hr[(x, y)] += gh[y * na + x];
            }
        }
        Ok(())
    }
}

/// Records where a GEMM of the assembly of `purpose` ran: a GEMM the device declined leaves
/// `cpu` with its reason for the rest.
fn record(records: &ResponseRecords, purpose: &str, at: PlaceRecord) {
    if let Some(r) = records
        .lock()
        .iter_mut()
        .rev()
        .find(|r| r.purpose == purpose)
    {
        if r.assembly.as_ref().is_none_or(|a| a.place == "gpu") {
            r.assembly = Some(at);
        }
    }
}

impl Gram for DeviceGram {
    fn gram(
        &self,
        hr: &mut Mat<f64>,
        f: f64,
        a: &[&Mat<f64>],
        b: &[&Mat<f64>],
    ) -> Result<bool, LinusError> {
        let (records, purpose) = &self.records;
        match self.run(hr, f, a, b) {
            Ok(()) => {
                record(records, purpose, PlaceRecord::gpu());
                Ok(true)
            }
            // nothing added: the host's GEMM runs
            Err(e @ GpuError::OutOfMemory { .. }) => {
                record(
                    records,
                    purpose,
                    PlaceRecord::cpu(format!("the device's memory ({e}): the host's GEMM")),
                );
                Ok(false)
            }
            Err(e) => Err(device_error_in(WHAT, e)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use faer::Par;
    use linus_scf::response::second_order::{refs, stacked_gram};

    /// The device's GEMM adds what the host's stacked_gram adds, to round-off, bitwise the same
    /// run to run, and records where it ran.
    #[test]
    fn the_device_gram_follows_the_host() {
        let Some(gpu) = crate::tests::harness::gpu_or_skip() else {
            return;
        };
        let records = ResponseRecords::default();
        records.start("hessian", None, "a test".into());
        let dg = DeviceGram::new(Arc::new(gpu), (records.clone(), "hessian"));
        let stacks = |seed: usize| -> Vec<Mat<f64>> {
            (0..12)
                .map(|k| {
                    Mat::from_fn(40, 7, |i, j| {
                        ((seed * 31 + k * 17 + i * 5 + j * 3) as f64 * 0.37).sin()
                    })
                })
                .collect()
        };
        let (a, b) = (stacks(1), stacks(2));
        let (ra, rb) = (refs(&a), refs(&b));
        let mut host = Mat::<f64>::zeros(12, 12);
        stacked_gram(&mut host, -4.0, &ra, &rb, Par::Seq);
        let run = || {
            let mut d = Mat::<f64>::zeros(12, 12);
            assert!(dg.gram(&mut d, -4.0, &ra, &rb).unwrap());
            d
        };
        let (d1, d2) = (run(), run());
        let mut worst: f64 = 0.0;
        for y in 0..12 {
            for x in 0..12 {
                assert_eq!(d1[(x, y)].to_bits(), d2[(x, y)].to_bits());
                worst = worst.max((d1[(x, y)] - host[(x, y)]).abs());
            }
        }
        assert!(worst < 1e-12, "{worst:e}");
        assert_eq!(records.lock()[0].assembly, Some(PlaceRecord::gpu()));
    }
}
