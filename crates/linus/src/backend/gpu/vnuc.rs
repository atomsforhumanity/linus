//! The nuclei's attraction V = -sum_C Z_C <a| 1/|r - C| |b> on the device, for large molecules
//! (`backend.nuclear_attraction`: the plan's rule, `Plan::nuclear_attraction_on_device`): the
//! device PCM engine's pair x point integrals ([`linus_gpu::pcm`]) with the charges as points of
//! zero width (q = Z_eff, the external point charges among them; scale -1), every pair with
//! primitives kept (pair cut 0), the integral context's pairs reused. The pairs' data are packed on the host beside the
//! device J/K setup ([`device_job`]); the pass runs after it ([`DeviceVnuc::compute`]). V agrees
//! with linus-ints' `OneEPlan::potential` to round-off (ubiquitin/def2-SVP 7.7e-13 of 444 Eh;
//! u1an39/def2-TZVP 4.3e-14) and is bitwise the same run to run. The overlap, kinetic energy and
//! ECPs stay on the host.

use crate::backend::gpu::device_error_in;
use crate::backend::{NuclearAttraction, Setup};
use faer::Mat;
use linus_base::error::LinusError;
use linus_gpu::pcm::{GpuPcmInts, PcmIntsHost, PcmPoint};
use linus_gpu::{Gpu, GpuError};
use linus_ints::{BasisSet, PairSet};
use std::sync::Arc;

/// The name of this device work in errors.
const WHAT: &str = "nuclear_attraction";

/// A packed device V, ready for its pass.
pub(crate) struct DeviceVnuc {
    gpu: Arc<Gpu>,
    host: PcmIntsHost,
    q: Vec<f64>,
    nbf: usize,
}

/// The host part of the device V of `charges` ((q, C), the electrons see -q / |r - C|) over the
/// pairs `pairs` of `basis`: declined with the reason where the device's kernels do not cover
/// the basis (Cartesian d and higher shells, l > 4); any other error the calculation's.
pub(crate) fn device_job(
    gpu: &Arc<Gpu>,
    basis: &BasisSet,
    pairs: &PairSet,
    charges: &[(f64, [f64; 3])],
) -> Result<Setup<DeviceVnuc>, LinusError> {
    let points: Vec<PcmPoint> = charges
        .iter()
        .map(|&(_, r)| PcmPoint { r, zeta: 0.0 })
        .collect();
    let host = match PcmIntsHost::with_pairs(basis, Some(pairs), &points, 0.0, false) {
        Ok(h) => h,
        Err(e) if e.falls_back_at_setup() => return Ok(Setup::Declined(e.to_string())),
        Err(e) => return Err(device_error_in(WHAT, e)),
    };
    Ok(Setup::Ready(DeviceVnuc {
        gpu: gpu.clone(),
        host,
        q: charges.iter().map(|&(q, _)| q).collect(),
        nbf: basis.nbf,
    }))
}

impl DeviceVnuc {
    /// V (nbf x nbf): the device's pass over the packed pairs and the charges.
    fn run(&self) -> Result<Mat<f64>, GpuError> {
        let gpu = &self.gpu;
        let n = self.nbf;
        let mut ints = GpuPcmInts::new(gpu, &self.host)?;
        let q = gpu.upload(&self.q)?;
        let mut m = gpu.alloc_zeros::<f64>(n * n)?;
        ints.pass(gpu, None, None, None, Some(&q), Some((&mut m, -1.0)))?;
        let v = gpu.download(&m)?;
        // column-major and symmetric
        Ok(Mat::from_fn(n, n, |i, j| v[i + j * n]))
    }
}

impl NuclearAttraction for DeviceVnuc {
    fn compute(self: Box<Self>) -> Result<Setup<Mat<f64>>, LinusError> {
        match self.run() {
            Ok(v) => Ok(Setup::Ready(v)),
            // set up after the J/K, in the memory it left: the host's V where that is too little
            Err(e @ GpuError::OutOfMemory { .. }) => Ok(Setup::Declined(e.to_string())),
            Err(e) => Err(device_error_in(WHAT, e)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use linus_ints::{AoMatrixMut, BasisDef, DerivOrder, OneEPlan, PairOptions};

    /// The device V against linus-ints' host V, element by element, and bitwise run to run:
    /// u1an39/def2-TZVP (its nuclei), CH2I/def2-SVP (iodine's ECP: Z_eff 25) with two external
    /// point charges.
    #[test]
    fn the_device_nuclear_attraction_follows_the_host() {
        let Some(gpu) = crate::tests::harness::gpu_or_skip().map(Arc::new) else {
            return;
        };
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        for (file, basis_name, extra) in [
            ("bench/molecules/u1an39.xyz", "def2-tzvp", false),
            ("tests/molecules/ch2i.xyz", "def2-svp", true),
        ] {
            let text = std::fs::read_to_string(root.join(file)).unwrap();
            let (z, pos) = linus_core::structure::parse_xyz(&text).unwrap();
            let def = BasisDef::load(basis_name, None).unwrap();
            let basis = BasisSet::new(&def, &z, &pos, true).unwrap();
            let n = basis.nbf;
            let mut charges: Vec<(f64, [f64; 3])> = z
                .iter()
                .zip(&pos)
                .map(|(&zi, &p)| {
                    let core = def.ecp_electrons.get(&zi).copied().unwrap_or(0);
                    ((zi - core) as f64, p)
                })
                .collect();
            if extra {
                charges.push((0.5, [3.0, 0.0, 1.0]));
                charges.push((-0.3, [-4.0, 1.0, 2.0]));
            }
            let pairs = PairSet::build(&basis, PairOptions::with_order(DerivOrder::Zero));
            let plan = OneEPlan::new(&basis, DerivOrder::Zero)
                .with_charges(&pairs, &charges)
                .unwrap();
            let mut host = vec![0.0; n * n];
            plan.potential(&mut AoMatrixMut::row_major(&mut host, n));
            let Setup::Ready(job) = device_job(&gpu, &basis, &pairs, &charges).unwrap() else {
                panic!("{file}: the device declined");
            };
            let a = job.run().unwrap();
            let b = job.run().unwrap();
            let mut worst: f64 = 0.0;
            let mut scale: f64 = 0.0;
            for i in 0..n {
                for j in 0..n {
                    assert_eq!(
                        a[(i, j)].to_bits(),
                        b[(i, j)].to_bits(),
                        "{file}: run to run"
                    );
                    worst = worst.max((a[(i, j)] - host[i * n + j]).abs());
                    scale = scale.max(host[i * n + j].abs());
                }
            }
            // measured: 4.3e-14 (u1an39/def2-TZVP), 1e-14 (CH2I); ubiquitin 1.7e-15 relative
            assert!(worst <= 1e-14 * scale, "{file}: {worst:e} of {scale:e}");
        }
    }
}
