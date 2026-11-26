//! The CPHF/CPKS solver's own algebra on the device (`linus_gpu::response::krylov`) behind
//! linus-scf's [`SubspaceSolver`]: a Hessian whose response J/K runs on the device solves its
//! response equations there too, the subspace, its products and the residuals resident on the
//! device, so the long projections and residual GEMMs of every iteration leave the host. The
//! iteration is linus-scf's one driver ([`subspace_solve_with`]: the steps, the stopping rule,
//! the progress events, the caller's stop request, the projected m x m system solved on the
//! host); [`DeviceAlgebra`] is its algebra on the device.
//!
//! Its memory (the subspace for [`CAP_ITERATIONS`] iterations of every right-hand side, the
//! right-hand sides, residuals and combined extras) is granted after the response's engines are
//! set up, from what they left beyond the products' workspaces per call: the stored exchange
//! operator keeps its own plan (a solver that took its device memory first could push the
//! operator's streamed part beyond the host memory it may use, and the exchange back onto the
//! tensor at several times the cost: ts61/def2-TZVP on the H200). Where it does not fit, the
//! host's solver runs (`resolved.record.backend.response`'s `solver`: `cpu`, with the reason); a
//! subspace that outgrows the grant and then the device goes back to the host's solver from the
//! start (recorded the same way). A draw beyond the grant is the response's `plan_violation`.

use crate::backend::gpu::device_error;
use crate::backend::{Setup, SolverShape};
use crate::driver::backend::{Family, ResponseRecords};
use crate::driver::report::PlaceRecord;
use faer::{Mat, Par};
use linus_base::error::LinusError;
use linus_gpu::pool::Grant;
use linus_gpu::response::krylov::{self, KrylovSpace};
use linus_gpu::{Gpu, GpuError};
use linus_scf::response::solver::{
    subspace_solve, subspace_solve_with, Apply, Solution, SolveControl, SolveError,
    SubspaceAlgebra, SubspaceSolver, DROP_RATIO, EXTRAS_BLOCK, GS_BLOCK,
};
use std::sync::Arc;

/// The subspace the solver's grant holds: this many full iterations of every right-hand side
/// (a Hessian's CPKS converges in 5-8 at the default tolerance, fewer columns per iteration as
/// they converge).
pub(crate) const CAP_ITERATIONS: usize = 8;

/// The device's solver for a response of `shape` (module docs), or why the host's runs: set
/// up after the response's engines (the J/K with its stored operator, the CPKS kernel), in the
/// device memory they left beyond `reserve` (the products' workspaces per call: the CPKS
/// kernel's and the exchange's).
pub(crate) fn setup(
    gpu: &Arc<Gpu>,
    shape: SolverShape,
    reserve: usize,
    records: (ResponseRecords, &'static str),
) -> Setup<Box<dyn SubspaceSolver>> {
    let cap = shape.nr * CAP_ITERATIONS;
    let need = krylov::device_bytes(shape.len, shape.nr, cap, shape.elen, EXTRAS_BLOCK);
    let free = gpu.budget().free();
    let gib = |b: usize| b as f64 / 1073741824.0;
    if need + reserve > free {
        return Setup::Declined(format!(
            "its subspace ({:.1} GiB) does not fit the {:.1} GiB the response's engines left \
             beside their {:.1} GiB reserve",
            gib(need),
            gib(free),
            gib(reserve)
        ));
    }
    match gpu.grant(need, "the CPHF/CPKS solver") {
        Ok(grant) => Setup::Ready(Box::new(DeviceKrylov {
            gpu: gpu.clone(),
            grant: Some(grant),
            cap,
            records,
        })),
        Err(e) => Setup::Declined(format!("its subspace did not fit the device: {e}")),
    }
}

/// The solver of [`setup`].
struct DeviceKrylov {
    gpu: Arc<Gpu>,
    /// Held from the setup to the solve's end.
    grant: Option<Grant>,
    cap: usize,
    records: (ResponseRecords, &'static str),
}

/// linus-scf's [`SubspaceAlgebra`] on the device's [`KrylovSpace`] (faer matrices to and from
/// its column-major vectors).
pub(crate) struct DeviceAlgebra<'g> {
    space: KrylovSpace<'g>,
    len: usize,
    nr: usize,
}

impl<'g> DeviceAlgebra<'g> {
    /// The algebra of A X = `b` with the Jacobi preconditioner `diag` on `gpu`, its subspace
    /// for `cap` columns from `grant`.
    pub(crate) fn new(
        gpu: &'g Gpu,
        b: &Mat<f64>,
        diag: &[f64],
        cap: usize,
        grant: Option<&'g Grant>,
    ) -> Result<Self, GpuError> {
        let (len, nr) = (b.nrows(), b.ncols());
        let bs: Vec<&[f64]> = (0..nr).map(|k| b.col_as_slice(k)).collect();
        let bc = cols(&bs, len);
        Ok(DeviceAlgebra {
            space: KrylovSpace::new(gpu, &bc, nr, diag, cap, grant)?,
            len,
            nr,
        })
    }
}

/// The column-major `rows` x `cols` matrix of `v`.
fn mat(v: &[f64], rows: usize, cols: usize) -> Mat<f64> {
    Mat::from_fn(rows, cols, |i, j| v[j * rows + i])
}

/// The columns `c` (each of `len`) one after another, in one allocation (copied as slices: the
/// products are tens of MB per iteration).
fn cols<C: AsRef<[f64]>>(c: &[C], len: usize) -> Vec<f64> {
    let mut out = Vec::with_capacity(len * c.len());
    for x in c {
        out.extend_from_slice(x.as_ref());
    }
    out
}

impl SubspaceAlgebra for DeviceAlgebra<'_> {
    type Error = GpuError;

    fn residual_max(&mut self) -> Result<Vec<f64>, GpuError> {
        self.space.residual_max()
    }

    fn precondition(&mut self, active: &[usize]) -> Result<Vec<f64>, GpuError> {
        self.space.precondition(active)
    }

    fn project_out(&mut self, m0: usize) -> Result<(), GpuError> {
        self.space.project_out(m0)
    }

    fn orthonormalize(&mut self, m0: usize, z0: &[f64]) -> Result<usize, GpuError> {
        self.space.orthonormalize(m0, z0, GS_BLOCK, DROP_RATIO)
    }

    fn new_directions(&mut self, m0: usize, nn: usize) -> Result<Vec<Vec<f64>>, GpuError> {
        self.space.new_directions(m0, nn)
    }

    fn append_products(&mut self, m0: usize, az: Vec<Vec<f64>>) -> Result<(), GpuError> {
        self.space.append_products(m0, &cols(&az, self.len))
    }

    fn new_rows(
        &mut self,
        m0: usize,
        nn: usize,
    ) -> Result<(Mat<f64>, Mat<f64>, Mat<f64>), GpuError> {
        let (x, y, vbn) = self.space.new_rows(m0, nn)?;
        Ok((
            mat(&x, m0 + nn, nn),
            mat(&y, nn, m0),
            mat(&vbn, nn, self.nr),
        ))
    }

    fn residuals(&mut self, c: &Mat<f64>) -> Result<(), GpuError> {
        let cs: Vec<&[f64]> = (0..c.ncols()).map(|k| c.col_as_slice(k)).collect();
        self.space.residuals(&cols(&cs, c.nrows()), c.nrows())
    }

    fn solution(&mut self, _c: &Mat<f64>) -> Result<Mat<f64>, GpuError> {
        // the coefficients are the device's already (the last residuals')
        let x = self.space.solution()?;
        Ok(mat(&x, self.len, self.nr))
    }

    fn combine_extras(
        &mut self,
        ex: &[Vec<f64>],
        elen: usize,
        _c: &Mat<f64>,
    ) -> Result<Mat<f64>, GpuError> {
        let fx = self.space.combine_extras(ex, elen, EXTRAS_BLOCK)?;
        Ok(mat(&fx, elen, self.nr))
    }
}

impl SubspaceSolver for DeviceKrylov {
    fn solve(
        &mut self,
        b: &Mat<f64>,
        diag: &[f64],
        elen: usize,
        control: SolveControl,
        par: Par,
        apply: &mut Apply<'_>,
    ) -> Result<Solution, LinusError> {
        let (records, purpose) = &self.records;
        let out = DeviceAlgebra::new(&self.gpu, b, diag, self.cap, self.grant.as_ref())
            .map_err(SolveError::Algebra)
            .and_then(|mut alg| {
                subspace_solve_with(&mut alg, b.ncols(), elen, control.clone(), par, apply)
            });
        // the grant's draws beyond it, then its bytes back to the pool
        if let Some(v) = self
            .grant
            .take()
            .and_then(|g| g.violations().first().cloned())
        {
            if let Some(r) = records
                .lock()
                .iter_mut()
                .rev()
                .find(|r| r.purpose == *purpose)
            {
                r.plan_violation.get_or_insert_with(|| {
                    format!(
                        "{}: {} bytes drawn with {} of the {} granted left",
                        v.what, v.requested, v.remaining, v.granted
                    )
                });
            }
        }
        let e = match out {
            Ok(x) => return Ok(x),
            Err(SolveError::Solve(e)) => return Err(e),
            Err(SolveError::Algebra(e)) => e,
        };
        if !matches!(e, GpuError::OutOfMemory { .. }) {
            return Err(device_error(Family::ResponseJk, e));
        }
        // the subspace outgrew the device: the host's solver from the start
        records.set_solver(
            purpose,
            PlaceRecord::cpu(format!(
                "the device's subspace outgrew the device ({e}): solved again on the host"
            )),
        );
        subspace_solve(b, diag, elen, control, par, apply)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use linus_scf::response::solver::reference::Problem;

    fn max_rel(a: &Mat<f64>, b: &Mat<f64>) -> f64 {
        let (mut d, mut m) = (0.0f64, 0.0f64);
        for j in 0..a.ncols() {
            for (x, y) in a.col_as_slice(j).iter().zip(b.col_as_slice(j)) {
                d = d.max((x - y).abs());
                m = m.max(x.abs());
            }
        }
        d / m.max(1e-300)
    }

    fn bits(a: &Mat<f64>) -> Vec<u64> {
        (0..a.ncols())
            .flat_map(|j| a.col_as_slice(j).iter().map(|x| x.to_bits()))
            .collect()
    }

    /// The device's algebra under the one driver takes the host algebra's iterations and
    /// reaches its solution and combined extras to round-off, bitwise the same run to run; a
    /// subspace beyond its first capacity grows (the second case's capacity of one iteration);
    /// near-dependent directions are dropped as on the host (the first case's duplicates). With
    /// a grant of the exact size ([`krylov::device_bytes`]), nothing is drawn beyond it.
    #[test]
    fn the_device_solver_follows_the_host_solver() {
        let Some(gpu) = crate::tests::harness::gpu_or_skip().map(Arc::new) else {
            return;
        };
        for (len, r, nr, dup, cap) in [(300, 9, 7, 2, 56), (4000, 40, 40, 0, 40)] {
            let p = Problem::new(len, r, nr, dup);
            let elen = len / 3;
            let par = Par::Seq;
            let (xh, fh, sh) = subspace_solve(
                &p.b,
                &p.d,
                elen,
                SolveControl::quiet(1e-10, 60),
                par,
                |zs: &[Vec<f64>]| Ok(p.apply(zs)),
            )
            .unwrap();
            let run = || {
                let need = krylov::device_bytes(len, nr, cap, elen, EXTRAS_BLOCK);
                let grant = gpu.grant(need, "the test's solver").unwrap();
                let (x, f, st) = {
                    let mut alg = DeviceAlgebra::new(&gpu, &p.b, &p.d, cap, Some(&grant)).unwrap();
                    let mut f = |zs: &[Vec<f64>]| Ok(p.apply(zs));
                    let ctl = SolveControl::quiet(1e-10, 60);
                    subspace_solve_with(&mut alg, nr, elen, ctl, par, &mut f).unwrap()
                };
                // the first case fits its grant; the second outgrows it (no longer drawn from)
                assert!(
                    grant.violations().is_empty(),
                    "len {len}: {:?}",
                    grant.violations()
                );
                (x, f, st)
            };
            let (xd, fd, sd) = run();
            assert_eq!(sd.iterations, sh.iterations, "iterations, len {len}");
            assert!(sd.max_residual <= 1e-10 && sh.max_residual <= 1e-10);
            let (ex, ef) = (max_rel(&xh, &xd), max_rel(&fh, &fd));
            assert!(
                ex < 1e-11 && ef < 1e-11,
                "len {len}: x {ex:.1e}, extras {ef:.1e}"
            );
            let (xd2, fd2, _) = run();
            assert_eq!(bits(&xd), bits(&xd2));
            assert_eq!(bits(&fd), bits(&fd2));
        }
    }

    /// A solve the device cannot hold (a grant-less space larger than the device) goes back to
    /// the host's solver from the start, recorded, with the host's bits.
    #[test]
    fn a_subspace_beyond_the_device_falls_back_to_the_host() {
        let Some(gpu) = crate::tests::harness::gpu_or_skip().map(Arc::new) else {
            return;
        };
        let p = Problem::new(300, 9, 7, 2);
        let records = ResponseRecords::default();
        records.start("hessian", Some("fp64"), "a test".into());
        // a capacity of 2^40 columns: no device holds V
        let mut dev = DeviceKrylov {
            gpu,
            grant: None,
            cap: 1 << 40,
            records: (records.clone(), "hessian"),
        };
        let mut f = |zs: &[Vec<f64>]| Ok(p.apply(zs));
        let ctl = SolveControl::quiet(1e-10, 60);
        let (xd, _, _) = dev
            .solve(&p.b, &p.d, 100, ctl.clone(), Par::Seq, &mut f)
            .unwrap();
        let (xh, _, _) =
            subspace_solve(&p.b, &p.d, 100, ctl, Par::Seq, |zs| Ok(p.apply(zs))).unwrap();
        assert_eq!(bits(&xd), bits(&xh));
        let solver = records.lock()[0].solver.clone().unwrap();
        assert_eq!(solver.place, "cpu");
        assert!(solver.reason.unwrap().contains("outgrew"));
    }
}
