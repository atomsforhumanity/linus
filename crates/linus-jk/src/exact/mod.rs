//! The integral-direct four-center J/K builder ([`FastJk`]): the SCF's [`JkBuilder`] over
//! linus-ints's [`JkPlan`] (the batched engine with the J/K digester; the scheme, the
//! screening and the determinism are described there). The builder adds the policy of an SCF:
//! a tighter threshold for the difference densities of incremental builds, and workspaces
//! kept between builds.

pub mod gradient;
pub mod hessian;
pub mod terms;

use faer::{Mat, Par};
use linus_base::linalg::{ao_view, ao_view_mut};
use linus_base::terms::{JkBuilder, JkOutput, JkRequest};
use linus_ints::eri4c::batch::BatchedPairs;
use linus_ints::operator::Operator;
use linus_ints::{
    AoMatrix, AoMatrixMut, BasisSet, JkArgs, JkOptions, JkOut, JkPlan, JkShape, JkStats,
    JkWorkspace, PairOptions, PairSet,
};

pub use linus_ints::eri4c::jk::PRIM_FACTOR;

/// Screening threshold of difference-density builds (incremental Fock builds) relative to the
/// full builds'. Each incremental build drops a different set of quartets, so at the full
/// threshold their errors add up over the builds since the last full one: caffeine/def2-SVP
/// RHF (thresh 1e-12) had incremental-minus-full energies of 0.4-2e-10 Eh at factor 1, 0.5-1e-11
/// at 0.1 and a constant 1.7e-11 (the full build's own screening error) at 0.01; u1an39 RHF
/// took 90 Fock builds at factor 1 (noise above e_tol 1e-10) and 19 at 0.1 (18 without
/// incremental builds), whose J/K time was 19% below the non-incremental SCF's.
pub const DELTA_THRESH_FACTOR: f64 = 0.1;

/// Workspaces kept between builds, one per request shape (an SCF alternates few).
const KEPT_WORKSPACES: usize = 4;

pub struct FastJk {
    plan: JkPlan,
    workspaces: Vec<JkWorkspace>,
    /// Screening threshold of difference-density builds (`compute_delta`) relative to the
    /// plan's.
    pub delta_thresh_factor: f64,
    /// Statistics of the last build.
    pub stats: JkStats,
}

impl FastJk {
    /// `thresh`: quartet screening threshold on Q_ab Q_cd max|D| (as PySCF's direct_scf_tol).
    pub fn new(basis: &BasisSet, thresh: f64) -> Self {
        Self::with_options(
            basis,
            JkOptions {
                thresh,
                ..Default::default()
            },
        )
    }

    /// Shell pairs that cannot reach thresh even with |D| = 100 are dropped; primitive pairs
    /// and primitive quartets are screened at `prim_factor * thresh`.
    pub fn with_prim_factor(basis: &BasisSet, thresh: f64, prim_factor: f64) -> Self {
        Self::with_options(
            basis,
            JkOptions {
                thresh,
                prim_factor,
                ..Default::default()
            },
        )
    }

    /// The builder of a plan with the options `opts`.
    pub fn with_options(basis: &BasisSet, opts: JkOptions) -> Self {
        let pairs = PairSet::build(basis, PairOptions::default());
        FastJk {
            plan: JkPlan::new(basis, &pairs, opts).expect("orbital pairs"),
            workspaces: Vec::new(),
            delta_thresh_factor: DELTA_THRESH_FACTOR,
            stats: JkStats::default(),
        }
    }

    /// K of the operator `op` (J is always of 1/r).
    pub fn with_k_operator(mut self, op: Operator) -> Self {
        self.plan.set_k_operator(op);
        self
    }

    /// The kernels of instruction set `isa` (tests), at most the CPU's best.
    #[cfg(test)]
    pub fn with_isa(mut self, isa: linus_num::simd::Isa) -> Self {
        self.plan = self.plan.with_isa(isa);
        self
    }

    /// The plan's batched layout (statistics and studies).
    pub fn pairs(&self) -> &BatchedPairs {
        self.plan.layout()
    }

    /// The screening threshold of full builds.
    pub fn thresh(&self) -> f64 {
        self.plan.options().thresh
    }

    /// J/K of `req` at the screening threshold `thresh`, with a workspace kept for the request's
    /// shape and the current pool.
    fn build(&mut self, req: &JkRequest, thresh: f64) -> JkOutput {
        let nbf = req.dens.first().map_or(0, |d| d.nrows());
        let shape = JkShape {
            ndens: req.dens.len(),
            j: req.want_j,
            k: req.want_k,
        };
        let nt = rayon::current_num_threads();
        let pos = self
            .workspaces
            .iter()
            .position(|w| w.shape() == shape && w.nthreads() == nt && self.plan.fits(w));
        let ws = match pos {
            Some(p) => &mut self.workspaces[p],
            None => {
                if self.workspaces.len() >= KEPT_WORKSPACES {
                    self.workspaces.remove(0);
                }
                self.workspaces.push(self.plan.workspace(shape, nt));
                self.workspaces.last_mut().unwrap()
            }
        };
        let zeros = |want: bool| -> Vec<Mat<f64>> {
            if want {
                (0..shape.ndens).map(|_| Mat::zeros(nbf, nbf)).collect()
            } else {
                Vec::new()
            }
        };
        let (mut j, mut k) = (zeros(req.want_j), zeros(req.want_k));
        let dens: Vec<AoMatrix> = req.dens.iter().map(|d| ao_view(d.as_ref())).collect();
        let mut jv: Vec<AoMatrixMut> = j.iter_mut().map(|m| ao_view_mut(m.as_mut())).collect();
        let mut kv: Vec<AoMatrixMut> = k.iter_mut().map(|m| ao_view_mut(m.as_mut())).collect();
        self.stats = self.plan.jk(
            &JkArgs {
                dens: &dens,
                thresh,
            },
            &mut JkOut {
                j: &mut jv,
                k: &mut kv,
            },
            ws,
        );
        JkOutput { j, k }
    }
}

impl JkBuilder for FastJk {
    fn name(&self) -> String {
        format!("fast(thresh={:e},isa={:?})", self.thresh(), self.plan.isa())
    }

    fn incremental_capable(&self) -> bool {
        true
    }

    /// One pass with J from 1/r and K from a/r + b erf(w r)/r: 11-14% faster than J + K and a
    /// second K_LR pass (caffeine wB97X core-guess density, 8 threads: def2-SVP 0.278 -> 0.239 s,
    /// def2-TZVP 5.22 -> 4.66 s).
    fn set_exchange_operator(&mut self, op: Operator) -> bool {
        self.plan.set_k_operator(op);
        true
    }

    fn compute_delta(
        &mut self,
        req: &JkRequest,
        _par: Par,
    ) -> Result<JkOutput, linus_base::error::LinusError> {
        let thresh = self.thresh() * self.delta_thresh_factor;
        Ok(self.build(req, thresh))
    }

    fn compute(
        &mut self,
        req: &JkRequest,
        _par: Par,
    ) -> Result<JkOutput, linus_base::error::LinusError> {
        let thresh = self.thresh();
        Ok(self.build(req, thresh))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::IncoreJk;
    use linus_core::structure::Structure;
    use linus_ints::basis::BasisDef;
    use linus_num::simd::Isa;

    fn mol(xyz: &str) -> Structure {
        Structure::from_xyz_str(xyz, 0, 1).unwrap()
    }

    const WATER2: &str = "6\n\nO 0.0 0.0 0.1173\nH 0.0 0.7572 -0.4692\nH 0.0 -0.7572 -0.4692\nO 2.9 0.1 0.2\nH 3.3 0.9 0.5\nH 3.4 -0.6 0.6\n";

    fn random_sym(n: usize, seed: u64) -> Mat<f64> {
        let mut s = seed;
        let mut r = || {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 11) as f64 / (1u64 << 53) as f64) - 0.5
        };
        let a = Mat::from_fn(n, n, |_, _| r());
        Mat::from_fn(n, n, |i, j| a[(i, j)] + a[(j, i)])
    }

    fn check(basis_name: &str, pure: bool, isa: Isa) {
        check_mol(WATER2, basis_name, pure, isa)
    }

    /// OH with def2-QZVP: every class pair up to (gg|gg) through the tiled kernels.
    const OH: &str = "2\n\nO 0.0 0.0 0.0\nH 0.3 -0.2 0.92\n";

    fn check_mol(xyz: &str, basis_name: &str, pure: bool, isa: Isa) {
        let m = mol(xyz);
        let def = BasisDef::load(basis_name, None).unwrap();
        let basis = BasisSet::new(&def, &m.z, &m.positions, pure).unwrap();
        let n = basis.nbf;
        let pairs = PairSet::build(&basis, PairOptions::default());
        let mut inc = IncoreJk::new(&basis, &pairs);
        let mut fast = FastJk::new(&basis, 1e-18).with_isa(isa);
        let dens = vec![random_sym(n, 7), random_sym(n, 11)];
        let req = JkRequest {
            dens: &dens,
            factors: None,
            want_j: true,
            want_k: true,
        };
        let a = inc.compute(&req, Par::Seq).unwrap();
        let b = fast.compute(&req, Par::Seq).unwrap();
        for di in 0..2 {
            for (x, y) in [(&a.j[di], &b.j[di]), (&a.k[di], &b.k[di])] {
                let mut worst: f64 = 0.0;
                let scale = (0..n)
                    .flat_map(|i| (0..n).map(move |j| (i, j)))
                    .map(|(i, j)| x[(i, j)].abs())
                    .fold(0.0, f64::max);
                for i in 0..n {
                    for j in 0..n {
                        worst = worst.max((x[(i, j)] - y[(i, j)]).abs());
                    }
                }
                assert!(
                    worst < 1e-12 * scale.max(1.0),
                    "{basis_name} {isa:?}: worst {worst:e} (scale {scale:e})"
                );
            }
        }
    }

    #[test]
    fn fast_jk_matches_incore() {
        for isa in [Isa::Generic, Isa::detect()] {
            check("def2-svp", true, isa);
            check("6-31g*", false, isa);
            check("def2-tzvp", true, isa);
        }
    }

    /// g shells (spherical, generated tiled pieces) and Cartesian f shells (run-time plans).
    #[test]
    fn fast_jk_matches_incore_high_l() {
        for isa in [Isa::Generic, Isa::detect()] {
            check_mol(OH, "def2-qzvp", true, isa);
            check_mol(OH, "cc-pvtz", false, isa);
        }
    }

    /// K for a range-separated operator against the contracted full tensor of that operator.
    #[test]
    fn range_separated_k() {
        use linus_ints::eri4c::eri_full;
        let m = mol(WATER2);
        for (bname, pure) in [("def2-svp", true), ("def2-tzvp", true), ("6-31g*", false)] {
            let def = BasisDef::load(bname, None).unwrap();
            let basis = BasisSet::new(&def, &m.z, &m.positions, pure).unwrap();
            let n = basis.nbf;
            let pairs = PairSet::build(&basis, PairOptions::default());
            let dens = vec![random_sym(n, 5)];
            let req = JkRequest {
                dens: &dens,
                factors: None,
                want_j: true,
                want_k: true,
            };
            let jref = IncoreJk::new(&basis, &pairs)
                .compute(&req, Par::Seq)
                .unwrap()
                .j
                .remove(0);
            for op in [
                Operator::LongRange { omega: 0.33 },
                Operator::Attenuated {
                    coulomb: 1.0,
                    long_range: -1.0,
                    omega: 0.4,
                },
                Operator::from_libxc(0.19, 0.46, 0.33),
            ] {
                let g = eri_full(&basis, &pairs, &op);
                let d = &dens[0];
                let kref = Mat::from_fn(n, n, |i, j| {
                    let mut s = 0.0;
                    for l in 0..n {
                        for t in 0..n {
                            s += g[((i * n + l) * n + j) * n + t] * d[(l, t)];
                        }
                    }
                    s
                });
                for isa in [Isa::Generic, Isa::detect()] {
                    let mut fast = FastJk::new(&basis, 1e-18).with_isa(isa).with_k_operator(op);
                    let out = fast.compute(&req, Par::Seq).unwrap();
                    for (x, y, what) in [(&jref, &out.j[0], "J"), (&kref, &out.k[0], "K")] {
                        let scale = (0..n * n)
                            .map(|e| x[(e / n, e % n)].abs())
                            .fold(0.0, f64::max);
                        let worst = (0..n * n)
                            .map(|e| (x[(e / n, e % n)] - y[(e / n, e % n)]).abs())
                            .fold(0.0, f64::max);
                        assert!(
                            worst < 1e-12 * scale.max(1.0),
                            "{bname} {op:?} {isa:?} {what}: {worst:e}"
                        );
                    }
                }
            }
        }
    }

    /// J-only requests go through the Hermite J-engine for every block.
    #[test]
    fn j_only_jengine_matches_incore() {
        let m = mol(WATER2);
        for (bname, pure) in [("def2-svp", true), ("def2-tzvp", true), ("6-31g*", false)] {
            let def = BasisDef::load(bname, None).unwrap();
            let basis = BasisSet::new(&def, &m.z, &m.positions, pure).unwrap();
            let n = basis.nbf;
            let pairs = PairSet::build(&basis, PairOptions::default());
            let dens = vec![random_sym(n, 23), random_sym(n, 29)];
            let req = JkRequest {
                dens: &dens,
                factors: None,
                want_j: true,
                want_k: false,
            };
            let a = IncoreJk::new(&basis, &pairs)
                .compute(&req, Par::Seq)
                .unwrap();
            for isa in [Isa::Generic, Isa::detect()] {
                let mut fast = FastJk::with_options(
                    &basis,
                    JkOptions {
                        thresh: 1e-18,
                        jengine: true,
                        ..Default::default()
                    },
                )
                .with_isa(isa);
                let b = fast.compute(&req, Par::Seq).unwrap();
                assert!(b.k.is_empty());
                for di in 0..2 {
                    let (x, y) = (&a.j[di], &b.j[di]);
                    let scale = (0..n * n)
                        .map(|e| x[(e / n, e % n)].abs())
                        .fold(0.0, f64::max);
                    let worst = (0..n * n)
                        .map(|e| (x[(e / n, e % n)] - y[(e / n, e % n)]).abs())
                        .fold(0.0, f64::max);
                    assert!(worst < 1e-12 * scale.max(1.0), "{bname} {isa:?}: {worst:e}");
                }
            }
        }
    }

    /// A builder reused for a second density (pooled accumulators) gives exactly what a fresh
    /// builder gives.
    #[test]
    fn reuse_matches_fresh() {
        let m = mol(WATER2);
        let def = BasisDef::load("def2-svp", None).unwrap();
        let basis = BasisSet::new(&def, &m.z, &m.positions, true).unwrap();
        let (d1, d2) = (
            vec![random_sym(basis.nbf, 13)],
            vec![random_sym(basis.nbf, 17)],
        );
        let r1 = JkRequest {
            dens: &d1,
            factors: None,
            want_j: true,
            want_k: true,
        };
        let r2 = JkRequest {
            dens: &d2,
            factors: None,
            want_j: true,
            want_k: true,
        };
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap();
        pool.install(|| {
            let mut reused = FastJk::new(&basis, 1e-12);
            reused.compute(&r1, Par::Seq).unwrap();
            let a = reused.compute(&r2, Par::Seq).unwrap();
            let b = FastJk::new(&basis, 1e-12).compute(&r2, Par::Seq).unwrap();
            assert!(a.j[0] == b.j[0] && a.k[0] == b.k[0]);
        });
    }

    #[test]
    fn deterministic_across_threads() {
        let m = mol(WATER2);
        let def = BasisDef::load("def2-svp", None).unwrap();
        let basis = BasisSet::new(&def, &m.z, &m.positions, true).unwrap();
        let dens = vec![random_sym(basis.nbf, 3)];
        let req = JkRequest {
            dens: &dens,
            factors: None,
            want_j: true,
            want_k: true,
        };
        let mut res = Vec::new();
        for nt in [1, 3, 8] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(nt)
                .build()
                .unwrap();
            let mut fast = FastJk::new(&basis, 1e-12);
            res.push(pool.install(|| fast.compute(&req, Par::Seq).unwrap()));
        }
        for r in &res[1..] {
            assert!(r.j[0] == res[0].j[0] && r.k[0] == res[0].k[0]);
        }
    }
}
