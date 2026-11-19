//! The PCM surface terms with the auxiliary basis of density-fitted J: the `predicted` SCF mode
//! (exact integrals, one pass per iteration) and the density-fitted PCM (`df`, an approximation).
//!
//! **Predicted** ([`super::PcmScfMode::Predicted`], the default with density-fitted J): every
//! SCF iteration needs the exact potential of the density D_n at the points (for the exact
//! energy) and the Fock term -sum_g q_n,g (ab|g) of its charges, two passes over all shell
//! pairs and points, the second depending on the first. Here one fused pass
//! ([`linus_ints::SurfacePlan::potentials_and_matrices`]) evaluates the exact
//! potential v_e(D_n) and the Fock term of predicted charges q~_n = Q phi~_n, with the potential
//! predicted from the previous iteration's exact one and the change of the RI-J fitted density:
//! phi~_n = phi_{n-1} - M (c_n - c_{n-1}) (the J build's fit coefficients c, M below; the first
//! iteration uses phi~_1 = v_n - M c_1). The energy of every iteration is exact; the Fock term
//! differs from the exact one by Q times the RI error of the potential change only, so the
//! fixed point (c_n = c_{n-1}) is the exact PCM solution and the convergence is that of the
//! exact two-pass SCF (the lag without the RI correction, `lagged_fock`, costs ~75% more
//! iterations). Gradients and Hessians are those of the exact PCM.
//!
//! The predicted charges (and the incremental passes) make every such build incremental
//! ([`linus_base::terms::EnvironmentTerm::built_incrementally`]): its Fock term is not of the
//! density alone. When the SCF asks for a full build (its confirmation of a converged iterate, a
//! restart: `reset_incremental`), the next step is the exact mode's: the potential of the density
//! in a full pass, its charges, then their Fock term in a second pass. So the SCF ends on the
//! exact PCM operator of its density, as the `exact` mode's SCF does, at the cost of one more
//! build (decision 25's one-build rule).
//!
//! **The density-fitted PCM** ([`super::PcmScfMode::Df`]): with the Coulomb-metric fit (ab|g) ~
//! sum_PQ (ab|P) [V^-1]_PQ (Q|g) (the RI-J approximation
//! applied to the surface charges' Coulomb integrals), the solute's electronic potential at
//! the points is v_e = M c with the fit coefficients c = V^-1 gamma, gamma_P = sum_ab (P|ab)
//! D_ab, and M_gP = (P|g) ([`linus_ints::AuxSurfacePlan::matrix`], closed form); the
//! Fock contribution -sum_g q_g (ab|g) becomes -sum_P (ab|P) d_P with d = V^-1 M^T q. Both
//! ride on the RI-J build of the SCF ([`linus_base::terms::CoulombHook`]): the J build computes c
//! anyway, the hook computes v_e, solves for the charges and returns -d/2 per density to be
//! added to the coefficients before the expansion, so the reaction field costs two
//! (npts x naux) products and the dense solve per iteration instead of two passes over all
//! shell pairs and points.
//!
//! Energy: E = 1/2 q_sym . phi, phi = v_n - M c, with q_sym = Q phi (Q the model's symmetrized
//! response). The SCF's Coulomb energy 1/2 Tr(D J) then holds -1/2 q_sym . v_e, and the XC
//! hook (`super::SolventXc`) adds 1/2 q_sym . v_n; the two parts are moved to the solvent
//! component after the SCF. The energy is variational in D (its Fock matrix is the exact
//! derivative of this RI energy), so gradients follow from the fixed-density derivative:
//!
//!   dE/dX = q_sym . dv_n/dX + 1/2 phi^T dQ/dX phi
//!           - sum_gP q_g c_P d(P|g)/dX - sum_P d_P sum_ab d(P|ab)/dX D_ab + d^T dV/dX c,
//!
//! (the first two as in the exact PCM; the last two are RI-J-like three- and two-centre
//! derivative terms with the coefficient pairs (d, D) and (d, c)).
//!
//! The RI fit is made for the Coulomb energy, not for potentials at points in the density's
//! tail: with def2-universal-jfit the potential at the surface points is off by ~1e-3 au (rms)
//! and the electrostatic solvation energy by up to 4% (PySCF's own fit gives the same), well above
//! the RI-J error of the total energy; hence `predicted` is the default and `ri` an option.

use super::PcmScf;
use faer::{Mat, MatMut, Par};
use linus_core::gradient::Gradient;
use linus_core::timing::Timers;
use linus_ints::eri3c;
use linus_ints::operator::Operator;
use linus_ints::points::pcm_ints::{AuxSurfacePlan, GaussPoint, SurfaceShape};
use linus_ints::{
    AoMatrix, BasisSet, DerivOrder, Eri3DerivOptions, Eri3DerivPlan, PairOptions, PairSet,
};
use linus_num::det::{add_into, nchunk_for, par_fold_det, DEFAULT_BUDGET};
use rayon::prelude::*;
use std::sync::{Arc, Mutex};

/// RI data of the PCM term of one SCF (`predicted` and `ri` modes).
pub struct RiPcm {
    pub mode: super::PcmScfMode,
    /// The auxiliary basis of the SCF's density-fitted J.
    pub aux: BasisSet,
    /// (P|g): npts x naux, row-major.
    m: Vec<f64>,
    /// Energy parts of the last Coulomb build: (1/2 q_sym . v_n, -1/2 q_sym . v_e); their sum
    /// is the solvent energy, the second part sits in the SCF's Coulomb energy.
    pub parts: (f64, f64),
    /// Coulomb builds seen and XC evaluations that consumed them.
    pub builds: u64,
    pub consumed: u64,
    /// Predicted mode: total fit coefficients of the last Coulomb build, and (exact phi, fit
    /// coefficients) of the previous iteration.
    c_cur: Vec<f64>,
    prev: Option<(Vec<f64>, Vec<f64>)>,
    /// Incremental passes: (density, predicted charges, potential, Fock integrals sum_g q~ (ab|g))
    /// of the last pass and the incremental passes since the last full one.
    #[allow(clippy::type_complexity, reason = "a one-off tuple of buffers")]
    inc: Option<(Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, usize)>,
    /// Predicted mode: the next step is exact (the SCF asked for a full build).
    pub(crate) full_next: bool,
    /// Predicted mode: the last step's Fock term was of predicted charges.
    pub(crate) last_incremental: bool,
}

/// Incremental passes between full ones in the predicted mode.
pub const INCREMENTAL_RESET: usize = 8;

impl RiPcm {
    pub fn new(aux: BasisSet, pts: &[GaussPoint], mode: super::PcmScfMode) -> Self {
        let mut m = vec![0.0; pts.len() * aux.nbf];
        AuxSurfacePlan::new(&aux, pts).matrix(&mut m);
        RiPcm {
            mode,
            aux,
            m,
            parts: (0.0, 0.0),
            builds: 0,
            consumed: 0,
            c_cur: Vec::new(),
            prev: None,
            inc: None,
            full_next: false,
            last_incremental: false,
        }
    }

    pub fn naux(&self) -> usize {
        self.aux.nbf
    }

    /// v = M c (rows summed in order: deterministic).
    pub fn apply_m(&self, c: &[f64]) -> Vec<f64> {
        let naux = self.naux();
        self.m
            .par_chunks(naux)
            .map(|row| row.iter().zip(c).map(|(a, b)| a * b).sum())
            .collect()
    }

    /// b = M^T q (fixed chunks of points, ordered reduction: deterministic).
    pub fn apply_mt(&self, q: &[f64]) -> Vec<f64> {
        let naux = self.naux();
        let np = q.len();
        par_fold_det(
            np,
            nchunk_for(np, naux, DEFAULT_BUDGET),
            || vec![0.0; naux],
            |acc, g| {
                let qg = q[g];
                if qg != 0.0 {
                    acc.iter_mut()
                        .zip(&self.m[g * naux..(g + 1) * naux])
                        .for_each(|(a, r)| *a += qg * r);
                }
            },
            |a, b| add_into(a, &b),
        )
    }

    /// Bytes of the stored (P|g).
    pub fn bytes(&self) -> usize {
        self.m.len() * 8
    }
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

impl PcmScf {
    /// One Coulomb build of the SCF ([`linus_base::terms::CoulombHook`]): fit coefficients `c` of
    /// the densities of the request (one: alpha of a closed shell, total = 2 c; two: alpha and
    /// beta). Solves for the charges of the fitted density, records the state and the energy
    /// parts, and returns the extra coefficients -d/2 per density (the J matrices then sum to
    /// J[D] - sum_P (ab|P) d_P, the reaction field's Fock term).
    pub(crate) fn ri_coulomb_step(
        &mut self,
        c: &[Vec<f64>],
        solve: &dyn Fn(&mut [f64]),
    ) -> Vec<Vec<f64>> {
        let t0 = std::time::Instant::now();
        let ri = self.ri.as_mut().expect("density-fitted PCM data");
        let naux = ri.naux();
        assert!(
            c.iter().all(|x| x.len() == naux),
            "PCM: the J builder's auxiliary basis differs from the solvent's"
        );
        let ct: Vec<f64> = match c.len() {
            1 => c[0].iter().map(|x| 2.0 * x).collect(),
            2 => c[0].iter().zip(&c[1]).map(|(a, b)| a + b).collect(),
            n => panic!("PCM: a Coulomb build of {n} densities is not an SCF step"),
        };
        if ri.mode == super::PcmScfMode::Predicted {
            // record the fit of this density for the XC side; J unchanged
            ri.c_cur = ct;
            ri.builds += 1;
            return Vec::new();
        }
        let ri = self.ri.as_ref().unwrap();
        let ve = ri.apply_m(&ct);
        let phi: Vec<f64> = self.op.v_nuc.iter().zip(&ve).map(|(a, b)| a - b).collect();
        let t1 = std::time::Instant::now();
        let ch = self.op.solve(&phi);
        let e = super::pcm::PcmOperator::energy(&phi, &ch);
        let t2 = std::time::Instant::now();
        let mut d = ri.apply_mt(&ch.q_sym);
        solve(&mut d);
        let parts = (
            0.5 * dot(&ch.q_sym, &self.op.v_nuc),
            -0.5 * dot(&ch.q_sym, &ve),
        );
        let ri = self.ri.as_mut().unwrap();
        ri.parts = parts;
        ri.builds += 1;
        self.last = Some((phi, ch, e));
        let t3 = std::time::Instant::now();
        self.timers.add("pcm_potential", (t1 - t0).as_secs_f64());
        self.timers.add("pcm_solve", (t2 - t1).as_secs_f64());
        self.timers.add("pcm_fock", (t3 - t2).as_secs_f64());
        let ek: Vec<f64> = d.iter().map(|x| -0.5 * x).collect();
        vec![ek; c.len()]
    }

    /// Predicted mode (module docs): exact energy and the Fock term of the predicted charges
    /// for total density `dt` (the density of the Coulomb build just done), one fused pass; after
    /// `full_next`, the exact mode's two passes (the exact charges' Fock term).
    pub(crate) fn predicted_energy_fock(&mut self, dt: &Mat<f64>) -> (f64, Mat<f64>) {
        let t0 = std::time::Instant::now();
        let n = dt.nrows();
        let ri = self.ri.as_mut().expect("RI data");
        assert_eq!(
            ri.builds,
            ri.consumed + 1,
            "PCM: the SCF evaluated the XC term without a Coulomb build before it"
        );
        ri.consumed = ri.builds;
        let c = std::mem::take(&mut ri.c_cur);
        if std::mem::take(&mut ri.full_next) {
            // of the density alone: the exact potential, its charges and their Fock term; the
            // next predicted step starts from them (and from a full pass)
            ri.inc = None;
            ri.last_incremental = false;
            let (e, v) = self.energy_fock(dt);
            let phi = self.last.as_ref().expect("the step's state").0.clone();
            self.ri.as_mut().expect("RI data").prev = Some((phi, c));
            return (e, v);
        }
        let d = linus_base::linalg::mat_to_rowmajor(dt.as_ref());
        let ri = self.ri.as_ref().unwrap();
        // phi~ = phi_prev - M (c - c_prev), or v_n - M c
        let phi_p: Vec<f64> = match &ri.prev {
            Some((phi0, c0)) => {
                let dc: Vec<f64> = c.iter().zip(c0).map(|(a, b)| a - b).collect();
                let dv = ri.apply_m(&dc);
                phi0.iter().zip(&dv).map(|(a, b)| a - b).collect()
            }
            None => {
                let ve = ri.apply_m(&c);
                self.op.v_nuc.iter().zip(&ve).map(|(a, b)| a - b).collect()
            }
        };
        let qp = self.op.solve(&phi_p).q_sym;
        let t1 = std::time::Instant::now();
        // incremental (the terms are linear in D and q~): the pass on the changes D_n - D_{n-1},
        // q~_n - q~_{n-1}, whose screening drops more as the SCF converges; a full pass every
        // INCREMENTAL_RESET passes
        let inc = if self.resolved.incremental {
            self.ri
                .as_mut()
                .unwrap()
                .inc
                .take()
                .filter(|x| x.4 < INCREMENTAL_RESET)
        } else {
            None
        };
        let x = self.exact();
        let mut ws = x.workspace(SurfaceShape {
            densities: 1,
            charge_sets: 1,
            gradient: false,
        });
        // the fused pass: the potential of a density and the potential matrix (row-major) of
        // charges
        let pass = |d: &[f64], q: &[f64], ws: &mut linus_ints::SurfaceWorkspace| {
            let n = self.basis.nbf;
            let (mut v, mut m) = (vec![0.0; x.points().len()], vec![0.0; n * n]);
            let mut mv = [linus_ints::AoMatrixMut::row_major(&mut m, n)];
            x.potentials_and_matrices((&[d], &[q]), &mut v, &mut mv, ws);
            (v, m)
        };
        let (ve, vm, since) = match inc {
            Some((d0, q0, v0, m0, k)) => {
                let dd: Vec<f64> = d.iter().zip(&d0).map(|(a, b)| a - b).collect();
                let dq: Vec<f64> = qp.iter().zip(&q0).map(|(a, b)| a - b).collect();
                let (v, m) = pass(&dd, &dq, &mut ws);
                let ve: Vec<f64> = v0.iter().zip(&v).map(|(a, b)| a + b).collect();
                let vm: Vec<f64> = m0.iter().zip(&m).map(|(a, b)| a + b).collect();
                (ve, vm, k + 1)
            }
            None => {
                let (v, m) = pass(&d, &qp, &mut ws);
                (v, m, 0)
            }
        };
        let t2 = std::time::Instant::now();
        if self.resolved.incremental {
            self.ri.as_mut().unwrap().inc =
                Some((d.clone(), qp.clone(), ve.clone(), vm.clone(), since));
        }
        let phi: Vec<f64> = self.op.v_nuc.iter().zip(&ve).map(|(a, b)| a - b).collect();
        let ch = self.op.solve(&phi);
        let e = super::pcm::PcmOperator::energy(&phi, &ch);
        let t3 = std::time::Instant::now();
        self.timers.add("pcm_predict", (t1 - t0).as_secs_f64());
        self.timers
            .add("pcm_potential_fock", (t2 - t1).as_secs_f64());
        self.timers.add("pcm_solve", (t3 - t2).as_secs_f64());
        let ri = self.ri.as_mut().unwrap();
        ri.prev = Some((phi.clone(), c));
        ri.last_incremental = true;
        self.last = Some((phi, ch, e));
        (e, Mat::from_fn(n, n, |i, j| -vm[i * n + j]))
    }

    /// The XC side of a density-fitted PCM SCF step: the energy part 1/2 q_sym . v_n of the Coulomb
    /// build just done (the Fock term is in J).
    pub(crate) fn ri_xc_energy(&mut self) -> f64 {
        let ri = self.ri.as_mut().expect("density-fitted PCM data");
        assert_eq!(
            ri.builds,
            ri.consumed + 1,
            "the density-fitted PCM: the SCF evaluated the XC term without a Coulomb \
                 build before it"
        );
        ri.consumed = ri.builds;
        ri.parts.0
    }

    /// The density-fitted PCM energy and fixed-density nuclear gradient (natom x 3) for the total
    /// density `d` (row-major nbf^2); see the module docs.
    pub(crate) fn ri_fixed_density_gradient(
        &self,
        z: &[u32],
        positions: &[[f64; 3]],
        d: &[f64],
        timers: &mut Timers,
    ) -> (f64, Vec<[f64; 3]>) {
        let ri = self.ri.as_ref().expect("density-fitted PCM data");
        let basis = &self.basis;
        let naux = ri.aux.nbf;
        // the RI-J integrals of the fit (linus-ints' Eri3DerivPlan: the orbital pairs with E^A
        // for the three-center term) and the metric's Cholesky factor
        let t0 = std::time::Instant::now();
        let pairs = PairSet::build(basis, PairOptions::with_order(DerivOrder::One));
        let ap = PairSet::aux(&ri.aux, PairOptions::default());
        let opts = Eri3DerivOptions::default();
        let plan = Eri3DerivPlan::new(basis, &pairs, &ri.aux, &ap, DerivOrder::One, opts)
            .expect("orbital pairs of order 1 in shell-index orientation");
        let v = eri3c::metric(plan.aux3c(), 0.0, &Operator::Coulomb);
        let vm = Mat::from_fn(naux, naux, |i, j| v[i * naux + j]);
        // sequential: the factor must not depend on the thread count
        let chol = linus_base::linalg::cholesky_in_place(vm, Par::Seq)
            .expect("auxiliary metric not positive definite");
        timers.add("grad_solvent_ri_setup", t0.elapsed().as_secs_f64());
        let mut ws = plan.workspace(rayon::current_num_threads());
        let dv = AoMatrix::row_major(d, basis.nbf);
        // x <- V^-1 x
        let solve = |x: &mut [f64]| {
            let mut m = MatMut::from_column_major_slice_mut(x, naux, 1);
            linus_base::linalg::solve_lower_in_place(chol.as_ref(), m.as_mut(), Par::Seq);
            linus_base::linalg::solve_upper_in_place(chol.transpose(), m, Par::Seq);
        };
        let mut c = vec![0.0; naux];
        timers.time("grad_solvent_ri_gamma", || plan.gamma(dv, &mut c, &mut ws));
        solve(&mut c);
        let ve = ri.apply_m(&c);
        let phi: Vec<f64> = self.op.v_nuc.iter().zip(&ve).map(|(a, b)| a - b).collect();
        let ch = self.op.solve(&phi);
        let mut dd = ri.apply_mt(&ch.q_sym);
        solve(&mut dd);
        let zf: Vec<f64> = z.iter().map(|&x| x as f64).collect();
        let mut g = timers.time("grad_solvent_surface", || {
            let mut g = Gradient::from(self.op.nuclear_gradient(&zf, positions, &ch.q_sym));
            g += self.op.solver_gradient(positions, &phi, &ch);
            g
        });
        // - sum_gP q_g c_P d(P|g)/dX (auxiliary centres and points)
        let (ga, gp) = timers.time("grad_solvent_ri_2c_points", || {
            let plan = AuxSurfacePlan::new(&ri.aux, self.points());
            let mut ga = vec![[0.0; 3]; ri.aux.natom()];
            let mut gp = vec![[0.0; 3]; self.points().len()];
            plan.gradient(&ch.q_sym, &c, &mut ga, &mut gp, &mut plan.workspace());
            (ga, gp)
        });
        g -= &ga;
        for (i, v) in gp.iter().enumerate() {
            let o = self.op.surf.owner[i];
            for k in 0..3 {
                g[o][k] -= v[k];
            }
        }
        // - sum_P d_P sum_ab d(P|ab)/dX D_ab
        let mut g3 = vec![[0.0; 3]; plan.natom()];
        timers.time("grad_solvent_ri_3c", || {
            plan.coulomb_gradient(dv, &dd, &mut g3, &mut ws)
        });
        g -= &g3;
        // + sum_PQ d_P c_Q d(P|Q)/dX
        let mut g2 = vec![[0.0; 3]; plan.natom()];
        timers.time("grad_solvent_ri_metric", || {
            plan.metric_gradient(&dd, &c, &mut g2, &mut ws)
        });
        g += &g2;
        (super::pcm::PcmOperator::energy(&phi, &ch), g.into_inner())
    }
}

/// The DfJk hook of a density-fitted PCM SCF: the PCM term shared with the SCF's XC hook.
pub struct PcmCoulombHook(pub Arc<Mutex<PcmScf>>);

impl linus_base::terms::CoulombHook for PcmCoulombHook {
    fn extra_coefficients(&mut self, c: &[Vec<f64>], solve: &dyn Fn(&mut [f64])) -> Vec<Vec<f64>> {
        self.0.lock().expect("PCM state").ri_coulomb_step(c, solve)
    }
}
