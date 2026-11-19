//! The static dipole polarizability, closed and open shells (HF and KS; exact four-center
//! integrals or density fitting), assembled from the electric field as a perturbation of the
//! reference (`crate::properties::dipole::ElectricField`), its response problem and the
//! stationary form. Unlike the dipole derivatives it needs the response to the field: a
//! three-right-hand-side electric-field CPHF/CPKS, (A + B) U_s^c = -C_vs^T r_c C_os, solved in a
//! shared subspace, and
//! α_cd = -sum_s Tr(dD_s^c r_d) = 2 sum_s U_s^c . (C_vs^T r_d C_os) (both spins of a closed
//! shell: 4 U^c . (C_v^T r_d C_o)). The field does not move the grid, so the XC kernel is the
//! only XC term.
//!
//! α is assembled in the stationary form of every second-order property
//! ([`linus_scf::response::stationary`]: the argument and the measurements),
//! α_cd = f sum_s [U_s^c . b_s^d + U_s^c . r_s^d] (f = 4 closed shell, 2 per spin;
//! b_s^c = -C_vs^T r_c C_os, r the solve's final residuals), with the second-order assembly's
//! one reduction (GEMMs of the stacked matrices). The three components are solved in one shared
//! subspace, a Galerkin method, so the correction U^c . r^d vanishes to round-off (about 1e-15
//! relative) and α is the Ritz value; the correction keeps it stationary with any solver. The
//! electric-field CPHF also serves the dipole derivatives of IR intensities
//! ([`crate::derivatives::hessian::HessianOptions::dipole_derivatives`]), whose stationary form
//! needs its U^c.

use crate::driver::ground::GroundState;
use crate::driver::{CalcContext, Purpose, Session, UnconvergedResponse};
use crate::properties::dipole::ElectricField;
use crate::two_electron::{ResponseRequest, ResponseUse};
use faer::{Mat, Par};
use linus_base::error::{LinusError, ResponseFailure};
use linus_base::terms::ResponseJk;
use linus_core::timing::Timers;
use linus_ints::DerivOrder;
use linus_scf::reference::Reference;
use linus_scf::response::operator::ResponseOperator;
use linus_scf::response::perturbation::Perturbation;
use linus_scf::response::problem::ResponseProblem;
use linus_scf::response::second_order::{refs, stacked_gram};
use linus_scf::response::solver::ResponseStats;
use linus_scf::response::stationary::Correction;

/// Static polarizability and its CPHF statistics.
#[derive(Clone, Debug)]
pub struct Polarizability {
    /// α (a.u.), symmetric.
    pub alpha: [[f64; 3]; 3],
    pub cphf_iterations: usize,
    /// Max-abs residual of the last CPHF/CPKS iteration (at most `response.field_tol`, or
    /// the polarizability is an error).
    pub max_residual: f64,
    pub timers: Timers,
}

/// The electric-field CPHF/CPKS of a polarizability: α, the solve's statistics, and its
/// solutions U_s^c (`[s][c]`: channel, field component).
pub(crate) struct FieldResponse {
    pub alpha: [[f64; 3]; 3],
    pub stats: ResponseStats,
    pub u: Vec<Vec<Mat<f64>>>,
}

impl FieldResponse {
    /// The polarizability of this response, with the timers of its setup and solve.
    pub fn polarizability(&self, timers: Timers) -> Polarizability {
        Polarizability {
            alpha: self.alpha,
            cphf_iterations: self.stats.iterations,
            max_residual: self.stats.max_residual,
            timers,
        }
    }
}

impl CalcContext {
    /// Electric-field CPHF/CPKS (three right-hand sides) and α_cd = 2 sum_s U_s^c . rhs_s^d plus
    /// the stationary correction (rhs = -C_v^T r C_o; closed shell: both spins; the module docs), on
    /// the XC kernel of `xc` (the SCF grid's). `jk`: a response J/K of these occupied orbitals
    /// that already serves these products ([`ResponseJk::serve`]: an IR run's Hessian's, kept
    /// alive past the Hessian with its stored exchange operator); None: one of its own, double
    /// precision. Timed as `pol_cphf` (its setup `pol_cphf_setup`), its products' J/K and XC
    /// kernel as `pol_cphf_jk` and `pol_cphf_xc` (with the builder's and a device kernel's
    /// phases as `pol_resp_*` and `pol_cphf_xc_dev_*`).
    #[allow(clippy::needless_range_loop, reason = "indexes several arrays in step")]
    pub(crate) fn field_response(
        &self,
        gs: &GroundState,
        reference: &Reference,
        xc: Option<&linus_dft::XcIntegrator>,
        jk: Option<Box<dyn ResponseJk>>,
        par: Par,
        timers: &mut Timers,
    ) -> Result<FieldResponse, LinusError> {
        let basis = &self.basis;
        // the method's exact exchange (range-separated hybrids: a_x = 1 and the builders'
        // exchange operator a/r + b erf(w r)/r)
        let xop = self.method.exchange;
        let a_x = linus_base::terms::exchange_parts(xop).0;
        let tol = self.plan.response().field_tol();
        // the field as a perturbation of the reference (crate::properties::dipole): its
        // right-hand sides b_s^c = -C_vs^T r_c C_os
        let field = ElectricField::new(reference, basis, par);
        // the geometry's pairs (crate::context)
        let pairs = self.integrals.pairs(basis, DerivOrder::Zero);
        // the reference's spin channels (crate::reference): the alpha orbitals of a closed shell
        // (occupation 2), or both spins
        let focc = reference.occupation();
        let co = reference.c_occ();
        let rhs = field.rhs_vo(None, par);
        // the Hessian's kind of response J/K (double precision, one level), or the one given
        let (jk, kernel_use) = match jk {
            // a Hessian's response J/K serves these products in double precision
            Some(jk) => (
                jk,
                crate::two_electron::KernelUse {
                    purpose: ResponseUse::Polarizability,
                    single: false,
                },
            ),
            None => {
                let req = ResponseRequest {
                    ground: gs,
                    occ: &co,
                    xop,
                    purpose: ResponseUse::Polarizability,
                };
                let s = self.response_jk(&req, false, &pairs, par, timers)?;
                (s.jk, s.kernel)
            }
        };
        // (a device semilocal kernel for these products: larger blocks and its ground state
        // kept across the calls, crate::backend::gpu::xc::response_kernel; VV10's on top)
        let vv10 = self.vv10_kernel(xc, &co, "pol", timers)?;
        let kernel = self.response_kernel(xc, vv10, &co, kernel_use, timers)?;
        // the response operator (crate::response::operator): the J/K, the CPKS kernel and the
        // solvent's reaction field (crate::solvent)
        let operator = ResponseOperator::new(&co, jk, a_x)
            .with_kernel(kernel.as_ref().map(|k| k.kernel()))
            .with_environment(self.solvent_response(gs)?);
        let mut problem = ResponseProblem::new(reference, operator);
        problem.report_to(self.progress.clone(), "polarizability");
        // the solve keeps dF[U] C_o (three products' worth) for the residuals of the stationary
        // correction
        let sol = timers.time("pol_cphf", || problem.solve(&rhs, tol, 100, true, par))?;
        let (u, st) = (sol.u, sol.stats);
        let fu = sol.fu.expect("the solve keeps its products");
        let resid = problem.residuals(&rhs, &u, &fu, par);
        drop(fu);
        // the products' phases: J/K and XC kernel (cphf_jk, cphf_xc, ...), the builder's own
        // (resp_*) and a device kernel's (xc_dev_*), under pol_
        let (jk, phases) = problem.into_parts();
        let builder = jk.timers();
        drop(jk);
        let device = kernel.as_ref().and_then(|k| k.device_timers());
        for (prefix, t) in [
            ("pol_", Some(phases)),
            ("pol_", builder),
            ("pol_cphf_", device),
        ] {
            for (k, v, _) in t.iter().flat_map(|t| &t.entries) {
                timers.add(&format!("{prefix}{k}"), *v);
            }
            for (k, v) in t.iter().flat_map(|t| &t.resources) {
                timers.resource(&format!("{prefix}{k}"), *v);
            }
        }
        if let Some(k) = &kernel {
            k.record_vv10("pol_cphf", timers);
        }
        // f [U^c . b^d + U^c . r^d] per channel (crate::response::stationary), the second-order
        // assembly's GEMMs of the stacked matrices; symmetric at convergence
        let f = 2.0 * focc;
        let correction = Correction {
            u: &u,
            residuals: &resid,
        };
        let mut m = Mat::<f64>::zeros(3, 3);
        for s in 0..u.len() {
            stacked_gram(&mut m, f, &refs(&u[s]), &refs(&rhs[s]), par);
            correction.add_gram(&mut m, s, f, par);
        }
        let alpha: [[f64; 3]; 3] =
            std::array::from_fn(|c| std::array::from_fn(|d| 0.5 * (m[(c, d)] + m[(d, c)])));
        Ok(FieldResponse {
            alpha,
            stats: st,
            u,
        })
    }
}

impl Session {
    /// Static dipole polarizability at the current geometry, closed or open shell, at an SCF
    /// converged as for Hessians: the last converged SCF when it belongs to this geometry and was
    /// converged for a response, else one of its own (so the result does not depend on which
    /// SCF ran before).
    pub fn polarizability(&mut self) -> Result<Polarizability, LinusError> {
        // what the method cannot do fails here, before any SCF
        self.cx
            .check_request(crate::derivatives::capabilities::Request::Polarizability)?;
        let positions = &self.cx.mol.positions;
        let gs = match self
            .ground
            .as_ref()
            .filter(|g| g.serves_response_at(positions))
        {
            Some(g) => g.clone(),
            None => self.scf_stage(Purpose::Hessian)?,
        };
        let mut timers = Timers::default();
        let par = self.par();
        let pool = &self.cx.pool;
        let xc = pool.install(|| self.cx.xc_on(gs.grid(), &mut timers))?;
        let reference = gs.reference(pool, par);
        let fr = pool.install(|| {
            self.cx
                .field_response(&gs, reference, xc.as_deref(), None, par, &mut timers)
        })?;
        self.accept_polarizability(fr.polarizability(timers))
    }

    /// `p` if its CPHF/CPKS reached `response.field_tol`; else an unconverged response is
    /// an error that keeps its result ([`Session::unconverged_response`]).
    pub(crate) fn accept_polarizability(
        &mut self,
        p: Polarizability,
    ) -> Result<Polarizability, LinusError> {
        let tol = self.cx.plan.response().field_tol();
        if p.max_residual.is_nan() || p.max_residual > tol {
            let failure = ResponseFailure {
                what: "polarizability",
                iterations: p.cphf_iterations,
                max_residual: p.max_residual,
                tol,
            };
            self.unconverged_response = Some(UnconvergedResponse::Polarizability(Box::new(p)));
            return Err(LinusError::ResponseNotConverged(Box::new(failure)));
        }
        self.unconverged_response = None;
        Ok(p)
    }
}

#[cfg(test)]
mod tests {
    use crate::driver::{CalcOptions, JkMethod, Session};
    use crate::Structure;
    use linus_base::linalg::dot;
    use linus_base::terms::XcPotential;
    use linus_core::timing::Timers;
    use linus_ints::{PairOptions, PairSet};
    use linus_scf::{run_scf, ScfConfig, ScfInput};

    const WATER: &str = "3\n\nO 0.0 0.05 0.11\nH 0.0 0.76 -0.47\nH 0.1 -0.75 -0.46\n";
    const CH2: &str = "3\n\nC 0.000 0.003 0.102\nH 0.000 0.995 -0.310\nH 0.010 -0.990 -0.302\n";

    /// Polarizability against central differences of the SCF dipole moment in a finite field
    /// (h + F r_c): water RHF exact and B3LYP RI, triplet CH2 UHF exact and UKS (PBE, RI); the
    /// VV10 kernel with water wB97M-V (exact) and triplet CH2 B97M-V (RI).
    #[test]
    fn polarizability_vs_finite_field() {
        let df = || JkMethod::Df {
            aux: "def2-universal-jkfit".into(),
        };
        for (xyz, mult, method, jk) in [
            (WATER, 1, "hf", JkMethod::Exact),
            (WATER, 1, "b3lyp", df()),
            (CH2, 3, "hf", JkMethod::Exact),
            (CH2, 3, "pbe", df()),
            (WATER, 1, "wb97x", JkMethod::Exact),
            (CH2, 3, "cam-b3lyp", df()),
            (WATER, 1, "wb97m-v", JkMethod::Exact),
            (CH2, 3, "b97m-v", df()),
        ] {
            let mol = Structure::from_xyz_str(xyz, 0, mult).unwrap();
            let o = CalcOptions {
                method: Some(method.into()),
                basis: Some("def2-svp".into()),
                jk: jk.into(),
                grid: crate::driver::options::GridSettings {
                    xc: (method != "hf").then_some(crate::driver::GridChoice::Pyscf { level: 2 }),
                    ..Default::default()
                },
                threads: Some(4),
                scf: crate::driver::ScfSettings {
                    e_tol: Some(crate::driver::options::Auto::Value(1e-12)),
                    g_tol: Some(crate::driver::options::Auto::Value(1e-10)),
                    ..Default::default()
                },
                ..Default::default()
            };
            let mut calc = Session::new(mol.clone(), o).unwrap();
            let an = calc.polarizability().unwrap().alpha;
            let par = calc.par();
            let mut t = Timers::default();
            let pairs = PairSet::build(&calc.cx.basis, PairOptions::default());
            let (s, h) = calc.one_electron(&pairs);
            let mut jk = calc.build_jk(pairs, &mut t).unwrap().0;
            let mut xc = calc.build_xc(&mut t).unwrap();
            let r = linus_ints::reference::oneshot::dipole(&calc.cx.basis, [0.0; 3]);
            let n = calc.cx.basis.nbf;
            let rm: Vec<faer::Mat<f64>> = r
                .iter()
                .map(|x| linus_base::linalg::mat_from_rowmajor(n, n, x))
                .collect();
            let (na, nb) = mol.nalpha_nbeta().unwrap();
            let mut so = ScfConfig {
                restricted: Some(mult == 1),
                ..Default::default()
            };
            so.e_tol = 1e-12;
            so.g_tol = 1e-10;
            let f = 1e-4;
            let mut err: f64 = 0.0;
            for c in 0..3 {
                let mut mu = [[0.0; 3]; 2];
                for (k, sg) in [1.0, -1.0].iter().enumerate() {
                    let hh = &h + &rm[c] * (sg * f);
                    let inp = ScfInput {
                        h: hh.as_ref(),
                        s: s.as_ref(),
                        e_nuc: mol.nuclear_repulsion(),
                        nalpha: na,
                        nbeta: nb,
                        exchange: calc.method().exchange,
                        guess_density: None,
                        geometry: None,
                    };
                    let rr = run_scf(
                        &inp,
                        jk.as_mut(),
                        xc.as_mut().map(|x| x as &mut dyn XcPotential),
                        None,
                        &so,
                        par,
                    )
                    .unwrap();
                    assert!(rr.converged);
                    let dt = &rr.d_a + &rr.d_b;
                    for d in 0..3 {
                        mu[k][d] = -dot(dt.as_ref(), rm[d].as_ref());
                    }
                }
                for d in 0..3 {
                    err = err.max(((mu[0][d] - mu[1][d]) / (2.0 * f) - an[c][d]).abs());
                }
            }
            eprintln!("{method} mult {mult}: |alpha - finite field| = {err:e}");
            assert!(
                err < 2e-4,
                "{method} mult {mult}: |alpha - finite field| = {err:e} (alpha {an:?})"
            );
        }
    }
}
