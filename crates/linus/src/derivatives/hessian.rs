//! Analytic nuclear Hessians (HF and KS-DFT, closed and open shells; exact four-center
//! integrals or density fitting with the Coulomb metric).
//!
//! The Hessian is an assembler (`spin_hessian`): the explicit second derivatives at fixed
//! density (the core Hamiltonian's and the method families' terms, `crate::derivatives::terms`),
//! then the response part over the framework of `linus_scf::response`: the nuclear displacements as
//! a perturbation of the reference (`NuclearDisplacements`: F^x C_o from the same terms, the
//! overlap derivative), their CPHF/CPKS (`ResponseProblem`), and the second-order assembly with
//! its stationary correction (`second_order::SecondOrder`, `stationary`).
//!
//! Closed and open shells are one formulation over spin channels: one channel for a closed
//! shell (the alpha orbitals, occupation 2), two for UHF/UKS, with F_s^x, C1_s, e1_s and the
//! CPHF/CPKS per channel (all channels of all 3N perturbations in one batched solve), the
//! two-electron density terms, RI exchange per channel
//! ([`linus_jk::df::hessian::df_hessian_2e`]) and the XC terms and kernel of the
//! channels. The closed-shell formulas below are the one-channel case. The response is solved
//! in one shared subspace for all perturbations (`ResponseProblem::solve`, about half the
//! iterations of independent conjugate gradients), which also yields dF\[U\] C_o.
//! Validation: tests/python/compare_hess_pyscf.py (max |H - H_PySCF| <= 6e-9 Eh/bohr^2 for
//! UHF/UKS, exact and RI, def2-SVP/TZVP; bench/results/2026-11-07-uhf-hess).
//!
//! Every reduction is thread-count independent (fixed-point XC, `linus_num::det` for the
//! derivative-integral contractions, including the second-derivative ones: Eri4DerivPlan, eri_hess_3c/2c).
//!
//! H = H_nuc + [explicit second-derivative terms with fixed D and W] +
//!   4 sum F^x . (C1^y C_o^T) - 4 sum S^x . (C1^y e_o C_o^T) - 2 sum s1oo^x . e1^y,
//! with C1^y = C U^y the first-order occupied orbitals (U_oo = -1/2 S^y_oo, U_vo from CPHF)
//! and e1^y_ij = h1_ij + v1_ij - (e_i + e_j)/2 s1_ij the first-order orbital-energy matrix
//! (the formulation of PySCF's rhf Hessian, with the batched CPHF solver of `linus_scf::response`).
//! F^x (the derivative Fock matrix at fixed density) only enters through F^x C_o. The term
//! -4 U^x . r^y with the final CPHF residual r^y makes the response part stationary in U, so
//! the Hessian error is quadratic in the CPHF residual (default tolerance 1e-6).
//!
//! Kohn-Sham: exchange scaled by the functional's exact-exchange fraction a_x, the explicit XC
//! second derivative E_xc^xy and the XC potential derivative V_xc^x (in F^x) from
//! `XcIntegrator::hessian_terms`, and the XC kernel in the CPKS response (only dF C_o is
//! needed anywhere, so the response uses `ResponseOperator::induced_fock_occ` and the
//! occupied-projected kernel `XcIntegrator::kernel_contract_occ`). By default
//! (`grid.response`), the Hessian is the derivative of the gradient computed without grid
//! response, the grid moving with the atoms ("scheme 1" of Bykov et al., Mol. Phys. 113, 1961
//! (2015), as in ORCA): first-order weight and point-motion terms from
//! `XcIntegrator::grid_terms` enter the CPKS right-hand sides, e1 and the explicit part, while
//! the F^x contracted with the response stays the fixed-grid one (the result is symmetrized).
//! With `response.grid_response: false` the XC terms are taken on a fixed grid (no grid-weight or
//! point-motion terms), like PySCF's RKS Hessian: that convention is for comparisons only (on
//! production grids it is off by up to ~100 cm^-1 in soft modes).
//!
//! With density fitting (jk = df) every two-electron quantity is that of the RI energy: F^x = h^x +
//! J^x\[D_t\] - a_x K^x\[D_a\] with the RI derivative matrices (including the metric response) and
//! the explicit terms, the second derivatives of the RI energy at fixed orbitals (3c/2c
//! second-derivative integrals plus the products of first derivatives that come from V^-1), both
//! from [`linus_jk::df::hessian`] (blocked over auxiliary functions), and the CPHF uses the same RI
//! J/K (`linus_jk::df::response::DfResponseJk`, specialized to densities A C_o^T + C_o A^T and
//! projected on C_o). The result is the exact Hessian of the RI energy (PySCF's df Hessian with
//! full auxiliary basis response).
//!
//! Range-separated hybrids (libxc's (alpha, beta, omega): exchange a K + b K_LR(omega) with
//! a = alpha + beta, b = -beta; wB97X, wB97M, CAM-B3LYP, the wPBE family) take every exchange
//! term with the operator w = a/r + b erf(omega r)/r and a_x = 1 (the coefficients are in the
//! operator, so the response code is unchanged): exact integrals: the explicit second
//! derivatives of the exchange with w (`linus_ints::Eri4DerivPlan`, J stays 1/r), K^x of
//! F^x (`linus_ints::eri4c::jk_deriv::jk_deriv`) and the response K from FastJk with the operator set; RI:
//! the SCF's wK scheme (Psi4's: Coulomb metric, Coulomb 3-center integrals on one side),
//! [`linus_jk::df::hessian::df_hessian_2e`] (its wK path) for the explicit terms and F^x and
//! `linus_jk::df::response::DfResponseJk::new_op` for the response (two-level and single-precision
//! modes included). The semilocal part of the functional (its short-range DFA exchange
//! included) is libxc's, through the usual XC terms and kernel; D3BJ dispersion adds its own
//! Hessian. Validation: tests/python/compare_hess_pyscf.py (exact integrals, PySCF's RSH
//! Hessians), the finite-difference tests below and in hess_df / hess_resp
//! (bench/results/2026-11-07-rsh-hess).

use crate::derivatives::terms::{
    ExplicitHessian, HessianTerm, SolventTerm, TermInputs, TwoElectronTerm, XcTerm,
};
use crate::driver::ground::GroundState;
use crate::driver::{CalcContext, GridChoice, Purpose, ResponsePrecision, Session};
use crate::properties::dipole::{DipoleTerms, HessianResponse};
use crate::properties::polarizability::Polarizability;
use crate::two_electron::{ResponseSetup, ResponseUse};
use faer::{Mat, Par};
use linus_base::error::{LinusError, ResponseFailure};
use linus_base::linalg::mat_to_rowmajor;
use linus_base::memory::mem_mark;
use linus_core::timing::Timers;
use linus_dft::XcIntegrator;
use linus_ints::{BasisSet, DerivOrder, OneEPlan};
use linus_scf::reference::Reference;
use linus_scf::response::operator::ResponseOperator;
use linus_scf::response::perturbation::{
    occupied_projections, NuclearDisplacements, Overlap, Perturbation,
};
use linus_scf::response::problem::ResponseProblem;
use linus_scf::response::second_order::{refs, stacked_gram, FirstOrderOrbitals, SecondOrder};
use linus_scf::response::stationary::Correction;

#[derive(Clone, Debug)]
pub struct HessianResult {
    pub energy: f64,
    pub gradient: Vec<[f64; 3]>,
    /// (3N x 3N) row-major, Eh/bohr^2, index 3 * atom + k
    pub hessian: Vec<f64>,
    pub cphf_iterations: usize,
    /// What the CPHF/CPKS used and reached (the result's `resolved.record.response`).
    pub response: ResolvedCphf,
    /// Dipole derivatives dmu_c/dX (3N rows, index 3 * atom + k; components c = x, y, z; e),
    /// dipole about the coordinate origin as [`Session::dipole_moment`]; with
    /// [`HessianOptions::dipole_derivatives`].
    pub dipole_derivatives: Option<Vec<[f64; 3]>>,
    /// The polarizability whose electric-field CPHF/CPKS the dipole derivatives' stationary
    /// form used (with [`HessianOptions::dipole_derivatives`]; its own timers; checked against
    /// `response.field_tol` by [`Session::ir_properties`], as by
    /// [`Session::polarizability`]).
    pub polarizability: Option<Polarizability>,
    pub timers: Timers,
}

/// What the CPHF/CPKS of a Hessian used and reached.
#[derive(Clone, Debug, serde::Serialize)]
pub struct ResolvedCphf {
    /// Max-abs residual the solve aimed at (`response.tol`, or its defaults).
    pub cphf_tol: f64,
    /// The response J/K computed in single precision.
    pub single_precision: bool,
    /// Two-level response: the iterations ran with a cheaper J/K.
    pub two_level: bool,
    /// The auxiliary basis of the iterations' J/K (two-level responses).
    pub iteration_aux: Option<String>,
    /// The response precision as set (`precision.response`, FP64 under `precision.fp32: off`).
    pub precision_setting: ResponsePrecision,
    /// Max-abs residual of the last iteration (at most [`HESSIAN_RESIDUAL_MARGIN`] x
    /// `cphf_tol`, or the Hessian is an error).
    pub max_residual: f64,
    /// The grid of the CPKS kernel when it differs from the SCF's (`cpks_grid`).
    pub grid: Option<GridChoice>,
}

/// The CPHF/CPKS of a Hessian counts as converged when its max-abs residual is at most this
/// multiple of the tolerance. The solver stops at the tolerance unless it runs out of
/// iterations or of new directions; the Hessian error is quadratic in the residual, so a solve
/// that stalled within this margin still gives a Hessian error within 100x the target's.
pub const HESSIAN_RESIDUAL_MARGIN: f64 = 10.0;

/// XC integrators of a Hessian: the SCF grid's for the explicit terms and F^x, and the one of
/// the CPKS kernel (`cpks_grid`; the same unless it differs).
#[derive(Clone, Copy)]
struct Xcs<'a> {
    main: Option<&'a XcIntegrator>,
    cpks: Option<&'a XcIntegrator>,
}

/// What [`Session::hessian_with`] computes besides the Hessian.
#[derive(Clone, Debug, Default)]
pub struct HessianOptions {
    /// Also the dipole derivatives (IR intensities) from the Hessian's own CPHF solutions, in
    /// their stationary form (`crate::properties::dipole`). They take the polarizability's
    /// electric-field CPHF/CPKS, which IR analyses need anyway: it runs after the Hessian is
    /// complete (the Hessian does not change) and comes back as
    /// [`HessianResult::polarizability`].
    pub dipole_derivatives: bool,
}

/// Max-abs residual of the CPHF/CPKS solves (`response.tol` overrides): 1e-6, and in two-level
/// responses (large systems, single-precision J/K) [`CPKS_TOL_TWO_LEVEL`]: the Hessian error is
/// quadratic in the residual, there below the single-precision noise (u1an39 wB97X-D3BJ/def2-SVP:
/// one CPKS iteration less, |dH| 9e-7 against 4e-7 from the precision, frequencies 0.001 cm^-1).
pub(crate) fn cpks_tol(tol: Option<f64>, two_level: bool) -> f64 {
    tol.unwrap_or(if two_level { CPKS_TOL_TWO_LEVEL } else { 1e-6 })
}

/// See `cpks_tol`.
pub const CPKS_TOL_TWO_LEVEL: f64 = 1e-5;

/// Merge the response J/K builders' own timers (e.g. DfResponseJk's integral and GEMM phases) as
/// resp_main_* and resp_iter_*.
fn merge_builder_timers(timers: &mut Timers, main: Option<Timers>, iter: Option<Timers>) {
    for (prefix, t) in [("resp_main_", main), ("resp_iter_", iter)] {
        if let Some(t) = t {
            for (k, v, _) in &t.entries {
                timers.add(&format!("{prefix}{}", k.trim_start_matches("resp_")), *v);
            }
            for (k, v) in &t.resources {
                timers.resource(&format!("{prefix}{}", k.trim_start_matches("resp_")), *v);
            }
        }
    }
}

/// Hessian of the nuclear repulsion (3N x 3N).
pub fn nuclear_repulsion_hessian(z: &[u32], xyz: &[[f64; 3]]) -> Vec<f64> {
    let n = z.len();
    let n3 = 3 * n;
    let mut h = vec![0.0; n3 * n3];
    for a in 0..n {
        for b in 0..n {
            if a == b {
                continue;
            }
            let r = [
                xyz[a][0] - xyz[b][0],
                xyz[a][1] - xyz[b][1],
                xyz[a][2] - xyz[b][2],
            ];
            let d2: f64 = r.iter().map(|x| x * x).sum();
            let d = d2.sqrt();
            let zz = (z[a] * z[b]) as f64;
            for k in 0..3 {
                for l in 0..3 {
                    // d^2/dA_k dA_l (Z/|A-B|) = Z (3 r_k r_l / d^5 - delta_kl / d^3)
                    let v = zz
                        * (3.0 * r[k] * r[l] / (d2 * d2 * d)
                            - if k == l { 1.0 / (d2 * d) } else { 0.0 });
                    h[(3 * a + k) * n3 + 3 * a + l] += v;
                    h[(3 * a + k) * n3 + 3 * b + l] -= v;
                }
            }
        }
    }
    h
}

impl CalcContext {
    /// The memory the fixed-size arrays of a response calculation need (bytes; `what`: the
    /// Hessian or the polarizability), estimated before any SCF from the basis, the occupied
    /// orbitals and the auxiliary basis: a lower bound of the peak, without the CPHF subspace
    /// (which grows with the iterations):
    /// - Hessian: the first-derivative AO matrices alive together (S^x with h^x, the channels'
    ///   V_xc^x or the solvent's F^x; 3N n^2 each), the nine response arrays per perturbation
    ///   and channel (F^x C_o, S^x C_o, dF C_o, C1, U, ...; 3N n o), and with density fitting the
    ///   explicit terms' Z (n naux o) or the response J/K's B (n naux o, 4 bytes per element in
    ///   single precision);
    /// - polarizability: six response arrays of the three field perturbations and the response
    ///   J/K's B (double precision).
    pub(crate) fn response_memory(&self, what: ResponseUse) -> Result<f64, LinusError> {
        let n = self.basis.nbf as f64;
        let n3 = 3.0 * self.mol.natom() as f64;
        let (na, nb) = self.mol.nalpha_nbeta().map_err(LinusError::Input)?;
        let restricted = self.plan.restricted().unwrap_or(na == nb);
        let (nch, o): (f64, f64) = if restricted {
            (1.0, na as f64)
        } else {
            (2.0, (na + nb) as f64)
        };
        let naux = match self.plan.jk().aux() {
            Some(a) => {
                let def = self.plan.load_basis(a.for_purpose(Purpose::Hessian))?;
                BasisSet::new(&def, &self.mol.z, &self.mol.positions, true)?.nbf as f64
            }
            None => 0.0,
        };
        let f64b = 8.0;
        Ok(match what {
            ResponseUse::Hessian => {
                let ao = f64b * n3 * n * n * (1.0 + nch.max(1.0));
                let mo = f64b * n3 * n * o * 9.0;
                let resp_bytes = match self.plan.response_precision() {
                    ResponsePrecision::Fp64 => 8.0,
                    _ => 4.0,
                };
                let df_explicit = f64b * n * naux * o;
                let df_response = resp_bytes * n * naux * o;
                mo + ao.max(df_explicit).max(df_response)
            }
            ResponseUse::Polarizability => f64b * 3.0 * n * o * 6.0 + f64b * n * naux * o,
        })
    }

    /// HF/KS Hessian (without dispersion) in spin channels: one for a closed shell (the alpha
    /// orbitals, occupation f = 2), two for an open shell (f = 1 per spin),
    ///   H = H_nuc + [explicit terms with D_t = sum_s f D_s, W = sum_s f W_s, the two-electron
    ///       density terms and the XC second derivative]
    ///     + sum_s f [2 F_s^x . (C1_s^y C_os^T) - 2 S^x . (C1_s^y e_os C_os^T) - s1oo_s^x . e1_s^y
    ///                - 2 U_s^x . r_s^y (- 2 dV_grid,s^x C_os . C1_s^y)],
    /// with F_s^x = h^x + J^x[D_t] - a_x K^x[D_s] + V_xc,s^x, C1_s = C_vs U_s - 1/2 C_os s1oo_s,
    /// e1_s the first-order occupied orbital energies and r_s the final CPHF residuals (the
    /// stationary term): the explicit terms, then [`NuclearDisplacements`] with their
    /// [`ResponseProblem`] (U_s from the CPHF/CPKS; dF only enters as dF_s C_os) and
    /// [`SecondOrder`], plus the grid terms. With `want_dip` also the dipole derivatives, before
    /// their stationary correction, with the residuals it needs.
    #[allow(clippy::type_complexity)]
    fn spin_hessian(
        &self,
        gs: &GroundState,
        reference: &Reference,
        xcs: Xcs<'_>,
        want_dip: bool,
        par: Par,
        timers: &mut Timers,
    ) -> Result<(Vec<f64>, usize, Option<DipoleTerms>, ResolvedCphf), LinusError> {
        let xc = xcs.main;
        let basis = &self.basis;
        let n = basis.nbf;
        let n2 = n * n;
        let n3 = 3 * self.mol.natom();
        // the reference's channels (crate::reference): the alpha orbitals of a closed shell
        // (occupation 2), or both spins; their densities D_s, D_t and W
        let nch = reference.nch();
        let focc = reference.occupation();
        let cor = reference.c_occ();
        let ds: Vec<&Mat<f64>> = reference.channels.iter().map(|b| &b.density).collect();
        let dt = &reference.total_density;
        let dtf = mat_to_rowmajor(dt.as_ref());
        let dsf: Vec<Vec<f64>> = ds.iter().map(|d| mat_to_rowmajor(d.as_ref())).collect();
        let ws: Vec<f64> = mat_to_rowmajor(reference.energy_weighted.as_ref())
            .into_iter()
            .map(|x| -x)
            .collect();
        // the method's exact exchange (crate::scf::exchange_parts: a_x K, or K of the operator)
        let xop = self.method.exchange;
        let a_x = linus_base::terms::exchange_parts(xop).0;
        let charges = self.mol.point_charges();
        // the geometry's pairs (crate::context) with second derivatives (its gradient built them)
        let pairs = timers.time("hess_pairs", || {
            self.integrals.pairs(basis, DerivOrder::Two)
        });
        // explicit second-derivative terms (nuclear, one-electron)
        let mut h = nuclear_repulsion_hessian(&self.mol.effective_z(), &self.mol.positions);
        let add = |h: &mut Vec<f64>, x: &[f64]| h.iter_mut().zip(x).for_each(|(a, b)| *a += b);
        let one_e = OneEPlan::new(basis, DerivOrder::Two)
            .with_charges(&pairs, &charges)
            .expect("pairs of derivative order two");
        let mut wk = one_e.workspace();
        let mut h1 = vec![0.0; h.len()];
        timers.time("hess_1e", || {
            one_e.overlap_kinetic_hess(Some(&ws), Some(&dtf), &mut h1, &mut wk)
        });
        add(&mut h, &h1);
        timers.time("hess_1e", || {
            one_e.nuclear_attraction_hess(&dtf, &mut h1, &mut wk)
        });
        add(&mut h, &h1);
        drop((wk, h1));
        // first-derivative matrices: S^x and F_s^x C_s = (h^x + J^x[D_t] - a_x K^x[D_s] +
        // V_xc,s^x) C_s (range-separated hybrids: a_x = 1 and K of the operator
        // a/r + b erf(omega r)/r), as fxo[s][x]
        let mut sx = vec![0.0; 3 * basis.natom() * n2];
        timers.time("hess_fock_deriv", || {
            one_e.overlap_kinetic_deriv(Some(&mut sx), None)
        });
        let mut hx = vec![0.0; sx.len()];
        timers.time("hess_fock_deriv", || one_e.hcore_deriv(&mut hx));
        timers.time("hess_ecp", || {
            self.add_ecp_hessian_terms(&pairs, &dtf, &mut h, &mut hx)
        });
        let fxo = occupied_projections(reference, &hx, par);
        drop(hx);
        // the method families' explicit terms in their fixed order (crate::derivatives::terms):
        // the two-electron method's, then its response J/K (the right-hand sides and the final
        // dF[U] C_o; CalcContext::response_jk. Two-level responses (large systems; the CPKS
        // kernel's grid unchanged): the iterations use a cheaper J/K, built from `iter_plan`
        // after the right-hand sides, when the main one has released its tensors; the Hessian
        // error is quadratic in the difference of the iteration J/K from the main one); the
        // solvent's (its reaction field's response joins the response operator); the XC
        // engine's (the semilocal terms, then VV10's at its response ground state on the NLC
        // grid, which the CPKS kernel shares; with the grid response its first-order terms:
        // dV_s^y C_s for the response, d/dy of the fixed-grid gradient in the explicit part);
        // then the grid terms join F^x
        let dsr: Vec<&[f64]> = dsf.iter().map(|d| &d[..]).collect();
        let inputs = TermInputs {
            ground: gs,
            reference,
            c_occ: cor.clone(),
            dtf: &dtf,
            dsf: dsr,
            par,
        };
        let mut explicit = ExplicitHessian::new(h, fxo);
        let mut two_e = TwoElectronTerm::new(self, xop, &pairs, want_dip);
        two_e.add_hessian(&inputs, &mut explicit, timers)?;
        let ResponseSetup {
            jk,
            plan: mut iter_plan,
            single_precision,
            kernel: kernel_use,
        } = two_e.into_response();
        let two_level = iter_plan.is_some();
        let mut solvent_term = SolventTerm::new(self);
        solvent_term.add_hessian(&inputs, &mut explicit, timers)?;
        let solvent = solvent_term.into_response();
        mem_mark(timers, "explicit_2e");
        // the pairs' derivative expansions are done with: freed (E stays for the response)
        drop(pairs);
        self.integrals.release_pair_derivatives();
        let pairs = self.integrals.pairs(basis, DerivOrder::Zero);
        let grid_response = self.plan.grid_response();
        let mut xc_term = XcTerm::new(self, xc, grid_response);
        xc_term.add_hessian(&inputs, &mut explicit, timers)?;
        let nl_ground = xc_term.into_vv10_ground();
        let nl = xc.and_then(|x| x.vv10());
        let (mut h, fxo, gxo) = explicit.finish();
        // CPHF/CPKS; with the stationary response term below the Hessian error is quadratic in
        // the residual, so a max-abs residual of 1e-6 gives ~1e-11 Eh/bohr^2 (response.tol
        // overrides). With first-order grid terms the response is contracted with the fixed-grid
        // F^x, which leaves an error linear in the residual times the (small) grid term.
        // Two-level responses (large systems, single precision): 1e-5 (CPKS_TOL_TWO_LEVEL).
        let cphf_tol = cpks_tol(self.plan.response_tol(), two_level);
        let iteration_aux = iter_plan.as_ref().map(|p| p.aux_name.clone());
        // the CPKS kernel (CPU, or the device's when the backend puts it there), with VV10's
        let vv10 = match (nl, &nl_ground) {
            (Some(nl), Some(g)) => Some(linus_dft::vv10_resp::Vv10Kernel::new(nl, g.clone())),
            _ => None,
        };
        let kernel = self.response_kernel(xcs.cpks, vv10, &cor, kernel_use, timers)?;
        // the response operator (crate::response::operator): the J/K, the CPKS kernel and the
        // solvent's reaction field, the last two the same at both levels of a two-level response
        let operator = ResponseOperator::new(&cor, jk, a_x)
            .with_kernel(kernel.as_ref().map(|k| k.kernel()))
            .with_environment(solvent);
        let mut problem = ResponseProblem::new(reference, operator);
        problem.report_to(self.progress.clone(), "hessian");
        // the nuclear displacements as a perturbation of the reference
        // (crate::response::perturbation): F^x C_o from the terms above, S^x C_o and s1oo
        let nuclear = NuclearDisplacements {
            reference,
            fock_occ: fxo,
            overlap: Overlap::new(reference, &sx, par),
        };
        drop(sx);
        // the induced Fock matrix of the occupied blocks' first-order density
        // D^x_oo,s = -C_os s1oo_s C_os^T, times C_os (only dF C_o is used below)
        let voo = timers.time("hess_cphf", || problem.induced_oo(&nuclear, par))?;
        if let Some(plan) = iter_plan.take() {
            // two-level: the main J/K is idle during the iterations: it frees its tensors until
            // the final product, then the iteration J/K is built
            problem.operator().begin_iterations(|| {
                Box::new(timers.time("hess_cphf_setup", || plan.build(basis, &pairs, &cor, par)))
            });
        }
        mem_mark(timers, "rhs");
        let rhs = nuclear.rhs_vo(voo.as_deref(), par);
        // shared-subspace solver (returns dF[U] C_o too, unless two-level)
        let sol = timers.time("hess_cphf", || {
            problem.solve(&rhs, cphf_tol, 100, !two_level, par)
        })?;
        let (u, st) = (sol.u, sol.stats);
        mem_mark(timers, "cpks_iterations");
        let vvo = match sol.fu {
            Some(fu) => fu,
            None => {
                // two-level: the final dF[U] C_o with the main operator (error quadratic in the
                // difference of the iteration operator)
                merge_builder_timers(timers, None, problem.operator().end_iterations());
                timers.time("hess_cphf_setup", || problem.operator().restore(par));
                timers.time("hess_cphf", || problem.induced_vo(&u, par))?
            }
        };
        merge_builder_timers(timers, problem.operator().jk_timers(), None);
        mem_mark(timers, "final_product");
        // CPHF residuals r^y = rhs^y - (A+B) U^y (free: (A+B) U = (e_a - e_i) U + C_v^T vvo C_o)
        let resid = problem.residuals(&rhs, &u, &vvo, par);
        // IR: the response J/K goes on to the electric-field CPHF when it serves those products
        // too (ResponseJk::serve: the device's in double precision, with the stored exchange
        // operator it built for the 3N perturbations); else it goes here
        let (mut jk, phases) = problem.into_parts();
        let field_jk = if want_dip && !two_level && jk.serve(ResponseUse::Polarizability.name()) {
            Some(jk)
        } else {
            drop(jk);
            None
        };
        timers.merge(&phases);
        if let Some(k) = &kernel {
            k.record_vv10("hess_cphf", timers);
        }
        drop(rhs);
        // the response's first-order occupied orbitals C1 = C_v U - 1/2 C_o s1oo and orbital
        // energies e1 (crate::response::second_order)
        let orbitals = FirstOrderOrbitals::new(&nuclear, &u, voo.as_deref(), &vvo, par);
        drop((voo, vvo));
        timers.time("hess_assemble", || {
            // f [2 (F^x C_o) . C1^y - 2 (S^x C_o) . (C1^y e_o) - s1oo^x . e1^y - 2 U^x . r^y]
            // per channel (crate::response::second_order), as GEMMs of the stacked (vectorized)
            // matrices; the U . r term makes the response part stationary in U (Hylleraas form,
            // crate::response::stationary): its error is quadratic in the CPHF residual. Then the
            // grid terms' - 2 f dV_grid^x C_o . C1^y
            let response = SecondOrder {
                rows: &nuclear,
                columns: &orbitals,
                correction: Some(Correction {
                    u: &u,
                    residuals: &resid,
                }),
            };
            let mut hr = Mat::<f64>::zeros(n3, n3);
            for s in 0..nch {
                response.add_channel(&mut hr, s, focc, par);
                if !gxo.is_empty() {
                    // the gradient's F^x is the fixed-grid one
                    let c1 = refs(&orbitals.c1[s]);
                    stacked_gram(&mut hr, -2.0 * focc, &refs(&gxo[s]), &c1, par);
                }
            }
            for x in 0..n3 {
                for y in 0..n3 {
                    h[x * n3 + y] += hr[(x, y)];
                }
            }
        });
        // symmetrize (the response part is symmetric only at convergence)
        for x in 0..n3 {
            for y in 0..x {
                let v = 0.5 * (h[x * n3 + y] + h[y * n3 + x]);
                h[x * n3 + y] = v;
                h[y * n3 + x] = v;
            }
        }
        // IR: the dipole derivatives of this response (crate::properties::dipole), before the
        // stationary correction that needs the field's response
        let dip = want_dip.then(|| {
            timers.time("hess_dipole", || {
                let h = HessianResponse {
                    nuclear: &nuclear,
                    dtf: &dtf,
                    orbitals: &orbitals,
                    residuals: resid,
                    jk: field_jk,
                };
                self.dipole_terms(h, par)
            })
        });
        let resolved = ResolvedCphf {
            cphf_tol,
            single_precision,
            two_level,
            iteration_aux,
            precision_setting: self.plan.response_precision(),
            max_residual: st.max_residual,
            grid: None,
        };
        Ok((h, st.iterations, dip, resolved))
    }
}

impl Session {
    /// Energy, gradient and analytic Hessian at the current geometry.
    pub fn hessian(&mut self) -> Result<HessianResult, LinusError> {
        self.hessian_with(&HessianOptions::default())
    }

    /// [`Self::hessian`] with optional extras (dipole derivatives).
    pub fn hessian_with(&mut self, hopts: &HessianOptions) -> Result<HessianResult, LinusError> {
        // what the method cannot do fails here, before any SCF
        self.cx
            .check_request(crate::derivatives::capabilities::Request::Hessian {
                dipole_derivatives: hopts.dipole_derivatives,
            })?;
        // the SCF and its reference (crate::reference), which the gradient, the Hessian and an
        // IR run's field response share
        let gs = self.scf_stage(Purpose::Hessian)?;
        let reference = gs.reference(&self.cx.pool, self.par());
        let g = self
            .cx
            .progress
            .stage("gradient", || self.cx.gradient_at(&gs, reference))?;
        // the gradient's phases (grad_*) are this call's too
        let mut timers = Timers::default();
        timers.merge(&g.timers);
        mem_mark(&mut timers, "scf_gradient");
        let par = self.par();
        let pool = &self.cx.pool;
        // the XC integrators (crate::xc): the SCF grid's (the geometry's, crate::context: the
        // SCF's and the gradient's), and the CPKS kernel's on its own (coarser) grid if
        // `cpks_grid` asks for one, alive through the response only
        let scf_grid = gs.grid();
        let xc = pool.install(|| self.cx.xc_on(scf_grid, &mut timers))?;
        let cpks_grid = self
            .cx
            .plan
            .response_grid(
                scf_grid,
                self.cx
                    .method
                    .functional
                    .as_ref()
                    .is_some_and(|f| f.is_meta_gga()),
            )
            .filter(|_| xc.is_some());
        // (VV10's kernel comes from the main integrator, on the NLC grid)
        let xc_cpks = match &cpks_grid {
            Some(g) => pool.install(|| self.cx.build_xc_on(g, false, &mut timers))?,
            None => None,
        };
        let xcs = Xcs {
            main: xc.as_deref(),
            cpks: xc_cpks.as_ref().or(xc.as_deref()),
        };
        let want_dip = hopts.dipole_derivatives;
        let (mut hess, iters, dip, mut response) = self
            .cx
            .progress
            .stage("hessian", || {
                pool.install(|| {
                    self.cx
                        .spin_hessian(&gs, reference, xcs, want_dip, par, &mut timers)
                })
            })
            // stopped on request: with the gradient it reached
            .map_err(|e| e.reached(Some(&g.gradient), None))?;
        drop(xc_cpks);
        response.grid = cpks_grid;
        if let Some(d) = self.cx.dispersion(2)? {
            if let Some(h) = d.hessian {
                hess.iter_mut().zip(&h).for_each(|(a, b)| *a += b);
            }
        }
        // the SCF's hand-off (crate::backend::gpu::session) is done with
        self.cx.release_handoff();
        let mut res = HessianResult {
            energy: gs.scf.energy,
            gradient: g.gradient,
            hessian: hess,
            cphf_iterations: iters,
            response,
            dipole_derivatives: dip.as_ref().map(|d| d.apt.clone()),
            polarizability: None,
            timers,
        };
        // an unconverged response is an error that keeps its result
        let rr = &res.response;
        if rr.max_residual.is_nan() || rr.max_residual > HESSIAN_RESIDUAL_MARGIN * rr.cphf_tol {
            let failure = ResponseFailure {
                what: "analytic Hessian",
                iterations: res.cphf_iterations,
                max_residual: rr.max_residual,
                tol: rr.cphf_tol,
            };
            self.unconverged_response =
                Some(crate::driver::UnconvergedResponse::Hessian(Box::new(res)));
            return Err(LinusError::ResponseNotConverged(Box::new(failure)));
        }
        // a residual within the tolerance over a non-finite Hessian is a numerical failure
        if res.hessian.iter().any(|x| !x.is_finite()) {
            return Err(LinusError::Numerical(
                "the analytic Hessian is not finite (NaN or infinite)".into(),
            ));
        }
        self.unconverged_response = None;
        // IR: the polarizability's electric-field CPHF/CPKS (the SCF grid's XC kernel, as
        // Session::polarizability), after the Hessian is complete, and with its solutions
        // the dipole derivatives' stationary correction (HessianOptions::dipole_derivatives);
        // on the Hessian's response J/K when it serves these products (the device's stored
        // exchange operator)
        if let Some(terms) = dip {
            let (apt, polarizability) = self
                .cx
                .stationary_dipole_derivatives(&gs, reference, xc.as_deref(), terms, par)
                .map_err(|e| e.reached(Some(&res.gradient), Some(&res.hessian)))?;
            res.dipole_derivatives = Some(apt);
            res.polarizability = Some(polarizability);
        }
        Ok(res)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::driver::options::{Auto, GridSettings, Reference};
    use crate::driver::{CalcOptions, GridChoice, JkMethod};
    use crate::Structure;

    const WATER: &str = "3\n\nO 0.000 0.000 0.118\nH 0.020 0.755 -0.471\nH -0.010 -0.760 -0.465\n";
    const CH2: &str = "3\n\nC 0.000 0.003 0.102\nH 0.000 0.995 -0.310\nH 0.010 -0.990 -0.302\n";

    fn opts(method: &str, jk: JkMethod, restricted: Option<bool>) -> CalcOptions {
        CalcOptions {
            method: Some(method.into()),
            basis: Some("def2-svp".into()),
            jk: jk.into(),
            // an XC grid only for DFT (Hartree-Fock has none: D41)
            grid: GridSettings {
                xc: (method != "hf").then_some(GridChoice::Pyscf { level: 1 }),
                ..Default::default()
            },
            reference: restricted.map(|r| {
                if r {
                    Reference::Restricted
                } else {
                    Reference::Unrestricted
                }
            }),
            threads: Some(4),
            scf: crate::driver::ScfSettings {
                e_tol: Some(Auto::Value(1e-11)),
                g_tol: Some(Auto::Value(1e-10)),
                ..Default::default()
            },
            ..Default::default()
        }
    }

    fn max_diff(a: &[f64], b: &[f64]) -> f64 {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, linus_base::linalg::max_nan)
    }

    /// The open-shell code on a closed shell (restricted = false) reproduces the closed-shell
    /// Hessian: HF exact, B3LYP RI, r2SCAN exact, and PBE RI with the grid-response terms.
    #[test]
    fn unrestricted_closed_shell_equals_restricted() {
        let df = || JkMethod::Df {
            aux: "def2-universal-jkfit".into(),
        };
        for (method, jk, grid_resp) in [
            ("hf", JkMethod::Exact, false),
            ("b3lyp", df(), false),
            ("r2scan", JkMethod::Exact, false),
            ("pbe", df(), true),
        ] {
            let mol = Structure::from_xyz_str(WATER, 0, 1).unwrap();
            let run = |restricted| {
                let mut o = opts(method, jk.clone(), Some(restricted));
                if method != "hf" {
                    o.grid.response = Some(grid_resp);
                }
                Session::new(mol.clone(), o)
                    .unwrap()
                    .hessian()
                    .unwrap()
                    .hessian
            };
            let (hr, hu) = (run(true), run(false));
            let e = max_diff(&hr, &hu);
            assert!(e < 1e-8, "{method}: |H_UHF - H_RHF| = {e:e}");
        }
    }

    /// Dipole derivatives from the Hessian's CPHF solutions against central differences of the
    /// SCF dipole moment: RHF (water, exact), RKS (water, B3LYP, RI) and UKS (triplet CH2, B3LYP,
    /// RI; and NH3+, HF, a charged molecule), range-separated hybrids (water wB97X RI, triplet
    /// CH2 CAM-B3LYP exact); KS with the first-order grid terms.
    #[test]
    fn dipole_derivatives_match_finite_differences() {
        let df = || JkMethod::Df {
            aux: "def2-universal-jkfit".into(),
        };
        let nh3p = "4\n\nN 0.000 0.000 0.040\nH 0.000 1.010 -0.050\nH 0.880 -0.505 -0.060\nH -0.885 -0.500 -0.045\n";
        for (xyz, charge, mult, method, jk) in [
            (WATER, 0, 1, "hf", JkMethod::Exact),
            (WATER, 0, 1, "b3lyp", df()),
            (CH2, 0, 3, "b3lyp", df()),
            (nh3p, 1, 2, "hf", JkMethod::Exact),
            (WATER, 0, 1, "wb97x", df()),
            (CH2, 0, 3, "cam-b3lyp", JkMethod::Exact),
        ] {
            let mol = Structure::from_xyz_str(xyz, charge, mult).unwrap();
            let o = opts(method, jk.clone(), None);
            // the differences move the grid with the atoms: the response must too (on this
            // coarse grid the fixed-grid response is off by ~2e-4)
            let mut calc = Session::new(mol.clone(), o).unwrap();
            let an = calc
                .hessian_with(&HessianOptions {
                    dipole_derivatives: true,
                })
                .unwrap()
                .dipole_derivatives
                .unwrap();
            let h = 1e-3;
            let mut err: f64 = 0.0;
            for x in 0..3 * mol.natom() {
                let mut mu = Vec::new();
                for s in [1.0, -1.0] {
                    let mut c = mol.positions.clone();
                    c[x / 3][x % 3] += s * h;
                    calc.set_positions(&c).unwrap();
                    let r = calc.energy().unwrap();
                    mu.push(calc.dipole_moment(&r.d_a, &r.d_b));
                }
                for k in 0..3 {
                    err = err.max(((mu[0][k] - mu[1][k]) / (2.0 * h) - an[x][k]).abs());
                }
            }
            assert!(
                err < 2e-5,
                "{method} {charge} {mult}: |dmu/dx - fd| = {err:e}"
            );
        }
    }

    /// The stationary dipole derivatives and polarizability: at a loose CPHF tolerance (1e-4,
    /// both solves) the dipole derivatives lie within 1.6e-9 (water B3LYP RI), 2.5e-8 (triplet
    /// CH2 B3LYP RI) and 1.0e-9 au (water HF exact) of a tightly converged run's (without the
    /// U^c . r^x term: 8.9e-6, 1.0e-4 and 9.6e-6 au), the polarizabilities within 1.2e-8,
    /// 9.9e-8 and 8.5e-10 au; and the Hessian is bitwise the same as without dipole derivatives.
    #[test]
    fn dipole_derivatives_and_polarizability_are_stationary() {
        let df = || JkMethod::Df {
            aux: "def2-universal-jkfit".into(),
        };
        for (xyz, mult, method, jk) in [
            (WATER, 1, "b3lyp", df()),
            (CH2, 3, "b3lyp", df()),
            (WATER, 1, "hf", JkMethod::Exact),
        ] {
            let mol = Structure::from_xyz_str(xyz, 0, mult).unwrap();
            let run = |tol: f64, dip: bool| {
                let mut o = opts(method, jk.clone(), None);
                if jk != JkMethod::Exact {
                    o.precision.response = Some(ResponsePrecision::Fp64);
                }
                o.response.tol = Some(Auto::Value(tol));
                o.response.field_tol = Some(tol);
                let mut calc = Session::new(mol.clone(), o).unwrap();
                let h = calc
                    .hessian_with(&HessianOptions {
                        dipole_derivatives: dip,
                    })
                    .unwrap();
                let ir = dip.then(|| calc.ir_properties(&h).unwrap());
                (h, ir)
            };
            let (h_loose, ir_loose) = run(1e-4, true);
            let (_, ir_tight) = run(1e-9, true);
            let (h_plain, _) = run(1e-4, false);
            for (a, b) in h_loose.hessian.iter().zip(&h_plain.hessian) {
                assert_eq!(a.to_bits(), b.to_bits(), "{method} mult {mult}: Hessian");
            }
            let (l, t) = (ir_loose.unwrap(), ir_tight.unwrap());
            let dmu = l
                .apt
                .iter()
                .flatten()
                .zip(t.apt.iter().flatten())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0, linus_base::linalg::max_nan);
            let dalpha = l
                .polarizability
                .iter()
                .flatten()
                .zip(t.polarizability.iter().flatten())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0, linus_base::linalg::max_nan);
            println!("{method} mult {mult}: |d dmu| {dmu:.1e}, |d alpha| {dalpha:.1e} at tol 1e-4");
            assert!(dmu < 1e-7, "{method} mult {mult}: |d dmu/dx| = {dmu:e}");
            assert!(
                dalpha < 2e-7,
                "{method} mult {mult}: |d alpha| = {dalpha:e}"
            );
        }
    }

    /// Open-shell analytic Hessians against central differences of analytic gradients (triplet
    /// CH2): UHF exact and RI, and UKS (PBE, RI) with the first-order grid terms against
    /// gradients without grid response on a moving grid.
    #[test]
    fn open_shell_matches_finite_differences() {
        let df = || JkMethod::Df {
            aux: "def2-universal-jkfit".into(),
        };
        for (method, jk, grid_resp) in [
            ("hf", JkMethod::Exact, false),
            ("hf", df(), false),
            ("pbe", df(), true),
        ] {
            let mol = Structure::from_xyz_str(CH2, 0, 3).unwrap();
            let mut o = opts(method, jk.clone(), None);
            if method != "hf" {
                o.grid.response = Some(grid_resp);
            }
            let mut calc = Session::new(mol.clone(), o).unwrap();
            let an = calc.hessian().unwrap().hessian;
            if method != "hf" {
                // the Hessian's grid response against gradients without one (one setting now)
                calc.cx.plan.request_mut().grid.response = Some(false);
            }
            let n3 = 9;
            let h = 2e-3;
            let mut fd = vec![0.0; n3 * n3];
            for y in 0..n3 {
                let mut g = Vec::new();
                for s in [1.0, -1.0, 2.0, -2.0] {
                    let mut c = mol.positions.clone();
                    c[y / 3][y % 3] += s * h;
                    calc.set_positions(&c).unwrap();
                    g.push(calc.gradient().unwrap().gradient);
                }
                for x in 0..n3 {
                    let d = |i: usize| g[i][x / 3][x % 3];
                    fd[x * n3 + y] = (8.0 * (d(0) - d(1)) - (d(2) - d(3))) / (12.0 * h);
                }
            }
            let fds: Vec<f64> = (0..n3 * n3)
                .map(|i| 0.5 * (fd[i] + fd[(i % n3) * n3 + i / n3]))
                .collect();
            let e = max_diff(&an, &fds);
            assert!(
                e < 2e-6,
                "{method} {jk:?} grid_response={grid_resp}: |H - H_fd| = {e:e}"
            );
        }
    }

    /// One-electron doublets with RI (no occupied beta orbitals: the RI code paths must skip an
    /// empty spin channel), HF, B3LYP and wB97X: energy, gradient and Hessian of the H atom
    /// (gradient and Hessian vanish) and H2+ against exact J/K, the H atom polarizability, and
    /// the H2+ RI Hessian against central differences of RI gradients. cc-pVTZ-jkfit: with
    /// def2-universal-jkfit the wK exchange of hydrogen carries a large fitting error (wB97X
    /// H atom -6.5e-4 Eh, H2 -2.2e-4; cc-pVTZ-jkfit +5e-5, +3e-5).
    #[test]
    fn one_electron_systems_with_ri() {
        let df = || JkMethod::Df {
            aux: "cc-pvtz-jkfit".into(),
        };
        let h = Structure::from_xyz_str("1\n\nH 0.0 0.0 0.0\n", 0, 2).unwrap();
        let h2p = Structure::from_xyz_str("2\n\nH 0.0 0.0 0.0\nH 0.0 0.1 1.05\n", 1, 2).unwrap();
        let flat = |g: Vec<[f64; 3]>| g.into_iter().flatten().collect::<Vec<f64>>();
        for method in ["hf", "b3lyp", "wb97x"] {
            let calc = |mol: &Structure, jk: JkMethod, grid_response: bool| {
                let mut o = opts(method, jk, None);
                if method != "hf" {
                    o.grid.response = Some(grid_response);
                }
                Session::new(mol.clone(), o).unwrap()
            };
            for (name, mol) in [("H", &h), ("H2+", &h2p)] {
                let (mut ri, mut ex) = (calc(mol, df(), true), calc(mol, JkMethod::Exact, true));
                let (e_ri, e_ex) = (ri.energy().unwrap().energy, ex.energy().unwrap().energy);
                assert!(
                    (e_ri - e_ex).abs() < 2e-4,
                    "{method} {name}: E_RI {e_ri} E_exact {e_ex}"
                );
                let (g_ri, g_ex) = (
                    flat(ri.gradient().unwrap().gradient),
                    flat(ex.gradient().unwrap().gradient),
                );
                let e = max_diff(&g_ri, &g_ex);
                assert!(e < 2e-4, "{method} {name}: |g_RI - g_exact| = {e:e}");
                let (h_ri, h_ex) = (ri.hessian().unwrap().hessian, ex.hessian().unwrap().hessian);
                let e = max_diff(&h_ri, &h_ex);
                assert!(e < 1e-3, "{method} {name}: |H_RI - H_exact| = {e:e}");
                if mol.natom() == 1 {
                    let m = g_ri.iter().chain(&h_ri).fold(0.0f64, |m, x| m.max(x.abs()));
                    assert!(
                        m < 1e-6,
                        "{method}: H atom RI gradient {g_ri:?} Hessian {h_ri:?}"
                    );
                    let (a_ri, a_ex) = (
                        ri.polarizability().unwrap().alpha,
                        ex.polarizability().unwrap().alpha,
                    );
                    assert!(
                        a_ri[0][0] > 0.5 && (a_ri[0][0] - a_ex[0][0]).abs() < 1e-2 * a_ex[0][0],
                        "{method}: H atom alpha RI {a_ri:?} exact {a_ex:?}"
                    );
                    continue;
                }
                // the Hessian's first-order grid terms against gradients without grid response
                // on a moving grid (as in open_shell_matches_finite_differences)
                let mut c = calc(mol, df(), true);
                let an = c.hessian().unwrap().hessian;
                if method != "hf" {
                    c.cx.plan.request_mut().grid.response = Some(false);
                }
                let (n3, step) = (3 * mol.natom(), 2e-3);
                let mut fd = vec![0.0; n3 * n3];
                for y in 0..n3 {
                    let mut g = Vec::new();
                    for s in [1.0, -1.0, 2.0, -2.0] {
                        let mut xyz = mol.positions.clone();
                        xyz[y / 3][y % 3] += s * step;
                        c.set_positions(&xyz).unwrap();
                        g.push(flat(c.gradient().unwrap().gradient));
                    }
                    for x in 0..n3 {
                        fd[x * n3 + y] =
                            (8.0 * (g[0][x] - g[1][x]) - (g[2][x] - g[3][x])) / (12.0 * step);
                    }
                }
                let fds: Vec<f64> = (0..n3 * n3)
                    .map(|i| 0.5 * (fd[i] + fd[(i % n3) * n3 + i / n3]))
                    .collect();
                let e = max_diff(&an, &fds);
                assert!(
                    e < 2e-6,
                    "{method} H2+: |H_RI - H_fd(RI gradients)| = {e:e}"
                );
            }
        }
    }

    /// Range-separated hybrids against central differences of analytic gradients (without grid
    /// response, the grid moving with the atoms; first-order grid terms in the Hessian):
    /// wB97X RKS with exact integrals, CAM-B3LYP UKS (triplet CH2) and wB97M-D3BJ RKS (meta-GGA,
    /// D3BJ) with RI (wK exchange).
    #[test]
    fn range_separated_matches_finite_differences() {
        let df = || JkMethod::Df {
            aux: "def2-universal-jkfit".into(),
        };
        for (xyz, mult, method, jk) in [
            (WATER, 1, "wb97x", JkMethod::Exact),
            (CH2, 3, "cam-b3lyp", df()),
            (WATER, 1, "wb97m-d3bj", df()),
        ] {
            let mol = Structure::from_xyz_str(xyz, 0, mult).unwrap();
            let mut o = opts(method, jk.clone(), None);
            o.grid.response = Some(true);
            let mut calc = Session::new(mol.clone(), o).unwrap();
            let an = calc.hessian().unwrap().hessian;
            // the Hessian's grid response against gradients without one (one setting now)
            calc.cx.plan.request_mut().grid.response = Some(false);
            let n3 = 9;
            let h = 2e-3;
            let mut fd = vec![0.0; n3 * n3];
            for y in 0..n3 {
                let mut g = Vec::new();
                for s in [1.0, -1.0, 2.0, -2.0] {
                    let mut c = mol.positions.clone();
                    c[y / 3][y % 3] += s * h;
                    calc.set_positions(&c).unwrap();
                    g.push(calc.gradient().unwrap().gradient);
                }
                for x in 0..n3 {
                    let d = |i: usize| g[i][x / 3][x % 3];
                    fd[x * n3 + y] = (8.0 * (d(0) - d(1)) - (d(2) - d(3))) / (12.0 * h);
                }
            }
            let fds: Vec<f64> = (0..n3 * n3)
                .map(|i| 0.5 * (fd[i] + fd[(i % n3) * n3 + i / n3]))
                .collect();
            let e = max_diff(&an, &fds);
            println!("{method} {jk:?} mult {mult}: |H - H_fd| = {e:.2e}");
            assert!(e < 2e-6, "{method} {jk:?}: |H - H_fd| = {e:e}");
        }
    }
}
