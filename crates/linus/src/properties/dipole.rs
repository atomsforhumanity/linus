//! The electric dipole moment ([`Session::dipole_moment`]) and the dipole integrals behind it,
//! the electric field as a perturbation of the reference ([`ElectricField`]: the
//! polarizability's and the dipole derivatives'), and the dipole derivatives (atomic polar
//! tensors) for IR intensities, closed and open shells (HF and KS; exact four-center integrals
//! or density fitting). The static polarizability that IR results carry is in
//! [`super::polarizability`].
//!
//! The dipole derivatives come from the analytic Hessian's own CPHF/CPKS solutions
//! ([`crate::derivatives::hessian::HessianOptions::dipole_derivatives`]) in a form stationary in
//! them (decision D35):
//!
//!   dμ_c/dX = Z δ_ck - Tr(D_t r_c^x)
//!             + f sum_s [U_s^x . b_s^c + 1/2 s1oo_s^x . (C_os^T r_c C_os) + U_s^c . r_s^x],
//!
//! f = 4 closed shell (alpha orbitals), 2 per spin open shell, with r_c the dipole integrals,
//! r_c^x their derivatives, b_s^c = -C_vs^T r_c C_os the electric field's right-hand side, U_s^x
//! and s1oo_s^x the nuclear displacements' response and occupied overlap derivative, U_s^c the
//! field's response and r_s^x the final residuals of the nuclear perturbations. The response
//! part is minus the generic second-order assembly with the field as rows and the nuclear
//! displacements as columns, -f sum_s (r_c C_os) . C1_s^x with C1^x = C_v U^x - 1/2 C_o s1oo^x
//! ([`linus_scf::response::second_order::SecondOrder`]); the last term is the stationary correction
//! ([`linus_scf::response::stationary`]): it makes the error quadratic in the residuals instead of
//! linear. U_s^c is the polarizability's electric-field CPHF/CPKS (the
//! SCF grid's XC kernel; the Hessian's response J/K when it serves these products too, as the
//! device's FP64 one does with its stored exchange operator, else one of its own), which IR
//! results carry anyway: it runs after the Hessian is complete (the Hessian does not change).
//! Besides it, the dipole derivatives cost the dipole integrals and their derivatives. KS: the
//! response is the Hessian's, so with the default first-order grid terms
//! (`grid.response`) it equals differences of SCF dipole moments on a grid that moves
//! with the atoms.

use crate::derivatives::hessian::{HessianOptions, HessianResult};
use crate::driver::ground::GroundState;
use crate::driver::{CalcContext, Session};
use crate::properties::polarizability::Polarizability;
use faer::{Mat, Par};
use linus_base::error::LinusError;
use linus_base::linalg::{ao_view_mut, mm};
use linus_base::terms::ResponseJk;
use linus_core::timing::Timers;
use linus_dft::XcIntegrator;
use linus_ints::{BasisSet, DerivOrder, OneEPlan};
use linus_scf::reference::Reference;
use linus_scf::response::perturbation::{NuclearDisplacements, Perturbation};
use linus_scf::response::second_order::{FirstOrderOrbitals, SecondOrder};
use linus_scf::response::stationary::Correction;
use std::cell::OnceCell;

#[derive(Clone, Debug)]
pub struct DipoleDerivatives {
    /// dμ_c/dX: 3N rows (3 * atom + k), columns c = x, y, z (e, atomic units).
    pub apt: Vec<[f64; 3]>,
    /// Static dipole polarizability (a.u., e^2 bohr^2 / Eh), symmetric.
    pub polarizability: [[f64; 3]; 3],
    /// Electric dipole moment (e bohr) about the origin.
    pub dipole: [f64; 3],
    /// Iterations of the electric-field CPHF (the polarizability's).
    pub cphf_iterations: usize,
    /// Max-abs residual of the last electric-field CPHF iteration (the polarizability's).
    pub max_residual: f64,
    pub timers: Timers,
}

/// The dipole integrals <a| r_c |b> about the coordinate origin, c = x, y, z: the dipole
/// moment's, the electric field's and the dipole derivatives' (one kernel for all).
pub(crate) fn dipole_integrals(basis: &BasisSet) -> [Mat<f64>; 3] {
    let n = basis.nbf;
    let mut r = [Mat::zeros(n, n), Mat::zeros(n, n), Mat::zeros(n, n)];
    let [x, y, z] = &mut r;
    let mut views = [x, y, z].map(|m| ao_view_mut(m.as_mut()));
    OneEPlan::new(basis, DerivOrder::Zero).dipole([0.0; 3], &mut views);
    r
}

impl CalcContext {
    /// Electric dipole moment (e bohr) about the coordinate origin, of the nuclei and the
    /// electron density D_a + D_b.
    pub fn dipole_moment(&self, d_a: &Mat<f64>, d_b: &Mat<f64>) -> [f64; 3] {
        let n = self.basis.nbf;
        let ints = self.pool.install(|| dipole_integrals(&self.basis));
        let mut mu = [0.0; 3];
        for (k, m) in ints.iter().enumerate() {
            let mut el = 0.0;
            for i in 0..n {
                for j in 0..n {
                    el += (d_a[(i, j)] + d_b[(i, j)]) * m[(i, j)];
                }
            }
            let nuc: f64 = self.mol.point_charges().iter().map(|(q, r)| q * r[k]).sum();
            mu[k] = nuc - el;
        }
        mu
    }
}

/// The electric field as a perturbation of a reference ([`linus_scf::response::perturbation`]): a
/// field F adds F . r to the core Hamiltonian and leaves the basis and the grid in place, so
/// F^c C_o = r_c C_o (the dipole integrals) is its whole fixed-density Fock derivative.
pub(crate) struct ElectricField<'a> {
    reference: &'a Reference,
    r: [Mat<f64>; 3],
    par: Par,
    fock_occ: OnceCell<Vec<Vec<Mat<f64>>>>,
}

impl<'a> ElectricField<'a> {
    /// The field's perturbation of `reference` (F^c C_o made on first use, at parallelism
    /// `par`: the dipole derivatives take it, the polarizability does not).
    pub fn new(reference: &'a Reference, basis: &BasisSet, par: Par) -> Self {
        ElectricField {
            reference,
            r: dipole_integrals(basis),
            par,
            fock_occ: OnceCell::new(),
        }
    }
}

impl Perturbation for ElectricField<'_> {
    fn reference(&self) -> &Reference {
        self.reference
    }

    fn components(&self) -> usize {
        3
    }

    fn fock_occ(&self) -> &[Vec<Mat<f64>>] {
        self.fock_occ.get_or_init(|| {
            self.reference
                .channels
                .iter()
                .map(|b| {
                    self.r
                        .iter()
                        .map(|rc| mm(rc.as_ref(), b.c_occ.as_ref(), self.par))
                        .collect()
                })
                .collect()
        })
    }
}

/// What the dipole derivatives take from an analytic Hessian (module docs): its nuclear
/// displacements (and the reference), D_t row-major, the first-order occupied orbitals C1_s^x
/// of their responses and the final residuals r_s^x (`[s][x]`), and the Hessian's response J/K
/// when it serves the electric-field CPHF too ([`ResponseJk::serve`]: the device's in double
/// precision, with its stored exchange operator), kept alive past the Hessian for it.
pub(crate) struct HessianResponse<'a> {
    pub nuclear: &'a NuclearDisplacements<'a>,
    pub dtf: &'a [f64],
    pub orbitals: &'a FirstOrderOrbitals,
    pub residuals: Vec<Vec<Mat<f64>>>,
    pub jk: Option<Box<dyn ResponseJk>>,
}

/// The dipole derivatives of a Hessian before their stationary correction, with what the
/// correction needs: the nuclear perturbations' final residuals (`[s][x]`), the channels'
/// factor f, and the Hessian's response J/K for the field's CPHF ([`HessianResponse`]).
pub(crate) struct DipoleTerms {
    pub apt: Vec<[f64; 3]>,
    residuals: Vec<Vec<Mat<f64>>>,
    f: f64,
    jk: Option<Box<dyn ResponseJk>>,
}

/// Adds -m(c, x) to `apt[x][c]`: a field x nuclear-displacements matrix of the second-order
/// assembly (rows the field's components c, columns the displacements x) in the dipole
/// derivatives' layout.
fn subtract_transposed(apt: &mut [[f64; 3]], m: &Mat<f64>) {
    for (x, o) in apt.iter_mut().enumerate() {
        for (c, v) in o.iter_mut().enumerate() {
            *v -= m[(c, x)];
        }
    }
}

impl CalcContext {
    /// The dipole derivatives of a Hessian's response before their stationary correction
    /// (module docs): Z δ_ck - Tr(D_t r_c^x) plus the response part, minus the second-order
    /// assembly of the field (rows) and the nuclear displacements (columns).
    pub(crate) fn dipole_terms(&self, h: HessianResponse<'_>, par: Par) -> DipoleTerms {
        let HessianResponse {
            nuclear,
            dtf,
            orbitals,
            residuals,
            jk,
        } = h;
        let reference = nuclear.reference;
        let field = ElectricField::new(reference, &self.basis, par);
        let mut apt = self.dipole_explicit(dtf);
        let f = 2.0 * reference.occupation();
        let response = SecondOrder {
            rows: &field,
            columns: orbitals,
            correction: None,
        };
        let mut m = Mat::<f64>::zeros(3, apt.len());
        for s in 0..reference.nch() {
            response.add_channel(&mut m, s, reference.occupation(), par);
        }
        subtract_transposed(&mut apt, &m);
        DipoleTerms {
            apt,
            residuals,
            f,
            jk,
        }
    }

    /// The explicit part of the dipole derivatives, Z δ_ck - Tr(D_t r_c^x), for the total
    /// density `dt` (row-major).
    fn dipole_explicit(&self, dt: &[f64]) -> Vec<[f64; 3]> {
        let plan = OneEPlan::new(&self.basis, DerivOrder::One);
        let mut g = vec![[[0.0; 3]; 3]; self.basis.natom()];
        plan.dipole_grad([0.0; 3], dt, &mut g, &mut plan.workspace());
        let mut out = vec![[0.0; 3]; 3 * self.mol.natom()];
        for (a, ga) in g.iter().enumerate() {
            for k in 0..3 {
                for c in 0..3 {
                    out[3 * a + k][c] =
                        if k == c { self.mol.zeff(a) as f64 } else { 0.0 } - ga[k][c];
                }
            }
        }
        out
    }

    /// An IR run's electric-field CPHF/CPKS after its Hessian (the polarizability's, on the SCF
    /// grid's XC kernel `xc`; on the Hessian's response J/K when it serves these products), and
    /// with its responses the stationary dipole derivatives: `terms` plus the correction
    /// f sum_s U_s^c . r_s^x ([`Correction::add_gram`]). The polarizability carries the field
    /// response's timers, and its wall as `polarizability`.
    pub(crate) fn stationary_dipole_derivatives(
        &self,
        gs: &GroundState,
        reference: &Reference,
        xc: Option<&XcIntegrator>,
        terms: DipoleTerms,
        par: Par,
    ) -> Result<(Vec<[f64; 3]>, Polarizability), LinusError> {
        let DipoleTerms {
            mut apt,
            residuals,
            f,
            jk,
        } = terms;
        let t0 = std::time::Instant::now();
        let mut timers = Timers::default();
        let fr = self
            .pool
            .install(|| self.field_response(gs, reference, xc, jk, par, &mut timers))?;
        let correction = Correction {
            u: &fr.u,
            residuals: &residuals,
        };
        let mut m = Mat::<f64>::zeros(3, apt.len());
        for s in 0..residuals.len() {
            correction.add_gram(&mut m, s, -f, par);
        }
        subtract_transposed(&mut apt, &m);
        drop(residuals);
        timers.add("polarizability", t0.elapsed().as_secs_f64());
        Ok((apt, fr.polarizability(timers)))
    }
}

impl Session {
    /// Dipole derivatives, polarizability and dipole moment at the current geometry. Computes
    /// the analytic Hessian (with dipole derivatives) on the way; when the Hessian is wanted as
    /// well, use [`Session::hessian_with`] and [`Session::ir_properties`] instead (as
    /// [`Session::hessian_analysis`] does).
    pub fn dipole_derivatives(&mut self) -> Result<DipoleDerivatives, LinusError> {
        let h = self.hessian_with(&HessianOptions {
            dipole_derivatives: true,
        })?;
        self.ir_properties(&h)
    }

    /// IR properties of a Hessian computed with [`HessianOptions::dipole_derivatives`] at the
    /// current geometry: its dipole derivatives and polarizability (the electric-field CPHF its
    /// dipole derivatives used; an error, as from [`Session::polarizability`], if that missed
    /// `response.field_tol`), plus the dipole moment of the last SCF.
    pub fn ir_properties(&mut self, h: &HessianResult) -> Result<DipoleDerivatives, LinusError> {
        let apt = h.dipole_derivatives.clone().ok_or_else(|| {
            LinusError::Input("the Hessian was computed without dipole derivatives".into())
        })?;
        let p = match &h.polarizability {
            Some(p) => self.accept_polarizability(p.clone())?,
            None => self.polarizability()?,
        };
        let r = self.last_here().ok_or_else(|| {
            LinusError::Numerical("no SCF result at this geometry after the polarizability".into())
        })?;
        let dipole = self.cx.dipole_moment(&r.d_a, &r.d_b);
        Ok(DipoleDerivatives {
            apt,
            polarizability: p.alpha,
            dipole,
            cphf_iterations: p.cphf_iterations,
            max_residual: p.max_residual,
            timers: p.timers,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::dipole_integrals;
    use crate::driver::{CalcOptions, JkMethod, Session};
    use crate::Structure;
    use linus_ints::{AoMatrixMut, DerivOrder, OneEPlan};

    const WATER: &str = "3\n\nO 0.0 0.05 0.11\nH 0.0 0.76 -0.47\nH 0.1 -0.75 -0.46\n";
    const CH2: &str = "3\n\nC 0.000 0.003 0.102\nH 0.000 0.995 -0.310\nH 0.010 -0.990 -0.302\n";

    /// The dipole integrals are the same bits in a faer matrix as row-major from a plan of
    /// derivative order one (the dipole derivatives' former layout): one kernel, one value per
    /// element.
    #[test]
    fn dipole_integrals_are_layout_independent() {
        let mol = Structure::from_xyz_str(CH2, 0, 3).unwrap();
        let o = CalcOptions {
            basis: Some("def2-tzvp".into()),
            ..Default::default()
        };
        let calc = Session::new(mol, o).unwrap();
        let basis = &calc.cx.basis;
        let n = basis.nbf;
        let r = dipole_integrals(basis);
        let mut ints = [vec![0.0; n * n], vec![0.0; n * n], vec![0.0; n * n]];
        let [x, y, z] = &mut ints;
        let views = [x, y, z].map(|m| AoMatrixMut::row_major(m, n));
        OneEPlan::new(basis, DerivOrder::One).dipole([0.0; 3], &mut { views });
        for c in 0..3 {
            for i in 0..n {
                for j in 0..n {
                    assert_eq!(r[c][(i, j)].to_bits(), ints[c][i * n + j].to_bits());
                }
            }
        }
    }

    /// Dipole derivatives through `dipole_derivatives` (the Hessian's): acoustic sum rule
    /// sum_A dμ_c/dX_A,k = q δ_kc, closed and open shell (B3LYP RI, first-order grid terms).
    #[test]
    fn apt_sum_rule() {
        for (xyz, mult) in [(WATER, 1), (CH2, 3)] {
            let mol = Structure::from_xyz_str(xyz, 0, mult).unwrap();
            let o = CalcOptions {
                method: Some("b3lyp".into()),
                basis: Some("def2-svp".into()),
                jk: JkMethod::Df {
                    aux: "def2-universal-jkfit".into(),
                }
                .into(),
                threads: Some(4),
                scf: crate::driver::ScfSettings {
                    e_tol: Some(crate::driver::options::Auto::Value(1e-12)),
                    g_tol: Some(crate::driver::options::Auto::Value(1e-10)),
                    ..Default::default()
                },
                ..Default::default()
            };
            let d = Session::new(mol, o).unwrap().dipole_derivatives().unwrap();
            for c in 0..3 {
                for k in 0..3 {
                    let s: f64 = (0..3).map(|a| d.apt[3 * a + k][c]).sum();
                    assert!(s.abs() < 1e-5, "mult {mult}: sum rule k={k} c={c}: {s}");
                }
            }
        }
    }
}
