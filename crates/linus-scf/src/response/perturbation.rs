//! Perturbations of the reference: what a static first-order perturbation hands the response
//! equations and the second-order properties.
//!
//! A real, static perturbation with components λ_k (the 3N nuclear displacements, the three
//! directions of an electric field) enters through two derivatives at fixed density, of which
//! the response and the properties only ever need the occupied projections (per spin channel
//! s, `[s][k]`):
//! - F^k C_os ([`Perturbation::fock_occ`]): the derivative of the Fock matrix at fixed density
//!   times the occupied orbitals. For a field the dipole integrals; for the nuclear
//!   displacements the core Hamiltonian's derivative and the method families' fixed-density
//!   terms (`linus::derivatives::terms`);
//! - S^k C_os ([`Perturbation::overlap`]), for a perturbation that moves the basis functions
//!   (none for a field).
//!
//! From them follow, the same for every perturbation:
//! - s1oo^k = C_os^T S^k C_os ([`Overlap`]): the orbitals stay orthonormal, so the
//!   occupied-occupied block of the response is fixed, U^k_oo = -1/2 s1oo^k, with the
//!   first-order density D^k_oo = -C_os s1oo^k C_os^T = A C_os^T + C_os A^T, A = -1/2 C_os s1oo^k;
//! - its induced Fock matrix dF[D^k_oo] C_os, one product of the response operator for all
//!   components ([`crate::response::problem::ResponseProblem::induced_oo`]);
//! - the right-hand sides of the response equations for the virtual-occupied block
//!   ([`Perturbation::rhs_vo`]),
//!   (A + B) U^k = -(C_vs^T F^k C_os - C_vs^T S^k C_os e_os + C_vs^T dF[D^k_oo] C_os);
//! - after the solve, the first-order occupied orbitals and orbital energies that second-order
//!   properties are assembled from ([`crate::response::second_order`]).
//!
//! The perturbations: [`NuclearDisplacements`] (the Hessian's and the dipole derivatives') and
//! the electric field (`linus::properties::dipole::ElectricField`: the polarizability's and the
//! dipole derivatives'). A new second-order property is a new perturbation, not a new copy of
//! this pattern: NMR shieldings pair a magnetic field (with London orbitals it moves the basis,
//! like a nuclear displacement) with the nuclear magnetic moments, magnetizabilities pair two
//! magnetic fields. Magnetic perturbations are imaginary: their response operator is A - B (no
//! Coulomb and no semilocal XC kernel), which such a perturbation would declare.

use crate::reference::Reference;
use faer::{Mat, Par};
use linus_base::linalg::{mat_from_rowmajor, mm};

/// A static perturbation of a reference determinant (see the module docs). Every matrix list
/// is `[s][k]`: spin channel s of the reference, component k.
pub trait Perturbation {
    /// The reference it perturbs.
    fn reference(&self) -> &Reference;

    /// Its components: the 3N nuclear displacements, the field's three directions.
    fn components(&self) -> usize;

    /// F^k C_os (nbf x nocc_s): the derivative of the Fock matrix at fixed density times the
    /// occupied orbitals.
    fn fock_occ(&self) -> &[Vec<Mat<f64>>];

    /// The overlap derivative's occupied projections, for a perturbation that moves the basis
    /// functions; None for one that leaves them (a field).
    fn overlap(&self) -> Option<&Overlap> {
        None
    }

    /// The right-hand sides of the response equations for U^k (nvir_s x nocc_s),
    /// -(C_vs^T F^k C_os - C_vs^T S^k C_os e_os + C_vs^T dF[D^k_oo] C_os), with `induced_oo` =
    /// dF[D^k_oo] C_os for a perturbation that moves the basis
    /// ([`crate::response::problem::ResponseProblem::induced_oo`]; None for one that does not).
    fn rhs_vo(&self, induced_oo: Option<&[Vec<Mat<f64>>]>, par: Par) -> Vec<Vec<Mat<f64>>> {
        let fxo = self.fock_occ();
        let moving = self.overlap().map(|o| {
            let voo = induced_oo.expect("the induced Fock matrix of a moving basis's D_oo");
            (&o.sxo, voo)
        });
        self.reference()
            .channels
            .iter()
            .enumerate()
            .map(|(s, b)| {
                let cvt = b.c_vir.transpose();
                (0..self.components())
                    .map(|k| {
                        let h1vo = mm(cvt, fxo[s][k].as_ref(), par);
                        match moving {
                            Some((sxo, voo)) => {
                                let s1vo = mm(cvt, sxo[s][k].as_ref(), par);
                                let vvo = mm(cvt, voo[s][k].as_ref(), par);
                                Mat::from_fn(b.c_vir.ncols(), b.nocc(), |a, i| {
                                    -(h1vo[(a, i)] - b.eps_occ[i] * s1vo[(a, i)] + vvo[(a, i)])
                                })
                            }
                            None => Mat::from_fn(h1vo.nrows(), h1vo.ncols(), |a, i| -h1vo[(a, i)]),
                        }
                    })
                    .collect()
            })
            .collect()
    }
}

/// The occupied projections of a perturbation's overlap derivative S^k: S^k C_os (`sxo`,
/// nbf x nocc_s) and s1oo^k = C_os^T S^k C_os (`s1oo`, nocc_s x nocc_s), `[s][k]`.
pub struct Overlap {
    pub sxo: Vec<Vec<Mat<f64>>>,
    pub s1oo: Vec<Vec<Mat<f64>>>,
}

/// The occupied projections X^k C_os (`[s][k]`, nbf x nocc_s) of AO matrices X^k given
/// row-major one after the other (component k at `ao[k n^2..(k + 1) n^2]`, the layout of the
/// one-electron derivative integrals).
pub fn occupied_projections(reference: &Reference, ao: &[f64], par: Par) -> Vec<Vec<Mat<f64>>> {
    let n = reference.channels.alpha().c_occ.nrows();
    reference
        .channels
        .iter()
        .map(|b| {
            ao.chunks_exact(n * n)
                .map(|m| mm(mat_from_rowmajor(n, n, m).as_ref(), b.c_occ.as_ref(), par))
                .collect()
        })
        .collect()
}

impl Overlap {
    /// The projections of the AO derivatives `sx` (row-major, [`occupied_projections`]) on the
    /// occupied orbitals of `reference`.
    pub fn new(reference: &Reference, sx: &[f64], par: Par) -> Self {
        let sxo = occupied_projections(reference, sx, par);
        let s1oo = reference
            .channels
            .iter()
            .zip(&sxo)
            .map(|(b, xs)| {
                xs.iter()
                    .map(|x| mm(b.c_occ.transpose(), x.as_ref(), par))
                    .collect()
            })
            .collect();
        Overlap { sxo, s1oo }
    }

    /// The factors A^k = -1/2 C_os s1oo^k of the occupied block's first-order density
    /// D^k_oo = A C_os^T + C_os A^T (the layout the response operator takes).
    pub fn occupied_factors(&self, reference: &Reference, par: Par) -> Vec<Vec<Mat<f64>>> {
        reference
            .channels
            .iter()
            .zip(&self.s1oo)
            .map(|(b, xs)| {
                xs.iter()
                    .map(|x| mm(b.c_occ.as_ref(), x.as_ref(), par) * -0.5)
                    .collect()
            })
            .collect()
    }
}

/// The 3N nuclear displacements x = 3 * atom + k: F^x C_os from the core Hamiltonian's
/// derivative and the method families' fixed-density terms (with the first-order grid terms of
/// a grid moving with the atoms: `linus::derivatives::terms::ExplicitHessian::finish`), and the
/// overlap derivative.
pub struct NuclearDisplacements<'a> {
    pub reference: &'a Reference,
    pub fock_occ: Vec<Vec<Mat<f64>>>,
    pub overlap: Overlap,
}

impl Perturbation for NuclearDisplacements<'_> {
    fn reference(&self) -> &Reference {
        self.reference
    }

    fn components(&self) -> usize {
        self.overlap.s1oo[0].len()
    }

    fn fock_occ(&self) -> &[Vec<Mat<f64>>] {
        &self.fock_occ
    }

    fn overlap(&self) -> Option<&Overlap> {
        Some(&self.overlap)
    }
}
