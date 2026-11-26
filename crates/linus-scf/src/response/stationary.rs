//! The stationary (Hylleraas) form of second-order response properties, written once.
//!
//! A second-order property E^{ab} of two perturbations (a Hessian element, a dipole derivative,
//! a polarizability) takes the first-order responses U_s^a, U_s^b (nvir_s x nocc_s per spin
//! channel s) from the CPHF/CPKS (A + B) U^k = b^k of [`super::problem::ResponseProblem`]. The
//! solver stops at a residual r^k = b^k - (A + B) U^k: written with one response only, the
//! property's error is linear in it. The Hylleraas functional of the pair,
//!
//!   E^{ab} = E^{ab}_fixed + f sum_s [U_s^b . b_s^a + U_s^a . r_s^b],
//!
//! is stationary in both responses: its error is -f r^a . (A + B)^-1 r^b, quadratic in the
//! residuals (f: the property's factor and the channels' occupation). The second term, the
//! **correction** f sum_s U_s^a . r_s^b ([`Correction`]), costs nothing but the residuals, which
//! come free from the solve ([`super::problem::ResponseProblem::residuals`]). With it a CPHF
//! tolerance of 1e-6 gives Hessians to ~1e-11 Eh/bohr^2.
//!
//! Every property adds it the same way, as one GEMM of the stacked (vectorized) matrices per
//! channel ([`Correction::add_gram`], the reduction of [`super::second_order`]):
//! - **the Hessian** (a, b both nuclear displacements x, y): -2 f_occ sum_s U_s^x . r_s^y over all
//!   pairs, among the Hessian's other response terms;
//! - **the dipole derivatives** (a the electric field's component c, b the nuclear displacement
//!   x): f sum_s U_s^c . r_s^x with U^c the polarizability's field response. Without it the
//!   error is linear in the residuals (-f r^x . (A + B)^-1 b^c); with it quadratic
//!   (-f r^x . (A + B)^-1 r^c, plus the residual times the difference between the two solves'
//!   operators). L-lactide B3LYP/def2-SVP at a CPHF tolerance of 1e-6: within 5e-11 au of the
//!   converged values instead of 1.2e-6;
//! - **the polarizability** (a, b both field components): f sum_s [U_s^c . b_s^d + U_s^c . r_s^d].
//!   Its three components are solved in one shared subspace, a Galerkin method (U = V c with
//!   V^T (b - (A + B) V c) = 0 for every right-hand side), so the correction vanishes to
//!   round-off there and α is the Ritz value b^c . V (V^T (A + B) V)^-1 V^T b^d. Measured
//!   (B3LYP/def2-SVP, max-abs residual 1e-5 / 1e-6): water 8e-11 / 1e-13 au, L-lactide
//!   1.1e-8 / 2.6e-11 au, its cation (UKS) 1.9e-8 / 1.3e-10 au; the correction moves α by about
//!   1e-15 relative. It is added all the same, so α stays stationary with any solver (a
//!   two-level or independent-vector solve is not Galerkin in the field's own subspace).
//!
//! Every response is `[s][k]` (channel s, component k), the layout of
//! [`super::problem::ResponseProblem::solve`].

use super::second_order::{refs, stacked_gram_on, Gram};
use faer::{Mat, Par};
use linus_base::error::LinusError;

/// The stationary correction f sum_s U_s^a . r_s^b of a second-order property (see the module
/// docs): `u[s][a]` the responses of one set of perturbations, `residuals[s][b]` the final
/// residuals of the other's solve (the same set for a Hessian).
pub struct Correction<'a> {
    pub u: &'a [Vec<Mat<f64>>],
    pub residuals: &'a [Vec<Mat<f64>>],
}

impl Correction<'_> {
    /// Adds f U_s^a . r_s^b of channel `s` to `out[(a, b)]` for all pairs, as one GEMM of the
    /// stacked matrices, by `gram` where it takes it (a device's; None: the host's,
    /// [`super::second_order::stacked_gram_on`]).
    pub fn add_gram(
        &self,
        gram: Option<&dyn Gram>,
        out: &mut Mat<f64>,
        s: usize,
        f: f64,
        par: Par,
    ) -> Result<(), LinusError> {
        stacked_gram_on(
            gram,
            out,
            f,
            &refs(&self.u[s]),
            &refs(&self.residuals[s]),
            par,
        )
    }
}
