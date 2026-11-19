//! Reference integrals and contractions: plain versions (quartet by quartet, through derivative
//! pseudo pairs, or without the production kernels' screening and blocking) of what the
//! production code computes, kept for the tests that check it.
//!
//! The module is compiled for tests and with the `reference` feature only: the dev-dependencies
//! of linus-ints itself, linus and linus-gpu turn the feature on, and release builds leave it
//! out.
//!
//! - [`eri`]: the density-fitting metric and three-center integrals over plain quartets
//!   ([`eri_2c`], [`eri_3c`]); raw first-derivative blocks of a quartet
//!   ([`eri_quartet_deriv1`]) and the pseudo-pair references of first and second derivatives;
//! - [`eri3c`]: Gamma-contracted three-center gradients and Hessians through the generic
//!   quartet kernels ([`eri_grad_3c`], [`eri_hess_3c`]); the production ones are
//!   `eri3c::dgrad` and `eri3c::dblock`;
//! - [`df_deriv`]: three-center derivative contractions for all perturbations of an atom
//!   ([`df3c_deriv_atoms`]);
//! - [`oneshot`]: one-call forms of the plans (a plan and a workspace per call, the result in
//!   a new vector), for tests that compute a quantity once;
//! - [`pcm`]: the surface-charge potentials and potential matrices pair by pair in the
//!   McMurchie-Davidson form, unscreened (the reference of `points::pcm_ints`'s plans).

pub mod df_deriv;
pub mod eri;
pub mod eri3c;
pub mod oneshot;
pub mod pcm;

pub use df_deriv::{apply_metric_deriv, df3c_deriv_atoms, Df3cAtom, Df3cInput};
pub use eri::{eri_2c, eri_3c, eri_quartet_deriv1, eri_quartet_deriv1_ref, eri_quartet_deriv2_ref};
pub use eri3c::{eri_grad_3c, eri_grad_3c_with, eri_hess_3c, Dens3c, Grad3cScreen};
