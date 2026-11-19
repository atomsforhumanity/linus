//! Analytic Hessians on the device: the explicit second-derivative terms that linus's CPU
//! computes after the SCF and before the CPHF/CPKS (the response itself runs on
//! [`crate::response`]'s engines).
//!
//! - [`df`]: the density-fitted two-electron terms (`GpuDfHess`, linus's CPU
//!   `two_electron::df::hessian::df_hessian_2e_op`): the derivative Fock matrices
//!   F^x C = (J^x\[D_t\] - a_x K^x\[D_s\]) C_s of the CPHF right-hand sides for every
//!   perturbation, and the cross terms of the RI energy's second derivative (the Coulomb
//!   r^x V^-1 r^y and the exchange R^x V^-1 R^y).
//! - [`xc`]: the explicit XC terms (`GpuXcHess`, linus's CPU `XcIntegrator::hessian_terms` and
//!   `grid_terms`): the XC potential's first derivatives V^x C (fixed grid) and dV^y C (grid
//!   response), the XC energy's second derivative and the derivative of the fixed-grid XC
//!   gradient.

pub mod df;
pub mod xc;
