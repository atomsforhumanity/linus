//! The XC term of a calculation at every derivative order, its VV10 nonlocal correlation
//! included: the one place that composes the semilocal part with VV10.
//!
//! - Energy and potential, gradient: [`XcIntegrator::potential`] and [`XcIntegrator::gradient`]
//!   add VV10 (on its NLC grid) themselves.
//! - CPKS kernel: [`XcKernel`], the semilocal kernel (this CPU integrator's, or a device kernel)
//!   and VV10's ([`crate::vv10_resp::Vv10Kernel`]), summed per product.
//! - Hessian terms: [`XcIntegrator::add_hessian_terms`], the semilocal terms ([`XcHessianEngine`]:
//!   this integrator's, or a device's) and VV10's at its response ground state, added to the
//!   Hessian's explicit terms.
//!
//! Placement (CPU or device, and VV10's device pair sweeps) is the backend's (`linus::backend`,
//! families `response_xc`, `hessian_xc` and `vv10`): the driver hands the engine the semilocal
//! kernel or terms and VV10's ground state. The engine fixes the order in which the parts are
//! added, which the results' bits depend on: the semilocal and VV10 kernel products are summed
//! before they join the response's Fock products, and the Hessian adds the semilocal terms, then
//! VV10's.

use crate::hessian::XcHessianTerms;
use crate::vv10::Vv10;
use crate::vv10_resp::{Vv10Ground, Vv10Kernel};
use crate::XcIntegrator;
use faer::{Mat, MatRef, Par};
use linus_base::error::LinusError;
use linus_base::terms::ResponseKernel;
use linus_core::channels::Channels;
use linus_core::timing::Timers;
use std::sync::Arc;

/// The XC kernel of a CPKS response: the semilocal kernel (the CPU integrator's or a device's,
/// as the backend set it up) and VV10's (-V functionals).
pub struct XcKernel<'a> {
    semilocal: Box<dyn ResponseKernel + 'a>,
    vv10: Option<Vv10Kernel<'a>>,
}

impl<'a> XcKernel<'a> {
    pub fn new(semilocal: Box<dyn ResponseKernel + 'a>, vv10: Option<Vv10Kernel<'a>>) -> Self {
        XcKernel { semilocal, vv10 }
    }

    pub fn kernel(&self) -> &dyn ResponseKernel {
        self
    }

    /// Records VV10's products as `<prefix>_vv10` (wall seconds) with `_pairs` and `_grid`
    /// (the pair sweeps, the NLC grid passes) and the resource `_perturbations` (nothing
    /// without VV10).
    pub fn record_vv10(&self, prefix: &str, timers: &mut Timers) {
        if let Some(k) = &self.vv10 {
            let (s, t) = k.seconds();
            timers.add(&format!("{prefix}_vv10"), s);
            timers.add(&format!("{prefix}_vv10_pairs"), t.pairs);
            timers.add(&format!("{prefix}_vv10_grid"), t.grid);
            timers.resource(
                &format!("{prefix}_vv10_perturbations"),
                t.perturbations as f64,
            );
        }
    }

    /// A device semilocal kernel's call times by phase (`xc_dev_*`; None on the CPU).
    pub fn device_timers(&self) -> Option<Timers> {
        self.semilocal.timers()
    }
}

impl ResponseKernel for XcKernel<'_> {
    fn contract_occ(
        &self,
        c_o: &[MatRef<'_, f64>],
        a: &[Vec<Mat<f64>>],
    ) -> Result<Vec<Vec<Mat<f64>>>, LinusError> {
        let mut v = self.semilocal.contract_occ(c_o, a)?;
        if let Some(k) = &self.vv10 {
            for (vs, ws) in v.iter_mut().zip(k.contract_occ(c_o, a)?) {
                for (x, y) in vs.iter_mut().zip(ws) {
                    *x += y;
                }
            }
        }
        Ok(v)
    }
}

/// What the XC terms of a Hessian are computed at: the channel densities `ds` (one: a closed
/// shell's alpha density; two: the spins) with their occupied orbitals `co`, the total density
/// `dt`, whether the grid moves with the atoms, and the products' parallelism.
pub struct HessianInput<'a> {
    pub ds: Channels<&'a Mat<f64>>,
    pub co: Channels<MatRef<'a, f64>>,
    pub dt: &'a Mat<f64>,
    pub grid_response: bool,
    pub par: Par,
}

/// The semilocal explicit XC terms of a Hessian as a backend computes them (family
/// `hessian_xc`): the CPU's integrator ([`XcIntegrator::hessian_terms`]) or a device's engine,
/// set up for the shapes of the input it is then given.
pub trait XcHessianEngine {
    /// V_s^x C_s, the grid terms and the explicit second derivative at `input`.
    fn terms(
        &mut self,
        input: &HessianInput<'_>,
        timers: &mut Timers,
    ) -> Result<XcHessianTerms, LinusError>;
}

/// What the XC terms of a Hessian add to: its explicit second derivative, its fixed-grid
/// F_s^x C_s and the grid response's dF_s^y C_s.
pub struct HessianOut<'a> {
    pub h: &'a mut [f64],
    pub fxo: &'a mut [Vec<Mat<f64>>],
    pub gxo: &'a mut Vec<Vec<Mat<f64>>>,
}

impl XcIntegrator {
    /// The XC terms of a Hessian added to `out`: the semilocal terms `semilocal` (of this
    /// integrator or a device's, [`XcHessianEngine`]), then, for a -V functional, VV10's at its
    /// response ground state (built by `vv10_ground`, where the backend places the pair sweeps;
    /// timed as `hess_vv10` with its parts). Returns that ground state for the CPKS kernel.
    pub fn add_hessian_terms(
        &self,
        semilocal: XcHessianTerms,
        vv10_ground: impl FnOnce(&Vv10, &mut Timers) -> Result<Vv10Ground, LinusError>,
        input: &HessianInput<'_>,
        out: &mut HessianOut<'_>,
        timers: &mut Timers,
    ) -> Result<Option<Arc<Vv10Ground>>, LinusError> {
        semilocal.add_to(out.h, out.fxo, out.gxo);
        let Some(nl) = self.vv10() else {
            return Ok(None);
        };
        let g = vv10_ground(nl, timers)?;
        let t = timers.time("hess_vv10", || {
            nl.hessian_terms(
                &g,
                input.co.as_slice(),
                input.dt,
                input.grid_response,
                input.par,
            )
        })?;
        timers.add("hess_vv10_potential_terms", t.times.potential_terms);
        timers.add("hess_vv10_pairs", t.times.pairs);
        timers.add("hess_vv10_grid", t.times.grid);
        t.terms.add_to(out.h, out.fxo, out.gxo);
        Ok(Some(Arc::new(g)))
    }
}

/// The integrator's semilocal Hessian terms ([`XcIntegrator::hessian_terms`]).
impl XcHessianEngine for &XcIntegrator {
    fn terms(
        &mut self,
        input: &HessianInput<'_>,
        timers: &mut Timers,
    ) -> Result<XcHessianTerms, LinusError> {
        Ok(self.hessian_terms(input.ds, input.co, input.grid_response, input.par, timers))
    }
}
