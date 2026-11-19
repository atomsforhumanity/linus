//! The method families' explicit derivative terms, each behind one shape, and the fixed order in
//! which a Hessian adds them.
//!
//! The families are the two-electron method (J/K: exact or density-fitted integrals), the
//! implicit solvent and the XC engine with VV10. Each family's terms come from the backend the
//! plan puts the family on (`crate::backend`): the device's provider, else the CPU's.
//!
//! **Gradient.** Each family's part is its own result (`GradientParts` reports them), a
//! [`GradientTerm`] of the XC engine and of the two-electron method (the solvent's part is
//! `CalcContext::solvent_gradient`). The gradient sums the parts in a fixed order: nuclear
//! repulsion, one-electron, two-electron, XC, dispersion, then the solvent. A term says where it
//! runs ([`Lane`]): the gradient's schedule runs the device's terms beside the CPU's
//! one-electron pass (each part computed as on its own: the same bits), the CPU's after it.
//!
//! **Hessian.** The families' terms are [`TwoElectronTerm`], [`SolventTerm`] and [`XcTerm`],
//! each a [`HessianTerm`]: it adds its explicit second derivative at fixed
//! density to h, its fixed-density Fock derivative F_s^x C_s to F^x C_o, and, for a grid moving
//! with the atoms, the first-order grid terms dV_s^y C_s ([`ExplicitHessian`]). The Hessian
//! adds the core Hamiltonian's terms (nuclear repulsion, one-electron, ECPs) first, then the
//! families in this order:
//! 1. the two-electron method;
//! 2. the solvent;
//! 3. the XC engine: the semilocal terms, then VV10's (`linus_dft::engine`);
//!
//! and last the grid terms join F^x ([`ExplicitHessian::finish`]). The results' bits depend
//! on this order, so it is written once, here and in the Hessian's assembly.
//!
//! A Hessian term keeps what its family hands on to the CPHF/CPKS operator: the two-electron
//! method its response J/K (set up right after its explicit terms, so the device sets it up at
//! the same point as before the solvent's and the XC's terms), the solvent its reaction field's
//! [`EnvironmentResponse`], the XC engine VV10's response ground state.

use crate::backend::Setup;
use crate::driver::backend::Family;
use crate::driver::ground::GroundState;
use crate::driver::CalcContext;
use crate::two_electron::{ResponseRequest, ResponseSetup, ResponseUse};
use faer::{Mat, MatRef, Par};
use linus_base::error::LinusError;
use linus_base::terms::EnvironmentResponse;
use linus_core::channels::Channels;
use linus_core::timing::Timers;
use linus_dft::engine::{HessianInput, HessianOut};
use linus_dft::vv10_resp::Vv10Ground;
use linus_dft::XcIntegrator;
use linus_ints::operator::Operator;
use linus_ints::PairSet;
use linus_scf::reference::Reference;
use std::sync::Arc;

/// The explicit terms of a Hessian as the terms add them: the explicit second derivative `h`
/// (3N x 3N row-major), the fixed-density F_s^x C_s `fxo[s][x]` (n x nocc_s), and the
/// first-order grid terms dV_s^y C_s `gxo[s][y]` (empty without a moving grid or XC).
pub(crate) struct ExplicitHessian {
    pub h: Vec<f64>,
    pub fxo: Vec<Vec<Mat<f64>>>,
    pub gxo: Vec<Vec<Mat<f64>>>,
}

impl ExplicitHessian {
    /// The terms so far: the core Hamiltonian's `h` and `fxo`.
    pub fn new(h: Vec<f64>, fxo: Vec<Vec<Mat<f64>>>) -> Self {
        ExplicitHessian {
            h,
            fxo,
            gxo: Vec::new(),
        }
    }

    /// The grid terms join F^x C_o (the response's right-hand sides and e1 take the moving
    /// grid's F^x): `(h, fxo + gxo, gxo)`, `gxo` kept for the assembly's fixed-grid correction.
    #[allow(clippy::type_complexity)]
    pub fn finish(mut self) -> (Vec<f64>, Vec<Vec<Mat<f64>>>, Vec<Vec<Mat<f64>>>) {
        for (fs, gs) in self.fxo.iter_mut().zip(&self.gxo) {
            for (f, g) in fs.iter_mut().zip(gs) {
                *f += g;
            }
        }
        (self.h, self.fxo, self.gxo)
    }
}

/// What every term reads: the ground state, its reference, its occupied orbitals per channel,
/// its total and channel densities row-major (the integral routines' layout), and the products'
/// parallelism.
pub(crate) struct TermInputs<'a> {
    pub ground: &'a GroundState,
    pub reference: &'a Reference,
    pub c_occ: Vec<MatRef<'a, f64>>,
    pub dtf: &'a [f64],
    pub dsf: Vec<&'a [f64]>,
    pub par: Par,
}

/// One method family's explicit Hessian terms (see the module docs).
pub(crate) trait HessianTerm {
    /// Adds the family's terms at the reference of `inputs` to `out`.
    fn add_hessian(
        &mut self,
        inputs: &TermInputs<'_>,
        out: &mut ExplicitHessian,
        timers: &mut Timers,
    ) -> Result<(), LinusError>;
}

/// The two-electron method's terms (family `hessian_jk`, `crate::backend`: exact four-center or the
/// density-fitted second derivatives, on the device when the backend puts them there) with the
/// exact exchange `xop`, then its response J/K ([`CalcContext::response_jk`]; `field`: an IR
/// run's electric-field response follows).
pub(crate) struct TwoElectronTerm<'a> {
    calc: &'a CalcContext,
    xop: Option<Operator>,
    pairs: &'a PairSet,
    field: bool,
    response: Option<ResponseSetup>,
}

impl<'a> TwoElectronTerm<'a> {
    pub fn new(
        calc: &'a CalcContext,
        xop: Option<Operator>,
        pairs: &'a PairSet,
        field: bool,
    ) -> Self {
        TwoElectronTerm {
            calc,
            xop,
            pairs,
            field,
            response: None,
        }
    }

    /// The response J/K set up after the explicit terms.
    pub fn into_response(self) -> ResponseSetup {
        self.response
            .expect("TwoElectronTerm: the response J/K comes with the explicit terms")
    }
}

impl HessianTerm for TwoElectronTerm<'_> {
    fn add_hessian(
        &mut self,
        inputs: &TermInputs<'_>,
        out: &mut ExplicitHessian,
        timers: &mut Timers,
    ) -> Result<(), LinusError> {
        let (calc, par) = (self.calc, inputs.par);
        let (xop, pairs) = (self.xop, self.pairs);
        let mut terms = calc.resolve(Family::HessianJk, |b| {
            b.hessian_jk(calc, inputs, xop, pairs, timers)
        })?;
        terms.add_hessian(inputs, out, timers)?;
        // a device engine's memory goes before the response J/K is set up
        drop(terms);
        // the response J/K (the right-hand sides and the final dF[U] C_o; a two-level response's
        // plan of the iterations' cheaper J/K) of these orbitals and exchange
        let req = ResponseRequest {
            ground: inputs.ground,
            occ: &inputs.c_occ,
            xop: self.xop,
            purpose: ResponseUse::Hessian,
        };
        self.response = Some(calc.response_jk(&req, self.field, self.pairs, par, timers)?);
        Ok(())
    }
}

/// The implicit solvent's terms (`crate::solvent::hessian_terms`; nothing in the gas phase),
/// keeping the reaction field's response for the response operator.
pub(crate) struct SolventTerm<'a> {
    calc: &'a CalcContext,
    response: Option<Arc<dyn EnvironmentResponse>>,
}

impl<'a> SolventTerm<'a> {
    pub fn new(calc: &'a CalcContext) -> Self {
        SolventTerm {
            calc,
            response: None,
        }
    }

    /// The reaction field's response (None in the gas phase).
    pub fn into_response(self) -> Option<Arc<dyn EnvironmentResponse>> {
        self.response
    }
}

impl HessianTerm for SolventTerm<'_> {
    fn add_hessian(
        &mut self,
        inputs: &TermInputs<'_>,
        out: &mut ExplicitHessian,
        timers: &mut Timers,
    ) -> Result<(), LinusError> {
        self.response = crate::solvent::hessian_terms(
            self.calc,
            inputs.ground,
            inputs.dtf,
            &inputs.c_occ,
            &mut out.h,
            &mut out.fxo,
            inputs.par,
            timers,
        )?;
        Ok(())
    }
}

/// The XC engine's terms on the integrator `xc` of the SCF grid (None for Hartree-Fock): the
/// semilocal terms, the device's when the backend puts them there (`crate::backend::gpu::xc`, both
/// parts in one pass), else the integrator's, then VV10's (-V functionals) at its response
/// ground state on the NLC grid (`linus_dft::engine`), which the CPKS kernel shares; with
/// `grid_response`, the first-order grid terms.
pub(crate) struct XcTerm<'a> {
    calc: &'a CalcContext,
    xc: Option<&'a XcIntegrator>,
    grid_response: bool,
    ground: Option<Arc<Vv10Ground>>,
}

impl<'a> XcTerm<'a> {
    pub fn new(calc: &'a CalcContext, xc: Option<&'a XcIntegrator>, grid_response: bool) -> Self {
        XcTerm {
            calc,
            xc,
            grid_response,
            ground: None,
        }
    }

    /// VV10's response ground state (-V functionals; None otherwise).
    pub fn into_vv10_ground(self) -> Option<Arc<Vv10Ground>> {
        self.ground
    }
}

impl HessianTerm for XcTerm<'_> {
    fn add_hessian(
        &mut self,
        inputs: &TermInputs<'_>,
        out: &mut ExplicitHessian,
        timers: &mut Timers,
    ) -> Result<(), LinusError> {
        let Some(x) = self.xc else {
            return Ok(());
        };
        let (calc, grid_response) = (self.calc, self.grid_response);
        let reference = inputs.reference;
        let ds: Vec<&Mat<f64>> = reference.channels.iter().map(|b| &b.density).collect();
        let dt = &reference.total_density;
        let input = HessianInput {
            ds: Channels::from_vec(ds),
            co: Channels::from_vec(inputs.c_occ.clone()),
            dt,
            grid_response,
            par: inputs.par,
        };
        // the semilocal terms on the backend the plan puts them on (a device engine's memory
        // goes before VV10's sweeps are set up)
        let semilocal = {
            let mut e =
                calc.resolve(Family::HessianXc, |b| b.hessian_xc(calc, x, &input, timers))?;
            e.terms(&input, timers)?
        };
        let mut o = HessianOut {
            h: &mut out.h,
            fxo: &mut out.fxo,
            gxo: &mut out.gxo,
        };
        let ground = |nl: &linus_dft::vv10::Vv10, t: &mut Timers| {
            calc.vv10_ground(nl, dt, grid_response, "hess", t)
        };
        self.ground = x.add_hessian_terms(semilocal, ground, &input, &mut o, timers)?;
        Ok(())
    }
}

/// What a gradient's family terms read: the ground state (its SCF's spin and occupations, which
/// the two-electron dispatch takes, and what it was set up with), the spins' occupied orbitals
/// and densities from the reference (a closed shell's one channel for both), the total density
/// row-major, and the products' parallelism.
pub(crate) struct GradientInputs<'a> {
    pub ground: &'a GroundState,
    pub ca: &'a Mat<f64>,
    pub cb: &'a Mat<f64>,
    pub da: &'a Mat<f64>,
    pub db: &'a Mat<f64>,
    pub dtf: &'a [f64],
    pub par: Par,
}

/// Where a gradient term runs in the gradient's schedule (`crate::derivatives::gradient`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Lane {
    /// Beside the host's one-electron pass: a device's term, whose work is the device's.
    Device,
    /// On the session's pool, after the one-electron pass.
    Host,
}

/// One method family's part of a nuclear gradient as a backend set it up (families
/// `gradient_xc` and `gradient_jk`, `crate::backend`). A device's term sets its engine up where
/// the gradient's schedule puts it: its host work before the overlap ([`GradientTerm::prepare`],
/// on the whole pool), its device work in it ([`GradientTerm::gradient`], beside the host's
/// one-electron pass). Either may decline (the setup or a call does not fit the device): nothing
/// of the term is then used, and the CPU's term runs after the overlap. A CPU term never
/// declines.
pub(crate) trait GradientTerm: Send {
    /// Where the term runs.
    fn lane(&self) -> Lane;

    /// The setup's host work, before the overlap (default nothing).
    fn prepare(
        &mut self,
        _inputs: &GradientInputs<'_>,
        _timers: &mut Timers,
    ) -> Result<Setup<()>, LinusError> {
        Ok(Setup::Ready(()))
    }

    /// The family's part (natom x 3).
    fn gradient(
        &mut self,
        inputs: &GradientInputs<'_>,
        timers: &mut Timers,
    ) -> Result<Setup<Vec<[f64; 3]>>, LinusError>;
}
