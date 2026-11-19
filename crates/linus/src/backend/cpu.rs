//! The CPU's provider: the CPU engines of every family, as the families' own modules choose them
//! (the J/K method's in `crate::two_electron`, the XC integrator's in `linus_dft`). It always
//! answers [`Setup::Ready`].

use super::scf::{ScfJk, ScfJkBuilt, ScfRun, ScfTerms, ScfXc};
use super::{Backend, Setup};
use crate::derivatives::terms::{
    ExplicitHessian, GradientInputs, GradientTerm, HessianTerm, Lane, TermInputs,
};
use crate::driver::ground::{GroundState, ScfSetup};
use crate::driver::CalcContext;
use crate::solvent::derivatives::{CpuSolventGradient, SolventGradient};
use crate::solvent::scf::{PcmSetup, ScfEnvironment};
use crate::solvent::SolventSettings;
use crate::two_electron::{KernelUse, ResponseRequest, ResponseSetup};
use faer::{Mat, MatRef, Par};
use linus_base::error::LinusError;
use linus_base::terms::JkBuilder;
use linus_base::terms::ResponseKernel;
use linus_base::terms::{XcDensity, XcPotential};
use linus_core::timing::Timers;
use linus_dft::engine::{HessianInput, XcHessianEngine};
use linus_dft::vv10::Vv10Stats;
use linus_dft::vv10_resp::{Vv10Ground, Vv10PairSweep};
use linus_dft::XcIntegrator;
use linus_ints::operator::Operator;
use linus_ints::PairSet;
use linus_scf::{HostLoop, ScfLoop};
use std::sync::Arc;

/// The CPU's provider (module docs).
pub(crate) struct CpuBackend;

impl Backend for CpuBackend {
    fn scf_jk(
        &self,
        cx: &CalcContext,
        pairs: &PairSet,
        setup: &ScfSetup,
        timers: &mut Timers,
    ) -> Result<Setup<ScfJkBuilt>, LinusError> {
        let (jk, df) = cx.cpu_scf_jk(pairs, setup, timers)?;
        Ok(Setup::Ready((Box::new(CpuScfJk(jk)), df)))
    }

    fn scf_xc(
        &self,
        _cx: &CalcContext,
        xc: Arc<XcIntegrator>,
        _timers: &mut Timers,
    ) -> Result<Setup<Box<dyn ScfXc>>, LinusError> {
        Ok(Setup::Ready(Box::new(CpuXc(xc))))
    }

    fn takes_scf_loop(
        &self,
        _cx: &CalcContext,
        _terms: &ScfTerms,
    ) -> Result<Setup<()>, LinusError> {
        Ok(Setup::Ready(()))
    }

    /// The host loop over the terms' J/K builder and XC term.
    fn scf_loop<'a>(
        &self,
        _cx: &'a CalcContext,
        terms: &'a mut ScfTerms,
        run: ScfRun<'a>,
    ) -> Box<dyn ScfLoop + 'a> {
        Box::new(HostLoop {
            inp: run.inp,
            jk: terms.jk.builder(),
            xc: terms.xc.as_deref_mut().map(|x| x as &mut dyn XcPotential),
            opts: run.opts,
            par: run.par,
        })
    }

    fn scf_environment(
        &self,
        cx: &CalcContext,
        setup: &ScfSetup,
        scf: &mut dyn ScfLoop,
        pcm: &PcmSetup,
    ) -> Result<Setup<Box<dyn ScfEnvironment>>, LinusError> {
        crate::solvent::scf::cpu_environment(cx, setup, scf, pcm).map(Setup::Ready)
    }

    fn gradient_xc<'a>(
        &self,
        cx: &'a CalcContext,
        grid_response: bool,
    ) -> Result<Setup<Box<dyn GradientTerm + 'a>>, LinusError> {
        Ok(Setup::Ready(Box::new(XcGradient { cx, grid_response })))
    }

    fn gradient_jk<'a>(
        &self,
        cx: &'a CalcContext,
        xop: Option<Operator>,
    ) -> Result<Setup<Box<dyn GradientTerm + 'a>>, LinusError> {
        Ok(Setup::Ready(Box::new(JkGradient { cx, xop })))
    }

    fn solvent_gradient<'a>(
        &self,
        cx: &'a CalcContext,
        gs: &GroundState,
        so: &SolventSettings,
        _timers: &mut Timers,
    ) -> Result<Setup<Box<dyn SolventGradient + 'a>>, LinusError> {
        let pcm = cx.solvent_state(gs, so)?;
        Ok(Setup::Ready(Box::new(CpuSolventGradient { cx, pcm })))
    }

    fn hessian_jk<'a>(
        &self,
        cx: &'a CalcContext,
        _inputs: &TermInputs<'_>,
        xop: Option<Operator>,
        pairs: &'a PairSet,
        _timers: &mut Timers,
    ) -> Result<Setup<Box<dyn HessianTerm + 'a>>, LinusError> {
        Ok(Setup::Ready(Box::new(HessianJk { cx, xop, pairs })))
    }

    fn hessian_xc<'a>(
        &self,
        _cx: &'a CalcContext,
        xc: &'a XcIntegrator,
        _input: &HessianInput<'_>,
        _timers: &mut Timers,
    ) -> Result<Setup<Box<dyn XcHessianEngine + 'a>>, LinusError> {
        Ok(Setup::Ready(Box::new(xc)))
    }

    fn response_jk(
        &self,
        cx: &CalcContext,
        req: &ResponseRequest<'_>,
        _field: bool,
        pairs: &PairSet,
        par: Par,
        timers: &mut Timers,
    ) -> Result<Setup<ResponseSetup>, LinusError> {
        cx.cpu_response_jk(req, pairs, par, timers)
            .map(Setup::Ready)
    }

    fn response_kernel<'a>(
        &self,
        _cx: &CalcContext,
        xc: &'a XcIntegrator,
        _occ: &[MatRef<'_, f64>],
        _kernel: KernelUse,
        _timers: &mut Timers,
    ) -> Result<Setup<Box<dyn ResponseKernel + 'a>>, LinusError> {
        Ok(Setup::Ready(Box::new(xc)))
    }

    fn vv10_sweeps(
        &self,
        _cx: &CalcContext,
        _ground: &Vv10Ground,
        _prefix: &str,
        _timers: &mut Timers,
    ) -> Result<Setup<Option<Box<dyn Vv10PairSweep>>>, LinusError> {
        Ok(Setup::Ready(None))
    }
}

/// The SCF's J/K builder on the CPU ([`CalcContext::cpu_scf_jk`]).
pub(crate) struct CpuScfJk(pub Box<dyn JkBuilder>);

impl ScfJk for CpuScfJk {
    fn builder(&mut self) -> &mut dyn JkBuilder {
        self.0.as_mut()
    }

    fn into_builder(self: Box<Self>) -> Box<dyn JkBuilder> {
        self.0
    }
}

/// The geometry's shared integrator (`crate::context::IntegralContext`) as the SCF's XC term:
/// its evaluations take it by shared reference.
struct CpuXc(Arc<XcIntegrator>);

impl XcPotential for CpuXc {
    fn compute(
        &mut self,
        d: &XcDensity<'_>,
        _par: Par,
    ) -> Result<(f64, Mat<f64>, Mat<f64>), LinusError> {
        Ok(self.0.potential(d.da, d.db, d.restricted))
    }

    fn keeps_state(&self) -> bool {
        self.0.vv10().is_some()
    }
}

impl ScfXc for CpuXc {
    fn vv10_stats(&self) -> Option<Vv10Stats> {
        self.0.vv10().map(|v| v.stats())
    }
}

/// The XC engine's part of a gradient on the CPU (VV10 included): the integrator of the SCF
/// grid's (timed as `grad_xc`, VV10's share as `grad_vv10`), with the grid's response when
/// `grid_response`; nothing for Hartree-Fock.
struct XcGradient<'a> {
    cx: &'a CalcContext,
    grid_response: bool,
}

impl GradientTerm for XcGradient<'_> {
    fn lane(&self) -> Lane {
        Lane::Host
    }

    fn gradient(
        &mut self,
        inputs: &GradientInputs<'_>,
        timers: &mut Timers,
    ) -> Result<Setup<Vec<[f64; 3]>>, LinusError> {
        let Some(x) = self.cx.xc_on(inputs.ground.grid(), timers)? else {
            return Ok(Setup::Ready(vec![[0.0; 3]; self.cx.mol.natom()]));
        };
        let restricted = inputs.ground.scf.restricted;
        let t_nl = x.vv10().map(|v| v.stats().total());
        let g = timers.time("grad_xc", || {
            x.gradient(inputs.da, inputs.db, restricted, self.grid_response)
        });
        if let (Some(v), Some(t0)) = (x.vv10(), t_nl) {
            timers.add("grad_vv10", v.stats().total() - t0);
        }
        Ok(Setup::Ready(g))
    }
}

/// The J/K method's part of a gradient on the CPU ([`CalcContext::cpu_gradient_2e`]) with the
/// exact exchange `xop`, on the gradient's derivative pairs (the one-electron pass's).
struct JkGradient<'a> {
    cx: &'a CalcContext,
    xop: Option<Operator>,
}

impl GradientTerm for JkGradient<'_> {
    fn lane(&self) -> Lane {
        Lane::Host
    }

    fn gradient(
        &mut self,
        inputs: &GradientInputs<'_>,
        timers: &mut Timers,
    ) -> Result<Setup<Vec<[f64; 3]>>, LinusError> {
        let pairs = self.cx.gradient_pairs(inputs.ground);
        self.cx
            .cpu_gradient_2e(inputs, &pairs, self.xop, timers)
            .map(Setup::Ready)
    }
}

/// The J/K method's explicit Hessian terms on the CPU ([`CalcContext::cpu_hessian_2e`]).
struct HessianJk<'a> {
    cx: &'a CalcContext,
    xop: Option<Operator>,
    pairs: &'a PairSet,
}

impl HessianTerm for HessianJk<'_> {
    fn add_hessian(
        &mut self,
        inputs: &TermInputs<'_>,
        out: &mut ExplicitHessian,
        timers: &mut Timers,
    ) -> Result<(), LinusError> {
        self.cx.cpu_hessian_2e(
            inputs.ground,
            &inputs.c_occ,
            &inputs.dsf,
            inputs.dtf,
            self.xop,
            self.pairs,
            &mut out.h,
            &mut out.fxo,
            inputs.par,
            timers,
        )
    }
}
