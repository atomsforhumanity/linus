//! The SCF on the device loop (`backend.scf_loop: "device"`, feature `gpu`; family `scf_loop` of
//! the GPU's provider, `crate::backend`): the device's J/K and XC engines as the loop's Fock terms
//! (`crate::backend::gpu::scf_control`), taken back from the SCF's terms the provider set up
//! ([`crate::backend::scf::ScfTerms`]), the loop as an [`ScfLoop`] that the SCF stage runs with the
//! solvent's reaction field (`CalcContext::scf_on`: on the device's buffers when the PCM runs
//! there, else from the CPU; the `predicted` mode's Coulomb hook fed by the device's J builds), and
//! the hand-over to the host loop of what it does beyond plain DIIS (a stability analysis beyond
//! the probe, a two-cycle; `crate::backend::gpu::scf_control`), which redoes the device's last
//! iteration from its state on the same engines and continues.

use crate::backend::gpu::device_error;
use crate::backend::gpu::device_jk::{self, DeviceJk};
use crate::backend::gpu::df::scf_build_error;
use crate::backend::gpu::scf_control::{DeviceEnv, DeviceFock, DeviceRun};
use crate::backend::gpu::xc::GpuXcPotential;
use crate::backend::scf::{ScfRun, ScfTerms, ScfXc};
use crate::backend::Setup;
use crate::driver::backend::Family;
use linus_base::error::LinusError;
use linus_base::terms::HookOutcome;
use linus_base::terms::{EnvironmentTerm, XcPotential};
use linus_gpu::scf::{DeviceJ, DeviceTerms};
use linus_scf::{ScfLoop, ScfResult};
use std::any::Any;

/// Whether the device loop takes the SCF's terms: where the plan puts the loop on the device and
/// the terms did not give it up while they were set up (`terms.loop_declined`), when the XC term
/// is the device's with VV10 on the device too (or there is none); declined otherwise, with why
/// (the host loop runs).
pub(crate) fn takes(terms: &ScfTerms) -> Setup<()> {
    if let Some(why) = &terms.loop_declined {
        return Setup::Declined(why.clone());
    }
    if !device_jk::is_device(&*terms.jk) {
        return Setup::Declined("the J/K setup fell back to the CPU, so the host loop ran".into());
    }
    let Some(xc) = &terms.xc else {
        return Setup::Ready(());
    };
    let xc: &dyn Any = &**xc;
    match xc.downcast_ref::<GpuXcPotential>() {
        Some(x) if x.vv10_on_device() => Setup::Ready(()),
        Some(_) => {
            Setup::Declined("the VV10 setup fell back to the CPU, so the host loop ran".into())
        }
        None => Setup::Declined("the XC setup fell back to the CPU, so the host loop ran".into()),
    }
}

/// The device loop over the SCF's terms (taken by [`takes`]).
pub(crate) struct DeviceScf<'a> {
    terms: &'a mut ScfTerms,
    run: ScfRun<'a>,
    /// The `predicted` PCM mode's Coulomb hook (the density-fitted PCM keeps the host loop and the
    /// CPU's J, `crate::driver::backend`).
    hook: Option<Box<dyn linus_base::terms::CoulombHook>>,
}

impl<'a> DeviceScf<'a> {
    pub(crate) fn new(terms: &'a mut ScfTerms, run: ScfRun<'a>) -> Self {
        DeviceScf {
            terms,
            run,
            hook: None,
        }
    }
}

impl ScfLoop for DeviceScf<'_> {
    /// Taken for the device's J builds ([`crate::backend::gpu::df::CoulombFit`]).
    fn set_coulomb_hook(
        &mut self,
        hook: Option<Box<dyn linus_base::terms::CoulombHook>>,
    ) -> Result<HookOutcome, LinusError> {
        // the hook rides on the DF engine's J builds: the memory-lean J has no fit to hand over
        let fit = device_jk::of(&mut *self.terms.jk).is_some_and(|b| b.coulomb_fit());
        if hook.is_some() && !fit {
            return Ok(HookOutcome::FellBack {
                reason: "the memory-lean J (no stored DF tensor) does not hand over its fit".into(),
            });
        }
        self.hook = hook;
        Ok(HookOutcome::Installed)
    }

    fn run(&mut self, env: Option<&mut dyn EnvironmentTerm>) -> Result<ScfResult, LinusError> {
        let ScfRun { inp, opts, par } = self.run;
        let twice = || LinusError::Internal("the device SCF ran twice".into());
        let builder = device_jk::of(&mut *self.terms.jk)
            .ok_or_else(|| LinusError::Internal("the device loop with the CPU's J/K".into()))?;
        let DeviceJk {
            gpu,
            mut j,
            aux,
            gridk,
        } = builder.take_device(par)?.ok_or_else(twice)?;
        let hook = match (self.hook.take(), &mut j, &aux) {
            (Some(h), DeviceJ::Df(jk), Some(aux)) => {
                let fit = crate::backend::gpu::df::CoulombFit::new(&gpu, aux)
                    .map_err(|e| device_error(Family::Jk, e))?;
                jk.set_keep_coulomb_fit(true);
                Some((fit, h))
            }
            (Some(_), ..) => {
                return Err(LinusError::Internal(
                    "a Coulomb hook on the device loop's memory-lean J".into(),
                ))
            }
            (None, ..) => None,
        };
        let (xc, nl, cpu) = match self.terms.xc.take() {
            Some(x) => {
                let x: Box<dyn Any> = x;
                let x = x.downcast::<GpuXcPotential>().map_err(|_| {
                    LinusError::Internal("the device loop with the CPU's XC term".into())
                })?;
                let (e, nl, c) = x.into_parts();
                (Some(e), nl, Some(c))
            }
            None => (None, None, None),
        };
        let env_name = env.as_ref().map(|e| e.name());
        let mut terms = DeviceFock {
            dev: DeviceTerms::new(j, xc).with_vv10(nl).with_gridk(gridk),
            env: env.map(|term| DeviceEnv::of(term, par)),
            hook,
        };
        let run = crate::backend::gpu::scf_control::run(&gpu, inp, &mut terms, opts, env_name, par)
            .map_err(|e| {
                // the environment term's own error, else the device's: the loop's input errors
                // (a scf.lindep that leaves fewer orbitals than occupied ones) are the
                // calculation's, as on the host loop
                terms
                    .env
                    .as_mut()
                    .and_then(|env| env.failure())
                    .unwrap_or_else(|| match e {
                        linus_gpu::GpuError::Input(m) => LinusError::Input(m),
                        e => scf_build_error(Family::ScfLoop, e),
                    })
            })?;
        // the engines back into the SCF's builders: the J/K's (the host loop continues on them,
        // their timers read them) and the XC term
        let DeviceFock { dev, env, hook, .. } = terms;
        let DeviceTerms {
            j, gridk, xc, vv10, ..
        } = dev;
        device_jk::of(&mut *self.terms.jk)
            .ok_or_else(|| LinusError::Internal("the device loop with the CPU's J/K".into()))?
            .put_device(DeviceJk {
                gpu: gpu.clone(),
                j,
                aux,
                gridk,
            })?;
        let hxc = match (xc, cpu) {
            (Some(e), Some(c)) => Some(GpuXcPotential::from_parts(gpu.clone(), e, vv10, c)),
            _ => None,
        };
        let (resume, device_timers) = match run {
            DeviceRun::Done(r) => {
                self.terms.xc = hxc.map(|x| Box::new(x) as Box<dyn ScfXc>);
                return Ok(*r);
            }
            DeviceRun::HandOver(resume, timers) => (resume, timers),
        };
        // what the host loop does beyond plain DIIS (a stability analysis, a two-cycle): it
        // redoes the device's last iteration from its state on the same engines and continues
        let jk = self.terms.jk.builder();
        if let Some((_, h)) = hook {
            if let HookOutcome::FellBack { reason } = jk.set_coulomb_hook(Some(h))? {
                return Err(LinusError::Unsupported(format!(
                    "the host loop after the device loop could not take the PCM's Coulomb hook: \
                     {reason}"
                )));
            }
        }
        let mut hxc = hxc;
        let xcp = hxc.as_mut().map(|x| x as &mut dyn XcPotential);
        let envt = env.map(|e| e.into_term());
        let mut host = linus_scf::run_scf_resumed(inp, jk, xcp, envt, opts, par, resume)?;
        host.timers.merge(&device_timers);
        self.terms.xc = hxc.map(|x| Box::new(x) as Box<dyn ScfXc>);
        Ok(host)
    }
}
