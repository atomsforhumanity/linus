//! The SCF's terms as the backends set them up: its J/K (family `jk`), its XC term (families `xc`
//! and `vv10`) and the loop that runs them (family `scf_loop`), in the order their device
//! memory is decided:
//! 1. the J/K ([`CalcContext::scf_terms`]), then the XC term on the memory it left;
//! 2. the J/K's fit before the first iteration ([`CalcContext::fit_scf_terms`]): a device J/K
//!    sizes the SCF's builds known in advance against the memory the other engines left, gives
//!    the device loop up when its state and builds do not fit, and hands over to the CPU's J/K
//!    when the host loop's builds do not fit either (decided here, never between iterations);
//! 3. the loop ([`CalcContext::scf_loop`]): the device loop over a device J/K and XC term where
//!    the plan puts the loop there, else the host loop.
//!
//! The SCF stage (`crate::driver::Session::scf_stage`) runs the loop with the solvent's
//! environment term (`CalcContext::scf_on`) and reads VV10's statistics from the XC term after it.

use super::{Backend, CpuBackend, Setup};
use crate::driver::backend::{Family, Placement};
use crate::driver::ground::ScfSetup;
use crate::driver::CalcContext;
use faer::Par;
use linus_base::error::LinusError;
use linus_base::terms::JkBuilder;
use linus_base::terms::XcPotential;
use linus_core::timing::Timers;
use linus_dft::vv10::Vv10Stats;
use linus_ints::PairSet;
use linus_jk::df::ResolvedDf;
use linus_scf::{ScfConfig, ScfInput, ScfLoop};
use std::any::Any;
use std::sync::Arc;

/// The SCF's J/K as a backend set it up.
pub(crate) trait ScfJk: Any + Send {
    /// The builder the host loop calls.
    fn builder(&mut self) -> &mut dyn JkBuilder;

    /// The builder alone (`crate::driver::Session::build_jk`).
    fn into_builder(self: Box<Self>) -> Box<dyn JkBuilder>;

    /// Before the first iteration, with every SCF engine set up: the builds of the SCF `run`
    /// known in advance (from its guess, electrons and settings) against what the J/K can hold.
    /// Declined: the CPU's J/K takes over (with the reason). A J/K that gives the device loop up
    /// says why in `loop_declined`. The CPU's J/K always fits.
    fn fit(
        &mut self,
        _cx: &CalcContext,
        _run: &ScfRun<'_>,
        _loop_declined: &mut Option<String>,
    ) -> Result<Setup<()>, LinusError> {
        Ok(Setup::Ready(()))
    }
}

/// The SCF's J/K as its backend set it up, and what its setup resolved to
/// (`resolved.record.df`).
pub(crate) type ScfJkBuilt = (Box<dyn ScfJk>, Option<ResolvedDf>);

/// The SCF's XC term as a backend set it up.
pub(crate) trait ScfXc: XcPotential + Any {
    /// VV10's statistics of the evaluations so far (-V functionals).
    fn vv10_stats(&self) -> Option<Vv10Stats>;
}

/// The SCF's terms (module docs).
pub(crate) struct ScfTerms {
    pub jk: Box<dyn ScfJk>,
    /// What the J/K's setup resolved to (`resolved.record.df`).
    pub df: Option<ResolvedDf>,
    /// The XC term (None for Hartree-Fock).
    pub xc: Option<Box<dyn ScfXc>>,
    /// Why the device loop was given up while the terms were set up, where the plan put the
    /// loop on the device: the J/K's setup fell back to the CPU, or the device loop's state and
    /// builds did not fit beside the engines.
    pub loop_declined: Option<String>,
}

/// What an SCF loop runs on: the SCF's input, its settings and the products' parallelism.
#[derive(Clone, Copy)]
pub(crate) struct ScfRun<'a> {
    pub inp: &'a ScfInput<'a>,
    pub opts: &'a ScfConfig,
    pub par: Par,
}

impl CalcContext {
    /// The SCF's J/K builder and XC term set up with `setup` on the backends the plan puts them
    /// on (the J/K first: the XC term's device engine is set up on the memory it leaves); the
    /// shell pairs `pairs` are done with once the J/K is set up.
    pub(crate) fn scf_terms(
        &self,
        pairs: Arc<PairSet>,
        setup: &ScfSetup,
        timers: &mut Timers,
    ) -> Result<ScfTerms, LinusError> {
        let (jk, df) = self.resolve(Family::Jk, |b| b.scf_jk(self, &pairs, setup, timers))?;
        // the J/K's setup fell back: the device loop with it
        let loop_declined = (self.placement(Family::ScfLoop) == Some(Placement::Device)
            && self.ran(Family::Jk) != Some(Placement::Device))
        .then(|| "the J/K setup fell back to the CPU, so the host loop ran".to_string());
        // the pairs are done with once the setups are built (crate::context)
        drop(pairs);
        self.integrals.release_pairs();
        // an energy's metric: its fitting holds it through the iterations
        if setup.purpose == crate::driver::Purpose::Energy {
            self.integrals.release_metrics();
        }
        let xc = match self.xc_on(&setup.grid, timers)? {
            Some(cpu) => Some(self.resolve(Family::Xc, |b| b.scf_xc(self, cpu.clone(), timers))?),
            None => None,
        };
        Ok(ScfTerms {
            jk,
            df,
            xc,
            loop_declined,
        })
    }

    /// The J/K's fit before the first iteration ([`ScfJk::fit`]; module docs): a J/K that does
    /// not fit is replaced by the CPU's (its setup's record replacing the device's), with the
    /// reason recorded.
    pub(crate) fn fit_scf_terms(
        &self,
        terms: ScfTerms,
        run: &ScfRun<'_>,
        setup: &ScfSetup,
        timers: &mut Timers,
    ) -> Result<ScfTerms, LinusError> {
        let ScfTerms {
            mut jk,
            df,
            xc,
            mut loop_declined,
        } = terms;
        let why = match jk.fit(self, run, &mut loop_declined)? {
            Setup::Ready(()) => {
                return Ok(ScfTerms {
                    jk,
                    df,
                    xc,
                    loop_declined,
                })
            }
            Setup::Declined(why) => why,
        };
        self.note_setup(Family::Jk, Some(why));
        // the device's J/K goes before the CPU's is set up
        drop(jk);
        let pairs = timers.time("pairs", || {
            self.integrals
                .pairs(&self.basis, linus_ints::DerivOrder::Zero)
        });
        let (jk, df) = self.cpu_scf_jk(&pairs, setup, timers)?;
        Ok(ScfTerms {
            jk: Box::new(super::cpu::CpuScfJk(jk)),
            df,
            xc,
            loop_declined,
        })
    }

    /// The loop that runs the SCF's terms (family `scf_loop`): the device's where the plan puts
    /// the loop there and the device takes the terms, else the host loop (the reason recorded).
    pub(crate) fn scf_loop<'a>(
        &'a self,
        terms: &'a mut ScfTerms,
        run: ScfRun<'a>,
    ) -> Result<Box<dyn ScfLoop + 'a>, LinusError> {
        let b: &dyn Backend = match (self.placement(Family::ScfLoop), self.backend.provider()) {
            (Some(Placement::Device), Some(device)) => match device.takes_scf_loop(self, terms)? {
                Setup::Ready(()) => {
                    self.note_setup(Family::ScfLoop, None);
                    device
                }
                Setup::Declined(why) => {
                    self.note_setup(Family::ScfLoop, Some(why));
                    &CpuBackend
                }
            },
            _ => &CpuBackend,
        };
        Ok(b.scf_loop(self, terms, run))
    }
}
