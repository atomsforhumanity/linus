//! The SCF's terms as the backends set them up: its J/K (family `jk`), its XC term (families `xc`
//! and `vv10`) and the loop that runs them (family `scf_loop`), in the order their device
//! memory is decided:
//! 1. the J/K, then the XC term on the memory it left ([`CalcContext::scf_setups`], with the
//!    host's own setup: S, h and the starting densities, beside a device J/K's setup);
//! 2. the J/K's fit before the first iteration ([`CalcContext::fit_scf_terms`]): a device J/K
//!    sizes the SCF's builds known in advance against the memory the other engines left, gives
//!    the device loop up when its state and builds do not fit, and hands over to the CPU's J/K
//!    when the host loop's builds do not fit either (decided here, never between iterations);
//! 3. the loop ([`CalcContext::scf_loop`]): the device loop over a device J/K and XC term where
//!    the plan puts the loop there, else the host loop.
//!
//! The SCF stage (`crate::driver::Session::scf_stage`) runs the loop with the solvent's
//! environment term (`CalcContext::scf_on`) and reads VV10's statistics from the XC term after it.

use super::{threads, Backend, CpuBackend, NuclearAttraction, Setup};
use crate::driver::backend::{Family, Placement};
use crate::driver::ground::ScfSetup;
use crate::driver::record::Decision;
use crate::driver::CalcContext;
use faer::{Mat, Par};
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

    /// What the J/K resolved to after [`ScfJk::fit`] (`resolved.record.df`), where the fit can
    /// change it (the device's K block); None: the setup's record stands.
    fn df_record(&mut self) -> Option<ResolvedDf> {
        None
    }

    /// The engines' own times of the SCF's builds by phase (a device engine's), merged into the
    /// SCF's timers when it is done (None: none beyond the SCF's `jk`).
    fn timers(&self) -> Option<Timers> {
        None
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

/// The SCF's J/K as [`CalcContext::scf_jk_term`] set it up, before its XC term.
pub(crate) struct JkTerm {
    jk: Box<dyn ScfJk>,
    df: Option<ResolvedDf>,
    loop_declined: Option<String>,
}

/// The SCF's host matrices ([`CalcContext::scf_setups`]): the overlap S, the core Hamiltonian h
/// (T + V + the ECPs) and the starting densities (None: the core guess).
pub(crate) struct ScfHost {
    pub s: Mat<f64>,
    pub h: Mat<f64>,
    pub guess: Option<(Mat<f64>, Mat<f64>)>,
}

/// [`ScfHost`] before its nuclear attraction is complete: h without V where the device computes
/// V (its packed job, or why the device declined), the ECPs apart.
struct HostParts {
    s: Mat<f64>,
    h: Mat<f64>,
    v: Option<Setup<Box<dyn NuclearAttraction>>>,
    ecp: Option<Mat<f64>>,
    guess: Option<(Mat<f64>, Mat<f64>)>,
}

impl CalcContext {
    /// The SCF's setups: its terms on the backends the plan puts them on (the J/K, then the XC
    /// term on the device memory it left) and the host's own (S, h, the starting densities
    /// `start` gives, and the molecular grid, which the XC term's setup then finds built). A
    /// device J/K's setup runs mostly on the device: the host's setup runs beside it, each on
    /// its share of the session's threads ([`threads::split_beside`]; with one thread, and with
    /// the CPU's J/K, one after another). The shell pairs `pairs` are released once both are
    /// done (a device PCM's setup releases them after screening them).
    pub(crate) fn scf_setups(
        &self,
        pairs: Arc<PairSet>,
        setup: &ScfSetup,
        start: impl FnOnce(&mut Timers) -> Option<(Mat<f64>, Mat<f64>)> + Send,
        timers: &mut Timers,
    ) -> Result<(ScfHost, ScfTerms), LinusError> {
        let device_jk = self.placement(Family::Jk) == Some(Placement::Device);
        // the nuclear attraction of large molecules on the device: packed beside the J/K setup,
        // its pass after it (crate::backend::gpu::vnuc)
        let charges = self.mol.point_charges();
        let vnuc = self.nuclear_attraction_place(&pairs, charges.len());
        let v_on_device = matches!(vnuc, Some((true, _)));
        let host = |timers: &mut Timers| {
            self.scf_host_parts(&pairs, &charges, v_on_device, setup, start, timers)
        };
        let (host, jk) = match device_jk
            .then(|| threads::split_beside(self.plan.threads()))
            .flatten()
        {
            Some((beside, own)) => {
                let mut th = Timers::default();
                let (host, jk) = rayon::join(
                    || threads::on_pool(beside, || host(&mut th)),
                    || threads::on_pool(own, || self.scf_jk_term(&pairs, setup, timers)),
                );
                timers.merge(&th);
                let jk = jk?;
                (host?, jk)
            }
            None => {
                let jk = self.scf_jk_term(&pairs, setup, timers)?;
                (host(timers)?, jk)
            }
        };
        let HostParts {
            s,
            mut h,
            v,
            ecp,
            guess,
        } = host;
        // h = T + V (the device's, or the host's where the device declined), then the ECPs
        if let Some(job) = v {
            let v = match job {
                Setup::Ready(j) => timers.time("one_electron_v_device", || j.compute())?,
                Setup::Declined(why) => Setup::Declined(why),
            };
            let (v, place, reason) = match v {
                Setup::Ready(v) => (v, "gpu", vnuc.map(|(_, r)| r).unwrap_or_default()),
                Setup::Declined(why) => (
                    timers.time("one_electron", || self.nuclear_attraction_host(&pairs)),
                    "cpu",
                    format!("the device declined ({why}): the host's"),
                ),
            };
            h += v;
            self.record
                .decide(Decision::NuclearAttraction { place, reason });
        } else if let Some((_, reason)) = vnuc {
            self.record.decide(Decision::NuclearAttraction {
                place: "cpu",
                reason,
            });
        }
        if let Some(u) = ecp {
            h += u;
        }
        // the pairs are done with once the setups are built (crate::context), unless a device
        // PCM's setup screens them next (crate::backend::gpu::pcm), which releases them
        // (crate::solvent::scf::open_environment)
        drop(pairs);
        let pcm_on_device =
            self.plan.solvent().is_some() && self.placement(Family::Pcm) == Some(Placement::Device);
        if !pcm_on_device {
            self.integrals.release_pairs();
        }
        let terms = self.scf_xc_term(jk, setup, timers)?;
        if device_jk {
            // the report's dipole integrals, built on the session's pool while the SCF iterates
            // on the device (its threads mostly wait then)
            self.integrals.prefetch_dipole(&self.pool, &self.basis);
        }
        Ok((ScfHost { s, h, guess }, terms))
    }

    /// The host's part of [`Self::scf_setups`]: S, h (T, and V unless `v_on_device`: then its
    /// device job, packed here), the ECP matrix, the molecular grid and the starting densities.
    fn scf_host_parts(
        &self,
        pairs: &PairSet,
        charges: &[(f64, [f64; 3])],
        v_on_device: bool,
        setup: &ScfSetup,
        start: impl FnOnce(&mut Timers) -> Option<(Mat<f64>, Mat<f64>)>,
        timers: &mut Timers,
    ) -> Result<HostParts, LinusError> {
        let (s, h, v) = if v_on_device {
            let (s, t) = timers.time("one_electron", || self.overlap_kinetic());
            let job = timers.time("one_electron_v_pack", || {
                self.nuclear_attraction(pairs, charges)
            })?;
            (s, t, Some(job))
        } else {
            let (s, h) = timers.time("one_electron", || self.one_electron_no_ecp(pairs));
            (s, h, None)
        };
        let ecp = timers.time("ecp", || self.ecp_matrix(pairs));
        self.xc_on(&setup.grid, timers)?;
        let guess = start(timers);
        Ok(HostParts {
            s,
            h,
            v,
            ecp,
            guess,
        })
    }

    /// The J/K half of [`Self::scf_setups`]: the SCF's J/K builder on the backend the plan puts
    /// it on.
    pub(crate) fn scf_jk_term(
        &self,
        pairs: &PairSet,
        setup: &ScfSetup,
        timers: &mut Timers,
    ) -> Result<JkTerm, LinusError> {
        let (jk, df) = self.resolve(Family::Jk, |b| b.scf_jk(self, pairs, setup, timers))?;
        // the J/K's setup fell back: the device loop with it
        let loop_declined = (self.placement(Family::ScfLoop) == Some(Placement::Device)
            && self.ran(Family::Jk) != Some(Placement::Device))
        .then(|| "the J/K setup fell back to the CPU, so the host loop ran".to_string());
        // an energy's metric: its fitting holds it through the iterations
        if setup.purpose == crate::driver::Purpose::Energy {
            self.integrals.release_metrics();
        }
        Ok(JkTerm {
            jk,
            df,
            loop_declined,
        })
    }

    /// The XC half of [`Self::scf_setups`]: the XC term set up after the J/K `jk` (on the device
    /// memory it left), on the geometry's molecular grid (from the integral context: built here
    /// unless an earlier call built it).
    pub(crate) fn scf_xc_term(
        &self,
        jk: JkTerm,
        setup: &ScfSetup,
        timers: &mut Timers,
    ) -> Result<ScfTerms, LinusError> {
        let JkTerm {
            jk,
            df,
            loop_declined,
        } = jk;
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
                let df = jk.df_record().or(df);
                return Ok(ScfTerms {
                    jk,
                    df,
                    xc,
                    loop_declined,
                });
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
