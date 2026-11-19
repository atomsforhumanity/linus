//! The device-resident SCF loop's controller (`backend.scf_loop: "device"`; its loop is
//! [`super::scf_loop`]): linus's SCF input mapped onto `linus_gpu::scf::run_scf`, its controller
//! ([`DeviceControls`]: the host loop's own controls), its Fock terms ([`DeviceFock`]: the device's
//! J/K and XC engines, and an environment term such as the solvent's reaction field on the CPU),
//! and its result back as linus's [`ScfResult`] (orbitals, densities and Fock matrices downloaded
//! once at the end).
//!
//! The device loop is the mechanism of linus's [`linus_scf::run_scf`] for what that runs with a
//! density-fitted J/K builder and the default algorithm (DIIS, the level shift, canonical
//! orthogonalization; no ADIIS, SOSCF, damping or incremental J/K builds). Its decisions are the
//! host loop's, from the same code: it asks [`DeviceControls`] after every Fock build, which
//! answers with `linus_scf::control`'s units and the accelerators' DIIS and level-shift rules
//! (a converged iterate of the PCM's predicted charges is confirmed by an exact build, as on the
//! host). It
//! agrees with the host loop to round-off (a different eigensolver and GEMM order) with the
//! same iteration counts, and is bitwise reproducible run to run.
//!
//! UHF/UKS stability, as the host loop decides it:
//! - the cheap probe (`scf.stability_probe`) runs here, on the device: the shared per-irrep
//!   Davidson searches (`stability::lowest_hessian_mode`, their starts built on the host from the
//!   device's orbitals, as the host loop builds them) with their Hessian-vector products from
//!   [`dev::Rotations`] (the rotated orbitals, their Fock build on the loop's own engines and
//!   the orbital gradient on the device; the small functions of the rotation on the host). When
//!   it finds no negative mode the SCF ends here, as on the host;
//! - the zero-mode check of a Hessian SCF (`mode_check`) is that probe, with the check's products
//!   where they are more: its search at the accepted point is the record, and a negative one hands
//!   over like the probe's (where no search ran at the accepted point, the check alone, after it);
//! - the closed-shell probe of a converged RHF/RKS point (`scf.stability_closed_probe`, decision
//!   D36) likewise: [`dev::Rotations`] of the closed-shell reference (alpha = beta, UKS-type Fock
//!   builds on the loop's engines) from the host's external start, and afterwards, when a term
//!   keeps something of the last state it saw (the solvent, the PCM's Coulomb fit, VV10's last
//!   energy), one build at the converged restricted state ([`dev::LoopConfig::keep_occupied`]);
//! - the escalation (when DIIS stagnates or at 2/3 of `max_iter`, within it) hands over after the
//!   step it is due at ([`dev::StepPlan::hand_over`]); the host loop decides there, as it would have
//!   (a plateau is accepted, else the escalation spends the rest of the budget);
//! - the soft-mode plateau rule applies to the device's iterations at `max_iter`; a plateau the
//!   stability trigger may analyse hands over, and the host loop analyses it before accepting it;
//! - a caller's stop request ends the loop where the host loop's would end it, after an iteration
//!   that steps ([`dev::Verdict::Stop`]), unconverged with its last step `stopped`;
//! - everything else the host loop does beyond plain DIIS (a negative probe, the saddle check's
//!   full analysis and its steps, `scf.stability`, a two-cycle), all rare, hands over: the
//!   device loop stops at that iteration ([`dev::HandOver`]) and the host loop redoes it from
//!   the device's state and the controls as they were before it ([`ScfResume`]), and continues
//!   as it would have from the start.

use faer::{Mat, Par};
use linus_base::error::LinusError;
use linus_base::linalg::{mat_from_colmajor, mat_to_colmajor};
use linus_base::terms::{exchange_parts, EnvironmentTerm, XcDensity};
use linus_core::timing::Timers;
use linus_gpu::scf as dev;
use linus_gpu::{DeviceBuf, Gpu, GpuError};
use linus_scf::accel::soscf::rotation_functions;
use linus_scf::accel::{level_shift, DiisRule};
use linus_scf::analysis;
use linus_scf::control::{AtBudget, Controls, Decision, Observation};
use linus_scf::fock::EnergyParts;
use linus_scf::result::Finished;
use linus_scf::stability::{
    gate_gap, lowest_hessian_mode, unflatten, RotatedGradient, Search, Starts, PROBE_NEGATIVE,
};
use linus_scf::{
    ClosedShellProbe, DiisDrop, ScfAlgorithm, ScfConfig, ScfInput, ScfIteration, ScfResult,
    ScfResume, StabilityCheck,
};

/// The device loop's mechanism for linus's settings (the device eigensolver is divide and
/// conquer).
fn loop_config(o: &ScfConfig, controls: &Controls) -> dev::LoopConfig {
    dev::LoopConfig {
        max_iter: o.max_iter,
        diis_space: o.algo.diis_space,
        lindep: o.algo.lindep,
        eigensolver: dev::Eigensolver::DivideConquer,
        // the lowest-energy iterate's orbitals: the saddle check's direction, the escalation's
        // start
        keep_lowest: controls.keeps_lowest_orbitals(),
        // a plateau at max_iter that the stability trigger may analyse: on the host from the
        // loop's state
        handover_at_max_iter: o.algo.plateau_window > 0,
        // the build at the converged state after the closed-shell probe or the mode check
        keep_occupied: o.algo.closed_shell_probe.unwrap_or(0) > 0 || o.mode_check > 0,
    }
}

/// The device loop's controller: the host loop's controls ([`Controls`]) and the accelerators'
/// DIIS and level-shift rules, so the two loops decide alike from the same code.
pub(crate) struct DeviceControls {
    controls: Controls,
    /// Where the iterations' progress events go (the SCF's settings' `progress`).
    progress: linus_base::progress::Progress,
    /// The controls before the last iteration the loop reported: a hand-over at a stop redoes that
    /// iteration on the host from them.
    before: Controls,
    diis: DiisRule,
    /// The device subspace's drop rule (it unpins as the host loop's does).
    drop: DiisDrop,
    algo: ScfAlgorithm,
}

impl DeviceControls {
    fn new(opts: &ScfConfig, restricted: bool) -> Self {
        let controls = Controls::new(opts, restricted);
        DeviceControls {
            controls,
            progress: opts.progress.clone(),
            before: controls,
            diis: DiisRule::new(&opts.algo),
            drop: opts.algo.diis_drop,
            algo: opts.algo.clone(),
        }
    }
}

impl dev::ScfDecide for DeviceControls {
    fn observe(&mut self, o: &dev::Observation<'_>) -> dev::Verdict {
        let verdict = self.decide(o);
        // the iteration's event, unless the host loop redoes it after a hand-over (it reports
        // it then)
        if !matches!(verdict, dev::Verdict::Analyse | dev::Verdict::TwoCycle) {
            self.progress
                .event(linus_base::progress::Event::ScfIteration {
                    iteration: o.iter,
                    energy: o.energy,
                    delta_energy: o.de,
                    gradient: o.gmax,
                });
        }
        // the caller asks to stop: where the host loop checks, after an iteration that steps (its
        // event reported)
        match verdict {
            dev::Verdict::Step(_) if self.progress.stop_due() => dev::Verdict::Stop,
            v => v,
        }
    }
}

impl DeviceControls {
    /// The verdict of one iteration (`dev::ScfDecide::observe`).
    fn decide(&mut self, o: &dev::Observation<'_>) -> dev::Verdict {
        self.before = self.controls;
        let (decision, lowest) = self.controls.observe(&Observation {
            iter: o.iter,
            guess_iter: o.guess_iter,
            energy: o.energy,
            de: o.de,
            gmax: o.gmax,
            gnorm_pyscf: o.gnorm_pyscf,
            previous: o.previous,
            rebuilt: o.rebuilt,
            incremental: o.incremental,
            switched: false,
            descending: false,
        });
        match decision {
            // the loop stops where the host loop may analyse (it decides after, with the gap)
            Decision::Converged { e_above } if self.controls.stability.may_analyse(e_above) => {
                dev::Verdict::Analyse
            }
            Decision::Converged { .. } => dev::Verdict::Converged,
            Decision::TwoCycle => dev::Verdict::TwoCycle,
            // an incremental build (the PCM's predicted charges) that meets the criteria
            Decision::Confirm => dev::Verdict::Confirm,
            Decision::Continue => {
                let usable = o.iter > o.guess_iter;
                let push = self.diis.admits(o.iter, usable);
                if push && self.diis.unpins(self.drop, o.error_norm, o.diis_norms) {
                    self.drop = DiisDrop::Oldest;
                }
                let stored = o.diis_norms.len() + usize::from(push);
                dev::Verdict::Step(dev::StepPlan {
                    lowest,
                    diis_push: push,
                    diis_victim: push
                        .then(|| self.drop.victim(o.diis_norms, o.diis_capacity))
                        .flatten(),
                    extrapolate: self.diis.extrapolates(o.iter, usable, stored),
                    level_shift: level_shift(&self.algo, o.gmax, false),
                    // the escalation is due after this step: the host loop decides there, as it
                    // would have at the top of its next iteration (the plateau rule first, then the
                    // escalation)
                    hand_over: self.controls.escalation_due(o.iter).is_some(),
                })
            }
        }
    }
}

/// How an SCF on the device loop ended.
pub(crate) enum DeviceRun {
    /// On the device: converged (or not, at `max_iter` or a numerical failure).
    Done(Box<ScfResult>),
    /// To be continued on the host loop from this state ([`linus_scf::run_scf_resumed`]); the
    /// device's timers.
    HandOver(Box<ScfResume>, Timers),
}

/// The Fock terms of the device loop: J/K and XC on the device engines
/// ([`dev::DeviceTerms`]), an environment term on the device ([`DeviceEnvironment`], the GPU's
/// PCM) or on the CPU ([`HostEnvironment`] through [`dev::OnHost`]: the densities down and its
/// Fock matrix up once per iteration), and a Coulomb hook (the `predicted` PCM mode) fed with
/// the fit coefficients of every J build.
pub(crate) struct DeviceFock<'a> {
    pub dev: dev::DeviceTerms,
    pub env: Option<DeviceEnv<'a>>,
    pub hook: Option<(
        crate::backend::gpu::df::CoulombFit,
        Box<dyn linus_base::terms::CoulombHook>,
    )>,
}

/// Where the device loop's environment term runs.
pub(crate) enum DeviceEnv<'a> {
    /// On the device's buffers ([`EnvironmentTerm::device_form`]).
    Device(&'a mut dyn EnvironmentTerm),
    /// On host copies.
    Host(dev::OnHost<HostEnvironment<'a>>),
}

impl<'a> DeviceEnv<'a> {
    /// The device form of `term` when it has one, else the host adapter.
    pub(crate) fn of(term: &'a mut dyn EnvironmentTerm, par: Par) -> Self {
        if device_env(term).is_some() {
            DeviceEnv::Device(term)
        } else {
            DeviceEnv::Host(dev::OnHost::new(HostEnvironment {
                term,
                par,
                failed: None,
            }))
        }
    }

    /// The term's own error that stopped the device loop (it reaches the loop as a device
    /// error; the caller returns this one instead).
    pub(crate) fn failure(&mut self) -> Option<LinusError> {
        match self {
            DeviceEnv::Host(h) => h.terms.failed.take(),
            DeviceEnv::Device(_) => None,
        }
    }

    /// The term (for the host loop that may continue the SCF).
    pub(crate) fn into_term(self) -> &'a mut dyn EnvironmentTerm {
        match self {
            DeviceEnv::Device(t) => t,
            DeviceEnv::Host(h) => h.terms.term,
        }
    }
}

/// An environment term evaluated on the device loop's buffers (the GPU's PCM term).
pub(crate) trait DeviceEnvironment {
    /// E_env at the loop's state, its Fock matrix (one nbf x nbf, the same for both spins) into
    /// `v`.
    fn compute_device(
        &mut self,
        gpu: &Gpu,
        st: &dev::FockState<'_>,
        v: &mut DeviceBuf<f64>,
    ) -> Result<f64, GpuError>;
}

/// The device form of the environment term `t` ([`EnvironmentTerm::device_form`]: the GPU's PCM
/// term, the one environment term with one), if it has one.
fn device_env(t: &mut dyn EnvironmentTerm) -> Option<&mut dyn DeviceEnvironment> {
    let d = t
        .device_form()?
        .downcast_mut::<crate::backend::gpu::pcm::GpuPcmTerm>()?;
    Some(d)
}

/// One device-loop iteration's state in the SCF trace (`linus_scf::trace`, `LINUS_SCF_TRACE`).
fn trace_iteration(iter: usize, step: &str, diis_size: usize, level_shift: bool) {
    linus_scf::trace::write(&format!(
        "{{\"loop\": \"device\", \"iter\": {iter}, \"step\": \"{step}\", \"diis_size\": {diis_size}, \
         \"level_shift\": {level_shift}}}"
    ));
}

impl dev::FockTerms for DeviceFock<'_> {
    fn coulomb_exchange(
        &mut self,
        gpu: &Gpu,
        st: &dev::FockState<'_>,
        j: &mut DeviceBuf<f64>,
        k: Option<&mut DeviceBuf<f64>>,
    ) -> Result<(), GpuError> {
        self.dev.coulomb_exchange(gpu, st, j, k)?;
        if let Some((fit, hook)) = self.hook.as_mut() {
            fit.feed(gpu, &self.dev.jk, hook.as_mut())?;
        }
        Ok(())
    }

    fn exchange_correlation(
        &mut self,
        gpu: &Gpu,
        st: &dev::FockState<'_>,
        v: &mut DeviceBuf<f64>,
    ) -> Result<Option<f64>, GpuError> {
        self.dev.exchange_correlation(gpu, st, v)
    }

    fn environment(
        &mut self,
        gpu: &Gpu,
        st: &dev::FockState<'_>,
        v: &mut DeviceBuf<f64>,
    ) -> Result<Option<(f64, bool)>, GpuError> {
        match &mut self.env {
            Some(DeviceEnv::Host(e)) => e.environment(gpu, st, v),
            Some(DeviceEnv::Device(t)) => {
                let d = device_env(&mut **t).expect("a device environment term");
                Ok(Some((d.compute_device(gpu, st, v)?, true)))
            }
            None => Ok(None),
        }
    }

    /// The environment term's (the PCM's predicted charges); the device's J/K and XC always
    /// build of the density alone.
    fn built_incrementally(&self) -> bool {
        match &self.env {
            Some(DeviceEnv::Host(e)) => e.terms.term.built_incrementally(),
            Some(DeviceEnv::Device(t)) => t.built_incrementally(),
            None => false,
        }
    }

    fn reset_incremental(&mut self) {
        match &mut self.env {
            Some(DeviceEnv::Host(e)) => e.terms.term.reset_incremental(),
            Some(DeviceEnv::Device(t)) => t.reset_incremental(),
            None => {}
        }
    }
}

/// linus's [`EnvironmentTerm`] on the host copies of the device loop's state.
pub(crate) struct HostEnvironment<'a> {
    pub term: &'a mut dyn EnvironmentTerm,
    pub par: Par,
    /// The term's error, which the device loop's trait cannot carry ([`DeviceEnv::failure`]).
    pub failed: Option<LinusError>,
}

impl dev::HostFockTerms for HostEnvironment<'_> {
    fn coulomb_exchange(
        &mut self,
        _st: &dev::HostState<'_>,
        _j: &mut [f64],
        _k: Option<&mut [f64]>,
    ) -> Result<(), GpuError> {
        Err(GpuError::NotCovered(
            "the environment term has no J/K".into(),
        ))
    }

    fn environment(
        &mut self,
        st: &dev::HostState<'_>,
        v: &mut [f64],
    ) -> Result<Option<(f64, bool)>, GpuError> {
        let n = st.nbf;
        let da = mat_from_colmajor(n, n, st.density(0));
        let db = if st.restricted {
            da.clone()
        } else {
            mat_from_colmajor(n, n, st.density(1))
        };
        let (e, m) = match self
            .term
            .compute(&XcDensity::of(&da, &db, st.restricted), self.par)
        {
            Ok(x) => x,
            Err(e) => {
                let msg = format!("the environment term ({}): {e}", self.term.name());
                self.failed = Some(e);
                return Err(GpuError::Input(msg));
            }
        };
        if let Some(m) = &m {
            v.copy_from_slice(&mat_to_colmajor(m.as_ref()));
        }
        Ok(Some((e, m.is_some())))
    }
}

/// The SCF of `inp` on the device loop with the Fock terms `terms` (`env_name`: the timer of
/// its environment term): finished on the device, or the state for the host loop to continue
/// from (see the module docs).
pub(crate) fn run(
    gpu: &Gpu,
    inp: &ScfInput,
    terms: &mut DeviceFock<'_>,
    opts: &ScfConfig,
    env_name: Option<&'static str>,
    par: Par,
) -> Result<DeviceRun, GpuError> {
    let t_all = std::time::Instant::now();
    let n = inp.h.nrows();
    let n2 = n * n;
    let h = mat_to_colmajor(inp.h);
    let s = mat_to_colmajor(inp.s);
    let guess = inp
        .guess_density
        .as_ref()
        .map(|(a, b)| (mat_to_colmajor(a.as_ref()), mat_to_colmajor(b.as_ref())));
    let a_x = exchange_parts(inp.exchange).0;
    let restricted = opts.restricted.unwrap_or(inp.nalpha == inp.nbeta);
    let dinp = dev::ScfInput {
        nbf: n,
        h: &h,
        s: &s,
        e_nuc: inp.e_nuc,
        nalpha: inp.nalpha,
        nbeta: inp.nbeta,
        restricted,
        exchange_fraction: a_x,
        guess: guess.as_ref().map(|(a, b)| (a.as_slice(), b.as_slice())),
    };
    let mut ctl = DeviceControls::new(opts, restricted);
    let cfg = loop_config(opts, &ctl.controls);
    let mut r = dev::run_scf(gpu, &dinp, terms, &cfg, &mut ctl)?;
    let DeviceControls {
        mut controls,
        before,
        ..
    } = ctl;
    let nmo = r.nmo;
    let nsp = if r.restricted { 1 } else { 2 };
    let (na, nb) = (r.nalpha, r.nbeta);
    let mut timers = Timers::default();
    let seconds =
        |f: fn(&dev::PhaseTimes) -> f64| r.history.iter().map(|h| f(&h.times)).sum::<f64>();
    timers.add("jk", seconds(|t| t.jk));
    timers.add("xc", seconds(|t| t.xc));
    if let Some(name) = env_name {
        timers.add(name, seconds(|t| t.environment));
    }
    // the host loop's names: Fock assembly, gradient and DIIS; level shift, diagonalization
    // and densities
    timers.add("diis", seconds(|t| t.evaluate + t.diis));
    timers.add("diag", seconds(|t| t.diag + t.density));
    timers.add("scf_setup", r.setup_seconds);
    timers.resource(
        "scf_loop_device_mib",
        r.memory.loop_bytes as f64 / 1048576.0,
    );
    let mut history: Vec<ScfIteration> = r.history.iter().map(iteration).collect();
    if linus_scf::trace::enabled() {
        // LINUS_SCF_TRACE: the device loop's DIIS subspace and level shift per iteration
        for h in &r.history {
            trace_iteration(h.iter, h.step.as_str(), h.diis_size, h.level_shift);
        }
    }
    let it = r.iterations;
    let mut stability: Vec<StabilityCheck> = Vec::new();
    let mut converged = r.converged;
    let mut plateau = None;
    let mut plateau_refused = None;
    let mut rots: Option<dev::Rotations> = None;
    // the search at the point the SCF accepts (the zero-mode check's record of a Hessian SCF)
    let mut point_check = None;
    let mut probe_trigger = None;
    // a two-cycle, or the escalation due: the host loop continues from the loop's state
    let mut hand_over = matches!(r.stop, dev::Stop::TwoCycle | dev::Stop::HandOver);
    // at max_iter: the plateau rule over the device's iterations
    if r.stop == dev::Stop::MaxIterations {
        match controls.at_budget(&history) {
            AtBudget::Plateau(p) => {
                // a plateau the stability trigger may analyse: the host loop analyses it before
                // it accepts it, from the loop's state at max_iter
                if r.handover.is_some()
                    && !r.restricted
                    && controls
                        .stability
                        .may_analyse(r.energy - controls.lowest.energy)
                {
                    hand_over = true;
                } else {
                    p.report(&opts.progress);
                    (converged, plateau) = (true, Some(p));
                }
            }
            AtBudget::Refused(f) => plateau_refused = Some(f),
            AtBudget::Stop => {}
        }
    }
    if r.stop == dev::Stop::StabilityCheck {
        // the host loop's decision at its converged point (the stability trigger)
        let ho = r
            .handover
            .as_ref()
            .expect("the loop's state at a stability check");
        let e_above = r.energy - controls.lowest.energy;
        let analysis = controls
            .stability
            .decide(e_above, 0, || gate_gap(&r.eps_a, &r.eps_b, na, nb, false));
        probe_trigger = controls.stability.record;
        match analysis {
            // the full analysis and its steps: on the host
            Some(a) if !a.probing => hand_over = true,
            None => {
                // the loop stopped for the trigger, which did not fire: converged as it is
                if let Some(last) = history.last_mut() {
                    last.step = "converged";
                }
                converged = true;
            }
            Some(a) => {
                let t_st = std::time::Instant::now();
                let mut rt = dev::Rotations::new(gpu, &h, &r.orbitals, nmo, [na, nb], a_x)?;
                let g0 = rt.gradient_at_reference(gpu, &r.fock)?;
                // the first trial direction: towards the lowest-energy iterate, when the point lies
                // above it (the saddle check's)
                let guess = match (&ho.min_orbitals, a.towards_lowest) {
                    (Some(m), true) => {
                        let sd = gpu.upload(&s)?;
                        Some(rt.towards(gpu, &sd, m)?)
                    }
                    _ => None,
                };
                let mut dr = DeviceRotations {
                    gpu,
                    rt,
                    terms: &mut *terms,
                    nmo,
                    nocc: [na, nb],
                    par,
                    restore: Some((&r.densities, &ho.occupied)),
                    failed: None,
                };
                let eps: [&[f64]; 2] = [&r.eps_a, &r.eps_b];
                // one search per irrep of the converged state's symmetry, as on the host
                let group = inp.geometry.as_ref().map(|g| g.group());
                let starts = Starts::new(&mut dr, &eps, group, inp.s);
                let m = lowest_hessian_mode(
                    &mut dr,
                    &starts,
                    Search {
                        products: a.products,
                        g0: Some(&g0),
                        guess,
                        guess_first: a.saddle_fired,
                        irrep: None,
                        stop_below: PROBE_NEGATIVE,
                    },
                );
                let (theta, rnorm, dav_it, nfb) = (m.theta, m.residual, m.iterations, m.builds);
                if let Some(e) = dr.failure() {
                    return Err(e);
                }
                opts.progress.line(format_args!("          stability: lowest Hessian eigenvalue {theta:+.3e} (residual {rnorm:.1e}, {dav_it} Davidson iterations, {nfb} Fock builds), converged {e_above:.2e} Eh above the lowest energy visited"));
                // the Ritz value bounds the lowest eigenvalue from above: a negative probe is a
                // negative mode, which the host loop converges and follows
                if theta >= PROBE_NEGATIVE {
                    // no negative mode: the SCF ends as it would have without the probe. Its builds
                    // left the environment term (and the PCM's Coulomb hook) and VV10's last energy
                    // at a rotated density: one build at the converged density puts what the result
                    // reads back (the device's J/K and semilocal XC keep nothing of the density)
                    let nfb = nfb + usize::from(dr.settle()?);
                    let secs = t_st.elapsed().as_secs_f64();
                    timers.add("stability", secs);
                    let chk = StabilityCheck {
                        iter: it,
                        e_above_min: e_above,
                        eigenvalue: theta,
                        residual: rnorm,
                        davidson_iterations: dav_it,
                        fock_builds: nfb,
                        step: 0.0,
                        de: 0.0,
                        probe: true,
                        irreps: starts.record(m.irrep, m.searched),
                    };
                    point_check = Some(analysis::point_check(&chk, secs));
                    stability.push(chk);
                    if let Some(last) = history.last_mut() {
                        last.step = "converged";
                    }
                    converged = true;
                    rots = Some(dr.rt);
                } else {
                    timers.add("stability", t_st.elapsed().as_secs_f64());
                    hand_over = true;
                }
            }
        }
    }
    if hand_over {
        let ho = r.handover.take().expect("the loop's state at a hand-over");
        let c = ho
            .step_orbitals
            .iter()
            .map(|b| {
                gpu.download(b)
                    .map(|v| mat_from_colmajor(n, nmo, &v[..n * nmo]))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let prev = match &ho.prev_densities {
            Some(b) => {
                let v = gpu.download(b)?;
                Some(
                    (0..nsp)
                        .map(|sp| mat_from_colmajor(n, n, &v[sp * n2..(sp + 1) * n2]))
                        .collect(),
                )
            }
            None => None,
        };
        // iteration `it` is the host loop's to redo, from the controls before it; after a step
        // (the escalation due, max_iter's plateau) the loop's state is that after the last
        // iteration's step, and the host continues with iteration it + 1 (the escalation's first)
        // or at its budget (the plateau's analysis)
        let after_step = matches!(r.stop, dev::Stop::MaxIterations | dev::Stop::HandOver);
        let resumed = if after_step { controls } else { before };
        // the occupied orbitals of the lowest-energy iterate (side by side), when there was one
        let min_occ = match (&ho.min_orbitals, resumed.lowest.iter > 0) {
            (Some(b), true) => {
                let v = gpu.download(b)?;
                Some(if r.restricted {
                    vec![mat_from_colmajor(n, na, &v[..n * na])]
                } else {
                    vec![
                        mat_from_colmajor(n, na, &v[..n * na]),
                        mat_from_colmajor(n, nb, &v[n * na..n * (na + nb)]),
                    ]
                })
            }
            _ => None,
        };
        let resume_at = if after_step {
            it + 1
        } else {
            history.truncate(it - 1);
            it
        };
        timers.add("scf_total", t_all.elapsed().as_secs_f64());
        let resume = ScfResume {
            iter: resume_at,
            c,
            prev,
            e_prev: ho.e_prev,
            controls: resumed,
            min_occ,
            guess_iter: ho.guess_iter,
            history,
        };
        return Ok(DeviceRun::HandOver(Box::new(resume), timers));
    }
    // the zero-mode check of a Hessian SCF: the probe's search at the accepted point (a negative
    // one handed over above); where none ran there (a plateau), the check alone, a record
    let mut mode_check = None;
    if opts.mode_check > 0 && converged && !r.restricted && point_check.is_some() {
        mode_check = point_check;
    } else if opts.mode_check > 0 && converged && !r.restricted {
        let t0 = std::time::Instant::now();
        let mut rt = match rots.take() {
            Some(rt) => rt,
            None => dev::Rotations::new(gpu, &h, &r.orbitals, nmo, [na, nb], a_x)?,
        };
        let g0 = rt.gradient_at_reference(gpu, &r.fock)?;
        // the converged state its builds are put back at: the probe's hand-over state, or the kept
        // occupied orbitals
        let occupied = r
            .occupied
            .as_ref()
            .or(r.handover.as_ref().map(|ho| &ho.occupied));
        let mut dr = DeviceRotations {
            gpu,
            rt,
            terms: &mut *terms,
            nmo,
            nocc: [na, nb],
            par,
            restore: occupied.map(|o| (&r.densities, o)),
            failed: None,
        };
        let eps = [&r.eps_a[..], &r.eps_b[..]];
        let group = inp.geometry.as_ref().map(|g| g.group());
        let starts = Starts::new(&mut dr, &eps, group, inp.s);
        mode_check = Some(analysis::mode_check(
            &mut dr,
            &starts,
            &g0,
            opts,
            t0,
            &mut timers,
        )?);
    }
    drop(rots);
    // the closed-shell probe (D36) at a converged RHF/RKS point below the gap gate, on the device
    let mut closed_shell_probe = None;
    if r.restricted {
        let (mut rec, trig) = ClosedShellProbe::gate_triggered(
            converged,
            &r.eps_a,
            na,
            opts.algo.closed_shell_gap,
            opts.algo.closed_shell_probe.unwrap_or(0),
            opts.probe_trigger.as_ref(),
        );
        probe_trigger = trig;
        if rec.runs() {
            let t0 = std::time::Instant::now();
            let mut rt = dev::Rotations::new(gpu, &h, &r.orbitals, nmo, [na, na], a_x)?;
            let g0 = rt.gradient_at_reference(gpu, &r.fock)?;
            let mut dr = DeviceRotations {
                gpu,
                rt,
                terms: &mut *terms,
                nmo,
                nocc: [na, na],
                par,
                restore: r.occupied.as_ref().map(|o| (&r.densities, o)),
                failed: None,
            };
            let group = inp.geometry.as_ref().map(|g| g.group());
            let starts = Starts::new(&mut dr, &[&r.eps_a, &r.eps_a], group, inp.s);
            analysis::closed_shell_probe(&mut dr, &mut rec, &starts, &g0, opts, t0, &mut timers)?;
        }
        closed_shell_probe = Some(rec);
    }
    // orbitals, densities and Fock matrices to the host, once
    let orbitals: Vec<Mat<f64>> = r
        .orbitals
        .iter()
        .map(|b| {
            gpu.download(b)
                .map(|v| mat_from_colmajor(n, nmo, &v[..n * nmo]))
        })
        .collect::<Result<_, _>>()?;
    let dens = gpu.download(&r.densities)?;
    let fock = gpu.download(&r.fock)?;
    let channel = |v: &[f64], s: usize| mat_from_colmajor(n, n, &v[s * n2..(s + 1) * n2]);
    timers.add("scf_total", t_all.elapsed().as_secs_f64());
    let eps_b = if r.restricted {
        r.eps_a.clone()
    } else {
        r.eps_b.clone()
    };
    let fin = Finished {
        energy: r.energy,
        e_nuc: r.e_nuc,
        parts: EnergyParts {
            one: r.e_one,
            coulomb: r.e_coulomb,
            exchange: r.e_exchange,
            xc: r.e_xc,
            environment: r.e_environment,
        },
        converged,
        restricted: r.restricted,
        nocc: [r.nalpha, r.nbeta],
        eps: [r.eps_a.clone(), eps_b],
        c: [orbitals[0].clone(), orbitals[nsp - 1].clone()],
        d: [channel(&dens, 0), channel(&dens, nsp - 1)],
        f: [channel(&fock, 0), channel(&fock, nsp - 1)],
        s2: r.s2,
        iterations: r.iterations,
        incremental_builds: r.history.iter().filter(|h| h.incremental).count(),
        history,
        stability,
        two_cycle_at: None,
        plateau,
        plateau_refused,
        escalation: controls
            .escalation
            .finish(it, converged, r.stop == dev::Stop::MaxIterations),
        mode_check,
        closed_shell_probe,
        probe_trigger,
        timers,
    };
    Ok(DeviceRun::Done(Box::new(ScfResult::finished(fin, opts))))
}

/// Whether a Fock term of the device loop keeps something of the last state it saw, which a
/// build at the converged state must put back after rotated builds (a probe, the mode check): the
/// environment term (a solvent's surface charges), the PCM's Coulomb fit, VV10's last energy and
/// statistics (read for the report). The device's J/K and semilocal XC keep nothing.
fn keeps_state(terms: &DeviceFock<'_>) -> bool {
    terms.env.is_some() || terms.hook.is_some() || terms.dev.vv10.is_some()
}

/// One device iteration as linus's.
fn iteration(it: &dev::Iteration) -> ScfIteration {
    ScfIteration {
        iter: it.iter,
        energy: it.energy,
        de: it.de,
        gmax: it.gmax,
        grms: it.grms,
        gnorm_pyscf: it.gnorm_pyscf,
        ddmax: it.ddmax,
        ddrms: it.ddrms,
        fock: if it.incremental {
            "incremental"
        } else {
            "full"
        },
        approx: 0,
        step: it.step.as_str(),
        t_jk: it.times.jk,
        t_xc: it.times.xc,
        t_step: it.times.own(),
    }
}

/// The parts of one spin's rotation for the device: kappa and the functions of kappa^T kappa,
/// column-major.
type RotationParts = (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>);

/// The device loop's rotations of its reference orbitals (`dev::Rotations`, the Fock builds on the
/// loop's own engines `terms`): the stability searches' products on the device (the small functions
/// of each rotation on the host). A failed build gives NaN and keeps its error; settling rebuilds at
/// the converged state `restore` (its densities and occupied orbitals) when a term keeps state.
struct DeviceRotations<'a, 'f> {
    gpu: &'a Gpu,
    rt: dev::Rotations,
    terms: &'a mut DeviceFock<'f>,
    nmo: usize,
    nocc: [usize; 2],
    par: Par,
    restore: Option<(&'a DeviceBuf<f64>, &'a DeviceBuf<f64>)>,
    failed: Option<GpuError>,
}

impl RotatedGradient for DeviceRotations<'_, '_> {
    type Error = GpuError;

    fn nmo(&self) -> usize {
        self.nmo
    }

    fn nocc(&self) -> [usize; 2] {
        self.nocc
    }

    fn orbital(&mut self, s: usize, p: usize) -> Vec<f64> {
        match self.rt.orbital(self.gpu, s, p) {
            Ok(c) => c,
            Err(e) => {
                self.failed.get_or_insert(e);
                Vec::new()
            }
        }
    }

    fn gradient(&mut self, v: &[f64], h: f64) -> Vec<f64> {
        let len = self.rt.len();
        if self.failed.is_some() {
            return vec![f64::NAN; len];
        }
        let k = unflatten(v, self.nmo, &self.nocc, h);
        let parts: Vec<RotationParts> = k
            .iter()
            .zip(self.nocc)
            .map(|(ks, no)| {
                let (cos, sinc, cosm1) = rotation_functions(ks, no, self.par);
                (
                    mat_to_colmajor(ks.as_ref()),
                    mat_to_colmajor(cos.as_ref()),
                    mat_to_colmajor(sinc.as_ref()),
                    mat_to_colmajor(cosm1.as_ref()),
                )
            })
            .collect();
        fn rot(p: &RotationParts) -> dev::Rotation<'_> {
            dev::Rotation {
                kappa: &p.0,
                cos: &p.1,
                sinc: &p.2,
                cosm1: &p.3,
            }
        }
        let terms: &mut dyn dev::FockTerms = &mut *self.terms;
        match self
            .rt
            .gradient(self.gpu, terms, &[rot(&parts[0]), rot(&parts[1])])
        {
            Ok(g) => g,
            Err(e) => {
                self.failed = Some(e);
                vec![f64::NAN; len]
            }
        }
    }

    fn failure(&mut self) -> Option<GpuError> {
        self.failed.take()
    }

    fn settle(&mut self) -> Result<bool, GpuError> {
        let keeps = keeps_state(self.terms);
        if keeps {
            let (d, o) = self.restore.ok_or_else(|| {
                GpuError::Contract("the device loop kept no occupied orbitals".into())
            })?;
            let terms: &mut dyn dev::FockTerms = &mut *self.terms;
            self.rt.rebuild(self.gpu, terms, d, o)?;
        }
        Ok(keeps)
    }
}
