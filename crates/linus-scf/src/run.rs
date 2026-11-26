//! The host SCF loop: evaluate → observe → decide → step.
//!
//! Each iteration evaluates the Fock matrices at the current state (`FockEvaluator`; the J/K
//! builder moves to its next approximation when the controls ask), reports the iteration's scalars
//! to the controls ([`Controls::observe`]), and acts on their decision: a step to new orbitals by
//! the accelerators (`Accelerators::step`), a confirming full build, a restart, or the end of the
//! SCF, after a stability analysis of a converged UHF/UKS point if the trigger asks for one
//! (`analysis::analyse`). When the escalation comes due (once, when DIIS stagnates or at 2/3 of
//! `max_iter`) and at the end of the budget, the plateau rule decides first: a plateau is accepted
//! (after the same analysis; on one more full build before the end of the budget), and a negative
//! mode is followed as the escalation, or refuses the plateau at the end; otherwise the escalation
//! starts, or the SCF stops at the end of the budget.
//! A caller's stop request ends the loop after an iteration's step
//! ([`linus_base::progress::Progress::stop_due`]): the SCF is unconverged, its last iteration's
//! step `stopped` ([`ScfResult::stopped`]).
//! After the loop come the final orbitals, the zero-mode check and the closed-shell probe.

use super::accel::{Accelerators, FockStep, Restart};
use super::analysis::{self, host_rotations, AnalysisInput, Outcome};
use super::control::{AtBudget, Controls, Decision, Observation};
use super::fock::{EnergyParts, FockEvaluator, FockTerms};
use super::result::Finished;
use super::stability::Starts;
use super::state::OrbitalState;
use super::{
    all_finite, density, diagonalize, gate_gap, trace, try_diagonalize, try_semicanonical,
    ClosedShellProbe, ModeCheck, PlateauRefusal, ScfConfig, ScfInput, ScfIteration, ScfResult,
    ScfResume, StabilityCheck,
};
use faer::{Mat, MatRef, Par};
use linus_base::error::LinusError;
use linus_base::linalg::{canonical_orthogonalizer, dot, mm};
use linus_core::timing::Timers;
use std::time::Instant;

/// The SCF of `inp` with the Fock terms `terms` from the start, or from `resume`.
pub(super) fn run(
    inp: &ScfInput,
    terms: FockTerms<'_>,
    opts: &ScfConfig,
    par: Par,
    resume: Option<ScfResume>,
) -> Result<ScfResult, LinusError> {
    let mut timers = Timers::default();
    let t_all = Instant::now();
    // faer's sequential matmul rounds differently from its parallel one for some shapes, while
    // the parallel results do not depend on the degree (caffeine RHF: 8 and 32 threads bitwise
    // equal, 1 thread not); a 1-thread SCF therefore runs the parallel code path too (in a
    // 1-thread pool it executes sequentially).
    let par = match par {
        Par::Seq => Par::rayon(2),
        p => p,
    };
    let (na, nb) = (inp.nalpha, inp.nbeta);
    let restricted = opts.restricted.unwrap_or(na == nb);
    assert!(
        !restricted || na == nb,
        "restricted SCF needs a closed shell (use ROHF/UHF)"
    );
    let (h, s) = (inp.h, inp.s);
    let nbf = h.nrows();
    let x = timers.time("orthogonalizer", || {
        canonical_orthogonalizer(s, opts.algo.lindep, par)
    });
    if x.ncols() < na.max(nb) {
        return Err(LinusError::Input(format!(
            "scf.lindep {:e} drops the basis to {} orbitals, fewer than the {} occupied",
            opts.algo.lindep,
            x.ncols(),
            na.max(nb)
        )));
    }
    let mut state = start(inp, resume.as_ref(), x.as_ref(), restricted, par);
    // the iteration whose density came from a guess (not from orbitals): kept out of DIIS
    let guess_iter = match &resume {
        Some(r) => r.guess_iter,
        None if state.occ.is_none() => 1,
        None => 0,
    };
    let mut fock = FockEvaluator::new(terms, inp, x.as_ref(), restricted, &opts.algo, par);
    let mut controls = Controls::new(opts, restricted);
    let mut accel = Accelerators::new(&opts.algo, s, x.as_ref(), par, opts.progress.clone());
    let mut history: Vec<ScfIteration> = Vec::new();
    let mut iters = 0;
    let mut e_old = 0.0;
    // the occupied orbitals of the lowest-energy iterate (Controls::keeps_lowest_orbitals)
    let mut lowest_occ: Option<Vec<Mat<f64>>> = None;
    if let Some(r) = resume {
        iters = r.iter.saturating_sub(1);
        e_old = r.e_prev;
        fock.last_dens = r.prev;
        controls = r.controls;
        lowest_occ = r.min_occ;
        history = r.history;
    }
    let mut energy = 0.0;
    let mut parts = EnergyParts::default();
    let mut fa = Mat::<f64>::zeros(nbf, nbf);
    let mut fb = Mat::<f64>::zeros(nbf, nbf);
    // this run has evaluated the Fock matrices (a loop resumed at its budget has not yet)
    let mut evaluated = false;
    let mut stability: Vec<StabilityCheck> = Vec::new();
    // the search at the point the SCF accepts (the zero-mode check's record of a Hessian SCF)
    let mut point_check: Option<ModeCheck> = None;
    let mut converged = false;
    let mut plateau = None;
    let mut plateau_refused = None;
    // a plateau accepted when the escalation came due, on the next (full) build
    let mut accept: Option<super::Plateau> = None;
    // a non-finite energy or Fock matrix (or a failed diagonalization) stops the SCF: the result
    // is unconverged with a NaN energy (the driver's Numerical error)
    let mut non_finite = false;
    // the next build is a full one at the loop's request (a confirmation, a restart)
    let mut rebuild = false;
    // LINUS_SCF_TRACE (scf/trace.rs): the entries traced so far (a resumed history is the device
    // loop's, which traces its own); each new entry's DIIS, ADIIS and SOSCF state is written at the
    // top of the next iteration, before anything changes it, and after the loop
    let tracing = trace::enabled();
    let mut traced = history.len();
    loop {
        if tracing {
            traced = accel.trace(&history, traced);
        }
        // the plateau rule decides at the end of the budget, and when the escalation comes due (never
        // between an incremental build that met the criteria and its confirmation): a plateau is
        // accepted there rather than escalated, whose quasi-Newton steps along a soft mode would
        // leave it
        let at_budget = controls.escalation.exhausted(iters);
        let due = if at_budget || rebuild {
            None
        } else {
            controls.escalation_due(iters)
        };
        let mut escalate = due;
        let verdict = (at_budget || due.is_some()).then(|| controls.at_budget(&history));
        match verdict {
            Some(AtBudget::Plateau(p)) => {
                // a plateau is the point the SCF would accept: the stability trigger's analysis
                // first. A negative mode refuses it at the end of the budget and is followed as
                // the escalation before it; a minimum or no analysis accepts it
                let wants = iters > 0
                    && !restricted
                    && controls
                        .stability
                        .may_analyse(energy - controls.lowest.energy);
                if wants && !evaluated {
                    // resumed at the budget (the device loop's plateau): the Fock matrices of the
                    // state it handed over
                    let ev = fock.evaluate(&state, true, &mut timers)?;
                    (energy, parts) = (ev.energy, ev.parts);
                    (fa, fb) = (ev.fa, ev.fb);
                }
                let e_above = energy - controls.lowest.energy;
                let analysis = wants
                    .then(|| {
                        let x = x.as_ref();
                        controls.stability.decide(e_above, stability.len(), || {
                            let ga = diagonalize(fa.as_ref(), x, par).0;
                            let gb = diagonalize(fb.as_ref(), x, par).0;
                            gate_gap(&ga, &gb, na, nb, false)
                        })
                    })
                    .flatten();
                let mut rotated = None;
                if let Some(a) = analysis {
                    let t_st = Instant::now();
                    let ainp = AnalysisInput {
                        iter: iters,
                        energy,
                        fa: &fa,
                        fb: &fb,
                        lowest: lowest_occ.as_ref(),
                        group: inp.geometry.as_ref().map(|g| g.group()),
                        progress: &opts.progress,
                    };
                    let (outcome, chk) = analysis::analyse(&mut fock, &state, &a, &ainp)?;
                    let secs = t_st.elapsed().as_secs_f64();
                    timers.add("stability", secs);
                    point_check = Some(analysis::point_check(&chk, secs));
                    stability.push(chk);
                    match outcome {
                        Outcome::Step(cr) if !at_budget => rotated = Some(cr),
                        Outcome::Step(_) => {
                            // the analysis' trial builds left the terms at a rotated density,
                            // which one build puts back
                            fock.settle(&state)?;
                            rotated = Some(Vec::new());
                        }
                        Outcome::Minimum => {
                            controls.stability.stable = true;
                            fock.settle(&state)?;
                        }
                        Outcome::ProbeMinimum => {}
                    }
                }
                match (rotated, at_budget) {
                    (Some(_), true) => {
                        // a negative mode with no iterations left to follow it
                        point_check = None;
                        plateau_refused = Some(PlateauRefusal {
                            plateau: p,
                            reason: "negative_mode",
                            gnorm_pyscf: history.last().map_or(f64::NAN, |h| h.gnorm_pyscf),
                            limit: None,
                        });
                        break;
                    }
                    (Some(cr), false) => {
                        // the escalation, from the orbitals rotated along the negative mode
                        let [ca, cb]: [Mat<f64>; 2] =
                            cr.try_into().expect("orbitals of both spins");
                        state.set_orbitals(ca, cb, par);
                        accel.restart(Restart::Escalation);
                        rebuild = true;
                        if let Some(last) = history.last_mut() {
                            last.step = "stability";
                        }
                        let trigger = due.expect("before the budget's end: the escalation is due");
                        controls
                            .escalation
                            .start(iters, trigger, iters, energy, history.len());
                        point_check = None;
                        escalate = None;
                    }
                    (None, true) => {
                        // accepted on the budget's last iteration, a full build
                        p.report(&opts.progress);
                        converged = true;
                        plateau = Some(p);
                        break;
                    }
                    (None, false) => {
                        // accepted on one full build at the state the last step reached
                        accept = Some(p);
                        rebuild = true;
                        escalate = None;
                    }
                }
            }
            // the accuracy targets refuse the plateau: the SCF fails at the end of the budget and
            // escalates before it
            Some(AtBudget::Refused(r)) if at_budget => {
                plateau_refused = Some(r);
                break;
            }
            Some(AtBudget::Stop) if at_budget => break,
            _ => {}
        }
        if let Some(trigger) = escalate {
            // once, from the occupied orbitals of the lowest-energy iteration (else the current
            // ones)
            let (from, from_iter, from_energy) = match lowest_occ.clone() {
                Some(o) => (o, controls.lowest.iter, controls.lowest.energy),
                None => match state.occ.clone() {
                    Some(o) => (o, iters, energy),
                    None => break,
                },
            };
            state.set_occupied(from, par);
            accel.restart(Restart::Escalation);
            rebuild = true;
            if let Some(last) = history.last_mut() {
                last.step = "escalation";
            }
            controls
                .escalation
                .start(iters, trigger, from_iter, from_energy, history.len());
            opts.progress.line(format_args!("          escalation ({}): restarting from iteration {from_iter} (E = {from_energy:.12}) with quasi-Newton steps", trigger.name()));
            opts.progress
                .event(linus_base::progress::Event::ScfEscalation {
                    iteration: iters,
                    trigger: trigger.name(),
                    from_iteration: from_iter,
                });
        }
        iters += 1;
        let it = iters;
        // evaluate; the budget's last iteration is a full build, so the SCF ends on one wherever
        // it ends (a convergence or a plateau at max_iter needs no confirming build)
        let rebuilt = std::mem::take(&mut rebuild);
        let last = controls.escalation.exhausted(it);
        let mut ev = fock.evaluate(&state, rebuilt || last, &mut timers)?;
        evaluated = true;
        // the builder's next approximation (COSX: the next SCF grid) once the orbital gradient is
        // small, and always before converging: K at the same density on it, the accelerators'
        // stored Fock matrices shifted by the change
        let switched = fock.may_switch()
            && controls.wants_next_approximation(ev.gmax, ev.gnorm_pyscf)
            && match fock.next_approximation(&state, &mut ev, &mut timers)? {
                Some(dk) => {
                    accel.operator_changed(&dk, &mut timers);
                    true
                }
                None => false,
            };
        energy = ev.energy;
        parts = ev.parts;
        non_finite = !(energy.is_finite() && all_finite(&ev.fa) && all_finite(&ev.fb));
        let de = energy - e_old;
        let fock_kind = if ev.incremental {
            "incremental"
        } else {
            "full"
        };
        if opts.progress.on() {
            let sw = if switched { "  (next grid)" } else { "" };
            opts.progress.line(format_args!(
                "iter {it:3}  E = {energy:.12}  dE = {de:+.3e}  max|[F,D]| = {:.3e}  rms {:.2e}  |dD| {:.2e}  {fock_kind}{sw}",
                ev.gmax, ev.grms, ev.ddmax
            ));
        }
        opts.progress
            .event(linus_base::progress::Event::ScfIteration {
                iteration: it,
                energy,
                delta_energy: de,
                gradient: ev.gmax,
            });
        let mut rec = ScfIteration {
            iter: it,
            energy: if non_finite { f64::NAN } else { energy },
            de,
            gmax: ev.gmax,
            grms: ev.grms,
            gnorm_pyscf: ev.gnorm_pyscf,
            ddmax: ev.ddmax,
            ddrms: ev.ddrms,
            fock: fock_kind,
            step: "converged",
            approx: fock.approx,
            t_jk: ev.t_jk,
            t_xc: ev.t_xc,
            t_step: 0.0,
        };
        let t_step0 = ev.assembled_at;
        if non_finite {
            rec.step = "non_finite";
            history.push(rec);
            (fa, fb) = (ev.fa, ev.fb);
            break;
        }
        // observe
        let previous = [
            history.last().map(|r| r.energy),
            history.len().checked_sub(2).map(|i| history[i].energy),
        ];
        let (decision, new_lowest) = controls.observe(&Observation {
            iter: it,
            guess_iter,
            energy,
            de,
            gmax: ev.gmax,
            gnorm_pyscf: ev.gnorm_pyscf,
            previous,
            rebuilt,
            incremental: ev.incremental,
            switched,
            descending: accel.descending(),
        });
        if new_lowest && controls.keeps_lowest_orbitals() {
            lowest_occ = state.occ.clone();
        }
        let gmax = ev.gmax;
        let (ea, eb) = (ev.ea, ev.eb);
        (fa, fb) = (ev.fa, ev.fb);
        // a plateau accepted when the escalation came due ends on this full build
        if let Some(p) = accept.take() {
            p.report(&opts.progress);
            converged = true;
            plateau = Some(p);
            rec.t_step = t_step0.elapsed().as_secs_f64();
            history.push(rec);
            break;
        }
        // decide
        match decision {
            Decision::Confirm => {
                // confirm with a full build at the same density
                rebuild = true;
                e_old = energy;
                accel.forget_last_step();
                rec.step = "rebuild";
                rec.t_step = t_step0.elapsed().as_secs_f64();
                history.push(rec);
                continue;
            }
            Decision::Converged { e_above } => {
                let x = x.as_ref();
                let analysis = controls.stability.decide(e_above, stability.len(), || {
                    let ga = diagonalize(fa.as_ref(), x, par).0;
                    let gb = diagonalize(fb.as_ref(), x, par).0;
                    gate_gap(&ga, &gb, na, nb, false)
                });
                if let Some(a) = analysis {
                    let t_st = Instant::now();
                    let inp = AnalysisInput {
                        iter: it,
                        energy,
                        fa: &fa,
                        fb: &fb,
                        lowest: lowest_occ.as_ref(),
                        group: inp.geometry.as_ref().map(|g| g.group()),
                        progress: &opts.progress,
                    };
                    let (outcome, chk) = analysis::analyse(&mut fock, &state, &a, &inp)?;
                    let secs = t_st.elapsed().as_secs_f64();
                    timers.add("stability", secs);
                    point_check = match outcome {
                        Outcome::Step(_) => None,
                        _ => Some(analysis::point_check(&chk, secs)),
                    };
                    stability.push(chk);
                    match outcome {
                        Outcome::ProbeMinimum => {
                            // the SCF ends as it would have without the probe
                            converged = true;
                            history.push(rec);
                            break;
                        }
                        Outcome::Step(cr) => {
                            // restart from the rotated orbitals with a new DIIS subspace
                            let [ca, cb]: [Mat<f64>; 2] =
                                cr.try_into().expect("orbitals of both spins");
                            state.set_orbitals(ca, cb, par);
                            accel.restart(Restart::Descent);
                            rec.step = "stability";
                        }
                        Outcome::Minimum => {
                            // a minimum: rebuild at the converged density (the analysis' builds
                            // changed the builders' state) and stop there
                            controls.stability.stable = true;
                            rec.step = "rebuild";
                        }
                    }
                    rebuild = true;
                    e_old = energy;
                    rec.t_step = t_st.elapsed().as_secs_f64();
                    history.push(rec);
                    continue;
                }
                converged = true;
                history.push(rec);
                break;
            }
            Decision::TwoCycle => {
                // occupations flip between two determinants and DIIS never ends it: quasi-Newton
                // steps from the current orbitals, a new DIIS subspace
                opts.progress.line(format_args!("          two-cycle: the energy repeats every other iteration; SOSCF steps from here"));
                accel.restart(Restart::Descent);
            }
            Decision::Continue => {}
        }
        // step
        e_old = energy;
        let step = accel.step(
            FockStep {
                iter: it,
                usable: it > guess_iter,
                keep_occ: fock.may_switch(),
                fa: &fa,
                fb: &fb,
                ea,
                eb,
                gmax,
            },
            &mut state,
            &mut timers,
        );
        rec.t_step = t_step0.elapsed().as_secs_f64();
        match step {
            Ok(name) => rec.step = name,
            Err(name) => {
                // the extrapolated Fock matrix is not finite
                rec.step = name;
                history.push(rec);
                non_finite = true;
                break;
            }
        }
        history.push(rec);
        // the caller asks to stop: the SCF ends unconverged at this iteration
        if opts.progress.stop_due() {
            if let Some(last) = history.last_mut() {
                last.step = "stopped";
            }
            break;
        }
    }
    if tracing {
        accel.trace(&history, traced);
    }
    let escalation = controls.escalation.finish(
        iters,
        converged,
        !non_finite && controls.escalation.exhausted(iters),
    );
    // final orbitals: the canonical orbitals of the SCF's occupied space in the converged Fock
    // matrices (without extrapolation)
    let fin_a = (!non_finite)
        .then(|| final_orbitals(&state, 0, &fa, x.as_ref(), s, par))
        .flatten();
    let fin_b = if restricted {
        fin_a.clone()
    } else {
        (!non_finite)
            .then(|| final_orbitals(&state, 1, &fb, x.as_ref(), s, par))
            .flatten()
    };
    let (eps_a, c_a, eps_b, c_b) = match (fin_a, fin_b) {
        (Some((ea, ca)), Some((eb, cb))) => (ea, ca, eb, cb),
        _ => {
            // a numerical failure: NaN energy and orbital energies (orbitals of the last step)
            converged = false;
            energy = f64::NAN;
            let nmo = x.ncols();
            let pad = |c: &Mat<f64>| {
                Mat::from_fn(
                    nbf,
                    nmo,
                    |i, j| if j < c.ncols() { c[(i, j)] } else { f64::NAN },
                )
            };
            (
                vec![f64::NAN; nmo],
                pad(&state.c_a),
                vec![f64::NAN; nmo],
                pad(&state.c_b),
            )
        }
    };
    let sz = 0.5 * (na as f64 - nb as f64);
    let s2 = if restricted {
        sz * (sz + 1.0)
    } else {
        sz * (sz + 1.0) + nb as f64
            - dot(
                mm(mm(s, state.da.as_ref(), par).as_ref(), s, par).as_ref(),
                state.db.as_ref(),
            )
    };
    // the zero-mode check of a Hessian SCF: the search at the accepted point (the trigger's probe
    // with the check's products, or the full analysis that found it a minimum); where none ran
    // there (`scf.stability_max` reached, a plateau), the check alone, a record
    let mut mode_check = None;
    if opts.mode_check > 0 && converged && !restricted && point_check.is_some() {
        mode_check = point_check;
    } else if opts.mode_check > 0 && converged && !restricted {
        let t0 = Instant::now();
        let cs = vec![c_a.clone(), c_b.clone()];
        let (mut rot, g0) = host_rotations(&mut fock, &state, cs, [&fa, &fb], state.nocc);
        let group = inp.geometry.as_ref().map(|g| g.group());
        let starts = Starts::new(&mut rot, &[&eps_a[..], &eps_b[..]], group, s);
        mode_check = Some(analysis::mode_check(
            &mut rot,
            &starts,
            &g0,
            opts,
            t0,
            &mut timers,
        )?);
    }
    // the closed-shell probe (D36): at a converged RHF/RKS point below the gap gate, one-sided
    // products of the UHF/UKS orbital Hessian; a record only, nothing of the result depends on it
    let mut probe_trigger = controls.stability.record;
    let mut closed_shell_probe = None;
    if restricted {
        let (mut rec, trig) = ClosedShellProbe::gate_triggered(
            converged,
            &eps_a,
            na,
            opts.algo.closed_shell_gap,
            opts.algo.closed_shell_probe.unwrap_or(0),
            opts.probe_trigger.as_ref(),
        );
        probe_trigger = trig;
        if rec.runs() {
            let t0 = Instant::now();
            let cs = vec![c_a.clone(), c_a.clone()];
            let (mut rot, g0) = host_rotations(&mut fock, &state, cs, [&fa, &fa], [na, na]);
            let group = inp.geometry.as_ref().map(|g| g.group());
            let starts = Starts::new(&mut rot, &[&eps_a[..], &eps_a[..]], group, s);
            analysis::closed_shell_probe(&mut rot, &mut rec, &starts, &g0, opts, t0, &mut timers)?;
        }
        closed_shell_probe = Some(rec);
    }
    let incremental_builds = fock.incremental_builds;
    drop(fock);
    timers.add("scf_total", t_all.elapsed().as_secs_f64());
    let fin = Finished {
        energy,
        e_nuc: inp.e_nuc,
        parts,
        converged,
        restricted,
        nocc: [na, nb],
        eps: [eps_a, eps_b],
        c: [c_a, c_b],
        d: [state.da, state.db],
        f: [fa, fb],
        s2,
        iterations: iters,
        incremental_builds,
        history,
        stability,
        two_cycle_at: controls.two_cycle.at,
        plateau,
        plateau_refused,
        escalation,
        mode_check,
        closed_shell_probe,
        probe_trigger,
        timers,
    };
    Ok(ScfResult::finished(fin, opts))
}

/// The final orbitals and orbital energies of spin channel `ch` (0 alpha, 1 beta) of the SCF at
/// `state` with its last Fock matrix `f`: the canonical orbitals of the occupied space of the
/// density `f` was built from, occupied first. Where that occupied set came from a diagonalization
/// (`OrbitalState::aufbau`) these are the eigenvectors of `f` with aufbau occupations; after
/// rotations they are its semicanonical orbitals ([`try_semicanonical`]), which keep the density
/// whether or not its occupied set is the aufbau set of `f` (docs/internal/deggrad.md: the
/// eigenvectors' aufbau set was a different determinant after the escalation, and the
/// derivatives used it). None when `f` has non-finite elements.
fn final_orbitals(
    state: &OrbitalState,
    ch: usize,
    f: &Mat<f64>,
    x: MatRef<'_, f64>,
    s: MatRef<'_, f64>,
    par: Par,
) -> Option<(Vec<f64>, Mat<f64>)> {
    match &state.occ {
        Some(occ) if !state.aufbau => try_semicanonical(&occ[ch], f, x, s, par),
        _ => try_diagonalize(f.as_ref(), x, par),
    }
}

/// The state the SCF starts from: the resumed iteration's orbitals, the guess densities of `inp`,
/// or the core Hamiltonian's orbitals (whose densities alone it keeps, as the start always did).
fn start(
    inp: &ScfInput,
    resume: Option<&ScfResume>,
    x: MatRef<'_, f64>,
    restricted: bool,
    par: Par,
) -> OrbitalState {
    let (na, nb) = (inp.nalpha, inp.nbeta);
    let nbf = inp.h.nrows();
    let nocc = [na, nb];
    match (resume, &inp.guess_density) {
        (Some(r), _) => {
            // the orbitals of the resumed iteration's density: a device iteration's
            // diagonalization (the device loop takes no quasi-Newton or stability steps; it hands
            // those over), occupied by aufbau
            let (ca, cb) = (&r.c[0], &r.c[r.c.len() - 1]);
            let (da, db) = (density(ca.as_ref(), na, par), density(cb.as_ref(), nb, par));
            let mut st = OrbitalState::new(restricted, nocc, nbf, da, db);
            st.occ = Some(st.occupied_of(ca, cb));
            (st.c_a, st.c_b) = (ca.clone(), cb.clone());
            st.aufbau = true;
            st
        }
        (None, Some((a, b))) => OrbitalState::new(restricted, nocc, nbf, a.clone(), b.clone()),
        (None, None) => {
            let (_, c) = try_diagonalize(inp.h, x, par)
                .unwrap_or_else(|| (vec![f64::NAN; nbf], Mat::from_fn(nbf, nbf, |_, _| f64::NAN)));
            let (da, db) = (density(c.as_ref(), na, par), density(c.as_ref(), nb, par));
            let mut st = OrbitalState::new(restricted, nocc, nbf, da, db);
            st.occ = Some(st.occupied_of(&c, &c));
            st.aufbau = true;
            st
        }
    }
}
