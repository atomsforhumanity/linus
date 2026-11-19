//! A calculation's progress, for whoever follows it while it runs: its progress lines
//! (`output.verbose`: the SCF's iterations, its escalations and two-cycle breaks, the stability
//! analyses, the plateau rule), its progress events (an in-process caller's listener: qc*'s
//! `ctx.progress`), and the follower's request that it stop (qc*'s `ctx.stop_requested()`).
//!
//! The front end decides where the lines go: the command line writes them to stderr as they come
//! ([`Progress::stderr`]); an in-process caller (the Python package) collects them
//! ([`Progress::collect`]) and returns a request's lines in the output's `logs`, because a
//! program does not print in its caller's process (qc* decision 24).
//!
//! Events ([`Event`]) are small records of what happened, handed to a listener
//! ([`Progress::with_events`]) as they happen: a stage's start and end, each SCF iteration and
//! escalation, each response iteration. They are linus's documented set (docs/python.md, "Progress events"); the
//! Python package turns each into a JSON object with its `event` name for the runner's
//! `ctx.progress`. Nothing in a result depends on them, and without a listener an event costs one
//! branch.
//!
//! A stop request ([`Progress::with_stop`]: the caller's check, asked on the calculation's
//! threads) ends a calculation at its next step with what it has reached
//! ([`crate::error::LinusError::Stopped`]): before each stage ([`Progress::stage`]), and between
//! the SCF's and the response solves' iterations ([`Progress::stop_due`], which asks at most every
//! [`STOP_INTERVAL`], so that a check that takes the caller's lock, Python's GIL, costs nothing
//! measurable in a fast SCF). Without a check, a stop costs one branch.
//!
//! A [`Progress`] is cheap to clone (the collected lines, the listener and the stop check are
//! shared), so the SCF's settings, its accelerators and its analyses each hold one. Off,
//! [`Progress::line`] formats nothing.

use crate::error::LinusError;
use std::fmt;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Where a calculation's progress lines and events go: nowhere (the default), stderr or a shared
/// buffer for the lines; a listener for the events; and whether its caller asks it to stop.
#[derive(Clone, Default)]
pub struct Progress {
    lines: Option<Sink>,
    events: Option<Listener>,
    stop: Option<Arc<Stop>>,
}

/// What receives a calculation's progress events: called on the calculation's own threads, while
/// it runs, so it returns quickly and never calls back into the calculation.
pub type Listener = Arc<dyn Fn(Event<'_>) + Send + Sync>;

/// Whether the caller asks the calculation to stop (an in-process caller's `ctx.stop_requested()`):
/// called on the calculation's threads, between its steps.
pub type StopCheck = Arc<dyn Fn() -> bool + Send + Sync>;

/// The shortest time between two asks of a [`StopCheck`] between iterations
/// ([`Progress::stop_due`]); stages always ask.
pub const STOP_INTERVAL: Duration = Duration::from_millis(50);

/// A caller's stop check and what it said: once it asks to stop, the calculation stops.
struct Stop {
    check: StopCheck,
    since: Instant,
    /// Nanoseconds after `since` of the last ask (`u64::MAX`: never asked).
    asked: AtomicU64,
    requested: AtomicBool,
}

impl Stop {
    /// Ask the caller (now).
    fn ask(&self) -> bool {
        if self.requested.load(Ordering::Relaxed) {
            return true;
        }
        self.asked
            .store(self.since.elapsed().as_nanos() as u64, Ordering::Relaxed);
        let stop = (self.check)();
        if stop {
            self.requested.store(true, Ordering::Relaxed);
        }
        stop
    }

    /// Ask the caller unless it was asked within [`STOP_INTERVAL`].
    fn due(&self) -> bool {
        if self.requested.load(Ordering::Relaxed) {
            return true;
        }
        let asked = self.asked.load(Ordering::Relaxed);
        if asked != u64::MAX {
            let now = self.since.elapsed().as_nanos() as u64;
            if now.saturating_sub(asked) < STOP_INTERVAL.as_nanos() as u64 {
                return false;
            }
        }
        self.ask()
    }
}

#[derive(Clone, Debug)]
enum Sink {
    Stderr,
    Lines(Arc<Mutex<String>>),
}

impl fmt::Debug for Progress {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Progress")
            .field("lines", &self.lines)
            .field("events", &self.events.is_some())
            .field("stop", &self.stop.is_some())
            .finish()
    }
}

/// One progress event: what a calculation did, as it does it. The names and fields are linus's
/// documented set; each becomes a JSON object `{"event": name, field: value, ...}` (qcdata's
/// units: Eh, Eh/bohr, s).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Event<'a> {
    /// `stage_start`: a stage of the calculation started: `scf`, `gradient`, `hessian` (its
    /// response and assembly), `polarizability` (an IR run's electric-field response).
    StageStart { stage: &'a str },
    /// `stage_end`: the stage ended after `seconds`, with `success` false when it failed (an SCF
    /// that did not converge, an error).
    StageEnd {
        stage: &'a str,
        seconds: f64,
        success: bool,
    },
    /// `scf_iteration`: an SCF iteration's Fock build: its energy (Eh), the change from the
    /// iteration before (`delta_energy`) and the orbital gradient (`gradient`: max |F_ai|).
    ScfIteration {
        iteration: usize,
        energy: f64,
        delta_energy: f64,
        gradient: f64,
    },
    /// `scf_escalation`: the SCF restarts after `iteration` from the orbitals of `from_iteration`
    /// (its lowest energy) with quasi-Newton steps, within its budget; `trigger` is
    /// `stagnation` (the lowest energy stopped improving) or `two_thirds` (of `scf.max_iter`).
    ScfEscalation {
        iteration: usize,
        trigger: &'a str,
        from_iteration: usize,
    },
    /// `response_iteration`: an iteration of a response solve (CPHF/CPKS) of `solve`
    /// (`hessian`, `polarizability`): the largest residual left, and the right-hand sides not yet
    /// converged (`unconverged`) of `total`.
    ResponseIteration {
        solve: &'a str,
        iteration: usize,
        residual: f64,
        unconverged: usize,
        total: usize,
    },
}

/// A field's value of an [`Event`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Value<'a> {
    Str(&'a str),
    /// A number; a non-finite one stands for no value (JSON null).
    Float(f64),
    Int(usize),
    Bool(bool),
}

impl<'a> Event<'a> {
    /// The event's name (the JSON object's `event`).
    pub fn name(&self) -> &'static str {
        match self {
            Event::StageStart { .. } => "stage_start",
            Event::StageEnd { .. } => "stage_end",
            Event::ScfIteration { .. } => "scf_iteration",
            Event::ScfEscalation { .. } => "scf_escalation",
            Event::ResponseIteration { .. } => "response_iteration",
        }
    }

    /// The event's fields, in order.
    pub fn fields(&self) -> Vec<(&'static str, Value<'a>)> {
        use Value::{Bool, Float, Int, Str};
        match *self {
            Event::StageStart { stage } => vec![("stage", Str(stage))],
            Event::StageEnd {
                stage,
                seconds,
                success,
            } => vec![
                ("stage", Str(stage)),
                ("seconds", Float(seconds)),
                ("success", Bool(success)),
            ],
            Event::ScfIteration {
                iteration,
                energy,
                delta_energy,
                gradient,
            } => vec![
                ("iteration", Int(iteration)),
                ("energy", Float(energy)),
                ("delta_energy", Float(delta_energy)),
                ("gradient", Float(gradient)),
            ],
            Event::ScfEscalation {
                iteration,
                trigger,
                from_iteration,
            } => vec![
                ("iteration", Int(iteration)),
                ("trigger", Str(trigger)),
                ("from_iteration", Int(from_iteration)),
            ],
            Event::ResponseIteration {
                solve,
                iteration,
                residual,
                unconverged,
                total,
            } => vec![
                ("solve", Str(solve)),
                ("iteration", Int(iteration)),
                ("residual", Float(residual)),
                ("unconverged", Int(unconverged)),
                ("total", Int(total)),
            ],
        }
    }
}

impl Progress {
    /// No progress lines, no events.
    pub fn off() -> Progress {
        Progress::default()
    }

    /// Each line to stderr as it comes (the command line's `--verbose`).
    pub fn stderr() -> Progress {
        Progress {
            lines: Some(Sink::Stderr),
            ..Progress::default()
        }
    }

    /// The lines collected in memory, for [`Progress::take`].
    pub fn collect() -> Progress {
        Progress {
            lines: Some(Sink::Lines(Arc::default())),
            ..Progress::default()
        }
    }

    /// The same events and stop check, and the lines, when they are written anywhere, collected
    /// for [`Progress::take`] instead.
    pub fn collecting(self) -> Progress {
        match self.lines {
            Some(_) => Progress {
                lines: Some(Sink::Lines(Arc::default())),
                ..self
            },
            None => self,
        }
    }

    /// The same lines, and every event to `listener`.
    pub fn with_events(self, listener: Listener) -> Progress {
        Progress {
            events: Some(listener),
            ..self
        }
    }

    /// The same lines and events, and `check` asked whether to stop.
    pub fn with_stop(self, check: StopCheck) -> Progress {
        Progress {
            stop: Some(Arc::new(Stop {
                check,
                since: Instant::now(),
                asked: AtomicU64::new(u64::MAX),
                requested: AtomicBool::new(false),
            })),
            ..self
        }
    }

    /// Whether the caller asks the calculation to stop, between iterations: its check, asked at
    /// most every [`STOP_INTERVAL`] (false in between; true once it asked to stop). False without
    /// a check.
    pub fn stop_due(&self) -> bool {
        self.stop.as_ref().is_some_and(|s| s.due())
    }

    /// Whether lines are written anywhere.
    pub fn on(&self) -> bool {
        self.lines.is_some()
    }

    /// The listener, if there is one.
    pub fn listener(&self) -> Option<Listener> {
        self.events.clone()
    }

    /// Whether anyone listens to the events: an event that costs work to make is made only then.
    pub fn listens(&self) -> bool {
        self.events.is_some()
    }

    /// Hand an event to the listener, if there is one.
    pub fn event(&self, event: Event<'_>) {
        if let Some(listener) = &self.events {
            listener(event);
        }
    }

    /// Write one line (`format_args!`; no trailing newline).
    pub fn line(&self, args: fmt::Arguments<'_>) {
        match &self.lines {
            None => {}
            Some(Sink::Stderr) => eprintln!("{args}"),
            Some(Sink::Lines(text)) => {
                use fmt::Write as _;
                let mut t = text.lock().unwrap_or_else(|e| e.into_inner());
                let _ = writeln!(t, "{args}");
            }
        }
    }

    /// The lines collected since the last call, emptied (None when nothing is collected).
    pub fn take(&self) -> Option<String> {
        match &self.lines {
            Some(Sink::Lines(text)) => Some(std::mem::take(
                &mut *text.lock().unwrap_or_else(|e| e.into_inner()),
            )),
            _ => None,
        }
    }

    /// A stage of the calculation: `run` between the events of its start and its end (with its
    /// wall time and whether it succeeded), when anyone listens; not run when the caller asks to
    /// stop before it starts (its check always asked here), which ends the calculation
    /// ([`LinusError::Stopped`]).
    pub fn stage<T>(
        &self,
        stage: &'static str,
        run: impl FnOnce() -> Result<T, LinusError>,
    ) -> Result<T, LinusError> {
        if self.stop.as_ref().is_some_and(|s| s.ask()) {
            return Err(LinusError::stopped(stage, None));
        }
        if !self.listens() {
            return run();
        }
        self.event(Event::StageStart { stage });
        let t0 = std::time::Instant::now();
        let out = run();
        self.event(Event::StageEnd {
            stage,
            seconds: t0.elapsed().as_secs_f64(),
            success: out.is_ok(),
        });
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn collects_lines_across_clones() {
        let p = Progress::collect();
        let q = p.clone();
        p.line(format_args!("iter {:3}", 1));
        q.line(format_args!("iter {:3}", 2));
        assert_eq!(p.take().as_deref(), Some("iter   1\niter   2\n"));
        assert_eq!(q.take().as_deref(), Some(""));
        assert!(!Progress::off().on() && Progress::off().take().is_none());
    }

    #[test]
    fn events_reach_the_listener_of_every_clone() {
        let seen: Arc<Mutex<Vec<String>>> = Arc::default();
        let s = seen.clone();
        let p = Progress::off().with_events(Arc::new(move |e: Event<'_>| {
            let fields: Vec<String> = e
                .fields()
                .iter()
                .map(|(k, v)| format!("{k}={v:?}"))
                .collect();
            s.lock()
                .unwrap()
                .push(format!("{} {}", e.name(), fields.join(" ")));
        }));
        assert!(p.listens() && !p.on());
        let q = p.clone();
        q.event(Event::ScfIteration {
            iteration: 3,
            energy: -1.5,
            delta_energy: -1e-3,
            gradient: 1e-4,
        });
        let r: Result<(), LinusError> =
            p.stage("gradient", || Err(LinusError::Internal("x".into())));
        assert!(r.is_err());
        let seen = seen.lock().unwrap();
        assert_eq!(
            seen[0],
            "scf_iteration iteration=Int(3) energy=Float(-1.5) delta_energy=Float(-0.001) gradient=Float(0.0001)"
        );
        assert_eq!(seen[1], "stage_start stage=Str(\"gradient\")");
        assert!(seen[2].starts_with("stage_end stage=Str(\"gradient\") seconds=Float("));
        assert!(seen[2].ends_with("success=Bool(false)"));
        // without a listener a stage runs as it is, and no event is made
        assert_eq!(
            Progress::off()
                .stage("scf", || Ok::<_, LinusError>(7))
                .unwrap(),
            7
        );
    }

    #[test]
    fn a_stop_request_ends_at_the_next_stage_or_due_iteration() {
        let asked = Arc::new(AtomicU64::new(0));
        let stop = Arc::new(AtomicBool::new(false));
        let (a, s) = (asked.clone(), stop.clone());
        let p = Progress::off().with_stop(Arc::new(move || {
            a.fetch_add(1, Ordering::Relaxed);
            s.load(Ordering::Relaxed)
        }));
        assert!(!p.stop_due() && asked.load(Ordering::Relaxed) == 1);
        // between iterations the check is asked again only after STOP_INTERVAL
        assert!(!p.stop_due() && asked.load(Ordering::Relaxed) == 1);
        // a stage always asks
        assert_eq!(p.stage("gradient", || Ok(3)).unwrap(), 3);
        assert_eq!(asked.load(Ordering::Relaxed), 2);
        stop.store(true, Ordering::Relaxed);
        std::thread::sleep(STOP_INTERVAL);
        assert!(p.clone().stop_due());
        // once asked to stop, every check says so without asking again
        assert!(p.stop_due() && asked.load(Ordering::Relaxed) == 3);
        let ran = std::cell::Cell::new(false);
        let r = p.stage("hessian", || {
            ran.set(true);
            Ok(())
        });
        assert!(!ran.get());
        match r {
            Err(LinusError::Stopped(s)) => {
                assert_eq!((s.stage, s.after), ("hessian", None));
                assert_eq!(
                    LinusError::Stopped(s).to_string(),
                    "stopped on request before the Hessian"
                );
            }
            r => panic!("{r:?}"),
        }
        // without a check nothing is asked and nothing stops
        assert!(!Progress::off().stop_due());
    }
}
