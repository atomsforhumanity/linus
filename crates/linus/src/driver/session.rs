//! A session's requests: one calculation at given positions in a lineage ([`Request`],
//! [`Session::compute`]), and where its SCF starts.
//!
//! A request moves the atoms, starts its SCF from its lineage's state (or its seed's, when its
//! own lineage has none) as [`crate::driver::lineage`] describes, runs its calctype, and commits
//! the lineage only when the whole request succeeds (and `commit` is set). The session's
//! direct calls (`energy`, `gradient`, `run`, ...) are requests in the default lineage, each
//! committed at its SCF.

use crate::driver::lineage::{
    FreshReason, GuessRecord, GuessSource, Lineage, LineageKey, ScfStart, GUESS_DISTANCE,
};
use crate::driver::report::{CalcType, Report};
use crate::driver::{Guess, Session};
use linus_base::error::LinusError;

/// One calculation of a session ([`Session::compute`]).
#[derive(Clone, Debug)]
pub struct Request {
    /// The positions (bohr; None: the current ones).
    pub positions: Option<Vec<[f64; 3]>>,
    /// What to compute.
    pub calctype: CalcType,
    /// The lineage the request continues and commits to.
    pub lineage: LineageKey,
    /// The lineages to start from, in order, when the request's own lineage has no state: the
    /// first of them with a state starts it (both branches of an IRC from the transition
    /// state's, path images from a neighbour's); none with a state is a fresh start.
    pub seed: Vec<LineageKey>,
    /// Whether a successful request commits its lineage (false: it stores nothing).
    pub commit: bool,
    /// A Hessian's analysis for this request (None: the session's `freq` settings).
    pub freq: Option<linus_vib::FrequencyOptions>,
    /// This request's accuracy targets, replacing the settings' `accuracy` (None: those).
    pub accuracy: Option<crate::driver::accuracy::Targets>,
}

impl Request {
    /// A request of `calctype` at the current positions in the default lineage, committed.
    pub fn new(calctype: CalcType) -> Request {
        Request {
            positions: None,
            calctype,
            lineage: LineageKey::default(),
            seed: Vec::new(),
            commit: true,
            freq: None,
            accuracy: None,
        }
    }
}

/// The request being run: its lineage, its SCF's start (taken by its first SCF) and the lineage
/// after its converged SCF (a later SCF of the request continues from it).
#[derive(Debug)]
pub(crate) struct Call {
    key: LineageKey,
    start: Option<ScfStart>,
    after: Option<Lineage>,
}

/// What `set_guess_orbitals` (the Python session's `ProgramInput.given`) or `set_guess_density`
/// set for the default lineage's next SCF.
#[derive(Debug, Default)]
pub(crate) struct Given {
    /// Spin densities the next SCF starts from, used once, and whether they come from orbitals
    /// made orthonormal here.
    pub guess: Option<(faer::Mat<f64>, faer::Mat<f64>, bool)>,
}

impl Session {
    /// Run the request `req` (see the module docs): its report, or the error (the lineage then
    /// stays as it was; [`Session::failure`] has what the calculation reached).
    pub fn compute(&mut self, req: &Request) -> Result<Report, LinusError> {
        if self.call.is_some() {
            return Err(LinusError::Internal("a request inside a request".into()));
        }
        if let Some(x) = &req.positions {
            self.set_positions(x)?;
        }
        // its own lineage's state (the default lineage's first: the given orbitals), else its
        // first seed's with one
        let own = self.lineages.get(&req.lineage);
        let mut given = None;
        let start = if own.is_none()
            && req.lineage == LineageKey::default()
            && self.given.guess.is_some()
        {
            let s = self.direct_start(true);
            // used once: kept again if the request fails
            given = self.given.guess.take();
            s
        } else {
            let from = std::iter::once(&req.lineage)
                .chain(&req.seed)
                .find_map(|k| self.lineages.get(k).map(|l| (k, l)));
            self.start_from(&req.lineage, from, true)
        };
        self.call = Some(Call {
            key: req.lineage.clone(),
            start: Some(start),
            after: None,
        });
        // the request's targets for its run only
        let targets = req.accuracy.map(|t| self.cx.plan.set_targets(t));
        let out = match &req.freq {
            Some(f) => self.run_with(req.calctype, f),
            None => self.run(req.calctype),
        };
        if let Some(t) = targets {
            self.cx.plan.set_targets(t);
        }
        let call = self.call.take().expect("the request's call");
        if out.is_err() && given.is_some() {
            self.given.guess = given;
        }
        if let (Ok(_), true, Some(after)) = (&out, req.commit, call.after) {
            self.lineages.insert(call.key, after);
        }
        out
    }

    /// Forget the lineage `lineage`: its next request starts fresh.
    pub fn reset(&mut self, lineage: &LineageKey) {
        self.lineages.remove(lineage);
    }

    /// Forget every lineage.
    pub fn reset_all(&mut self) {
        self.lineages.clear();
    }

    /// The lineages with a state, in key order.
    pub fn lineages(&self) -> Vec<LineageKey> {
        self.lineages.keys().cloned().collect()
    }

    /// The configured guess (`scf.guess`) as a guess source.
    fn configured_guess(&self) -> GuessSource {
        match self.cx.plan.guess() {
            Guess::Sad => GuessSource::Sad,
            Guess::Core => GuessSource::Core,
        }
    }

    /// The start of an SCF of the lineage `key` at the current positions from the state of the
    /// lineage `from` (its own or its seed's; None: no state): a continuation within
    /// [`GUESS_DISTANCE`], with its densities at the same positions and its orbitals made
    /// orthonormal here elsewhere (when `project`; else the guess is left out), else a fresh
    /// start from the configured guess.
    pub(crate) fn start_from(
        &self,
        key: &LineageKey,
        from: Option<(&LineageKey, &Lineage)>,
        project: bool,
    ) -> ScfStart {
        let configured = self.configured_guess();
        let Some((fk, l)) = from else {
            return ScfStart::fresh(configured, key, None, FreshReason::NoHistory);
        };
        let d = l.distance(&self.cx.mol.positions);
        if d > GUESS_DISTANCE {
            return ScfStart::fresh(configured, key, Some(d), FreshReason::Distance);
        }
        let (guess, orthonormalized) = if !project {
            (None, false)
        } else if d == 0.0 {
            (Some(l.densities()), false)
        } else {
            let (c_a, c_b, restricted) = l.occupied();
            match self.cx.projected_guess(c_a, c_b, restricted) {
                Some(g) => (Some(g), true),
                // linearly dependent here: the densities as they are
                None => (Some(l.densities()), false),
            }
        };
        ScfStart {
            guess,
            continuation: true,
            last_probe_gap: l.probe_gap(),
            record: GuessRecord {
                source: GuessSource::Lineage,
                lineage: key.value(),
                from_lineage: Some(fk.value()),
                distance: Some(d),
                orthonormalized,
                reason: None,
            },
        }
    }

    /// The start of a direct call's SCF: the default lineage's, or the guess `set_guess_orbitals`
    /// or `set_guess_density` gave (only when `project`).
    fn direct_start(&self, project: bool) -> ScfStart {
        let key = LineageKey::default();
        let own = self.lineages.get(&key);
        match &self.given.guess {
            Some((d_a, d_b, orthonormalized)) => ScfStart {
                guess: project.then(|| (d_a.clone(), d_b.clone())),
                continuation: false,
                last_probe_gap: None,
                record: GuessRecord {
                    source: GuessSource::Given,
                    lineage: key.value(),
                    from_lineage: None,
                    distance: None,
                    orthonormalized: *orthonormalized,
                    reason: None,
                },
            },
            None => self.start_from(&key, own.map(|l| (&key, l)), project),
        }
    }

    /// The start of the next SCF without its guess (what `stability_probe`,
    /// `closed_shell_probe` and the plan report).
    pub(crate) fn next_start(&self) -> ScfStart {
        match &self.call {
            Some(Call { start: Some(s), .. }) => ScfStart {
                guess: None,
                ..s.clone()
            },
            Some(Call {
                key,
                after: Some(l),
                ..
            }) => self.start_from(key, Some((key, l)), false),
            _ => self.direct_start(false),
        }
    }

    /// Where the SCF that runs now starts: the request's start (its first SCF), a continuation of
    /// the request's converged SCF (a later one), or a direct call's (its given guess used up).
    pub(crate) fn take_start(&mut self) -> ScfStart {
        if let Some(s) = self.call.as_mut().and_then(|c| c.start.take()) {
            return s;
        }
        if let Some(Call {
            key,
            after: Some(l),
            ..
        }) = &self.call
        {
            return self.start_from(key, Some((key, l)), true);
        }
        let s = self.direct_start(true);
        self.given.guess = None;
        s
    }

    /// The lineage after the SCF that started at `start` converged as `r` here: the request's
    /// (committed when it succeeds), or the default lineage's for a direct call (committed now).
    pub(crate) fn scf_converged(&mut self, start: &ScfStart, r: &linus_scf::ScfResult) {
        let after = Lineage::after(start, r, &self.cx.mol.positions);
        match self.call.as_mut() {
            Some(call) => call.after = Some(after),
            None => {
                self.lineages.insert(LineageKey::default(), after);
            }
        }
    }
}
