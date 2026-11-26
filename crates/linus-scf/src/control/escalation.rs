//! The iteration budget and the escalation within it ([`crate::ScfAlgorithm::escalate`]).
//!
//! `max_iter` is the SCF's whole budget of iterations. An SCF whose DIIS stagnates restarts once,
//! within that budget, from the orbitals of its lowest-energy iteration with quasi-Newton steps
//! (which descend in energy), a new DIIS subspace, no level shift and no damping. The escalation
//! starts after the iteration at which the lowest energy has not improved for `stagnation_window`
//! iterations, or after 2/3 of `max_iter`, or after `blowup_window` iterations of a DIIS blow-up
//! ([`super::BlowUp`]), whichever comes first, and only while at least [`MIN_ESCALATION`]
//! iterations are left: its first iteration only builds the lowest-energy density again, so it
//! needs a second to take a step.

use crate::{Escalation, EscalationRun};

/// The fewest iterations the escalation needs to be left: the restart's build of the
/// lowest-energy density, and one after the first quasi-Newton step.
pub const MIN_ESCALATION: usize = 2;

/// Why the escalation starts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Trigger {
    /// The lowest energy has not improved for `stagnation_window` iterations.
    Stagnation,
    /// 2/3 of `max_iter` are spent.
    TwoThirds,
    /// DIIS blew up: the energy and the orbital gradient far above their lowest for
    /// `blowup_window` iterations ([`super::BlowUp`]).
    BlowUp,
}

impl Trigger {
    /// The record's name.
    pub fn name(self) -> &'static str {
        match self {
            Trigger::Stagnation => "stagnation",
            Trigger::TwoThirds => "two_thirds",
            Trigger::BlowUp => "blowup",
        }
    }
}

/// The iteration budget and the escalation's state.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EscalationRule {
    enabled: bool,
    max_iter: usize,
    /// Iterations without a new lowest energy that start the escalation (0: only 2/3 of
    /// `max_iter` does).
    window: usize,
    /// The history index where the current phase (the first SCF, or the escalation) starts.
    pub phase_start: usize,
    /// The escalation, once it started.
    pub run: Option<EscalationRun>,
}

impl EscalationRule {
    pub fn new(enabled: bool, max_iter: usize, window: usize) -> Self {
        EscalationRule {
            enabled,
            max_iter,
            window,
            phase_start: 0,
            run: None,
        }
    }

    /// The SCF has used its budget after `iters` iterations.
    pub fn exhausted(&self, iters: usize) -> bool {
        iters >= self.max_iter
    }

    /// The last iteration after which the escalation starts without stagnation: 2/3 of
    /// `max_iter`.
    pub fn latest(&self) -> usize {
        2 * self.max_iter / 3
    }

    /// It may still start (on, and not yet).
    pub fn available(&self) -> bool {
        self.enabled && self.run.is_none()
    }

    /// Why the escalation starts after iteration `iters`, the lowest energy (of a density from
    /// orbitals) having last improved at iteration `lowest_iter` (0: none yet) and DIIS having
    /// `blown_up` ([`super::BlowUp::blown_up`]); None while it is not due, has run, is off, or the
    /// budget leaves it fewer than [`MIN_ESCALATION`] iterations. Where a blow-up coincides with
    /// another trigger the other names it: an SCF whose escalation the blow-up does not bring
    /// forward keeps its record.
    pub fn due(&self, iters: usize, lowest_iter: usize, blown_up: bool) -> Option<Trigger> {
        if !self.available() || lowest_iter == 0 || iters + MIN_ESCALATION > self.max_iter {
            return None;
        }
        if self.window > 0 && iters - lowest_iter >= self.window {
            Some(Trigger::Stagnation)
        } else if iters >= self.latest() {
            Some(Trigger::TwoThirds)
        } else if blown_up {
            Some(Trigger::BlowUp)
        } else {
            None
        }
    }

    /// The escalation starts after iteration `at` (`trigger`), from the iteration `from_iter` of
    /// energy `from_energy`; its phase starts at history index `phase_start`.
    pub fn start(
        &mut self,
        at: usize,
        trigger: Trigger,
        from_iter: usize,
        from_energy: f64,
        phase_start: usize,
    ) {
        self.run = Some(EscalationRun {
            trigger: trigger.name(),
            at,
            from_iter,
            from_energy,
            iterations: 0,
            converged: false,
        });
        self.phase_start = phase_start;
    }

    /// The record of an SCF that ended after `iters` iterations, `converged` (by the criteria or
    /// the plateau rule) or not, `at_budget` (it used its whole budget): the escalation if it
    /// ran; why it did not when the SCF used its budget; None when it converged before the
    /// escalation was due (or stopped on a numerical failure).
    pub fn finish(&mut self, iters: usize, converged: bool, at_budget: bool) -> Option<Escalation> {
        if let Some(e) = self.run.as_mut() {
            e.iterations = iters - e.at;
            e.converged = converged;
            return Some(Escalation::Ran(*e));
        }
        if !at_budget {
            return None;
        }
        Some(Escalation::NotRun {
            reason: if self.enabled { "max_iter" } else { "off" },
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stagnation_or_two_thirds_within_the_budget() {
        let mut e = EscalationRule::new(true, 30, 8);
        assert_eq!(e.latest(), 20);
        assert!(!e.exhausted(29) && e.exhausted(30));
        // no lowest-energy iterate yet (a guess density's iteration)
        assert_eq!(e.due(1, 0, false), None);
        assert_eq!(e.due(1, 0, true), None);
        assert_eq!(
            e.due(10, 3, false),
            None,
            "7 iterations without a lower energy"
        );
        assert_eq!(e.due(10, 3, true), Some(Trigger::BlowUp));
        assert_eq!(e.due(11, 3, false), Some(Trigger::Stagnation));
        assert_eq!(
            e.due(11, 3, true),
            Some(Trigger::Stagnation),
            "the stagnation names it"
        );
        assert_eq!(e.due(20, 19, false), Some(Trigger::TwoThirds));
        assert_eq!(e.due(20, 19, true), Some(Trigger::TwoThirds));
        // the budget's last iteration: too late for any
        assert_eq!(e.due(29, 3, true), None);
        e.start(11, Trigger::Stagnation, 3, -1.5, 11);
        assert!(!e.available() && e.due(20, 3, true).is_none(), "once");
        match e.finish(30, false, true) {
            Some(Escalation::Ran(r)) => assert_eq!(
                (r.trigger, r.at, r.from_iter, r.iterations, r.converged),
                ("stagnation", 11, 3, 19, false)
            ),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn a_small_budget_or_off_says_why_it_did_not_run() {
        // max_iter 2: after iteration 1 one iteration is left, fewer than the escalation needs
        let mut e = EscalationRule::new(true, 2, 8);
        assert_eq!(e.due(1, 1, true), None);
        assert_eq!(
            e.finish(2, false, true),
            Some(Escalation::NotRun { reason: "max_iter" })
        );
        let mut off = EscalationRule::new(false, 128, 8);
        assert_eq!(off.due(100, 1, true), None);
        assert_eq!(
            off.finish(128, false, true),
            Some(Escalation::NotRun { reason: "off" })
        );
        // converged before it was due: no record
        assert_eq!(
            EscalationRule::new(true, 128, 8).finish(20, true, false),
            None
        );
        // window 0: only 2/3 of max_iter
        let w0 = EscalationRule::new(true, 30, 0);
        assert_eq!(
            (w0.due(19, 2, false), w0.due(20, 2, false)),
            (None, Some(Trigger::TwoThirds))
        );
    }
}
