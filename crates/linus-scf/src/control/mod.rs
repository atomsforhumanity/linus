//! The SCF's control policy: pure state machines on the scalars of each iteration, which decide
//! when the SCF has converged, when it restarts, and when a converged point is analysed. They see
//! numbers, never matrices, so the host loop and the device loop make the same decisions from the
//! same code ([`Controls`]).
//!
//! | unit | decides |
//! |---|---|
//! | [`ConvergenceTest`] | converged: \|dE\| and the orbital gradient in its norm |
//! | [`LowestEnergy`] | the lowest energy visited (the saddle check's reference, the escalation's start) |
//! | [`TwoCycle`] | a two-cycle, answered with quasi-Newton steps |
//! | [`PlateauRule`] | an SCF accepted on a soft-mode plateau when the escalation comes due or at `max_iter`, or refused by the accuracy targets |
//! | [`BlowUp`] | a DIIS blow-up: the energy and the orbital gradient far above their lowest, iteration after iteration |
//! | [`EscalationRule`] | the iteration budget (`max_iter`, hard) and the restart within it when DIIS stagnates or blows up, or at 2/3 of it |
//! | [`StabilityTrigger`] | which converged UHF/UKS points are analysed, and how |
//!
//! Each iteration the loop reports an [`Observation`]; [`Controls::observe`] answers with a
//! [`Decision`]. The accelerators' own switches (when DIIS, ADIIS, damping, SOSCF and the level
//! shift apply) live with them in `super::accel`.

#[doc(hidden)]
pub mod blowup;
#[doc(hidden)]
pub mod convergence;
#[doc(hidden)]
pub mod escalation;
#[doc(hidden)]
pub mod lowest;
#[doc(hidden)]
pub mod plateau;
#[doc(hidden)]
pub mod trigger;
#[doc(hidden)]
pub mod two_cycle;

pub use blowup::{BlowUp, BLOWUP_ENERGY, BLOWUP_GRADIENT};
pub(crate) use convergence::ConvergenceTest;
pub(crate) use escalation::EscalationRule;
pub use escalation::{Trigger, MIN_ESCALATION};
pub(crate) use lowest::LowestEnergy;
pub(crate) use plateau::PlateauRule;
pub(crate) use trigger::{Analysis, StabilityTrigger, FULL_PRODUCTS};
pub use two_cycle::TwoCycle;

use super::{Plateau, PlateauRefusal, ScfConfig};

/// The scalars of one iteration's accepted Fock build, as the controls see them.
#[derive(Clone, Copy, Debug)]
pub struct Observation {
    pub iter: usize,
    /// 1 when iteration 1's density was a guess (not from orbitals), else 0.
    pub guess_iter: usize,
    pub energy: f64,
    /// E_iter - E_iter-1.
    pub de: f64,
    /// max |F_ai| and PySCF's gradient norm.
    pub gmax: f64,
    pub gnorm_pyscf: f64,
    /// The energies of the two iterations before (E_iter-1, E_iter-2; None before there were any).
    pub previous: [Option<f64>; 2],
    /// A full build the loop asked for at an unchanged density or after a restart (the confirmation of
    /// an incremental build, a restart's first build).
    pub rebuilt: bool,
    /// The build was incremental: of a difference density, or a term built from its history
    /// (COSX's incremental K, the PCM's predicted charges); not of the density alone.
    pub incremental: bool,
    /// The J/K builder moved to its next approximation in this iteration.
    pub switched: bool,
    /// The steps are quasi-Newton since a restart (the two-cycle detector is off).
    pub descending: bool,
}

/// What the loop does after an iteration.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Decision {
    /// Not converged: take a step.
    Continue,
    /// The criteria are met on an incremental build: rebuild in full at the same density and test
    /// again.
    Confirm,
    /// Converged, `e_above` above the lowest energy visited (the stability trigger's input).
    Converged { e_above: f64 },
    /// A two-cycle: restart the accelerators with quasi-Newton steps, then take a step.
    TwoCycle,
}

/// What the controls decide when the SCF has used its budget (`max_iter`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum AtBudget {
    /// Accepted on a plateau: converged (after the stability trigger's analysis of an open shell).
    Plateau(Plateau),
    /// A plateau the accuracy targets refuse: stop unconverged.
    Refused(PlateauRefusal),
    /// Stop unconverged.
    Stop,
}

/// The SCF's controls: its units' settings and state (scalars only).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Controls {
    pub convergence: ConvergenceTest,
    pub lowest: LowestEnergy,
    pub blowup: BlowUp,
    pub two_cycle: TwoCycle,
    pub plateau: PlateauRule,
    pub escalation: EscalationRule,
    pub stability: StabilityTrigger,
    /// The J/K builder's next approximation once max |F_ai| is below this (`grid_switch`).
    grid_switch: f64,
}

impl Controls {
    pub fn new(opts: &ScfConfig, restricted: bool) -> Self {
        Controls {
            convergence: ConvergenceTest {
                e_tol: opts.e_tol,
                g_tol: opts.g_tol,
                g_norm: opts.g_norm,
            },
            lowest: LowestEnergy::new(),
            blowup: BlowUp::new(opts.algo.blowup_window),
            two_cycle: TwoCycle::new(opts.algo.two_cycle),
            plateau: PlateauRule::new(opts),
            escalation: EscalationRule::new(
                opts.algo.escalate,
                opts.max_iter,
                opts.algo.stagnation_window,
            ),
            stability: StabilityTrigger::new(opts, restricted),
            grid_switch: opts.algo.grid_switch,
        }
    }

    /// One iteration: the lowest energy and the blow-up detector (both forgotten at a change of
    /// the J/K approximation), then the decision. Also returns whether the iteration is the new
    /// lowest.
    pub fn observe(&mut self, o: &Observation) -> (Decision, bool) {
        if o.switched {
            self.lowest.reset();
            self.blowup.reset();
        }
        let new_lowest = self.lowest.observe(o.iter, o.guess_iter, o.energy);
        self.blowup
            .observe(o.iter, o.guess_iter, o.energy - self.lowest.energy, o.gmax);
        // after a switch to the next approximation, dE compares two different operators
        let gconv = self.convergence.gconv(o.gmax, o.gnorm_pyscf);
        if !o.switched
            && self
                .convergence
                .met(o.iter, o.guess_iter, o.de, gconv, o.rebuilt)
        {
            // the SCF ends on a full build: an incremental one that meets the criteria is
            // rebuilt in full at the same density and tested again, so the returned Fock matrix
            // is the operator of one build (and requesting it changes no number)
            let decision = if o.incremental {
                Decision::Confirm
            } else {
                Decision::Converged {
                    e_above: o.energy - self.lowest.energy,
                }
            };
            return (decision, new_lowest);
        }
        let cycle = self
            .two_cycle
            .observe(o.iter, o.energy, o.previous, o.gmax, o.descending);
        let decision = if cycle {
            Decision::TwoCycle
        } else {
            Decision::Continue
        };
        (decision, new_lowest)
    }

    /// The decision when the SCF has used its iterations: the plateau rule over its iterations
    /// (`history`), else stop.
    pub fn at_budget(&self, history: &[super::ScfIteration]) -> AtBudget {
        match self.plateau.verdict(history) {
            plateau::Verdict::Accept(p) => AtBudget::Plateau(p),
            plateau::Verdict::Refuse(r) => AtBudget::Refused(r),
            plateau::Verdict::None => AtBudget::Stop,
        }
    }

    /// Whether the escalation is due after iteration `iters` (the step of iteration `iters`
    /// taken), and why: the lowest energy has not improved for `stagnation_window` iterations,
    /// 2/3 of `max_iter` are spent, or DIIS blew up over the last `blowup_window` iterations
    /// ([`EscalationRule::due`]). Both loops ask it at the same point; the host loop then lets the
    /// plateau rule decide first.
    pub fn escalation_due(&self, iters: usize) -> Option<Trigger> {
        self.escalation
            .due(iters, self.lowest.iter, self.blowup.blown_up())
    }

    /// The J/K builder moves to its next approximation (COSX: the next grid) once max |F_ai| is
    /// below `grid_switch`, and always before the SCF may converge.
    pub fn wants_next_approximation(&self, gmax: f64, gnorm_pyscf: f64) -> bool {
        gmax < self.grid_switch
            || self.convergence.gconv(gmax, gnorm_pyscf) < self.convergence.g_tol
    }

    /// The SCF keeps the occupied orbitals of its lowest-energy iterate: the start of the
    /// escalation, and the saddle check's first direction.
    pub fn keeps_lowest_orbitals(&self) -> bool {
        self.escalation.available() || self.stability.wants_lowest_orbitals()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn obs(iter: usize, energy: f64, de: f64, gmax: f64) -> Observation {
        Observation {
            iter,
            guess_iter: 1,
            energy,
            de,
            gmax,
            gnorm_pyscf: gmax,
            previous: [None, None],
            rebuilt: false,
            incremental: false,
            switched: false,
            descending: false,
        }
    }

    #[test]
    fn converged_confirmed_or_continued() {
        let o = ScfConfig::default();
        let mut c = Controls::new(&o, false);
        assert_eq!(
            c.observe(&obs(2, -1.0, -1.0, 1e-2)),
            (Decision::Continue, true)
        );
        let (d, low) = c.observe(&obs(3, -1.0 + 1e-12, 1e-12, 1e-9));
        assert_eq!(
            d,
            Decision::Converged {
                e_above: 1e-12 + -1.0 - -1.0
            }
        );
        assert!(!low);
        let inc = Observation {
            incremental: true,
            ..obs(4, -1.0, 0.0, 1e-9)
        };
        assert_eq!(c.observe(&inc).0, Decision::Confirm);
        // a switch: the lowest energy restarts and the step is not a convergence
        let sw = Observation {
            switched: true,
            ..obs(5, -0.5, 0.0, 1e-9)
        };
        assert_eq!(c.observe(&sw), (Decision::Continue, true));
        assert_eq!(c.lowest.energy, -0.5);
    }

    #[test]
    fn the_escalation_is_due_once_within_the_budget() {
        let o = ScfConfig::default();
        let mut c = Controls::new(&o, false);
        assert_eq!(c.at_budget(&[]), AtBudget::Stop);
        assert!(c.keeps_lowest_orbitals());
        // the lowest energy at iteration 2, then no lower one
        c.observe(&obs(2, -1.0, -1.0, 1e-2));
        let w = o.algo.stagnation_window;
        assert_eq!(c.escalation_due(1 + w), None);
        assert_eq!(c.escalation_due(2 + w), Some(Trigger::Stagnation));
        c.escalation.start(2 + w, Trigger::Stagnation, 2, -1.0, 0);
        assert_eq!(c.escalation_due(3 + w), None, "once");
        // closed shell with the escalation off: nothing needs the lowest iterate
        let mut o = ScfConfig::default();
        o.algo.escalate = false;
        assert!(!Controls::new(&o, true).keeps_lowest_orbitals());
        assert!(
            Controls::new(&o, false).keeps_lowest_orbitals(),
            "the saddle check's direction"
        );
    }

    /// Trp-cage PBE/def2-SVP: the lowest energy at iteration 23 (max |F_ai| 6.8e-3, its smallest),
    /// then 11 Eh higher at 0.146: the escalation is due after the window's 4th such iteration
    /// (`blowup`), 60 iterations before the stagnation window; a change of the J/K approximation
    /// starts the count again, and with the window 0 only the stagnation comes due.
    #[test]
    fn a_blowup_brings_the_escalation_forward() {
        let trace = |c: &mut Controls, switch_at: Option<usize>| {
            c.observe(&obs(2, -7464.3, 0.0, 8.7e-2));
            for k in 3..=23 {
                let e = -7470.0 - 0.0167 * (k - 2) as f64;
                c.observe(&obs(k, e, -1e-2, 6.8e-3 * (24 - k) as f64));
            }
            assert_eq!(c.lowest.iter, 23);
            let mut due = None;
            for k in 24..=40 {
                let o = Observation {
                    switched: switch_at == Some(k),
                    ..obs(k, -7459.2, 1e-2, 0.146)
                };
                c.observe(&o);
                if due.is_none() {
                    due = c.escalation_due(k).map(|t| (t, k));
                }
            }
            due
        };
        let o = ScfConfig::default();
        let mut c = Controls::new(&o, true);
        assert_eq!(trace(&mut c, None), Some((Trigger::BlowUp, 27)));
        assert_eq!(Trigger::BlowUp.name(), "blowup");
        // the switch at iteration 26 forgets the lowest energy and the smallest gradient
        let mut c = Controls::new(&o, true);
        assert_eq!(trace(&mut c, Some(26)), None);
        // off: nothing until the stagnation window
        let mut o = ScfConfig::default();
        o.algo.blowup_window = 0;
        let mut c = Controls::new(&o, true);
        assert_eq!(trace(&mut c, None), None);
        assert_eq!(c.escalation_due(23 + 64), Some(Trigger::Stagnation));
    }
}
