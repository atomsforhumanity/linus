//! The stability trigger: which converged UHF/UKS points the SCF analyses, and how.
//!
//! - **The saddle check** (`scf.stability_saddle_check`): a point converged more than this above the lowest
//!   energy visited is not a minimum along the path; the full analysis starts towards the
//!   lowest-energy iterate.
//! - **`scf.stability`**: every converged point gets the full analysis.
//! - **The cheap probe** (`scf.stability_probe`, the D25 default at a session's first
//!   geometry): a few one-sided Davidson products per irrep; the full analysis only when they
//!   already find a negative mode, in that irrep.
//!
//! Every search runs per irrep of the converged state's symmetry (`crate::stability::Starts`).
//! - **The gap trigger** of a continued SCF ([`ProbeTrigger`]): where the probe is off, it runs when
//!   the converged point's gap fires the trigger.
//! - **The zero-mode check** of a Hessian SCF (`scf.stability_hessian_check`, [`ScfConfig::mode_check`]):
//!   every converged point is probed, with the check's products where they are more than the
//!   probe's; a negative mode takes the probe's path (the full analysis, the step), and the search
//!   at the accepted point is the check's record.
//!
//! At most `scf.stability_max` analyses per SCF, and none after one found a minimum.

use crate::stability::{ProbeTrigger, TriggerRecord};
use crate::ScfConfig;

/// Davidson products of the full analysis.
pub(crate) const FULL_PRODUCTS: usize = 8;

/// An analysis the trigger asks for.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Analysis {
    /// Only the cheap probe first (the full analysis follows when it finds a negative mode).
    pub probing: bool,
    /// Davidson products of the first search per irrep (`FULL_PRODUCTS`, or the probe's).
    pub products: usize,
    /// The converged energy above the lowest visited (Eh).
    pub e_above: f64,
    /// Start the search towards the lowest-energy iterate (the saddle check's direction).
    pub towards_lowest: bool,
    /// The saddle check fired: that direction is the path's (its irreps are searched first).
    pub saddle_fired: bool,
}

/// The trigger's settings and state.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StabilityTrigger {
    restricted: bool,
    saddle_check: f64,
    always: bool,
    max: usize,
    /// The probe's products (`scf.stability_probe` resolved; the gap trigger may set them).
    probe: usize,
    /// The zero-mode check's products (`ScfConfig::mode_check`; 0: not a Hessian SCF).
    check: usize,
    trigger: Option<ProbeTrigger>,
    /// An analysis found the converged point a minimum.
    pub stable: bool,
    /// The gap trigger's decision (at the first converged point it looked at).
    pub record: Option<TriggerRecord>,
}

impl StabilityTrigger {
    pub fn new(opts: &ScfConfig, restricted: bool) -> Self {
        StabilityTrigger {
            restricted,
            saddle_check: opts.algo.saddle_check,
            always: opts.algo.stability,
            max: opts.algo.stability_max,
            probe: opts.algo.stability_probe.unwrap_or(0),
            check: opts.mode_check,
            trigger: opts.probe_trigger,
            stable: false,
            record: None,
        }
    }

    /// The saddle check fires at a point `e_above` above the lowest energy visited.
    fn saddle_fired(&self, e_above: f64) -> bool {
        self.saddle_check > 0.0 && e_above > self.saddle_check
    }

    /// The trigger keeps the lowest-energy iterate's occupied orbitals (the saddle check's first
    /// direction) for analyses that may come.
    pub fn wants_lowest_orbitals(&self) -> bool {
        !self.restricted && self.saddle_check > 0.0 && self.max > 0
    }

    /// Whether a converged point `e_above` above the lowest energy may need an analysis, before its
    /// gap is known: the points where the device loop stops for [`StabilityTrigger::decide`].
    pub fn may_analyse(&self, e_above: f64) -> bool {
        !self.restricted
            && self.max > 0
            && (self.saddle_fired(e_above)
                || self.always
                || self.probe > 0
                || self.check > 0
                || self.trigger.is_some_and(|t| t.stability_products > 0))
    }

    /// The analysis of a converged point `e_above` above the lowest energy visited, after `done`
    /// analyses of this SCF; `gap` gives the point's gap when the gap trigger needs it (it decides
    /// once, at the first converged point it sees). None: the point is accepted as it is.
    pub fn decide(
        &mut self,
        e_above: f64,
        done: usize,
        gap: impl FnOnce() -> Option<f64>,
    ) -> Option<Analysis> {
        if self.restricted {
            return None;
        }
        let saddle_fired = self.saddle_fired(e_above);
        let full_check = saddle_fired || self.always;
        if self.probe == 0 && self.record.is_none() {
            if let Some(t) = self.trigger.filter(|t| t.stability_products > 0) {
                let rec = t.decide(gap());
                if rec.fired() {
                    self.probe = t.stability_products;
                }
                self.record = Some(rec);
            }
        }
        // a Hessian SCF probes every converged point, with the zero-mode check's products where they
        // are more
        let products = self.probe.max(self.check);
        let probing = !full_check && products > 0;
        if self.stable || done >= self.max || !(full_check || probing) {
            return None;
        }
        Some(Analysis {
            probing,
            products: if probing { products } else { FULL_PRODUCTS },
            e_above,
            towards_lowest: e_above > 0.0 && self.saddle_check > 0.0,
            saddle_fired,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn opts(f: impl FnOnce(&mut ScfConfig)) -> ScfConfig {
        let mut o = ScfConfig::default();
        f(&mut o);
        o
    }

    #[test]
    fn saddle_check_probe_and_full_analysis() {
        // the saddle check: the full analysis, towards the lowest iterate
        let mut t = StabilityTrigger::new(&opts(|_| {}), false);
        assert_eq!(t.decide(1e-12, 0, || None), None, "at the lowest energy");
        let a = t.decide(1e-6, 0, || None).unwrap();
        assert_eq!((a.probing, a.products), (false, FULL_PRODUCTS));
        assert!(a.towards_lowest);
        assert_eq!(t.decide(1e-6, 2, || None), None, "stability_max reached");
        t.stable = true;
        assert_eq!(t.decide(1e-6, 0, || None), None, "found a minimum");
        // the probe: its products
        let mut p = StabilityTrigger::new(&opts(|o| o.algo.stability_probe = Some(2)), false);
        let a = p.decide(0.0, 0, || None).unwrap();
        assert_eq!((a.probing, a.products, a.towards_lowest), (true, 2, false));
        // scf.stability: the full analysis
        let mut s = StabilityTrigger::new(&opts(|o| o.algo.stability = true), false);
        let a = s.decide(0.0, 0, || None).unwrap();
        assert_eq!((a.probing, a.products), (false, FULL_PRODUCTS));
        // closed shells: never
        let mut r = StabilityTrigger::new(&opts(|o| o.algo.stability = true), true);
        assert!(r.decide(1.0, 0, || None).is_none() && !r.may_analyse(1.0));
    }

    #[test]
    fn a_hessian_scf_probes_with_the_checks_products() {
        // the probe off (a later geometry): the check probes alone
        let mut t = StabilityTrigger::new(&opts(|o| o.mode_check = 6), false);
        assert!(t.may_analyse(0.0));
        let a = t.decide(0.0, 0, || None).unwrap();
        assert_eq!((a.probing, a.products), (true, 6));
        // the probe on: the larger of the two
        let o = opts(|o| {
            o.mode_check = 6;
            o.algo.stability_probe = Some(2);
        });
        assert_eq!(
            StabilityTrigger::new(&o, false)
                .decide(0.0, 0, || None)
                .unwrap()
                .products,
            6
        );
        let o = opts(|o| {
            o.mode_check = 2;
            o.algo.stability_probe = Some(4);
        });
        assert_eq!(
            StabilityTrigger::new(&o, false)
                .decide(0.0, 0, || None)
                .unwrap()
                .products,
            4
        );
        // scf.stability: the full analysis
        let o = opts(|o| {
            o.mode_check = 6;
            o.algo.stability = true;
        });
        let a = StabilityTrigger::new(&o, false)
            .decide(0.0, 0, || None)
            .unwrap();
        assert_eq!((a.probing, a.products), (false, FULL_PRODUCTS));
        // the gap trigger still decides and records, its products raised to the check's
        let trig = ProbeTrigger {
            last_gap: None,
            threshold: 0.15,
            stability_products: 3,
            closed_products: 0,
        };
        let o = opts(|o| {
            o.mode_check = 6;
            o.probe_trigger = Some(trig);
        });
        let mut g = StabilityTrigger::new(&o, false);
        assert_eq!(g.decide(0.0, 0, || Some(0.3)).map(|a| a.products), Some(6));
        assert!(!g.record.unwrap().fired());
    }

    #[test]
    fn the_gap_trigger_decides_once() {
        let trig = ProbeTrigger {
            last_gap: None,
            threshold: 0.15,
            stability_products: 3,
            closed_products: 0,
        };
        let o = opts(|o| o.probe_trigger = Some(trig));
        let mut t = StabilityTrigger::new(&o, false);
        assert!(
            t.may_analyse(0.0),
            "armed: the device loop stops at its converged point"
        );
        let mut asked = 0;
        let a = t.decide(0.0, 0, || {
            asked += 1;
            Some(0.1)
        });
        assert_eq!(a.map(|a| (a.probing, a.products)), Some((true, 3)));
        assert!(t.record.unwrap().fired());
        t.decide(0.0, 1, || {
            asked += 1;
            Some(0.1)
        });
        assert_eq!(asked, 1, "the gap is read once");
        // a gap above the gate: recorded, no probe
        let mut u = StabilityTrigger::new(&o, false);
        assert!(u.decide(0.0, 0, || Some(0.3)).is_none());
        assert!(!u.record.unwrap().fired());
        // nothing armed: the device loop does not stop
        let off = StabilityTrigger::new(&opts(|o| o.algo.stability_max = 0), false);
        assert!(!off.may_analyse(1.0));
    }
}
