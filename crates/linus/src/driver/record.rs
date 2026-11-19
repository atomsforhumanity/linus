//! The run record of a session: what it decided at run time, written by the code that
//! decides, at the moment it decides ([`linus_core::RunRecord`] over [`Decision`]). The report's
//! `resolved.record` and `warnings` are generated from it and the settings, not reconstructed
//! from the session's state afterwards.
//!
//! A decision supersedes the earlier one of its kind and family (the last device setup of a
//! family is the one that ran), so the record stays small across a session's calls. A report's
//! warnings are generated from the record and the run's results (`crate::driver::report`).
//!
//! What only a device decides inside its setups (the FP64 emulation that ran, how its engines
//! kept to their workspace grants) is its provider's record, read through
//! [`crate::backend::Backend::emulation`] and [`crate::backend::Backend::grant_violations`].

use crate::driver::backend::Family;
use linus_core::RunRecord;
use std::sync::{Mutex, MutexGuard};

/// A run-time decision.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum Decision {
    /// A family's device setup ran on the device (None), or the CPU took over, and why.
    Setup {
        kernel: Family,
        fallback: Option<String>,
    },
    /// The last SCF's probes: the Hessian-vector products of the stability probe and of the
    /// closed-shell probe it was given (the probes' rule at its geometry).
    Probes {
        stability: usize,
        closed_shell: usize,
    },
}

impl Decision {
    /// Whether `self` supersedes `earlier` (the same kind of decision for the same family).
    fn supersedes(&self, earlier: &Decision) -> bool {
        match (self, earlier) {
            (Decision::Setup { kernel: a, .. }, Decision::Setup { kernel: b, .. }) => a == b,
            (Decision::Probes { .. }, Decision::Probes { .. }) => true,
            _ => false,
        }
    }
}

/// The record of a session, behind a lock: the engines record from `&Session`.
#[derive(Debug, Default)]
pub(crate) struct Recorder(Mutex<RunRecord<Decision>>);

impl Recorder {
    fn lock(&self) -> MutexGuard<'_, RunRecord<Decision>> {
        self.0.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Records `d`, superseding the earlier decision of its kind and family.
    pub(crate) fn decide(&self, d: Decision) {
        let mut r = self.lock();
        r.decisions.retain(|e| !d.supersedes(e));
        r.decide(d);
    }

    /// The last device setup of `k`: None when none ran, Some(None) on the device, Some(reason)
    /// when the CPU took over.
    pub(crate) fn setup(&self, k: Family) -> Option<Option<String>> {
        self.lock().decisions.iter().find_map(|d| match d {
            Decision::Setup { kernel, fallback } if *kernel == k => Some(fallback.clone()),
            _ => None,
        })
    }

    /// The last SCF's probes (stability, closed shell).
    pub(crate) fn probes(&self) -> Option<(usize, usize)> {
        self.lock().decisions.iter().find_map(|d| match d {
            Decision::Probes {
                stability,
                closed_shell,
            } => Some((*stability, *closed_shell)),
            _ => None,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A later decision of a family supersedes the earlier one.
    #[test]
    fn decisions_supersede() {
        let r = Recorder::default();
        assert_eq!(r.setup(Family::Jk), None);
        r.decide(Decision::Setup {
            kernel: Family::Jk,
            fallback: Some("out of memory".into()),
        });
        r.decide(Decision::Setup {
            kernel: Family::Xc,
            fallback: None,
        });
        r.decide(Decision::Setup {
            kernel: Family::Jk,
            fallback: None,
        });
        assert_eq!(r.setup(Family::Jk), Some(None));
        assert_eq!(r.setup(Family::Xc), Some(None));
        assert_eq!(r.lock().decisions.len(), 2);
    }
}
