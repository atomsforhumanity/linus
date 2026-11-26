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
    /// The engine of the last SCF's density-fitted J on the device (`backend.df_j`). Decided by
    /// the GPU backend only.
    #[cfg_attr(not(feature = "gpu"), allow(dead_code))]
    ScfDfJ(DfJChoice),
    /// Where the last SCF setup's nuclear attraction V ran (`gpu`, `cpu`), and why. Decided
    /// beside a device J/K's setup only.
    NuclearAttraction { place: &'static str, reason: String },
    /// The kernels of the last device two-electron gradient (`hermite_j`, `stored_planes`,
    /// `fused`: `linus_gpu::grad::df::GpuDfGrad::kernels`), and why. Decided by the GPU backend
    /// only.
    #[cfg_attr(not(feature = "gpu"), allow(dead_code))]
    GradientJkKernels {
        kernels: &'static str,
        reason: String,
    },
}

/// The engine of an SCF's density-fitted J on the device (`backend.df_j`'s resolution,
/// `resolved.record.backend.j`): `stored` (the DF engine's tensor) or `direct` (the memory-lean
/// J, `linus_gpu::jdirect`), and why; the memory-lean J's kernels ("hermite" or "eri3c") and why;
/// and, where the memory-lean J was chosen but its setup did not fit, why (the DF engine took
/// over).
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct DfJChoice {
    pub engine: &'static str,
    pub reason: String,
    pub kernels: Option<(&'static str, String)>,
    pub declined: Option<String>,
}

impl Decision {
    /// Whether `self` supersedes `earlier` (the same kind of decision for the same family).
    fn supersedes(&self, earlier: &Decision) -> bool {
        match (self, earlier) {
            (Decision::Setup { kernel: a, .. }, Decision::Setup { kernel: b, .. }) => a == b,
            (Decision::Probes { .. }, Decision::Probes { .. }) => true,
            (Decision::ScfDfJ(_), Decision::ScfDfJ(_)) => true,
            (Decision::NuclearAttraction { .. }, Decision::NuclearAttraction { .. }) => true,
            (Decision::GradientJkKernels { .. }, Decision::GradientJkKernels { .. }) => true,
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

    /// The engine of the last SCF's density-fitted J on the device and why (None: not decided:
    /// no device J/K).
    pub(crate) fn scf_df_j(&self) -> Option<DfJChoice> {
        self.lock().decisions.iter().find_map(|d| match d {
            Decision::ScfDfJ(c) => Some(c.clone()),
            _ => None,
        })
    }

    /// Where the last SCF setup's nuclear attraction ran, and why (None: not decided).
    pub(crate) fn nuclear_attraction(&self) -> Option<(&'static str, String)> {
        self.lock().decisions.iter().find_map(|d| match d {
            Decision::NuclearAttraction { place, reason } => Some((*place, reason.clone())),
            _ => None,
        })
    }

    /// The kernels of the last device two-electron gradient, and why (None: none ran).
    pub(crate) fn gradient_jk_kernels(&self) -> Option<(&'static str, String)> {
        self.lock().decisions.iter().find_map(|d| match d {
            Decision::GradientJkKernels { kernels, reason } => Some((*kernels, reason.clone())),
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
