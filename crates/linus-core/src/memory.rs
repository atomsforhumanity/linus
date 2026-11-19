//! Memory: what a piece of work needs ([`MemoryPlan`]) and what it may use ([`Budget`]).
//!
//! These are the types only. The policy (the limits, the reserves left for other engines,
//! which optional storage to grant) belongs to the driver's plan, which checks each engine's
//! plan against its budget once, before the work, and records the decision with the bytes it
//! was based on. An engine never reads the free memory itself.

use std::fmt;

/// The bytes a piece of work needs: `persistent` for its whole life, `per_call` during each
/// call (freed between calls), and storage it can use when granted (faster, never needed).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct MemoryPlan {
    pub persistent: usize,
    pub per_call: usize,
    /// Optional storage, in the order the work prefers it granted.
    pub optional: Vec<OptionalStorage>,
}

/// Storage a piece of work uses when it fits (a stored operator, precomputed planes, a cache).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OptionalStorage {
    pub what: &'static str,
    pub bytes: usize,
}

impl MemoryPlan {
    /// The bytes the work cannot run without.
    pub fn required(&self) -> usize {
        self.persistent.saturating_add(self.per_call)
    }

    /// The bytes with every optional storage granted too.
    pub fn with_all_optional(&self) -> usize {
        self.optional
            .iter()
            .fold(self.required(), |s, o| s.saturating_add(o.bytes))
    }
}

/// Memory a piece of work may use: a limit and what is in use against it (bytes).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Budget {
    pub limit: usize,
    pub in_use: usize,
}

impl Budget {
    pub fn new(limit: usize, in_use: usize) -> Self {
        Budget { limit, in_use }
    }

    /// The bytes still free (zero when the use is above the limit).
    pub fn free(&self) -> usize {
        self.limit.saturating_sub(self.in_use)
    }

    /// Whether `bytes` more fit.
    pub fn fits(&self, bytes: usize) -> bool {
        bytes <= self.free()
    }

    /// Ok when `bytes` more fit, else what was asked for and why it did not fit.
    pub fn check(&self, what: &'static str, bytes: usize) -> Result<(), OverBudget> {
        if self.fits(bytes) {
            Ok(())
        } else {
            Err(self.over(what, bytes))
        }
    }

    /// The error of a request of `requested` bytes for `what` that does not fit this budget,
    /// for a caller whose rule is more than [`Budget::check`]'s comparison (a request that
    /// must leave a reserve free, or a block count derived from the free bytes).
    pub fn over(&self, what: &'static str, requested: usize) -> OverBudget {
        OverBudget {
            what,
            requested,
            in_use: self.in_use,
            limit: self.limit,
        }
    }
}

/// A request that does not fit its [`Budget`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OverBudget {
    pub what: &'static str,
    pub requested: usize,
    pub in_use: usize,
    pub limit: usize,
}

impl fmt::Display for OverBudget {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}: {} bytes requested with {} in use, over the limit of {} bytes",
            self.what, self.requested, self.in_use, self.limit
        )
    }
}

impl std::error::Error for OverBudget {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn budget_and_plan() {
        let b = Budget::new(10, 4);
        assert_eq!(b.free(), 6);
        assert!(b.fits(6) && !b.fits(7));
        assert_eq!(Budget::new(4, 10).free(), 0);
        let e = b.check("tensor", 7).unwrap_err();
        assert_eq!(
            e.to_string(),
            "tensor: 7 bytes requested with 4 in use, over the limit of 10 bytes"
        );
        let p = MemoryPlan {
            persistent: 3,
            per_call: 2,
            optional: vec![OptionalStorage {
                what: "planes",
                bytes: 4,
            }],
        };
        assert_eq!((p.required(), p.with_all_optional()), (5, 9));
    }
}
