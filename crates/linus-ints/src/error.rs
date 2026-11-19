//! The errors of the integral API: what a plan's constructor refuses, by kind.

use crate::pairs::{DerivOrder, PairKind};
use linus_core::ErrorKind;

/// Why a plan cannot be built for its inputs.
#[derive(Clone, Debug, PartialEq, thiserror::Error)]
pub enum IntsError {
    /// The pair set's derivative expansions are of lower order than the plan needs.
    #[error("the pair set has derivative expansions of order {have:?}; the plan needs {need:?}")]
    Order { need: DerivOrder, have: DerivOrder },
    /// The pair set is of orbital pairs where the plan takes the (P, 1) pairs of an auxiliary
    /// basis, or the other way round.
    #[error("the plan takes {need:?} pairs")]
    PairKind { need: PairKind },
    /// A shell of higher angular momentum than the kernels support.
    #[error("angular momentum {l} is beyond the kernels' {max}")]
    AngularMomentum { l: usize, max: usize },
}

impl IntsError {
    /// The cross-package error kind: a request the kernels do not cover is unsupported; a pair
    /// set of the wrong order or kind is a caller's bug.
    pub fn kind(&self) -> ErrorKind {
        match self {
            IntsError::AngularMomentum { .. } => ErrorKind::Unsupported,
            IntsError::Order { .. } | IntsError::PairKind { .. } => ErrorKind::Internal,
        }
    }
}
