//! The inside of every plan: batch engines that walk the integral batches of a pair set in a
//! fixed order and hand each batch to a digester (SHARK's integral engine and consumers).
//!
//! - The engine owns the walk: the task list, the order of the batches within a task, the
//!   integral (Schwarz) bounds that end a walk early, and the schedule over the threads.
//! - The digester owns what is done with a batch: the density bounds of its lanes, the
//!   kernels it runs (integrals of an operator, or a fused contraction such as the Hermite
//!   J-engine), and the contraction into its accumulators.
//! - Kernels are pure functions of pair data ([`crate::eri4c::batch::BatchKernels`] and the
//!   per-family kernels).
//!
//! Digesters are generic parameters, monomorphized with the engine, never trait objects: a
//! batch is a few hundred floating-point operations, and the digestion is inlined into the
//! walk.
//!
//! **The determinism contract.** Every digester declares how it adds its contributions up
//! ([`Reduction`]): exactly, in a fixed-point format, or in floating point. The engines take
//! their schedule from that declaration through the one scheduler, [`schedule`], so a
//! floating-point digester cannot be scheduled in an order that depends on the thread count.
//! Results therefore have the same bits for any number of threads.
//!
//! Engines:
//! - [`block`]: four-center integrals of one bra pair against blocks of `LANES` ket pairs of one
//!   class (the batched kernels; the J/K digester of [`crate::eri4c::jk`]).
//!
//! Plans report the memory their caller-owned workspaces take before they run
//! ([`WorkspaceSize`]).

pub mod block;
pub mod schedule;

pub use schedule::Reduction;

/// The memory a workspace takes, in bytes: per pool thread, and shared by the threads.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct WorkspaceSize {
    pub per_thread: usize,
    pub shared: usize,
}

impl WorkspaceSize {
    /// The bytes for a pool of `nthreads` threads.
    pub fn total(&self, nthreads: usize) -> usize {
        self.per_thread * nthreads + self.shared
    }
}
