//! The batched four-center engine: integrals of one bra shell pair against blocks of `LANES`
//! ket pairs of one class, one ket pair per SIMD lane ([`crate::eri4c::batch`] has the
//! layout and the kernels).
//!
//! A task is one retained bra pair i. Its walk visits the ket groups (pairs of one class and
//! primitive count) in layout order, and within a group the blocks in order of falling
//! Schwarz factor, over the ket pairs j <= i only, so every symmetry-unique shell quartet is
//! seen once (8-fold permutational symmetry). A group ends at its first block that cannot
//! pass Q_i Q_blk >= thresh, or Q_i Q_blk * bound >= thresh with the task's density bound
//! from the digester: the blocks are sorted by Q, so no later block of the group can pass.
//!
//! The digester ([`BlockDigester`]) screens the lanes of each block that reaches it, runs the
//! kernels it needs and digests; it flushes per task. Tasks are scheduled by
//! [`crate::engine::schedule`] according to the digester's [`crate::engine::Reduction`].

use crate::eri4c::batch::{BatchKernels, BatchedPairs, Block, Pair, PairClass};

/// The task of one bra pair.
pub struct BraTask<'a> {
    /// The bra pair's position in the layout's retained pairs.
    pub i: usize,
    pub pair: &'a Pair,
    pub class: &'a PairClass,
}

/// A ket block the walk hands to the digester.
pub struct KetBlock<'a> {
    /// The block's position in the layout.
    pub index: usize,
    pub block: &'a Block,
    pub class: &'a PairClass,
}

/// What the batched four-center engine does with each block of a task (see the module docs).
///
/// Determinism: an implementor declares its reduction where it schedules the walk
/// ([`crate::engine::schedule`]); a digester that adds into fixed-point accumulators may run
/// under the dynamic schedule ([`crate::engine::schedule::exact`]), any other must not.
pub trait BlockDigester<V: BatchKernels> {
    /// The digester's state on one thread: accumulators and scratch.
    type Thread;

    /// Start the task; returns its density bound, the largest density factor any lane of
    /// the task can carry (the walk ends a group once Q_i Q_blk * bound < thresh).
    fn begin_task(&self, t: &mut Self::Thread, task: &BraTask<'_>) -> f64;

    /// One ket block that passed the walk's bounds: screen its lanes, run the kernels the
    /// digester needs and digest.
    fn block(&self, t: &mut Self::Thread, task: &BraTask<'_>, ket: &KetBlock<'_>);

    /// End the task: flush its contributions.
    fn end_task(&self, t: &mut Self::Thread, task: &BraTask<'_>);
}

/// Walk the task of bra pair `i` (see the module docs), screening blocks at `thresh`.
#[inline(always)]
pub fn walk_task<V, D>(bp: &BatchedPairs, thresh: f64, d: &D, t: &mut D::Thread, i: usize)
where
    V: BatchKernels,
    D: BlockDigester<V>,
{
    let pair = &bp.pairs[i];
    let task = BraTask {
        i,
        pair,
        class: &bp.classes[pair.class as usize],
    };
    let bound = d.begin_task(t, &task);
    let qi = pair.q;
    for g in &bp.groups {
        if g.pairs.start > i {
            break;
        }
        let class = &bp.classes[g.class as usize];
        for index in g.blocks.clone() {
            let block = &bp.blocks[index];
            // pairs are sorted by Q within a group: once Q_i Q_j fails, every later block does
            if block.first as usize > i
                || qi * block.qmax < thresh
                || qi * block.qmax * bound < thresh
            {
                break;
            }
            d.block(
                t,
                &task,
                &KetBlock {
                    index,
                    block,
                    class,
                },
            );
        }
    }
    d.end_task(t, &task);
}
