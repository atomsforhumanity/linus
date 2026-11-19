//! How the batch engines deal their tasks to threads: the determinism contract of
//! [`crate::engine`], written once for every engine.
//!
//! A digester declares how it adds up its contributions ([`Reduction`]) and the engine takes
//! the schedule from that:
//! - [`Reduction::Exact`]: the digester accumulates in a fixed-point format
//!   (`linus_num::det`), where addition is exact and associative, so any order of the tasks
//!   gives the same bits. [`exact`] hands the tasks to the pool's threads dynamically, from one
//!   shared counter over a list ordered most expensive first, each thread adding into its own
//!   slot of the caller's workspace.
//! - [`Reduction::Ordered`]: the digester accumulates in floating point. [`ordered`] deals
//!   the tasks to a number of strided chunks fixed by the problem (task i to chunk
//!   i mod nchunk), folds each chunk in task order into its own accumulator of the caller's
//!   workspace, and adds the accumulators up in chunk order
//!   (`linus_num::det::par_fold_det_into`). With large per-task scratch the same order runs
//!   on scratch per thread and partials per chunk, the chunks dealt to the threads by one
//!   dispatcher: [`ordered_strided`] (strided chunks, the partials the caller's),
//!   [`ordered_pool`] (strided chunks, flat partials carved from one workspace pool) and
//!   [`ordered_ranges`] (contiguous chunks, for digesters that read consecutive tasks).
//!
//! Either way the result does not depend on the number of threads or on the schedule. Both
//! run on the caller's current rayon pool. Independent tasks that each write their own part
//! of an output use the same dispatcher: [`per_part`].

use std::ops::Range;
use std::sync::atomic::{AtomicUsize, Ordering};

/// How a digester's contributions are added up (see the module docs).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Reduction {
    /// Fixed-point accumulation: exact in any order; the engine schedules dynamically.
    Exact,
    /// Floating-point accumulation: the engine fixes the order (problem-fixed chunks, each
    /// folded in task order, combined in chunk order).
    Ordered,
}

/// The tasks of one thread under [`exact`]: the next task of the shared list, until it is
/// exhausted.
pub struct Tasks<'a> {
    order: &'a [u32],
    next: &'a AtomicUsize,
}

impl Iterator for Tasks<'_> {
    type Item = u32;

    #[inline]
    fn next(&mut self) -> Option<u32> {
        let t = self.next.fetch_add(1, Ordering::Relaxed);
        self.order.get(t).copied()
    }
}

/// Run the tasks `order` (task ids, most expensive first) on every thread of the current
/// pool: pool thread t calls `worker(&mut slots[t], tasks)` once, and its [`Tasks`] hand out
/// the next task of the shared list. For [`Reduction::Exact`] digesters only.
///
/// # Panics
/// When `slots` does not have one slot per pool thread: a workspace made for another thread
/// count.
pub fn exact<T: Send>(order: &[u32], slots: &mut [T], worker: impl Fn(&mut T, Tasks<'_>) + Sync) {
    let nt = rayon::current_num_threads();
    assert_eq!(
        slots.len(),
        nt,
        "workspace made for {} threads used in a pool of {nt} threads",
        slots.len()
    );
    let next = AtomicUsize::new(0);
    let base = SlotPtr(slots.as_mut_ptr());
    rayon::broadcast(|bc| {
        // SAFETY: broadcast runs the closure once on each pool thread, with distinct indices
        // below nt = slots.len(), so every slot is borrowed by one thread only
        let slot = unsafe { &mut *base.get().add(bc.index()) };
        worker(slot, Tasks { order, next: &next });
    });
}

/// Fold the tasks 0..n into the caller's accumulators `accs`, one per chunk (a chunk count
/// fixed by the problem, `linus_num::det::nchunk_for`): task i goes to chunk i mod nchunk,
/// each chunk folds its tasks in order, and the chunks are combined into `accs[0]` in chunk
/// order. For [`Reduction::Ordered`] digesters.
pub fn ordered<A: Send>(
    n: usize,
    accs: &mut [A],
    body: impl Fn(&mut A, usize) + Sync + Send,
    combine: impl FnMut(&mut A, &A),
) {
    linus_num::det::par_fold_det_into(n, accs, body, combine);
}

/// The one dispatcher of the Ordered schedules with per-thread scratch: the chunks 0..nc go
/// to the current pool's threads from one shared counter, each run by `chunk(scratch, c)` on
/// the scratch of the thread that takes it. A chunk's result must depend on its own tasks and
/// their order only (the scratch carries nothing from one task to the next); then the result
/// has the same bits for any thread count and schedule.
///
/// # Panics
/// When `scratch` has fewer slots than the pool has threads.
fn dispatch<S: Send>(nc: usize, scratch: &mut [S], chunk: impl Fn(&mut S, usize) + Sync) {
    let nt = rayon::current_num_threads();
    assert!(
        scratch.len() >= nt,
        "a workspace with scratch for {} threads used in a pool of {nt} threads",
        scratch.len()
    );
    let next = AtomicUsize::new(0);
    let sp = SlotPtr(scratch.as_mut_ptr());
    rayon::broadcast(|bc| {
        // SAFETY: broadcast runs the closure once on each pool thread, with distinct indices
        // below nt <= scratch.len()
        let s = unsafe { &mut *sp.get().add(bc.index()) };
        loop {
            let c = next.fetch_add(1, Ordering::Relaxed);
            if c >= nc {
                break;
            }
            chunk(s, c);
        }
    });
}

/// Fold the tasks 0..n, cut into contiguous chunks of `n.div_ceil(nch)` tasks (the last one
/// shorter; at most `nch` chunks, sized from the problem only, as
/// `linus_num::det::ordered_chunks`), into one partial result per chunk: chunk c folds its range
/// of tasks in order into `parts[c]` with the scratch of the thread that takes it (one
/// dispatcher for all three forms). Returns the number of chunks; the caller adds
/// `parts[..nchunk]` up in chunk order. For [`Reduction::Ordered`] digesters whose chunks
/// read consecutive tasks (slices of a pair list). `body` starts the partial from zero and resets any accumulating scratch.
///
/// # Panics
/// When `scratch` has fewer slots than the pool has threads, or `parts` fewer than the chunk
/// count.
pub fn ordered_ranges<S: Send, P: Send>(
    n: usize,
    nch: usize,
    scratch: &mut [S],
    parts: &mut [P],
    body: impl Fn(&mut S, &mut P, Range<usize>) + Sync,
) -> usize {
    let per = n.div_ceil(nch.max(1)).max(1);
    let nc = n.div_ceil(per);
    assert!(
        parts.len() >= nc,
        "{} partials for {nc} chunks",
        parts.len()
    );
    let pp = SlotPtr(parts.as_mut_ptr());
    dispatch(nc, scratch, |s, c| {
        // SAFETY: the dispatcher hands chunk c (< nc <= parts.len()) to one thread only
        let p = unsafe { &mut *pp.get().add(c) };
        body(s, p, c * per..((c + 1) * per).min(n));
    });
    nc
}

/// [`ordered`] with per-thread scratch: task i goes to chunk i mod nchunk (`parts.len()`
/// chunks, fixed by the problem), chunk c folds its tasks in order into `parts[c]` from the
/// state the caller left it in (zeroed, for a sum) with the scratch of the thread that takes
/// it (one dispatcher for all three forms). The caller adds the partials up in chunk order. For
/// [`Reduction::Ordered`] digesters whose scratch is large (the lane kernels): the scratch is
/// per thread, the partials per chunk.
///
/// # Panics
/// When `scratch` has fewer slots than the pool has threads.
pub fn ordered_strided<S: Send, P: Send>(
    n: usize,
    scratch: &mut [S],
    parts: &mut [P],
    body: impl Fn(&mut S, &mut P, usize) + Sync,
) {
    let nc = parts.len();
    let pp = SlotPtr(parts.as_mut_ptr());
    dispatch(nc, scratch, |s, c| {
        // SAFETY: the dispatcher hands chunk c (< nc = parts.len()) to one thread only
        let p = unsafe { &mut *pp.get().add(c) };
        let mut i = c;
        while i < n {
            body(s, p, i);
            i += nc;
        }
    });
}

/// Independent tasks, one per part: task i writes `parts[i]` only, with the scratch of the
/// thread that takes it (the one dispatcher). Nothing is reduced, so the result does not depend
/// on the schedule (projections of shells into their own rows, say).
///
/// # Panics
/// When `scratch` has fewer slots than the pool has threads.
pub fn per_part<S: Send, P: Send>(
    scratch: &mut [S],
    parts: &mut [P],
    body: impl Fn(&mut S, &mut P, usize) + Sync,
) {
    ordered_strided(parts.len(), scratch, parts, body);
}

/// [`ordered_strided`] over `nchunk` flat accumulators of `len` values carved from one pool
/// (one pool serves the operations of a plan with different accumulator lengths and chunk
/// counts): chunk c zeroes `pool[c * len..(c + 1) * len]` and folds the tasks c, c + nchunk,
/// ... into it in order, and the chunks are then added into the first in chunk order (the
/// result is `pool[..len]`). The same chunking, order and sums as [`ordered`] with
/// accumulators that start at zero. For [`Reduction::Ordered`] digesters.
///
/// # Panics
/// When the pool holds fewer than `nchunk` accumulators, or `scratch` has fewer slots than the
/// pool has threads.
pub fn ordered_pool<S: Send>(
    n: usize,
    pool: &mut [f64],
    len: usize,
    nchunk: usize,
    scratch: &mut [S],
    body: impl Fn(&mut [f64], &mut S, usize) + Sync,
) {
    assert!(
        nchunk >= 1 && pool.len() >= nchunk * len,
        "the workspace's pool holds {} values, not {nchunk} accumulators of {len}",
        pool.len()
    );
    let pool = &mut pool[..nchunk * len];
    let pp = SlotPtr(pool.as_mut_ptr());
    dispatch(nchunk, scratch, |s, c| {
        // SAFETY: the dispatcher hands chunk c (< nchunk) to one thread only, and the chunks'
        // accumulators are disjoint ranges of the pool
        let acc = unsafe { std::slice::from_raw_parts_mut(pp.get().add(c * len), len) };
        acc.fill(0.0);
        let mut i = c;
        while i < n {
            body(acc, s, i);
            i += nchunk;
        }
    });
    let (first, rest) = pool.split_at_mut(len);
    for p in rest.chunks_exact(len.max(1)) {
        linus_num::det::add_into(first, p);
    }
}

/// A pointer to the slots shared by the threads of [`exact`], each using its own.
struct SlotPtr<T>(*mut T);
impl<T> Clone for SlotPtr<T> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<T> Copy for SlotPtr<T> {}
// SAFETY: each thread dereferences a distinct slot (see `exact`)
unsafe impl<T: Send> Send for SlotPtr<T> {}
unsafe impl<T: Send> Sync for SlotPtr<T> {}
impl<T> SlotPtr<T> {
    #[inline]
    fn get(self) -> *mut T {
        self.0
    }
}
