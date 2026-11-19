//! Ordered strided chunks: deterministic parallel folds of floating-point accumulators. The
//! result does not depend on the number of threads or on the work-stealing schedule (rayon's
//! `fold`/`reduce` sum floating-point partial results in a schedule-dependent order).
//!
//! Items are dealt to a fixed number of strided chunks (item i to chunk i mod nchunk, so
//! expensive and cheap items spread evenly), each chunk is folded sequentially in item order
//! into its own accumulator, chunks run in parallel, and the chunk accumulators are combined
//! in chunk order. The chunk count must depend on the problem only ([`nchunk_for`]).
//! [`ordered_chunks`] cuts the items into contiguous ranges instead, for folds that read
//! consecutive items (rows, slices of a pair list).

use rayon::prelude::*;
use std::ops::Range;

/// Chunk count for `n_items` items with accumulators of `acc_len` doubles: up to 64, fewer
/// when 64 copies would exceed `budget` doubles in all (at least 2 when there are 2 items).
/// Depends on the problem size only.
pub fn nchunk_for(n_items: usize, acc_len: usize, budget: usize) -> usize {
    (budget / acc_len.max(1)).clamp(2, 64).min(n_items).max(1)
}

/// Default memory budget for the chunk accumulators of [`nchunk_for`] (2^25 doubles, 256 MB).
pub const DEFAULT_BUDGET: usize = 1 << 25;

/// Deterministic parallel fold over items 0..n (see the module docs): `init` creates a chunk
/// accumulator (with any scratch it needs), `body` folds one item into it, `combine` adds a
/// later chunk's accumulator into the running total.
pub fn par_fold_det<A, I, B, C>(n: usize, nchunk: usize, init: I, body: B, mut combine: C) -> A
where
    A: Send,
    I: Fn() -> A + Sync + Send,
    B: Fn(&mut A, usize) + Sync + Send,
    C: FnMut(&mut A, A),
{
    let nchunk = nchunk.max(1);
    let parts: Vec<A> = (0..nchunk)
        .into_par_iter()
        .with_max_len(1)
        .map(|c| {
            let mut a = init();
            let mut i = c;
            while i < n {
                body(&mut a, i);
                i += nchunk;
            }
            a
        })
        .collect();
    let mut it = parts.into_iter();
    let mut acc = it.next().unwrap_or_else(&init);
    for p in it {
        combine(&mut acc, p);
    }
    acc
}

/// [`par_fold_det`] over caller-owned accumulators, one per chunk (`accs.len()` chunks):
/// chunk c folds items c, c + nchunk, ... in order into `accs[c]` from the state the caller
/// left it in (zeroed, for a sum), the chunks run in parallel, and `accs[1..]` are then
/// combined into `accs[0]` in chunk order. The same chunking and order as [`par_fold_det`]
/// with that chunk count, so the same bits for accumulators that start as `init()` would;
/// no allocation of its own.
pub fn par_fold_det_into<A, B, C>(n: usize, accs: &mut [A], body: B, mut combine: C)
where
    A: Send,
    B: Fn(&mut A, usize) + Sync + Send,
    C: FnMut(&mut A, &A),
{
    let nchunk = accs.len();
    assert!(
        nchunk >= 1,
        "par_fold_det_into needs at least one accumulator"
    );
    accs.par_iter_mut()
        .with_max_len(1)
        .enumerate()
        .for_each(|(c, a)| {
            let mut i = c;
            while i < n {
                body(a, i);
                i += nchunk;
            }
        });
    let (first, rest) = accs.split_first_mut().expect("nonempty");
    for p in rest.iter() {
        combine(first, p);
    }
}

/// The contiguous chunks of the items `0..n` for an ordered reduction, as a parallel iterator of
/// ranges in chunk order: `n.div_ceil(nch)` items per chunk (the last one shorter), so at most
/// `nch` chunks, sized from the problem only. Fold each range into its own partial result and
/// add the partials in chunk order (`collect` keeps the order): the result does not depend on
/// the thread count. [`par_fold_det`] deals the items to strided chunks instead, which spreads
/// uneven items better; contiguous chunks keep each partial's items together (slices of the
/// item list).
pub fn ordered_chunks(n: usize, nch: usize) -> impl IndexedParallelIterator<Item = Range<usize>> {
    let per = n.div_ceil(nch.max(1)).max(1);
    (0..n.div_ceil(per))
        .into_par_iter()
        .map(move |c| c * per..((c + 1) * per).min(n))
}

/// `a += b` elementwise (a combine step for plain vectors).
#[inline]
pub fn add_into(a: &mut [f64], b: &[f64]) {
    for (x, y) in a.iter_mut().zip(b) {
        *x += y;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ordered_chunks_tile_in_order() {
        for (n, nch) in [(0, 4), (1, 4), (10, 4), (12, 4), (1000, 192), (5, 256)] {
            let r: Vec<_> = ordered_chunks(n, nch).collect();
            assert!(r.len() <= nch.max(1));
            let flat: Vec<usize> = r.iter().flat_map(|c| c.clone()).collect();
            assert_eq!(flat, (0..n).collect::<Vec<_>>(), "{n} {nch}");
        }
        assert_eq!(
            ordered_chunks(10, 4).collect::<Vec<_>>(),
            [0..3, 3..6, 6..9, 9..10]
        );
    }

    #[test]
    fn independent_of_thread_count() {
        let vals: Vec<f64> = (0..10000)
            .map(|i| ((i * 7919 % 1013) as f64 - 500.0) * 10f64.powi(i % 7 - 3))
            .collect();
        let run = |t: usize| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(t)
                .build()
                .unwrap()
                .install(|| {
                    par_fold_det(
                        vals.len(),
                        37,
                        || vec![0.0; 3],
                        |a, i| a[i % 3] += vals[i],
                        |a, b| add_into(a, &b),
                    )
                })
        };
        let r1 = run(1);
        for t in [2, 5, 8] {
            assert_eq!(run(t), r1);
        }
    }

    /// par_fold_det_into over zeroed accumulators has par_fold_det's bits, at any thread count.
    #[test]
    fn into_matches_par_fold_det() {
        let vals: Vec<f64> = (0..10000)
            .map(|i| ((i * 7919 % 1013) as f64 - 500.0) * 10f64.powi(i % 7 - 3))
            .collect();
        let pool = |t: usize| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(t)
                .build()
                .unwrap()
        };
        let reference = pool(3).install(|| {
            par_fold_det(
                vals.len(),
                37,
                || vec![0.0; 3],
                |a, i| a[i % 3] += vals[i],
                |a, b| add_into(a, &b),
            )
        });
        for t in [1, 2, 5, 8] {
            let mut accs = vec![vec![0.0; 3]; 37];
            pool(t).install(|| {
                par_fold_det_into(
                    vals.len(),
                    &mut accs,
                    |a, i| a[i % 3] += vals[i],
                    |a, b| add_into(a, b),
                )
            });
            let same = accs[0]
                .iter()
                .zip(&reference)
                .all(|(x, y)| x.to_bits() == y.to_bits());
            assert!(same, "{t} threads: {:?} vs {:?}", accs[0], reference);
        }
    }
}
