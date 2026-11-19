// SPDX-License-Identifier: LGPL-3.0-or-later
//! Deterministic parallel reductions: the work is cut into chunks that depend only on the
//! problem size, every chunk is reduced on its own, and the partial results are summed in
//! chunk order. The result is therefore bitwise identical for any rayon pool size.

use rayon::prelude::*;

/// Maximum number of chunks of a reduction (also the upper bound of useful threads).
pub(crate) const NCHUNK: usize = 64;

/// Split `0..n` into at most [`NCHUNK`] contiguous ranges of about equal `cost(i)`.
pub(crate) fn balanced_ranges(n: usize, cost: impl Fn(usize) -> f64) -> Vec<(usize, usize)> {
    if n == 0 {
        return Vec::new();
    }
    let total: f64 = (0..n).map(&cost).sum::<f64>().max(1.0);
    let nchunk = NCHUNK.min(n);
    let mut out = Vec::with_capacity(nchunk);
    let (mut start, mut acc) = (0usize, 0.0f64);
    for i in 0..n {
        acc += cost(i);
        if out.len() + 1 < nchunk && acc >= total * (out.len() + 1) as f64 / nchunk as f64 {
            out.push((start, i + 1));
            start = i + 1;
        }
    }
    if start < n {
        out.push((start, n));
    }
    out
}

/// Run `f` on every range in parallel and fold the results in range order with `add`.
pub(crate) fn reduce_ranges<T: Send>(
    ranges: &[(usize, usize)],
    f: impl Fn(usize, usize) -> T + Sync,
    mut add: impl FnMut(T),
) {
    let parts: Vec<T> = ranges.par_iter().map(|&(a, b)| f(a, b)).collect();
    for p in parts {
        add(p);
    }
}

/// `C += A B` for row-major dense matrices (`A`: m x k, `B`: k x n, `C`: m x n), rows of C
/// in parallel; each element is accumulated by one thread in a fixed order.
pub(crate) fn gemm_acc(m: usize, k: usize, n: usize, a: &[f64], b: &[f64], c: &mut [f64]) {
    assert!(a.len() >= m * k && b.len() >= k * n && c.len() >= m * n);
    c[..m * n]
        .par_chunks_mut(n)
        .enumerate()
        .for_each(|(i, crow)| {
            let arow = &a[i * k..(i + 1) * k];
            for (l, &x) in arow.iter().enumerate() {
                if x != 0.0 {
                    let brow = &b[l * n..(l + 1) * n];
                    for (cv, bv) in crow.iter_mut().zip(brow) {
                        *cv += x * bv;
                    }
                }
            }
        });
}

/// Row-major transpose of an m x n matrix.
pub(crate) fn transpose(m: usize, n: usize, a: &[f64]) -> Vec<f64> {
    let mut t = vec![0.0; m * n];
    for i in 0..m {
        for j in 0..n {
            t[j * m + i] = a[i * n + j];
        }
    }
    t
}

/// `H += X + X^T` for a square row-major matrix.
pub(crate) fn add_sym(n: usize, x: &[f64], h: &mut [f64]) {
    h[..n * n]
        .par_chunks_mut(n)
        .enumerate()
        .for_each(|(i, row)| {
            for (j, v) in row.iter_mut().enumerate() {
                *v += x[i * n + j] + x[j * n + i];
            }
        });
}
