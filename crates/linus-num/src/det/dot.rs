//! Fixed-chunk dot products: dot products and linear combinations of long vectors whose bits do
//! not depend on the thread count or on the machine's SIMD level.
//!
//! A vector is cut into chunks of [`DOT_CHUNK`] elements (fixed, not derived from the thread
//! count). Each chunk's dot product is four interleaved partial sums in index order, added
//! (s0 + s1) + (s2 + s3), then the tail, with plain multiplies and adds (no fused
//! multiply-add), so a chunk gives the same value on every x86-64 SIMD level and with any
//! vectorization. The chunks' values are added in chunk order. Chunks may run in parallel; the
//! order of the additions never changes.
//!
//! The block Krylov solver of linear response orthonormalizes its new directions with these
//! (linus's `response::solver`).

use faer::{Mat, Par};
use rayon::prelude::*;
use std::ops::Range;

/// Elements per chunk of the fixed-chunk reductions ([`dot`], [`col_dots`], [`sub_cols`]).
pub const DOT_CHUNK: usize = 4096;

/// The dot product of one chunk (see the module docs).
fn chunk_dot(a: &[f64], b: &[f64]) -> f64 {
    let (a4, at) = a.as_chunks::<4>();
    let (b4, bt) = b.as_chunks::<4>();
    let mut s = [0.0f64; 4];
    for (x, y) in a4.iter().zip(b4) {
        for l in 0..4 {
            s[l] += x[l] * y[l];
        }
    }
    let mut t = (s[0] + s[1]) + (s[2] + s[3]);
    for (x, y) in at.iter().zip(bt) {
        t += x * y;
    }
    t
}

/// a . b as the chunks' dot products added in chunk order: the same value at any parallelism.
pub fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.chunks(DOT_CHUNK)
        .zip(b.chunks(DOT_CHUNK))
        .fold(0.0, |s, (x, y)| s + chunk_dot(x, y))
}

/// q_i . z for the columns i in `cols` of `q`, each as [`dot`] (the chunks in parallel when
/// `par` has more than one thread, their partial sums added in chunk order).
pub fn col_dots(q: &Mat<f64>, cols: Range<usize>, z: &[f64], par: Par) -> Vec<f64> {
    let nch = z.len().div_ceil(DOT_CHUNK).max(1);
    let part = |c: usize| -> Vec<f64> {
        let (r0, r1) = (c * DOT_CHUNK, ((c + 1) * DOT_CHUNK).min(z.len()));
        cols.clone()
            .map(|i| chunk_dot(&q.col_as_slice(i)[r0..r1], &z[r0..r1]))
            .collect()
    };
    let parts: Vec<Vec<f64>> = if par.degree() > 1 {
        (0..nch).into_par_iter().map(part).collect()
    } else {
        (0..nch).map(part).collect()
    };
    (0..cols.len())
        .map(|i| parts.iter().fold(0.0, |s, p| s + p[i]))
        .collect()
}

/// z <- z - sum_i p_i q_i over the columns i in `cols` of `q` (`p[i - cols.start]`), each
/// element's terms subtracted in the order of i (rows in parallel chunks when `par` has more
/// than one thread: the same values at any parallelism).
pub fn sub_cols(z: &mut [f64], q: &Mat<f64>, cols: Range<usize>, p: &[f64], par: Par) {
    let cols: Vec<&[f64]> = cols.map(|i| q.col_as_slice(i)).collect();
    let f = |(c, zc): (usize, &mut [f64])| {
        let (r0, nz) = (c * DOT_CHUNK, zc.len());
        for (i, col) in cols.iter().enumerate() {
            let pi = p[i];
            for (t, s) in zc.iter_mut().zip(&col[r0..r0 + nz]) {
                *t -= pi * s;
            }
        }
    };
    if par.degree() > 1 {
        z.par_chunks_mut(DOT_CHUNK).enumerate().for_each(f);
    } else {
        z.chunks_mut(DOT_CHUNK).enumerate().for_each(f);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Fixed pseudo-random numbers in [-0.5, 0.5) (a 64-bit LCG).
    fn noise(seed: u64, len: usize) -> Vec<f64> {
        let mut s = seed.wrapping_mul(6364136223846793005);
        (0..len)
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((s >> 11) as f64) / ((1u64 << 53) as f64) - 0.5
            })
            .collect()
    }

    /// The reductions spelled out sequentially, element by element, in the documented order.
    fn chunk_dot_spelled(a: &[f64], b: &[f64]) -> f64 {
        let n4 = a.len() / 4 * 4;
        let mut s = [0.0f64; 4];
        for j in (0..n4).step_by(4) {
            for l in 0..4 {
                s[l] += a[j + l] * b[j + l];
            }
        }
        let mut t = (s[0] + s[1]) + (s[2] + s[3]);
        for j in n4..a.len() {
            t += a[j] * b[j];
        }
        t
    }

    #[test]
    fn the_reductions_have_the_same_bits_at_any_parallelism() {
        // lengths below, at and across the chunk size, with tails of every length mod 4
        for len in [0, 1, 3, 4, 7, 4096, 4097, 9000, 3 * 4096 + 2] {
            let a = noise(1, len);
            let b = noise(2, len);
            let want = a
                .chunks(DOT_CHUNK)
                .zip(b.chunks(DOT_CHUNK))
                .fold(0.0, |s, (x, y)| s + chunk_dot_spelled(x, y));
            assert_eq!(dot(&a, &b).to_bits(), want.to_bits(), "dot, len {len}");
            let ncol = 5;
            let qv = noise(3, len * ncol);
            let q = Mat::from_fn(len, ncol, |i, j| qv[j * len + i]);
            let p: Vec<f64> = noise(4, ncol);
            let mut first: Option<(Vec<f64>, Vec<f64>)> = None;
            for threads in [1, 3, 8] {
                let pool = rayon::ThreadPoolBuilder::new()
                    .num_threads(threads)
                    .build()
                    .unwrap();
                let par = if threads == 1 {
                    Par::Seq
                } else {
                    Par::rayon(threads)
                };
                let (d, z) = pool.install(|| {
                    let d = col_dots(&q, 1..ncol, &b, par);
                    let mut z = b.clone();
                    sub_cols(&mut z, &q, 1..ncol, &p[1..], par);
                    (d, z)
                });
                for (i, x) in d.iter().enumerate() {
                    assert_eq!(
                        x.to_bits(),
                        dot(q.col_as_slice(i + 1), &b).to_bits(),
                        "col_dots, len {len}, column {}, {threads} threads",
                        i + 1
                    );
                }
                for (r, x) in z.iter().enumerate() {
                    let mut t = b[r];
                    for i in 1..ncol {
                        t -= p[i] * q[(r, i)];
                    }
                    assert_eq!(x.to_bits(), t.to_bits(), "sub_cols, len {len}, row {r}");
                }
                match &first {
                    None => first = Some((d, z)),
                    Some((d0, z0)) => assert!(d0 == &d && z0 == &z),
                }
            }
        }
    }
}
