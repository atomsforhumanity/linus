//! Shape-blocked products: a parallel matrix product whose rounding does not depend on the
//! thread count. faer's parallel GEMM splits its work by the thread count, and with it the
//! order of the partial sums; here the output is cut into blocks sized from the shape only,
//! and each block is one sequential GEMM.

use faer::{Accum, MatMut, MatRef, Par};
use rayon::prelude::*;

/// dst = alpha A B (+ dst with Accum::Add) independent of the thread count: the output is cut
/// into at most 64 blocks along its larger dimension, sized from the shape only, each a
/// sequential GEMM (faer's parallel GEMM rounds differently with the thread count).
pub fn det_matmul(
    dst: MatMut<'_, f64>,
    acc: Accum,
    a: MatRef<'_, f64>,
    b: MatRef<'_, f64>,
    alpha: f64,
) {
    let (m, n) = (dst.nrows(), dst.ncols());
    if m == 0 || n == 0 {
        return;
    }
    if n >= m {
        let cw = n.div_ceil(64).max(16);
        dst.par_col_chunks_mut(cw).enumerate().for_each(|(i, d)| {
            let w = d.ncols();
            seq_matmul(d, acc, a, b.subcols(i * cw, w), alpha);
        });
    } else {
        let rw = m.div_ceil(64).max(16);
        dst.par_row_chunks_mut(rw).enumerate().for_each(|(i, d)| {
            let h = d.nrows();
            seq_matmul(d, acc, a.subrows(i * rw, h), b, alpha);
        });
    }
}

/// One sequential faer GEMM (the thread count cannot reach it).
#[inline]
// faer's matmul is disallowed outside linus's util::linalg for its parallel modes (clippy.toml);
// a sequential product is thread-count independent by construction
#[allow(clippy::disallowed_methods)]
fn seq_matmul(
    dst: MatMut<'_, f64>,
    acc: Accum,
    a: MatRef<'_, f64>,
    b: MatRef<'_, f64>,
    alpha: f64,
) {
    faer::linalg::matmul::matmul(dst, acc, a, b, alpha, Par::Seq);
}

#[cfg(test)]
mod tests {
    use super::*;
    use faer::Mat;

    #[test]
    fn independent_of_thread_count() {
        let a = Mat::from_fn(70, 300, |i, j| ((i * 31 + j * 17) % 23) as f64 * 0.1 - 1.0);
        let b = Mat::from_fn(300, 90, |i, j| ((i * 13 + j * 7) % 19) as f64 * 0.05 - 0.4);
        let run = |t: usize| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(t)
                .build()
                .unwrap()
                .install(|| {
                    let mut c = Mat::<f64>::zeros(70, 90);
                    det_matmul(c.as_mut(), Accum::Replace, a.as_ref(), b.as_ref(), 0.5);
                    c
                })
        };
        let c1 = run(1);
        for t in [3, 8] {
            assert!(run(t) == c1);
        }
    }
}
