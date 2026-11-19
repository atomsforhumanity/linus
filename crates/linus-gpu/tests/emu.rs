//! The Ozaki-scheme FP64 emulation (`linus_gpu::emu`): the product plan on the host; on the
//! device the INT8 GEMM against exact integer products, and the emulated GEMMs against a
//! double-double host reference, with their error bound, bitwise reproducibility and the
//! symmetric (lower-triangle) form.

mod common;

use linus_gpu::emu::{diagonal, groups, INT32_TERMS};

/// Exact (double-double) dot product: Ogita-Rump-Oishi Dot2, error ~2^-106 relative.
#[cfg_attr(not(feature = "gpu"), allow(dead_code))]
fn dot2(a: impl Iterator<Item = f64>, b: impl Iterator<Item = f64>) -> f64 {
    let (mut s, mut c) = (0.0f64, 0.0f64);
    for (x, y) in a.zip(b) {
        let p = x * y;
        let pe = x.mul_add(y, -p);
        let t = s + p;
        let z = t - s;
        let se = (s - (t - z)) + (p - z);
        s = t;
        c += pe + se;
    }
    s + c
}

#[test]
fn diagonals_cover_the_triangle_and_groups_bound_int32() {
    for s in 1..=8 {
        let mut all = vec![];
        for g in 0..s {
            let d = diagonal(g, s, s);
            assert_eq!(d.len(), g + 1);
            assert!(d.iter().all(|&(p, q)| p + q == g && p < s && q < s));
            all.extend(d);
        }
        assert_eq!(all.len(), s * (s + 1) / 2);
    }
    // unequal slice counts: only digits that exist
    assert_eq!(diagonal(3, 2, 7), vec![(0, 3), (1, 2)]);
    let d = diagonal(6, 7, 7);
    for k in [1, 1000, 30000, 40000, INT32_TERMS] {
        let gs = groups(&d, k);
        assert_eq!(gs.iter().map(|g| g.len()).sum::<usize>(), 7);
        assert!(gs
            .iter()
            .all(|g| g.len() * k <= INT32_TERMS || g.len() == 1));
    }
}

/// The host model of the device digits: x 2^(8s-1-e) rounded, cut into signed bytes.
fn digits(x: f64, e: i32, s: usize) -> Vec<i64> {
    let mut i = (x * 2f64.powi(8 * s as i32 - 1 - e)).round_ties_even() as i64;
    let mut d = vec![0i64; s];
    for p in (1..s).rev() {
        let mut v = i & 0xff;
        if v > 127 {
            v -= 256;
        }
        i = (i - v) >> 8;
        d[p] = v;
    }
    d[0] = i;
    d
}

#[test]
fn digit_model_is_exact_and_in_range() {
    let mut rng = common::Rng(11);
    for s in 1..=7usize {
        for _ in 0..20000 {
            let x = rng.wide();
            // a vector maximum at or above |x|, its exponent as the kernel picks it
            let mx = x.abs() * (1.0 + 3.0 * rng.uniform());
            let (f, mut e) = frexp(mx);
            if f > 127.0 / 128.0 {
                e += 1;
            }
            let d = digits(x, e, s);
            assert!(d[0].abs() <= 127 && d[1..].iter().all(|v| (-128..=127).contains(v)));
            let back: f64 = d
                .iter()
                .enumerate()
                .map(|(p, &v)| v as f64 * 2f64.powi(e - 7 - 8 * p as i32))
                .sum();
            // rounding to 8s - 1 fractional bits of the vector's scale
            assert!((back - x).abs() <= 2f64.powi(e - (8 * s as i32 - 1) - 1) * (1.0 + 1e-15));
        }
    }
}

fn frexp(x: f64) -> (f64, i32) {
    if x == 0.0 {
        return (0.0, 0);
    }
    let e = x.abs().log2().floor() as i32 + 1;
    let mut f = x / 2f64.powi(e);
    let mut e = e;
    if f.abs() >= 1.0 {
        f /= 2.0;
        e += 1;
    } else if f.abs() < 0.5 {
        f *= 2.0;
        e -= 1;
    }
    (f, e)
}

#[cfg(feature = "gpu")]
mod gpu {
    use super::common::{self, Rng};
    use super::dot2;
    use linus_gpu::blas::{Blas, GemmDims, MatMut, MatRef, Op};
    use linus_gpu::emu::{Order, Ozaki, OzakiWork, Sliced, Vectors};

    #[test]
    fn int8_gemm_is_exact() {
        let Some(gpu) = common::gpu() else { return };
        let blas = Blas::new(&gpu).unwrap();
        let mut rng = Rng(3);
        // TN layout (both operands contiguous along k), k a multiple of 4
        // m a multiple of 16: cuBLAS 13.2 refuses some other INT8 output row counts (emu::ROW_ALIGN)
        for &(m, n, k) in &[(16usize, 3usize, 4usize), (48, 23, 64), (128, 65, 1000)] {
            let a: Vec<i8> = (0..k * m)
                .map(|_| (rng.next_u64() % 256) as u8 as i8)
                .collect();
            let b: Vec<i8> = (0..k * n)
                .map(|_| (rng.next_u64() % 256) as u8 as i8)
                .collect();
            let c0: Vec<i32> = (0..m * n)
                .map(|_| (rng.next_u64() % 1000) as i32 - 500)
                .collect();
            let (da, db) = (gpu.upload(&a).unwrap(), gpu.upload(&b).unwrap());
            let mut dc = gpu.upload(&c0).unwrap();
            blas.gemm_i8(
                GemmDims::new(Op::T, Op::N, m, n, k),
                (2, -1),
                MatRef::new(&da, 0, k),
                MatRef::new(&db, 0, k),
                MatMut::new(&mut dc, 0, m),
            )
            .unwrap();
            let c = gpu.download(&dc).unwrap();
            for j in 0..n {
                for i in 0..m {
                    let s: i64 = (0..k)
                        .map(|l| a[i * k + l] as i64 * b[j * k + l] as i64)
                        .sum();
                    assert_eq!(
                        c[j * m + i] as i64,
                        2 * s - c0[j * m + i] as i64,
                        "({m},{n},{k})"
                    );
                }
            }
        }
        // misaligned leading dimension is refused
        let da = gpu.alloc_zeros::<i8>(64).unwrap();
        let mut dc = gpu.alloc_zeros::<i32>(16).unwrap();
        assert!(blas
            .gemm_i8(
                GemmDims::new(Op::T, Op::N, 2, 2, 3),
                (1, 0),
                MatRef::new(&da, 0, 3),
                MatRef::new(&da, 0, 3),
                MatMut::new(&mut dc, 0, 2)
            )
            .is_err());
    }

    /// Column-major m x n of op(A) op(B) for a stored as (ta ? k x m : m x k), b likewise.
    #[allow(clippy::too_many_arguments)]
    fn reference(
        ta: Op,
        tb: Op,
        m: usize,
        n: usize,
        k: usize,
        a: &[f64],
        lda: usize,
        b: &[f64],
        ldb: usize,
    ) -> Vec<f64> {
        let av = |i: usize, l: usize| {
            if ta == Op::T {
                a[i * lda + l]
            } else {
                a[l * lda + i]
            }
        };
        let bv = |l: usize, j: usize| {
            if tb == Op::N {
                b[j * ldb + l]
            } else {
                b[l * ldb + j]
            }
        };
        let mut c = vec![0.0; m * n];
        for j in 0..n {
            for i in 0..m {
                c[j * m + i] = dot2((0..k).map(|l| av(i, l)), (0..k).map(|l| bv(l, j)));
            }
        }
        c
    }

    /// max over elements of |c - ref| / (max_l |a_il| max_l |b_lj| k): the fixed-point bound's
    /// scale.
    #[allow(clippy::too_many_arguments)]
    fn scaled_error(
        ta: Op,
        tb: Op,
        m: usize,
        n: usize,
        k: usize,
        a: &[f64],
        lda: usize,
        b: &[f64],
        ldb: usize,
        c: &[f64],
        r: &[f64],
    ) -> f64 {
        let av = |i: usize, l: usize| {
            if ta == Op::T {
                a[i * lda + l]
            } else {
                a[l * lda + i]
            }
        };
        let bv = |l: usize, j: usize| {
            if tb == Op::N {
                b[j * ldb + l]
            } else {
                b[l * ldb + j]
            }
        };
        let ra: Vec<f64> = (0..m)
            .map(|i| (0..k).map(|l| av(i, l).abs()).fold(0.0, f64::max))
            .collect();
        let rb: Vec<f64> = (0..n)
            .map(|j| (0..k).map(|l| bv(l, j).abs()).fold(0.0, f64::max))
            .collect();
        let mut worst = 0.0f64;
        for j in 0..n {
            for i in 0..m {
                let scale = ra[i] * rb[j] * k as f64;
                if scale > 0.0 {
                    worst = worst.max((c[j * m + i] - r[j * m + i]).abs() / scale);
                } else {
                    assert_eq!(c[j * m + i], 0.0);
                }
            }
        }
        worst
    }

    #[test]
    fn emulated_gemm_meets_the_fixed_point_bound() {
        let Some(gpu) = common::gpu() else { return };
        let oz = Ozaki::new(&gpu).unwrap();
        let mut work = OzakiWork::default();
        let mut rng = Rng(7);
        for &(ta, tb) in &[
            (Op::T, Op::N),
            (Op::N, Op::N),
            (Op::T, Op::T),
            (Op::N, Op::T),
        ] {
            let (m, n, k) = (45usize, 21usize, 300usize);
            let (lda, ldb) = (
                if ta == Op::T { k + 3 } else { m + 2 },
                if tb == Op::N { k + 1 } else { n + 5 },
            );
            let (acols, bcols) = (
                if ta == Op::T { m } else { k },
                if tb == Op::N { n } else { k },
            );
            // values over 17 decades, plus an all-zero vector in each operand
            let mut a: Vec<f64> = (0..lda * acols).map(|_| rng.wide()).collect();
            let mut b: Vec<f64> = (0..ldb * bcols).map(|_| rng.wide()).collect();
            for l in 0..k {
                if ta == Op::T {
                    a[4 * lda + l] = 0.0
                } else {
                    a[l * lda + 4] = 0.0
                }
                if tb == Op::N {
                    b[2 * ldb + l] = 0.0
                } else {
                    b[l * ldb + 2] = 0.0
                }
            }
            let r = reference(ta, tb, m, n, k, &a, lda, &b, ldb);
            let (da, db) = (gpu.upload(&a).unwrap(), gpu.upload(&b).unwrap());
            for s in [3usize, 5, 7] {
                let c0: Vec<f64> = (0..m * n).map(|_| rng.uniform() - 0.5).collect();
                let mut dc = gpu.upload(&c0).unwrap();
                oz.dgemm_at(
                    &gpu,
                    s,
                    ta,
                    tb,
                    m,
                    n,
                    k,
                    0.5,
                    (&da, 0, lda),
                    (&db, 0, ldb),
                    2.0,
                    (&mut dc, 0, m),
                    &mut work,
                )
                .unwrap();
                let c = gpu.download(&dc).unwrap();
                // undo alpha, beta exactly (powers of two)
                let c: Vec<f64> = c
                    .iter()
                    .zip(&c0)
                    .map(|(x, y)| (x - 2.0 * y) * 2.0)
                    .collect();
                let err = scaled_error(ta, tb, m, n, k, &a, lda, &b, ldb, &c, &r);
                // each operand within 2^-(8s-1) of its vector's max, plus the dropped
                // diagonals and the FP64 recombination
                let bound = 4.0 * 2f64.powi(-(8 * s as i32 - 1)) + 1e-15;
                assert!(err <= bound, "{ta:?}{tb:?} s={s}: {err:e} > {bound:e}");
                if s == 7 {
                    assert!(err < 5e-16, "{ta:?}{tb:?}: {err:e}");
                }
            }
        }
    }

    #[test]
    fn large_inner_dimension_splits_the_int32_sums() {
        let Some(gpu) = common::gpu() else { return };
        let oz = Ozaki::new(&gpu).unwrap();
        let mut work = OzakiWork::default();
        let mut rng = Rng(9);
        // k = 50000: at most 2 products of a diagonal per INT32 accumulator; values of one
        // sign so an overflow would show
        let (m, n, k) = (6usize, 4usize, 50000usize);
        let a: Vec<f64> = (0..k * m).map(|_| 0.5 + 0.5 * rng.uniform()).collect();
        let b: Vec<f64> = (0..k * n).map(|_| 0.5 + 0.5 * rng.uniform()).collect();
        let r = reference(Op::T, Op::N, m, n, k, &a, k, &b, k);
        let (da, db) = (gpu.upload(&a).unwrap(), gpu.upload(&b).unwrap());
        let mut dc = gpu.alloc_zeros::<f64>(m * n).unwrap();
        oz.dgemm_at(
            &gpu,
            7,
            Op::T,
            Op::N,
            m,
            n,
            k,
            1.0,
            (&da, 0, k),
            (&db, 0, k),
            0.0,
            (&mut dc, 0, m),
            &mut work,
        )
        .unwrap();
        let c = gpu.download(&dc).unwrap();
        for (x, y) in c.iter().zip(&r) {
            assert!(((x - y) / y).abs() < 1e-15, "{x} vs {y}");
        }
    }

    #[test]
    fn presliced_operands_symmetric_form_and_reproducibility() {
        let Some(gpu) = common::gpu() else { return };
        let oz = Ozaki::new(&gpu).unwrap();
        let mut work = OzakiWork::default();
        let mut rng = Rng(13);
        // X: n x J column-major; K = X X^T from rows of X (vectors strided along J)
        let (n, jj) = (70usize, 517usize);
        let x: Vec<f64> = (0..n * jj).map(|_| rng.uniform() - 0.5).collect();
        let dx = gpu.upload(&x).unwrap();
        let rows = Vectors {
            buf: &dx,
            off: 0,
            ld: n,
            contiguous: false,
        };
        let mut asc = Sliced::new(&gpu, n, jj, 7, Order::Ascending).unwrap();
        let mut desc = Sliced::new(&gpu, n, jj, 7, Order::Descending).unwrap();
        oz.slice(&gpu, rows, &mut asc, &mut work).unwrap();
        oz.slice(&gpu, rows, &mut desc, &mut work).unwrap();
        let mut full = gpu.alloc_zeros::<f64>(n * n).unwrap();
        let mut low = gpu.alloc_zeros::<f64>(n * n).unwrap();
        let mut runs = vec![];
        for _ in 0..3 {
            oz.gemm(
                &gpu,
                asc.view(),
                desc.view(),
                1.0,
                0.0,
                (&mut full, 0, n),
                &mut work,
            )
            .unwrap();
            runs.push(gpu.download(&full).unwrap());
        }
        assert!(runs.windows(2).all(|w| w[0]
            .iter()
            .zip(&w[1])
            .all(|(p, q)| p.to_bits() == q.to_bits())));
        // the same product slice by slice (ascending x ascending: no concatenation)
        let mut pp = gpu.alloc_zeros::<f64>(n * n).unwrap();
        oz.gemm(
            &gpu,
            asc.view(),
            asc.view(),
            1.0,
            0.0,
            (&mut pp, 0, n),
            &mut work,
        )
        .unwrap();
        let pp = gpu.download(&pp).unwrap();
        assert!(pp
            .iter()
            .zip(&runs[0])
            .all(|(p, q)| p.to_bits() == q.to_bits()));
        for blocks in [1, 3, 8] {
            // product by product, and concatenated with the descending copy
            for right in [asc.view(), desc.view()] {
                oz.syrk_lower(
                    &gpu,
                    asc.view(),
                    right,
                    1.0,
                    0.0,
                    (&mut low, 0, n),
                    blocks,
                    &mut work,
                )
                .unwrap();
                let l = gpu.download(&low).unwrap();
                for j in 0..n {
                    for i in j..n {
                        assert_eq!(
                            l[j * n + i].to_bits(),
                            runs[0][j * n + i].to_bits(),
                            "({i},{j}) blocks {blocks}"
                        );
                    }
                }
            }
        }
        // a range of vectors as a view: exactly those rows of the full product
        let (r0, nr) = (16usize, 37usize);
        let mut part = gpu.alloc_zeros::<f64>(nr * n).unwrap();
        oz.gemm(
            &gpu,
            asc.rows_view(r0, nr).unwrap(),
            desc.view(),
            1.0,
            0.0,
            (&mut part, 0, nr),
            &mut work,
        )
        .unwrap();
        let part = gpu.download(&part).unwrap();
        for j in 0..n {
            for i in 0..nr {
                assert_eq!(
                    part[j * nr + i].to_bits(),
                    runs[0][j * n + r0 + i].to_bits()
                );
            }
        }
        // a view's vectors must be held (70, padded to 80); as the left operand its padded
        // rows too (60.. padded to 92: refused by the product), not as the right one
        assert!(asc.rows_view(70, 20).is_err());
        let tail = asc.rows_view(60, 20).unwrap();
        let mut right = gpu.alloc_zeros::<f64>(n * 20).unwrap();
        assert!(oz
            .gemm(
                &gpu,
                tail,
                desc.view(),
                1.0,
                0.0,
                (&mut right, 0, n),
                &mut work
            )
            .is_err());
        let rv = desc.rows_view(60, 10).unwrap();
        oz.gemm(
            &gpu,
            asc.view(),
            rv,
            1.0,
            0.0,
            (&mut right, 0, n),
            &mut work,
        )
        .unwrap();
        let right = gpu.download(&right).unwrap();
        for j in 0..10 {
            for i in 0..n {
                assert_eq!(
                    right[j * n + i].to_bits(),
                    runs[0][(60 + j) * n + i].to_bits()
                );
            }
        }
        // against the exact product
        let mut worst = 0.0f64;
        for j in 0..n {
            for i in 0..n {
                let r = dot2((0..jj).map(|l| x[l * n + i]), (0..jj).map(|l| x[l * n + j]));
                worst = worst.max((runs[0][j * n + i] - r).abs());
            }
        }
        assert!(worst < 1e-15 * jj as f64, "{worst:e}");
    }

    /// Neighbouring diagonals in pairs (skinny right operands) give the same bits as one GEMM
    /// per diagonal, for even and odd diagonal counts.
    #[test]
    fn paired_diagonals_are_bitwise_the_same() {
        let Some(gpu) = common::gpu() else { return };
        let mut oz = Ozaki::new(&gpu).unwrap();
        let mut work = OzakiWork::default();
        let mut rng = Rng(17);
        let (m, n, k) = (130usize, 37usize, 400usize);
        let a: Vec<f64> = (0..k * m).map(|_| rng.wide()).collect();
        let b: Vec<f64> = (0..k * n).map(|_| rng.uniform() - 0.5).collect();
        let (da, db) = (gpu.upload(&a).unwrap(), gpu.upload(&b).unwrap());
        for s in [3usize, 6, 7] {
            let mut out = vec![];
            for pair in [512usize, 0] {
                oz.pair_max_n = pair;
                let mut dc = gpu.alloc_zeros::<f64>(m * n).unwrap();
                oz.dgemm_at(
                    &gpu,
                    s,
                    Op::T,
                    Op::N,
                    m,
                    n,
                    k,
                    1.0,
                    (&da, 0, k),
                    (&db, 0, k),
                    0.0,
                    (&mut dc, 0, m),
                    &mut work,
                )
                .unwrap();
                out.push(gpu.download(&dc).unwrap());
            }
            assert!(
                out[0]
                    .iter()
                    .zip(&out[1])
                    .all(|(p, q)| p.to_bits() == q.to_bits()),
                "{s} slices"
            );
        }
    }
}
