//! Rys quadrature nodes and weights for orders n = 1..=5.
//!
//! For the Boys-type integrals F_k(T) = ∫_0^1 t^{2k} exp(-T t^2) dt the Rys quadrature of
//! order n gives nodes x_i = t_i^2 and weights w_i with Σ_i w_i x_i^k = F_k(T) for
//! k = 0..2n-1. On [0, T_ASYM\[n\]) the nodes and weights come from piecewise polynomial fits
//! (per order: intervals of width 1/4..1 in T, degree 7..11, evaluated by Estrin's scheme) generated in 60-digit arithmetic by
//! `tools/gen_rys_tables.py`; from T_ASYM\[n\] on, the Gauss-Hermite limit x_i = r_i^2 / T,
//! w_i = W_i / sqrt(T) is exact to double precision. Accuracy: <= 1e-15 relative for every
//! node and weight (tested against the generator's 60-digit reference values).

#[allow(clippy::excessive_precision, clippy::unreadable_literal)]
#[rustfmt::skip]
mod tables {
    include!("rys_tables.rs");
}
use linus_num::simd::Lanes;
use tables::{H, HERM_R2, HERM_W, NCS, WIDTH};
pub use tables::{NMAX, T_ASYM};

/// Coefficient table of order n.
#[inline(always)]
fn table(n: usize) -> &'static [f64] {
    match n {
        1 => &tables::RYS1,
        2 => &tables::RYS2,
        3 => &tables::RYS3,
        4 => &tables::RYS4,
        _ => &tables::RYS5,
    }
}

#[inline(always)]
fn fma(a: f64, b: f64, c: f64) -> f64 {
    // lowers to a single instruction inside the `fma`-enabled SIMD variants
    a.mul_add(b, c)
}

/// Nodes (first N entries) and weights (next N entries) of order N at T, in a W-wide array
/// (W = WIDTH[N]; trailing entries are unspecified). Meant to be inlined into SIMD kernels:
/// the Horner steps run W functions at once.
#[inline(always)]
pub(crate) fn roots_w<const N: usize, const W: usize>(t: f64) -> [f64; W] {
    debug_assert!(W == WIDTH[N]);
    let mut r = [0.0; W];
    if t >= T_ASYM[N] {
        let inv = 1.0 / t;
        let s = inv.sqrt();
        for i in 0..N {
            r[i] = HERM_R2[N][i] * inv;
            r[N + i] = HERM_W[N][i] * s;
        }
        return r;
    }
    fitted::<N, W>(t)
}

/// W-wide fused multiply-add a * x + b.
#[inline(always)]
fn vfma<const W: usize>(a: &[f64; W], x: f64, b: &[f64; W]) -> [f64; W] {
    let mut r = [0.0; W];
    for f in 0..W {
        r[f] = fma(a[f], x, b[f]);
    }
    r
}

/// The fitted nodes and weights for T < T_ASYM[N]: the local polynomial of the interval
/// (degree NCS[N] - 1) by Estrin's scheme (dependency depth 3-4); each step runs the W
/// functions as one vector.
#[inline(always)]
pub(crate) fn fitted<const N: usize, const W: usize>(t: f64) -> [f64; W] {
    let nc = NCS[N];
    let h = H[N];
    let k = (t * (1.0 / h)) as usize;
    // local variable (T - T_mid) / (H / 2)
    let x = t * (2.0 / h) - (2 * k + 1) as f64;
    let tab = table(N);
    debug_assert!(tab.len().is_multiple_of(W));
    // SAFETY: [f64; W] has the layout of W consecutive f64; the length is rounded down.
    let rows: &[[f64; W]] =
        unsafe { std::slice::from_raw_parts(tab.as_ptr() as *const [f64; W], tab.len() / W) };
    let c = &rows[k * nc..(k + 1) * nc];
    let x2 = x * x;
    let x4 = x2 * x2;
    match nc {
        8 => {
            let c: &[[f64; W]; 8] = c.try_into().unwrap();
            let a0 = vfma(&c[1], x, &c[0]);
            let a1 = vfma(&c[3], x, &c[2]);
            let a2 = vfma(&c[5], x, &c[4]);
            let a3 = vfma(&c[7], x, &c[6]);
            let b0 = vfma(&a1, x2, &a0);
            let b1 = vfma(&a3, x2, &a2);
            vfma(&b1, x4, &b0)
        }
        9 => {
            let c: &[[f64; W]; 9] = c.try_into().unwrap();
            let a0 = vfma(&c[1], x, &c[0]);
            let a1 = vfma(&c[3], x, &c[2]);
            let a2 = vfma(&c[5], x, &c[4]);
            let a3 = vfma(&c[7], x, &c[6]);
            let b0 = vfma(&a1, x2, &a0);
            let b1 = vfma(&a3, x2, &a2);
            let d0 = vfma(&b1, x4, &b0);
            vfma(&c[8], x4 * x4, &d0)
        }
        10 => {
            let c: &[[f64; W]; 10] = c.try_into().unwrap();
            let a0 = vfma(&c[1], x, &c[0]);
            let a1 = vfma(&c[3], x, &c[2]);
            let a2 = vfma(&c[5], x, &c[4]);
            let a3 = vfma(&c[7], x, &c[6]);
            let a4 = vfma(&c[9], x, &c[8]);
            let b0 = vfma(&a1, x2, &a0);
            let b1 = vfma(&a3, x2, &a2);
            let d0 = vfma(&b1, x4, &b0);
            vfma(&a4, x4 * x4, &d0)
        }
        _ => {
            let c: &[[f64; W]; 12] = c.try_into().unwrap();
            let x8 = x4 * x4;
            let a0 = vfma(&c[1], x, &c[0]);
            let a1 = vfma(&c[3], x, &c[2]);
            let a2 = vfma(&c[5], x, &c[4]);
            let a3 = vfma(&c[7], x, &c[6]);
            let a4 = vfma(&c[9], x, &c[8]);
            let a5 = vfma(&c[11], x, &c[10]);
            let b0 = vfma(&a1, x2, &a0);
            let b1 = vfma(&a3, x2, &a2);
            let b2 = vfma(&a5, x2, &a4);
            let d0 = vfma(&b1, x4, &b0);
            vfma(&b2, x8, &d0)
        }
    }
}

/// Nodes and weights of order N for eight values of T at once: u[i][l], w[i][l] = scale *
/// weight. The large-T limit is evaluated for all lanes as vectors, the fits per lane.
#[inline(always)]
pub(crate) fn roots_lanes<const N: usize, const W: usize>(
    t: &[f64; 8],
    scale: f64,
    u: &mut [[f64; 8]; N],
    w: &mut [[f64; 8]; N],
) {
    let mut inv = [0.0; 8];
    let mut s = [0.0; 8];
    let mut all_asym = true;
    for l in 0..8 {
        // T = 0 only reaches the fits below; avoid the division by zero anyway
        inv[l] = 1.0 / t[l].max(1.0);
        s[l] = inv[l].sqrt() * scale;
        all_asym &= t[l] >= T_ASYM[N];
    }
    for i in 0..N {
        for l in 0..8 {
            u[i][l] = HERM_R2[N][i] * inv[l];
            w[i][l] = HERM_W[N][i] * s[l];
        }
    }
    if all_asym {
        return;
    }
    // lanes unrolled with literal indices, so that the per-lane results are inserted into the
    // vectors in registers (a scalar store per lane followed by a vector load stalls on
    // store forwarding)
    macro_rules! lane {
        ($l:literal) => {
            if t[$l] < T_ASYM[N] {
                let r = fitted::<N, W>(t[$l]);
                for i in 0..N {
                    u[i][$l] = r[i];
                    w[i][$l] = r[N + i] * scale;
                }
            }
        };
    }
    lane!(0);
    lane!(1);
    lane!(2);
    lane!(3);
    lane!(4);
    lane!(5);
    lane!(6);
    lane!(7);
}

/// [`roots_lanes`] with the T values in a vector: orders 1 and 2 from the Boys functions of
/// all eight lanes at once ([`crate::core::boys::boys_lanes`]: one table row per lane and a vector
/// exp), the higher orders per lane from the fits. Order 1: w = F_0, x = F_1 / F_0. Order 2:
/// the nodes are the roots of x^2 + a x + b, the monic polynomial orthogonal to 1 and x under
/// the moments F_0..F_3 (a = (F_1 F_2 - F_0 F_3) / d, b = (F_1 F_3 - F_2^2) / d,
/// d = F_0 F_2 - F_1^2, never close to cancelling: F_0 F_2 / F_1^2 lies in [1.8, 3]), the
/// larger root from the quadratic formula and the smaller as b over it, and the weights
/// w_2 = (F_1 - x_1 F_0) / (x_2 - x_1), w_1 = F_0 - w_2. Lanes from T_ASYM on take the
/// Gauss-Hermite limit as [`roots_lanes`] does. Agrees with the fits to a few 1e-15 relative
/// (tests: the 60-digit reference values and the Boys moments). Replaces 8 scalar polynomial
/// evaluations (table rows per lane, inserted lane by lane) by ~40 vector instructions.
#[inline(always)]
pub(crate) fn roots_vec<const N: usize, const W: usize, S: Lanes>(
    tv: S,
    scale: f64,
    u: &mut [[f64; 8]; N],
    w: &mut [[f64; 8]; N],
) {
    if N > 2 {
        roots_lanes::<N, W>(&tv.to_array(), scale, u, w);
        return;
    }
    let fits = tv.lt(S::splat(T_ASYM[N]));
    // the Gauss-Hermite limit (all lanes, or the lanes from T_ASYM on)
    let asym = |i: usize| {
        // (T >= T_ASYM > 1 in the lanes that use it)
        let inv = S::splat(1.0) / tv.max(S::splat(1.0));
        (
            S::splat(HERM_R2[N][i]) * inv,
            S::splat(HERM_W[N][i] * scale) * inv.sqrt(),
        )
    };
    if fits == 0 {
        for i in 0..N {
            let (x, wt) = asym(i);
            x.store(&mut u[i]);
            wt.store(&mut w[i]);
        }
        return;
    }
    let mut f = [S::zero(); 4];
    crate::core::boys::boys_lanes(2 * N - 1, tv, &mut f);
    let sc = S::splat(scale);
    let mut r = [[S::zero(); 2]; 2];
    if N == 1 {
        r[0] = [f[1] / f[0], f[0] * sc];
    } else {
        let (m0, m1, m2, m3) = (f[0], f[1], f[2], f[3]);
        let rd = S::splat(1.0) / m0.mul_add(m2, -(m1 * m1));
        // a < 0 (minus the sum of the nodes), b > 0 (their product)
        let a = m1.mul_add(m2, -(m0 * m3)) * rd;
        let b = m1.mul_add(m3, -(m2 * m2)) * rd;
        let sq = a.mul_add(a, S::splat(-4.0) * b).sqrt();
        let x2 = (sq - a) * S::splat(0.5);
        let x1 = b / x2;
        let w2 = (-x1).mul_add(m0, m1) / sq;
        let w1 = m0 - w2;
        r[0] = [x1, w1 * sc];
        r[1] = [x2, w2 * sc];
    }
    for i in 0..N {
        let [mut x, mut wt] = r[i];
        if fits != 0xff {
            let (xa, wa) = asym(i);
            x = S::select(fits, x, xa);
            wt = S::select(fits, wt, wa);
        }
        x.store(&mut u[i]);
        wt.store(&mut w[i]);
    }
}

/// Nodes `x[..n]` and weights `w[..n]` of the order-n Rys quadrature at T >= 0 (n <= NMAX).
pub fn rys_roots(n: usize, t: f64, x: &mut [f64], w: &mut [f64]) {
    assert!((1..=NMAX).contains(&n), "Rys order {n} not tabulated");
    macro_rules! go {
        ($n:literal, $w:literal) => {{
            let r = roots_w::<$n, $w>(t);
            x[..$n].copy_from_slice(&r[..$n]);
            w[..$n].copy_from_slice(&r[$n..2 * $n]);
        }};
    }
    match n {
        1 => go!(1, 2),
        2 => go!(2, 4),
        3 => go!(3, 8),
        4 => go!(4, 8),
        _ => go!(5, 12),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn widths_consistent() {
        for n in 1..=NMAX {
            assert!(WIDTH[n] >= 2 * n);
            assert_eq!(table(n).len() % (NCS[n] * WIDTH[n]), 0);
            assert_eq!(
                table(n).len() / (NCS[n] * WIDTH[n]),
                (T_ASYM[n] / H[n]).round() as usize
            );
            assert!([8, 9, 10, 12].contains(&NCS[n]));
        }
    }

    /// Every node and weight against 60-digit reference values (random T over the whole
    /// range, all interval edges, tiny T and both sides of the switch to the Hermite limit).
    #[test]
    fn matches_reference() {
        let text = include_str!("rys_ref.txt");
        let (mut x, mut w) = ([0.0; NMAX], [0.0; NMAX]);
        let mut worst: f64 = 0.0;
        let mut count = 0;
        for line in text.lines().filter(|l| !l.starts_with('#')) {
            let v: Vec<f64> = line
                .split_whitespace()
                .map(|s| s.parse().unwrap())
                .collect();
            let n = v[0] as usize;
            let t = v[1];
            rys_roots(n, t, &mut x, &mut w);
            for i in 0..n {
                let (rx, rw) = (v[2 + i], v[2 + n + i]);
                worst = worst
                    .max(((x[i] - rx) / rx).abs())
                    .max(((w[i] - rw) / rw).abs());
            }
            count += 1;
        }
        assert!(count > 250);
        assert!(worst <= 1e-14, "worst relative error {worst:e}");
        eprintln!("rys: {count} reference points, worst relative error {worst:.2e}");
    }

    /// Orders 1 and 2 of [`roots_vec`] against the 60-digit reference values (eight T at a
    /// time, also mixed with lanes in the Gauss-Hermite regime) and against the fits.
    fn check_vec<S: Lanes>() -> f64 {
        let text = include_str!("rys_ref.txt");
        let rows: Vec<Vec<f64>> = text
            .lines()
            .filter(|l| !l.starts_with('#'))
            .map(|l| l.split_whitespace().map(|s| s.parse().unwrap()).collect())
            .collect();
        let mut worst: f64 = 0.0;
        for n in 1..=2usize {
            let mut sel: Vec<&Vec<f64>> = rows.iter().filter(|v| v[0] as usize == n).collect();
            // pad to a multiple of 8 with large-T lanes (the Gauss-Hermite limit)
            while !sel.len().is_multiple_of(8) {
                sel.push(sel[sel.len() % 3]);
            }
            for chunk in sel.chunks(8) {
                let t: [f64; 8] = std::array::from_fn(|l| chunk[l][1]);
                let scale = 0.7;
                macro_rules! go {
                    ($n:literal, $w:literal) => {{
                        let (mut u, mut w) = ([[0.0; 8]; $n], [[0.0; 8]; $n]);
                        roots_vec::<$n, $w, S>(S::from_array(t), scale, &mut u, &mut w);
                        for l in 0..8 {
                            for i in 0..$n {
                                let (rx, rw) = (chunk[l][2 + i], chunk[l][2 + $n + i] * scale);
                                worst = worst
                                    .max(((u[i][l] - rx) / rx).abs())
                                    .max(((w[i][l] - rw) / rw).abs());
                            }
                        }
                    }};
                }
                if n == 1 {
                    go!(1, 2)
                } else {
                    go!(2, 4)
                }
            }
        }
        // against the fits on a dense grid, lanes spread over [0, 80)
        for i in 0..6000 {
            let t: [f64; 8] =
                std::array::from_fn(|l| (i as f64 * 0.00731 + l as f64 * 9.77) % 80.0);
            let (mut u1, mut w1, mut u2, mut w2) =
                ([[0.0; 8]; 1], [[0.0; 8]; 1], [[0.0; 8]; 2], [[0.0; 8]; 2]);
            roots_vec::<1, 2, S>(S::from_array(t), 1.0, &mut u1, &mut w1);
            roots_vec::<2, 4, S>(S::from_array(t), 1.0, &mut u2, &mut w2);
            for l in 0..8 {
                let (r1, r2) = (roots_w::<1, 2>(t[l]), roots_w::<2, 4>(t[l]));
                worst = worst
                    .max(((u1[0][l] - r1[0]) / r1[0]).abs())
                    .max(((w1[0][l] - r1[1]) / r1[1]).abs());
                for k in 0..2 {
                    worst = worst
                        .max(((u2[k][l] - r2[k]) / r2[k]).abs())
                        .max(((w2[k][l] - r2[2 + k]) / r2[2 + k]).abs());
                }
            }
        }
        worst
    }

    #[test]
    fn vector_roots_match_reference() {
        let w = check_vec::<linus_num::simd::Portable<false>>();
        eprintln!("roots_vec portable: worst relative error {w:.2e}");
        assert!(w < 1e-14, "portable: {w:e}");
        #[cfg(target_arch = "x86_64")]
        {
            if linus_num::simd::Isa::detect() != linus_num::simd::Isa::Generic {
                #[target_feature(enable = "avx2,fma")]
                unsafe fn run2() -> f64 {
                    check_vec::<linus_num::simd::Avx2>()
                }
                let w = unsafe { run2() };
                eprintln!("roots_vec avx2: worst relative error {w:.2e}");
                assert!(w < 1e-14, "avx2: {w:e}");
            }
            if linus_num::simd::Isa::detect() == linus_num::simd::Isa::Avx512 {
                #[target_feature(enable = "avx512f,avx512vl,avx512dq,fma")]
                unsafe fn run5() -> f64 {
                    check_vec::<linus_num::simd::Avx512>()
                }
                let w = unsafe { run5() };
                eprintln!("roots_vec avx512: worst relative error {w:.2e}");
                assert!(w < 1e-14, "avx512: {w:e}");
            }
        }
    }

    /// The quadrature reproduces the Boys moments F_0..F_{2n-1}.
    #[test]
    #[allow(clippy::needless_range_loop, reason = "indexes several arrays in step")]
    fn reproduces_boys_moments() {
        let mut f = vec![0.0; 2 * NMAX];
        let (mut x, mut w) = ([0.0; NMAX], [0.0; NMAX]);
        let mut worst: f64 = 0.0;
        for i in 0..4000 {
            let t = i as f64 * 0.015;
            for n in 1..=NMAX {
                crate::core::boys::boys_series(2 * n - 1, t, &mut f);
                rys_roots(n, t, &mut x, &mut w);
                for k in 0..2 * n {
                    let q: f64 = (0..n).map(|i| w[i] * x[i].powi(k as i32)).sum();
                    worst = worst.max(((q - f[k]) / f[k]).abs());
                }
            }
        }
        assert!(worst < 1e-13, "worst relative moment error {worst:e}");
    }
}
