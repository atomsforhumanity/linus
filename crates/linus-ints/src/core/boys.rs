//! Boys function F_m(T) = ∫_0^1 t^{2m} exp(-T t^2) dt.
//!
//! Small T (< `T_SWITCH`): Taylor expansion about the nearest point of a table spaced
//! `DELTA` apart, evaluated for the highest order needed and recursed downward
//! (stable). Large T: F_0 = sqrt(pi/T)/2 (erf(sqrt T) = 1 to double precision) and upward
//! recursion, stable for T > m.
//!
//! The table is filled once from the convergent series
//! F_m(T) = exp(-T) Σ_k (2T)^k / ((2m+1)(2m+3)...(2m+2k+1)).

use linus_num::simd::{exp_nonpos, Lanes};
use std::sync::OnceLock;

/// Highest order supported (enough for second derivatives of (hh|hh) quartets).
pub const MAX_M: usize = 40;
const N_TAYLOR: usize = 8;
const DELTA: f64 = 0.1;
const T_SWITCH: f64 = 45.0;
const N_GRID: usize = (T_SWITCH / DELTA) as usize + 2;
const TABLE_M: usize = MAX_M + N_TAYLOR + 1;

struct Table {
    /// f[i * TABLE_M + m] = F_m(i * DELTA)
    f: Vec<f64>,
}

fn table() -> &'static Table {
    static T: OnceLock<Table> = OnceLock::new();
    T.get_or_init(|| {
        let mut f = vec![0.0; N_GRID * TABLE_M];
        for i in 0..N_GRID {
            let t = i as f64 * DELTA;
            let row = &mut f[i * TABLE_M..(i + 1) * TABLE_M];
            boys_series(TABLE_M - 1, t, row);
        }
        Table { f }
    })
}

/// Reference evaluation: series for the highest order, downward recursion for the rest.
/// Accurate to ~1e-16 relative for T up to ~60; used to fill the table and in tests.
pub fn boys_series(mmax: usize, t: f64, out: &mut [f64]) {
    let mut term = 1.0 / (2 * mmax + 1) as f64;
    let mut sum = term;
    let mut k = 1usize;
    loop {
        term *= 2.0 * t / (2 * mmax + 2 * k + 1) as f64;
        sum += term;
        if term < sum * 1e-18 {
            break;
        }
        k += 1;
        if k > 10_000 {
            break;
        }
    }
    let et = (-t).exp();
    out[mmax] = et * sum;
    for m in (0..mmax).rev() {
        out[m] = (2.0 * t * out[m + 1] + et) / (2 * m + 1) as f64;
    }
}

/// F_0..=F_mmax at T into out[0..=mmax].
#[inline]
pub fn boys(mmax: usize, t: f64, out: &mut [f64]) {
    debug_assert!(mmax <= MAX_M);
    if t < T_SWITCH {
        let tab = table();
        let i = (t / DELTA + 0.5) as usize;
        let dt = i as f64 * DELTA - t; // F_m(T) = Σ_k F_{m+k}(T_i) (dt)^k / k!
        let row = &tab.f[i * TABLE_M..(i + 1) * TABLE_M];
        // Taylor for the top order
        let mut acc = row[mmax + N_TAYLOR - 1];
        for k in (0..N_TAYLOR - 1).rev() {
            acc = row[mmax + k] + acc * dt / (k + 1) as f64;
        }
        out[mmax] = acc;
        if mmax > 0 {
            let et = (-t).exp();
            let t2 = 2.0 * t;
            for m in (0..mmax).rev() {
                out[m] = (t2 * out[m + 1] + et) * INV_ODD[m];
            }
        }
    } else {
        let inv_t = 1.0 / t;
        out[0] = 0.5 * (std::f64::consts::PI * inv_t).sqrt();
        if mmax > 0 {
            let et = (-t).exp();
            let half_inv_t = 0.5 * inv_t;
            for m in 0..mmax {
                out[m + 1] = ((2 * m + 1) as f64 * out[m] - et) * half_inv_t;
            }
        }
    }
}

/// The Taylor table of [`boys_lanes_tab`] (built on first use). Kernels fetch it once per
/// call, so their inner loops carry no initialization check (whose cold call path made LLVM
/// spill live vectors around every Boys evaluation).
#[inline]
pub fn boys_table() -> &'static [f64] {
    &table().f
}

/// F_0..=F_mmax for eight values of T at once: out\[m\] holds F_m in each lane. Same scheme as
/// [`boys`] (Taylor about the nearest grid point below `T_SWITCH`, asymptotic form above),
/// with the table rows gathered per lane.
#[inline(always)]
pub fn boys_lanes<V: Lanes>(mmax: usize, t: V, out: &mut [V]) {
    boys_lanes_tab(boys_table(), mmax, t, out)
}

/// [`boys_lanes`] with the table from [`boys_table`] passed in.
#[inline(always)]
pub fn boys_lanes_tab<V: Lanes>(tab: &[f64], mmax: usize, t: V, out: &mut [V]) {
    debug_assert!(mmax <= MAX_M);
    debug_assert_eq!(tab.len(), N_GRID * TABLE_M);
    let small = t.lt(V::splat(T_SWITCH));
    // exp(-T) is only needed by the recursions (F_0 alone comes from the table or sqrt(pi/T))
    let et = if mmax > 0 { exp_nonpos(-t) } else { V::zero() };
    if small != 0 {
        // the table row of every lane is in bounds: T clamped to [0, T_SWITCH] (lanes with
        // T >= T_SWITCH get a valid, unused row; a negative or NaN T, from bad input, row 0)
        let tc = t.max(V::zero()).min(V::splat(T_SWITCH));
        let i = (tc * V::splat(1.0 / DELTA)).round();
        let dt = i.mul_add(V::splat(DELTA), -tc);
        const _: () = assert!(N_TAYLOR == 8);
        // SAFETY: i is an integer in [0, T_SWITCH / DELTA] = [0, N_GRID - 2] (the clamp above),
        // and mmax + 8 <= MAX_M + N_TAYLOR < TABLE_M, so every row read is inside the table
        let rows = unsafe { V::gather_rows8(tab, i, TABLE_M, mmax) };
        let mut acc = rows[N_TAYLOR - 1];
        for k in (0..N_TAYLOR - 1).rev() {
            acc = (acc * dt).mul_add(V::splat(INV_INT[k + 1]), rows[k]);
        }
        out[mmax] = acc;
        let t2 = t + t;
        for m in (0..mmax).rev() {
            out[m] = t2.mul_add(out[m + 1], et) * V::splat(INV_ODD[m]);
        }
    }
    if small != 0xff {
        let inv_t = V::splat(1.0) / t;
        let half_inv_t = inv_t * V::splat(0.5);
        let mut f = (inv_t * V::splat(std::f64::consts::PI)).sqrt() * V::splat(0.5);
        if small == 0 {
            out[0] = f;
            for m in 0..mmax {
                f = f.mul_add(V::splat((2 * m + 1) as f64), -et) * half_inv_t;
                out[m + 1] = f;
            }
        } else {
            out[0] = V::select(small, out[0], f);
            for m in 0..mmax {
                f = f.mul_add(V::splat((2 * m + 1) as f64), -et) * half_inv_t;
                out[m + 1] = V::select(small, out[m + 1], f);
            }
        }
    }
}

/// 1/k
const INV_INT: [f64; N_TAYLOR + 1] = {
    let mut a = [0.0; N_TAYLOR + 1];
    let mut k = 1;
    while k <= N_TAYLOR {
        a[k] = 1.0 / k as f64;
        k += 1;
    }
    a
};

/// 1/(2m+1)
const INV_ODD: [f64; MAX_M + 1] = {
    let mut a = [0.0; MAX_M + 1];
    let mut m = 0;
    while m <= MAX_M {
        a[m] = 1.0 / (2 * m + 1) as f64;
        m += 1;
    }
    a
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn boys_matches_series_everywhere() {
        let mut a = vec![0.0; MAX_M + 1];
        let mut b = vec![0.0; MAX_M + 1];
        let mut worst: f64 = 0.0;
        for i in 0..20000 {
            let t = i as f64 * 0.00731;
            for &mmax in &[0usize, 1, 4, 8, 16, 24, MAX_M] {
                boys(mmax, t, &mut a);
                boys_series(mmax, t, &mut b);
                for m in 0..=mmax {
                    let rel = ((a[m] - b[m]) / b[m]).abs();
                    worst = worst.max(rel);
                }
            }
        }
        assert!(worst < 5e-14, "worst relative error {worst:e}");
    }

    fn check_lanes<V: Lanes>() {
        let mut b = vec![0.0; MAX_M + 1];
        let mut out = vec![V::zero(); MAX_M + 1];
        let mut worst: f64 = 0.0;
        for i in 0..4000 {
            // mix small and large T in one vector
            let ts: [f64; 8] =
                std::array::from_fn(|k| (i as f64 * 0.0173 + k as f64 * 7.31) % 80.0);
            for &mmax in &[0usize, 1, 2, 4, 8, 12, 16, 24] {
                boys_lanes(mmax, V::from_array(ts), &mut out);
                for (k, &t) in ts.iter().enumerate() {
                    boys_series(mmax, t, &mut b);
                    for m in 0..=mmax {
                        let v = out[m].to_array()[k];
                        worst = worst.max(((v - b[m]) / b[m]).abs());
                    }
                }
            }
        }
        assert!(worst < 5e-14, "worst relative error {worst:e}");
    }

    #[test]
    fn boys_lanes_portable() {
        check_lanes::<linus_num::simd::Portable<false>>();
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn boys_lanes_avx2() {
        if linus_num::simd::Isa::detect() != linus_num::simd::Isa::Generic {
            #[target_feature(enable = "avx2,fma")]
            unsafe fn run() {
                check_lanes::<linus_num::simd::Avx2>();
            }
            unsafe { run() }
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn boys_lanes_avx512() {
        if linus_num::simd::Isa::detect() == linus_num::simd::Isa::Avx512 {
            #[target_feature(enable = "avx512f,avx512dq,fma")]
            unsafe fn run() {
                check_lanes::<linus_num::simd::Avx512>();
            }
            unsafe { run() }
        }
    }

    #[test]
    fn boys_known_values() {
        let mut f = [0.0; 3];
        boys(2, 0.0, &mut f);
        assert!((f[0] - 1.0).abs() < 1e-15);
        assert!((f[1] - 1.0 / 3.0).abs() < 1e-15);
        assert!((f[2] - 0.2).abs() < 1e-15);
        // F0(T) = sqrt(pi/T)/2 erf(sqrt(T)); erf(1) = 0.8427007929497149
        boys(0, 1.0, &mut f);
        let r = 0.5 * std::f64::consts::PI.sqrt() * 0.842_700_792_949_714_9;
        assert!((f[0] - r).abs() < 1e-15, "{} {}", f[0], r);
    }
}
