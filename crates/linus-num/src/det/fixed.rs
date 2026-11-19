//! The 2^-82 fixed-point format and its accumulator [`FixedAcc`]: order-independent
//! accumulation, so that parallel reductions give the same bits for any number of threads and
//! any schedule.
//!
//! A value v is split exactly into hi * 2^-30 + r (hi = trunc(v 2^30)), r is truncated to a
//! multiple of 2^-82, and both parts are summed as integers (exact and associative). The
//! absolute resolution is 2^-82 (2e-25) per addition; values must stay below 2^32 in
//! magnitude. An accumulator records a value it cannot represent (NaN, infinite, or above 2^32
//! in magnitude) and then reads as NaN ([`FixedAcc::value`]), so a bad input shows in the
//! result instead of being clamped away. Used by the XC integration, the VV10 and COSX exchange
//! builds and gradients; the GPU's accumulator (linus-gpu `detsum`) adds the same words with
//! integer atomics ([`split`]).

use std::sync::Mutex;

const S_HI: f64 = (1u64 << 30) as f64;
const INV_S_HI: f64 = 1.0 / S_HI;
const S_LO: f64 = (1u128 << 82) as f64;
/// Clamp before the float-to-integer conversion (keeps NaN/inf out of range checks).
const CLAMP: f64 = (1u64 << 62) as f64;
/// The largest magnitude a value may have (2^32): larger ones, infinities and NaN are invalid.
const LIMIT: f64 = CLAMP * INV_S_HI;

/// Fixed-point accumulator for `n` values.
#[derive(Clone, Debug)]
pub struct FixedAcc {
    hi: Vec<i64>,
    lo: Vec<i64>,
    ticks: u32,
    /// An added value was NaN, infinite or too large ([`LIMIT`]): every value reads as NaN.
    invalid: bool,
}

/// Whether every value of `v` can be accumulated (finite, magnitude at most 2^32).
#[inline]
pub fn all_valid(v: &[f64]) -> bool {
    v.iter().fold(true, |ok, x| ok & (x.abs() <= LIMIT))
}

/// The words (hi, lo) one addition of `v` contributes: hi = trunc(v 2^30), lo = the rest in
/// units of 2^-82 (|lo| < 2^52). Values beyond 2^32 in magnitude are clamped to it, NaN to
/// -2^32 (callers record them as invalid: [`all_valid`]).
#[inline(always)]
// f64::clamp keeps NaN; max/min map it to a bound
#[allow(clippy::manual_clamp)]
pub fn split(v: f64) -> (i64, i64) {
    // max/min map NaN to the bound, so the conversion below is always in range
    let v = v.max(-CLAMP * INV_S_HI).min(CLAMP * INV_S_HI);
    // SAFETY: |v 2^30| <= 2^62 and |r 2^82| < 2^52 after the clamp; neither is NaN.
    let h = unsafe { (v * S_HI).to_int_unchecked::<i64>() };
    let r = v - h as f64 * INV_S_HI;
    let l = unsafe { (r * S_LO).to_int_unchecked::<i64>() };
    (h, l)
}

/// The value of an accumulated sum `t` in units of 2^-82 (hi 2^52 + lo), rounded to f64.
#[inline]
pub fn value_of(t: i128) -> f64 {
    t as f64 / S_LO
}

/// Adds `v`; false if a value was invalid ([`LIMIT`]).
#[inline(always)]
fn add_generic(hi: &mut [i64], lo: &mut [i64], v: &[f64]) -> bool {
    let mut ok = true;
    for ((h, l), &x) in hi.iter_mut().zip(lo.iter_mut()).zip(v) {
        let (a, b) = split(x);
        *h += a;
        *l += b;
        ok &= x.abs() <= LIMIT;
    }
    ok
}

#[cfg(test)]
#[inline(always)]
fn add_scaled_generic(hi: &mut [i64], lo: &mut [i64], v: &[f64], s: f64) -> bool {
    let mut ok = true;
    for ((h, l), &x) in hi.iter_mut().zip(lo.iter_mut()).zip(v) {
        let (a, b) = split(s * x);
        *h += a;
        *l += b;
        ok &= (s * x).abs() <= LIMIT;
    }
    ok
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512dq,avx512vl,avx2,fma")]
unsafe fn add_avx512(hi: &mut [i64], lo: &mut [i64], v: &[f64]) -> bool {
    add_generic(hi, lo, v)
}

#[cfg(all(test, target_arch = "x86_64"))]
#[target_feature(enable = "avx512f,avx512dq,avx512vl,avx2,fma")]
unsafe fn add_scaled_avx512(hi: &mut [i64], lo: &mut [i64], v: &[f64], s: f64) -> bool {
    add_scaled_generic(hi, lo, v, s)
}

#[cfg(target_arch = "x86_64")]
#[inline]
fn has_avx512() -> bool {
    crate::simd::Isa::detect() == crate::simd::Isa::Avx512
}

impl FixedAcc {
    pub fn new(n: usize) -> Self {
        FixedAcc {
            hi: vec![0; n],
            lo: vec![0; n],
            ticks: 0,
            invalid: false,
        }
    }

    /// acc[off..off + v.len()] += v
    #[inline]
    pub fn add(&mut self, off: usize, v: &[f64]) {
        let (h, l) = (
            &mut self.hi[off..off + v.len()],
            &mut self.lo[off..off + v.len()],
        );
        #[cfg(target_arch = "x86_64")]
        if has_avx512() {
            // SAFETY: AVX-512 F/DQ/VL checked at run time
            self.invalid |= !unsafe { add_avx512(h, l, v) };
            return;
        }
        self.invalid |= !add_generic(h, l, v);
    }

    /// acc[off..off + v.len()] += s v
    #[cfg(test)]
    #[inline]
    pub fn add_scaled(&mut self, off: usize, v: &[f64], s: f64) {
        let (h, l) = (
            &mut self.hi[off..off + v.len()],
            &mut self.lo[off..off + v.len()],
        );
        #[cfg(target_arch = "x86_64")]
        if has_avx512() {
            // SAFETY: AVX-512 F/DQ/VL checked at run time
            self.invalid |= !unsafe { add_scaled_avx512(h, l, v, s) };
            return;
        }
        self.invalid |= !add_scaled_generic(h, l, v, s);
    }

    /// acc\[off + rows\[i\] * stride + cols\[j\]\] += blk\[i\]\[j\] for the row-major block `blk`
    /// (rows.len() x ncols), with `runs` the maximal runs (first position, first value, length)
    /// of consecutive entries of `cols` ([`index_runs`]).
    #[inline]
    pub fn add_block(
        &mut self,
        off: usize,
        stride: usize,
        rows: &[usize],
        runs: &[(usize, usize, usize)],
        ncols: usize,
        blk: &[f64],
    ) {
        for (i, &m) in rows.iter().enumerate() {
            let base = off + m * stride;
            let row = &blk[i * ncols..(i + 1) * ncols];
            for &(j0, n0, len) in runs {
                self.add(base + n0, &row[j0..j0 + len]);
            }
        }
    }

    /// [`Self::tick`] for a batch that added at most once to every element.
    #[inline]
    pub fn batch_done(&mut self) {
        self.tick(1);
    }

    /// The high and low words, for callers that add the values `v` with their own loops over
    /// [`split`] (they must call [`Self::tick`] as usual); an invalid value in `v` is recorded
    /// here.
    #[inline]
    pub fn parts_for(&mut self, v: &[f64]) -> (&mut [i64], &mut [i64]) {
        self.invalid |= !all_valid(v);
        (&mut self.hi, &mut self.lo)
    }

    /// acc\[i\] += v (single element)
    #[inline]
    #[allow(clippy::neg_cmp_op_on_partial_ord, reason = "true for NaN too")]
    pub fn add1(&mut self, i: usize, v: f64) {
        let (a, b) = split(v);
        self.hi[i] += a;
        self.lo[i] += b;
        self.invalid |= !(v.abs() <= LIMIT);
    }

    /// Call once per batch in which every element received at most `k` additions (k <= 64):
    /// carries the low words into the high ones before they can overflow.
    #[inline]
    pub fn tick(&mut self, k: u32) {
        self.ticks += k;
        if self.ticks >= 1024 {
            self.normalize();
        }
    }

    fn normalize(&mut self) {
        for (h, l) in self.hi.iter_mut().zip(self.lo.iter_mut()) {
            let c = *l >> 52;
            *h += c;
            *l -= c << 52;
        }
        self.ticks = 0;
    }

    pub fn merge(&mut self, o: &FixedAcc) {
        self.invalid |= o.invalid;
        self.normalize();
        for (i, (&h, &l)) in o.hi.iter().zip(&o.lo).enumerate() {
            let c = l >> 52;
            self.hi[i] += h + c;
            self.lo[i] += l - (c << 52);
        }
        self.normalize();
    }

    /// The sum at `i`; NaN once an invalid value was added anywhere in the accumulator.
    #[inline]
    pub fn value(&self, i: usize) -> f64 {
        if self.invalid {
            return f64::NAN;
        }
        value_of(((self.hi[i] as i128) << 52) + self.lo[i] as i128)
    }

    pub fn to_f64(&self) -> Vec<f64> {
        (0..self.hi.len()).map(|i| self.value(i)).collect()
    }
}

/// One lazily created accumulator per pool thread, merged exactly at the end.
///
/// Invariant: [`ThreadAccs::with`] holds the slot's (non-reentrant) mutex while its closure
/// runs, so the closure must not start rayon work (`par_iter`, `join`, a parallel faer
/// product): while waiting, the thread could pick up a task that calls `with` on the same slot
/// and deadlock. Every current closure is sequential.
pub struct ThreadAccs<T> {
    slots: Vec<Mutex<Option<T>>>,
}

impl<T> ThreadAccs<T> {
    pub fn new() -> Self {
        ThreadAccs {
            slots: (0..rayon::current_num_threads().max(1))
                .map(|_| Mutex::new(None))
                .collect(),
        }
    }

    /// Runs `f` on this thread's accumulator (created by `init` on first use). `f` must be
    /// sequential (see the type's invariant).
    pub fn with<R>(&self, init: impl FnOnce() -> T, f: impl FnOnce(&mut T) -> R) -> R {
        let t = rayon::current_thread_index().unwrap_or(0) % self.slots.len();
        let mut g = self.slots[t].lock().unwrap();
        f(g.get_or_insert_with(init))
    }

    pub fn into_vec(self) -> Vec<T> {
        self.slots
            .into_iter()
            .filter_map(|m| m.into_inner().unwrap())
            .collect()
    }
}

impl<T> Default for ThreadAccs<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// Merge fixed-point accumulators (any order gives the same bits).
pub fn merge_all(accs: Vec<Vec<FixedAcc>>) -> Option<Vec<FixedAcc>> {
    let mut it = accs.into_iter();
    let mut tot = it.next()?;
    for a in it {
        for (t, x) in tot.iter_mut().zip(&a) {
            t.merge(x);
        }
    }
    Some(tot)
}

/// Offset of column j of a packed lower triangle (column-major, rows j..n): the layout of the
/// accumulators of symmetric matrices.
#[inline(always)]
pub fn packed_col(n: usize, j: usize) -> usize {
    j * n - j * j.saturating_sub(1) / 2
}

/// Contiguous runs (local start, global start, length) of ascending indices: the `runs` of
/// [`FixedAcc::add_block`].
#[inline]
pub fn index_runs(idx: &[usize], runs: &mut Vec<(usize, usize, usize)>) {
    runs.clear();
    let mut i = 0;
    while i < idx.len() {
        let mut j = i + 1;
        while j < idx.len() && idx[j] == idx[j - 1] + 1 {
            j += 1;
        }
        runs.push((i, idx[i], j - i));
        i = j;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn order_independent() {
        let vals: Vec<f64> = (0..5000)
            .map(|i| {
                ((i * 7919 % 1000) as f64 - 500.0)
                    * 1e-3
                    * (1.0 + (i % 13) as f64 * 1e3_f64.powi(i % 5 - 2))
            })
            .collect();
        let mut a = FixedAcc::new(1);
        for v in &vals {
            a.add(0, &[*v]);
            a.tick(1);
        }
        let mut b = FixedAcc::new(1);
        let mut c = FixedAcc::new(1);
        for (i, v) in vals.iter().rev().enumerate() {
            if i % 2 == 0 {
                b.add1(0, *v);
                b.tick(1);
            } else {
                c.add_scaled(0, &[*v], 1.0);
                c.tick(1);
            }
        }
        b.merge(&c);
        assert_eq!(a.value(0).to_bits(), b.value(0).to_bits());
        let exact: f64 = vals.iter().sum();
        assert!((a.value(0) - exact).abs() < 1e-9 * exact.abs().max(1.0));
    }

    #[test]
    fn invalid_values_read_as_nan() {
        for bad in [f64::NAN, f64::INFINITY, -f64::INFINITY, 1e10, -1e10] {
            let mut a = FixedAcc::new(2);
            a.add(0, &[1.0, 2.0]);
            assert_eq!(a.value(1), 2.0);
            let mut b = FixedAcc::new(2);
            b.add1(1, bad);
            a.merge(&b);
            assert!(a.value(0).is_nan() && a.value(1).is_nan(), "{bad}");
            let mut c = FixedAcc::new(1);
            let _ = c.parts_for(&[0.5, bad]);
            assert!(c.value(0).is_nan(), "{bad}");
        }
        // the largest valid magnitude
        let mut a = FixedAcc::new(1);
        a.add(0, &[LIMIT]);
        a.add1(0, -LIMIT);
        assert_eq!(a.value(0), 0.0);
    }

    #[test]
    fn packed_offsets() {
        let n = 7;
        let mut o = 0;
        for j in 0..n {
            assert_eq!(packed_col(n, j), o);
            o += n - j;
        }
    }
}
