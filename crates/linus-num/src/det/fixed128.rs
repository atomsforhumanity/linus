//! The 2^-64 fixed-point format: a value v as the 128-bit two's-complement integer v 2^64
//! (the integer part floor(v) in the high word, the fraction truncated to 2^-64 in the low
//! word). Sums of these integers are exact and associative, so any order of additions gives
//! the same bits. Used by the integral-direct J/K builder (FastJk), which adds the J
//! contributions of eight lanes at once ([`Fixed128x8`], with the lane form of the split in
//! [`crate::simd::Lanes::fixed_add`]).

use crate::simd::LANES;
use std::ops::{Add, AddAssign};

/// A value in the 2^-64 fixed-point format.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(transparent)]
pub struct Fixed128(pub i128);

impl Fixed128 {
    pub const ZERO: Fixed128 = Fixed128(0);

    /// `x` in the format: floor(x) 2^64 plus the fraction truncated to 2^-64.
    #[inline(always)]
    pub fn from_f64(x: f64) -> Self {
        const TWO64: f64 = 18_446_744_073_709_551_616.0;
        let h = x.floor();
        let f = (x - h) * TWO64; // in [0, 2^64]
        Fixed128(((h as i64 as i128) << 64) + (f as u64 as i128))
    }

    /// The value, rounded to f64 (the high word plus the low word times 2^-64).
    #[inline(always)]
    pub fn to_f64(self) -> f64 {
        const INV_TWO64: f64 = 1.0 / 18_446_744_073_709_551_616.0;
        let a = self.0;
        ((a >> 64) as i64) as f64 + ((a as u64) as f64) * INV_TWO64
    }

    /// The sum, wrapping on overflow (two's complement).
    #[inline(always)]
    pub fn wrapping_add(self, o: Fixed128) -> Fixed128 {
        Fixed128(self.0.wrapping_add(o.0))
    }
}

impl Add for Fixed128 {
    type Output = Fixed128;
    #[inline(always)]
    fn add(self, o: Fixed128) -> Fixed128 {
        Fixed128(self.0 + o.0)
    }
}

impl AddAssign for Fixed128 {
    #[inline(always)]
    fn add_assign(&mut self, o: Fixed128) {
        self.0 += o.0;
    }
}

/// Eight [`Fixed128`] accumulators with their high and low words in separate arrays, so a
/// vector of eight lanes adds with whole-vector operations
/// ([`crate::simd::Lanes::fixed_add`]`(&mut a.hi, &mut a.lo)`); lane l is hi\[l\] 2^64 + lo\[l\].
#[derive(Clone, Copy)]
#[repr(C, align(64))]
pub struct Fixed128x8 {
    pub hi: [i64; LANES],
    pub lo: [u64; LANES],
}

impl Fixed128x8 {
    pub const ZERO: Fixed128x8 = Fixed128x8 {
        hi: [0; LANES],
        lo: [0; LANES],
    };

    /// Lane `l`.
    #[inline(always)]
    pub fn get(&self, l: usize) -> Fixed128 {
        Fixed128(((self.hi[l] as i128) << 64) | self.lo[l] as i128)
    }

    /// Lane `l` += x (wrapping).
    #[inline(always)]
    pub fn add(&mut self, l: usize, x: Fixed128) {
        let v = self.get(l).wrapping_add(x);
        self.hi[l] = (v.0 >> 64) as i64;
        self.lo[l] = v.0 as u64;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simd::{Lanes, Portable};

    #[test]
    fn exact_and_order_independent() {
        let vals: Vec<f64> = (0..3000)
            .map(|i| ((i * 7919 % 1013) as f64 - 500.0) * 10f64.powi(i % 7 - 3))
            .collect();
        let fwd = vals
            .iter()
            .fold(Fixed128::ZERO, |s, &v| s + Fixed128::from_f64(v));
        let rev = vals.iter().rev().fold(Fixed128::ZERO, |s, &v| {
            s.wrapping_add(Fixed128::from_f64(v))
        });
        assert_eq!(fwd, rev);
        let exact: f64 = vals.iter().sum();
        assert!((fwd.to_f64() - exact).abs() < 1e-12 * exact.abs().max(1.0));
        assert_eq!(Fixed128::from_f64(-2.5).to_f64(), -2.5);
    }

    #[test]
    fn lanes_add_as_scalars() {
        let mut a = Fixed128x8::ZERO;
        let mut b = Fixed128x8::ZERO;
        for k in 0..50 {
            let x: [f64; LANES] = std::array::from_fn(|l| (k * 8 + l) as f64 * -0.37 + 1e-9);
            unsafe { Portable::<false>::from_array(x).fixed_add(&mut a.hi, &mut a.lo) };
            for (l, &v) in x.iter().enumerate() {
                b.add(l, Fixed128::from_f64(v));
            }
        }
        for l in 0..LANES {
            assert_eq!(a.get(l), b.get(l));
        }
    }
}
