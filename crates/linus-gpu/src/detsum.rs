//! Deterministic accumulation: the device side of the CPU's fixed-point accumulator.
//!
//! The CPU sums in parallel without depending on the thread count or the schedule by adding
//! values as fixed-point integers (linus-num's `det::FixedAcc`): a value v is split exactly
//! into h 2^-30 + r with h = trunc(v 2^30), and l = trunc(r 2^82); integer sums are exact and
//! associative, so any order gives the same bits. The device does the same split
//! ([`DEVICE_SRC`], `device/detsum.h`) and adds the words with 64-bit integer atomics, never
//! floating-point atomics. Because device additions come in any order, the low word is split
//! once more (l = m 2^26 + q, q in [0, 2^26)) so that no word can overflow before 2^37
//! additions to one element; an element is three words, and its value
//! (H 2^52 + M 2^26 + Q) 2^-82 is the same integer as the CPU's (hi 2^52 + lo), hence the
//! same f64.
//!
//! An invalid value (NaN, infinite, or beyond 2^32 in magnitude) poisons its accumulator, as the
//! CPU's `FixedAcc` records one: a flag word after the elements is set and every element reads
//! NaN, so a bad input surfaces as NaN instead of a clamped finite number. Finite inputs give
//! the same bits as without the flag.
//!
//! [`split`] and [`value`] are the host halves, built on linus-num's (which documents the
//! format); [`words`] sizes an accumulator's buffer; the device accumulator itself is `DetAcc`
//! (feature `gpu`).

/// Device source of `det_split`, `det_poison`, `det_add`, `det_add_block` and `det_value`
/// (needs the portability header, which the device layer pastes in front of every kernel).
pub const DEVICE_SRC: &str = include_str!("../device/detsum.h");

/// The u64 words of a device accumulator of `n` elements: three per element (structure of
/// arrays) and the invalid flag at index 3 n.
#[inline]
pub const fn words(n: usize) -> usize {
    3 * n + 1
}

/// The words (h, m, q) that one addition of `v` contributes: h and l exactly as the CPU
/// accumulator splits v (linus-num's `det::split`, which clamps an invalid value; the device
/// then also poisons the accumulator), then l = m 2^26 + q.
#[inline]
pub fn split(v: f64) -> (i64, i64, i64) {
    let (h, l) = linus_num::det::split(v);
    (h, l >> 26, l & ((1 << 26) - 1))
}

/// The value of an element from its three (wrapping) word sums.
#[inline]
pub fn value(h: u64, m: u64, q: u64) -> f64 {
    let t = ((h as i64 as i128) << 52) + ((m as i64 as i128) << 26) + q as i64 as i128;
    linus_num::det::value_of(t)
}

#[cfg(feature = "gpu")]
pub use device::*;

#[cfg(feature = "gpu")]
mod device {
    use crate::error::GpuError;
    use crate::kernels::registry::KernelModule;
    use crate::{Arg, DeviceBuf, Gpu, Launch};
    use std::borrow::Cow;

    const KERNELS: &str = include_str!("../device/detsum.cu");

    fn source() -> String {
        format!("{}\n{}", super::DEVICE_SRC, KERNELS)
    }

    /// The accumulator's module: the device functions, then its two kernels.
    pub(crate) const MODULE: KernelModule = KernelModule::new(
        "detsum",
        || Cow::Owned(source()),
        &["det_scatter_add", "det_sum"],
    );

    /// A device array of `n` fixed-point accumulators (three u64 words each, structure of
    /// arrays, then the invalid flag: [`super::words`]), zeroed on creation.
    pub struct DetAcc {
        n: usize,
        words: DeviceBuf<u64>,
    }

    impl DetAcc {
        pub fn new(gpu: &Gpu, n: usize) -> Result<Self, GpuError> {
            Ok(DetAcc {
                n,
                words: gpu.alloc_zeros::<u64>(super::words(n))?,
            })
        }

        pub fn len(&self) -> usize {
            self.n
        }

        pub fn is_empty(&self) -> bool {
            self.n == 0
        }

        /// The argument for kernels that call `det_add(acc, n, i, v)` (`n` is [`Self::len`]).
        pub fn arg(&mut self) -> Arg {
            Arg::buf_mut(&mut self.words)
        }

        pub fn zero(&mut self, gpu: &Gpu) -> Result<(), GpuError> {
            gpu.zero(&mut self.words)
        }

        /// The accumulated values (synchronizes the stream); every one NaN once an invalid
        /// value was added.
        pub fn to_f64(&self, gpu: &Gpu) -> Result<Vec<f64>, GpuError> {
            let w = gpu.download(&self.words)?;
            let n = self.n;
            if w[3 * n] != 0 {
                return Ok(vec![f64::NAN; n]);
            }
            Ok((0..n)
                .map(|i| super::value(w[i], w[n + i], w[2 * n + i]))
                .collect())
        }

        /// acc\[idx\[i\]\] += v\[i\] for every i, in whatever order the device runs them. Every
        /// index must be below [`Self::len`].
        pub fn scatter_add(
            &mut self,
            gpu: &Gpu,
            idx: &DeviceBuf<i64>,
            v: &DeviceBuf<f64>,
        ) -> Result<(), GpuError> {
            assert_eq!(idx.len(), v.len());
            let k = gpu.kernel(MODULE.name, &MODULE.source(), "det_scatter_add")?;
            let args = [
                self.arg(),
                Arg::I64(self.n as i64),
                Arg::buf(idx),
                Arg::buf(v),
                Arg::I64(v.len() as i64),
            ];
            // SAFETY: matches det_scatter_add; reads count entries of idx and v and writes the
            // 3 n words (indices below n by this function's contract) and the flag word.
            unsafe { gpu.launch(&k, Launch::linear(v.len(), 256), &args) }
        }

        /// acc\[0\] += sum of v (block-level integer partial sums, then one atomic per word per
        /// block). `block` is a multiple of the warp size, at most 1024.
        pub fn sum_into_first(
            &mut self,
            gpu: &Gpu,
            v: &DeviceBuf<f64>,
            block: u32,
        ) -> Result<(), GpuError> {
            assert!(block > 0 && block <= 1024 && block.is_multiple_of(gpu.warp_size()));
            let k = gpu.kernel(MODULE.name, &MODULE.source(), "det_sum")?;
            let args = [self.arg(), Arg::buf(v), Arg::I64(v.len() as i64)];
            // SAFETY: matches det_sum; reads count values, writes the first element's words and
            // the flag word.
            unsafe { gpu.launch(&k, Launch::linear(v.len(), block), &args) }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_reassembles_exactly() {
        for &v in &[
            0.0,
            1.0,
            -1.0,
            1e-30,
            -3.5e-7,
            123456.789,
            -4.0e9,
            2.0f64.powi(-82),
            f64::NAN,
            1e300,
        ] {
            let (h, m, q) = split(v);
            assert!((0..1 << 26).contains(&q));
            let back = value(h as u64, m as u64, q as u64);
            let vc = if v.is_nan() {
                -4294967296.0
            } else {
                v.clamp(-4294967296.0, 4294967296.0)
            };
            // exact to the 2^-82 resolution
            assert!(
                (back - vc).abs() <= 2.0f64.powi(-82) * 1.0001,
                "{v}: {back}"
            );
        }
    }
}
