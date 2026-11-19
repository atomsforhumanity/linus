//! The arithmetic vocabulary: what a kernel family computes in.
//!
//! linus is FP64-correct by default (docs/gpu.md). On devices whose FP64 rate is low, the large
//! GEMMs run in an FP64 emulation that keeps FP64 accuracy (the vendor library's fixed-point
//! emulation, or the Ozaki scheme on INT8 tensor cores), and a few families have reduced
//! arithmetic after an accuracy study (FP32 products, double-float pair sums). [`Arith`] names
//! these choices once, for every family: the precision policy resolves one value per kernel
//! family and calculation (with the reason), and the engines take it instead of reading the
//! card or a setting themselves.

use std::fmt;

/// The arithmetic of a kernel family.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Arith {
    /// Native FP64.
    Fp64,
    /// FP64 emulated by the vendor BLAS (cuBLAS's fixed-point emulation) with a fixed
    /// mantissa of `bits` bits (55: FP64 quality on density-fitting data).
    Fp64Emulated { bits: u8 },
    /// FP64 emulated by the Ozaki scheme on INT8 tensor cores with `slices` 8-bit digits per
    /// value (8 slices - 1 bits; 7: FP64 quality on density-fitting data).
    Ozaki { slices: u8 },
    /// FP32 values and products, summed in FP64 above a block.
    Fp32,
    /// Double-float (an unevaluated sum of two FP32 values, about 48 bits): VV10's pair sums.
    DoubleFloat,
}

impl Arith {
    /// The name in reports: `fp64`, `cublas`, `ozaki`, `fp32` or `df64`.
    pub fn name(self) -> &'static str {
        match self {
            Arith::Fp64 => "fp64",
            Arith::Fp64Emulated { .. } => "cublas",
            Arith::Ozaki { .. } => "ozaki",
            Arith::Fp32 => "fp32",
            Arith::DoubleFloat => "df64",
        }
    }

    /// The mantissa bits per operand of an FP64 emulation (None for the other kinds).
    pub fn emulated_bits(self) -> Option<u32> {
        match self {
            Arith::Fp64Emulated { bits } => Some(bits as u32),
            Arith::Ozaki { slices } => Some(8 * slices as u32 - 1),
            _ => None,
        }
    }

    /// FP64 or an FP64 emulation, as opposed to reduced arithmetic.
    pub fn is_fp64_class(self) -> bool {
        matches!(
            self,
            Arith::Fp64 | Arith::Fp64Emulated { .. } | Arith::Ozaki { .. }
        )
    }
}

impl fmt::Display for Arith {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.emulated_bits() {
            Some(b) => write!(f, "{} ({b} bits)", self.name()),
            None => f.write_str(self.name()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_and_bits() {
        assert_eq!(Arith::Ozaki { slices: 7 }.emulated_bits(), Some(55));
        assert_eq!(
            Arith::Fp64Emulated { bits: 55 }.to_string(),
            "cublas (55 bits)"
        );
        assert_eq!(Arith::DoubleFloat.to_string(), "df64");
        assert!(Arith::Ozaki { slices: 7 }.is_fp64_class() && !Arith::Fp32.is_fp64_class());
    }
}
