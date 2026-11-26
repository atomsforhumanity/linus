//! Kernels. Each module has a source generator (plain Rust that emits CUDA C++ specialized
//! for one class of work, with the angular-momentum tables folded into constants; no GPU
//! needed, so the generators are tested on any machine) and, with the `gpu` feature, the host
//! side that uploads the data and launches.
//!
//! - [`colloc`]: basis functions and their first derivatives on grid points.
//! - [`boys`]: the Boys function table of every three-center kernel (the CPU's, bit for bit).
//! - `eri3c` (feature `testkit`): three-center Coulomb integrals (k|mn) of one shell class,
//!   McMurchie-Davidson as linus-ints' CPU kernel: the spike's hand-written generator, kept as
//!   the reference the generated kernels are compared with (tests and benchmarks only).
//! - [`eri3c_gen`]: the fused three-center kernels and first derivatives generated offline by
//!   tools/intgen (cuda_eri3c.py), one per class, with the schedule the tuner chose;
//!   [`eri3c_class`] derives what the host needs to feed them. Used by [`crate::eri3c`].
//! - [`pcm_gen`]: the PCM surface-charge kernels generated offline by tools/intgen (cuda_pcm.py),
//!   two per Hermite degree (potentials and charge accumulators; first derivatives). Used by
//!   [`crate::pcm`].
//! - [`jengine_gen`]: the J-engine contractions of the density-fitted Coulomb terms in Hermite
//!   space, generated offline by tools/intgen (cuda_jengine.py): three families (orbital-pair or
//!   auxiliary bras and kets), one kernel per (bra accumulator degree, ket degree). Used by the
//!   J-only DF gradient (`grad::df_j`) and the memory-lean DF-J (`jdirect`).
//! - `registry` (feature `gpu`): every hand-written module (name, source, entry points), for
//!   the compile test and the kernel manifest.
//! - [`vgemm`]: variable-size batched FP64 GEMMs (many small products of different shapes in
//!   one launch), for the XC engines' per-batch products.

pub mod boys;
pub mod colloc;
#[cfg(any(test, feature = "testkit"))]
pub mod eri3c;
pub mod eri3c_class;
pub mod gridk;
#[rustfmt::skip]
pub mod eri3c_gen;
/// The per-architecture variants of [`eri3c_gen`] (generated as eri3c_gen/arch.rs).
#[rustfmt::skip]
#[path = "eri3c_gen/arch.rs"]
pub mod eri3c_gen_arch;
#[rustfmt::skip]
pub mod jengine_gen;
#[rustfmt::skip]
pub mod pcm_gen;
#[cfg(feature = "gpu")]
pub mod registry;
pub mod vgemm;

/// A C++ literal that parses to exactly `v` (Rust's shortest round-trip form).
pub(crate) fn lit(v: f64) -> String {
    let s = format!("{v:?}");
    if s.contains(['.', 'e', 'E', 'n', 'i']) {
        s
    } else {
        format!("{s}.0")
    }
}

/// Source statistics of a generated kernel.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct KernelStats {
    /// FP64 operations per unit of work as written in the source (adds, multiplies, FMAs
    /// counted as 2), before the compiler's common-subexpression elimination; transcendental
    /// calls not included.
    pub flops: usize,
    /// Lines of generated source.
    pub lines: usize,
}

#[cfg(test)]
mod tests {
    #[test]
    fn literals_round_trip() {
        for v in [0.5, 1.0, -3.0, 1e-7, 2.0f64.sqrt(), -0.0, 1e300] {
            let s = super::lit(v);
            assert_eq!(s.parse::<f64>().unwrap().to_bits(), v.to_bits(), "{s}");
            assert!(s.contains('.') || s.contains('e'));
        }
    }
}
