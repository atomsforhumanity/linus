//! linus-num: the numerical primitives every linus crate builds on.
//!
//! - [`det`]: deterministic accumulation, so parallel results have the same bits at any thread
//!   count (two fixed-point formats, ordered chunks, shape-blocked products, fixed-chunk dot
//!   products);
//! - [`simd`]: lane-parallel f64 arithmetic and the run-time choice of the instruction set
//!   ([`simd::Isa`], `LINUS_ISA`), with the [`multiversion!`], [`multiversion_typed!`] and [`multiversion_lanes!`]
//!   macros that compile one body per instruction set.
//!
//! A leaf crate: it depends on no other linus crate (only rayon and faer).

pub mod det;
pub mod simd;
