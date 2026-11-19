//! Deterministic accumulation: parallel sums whose bits do not depend on the number of threads
//! or the schedule (CONTRIBUTING.md, "Determinism").
//!
//! Floating-point addition is not associative, so a parallel sum is reproducible only if its
//! order is fixed or its additions are exact. linus uses five techniques, each where it fits:
//!
//! | technique | items | used by |
//! |---|---|---|
//! | 2^-82 fixed point ([`FixedAcc`]): hi = trunc(v 2^30) and lo = the rest in 2^-82 units, summed as two i64 words; exact, any order; invalid values read as NaN | [`fixed`] | XC quadrature, VV10, COSX; the GPU's integer-atomic accumulator adds the same words |
//! | 2^-64 fixed point ([`Fixed128`]): v 2^64 as one i128 (floor in the high word); exact, any order | [`fixed128`] | the integral-direct J/K builder (FastJk), eight lanes at once ([`Fixed128x8`]) |
//! | ordered chunks ([`par_fold_det`] strided, [`ordered_chunks`] contiguous): a chunk count fixed by the problem, each chunk folded in item order, the chunks combined in order | [`chunks`] | integral derivatives, PCM, the DF builder's row contractions, RI-J gradients |
//! | shape-blocked products ([`det_matmul`]): output blocks sized from the shape, each a sequential GEMM | [`matmul`] | the DF gradient |
//! | fixed-chunk dot products ([`dot()`], [`col_dots`], [`sub_cols`]): chunks of a fixed length, each a fixed-order SIMD-independent dot, added in chunk order | [`mod@dot`] | the response's block Krylov solver |
//!
//! [`ThreadAccs`] keeps one accumulator per pool thread for the fixed-point formats, whose
//! merges are exact in any order. The two fixed-point formats differ in resolution and range
//! (2^-82 below 2^32 against 2^-64 below 2^63) and give different bits; unifying them is a
//! numerics change, recorded for the numerics epoch.

pub mod chunks;
pub mod dot;
pub mod fixed;
pub mod fixed128;
pub mod matmul;

pub use chunks::{
    add_into, nchunk_for, ordered_chunks, par_fold_det, par_fold_det_into, DEFAULT_BUDGET,
};
pub use dot::{col_dots, dot, sub_cols, DOT_CHUNK};
pub use fixed::{
    all_valid, index_runs, merge_all, packed_col, split, value_of, FixedAcc, ThreadAccs,
};
pub use fixed128::{Fixed128, Fixed128x8};
pub use matmul::det_matmul;
