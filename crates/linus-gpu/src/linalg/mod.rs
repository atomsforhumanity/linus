//! Numerical services on the device, above the vendor layer:
//! - [`Gemm`], the GEMM service that takes the arithmetic of every product explicitly
//!   ([`Arith`], linus-core's vocabulary);
//! - [`Blas1`], the vector operations (scale, axpby) every engine shares, each with its
//!   rounding fixed by its definition.
//!
//! The vendor layer (`crate::cuda`) wraps one cuBLAS handle per [`crate::blas::Blas`], with
//! typed operands; this layer decides which handle runs a product from its arithmetic, so no
//! engine holds a handle whose mode is hidden state.

pub(crate) mod blas1;
mod gemm;

pub use blas1::{Blas1, Operand, Scale};
pub use gemm::Gemm;
pub use linus_core::Arith;
