//! The GEMM service: every product names its arithmetic.

use crate::blas::{Blas, Fp64Mode, GemmDims, GemmScalar, GemmShape, MatMut, MatPtr, MatRef, Op};
use crate::{DeviceBuf, Gpu, GpuError};
use linus_core::Arith;

/// The GEMMs, SYRKs and TRSMs of an engine, each in the arithmetic its call names.
///
/// It keeps one cuBLAS handle per arithmetic the engine has prepared ([`Gemm::prepare`]): the
/// native handle (FP64, and FP32 with f32 operands) and an emulated FP64 handle per mantissa
/// width (`Arith::Fp64Emulated { bits }`: cuBLAS's fixed-point emulation, eager, `bits` fixed).
/// A handle's configuration is set once; switching an engine's arithmetic means naming another
/// one in its calls, never reconfiguring a handle that other calls share.
///
/// The Ozaki scheme and double-float pair sums are engines of their own ([`crate::emu`], the
/// VV10 kernels), not cuBLAS configurations: a GEMM call in them is a contract error. SYRK and
/// TRSM are always native FP64 (cuBLAS emulates GEMMs only).
pub struct Gemm {
    native: Blas,
    emulated: Vec<(u8, Blas)>,
}

impl Gemm {
    /// The service with its native handle.
    pub fn new(gpu: &Gpu) -> Result<Gemm, GpuError> {
        Ok(Gemm {
            native: Blas::new(gpu)?,
            emulated: Vec::new(),
        })
    }

    /// Prepares the handle that runs `arith`: for `Fp64Emulated { bits }` an emulated handle
    /// (created and configured the first time); for FP64 and FP32 the native handle, its math
    /// set to cuBLAS's default.
    pub fn prepare(&mut self, gpu: &Gpu, arith: Arith) -> Result<(), GpuError> {
        match arith {
            Arith::Fp64 | Arith::Fp32 => self.native.configure(gpu, Fp64Mode::Native),
            Arith::Fp64Emulated { bits } => {
                if !self.emulated.iter().any(|(b, _)| *b == bits) {
                    let blas = Blas::with_mode(gpu, Fp64Mode::fixed(bits))?;
                    self.emulated.push((bits, blas));
                }
                Ok(())
            }
            a => Err(GpuError::Contract(format!(
                "GEMM: {a} is not an arithmetic of cuBLAS's handles"
            ))),
        }
    }

    /// The handle that runs `T` products in `arith`.
    fn handle<T: GemmScalar>(&self, arith: Arith) -> Result<&Blas, GpuError> {
        if !T::runs_in(arith) {
            return Err(GpuError::Contract(format!("{} in {arith}", T::NAME)));
        }
        match arith {
            Arith::Fp64Emulated { bits } => self
                .emulated
                .iter()
                .find(|(b, _)| *b == bits)
                .map(|(_, h)| h)
                .ok_or_else(|| {
                    GpuError::Contract(format!("GEMM: {arith} was not prepared (Gemm::prepare)"))
                }),
            _ => Ok(&self.native),
        }
    }

    /// One product C = alpha op(A) op(B) + beta C in `arith` ([`Blas::gemm`]).
    pub fn gemm<T: GemmScalar>(
        &self,
        arith: Arith,
        d: GemmDims,
        s: (T, T),
        a: MatRef<'_, T>,
        b: MatRef<'_, T>,
        c: MatMut<'_, T>,
    ) -> Result<(), GpuError> {
        self.handle::<T>(arith)?.gemm(d, s, a, b, c)
    }

    /// A strided batch of `d.batch` products in `arith` ([`Blas::gemm_strided`]).
    pub fn gemm_strided<T: GemmScalar>(
        &self,
        arith: Arith,
        d: GemmDims,
        s: (T, T),
        a: MatRef<'_, T>,
        b: MatRef<'_, T>,
        c: MatMut<'_, T>,
    ) -> Result<(), GpuError> {
        self.handle::<T>(arith)?.gemm_strided(d, s, a, b, c)
    }

    /// [`Gemm::gemm_strided`] on operands by address ([`Blas::gemm_raw`]).
    ///
    /// # Safety
    /// As [`Blas::gemm_raw`].
    pub unsafe fn gemm_raw<T: GemmScalar>(
        &self,
        arith: Arith,
        d: GemmDims,
        s: (T, T),
        a: MatPtr<T>,
        b: MatPtr<T>,
        c: MatPtr<T>,
    ) -> Result<(), GpuError> {
        // SAFETY: the caller's contract.
        unsafe { self.handle::<T>(arith)?.gemm_raw(d, s, a, b, c) }
    }

    /// A grouped GEMM in `arith` ([`Blas::gemm_grouped`]).
    ///
    /// # Safety
    /// As [`Blas::gemm_grouped`].
    pub unsafe fn gemm_grouped<T: GemmScalar>(
        &self,
        arith: Arith,
        op: (Op, Op),
        s: (T, T),
        shapes: &[GemmShape],
        (ptrs, off): (&DeviceBuf<u64>, usize),
    ) -> Result<(), GpuError> {
        // SAFETY: the caller's contract.
        unsafe {
            self.handle::<T>(arith)?
                .gemm_grouped(op, s, shapes, ptrs, off)
        }
    }

    /// C = alpha A A^T + beta C, lower triangle, in native FP64 ([`Blas::syrk_lower`]).
    pub fn syrk_lower(
        &self,
        nk: (usize, usize),
        s: (f64, f64),
        a: MatRef<'_, f64>,
        c: MatMut<'_, f64>,
    ) -> Result<(), GpuError> {
        self.native.syrk_lower(nk, s, a, c)
    }

    /// B <- L^-1 B or L^-T B in native FP64 ([`Blas::trsm_lower_left`]).
    pub fn trsm_lower_left(
        &self,
        op: Op,
        mn: (usize, usize),
        l: MatRef<'_, f64>,
        b: MatMut<'_, f64>,
    ) -> Result<(), GpuError> {
        self.native.trsm_lower_left(op, mn, l, b)
    }
}
