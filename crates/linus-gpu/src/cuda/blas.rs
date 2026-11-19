//! cuBLAS: GEMMs on typed operands (FP64, native or with cuBLAS's fixed-point FP64 emulation;
//! FP32; INT8 with INT32 accumulation in `ozaki.rs`), SYRK and TRSM.
//!
//! An operand is a column-major sub-matrix of a device buffer: [`MatRef`] (read) or [`MatMut`]
//! (written), its first element's offset, leading dimension and, in a strided batch, the
//! stride between products. Every safe entry point checks every operand against its buffer in
//! one place ([`MatRef::check`]); that check is the safety argument of the cuBLAS call. The
//! element type picks the routine ([`GemmScalar`]: DGEMM for f64, SGEMM for f32), so one
//! generic method serves both precisions.
//!
//! **Entry points are fixed per call shape.** A single product runs `cublasDgemm` in FP64 and
//! the strided-batched SGEMM with one product in FP32; a strided batch runs the strided-batched
//! routine whatever its count; an INT8 product runs `cublasGemmEx` for one product and
//! `cublasGemmStridedBatchedEx` for more. cuBLAS may choose different kernels for different
//! entry points of the same product, so these are part of linus's bitwise results and must not
//! be "simplified".
//!
//! Emulation is a handle's configuration, set once when it is created ([`Blas::with_mode`]):
//! the math mode `CUBLAS_FP64_EMULATED_FIXEDPOINT_MATH`, a strategy (`performant`: only where
//! cuBLAS expects a gain; `eager`: whenever possible), and the number of retained mantissa
//! bits, either chosen per call by cuBLAS from the inputs (`dynamic`, optionally biased) or
//! fixed. cuBLAS can report the bits it retained through a device pointer;
//! [`Blas::retained_bits`] reads it after a call. The engines do not hold handles: they run
//! their products through [`crate::linalg::Gemm`], which keeps one handle per arithmetic and
//! takes the arithmetic of every call explicitly.

use super::device::Gpu;
use super::pool::{DevPtr, DeviceBuf};
use crate::error::GpuError;
use cudarc::cublas::sys as cb;
use cudarc::cublas::CudaBlas;
use cudarc::driver::DeviceRepr;

/// Mantissa control of the fixed-point emulation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mantissa {
    /// cuBLAS chooses the bits per call from the inputs, plus `offset` (may be negative),
    /// capped at `max_bits` (0: the library default).
    Dynamic { offset: i32, max_bits: i32 },
    /// Always `bits` (quantized up by cuBLAS to what its slices hold).
    Fixed { bits: i32 },
}

/// When cuBLAS emulates.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Strategy {
    Performant,
    Eager,
}

/// The FP64 arithmetic of a handle.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fp64Mode {
    Native,
    Emulated {
        strategy: Strategy,
        mantissa: Mantissa,
    },
}

impl Fp64Mode {
    /// linus's FP64 emulation: eager, with a fixed mantissa of `bits`.
    pub fn fixed(bits: u8) -> Fp64Mode {
        Fp64Mode::Emulated {
            strategy: Strategy::Eager,
            mantissa: Mantissa::Fixed { bits: bits as i32 },
        }
    }
}

/// Column-major operand transposition.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Op {
    N,
    T,
}

impl Op {
    pub(super) fn cb(self) -> cb::cublasOperation_t {
        match self {
            Op::N => cb::cublasOperation_t::CUBLAS_OP_N,
            Op::T => cb::cublasOperation_t::CUBLAS_OP_T,
        }
    }
}

/// The shape of a GEMM: C (m x n) = alpha op(A) op(B) + beta C with op(A) m x k and op(B)
/// k x n, `batch` products for a strided batch (1 otherwise).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GemmDims {
    pub ta: Op,
    pub tb: Op,
    pub m: usize,
    pub n: usize,
    pub k: usize,
    pub batch: usize,
}

impl GemmDims {
    /// One product.
    pub fn new(ta: Op, tb: Op, m: usize, n: usize, k: usize) -> GemmDims {
        GemmDims {
            ta,
            tb,
            m,
            n,
            k,
            batch: 1,
        }
    }

    /// `batch` products of this shape.
    pub fn batch(self, batch: usize) -> GemmDims {
        GemmDims { batch, ..self }
    }

    /// The stored (rows, cols) of A and of B.
    pub(super) fn stored(&self) -> ((usize, usize), (usize, usize)) {
        let a = if self.ta == Op::N {
            (self.m, self.k)
        } else {
            (self.k, self.m)
        };
        let b = if self.tb == Op::N {
            (self.k, self.n)
        } else {
            (self.n, self.k)
        };
        (a, b)
    }

    /// m, n, k as cuBLAS's `int`.
    pub(super) fn i32s(&self) -> Result<[i32; 3], GpuError> {
        Ok([dim(self.m)?, dim(self.n)?, dim(self.k)?])
    }
}

/// A dimension, leading dimension or count as cuBLAS's `int`.
pub(super) fn dim(x: usize) -> Result<i32, GpuError> {
    i32::try_from(x).map_err(|_| GpuError::Contract(format!("BLAS dimension {x} exceeds i32")))
}

/// Whether `rows` x `cols` elements at `off` with leading dimension `ld`, `batch` times at
/// `stride`, lie inside `len` elements (and `ld` covers `rows`). An empty matrix needs only its
/// offset inside the buffer.
fn fits(
    len: usize,
    (off, ld, stride): (usize, usize, usize),
    (rows, cols): (usize, usize),
    batch: usize,
) -> bool {
    if ld < rows {
        return false;
    }
    if rows == 0 || cols == 0 || batch == 0 {
        return off <= len;
    }
    let last = (batch - 1)
        .checked_mul(stride)
        .and_then(|s| (cols - 1).checked_mul(ld).and_then(|c| c.checked_add(s)))
        .and_then(|s| s.checked_add(rows))
        .and_then(|s| s.checked_add(off));
    last.is_some_and(|end| end <= len)
}

/// A column-major sub-matrix a routine reads: `buf` from element `off` with leading dimension
/// `ld`, and in a strided batch the next product's `stride` elements on (0: shared).
#[derive(Debug)]
pub struct MatRef<'a, T> {
    buf: &'a DeviceBuf<T>,
    off: usize,
    ld: usize,
    stride: usize,
}

// by hand: the derives would require `T: Copy`
impl<T> Clone for MatRef<'_, T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> Copy for MatRef<'_, T> {}

impl<'a, T> MatRef<'a, T> {
    pub fn new(buf: &'a DeviceBuf<T>, off: usize, ld: usize) -> Self {
        MatRef {
            buf,
            off,
            ld,
            stride: 0,
        }
    }

    /// The same operand with `stride` elements between the products of a batch.
    pub fn stride(self, stride: usize) -> Self {
        MatRef { stride, ..self }
    }

    /// Whether `rows` x `cols` elements (`batch` products) lie inside the buffer.
    pub fn check(&self, rows: usize, cols: usize, batch: usize) -> bool {
        fits(
            self.buf.len(),
            (self.off, self.ld, self.stride),
            (rows, cols),
            batch,
        )
    }

    /// The address of the first element.
    pub fn ptr(&self) -> DevPtr<T> {
        self.buf.at(self.off)
    }

    /// The first element's offset in the buffer.
    pub fn offset(&self) -> usize {
        self.off
    }

    pub fn ld(&self) -> usize {
        self.ld
    }

    /// The stride between the products of a batch.
    pub fn batch_stride(&self) -> usize {
        self.stride
    }
}

/// A column-major sub-matrix a routine writes (as [`MatRef`]).
#[derive(Debug)]
pub struct MatMut<'a, T> {
    buf: &'a mut DeviceBuf<T>,
    off: usize,
    ld: usize,
    stride: usize,
}

impl<'a, T> MatMut<'a, T> {
    pub fn new(buf: &'a mut DeviceBuf<T>, off: usize, ld: usize) -> Self {
        MatMut {
            buf,
            off,
            ld,
            stride: 0,
        }
    }

    /// The same operand with `stride` elements between the products of a batch.
    pub fn stride(self, stride: usize) -> Self {
        MatMut { stride, ..self }
    }

    /// Whether `rows` x `cols` elements (`batch` products, which must not overlap) lie inside
    /// the buffer.
    pub fn check(&self, rows: usize, cols: usize, batch: usize) -> bool {
        fits(
            self.buf.len(),
            (self.off, self.ld, self.stride),
            (rows, cols),
            batch,
        ) && !(batch > 1 && self.stride == 0)
    }

    /// The address of the first element.
    pub fn ptr(&self) -> DevPtr<T> {
        self.buf.at(self.off)
    }

    pub fn ld(&self) -> usize {
        self.ld
    }

    /// The stride between the products of a batch.
    pub fn batch_stride(&self) -> usize {
        self.stride
    }
}

/// An operand by its device address alone (unchecked: [`Blas::gemm_raw`] is unsafe), with its
/// leading dimension and batch stride.
#[derive(Debug)]
pub struct MatPtr<T> {
    pub ptr: DevPtr<T>,
    pub ld: usize,
    pub stride: usize,
}

impl<T> Clone for MatPtr<T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> Copy for MatPtr<T> {}

/// The shape of one product of a grouped GEMM ([`Blas::gemm_grouped`]): C (m x n) =
/// alpha op(A) op(B) + beta C, column-major, with leading dimensions; the operands' device
/// addresses go in the pointer arrays.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GemmShape {
    pub m: i32,
    pub n: i32,
    pub k: i32,
    pub lda: i32,
    pub ldb: i32,
    pub ldc: i32,
}

/// The element of a floating-point GEMM: f64 (DGEMM; emulated on a handle in an emulated FP64
/// mode) or f32 (SGEMM, true FP32: the handles leave TF32 off). The methods are cuBLAS's
/// routines of the type; callers use [`Blas`]'s checked methods.
pub trait GemmScalar: DeviceRepr + Copy + std::fmt::Debug + 'static {
    /// Whether a single product runs as a strided batch of one (FP32: the entry point linus has
    /// always used, see the module docs).
    #[doc(hidden)]
    const ONE_AS_BATCH: bool;

    /// The routine's name in errors.
    #[doc(hidden)]
    const NAME: &'static str;

    /// Whether products of this element run in `arith` (f64: FP64, native or emulated; f32: FP32).
    fn runs_in(arith: linus_core::Arith) -> bool;

    /// cuBLAS's single GEMM.
    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    unsafe fn gemm(
        h: cb::cublasHandle_t,
        op: (cb::cublasOperation_t, cb::cublasOperation_t),
        mnk: [i32; 3],
        alpha: *const Self,
        a: (*const Self, i32),
        b: (*const Self, i32),
        beta: *const Self,
        c: (*mut Self, i32),
    ) -> cb::cublasStatus_t;

    /// cuBLAS's strided-batched GEMM (operands: address, leading dimension, stride).
    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    unsafe fn strided(
        h: cb::cublasHandle_t,
        op: (cb::cublasOperation_t, cb::cublasOperation_t),
        mnk: [i32; 3],
        alpha: *const Self,
        a: (*const Self, i32, i64),
        b: (*const Self, i32, i64),
        beta: *const Self,
        c: (*mut Self, i32, i64),
        batch: i32,
    ) -> cb::cublasStatus_t;

    /// cuBLAS's grouped GEMM (one product per group; the pointer arrays' device addresses).
    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    unsafe fn grouped(
        h: cb::cublasHandle_t,
        op: (*const cb::cublasOperation_t, *const cb::cublasOperation_t),
        mnk: (*const i32, *const i32, *const i32),
        alpha: *const Self,
        a: (u64, *const i32),
        b: (u64, *const i32),
        beta: *const Self,
        c: (u64, *const i32),
        groups: i32,
        size: *const i32,
    ) -> cb::cublasStatus_t;
}

/// The f64 and f32 impls of [`GemmScalar`], from cuBLAS's routine names.
macro_rules! gemm_scalar {
    (
        $t:ty,
        $name:literal,
        $one_as_batch:literal,
        $runs:pat,
        $gemm:ident,
        $strided:ident,
        $grouped:ident
    ) => {
        impl GemmScalar for $t {
            const ONE_AS_BATCH: bool = $one_as_batch;
            const NAME: &'static str = $name;

            fn runs_in(arith: linus_core::Arith) -> bool {
                matches!(arith, $runs)
            }

            unsafe fn gemm(
                h: cb::cublasHandle_t,
                op: (cb::cublasOperation_t, cb::cublasOperation_t),
                mnk: [i32; 3],
                alpha: *const Self,
                a: (*const Self, i32),
                b: (*const Self, i32),
                beta: *const Self,
                c: (*mut Self, i32),
            ) -> cb::cublasStatus_t {
                // SAFETY: the caller's contract (Blas::gemm).
                unsafe {
                    cb::$gemm(
                        h, op.0, op.1, mnk[0], mnk[1], mnk[2], alpha, a.0, a.1, b.0, b.1, beta,
                        c.0, c.1,
                    )
                }
            }

            unsafe fn strided(
                h: cb::cublasHandle_t,
                op: (cb::cublasOperation_t, cb::cublasOperation_t),
                mnk: [i32; 3],
                alpha: *const Self,
                a: (*const Self, i32, i64),
                b: (*const Self, i32, i64),
                beta: *const Self,
                c: (*mut Self, i32, i64),
                batch: i32,
            ) -> cb::cublasStatus_t {
                // SAFETY: the caller's contract (Blas::gemm_strided, Blas::gemm_raw).
                unsafe {
                    cb::$strided(
                        h, op.0, op.1, mnk[0], mnk[1], mnk[2], alpha, a.0, a.1, a.2, b.0, b.1, b.2,
                        beta, c.0, c.1, c.2, batch,
                    )
                }
            }

            unsafe fn grouped(
                h: cb::cublasHandle_t,
                op: (*const cb::cublasOperation_t, *const cb::cublasOperation_t),
                mnk: (*const i32, *const i32, *const i32),
                alpha: *const Self,
                a: (u64, *const i32),
                b: (u64, *const i32),
                beta: *const Self,
                c: (u64, *const i32),
                groups: i32,
                size: *const i32,
            ) -> cb::cublasStatus_t {
                // SAFETY: the caller's contract (Blas::gemm_grouped).
                unsafe {
                    cb::$grouped(
                        h,
                        op.0,
                        op.1,
                        mnk.0,
                        mnk.1,
                        mnk.2,
                        alpha,
                        a.0 as *const *const $t,
                        a.1,
                        b.0 as *const *const $t,
                        b.1,
                        beta,
                        c.0 as *const *mut $t,
                        c.1,
                        groups,
                        size,
                    )
                }
            }
        }
    };
}

gemm_scalar!(
    f64,
    "DGEMM",
    false,
    linus_core::Arith::Fp64 | linus_core::Arith::Fp64Emulated { .. },
    cublasDgemm_v2,
    cublasDgemmStridedBatched,
    cublasDgemmGroupedBatched
);
gemm_scalar!(
    f32,
    "SGEMM",
    true,
    linus_core::Arith::Fp32,
    cublasSgemm_v2,
    cublasSgemmStridedBatched,
    cublasSgemmGroupedBatched
);

type SetI32 = unsafe extern "C" fn(cb::cublasHandle_t, i32) -> cb::cublasStatus_t;
type SetPtr = unsafe extern "C" fn(cb::cublasHandle_t, *mut i32) -> cb::cublasStatus_t;

/// The emulation setters missing from cudarc's bindings, looked up in the loaded libcublas.
fn sym<T: Copy>(name: &str) -> Result<T, GpuError> {
    // SAFETY: cudarc has loaded libcublas; the symbol types match cublas_api.h (13.1).
    unsafe {
        let lib = cb::culib();
        lib.get::<T>(name.as_bytes()).map(|s| *s).map_err(|e| {
            GpuError::NotCovered(format!(
                "libcublas lacks {name} (cuBLAS 13.1 or newer needed): {e}"
            ))
        })
    }
}

/// A cuBLAS handle on the device's stream, in one FP64 arithmetic for its lifetime.
pub struct Blas {
    blas: CudaBlas,
    mode: Fp64Mode,
    bits: Option<DeviceBuf<i32>>,
}

impl Blas {
    /// A handle in cuBLAS's default math (native FP64).
    pub fn new(gpu: &Gpu) -> Result<Self, GpuError> {
        let blas = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            CudaBlas::new(gpu.stream().clone())
        }))
        .map_err(|_| GpuError::Unavailable("cannot load libcublas".into()))??;
        Ok(Blas {
            blas,
            mode: Fp64Mode::Native,
            bits: None,
        })
    }

    /// A handle whose FP64 routines run in `mode` (configured once, here).
    pub fn with_mode(gpu: &Gpu, mode: Fp64Mode) -> Result<Self, GpuError> {
        let mut b = Blas::new(gpu)?;
        b.configure(gpu, mode)?;
        Ok(b)
    }

    pub fn handle(&self) -> cb::cublasHandle_t {
        *self.blas.handle()
    }

    pub fn mode(&self) -> Fp64Mode {
        self.mode
    }

    /// cuBLAS's version (e.g. 130201).
    pub fn version(&self) -> Result<i32, GpuError> {
        let mut v = 0;
        // SAFETY: valid handle, out pointer.
        unsafe { cb::cublasGetVersion_v2(self.handle(), &mut v) }.result()?;
        Ok(v)
    }

    /// Sets the handle's FP64 arithmetic: cuBLAS's math mode and emulation settings (native:
    /// the default math, set explicitly).
    pub(crate) fn configure(&mut self, gpu: &Gpu, mode: Fp64Mode) -> Result<(), GpuError> {
        let h = self.handle();
        match mode {
            Fp64Mode::Native => {
                // SAFETY: valid handle.
                unsafe { cb::cublasSetMathMode(h, cb::cublasMath_t::CUBLAS_DEFAULT_MATH) }
                    .result()?;
                if self.bits.is_some() {
                    let f: SetPtr = sym("cublasSetFixedPointEmulationMantissaBitCountPointer")?;
                    unsafe { f(h, std::ptr::null_mut()) }.result()?;
                    self.bits = None;
                }
            }
            Fp64Mode::Emulated { strategy, mantissa } => {
                unsafe {
                    cb::cublasSetMathMode(h, cb::cublasMath_t::CUBLAS_FP64_EMULATED_FIXEDPOINT_MATH)
                }
                .result()?;
                let st = match strategy {
                    Strategy::Performant => {
                        cb::cublasEmulationStrategy_t::CUBLAS_EMULATION_STRATEGY_PERFORMANT
                    }
                    Strategy::Eager => {
                        cb::cublasEmulationStrategy_t::CUBLAS_EMULATION_STRATEGY_EAGER
                    }
                };
                unsafe { cb::cublasSetEmulationStrategy(h, st) }.result()?;
                let control: SetI32 = sym("cublasSetFixedPointEmulationMantissaControl")?;
                let max_bits: SetI32 = sym("cublasSetFixedPointEmulationMaxMantissaBitCount")?;
                let offset: SetI32 = sym("cublasSetFixedPointEmulationMantissaBitOffset")?;
                // cudaEmulationMantissaControl: DYNAMIC = 0, FIXED = 1
                match mantissa {
                    Mantissa::Dynamic {
                        offset: o,
                        max_bits: m,
                    } => unsafe {
                        control(h, 0).result()?;
                        max_bits(h, m).result()?;
                        offset(h, o).result()?;
                    },
                    Mantissa::Fixed { bits } => unsafe {
                        control(h, 1).result()?;
                        max_bits(h, bits).result()?;
                        offset(h, 0).result()?;
                    },
                }
                if self.bits.is_none() {
                    let buf = gpu.alloc_zeros::<i32>(1)?;
                    let f: SetPtr = sym("cublasSetFixedPointEmulationMantissaBitCountPointer")?;
                    unsafe { f(h, buf.ptr().as_mut_ptr()) }.result()?;
                    self.bits = Some(buf);
                }
            }
        }
        self.mode = mode;
        Ok(())
    }

    /// The mantissa bits the last emulated call retained (-1: it ran in native FP64), or None
    /// in native mode. Synchronizes.
    pub fn retained_bits(&self, gpu: &Gpu) -> Result<Option<i32>, GpuError> {
        match &self.bits {
            Some(b) => Ok(Some(gpu.download(b)?[0])),
            None => Ok(None),
        }
    }

    /// One product C = alpha op(A) op(B) + beta C (`d.batch` 1), `s` = (alpha, beta).
    pub fn gemm<T: GemmScalar>(
        &self,
        d: GemmDims,
        s: (T, T),
        a: MatRef<'_, T>,
        b: MatRef<'_, T>,
        c: MatMut<'_, T>,
    ) -> Result<(), GpuError> {
        if d.batch != 1 {
            return Err(GpuError::Contract(format!(
                "{}: one product, not a batch of {}",
                T::NAME,
                d.batch
            )));
        }
        if T::ONE_AS_BATCH {
            return self.gemm_strided(d, s, a.stride(0), b.stride(0), c.stride(0));
        }
        let mnk = d.i32s()?;
        let (la, lb, lc) = (dim(a.ld())?, dim(b.ld())?, dim(c.ld())?);
        let ((ar, ac), (br, bc)) = d.stored();
        if !a.check(ar, ac, 1) || !b.check(br, bc, 1) || !c.check(d.m, d.n, 1) {
            return Err(GpuError::Contract(format!(
                "{} operand outside its buffer",
                T::NAME
            )));
        }
        // SAFETY: device pointers of live buffers; every operand checked to lie inside its
        // buffer above (an empty product too: cuBLAS returns at once, or scales C by beta).
        unsafe {
            T::gemm(
                self.handle(),
                (d.ta.cb(), d.tb.cb()),
                mnk,
                &s.0,
                (a.ptr().as_ptr(), la),
                (b.ptr().as_ptr(), lb),
                &s.1,
                (c.ptr().as_mut_ptr(), lc),
            )
        }
        .result()?;
        Ok(())
    }

    /// `d.batch` products C_b = alpha op(A_b) op(B_b) + beta C_b, the operands of product b at
    /// their offsets plus b times their strides (stride 0 shares an operand; the outputs must
    /// not overlap): cuBLAS's strided-batched routine whatever the count. Nothing when the batch
    /// or C is empty.
    pub fn gemm_strided<T: GemmScalar>(
        &self,
        d: GemmDims,
        s: (T, T),
        a: MatRef<'_, T>,
        b: MatRef<'_, T>,
        c: MatMut<'_, T>,
    ) -> Result<(), GpuError> {
        let mnk = d.i32s()?;
        let (la, lb, lc, nb) = (dim(a.ld())?, dim(b.ld())?, dim(c.ld())?, dim(d.batch)?);
        if d.batch == 0 || d.m == 0 || d.n == 0 {
            return Ok(());
        }
        let ((ar, ac), (br, bc)) = d.stored();
        if !a.check(ar, ac, d.batch) || !b.check(br, bc, d.batch) || !c.check(d.m, d.n, d.batch) {
            return Err(GpuError::Contract(format!(
                "batched {} operand outside its buffer",
                T::NAME
            )));
        }
        // SAFETY: device pointers of live buffers; every product's operands checked to lie
        // inside their buffers above.
        unsafe {
            T::strided(
                self.handle(),
                (d.ta.cb(), d.tb.cb()),
                mnk,
                &s.0,
                (a.ptr().as_ptr(), la, a.batch_stride() as i64),
                (b.ptr().as_ptr(), lb, b.batch_stride() as i64),
                &s.1,
                (c.ptr().as_mut_ptr(), lc, c.batch_stride() as i64),
                nb,
            )
        }
        .result()?;
        Ok(())
    }

    /// [`Blas::gemm_strided`] on operands given by address alone (one product is a batch of
    /// one). Nothing when the batch or C is empty.
    ///
    /// # Safety
    /// Every product's operands must lie in live device buffers of `T`; the C matrices must
    /// not overlap.
    pub unsafe fn gemm_raw<T: GemmScalar>(
        &self,
        d: GemmDims,
        s: (T, T),
        a: MatPtr<T>,
        b: MatPtr<T>,
        c: MatPtr<T>,
    ) -> Result<(), GpuError> {
        let mnk = d.i32s()?;
        let (la, lb, lc) = (dim(a.ld.max(1))?, dim(b.ld.max(1))?, dim(c.ld.max(1))?);
        let nb = dim(d.batch)?;
        if d.batch == 0 || d.m == 0 || d.n == 0 {
            return Ok(());
        }
        // SAFETY: the caller's contract.
        unsafe {
            T::strided(
                self.handle(),
                (d.ta.cb(), d.tb.cb()),
                mnk,
                &s.0,
                (a.ptr.as_ptr(), la, a.stride as i64),
                (b.ptr.as_ptr(), lb, b.stride as i64),
                &s.1,
                (c.ptr.as_mut_ptr(), lc, c.stride as i64),
                nb,
            )
        }
        .result()?;
        Ok(())
    }

    /// One group per product (sizes 1): C_i = alpha op(A_i) op(B_i) + beta C_i with each
    /// product's own m, n, k and leading dimensions in `shapes` and its operands' device
    /// addresses in `ptrs[off..]`: the A pointers, then the B pointers, then the C pointers
    /// (`shapes.len()` each, as u64). One launch for many small products of different sizes;
    /// in f64 it is emulated on an emulated handle, as cuBLAS's other GEMMs.
    ///
    /// # Safety
    /// Every address must point to a live device matrix of its product's shape and leading
    /// dimension; the C matrices must not overlap each other or any A or B.
    pub unsafe fn gemm_grouped<T: GemmScalar>(
        &self,
        op: (Op, Op),
        s: (T, T),
        shapes: &[GemmShape],
        ptrs: &DeviceBuf<u64>,
        off: usize,
    ) -> Result<(), GpuError> {
        let np = shapes.len();
        if np == 0 {
            return Ok(());
        }
        let end = np.checked_mul(3).and_then(|n3| n3.checked_add(off));
        if end.is_none_or(|end| end > ptrs.len()) || np > i32::MAX as usize {
            return Err(GpuError::Contract(
                "grouped GEMM: pointer arrays outside their buffer".into(),
            ));
        }
        let (ta, tb) = op;
        for s in shapes {
            let (ar, br) = (
                if ta == Op::N { s.m } else { s.k },
                if tb == Op::N { s.k } else { s.n },
            );
            if s.m < 0
                || s.n < 0
                || s.k < 0
                || s.lda < ar.max(1)
                || s.ldb < br.max(1)
                || s.ldc < s.m.max(1)
            {
                return Err(GpuError::Contract(format!("grouped GEMM: bad shape {s:?}")));
            }
        }
        let tav = vec![ta.cb(); np];
        let tbv = vec![tb.cb(); np];
        let m: Vec<i32> = shapes.iter().map(|s| s.m).collect();
        let n: Vec<i32> = shapes.iter().map(|s| s.n).collect();
        let k: Vec<i32> = shapes.iter().map(|s| s.k).collect();
        let lda: Vec<i32> = shapes.iter().map(|s| s.lda).collect();
        let ldb: Vec<i32> = shapes.iter().map(|s| s.ldb).collect();
        let ldc: Vec<i32> = shapes.iter().map(|s| s.ldc).collect();
        let al = vec![s.0; np];
        let be = vec![s.1; np];
        let size = vec![1i32; np];
        // the A, B and C pointer arrays, one after the other
        let base = ptrs.at(off);
        // SAFETY: host arrays of np entries; the pointer arrays lie in `ptrs` (checked); the
        // matrices are the caller's contract.
        unsafe {
            T::grouped(
                self.handle(),
                (tav.as_ptr(), tbv.as_ptr()),
                (m.as_ptr(), n.as_ptr(), k.as_ptr()),
                al.as_ptr(),
                (base.addr(), lda.as_ptr()),
                ((base + np).addr(), ldb.as_ptr()),
                be.as_ptr(),
                ((base + 2 * np).addr(), ldc.as_ptr()),
                np as i32,
                size.as_ptr(),
            )
        }
        .result()?;
        Ok(())
    }

    /// B <- L^-1 B (op = N) or L^-T B (op = T) for the lower-triangular m x m matrix L
    /// (non-unit diagonal; its upper triangle is not read) and the m x n matrix B.
    pub fn trsm_lower_left(
        &self,
        op: Op,
        (m, n): (usize, usize),
        l: MatRef<'_, f64>,
        b: MatMut<'_, f64>,
    ) -> Result<(), GpuError> {
        let (mi, ni, ll, lb) = (dim(m)?, dim(n)?, dim(l.ld())?, dim(b.ld())?);
        if m == 0 || n == 0 {
            return Ok(());
        }
        if !l.check(m, m, 1) || !b.check(m, n, 1) {
            return Err(GpuError::Contract("TRSM operand outside its buffer".into()));
        }
        let one = 1.0f64;
        // SAFETY: device pointers of live buffers, both operands checked above.
        unsafe {
            cb::cublasDtrsm_v2(
                self.handle(),
                cb::cublasSideMode_t::CUBLAS_SIDE_LEFT,
                cb::cublasFillMode_t::CUBLAS_FILL_MODE_LOWER,
                op.cb(),
                cb::cublasDiagType_t::CUBLAS_DIAG_NON_UNIT,
                mi,
                ni,
                &one,
                l.ptr().as_ptr(),
                ll,
                b.ptr().as_mut_ptr(),
                lb,
            )
        }
        .result()?;
        Ok(())
    }

    /// C = alpha A A^T + beta C, lower triangle of the n x n matrix C only (the upper one is
    /// not written), A n x k, `s` = (alpha, beta). Native FP64 (cuBLAS emulates GEMMs, not
    /// SYRK).
    pub fn syrk_lower(
        &self,
        (n, k): (usize, usize),
        s: (f64, f64),
        a: MatRef<'_, f64>,
        c: MatMut<'_, f64>,
    ) -> Result<(), GpuError> {
        let (ni, ki, la, lc) = (dim(n)?, dim(k)?, dim(a.ld())?, dim(c.ld())?);
        if n == 0 {
            return Ok(());
        }
        if !a.check(n, k, 1) || !c.check(n, n, 1) {
            return Err(GpuError::Contract("SYRK operand outside its buffer".into()));
        }
        // SAFETY: device pointers of live buffers, both operands checked above.
        unsafe {
            cb::cublasDsyrk_v2(
                self.handle(),
                cb::cublasFillMode_t::CUBLAS_FILL_MODE_LOWER,
                cb::cublasOperation_t::CUBLAS_OP_N,
                ni,
                ki,
                &s.0,
                a.ptr().as_ptr(),
                la,
                &s.1,
                c.ptr().as_mut_ptr(),
                lc,
            )
        }
        .result()?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::fits;

    #[test]
    fn operand_bounds() {
        // 3 x 2 at offset 1 with ld 4: elements 1..8 (the last column 5..8)
        assert!(fits(8, (1, 4, 0), (3, 2), 1));
        assert!(!fits(7, (1, 4, 0), (3, 2), 1));
        // ld below the rows
        assert!(!fits(100, (0, 2, 0), (3, 2), 1));
        // a batch of 3 with stride 10: the last product ends at 20 + 4 + 3 = 27
        assert!(fits(27, (0, 4, 10), (3, 2), 3));
        assert!(!fits(26, (0, 4, 10), (3, 2), 3));
        // empty: only the offset
        assert!(fits(5, (5, 3, 0), (3, 0), 1) && !fits(5, (6, 3, 0), (3, 0), 1));
        // overflow is outside, not a wrap
        assert!(!fits(usize::MAX, (usize::MAX - 1, 4, 0), (3, 2), 1));
    }
}
