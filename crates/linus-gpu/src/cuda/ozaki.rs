//! INT8 GEMMs with exact INT32 accumulation (cuBLAS `cublasGemmEx`, `CUBLAS_COMPUTE_32I`): the
//! vendor part of the Ozaki-scheme FP64 emulation in [`crate::emu`]. rocBLAS has the
//! equivalent (`rocblas_gemm_ex` with `rocblas_datatype_i8_r` in and `i32_r` out), so a HIP
//! backend exports the same method.
//!
//! Integer products are exact: no rounding, only overflow, which the caller excludes by
//! bounding the inner dimension (digits of at most 2^7 in magnitude give products of at most
//! 2^14, so 2^17 of them fit an INT32 accumulator).

use super::blas::{dim, Blas, GemmDims, MatMut, MatRef};
use crate::error::GpuError;
use cudarc::cublas::sys as cb;

impl Blas {
    /// `d.batch` integer products C_b = alpha op(A_b) op(B_b) + beta C_b, `s` = (alpha, beta):
    /// A and B INT8, C INT32 (exact; the caller keeps the sums below 2^31), operands as for
    /// [`Blas::gemm_strided`] (stride 0 shares an operand; the outputs must not overlap). One
    /// product runs `cublasGemmEx`, more the strided-batched one. Offsets, leading dimensions
    /// and strides of A and B must be multiples of 4; the tensor-core kernels want op(A) = T
    /// and op(B) = N (both operands contiguous along the inner dimension).
    pub fn gemm_i8(
        &self,
        d: GemmDims,
        s: (i32, i32),
        a: MatRef<'_, i8>,
        b: MatRef<'_, i8>,
        c: MatMut<'_, i32>,
    ) -> Result<(), GpuError> {
        let mnk = d.i32s()?;
        let (la, lb, lc, nb) = (dim(a.ld())?, dim(b.ld())?, dim(c.ld())?, dim(d.batch)?);
        if d.batch == 0 || d.m == 0 || d.n == 0 {
            return Ok(());
        }
        let ((ar, ac), (br, bc)) = d.stored();
        if !a.check(ar, ac, d.batch) || !b.check(br, bc, d.batch) || !c.check(d.m, d.n, d.batch) {
            return Err(GpuError::Contract(
                "INT8 GEMM operand outside its buffer".into(),
            ));
        }
        let aligned = |x: usize| x.is_multiple_of(4);
        if ![
            a.offset(),
            a.ld(),
            a.batch_stride(),
            b.offset(),
            b.ld(),
            b.batch_stride(),
        ]
        .into_iter()
        .all(aligned)
        {
            return Err(GpuError::Contract(
                "INT8 GEMM: offsets, leading dimensions and strides of the INT8 operands must be multiples of 4".into(),
            ));
        }
        let (pa, pb, pc) = (a.ptr().as_ptr(), b.ptr().as_ptr(), c.ptr().as_mut_ptr());
        let (alpha, beta) = s;
        // SAFETY: device pointers of live buffers; every product's operands checked to lie
        // inside their buffers; the scalars are INT32 as CUBLAS_COMPUTE_32I requires.
        unsafe {
            if d.batch == 1 {
                cb::cublasGemmEx(
                    self.handle(),
                    d.ta.cb(),
                    d.tb.cb(),
                    mnk[0],
                    mnk[1],
                    mnk[2],
                    &alpha as *const i32 as *const _,
                    pa as *const _,
                    cb::cudaDataType::CUDA_R_8I,
                    la,
                    pb as *const _,
                    cb::cudaDataType::CUDA_R_8I,
                    lb,
                    &beta as *const i32 as *const _,
                    pc as *mut _,
                    cb::cudaDataType::CUDA_R_32I,
                    lc,
                    cb::cublasComputeType_t::CUBLAS_COMPUTE_32I,
                    cb::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
                )
            } else {
                cb::cublasGemmStridedBatchedEx(
                    self.handle(),
                    d.ta.cb(),
                    d.tb.cb(),
                    mnk[0],
                    mnk[1],
                    mnk[2],
                    &alpha as *const i32 as *const _,
                    pa as *const _,
                    cb::cudaDataType::CUDA_R_8I,
                    la,
                    a.batch_stride() as i64,
                    pb as *const _,
                    cb::cudaDataType::CUDA_R_8I,
                    lb,
                    b.batch_stride() as i64,
                    &beta as *const i32 as *const _,
                    pc as *mut _,
                    cb::cudaDataType::CUDA_R_32I,
                    lc,
                    c.batch_stride() as i64,
                    nb,
                    cb::cublasComputeType_t::CUBLAS_COMPUTE_32I,
                    cb::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
                )
            }
        }
        .result()?;
        Ok(())
    }
}
