//! The CUDA backend of the device layer: the only code in linus-gpu that talks to a vendor
//! runtime (through cudarc: driver, NVRTC, cuBLAS, cuSOLVER).
//!
//! Everything above it (kernels, detsum, harness) uses the items re-exported at the crate
//! root: [`Gpu`] (device, stream, memory, copies, run-time compilation, launch, timing),
//! [`DeviceBuf`], [`Kernel`], [`Launch`], [`Arg`], [`blas::Blas`] and [`solver::Solver`]. A
//! HIP backend would be a sibling module (`hip/`, on hiprtc, rocBLAS and rocSOLVER) exporting
//! the same items under its own feature; kernel sources are already in the common CUDA/HIP
//! subset (`device/portable.h`).

pub mod blas;
pub mod device;
pub mod host;
pub mod nvrtc;
mod ozaki;
pub mod pool;
pub mod solver;
pub mod stream;

pub use device::{kernel_source, Arg, DeviceInfo, Gpu, Kernel, Launch, Mark};
pub use host::HostBuf;
pub use pool::{DevPtr, DeviceBuf, PoolUsage};
pub use stream::{Copies, Fence, StreamedStore};

/// The out-of-memory error of a CUDA library's or the driver's own allocation, with the
/// device's used and total bytes as the driver reports them now (0 when it cannot).
pub(crate) fn device_oom() -> crate::GpuError {
    let (free, total) = cudarc::driver::result::mem_get_info().unwrap_or((0, 0));
    crate::GpuError::OutOfMemory {
        requested: 0,
        in_use: total.saturating_sub(free),
        limit: total,
    }
}

/// Whether linus's own budget on the device is nearly spent (less than a tenth of the handle's
/// limit free): a library's generic failure (cuSOLVER's internal error, which is how it reports
/// a failed allocation inside a factorization) is then taken for an out-of-memory. It reads
/// linus's accounting, not the device's free memory, so that on a shared card another process's
/// memory does not decide how a failure of this one is classified.
pub(crate) fn memory_pressure(budget: linus_core::Budget) -> bool {
    budget.free() < budget.limit / 10
}

/// Whether a driver status leaves the CUDA context unusable for the rest of the process (a
/// "sticky" error: a kernel faulted, the device asserted or failed).
fn poisons_context(r: cudarc::driver::sys::CUresult) -> bool {
    use cudarc::driver::sys::CUresult::*;
    matches!(
        r,
        CUDA_ERROR_ILLEGAL_ADDRESS
            | CUDA_ERROR_LAUNCH_FAILED
            | CUDA_ERROR_HARDWARE_STACK_ERROR
            | CUDA_ERROR_ILLEGAL_INSTRUCTION
            | CUDA_ERROR_MISALIGNED_ADDRESS
            | CUDA_ERROR_INVALID_ADDRESS_SPACE
            | CUDA_ERROR_INVALID_PC
            | CUDA_ERROR_ASSERT
            | CUDA_ERROR_ECC_UNCORRECTABLE
            | CUDA_ERROR_LAUNCH_TIMEOUT
    )
}

impl From<cudarc::driver::DriverError> for crate::GpuError {
    fn from(e: cudarc::driver::DriverError) -> Self {
        // the driver's own allocations (the stream-ordered pool, library workspaces)
        if e.0 == cudarc::driver::sys::CUresult::CUDA_ERROR_OUT_OF_MEMORY {
            return device_oom();
        }
        // a fault that poisons the context: no later call of this process can succeed
        if poisons_context(e.0) {
            crate::error::mark_context_lost();
        }
        let status = match e.error_string() {
            Ok(s) => format!("{:?} ({})", e.0, s.to_string_lossy()),
            Err(_) => format!("{:?}", e.0),
        };
        crate::GpuError::Library {
            lib: crate::error::Lib::Driver,
            call: None,
            status,
        }
    }
}

impl From<cudarc::cublas::result::CublasError> for crate::GpuError {
    fn from(e: cudarc::cublas::result::CublasError) -> Self {
        if e.0 == cudarc::cublas::sys::cublasStatus_t::CUBLAS_STATUS_ALLOC_FAILED {
            return device_oom();
        }
        crate::GpuError::Library {
            lib: crate::error::Lib::Cublas,
            call: None,
            status: format!("{:?}", e.0),
        }
    }
}

/// An NVRTC status of `call` (not a compile error: those carry the log, `GpuError::Compile`).
pub(crate) fn nvrtc_error(
    call: &'static str,
) -> impl Fn(cudarc::nvrtc::result::NvrtcError) -> crate::GpuError {
    move |e| crate::GpuError::Library {
        lib: crate::error::Lib::Nvrtc,
        call: Some(call),
        status: format!("{:?}", e.0),
    }
}
