//! linus-gpu: GPU kernels and engines for linus. The `linus` crate's `gpu` feature wraps the
//! engines behind its builder traits (`backend: "gpu"`, docs/gpu.md); this crate's API is
//! unstable.
//!
//! Everything that touches a device is behind the `gpu` feature, off by default, so CPU-only
//! builds never need CUDA. With the feature on, CUDA is still not needed at build time: the
//! driver, NVRTC, cuBLAS and cuSOLVER are loaded when a GPU is first opened, and a machine
//! without them gets [`GpuError::Unavailable`] from `Gpu::new`, not a link failure.
//!
//! Layers:
//! - The device layer (`cuda`, feature `gpu`): `Gpu` (device, stream, memory, copies,
//!   run-time compilation with an on-disk cache, launch, timing marks), `DeviceBuf` and
//!   `DevPtr` (the device address of a buffer's element, typed by the element: kernel
//!   arguments and library operands never spell an element size), `HostBuf` (page-locked
//!   host memory), `StreamedStore` (host memory streamed through device staging buffers, the
//!   copy stream and its fences inside), `Kernel`, `Launch`, `Arg`, `blas` (a cuBLAS handle in
//!   one FP64 arithmetic: GEMMs on typed operands in FP64, FP32 and INT8, SYRK, TRSM) and
//!   `solver` (symmetric eigensolver, Cholesky). It is the only code that calls a vendor
//!   runtime; a HIP backend would sit beside it and export the same items.
//! - The engines, each the device counterpart of a linus CPU builder, with plain host
//!   matrices at their boundary (linus wraps them behind its own traits):
//!   - [`df`]: density-fitted J and K (`GpuDfJk`; the CPU's `DfJk`);
//!   - [`xc`]: the XC energy and potential (`GpuXc`; the CPU's `XcIntegrator`), and the XC
//!     family's shared parts: the per-batch layout, the density pass and the host functional
//!     evaluator the other XC engines are composed on;
//!   - [`grad`]: analytic gradients, the DF two-electron term (`GpuDfGrad`) and the XC term
//!     with grid response (`GpuXcGrad`);
//!   - [`hess`]: analytic Hessians' explicit terms, the DF two-electron part (`GpuDfHess`: the
//!     derivative Fock matrices of the CPHF right-hand sides and the RI cross terms);
//!   - [`pcm`]: the implicit solvent's surface terms, the integrals of the cavity's surface
//!     charges (`GpuPcmInts`; the CPU's `pcm_ints::SurfacePlan`) and the dense charge solve
//!     (`GpuPcmSolver`; the CPU's `PcmOperator`);
//!   - [`response`]: the CPHF/CPKS products of Hessians and polarizabilities on the tensor
//!     and grid of those engines, the projected response J/K (`GpuDfResponse`; the CPU's
//!     `DfResponseJk`) and the XC kernel contraction (`GpuXcKernel`; the CPU's
//!     `XcIntegrator::kernel_contract_occ`).
//! - `engine` (feature `gpu`): what every engine shares: `Phases` (an engine's record as named
//!   phases in seconds, which its caller reports with one loop) and `Timeline` (phase timing by
//!   events on the stream).
//! - `linalg` (feature `gpu`): `Gemm`, the GEMM service of the engines: every product names its
//!   arithmetic (linus-core's `Arith`), and the service runs it on the cuBLAS handle of that
//!   arithmetic (native FP64 and FP32, or FP64 emulated with a fixed mantissa); `Blas1`, the
//!   vector operations (scale, axpby) every engine shares, each with its rounding fixed by its
//!   definition.
//! - [`emu`]: FP64 GEMMs emulated on INT8 tensor cores (Ozaki scheme I), with operands sliced
//!   once and reused (the vendor INT8 GEMM is `blas::Blas::gemm_i8`).
//! - [`detsum`]: fixed-point accumulation on integer atomics, bit-identical to the CPU's
//!   accumulator (linus-num's `det::FixedAcc`).
//! - [`kernels`]: the kernel source generators (collocation, vgemm; the three-center and PCM
//!   kernels generated offline by tools/intgen) and, with `gpu`, the registry of the
//!   hand-written modules. Hand-written kernels are files under `crates/linus-gpu/device/`
//!   (`include_str!`), each module declared once by the engine that owns it.
//! - [`eri3c`]: three-center integrals and their first derivatives built on the device.
//! - [`scf`]: the device-resident SCF loop (orthogonalizer, diagonalization, densities, DIIS,
//!   energy and convergence checks on the device), with the Fock terms behind a trait.
//! - `harness` (feature `testkit`, on in the crate's tests and examples): fixtures (linus-core
//!   structures, basis sets, grids) and comparisons with the CPU code.
//! - [`cpu`]: the items taken from linus-ints, linus-grid and linus-xc, in one place.
//!
//! Kernel rules:
//! - FP64 unless a kernel's documentation says otherwise.
//! - Deterministic run to run on a device: each output written by one thread, or summed with
//!   [`detsum`]; no floating-point atomics.
//! - Sources in the common CUDA/HIP subset: C-style, no inline PTX, the few differing names
//!   behind `device/portable.h`, the warp size from the device (`LINUS_WARP_SIZE`), never a
//!   literal 32. Vendor-specific variants (tensor cores, emulated GEMMs) only as optional
//!   kernels beside a portable one.
//! - Large GEMMs go through the vendor BLAS, never hand-written kernels.

pub mod cpu;
pub mod detsum;
pub mod df;
#[cfg(feature = "gpu")]
mod df_deriv;
pub mod emu;
pub mod eri3c;
pub mod error;
pub mod grad;
pub mod gridk;
#[cfg(any(test, feature = "testkit"))]
pub mod harness;
pub mod hess;
pub mod jdirect;
mod jdirect_hermite;
#[cfg(feature = "gpu")]
mod jengine;
pub mod kernels;
pub mod pcm;
pub mod response;
pub mod scf;
pub mod xc;

#[cfg(feature = "gpu")]
mod cuda;
#[cfg(feature = "gpu")]
pub mod engine;
#[cfg(feature = "gpu")]
pub mod linalg;

#[cfg(feature = "gpu")]
pub use cuda::{
    blas, kernel_source, nvrtc, pool, solver, Arg, Copies, DevPtr, DeviceBuf, DeviceInfo, Fence,
    Gpu, HostBuf, Kernel, Launch, Mark, PoolUsage, StreamedStore,
};
pub use error::GpuError;
