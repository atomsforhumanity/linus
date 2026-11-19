//! cuSOLVER: the symmetric eigensolver (the SCF's `eigh`), Cholesky and LU factorizations
//! and their solves (the density-fitting metric, the PCM's charge solve).

use super::device::Gpu;
use super::pool::{DeviceBuf, PoolMeter};
use crate::error::{GpuError, Lib};
use cudarc::cusolver::sys as cs;

/// A cuSOLVER dense handle on the device's stream.
pub struct Solver {
    h: cs::cusolverDnHandle_t,
    /// linus's memory accounting on the device (the classification of internal errors, [`err`]).
    meter: PoolMeter,
}

// SAFETY: a cuSOLVER handle may be used from any host thread, one call at a time. Solver is not
// Sync (the handle is a raw pointer), so every call through it comes from the one thread that
// owns it at that moment: moving it to another thread cannot make two calls concurrent.
unsafe impl Send for Solver {}

/// The error of cuSOLVER's status of `call`: its allocation failure, and its internal error
/// while linus's memory budget on the device is nearly spent (as cuSOLVER reports a failed
/// workspace allocation inside a factorization; [`super::memory_pressure`]), are out-of-memory
/// errors, so that a setup falls back as for any other; any other status is the library's
/// ([`GpuError::Library`]: an error of the calculation).
fn err<'a>(
    meter: &'a PoolMeter,
    call: &'static str,
) -> impl Fn(cudarc::cusolver::result::CusolverError) -> GpuError + 'a {
    move |e| {
        use cs::cusolverStatus_t as S;
        match e.0 {
            S::CUSOLVER_STATUS_ALLOC_FAILED => super::device_oom(),
            S::CUSOLVER_STATUS_INTERNAL_ERROR if super::memory_pressure(meter.budget()) => {
                super::device_oom()
            }
            S::CUSOLVER_STATUS_NOT_SUPPORTED => {
                GpuError::NotCovered(format!("cuSOLVER {call}: not supported"))
            }
            s => GpuError::Library {
                lib: Lib::Cusolver,
                call: Some(call),
                status: format!("{s:?}"),
            },
        }
    }
}

/// A nonzero `info` of the cuSOLVER call `op` as its numerical failure.
fn check_info(op: &'static str, info: i32) -> Result<(), GpuError> {
    match info {
        0 => Ok(()),
        info => Err(GpuError::Numerical { op, info }),
    }
}

impl Solver {
    pub fn new(gpu: &Gpu) -> Result<Self, GpuError> {
        let meter = gpu.pool().meter();
        let mut h: cs::cusolverDnHandle_t = std::ptr::null_mut();
        // cudarc panics when it cannot load libcusolver
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            unsafe { cs::cusolverDnCreate(&mut h) }.result()
        }))
        .map_err(|_| GpuError::Unavailable("cannot load libcusolver".into()))?
        .map_err(err(&meter, "Create"))?;
        // SAFETY: live handle; the stream outlives it (owned by gpu).
        unsafe { cs::cusolverDnSetStream(h, gpu.stream().cu_stream() as cs::cudaStream_t) }
            .result()
            .map_err(err(&meter, "SetStream"))?;
        // Bitwise reproducible results run to run (cuSOLVER's default, made explicit; the
        // symbol exists from CUDA 12.2, and cudarc panics when it is missing).
        let det = std::panic::catch_unwind(|| unsafe {
            cs::cusolverDnSetDeterministicMode(
                h,
                cs::cusolverDeterministicMode_t::CUSOLVER_DETERMINISTIC_RESULTS,
            )
        });
        if let Ok(st) = det {
            st.result().map_err(err(&meter, "SetDeterministicMode"))?;
        }
        Ok(Solver { h, meter })
    }

    /// Eigenvalues (ascending) into `w` and eigenvectors of the symmetric n x n matrix `a`
    /// (column-major; only its lower triangle is read), which is overwritten with the
    /// eigenvectors as columns.
    pub fn syevd(
        &self,
        gpu: &Gpu,
        a: &mut DeviceBuf<f64>,
        n: usize,
        w: &mut DeviceBuf<f64>,
    ) -> Result<(), GpuError> {
        if a.len() < n * n || w.len() < n {
            return Err(GpuError::Contract("syevd: buffers too small".into()));
        }
        let ni = i32::try_from(n).map_err(|_| GpuError::Contract("syevd: n exceeds i32".into()))?;
        let (jobz, uplo) = (
            cs::cusolverEigMode_t::CUSOLVER_EIG_MODE_VECTOR,
            cs::cublasFillMode_t::CUBLAS_FILL_MODE_LOWER,
        );
        let (pa, pw) = (a.ptr().as_mut_ptr(), w.ptr().as_mut_ptr());
        let mut lwork = 0;
        // SAFETY: device pointers of buffers sized above.
        unsafe { cs::cusolverDnDsyevd_bufferSize(self.h, jobz, uplo, ni, pa, ni, pw, &mut lwork) }
            .result()
            .map_err(err(&self.meter, "syevd_bufferSize"))?;
        let work = gpu.alloc_zeros::<f64>(lwork.max(1) as usize)?;
        let info = gpu.alloc_zeros::<i32>(1)?;
        unsafe {
            cs::cusolverDnDsyevd(
                self.h,
                jobz,
                uplo,
                ni,
                pa,
                ni,
                pw,
                work.ptr().as_mut_ptr(),
                lwork,
                info.ptr().as_mut_ptr(),
            )
        }
        .result()
        .map_err(err(&self.meter, "syevd"))?;
        check_info("cuSOLVER syevd", gpu.download(&info)?[0])
    }
}

impl Solver {
    /// Cholesky factor A = L L^T of the symmetric positive definite n x n matrix `a`
    /// (column-major; the lower triangle is read and overwritten with L, the upper triangle
    /// is left as it was). Err(Numerical) names the failing leading minor when A is not
    /// positive definite.
    pub fn potrf_lower(&self, gpu: &Gpu, a: &mut DeviceBuf<f64>, n: usize) -> Result<(), GpuError> {
        if a.len() < n * n {
            return Err(GpuError::Contract("potrf: buffer too small".into()));
        }
        if n == 0 {
            return Ok(());
        }
        let ni = i32::try_from(n).map_err(|_| GpuError::Contract("potrf: n exceeds i32".into()))?;
        let uplo = cs::cublasFillMode_t::CUBLAS_FILL_MODE_LOWER;
        let pa = a.ptr().as_mut_ptr();
        let mut lwork = 0;
        // SAFETY: device pointer of a buffer of n x n elements (checked above).
        unsafe { cs::cusolverDnDpotrf_bufferSize(self.h, uplo, ni, pa, ni, &mut lwork) }
            .result()
            .map_err(err(&self.meter, "potrf_bufferSize"))?;
        let work = gpu.alloc_zeros::<f64>(lwork.max(1) as usize)?;
        let info = gpu.alloc_zeros::<i32>(1)?;
        unsafe {
            cs::cusolverDnDpotrf(
                self.h,
                uplo,
                ni,
                pa,
                ni,
                work.ptr().as_mut_ptr(),
                lwork,
                info.ptr().as_mut_ptr(),
            )
        }
        .result()
        .map_err(err(&self.meter, "potrf"))?;
        check_info("cuSOLVER potrf", gpu.download(&info)?[0])
    }
}

impl Solver {
    /// LU factorization with partial pivoting, P A = L U (unit L), of the n x n matrix `a`
    /// (column-major, overwritten with L and U), the pivots into `ipiv` (n, 1-based as LAPACK).
    /// Err(Numerical) when U is singular.
    pub fn getrf(
        &self,
        gpu: &Gpu,
        a: &mut DeviceBuf<f64>,
        n: usize,
        ipiv: &mut DeviceBuf<i32>,
    ) -> Result<(), GpuError> {
        if a.len() < n * n || ipiv.len() < n {
            return Err(GpuError::Contract("getrf: buffers too small".into()));
        }
        if n == 0 {
            return Ok(());
        }
        let ni = i32::try_from(n).map_err(|_| GpuError::Contract("getrf: n exceeds i32".into()))?;
        let pa = a.ptr().as_mut_ptr();
        let mut lwork = 0;
        // SAFETY: device pointer of a buffer of n x n elements (checked above).
        unsafe { cs::cusolverDnDgetrf_bufferSize(self.h, ni, ni, pa, ni, &mut lwork) }
            .result()
            .map_err(err(&self.meter, "getrf_bufferSize"))?;
        let work = gpu.alloc_zeros::<f64>(lwork.max(1) as usize)?;
        let info = gpu.alloc_zeros::<i32>(1)?;
        // SAFETY: a (n x n), the workspace of the queried size, n pivots, one info word.
        unsafe {
            cs::cusolverDnDgetrf(
                self.h,
                ni,
                ni,
                pa,
                ni,
                work.ptr().as_mut_ptr(),
                ipiv.ptr().as_mut_ptr(),
                info.ptr().as_mut_ptr(),
            )
        }
        .result()
        .map_err(err(&self.meter, "getrf"))?;
        check_info("cuSOLVER getrf", gpu.download(&info)?[0])
    }

    /// Solves A X = B (`trans` false) or A^T X = B (`trans` true) with the factors of
    /// [`Solver::getrf`]; `b` (n x nrhs, column-major) is overwritten with X. Queued without
    /// synchronizing: `info` receives cuSOLVER's argument check, which the caller may read.
    #[allow(clippy::too_many_arguments)]
    pub fn getrs(
        &self,
        a: &DeviceBuf<f64>,
        n: usize,
        ipiv: &DeviceBuf<i32>,
        trans: bool,
        b: &mut DeviceBuf<f64>,
        nrhs: usize,
        info: &mut DeviceBuf<i32>,
    ) -> Result<(), GpuError> {
        if a.len() < n * n || ipiv.len() < n || b.len() < n * nrhs || info.is_empty() {
            return Err(GpuError::Contract("getrs: buffers too small".into()));
        }
        if n == 0 || nrhs == 0 {
            return Ok(());
        }
        let ni = i32::try_from(n).map_err(|_| GpuError::Contract("getrs: n exceeds i32".into()))?;
        let nr = i32::try_from(nrhs)
            .map_err(|_| GpuError::Contract("getrs: nrhs exceeds i32".into()))?;
        let op = if trans {
            cs::cublasOperation_t::CUBLAS_OP_T
        } else {
            cs::cublasOperation_t::CUBLAS_OP_N
        };
        // SAFETY: a (n x n), n pivots, b (n x nrhs), one info word (checked above).
        unsafe {
            cs::cusolverDnDgetrs(
                self.h,
                op,
                ni,
                nr,
                a.ptr().as_ptr(),
                ni,
                ipiv.ptr().as_ptr(),
                b.ptr().as_mut_ptr(),
                ni,
                info.ptr().as_mut_ptr(),
            )
        }
        .result()
        .map_err(err(&self.meter, "getrs"))
    }

    /// Solves A X = B with the lower Cholesky factor of [`Solver::potrf_lower`]; `b` (n x
    /// nrhs, column-major) is overwritten with X. Queued as [`Solver::getrs`].
    pub fn potrs_lower(
        &self,
        a: &DeviceBuf<f64>,
        n: usize,
        b: &mut DeviceBuf<f64>,
        nrhs: usize,
        info: &mut DeviceBuf<i32>,
    ) -> Result<(), GpuError> {
        if a.len() < n * n || b.len() < n * nrhs || info.is_empty() {
            return Err(GpuError::Contract("potrs: buffers too small".into()));
        }
        if n == 0 || nrhs == 0 {
            return Ok(());
        }
        let ni = i32::try_from(n).map_err(|_| GpuError::Contract("potrs: n exceeds i32".into()))?;
        let nr = i32::try_from(nrhs)
            .map_err(|_| GpuError::Contract("potrs: nrhs exceeds i32".into()))?;
        // SAFETY: a (n x n), b (n x nrhs), one info word (checked above).
        unsafe {
            cs::cusolverDnDpotrs(
                self.h,
                cs::cublasFillMode_t::CUBLAS_FILL_MODE_LOWER,
                ni,
                nr,
                a.ptr().as_ptr(),
                ni,
                b.ptr().as_mut_ptr(),
                ni,
                info.ptr().as_mut_ptr(),
            )
        }
        .result()
        .map_err(err(&self.meter, "potrs"))
    }
}

impl Drop for Solver {
    fn drop(&mut self) {
        // SAFETY: created in new, destroyed once.
        let _ = unsafe { cs::cusolverDnDestroy(self.h) };
    }
}

/// The algorithm of an [`EighPlan`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EighMethod {
    /// Divide and conquer, `syevd` (the 32-bit legacy API).
    Syevd,
    /// Divide and conquer through the 64-bit generic API, `Xsyevd` (with the host workspace it
    /// asks for).
    Xsyevd,
    /// Cyclic Jacobi, `syevj`, to machine precision (at most 100 sweeps).
    Syevj,
}

/// Repeated symmetric eigendecompositions of one order (the SCF's): the workspace is allocated
/// once, and the calls only queue work (a method with a host workspace may wait for its host
/// part). Each call writes its status word into a slot of [`EighPlan::info`] (0: success),
/// which the caller reads when it next synchronizes.
pub struct EighPlan {
    n: usize,
    method: EighMethod,
    /// device workspace (whole doubles) and its size in bytes; host workspace
    work: DeviceBuf<f64>,
    work_bytes: usize,
    host: Vec<u8>,
    info: DeviceBuf<i32>,
    params: cs::cusolverDnParams_t,
    syevj: cs::syevjInfo_t,
}

// SAFETY: the parameter objects are plain cuSOLVER handles, used through &mut self.
unsafe impl Send for EighPlan {}

impl EighPlan {
    pub fn n(&self) -> usize {
        self.n
    }

    pub fn method(&self) -> EighMethod {
        self.method
    }

    /// The status words (one per slot) of the last calls.
    pub fn info(&self) -> &DeviceBuf<i32> {
        &self.info
    }

    /// Device bytes of the workspace.
    pub fn bytes(&self) -> usize {
        8 * self.work.len() + 4 * self.info.len()
    }
}

impl Drop for EighPlan {
    fn drop(&mut self) {
        // SAFETY: created by eigh_plan (or null), destroyed once.
        unsafe {
            if !self.params.is_null() {
                let _ = cs::cusolverDnDestroyParams(self.params);
            }
            if !self.syevj.is_null() {
                let _ = cs::cusolverDnDestroySyevjInfo(self.syevj);
            }
        }
    }
}

const JOBZ: cs::cusolverEigMode_t = cs::cusolverEigMode_t::CUSOLVER_EIG_MODE_VECTOR;
const LOWER: cs::cublasFillMode_t = cs::cublasFillMode_t::CUBLAS_FILL_MODE_LOWER;
const R64: cs::cudaDataType = cs::cudaDataType::CUDA_R_64F;

impl Solver {
    /// A plan for n x n eigendecompositions by `method` with `slots` status words.
    pub fn eigh_plan(
        &self,
        gpu: &Gpu,
        n: usize,
        method: EighMethod,
        slots: usize,
    ) -> Result<EighPlan, GpuError> {
        let ni = i32::try_from(n).map_err(|_| GpuError::Contract("eigh: n exceeds i32".into()))?;
        let mut plan = EighPlan {
            n,
            method,
            work: gpu.alloc_zeros::<f64>(1)?,
            work_bytes: 8,
            host: Vec::new(),
            info: gpu.alloc_zeros::<i32>(slots.max(1))?,
            params: std::ptr::null_mut(),
            syevj: std::ptr::null_mut(),
        };
        // SAFETY (the size queries): they read only the dimensions (null matrix pointers).
        match method {
            EighMethod::Syevd => {
                let mut lwork = 0;
                unsafe {
                    cs::cusolverDnDsyevd_bufferSize(
                        self.h,
                        JOBZ,
                        LOWER,
                        ni,
                        std::ptr::null(),
                        ni.max(1),
                        std::ptr::null(),
                        &mut lwork,
                    )
                }
                .result()
                .map_err(err(&self.meter, "syevd_bufferSize"))?;
                plan.work = gpu.alloc_zeros::<f64>(lwork.max(1) as usize)?;
                plan.work_bytes = 8 * lwork.max(1) as usize;
            }
            EighMethod::Xsyevd => {
                unsafe { cs::cusolverDnCreateParams(&mut plan.params) }
                    .result()
                    .map_err(err(&self.meter, "CreateParams"))?;
                let (mut dev, mut host) = (0usize, 0usize);
                unsafe {
                    cs::cusolverDnXsyevd_bufferSize(
                        self.h,
                        plan.params,
                        JOBZ,
                        LOWER,
                        n as i64,
                        R64,
                        std::ptr::null(),
                        n.max(1) as i64,
                        R64,
                        std::ptr::null(),
                        R64,
                        &mut dev,
                        &mut host,
                    )
                }
                .result()
                .map_err(err(&self.meter, "Xsyevd_bufferSize"))?;
                plan.work = gpu.alloc_zeros::<f64>(dev.div_ceil(8).max(1))?;
                plan.work_bytes = dev;
                plan.host = vec![0u8; host];
            }
            EighMethod::Syevj => {
                unsafe { cs::cusolverDnCreateSyevjInfo(&mut plan.syevj) }
                    .result()
                    .map_err(err(&self.meter, "CreateSyevjInfo"))?;
                unsafe { cs::cusolverDnXsyevjSetSortEig(plan.syevj, 1) }
                    .result()
                    .map_err(err(&self.meter, "XsyevjSetSortEig"))?;
                let mut lwork = 0;
                unsafe {
                    cs::cusolverDnDsyevj_bufferSize(
                        self.h,
                        JOBZ,
                        LOWER,
                        ni,
                        std::ptr::null(),
                        ni.max(1),
                        std::ptr::null(),
                        &mut lwork,
                        plan.syevj,
                    )
                }
                .result()
                .map_err(err(&self.meter, "syevj_bufferSize"))?;
                plan.work = gpu.alloc_zeros::<f64>(lwork.max(1) as usize)?;
                plan.work_bytes = 8 * lwork.max(1) as usize;
            }
        }
        Ok(plan)
    }

    /// Queues the eigendecomposition of the symmetric n x n matrix `a` (column-major; the lower
    /// triangle is read) with `plan`: eigenvalues (ascending) into w\[..n\], eigenvectors over
    /// `a` as columns, the status into `plan.info()[slot]`.
    pub fn eigh_queued(
        &self,
        plan: &mut EighPlan,
        a: &mut DeviceBuf<f64>,
        w: &mut DeviceBuf<f64>,
        slot: usize,
    ) -> Result<(), GpuError> {
        let n = plan.n;
        if a.len() < n * n || w.len() < n || slot >= plan.info.len() {
            return Err(GpuError::Contract(
                "eigh: buffers too small or no such status slot".into(),
            ));
        }
        if n == 0 {
            return Ok(());
        }
        let ni = n as i32;
        let (pa, pw) = (a.ptr().as_mut_ptr(), w.ptr().as_mut_ptr());
        let pinfo = plan.info.at(slot).as_mut_ptr();
        let pwork = plan.work.ptr().as_mut_ptr();
        let host = if plan.host.is_empty() {
            std::ptr::null_mut()
        } else {
            plan.host.as_mut_ptr() as *mut std::ffi::c_void
        };
        // SAFETY: device pointers of buffers sized above; the workspaces have the queried
        // sizes; the status slot lies in the info buffer; the host workspace lives in the plan.
        unsafe {
            match plan.method {
                EighMethod::Syevd => cs::cusolverDnDsyevd(
                    self.h,
                    JOBZ,
                    LOWER,
                    ni,
                    pa,
                    ni,
                    pw,
                    pwork,
                    (plan.work_bytes / 8) as i32,
                    pinfo,
                ),
                EighMethod::Xsyevd => cs::cusolverDnXsyevd(
                    self.h,
                    plan.params,
                    JOBZ,
                    LOWER,
                    n as i64,
                    R64,
                    pa as *mut std::ffi::c_void,
                    n as i64,
                    R64,
                    pw as *mut std::ffi::c_void,
                    R64,
                    pwork as *mut std::ffi::c_void,
                    plan.work_bytes,
                    host,
                    plan.host.len(),
                    pinfo,
                ),
                EighMethod::Syevj => cs::cusolverDnDsyevj(
                    self.h,
                    JOBZ,
                    LOWER,
                    ni,
                    pa,
                    ni,
                    pw,
                    pwork,
                    (plan.work_bytes / 8) as i32,
                    pinfo,
                    plan.syevj,
                ),
            }
        }
        .result()
        .map_err(err(&self.meter, "eigh"))
    }
}
