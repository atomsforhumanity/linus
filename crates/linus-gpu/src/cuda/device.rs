//! The device: context, stream, properties; memory, copies, run-time compilation, launch and
//! timing for everything above the backend.

use super::host::HostBuf;
use super::nvrtc::KernelCache;
use super::pool::{DevPtr, DeviceBuf, MemPool, PoolUsage};
use super::stream::{CopyStream, Fence};
use crate::error::GpuError;
use cudarc::driver::sys::CUdevice_attribute as Attr;
use cudarc::driver::{CudaContext, CudaFunction, CudaStream, DeviceRepr, ValidAsZeroBits};
use std::ffi::c_void;
use std::path::PathBuf;
use std::sync::Arc;

/// The portability header pasted in front of every kernel source.
pub const PORTABLE_H: &str = include_str!("../../device/portable.h");

/// The full source the compiler sees for a device of warp size `warp_size`: the warp size, the
/// portability header, `src` ([`Gpu::kernel_source`]; no device needed, for compile tests).
pub fn kernel_source(warp_size: u32, src: &str) -> String {
    format!("#define LINUS_WARP_SIZE {warp_size}\n{PORTABLE_H}\n{src}")
}

/// Properties of a device.
#[derive(Clone, Debug)]
pub struct DeviceInfo {
    pub ordinal: usize,
    pub name: String,
    /// Compute capability (major, minor).
    pub cc: (i32, i32),
    pub sm_count: u32,
    pub warp_size: u32,
    pub total_mem: usize,
    /// Core clock (kHz).
    pub clock_khz: u32,
    /// Memory clock (kHz) and bus width (bits), as the driver reports them.
    pub mem_clock_khz: u32,
    pub mem_bus_bits: u32,
    pub l2_bytes: u32,
    pub max_regs_per_block: u32,
    pub max_smem_per_block_optin: u32,
    /// CUDA driver version (e.g. 13030).
    pub driver_version: i32,
}

impl DeviceInfo {
    /// `sm_XY`.
    pub fn arch(&self) -> String {
        format!("sm_{}{}", self.cc.0, self.cc.1)
    }

    /// The nominal ratio of the device class's dense INT8 tensor-core throughput to its FP64
    /// throughput (tensor-core FP64 where it has one), from NVIDIA's data sheets by compute
    /// capability; None for devices without INT8 tensor cores (compute capability below 7.2,
    /// e.g. P100, V100), on which FP64 emulation cannot run.
    ///
    /// | cc | example | FP64 TFLOP/s | INT8 TOPS | ratio |
    /// |---|---|---|---|---|
    /// | 8.0 | A100 | 19.5 | 624 | 32 |
    /// | 9.0 | H100/H200 | 67 | 1979 | 30 |
    /// | 10.0 | B200 | 40 | 4500 | 112 (extrapolated: no B200 measured) |
    /// | 7.5, 8.6, 8.9, 12.0 | T4, A40, L40S, RTX PRO 6000 | FP64 at 1/32-1/64 of FP32 | | 500-600 |
    /// | 10.3, 11.0, 12.1 | B300, Thor, GB10 | FP64 at a few TFLOP/s or less | | > 1000 |
    pub fn int8_per_fp64(&self) -> Option<f64> {
        match self.cc {
            (major, minor) if (major, minor) < (7, 2) => None,
            (8, 0) => Some(32.0),
            (9, 0) => Some(30.0),
            (10, 0) => Some(112.0),
            (10, _) | (11, _) | (12, 1..) => Some(1000.0),
            _ => Some(500.0),
        }
    }

    /// Whether native FP64 outruns FP64 emulation on INT8 tensor cores on this device class:
    /// INT8 throughput below 64x FP64, or no INT8 tensor cores. Measured on the DF exchange's
    /// GEMMs (docs/internal/gpu-datacenter.md): the Ozaki scheme with 7 slices (28 INT8
    /// products per FP64 product) ran at about 0.4 x ratio / 28 of native FP64's speed, 0.47x
    /// on an H200 (ratio 30) and 5.6-7.6x on an RTX PRO 6000 (ratio about 500), so it breaks
    /// even near a ratio of 70. B200 (112) is on the emulated side by that model, not by a
    /// measurement. The one classification of linus's card-aware choices (decision D27).
    pub fn fp64_rich(&self) -> bool {
        self.int8_per_fp64().is_none_or(|r| r < 64.0)
    }
}

/// A compiled kernel.
#[derive(Clone, Debug)]
pub struct Kernel {
    f: CudaFunction,
    name: String,
}

impl Kernel {
    /// The kernel `entry` of the loaded module `m`.
    fn load(m: &Arc<cudarc::driver::CudaModule>, entry: &str) -> Result<Kernel, GpuError> {
        Ok(Kernel {
            f: m.load_function(entry)?,
            name: entry.to_string(),
        })
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    /// Registers per thread.
    pub fn registers(&self) -> Option<i32> {
        self.f.num_regs().ok()
    }

    /// Local memory per thread in bytes (register spills and local arrays).
    pub fn local_bytes(&self) -> Option<i32> {
        self.f.local_size_bytes().ok()
    }

    /// Allows launches with up to `bytes` of dynamic shared memory (above the default 48 KiB,
    /// up to the device's opt-in limit, [`DeviceInfo::max_smem_per_block_optin`]).
    pub fn set_max_dynamic_shared(&self, bytes: u32) -> Result<(), GpuError> {
        use cudarc::driver::sys::CUfunction_attribute_enum as A;
        Ok(self.f.set_attribute(
            A::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
            bytes as i32,
        )?)
    }
}

/// Launch geometry.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Launch {
    pub grid: (u32, u32, u32),
    pub block: (u32, u32, u32),
    pub shared_bytes: u32,
}

impl Launch {
    /// `n` threads in blocks of `block` (one-dimensional).
    pub fn linear(n: usize, block: u32) -> Launch {
        Launch {
            grid: (n.div_ceil(block as usize).max(1) as u32, 1, 1),
            block: (block, 1, 1),
            shared_bytes: 0,
        }
    }
}

/// A kernel argument: a device address or a scalar.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Arg {
    /// A device address (from a [`DevPtr`]; 0 for a pointer the kernel does not read).
    Ptr(u64),
    I32(i32),
    U32(u32),
    I64(i64),
    F64(f64),
}

impl Arg {
    /// The device address of `b` (for a `T *` parameter).
    pub fn buf<T>(b: &DeviceBuf<T>) -> Arg {
        b.ptr().into()
    }

    /// As [`Arg::buf`] for a buffer the kernel writes (documents intent; same value).
    pub fn buf_mut<T>(b: &mut DeviceBuf<T>) -> Arg {
        b.ptr().into()
    }

    fn word(self) -> u64 {
        match self {
            Arg::Ptr(p) => p,
            Arg::I32(v) => v as u32 as u64,
            Arg::U32(v) => v as u64,
            Arg::I64(v) => v as u64,
            Arg::F64(v) => v.to_bits(),
        }
    }
}

impl<T> From<DevPtr<T>> for Arg {
    /// The address for a `T *` parameter.
    fn from(p: DevPtr<T>) -> Arg {
        Arg::Ptr(p.addr())
    }
}

/// An open device with one stream; the entry point to device memory, kernels and timing.
/// Work is issued in order on the stream; results are read back with a synchronizing copy.
pub struct Gpu {
    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    info: DeviceInfo,
    pool: MemPool,
    kernels: KernelCache,
    /// The cap on the memory limit the handle opened with (bytes), if any.
    memory_cap: Option<usize>,
}

impl Drop for Gpu {
    /// Returns what the device's pool keeps of freed buffers to the driver: the pool never
    /// releases on its own (its release threshold is unlimited), so without this a later
    /// handle in the same process (the next session of a workflow) would see the earlier
    /// one's peak as used and plan its engines against less memory (on an H200 a harness
    /// opened its device with 25 of 141 GB free after a range-separated ts61 SCF in the same
    /// process, and its exchange setup ran out of memory).
    fn drop(&mut self) {
        let _ = self.pool.trim(&self.stream);
    }
}

/// A point on the device's stream ([`Gpu::mark`]), for timing the work between two marks
/// without synchronizing in between.
pub struct Mark(cudarc::driver::CudaEvent);

impl Mark {
    /// Device seconds from `self` to `later` (waits for `later`).
    pub fn seconds_until(&self, later: &Mark) -> Result<f64, GpuError> {
        later.0.synchronize()?;
        Ok(self.0.elapsed_ms(&later.0)? as f64 * 1e-3)
    }
}

/// Error unless offset..offset + len lies within `n` elements.
pub(super) fn check_range(what: &str, offset: usize, len: usize, n: usize) -> Result<(), GpuError> {
    match offset.checked_add(len) {
        Some(end) if end <= n => Ok(()),
        _ => Err(GpuError::Contract(format!(
            "{what}: elements {offset}..{} outside a buffer of {n}",
            offset.saturating_add(len)
        ))),
    }
}

/// The device memory a handle may use: 95 % of what is free when it opens, at most `cap` bytes
/// when given (processes sharing a card: each engine plans its storage and opportunistic
/// copies, such as the exchange's pre-sliced planes, against the cap instead of against the
/// whole card).
fn memory_limit(free: usize, cap: Option<usize>) -> usize {
    let limit = free / 20 * 19;
    cap.map_or(limit, |c| limit.min(c))
}

fn attr(ctx: &CudaContext, a: Attr) -> u32 {
    ctx.attribute(a).map(|v| v.max(0) as u32).unwrap_or(0)
}

impl Gpu {
    /// Opens device `ordinal` with the default kernel cache directory
    /// ([`super::nvrtc::default_cache_dir`]) and a memory limit of 95 % of the memory free now.
    pub fn new(ordinal: usize) -> Result<Gpu, GpuError> {
        Self::open(ordinal, super::nvrtc::default_cache_dir(), None)
    }

    /// Opens device `ordinal` with the kernel cache in `dir` (None: memory only) and a memory
    /// limit of 95 % of the memory free now, at most `memory_cap` bytes when given (the caller's
    /// setting: linus's `device_memory_gib`).
    pub fn open(
        ordinal: usize,
        dir: Option<PathBuf>,
        memory_cap: Option<usize>,
    ) -> Result<Gpu, GpuError> {
        // cudarc panics when it cannot load libcuda; turn that into an error
        let ctx = std::panic::catch_unwind(|| CudaContext::new(ordinal))
            .map_err(|p| {
                let msg = p
                    .downcast_ref::<String>()
                    .cloned()
                    .or_else(|| p.downcast_ref::<&str>().map(|s| s.to_string()))
                    .unwrap_or_default();
                GpuError::Unavailable(format!("cannot load the CUDA driver: {msg}"))
            })?
            .map_err(|e| GpuError::Unavailable(format!("device {ordinal}: {e:?}")))?;
        // linus orders its streams itself: one compute stream, and copy streams that wait on
        // explicit fences (`CopyStream`). Kernels and libraries take raw addresses, which
        // cudarc's per-buffer read and write events never see, so the events (two per
        // allocation, waited on at every free once a second stream exists) protect nothing.
        // SAFETY: every cross-stream use is ordered by a `Fence` (see `CopyStream`), and a
        // buffer is freed on the compute stream after the fences of its copies were waited for.
        unsafe { ctx.disable_event_tracking() };
        let cc = ctx.compute_capability()?;
        let mut driver_version = 0;
        // SAFETY: out pointer.
        unsafe { cudarc::driver::sys::cuDriverGetVersion(&mut driver_version) }.result()?;
        let info = DeviceInfo {
            ordinal,
            name: ctx.name()?,
            cc,
            sm_count: attr(&ctx, Attr::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT),
            warp_size: attr(&ctx, Attr::CU_DEVICE_ATTRIBUTE_WARP_SIZE),
            total_mem: ctx.total_mem()?,
            clock_khz: attr(&ctx, Attr::CU_DEVICE_ATTRIBUTE_CLOCK_RATE),
            mem_clock_khz: attr(&ctx, Attr::CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE),
            mem_bus_bits: attr(&ctx, Attr::CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH),
            l2_bytes: attr(&ctx, Attr::CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE),
            max_regs_per_block: attr(&ctx, Attr::CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK),
            max_smem_per_block_optin: attr(
                &ctx,
                Attr::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,
            ),
            driver_version,
        };
        let stream = ctx.default_stream();
        let (free, _) = ctx.mem_get_info()?;
        let limit = memory_limit(free, memory_cap);
        let pool = MemPool::new(&ctx, limit)?;
        let kernels = KernelCache::new(ctx.clone(), cc, dir)?;
        Ok(Gpu {
            ctx,
            stream,
            info,
            pool,
            kernels,
            memory_cap,
        })
    }

    /// The cap on the memory limit the handle opened with (bytes), if any; the limit itself is
    /// [`Gpu::memory`]'s.
    pub fn memory_cap(&self) -> Option<usize> {
        self.memory_cap
    }

    pub fn info(&self) -> &DeviceInfo {
        &self.info
    }

    pub fn warp_size(&self) -> u32 {
        self.info.warp_size
    }

    pub(crate) fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }

    /// The kernel cache's counts so far (compiled, memory and disk hits), for reports.
    pub fn cache_stats(&self) -> super::nvrtc::CacheStats {
        self.kernels.stats()
    }

    /// The kernel cache itself (feature `testkit`: tests and benchmarks of the cache and the
    /// compiler).
    #[cfg(feature = "testkit")]
    pub fn kernel_cache(&self) -> &KernelCache {
        &self.kernels
    }

    pub fn memory(&self) -> PoolUsage {
        self.pool.usage()
    }

    /// Starts a new measure of the peak device memory ([`Gpu::peak`]): the high-water mark
    /// becomes what the pool holds now. A caller measuring one calculation calls it at the
    /// calculation's start.
    pub fn reset_peak(&self) {
        self.pool.reset_peak();
    }

    /// The most device memory the pool held at once since the handle opened or the last
    /// [`Gpu::reset_peak`] (bytes): the pool's accounting, live buffers and grants (`Gpu::grant`),
    /// not the driver's (the context, the libraries' own workspaces).
    pub fn peak(&self) -> usize {
        self.pool.usage().peak
    }

    /// The device memory the handle's engines may still allocate: the handle's limit and what
    /// its pool holds now. The one way code learns what is free (`budget().free()`) or checks a
    /// request before allocating (`budget().check(what, bytes)?`, an out-of-memory error).
    pub fn budget(&self) -> linus_core::Budget {
        let u = self.pool.usage();
        linus_core::Budget::new(u.limit, u.in_use)
    }

    /// `bytes` of device memory set aside for the engine `what` ([`crate::pool::Grant`]): counted
    /// as in use from now on, drawn by the engine's planned allocations, the rest released on
    /// drop. An out-of-memory error when they do not fit.
    pub fn grant(&self, bytes: usize, what: &'static str) -> Result<crate::pool::Grant, GpuError> {
        self.pool.grant(bytes, what)
    }

    pub(crate) fn pool(&self) -> &MemPool {
        &self.pool
    }

    // ---- memory and copies -----------------------------------------------------------

    /// `n` zeroed elements.
    pub fn alloc_zeros<T: DeviceRepr + ValidAsZeroBits>(
        &self,
        n: usize,
    ) -> Result<DeviceBuf<T>, GpuError> {
        self.pool.alloc_zeros(&self.stream, n)
    }

    /// [`Self::alloc_zeros`] of an allocation the memory plan granted: drawn from `grant` (when
    /// the engine holds one) just before it is made, its bytes back in the grant when the
    /// buffer is freed ([`crate::pool::Grant`]).
    pub fn alloc_zeros_granted<T: DeviceRepr + ValidAsZeroBits>(
        &self,
        grant: Option<&crate::pool::Grant>,
        n: usize,
    ) -> Result<DeviceBuf<T>, GpuError> {
        self.pool.alloc_zeros_granted(&self.stream, grant, n)
    }

    /// A device copy of `host`.
    pub fn upload<T: DeviceRepr>(&self, host: &[T]) -> Result<DeviceBuf<T>, GpuError> {
        self.pool.upload(&self.stream, host)
    }

    /// [`Self::upload`] of an allocation the memory plan granted, as
    /// [`Self::alloc_zeros_granted`].
    pub fn upload_granted<T: DeviceRepr>(
        &self,
        grant: Option<&crate::pool::Grant>,
        host: &[T],
    ) -> Result<DeviceBuf<T>, GpuError> {
        self.pool.upload_granted(&self.stream, grant, host)
    }

    /// dst[offset..offset + host.len()] = host.
    pub fn upload_into<T: DeviceRepr>(
        &self,
        dst: &mut DeviceBuf<T>,
        offset: usize,
        host: &[T],
    ) -> Result<(), GpuError> {
        check_range("upload_into", offset, host.len(), dst.len())?;
        let mut view = dst.raw_mut().slice_mut(offset..offset + host.len());
        Ok(self.stream.memcpy_htod(host, &mut view)?)
    }

    /// A host copy of `buf` (synchronizes the stream).
    pub fn download<T: DeviceRepr>(&self, buf: &DeviceBuf<T>) -> Result<Vec<T>, GpuError> {
        Ok(self.stream.clone_dtoh(buf.raw())?)
    }

    /// dst[..src.len()] = src on the device.
    pub fn copy<T: DeviceRepr>(
        &self,
        src: &DeviceBuf<T>,
        dst: &mut DeviceBuf<T>,
    ) -> Result<(), GpuError> {
        if src.len() > dst.len() {
            return Err(GpuError::Contract("copy: destination too small".into()));
        }
        let mut view = dst.raw_mut().slice_mut(0..src.len());
        Ok(self.stream.memcpy_dtod(src.raw(), &mut view)?)
    }

    /// Sets every byte of `buf` to zero.
    pub fn zero<T: DeviceRepr + ValidAsZeroBits>(
        &self,
        buf: &mut DeviceBuf<T>,
    ) -> Result<(), GpuError> {
        Ok(self.stream.memset_zeros(buf.raw_mut())?)
    }

    /// Sets buf[offset..offset + len] to zero.
    pub fn zero_range<T: DeviceRepr + ValidAsZeroBits>(
        &self,
        buf: &mut DeviceBuf<T>,
        offset: usize,
        len: usize,
    ) -> Result<(), GpuError> {
        check_range("zero_range", offset, len, buf.len())?;
        if len == 0 {
            return Ok(());
        }
        let mut view = buf.raw_mut().slice_mut(offset..offset + len);
        Ok(self.stream.memset_zeros(&mut view)?)
    }

    /// A host copy of buf[offset..offset + len] (synchronizes the stream).
    pub fn download_range<T: DeviceRepr + ValidAsZeroBits + Clone + Default>(
        &self,
        buf: &DeviceBuf<T>,
        offset: usize,
        len: usize,
    ) -> Result<Vec<T>, GpuError> {
        check_range("download_range", offset, len, buf.len())?;
        let mut out = vec![T::default(); len];
        if len > 0 {
            let view = buf.raw().slice(offset..offset + len);
            self.stream.memcpy_dtoh(&view, &mut out)?;
            self.stream.synchronize()?;
        }
        Ok(out)
    }

    /// dst[dst_off..dst_off + len] = src[src_off..src_off + len] on the device.
    pub fn copy_range<T: DeviceRepr>(
        &self,
        src: &DeviceBuf<T>,
        src_off: usize,
        dst: &mut DeviceBuf<T>,
        dst_off: usize,
        len: usize,
    ) -> Result<(), GpuError> {
        check_range("copy_range (source)", src_off, len, src.len())?;
        check_range("copy_range (destination)", dst_off, len, dst.len())?;
        if len == 0 {
            return Ok(());
        }
        let view = src.raw().slice(src_off..src_off + len);
        let mut out = dst.raw_mut().slice_mut(dst_off..dst_off + len);
        Ok(self.stream.memcpy_dtod(&view, &mut out)?)
    }

    /// `len` zeroed elements of page-locked host memory, for copies that overlap device work.
    pub fn alloc_host<T: DeviceRepr + ValidAsZeroBits>(
        &self,
        len: usize,
    ) -> Result<HostBuf<T>, GpuError> {
        HostBuf::new(&self.stream, len)
    }

    /// Queues dst[dst_off..dst_off + len] = host[host_off..host_off + len] on the stream
    /// (asynchronous; the host buffer is not touched by the host until the stream is waited
    /// for, see [`HostBuf`]).
    pub fn upload_from_host<T: DeviceRepr>(
        &self,
        host: &HostBuf<T>,
        host_off: usize,
        dst: &mut DeviceBuf<T>,
        dst_off: usize,
        len: usize,
    ) -> Result<(), GpuError> {
        check_range("upload_from_host (host)", host_off, len, host.len())?;
        check_range("upload_from_host (device)", dst_off, len, dst.len())?;
        if len == 0 {
            return Ok(());
        }
        let sz = std::mem::size_of::<T>();
        self.ctx.bind_to_thread()?;
        // SAFETY: both ranges checked above; the pinned host buffer outlives the copy (its
        // drop waits for the stream), and the device buffer is freed in stream order.
        unsafe {
            cudarc::driver::sys::cuMemcpyHtoDAsync_v2(
                dst.at(dst_off).addr(),
                host.ptr().add(host_off) as *const c_void,
                len * sz,
                self.stream.cu_stream(),
            )
        }
        .result()?;
        Ok(())
    }

    /// Queues the copy of a column-major block (`rows` x `cols` elements, leading dimensions
    /// `src_ld` on the device and `host_ld` on the host) from src[src_off..] to
    /// host[host_off..] on the stream (asynchronous, see [`Gpu::upload_from_host`]).
    #[allow(clippy::too_many_arguments)]
    pub fn download_block_to_host<T: DeviceRepr>(
        &self,
        src: &DeviceBuf<T>,
        src_off: usize,
        src_ld: usize,
        host: &mut HostBuf<T>,
        host_off: usize,
        host_ld: usize,
        rows: usize,
        cols: usize,
    ) -> Result<(), GpuError> {
        if rows == 0 || cols == 0 {
            return Ok(());
        }
        if rows > src_ld || rows > host_ld {
            return Err(GpuError::Contract(
                "download_block_to_host: rows exceed a leading dimension".into(),
            ));
        }
        // the elements from the block's first to its last (a span beyond the address space is
        // outside any buffer)
        let span = |ld: usize| {
            (cols - 1)
                .checked_mul(ld)
                .and_then(|s| s.checked_add(rows))
                .unwrap_or(usize::MAX)
        };
        check_range(
            "download_block_to_host (device)",
            src_off,
            span(src_ld),
            src.len(),
        )?;
        check_range(
            "download_block_to_host (host)",
            host_off,
            span(host_ld),
            host.len(),
        )?;
        let sz = std::mem::size_of::<T>();
        use cudarc::driver::sys::{CUmemorytype, CUDA_MEMCPY2D};
        let p = CUDA_MEMCPY2D {
            srcXInBytes: 0,
            srcY: 0,
            srcMemoryType: CUmemorytype::CU_MEMORYTYPE_DEVICE,
            srcHost: std::ptr::null(),
            srcDevice: src.at(src_off).addr(),
            srcArray: std::ptr::null_mut(),
            srcPitch: src_ld * sz,
            dstXInBytes: 0,
            dstY: 0,
            dstMemoryType: CUmemorytype::CU_MEMORYTYPE_HOST,
            // SAFETY (pointer arithmetic): host_off is inside the buffer (checked above).
            dstHost: unsafe { host.ptr().add(host_off) } as *mut c_void,
            dstDevice: 0,
            dstArray: std::ptr::null_mut(),
            dstPitch: host_ld * sz,
            WidthInBytes: rows * sz,
            Height: cols,
        };
        self.ctx.bind_to_thread()?;
        // SAFETY: both spans checked above; the host buffer outlives the copy (see
        // upload_from_host).
        unsafe { cudarc::driver::sys::cuMemcpy2DAsync_v2(&p, self.stream.cu_stream()) }.result()?;
        Ok(())
    }

    // ---- kernels -------------------------------------------------------------------------

    /// The full source the compiler sees: the warp size, the portability header, `src`.
    pub fn kernel_source(&self, src: &str) -> String {
        kernel_source(self.info.warp_size, src)
    }

    /// Kernel `entry` of `src` (compiled once per source and options, see [`super::nvrtc`]).
    pub fn kernel(&self, name: &str, src: &str, entry: &str) -> Result<Kernel, GpuError> {
        self.kernel_with(name, src, entry, &[])
    }

    /// As [`Gpu::kernel_with`] compiled for the device's architecture-specific target (`sm_XYa`,
    /// [`crate::nvrtc::KernelCache::arch_specific`]): for kernels that use features of that
    /// architecture alone (sm_90a's register hand-over). An error where the cache compiles PTX
    /// for an older architecture.
    pub fn kernel_arch_specific(
        &self,
        name: &str,
        src: &str,
        entry: &str,
        extra: &[&str],
    ) -> Result<Kernel, GpuError> {
        let target = self.kernels.arch_specific()?;
        let m = self
            .kernels
            .module_for(name, &self.kernel_source(src), extra, &target)?;
        Kernel::load(&m, entry)
    }

    /// The kernels `entries` of one module compiled from `src` (one compilation, one cache
    /// entry, one load for all of them; [`Gpu::kernel`] per entry would hash and look up the
    /// source each time).
    pub fn kernels(
        &self,
        name: &str,
        src: &str,
        entries: &[&str],
    ) -> Result<Vec<Kernel>, GpuError> {
        let m = self.kernels.module(name, &self.kernel_source(src), &[])?;
        entries.iter().map(|e| Kernel::load(&m, e)).collect()
    }

    /// As [`Gpu::kernel`] with extra compiler options.
    pub fn kernel_with(
        &self,
        name: &str,
        src: &str,
        entry: &str,
        extra: &[&str],
    ) -> Result<Kernel, GpuError> {
        let m = self.kernels.module(name, &self.kernel_source(src), extra)?;
        Kernel::load(&m, entry)
    }

    /// Queues `k` on the stream.
    ///
    /// # Safety
    /// `args` must match the kernel's parameter list in number and type (a device address
    /// for each pointer, `I32`/`U32` for 32-bit and `I64`/`F64` for 64-bit scalars), and every
    /// buffer must hold all the elements the kernel reads or writes for this geometry.
    pub unsafe fn launch(&self, k: &Kernel, cfg: Launch, args: &[Arg]) -> Result<(), GpuError> {
        self.ctx.bind_to_thread()?;
        // each parameter points at a little-endian 64-bit word; the driver reads its size
        let mut words: Vec<u64> = args.iter().map(|a| a.word()).collect();
        let mut params: Vec<*mut c_void> = words
            .iter_mut()
            .map(|w| w as *mut u64 as *mut c_void)
            .collect();
        unsafe {
            cudarc::driver::result::launch_kernel(
                k.f.cu_function(),
                cfg.grid,
                cfg.block,
                cfg.shared_bytes,
                self.stream.cu_stream(),
                &mut params,
            )
        }?;
        Ok(())
    }

    // ---- synchronization and timing ------------------------------------------------------

    pub fn synchronize(&self) -> Result<(), GpuError> {
        Ok(self.stream.synchronize()?)
    }

    /// A timing mark recorded on the stream now (after the work queued so far).
    pub fn mark(&self) -> Result<Mark, GpuError> {
        let e = self
            .ctx
            .new_event(Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT))?;
        e.record(&self.stream)?;
        Ok(Mark(e))
    }

    // ---- a second stream for overlapped copies -------------------------------------------

    /// A second (non-blocking) stream of the device's context for copies between pinned host
    /// memory and device memory that overlap the work on the device's stream. The two streams
    /// are ordered only by [`Fence`]s: before a copy reads or writes a device buffer that the
    /// device's stream uses, the copy stream waits for a fence of the device's stream
    /// ([`CopyStream::wait_for`] with [`Gpu::fence`]), and the device's stream waits for the
    /// copy's fence ([`Gpu::wait_for`] with [`CopyStream::fence`]) before it uses the result.
    pub(crate) fn copy_stream(&self) -> Result<CopyStream, GpuError> {
        Ok(CopyStream {
            stream: self.ctx.new_stream()?,
        })
    }

    /// A fence at the end of the work queued on the device's stream so far.
    pub fn fence(&self) -> Result<Fence, GpuError> {
        Ok(Fence(self.stream.record_event(None)?))
    }

    /// The device's stream waits (on the device) for the work before `f`.
    pub fn wait_for(&self, f: &Fence) -> Result<(), GpuError> {
        Ok(self.stream.wait(&f.0)?)
    }
}
