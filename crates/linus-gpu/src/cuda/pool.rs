//! Device memory: stream-ordered allocation from the driver's memory pool, with accounting
//! and a limit.
//!
//! The pool is the device's default stream-ordered pool (`cuMemAllocAsync`), told to keep
//! freed memory (release threshold = unlimited), so a freed block is reused by the next
//! allocation on the stream without going back to the driver: temporaries are cheap, and the
//! driver handles fragmentation and stream ordering. On top of it [`MemPool`] counts the bytes
//! of live [`DeviceBuf`]s against a limit, so a calculation that would not fit fails with a
//! clear [`GpuError::OutOfMemory`] before the driver runs out.
//!
//! Kernels and library calls address a buffer's elements through [`DevPtr`], a device address
//! typed by its element: [`DeviceBuf::at`] gives the address of an element, and the element
//! size comes from the type, never from a literal.

use crate::error::GpuError;
use cudarc::driver::{
    sys, CudaContext, CudaSlice, CudaStream, DevicePtr, DeviceRepr, ValidAsZeroBits,
};
use std::marker::PhantomData;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

/// The device address of an element of type `T` ([`DeviceBuf::at`]): what a kernel's `T *`
/// parameter or a library's device pointer receives. Offsets move in elements of `T`, so no
/// call site spells an element size.
pub struct DevPtr<T> {
    addr: u64,
    _elem: PhantomData<*const T>,
}

impl<T> DevPtr<T> {
    /// The null address (a pointer parameter the kernel does not read in this launch).
    pub const fn null() -> Self {
        DevPtr {
            addr: 0,
            _elem: PhantomData,
        }
    }

    /// The typed address of a raw device address (from a pointer array or an address
    /// computed elsewhere).
    ///
    /// # Safety
    /// `addr` must be the address of an element of type `T` (or 0) wherever the result is
    /// read or written through.
    pub unsafe fn from_addr(addr: u64) -> Self {
        DevPtr {
            addr,
            _elem: PhantomData,
        }
    }

    /// The raw device address (the word a kernel argument or a device pointer array holds).
    pub fn addr(self) -> u64 {
        self.addr
    }

    /// As a pointer for a library call that only reads through it.
    pub(crate) fn as_ptr(self) -> *const T {
        self.addr as *const T
    }

    /// As a pointer for a library call that writes through it.
    pub(crate) fn as_mut_ptr(self) -> *mut T {
        self.addr as *mut T
    }
}

impl<T> std::ops::Add<usize> for DevPtr<T> {
    type Output = DevPtr<T>;

    /// The address `n` elements further on.
    fn add(self, n: usize) -> DevPtr<T> {
        DevPtr {
            addr: self.addr + (n * std::mem::size_of::<T>()) as u64,
            _elem: PhantomData,
        }
    }
}

// by hand: the derives would require `T: Copy` (and `T: Debug`, `T: PartialEq`)
impl<T> Clone for DevPtr<T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> Copy for DevPtr<T> {}

impl<T> std::fmt::Debug for DevPtr<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "DevPtr({:#x})", self.addr)
    }
}

impl<T> PartialEq for DevPtr<T> {
    fn eq(&self, other: &Self) -> bool {
        self.addr == other.addr
    }
}

impl<T> Eq for DevPtr<T> {}

#[derive(Debug, Default)]
struct Counters {
    in_use: AtomicUsize,
    peak: AtomicUsize,
    limit: AtomicUsize,
}

impl Counters {
    fn usage(&self) -> PoolUsage {
        PoolUsage {
            in_use: self.in_use.load(Ordering::Relaxed),
            peak: self.peak.load(Ordering::Relaxed),
            limit: self.limit.load(Ordering::Relaxed),
        }
    }
}

/// A read-only view of a pool's accounting, for the library wrappers that classify a library's
/// failure by linus's own memory budget ([`super::memory_pressure`]) without the `Gpu` handle.
#[derive(Clone, Debug)]
pub(crate) struct PoolMeter(Arc<Counters>);

impl PoolMeter {
    /// The pool's limit and its use now.
    pub(crate) fn budget(&self) -> linus_core::Budget {
        let u = self.0.usage();
        linus_core::Budget::new(u.limit, u.in_use)
    }
}

/// Device bytes set aside for one engine before it allocates them ([`super::Gpu::grant`]; the
/// memory plan's grant, package PLAN): the pool counts them as in use from the grant on, so
/// nothing allocated elsewhere afterwards can take them. The engine draws from the grant just
/// before each allocation it planned: [`Grant::draw`] for an allocation it keeps (the bytes go
/// back to the pool, which the allocation then takes as usual), or a granted allocation
/// ([`super::Gpu::alloc_zeros_granted`]) whose bytes return to the grant when it is freed, so
/// a workspace reused by the engine's next phase or call draws them again. A draw beyond what
/// remains (the engine holding more at once than the plan gave it) is a plan violation,
/// recorded ([`Grant::violations`]) and not refused: the allocation then competes for the free
/// memory like any other. What remains returns to the pool when the grant drops.
#[derive(Debug)]
pub struct Grant {
    state: Arc<GrantState>,
}

/// A grant's accounting, shared with its granted allocations.
#[derive(Debug)]
pub(crate) struct GrantState {
    counters: Arc<Counters>,
    what: &'static str,
    granted: usize,
    inner: std::sync::Mutex<GrantInner>,
}

#[derive(Debug, Default)]
struct GrantInner {
    remaining: usize,
    /// The grant dropped: a granted allocation freed later returns its bytes to the pool.
    closed: bool,
    violations: Vec<PlanViolation>,
}

/// A draw beyond a [`Grant`]: what it was for, the bytes granted, the bytes asked for and the
/// bytes the grant still held.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PlanViolation {
    pub what: &'static str,
    pub granted: usize,
    pub requested: usize,
    pub remaining: usize,
}

impl GrantState {
    fn lock(&self) -> std::sync::MutexGuard<'_, GrantInner> {
        self.inner.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Draws `bytes`: what the grant held of them leaves it and goes back to the pool; beyond
    /// that, a recorded plan violation. The bytes taken from the grant.
    fn take(&self, bytes: usize) -> usize {
        let mut g = self.lock();
        let take = bytes.min(g.remaining);
        g.remaining -= take;
        self.counters.in_use.fetch_sub(take, Ordering::Relaxed);
        if bytes > take {
            g.violations.push(PlanViolation {
                what: self.what,
                granted: self.granted,
                requested: bytes,
                remaining: take,
            });
        }
        take
    }

    /// The `drawn` bytes of a granted allocation just freed back in the grant (counted as in
    /// use again), unless the grant has dropped.
    fn give_back(&self, drawn: usize) {
        let mut g = self.lock();
        if !g.closed {
            g.remaining += drawn;
            self.counters.in_use.fetch_add(drawn, Ordering::Relaxed);
        }
    }
}

impl Grant {
    /// Draws `bytes` for an allocation the engine makes next and keeps: they leave the grant
    /// and go back to the pool. Beyond what remains: a recorded plan violation, the rest drawn.
    pub fn draw(&self, bytes: usize) {
        self.state.take(bytes);
    }

    /// The bytes the grant still holds.
    pub fn remaining(&self) -> usize {
        self.state.lock().remaining
    }

    /// The bytes granted.
    pub fn granted(&self) -> usize {
        self.state.granted
    }

    /// The draws beyond the grant so far.
    pub fn violations(&self) -> Vec<PlanViolation> {
        self.state.lock().violations.clone()
    }
}

impl Drop for Grant {
    fn drop(&mut self) {
        let mut g = self.state.lock();
        g.closed = true;
        let left = std::mem::take(&mut g.remaining);
        self.state
            .counters
            .in_use
            .fetch_sub(left, Ordering::Relaxed);
    }
}

/// Memory accounting of one device (see the module docs).
#[derive(Debug)]
pub struct MemPool {
    counters: Arc<Counters>,
    pool: Option<sys::CUmemoryPool>,
}

// SAFETY: CUmemoryPool is a driver handle usable from any thread.
unsafe impl Send for MemPool {}
unsafe impl Sync for MemPool {}

/// Bytes in use, the high-water mark and the limit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PoolUsage {
    pub in_use: usize,
    pub peak: usize,
    pub limit: usize,
}

impl MemPool {
    /// Accounting with `limit` bytes; keeps freed memory in the driver's pool when the device
    /// supports stream-ordered allocation.
    pub(crate) fn new(ctx: &Arc<CudaContext>, limit: usize) -> Result<Self, GpuError> {
        let pool = if ctx.has_async_alloc() {
            ctx.bind_to_thread()?;
            let mut pool: sys::CUmemoryPool = std::ptr::null_mut();
            // SAFETY: valid device handle and out pointer.
            unsafe { sys::cuDeviceGetDefaultMemPool(&mut pool, ctx.cu_device()) }.result()?;
            let mut keep: u64 = u64::MAX;
            // SAFETY: the release threshold attribute takes a cuuint64_t.
            unsafe {
                sys::cuMemPoolSetAttribute(
                    pool,
                    sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_RELEASE_THRESHOLD,
                    &mut keep as *mut u64 as *mut _,
                )
            }
            .result()?;
            Some(pool)
        } else {
            None
        };
        let counters = Arc::new(Counters::default());
        counters.limit.store(limit, Ordering::Relaxed);
        Ok(MemPool { counters, pool })
    }

    pub fn usage(&self) -> PoolUsage {
        self.counters.usage()
    }

    pub(crate) fn meter(&self) -> PoolMeter {
        PoolMeter(self.counters.clone())
    }

    pub fn set_limit(&self, bytes: usize) {
        self.counters.limit.store(bytes, Ordering::Relaxed);
    }

    /// Sets the high-water mark to the bytes in use now ([`super::Gpu::reset_peak`]).
    pub fn reset_peak(&self) {
        self.counters.peak.store(
            self.counters.in_use.load(Ordering::Relaxed),
            Ordering::Relaxed,
        );
    }

    /// Returns cached free memory to the driver (call after a large calculation).
    pub fn trim(&self, stream: &CudaStream) -> Result<(), GpuError> {
        if let Some(pool) = self.pool {
            stream.synchronize()?;
            // SAFETY: valid pool handle.
            unsafe { sys::cuMemPoolTrimTo(pool, 0) }.result()?;
        }
        Ok(())
    }

    fn reserve(&self, bytes: usize) -> Result<(), GpuError> {
        let c = &self.counters;
        let limit = c.limit.load(Ordering::Relaxed);
        let mut cur = c.in_use.load(Ordering::Relaxed);
        let next = loop {
            let next = match cur.checked_add(bytes) {
                Some(next) if next <= limit => next,
                _ => {
                    return Err(GpuError::OutOfMemory {
                        requested: bytes,
                        in_use: cur,
                        limit,
                    })
                }
            };
            match c
                .in_use
                .compare_exchange_weak(cur, next, Ordering::Relaxed, Ordering::Relaxed)
            {
                Ok(_) => break next,
                Err(v) => cur = v,
            }
        };
        c.peak.fetch_max(next, Ordering::Relaxed);
        Ok(())
    }

    /// `bytes` set aside for one engine ([`Grant`]); an out-of-memory error when they do not
    /// fit the limit beside what is in use.
    pub(crate) fn grant(&self, bytes: usize, what: &'static str) -> Result<Grant, GpuError> {
        self.reserve(bytes)?;
        Ok(Grant {
            state: Arc::new(GrantState {
                counters: self.counters.clone(),
                what,
                granted: bytes,
                inner: std::sync::Mutex::new(GrantInner {
                    remaining: bytes,
                    ..GrantInner::default()
                }),
            }),
        })
    }

    /// [`MemPool::alloc_zeros`] drawn from `grant` (when given), the bytes back in the grant
    /// when the buffer is freed ([`Grant`]).
    pub(crate) fn alloc_zeros_granted<T: DeviceRepr + ValidAsZeroBits>(
        &self,
        stream: &Arc<CudaStream>,
        grant: Option<&Grant>,
        n: usize,
    ) -> Result<DeviceBuf<T>, GpuError> {
        let Some(g) = grant else {
            return self.alloc_zeros(stream, n);
        };
        let drawn = g.state.take(n.saturating_mul(std::mem::size_of::<T>()));
        match self.alloc_zeros(stream, n) {
            Ok(mut b) => {
                b.grant = Some((g.state.clone(), drawn));
                Ok(b)
            }
            Err(e) => {
                g.state.give_back(drawn);
                Err(e)
            }
        }
    }

    /// [`MemPool::upload`] drawn from `grant` (when given), as [`MemPool::alloc_zeros_granted`].
    pub(crate) fn upload_granted<T: DeviceRepr>(
        &self,
        stream: &Arc<CudaStream>,
        grant: Option<&Grant>,
        host: &[T],
    ) -> Result<DeviceBuf<T>, GpuError> {
        let Some(g) = grant else {
            return self.upload(stream, host);
        };
        let drawn = g.state.take(std::mem::size_of_val(host));
        match self.upload(stream, host) {
            Ok(mut b) => {
                b.grant = Some((g.state.clone(), drawn));
                Ok(b)
            }
            Err(e) => {
                g.state.give_back(drawn);
                Err(e)
            }
        }
    }

    fn release(&self, bytes: usize) {
        self.counters.in_use.fetch_sub(bytes, Ordering::Relaxed);
    }

    /// `n` zeroed elements on `stream`.
    pub(crate) fn alloc_zeros<T: DeviceRepr + ValidAsZeroBits>(
        &self,
        stream: &Arc<CudaStream>,
        n: usize,
    ) -> Result<DeviceBuf<T>, GpuError> {
        // a size beyond the address space is more than any limit
        let bytes = n.saturating_mul(std::mem::size_of::<T>());
        self.reserve(bytes)?;
        match stream.alloc_zeros::<T>(n) {
            Ok(buf) => Ok(DeviceBuf::new(buf, stream, bytes, self.counters.clone())),
            Err(e) => {
                self.release(bytes);
                Err(e.into())
            }
        }
    }

    /// A copy of `host` on `stream`.
    pub(crate) fn upload<T: DeviceRepr>(
        &self,
        stream: &Arc<CudaStream>,
        host: &[T],
    ) -> Result<DeviceBuf<T>, GpuError> {
        let bytes = std::mem::size_of_val(host);
        self.reserve(bytes)?;
        match stream.clone_htod(host) {
            Ok(buf) => Ok(DeviceBuf::new(buf, stream, bytes, self.counters.clone())),
            Err(e) => {
                self.release(bytes);
                Err(e.into())
            }
        }
    }
}

/// A device array of `T`, counted by its [`MemPool`]. Its memory is freed in stream order
/// when it is dropped, so dropping it right after queuing a kernel that uses it is safe.
#[derive(Debug)]
pub struct DeviceBuf<T> {
    buf: CudaSlice<T>,
    ptr: u64,
    bytes: usize,
    counters: Arc<Counters>,
    /// The grant the buffer was drawn from and the bytes it took ([`MemPool::alloc_zeros_granted`]).
    grant: Option<(Arc<GrantState>, usize)>,
}

impl<T> DeviceBuf<T> {
    fn new(
        buf: CudaSlice<T>,
        stream: &Arc<CudaStream>,
        bytes: usize,
        counters: Arc<Counters>,
    ) -> Self {
        let ptr = {
            let (p, _sync) = buf.device_ptr(stream);
            p
        };
        DeviceBuf {
            buf,
            ptr,
            bytes,
            counters,
            grant: None,
        }
    }

    pub fn len(&self) -> usize {
        self.buf.len()
    }

    pub fn is_empty(&self) -> bool {
        self.buf.len() == 0
    }

    /// The device address of the first element.
    pub fn ptr(&self) -> DevPtr<T> {
        DevPtr {
            addr: self.ptr,
            _elem: PhantomData,
        }
    }

    /// The device address of element `off` (at most [`DeviceBuf::len`], the end; checked in
    /// debug builds).
    pub fn at(&self, off: usize) -> DevPtr<T> {
        debug_assert!(
            off <= self.len(),
            "element {off} of a buffer of {}",
            self.len()
        );
        self.ptr() + off
    }

    pub(crate) fn raw(&self) -> &CudaSlice<T> {
        &self.buf
    }

    pub(crate) fn raw_mut(&mut self) -> &mut CudaSlice<T> {
        &mut self.buf
    }
}

impl<T> Drop for DeviceBuf<T> {
    fn drop(&mut self) {
        self.counters
            .in_use
            .fetch_sub(self.bytes, Ordering::Relaxed);
        if let Some((g, drawn)) = self.grant.take() {
            g.give_back(drawn);
        }
    }
}
