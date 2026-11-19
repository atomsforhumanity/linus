//! Page-locked (pinned) host memory: the host side of streamed device work. Copies between a
//! [`HostBuf`] and device memory are queued on the device's stream and run asynchronously
//! (DMA, full PCIe bandwidth); the host may read or write the buffer only through the methods
//! here, which first wait for the stream.
//!
//! On Unix the memory is an anonymous mapping (zero-filled by the OS; huge pages advised on
//! Linux) whose pages the current rayon pool's threads fault in, then page-locked and
//! registered with the driver (`cuMemHostRegister`). For 51 GiB that takes 1.5 s (touch 1.1 s
//! on 8 threads, registration 0.4 s) against 8.4 s for `cuMemHostAlloc` and a zeroing pass
//! (7.1 s + 1.3 s; its free takes another 2.6 s), whose single-threaded pinning dominated the
//! density-fitting setup of a host-streamed tensor. Elsewhere: `cuMemHostAlloc`, then zeroed.

use crate::error::GpuError;
use cudarc::driver::{CudaStream, DeviceRepr, ValidAsZeroBits};
use std::sync::Arc;

/// `len` elements of `T` in page-locked host memory, zero-initialized.
pub struct HostBuf<T> {
    ptr: *mut T,
    len: usize,
    /// Bytes of the allocation (at least one element's).
    bytes: usize,
    stream: Arc<CudaStream>,
}

// SAFETY: the allocation is plain host memory owned by this value; accesses go through &self
// / &mut self, and device copies are ordered on the stream, which is waited for before host
// access and before freeing.
unsafe impl<T: Send> Send for HostBuf<T> {}
unsafe impl<T: Sync> Sync for HostBuf<T> {}

/// Bytes per page touched when faulting a mapping in (the smallest page size).
#[cfg(unix)]
const PAGE: usize = 4096;

/// A zero-filled anonymous mapping of `bytes`, its pages faulted in by the current rayon pool,
/// page-locked and registered with the driver (the context bound to this thread).
#[cfg(unix)]
fn alloc_registered(bytes: usize) -> Result<*mut u8, GpuError> {
    use cudarc::driver::sys;
    use rayon::prelude::*;
    // SAFETY: a new private anonymous mapping (no file, no fixed address).
    let p = unsafe {
        libc::mmap(
            std::ptr::null_mut(),
            bytes,
            libc::PROT_READ | libc::PROT_WRITE,
            libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
            -1,
            0,
        )
    };
    if p == libc::MAP_FAILED {
        return Err(GpuError::HostMemory {
            bytes,
            why: format!("mmap: {}", std::io::Error::last_os_error()),
        });
    }
    // huge pages: fewer faults and much cheaper locking (advice only; failure is harmless)
    #[cfg(target_os = "linux")]
    // SAFETY: advice on the mapping just created.
    unsafe {
        libc::madvise(p, bytes, libc::MADV_HUGEPAGE);
    }
    // fault the pages in from several threads (the OS zero-fills them; writing a zero byte
    // per page changes nothing)
    let base = p as usize;
    let span = 1usize << 26;
    (0..bytes.div_ceil(span)).into_par_iter().for_each(|i| {
        let end = ((i + 1) * span).min(bytes);
        let mut at = i * span;
        while at < end {
            // SAFETY: inside the mapping; a volatile write so that the touch is not removed.
            unsafe { std::ptr::write_volatile((base + at) as *mut u8, 0) };
            at += PAGE;
        }
    });
    // SAFETY: the mapping is `bytes` long and stays mapped until it is unregistered (drop).
    let r = unsafe { sys::cuMemHostRegister_v2(p, bytes, 0) };
    if r != sys::cudaError_enum::CUDA_SUCCESS {
        // SAFETY: the mapping above, unused elsewhere.
        unsafe { libc::munmap(p, bytes) };
        return Err(GpuError::HostMemory {
            bytes,
            why: format!("cuMemHostRegister: {r:?}"),
        });
    }
    Ok(p as *mut u8)
}

impl<T: DeviceRepr + ValidAsZeroBits> HostBuf<T> {
    /// Allocates `len` zeroed elements (page-locked; see the module docs).
    pub(crate) fn new(stream: &Arc<CudaStream>, len: usize) -> Result<Self, GpuError> {
        let bytes = len
            .max(1)
            .checked_mul(std::mem::size_of::<T>())
            .ok_or_else(|| GpuError::HostMemory {
                bytes: usize::MAX,
                why: format!("{len} elements overflow the address space"),
            })?;
        stream.context().bind_to_thread()?;
        #[cfg(unix)]
        let p = alloc_registered(bytes)? as *mut T;
        #[cfg(not(unix))]
        let p = {
            // SAFETY: plain page-locked allocation (no flags: cached, readable by the host).
            let p = unsafe { cudarc::driver::result::malloc_host(bytes, 0) }.map_err(|e| {
                GpuError::HostMemory {
                    bytes,
                    why: format!("cuMemHostAlloc: {e:?}"),
                }
            })? as *mut T;
            // SAFETY: freshly allocated, `bytes` long; all-zero bytes are a valid T.
            unsafe { std::ptr::write_bytes(p as *mut u8, 0, bytes) };
            p
        };
        Ok(HostBuf {
            ptr: p,
            len,
            bytes,
            stream: stream.clone(),
        })
    }
}

impl<T> HostBuf<T> {
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn bytes(&self) -> usize {
        self.len * std::mem::size_of::<T>()
    }

    pub(crate) fn ptr(&self) -> *mut T {
        self.ptr
    }

    /// The contents (waits for the copies queued on the stream).
    pub fn as_slice(&self) -> Result<&[T], GpuError> {
        self.stream.synchronize()?;
        // SAFETY: len elements owned by self; no device copy is pending after the wait.
        Ok(unsafe { std::slice::from_raw_parts(self.ptr, self.len) })
    }

    /// The contents, writable (waits for the copies queued on the stream).
    pub fn as_mut_slice(&mut self) -> Result<&mut [T], GpuError> {
        self.stream.synchronize()?;
        // SAFETY: as as_slice, exclusive through &mut self.
        Ok(unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) })
    }
}

impl<T> Drop for HostBuf<T> {
    fn drop(&mut self) {
        // copies that still read or write the buffer must finish before it is freed
        let _ = self.stream.synchronize();
        let _ = self.stream.context().bind_to_thread();
        #[cfg(unix)]
        // SAFETY: registered and mapped in new, released once.
        unsafe {
            let _ = cudarc::driver::sys::cuMemHostUnregister(self.ptr as *mut _);
            libc::munmap(self.ptr as *mut _, self.bytes);
        }
        #[cfg(not(unix))]
        // SAFETY: allocated by malloc_host in new, freed once.
        let _ = unsafe { cudarc::driver::result::free_host(self.ptr as *mut _) };
    }
}
