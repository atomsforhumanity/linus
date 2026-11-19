//! Streams beside the device's compute stream, and host memory streamed through the device.
//!
//! All device work runs in order on the compute stream (the [`Gpu`]'s). A [`CopyStream`] is a
//! second stream for copies between page-locked host memory and device memory that overlap
//! that work; the two are ordered only by [`Fence`]s. [`StreamedStore`] owns that whole
//! pattern: the host memory, a ring of device staging buffers, the copy stream and the fence of
//! each staging buffer's last use. An engine that keeps data on the host (the DF tensor's host
//! storage, the response's stored exchange operator) says which block goes through which
//! staging buffer; the store orders the copies with the work.

use super::device::Gpu;
use super::host::HostBuf;
use super::pool::DeviceBuf;
use crate::error::GpuError;
use cudarc::driver::{CudaStream, DeviceRepr, ValidAsZeroBits};
use std::ffi::c_void;
use std::sync::Arc;

/// A point on a stream ([`Gpu::fence`], or a copy stream's) that the other stream can wait for,
/// on the device.
pub struct Fence(pub(super) cudarc::driver::CudaEvent);

/// A second stream for copies that overlap the device's work ([`Gpu::copy_stream`]). Dropping
/// it waits for its copies.
pub(crate) struct CopyStream {
    pub(super) stream: Arc<CudaStream>,
}

impl CopyStream {
    /// A fence at the end of the copies queued so far.
    pub fn fence(&self) -> Result<Fence, GpuError> {
        Ok(Fence(self.stream.record_event(None)?))
    }

    /// The copies queued after this wait (on the device) for the work before `f`.
    pub fn wait_for(&self, f: &Fence) -> Result<(), GpuError> {
        Ok(self.stream.wait(&f.0)?)
    }

    /// Waits on the host for the copies queued so far.
    pub fn synchronize(&self) -> Result<(), GpuError> {
        Ok(self.stream.synchronize()?)
    }

    /// Queues dst[dst_off..dst_off + len] = host[host_off..host_off + len] on this stream.
    ///
    /// # Safety
    /// Neither buffer may be used by other work until the copy is done: the device's stream
    /// waits for a [`CopyStream::fence`] after it before it reads `dst`, this stream waited for
    /// a [`Gpu::fence`] after the device's last use of `dst`, and neither buffer is dropped or
    /// accessed from the host before this stream has passed the copy (its drop waits for it).
    pub unsafe fn upload<T: DeviceRepr>(
        &self,
        host: &HostBuf<T>,
        host_off: usize,
        dst: &DeviceBuf<T>,
        dst_off: usize,
        len: usize,
    ) -> Result<(), GpuError> {
        super::device::check_range("CopyStream::upload (host)", host_off, len, host.len())?;
        super::device::check_range("CopyStream::upload (device)", dst_off, len, dst.len())?;
        if len == 0 {
            return Ok(());
        }
        let sz = std::mem::size_of::<T>();
        self.stream.context().bind_to_thread()?;
        // SAFETY: both ranges checked above; ordering and lifetimes are the caller's contract.
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

    /// Queues host[host_off..host_off + len] = src[src_off..src_off + len] on this stream.
    ///
    /// # Safety
    /// As [`CopyStream::upload`]: this stream waited for a [`Gpu::fence`] after the device's
    /// last write of `src`, `src` is not written again before a later [`CopyStream::fence`] has
    /// been waited for, and the host part is not read before this stream has passed the copy.
    pub unsafe fn download<T: DeviceRepr>(
        &self,
        src: &DeviceBuf<T>,
        src_off: usize,
        host: &HostBuf<T>,
        host_off: usize,
        len: usize,
    ) -> Result<(), GpuError> {
        super::device::check_range("CopyStream::download (device)", src_off, len, src.len())?;
        super::device::check_range("CopyStream::download (host)", host_off, len, host.len())?;
        if len == 0 {
            return Ok(());
        }
        let sz = std::mem::size_of::<T>();
        self.stream.context().bind_to_thread()?;
        // SAFETY: both ranges checked above; ordering and lifetimes are the caller's contract.
        unsafe {
            cudarc::driver::sys::cuMemcpyDtoHAsync_v2(
                host.ptr().add(host_off) as *mut c_void,
                src.at(src_off).addr(),
                len * sz,
                self.stream.cu_stream(),
            )
        }
        .result()?;
        Ok(())
    }
}

impl Drop for CopyStream {
    fn drop(&mut self) {
        // copies that still read or write host or device buffers finish first
        let _ = self.stream.synchronize();
    }
}

/// How a [`StreamedStore`] orders its copies with the device's work.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Copies {
    /// On the compute stream, in order with the work (no overlap and no fences).
    InOrder,
    /// On a copy stream of the store's own, overlapping the work: each staging buffer's uses
    /// on the two streams are ordered by the fence of its last use.
    Overlapped,
}

/// The stream that used a staging buffer last.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Side {
    Compute,
    Copy,
}

/// Page-locked host memory streamed through a ring of device staging buffers ("slots").
///
/// The protocol, the same for every user: the compute stream takes a slot with
/// [`StreamedStore::slot`] (it waits for the slot's last copy), works on it, and gives it back
/// with [`StreamedStore::release`]; [`StreamedStore::load`] (host block to slot) and
/// [`StreamedStore::store`] (slot to host block) wait for the slot's last compute use. A slot is
/// never written by one stream while the other still uses it, and the host memory is not
/// touched by the host while a copy is pending (its accessors wait). Dropping the store waits
/// for its copies before the host memory and the staging buffers go: explicitly, not by the
/// order of fields.
///
/// With [`Copies::InOrder`] every copy is queued on the compute stream and the protocol's
/// waits are no-ops: the copies run between the work that surrounds them, as plain in-order
/// code would.
pub struct StreamedStore<T> {
    copy: Option<CopyStream>,
    host: HostBuf<T>,
    staging: Vec<DeviceBuf<T>>,
    /// Per slot, the fence of its last use and the stream that recorded it.
    last: Vec<Option<(Side, Fence)>>,
}

impl<T: DeviceRepr + ValidAsZeroBits> StreamedStore<T> {
    /// `len` zeroed elements of host memory and `slots` staging buffers of `slot_len` elements
    /// (allocated in that order), copies as `copies` says.
    pub fn new(
        gpu: &Gpu,
        len: usize,
        (slots, slot_len): (usize, usize),
        copies: Copies,
    ) -> Result<Self, GpuError> {
        let host = gpu.alloc_host::<T>(len)?;
        Self::with_host(gpu, host, (slots, slot_len), copies)
    }

    /// The store of `host` (already filled, e.g. by a setup's downloads): `slots` staging
    /// buffers of `slot_len` elements, copies as `copies` says.
    pub fn with_host(
        gpu: &Gpu,
        host: HostBuf<T>,
        (slots, slot_len): (usize, usize),
        copies: Copies,
    ) -> Result<Self, GpuError> {
        let staging = (0..slots)
            .map(|_| gpu.alloc_zeros::<T>(slot_len))
            .collect::<Result<Vec<_>, _>>()?;
        let copy = match copies {
            Copies::InOrder => None,
            Copies::Overlapped => Some(gpu.copy_stream()?),
        };
        Ok(StreamedStore {
            copy,
            host,
            last: (0..slots).map(|_| None).collect(),
            staging,
        })
    }
}

impl<T: DeviceRepr> StreamedStore<T> {
    /// The number of staging buffers.
    pub fn slots(&self) -> usize {
        self.staging.len()
    }

    /// Bytes of the staging buffers on the device and of the host memory.
    pub fn bytes(&self) -> (usize, usize) {
        let dev = self.staging.iter().map(|s| s.len()).sum::<usize>() * std::mem::size_of::<T>();
        (dev, self.host.bytes())
    }

    /// The host memory (its accessors wait for the compute stream).
    pub fn host(&self) -> &HostBuf<T> {
        &self.host
    }

    /// The host memory, for the compute stream's own copies into it (a setup's downloads).
    pub fn host_mut(&mut self) -> &mut HostBuf<T> {
        &mut self.host
    }

    fn check_slot(&self, k: usize) -> Result<(), GpuError> {
        if k < self.staging.len() {
            Ok(())
        } else {
            Err(GpuError::Contract(format!(
                "streamed store: slot {k} of {}",
                self.staging.len()
            )))
        }
    }

    /// The copy stream waits for slot k's last use on the compute stream.
    fn copy_after_compute(&mut self, k: usize) -> Result<(), GpuError> {
        if let (Some(copy), Some((Side::Compute, f))) = (&self.copy, &self.last[k]) {
            copy.wait_for(f)?;
            self.last[k] = None;
        }
        Ok(())
    }

    /// Slot k for the compute stream: it waits for the slot's last copy (a load into it, or a
    /// store out of it that must finish before the slot is overwritten).
    pub fn slot(&mut self, gpu: &Gpu, k: usize) -> Result<&mut DeviceBuf<T>, GpuError> {
        self.check_slot(k)?;
        if let Some((Side::Copy, f)) = &self.last[k] {
            gpu.wait_for(f)?;
            self.last[k] = None;
        }
        Ok(&mut self.staging[k])
    }

    /// The compute stream's work on slot k so far is its last use (the copies after it wait).
    pub fn release(&mut self, gpu: &Gpu, k: usize) -> Result<(), GpuError> {
        self.check_slot(k)?;
        if self.copy.is_some() {
            self.last[k] = Some((Side::Compute, gpu.fence()?));
        }
        Ok(())
    }

    /// Queues host[off..off + len] into slot k from its start.
    pub fn load(&mut self, gpu: &Gpu, k: usize, off: usize, len: usize) -> Result<(), GpuError> {
        self.check_slot(k)?;
        self.copy_after_compute(k)?;
        match &self.copy {
            None => gpu.upload_from_host(&self.host, off, &mut self.staging[k], 0, len),
            Some(copy) => {
                // SAFETY: the copy stream waited for the slot's last compute use above; the
                // compute stream waits for this copy's fence (`slot`) before it reads the slot;
                // the host memory and the slot live in the store, whose drop waits for the
                // copy stream.
                unsafe { copy.upload(&self.host, off, &self.staging[k], 0, len)? };
                self.last[k] = Some((Side::Copy, copy.fence()?));
                Ok(())
            }
        }
    }

    /// Queues slot k's first `len` elements into host[off..off + len], after the slot's last
    /// compute use ([`StreamedStore::release`]).
    pub fn store(&mut self, gpu: &Gpu, k: usize, off: usize, len: usize) -> Result<(), GpuError> {
        self.check_slot(k)?;
        self.copy_after_compute(k)?;
        match &self.copy {
            None => gpu.download_block_to_host(
                &self.staging[k],
                0,
                len,
                &mut self.host,
                off,
                len,
                len,
                1,
            ),
            Some(copy) => {
                // SAFETY: the copy stream waited for the slot's last compute use (its fill)
                // above; the compute stream waits for this copy's fence (`slot`) before it
                // writes the slot again; the store's drop waits for the copy stream.
                unsafe { copy.download(&self.staging[k], 0, &self.host, off, len)? };
                self.last[k] = Some((Side::Copy, copy.fence()?));
                Ok(())
            }
        }
    }

    /// The compute stream waits (on the device) for every copy queued so far.
    pub fn join(&self, gpu: &Gpu) -> Result<(), GpuError> {
        match &self.copy {
            Some(c) => gpu.wait_for(&c.fence()?),
            None => Ok(()),
        }
    }
}

impl<T> Drop for StreamedStore<T> {
    /// The copies finish before the host memory is unmapped and the staging buffers are freed
    /// (the host buffer's own drop waits for the compute stream).
    fn drop(&mut self) {
        if let Some(c) = &self.copy {
            let _ = c.synchronize();
        }
    }
}
