//! What the linus-gpu examples share.

use linus_gpu::{Gpu, GpuError};

/// Device milliseconds of the work `f` queues on `gpu`'s stream, from timing marks around it
/// (waits for the work).
pub fn device_ms(gpu: &Gpu, f: impl FnOnce() -> Result<(), GpuError>) -> Result<f64, GpuError> {
    let start = gpu.mark()?;
    f()?;
    Ok(start.seconds_until(&gpu.mark()?)? * 1e3)
}
