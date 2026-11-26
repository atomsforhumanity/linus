//! Helpers of the facade's unit tests.

/// The device of a GPU test, or None (skipped, with a line on stderr) without a usable one,
/// unless `LINUS_GPU_REQUIRED=1`: then a panic, so a GPU run of the tests cannot pass by
/// skipping.
#[cfg(feature = "gpu")]
pub(crate) fn gpu_or_skip() -> Option<linus_gpu::Gpu> {
    match linus_gpu::Gpu::new(0) {
        Ok(g) => Some(g),
        Err(e) if std::env::var("LINUS_GPU_REQUIRED").as_deref() != Ok("1") => {
            eprintln!("no usable GPU, skipping: {e}");
            None
        }
        Err(e) => panic!("LINUS_GPU_REQUIRED=1 and no usable GPU: {e}"),
    }
}
