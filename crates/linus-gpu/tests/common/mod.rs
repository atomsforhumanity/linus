#![allow(dead_code)]

/// Device 0, or None (test skipped) when this machine has no usable GPU. With
/// LINUS_GPU_REQUIRED=1 a missing GPU fails the test instead.
#[cfg(feature = "gpu")]
pub fn gpu() -> Option<linus_gpu::Gpu> {
    linus_gpu::harness::gpu()
}

/// Deterministic pseudo-random numbers (xorshift64*).
pub struct Rng(pub u64);

impl Rng {
    pub fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }
    /// Uniform in [0, 1).
    pub fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
    /// Signed values spread over many decades, as quadrature contributions are.
    pub fn wide(&mut self) -> f64 {
        let mag = 10f64.powf(-14.0 + 17.0 * self.uniform());
        if self.uniform() < 0.5 {
            -mag
        } else {
            mag
        }
    }
}
