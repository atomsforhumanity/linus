//! What the linus-gpu examples share: the command line's options, the median of repeated
//! timings, orthonormal test orbitals and their densities, device timing.
#![allow(
    dead_code,
    reason = "each example compiles this module as its own and uses a part"
)]

use linus_gpu::{Gpu, GpuError};

/// Device milliseconds of the work `f` queues on `gpu`'s stream, from timing marks around it
/// (waits for the work).
pub fn device_ms(gpu: &Gpu, f: impl FnOnce() -> Result<(), GpuError>) -> Result<f64, GpuError> {
    let start = gpu.mark()?;
    f()?;
    Ok(start.seconds_until(&gpu.mark()?)? * 1e3)
}

/// The value after `name` on the command line (`name value`), if given.
pub fn arg(args: &[String], name: &str) -> Option<String> {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1).cloned())
}

/// Whether the switch `name` is on the command line.
pub fn flag(args: &[String], name: &str) -> bool {
    args.iter().any(|a| a == name)
}

/// The median of `v` (the upper of the two middle values for an even count).
pub fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

/// The median of repeated calls after the first (which compiles and warms up), or the first
/// alone.
pub fn median_warm(v: &[f64]) -> f64 {
    median(if v.len() > 1 { &v[1..] } else { v }.to_vec())
}

/// n x o column-major orbitals: a deterministic pseudo-random matrix, columns made orthonormal
/// (Gram-Schmidt) so the magnitudes are those of a molecule's orbitals (the engines' work
/// depends on the sizes, not on the values).
pub fn orbitals(n: usize, o: usize, seed: u64) -> Vec<f64> {
    let mut s = seed;
    let mut c: Vec<f64> = (0..n * o)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 11) as f64 / (1u64 << 53) as f64) - 0.5
        })
        .collect();
    for j in 0..o {
        for k in 0..j {
            let d: f64 = (0..n).map(|i| c[i + j * n] * c[i + k * n]).sum();
            for i in 0..n {
                c[i + j * n] -= d * c[i + k * n];
            }
        }
        let nrm = (0..n).map(|i| c[i + j * n].powi(2)).sum::<f64>().sqrt();
        for i in 0..n {
            c[i + j * n] /= nrm;
        }
    }
    c
}

/// f C C^T of the first `o` columns of `c` (n x n, column-major).
pub fn density(c: &[f64], n: usize, o: usize, f: f64) -> Vec<f64> {
    let mut d = vec![0.0; n * n];
    for j in 0..n {
        for i in 0..n {
            d[i + j * n] = f * (0..o).map(|k| c[i + k * n] * c[j + k * n]).sum::<f64>();
        }
    }
    d
}
