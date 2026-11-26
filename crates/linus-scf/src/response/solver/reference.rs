//! The shared-subspace solver's test problem (feature `reference`): linus-scf's solver tests and
//! the facade's test of a device's algebra solve it.

use faer::Mat;

/// Fixed pseudo-random numbers in [-0.5, 0.5) (a 64-bit LCG).
pub fn noise(seed: u64, len: usize) -> Vec<f64> {
    let mut s = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (0..len)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 11) as f64) / ((1u64 << 53) as f64) - 0.5
        })
        .collect()
}

/// A symmetric positive definite test operator d + U U^T (U len x r) with right-hand sides
/// `nr` (the last `dup` of them copies of earlier ones plus a tiny multiple of another, so that
/// their directions become near-dependent and are dropped).
pub struct Problem {
    pub d: Vec<f64>,
    pub u: Mat<f64>,
    pub b: Mat<f64>,
}

impl Problem {
    /// The problem of `len` unknowns, U of rank `r`, `nr` right-hand sides, `dup` near-copies.
    pub fn new(len: usize, r: usize, nr: usize, dup: usize) -> Problem {
        let d: Vec<f64> = noise(1, len)
            .iter()
            .map(|x| 1.0 + 4.0 * (x + 0.5))
            .collect();
        let un = noise(2, len * r);
        let u = Mat::from_fn(len, r, |i, j| 0.05 * un[j * len + i]);
        let bn = noise(3, len * nr);
        let b = Mat::from_fn(len, nr, |i, k| {
            if k + dup >= nr && k >= dup {
                bn[(k - dup) * len + i] + 1e-13 * bn[(k - 1) * len + i]
            } else {
                bn[k * len + i]
            }
        });
        Problem { d, u, b }
    }

    /// A z of each direction, and an "extra" of a third of the length (its first entries).
    pub fn apply(&self, zs: &[Vec<f64>]) -> Vec<(Vec<f64>, Vec<f64>)> {
        zs.iter()
            .map(|z| {
                let ut: Vec<f64> = (0..self.u.ncols())
                    .map(|j| {
                        self.u
                            .col_as_slice(j)
                            .iter()
                            .zip(z)
                            .map(|(a, b)| a * b)
                            .sum()
                    })
                    .collect();
                let az: Vec<f64> = (0..z.len())
                    .map(|i| {
                        self.d[i] * z[i]
                            + (0..ut.len()).map(|j| self.u[(i, j)] * ut[j]).sum::<f64>()
                    })
                    .collect();
                let e = z[..z.len() / 3].to_vec();
                (az, e)
            })
            .collect()
    }
}
