//! Every Lebedev set integrates all spherical harmonics up to its degree exactly (to the
//! ~1e-14 accuracy of the published tables) and fails at the next even degree.

use linus_grid::lebedev::{lebedev, LEBEDEV_DEGREE, LEBEDEV_NPOINTS};

/// max_m |sum_i w_i Y_lm(u_i)| for each l = 0..=lmax, with 4pi-normalised real harmonics
/// (mean of Y^2 over the sphere = 1), so exact integrals are delta_l0.
#[allow(clippy::needless_range_loop, reason = "indexes several arrays in step")]
fn harmonic_integrals(n: usize, lmax: usize) -> Vec<f64> {
    let g = lebedev(n).unwrap();
    let mut acc = vec![vec![0.0f64; 2 * lmax + 1]; lmax + 1];
    let mut p = vec![vec![0.0f64; lmax + 1]; lmax + 1];
    for i in 0..n {
        let (x, y, z) = (g.x[i], g.y[i], g.z[i]);
        let ct = z;
        let st = (x * x + y * y).sqrt();
        let phi = y.atan2(x);
        // fully normalised associated Legendre functions (geodesy normalisation)
        p[0][0] = 1.0;
        for m in 1..=lmax {
            let f = if m == 1 {
                3f64.sqrt()
            } else {
                ((2 * m + 1) as f64 / (2 * m) as f64).sqrt()
            };
            p[m][m] = f * st * p[m - 1][m - 1];
        }
        for m in 0..lmax {
            p[m + 1][m] = ((2 * m + 3) as f64).sqrt() * ct * p[m][m];
        }
        for m in 0..=lmax {
            for l in (m + 2)..=lmax {
                let (lf, mf) = (l as f64, m as f64);
                let a = ((4.0 * lf * lf - 1.0) / (lf * lf - mf * mf)).sqrt();
                let b = (((lf - 1.0) * (lf - 1.0) - mf * mf)
                    / (4.0 * (lf - 1.0) * (lf - 1.0) - 1.0))
                    .sqrt();
                p[l][m] = a * (ct * p[l - 1][m] - b * p[l - 2][m]);
            }
        }
        for l in 0..=lmax {
            acc[l][lmax] += g.w[i] * p[l][0];
            for m in 1..=l {
                let s2 = std::f64::consts::SQRT_2;
                let (s, c) = (m as f64 * phi).sin_cos();
                acc[l][lmax + m] += g.w[i] * s2 * p[l][m] * c;
                acc[l][lmax - m] += g.w[i] * s2 * p[l][m] * s;
            }
        }
    }
    acc.iter()
        .map(|v| v.iter().fold(0.0f64, |a, &b| a.max(b.abs())))
        .collect()
}

#[test]
fn lebedev_sets_exact_to_their_degree() {
    for (k, &n) in LEBEDEV_NPOINTS.iter().enumerate() {
        let deg = LEBEDEV_DEGREE[k];
        let ints = harmonic_integrals(n, deg + 2);
        assert!(
            (ints[0] - 1.0).abs() < 3e-14,
            "n={n}: sum of weights {}",
            ints[0]
        );
        let worst = ints[1..=deg].iter().cloned().fold(0.0f64, f64::max);
        assert!(
            worst < 1e-13,
            "n={n} (degree {deg}): max |int Y_lm| = {worst:e}"
        );
        // the next even degree is not integrated exactly (the degree is tight)
        assert!(
            ints[deg + 1] > 1e-6,
            "n={n}: degree {} also exact ({:e})",
            deg + 1,
            ints[deg + 1]
        );
        println!(
            "n={n:5} degree={deg:3} max|int Y_lm|(1<=l<=deg)={worst:.1e} l=deg+1: {:.1e}",
            ints[deg + 1]
        );
    }
}
