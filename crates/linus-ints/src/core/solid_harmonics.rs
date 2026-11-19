//! Real solid harmonics as combinations of Cartesian Gaussian components.
//!
//! `cart2sph(l)` returns a (2l+1) x ncart(l) row-major matrix `T` such that spherical function
//! m = sum_k T\[m\]\[k\] * cart_k, where cart_k = x^lx y^ly z^lz exp(-a r^2) carries the
//! normalization of x^l (the libcint/PySCF convention). Rows are ordered m = -l..l, except
//! for l = 1 where the order is x, y, z (m = +1, -1, 0), as in PySCF. Each row is normalized
//! to unit norm in the Cartesian metric. Signs follow the standard real solid harmonics
//! (Helgaker, Jørgensen, Olsen, eq. 6.4.47-6.4.50), which are also PySCF's.

use crate::basis::{cart_components, dfact, ncart};

fn binom(n: i64, k: i64) -> f64 {
    if k < 0 || k > n {
        return 0.0;
    }
    let mut r = 1.0;
    for i in 0..k {
        r = r * (n - i) as f64 / (i + 1) as f64;
    }
    r
}

/// Unnormalized coefficients of S_lm over Cartesian monomials, keyed by component index.
fn solid_harmonic_coefs(l: usize, m: i64) -> Vec<f64> {
    let comps = cart_components(l);
    let mut c = vec![0.0; comps.len()];
    let li = l as i64;
    let am = m.abs();
    // v_m = 0 for m >= 0, 1/2 for m < 0. Work with 2v to stay in integers.
    let vm2: i64 = if m < 0 { 1 } else { 0 };
    for t in 0..=((li - am) / 2) {
        for u in 0..=t {
            // v runs over v_m, v_m + 1, ..., <= floor(|m|/2 - v_m) + v_m
            let vmax2 = 2 * ((am - vm2) / 2) + vm2; // largest 2v
            let mut v2 = vm2;
            while v2 <= vmax2 {
                let sign_exp = t + (v2 - vm2) / 2;
                let sign = if sign_exp % 2 == 0 { 1.0 } else { -1.0 };
                let coef = sign
                    * 0.25f64.powi(t as i32)
                    * binom(li, t)
                    * binom(li - t, am + t)
                    * binom(t, u)
                    * binom(am, v2);
                let px = 2 * t + am - 2 * u - v2; // 2t + |m| - 2(u+v)
                let py = 2 * u + v2; // 2(u+v)
                let pz = li - 2 * t - am;
                if px >= 0 && py >= 0 && pz >= 0 {
                    let idx = comps
                        .iter()
                        .position(|&[a, b, cz]| a as i64 == px && b as i64 == py && cz as i64 == pz)
                        .expect("component");
                    c[idx] += coef;
                }
                v2 += 2;
            }
        }
    }
    c
}

/// Overlap of Cartesian components (x^l normalization) of one shell with itself, relative.
pub fn cart_metric(l: usize) -> Vec<f64> {
    let comps = cart_components(l);
    let n = comps.len();
    let norm = dfact(2 * l as i64 - 1);
    let mut m = vec![0.0; n * n];
    for (i, a) in comps.iter().enumerate() {
        for (j, b) in comps.iter().enumerate() {
            let mut v = 1.0;
            for k in 0..3 {
                let s = a[k] + b[k];
                if s % 2 == 1 {
                    v = 0.0;
                    break;
                }
                v *= dfact(s as i64 - 1);
            }
            m[i * n + j] = v / norm;
        }
    }
    m
}

/// Cached transformation matrices for l <= 12 (computed once).
pub fn cart2sph_cached(l: usize) -> &'static [f64] {
    use std::sync::OnceLock;
    static T: OnceLock<Vec<Vec<f64>>> = OnceLock::new();
    let t = T.get_or_init(|| (0..=12).map(cart2sph_compute).collect());
    &t[l]
}

pub fn cart2sph(l: usize) -> Vec<f64> {
    if l <= 12 {
        return cart2sph_cached(l).to_vec();
    }
    cart2sph_compute(l)
}

fn cart2sph_compute(l: usize) -> Vec<f64> {
    let nc = ncart(l);
    if l == 0 {
        return vec![1.0];
    }
    if l == 1 {
        return vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
    }
    let metric = cart_metric(l);
    let mut out = Vec::with_capacity((2 * l + 1) * nc);
    for m in -(l as i64)..=(l as i64) {
        let mut c = solid_harmonic_coefs(l, m);
        let mut n2 = 0.0;
        for i in 0..nc {
            for j in 0..nc {
                n2 += c[i] * metric[i * nc + j] * c[j];
            }
        }
        let f = 1.0 / n2.sqrt();
        c.iter_mut().for_each(|x| *x *= f);
        out.extend_from_slice(&c);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[allow(clippy::erasing_op, clippy::identity_op)] // row * nc + column, written out for row 0 too
    fn d_functions() {
        let t = cart2sph(2);
        // m=-2: xy ; m=0: (2zz - xx - yy)/2 normalized ; m=2: (xx-yy)*sqrt(3)/2
        let nc = 6;
        // xy coefficient: component index 1, metric <xy|xy> = 1/3 -> coef sqrt(3)
        assert!((t[0 * nc + 1] - 3f64.sqrt()).abs() < 1e-14);
        assert!((t[2 * nc + 5] - 1.0).abs() < 1e-14);
        assert!((t[2 * nc + 0] + 0.5).abs() < 1e-14);
        assert!((t[4 * nc + 0] - 3f64.sqrt() / 2.0).abs() < 1e-14);
        assert!((t[4 * nc + 3] + 3f64.sqrt() / 2.0).abs() < 1e-14);
    }

    #[test]
    fn rows_orthonormal() {
        for l in 2..=6 {
            let t = cart2sph(l);
            let m = cart_metric(l);
            let nc = ncart(l);
            let ns = 2 * l + 1;
            for a in 0..ns {
                for b in 0..ns {
                    let mut s = 0.0;
                    for i in 0..nc {
                        for j in 0..nc {
                            s += t[a * nc + i] * m[i * nc + j] * t[b * nc + j];
                        }
                    }
                    let e = if a == b { 1.0 } else { 0.0 };
                    assert!((s - e).abs() < 1e-12, "l={l} a={a} b={b} s={s}");
                }
            }
        }
    }
}
