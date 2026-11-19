//! Lebedev-Laikov angular quadratures on the unit sphere.
//!
//! The orbit generators come from `lebedev_data.rs` (see its header for provenance and
//! license: Robert Parrish's BSD-2-Clause `getLebedevSphere` as shipped in SciPy).  The
//! points are generated from the octahedral orbits in the same order as PySCF's
//! `SphGenOh`, with the same floating-point formulas (`b = sqrt(1 - 2 a^2)`, ...), so the
//! unit vectors are bit-identical to PySCF's.  Weights are normalised to sum to 1 (the
//! 4*pi lives in the radial weights, as in PySCF).

use crate::lebedev_data::LEBEDEV_TABLE;
use std::sync::OnceLock;

/// All available Lebedev point counts (ascending).
pub const LEBEDEV_NPOINTS: [usize; 32] = [
    6, 14, 26, 38, 50, 74, 86, 110, 146, 170, 194, 230, 266, 302, 350, 434, 590, 770, 974, 1202,
    1454, 1730, 2030, 2354, 2702, 3074, 3470, 3890, 4334, 4802, 5294, 5810,
];

/// Algebraic degree of exactness of each set in [`LEBEDEV_NPOINTS`].
pub const LEBEDEV_DEGREE: [usize; 32] = [
    3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 35, 41, 47, 53, 59, 65, 71, 77, 83, 89,
    95, 101, 107, 113, 119, 125, 131,
];

/// A Lebedev grid: unit vectors and weights (sum of weights = 1).
#[derive(Clone, Debug)]
pub struct LebedevGrid {
    pub npoints: usize,
    pub degree: usize,
    pub x: Vec<f64>,
    pub y: Vec<f64>,
    pub z: Vec<f64>,
    pub w: Vec<f64>,
}

static CACHE: [OnceLock<LebedevGrid>; 32] = [const { OnceLock::new() }; 32];

/// Index of a point count in [`LEBEDEV_NPOINTS`].
pub fn lebedev_index(npoints: usize) -> Option<usize> {
    LEBEDEV_NPOINTS.iter().position(|&n| n == npoints)
}

/// Degree of exactness for a point count.
pub fn lebedev_degree(npoints: usize) -> Option<usize> {
    lebedev_index(npoints).map(|i| LEBEDEV_DEGREE[i])
}

/// Point count of the Lebedev set of a given algebraic degree (exact match).
pub fn lebedev_npoints_for_degree(degree: usize) -> Option<usize> {
    LEBEDEV_DEGREE
        .iter()
        .position(|&d| d == degree)
        .map(|i| LEBEDEV_NPOINTS[i])
}

/// Smallest Lebedev set with degree >= `degree`.
pub fn lebedev_npoints_at_least_degree(degree: usize) -> Option<usize> {
    LEBEDEV_DEGREE
        .iter()
        .position(|&d| d >= degree)
        .map(|i| LEBEDEV_NPOINTS[i])
}

/// Returns the (cached) Lebedev grid with `npoints` points, or `None` if no such set exists.
pub fn lebedev(npoints: usize) -> Option<&'static LebedevGrid> {
    let idx = lebedev_index(npoints)?;
    Some(CACHE[idx].get_or_init(|| build(idx)))
}

fn build(idx: usize) -> LebedevGrid {
    let (n, degree, orbits) = LEBEDEV_TABLE[idx];
    assert_eq!(n, LEBEDEV_NPOINTS[idx]);
    let mut g = LebedevGrid {
        npoints: n,
        degree,
        x: Vec::with_capacity(n),
        y: Vec::with_capacity(n),
        z: Vec::with_capacity(n),
        w: Vec::with_capacity(n),
    };
    for &(t, a, b, v) in orbits {
        gen_oh(t, a, b, v, &mut g);
    }
    assert_eq!(g.x.len(), n);
    g
}

/// Octahedral sign pattern (x fastest, then y, then z), as in PySCF's SphGenOh.
const SIGNS8: [[f64; 3]; 8] = [
    [1., 1., 1.],
    [-1., 1., 1.],
    [1., -1., 1.],
    [-1., -1., 1.],
    [1., 1., -1.],
    [-1., 1., -1.],
    [1., -1., -1.],
    [-1., -1., -1.],
];

fn push(g: &mut LebedevGrid, x: f64, y: f64, z: f64, v: f64) {
    g.x.push(x);
    g.y.push(y);
    g.z.push(z);
    g.w.push(v);
}

fn gen_oh(t: u8, a: f64, b: f64, v: f64, g: &mut LebedevGrid) {
    match t {
        1 => {
            let a = 1.0;
            push(g, a, 0., 0., v);
            push(g, -a, 0., 0., v);
            push(g, 0., a, 0., v);
            push(g, 0., -a, 0., v);
            push(g, 0., 0., a, v);
            push(g, 0., 0., -a, v);
        }
        2 => {
            let a = 0.5f64.sqrt();
            for &(x, y, z) in &[
                (0., a, a),
                (0., -a, a),
                (0., a, -a),
                (0., -a, -a),
                (a, 0., a),
                (-a, 0., a),
                (a, 0., -a),
                (-a, 0., -a),
                (a, a, 0.),
                (-a, a, 0.),
                (a, -a, 0.),
                (-a, -a, 0.),
            ] {
                push(g, x, y, z, v);
            }
        }
        3 => {
            let a = (1.0f64 / 3.0).sqrt();
            for s in SIGNS8 {
                push(g, s[0] * a, s[1] * a, s[2] * a, v);
            }
        }
        4 => {
            let b = (1.0 - 2.0 * a * a).sqrt();
            for p in [[a, a, b], [a, b, a], [b, a, a]] {
                for s in SIGNS8 {
                    push(g, s[0] * p[0], s[1] * p[1], s[2] * p[2], v);
                }
            }
        }
        5 => {
            let b = (1.0 - a * a).sqrt();
            for &(x, y, z) in &[
                (a, b, 0.),
                (-a, b, 0.),
                (a, -b, 0.),
                (-a, -b, 0.),
                (b, a, 0.),
                (-b, a, 0.),
                (b, -a, 0.),
                (-b, -a, 0.),
                (a, 0., b),
                (-a, 0., b),
                (a, 0., -b),
                (-a, 0., -b),
                (b, 0., a),
                (-b, 0., a),
                (b, 0., -a),
                (-b, 0., -a),
                (0., a, b),
                (0., -a, b),
                (0., a, -b),
                (0., -a, -b),
                (0., b, a),
                (0., -b, a),
                (0., b, -a),
                (0., -b, -a),
            ] {
                push(g, x, y, z, v);
            }
        }
        6 => {
            let c = (1.0 - a * a - b * b).sqrt();
            for p in [
                [a, b, c],
                [a, c, b],
                [b, a, c],
                [b, c, a],
                [c, a, b],
                [c, b, a],
            ] {
                for s in SIGNS8 {
                    push(g, s[0] * p[0], s[1] * p[1], s[2] * p[2], v);
                }
            }
        }
        _ => unreachable!("invalid Lebedev orbit type {t}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counts_and_normalisation() {
        for &n in LEBEDEV_NPOINTS.iter() {
            let g = lebedev(n).unwrap();
            assert_eq!(g.x.len(), n);
            let s: f64 = g.w.iter().sum();
            // the published tables are accurate to ~2e-14 (the 3470-point set sums to 1 - 1.5e-14)
            assert!((s - 1.0).abs() < 3e-14, "n={n} sum={s}");
            for i in 0..n {
                let r2 = g.x[i] * g.x[i] + g.y[i] * g.y[i] + g.z[i] * g.z[i];
                assert!((r2 - 1.0).abs() < 4e-16 * 4.0, "n={n} |u|^2={r2}");
            }
        }
    }
}
