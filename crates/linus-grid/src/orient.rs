//! Orientation of the angular (Lebedev) grids.

use serde::{Deserialize, Serialize};

/// How the Lebedev unit vectors are rotated before being placed on the atoms.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum Orientation {
    /// Lab frame (PySCF, ORCA default).
    Identity,
    /// Explicit rotation applied to every unit vector: u' = M u (row-major). Use this to
    /// reproduce Psi4 exactly with its `DFTGrid.orientation()` matrix.
    Matrix([[f64; 3]; 3]),
    /// Principal axes of the nuclear-charge distribution (makes the grid, and hence energies,
    /// invariant to rigid rotations of non-degenerate molecules). Eigenvector signs are fixed
    /// deterministically; for degenerate moments the orientation is not unique.
    Standard,
}

/// Rotation matrix for an orientation (row-major, applied as u' = M u).
pub fn rotation_matrix(o: &Orientation, z: &[u32], xyz: &[[f64; 3]]) -> [[f64; 3]; 3] {
    match o {
        Orientation::Identity => [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        Orientation::Matrix(m) => *m,
        Orientation::Standard => standard_orientation(z, xyz),
    }
}

fn standard_orientation(z: &[u32], xyz: &[[f64; 3]]) -> [[f64; 3]; 3] {
    let q: f64 = z.iter().map(|&v| v as f64).sum();
    if xyz.len() < 2 || q == 0.0 {
        return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    }
    let mut c = [0.0; 3];
    for (zi, p) in z.iter().zip(xyz) {
        for k in 0..3 {
            c[k] += *zi as f64 * p[k];
        }
    }
    for v in c.iter_mut() {
        *v /= q;
    }
    let mut m = [[0.0; 3]; 3];
    for (zi, p) in z.iter().zip(xyz) {
        let d = [p[0] - c[0], p[1] - c[1], p[2] - c[2]];
        for i in 0..3 {
            for j in 0..3 {
                m[i][j] += *zi as f64 * d[i] * d[j];
            }
        }
    }
    let (_, v) = jacobi3(m);
    // rows = eigenvectors (ascending eigenvalues); deterministic signs, right-handed
    let mut r = [[0.0; 3]; 3];
    for k in 0..3 {
        let mut e = [v[0][k], v[1][k], v[2][k]];
        let imax = (0..3)
            .max_by(|&a, &b| e[a].abs().partial_cmp(&e[b].abs()).unwrap())
            .unwrap();
        if e[imax] < 0.0 {
            for x in e.iter_mut() {
                *x = -*x;
            }
        }
        r[k] = e;
    }
    let cr = [
        r[0][1] * r[1][2] - r[0][2] * r[1][1],
        r[0][2] * r[1][0] - r[0][0] * r[1][2],
        r[0][0] * r[1][1] - r[0][1] * r[1][0],
    ];
    if cr[0] * r[2][0] + cr[1] * r[2][1] + cr[2] * r[2][2] < 0.0 {
        for x in r[2].iter_mut() {
            *x = -*x;
        }
    }
    // u' = M u places the grid's x-axis along the first principal axis: M = R^T
    let mut mt = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            mt[i][j] = r[j][i];
        }
    }
    mt
}

/// Jacobi eigen-decomposition of a symmetric 3x3 matrix: (eigenvalues ascending, eigenvectors
/// as columns).
#[allow(clippy::needless_range_loop, reason = "indexes several arrays in step")]
fn jacobi3(mut a: [[f64; 3]; 3]) -> ([f64; 3], [[f64; 3]; 3]) {
    let mut v = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    for _ in 0..100 {
        let off = a[0][1].abs() + a[0][2].abs() + a[1][2].abs();
        if off < 1e-300 {
            break;
        }
        for (p, q) in [(0, 1), (0, 2), (1, 2)] {
            if a[p][q].abs() < 1e-300 {
                continue;
            }
            let theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q]);
            let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
            let t = if theta == 0.0 { 1.0 } else { t };
            let c = 1.0 / (t * t + 1.0).sqrt();
            let s = t * c;
            for k in 0..3 {
                let akp = a[k][p];
                let akq = a[k][q];
                a[k][p] = c * akp - s * akq;
                a[k][q] = s * akp + c * akq;
            }
            for k in 0..3 {
                let apk = a[p][k];
                let aqk = a[q][k];
                a[p][k] = c * apk - s * aqk;
                a[q][k] = s * apk + c * aqk;
            }
            for k in 0..3 {
                let vkp = v[k][p];
                let vkq = v[k][q];
                v[k][p] = c * vkp - s * vkq;
                v[k][q] = s * vkp + c * vkq;
            }
        }
    }
    let mut idx = [0usize, 1, 2];
    idx.sort_by(|&i, &j| a[i][i].partial_cmp(&a[j][j]).unwrap());
    let ev = [a[idx[0]][idx[0]], a[idx[1]][idx[1]], a[idx[2]][idx[2]]];
    let mut vs = [[0.0; 3]; 3];
    for (col, &i) in idx.iter().enumerate() {
        for k in 0..3 {
            vs[k][col] = v[k][i];
        }
    }
    (ev, vs)
}
