// SPDX-License-Identifier: LGPL-3.0-or-later
//! Dense LU factorization with partial pivoting for the (symmetric, indefinite) bordered
//! EEQ matrix. Serial and deterministic; O(n^3/3) with n = atoms + 1.

/// LU factors of a square row-major matrix, `P A = L U`.
pub(crate) struct Lu {
    n: usize,
    lu: Vec<f64>,
    piv: Vec<usize>,
}

impl Lu {
    /// Factorize `a` (row-major n x n). Returns `None` for an exactly singular pivot.
    pub(crate) fn new(n: usize, mut a: Vec<f64>) -> Option<Self> {
        assert_eq!(a.len(), n * n);
        let mut piv = vec![0usize; n];
        for k in 0..n {
            let mut p = k;
            let mut best = a[k * n + k].abs();
            for i in k + 1..n {
                let v = a[i * n + k].abs();
                if v > best {
                    best = v;
                    p = i;
                }
            }
            if best == 0.0 {
                return None;
            }
            piv[k] = p;
            if p != k {
                for j in 0..n {
                    a.swap(k * n + j, p * n + j);
                }
            }
            let inv = 1.0 / a[k * n + k];
            let (top, bottom) = a.split_at_mut((k + 1) * n);
            let rowk = &top[k * n..k * n + n];
            for row in bottom.chunks_mut(n) {
                let f = row[k] * inv;
                row[k] = f;
                if f != 0.0 {
                    for j in k + 1..n {
                        row[j] -= f * rowk[j];
                    }
                }
            }
        }
        Some(Self { n, lu: a, piv })
    }

    /// Solve `A x = b` in place.
    pub(crate) fn solve(&self, b: &mut [f64]) {
        let n = self.n;
        for k in 0..n {
            let p = self.piv[k];
            if p != k {
                b.swap(k, p);
            }
        }
        for i in 0..n {
            let mut s = b[i];
            let row = &self.lu[i * n..i * n + i];
            for (j, l) in row.iter().enumerate() {
                s -= l * b[j];
            }
            b[i] = s;
        }
        for i in (0..n).rev() {
            let mut s = b[i];
            for j in i + 1..n {
                s -= self.lu[i * n + j] * b[j];
            }
            b[i] = s / self.lu[i * n + i];
        }
    }

    /// Solve `A X = B` for the columns of a row-major n x m matrix `B`, in place, columns
    /// in parallel (each column is an independent serial solve, so the result does not
    /// depend on the thread count).
    pub(crate) fn solve_columns(&self, m: usize, b: &mut [f64]) {
        use rayon::prelude::*;
        let n = self.n;
        assert_eq!(b.len(), n * m);
        let cols: Vec<Vec<f64>> = (0..m)
            .into_par_iter()
            .map(|c| {
                let mut x: Vec<f64> = (0..n).map(|i| b[i * m + c]).collect();
                self.solve(&mut x);
                x
            })
            .collect();
        for (c, x) in cols.iter().enumerate() {
            for i in 0..n {
                b[i * m + c] = x[i];
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Lu;

    #[test]
    fn solves_indefinite_bordered_system() {
        // [[2, 1, 1], [1, 3, 1], [1, 1, 0]]
        let a = vec![2.0, 1.0, 1.0, 1.0, 3.0, 1.0, 1.0, 1.0, 0.0];
        let lu = Lu::new(3, a.clone()).unwrap();
        let x0 = [0.3, -0.7, 1.1];
        let mut b: Vec<f64> = (0..3)
            .map(|i| (0..3).map(|j| a[i * 3 + j] * x0[j]).sum())
            .collect();
        lu.solve(&mut b);
        for i in 0..3 {
            assert!((b[i] - x0[i]).abs() < 1e-14);
        }
    }
}
