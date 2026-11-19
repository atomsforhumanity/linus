//! DIIS (Pulay) for the device loop. The Fock and error matrices of the subspace stay on the
//! device (`DiisStore`, feature `gpu`); the host keeps only their inner products
//! ([`DiisBook`]), at most `diis_space`^2 numbers, and solves the small DIIS system there
//! as linus's CPU `Diis::extrapolate` does. Which entry leaves a full subspace is the
//! controller's decision ([`super::StepPlan::diis_victim`]), from the entries' squared norms
//! ([`DiisBook::norms`]).

/// The host side of the DIIS subspace: the inner products B_ij = sum_s <e_i,s, e_j,s> of the
/// stored error matrices, in insertion order.
#[derive(Clone, Debug)]
pub struct DiisBook {
    pub max: usize,
    b: Vec<Vec<f64>>,
}

impl DiisBook {
    pub fn new(max: usize) -> Self {
        DiisBook { max, b: Vec::new() }
    }

    pub fn len(&self) -> usize {
        self.b.len()
    }

    pub fn is_empty(&self) -> bool {
        self.b.is_empty()
    }

    /// The squared error norms of the entries, in order.
    pub fn norms(&self) -> Vec<f64> {
        (0..self.len()).map(|i| self.b[i][i]).collect()
    }

    /// Adds an entry: `row` holds its inner products with the current entries (in order) and
    /// `self_dot` its squared norm. Removes the entry `victim` first; a full subspace needs one,
    /// one with room has none.
    pub fn push(&mut self, row: &[f64], self_dot: f64, victim: Option<usize>) {
        assert_eq!(row.len(), self.len(), "one inner product per stored entry");
        assert_eq!(
            victim.is_some(),
            self.len() >= self.max,
            "a victim exactly when the subspace is full"
        );
        let mut row = row.to_vec();
        if let Some(o) = victim {
            self.b.remove(o);
            for r in self.b.iter_mut() {
                r.remove(o);
            }
            row.remove(o);
        }
        for (r, &v) in self.b.iter_mut().zip(&row) {
            r.push(v);
        }
        row.push(self_dot);
        self.b.push(row);
    }

    /// The extrapolation coefficients c_i (F = sum_i c_i F_i): the CPU's solve of the bordered
    /// system [B -1; -1 0] (c, lambda) = (0, -1), B scaled by its largest diagonal element, by
    /// eigendecomposition with eigenvalues below 1e-14 of the largest dropped.
    pub fn coefficients(&self) -> Vec<f64> {
        let n = self.len();
        let m = n + 1;
        let mut b = vec![0.0; m * m];
        for i in 0..n {
            for j in 0..n {
                b[i * m + j] = self.b[i][j];
            }
            b[i * m + n] = -1.0;
            b[n * m + i] = -1.0;
        }
        let s = (0..n).map(|i| b[i * m + i]).fold(0.0, f64::max).max(1e-300);
        for i in 0..n {
            for j in 0..n {
                b[i * m + j] /= s;
            }
        }
        let (w, u) = jacobi_eigh(&b, m);
        let wmax = w.iter().fold(0.0f64, |a, &v| a.max(v.abs()));
        let mut x = vec![0.0; m];
        for k in 0..m {
            if w[k].abs() < 1e-14 * wmax {
                continue;
            }
            // rhs = (0, ..., 0, -1)
            let proj = -u[n * m + k] / w[k];
            for i in 0..m {
                x[i] += proj * u[i * m + k];
            }
        }
        x.truncate(n);
        x
    }
}

/// Eigenvalues (ascending) and eigenvectors of the symmetric m x m matrix `a` (row-major) by
/// cyclic Jacobi rotations: (w, u) with eigenvector k in column k, u[i m + k]. For the small
/// DIIS system only (m <= a few tens); deterministic.
pub fn jacobi_eigh(a: &[f64], m: usize) -> (Vec<f64>, Vec<f64>) {
    assert_eq!(a.len(), m * m);
    let mut a = a.to_vec();
    let mut v = vec![0.0; m * m];
    for i in 0..m {
        v[i * m + i] = 1.0;
    }
    let off = |a: &[f64]| -> f64 {
        let mut s = 0.0;
        for i in 0..m {
            for j in 0..m {
                if i != j {
                    s += a[i * m + j] * a[i * m + j];
                }
            }
        }
        s
    };
    let total: f64 = a.iter().map(|x| x * x).sum();
    for _sweep in 0..100 {
        if off(&a) <= 1e-34 * total || off(&a) == 0.0 {
            break;
        }
        for p in 0..m {
            for q in p + 1..m {
                let apq = a[p * m + q];
                if apq == 0.0 {
                    continue;
                }
                let theta = (a[q * m + q] - a[p * m + p]) / (2.0 * apq);
                let t = if theta.abs() > 1e150 {
                    0.5 / theta
                } else {
                    theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt())
                };
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                // A <- J^T A J with the rotation in the (p, q) plane
                for k in 0..m {
                    let akp = a[k * m + p];
                    let akq = a[k * m + q];
                    a[k * m + p] = c * akp - s * akq;
                    a[k * m + q] = s * akp + c * akq;
                }
                for k in 0..m {
                    let apk = a[p * m + k];
                    let aqk = a[q * m + k];
                    a[p * m + k] = c * apk - s * aqk;
                    a[q * m + k] = s * apk + c * aqk;
                }
                for k in 0..m {
                    let vkp = v[k * m + p];
                    let vkq = v[k * m + q];
                    v[k * m + p] = c * vkp - s * vkq;
                    v[k * m + q] = s * vkp + c * vkq;
                }
            }
        }
    }
    let mut order: Vec<usize> = (0..m).collect();
    order.sort_by(|&i, &j| a[i * m + i].total_cmp(&a[j * m + j]));
    let w: Vec<f64> = order.iter().map(|&k| a[k * m + k]).collect();
    let mut u = vec![0.0; m * m];
    for (c, &k) in order.iter().enumerate() {
        for i in 0..m {
            u[i * m + c] = v[i * m + k];
        }
    }
    (w, u)
}

#[cfg(feature = "gpu")]
pub use device::DiisStore;

#[cfg(feature = "gpu")]
mod device {
    use crate::error::GpuError;
    use crate::{DeviceBuf, Gpu};

    /// One stored entry: the Fock and error matrices of each spin channel.
    pub struct Entry {
        pub f: Vec<DeviceBuf<f64>>,
        pub e: Vec<DeviceBuf<f64>>,
    }

    /// The device side of the DIIS subspace: the entries' matrices, in the order of
    /// [`super::DiisBook`]. A removed entry's buffers are reused by the next push.
    #[derive(Default)]
    pub struct DiisStore {
        pub entries: Vec<Entry>,
        spare: Vec<Entry>,
    }

    impl DiisStore {
        /// Removes entry `out` (if any) and appends copies of `f` and `e`.
        pub fn push(
            &mut self,
            gpu: &Gpu,
            out: Option<usize>,
            f: &[&DeviceBuf<f64>],
            e: &[&DeviceBuf<f64>],
        ) -> Result<(), GpuError> {
            if let Some(o) = out {
                let old = self.entries.remove(o);
                self.spare.push(old);
            }
            let mut ent = match self.spare.pop() {
                Some(x) => x,
                None => Entry {
                    f: f.iter()
                        .map(|m| gpu.alloc_zeros(m.len()))
                        .collect::<Result<_, _>>()?,
                    e: e.iter()
                        .map(|m| gpu.alloc_zeros(m.len()))
                        .collect::<Result<_, _>>()?,
                },
            };
            for (dst, src) in ent.f.iter_mut().zip(f) {
                gpu.copy(src, dst)?;
            }
            for (dst, src) in ent.e.iter_mut().zip(e) {
                gpu.copy(src, dst)?;
            }
            self.entries.push(ent);
            Ok(())
        }

        pub fn len(&self) -> usize {
            self.entries.len()
        }

        pub fn is_empty(&self) -> bool {
            self.entries.is_empty()
        }

        /// Device bytes held (entries and spares).
        pub fn bytes(&self) -> usize {
            self.entries
                .iter()
                .chain(&self.spare)
                .flat_map(|x| x.f.iter().chain(&x.e))
                .map(|b| 8 * b.len())
                .sum()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sym(m: usize, seed: f64) -> Vec<f64> {
        let mut a = vec![0.0; m * m];
        for i in 0..m {
            for j in 0..=i {
                let v = ((i * 7 + j * 3) as f64 * seed).sin() + if i == j { 2.0 } else { 0.0 };
                a[i * m + j] = v;
                a[j * m + i] = v;
            }
        }
        a
    }

    #[test]
    fn jacobi_diagonalizes() {
        for m in [1, 2, 5, 11] {
            let a = sym(m, 0.37);
            let (w, u) = jacobi_eigh(&a, m);
            assert!(w.windows(2).all(|p| p[0] <= p[1]));
            for k in 0..m {
                // A u_k = w_k u_k, |u_k| = 1
                let mut r = 0.0f64;
                let mut nrm = 0.0;
                for i in 0..m {
                    let au: f64 = (0..m).map(|j| a[i * m + j] * u[j * m + k]).sum();
                    r = r.max((au - w[k] * u[i * m + k]).abs());
                    nrm += u[i * m + k] * u[i * m + k];
                }
                assert!(r < 1e-13, "m {m} k {k}: residual {r}");
                assert!((nrm - 1.0).abs() < 1e-14);
            }
        }
    }

    /// The coefficients minimize |sum c_i e_i| with sum c_i = 1: for errors with inner
    /// products B, B c = lambda 1, and the sum is one.
    #[test]
    fn coefficients_solve_the_diis_system() {
        let mut book = DiisBook::new(10);
        // error vectors e_i in R^6
        let e: Vec<Vec<f64>> = (0..4)
            .map(|i| {
                (0..6)
                    .map(|k| {
                        (1.3 * (i + 1) as f64 * ((k + 1) as f64).powf(1.5)).sin() * 0.1f64.powi(i)
                    })
                    .collect()
            })
            .collect();
        let d = |a: &[f64], b: &[f64]| a.iter().zip(b).map(|(x, y)| x * y).sum::<f64>();
        for i in 0..e.len() {
            let row: Vec<f64> = (0..i).map(|j| d(&e[i], &e[j])).collect();
            book.push(&row, d(&e[i], &e[i]), None);
        }
        let c = book.coefficients();
        assert!((c.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        let bc: Vec<f64> = (0..4)
            .map(|i| (0..4).map(|j| d(&e[i], &e[j]) * c[j]).sum())
            .collect();
        for v in &bc {
            // B entries are O(1): equal to round-off on that scale
            assert!((v - bc[0]).abs() < 1e-13, "{bc:?}");
        }
    }

    #[test]
    fn a_push_removes_its_victim() {
        let mut b = DiisBook::new(3);
        for (i, n) in [4.0, 9.0, 1.0].iter().enumerate() {
            b.push(&vec![0.0; i], *n, None);
        }
        assert_eq!(b.norms(), vec![4.0, 9.0, 1.0]);
        b.push(&[0.1, 0.2, 0.3], 2.0, Some(1));
        // entries now 4, 1, 2; the new row lost the removed entry's product
        assert_eq!(b.norms(), vec![4.0, 1.0, 2.0]);
        assert_eq!(b.b[2], vec![0.1, 0.3, 2.0]);
        assert_eq!(b.b[0][2], 0.1);
    }

    #[test]
    #[should_panic(expected = "a victim exactly when the subspace is full")]
    fn a_full_subspace_needs_a_victim() {
        let mut b = DiisBook::new(1);
        b.push(&[], 1.0, None);
        b.push(&[0.0], 1.0, None);
    }
}
