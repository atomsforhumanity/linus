//! ADIIS (Hu & Yang, JCP 132, 054109, 2010) while the orbital gradient is large
//! ([`crate::ScfAlgorithm::adiis`]): the history of densities and Fock matrices, and the
//! coefficients of the augmented-Roothaan-Hall energy's minimum on the simplex.

use faer::Mat;
use linus_base::linalg::dot;

/// The ADIIS history: the densities and Fock matrices of the last iterations (per channel), at
/// most max(`diis_space`, 2).
pub(crate) struct Adiis {
    max: usize,
    pub(crate) dens: Vec<Vec<Mat<f64>>>,
    pub(crate) focks: Vec<Vec<Mat<f64>>>,
}

impl Adiis {
    pub(crate) fn new(diis_space: usize) -> Self {
        Adiis {
            max: diis_space.max(2),
            dens: Vec::new(),
            focks: Vec::new(),
        }
    }

    /// Adds an iteration's densities and Fock matrices (the oldest leaves a full history).
    pub(crate) fn push(&mut self, d: Vec<Mat<f64>>, f: Vec<Mat<f64>>) {
        self.dens.push(d);
        self.focks.push(f);
        if self.dens.len() > self.max {
            self.dens.remove(0);
            self.focks.remove(0);
        }
    }

    pub(crate) fn len(&self) -> usize {
        self.dens.len()
    }

    pub(crate) fn clear(&mut self) {
        self.dens.clear();
        self.focks.clear();
    }

    /// Adds dF (per channel) to every stored Fock matrix: a changed Fock operator.
    pub(crate) fn shift(&mut self, df: &[Mat<f64>]) {
        for f in self.focks.iter_mut() {
            for (m, d) in f.iter_mut().zip(df) {
                *m += d;
            }
        }
    }

    /// The extrapolated Fock matrices sum_i c_i F_i (per channel) with the coefficients of
    /// [`adiis_coefficients`] (`weight`: 2 for a closed shell's one channel, else 1).
    pub(crate) fn extrapolate(&self, weight: f64) -> Vec<Mat<f64>> {
        let c = adiis_coefficients(&self.dens, &self.focks, weight);
        let mut fx: Vec<Mat<f64>> = self.focks[0]
            .iter()
            .map(|m| Mat::zeros(m.nrows(), m.ncols()))
            .collect();
        for (ci, fs) in c.iter().zip(&self.focks) {
            for (acc, f) in fx.iter_mut().zip(fs) {
                *acc += f * *ci;
            }
        }
        fx
    }
}

/// ADIIS (Hu & Yang, JCP 132, 054109, 2010): minimize the augmented-Roothaan-Hall energy
///   f(c) = sum_i c_i d_i + 1/2 sum_ij c_i c_j M_ij,  c_i >= 0, sum c_i = 1,
/// with d_i = <D_i - D_n, F_n>, M_ij = <D_i - D_n, F_j - F_n> summed over spins.
/// Returns the coefficients (simplex), found by exponentiated-gradient descent.
fn adiis_coefficients(dens: &[Vec<Mat<f64>>], focks: &[Vec<Mat<f64>>], weight: f64) -> Vec<f64> {
    let n = dens.len();
    let last = n - 1;
    let nsp = dens[0].len();
    let mut d = vec![0.0; n];
    let mut m = vec![0.0; n * n];
    for i in 0..n {
        for sp in 0..nsp {
            let ddi = &dens[i][sp] - &dens[last][sp];
            d[i] += weight * dot(ddi.as_ref(), focks[last][sp].as_ref());
            for j in 0..n {
                let dfj = &focks[j][sp] - &focks[last][sp];
                m[i * n + j] += weight * dot(ddi.as_ref(), dfj.as_ref());
            }
        }
    }
    for i in 0..n {
        for j in 0..i {
            let v = 0.5 * (m[i * n + j] + m[j * n + i]);
            m[i * n + j] = v;
            m[j * n + i] = v;
        }
    }
    let f = |c: &[f64]| -> f64 {
        let mut v = 0.0;
        for i in 0..n {
            v += c[i] * d[i];
            for j in 0..n {
                v += 0.5 * c[i] * c[j] * m[i * n + j];
            }
        }
        v
    };
    // start from the latest point, mirror descent with backtracking
    let mut c = vec![0.1 / (n as f64); n];
    c[last] += 0.9;
    let s: f64 = c.iter().sum();
    c.iter_mut().for_each(|x| *x /= s);
    let mut fc = f(&c);
    let mut step = 1.0;
    for _ in 0..500 {
        let g: Vec<f64> = (0..n)
            .map(|i| d[i] + (0..n).map(|j| m[i * n + j] * c[j]).sum::<f64>())
            .collect();
        let gmax = g.iter().fold(0.0f64, |a, &b| a.max(b.abs())).max(1e-300);
        let mut improved = false;
        for _ in 0..30 {
            let mut t: Vec<f64> = (0..n).map(|i| c[i] * (-step * g[i] / gmax).exp()).collect();
            let s: f64 = t.iter().sum();
            t.iter_mut().for_each(|x| *x /= s);
            let ft = f(&t);
            if ft < fc - 1e-16 * fc.abs() {
                c = t;
                fc = ft;
                improved = true;
                step *= 1.5;
                break;
            }
            step *= 0.5;
        }
        if !improved {
            break;
        }
    }
    c
}
