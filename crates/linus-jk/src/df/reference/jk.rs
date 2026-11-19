//! Reference DF J/K (tests only): the three-index tensor B in memory, K from (signed) factors of
//! the density. The RI J/K of the SCF ([`crate::df::DfJk`]) and of the Hessian
//! response ([`crate::df::response::DfResponseJk`]) are checked against it.

use faer::{Mat, Par};
use linus_base::linalg::{mat_from_rowmajor, mm};
use linus_base::terms::{JkBuilder, JkOutput, JkRequest};
use linus_ints::{BasisSet, PairOptions, PairSet};

/// Density-fitted J and K with the Coulomb metric: B^Q_mn = sum_P (mn|P) [L^-1]_QP, V = L L^T.
/// B is held in memory as naux blocks of nbf x nbf.
pub struct DenseDfJk {
    nbf: usize,
    naux: usize,
    /// column-major (nbf*nbf) x naux; column Q is the symmetric matrix B^Q
    b: Vec<f64>,
}

impl DenseDfJk {
    pub fn new(basis: &BasisSet, pairs: &PairSet, aux: &BasisSet, par: Par) -> Self {
        let nbf = basis.nbf;
        let naux = aux.nbf;
        let apairs = PairSet::aux(aux, PairOptions::default());
        let v = mat_from_rowmajor(naux, naux, &linus_ints::reference::eri_2c(aux, &apairs));
        // (P|mn) as row-major naux x nbf^2; B^T = L^{-1} A^T is solved in place on it, so the
        // 3-index tensor is held once (row-major naux x nbf^2 = column-major nbf^2 x naux).
        let mut b = linus_ints::reference::eri_3c(basis, pairs, aux, &apairs);
        let l = linus_base::linalg::cholesky(v.as_ref(), par)
            .expect("auxiliary metric not positive definite");
        {
            let x = faer::MatMut::from_row_major_slice_mut(&mut b, naux, nbf * nbf);
            linus_base::linalg::solve_lower_in_place(l.as_ref(), x, par);
        }
        DenseDfJk { nbf, naux, b }
    }
}

impl DenseDfJk {
    fn k_from_factor(&self, c: &Mat<f64>, par: Par) -> Mat<f64> {
        let (n, naux) = (self.nbf, self.naux);
        let nocc = c.ncols();
        if nocc == 0 {
            return Mat::zeros(n, n);
        }
        let bwide = faer::MatRef::from_column_major_slice(&self.b, n, n * naux);
        let mut zbuf = vec![0.0; n * naux * nocc];
        {
            let zm = faer::MatMut::from_column_major_slice_mut(&mut zbuf, n * naux, nocc);
            linus_base::linalg::matmul(
                zm,
                faer::Accum::Replace,
                bwide.transpose(),
                c.as_ref(),
                1.0,
                par,
            );
        }
        let w = faer::MatRef::from_column_major_slice(&zbuf, n, naux * nocc);
        mm(w, w.transpose(), par)
    }
}

impl JkBuilder for DenseDfJk {
    fn name(&self) -> String {
        format!("df(naux={})", self.naux)
    }
    fn compute(
        &mut self,
        req: &JkRequest,
        par: Par,
    ) -> Result<JkOutput, linus_base::error::LinusError> {
        let (n, naux) = (self.nbf, self.naux);
        let mut out = JkOutput::default();
        let bmat = faer::MatRef::from_column_major_slice(&self.b, n * n, naux);
        for (di, d) in req.dens.iter().enumerate() {
            if req.want_j {
                let dv = Mat::from_fn(n * n, 1, |x, _| d[(x % n, x / n)]);
                let gamma = mm(bmat.transpose(), dv.as_ref(), par); // naux x 1
                let jv = mm(bmat, gamma.as_ref(), par); // n^2 x 1
                out.j.push(Mat::from_fn(n, n, |i, j| jv[(j * n + i, 0)]));
            }
            if req.want_k {
                // factor C (n x nocc): K = sum_Q (B^Q C)(B^Q C)^T
                let (c, cneg): (Mat<f64>, Mat<f64>) = match req.factors.and_then(|f| f.symmetric())
                {
                    Some(f) => (f[di].clone(), Mat::zeros(n, 0)),
                    None => factor_density_signed(d.as_ref(), par),
                };
                if cneg.ncols() > 0 {
                    // indefinite density: K[D] = K[C+ C+^T] - K[C- C-^T]
                    let kp = self.k_from_factor(&c, par);
                    let km = self.k_from_factor(&cneg, par);
                    out.k.push(kp - km);
                    continue;
                }
                let nocc = c.ncols();
                if nocc == 0 {
                    out.k.push(Mat::zeros(n, n));
                    continue;
                }
                // Bt as (n*naux) x n: rows (Q, nu), since each B^Q is symmetric the column-major
                // storage of b reinterpreted as n x (n*naux) gives [B^1 B^2 ...].
                let bwide = faer::MatRef::from_column_major_slice(&self.b, n, n * naux);
                // Z = Bwide^T C : (n*naux) x nocc, Z[(Q,nu), i]
                let mut zbuf = vec![0.0; n * naux * nocc];
                {
                    let zm = faer::MatMut::from_column_major_slice_mut(&mut zbuf, n * naux, nocc);
                    linus_base::linalg::matmul(
                        zm,
                        faer::Accum::Replace,
                        bwide.transpose(),
                        c.as_ref(),
                        1.0,
                        par,
                    );
                }
                // reinterpret Z as W: n x (naux*nocc); K = W W^T
                let w = faer::MatRef::from_column_major_slice(&zbuf, n, naux * nocc);
                let k = mm(w, w.transpose(), par);
                out.k.push(k);
            }
        }
        Ok(out)
    }
}

/// Factor a symmetric density as C+ C+^T - C- C-^T (eigendecomposition; eigenvalues with
/// |w| below 1e-14 relative to the largest are dropped). Response densities are indefinite.
pub(crate) fn factor_density_signed(d: faer::MatRef<'_, f64>, par: Par) -> (Mat<f64>, Mat<f64>) {
    let (w, u) = linus_base::linalg::eigh(d, par);
    let wmax = w.iter().fold(0.0f64, |a, &b| a.max(b.abs()));
    let cut = 1e-14 * wmax.max(1e-300);
    let pos: Vec<usize> = (0..w.len()).filter(|&i| w[i] > cut).collect();
    let neg: Vec<usize> = (0..w.len()).filter(|&i| w[i] < -cut).collect();
    (
        Mat::from_fn(d.nrows(), pos.len(), |i, k| {
            u[(i, pos[k])] * w[pos[k]].sqrt()
        }),
        Mat::from_fn(d.nrows(), neg.len(), |i, k| {
            u[(i, neg[k])] * (-w[neg[k]]).sqrt()
        }),
    )
}
