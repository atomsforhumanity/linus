//! Second derivatives of the PCM energy with respect to the nuclei at fixed density, the
//! fixed-density derivatives of the solvent charges, and the multi-vector helpers they need.
//!
//! With phi the total potential at the points, Q = K^-1 R, q = Q phi, y = K^-T phi and
//! q_sym = Q_sym phi (Q_sym = (Q + Q^T)/2), the fixed-density energy E = 1/2 phi^T Q phi has
//!
//!   d2E/dXdY = dphi_X . Q_sym dphi_Y + dphi_X . w_Y + dphi_Y . w_X + q_sym . d2phi_XY
//!              + 1/2 phi^T d2Q_XY phi,
//!   w_X = (dQ_sym/dX) phi = 1/2 [K^-1 (dR phi - dK q) + dR^T y - R^T K^-T (dK^T y)],
//!   1/2 phi^T d2Q phi = 1/2 [-(dK_Y^T y).(dQ_X phi) - (dK_X^T y).(dQ_Y phi)
//!                            + y^T d2R_XY phi - y^T d2K_XY q],
//!
//! and the charges change at fixed density by dq_sym/dX = Q_sym dphi_X + w_X. dK and d2K are
//! assembled from the derivatives of S, D and the areas A (through the switching factors F):
//! derivative-times-vector arrays [3N][npts] (`ds_mv`, `dd_mv`, `ddt_mv`, `switch_deriv`) and
//! second-derivative bilinear forms (3N x 3N: `d2s`, `d2d`, `d2f`). A point moves rigidly with
//! the atom of its sphere, so the matrix elements depend on the atoms through the point
//! separations (S, D) and the point-to-sphere-centre distances (F).
//!
//! Arrays [3N][npts] are stored row-major by perturbation X (index X * npts + i); as faer
//! matrices they are npts x 3N column-major.

use super::pcm::{coul_erf3, dense_par, dot3, norm, sub, zeta_ij, Charges, Factor, PcmOperator};
use super::surface::SwitchTerm;
use super::PcmModel;
use faer::linalg::matmul::matmul;
use faer::linalg::triangular_solve as tri;
use faer::{Accum, Mat, MatRef};
use linus_num::det::{add_into, nchunk_for, par_fold_det, DEFAULT_BUDGET};
use rayon::prelude::*;
use std::f64::consts::PI;

type M3 = [[f64; 3]; 3];

#[inline]
fn outer(a: &[f64; 3], b: &[f64; 3]) -> M3 {
    let mut m = [[0.0; 3]; 3];
    for k in 0..3 {
        for l in 0..3 {
            m[k][l] = a[k] * b[l];
        }
    }
    m
}

/// Add c M to the (x, y) atom block of an n3 x n3 matrix.
#[inline]
fn add_block(h: &mut [f64], n3: usize, x: usize, y: usize, c: f64, m: &M3) {
    for k in 0..3 {
        for l in 0..3 {
            h[(3 * x + k) * n3 + 3 * y + l] += c * m[k][l];
        }
    }
}

/// Add c M at the four blocks of a function of r_a - r_b: (a,a) +, (b,b) +, (a,b) -, (b,a) -.
#[inline]
fn add_pair_blocks(h: &mut [f64], n3: usize, a: usize, b: usize, c: f64, m: &M3) {
    if a == b {
        return;
    }
    add_block(h, n3, a, a, c, m);
    add_block(h, n3, b, b, c, m);
    add_block(h, n3, a, b, -c, m);
    // the (b, a) block is the transpose of (a, b); the M here are symmetric or handled by callers
    let mut mt = [[0.0; 3]; 3];
    for k in 0..3 {
        for l in 0..3 {
            mt[k][l] = m[l][k];
        }
    }
    add_block(h, n3, b, a, -c, &mt);
}

/// Hessian of s(|u|) with respect to u: s'' u^ u^T + (s'/r)(I - u^ u^T).
#[inline]
fn radial_hess(u: &[f64; 3], r: f64, s1: f64, s2: f64) -> M3 {
    let mut m = [[0.0; 3]; 3];
    for k in 0..3 {
        for l in 0..3 {
            let uu = u[k] * u[l] / (r * r);
            m[k][l] = s2 * uu + s1 / r * (if k == l { 1.0 } else { 0.0 } - uu);
        }
    }
    m
}

/// Gradient and Hessian of D_ij = g(r) (u . n), g = -S'(r)/r, with respect to u = r_i - r_j.
#[inline]
fn d_derivs(zij: f64, u: &[f64; 3], n: &[f64; 3]) -> ([f64; 3], M3) {
    let r = norm(u);
    let (_, s1, s2, s3) = coul_erf3(zij, r);
    let g = -s1 / r;
    let g1 = -s2 / r + s1 / (r * r);
    let g2 = -s3 / r + 2.0 * s2 / (r * r) - 2.0 * s1 / (r * r * r);
    let un = dot3(u, n);
    let uh = [u[0] / r, u[1] / r, u[2] / r];
    let grad = [
        g1 * uh[0] * un + g * n[0],
        g1 * uh[1] * un + g * n[1],
        g1 * uh[2] * un + g * n[2],
    ];
    let mut h = [[0.0; 3]; 3];
    for k in 0..3 {
        for l in 0..3 {
            let uu = uh[k] * uh[l];
            h[k][l] = g2 * un * uu
                + g1 / r * un * (if k == l { 1.0 } else { 0.0 } - uu)
                + g1 * (uh[k] * n[l] + n[k] * uh[l]);
        }
    }
    (grad, h)
}

impl PcmOperator {
    /// Switching terms of every point (see [`super::surface::Surface::switch_terms`]).
    pub(crate) fn all_switch_terms(&self, positions: &[[f64; 3]]) -> Vec<Vec<SwitchTerm>> {
        (0..self.surf.len())
            .into_par_iter()
            .map(|i| self.surf.switch_terms(positions, i))
            .collect()
    }

    /// dF_i/dX for all perturbations X: [3N][npts].
    pub(crate) fn switch_deriv(&self, natom: usize, terms: &[Vec<SwitchTerm>]) -> Vec<f64> {
        let n = self.surf.len();
        let mut out = vec![0.0; 3 * natom * n];
        for (i, ts) in terms.iter().enumerate() {
            let own = self.surf.owner[i];
            let f = self.surf.switch[i];
            for t in ts {
                let c = f * t.df / t.f;
                for k in 0..3 {
                    out[(3 * own + k) * n + i] += c * t.u[k];
                    out[(3 * t.atom + k) * n + i] -= c * t.u[k];
                }
            }
        }
        out
    }

    /// Columns (3N each) of per-point derivative vectors into an X-major [3N][npts] array.
    fn assemble(cols: Vec<Vec<f64>>, n3: usize) -> Vec<f64> {
        let n = cols.len();
        let mut out = vec![0.0; n3 * n];
        for (i, c) in cols.iter().enumerate() {
            for x in 0..n3 {
                out[x * n + i] = c[x];
            }
        }
        out
    }

    /// (dS/dX v)_i for all X: [3N][npts]; `df` from [`Self::switch_deriv`].
    pub(crate) fn ds_mv(&self, natom: usize, v: &[f64], df: &[f64]) -> Vec<f64> {
        let surf = &self.surf;
        let pts = &surf.pts;
        let n = surf.len();
        let n3 = 3 * natom;
        let cols: Vec<Vec<f64>> = (0..n)
            .into_par_iter()
            .map(|i| {
                let mut col = vec![0.0; n3];
                let oi = surf.owner[i];
                for j in 0..n {
                    if j == i || v[j] == 0.0 {
                        continue;
                    }
                    let u = sub(&pts[i].r, &pts[j].r);
                    let r = norm(&u);
                    let s1 = coul_erf3(zeta_ij(pts[i].zeta, pts[j].zeta), r).1;
                    let c = s1 / r * v[j];
                    let oj = surf.owner[j];
                    for k in 0..3 {
                        col[3 * oi + k] += c * u[k];
                        col[3 * oj + k] -= c * u[k];
                    }
                }
                // diagonal S_ii = zeta_i sqrt(2/pi) / F_i
                let c = -pts[i].zeta * (2.0 / PI).sqrt() / (surf.switch[i] * surf.switch[i]) * v[i];
                for (x, cx) in col.iter_mut().enumerate() {
                    *cx += c * df[x * n + i];
                }
                col
            })
            .collect();
        Self::assemble(cols, n3)
    }

    /// (dD/dX v)_i = sum_j dD_ij/dX v_j for all X.
    pub(crate) fn dd_mv(&self, natom: usize, v: &[f64]) -> Vec<f64> {
        let surf = &self.surf;
        let pts = &surf.pts;
        let n = surf.len();
        let n3 = 3 * natom;
        let cols: Vec<Vec<f64>> = (0..n)
            .into_par_iter()
            .map(|i| {
                let mut col = vec![0.0; n3];
                let oi = surf.owner[i];
                for j in 0..n {
                    if j == i || v[j] == 0.0 {
                        continue;
                    }
                    let u = sub(&pts[i].r, &pts[j].r);
                    let (g, _) = d_derivs(zeta_ij(pts[i].zeta, pts[j].zeta), &u, &surf.normal[j]);
                    let oj = surf.owner[j];
                    for k in 0..3 {
                        col[3 * oi + k] += v[j] * g[k];
                        col[3 * oj + k] -= v[j] * g[k];
                    }
                }
                col
            })
            .collect();
        Self::assemble(cols, n3)
    }

    /// (dD^T/dX v)_j = sum_i dD_ij/dX v_i for all X.
    pub(crate) fn ddt_mv(&self, natom: usize, v: &[f64]) -> Vec<f64> {
        let surf = &self.surf;
        let pts = &surf.pts;
        let n = surf.len();
        let n3 = 3 * natom;
        let cols: Vec<Vec<f64>> = (0..n)
            .into_par_iter()
            .map(|j| {
                let mut col = vec![0.0; n3];
                let oj = surf.owner[j];
                for i in 0..n {
                    if j == i || v[i] == 0.0 {
                        continue;
                    }
                    let u = sub(&pts[i].r, &pts[j].r);
                    let (g, _) = d_derivs(zeta_ij(pts[i].zeta, pts[j].zeta), &u, &surf.normal[j]);
                    let oi = surf.owner[i];
                    for k in 0..3 {
                        col[3 * oi + k] += v[i] * g[k];
                        col[3 * oj + k] -= v[i] * g[k];
                    }
                }
                col
            })
            .collect();
        Self::assemble(cols, n3)
    }

    /// Gradient (sparse: (atom, 3-vector)) and Hessian blocks ((atom, atom, 3x3)) of F_i.
    #[allow(clippy::needless_range_loop, reason = "indexes several arrays in step")]
    #[allow(clippy::type_complexity, reason = "a one-off tuple of buffers")]
    fn switch_point_derivs(
        &self,
        i: usize,
        ts: &[SwitchTerm],
    ) -> (Vec<(usize, [f64; 3])>, Vec<(usize, usize, M3)>) {
        let own = self.surf.owner[i];
        let f = self.surf.switch[i];
        // grad of ln F: sum_K lambda_K (+ at own, - at K)
        let mut grad: Vec<(usize, [f64; 3])> = Vec::new();
        let mut addg = |a: usize, v: [f64; 3], c: f64| {
            if let Some(e) = grad.iter_mut().find(|e| e.0 == a) {
                for k in 0..3 {
                    e.1[k] += c * v[k];
                }
            } else {
                grad.push((a, [c * v[0], c * v[1], c * v[2]]));
            }
        };
        for t in ts {
            let lam = [
                t.df / t.f * t.u[0],
                t.df / t.f * t.u[1],
                t.df / t.f * t.u[2],
            ];
            addg(own, lam, 1.0);
            addg(t.atom, lam, -1.0);
        }
        // d2F = F [g g^T + sum_K Lambda_K]
        let mut hess: Vec<(usize, usize, M3)> = Vec::new();
        for (a, ga) in &grad {
            for (b, gb) in &grad {
                let m = outer(ga, gb);
                let mut mf = [[0.0; 3]; 3];
                for k in 0..3 {
                    for l in 0..3 {
                        mf[k][l] = f * m[k][l];
                    }
                }
                hess.push((*a, *b, mf));
            }
        }
        for t in ts {
            let rho = [t.u[0] * t.r, t.u[1] * t.r, t.u[2] * t.r];
            // Lambda = (f''/f - (f'/f)^2) u u^T + f'/(f r) (I - u u^T)
            let a2 = t.d2f / t.f - (t.df / t.f).powi(2);
            let a1 = t.df / t.f;
            let h = radial_hess(&rho, t.r, a1, a2);
            let mut hf = [[0.0; 3]; 3];
            for k in 0..3 {
                for l in 0..3 {
                    hf[k][l] = f * h[k][l];
                }
            }
            if own != t.atom {
                hess.push((own, own, hf));
                hess.push((t.atom, t.atom, hf));
                let mut neg = hf;
                neg.iter_mut()
                    .for_each(|row| row.iter_mut().for_each(|x| *x = -*x));
                hess.push((own, t.atom, neg));
                hess.push((t.atom, own, neg));
            }
        }
        let gradf = grad
            .into_iter()
            .map(|(a, g)| (a, [f * g[0], f * g[1], f * g[2]]))
            .collect();
        (gradf, hess)
    }

    /// sum_i c_i d2F_i/dXdY (3N x 3N).
    pub(crate) fn d2f(&self, natom: usize, c: &[f64], terms: &[Vec<SwitchTerm>]) -> Vec<f64> {
        let n3 = 3 * natom;
        let n = self.surf.len();
        par_fold_det(
            n,
            nchunk_for(n, n3 * n3, DEFAULT_BUDGET),
            || vec![0.0; n3 * n3],
            |h, i| {
                if c[i] == 0.0 || terms[i].is_empty() {
                    return;
                }
                let (_, hb) = self.switch_point_derivs(i, &terms[i]);
                for (a, b, m) in hb {
                    add_block(h, n3, a, b, c[i], &m);
                }
            },
            |a, b| add_into(a, &b),
        )
    }

    /// sum_ij a_i b_j d2S_ij/dXdY (3N x 3N), diagonal S_ii = zeta_i sqrt(2/pi) / F_i included.
    pub(crate) fn d2s(
        &self,
        natom: usize,
        a: &[f64],
        b: &[f64],
        terms: &[Vec<SwitchTerm>],
    ) -> Vec<f64> {
        let surf = &self.surf;
        let pts = &surf.pts;
        let n = surf.len();
        let n3 = 3 * natom;
        let mut h = par_fold_det(
            n,
            nchunk_for(n, n3 * n3, DEFAULT_BUDGET),
            || vec![0.0; n3 * n3],
            |h, i| {
                let oi = surf.owner[i];
                for j in (i + 1)..n {
                    let c = a[i] * b[j] + a[j] * b[i];
                    let oj = surf.owner[j];
                    if c == 0.0 || oi == oj {
                        continue;
                    }
                    let u = sub(&pts[i].r, &pts[j].r);
                    let r = norm(&u);
                    let (_, s1, s2, _) = coul_erf3(zeta_ij(pts[i].zeta, pts[j].zeta), r);
                    add_pair_blocks(h, n3, oi, oj, c, &radial_hess(&u, r, s1, s2));
                }
            },
            |x, y| add_into(x, &y),
        );
        // diagonal: d2(c/F) = c (2 dF dF^T / F^3 - d2F / F^2)
        let cd: Vec<f64> = (0..n)
            .map(|i| a[i] * b[i] * pts[i].zeta * (2.0 / PI).sqrt())
            .collect();
        let hd = par_fold_det(
            n,
            nchunk_for(n, n3 * n3, DEFAULT_BUDGET),
            || vec![0.0; n3 * n3],
            |h, i| {
                if cd[i] == 0.0 || terms[i].is_empty() {
                    return;
                }
                let f = surf.switch[i];
                let (g, hb) = self.switch_point_derivs(i, &terms[i]);
                for (x, gx) in &g {
                    for (y, gy) in &g {
                        add_block(h, n3, *x, *y, 2.0 * cd[i] / (f * f * f), &outer(gx, gy));
                    }
                }
                for (x, y, m) in hb {
                    add_block(h, n3, x, y, -cd[i] / (f * f), &m);
                }
            },
            |x, y| add_into(x, &y),
        );
        add_into(&mut h, &hd);
        h
    }

    /// sum_{i != j} a_i b_j d2D_ij/dXdY (3N x 3N).
    pub(crate) fn d2d(&self, natom: usize, a: &[f64], b: &[f64]) -> Vec<f64> {
        let surf = &self.surf;
        let pts = &surf.pts;
        let n = surf.len();
        let n3 = 3 * natom;
        par_fold_det(
            n,
            nchunk_for(n, n3 * n3, DEFAULT_BUDGET),
            || vec![0.0; n3 * n3],
            |h, i| {
                if a[i] == 0.0 {
                    return;
                }
                let oi = surf.owner[i];
                for j in 0..n {
                    let oj = surf.owner[j];
                    if j == i || b[j] == 0.0 || oi == oj {
                        continue;
                    }
                    let u = sub(&pts[i].r, &pts[j].r);
                    let (_, hm) = d_derivs(zeta_ij(pts[i].zeta, pts[j].zeta), &u, &surf.normal[j]);
                    add_pair_blocks(h, n3, oi, oj, a[i] * b[j], &hm);
                }
            },
            |x, y| add_into(x, &y),
        )
    }

    /// Solve K X = B for the columns of B (npts x m).
    // one right-hand side with the fixed degree of dense_par (thread-count independent;
    // util::linalg would run it sequentially and change the result)
    #[allow(clippy::disallowed_methods)]
    pub(crate) fn solve_k_many(&self, b: &mut Mat<f64>) {
        let n = b.nrows();
        let par = dense_par(n);
        match &self.factor {
            Factor::Llt(l) => {
                tri::solve_lower_triangular_in_place(l.as_ref(), b.as_mut(), par);
                tri::solve_upper_triangular_in_place(l.transpose(), b.as_mut(), par);
            }
            Factor::Lu { lu, fwd } => {
                let p = Mat::from_fn(n, b.ncols(), |i, k| b[(fwd[i], k)]);
                b.copy_from(p.as_ref());
                tri::solve_unit_lower_triangular_in_place(lu.as_ref(), b.as_mut(), par);
                tri::solve_upper_triangular_in_place(lu.as_ref(), b.as_mut(), par);
            }
            Factor::Elsewhere => panic!("PCM: K is factored on a device, not on the host"),
        }
    }

    /// Solve K^T X = B for the columns of B.
    // one right-hand side with the fixed degree of dense_par (thread-count independent;
    // util::linalg would run it sequentially and change the result)
    #[allow(clippy::disallowed_methods)]
    pub(crate) fn solve_kt_many(&self, b: &mut Mat<f64>) {
        let n = b.nrows();
        let par = dense_par(n);
        match &self.factor {
            Factor::Llt(_) => self.solve_k_many(b),
            Factor::Lu { lu, fwd } => {
                tri::solve_lower_triangular_in_place(lu.transpose(), b.as_mut(), par);
                tri::solve_unit_upper_triangular_in_place(lu.transpose(), b.as_mut(), par);
                let mut p = Mat::<f64>::zeros(n, b.ncols());
                for i in 0..n {
                    for k in 0..b.ncols() {
                        p[(fwd[i], k)] = b[(i, k)];
                    }
                }
                b.copy_from(p.as_ref());
            }
            Factor::Elsewhere => panic!("PCM: K is factored on a device, not on the host"),
        }
    }

    /// m X for X given as [ncol][npts] (X-major); returns the same layout.
    // one right-hand side with the fixed degree of dense_par (thread-count independent;
    // util::linalg would run it sequentially and change the result)
    #[allow(clippy::disallowed_methods)]
    fn mat_many(&self, m: MatRef<'_, f64>, x: &[f64]) -> Vec<f64> {
        let n = self.surf.len();
        let ncol = x.len() / n.max(1);
        let xm = MatRef::from_column_major_slice(x, n, ncol);
        let mut out = vec![0.0; n * ncol];
        let om = faer::MatMut::from_column_major_slice_mut(&mut out, n, ncol);
        matmul(om, Accum::Replace, m, xm, 1.0, dense_par(n));
        out
    }

    /// Q_sym applied to the columns of `phis` (\[ncol\]\[npts\]).
    pub fn q_sym_many(&self, phis: &[f64]) -> Vec<f64> {
        let n = self.surf.len();
        let ncol = phis.len() / n.max(1);
        let f = self.f_eps;
        match self.model {
            PcmModel::Cpcm | PcmModel::Cosmo => {
                let mut b = Mat::from_fn(n, ncol, |i, k| phis[k * n + i]);
                self.solve_k_many(&mut b);
                (0..ncol * n).map(|x| -f * b[(x % n, x / n)]).collect()
            }
            _ => {
                // q = K^-1 R phi, qt = R^T K^-T phi
                let rphi = self.r_many(phis);
                let mut b = Mat::from_fn(n, ncol, |i, k| rphi[k * n + i]);
                self.solve_k_many(&mut b);
                let mut c = Mat::from_fn(n, ncol, |i, k| phis[k * n + i]);
                self.solve_kt_many(&mut c);
                let cv: Vec<f64> = (0..ncol * n).map(|x| c[(x % n, x / n)]).collect();
                let qt = self.rt_many(&cv);
                (0..ncol * n)
                    .map(|x| 0.5 * (b[(x % n, x / n)] + qt[x]))
                    .collect()
            }
        }
    }

    /// R X for columns X.
    fn r_many(&self, x: &[f64]) -> Vec<f64> {
        let n = self.surf.len();
        let f = self.f_eps;
        match self.model {
            PcmModel::Cpcm | PcmModel::Cosmo => x.iter().map(|v| -f * v).collect(),
            _ => {
                let ax: Vec<f64> = x
                    .iter()
                    .enumerate()
                    .map(|(k, v)| v * self.surf.area[k % n])
                    .collect();
                let dax = self.mat_many(self.d.as_ref().unwrap().as_ref(), &ax);
                x.iter()
                    .zip(&dax)
                    .map(|(v, w)| -f * (v - w / (2.0 * PI)))
                    .collect()
            }
        }
    }

    /// R^T X for columns X.
    fn rt_many(&self, x: &[f64]) -> Vec<f64> {
        let n = self.surf.len();
        let f = self.f_eps;
        match self.model {
            PcmModel::Cpcm | PcmModel::Cosmo => x.iter().map(|v| -f * v).collect(),
            _ => {
                let dtx = self.mat_many(self.d.as_ref().unwrap().transpose(), x);
                x.iter()
                    .zip(&dtx)
                    .enumerate()
                    .map(|(k, (v, t))| -f * (v - self.surf.area[k % n] * t / (2.0 * PI)))
                    .collect()
            }
        }
    }

    /// (w, h2): w_X = (dQ_sym/dX) phi (\[3N\]\[npts\]) and 1/2 phi^T d2Q/dXdY phi (3N x 3N), for
    /// the potential `phi` with charges `c` = solve(phi).
    #[allow(clippy::type_complexity, reason = "a one-off tuple of buffers")]
    pub fn solver_second(
        &self,
        positions: &[[f64; 3]],
        phi: &[f64],
        c: &Charges,
    ) -> (Vec<f64>, Vec<f64>) {
        let natom = positions.len();
        let n3 = 3 * natom;
        let n = self.surf.len();
        let (q, y) = (&c.q, &c.y);
        let terms = self.all_switch_terms(positions);
        let df = self.switch_deriv(natom, &terms);
        let area = &self.surf.area;
        // dA = W R^2 dF
        let wr2: Vec<f64> = (0..n)
            .map(|i| self.surf.weight[i] * self.surf.radius_of(i).powi(2))
            .collect();
        let da: Vec<f64> = df.iter().enumerate().map(|(x, v)| v * wr2[x % n]).collect();
        let hadamard = |arr: &[f64], v: &[f64]| -> Vec<f64> {
            arr.iter().enumerate().map(|(x, a)| a * v[x % n]).collect()
        };
        let scale_rows = |arr: &[f64], v: &[f64]| -> Vec<f64> { hadamard(arr, v) };
        let dotx = |a: &[f64], b: &[f64]| -> Vec<f64> {
            // [X][Y] = a_X . b_Y
            let am = MatRef::from_column_major_slice(a, n, n3);
            let bm = MatRef::from_column_major_slice(b, n, n3);
            let mut m = Mat::<f64>::zeros(n3, n3);
            linus_base::linalg::matmul(
                m.as_mut(),
                Accum::Replace,
                am.transpose(),
                bm,
                1.0,
                dense_par(n),
            );
            (0..n3 * n3).map(|x| m[(x / n3, x % n3)]).collect()
        };
        let sym_add = |h: &mut Vec<f64>, m: &[f64], c: f64| {
            for x in 0..n3 {
                for yy in 0..n3 {
                    h[x * n3 + yy] += c * (m[x * n3 + yy] + m[yy * n3 + x]);
                }
            }
        };
        let to_mat = |v: &[f64]| Mat::from_fn(n, n3, |i, k| v[k * n + i]);
        let from_mat =
            |m: &Mat<f64>| -> Vec<f64> { (0..n3 * n).map(|x| m[(x % n, x / n)]).collect() };
        let f = self.f_eps;
        let (dkq, dkty, drphi, drty, d2r, d2k): (
            Vec<f64>,
            Vec<f64>,
            Vec<f64>,
            Vec<f64>,
            Vec<f64>,
            Vec<f64>,
        ) = match self.model {
            PcmModel::Cpcm | PcmModel::Cosmo => {
                let dsq = self.ds_mv(natom, q, &df);
                let dsy = self.ds_mv(natom, y, &df);
                let d2s = self.d2s(natom, y, q, &terms);
                (
                    dsq,
                    dsy,
                    vec![0.0; n3 * n],
                    vec![0.0; n3 * n],
                    vec![0.0; n3 * n3],
                    d2s,
                )
            }
            PcmModel::Iefpcm | PcmModel::Ssvpe | PcmModel::Smd => {
                let s = self.s.as_ref().unwrap().as_ref();
                let d = self.d.as_ref().unwrap().as_ref();
                let cc = f / (2.0 * PI);
                let mv = |m: MatRef<'_, f64>, v: &[f64]| super::pcm::matvec(m, v);
                let hv = |a: &[f64], b: &[f64]| -> Vec<f64> {
                    a.iter().zip(b).map(|(x, y)| x * y).collect()
                };
                // d(DAS) v and d(S A D^T) v for all X
                let d_das = |v: &[f64]| -> Vec<f64> {
                    let sv = mv(s, v);
                    let t1 = self.dd_mv(natom, &hv(area, &sv));
                    let t2 = self.mat_many(d, &hadamard(&da, &sv));
                    let dsv = self.ds_mv(natom, v, &df);
                    let t3 = self.mat_many(d, &scale_rows(&dsv, area));
                    t1.iter()
                        .zip(&t2)
                        .zip(&t3)
                        .map(|((a, b), c)| a + b + c)
                        .collect()
                };
                let d_sadt = |v: &[f64]| -> Vec<f64> {
                    let dtv = super::pcm::matvec_t(d, v);
                    let t1 = self.ds_mv(natom, &hv(area, &dtv), &df);
                    let t2 = self.mat_many(s, &hadamard(&da, &dtv));
                    let ddt = self.ddt_mv(natom, v);
                    let t3 = self.mat_many(s, &scale_rows(&ddt, area));
                    t1.iter()
                        .zip(&t2)
                        .zip(&t3)
                        .map(|((a, b), c)| a + b + c)
                        .collect()
                };
                // left^T d2(DAS) right (3N x 3N)
                let d2_das = |l: &[f64], rgt: &[f64]| -> Vec<f64> {
                    let sr = mv(s, rgt);
                    let dtl = super::pcm::matvec_t(d, l);
                    let mut h = self.d2d(natom, l, &hv(area, &sr));
                    let c2: Vec<f64> = (0..n).map(|j| dtl[j] * sr[j] * wr2[j]).collect();
                    add_into(&mut h, &self.d2f(natom, &c2, &terms));
                    add_into(&mut h, &self.d2s(natom, &hv(area, &dtl), rgt, &terms));
                    let ddt_l = self.ddt_mv(natom, l);
                    let dsr = self.ds_mv(natom, rgt, &df);
                    // dD_X A... cross terms (X <-> Y symmetrized)
                    sym_add(&mut h, &dotx(&ddt_l, &hadamard(&da, &sr)), 1.0);
                    sym_add(&mut h, &dotx(&scale_rows(&ddt_l, area), &dsr), 1.0);
                    sym_add(&mut h, &dotx(&hadamard(&da, &dtl), &dsr), 1.0);
                    h
                };
                let aphi = hv(area, phi);
                let dty = super::pcm::matvec_t(d, y);
                // dR phi = c (dD (A phi) + D (dA phi)); dR^T y = c (A dD^T y + dA D^T y)
                let drphi: Vec<f64> = self
                    .dd_mv(natom, &aphi)
                    .iter()
                    .zip(self.mat_many(d, &hadamard(&da, phi)))
                    .map(|(a, b)| cc * (a + b))
                    .collect();
                let ddty = self.ddt_mv(natom, y);
                let drty: Vec<f64> = scale_rows(&ddty, area)
                    .iter()
                    .zip(hadamard(&da, &dty))
                    .map(|(a, b)| cc * (a + b))
                    .collect();
                // y^T d2R phi
                let mut d2r = self.d2d(natom, y, &aphi);
                let c1: Vec<f64> = (0..n).map(|j| dty[j] * phi[j] * wr2[j]).collect();
                add_into(&mut d2r, &self.d2f(natom, &c1, &terms));
                sym_add(&mut d2r, &dotx(&ddty, &hadamard(&da, phi)), 1.0);
                d2r.iter_mut().for_each(|x| *x *= cc);
                let dsq = self.ds_mv(natom, q, &df);
                let dsy = self.ds_mv(natom, y, &df);
                let mut d2k = self.d2s(natom, y, q, &terms);
                let (dkq, dkty) = if self.model == PcmModel::Iefpcm {
                    let dkq: Vec<f64> = dsq.iter().zip(d_das(q)).map(|(a, b)| a - cc * b).collect();
                    let dkty: Vec<f64> =
                        dsy.iter().zip(d_sadt(y)).map(|(a, b)| a - cc * b).collect();
                    let x = d2_das(y, q);
                    d2k.iter_mut().zip(&x).for_each(|(a, b)| *a -= cc * b);
                    (dkq, dkty)
                } else {
                    let h = 0.5 * cc;
                    let dkq: Vec<f64> = dsq
                        .iter()
                        .zip(d_das(q))
                        .zip(d_sadt(q))
                        .map(|((a, b), c)| a - h * (b + c))
                        .collect();
                    let dkty: Vec<f64> = dsy
                        .iter()
                        .zip(d_das(y))
                        .zip(d_sadt(y))
                        .map(|((a, b), c)| a - h * (b + c))
                        .collect();
                    let x1 = d2_das(y, q);
                    let x2 = d2_das(q, y);
                    d2k.iter_mut()
                        .zip(x1.iter().zip(&x2))
                        .for_each(|(a, (b, c))| *a -= h * (b + c));
                    (dkq, dkty)
                };
                (dkq, dkty, drphi, drty, d2r, d2k)
            }
        };
        // dQ phi = K^-1 (dR phi - dK q)
        let rhs: Vec<f64> = drphi.iter().zip(&dkq).map(|(a, b)| a - b).collect();
        let mut m = to_mat(&rhs);
        self.solve_k_many(&mut m);
        let dqphi = from_mat(&m);
        // dQ^T phi = dR^T y - R^T K^-T (dK^T y)
        let mut m = to_mat(&dkty);
        self.solve_kt_many(&mut m);
        let rt = self.rt_many(&from_mat(&m));
        let dqtphi: Vec<f64> = drty.iter().zip(&rt).map(|(a, b)| a - b).collect();
        let w: Vec<f64> = dqphi
            .iter()
            .zip(&dqtphi)
            .map(|(a, b)| 0.5 * (a + b))
            .collect();
        // 1/2 phi^T d2Q phi
        let mut h2 = vec![0.0; n3 * n3];
        sym_add(&mut h2, &dotx(&dqphi, &dkty), -0.5);
        for x in 0..n3 * n3 {
            h2[x] += 0.5 * (d2r[x] - d2k[x]);
        }
        (w, h2)
    }

    /// Nuclear potential at the points: first derivatives \[3N\]\[npts\] and q . d2v_n (3N x 3N).
    pub fn nuclear_derivs(
        &self,
        z: &[f64],
        positions: &[[f64; 3]],
        q: &[f64],
    ) -> (Vec<f64>, Vec<f64>) {
        let surf = &self.surf;
        let n = surf.len();
        let natom = positions.len();
        let n3 = 3 * natom;
        let cols: Vec<Vec<f64>> = (0..n)
            .into_par_iter()
            .map(|g| {
                let mut col = vec![0.0; n3];
                let pt = &surf.pts[g];
                let o = surf.owner[g];
                for (a, c) in positions.iter().enumerate() {
                    if a == o {
                        continue;
                    }
                    let u = sub(&pt.r, c);
                    let r = norm(&u);
                    let s1 = coul_erf3(self.pot_zeta[g], r).1;
                    for k in 0..3 {
                        col[3 * o + k] += z[a] * s1 / r * u[k];
                        col[3 * a + k] -= z[a] * s1 / r * u[k];
                    }
                }
                col
            })
            .collect();
        let dv = Self::assemble(cols, n3);
        let h = par_fold_det(
            n,
            nchunk_for(n, n3 * n3, DEFAULT_BUDGET),
            || vec![0.0; n3 * n3],
            |h, g| {
                if q[g] == 0.0 {
                    return;
                }
                let pt = &surf.pts[g];
                let o = surf.owner[g];
                for (a, c) in positions.iter().enumerate() {
                    if a == o {
                        continue;
                    }
                    let u = sub(&pt.r, c);
                    let r = norm(&u);
                    let (_, s1, s2, _) = coul_erf3(self.pot_zeta[g], r);
                    add_pair_blocks(h, n3, o, a, q[g] * z[a], &radial_hess(&u, r, s1, s2));
                }
            },
            |x, y| add_into(x, &y),
        );
        (dv, h)
    }
}
