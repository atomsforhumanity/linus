//! The PCM linear response: matrices of the discretized surface, the solve for the surface
//! charges, the energy and Fock contribution of a density, and the explicit (fixed-density)
//! nuclear derivatives of the energy.
//!
//! With the total electrostatic potential of the solute at the points, phi = v_n - v_e
//! (v_n: nuclei, v_e: electrons, both as Gaussian-charge potentials), the charges solve
//! K q = R phi (Lange & Herbert, J. Chem. Phys. 133, 244111 (2010), PySCF's conventions):
//!
//!   C-PCM  K = S,                                R = -f I,  f = (eps - 1) / eps
//!   COSMO  K = S,                                R = -f I,  f = (eps - 1) / (eps + x)
//!   IEF    K = S - f/(2 pi) D A S,               R = -f (I - D A / (2 pi)), f = (eps-1)/(eps+1)
//!   SS(V)PE K = S - f/(4 pi) (D A S + S A D^T),  R as IEF
//!
//! S_ij = erf(zeta_ij r_ij) / r_ij, zeta_ij = zeta_i zeta_j / sqrt(zeta_i^2 + zeta_j^2),
//! S_ii = zeta_i sqrt(2/pi) / F_i; D_ij = -S'(r_ij) (r_i - r_j).n_j / r_ij,
//! D_ii = -zeta_i sqrt(2/pi) / (2 R_i); A = diag(areas). The energy is
//! E = 1/2 phi^T K^-1 R phi = 1/2 q_sym . phi with q_sym = (K^-1 R + R^T K^-T) phi / 2, and the
//! Fock contribution of electrons dE/dD_ab = -sum_g q_sym,g (ab|g).
//!
//! K is factored once per geometry (Cholesky for C-PCM/COSMO, LU otherwise) with a fixed
//! parallel degree, so results do not depend on the thread count.

use super::surface::{dist, libm_erf, Surface};
use super::PcmModel;
use faer::dyn_stack::{MemBuffer, MemStack};
use faer::linalg::triangular_solve as tri;
use faer::{Accum, Mat, MatRef, Par};
use linus_base::linalg::cholesky_in_place;
use rayon::prelude::*;
use std::f64::consts::PI;

/// Fixed parallel degree of the dense factorizations and products (thread-count independence,
/// as `linus_base::linalg::EIGH_DEGREE`).
const DENSE_DEGREE: usize = 8;

pub(crate) fn dense_par(n: usize) -> Par {
    if n >= 256 {
        Par::rayon(DENSE_DEGREE)
    } else {
        Par::Seq
    }
}

/// erf(z r)/r and its first two derivatives with respect to r.
#[inline]
pub(crate) fn coul_erf(z: f64, r: f64) -> (f64, f64, f64) {
    let (s, s1, s2, _) = coul_erf3(z, r);
    (s, s1, s2)
}

/// erf(z r)/r and its first three derivatives with respect to r (a series for z r < 0.02).
#[inline]
pub(crate) fn coul_erf3(z: f64, r: f64) -> (f64, f64, f64, f64) {
    if z.is_infinite() {
        // point charge: 1/r
        let r2 = r * r;
        return (1.0 / r, -1.0 / r2, 2.0 / (r2 * r), -6.0 / (r2 * r2));
    }
    let x = z * r;
    let c = 2.0 * z / PI.sqrt();
    if x < 0.02 {
        let x2 = x * x;
        return (
            c * (1.0 - x2 / 3.0 + x2 * x2 / 10.0 - x2 * x2 * x2 / 42.0),
            c * z * x * (-2.0 / 3.0 + 0.4 * x2 - x2 * x2 / 7.0),
            c * z * z * (-2.0 / 3.0 + 1.2 * x2 - 5.0 * x2 * x2 / 7.0),
            c * z * z * z * x * (2.4 - 20.0 * x2 / 7.0),
        );
    }
    let e = libm_erf(x);
    let g = c * (-x * x).exp();
    let (r2, r3) = (r * r, r * r * r);
    let s = e / r;
    let s1 = g / r - e / r2;
    let s2 = -g * (2.0 * z * z + 2.0 / r2) + 2.0 * e / r3;
    let s3 = g * (4.0 * z.powi(4) * r + 4.0 * z * z / r + 6.0 / r3) - 6.0 * e / (r2 * r2);
    (s, s1, s2, s3)
}

pub(crate) enum Factor {
    /// Lower Cholesky factor of K.
    Llt(Mat<f64>),
    /// LU of K with partial pivoting (P K = L U, unit L).
    Lu { lu: Mat<f64>, fwd: Vec<usize> },
    /// K is built and factored elsewhere (a device, `super::gpu`): the operator holds the
    /// surface, f and the nuclear potential only ([`PcmOperator::surface_only`]).
    Elsewhere,
}

/// What [`PcmOperator::solver_gradient_by`] needs besides the surface: the products with S
/// and D^T and the O(n^2) off-diagonal sums of the S and D derivative contractions per point.
/// The host's operator gives them from its matrices and points; a device from its own
/// (`super::gpu`).
pub trait SolverTerms {
    /// S x or D^T x.
    fn product(&self, m: MatProduct, x: &[f64]) -> Vec<f64>;
    /// Per point: `PcmOperator::ds_points`.
    fn ds_points(&self, a: &[f64], b: &[f64]) -> Vec<[f64; 3]>;
    /// Per point: `PcmOperator::dd_points`.
    fn dd_points(&self, a: &[f64], b: &[f64]) -> Vec<[f64; 3]>;
}

impl SolverTerms for PcmOperator {
    fn product(&self, m: MatProduct, x: &[f64]) -> Vec<f64> {
        match m {
            MatProduct::S => matvec(self.s.as_ref().unwrap().as_ref(), x),
            MatProduct::Dt => matvec_t(self.d.as_ref().unwrap().as_ref(), x),
        }
    }

    fn ds_points(&self, a: &[f64], b: &[f64]) -> Vec<[f64; 3]> {
        PcmOperator::ds_points(self, a, b)
    }

    fn dd_points(&self, a: &[f64], b: &[f64]) -> Vec<[f64; 3]> {
        PcmOperator::dd_points(self, a, b)
    }
}

/// The matrix-vector products of the solver gradient ([`PcmOperator::solver_gradient_by`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MatProduct {
    /// S x.
    S,
    /// D^T x.
    Dt,
}

/// Charges of one potential: q = K^-1 R phi, y = K^-T phi, q_sym.
#[derive(Clone, Debug)]
pub struct Charges {
    pub q: Vec<f64>,
    pub y: Vec<f64>,
    pub q_sym: Vec<f64>,
}

/// The PCM response at one geometry (independent of the density).
pub struct PcmOperator {
    pub model: PcmModel,
    pub eps: f64,
    pub f_eps: f64,
    pub surf: Surface,
    /// S, kept for IEF-PCM and SS(V)PE (C-PCM and COSMO factor S in place).
    pub(crate) s: Option<Mat<f64>>,
    /// D, for IEF-PCM and SS(V)PE.
    pub(crate) d: Option<Mat<f64>>,
    pub(crate) factor: Factor,
    /// Exponent of each point's potential: its Gaussian's (the potential smeared over the
    /// Gaussian, PySCF) or infinity (the potential at the point, ORCA).
    pub pot_zeta: Vec<f64>,
    /// Potential of the nuclei at the points.
    pub v_nuc: Vec<f64>,
}

/// Row-parallel matrix-vector product m x (each row summed in order: deterministic).
pub(crate) fn matvec(m: MatRef<'_, f64>, x: &[f64]) -> Vec<f64> {
    (0..m.nrows())
        .into_par_iter()
        .map(|i| (0..m.ncols()).map(|j| m[(i, j)] * x[j]).sum())
        .collect()
}

/// m^T x (column-parallel, deterministic).
pub(crate) fn matvec_t(m: MatRef<'_, f64>, x: &[f64]) -> Vec<f64> {
    (0..m.ncols())
        .into_par_iter()
        .map(|j| (0..m.nrows()).map(|i| m[(i, j)] * x[i]).sum())
        .collect()
}

fn dotv(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// S of the surface (dense).
pub(crate) fn s_matrix(surf: &Surface) -> Mat<f64> {
    let n = surf.len();
    let pts = &surf.pts;
    let rows: Vec<Vec<f64>> = (0..n)
        .into_par_iter()
        .map(|i| {
            (0..n)
                .map(|j| {
                    if i == j {
                        pts[i].zeta * (2.0 / PI).sqrt() / surf.switch[i]
                    } else {
                        let zij = zeta_ij(pts[i].zeta, pts[j].zeta);
                        coul_erf(zij, dist(&pts[i].r, &pts[j].r)).0
                    }
                })
                .collect()
        })
        .collect();
    Mat::from_fn(n, n, |i, j| rows[i][j])
}

#[inline]
pub(crate) fn zeta_ij(a: f64, b: f64) -> f64 {
    a * b / (a * a + b * b).sqrt()
}

/// D of the surface (dense).
pub(crate) fn d_matrix(surf: &Surface) -> Mat<f64> {
    let n = surf.len();
    let pts = &surf.pts;
    let rows: Vec<Vec<f64>> = (0..n)
        .into_par_iter()
        .map(|i| {
            (0..n)
                .map(|j| {
                    if i == j {
                        -pts[i].zeta * (2.0 / PI).sqrt() / (2.0 * surf.radius_of(i))
                    } else {
                        let u = sub(&pts[i].r, &pts[j].r);
                        let r = norm(&u);
                        let (_, s1, _) = coul_erf(zeta_ij(pts[i].zeta, pts[j].zeta), r);
                        -s1 * dot3(&u, &surf.normal[j]) / r
                    }
                })
                .collect()
        })
        .collect();
    Mat::from_fn(n, n, |i, j| rows[i][j])
}

#[inline]
pub(crate) fn sub(a: &[f64; 3], b: &[f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
#[inline]
pub(crate) fn norm(a: &[f64; 3]) -> f64 {
    (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt()
}
#[inline]
pub(crate) fn dot3(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Peak bytes of the dense surface matrices of `n` points while [`PcmOperator::new`] builds and
/// factors them: C-PCM and COSMO hold S twice (its rows and the matrix; the Cholesky factor is in
/// place), IEF-PCM, SS(V)PE and SMD hold S, D (and D's rows), DA, DAS and K at once.
pub fn dense_peak_bytes(model: PcmModel, n: usize) -> f64 {
    let n2 = (n as f64) * (n as f64) * std::mem::size_of::<f64>() as f64;
    match model {
        PcmModel::Cpcm | PcmModel::Cosmo => 2.0 * n2,
        PcmModel::Iefpcm | PcmModel::Ssvpe | PcmModel::Smd => 6.0 * n2,
    }
}

impl PcmOperator {
    /// Matrices and factorization for the surface `surf` of atoms (`z`, `positions`, bohr).
    /// `cosmo_x` is the x of COSMO's f = (eps - 1) / (eps + x).
    /// With `point_potential` the solute potential is taken at the points (ORCA) instead of
    /// over the Gaussians (PySCF); the charge-charge interactions S, D are Gaussian either way.
    pub fn new(
        model: PcmModel,
        eps: f64,
        cosmo_x: f64,
        surf: Surface,
        point_potential: bool,
        z: &[f64],
        positions: &[[f64; 3]],
    ) -> Result<Self, String> {
        let (model, f_eps) = model_f(model, eps, cosmo_x);
        let n = surf.len();
        let par = dense_par(n);
        let s = s_matrix(&surf);
        let (s, d, factor) = match model {
            PcmModel::Cpcm | PcmModel::Cosmo => {
                let l = cholesky_in_place(s, par)
                    .map_err(|e| format!("PCM matrix S not positive definite: {e:?}"))?;
                (None, None, Factor::Llt(l))
            }
            PcmModel::Iefpcm | PcmModel::Ssvpe | PcmModel::Smd => {
                let d = d_matrix(&surf);
                // DA S
                let da = Mat::from_fn(n, n, |i, j| d[(i, j)] * surf.area[j]);
                let mut das = Mat::<f64>::zeros(n, n);
                linus_base::linalg::matmul(
                    das.as_mut(),
                    Accum::Replace,
                    da.as_ref(),
                    s.as_ref(),
                    1.0,
                    par,
                );
                let k = if model == PcmModel::Iefpcm {
                    let c = f_eps / (2.0 * PI);
                    Mat::from_fn(n, n, |i, j| s[(i, j)] - c * das[(i, j)])
                } else {
                    let c = f_eps / (4.0 * PI);
                    Mat::from_fn(n, n, |i, j| s[(i, j)] - c * (das[(i, j)] + das[(j, i)]))
                };
                drop(das);
                drop(da);
                let (lu, fwd) = lu(k, par);
                (Some(s), Some(d), Factor::Lu { lu, fwd })
            }
        };
        let pot_zeta: Vec<f64> = surf
            .pts
            .iter()
            .map(|p| {
                if point_potential {
                    f64::INFINITY
                } else {
                    p.zeta
                }
            })
            .collect();
        let v_nuc = nuclear_potential(&surf, &pot_zeta, z, positions);
        Ok(PcmOperator {
            model,
            eps,
            f_eps,
            surf,
            s,
            d,
            factor,
            pot_zeta,
            v_nuc,
        })
    }

    /// The operator's surface data, f and the nuclear potential at the points without the
    /// dense matrices and K's factorization, for a device that builds and factors K itself
    /// (`super::gpu`): [`Self::solve`] and the matrix products are not available; the surface
    /// derivative contractions (`ds_contract`, `dd_contract`, `da_contract`,
    /// `nuclear_gradient`) are.
    pub fn surface_only(
        model: PcmModel,
        eps: f64,
        cosmo_x: f64,
        surf: Surface,
        point_potential: bool,
        z: &[f64],
        positions: &[[f64; 3]],
    ) -> Self {
        let (model, f_eps) = model_f(model, eps, cosmo_x);
        let pot_zeta: Vec<f64> = surf
            .pts
            .iter()
            .map(|p| {
                if point_potential {
                    f64::INFINITY
                } else {
                    p.zeta
                }
            })
            .collect();
        let v_nuc = nuclear_potential(&surf, &pot_zeta, z, positions);
        PcmOperator {
            model,
            eps,
            f_eps,
            surf,
            s: None,
            d: None,
            factor: Factor::Elsewhere,
            pot_zeta,
            v_nuc,
        }
    }

    pub fn npoints(&self) -> usize {
        self.surf.len()
    }

    /// x = K^-1 b
    // one right-hand side with the fixed degree of dense_par (thread-count independent;
    // util::linalg would run it sequentially and change the result)
    #[allow(clippy::disallowed_methods)]
    pub(crate) fn solve_k(&self, b: &[f64]) -> Vec<f64> {
        let n = b.len();
        let par = dense_par(n);
        match &self.factor {
            Factor::Llt(l) => {
                let mut x = Mat::from_fn(n, 1, |i, _| b[i]);
                tri::solve_lower_triangular_in_place(l.as_ref(), x.as_mut(), par);
                tri::solve_upper_triangular_in_place(l.transpose(), x.as_mut(), par);
                (0..n).map(|i| x[(i, 0)]).collect()
            }
            Factor::Lu { lu, fwd } => {
                let mut x = Mat::from_fn(n, 1, |i, _| b[fwd[i]]);
                tri::solve_unit_lower_triangular_in_place(lu.as_ref(), x.as_mut(), par);
                tri::solve_upper_triangular_in_place(lu.as_ref(), x.as_mut(), par);
                (0..n).map(|i| x[(i, 0)]).collect()
            }
            Factor::Elsewhere => panic!("PCM: K is factored on a device, not on the host"),
        }
    }

    /// x = K^-T b
    // one right-hand side with the fixed degree of dense_par (thread-count independent;
    // util::linalg would run it sequentially and change the result)
    #[allow(clippy::disallowed_methods)]
    pub(crate) fn solve_kt(&self, b: &[f64]) -> Vec<f64> {
        let n = b.len();
        let par = dense_par(n);
        match &self.factor {
            Factor::Llt(_) => self.solve_k(b),
            Factor::Lu { lu, fwd } => {
                // K^T = U^T L^T P: U^T z = b, L^T w = z, x = P^T w
                let mut x = Mat::from_fn(n, 1, |i, _| b[i]);
                tri::solve_lower_triangular_in_place(lu.transpose(), x.as_mut(), par);
                tri::solve_unit_upper_triangular_in_place(lu.transpose(), x.as_mut(), par);
                let mut out = vec![0.0; n];
                for i in 0..n {
                    out[fwd[i]] = x[(i, 0)];
                }
                out
            }
            Factor::Elsewhere => panic!("PCM: K is factored on a device, not on the host"),
        }
    }

    /// R phi
    pub(crate) fn apply_r(&self, phi: &[f64]) -> Vec<f64> {
        let f = self.f_eps;
        match self.model {
            PcmModel::Cpcm | PcmModel::Cosmo => phi.iter().map(|x| -f * x).collect(),
            _ => {
                let a_phi: Vec<f64> = phi
                    .iter()
                    .zip(&self.surf.area)
                    .map(|(x, a)| x * a)
                    .collect();
                let dap = matvec(self.d.as_ref().unwrap().as_ref(), &a_phi);
                phi.iter()
                    .zip(&dap)
                    .map(|(x, y)| -f * (x - y / (2.0 * PI)))
                    .collect()
            }
        }
    }

    /// R^T y
    pub(crate) fn apply_rt(&self, y: &[f64]) -> Vec<f64> {
        let f = self.f_eps;
        match self.model {
            PcmModel::Cpcm | PcmModel::Cosmo => y.iter().map(|x| -f * x).collect(),
            _ => {
                let dty = matvec_t(self.d.as_ref().unwrap().as_ref(), y);
                y.iter()
                    .zip(&dty)
                    .zip(&self.surf.area)
                    .map(|((x, t), a)| -f * (x - a * t / (2.0 * PI)))
                    .collect()
            }
        }
    }

    /// Charges of the potential phi.
    pub fn solve(&self, phi: &[f64]) -> Charges {
        let y = self.solve_kt(phi);
        match self.model {
            PcmModel::Cpcm | PcmModel::Cosmo => {
                let q: Vec<f64> = y.iter().map(|x| -self.f_eps * x).collect();
                Charges {
                    q_sym: q.clone(),
                    q,
                    y,
                }
            }
            _ => {
                let q = self.solve_k(&self.apply_r(phi));
                let qt = self.apply_rt(&y);
                let q_sym = q.iter().zip(&qt).map(|(a, b)| 0.5 * (a + b)).collect();
                Charges { q, y, q_sym }
            }
        }
    }

    /// Gradient of the energy at fixed potential phi (the matrix part 1/2 phi^T dQ phi) with
    /// charges `c` = solve(phi): 1/2 y^T dR phi - 1/2 y^T dK q (natom x 3).
    pub fn solver_gradient(
        &self,
        positions: &[[f64; 3]],
        phi: &[f64],
        c: &Charges,
    ) -> Vec<[f64; 3]> {
        self.solver_gradient_by(positions, phi, c, self)
    }

    /// [`Self::solver_gradient`] with the products S x and D^T x and the surface contractions'
    /// off-diagonal sums of `t` (the host's, or a device's for an operator built with
    /// [`Self::surface_only`]).
    pub fn solver_gradient_by(
        &self,
        positions: &[[f64; 3]],
        phi: &[f64],
        c: &Charges,
        t: &dyn SolverTerms,
    ) -> Vec<[f64; 3]> {
        let mv = |m: MatProduct, x: &[f64]| t.product(m, x);
        let ds = |a: &[f64], b: &[f64]| self.ds_fold(positions, a, b, t.ds_points(a, b));
        let dd = |a: &[f64], b: &[f64]| self.dd_fold(positions, t.dd_points(a, b));
        let natom = positions.len();
        let surf = &self.surf;
        let (y, q) = (&c.y, &c.q);
        let mut g = vec![[0.0; 3]; natom];
        let add = |g: &mut Vec<[f64; 3]>, h: &[[f64; 3]], f: f64| {
            for (a, b) in g.iter_mut().zip(h) {
                for k in 0..3 {
                    a[k] += f * b[k];
                }
            }
        };
        match self.model {
            PcmModel::Cpcm | PcmModel::Cosmo => {
                add(&mut g, &ds(y, q), -0.5);
            }
            PcmModel::Iefpcm | PcmModel::Ssvpe | PcmModel::Smd => {
                let a = &surf.area;
                let fac = self.f_eps / (2.0 * PI);
                let dty = mv(MatProduct::Dt, y);
                let sq = mv(MatProduct::S, q);
                let a_phi: Vec<f64> = a.iter().zip(phi).map(|(x, y)| x * y).collect();
                // 1/2 y^T dR phi = fac/2 ([dD](y, A phi) + [dA](D^T y o phi))
                add(&mut g, &dd(y, &a_phi), 0.5 * fac);
                let c1: Vec<f64> = dty.iter().zip(phi).map(|(x, y)| x * y).collect();
                add(&mut g, &self.da_contract(positions, &c1), 0.5 * fac);
                // -1/2 y^T dK q
                add(&mut g, &ds(y, q), -0.5);
                let fk = if self.model == PcmModel::Iefpcm {
                    fac
                } else {
                    self.f_eps / (4.0 * PI)
                };
                let a_sq: Vec<f64> = a.iter().zip(&sq).map(|(x, y)| x * y).collect();
                add(&mut g, &dd(y, &a_sq), 0.5 * fk);
                let c2: Vec<f64> = dty.iter().zip(&sq).map(|(x, y)| x * y).collect();
                add(&mut g, &self.da_contract(positions, &c2), 0.5 * fk);
                let a_dty: Vec<f64> = a.iter().zip(&dty).map(|(x, y)| x * y).collect();
                add(&mut g, &ds(&a_dty, q), 0.5 * fk);
                if self.model == PcmModel::Ssvpe {
                    let dtq = mv(MatProduct::Dt, q);
                    let sy = mv(MatProduct::S, y);
                    let a_dtq: Vec<f64> = a.iter().zip(&dtq).map(|(x, y)| x * y).collect();
                    add(&mut g, &ds(y, &a_dtq), 0.5 * fk);
                    let c3: Vec<f64> = sy.iter().zip(&dtq).map(|(x, y)| x * y).collect();
                    add(&mut g, &self.da_contract(positions, &c3), 0.5 * fk);
                    let a_sy: Vec<f64> = a.iter().zip(&sy).map(|(x, y)| x * y).collect();
                    add(&mut g, &dd(q, &a_sy), 0.5 * fk);
                }
            }
        }
        g
    }

    /// sum_ij a_i b_j dS_ij/dX (natom x 3), diagonal S_ii = zeta_i sqrt(2/pi) / F_i included.
    pub fn ds_contract(&self, positions: &[[f64; 3]], a: &[f64], b: &[f64]) -> Vec<[f64; 3]> {
        self.ds_fold(positions, a, b, self.ds_points(a, b))
    }

    /// The off-diagonal part of [`Self::ds_contract`] per point: the coefficient of dr_i,
    /// sum_{j != i} (a_i b_j + a_j b_i) S'(r_ij) u_ij / r_ij.
    pub(crate) fn ds_points(&self, a: &[f64], b: &[f64]) -> Vec<[f64; 3]> {
        let surf = &self.surf;
        let pts = &surf.pts;
        let n = surf.len();
        (0..n)
            .into_par_iter()
            .map(|i| {
                let mut v = [0.0; 3];
                for j in 0..n {
                    if j == i {
                        continue;
                    }
                    let c = a[i] * b[j] + a[j] * b[i];
                    if c == 0.0 {
                        continue;
                    }
                    let u = sub(&pts[i].r, &pts[j].r);
                    let r = norm(&u);
                    let (_, s1, _) = coul_erf(zeta_ij(pts[i].zeta, pts[j].zeta), r);
                    let f = c * s1 / r;
                    for k in 0..3 {
                        v[k] += f * u[k];
                    }
                }
                v
            })
            .collect()
    }

    /// [`Self::ds_contract`] from its off-diagonal part per point `per`: folded onto the points'
    /// atoms, plus the diagonal through the switching factors.
    fn ds_fold(
        &self,
        positions: &[[f64; 3]],
        a: &[f64],
        b: &[f64],
        per: Vec<[f64; 3]>,
    ) -> Vec<[f64; 3]> {
        let surf = &self.surf;
        let pts = &surf.pts;
        let n = surf.len();
        let mut g = vec![[0.0; 3]; positions.len()];
        for (i, v) in per.iter().enumerate() {
            for k in 0..3 {
                g[surf.owner[i]][k] += v[k];
            }
        }
        // diagonal through the switching factors
        let c: Vec<f64> = (0..n)
            .map(|i| {
                -a[i] * b[i] * pts[i].zeta * (2.0 / PI).sqrt() / (surf.switch[i] * surf.switch[i])
            })
            .collect();
        let gd = surf.switch_grad(positions, &c);
        for (x, y) in g.iter_mut().zip(&gd) {
            for k in 0..3 {
                x[k] += y[k];
            }
        }
        g
    }

    /// sum_{i != j} a_i b_j dD_ij/dX (natom x 3); D_ij = g(r) (u . n_j), u = r_i - r_j,
    /// g = -S'(r)/r.
    pub fn dd_contract(&self, positions: &[[f64; 3]], a: &[f64], b: &[f64]) -> Vec<[f64; 3]> {
        self.dd_fold(positions, self.dd_points(a, b))
    }

    /// [`Self::dd_contract`] per point before the fold: sum_{j != i} (a_i b_j grad_u D_ij -
    /// a_j b_i grad_u D_ji).
    pub(crate) fn dd_points(&self, a: &[f64], b: &[f64]) -> Vec<[f64; 3]> {
        let surf = &self.surf;
        let pts = &surf.pts;
        let n = surf.len();
        // grad_u D_ij = g'(r) u_hat (u . n_j) + g(r) n_j
        let grad = |i: usize, j: usize| -> [f64; 3] {
            let u = sub(&pts[i].r, &pts[j].r);
            let r = norm(&u);
            let (_, s1, s2) = coul_erf(zeta_ij(pts[i].zeta, pts[j].zeta), r);
            let g = -s1 / r;
            let gp = -s2 / r + s1 / (r * r);
            let un = dot3(&u, &surf.normal[j]);
            let nj = &surf.normal[j];
            [
                gp * u[0] / r * un + g * nj[0],
                gp * u[1] / r * un + g * nj[1],
                gp * u[2] / r * un + g * nj[2],
            ]
        };
        (0..n)
            .into_par_iter()
            .map(|i| {
                let mut v = [0.0; 3];
                for j in 0..n {
                    if j == i {
                        continue;
                    }
                    let c1 = a[i] * b[j];
                    if c1 != 0.0 {
                        let gr = grad(i, j);
                        for k in 0..3 {
                            v[k] += c1 * gr[k];
                        }
                    }
                    let c2 = a[j] * b[i];
                    if c2 != 0.0 {
                        let gr = grad(j, i);
                        for k in 0..3 {
                            v[k] -= c2 * gr[k];
                        }
                    }
                }
                v
            })
            .collect()
    }

    /// [`Self::dd_contract`] from its per-point sums: folded onto the points' atoms.
    fn dd_fold(&self, positions: &[[f64; 3]], per: Vec<[f64; 3]>) -> Vec<[f64; 3]> {
        let surf = &self.surf;
        let mut g = vec![[0.0; 3]; positions.len()];
        for (i, v) in per.iter().enumerate() {
            for k in 0..3 {
                g[surf.owner[i]][k] += v[k];
            }
        }
        g
    }

    /// sum_j c_j dA_j/dX, A_j = W_j R_j^2 F_j.
    pub fn da_contract(&self, positions: &[[f64; 3]], c: &[f64]) -> Vec<[f64; 3]> {
        let s = &self.surf;
        let cw: Vec<f64> = (0..s.len())
            .map(|j| c[j] * s.weight[j] * s.radius_of(j).powi(2))
            .collect();
        s.switch_grad(positions, &cw)
    }

    /// q . dv_n/dX: derivative of the nuclear potential at the points (points moving with
    /// their atoms) contracted with charges q (natom x 3).
    pub fn nuclear_gradient(&self, z: &[f64], positions: &[[f64; 3]], q: &[f64]) -> Vec<[f64; 3]> {
        let surf = &self.surf;
        let per: Vec<Vec<[f64; 3]>> = (0..surf.len())
            .into_par_iter()
            .map(|g| {
                let mut v = vec![[0.0; 3]; positions.len()];
                let pt = &surf.pts[g];
                for (a, c) in positions.iter().enumerate() {
                    let u = sub(&pt.r, c);
                    let r = norm(&u);
                    let (_, s1, _) = coul_erf(self.pot_zeta[g], r);
                    let f = q[g] * z[a] * s1 / r;
                    for k in 0..3 {
                        v[a][k] -= f * u[k];
                        v[surf.owner[g]][k] += f * u[k];
                    }
                }
                v
            })
            .collect();
        let mut out = vec![[0.0; 3]; positions.len()];
        for v in per {
            for (o, x) in out.iter_mut().zip(&v) {
                for k in 0..3 {
                    o[k] += x[k];
                }
            }
        }
        out
    }

    /// Sum of the charges q_sym of a potential.
    pub fn energy(phi: &[f64], c: &Charges) -> f64 {
        0.5 * dotv(&c.q_sym, phi)
    }
}

/// The model as the operator treats it (SMD's electrostatics resolve to IEF-PCM or C-PCM in the
/// settings; IEF-PCM here) and its f(eps).
fn model_f(model: PcmModel, eps: f64, cosmo_x: f64) -> (PcmModel, f64) {
    let model = if model == PcmModel::Smd {
        PcmModel::Iefpcm
    } else {
        model
    };
    let f_eps = if eps.is_infinite() {
        1.0
    } else {
        match model {
            PcmModel::Cpcm => (eps - 1.0) / eps,
            PcmModel::Cosmo => (eps - 1.0) / (eps + cosmo_x),
            PcmModel::Iefpcm | PcmModel::Ssvpe | PcmModel::Smd => (eps - 1.0) / (eps + 1.0),
        }
    };
    (model, f_eps)
}

/// Potential of point nuclei at the points: sum_A Z_A erf(zeta_g r)/r (zeta_g = `pot_zeta`).
pub(crate) fn nuclear_potential(
    surf: &Surface,
    pot_zeta: &[f64],
    z: &[f64],
    positions: &[[f64; 3]],
) -> Vec<f64> {
    surf.pts
        .par_iter()
        .zip(pot_zeta)
        .map(|(pt, &zg)| {
            z.iter()
                .zip(positions)
                .map(|(&za, c)| za * coul_erf(zg, dist(&pt.r, c)).0)
                .sum()
        })
        .collect()
}

fn lu(mut a: Mat<f64>, par: Par) -> (Mat<f64>, Vec<usize>) {
    use faer::linalg::lu::partial_pivoting::factor;
    let n = a.nrows();
    let mut fwd = vec![0usize; n];
    let mut inv = vec![0usize; n];
    let mut mem = MemBuffer::new(factor::lu_in_place_scratch::<usize, f64>(
        n,
        n,
        par,
        Default::default(),
    ));
    let _ = factor::lu_in_place(
        a.as_mut(),
        &mut fwd,
        &mut inv,
        par,
        MemStack::new(&mut mem),
        Default::default(),
    );
    (a, fwd)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lu_and_cholesky_solves() {
        let n = 300;
        let m = Mat::from_fn(n, n, |i, j| {
            if i == j {
                4.0 + (i % 7) as f64
            } else {
                ((i * 31 + j * 17) % 13) as f64 / 13.0 - 0.45
            }
        });
        let b: Vec<f64> = (0..n).map(|i| ((i * 7) % 11) as f64 - 5.0).collect();
        let (luf, fwd) = lu(m.clone(), dense_par(n));
        let op = |factor: Factor| PcmOperator {
            model: PcmModel::Iefpcm,
            eps: 2.0,
            f_eps: 0.3,
            surf: Surface {
                scheme: super::super::Discretization::Swig,
                cut: super::super::surface::PointCut::Pyscf,
                nleb: vec![],
                pts: vec![],
                owner: vec![],
                normal: vec![],
                weight: vec![],
                switch: vec![],
                area: vec![],
                atom_radius: vec![],
                r_sw: vec![],
                r_in: vec![],
                atom_start: vec![],
            },
            s: None,
            d: None,
            factor,
            pot_zeta: vec![],
            v_nuc: vec![],
        };
        let p = op(Factor::Lu { lu: luf, fwd });
        let x = p.solve_k(&b);
        let r = matvec(m.as_ref(), &x);
        assert!(r.iter().zip(&b).all(|(a, c)| (a - c).abs() < 1e-10));
        let x = p.solve_kt(&b);
        let r = matvec_t(m.as_ref(), &x);
        assert!(r.iter().zip(&b).all(|(a, c)| (a - c).abs() < 1e-10));
        let sym = Mat::from_fn(n, n, |i, j| {
            m[(i, j)] + m[(j, i)] + if i == j { 400.0 } else { 0.0 }
        });
        let p = op(Factor::Llt(
            cholesky_in_place(sym.clone(), dense_par(n)).unwrap(),
        ));
        let x = p.solve_k(&b);
        let r = matvec(sym.as_ref(), &x);
        assert!(r.iter().zip(&b).all(|(a, c)| (a - c).abs() < 1e-10));
    }
}
