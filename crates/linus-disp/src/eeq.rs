// SPDX-License-Identifier: LGPL-3.0-or-later
// Ported from multicharge 0.6.0 (Apache-2.0, see LICENSE-APACHE): model/eeq.f90,
// model/type.F90, param/eeq2019.f90 and mctc-lib 0.7.0 ncoord/erf.f90, ncoord/type.f90.
//! Electronegativity equilibration (EEQ 2019) charges of DFT-D4 (Caldeweyher et al.,
//! JCP 150, 154122 (2019)) with first and second geometric derivatives, molecules only.
//!
//! The charges `q` and the Lagrange multiplier `mu` solve the bordered system
//! `A (q, mu) = (x, Q_tot)` with `A_ii = eta_i + sqrt(2/pi) / rad_i`,
//! `A_ij = erf(r_ij / sqrt(rad_i^2 + rad_j^2)) / r_ij`, and
//! `x_i = -chi_i + kcnchi_i CN_i / sqrt(CN_i + 1e-14)` with the capped error-function
//! coordination number `CN = ln(1 + e^8) - ln(1 + e^(8 - CN_raw))`,
//! `CN_raw_i = sum_j (1 + erf(-7.5 (r_ij - R_ij) / R_ij)) / 2` (D3 covalent radii, 25 bohr).
//!
//! Derivatives use the adjoint `u = A^-1 (v, 0)` of a charge weight `v` (dE/dq):
//! `v . dq/dR_a = -u . (A_a y - x_a)` and
//! `sum_i v_i d2q_i/dR_a dR_b = u . (x_ab - A_ab y - A_a y_b - A_b y_a)`.

use crate::data::eeq as tab;
use crate::linalg::Lu;
use crate::par;
use crate::units::covalent_radius_d3;
use crate::DispError;
use rayon::prelude::*;

extern "C" {
    /// C99 libm error function (the routine gfortran's ERF intrinsic calls).
    fn erf(x: f64) -> f64;
}

#[inline]
fn erf_f64(x: f64) -> f64 {
    // SAFETY: pure libm function without side effects
    unsafe { erf(x) }
}

/// multicharge `new_eeq2019_model`: CN cutoff, steepness and cap.
const CN_CUTOFF: f64 = 25.0;
const CN_EXP: f64 = 7.5;
const CN_MAX: f64 = 8.0;
/// regularization of the CN term of the right-hand side (`reg` in get_xvec)
const REG: f64 = 1.0e-14;
const SQRTPI: f64 = 1.7724538509055159;

#[inline]
fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

/// EEQ 2019 model of one molecule.
#[derive(Clone, Debug)]
pub struct EeqModel {
    n: usize,
    chi: Vec<f64>,
    eta: Vec<f64>,
    kcnchi: Vec<f64>,
    rad: Vec<f64>,
    rcov: Vec<f64>,
}

/// Solution of the EEQ equations at one geometry.
pub struct EeqState {
    /// atomic partial charges
    pub q: Vec<f64>,
    /// capped coordination numbers
    pub cn: Vec<f64>,
    /// d CN / d CN_raw and d2 CN / d CN_raw^2 of the cap
    cut1: Vec<f64>,
    cut2: Vec<f64>,
    lu: Lu,
}

/// Error-function counting function and its first two r derivatives.
#[inline]
fn count_erf(rc: f64, r: f64) -> (f64, f64, f64) {
    let e = CN_EXP * (r - rc) / rc;
    let c = 0.5 * (1.0 + erf_f64(-e));
    let ex = (-(e * e)).exp();
    let d1 = -CN_EXP * ex / (SQRTPI * rc);
    let d2 = 2.0 * e * CN_EXP * CN_EXP * ex / (SQRTPI * rc * rc);
    (c, d1, d2)
}

/// `erf(gam r) / r` and its first two r derivatives (multicharge `coulomb_kernel`).
#[inline]
fn coulomb_kernel(r: f64, gam: f64) -> (f64, f64, f64) {
    let erfv = erf_f64(gam * r);
    let ex = 2.0 / SQRTPI * (-(gam * r) * (gam * r)).exp();
    let val = erfv / r;
    let dr = gam * ex / r - erfv / (r * r);
    let drr = -2.0 * gam * gam * gam * ex - 2.0 * gam * ex / (r * r) + 2.0 * erfv / (r * r * r);
    (val, dr, drr)
}

/// Hessian block of a radial pair function (`dE/dr = e1`, `d2E/dr2 = e2`, unit vector `u`).
#[inline]
fn radial_block(u: [f64; 3], r: f64, e1: f64, e2: f64) -> [[f64; 3]; 3] {
    let mut b = [[0.0; 3]; 3];
    let er = e1 / r;
    for x in 0..3 {
        for y in 0..3 {
            b[x][y] = (e2 - er) * u[x] * u[y];
        }
        b[x][x] += er;
    }
    b
}

impl EeqModel {
    /// EEQ model for atomic numbers 1-103.
    pub fn new(numbers: &[u32]) -> Result<Self, DispError> {
        for (iat, &z) in numbers.iter().enumerate() {
            if z == 0 || z as usize > tab::EEQ_CHI.len() {
                return Err(DispError::UnsupportedElement {
                    model: "EEQ (D4)",
                    atom: iat,
                    number: z,
                });
            }
        }
        let p = |t: &[f64; 103]| {
            numbers
                .iter()
                .map(|&z| t[z as usize - 1])
                .collect::<Vec<_>>()
        };
        Ok(Self {
            n: numbers.len(),
            chi: p(&tab::EEQ_CHI),
            eta: p(&tab::EEQ_ETA),
            kcnchi: p(&tab::EEQ_KCNCHI),
            rad: p(&tab::EEQ_RAD),
            rcov: numbers.iter().map(|&z| covalent_radius_d3(z)).collect(),
        })
    }

    /// `1 / sqrt(rad_i^2 + rad_j^2)`
    #[inline]
    fn gam(&self, i: usize, j: usize) -> f64 {
        1.0 / (self.rad[i] * self.rad[i] + self.rad[j] * self.rad[j]).sqrt()
    }

    /// d x_i / d CN_i and d2 x_i / d CN_i^2
    #[inline]
    fn dx(&self, i: usize, cn: f64) -> (f64, f64) {
        let k = self.kcnchi[i];
        (
            k * (0.5 * cn + REG) / (cn + REG).powf(1.5),
            -k * (0.25 * cn + REG) / (cn + REG).powf(2.5),
        )
    }

    /// Solve the EEQ equations for total charge `charge`.
    pub fn solve(&self, xyz: &[[f64; 3]], charge: f64) -> Result<EeqState, DispError> {
        let n = self.n;
        let cutoff2 = CN_CUTOFF * CN_CUTOFF;
        // raw CN: every atom sums its neighbours in index order
        let raw: Vec<f64> = (0..n)
            .into_par_iter()
            .map(|i| {
                let mut c = 0.0;
                for j in 0..n {
                    let v = sub(xyz[i], xyz[j]);
                    let r2 = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
                    if r2 > cutoff2 || r2 < 1.0e-12 {
                        continue;
                    }
                    c += count_erf(self.rcov[i] + self.rcov[j], r2.sqrt()).0;
                }
                c
            })
            .collect();
        let em = CN_MAX.exp();
        let cn: Vec<f64> = raw
            .iter()
            .map(|&c| (1.0 + em).ln() - (1.0 + (CN_MAX - c).exp()).ln())
            .collect();
        let cut1: Vec<f64> = raw.iter().map(|&c| em / (em + c.exp())).collect();
        let cut2: Vec<f64> = raw
            .iter()
            .map(|&c| {
                let arg = (-(CN_MAX - c).abs()).exp();
                -arg / ((1.0 + arg) * (1.0 + arg))
            })
            .collect();
        // bordered Coulomb matrix and right-hand side
        let m = n + 1;
        let mut a = vec![0.0; m * m];
        a.par_chunks_mut(m).enumerate().for_each(|(i, row)| {
            if i == n {
                row[..n].fill(1.0);
                row[n] = 0.0;
                return;
            }
            let sqrt2pi = (2.0 / crate::data::common::PI).sqrt();
            for j in 0..n {
                row[j] = if i == j {
                    self.eta[i] + sqrt2pi / self.rad[i]
                } else {
                    let v = sub(xyz[j], xyz[i]);
                    let r2 = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
                    let gam = 1.0 / (self.rad[i] * self.rad[i] + self.rad[j] * self.rad[j]);
                    erf_f64((r2 * gam).sqrt()) / r2.sqrt()
                };
            }
            row[n] = 1.0;
        });
        let mut x = vec![0.0; m];
        for i in 0..n {
            let tmp = self.kcnchi[i] / (cn[i] + REG).sqrt();
            x[i] = -self.chi[i] + tmp * cn[i];
        }
        x[n] = charge;
        let lu = Lu::new(m, a).ok_or_else(|| DispError::Numerical("singular EEQ matrix".into()))?;
        lu.solve(&mut x);
        Ok(EeqState {
            q: x[..n].to_vec(),
            cn,
            cut1,
            cut2,
            lu,
        })
    }

    /// Adjoint `u = A^-1 (v, 0)` (first n entries).
    fn adjoint(&self, st: &EeqState, v: &[f64]) -> Vec<f64> {
        let mut u = v[..self.n].to_vec();
        u.push(0.0);
        st.lu.solve(&mut u);
        u.truncate(self.n);
        u
    }

    /// `sum_i v_i dq_i/dR`, added to `grad`.
    pub fn add_charge_gradient(
        &self,
        xyz: &[[f64; 3]],
        st: &EeqState,
        v: &[f64],
        grad: &mut [[f64; 3]],
    ) {
        let n = self.n;
        let u = self.adjoint(st, v);
        // weights of the CN derivative: u_i x'_i cut'_i (chain through the cap)
        let w: Vec<f64> = (0..n)
            .map(|i| u[i] * self.dx(i, st.cn[i]).0 * st.cut1[i])
            .collect();
        let q = &st.q;
        let cutoff2 = CN_CUTOFF * CN_CUTOFF;
        grad.par_iter_mut().enumerate().for_each(|(a, ga)| {
            for l in 0..n {
                if l == a {
                    continue;
                }
                let v = sub(xyz[a], xyz[l]);
                let r2 = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
                let r = r2.sqrt();
                let (_, da, _) = coulomb_kernel(r, self.gam(a, l));
                let mut s = -(u[a] * q[l] + u[l] * q[a]) * da;
                if r2 <= cutoff2 && r2 >= 1.0e-12 {
                    let (_, dc, _) = count_erf(self.rcov[a] + self.rcov[l], r);
                    s += (w[a] + w[l]) * dc;
                }
                for x in 0..3 {
                    ga[x] += s * v[x] / r;
                }
            }
        });
    }

    /// Jacobian of the raw CN, `Jr[k][3a + x] = dCN_raw_k / dR_ax` (row-major n x 3n).
    fn raw_cn_jacobian(&self, xyz: &[[f64; 3]]) -> Vec<f64> {
        let n = self.n;
        let n3 = 3 * n;
        let cutoff2 = CN_CUTOFF * CN_CUTOFF;
        let mut jac = vec![0.0; n * n3];
        jac.par_chunks_mut(n3.max(1))
            .enumerate()
            .for_each(|(k, row)| {
                for l in 0..n {
                    let v = sub(xyz[k], xyz[l]);
                    let r2 = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
                    if r2 > cutoff2 || r2 < 1.0e-12 {
                        continue;
                    }
                    let r = r2.sqrt();
                    let (_, dc, _) = count_erf(self.rcov[k] + self.rcov[l], r);
                    for x in 0..3 {
                        let g = dc * v[x] / r;
                        row[3 * k + x] += g;
                        row[3 * l + x] -= g;
                    }
                }
            });
        jac
    }

    /// Charge Jacobian `Q[i][3a + x] = dq_i / dR_ax` (row-major n x 3n).
    pub fn charge_jacobian(&self, xyz: &[[f64; 3]], st: &EeqState) -> Vec<f64> {
        let n = self.n;
        let n3 = 3 * n;
        let m = n + 1;
        let jr = self.raw_cn_jacobian(xyz);
        let q = &st.q;
        // right-hand sides R[i][c] = (A_c y)_i - (x_c)_i, R[n][c] = 0
        let mut rhs = vec![0.0; m * n3];
        rhs[..n * n3]
            .par_chunks_mut(n3.max(1))
            .enumerate()
            .for_each(|(i, row)| {
                let xp = self.dx(i, st.cn[i]).0 * st.cut1[i];
                for c in 0..n3 {
                    row[c] = -xp * jr[i * n3 + c];
                }
                for j in 0..n {
                    if j == i {
                        continue;
                    }
                    let v = sub(xyz[i], xyz[j]);
                    let r = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
                    let (_, da, _) = coulomb_kernel(r, self.gam(i, j));
                    for x in 0..3 {
                        let g = da * v[x] / r; // dA_ij / dR_i
                        row[3 * i + x] += g * q[j];
                        row[3 * j + x] -= g * q[j];
                    }
                }
            });
        st.lu.solve_columns(n3, &mut rhs);
        rhs.truncate(n * n3);
        for v in rhs.iter_mut() {
            *v = -*v;
        }
        rhs
    }

    /// Second-order charge response contracted with the weights `v` (dE/dq), split into
    /// the parts the D4 Hessian needs:
    /// adds `u . (x_ab - A_ab y)` to `h` (3n x 3n) and returns `D[a][j] = (u^T A_a)_j`
    /// (row-major 3n x n), so that the remaining term is `-(D Q + (D Q)^T)`.
    pub fn add_charge_hessian_parts(
        &self,
        xyz: &[[f64; 3]],
        st: &EeqState,
        v: &[f64],
        h: &mut [f64],
    ) -> Vec<f64> {
        let n = self.n;
        let n3 = 3 * n;
        let u = self.adjoint(st, v);
        let q = &st.q;
        let cutoff2 = CN_CUTOFF * CN_CUTOFF;
        // u . x_ab: sum_i alpha_i grad CN_raw_i grad CN_raw_i^T + sum_i beta_i d2CN_raw_i
        let mut alpha = vec![0.0; n];
        let mut beta = vec![0.0; n];
        for i in 0..n {
            let (x1, x2) = self.dx(i, st.cn[i]);
            alpha[i] = u[i] * (x2 * st.cut1[i] * st.cut1[i] + x1 * st.cut2[i]);
            beta[i] = u[i] * x1 * st.cut1[i];
        }
        let jr = self.raw_cn_jacobian(xyz);
        let mut aj = jr.clone();
        for i in 0..n {
            for c in 0..n3 {
                aj[i * n3 + c] *= alpha[i];
            }
        }
        let jt = par::transpose(n, n3, &jr);
        par::gemm_acc(n3, n, n3, &jt, &aj, h);
        let mut d = vec![0.0; n3 * n];
        h.par_chunks_mut(3 * n3)
            .zip(d.par_chunks_mut(3 * n))
            .enumerate()
            .for_each(|(a, (hrows, drows))| {
                for l in 0..n {
                    if l == a {
                        continue;
                    }
                    let vv = sub(xyz[a], xyz[l]);
                    let r2 = vv[0] * vv[0] + vv[1] * vv[1] + vv[2] * vv[2];
                    let r = r2.sqrt();
                    let un = [vv[0] / r, vv[1] / r, vv[2] / r];
                    // - (u_a q_l + u_l q_a) d2A_al
                    let (_, da, dda) = coulomb_kernel(r, self.gam(a, l));
                    let wq = -(u[a] * q[l] + u[l] * q[a]);
                    let mut e1 = wq * da;
                    let mut e2 = wq * dda;
                    if r2 <= cutoff2 && r2 >= 1.0e-12 {
                        let (_, dc, ddc) = count_erf(self.rcov[a] + self.rcov[l], r);
                        e1 += (beta[a] + beta[l]) * dc;
                        e2 += (beta[a] + beta[l]) * ddc;
                    }
                    let blk = radial_block(un, r, e1, e2);
                    for x in 0..3 {
                        for y in 0..3 {
                            hrows[x * n3 + 3 * a + y] += blk[x][y];
                            hrows[x * n3 + 3 * l + y] -= blk[x][y];
                        }
                        // D[a][l] = u_a dA_al/dR_a, D[a][a] = sum_l u_l dA_al/dR_a
                        drows[x * n + l] += u[a] * da * un[x];
                        drows[x * n + a] += u[l] * da * un[x];
                    }
                }
            });
        d
    }
}

/// EEQ 2019 charges of a molecule with total charge `charge` (as dftd4 computes them).
pub fn eeq_charges(z: &[u32], xyz_bohr: &[[f64; 3]], charge: f64) -> Result<Vec<f64>, DispError> {
    let m = EeqModel::new(z)?;
    Ok(m.solve(xyz_bohr, charge)?.q)
}
