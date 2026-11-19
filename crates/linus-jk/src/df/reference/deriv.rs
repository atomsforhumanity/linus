//! Reference implementation (tests only) of the two-electron pieces of the density-fitted
//! (RI, Coulomb metric) Hessian with the three-center tensor (P|mn) in memory, the code that
//! [`crate::df::hessian`] (blocked over auxiliary functions, no 3-index AO tensor) replaced; the
//! tests there check the blocked terms against it.
//!
//! With (P|mn) three-center integrals, V the metric (P|Q) = L L^T, γ = (P|mn) D_mn,
//! c = V^-1 γ, Z_P = V^-1 (P|m i) (i an occupied orbital), X_P = C_o^T Z_P (o x o),
//! Y_P = C_o X_P, Λ^x_P = C_o^T (P|..)^x C_o, R^x = Λ^x - V^x X and B_o = L^-1 (P|m i):
//!
//!   K^x[C_o C_o^T] C_o = sum_P sum_l (P|ml)^x Y_P,l: + sum_Q B_o,Q (L^-1 R^x)_Q,
//!   J^x[D] C_o        = (sum_P (P|mn)^x c_P) C_o + sum_Q B_o,Q (L^-1 r^x)_Q,  r^x = γ^x - V^x c,
//!
//! and the second derivative of E_2e = 1/2 γ^T V^-1 γ - a_x sum_ij (P|ij) V^-1_PQ (Q|ij) at
//! fixed orbitals is (Bykov et al., Mol. Phys. 113, 1961 (2015), eq. 39 for J)
//!
//!   sum (P|mn)^xy Γ_Pmn + sum (P|Q)^xy γ_PQ + (L^-1 r^x).(L^-1 r^y) - 2 a_x (L^-1 R^x).(L^-1 R^y)
//!
//! with the gradient's Γ_P = c_P D - 2 a_x C_o X_P C_o^T and γ_PQ = -1/2 c_P c_Q + a_x tr(X_P X_Q).

use faer::{Accum, Mat, MatMut, MatRef, Par};
use linus_base::linalg::abt_split;
use linus_base::linalg::matmul;
use linus_base::linalg::{solve_lower_in_place, solve_upper_in_place};
use linus_core::timing::Timers;
use linus_ints::reference::oneshot;
use linus_ints::reference::Dens3c;
use linus_ints::reference::Df3cInput;
use linus_ints::{BasisSet, DerivOrder, PairOptions, PairSet};
use rayon::prelude::*;

/// RI pieces at one geometry: three-center integrals, metric Cholesky factor and the
/// auxiliary pair data for derivative integrals.
pub struct DfDerivJk {
    pub basis: BasisSet,
    pub aux: BasisSet,
    pub pairs: PairSet,
    /// the auxiliary pairs, with second-order derivative expansions
    apairs: PairSet,
    nbf: usize,
    naux: usize,
    natom: usize,
    /// V = L L^T (lower triangle)
    l: Mat<f64>,
    /// (P|mn), naux x nbf^2 row-major
    j3: Vec<f64>,
    /// screening threshold of the derivative-integral contractions
    pub thresh: f64,
}

/// Two-electron pieces of a density-fitted closed-shell Hessian.
pub struct DfHessian2e {
    /// (J^x[D_t] - a_x K^x[D_a]) C_o for every perturbation (nbf x nocc).
    pub fxo: Vec<Mat<f64>>,
    /// Second derivative of the RI two-electron energy at fixed orbitals, (3N)^2 row-major.
    pub hess: Vec<f64>,
}

fn add_into(dst: &mut [f64], src: &[f64]) {
    dst.iter_mut().zip(src).for_each(|(a, b)| *a += b);
}

/// [P][m][i] (naux x n x r) -> [m][P][i] (n x naux x r).
fn to_m_major(x: &[f64], naux: usize, n: usize, r: usize) -> Vec<f64> {
    let mut out = vec![0.0; naux * n * r];
    if r == 0 {
        // no occupied orbitals in this channel (one-electron systems)
        return out;
    }
    out.par_chunks_mut(naux * r)
        .enumerate()
        .for_each(|(m, dst)| {
            for p in 0..naux {
                dst[p * r..(p + 1) * r].copy_from_slice(&x[(p * n + m) * r..(p * n + m + 1) * r]);
            }
        });
    out
}

impl DfDerivJk {
    pub fn new(basis: &BasisSet, aux: &BasisSet, par: Par) -> Self {
        let pairs = PairSet::build(basis, PairOptions::with_order(DerivOrder::Two));
        let apairs = PairSet::aux(aux, PairOptions::with_order(DerivOrder::Two));
        let naux = aux.nbf;
        let v = linus_base::linalg::mat_from_rowmajor(
            naux,
            naux,
            &linus_ints::reference::eri_2c(aux, &apairs),
        );
        let l = linus_base::linalg::cholesky(v.as_ref(), par)
            .expect("auxiliary metric not positive definite");
        let j3 = linus_ints::reference::eri_3c(basis, &pairs, aux, &apairs);
        DfDerivJk {
            basis: basis.clone(),
            aux: aux.clone(),
            pairs,
            apairs,
            nbf: basis.nbf,
            naux,
            natom: basis.atom_shells.len().max(aux.atom_shells.len()),
            l,
            j3,
            thresh: 1e-14,
        }
    }

    /// X <- L^-1 X for X naux x w (row-major).
    fn solve_l(&self, x: &mut [f64], w: usize, par: Par) {
        if w > 0 {
            solve_lower_in_place(
                self.l.as_ref(),
                MatMut::from_row_major_slice_mut(x, self.naux, w),
                par,
            );
        }
    }

    /// X <- L^-T X for X naux x w (row-major).
    fn solve_lt(&self, x: &mut [f64], w: usize, par: Par) {
        if w > 0 {
            solve_upper_in_place(
                self.l.transpose(),
                MatMut::from_row_major_slice_mut(x, self.naux, w),
                par,
            );
        }
    }

    /// γ_t = (P|mn) D_t,mn for several densities: naux x nd (row-major).
    fn fit_rhs(&self, dens: &[&[f64]], par: Par) -> Vec<f64> {
        let (n2, naux, nd) = (self.nbf * self.nbf, self.naux, dens.len());
        let mut dm = Mat::<f64>::zeros(n2, nd);
        for (t, d) in dens.iter().enumerate() {
            for x in 0..n2 {
                dm[(x, t)] = d[x];
            }
        }
        let mut g = vec![0.0; naux * nd];
        matmul(
            MatMut::from_row_major_slice_mut(&mut g, naux, nd),
            Accum::Replace,
            MatRef::from_row_major_slice(&self.j3, naux, n2),
            dm.as_ref(),
            1.0,
            par,
        );
        g
    }

    /// V^x X for the three perturbations of `atom` (X naux x w row-major), 3 x naux x w.
    fn metric_apply(
        &self,
        rows: &(std::ops::Range<usize>, Vec<f64>),
        x: &[f64],
        w: usize,
        par: Par,
    ) -> Vec<f64> {
        let (range, g) = rows;
        let (naux, na) = (self.naux, range.len());
        let mut out = vec![0.0; 3 * naux * w];
        if na == 0 || w == 0 {
            return out;
        }
        let xa = MatRef::from_row_major_slice(&x[range.start * w..range.end * w], na, w);
        let xm = MatRef::from_row_major_slice(x, naux, w);
        for k in 0..3 {
            let gk = MatRef::from_row_major_slice(&g[k * na * naux..(k + 1) * na * naux], na, naux);
            let ok = &mut out[k * naux * w..(k + 1) * naux * w];
            // out[P] += sum_{q on atom} G[q][P] X_q  (all rows)
            matmul(
                MatMut::from_row_major_slice_mut(ok, naux, w),
                Accum::Replace,
                gk.transpose(),
                xa,
                1.0,
                par,
            );
            // out[p on atom] += sum_Q G[p][Q] X_Q
            matmul(
                MatMut::from_row_major_slice_mut(&mut ok[range.start * w..range.end * w], na, w),
                Accum::Add,
                gk,
                xm,
                1.0,
                par,
            );
        }
        out
    }

    /// Two-electron pieces of the closed-shell RI Hessian for occupied orbitals `co`
    /// (D_a = C_o C_o^T, D_t = 2 D_a) and exact-exchange fraction `a_x`.
    pub fn hessian_2e_restricted(
        &self,
        co: MatRef<'_, f64>,
        a_x: f64,
        par: Par,
        timers: &mut Timers,
    ) -> DfHessian2e {
        let (n, naux) = (self.nbf, self.naux);
        let n2 = n * n;
        let o = co.ncols();
        let oo = o * o;
        let n3 = 3 * self.natom;
        let with_k = a_x != 0.0 && o > 0;
        let cof = linus_base::linalg::mat_to_rowmajor(co.as_ref());
        let dt: Vec<f64> = {
            let d = linus_base::linalg::mm(co, co.transpose(), par) * 2.0;
            linus_base::linalg::mat_to_rowmajor(d.as_ref())
        };
        // J fit
        let gamma = self.fit_rhs(&[&dt], par);
        let mut c = gamma.clone();
        self.solve_l(&mut c, 1, par);
        self.solve_lt(&mut c, 1, par);
        // exchange intermediates
        let (bo, zt, x, y) = timers.time("hess_df_setup", || {
            if !with_k {
                return (Vec::new(), Vec::new(), Vec::new(), Vec::new());
            }
            // Q[P][m][i] = (P|m i), B_o = L^-1 Q
            let mut q = vec![0.0; naux * n * o];
            matmul(
                MatMut::from_row_major_slice_mut(&mut q, naux * n, o),
                Accum::Replace,
                MatRef::from_row_major_slice(&self.j3, naux * n, n),
                co,
                1.0,
                par,
            );
            // Λ0[P] = C_o^T Q_P, X = V^-1 Λ0
            let mut x = vec![0.0; naux * oo];
            x.par_chunks_mut(oo).enumerate().for_each(|(p, xp)| {
                let qp = MatRef::from_row_major_slice(&q[p * n * o..(p + 1) * n * o], n, o);
                matmul(
                    MatMut::from_row_major_slice_mut(xp, o, o),
                    Accum::Replace,
                    co.transpose(),
                    qp,
                    1.0,
                    Par::Seq,
                );
            });
            self.solve_l(&mut x, oo, par);
            self.solve_lt(&mut x, oo, par);
            let mut bo = q;
            self.solve_l(&mut bo, n * o, par);
            // Z = V^-1 (P|m i) = L^-T B_o, as [m][P][i]
            let mut z = bo.clone();
            self.solve_lt(&mut z, n * o, par);
            let zt = to_m_major(&z, naux, n, o);
            // Y_P = C_o X_P
            let mut y = vec![0.0; naux * n * o];
            y.par_chunks_mut(n * o).enumerate().for_each(|(p, yp)| {
                let xp = MatRef::from_row_major_slice(&x[p * oo..(p + 1) * oo], o, o);
                matmul(
                    MatMut::from_row_major_slice_mut(yp, n, o),
                    Accum::Replace,
                    co,
                    xp,
                    1.0,
                    Par::Seq,
                );
            });
            (bo, zt, x, y)
        });
        // per-atom first-derivative contractions
        let input = Df3cInput {
            vecs: vec![&c],
            dens: vec![&dt],
            right: if with_k { Some((&y, o)) } else { None },
            orbitals: if with_k { Some((&cof, o)) } else { None },
        };
        let npk = o * (o + 1) / 2;
        let mut rt = vec![0.0; naux * n3]; // L^-1 r^x, naux x n3
                                           // packed R^x, [x][P][ij], of the first `blk` atoms (all of them when they fit in memory)
        let blk = cross_block_atoms(naux * npk, self.natom);
        let mut rp = if with_k {
            vec![0.0; 3 * blk * naux * npk]
        } else {
            Vec::new()
        };
        let mut jxo = vec![Mat::<f64>::zeros(n, o); n3];
        let mut kxo = vec![Mat::<f64>::zeros(n, o); n3];
        for atom in 0..self.natom {
            let res = timers.time("hess_df_3c_deriv", || {
                linus_ints::reference::df3c_deriv_atoms(
                    &self.basis,
                    &self.pairs,
                    &self.aux,
                    &self.apairs,
                    &input,
                    &[atom],
                    self.thresh,
                )
            });
            let a = &res[0];
            timers.time("hess_df_assemble1", || {
                let rows = oneshot::metric_deriv_rows(&self.aux, &self.apairs, atom);
                let vc = linus_ints::reference::apply_metric_deriv(&rows.0, &rows.1, naux, &c, 1);
                let vx = if with_k {
                    self.metric_apply(&rows, &x, oo, par)
                } else {
                    Vec::new()
                };
                for k in 0..3 {
                    let xi = 3 * atom + k;
                    for p in 0..naux {
                        rt[p * n3 + xi] = a.gamma[0][k * naux + p] - vc[k * naux + p];
                    }
                    // M^x C_o
                    let mk = MatRef::from_row_major_slice(&a.m[0][k * n2..(k + 1) * n2], n, n);
                    matmul(jxo[xi].as_mut(), Accum::Replace, mk, co, 1.0, par);
                    if with_k {
                        // R^x = Λ^x - V^x X
                        let mut rk = a.lambda[k * naux * oo..(k + 1) * naux * oo].to_vec();
                        rk.iter_mut()
                            .zip(&vx[k * naux * oo..(k + 1) * naux * oo])
                            .for_each(|(r, v)| *r -= v);
                        // K^x C_o = Y^x + sum_P Z_P R^x_P
                        let kx = &mut kxo[xi];
                        for i in 0..n {
                            for j in 0..o {
                                kx[(i, j)] = a.right[(k * n + i) * o + j];
                            }
                        }
                        // long contraction (naux o), small output: split over threads
                        *kx += abt_split(
                            MatRef::from_row_major_slice(&zt, n, naux * o),
                            MatRef::from_row_major_slice(&rk, naux * o, o).transpose(),
                        );
                        // packed copy (sqrt 2 off the diagonal) for the cross term
                        if atom < blk {
                            pack_tri(&rk, o, &mut rp[xi * naux * npk..(xi + 1) * naux * npk]);
                        }
                    }
                }
            });
        }
        let mut hess = vec![0.0; n3 * n3];
        timers.time("hess_df_assemble2", || {
            self.solve_l(&mut rt, n3, par);
            let rtm = MatRef::from_row_major_slice(&rt, naux, n3);
            // J^x C_o += sum_Q B_o,Q rt^x_Q needs (P|mi): recompute L^-1 (P|m i) if K is off
            let bo_j: Vec<f64> = if with_k {
                Vec::new()
            } else {
                let mut q = vec![0.0; naux * n * o];
                matmul(
                    MatMut::from_row_major_slice_mut(&mut q, naux * n, o),
                    Accum::Replace,
                    MatRef::from_row_major_slice(&self.j3, naux * n, n),
                    co,
                    1.0,
                    par,
                );
                self.solve_l(&mut q, n * o, par);
                q
            };
            let bref = if with_k { &bo } else { &bo_j };
            let mut jadd = vec![0.0; n3 * n * o];
            matmul(
                MatMut::from_row_major_slice_mut(&mut jadd, n3, n * o),
                Accum::Replace,
                rtm.transpose(),
                MatRef::from_row_major_slice(bref, naux, n * o),
                1.0,
                par,
            );
            for xi in 0..n3 {
                for i in 0..n {
                    for j in 0..o {
                        jxo[xi][(i, j)] += jadd[(xi * n + i) * o + j];
                    }
                }
            }
            // cross terms
            matmul(
                MatMut::from_row_major_slice_mut(&mut hess, n3, n3),
                Accum::Replace,
                rtm.transpose(),
                rtm,
                1.0,
                par,
            );
            if with_k {
                // L^-1 R^x (packed), then the Gram matrix with the contraction split over threads;
                // atoms beyond the stored block are recomputed (R^x = Λ^x - V^x X only)
                let lam_in = Df3cInput {
                    vecs: vec![],
                    dens: vec![],
                    right: None,
                    orbitals: Some((&cof, o)),
                };
                let mut recompute = |a0: usize, a1: usize| -> Vec<Vec<f64>> {
                    let mut w = vec![0.0; 3 * (a1 - a0) * naux * npk];
                    for atom in a0..a1 {
                        let res = linus_ints::reference::df3c_deriv_atoms(
                            &self.basis,
                            &self.pairs,
                            &self.aux,
                            &self.apairs,
                            &lam_in,
                            &[atom],
                            self.thresh,
                        );
                        let rows = oneshot::metric_deriv_rows(&self.aux, &self.apairs, atom);
                        let vx = self.metric_apply(&rows, &x, oo, par);
                        for k in 0..3 {
                            let mut rk = res[0].lambda[k * naux * oo..(k + 1) * naux * oo].to_vec();
                            rk.iter_mut()
                                .zip(&vx[k * naux * oo..(k + 1) * naux * oo])
                                .for_each(|(r, v)| *r -= v);
                            let xi = 3 * (atom - a0) + k;
                            pack_tri(&rk, o, &mut w[xi * naux * npk..(xi + 1) * naux * npk]);
                        }
                    }
                    vec![w]
                };
                let g = self.cross_gram(
                    n3,
                    blk,
                    &[naux * npk],
                    vec![std::mem::take(&mut rp)],
                    &mut recompute,
                );
                for i in 0..n3 {
                    for j in 0..n3 {
                        hess[i * n3 + j] -= 2.0 * a_x * g[0][(i, j)];
                    }
                }
            }
        });
        drop(rp);
        // explicit second-derivative integrals
        timers.time("hess_df_2nd_deriv", || {
            let mut gam2 = vec![0.0; naux * naux];
            for p in 0..naux {
                for q in 0..naux {
                    gam2[p * naux + q] = -0.5 * c[p] * c[q];
                }
            }
            if with_k {
                let xm = MatRef::from_row_major_slice(&x, naux, oo);
                matmul(
                    MatMut::from_row_major_slice_mut(&mut gam2, naux, naux),
                    Accum::Add,
                    xm,
                    xm.transpose(),
                    a_x,
                    par,
                );
            }
            let h2 = oneshot::hess_2c(
                &self.aux,
                &self.apairs,
                &gam2,
                1e-14,
                &linus_ints::operator::Operator::Coulomb,
            );
            add_into(&mut hess, &h2);
            // aux-shell batches of Γ_P = -2 a_x C_o X_P C_o^T (<= 2^25 doubles per batch)
            let ash = &self.aux.shells;
            let nsh = ash.len();
            let maxf = if with_k {
                ((1usize << 25) / n2).max(1)
            } else {
                naux
            };
            let mut batches: Vec<std::ops::Range<usize>> = Vec::new();
            let mut s0 = 0;
            while s0 < nsh {
                let f0 = ash[s0].ao_start;
                let mut s1 = s0 + 1;
                while s1 < nsh && ash[s1].ao_start + ash[s1].nfunc() - f0 <= maxf {
                    s1 += 1;
                }
                batches.push(s0..s1);
                s0 = s1;
            }
            for (bi, sb) in batches.into_iter().enumerate() {
                let first = bi == 0;
                let (s0, s1) = (sb.start, sb.end);
                let f0 = ash[s0].ao_start;
                let f1 = if s1 < nsh { ash[s1].ao_start } else { naux };
                let full: Vec<f64> = if with_k {
                    let mut g = vec![0.0; (f1 - f0) * n2];
                    g.par_chunks_mut(n2).enumerate().for_each(|(pi, gp)| {
                        let p = f0 + pi;
                        let yp = MatRef::from_row_major_slice(&y[p * n * o..(p + 1) * n * o], n, o);
                        matmul(
                            MatMut::from_row_major_slice_mut(gp, n, n),
                            Accum::Replace,
                            yp,
                            co.transpose(),
                            -2.0 * a_x,
                            Par::Seq,
                        );
                    });
                    g
                } else {
                    Vec::new()
                };
                let dens = Dens3c {
                    full: if with_k { Some(&full) } else { None },
                    aux_shells: s0..s1,
                    separable: if first {
                        vec![(1.0, &c[..], &dt[..])]
                    } else {
                        vec![]
                    },
                };
                let h3 = linus_ints::reference::eri_hess_3c(
                    &self.basis,
                    &self.pairs,
                    &self.aux,
                    &self.apairs,
                    &dens,
                    1e-14,
                    &linus_ints::operator::Operator::Coulomb,
                );
                add_into(&mut hess, &h3);
            }
        });
        let fxo: Vec<Mat<f64>> = (0..n3)
            .map(|xi| {
                if with_k {
                    &jxo[xi] - &kxo[xi] * a_x
                } else {
                    jxo[xi].clone()
                }
            })
            .collect();
        DfHessian2e { fxo, hess }
    }

    /// Two-electron pieces of the open-shell RI Hessian for the occupied orbitals of both
    /// spins (D_s = C_s C_s^T, D_t = D_a + D_b) and exact-exchange fraction `a_x`: the
    /// closed-shell formulas with the exchange split per spin,
    ///   Γ_P = c_P D_t - a_x sum_s C_s X_s,P C_s^T,  γ_PQ = -1/2 c_P c_Q + a_x/2 sum_s tr(X_s,P X_s,Q),
    ///   cross terms (L^-1 r^x).(L^-1 r^y) - a_x sum_s (L^-1 R_s^x).(L^-1 R_s^y),
    /// and F_s^x C_s = (J^x[D_t] - a_x K^x[D_s]) C_s. The derivative integrals are contracted
    /// once for both spins (orbitals and right factors side by side).
    pub fn hessian_2e_unrestricted(
        &self,
        co: [MatRef<'_, f64>; 2],
        a_x: f64,
        par: Par,
        timers: &mut Timers,
    ) -> DfHessian2eU {
        let (n, naux) = (self.nbf, self.naux);
        let n2 = n * n;
        let os = [co[0].ncols(), co[1].ncols()];
        let oc = os[0] + os[1];
        let off = [0, os[0]];
        let n3 = 3 * self.natom;
        let with_k = a_x != 0.0 && oc > 0;
        // both spins' orbitals side by side, [m][i] row-major
        let cc = Mat::from_fn(n, oc, |m, i| {
            if i < os[0] {
                co[0][(m, i)]
            } else {
                co[1][(m, i - os[0])]
            }
        });
        let cof = linus_base::linalg::mat_to_rowmajor(cc.as_ref());
        let dt: Vec<f64> = {
            let d = linus_base::linalg::mm(cc.as_ref(), cc.transpose(), par);
            linus_base::linalg::mat_to_rowmajor(d.as_ref())
        };
        let gamma = self.fit_rhs(&[&dt], par);
        let mut c = gamma.clone();
        self.solve_l(&mut c, 1, par);
        self.solve_lt(&mut c, 1, par);
        // exchange intermediates per spin: B_o,s = L^-1 (P|m i_s), Z_s (m-major), X_s = V^-1 (P|i_s j_s)
        // and the joint right factor Y[P] = [C_a X_a,P | C_b X_b,P] (n x oc)
        struct SpinK {
            bo: Vec<f64>,
            zt: Vec<f64>,
            x: Vec<f64>,
        }
        let (sk, y): (Vec<SpinK>, Vec<f64>) = timers.time("hess_df_setup", || {
            if !with_k {
                return (Vec::new(), Vec::new());
            }
            let mut y = vec![0.0; naux * n * oc];
            let sk = (0..2)
                .map(|s| {
                    let (o, cs) = (os[s], co[s]);
                    let oo = o * o;
                    let mut q = vec![0.0; naux * n * o];
                    if o > 0 {
                        matmul(
                            MatMut::from_row_major_slice_mut(&mut q, naux * n, o),
                            Accum::Replace,
                            MatRef::from_row_major_slice(&self.j3, naux * n, n),
                            cs,
                            1.0,
                            par,
                        );
                    }
                    let mut x = vec![0.0; naux * oo];
                    if o > 0 {
                        x.par_chunks_mut(oo).enumerate().for_each(|(p, xp)| {
                            let qp =
                                MatRef::from_row_major_slice(&q[p * n * o..(p + 1) * n * o], n, o);
                            matmul(
                                MatMut::from_row_major_slice_mut(xp, o, o),
                                Accum::Replace,
                                cs.transpose(),
                                qp,
                                1.0,
                                Par::Seq,
                            );
                        });
                    }
                    self.solve_l(&mut x, oo, par);
                    self.solve_lt(&mut x, oo, par);
                    let mut bo = q;
                    self.solve_l(&mut bo, n * o, par);
                    let mut z = bo.clone();
                    self.solve_lt(&mut z, n * o, par);
                    let zt = to_m_major(&z, naux, n, o);
                    drop(z);
                    if o > 0 {
                        y.par_chunks_mut(n * oc).enumerate().for_each(|(p, yp)| {
                            let xp = MatRef::from_row_major_slice(&x[p * oo..(p + 1) * oo], o, o);
                            let t = linus_base::linalg::mm(cs, xp, Par::Seq);
                            for m in 0..n {
                                for i in 0..o {
                                    yp[m * oc + off[s] + i] = t[(m, i)];
                                }
                            }
                        });
                    }
                    SpinK { bo, zt, x }
                })
                .collect();
            (sk, y)
        });
        let input = Df3cInput {
            vecs: vec![&c],
            dens: vec![&dt],
            right: if with_k { Some((&y, oc)) } else { None },
            orbitals: if with_k { Some((&cof, oc)) } else { None },
        };
        let npk = [0, 1].map(|s| os[s] * (os[s] + 1) / 2);
        let mut rt = vec![0.0; naux * n3]; // L^-1 r^x, naux x n3
                                           // packed R_s^x of the first `blk` atoms (all of them when they fit in memory)
        let blk = cross_block_atoms(naux * (npk[0] + npk[1]), self.natom);
        let mut rp: [Vec<f64>; 2] = [0, 1].map(|s| {
            if with_k {
                vec![0.0; 3 * blk * naux * npk[s]]
            } else {
                Vec::new()
            }
        });
        let mut jxo: [Vec<Mat<f64>>; 2] = [0, 1].map(|s| vec![Mat::<f64>::zeros(n, os[s]); n3]);
        let mut kxo: [Vec<Mat<f64>>; 2] = [0, 1].map(|s| vec![Mat::<f64>::zeros(n, os[s]); n3]);
        for atom in 0..self.natom {
            let res = timers.time("hess_df_3c_deriv", || {
                linus_ints::reference::df3c_deriv_atoms(
                    &self.basis,
                    &self.pairs,
                    &self.aux,
                    &self.apairs,
                    &input,
                    &[atom],
                    self.thresh,
                )
            });
            let a = &res[0];
            timers.time("hess_df_assemble1", || {
                let rows = oneshot::metric_deriv_rows(&self.aux, &self.apairs, atom);
                let vc = linus_ints::reference::apply_metric_deriv(&rows.0, &rows.1, naux, &c, 1);
                let vx: Vec<Vec<f64>> = if with_k {
                    sk.iter()
                        .zip(os)
                        .map(|(k, o)| self.metric_apply(&rows, &k.x, o * o, par))
                        .collect()
                } else {
                    Vec::new()
                };
                for k in 0..3 {
                    let xi = 3 * atom + k;
                    for p in 0..naux {
                        rt[p * n3 + xi] = a.gamma[0][k * naux + p] - vc[k * naux + p];
                    }
                    let mk = MatRef::from_row_major_slice(&a.m[0][k * n2..(k + 1) * n2], n, n);
                    for s in 0..2 {
                        matmul(jxo[s][xi].as_mut(), Accum::Replace, mk, co[s], 1.0, par);
                    }
                    if !with_k {
                        continue;
                    }
                    for s in 0..2 {
                        let o = os[s];
                        if o == 0 {
                            continue;
                        }
                        let oo = o * o;
                        // R_s^x = Λ_s^x - V^x X_s (the spin's diagonal block of the joint Λ)
                        let lam = &a.lambda[k * naux * oc * oc..(k + 1) * naux * oc * oc];
                        let vxs = &vx[s][k * naux * oo..(k + 1) * naux * oo];
                        let mut rk = vec![0.0; naux * oo];
                        rk.par_chunks_mut(oo).enumerate().for_each(|(p, rp_)| {
                            for i in 0..o {
                                for j in 0..o {
                                    rp_[i * o + j] = lam
                                        [p * oc * oc + (off[s] + i) * oc + off[s] + j]
                                        - vxs[p * oo + i * o + j];
                                }
                            }
                        });
                        // K_s^x C_s = Y_s^x + sum_P Z_s,P R_s^x_P
                        let kx = &mut kxo[s][xi];
                        for i in 0..n {
                            for j in 0..o {
                                kx[(i, j)] = a.right[(k * n + i) * oc + off[s] + j];
                            }
                        }
                        *kx += abt_split(
                            MatRef::from_row_major_slice(&sk[s].zt, n, naux * o),
                            MatRef::from_row_major_slice(&rk, naux * o, o).transpose(),
                        );
                        if atom < blk {
                            pack_tri(
                                &rk,
                                o,
                                &mut rp[s][xi * naux * npk[s]..(xi + 1) * naux * npk[s]],
                            );
                        }
                    }
                }
            });
        }
        let mut hess = vec![0.0; n3 * n3];
        timers.time("hess_df_assemble2", || {
            self.solve_l(&mut rt, n3, par);
            let rtm = MatRef::from_row_major_slice(&rt, naux, n3);
            // J^x C_s += sum_Q B_o,s,Q rt^x_Q
            for s in 0..2 {
                let o = os[s];
                if o == 0 {
                    continue;
                }
                let bo_j: Vec<f64> = if with_k {
                    Vec::new()
                } else {
                    let mut q = vec![0.0; naux * n * o];
                    matmul(
                        MatMut::from_row_major_slice_mut(&mut q, naux * n, o),
                        Accum::Replace,
                        MatRef::from_row_major_slice(&self.j3, naux * n, n),
                        co[s],
                        1.0,
                        par,
                    );
                    self.solve_l(&mut q, n * o, par);
                    q
                };
                let bref = if with_k { &sk[s].bo } else { &bo_j };
                let mut jadd = vec![0.0; n3 * n * o];
                matmul(
                    MatMut::from_row_major_slice_mut(&mut jadd, n3, n * o),
                    Accum::Replace,
                    rtm.transpose(),
                    MatRef::from_row_major_slice(bref, naux, n * o),
                    1.0,
                    par,
                );
                for xi in 0..n3 {
                    for i in 0..n {
                        for j in 0..o {
                            jxo[s][xi][(i, j)] += jadd[(xi * n + i) * o + j];
                        }
                    }
                }
            }
            matmul(
                MatMut::from_row_major_slice_mut(&mut hess, n3, n3),
                Accum::Replace,
                rtm.transpose(),
                rtm,
                1.0,
                par,
            );
            if with_k {
                // atoms beyond the stored block: R_s^x = Λ_s^x - V^x X_s recomputed (joint Λ)
                let lam_in = Df3cInput {
                    vecs: vec![],
                    dens: vec![],
                    right: None,
                    orbitals: Some((&cof, oc)),
                };
                let mut recompute = |a0: usize, a1: usize| -> Vec<Vec<f64>> {
                    let mut w: Vec<Vec<f64>> = (0..2)
                        .map(|s| vec![0.0; 3 * (a1 - a0) * naux * npk[s]])
                        .collect();
                    for atom in a0..a1 {
                        let res = linus_ints::reference::df3c_deriv_atoms(
                            &self.basis,
                            &self.pairs,
                            &self.aux,
                            &self.apairs,
                            &lam_in,
                            &[atom],
                            self.thresh,
                        );
                        let rows = oneshot::metric_deriv_rows(&self.aux, &self.apairs, atom);
                        for s in 0..2 {
                            let o = os[s];
                            if o == 0 {
                                continue;
                            }
                            let oo = o * o;
                            let vx = self.metric_apply(&rows, &sk[s].x, oo, par);
                            for k in 0..3 {
                                let lam =
                                    &res[0].lambda[k * naux * oc * oc..(k + 1) * naux * oc * oc];
                                let mut rk = vec![0.0; naux * oo];
                                rk.par_chunks_mut(oo).enumerate().for_each(|(p, rp_)| {
                                    for i in 0..o {
                                        for j in 0..o {
                                            rp_[i * o + j] = lam
                                                [p * oc * oc + (off[s] + i) * oc + off[s] + j]
                                                - vx[k * naux * oo + p * oo + i * o + j];
                                        }
                                    }
                                });
                                let xi = 3 * (atom - a0) + k;
                                pack_tri(
                                    &rk,
                                    o,
                                    &mut w[s][xi * naux * npk[s]..(xi + 1) * naux * npk[s]],
                                );
                            }
                        }
                    }
                    w
                };
                let first = vec![std::mem::take(&mut rp[0]), std::mem::take(&mut rp[1])];
                let g = self.cross_gram(
                    n3,
                    blk,
                    &[naux * npk[0], naux * npk[1]],
                    first,
                    &mut recompute,
                );
                for gs in &g {
                    for i in 0..n3 {
                        for j in 0..n3 {
                            hess[i * n3 + j] -= a_x * gs[(i, j)];
                        }
                    }
                }
            }
        });
        drop(rp);
        timers.time("hess_df_2nd_deriv", || {
            let mut gam2 = vec![0.0; naux * naux];
            for p in 0..naux {
                for q in 0..naux {
                    gam2[p * naux + q] = -0.5 * c[p] * c[q];
                }
            }
            if with_k {
                for (k, o) in sk.iter().zip(os) {
                    let xm = MatRef::from_row_major_slice(&k.x, naux, o * o);
                    matmul(
                        MatMut::from_row_major_slice_mut(&mut gam2, naux, naux),
                        Accum::Add,
                        xm,
                        xm.transpose(),
                        0.5 * a_x,
                        par,
                    );
                }
            }
            let h2 = oneshot::hess_2c(
                &self.aux,
                &self.apairs,
                &gam2,
                1e-14,
                &linus_ints::operator::Operator::Coulomb,
            );
            add_into(&mut hess, &h2);
            // aux-shell batches of Γ_P = -a_x sum_s C_s X_s,P C_s^T = -a_x Y_P C^T (joint factors)
            let ash = &self.aux.shells;
            let nsh = ash.len();
            let maxf = if with_k {
                ((1usize << 25) / n2).max(1)
            } else {
                naux
            };
            let mut batches: Vec<std::ops::Range<usize>> = Vec::new();
            let mut s0 = 0;
            while s0 < nsh {
                let f0 = ash[s0].ao_start;
                let mut s1 = s0 + 1;
                while s1 < nsh && ash[s1].ao_start + ash[s1].nfunc() - f0 <= maxf {
                    s1 += 1;
                }
                batches.push(s0..s1);
                s0 = s1;
            }
            for (bi, sb) in batches.into_iter().enumerate() {
                let first = bi == 0;
                let (s0, s1) = (sb.start, sb.end);
                let f0 = ash[s0].ao_start;
                let f1 = if s1 < nsh { ash[s1].ao_start } else { naux };
                let full: Vec<f64> = if with_k {
                    let mut g = vec![0.0; (f1 - f0) * n2];
                    g.par_chunks_mut(n2).enumerate().for_each(|(pi, gp)| {
                        let p = f0 + pi;
                        let yp =
                            MatRef::from_row_major_slice(&y[p * n * oc..(p + 1) * n * oc], n, oc);
                        matmul(
                            MatMut::from_row_major_slice_mut(gp, n, n),
                            Accum::Replace,
                            yp,
                            cc.transpose(),
                            -a_x,
                            Par::Seq,
                        );
                    });
                    g
                } else {
                    Vec::new()
                };
                let dens = Dens3c {
                    full: if with_k { Some(&full) } else { None },
                    aux_shells: s0..s1,
                    separable: if first {
                        vec![(1.0, &c[..], &dt[..])]
                    } else {
                        vec![]
                    },
                };
                let h3 = linus_ints::reference::eri_hess_3c(
                    &self.basis,
                    &self.pairs,
                    &self.aux,
                    &self.apairs,
                    &dens,
                    1e-14,
                    &linus_ints::operator::Operator::Coulomb,
                );
                add_into(&mut hess, &h3);
            }
        });
        let fxo: Vec<[Mat<f64>; 2]> = (0..n3)
            .map(|xi| {
                [0, 1].map(|s| {
                    if with_k {
                        &jxo[s][xi] - &kxo[s][xi] * a_x
                    } else {
                        jxo[s][xi].clone()
                    }
                })
            })
            .collect();
        DfHessian2eU { fxo, hess }
    }
}

/// Packed lower triangle (diagonal, then sqrt 2 times the off-diagonal elements of each row) of
/// the naux symmetric o x o blocks of `src` ([P][i][j]) into `dst` ([P][o(o+1)/2]): dot
/// products of packed blocks are full Frobenius products.
fn pack_tri(src: &[f64], o: usize, dst: &mut [f64]) {
    let npk = o * (o + 1) / 2;
    let sq2 = std::f64::consts::SQRT_2;
    dst.par_chunks_mut(npk).enumerate().for_each(|(p, dp)| {
        let s = &src[p * o * o..(p + 1) * o * o];
        let mut t = 0;
        for i in 0..o {
            dp[t] = s[i * o + i];
            t += 1;
            for j in 0..i {
                dp[t] = sq2 * s[i * o + j];
                t += 1;
            }
        }
    });
}

/// Atoms whose exchange cross-term rows (three perturbations per atom, `per_x` doubles each) are
/// kept in memory at once: all of them when they fit in 3000 MB, else as many as fit (at least
/// one); the others are recomputed and streamed by [`DfDerivJk::cross_gram`]. The stored rows
/// grow as N^4 (3N x naux x o^2/2).
fn cross_block_atoms(per_x: usize, natom: usize) -> usize {
    let mb = 3000.0;
    let per_atom = (3 * per_x * 8) as f64 / 1e6;
    if per_atom * natom as f64 <= mb {
        natom
    } else {
        ((mb / per_atom) as usize).clamp(1, natom)
    }
}

impl DfDerivJk {
    /// Exchange cross terms sum_s W_s W_s^T, W_s = L^-1 R_s (rows: the 3N perturbations, `rows[s]`
    /// doubles each), with only one block of `blk` atoms in memory: `first` holds the first
    /// block's R_s (not yet multiplied by L^-1), `recompute(a0, a1)` gives R_s for atoms a0..a1.
    /// Each outer block is paired with every later atom streamed one at a time, so the
    /// recomputation is about (natom / blk - 1) / 2 times the R of all atoms; with everything in
    /// memory (blk = natom) nothing is recomputed. Returns one n3 x n3 matrix per spin.
    fn cross_gram(
        &self,
        n3: usize,
        blk: usize,
        rows: &[usize],
        first: Vec<Vec<f64>>,
        recompute: &mut dyn FnMut(usize, usize) -> Vec<Vec<f64>>,
    ) -> Vec<Mat<f64>> {
        let natom = n3 / 3;
        let naux = self.naux;
        let mut g: Vec<Mat<f64>> = rows.iter().map(|_| Mat::zeros(n3, n3)).collect();
        // one chunk per perturbation; a streamed atom has only 3, so each chunk's solve gets its
        // share of the threads (it ran on 3 of 8 threads, ts61: 2 h 43 min)
        let threads = rayon::current_num_threads();
        let linv = |w: &mut [Vec<f64>]| {
            for (ws, &r) in w.iter_mut().zip(rows) {
                if let Some(nch) = ws.len().checked_div(r) {
                    let inner = if nch >= threads {
                        Par::Seq
                    } else {
                        Par::rayon(threads.div_ceil(nch.max(1)))
                    };
                    ws.par_chunks_mut(r)
                        .for_each(|x| self.solve_l(x, r / naux, inner));
                }
            }
        };
        let mut first = Some(first);
        let mut a0 = 0;
        while a0 < natom {
            let a1 = (a0 + blk).min(natom);
            let mut wb = match first.take() {
                Some(f) => f,
                None => recompute(a0, a1),
            };
            linv(&mut wb);
            for (s, &r) in rows.iter().enumerate() {
                if r == 0 {
                    continue;
                }
                let mb = MatRef::from_row_major_slice(&wb[s], 3 * (a1 - a0), r);
                let gbb = abt_split(mb, mb);
                for i in 0..3 * (a1 - a0) {
                    for j in 0..3 * (a1 - a0) {
                        g[s][(3 * a0 + i, 3 * a0 + j)] = gbb[(i, j)];
                    }
                }
            }
            for c in a1..natom {
                let mut wc = recompute(c, c + 1);
                linv(&mut wc);
                for (s, &r) in rows.iter().enumerate() {
                    if r == 0 {
                        continue;
                    }
                    let mb = MatRef::from_row_major_slice(&wb[s], 3 * (a1 - a0), r);
                    let mc = MatRef::from_row_major_slice(&wc[s], 3, r);
                    let gbc = abt_split(mb, mc);
                    for i in 0..3 * (a1 - a0) {
                        for j in 0..3 {
                            g[s][(3 * a0 + i, 3 * c + j)] = gbc[(i, j)];
                            g[s][(3 * c + j, 3 * a0 + i)] = gbc[(i, j)];
                        }
                    }
                }
            }
            a0 = a1;
        }
        g
    }
}

/// Two-electron pieces of a density-fitted open-shell Hessian.
pub struct DfHessian2eU {
    /// (J^x[D_t] - a_x K^x[D_s]) C_s for every perturbation and spin (nbf x nocc_s).
    pub fxo: Vec<[Mat<f64>; 2]>,
    /// Second derivative of the RI two-electron energy at fixed orbitals, (3N)^2 row-major.
    pub hess: Vec<f64>,
}
