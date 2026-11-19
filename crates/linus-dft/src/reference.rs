//! Reference implementations of the XC quadrature as it was before the batch engine of
//! [`crate`] (kept to check the optimized code: same grid, results agree to roundoff). Not
//! used in production.

use crate::hessian::{any_above, cm, cm_mut, comp_index, scatter_sym, zeroed, YBIG_MAX};
use crate::points::{mul_add, mul_set, weighted_derivs, xc_input};
use crate::{select_shells, XcIntegrator};
use faer::{Accum, Mat, MatRef, Par};
use linus_base::linalg::matmul;
use linus_ints::points::ao::{eval_ao, ncomp};
use linus_xc::{Family, XcInput, XcOutput};
use rayon::prelude::*;
use std::sync::Mutex;

impl XcIntegrator {
    /// Reference (pre-optimization) XC energy and potential matrices.
    #[allow(clippy::needless_range_loop, reason = "indexes several arrays in step")]
    #[allow(clippy::type_complexity, reason = "a one-off tuple of buffers")]
    pub fn compute_ref(
        &mut self,
        da: &Mat<f64>,
        db: &Mat<f64>,
        restricted: bool,
    ) -> (f64, Mat<f64>, Mat<f64>) {
        let basis = &self.basis;
        let nbf = basis.nbf;
        let grid = &self.grid;
        let func = if restricted {
            &self.func_u
        } else {
            &self.func_p
        };
        let fam = func.family();
        let gga = fam >= Family::Gga;
        let mgga = fam == Family::Mgga && func.needs_tau();
        let order = if gga { 1 } else { 0 };
        let nspin = if restricted { 1 } else { 2 };
        let ncomp = if gga { 4 } else { 1 };
        let dt: Mat<f64> = da + db;
        let flat =
            |m: &Mat<f64>| -> Vec<f64> { (0..nbf * nbf).map(|x| m[(x % nbf, x / nbf)]).collect() };
        let dens_flat: Vec<Vec<f64>> = if restricted {
            vec![flat(&dt)]
        } else {
            vec![flat(da), flat(db)]
        };
        // the shells and functions that reach each batch
        let (bsh, bidx): (Vec<Vec<usize>>, Vec<Vec<usize>>) = self
            .batch_info()
            .iter()
            .map(|b| {
                let mut idx = Vec::new();
                for &s in &b.shells {
                    let x = &basis.shells[s];
                    idx.extend(x.ao_start..x.ao_start + x.nfunc());
                }
                (b.shells.clone(), idx)
            })
            .unzip();
        let nb = grid.batches.len();
        let fthresh = self.func_thresh;
        // one accumulator per worker thread (uncontended locks), summed in thread order
        let nthr = rayon::current_num_threads();
        let accs: Vec<std::sync::Mutex<(f64, Vec<Vec<f64>>, BatchWork)>> = (0..nthr)
            .map(|_| {
                std::sync::Mutex::new((
                    0.0,
                    vec![vec![0.0f64; nbf * nbf]; nspin],
                    BatchWork::default(),
                ))
            })
            .collect();
        (0..nb).into_par_iter().for_each(|bi| {
            let mut lap = crate::profile::Lap::start();
            let tid = rayon::current_thread_index().unwrap_or(0) % nthr;
            let mut guard = accs[tid].lock().unwrap();
            let (eacc, vacc, wk) = &mut *guard;
            let b = &grid.batches[bi];
            let np = b.end - b.start;
            let shells = &bsh[bi];
            if shells.is_empty() {
                return;
            }
            let idx_all = &bidx[bi];
            let nao_all = idx_all.len();
            let (x, y, z, w) = (
                &grid.x[b.start..b.end],
                &grid.y[b.start..b.end],
                &grid.z[b.start..b.end],
                &grid.w[b.start..b.end],
            );
            wk.ao_raw.resize(ncomp * nao_all * np, 0.0);
            lap.lap(crate::profile::SETUP);
            eval_ao(basis, shells, x, y, z, order, &mut wk.ao_raw);
            lap.lap(crate::profile::COLLOC);
            // function screening: keep functions whose value or gradient exceeds fthresh in this batch
            wk.keep.clear();
            for mu in 0..nao_all {
                let mut m: f64 = 0.0;
                for k in 0..ncomp {
                    for &v in &wk.ao_raw[(k * nao_all + mu) * np..(k * nao_all + mu + 1) * np] {
                        m = m.max(v.abs());
                    }
                }
                if m > fthresh {
                    wk.keep.push(mu);
                }
            }
            let nao = wk.keep.len();
            if nao == 0 {
                return;
            }
            wk.idx.clear();
            wk.idx.extend(wk.keep.iter().map(|&mu| idx_all[mu]));
            wk.ao.resize(ncomp * nao * np, 0.0);
            for k in 0..ncomp {
                for (j, &mu) in wk.keep.iter().enumerate() {
                    let (src, dst) = ((k * nao_all + mu) * np, (k * nao + j) * np);
                    wk.ao[dst..dst + np].copy_from_slice(&wk.ao_raw[src..src + np]);
                }
            }
            lap.lap(crate::profile::SCREEN);
            let idx = &wk.idx;
            let ao = &wk.ao;
            let phis = |k: usize, mu: usize| -> &[f64] {
                &ao[(k * nao + mu) * np..(k * nao + mu + 1) * np]
            };
            let phim = |k: usize| {
                MatRef::from_column_major_slice(&ao[k * nao * np..(k + 1) * nao * np], np, nao)
            };
            // densities
            wk.rho.clear();
            wk.rho.resize(nspin * np, 0.0);
            wk.grad.clear();
            wk.grad.resize(nspin * 3 * np, 0.0);
            wk.tau.clear();
            wk.tau.resize(nspin * np, 0.0);
            wk.dsub.resize(nao * nao, 0.0);
            wk.xm.resize(nspin * nao * np, 0.0);
            for s in 0..nspin {
                let d = &dens_flat[s];
                for j in 0..nao {
                    let row = &d[idx[j] * nbf..(idx[j] + 1) * nbf];
                    for i in 0..nao {
                        wk.dsub[j * nao + i] = row[idx[i]];
                    }
                }
                let dm = MatRef::from_column_major_slice(&wk.dsub, nao, nao);
                {
                    let xmut = faer::MatMut::from_column_major_slice_mut(
                        &mut wk.xm[s * nao * np..(s + 1) * nao * np],
                        np,
                        nao,
                    );
                    matmul(xmut, Accum::Replace, phim(0), dm, 1.0, Par::Seq);
                }
                let xs = &wk.xm[s * nao * np..(s + 1) * nao * np];
                let rho = &mut wk.rho[s * np..(s + 1) * np];
                for mu in 0..nao {
                    let (p0, xc) = (phis(0, mu), &xs[mu * np..(mu + 1) * np]);
                    for g in 0..np {
                        rho[g] += p0[g] * xc[g];
                    }
                }
                if gga {
                    for k in 0..3 {
                        let gk = &mut wk.grad[(s * 3 + k) * np..(s * 3 + k + 1) * np];
                        for mu in 0..nao {
                            let (pk, xc) = (phis(k + 1, mu), &xs[mu * np..(mu + 1) * np]);
                            for g in 0..np {
                                gk[g] += 2.0 * pk[g] * xc[g];
                            }
                        }
                    }
                }
                if mgga {
                    wk.ym.resize(nao * np, 0.0);
                    for k in 0..3 {
                        {
                            let ymut =
                                faer::MatMut::from_column_major_slice_mut(&mut wk.ym, np, nao);
                            matmul(ymut, Accum::Replace, phim(k + 1), dm, 1.0, Par::Seq);
                        }
                        let tau = &mut wk.tau[s * np..(s + 1) * np];
                        for mu in 0..nao {
                            let (pk, yc) = (phis(k + 1, mu), &wk.ym[mu * np..(mu + 1) * np]);
                            for g in 0..np {
                                tau[g] += 0.5 * pk[g] * yc[g];
                            }
                        }
                    }
                }
            }
            lap.lap(crate::profile::DENSITY);
            // libxc layout
            let grad = &wk.grad;
            let gr = |s: usize, k: usize, g: usize| grad[(s * 3 + k) * np + g];
            if restricted {
                wk.rin.clear();
                wk.rin.extend_from_slice(&wk.rho[..np]);
                wk.sin.clear();
                if gga {
                    wk.sin.extend(
                        (0..np).map(|g| {
                            gr(0, 0, g).powi(2) + gr(0, 1, g).powi(2) + gr(0, 2, g).powi(2)
                        }),
                    );
                }
                wk.tin.clear();
                wk.tin.extend_from_slice(&wk.tau[..np]);
            } else {
                wk.rin.resize(2 * np, 0.0);
                wk.sin.resize(3 * np, 0.0);
                wk.tin.resize(2 * np, 0.0);
                for g in 0..np {
                    wk.rin[2 * g] = wk.rho[g];
                    wk.rin[2 * g + 1] = wk.rho[np + g];
                    let dot = |a: usize, c: usize| {
                        gr(a, 0, g) * gr(c, 0, g)
                            + gr(a, 1, g) * gr(c, 1, g)
                            + gr(a, 2, g) * gr(c, 2, g)
                    };
                    if gga {
                        wk.sin[3 * g] = dot(0, 0);
                        wk.sin[3 * g + 1] = dot(0, 1);
                        wk.sin[3 * g + 2] = dot(1, 1);
                    }
                    wk.tin[2 * g] = wk.tau[g];
                    wk.tin[2 * g + 1] = wk.tau[np + g];
                }
            }
            let input = if mgga {
                XcInput::mgga(&wk.rin, &wk.sin, &wk.tin)
            } else if gga {
                XcInput::gga(&wk.rin, &wk.sin)
            } else {
                XcInput::lda(&wk.rin)
            };
            func.eval(1, &input, &mut wk.out)
                .expect("functional evaluation");
            lap.lap(crate::profile::LIBXC);
            let zk = wk.out.zk();
            let vrho = wk.out.vrho();
            for g in 0..np {
                let rt = if restricted {
                    wk.rho[g]
                } else {
                    wk.rho[g] + wk.rho[np + g]
                };
                *eacc += w[g] * rt * zk[g];
            }
            // potential per spin
            wk.wm.resize(nao * np, 0.0);
            wk.vsub.resize(nao * nao, 0.0);
            let mut vr = vec![0.0; np];
            let mut vg = vec![0.0; 3 * np];
            for s in 0..nspin {
                for g in 0..np {
                    vr[g] = 0.5 * w[g] * if restricted { vrho[g] } else { vrho[2 * g + s] };
                }
                if gga {
                    let vs = wk.out.vsigma();
                    for k in 0..3 {
                        for g in 0..np {
                            vg[k * np + g] = w[g]
                                * if restricted {
                                    2.0 * vs[g] * gr(0, k, g)
                                } else {
                                    2.0 * vs[3 * g + 2 * s] * gr(s, k, g)
                                        + vs[3 * g + 1] * gr(1 - s, k, g)
                                };
                        }
                    }
                }
                for mu in 0..nao {
                    let wc = &mut wk.wm[mu * np..(mu + 1) * np];
                    let p0 = phis(0, mu);
                    for g in 0..np {
                        wc[g] = vr[g] * p0[g];
                    }
                    if gga {
                        let (p1, p2, p3) = (phis(1, mu), phis(2, mu), phis(3, mu));
                        for g in 0..np {
                            wc[g] += vg[g] * p1[g] + vg[np + g] * p2[g] + vg[2 * np + g] * p3[g];
                        }
                    }
                }
                lap.lap(crate::profile::WEIGHTS);
                {
                    let wmr = MatRef::from_column_major_slice(&wk.wm, np, nao);
                    let vm = faer::MatMut::from_column_major_slice_mut(&mut wk.vsub, nao, nao);
                    matmul(vm, Accum::Replace, phim(0).transpose(), wmr, 1.0, Par::Seq);
                }
                if mgga {
                    let vtau = wk.out.vtau();
                    for k in 0..3 {
                        for mu in 0..nao {
                            let wc = &mut wk.wm[mu * np..(mu + 1) * np];
                            let pk = phis(k + 1, mu);
                            for g in 0..np {
                                // 1/4 here and the symmetrization below give 1/2 vtau
                                wc[g] = 0.25
                                    * w[g]
                                    * (if restricted { vtau[g] } else { vtau[2 * g + s] })
                                    * pk[g];
                            }
                        }
                        let wmr = MatRef::from_column_major_slice(&wk.wm, np, nao);
                        let vm = faer::MatMut::from_column_major_slice_mut(&mut wk.vsub, nao, nao);
                        matmul(vm, Accum::Add, phim(k + 1).transpose(), wmr, 1.0, Par::Seq);
                    }
                }
                lap.lap(crate::profile::GEMM_V);
                // V += vsub + vsub^T, scattered (lower triangle, mirrored at the end)
                let acc = &mut vacc[s];
                for j in 0..nao {
                    let cj = idx[j] * nbf;
                    for i in j..nao {
                        acc[cj + idx[i]] += wk.vsub[j * nao + i] + wk.vsub[i * nao + j];
                    }
                }
                lap.lap(crate::profile::SCATTER);
            }
        });
        let mut lap = crate::profile::Lap::start();
        let mut exc = 0.0;
        let mut vacc = vec![vec![0.0f64; nbf * nbf]; nspin];
        for m in accs {
            let (e, v, _) = m.into_inner().unwrap();
            exc += e;
            for (a, b) in vacc.iter_mut().zip(&v) {
                a.iter_mut().zip(b).for_each(|(x, y)| *x += y);
            }
        }
        // batch indices are ascending, so the scatter filled only the lower triangle
        // (row >= column); mirror it.
        for v in vacc.iter_mut() {
            for j in 0..nbf {
                for i in 0..j {
                    let s = v[j * nbf + i] + v[i * nbf + j];
                    v[j * nbf + i] = s;
                    v[i * nbf + j] = s;
                }
            }
        }
        let va = Mat::from_fn(nbf, nbf, |i, j| vacc[0][j * nbf + i]);
        let vb = if restricted {
            va.clone()
        } else {
            Mat::from_fn(nbf, nbf, |i, j| vacc[1][j * nbf + i])
        };
        lap.lap(crate::profile::REDUCE);
        (exc, va, vb)
    }

    /// Reference (pre-optimization) XC nuclear gradient.
    #[allow(clippy::needless_range_loop, reason = "indexes several arrays in step")]
    #[allow(clippy::type_complexity, reason = "a one-off tuple of buffers")]
    pub fn gradient_ref(
        &self,
        da: &Mat<f64>,
        db: &Mat<f64>,
        restricted: bool,
        grid_response: bool,
    ) -> Vec<[f64; 3]> {
        let basis = &self.basis;
        let grid = &self.grid;
        let natom = grid.atom_z.len();
        let func = if restricted {
            &self.func_u
        } else {
            &self.func_p
        };
        let fam = func.family();
        let gga = fam >= Family::Gga;
        let mgga = fam == Family::Mgga && func.needs_tau();
        let order = if gga { 2 } else { 1 };
        let nspin = if restricted { 1 } else { 2 };
        let dt: Mat<f64> = da + db;
        let nbf = basis.nbf;
        let flat =
            |m: &Mat<f64>| -> Vec<f64> { (0..nbf * nbf).map(|x| m[(x % nbf, x / nbf)]).collect() };
        let dens_flat: Vec<Vec<f64>> = if restricted {
            vec![flat(&dt)]
        } else {
            vec![flat(da), flat(db)]
        };
        let extents = &self.extents;
        let npts_all = grid.x.len();
        // per batch: (atom gradient contributions, (point index, energy density, per-point translation sum))
        let parts: Vec<(Vec<[f64; 3]>, Vec<(usize, f64, [f64; 3])>)> = grid
            .batches
            .par_iter()
            .map(|b| {
                let mut lap = crate::profile::Lap::start();
                let mut gat = vec![[0.0; 3]; natom];
                let np = b.end - b.start;
                let (x, y, z, w) = (
                    &grid.x[b.start..b.end],
                    &grid.y[b.start..b.end],
                    &grid.z[b.start..b.end],
                    &grid.w[b.start..b.end],
                );
                let shells = select_shells(basis, extents, b.center, b.radius);
                if shells.is_empty() {
                    return (gat, vec![]);
                }
                let mut idx = Vec::new();
                let mut owner = Vec::new();
                for &s in &shells {
                    let sh = &basis.shells[s];
                    idx.extend(sh.ao_start..sh.ao_start + sh.nfunc());
                    owner.extend(std::iter::repeat_n(sh.atom, sh.nfunc()));
                }
                let nao = idx.len();
                let nc = linus_ints::points::ao::ncomp(order);
                let mut ao = vec![0.0; nc * nao * np];
                lap.lap(crate::profile::SETUP);
                eval_ao(basis, &shells, x, y, z, order, &mut ao);
                lap.lap(crate::profile::COLLOC);
                let phi = |k: usize| {
                    MatRef::from_column_major_slice(&ao[k * nao * np..(k + 1) * nao * np], np, nao)
                };
                // second-derivative component index for (i, k): xx xy xz yy yz zz at 4..10
                let d2 = |i: usize, k: usize| -> usize {
                    let (a, b) = if i <= k { (i, k) } else { (k, i) };
                    4 + match (a, b) {
                        (0, 0) => 0,
                        (0, 1) => 1,
                        (0, 2) => 2,
                        (1, 1) => 3,
                        (1, 2) => 4,
                        _ => 5,
                    }
                };
                let dens = &dens_flat;
                // X_s = phi D_s, Y_sk = dphi_k D_s
                let mut xs = Vec::new();
                let mut ys: Vec<Vec<Mat<f64>>> = Vec::new();
                let mut rho = vec![vec![0.0; np]; nspin];
                let mut grad = vec![[vec![0.0; np], vec![0.0; np], vec![0.0; np]]; nspin];
                let mut tau = vec![vec![0.0; np]; nspin];
                for s in 0..nspin {
                    let dsub = Mat::from_fn(nao, nao, |i, j| dens[s][idx[j] * nbf + idx[i]]);
                    let xm = linus_base::linalg::mm(phi(0), dsub.as_ref(), Par::Seq);
                    let mut yk = Vec::new();
                    for k in 0..3 {
                        yk.push(linus_base::linalg::mm(phi(k + 1), dsub.as_ref(), Par::Seq));
                    }
                    lap.lap(crate::profile::GEMM_D);
                    for mu in 0..nao {
                        for g in 0..np {
                            rho[s][g] += phi(0)[(g, mu)] * xm[(g, mu)];
                            for k in 0..3 {
                                grad[s][k][g] += 2.0 * phi(k + 1)[(g, mu)] * xm[(g, mu)];
                                tau[s][g] += 0.5 * phi(k + 1)[(g, mu)] * yk[k][(g, mu)];
                            }
                        }
                    }
                    xs.push(xm);
                    ys.push(yk);
                    lap.lap(crate::profile::DENSITY);
                }
                let (rho_in, sigma_in, tau_in): (Vec<f64>, Vec<f64>, Vec<f64>) = if restricted {
                    let sig = (0..np)
                        .map(|g| {
                            grad[0][0][g].powi(2) + grad[0][1][g].powi(2) + grad[0][2][g].powi(2)
                        })
                        .collect();
                    (rho[0].clone(), sig, tau[0].clone())
                } else {
                    let mut r = vec![0.0; 2 * np];
                    let mut sg = vec![0.0; 3 * np];
                    let mut t = vec![0.0; 2 * np];
                    for g in 0..np {
                        r[2 * g] = rho[0][g];
                        r[2 * g + 1] = rho[1][g];
                        let dot = |a: usize, c: usize| {
                            grad[a][0][g] * grad[c][0][g]
                                + grad[a][1][g] * grad[c][1][g]
                                + grad[a][2][g] * grad[c][2][g]
                        };
                        sg[3 * g] = dot(0, 0);
                        sg[3 * g + 1] = dot(0, 1);
                        sg[3 * g + 2] = dot(1, 1);
                        t[2 * g] = tau[0][g];
                        t[2 * g + 1] = tau[1][g];
                    }
                    (r, sg, t)
                };
                let input = if mgga {
                    XcInput::mgga(&rho_in, &sigma_in, &tau_in)
                } else if gga {
                    XcInput::gga(&rho_in, &sigma_in)
                } else {
                    XcInput::lda(&rho_in)
                };
                let mut out = XcOutput::new();
                func.eval(1, &input, &mut out)
                    .expect("functional evaluation");
                lap.lap(crate::profile::LIBXC);
                let (zk, vrho) = (out.zk(), out.vrho());
                // per-point translation sums (sum over all functions) for the point-motion term
                let mut tsum = vec![[0.0f64; 3]; np];
                for s in 0..nspin {
                    let mut wv = vec![0.0; np];
                    let mut vk = [vec![0.0; np], vec![0.0; np], vec![0.0; np]];
                    let mut vt = vec![0.0; np];
                    for g in 0..np {
                        wv[g] = w[g] * if restricted { vrho[g] } else { vrho[2 * g + s] };
                        if gga {
                            let vs = out.vsigma();
                            for k in 0..3 {
                                vk[k][g] = w[g]
                                    * if restricted {
                                        2.0 * vs[g] * grad[0][k][g]
                                    } else {
                                        2.0 * vs[3 * g + 2 * s] * grad[s][k][g]
                                            + vs[3 * g + 1] * grad[1 - s][k][g]
                                    };
                            }
                        }
                        if mgga {
                            let vtau = out.vtau();
                            vt[g] = w[g] * if restricted { vtau[g] } else { vtau[2 * g + s] };
                        }
                    }
                    let xm = &xs[s];
                    let yk = &ys[s];
                    for mu in 0..nao {
                        let a = owner[mu];
                        for g in 0..np {
                            let xv = xm[(g, mu)];
                            let mut bsc = wv[g] * xv;
                            if gga {
                                bsc += vk[0][g] * yk[0][(g, mu)]
                                    + vk[1][g] * yk[1][(g, mu)]
                                    + vk[2][g] * yk[2][(g, mu)];
                            }
                            for i in 0..3 {
                                let mut c = phi(i + 1)[(g, mu)] * bsc;
                                if gga {
                                    for k in 0..3 {
                                        c += phi(d2(i, k))[(g, mu)] * vk[k][g] * xv;
                                    }
                                }
                                if mgga {
                                    for k in 0..3 {
                                        c += 0.5 * vt[g] * phi(d2(i, k))[(g, mu)] * yk[k][(g, mu)];
                                    }
                                }
                                gat[a][i] -= 2.0 * c;
                                tsum[g][i] -= 2.0 * c;
                            }
                        }
                    }
                }
                let pts: Vec<(usize, f64, [f64; 3])> = (0..np)
                    .map(|g| {
                        let rt: f64 = (0..nspin).map(|s| rho[s][g]).sum();
                        (b.start + g, rt * zk[g], tsum[g])
                    })
                    .collect();
                lap.lap(crate::profile::OTHER);
                (gat, pts)
            })
            .collect();
        let mut lap = crate::profile::Lap::start();
        let mut gtot = vec![[0.0; 3]; natom];
        let mut eps = vec![0.0; npts_all];
        for (gat, pts) in &parts {
            for a in 0..natom {
                for i in 0..3 {
                    gtot[a][i] += gat[a][i];
                }
            }
            if grid_response {
                for &(g, e, t) in pts {
                    eps[g] = e;
                    let own = grid.atom[g] as usize;
                    for i in 0..3 {
                        gtot[own][i] -= t[i];
                    }
                }
            }
        }
        lap.lap(crate::profile::REDUCE);
        if grid_response {
            let gw = grid.weight_gradient_contract_pointwise(&eps);
            for a in 0..natom {
                for i in 0..3 {
                    gtot[a][i] += gw[a][i];
                }
            }
        }
        lap.lap(crate::profile::GRID_RESP);
        gtot
    }

    /// Reference (pre-optimization) kernel contraction (restricted or unrestricted).
    #[allow(clippy::needless_range_loop, reason = "indexes several arrays in step")]
    pub fn kernel_contract_ref(
        &self,
        da: &Mat<f64>,
        db: &Mat<f64>,
        restricted: bool,
        d1: &[(Mat<f64>, Mat<f64>)],
    ) -> Vec<(Mat<f64>, Mat<f64>)> {
        let basis = &self.basis;
        let nbf = basis.nbf;
        let grid = &self.grid;
        if restricted {
            let d0: Mat<f64> = da + db;
            let d1t: Vec<Mat<f64>> = d1.iter().map(|p| p.0.clone()).collect();
            return self
                .kernel_contract_restricted_ref(&d0, &d1t)
                .into_iter()
                .map(|v| (v.clone(), v))
                .collect();
        }
        let func = &self.func_p;
        let fam = func.family();
        assert!(
            fam != Family::Mgga,
            "open-shell meta-GGA kernels are not implemented yet"
        );
        let gga = fam >= Family::Gga;
        let order = if gga { 1 } else { 0 };
        let ncomp = if gga { 4 } else { 1 };
        let nspin = if restricted { 1 } else { 2 };
        let npert = d1.len();
        let dt: Mat<f64> = da + db;
        let d0: Vec<&Mat<f64>> = if restricted { vec![&dt] } else { vec![da, db] };
        let extents = &self.extents;
        // fixed-point accumulators (one set per thread): the sum does not depend on the schedule
        let taccs: linus_num::det::ThreadAccs<Vec<Vec<linus_num::det::FixedAcc>>> =
            linus_num::det::ThreadAccs::new();
        grid.batches.par_iter().for_each(|b| {
            taccs.with(
                || {
                    (0..npert)
                        .map(|_| {
                            (0..nspin)
                                .map(|_| linus_num::det::FixedAcc::new(nbf * nbf))
                                .collect()
                        })
                        .collect()
                },
                |acc| {
                    let np = b.end - b.start;
                    let (x, y, z, w) = (
                        &grid.x[b.start..b.end],
                        &grid.y[b.start..b.end],
                        &grid.z[b.start..b.end],
                        &grid.w[b.start..b.end],
                    );
                    let shells = select_shells(basis, extents, b.center, b.radius);
                    if shells.is_empty() {
                        return;
                    }
                    let mut idx = Vec::new();
                    for &s in &shells {
                        let sh = &basis.shells[s];
                        idx.extend(sh.ao_start..sh.ao_start + sh.nfunc());
                    }
                    let nao = idx.len();
                    let mut ao = vec![0.0; ncomp * nao * np];
                    eval_ao(basis, &shells, x, y, z, order, &mut ao);
                    let phi = |k: usize| {
                        MatRef::from_column_major_slice(
                            &ao[k * nao * np..(k + 1) * nao * np],
                            np,
                            nao,
                        )
                    };
                    // density and gradient for a set of spin densities
                    let dens_on_grid = |d: &Mat<f64>| -> (Vec<f64>, [Vec<f64>; 3]) {
                        let dsub = Mat::from_fn(nao, nao, |i, j| d[(idx[i], idx[j])]);
                        let xm = linus_base::linalg::mm(phi(0), dsub.as_ref(), Par::Seq);
                        let mut r = vec![0.0; np];
                        let mut g = [vec![0.0; np], vec![0.0; np], vec![0.0; np]];
                        for mu in 0..nao {
                            for p in 0..np {
                                r[p] += phi(0)[(p, mu)] * xm[(p, mu)];
                                if gga {
                                    for k in 0..3 {
                                        g[k][p] += 2.0 * phi(k + 1)[(p, mu)] * xm[(p, mu)];
                                    }
                                }
                            }
                        }
                        (r, g)
                    };
                    let g0: Vec<(Vec<f64>, [Vec<f64>; 3])> =
                        d0.iter().map(|d| dens_on_grid(d)).collect();
                    let (rin, sin): (Vec<f64>, Vec<f64>) = if restricted {
                        let s = (0..np)
                            .map(|p| {
                                g0[0].1[0][p].powi(2)
                                    + g0[0].1[1][p].powi(2)
                                    + g0[0].1[2][p].powi(2)
                            })
                            .collect();
                        (g0[0].0.clone(), s)
                    } else {
                        let mut r = vec![0.0; 2 * np];
                        let mut s = vec![0.0; 3 * np];
                        for p in 0..np {
                            r[2 * p] = g0[0].0[p];
                            r[2 * p + 1] = g0[1].0[p];
                            let dot = |a: usize, c: usize| {
                                (0..3).map(|k| g0[a].1[k][p] * g0[c].1[k][p]).sum::<f64>()
                            };
                            s[3 * p] = dot(0, 0);
                            s[3 * p + 1] = dot(0, 1);
                            s[3 * p + 2] = dot(1, 1);
                        }
                        (r, s)
                    };
                    let input = if gga {
                        XcInput::gga(&rin, &sin)
                    } else {
                        XcInput::lda(&rin)
                    };
                    let mut out = XcOutput::new();
                    func.eval(2, &input, &mut out)
                        .expect("functional evaluation");
                    for (ip, (d1a, d1b)) in d1.iter().enumerate() {
                        let g1: Vec<(Vec<f64>, [Vec<f64>; 3])> = if restricted {
                            vec![dens_on_grid(d1a)]
                        } else {
                            vec![dens_on_grid(d1a), dens_on_grid(d1b)]
                        };
                        for s in 0..nspin {
                            // W[p][mu] = w (1/2 a_p phi + c_p . grad phi); V1 = phi^T W + W^T phi
                            let mut a = vec![0.0; np];
                            let mut c = [vec![0.0; np], vec![0.0; np], vec![0.0; np]];
                            for p in 0..np {
                                if restricted {
                                    let r1 = g1[0].0[p];
                                    let v2rr = out.v2rho2()[p];
                                    let mut ap = v2rr * r1;
                                    if gga {
                                        let dsig = 2.0
                                            * (0..3)
                                                .map(|k| g0[0].1[k][p] * g1[0].1[k][p])
                                                .sum::<f64>();
                                        let (frs, fss, fs) = (
                                            out.v2rhosigma()[p],
                                            out.v2sigma2()[p],
                                            out.vsigma()[p],
                                        );
                                        ap += frs * dsig;
                                        let dfs = frs * r1 + fss * dsig;
                                        for k in 0..3 {
                                            c[k][p] = w[p]
                                                * 2.0
                                                * (dfs * g0[0].1[k][p] + fs * g1[0].1[k][p]);
                                        }
                                    }
                                    a[p] = w[p] * ap;
                                } else {
                                    let r1 = [g1[0].0[p], g1[1].0[p]];
                                    let v2 = &out.v2rho2()[3 * p..3 * p + 3]; // aa ab bb
                                    let frr =
                                        |i: usize, j: usize| if i == j { v2[2 * i] } else { v2[1] };
                                    let mut ap = frr(s, 0) * r1[0] + frr(s, 1) * r1[1];
                                    if gga {
                                        let gd = |a: usize, b: usize| {
                                            (0..3)
                                                .map(|k| g0[a].1[k][p] * g1[b].1[k][p])
                                                .sum::<f64>()
                                        };
                                        // delta sigma: aa, ab, bb
                                        let ds =
                                            [2.0 * gd(0, 0), gd(0, 1) + gd(1, 0), 2.0 * gd(1, 1)];
                                        let vrs = &out.v2rhosigma()[6 * p..6 * p + 6]; // (ra: saa sab sbb, rb: ...)
                                        let vss = &out.v2sigma2()[6 * p..6 * p + 6]; // aa-aa aa-ab aa-bb ab-ab ab-bb bb-bb
                                        let fss = |i: usize, j: usize| -> f64 {
                                            let (i, j) = if i <= j { (i, j) } else { (j, i) };
                                            match (i, j) {
                                                (0, 0) => vss[0],
                                                (0, 1) => vss[1],
                                                (0, 2) => vss[2],
                                                (1, 1) => vss[3],
                                                (1, 2) => vss[4],
                                                _ => vss[5],
                                            }
                                        };
                                        let frs = |r: usize, t: usize| vrs[3 * r + t];
                                        let vsig = &out.vsigma()[3 * p..3 * p + 3];
                                        for t in 0..3 {
                                            ap += frs(s, t) * ds[t];
                                        }
                                        // delta of f_sigma_{ss}, f_sigma_ab
                                        let tss = if s == 0 { 0 } else { 2 };
                                        let dfss = frs(0, tss) * r1[0]
                                            + frs(1, tss) * r1[1]
                                            + (0..3).map(|t| fss(tss, t) * ds[t]).sum::<f64>();
                                        let dfab = frs(0, 1) * r1[0]
                                            + frs(1, 1) * r1[1]
                                            + (0..3).map(|t| fss(1, t) * ds[t]).sum::<f64>();
                                        let o = 1 - s;
                                        for k in 0..3 {
                                            c[k][p] = w[p]
                                                * (2.0 * dfss * g0[s].1[k][p]
                                                    + 2.0 * vsig[tss] * g1[s].1[k][p]
                                                    + dfab * g0[o].1[k][p]
                                                    + vsig[1] * g1[o].1[k][p]);
                                        }
                                    }
                                    a[p] = w[p] * ap;
                                }
                            }
                            let wm = Mat::from_fn(np, nao, |p, mu| {
                                let mut v = 0.5 * a[p] * phi(0)[(p, mu)];
                                if gga {
                                    v += c[0][p] * phi(1)[(p, mu)]
                                        + c[1][p] * phi(2)[(p, mu)]
                                        + c[2][p] * phi(3)[(p, mu)];
                                }
                                v
                            });
                            let v =
                                linus_base::linalg::mm(phi(0).transpose(), wm.as_ref(), Par::Seq);
                            let accm = &mut acc[ip][s];
                            for j in 0..nao {
                                for i in 0..nao {
                                    accm.add1(idx[j] * nbf + idx[i], v[(i, j)] + v[(j, i)]);
                                }
                            }
                            accm.tick(1);
                        }
                    }
                },
            )
        });
        let tot = linus_num::det::merge_all(
            taccs
                .into_vec()
                .into_iter()
                .map(|v| v.into_iter().flatten().collect())
                .collect(),
        );
        let result: Vec<Vec<Mat<f64>>> = match tot {
            Some(t) => t
                .chunks(nspin)
                .map(|c| {
                    c.iter()
                        .map(|a| {
                            let v = a.to_f64();
                            Mat::from_fn(nbf, nbf, |i, j| v[j * nbf + i])
                        })
                        .collect()
                })
                .collect(),
            None => vec![vec![Mat::<f64>::zeros(nbf, nbf); nspin]; npert],
        };
        result
            .into_iter()
            .map(|mut v| {
                if restricted {
                    let a = v.remove(0);
                    (a.clone(), a)
                } else {
                    let b = v.remove(1);
                    (v.remove(0), b)
                }
            })
            .collect()
    }

    /// Reference (pre-optimization) closed-shell kernel contraction.
    pub fn kernel_contract_restricted_ref(&self, d0: &Mat<f64>, d1: &[Mat<f64>]) -> Vec<Mat<f64>> {
        let basis = &self.basis;
        let nbf = basis.nbf;
        let grid = &self.grid;
        let func = &self.func_u;
        let fam = func.family();
        let gga = fam >= Family::Gga;
        let mgga = fam == Family::Mgga && func.needs_tau();
        let nu = if mgga {
            5
        } else if gga {
            4
        } else {
            1
        };
        let order = usize::from(gga);
        let ci = comp_index(order.max(1));
        let nx = if mgga { 4 } else { 1 };
        let npert = d1.len();
        if npert == 0 {
            return vec![];
        }
        let nb = grid.batches.len();
        let vout: Vec<Mutex<Mat<f64>>> = (0..npert)
            .map(|_| Mutex::new(Mat::zeros(nbf, nbf)))
            .collect();
        let nthr = rayon::current_num_threads();
        let works: Vec<Mutex<Work>> = (0..nthr).map(|_| Mutex::new(Work::default())).collect();
        (0..nb).into_par_iter().for_each(|bi| {
            let mut lap = crate::profile::Lap::start();
            let tid = rayon::current_thread_index().unwrap_or(0) % nthr;
            let mut guard = works[tid].lock().unwrap();
            let wk = &mut *guard;
            let b = &grid.batches[bi];
            let np = b.end - b.start;
            let w = &grid.w[b.start..b.end];
            let nao = self.batch_functions_ref(bi, order, wk);
            lap.lap(crate::profile::COLLOC);
            if nao == 0 {
                return;
            }
            let Work {
                ao,
                idx,
                dsub,
                x,
                rho,
                grad,
                sigma,
                tau,
                out,
                wv,
                wf,
                u,
                t,
                ybig,
                zbig,
                mbig,
                ..
            } = wk;
            let ao: &[f64] = ao;
            let phi =
                |k: usize, m: usize| -> &[f64] { &ao[(k * nao + m) * np..(k * nao + m + 1) * np] };
            let phim = |k: usize| cm(&ao[k * nao * np..], np, nao);
            // ground state
            {
                let ds = zeroed(dsub, nao * nao);
                for j in 0..nao {
                    for i in 0..nao {
                        ds[j * nao + i] = d0[(idx[i], idx[j])];
                    }
                }
            }
            x.resize(nx * nao * np, 0.0);
            for c in 0..nx {
                let comp = if c == 0 { 0 } else { ci.d1[c - 1] };
                matmul(
                    cm_mut(&mut x[c * nao * np..], np, nao),
                    Accum::Replace,
                    phim(comp),
                    cm(dsub, nao, nao),
                    1.0,
                    Par::Seq,
                );
            }
            let rho = zeroed(rho, np);
            let grad = zeroed(grad, 3 * np);
            let tau = zeroed(tau, np);
            for m in 0..nao {
                let (p0, x0) = (phi(0, m), &x[m * np..(m + 1) * np]);
                mul_add(rho, 1.0, p0, x0);
                if gga {
                    for k in 0..3 {
                        let pk = phi(ci.d1[k], m);
                        for gi in 0..np {
                            grad[k * np + gi] += 2.0 * pk[gi] * x0[gi];
                        }
                    }
                }
                if mgga {
                    for k in 0..3 {
                        let (pk, xk) = (
                            phi(ci.d1[k], m),
                            &x[((k + 1) * nao + m) * np..((k + 1) * nao + m + 1) * np],
                        );
                        mul_add(tau, 0.5, pk, xk);
                    }
                }
            }
            let sigma = zeroed(sigma, if gga { np } else { 0 });
            if gga {
                for gi in 0..np {
                    sigma[gi] =
                        grad[gi].powi(2) + grad[np + gi].powi(2) + grad[2 * np + gi].powi(2);
                }
            }
            lap.lap(crate::profile::DENSITY);
            func.eval(2, &xc_input(gga, mgga, rho, sigma, tau), out)
                .expect("functional evaluation");
            lap.lap(crate::profile::LIBXC);
            weighted_derivs(
                out,
                gga,
                mgga,
                w,
                grad,
                np,
                np,
                zeroed(wv, nu * np),
                zeroed(wf, nu * nu * np),
            );
            let wf: &[f64] = wf;
            // perturbations in blocks: X1 = phi D1 (and d_k phi D1 for tau), u1, t = w F u1,
            // W = 1/2 t_rho phi + t_k d_k phi, M = phi^T W (+ d_k phi^T 1/4 t_tau d_k phi)
            let chunk = (YBIG_MAX / (nao * np.max(nao) * nx).max(1)).clamp(1, npert);
            let mut p0 = 0;
            while p0 < npert {
                let p1 = (p0 + chunk).min(npert);
                let pc = p1 - p0;
                let ncol = pc * nao;
                let dbig = zeroed(dsub, nao * ncol);
                for pl in 0..pc {
                    let d = &d1[p0 + pl];
                    for j in 0..nao {
                        let col = &mut dbig[(pl * nao + j) * nao..(pl * nao + j + 1) * nao];
                        for i in 0..nao {
                            col[i] = d[(idx[i], idx[j])];
                        }
                    }
                }
                lap.lap(crate::profile::SETUP);
                let xb = zeroed(ybig, np * ncol);
                matmul(
                    cm_mut(xb, np, ncol),
                    Accum::Replace,
                    phim(0),
                    cm(dbig, nao, ncol),
                    1.0,
                    Par::Seq,
                );
                lap.lap(crate::profile::GEMM_D);
                let zb = zeroed(zbig, if mgga { 3 * np * ncol } else { 0 });
                if mgga {
                    for k in 0..3 {
                        matmul(
                            cm_mut(&mut zb[k * np * ncol..], np, ncol),
                            Accum::Replace,
                            phim(ci.d1[k]),
                            cm(dbig, nao, ncol),
                            1.0,
                            Par::Seq,
                        );
                    }
                }
                let uu = zeroed(u, pc * nu * np);
                for pl in 0..pc {
                    let ub = &mut uu[pl * nu * np..(pl + 1) * nu * np];
                    for m in 0..nao {
                        let xm = &xb[(pl * nao + m) * np..(pl * nao + m + 1) * np];
                        let p0v = phi(0, m);
                        mul_add(ub, 1.0, p0v, xm);
                        if gga {
                            for k in 0..3 {
                                let pk = phi(ci.d1[k], m);
                                let ur = &mut ub[(1 + k) * np..(2 + k) * np];
                                mul_add(ur, 2.0, pk, xm);
                            }
                        }
                        if mgga {
                            for k in 0..3 {
                                let pk = phi(ci.d1[k], m);
                                let xk = &zb[k * np * ncol + (pl * nao + m) * np
                                    ..k * np * ncol + (pl * nao + m + 1) * np];
                                let ur = &mut ub[4 * np..5 * np];
                                mul_add(ur, 0.5, pk, xk);
                            }
                        }
                    }
                }
                let tt = zeroed(t, pc * nu * np);
                for pl in 0..pc {
                    for c in 0..nu {
                        let tr = &mut tt[(pl * nu + c) * np..(pl * nu + c + 1) * np];
                        for dd in 0..nu {
                            let (f, ur) = (
                                &wf[(c * nu + dd) * np..(c * nu + dd + 1) * np],
                                &uu[(pl * nu + dd) * np..(pl * nu + dd + 1) * np],
                            );
                            mul_add(tr, 1.0, f, ur);
                        }
                    }
                }
                // W overwrites X1
                for pl in 0..pc {
                    let tp = |c: usize| &tt[(pl * nu + c) * np..(pl * nu + c + 1) * np];
                    for m in 0..nao {
                        let y = &mut xb[(pl * nao + m) * np..(pl * nao + m + 1) * np];
                        let (p0v, tr) = (phi(0, m), tp(0));
                        mul_set(y, 0.5, tr, p0v);
                        if gga {
                            for k in 0..3 {
                                let (pk, tk) = (phi(ci.d1[k], m), tp(1 + k));
                                mul_add(y, 1.0, tk, pk);
                            }
                        }
                    }
                }
                lap.lap(crate::profile::WEIGHTS);
                let mb = zeroed(mbig, nao * ncol);
                matmul(
                    cm_mut(mb, nao, ncol),
                    Accum::Replace,
                    phim(0).transpose(),
                    cm(xb, np, ncol),
                    1.0,
                    Par::Seq,
                );
                if mgga {
                    for k in 0..3 {
                        let zk = &mut zb[k * np * ncol..(k + 1) * np * ncol];
                        for pl in 0..pc {
                            let ttau = &tt[(pl * nu + 4) * np..(pl * nu + 5) * np];
                            for m in 0..nao {
                                let pk = phi(ci.d1[k], m);
                                let z = &mut zk[(pl * nao + m) * np..(pl * nao + m + 1) * np];
                                mul_set(z, 0.25, ttau, pk);
                            }
                        }
                        matmul(
                            cm_mut(mb, nao, ncol),
                            Accum::Add,
                            phim(ci.d1[k]).transpose(),
                            cm(zk, np, ncol),
                            1.0,
                            Par::Seq,
                        );
                    }
                }
                lap.lap(crate::profile::GEMM_V);
                // scatter V1_p += M_p + M_p^T (rotated start to spread lock traffic)
                for j in 0..pc {
                    let pl = (j + bi) % pc;
                    let mp = &mb[pl * nao * nao..(pl + 1) * nao * nao];
                    scatter_sym(&mut vout[p0 + pl].lock().unwrap(), idx, mp);
                }
                lap.lap(crate::profile::SCATTER);
                p0 = p1;
            }
        });
        vout.into_iter().map(|m| m.into_inner().unwrap()).collect()
    }

    pub(crate) fn batch_functions_ref(&self, bi: usize, order: usize, wk: &mut Work) -> usize {
        let (basis, grid) = (&self.basis, &self.grid);
        let b = &grid.batches[bi];
        let np = b.end - b.start;
        wk.idx.clear();
        let shells = select_shells(basis, &self.extents, b.center, b.radius);
        if shells.is_empty() || np == 0 {
            return 0;
        }
        let nc = ncomp(order);
        let nao_all: usize = shells.iter().map(|&s| basis.shells[s].nfunc()).sum();
        wk.ao_raw.resize(nc * nao_all * np, 0.0);
        eval_ao(
            basis,
            &shells,
            &grid.x[b.start..b.end],
            &grid.y[b.start..b.end],
            &grid.z[b.start..b.end],
            order,
            &mut wk.ao_raw,
        );
        wk.keep.clear();
        let mut mu = 0;
        for &s in shells.iter() {
            let sh = &basis.shells[s];
            for f in 0..sh.nfunc() {
                let big = (0..nc).any(|k| {
                    any_above(
                        &wk.ao_raw[(k * nao_all + mu) * np..(k * nao_all + mu + 1) * np],
                        self.func_thresh,
                    )
                });
                if big {
                    wk.keep.push(mu);
                    wk.idx.push(sh.ao_start + f);
                }
                mu += 1;
            }
        }
        let nao = wk.keep.len();
        wk.ao.resize(nc * nao * np, 0.0);
        for k in 0..nc {
            for (j, &m) in wk.keep.iter().enumerate() {
                let (src, dst) = ((k * nao_all + m) * np, (k * nao + j) * np);
                wk.ao[dst..dst + np].copy_from_slice(&wk.ao_raw[src..src + np]);
            }
        }
        nao
    }
}

/// Per-thread scratch of the reference kernel (the Hessian code's component-major layout).
#[derive(Default)]
pub(crate) struct Work {
    ao_raw: Vec<f64>,
    ao: Vec<f64>,
    keep: Vec<usize>,
    idx: Vec<usize>,
    dsub: Vec<f64>,
    x: Vec<f64>,
    rho: Vec<f64>,
    grad: Vec<f64>,
    sigma: Vec<f64>,
    tau: Vec<f64>,
    out: XcOutput,
    wv: Vec<f64>,
    wf: Vec<f64>,
    u: Vec<f64>,
    t: Vec<f64>,
    ybig: Vec<f64>,
    zbig: Vec<f64>,
    mbig: Vec<f64>,
}

#[derive(Default)]
struct BatchWork {
    ao: Vec<f64>,
    ao_raw: Vec<f64>,
    keep: Vec<usize>,
    idx: Vec<usize>,
    rho: Vec<f64>,
    grad: Vec<f64>,
    tau: Vec<f64>,
    dsub: Vec<f64>,
    xm: Vec<f64>,
    ym: Vec<f64>,
    wm: Vec<f64>,
    vsub: Vec<f64>,
    rin: Vec<f64>,
    sin: Vec<f64>,
    tin: Vec<f64>,
    out: XcOutput,
}
