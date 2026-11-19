//! Density-fitted two-electron gradient (Coulomb metric V = (P|Q), one auxiliary basis for
//! J, K and, for range-separated hybrids, K_LR in Psi4's wK form: Coulomb 3-center integrals
//! on one side and erf-attenuated ones on the other, as [`crate::df::DfJk`] builds it):
//!
//!   E_2e = 1/2 gamma^T V^-1 gamma - 1/2 sum_s w_s sum_ij [a X_s^T V^-1 X_s + b X_s^T V^-1 Y_s]_ij,ij
//!
//! with X^s_P,ij = (P|ij), Y^s_P,ij = (P|erf|ij) over the occupied orbitals C_s of spin s
//! (w_s = 2 for restricted, 1 for unrestricted) and gamma_P = (P|mn) D_mn. With
//! c = V^-1 gamma, X~ = V^-1 X and Y~ = V^-1 Y, the integral derivatives contract as
//!
//!   E^x = sum_Pmn (P|mn)^x G_Pmn + sum_Pmn (P|erf|mn)^x G^w_Pmn + sum_PQ (P|Q)^x g_PQ,
//!   G_P   = c_P D - sum_s w_s C_s (a X~^s_P + b/2 Y~^s_P) C_s^T,
//!   G^w_P = -b/2 sum_s w_s C_s X~^s_P C_s^T,
//!   g_PQ  = -1/2 c_P c_Q + sum_s w_s [a/2 <X~_P, X~_Q> + b/4 (<X~_P, Y~_Q> + <Y~_P, X~_Q>)]_s.
//!
//! The wK scheme fits only one side, so the erf metric (P|erf|Q) never appears and the
//! Coulomb metric gets the mixed X~/Y~ term.
//!
//! Work and memory: pass 1 builds the MO integrals X (and Y) from 3-center integrals computed one
//! block of auxiliary shells at a time (stored pair-major for the significant shell pairs,
//! half-transformed per shell panel with GEMMs, as DfJk's exchange), so the 3-index tensor is never
//! held whole (~1 GB of block buffers instead of 3 x naux x nbf^2 doubles in the dense code it
//! replaced). X and Y are symmetric in ij and kept packed (i >= j, off-diagonal entries scaled by
//! sqrt 2 after the fit so plain dot products give the full sums): half the memory, triangular
//! solves and metric-term GEMMs. The fit is two triangular solves with the Cholesky factor of V.
//! Pass 2 forms G (and G^w) per block of auxiliary functions with two GEMMs, in the layout
//! \[n\]\[m\]\[P\] so each pair's weights are contiguous, folds the Coulomb term c_P D into G's
//! pair weights and contracts with the 3-center derivative integrals on the auxiliary-lane kernel
//! ([`Eri3DerivPlan::gradient`]). Without exchange (pure functionals) E_2e is the RI-J energy and
//! the gradient is `crate::df::rij_gradient::rij_gradient` (integral-direct).

use crate::df::common::{aux_blocks, Mo3c};
use crate::df::CoulombMetric;
use faer::{Accum, Mat, MatMut, MatRef, Par};
use linus_base::linalg::matmul;
use linus_base::linalg::mm;
use linus_ints::core::hermite::ShellPair;
use linus_ints::eri3c::DfThresholds;
use linus_ints::operator::Operator;
use linus_ints::{
    AoMatrix, BasisSet, DerivOrder, Eri2DerivOptions, Eri2DerivPlan, Eri3Block, Eri3DerivOptions,
    Eri3DerivPlan, PairOptions, PairSet,
};
use linus_num::det::det_matmul;
use rayon::prelude::*;

/// Packed index of the orbital pair (i, j), i >= j.
#[inline]
fn pk(i: usize, j: usize) -> usize {
    i * (i + 1) / 2 + j
}

/// Packed pairs of o orbitals.
#[inline]
fn npk(o: usize) -> usize {
    o * (o + 1) / 2
}

/// Memory for the pass-1 block buffers (bytes): 3-center rows x block and half-transformed.
const BLOCK_BYTES: usize = 1 << 30;

/// Memory for one G block of pass 2 (nq x nbf^2 doubles; G and G^w are built one after the
/// other). Small blocks starve the per-block parallel contraction (one task per auxiliary
/// shell of the block): u1an39/def2-TZVP wB97X with 1 GiB for G and G^w together (58
/// functions, 43 blocks) spent 42.9 s in the 3-center pass.
const G_BLOCK_BYTES: usize = 3 << 30;

/// The bytes [`df_gradient`] holds at its peak beside its inputs, from its block sizes
/// (`BLOCK_BYTES`, `G_BLOCK_BYTES`: fixed by the problem, never by the memory): the larger of
/// pass 1 (the MO integrals H, a block of 3-center rows and its half-transform), the fit (H, the
/// metric term's naux^2 matrices) and pass 2 (H, the metric term, a G block (its lower triangle
/// when one block holds every auxiliary function), its half-transformed factor and M blocks). `nx`: the significant pair-function rows of the
/// 3-center integrals; `occ`: occupied orbitals per spin; `range_separated`: a long-range
/// exchange (Y and G^w beside X and G). Without exchange (pure functionals) the gradient is
/// integral-direct and holds no such arrays (0).
pub fn df_gradient_workspace(
    nbf: usize,
    aux_shells: &[linus_ints::basis::Shell],
    nx: usize,
    occ: &[usize],
    with_k: bool,
    range_separated: bool,
) -> usize {
    if !with_k {
        return 0;
    }
    let (n, naux) = (nbf, aux_shells.iter().map(|s| s.nfunc()).sum::<usize>());
    let nops = if range_separated { 2 } else { 1 };
    let omax = occ.iter().copied().max().unwrap_or(0).max(1);
    let h = 8 * nops * occ.iter().map(|&o| naux * npk(o)).sum::<usize>();
    let maxnq = |maxf: usize| {
        aux_blocks(aux_shells, maxf, false)
            .iter()
            .map(|b| b.nq)
            .max()
            .unwrap_or(0)
    };
    let maxf = (BLOCK_BYTES / (8 * (nx + n * omax * occ.len()).max(1))).clamp(16, naux.max(16));
    let nq1 = maxnq(maxf);
    let pass1 = h + 8 * nq1 * (nx + n * omax + omax * omax);
    let fit = h + 8 * (3 * naux * naux + naux * npk(omax));
    // a G block writes the lower triangle of its slabs; blocks of other sizes shift the slabs,
    // so with several blocks the whole buffer is touched
    let blocks2 = aux_blocks(
        aux_shells,
        (G_BLOCK_BYTES / (8 * n * n).max(1)).clamp(8, naux.max(8)),
        false,
    );
    let nq2 = blocks2.iter().map(|b| b.nq).max().unwrap_or(0);
    let g = if blocks2.len() > 1 {
        n * n
    } else {
        n * (n + 1) / 2
    };
    let pass2 = h + 8 * (naux * naux + nq2 * (g + n * omax + omax * omax));
    pass1.max(fit).max(pass2)
}

/// Density-fitted two-electron gradient (J and the exchange of `xop` = a/r + b erf(omega r)/r:
/// a K and b K_LR in the wK scheme; global hybrids a_x/r, pure functionals 0; see
/// the module docs). `metric`: the Coulomb metric of `aux`; `dt`: total density (row-major);
/// `occ`: occupied orbitals per spin (restricted: one set standing for both spins).
#[allow(clippy::too_many_arguments)]
#[allow(clippy::needless_range_loop, reason = "indexes several arrays in step")]
pub fn df_gradient(
    basis: &BasisSet,
    pairs: &PairSet,
    aux: &BasisSet,
    metric: &CoulombMetric,
    dt: &[f64],
    occ: &[&Mat<f64>],
    restricted: bool,
    xop: Operator,
    par: Par,
    timers: &mut linus_core::timing::Timers,
) -> Vec<[f64; 3]> {
    let n = basis.nbf;
    let naux = aux.nbf;
    let spin_w = if restricted { 2.0 } else { 1.0 };
    let a = xop.coulomb();
    let with_lr = xop.has_erf();
    let (b, omega) = if with_lr {
        (xop.long_range(), xop.omega())
    } else {
        (0.0, 0.0)
    };
    let with_k = a != 0.0 || with_lr;
    if !with_k {
        // pure functionals: E_2e = 1/2 gamma^T V^-1 gamma is the RI-J energy; its gradient is
        // integral-direct in Hermite space (split-RI-J kernels, crate::two_electron::df::rij_gradient)
        let t0 = std::time::Instant::now();
        let dm = Mat::from_fn(n, n, |i, j| dt[i * n + j]);
        let g = crate::df::rij_gradient::rij_gradient(
            basis,
            pairs,
            aux,
            &dm,
            &DfThresholds::default(),
            par,
            timers,
        );
        timers.add("grad_df_3c", t0.elapsed().as_secs_f64());
        return g;
    }
    let t0 = std::time::Instant::now();
    let ap = PairSet::aux(aux, PairOptions::with_order(DerivOrder::One));
    // the three-center derivative contractions (their pair layout is built on first use)
    let plan3 = Eri3DerivPlan::new(
        basis,
        pairs,
        aux,
        &ap,
        DerivOrder::One,
        Eri3DerivOptions::default(),
    )
    .expect("orbital pairs of order 1 in shell-index orientation");
    let aux3c = plan3.aux3c();
    assert_eq!(
        metric.naux(),
        naux,
        "the metric of another auxiliary basis set"
    );
    let chol = metric.factor();
    timers.add("grad_df_mo_metric", t0.elapsed().as_secs_f64());
    let ts = std::time::Instant::now();
    let mo = Mo3c::new(basis, pairs, aux3c);
    timers.add("grad_df_mo_pairs", ts.elapsed().as_secs_f64());
    // occupied orbitals row-major
    let cs: Vec<(usize, Vec<f64>)> = occ
        .iter()
        .map(|c| {
            let o = c.ncols();
            let mut v = vec![0.0; n * o];
            for i in 0..n {
                for j in 0..o {
                    v[i * o + j] = c[(i, j)];
                }
            }
            (o, v)
        })
        .collect();
    let omax = cs.iter().map(|x| x.0).max().unwrap_or(0).max(1);
    let nops = if with_lr { 2 } else { 1 };
    let maxf = (BLOCK_BYTES / (8 * (mo.nx + n * omax * occ.len()).max(1))).clamp(16, naux.max(16));
    let blocks = aux_blocks(&aux.shells, maxf, false);
    // pass 1: raw MO integrals H[op][s] = (P|op|ij), naux x o_s (o_s + 1) / 2 ([P][ij], packed
    // i >= j, see `pk`); gamma_P = (P|mn) D_mn
    let mut h: Vec<Vec<Vec<f64>>> = if with_k {
        (0..nops)
            .map(|_| cs.iter().map(|(o, _)| vec![0.0; naux * npk(*o)]).collect())
            .collect()
    } else {
        vec![]
    };
    // J only: gamma_P = sum_rows (P|row) D_row (off-diagonal shell pairs count twice)
    let drow = if with_k { vec![] } else { mo.density_rows(dt) };
    let mut gamma = vec![0.0; naux];
    let mut buf = vec![0.0; mo.nx * maxf];
    let (mut t_ints, mut t_half) = (0.0, 0.0);
    let (mut zbuf, mut hbuf): (Vec<f64>, Vec<f64>) = (vec![], vec![]);
    for blk in &blocks {
        let (p0, nq) = (blk.p0, blk.nq);
        for op in 0..nops {
            let ti = std::time::Instant::now();
            mo.ints(
                blk.shells.clone(),
                nq,
                if op == 0 { 0.0 } else { omega },
                &mut buf,
            );
            t_ints += ti.elapsed().as_secs_f64();
            if !with_k {
                let am = MatRef::from_row_major_slice(&buf[..mo.nx * nq], mo.nx, nq);
                let dv = MatRef::from_column_major_slice(&drow, mo.nx, 1);
                let gb = mm(am.transpose(), dv, par);
                for q in 0..nq {
                    gamma[p0 + q] = gb[(q, 0)];
                }
                continue;
            }
            for (s, (o, c)) in cs.iter().enumerate() {
                let o = *o;
                if o == 0 {
                    continue;
                }
                let th = std::time::Instant::now();
                mo.half_into(&buf, nq, c, o, &mut zbuf);
                t_half += th.elapsed().as_secs_f64();
                // H[(P, j)][i] = sum_m Z[m][(P, j)] C_mi: one o x o block per P (row-major)
                let cm = MatRef::from_row_major_slice(c, n, o);
                let zm = MatRef::from_row_major_slice(&zbuf[..n * nq * o], n, nq * o);
                hbuf.resize(nq * o * o, 0.0);
                det_matmul(
                    MatMut::from_row_major_slice_mut(&mut hbuf[..nq * o * o], nq * o, o),
                    Accum::Replace,
                    zm.transpose(),
                    cm,
                    1.0,
                );
                let hp = &hbuf[..nq * o * o];
                let hs = &mut h[op][s];
                let np = npk(o);
                hs[p0 * np..(p0 + nq) * np]
                    .par_chunks_mut(np)
                    .enumerate()
                    .for_each(|(q, dst)| {
                        let hq = &hp[q * o * o..(q + 1) * o * o];
                        for i in 0..o {
                            for j in 0..=i {
                                dst[pk(i, j)] = 0.5 * (hq[j * o + i] + hq[i * o + j]);
                            }
                        }
                    });
                if op == 0 {
                    for q in 0..nq {
                        let tr: f64 = (0..o).map(|i| hs[(p0 + q) * np + pk(i, i)]).sum();
                        gamma[p0 + q] += spin_w * tr;
                    }
                }
            }
        }
    }
    drop(buf);
    timers.add("grad_df_mo", t0.elapsed().as_secs_f64());
    timers.add("grad_df_mo_ints", t_ints);
    timers.add("grad_df_mo_half", t_half);
    let t1 = std::time::Instant::now();
    // fit: c = V^-1 gamma, X~ = V^-1 X, Y~ = V^-1 Y by two triangular solves (V = L L^T) on a
    // column-major copy (in place on the row-major naux x o^2 layout faer's solves are slow;
    // an explicit V^-1 GEMM is fast but loses ~1e-10 relative in the ill-conditioned fit)
    let solve = |m: &mut [f64], ncol: usize| {
        if ncol == 0 {
            return;
        }
        let mut x = Mat::<f64>::zeros(naux, ncol);
        x.par_col_chunks_mut(1)
            .enumerate()
            .for_each(|(k, mut col)| {
                for p in 0..naux {
                    col[(p, 0)] = m[p * ncol + k];
                }
            });
        // fixed column blocks, each solved sequentially: independent of the thread count
        let cw = ncol.div_ceil(64).max(8);
        x.par_col_chunks_mut(cw).for_each(|mut xc| {
            linus_base::linalg::solve_lower_in_place(chol, xc.as_mut(), Par::Seq);
            linus_base::linalg::solve_upper_in_place(chol.transpose(), xc.as_mut(), Par::Seq);
        });
        m.par_chunks_mut(ncol).enumerate().for_each(|(p, row)| {
            for (k, v) in row.iter_mut().enumerate() {
                *v = x[(p, k)];
            }
        });
    };
    solve(&mut gamma, 1);
    let cfit = gamma;
    if with_k {
        for hop in h.iter_mut() {
            for (s, hs) in hop.iter_mut().enumerate() {
                let np = npk(cs[s].0);
                if np == 0 {
                    // no occupied orbitals in this spin (one-electron systems)
                    continue;
                }
                solve(hs, np);
                // weights of the packed storage: sum_ij x_ij y_ij = sum_{i>=j} w_ij x_ij y_ij with
                // w = 2 off the diagonal; the stored X~ carries sqrt(w) (unpacked by `unpack_m`)
                let o = cs[s].0;
                hs.par_chunks_mut(np).for_each(|row| {
                    for i in 0..o {
                        for j in 0..i {
                            row[pk(i, j)] *= std::f64::consts::SQRT_2;
                        }
                    }
                });
            }
        }
    }
    // metric term g_PQ
    let mut gm = vec![0.0; naux * naux];
    for p in 0..naux {
        for q in 0..naux {
            gm[p * naux + q] = -0.5 * cfit[p] * cfit[q];
        }
    }
    if with_k {
        for (s, (o, _)) in cs.iter().enumerate() {
            let np = npk(*o);
            if np == 0 {
                continue;
            }
            let xm = MatRef::from_column_major_slice(&h[0][s], np, naux);
            let mut add = |m: Mat<f64>, f: f64, lower: bool| {
                gm.par_chunks_mut(naux).enumerate().for_each(|(p, row)| {
                    for q in 0..naux {
                        row[q] += f * if lower && q > p { m[(q, p)] } else { m[(p, q)] };
                    }
                });
            };
            if a != 0.0 {
                // X~^T X~: lower triangle only (half the flops)
                let mut xx = Mat::<f64>::zeros(naux, naux);
                // fixed column blocks j0..j0+w, rows j0.. (the lower part), sequential GEMMs
                let cw = naux.div_ceil(64).max(16);
                xx.par_col_chunks_mut(cw).enumerate().for_each(|(ci, xc)| {
                    let (j0, w) = (ci * cw, xc.ncols());
                    let d = xc.subrows_mut(j0, naux - j0);
                    matmul(
                        d,
                        Accum::Replace,
                        xm.transpose().subrows(j0, naux - j0),
                        xm.subcols(j0, w),
                        1.0,
                        Par::Seq,
                    );
                });
                add(xx, 0.5 * a * spin_w, true);
            }
            if with_lr {
                let ym = MatRef::from_column_major_slice(&h[1][s], np, naux);
                let mut xy = Mat::<f64>::zeros(naux, naux);
                det_matmul(xy.as_mut(), Accum::Replace, xm.transpose(), ym, 1.0);
                let sym = Mat::from_fn(naux, naux, |p, q| xy[(p, q)] + xy[(q, p)]);
                add(sym, 0.25 * b * spin_w, false);
            }
        }
    }
    let t2c = std::time::Instant::now();
    let opts = Eri2DerivOptions {
        thresh: DfThresholds::default().metric_deriv,
        op: Operator::Coulomb,
    };
    let plan =
        Eri2DerivPlan::new(aux, &ap, DerivOrder::One, opts).expect("auxiliary pairs of order 1");
    let mut g2 = vec![[0.0; 3]; aux.natom()];
    plan.gradient(
        AoMatrix::row_major(&gm, naux),
        &mut g2,
        &mut plan.workspace(),
    );
    timers.add("grad_df_2c", t2c.elapsed().as_secs_f64());
    timers.add("grad_df_fit", t1.elapsed().as_secs_f64());
    let t2 = std::time::Instant::now();
    // pass 2: 3-center derivative contractions per auxiliary block
    let mut g = g2;
    let addg = |g: &mut Vec<[f64; 3]>, x: Vec<[f64; 3]>| {
        for (p, q) in g.iter_mut().zip(&x) {
            for k in 0..3 {
                p[k] += q[k];
            }
        }
    };
    addg(
        &mut g,
        pass2_lanes(
            basis,
            &plan3,
            aux,
            dt,
            &cs,
            &h,
            &cfit,
            spin_w,
            a,
            with_lr.then_some((b, omega)),
            par,
            timers,
        ),
    );
    timers.add("grad_df_3c", t2.elapsed().as_secs_f64());
    g
}

/// Pass 2 on the auxiliary-lane derivative kernel ([`Eri3DerivPlan::gradient`]): per block of
/// auxiliary functions, the lower triangle of G (and G^w) in the layout [m][m'][P], m' <= m
/// (P fastest, so a pair's weights for the block are contiguous runs; one GEMM per row m),
/// the Coulomb term c_P D folded into G's weights, one kernel call per operator. `h`: the
/// fitted MO integrals X~ (and Y~), packed, `cfit` = V^-1 gamma.
#[allow(clippy::too_many_arguments)]
fn pass2_lanes(
    basis: &BasisSet,
    plan3: &Eri3DerivPlan<'_>,
    aux: &BasisSet,
    dt: &[f64],
    cs: &[(usize, Vec<f64>)],
    h: &[Vec<Vec<f64>>],
    cfit: &[f64],
    spin_w: f64,
    a: f64,
    lr: Option<(f64, f64)>,
    _par: Par,
    timers: &mut linus_core::timing::Timers,
) -> Vec<[f64; 3]> {
    let n = basis.nbf;
    let n2 = n * n;
    let naux = aux.nbf;
    let natom = basis
        .shells
        .iter()
        .chain(&aux.shells)
        .map(|s| s.atom + 1)
        .max()
        .unwrap_or(0);
    let (b, omega) = lr.unwrap_or((0.0, 0.0));
    let with_lr = lr.is_some() && b != 0.0;
    let ao: Vec<(usize, usize)> = basis
        .shells
        .iter()
        .map(|s| (s.ao_start, s.nfunc()))
        .collect();
    // the pair layout of the 3-center kernel, before the G buffers
    let tp = std::time::Instant::now();
    plan3.prepare_gradient();
    timers.add("grad_df_3c_pairs", tp.elapsed().as_secs_f64());
    let mut ws3 = plan3.workspace(rayon::current_num_threads());
    let mut gk = vec![[0.0; 3]; plan3.natom()];
    let maxf2 = (G_BLOCK_BYTES / (8 * n2).max(1)).clamp(8, naux.max(8));
    let blocks2 = aux_blocks(&aux.shells, maxf2, false);
    let maxnq = blocks2.iter().map(|b| b.nq).max().unwrap_or(0);
    // G[n'][(m, P)] = sum_s (C_s M^s_P C_s^T)[m][n'] for M^s_P = sum_t f_t H[t][s]_P:
    // T_s[m][(P, j)] = C_s Mcat_s (one GEMM, Mcat[i][(P, j)] = M_P[i][j]), then
    // G = C_s T_s'^T with T_s' the (n nq) x o view of T_s (second GEMM)
    let omax = cs.iter().map(|x| x.0).max().unwrap_or(0);
    let mut gbuf = vec![0.0; maxnq * n2];
    let mut tbuf = vec![0.0; maxnq * n * omax];
    let mut mbuf = vec![0.0; maxnq * omax * omax];
    let mut build_g = |p0: usize, nq: usize, coefs: &[(usize, f64)], gb: &mut [f64]| {
        let gb = &mut gb[..nq * n2];
        let mut first_spin = true;
        for (s, (o, c)) in cs.iter().enumerate() {
            let o = *o;
            if o == 0 {
                continue;
            }
            let cm = MatRef::from_row_major_slice(c, n, o);
            let mc = &mut mbuf[..o * nq * o];
            let np = npk(o);
            mc.par_chunks_mut(nq * o).enumerate().for_each(|(i, row)| {
                for qq in 0..nq {
                    for j in 0..o {
                        let x = (p0 + qq) * np + if i >= j { pk(i, j) } else { pk(j, i) };
                        let mut v = 0.0;
                        for &(t, f) in coefs {
                            v += f * spin_w * h[t][s][x];
                        }
                        row[qq * o + j] = if i == j {
                            v
                        } else {
                            v * std::f64::consts::FRAC_1_SQRT_2
                        };
                    }
                }
            });
            let tb = &mut tbuf[..n * nq * o];
            det_matmul(
                MatMut::from_row_major_slice_mut(tb, n, nq * o),
                Accum::Replace,
                cm,
                MatRef::from_row_major_slice(mc, o, nq * o),
                1.0,
            );
            let tb = &tbuf[..n * nq * o];
            let acc = if first_spin {
                Accum::Replace
            } else {
                Accum::Add
            };
            // G is symmetric: only the lower triangle, G[m][m'][P] for m' <= m, one sequential
            // GEMM per row m into its contiguous slab (half the flops; `wfn` reads G[max][min])
            gb.par_chunks_mut(n * nq).enumerate().for_each(|(m, slab)| {
                let tm = MatRef::from_row_major_slice(&tb[m * nq * o..(m + 1) * nq * o], nq, o);
                let dst = MatMut::from_row_major_slice_mut(&mut slab[..(m + 1) * nq], m + 1, nq);
                matmul(
                    dst,
                    acc,
                    cm.subrows(0, m + 1),
                    tm.transpose(),
                    1.0,
                    Par::Seq,
                );
            });
            first_spin = false;
        }
    };
    let mut g = vec![[0.0; 3]; natom];
    let (mut t_g, mut t_k, mut t_w) = (0.0, 0.0, 0.0);
    for blk in &blocks2 {
        let (p0, nq) = (blk.p0, blk.nq);
        for op in 0..if with_lr { 2 } else { 1 } {
            let tg = std::time::Instant::now();
            // G: -sum_s w_s C_s (a X~ + b/2 Y~) C_s^T (plus c_P D in the weights);
            // G^w: -b/2 sum_s w_s C_s X~ C_s^T
            let coefs: Vec<(usize, f64)> = if op == 0 {
                let mut v = vec![];
                if a != 0.0 {
                    v.push((0, -a));
                }
                if with_lr {
                    v.push((1, -0.5 * b));
                }
                v
            } else {
                vec![(0, -0.5 * b)]
            };
            build_g(p0, nq, &coefs, &mut gbuf);
            t_g += tg.elapsed().as_secs_f64();
            let tk = std::time::Instant::now();
            let gb = &gbuf[..nq * n2];
            let with_j = op == 0;
            let wfn = |sp: &ShellPair, w: &mut [f64]| -> f64 {
                let (a0, na) = ao[sp.sa];
                let (b0, nb) = ao[sp.sb];
                let f = if sp.sa == sp.sb { 1.0 } else { 2.0 };
                let nab = na * nb;
                let mut gm: f64 = 0.0;
                for ia in 0..na {
                    for ib in 0..nb {
                        let (m, nn) = (a0 + ia, b0 + ib);
                        let (hi, lo) = if m >= nn { (m, nn) } else { (nn, m) };
                        let r1 = &gb[(hi * n + lo) * nq..(hi * n + lo + 1) * nq];
                        let dj = if with_j {
                            0.5 * f * (dt[m * n + nn] + dt[nn * n + m])
                        } else {
                            0.0
                        };
                        let ab = ia * nb + ib;
                        for p in 0..nq {
                            let v = f * r1[p] + dj * cfit[p0 + p];
                            w[p * nab + ab] = v;
                            gm = gm.max(v.abs());
                        }
                    }
                }
                gm
            };
            let opr = if op == 0 {
                Operator::Coulomb
            } else {
                Operator::LongRange { omega }
            };
            let b3 = Eri3Block {
                shells: blk.shells.clone(),
                op: opr,
                thresh: plan3.options().thresholds.deriv_weighted,
            };
            plan3.gradient(&b3, wfn, &mut gk, &mut ws3);
            for (x, y) in g.iter_mut().zip(&gk) {
                for k in 0..3 {
                    x[k] += y[k];
                }
            }
            if op == 0 {
                t_k += tk.elapsed().as_secs_f64();
            } else {
                t_w += tk.elapsed().as_secs_f64();
            }
        }
    }
    timers.add("grad_df_3c_build_g", t_g);
    timers.add("grad_df_3c_kernel", t_k);
    timers.add("grad_df_3c_erf", t_w);
    timers.resource("grad_df_3c_blocks", blocks2.len() as f64);
    g
}
