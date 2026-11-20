//! Density-fitted (Coulomb metric) two-electron Hessian terms, blocked over auxiliary
//! functions: the explicit second derivatives of the RI energy at fixed orbitals and
//! F^x C_s = (J^x\[D_t\] - a_x K^x\[D_s\]) C_s for all 3N perturbations, without the 3-index
//! tensor (P|mn) or the per-perturbation exchange intermediates (3N x naux x o^2) of
//! `crate::df::reference::DfDerivJk` (a test oracle), which this replaces for Hessians.
//!
//! Channels s (closed shell: one, C_o with weight f = 2; open shell: alpha and beta, f = 1),
//! D_t = sum_s f_s C_s C_s^T, E_2e = 1/2 gamma^T V^-1 gamma - a_x/2 sum_s f_s sum_ij B_s,ij V^-1 B_s,ij
//! with B_s,P,ij = (P|ij). Derivative integrals enter only through the blocks
//! I^k_P\[a\]\[b\] = (∂_k a b|P) (derivative of AO a's center; [`linus_ints::eri3c::dblock`]), the
//! auxiliary-center derivative being -(I + I^T) (translational invariance), and their
//! half-transforms H^k_s,P = I^k_P C_s (n x o_s). Per perturbation x = (A, k):
//!
//!   Lambda^x_s,P = C_s^T (P|..)^x C_s = delta(P in A) Dk_P + S^{A,k}_P + S^{A,k}_P^T,
//!     S^{A,k}_P = C_s,A^T H^k_s,P,A (rows of atom A),  Dk_P = -(C_s^T H^k_s,P + h.c.),
//!   R^x_s = Lambda^x_s - V^x X_s = a^x + b^x + c^x with (X_s = V^-1 B_s, G^k_PQ = (∂_k P|Q))
//!     a^x_P = delta(P in A) E^k_P,  E^k_P = Dk_P - sum_Q G^k_PQ X_Q,
//!     b^x_P = S^{A,k}_P + S^{A,k}_P^T,   c^x_P = -sum_{Q in A} G^k_QP X_Q,
//!
//! and the exchange cross term of the Hessian, -a_x sum_s f_s R^x_s V^-1 R^y_s, is
//! G = (a+b+c) V^-1 (b+c) + [a V^-1 (b+c)]^T + a V^-1 a. The first two are accumulated block by
//! block of auxiliary functions P from (V^-1 (b+c)^y)_P = C_B^T Hbar^l_P,B + h.c. -
//! sum_{Q in B} gbar^l_PQ X_Q with Hbar = V^-1 H and gbar^l = V^-1 G^l^T stored (Hbar for one
//! block of atoms B at a time when it does not fit: 1500 MB or a quarter of all Hbar if larger,
//! within the machine's memory or `memory_gib` ([`linus_base::memory::MemoryLimit::hbar_mb`]; the
//! first block, which shares the budget with Z, gets at least half of it);
//! each further atom block costs one more integral pass, which needs R^x only for x on the
//! atoms of that block and later ones: G is symmetric); a V^-1 a = sum Vinv_PQ <E_P, E_Q>
//! at the end. So only 3 n o naux (Hbar) is stored instead of 3N naux o^2. The o x o blocks
//! are kept packed (symmetric lower triangle, sqrt 2 off the diagonal). G enters the Hessian in
//! one orientation fixed by the atoms, G\[x\]\[y\] with atom(x) >= atom(y) (the average with
//! G\[y\]\[x\] within an atom), which every blocking computes, and a pass's products (H-bar, its
//! solve, the Gram blocks of G) are parts of the single pass's, with its kernels
//! ([`linus_base::linalg::matmul_part`], [`linus_base::linalg::column_blocks`]): the Hessian has
//! the same bits for any number of passes, so it does not depend on the H-bar budget (D43).
//!
//! The explicit second-derivative integrals go through the auxiliary-lane kernel
//! ([`linus_ints::Eri3DerivPlan::hessian`]) in batches of auxiliary functions with
//! Gamma_P = c_P D_t - a_x sum_s f_s C_s X_s,P C_s^T formed per batch (the Coulomb part inside
//! the batches: as a separable term of the generic quartet kernel it made the first batch visit
//! every auxiliary shell, twice the work).
//!
//! F^x C_s (the gradient's first-order Fock matrices):
//!   J^x\[D_t\] C_s = M^x C_s + sum_P Z_s,P r^x_P,  r^x = gamma^x - V^x c,  Z_s = V^-1 (P|m i_s),
//!   M^x = sum_P c_P (P|mn)^x = P_A J1 + J1^T P_A - (J1_A + J1_A^T),  J1 = sum_P c_P I_P,
//!   gamma^x_P = 2 sum_{m in A} h_P\[m\] - delta(P in A) 2 sum_m h_P\[m\],  h_P\[m\] = (I_P D_t)_mm,
//!   K^x[C_s C_s^T] C_s = [rows A of sum_P H_P X_P] + sum_P I_P\[A,:\]^T Y_P\[A,:\]
//!     - delta(P in A)(H_P X_P + I_P^T Y_P) + sum_P Z_P R^x_P,   Y_P = C_s X_s,P.
//! The 2-center second-derivative term is Eri2DerivPlan::hessian with gamma_PQ = -c_P c_Q / 2 + a_x/2 sum_s f_s tr(X_s,P X_s,Q).
//! Validation: equal to DfDerivJk's terms (tests below) and through the Hessian checks
//! (tests/python/compare_hess_pyscf.py).
//!
//! Range-separated hybrids (`wk_hessian_2e`): the SCF's RI exchange in Psi4's wK scheme,
//! E_x = -1/2 sum_s f_s sum_ij B_s,ij^T V^-1 W_s,ij with W = (P|w|ij) for the operator
//! w = a/r + b erf(omega r)/r and the Coulomb metric V, is a bilinear form in B and W; by the
//! polarization identity, sym(B^T V^-1 W) = 1/4 [(B+W)^T V^-1 (B+W) - (B-W)^T V^-1 (B-W)], it
//! is the sum of two exchange energies of the global-hybrid form above with the 3-center
//! integrals of the operators w+ = (1 + a)/r + b erf and w- = (1 - a)/r - b erf and a_x = 1/4
//! and -1/4 (`df_hessian_2e_op`; the metric stays Coulomb). Every term of the Hessian
//! (explicit 3c/2c second derivatives, the cross term R V^-1 R, F^x C) is linear in such a
//! form, so the two calls plus a Coulomb-only call give the wK Hessian exactly, with the tested
//! global-hybrid code (twice the exchange work of a global hybrid, as the wK SCF).

use crate::df::common::{aux_blocks, solve_metric_mmajor, MetricSolve, Mo3c, SendPtr};
use crate::df::CoulombMetric;
use faer::{Accum, Mat, MatMut, MatRef, Par};
use linus_base::linalg::{abt_split_part, column_blocks, matmul, matmul_part, Whole};
use linus_base::linalg::{solve_lower_in_place, solve_upper_in_place};
use linus_base::memory::MemoryLimit;
use linus_core::timing::Timers;
use linus_ints::eri3c::DfThresholds;
use linus_ints::operator::Operator;
use linus_ints::{
    AoMatrix, BasisSet, DerivOrder, Eri2DerivOptions, Eri2DerivPlan, Eri3Block, Eri3DerivOptions,
    Eri3DerivPlan, Eri3DerivWorkspace, PairOptions, PairSet,
};
use rayon::prelude::*;
use std::ops::Range;
use std::time::Instant;

/// Two-electron pieces of a density-fitted Hessian.
pub struct DfHess2e {
    /// (J^x\[D_t\] - a_x K^x\[D_s\]) C_s for every perturbation x and channel s (n x o_s).
    pub fxo: Vec<Vec<Mat<f64>>>,
    /// Second derivative of the RI two-electron energy at fixed orbitals, (3N)^2 row-major.
    pub hess: Vec<f64>,
}

/// Memory for the 3-center MO pass buffers (as the DF gradient).
const MO_BLOCK_BYTES: usize = 1 << 30;
/// Memory for one block of derivative integrals (3 n^2 per auxiliary function).
const DBLOCK_BYTES: usize = 256 << 20;
/// Auxiliary functions per sub-chunk of the exchange cross term (R and V^-1 R buffers of
/// 3N x sub x o^2 each).
const RZ_BYTES: usize = 192 << 20;

/// Solve V X = B (V = L L^T) for B naux x w stored row-major (`data`), in column blocks
/// (column-major copies; faer's solves are slow on the row-major layout). The blocks
/// ([`column_blocks`]) give every column the bits it has in any other blocking (an H-bar pass's
/// columns are a slice of the single pass's).
fn solve_v_rows(l: MatRef<'_, f64>, data: &mut [f64], w: usize) {
    let naux = l.nrows();
    if w == 0 || naux == 0 {
        return;
    }
    // the raw-pointer reads and writes below stay inside data[..naux * w]
    assert!(
        data.len() >= naux * w,
        "solve_v_rows: buffer shorter than naux x w"
    );
    let blocks: Vec<Range<usize>> = column_blocks(w, 256).collect();
    let ptr = SendPtr(data.as_mut_ptr());
    blocks.into_par_iter().for_each(|cols| {
        let (c0, wc) = (cols.start, cols.len());
        let mut t = Mat::<f64>::zeros(naux, wc);
        // SAFETY: each task reads and writes only its block of columns of the naux x w buffer
        unsafe {
            let p = ptr.get();
            for r in 0..naux {
                for j in 0..wc {
                    t[(r, j)] = *p.add(r * w + c0 + j);
                }
            }
            solve_lower_in_place(l, t.as_mut(), Par::Seq);
            solve_upper_in_place(l.transpose(), t.as_mut(), Par::Seq);
            for r in 0..naux {
                for j in 0..wc {
                    *p.add(r * w + c0 + j) = t[(r, j)];
                }
            }
        }
    });
}

/// Packed lower triangle with sqrt 2 off the diagonal of the symmetrized o x o blocks of `src`
/// ([q][i][j]) into `dst` ([q][o(o+1)/2]): dot products of packed blocks are Frobenius products
/// of the symmetric parts.
fn pack_sym(src: &[f64], o: usize, dst: &mut [f64]) {
    let npk = o * (o + 1) / 2;
    if npk == 0 {
        // a channel without occupied orbitals (e.g. the beta spin of a one-electron system)
        return;
    }
    let h = std::f64::consts::FRAC_1_SQRT_2;
    dst.par_chunks_mut(npk)
        .zip(src.par_chunks(o * o))
        .for_each(|(d, s)| {
            let mut t = 0;
            for i in 0..o {
                for j in 0..i {
                    d[t] = h * (s[i * o + j] + s[j * o + i]);
                    t += 1;
                }
                d[t] = s[i * o + i];
                t += 1;
            }
        });
}

/// Inverse of [`pack_sym`] for symmetric blocks: [q][npk] -> [q][i][j].
#[allow(dead_code)]
fn unpack_sym(src: &[f64], o: usize, dst: &mut [f64]) {
    let npk = o * (o + 1) / 2;
    let h = std::f64::consts::FRAC_1_SQRT_2;
    for (d, s) in dst.chunks_mut(o * o).zip(src.chunks(npk)) {
        let mut t = 0;
        for i in 0..o {
            for j in 0..i {
                let v = h * s[t];
                d[i * o + j] = v;
                d[j * o + i] = v;
                t += 1;
            }
            d[i * o + i] = s[t];
            t += 1;
        }
    }
}

const SQRT2: f64 = std::f64::consts::SQRT_2;

/// Per-channel data.
struct Chan {
    o: usize,
    off: usize,
    f: f64,
    /// orbitals, row-major n x o
    c: Vec<f64>,
}

/// Two-electron pieces of a density-fitted HF/KS Hessian for occupied orbitals `occ` (one
/// matrix per channel, n x o_s) with occupation weights `f` (closed shell: [2.0]; open shell:
/// [1.0, 1.0]) and the exact exchange `xop` (None: Coulomb only; a global operator a/r: the
/// exchange with fraction a; a range-separated operator: Psi4's wK scheme of the module docs);
/// `mem` sets the H-bar block budget. `pairs`: index-ordered, derivative order 2; `metric`:
/// the Coulomb metric of `aux`. Runs in the caller's rayon pool.
#[allow(clippy::too_many_arguments)]
pub fn df_hessian_2e(
    basis: &BasisSet,
    pairs: &PairSet,
    aux: &BasisSet,
    metric: &CoulombMetric,
    occ: &[MatRef<'_, f64>],
    f: &[f64],
    xop: Option<Operator>,
    mem: MemoryLimit,
    par: Par,
    timers: &mut Timers,
) -> DfHess2e {
    assert!(
        pairs.deriv_order() >= DerivOrder::Two,
        "df_hessian_2e needs pairs with second-order derivative expansions"
    );
    match xop.map(|op| (op, op.global_fraction())) {
        Some((op, None)) => wk_hessian_2e(basis, pairs, aux, metric, occ, f, op, mem, par, timers),
        global => df_hessian_2e_op(
            basis,
            pairs,
            aux,
            metric,
            occ,
            f,
            global.and_then(|g| g.1).unwrap_or(0.0),
            true,
            &Operator::Coulomb,
            mem,
            par,
            timers,
        ),
    }
}

/// [`df_hessian_2e`] of a range-separated hybrid with the exact exchange of the operator
/// `x` = a/r + b erf(omega r)/r, a K + b K_LR(omega), in Psi4's wK scheme (the SCF's RI-K): J
/// plus the two polarization terms of the module docs.
#[allow(clippy::too_many_arguments)]
pub fn wk_hessian_2e(
    basis: &BasisSet,
    pairs: &PairSet,
    aux: &BasisSet,
    metric: &CoulombMetric,
    occ: &[MatRef<'_, f64>],
    f: &[f64],
    x: Operator,
    mem: MemoryLimit,
    par: Par,
    timers: &mut Timers,
) -> DfHess2e {
    let mut out = df_hessian_2e_op(
        basis,
        pairs,
        aux,
        metric,
        occ,
        f,
        0.0,
        true,
        &Operator::Coulomb,
        mem,
        par,
        timers,
    );
    for (ax, op) in [
        (
            0.25,
            Operator::Attenuated {
                coulomb: 1.0 + x.coulomb(),
                long_range: x.long_range(),
                omega: x.omega(),
            },
        ),
        (
            -0.25,
            Operator::Attenuated {
                coulomb: 1.0 - x.coulomb(),
                long_range: -x.long_range(),
                omega: x.omega(),
            },
        ),
    ] {
        if op.coulomb() == 0.0 && !op.has_erf() {
            continue;
        }
        let p = df_hessian_2e_op(
            basis, pairs, aux, metric, occ, f, ax, false, &op, mem, par, timers,
        );
        out.hess.iter_mut().zip(&p.hess).for_each(|(u, v)| *u += v);
        for (fo, fp) in out.fxo.iter_mut().zip(&p.fxo) {
            for (u, v) in fo.iter_mut().zip(fp) {
                *u += v;
            }
        }
    }
    out
}

/// [`df_hessian_2e`] with the exchange energy -a_x/2 sum_s f_s B_s^T V^-1 B_s of the 3-center
/// integrals B_s,P,ij = (P|xop|ij) of the operator `xop` (Coulomb metric V), and the Coulomb
/// terms only with `want_j` (which needs xop = 1/r).
#[allow(clippy::too_many_arguments)]
#[allow(clippy::needless_range_loop, reason = "indexes several arrays in step")]
fn df_hessian_2e_op(
    basis: &BasisSet,
    pairs: &PairSet,
    aux: &BasisSet,
    metric: &CoulombMetric,
    occ: &[MatRef<'_, f64>],
    f: &[f64],
    a_x: f64,
    want_j: bool,
    xop: &Operator,
    mem: MemoryLimit,
    par: Par,
    timers: &mut Timers,
) -> DfHess2e {
    assert!(
        !want_j || xop.is_coulomb(),
        "df_hessian_2e_op: the Coulomb terms need the Coulomb operator"
    );
    let coul = xop.is_coulomb();
    let n = basis.nbf;
    let n2 = n * n;
    let naux = aux.nbf;
    let natom = basis.atom_shells.len().max(aux.atom_shells.len());
    let n3 = 3 * natom;
    let mut off = 0;
    let ch: Vec<Chan> = occ
        .iter()
        .zip(f)
        .map(|(c, &fs)| {
            let o = c.ncols();
            let v = linus_base::linalg::mat_to_rowmajor(c.as_ref());
            let r = Chan {
                o,
                off,
                f: fs,
                c: v,
            };
            off += o;
            r
        })
        .collect();
    let ot = off;
    let with_k = a_x != 0.0 && ot > 0;
    // joint orbitals [m][j] and the occupation weight of each joint column
    let cj: Vec<f64> = (0..n * ot)
        .map(|x| {
            let (m, j) = (x / ot, x % ot);
            let s = ch
                .iter()
                .position(|c| j >= c.off && j < c.off + c.o)
                .unwrap();
            ch[s].c[m * ch[s].o + j - ch[s].off]
        })
        .collect();
    let fcol: Vec<f64> = (0..ot)
        .map(|j| ch.iter().find(|c| j >= c.off && j < c.off + c.o).unwrap().f)
        .collect();
    let cjm = MatRef::from_row_major_slice(&cj, n, ot);
    let aor: Vec<Range<usize>> = (0..natom).map(|a| basis.atom_functions(a)).collect();
    let xr: Vec<Range<usize>> = (0..natom).map(|a| aux.atom_functions(a)).collect();
    let dt: Vec<f64> = {
        let mut d = vec![0.0; n2];
        d.par_chunks_mut(n).enumerate().for_each(|(m, row)| {
            for (nn, v) in row.iter_mut().enumerate() {
                *v = (0..ot)
                    .map(|j| fcol[j] * cj[m * ot + j] * cj[nn * ot + j])
                    .sum();
            }
        });
        d
    };

    // ---- setup: metric, 3-center MO integrals, fit ----
    let t0 = Instant::now();
    // the auxiliary pairs with second-order expansions: E^A for the metric derivative rows,
    // E^AA and their bounds for the second-derivative terms; the three-center derivative
    // contractions (their pair layouts are built on first use)
    let ap = PairSet::aux(aux, PairOptions::with_order(DerivOrder::Two));
    let opts3 = Eri3DerivOptions::default();
    let mut plan3 = Eri3DerivPlan::new(basis, pairs, aux, &ap, DerivOrder::Two, opts3)
        .expect("orbital and auxiliary pairs of order 2 in shell-index orientation");
    let mut ws3 = plan3.workspace(rayon::current_num_threads());
    let aux3c = plan3.aux3c();
    assert_eq!(
        metric.naux(),
        naux,
        "the metric of another auxiliary basis set"
    );
    let lr = metric.factor();
    let mo = Mo3c::new(basis, pairs, aux3c);
    let nbuf = if coul { 1 } else { 2 };
    let maxf = (MO_BLOCK_BYTES / (8 * (nbuf * mo.nx + n * ot).max(1))).clamp(16, naux.max(16));
    let mut z: Vec<Vec<f64>> = ch.iter().map(|c| vec![0.0; n * naux * c.o]).collect(); // [m][P][i]
    let mut gamma = vec![0.0; naux];
    {
        let mut buf = vec![0.0; mo.nx * maxf];
        let mut cbuf = if coul {
            Vec::new()
        } else {
            vec![0.0; mo.nx * maxf]
        };
        for b in aux_blocks(&aux.shells, maxf, false) {
            let (p0, nq, blk) = (b.p0, b.nq, b.shells);
            if coul {
                mo.ints(blk.clone(), nq, 0.0, &mut buf);
            } else {
                // (P|xop|mn) = full (P|mn) + erf (P|erf|mn)
                mo.ints(blk.clone(), nq, 0.0, &mut cbuf);
                crate::df::response::op_ints(&mo, xop, blk.clone(), nq, &cbuf, &mut buf);
            }
            for (s, c) in ch.iter().enumerate() {
                let o = c.o;
                if o == 0 {
                    continue;
                }
                let zb = mo.half(&buf, nq, &c.c, o); // [m][(q, i)]
                if want_j {
                    let g: Vec<f64> = (0..nq)
                        .into_par_iter()
                        .map(|q| {
                            let mut t = 0.0;
                            for m in 0..n {
                                let zr = &zb[m * nq * o + q * o..m * nq * o + (q + 1) * o];
                                let cr = &c.c[m * o..(m + 1) * o];
                                t += zr.iter().zip(cr).map(|(a, b)| a * b).sum::<f64>();
                            }
                            c.f * t
                        })
                        .collect();
                    for q in 0..nq {
                        gamma[p0 + q] += g[q];
                    }
                }
                z[s].par_chunks_mut(naux * o)
                    .enumerate()
                    .for_each(|(m, zm)| {
                        zm[p0 * o..(p0 + nq) * o].copy_from_slice(&zb[m * nq * o..(m + 1) * nq * o])
                    });
            }
        }
    }
    drop(mo);
    timers.add("hess_df_mo", t0.elapsed().as_secs_f64());
    let t0 = Instant::now();
    // c = V^-1 gamma, Z_s = V^-1 (P|m i), X_s = C_s^T Z_s ([P][i][j])
    let mut cfit = gamma.clone();
    solve_v_rows(lr, &mut cfit, 1);
    for (s, c) in ch.iter().enumerate() {
        solve_metric_mmajor(lr, &mut z[s], c.o, MetricSolve::Full);
    }
    let x: Vec<Vec<f64>> = ch
        .iter()
        .enumerate()
        .map(|(s, c)| {
            let o = c.o;
            if !with_k || o == 0 {
                return Vec::new();
            }
            let cm = MatRef::from_row_major_slice(&c.c, n, o);
            let t = linus_base::linalg::mm(
                cm.transpose(),
                MatRef::from_row_major_slice(&z[s], n, naux * o),
                par,
            ); // [i][(P, j)]
            let mut xs = vec![0.0; naux * o * o];
            xs.par_chunks_mut(o * o).enumerate().for_each(|(p, xp)| {
                for i in 0..o {
                    for j in 0..o {
                        // symmetrized: X_P is symmetric up to rounding
                        xp[i * o + j] = 0.5 * (t[(i, p * o + j)] + t[(j, p * o + i)]);
                    }
                }
            });
            xs
        })
        .collect();
    timers.add("hess_df_fit", t0.elapsed().as_secs_f64());
    // peak-memory checkpoints per call (mem_hwm_gb_df_<call>_<stage>)
    let call = if want_j {
        "j"
    } else if a_x > 0.0 {
        "k"
    } else {
        "km"
    };
    linus_base::memory::mem_mark(timers, &format!("df_{call}_fit"));

    // ---- metric derivatives: G^k (rows (∂_k P|Q)), V^x c, gbar^k = V^-1 G^k^T, Vinv ----
    let t0 = Instant::now();
    let mut gd = vec![0.0; 3 * naux * naux];
    let metric = Eri2DerivOptions {
        thresh: DfThresholds::default().metric_deriv,
        op: Operator::Coulomb,
    };
    Eri2DerivPlan::new(aux, &ap, DerivOrder::One, metric)
        .expect("auxiliary pairs of order 2")
        .deriv_integrals(&mut gd);
    let g: Vec<&[f64]> = gd.chunks_exact(naux * naux).collect();
    // vc[P][x] = (V^x c)_P
    let mut vc = vec![0.0; naux * n3];
    for k in 0..3 {
        let gk = MatRef::from_row_major_slice(g[k], naux, naux);
        let gc = linus_base::linalg::mm(gk, MatRef::from_column_major_slice(&cfit, naux, 1), par);
        for (a, r) in xr.iter().enumerate() {
            let x = 3 * a + k;
            for p in r.clone() {
                vc[p * n3 + x] += gc[(p, 0)];
            }
            // sum_{Q in A} G_QP c_Q for all P
            let ca = MatRef::from_column_major_slice(&cfit[r.clone()], 1, r.len());
            let t = linus_base::linalg::mm(ca, gk.subrows(r.start, r.len()), Par::Seq);
            for p in 0..naux {
                vc[p * n3 + x] += t[(0, p)];
            }
        }
    }
    let (gbar, vinv): (Vec<Vec<f64>>, Mat<f64>) = if with_k {
        let gbar = (0..3)
            .map(|k| {
                // G^k row-major read column-major is G^k^T
                let mut y = g[k].to_vec();
                solve_lower_in_place(
                    lr,
                    MatMut::from_column_major_slice_mut(&mut y, naux, naux),
                    par,
                );
                solve_upper_in_place(
                    lr.transpose(),
                    MatMut::from_column_major_slice_mut(&mut y, naux, naux),
                    par,
                );
                y
            })
            .collect();
        let mut li = Mat::<f64>::identity(naux, naux);
        solve_lower_in_place(lr, li.as_mut(), par);
        let vinv = linus_base::linalg::mm(li.transpose(), li.as_ref(), par);
        (gbar, vinv)
    } else {
        (Vec::new(), Mat::zeros(0, 0))
    };
    timers.add("hess_df_metric", t0.elapsed().as_secs_f64());

    // ---- derivative-integral pair data and blocks ----
    let t0 = Instant::now();
    // the derivative blocks' pair layout, before the block buffers
    plan3.prepare_deriv_blocks();
    let maxq = (DBLOCK_BYTES / (24 * n2).max(1)).max(1);
    let dblocks = aux_blocks(&aux.shells, maxq, true);
    let maxnq = dblocks.iter().map(|b| b.nq).max().unwrap_or(0);
    let mut ibuf = vec![0.0; 3 * n2 * maxnq];
    let mut last_nq = usize::MAX;
    timers.add("hess_df_setup", t0.elapsed().as_secs_f64());

    // atom blocks of the V^-1 side (Hbar of one block in memory at a time)
    let ybl: Vec<Range<usize>> = if with_k {
        // default: 1500 MB, at least a quarter of all Hbar (<= ~5 passes), within what is
        // available (MemoryLimit::hbar_mb). Sizing it to the response J/K instead (2-4 GB on u1an39)
        // saved one pass but set the Hessian's memory peak (B3LYP 6.2 GB, wB97X-D3BJ 8.4 GB).
        let hbar_all_mb = (3 * n * ot * naux * 8) as f64 / 1e6;
        let mb = mem.hbar_mb(0.25 * hbar_all_mb);
        timers.resource("hess_df_hbar_mb", mb);
        // the first block shares the budget with Z = V^-1 (P|mi), freed after the first pass, but
        // gets at least half of it (with Z larger than the budget, the old limit of zero made
        // every atom its own pass: ts61 needed a 4000 MB budget)
        let zmb = (n * naux * ot * 8) as f64 / 1e6;
        let mut out: Vec<Range<usize>> = vec![];
        let mut a0 = 0;
        let mut acc = 0.0;
        for a in 0..natom {
            let sz = (3 * aor[a].len() * ot * naux * 8) as f64 / 1e6;
            let lim = if out.is_empty() {
                (mb - zmb).max(0.5 * mb)
            } else {
                mb
            };
            if a > a0 && acc + sz > lim {
                out.push(a0..a);
                a0 = a;
                acc = 0.0;
            }
            acc += sz;
        }
        out.push(a0..natom);
        out
    } else {
        // one pass over no atom block
        std::iter::once(0..0).collect()
    };
    timers.resource("hess_df_passes", ybl.len() as f64);

    // ---- accumulators ----
    let mut gx = vec![0.0; naux * n3]; // gamma^x [P][x]
                                       // J^x C_s second part sum_P Z_s,P r^x_P (r = gamma^x - V^x c), per channel and AO row m: 3N x o
    let mut jz: Vec<Vec<Mat<f64>>> = ch
        .iter()
        .map(|c| (0..n).map(|_| Mat::zeros(n3, c.o)).collect())
        .collect();
    let mut j1: Vec<Vec<f64>> = (0..3).map(|_| vec![0.0; n2]).collect();
    let mut fj = vec![Mat::<f64>::zeros(n, ot); n3]; // J aux-center part, joint columns
    let mut ky1 = vec![Mat::<f64>::zeros(n, ot); 3];
    let mut kacc = vec![Mat::<f64>::zeros(n, ot); n3];
    let mut gm: Vec<Mat<f64>> = ch.iter().map(|_| Mat::zeros(n3, n3)).collect();
    let mut p2: Vec<Mat<f64>> = ch.iter().map(|_| Mat::zeros(n3, n3)).collect();
    let mut ep: Vec<Vec<f64>> = ch
        .iter()
        .map(|c| {
            if with_k {
                vec![0.0; 3 * naux * c.o * (c.o + 1) / 2]
            } else {
                Vec::new()
            }
        })
        .collect();

    let (mut rbuf, mut zbuf, mut ebuf, mut hbsbuf): (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    // packed X_s ([P][o(o+1)/2])
    let xp: Vec<Vec<f64>> = ch
        .iter()
        .enumerate()
        .map(|(s, c)| {
            if !with_k {
                return Vec::new();
            }
            let mut v = vec![0.0; naux * c.o * (c.o + 1) / 2];
            pack_sym(&x[s], c.o, &mut v);
            v
        })
        .collect();
    let mut hbuf: Vec<f64> = Vec::new();
    let mut ubuf: Vec<f64> = Vec::new();
    let (mut t_r, mut t_e, mut t_z, mut t_g, mut t_t2) = (0.0, 0.0, 0.0, 0.0, 0.0);
    for (bi, yb) in ybl.iter().enumerate() {
        let first = bi == 0;
        // Hbar = V^-1 H for the AO functions of the atoms in yb: row-major [P][(l, b, j)]
        let rb = if yb.is_empty() {
            0..0
        } else {
            aor[yb.start].start..aor[yb.end - 1].end
        };
        let nb = rb.len();
        let wbar = 3 * nb * ot;
        let hbar: Vec<f64> = if with_k {
            let t0 = Instant::now();
            let mut hb = vec![0.0; naux * wbar];
            for b in &dblocks {
                let (p0, nq, shells) = (b.p0, b.nq, &b.shells);
                if nq != last_nq {
                    ibuf[..3 * n2 * nq]
                        .par_chunks_mut(1 << 16)
                        .for_each(|c| c.fill(0.0));
                    last_nq = nq;
                }
                let blk = Eri3Block {
                    shells: shells.clone(),
                    op: *xop,
                    thresh: DfThresholds::default().deriv_block,
                };
                plan3.deriv_block(&blk, rb.clone(), &mut ibuf, &mut ws3);
                // H rows (k, b in rb, q): one GEMM per k, the rows of this pass's atoms of the
                // single pass's product
                for k in 0..3 {
                    let ik = MatRef::from_row_major_slice(
                        &ibuf[(k * n + rb.start) * nq * n..(k * n + rb.end) * nq * n],
                        nb * nq,
                        n,
                    );
                    let mut hk = Mat::<f64>::zeros(nb * nq, ot); // rows (b, q), cols j
                    let whole = Whole {
                        rows: n * nq,
                        cols: ot,
                    };
                    matmul_part(whole, hk.as_mut(), Accum::Replace, ik, cjm, 1.0, par);
                    hb[p0 * wbar..(p0 + nq) * wbar]
                        .par_chunks_mut(wbar)
                        .enumerate()
                        .for_each(|(q, row)| {
                            for b in 0..nb {
                                for j in 0..ot {
                                    row[(k * nb + b) * ot + j] = hk[(b * nq + q, j)];
                                }
                            }
                        });
                }
            }
            timers.add("hess_df_hbar_ints", t0.elapsed().as_secs_f64());
            let t1 = Instant::now();
            solve_v_rows(lr, &mut hb, wbar);
            timers.add("hess_df_hbar_solve", t1.elapsed().as_secs_f64());
            hb
        } else {
            Vec::new()
        };
        let nyb = 3 * yb.len();
        for b in &dblocks {
            let (ablk, p0, nq, shells) = (b.atom, b.p0, b.nq, &b.shells);
            let t0 = Instant::now();
            if nq != last_nq {
                ibuf[..3 * n2 * nq]
                    .par_chunks_mut(1 << 16)
                    .for_each(|c| c.fill(0.0));
                last_nq = nq;
            }
            let blk = Eri3Block {
                shells: shells.clone(),
                op: *xop,
                thresh: DfThresholds::default().deriv_block,
            };
            plan3.deriv_block(&blk, 0..n, &mut ibuf, &mut ws3);
            let ib = &ibuf[..3 * n2 * nq];
            // H[k][a][q][j] = sum_b I[k][a][q][b] C[b][j]
            hbuf.resize(3 * n * nq * ot, 0.0);
            let h = &mut hbuf[..3 * n * nq * ot];
            matmul(
                MatMut::from_row_major_slice_mut(h, 3 * n * nq, ot),
                Accum::Replace,
                MatRef::from_row_major_slice(ib, 3 * n * nq, n),
                cjm,
                1.0,
                par,
            );
            let h = &hbuf[..3 * n * nq * ot];
            timers.add("hess_df_ints", t0.elapsed().as_secs_f64());
            // per-channel copies H_s[k][a][q][i] (the joint array itself for one channel)
            let hs: Vec<std::borrow::Cow<[f64]>> = ch
                .iter()
                .map(|c| {
                    if ch.len() == 1 {
                        std::borrow::Cow::Borrowed(h)
                    } else {
                        let o = c.o;
                        let mut v = vec![0.0; 3 * n * nq * o];
                        v.par_chunks_mut(o.max(1)).enumerate().for_each(|(r, d)| {
                            d.copy_from_slice(&h[r * ot + c.off..r * ot + c.off + o])
                        });
                        std::borrow::Cow::Owned(v)
                    }
                })
                .collect();
            if first && want_j {
                let t0 = Instant::now();
                // gamma^x: hq[k][q][m] = sum_j f_j H[k][m][q][j] C[m][j]
                let hq: Vec<f64> = (0..3 * nq * n)
                    .into_par_iter()
                    .map(|idx| {
                        let (k, q, m) = (idx / (nq * n), (idx / n) % nq, idx % n);
                        let hr = &h[((k * n + m) * nq + q) * ot..((k * n + m) * nq + q + 1) * ot];
                        let cr = &cj[m * ot..(m + 1) * ot];
                        (0..ot).map(|j| fcol[j] * hr[j] * cr[j]).sum::<f64>()
                    })
                    .collect();
                for k in 0..3 {
                    for q in 0..nq {
                        let row = &hq[(k * nq + q) * n..(k * nq + q + 1) * n];
                        let tot: f64 = row.iter().sum();
                        for (a, r) in aor.iter().enumerate() {
                            let x = 3 * a + k;
                            let mut v = 2.0 * row[r.clone()].iter().sum::<f64>();
                            if a == ablk {
                                v -= 2.0 * tot;
                            }
                            gx[(p0 + q) * n3 + x] = v;
                        }
                    }
                }
                // J1 = sum_P c_P I_P (block part J1b); the auxiliary-center part of M^x C
                let cb = &cfit[p0..p0 + nq];
                let mut j1b = vec![0.0; 3 * n2];
                j1b.par_chunks_mut(n).enumerate().for_each(|(ka, row)| {
                    for q in 0..nq {
                        let src = &ib[(ka * nq + q) * n..(ka * nq + q + 1) * n];
                        row.iter_mut().zip(src).for_each(|(u, v)| *u += cb[q] * v);
                    }
                });
                for k in 0..3 {
                    let jb = &j1b[k * n2..(k + 1) * n2];
                    j1[k].iter_mut().zip(jb).for_each(|(u, v)| *u += v);
                    let jm = MatRef::from_row_major_slice(jb, n, n);
                    let x = 3 * ablk + k;
                    matmul(fj[x].as_mut(), Accum::Add, jm, cjm, -1.0, par);
                    matmul(fj[x].as_mut(), Accum::Add, jm.transpose(), cjm, -1.0, par);
                }
                // sum_P Z_s,P r^x_P over this block's P (Z is freed after the first pass)
                let rblk =
                    Mat::from_fn(nq, n3, |q, x| gx[(p0 + q) * n3 + x] - vc[(p0 + q) * n3 + x]);
                for (s, c) in ch.iter().enumerate() {
                    let o = c.o;
                    if o == 0 {
                        continue;
                    }
                    let zs = &z[s];
                    jz[s].par_iter_mut().enumerate().for_each(|(m, acc)| {
                        let zm = MatRef::from_row_major_slice(
                            &zs[(m * naux + p0) * o..(m * naux + p0 + nq) * o],
                            nq,
                            o,
                        );
                        matmul(
                            acc.as_mut(),
                            Accum::Add,
                            rblk.transpose(),
                            zm,
                            1.0,
                            Par::Seq,
                        );
                    });
                }
                timers.add("hess_df_j1", t0.elapsed().as_secs_f64());
            }
            if first && with_k {
                let t0 = Instant::now();
                // Y_s,q = C_s X_s,q ([a][q][i] per channel); sum_q H_q X_q; I_A^T Y_A per atom
                for (s, c) in ch.iter().enumerate() {
                    let o = c.o;
                    if o == 0 {
                        continue;
                    }
                    let cm = MatRef::from_row_major_slice(&c.c, n, o);
                    let xb = &x[s][p0 * o * o..(p0 + nq) * o * o];
                    let mut yt = vec![0.0; n * nq * o];
                    {
                        let ptr = SendPtr(yt.as_mut_ptr());
                        (0..nq).into_par_iter().for_each(|q| {
                            let xq =
                                MatRef::from_row_major_slice(&xb[q * o * o..(q + 1) * o * o], o, o);
                            // SAFETY: column block q of the n x (nq o) row-major buffer belongs to this task
                            let dst = unsafe {
                                MatMut::from_raw_parts_mut(
                                    ptr.get().add(q * o),
                                    n,
                                    o,
                                    (nq * o) as isize,
                                    1,
                                )
                            };
                            matmul(dst, Accum::Replace, cm, xq, 1.0, Par::Seq);
                        });
                    }
                    // KY_b^k = sum_q H^k_q X_q (n x o), one GEMM per k: [m][(q, j)] x [(q, j)][i]
                    let kyb: Vec<Mat<f64>> = (0..3)
                        .map(|k| {
                            let hk = MatRef::from_row_major_slice(
                                &hs[s][k * n * nq * o..(k + 1) * n * nq * o],
                                n,
                                nq * o,
                            );
                            linus_base::linalg::mm(
                                hk,
                                MatRef::from_row_major_slice(xb, nq * o, o),
                                par,
                            )
                        })
                        .collect();
                    // T^{A,k} = I^k[A rows]^T Y[A rows] (n x o)
                    let tak: Vec<Mat<f64>> = (0..n3)
                        .into_par_iter()
                        .map(|xi| {
                            let (a, k) = (xi / 3, xi % 3);
                            let r = &aor[a];
                            if r.is_empty() {
                                return Mat::zeros(n, o);
                            }
                            let ia = MatRef::from_row_major_slice(
                                &ib[(k * n + r.start) * nq * n..(k * n + r.end) * nq * n],
                                r.len() * nq,
                                n,
                            );
                            let ya = MatRef::from_row_major_slice(
                                &yt[r.start * nq * o..r.end * nq * o],
                                r.len() * nq,
                                o,
                            );
                            linus_base::linalg::mm(ia.transpose(), ya, Par::Seq)
                        })
                        .collect();
                    for k in 0..3 {
                        let mut tall = Mat::<f64>::zeros(n, o);
                        for a in 0..natom {
                            let xi = 3 * a + k;
                            add_to(kacc[xi].subcols_mut(c.off, o), tak[xi].as_ref(), 1.0);
                            tall += &tak[xi];
                        }
                        add_to(ky1[k].subcols_mut(c.off, o), kyb[k].as_ref(), 1.0);
                        add_to(
                            kacc[3 * ablk + k].subcols_mut(c.off, o),
                            kyb[k].as_ref(),
                            -1.0,
                        );
                        add_to(
                            kacc[3 * ablk + k].subcols_mut(c.off, o),
                            tall.as_ref(),
                            -1.0,
                        );
                    }
                }
                timers.add("hess_df_k1", t0.elapsed().as_secs_f64());
            }
            if !with_k {
                continue;
            }
            // ---- exchange: R^x (all x), V^-1 (b+c)^y (y in yb), Gram, term 2 ----
            let t0 = Instant::now();
            for (s, c) in ch.iter().enumerate() {
                let o = c.o;
                if o == 0 {
                    continue;
                }
                let npk = o * (o + 1) / 2;
                // packed symmetric o x o blocks throughout (lower triangle, sqrt 2 off the
                // diagonal: dot products are Frobenius products)
                let xpm = MatRef::from_row_major_slice(&xp[s], naux, npk);
                // R^x rows of this pass: x on atoms >= yb.start, which holds every G[x][y] with
                // atom(x) >= atom(y), the orientation the assembly takes (G is symmetric)
                let x0 = 3 * yb.start;
                let nr = n3 - x0;
                let sub = (RZ_BYTES / (8 * n3 * npk).max(1)).clamp(1, nq);
                // buffers reused across sub-chunks (fresh multi-100-MB allocations page-fault)
                rbuf.resize(n3 * sub * npk, 0.0);
                zbuf.resize(nyb * sub * npk, 0.0);
                ebuf.resize(3 * sub * npk, 0.0);
                let mut q0 = 0;
                while q0 < nq {
                    let q1 = (q0 + sub).min(nq);
                    let ns = q1 - q0;
                    let len = ns * npk;
                    let (pa, pb) = (p0 + q0, p0 + q1);
                    // S^{A,k} = C_A^T H^k[A rows][q0..q1] (o x ns o); R^x = S + S^T (per q), packed
                    let tr = Instant::now();
                    let pack_q = |d: &mut [f64], sm: &Mat<f64>, f: f64, q: usize| {
                        let mut t = 0;
                        for i in 0..o {
                            for j in 0..i {
                                d[t] = f * SQRT2 * (sm[(i, q * o + j)] + sm[(j, q * o + i)]);
                                t += 1;
                            }
                            d[t] = f * 2.0 * sm[(i, q * o + i)];
                            t += 1;
                        }
                    };
                    let pack_s = |dst: &mut [f64], sm: &Mat<f64>, f: f64| {
                        for q in 0..ns {
                            pack_q(&mut dst[q * npk..(q + 1) * npk], sm, f, q);
                        }
                    };
                    let fill_b = |dst: &mut [f64],
                                  hsrc: &[f64],
                                  row_stride: usize,
                                  col0: usize,
                                  rows: Range<usize>,
                                  crow: Range<usize>| {
                        let hm = MatRef::from_row_major_slice_with_stride(
                            &hsrc[rows.start * row_stride + col0..],
                            rows.len(),
                            ns * o,
                            row_stride,
                        );
                        let ca = MatRef::from_row_major_slice(
                            &c.c[crow.start * o..crow.end * o],
                            crow.len(),
                            o,
                        );
                        let sm = linus_base::linalg::mm(ca.transpose(), hm, Par::Seq); // o x (ns o)
                        pack_s(dst, &sm, 1.0);
                    };
                    let hsk = &hs[s];
                    let r = &mut rbuf[..nr * len];
                    r.par_chunks_mut(len).enumerate().for_each(|(xr_, rx)| {
                        let xi = x0 + xr_;
                        let (a, k) = (xi / 3, xi % 3);
                        let rg = aor[a].clone();
                        if !rg.is_empty() {
                            // rows a of H^k: hs[((k n + a) nq + q) o + i]
                            fill_b(rx, &hsk[k * n * nq * o..], nq * o, q0 * o, rg.clone(), rg);
                        } else {
                            rx.fill(0.0);
                        }
                        // c part: - sum_{Q in A} G^k_QP X_Q
                        let qa = xr[a].clone();
                        if !qa.is_empty() {
                            let gq = MatRef::from_row_major_slice_with_stride(
                                &g[k][qa.start * naux + pa..],
                                qa.len(),
                                ns,
                                naux,
                            );
                            matmul(
                                MatMut::from_row_major_slice_mut(rx, ns, npk),
                                Accum::Add,
                                gq.transpose(),
                                xpm.subrows(qa.start, qa.len()),
                                -1.0,
                                Par::Seq,
                            );
                        }
                    });
                    t_r += tr.elapsed().as_secs_f64();
                    let te = Instant::now();
                    // a part: E^k = -(C^T H^k + h.c.) - G^k[P rows] X (x = (ablk, k)), parallel GEMMs
                    // (computed in the first pass and stored in ep; later passes read it back)
                    let e = &mut ebuf[..3 * len];
                    for k in 0..3 {
                        let ek = &mut e[k * len..(k + 1) * len];
                        if !first {
                            ek.copy_from_slice(
                                &ep[s][(k * naux + pa) * npk..(k * naux + pb) * npk],
                            );
                            continue;
                        }
                        let hm = MatRef::from_row_major_slice_with_stride(
                            &hsk[k * n * nq * o + q0 * o..],
                            n,
                            ns * o,
                            nq * o,
                        );
                        let sm = linus_base::linalg::mm(
                            MatRef::from_row_major_slice(&c.c, n, o).transpose(),
                            hm,
                            par,
                        ); // o x (ns o)
                        ek.par_chunks_mut(npk)
                            .enumerate()
                            .for_each(|(q, d)| pack_q(d, &sm, -1.0, q));
                        let gk =
                            MatRef::from_row_major_slice(&g[k][pa * naux..pb * naux], ns, naux);
                        matmul(
                            MatMut::from_row_major_slice_mut(ek, ns, npk),
                            Accum::Add,
                            gk,
                            xpm,
                            -1.0,
                            par,
                        );
                    }
                    let e = &ebuf[..3 * len];
                    let r = &mut rbuf[..nr * len];
                    for k in 0..3 {
                        let xi = 3 * ablk + k;
                        if xi < x0 {
                            continue;
                        }
                        let xi = xi - x0;
                        r[xi * len..(xi + 1) * len]
                            .iter_mut()
                            .zip(&e[k * len..(k + 1) * len])
                            .for_each(|(u, v)| *u += v);
                    }
                    if first {
                        for k in 0..3 {
                            ep[s][(k * naux + pa) * npk..(k * naux + pb) * npk]
                                .copy_from_slice(&e[k * len..(k + 1) * len]);
                        }
                    }
                    t_e += te.elapsed().as_secs_f64();
                    let tz = Instant::now();
                    // V^-1 (b + c)^y for y in yb: C_B^T Hbar^l_B + h.c. - sum_{Q in B} gbar^l_PQ X_Q
                    let zb = &mut zbuf[..nyb * len];
                    {
                        // gather Hbar rows of the sub-chunk, channel s: [l][b][q][i]
                        hbsbuf.resize(3 * nb * ns * o, 0.0);
                        let hbs = &mut hbsbuf[..3 * nb * ns * o];
                        hbs.par_chunks_mut(ns * o).enumerate().for_each(|(lb, d)| {
                            for q in 0..ns {
                                let row = &hbar[(pa + q) * wbar..(pa + q + 1) * wbar];
                                d[q * o..(q + 1) * o]
                                    .copy_from_slice(&row[lb * ot + c.off..lb * ot + c.off + o]);
                            }
                        });
                        let hbs = &hbsbuf[..3 * nb * ns * o];
                        zb.par_chunks_mut(len).enumerate().for_each(|(yi, zy)| {
                            let (bb, l) = (yb.start + yi / 3, yi % 3);
                            let rg = aor[bb].clone();
                            if !rg.is_empty() {
                                let rel = (rg.start - rb.start)..(rg.end - rb.start);
                                fill_b(zy, &hbs[l * nb * ns * o..], ns * o, 0, rel, rg);
                            } else {
                                zy.fill(0.0);
                            }
                            let qb = xr[bb].clone();
                            if !qb.is_empty() {
                                // gbar^l column-major: element (P, Q) at P + naux Q
                                let gq = MatRef::from_column_major_slice_with_stride(
                                    &gbar[l][qb.start * naux + pa..],
                                    ns,
                                    qb.len(),
                                    naux,
                                );
                                matmul(
                                    MatMut::from_row_major_slice_mut(zy, ns, npk),
                                    Accum::Add,
                                    gq,
                                    xpm.subrows(qb.start, qb.len()),
                                    -1.0,
                                    Par::Seq,
                                );
                            }
                        });
                    }
                    t_z += tz.elapsed().as_secs_f64();
                    let tg = Instant::now();
                    // Gram and the a V^-1 (b+c) part
                    let r = &rbuf[..nr * len];
                    let zb = &zbuf[..nyb * len];
                    let rm = MatRef::from_row_major_slice(r, nr, len);
                    let zm = MatRef::from_row_major_slice(zb, nyb, len);
                    // parts of the single pass's products (x on all atoms, y on all atoms): the
                    // same kernels, so the same bits, for any blocking
                    let gb = abt_split_part(rm, zm, Whole { rows: n3, cols: n3 });
                    let em = MatRef::from_row_major_slice(e, 3, len);
                    let pb2 = abt_split_part(em, zm, Whole { rows: 3, cols: n3 });
                    for xi in 0..nr {
                        for yi in 0..nyb {
                            gm[s][(x0 + xi, 3 * yb.start + yi)] += gb[(xi, yi)];
                        }
                    }
                    for k in 0..3 {
                        for yi in 0..nyb {
                            p2[s][(3 * ablk + k, 3 * yb.start + yi)] += pb2[(k, yi)];
                        }
                    }
                    t_g += tg.elapsed().as_secs_f64();
                    let t2t = Instant::now();
                    // K^x C_s term 2: sum_P Z_s,P R^x_P (first pass), one GEMM per group of x:
                    // Z (n x ns o) [R^x unpacked side by side] ((ns o) x nx o)
                    if first {
                        debug_assert_eq!(x0, 0);
                        let zsm = MatRef::from_row_major_slice_with_stride(
                            &z[s][pa * o..],
                            n,
                            ns * o,
                            naux * o,
                        );
                        let xg = n3.div_ceil(2);
                        for xa in (0..n3).step_by(xg) {
                            let xb = (xa + xg).min(n3);
                            let w = (xb - xa) * o;
                            ubuf.resize(ns * o * w, 0.0);
                            {
                                let ptr = SendPtr(ubuf.as_mut_ptr());
                                (xa..xb).into_par_iter().for_each(|xi| {
                                    let rx = &r[xi * len..(xi + 1) * len];
                                    let c0 = (xi - xa) * o;
                                    let h = std::f64::consts::FRAC_1_SQRT_2;
                                    // SAFETY: columns c0 .. c0 + o of the (ns o) x w buffer
                                    let p = ptr.get();
                                    for q in 0..ns {
                                        let src = &rx[q * npk..(q + 1) * npk];
                                        let mut t = 0;
                                        for i in 0..o {
                                            for j in 0..i {
                                                let v = h * src[t];
                                                unsafe {
                                                    *p.add((q * o + i) * w + c0 + j) = v;
                                                    *p.add((q * o + j) * w + c0 + i) = v;
                                                }
                                                t += 1;
                                            }
                                            unsafe { *p.add((q * o + i) * w + c0 + i) = src[t] };
                                            t += 1;
                                        }
                                    }
                                });
                            }
                            let t2 = linus_base::linalg::mm(
                                zsm,
                                MatRef::from_row_major_slice(&ubuf[..ns * o * w], ns * o, w),
                                par,
                            );
                            kacc[xa..xb]
                                .par_iter_mut()
                                .enumerate()
                                .for_each(|(xi, kx)| {
                                    add_to(kx.subcols_mut(c.off, o), t2.subcols(xi * o, o), 1.0)
                                });
                        }
                    }
                    t_t2 += t2t.elapsed().as_secs_f64();
                    q0 = q1;
                }
            }
            timers.add("hess_df_rz", t0.elapsed().as_secs_f64());
        }
        if first {
            // Z is only needed in the first pass (K^x term 2 and the J term in jz)
            z = Vec::new();
        }
    }
    drop(z);
    drop(ibuf);
    // the derivative blocks' pair layout
    plan3.free_layouts();
    drop((rbuf, zbuf, ebuf, hbuf, hbsbuf, xp, ubuf));
    for (name, t) in [
        ("hess_df_rz_r", t_r),
        ("hess_df_rz_e", t_e),
        ("hess_df_rz_z", t_z),
        ("hess_df_rz_gram", t_g),
        ("hess_df_rz_t2", t_t2),
    ] {
        timers.add(name, t);
    }

    // ---- assemble F^x C_s ----
    let t0 = Instant::now();
    // r^x = gamma^x - V^x c (the J cross term below); sum_P Z_s,P r^x_P is in jz
    let rv: Vec<f64> = gx.iter().zip(&vc).map(|(a, b)| a - b).collect();
    drop(gx);
    let mut fxo: Vec<Vec<Mat<f64>>> = (0..n3)
        .map(|_| ch.iter().map(|c| Mat::zeros(n, c.o)).collect())
        .collect();
    for (s, c) in ch.iter().enumerate() {
        let o = c.o;
        if o == 0 {
            continue;
        }
        let zr = &jz[s];
        let cm = MatRef::from_row_major_slice(&c.c, n, o);
        let j1c: Vec<Mat<f64>> = (0..3)
            .map(|k| linus_base::linalg::mm(MatRef::from_row_major_slice(&j1[k], n, n), cm, par))
            .collect();
        fxo.par_iter_mut().enumerate().for_each(|(xi, fx)| {
            let (a, k) = (xi / 3, xi % 3);
            let rg = aor[a].clone();
            let mut jv = fj[xi].subcols(c.off, o).to_owned();
            for m in 0..n {
                for i in 0..o {
                    jv[(m, i)] += zr[m][(xi, i)];
                }
            }
            if !rg.is_empty() {
                // M^x C: rows A of J1 C + J1[A,:]^T C[A,:]
                for m in rg.clone() {
                    for i in 0..o {
                        jv[(m, i)] += j1c[k][(m, i)];
                    }
                }
                let ja =
                    MatRef::from_row_major_slice(&j1[k][rg.start * n..rg.end * n], rg.len(), n);
                matmul(
                    jv.as_mut(),
                    Accum::Add,
                    ja.transpose(),
                    cm.subrows(rg.start, rg.len()),
                    1.0,
                    Par::Seq,
                );
            }
            if with_k {
                let mut kv = kacc[xi].subcols(c.off, o).to_owned();
                for m in rg {
                    for i in 0..o {
                        kv[(m, i)] += ky1[k][(m, c.off + i)];
                    }
                }
                jv -= kv * a_x;
            }
            fx[s] = jv;
        });
    }
    timers.add("hess_df_fock", t0.elapsed().as_secs_f64());

    // ---- cross terms: J (r V^-1 r) and exchange G = M + P2^T + a V^-1 a ----
    let t0 = Instant::now();
    let mut hess = vec![0.0; n3 * n3];
    {
        let mut w = Mat::from_fn(naux, n3, |p, x| rv[p * n3 + x]);
        solve_lower_in_place(lr, w.as_mut(), par);
        let hj = linus_base::linalg::mm(w.transpose(), w.as_ref(), par);
        for i in 0..n3 {
            for j in 0..n3 {
                hess[i * n3 + j] = hj[(i, j)];
            }
        }
    }
    if with_k {
        for (s, c) in ch.iter().enumerate() {
            let npk = c.o * (c.o + 1) / 2;
            if npk == 0 {
                continue;
            }
            // a V^-1 a: sum_{P in A, Q in B} Vinv_PQ <E^k_P, E^l_Q>
            let mut p3 = Mat::<f64>::zeros(n3, n3);
            for k in 0..3 {
                for l in 0..3 {
                    let ek = MatRef::from_row_major_slice(
                        &ep[s][k * naux * npk..(k + 1) * naux * npk],
                        naux,
                        npk,
                    );
                    let el = MatRef::from_row_major_slice(
                        &ep[s][l * naux * npk..(l + 1) * naux * npk],
                        naux,
                        npk,
                    );
                    let gkl = abt_split_wide(ek, el, par);
                    let blocks: Vec<(usize, usize, f64)> = (0..natom * natom)
                        .into_par_iter()
                        .map(|ab| {
                            let (a, b) = (ab / natom, ab % natom);
                            let mut v = 0.0;
                            for p in xr[a].clone() {
                                for q in xr[b].clone() {
                                    v += vinv[(p, q)] * gkl[(p, q)];
                                }
                            }
                            (a, b, v)
                        })
                        .collect();
                    for (a, b, v) in blocks {
                        p3[(3 * a + k, 3 * b + l)] += v;
                    }
                }
            }
            // G is symmetric; it is taken in one orientation fixed by the atoms, G[x][y] with
            // atom(x) >= atom(y), averaged with G[y][x] only within an atom. Every pass computes
            // those entries (M = (a+b+c) V^-1 (b+c) for x on atoms >= its block's first), so the
            // bits do not depend on how the atoms are blocked (the H-bar budget, `memory_gib`)
            let gl = |i: usize, j: usize| gm[s][(i, j)] + p2[s][(j, i)] + p3[(i, j)];
            let gs = Mat::from_fn(
                n3,
                n3,
                |i, j| {
                    if i / 3 >= j / 3 {
                        gl(i, j)
                    } else {
                        gl(j, i)
                    }
                },
            );
            for i in 0..n3 {
                for j in 0..n3 {
                    hess[i * n3 + j] -= a_x * c.f * 0.5 * (gs[(i, j)] + gs[(j, i)]);
                }
            }
        }
    }
    drop(ep);
    timers.add("hess_df_cross", t0.elapsed().as_secs_f64());
    linus_base::memory::mem_mark(timers, &format!("df_{call}_passes_cross"));
    drop(vinv);
    drop(gbar);
    drop(g);

    // ---- explicit second-derivative integrals ----
    timers.time("hess_df_2nd_deriv", || {
        second_derivative_terms(
            basis, &plan3, &mut ws3, aux, &ap, &ch, &dt, &cfit, &x, a_x, with_k, xop, &mut hess,
            par,
        )
    });
    linus_base::memory::mem_mark(timers, &format!("df_{call}_2nd_deriv"));
    DfHess2e { fxo, hess }
}

/// The explicit second-derivative integrals of the RI two-electron energy at fixed orbitals,
/// added to `hess` ((3 natom)^2 row-major): the two-center term (`Eri2DerivPlan::hessian` with
/// gamma_PQ = -c_P c_Q / 2 + a_x/2 sum_s f_s tr(X_s,P X_s,Q)) and the three-center term (the
/// auxiliary-lane kernel `Eri3DerivPlan::hessian` over batches of Gamma_P = c_P D_t - a_x sum_s f_s C_s
/// X_s,P C_s^T), for the fit coefficients `cfit`, the channels' symmetric X_s (\[P\]\[i\]\[j\];
/// unused without exchange), the total density `dt` (n^2) and the operator `xop` of the
/// three-center integrals: [`df_hessian_2e_op`]'s last step.
#[allow(clippy::too_many_arguments)]
fn second_derivative_terms(
    basis: &BasisSet,
    plan3: &Eri3DerivPlan<'_>,
    ws3: &mut Eri3DerivWorkspace,
    aux: &BasisSet,
    ap: &PairSet,
    ch: &[Chan],
    dt: &[f64],
    cfit: &[f64],
    x: &[Vec<f64>],
    a_x: f64,
    with_k: bool,
    xop: &Operator,
    hess: &mut [f64],
    par: Par,
) {
    let n = basis.nbf;
    let n2 = n * n;
    let naux = aux.nbf;
    let natom = basis.atom_shells.len().max(aux.atom_shells.len());
    {
        let mut gam2 = vec![0.0; naux * naux];
        for p in 0..naux {
            for q in 0..naux {
                gam2[p * naux + q] = -0.5 * cfit[p] * cfit[q];
            }
        }
        if with_k {
            for (s, c) in ch.iter().enumerate() {
                let xm = MatRef::from_row_major_slice(&x[s], naux, c.o * c.o);
                matmul(
                    MatMut::from_row_major_slice_mut(&mut gam2, naux, naux),
                    Accum::Add,
                    xm,
                    xm.transpose(),
                    0.5 * a_x * c.f,
                    par,
                );
            }
        }
        let opts = Eri2DerivOptions {
            thresh: DfThresholds::default().metric_deriv,
            op: Operator::Coulomb,
        };
        let plan =
            Eri2DerivPlan::new(aux, ap, DerivOrder::Two, opts).expect("auxiliary pairs of order 2");
        let n3a = 3 * aux.natom();
        let mut h2 = vec![0.0; n3a * n3a];
        plan.hessian(
            AoMatrix::row_major(&gam2, naux),
            &mut h2,
            &mut plan.workspace(),
        );
        hess.iter_mut().zip(&h2).for_each(|(a, b)| *a += b);
        // 3-center term: lane kernel (linus_ints::Eri3DerivPlan::hessian) over batches of the full
        // Gamma_P, also for Coulomb-only calls (pure functionals, the J call of a range-separated
        // hybrid) on Gamma_P = c_P D_t (u1an39/def2-SVP: 9.0 s with the separable generic
        // quartet kernel, 5.0 s on the lanes, same result)
        // the Hessian's pair layout and bounds, before the Gamma batches
        plan3.prepare_hessian();
        let mut h3 = vec![0.0; 9 * natom * natom];
        // aux-shell batches of Gamma_P = -a_x sum_s f_s C_s X_s,P C_s^T (<= 2^25 doubles per batch)
        let ash = &aux.shells;
        let nsh = ash.len();
        let maxf = ((1usize << 25) / n2).max(1);
        let mut batches: Vec<Range<usize>> = Vec::new();
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
        for sb in batches {
            let (s0, s1) = (sb.start, sb.end);
            let f0 = ash[s0].ao_start;
            let f1 = if s1 < nsh { ash[s1].ao_start } else { naux };
            // the Coulomb part c_P D_t goes into the same batch
            let full: Vec<f64> = {
                let mut gfull = vec![0.0; (f1 - f0) * n2];
                gfull.par_chunks_mut(n2).enumerate().for_each(|(pi, gp)| {
                    let p = f0 + pi;
                    gp.iter_mut().zip(dt).for_each(|(g, d)| *g = cfit[p] * d);
                    for (s, c) in ch.iter().enumerate() {
                        let o = c.o;
                        if o == 0 || !with_k {
                            continue;
                        }
                        let cm = MatRef::from_row_major_slice(&c.c, n, o);
                        let xp =
                            MatRef::from_row_major_slice(&x[s][p * o * o..(p + 1) * o * o], o, o);
                        let y = linus_base::linalg::mm(cm, xp, Par::Seq);
                        matmul(
                            MatMut::from_row_major_slice_mut(gp, n, n),
                            Accum::Add,
                            y.as_ref(),
                            cm.transpose(),
                            -a_x * c.f,
                            Par::Seq,
                        );
                    }
                });
                gfull
            };
            let blk = Eri3Block {
                shells: s0..s1,
                op: *xop,
                thresh: DfThresholds::default().deriv_weighted,
            };
            plan3.hessian(&blk, &full, &mut h3, ws3);
            linus_num::det::add_into(hess, &h3);
        }
    }
}

/// dst += alpha src.
fn add_to(mut dst: MatMut<'_, f64>, src: MatRef<'_, f64>, alpha: f64) {
    for j in 0..dst.ncols() {
        for i in 0..dst.nrows() {
            dst[(i, j)] += alpha * src[(i, j)];
        }
    }
}

/// A B^T for A, B (m x k, n x k) with large m, n and a long k: faer's parallel GEMM.
fn abt_split_wide(a: MatRef<'_, f64>, b: MatRef<'_, f64>, par: Par) -> Mat<f64> {
    linus_base::linalg::mm(a, b.transpose(), par)
}
