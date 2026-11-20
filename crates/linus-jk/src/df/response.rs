//! Density-fitted J and K of CPHF/CPKS first-order densities with occupied factors,
//! memory-lean (no 3-index AO tensor): for channel s with occupied orbitals C (n x o) and a
//! first-order density dD = A C^T + C A^T (A n x o), only the products with C are formed,
//!
//!   K\[dD\] C = sum_P [(P|..) A] X_P + sum_Q Bt_Q (A^T Bt_Q),
//!   J\[dD\] C = sum_Q Bt_Q gt_Q,   gt_Q = 2 sum_s tr(A_s^T Bt_s,Q)  (J of the total density),
//!
//! with X_P = V^-1 (P|ij) (fitted occupied pairs) and Bt = L^-1 (P|m i) (V = L L^T), both
//! stored (naux o^2 and naux n o), and the raw 3-center integrals (P|mn) recomputed per call,
//! one block of auxiliary functions at a time (the first term: per block one GEMM over
//! (P, n') for all densities side by side). Densities inside the occupied space, A = C M (the
//! D^x_oo of the Hessian), need neither: K\[dD\] C = sum_Q Bt_Q (M + M^T) Xt_Q, Xt_Q = C^T Bt_Q.
//! Same RI J/K as DfJk (Coulomb metric); per density about 2 n^2 o naux
//! + 6 n o^2 naux flops and no n x n intermediates.
//!
//! Range-separated hybrids (the exchange operator given at setup, [`DfResponseJk::new_op`]): the exchange is
//! that of the SCF's RI-K in Psi4's wK scheme for the operator w = a/r + b erf(omega r)/r,
//! K_w\[D\]_mn = sym[(ml|P) V^-1 (P|w|ns)] D_ls (Coulomb metric, Coulomb integrals on one side),
//! so with Wt = L^-1 (P|w|m i) and Xw = V^-1 (P|w|ij) stored next to Bt and X,
//!
//!   K_w\[dD\] C = 1/2 { sum_P [(P|..) A] Xw_P + sum_P [(P|w|..) A] X_P
//!                     + sum_Q [Bt_Q (A^T Wt_Q) + Wt_Q (A^T Bt_Q)] },
//!   occupied space: K_w\[dD\] C = 1/2 sum_Q [Bt_Q N Xwt_Q + Wt_Q N Xt_Q],  N = M + M^T,
//!
//! (for w = 1/r both reduce to the expressions above): twice the exchange work and storage of a
//! global hybrid; J is unchanged.
//!
//! The CPKS iterations of a two-level response only need an operator close to the main one (the
//! Hessian error is quadratic in the difference): for range-separated hybrids they use the
//! symmetric RI of w in its own metric V_w = (P|w|Q), (ml|w|ns) ~ (ml|w|P) V_w^-1 (Q|w|ns)
//! (`DfResponseJk::new_sym_op`; another RI approximation of the same exact exchange), which
//! costs what a global hybrid's exchange does: with Wt = L_w^-1 (P|w|m i), Xw = V_w^-1 (P|w|ij),
//! K C = sum_P [(P|w|..) A] Xw_P + sum_Q Wt_Q (A^T Wt_Q) (J from the Coulomb Bt as always).
//!
//! Precision and the two-level response (large systems, see [`SINGLE_WORK`] and
//! [`IterPlan`]): single-precision exchange GEMMs (u1an39/def2-SVP: 48 -> 27 s per CPKS
//! build of 117 densities, K C relative error 4e-6) and a second, cheaper J/K (def2-SVP-RI, single
//! precision) for the CPKS iterations, the main one giving the right-hand sides and the final
//! dF\[U\] C_o. Single precision only for CPKS tolerances safely above the residual where its
//! round-off stalls the solve ([`SINGLE_FLOOR`], [`auto_single`]).

use crate::df::common::{
    aux_blocks, solve_metric_mmajor, tensor_operator, AuxBlock, MetricSolve, Mo3c,
};
use crate::df::{metric_cholesky, CoulombMetric};
use faer::{Accum, Mat, MatMut, MatRef, Par};
use linus_base::linalg::{matmul, matmul_part, solve_upper_in_place, Whole};
use linus_base::terms::SINGLE_FLOOR;
use linus_core::timing::Timers;
use linus_ints::eri3c::{self, Aux3c};
use linus_ints::operator::Operator;
use linus_ints::{BasisSet, PairSet};
use rayon::prelude::*;
use std::ops::Range;
use std::time::Instant;

struct RespChan {
    o: usize,
    /// occupied orbitals, row-major n x o
    c: Vec<f64>,
    /// S C (n x o), to recognize densities inside the occupied space
    sc: Mat<f64>,
    /// L^-1 (P|m i), m-major [m][P][i]
    bt: Vec<f64>,
    /// V^-1 (P|ij), [P][i][j]
    x: Vec<f64>,
    /// exchange operator w (range separation; empty otherwise): L^-1 (P|w|m i), m-major
    wt: Vec<f64>,
    /// V^-1 (P|w|ij), [P][i][j]
    xw: Vec<f64>,
}

/// Memory-lean RI J/K for factored first-order densities (see the module docs).
pub struct DfResponseJk {
    n: usize,
    naux: usize,
    chans: Vec<RespChan>,
    mo: Mo3c<'static>,
    /// auxiliary basis (blocks of the integral-direct term)
    aux: BasisSet,
    /// the exchange operator a/r + b erf(omega r)/r of a range-separated hybrid (wK scheme)
    xop: Option<Operator>,
    /// the exchange is the symmetric RI of `xop` in its own metric (iteration J/K of a
    /// two-level response; see the module docs) instead of wK
    sym: bool,
    /// single-precision GEMMs for the exchange (Bt and X copies in f32): about twice the speed
    /// and ~4e-6 relative error in K C ([`DfResponseJk::configured`],
    /// `DfResponseJk::into_single_only`); J stays double precision unless the double copies were
    /// freed
    pub single: bool,
    bt32: Vec<Vec<f32>>,
    x32: Vec<Vec<f32>>,
    wt32: Vec<Vec<f32>>,
    xw32: Vec<Vec<f32>>,
    /// the stored tensors were freed by `release` (rebuilt by `restore`)
    released: bool,
    /// memory for the per-thread buffers of exchange term 1 ([`T1_SLOT_BYTES`]; sets the size of
    /// the density groups, `term1_slots`)
    pub t1_slot_bytes: usize,
    pub timers: Timers,
}

/// Dense integral block(s) of exchange term 1 (n x nq n doubles each): sets the block size nq.
const T1_DENSE_BYTES: usize = 256 << 20;
/// Buffers of the slots of exchange term 1 (two n x (ng o) each, `term1_slots`): sets the
/// density group size ng.
pub const T1_SLOT_BYTES: usize = 1 << 30;
/// Slots of exchange term 1 ([`term1_slots`]): accumulators the auxiliary functions are dealt
/// to, a fixed number (not the thread count) so that the sums do not depend on the thread count.
const T1_SLOTS: usize = 8;
/// Exchange work per density (n^2 o naux) above which the CPKS exchange runs in single
/// precision ([`DfResponseJk::configured`]).
pub const SINGLE_WORK: f64 = 1e9;
/// ... and above which the iterations use a separate, cheaper J/K ([`IterPlan`]).
pub const TWO_LEVEL_WORK: f64 = 1e10;
/// The smallest `response.tol` for which `precision.response = auto` runs single precision
/// (10 x [`SINGLE_FLOOR`]; [`auto_single`]): below it the solve would stall at the floor.
pub const SINGLE_MIN_TOL: f64 = 10.0 * SINGLE_FLOOR;

/// `precision.response = auto`: single-precision response exchange when the exchange work per
/// density, n^2 o naux, exceeds [`SINGLE_WORK`] (u1an39/def2-SVP with cc-pVTZ-jkfit: 7.7e10,
/// L-lactide: 1.8e9, CH3SH: ~1e7) and the requested max-abs residual `cphf_tol` is at least
/// [`SINGLE_MIN_TOL`]; unset (None), the defaults (1e-6, and 1e-5 in two-level responses) are.
pub fn auto_single(work: f64, cphf_tol: Option<f64>) -> bool {
    work > SINGLE_WORK && cphf_tol.is_none_or(|t| t >= SINGLE_MIN_TOL)
}

/// Exchange term 1 of one block of auxiliary functions for one group of ng densities:
/// K_t += f sum_q I_q T_q, T_q = [A_d X_q] (n x ng o) and I_q = im[:, q n..(q + 1) n] (`im` the
/// dense m-major block, n x nq n; `ap` the ng factors interleaved by AO row, (n ng) x o row-major,
/// which read as n x (ng o) is T_q's layout, see [`interleave_factors`]; `xb` the block's X_q,
/// [q][i][j]). The auxiliary functions are dealt to the slots ([`T1_SLOTS`] of them, each a T_q
/// buffer and an accumulator K_t, column-major n x (ng o)) round robin, each slot's in order with
/// sequential GEMMs, and the slots run in parallel: up to [`T1_SLOTS`] threads stay busy, and
/// the sums depend neither on the thread count nor on the run (the slots are summed in order,
/// [`flush_slots`]). The products of a group are parts of the products of all `nd` densities in
/// one group ([`matmul_part`]): the bits do not depend on how the memory budget groups them.
/// (The batched
/// form it replaces, T for the block's nq functions and one parallel GEMM, ran at 4.5-5 of 8
/// cores: the T buffer's budget left nq at 5-14, so its T step had fewer tasks than threads, and K was
/// scattered into the outputs serially after every block.)
#[allow(clippy::too_many_arguments)]
fn term1_slots<T: faer::traits::ComplexField + Copy + Default + Send + Sync>(
    im: MatRef<'_, T>,
    ap: &[T],
    ng: usize,
    nd: usize,
    xb: &[T],
    n: usize,
    nq: usize,
    o: usize,
    f: T,
    slots: &mut [(Vec<T>, Vec<T>)],
) {
    let w = ng * o;
    let (t_whole, k_whole) = (
        Whole {
            rows: n * nd,
            cols: o,
        },
        Whole {
            rows: n,
            cols: nd * o,
        },
    );
    let nslot = slots.len();
    let one = faer::traits::ComplexField::one_impl();
    let apm = MatRef::from_row_major_slice(&ap[..n * w], n * ng, o);
    slots.par_iter_mut().enumerate().for_each(|(t, (tb, kb))| {
        for q in (t..nq).step_by(nslot) {
            let xq = MatRef::from_row_major_slice(&xb[q * o * o..(q + 1) * o * o], o, o);
            matmul_part(
                t_whole,
                MatMut::from_row_major_slice_mut(&mut tb[..n * w], n * ng, o),
                Accum::Replace,
                apm,
                xq,
                one,
                Par::Seq,
            );
            let tq = MatRef::from_row_major_slice(&tb[..n * w], n, w);
            matmul_part(
                k_whole,
                MatMut::from_column_major_slice_mut(&mut kb[..n * w], n, w),
                Accum::Add,
                im.subcols(q * n, n),
                tq,
                f,
                Par::Seq,
            );
        }
    });
}

/// Add the slots' accumulators of [`term1_slots`] (group `grp` of the densities) to `kout`
/// (n x o per density) and zero them.
fn flush_slots<T: Copy + Default + Send + Sync>(
    kout: &mut [Mat<f64>],
    grp: &[usize],
    n: usize,
    o: usize,
    slots: &mut [(Vec<T>, Vec<T>)],
    to64: impl Fn(T) -> f64 + Sync,
) {
    let w = grp.len() * o;
    {
        let ks: Vec<&[T]> = slots.iter().map(|s| &s.1[..n * w]).collect();
        let mut targets: Vec<(usize, &mut Mat<f64>)> = kout
            .iter_mut()
            .enumerate()
            .filter_map(|(d, k)| grp.iter().position(|&g| g == d).map(|gi| (gi, k)))
            .collect();
        targets.par_iter_mut().for_each(|(gi, k)| {
            for j in 0..o {
                let c = (*gi * o + j) * n;
                for i in 0..n {
                    let s: f64 = ks.iter().map(|kb| to64(kb[c + i])).sum();
                    k[(i, j)] += s;
                }
            }
        });
    }
    slots
        .par_iter_mut()
        .for_each(|s| s.1[..n * w].iter_mut().for_each(|v| *v = T::default()));
}

/// The factors A_d (n x o) of `idx` interleaved by AO row for [`term1_slots`]: row-major
/// (n nd) x o with row m nd + d.
fn interleave_factors<T: Copy + Default + Send + Sync>(
    a: &[Mat<f64>],
    idx: &[usize],
    conv: impl Fn(f64) -> T + Sync,
) -> Vec<T> {
    let (nd, n) = (idx.len(), a.first().map_or(0, |m| m.nrows()));
    let o = a.first().map_or(0, |m| m.ncols());
    let mut v = vec![T::default(); n * nd * o];
    if v.is_empty() {
        // no densities, or a channel without occupied orbitals
        return v;
    }
    v.par_chunks_mut(nd * o).enumerate().for_each(|(m, row)| {
        for (dd, &d) in idx.iter().enumerate() {
            for j in 0..o {
                row[dd * o + j] = conv(a[d][(m, j)]);
            }
        }
    });
    v
}

/// X = L^-T C^T Bt ([P][i][j], symmetrized in ij) from the m-major Bt = L^-1 (P|m i).
fn fit_occ_pairs(
    l: MatRef<'_, f64>,
    c: &[f64],
    bt: &[f64],
    n: usize,
    naux: usize,
    o: usize,
    par: Par,
) -> Vec<f64> {
    let cm = MatRef::from_row_major_slice(c, n, o);
    let xt = linus_base::linalg::mm(
        cm.transpose(),
        MatRef::from_row_major_slice(bt, n, naux * o),
        par,
    );
    let mut xc = Mat::<f64>::from_fn(naux, o * o, |p, ij| {
        let (i, j) = (ij / o, ij % o);
        0.5 * (xt[(i, p * o + j)] + xt[(j, p * o + i)])
    });
    solve_upper_in_place(l.transpose(), xc.as_mut(), par);
    (0..naux * o * o)
        .map(|t| xc[(t / (o * o), t % (o * o))])
        .collect()
}

/// Pair-major integrals of the operator w = full/r + erf erf(omega r)/r for an auxiliary block:
/// `buf` holds the Coulomb ones on entry, `wbuf` receives w's (one extra erf pass).
pub(crate) fn op_ints(
    mo: &Mo3c<'_>,
    op: &Operator,
    sh: Range<usize>,
    nq: usize,
    buf: &[f64],
    wbuf: &mut [f64],
) {
    let len = mo.nx * nq;
    if op.long_range() != 0.0 && op.omega() > 0.0 {
        mo.ints(sh, nq, op.omega(), wbuf);
        let (a, b) = (op.coulomb(), op.long_range());
        wbuf[..len]
            .par_chunks_mut(1 << 14)
            .zip(buf[..len].par_chunks(1 << 14))
            .for_each(|(w, c)| w.iter_mut().zip(c).for_each(|(u, v)| *u = a * v + b * *u));
    } else {
        let a = op.coulomb();
        wbuf[..len]
            .par_chunks_mut(1 << 14)
            .zip(buf[..len].par_chunks(1 << 14))
            .for_each(|(w, c)| w.iter_mut().zip(c).for_each(|(u, v)| *u = a * v));
    }
}

/// The exchange form of term 2 (see [`term2_chunk`]).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum K2 {
    /// K += Bt G (global hybrids and HF)
    Coul,
    /// K += 1/2 (Bt Gw + Wt G) (range separation, wK)
    Wk,
    /// K += Wt Gw (symmetric RI of the operator)
    Sym,
}

/// The factors of a chunk of densities for [`term2_chunk`].
enum Fac<'a, T> {
    /// general densities: the A_d side by side, n x (o nds), column j nds + d
    Gen(MatRef<'a, T>),
    /// occupied space: the N_d = M_d + M_d^T stacked, (o nds) x o with row j nds + d, and C
    Occ(MatRef<'a, T>, MatRef<'a, T>),
}

/// Term 2 of the projected exchange and the J contraction gamma for a chunk of nds densities
/// (see [`DfResponseJk::jk_proj`]): for every auxiliary function Q, G_Q = F^T Bt_Q (general) or
/// N (C^T Bt_Q) (occupied space), stored row-major as (o nds) x o, which is [j][(d, i)];
/// gamma_Q,d = fg sum_j G_Q[j][(d, j)]; K_d += Bt_Q G_Q (wK: 1/2 [Bt_Q Gw_Q + Wt_Q G_Q];
/// symmetric operator RI: Wt_Q Gw_Q; Gw from Wt). All densities of the chunk go through one pair
/// of GEMMs per Q with no transposed copies; K accumulates in T and is added to the f64 result
/// every 32 Q. Returns gamma [Q][d] and K (n x (nds o), column d o + i).
#[allow(clippy::too_many_arguments)]
fn term2_chunk<T>(
    fac: Fac<'_, T>,
    bt: &[T],
    wt: &[T],
    n: usize,
    o: usize,
    nds: usize,
    naux: usize,
    mode: K2,
    fg: f64,
    want_k: bool,
) -> (Vec<f64>, Mat<f64>)
where
    T: faer::traits::ComplexField + Copy + Default + Send + Sync + Into<f64> + From<f32>,
{
    let one: T = T::from(1.0f32);
    let half: T = T::from(0.5f32);
    let w = nds * o;
    let mut gamma = vec![0.0; naux * nds];
    let mut gbuf = vec![T::default(); w * o];
    let mut gwbuf = vec![T::default(); if mode != K2::Coul && want_k { w * o } else { 0 }];
    let mut xt = Mat::<T>::zeros(o, o);
    let kw = if want_k { w } else { 0 };
    let mut kacc = Mat::<T>::zeros(n, kw);
    let mut k64 = Mat::<f64>::zeros(n, kw);
    let form = |src: MatRef<'_, T>, dst: &mut [T], xt: &mut Mat<T>| {
        let g = MatMut::from_row_major_slice_mut(dst, w, o);
        match &fac {
            Fac::Gen(f) => matmul(g, Accum::Replace, f.transpose(), src, one, Par::Seq),
            Fac::Occ(nm, c) => {
                matmul(
                    xt.as_mut(),
                    Accum::Replace,
                    c.transpose(),
                    src,
                    one,
                    Par::Seq,
                );
                matmul(g, Accum::Replace, *nm, xt.as_ref(), one, Par::Seq);
            }
        }
    };
    let flush = |kacc: &mut Mat<T>, k64: &mut Mat<f64>| {
        for j in 0..kw {
            for i in 0..n {
                k64[(i, j)] += kacc[(i, j)].into();
            }
        }
        *kacc = Mat::zeros(n, kw);
    };
    for q in 0..naux {
        let bq = MatRef::from_row_major_slice_with_stride(&bt[q * o..], n, o, naux * o);
        form(bq, &mut gbuf, &mut xt);
        for dd in 0..nds {
            let mut sg = 0.0;
            for j in 0..o {
                sg += gbuf[(j * nds + dd) * o + j].into();
            }
            gamma[q * nds + dd] = fg * sg;
        }
        if !want_k {
            continue;
        }
        let gv = MatRef::from_row_major_slice(&gbuf, o, w);
        match mode {
            K2::Coul => matmul(kacc.as_mut(), Accum::Add, bq, gv, one, Par::Seq),
            K2::Wk | K2::Sym => {
                let wq = MatRef::from_row_major_slice_with_stride(&wt[q * o..], n, o, naux * o);
                form(wq, &mut gwbuf, &mut xt);
                let gwv = MatRef::from_row_major_slice(&gwbuf, o, w);
                if mode == K2::Sym {
                    matmul(kacc.as_mut(), Accum::Add, wq, gwv, one, Par::Seq);
                } else {
                    matmul(kacc.as_mut(), Accum::Add, bq, gwv, half, Par::Seq);
                    matmul(kacc.as_mut(), Accum::Add, wq, gv, half, Par::Seq);
                }
            }
        }
        if q % 32 == 31 {
            flush(&mut kacc, &mut k64);
        }
    }
    if want_k {
        flush(&mut kacc, &mut k64);
    }
    (gamma, k64)
}

impl DfResponseJk {
    /// Setup for the occupied orbitals `occ` (one matrix per channel) and the AO overlap `s` on
    /// the Coulomb metric `metric` of `aux`: 3-center MO integrals (one pass),
    /// Bt = L^-1 (P|m i), X = V^-1 (P|ij).
    #[cfg(any(test, feature = "reference"))]
    pub fn new(
        basis: &BasisSet,
        pairs: &PairSet,
        aux: &BasisSet,
        metric: &CoulombMetric,
        occ: &[MatRef<'_, f64>],
        s: MatRef<'_, f64>,
        par: Par,
    ) -> Self {
        Self::new_op(basis, pairs, aux, metric, occ, s, None, par)
    }

    /// `DfResponseJk::new` with the exchange operator `xop` (range-separated hybrids, wK scheme;
    /// see the module docs): also Wt = L^-1 (P|w|m i) and Xw = V^-1 (P|w|ij).
    #[allow(clippy::too_many_arguments)]
    pub fn new_op(
        basis: &BasisSet,
        pairs: &PairSet,
        aux: &BasisSet,
        metric: &CoulombMetric,
        occ: &[MatRef<'_, f64>],
        s: MatRef<'_, f64>,
        xop: Option<Operator>,
        par: Par,
    ) -> Self {
        Self::with_mode(basis, pairs, aux, metric, occ, s, xop, false, false, par)
    }

    /// The symmetric RI of the exchange operator `xop` in its own metric (P|xop|Q), for the
    /// iterations of a two-level response (module docs); needs a positive definite operator
    /// metric ([`sym_op_usable`]).
    #[cfg(any(test, feature = "reference"))]
    #[allow(clippy::too_many_arguments)]
    pub fn new_sym_op(
        basis: &BasisSet,
        pairs: &PairSet,
        aux: &BasisSet,
        metric: &CoulombMetric,
        occ: &[MatRef<'_, f64>],
        s: MatRef<'_, f64>,
        xop: Operator,
        par: Par,
    ) -> Self {
        assert!(
            sym_op_usable(&xop),
            "symmetric operator RI needs a/r + b erf with a > 0, a + b > 0"
        );
        Self::with_mode(
            basis,
            pairs,
            aux,
            metric,
            occ,
            s,
            Some(xop),
            true,
            false,
            par,
        )
    }

    #[allow(clippy::too_many_arguments)]
    /// `f32_only`: single precision only from the start (as `DfResponseJk::into_single_only`
    /// without the transient double-precision copies of every tensor).
    #[allow(clippy::too_many_arguments)]
    pub fn with_mode(
        basis: &BasisSet,
        pairs: &PairSet,
        aux: &BasisSet,
        metric: &CoulombMetric,
        occ: &[MatRef<'_, f64>],
        s: MatRef<'_, f64>,
        xop: Option<Operator>,
        sym: bool,
        f32_only: bool,
        par: Par,
    ) -> Self {
        let n = basis.nbf;
        let naux = aux.nbf;
        let mo = Mo3c::new_owned(basis, pairs, Aux3c::new(aux));
        let chans: Vec<RespChan> = occ
            .iter()
            .map(|c| {
                let o = c.ncols();
                RespChan {
                    o,
                    c: (0..n * o).map(|x| c[(x / o, x % o)]).collect(),
                    sc: linus_base::linalg::mm(s, *c, par),
                    bt: Vec::new(),
                    x: Vec::new(),
                    wt: Vec::new(),
                    xw: Vec::new(),
                }
            })
            .collect();
        let mut r = DfResponseJk {
            n,
            naux,
            chans,
            mo,
            aux: aux.clone(),
            xop,
            sym,
            single: false,
            bt32: Vec::new(),
            x32: Vec::new(),
            wt32: Vec::new(),
            xw32: Vec::new(),
            released: false,
            t1_slot_bytes: T1_SLOT_BYTES,
            timers: Timers::default(),
        };
        r.build(metric, par, f32_only);
        r
    }

    /// The stored tensors from the 3-center integrals: Bt, X (and Wt, Xw), double precision
    /// (symmetric operator RI: Bt for J, Wt and Xw in the operator's metric; no X).
    /// `f32_only`: each tensor goes to single precision as soon as it is fitted (its double copy
    /// freed), and the builder runs in single precision only.
    fn build(&mut self, metric: &CoulombMetric, par: Par, f32_only: bool) {
        let (n, naux) = (self.n, self.naux);
        assert_eq!(
            metric.naux(),
            naux,
            "the metric of another auxiliary basis set"
        );
        // the symmetric RI's operator metric a (P|Q) + b (P|erf|Q), factored here
        let lw = match (self.sym, self.xop) {
            (true, Some(op)) => {
                let aux3c = Aux3c::new(&self.aux);
                let mut v = eri3c::metric(&aux3c, 0.0, &Operator::Coulomb);
                let ve = eri3c::metric(&aux3c, 0.0, &tensor_operator(op.omega()));
                v.iter_mut()
                    .zip(&ve)
                    .for_each(|(c, e)| *c = op.coulomb() * *c + op.long_range() * e);
                Some(
                    metric_cholesky(Mat::from_fn(naux, naux, |i, j| v[i * naux + j]), par)
                        .expect("auxiliary metric not positive definite"),
                )
            }
            _ => None,
        };
        let lr = metric.factor();
        let lwr = lw.as_ref().map_or(lr, |m| m.as_ref());
        let sym = self.sym;
        let xop = self.xop;
        let long_range = xop.is_some();
        let (mo, aux, chans) = (&self.mo, &self.aux, &mut self.chans);
        for ch in chans.iter_mut() {
            ch.bt = vec![0.0; n * naux * ch.o];
            ch.wt = if long_range {
                vec![0.0; n * naux * ch.o]
            } else {
                Vec::new()
            };
        }
        // pass over auxiliary blocks: (P|m i) into Bt (m-major), (P|w|m i) into Wt
        let ot: usize = chans.iter().map(|c| c.o).sum();
        let nbuf = if long_range { 2 } else { 1 };
        let maxf = ((1usize << 30) / (8 * nbuf * (mo.nx + n * ot).max(1))).clamp(16, naux.max(16));
        let mut buf = vec![0.0; mo.nx * maxf];
        let mut wbuf = if long_range {
            vec![0.0; mo.nx * maxf]
        } else {
            Vec::new()
        };
        for AuxBlock {
            p0, nq, shells: sh, ..
        } in aux_blocks(&aux.shells, maxf, false)
        {
            mo.ints(sh.clone(), nq, 0.0, &mut buf);
            if let Some(op) = &xop {
                op_ints(mo, op, sh, nq, &buf, &mut wbuf);
            }
            for ch in chans.iter_mut() {
                let o = ch.o;
                if o == 0 {
                    continue;
                }
                let zb = mo.half(&buf, nq, &ch.c, o); // [m][(q, i)]
                ch.bt
                    .par_chunks_mut(naux * o)
                    .enumerate()
                    .for_each(|(m, zm)| {
                        zm[p0 * o..(p0 + nq) * o].copy_from_slice(&zb[m * nq * o..(m + 1) * nq * o])
                    });
                if long_range {
                    let zw = mo.half(&wbuf, nq, &ch.c, o);
                    ch.wt
                        .par_chunks_mut(naux * o)
                        .enumerate()
                        .for_each(|(m, zm)| {
                            zm[p0 * o..(p0 + nq) * o]
                                .copy_from_slice(&zw[m * nq * o..(m + 1) * nq * o])
                        });
                }
            }
        }
        drop((buf, wbuf));
        let to_f32 = |v: Vec<f64>| -> Vec<f32> { v.par_iter().map(|&x| x as f32).collect() };
        #[allow(clippy::type_complexity)]
        let mut f32s: Vec<(Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>)> =
            (0..chans.len()).map(|_| Default::default()).collect();
        for (s_, ch) in chans.iter_mut().enumerate() {
            let o = ch.o;
            if o == 0 {
                continue;
            }
            // Bt = L^-1 B per m slab; X = L^-T C^T Bt (and the same for w; in w's metric for
            // the symmetric operator RI, whose exchange needs no X)
            solve_metric_mmajor(lr, &mut ch.bt, o, MetricSolve::Lower);
            if !sym {
                ch.x = fit_occ_pairs(lr, &ch.c, &ch.bt, n, naux, o, par);
            }
            if f32_only {
                f32s[s_].0 = to_f32(std::mem::take(&mut ch.bt));
                f32s[s_].1 = to_f32(std::mem::take(&mut ch.x));
            }
            if long_range {
                solve_metric_mmajor(lwr, &mut ch.wt, o, MetricSolve::Lower);
                ch.xw = fit_occ_pairs(lwr, &ch.c, &ch.wt, n, naux, o, par);
                if f32_only {
                    f32s[s_].2 = to_f32(std::mem::take(&mut ch.wt));
                    f32s[s_].3 = to_f32(std::mem::take(&mut ch.xw));
                }
            }
        }
        if f32_only {
            self.single = true;
            self.bt32 = f32s.iter_mut().map(|t| std::mem::take(&mut t.0)).collect();
            self.x32 = f32s.iter_mut().map(|t| std::mem::take(&mut t.1)).collect();
            self.wt32 = f32s.iter_mut().map(|t| std::mem::take(&mut t.2)).collect();
            self.xw32 = f32s.into_iter().map(|t| t.3).collect();
        }
    }

    /// Free the stored tensors (the builder is idle; see
    /// [`linus_base::terms::ResponseJk::release`]).
    pub fn release_storage(&mut self) {
        for ch in self.chans.iter_mut() {
            ch.bt = Vec::new();
            ch.x = Vec::new();
            ch.wt = Vec::new();
            ch.xw = Vec::new();
        }
        self.bt32 = Vec::new();
        self.x32 = Vec::new();
        self.wt32 = Vec::new();
        self.xw32 = Vec::new();
        self.released = true;
    }

    /// Rebuild the tensors freed by [`DfResponseJk::release_storage`] (one 3-center integral pass and
    /// the fit, as at setup): single precision only for a single-precision builder (its last use
    /// in a two-level response is the final dF\[U\] C_o, general densities only), else double.
    pub fn restore_storage(&mut self, par: Par) {
        if !self.released {
            return;
        }
        // single-precision builders come back in single precision only (J too): the restored
        // builder serves the final dF[U] C_o of general densities, whose exchange was single
        // precision anyway; no double copies at the memory peak
        let single = self.single;
        // a metric of its own: the geometry's is released after the response's setup, and
        // holding one through the iterations would raise their peak (crate::context)
        let metric =
            CoulombMetric::new(&self.aux, par).expect("auxiliary metric not positive definite");
        self.build(&metric, par, single);
        self.released = false;
    }

    /// The exchange operator (range-separated hybrids), if any.
    #[cfg(any(test, feature = "reference"))]
    pub fn exchange_operator(&self) -> Option<Operator> {
        self.xop
    }

    /// Single-precision copies of Bt and X (and Wt, Xw).
    fn make_single(&mut self) {
        let f32s = |v: &Vec<f64>| -> Vec<f32> { v.par_iter().map(|&x| x as f32).collect() };
        self.bt32 = self.chans.iter().map(|ch| f32s(&ch.bt)).collect();
        self.x32 = self.chans.iter().map(|ch| f32s(&ch.x)).collect();
        self.wt32 = self.chans.iter().map(|ch| f32s(&ch.wt)).collect();
        self.xw32 = self.chans.iter().map(|ch| f32s(&ch.xw)).collect();
    }

    /// Single precision only (J too): the double-precision Bt and X are freed. For the CPKS
    /// iteration J/K of a two-level response, whose result only steers the iterations.
    #[cfg(any(test, feature = "reference"))]
    pub fn into_single_only(mut self) -> Self {
        self.single = true;
        self.make_single();
        for ch in self.chans.iter_mut() {
            ch.bt = Vec::new();
            ch.x = Vec::new();
            ch.wt = Vec::new();
            ch.xw = Vec::new();
        }
        self
    }

    /// The precision policy of the CPHF/CPKS exchange (`precision.response`: fp32, fp64 or
    /// auto; `cphf_tol`: the `response.tol` requested, None for the defaults) and the
    /// memory of the term-1 buffers (`slot_bytes`, [`DfResponseJk::t1_slot_bytes`]). Auto:
    /// [`auto_single`]. K C then carries ~4e-6 relative error (the default CPKS tolerance is
    /// 1e-6) and the Hessian changes by ~1e-8 Eh/bohr^2 (frequencies by ~1e-5 cm^-1).
    pub fn configured(mut self, single: bool, slot_bytes: usize) -> Self {
        self.t1_slot_bytes = slot_bytes;
        self.single = single;
        if self.single {
            // the double-precision X (Xw) serves only the double-precision exchange
            self.make_single();
            for ch in self.chans.iter_mut() {
                ch.x = Vec::new();
                ch.xw = Vec::new();
            }
        }
        self
    }

    /// For densities d and channels s with factors a\[s\]\[d\] (n x o_s): J[sum_s dD_s,d] C_s and
    /// (with `want_k`) K\[dD_s,d\] C_s, as \[s\]\[d\] lists of n x o_s matrices (K of the exchange
    /// operator in the wK scheme when one is set). Runs on the current rayon pool (`_par`: the
    /// work is split into per-thread tasks of sequential kernels).
    #[allow(clippy::type_complexity)]
    pub fn jk_proj(
        &mut self,
        a: &[Vec<Mat<f64>>],
        want_k: bool,
        _par: Par,
    ) -> (Vec<Vec<Mat<f64>>>, Vec<Vec<Mat<f64>>>) {
        let (n, naux) = (self.n, self.naux);
        let nd = a.first().map_or(0, |v| v.len());
        let nch = self.chans.len();
        let single = self.single;
        assert!(
            !self.released,
            "DfResponseJk: tensors released (restore first)"
        );
        if single && self.bt32.len() != nch {
            self.make_single();
        }
        let long_range = self.xop.is_some();
        let sym = self.sym;
        // without the double-precision copies (iteration J/K) everything runs in single precision
        let have64 = self.chans.iter().all(|c| !c.bt.is_empty() || c.o == 0);
        let t0 = Instant::now();
        // densities inside the occupied space: A = C M, M = C^T S A (per channel and density)
        let occm: Vec<Vec<Option<Mat<f64>>>> = (0..nch)
            .map(|s| {
                let ch = &self.chans[s];
                let cm = MatRef::from_row_major_slice(&ch.c, n, ch.o);
                a[s].par_iter()
                    .map(|ad| {
                        if ch.o == 0 || !have64 {
                            return None;
                        }
                        let m = linus_base::linalg::mm(ch.sc.transpose(), ad.as_ref(), Par::Seq);
                        let back = linus_base::linalg::mm(cm, m.as_ref(), Par::Seq);
                        let mx = linus_base::linalg::max_abs(ad.as_ref());
                        let err = linus_base::linalg::max_abs((ad - &back).as_ref());
                        (err <= 1e-12 * mx.max(1e-300)).then_some(m)
                    })
                    .collect()
            })
            .collect();
        self.timers
            .add("resp_occ_check", t0.elapsed().as_secs_f64());
        // exchange term 2 (and the occupied-space densities' K) and gt, per auxiliary function Q
        // for chunks of densities side by side (term2_chunk; range separation: of Bt and Wt,
        // K C += 1/2 [Bt (A^T Wt) + Wt (A^T Bt)])
        let t0 = Instant::now();
        let mut gt = vec![0.0; naux * nd]; // [Q][d]
        let mut kout: Vec<Vec<Mat<f64>>> = self
            .chans
            .iter()
            .map(|ch| (0..nd).map(|_| Mat::zeros(n, ch.o)).collect())
            .collect();
        let mode = if !long_range {
            K2::Coul
        } else if sym {
            K2::Sym
        } else {
            K2::Wk
        };
        for (s, ch) in self.chans.iter().enumerate() {
            let o = ch.o;
            if o == 0 {
                continue;
            }
            let cm = MatRef::from_row_major_slice(&ch.c, n, o);
            let occ: Vec<usize> = (0..nd).filter(|&d| occm[s][d].is_some()).collect();
            let gen: Vec<usize> = (0..nd).filter(|&d| occm[s][d].is_none()).collect();
            // fixed chunks (independent of the thread count): about 8 per group
            let chunks = |v: &[usize]| -> Vec<Vec<usize>> {
                v.chunks(v.len().div_ceil(8).clamp(1, 32))
                    .map(|c| c.to_vec())
                    .collect()
            };
            type Res = (Vec<usize>, Vec<f64>, Mat<f64>);
            let mut res: Vec<Res> = Vec::new();
            // occupied space (double precision): G_Q = N (C^T Bt_Q), N = M + M^T, rows (j, d)
            res.extend(
                chunks(&occ)
                    .into_par_iter()
                    .map(|dl| {
                        let nds = dl.len();
                        let nm: Vec<Mat<f64>> = dl
                            .iter()
                            .map(|&d| {
                                let m = occm[s][d].as_ref().unwrap();
                                m + m.transpose()
                            })
                            .collect();
                        let mut f = vec![0.0f64; o * nds * o];
                        for (dd, nn) in nm.iter().enumerate() {
                            for j in 0..o {
                                for k in 0..o {
                                    f[(j * nds + dd) * o + k] = nn[(j, k)];
                                }
                            }
                        }
                        let (g, k) = term2_chunk(
                            Fac::Occ(MatRef::from_row_major_slice(&f, o * nds, o), cm),
                            &ch.bt,
                            &ch.wt,
                            n,
                            o,
                            nds,
                            naux,
                            mode,
                            1.0,
                            want_k,
                        );
                        (dl, g, k)
                    })
                    .collect::<Vec<Res>>(),
            );
            // general densities: G_Q = A^T Bt_Q with the factors side by side as columns (j, d)
            res.extend(
                chunks(&gen)
                    .into_par_iter()
                    .map(|dl| {
                        let nds = dl.len();
                        if single {
                            let f = Mat::<f32>::from_fn(n, o * nds, |m, c| {
                                a[s][dl[c % nds]][(m, c / nds)] as f32
                            });
                            let (g, k) = term2_chunk(
                                Fac::Gen(f.as_ref()),
                                &self.bt32[s],
                                &self.wt32[s],
                                n,
                                o,
                                nds,
                                naux,
                                mode,
                                2.0,
                                want_k,
                            );
                            (dl, g, k)
                        } else {
                            let f = Mat::<f64>::from_fn(n, o * nds, |m, c| {
                                a[s][dl[c % nds]][(m, c / nds)]
                            });
                            let (g, k) = term2_chunk(
                                Fac::Gen(f.as_ref()),
                                &ch.bt,
                                &ch.wt,
                                n,
                                o,
                                nds,
                                naux,
                                mode,
                                2.0,
                                want_k,
                            );
                            (dl, g, k)
                        }
                    })
                    .collect::<Vec<Res>>(),
            );
            for (dl, g, k) in res {
                let nds = dl.len();
                for (dd, &d) in dl.iter().enumerate() {
                    for q in 0..naux {
                        gt[q * nd + d] += g[q * nds + dd];
                    }
                    if want_k {
                        kout[s][d] += k.subcols(dd * o, o);
                    }
                }
            }
        }
        self.timers.add("resp_k2", t0.elapsed().as_secs_f64());
        // J C_s = sum_Q Bt_s,Q gt_Q: per m, (o x naux) (naux x nd)
        let t0 = Instant::now();
        let gm = MatRef::from_row_major_slice(&gt, naux, nd);
        let gt32: Vec<f32> = if have64 {
            Vec::new()
        } else {
            gt.iter().map(|&v| v as f32).collect()
        };
        let jout: Vec<Vec<Mat<f64>>> = self
            .chans
            .iter()
            .enumerate()
            .map(|(s, ch)| {
                let o = ch.o;
                let rows: Vec<Mat<f64>> = if have64 {
                    (0..n)
                        .into_par_iter()
                        .map(|m| {
                            linus_base::linalg::mm(
                                MatRef::from_row_major_slice(
                                    &ch.bt[m * naux * o..(m + 1) * naux * o],
                                    naux,
                                    o,
                                )
                                .transpose(),
                                gm,
                                Par::Seq,
                            )
                        })
                        .collect()
                } else {
                    let g32 = MatRef::from_row_major_slice(&gt32, naux, nd);
                    (0..n)
                        .into_par_iter()
                        .map(|m| {
                            let mut r = Mat::<f32>::zeros(o, nd);
                            matmul(
                                r.as_mut(),
                                Accum::Replace,
                                MatRef::from_row_major_slice(
                                    &self.bt32[s][m * naux * o..(m + 1) * naux * o],
                                    naux,
                                    o,
                                )
                                .transpose(),
                                g32,
                                1.0f32,
                                Par::Seq,
                            );
                            Mat::from_fn(o, nd, |i, d| r[(i, d)] as f64)
                        })
                        .collect()
                };
                (0..nd)
                    .map(|d| Mat::from_fn(n, o, |m, i| rows[m][(i, d)]))
                    .collect()
            })
            .collect();
        self.timers.add("resp_j", t0.elapsed().as_secs_f64());
        if !want_k {
            return (jout, kout);
        }
        // exchange term 1 for the general densities, integral-direct:
        // K C += sum_P (P|..) (A X_P), per block of auxiliary functions with the dense m-major
        // integral block [m][(P, n')] (range separation: 1/2 [(P|..) (A Xw_P) + (P|w|..) (A X_P)],
        // a second dense block of the operator's integrals), one task per thread (term1_slots)
        // for groups of densities whose slot buffers fit in T1_SLOT_BYTES
        let t0 = Instant::now();
        let gen: Vec<Vec<usize>> = (0..nch)
            .map(|s| {
                (0..nd)
                    .filter(|&d| occm[s][d].is_none() && self.chans[s].o > 0)
                    .collect()
            })
            .collect();
        if gen.iter().any(|g| !g.is_empty()) {
            let tsz = if single { 4 } else { 8 };
            let nblk = if long_range && !sym { 2 } else { 1 };
            let maxq = (T1_DENSE_BYTES / (8 * nblk * n * n).max(1)).max(1);
            let blocks = aux_blocks(&self.aux.shells, maxq, false);
            let maxnq = blocks.iter().map(|b| b.nq).max().unwrap_or(0);
            let nslot = T1_SLOTS;
            let groups: Vec<Vec<Vec<usize>>> = (0..nch)
                .map(|s| {
                    let per = (self.t1_slot_bytes
                        / (2 * nslot * n * self.chans[s].o.max(1) * tsz).max(1))
                    .max(1);
                    gen[s].chunks(per).map(|c| c.to_vec()).collect()
                })
                .collect();
            let maxw = (0..nch)
                .flat_map(|s| groups[s].iter().map(move |g| (s, g.len())))
                .map(|(s, l)| l * self.chans[s].o)
                .max()
                .unwrap_or(0);
            let mut ibuf = vec![0.0; self.mo.nx * maxnq];
            let mut wbuf = if long_range {
                vec![0.0; self.mo.nx * maxnq]
            } else {
                Vec::new()
            };
            let mut dense = if sym {
                Vec::new()
            } else {
                vec![0.0; n * maxnq * n]
            };
            let mut densew = if long_range {
                vec![0.0; n * maxnq * n]
            } else {
                Vec::new()
            };
            let mut dense32: Vec<f32> = if single && !sym {
                vec![0.0; n * maxnq * n]
            } else {
                Vec::new()
            };
            let mut densew32: Vec<f32> = if single && long_range {
                vec![0.0; n * maxnq * n]
            } else {
                Vec::new()
            };
            // the factors of each group interleaved by AO row (term1_slots), per channel
            let ap32: Vec<Vec<Vec<f32>>> = if single {
                (0..nch)
                    .map(|s| {
                        groups[s]
                            .iter()
                            .map(|g| interleave_factors(&a[s], g, |x| x as f32))
                            .collect()
                    })
                    .collect()
            } else {
                Vec::new()
            };
            let ap64: Vec<Vec<Vec<f64>>> = if single {
                Vec::new()
            } else {
                (0..nch)
                    .map(|s| {
                        groups[s]
                            .iter()
                            .map(|g| interleave_factors(&a[s], g, |x| x))
                            .collect()
                    })
                    .collect()
            };
            let mut slots32: Vec<(Vec<f32>, Vec<f32>)> = if single {
                (0..nslot)
                    .map(|_| (vec![0.0; n * maxw], vec![0.0; n * maxw]))
                    .collect()
            } else {
                Vec::new()
            };
            let mut slots64: Vec<(Vec<f64>, Vec<f64>)> = if single {
                Vec::new()
            } else {
                (0..nslot)
                    .map(|_| (vec![0.0; n * maxw], vec![0.0; n * maxw]))
                    .collect()
            };
            let to32 = |dst: &mut [f32], src: &[f64]| {
                dst.par_chunks_mut(1 << 14)
                    .zip(src.par_chunks(1 << 14))
                    .for_each(|(d, s)| d.iter_mut().zip(s).for_each(|(u, v)| *u = *v as f32))
            };
            let (mut t_int, mut t_gemm, mut t_acc) = (0.0, 0.0, 0.0);
            for AuxBlock {
                p0, nq, shells: sh, ..
            } in &blocks
            {
                let (p0, nq) = (*p0, *nq);
                let ti = Instant::now();
                self.mo.ints(sh.clone(), nq, 0.0, &mut ibuf);
                if !sym {
                    self.mo.dense_mmajor(&ibuf, nq, &mut dense);
                }
                if let Some(op) = &self.xop {
                    op_ints(&self.mo, op, sh.clone(), nq, &ibuf, &mut wbuf);
                    self.mo.dense_mmajor(&wbuf, nq, &mut densew);
                }
                if single {
                    if !sym {
                        to32(&mut dense32[..n * nq * n], &dense[..n * nq * n]);
                    }
                    if long_range {
                        to32(&mut densew32[..n * nq * n], &densew[..n * nq * n]);
                    }
                }
                t_int += ti.elapsed().as_secs_f64();
                for (s, ch) in self.chans.iter().enumerate() {
                    let o = ch.o;
                    let xs = p0 * o * o..(p0 + nq) * o * o;
                    for (g, grp) in groups[s].iter().enumerate() {
                        let ng = grp.len();
                        let tg = Instant::now();
                        // (integral block, fitted pairs, factor) terms
                        if single {
                            let im =
                                || MatRef::from_row_major_slice(&dense32[..n * nq * n], n, nq * n);
                            let imw =
                                || MatRef::from_row_major_slice(&densew32[..n * nq * n], n, nq * n);
                            let terms: Vec<(MatRef<'_, f32>, &[f32], f32)> = if sym {
                                vec![(imw(), &self.xw32[s][xs.clone()], 1.0)]
                            } else if long_range {
                                vec![
                                    (im(), &self.xw32[s][xs.clone()], 0.5),
                                    (imw(), &self.x32[s][xs.clone()], 0.5),
                                ]
                            } else {
                                vec![(im(), &self.x32[s][xs.clone()], 1.0)]
                            };
                            for (im, xb, f) in terms {
                                term1_slots(
                                    im,
                                    &ap32[s][g],
                                    ng,
                                    gen[s].len(),
                                    xb,
                                    n,
                                    nq,
                                    o,
                                    f,
                                    &mut slots32,
                                );
                            }
                            t_gemm += tg.elapsed().as_secs_f64();
                            let ta = Instant::now();
                            flush_slots(&mut kout[s], grp, n, o, &mut slots32, |x| x as f64);
                            t_acc += ta.elapsed().as_secs_f64();
                        } else {
                            let im =
                                || MatRef::from_row_major_slice(&dense[..n * nq * n], n, nq * n);
                            let imw =
                                || MatRef::from_row_major_slice(&densew[..n * nq * n], n, nq * n);
                            let terms: Vec<(MatRef<'_, f64>, &[f64], f64)> = if sym {
                                vec![(imw(), &ch.xw[xs.clone()], 1.0)]
                            } else if long_range {
                                vec![
                                    (im(), &ch.xw[xs.clone()], 0.5),
                                    (imw(), &ch.x[xs.clone()], 0.5),
                                ]
                            } else {
                                vec![(im(), &ch.x[xs.clone()], 1.0)]
                            };
                            for (im, xb, f) in terms {
                                term1_slots(
                                    im,
                                    &ap64[s][g],
                                    ng,
                                    gen[s].len(),
                                    xb,
                                    n,
                                    nq,
                                    o,
                                    f,
                                    &mut slots64,
                                );
                            }
                            t_gemm += tg.elapsed().as_secs_f64();
                            let ta = Instant::now();
                            flush_slots(&mut kout[s], grp, n, o, &mut slots64, |x| x);
                            t_acc += ta.elapsed().as_secs_f64();
                        }
                    }
                }
            }
            self.timers.add("resp_k1_ints", t_int);
            self.timers.add("resp_k1_gemm", t_gemm);
            self.timers.add("resp_k1_acc", t_acc);
            self.timers.resource("resp_k1_blocks", blocks.len() as f64);
            self.timers.resource(
                "resp_k1_groups",
                groups.iter().map(|g| g.len()).max().unwrap_or(0) as f64,
            );
        }
        self.timers.add("resp_k1", t0.elapsed().as_secs_f64());
        (jout, kout)
    }
}

/// Default auxiliary basis of the CPKS iteration J/K of large systems (Weigend's RI-MP2 fitting
/// basis: it fits the occupied-virtual products of the response well at 60% of cc-pVTZ-jkfit's
/// size; L-lactide frequencies with it in the whole response change by <= 0.09 cm^-1).
pub const ITER_AUX_DEFAULT: &str = "def2-svp-rifit|def2-universal-jkfit";

/// The CPKS iteration J/K of a large system ([`IterPlan`], from [`iteration_plan`]): when `main`
/// (the response J/K of the right-hand sides and of the final dF\[U\] C_o) runs in single
/// precision, RI J/K in single precision with the smaller auxiliary basis `iter_aux`
/// (`response.iteration_aux`; "auto": [`ITER_AUX_DEFAULT`]; "none": no separate iteration J/K;
/// "main": main's), with main's exchange operator; range-separated hybrids whenever main runs in
/// single precision: the symmetric RI of the operator (module docs; in main's auxiliary basis
/// below [`TWO_LEVEL_WORK`]). The Hessian error is quadratic in the difference of the two
/// operators (the stationary response term uses the main one's residual). The plan is decided
/// from the main J/K, so the build can wait until the main one has released its tensors (the
/// two never need to coexist).
pub struct IterPlan {
    aux: BasisSet,
    /// The auxiliary basis set's name ("main": that of the main J/K).
    pub aux_name: String,
    xop: Option<Operator>,
    sym: bool,
    /// AO overlap (recognizes densities inside the occupied space)
    s: Mat<f64>,
}

impl IterPlan {
    /// The iteration J/K (single precision only) for the occupied orbitals `occ` of the main one.
    pub fn build(
        &self,
        basis: &BasisSet,
        pairs: &PairSet,
        occ: &[MatRef<'_, f64>],
        par: Par,
    ) -> DfResponseJk {
        // the iteration basis set's own metric (main's is released after its setup)
        let metric =
            CoulombMetric::new(&self.aux, par).expect("auxiliary metric not positive definite");
        DfResponseJk::with_mode(
            basis,
            pairs,
            &self.aux,
            &metric,
            occ,
            self.s.as_ref(),
            self.xop,
            self.sym,
            true,
            par,
        )
    }
}

/// The [`IterPlan`] of a main response J/K whose iterations use the auxiliary basis `name`
/// (`iteration_aux`; None when it does not cover the atoms: one level); `s` the AO overlap.
pub fn iteration_plan(
    main: &DfResponseJk,
    aux_def: impl Fn(&str) -> Option<linus_ints::BasisDef>,
    mol: &linus_core::structure::Structure,
    name: &str,
    s: Mat<f64>,
) -> Option<IterPlan> {
    let sym = main.single && main.xop.is_some_and(|op| sym_op_usable(&op));
    let name = name.to_string();
    let aux = if name == "main" {
        main.aux.clone()
    } else {
        BasisSet::new(&aux_def(&name)?, &mol.z, &mol.positions, true).ok()?
    };
    Some(IterPlan {
        aux,
        aux_name: name,
        xop: main.xop,
        sym: sym && main.xop.is_some(),
        s,
    })
}

/// The operator a/r + b erf(omega r)/r has a positive definite auxiliary metric that is not
/// near-singular (a > 0, a + b > 0: its Fourier transform 4 pi/k^2 [a + b exp(-k^2/4 omega^2)]
/// is bounded below by a 4 pi/k^2): the symmetric operator RI of the iteration J/K applies.
pub fn sym_op_usable(op: &Operator) -> bool {
    op.coulomb() > 0.0 && op.coulomb() + op.long_range() > 0.0 && op.omega() > 0.0
}

impl linus_base::terms::ResponseJk for DfResponseJk {
    fn name(&self) -> String {
        format!("df-resp(naux={})", self.naux)
    }

    /// (c_j J C_s - c_k K C_s) from [`DfResponseJk::jk_proj`].
    fn fock_occ(
        &mut self,
        a: &[Vec<Mat<f64>>],
        c_j: f64,
        c_k: f64,
        par: Par,
    ) -> Result<Vec<Vec<Mat<f64>>>, linus_base::error::LinusError> {
        assert_eq!(
            a.len(),
            self.chans.len(),
            "DfResponseJk: one list of factors per channel"
        );
        let want_k = c_k != 0.0;
        let (j, k) = self.jk_proj(a, want_k, par);
        Ok(j.into_iter()
            .enumerate()
            .map(|(s, js)| {
                js.into_iter()
                    .enumerate()
                    .map(|(d, jd)| {
                        let mut f = &jd * c_j;
                        if want_k {
                            f -= &k[s][d] * c_k;
                        }
                        f
                    })
                    .collect()
            })
            .collect())
    }

    fn release(&mut self) {
        self.release_storage()
    }

    fn restore(&mut self, par: Par) {
        self.restore_storage(par)
    }

    fn timers(&self) -> Option<Timers> {
        Some(self.timers.clone())
    }
}
