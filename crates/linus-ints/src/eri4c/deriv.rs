//! Nuclear first and second derivatives of two-electron integrals (4-center (ab|cd),
//! 3-center (P|mn), 2-center (P|Q)) in the factorized McMurchie-Davidson form.
//!
//! With the pair data of [`crate::core::deriv`], every derivative is an E^T W E product where
//! W\[r\]\[s\] = (-1)^{|s|} R_{r+s}: derivatives of the bra (ket) functions use the derivative
//! E-matrices E^A (E^AA) of the bra (ket) pair, and ∂/∂P_k (the total translation of the bra
//! pair) raises the bra Hermite index. Derivatives with respect to the second center of a pair
//! follow from ∂B = ∂P - ∂A and those of the fourth center from translational invariance,
//! ∂A + ∂B + ∂C + ∂D = 0 (SHARK Scheme 4 / Helgaker-Taylor).
//!
//! Kernels:
//! - [`quartet_grad`]: Γ-contracted first derivatives (gradient contributions) without
//!   forming derivative integrals; Coulomb-type (separable) terms go through Hermite
//!   densities (J-engine style);
//! - [`quartet_hess`]: Γ-contracted second derivatives.
//!
//! The drivers of the four-center contractions (8-fold permutational symmetry, Schwarz x
//! density screening, ordered chunks) run under [`super::deriv_plan::Eri4DerivPlan`]; the
//! two-center ones under [`crate::eri2c::Eri2DerivPlan`] (2-center integrals are 4-center ones
//! with unit s functions (exponent 0) as second shells of the auxiliary pairs, whose
//! derivatives vanish).
//!
//! The raw derivative blocks of a quartet and the plain three-center gradient and Hessian
//! contractions the tests check these against are in `crate::reference`.

use crate::basis::BasisSet;
use crate::contract::{DensityTerm, DensityTerms};
use crate::core::boys;
use crate::core::deriv::{
    d2_idx, deriv2_pair, deriv_pair_a, deriv_pair_b, r_tensor_batch, r_tensor_fast, shift1, wtable,
    D2Kind, PairDeriv, WTable,
};
use crate::core::hermite::{nherm, ShellPair};
use crate::core::kernels::{dot3_long, dot3_shift, wmul_acc, wmul_t_acc};
use crate::core::util::{add_grads_into, shell_pair_dmax};
use crate::eri4c::{EriWork, ERI_PREF};
use crate::operator::{boys_operator, Operator};
use crate::pairs::{diag_bound, DerivOrder, PairSet};
use rayon::prelude::*;

/// Center labels of a quartet (ab|cd).
pub const CENTER_A: usize = 0;
pub const CENTER_B: usize = 1;
pub const CENTER_C: usize = 2;
pub const CENTER_D: usize = 3;

/// Scratch space for the derivative kernels (one per thread).
#[derive(Default)]
pub struct DerivWork {
    pub(crate) fm: Vec<f64>,
    pub(crate) r: Vec<f64>,
    pub(crate) rwork: Vec<f64>,
    pub(crate) w: Vec<f64>,
    pub(crate) t: Vec<f64>,
    h: Vec<f64>,
    ha: Vec<f64>,
    z: Vec<f64>,
    za: Vec<f64>,
    y: Vec<f64>,
    pub(crate) tmp: Vec<f64>,
    v1: Vec<f64>,
    v2: Vec<f64>,
    // batched R over ket primitives
    rb: Vec<f64>,
    rbw: Vec<f64>,
    fmb: Vec<f64>,
    qa: Vec<f64>,
    qpref: Vec<f64>,
    qx: Vec<[f64; 3]>,
    /// Two-electron operator of the derivative integrals (Coulomb by default; erf-attenuated
    /// and CAM combinations change only the Boys stage, see [`boys_operator`]).
    pub op: Operator,
}

/// R tensors of bra primitive `ip` with every ket primitive (structure of arrays in `rb`).
#[allow(clippy::too_many_arguments)]
fn batch_r(
    l: usize,
    bra: &ShellPair,
    ip: usize,
    ket: &ShellPair,
    op: &Operator,
    rb: &mut Vec<f64>,
    rbw: &mut Vec<f64>,
    fmb: &mut Vec<f64>,
    qa: &mut Vec<f64>,
    qpref: &mut Vec<f64>,
    qx: &mut Vec<[f64; 3]>,
) {
    let nq = ket.nprim;
    let nh = nherm(l);
    grow(rb, nh * nq);
    grow(rbw, (2 * nh + l + 4) * nq);
    grow(fmb, (l + 1) * nq);
    grow(qa, nq);
    grow(qpref, nq);
    if qx.len() < nq {
        qx.resize(nq, [0.0; 3]);
    }
    let (p, pc) = (bra.p[ip], bra.pc[ip]);
    for kq in 0..nq {
        let q = ket.p[kq];
        let qc = ket.pc[kq];
        let pq = [pc[0] - qc[0], pc[1] - qc[1], pc[2] - qc[2]];
        let alpha = p * q / (p + q);
        boys_operator(
            l,
            alpha,
            pq[0] * pq[0] + pq[1] * pq[1] + pq[2] * pq[2],
            op,
            &mut fmb[kq * (l + 1)..(kq + 1) * (l + 1)],
        );
        qa[kq] = alpha;
        qpref[kq] = ERI_PREF / (p * q * (p + q).sqrt());
        qx[kq] = pq;
    }
    r_tensor_batch(l, nq, qa, qx, fmb, qpref, rb, rbw);
}

impl DerivWork {
    pub fn new() -> Self {
        DerivWork {
            fm: vec![0.0; boys::MAX_M + 1],
            ..Default::default()
        }
    }

    /// Scratch space for derivative integrals of the operator `op`.
    pub fn with_op(op: Operator) -> Self {
        DerivWork { op, ..Self::new() }
    }
}

/// One chunk's accumulator of the 4c gradient's deterministic reduction: the integral
/// scratch, the partial gradient (natom rows) and the Γ block of a quartet.
pub(crate) type GradAcc = (DerivWork, Vec<[f64; 3]>, Vec<f64>);
/// One chunk's accumulator of the 4c Hessian's: the scratch, the partial (3 natom)^2 Hessian
/// and the Γ block.
pub(crate) type HessAcc = (DerivWork, Vec<f64>, Vec<f64>);

/// The lengths [`DerivWork`]'s buffers reach in [`quartet_grad`] (order 1, with `nj` Coulomb
/// terms) or [`quartet_hess`] (order 2) for pairs of angular momentum sum at most `lab`, at
/// most `nab` functions and at most `kq` primitives: the kernels' `grow` sizes at those
/// maxima (each size grows with each argument, so these bound every quartet's).
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct ScratchLens {
    w: usize,
    t: usize,
    h: usize,
    ha: usize,
    z: usize,
    za: usize,
    y: usize,
    tmp: usize,
    v1: usize,
    v2: usize,
    r: usize,
    rwork: usize,
    rb: usize,
    rbw: usize,
    fmb: usize,
    kq: usize,
}

impl ScratchLens {
    pub(crate) fn new(order: usize, lab: usize, nab: usize, kq: usize, nj: usize) -> Self {
        let l = 2 * lab + order;
        let (h0, h1, h2) = (nherm(lab), nherm(lab + 1), nherm(lab + 2));
        if order == 1 {
            ScratchLens {
                w: h1 * h1,
                t: h1 * nab,
                h: h0 * nab,
                z: kq * h1 * nab,
                y: h1 * nab,
                tmp: nab * nab,
                v1: nj * h1,
                v2: nj * kq * h1,
                rb: nherm(l) * kq,
                rbw: (2 * nherm(l) + l + 4) * kq,
                fmb: (l + 1) * kq,
                kq,
                ..Default::default()
            }
        } else {
            ScratchLens {
                w: h2 * h2,
                t: h2 * nab,
                h: h0 * nab,
                ha: 3 * h1 * nab,
                z: kq * h2 * nab,
                za: kq * 3 * h1 * nab,
                y: h2 * nab,
                r: nherm(l) + 1,
                rwork: 2 * nherm(l),
                ..Default::default()
            }
        }
    }

    /// The bytes of a [`DerivWork`] reserved to these lengths.
    pub(crate) fn bytes(&self) -> usize {
        let f = self.w
            + self.t
            + self.h
            + self.ha
            + self.z
            + self.za
            + self.y
            + self.tmp
            + self.v1
            + self.v2
            + self.r
            + self.rwork
            + self.rb
            + self.rbw
            + self.fmb
            + 2 * self.kq
            + boys::MAX_M
            + 1;
        8 * f + 24 * self.kq
    }
}

impl DerivWork {
    /// Grow the buffers to `s` up front, so the kernels' own `grow` calls never allocate.
    pub(crate) fn reserve(&mut self, s: &ScratchLens) {
        grow(&mut self.w, s.w);
        grow(&mut self.t, s.t);
        grow(&mut self.h, s.h);
        grow(&mut self.ha, s.ha);
        grow(&mut self.z, s.z);
        grow(&mut self.za, s.za);
        grow(&mut self.y, s.y);
        grow(&mut self.tmp, s.tmp);
        grow(&mut self.v1, s.v1);
        grow(&mut self.v2, s.v2);
        grow(&mut self.r, s.r);
        grow(&mut self.rwork, s.rwork);
        grow(&mut self.rb, s.rb);
        grow(&mut self.rbw, s.rbw);
        grow(&mut self.fmb, s.fmb);
        grow(&mut self.qa, s.kq);
        grow(&mut self.qpref, s.kq);
        if self.qx.len() < s.kq {
            self.qx.resize(s.kq, [0.0; 3]);
        }
    }

    /// The buffers' lengths (tests: a reserved scratch does not grow in a contraction).
    #[cfg(test)]
    pub(crate) fn lens(&self) -> [usize; 19] {
        [
            self.fm.len(),
            self.r.len(),
            self.rwork.len(),
            self.w.len(),
            self.t.len(),
            self.h.len(),
            self.ha.len(),
            self.z.len(),
            self.za.len(),
            self.y.len(),
            self.tmp.len(),
            self.v1.len(),
            self.v2.len(),
            self.rb.len(),
            self.rbw.len(),
            self.fmb.len(),
            self.qa.len(),
            self.qpref.len(),
            self.qx.len(),
        ]
    }
}

#[inline]
pub(crate) fn grow(v: &mut Vec<f64>, n: usize) {
    if v.len() < n {
        v.resize(n, 0.0);
    }
}

/// W[r][s] = (-1)^{|s|} R_{r+s}(p, q, PQ) on the valid entries of `wt` (zero elsewhere), with
/// R up to degree `l`.
#[inline]
#[allow(clippy::too_many_arguments)]
pub(crate) fn fill_w(
    l: usize,
    p: f64,
    pc: [f64; 3],
    q: f64,
    qc: [f64; 3],
    wt: &WTable,
    op: &Operator,
    fm: &mut [f64],
    r: &mut Vec<f64>,
    rwork: &mut Vec<f64>,
    w: &mut [f64],
) {
    debug_assert_eq!(wt.lmax, l);
    let nh = nherm(l);
    grow(r, nh + 1);
    grow(rwork, 2 * nh);
    let pq = [pc[0] - qc[0], pc[1] - qc[1], pc[2] - qc[2]];
    let alpha = p * q / (p + q);
    boys_operator(
        l,
        alpha,
        pq[0] * pq[0] + pq[1] * pq[1] + pq[2] * pq[2],
        op,
        fm,
    );
    let pref = ERI_PREF / (p * q * (p + q).sqrt());
    r_tensor_fast(l, alpha, pq, fm, pref, r, rwork);
    r[nh] = 0.0;
    wt.gather(&r[..nh + 1], w);
}

#[inline]
// the unrolled loop as compiled (as_chunks would change its code)
#[allow(clippy::chunks_exact_to_as_chunks)]
fn dot(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    let (a, b) = (&a[..n], &b[..n]);
    let mut s = [0.0f64; 4];
    let (ca, cb) = (a.chunks_exact(4), b.chunks_exact(4));
    let (ra, rb) = (ca.remainder(), cb.remainder());
    for (x, y) in ca.zip(cb) {
        s[0] += x[0] * y[0];
        s[1] += x[1] * y[1];
        s[2] += x[2] * y[2];
        s[3] += x[3] * y[3];
    }
    let mut t = (s[0] + s[2]) + (s[1] + s[3]);
    for (x, y) in ra.iter().zip(rb) {
        t += x * y;
    }
    t
}

/// Relative cost of the Γ-contracted gradient kernel with `bra` as bra (the T and Z passes
/// scale with the ket dimension).
#[inline]
fn grad_cost(bra: &ShellPair, ket: &ShellPair) -> usize {
    let (lab, lcd) = (bra.lab(), ket.lab());
    (nherm(lab + 1) * nherm(lcd) + nherm(lab) * nherm(lcd + 1)) * ket.nab() * bra.nprim * ket.nprim
        + nherm(lab + 1) * bra.nab() * ket.nab() * bra.nprim
}

/// Center pairs of the second-derivative blocks, in output order.
pub const D2_CENTER_PAIRS: [(usize, usize); 10] = [
    (0, 0),
    (0, 1),
    (0, 2),
    (0, 3),
    (1, 1),
    (1, 2),
    (1, 3),
    (2, 2),
    (2, 3),
    (3, 3),
];

// ---------------------------------------------------------------------------------------
// Γ-contracted first derivatives
// ---------------------------------------------------------------------------------------

/// Hermite densities of a pair for a separable (Coulomb-type) term: per primitive,
/// e[ip * nherm(lab) + r] = sum_ab E\[r\]\[ab\] D_ab and ea[(ip * 3 + k) * nherm(lab+1) + r] =
/// sum_ab EA\[r\]\[k\]\[ab\] D_ab (pair-permutation factors included by the caller).
pub struct HermDens<'a> {
    pub e: &'a [f64],
    pub ea: &'a [f64],
}

/// One separable term coef * D1_ab D2_cd of the quartet density.
pub struct JTerm<'a> {
    pub coef: f64,
    pub bra: HermDens<'a>,
    pub ket: HermDens<'a>,
}

/// Γ-contracted first derivatives of one quartet:
/// g\[X\]\[k\] = sum_{ab,cd} ∂(ab|cd)/∂X_k (Γ\[ab\]\[cd\] + sum_t coef_t D1t_ab D2t_cd), X = A..D.
/// `gamma` is nab x ncd (row-major) or None.
pub fn quartet_grad(
    bra: &ShellPair,
    bd: &PairDeriv,
    ket: &ShellPair,
    kd: &PairDeriv,
    gamma: Option<&[f64]>,
    jterms: &[JTerm],
    w: &mut DerivWork,
) -> [[f64; 3]; 4] {
    let mut g = [[0.0; 3]; 4];
    if bra.nprim == 0 || ket.nprim == 0 || (gamma.is_none() && jterms.is_empty()) {
        return g;
    }
    let (lab, lcd) = (bra.lab(), ket.lab());
    let (nab, ncd) = (bra.nab(), ket.nab());
    let l = lab + lcd + 1;
    let (nhb, nhb1, nhk, nhk1) = (nherm(lab), nherm(lab + 1), nherm(lcd), nherm(lcd + 1));
    let (kp, kq_n) = (bra.nprim, ket.nprim);
    let nj = jterms.len();
    let wt = wtable(lab + 1, lcd + 1, 1);
    let sh = shift1();
    let DerivWork {
        w: wm,
        t,
        h,
        z,
        y,
        v1,
        v2,
        tmp: gt,
        rb,
        rbw,
        fmb,
        qa,
        qpref,
        qx,
        op,
        ..
    } = w;
    grow(wm, nhb1 * nhk1);
    let (mut ga, mut gp, mut gc) = ([0.0; 3], [0.0; 3], [0.0; 3]);
    if let Some(gm) = gamma {
        grow(t, nhb1 * ncd);
        grow(h, nhb * ncd);
        grow(z, kq_n * nhk1 * ncd);
        grow(y, nhb1 * nab);
        grow(gt, nab * ncd);
        z[..kq_n * nhk1 * ncd].fill(0.0);
        // Γ^T (ncd x nab) for the Y = T Γ^T pass
        for ab in 0..nab {
            for cd in 0..ncd {
                gt[cd * nab + ab] = gm[ab * ncd + cd];
            }
        }
    }
    // J (separable) terms: v1[t][r'] = sum_q W_pq[r'][s] hk_q[s] per bra primitive,
    // v2[t][q][s'] = sum_p hb_p[r] W_pq[r][s'] over all bra primitives.
    grow(v1, nj * nhb1);
    grow(v2, nj * kq_n * nhk1);
    v2[..nj * kq_n * nhk1].fill(0.0);
    for ip in 0..kp {
        let ep = bra.e_prim(ip);
        if let Some(gm) = gamma {
            let h = &mut h[..nhb * ncd];
            h.fill(0.0);
            wmul_acc(ep, nab, nhb, nab, gm, ncd, h);
            t[..nhb1 * ncd].fill(0.0);
        }
        v1[..nj * nhb1].fill(0.0);
        batch_r(l, bra, ip, ket, op, rb, rbw, fmb, qa, qpref, qx);
        for kq in 0..kq_n {
            wt.gather_soa(rb, kq_n, kq, wm, nhb, nhk);
            if gamma.is_some() {
                wmul_acc(wm, nhk1, nhb1, nhk, ket.e_prim(kq), ncd, t);
                wmul_t_acc(
                    wm,
                    nhk1,
                    nhb,
                    nhk1,
                    h,
                    ncd,
                    &mut z[kq * nhk1 * ncd..(kq + 1) * nhk1 * ncd],
                );
            }
            for (it, jt) in jterms.iter().enumerate() {
                let hb = &jt.bra.e[ip * nhb..(ip + 1) * nhb];
                let hk = &jt.ket.e[kq * nhk..(kq + 1) * nhk];
                wmul_acc(
                    wm,
                    nhk1,
                    nhb1,
                    nhk,
                    hk,
                    1,
                    &mut v1[it * nhb1..(it + 1) * nhb1],
                );
                let o = (it * kq_n + kq) * nhk1;
                wmul_t_acc(wm, nhk1, nhb, nhk1, hb, 1, &mut v2[o..o + nhk1]);
            }
        }
        for (it, jt) in jterms.iter().enumerate() {
            let hb = &jt.bra.e[ip * nhb..(ip + 1) * nhb];
            let hba = &jt.bra.ea[ip * 3 * nhb1..(ip + 1) * 3 * nhb1];
            let v = &v1[it * nhb1..(it + 1) * nhb1];
            for k in 0..3 {
                ga[k] += jt.coef * dot(&hba[k * nhb1..(k + 1) * nhb1], v);
                let mut s = 0.0;
                for rr in 0..nhb {
                    s += hb[rr] * v[sh[rr][k] as usize];
                }
                gp[k] += jt.coef * s;
            }
        }
        if gamma.is_some() {
            // Y[r'][ab] = sum_cd T[r'][cd] Γ[ab][cd]
            let y = &mut y[..nhb1 * nab];
            y.fill(0.0);
            wmul_acc(t, ncd, nhb1, ncd, gt, nab, y);
            let a3 = dot3_long(bd.ea_t_prim(ip), y, nhb1 * nab);
            let p3 = dot3_shift(ep, y, sh, nhb, nab);
            for k in 0..3 {
                ga[k] += a3[k];
                gp[k] += p3[k];
            }
        }
    }
    for (it, jt) in jterms.iter().enumerate() {
        for kq in 0..kq_n {
            let hka = &jt.ket.ea[kq * 3 * nhk1..(kq + 1) * 3 * nhk1];
            let o = (it * kq_n + kq) * nhk1;
            for k in 0..3 {
                gc[k] += jt.coef * dot(&hka[k * nhk1..(k + 1) * nhk1], &v2[o..o + nhk1]);
            }
        }
    }
    if gamma.is_some() {
        for kq in 0..kq_n {
            let zq = &z[kq * nhk1 * ncd..(kq + 1) * nhk1 * ncd];
            let c3 = dot3_long(kd.ea_t_prim(kq), zq, nhk1 * ncd);
            for k in 0..3 {
                gc[k] += c3[k];
            }
        }
    }
    for k in 0..3 {
        g[CENTER_A][k] = ga[k];
        g[CENTER_B][k] = gp[k] - ga[k];
        g[CENTER_C][k] = gc[k];
        g[CENTER_D][k] = -(gp[k] + gc[k]);
    }
    g
}

// ---------------------------------------------------------------------------------------
// Γ-contracted second derivatives
// ---------------------------------------------------------------------------------------

/// Second-derivative pieces of a quartet: 3x3 blocks ∂X_k ∂Y_l for X, Y in
/// {A, P = A + B, C}: AA, AP, PP, AC, PC, CC.
#[derive(Clone, Copy, Default, Debug)]
pub struct HessPieces {
    pub aa: [[f64; 3]; 3],
    pub ap: [[f64; 3]; 3],
    pub pp: [[f64; 3]; 3],
    pub ac: [[f64; 3]; 3],
    pub pc: [[f64; 3]; 3],
    pub cc: [[f64; 3]; 3],
}

impl HessPieces {
    /// Blocks for the 10 center pairs of [`D2_CENTER_PAIRS`], \[XY\]\[k\]\[l\] = ∂X_k ∂Y_l.
    pub fn center_blocks(&self) -> [[[f64; 3]; 3]; 10] {
        let mut o = [[[0.0; 3]; 3]; 10];
        let (aa, ap, pp, ac, pc, cc) = (&self.aa, &self.ap, &self.pp, &self.ac, &self.pc, &self.cc);
        for k in 0..3 {
            for l in 0..3 {
                let ab = ap[k][l] - aa[k][l];
                let bb = pp[k][l] - ap[l][k] - ap[k][l] + aa[k][l];
                let bc = pc[k][l] - ac[k][l];
                let ad = -(ap[k][l] + ac[k][l]);
                let bp = pp[k][l] - ap[k][l];
                let bd = -(bp + bc);
                let cd = -(pc[l][k] + cc[k][l]);
                let dd = pp[k][l] + pc[k][l] + pc[l][k] + cc[k][l];
                let vals = [aa[k][l], ab, ac[k][l], ad, bb, bc, bd, cc[k][l], cd, dd];
                for (i, v) in vals.iter().enumerate() {
                    o[i][k][l] = *v;
                }
            }
        }
        o
    }
}

/// Γ-contracted second derivatives of one quartet (needs order-2 pair data):
/// sum_{ab,cd} ∂²(ab|cd)/∂X_k∂Y_l Γ\[ab\]\[cd\], returned as [`HessPieces`].
#[allow(clippy::needless_range_loop, reason = "indexes several arrays in step")]
pub fn quartet_hess(
    bra: &ShellPair,
    bd: &PairDeriv,
    ket: &ShellPair,
    kd: &PairDeriv,
    gamma: &[f64],
    w: &mut DerivWork,
) -> HessPieces {
    let mut hp = HessPieces::default();
    if bra.nprim == 0 || ket.nprim == 0 {
        return hp;
    }
    let (lab, lcd) = (bra.lab(), ket.lab());
    let (nab, ncd) = (bra.nab(), ket.nab());
    let l = lab + lcd + 2;
    let (nhb, nhb1, nhb2) = (nherm(lab), nherm(lab + 1), nherm(lab + 2));
    let (nhk, nhk1, nhk2) = (nherm(lcd), nherm(lcd + 1), nherm(lcd + 2));
    let wt = wtable(lab + 2, lcd + 2, 2);
    let sh = shift1();
    let DerivWork {
        fm,
        r,
        rwork,
        w: wm,
        t,
        h,
        ha,
        z,
        za,
        y,
        op,
        ..
    } = w;
    grow(wm, nhb2 * nhk2);
    grow(t, nhb2 * ncd);
    grow(h, nhb * ncd);
    grow(ha, 3 * nhb1 * ncd);
    grow(z, ket.nprim * nhk2 * ncd);
    grow(za, ket.nprim * 3 * nhk1 * ncd);
    grow(y, nhb2 * nab);
    let z = &mut z[..ket.nprim * nhk2 * ncd];
    let za = &mut za[..ket.nprim * 3 * nhk1 * ncd];
    z.fill(0.0);
    za.fill(0.0);
    for ip in 0..bra.nprim {
        let ep = bra.e_prim(ip);
        let eap = bd.ea_prim(ip);
        let eaap = bd.eaa_prim(ip);
        // H = E^T Γ, HA_k = EA_k^T Γ
        let h = &mut h[..nhb * ncd];
        h.fill(0.0);
        let ha = &mut ha[..3 * nhb1 * ncd];
        ha.fill(0.0);
        for rr in 0..nhb1 {
            for c in 0..4 {
                if c == 0 && rr >= nhb {
                    continue;
                }
                let (erow, hrow) = if c == 0 {
                    (
                        &ep[rr * nab..(rr + 1) * nab],
                        &mut h[rr * ncd..(rr + 1) * ncd],
                    )
                } else {
                    let k = c - 1;
                    (
                        &eap[(rr * 3 + k) * nab..(rr * 3 + k + 1) * nab],
                        &mut ha[(k * nhb1 + rr) * ncd..(k * nhb1 + rr + 1) * ncd],
                    )
                };
                for (ab, &ev) in erow.iter().enumerate() {
                    if ev == 0.0 {
                        continue;
                    }
                    for (x, &gv) in hrow.iter_mut().zip(&gamma[ab * ncd..(ab + 1) * ncd]) {
                        *x += ev * gv;
                    }
                }
            }
        }
        let t = &mut t[..nhb2 * ncd];
        t.fill(0.0);
        for kq in 0..ket.nprim {
            fill_w(
                l, bra.p[ip], bra.pc[ip], ket.p[kq], ket.pc[kq], wt, op, fm, r, rwork, wm,
            );
            wmul_acc(wm, nhk2, nhb2, nhk, ket.e_prim(kq), ncd, t);
            wmul_t_acc(
                wm,
                nhk2,
                nhb,
                nhk2,
                h,
                ncd,
                &mut z[kq * nhk2 * ncd..(kq + 1) * nhk2 * ncd],
            );
            for k in 0..3 {
                let o = (kq * 3 + k) * nhk1 * ncd;
                wmul_t_acc(
                    wm,
                    nhk2,
                    nhb1,
                    nhk1,
                    &ha[k * nhb1 * ncd..(k + 1) * nhb1 * ncd],
                    ncd,
                    &mut za[o..o + nhk1 * ncd],
                );
            }
        }
        // Y'' = T'' Γ^T
        for rr in 0..nhb2 {
            let trow = &t[rr * ncd..(rr + 1) * ncd];
            for ab in 0..nab {
                y[rr * nab + ab] = dot(trow, &gamma[ab * ncd..(ab + 1) * ncd]);
            }
        }
        let yr = |rr: usize| &y[rr * nab..(rr + 1) * nab];
        for rr in 0..nhb2 {
            for kl in 0..6 {
                let v = dot(&eaap[(rr * 6 + kl) * nab..(rr * 6 + kl + 1) * nab], yr(rr));
                let (k, l2) = crate::core::deriv::D2_COMPS[kl];
                hp.aa[k][l2] += v;
                if k != l2 {
                    hp.aa[l2][k] += v;
                }
            }
        }
        for rr in 0..nhb1 {
            for k in 0..3 {
                let erow = &eap[(rr * 3 + k) * nab..(rr * 3 + k + 1) * nab];
                for l2 in 0..3 {
                    hp.ap[k][l2] += dot(erow, yr(sh[rr][l2] as usize));
                }
            }
        }
        for rr in 0..nhb {
            let erow = &ep[rr * nab..(rr + 1) * nab];
            for k in 0..3 {
                let r1 = sh[rr][k] as usize;
                for l2 in k..3 {
                    let v = dot(erow, yr(sh[r1][l2] as usize));
                    hp.pp[k][l2] += v;
                    if l2 != k {
                        hp.pp[l2][k] += v;
                    }
                }
            }
        }
    }
    for kq in 0..ket.nprim {
        let zq = &z[kq * nhk2 * ncd..(kq + 1) * nhk2 * ncd];
        let eaq = kd.ea_prim(kq);
        let eaaq = kd.eaa_prim(kq);
        let zr = |s: usize| &zq[s * ncd..(s + 1) * ncd];
        for s in 0..nhk2 {
            for kl in 0..6 {
                let v = dot(&eaaq[(s * 6 + kl) * ncd..(s * 6 + kl + 1) * ncd], zr(s));
                let (k, l2) = crate::core::deriv::D2_COMPS[kl];
                hp.cc[k][l2] += v;
                if k != l2 {
                    hp.cc[l2][k] += v;
                }
            }
        }
        for s in 0..nhk1 {
            for l2 in 0..3 {
                let erow = &eaq[(s * 3 + l2) * ncd..(s * 3 + l2 + 1) * ncd];
                for k in 0..3 {
                    hp.pc[k][l2] -= dot(erow, zr(sh[s][k] as usize));
                    let o = ((kq * 3 + k) * nhk1 + s) * ncd;
                    hp.ac[k][l2] += dot(erow, &za[o..o + ncd]);
                }
            }
        }
    }
    let _ = d2_idx(0, 0);
    hp
}

// ---------------------------------------------------------------------------------------
// Drivers
// ---------------------------------------------------------------------------------------

/// Schwarz-type factors for derivative integrals of each pair: (Q1, Q2) with
/// Q1 = sqrt(max |(∂X_k ab|∂X_k ab)|) over X in {A, B}, k, ab and, if the pair data is of
/// order 2, Q2 = sqrt(max |(∂X_k∂Y_l ab|∂X_k∂Y_l ab)|) (else 0). Then
/// |∂(ab|cd)| <= Q1_ab Q_cd + Q_ab Q1_cd and |∂²(ab|cd)| <= Q2_ab Q_cd + 2 Q1_ab Q1_cd + Q_ab Q2_cd
/// (Cauchy-Schwarz on each term of the product rule).
pub(crate) fn schwarz_deriv(pairs: &[ShellPair], dpairs: &[PairDeriv]) -> (Vec<f64>, Vec<f64>) {
    pairs
        .par_iter()
        .zip(dpairs)
        .map_init(
            || (EriWork::new(), Vec::new()),
            |(w, buf), (pr, pd)| {
                if pr.nprim == 0 {
                    return (0.0, 0.0);
                }
                let q1 = diag_bound(&deriv_pair_a(pr, pd), w, buf).max(diag_bound(
                    &deriv_pair_b(pr, pd),
                    w,
                    buf,
                ));
                let q2 = if pd.order >= 2 {
                    [D2Kind::AA, D2Kind::AB, D2Kind::BB]
                        .iter()
                        .map(|&k| diag_bound(&deriv2_pair(pr, pd, k), w, buf))
                        .fold(0.0, f64::max)
                } else {
                    0.0
                };
                (q1, q2)
            },
        )
        .unzip()
}

/// Hermite densities (with pair factor 2 for distinct shells) of every pair for one density.
pub(crate) struct PairHerm {
    pub(crate) e: Vec<Vec<f64>>,
    pub(crate) ea: Vec<Vec<f64>>,
}

pub(crate) fn pair_herm(basis: &BasisSet, pairs: &PairSet, d: &[f64]) -> PairHerm {
    let nbf = basis.nbf;
    let sh = &basis.shells;
    let (e, ea): (Vec<Vec<f64>>, Vec<Vec<f64>>) = pairs
        .pairs()
        .par_iter()
        .zip(pairs.derivs())
        .map(|(pr, pd)| {
            let lab = pr.lab();
            let (nh0, nh1) = (nherm(lab), nherm(lab + 1));
            let f = if pr.sa == pr.sb { 1.0 } else { 2.0 };
            let (a0, b0) = (sh[pr.sa].ao_start, sh[pr.sb].ao_start);
            let mut e = vec![0.0; pr.nprim * nh0];
            let mut ea = vec![0.0; pr.nprim * 3 * nh1];
            for ip in 0..pr.nprim {
                crate::one_e::deriv::herm_dens(
                    pr.e_prim(ip),
                    nh0,
                    1,
                    d,
                    nbf,
                    a0,
                    b0,
                    pr.na,
                    pr.nb,
                    f,
                    &mut e[ip * nh0..(ip + 1) * nh0],
                );
                crate::one_e::deriv::herm_dens(
                    pd.ea_prim(ip),
                    nh1,
                    3,
                    d,
                    nbf,
                    a0,
                    b0,
                    pr.na,
                    pr.nb,
                    f,
                    &mut ea[ip * 3 * nh1..(ip + 1) * 3 * nh1],
                );
            }
            (e, ea)
        })
        .unzip();
    PairHerm { e, ea }
}

/// Symmetrized exchange-type quartet density times `deg`:
/// Γ[ab][cd] = deg sum_u x_u/4 (X1_ac X2_bd + X1_bc X2_ad + X1_ad X2_bc + X1_bd X2_ac).
#[allow(clippy::too_many_arguments)]
fn exchange_block(
    exch: &[DensityTerm],
    nbf: usize,
    o: [usize; 4],
    n: [usize; 4],
    deg: f64,
    out: &mut [f64],
) {
    let (na, nb, nc, nd) = (n[0], n[1], n[2], n[3]);
    let ncd = nc * nd;
    out[..na * nb * ncd].fill(0.0);
    for &DensityTerm {
        coef: x,
        left: d1,
        right: d2,
    } in exch
    {
        let f = 0.25 * x * deg;
        for a in 0..na {
            let ia = o[0] + a;
            for b in 0..nb {
                let ib = o[1] + b;
                let row = &mut out[(a * nb + b) * ncd..(a * nb + b + 1) * ncd];
                for c in 0..nc {
                    let ic = o[2] + c;
                    let (d1ac, d1bc, d2ac, d2bc) = (
                        d1[ia * nbf + ic],
                        d1[ib * nbf + ic],
                        d2[ia * nbf + ic],
                        d2[ib * nbf + ic],
                    );
                    for dd in 0..nd {
                        let id = o[3] + dd;
                        row[c * nd + dd] += f
                            * (d1ac * d2[ib * nbf + id]
                                + d1bc * d2[ia * nbf + id]
                                + d1[ia * nbf + id] * d2bc
                                + d1[ib * nbf + id] * d2ac);
                    }
                }
            }
        }
    }
}

/// Coulomb-type quartet density block (for the Hessian path, which has no J engine):
/// Γ[ab][cd] += deg sum_t c_t/2 (D1_ab D2_cd + D2_ab D1_cd).
fn coulomb_block(
    coul: &[DensityTerm],
    nbf: usize,
    o: [usize; 4],
    n: [usize; 4],
    deg: f64,
    out: &mut [f64],
) {
    let (na, nb, nc, nd) = (n[0], n[1], n[2], n[3]);
    let ncd = nc * nd;
    for &DensityTerm {
        coef: cf,
        left: d1,
        right: d2,
    } in coul
    {
        let f = 0.5 * cf * deg;
        for a in 0..na {
            for b in 0..nb {
                let (x1, x2) = (
                    d1[(o[0] + a) * nbf + o[1] + b],
                    d2[(o[0] + a) * nbf + o[1] + b],
                );
                let row = &mut out[(a * nb + b) * ncd..(a * nb + b + 1) * ncd];
                for c in 0..nc {
                    for dd in 0..nd {
                        let (y1, y2) = (
                            d1[(o[2] + c) * nbf + o[3] + dd],
                            d2[(o[2] + c) * nbf + o[3] + dd],
                        );
                        row[c * nd + dd] += f * (x1 * y2 + x2 * y1);
                    }
                }
            }
        }
    }
}

/// Shared setup for the 4-center contractions.
struct Dens4c<'a> {
    /// Coulomb density list (deduplicated by address) and per-term indices
    jd: Vec<&'a [f64]>,
    jterm_idx: Vec<(f64, usize, usize)>,
    jdmax: Vec<Vec<f64>>,
    kdmax: Vec<(f64, Vec<f64>, Vec<f64>)>,
}

impl<'a> Dens4c<'a> {
    fn new(basis: &BasisSet, terms: &'a DensityTerms<'a>) -> Self {
        let mut jd: Vec<&[f64]> = Vec::new();
        let mut find = |d: &'a [f64]| -> usize {
            if let Some(i) = jd
                .iter()
                .position(|x| std::ptr::eq(x.as_ptr(), d.as_ptr()) && x.len() == d.len())
            {
                i
            } else {
                jd.push(d);
                jd.len() - 1
            }
        };
        let mut jterm_idx = Vec::new();
        for &DensityTerm {
            coef: c,
            left: d1,
            right: d2,
        } in &terms.coulomb
        {
            let (i1, i2) = (find(d1), find(d2));
            if i1 == i2 {
                jterm_idx.push((c, i1, i1));
            } else {
                jterm_idx.push((0.5 * c, i1, i2));
                jterm_idx.push((0.5 * c, i2, i1));
            }
        }
        let jdmax = jd.iter().map(|d| shell_pair_dmax(basis, d)).collect();
        let kdmax = terms
            .exchange
            .iter()
            .map(|t| {
                (
                    t.coef,
                    shell_pair_dmax(basis, t.left),
                    shell_pair_dmax(basis, t.right),
                )
            })
            .collect();
        Dens4c {
            jd,
            jterm_idx,
            jdmax,
            kdmax,
        }
    }

    /// (Coulomb, exchange) density estimates of a quartet of shells.
    fn estimates(&self, ns: usize, s: [usize; 4]) -> (f64, f64) {
        let mut ej: f64 = 0.0;
        for &(c, i1, i2) in &self.jterm_idx {
            ej = ej
                .max(c.abs() * self.jdmax[i1][s[0] * ns + s[1]] * self.jdmax[i2][s[2] * ns + s[3]]);
        }
        let mut ek: f64 = 0.0;
        for (x, m1, m2) in &self.kdmax {
            let v = m1[s[0] * ns + s[2]] * m2[s[1] * ns + s[3]]
                + m1[s[1] * ns + s[2]] * m2[s[0] * ns + s[3]]
                + m1[s[0] * ns + s[3]] * m2[s[1] * ns + s[2]]
                + m1[s[1] * ns + s[3]] * m2[s[0] * ns + s[2]];
            ek = ek.max(0.25 * x.abs() * v);
        }
        (ej, ek)
    }
}

/// Two-electron gradient sum_{abcd} ∂(ab|cd)/∂X Γ_abcd for the density terms `terms`,
/// natom x 3, into `accs[0].1` (zeroed; owned by an
/// [`Eri4DerivPlan`](super::deriv_plan::Eri4DerivPlan)). Uses 8-fold permutational symmetry,
/// Schwarz x density screening at `thresh` (e.g. 1e-12), and rayon over bra pairs. `pairs`:
/// index-ordered, derivative order >= 1.
/// For the two-electron operator `op` (a/r + b erf(omega r)/r) in every term; the pairs'
/// Schwarz factors stay the Coulomb ones (they bound the erf integrals too, scaled by
/// [`Operator::schwarz_factor`]).
pub(crate) fn grad_4c_into(
    basis: &BasisSet,
    pairs: &PairSet,
    terms: &DensityTerms,
    thresh: f64,
    op: &Operator,
    accs: &mut [GradAcc],
) {
    // slices, not the set: their loads hoist out of the loops (a &PairSet is not Freeze)
    let (prs, drs) = (pairs.pairs(), pairs.derivs());
    let thresh = op.screen_threshold(thresh);
    let sh = &basis.shells;
    let ns = sh.len();
    let nbf = basis.nbf;
    let dens = Dens4c::new(basis, terms);
    let herms: Vec<PairHerm> = dens.jd.iter().map(|d| pair_herm(basis, pairs, d)).collect();
    let np = pairs.len();
    let q = pairs.schwarz();
    let qmax = pairs.schwarz_max();
    let (qd, _) = pairs.deriv_bounds();
    let qdmax = qd.iter().cloned().fold(0.0, f64::max);
    let have_k = !terms.exchange.is_empty();
    // deterministic reduction (independent of the thread count)
    linus_num::det::par_fold_det_into(
        np,
        accs,
        |(w, g, gbuf), idx| {
            // most expensive bra pairs first
            let ij = np - 1 - idx;
            let bra = &prs[ij];
            if bra.nprim == 0 || qd[ij] * qmax + q[ij] * qdmax < thresh {
                return;
            }
            let bd = &drs[ij];
            let (s1, s2) = (bra.sa, bra.sb);
            // the separable terms of a quartet: one list per bra pair, refilled per ket
            let mut jterms: Vec<JTerm> = Vec::with_capacity(dens.jterm_idx.len());
            for kl in 0..=ij {
                let ket = &prs[kl];
                let qq = qd[ij] * q[kl] + q[ij] * qd[kl];
                if ket.nprim == 0 || qq < thresh {
                    continue;
                }
                let (s3, s4) = (ket.sa, ket.sb);
                let (ej, ek) = dens.estimates(ns, [s1, s2, s3, s4]);
                let degq = (if s1 == s2 { 1.0 } else { 2.0 })
                    * (if s3 == s4 { 1.0 } else { 2.0 })
                    * (if ij == kl { 1.0 } else { 2.0 });
                let do_j = degq * qq * ej >= thresh;
                let do_k = have_k && degq * qq * ek >= thresh;
                if !do_j && !do_k {
                    continue;
                }
                let fbk = if ij == kl { 1.0 } else { 2.0 };
                // orientation: the kernel cost scales with the ket dimension
                let swap = grad_cost(ket, bra) < grad_cost(bra, ket);
                let (pb, pk, ib, ik) = if swap {
                    (kl, ij, 3usize, 1usize)
                } else {
                    (ij, kl, 1, 3)
                };
                let (b2, k2) = (&prs[pb], &prs[pk]);
                let sq = if swap {
                    [s3, s4, s1, s2]
                } else {
                    [s1, s2, s3, s4]
                };
                let _ = (ib, ik);
                let gamma = if do_k {
                    let deg = (if s1 == s2 { 1.0 } else { 2.0 })
                        * (if s3 == s4 { 1.0 } else { 2.0 })
                        * fbk;
                    let n = [b2.na, b2.nb, k2.na, k2.nb];
                    gbuf.resize(b2.nab() * k2.nab(), 0.0);
                    exchange_block(
                        &terms.exchange,
                        nbf,
                        [
                            sh[sq[0]].ao_start,
                            sh[sq[1]].ao_start,
                            sh[sq[2]].ao_start,
                            sh[sq[3]].ao_start,
                        ],
                        n,
                        deg,
                        gbuf,
                    );
                    Some(&gbuf[..])
                } else {
                    None
                };
                jterms.clear();
                if do_j {
                    for &(c, i1, i2) in &dens.jterm_idx {
                        let (j1, j2) = if swap { (i2, i1) } else { (i1, i2) };
                        jterms.push(JTerm {
                            coef: c * fbk,
                            bra: HermDens {
                                e: &herms[j1].e[pb],
                                ea: &herms[j1].ea[pb],
                            },
                            ket: HermDens {
                                e: &herms[j2].e[pk],
                                ea: &herms[j2].ea[pk],
                            },
                        });
                    }
                }
                let gq = quartet_grad(b2, &drs[pb], k2, &drs[pk], gamma, &jterms, w);
                let _ = bd;
                for (x, s) in sq.iter().enumerate() {
                    let at = sh[*s].atom;
                    for k in 0..3 {
                        g[at][k] += gq[x][k];
                    }
                }
            }
        },
        |acc, b| add_grads_into(&mut acc.1, &b.1),
    );
}

/// Accumulate the 10 center-pair blocks of a quartet into a (3 natom)^2 Hessian.
pub(crate) fn add_hess_blocks(
    h: &mut [f64],
    n3: usize,
    atoms: [usize; 4],
    blocks: &[[[f64; 3]; 3]; 10],
) {
    for (i, &(x, y)) in D2_CENTER_PAIRS.iter().enumerate() {
        let (ax, ay) = (atoms[x], atoms[y]);
        let m = &blocks[i];
        for k in 0..3 {
            for l in 0..3 {
                h[(3 * ax + k) * n3 + 3 * ay + l] += m[k][l];
                if x != y {
                    h[(3 * ay + l) * n3 + 3 * ax + k] += m[k][l];
                }
            }
        }
    }
}

/// Two-electron Hessian contribution sum_{abcd} ∂²(ab|cd)/∂X∂Y Γ_abcd, (3 natom)^2, into
/// `accs[0].1` (zeroed; owned by an [`Eri4DerivPlan`](super::deriv_plan::Eri4DerivPlan)).
/// Needs index-ordered pairs of derivative order 2.
/// For the two-electron operator `op` (a/r + b erf(omega r)/r) in every term; the pairs'
/// Schwarz factors stay the Coulomb ones (with the derivative bounds they bound the erf
/// integrals too, scaled by [`Operator::schwarz_factor`]).
pub(crate) fn hess_4c_into(
    basis: &BasisSet,
    pairs: &PairSet,
    terms: &DensityTerms,
    thresh: f64,
    op: &Operator,
    accs: &mut [HessAcc],
) {
    // slices, not the set: their loads hoist out of the loops (a &PairSet is not Freeze)
    let (prs, drs) = (pairs.pairs(), pairs.derivs());
    assert!(
        pairs.deriv_order() >= DerivOrder::Two,
        "eri_hess_4c needs second-order pair derivatives"
    );
    let thresh = op.screen_threshold(thresh);
    let natom = basis.natom();
    let n3 = 3 * natom;
    let sh = &basis.shells;
    let ns = sh.len();
    let nbf = basis.nbf;
    let dens = Dens4c::new(basis, terms);
    let np = pairs.len();
    let q = pairs.schwarz();
    let qmax = pairs.schwarz_max();
    let (qd, qd2) = pairs.deriv_bounds();
    let qdmax = qd.iter().cloned().fold(0.0, f64::max);
    let qd2max = qd2.iter().cloned().fold(0.0, f64::max);
    // deterministic reduction (independent of the thread count); the most expensive bra
    // pairs (most kets) come first
    linus_num::det::par_fold_det_into(
        np,
        accs,
        |(w, h, gbuf), item| {
            let ij = np - 1 - item;
            let bra = &prs[ij];
            if bra.nprim == 0 || qd2[ij] * qmax + 2.0 * qd[ij] * qdmax + q[ij] * qd2max < thresh {
                return;
            }
            let bd = &drs[ij];
            let (s1, s2) = (bra.sa, bra.sb);
            for kl in 0..=ij {
                let ket = &prs[kl];
                let qq = qd2[ij] * q[kl] + 2.0 * qd[ij] * qd[kl] + q[ij] * qd2[kl];
                if ket.nprim == 0 || qq < thresh {
                    continue;
                }
                let (s3, s4) = (ket.sa, ket.sb);
                let (ej, ek) = dens.estimates(ns, [s1, s2, s3, s4]);
                let deg = (if s1 == s2 { 1.0 } else { 2.0 })
                    * (if s3 == s4 { 1.0 } else { 2.0 })
                    * (if ij == kl { 1.0 } else { 2.0 });
                if deg * qq * ej.max(ek) < thresh {
                    continue;
                }
                let deg = (if s1 == s2 { 1.0 } else { 2.0 })
                    * (if s3 == s4 { 1.0 } else { 2.0 })
                    * (if ij == kl { 1.0 } else { 2.0 });
                let o = [
                    sh[s1].ao_start,
                    sh[s2].ao_start,
                    sh[s3].ao_start,
                    sh[s4].ao_start,
                ];
                let n = [bra.na, bra.nb, ket.na, ket.nb];
                gbuf.resize(bra.nab() * ket.nab(), 0.0);
                exchange_block(&terms.exchange, nbf, o, n, deg, gbuf);
                coulomb_block(&terms.coulomb, nbf, o, n, deg, gbuf);
                let hp = quartet_hess(bra, bd, ket, &drs[kl], gbuf, w);
                let atoms = [sh[s1].atom, sh[s2].atom, sh[s3].atom, sh[s4].atom];
                add_hess_blocks(h, n3, atoms, &hp.center_blocks());
            }
        },
        |acc, b| linus_num::det::add_into(&mut acc.1, &b.1),
    );
}
