//! Fast three-center (k|mn) and two-center (k|l) Coulomb integrals for density fitting.
//!
//! McMurchie-Davidson in Hermite space, organised as in SHARK's split-RI-J: an auxiliary
//! solid-harmonic Gaussian S_k(r_A) exp(-a r_A^2) equals (2a)^{-l} S_k(d/dA) exp(-a r_A^2)
//! (Hobson's theorem), so its Hermite expansion has only the degree-l terms and its "E matrix"
//! is the solid-harmonic coefficient table scaled by (2a)^{-l}. With the orbital pair in the
//! SHARK layout (E_ip[t']\[mn\] for each primitive pair ip, centre P_ip):
//!
//!   (k|mn) = (-1)^l sum_ip sum_kp c_kp (2a_kp)^{-l} sum_{|t|=l} S_k\[t\] sum_t' R_{t+t'}(P_ip - A) E_ip[t']\[mn\]
//!
//! Auxiliary primitives of equal l are processed W = 8 at a time in SIMD lanes (Boys function,
//! Hermite R recursion, the S_k contraction and the E_ip contraction are all lane-parallel),
//! and the hot kernels are compiled for AVX-512 / AVX2 with run-time dispatch.
//!
//! Screening: shell pairs by Schwarz (Q_mn = max (mn|mn)^1/2 times max_k (k|k)^1/2), primitive
//! pairs by the primitive Schwarz bound (ip|ip)^1/2 times the largest lane bound (kp|kp)^1/2.
//! Both bounds are rigorous (Cauchy-Schwarz for the Coulomb metric).

pub mod dblock;
pub mod deriv;
pub mod deriv_plan;
pub mod dgrad;
mod layout;
mod thresholds;

pub use layout::{df_kept_pairs, DfRows};
pub use thresholds::DfThresholds;

use crate::basis::{cart_components, ncart, BasisSet, Shell};
use crate::core::boys;
use crate::core::hermite::{build_pair, herm_idx, herm_tuv, nherm, ShellPair, ShellView};
use crate::core::solid_harmonics::{cart2sph, cart2sph_cached};
use crate::eri4c::ERI_PREF;
use crate::operator::Operator;
use std::sync::OnceLock;

/// SIMD lanes (auxiliary primitives per batch).
pub const W: usize = 8;
pub type Lanes = [f64; W];
/// Highest total Hermite degree l_aux + l_a + l_b supported.
pub const MAX_L3C: usize = 18;
/// Highest auxiliary angular momentum supported.
pub const MAX_L_AUX: usize = 6;

pub(crate) const ZERO: Lanes = [0.0; W];

// ------------------------------------------------------------------------------------------
// Boys function on lanes: table at T_i = i * 0.1 (i < 1171) with F_0..F_{MAX_L3C+7} and
// exp(-T_i); 8-term Taylor for the top order, exp(-T) = exp(-T_i) exp(T_i - T) by a short
// series, downward recursion. T >= 117: asymptotic form (exp(-T) < 2e-51 is negligible).
// Deliberately not crate::core::boys (boys_lanes_tab): that one switches to the asymptotic form at
// T = 45, takes exp(-T) from simd::exp_nonpos instead of the table and orders its operations
// differently, so sharing it would change every density-fitted result at round-off (all the
// DF validation and benchmark records were made with this one).
// ------------------------------------------------------------------------------------------

const BT_DT: f64 = 0.1;
const BT_INV_DT: f64 = 10.0;
const BT_TMAX: f64 = 117.0;
const BT_ROWS: usize = 1172;
const BT_TAY: usize = 8;
const BT_COLS: usize = MAX_L3C + BT_TAY + 1;

pub(crate) fn boys_table() -> &'static [f64] {
    static T: OnceLock<Vec<f64>> = OnceLock::new();
    T.get_or_init(|| {
        let mut f = vec![0.0; BT_ROWS * BT_COLS];
        for i in 0..BT_ROWS {
            let t = i as f64 * BT_DT;
            let row = &mut f[i * BT_COLS..(i + 1) * BT_COLS];
            boys::boys_series(BT_COLS - 2, t, &mut row[..BT_COLS - 1]);
            row[BT_COLS - 1] = (-t).exp();
        }
        f
    })
}

const INV_INT: [f64; 10] = [
    0.0,
    1.0,
    0.5,
    1.0 / 3.0,
    0.25,
    0.2,
    1.0 / 6.0,
    1.0 / 7.0,
    0.125,
    1.0 / 9.0,
];
const INV_FACT: [f64; 10] = [
    1.0,
    1.0,
    0.5,
    1.0 / 6.0,
    1.0 / 24.0,
    1.0 / 120.0,
    1.0 / 720.0,
    1.0 / 5040.0,
    1.0 / 40320.0,
    1.0 / 362880.0,
];

/// Top-order Taylor sum F_l(T_i - d) = sum_k F_{l+k}(T_i) d^k / k! and exp(-T) =
/// exp(-T_i) exp(d) on lanes.
#[inline(always)]
fn boys_taylor(rows: &[Lanes; BT_TAY], dt: &Lanes, e0: &Lanes) -> (Lanes, Lanes) {
    let mut acc = rows[BT_TAY - 1];
    for k in (0..BT_TAY - 1).rev() {
        let rk = &rows[k];
        for q in 0..W {
            acc[q] = rk[q] + acc[q] * dt[q] * INV_INT[k + 1];
        }
    }
    let mut et = ZERO;
    for q in 0..W {
        let d = dt[q];
        let e = 1.0
            + d * (1.0
                + d * (0.5
                    + d * (INV_FACT[3]
                        + d * (INV_FACT[4]
                            + d * (INV_FACT[5]
                                + d * (INV_FACT[6] + d * (INV_FACT[7] + d * INV_FACT[8])))))));
        et[q] = e0[q] * e;
    }
    (acc, et)
}

/// F_0..=F_lmax at T (per lane) into f[0..=lmax].
#[inline(always)]
#[allow(clippy::needless_range_loop, reason = "indexes several arrays in step")]
fn boys_lanes(lmax: usize, t: &Lanes, f: &mut [Lanes], tab: &[f64]) {
    let mut idx = [0usize; W];
    let mut dt = ZERO;
    let mut tc = ZERO;
    for q in 0..W {
        let tq = if t[q] < BT_TMAX { t[q] } else { 0.0 };
        tc[q] = tq;
        let i = (tq * BT_INV_DT + 0.5) as usize;
        idx[q] = i * BT_COLS;
        dt[q] = i as f64 * BT_DT - tq;
    }
    // gather the table rows into lanes, then evaluate on lanes
    let mut rows = [ZERO; BT_TAY];
    let mut e0 = ZERO;
    for q in 0..W {
        let row = &tab[idx[q]..idx[q] + BT_COLS];
        for k in 0..BT_TAY {
            rows[k][q] = row[lmax + k];
        }
        e0[q] = row[BT_COLS - 1];
    }
    let (top, et) = boys_taylor(&rows, &dt, &e0);
    f[lmax] = top;
    for m in (0..lmax).rev() {
        let inv = 1.0 / (2 * m + 1) as f64;
        let (lo, hi) = f.split_at_mut(m + 1);
        let fm1 = &hi[0];
        let fm = &mut lo[m];
        for q in 0..W {
            fm[q] = (2.0 * tc[q] * fm1[q] + et[q]) * inv;
        }
    }
    // asymptotic lanes
    let mut any_big = false;
    for q in 0..W {
        any_big |= t[q] >= BT_TMAX;
    }
    if any_big {
        for q in 0..W {
            if t[q] >= BT_TMAX {
                let ti = 1.0 / t[q];
                let mut v = 0.5 * (std::f64::consts::PI * ti).sqrt();
                f[0][q] = v;
                for m in 1..=lmax {
                    v *= (2 * m - 1) as f64 * 0.5 * ti;
                    f[m][q] = v;
                }
            }
        }
    }
}

// ------------------------------------------------------------------------------------------
// Hermite R recursion on lanes
// ------------------------------------------------------------------------------------------

#[derive(Clone, Copy)]
pub(crate) struct RStep {
    axis: u8,
    j1: u16,
    j2: u16,
    c: f64,
}

pub(crate) fn r_prog() -> &'static [RStep] {
    static P: OnceLock<Vec<RStep>> = OnceLock::new();
    P.get_or_init(|| {
        let tuv = herm_tuv();
        (0..nherm(MAX_L3C))
            .map(|h| {
                if h == 0 {
                    return RStep {
                        axis: 0,
                        j1: 0,
                        j2: 0,
                        c: 0.0,
                    };
                }
                let [t, u, v] = tuv[h];
                let (t, u, v) = (t as usize, u as usize, v as usize);
                if t > 0 {
                    let j2 = if t > 1 { herm_idx(t - 2, u, v) } else { 0 };
                    RStep {
                        axis: 0,
                        j1: herm_idx(t - 1, u, v) as u16,
                        j2: j2 as u16,
                        c: (t as f64 - 1.0).max(0.0),
                    }
                } else if u > 0 {
                    let j2 = if u > 1 { herm_idx(t, u - 2, v) } else { 0 };
                    RStep {
                        axis: 1,
                        j1: herm_idx(t, u - 1, v) as u16,
                        j2: j2 as u16,
                        c: (u as f64 - 1.0).max(0.0),
                    }
                } else {
                    let j2 = if v > 1 { herm_idx(t, u, v - 2) } else { 0 };
                    RStep {
                        axis: 2,
                        j1: herm_idx(t, u, v - 1) as u16,
                        j2: j2 as u16,
                        c: (v as f64 - 1.0).max(0.0),
                    }
                }
            })
            .collect()
    })
}

/// x * p1 + c * p2 (separate reference arguments keep the lanes vectorized after inlining)
#[inline(always)]
fn lane_fma(x: &Lanes, p1: &Lanes, c: f64, p2: &Lanes) -> Lanes {
    let mut v = ZERO;
    for q in 0..W {
        v[q] = x[q] * p1[q] + c * p2[q];
    }
    v
}

/// acc + a * b (lanes)
#[inline(always)]
fn lane_mul_add(acc: &Lanes, a: &Lanes, b: &Lanes) -> Lanes {
    let mut v = ZERO;
    for q in 0..W {
        v[q] = acc[q] + a[q] * b[q];
    }
    v
}

/// R_tuv for t+u+v <= l into out[0..nherm(l)], from rn[n] = R^{(n)}_{000} (n = 0..=l).
/// `work` needs 2 * nherm(l) entries.
#[inline(always)]
fn r_lanes(
    l: usize,
    rn: &[Lanes],
    pq: &[Lanes; 3],
    out: &mut [Lanes],
    work: &mut [Lanes],
    prog: &[RStep],
) {
    if l == 0 {
        out[0] = rn[0];
        return;
    }
    let nh = nherm(l);
    let (a, b) = work[..2 * nh].split_at_mut(nh);
    let mut prev: &mut [Lanes] = a;
    let mut cur: &mut [Lanes] = b;
    prev[0] = rn[l];
    for n in (0..l).rev() {
        let target: &mut [Lanes] = if n == 0 { &mut *out } else { &mut *cur };
        target[0] = rn[n];
        let cnt = nherm(l - n);
        for h in 1..cnt {
            let s = prog[h];
            target[h] = lane_fma(
                &pq[s.axis as usize],
                &prev[s.j1 as usize],
                s.c,
                &prev[s.j2 as usize],
            );
        }
        if n > 0 {
            std::mem::swap(&mut prev, &mut cur);
        }
    }
}

// ------------------------------------------------------------------------------------------
// Solid-harmonic contraction tables
// ------------------------------------------------------------------------------------------

/// For auxiliary l and orbital-pair degree lab: idx[c * nherm(lab) + t'] = herm_idx(c + t')
/// for the Cartesian components c of degree l, and the nonzero solid-harmonic coefficients
/// (k, c, S_k[c]).
pub(crate) struct HTab {
    pub(crate) idx: Vec<u16>,
    /// per function k: (c * nherm(lab), S_k[c]) for the nonzero coefficients
    nzk: Vec<Vec<(u32, f64)>>,
}

pub(crate) fn htab(l: usize, lab: usize) -> &'static HTab {
    const NL: usize = MAX_L3C + 1;
    static TABS: OnceLock<Vec<OnceLock<HTab>>> = OnceLock::new();
    let tabs = TABS.get_or_init(|| (0..NL * NL).map(|_| OnceLock::new()).collect());
    assert!(
        l <= MAX_L_AUX && l + lab <= MAX_L3C,
        "l_aux={l} l_ab={lab} beyond MAX_L3C"
    );
    tabs[l * NL + lab].get_or_init(|| {
        let tuv = herm_tuv();
        let nh = nherm(lab);
        let cc = cart_components(l);
        let mut idx = Vec::with_capacity(cc.len() * nh);
        for c in &cc {
            for s in &tuv[..nh] {
                idx.push(herm_idx(
                    c[0] + s[0] as usize,
                    c[1] + s[1] as usize,
                    c[2] + s[2] as usize,
                ) as u16);
            }
        }
        let t = cart2sph(l);
        let nc = ncart(l);
        let nk = if l <= 1 { nc } else { 2 * l + 1 };
        let nzk = (0..nk)
            .map(|k| {
                (0..nc)
                    .filter(|&c| t[k * nc + c] != 0.0)
                    .map(|c| ((c * nh) as u32, t[k * nc + c]))
                    .collect()
            })
            .collect();
        HTab { idx, nzk }
    })
}

// ------------------------------------------------------------------------------------------
// Data: orbital shell pairs and auxiliary primitive batches
// ------------------------------------------------------------------------------------------

/// An orbital shell pair prepared for 3-center integrals: transposed E matrices
/// Et_ip\[mn\]\[t'\] and primitive Schwarz bounds.
#[derive(Clone, Debug)]
pub struct Pair3c {
    pub sa: usize,
    pub sb: usize,
    pub na: usize,
    pub nb: usize,
    pub lab: usize,
    pub p: Vec<f64>,
    pub pc: Vec<[f64; 3]>,
    /// nprim * nab * nherm(lab)
    pub et: Vec<f64>,
    /// (ip|ip)^1/2 per primitive pair (max over mn)
    pub qprim: Vec<f64>,
}

impl Pair3c {
    pub fn nab(&self) -> usize {
        self.na * self.nb
    }
    pub fn nprim(&self) -> usize {
        self.p.len()
    }
    pub fn from_shell_pair(sp: &ShellPair) -> Self {
        let lab = sp.lab();
        let nh = nherm(lab);
        let nab = sp.nab();
        let mut et = vec![0.0; sp.nprim * nab * nh];
        let mut qprim = vec![0.0; sp.nprim];
        let tuv = herm_tuv();
        // R at PQ = 0 with alpha = p/2: only even (t,u,v) survive
        let mut fm = vec![0.0; boys::MAX_M + 1];
        let mut r = vec![0.0; nherm(2 * lab)];
        let mut rw = vec![0.0; 2 * nherm(2 * lab)];
        // per (t, u) of Herm(lab): the R slot of t + u and (-1)^|u|
        let rtab: Vec<(usize, f64)> = (0..nh * nh)
            .map(|x| {
                let ([a0, a1, a2], [b0, b1, b2]) = (tuv[x / nh], tuv[x % nh]);
                let sign = if (b0 + b1 + b2) % 2 == 0 { 1.0 } else { -1.0 };
                (
                    herm_idx((a0 + b0) as usize, (a1 + b1) as usize, (a2 + b2) as usize),
                    sign,
                )
            })
            .collect();
        let mut nz: Vec<(usize, f64)> = Vec::with_capacity(nh);
        for ip in 0..sp.nprim {
            let e = sp.e_prim(ip);
            let dst = &mut et[ip * nab * nh..(ip + 1) * nab * nh];
            for t in 0..nh {
                for ab in 0..nab {
                    dst[ab * nh + t] = e[t * nab + ab];
                }
            }
            let p = sp.p[ip];
            boys::boys(2 * lab, 0.0, &mut fm);
            let alpha = 0.5 * p;
            let pref = ERI_PREF / (p * p * (2.0 * p).sqrt());
            crate::core::hermite::r_tensor(2 * lab, alpha, [0.0; 3], &fm, pref, &mut r, &mut rw);
            let mut qmax: f64 = 0.0;
            for ab in 0..nab {
                let row = &dst[ab * nh..(ab + 1) * nh];
                // the row's nonzero E (u ascending): a zero factor adds a zero product, which
                // leaves |s| as it was, so the sum below is the dense loop's, bit for bit
                nz.clear();
                nz.extend((0..nh).filter(|&u| row[u] != 0.0).map(|u| (u, row[u])));
                let mut s = 0.0;
                for (t, &et_) in row.iter().enumerate() {
                    if et_ == 0.0 {
                        continue;
                    }
                    let tab = &rtab[t * nh..(t + 1) * nh];
                    for &(u, eu) in &nz {
                        let (idx, sign) = tab[u];
                        s += et_ * sign * r[idx] * eu;
                    }
                }
                qmax = qmax.max(s.abs());
            }
            qprim[ip] = qmax.sqrt();
        }
        Pair3c {
            sa: sp.sa,
            sb: sp.sb,
            na: sp.na,
            nb: sp.nb,
            lab,
            p: sp.p.clone(),
            pc: sp.pc.clone(),
            et,
            qprim,
        }
    }
}

/// W auxiliary primitives of equal l (padding lanes have coef = 0).
#[derive(Clone, Debug)]
pub struct AuxChunk {
    pub l: usize,
    pub exp: Lanes,
    /// c_kp (2a)^{-l} (-1)^l
    pub coef: Lanes,
    pub cx: Lanes,
    pub cy: Lanes,
    pub cz: Lanes,
    /// largest lane Schwarz bound (kp|kp)^1/2
    pub q: f64,
    /// first auxiliary function index of each lane's shell (u32::MAX for padding)
    pub col: [u32; W],
}

/// Auxiliary basis prepared for 3-center integrals.
#[derive(Clone, Debug)]
pub struct Aux3c {
    pub naux: usize,
    pub shells: Vec<Shell>,
    /// (k|k)^1/2 per shell (max over its functions), contracted
    pub qshell: Vec<f64>,
    pub qmax: f64,
}

/// Self-repulsion (k|k) of the functions of an auxiliary shell (max over k), contracted.
fn aux_self_repulsion(s: &Shell) -> f64 {
    let l = s.l;
    let sp = build_pair(&ShellView::from_shell(s), &ShellView::unit(s.center), 0, 0);
    let p3 = Pair3c::from_shell_pair(&sp);
    // reuse the pair machinery: (k|k) for the (k, unit) "pair" is exactly what qprim bounds
    // per primitive; for the contracted value compute directly
    let nh = nherm(l);
    let tuv = herm_tuv();
    let mut fm = vec![0.0; boys::MAX_M + 1];
    let mut r = vec![0.0; nherm(2 * l)];
    let mut rw = vec![0.0; 2 * nherm(2 * l)];
    let nk = s.nfunc();
    let mut best: f64 = 0.0;
    for k in 0..nk {
        let mut v = 0.0;
        for i in 0..p3.nprim() {
            for j in 0..p3.nprim() {
                let (p, q) = (p3.p[i], p3.p[j]);
                let alpha = p * q / (p + q);
                boys::boys(2 * l, 0.0, &mut fm);
                let pref = ERI_PREF / (p * q * (p + q).sqrt());
                crate::core::hermite::r_tensor(2 * l, alpha, [0.0; 3], &fm, pref, &mut r, &mut rw);
                let ei = &p3.et[i * nk * nh + k * nh..i * nk * nh + (k + 1) * nh];
                let ej = &p3.et[j * nk * nh + k * nh..j * nk * nh + (k + 1) * nh];
                for t in 0..nh {
                    for u in 0..nh {
                        let [a0, a1, a2] = tuv[t];
                        let [b0, b1, b2] = tuv[u];
                        let sign = if (b0 + b1 + b2) % 2 == 0 { 1.0 } else { -1.0 };
                        v += ei[t]
                            * sign
                            * r[herm_idx(
                                (a0 + b0) as usize,
                                (a1 + b1) as usize,
                                (a2 + b2) as usize,
                            )]
                            * ej[u];
                    }
                }
            }
        }
        best = best.max(v.abs());
    }
    best
}

impl Aux3c {
    pub fn new(aux: &BasisSet) -> Self {
        for s in &aux.shells {
            assert!(
                s.l <= 1 || s.pure,
                "auxiliary basis must use spherical functions"
            );
            assert!(s.l <= MAX_L_AUX, "auxiliary l = {} not supported", s.l);
        }
        let qshell: Vec<f64> = aux
            .shells
            .iter()
            .map(|s| aux_self_repulsion(s).sqrt())
            .collect();
        let qmax = qshell.iter().cloned().fold(0.0, f64::max);
        Aux3c {
            naux: aux.nbf,
            shells: aux.shells.clone(),
            qshell,
            qmax,
        }
    }

    /// Lane batches for auxiliary shells in `range` (grouped by l; columns relative to the
    /// first function of the range).
    pub fn chunks(&self, range: std::ops::Range<usize>) -> Vec<AuxChunk> {
        let col0 = if range.start < self.shells.len() {
            self.shells[range.start].ao_start
        } else {
            self.naux
        };
        let lmax = self.shells[range.clone()]
            .iter()
            .map(|s| s.l)
            .max()
            .unwrap_or(0);
        let mut out = vec![];
        for l in 0..=lmax {
            let mut lanes: Vec<(f64, f64, [f64; 3], u32, f64)> = vec![];
            for si in range.clone() {
                let s = &self.shells[si];
                if s.l != l {
                    continue;
                }
                for (&a, &c) in s.exps.iter().zip(&s.coefs) {
                    let sign = if l % 2 == 0 { 1.0 } else { -1.0 };
                    let cf = c * (2.0 * a).powi(-(l as i32)) * sign;
                    // primitive bound: |c| times the self-repulsion of the normalised primitive
                    let prim = Shell {
                        l,
                        pure: s.pure,
                        atom: s.atom,
                        center: s.center,
                        exps: vec![a],
                        coefs: vec![c],
                        ao_start: 0,
                    };
                    let q = aux_self_repulsion(&prim).sqrt();
                    lanes.push((a, cf, s.center, (s.ao_start - col0) as u32, q));
                }
            }
            for ch in lanes.chunks(W) {
                let mut c = AuxChunk {
                    l,
                    exp: [1.0; W],
                    coef: ZERO,
                    cx: ZERO,
                    cy: ZERO,
                    cz: ZERO,
                    q: 0.0,
                    col: [u32::MAX; W],
                };
                for (q, &(a, cf, cen, col, qb)) in ch.iter().enumerate() {
                    c.exp[q] = a;
                    c.coef[q] = cf;
                    c.cx[q] = cen[0];
                    c.cy[q] = cen[1];
                    c.cz[q] = cen[2];
                    c.col[q] = col;
                    c.q = c.q.max(qb);
                }
                if let Some(&(_, _, cen, _, _)) = ch.first() {
                    for q in ch.len()..W {
                        c.cx[q] = cen[0];
                        c.cy[q] = cen[1];
                        c.cz[q] = cen[2];
                    }
                }
                out.push(c);
            }
        }
        out
    }
}

// ------------------------------------------------------------------------------------------
// Kernel
// ------------------------------------------------------------------------------------------

/// Scratch space for the kernels (one per thread).
pub struct Work3c {
    pub(crate) fm: Vec<Lanes>,
    pub(crate) rn: Vec<Lanes>,
    pub(crate) r: Vec<Lanes>,
    pub(crate) rw: Vec<Lanes>,
    pub(crate) h: Vec<Lanes>,
    pub(crate) acc: Vec<Lanes>,
    pub prim_skipped: usize,
    pub prim_done: usize,
}

impl Default for Work3c {
    fn default() -> Self {
        Self::new()
    }
}

impl Work3c {
    pub fn new() -> Self {
        Work3c {
            fm: vec![ZERO; MAX_L3C + 1],
            rn: vec![ZERO; MAX_L3C + 1],
            r: vec![ZERO; nherm(MAX_L3C)],
            rw: vec![ZERO; 2 * nherm(MAX_L3C)],
            h: vec![],
            acc: vec![],
            prim_skipped: 0,
            prim_done: 0,
        }
    }
}

/// Hermite integrals R_{t}(P_ip - A_q) (t <= lt) for one primitive pair and W auxiliary
/// primitives, scaled by the prefactor and the lane coefficients.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn prim_hermite(
    p: f64,
    pc: [f64; 3],
    ch: &AuxChunk,
    lt: usize,
    omega: f64,
    tab: &[f64],
    prog: &[RStep],
    fm: &mut [Lanes],
    rn: &mut [Lanes],
    r: &mut [Lanes],
    rw: &mut [Lanes],
) {
    let w2 = omega * omega;
    let mut t = ZERO;
    let mut alpha2 = ZERO;
    let mut pref = ZERO;
    let mut pq = [ZERO; 3];
    for q in 0..W {
        let a = ch.exp[q];
        let s = p + a;
        let al = p * a / s;
        // erf(omega r)/r: alpha -> alpha w^2/(alpha + w^2), prefactor * sqrt(alpha_w/alpha)
        let (al, scale) = if omega > 0.0 {
            let aw = al * w2 / (al + w2);
            (aw, (aw / al).sqrt())
        } else {
            (al, 1.0)
        };
        let x = pc[0] - ch.cx[q];
        let y = pc[1] - ch.cy[q];
        let z = pc[2] - ch.cz[q];
        pq[0][q] = x;
        pq[1][q] = y;
        pq[2][q] = z;
        t[q] = al * (x * x + y * y + z * z);
        alpha2[q] = -2.0 * al;
        pref[q] = ERI_PREF / (p * a * s.sqrt()) * ch.coef[q] * scale;
    }
    boys_lanes(lt, &t, fm, tab);
    scale_rn(lt, pref, &alpha2, fm, rn);
    r_lanes(lt, rn, &pq, r, rw, prog);
}

/// As [`prim_hermite`] for the operator `full`/r + `erf` erf(omega r)/r: the R recursion is
/// linear in R^(n)_000 and independent of the exponent, so only the starting values combine
/// (two Boys evaluations per lane batch; see [`crate::operator`]).
#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn prim_hermite_comb(
    p: f64,
    pc: [f64; 3],
    ch: &AuxChunk,
    lt: usize,
    full: f64,
    erf: f64,
    omega: f64,
    tab: &[f64],
    prog: &[RStep],
    fm: &mut [Lanes],
    rn: &mut [Lanes],
    r: &mut [Lanes],
    rw: &mut [Lanes],
) {
    let w2 = omega * omega;
    let (mut t1, mut t2) = (ZERO, ZERO);
    let (mut a1, mut a2) = (ZERO, ZERO);
    let (mut p1, mut p2) = (ZERO, ZERO);
    let mut pq = [ZERO; 3];
    for q in 0..W {
        let a = ch.exp[q];
        let s = p + a;
        let al = p * a / s;
        let aw = al * w2 / (al + w2);
        let x = pc[0] - ch.cx[q];
        let y = pc[1] - ch.cy[q];
        let z = pc[2] - ch.cz[q];
        pq[0][q] = x;
        pq[1][q] = y;
        pq[2][q] = z;
        let r2 = x * x + y * y + z * z;
        t1[q] = al * r2;
        t2[q] = aw * r2;
        a1[q] = -2.0 * al;
        a2[q] = -2.0 * aw;
        let base = ERI_PREF / (p * a * s.sqrt()) * ch.coef[q];
        p1[q] = base * full;
        p2[q] = base * (aw / al).sqrt() * erf;
    }
    boys_lanes(lt, &t1, fm, tab);
    scale_rn(lt, p1, &a1, fm, rn);
    boys_lanes(lt, &t2, fm, tab);
    let mut f = p2;
    for n in 0..=lt {
        let (fmn, rnn) = (&fm[n], &mut rn[n]);
        for q in 0..W {
            rnn[q] += f[q] * fmn[q];
            f[q] *= a2[q];
        }
    }
    r_lanes(lt, rn, &pq, r, rw, prog);
}

#[inline(always)]
fn scale_rn(lt: usize, pref: Lanes, alpha2: &Lanes, fm: &[Lanes], rn: &mut [Lanes]) {
    let mut f = pref;
    for n in 0..=lt {
        let fmn = &fm[n];
        let rnn = &mut rn[n];
        for q in 0..W {
            rnn[q] = f[q] * fmn[q];
            f[q] *= alpha2[q];
        }
    }
}

/// H[k][t'] = sum_c S_k[c] R[c + t']
#[inline(always)]
pub(crate) fn h_contract(h: &mut [Lanes], r: &[Lanes], ht: &HTab, nh: usize) {
    for (k, nzk) in ht.nzk.iter().enumerate() {
        let hk = &mut h[k * nh..(k + 1) * nh];
        for (tt, hv) in hk.iter_mut().enumerate() {
            let mut v = ZERO;
            for &(coff, coef) in nzk {
                let rv = &r[ht.idx[coff as usize + tt] as usize];
                for q in 0..W {
                    v[q] += coef * rv[q];
                }
            }
            *hv = v;
        }
    }
}

/// acc[k][mn] += sum_t' H[k][t'] Et[mn][t']
#[inline(always)]
pub(crate) fn acc_contract(
    acc: &mut [Lanes],
    h: &[Lanes],
    et: &[f64],
    nk: usize,
    nab: usize,
    nh: usize,
) {
    for k in 0..nk {
        let hk = &h[k * nh..(k + 1) * nh];
        let ak = &mut acc[k * nab..(k + 1) * nab];
        let mut ab = 0;
        while ab + 4 <= nab {
            let e0 = &et[ab * nh..(ab + 1) * nh];
            let e1 = &et[(ab + 1) * nh..(ab + 2) * nh];
            let e2 = &et[(ab + 2) * nh..(ab + 3) * nh];
            let e3 = &et[(ab + 3) * nh..(ab + 4) * nh];
            let (mut s0, mut s1, mut s2, mut s3) = (ak[ab], ak[ab + 1], ak[ab + 2], ak[ab + 3]);
            for tt in 0..nh {
                let hv = &hk[tt];
                let (v0, v1, v2, v3) = (e0[tt], e1[tt], e2[tt], e3[tt]);
                for q in 0..W {
                    s0[q] += hv[q] * v0;
                    s1[q] += hv[q] * v1;
                    s2[q] += hv[q] * v2;
                    s3[q] += hv[q] * v3;
                }
            }
            ak[ab] = s0;
            ak[ab + 1] = s1;
            ak[ab + 2] = s2;
            ak[ab + 3] = s3;
            ab += 4;
        }
        while ab < nab {
            let e0 = &et[ab * nh..(ab + 1) * nh];
            let mut s0 = ak[ab];
            for tt in 0..nh {
                let hv = &hk[tt];
                let v0 = e0[tt];
                for q in 0..W {
                    s0[q] += hv[q] * v0;
                }
            }
            ak[ab] = s0;
            ab += 1;
        }
    }
}

/// Integrals (k|mn) of one orbital shell pair with the auxiliary primitives in `chunks`,
/// accumulated into out[mn * ld + col(k)] (mn = ia * nb + ib).
#[inline(always)]
#[allow(clippy::needless_range_loop, reason = "indexes several arrays in step")]
fn pair_block_impl(
    pair: &Pair3c,
    chunks: &[AuxChunk],
    out: &mut [f64],
    ld: usize,
    thresh: f64,
    op: Operator,
    w: &mut Work3c,
) {
    let tab = boys_table();
    let prog = r_prog();
    let lab = pair.lab;
    let nh = nherm(lab);
    let nab = pair.nab();
    let nprim = pair.nprim();
    let qpmax = pair.qprim.iter().cloned().fold(0.0, f64::max);
    let Work3c {
        fm,
        rn,
        r,
        rw,
        h,
        acc,
        prim_skipped,
        prim_done,
    } = w;
    for ch in chunks {
        if qpmax * ch.q < thresh {
            *prim_skipped += nprim;
            continue;
        }
        let l = ch.l;
        let lt = l + lab;
        let ht = htab(l, lab);
        let nk = if l <= 1 { ncart(l) } else { 2 * l + 1 };
        if acc.len() < nk * nab {
            acc.resize(nk * nab, ZERO);
        }
        if h.len() < nk * nh {
            h.resize(nk * nh, ZERO);
        }
        let acc = &mut acc[..nk * nab];
        acc.fill(ZERO);
        let h = &mut h[..nk * nh];
        let mut any = false;
        for ip in 0..nprim {
            if pair.qprim[ip] * ch.q < thresh {
                *prim_skipped += 1;
                continue;
            }
            *prim_done += 1;
            any = true;
            let et = &pair.et[ip * nab * nh..(ip + 1) * nab * nh];
            prim_hermite_sel(
                pair.p[ip],
                pair.pc[ip],
                ch,
                lt,
                op,
                tab,
                prog,
                fm,
                rn,
                r,
                rw,
            );
            if l == 0 {
                acc_contract(acc, &r[..nh], et, 1, nab, nh);
            } else {
                h_contract(h, r, ht, nh);
                acc_contract(acc, h, et, nk, nab, nh);
            }
        }
        if !any {
            continue;
        }
        for q in 0..W {
            let col = ch.col[q];
            if col == u32::MAX {
                continue;
            }
            let col = col as usize;
            for k in 0..nk {
                for ab in 0..nab {
                    out[ab * ld + col + k] += acc[k * nab + ab][q];
                }
            }
        }
    }
}

// ------------------------------------------------------------------------------------------
// Split-RI-J kernels (integral-direct Coulomb in Hermite space)
//
// gamma_k = sum_mn (k|mn) D_mn = (-1)^l c (2a)^-l sum_c S_k[c] sum_ip sum_t' R_{c+t'} Dh_ip[t'],
//   Dh_ip[t'] = sum_mn E_ip[t'][mn] D_mn  (density in the pair's Hermite basis),
// J_mn = sum_k (k|mn) d_k = sum_ip sum_t' E_ip[t'][mn] Jh_ip[t'],
//   Jh_ip[t'] = sum_lanes sum_c R_{c+t'} dc[c],  dc[c] = sum_k S_k[c] d_k (per lane).
// No (k|mn) block is ever formed: per primitive pair and lane batch the work is the Boys
// function, the R recursion and one ncart(l) x nherm(lab) contraction.
// ------------------------------------------------------------------------------------------

/// Hermite density of a pair: dh[ip * nh + t'] = sum_mn Et_ip\[mn\]\[t'\] dw\[mn\].
pub fn pair_hermite_density(pair: &Pair3c, dw: &[f64], dh: &mut [f64]) {
    let nh = nherm(pair.lab);
    let nab = pair.nab();
    for ip in 0..pair.nprim() {
        let et = &pair.et[ip * nab * nh..(ip + 1) * nab * nh];
        let o = &mut dh[ip * nh..(ip + 1) * nh];
        o.fill(0.0);
        for (ab, &d) in dw[..nab].iter().enumerate() {
            if d == 0.0 {
                continue;
            }
            for (ov, &e) in o.iter_mut().zip(&et[ab * nh..(ab + 1) * nh]) {
                *ov += e * d;
            }
        }
    }
}

/// J rows of a pair from its Hermite potential: jw\[mn\] += sum_ip sum_t' Et_ip\[mn\]\[t'\] jh[ip * nh + t'].
pub fn pair_from_hermite(pair: &Pair3c, jh: &[f64], jw: &mut [f64]) {
    let nh = nherm(pair.lab);
    let nab = pair.nab();
    for ip in 0..pair.nprim() {
        let et = &pair.et[ip * nab * nh..(ip + 1) * nab * nh];
        let h = &jh[ip * nh..(ip + 1) * nh];
        for (ab, o) in jw[..nab].iter_mut().enumerate() {
            let mut s = 0.0;
            for (&e, &hv) in et[ab * nh..(ab + 1) * nh].iter().zip(h) {
                s += e * hv;
            }
            *o += s;
        }
    }
}

/// Fitted coefficients of a lane batch in the Cartesian-Hermite basis into `out` (ncart(l)
/// lanes, overwritten): out\[c\]\[lane\] = sum_k S_k\[c\] d[col(lane) + k] (d indexed
/// relative to the batch's first function).
#[allow(clippy::needless_range_loop, reason = "indexes several arrays in step")]
pub fn chunk_coefficients(ch: &AuxChunk, d: &[f64], out: &mut Vec<Lanes>) {
    let l = ch.l;
    let nc = ncart(l);
    let nk = if l <= 1 { nc } else { 2 * l + 1 };
    let t = cart2sph(l);
    out.clear();
    out.resize(nc, ZERO);
    for q in 0..W {
        if ch.col[q] == u32::MAX {
            continue;
        }
        let c0 = ch.col[q] as usize;
        for k in 0..nk {
            let dk = d[c0 + k];
            for c in 0..nc {
                out[c][q] += t[k * nc + c] * dk;
            }
        }
    }
}

#[inline(always)]
#[allow(clippy::needless_range_loop, reason = "indexes several arrays in step")]
fn pair_gamma_impl(
    pair: &Pair3c,
    chunks: &[AuxChunk],
    dh: &[f64],
    gamma: &mut [f64],
    thresh: f64,
    w: &mut Work3c,
) {
    let tab = boys_table();
    let prog = r_prog();
    let lab = pair.lab;
    let nh = nherm(lab);
    let nprim = pair.nprim();
    let qpmax = pair.qprim.iter().cloned().fold(0.0, f64::max);
    let dmax = dh.iter().fold(0.0f64, |a, &x| a.max(x.abs()));
    if dmax == 0.0 {
        return;
    }
    let Work3c {
        fm, rn, r, rw, acc, ..
    } = w;
    for ch in chunks {
        if qpmax * ch.q < thresh {
            continue;
        }
        let l = ch.l;
        let lt = l + lab;
        let ht = htab(l, lab);
        let nc = ncart(l);
        if acc.len() < nc {
            acc.resize(nc, ZERO);
        }
        let g = &mut acc[..nc];
        g.fill(ZERO);
        for ip in 0..nprim {
            let d = &dh[ip * nh..(ip + 1) * nh];
            if pair.qprim[ip] * ch.q < thresh {
                continue;
            }
            prim_hermite(
                pair.p[ip],
                pair.pc[ip],
                ch,
                lt,
                0.0,
                tab,
                prog,
                fm,
                rn,
                r,
                rw,
            );
            hermite_dot(g, r, &ht.idx, d, nh);
        }
        // gamma_k += sum_c S_k[c] g[c] per lane
        let t = cart2sph_cached(l);
        let nk = if l <= 1 { nc } else { 2 * l + 1 };
        for q in 0..W {
            if ch.col[q] == u32::MAX {
                continue;
            }
            let c0 = ch.col[q] as usize;
            for k in 0..nk {
                let mut s = 0.0;
                for c in 0..nc {
                    s += t[k * nc + c] * g[c][q];
                }
                gamma[c0 + k] += s;
            }
        }
    }
}

/// g[c] += sum_t' R[idx[c * nh + t']] d[t'] (lanes)
#[inline(always)]
fn hermite_dot(g: &mut [Lanes], r: &[Lanes], idx: &[u16], d: &[f64], nh: usize) {
    for (c, gc) in g.iter_mut().enumerate() {
        let ix = &idx[c * nh..(c + 1) * nh];
        let mut s = *gc;
        for (&i, &dv) in ix.iter().zip(d) {
            let rv = &r[i as usize];
            for q in 0..W {
                s[q] += rv[q] * dv;
            }
        }
        *gc = s;
    }
}

/// jl[t'] += sum_c R[idx[c * nh + t']] dc[c] (lanes)
#[inline(always)]
fn hermite_axpy(jl: &mut [Lanes], r: &[Lanes], idx: &[u16], dc: &[Lanes], nh: usize) {
    for (c, dcv) in dc.iter().enumerate() {
        let ix = &idx[c * nh..(c + 1) * nh];
        for (jv, &i) in jl.iter_mut().zip(ix) {
            *jv = lane_mul_add(jv, &r[i as usize], dcv);
        }
    }
}

#[inline(always)]
fn pair_jh_impl(
    pair: &Pair3c,
    chunks: &[AuxChunk],
    dc: &[Vec<Lanes>],
    cmax: &[f64],
    jh: &mut [f64],
    thresh: f64,
    w: &mut Work3c,
) {
    let tab = boys_table();
    let prog = r_prog();
    let lab = pair.lab;
    let nh = nherm(lab);
    let nprim = pair.nprim();
    let qpmax = pair.qprim.iter().cloned().fold(0.0, f64::max);
    let Work3c {
        fm, rn, r, rw, acc, ..
    } = w;
    if acc.len() < nprim * nh {
        acc.resize(nprim * nh, ZERO);
    }
    let jl = &mut acc[..nprim * nh];
    jl.fill(ZERO);
    for (ci, ch) in chunks.iter().enumerate() {
        if qpmax * ch.q < thresh || cmax[ci] == 0.0 {
            continue;
        }
        let l = ch.l;
        let lt = l + lab;
        let ht = htab(l, lab);
        for ip in 0..nprim {
            if pair.qprim[ip] * ch.q < thresh {
                continue;
            }
            prim_hermite(
                pair.p[ip],
                pair.pc[ip],
                ch,
                lt,
                0.0,
                tab,
                prog,
                fm,
                rn,
                r,
                rw,
            );
            hermite_axpy(&mut jl[ip * nh..(ip + 1) * nh], r, &ht.idx, &dc[ci], nh);
        }
    }
    for (o, v) in jh[..nprim * nh].iter_mut().zip(jl.iter()) {
        *o = v.iter().sum();
    }
}

linus_num::multiversion!(
    /// gamma[col(k)] += sum_mn (k|mn) D_mn for one pair, from its Hermite density `dh`
    /// (see [`pair_hermite_density`]). Primitive batches with a Schwarz bound below `thresh` are skipped.
    pub fn pair_gamma(pair: &Pair3c, chunks: &[AuxChunk], dh: &[f64], gamma: &mut [f64], thresh: f64, w: &mut Work3c) -> () = pair_gamma_impl);

linus_num::multiversion!(
    /// Hermite potential jh[ip * nh + t'] = sum_k d_k (k|Lambda_t'^{ip}) of one pair for fitted
    /// coefficients given per chunk as [`chunk_coefficients`] (`cmax`: max |dc| per chunk; zero
    /// chunks are skipped).
    pub fn pair_jh(pair: &Pair3c, chunks: &[AuxChunk], dc: &[Vec<Lanes>], cmax: &[f64], jh: &mut [f64], thresh: f64, w: &mut Work3c) -> () = pair_jh_impl);

/// Accumulate (k|op|mn) for one shell pair and the auxiliary lanes in `chunks` into
/// out[mn * ld + col(k)], for the operator `op` (evaluated with its [`Operator::kernel`]).
/// Primitive batches are screened with the Coulomb Schwarz bounds at the operator's
/// [`Operator::screen_threshold`] of `thresh`, as in the four-center code.
pub fn pair_block(
    pair: &Pair3c,
    chunks: &[AuxChunk],
    out: &mut [f64],
    ld: usize,
    thresh: f64,
    op: &Operator,
    w: &mut Work3c,
) {
    pair_block_sel(
        pair,
        chunks,
        out,
        ld,
        op.screen_threshold(thresh),
        op.kernel(),
        w,
    )
}

linus_num::multiversion!(fn pair_block_sel(pair: &Pair3c, chunks: &[AuxChunk], out: &mut [f64], ld: usize, thresh: f64, op: Operator, w: &mut Work3c) -> () = pair_block_impl);

/// [`prim_hermite`] or [`prim_hermite_comb`] for the operator `op`.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn prim_hermite_sel(
    p: f64,
    pc: [f64; 3],
    ch: &AuxChunk,
    lt: usize,
    op: Operator,
    tab: &[f64],
    prog: &[RStep],
    fm: &mut [Lanes],
    rn: &mut [Lanes],
    r: &mut [Lanes],
    rw: &mut [Lanes],
) {
    match op {
        Operator::Attenuated {
            coulomb: full,
            long_range: erf,
            omega,
        } => prim_hermite_comb(p, pc, ch, lt, full, erf, omega, tab, prog, fm, rn, r, rw),
        single => prim_hermite(p, pc, ch, lt, single.omega(), tab, prog, fm, rn, r, rw),
    }
}

/// Two-center metric (k|op|l) over the auxiliary shells, naux x naux row-major (the density
/// fitting metric for op = 1/r, the long-range one for erf(omega r)/r).
pub fn metric(aux: &Aux3c, thresh: f64, op: &Operator) -> Vec<f64> {
    use rayon::prelude::*;
    let kernel = op.kernel();
    let n = aux.naux;
    let chunks = aux.chunks(0..aux.shells.len());
    let pairs: Vec<Pair3c> = aux
        .shells
        .iter()
        .map(|s| {
            Pair3c::from_shell_pair(&build_pair(
                &ShellView::from_shell(s),
                &ShellView::unit(s.center),
                0,
                usize::MAX,
            ))
        })
        .collect();
    let mut v = vec![0.0; n * n];
    // rows of shell s: functions ao_start .. ao_start + nfunc
    let mut slabs: Vec<(&Pair3c, &mut [f64])> = Vec::with_capacity(pairs.len());
    let mut rest: &mut [f64] = &mut v;
    for (s, pr) in aux.shells.iter().zip(&pairs) {
        let (a, b) = rest.split_at_mut(s.nfunc() * n);
        slabs.push((pr, a));
        rest = b;
    }
    slabs
        .into_par_iter()
        .for_each_init(Work3c::new, |w, (pr, slab)| {
            pair_block_sel(pr, &chunks, slab, n, thresh, kernel, w);
        });
    // symmetrize (exact up to rounding)
    for i in 0..n {
        for j in 0..i {
            let s = 0.5 * (v[i * n + j] + v[j * n + i]);
            v[i * n + j] = s;
            v[j * n + i] = s;
        }
    }
    v
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::basis::BasisDef;
    use crate::pairs::{PairOptions, PairSet};

    fn water() -> (Vec<u32>, Vec<[f64; 3]>) {
        (
            vec![8, 1, 1],
            vec![
                [0.0, 0.0, 0.2214],
                [0.0, 1.4309, -0.8857],
                [0.3, -1.4309, -0.8857],
            ],
        )
    }

    /// The primitive Schwarz bounds of `Pair3c::from_shell_pair` against the dense loop they were
    /// first computed with, bit for bit (def2-QZVP: g functions, high-degree Hermite sets).
    #[test]
    fn qprim_matches_the_dense_loop_bitwise() {
        let (z, c) = water();
        let b = BasisSet::new(&BasisDef::load("def2-qzvp", None).unwrap(), &z, &c, true).unwrap();
        let pl = PairSet::build(&b, PairOptions::default());
        let tuv = herm_tuv();
        for i in 0..pl.len() {
            let sp = pl.pair(i);
            if sp.nprim == 0 {
                continue;
            }
            let p3 = Pair3c::from_shell_pair(sp);
            let (lab, nab) = (sp.lab(), sp.nab());
            let nh = nherm(lab);
            let mut fm = vec![0.0; boys::MAX_M + 1];
            let mut r = vec![0.0; nherm(2 * lab)];
            let mut rw = vec![0.0; 2 * nherm(2 * lab)];
            for ip in 0..sp.nprim {
                let dst = &p3.et[ip * nab * nh..(ip + 1) * nab * nh];
                let p = sp.p[ip];
                boys::boys(2 * lab, 0.0, &mut fm);
                let pref = ERI_PREF / (p * p * (2.0 * p).sqrt());
                crate::core::hermite::r_tensor(
                    2 * lab,
                    0.5 * p,
                    [0.0; 3],
                    &fm,
                    pref,
                    &mut r,
                    &mut rw,
                );
                let mut qmax: f64 = 0.0;
                for ab in 0..nab {
                    let mut s = 0.0;
                    for t in 0..nh {
                        let et_ = dst[ab * nh + t];
                        if et_ == 0.0 {
                            continue;
                        }
                        for u in 0..nh {
                            let [a0, a1, a2] = tuv[t];
                            let [b0, b1, b2] = tuv[u];
                            let sign = if (b0 + b1 + b2) % 2 == 0 { 1.0 } else { -1.0 };
                            s += et_
                                * sign
                                * r[herm_idx(
                                    (a0 + b0) as usize,
                                    (a1 + b1) as usize,
                                    (a2 + b2) as usize,
                                )]
                                * dst[ab * nh + u];
                        }
                    }
                    qmax = qmax.max(s.abs());
                }
                assert_eq!(
                    p3.qprim[ip].to_bits(),
                    qmax.sqrt().to_bits(),
                    "pair {i} primitive {ip}"
                );
            }
        }
    }

    #[test]
    #[allow(clippy::needless_range_loop, reason = "indexes several arrays in step")]
    fn boys_lanes_accuracy() {
        let tab = boys_table();
        let mut f = vec![ZERO; MAX_L3C + 1];
        let mut refv = vec![0.0; MAX_L3C + 1];
        let mut worst: f64 = 0.0;
        for i in 0..4000 {
            let mut t = ZERO;
            for q in 0..W {
                t[q] = (i * W + q) as f64 * 0.0049;
            }
            for &l in &[0usize, 3, 8, MAX_L3C] {
                boys_lanes(l, &t, &mut f, tab);
                for q in 0..W {
                    boys::boys_series(l, t[q], &mut refv);
                    for m in 0..=l {
                        worst = worst.max(((f[m][q] - refv[m]) / refv[m]).abs());
                    }
                }
            }
        }
        assert!(worst < 1e-13, "worst relative error {worst:e}");
    }

    /// The combined-operator kernel equals a (k|mn) + b (k|erf|mn), and pure operators take the
    /// single-operator kernel bitwise.
    #[test]
    fn combined_operator_blocks() {
        let (z, c) = water();
        let b = BasisSet::new(&BasisDef::load("def2-tzvp", None).unwrap(), &z, &c, true).unwrap();
        let a = BasisSet::new(
            &BasisDef::load("def2-universal-jkfit", None).unwrap(),
            &z,
            &c,
            true,
        )
        .unwrap();
        let pl = PairSet::build(&b, PairOptions::default());
        let aux = Aux3c::new(&a);
        let chunks = aux.chunks(0..a.shells.len());
        let naux = a.nbf;
        let mut w = Work3c::new();
        let (ca, cb, om) = (0.157706, 0.842294, 0.3);
        let mut worst: f64 = 0.0;
        for sp in pl.pairs() {
            let p3 = Pair3c::from_shell_pair(sp);
            let n = p3.nab() * naux;
            let (mut c0, mut c1, mut cc, mut s0, mut s1) = (
                vec![0.0; n],
                vec![0.0; n],
                vec![0.0; n],
                vec![0.0; n],
                vec![0.0; n],
            );
            let comb = |coulomb, long_range, omega| Operator::Attenuated {
                coulomb,
                long_range,
                omega,
            };
            pair_block(&p3, &chunks, &mut c0, naux, 0.0, &Operator::Coulomb, &mut w);
            pair_block(
                &p3,
                &chunks,
                &mut c1,
                naux,
                0.0,
                &Operator::LongRange { omega: om },
                &mut w,
            );
            pair_block(&p3, &chunks, &mut cc, naux, 0.0, &comb(ca, cb, om), &mut w);
            // pure operators written as combinations select the single-operator kernel
            pair_block(
                &p3,
                &chunks,
                &mut s0,
                naux,
                0.0,
                &comb(1.0, 0.0, 0.0),
                &mut w,
            );
            pair_block(
                &p3,
                &chunks,
                &mut s1,
                naux,
                0.0,
                &comb(0.0, 1.0, om),
                &mut w,
            );
            assert!(
                s0 == c0 && s1 == c1,
                "pure operators must take the single-operator kernel"
            );
            for i in 0..n {
                worst = worst.max((cc[i] - (ca * c0[i] + cb * c1[i])).abs());
            }
        }
        assert!(worst < 1e-14, "combined vs a K + b K_erf: {worst:e}");
    }

    #[test]
    fn three_center_matches_reference() {
        let (z, c) = water();
        for (bname, aname) in [
            ("def2-svp", "def2-universal-jkfit"),
            ("def2-tzvp", "def2-universal-jkfit"),
        ] {
            let b = BasisSet::new(&BasisDef::load(bname, None).unwrap(), &z, &c, true).unwrap();
            let a = BasisSet::new(&BasisDef::load(aname, None).unwrap(), &z, &c, true).unwrap();
            let pl = PairSet::build(&b, PairOptions::default());
            let ap = PairSet::aux(&a, PairOptions::default());
            let refv = crate::reference::eri_3c(&b, &pl, &a, &ap); // naux x nbf^2
            let aux = Aux3c::new(&a);
            let chunks = aux.chunks(0..a.shells.len());
            let nbf = b.nbf;
            let naux = a.nbf;
            let mut w = Work3c::new();
            let mut worst: f64 = 0.0;
            for sp in pl.pairs() {
                let p3 = Pair3c::from_shell_pair(sp);
                let mut out = vec![0.0; p3.nab() * naux];
                pair_block(
                    &p3,
                    &chunks,
                    &mut out,
                    naux,
                    0.0,
                    &Operator::Coulomb,
                    &mut w,
                );
                let (a0, b0) = (b.shells[sp.sa].ao_start, b.shells[sp.sb].ao_start);
                for ia in 0..sp.na {
                    for ib in 0..sp.nb {
                        for k in 0..naux {
                            let x = out[(ia * sp.nb + ib) * naux + k];
                            let y = refv[k * nbf * nbf + (a0 + ia) * nbf + b0 + ib];
                            worst = worst.max((x - y).abs());
                        }
                    }
                }
            }
            assert!(worst < 1e-12, "{bname}: max |diff| = {worst:e}");
            let v = metric(&aux, 0.0, &Operator::Coulomb);
            let vr = crate::reference::eri_2c(&a, &ap);
            let d = v
                .iter()
                .zip(&vr)
                .map(|(x, y)| (x - y).abs())
                .fold(0.0, f64::max);
            assert!(d < 1e-11, "metric max |diff| = {d:e}");
        }
    }
}
