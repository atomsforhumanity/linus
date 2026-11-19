//! Derivative Hermite expansion (E) matrices of shell pairs, for nuclear first and second
//! derivatives in the McMurchie-Davidson / SHARK form.
//!
//! For a primitive Cartesian Gaussian G_i(x) = (x - A_x)^i exp(-a (x - A_x)^2),
//!
//!   dG_i/dA_x   = 2a G_{i+1} - i G_{i-1},
//!   d²G_i/dA_x² = 4a² G_{i+2} - 2a (2i + 1) G_i + i (i - 1) G_{i-2},
//!
//! so the derivatives of a product G_a G_b with respect to the center A of the first shell
//! expand in Hermite Gaussians Λ_t(r - P) with E-coefficients of degree up to
//! la + lb + order (the "differentiated basis function" form). Derivatives with respect to the
//! second center B follow from translation (Helgaker-Taylor): (∂/∂A + ∂/∂B) Ω = ∂/∂P Ω, and
//! ∂/∂P_k raises the Hermite index, ∂/∂P_k Λ_t = Λ_{t + 1_k}. Hence
//!
//!   E^B_k   = shift_k(E) - E^A_k,
//!   E^AB_kl = shift_l(E^A_k) - E^AA_kl,
//!   E^BB_kl = shift_k(E^B_l) - E^AB_kl,
//!
//! where shift_k maps the coefficient of Λ_t to Λ_{t+1_k}. Only E^A (3 components) and E^AA
//! (6 components) are stored; everything else is applied as an index shift in the kernels.
//!
//! A [`PairDeriv`] holds these matrices for the same retained primitive pairs, in the same
//! order, as the [`ShellPair`] of the same shells: both come from the one primitive loop
//! (`core::pair_data`), and a [`crate::PairSet`] keeps them side by side.

use crate::basis::BasisSet;
use crate::core::hermite::{herm_idx, herm_tuv, nherm, ShellPair, ShellView};
use crate::core::pair_data::{self, PRIM_PAIR_CUT};
use std::sync::OnceLock;

/// Second-derivative component order (k <= l): xx, xy, xz, yy, yz, zz.
pub const D2_COMPS: [(usize, usize); 6] = [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)];

/// Index of (k, l) in [`D2_COMPS`] for any k, l.
#[inline]
pub const fn d2_idx(k: usize, l: usize) -> usize {
    const T: [[usize; 3]; 3] = [[0, 1, 2], [1, 3, 4], [2, 4, 5]];
    T[k][l]
}

/// Derivative E-matrices of the shell pair (a, b) with respect to the center A of `a`.
#[derive(Clone, Debug)]
pub struct PairDeriv {
    /// 1: only `ea`; 2: `ea` and `eaa`.
    pub order: usize,
    pub nprim: usize,
    pub la: usize,
    pub lb: usize,
    pub na: usize,
    pub nb: usize,
    /// Per primitive: nherm(la+lb+1) x 3 x na x nb, layout \[r\]\[k\]\[mu\]\[nu\].
    pub ea: Vec<f64>,
    /// Per primitive: nherm(la+lb+2) x 6 x na x nb, layout \[r\]\[kl\]\[mu\]\[nu\] with kl in
    /// [`D2_COMPS`] order (empty for order 1).
    pub eaa: Vec<f64>,
    /// `ea` with the component index first: per primitive \[k\]\[r\]\[mu\]\[nu\].
    pub ea_t: Vec<f64>,
}

impl PairDeriv {
    #[inline]
    pub fn lab(&self) -> usize {
        self.la + self.lb
    }
    #[inline]
    pub fn nab(&self) -> usize {
        self.na * self.nb
    }
    #[inline]
    pub fn ea_prim(&self, ip: usize) -> &[f64] {
        let sz = nherm(self.lab() + 1) * 3 * self.nab();
        &self.ea[ip * sz..(ip + 1) * sz]
    }
    #[inline]
    pub fn ea_t_prim(&self, ip: usize) -> &[f64] {
        let sz = nherm(self.lab() + 1) * 3 * self.nab();
        &self.ea_t[ip * sz..(ip + 1) * sz]
    }
    #[inline]
    pub fn eaa_prim(&self, ip: usize) -> &[f64] {
        let sz = nherm(self.lab() + 2) * 6 * self.nab();
        &self.eaa[ip * sz..(ip + 1) * sz]
    }
}

/// The pair (a, b) of two (ordered) shells of a basis with its derivative expansions up to
/// `order` (1 or 2), from the one primitive loop (`core::pair_data::build`).
pub fn pair_with_deriv(
    basis: &BasisSet,
    i: usize,
    j: usize,
    order: usize,
) -> (ShellPair, PairDeriv) {
    assert!(order == 1 || order == 2, "derivative order must be 1 or 2");
    let a = ShellView::from_shell(&basis.shells[i]);
    let b = ShellView::from_shell(&basis.shells[j]);
    let (sp, pd) = pair_data::build(&a, &b, i, j, order, PRIM_PAIR_CUT);
    (sp, pd.expect("derivative order >= 1"))
}

/// Table of Hermite index shifts: shift1()\[r\]\[k\] = index of r + 1_k.
pub fn shift1() -> &'static [[u32; 3]] {
    static T: OnceLock<Vec<[u32; 3]>> = OnceLock::new();
    T.get_or_init(|| {
        const L: usize = 44;
        herm_tuv()[..nherm(L)]
            .iter()
            .map(|t| {
                let (x, y, z) = (t[0] as usize, t[1] as usize, t[2] as usize);
                [
                    herm_idx(x + 1, y, z) as u32,
                    herm_idx(x, y + 1, z) as u32,
                    herm_idx(x, y, z + 1) as u32,
                ]
            })
            .collect()
    })
}

/// Gather table for W\[r\]\[s\] = (-1)^{|s|} R[r + s] with r in Herm(lr), s in Herm(ls) and
/// |r| + |s| <= lmax = lr + ls - ex; out-of-range entries point to `sentinel` = nherm(lmax),
/// where the R buffer must hold a zero.
pub struct WTable {
    pub(crate) nr: usize,
    pub(crate) ns: usize,
    pub(crate) lmax: usize,
    /// Every index of `idx` is at most `sentinel` (the gathers rely on it; built by `wtable`).
    pub(crate) sentinel: usize,
    pub(crate) idx: Vec<u32>,
    pub(crate) sign: Vec<f64>,
}

impl WTable {
    /// w[r * ns + s] = sign\[s\] * r_ext\[idx\]; `r_ext` has r_ext\[sentinel\] == 0.
    #[inline]
    pub fn gather(&self, r_ext: &[f64], w: &mut [f64]) {
        let ns = self.ns;
        assert!(r_ext.len() > self.sentinel && w.len() >= self.nr * ns);
        for (wrow, irow) in w[..self.nr * ns]
            .chunks_exact_mut(ns)
            .zip(self.idx.chunks_exact(ns))
        {
            for ((o, &ix), &sg) in wrow.iter_mut().zip(irow).zip(&self.sign) {
                // SAFETY: every index is <= sentinel < r_ext.len() (built in wtable).
                *o = sg * unsafe { *r_ext.get_unchecked(ix as usize) };
            }
        }
    }
}

impl WTable {
    /// Gather W for batch member `q` from a structure-of-arrays R (r[i * nq + q], see
    /// [`r_tensor_batch`]): rows < `nfull` get all columns, later rows only the first
    /// `nshort` (the corner beyond is left untouched); all gathered entries must be valid.
    #[inline]
    pub fn gather_soa(
        &self,
        r: &[f64],
        nq: usize,
        q: usize,
        w: &mut [f64],
        nfull: usize,
        nshort: usize,
    ) {
        let ns = self.ns;
        assert!(q < nq && w.len() >= self.nr * ns && nshort <= ns && r.len() >= self.sentinel * nq);
        for row in 0..self.nr {
            let ncol = if row < nfull { ns } else { nshort };
            let irow = &self.idx[row * ns..row * ns + ncol];
            let wrow = &mut w[row * ns..row * ns + ncol];
            for ((o, &ix), &sg) in wrow.iter_mut().zip(irow).zip(&self.sign) {
                debug_assert!((ix as usize) < self.sentinel);
                // SAFETY: gathered entries are valid (index < sentinel = nherm(lmax)) and
                // r.len() >= sentinel * nq was checked above.
                *o = sg * unsafe { *r.get_unchecked(ix as usize * nq + q) };
            }
        }
    }
}

/// Recursion table for the Hermite Coulomb integrals: for each Hermite index i >= 1 in
/// Herm order, the direction d of the last raised index and the indices of i - 1_d and
/// i - 2_d (or 0 with coefficient 0) so that R^{(n)}_i = X_d R^{(n+1)}_{i-1_d}
/// + (t_d - 1) R^{(n+1)}_{i-2_d}.
struct RTab {
    dir: Vec<u8>,
    i1: Vec<u32>,
    i2: Vec<u32>,
    c: Vec<f64>,
}

const MAX_RTAB_L: usize = 44;

#[allow(clippy::needless_range_loop, reason = "indexes several arrays in step")]
fn rtab() -> &'static RTab {
    static T: OnceLock<RTab> = OnceLock::new();
    T.get_or_init(|| {
        let n = nherm(MAX_RTAB_L);
        let tuv = herm_tuv();
        let mut t = RTab {
            dir: vec![0; n],
            i1: vec![0; n],
            i2: vec![0; n],
            c: vec![0.0; n],
        };
        for i in 1..n {
            let [x, y, z] = tuv[i];
            let (x, y, z) = (x as usize, y as usize, z as usize);
            let (d, i1, i2, c) = if x > 0 {
                (
                    0,
                    herm_idx(x - 1, y, z),
                    if x > 1 { herm_idx(x - 2, y, z) } else { 0 },
                    (x as f64 - 1.0).max(0.0),
                )
            } else if y > 0 {
                (
                    1,
                    herm_idx(x, y - 1, z),
                    if y > 1 { herm_idx(x, y - 2, z) } else { 0 },
                    (y as f64 - 1.0).max(0.0),
                )
            } else {
                (
                    2,
                    herm_idx(x, y, z - 1),
                    if z > 1 { herm_idx(x, y, z - 2) } else { 0 },
                    (z as f64 - 1.0).max(0.0),
                )
            };
            t.dir[i] = d;
            t.i1[i] = i1 as u32;
            t.i2[i] = i2 as u32;
            t.c[i] = c;
        }
        t
    })
}

linus_num::multiversion!(
    /// R tensors for a batch of nq primitive pairs at once (structure of arrays, the recursion
    /// vectorizes over the batch): out[i * nq + q] = pref\[q\] R_i(alpha\[q\], pq\[q\]) for i in
    /// Herm(l); fm[q * (l + 1) + m] = F_m(T_q). `work` needs (2 nherm(l) + l + 4) nq entries.
    pub fn r_tensor_batch(l: usize, nq: usize, alpha: &[f64], pq: &[[f64; 3]], fm: &[f64], pref: &[f64], out: &mut [f64], work: &mut [f64]) -> () = r_tensor_batch_body);

#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn r_tensor_batch_body(
    l: usize,
    nq: usize,
    alpha: &[f64],
    pq: &[[f64; 3]],
    fm: &[f64],
    pref: &[f64],
    out: &mut [f64],
    work: &mut [f64],
) {
    let nh = nherm(l);
    assert!(
        l <= MAX_RTAB_L
            && alpha.len() >= nq
            && pq.len() >= nq
            && pref.len() >= nq
            && fm.len() >= nq * (l + 1)
    );
    assert!(out.len() >= nh * nq && work.len() >= (2 * nh + l + 4) * nq);
    let (work, extra) = work.split_at_mut(2 * nh * nq);
    let (rn, xs) = extra.split_at_mut((l + 1) * nq);
    // rn[n][q]

    for q in 0..nq {
        let m2a = -2.0 * alpha[q];
        let mut f = pref[q];
        for n in 0..=l {
            rn[n * nq + q] = f * fm[q * (l + 1) + n];
            f *= m2a;
        }
    }
    if l == 0 {
        out[..nq].copy_from_slice(&rn[..nq]);
        return;
    }

    for q in 0..nq {
        for d in 0..3 {
            xs[d * nq + q] = pq[q][d];
        }
    }
    let tab = rtab();
    let (a, b) = work[..2 * nh * nq].split_at_mut(nh * nq);
    let mut prev: &mut [f64] = a;
    let mut cur: &mut [f64] = b;
    prev[..nq].copy_from_slice(&rn[l * nq..(l + 1) * nq]);
    for n in (0..l).rev() {
        let target: &mut [f64] = if n == 0 { &mut *out } else { &mut *cur };
        target[..nq].copy_from_slice(&rn[n * nq..(n + 1) * nq]);
        let cnt = nherm(l - n);
        for i in 1..cnt {
            let d = tab.dir[i] as usize;
            let (i1, i2, c) = (tab.i1[i] as usize, tab.i2[i] as usize, tab.c[i]);
            let x = &xs[d * nq..(d + 1) * nq];
            let p1 = &prev[i1 * nq..(i1 + 1) * nq];
            let p2 = &prev[i2 * nq..(i2 + 1) * nq];
            let tg = &mut target[i * nq..(i + 1) * nq];
            for q in 0..nq {
                tg[q] = x[q] * p1[q] + c * p2[q];
            }
        }
        if n > 0 {
            std::mem::swap(&mut prev, &mut cur);
        }
    }
}

/// Same as [`crate::core::hermite::r_tensor`] (R_tuv(alpha, PQ) scaled by `pref`, Hermite order,
/// t+u+v <= l) with a table-driven recursion; `work` needs 2 nherm(l) entries.
#[inline]
pub fn r_tensor_fast(
    l: usize,
    alpha: f64,
    pq: [f64; 3],
    fm: &[f64],
    pref: f64,
    out: &mut [f64],
    work: &mut [f64],
) {
    let m2a = -2.0 * alpha;
    let mut rn = [0.0f64; crate::core::boys::MAX_M + 1];
    let mut f = pref;
    for n in 0..=l {
        rn[n] = f * fm[n];
        f *= m2a;
    }
    if l == 0 {
        out[0] = rn[0];
        return;
    }
    assert!(l <= MAX_RTAB_L);
    let tab = rtab();
    let nh = nherm(l);
    let (a, b) = work[..2 * nh].split_at_mut(nh);
    let mut prev: &mut [f64] = a;
    let mut cur: &mut [f64] = b;
    prev[0] = rn[l];
    for n in (0..l).rev() {
        let target: &mut [f64] = if n == 0 { &mut *out } else { &mut *cur };
        target[0] = rn[n];
        let cnt = nherm(l - n);
        let (dir, i1, i2, c) = (
            &tab.dir[..cnt],
            &tab.i1[..cnt],
            &tab.i2[..cnt],
            &tab.c[..cnt],
        );
        let tg = &mut target[..cnt];
        for i in 1..cnt {
            // SAFETY: i1, i2 < i < cnt <= prev.len() (Herm order), dir < 3.
            unsafe {
                let d = *dir.get_unchecked(i) as usize;
                *tg.get_unchecked_mut(i) = *pq.get_unchecked(d)
                    * *prev.get_unchecked(*i1.get_unchecked(i) as usize)
                    + *c.get_unchecked(i) * *prev.get_unchecked(*i2.get_unchecked(i) as usize);
            }
        }
        if n > 0 {
            std::mem::swap(&mut prev, &mut cur);
        }
    }
}

const MAXW: usize = 30;

pub fn wtable(lr: usize, ls: usize, ex: usize) -> &'static WTable {
    static TABLES: OnceLock<Vec<OnceLock<WTable>>> = OnceLock::new();
    assert!(lr <= MAXW && ls <= MAXW && ex <= 2);
    let tabs = TABLES.get_or_init(|| {
        (0..(MAXW + 1) * (MAXW + 1) * 3)
            .map(|_| OnceLock::new())
            .collect()
    });
    tabs[(lr * (MAXW + 1) + ls) * 3 + ex].get_or_init(|| {
        let tuv = herm_tuv();
        let (nr, ns) = (nherm(lr), nherm(ls));
        let lmax = (lr + ls).saturating_sub(ex);
        let sentinel = nherm(lmax);
        let mut idx = Vec::with_capacity(nr * ns);
        for r in &tuv[..nr] {
            for s in &tuv[..ns] {
                let (t, u, v) = (
                    (r[0] + s[0]) as usize,
                    (r[1] + s[1]) as usize,
                    (r[2] + s[2]) as usize,
                );
                idx.push(if t + u + v <= lmax {
                    herm_idx(t, u, v) as u32
                } else {
                    sentinel as u32
                });
            }
        }
        let sign = tuv[..ns]
            .iter()
            .map(|s| {
                if (s[0] + s[1] + s[2]) % 2 == 0 {
                    1.0
                } else {
                    -1.0
                }
            })
            .collect();
        WTable {
            nr,
            ns,
            lmax,
            sentinel,
            idx,
            sign,
        }
    })
}

fn fake_pair(base: &ShellPair, extra_l: usize, ncomp: usize, e: Vec<f64>) -> ShellPair {
    let emax = e.iter().fold(0.0f64, |m, x| m.max(x.abs()));
    ShellPair {
        sa: base.sa,
        sb: base.sb,
        la: base.la + extra_l,
        lb: base.lb,
        na: ncomp * base.na,
        nb: base.nb,
        nprim: base.nprim,
        p: base.p.clone(),
        pc: base.pc.clone(),
        e,
        emax,
    }
}

/// A pseudo pair whose first-shell functions run over (k, mu): any integral over it is the
/// derivative d/dA_k of the integral over (a, b). For reference evaluations and tests.
pub fn deriv_pair_a(base: &ShellPair, d: &PairDeriv) -> ShellPair {
    fake_pair(base, 1, 3, d.ea.clone())
}

/// E^B_k = shift_k(E) - E^A_k per primitive, layout \[r\]\[k\]\[mu\]\[nu\] over Herm(lab+1).
pub fn eb_prim(base: &ShellPair, d: &PairDeriv, ip: usize, out: &mut [f64]) {
    let lab = base.lab();
    let nab = base.nab();
    let nh1 = nherm(lab + 1);
    let ea = d.ea_prim(ip);
    for (o, x) in out[..nh1 * 3 * nab].iter_mut().zip(ea) {
        *o = -x;
    }
    let e = base.e_prim(ip);
    let sh = shift1();
    for r in 0..nherm(lab) {
        for k in 0..3 {
            let dst = sh[r][k] as usize;
            let o = &mut out[(dst * 3 + k) * nab..(dst * 3 + k + 1) * nab];
            for (x, y) in o.iter_mut().zip(&e[r * nab..(r + 1) * nab]) {
                *x += y;
            }
        }
    }
}

/// Pseudo pair for d/dB_k (see [`deriv_pair_a`]).
pub fn deriv_pair_b(base: &ShellPair, d: &PairDeriv) -> ShellPair {
    let nab = base.nab();
    let sz = nherm(base.lab() + 1) * 3 * nab;
    let mut e = vec![0.0; base.nprim * sz];
    for ip in 0..base.nprim {
        eb_prim(base, d, ip, &mut e[ip * sz..(ip + 1) * sz]);
    }
    fake_pair(base, 1, 3, e)
}

/// Which second derivative of a pair.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum D2Kind {
    AA,
    AB,
    BB,
}

/// Pseudo pair for second derivatives d²/dX_k dY_l (X, Y in {A, B} by `kind`), with the
/// first-shell functions running over (k, l, mu) (9 components, k-major).
pub fn deriv2_pair(base: &ShellPair, d: &PairDeriv, kind: D2Kind) -> ShellPair {
    assert!(d.order >= 2);
    let lab = base.lab();
    let nab = base.nab();
    let (nh1, nh2) = (nherm(lab + 1), nherm(lab + 2));
    let sz = nh2 * 9 * nab;
    let sh = shift1();
    let mut out = vec![0.0; base.nprim * sz];
    let mut eb = vec![0.0; nh1 * 3 * nab];
    for ip in 0..base.nprim {
        let o = &mut out[ip * sz..(ip + 1) * sz];
        let eaa = d.eaa_prim(ip);
        let ea = d.ea_prim(ip);
        // AA
        let mut aa = vec![0.0; sz];
        for r in 0..nh2 {
            for k in 0..3 {
                for l in 0..3 {
                    let src = &eaa[(r * 6 + d2_idx(k, l)) * nab..(r * 6 + d2_idx(k, l) + 1) * nab];
                    aa[(r * 9 + k * 3 + l) * nab..(r * 9 + k * 3 + l + 1) * nab]
                        .copy_from_slice(src);
                }
            }
        }
        // AB_kl = shift_l(EA_k) - AA_kl
        let mut abm = vec![0.0; sz];
        for x in 0..sz {
            abm[x] = -aa[x];
        }
        for r in 0..nh1 {
            for k in 0..3 {
                for l in 0..3 {
                    let dst = sh[r][l] as usize;
                    for x in 0..nab {
                        abm[(dst * 9 + k * 3 + l) * nab + x] += ea[(r * 3 + k) * nab + x];
                    }
                }
            }
        }
        match kind {
            D2Kind::AA => o.copy_from_slice(&aa),
            D2Kind::AB => o.copy_from_slice(&abm),
            D2Kind::BB => {
                // BB_kl = shift_k(EB_l) - AB_kl
                eb_prim(base, d, ip, &mut eb);
                for x in 0..sz {
                    o[x] = -abm[x];
                }
                for r in 0..nh1 {
                    for k in 0..3 {
                        for l in 0..3 {
                            let dst = sh[r][k] as usize;
                            for x in 0..nab {
                                o[(dst * 9 + k * 3 + l) * nab + x] += eb[(r * 3 + l) * nab + x];
                            }
                        }
                    }
                }
            }
        }
    }
    fake_pair(base, 2, 9, out)
}
