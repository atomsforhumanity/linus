//! Blocks of first-derivative three-center integrals (∂_k a b|P), the derivative with respect
//! to the center of the AO function a, for every ordered AO function pair and a block of
//! auxiliary functions, on the lane kernels of [`crate::eri3c`] (W auxiliary primitives per
//! SIMD batch). The integral layer of the aux-blocked density-fitted Hessian.
//!
//! For an orbital shell pair (A, B) with Hermite expansions E (of ab), E_A (of ∂ab/∂A, degree
//! l_ab + 1, [`crate::core::deriv::PairDeriv::ea`]) and E_B = shift(E) - E_A (translational
//! invariance of the product: (∂_A + ∂_B) ab = sum_t E_t Λ_{t+1_k}), one Hermite potential per
//! primitive pair and lane batch at degree l_aux + l_ab + 1,
//!
//!   H\[k\]\[t'\] = sum_c S_k\[c\] R_{c+t'},   |t'| <= l_ab + 1,
//!
//! gives both (∂a b|P) = sum_t' H[t'] E_A[t']\[ab\] and (a ∂b|P) = sum_t' H[t'] E_B[t']\[ab\], i.e.
//! the blocks of both orderings (a, b) and (b, a) of the pair. The auxiliary-center derivative
//! follows from translational invariance, (ab|∂P) = -(∂a b|P) - (a ∂b|P).
//!
//! Screening: per primitive pair, the Schwarz bound of the derivative products,
//! max_k max(|(∂_A ab|∂_A ab)|, |(∂_B ab|∂_B ab)|)^1/2, times the largest lane bound.

use crate::core::hermite::{herm_idx, herm_tuv, nherm, ShellPair};
use crate::engine::schedule;
use crate::eri3c::deriv_plan::{BlockCtx, Eri3Thread};
use crate::eri3c::{
    acc_contract, boys_table, h_contract, htab, prim_hermite_sel, r_prog, AuxChunk, Lanes, Work3c,
    ZERO,
};
use crate::eri4c::ERI_PREF;
use crate::operator::Operator;
use crate::{core::boys, core::deriv::PairDeriv};
use linus_num::simd::Isa;

/// An orbital shell pair prepared for derivative 3-center blocks.
#[derive(Clone, Debug)]
pub(crate) struct Pair3cD {
    pub(crate) sa: usize,
    pub(crate) sb: usize,
    pub(crate) na: usize,
    pub(crate) nb: usize,
    pub(crate) lab: usize,
    p: Vec<f64>,
    pc: Vec<[f64; 3]>,
    /// per primitive [k][ab][t'] (t' < nherm(lab + 1)): ∂/∂A of the product
    eat: Vec<f64>,
    /// per primitive [k][ab][t']: ∂/∂B (empty for a diagonal shell pair)
    ebt: Vec<f64>,
    /// derivative Schwarz bound per primitive pair
    qprim: Vec<f64>,
}

impl Pair3cD {
    pub(crate) fn nab(&self) -> usize {
        self.na * self.nb
    }

    /// From a shell pair and its first-derivative data (same primitive order: a pair of a
    /// [`crate::PairSet`] and its derivative expansion).
    pub(crate) fn new(sp: &ShellPair, pd: &PairDeriv) -> Self {
        let lab = sp.la + sp.lb;
        let (nh, nh1) = (nherm(lab), nherm(lab + 1));
        let nab = sp.na * sp.nb;
        let diag = sp.sa == sp.sb;
        let per = 3 * nab * nh1;
        let mut eat = vec![0.0; sp.nprim * per];
        let mut ebt = if diag {
            Vec::new()
        } else {
            vec![0.0; sp.nprim * per]
        };
        let sh = crate::core::deriv::shift1();
        for ip in 0..sp.nprim {
            let ea = pd.ea_prim(ip);
            let e = sp.e_prim(ip);
            let da = &mut eat[ip * per..(ip + 1) * per];
            for k in 0..3 {
                for ab in 0..nab {
                    for t in 0..nh1 {
                        da[(k * nab + ab) * nh1 + t] = ea[(t * 3 + k) * nab + ab];
                    }
                }
            }
            if !diag {
                let db = &mut ebt[ip * per..(ip + 1) * per];
                for k in 0..3 {
                    for ab in 0..nab {
                        let row = &mut db[(k * nab + ab) * nh1..(k * nab + ab + 1) * nh1];
                        for t in 0..nh1 {
                            row[t] = -ea[(t * 3 + k) * nab + ab];
                        }
                        for t in 0..nh {
                            row[sh[t][k] as usize] += e[t * nab + ab];
                        }
                    }
                }
            }
        }
        // primitive bounds: (x|x)^1/2 for the derivative expansions x (R at PQ = 0, alpha = p/2)
        let l2 = 2 * (lab + 1);
        let tuv = herm_tuv();
        let mut fm = vec![0.0; boys::MAX_M + 1];
        let mut r = vec![0.0; nherm(l2)];
        let mut rw = vec![0.0; 2 * nherm(l2)];
        let mut qprim = vec![0.0; sp.nprim];
        for ip in 0..sp.nprim {
            let p = sp.p[ip];
            boys::boys(l2, 0.0, &mut fm);
            let pref = ERI_PREF / (p * p * (2.0 * p).sqrt());
            crate::core::hermite::r_tensor(l2, 0.5 * p, [0.0; 3], &fm, pref, &mut r, &mut rw);
            let mut qmax: f64 = 0.0;
            for src in [&eat, &ebt] {
                if src.is_empty() {
                    continue;
                }
                let x = &src[ip * per..(ip + 1) * per];
                for kab in 0..3 * nab {
                    let v = &x[kab * nh1..(kab + 1) * nh1];
                    let mut s = 0.0;
                    for t in 0..nh1 {
                        if v[t] == 0.0 {
                            continue;
                        }
                        let [a0, a1, a2] = tuv[t];
                        for u in 0..nh1 {
                            let [b0, b1, b2] = tuv[u];
                            let sign = if (b0 + b1 + b2) % 2 == 0 { 1.0 } else { -1.0 };
                            s += v[t]
                                * sign
                                * r[herm_idx(
                                    (a0 + b0) as usize,
                                    (a1 + b1) as usize,
                                    (a2 + b2) as usize,
                                )]
                                * v[u];
                        }
                    }
                    qmax = qmax.max(s.abs());
                }
            }
            qprim[ip] = qmax.sqrt();
        }
        Pair3cD {
            sa: sp.sa,
            sb: sp.sb,
            na: sp.na,
            nb: sp.nb,
            lab,
            p: sp.p.clone(),
            pc: sp.pc.clone(),
            eat,
            ebt,
            qprim,
        }
    }

    /// Largest primitive derivative bound.
    pub(crate) fn qmax(&self) -> f64 {
        self.qprim.iter().cloned().fold(0.0, f64::max)
    }
}

/// Raw output pointer shared by tasks that write disjoint regions.
#[derive(Clone, Copy)]
struct OutPtr(*mut f64);
unsafe impl Send for OutPtr {}
unsafe impl Sync for OutPtr {}

/// Where a pair's blocks go: out[((k * nbf + row) * nq + P) * nbf + col].
#[derive(Clone, Copy)]
struct Dest {
    ptr: OutPtr,
    nbf: usize,
    nq: usize,
}

impl Dest {
    /// Zero the rows `r0..r0+nr` x columns `c0..c0+nc` of all P and k.
    unsafe fn zero(&self, r0: usize, nr: usize, c0: usize, nc: usize) {
        for k in 0..3 {
            for r in r0..r0 + nr {
                for p in 0..self.nq {
                    let o = ((k * self.nbf + r) * self.nq + p) * self.nbf + c0;
                    std::slice::from_raw_parts_mut(self.ptr.0.add(o), nc).fill(0.0);
                }
            }
        }
    }
    #[inline(always)]
    unsafe fn add(&self, k: usize, row: usize, p: usize, col: usize, v: f64) {
        *self
            .ptr
            .0
            .add(((k * self.nbf + row) * self.nq + p) * self.nbf + col) += v;
    }
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::needless_range_loop, reason = "indexes several arrays in step")]
fn pair_dblock_impl(
    pair: &Pair3cD,
    chunks: &[AuxChunk],
    a0: usize,
    b0: usize,
    do_a: bool,
    do_b: bool,
    dest: Dest,
    thresh: f64,
    op: Operator,
    w: &mut Work3c,
    acc: &mut Vec<Lanes>,
) {
    let tab = boys_table();
    let prog = r_prog();
    let lab = pair.lab;
    let nh1 = nherm(lab + 1);
    let nab = pair.nab();
    let per = 3 * nab * nh1;
    let nprim = pair.p.len();
    let qpmax = pair.qmax();
    let Work3c {
        fm, rn, r, rw, h, ..
    } = w;
    for ch in chunks {
        if qpmax * ch.q < thresh {
            continue;
        }
        let l = ch.l;
        let lt = l + lab + 1;
        let ht = htab(l, lab + 1);
        let nk = if l <= 1 {
            crate::basis::ncart(l)
        } else {
            2 * l + 1
        };
        let blk = nk * nab;
        if acc.len() < 6 * blk {
            acc.resize(6 * blk, ZERO);
        }
        let acc = &mut acc[..6 * blk];
        acc.fill(ZERO);
        if h.len() < nk * nh1 {
            h.resize(nk * nh1, ZERO);
        }
        let h = &mut h[..nk * nh1];
        let mut any = false;
        for ip in 0..nprim {
            if pair.qprim[ip] * ch.q < thresh {
                continue;
            }
            any = true;
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
            let hv: &[Lanes] = if l == 0 {
                &r[..nh1]
            } else {
                h_contract(h, r, ht, nh1);
                h
            };
            for k in 0..3 {
                if do_a {
                    let e = &pair.eat[ip * per + k * nab * nh1..ip * per + (k + 1) * nab * nh1];
                    acc_contract(&mut acc[k * blk..(k + 1) * blk], hv, e, nk, nab, nh1);
                }
                if do_b {
                    let e = &pair.ebt[ip * per + k * nab * nh1..ip * per + (k + 1) * nab * nh1];
                    acc_contract(&mut acc[(3 + k) * blk..(4 + k) * blk], hv, e, nk, nab, nh1);
                }
            }
        }
        if !any {
            continue;
        }
        let (na, nb) = (pair.na, pair.nb);
        for q in 0..crate::eri3c::W {
            let col = ch.col[q];
            if col == u32::MAX {
                continue;
            }
            let col = col as usize;
            for k in 0..3 {
                for kk in 0..nk {
                    let pidx = col + kk;
                    for ia in 0..na {
                        for ib in 0..nb {
                            let ab = ia * nb + ib;
                            // SAFETY: rows a0..a0+na (A part) and b0..b0+nb (B part) belong to this
                            // pair only (see `deriv_block`)
                            unsafe {
                                if do_a {
                                    dest.add(
                                        k,
                                        a0 + ia,
                                        pidx,
                                        b0 + ib,
                                        acc[k * blk + kk * nab + ab][q],
                                    );
                                }
                                if do_b {
                                    dest.add(
                                        k,
                                        b0 + ib,
                                        pidx,
                                        a0 + ia,
                                        acc[(3 + k) * blk + kk * nab + ab][q],
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512vl,avx512dq,avx2,fma")]
#[allow(clippy::too_many_arguments)]
unsafe fn pair_dblock_avx512(
    pair: &Pair3cD,
    chunks: &[AuxChunk],
    a0: usize,
    b0: usize,
    do_a: bool,
    do_b: bool,
    dest: Dest,
    thresh: f64,
    op: Operator,
    w: &mut Work3c,
    acc: &mut Vec<Lanes>,
) {
    pair_dblock_impl(pair, chunks, a0, b0, do_a, do_b, dest, thresh, op, w, acc)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[allow(clippy::too_many_arguments)]
unsafe fn pair_dblock_avx2(
    pair: &Pair3cD,
    chunks: &[AuxChunk],
    a0: usize,
    b0: usize,
    do_a: bool,
    do_b: bool,
    dest: Dest,
    thresh: f64,
    op: Operator,
    w: &mut Work3c,
    acc: &mut Vec<Lanes>,
) {
    pair_dblock_impl(pair, chunks, a0, b0, do_a, do_b, dest, thresh, op, w, acc)
}

/// Derivative integrals of the block of auxiliary shells `cx` (nq functions):
/// out[((k * nbf + a) * nq + P) * nbf + b] = (∂_k a b|op|P) for the AO functions a in `rows`
/// (whole shells) and all b, P relative to the first function of the block. Function pairs
/// of shell pairs not in `pairs` are never written (they stay as they were: zero the buffer
/// once). `cx.thresh`: primitive screening (the Coulomb bounds stay valid, scaled by
/// [`Operator::schwarz_factor`]). The pairs (tasks `order`) go to the threads dynamically, each
/// writing its own region of `out` with its thread's scratch.
pub(crate) fn deriv_block(
    pairs: &[Pair3cD],
    cx: &BlockCtx<'_>,
    rows: std::ops::Range<usize>,
    out: &mut [f64],
    threads: &mut [Eri3Thread],
    order: &[u32],
) {
    let (nbf, nq) = (cx.nbf, cx.nq);
    assert!(out.len() >= 3 * nbf * nq * nbf);
    let thresh = cx.op.screen_threshold(cx.thresh);
    let sel = cx.op.kernel();
    let (chunks, ao, isa) = (&cx.chunks[..], cx.ao, cx.isa);
    let dest = Dest {
        ptr: OutPtr(out.as_mut_ptr()),
        nbf,
        nq,
    };
    schedule::exact(order, threads, |t, tasks| {
        let (w, acc) = (&mut t.work, &mut t.lanes);
        for pi in tasks {
            let pair = &pairs[pi as usize];
            let (a0, na) = ao[pair.sa];
            let (b0, nb) = ao[pair.sb];
            let do_a = rows.contains(&a0);
            let do_b = pair.sa != pair.sb && rows.contains(&b0);
            if !do_a && !do_b {
                continue;
            }
            // SAFETY: each (row shell, column shell) region belongs to exactly one pair of the
            // list (pairs are unordered and distinct), so tasks write disjoint memory
            unsafe {
                if do_a {
                    dest.zero(a0, na, b0, nb);
                }
                if do_b {
                    dest.zero(b0, nb, a0, na);
                }
                match isa {
                    #[cfg(target_arch = "x86_64")]
                    Isa::Avx512 => pair_dblock_avx512(
                        pair, chunks, a0, b0, do_a, do_b, dest, thresh, sel, w, acc,
                    ),
                    #[cfg(target_arch = "x86_64")]
                    Isa::Avx2 => pair_dblock_avx2(
                        pair, chunks, a0, b0, do_a, do_b, dest, thresh, sel, w, acc,
                    ),
                    _ => pair_dblock_impl(
                        pair, chunks, a0, b0, do_a, do_b, dest, thresh, sel, w, acc,
                    ),
                }
            }
        }
    });
}

// ------------------------------------------------------------------------------------------
// Second derivatives contracted with a 3-index density (the explicit 3-center term of a
// density-fitted Hessian), on the same lane kernels
// ------------------------------------------------------------------------------------------

/// An orbital shell pair prepared for Gamma-contracted second-derivative 3-center integrals:
/// E (degree l_ab), E_A (l_ab + 1) and E_AA (l_ab + 2, the six components of
/// [`crate::core::deriv::D2_COMPS`]), each transposed to \[comp\]\[ab\]\[t\]. With the Hermite shift S_k
/// ((∂_A + ∂_B)_k of the product) and ∂_C = -S on the integral (translational invariance),
/// every second derivative of (ab|P) is a combination of
///   X1_kl = E_AA,kl,  X2_kl = S_l E_A,k,  X3_kl = S_k S_l E:
///   AA = X1, AB_kl = X2_kl - X1_kl, BB_kl = X3_kl - X2_lk - X2_kl + X1_kl,
///   AC_kl = -X2_kl, BC_kl = X2_kl - X3_kl, CC_kl = X3_kl.
#[derive(Clone, Debug)]
pub(crate) struct Pair3cH {
    pub(crate) sa: usize,
    pub(crate) sb: usize,
    pub(crate) na: usize,
    pub(crate) nb: usize,
    pub(crate) lab: usize,
    p: Vec<f64>,
    pc: Vec<[f64; 3]>,
    /// per primitive [ab][t] (nherm(lab))
    et: Vec<f64>,
    /// per primitive [k][ab][t] (nherm(lab + 1))
    eat: Vec<f64>,
    /// per primitive [kl][ab][t] (nherm(lab + 2), D2_COMPS order)
    eaat: Vec<f64>,
    /// pair bound of the second derivatives (screening)
    pub(crate) q2: f64,
}

impl Pair3cH {
    /// From a shell pair and its order-2 derivative data (same primitive order); `q`, `q1`,
    /// `q2`: the pair's Schwarz factors of the integrals and their first and second
    /// derivatives ([`crate::PairSet::schwarz`], [`crate::PairSet::deriv_bounds`]), combined into one screening bound
    /// with the auxiliary basis' (`aq`, `aq1`, `aq2`: largest over shells).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        sp: &ShellPair,
        pd: &PairDeriv,
        q: f64,
        q1: f64,
        q2: f64,
        aq: f64,
        aq1: f64,
        aq2: f64,
    ) -> Self {
        assert!(pd.order >= 2, "Pair3cH needs second-derivative pair data");
        let lab = sp.la + sp.lb;
        let (nh, nh1, nh2) = (nherm(lab), nherm(lab + 1), nherm(lab + 2));
        let nab = sp.na * sp.nb;
        let mut et = vec![0.0; sp.nprim * nab * nh];
        let mut eat = vec![0.0; sp.nprim * 3 * nab * nh1];
        let mut eaat = vec![0.0; sp.nprim * 6 * nab * nh2];
        for ip in 0..sp.nprim {
            let e = sp.e_prim(ip);
            let ea = pd.ea_prim(ip);
            let eaa = pd.eaa_prim(ip);
            let (d0, d1, d2) = (
                &mut et[ip * nab * nh..],
                &mut eat[ip * 3 * nab * nh1..],
                &mut eaat[ip * 6 * nab * nh2..],
            );
            for ab in 0..nab {
                for t in 0..nh {
                    d0[ab * nh + t] = e[t * nab + ab];
                }
                for k in 0..3 {
                    for t in 0..nh1 {
                        d1[(k * nab + ab) * nh1 + t] = ea[(t * 3 + k) * nab + ab];
                    }
                }
                for kl in 0..6 {
                    for t in 0..nh2 {
                        d2[(kl * nab + ab) * nh2 + t] = eaa[(t * 6 + kl) * nab + ab];
                    }
                }
            }
        }
        Pair3cH {
            sa: sp.sa,
            sb: sp.sb,
            na: sp.na,
            nb: sp.nb,
            lab,
            p: sp.p.clone(),
            pc: sp.pc.clone(),
            et,
            eat,
            eaat,
            q2: q2 * aq + 2.0 * q1 * aq1 + q * aq2,
        }
    }
}

/// Scratch of [`hess_block`] per task.
#[derive(Default)]
pub(crate) struct H3Work {
    acc: Vec<Lanes>,
    hs: Vec<Lanes>,
    hss: Vec<Lanes>,
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::needless_range_loop, reason = "indexes several arrays in step")]
fn pair_hess_impl(
    pair: &Pair3cH,
    chunks: &[AuxChunk],
    gamma: &[f64],
    nbf: usize,
    ao: &[(usize, usize)],
    op: Operator,
    w: &mut Work3c,
    hw: &mut H3Work,
    out: &mut [[f64; 21]],
    aux_fn: &[usize],
) {
    let tab = boys_table();
    let prog = r_prog();
    let lab = pair.lab;
    let (nh, nh1, nh2) = (nherm(lab), nherm(lab + 1), nherm(lab + 2));
    let nab = pair.na * pair.nb;
    let n2 = nbf * nbf;
    let sh = crate::core::deriv::shift1();
    let comps = crate::core::deriv::D2_COMPS;
    let (a0, na) = ao[pair.sa];
    let (b0, nb) = ao[pair.sb];
    let diag = pair.sa == pair.sb;
    let Work3c {
        fm, rn, r, rw, h, ..
    } = w;
    let nprim = pair.p.len();
    for ch in chunks {
        let l = ch.l;
        let lt = l + lab + 2;
        let ht = htab(l, lab + 2);
        let nk = if l <= 1 {
            crate::basis::ncart(l)
        } else {
            2 * l + 1
        };
        let blk = nk * nab;
        // acc: X1 (6), X2 (9, [k][l]), X3 (6)
        hw.acc.resize(21 * blk, ZERO);
        let acc = &mut hw.acc[..21 * blk];
        acc.fill(ZERO);
        if h.len() < nk * nh2 {
            h.resize(nk * nh2, ZERO);
        }
        hw.hs.resize(3 * nk * nh1, ZERO);
        hw.hss.resize(6 * nk * nh, ZERO);
        for ip in 0..nprim {
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
            let hv: &[Lanes] = if l == 0 {
                &r[..nh2]
            } else {
                h_contract(&mut h[..nk * nh2], r, ht, nh2);
                &h[..nk * nh2]
            };
            // shifted potentials: hs[l][kk][t] = H[kk][t + 1_l] (t < nh1), hss[kl][kk][t] = H[kk][t + 1_k + 1_l]
            for kk in 0..nk {
                let hk = &hv[kk * nh2..(kk + 1) * nh2];
                for d in 0..3 {
                    let o = &mut hw.hs[(d * nk + kk) * nh1..(d * nk + kk + 1) * nh1];
                    for t in 0..nh1 {
                        o[t] = hk[sh[t][d] as usize];
                    }
                }
                for (c, &(k, ll)) in comps.iter().enumerate() {
                    let o = &mut hw.hss[(c * nk + kk) * nh..(c * nk + kk + 1) * nh];
                    for t in 0..nh {
                        o[t] = hk[sh[sh[t][k] as usize][ll] as usize];
                    }
                }
            }
            let e = &pair.et[ip * nab * nh..(ip + 1) * nab * nh];
            let ea = &pair.eat[ip * 3 * nab * nh1..(ip + 1) * 3 * nab * nh1];
            let eaa = &pair.eaat[ip * 6 * nab * nh2..(ip + 1) * 6 * nab * nh2];
            for c in 0..6 {
                acc_contract(
                    &mut acc[c * blk..(c + 1) * blk],
                    hv,
                    &eaa[c * nab * nh2..(c + 1) * nab * nh2],
                    nk,
                    nab,
                    nh2,
                );
                acc_contract(
                    &mut acc[(15 + c) * blk..(16 + c) * blk],
                    &hw.hss[c * nk * nh..(c + 1) * nk * nh],
                    e,
                    nk,
                    nab,
                    nh,
                );
            }
            for k in 0..3 {
                for d in 0..3 {
                    acc_contract(
                        &mut acc[(6 + 3 * k + d) * blk..(7 + 3 * k + d) * blk],
                        &hw.hs[d * nk * nh1..(d + 1) * nk * nh1],
                        &ea[k * nab * nh1..(k + 1) * nab * nh1],
                        nk,
                        nab,
                        nh1,
                    );
                }
            }
        }
        // contract with Gamma_P (W = Gamma_ab + Gamma_ba off the diagonal shell pair)
        for q in 0..crate::eri3c::W {
            let col = ch.col[q];
            if col == u32::MAX {
                continue;
            }
            for kk in 0..nk {
                let prel = col as usize + kk;
                let g = &gamma[prel * n2..(prel + 1) * n2];
                let o = &mut out[aux_fn[prel]];
                let mut v = [0.0; 21];
                for ia in 0..na {
                    for ib in 0..nb {
                        let (m, nn) = (a0 + ia, b0 + ib);
                        let wv = if diag {
                            g[m * nbf + nn]
                        } else {
                            g[m * nbf + nn] + g[nn * nbf + m]
                        };
                        if wv == 0.0 {
                            continue;
                        }
                        let ab = ia * nb + ib;
                        for (c, vc) in v.iter_mut().enumerate() {
                            *vc += wv * acc[c * blk + kk * nab + ab][q];
                        }
                    }
                }
                for c in 0..21 {
                    o[c] += v[c];
                }
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512vl,avx512dq,avx2,fma")]
#[allow(clippy::too_many_arguments)]
unsafe fn pair_hess_avx512(
    pair: &Pair3cH,
    chunks: &[AuxChunk],
    gamma: &[f64],
    nbf: usize,
    ao: &[(usize, usize)],
    op: Operator,
    w: &mut Work3c,
    hw: &mut H3Work,
    out: &mut [[f64; 21]],
    aux_fn: &[usize],
) {
    pair_hess_impl(pair, chunks, gamma, nbf, ao, op, w, hw, out, aux_fn)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[allow(clippy::too_many_arguments)]
unsafe fn pair_hess_avx2(
    pair: &Pair3cH,
    chunks: &[AuxChunk],
    gamma: &[f64],
    nbf: usize,
    ao: &[(usize, usize)],
    op: Operator,
    w: &mut Work3c,
    hw: &mut H3Work,
    out: &mut [[f64; 21]],
    aux_fn: &[usize],
) {
    pair_hess_impl(pair, chunks, gamma, nbf, ao, op, w, hw, out, aux_fn)
}

/// sum_{P in the block, mn} ∂²(P|op|mn)/∂X∂Y Gamma_P\[m\]\[n\] for the block of auxiliary shells
/// `cx` (nq functions, Gamma row-major [P - first]\[m\]\[n\], symmetric in mn) into the partials
/// `parts` ((3 natom)^2 each, row-major, one per chunk of the ordered reduction, zeroed here;
/// the caller adds them in chunk order), on the threads' scratch. Pairs whose second-derivative
/// bound times max |Gamma| is below `cx.thresh` are skipped.
pub(crate) fn hess_block(
    pairs: &[Pair3cH],
    cx: &BlockCtx<'_>,
    gamma: &[f64],
    threads: &mut [Eri3Thread],
    parts: &mut [Vec<f64>],
) {
    let (nbf, nq) = (cx.nbf, cx.nq);
    let thresh = cx.op.screen_threshold(cx.thresh);
    let sel = cx.op.kernel();
    let n3 = 3 * cx.natom;
    let (chunks, ao, isa) = (&cx.chunks[..], cx.ao, cx.isa);
    let gmax = gamma[..nq * nbf * nbf]
        .iter()
        .fold(0.0f64, |a, x| a.max(x.abs()));
    for t in threads.iter_mut() {
        t.h21.clear();
        t.h21.resize(nq, [0.0; 21]);
        t.fidx.clear();
        t.fidx.extend(0..nq);
    }
    for h in parts.iter_mut() {
        h.fill(0.0);
    }
    // per pair: the 21 contracted components for every auxiliary function of the block, then
    // the 6 center blocks into the Hessian (atoms of A, B and P)
    schedule::ordered_strided(pairs.len(), threads, parts, |t, h, pi| {
        let (w, hw, out, fidx) = (&mut t.work, &mut t.h3, &mut t.h21[..], &t.fidx[..]);

        let pair = &pairs[pi];
        if pair.q2 * gmax < thresh {
            return;
        }
        out.iter_mut().for_each(|o| *o = [0.0; 21]);
        match isa {
            // SAFETY: plain computation; the target features are checked by `isa`
            #[cfg(target_arch = "x86_64")]
            Isa::Avx512 => unsafe {
                pair_hess_avx512(pair, chunks, gamma, nbf, ao, sel, w, hw, out, fidx)
            },
            // SAFETY: as above
            #[cfg(target_arch = "x86_64")]
            Isa::Avx2 => unsafe {
                pair_hess_avx2(pair, chunks, gamma, nbf, ao, sel, w, hw, out, fidx)
            },
            _ => pair_hess_impl(pair, chunks, gamma, nbf, ao, sel, w, hw, out, fidx),
        }
        let (at_a, at_b) = (cx.ao_atom[pair.sa], cx.ao_atom[pair.sb]);
        for (p, v) in out.iter().enumerate() {
            if v.iter().all(|x| *x == 0.0) {
                continue;
            }
            let at_c = cx.aux_atom[p];
            let x1 = |k: usize, l: usize| v[crate::core::deriv::d2_idx(k, l)];
            let x2 = |k: usize, l: usize| v[6 + 3 * k + l];
            let x3 = |k: usize, l: usize| v[15 + crate::core::deriv::d2_idx(k, l)];
            let mut add = |x: usize, y: usize, f: &dyn Fn(usize, usize) -> f64| {
                for k in 0..3 {
                    for l in 0..3 {
                        let val = f(k, l);
                        h[(3 * x + k) * n3 + 3 * y + l] += val;
                    }
                }
            };
            // center pairs: AA, BB, CC once; AB, AC, BC in both orders
            add(at_a, at_a, &|k, l| x1(k, l));
            add(at_b, at_b, &|k, l| {
                x3(k, l) - x2(l, k) - x2(k, l) + x1(k, l)
            });
            add(at_c, at_c, &|k, l| x3(k, l));
            add(at_a, at_b, &|k, l| x2(k, l) - x1(k, l));
            add(at_b, at_a, &|k, l| x2(l, k) - x1(l, k));
            add(at_a, at_c, &|k, l| -x2(k, l));
            add(at_c, at_a, &|k, l| -x2(l, k));
            add(at_b, at_c, &|k, l| x2(k, l) - x3(k, l));
            add(at_c, at_b, &|k, l| x2(l, k) - x3(l, k));
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::basis::BasisDef;
    use crate::core::hermite::ShellView;
    use crate::eri3c::deriv_plan::{Eri3Block, Eri3DerivOptions, Eri3DerivPlan};
    use crate::eri3c::DfThresholds;
    use crate::eri4c::jk_deriv::{eri_quartet_deriv_a, DerivAWork};
    use crate::pairs::{DerivOrder, PairOptions, PairSet};
    use crate::BasisSet;

    /// Operators of the tests: Coulomb, erf (single kernel) and a CAM combination.
    fn test_ops() -> [Operator; 3] {
        [
            Operator::Coulomb,
            Operator::LongRange { omega: 0.33 },
            Operator::Attenuated {
                coulomb: 0.19,
                long_range: 0.46,
                omega: 0.33,
            },
        ]
    }

    /// deriv_block equals the reference quartet kernel (derivative of the bra's first center)
    /// for every ordered shell pair and auxiliary shell, with rows restricted or not, for the
    /// Coulomb, erf and CAM operators.
    #[test]
    fn deriv_block_matches_quartet_kernel() {
        let z = vec![8, 1, 1, 6];
        let positions = vec![
            [0.1, -0.05, 0.12],
            [1.6, 0.98, -0.06],
            [-1.28, 1.04, 0.09],
            [0.3, -2.4, 1.1],
        ];
        let basis = BasisSet::new(
            &BasisDef::load("def2-svp", None).unwrap(),
            &z,
            &positions,
            true,
        )
        .unwrap();
        let aux = BasisSet::new(
            &BasisDef::load("def2-universal-jkfit", None).unwrap(),
            &z,
            &positions,
            true,
        )
        .unwrap();
        let n = basis.nbf;
        let pairs = PairSet::build(&basis, PairOptions::with_order(DerivOrder::One));
        let ao: Vec<(usize, usize)> = basis
            .shells
            .iter()
            .map(|s| (s.ao_start, s.nfunc()))
            .collect();
        let apairs = PairSet::aux(&aux, PairOptions::with_order(DerivOrder::One));
        let opts = Eri3DerivOptions {
            thresholds: DfThresholds {
                deriv_pair: 0.0,
                ..Default::default()
            },
        };
        let plan =
            Eri3DerivPlan::new(&basis, &pairs, &aux, &apairs, DerivOrder::One, opts).unwrap();
        let mut ws = plan.workspace(rayon::current_num_threads());
        // aux shells 3..9 as one block
        let ash = 3..9;
        let p0 = aux.shells[ash.start].ao_start;
        let nq = aux.shells[ash.end - 1].ao_start + aux.shells[ash.end - 1].nfunc() - p0;
        for (rows, op) in [
            (0..n, Operator::Coulomb),
            (
                basis.shells[4].ao_start..basis.shells[9].ao_start,
                Operator::Coulomb,
            ),
        ]
        .into_iter()
        .chain(test_ops()[1..].iter().map(|o| (0..n, *o)))
        {
            let mut out = vec![0.0; 3 * n * nq * n];
            let blk = Eri3Block {
                shells: ash.clone(),
                op,
                thresh: 1e-20,
            };
            plan.deriv_block(&blk, rows.clone(), &mut out, &mut ws);
            let views: Vec<ShellView> = basis.shells.iter().map(ShellView::from_shell).collect();
            let mut w = DerivAWork::with_op(op);
            let mut maxd: f64 = 0.0;
            for a in 0..basis.shells.len() {
                for b in 0..basis.shells.len() {
                    let (pr, pd) = crate::core::pair_data::build(
                        &views[a],
                        &views[b],
                        a,
                        b,
                        1,
                        crate::core::pair_data::PRIM_PAIR_CUT,
                    );
                    let pd = pd.unwrap();
                    let (a0, na) = ao[a];
                    let (b0, nb) = ao[b];
                    for ps in ash.clone() {
                        let ap = apairs.pair(ps);
                        let np = ap.na;
                        let pq0 = aux.shells[ps].ao_start - p0;
                        let mut buf = vec![0.0; 3 * na * nb * np];
                        eri_quartet_deriv_a(&pr, &pd, ap, &mut buf, &mut w);
                        for k in 0..3 {
                            for ia in 0..na {
                                for ib in 0..nb {
                                    for p in 0..np {
                                        let r = buf[(k * na * nb + ia * nb + ib) * np + p];
                                        let v =
                                            out[((k * n + a0 + ia) * nq + pq0 + p) * n + b0 + ib];
                                        let want = if rows.contains(&(a0 + ia)) { r } else { 0.0 };
                                        maxd = maxd.max((v - want).abs());
                                    }
                                }
                            }
                        }
                    }
                }
            }
            assert!(maxd < 1e-12, "{op:?}: max deviation {maxd:e}");
        }
    }

    /// The plan's Hessian equals eri_hess_3c (generic quartet kernel) for a random symmetric
    /// Gamma, for the Coulomb, erf and CAM operators.
    #[test]
    fn hessian_matches_eri_hess_3c() {
        use crate::reference::{eri_hess_3c, Dens3c};
        let z = vec![8, 1, 1, 6];
        let positions = vec![
            [0.1, -0.05, 0.12],
            [1.6, 0.98, -0.06],
            [-1.28, 1.04, 0.09],
            [0.3, -2.4, 1.1],
        ];
        let basis = BasisSet::new(
            &BasisDef::load("def2-svp", None).unwrap(),
            &z,
            &positions,
            true,
        )
        .unwrap();
        let aux = BasisSet::new(
            &BasisDef::load("def2-universal-jkfit", None).unwrap(),
            &z,
            &positions,
            true,
        )
        .unwrap();
        let n = basis.nbf;
        let pairs = PairSet::build(&basis, PairOptions::with_order(DerivOrder::Two));
        let apairs = PairSet::aux(&aux, PairOptions::with_order(DerivOrder::Two));
        let plan = Eri3DerivPlan::new(
            &basis,
            &pairs,
            &aux,
            &apairs,
            DerivOrder::Two,
            Eri3DerivOptions::default(),
        )
        .unwrap();
        let mut ws = plan.workspace(rayon::current_num_threads());
        let ash = 2..11;
        let p0 = aux.shells[ash.start].ao_start;
        let nq = aux.shells[ash.end - 1].ao_start + aux.shells[ash.end - 1].nfunc() - p0;
        // random symmetric Gamma rows
        let mut seed = 12345u64;
        let mut rnd = || {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((seed >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
        };
        let mut g = vec![0.0; nq * n * n];
        for p in 0..nq {
            for i in 0..n {
                for j in 0..=i {
                    let v = rnd();
                    g[p * n * n + i * n + j] = v;
                    g[p * n * n + j * n + i] = v;
                }
            }
        }
        let natom = z.len();
        for op in test_ops() {
            let mut h = vec![0.0; 9 * natom * natom];
            let blk = Eri3Block {
                shells: ash.clone(),
                op,
                thresh: 0.0,
            };
            plan.hessian(&blk, &g, &mut h, &mut ws);
            let dens = Dens3c {
                full: Some(&g),
                aux_shells: ash.clone(),
                separable: vec![],
            };
            let hr = eri_hess_3c(&basis, &pairs, &aux, &apairs, &dens, 0.0, &op);
            let md = h
                .iter()
                .zip(&hr)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0, f64::max);
            let mx = hr.iter().map(|a| a.abs()).fold(0.0, f64::max);
            assert!(
                md < 1e-11 * mx.max(1.0),
                "{op:?}: max deviation {md:e} (|H| {mx:e})"
            );
        }
    }
}
