//! Gamma-contracted first derivatives of three-center integrals, the 3-center term of a
//! density-fitted gradient with a non-separable 3-index density (exchange):
//!
//!   g_X = sum_{P, ab} Gamma_P\[ab\] d(ab|P)/dX,
//!
//! on the SIMD auxiliary-lane kernels of [`crate::eri3c`] (W auxiliary primitives per batch).
//!
//! For an orbital shell pair (A, B) with Hermite expansions E (of the product ab) and E_A (of
//! its derivative with respect to A, degree l_ab + 1, [`crate::core::deriv::PairDeriv::ea_t`]), one
//! Hermite potential per primitive pair and lane batch at degree l_aux + l_ab + 1,
//!
//!   H\[k\]\[t'\] = sum_c S_k\[c\] R_{c+t'},   |t'| <= l_ab + 1,
//!
//! gives the derivative with respect to A, (d_A ab|P) = sum_t' H[t'] E_A[t']\[ab\], and the
//! total translation of the product, (S_x ab|P) = (d_A + d_B)(ab|P) = sum_t H[t + 1_x] E\[t\]\[ab\]
//! (shifted potential, degree l_ab only). Then
//!
//!   g_A += Gamma (d_A ab|P),   g_B += Gamma ((S ab|P) - (d_A ab|P)),   g_C -= Gamma (S ab|P)
//!
//! (the auxiliary center by translational invariance). Pairs with both shells on one atom need
//! only the S part, and their lane batches entirely on that atom contribute nothing.
//!
//! Gamma first: per lane batch the weights Gamma_P\[ab\] of its auxiliary functions are gathered
//! into lanes and contracted with each primitive pair's E (and E_A) into Hermite densities,
//! which meet the potential in short dot products; the density of the S part is shared by the
//! three directions (shifted potential). So all heavy work (R recursion, potential, E and Gamma
//! contractions) is lane-parallel, at nk nab (nherm(l_ab) + 3 nherm(l_ab + 1)) lane FMAs per
//! primitive pair (nk nab nherm(l_ab) for one-center pairs) on top of the R recursion one
//! degree above the energy's, instead of 3 nk nab (nherm(l_ab) + nherm(l_ab + 1)) for
//! accumulating the derivative integrals first. The pair data are borrowed from the pair lists
//! (no copies).
//!
//! Screening: per primitive pair the Schwarz bound of the derivative products (max over the
//! d_A and S expansions) times the lane bound times the pair's largest |Gamma| weight.
//! Deterministic: pairs are folded in fixed strided chunks combined in order
//! ([`linus_num::det`]), so the result does not depend on the thread count.

use crate::basis::ncart;
use crate::core::boys;
use crate::core::deriv::PairDeriv;
use crate::core::hermite::{herm_idx, herm_tuv, nherm, ShellPair};
use crate::engine::schedule;
use crate::eri3c::deriv_plan::{BlockCtx, Eri3Thread};
use crate::eri3c::{
    boys_table, h_contract, htab, prim_hermite_sel, r_prog, AuxChunk, Lanes, Work3c, W, ZERO,
};
use crate::eri4c::ERI_PREF;
use crate::operator::Operator;
use linus_num::simd;
use linus_num::simd::Isa;

/// An orbital shell pair prepared for Gamma-contracted 3-center first derivatives: borrows
/// the pair's Hermite expansions (E per primitive \[t\]\[ab\], E_A per primitive \[k\]\[t'\]\[ab\]) and
/// adds primitive derivative bounds.
#[derive(Clone, Debug)]
pub(crate) struct Pair3cG<'a> {
    pub(crate) sp: &'a ShellPair,
    pd: &'a PairDeriv,
    /// both shells on one atom: only the total translation S is needed
    pub(crate) one_center: bool,
    /// derivative Schwarz bound per primitive pair
    qprim: Vec<f64>,
    /// largest primitive bound
    pub(crate) qmax: f64,
}

impl<'a> Pair3cG<'a> {
    pub(crate) fn nab(&self) -> usize {
        self.sp.na * self.sp.nb
    }

    /// From a shell pair and its first-derivative data (same primitive order: a pair of a
    /// [`crate::PairSet`] and its derivative expansion);
    /// `one_center`: both shells sit on the same atom.
    #[allow(clippy::needless_range_loop, reason = "indexes several arrays in step")]
    pub(crate) fn new(sp: &'a ShellPair, pd: &'a PairDeriv, one_center: bool) -> Self {
        let lab = sp.la + sp.lb;
        let (nh, nh1) = (nherm(lab), nherm(lab + 1));
        let nab = sp.na * sp.nb;
        // primitive bounds of the derivative expansions x = E_A,k[ab] and S_k E[ab] in the
        // Coulomb norm, by the triangle inequality over Hermite functions:
        // (x|x)^1/2 <= sum_t |x_t| (L_t|L_t)^1/2, (L_t|L_t) = (-1)^|t| R_2t (PQ = 0, alpha = p/2)
        let l2 = 2 * (lab + 1);
        let tuv = herm_tuv();
        let sh = crate::core::deriv::shift1();
        let mut fm = vec![0.0; boys::MAX_M + 1];
        let mut r = vec![0.0; nherm(l2)];
        let mut rw = vec![0.0; 2 * nherm(l2)];
        let mut dg = vec![0.0; nh1];
        let mut qprim = vec![0.0; sp.nprim];
        for ip in 0..sp.nprim {
            let p = sp.p[ip];
            boys::boys(l2, 0.0, &mut fm);
            let pref = ERI_PREF / (p * p * (2.0 * p).sqrt());
            crate::core::hermite::r_tensor(l2, 0.5 * p, [0.0; 3], &fm, pref, &mut r, &mut rw);
            for (t, d) in dg.iter_mut().enumerate() {
                let [a0, a1, a2] = tuv[t];
                *d = r[herm_idx(2 * a0 as usize, 2 * a1 as usize, 2 * a2 as usize)]
                    .abs()
                    .sqrt();
            }
            let mut qm: f64 = 0.0;
            let e = sp.e_prim(ip);
            let ea = pd.ea_t_prim(ip);
            for k in 0..3 {
                for ab in 0..nab {
                    let mut s = 0.0;
                    for t in 0..nh {
                        s += e[t * nab + ab].abs() * dg[sh[t][k] as usize];
                    }
                    qm = qm.max(s);
                    if !one_center {
                        let mut s = 0.0;
                        for t in 0..nh1 {
                            s += ea[(k * nh1 + t) * nab + ab].abs() * dg[t];
                        }
                        qm = qm.max(s);
                    }
                }
            }
            qprim[ip] = qm;
        }
        let qmax = qprim.iter().cloned().fold(0.0, f64::max);
        Pair3cG {
            sp,
            pd,
            one_center,
            qprim,
            qmax,
        }
    }
}

/// Scratch of the kernel per task.
#[derive(Default)]
pub(crate) struct GWork {
    wl: Vec<Lanes>,
    wbuf: Vec<f64>,
}

/// What one pair's task needs besides the pair and its weights.
struct PairCtx<'a> {
    chunks: &'a [AuxChunk],
    /// common atom of each lane batch's lanes (u32::MAX if mixed)
    chunk_atom: &'a [u32],
    aux_atom: &'a [usize],
    at_a: usize,
    at_b: usize,
    gmax: f64,
    thresh: f64,
    op: Operator,
}

/// out[j] = sum_ab wl[ab] e[t0 + j][ab] (lanes, E t-major), four t at a time: one load of the
/// weights and four broadcast FMAs per ab.
#[inline(always)]
fn wdot4<V: simd::Lanes>(wl: &[Lanes], e: &[f64], nab: usize, t0: usize) -> [V; 4] {
    let (e0, e1, e2, e3) = (
        &e[t0 * nab..(t0 + 1) * nab],
        &e[(t0 + 1) * nab..(t0 + 2) * nab],
        &e[(t0 + 2) * nab..(t0 + 3) * nab],
        &e[(t0 + 3) * nab..(t0 + 4) * nab],
    );
    let (mut s0, mut s1, mut s2, mut s3) = (V::zero(), V::zero(), V::zero(), V::zero());
    for (ab, wv) in wl.iter().enumerate() {
        let wv = V::load8(wv);
        s0 = wv.mul_add(V::splat(e0[ab]), s0);
        s1 = wv.mul_add(V::splat(e1[ab]), s1);
        s2 = wv.mul_add(V::splat(e2[ab]), s2);
        s3 = wv.mul_add(V::splat(e3[ab]), s3);
    }
    [s0, s1, s2, s3]
}

#[inline(always)]
fn wdot1<V: simd::Lanes>(wl: &[Lanes], e: &[f64], nab: usize, t: usize) -> V {
    let e0 = &e[t * nab..(t + 1) * nab];
    let mut s0 = V::zero();
    for (ab, wv) in wl.iter().enumerate() {
        s0 = V::load8(wv).mul_add(V::splat(e0[ab]), s0);
    }
    s0
}

/// One pair's contribution, Gamma first (module docs). The weights are screened per lane batch
/// (their largest magnitude in the batch). Generic over the SIMD type of the contractions
/// (explicit FMAs).
#[inline(always)]
fn pair_grad_dens_impl<V: simd::Lanes>(
    pair: &Pair3cG,
    cx: &PairCtx,
    wb: &[f64],
    w: &mut Work3c,
    gw: &mut GWork,
    g: &mut [[f64; 3]],
) {
    let tab = boys_table();
    let prog = r_prog();
    let sp = pair.sp;
    let lab = sp.la + sp.lb;
    let (nh, nh1) = (nherm(lab), nherm(lab + 1));
    let nab = pair.nab();
    let one = pair.one_center;
    let sh = crate::core::deriv::shift1();
    let nprim = sp.nprim;
    let Work3c {
        fm, rn, r, rw, h, ..
    } = w;
    let GWork { wl, .. } = gw;
    let mut ga = [0.0f64; 3];
    let mut gs = [0.0f64; 3];
    for (ci, ch) in cx.chunks.iter().enumerate() {
        if pair.qmax * ch.q * cx.gmax < cx.thresh {
            continue;
        }
        if one && cx.chunk_atom[ci] == cx.at_a as u32 {
            continue;
        }
        let l = ch.l;
        let nk = if l <= 1 { ncart(l) } else { 2 * l + 1 };
        let blk = nk * nab;
        // weights in lanes: wl[kk][ab][q] = Gamma_{col_q + kk}[ab], and their largest magnitude
        if wl.len() < blk {
            wl.resize(blk, ZERO);
        }
        let wl = &mut wl[..blk];
        let mut wmax: f64 = 0.0;
        for q in 0..W {
            let col = ch.col[q];
            if col == u32::MAX {
                for v in wl.iter_mut() {
                    v[q] = 0.0;
                }
                continue;
            }
            for kk in 0..nk {
                let src = &wb[(col as usize + kk) * nab..(col as usize + kk + 1) * nab];
                for (ab, &v) in src.iter().enumerate() {
                    wl[kk * nab + ab][q] = v;
                    wmax = wmax.max(v.abs());
                }
            }
        }
        let thr = cx.thresh / wmax.max(1e-300);
        if pair.qmax * ch.q < thr {
            continue;
        }
        let lt = l + lab + 1;
        let ht = htab(l, lab + 1);
        if h.len() < nk * nh1 {
            h.resize(nk * nh1, ZERO);
        }
        let mut ls = [V::zero(); 3];
        let mut la = [V::zero(); 3];
        for ip in 0..nprim {
            if pair.qprim[ip] * ch.q < thr {
                continue;
            }
            prim_hermite_sel(sp.p[ip], sp.pc[ip], ch, lt, cx.op, tab, prog, fm, rn, r, rw);
            let hv: &[Lanes] = if l == 0 {
                &r[..nh1]
            } else {
                h_contract(&mut h[..nk * nh1], r, ht, nh1);
                &h[..nk * nh1]
            };
            let e = sp.e_prim(ip);
            let ea = if one { &[][..] } else { pair.pd.ea_t_prim(ip) };
            for kk in 0..nk {
                let wk = &wl[kk * nab..(kk + 1) * nab];
                let hk = &hv[kk * nh1..(kk + 1) * nh1];
                // S: sum_t H[t + 1_x] (sum_ab W[ab] E[t][ab])
                let mut t = 0;
                while t + 4 <= nh {
                    let d: [V; 4] = wdot4(wk, e, nab, t);
                    for (j, &dj) in d.iter().enumerate() {
                        let s = &sh[t + j];
                        for x in 0..3 {
                            ls[x] = V::load8(&hk[s[x] as usize]).mul_add(dj, ls[x]);
                        }
                    }
                    t += 4;
                }
                while t < nh {
                    let dj: V = wdot1(wk, e, nab, t);
                    let s = &sh[t];
                    for x in 0..3 {
                        ls[x] = V::load8(&hk[s[x] as usize]).mul_add(dj, ls[x]);
                    }
                    t += 1;
                }
                if !one {
                    // d_A: sum_t' H[t'] (sum_ab W[ab] E_A,x[t'][ab])
                    for x in 0..3 {
                        let ex = &ea[x * nh1 * nab..(x + 1) * nh1 * nab];
                        let mut t = 0;
                        while t + 4 <= nh1 {
                            let d: [V; 4] = wdot4(wk, ex, nab, t);
                            for (j, &dj) in d.iter().enumerate() {
                                la[x] = V::load8(&hk[t + j]).mul_add(dj, la[x]);
                            }
                            t += 4;
                        }
                        while t < nh1 {
                            let dj: V = wdot1(wk, ex, nab, t);
                            la[x] = V::load8(&hk[t]).mul_add(dj, la[x]);
                            t += 1;
                        }
                    }
                }
            }
        }
        let (ls, la) = (ls.map(|v| v.to_array()), la.map(|v| v.to_array()));
        for q in 0..W {
            let col = ch.col[q];
            if col == u32::MAX {
                continue;
            }
            let at = cx.aux_atom[col as usize];
            for x in 0..3 {
                g[at][x] -= ls[x][q];
                gs[x] += ls[x][q];
                ga[x] += la[x][q];
            }
        }
    }
    for x in 0..3 {
        g[cx.at_a][x] += ga[x];
        g[cx.at_b][x] += gs[x] - ga[x];
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512vl,avx512dq,avx2,fma")]
unsafe fn pair_grad_avx512(
    pair: &Pair3cG,
    cx: &PairCtx,
    wb: &[f64],
    w: &mut Work3c,
    gw: &mut GWork,
    g: &mut [[f64; 3]],
) {
    pair_grad_dens_impl::<simd::Avx512>(pair, cx, wb, w, gw, g)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn pair_grad_avx2(
    pair: &Pair3cG,
    cx: &PairCtx,
    wb: &[f64],
    w: &mut Work3c,
    gw: &mut GWork,
    g: &mut [[f64; 3]],
) {
    pair_grad_dens_impl::<simd::Avx2>(pair, cx, wb, w, gw, g)
}

fn pair_grad_generic(
    pair: &Pair3cG,
    cx: &PairCtx,
    wb: &[f64],
    w: &mut Work3c,
    gw: &mut GWork,
    g: &mut [[f64; 3]],
) {
    pair_grad_dens_impl::<simd::Portable<false>>(pair, cx, wb, w, gw, g)
}

/// sum_{P in the block, ab} Gamma_P\[ab\] d(ab|op|P)/dX over the shell pairs `pairs` for the
/// block of auxiliary shells `cx`, into the partials `parts` (one per chunk of the ordered
/// reduction, zeroed here; the caller adds them in chunk order), on the threads' scratch.
/// `weights` fills a pair's Gamma for the block (nq x nab, \[P\]\[ab\] with
/// ab = ia * nb + ib, the factor 2 of an off-diagonal shell pair included: it stands for (a, b)
/// and (b, a)) and returns max |Gamma| (0: the pair is skipped); pairs, lane batches and
/// primitive pairs whose derivative bound times the pair's max |Gamma| is below the operator's
/// [`Operator::screen_threshold`] of `cx.thresh` are skipped (the Coulomb bounds scaled
/// by the operator's factor, as in the four-center code).
pub(crate) fn grad_block<F>(
    pairs: &[Pair3cG],
    cx: &BlockCtx<'_>,
    weights: &F,
    threads: &mut [Eri3Thread],
    parts: &mut [Vec<[f64; 3]>],
) where
    F: Fn(&ShellPair, &mut [f64]) -> f64 + Sync,
{
    let chunk_atom: Vec<u32> = cx
        .chunks
        .iter()
        .map(|ch| {
            let mut at = u32::MAX;
            for q in 0..W {
                let col = ch.col[q];
                if col == u32::MAX {
                    continue;
                }
                let a = cx.aux_atom[col as usize] as u32;
                if at == u32::MAX {
                    at = a;
                } else if at != a {
                    return u32::MAX;
                }
            }
            at
        })
        .collect();
    let opk = cx.op.kernel();
    let (isa, nq, qaux) = (cx.isa, cx.nq, cx.aux3c.qmax);
    let thresh = cx.op.screen_threshold(cx.thresh);
    for g in parts.iter_mut() {
        g.fill([0.0; 3]);
    }
    schedule::ordered_strided(pairs.len(), threads, parts, |t, g, pi| {
        let pair = &pairs[pi];
        let nab = pair.nab();
        let (w, gw) = (&mut t.work, &mut t.gw);
        let mut wb = std::mem::take(&mut gw.wbuf);
        wb.resize(nq * nab, 0.0);
        let gmax = weights(pair.sp, &mut wb[..nq * nab]);
        if gmax > 0.0 && pair.qmax * qaux * gmax >= thresh {
            let pc = PairCtx {
                chunks: &cx.chunks,
                chunk_atom: &chunk_atom,
                aux_atom: cx.aux_atom,
                at_a: cx.ao_atom[pair.sp.sa],
                at_b: cx.ao_atom[pair.sp.sb],
                gmax,
                thresh,
                op: opk,
            };
            match isa {
                // SAFETY: plain computation; the target features are checked by `isa`
                #[cfg(target_arch = "x86_64")]
                Isa::Avx512 => unsafe { pair_grad_avx512(pair, &pc, &wb[..nq * nab], w, gw, g) },
                // SAFETY: as above
                #[cfg(target_arch = "x86_64")]
                Isa::Avx2 => unsafe { pair_grad_avx2(pair, &pc, &wb[..nq * nab], w, gw, g) },
                _ => pair_grad_generic(pair, &pc, &wb[..nq * nab], w, gw, g),
            }
        }
        gw.wbuf = wb;
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::basis::BasisDef;
    use crate::eri3c::deriv_plan::{Eri3Block, Eri3DerivOptions, Eri3DerivPlan};
    use crate::eri3c::DfThresholds;
    use crate::pairs::{DerivOrder, PairOptions, PairSet};
    use crate::reference::{eri_grad_3c, Dens3c};
    use crate::BasisSet;

    /// The plan's Gamma-contracted gradient equals the generic quartet kernel (eri_grad_3c)
    /// for a random symmetric Gamma: Coulomb, erf and a combined operator, on blocks of
    /// auxiliary shells.
    #[test]
    fn gradient_matches_generic_kernel() {
        let z = vec![8, 1, 1, 6, 8];
        let positions = vec![
            [0.1, -0.05, 0.12],
            [1.6, 0.98, -0.06],
            [-1.28, 1.04, 0.09],
            [0.3, -2.4, 1.1],
            [0.35, -2.5, 3.4],
        ];
        let basis = BasisSet::new(
            &BasisDef::load("def2-tzvp", None).unwrap(),
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
        let ao: Vec<(usize, usize)> = basis
            .shells
            .iter()
            .map(|s| (s.ao_start, s.nfunc()))
            .collect();
        let natom = z.len();
        assert_eq!(plan.natom(), natom);
        let mut seed = 987654321u64;
        let mut rnd = || {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((seed >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
        };
        for (ash, op) in [
            (0..aux.shells.len(), Operator::Coulomb),
            (5..40, Operator::LongRange { omega: 0.3 }),
            (
                17..60,
                Operator::Attenuated {
                    coulomb: 0.6,
                    long_range: -0.4,
                    omega: 0.33,
                },
            ),
        ] {
            let p0 = aux.shells[ash.start].ao_start;
            let nq = aux.shells[ash.end - 1].ao_start + aux.shells[ash.end - 1].nfunc() - p0;
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
            let wfn = |sp: &ShellPair, w: &mut [f64]| -> f64 {
                let (a0, na) = ao[sp.sa];
                let (b0, nb) = ao[sp.sb];
                let f = if sp.sa == sp.sb { 1.0 } else { 2.0 };
                let nab = na * nb;
                let mut gm: f64 = 0.0;
                for p in 0..nq {
                    for ia in 0..na {
                        for ib in 0..nb {
                            let v = 0.5
                                * f
                                * (g[p * n * n + (a0 + ia) * n + b0 + ib]
                                    + g[p * n * n + (b0 + ib) * n + a0 + ia]);
                            w[p * nab + ia * nb + ib] = v;
                            gm = gm.max(v.abs());
                        }
                    }
                }
                gm
            };
            let dens = Dens3c {
                full: Some(&g),
                aux_shells: ash.clone(),
                separable: vec![],
            };
            let gr = eri_grad_3c(&basis, &pairs, &aux, &apairs, &dens, 0.0, &op);
            let mx = gr
                .iter()
                .flat_map(|a| a.iter().map(|x| x.abs()))
                .fold(0.0, f64::max);
            let mut gn = vec![[0.0; 3]; natom];
            let blk = Eri3Block {
                shells: ash.clone(),
                op,
                thresh: 0.0,
            };
            plan.gradient(&blk, wfn, &mut gn, &mut ws);
            let md = gn
                .iter()
                .zip(&gr)
                .flat_map(|(a, b)| (0..3).map(move |k| (a[k] - b[k]).abs()))
                .fold(0.0, f64::max);
            assert!(
                md < 1e-11 * mx.max(1.0),
                "{op:?}: max deviation {md:e} (|g| {mx:e})"
            );
            // translational invariance
            for k in 0..3 {
                let s: f64 = gn.iter().map(|v| v[k]).sum();
                assert!(s.abs() < 1e-10 * mx.max(1.0), "sum {s:e}");
            }
        }
    }
}
