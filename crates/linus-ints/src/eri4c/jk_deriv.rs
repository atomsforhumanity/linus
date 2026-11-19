//! First-derivative Coulomb and exchange matrices for nuclear perturbations (the
//! two-electron part of the CPHF right-hand side F^x):
//!
//!   J^x\[D\]_mn = sum_ls ∂(mn|ls)/∂x D_ls,   K^x\[D\]_mn = sum_ls ∂(ml|ns)/∂x D_ls,
//!
//! for x = 3 * atom + k. With B(a,b,c,d) = (∂_A ab|cd), the derivative with respect to the
//! center of the first function (a on atom X), and symmetric D,
//!
//!   J^X = A + A^T + 2 C,   A_ab = sum_cd B(a,b,c,d) D_cd,  C_cd = sum_ab B(a,b,c,d) D_ab,
//!   K^X = P + P^T + Q + Q^T,  P_ac = sum_bd B(a,b,c,d) D_bd,  Q_bc = sum_ad B(a,b,c,d) D_ad,
//!
//! so only bra-side first-center derivatives are needed (one "T" pass per primitive quartet,
//! no ket derivatives). Each atom is processed separately, which bounds the scratch memory to
//! a few nbf x nbf matrices per thread.

use crate::basis::BasisSet;
use crate::core::boys;
use crate::core::deriv::{r_tensor_fast, wtable, PairDeriv};
use crate::core::hermite::{nherm, ShellPair, ShellView};
use crate::core::pair_data::{self, PRIM_PAIR_CUT};
use crate::core::util::shell_pair_dmax;
use crate::eri4c::ERI_PREF;
use crate::operator::{boys_operator, Operator};
use crate::pairs::PairSet;
use rayon::prelude::*;

/// Scratch for [`eri_quartet_deriv_a`].
#[derive(Default)]
pub struct DerivAWork {
    fm: Vec<f64>,
    r: Vec<f64>,
    rwork: Vec<f64>,
    w: Vec<f64>,
    t: Vec<f64>,
    /// Two-electron operator (Coulomb by default; erf-attenuated and CAM combinations change
    /// only the Boys stage, see [`boys_operator`]).
    pub op: Operator,
}

impl DerivAWork {
    pub fn new() -> Self {
        DerivAWork {
            fm: vec![0.0; boys::MAX_M + 1],
            ..Default::default()
        }
    }

    /// Scratch for derivative integrals of the operator `op`.
    pub fn with_op(op: Operator) -> Self {
        DerivAWork { op, ..Self::new() }
    }
}

/// Derivatives with respect to the center A of the first shell of the bra only:
/// out[(k * nab + ab) * ncd + cd] = ∂(ab|cd)/∂A_k.
pub fn eri_quartet_deriv_a(
    bra: &ShellPair,
    bd: &PairDeriv,
    ket: &ShellPair,
    out: &mut [f64],
    w: &mut DerivAWork,
) {
    let (lab, lcd) = (bra.lab(), ket.lab());
    let (nab, ncd) = (bra.nab(), ket.nab());
    let blk = nab * ncd;
    let out = &mut out[..3 * blk];
    out.fill(0.0);
    if bra.nprim == 0 || ket.nprim == 0 {
        return;
    }
    let l = lab + lcd + 1;
    let (nhb1, nhk) = (nherm(lab + 1), nherm(lcd));
    let wt = wtable(lab + 1, lcd, 0);
    let DerivAWork {
        fm,
        r,
        rwork,
        w: wm,
        t,
        op,
    } = w;
    let nh = nherm(l);
    if r.len() < nh + 1 {
        r.resize(nh + 1, 0.0);
        rwork.resize(2 * nh, 0.0);
    }
    if wm.len() < nhb1 * nhk {
        wm.resize(nhb1 * nhk, 0.0);
    }
    if t.len() < nhb1 * ncd {
        t.resize(nhb1 * ncd, 0.0);
    }
    for ip in 0..bra.nprim {
        let p = bra.p[ip];
        let pc = bra.pc[ip];
        let t = &mut t[..nhb1 * ncd];
        t.fill(0.0);
        for kq in 0..ket.nprim {
            let q = ket.p[kq];
            let qc = ket.pc[kq];
            let pq = [pc[0] - qc[0], pc[1] - qc[1], pc[2] - qc[2]];
            let alpha = p * q / (p + q);
            boys_operator(
                l,
                alpha,
                pq[0] * pq[0] + pq[1] * pq[1] + pq[2] * pq[2],
                op,
                fm,
            );
            r_tensor_fast(
                l,
                alpha,
                pq,
                fm,
                ERI_PREF / (p * q * (p + q).sqrt()),
                r,
                rwork,
            );
            r[nh] = 0.0;
            wt.gather(&r[..nh + 1], wm);
            crate::core::kernels::wmul_acc(wm, nhk, nhb1, nhk, ket.e_prim(kq), ncd, t);
        }
        let eap = bd.ea_prim(ip);
        for rr in 0..nhb1 {
            let trow = &t[rr * ncd..(rr + 1) * ncd];
            for k in 0..3 {
                let erow = &eap[(rr * 3 + k) * nab..(rr * 3 + k + 1) * nab];
                let o = &mut out[k * blk..(k + 1) * blk];
                for (ab, &ev) in erow.iter().enumerate() {
                    if ev == 0.0 {
                        continue;
                    }
                    for (x, &y) in o[ab * ncd..(ab + 1) * ncd].iter_mut().zip(trow) {
                        *x += ev * y;
                    }
                }
            }
        }
    }
}

/// Per-atom accumulators for one density: A, C (J) and P, Q (K), each 3 x nbf x nbf.
struct Acc {
    a: Vec<f64>,
    c: Vec<f64>,
    p: Vec<f64>,
    q: Vec<f64>,
}

impl Acc {
    fn new(n2: usize, want_j: bool, want_k: bool) -> Self {
        let z = |b: bool| if b { vec![0.0; 3 * n2] } else { Vec::new() };
        Acc {
            a: z(want_j),
            c: z(want_j),
            p: z(want_k),
            q: z(want_k),
        }
    }
    fn add(&mut self, o: &Acc) {
        for (x, y) in [
            (&mut self.a, &o.a),
            (&mut self.c, &o.c),
            (&mut self.p, &o.p),
            (&mut self.q, &o.q),
        ] {
            for (u, v) in x.iter_mut().zip(y.iter()) {
                *u += v;
            }
        }
    }
}

/// Digest one block B[k][a][b][c][d] (bra (sa, sb) ordered with sa on the perturbed atom,
/// ket pair (sc >= sd)) into the accumulators of each density.
#[allow(clippy::too_many_arguments)]
fn digest(
    basis: &BasisSet,
    blk: &[f64],
    sa: usize,
    sb: usize,
    sc: usize,
    sd: usize,
    dens: &[&[f64]],
    acc: &mut [Acc],
    want_j: bool,
    want_k: bool,
) {
    let sh = &basis.shells;
    let nbf = basis.nbf;
    let n2 = nbf * nbf;
    let (na, nb, nc, nd) = (
        sh[sa].nfunc(),
        sh[sb].nfunc(),
        sh[sc].nfunc(),
        sh[sd].nfunc(),
    );
    let (a0, b0, c0, d0) = (
        sh[sa].ao_start,
        sh[sb].ao_start,
        sh[sc].ao_start,
        sh[sd].ao_start,
    );
    let ncd = nc * nd;
    let nabcd = na * nb * ncd;
    let swap = sc != sd;
    let fk = if swap { 2.0 } else { 1.0 };
    for (dm, ac) in dens.iter().zip(acc.iter_mut()) {
        for k in 0..3 {
            let bk = &blk[k * nabcd..(k + 1) * nabcd];
            for a in 0..na {
                let ia = a0 + a;
                for b in 0..nb {
                    let ib = b0 + b;
                    let row = &bk[(a * nb + b) * ncd..(a * nb + b + 1) * ncd];
                    if want_j {
                        let dab = dm[ia * nbf + ib];
                        let mut s = 0.0;
                        for c in 0..nc {
                            let ic = c0 + c;
                            let crow =
                                &mut ac.c[k * n2 + ic * nbf + d0..k * n2 + ic * nbf + d0 + nd];
                            let drow = &dm[ic * nbf + d0..ic * nbf + d0 + nd];
                            for d in 0..nd {
                                let v = row[c * nd + d];
                                s += v * drow[d];
                                crow[d] += v * dab;
                            }
                        }
                        ac.a[k * n2 + ia * nbf + ib] += fk * s;
                        if swap {
                            for c in 0..nc {
                                let ic = c0 + c;
                                for d in 0..nd {
                                    ac.c[k * n2 + (d0 + d) * nbf + ic] += row[c * nd + d] * dab;
                                }
                            }
                        }
                    }
                    if want_k {
                        // P_ac += B D_bd ; Q_bc += B D_ad ; swapped ket: P_ad += B D_bc ; Q_bd += B D_ac
                        for c in 0..nc {
                            let ic = c0 + c;
                            let mut sp = 0.0;
                            let mut sq = 0.0;
                            for d in 0..nd {
                                let id = d0 + d;
                                let v = row[c * nd + d];
                                sp += v * dm[ib * nbf + id];
                                sq += v * dm[ia * nbf + id];
                            }
                            ac.p[k * n2 + ia * nbf + ic] += sp;
                            ac.q[k * n2 + ib * nbf + ic] += sq;
                        }
                        if swap {
                            for d in 0..nd {
                                let id = d0 + d;
                                let mut sp = 0.0;
                                let mut sq = 0.0;
                                for c in 0..nc {
                                    let ic = c0 + c;
                                    let v = row[c * nd + d];
                                    sp += v * dm[ib * nbf + ic];
                                    sq += v * dm[ia * nbf + ic];
                                }
                                ac.p[k * n2 + ia * nbf + id] += sp;
                                ac.q[k * n2 + ib * nbf + id] += sq;
                            }
                        }
                    }
                }
            }
        }
    }
}

/// Molecules with at least this many atoms are done one atom per task in [`jk_deriv`].
const JK_DERIV_ATOM_TASKS: usize = 16;

/// First-derivative J and K matrices for all 3 natom nuclear perturbations and every
/// (symmetric) density in `dens`: returns (J, K), each a Vec over densities of
/// (3 natom) x nbf x nbf row-major matrices (empty Vec when not requested). `pairs`: the
/// index-ordered ket pairs with their Schwarz factors; `thresh` screens Q_ab' Q_cd max|D|
/// (e.g. 1e-12).
/// With the integrals of the two-electron operator `op` (a/r + b erf(omega r)/r)
/// in J and K, e.g. the exchange of a range-separated hybrid (K only, the coefficients in
/// `op`); the Schwarz factors stay the Coulomb ones (they bound the erf integrals too, scaled by
/// [`Operator::schwarz_factor`]).
#[allow(clippy::too_many_arguments)]
#[allow(clippy::type_complexity, reason = "a one-off tuple of buffers")]
pub fn jk_deriv(
    basis: &BasisSet,
    pairs: &PairSet,
    dens: &[&[f64]],
    want_j: bool,
    want_k: bool,
    thresh: f64,
    op: &Operator,
) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let thresh = op.screen_threshold(thresh);
    let nbf = basis.nbf;
    let n2 = nbf * nbf;
    let natom = basis.atom_shells.len();
    let sh = &basis.shells;
    let ns = sh.len();
    let nd = dens.len();
    let views: Vec<ShellView> = sh.iter().map(ShellView::from_shell).collect();
    let dmax: Vec<Vec<f64>> = dens.iter().map(|d| shell_pair_dmax(basis, d)).collect();
    let dm_all: Vec<f64> = (0..ns * ns)
        .map(|x| dmax.iter().map(|m| m[x]).fold(0.0, f64::max))
        .collect();
    let schwarz = pairs.schwarz();
    let qmax = pairs.schwarz_max();
    let atom_job = |x: usize, inner_par: bool| -> Vec<Acc> {
        // ordered bra pairs (a on atom x, b any) with first-center derivatives and their bounds
        let bras: Vec<(usize, usize, ShellPair, PairDeriv, f64)> = basis.atom_shells[x]
            .clone()
            .flat_map(|a| (0..ns).map(move |b| (a, b)))
            .map(|(a, b)| {
                let (pr, pd) = pair_data::build(&views[a], &views[b], a, b, 1, PRIM_PAIR_CUT);
                let pd = pd.expect("first-order pair data");
                let q = crate::eri4c::deriv::schwarz_deriv(
                    std::slice::from_ref(&pr),
                    std::slice::from_ref(&pd),
                )
                .0[0];
                (a, b, pr, pd, q)
            })
            .collect();
        let work = |accs: &mut Vec<Acc>, w: &mut DerivAWork, buf: &mut Vec<f64>, bi: usize| {
            let (sa, sb, bra, bd, qb) = &bras[bi];
            if bra.nprim == 0 || qb * qmax < thresh {
                return;
            }
            for (kl, ket) in pairs.pairs().iter().enumerate() {
                let qq = qb * schwarz[kl];
                if ket.nprim == 0 || qq < thresh {
                    continue;
                }
                let (sc, sd) = (ket.sa, ket.sb);
                let mut dm: f64 = 0.0;
                if want_j {
                    dm = dm.max(dm_all[sc * ns + sd]).max(dm_all[sa * ns + sb]);
                }
                if want_k {
                    dm = dm
                        .max(dm_all[sb * ns + sd])
                        .max(dm_all[sa * ns + sd])
                        .max(dm_all[sb * ns + sc])
                        .max(dm_all[sa * ns + sc]);
                }
                if 2.0 * qq * dm < thresh {
                    continue;
                }
                let blk = 3 * bra.nab() * ket.nab();
                buf.resize(blk, 0.0);
                eri_quartet_deriv_a(bra, bd, ket, buf, w);
                digest(basis, buf, *sa, *sb, sc, sd, dens, accs, want_j, want_k);
            }
        };
        let new_accs = || {
            (0..nd)
                .map(|_| Acc::new(n2, want_j, want_k))
                .collect::<Vec<_>>()
        };
        if inner_par {
            // deterministic reduction over fixed chunks of bras (independent of the thread count)
            let acc_len = nd * 3 * n2 * (if want_j { 2 } else { 0 } + if want_k { 2 } else { 0 });
            let nch =
                linus_num::det::nchunk_for(bras.len(), acc_len, linus_num::det::DEFAULT_BUDGET);
            linus_num::det::par_fold_det(
                bras.len(),
                nch,
                || (new_accs(), DerivAWork::with_op(*op), Vec::new()),
                |(accs, w, buf), bi| work(accs, w, buf, bi),
                |a, b| {
                    for (x, y) in a.0.iter_mut().zip(&b.0) {
                        x.add(y);
                    }
                },
            )
            .0
        } else {
            let mut accs = new_accs();
            let mut w = DerivAWork::with_op(*op);
            let mut buf = Vec::new();
            for bi in 0..bras.len() {
                work(&mut accs, &mut w, &mut buf, bi);
            }
            accs
        }
    };
    // J^X, K^X (3 x nbf² per density) of one atom from its accumulators.
    let finish = |accs: Vec<Acc>| -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
        let mut js = Vec::new();
        let mut ks = Vec::new();
        for ac in &accs {
            if want_j {
                let mut m = vec![0.0; 3 * n2];
                for k in 0..3 {
                    let (a, c) = (&ac.a[k * n2..(k + 1) * n2], &ac.c[k * n2..(k + 1) * n2]);
                    let mk = &mut m[k * n2..(k + 1) * n2];
                    for i in 0..nbf {
                        for j in 0..nbf {
                            mk[i * nbf + j] =
                                a[i * nbf + j] + a[j * nbf + i] + 2.0 * c[i * nbf + j];
                        }
                    }
                }
                js.push(m);
            }
            if want_k {
                let mut m = vec![0.0; 3 * n2];
                for k in 0..3 {
                    let (p, q) = (&ac.p[k * n2..(k + 1) * n2], &ac.q[k * n2..(k + 1) * n2]);
                    let mk = &mut m[k * n2..(k + 1) * n2];
                    for i in 0..nbf {
                        for j in 0..nbf {
                            mk[i * nbf + j] =
                                p[i * nbf + j] + p[j * nbf + i] + q[i * nbf + j] + q[j * nbf + i];
                        }
                    }
                }
                ks.push(m);
            }
        }
        (js, ks)
    };
    let mut jout: Vec<Vec<f64>> = if want_j {
        (0..nd).map(|_| vec![0.0; 3 * natom * n2]).collect()
    } else {
        Vec::new()
    };
    let mut kout: Vec<Vec<f64>> = if want_k {
        (0..nd).map(|_| vec![0.0; 3 * natom * n2]).collect()
    } else {
        Vec::new()
    };
    let mut store = |x: usize, (js, ks): (Vec<Vec<f64>>, Vec<Vec<f64>>)| {
        for (di, m) in js.into_iter().enumerate() {
            jout[di][3 * x * n2..(3 * x + 3) * n2].copy_from_slice(&m);
        }
        for (di, m) in ks.into_iter().enumerate() {
            kout[di][3 * x * n2..(3 * x + 3) * n2].copy_from_slice(&m);
        }
    };
    // one atom per task for many atoms, else parallel within each atom; the choice depends on
    // the molecule only, so results do not depend on the thread count
    if natom >= JK_DERIV_ATOM_TASKS {
        // one atom per task (bounded scratch: one set of accumulators per running task)
        let res: Vec<(usize, (Vec<Vec<f64>>, Vec<Vec<f64>>))> = (0..natom)
            .into_par_iter()
            .map(|x| (x, finish(atom_job(x, false))))
            .collect();
        for (x, r) in res {
            store(x, r);
        }
    } else {
        for x in 0..natom {
            let r = finish(atom_job(x, true));
            store(x, r);
        }
    }
    (jout, kout)
}
