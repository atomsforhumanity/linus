//! Two-electron repulsion integrals in the factorized McMurchie-Davidson form
//! (ab|cd) = sum_{bra prims} E_bra^T [ sum_{ket prims} R E_ket ]   (SHARK, Scheme 1).
//!
//! This is the generic (any angular momentum) reference path; specialized kernels for low
//! angular momentum and fused J/K digestion live elsewhere and are tested against it.

pub mod batch;
pub mod deriv;
pub mod deriv_plan;
pub mod jk;
pub mod jk_deriv;

use crate::basis::BasisSet;
use crate::core::boys;
use crate::core::hermite::{nherm, r_tensor, sum_table, ShellPair};
use crate::operator::{rn_operator_scalar, Operator};
use crate::pairs::PairSet;
use rayon::prelude::*;

/// 2 pi^{5/2}
pub const ERI_PREF: f64 = 2.0 * 17.493_418_327_624_862; // pi^{5/2} = 17.4934183276248628...

#[derive(Default)]
pub struct EriWork {
    fm: Vec<f64>,
    r: Vec<f64>,
    rwork: Vec<f64>,
    rmat: Vec<f64>,
    t: Vec<f64>,
    rn: Vec<f64>,
}

impl EriWork {
    pub fn new() -> Self {
        EriWork {
            fm: vec![0.0; boys::MAX_M + 1],
            ..Default::default()
        }
    }
    fn ensure(&mut self, l: usize, nhb: usize, nhk: usize, ncd: usize) {
        let nh = nherm(l);
        if self.r.len() < nh {
            self.r.resize(nh, 0.0);
            self.rwork.resize(2 * nh, 0.0);
        }
        if self.rmat.len() < nhb * nhk {
            self.rmat.resize(nhb * nhk, 0.0);
        }
        if self.t.len() < nhb * ncd {
            self.t.resize(nhb * ncd, 0.0);
        }
    }
}

/// Integrals (ab|op|cd) of one shell quartet for the two-electron operator `op`
/// (a/r + b erf(omega r)/r); out[(a*nb + b) * ncd + c*nd + d].
pub fn eri_quartet(
    bra: &ShellPair,
    ket: &ShellPair,
    op: &Operator,
    out: &mut [f64],
    w: &mut EriWork,
) {
    let (lbra, lket) = (bra.lab(), ket.lab());
    let l = lbra + lket;
    let (nhb, nhk) = (nherm(lbra), nherm(lket));
    let (nab, ncd) = (bra.nab(), ket.nab());
    let out = &mut out[..nab * ncd];
    out.fill(0.0);
    if bra.nprim == 0 || ket.nprim == 0 {
        return;
    }
    w.ensure(l, nhb, nhk, ncd);
    let st = sum_table(lbra, lket);
    let coulomb = op.is_coulomb();
    if !coulomb && w.rn.len() < l + 1 {
        w.rn.resize(l + 1, 0.0);
    }
    let EriWork {
        fm,
        r,
        rwork,
        rmat,
        t,
        rn,
    } = w;
    let t = &mut t[..nhb * ncd];
    let rmat = &mut rmat[..nhb * nhk];
    for ip in 0..bra.nprim {
        let p = bra.p[ip];
        let pc = bra.pc[ip];
        t.fill(0.0);
        for kq in 0..ket.nprim {
            let q = ket.p[kq];
            let qc = ket.pc[kq];
            let pq = [pc[0] - qc[0], pc[1] - qc[1], pc[2] - qc[2]];
            let alpha = p * q / (p + q);
            let tt = alpha * (pq[0] * pq[0] + pq[1] * pq[1] + pq[2] * pq[2]);
            let pref = ERI_PREF / (p * q * (p + q).sqrt());
            if coulomb {
                boys::boys(l, tt, fm);
                r_tensor(l, alpha, pq, fm, pref, r, rwork);
            } else {
                let r2 = pq[0] * pq[0] + pq[1] * pq[1] + pq[2] * pq[2];
                rn_operator_scalar(l, alpha, r2, pref, op, fm, rn);
                crate::eri4c::batch::r_tensor_loop(l, rn, pq[0], pq[1], pq[2], r, rwork);
            }
            for (k, (rm, &ix)) in rmat.iter_mut().zip(st.idx.iter()).enumerate() {
                *rm = st.sign[k % nhk] * r[ix as usize];
            }
            // t[r][cd] += sum_s rmat[r][s] * eket[s][cd]
            let ek = ket.e_prim(kq);
            for rr in 0..nhb {
                let trow = &mut t[rr * ncd..(rr + 1) * ncd];
                for s in 0..nhk {
                    let v = rmat[rr * nhk + s];
                    if v == 0.0 {
                        continue;
                    }
                    let erow = &ek[s * ncd..(s + 1) * ncd];
                    for (tv, &ev) in trow.iter_mut().zip(erow) {
                        *tv += v * ev;
                    }
                }
            }
        }
        // out[ab][cd] += sum_r ebra[r][ab] t[r][cd]
        let eb = bra.e_prim(ip);
        for rr in 0..nhb {
            let trow = &t[rr * ncd..(rr + 1) * ncd];
            let erow = &eb[rr * nab..(rr + 1) * nab];
            for (ab, &ev) in erow.iter().enumerate() {
                if ev == 0.0 {
                    continue;
                }
                let orow = &mut out[ab * ncd..(ab + 1) * ncd];
                for (o, &tv) in orow.iter_mut().zip(trow) {
                    *o += ev * tv;
                }
            }
        }
    }
}

/// The full ERI tensor (nbf^4, chemist's notation, row-major) of the two-electron operator
/// `op`, for tests and small systems.
pub fn eri_full(basis: &BasisSet, pairs: &PairSet, op: &Operator) -> Vec<f64> {
    let nbf = basis.nbf;
    let np = pairs.len();
    let sh = &basis.shells;
    let blocks: Vec<(usize, usize, Vec<f64>)> = (0..np)
        .into_par_iter()
        .flat_map_iter(|ij| (0..=ij).map(move |kl| (ij, kl)))
        .map_init(EriWork::new, |w, (ij, kl)| {
            let (b, k) = (pairs.pair(ij), pairs.pair(kl));
            let mut buf = vec![0.0; b.nab() * k.nab()];
            eri_quartet(b, k, op, &mut buf, w);
            (ij, kl, buf)
        })
        .collect();
    let mut g = vec![0.0; nbf * nbf * nbf * nbf];
    let n2 = nbf * nbf;
    for (ij, kl, buf) in blocks {
        let (b, k) = (pairs.pair(ij), pairs.pair(kl));
        let (a0, b0) = (sh[b.sa].ao_start, sh[b.sb].ao_start);
        let (c0, d0) = (sh[k.sa].ao_start, sh[k.sb].ao_start);
        let (na, nb, nc, nd) = (b.na, b.nb, k.na, k.nb);
        for ia in 0..na {
            for ib in 0..nb {
                for ic in 0..nc {
                    for id in 0..nd {
                        let v = buf[((ia * nb + ib) * nc + ic) * nd + id];
                        let (p, q, r, s) = (a0 + ia, b0 + ib, c0 + ic, d0 + id);
                        for (x, y) in [
                            (p * nbf + q, r * nbf + s),
                            (q * nbf + p, r * nbf + s),
                            (p * nbf + q, s * nbf + r),
                            (q * nbf + p, s * nbf + r),
                        ] {
                            g[x * n2 + y] = v;
                            g[y * n2 + x] = v;
                        }
                    }
                }
            }
        }
    }
    g
}
