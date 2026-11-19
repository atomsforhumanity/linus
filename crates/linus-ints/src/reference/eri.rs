//! Two-electron integrals over plain quartets ([`crate::eri4c::eri_quartet`]): the
//! density-fitting metric and three-center integrals, raw first-derivative blocks of a quartet,
//! and first and second derivatives through derivative pseudo pairs.

use crate::basis::BasisSet;
use crate::core::deriv::{
    deriv2_pair, deriv_pair_a, deriv_pair_b, shift1, wtable, D2Kind, PairDeriv,
};
use crate::core::hermite::{nherm, ShellPair};
use crate::core::kernels::{wmul_acc, wmul_t_acc};
use crate::eri4c::deriv::{fill_w, grow, DerivWork, D2_CENTER_PAIRS};
use crate::eri4c::{eri_quartet, EriWork};
use crate::operator::Operator;
use crate::pairs::PairSet;
use rayon::prelude::*;

/// Two-center Coulomb metric (P|Q), naux x naux.
pub fn eri_2c(aux: &BasisSet, apairs: &PairSet) -> Vec<f64> {
    let n = aux.nbf;
    let ns = apairs.len();
    let blocks: Vec<(usize, usize, Vec<f64>)> = (0..ns)
        .into_par_iter()
        .flat_map_iter(|i| (0..=i).map(move |j| (i, j)))
        .map_init(EriWork::new, |w, (i, j)| {
            let mut buf = vec![0.0; apairs.pair(i).na * apairs.pair(j).na];
            eri_quartet(
                apairs.pair(i),
                apairs.pair(j),
                &Operator::Coulomb,
                &mut buf,
                w,
            );
            (i, j, buf)
        })
        .collect();
    let mut m = vec![0.0; n * n];
    for (i, j, buf) in blocks {
        let (p0, q0) = (aux.shells[i].ao_start, aux.shells[j].ao_start);
        let (np, nq) = (apairs.pair(i).na, apairs.pair(j).na);
        for a in 0..np {
            for b in 0..nq {
                m[(p0 + a) * n + q0 + b] = buf[a * nq + b];
                m[(q0 + b) * n + p0 + a] = buf[a * nq + b];
            }
        }
    }
    m
}

/// Three-center integrals (P|mn) as a naux x (nbf*nbf) row-major array (both triangles).
pub fn eri_3c(basis: &BasisSet, pairs: &PairSet, aux: &BasisSet, apairs: &PairSet) -> Vec<f64> {
    let nbf = basis.nbf;
    let naux = aux.nbf;
    let sh = &basis.shells;
    let blocks: Vec<(usize, usize, Vec<f64>)> = (0..apairs.len())
        .into_par_iter()
        .flat_map_iter(|p| (0..pairs.len()).map(move |ij| (p, ij)))
        .map_init(EriWork::new, |w, (p, ij)| {
            let pr = pairs.pair(ij);
            let mut buf = vec![0.0; apairs.pair(p).na * pr.nab()];
            eri_quartet(apairs.pair(p), pr, &Operator::Coulomb, &mut buf, w);
            (p, ij, buf)
        })
        .collect();
    let mut out = vec![0.0; naux * nbf * nbf];
    for (p, ij, buf) in blocks {
        let pr = pairs.pair(ij);
        let p0 = aux.shells[p].ao_start;
        let (a0, b0) = (sh[pr.sa].ao_start, sh[pr.sb].ao_start);
        let nab = pr.nab();
        for x in 0..apairs.pair(p).na {
            let row = &mut out[(p0 + x) * nbf * nbf..(p0 + x + 1) * nbf * nbf];
            for ia in 0..pr.na {
                for ib in 0..pr.nb {
                    let v = buf[x * nab + ia * pr.nb + ib];
                    row[(a0 + ia) * nbf + b0 + ib] = v;
                    row[(b0 + ib) * nbf + a0 + ia] = v;
                }
            }
        }
    }
    out
}

/// First derivatives of the integrals of one shell quartet (ab|cd) with respect to the four
/// centers: out[((X * 3 + k) * nab + ab) * ncd + cd] = ∂(ab|cd)/∂X_k, X = A, B, C, D.
/// `bd` and `kd` are the derivative data of the bra and ket pairs.
pub fn eri_quartet_deriv1(
    bra: &ShellPair,
    bd: &PairDeriv,
    ket: &ShellPair,
    kd: &PairDeriv,
    out: &mut [f64],
    w: &mut DerivWork,
) {
    let (lab, lcd) = (bra.lab(), ket.lab());
    let (nab, ncd) = (bra.nab(), ket.nab());
    let blk = nab * ncd;
    let out = &mut out[..12 * blk];
    out.fill(0.0);
    if bra.nprim == 0 || ket.nprim == 0 {
        return;
    }
    let l = lab + lcd + 1;
    let (nhb, nhb1, nhk, nhk1) = (nherm(lab), nherm(lab + 1), nherm(lcd), nherm(lcd + 1));
    let wt = wtable(lab + 1, lcd + 1, 1);
    let sh = shift1();
    let DerivWork {
        fm,
        r,
        rwork,
        w: wm,
        t,
        tmp,
        op,
        ..
    } = w;
    grow(wm, nhb1 * nhk1);
    grow(t, nhb1 * ncd);
    grow(tmp, 3 * blk);
    let mut u_buf = vec![0.0; ket.nprim * nhk1 * nab];
    let u = &mut u_buf[..];
    let dp = &mut tmp[..3 * blk];
    dp.fill(0.0);
    for ip in 0..bra.nprim {
        let ep = bra.e_prim(ip);
        let eap = bd.ea_prim(ip);
        let t = &mut t[..nhb1 * ncd];
        t.fill(0.0);
        for kq in 0..ket.nprim {
            fill_w(
                l, bra.p[ip], bra.pc[ip], ket.p[kq], ket.pc[kq], wt, op, fm, r, rwork, wm,
            );
            wmul_acc(wm, nhk1, nhb1, nhk, ket.e_prim(kq), ncd, t);
            wmul_t_acc(
                wm,
                nhk1,
                nhb,
                nhk1,
                ep,
                nab,
                &mut u[kq * nhk1 * nab..(kq + 1) * nhk1 * nab],
            );
        }
        // dA[k][ab][cd] += EA[r'][k][ab] T[r'][cd]; dP[k][ab][cd] += E[r][ab] T[r+1_k][cd]
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
        for rr in 0..nhb {
            let erow = &ep[rr * nab..(rr + 1) * nab];
            for k in 0..3 {
                let r1 = sh[rr][k] as usize;
                let trow = &t[r1 * ncd..(r1 + 1) * ncd];
                let o = &mut dp[k * blk..(k + 1) * blk];
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
    // dC[l][ab][cd] = sum_q sum_s' U_q[s'][ab] EA_q[s'][l][cd]
    for kq in 0..ket.nprim {
        let uq = &u[kq * nhk1 * nab..(kq + 1) * nhk1 * nab];
        let eaq = kd.ea_prim(kq);
        for s in 0..nhk1 {
            let urow = &uq[s * nab..(s + 1) * nab];
            for k in 0..3 {
                let erow = &eaq[(s * 3 + k) * ncd..(s * 3 + k + 1) * ncd];
                let o = &mut out[(6 + k) * blk..(7 + k) * blk];
                for (ab, &uv) in urow.iter().enumerate() {
                    if uv == 0.0 {
                        continue;
                    }
                    for (x, &y) in o[ab * ncd..(ab + 1) * ncd].iter_mut().zip(erow) {
                        *x += uv * y;
                    }
                }
            }
        }
    }
    // B = P - A, D = -(P + C)
    for x in 0..3 * blk {
        let (a, p, c) = (out[x], dp[x], out[6 * blk + x]);
        out[3 * blk + x] = p - a;
        out[9 * blk + x] = -(p + c);
    }
}

/// Reference first derivatives of a quartet through derivative pseudo pairs and the plain
/// integral routine (same layout as [`eri_quartet_deriv1`]). Slow; for tests.
pub fn eri_quartet_deriv1_ref(
    bra: &ShellPair,
    bd: &PairDeriv,
    ket: &ShellPair,
    kd: &PairDeriv,
    out: &mut [f64],
) {
    let (nab, ncd) = (bra.nab(), ket.nab());
    let blk = nab * ncd;
    let mut w = EriWork::new();
    let mut buf = vec![0.0; 3 * blk];
    let c = &Operator::Coulomb;
    eri_quartet(&deriv_pair_a(bra, bd), ket, c, &mut out[..3 * blk], &mut w);
    eri_quartet(
        &deriv_pair_b(bra, bd),
        ket,
        c,
        &mut out[3 * blk..6 * blk],
        &mut w,
    );
    eri_quartet(bra, &deriv_pair_a(ket, kd), c, &mut buf, &mut w);
    // buf layout [ab][(k, c, d)] -> [k][ab][cd]
    for ab in 0..nab {
        for k in 0..3 {
            for cd in 0..ncd {
                out[(6 + k) * blk + ab * ncd + cd] = buf[ab * 3 * ncd + k * ncd + cd];
            }
        }
    }
    for x in 0..3 * blk {
        out[9 * blk + x] = -(out[x] + out[3 * blk + x] + out[6 * blk + x]);
    }
}

/// Reference second derivatives of a quartet: out[((XY * 9) + k * 3 + l) * nab * ncd + ab * ncd + cd]
/// = ∂²(ab|cd)/∂X_k∂Y_l for the center pairs XY of [`D2_CENTER_PAIRS`]. Needs order-2 pair
/// data. Slow (pseudo pairs); for tests and small systems.
/// For the two-electron operator `op` (a/r + b erf(omega r)/r).
#[allow(clippy::needless_range_loop, reason = "indexes several arrays in step")]
pub fn eri_quartet_deriv2_ref(
    bra: &ShellPair,
    bd: &PairDeriv,
    ket: &ShellPair,
    kd: &PairDeriv,
    op: &Operator,
    out: &mut [f64],
) {
    let (nab, ncd) = (bra.nab(), ket.nab());
    let blk = nab * ncd;
    let mut w = EriWork::new();
    let eri_quartet = |b: &ShellPair, k: &ShellPair, o: &mut [f64], w: &mut EriWork| {
        crate::eri4c::eri_quartet(b, k, op, o, w)
    };
    // blocks over (k,l) for each center pair among A,B,C (D eliminated afterwards)
    let mut m = vec![vec![0.0; 9 * blk]; 16];
    let idx = |x: usize, y: usize| x * 4 + y;
    let mut buf = vec![0.0; 9 * blk];
    for (kind, (x, y)) in [
        (D2Kind::AA, (0, 0)),
        (D2Kind::AB, (0, 1)),
        (D2Kind::BB, (1, 1)),
    ] {
        eri_quartet(&deriv2_pair(bra, bd, kind), ket, &mut m[idx(x, y)], &mut w);
    }
    // CC: ket pseudo pair, layout [ab][(k,l,c,d)]
    eri_quartet(bra, &deriv2_pair(ket, kd, D2Kind::AA), &mut buf, &mut w);
    for ab in 0..nab {
        for kl in 0..9 {
            for cd in 0..ncd {
                m[idx(2, 2)][kl * blk + ab * ncd + cd] = buf[ab * 9 * ncd + kl * ncd + cd];
            }
        }
    }
    // AC, BC: (3 x 3) with bra derivative k (outer) and ket derivative l
    for (x, bp) in [(0, deriv_pair_a(bra, bd)), (1, deriv_pair_b(bra, bd))] {
        let kp = deriv_pair_a(ket, kd);
        let mut b2 = vec![0.0; 9 * blk];
        eri_quartet(&bp, &kp, &mut b2, &mut w);
        // layout [(k, a, b)][(l, c, d)]
        for k in 0..3 {
            for ab in 0..nab {
                for l in 0..3 {
                    for cd in 0..ncd {
                        m[idx(x, 2)][(k * 3 + l) * blk + ab * ncd + cd] =
                            b2[(k * nab + ab) * 3 * ncd + l * ncd + cd];
                    }
                }
            }
        }
    }
    // transposes: M_YX[k][l] = M_XY[l][k]
    for (x, y) in [(0, 1), (0, 2), (1, 2)] {
        let src = m[idx(x, y)].clone();
        let dst = &mut m[idx(y, x)];
        for k in 0..3 {
            for l in 0..3 {
                dst[(k * 3 + l) * blk..(k * 3 + l + 1) * blk]
                    .copy_from_slice(&src[(l * 3 + k) * blk..(l * 3 + k + 1) * blk]);
            }
        }
    }
    // D = -(A + B + C)
    for x in 0..3 {
        for kl in 0..9 * blk {
            let v = -(m[idx(x, 0)][kl] + m[idx(x, 1)][kl] + m[idx(x, 2)][kl]);
            m[idx(x, 3)][kl] = v;
        }
    }
    for kl in 0..9 {
        let (k, l) = (kl / 3, kl % 3);
        for x in 0..blk {
            // M_DX[k][l] = M_XD[l][k]
            let v = -(m[idx(0, 3)][(l * 3 + k) * blk + x]
                + m[idx(1, 3)][(l * 3 + k) * blk + x]
                + m[idx(2, 3)][(l * 3 + k) * blk + x]);
            m[idx(3, 3)][kl * blk + x] = v;
        }
    }
    for (o, &(x, y)) in D2_CENTER_PAIRS.iter().enumerate() {
        out[o * 9 * blk..(o + 1) * 9 * blk].copy_from_slice(&m[idx(x, y)]);
    }
}
