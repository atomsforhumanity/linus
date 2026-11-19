//! One-electron integrals: overlap, kinetic energy, nuclear attraction (point charges),
//! multipoles, effective core potentials.
//!
//! - [`OneEPlan`] (`plan`): the matrices of a basis, their derivative matrices and their
//!   gradient and Hessian contractions with densities, into the caller's buffers;
//! - the kernels of one shell pair (this module's blocks, [`deriv`]'s derivative blocks);
//! - effective core potentials ([`ecp`], [`ecp_deriv`]).

pub mod deriv;
pub mod ecp;
pub mod ecp_deriv;
pub mod plan;

pub use plan::{OneEPlan, OneEWorkspace};

use crate::basis::{cart_components, Shell};
use crate::contract::AoMatrixMut;
use crate::core::boys;
use crate::core::hermite::{hermite_e_1d, nherm, r_tensor, ShellPair};
use crate::core::util::PairTransform;
use std::f64::consts::PI;

/// Scatter a shell-pair block (na x nb, row-major) into a symmetric matrix view: element
/// (a0 + i, b0 + j) and its transpose, in row order (on a diagonal block the lower triangle's
/// values stay).
pub(crate) fn scatter_sym_ao(
    m: &mut AoMatrixMut<'_>,
    (a0, b0): (usize, usize),
    (na, nb): (usize, usize),
    blk: &[f64],
) {
    for i in 0..na {
        for j in 0..nb {
            let v = blk[i * nb + j];
            *m.at(a0 + i, b0 + j) = v;
            *m.at(b0 + j, a0 + i) = v;
        }
    }
}

/// Overlap and kinetic energy blocks for a shell pair (in the shells' functions).
pub fn overlap_kinetic_block(a: &Shell, b: &Shell, s_out: &mut [f64], t_out: &mut [f64]) {
    let (la, lb) = (a.l, b.l);
    let lb2 = lb + 2;
    let nt = la + lb2 + 1;
    let len = (la + 1) * (lb2 + 1) * nt;
    let mut ex = vec![0.0; len];
    let mut ey = vec![0.0; len];
    let mut ez = vec![0.0; len];
    let ca = cart_components(la);
    let cb = cart_components(lb);
    let (nca, ncb) = (ca.len(), cb.len());
    let mut sc = vec![0.0; nca * ncb];
    let mut tc = vec![0.0; nca * ncb];
    let ab = [
        a.center[0] - b.center[0],
        a.center[1] - b.center[1],
        a.center[2] - b.center[2],
    ];
    // 1D overlaps S[i][j] for j <= lb + 2
    let mut s1 = [
        vec![0.0; (la + 1) * (lb2 + 1)],
        vec![0.0; (la + 1) * (lb2 + 1)],
        vec![0.0; (la + 1) * (lb2 + 1)],
    ];
    for (&ai, &ci) in a.exps.iter().zip(&a.coefs) {
        for (&bj, &cj) in b.exps.iter().zip(&b.coefs) {
            let p = ai + bj;
            hermite_e_1d(la, lb2, ai, bj, ab[0], &mut ex);
            hermite_e_1d(la, lb2, ai, bj, ab[1], &mut ey);
            hermite_e_1d(la, lb2, ai, bj, ab[2], &mut ez);
            let sq = (PI / p).sqrt();
            for (d, e) in [&ex, &ey, &ez].iter().enumerate() {
                for i in 0..=la {
                    for j in 0..=lb2 {
                        s1[d][i * (lb2 + 1) + j] = e[(i * (lb2 + 1) + j) * nt] * sq;
                    }
                }
            }
            let cc = ci * cj;
            let kin1 = |d: usize, i: usize, j: usize| -> f64 {
                let s = &s1[d];
                let w = lb2 + 1;
                let mut v =
                    -2.0 * bj * bj * s[i * w + j + 2] + bj * (2 * j + 1) as f64 * s[i * w + j];
                if j >= 2 {
                    v -= 0.5 * (j * (j - 1)) as f64 * s[i * w + j - 2];
                }
                v
            };
            let w = lb2 + 1;
            for (ia, pa) in ca.iter().enumerate() {
                for (ib, pb) in cb.iter().enumerate() {
                    let sx = s1[0][pa[0] * w + pb[0]];
                    let sy = s1[1][pa[1] * w + pb[1]];
                    let sz = s1[2][pa[2] * w + pb[2]];
                    let tx = kin1(0, pa[0], pb[0]);
                    let ty = kin1(1, pa[1], pb[1]);
                    let tz = kin1(2, pa[2], pb[2]);
                    sc[ia * ncb + ib] += cc * sx * sy * sz;
                    tc[ia * ncb + ib] += cc * (tx * sy * sz + sx * ty * sz + sx * sy * tz);
                }
            }
        }
    }
    let tf = PairTransform::of_shells(a, b);
    let mut tmp = Vec::new();
    tf.rows(&sc, 1, s_out, &mut tmp);
    tf.rows(&tc, 1, t_out, &mut tmp);
}

/// Dipole blocks <a| (r - origin)_k |b> of a shell pair for k = x, y, z: `out` holds the three
/// na x nb blocks, \[k\]\[a\]\[b\].
pub fn dipole_block(a: &Shell, b: &Shell, origin: [f64; 3], out: &mut [f64]) {
    let (la, lb) = (a.l, b.l);
    let nt = la + lb + 1;
    let len = (la + 1) * (lb + 1) * nt;
    let mut e = [vec![0.0; len], vec![0.0; len], vec![0.0; len]];
    let ca = cart_components(la);
    let cb = cart_components(lb);
    let (nca, ncb) = (ca.len(), cb.len());
    let mut cart = [
        vec![0.0; nca * ncb],
        vec![0.0; nca * ncb],
        vec![0.0; nca * ncb],
    ];
    for (&ai, &ci) in a.exps.iter().zip(&a.coefs) {
        for (&bj, &cj) in b.exps.iter().zip(&b.coefs) {
            let p = ai + bj;
            let sq = (PI / p).sqrt();
            let mut pcen = [0.0; 3];
            for d in 0..3 {
                hermite_e_1d(la, lb, ai, bj, a.center[d] - b.center[d], &mut e[d]);
                pcen[d] = (ai * a.center[d] + bj * b.center[d]) / p;
            }
            // 1D: S_ij = E0 sq ; M_ij = (E1 + X_PC E0) sq
            for (ia, pa) in ca.iter().enumerate() {
                for (ib, pb) in cb.iter().enumerate() {
                    let mut s = [0.0; 3];
                    let mut m = [0.0; 3];
                    for d in 0..3 {
                        let o = (pa[d] * (lb + 1) + pb[d]) * nt;
                        let e0 = e[d][o];
                        let e1 = if pa[d] + pb[d] >= 1 { e[d][o + 1] } else { 0.0 };
                        s[d] = e0 * sq;
                        m[d] = (e1 + (pcen[d] - origin[d]) * e0) * sq;
                    }
                    let cc = ci * cj;
                    cart[0][ia * ncb + ib] += cc * m[0] * s[1] * s[2];
                    cart[1][ia * ncb + ib] += cc * s[0] * m[1] * s[2];
                    cart[2][ia * ncb + ib] += cc * s[0] * s[1] * m[2];
                }
            }
        }
    }
    let n = a.nfunc() * b.nfunc();
    let (tf, mut tmp) = (PairTransform::of_shells(a, b), Vec::new());
    for d in 0..3 {
        tf.rows(&cart[d], 1, &mut out[d * n..(d + 1) * n], &mut tmp);
    }
}

/// Potential integrals of one shell pair at a batch of unit point charges:
/// out[ab * npts + g] = <a| 1/|r - r_g| |b>  (positive sign; no charge factor).
/// One Hermite R-tensor per (primitive pair, point), then a dense product over Hermite
/// functions: out = sum_prim E^T R (SHARK form).
pub fn potential_pair_points(
    pair: &ShellPair,
    x: &[f64],
    y: &[f64],
    z: &[f64],
    out: &mut [f64],
    rbuf: &mut Vec<f64>,
    work: &mut Vec<f64>,
) {
    let npts = x.len();
    let lab = pair.lab();
    let nh = nherm(lab);
    let nab = pair.nab();
    out[..nab * npts].fill(0.0);
    rbuf.resize(nh * npts, 0.0);
    work.resize(2 * nh + nh, 0.0);
    let mut fm = [0.0; boys::MAX_M + 1];
    for ip in 0..pair.nprim {
        let p = pair.p[ip];
        let pc = pair.pc[ip];
        let pref = 2.0 * PI / p;
        // R[r][g]
        for g in 0..npts {
            let pq = [pc[0] - x[g], pc[1] - y[g], pc[2] - z[g]];
            let t = p * (pq[0] * pq[0] + pq[1] * pq[1] + pq[2] * pq[2]);
            boys::boys(lab, t, &mut fm);
            let (rv, wk) = work.split_at_mut(nh);
            r_tensor(lab, p, pq, &fm, pref, rv, wk);
            for r in 0..nh {
                rbuf[r * npts + g] = rv[r];
            }
        }
        // out[ab][g] += sum_r E[r][ab] R[r][g]
        let e = pair.e_prim(ip);
        for r in 0..nh {
            let rrow = &rbuf[r * npts..(r + 1) * npts];
            let erow = &e[r * nab..(r + 1) * nab];
            for (ab, &ev) in erow.iter().enumerate() {
                if ev == 0.0 {
                    continue;
                }
                let o = &mut out[ab * npts..(ab + 1) * npts];
                for (ov, &rv) in o.iter_mut().zip(rrow) {
                    *ov += ev * rv;
                }
            }
        }
    }
}
