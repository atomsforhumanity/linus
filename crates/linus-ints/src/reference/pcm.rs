//! The McMurchie-Davidson reference of the surface-charge integrals
//! ([`crate::points::pcm_ints`]): the potentials of densities at the points and the potential
//! matrices of charge sets, pair by pair and primitive by primitive, without screening. The
//! tests check the batched plans against them.

use crate::basis::BasisSet;
use crate::contract::AoMatrixMut;
use crate::core::hermite::nherm;
use crate::one_e::deriv::herm_dens;
use crate::one_e::scatter_sym_ao;
use crate::pairs::PairSet;
use crate::points::pcm_ints::GaussPoint;
use crate::potential::{coulomb_hermite, HermiteCtx, HermiteWork};
use linus_num::det::{add_into, nchunk_for, par_fold_det, DEFAULT_BUDGET};
use rayon::prelude::*;

/// The potentials of the densities at the points: out\[k\]\[g\] = sum_ab D^k_ab (ab|g)
/// (positive operator; D^k symmetric nbf x nbf row-major).
pub fn potential_points(
    basis: &BasisSet,
    pairs: &PairSet,
    pts: &[GaussPoint],
    ds: &[&[f64]],
) -> Vec<Vec<f64>> {
    let nbf = basis.nbf;
    let np = pts.len();
    let nd = ds.len();
    let sh = &basis.shells;
    let npairs = pairs.len();
    let acc = par_fold_det(
        npairs,
        nchunk_for(npairs, nd * np, DEFAULT_BUDGET),
        || vec![0.0; nd * np],
        |acc, kp| {
            let pr = pairs.pair(kp);
            let (s1, s2) = (&sh[pr.sa], &sh[pr.sb]);
            let lab = pr.lab();
            let nh = nherm(lab);
            let f = if pr.sa == pr.sb { 1.0 } else { 2.0 };
            let mut hd = vec![0.0; nd * nh];
            let mut r = vec![0.0; nh];
            let mut hw = HermiteWork::new(lab);
            for ip in 0..pr.nprim {
                for (k, d) in ds.iter().enumerate() {
                    herm_dens(
                        pr.e_prim(ip),
                        nh,
                        1,
                        d,
                        nbf,
                        s1.ao_start,
                        s2.ao_start,
                        pr.na,
                        pr.nb,
                        f,
                        &mut hd[k * nh..(k + 1) * nh],
                    );
                }
                for (g, pt) in pts.iter().enumerate() {
                    coulomb_hermite(
                        pt.r,
                        pt.zeta,
                        1.0,
                        &mut HermiteCtx {
                            l: lab,
                            p: pr.p[ip],
                            pc: pr.pc[ip],
                            r: &mut r,
                            work: &mut hw,
                        },
                    );
                    for k in 0..nd {
                        let h = &hd[k * nh..(k + 1) * nh];
                        let mut v = 0.0;
                        for (x, y) in h.iter().zip(&r) {
                            v += x * y;
                        }
                        acc[k * np + g] += v;
                    }
                }
            }
        },
        |a, b| add_into(a, &b),
    );
    (0..nd)
        .map(|k| acc[k * np..(k + 1) * np].to_vec())
        .collect()
}

/// The potential matrices of charge sets: out\[k\]_ab = sum_g q^k_g (ab|g) (symmetric nbf x nbf
/// row-major).
pub fn potential_matrices(
    basis: &BasisSet,
    pairs: &PairSet,
    pts: &[GaussPoint],
    qs: &[&[f64]],
) -> Vec<Vec<f64>> {
    let nbf = basis.nbf;
    let nd = qs.len();
    let sh = &basis.shells;
    let blocks: Vec<Vec<f64>> = pairs
        .pairs()
        .par_iter()
        .map(|pr| {
            let lab = pr.lab();
            let nh = nherm(lab);
            let nab = pr.nab();
            let mut out = vec![0.0; nd * nab];
            let mut acc = vec![0.0; nd * nh];
            let mut r = vec![0.0; nh];
            let mut hw = HermiteWork::new(lab);
            for ip in 0..pr.nprim {
                acc.fill(0.0);
                for (g, pt) in pts.iter().enumerate() {
                    coulomb_hermite(
                        pt.r,
                        pt.zeta,
                        1.0,
                        &mut HermiteCtx {
                            l: lab,
                            p: pr.p[ip],
                            pc: pr.pc[ip],
                            r: &mut r,
                            work: &mut hw,
                        },
                    );
                    for k in 0..nd {
                        let q = qs[k][g];
                        if q == 0.0 {
                            continue;
                        }
                        for (a, &y) in acc[k * nh..(k + 1) * nh].iter_mut().zip(&r) {
                            *a += q * y;
                        }
                    }
                }
                let e = pr.e_prim(ip);
                for k in 0..nd {
                    let o = &mut out[k * nab..(k + 1) * nab];
                    for rr in 0..nh {
                        let a = acc[k * nh + rr];
                        if a == 0.0 {
                            continue;
                        }
                        for (x, &ev) in o.iter_mut().zip(&e[rr * nab..(rr + 1) * nab]) {
                            *x += ev * a;
                        }
                    }
                }
            }
            out
        })
        .collect();
    let mut mats = vec![vec![0.0; nbf * nbf]; nd];
    for (pr, blk) in pairs.pairs().iter().zip(blocks) {
        let (a, b) = (&sh[pr.sa], &sh[pr.sb]);
        let nab = pr.nab();
        for (k, m) in mats.iter_mut().enumerate() {
            scatter_sym_ao(
                &mut AoMatrixMut::row_major(m, nbf),
                (a.ao_start, b.ao_start),
                (pr.na, pr.nb),
                &blk[k * nab..(k + 1) * nab],
            );
        }
    }
    mats
}
