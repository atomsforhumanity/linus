//! Finite-difference validation of the analytic derivative integrals and contractions:
//! central differences (h = 1e-4 bohr, Richardson extrapolation) of our own integrals.

mod common;
use common::*;
use linus_ints::contract::{DensityTerm, DensityTerms};
use linus_ints::core::hermite::{build_pair, ShellView};
use linus_ints::core::pair_data;
use linus_ints::eri4c::deriv::{self, DerivWork};
use linus_ints::eri4c::{self, eri_quartet, EriWork};
use linus_ints::operator::Operator;
use linus_ints::reference::oneshot;
use linus_ints::reference::Dens3c;
use linus_ints::{one_e, BasisSet, DerivOrder, PairOptions, PairSet};

const TOL: f64 = 1e-9;

fn nuclear_charges(z: &[u32], positions: &[[f64; 3]]) -> Vec<(f64, [f64; 3])> {
    z.iter()
        .zip(positions)
        .map(|(&q, &c)| (q as f64, c))
        .collect()
}

#[test]
fn fd_one_electron_matrices() {
    let (def, z, positions) = test_system(true);
    let basis = make_basis(&def, &z, &positions);
    let n = basis.nbf;
    let n2 = n * n;
    let natom = z.len();
    let pairs = PairSet::build(&basis, PairOptions::with_order(DerivOrder::One));
    let charges = nuclear_charges(&z, &positions);
    let (ds, dt) = oneshot::overlap_kinetic_deriv(&basis);
    let dvc = oneshot::potential_deriv(&basis, &pairs, &charges);
    let (dv, dc) = dvc.split_at(3 * natom * n2);
    let origin = [0.3, -0.1, 0.2];
    let dm = oneshot::dipole_deriv(&basis, origin);
    let hd = oneshot::hcore_deriv(&basis, &pairs, &charges);
    let (mut es, mut et, mut ev, mut ec, mut ed, mut eh) =
        (0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64);
    for atom in 0..natom {
        for k in 0..3 {
            let x = 3 * atom + k;
            let num = fd(
                |h| {
                    let c = displaced(&positions, atom, k, h);
                    let b = make_basis(&def, &z, &c);
                    let p = PairSet::build(&b, PairOptions::default());
                    let (s, t) = oneshot::overlap_kinetic(&b);
                    let v = oneshot::potential(&b, &p, &charges); // charges fixed
                    let vn = oneshot::potential(&b, &p, &nuclear_charges(&z, &c)); // nuclei move too
                    let d = oneshot::dipole(&b, origin);
                    let mut out = s;
                    out.extend(t);
                    out.extend(v);
                    out.extend(d[0].iter().chain(&d[1]).chain(&d[2]));
                    let tt = oneshot::overlap_kinetic(&b).1;
                    out.extend(tt.iter().zip(&vn).map(|(a, b)| a + b));
                    out
                },
                H,
            );
            es = es.max(check(
                &format!("dS/dX[{x}]"),
                &ds[x * n2..(x + 1) * n2],
                &num[..n2],
                TOL,
            ));
            et = et.max(check(
                &format!("dT/dX[{x}]"),
                &dt[x * n2..(x + 1) * n2],
                &num[n2..2 * n2],
                TOL,
            ));
            ev = ev.max(check(
                &format!("dV/dX[{x}] (basis)"),
                &dv[x * n2..(x + 1) * n2],
                &num[2 * n2..3 * n2],
                TOL,
            ));
            for c in 0..3 {
                let o = (x * 3 + c) * n2;
                ed = ed.max(check(
                    &format!("dM{c}/dX[{x}]"),
                    &dm[o..o + n2],
                    &num[(3 + c) * n2..(4 + c) * n2],
                    TOL,
                ));
            }
            eh = eh.max(check(
                &format!("dh/dX[{x}]"),
                &hd[x * n2..(x + 1) * n2],
                &num[6 * n2..7 * n2],
                TOL,
            ));
            // operator center only
            let numc = fd(
                |h| {
                    let mut ch = charges.clone();
                    ch[atom].1[k] += h;
                    oneshot::potential(&basis, &pairs, &ch)
                },
                H,
            );
            ec = ec.max(check(
                &format!("dV/dC[{x}]"),
                &dc[x * n2..(x + 1) * n2],
                &numc,
                TOL,
            ));
        }
    }
    println!("SUMMARY fd 1e: S {es:.2e} T {et:.2e} V(basis) {ev:.2e} V(op) {ec:.2e} dipole {ed:.2e} hcore {eh:.2e}");
}

#[test]
fn fd_one_electron_blocks_second() {
    // second derivatives of S and T blocks vs FD of first-derivative blocks (moving shell a)
    let (def, z, positions) = test_system(true);
    let basis = make_basis(&def, &z, &positions);
    let sh = &basis.shells;
    let mut worst: f64 = 0.0;
    for i in 0..sh.len() {
        for j in 0..sh.len() {
            let (a, b) = (&sh[i], &sh[j]);
            let n = a.nfunc() * b.nfunc();
            let mut s2 = vec![0.0; 9 * n];
            let mut t2 = vec![0.0; 9 * n];
            one_e::deriv::overlap_kinetic_deriv2_block(a, b, &mut s2, &mut t2);
            let mut s1 = vec![0.0; 3 * n];
            let mut t1 = vec![0.0; 3 * n];
            one_e::deriv::overlap_kinetic_deriv1_block(a, b, &mut s1, &mut t1);
            for l in 0..3 {
                // d/dA_l of the k-th first derivative block
                let num = fd(
                    |h| {
                        let mut am = a.clone();
                        am.center[l] += h;
                        let mut s = vec![0.0; 3 * n];
                        let mut t = vec![0.0; 3 * n];
                        one_e::deriv::overlap_kinetic_deriv1_block(&am, b, &mut s, &mut t);
                        s.extend(t);
                        s
                    },
                    H,
                );
                for k in 0..3 {
                    let (ns, nt) = (&num[k * n..(k + 1) * n], &num[(3 + k) * n..(4 + k) * n]);
                    worst = worst.max(check(
                        &format!("d2S/dA{k}dA{l} ({i},{j})"),
                        &s2[(k * 3 + l) * n..(k * 3 + l + 1) * n],
                        ns,
                        TOL,
                    ));
                    worst = worst.max(check(
                        &format!("d2T/dA{k}dA{l} ({i},{j})"),
                        &t2[(k * 3 + l) * n..(k * 3 + l + 1) * n],
                        nt,
                        TOL,
                    ));
                }
                // first derivative blocks themselves vs FD of S, T blocks
                let num1 = fd(
                    |h| {
                        let mut am = a.clone();
                        am.center[l] += h;
                        let mut s = vec![0.0; n];
                        let mut t = vec![0.0; n];
                        one_e::overlap_kinetic_block(&am, b, &mut s, &mut t);
                        s.extend(t);
                        s
                    },
                    H,
                );
                worst = worst.max(check(
                    &format!("dS/dA{l} ({i},{j})"),
                    &s1[l * n..(l + 1) * n],
                    &num1[..n],
                    TOL,
                ));
                worst = worst.max(check(
                    &format!("dT/dA{l} ({i},{j})"),
                    &t1[l * n..(l + 1) * n],
                    &num1[n..],
                    TOL,
                ));
            }
        }
    }
    println!("SUMMARY fd S/T blocks (1st, 2nd): {worst:.2e}");
}

#[allow(clippy::needless_range_loop, reason = "indexes several arrays in step")]
fn shifted_pair(
    basis: &BasisSet,
    i: usize,
    j: usize,
    mv: [[f64; 3]; 2],
) -> linus_ints::core::hermite::ShellPair {
    let mut a = ShellView::from_shell(&basis.shells[i]);
    let mut b = ShellView::from_shell(&basis.shells[j]);
    for k in 0..3 {
        a.center[k] += mv[0][k];
        b.center[k] += mv[1][k];
    }
    build_pair(&a, &b, i, j)
}

#[allow(clippy::needless_range_loop, reason = "indexes several arrays in step")]
fn shifted_pair_deriv(
    basis: &BasisSet,
    i: usize,
    j: usize,
    mv: [[f64; 3]; 2],
) -> (
    linus_ints::core::hermite::ShellPair,
    linus_ints::core::deriv::PairDeriv,
) {
    let mut a = ShellView::from_shell(&basis.shells[i]);
    let mut b = ShellView::from_shell(&basis.shells[j]);
    for k in 0..3 {
        a.center[k] += mv[0][k];
        b.center[k] += mv[1][k];
    }
    let (sp, pd) = pair_data::build(&a, &b, i, j, 2, pair_data::PRIM_PAIR_CUT);
    (sp, pd.unwrap())
}

/// Quartets (i j | k l) of shells to test: every shell appears in every position with a
/// few partners.
fn quartet_list(ns: usize) -> Vec<[usize; 4]> {
    let mut v = Vec::new();
    for i in 0..ns {
        for j in 0..ns {
            v.push([i, j, (i + 2) % ns, (j + 5) % ns]);
            v.push([(i + 3) % ns, (j + 1) % ns, i, j]);
        }
    }
    v
}

#[test]
fn fd_eri4_first_derivative_blocks() {
    let (def, z, positions) = test_system(true);
    let basis = make_basis(&def, &z, &positions);
    let ns = basis.shells.len();
    let zero = [[0.0; 3]; 2];
    let mut w = DerivWork::new();
    let mut ew = EriWork::new();
    let (mut worst_fd, mut worst_ref) = (0.0f64, 0.0f64);
    for q in quartet_list(ns) {
        let (bra, bd) = shifted_pair_deriv(&basis, q[0], q[1], zero);
        let (ket, kd) = shifted_pair_deriv(&basis, q[2], q[3], zero);
        let blk = bra.nab() * ket.nab();
        let mut d1 = vec![0.0; 12 * blk];
        linus_ints::reference::eri_quartet_deriv1(&bra, &bd, &ket, &kd, &mut d1, &mut w);
        let mut dr = vec![0.0; 12 * blk];
        linus_ints::reference::eri_quartet_deriv1_ref(&bra, &bd, &ket, &kd, &mut dr);
        worst_ref = worst_ref.max(check(&format!("deriv1 vs ref {q:?}"), &d1, &dr, 1e-12));
        for x in 0..4 {
            for k in 0..3 {
                let num = fd(
                    |h| {
                        let mut mv = [[[0.0; 3]; 2]; 2];
                        mv[x / 2][x % 2][k] = h;
                        let b = shifted_pair(&basis, q[0], q[1], mv[0]);
                        let kt = shifted_pair(&basis, q[2], q[3], mv[1]);
                        let mut out = vec![0.0; blk];
                        eri_quartet(&b, &kt, &Operator::Coulomb, &mut out, &mut ew);
                        out
                    },
                    H,
                );
                worst_fd = worst_fd.max(check(
                    &format!("d(ab|cd)/d{x}{k} {q:?}"),
                    &d1[(x * 3 + k) * blk..(x * 3 + k + 1) * blk],
                    &num,
                    TOL,
                ));
            }
        }
    }
    println!("SUMMARY fd ERI 1st blocks: vs FD {worst_fd:.2e}, fast vs reference {worst_ref:.2e}");
}

#[test]
fn fd_eri4_second_derivative_blocks() {
    let (def, z, positions) = test_system(false);
    let basis = make_basis(&def, &z, &positions);
    let ns = basis.shells.len();
    let mut w = DerivWork::new();
    let mut worst: f64 = 0.0;
    for (n, q) in quartet_list(ns).into_iter().enumerate() {
        if n % 3 != 0 {
            continue;
        }
        let zero = [[0.0; 3]; 2];
        let (bra, bd) = shifted_pair_deriv(&basis, q[0], q[1], zero);
        let (ket, kd) = shifted_pair_deriv(&basis, q[2], q[3], zero);
        let blk = bra.nab() * ket.nab();
        let mut d2 = vec![0.0; 90 * blk];
        linus_ints::reference::eri_quartet_deriv2_ref(
            &bra,
            &bd,
            &ket,
            &kd,
            &linus_ints::operator::Operator::Coulomb,
            &mut d2,
        );
        // FD of first derivatives w.r.t. each of A, B, C (D by invariance inside deriv2)
        for y in 0..4 {
            for l in 0..3 {
                let num = fd(
                    |h| {
                        let mut mv = [[[0.0; 3]; 2]; 2];
                        mv[y / 2][y % 2][l] = h;
                        let (b, bdd) = shifted_pair_deriv(&basis, q[0], q[1], mv[0]);
                        let (kt, kdd) = shifted_pair_deriv(&basis, q[2], q[3], mv[1]);
                        let mut out = vec![0.0; 12 * blk];
                        linus_ints::reference::eri_quartet_deriv1(
                            &b, &bdd, &kt, &kdd, &mut out, &mut w,
                        );
                        out
                    },
                    H,
                );
                for (o, &(xx, yy)) in deriv::D2_CENTER_PAIRS.iter().enumerate() {
                    // block (X, Y) [k][l] = d/dY_l (d/dX_k); compare where Y == y
                    let (xc, yc, transposed) = if yy == y {
                        (xx, yy, false)
                    } else if xx == y {
                        (yy, xx, true)
                    } else {
                        continue;
                    };
                    let _ = yc;
                    for k in 0..3 {
                        let (kk, ll) = if transposed { (l, k) } else { (k, l) };
                        let ana = &d2[(o * 9 + kk * 3 + ll) * blk..(o * 9 + kk * 3 + ll + 1) * blk];
                        let numk = &num[(xc * 3 + k) * blk..(xc * 3 + k + 1) * blk];
                        worst = worst.max(check(
                            &format!("d2(ab|cd) {q:?} X{xc}{k} Y{y}{l}"),
                            ana,
                            numk,
                            TOL,
                        ));
                    }
                }
            }
        }
    }
    println!("SUMMARY fd ERI 2nd blocks: {worst:.2e}");
}

/// Full nuclear-derivative ERI tensors d(ab|cd)/dX_{atom,k}, (3 natom) x nbf^4, assembled
/// from raw quartet derivatives over all ordered shell quartets.
fn full_eri_deriv(basis: &BasisSet) -> Vec<f64> {
    let n = basis.nbf;
    let n4 = n * n * n * n;
    let sh = &basis.shells;
    let ns = sh.len();
    let natom = basis.atom_shells.len();
    let mut out = vec![0.0; 3 * natom * n4];
    let prs: Vec<Vec<_>> = (0..ns)
        .map(|i| {
            (0..ns)
                .map(|j| shifted_pair_deriv(basis, i, j, [[0.0; 3]; 2]))
                .collect()
        })
        .collect();
    let mut w = DerivWork::new();
    for i in 0..ns {
        for j in 0..ns {
            for k in 0..ns {
                for l in 0..ns {
                    let (bra, bd) = &prs[i][j];
                    let (ket, kd) = &prs[k][l];
                    let blk = bra.nab() * ket.nab();
                    let mut d = vec![0.0; 12 * blk];
                    linus_ints::reference::eri_quartet_deriv1(bra, bd, ket, kd, &mut d, &mut w);
                    let s = [i, j, k, l];
                    for x in 0..4 {
                        let at = sh[s[x]].atom;
                        for c in 0..3 {
                            let o = (3 * at + c) * n4;
                            for a in 0..bra.na {
                                for b in 0..bra.nb {
                                    for cc in 0..ket.na {
                                        for dd in 0..ket.nb {
                                            let v = d[(x * 3 + c) * blk
                                                + (a * bra.nb + b) * ket.nab()
                                                + cc * ket.nb
                                                + dd];
                                            let (p, q, r, t) = (
                                                sh[i].ao_start + a,
                                                sh[j].ao_start + b,
                                                sh[k].ao_start + cc,
                                                sh[l].ao_start + dd,
                                            );
                                            out[o + ((p * n + q) * n + r) * n + t] += v;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    out
}

/// Hessian contraction sum_abcd d²(ab|cd)/dXdY Γ_abcd from raw second-derivative blocks of
/// every ordered shell quartet (exact reference for the 4c Hessian).
fn hess_from_raw(basis: &BasisSet, terms: &DensityTerms) -> Vec<f64> {
    let n = basis.nbf;
    let sh = &basis.shells;
    let ns = sh.len();
    let n3 = 3 * basis.atom_shells.len();
    let mut h = vec![0.0; n3 * n3];
    let prs: Vec<Vec<_>> = (0..ns)
        .map(|i| {
            (0..ns)
                .map(|j| shifted_pair_deriv(basis, i, j, [[0.0; 3]; 2]))
                .collect()
        })
        .collect();
    let gam = |a: usize, b: usize, c: usize, d: usize| {
        let mut g = 0.0;
        for &DensityTerm {
            coef: f,
            left: d1,
            right: d2,
        } in &terms.coulomb
        {
            g += f * d1[a * n + b] * d2[c * n + d];
        }
        for &DensityTerm {
            coef: f,
            left: d1,
            right: d2,
        } in &terms.exchange
        {
            g += f * d1[a * n + c] * d2[b * n + d];
        }
        g
    };
    for i in 0..ns {
        for j in 0..ns {
            for k in 0..ns {
                for l in 0..ns {
                    let (bra, bd) = &prs[i][j];
                    let (ket, kd) = &prs[k][l];
                    let (nab, ncd) = (bra.nab(), ket.nab());
                    let blk = nab * ncd;
                    let mut d2 = vec![0.0; 90 * blk];
                    linus_ints::reference::eri_quartet_deriv2_ref(
                        bra,
                        bd,
                        ket,
                        kd,
                        &linus_ints::operator::Operator::Coulomb,
                        &mut d2,
                    );
                    let s = [i, j, k, l];
                    let mut gq = vec![0.0; blk];
                    for a in 0..bra.na {
                        for b in 0..bra.nb {
                            for c in 0..ket.na {
                                for d in 0..ket.nb {
                                    gq[(a * bra.nb + b) * ncd + c * ket.nb + d] = gam(
                                        sh[i].ao_start + a,
                                        sh[j].ao_start + b,
                                        sh[k].ao_start + c,
                                        sh[l].ao_start + d,
                                    );
                                }
                            }
                        }
                    }
                    for (o, &(x, y)) in deriv::D2_CENTER_PAIRS.iter().enumerate() {
                        let (ax, ay) = (sh[s[x]].atom, sh[s[y]].atom);
                        for kl in 0..9 {
                            let v: f64 = d2[(o * 9 + kl) * blk..(o * 9 + kl + 1) * blk]
                                .iter()
                                .zip(&gq)
                                .map(|(p, q)| p * q)
                                .sum();
                            let (kk, ll) = (kl / 3, kl % 3);
                            h[(3 * ax + kk) * n3 + 3 * ay + ll] += v;
                            if x != y {
                                h[(3 * ay + ll) * n3 + 3 * ax + kk] += v;
                            }
                        }
                    }
                }
            }
        }
    }
    h
}

fn eri_energy(basis: &BasisSet, terms: &DensityTerms) -> f64 {
    let pairs = PairSet::build(basis, PairOptions::default());
    let g = eri4c::eri_full(basis, &pairs, &linus_ints::operator::Operator::Coulomb);
    let n = basis.nbf;
    let mut e = 0.0;
    for a in 0..n {
        for b in 0..n {
            for c in 0..n {
                for d in 0..n {
                    let v = g[((a * n + b) * n + c) * n + d];
                    for &DensityTerm {
                        coef: f,
                        left: d1,
                        right: d2,
                    } in &terms.coulomb
                    {
                        e += f * v * d1[a * n + b] * d2[c * n + d];
                    }
                    for &DensityTerm {
                        coef: f,
                        left: d1,
                        right: d2,
                    } in &terms.exchange
                    {
                        e += f * v * d1[a * n + c] * d2[b * n + d];
                    }
                }
            }
        }
    }
    e
}

#[test]
fn fd_eri4_gradient_and_hessian_contractions() {
    let (def, z, positions) = test_system(false);
    let basis = make_basis(&def, &z, &positions);
    let n = basis.nbf;
    let (d1, d2, d3) = (random_sym(n, 1), random_sym(n, 2), random_sym(n, 3));
    let terms = DensityTerms::default()
        .coulomb(0.7, &d1, &d1)
        .coulomb(0.3, &d1, &d2)
        .exchange(-0.4, &d1, &d1)
        .exchange(0.2, &d2, &d3);
    let natom = z.len();
    let grad_at = |c: &[[f64; 3]]| {
        let b = make_basis(&def, &z, c);
        let p = PairSet::build(&b, PairOptions::with_order(DerivOrder::One));
        flat(&oneshot::grad_4c(
            &b,
            &p,
            &terms,
            0.0,
            &linus_ints::operator::Operator::Coulomb,
        ))
    };
    let g = grad_at(&positions);
    let mut num = Vec::new();
    for atom in 0..natom {
        for k in 0..3 {
            num.push(
                fd(
                    |h| {
                        vec![eri_energy(
                            &make_basis(&def, &z, &displaced(&positions, atom, k, h)),
                            &terms,
                        )]
                    },
                    h_step(),
                )[0],
            );
        }
    }
    let eg = check("4c gradient contraction", &g, &num, TOL);
    // exact check: contraction of the fully assembled derivative tensors
    let dg = full_eri_deriv(&basis);
    let n4 = n * n * n * n;
    let mut gex = vec![0.0; 3 * natom];
    for x in 0..3 * natom {
        let t = &dg[x * n4..(x + 1) * n4];
        let mut s = 0.0;
        for a in 0..n {
            for b in 0..n {
                for c in 0..n {
                    for d in 0..n {
                        let v = t[((a * n + b) * n + c) * n + d];
                        for &DensityTerm {
                            coef: f,
                            left: d1,
                            right: d2,
                        } in &terms.coulomb
                        {
                            s += f * v * d1[a * n + b] * d2[c * n + d];
                        }
                        for &DensityTerm {
                            coef: f,
                            left: d1,
                            right: d2,
                        } in &terms.exchange
                        {
                            s += f * v * d1[a * n + c] * d2[b * n + d];
                        }
                    }
                }
            }
        }
        gex[x] = s;
    }
    check("4c gradient contraction vs raw tensors", &g, &gex, 1e-12);
    // Hessian
    let pairs = PairSet::build(&basis, PairOptions::with_order(DerivOrder::Two));
    let hss = oneshot::hess_4c(
        &basis,
        &pairs,
        &terms,
        0.0,
        &linus_ints::operator::Operator::Coulomb,
    );
    let n3 = 3 * natom;
    let mut numh = vec![0.0; n3 * n3];
    for atom in 0..natom {
        for k in 0..3 {
            let col = fd(|h| grad_at(&displaced(&positions, atom, k, h)), H);
            for i in 0..n3 {
                numh[i * n3 + 3 * atom + k] = col[i];
            }
        }
    }
    let eh = check("4c Hessian contraction", &hss, &numh, TOL);
    check(
        "4c Hessian contraction vs raw blocks",
        &hss,
        &hess_from_raw(&basis, &terms),
        1e-12,
    );
    println!("SUMMARY fd 4c contractions: gradient {eg:.2e} Hessian {eh:.2e}");
}

/// J[D] and K[D] from the full ERI tensor.
fn jk_full(basis: &BasisSet, d: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let pairs = PairSet::build(basis, PairOptions::default());
    let g = eri4c::eri_full(basis, &pairs, &linus_ints::operator::Operator::Coulomb);
    let n = basis.nbf;
    let (mut j, mut k) = (vec![0.0; n * n], vec![0.0; n * n]);
    for m in 0..n {
        for nn in 0..n {
            let (mut sj, mut sk) = (0.0, 0.0);
            for l in 0..n {
                for s in 0..n {
                    sj += g[((m * n + nn) * n + l) * n + s] * d[l * n + s];
                    sk += g[((m * n + l) * n + nn) * n + s] * d[l * n + s];
                }
            }
            j[m * n + nn] = sj;
            k[m * n + nn] = sk;
        }
    }
    (j, k)
}

#[test]
fn fd_jk_derivative_matrices() {
    let (def, z, positions) = test_system(false);
    let basis = make_basis(&def, &z, &positions);
    let n = basis.nbf;
    let n2 = n * n;
    let (d1, d2) = (random_sym(n, 21), random_sym(n, 22));
    let pairs = PairSet::build(&basis, PairOptions::default());
    let (jx, kx) = linus_ints::eri4c::jk_deriv::jk_deriv(
        &basis,
        &pairs,
        &[&d1, &d2],
        true,
        true,
        0.0,
        &linus_ints::operator::Operator::Coulomb,
    );
    let (mut ej, mut ek) = (0.0f64, 0.0f64);
    for atom in 0..z.len() {
        for k in 0..3 {
            let x = 3 * atom + k;
            let num = fd(
                |h| {
                    let b = make_basis(&def, &z, &displaced(&positions, atom, k, h));
                    let (j1, k1) = jk_full(&b, &d1);
                    let (j2, k2) = jk_full(&b, &d2);
                    j1.into_iter().chain(k1).chain(j2).chain(k2).collect()
                },
                H,
            );
            for (di, o) in [(0, 0), (1, 2)] {
                ej = ej.max(check(
                    &format!("J^x[D{di}] x={x}"),
                    &jx[di][x * n2..(x + 1) * n2],
                    &num[o * n2..(o + 1) * n2],
                    TOL,
                ));
                ek = ek.max(check(
                    &format!("K^x[D{di}] x={x}"),
                    &kx[di][x * n2..(x + 1) * n2],
                    &num[(o + 1) * n2..(o + 2) * n2],
                    TOL,
                ));
            }
        }
    }
    println!("SUMMARY fd J^x/K^x: J {ej:.2e} K {ek:.2e}");
    // the one-atom-per-task path (natom >= 2 threads) must agree
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap();
    let (j1, k1) = pool.install(|| {
        linus_ints::eri4c::jk_deriv::jk_deriv(
            &basis,
            &pairs,
            &[&d1, &d2],
            true,
            true,
            0.0,
            &linus_ints::operator::Operator::Coulomb,
        )
    });
    for di in 0..2 {
        check("J^x per-atom path", &j1[di], &jx[di], 1e-13);
        check("K^x per-atom path", &k1[di], &kx[di], 1e-13);
    }
}

fn aux_def() -> linus_ints::BasisDef {
    basis_def(&[
        (
            8,
            vec![
                (0, vec![3.0], vec![1.0]),
                (1, vec![1.5], vec![1.0]),
                (2, vec![1.2], vec![1.0]),
                (3, vec![1.0], vec![1.0]),
                (4, vec![0.8], vec![1.0]),
            ],
        ),
        (
            1,
            vec![
                (0, vec![1.0, 0.3], vec![0.6, 0.5]),
                (1, vec![0.9], vec![1.0]),
            ],
        ),
        (
            7,
            vec![(0, vec![1.3], vec![1.0]), (2, vec![0.7], vec![1.0])],
        ),
    ])
}

#[test]
fn fd_df_gradient_and_hessian_contractions() {
    let (def, z, positions) = test_system(false);
    let adef = aux_def();
    let basis = make_basis(&def, &z, &positions);
    let aux = make_basis(&adef, &z, &positions);
    let (n, naux) = (basis.nbf, aux.nbf);
    let natom = z.len();
    // Γ_Pmn: random, symmetric in mn; separable part c d_P D_mn; γ_PQ random symmetric
    let mut full = vec![0.0; naux * n * n];
    for p in 0..naux {
        let m = random_sym(n, 100 + p as u64);
        full[p * n * n..(p + 1) * n * n].copy_from_slice(&m);
    }
    let dvec = random_vec(naux, 7);
    let dmat = random_sym(n, 8);
    let gam2 = random_sym(naux, 9);
    let dens = Dens3c {
        full: Some(&full),
        aux_shells: 0..aux.shells.len(),
        separable: vec![(0.6, &dvec, &dmat)],
    };
    let e3 = |c: &[[f64; 3]]| {
        let b = make_basis(&def, &z, c);
        let a = make_basis(&adef, &z, c);
        let p = PairSet::build(&b, PairOptions::default());
        let ap = PairSet::aux(&a, PairOptions::default());
        let j3 = linus_ints::reference::eri_3c(&b, &p, &a, &ap);
        let j2 = linus_ints::reference::eri_2c(&a, &ap);
        let mut e = 0.0;
        for pp in 0..naux {
            for mn in 0..n * n {
                e += j3[pp * n * n + mn] * (full[pp * n * n + mn] + 0.6 * dvec[pp] * dmat[mn]);
            }
        }
        let e2: f64 = j2.iter().zip(&gam2).map(|(x, y)| x * y).sum();
        vec![e, e2]
    };
    let grads = |c: &[[f64; 3]], order: usize| {
        let b = make_basis(&def, &z, c);
        let a = make_basis(&adef, &z, c);
        let p = PairSet::build(
            &b,
            PairOptions::with_order(DerivOrder::try_from(order).unwrap()),
        );
        let ap = PairSet::aux(
            &a,
            PairOptions::with_order(DerivOrder::try_from(order).unwrap()),
        );
        let g3 = flat(&linus_ints::reference::eri_grad_3c(
            &b,
            &p,
            &a,
            &ap,
            &dens,
            0.0,
            &linus_ints::operator::Operator::Coulomb,
        ));
        let g2 = flat(&linus_ints::reference::oneshot::grad_2c(
            &a,
            &ap,
            &gam2,
            0.0,
            &linus_ints::operator::Operator::Coulomb,
        ));
        (g3, g2, b, a, p, ap)
    };
    let (g3, g2, b, a, p, ap) = grads(&positions, 2);
    let mut n3 = Vec::new();
    let mut n2 = Vec::new();
    for atom in 0..natom {
        for k in 0..3 {
            let v = fd(|h| e3(&displaced(&positions, atom, k, h)), H);
            n3.push(v[0]);
            n2.push(v[1]);
        }
    }
    let e_g3 = check("3c gradient contraction", &g3, &n3, TOL);
    let e_g2 = check("2c gradient contraction", &g2, &n2, TOL);
    let h3 = linus_ints::reference::eri_hess_3c(
        &b,
        &p,
        &a,
        &ap,
        &dens,
        0.0,
        &linus_ints::operator::Operator::Coulomb,
    );
    let h2 = linus_ints::reference::oneshot::hess_2c(
        &a,
        &ap,
        &gam2,
        0.0,
        &linus_ints::operator::Operator::Coulomb,
    );
    let m = 3 * natom;
    let (mut nh3, mut nh2) = (vec![0.0; m * m], vec![0.0; m * m]);
    for atom in 0..natom {
        for k in 0..3 {
            let col = fd(
                |h| {
                    let (g3, g2, ..) = grads(&displaced(&positions, atom, k, h), 1);
                    g3.into_iter().chain(g2).collect()
                },
                H,
            );
            for i in 0..m {
                nh3[i * m + 3 * atom + k] = col[i];
                nh2[i * m + 3 * atom + k] = col[m + i];
            }
        }
    }
    let e_h3 = check("3c Hessian contraction", &h3, &nh3, TOL);
    let e_h2 = check("2c Hessian contraction", &h2, &nh2, TOL);
    println!("SUMMARY fd DF contractions: 3c grad {e_g3:.2e} 2c grad {e_g2:.2e} 3c hess {e_h3:.2e} 2c hess {e_h2:.2e}");
}

#[test]
fn fd_one_electron_contractions() {
    let (def, z, positions) = test_system(true);
    let basis = make_basis(&def, &z, &positions);
    let n = basis.nbf;
    let natom = z.len();
    let m = 3 * natom;
    let (dd, ww) = (random_sym(n, 11), random_sym(n, 12));
    let origin = [0.2, 0.1, -0.3];
    // energy-like scalars at displaced geometries: tr(W S), tr(D T), tr(D V) (nuclei move),
    // dipole components tr(D M_c)
    let scal = |c: &[[f64; 3]]| {
        let b = make_basis(&def, &z, c);
        let p = PairSet::build(&b, PairOptions::default());
        let (s, t) = oneshot::overlap_kinetic(&b);
        let v = oneshot::potential(&b, &p, &nuclear_charges(&z, c));
        let dm = oneshot::dipole(&b, origin);
        let tr = |x: &[f64], y: &[f64]| x.iter().zip(y).map(|(a, b)| a * b).sum::<f64>();
        vec![
            tr(&ww, &s),
            tr(&dd, &t),
            tr(&dd, &v),
            tr(&dd, &dm[0]),
            tr(&dd, &dm[1]),
            tr(&dd, &dm[2]),
        ]
    };
    let grads = |c: &[[f64; 3]], order: usize| {
        let b = make_basis(&def, &z, c);
        let p = PairSet::build(
            &b,
            PairOptions::with_order(DerivOrder::try_from(order).unwrap()),
        );
        let gs = flat(&oneshot::overlap_kinetic_grad(&b, Some(&ww), None));
        let gt = flat(&oneshot::overlap_kinetic_grad(&b, None, Some(&dd)));
        let gv = flat(&oneshot::nuclear_attraction_grad(
            &b,
            &p,
            &nuclear_charges(&z, c),
            &dd,
        ));
        let gd: Vec<f64> = oneshot::dipole_grad(&b, origin, &dd)
            .iter()
            .flat_map(|x| x.iter().flat_map(|y| y.iter().cloned()).collect::<Vec<_>>())
            .collect();
        (gs, gt, gv, gd)
    };
    let (gs, gt, gv, gd) = grads(&positions, 1);
    let mut num = vec![Vec::new(); 6];
    for atom in 0..natom {
        for k in 0..3 {
            let v = fd(|h| scal(&displaced(&positions, atom, k, h)), H);
            for i in 0..6 {
                num[i].push(v[i]);
            }
        }
    }
    let es = check("grad tr(W dS)", &gs, &num[0], TOL);
    let et = check("grad tr(D dT)", &gt, &num[1], TOL);
    let ev = check("grad tr(D dV)", &gv, &num[2], TOL);
    // dipole: gd[(atom*3+k)*3 + c]
    let mut numd = vec![0.0; 9 * natom];
    for x in 0..m {
        for c in 0..3 {
            numd[x * 3 + c] = num[3 + c][x];
        }
    }
    let ed = check("grad tr(D dM)", &gd, &numd, TOL);
    // Hessians vs FD of gradients
    let pairs = PairSet::build(&basis, PairOptions::with_order(DerivOrder::Two));
    let hs = oneshot::overlap_kinetic_hess(&basis, Some(&ww), None);
    let ht = oneshot::overlap_kinetic_hess(&basis, None, Some(&dd));
    let hv =
        oneshot::nuclear_attraction_hess(&basis, &pairs, &nuclear_charges(&z, &positions), &dd);
    let mut nh = vec![vec![0.0; m * m]; 3];
    for atom in 0..natom {
        for k in 0..3 {
            let col = fd(
                |h| {
                    let (a, b, c, _) = grads(&displaced(&positions, atom, k, h), 1);
                    a.into_iter().chain(b).chain(c).collect()
                },
                H,
            );
            for t in 0..3 {
                for i in 0..m {
                    nh[t][i * m + 3 * atom + k] = col[t * m + i];
                }
            }
        }
    }
    let ehs = check("hess tr(W d2S)", &hs, &nh[0], TOL);
    let eht = check("hess tr(D d2T)", &ht, &nh[1], TOL);
    let ehv = check("hess tr(D d2V)", &hv, &nh[2], TOL);
    // point charges away from atoms: charge-position gradient and Hessian blocks
    let ch = vec![(0.7, [0.5, 0.4, -0.9]), (-0.3, [-1.0, 0.2, 0.6])];
    let (_, gcq) = oneshot::potential_grad(&basis, &pairs, &ch, &dd);
    let mut numq = Vec::new();
    for ic in 0..2 {
        for k in 0..3 {
            numq.push(
                fd(
                    |h| {
                        let mut c2 = ch.clone();
                        c2[ic].1[k] += h;
                        let v = oneshot::potential(&basis, &pairs, &c2);
                        vec![v.iter().zip(&dd).map(|(a, b)| a * b).sum()]
                    },
                    H,
                )[0],
            );
        }
    }
    let eq = check("grad tr(D dV/dQ) (point charges)", &flat(&gcq), &numq, TOL);
    let hq = oneshot::potential_hess(&basis, &pairs, &ch, &dd);
    let nq3 = 3 * (natom + 2);
    let mut numhq = vec![0.0; nq3 * nq3];
    for col in 0..nq3 {
        let v = fd(
            |h| {
                let mut c = positions.clone();
                let mut c2 = ch.clone();
                if col < m {
                    c[col / 3][col % 3] += h;
                } else {
                    c2[(col - m) / 3].1[(col - m) % 3] += h;
                }
                let b = make_basis(&def, &z, &c);
                let p = PairSet::build(&b, PairOptions::with_order(DerivOrder::One));
                let (ga, gc) = oneshot::potential_grad(&b, &p, &c2, &dd);
                flat(&ga).into_iter().chain(flat(&gc)).collect()
            },
            H,
        );
        for i in 0..nq3 {
            numhq[i * nq3 + col] = v[i];
        }
    }
    let ehq = check("hess tr(D d2V) (point charges)", &hq, &numhq, TOL);
    println!("SUMMARY fd 1e contractions: S {es:.2e} T {et:.2e} V {ev:.2e} dip {ed:.2e} | hess S {ehs:.2e} T {eht:.2e} V {ehv:.2e} | charges grad {eq:.2e} hess {ehq:.2e}");
}
