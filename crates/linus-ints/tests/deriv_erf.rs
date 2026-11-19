//! Derivative integrals of the erf-attenuated operators (erf(w r)/r, erfc(w r)/r and CAM
//! combinations a/r + b erf(w r)/r): raw first-derivative blocks and the gradient
//! contractions (4-, 3- and 2-center) against Richardson-extrapolated central differences of
//! the integrals themselves.

mod common;
use common::*;
use linus_ints::contract::{DensityTerm, DensityTerms};
use linus_ints::core::hermite::ShellView;
use linus_ints::core::pair_data;
use linus_ints::eri4c::deriv::{self, DerivWork};
use linus_ints::eri4c::{self, eri_quartet, EriWork};
use linus_ints::operator::Operator;
use linus_ints::reference::oneshot;
use linus_ints::reference::Dens3c;
use linus_ints::{BasisSet, DerivOrder, PairOptions, PairSet};

const TOL: f64 = 1e-9;

fn ops() -> Vec<(&'static str, Operator)> {
    vec![
        ("erf(0.3)", Operator::LongRange { omega: 0.3 }),
        (
            "erfc(0.4)",
            Operator::Attenuated {
                coulomb: 1.0,
                long_range: -1.0,
                omega: 0.4,
            },
        ),
        (
            "cam(0.19, 0.46, 0.33)",
            Operator::Attenuated {
                coulomb: 0.19,
                long_range: 0.46,
                omega: 0.33,
            },
        ),
        ("erf(2.5)", Operator::LongRange { omega: 2.5 }),
    ]
}

#[allow(clippy::needless_range_loop, reason = "indexes several arrays in step")]
fn shifted(
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
    let (sp, pd) = pair_data::build(&a, &b, i, j, 1, pair_data::PRIM_PAIR_CUT);
    (sp, pd.unwrap())
}

#[test]
fn fd_erf_eri4_first_derivative_blocks() {
    let (def, z, positions) = test_system(true);
    let basis = make_basis(&def, &z, &positions);
    let ns = basis.shells.len();
    let zero = [[0.0; 3]; 2];
    let mut ew = EriWork::new();
    let mut quartets = vec![];
    for i in 0..ns {
        for j in 0..ns {
            quartets.push([i, j, (i + 2) % ns, (j + 5) % ns]);
        }
    }
    for (name, op) in ops() {
        let mut w = DerivWork::with_op(op);
        let mut worst: f64 = 0.0;
        for q in &quartets {
            let (bra, bd) = shifted(&basis, q[0], q[1], zero);
            let (ket, kd) = shifted(&basis, q[2], q[3], zero);
            let blk = bra.nab() * ket.nab();
            let mut d1 = vec![0.0; 12 * blk];
            linus_ints::reference::eri_quartet_deriv1(&bra, &bd, &ket, &kd, &mut d1, &mut w);
            for x in 0..4 {
                for k in 0..3 {
                    let num = fd(
                        |h| {
                            let mut mv = [[[0.0; 3]; 2]; 2];
                            mv[x / 2][x % 2][k] = h;
                            let (b, _) = shifted(&basis, q[0], q[1], mv[0]);
                            let (kt, _) = shifted(&basis, q[2], q[3], mv[1]);
                            let mut out = vec![0.0; blk];
                            eri_quartet(&b, &kt, &op, &mut out, &mut ew);
                            out
                        },
                        H,
                    );
                    worst = worst.max(check(
                        &format!("{name} d(ab|cd)/d{x}{k} {q:?}"),
                        &d1[(x * 3 + k) * blk..(x * 3 + k + 1) * blk],
                        &num,
                        TOL,
                    ));
                }
            }
        }
        println!("SUMMARY fd {name} ERI 1st blocks: {worst:.2e}");
    }
}

fn eri_energy_op(basis: &BasisSet, terms: &DensityTerms, op: &Operator) -> f64 {
    let pairs = PairSet::build(basis, PairOptions::default());
    let g = eri4c::eri_full(basis, &pairs, op);
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
fn fd_erf_eri4_gradient_contraction() {
    let (def, z, positions) = test_system(false);
    let n = make_basis(&def, &z, &positions).nbf;
    let (d1, d2, d3) = (random_sym(n, 11), random_sym(n, 12), random_sym(n, 13));
    let terms = DensityTerms::default()
        .coulomb(0.7, &d1, &d1)
        .coulomb(0.3, &d1, &d2)
        .exchange(-0.4, &d1, &d1)
        .exchange(0.2, &d2, &d3);
    for (name, op) in ops() {
        let b = make_basis(&def, &z, &positions);
        let p = PairSet::build(&b, PairOptions::with_order(DerivOrder::One));
        let g = flat(&oneshot::grad_4c(&b, &p, &terms, 0.0, &op));
        let mut num = Vec::new();
        for atom in 0..z.len() {
            for k in 0..3 {
                num.push(
                    fd(
                        |h| {
                            vec![eri_energy_op(
                                &make_basis(&def, &z, &displaced(&positions, atom, k, h)),
                                &terms,
                                &op,
                            )]
                        },
                        h_step(),
                    )[0],
                );
            }
        }
        let e = check(&format!("{name} 4c gradient contraction"), &g, &num, TOL);
        println!("SUMMARY fd {name} 4c gradient contraction: {e:.2e}");
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

/// (P|op|mn) (naux x n^2) and (P|op|Q) from the generic quartet code.
fn ints_3c_2c(basis: &BasisSet, aux: &BasisSet, op: &Operator) -> (Vec<f64>, Vec<f64>) {
    let (n, naux) = (basis.nbf, aux.nbf);
    let pairs = PairSet::build(basis, PairOptions::default());
    let ap = PairSet::aux(aux, PairOptions::default());
    let mut w = EriWork::new();
    let mut j3 = vec![0.0; naux * n * n];
    for (pi, pa) in ap.pairs().iter().enumerate() {
        let p0 = aux.shells[pi].ao_start;
        for pr in pairs.pairs() {
            let mut buf = vec![0.0; pa.na * pr.nab()];
            eri_quartet(pa, pr, op, &mut buf, &mut w);
            let (a0, b0) = (basis.shells[pr.sa].ao_start, basis.shells[pr.sb].ao_start);
            for x in 0..pa.na {
                for ia in 0..pr.na {
                    for ib in 0..pr.nb {
                        let v = buf[x * pr.nab() + ia * pr.nb + ib];
                        j3[(p0 + x) * n * n + (a0 + ia) * n + b0 + ib] = v;
                        j3[(p0 + x) * n * n + (b0 + ib) * n + a0 + ia] = v;
                    }
                }
            }
        }
    }
    let mut j2 = vec![0.0; naux * naux];
    for (pi, pa) in ap.pairs().iter().enumerate() {
        for (qi, qa) in ap.pairs().iter().enumerate() {
            let mut buf = vec![0.0; pa.na * qa.na];
            eri_quartet(pa, qa, op, &mut buf, &mut w);
            let (p0, q0) = (aux.shells[pi].ao_start, aux.shells[qi].ao_start);
            for x in 0..pa.na {
                for y in 0..qa.na {
                    j2[(p0 + x) * naux + q0 + y] = buf[x * qa.na + y];
                }
            }
        }
    }
    (j3, j2)
}

#[test]
fn fd_erf_df_gradient_contractions() {
    let (def, z, positions) = test_system(false);
    let adef = aux_def();
    let basis = make_basis(&def, &z, &positions);
    let aux = make_basis(&adef, &z, &positions);
    let (n, naux) = (basis.nbf, aux.nbf);
    let mut full = vec![0.0; naux * n * n];
    for p in 0..naux {
        full[p * n * n..(p + 1) * n * n].copy_from_slice(&random_sym(n, 200 + p as u64));
    }
    let dvec = random_vec(naux, 17);
    let dmat = random_sym(n, 18);
    let gam2 = random_sym(naux, 19);
    let dens = Dens3c {
        full: Some(&full),
        aux_shells: 0..aux.shells.len(),
        separable: vec![(0.6, &dvec, &dmat)],
    };
    for (name, op) in ops() {
        let e32 = |c: &[[f64; 3]]| {
            let b = make_basis(&def, &z, c);
            let a = make_basis(&adef, &z, c);
            let (j3, j2) = ints_3c_2c(&b, &a, &op);
            let mut e = 0.0;
            for pp in 0..naux {
                for mn in 0..n * n {
                    e += j3[pp * n * n + mn] * (full[pp * n * n + mn] + 0.6 * dvec[pp] * dmat[mn]);
                }
            }
            vec![e, j2.iter().zip(&gam2).map(|(x, y)| x * y).sum()]
        };
        let p = PairSet::build(&basis, PairOptions::with_order(DerivOrder::One));
        let ap = PairSet::aux(&aux, PairOptions::with_order(DerivOrder::One));
        let g3 = flat(&linus_ints::reference::eri_grad_3c(
            &basis, &p, &aux, &ap, &dens, 0.0, &op,
        ));
        let g2 = flat(&linus_ints::reference::oneshot::grad_2c(
            &aux, &ap, &gam2, 0.0, &op,
        ));
        let (mut n3, mut n2) = (vec![], vec![]);
        for atom in 0..z.len() {
            for k in 0..3 {
                let v = fd(|h| e32(&displaced(&positions, atom, k, h)), H);
                n3.push(v[0]);
                n2.push(v[1]);
            }
        }
        let e3 = check(&format!("{name} 3c gradient contraction"), &g3, &n3, TOL);
        let e2 = check(&format!("{name} 2c gradient contraction"), &g2, &n2, TOL);
        println!("SUMMARY fd {name} DF contractions: 3c {e3:.2e} 2c {e2:.2e}");
    }
}

/// The fast 3-center kernel's erf integrals (used by RI-K_LR) agree with the generic
/// quartet code (used by the derivative integrals).
#[test]
fn erf_3c_fast_kernel_matches_quartets() {
    use linus_ints::eri3c::{pair_block, Aux3c, Pair3c, Work3c};
    let (def, z, positions) = test_system(false);
    let basis = make_basis(&def, &z, &positions);
    let aux = make_basis(&aux_def(), &z, &positions);
    let (n, naux) = (basis.nbf, aux.nbf);
    let w_ = 0.37;
    let (j3, _) = ints_3c_2c(&basis, &aux, &Operator::LongRange { omega: w_ });
    let a3 = Aux3c::new(&aux);
    let chunks = a3.chunks(0..aux.shells.len());
    let pairs = PairSet::build(&basis, PairOptions::default());
    let mut w = Work3c::new();
    let mut worst: f64 = 0.0;
    for pr in pairs.pairs() {
        let p3 = Pair3c::from_shell_pair(pr);
        let mut out = vec![0.0; pr.nab() * naux];
        pair_block(
            &p3,
            &chunks,
            &mut out,
            naux,
            0.0,
            &Operator::LongRange { omega: w_ },
            &mut w,
        );
        let (a0, b0) = (basis.shells[pr.sa].ao_start, basis.shells[pr.sb].ao_start);
        for ia in 0..pr.na {
            for ib in 0..pr.nb {
                for p in 0..naux {
                    let r = j3[p * n * n + (a0 + ia) * n + b0 + ib];
                    worst = worst.max((out[(ia * pr.nb + ib) * naux + p] - r).abs());
                }
            }
        }
    }
    println!("SUMMARY erf 3c fast kernel vs quartets: {worst:.2e}");
    assert!(worst < 1e-13, "{worst}");
}

/// out[c][offs[keep] + x_keep] += f * sum blk[c][x...] prod_{p != keep} vecs[p][offs[p] + x_p]
#[allow(clippy::too_many_arguments)]
fn acc_proj(
    blk: &[f64],
    ncomp: usize,
    dims: &[usize],
    offs: &[usize],
    keep: usize,
    vecs: &[&[f64]],
    f: f64,
    nkeep: usize,
    out: &mut [f64],
) {
    let m = dims.len();
    let size: usize = dims.iter().product();
    let mut idx = vec![0usize; m];
    for c in 0..ncomp {
        for flat in 0..size {
            let mut r = flat;
            for p in (0..m).rev() {
                idx[p] = r % dims[p];
                r /= dims[p];
            }
            let mut w = f * blk[c * size + flat];
            for p in 0..m {
                if p != keep {
                    w *= vecs[p][offs[p] + idx[p]];
                }
            }
            out[c * nkeep + offs[keep] + idx[keep]] += w;
        }
    }
}

/// First-derivative erf(0.3 r)/r integrals against PySCF/libcint (mol.with_range_coulomb):
/// int2e_ip1, int3c2e_ip1, int3c2e_ip2, int2c2e_ip1, as random projections
/// (tests/python/gen_deriv_fixtures.py erf; same system and vectors as deriv_pyscf.rs).
#[test]
fn pyscf_erf_first_derivatives() {
    use linus_ints::core::deriv::pair_with_deriv;
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    let read = |f: &str| -> serde_json::Value {
        serde_json::from_str(&std::fs::read_to_string(dir.join(f)).unwrap()).unwrap()
    };
    let (base, fx) = (
        read("pyscf_deriv_ints.json"),
        read("pyscf_deriv_ints_erf.json"),
    );
    let arr = |v: &serde_json::Value| -> Vec<f64> {
        v.as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_f64().unwrap())
            .collect()
    };
    // the fixture system is test_system(true) with aux_def()
    let (def, z, positions) = test_system(true);
    let basis = make_basis(&def, &z, &positions);
    let aux = make_basis(&aux_def(), &z, &positions);
    let (n, naux) = (basis.nbf, aux.nbf);
    assert_eq!(
        (n as u64, naux as u64),
        (fx["nbf"].as_u64().unwrap(), fx["naux"].as_u64().unwrap())
    );
    let op = Operator::LongRange {
        omega: fx["omega"].as_f64().unwrap(),
    };
    let (v1, v2, v3, v4, u1, u2) = (
        arr(&base["v1"]),
        arr(&base["v2"]),
        arr(&base["v3"]),
        arr(&base["v4"]),
        arr(&base["u1"]),
        arr(&base["u2"]),
    );
    let one = vec![1.0; n.max(naux)];
    let sh = &basis.shells;
    let ns = sh.len();
    let prs: Vec<Vec<_>> = (0..ns)
        .map(|i| (0..ns).map(|j| pair_with_deriv(&basis, i, j, 1)).collect())
        .collect();
    let mut w = DerivWork::with_op(op);
    let mut ip1 = vec![0.0; 3 * n];
    for i in 0..ns {
        for j in 0..ns {
            for k in 0..ns {
                for l in 0..ns {
                    let ((bra, bd), (ket, kd)) = (&prs[i][j], &prs[k][l]);
                    let blk = bra.nab() * ket.nab();
                    let mut d1 = vec![0.0; 12 * blk];
                    linus_ints::reference::eri_quartet_deriv1(bra, bd, ket, kd, &mut d1, &mut w);
                    let dims = [bra.na, bra.nb, ket.na, ket.nb];
                    let offs = [
                        sh[i].ao_start,
                        sh[j].ao_start,
                        sh[k].ao_start,
                        sh[l].ao_start,
                    ];
                    acc_proj(
                        &d1[..3 * blk],
                        3,
                        &dims,
                        &offs,
                        0,
                        &[&one, &v2, &v3, &v4],
                        -1.0,
                        n,
                        &mut ip1,
                    );
                }
            }
        }
    }
    let ap = PairSet::aux(&aux, PairOptions::with_order(DerivOrder::One));
    let ash = &aux.shells;
    let (mut t_ip1, mut t_ip2) = (vec![0.0; 3 * n], vec![0.0; 3 * naux]);
    #[allow(clippy::needless_range_loop, reason = "shells and their pairs in step")]
    for ps in 0..ash.len() {
        for i in 0..ns {
            for j in 0..ns {
                let (ket, kd) = &prs[i][j];
                let bra = ap.pair(ps);
                let blk = bra.nab() * ket.nab();
                let dims = [bra.na, 1, ket.na, ket.nb];
                let offs = [ash[ps].ao_start, 0, sh[i].ao_start, sh[j].ao_start];
                let mut d1 = vec![0.0; 12 * blk];
                linus_ints::reference::eri_quartet_deriv1(
                    bra,
                    ap.deriv(ps),
                    ket,
                    kd,
                    &mut d1,
                    &mut w,
                );
                acc_proj(
                    &d1[6 * blk..9 * blk],
                    3,
                    &dims,
                    &offs,
                    2,
                    &[&u1, &one, &one, &v2],
                    -1.0,
                    n,
                    &mut t_ip1,
                );
                acc_proj(
                    &d1[0..3 * blk],
                    3,
                    &dims,
                    &offs,
                    0,
                    &[&one, &one, &v1, &v2],
                    -1.0,
                    naux,
                    &mut t_ip2,
                );
            }
        }
    }
    let mut c_ip1 = vec![0.0; 3 * naux];
    for p in 0..ash.len() {
        for q in 0..ash.len() {
            let (bra, ket) = (ap.pair(p), ap.pair(q));
            let blk = bra.na * ket.na;
            let mut d1 = vec![0.0; 12 * blk];
            linus_ints::reference::eri_quartet_deriv1(
                bra,
                ap.deriv(p),
                ket,
                ap.deriv(q),
                &mut d1,
                &mut w,
            );
            acc_proj(
                &d1[0..3 * blk],
                3,
                &[bra.na, 1, ket.na, 1],
                &[ash[p].ao_start, 0, ash[q].ao_start, 0],
                0,
                &[&one, &one, &u2, &one],
                -1.0,
                naux,
                &mut c_ip1,
            );
        }
    }
    for (name, ours) in [
        ("int2e_ip1", &ip1),
        ("int3c2e_ip1", &t_ip1),
        ("int3c2e_ip2", &t_ip2),
        ("int2c2e_ip1", &c_ip1),
    ] {
        let r = arr(&fx["ints"][name]["data"]);
        let e = check(&format!("erf vs PySCF {name}"), ours, &r, 1e-10);
        println!(
            "SUMMARY pyscf erf {name:14} {e:.2e} (max |ref| {:.2e})",
            max_abs(&r)
        );
    }
}

/// The gradient contractions do not depend on the number of threads (bitwise).
#[test]
fn gradient_contractions_are_thread_count_independent() {
    let (def, z, positions) = test_system(false);
    let basis = make_basis(&def, &z, &positions);
    let aux = make_basis(&aux_def(), &z, &positions);
    let (n, naux) = (basis.nbf, aux.nbf);
    let (d1, d2) = (random_sym(n, 31), random_sym(n, 32));
    let terms = DensityTerms::default()
        .coulomb(0.5, &d1, &d1)
        .exchange(-0.25, &d1, &d1)
        .exchange(0.1, &d2, &d2);
    let mut full = vec![0.0; naux * n * n];
    for p in 0..naux {
        full[p * n * n..(p + 1) * n * n].copy_from_slice(&random_sym(n, 300 + p as u64));
    }
    let dvec = random_vec(naux, 33);
    let gam2 = random_sym(naux, 34);
    let op = Operator::Attenuated {
        coulomb: 0.19,
        long_range: 0.46,
        omega: 0.33,
    };
    let run = |threads: usize| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap()
            .install(|| {
                let p = PairSet::build(&basis, PairOptions::with_order(DerivOrder::One));
                let ap = PairSet::aux(&aux, PairOptions::with_order(DerivOrder::One));
                let dens = Dens3c {
                    full: Some(&full),
                    aux_shells: 0..aux.shells.len(),
                    separable: vec![(0.6, &dvec, &d1)],
                };
                let mut g = flat(&oneshot::grad_4c(&basis, &p, &terms, 0.0, &op));
                g.extend(flat(&linus_ints::reference::eri_grad_3c(
                    &basis, &p, &aux, &ap, &dens, 0.0, &op,
                )));
                g.extend(flat(&linus_ints::reference::oneshot::grad_2c(
                    &aux, &ap, &gam2, 0.0, &op,
                )));
                g
            })
    };
    let (a, b) = (run(1), run(5));
    for (x, y) in a.iter().zip(&b) {
        assert_eq!(x.to_bits(), y.to_bits(), "{x} {y}");
    }
}

// ---------------------------------------------------------------------------------------
// Second derivatives (range-separated hybrid Hessians)
// ---------------------------------------------------------------------------------------

/// Hessian contraction sum_abcd d²(ab|op|cd)/dXdY Γ_abcd from raw second-derivative blocks of
/// every ordered shell quartet (exact reference for the 4c Hessian).
fn hess_from_raw_op(basis: &BasisSet, terms: &DensityTerms, op: &Operator) -> Vec<f64> {
    let n = basis.nbf;
    let sh = &basis.shells;
    let ns = sh.len();
    let n3 = 3 * basis.atom_shells.len();
    let mut h = vec![0.0; n3 * n3];
    let prs: Vec<Vec<_>> = (0..ns)
        .map(|i| {
            (0..ns)
                .map(|j| linus_ints::core::deriv::pair_with_deriv(basis, i, j, 2))
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
                    linus_ints::reference::eri_quartet_deriv2_ref(bra, bd, ket, kd, op, &mut d2);
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

/// the 4c Hessian against central differences of the 4c gradient and against the contraction of
/// the raw second-derivative blocks of the operator.
#[test]
fn fd_erf_eri4_hessian_contraction() {
    let (def, z, positions) = test_system(false);
    let n = make_basis(&def, &z, &positions).nbf;
    let natom = z.len();
    let n3 = 3 * natom;
    let (d1, d2, d3) = (random_sym(n, 41), random_sym(n, 42), random_sym(n, 43));
    let terms = DensityTerms::default()
        .coulomb(0.7, &d1, &d1)
        .coulomb(0.3, &d1, &d2)
        .exchange(-0.4, &d1, &d1)
        .exchange(0.2, &d2, &d3);
    for (name, op) in ops() {
        let grad_at = |c: &[[f64; 3]]| {
            let b = make_basis(&def, &z, c);
            let p = PairSet::build(&b, PairOptions::with_order(DerivOrder::One));
            flat(&oneshot::grad_4c(&b, &p, &terms, 0.0, &op))
        };
        let basis = make_basis(&def, &z, &positions);
        let pairs = PairSet::build(&basis, PairOptions::with_order(DerivOrder::Two));
        let hss = oneshot::hess_4c(&basis, &pairs, &terms, 0.0, &op);
        let mut numh = vec![0.0; n3 * n3];
        for atom in 0..natom {
            for k in 0..3 {
                let col = fd(|h| grad_at(&displaced(&positions, atom, k, h)), H);
                for i in 0..n3 {
                    numh[i * n3 + 3 * atom + k] = col[i];
                }
            }
        }
        let eh = check(&format!("{name} 4c Hessian contraction"), &hss, &numh, TOL);
        let er = check(
            &format!("{name} 4c Hessian contraction vs raw blocks"),
            &hss,
            &hess_from_raw_op(&basis, &terms, &op),
            1e-12,
        );
        println!("SUMMARY fd {name} 4c Hessian contraction: fd {eh:.2e} raw {er:.2e}");
    }
}

/// eri_hess_3c and eri_hess_2c against central differences of the gradient contractions.
#[test]
fn fd_erf_df_hessian_contractions() {
    let (def, z, positions) = test_system(false);
    let adef = aux_def();
    let basis = make_basis(&def, &z, &positions);
    let aux = make_basis(&adef, &z, &positions);
    let (n, naux) = (basis.nbf, aux.nbf);
    let natom = z.len();
    let n3 = 3 * natom;
    let mut full = vec![0.0; naux * n * n];
    for p in 0..naux {
        full[p * n * n..(p + 1) * n * n].copy_from_slice(&random_sym(n, 400 + p as u64));
    }
    let dvec = random_vec(naux, 47);
    let dmat = random_sym(n, 48);
    let gam2 = random_sym(naux, 49);
    let dens = Dens3c {
        full: Some(&full),
        aux_shells: 0..aux.shells.len(),
        separable: vec![(0.6, &dvec, &dmat)],
    };
    for (name, op) in ops() {
        let grads = |c: &[[f64; 3]]| {
            let b = make_basis(&def, &z, c);
            let a = make_basis(&adef, &z, c);
            let p = PairSet::build(&b, PairOptions::with_order(DerivOrder::One));
            let ap = PairSet::aux(&a, PairOptions::with_order(DerivOrder::One));
            let mut g = flat(&linus_ints::reference::eri_grad_3c(
                &b, &p, &a, &ap, &dens, 0.0, &op,
            ));
            g.extend(flat(&linus_ints::reference::oneshot::grad_2c(
                &a, &ap, &gam2, 0.0, &op,
            )));
            g
        };
        let p = PairSet::build(&basis, PairOptions::with_order(DerivOrder::Two));
        let ap = PairSet::aux(&aux, PairOptions::with_order(DerivOrder::Two));
        let h3 = linus_ints::reference::eri_hess_3c(&basis, &p, &aux, &ap, &dens, 0.0, &op);
        let h2 = linus_ints::reference::oneshot::hess_2c(&aux, &ap, &gam2, 0.0, &op);
        let (mut n3h, mut n2h) = (vec![0.0; n3 * n3], vec![0.0; n3 * n3]);
        for atom in 0..natom {
            for k in 0..3 {
                let col = fd(|h| grads(&displaced(&positions, atom, k, h)), H);
                for i in 0..n3 {
                    n3h[i * n3 + 3 * atom + k] = col[i];
                    n2h[i * n3 + 3 * atom + k] = col[n3 + i];
                }
            }
        }
        let e3 = check(&format!("{name} 3c Hessian contraction"), &h3, &n3h, TOL);
        let e2 = check(&format!("{name} 2c Hessian contraction"), &h2, &n2h, TOL);
        println!("SUMMARY fd {name} DF Hessian contractions: 3c {e3:.2e} 2c {e2:.2e}");
    }
}

/// J[D], K[D] of the operator from its full ERI tensor.
fn jk_full_op(basis: &BasisSet, d: &[f64], op: &Operator) -> (Vec<f64>, Vec<f64>) {
    let pairs = PairSet::build(basis, PairOptions::default());
    let g = eri4c::eri_full(basis, &pairs, op);
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

/// jk_deriv (the derivative Fock matrices of the Hessian's right-hand sides) against central
/// differences of J and K of the operator.
#[test]
fn fd_erf_jk_derivative_matrices() {
    let (def, z, positions) = test_system(false);
    let basis = make_basis(&def, &z, &positions);
    let n = basis.nbf;
    let n2 = n * n;
    let (d1, d2) = (random_sym(n, 51), random_sym(n, 52));
    let pairs = PairSet::build(&basis, PairOptions::default());
    for (name, op) in ops() {
        let (jx, kx) = linus_ints::eri4c::jk_deriv::jk_deriv(
            &basis,
            &pairs,
            &[&d1, &d2],
            true,
            true,
            0.0,
            &op,
        );
        let (mut ej, mut ek) = (0.0f64, 0.0f64);
        for atom in 0..z.len() {
            for k in 0..3 {
                let x = 3 * atom + k;
                let num = fd(
                    |h| {
                        let b = make_basis(&def, &z, &displaced(&positions, atom, k, h));
                        let (j1, k1) = jk_full_op(&b, &d1, &op);
                        let (j2, k2) = jk_full_op(&b, &d2, &op);
                        j1.into_iter().chain(k1).chain(j2).chain(k2).collect()
                    },
                    H,
                );
                for (di, o) in [(0, 0), (1, 2)] {
                    ej = ej.max(check(
                        &format!("{name} J^x[D{di}] x={x}"),
                        &jx[di][x * n2..(x + 1) * n2],
                        &num[o * n2..(o + 1) * n2],
                        TOL,
                    ));
                    ek = ek.max(check(
                        &format!("{name} K^x[D{di}] x={x}"),
                        &kx[di][x * n2..(x + 1) * n2],
                        &num[(o + 1) * n2..(o + 2) * n2],
                        TOL,
                    ));
                }
            }
        }
        // the one-atom-per-task path must agree
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap();
        let (j1, k1) = pool.install(|| {
            linus_ints::eri4c::jk_deriv::jk_deriv(&basis, &pairs, &[&d1, &d2], true, true, 0.0, &op)
        });
        for di in 0..2 {
            check("J^x per-atom path", &j1[di], &jx[di], 1e-13);
            check("K^x per-atom path", &k1[di], &kx[di], 1e-13);
        }
        println!("SUMMARY fd {name} J^x/K^x: J {ej:.2e} K {ek:.2e}");
    }
}

/// Second-derivative erf(0.3 r)/r integrals against PySCF/libcint (mol.with_range_coulomb):
/// int2e_ipip1, int2e_ipvip1, int2e_ip1ip2, int3c2e_ipip1, int3c2e_ipvip1, int3c2e_ip1ip2,
/// int3c2e_ipip2, int2c2e_ipip1, int2c2e_ip1ip2, as the random projections of
/// tests/python/gen_deriv_fixtures.py erf (same layout as deriv_pyscf.rs).
#[test]
fn pyscf_erf_second_derivatives() {
    use linus_ints::core::deriv::pair_with_deriv;
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    let read = |f: &str| -> serde_json::Value {
        serde_json::from_str(&std::fs::read_to_string(dir.join(f)).unwrap()).unwrap()
    };
    let (base, fx) = (
        read("pyscf_deriv_ints.json"),
        read("pyscf_deriv_ints_erf.json"),
    );
    let arr = |v: &serde_json::Value| -> Vec<f64> {
        v.as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_f64().unwrap())
            .collect()
    };
    let (def, z, positions) = test_system(true);
    let basis = make_basis(&def, &z, &positions);
    let aux = make_basis(&aux_def(), &z, &positions);
    let (n, naux) = (basis.nbf, aux.nbf);
    let op = Operator::LongRange {
        omega: fx["omega"].as_f64().unwrap(),
    };
    let (v1, v2, v3, v4, u1, u2) = (
        arr(&base["v1"]),
        arr(&base["v2"]),
        arr(&base["v3"]),
        arr(&base["v4"]),
        arr(&base["u1"]),
        arr(&base["u2"]),
    );
    let one = vec![1.0; n.max(naux)];
    let sh = &basis.shells;
    let ns = sh.len();
    let prs: Vec<Vec<_>> = (0..ns)
        .map(|i| (0..ns).map(|j| pair_with_deriv(&basis, i, j, 2)).collect())
        .collect();
    // 4-center: d2 blocks [AA, AB, AC, ...] (D2_CENTER_PAIRS order); ipip1 = AA, ipvip1 = AB,
    // ip1ip2 = AC (both "ip" signs cancel)
    let (mut ipip1, mut ipvip1, mut ip1ip2) =
        (vec![0.0; 9 * n], vec![0.0; 9 * n], vec![0.0; 9 * n]);
    for i in 0..ns {
        for j in 0..ns {
            for k in 0..ns {
                for l in 0..ns {
                    let ((bra, bd), (ket, kd)) = (&prs[i][j], &prs[k][l]);
                    let blk = bra.nab() * ket.nab();
                    let mut d2 = vec![0.0; 90 * blk];
                    linus_ints::reference::eri_quartet_deriv2_ref(bra, bd, ket, kd, &op, &mut d2);
                    let dims = [bra.na, bra.nb, ket.na, ket.nb];
                    let offs = [
                        sh[i].ao_start,
                        sh[j].ao_start,
                        sh[k].ao_start,
                        sh[l].ao_start,
                    ];
                    let vecs = [&one[..], &v2, &v3, &v4];
                    acc_proj(
                        &d2[0..9 * blk],
                        9,
                        &dims,
                        &offs,
                        0,
                        &vecs,
                        1.0,
                        n,
                        &mut ipip1,
                    );
                    acc_proj(
                        &d2[9 * blk..18 * blk],
                        9,
                        &dims,
                        &offs,
                        0,
                        &vecs,
                        1.0,
                        n,
                        &mut ipvip1,
                    );
                    acc_proj(
                        &d2[18 * blk..27 * blk],
                        9,
                        &dims,
                        &offs,
                        0,
                        &vecs,
                        1.0,
                        n,
                        &mut ip1ip2,
                    );
                }
            }
        }
    }
    // 3-center (P|ij): bra = aux pair (A = P), ket = (C = i, D = j)
    let ap = PairSet::aux(&aux, PairOptions::with_order(DerivOrder::Two));
    let ash = &aux.shells;
    let (mut t_ipip1, mut t_ipvip1, mut t_ip1ip2, mut t_ipip2) = (
        vec![0.0; 9 * n],
        vec![0.0; 9 * n],
        vec![0.0; 9 * n],
        vec![0.0; 9 * naux],
    );
    #[allow(clippy::needless_range_loop, reason = "shells and their pairs in step")]
    for ps in 0..ash.len() {
        for i in 0..ns {
            for j in 0..ns {
                let (ket, kd) = &prs[i][j];
                let bra = ap.pair(ps);
                let blk = bra.nab() * ket.nab();
                let dims = [bra.na, 1, ket.na, ket.nb];
                let offs = [ash[ps].ao_start, 0, sh[i].ao_start, sh[j].ao_start];
                let mut d2 = vec![0.0; 90 * blk];
                linus_ints::reference::eri_quartet_deriv2_ref(
                    bra,
                    ap.deriv(ps),
                    ket,
                    kd,
                    &op,
                    &mut d2,
                );
                let vk_i = [&u1[..], &one[..], &one[..], &v2[..]];
                let vk_p = [&one[..], &one[..], &v1[..], &v2[..]];
                // CC (index 7), CD (index 8), AA (0); ip1ip2 = d_i d_P = AC transposed
                acc_proj(
                    &d2[63 * blk..72 * blk],
                    9,
                    &dims,
                    &offs,
                    2,
                    &vk_i,
                    1.0,
                    n,
                    &mut t_ipip1,
                );
                acc_proj(
                    &d2[72 * blk..81 * blk],
                    9,
                    &dims,
                    &offs,
                    2,
                    &vk_i,
                    1.0,
                    n,
                    &mut t_ipvip1,
                );
                let ac = &d2[18 * blk..27 * blk];
                let mut act = vec![0.0; 9 * blk];
                for kk in 0..3 {
                    for ll in 0..3 {
                        act[(kk * 3 + ll) * blk..(kk * 3 + ll + 1) * blk]
                            .copy_from_slice(&ac[(ll * 3 + kk) * blk..(ll * 3 + kk + 1) * blk]);
                    }
                }
                acc_proj(&act, 9, &dims, &offs, 2, &vk_i, 1.0, n, &mut t_ip1ip2);
                acc_proj(
                    &d2[0..9 * blk],
                    9,
                    &dims,
                    &offs,
                    0,
                    &vk_p,
                    1.0,
                    naux,
                    &mut t_ipip2,
                );
            }
        }
    }
    let (mut c_ipip1, mut c_ip1ip2) = (vec![0.0; 9 * naux], vec![0.0; 9 * naux]);
    for p in 0..ash.len() {
        for q in 0..ash.len() {
            let (bra, ket) = (ap.pair(p), ap.pair(q));
            let blk = bra.na * ket.na;
            let mut d2 = vec![0.0; 90 * blk];
            linus_ints::reference::eri_quartet_deriv2_ref(
                bra,
                ap.deriv(p),
                ket,
                ap.deriv(q),
                &op,
                &mut d2,
            );
            let dims = [bra.na, 1, ket.na, 1];
            let offs = [ash[p].ao_start, 0, ash[q].ao_start, 0];
            let vecs = [&one[..], &one[..], &u2[..], &one[..]];
            acc_proj(
                &d2[0..9 * blk],
                9,
                &dims,
                &offs,
                0,
                &vecs,
                1.0,
                naux,
                &mut c_ipip1,
            );
            acc_proj(
                &d2[18 * blk..27 * blk],
                9,
                &dims,
                &offs,
                0,
                &vecs,
                1.0,
                naux,
                &mut c_ip1ip2,
            );
        }
    }
    for (name, ours) in [
        ("int2e_ipip1", &ipip1),
        ("int2e_ipvip1", &ipvip1),
        ("int2e_ip1ip2", &ip1ip2),
        ("int3c2e_ipip1", &t_ipip1),
        ("int3c2e_ipvip1", &t_ipvip1),
        ("int3c2e_ip1ip2", &t_ip1ip2),
        ("int3c2e_ipip2", &t_ipip2),
        ("int2c2e_ipip1", &c_ipip1),
        ("int2c2e_ip1ip2", &c_ip1ip2),
    ] {
        let r = arr(&fx["ints"][name]["data"]);
        let e = check(&format!("erf vs PySCF {name}"), ours, &r, 1e-10);
        println!(
            "SUMMARY pyscf erf {name:15} {e:.2e} (max |ref| {:.2e})",
            max_abs(&r)
        );
    }
}

/// The second-derivative contractions of the operator are bitwise independent of the thread
/// count.
#[test]
fn hessian_contractions_are_thread_count_independent() {
    let (def, z, positions) = test_system(false);
    let basis = make_basis(&def, &z, &positions);
    let aux = make_basis(&aux_def(), &z, &positions);
    let (n, naux) = (basis.nbf, aux.nbf);
    let (d1, d2) = (random_sym(n, 61), random_sym(n, 62));
    let terms = DensityTerms::default()
        .coulomb(0.5, &d1, &d1)
        .exchange(-0.25, &d1, &d1)
        .exchange(0.1, &d2, &d2);
    let mut full = vec![0.0; naux * n * n];
    for p in 0..naux {
        full[p * n * n..(p + 1) * n * n].copy_from_slice(&random_sym(n, 500 + p as u64));
    }
    let gam2 = random_sym(naux, 64);
    let op = Operator::Attenuated {
        coulomb: 0.19,
        long_range: 0.46,
        omega: 0.33,
    };
    let run = |threads: usize| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap()
            .install(|| {
                let p = PairSet::build(&basis, PairOptions::with_order(DerivOrder::Two));
                let ap = PairSet::aux(&aux, PairOptions::with_order(DerivOrder::Two));
                let dens = Dens3c {
                    full: Some(&full),
                    aux_shells: 0..aux.shells.len(),
                    separable: vec![],
                };
                let mut h = oneshot::hess_4c(&basis, &p, &terms, 0.0, &op);
                h.extend(linus_ints::reference::eri_hess_3c(
                    &basis, &p, &aux, &ap, &dens, 0.0, &op,
                ));
                h.extend(linus_ints::reference::oneshot::hess_2c(
                    &aux, &ap, &gam2, 0.0, &op,
                ));
                h
            })
    };
    let (a, b) = (run(1), run(5));
    for (x, y) in a.iter().zip(&b) {
        assert_eq!(x.to_bits(), y.to_bits(), "{x} {y}");
    }
}
