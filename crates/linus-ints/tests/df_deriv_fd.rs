//! Finite-difference validation of the three- and two-center derivative contractions of
//! [`linus_ints::reference::df_deriv`] (random inputs, all perturbations; atoms, AO and auxiliary
//! functions move together).

mod common;
use common::*;
use linus_ints::reference::oneshot::metric_deriv_rows;
use linus_ints::reference::{apply_metric_deriv, df3c_deriv_atoms, Df3cInput};
use linus_ints::{DerivOrder, PairOptions, PairSet};

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
fn fd_df3c_contractions() {
    for with_g in [false, true] {
        let (def, z, positions) = test_system(with_g);
        let adef = aux_def();
        let basis = make_basis(&def, &z, &positions);
        let aux = make_basis(&adef, &z, &positions);
        let (n, naux) = (basis.nbf, aux.nbf);
        let natom = z.len();
        let (r, o) = (3usize, 2usize);
        let c = random_vec(naux, 11);
        let d = random_sym(n, 12);
        let y = random_vec(naux * n * r, 13);
        let cm = random_vec(n * o, 14);
        let xw = 2usize;
        let xm = random_vec(naux * xw, 15);
        // reference: contractions with plain integrals at displaced geometries
        let contract = |cc: &[[f64; 3]]| -> Vec<f64> {
            let b = make_basis(&def, &z, cc);
            let a = make_basis(&adef, &z, cc);
            let p = PairSet::build(&b, PairOptions::default());
            let ap = PairSet::aux(&a, PairOptions::default());
            let j3 = linus_ints::reference::eri_3c(&b, &p, &a, &ap);
            let j2 = linus_ints::reference::eri_2c(&a, &ap);
            let mut out = Vec::new();
            // M_mn = sum_P (P|mn) c_P
            for mn in 0..n * n {
                out.push((0..naux).map(|q| j3[q * n * n + mn] * c[q]).sum());
            }
            // γ_P = sum (P|mn) D_mn
            for q in 0..naux {
                out.push(
                    j3[q * n * n..(q + 1) * n * n]
                        .iter()
                        .zip(&d)
                        .map(|(x, y)| x * y)
                        .sum(),
                );
            }
            // Y_mj = sum_P sum_l (P|ml) Y_P,lj
            for m in 0..n {
                for j in 0..r {
                    let mut s = 0.0;
                    for q in 0..naux {
                        for l in 0..n {
                            s += j3[q * n * n + m * n + l] * y[(q * n + l) * r + j];
                        }
                    }
                    out.push(s);
                }
            }
            // Λ_P,ij = C^T (P|..) C
            for q in 0..naux {
                for i in 0..o {
                    for j in 0..o {
                        let mut s = 0.0;
                        for m in 0..n {
                            for l in 0..n {
                                s += cm[m * o + i] * j3[q * n * n + m * n + l] * cm[l * o + j];
                            }
                        }
                        out.push(s);
                    }
                }
            }
            // (V X)_P
            for q in 0..naux {
                for w in 0..xw {
                    out.push((0..naux).map(|s| j2[q * naux + s] * xm[s * xw + w]).sum());
                }
            }
            out
        };
        let pairs = PairSet::build(&basis, PairOptions::default());
        let ap = PairSet::aux(&aux, PairOptions::with_order(DerivOrder::One));
        let input = Df3cInput {
            vecs: vec![&c],
            dens: vec![&d],
            right: Some((&y, r)),
            orbitals: Some((&cm, o)),
        };
        let atoms: Vec<usize> = (0..natom).collect();
        let res = df3c_deriv_atoms(&basis, &pairs, &aux, &ap, &input, &atoms, 0.0);
        let sizes = [n * n, naux, n * r, naux * o * o, naux * xw];
        let names = [
            "M^x = (P|mn)^x c_P",
            "γ^x = (P|mn)^x D_mn",
            "Y^x = (P|ml)^x Y_P,l",
            "Λ^x = C^T (P|..)^x C",
            "V^x X",
        ];
        let mut worst = [0.0f64; 5];
        for a in &res {
            let rows = metric_deriv_rows(&aux, &ap, a.atom);
            let vx = apply_metric_deriv(&rows.0, &rows.1, naux, &xm, xw);
            for k in 0..3 {
                let num = fd(|h| contract(&displaced(&positions, a.atom, k, h)), H);
                let mut off = 0;
                let ana: [&[f64]; 5] = [
                    &a.m[0][k * n * n..(k + 1) * n * n],
                    &a.gamma[0][k * naux..(k + 1) * naux],
                    &a.right[k * n * r..(k + 1) * n * r],
                    &a.lambda[k * naux * o * o..(k + 1) * naux * o * o],
                    &vx[k * naux * xw..(k + 1) * naux * xw],
                ];
                for t in 0..5 {
                    let e = check(
                        &format!("{} atom {} k {} (g={with_g})", names[t], a.atom, k),
                        ana[t],
                        &num[off..off + sizes[t]],
                        1e-8,
                    );
                    worst[t] = worst[t].max(e);
                    off += sizes[t];
                }
            }
        }
        println!(
            "SUMMARY fd df3c contractions (g={with_g}): {:?}",
            worst.map(|e| format!("{e:.2e}"))
        );
    }
}
