//! linus-jk's density-fitted gradient against whole calculations (linus_jk::df::gradient's
//! tests that run the Session).

use crate::driver::{CalcOptions, JkMethod, Session};
use crate::Structure;
use faer::{Mat, MatRef, Par};
use linus_base::linalg::mm;
use linus_ints::operator::Operator;
use linus_ints::reference::Dens3c;
use linus_ints::{BasisSet, DerivOrder, PairOptions, PairSet};
use linus_jk::df::gradient::*;
use rayon::prelude::*;

const WATER: &str = "3\nwater\nO 0.10 -0.05 0.12\nH 0.85 0.52 -0.03\nH -0.68 0.55 0.05\n";

/// Density-fitted two-electron gradient (J with the total density, K from occupied orbitals):
/// dense reference holding naux x nbf^2 tensors (the code the blocked gradient replaced).
#[allow(clippy::too_many_arguments)]
fn dense_df_gradient(
    basis: &BasisSet,
    pairs: &PairSet,
    aux: &BasisSet,
    dt: &[f64],
    occ: &[&Mat<f64>],
    restricted: bool,
    a_x: f64,
    par: Par,
) -> Vec<[f64; 3]> {
    let n = basis.nbf;
    let n2 = n * n;
    let naux = aux.nbf;
    let ap = PairSet::aux(aux, PairOptions::with_order(DerivOrder::One));
    let v =
        linus_base::linalg::mat_from_rowmajor(naux, naux, &linus_ints::reference::eri_2c(aux, &ap));
    let j3 = linus_ints::reference::eri_3c(basis, pairs, aux, &ap); // naux x n^2 row-major
    let l = linus_base::linalg::cholesky(v.as_ref(), Par::Seq)
        .expect("auxiliary metric not positive definite");
    // c = V^-1 gamma
    let gamma = Mat::from_fn(naux, 1, |p, _| {
        j3[p * n2..(p + 1) * n2]
            .iter()
            .zip(dt)
            .map(|(a, b)| a * b)
            .sum()
    });
    let c = linus_base::linalg::cholesky_solve(l.as_ref(), gamma.as_ref(), Par::Seq);
    let cv: Vec<f64> = (0..naux).map(|p| c[(p, 0)]).collect();
    // gamma_PQ (metric term): J part -1/2 c c^T
    let mut gam = vec![0.0; naux * naux];
    for p in 0..naux {
        for q in 0..naux {
            gam[p * naux + q] = -0.5 * cv[p] * cv[q];
        }
    }
    let mut gk: Option<Vec<f64>> = None;
    if a_x != 0.0 {
        // B~ (n^2 x naux, column-major): column P is the symmetric n x n matrix B~_P = sum_Q V^-1_PQ (Q|mn)
        let id = Mat::<f64>::identity(naux, naux);
        let vinv = linus_base::linalg::cholesky_solve(l.as_ref(), id.as_ref(), Par::Seq);
        let a = MatRef::from_column_major_slice(&j3, n2, naux);
        let mut bt = vec![0.0; n2 * naux];
        {
            let btm = faer::MatMut::from_column_major_slice_mut(&mut bt, n2, naux);
            linus_base::linalg::matmul(btm, faer::Accum::Replace, a, vinv.as_ref(), 1.0, par);
        }
        // restricted: one set of alpha orbitals stands for both spins (factor 2)
        let spin_w = if restricted { 2.0 } else { 1.0 };
        let mut g = vec![0.0; naux * n2];
        for cs in occ {
            let no = cs.ncols();
            if no == 0 {
                continue;
            }
            let no2 = no * no;
            // X_P = C^T B~_P C, stored as rows of an naux x no^2 matrix (column-major per P)
            let mut xflat = vec![0.0; naux * no2];
            xflat.par_chunks_mut(no2).enumerate().for_each(|(p, xp)| {
                let bp = MatRef::from_column_major_slice(&bt[p * n2..(p + 1) * n2], n, n);
                let y = mm(bp, cs.as_ref(), Par::Seq);
                let x = mm(cs.transpose(), y.as_ref(), Par::Seq);
                for j in 0..no {
                    for i in 0..no {
                        xp[j * no + i] = x[(i, j)];
                    }
                }
            });
            // G_P += -a_x w C X_P C^T   (row-major naux x n^2)
            g.par_chunks_mut(n2).enumerate().for_each(|(p, gp)| {
                let xp = MatRef::from_column_major_slice(&xflat[p * no2..(p + 1) * no2], no, no);
                let t = mm(cs.as_ref(), xp, Par::Seq);
                let full = mm(t.as_ref(), cs.transpose(), Par::Seq);
                for i in 0..n {
                    for j in 0..n {
                        gp[i * n + j] -= a_x * spin_w * full[(i, j)];
                    }
                }
            });
            // g_PQ += a_x/2 w tr(X_P X_Q) = a_x/2 w sum_ij X_P[i,j] X_Q[j,i]; X symmetric (D B~ D is),
            // so tr(X_P X_Q) = <X_P, X_Q>: one GEMM of the naux x no^2 matrix with itself
            let xm = MatRef::from_column_major_slice(&xflat, no2, naux);
            let gq = mm(xm.transpose(), xm, par);
            for p in 0..naux {
                for q in 0..naux {
                    gam[p * naux + q] += 0.5 * a_x * spin_w * gq[(p, q)];
                }
            }
        }
        gk = Some(g);
    }
    let dens = Dens3c {
        full: gk.as_deref(),
        aux_shells: 0..aux.shells.len(),
        separable: vec![(1.0, &cv, dt)],
    };
    let g3 = linus_ints::reference::eri_grad_3c(
        basis,
        pairs,
        aux,
        &ap,
        &dens,
        1e-13,
        &linus_ints::operator::Operator::Coulomb,
    );
    let g2 = linus_ints::reference::oneshot::grad_2c(
        aux,
        &ap,
        &gam,
        1e-13,
        &linus_ints::operator::Operator::Coulomb,
    );
    g3.iter()
        .zip(&g2)
        .map(|(a, b)| [a[0] + b[0], a[1] + b[1], a[2] + b[2]])
        .collect()
}

fn opts(method: &str, jk: JkMethod) -> CalcOptions {
    CalcOptions {
        method: Some(method.into()),
        basis: Some("def2-svp".into()),
        jk: jk.into(),
        threads: Some(2),
        scf: crate::driver::ScfSettings {
            convergence: Some(crate::driver::options::Auto::Value(
                crate::driver::ConvergencePreset::Extreme,
            )),
            ..Default::default()
        },
        ..Default::default()
    }
}

/// Without range separation the blocked DF gradient equals the dense reference code
/// (global hybrid and J only, RKS and UKS).
#[test]
fn df_gradient_of_global_hybrid_matches_dense_code() {
    for (mult, method, a) in [
        (1, "b3lyp", 0.2),
        (3, "b3lyp", 0.2),
        (1, "pbe", 0.0),
        (3, "pbe", 0.0),
    ] {
        let mol = Structure::from_xyz_str(WATER, 0, mult).unwrap();
        let mut c = Session::new(
            mol,
            opts(
                method,
                JkMethod::Df {
                    aux: "def2-universal-jkfit".into(),
                },
            ),
        )
        .unwrap();
        c.gradient().unwrap();
        let r = c.last().unwrap().clone();
        let b = &c.cx.basis;
        let pairs = PairSet::build(b, PairOptions::with_order(DerivOrder::One));
        let adef = c.cx.plan.load_basis("def2-universal-jkfit").unwrap();
        let aux = BasisSet::new(&adef, &c.cx.mol.z, &c.cx.mol.positions, true).unwrap();
        // densities from the occupied orbitals, as the gradient code forms them
        let ca = r.c_a.subcols(0, r.nalpha).to_owned();
        let cb = r.c_b.subcols(0, r.nbeta).to_owned();
        let d = &ca * ca.transpose() + &cb * cb.transpose();
        let dt = linus_base::linalg::mat_to_rowmajor(d.as_ref());
        let occ: Vec<&Mat<f64>> = if r.restricted {
            vec![&ca]
        } else {
            vec![&ca, &cb]
        };
        let mut t = linus_core::timing::Timers::default();
        let g = df_gradient(
            b,
            &pairs,
            &aux,
            &linus_jk::df::CoulombMetric::new(&aux, faer::Par::Seq).unwrap(),
            &dt,
            &occ,
            r.restricted,
            Operator::Attenuated {
                coulomb: a,
                long_range: 0.0,
                omega: 0.0,
            },
            Par::Seq,
            &mut t,
        );
        let g_ref = dense_df_gradient(b, &pairs, &aux, &dt, &occ, r.restricted, a, Par::Seq);
        for (u, v) in g.iter().zip(&g_ref) {
            for k in 0..3 {
                // the 2e part is large (~4 Eh/bohr); the codes' 3-center integrals differ in screening
                assert!(
                    (u[k] - v[k]).abs() < 1e-11 * v[k].abs().max(1.0),
                    "{method} mult {mult}: {u:?} {v:?}"
                );
            }
        }
    }
}

/// Pure functionals (J only) and the dense code agree; the gradient matches finite
/// differences (driver energies).
#[test]
fn pure_functional_df_gradient() {
    let mol = Structure::from_xyz_str(WATER, 0, 1).unwrap();
    let o = opts(
        "pbe",
        JkMethod::Df {
            aux: "def2-universal-jkfit".into(),
        },
    );
    let mut c = Session::new(mol.clone(), o.clone()).unwrap();
    let g = c.gradient().unwrap().gradient;
    let h = 2e-3;
    let e = |d: f64| {
        let mut m = mol.clone();
        m.positions[1][0] += d;
        Session::new(m, o.clone()).unwrap().energy().unwrap().energy
    };
    let fd = (8.0 * (e(h) - e(-h)) - (e(2.0 * h) - e(-2.0 * h))) / (12.0 * h);
    assert!((fd - g[1][0]).abs() < 2e-9, "analytic {} fd {fd}", g[1][0]);
}
