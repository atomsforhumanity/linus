//! Memory-lean RI-J nuclear gradient (Coulomb part of jk = rij-snk), integral-direct in
//! Hermite space (split-RI-J derivatives; no three-index tensor):
//!
//!   E_J = 1/2 gamma^T V^-1 gamma,   gamma_P = sum_mn (P|mn) D_mn,   c = V^-1 gamma,
//!   dE_J/dX = sum_P c_P sum_mn d(P|mn)/dX D_mn - 1/2 sum_PQ c_P c_Q d(P|Q)/dX.
//!
//! The integral work is linus-ints' [`Eri3DerivPlan`]: gamma ([`Eri3DerivPlan::gamma`]), the
//! three-center term ([`Eri3DerivPlan::coulomb_gradient`]: all three centres of every
//! significant orbital shell pair against all auxiliary lane batches) and the metric term
//! ([`Eri3DerivPlan::metric_gradient`] with w = -c/2). Here: the fit. Memory: the metric
//! (naux^2) for the fit, O(naux) per thread otherwise. The plan's reductions run over fixed
//! chunks in order, so the result does not depend on the thread count.

use faer::{Mat, MatMut, Par};
use linus_base::linalg::ao_view;
use linus_core::gradient::Gradient;
use linus_core::timing::Timers;
use linus_ints::eri3c::{self, DfThresholds};
use linus_ints::operator::Operator;
use linus_ints::{BasisSet, DerivOrder, Eri3DerivOptions, Eri3DerivPlan, PairOptions, PairSet};

/// RI-J energy gradient for the total density `d` (symmetric, nbf x nbf) with the screening
/// `thresholds` (`pair` and `prim`). Returns one row per atom up to the last with shells;
/// timings go to `timers` ("rij_grad_gamma", "rij_grad_3c", "rij_grad_2c", "rij_grad_fit").
pub fn rij_gradient(
    basis: &BasisSet,
    pairs: &PairSet,
    aux: &BasisSet,
    d: &Mat<f64>,
    thresholds: &DfThresholds,
    par: Par,
    timers: &mut Timers,
) -> Vec<[f64; 3]> {
    let natom = basis
        .shells
        .iter()
        .chain(&aux.shells)
        .map(|s| s.atom + 1)
        .max()
        .unwrap_or(0);
    let naux = aux.nbf;
    let ap = PairSet::aux(aux, PairOptions::default());
    let opts = Eri3DerivOptions {
        thresholds: *thresholds,
    };
    let plan = Eri3DerivPlan::new(basis, pairs, aux, &ap, DerivOrder::One, opts)
        .expect("orbital pairs of order 1 in shell-index orientation");
    let mut ws = plan.workspace(rayon::current_num_threads());
    let dv = ao_view(d.as_ref());
    let mut gamma = vec![0.0; naux];
    timers.time("rij_grad_gamma", || plan.gamma(dv, &mut gamma, &mut ws));
    // c = V^-1 gamma
    let c = timers.time("rij_grad_fit", || {
        let v = eri3c::metric(plan.aux3c(), 0.0, &Operator::Coulomb);
        let vm = Mat::from_fn(naux, naux, |i, j| v[i * naux + j]);
        // sequential: the result must not depend on the thread count (naux^3/3 flops, ~0.1 s
        // at naux = 3000)
        let _ = par;
        let l = linus_base::linalg::cholesky_in_place(vm, Par::Seq)
            .expect("auxiliary metric not positive definite");
        let mut c = gamma.clone();
        let mut m = MatMut::from_column_major_slice_mut(&mut c, naux, 1);
        linus_base::linalg::solve_lower_in_place(l.as_ref(), m.as_mut(), Par::Seq);
        linus_base::linalg::solve_upper_in_place(l.transpose(), m, Par::Seq);
        c
    });
    // three-center term: sum_P c_P sum_mn d(P|mn) D_mn
    let mut g3 = vec![[0.0; 3]; plan.natom()];
    timers.time("rij_grad_3c", || {
        plan.coulomb_gradient(dv, &c, &mut g3, &mut ws)
    });
    // metric term: -1/2 sum_PQ c_P c_Q d(P|Q)
    let w: Vec<f64> = c.iter().map(|x| -0.5 * x).collect();
    let mut g2 = vec![[0.0; 3]; plan.natom()];
    timers.time("rij_grad_2c", || {
        plan.metric_gradient(&w, &c, &mut g2, &mut ws)
    });
    let mut g = Gradient::from(g3);
    g += &g2;
    let mut g = g.into_inner();
    g.truncate(natom);
    g
}

#[cfg(test)]
mod tests {
    use super::*;
    use linus_ints::basis::BasisDef;
    use linus_ints::reference::Dens3c;
    use linus_ints::DerivOrder;

    #[test]
    fn matches_reference_df_gradient() {
        let json = r#"{"name":"t","elements":{"8":{"electron_shells":[
          {"function_type":"gto","angular_momentum":[0],"exponents":["130.7","23.81","6.44"],"coefficients":[["0.154","0.535","0.444"]]},
          {"function_type":"gto","angular_momentum":[1],"exponents":["5.03","1.17","0.38"],"coefficients":[["0.156","0.607","0.392"]]},
          {"function_type":"gto","angular_momentum":[2],"exponents":["0.8"],"coefficients":[["1.0"]]},
          {"function_type":"gto","angular_momentum":[3],"exponents":["0.7"],"coefficients":[["1.0"]]}]},
          "1":{"electron_shells":[{"function_type":"gto","angular_momentum":[0],"exponents":["3.42","0.62"],"coefficients":[["0.15","0.53"]]},
          {"function_type":"gto","angular_momentum":[1],"exponents":["0.9"],"coefficients":[["1.0"]]}]}}}"#;
        let ajson = r#"{"name":"a","elements":{"8":{"electron_shells":[
          {"function_type":"gto","angular_momentum":[0],"exponents":["20.0"],"coefficients":[["1.0"]]},
          {"function_type":"gto","angular_momentum":[0],"exponents":["2.0"],"coefficients":[["1.0"]]},
          {"function_type":"gto","angular_momentum":[1],"exponents":["1.5"],"coefficients":[["1.0"]]},
          {"function_type":"gto","angular_momentum":[2],"exponents":["1.0"],"coefficients":[["1.0"]]},
          {"function_type":"gto","angular_momentum":[3],"exponents":["0.9"],"coefficients":[["1.0"]]}]},
          "1":{"electron_shells":[{"function_type":"gto","angular_momentum":[0],"exponents":["1.0"],"coefficients":[["1.0"]]},
          {"function_type":"gto","angular_momentum":[1],"exponents":["0.8"],"coefficients":[["1.0"]]},
          {"function_type":"gto","angular_momentum":[2],"exponents":["0.7"],"coefficients":[["1.0"]]}]}}}"#;
        let xyz = [[0.0, 0.1, 0.0], [0.2, 1.4, 1.1], [0.0, -1.4, 1.0]];
        let basis = BasisSet::new(
            &BasisDef::from_bse_json(json).unwrap(),
            &[8, 1, 1],
            &xyz,
            true,
        )
        .unwrap();
        let aux = BasisSet::new(
            &BasisDef::from_bse_json(ajson).unwrap(),
            &[8, 1, 1],
            &xyz,
            true,
        )
        .unwrap();
        let pairs = PairSet::build(&basis, PairOptions::with_order(DerivOrder::One));
        let n = basis.nbf;
        let d = Mat::from_fn(n, n, |i, j| {
            (((i + 1) * (j + 1)) as f64 * 0.37).cos() * 0.1 + if i == j { 1.0 } else { 0.0 }
        });
        let d = Mat::from_fn(n, n, |i, j| 0.5 * (d[(i, j)] + d[(j, i)]));
        let mut t = Timers::default();
        let got = rij_gradient(
            &basis,
            &pairs,
            &aux,
            &d,
            &DfThresholds::none(),
            Par::Seq,
            &mut t,
        );
        // reference: generic 3c/2c derivative contractions
        let naux = aux.nbf;
        let ap = PairSet::aux(&aux, PairOptions::with_order(DerivOrder::One));
        let j3 = linus_ints::reference::eri_3c(&basis, &pairs, &aux, &ap);
        let df = linus_base::linalg::mat_to_rowmajor(d.as_ref());
        let gamma: Vec<f64> = (0..naux)
            .map(|p| {
                j3[p * n * n..(p + 1) * n * n]
                    .iter()
                    .zip(&df)
                    .map(|(a, b)| a * b)
                    .sum()
            })
            .collect();
        let v = linus_base::linalg::mat_from_rowmajor(
            naux,
            naux,
            &linus_ints::reference::eri_2c(&aux, &ap),
        );
        let l = linus_base::linalg::cholesky(v.as_ref(), Par::Seq).unwrap();
        let g = Mat::from_fn(naux, 1, |i, _| gamma[i]);
        let cm = linus_base::linalg::cholesky_solve(l.as_ref(), g.as_ref(), Par::Seq);
        let cv: Vec<f64> = (0..naux).map(|i| cm[(i, 0)]).collect();
        let gam: Vec<f64> = (0..naux * naux)
            .map(|x| -0.5 * cv[x / naux] * cv[x % naux])
            .collect();
        let dens = Dens3c {
            full: None,
            aux_shells: 0..aux.shells.len(),
            separable: vec![(1.0, &cv, &df)],
        };
        let g3 = linus_ints::reference::eri_grad_3c(
            &basis,
            &pairs,
            &aux,
            &ap,
            &dens,
            0.0,
            &linus_ints::operator::Operator::Coulomb,
        );
        let g2 = linus_ints::reference::oneshot::grad_2c(
            &aux,
            &ap,
            &gam,
            0.0,
            &linus_ints::operator::Operator::Coulomb,
        );
        let mut worst: f64 = 0.0;
        for a in 0..3 {
            for k in 0..3 {
                let want = g3[a][k] + g2[a][k];
                worst = worst.max((got[a][k] - want).abs());
            }
        }
        assert!(worst < 1e-11, "worst {worst:e}");
        // thread-count independence
        let run = |t: usize| {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(t)
                .build()
                .unwrap();
            pool.install(|| {
                rij_gradient(
                    &basis,
                    &pairs,
                    &aux,
                    &d,
                    &DfThresholds::default(),
                    Par::rayon(t),
                    &mut Timers::default(),
                )
            })
        };
        let (g1, g5) = (run(1), run(5));
        for (a, b) in g1.iter().zip(&g5) {
            for k in 0..3 {
                assert_eq!(a[k].to_bits(), b[k].to_bits());
            }
        }
        // translational invariance
        for k in 0..3 {
            let s: f64 = got.iter().map(|g| g[k]).sum();
            assert!(s.abs() < 1e-11, "sum {s:e}");
        }
    }
}
