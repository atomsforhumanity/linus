//! The second-derivative integral contractions of the Hessians (Eri4DerivPlan::hessian,
//! eri_hess_3c, the 2c Hessian plan) are bitwise independent of the thread count: 1 and 8 threads give
//! identical bits (deterministic reductions of `linus_num::det`).

mod common;
use common::*;
use linus_ints::basis::BasisDef;
use linus_ints::contract::DensityTerms;
use linus_ints::reference::oneshot;
use linus_ints::reference::Dens3c;
use linus_ints::{BasisSet, DerivOrder, PairOptions, PairSet};

/// Methanol (bohr), low symmetry.
fn methanol() -> (Vec<u32>, Vec<[f64; 3]>) {
    let z = vec![6, 8, 1, 1, 1, 1];
    let c = vec![
        [0.02, -0.03, 0.01],
        [2.69, 0.05, -0.04],
        [3.27, 1.71, 0.12],
        [-0.72, 1.93, 0.08],
        [-0.71, -1.02, 1.68],
        [-0.69, -0.98, -1.71],
    ];
    (z, c)
}

#[test]
fn eri_hess_contractions_are_thread_count_independent() {
    let (z, positions) = methanol();
    let basis = BasisSet::new(
        &BasisDef::load("def2-svp", None).unwrap(),
        &z,
        &positions,
        true,
    )
    .unwrap();
    let aux = BasisSet::new(
        &BasisDef::load("def2-universal-jkfit", None).unwrap(),
        &z,
        &positions,
        true,
    )
    .unwrap();
    let (n, naux) = (basis.nbf, aux.nbf);
    let dt = random_sym(n, 1);
    let da: Vec<f64> = random_sym(n, 2).iter().map(|x| 0.3 * x).collect();
    let full: Vec<f64> = (0..naux)
        .flat_map(|p| random_sym(n, 100 + p as u64))
        .collect();
    let cvec = random_vec(naux, 3);
    let gam2 = random_sym(naux, 4);
    let run = |t: usize| -> Vec<Vec<f64>> {
        rayon::ThreadPoolBuilder::new()
            .num_threads(t)
            .build()
            .unwrap()
            .install(|| {
                let pairs = PairSet::build(&basis, PairOptions::with_order(DerivOrder::Two));
                let ap = PairSet::aux(&aux, PairOptions::with_order(DerivOrder::Two));
                let dens = Dens3c {
                    full: Some(&full),
                    aux_shells: 0..aux.shells.len(),
                    separable: vec![(0.7, &cvec, &dt)],
                };
                vec![
                    oneshot::hess_4c(
                        &basis,
                        &pairs,
                        &DensityTerms::default().coulomb(0.5, &dt, &dt).exchange(
                            -0.25 * 0.25,
                            &dt,
                            &dt,
                        ),
                        1e-12,
                        &linus_ints::operator::Operator::Coulomb,
                    ),
                    oneshot::hess_4c(
                        &basis,
                        &pairs,
                        &DensityTerms::default()
                            .coulomb(0.5, &dt, &dt)
                            .exchange(-0.5 * 0.25, &da, &da)
                            .exchange(-0.5 * 0.25, &dt, &dt),
                        1e-12,
                        &linus_ints::operator::Operator::Coulomb,
                    ),
                    linus_ints::reference::eri_hess_3c(
                        &basis,
                        &pairs,
                        &aux,
                        &ap,
                        &dens,
                        1e-14,
                        &linus_ints::operator::Operator::Coulomb,
                    ),
                    linus_ints::reference::oneshot::hess_2c(
                        &aux,
                        &ap,
                        &gam2,
                        1e-14,
                        &linus_ints::operator::Operator::Coulomb,
                    ),
                ]
            })
    };
    let r1 = run(1);
    let r8 = run(8);
    for (name, (a, b)) in [
        "eri_hess_4c rhf",
        "eri_hess_4c uhf",
        "eri_hess_3c",
        "eri_hess_2c",
    ]
    .iter()
    .zip(r1.iter().zip(&r8))
    {
        assert!(a.iter().any(|x| *x != 0.0), "{name}: all zero");
        let same = a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits());
        let md = a
            .iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f64::max);
        assert!(same, "{name}: 1 and 8 threads differ (max {md:.2e})");
    }
}
