//! Explicit two-electron second derivatives of Hessians with exact four-center integrals: the
//! second derivatives of the integrals contracted with the densities and the two-electron part
//! of F^x C (J^x and K^x), for HF, global and range-separated hybrids.

use crate::exact::terms::SpinDensities;
use faer::{Mat, MatRef, Par};
use linus_base::linalg::mat_from_rowmajor;
use linus_base::linalg::mm;
use linus_core::timing::Timers;
use linus_ints::contract::DensityTerms;
use linus_ints::operator::Operator;
use linus_ints::{
    eri4c::jk_deriv, BasisSet, DerivOrder, Eri4DerivOptions, Eri4DerivPlan, Eri4DerivWorkspace,
    PairSet,
};

/// The four-center Hessian contribution of `terms` with `op` ((3 natom)^2), on the workspace
/// `ws` (made on first use; one workspace serves every operator over the same pairs).
fn hess_4c(
    basis: &BasisSet,
    pairs: &PairSet,
    terms: &DensityTerms,
    thresh: f64,
    op: &Operator,
    ws: &mut Option<Eri4DerivWorkspace>,
) -> Vec<f64> {
    let opts = Eri4DerivOptions { thresh, op: *op };
    let plan = Eri4DerivPlan::new(basis, pairs, DerivOrder::Two, opts)
        .expect("exact Hessians take index-ordered pairs of derivative order 2");
    let ws = ws.get_or_insert_with(|| plan.workspace(terms.coulomb.len()));
    let n3 = 3 * basis.natom();
    let mut h = vec![0.0; n3 * n3];
    plan.hessian(terms, &mut h, ws);
    h
}

/// The explicit two-electron terms of a Hessian with exact integrals, added to `h` ((3N)^2
/// row-major), and the two-electron part of F_s^x C_s = (J^x\[D_t\] - a_x K^x\[D_s\]) C_s, added to
/// `fxo[s][x]` (see `linus::two_electron`'s `cpu_hessian_2e` for the arguments): four-center
/// second derivatives and J^x/K^x; range-separated hybrids: the exchange with the operator `xop`
/// in one pass (J with 1/r).
#[allow(clippy::too_many_arguments)]
pub fn hessian_2e(
    basis: &BasisSet,
    pairs: &PairSet,
    occ: &[MatRef<'_, f64>],
    ds: &[&[f64]],
    dtf: &[f64],
    xop: Option<Operator>,
    thresh: f64,
    h: &mut [f64],
    fxo: &mut [Vec<Mat<f64>>],
    par: Par,
    timers: &mut Timers,
) {
    let n = basis.nbf;
    let n2 = n * n;
    let nch = occ.len();
    let add = |h: &mut [f64], x: &[f64]| h.iter_mut().zip(x).for_each(|(a, b)| *a += b);
    // closed shell: J^x of D_t and K^x of D_a; open shell: J^x and K^x of both spins,
    // J^x[D_t] = J^x[D_a] + J^x[D_b]
    let jk_dens: Vec<&[f64]> = if nch == 1 {
        vec![dtf, ds[0]]
    } else {
        vec![ds[0], ds[1]]
    };
    let spins = if nch == 1 {
        SpinDensities::Closed { dt: dtf }
    } else {
        SpinDensities::Open {
            dt: dtf,
            da: ds[0],
            db: ds[1],
        }
    };
    let terms = |a_x: f64| spins.coulomb_exchange(a_x);
    let (a_x, op) = linus_base::terms::exchange_parts(xop);
    let (jx, kx) = match op {
        None => {
            add(
                h,
                &timers.time("hess_2e", || {
                    hess_4c(
                        basis,
                        pairs,
                        &terms(a_x),
                        thresh,
                        &linus_ints::operator::Operator::Coulomb,
                        &mut None,
                    )
                }),
            );
            timers.time("hess_fock_deriv", || {
                jk_deriv::jk_deriv(
                    basis,
                    pairs,
                    &jk_dens,
                    true,
                    a_x != 0.0,
                    thresh,
                    &linus_ints::operator::Operator::Coulomb,
                )
            })
        }
        Some(op) => {
            // J with 1/r; the exchange in one pass with a/r + b erf(omega r)/r, on one workspace
            let mut ws = None;
            add(
                h,
                &timers.time("hess_2e", || {
                    hess_4c(
                        basis,
                        pairs,
                        &terms(0.0),
                        thresh,
                        &linus_ints::operator::Operator::Coulomb,
                        &mut ws,
                    )
                }),
            );
            // the exchange with unit coefficient (the operator carries a, b)
            let tk = spins.exchange(DensityTerms::default(), 1.0);
            add(
                h,
                &timers.time("hess_2e", || {
                    hess_4c(basis, pairs, &tk, thresh, &op, &mut ws)
                }),
            );
            let jdens: Vec<&[f64]> = if nch == 1 {
                vec![dtf]
            } else {
                vec![ds[0], ds[1]]
            };
            let (jx, _) = timers.time("hess_fock_deriv", || {
                jk_deriv::jk_deriv(
                    basis,
                    pairs,
                    &jdens,
                    true,
                    false,
                    thresh,
                    &linus_ints::operator::Operator::Coulomb,
                )
            });
            let (_, kx) = timers.time("hess_fock_deriv", || {
                jk_deriv::jk_deriv(basis, pairs, &ds[..nch], false, true, thresh, &op)
            });
            // the layout of the global-hybrid call: K of D_a second for a closed shell
            let kx = if nch == 1 {
                vec![Vec::new(), kx.into_iter().next().unwrap()]
            } else {
                kx
            };
            (jx, kx)
        }
    };
    for p in 0..fxo[0].len() {
        let sl = p * n2..(p + 1) * n2;
        let jt = if nch == 1 {
            mat(&jx[0][sl.clone()], n)
        } else {
            mat(&jx[0][sl.clone()], n) + mat(&jx[1][sl.clone()], n)
        };
        for (s, fs) in fxo.iter_mut().enumerate() {
            let mut g = jt.clone();
            if a_x != 0.0 {
                g -= mat(&kx[if nch == 1 { 1 } else { s }][sl.clone()], n) * a_x;
            }
            fs[p] += mm(g.as_ref(), occ[s], par);
        }
    }
}

fn mat(v: &[f64], n: usize) -> Mat<f64> {
    mat_from_rowmajor(n, n, v)
}
