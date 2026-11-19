//! Two-electron gradient terms of hybrids with range-separated exact exchange: exchange
//! a K + b K_LR(omega), K_LR the exchange of erf(omega r)/r (libxc's (alpha, beta, omega):
//! a = alpha + beta, b = -beta).
//!
//! Exact integrals ([`exact_gradient`]): J with 1/r and the whole exchange with the combined
//! operator a/r + b erf(omega r)/r ([`Eri4DerivPlan`]; one pass over the exchange
//! quartets, two Boys evaluations per primitive quartet). The J/K method picks this or the
//! routines below in `linus::two_electron`.
//!
//! Density fitting: [`crate::df::gradient::df_gradient`] with the long-range term (Psi4's wK
//! scheme, see there).
//!
//! RIJ + COSX: RI-J (`crate::df::rij_gradient`) and the COSX exchange gradient
//! ([`crate::cosx::gradient`]) with the combined operator, on the grid of the final exchange energy
//! (as for global hybrids, [`crate::cosx::rij_cosx`]).

use crate::exact::terms::SpinDensities;
use faer::Mat;
use linus_base::linalg::mat_to_rowmajor;
use linus_core::gradient::Gradient;
use linus_ints::contract::DensityTerms;
use linus_ints::operator::Operator;
use linus_ints::{
    BasisSet, DerivOrder, Eri4DerivOptions, Eri4DerivPlan, Eri4DerivWorkspace, PairSet,
};

/// The four-center gradient of `terms` with `op`, on the workspace `ws` (made on first use;
/// one workspace serves every operator over the same pairs, J's first: its scratch is sized
/// for the Coulomb terms).
pub fn grad_4c(
    basis: &BasisSet,
    pairs: &PairSet,
    terms: &DensityTerms,
    thresh: f64,
    op: &Operator,
    ws: &mut Option<Eri4DerivWorkspace>,
) -> Vec<[f64; 3]> {
    let opts = Eri4DerivOptions { thresh, op: *op };
    let plan = Eri4DerivPlan::new(basis, pairs, DerivOrder::One, opts)
        .expect("exact gradients take index-ordered pairs of derivative order >= 1");
    let ws = ws.get_or_insert_with(|| plan.workspace(terms.coulomb.len()));
    let mut g = vec![[0.0; 3]; basis.natom()];
    plan.gradient(terms, &mut g, ws);
    g
}

/// Exact four-center two-electron gradient (J + a K + b K_LR): J with 1/r, then all of the
/// exchange with the combined operator a/r + b erf(w r)/r (two Boys evaluations per primitive
/// quartet in the exchange pass; la18 wB97X-D3BJ/def2-SVP, 8 threads: 2.0 s against 2.6 s for
/// J + a K with 1/r and b K_LR in a second pass). `dt`, `da`, `db` are the total and spin
/// densities (row-major nbf x nbf); restricted: `da`/`db` unused, exchange from dt. `op` the
/// exchange operator (coefficients included). `pairs`: index-ordered, derivative order >= 1.
#[allow(clippy::too_many_arguments)]
pub fn exact_gradient(
    basis: &BasisSet,
    pairs: &PairSet,
    dt: &[f64],
    da: &[f64],
    db: &[f64],
    restricted: bool,
    op: Operator,
    thresh: f64,
) -> Vec<[f64; 3]> {
    let spins = SpinDensities::new(restricted, dt, da, db);
    let j = spins.coulomb(DensityTerms::default());
    let mut ws = None;
    let mut g = Gradient::from(grad_4c(
        basis,
        pairs,
        &j,
        thresh,
        &linus_ints::operator::Operator::Coulomb,
        &mut ws,
    ));
    // exchange terms with unit coefficient (the operator carries a, b)
    let k = spins.exchange(DensityTerms::default(), 1.0);
    g += grad_4c(basis, pairs, &k, thresh, &op, &mut ws);
    g.into_inner()
}

/// The exact four-center two-electron gradient at the spin densities `da`, `db` (and `dtf` =
/// D_a + D_b row-major) with the exact exchange `xop`: J - a_x K for HF and global hybrids; for
/// range-separated hybrids J with 1/r and the exchange with the combined operator
/// ([`exact_gradient`]).
#[allow(clippy::too_many_arguments)]
pub fn gradient_2e(
    basis: &BasisSet,
    pairs: &PairSet,
    dtf: &[f64],
    da: &Mat<f64>,
    db: &Mat<f64>,
    restricted: bool,
    xop: Option<Operator>,
    thresh: f64,
) -> Vec<[f64; 3]> {
    let (daf, dbf) = (mat_to_rowmajor(da.as_ref()), mat_to_rowmajor(db.as_ref()));
    let (a_x, op) = linus_base::terms::exchange_parts(xop);
    match op {
        // J with 1/r, the exchange with the combined operator
        Some(x) => exact_gradient(basis, pairs, dtf, &daf, &dbf, restricted, x, thresh),
        None => {
            let terms = SpinDensities::new(restricted, dtf, &daf, &dbf).coulomb_exchange(a_x);
            grad_4c(
                basis,
                pairs,
                &terms,
                thresh,
                &linus_ints::operator::Operator::Coulomb,
                &mut None,
            )
        }
    }
}
