//! linus-dev: the study, check and benchmark programs of linus development, in `examples/`
//! (`cargo build --release -p linus-dev --examples`; the binaries land in
//! `target/release/examples/`, where the scripts in bench/ and tests/python/ look for them).
//! They use the `linus` crate's internals (its `#[doc(hidden)]` modules), which are not a
//! stable interface. Not published.
//!
//! The library holds what several programs share.

use linus_ints::{AoMatrixMut, BasisSet, DerivOrder, OneEPlan, PairSet};

/// The overlap S, the kinetic energy T and the potential V of the point charges (q, C)
/// (nbf x nbf, row-major), from linus-ints' one-electron plan.
pub fn one_electron(
    basis: &BasisSet,
    pairs: &PairSet,
    charges: &[(f64, [f64; 3])],
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let n = basis.nbf;
    let (mut s, mut t, mut v) = (vec![0.0; n * n], vec![0.0; n * n], vec![0.0; n * n]);
    let plan = OneEPlan::new(basis, DerivOrder::Zero)
        .with_charges(pairs, charges)
        .expect("the potential takes pairs of any order");
    plan.overlap_kinetic(
        &mut AoMatrixMut::row_major(&mut s, n),
        &mut AoMatrixMut::row_major(&mut t, n),
    );
    plan.potential(&mut AoMatrixMut::row_major(&mut v, n));
    (s, t, v)
}

/// The overlap matrix S (nbf x nbf, row-major), from linus-ints' one-electron plan.
pub fn overlap(basis: &BasisSet) -> Vec<f64> {
    let n = basis.nbf;
    let mut s = vec![0.0; n * n];
    OneEPlan::new(basis, DerivOrder::Zero).overlap(&mut AoMatrixMut::row_major(&mut s, n));
    s
}
