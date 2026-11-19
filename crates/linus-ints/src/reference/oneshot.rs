//! One-call forms of the plans for tests and tools: a plan and its workspace for a single
//! operation, the result in a new vector. Code that computes more than once (or cares for its
//! memory) builds the plan and keeps the workspace.

use crate::basis::BasisSet;
use crate::contract::{AoMatrix, AoMatrixMut, DensityTerms};
use crate::eri2c::{Eri2DerivOptions, Eri2DerivPlan};
use crate::eri4c::deriv_plan::{Eri4DerivOptions, Eri4DerivPlan};
use crate::one_e::ecp::{EcpSet, EcpStats};
use crate::one_e::ecp_deriv::EcpPlan;
use crate::one_e::OneEPlan;
use crate::operator::Operator;
use crate::pairs::{DerivOrder, PairSet};
use crate::points::pcm_ints::{
    AuxSurfacePlan, GaussPoint, SurfaceDerivPlan, SurfacePlan, SurfaceShape,
};

/// Point charges (q, C).
type Charges<'a> = &'a [(f64, [f64; 3])];

/// [`OneEPlan::overlap_kinetic`]: S and T (nbf x nbf, row-major).
pub fn overlap_kinetic(basis: &BasisSet) -> (Vec<f64>, Vec<f64>) {
    let n = basis.nbf;
    let (mut s, mut t) = (vec![0.0; n * n], vec![0.0; n * n]);
    OneEPlan::new(basis, DerivOrder::Zero).overlap_kinetic(
        &mut AoMatrixMut::row_major(&mut s, n),
        &mut AoMatrixMut::row_major(&mut t, n),
    );
    (s, t)
}

/// [`OneEPlan::potential`] of the point charges (nbf x nbf, row-major).
pub fn potential(basis: &BasisSet, pairs: &PairSet, charges: Charges) -> Vec<f64> {
    let n = basis.nbf;
    let mut v = vec![0.0; n * n];
    let plan = OneEPlan::new(basis, DerivOrder::Zero).with_charges(pairs, charges);
    plan.unwrap()
        .potential(&mut AoMatrixMut::row_major(&mut v, n));
    v
}

/// [`OneEPlan::dipole`] (three nbf x nbf row-major matrices).
pub fn dipole(basis: &BasisSet, origin: [f64; 3]) -> [Vec<f64>; 3] {
    let n = basis.nbf;
    let mut out = [vec![0.0; n * n], vec![0.0; n * n], vec![0.0; n * n]];
    let [x, y, z] = &mut out;
    OneEPlan::new(basis, DerivOrder::Zero).dipole(
        origin,
        &mut [
            AoMatrixMut::row_major(x, n),
            AoMatrixMut::row_major(y, n),
            AoMatrixMut::row_major(z, n),
        ],
    );
    out
}

/// [`OneEPlan::overlap_kinetic_deriv`]: dS/dX and dT/dX, each (3 natom) x nbf x nbf.
pub fn overlap_kinetic_deriv(basis: &BasisSet) -> (Vec<f64>, Vec<f64>) {
    let len = 3 * basis.natom() * basis.nbf * basis.nbf;
    let (mut ds, mut dt) = (vec![0.0; len], vec![0.0; len]);
    OneEPlan::new(basis, DerivOrder::One).overlap_kinetic_deriv(Some(&mut ds), Some(&mut dt));
    (ds, dt)
}

/// [`OneEPlan::overlap_kinetic_grad`] (natom rows).
pub fn overlap_kinetic_grad(
    basis: &BasisSet,
    ws: Option<&[f64]>,
    dt: Option<&[f64]>,
) -> Vec<[f64; 3]> {
    let plan = OneEPlan::new(basis, DerivOrder::One);
    let mut g = vec![[0.0; 3]; basis.natom()];
    plan.overlap_kinetic_grad(ws, dt, &mut g, &mut plan.workspace());
    g
}

/// [`OneEPlan::overlap_kinetic_hess`] ((3 natom)^2, row-major).
pub fn overlap_kinetic_hess(basis: &BasisSet, ws: Option<&[f64]>, dt: Option<&[f64]>) -> Vec<f64> {
    let plan = OneEPlan::new(basis, DerivOrder::Two);
    let n3 = 3 * basis.natom();
    let mut h = vec![0.0; n3 * n3];
    plan.overlap_kinetic_hess(ws, dt, &mut h, &mut plan.workspace());
    h
}

/// [`OneEPlan::dipole_deriv`].
pub fn dipole_deriv(basis: &BasisSet, origin: [f64; 3]) -> Vec<f64> {
    let mut out = vec![0.0; 9 * basis.natom() * basis.nbf * basis.nbf];
    OneEPlan::new(basis, DerivOrder::One).dipole_deriv(origin, &mut out);
    out
}

/// [`OneEPlan::dipole_grad`] (out\[atom\]\[k\]\[c\]).
pub fn dipole_grad(basis: &BasisSet, origin: [f64; 3], d: &[f64]) -> Vec<[[f64; 3]; 3]> {
    let plan = OneEPlan::new(basis, DerivOrder::One);
    let mut g = vec![[[0.0; 3]; 3]; basis.natom()];
    plan.dipole_grad(origin, d, &mut g, &mut plan.workspace());
    g
}

/// The plan of the point charges' potential of order `order`.
fn charges_plan<'a>(
    basis: &'a BasisSet,
    pairs: &'a PairSet,
    charges: Charges<'a>,
    order: DerivOrder,
) -> OneEPlan<'a> {
    OneEPlan::new(basis, order)
        .with_charges(pairs, charges)
        .unwrap()
}

/// [`OneEPlan::potential_deriv`]: (3 (natom + ncharge)) x nbf x nbf.
pub fn potential_deriv(basis: &BasisSet, pairs: &PairSet, charges: Charges) -> Vec<f64> {
    let nslots = basis.natom() + charges.len();
    let mut dv = vec![0.0; 3 * nslots * basis.nbf * basis.nbf];
    charges_plan(basis, pairs, charges, DerivOrder::One).potential_deriv(&mut dv);
    dv
}

/// [`OneEPlan::potential_grad`]: (the basis-center part per atom, the charge-position part
/// per charge).
pub fn potential_grad(
    basis: &BasisSet,
    pairs: &PairSet,
    charges: Charges,
    d: &[f64],
) -> (Vec<[f64; 3]>, Vec<[f64; 3]>) {
    let plan = charges_plan(basis, pairs, charges, DerivOrder::One);
    let mut g = vec![[0.0; 3]; basis.natom() + charges.len()];
    plan.potential_grad(d, &mut g, &mut plan.workspace());
    let gc = g.split_off(basis.natom());
    (g, gc)
}

/// [`OneEPlan::potential_hess`]: (3 (natom + ncharge))^2.
pub fn potential_hess(basis: &BasisSet, pairs: &PairSet, charges: Charges, d: &[f64]) -> Vec<f64> {
    let plan = charges_plan(basis, pairs, charges, DerivOrder::Two);
    let n3 = 3 * (basis.natom() + charges.len());
    let mut h = vec![0.0; n3 * n3];
    plan.potential_hess(d, &mut h, &mut plan.workspace());
    h
}

/// [`OneEPlan::nuclear_attraction_grad`] (natom rows; charge i on atom i).
pub fn nuclear_attraction_grad(
    basis: &BasisSet,
    pairs: &PairSet,
    charges: Charges,
    d: &[f64],
) -> Vec<[f64; 3]> {
    let plan = charges_plan(basis, pairs, charges, DerivOrder::One);
    let mut g = vec![[0.0; 3]; basis.natom()];
    plan.nuclear_attraction_grad(d, &mut g, &mut plan.workspace());
    g
}

/// [`OneEPlan::nuclear_attraction_hess`] ((3 natom)^2; charge i on atom i).
pub fn nuclear_attraction_hess(
    basis: &BasisSet,
    pairs: &PairSet,
    charges: Charges,
    d: &[f64],
) -> Vec<f64> {
    let plan = charges_plan(basis, pairs, charges, DerivOrder::Two);
    let n3 = 3 * basis.natom();
    let mut h = vec![0.0; n3 * n3];
    plan.nuclear_attraction_hess(d, &mut h, &mut plan.workspace());
    h
}

/// [`OneEPlan::hcore_deriv`]: (3 natom) x nbf x nbf (charge i on atom i).
pub fn hcore_deriv(basis: &BasisSet, pairs: &PairSet, charges: Charges) -> Vec<f64> {
    let mut h = vec![0.0; 3 * basis.natom() * basis.nbf * basis.nbf];
    charges_plan(basis, pairs, charges, DerivOrder::One).hcore_deriv(&mut h);
    h
}

/// [`EcpPlan::matrix`] (nbf x nbf, row-major) with its statistics.
pub fn ecp_matrix_stats(basis: &BasisSet, pairs: &PairSet, ecp: &EcpSet) -> (Vec<f64>, EcpStats) {
    let n = basis.nbf;
    let mut v = vec![0.0; n * n];
    let plan = EcpPlan::new(basis, pairs, ecp, DerivOrder::Zero);
    let stats = plan.matrix(&mut AoMatrixMut::row_major(&mut v, n));
    (v, stats)
}

/// [`EcpPlan::matrix`] (nbf x nbf, row-major).
pub fn ecp_matrix(basis: &BasisSet, pairs: &PairSet, ecp: &EcpSet) -> Vec<f64> {
    ecp_matrix_stats(basis, pairs, ecp).0
}

/// [`EcpPlan::gradient`] (natom rows).
pub fn ecp_grad(basis: &BasisSet, pairs: &PairSet, ecp: &EcpSet, d: &[f64]) -> Vec<[f64; 3]> {
    let plan = EcpPlan::new(basis, pairs, ecp, DerivOrder::One);
    let mut g = vec![[0.0; 3]; basis.natom()];
    plan.gradient(d, &mut g, &mut plan.workspace());
    g
}

/// [`EcpPlan::add_deriv_matrices`] into zeros: (3 natom) x nbf x nbf.
pub fn ecp_deriv(basis: &BasisSet, pairs: &PairSet, ecp: &EcpSet) -> Vec<f64> {
    let mut dv = vec![0.0; 3 * basis.natom() * basis.nbf * basis.nbf];
    let plan = EcpPlan::new(basis, pairs, ecp, DerivOrder::One);
    plan.add_deriv_matrices(&mut dv, &mut plan.workspace());
    dv
}

/// [`EcpPlan::hessian`] ((3 natom)^2, row-major).
pub fn ecp_hess(basis: &BasisSet, pairs: &PairSet, ecp: &EcpSet, d: &[f64]) -> Vec<f64> {
    let plan = EcpPlan::new(basis, pairs, ecp, DerivOrder::Two);
    let n3 = 3 * basis.natom();
    let mut h = vec![0.0; n3 * n3];
    plan.hessian(d, &mut h, &mut plan.workspace());
    h
}

/// [`Eri4DerivPlan::gradient`] of `terms` with `op` at `thresh` (natom rows).
pub fn grad_4c(
    basis: &BasisSet,
    pairs: &PairSet,
    terms: &DensityTerms,
    thresh: f64,
    op: &Operator,
) -> Vec<[f64; 3]> {
    let opts = Eri4DerivOptions { thresh, op: *op };
    let plan = Eri4DerivPlan::new(basis, pairs, DerivOrder::One, opts).unwrap();
    let mut ws = plan.workspace(terms.coulomb.len());
    let mut g = vec![[0.0; 3]; basis.natom()];
    plan.gradient(terms, &mut g, &mut ws);
    g
}

/// [`Eri4DerivPlan::hessian`] of `terms` with `op` at `thresh` ((3 natom)^2, row-major).
pub fn hess_4c(
    basis: &BasisSet,
    pairs: &PairSet,
    terms: &DensityTerms,
    thresh: f64,
    op: &Operator,
) -> Vec<f64> {
    let opts = Eri4DerivOptions { thresh, op: *op };
    let plan = Eri4DerivPlan::new(basis, pairs, DerivOrder::Two, opts).unwrap();
    let mut ws = plan.workspace(terms.coulomb.len());
    let n3 = 3 * basis.natom();
    let mut h = vec![0.0; n3 * n3];
    plan.hessian(terms, &mut h, &mut ws);
    h
}

/// [`Eri2DerivPlan::gradient`] of the naux x naux row-major `gamma` with `op` at `thresh`
/// (natom rows); `apairs`: [`PairSet::aux`] of order >= 1.
pub fn grad_2c(
    aux: &BasisSet,
    apairs: &PairSet,
    gamma: &[f64],
    thresh: f64,
    op: &Operator,
) -> Vec<[f64; 3]> {
    let opts = Eri2DerivOptions { thresh, op: *op };
    let plan = Eri2DerivPlan::new(aux, apairs, DerivOrder::One, opts).unwrap();
    let mut ws = plan.workspace();
    let mut g = vec![[0.0; 3]; aux.natom()];
    plan.gradient(AoMatrix::row_major(gamma, aux.nbf), &mut g, &mut ws);
    g
}

/// [`Eri2DerivPlan::hessian`] of the naux x naux row-major `gamma` with `op` at `thresh`
/// ((3 natom)^2, row-major); `apairs`: [`PairSet::aux`] of order 2.
pub fn hess_2c(
    aux: &BasisSet,
    apairs: &PairSet,
    gamma: &[f64],
    thresh: f64,
    op: &Operator,
) -> Vec<f64> {
    let opts = Eri2DerivOptions { thresh, op: *op };
    let plan = Eri2DerivPlan::new(aux, apairs, DerivOrder::Two, opts).unwrap();
    let mut ws = plan.workspace();
    let n3 = 3 * aux.natom();
    let mut h = vec![0.0; n3 * n3];
    plan.hessian(AoMatrix::row_major(gamma, aux.nbf), &mut h, &mut ws);
    h
}

/// The rows of the Coulomb metric's first-derivative integrals for the auxiliary functions
/// on `atom`, from [`Eri2DerivPlan::deriv_integrals`]: the atom's functions and
/// `rows[(k * na + p) * naux + q]` = (∂_k p|q) for its na functions p (3 x na x naux).
pub fn metric_deriv_rows(
    aux: &BasisSet,
    apairs: &PairSet,
    atom: usize,
) -> (std::ops::Range<usize>, Vec<f64>) {
    let naux = aux.nbf;
    let plan =
        Eri2DerivPlan::new(aux, apairs, DerivOrder::One, Eri2DerivOptions::default()).unwrap();
    let mut d = vec![0.0; 3 * naux * naux];
    plan.deriv_integrals(&mut d);
    let range = aux.atom_functions(atom);
    let mut rows = Vec::with_capacity(3 * range.len() * naux);
    for k in 0..3 {
        rows.extend_from_slice(&d[(k * naux + range.start) * naux..(k * naux + range.end) * naux]);
    }
    (range, rows)
}

/// [`SurfacePlan::potentials`] of the densities `ds` ([k][point], ndens x npoints).
pub fn surface_potentials(plan: &SurfacePlan, ds: &[&[f64]]) -> Vec<f64> {
    let mut ws = plan.workspace(SurfaceShape {
        densities: ds.len(),
        ..Default::default()
    });
    let mut v = vec![0.0; ds.len() * plan.points().len()];
    plan.potentials(ds, &mut v, &mut ws);
    v
}

/// The nbf x nbf row-major matrices `out` as views.
fn views(out: &mut [Vec<f64>], n: usize) -> Vec<AoMatrixMut<'_>> {
    out.iter_mut()
        .map(|m| AoMatrixMut::row_major(m, n))
        .collect()
}

/// [`SurfacePlan::matrices`] of the charge sets `qs` (nbf x nbf row-major each).
pub fn surface_matrices(plan: &SurfacePlan, qs: &[&[f64]]) -> Vec<Vec<f64>> {
    let n = plan.basis().nbf;
    let mut ws = plan.workspace(SurfaceShape {
        charge_sets: qs.len(),
        ..Default::default()
    });
    let mut m = vec![vec![0.0; n * n]; qs.len()];
    plan.matrices(qs, &mut views(&mut m, n), &mut ws);
    m
}

/// [`SurfacePlan::potentials_and_matrices`]: the potentials ([k][point]) and the matrices.
pub fn surface_potentials_and_matrices(
    plan: &SurfacePlan,
    ds: &[&[f64]],
    qs: &[&[f64]],
) -> (Vec<f64>, Vec<Vec<f64>>) {
    let n = plan.basis().nbf;
    let mut ws = plan.workspace(SurfaceShape {
        densities: ds.len(),
        charge_sets: qs.len(),
        gradient: false,
    });
    let mut v = vec![0.0; ds.len() * plan.points().len()];
    let mut m = vec![vec![0.0; n * n]; qs.len()];
    plan.potentials_and_matrices((ds, qs), &mut v, &mut views(&mut m, n), &mut ws);
    (v, m)
}

/// [`SurfacePlan::gradient`]: (the shell-centre part per atom, the point part per point).
pub fn surface_gradient(
    plan: &SurfacePlan,
    q: &[f64],
    d: &[f64],
) -> (Vec<[f64; 3]>, Vec<[f64; 3]>) {
    let mut ws = plan.workspace(SurfaceShape {
        gradient: true,
        ..Default::default()
    });
    let mut ga = vec![[0.0; 3]; plan.basis().natom()];
    let mut gc = vec![[0.0; 3]; plan.points().len()];
    plan.gradient(q, d, &mut ga, &mut gc, &mut ws);
    (ga, gc)
}

/// [`SurfaceDerivPlan::gradient`]: (the shell-centre part per atom, the point part per point).
pub fn surface_deriv_gradient(
    plan: &SurfaceDerivPlan,
    q: &[f64],
    d: &[f64],
) -> (Vec<[f64; 3]>, Vec<[f64; 3]>) {
    let (natom, np) = (
        plan.surface().basis().natom(),
        plan.surface().points().len(),
    );
    let mut ga = vec![[0.0; 3]; natom];
    let mut gc = vec![[0.0; 3]; np];
    plan.gradient(q, d, &mut ga, &mut gc, &mut plan.workspace());
    (ga, gc)
}

/// [`SurfaceDerivPlan::potentials_deriv`]: (3 natom) x npoints.
pub fn surface_potentials_deriv(plan: &SurfaceDerivPlan, d: &[f64]) -> Vec<f64> {
    let (natom, np) = (
        plan.surface().basis().natom(),
        plan.surface().points().len(),
    );
    let mut out = vec![0.0; 3 * natom * np];
    plan.potentials_deriv(d, &mut out, &mut plan.workspace());
    out
}

/// [`SurfaceDerivPlan::matrices_deriv`]: (3 natom) x nbf x nbf.
pub fn surface_matrices_deriv(plan: &SurfaceDerivPlan, q: &[f64]) -> Vec<f64> {
    let (natom, nbf) = (plan.surface().basis().natom(), plan.surface().basis().nbf);
    let mut out = vec![0.0; 3 * natom * nbf * nbf];
    plan.matrices_deriv(q, &mut out);
    out
}

/// [`SurfaceDerivPlan::hessian`]: (3 natom)^2, row-major.
pub fn surface_hessian(plan: &SurfaceDerivPlan, q: &[f64], d: &[f64]) -> Vec<f64> {
    let natom = plan.surface().basis().natom();
    let mut h = vec![0.0; 9 * natom * natom];
    plan.hessian(q, d, &mut h, &mut plan.workspace());
    h
}

/// [`AuxSurfacePlan::matrix`]: (P|g), npoints x naux row-major.
pub fn aux_point_matrix(aux: &BasisSet, pts: &[GaussPoint]) -> Vec<f64> {
    let mut m = vec![0.0; pts.len() * aux.nbf];
    AuxSurfacePlan::new(aux, pts).matrix(&mut m);
    m
}

/// [`AuxSurfacePlan::gradient`]: (the auxiliary atoms' part, the points' part).
pub fn aux_point_grad(
    aux: &BasisSet,
    pts: &[GaussPoint],
    q: &[f64],
    c: &[f64],
) -> (Vec<[f64; 3]>, Vec<[f64; 3]>) {
    let plan = AuxSurfacePlan::new(aux, pts);
    let mut ga = vec![[0.0; 3]; aux.natom()];
    let mut gc = vec![[0.0; 3]; pts.len()];
    plan.gradient(q, c, &mut ga, &mut gc, &mut plan.workspace());
    (ga, gc)
}
