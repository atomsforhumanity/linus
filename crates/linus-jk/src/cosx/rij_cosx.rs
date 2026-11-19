//! Two-electron gradient of jk = rij-snk (RIJCOSX): RI-J (`crate::df::rij_gradient`) plus the COSX
//! exchange gradient ([`crate::cosx::gradient`]) on the grid of the final exchange energy.
//!
//! With the ORCA-like COSX grids (`k_grid` = cosx:N) the SCF runs on GridX1/GridX2 and the
//! energy's exchange is re-evaluated on GridX3 (`linus::two_electron::finalize`); the gradient
//! is the derivative of that final energy at the SCF density, as in ORCA (whose COSX gradient
//! also uses GridX3; changing ORCA's GridX3 changes its gradient):
//!
//! * overlap fitting differentiated consistently (the S and S_num derivatives of
//!   M = S S_num^-1). Treating M as fixed costs 1e-3 Eh/bohr (la18 B3LYP/def2-SVP, max error
//!   against exact integrals: 1.1e-3 instead of 3.9e-5);
//! * with `grid_response` (default) the grid moves with the atoms and the partition weights
//!   are differentiated: the exact derivative of the GridX3 energy at fixed density,
//!   translation invariant. The point-motion and weight terms are each ~1e-3 Eh/bohr and cancel
//!   to ~5e-5, so they are switched together (`grid_response = false` drops both, a fixed grid;
//!   point motion alone would leave a 1e-3 error). Weight derivatives cost ~15% of the COSX
//!   gradient;
//! * on GridX2 (the SCF grid) instead of GridX3 the error against exact integrals grows 2-5x.
//!
//! Neglected (as in ORCA): the SCF density is stationary for neither the GridX3 energy nor any
//! overlap-fitted energy (B(D1, D2) = D1 . M K'\[D2\] is not symmetric), so the gradient differs
//! from finite differences of the energy by the orbital response. Measured (max |FD -
//! analytic|, Eh/bohr): one fitted grid (gridx:3) 7e-7 (water), 2.3e-6 (la18 def2-SVP); the
//! production scheme cosx:2 7e-6 (water), 1.6e-5 (la18 def2-SVP RKS), 1.5e-5 (la18+ UKS).
//! Everything else is exact: with any non-COSX `k_grid` (one grid, no fitting) the energy is
//! variational and the gradient matches finite differences to 5e-10 (la18 B3LYP/def2-SVP RKS),
//! 2e-9 (la18+ UKS), 4e-10 (water).
//!
//! Accuracy against exact integrals (B3LYP, DefGrid2, max |dG| Eh/bohr; RI-J-only / COSX-only
//! parts from RI-J with exact K): la18 SVP 3.9e-5, la18 TZVP 5.5e-5, u1an39 SVP 1.1e-4,
//! u1an39 TZVP 9.7e-5, ts61 SVP 2.4e-4, ts61 TZVP 1.1e-4. Fitting the derivative functions on
//! the grid (Izsák & Neese 2011, eq. 21) instead of differentiating the fitted energy is
//! implemented as [`CosxGradientConfig::deriv_fit`] (full fitting matrices); it is less accurate
//! than the consistent derivative (def2-SVP: la18 9.9e-5, u1an39 4.3e-4, ts61 1.5e-3 vs
//! 3.9e-5, 1.1e-4, 2.4e-4).

use crate::cosx::gradient::{exchange_gradient, CosxGradientConfig};
use crate::df::rij_gradient::rij_gradient;
use faer::{Mat, Par};
use linus_base::error::LinusError;
use linus_core::timing::Timers;
use linus_grid::{GridSpec, MolecularGrid};
use linus_ints::eri3c::DfThresholds;
use linus_ints::operator::Operator;
use linus_ints::{BasisSet, PairSet};

/// RI-J + COSX two-electron gradient at the spin densities `da`, `db` (`restricted`: a closed
/// shell) of molecule `z`, `positions` with auxiliary basis `aux` on the COSX gradient's `grid`
/// (its spec and whether it fits the overlap; the facade's `gradient_grid` of the `k_grid`
/// setting): the exact exchange `xop` (a_x K for HF and global hybrids, K of a/r + b erf(omega r)/r
/// for range-separated ones; [`linus_base::terms::exchange_parts`]); `grid_response`: the COSX grid
/// moves with the atoms.
#[allow(clippy::too_many_arguments)]
pub fn rijcosx_gradient(
    basis: &BasisSet,
    z: &[u32],
    positions: &[[f64; 3]],
    pairs: &PairSet,
    aux: &BasisSet,
    grid: (GridSpec, bool),
    restricted: bool,
    da: &Mat<f64>,
    db: &Mat<f64>,
    xop: Option<Operator>,
    grid_response: bool,
    par: Par,
    timers: &mut Timers,
) -> Result<Vec<[f64; 3]>, LinusError> {
    let (a_x, op) = linus_base::terms::exchange_parts(xop);
    let op = op.unwrap_or(Operator::Coulomb);
    let dt = da + db;
    let t = std::time::Instant::now();
    let mut g = rij_gradient(
        basis,
        pairs,
        aux,
        &dt,
        &DfThresholds::default(),
        par,
        timers,
    );
    timers.add("grad_rij", t.elapsed().as_secs_f64());
    if a_x != 0.0 {
        let t = std::time::Instant::now();
        let (spec, fitting) = grid;
        let mg = timers.time("grad_cosx_grid", || {
            MolecularGrid::build(z, positions, &spec)
        })?;
        let (dens, kappa) = if restricted {
            (vec![da.clone()], vec![-a_x])
        } else {
            (vec![da.clone(), db.clone()], vec![-0.5 * a_x; 2])
        };
        // grid response: weight derivatives and point motion together (they cancel to a large
        // extent; point motion alone is far worse than neither)
        let gr = grid_response;
        let opts = CosxGradientConfig {
            fitting,
            weight_derivs: gr,
            point_motion: gr,
            xop: op,
            ..Default::default()
        };
        let k = exchange_gradient(basis, &mg, &dens, &kappa, &opts, par)?;
        timers.merge(&k.timers);
        for (x, y) in g.iter_mut().zip(&k.gradient) {
            for c in 0..3 {
                x[c] += y[c];
            }
        }
        timers.add("grad_cosx", t.elapsed().as_secs_f64());
    }
    Ok(g)
}
