//! Grids for VV10 nonlocal correlation ("NLC grids"), as the reference programs choose them.
//!
//! The VV10 double sum costs O(N^2) in the grid points, so every program evaluates it on a
//! coarser grid than the semilocal XC:
//!
//! | program | NLC grid | linus |
//! |---|---|---|
//! | PySCF 2.15 | `mf.nlcgrids`, level 3 whatever the XC level (`Grids`' default; older PySCF: 1) | `GridSpec::pyscf(3)` (exact) |
//! | ORCA 6.0.0 DefGrid3 | the DefGrid1 XC grid | [`GridSpec::orca_nl`]`(3)` = `GridSpec::orca_defgrid(1)` |
//! | ORCA 6.0.0 DefGrid1/2 | AngularGrid 1/2 | [`GridSpec::orca_nl`]`(1/2)` (approximate) |
//! | Psi4 1.11 | `DFT_VV10_RADIAL_POINTS` 50 x `DFT_VV10_SPHERICAL_POINTS` 146 | [`GridSpec::psi4_vv10`] |

use crate::grid::{AtomSizes, GridSpec};
use crate::prune::Pruning;
use crate::radial::RadialScheme;

impl GridSpec {
    /// ORCA 6.0.0's VV10 ("NL") grid for `DefGrid n` (1..=3), from ORCA's tables
    /// ([`crate::orca`]).
    ///
    /// * DefGrid3: ORCA's DefGrid1 XC grid, i.e. [`GridSpec::orca_defgrid`]`(1)`. OMol25's level
    ///   (wB97M-V/def2-TZVPD DEFGRID3).
    /// * DefGrid1/2: AngularGrid 1/2, approximated: the DefGrid1 radial grid (xi and region
    ///   boundaries of AngularGrid 3) with the Lebedev regions of AngularGrid 1/2 and the shell
    ///   counts of ORCA's tables ([`AtomSizes::OrcaNl`]). Not ORCA's points.
    pub fn orca_nl(defgrid: u8) -> GridSpec {
        let n = defgrid.clamp(1, 3);
        if n == 3 {
            return GridSpec::orca_defgrid(1);
        }
        let mut s = GridSpec::orca_defgrid(1);
        // AngularGrid 1 (DefGrid1) or 2 (DefGrid2)
        s.pruning = Pruning::OrcaMeasured { angular_grid: n };
        s.sizes = AtomSizes::OrcaNl { defgrid: n };
        s
    }

    /// Psi4 1.11's VV10 grid: [`GridSpec::psi4_default`] with `DFT_VV10_RADIAL_POINTS` (50) radial
    /// and `DFT_VV10_SPHERICAL_POINTS` (146) Lebedev points, unpruned. As for the XC grid, Psi4's
    /// standard orientation and its removal of points beyond the basis extents are not applied.
    pub fn psi4_vv10() -> GridSpec {
        let mut s = GridSpec::psi4_default();
        s.sizes = AtomSizes::Uniform {
            n_rad: 50,
            n_ang: 146,
        };
        debug_assert!(matches!(s.radial, RadialScheme::Psi4Treutler { .. }));
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::MolecularGrid;

    /// Psi4's VV10 grid builds (the ORCA NL grids are tested with their data,
    /// tests/orca_defgrid.rs); orca_nl(3) is the DefGrid1 XC grid.
    #[test]
    fn nl_grids() {
        let z = [8, 1, 1];
        let xyz = [
            [0.0, 0.0, 0.2217],
            [0.0, 1.4309, -0.8867],
            [0.0, -1.4309, -0.8867],
        ];
        let n = |s: &GridSpec| MolecularGrid::build(&z, &xyz, s).unwrap().len();
        assert_eq!(GridSpec::orca_nl(3), GridSpec::orca_defgrid(1));
        assert!(n(&GridSpec::psi4_vv10()) > 0);
    }
}
