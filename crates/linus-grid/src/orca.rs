//! ORCA 6.0.0's DefGrid tables for the `orca:N` presets: `data/orca/grids.json` of the source
//! tree, compiled in and parsed once per process on first use.
//!
//! The file holds what ORCA does not publish about its DefGrid1/2/3 XC grids and NL grids,
//! determined from ORCA's printed output (`tools/orca_probe/`; included with permission of
//! FAccTs GmbH): the per-element xi of its radial grid for AngularGrid 3, 4 and 6, its radial
//! shell counts (by element and by row), its pruning-region boundaries (in units of xi, by row
//! and AngularGrid), and the shell counts that approximate its DefGrid1/2 NL grids. Every grid
//! that uses [`crate::RadialScale::OrcaXi`], [`crate::AtomSizes::OrcaDefGrid`],
//! [`crate::AtomSizes::OrcaNl`] or [`crate::Pruning::OrcaMeasured`] reads it.

use crate::elements::period_index;
use serde::Deserialize;
use std::sync::OnceLock;

/// The tables as JSON: the generator's literals, which serde_json's `float_roundtrip` parses
/// to the same bits.
const GRIDS_JSON: &str = include_str!("../../../data/orca/grids.json");

/// ORCA's measured DefGrid tables.
#[derive(Clone, Debug, PartialEq)]
pub struct OrcaGrids {
    /// xi (bohr) by Z for AngularGrid 3, 4, 6; 0 = not measured.
    xi: Vec<[f64; 3]>,
    /// Radial shells by Z for DefGrid1/2/3; 0 = not measured.
    nr: Vec<[u16; 3]>,
    /// Radial shells by row (H-He ... Fr-) for DefGrid1/2/3.
    row_nr: [[usize; 3]; 7],
    /// Pruning-region boundaries (units of xi) by row (H-He, Li-Ne, Na-Ar, K-Kr) and
    /// AngularGrid (3, 4, 6).
    region_t: [[[f64; 4]; 3]; 4],
    /// NL grids of DefGrid1/2: (H shells, extra shells over DefGrid1's for heavier elements).
    nl: [(usize, usize); 2],
}

#[derive(Deserialize)]
struct ByZ<T> {
    by_z: Vec<[T; 3]>,
}

#[derive(Deserialize)]
struct ByRow<T> {
    by_row: T,
}

#[derive(Deserialize)]
struct Nl {
    h_shells: usize,
    extra_shells: usize,
}

#[derive(Deserialize)]
struct NlGrids {
    defgrid1: Nl,
    defgrid2: Nl,
}

#[derive(Deserialize)]
struct Raw {
    radial_xi: ByZ<f64>,
    radial_shells: ByZ<u16>,
    radial_shells_by_row: ByRow<[[usize; 3]; 7]>,
    region_boundaries: ByRow<[[[f64; 4]; 3]; 4]>,
    nl_grids: NlGrids,
}

/// Column of an AngularGrid in the xi and region tables: <= 3 -> 3, 4, >= 5 -> 6.
fn column(angular_grid: u8) -> usize {
    match angular_grid {
        0..=3 => 0,
        4 => 1,
        _ => 2,
    }
}

impl OrcaGrids {
    fn from_raw(r: Raw) -> OrcaGrids {
        OrcaGrids {
            xi: r.radial_xi.by_z,
            nr: r.radial_shells.by_z,
            row_nr: r.radial_shells_by_row.by_row,
            region_t: r.region_boundaries.by_row,
            nl: [
                (
                    r.nl_grids.defgrid1.h_shells,
                    r.nl_grids.defgrid1.extra_shells,
                ),
                (
                    r.nl_grids.defgrid2.h_shells,
                    r.nl_grids.defgrid2.extra_shells,
                ),
            ],
        }
    }

    /// ORCA's xi for element `z` and AngularGrid `angular_grid`, if measured.
    pub fn xi(&self, z: u32, angular_grid: u8) -> Option<f64> {
        let v = *self.xi.get(z as usize)?.get(column(angular_grid))?;
        (v > 0.0).then_some(v)
    }

    /// Radial shells of DefGrid `level` (1..=3) for element `z`, if measured.
    pub fn defgrid_nr(&self, z: u32, level: u8) -> Option<usize> {
        let v = *self
            .nr
            .get(z as usize)?
            .get(level.clamp(1, 3) as usize - 1)?;
        (v > 0).then_some(v as usize)
    }

    /// Radial shells of DefGrid `level` for `z`: measured, else its row's.
    pub fn defgrid_nr_or_row(&self, z: u32, level: u8) -> usize {
        self.defgrid_nr(z, level)
            .unwrap_or(self.row_nr[period_index(z).min(6)][level.clamp(1, 3) as usize - 1])
    }

    /// Pruning-region boundaries (units of xi) of element `z` for AngularGrid `angular_grid`
    /// (rows beyond K-Kr use the K-Kr values).
    pub fn region_boundaries(&self, z: u32, angular_grid: u8) -> &[f64; 4] {
        &self.region_t[period_index(z).min(3)][column(angular_grid)]
    }

    /// Radial shells of the approximate NL grid of DefGrid `defgrid` (1 or 2) for element `z`.
    pub fn nl_shells(&self, z: u32, defgrid: u8) -> usize {
        let (nr_h, extra) = self.nl[(defgrid.clamp(1, 2) - 1) as usize];
        if z <= 2 {
            nr_h
        } else {
            self.defgrid_nr_or_row(z, 1) + extra
        }
    }
}

/// The tables, parsed on first use.
pub fn grids() -> &'static OrcaGrids {
    static TABLES: OnceLock<OrcaGrids> = OnceLock::new();
    TABLES.get_or_init(|| {
        let raw: Raw = serde_json::from_str(GRIDS_JSON)
            .unwrap_or_else(|e| panic!("data/orca/grids.json is malformed: {e}"));
        OrcaGrids::from_raw(raw)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The compiled-in file parses (tests/orca_defgrid.rs checks the values against ORCA's).
    #[test]
    fn embedded_tables_parse() {
        let t = grids();
        assert_eq!((t.xi.len(), t.nr.len()), (37, 37));
    }
}
