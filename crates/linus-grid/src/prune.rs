//! Angular pruning: number of Lebedev points on each radial shell.

use crate::elements::{bragg_radius, sg1_radius, RadiiTable};
use crate::lebedev::{lebedev_index, LEBEDEV_NPOINTS};
use crate::GridError;
use serde::{Deserialize, Serialize};

/// Radial-region boundaries (fractions of an atomic radius) used by NWChem/SG-1 style pruning,
/// per row group: H-He, Li-Ne, Na and heavier (PySCF's `alphas`).
pub const NWCHEM_ALPHAS: [[f64; 4]; 3] = [
    [0.25, 0.5, 1.0, 4.5],
    [0.1667, 0.5, 0.9, 3.5],
    [0.1, 0.4, 0.8, 2.5],
];

/// ORCA 6 angular grid schemes (manual, Table 7.1): Lebedev sets in the five radial regions for
/// `AngularGrid` 1..=7.
pub const ORCA_ANGULAR_REGIONS: [[usize; 5]; 7] = [
    [14, 26, 50, 50, 26],
    [14, 26, 50, 110, 50],
    [26, 50, 110, 194, 110],
    [26, 110, 194, 302, 194],
    [26, 194, 302, 434, 302],
    [50, 302, 434, 590, 434],
    [110, 434, 590, 770, 590],
];

/// Maximum Lebedev set of ORCA `AngularGrid` g (1..=7): 50, 110, 194, 302, 434, 590, 770.
pub fn orca_angular_max(g: usize) -> usize {
    ORCA_ANGULAR_REGIONS[g - 1][3]
}

/// Pruning schemes.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum Pruning {
    /// Same angular grid on every shell.
    None,
    /// NWChem-style pruning exactly as PySCF's `gen_grid.nwchem_prune` (Bragg radii).
    NWChem,
    /// SG-1 style pruning exactly as PySCF's `gen_grid.sg1_prune` (H-Ar only; ignores n_ang).
    SG1,
    /// Treutler-Ahlrichs index regions as PySCF's `gen_grid.treutler_prune`:
    /// first third of the shells 14 points, up to half 50 points, the rest n_ang.
    Treutler,
    /// ORCA-like five-region pruning: the region Lebedev sets of ORCA's AngularGrid scheme whose
    /// largest set equals n_ang (Table 7.1 of the ORCA 6 manual), with region boundaries at
    /// `NWCHEM_ALPHAS` x Bragg radius (ORCA's optimised per-element boundaries are not published).
    OrcaRegions,
    /// Fully custom five-region pruning: region k (0..5) uses `grids[k]` (capped at n_ang when
    /// `cap` is true); boundaries at `alphas[row group] x radius(radii)`.
    Regions {
        radii: RadiiTable,
        alphas: [[f64; 4]; 3],
        grids: [usize; 5],
        cap: bool,
    },
    /// ORCA 6.0.0's pruning for `angular_grid` 3, 4 or 6 (DefGrid1/2/3): the five Lebedev sets
    /// of that AngularGrid (one AngularGrid lower for H/He) with region boundaries from ORCA's
    /// tables ([`crate::orca::OrcaGrids::region_boundaries`]) x the element's ORCA xi. Meant for ORCA's radial grid (`RadialScale::OrcaXi` with the same AngularGrid).
    OrcaMeasured { angular_grid: u8 },
}

fn row_group(z: u32) -> usize {
    if z <= 2 {
        0
    } else if z <= 10 {
        1
    } else {
        2
    }
}

/// Region index 0..=4 = number of boundaries exceeded (PySCF: `(rads/r_atom > alphas).sum()`).
fn place(r: f64, r_atom: f64, alphas: &[f64; 4]) -> usize {
    let x = r / r_atom;
    alphas.iter().filter(|&&a| x > a).count()
}

/// Number of angular points per radial shell.
///
/// `r` are the radial nodes (in the scheme's native order), `n_ang` the maximum (nominal)
/// angular grid of the element.
pub fn angular_counts(
    pruning: &Pruning,
    z: u32,
    r: &[f64],
    n_ang: usize,
) -> Result<Vec<usize>, GridError> {
    match pruning {
        Pruning::None => Ok(vec![n_ang; r.len()]),
        Pruning::NWChem => {
            // leb_ngrid = LEBEDEV_NGRID[4:] = [38, 50, 74, 86, ...] = LEBEDEV_NPOINTS[3..]
            let leb = &LEBEDEV_NPOINTS[3..];
            if n_ang < 50 {
                return Ok(vec![n_ang; r.len()]);
            }
            let leb_l: [usize; 5] = if n_ang == 50 {
                [1, 2, 2, 2, 1]
            } else {
                let idx = leb
                    .iter()
                    .position(|&n| n == n_ang)
                    .ok_or(GridError::UnsupportedAngular(n_ang))?;
                [1, 3, idx - 1, idx, idx - 1]
            };
            let r_atom = bragg_radius(z) + 1e-200;
            let a = &NWCHEM_ALPHAS[row_group(z)];
            Ok(r.iter()
                .map(|&ri| leb[leb_l[place(ri, r_atom, a)]])
                .collect())
        }
        Pruning::SG1 => {
            let leb = [6usize, 38, 86, 194, 86];
            let r_atom = sg1_radius(z).ok_or(GridError::UnsupportedElement {
                z,
                what: "SG-1 pruning",
            })? + 1e-200;
            let a = &NWCHEM_ALPHAS[row_group(z)];
            Ok(r.iter().map(|&ri| leb[place(ri, r_atom, a)]).collect())
        }
        Pruning::Treutler => {
            let nr = r.len();
            let mut v = vec![n_ang; nr];
            for (i, x) in v.iter_mut().enumerate() {
                if i < nr / 3 {
                    *x = 14;
                } else if i < nr / 2 {
                    *x = 50;
                }
            }
            Ok(v)
        }
        Pruning::OrcaRegions => {
            let g = ORCA_ANGULAR_REGIONS
                .iter()
                .position(|row| row[3] == n_ang)
                .ok_or(GridError::UnsupportedAngular(n_ang))?;
            let grids = ORCA_ANGULAR_REGIONS[g];
            let r_atom = bragg_radius(z);
            let a = &NWCHEM_ALPHAS[row_group(z)];
            Ok(r.iter().map(|&ri| grids[place(ri, r_atom, a)]).collect())
        }
        Pruning::OrcaMeasured { angular_grid } => {
            let g = (*angular_grid).clamp(1, 7);
            let gl = (if z <= 2 && g > 1 { g - 1 } else { g }) as usize;
            let grids = ORCA_ANGULAR_REGIONS[gl - 1];
            let orca = crate::orca::grids();
            let t = orca.region_boundaries(z, g);
            let xi = orca
                .xi(z, g)
                .unwrap_or_else(|| crate::elements::treutler_xi(z));
            Ok(r.iter()
                .map(|&ri| grids[t.iter().filter(|&&tk| tk < ri / xi).count()])
                .collect())
        }
        Pruning::Regions {
            radii,
            alphas,
            grids,
            cap,
        } => {
            for &g in grids.iter() {
                lebedev_index(g).ok_or(GridError::UnsupportedAngular(g))?;
            }
            let r_atom = radii.radius(z);
            let a = &alphas[row_group(z)];
            Ok(r.iter()
                .map(|&ri| {
                    let g = grids[place(ri, r_atom, a)];
                    if *cap {
                        g.min(n_ang)
                    } else {
                        g
                    }
                })
                .collect())
        }
    }
}
