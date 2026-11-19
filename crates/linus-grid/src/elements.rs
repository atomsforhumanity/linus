//! Per-element data: radii tables, radial scaling parameters, periodic-table rows.

use crate::element_data::*;
use linus_core::units::BOHR_ANGSTROM_PYSCF;
use serde::{Deserialize, Serialize};

/// Atomic radii tables that can be used for Becke/Treutler size adjustment and pruning.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum RadiiTable {
    /// Bragg-Slater radii as tabulated by PySCF (`pyscf.data.radii.BRAGG`).
    BraggPyscf,
    /// Covalent radii as tabulated by PySCF (`pyscf.data.radii.COVALENT`).
    CovalentPyscf,
    /// Bragg-Slater radii (bohr) used by Psi4 (`GetBSRadius` in libfock/cubature.cc).
    BraggSlaterPsi4,
}

impl RadiiTable {
    /// Radius in bohr for nuclear charge `z`.
    pub fn radius(self, z: u32) -> f64 {
        match self {
            RadiiTable::BraggPyscf => bragg_radius(z),
            RadiiTable::CovalentPyscf => {
                let z = z as usize;
                let v = if z < COVALENT_ANGSTROM.len() {
                    COVALENT_ANGSTROM[z]
                } else {
                    COVALENT_ANGSTROM[0]
                };
                (1.0 / BOHR_ANGSTROM_PYSCF) * v
            }
            RadiiTable::BraggSlaterPsi4 => psi4_bs_radius(z),
        }
    }
}

/// Bragg-Slater radius in bohr as used by PySCF (`radi.BRAGG_RADII[z]`), bit-identical.
pub fn bragg_radius(z: u32) -> f64 {
    let z = z as usize;
    let v = if z < BRAGG_ANGSTROM.len() {
        BRAGG_ANGSTROM[z]
    } else {
        BRAGG_ANGSTROM[0]
    };
    (1.0 / BOHR_ANGSTROM_PYSCF) * v
}

/// Psi4's Bragg-Slater radius in bohr (3.30 beyond the table, as in Psi4).
pub fn psi4_bs_radius(z: u32) -> f64 {
    let z = z as usize;
    if z < PSI4_BS_RADII_BOHR.len() {
        PSI4_BS_RADII_BOHR[z]
    } else {
        3.30
    }
}

/// Treutler-Ahlrichs radial scaling parameter xi (1.0 beyond the table).
pub fn treutler_xi(z: u32) -> f64 {
    let z = z as usize;
    if z < TREUTLER_XI.len() {
        TREUTLER_XI[z]
    } else {
        1.5
    }
}

/// SG-1 radius in bohr (only defined for H-Ar).
pub fn sg1_radius(z: u32) -> Option<f64> {
    SG1_RADII_BOHR.get(z as usize).copied()
}

/// Zero-based period index as used by PySCF's grid tables:
/// 0 for H-He, 1 for Li-Ne, 2 for Na-Ar, 3 for K-Kr, 4 for Rb-Xe, 5 for Cs-Rn, 6 beyond.
pub fn period_index(z: u32) -> usize {
    [2u32, 10, 18, 36, 54, 86, 118]
        .iter()
        .filter(|&&t| z > t)
        .count()
}

/// Periodic-table row (1-based).
pub fn row(z: u32) -> usize {
    period_index(z) + 1
}
