//! Atomic radii of the solute cavity.
//!
//! [`BONDI_PYSCF`]: the van der Waals radii of PySCF's `pyscf.data.radii.VDW` (Bondi 1964 via
//! Macmillan's Chemical and Physical Data, and Mantina, Chamberlin, Valero, Cramer & Truhlar,
//! J. Phys. Chem. A 113, 5806 (2009) for the elements Bondi lacks; 1.999999 Å where PySCF has
//! no value), which PySCF's PCM uses with H = 1.1 Å ("modified Bondi") and a scale of 1.2.
//! Ported from PySCF 2.15 (Apache-2.0; see LICENSE). The table is in Å; PySCF converts with
//! its Bohr radius (linus-core's `units::BOHR_ANGSTROM_PYSCF`).

const UNKNOWN: f64 = 1.999999;

/// van der Waals radii (Å) of PySCF's `radii.VDW`, index = atomic number (0: ghost), up to 103.
pub static BONDI_PYSCF: [f64; 104] = [
    UNKNOWN, // ghost
    1.20, 1.40, // H He
    1.82, 1.53, 1.92, 1.70, 1.55, 1.52, 1.47, 1.54, // Li-Ne
    2.27, 1.73, 1.84, 2.10, 1.80, 1.80, 1.75, 1.88, // Na-Ar
    2.75, 2.31, // K Ca
    UNKNOWN, UNKNOWN, UNKNOWN, UNKNOWN, UNKNOWN, UNKNOWN, UNKNOWN, // Sc-Co
    1.63, 1.40, 1.39, // Ni Cu Zn
    1.87, 2.11, 1.85, 1.90, 1.85, 2.02, // Ga-Kr
    3.03, 2.49, // Rb Sr
    UNKNOWN, UNKNOWN, UNKNOWN, UNKNOWN, UNKNOWN, UNKNOWN, UNKNOWN, // Y-Rh
    1.63, 1.72, 1.58, // Pd Ag Cd
    1.93, 2.17, 2.06, 2.06, 1.98, 2.16, // In-Xe
    3.43, 2.49, // Cs Ba
    UNKNOWN, UNKNOWN, UNKNOWN, UNKNOWN, UNKNOWN, UNKNOWN, UNKNOWN, UNKNOWN, // La-Gd
    UNKNOWN, UNKNOWN, UNKNOWN, UNKNOWN, UNKNOWN, UNKNOWN, UNKNOWN, // Tb-Lu
    UNKNOWN, UNKNOWN, UNKNOWN, UNKNOWN, UNKNOWN, UNKNOWN, // Hf-Ir
    1.75, 1.66, 1.55, // Pt Au Hg
    1.96, 2.02, 2.07, 1.97, 2.02, 2.20, // Tl-Rn
    3.48, 2.83, // Fr Ra
    UNKNOWN, UNKNOWN, UNKNOWN, // Ac Th Pa
    1.86,    // U
    UNKNOWN, UNKNOWN, UNKNOWN, UNKNOWN, UNKNOWN, UNKNOWN, UNKNOWN, UNKNOWN, UNKNOWN, UNKNOWN,
    UNKNOWN, // Np-Lr
];

/// PySCF's "modified Bondi" radius (Å): [`BONDI_PYSCF`] with H = 1.1 Å.
pub fn modified_bondi(z: u32) -> Option<f64> {
    match z {
        1 => Some(1.1),
        z if (z as usize) < BONDI_PYSCF.len() => Some(BONDI_PYSCF[z as usize]),
        _ => None,
    }
}

/// True where PySCF has no radius of its own (1.999999 Å placeholder).
pub fn is_placeholder(r: f64) -> bool {
    r == UNKNOWN
}

#[cfg(test)]
mod tests {
    #[test]
    fn table_spot_checks() {
        use super::*;
        assert_eq!(BONDI_PYSCF[6], 1.70);
        assert_eq!(BONDI_PYSCF[16], 1.80);
        assert_eq!(BONDI_PYSCF[35], 1.85);
        assert_eq!(BONDI_PYSCF[54], 2.16);
        assert_eq!(BONDI_PYSCF[78], 1.75);
        assert_eq!(BONDI_PYSCF[92], 1.86);
        assert_eq!(modified_bondi(1), Some(1.1));
        assert_eq!(modified_bondi(8), Some(1.52));
    }
}
