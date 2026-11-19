// SPDX-License-Identifier: LGPL-3.0-or-later
//! Constants that are part of the D3/D4/EEQ parametrizations, evaluated exactly as
//! mctc-lib 0.7.0 evaluates them (`io/convert.f90`, `data/covrad.f90`).

use crate::data::common as c;

/// Angstrom -> bohr as mctc-lib computes it from CODATA 2018 (`aatoau`). The reference
/// programs convert their tabulated radii with this factor, so it is part of the
/// parametrization; it is never used to convert input geometries (linus works in bohr).
pub const AATOAU: f64 = {
    let hbar = c::PLANCK / (2.0 * c::PI);
    let bohr = hbar / (c::ELECTRON_MASS * c::SPEED_OF_LIGHT * c::FINE_STRUCTURE);
    let autoaa = bohr * 1e10;
    1.0 / autoaa
};

/// D3 covalent radius in bohr (`covalent_rad_d3 = 4/3 * aatoau * covalent_rad_2009`),
/// used by the D3 and the EEQ coordination numbers. 0 for unknown elements.
#[inline]
pub fn covalent_radius_d3(z: u32) -> f64 {
    let z = z as usize;
    if z == 0 || z > c::COVALENT_RAD_2009.len() {
        return 0.0;
    }
    4.0 / 3.0 * (AATOAU * c::COVALENT_RAD_2009[z - 1])
}

#[cfg(test)]
mod tests {
    #[test]
    fn aatoau_matches_axtb() {
        // axtb-d4 evaluates the same expression from dftd4's copy of the constants
        assert_eq!(super::AATOAU.to_bits(), axtb_d4::AATOAU.to_bits());
        for z in 1..=118u32 {
            assert_eq!(
                super::covalent_radius_d3(z).to_bits(),
                axtb_d4::covalent_radius(z as usize).to_bits(),
                "Z = {z}"
            );
        }
    }
}
