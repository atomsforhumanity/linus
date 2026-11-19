//! Atomic masses (u) for vibrational analysis.
//!
//! - `Isotope` (default): the mass of the most abundant isotope (for elements without a stable
//!   isotope, Tc, Pm, Po, At, Rn: the longest-lived one), from NIST SRD 144 "Atomic Weights and
//!   Isotopic Compositions" (relative atomic masses of the Atomic Mass Evaluation; H-1
//!   1.00782503223, O-16 15.99491461957). These are exactly the masses of qcelemental and so of
//!   Psi4, and agree with Gaussian's and Q-Chem's defaults to the digits those print.
//! - `Average`: IUPAC 2013 standard atomic weights (conventional values for elements given as
//!   intervals: H 1.008, C 12.011, O 15.999, ...), as PySCF's `elements.MASSES`. ORCA 6 uses
//!   these for its frequencies and thermochemistry (its .hess files list H 1.008, O 15.999);
//!   PySCF's `thermo` and its default `harmonic_analysis` use them too.
//!
//! The choice moves frequencies by up to ~0.4% (X-H stretches, through 1.00783 vs 1.008) and
//! is the largest single difference between programs' harmonic analyses.

use serde::{Deserialize, Serialize};

/// Which mass table gives the default masses.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MassTable {
    /// Most abundant isotope (NIST SRD 144 / AME; qcelemental, Psi4).
    #[default]
    Isotope,
    /// IUPAC 2013 standard atomic weights, conventional values (ORCA, PySCF thermo).
    Average,
}

/// (mass number of the most abundant isotope, its mass, standard atomic weight) for Z = 1..86.
const TABLE: [(u32, f64, f64); 86] = [
    (1, 1.00782503223, 1.008),           // H
    (4, 4.00260325413, 4.002602),        // He
    (7, 7.0160034366, 6.94),             // Li
    (9, 9.012183065, 9.0121831),         // Be
    (11, 11.00930536, 10.81),            // B
    (12, 12.0, 12.011),                  // C
    (14, 14.00307400443, 14.007),        // N
    (16, 15.99491461957, 15.999),        // O
    (19, 18.99840316273, 18.998403163),  // F
    (20, 19.9924401762, 20.1797),        // Ne
    (23, 22.9897692820, 22.98976928),    // Na
    (24, 23.985041697, 24.305),          // Mg
    (27, 26.98153853, 26.9815385),       // Al
    (28, 27.97692653465, 28.085),        // Si
    (31, 30.97376199842, 30.973761998),  // P
    (32, 31.9720711744, 32.06),          // S
    (35, 34.968852682, 35.45),           // Cl
    (40, 39.9623831237, 39.948),         // Ar
    (39, 38.9637064864, 39.0983),        // K
    (40, 39.962590863, 40.078),          // Ca
    (45, 44.95590828, 44.955908),        // Sc
    (48, 47.94794198, 47.867),           // Ti
    (51, 50.94395704, 50.9415),          // V
    (52, 51.94050623, 51.9961),          // Cr
    (55, 54.93804391, 54.938044),        // Mn
    (56, 55.93493633, 55.845),           // Fe
    (59, 58.93319429, 58.933194),        // Co
    (58, 57.93534241, 58.6934),          // Ni
    (63, 62.92959772, 63.546),           // Cu
    (64, 63.92914201, 65.38),            // Zn
    (69, 68.9255735, 69.723),            // Ga
    (74, 73.921177761, 72.63),           // Ge
    (75, 74.92159457, 74.921595),        // As
    (80, 79.9165218, 78.971),            // Se
    (79, 78.9183376, 79.904),            // Br
    (84, 83.9114977282, 83.798),         // Kr
    (85, 84.9117897379, 85.4678),        // Rb
    (88, 87.9056125, 87.62),             // Sr
    (89, 88.9058403, 88.90584),          // Y
    (90, 89.9046977, 91.224),            // Zr
    (93, 92.9063730, 92.90637),          // Nb
    (98, 97.90540482, 95.95),            // Mo
    (98, 97.9072124, 97.90721),          // Tc
    (102, 101.9043441, 101.07),          // Ru
    (103, 102.9054980, 102.9055),        // Rh
    (106, 105.9034804, 106.42),          // Pd
    (107, 106.9050916, 107.8682),        // Ag
    (114, 113.90336509, 112.414),        // Cd
    (115, 114.903878776, 114.818),       // In
    (120, 119.90220163, 118.71),         // Sn
    (121, 120.9038120, 121.76),          // Sb
    (130, 129.906222748, 127.6),         // Te
    (127, 126.9044719, 126.90447),       // I
    (132, 131.9041550856, 131.293),      // Xe
    (133, 132.9054519610, 132.90545196), // Cs
    (138, 137.90524700, 137.327),        // Ba
    (139, 138.9063563, 138.90547),       // La
    (140, 139.9054431, 140.116),         // Ce
    (141, 140.9076576, 140.90766),       // Pr
    (142, 141.9077290, 144.242),         // Nd
    (145, 144.9127559, 144.91276),       // Pm
    (152, 151.9197397, 150.36),          // Sm
    (153, 152.9212380, 151.964),         // Eu
    (158, 157.9241123, 157.25),          // Gd
    (159, 158.9253547, 158.92535),       // Tb
    (164, 163.9291819, 162.5),           // Dy
    (165, 164.9303288, 164.93033),       // Ho
    (166, 165.9302995, 167.259),         // Er
    (169, 168.9342179, 168.93422),       // Tm
    (174, 173.9388664, 173.054),         // Yb
    (175, 174.9407752, 174.9668),        // Lu
    (180, 179.9465570, 178.49),          // Hf
    (181, 180.9479958, 180.94788),       // Ta
    (184, 183.95093092, 183.84),         // W
    (187, 186.9557501, 186.207),         // Re
    (192, 191.9614770, 190.23),          // Os
    (193, 192.9629216, 192.217),         // Ir
    (195, 194.9647917, 195.084),         // Pt
    (197, 196.96656879, 196.966569),     // Au
    (202, 201.97064340, 200.592),        // Hg
    (205, 204.9744278, 204.38),          // Tl
    (208, 207.9766525, 207.2),           // Pb
    (209, 208.9803991, 208.9804),        // Bi
    (209, 208.9824308, 208.98243),       // Po
    (210, 209.9871479, 209.98715),       // At
    (222, 222.0175782, 222.01758),       // Rn
];

/// Some isotopes besides the most abundant ones (NIST SRD 144), for per-atom overrides:
/// (Z, mass number, mass).
pub const ISOTOPES: [(u32, u32, f64); 11] = [
    (1, 2, 2.01410177812),   // D
    (1, 3, 3.0160492779),    // T
    (3, 6, 6.0151228874),    // Li-6
    (5, 10, 10.01293695),    // B-10
    (6, 13, 13.00335483507), // C-13
    (7, 15, 15.00010889888), // N-15
    (8, 17, 16.99913175650), // O-17
    (8, 18, 17.99915961286), // O-18
    (16, 34, 33.967867004),  // S-34
    (17, 37, 36.965902602),  // Cl-37
    (35, 81, 80.9162897),    // Br-81
];

/// Default mass (u) of element `z` from `table`; None outside 1..=86.
pub fn mass(z: u32, table: MassTable) -> Option<f64> {
    let (_, iso, avg) = *TABLE.get((z as usize).checked_sub(1)?)?;
    Some(match table {
        MassTable::Isotope => iso,
        MassTable::Average => avg,
    })
}

/// Mass number of the most abundant (or longest-lived) isotope of element `z`.
pub fn mass_number(z: u32) -> Option<u32> {
    TABLE.get((z as usize).checked_sub(1)?).map(|t| t.0)
}

/// Mass (u) of the isotope with mass number `a` of element `z`: the default isotope or one of
/// [`ISOTOPES`].
pub fn isotope_mass(z: u32, a: u32) -> Option<f64> {
    if mass_number(z) == Some(a) {
        return mass(z, MassTable::Isotope);
    }
    ISOTOPES.iter().find(|t| t.0 == z && t.1 == a).map(|t| t.2)
}

/// Per-atom masses: `table` for every atom, then `overrides` (atom index, mass in u).
pub fn masses(z: &[u32], table: MassTable, overrides: &[(usize, f64)]) -> Result<Vec<f64>, String> {
    let mut m: Vec<f64> = z
        .iter()
        .map(|&zi| mass(zi, table).ok_or_else(|| format!("no mass for atomic number {zi}")))
        .collect::<Result<_, _>>()?;
    for &(i, mi) in overrides {
        if i >= m.len() {
            return Err(format!(
                "mass override for atom {i}, but there are {} atoms",
                m.len()
            ));
        }
        if !(mi > 0.0 && mi.is_finite()) {
            return Err(format!("mass of atom {i} must be positive, not {mi}"));
        }
        m[i] = mi;
    }
    Ok(m)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tables() {
        assert_eq!(mass(1, MassTable::Isotope), Some(1.00782503223));
        assert_eq!(mass(8, MassTable::Average), Some(15.999));
        assert_eq!(mass(87, MassTable::Isotope), None);
        assert_eq!(mass(0, MassTable::Isotope), None);
        assert_eq!(isotope_mass(1, 2), Some(2.01410177812));
        assert_eq!(isotope_mass(6, 12), Some(12.0));
        // the table's mass numbers round its masses
        for (i, t) in TABLE.iter().enumerate() {
            assert!((t.1 - t.0 as f64).abs() < 0.5, "Z = {}", i + 1);
        }
        let m = masses(&[8, 1, 1], MassTable::Isotope, &[(1, 2.01410177812)]).unwrap();
        assert_eq!(m[1], 2.01410177812);
        assert!(masses(&[8], MassTable::Isotope, &[(1, 2.0)]).is_err());
    }
}
