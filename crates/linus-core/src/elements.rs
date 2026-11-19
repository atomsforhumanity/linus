//! The elements linus supports: H-Rn, by atomic number and symbol.

/// Element symbols by atomic number (index 0: dummy "X"); linus supports H-Rn (1-86).
pub const SYMBOLS: [&str; 87] = [
    "X", "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si", "P", "S",
    "Cl", "Ar", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge",
    "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
    "In", "Sn", "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd",
    "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi", "Po", "At", "Rn",
];

/// Highest atomic number linus supports (Rn).
pub const MAX_Z: u32 = 86;

/// Atomic number of an element symbol (any case), or of a number given as text ("6").
pub fn symbol_to_z(s: &str) -> Option<u32> {
    let t = s.trim();
    if let Ok(z) = t.parse::<u32>() {
        return (1..=MAX_Z).contains(&z).then_some(z);
    }
    SYMBOLS
        .iter()
        .position(|x| x.eq_ignore_ascii_case(t))
        .filter(|&i| i > 0)
        .map(|i| i as u32)
}

/// Element symbol of atomic number `z` (1-86).
pub fn z_to_symbol(z: u32) -> Option<&'static str> {
    (1..=MAX_Z).contains(&z).then(|| SYMBOLS[z as usize])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn symbols_and_numbers() {
        assert_eq!(symbol_to_z("cl"), Some(17));
        assert_eq!(symbol_to_z(" 6 "), Some(6));
        assert_eq!(symbol_to_z("X"), None);
        assert_eq!(symbol_to_z("87"), None);
        assert_eq!(z_to_symbol(86), Some("Rn"));
        assert_eq!(z_to_symbol(0), None);
    }
}
