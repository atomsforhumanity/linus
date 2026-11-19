//! Units and physical constants: one table of CODATA adjustments ([`Constants`]), and the
//! Bohr radius in Å that reads Å input ([`BOHR_ANGSTROM`]).
//!
//! linus uses CODATA 2022 (h, k_B, N_A, c and e exact since the 2019 SI). CODATA 2014 is
//! available to reproduce programs that use it: Psi4 and PySCF (through qcelemental and
//! `pyscf.data.nist`; PySCF's Bohr radius is the 2010 value, 4.7e-10 relative off, which no
//! printed digit sees). The largest 2014 -> 2022 change that matters here is k_B (3.5e-7
//! relative), so thermochemistry differs between the two sets by ~3e-7 relative, e.g.
//! ~1e-8 Eh in G - E_el for a small molecule; frequencies by ~1e-8 relative.
//!
//! PySCF's 2010 Bohr radius ([`BOHR_ANGSTROM_PYSCF`]) is used only where a PySCF table in Å
//! must give PySCF's values in bohr bit for bit (grid radii, cavity radii, SMD).

use serde::{Deserialize, Serialize};

/// Bohr radius in Å (CODATA 2022; the `bohr` of [`Constants::CODATA2022`]): what Å input is
/// divided by.
pub const BOHR_ANGSTROM: f64 = 0.529_177_210_544;

/// PySCF's Bohr radius in Å (CODATA 2010, `pyscf.data.nist.BOHR`, `pyscf.lib.parameters.BOHR`),
/// for PySCF's tables in Å only (bit-identical radii in bohr).
pub const BOHR_ANGSTROM_PYSCF: f64 = 0.52917721092;

/// Which CODATA adjustment to take constants from.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Codata {
    /// CODATA 2022 (default).
    #[default]
    #[serde(alias = "2022")]
    Codata2022,
    /// CODATA 2014, as Psi4 1.11 (qcelemental) and PySCF 2.15 use.
    #[serde(alias = "2014")]
    Codata2014,
}

/// Constants in SI units unless stated.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Constants {
    /// Planck constant (J s)
    pub h: f64,
    /// Boltzmann constant (J/K)
    pub kb: f64,
    /// Avogadro constant (1/mol)
    pub na: f64,
    /// speed of light (m/s)
    pub c: f64,
    /// Hartree energy (J)
    pub hartree: f64,
    /// atomic mass constant, 1 u (kg)
    pub amu: f64,
    /// Bohr radius (m)
    pub bohr: f64,
    /// electron mass in u
    pub me_u: f64,
    /// fine-structure constant
    pub alpha: f64,
}

impl Constants {
    pub const CODATA2022: Constants = Constants {
        h: 6.626_070_15e-34,
        kb: 1.380_649e-23,
        na: 6.022_140_76e23,
        c: 299_792_458.0,
        hartree: 4.359_744_722_206_0e-18,
        amu: 1.660_539_068_92e-27,
        bohr: 0.529_177_210_544e-10,
        me_u: 5.485_799_090_441e-4,
        alpha: 7.297_352_564_3e-3,
    };
    pub const CODATA2014: Constants = Constants {
        h: 6.626_070_040e-34,
        kb: 1.380_648_52e-23,
        na: 6.022_140_857e23,
        c: 299_792_458.0,
        hartree: 4.359_744_650e-18,
        amu: 1.660_539_040e-27,
        bohr: 0.529_177_210_67e-10,
        me_u: 5.485_799_090_70e-4,
        alpha: 7.297_352_566_4e-3,
    };

    pub fn of(c: Codata) -> Constants {
        match c {
            Codata::Codata2022 => Self::CODATA2022,
            Codata::Codata2014 => Self::CODATA2014,
        }
    }

    /// cm^-1 per sqrt(Eh / (bohr^2 u)): wavenumber of a mass-weighted Hessian eigenvalue.
    pub fn wavenumber_per_sqrt_au(&self) -> f64 {
        (self.hartree / (self.amu * self.bohr * self.bohr)).sqrt()
            / (2.0 * std::f64::consts::PI * self.c * 100.0)
    }

    /// Boltzmann constant in Eh/K.
    pub fn kb_hartree(&self) -> f64 {
        self.kb / self.hartree
    }

    /// Second radiation constant h c / k_B in K cm (a wavenumber in cm^-1 times this is a
    /// temperature).
    pub fn kelvin_per_wavenumber(&self) -> f64 {
        self.h * self.c * 100.0 / self.kb
    }

    /// Eh/bohr^2 -> mdyn/Å (1 mdyn/Å = 100 N/m).
    pub fn mdyn_per_angstrom_per_au(&self) -> f64 {
        self.hartree / (self.bohr * self.bohr) / 100.0
    }

    /// km/mol per (e^2/u): IR intensity N_A π / (3 c^2 4π ε0) |dμ/dQ|^2 with dμ/dQ in atomic
    /// units of dipole per bohr per sqrt(u), = N_A π α^2 a0 (m_e/u) / 3 (as Psi4's vib.py).
    pub fn ir_km_per_mol_per_au(&self) -> f64 {
        self.na * std::f64::consts::PI * self.alpha * self.alpha * self.bohr * self.me_u / 3.0
            * 1e-3
    }

    /// Rotational constant in cm^-1 for a moment of inertia in u bohr^2.
    pub fn rotational_constant_wavenumber(&self, moment: f64) -> f64 {
        self.h
            / (8.0
                * std::f64::consts::PI.powi(2)
                * self.c
                * 100.0
                * moment
                * self.amu
                * self.bohr
                * self.bohr)
    }
}

/// kcal/mol per Eh with CODATA 2022 (thermochemical calorie, 4.184 J).
pub const HARTREE_KCAL_MOL: f64 = 627.509_474_063_1;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn one_bohr_per_edition() {
        let c = Constants::CODATA2022;
        assert!((c.bohr * 1e10 / BOHR_ANGSTROM - 1.0).abs() < 1e-15);
        // the constant carries 13 significant digits
        let k = c.hartree * c.na / 4184.0;
        assert!((k / HARTREE_KCAL_MOL - 1.0).abs() < 1e-12);
        assert!((BOHR_ANGSTROM_PYSCF / BOHR_ANGSTROM - 1.0).abs() < 1e-9);
    }
}
