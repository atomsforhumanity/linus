//! What a Hessian's analysis computes (decision D42), resolved from the `freq` settings
//! ([`FrequencyOptions`]): the harmonic analysis, one thermochemistry per temperature and, with
//! `ir`, the dipole derivatives. The facade runs it (`linus::vib`'s `FrequencyResult` and
//! `Session::hessian_analysis`).

use super::{ThermoOptions, VibOptions};
use linus_base::error::LinusError;

/// What a Hessian's analysis computes, resolved from the `freq` settings.
#[derive(Clone, Debug, PartialEq)]
pub struct FrequencyOptions {
    /// Dipole derivatives and IR intensities (from the Hessian's CPHF solutions) and the static
    /// polarizability (a three-right-hand-side field CPHF).
    pub ir: bool,
    /// The harmonic analysis's settings (None: no analysis).
    pub analysis: Option<VibOptions>,
    /// One thermochemistry per temperature, in order (empty: none).
    pub thermo: Vec<ThermoOptions>,
}

impl Default for FrequencyOptions {
    /// The defaults of the `freq` settings: the harmonic analysis and RRHO thermochemistry at
    /// 298.15 K, no IR intensities.
    fn default() -> Self {
        FrequencyOptions {
            ir: false,
            analysis: Some(VibOptions::default()),
            thermo: vec![ThermoOptions::default()],
        }
    }
}

impl FrequencyOptions {
    /// The options in range for the atoms `z` (masses, symmetry number, thermochemistry), before
    /// any Hessian is computed.
    pub fn check(&self, z: &[u32]) -> Result<(), LinusError> {
        if let Some(v) = &self.analysis {
            v.check()?;
            v.masses_for(z)?;
        }
        self.thermo.iter().try_for_each(ThermoOptions::check)
    }
}
