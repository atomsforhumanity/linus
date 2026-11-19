//! Vibrations: linus-vib's harmonic analysis and thermochemistry (pure functions of any
//! program's Hessian, re-exported here), and the analysis a Hessian carries (decision D42): the
//! analytic Hessian, optionally its dipole derivatives (IR intensities) and the polarizability,
//! the harmonic analysis and one thermochemistry per temperature, in one call
//! ([`Session::hessian_analysis`], [`FrequencyResult`]). The settings are the `freq` group's
//! (`crate::driver::CalcOptions::freq_options`).

pub use linus_vib::*;

use crate::derivatives::hessian::{HessianOptions, HessianResult};
use crate::driver::Session;
use crate::properties::dipole::DipoleDerivatives;
use crate::Structure;
use linus_base::error::LinusError;

/// Everything a Hessian with its analysis produces.
#[derive(Clone, Debug)]
pub struct FrequencyResult {
    /// Energy, gradient and Hessian.
    pub hessian: HessianResult,
    pub dipole_derivatives: Option<DipoleDerivatives>,
    /// The harmonic analysis (None: `freq.analysis: off`).
    pub vib: Option<VibrationalAnalysis>,
    /// One thermochemistry per temperature (empty: `freq.thermo: off`).
    pub thermo: Vec<Thermochemistry>,
}

impl FrequencyResult {
    /// Largest absolute gradient component (Eh/bohr): how far from a stationary point.
    pub fn max_gradient(&self) -> f64 {
        self.hessian
            .gradient
            .iter()
            .flatten()
            .fold(0.0, |a: f64, g| linus_base::linalg::max_nan(a, g.abs()))
    }
}

impl Session {
    /// Analytic Hessian at the current geometry (with `opts.ir` also its dipole derivatives and
    /// the polarizability), then the harmonic analysis and thermochemistry at the SCF energy, as
    /// `opts` asks.
    pub fn hessian_analysis(
        &mut self,
        opts: &FrequencyOptions,
    ) -> Result<FrequencyResult, LinusError> {
        // check the options before the expensive part
        opts.check(&self.cx.mol.z)?;
        let hessian = self.hessian_with(&HessianOptions {
            dipole_derivatives: opts.ir,
        })?;
        let dipole_derivatives = if opts.ir {
            let t = std::time::Instant::now();
            let progress = self.cx.progress.clone();
            let d = progress
                .stage("polarizability", || self.ir_properties(&hessian))
                // stopped on request: with the gradient and Hessian it reached
                .map_err(|e| e.reached(Some(&hessian.gradient), Some(&hessian.hessian)))?;
            self.timers.add("polarizability", t.elapsed().as_secs_f64());
            Some(d)
        } else {
            None
        };
        FrequencyResult::from_hessian(&self.cx.mol, hessian, dipole_derivatives, opts)
    }
}

impl FrequencyResult {
    /// Harmonic analysis and thermochemistry of any `HessianResult` of `mol` (closed or open
    /// shell: only the Hessian, the energy and the multiplicity enter), with optional dipole
    /// derivatives for IR intensities. `opts.ir` is not looked at.
    pub fn from_hessian(
        mol: &Structure,
        hessian: HessianResult,
        dipole_derivatives: Option<DipoleDerivatives>,
        opts: &FrequencyOptions,
    ) -> Result<Self, LinusError> {
        let vib = opts
            .analysis
            .as_ref()
            .map(|v| {
                harmonic_analysis(
                    &mol.z,
                    &mol.positions,
                    &hessian.hessian,
                    v,
                    dipole_derivatives.as_ref().map(|d| &d.apt[..]),
                )
            })
            .transpose()?;
        let thermo = match &vib {
            Some(v) => opts
                .thermo
                .iter()
                .map(|t| thermochemistry(v, hessian.energy, mol.multiplicity, t))
                .collect::<Result<_, _>>()?,
            None => Vec::new(),
        };
        Ok(FrequencyResult {
            hessian,
            dipole_derivatives,
            vib,
            thermo,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::driver::CalcOptions;

    #[test]
    fn water_frequencies() {
        let xyz = "3\n\nO 0.0 0.0 0.1178\nH 0.0 0.7555 -0.4712\nH 0.0 -0.7555 -0.4712\n";
        let mol = Structure::from_xyz_str(xyz, 0, 1).unwrap();
        let o = CalcOptions::from_json(serde_json::json!({
            "basis": "def2-svp", "threads": 2,
            "freq": {"ir": true, "temperature": [300.0, 400.0], "thermo": "qrrho"}}))
        .unwrap();
        let opts = o.freq_options();
        let mut calc = Session::new(mol, o).unwrap();
        let f = calc.hessian_analysis(&opts).unwrap();
        let v = f.vib.as_ref().unwrap();
        assert_eq!((v.frequencies.len(), v.n_imaginary), (3, 0));
        assert_eq!(
            (v.symmetry.point_group.as_str(), v.symmetry.symmetry_number),
            ("C2v", 2)
        );
        assert!(
            v.frequencies[0] > 1700.0 && v.frequencies[2] < 4200.0,
            "{:?}",
            v.frequencies
        );
        let ir = v.ir_intensities.as_ref().unwrap();
        assert!(ir.iter().all(|&i| i > 1.0), "{ir:?}");
        // one thermochemistry per temperature, in order
        assert_eq!(f.thermo.len(), 2);
        let t = &f.thermo[0];
        assert_eq!(t.options.temperature, 300.0);
        assert!(t.options.qrrho_entropy && !t.options.qrrho_enthalpy);
        assert!((t.total.gibbs - (t.total.enthalpy - 300.0 * t.total.entropy)).abs() < 1e-12);
        assert!((t.gibbs_correction - (t.total.gibbs - f.hessian.energy)).abs() < 1e-12);
        assert_eq!(f.thermo[1].options.temperature, 400.0);
        assert!(f.max_gradient() < 0.05);
        assert!(
            CalcOptions::from_json(serde_json::json!({"freq": {"temprature": 300.0}})).is_err()
        );
    }
}
