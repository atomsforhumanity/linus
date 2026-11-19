//! Ideal-gas thermochemistry (rigid rotor, harmonic oscillator; optionally quasi-RRHO) from a
//! harmonic analysis.
//!
//! Per molecule, in Eh and Eh/K (R stands for k_B):
//! - electronic: E = E_el (the input energy), S = R ln g with g the spin multiplicity (no
//!   excited states, no orbital degeneracy);
//! - translational: E = 3/2 RT, H = 5/2 RT (the pV = RT of the ideal gas is put here, as
//!   PySCF and Psi4 do), S = R (ln q_t + 5/2), q_t = (2π M k T / h^2)^(3/2) V, V = kT / p, or
//!   1 / (c N_A) for a concentration standard state c;
//! - rotational: linear E = RT, S = R (ln(8π^2 I kT / (σ h^2)) + 1); nonlinear E = 3/2 RT,
//!   S = R (ln(sqrt(π I_A I_B I_C) (8π^2 kT / h^2)^(3/2) / σ) + 3/2);
//! - vibrational, per real mode with θ = h c ν / k and x = θ / T: ZPE = R θ / 2,
//!   E = R θ (1/2 + 1/(e^x - 1)), S = R (x / (e^x - 1) - ln(1 - e^-x)), Cv = R x^2 e^x / (e^x - 1)^2.
//!
//! Imaginary modes are left out of every vibrational term and listed in
//! `excluded_imaginary` (a transition state's reaction coordinate; also any spurious ones,
//! which the caller should look at). `imaginary_as_real` (cm^-1) instead treats imaginary
//! modes smaller than that as real modes of |ν| (and lists them in `inverted_imaginary`).
//!
//! Low-frequency treatments (all off by default, which gives plain RRHO as Gaussian, Psi4 and
//! PySCF):
//! - `qrrho_entropy` (Grimme, Chem. Eur. J. 18, 9955 (2012), eqs 3-8):
//!   S = w S_harm + (1 - w) S_rot, w = 1 / (1 + (ν0/ν)^α), S_rot = R (1/2 + ln sqrt(8π^3 μ' kT / h^2))
//!   with μ = h / (8π^2 c ν), μ' = μ B_av / (μ + B_av), ν0 = `qrrho_frequency` (100 cm^-1),
//!   α = `qrrho_alpha` (4), B_av = `qrrho_bav` (1e-44 kg m^2). ORCA does this by default.
//! - `qrrho_enthalpy` (Li, Gomes, Sharada, Bell, Head-Gordon, J. Phys. Chem. C 119, 1840
//!   (2015), eqs 4-6): E = w E_harm + (1 - w) RT/2 with the same w; E_harm includes the mode's
//!   zero-point energy, so the ZPE of soft modes is damped too (the `zpe` reported stays the
//!   harmonic one).
//! - `raise_low_frequencies` (Ribeiro, Marenich, Cramer, Truhlar, J. Phys. Chem. B 115, 14556
//!   (2011)): real frequencies below this value (100 cm^-1 there) are raised to it for all
//!   vibrational terms (ZPE, E, S, Cv), "quasiharmonic".
//! - `frequency_scale`: frequencies are multiplied by this factor before everything above.

use super::units::Constants;
use super::{RotorType, VibrationalAnalysis};
use linus_base::error::LinusError;
use serde::{Deserialize, Serialize};

/// Settings of the thermochemistry (see `super::from_json_strict`).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ThermoOptions {
    /// Temperature (K).
    pub temperature: f64,
    /// Pressure (Pa) of the ideal-gas standard state.
    pub pressure: f64,
    /// Concentration standard state (mol/L) instead of `pressure`, e.g. 1.0 for 1 M.
    pub concentration: Option<f64>,
    /// Grimme's quasi-RRHO entropy.
    pub qrrho_entropy: bool,
    /// Head-Gordon's quasi-RRHO energy (enthalpy).
    pub qrrho_enthalpy: bool,
    /// ν0 (cm^-1) of the quasi-RRHO weight.
    pub qrrho_frequency: f64,
    /// Exponent α of the quasi-RRHO weight.
    pub qrrho_alpha: f64,
    /// Grimme's average moment of inertia B_av (kg m^2).
    pub qrrho_bav: f64,
    /// Truhlar's quasiharmonic treatment: real frequencies below this (cm^-1) are raised to it.
    pub raise_low_frequencies: Option<f64>,
    /// Imaginary modes smaller than this (cm^-1) are treated as real modes of |ν|.
    pub imaginary_as_real: f64,
    /// Scale factor for the frequencies.
    pub frequency_scale: f64,
    /// Electronic degeneracy; None: the spin multiplicity.
    pub electronic_degeneracy: Option<f64>,
}

impl Default for ThermoOptions {
    fn default() -> Self {
        ThermoOptions {
            temperature: 298.15,
            pressure: 101_325.0,
            concentration: None,
            qrrho_entropy: false,
            qrrho_enthalpy: false,
            qrrho_frequency: 100.0,
            qrrho_alpha: 4.0,
            qrrho_bav: 1e-44,
            raise_low_frequencies: None,
            imaginary_as_real: 0.0,
            frequency_scale: 1.0,
            electronic_degeneracy: None,
        }
    }
}

/// One contribution (electronic, translational, rotational, vibrational) or a total.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ThermoTerm {
    /// Internal energy (Eh). Electronic: E_el; vibrational: includes the ZPE.
    pub energy: f64,
    /// Enthalpy (Eh).
    pub enthalpy: f64,
    /// Entropy (Eh/K).
    pub entropy: f64,
    /// Heat capacity at constant volume (Eh/K).
    pub cv: f64,
    /// Heat capacity at constant pressure (Eh/K).
    pub cp: f64,
    /// Gibbs free energy H - T S (Eh).
    pub gibbs: f64,
}

impl ThermoTerm {
    fn new(energy: f64, enthalpy: f64, entropy: f64, cv: f64, cp: f64, t: f64) -> Self {
        ThermoTerm {
            energy,
            enthalpy,
            entropy,
            cv,
            cp,
            gibbs: enthalpy - t * entropy,
        }
    }
}

/// One vibrational mode's contributions.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct ModeThermo {
    /// Frequency as analysed (cm^-1, negative if imaginary).
    pub frequency: f64,
    /// Frequency used (cm^-1; scaled, raised or inverted), None if left out.
    pub used_frequency: Option<f64>,
    pub zpe: f64,
    /// Energy (Eh), including the ZPE (after the quasi-RRHO enthalpy damping, if on).
    pub energy: f64,
    /// Entropy (Eh/K), after the quasi-RRHO interpolation, if on.
    pub entropy: f64,
    pub cv: f64,
    /// Quasi-RRHO weight w (1: harmonic).
    pub qrrho_weight: f64,
}

/// Thermochemistry at one temperature and pressure. Serialized with qcdata's `Thermochemistry`
/// names (decision D42): `temperature`, `pressure` (absent with a concentration standard state),
/// `concentration`, `electronic_energy`, `zero_point_energy`, the five terms, `symmetry_number`,
/// `electronic_degeneracy` (the one used), `entropy_model` (`rrho`, `qrrho_grimme`,
/// `quasiharmonic_truhlar`), `enthalpy_model` (`rrho`, `qrrho_head_gordon`), `qrrho_frequency`
/// (the quasi-RRHO crossover, or the quasiharmonic cutoff), `qrrho_alpha` and `qrrho_bav` (the
/// quasi-RRHO models), `frequency_scale_factor`, `excluded_imaginary`, `inverted_imaginary`; and
/// linus's own `multiplicity`, `rotor_type`, `mass`, `energy_0k`, the three corrections,
/// `low_frequencies`, `modes` and `warnings`.
#[derive(Clone, Debug, PartialEq)]
pub struct Thermochemistry {
    pub options: ThermoOptions,
    /// Electronic energy E_el (Eh).
    pub electronic_energy: f64,
    pub multiplicity: u32,
    pub symmetry_number: u32,
    pub rotor_type: RotorType,
    /// Total mass (u).
    pub mass: f64,
    /// Zero-point energy (Eh) of the modes used.
    pub zpe: f64,
    pub electronic: ThermoTerm,
    pub translational: ThermoTerm,
    pub rotational: ThermoTerm,
    pub vibrational: ThermoTerm,
    /// Sums of the four terms (E_el included in energy, enthalpy and gibbs).
    pub total: ThermoTerm,
    /// E_el + ZPE (Eh).
    pub energy_0k: f64,
    /// total.energy - E_el, total.enthalpy - E_el, total.gibbs - E_el (Eh): the corrections
    /// to add to an electronic energy (also from another level).
    pub thermal_energy_correction: f64,
    pub enthalpy_correction: f64,
    pub gibbs_correction: f64,
    /// Imaginary frequencies left out (cm^-1).
    pub excluded_imaginary: Vec<f64>,
    /// Imaginary frequencies treated as real (`imaginary_as_real`).
    pub inverted_imaginary: Vec<f64>,
    /// Real frequencies below the quasi-RRHO ν0 (cm^-1), whatever the treatment.
    pub low_frequencies: Vec<f64>,
    /// Per vibrational mode (external modes left out).
    pub modes: Vec<ModeThermo>,
    /// Things to check: imaginary modes left out, soft modes treated as harmonic oscillators.
    pub warnings: Vec<String>,
}

impl Serialize for Thermochemistry {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeMap;
        let o = &self.options;
        let qrrho = o.qrrho_entropy || o.qrrho_enthalpy;
        let mut m = s.serialize_map(None)?;
        m.serialize_entry("temperature", &o.temperature)?;
        if o.concentration.is_none() {
            m.serialize_entry("pressure", &o.pressure)?;
        }
        if let Some(c) = o.concentration {
            m.serialize_entry("concentration", &c)?;
        }
        m.serialize_entry("electronic_energy", &self.electronic_energy)?;
        m.serialize_entry("zero_point_energy", &self.zpe)?;
        m.serialize_entry("electronic", &self.electronic)?;
        m.serialize_entry("translational", &self.translational)?;
        m.serialize_entry("rotational", &self.rotational)?;
        m.serialize_entry("vibrational", &self.vibrational)?;
        m.serialize_entry("total", &self.total)?;
        m.serialize_entry("symmetry_number", &self.symmetry_number)?;
        m.serialize_entry(
            "electronic_degeneracy",
            &o.electronic_degeneracy.unwrap_or(self.multiplicity as f64),
        )?;
        let entropy_model = match (o.qrrho_entropy, o.raise_low_frequencies) {
            (true, _) => "qrrho_grimme",
            (false, Some(_)) => "quasiharmonic_truhlar",
            (false, None) => "rrho",
        };
        m.serialize_entry("entropy_model", entropy_model)?;
        let enthalpy_model = if o.qrrho_enthalpy {
            "qrrho_head_gordon"
        } else {
            "rrho"
        };
        m.serialize_entry("enthalpy_model", enthalpy_model)?;
        if qrrho {
            m.serialize_entry("qrrho_frequency", &o.qrrho_frequency)?;
            m.serialize_entry("qrrho_alpha", &o.qrrho_alpha)?;
            m.serialize_entry("qrrho_bav", &o.qrrho_bav)?;
        } else if let Some(x) = o.raise_low_frequencies {
            m.serialize_entry("qrrho_frequency", &x)?;
        }
        m.serialize_entry("frequency_scale_factor", &o.frequency_scale)?;
        m.serialize_entry("imaginary_as_real", &o.imaginary_as_real)?;
        m.serialize_entry("excluded_imaginary", &self.excluded_imaginary)?;
        m.serialize_entry("inverted_imaginary", &self.inverted_imaginary)?;
        m.serialize_entry("multiplicity", &self.multiplicity)?;
        m.serialize_entry("rotor_type", &self.rotor_type)?;
        m.serialize_entry("mass", &self.mass)?;
        m.serialize_entry("energy_0k", &self.energy_0k)?;
        m.serialize_entry("thermal_energy_correction", &self.thermal_energy_correction)?;
        m.serialize_entry("enthalpy_correction", &self.enthalpy_correction)?;
        m.serialize_entry("gibbs_correction", &self.gibbs_correction)?;
        m.serialize_entry("low_frequencies", &self.low_frequencies)?;
        m.serialize_entry("modes", &self.modes)?;
        m.serialize_entry("warnings", &self.warnings)?;
        m.end()
    }
}

impl ThermoOptions {
    /// The values in range (temperature, pressure or concentration, scales and quasi-RRHO
    /// parameters positive), so that a frequency calculation fails before its Hessian.
    #[allow(clippy::neg_cmp_op_on_partial_ord, reason = "true for NaN too")]
    pub fn check(&self) -> Result<(), LinusError> {
        let t = self.temperature;
        if !(t > 0.0 && t.is_finite()) {
            return Err(LinusError::Input(format!(
                "temperature must be positive, not {t}"
            )));
        }
        if !(self.pressure > 0.0 && self.pressure.is_finite())
            || self
                .concentration
                .is_some_and(|c| !(c > 0.0 && c.is_finite()))
        {
            return Err(LinusError::Input(
                "pressure and concentration must be positive".into(),
            ));
        }
        if !(self.frequency_scale > 0.0) || !(self.qrrho_frequency > 0.0) || !(self.qrrho_bav > 0.0)
        {
            return Err(LinusError::Input(
                "frequency_scale, qrrho_frequency and qrrho_bav must be positive".into(),
            ));
        }
        Ok(())
    }
}

/// Thermochemistry of a molecule with harmonic analysis `vib`, electronic energy `energy`
/// (Eh) and spin `multiplicity`.
pub fn thermochemistry(
    vib: &VibrationalAnalysis,
    energy: f64,
    multiplicity: u32,
    opts: &ThermoOptions,
) -> Result<Thermochemistry, LinusError> {
    opts.check()?;
    let t = opts.temperature;
    if multiplicity < 1 {
        return Err(LinusError::Input("multiplicity must be at least 1".into()));
    }
    let cst = Constants::of(vib.codata);
    let pi = std::f64::consts::PI;
    let (h, kb, c) = (cst.h, cst.kb, cst.c);
    let r = cst.kb_hartree(); // Eh/K per molecule
    let rt = r * t;
    // electronic
    let g = opts.electronic_degeneracy.unwrap_or(multiplicity as f64);
    let electronic = ThermoTerm::new(energy, energy, r * g.ln(), 0.0, 0.0, t);
    // translational
    let mass = vib.total_mass();
    let mkg = mass * cst.amu;
    let volume = match opts.concentration {
        Some(conc) => 1.0 / (conc * 1000.0 * cst.na),
        None => kb * t / opts.pressure,
    };
    let q_t = (2.0 * pi * mkg * kb * t / (h * h)).powf(1.5) * volume;
    let translational = ThermoTerm::new(
        1.5 * rt,
        2.5 * rt,
        r * (q_t.ln() + 2.5),
        1.5 * r,
        2.5 * r,
        t,
    );
    // rotational
    let sigma = vib.symmetry.symmetry_number as f64;
    let si = |i: f64| i * cst.amu * cst.bohr * cst.bohr; // u bohr^2 -> kg m^2
    let rotational = match vib.rotor_type {
        RotorType::Atom => ThermoTerm::default(),
        RotorType::Linear => {
            let q = 8.0 * pi * pi * si(vib.moments_of_inertia[2]) * kb * t / (sigma * h * h);
            ThermoTerm::new(rt, rt, r * (q.ln() + 1.0), r, r, t)
        }
        RotorType::Nonlinear => {
            let prod: f64 = vib.moments_of_inertia.iter().map(|&i| si(i)).product();
            let q = (pi * prod).sqrt() * (8.0 * pi * pi * kb * t / (h * h)).powf(1.5) / sigma;
            ThermoTerm::new(1.5 * rt, 1.5 * rt, r * (q.ln() + 1.5), 1.5 * r, 1.5 * r, t)
        }
    };
    // vibrational
    let theta_per_cm = cst.kelvin_per_wavenumber();
    let mut modes = Vec::new();
    let (mut excluded, mut inverted, mut low) = (Vec::new(), Vec::new(), Vec::new());
    let (mut zpe, mut e_vib, mut s_vib, mut cv_vib) = (0.0, 0.0, 0.0, 0.0);
    for (&f, &ext) in vib.frequencies.iter().zip(&vib.external) {
        if ext {
            continue;
        }
        let mut nu = f * opts.frequency_scale;
        if f < 0.0 {
            if -f < opts.imaginary_as_real {
                inverted.push(f);
                nu = -nu;
            } else {
                excluded.push(f);
                modes.push(ModeThermo {
                    frequency: f,
                    used_frequency: None,
                    zpe: 0.0,
                    energy: 0.0,
                    entropy: 0.0,
                    cv: 0.0,
                    qrrho_weight: 0.0,
                });
                continue;
            }
        }
        if nu < opts.qrrho_frequency {
            low.push(f);
        }
        if let Some(floor) = opts.raise_low_frequencies {
            nu = nu.max(floor);
        }
        if nu <= 0.0 {
            // a zero frequency has no harmonic partition function
            excluded.push(f);
            modes.push(ModeThermo {
                frequency: f,
                used_frequency: None,
                zpe: 0.0,
                energy: 0.0,
                entropy: 0.0,
                cv: 0.0,
                qrrho_weight: 0.0,
            });
            continue;
        }
        let theta = nu * theta_per_cm;
        let x = theta / t;
        let em1 = x.exp_m1();
        let z0 = 0.5 * r * theta;
        let mut e = z0 + r * theta / em1;
        let mut s = r * (x / em1 - (-(-x).exp_m1()).ln());
        let cv = r * x * x * x.exp() / (em1 * em1);
        let w = 1.0 / (1.0 + (opts.qrrho_frequency / nu).powf(opts.qrrho_alpha));
        if opts.qrrho_entropy {
            let mu = h / (8.0 * pi * pi * c * nu * 100.0);
            let mu_eff = mu * opts.qrrho_bav / (mu + opts.qrrho_bav);
            let s_rot = r * (0.5 + (8.0 * pi.powi(3) * mu_eff * kb * t / (h * h)).sqrt().ln());
            s = w * s + (1.0 - w) * s_rot;
        }
        if opts.qrrho_enthalpy {
            e = w * e + (1.0 - w) * 0.5 * rt;
        }
        zpe += z0;
        e_vib += e;
        s_vib += s;
        cv_vib += cv;
        let qw = if opts.qrrho_entropy || opts.qrrho_enthalpy {
            w
        } else {
            1.0
        };
        modes.push(ModeThermo {
            frequency: f,
            used_frequency: Some(nu),
            zpe: z0,
            energy: e,
            entropy: s,
            cv,
            qrrho_weight: qw,
        });
    }
    let vibrational = ThermoTerm::new(e_vib, e_vib, s_vib, cv_vib, cv_vib, t);
    let mut warnings = Vec::new();
    if !excluded.is_empty() {
        warnings.push(format!("{} imaginary mode(s) left out of the vibrational partition function: {excluded:.1?} cm^-1", excluded.len()));
    }
    if !low.is_empty() && !opts.qrrho_entropy && opts.raise_low_frequencies.is_none() {
        warnings.push(format!(
            "{} real mode(s) below {} cm^-1 treated as harmonic oscillators (their entropy is overestimated; \
             qrrho_entropy or raise_low_frequencies treat them as hindered rotors): {low:.1?} cm^-1",
            low.len(),
            opts.qrrho_frequency
        ));
    }
    let terms = [&electronic, &translational, &rotational, &vibrational];
    let sum = |f: fn(&ThermoTerm) -> f64| terms.iter().map(|x| f(x)).sum::<f64>();
    let total = ThermoTerm {
        energy: sum(|x| x.energy),
        enthalpy: sum(|x| x.enthalpy),
        entropy: sum(|x| x.entropy),
        cv: sum(|x| x.cv),
        cp: sum(|x| x.cp),
        gibbs: sum(|x| x.gibbs),
    };
    Ok(Thermochemistry {
        options: opts.clone(),
        electronic_energy: energy,
        multiplicity,
        symmetry_number: vib.symmetry.symmetry_number,
        rotor_type: vib.rotor_type,
        mass,
        zpe,
        electronic,
        translational,
        rotational,
        vibrational,
        total,
        energy_0k: energy + zpe,
        thermal_energy_correction: total.energy - energy,
        enthalpy_correction: total.enthalpy - energy,
        gibbs_correction: total.gibbs - energy,
        excluded_imaginary: excluded,
        inverted_imaginary: inverted,
        low_frequencies: low,
        modes,
        warnings,
    })
}
