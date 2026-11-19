//! The continuum models and the parameters they run with: the model ([`PcmModel`]), presets,
//! discretization and SCF modes, and the parameters resolved for one molecule ([`ResolvedPcm`]:
//! radii, Lebedev grids, permittivity from the solvent database). The `solvent` settings group
//! that resolves into them is the facade's.

use super::{db, surface};
use serde::{Deserialize, Serialize};

/// The continuum model (the form of K and R; see the `pcm` module).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(try_from = "String", into = "String")]
pub enum PcmModel {
    /// Conductor-like PCM, f = (eps - 1) / eps.
    #[default]
    Cpcm,
    /// COSMO, f = (eps - 1) / (eps + x), x = 0.5 by default.
    Cosmo,
    /// Integral-equation formalism PCM.
    Iefpcm,
    /// Surface and simulation of volume polarization for electrostatics (symmetric IEF).
    Ssvpe,
    /// SMD (Marenich, Cramer & Truhlar 2009): bulk electrostatics with SMD's Coulomb radii (IEF-PCM
    /// as PySCF and Gaussian; C-PCM with the `orca` preset, as ORCA) plus the non-electrostatic
    /// term G_CDS ([`smd::cds`](crate::smd::cds)). Needs a solvent name with SMD descriptors.
    Smd,
}

impl PcmModel {
    pub fn name(&self) -> &'static str {
        match self {
            PcmModel::Cpcm => "cpcm",
            PcmModel::Cosmo => "cosmo",
            PcmModel::Iefpcm => "iefpcm",
            PcmModel::Ssvpe => "ssvpe",
            PcmModel::Smd => "smd",
        }
    }
}

impl std::str::FromStr for PcmModel {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, String> {
        let k: String = s
            .to_ascii_lowercase()
            .chars()
            .filter(|c| c.is_ascii_alphanumeric())
            .collect();
        Ok(match k.as_str() {
            "cpcm" | "cosmocpcm" => PcmModel::Cpcm,
            "cosmo" => PcmModel::Cosmo,
            "iefpcm" | "ief" | "pcm" => PcmModel::Iefpcm,
            "ssvpe" => PcmModel::Ssvpe,
            "smd" => PcmModel::Smd,
            _ => {
                return Err(format!(
                    "unknown PCM model {s:?}; use cpcm, cosmo, iefpcm, ssvpe or smd"
                ))
            }
        })
    }
}

impl TryFrom<String> for PcmModel {
    type Error = String;
    fn try_from(s: String) -> Result<Self, String> {
        s.parse()
    }
}

impl From<PcmModel> for String {
    fn from(m: PcmModel) -> String {
        m.name().into()
    }
}

/// Switching function of the surface (`surface`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Discretization {
    /// Switching/Gaussian, 5th-order polynomial switching (Lange & Herbert 2010).
    #[serde(alias = "SWIG", alias = "Swig")]
    Swig,
    /// Improved SWIG, erf switching.
    #[serde(alias = "ISWIG", alias = "Iswig", alias = "ISwIG")]
    Iswig,
}

/// How the SCF evaluates the surface terms (the solute potential at the points and the
/// charges' Fock contribution) in every iteration (`SolventSettings::scf_terms`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PcmScfMode {
    /// Two passes of exact surface integrals per iteration: the potential of the density, then
    /// the Fock term of its charges.
    #[serde(
        alias = "EXACT",
        alias = "Exact",
        alias = "two-pass",
        alias = "two_pass"
    )]
    Exact,
    /// One fused pass of exact integrals per iteration: the exact potential of the density
    /// (exact energy) and the Fock term of charges predicted from the previous iteration's
    /// exact potential plus the RI-J fitted change of the density (the `ri` module). The fixed point is
    /// the exact PCM solution (at convergence the prediction is exact); needs density-fitted J.
    #[serde(alias = "PREDICTED", alias = "Predicted")]
    Predicted,
    /// The density-fitted PCM (the `ri` module): potential and Fock term through the auxiliary
    /// basis of the density-fitted J, riding on the J build; an approximation (with
    /// def2-universal-jfit about up to 4% of the electrostatic solvation energy too negative), for
    /// screening. `df`, the glossary's name for density fitting (docs/conventions.md); `ri`, its
    /// earlier name, and the other spellings below are accepted as aliases.
    #[serde(
        alias = "DF",
        alias = "Df",
        alias = "df-pcm",
        alias = "df_pcm",
        alias = "ri",
        alias = "RI",
        alias = "Ri",
        alias = "ri-pcm",
        alias = "RI-PCM",
        alias = "ri_pcm"
    )]
    Df,
}

impl PcmScfMode {
    pub fn name(&self) -> &'static str {
        match self {
            PcmScfMode::Exact => "exact",
            PcmScfMode::Predicted => "predicted",
            PcmScfMode::Df => "df",
        }
    }
}

/// Sets of cavity and discretization defaults.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum PcmPreset {
    /// PySCF 2.15 `pyscf.solvent.pcm`: modified Bondi radii (PySCF's table, H 1.1 Å) x 1.2,
    /// 302-point Lebedev spheres, SWIG, PySCF's Bohr radius (the default; its discretization
    /// error against dense surfaces is 0.4-0.8% of E_solv, bench/results/2026-11-08-pcm-perf).
    #[default]
    #[serde(alias = "PySCF", alias = "PYSCF", alias = "Pyscf")]
    Pyscf,
    /// ORCA 6.0.0's default C-PCM ("GAUSSIAN VDW" surface; Garcia-Ratés & Neese, J. Comput.
    /// Chem. 41, 922 (2020)): Bondi radii with H 1.1 Å (and Mantina's) x 1.2; per sphere the
    /// Lebedev grid of ORCA's list picked by a point density; ORCA's exponent constants; ISWIG
    /// with points dropped below a switching cut; the solute potential taken at the points
    /// (Gaussian charges only in S). Its constants, read off ORCA's output, are
    /// [`crate::surface::OrcaPcm`].
    /// Give ORCA's own eps for exact comparisons.
    #[serde(alias = "ORCA", alias = "Orca")]
    Orca,
}

/// Settings resolved for one molecule.
#[derive(Clone, Debug)]
pub struct ResolvedPcm {
    pub model: PcmModel,
    pub eps: f64,
    pub cosmo_x: f64,
    /// Lebedev points and exponent constant xi_N of each atom's sphere.
    pub lebedev: Vec<usize>,
    pub xi: Vec<f64>,
    pub discretization: Discretization,
    pub cut: surface::PointCut,
    /// Solute potential at the points (ORCA) instead of over the Gaussians (PySCF).
    pub point_potential: bool,
    /// Sphere radius per atom (bohr).
    pub radii: Vec<f64>,
    pub int_thresh: f64,
    /// Surface-integral gradient on the Rys kernels (None: unless the SCF mode is exact).
    pub grad_rys: Option<bool>,
    /// Incremental fused passes in the predicted SCF mode.
    pub incremental: bool,
    /// SMD: the solvent's descriptors (G_CDS is added to the energy and its derivatives).
    pub smd: Option<db::SmdDescriptors>,
}
