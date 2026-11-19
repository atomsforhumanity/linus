//! The `solvent` settings group ([`SolventSettings`]) and its resolution for one molecule into
//! linus-solvent's [`ResolvedPcm`](linus_solvent::ResolvedPcm) (radii, Lebedev grids,
//! permittivity from the solvent database).

use crate::driver::schema::Auto;
use linus_base::error::LinusError;
use linus_solvent::model::{Discretization, PcmModel, PcmPreset, ResolvedPcm};
use linus_solvent::{db, radii, smd, surface};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// How the SCF evaluates the surface terms (`solvent.scf_terms`; the density-fitted PCM is
/// `solvent.df`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PcmScfTerms {
    /// Two passes of exact surface integrals per iteration (`PcmScfMode::Exact`).
    #[serde(alias = "two_pass")]
    Exact,
    /// One fused pass per iteration with predicted charges (`PcmScfMode::Predicted`).
    Predicted,
}

crate::driver::schema::setting_enum!(PcmModel: "cpcm", "cosmo", "iefpcm", "ssvpe", "smd");
crate::driver::schema::setting_enum!(PcmPreset: "pyscf", "orca");
crate::driver::schema::setting_enum!(Discretization: "swig", "iswig");
crate::driver::schema::setting_enum!(PcmScfTerms: "exact", "predicted");

crate::driver::schema::settings_group! {
    /// The implicit solvent: a polarizable continuum model with Gaussian surface charges
    /// (absent: gas phase). Shorthand: a solvent name or a permittivity.
    pub struct SolventSettings at "solvent." {
        /// The solvent, by name (`--list-solvents`; the rest of the CRC table through
        /// `LINUS_SOLVENT_TABLE`): its permittivity and, for SMD, its descriptors.
        name: Option<String> = None, [common, model];
        /// The static permittivity (overrides the named solvent's; at least 1).
        eps: Option<f64> = None, [common, model];
        /// The continuum model: `cpcm`, `cosmo`, `iefpcm`, `ssvpe` or `smd` (needs a solvent
        /// name with SMD descriptors).
        model: PcmModel = PcmModel::Cpcm, [common, model];
        /// The cavity and discretization defaults: `pyscf` (PySCF 2.15's pyscf.solvent.pcm) or
        /// `orca` (ORCA 6.0.0's C-PCM).
        preset: PcmPreset = PcmPreset::Pyscf, [expert, model];
        /// The density-fitted PCM: the surface terms through the auxiliary basis of the
        /// density-fitted J (about up to 4% of the electrostatic solvation energy; no Hessians
        /// or response properties).
        df: bool = false, [expert, approximation];
        /// Lebedev points per sphere. `auto`: 302 (SMD 590) with the pyscf preset, by point
        /// density with orca.
        lebedev: Auto<usize> = Auto::Auto, [expert, approximation];
        /// The switching of the surface: `swig` or `iswig`. `auto`: by the preset (pyscf swig,
        /// orca iswig).
        discretization: Auto<Discretization> = Auto::Auto, [expert, approximation];
        /// Per-element radii before scaling, by symbol or atomic number (`{"H": 1.2}`); they
        /// override the preset's table.
        radii: std::collections::BTreeMap<String, f64> = Default::default(), [expert, model, unit = "Å"];
        /// The scale of the radii. `auto`: 1.2 (SMD 1.0).
        radii_scale: Auto<f64> = Auto::Auto, [expert, model];
        /// A probe radius added to the scaled radii.
        r_probe: f64 = 0.0, [expert, model, unit = "Å"];
        /// COSMO's x in f = (eps - 1) / (eps + x).
        cosmo_x: f64 = 0.5, [expert, model];
        /// SMD: the window over which new sphere contacts of the CDS surface area are smoothed.
        /// `auto`: 0 (the exact area) for single points, 0.1 for gradients and Hessians.
        cds_contact_smoothing: Auto<f64> = Auto::Auto, [expert, model, unit = "Å"];
        /// How the SCF evaluates the surface terms: `exact` (two passes of exact integrals per
        /// iteration) or `predicted` (one fused pass with predicted charges; the same converged
        /// result). `auto`: predicted with density-fitted J, else exact.
        scf_terms: Auto<PcmScfTerms> = Auto::Auto, [expert, algorithm];
        /// The predicted mode's passes on the changes of the density and the charges.
        incremental: bool = true, [expert, algorithm];
        /// The screening threshold of the surface-charge integrals.
        screening: f64 = 1e-14, [expert, approximation];
        /// Gradients: the surface-integral derivatives on the batched Rys kernels rather than the
        /// McMurchie-Davidson reference. `auto`: Rys in every SCF mode.
        rys_gradient: Auto<bool> = Auto::Auto, [dev, algorithm];
    }
}

impl SolventSettings {
    /// Permittivity: `eps`, else the solvent database's value for `solvent`.
    #[allow(clippy::neg_cmp_op_on_partial_ord, reason = "true for NaN too")]
    pub fn permittivity(&self) -> Result<f64, String> {
        if let Some(e) = self.eps() {
            if !(e >= 1.0) {
                return Err(format!("solvent.eps must be >= 1, not {e}"));
            }
            return Ok(e);
        }
        match &self.name() {
            Some(name) => lookup_eps(name),
            None => Err("a solvent needs solvent.name or solvent.eps".into()),
        }
    }

    /// Resolve the settings for atoms `z`.
    #[allow(clippy::neg_cmp_op_on_partial_ord, reason = "true for NaN too")]
    pub fn resolve(&self, z: &[u32]) -> Result<ResolvedPcm, String> {
        let name = self.name();
        let smd = match self.model() {
            PcmModel::Smd => {
                let name = name
                    .as_deref()
                    .ok_or("SMD needs solvent.name (its descriptors come from solvent::db)")?;
                let s = db::lookup(name).ok_or_else(|| unknown_solvent(name))?;
                Some(
                    s.smd
                        .ok_or_else(|| format!("solvent {name:?} has no SMD descriptors"))?,
                )
            }
            _ => None,
        };
        // eps: `eps` if given (program values for exact comparisons), else the solvent database's
        // default (qcref/CRC at 298.15 K, measured before fitted), SMD included
        let eps = self.permittivity()?;
        // the orca preset's constants (read off ORCA's output)
        let preset = self.preset();
        let orca = match preset {
            PcmPreset::Orca => Some(surface::OrcaPcm::get()),
            PcmPreset::Pyscf => None,
        };
        // (Lebedev points, switching, radius scale, Bohr radius in Å of the radius conversion)
        let (disc, scale, bohr) = match preset {
            PcmPreset::Pyscf => (
                Discretization::Swig,
                1.2,
                linus_core::units::BOHR_ANGSTROM_PYSCF,
            ),
            PcmPreset::Orca => (Discretization::Iswig, 1.2, linus_core::units::BOHR_ANGSTROM),
        };
        let disc = self.discretization().or_else(|| disc);
        // SMD's Coulomb radii are used unscaled
        let scale = self
            .radii_scale()
            .or_else(|| if smd.is_some() { 1.0 } else { scale });
        let probe = self.r_probe();
        let mut custom: BTreeMap<u32, f64> = BTreeMap::new();
        for (k, &v) in &self.radii() {
            let zz = match k.trim().parse::<u32>() {
                Ok(n) => n,
                Err(_) => crate::structure::symbol_to_z(k)
                    .ok_or_else(|| format!("radii: unknown element {k:?}"))?,
            };
            if !(v > 0.0) {
                return Err(format!("radii: radius of {k} must be positive"));
            }
            custom.insert(zz, v);
        }
        // SMD Coulomb radii (Å): PySCF's table with the pyscf preset, SMD/ORCA's otherwise
        let smd_radii: Option<Vec<f64>> = smd.map(|d| {
            let scheme = match preset {
                PcmPreset::Pyscf => smd::RadiiScheme::Pyscf,
                _ => smd::RadiiScheme::Smd,
            };
            smd::coulomb_radii_angstrom(z, d.alpha, scheme)
        });
        let mut radii = Vec::with_capacity(z.len());
        for (i, &zz) in z.iter().enumerate() {
            let r = match custom.get(&zz) {
                Some(&r) => r,
                None if smd_radii.is_some() => smd_radii.as_ref().unwrap()[i],
                None => {
                    let r = radii::modified_bondi(zz)
                        .ok_or_else(|| format!("no cavity radius for element {zz}"))?;
                    match preset {
                        PcmPreset::Pyscf => r,
                        PcmPreset::Orca if radii::is_placeholder(r) => {
                            orca.expect("orca data").fallback_radius_angstrom
                        }
                        PcmPreset::Orca => r,
                    }
                }
            };
            radii.push((scale * r + probe) / bohr);
        }
        let (lebedev, xi, cut, point_potential): (Vec<usize>, Vec<f64>, surface::PointCut, bool) =
            match preset {
                PcmPreset::Pyscf => {
                    // PySCF's SMD discretizes the cavity with its SASA grid (590 points)
                    let n = self
                        .lebedev()
                        .or_else(|| if smd.is_some() { 590 } else { 302 });
                    let x = surface::xi_for(n).ok_or_else(|| {
                    format!("lebedev: {n} points per sphere has no Gaussian exponent constant; use one of {}", surface::XI.iter().map(|x| x.0.to_string()).collect::<Vec<_>>().join(", "))
                })?;
                    (
                        vec![n; z.len()],
                        vec![x; z.len()],
                        surface::PointCut::Pyscf,
                        false,
                    )
                }
                PcmPreset::Orca => {
                    let o = orca.expect("orca data");
                    let ns: Vec<usize> = match self.lebedev() {
                        Auto::Value(n) => vec![n; z.len()],
                        Auto::Auto => radii.iter().map(|r| o.lebedev_for(r * bohr)).collect(),
                    };
                    let xs = ns
                    .iter()
                    .map(|&n| o.xi_for(n).ok_or_else(|| format!("lebedev: the orca preset has exponent constants for {} points per sphere, not {n}", o.xi.iter().map(|x| x.0.to_string()).collect::<Vec<_>>().join(", "))))
                    .collect::<Result<Vec<f64>, String>>()?;
                    (ns, xs, surface::PointCut::Orca(o.cut_swf), true)
                }
            };
        // SMD's bulk electrostatics: IEF-PCM (PySCF, Gaussian), C-PCM with the orca preset (ORCA)
        let model = match (self.model(), preset) {
            (PcmModel::Smd, PcmPreset::Orca) => PcmModel::Cpcm,
            (PcmModel::Smd, _) => PcmModel::Iefpcm,
            (m, _) => m,
        };
        Ok(ResolvedPcm {
            model,
            eps,
            cosmo_x: self.cosmo_x(),
            lebedev,
            xi,
            discretization: disc,
            cut,
            point_potential,
            radii,
            int_thresh: self.screening(),
            grad_rys: self.rys_gradient().value(),
            incremental: self.incremental(),
            smd,
        })
    }
}

/// Permittivity of a named solvent: the solvent database's default ([`db::Solvent::eps`]:
/// qcref/CRC at 298.15 K).
fn lookup_eps(name: &str) -> Result<f64, String> {
    db::lookup(name)
        .map(|s| s.eps)
        .ok_or_else(|| unknown_solvent(name))
}

/// An unknown solvent name: the closest names of the database, where to list them, and the
/// state of the full CRC table that names outside the shipped one resolve through.
fn unknown_solvent(name: &str) -> String {
    let names: Vec<String> = db::solvents()
        .iter()
        .chain(db::full_table_solvents())
        .map(|s| s.name.to_lowercase())
        .collect();
    let hint = linus_base::error::did_you_mean(&name.to_lowercase(), &names);
    format!(
        "unknown solvent '{name}'.{hint} {} linus.available_solvents() and `linus --list-solvents` list the shipped ones (or give eps)",
        db::full_table_note()
    )
}

/// Validate solvent settings (from `CalcOptions::check`).
#[allow(clippy::neg_cmp_op_on_partial_ord, reason = "true for NaN too")]
pub fn check(opts: &Option<SolventSettings>) -> Result<(), LinusError> {
    if let Some(s) = opts {
        s.permittivity().map_err(LinusError::Input)?;
        if !(s.cosmo_x() >= 0.0) {
            return Err(LinusError::Input(format!(
                "solvent.cosmo_x must be >= 0, not {}",
                s.cosmo_x()
            )));
        }
        if s.df() && s.scf_terms.is_some() {
            return Err(LinusError::Setting {
                code: "no_effect",
                path: "solvent.scf_terms".into(),
                message:
                    "the density-fitted PCM (solvent.df) has its own SCF terms (remove the setting)"
                        .into(),
            });
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use linus_solvent::model::*;

    /// `solvent.scf_terms`: the density-fitted PCM is `df` (reported so), with `ri` and the other
    /// spellings as input aliases.
    #[test]
    fn scf_mode_df_and_its_aliases() {
        for s in [
            "df", "DF", "df-pcm", "df_pcm", "ri", "RI", "ri-pcm", "RI-PCM", "ri_pcm",
        ] {
            let m: PcmScfMode = serde_json::from_value(serde_json::json!(s)).unwrap();
            assert_eq!(m, PcmScfMode::Df, "{s}");
        }
        assert_eq!(PcmScfMode::Df.name(), "df");
        assert_eq!(serde_json::to_value(PcmScfMode::Df).unwrap(), "df");
        let p: PcmScfMode = serde_json::from_value(serde_json::json!("predicted")).unwrap();
        assert_eq!(p, PcmScfMode::Predicted);
    }
}
