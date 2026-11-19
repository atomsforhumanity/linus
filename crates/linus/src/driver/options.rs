//! The settings of a calculation ([`CalcOptions`]): every setting declared once
//! ([`crate::driver::schema`]), with the grammar of its values. The CLI, the JSON request,
//! Python and qcdata's `keywords` use the one tree (docs/cli.md has the generated tables).
//!
//! - **The tree.** Top-level settings say *what model* (`method`, `basis`, `dispersion`,
//!   `reference`, `cartesian`) and *which resources* (`threads`, `memory_gib`); nine groups hold
//!   the rest by concept: the J/K approximation (`jk`), the grids (`grid`), the implicit
//!   solvent (`solvent`), the SCF (`scf`), the response equations of Hessians (`response`),
//!   the hardware (`backend`) and the optional outputs (`output`). A key never repeats its
//!   group; families share a prefix (`scf.stability_*`, `jk.df_*`).
//! - **One spelling.** Keys and enum values ignore case and `-` equals `_`. `"auto"` means
//!   linus decides; null or absent means unset (the default). Other programs' vocabulary is
//!   accepted as synonyms (`rijcosx`, `ri`, `rhf`/`uhf`, `d3(bj)`, `defgrid2`) and never echoed.
//!   Value shorthands stand for a group where they read better: `jk: "df"`, `grid: "linus:3"`
//!   (or a PySCF level), `solvent: "thf"` or a permittivity, `backend: "gpu"`.
//! - **Presence.** Every setting is an `Option` in Rust: `Some` is given, `None` unset; the
//!   accessors (`opts.scf.max_iter()`) apply the defaults. Decision D41's checks read what was
//!   given, so settings built in Rust are checked as strictly as JSON.
//! - **Values.** Grids ([`GridChoice`]) parse from strings (`"pyscf:3"`), integers (a PySCF
//!   grid level) and their tagged objects, and serialize as the strings; the J/K method
//!   resolves to a [`JkMethod`]; [`CalcOptions::normalized`] splits dispersion suffixes off the
//!   method (`b3lyp-d3bj` = `b3lyp` with `dispersion: "d3bj"`, unless linus knows the whole
//!   name, as `wb97x-d3bj`); [`CalcOptions::check`] rejects what the calculation cannot honour
//!   before any expensive work.
//! - The resolvers of `auto` settings ([`CalcOptions::grid_for`] per [`GridRole`],
//!   [`AuxBasis::for_purpose`], [`CalcOptions::convergence_for`], [`CalcOptions::memory`]) live
//!   here, next to the settings they resolve; the plan (`crate::driver::plan`) is their only
//!   caller.

pub use crate::driver::schema::Auto;
use crate::driver::schema::{setting_enum, settings_group, Group, SettingType, Shorthand};
use crate::solvent::SolventSettings;
use linus_base::convergence::{Convergence, GradNorm};
use linus_base::error::LinusError;
use linus_grid::GridSpec;
use linus_ints::basis::BasisDef;
use linus_scf::DiisDrop;
pub use linus_scf::Guess;
use serde::de::Error as _;
use serde::{Deserialize, Deserializer, Serialize};
use serde_json::{json, Map, Value};
use std::path::PathBuf;

use linus_base::memory::MemoryLimit;
pub use linus_jk::df::{DfLongRange, DfPrecision};

/// Lower case, trimmed, `_` as `-`: the key form of the names of the value grammars (J/K
/// methods, grids, auxiliary bases).
pub fn norm_key(s: &str) -> String {
    s.trim().to_ascii_lowercase().replace('_', "-")
}

// ---------------------------------------------------------------------------------- grids

// ---------------------------------------------------------------------------------- grids

/// A grid: `auto` (chosen by the calculation), a reference program's preset, linus's, or a full
/// specification. Serialized as its string form (`"linus:3"`; [`GridChoice::label`]).
#[derive(Clone, Debug, PartialEq)]
pub enum GridChoice {
    /// Chosen by the calculation (the default): [`GridChoice::ENERGY_DEFAULT`] for energies and
    /// gradients, [`GridChoice::HESSIAN_DEFAULT`] for Hessians, frequencies and IR intensities.
    Auto,
    /// PySCF's grid of `level` (0-9).
    Pyscf {
        level: u8,
    },
    /// linus's grids (levels 1-4; `linus_grid::GridSpec::linus`).
    Linus {
        level: u8,
    },
    /// ORCA 6.0.0's DefGrid1-3.
    Orca {
        defgrid: u8,
    },
    Psi4,
    /// ORCA-like COSX grids of DefGrid n (1-3; seminumerical exchange only; see `two_electron::cosx::defgrid_levels`).
    Cosx {
        defgrid: u8,
    },
    /// COSX grids by level (1-4, `GridSpec::cosx`): SCF on levels\[0\], then levels\[1\]; final
    /// energy and gradient on levels\[2\]. Equal levels give one grid and a variational energy
    /// apart from the overlap fitting (seminumerical exchange only).
    CosxLevels {
        levels: [u8; 3],
    },
    /// Any grid, given by its full specification (`linus_grid::GridSpec`: radial scheme, sizes,
    /// pruning, partitioning, weight cutoff), e.g. a preset's with changes.
    Spec(Box<GridSpec>),
}

/// The object form of [`GridChoice`] (what it serializes to), for deserialization.
#[derive(Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
enum GridTagged {
    Auto,
    Pyscf { level: u8 },
    Linus { level: u8 },
    Orca { defgrid: u8 },
    Psi4,
    Cosx { defgrid: u8 },
    CosxLevels { levels: [u8; 3] },
    Spec(Box<GridSpec>),
}

impl From<GridTagged> for GridChoice {
    fn from(g: GridTagged) -> Self {
        match g {
            GridTagged::Auto => GridChoice::Auto,
            GridTagged::Pyscf { level } => GridChoice::Pyscf { level },
            GridTagged::Linus { level } => GridChoice::Linus { level },
            GridTagged::Orca { defgrid } => GridChoice::Orca { defgrid },
            GridTagged::Psi4 => GridChoice::Psi4,
            GridTagged::Cosx { defgrid } => GridChoice::Cosx { defgrid },
            GridTagged::CosxLevels { levels } => GridChoice::CosxLevels { levels },
            GridTagged::Spec(s) => GridChoice::Spec(s),
        }
    }
}

const GRID_FORMS: &str = "auto, a PySCF level N (0-9), pyscf:N, linus:N (1-4), orca:N (1-3), psi4, cosx:N (1-3) or gridx:A,B,C (1-4)";

impl std::str::FromStr for GridChoice {
    type Err = String;

    /// `auto`, `3` or `pyscf:3` (PySCF level), `linus:2`, `orca:2` / `defgrid2`, `psi4`, `cosx:2`
    /// (the ORCA-like COSX grid scheme of DefGrid2, for `k_grid` of rij_snk), `gridx:1,2,3` (COSX
    /// grid levels: first SCF iterations, last ones, final energy and gradient; `gridx:2` = one
    /// grid). `-` and `_` may replace the colon (`pyscf-3`), or it may be left out (`linus2`).
    fn from_str(s: &str) -> Result<Self, String> {
        let t = s.trim().to_ascii_lowercase().replace(['-', '_'], ":");
        let bad = || format!("unknown grid {s:?}; use {GRID_FORMS}");
        let g = match t.as_str() {
            "auto" => GridChoice::Auto,
            "psi4" => GridChoice::Psi4,
            _ if t.bytes().all(|b| b.is_ascii_digit()) => GridChoice::Pyscf {
                level: t.parse().map_err(|_| bad())?,
            },
            _ => {
                let (kind, rest) = match t.split_once(':') {
                    Some((k, l)) => (k.to_string(), l.to_string()),
                    None => {
                        let k = t.trim_end_matches(|c: char| c.is_ascii_digit() || c == ',');
                        (k.to_string(), t[k.len()..].to_string())
                    }
                };
                let num = |x: &str| x.trim().parse::<u8>().map_err(|_| bad());
                match kind.as_str() {
                    "pyscf" => GridChoice::Pyscf { level: num(&rest)? },
                    "linus" => GridChoice::Linus { level: num(&rest)? },
                    "orca" | "defgrid" => GridChoice::Orca {
                        defgrid: num(&rest)?,
                    },
                    "cosx" => GridChoice::Cosx {
                        defgrid: num(&rest)?,
                    },
                    "gridx" => {
                        let v: Vec<u8> = rest.split(',').map(num).collect::<Result<_, _>>()?;
                        match v[..] {
                            [a] => GridChoice::CosxLevels { levels: [a, a, a] },
                            [a, b, c] => GridChoice::CosxLevels { levels: [a, b, c] },
                            _ => return Err(bad()),
                        }
                    }
                    _ => return Err(bad()),
                }
            }
        };
        g.validate()?;
        Ok(g)
    }
}

impl<'de> Deserialize<'de> for GridChoice {
    /// A string (as [`GridChoice`]'s `from_str`), an integer (a PySCF level) or the tagged object.
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        GridChoice::from_value(Value::deserialize(d)?).map_err(D::Error::custom)
    }
}

impl GridChoice {
    /// Grid of energies and gradients unless set: PySCF's level 3 (the PySCF comparisons rely
    /// on it; reviewed and kept on 2026-11-07). For speed, `GridChoice::Linus { level: 2 }` has
    /// about half the points (5-30% less wall time) at twice the gradient error and a slightly
    /// larger energy error (39-atom anion: 4e-5 vs 1.6e-5 Eh); see docs/grids.md.
    pub const ENERGY_DEFAULT: GridChoice = GridChoice::Pyscf { level: 3 };
    /// Grid of analytic Hessians, frequencies and IR intensities unless set: linus level 3
    /// (SSF partitioning, Treutler-Ahlrichs radial grids, 434-point valence shells): harmonic
    /// frequencies within 0.15 cm^-1 of the converged grid on the lactide benchmarks, at about
    /// the points of PySCF's level 3, which is off by up to 1.7 cm^-1 there
    /// (bench/results/2026-11-07-hess-accuracy).
    pub const HESSIAN_DEFAULT: GridChoice = GridChoice::Linus { level: 3 };
    /// Grid of the CPKS XC kernel on linus's XC grids (the default one among them, [`kernel_grid_for`]):
    /// linus level 1 (about a quarter of the points). The kernel's grid error barely reaches the frequencies (L-lactide
    /// B3LYP: kernel on linus:1 or ORCA-like DefGrid1 instead of the SCF grid changes them by
    /// < 0.005 cm^-1), unlike that of the explicit XC terms.
    pub const CPKS_DEFAULT: GridChoice = GridChoice::Linus { level: 1 };
    /// Grid of the CPKS XC kernel of meta-GGAs on linus's XC grids ([`kernel_grid_for`]): linus
    /// level 2. Their kernel is far more grid-sensitive (ROBUST sweep, Phase 2's 80 Hessian
    /// molecules against linus:4, bench/results/2026-11-10-robust/cpksgrid_*): frequency MAD / max
    /// r2SCAN 3.27 / 14.4 cm^-1 with the kernel on linus:1, 2.92 / 9.1 on linus:2, 2.78 / 11.0 on
    /// the SCF grid; TPSS 1.01 / 13.3, 0.84 / 7.6, 0.83 / 7.6; Hessian wall time +15% (median;
    /// the SCF grid +38%); a 39-atom r2SCAN/def2-SVP Hessian (u1an39) +18%.
    pub const CPKS_META_GGA: GridChoice = GridChoice::Linus { level: 2 };

    /// A grid from JSON: a string, an integer (PySCF level) or the tagged object (its `kind`
    /// normalized like the strings).
    pub fn from_value(v: Value) -> Result<GridChoice, String> {
        match v {
            Value::String(s) => s.parse(),
            Value::Number(n) => match n.as_u64() {
                Some(l) if l <= u8::MAX as u64 => {
                    let g = GridChoice::Pyscf { level: l as u8 };
                    g.validate()?;
                    Ok(g)
                }
                _ => Err(format!("grid {n}: a number is a PySCF grid level (0-9)")),
            },
            Value::Object(mut m) => {
                if let Some(Value::String(k)) = m.get("kind") {
                    let k = norm_key(k).replace('-', "_");
                    if matches!(k.as_str(), "auto" | "psi4") && m.len() > 1 {
                        return Err(format!(
                            "grid {k} takes no other fields, not {:?}",
                            m.keys().filter(|x| *x != "kind").collect::<Vec<_>>()
                        ));
                    }
                    m.insert("kind".into(), Value::String(k));
                }
                let g: GridChoice = serde_json::from_value::<GridTagged>(Value::Object(m))
                    .map_err(|e| format!("grid: {e}"))?
                    .into();
                g.validate()?;
                Ok(g)
            }
            other => Err(format!(
                "grid must be a string, a PySCF level or an object, not {other}"
            )),
        }
    }

    /// Levels in range: PySCF 0-9, linus 1-4, ORCA DefGrid 1-3, COSX DefGrid 1-3, COSX levels 1-4.
    pub fn validate(&self) -> Result<(), String> {
        let check = |what: &str, v: u8, lo: u8, hi: u8| -> Result<(), String> {
            if (lo..=hi).contains(&v) {
                Ok(())
            } else {
                Err(format!("{what} {v} does not exist; use {lo}-{hi}"))
            }
        };
        match self {
            GridChoice::Pyscf { level } => check("PySCF grid level", *level, 0, 9),
            GridChoice::Linus { level } => check("linus grid level", *level, 1, 4),
            GridChoice::Orca { defgrid } => check("ORCA DefGrid", *defgrid, 1, 3),
            GridChoice::Cosx { defgrid } => check("COSX DefGrid", *defgrid, 1, 3),
            GridChoice::CosxLevels { levels } => levels
                .iter()
                .try_for_each(|&l| check("COSX grid level", l, 1, 4)),
            GridChoice::Auto | GridChoice::Psi4 | GridChoice::Spec(_) => Ok(()),
        }
    }

    /// A short name (`pyscf:3`, `linus:1`, `orca:2`, `cosx:2`, `gridx:1,2,3`, `psi4`, `auto`,
    /// `spec`), as the strings that parse to it.
    pub fn label(&self) -> String {
        match self {
            GridChoice::Auto => "auto".into(),
            GridChoice::Pyscf { level } => format!("pyscf:{level}"),
            GridChoice::Linus { level } => format!("linus:{level}"),
            GridChoice::Orca { defgrid } => format!("orca:{defgrid}"),
            GridChoice::Psi4 => "psi4".into(),
            GridChoice::Cosx { defgrid } => format!("cosx:{defgrid}"),
            GridChoice::CosxLevels { levels } => {
                format!("gridx:{},{},{}", levels[0], levels[1], levels[2])
            }
            GridChoice::Spec(_) => "spec".into(),
        }
    }

    /// The grid a calculation for `purpose` uses (resolves [`GridChoice::Auto`]).
    pub fn for_purpose(&self, purpose: Purpose) -> GridChoice {
        match (self, purpose) {
            (GridChoice::Auto, Purpose::Hessian) => GridChoice::HESSIAN_DEFAULT,
            (GridChoice::Auto, _) => GridChoice::ENERGY_DEFAULT,
            (g, _) => g.clone(),
        }
    }

    /// The grid specification ([`GridChoice::Auto`]: that of energies).
    pub fn spec(&self) -> GridSpec {
        match self {
            GridChoice::Auto => GridChoice::ENERGY_DEFAULT.spec(),
            GridChoice::Pyscf { level } => GridSpec::pyscf(*level),
            GridChoice::Linus { level } => GridSpec::linus(*level),
            GridChoice::Orca { defgrid } => GridSpec::orca_defgrid(*defgrid),
            GridChoice::Psi4 => GridSpec::psi4_default(),
            GridChoice::Cosx { defgrid } => {
                GridSpec::cosx(linus_jk::cosx::defgrid_levels(*defgrid)[1])
            }
            GridChoice::CosxLevels { levels } => GridSpec::cosx(levels[1]),
            GridChoice::Spec(s) => (**s).clone(),
        }
    }

    /// The (first SCF, last SCF, final energy) COSX grid levels of the COSX grid schemes.
    pub fn cosx_levels(&self) -> Option<[u8; 3]> {
        match self {
            GridChoice::Cosx { defgrid } => Some(linus_jk::cosx::defgrid_levels(*defgrid)),
            GridChoice::CosxLevels { levels } => Some(*levels),
            _ => None,
        }
    }
}

/// The production NLC grid (`nlc_grid = "auto"` with the default XC grid, or any XC grid other
/// than a PySCF, ORCA or Psi4 preset): linus level 1. VV10 energy and VV10 gradient at fixed
/// wB97M-V/def2-TZVPD density against PySCF level 6/4 (bench/results/2026-11-07-vv10/nlc-grids):
/// L-lactide -0.8 µEh and 7e-6 Eh/bohr, the 39-atom anion +0.5 µEh and 1.0e-5, with 0.65x the
/// points of PySCF's level 1 (-0.6 and +1.3 µEh; the pair sums cost the square). Its errors are below those
/// of the XC grid (PySCF level 3: 2-16 µEh, 3e-5-1e-4 Eh/bohr). See docs/grids.md.
pub const NLC_DEFAULT: GridChoice = GridChoice::Linus { level: 1 };

/// The NLC grid a calculation uses: `nlc` unless it is `Auto`; `Auto` follows the XC grid
/// `xc` (as the user gave it, before `Auto` is resolved), so that a reference program's XC grid
/// comes with its own NLC grid: PySCF levels -> PySCF 2.15's `nlcgrids` default, level 3 (whatever
/// the XC level: `Grids`' default; older PySCF used level 1); ORCA DefGrid N ->
/// ORCA 6's NL grid of DefGrid N ([`GridSpec::orca_nl`]); Psi4 -> Psi4's VV10 grid
/// ([`GridSpec::psi4_vv10`]); everything else (the default `Auto`, linus grids, specs) ->
/// [`NLC_DEFAULT`].
pub fn nlc_grid_for(nlc: &GridChoice, xc: &GridChoice) -> GridChoice {
    match (nlc, xc) {
        (GridChoice::Auto, GridChoice::Pyscf { .. }) => GridChoice::Pyscf { level: 3 },
        (GridChoice::Auto, GridChoice::Orca { defgrid }) => {
            GridChoice::Spec(Box::new(GridSpec::orca_nl(*defgrid)))
        }
        (GridChoice::Auto, GridChoice::Psi4) => GridChoice::Spec(Box::new(GridSpec::psi4_vv10())),
        (GridChoice::Auto, _) => NLC_DEFAULT,
        (g, _) => g.clone(),
    }
}

/// The CPKS kernel's grid (`grid.kernel`) of a Hessian whose XC grid is `xc` (resolved: `auto` is
/// [`GridChoice::HESSIAN_DEFAULT`]): `kernel` unless it is `Auto`. `Auto` follows the XC grid's
/// family, as [`nlc_grid_for`] does, and never whether `grid.xc` was given (decision F5): a
/// reference program's grid (PySCF levels, ORCA's DefGrids, Psi4) is the kernel's too, as in that
/// program; linus's grids and full specifications give [`GridChoice::CPKS_DEFAULT`], or
/// [`GridChoice::CPKS_META_GGA`] for a meta-GGA (`meta_gga`).
pub fn kernel_grid_for(kernel: &GridChoice, xc: &GridChoice, meta_gga: bool) -> GridChoice {
    match (kernel, xc) {
        (
            GridChoice::Auto,
            GridChoice::Pyscf { .. } | GridChoice::Orca { .. } | GridChoice::Psi4,
        ) => xc.clone(),
        (GridChoice::Auto, _) if meta_gga => GridChoice::CPKS_META_GGA,
        (GridChoice::Auto, _) => GridChoice::CPKS_DEFAULT,
        (g, _) => g.clone(),
    }
}

impl Serialize for GridChoice {
    /// The string form (`"linus:3"`), a full specification as its tagged object.
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        match self {
            GridChoice::Spec(spec) => {
                #[derive(Serialize)]
                #[serde(tag = "kind")]
                enum Tagged<'a> {
                    #[serde(rename = "spec")]
                    Spec(&'a GridSpec),
                }
                Tagged::Spec(spec).serialize(s)
            }
            g => s.serialize_str(&g.label()),
        }
    }
}

impl SettingType for GridChoice {
    fn schema() -> Value {
        json!({
            "type": ["string", "integer", "object"],
            "x-type": "grid",
            "x-python": "str | int | dict[str, Any]",
            "description": GRID_FORMS,
        })
    }

    fn parse(v: Value) -> Result<Self, LinusError> {
        GridChoice::from_value(v).map_err(LinusError::Input)
    }
}

// ---------------------------------------------------------------------------------- J/K

/// The kind of J/K approximation (`jk.kind`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum JkKind {
    /// Exact four-center integrals (integral-direct unless `jk.exact_storage` says otherwise).
    #[default]
    #[serde(alias = "direct", alias = "conventional")]
    Exact,
    /// Density fitting of J and K with one auxiliary basis.
    #[serde(alias = "ri", alias = "rijk", alias = "ri_jk", alias = "dfjk")]
    Df,
    /// Density-fitted J and seminumerical K on a separate grid (ORCA's RIJCOSX).
    #[serde(alias = "rij_cosx", alias = "rijcosx", alias = "snk")]
    RijSnk,
}
setting_enum!(JkKind: "exact", "df", "rij_snk");

/// The J/K method a calculation uses, resolved from the `jk` settings
/// ([`CalcOptions::jk_method`]): how the Coulomb and exact-exchange matrices are approximated,
/// with the auxiliary basis and exchange grid as set (`auto` resolved per purpose where used).
#[derive(Clone, Debug, PartialEq)]
pub enum JkMethod {
    /// Exact four-center integrals.
    Exact,
    /// Density fitting for J and K with one auxiliary basis (default [`AuxBasis::Auto`]).
    Df { aux: AuxBasis },
    /// Density-fitted J (default auxiliary basis [`AUX_RIJ_DEFAULT`]) and seminumerical K on a
    /// separate grid (default [`K_GRID_DEFAULT`]).
    RijSnk { aux: AuxBasis, k_grid: GridChoice },
}

/// An auxiliary basis setting: `"auto"` (chosen by the calculation's purpose,
/// [`AuxBasis::for_purpose`]) or a basis set name (`"a|b"`: a composite, each element from the
/// first set that has it). Serialized as the string.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AuxBasis {
    Auto,
    Named(String),
}

impl AuxBasis {
    /// The setting as written: `auto` or the name.
    pub fn as_str(&self) -> &str {
        match self {
            AuxBasis::Auto => AUX_AUTO,
            AuxBasis::Named(n) => n,
        }
    }

    /// The basis set a calculation for `purpose` uses ([`AUX_ENERGY_DEFAULT`] or
    /// [`AUX_HESSIAN_DEFAULT`] for `auto`).
    pub fn for_purpose(&self, purpose: Purpose) -> &str {
        match (self, purpose) {
            (AuxBasis::Auto, Purpose::Hessian) => AUX_HESSIAN_DEFAULT,
            (AuxBasis::Auto, _) => AUX_ENERGY_DEFAULT,
            (AuxBasis::Named(n), _) => n,
        }
    }
}

impl From<&str> for AuxBasis {
    fn from(s: &str) -> Self {
        if norm_key(s) == AUX_AUTO {
            AuxBasis::Auto
        } else {
            AuxBasis::Named(s.trim().to_string())
        }
    }
}

impl From<String> for AuxBasis {
    fn from(s: String) -> Self {
        AuxBasis::from(s.as_str())
    }
}

impl From<&String> for AuxBasis {
    fn from(s: &String) -> Self {
        AuxBasis::from(s.as_str())
    }
}

impl std::fmt::Display for AuxBasis {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl Serialize for AuxBasis {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(self.as_str())
    }
}

impl<'de> Deserialize<'de> for AuxBasis {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let s = String::deserialize(d)?;
        if s.trim().is_empty() {
            return Err(D::Error::custom(
                "an auxiliary basis needs a name (or \"auto\")",
            ));
        }
        Ok(AuxBasis::from(s))
    }
}

impl SettingType for AuxBasis {
    fn schema() -> Value {
        json!({"type": "string", "x-type": "auto | basis set (\"a|b\": a composite)"})
    }
}

/// Auxiliary basis of rij_snk unless set (ORCA's def2/J).
pub const AUX_RIJ_DEFAULT: &str = "def2-universal-jfit";
/// Grid of the seminumerical exchange of rij_snk unless set, and what `k_grid: "auto"` means:
/// the ORCA-like COSX grids of DefGrid2.
pub const K_GRID_DEFAULT: GridChoice = GridChoice::Cosx { defgrid: 2 };

impl JkMethod {
    /// The auxiliary basis of the density-fitted J (and K, with df) as set (None for exact).
    pub fn aux(&self) -> Option<&AuxBasis> {
        match self {
            JkMethod::Df { aux } | JkMethod::RijSnk { aux, .. } => Some(aux),
            JkMethod::Exact => None,
        }
    }

    /// The grid of the seminumerical exchange (rij_snk; None for the others).
    pub fn k_grid(&self) -> Option<&GridChoice> {
        match self {
            JkMethod::RijSnk { k_grid, .. } => Some(k_grid),
            _ => None,
        }
    }

    /// The highest orbital angular momentum the method supports, if it has a limit of its own:
    /// [`MAX_L_COSX`] for the seminumerical exchange of rij_snk.
    pub fn max_orbital_l(&self) -> Option<usize> {
        match self {
            JkMethod::RijSnk { .. } => Some(MAX_L_COSX),
            _ => None,
        }
    }

    /// Whether analytic second derivatives (Hessians, polarizabilities) exist for the method:
    /// exact integrals and density fitting, not RIJ + COSX.
    pub fn has_second_derivatives(&self) -> bool {
        match self {
            JkMethod::Exact | JkMethod::Df { .. } => true,
            JkMethod::RijSnk { .. } => false,
        }
    }

    /// The canonical name: exact, df, rij_snk.
    pub fn name(&self) -> &'static str {
        match self {
            JkMethod::Exact => "exact",
            JkMethod::Df { .. } => "df",
            JkMethod::RijSnk { .. } => "rij_snk",
        }
    }
}

/// Where the exact four-center integrals of `jk.kind: "exact"` are kept.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExactStorage {
    /// Chosen by linus: integral-direct.
    #[default]
    Auto,
    /// Integral-direct: integrals recomputed (screened) in every Fock build.
    Direct,
    /// All integrals held in memory (nbf^4 doubles; tiny systems and tests).
    #[serde(alias = "in_core")]
    Incore,
}
setting_enum!(ExactStorage: "auto", "direct", "incore");
setting_enum!(DfPrecision: "auto", "f64", "f48", "f40", "f32");
setting_enum!(DfLongRange: "auto", "stored", "one_sided");

impl ExactStorage {
    /// The storage a calculation uses (`auto`: direct).
    pub fn resolved(self) -> ExactStorage {
        match self {
            ExactStorage::Auto => ExactStorage::Direct,
            s => s,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            ExactStorage::Auto => "auto",
            ExactStorage::Direct => "direct",
            ExactStorage::Incore => "incore",
        }
    }
}

settings_group! {
    /// The J/K approximation: how the Coulomb and exact-exchange matrices are computed, and the
    /// storage and screening of their integrals. Shorthand: the kind (`"exact"`; `"incore"` is
    /// exact with `exact_storage: "incore"`).
    pub struct JkSettings at "jk." {
        /// The approximation: `df` (density fitting of J and K with the calculation's auxiliary
        /// basis, `aux`; synonyms `ri`, `rijk`), `exact` four-center integrals, or `rij_snk`
        /// (density-fitted J with seminumerical K, ORCA's RIJCOSX; synonym `rijcosx`).
        kind: JkKind = JkKind::Df, [common, approximation];
        /// The auxiliary basis of `df` and of `rij_snk`'s J. `auto`: def2-universal-jkfit for
        /// energies and gradients, `cc-pvtz-jkfit|def2-universal-jkfit` for Hessians (df);
        /// def2-universal-jfit (rij_snk). `"a|b"` takes each element from the first set that has it.
        aux: AuxBasis = AuxBasis::Auto, [common, approximation];
        /// The grid of `rij_snk`'s seminumerical exchange: `cosx:N` (ORCA-like COSX grids of
        /// DefGrid N), `gridx:A,B,C` (COSX grid levels of the first and last SCF iterations and
        /// the final energy) or any grid. `auto`: `cosx:2`.
        k_grid: GridChoice = GridChoice::Auto, [expert, approximation];
        /// Screening threshold of the exact four-center integrals (derivatives use at most 1e-12).
        screening: f64 = 1e-12, [expert, approximation];
        /// Where `exact` keeps its integrals: `direct` (recomputed in every Fock build) or
        /// `incore` (all held in memory: tiny systems and tests). `auto`: direct.
        exact_storage: ExactStorage = ExactStorage::Auto, [expert, algorithm];
        /// Storage format of `df`'s three-index exchange tensor: `f64`, or the compressed `f48`
        /// (relative error at most 9e-13), `f40` (2e-10), `f32` (6e-8). `auto`: f64; a memory
        /// budget too small for it is a `host_memory` error that names the `memory_gib` needed (the
        /// compressed formats only when set, so memory never changes the arithmetic).
        df_storage: DfPrecision = DfPrecision::Auto, [expert, approximation];
        /// Long-range exchange of range-separated hybrids with `df`: `stored` (B and B^omega in
        /// memory) or `one_sided` (raw integrals of one side per build). `auto`: stored when
        /// both fit the budget.
        df_long_range: DfLongRange = DfLongRange::Auto, [expert, algorithm];
        /// Incremental Fock builds (from the change of the density) for integral-direct exact
        /// J/K and the seminumerical exchange.
        incremental: bool = true, [expert, algorithm];
        /// A full rebuild after this many incremental builds.
        incremental_reset: usize = 20, [expert, algorithm];
        /// The seminumerical exchange moves to its next (finer) grid once max|F_ai| is below
        /// this.
        grid_switch: f64 = 1e-3, [expert, algorithm];
        /// At a grid switch, shift the DIIS subspace's Fock matrices by the change of the
        /// exchange matrix.
        grid_switch_shift: bool = true, [dev, algorithm];
    }
}

impl From<JkMethod> for JkSettings {
    /// The settings that name the J/K method `m` (its auxiliary basis and exchange grid given
    /// as set): `CalcOptions { jk: JkMethod::Df { aux }.into(), ..Default::default() }`.
    fn from(m: JkMethod) -> JkSettings {
        match m {
            JkMethod::Exact => JkSettings {
                kind: Some(JkKind::Exact),
                ..Default::default()
            },
            JkMethod::Df { aux } => JkSettings {
                kind: Some(JkKind::Df),
                aux: Some(aux),
                ..Default::default()
            },
            JkMethod::RijSnk { aux, k_grid } => JkSettings {
                kind: Some(JkKind::RijSnk),
                aux: Some(aux),
                k_grid: Some(k_grid),
                ..Default::default()
            },
        }
    }
}

impl Shorthand for JkSettings {
    fn expand(path: &str, v: Value) -> Result<Map<String, Value>, String> {
        match v {
            Value::String(s) if matches!(norm_key(&s).as_str(), "incore" | "in-core") => {
                Ok(Map::from_iter([
                    ("kind".to_string(), json!("exact")),
                    ("exact_storage".to_string(), json!("incore")),
                ]))
            }
            Value::String(s) => Ok(Map::from_iter([("kind".to_string(), Value::String(s))])),
            v => Err(format!(
                "{path} must be a J/K kind (exact, incore, df, rij_snk) or an object, not {v}"
            )),
        }
    }

    fn schema() -> Option<Value> {
        Some(json!({"type": "string", "x-type": "exact | incore | df | rij_snk"}))
    }
}

// ---------------------------------------------------------------------------------- backend

/// Which hardware runs the calculation's kernels (`backend.kind`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BackendKind {
    #[default]
    Cpu,
    Gpu,
    /// The GPU when this build has one and a device opens, else the CPU.
    Auto,
}
setting_enum!(BackendKind: "cpu", "gpu", "auto");

impl BackendKind {
    /// cpu, gpu or auto.
    pub fn name(self) -> &'static str {
        match self {
            BackendKind::Cpu => "cpu",
            BackendKind::Gpu => "gpu",
            BackendKind::Auto => "auto",
        }
    }
}

/// Where the SCF loop's own work runs (`backend.scf_loop`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ScfLoop {
    /// The device loop where it can run, closed and open shells (decisions D27, D31).
    #[default]
    Auto,
    /// On the CPU, driving the Fock builds of the backend.
    Host,
    /// On the GPU: orbitals, densities, Fock matrices and DIIS stay on the device.
    Device,
}
setting_enum!(ScfLoop: "auto", "host", "device");

/// The kernel cache of a GPU backend (`backend.kernel_cache`): where compiled device kernels
/// are kept between runs. Serialized as its string.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub enum KernelCache {
    /// `$LINUS_GPU_CACHE` (a directory, or `off`), else `$XDG_CACHE_HOME/linus/gpu-kernels`, else
    /// `~/.cache/linus/gpu-kernels`.
    #[default]
    Auto,
    /// In memory only: every run compiles its kernels.
    Off,
    /// This directory.
    Dir(PathBuf),
}

impl KernelCache {
    /// `auto`, `off` or the directory.
    pub fn text(&self) -> String {
        match self {
            KernelCache::Auto => "auto".into(),
            KernelCache::Off => "off".into(),
            KernelCache::Dir(d) => d.display().to_string(),
        }
    }
}

impl Serialize for KernelCache {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(&self.text())
    }
}

impl<'de> Deserialize<'de> for KernelCache {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        KernelCache::parse(Value::deserialize(d)?).map_err(serde::de::Error::custom)
    }
}

impl SettingType for KernelCache {
    fn schema() -> Value {
        json!({"type": "string", "x-type": "auto | off | directory"})
    }

    fn parse(v: Value) -> Result<Self, LinusError> {
        match v {
            Value::String(s) if s.trim().eq_ignore_ascii_case("auto") => Ok(KernelCache::Auto),
            Value::String(s) if s.trim().eq_ignore_ascii_case("off") => Ok(KernelCache::Off),
            Value::String(s) if !s.trim().is_empty() => {
                Ok(KernelCache::Dir(PathBuf::from(s.trim())))
            }
            v => Err(LinusError::Input(format!(
                "auto, off or a directory, not {v}"
            ))),
        }
    }
}

/// GPU device ids (`backend.devices`): one id or a list of them; one device per calculation
/// today.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct Devices(pub Vec<u32>);

impl<'de> Deserialize<'de> for Devices {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        match Value::deserialize(d)? {
            Value::Array(a) => a
                .into_iter()
                .map(serde_json::from_value::<u32>)
                .collect::<Result<Vec<_>, _>>()
                .map(Devices)
                .map_err(D::Error::custom),
            v => serde_json::from_value::<u32>(v)
                .map(|i| Devices(vec![i]))
                .map_err(|_| D::Error::custom("a device id or a list of device ids")),
        }
    }
}

impl SettingType for Devices {
    fn schema() -> Value {
        json!({"anyOf": [{"type": "integer", "minimum": 0}, {"type": "array", "items": {"type": "integer", "minimum": 0}}]})
    }
}

settings_group! {
    /// The hardware that runs the kernels. Shorthand: the kind (`"gpu"`).
    pub struct BackendSettings at "backend." {
        /// `cpu`, `gpu` (every kernel family that has device kernels on an NVIDIA GPU, the
        /// others on the CPU, each with its reason; a build with the `gpu` feature) or `auto`
        /// (the GPU when the build has it and a device opens, else the CPU).
        kind: BackendKind = BackendKind::Cpu, [common, resource];
        /// The GPU to use: a device id (or a list of one).
        devices: Devices = Devices(vec![0]), [expert, resource];
        /// Where the SCF loop's own work runs with a GPU: `host` (the CPU drives the device's
        /// Fock builds) or `device` (the whole loop on the GPU). `auto`: device, closed and open
        /// shells.
        scf_loop: ScfLoop = ScfLoop::Auto, [expert, algorithm];
        /// Where a GPU backend keeps its compiled kernels between runs: a directory or `off`
        /// (memory only). `auto`: `$LINUS_GPU_CACHE`, else `$XDG_CACHE_HOME/linus/gpu-kernels`,
        /// else `~/.cache/linus/gpu-kernels`.
        kernel_cache: KernelCache = KernelCache::Auto, [expert, resource];
    }
}

impl Shorthand for BackendSettings {
    fn expand(path: &str, v: Value) -> Result<Map<String, Value>, String> {
        match v {
            Value::String(s) => Ok(Map::from_iter([("kind".to_string(), Value::String(s))])),
            v => Err(format!(
                "{path} must be cpu, gpu, auto or an object, not {v}"
            )),
        }
    }

    fn schema() -> Option<Value> {
        Some(json!({"type": "string", "enum": ["cpu", "gpu", "auto"]}))
    }
}

impl BackendSettings {
    /// Whether this build can run the settings: `gpu` needs the `gpu` cargo feature; the CPU
    /// computes in FP64 without devices; a GPU backend takes one device.
    pub fn check(&self) -> Result<(), LinusError> {
        let kind = self.kind();
        match kind {
            BackendKind::Gpu if !crate::backend::GPU_BUILT => Err(LinusError::Unsupported(
                "backend \"gpu\": this linus was built without GPU support (the gpu cargo feature); use backend \"cpu\" (the default)"
                    .into(),
            )),
            BackendKind::Gpu | BackendKind::Auto => match self.devices().0.as_slice() {
                [] => Err(LinusError::Input(format!(
                    "backend {}: devices is empty; give one device id (or leave it out for device 0)",
                    kind.name()
                ))),
                [_, _, ..] => Err(LinusError::Unsupported(format!(
                    "backend {}: a calculation runs on one device; give one device id",
                    kind.name()
                ))),
                _ => Ok(()),
            },
            BackendKind::Cpu => {
                if self.devices.is_some() {
                    return Err(LinusError::Input(
                        "backend cpu: devices are GPU device ids; the cpu backend takes none (threads sets its cores)".into(),
                    ));
                }
                if self.scf_loop == Some(ScfLoop::Device) {
                    return Err(LinusError::Input(
                        "backend cpu: scf_loop \"device\" is the GPU's SCF loop; use backend \"gpu\" or \"auto\"".into(),
                    ));
                }
                Ok(())
            }
        }
    }

    /// The device id a GPU backend opens (default 0).
    pub fn device_id(&self) -> u32 {
        self.devices().0.first().copied().unwrap_or(0)
    }
}

// ---------------------------------------------------------------------------------- dispersion
// ---------------------------------------------------------------------------------- dispersion

/// An empirical dispersion model linus computes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Dispersion {
    D3bj,
    /// D3(BJ) with the ATM three-body term.
    D3bjAtm,
    D4,
}

impl Dispersion {
    pub fn name(&self) -> &'static str {
        match self {
            Dispersion::D3bj => "d3bj",
            Dispersion::D3bjAtm => "d3bj-atm",
            Dispersion::D4 => "d4",
        }
    }
}

impl Serialize for Dispersion {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(self.name())
    }
}

/// The `dispersion` setting: `auto` (implied by the method name, e.g. `wb97x-d3bj`; none
/// otherwise), off (`"none"`, also turning an implied model off) or a model (`"d3bj"`,
/// `"d3bj-atm"`, `"d4"`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum DispersionSetting {
    #[default]
    Auto,
    Off,
    Model(Dispersion),
}

impl DispersionSetting {
    /// The setting a dispersion name stands for (`auto`; [`parse_dispersion`]: `none` is
    /// [`DispersionSetting::Off`]).
    pub fn parse(s: &str) -> Result<DispersionSetting, LinusError> {
        if norm_key(s) == "auto" {
            return Ok(DispersionSetting::Auto);
        }
        Ok(match parse_dispersion(s)? {
            None => DispersionSetting::Off,
            Some(d) => DispersionSetting::Model(d),
        })
    }
}

impl Serialize for DispersionSetting {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        match self {
            DispersionSetting::Auto => s.serialize_str("auto"),
            DispersionSetting::Off => s.serialize_str("none"),
            DispersionSetting::Model(d) => s.serialize_str(d.name()),
        }
    }
}

impl<'de> Deserialize<'de> for DispersionSetting {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let s = String::deserialize(d)?;
        DispersionSetting::parse(&s).map_err(D::Error::custom)
    }
}

impl SettingType for DispersionSetting {
    fn schema() -> Value {
        json!({"type": "string", "x-type": "auto | none | d3bj | d3bj-atm | d4"})
    }

    /// Keeps the error's kind (D3 with zero damping is unsupported, bare `d3` an input error).
    fn parse(v: Value) -> Result<Self, LinusError> {
        match v {
            Value::String(s) => DispersionSetting::parse(&s),
            v => Err(LinusError::Input(format!("a dispersion name, not {v}"))),
        }
    }
}

/// A dispersion name: `Ok(None)` for `none` (or empty), the model for `d3bj` (also `d3(bj)`),
/// `d3bj-atm` (`d3bjatm`, `d3bj-abc`), `d4`. Bare `d3` is ambiguous ([`D3_AMBIGUOUS`]) and D3
/// with zero damping is [`LinusError::Unsupported`]; anything else [`LinusError::Input`].
pub fn parse_dispersion(s: &str) -> Result<Option<Dispersion>, LinusError> {
    let k: String = s
        .trim()
        .to_ascii_lowercase()
        .chars()
        .filter(|c| !matches!(c, '(' | ')' | ' '))
        .collect::<String>()
        .replace('_', "-");
    Ok(Some(match k.as_str() {
        "none" | "" => return Ok(None),
        "d3bj" => Dispersion::D3bj,
        "d3" => return Err(LinusError::Input(D3_AMBIGUOUS.into())),
        "d3bj-atm" | "d3bjatm" | "d3bj-abc" | "d3bjabc" => Dispersion::D3bjAtm,
        "d4" => Dispersion::D4,
        "d3zero" | "d3-zero" | "d30" | "d3-0" | "zero" | "d3zero-atm" | "d3m" | "d3mbj" | "d3-m" => {
            return Err(LinusError::Unsupported(format!(
                "dispersion {s:?}: only D3 with Becke-Johnson damping (d3bj, d3bj-atm) and D4 are implemented, not zero or modified damping"
            )))
        }
        _ => return Err(LinusError::Input(format!("unknown dispersion {s:?}; use d3bj, d3bj-atm, d4 or none"))),
    }))
}

/// Bare `d3` names two models: zero damping for Psi4 and qcdata, Becke-Johnson for ORCA.
pub const D3_AMBIGUOUS: &str = "'d3' is ambiguous (Psi4/qcdata: zero damping; ORCA: Becke-Johnson). linus implements D3 with BJ damping: write d3bj (or d3bj-atm, d4).";

/// Dispersion suffixes split off a method name (`b3lyp-d3bj`), longest first, with the
/// canonical dispersion name each stands for (bare `-d3` stays `d3`, which is rejected).
const DISPERSION_SUFFIXES: [(&str, &str); 9] = [
    ("-d3bj-atm", "d3bj-atm"),
    ("-d3(bj)-atm", "d3bj-atm"),
    ("-d3bjatm", "d3bj-atm"),
    ("-d3(bj)", "d3bj"),
    ("-d3bj", "d3bj"),
    ("-d3zero", "d3zero"),
    ("-d3(0)", "d3zero"),
    ("-d3", "d3"),
    ("-d4", "d4"),
];

/// Whether linus-xc knows `method` as a functional (or it is HF).
pub fn functional_known(method: &str) -> bool {
    method.trim().eq_ignore_ascii_case("hf") || linus_xc::Functional::new(method, false).is_ok()
}

/// `("b3lyp", Some("d3bj"))` for `"b3lyp-d3bj"` (also `-d3(bj)`, bare `-d3` (as `d3`, which
/// [`parse_dispersion`] rejects as ambiguous),
/// `-d3bj-atm`, `-d4`, and `-d3zero`/`-d3(0)`, which the dispersion check then rejects), any
/// case; a name linus knows whole (`wb97x-d3bj`: a functional of its own) or that has no such
/// suffix stays whole (`(method, None)`).
pub fn split_dispersion(method: &str) -> (String, Option<String>) {
    let m = method.trim();
    let low = m.to_ascii_lowercase();
    if functional_known(m) {
        return (m.to_string(), None);
    }
    for (suffix, name) in DISPERSION_SUFFIXES {
        if low.ends_with(suffix) && low.len() > suffix.len() {
            let base = &m[..m.len() - suffix.len()];
            return (base.to_string(), Some(name.to_string()));
        }
    }
    (m.to_string(), None)
}

// ---------------------------------------------------------------------------------- settings

/// What the SCF is for; picks the default convergence thresholds, grids and auxiliary basis.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Purpose {
    Energy,
    Gradient,
    Hessian,
}

impl Purpose {
    pub fn name(self) -> &'static str {
        match self {
            Purpose::Energy => "energy",
            Purpose::Gradient => "gradient",
            Purpose::Hessian => "hessian",
        }
    }
}

/// Default SCF thresholds per purpose (used when `scf.convergence` is auto), from the errors
/// against SCFs converged to max |F_ai| < 1e-10 (bench/conv_accuracy.py,
/// bench/hess_conv_accuracy.py; caffeine RHF/def2-SVP exact, la18 and u1an39 B3LYP/def2-SVP
/// RI): the energy error is quadratic in the final max |F_ai| (|dE| <= 2e-11 Eh at the tight
/// preset), the gradient error linear (max |dG| = 3-10 x max |F_ai|), and the analytic
/// Hessian error linear too (max |dH| = 4-5 x max |F_ai|, caffeine RHF RI).
/// - energies: the `tight` preset (|dE| < 1e-8, max |F_ai| < 5e-7): energy errors ~1e-11 Eh.
/// - gradients: max |F_ai| < 1e-7: max |dG| <= ~1e-6 Eh/bohr (measured 2-8e-7), 30x below
///   ORCA's VeryTightOpt TolMAXG (3e-5); ORCA itself optimizes with TightSCF, whose final
///   orbital gradient (~1e-6) means gradient errors of ~1e-5.
/// - Hessians: max |F_ai| < 1e-9: max |dH| ~ 2e-9 Eh/bohr^2 (caffeine RHF RI); the SCF is a
///   small part of a Hessian's cost.
///
/// e_tol is 1e-9 for gradients and Hessians, so that the orbital-gradient criterion decides
/// (the energy is then converged far below the ~1e-11 Eh screening noise of direct J/K).
pub const DEFAULT_ENERGY_CONVERGENCE: Convergence = Convergence {
    e_tol: 1e-8,
    g_tol: 5e-7,
    g_norm: GradNorm::Max,
};
pub const DEFAULT_GRADIENT_CONVERGENCE: Convergence = Convergence {
    e_tol: 1e-9,
    g_tol: 1e-7,
    g_norm: GradNorm::Max,
};
pub const DEFAULT_HESSIAN_CONVERGENCE: Convergence = Convergence {
    e_tol: 1e-9,
    g_tol: 1e-9,
    g_norm: GradNorm::Max,
};

/// The default SCF thresholds of `purpose` (`scf.convergence: auto`).
pub fn default_convergence(purpose: Purpose) -> Convergence {
    match purpose {
        Purpose::Energy => DEFAULT_ENERGY_CONVERGENCE,
        Purpose::Gradient => DEFAULT_GRADIENT_CONVERGENCE,
        Purpose::Hessian => DEFAULT_HESSIAN_CONVERGENCE,
    }
}

/// A named set of SCF thresholds (`scf.convergence`; [`Convergence::preset`] has the values).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConvergencePreset {
    Loose,
    Normal,
    Tight,
    VeryTight,
    Extreme,
    /// PySCF's defaults (conv_tol 1e-9, conv_tol_grad its square root, on PySCF's norm).
    Pyscf,
}

impl ConvergencePreset {
    pub const ALL: [ConvergencePreset; 6] = [
        ConvergencePreset::Loose,
        ConvergencePreset::Normal,
        ConvergencePreset::Tight,
        ConvergencePreset::VeryTight,
        ConvergencePreset::Extreme,
        ConvergencePreset::Pyscf,
    ];

    /// loose, normal, tight, verytight, extreme, pyscf.
    pub fn name(self) -> &'static str {
        match self {
            ConvergencePreset::Loose => "loose",
            ConvergencePreset::Normal => "normal",
            ConvergencePreset::Tight => "tight",
            ConvergencePreset::VeryTight => "verytight",
            ConvergencePreset::Extreme => "extreme",
            ConvergencePreset::Pyscf => "pyscf",
        }
    }

    /// The preset's thresholds.
    pub fn thresholds(self) -> Convergence {
        Convergence::preset(self.name()).expect("every preset has thresholds")
    }
}

impl std::str::FromStr for ConvergencePreset {
    type Err = String;

    /// A preset name, any case, `_` and `-` ignored (`very_tight`).
    fn from_str(s: &str) -> Result<Self, String> {
        let k = s.trim().to_ascii_lowercase().replace(['_', '-'], "");
        ConvergencePreset::ALL
            .into_iter()
            .find(|p| p.name() == k)
            .ok_or_else(|| {
                format!(
                    "unknown SCF convergence preset {s:?}; use one of {}",
                    Convergence::PRESETS.join(", ")
                )
            })
    }
}

impl Serialize for ConvergencePreset {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(self.name())
    }
}

impl<'de> Deserialize<'de> for ConvergencePreset {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        String::deserialize(d)?.parse().map_err(D::Error::custom)
    }
}

setting_enum!(ConvergencePreset: "loose", "normal", "tight", "verytight", "extreme", "pyscf");
setting_enum!(Guess: "sad", "core");
setting_enum!(DiisDrop: "largest_error", "oldest");

/// The determinant (`reference`): restricted (RHF/RKS) or unrestricted (UHF/UKS).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Reference {
    /// Restricted for closed shells, unrestricted for open shells.
    #[default]
    Auto,
    #[serde(alias = "rhf", alias = "rks")]
    Restricted,
    #[serde(alias = "uhf", alias = "uks")]
    Unrestricted,
}
setting_enum!(Reference: "auto", "restricted", "unrestricted");

impl Reference {
    /// Restricted or not (None: by shell).
    pub fn restricted(self) -> Option<bool> {
        match self {
            Reference::Auto => None,
            Reference::Restricted => Some(true),
            Reference::Unrestricted => Some(false),
        }
    }
}

/// The stability analysis of UHF/UKS determinants (`scf.stability`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Stability {
    /// The probes, and a full analysis when one fires or the SCF looks like a saddle point.
    #[default]
    Auto,
    /// No probe and no analysis.
    Off,
    /// A full analysis of every converged UHF/UKS determinant.
    Always,
}
setting_enum!(Stability: "auto", "off", "always");

/// Precision of the CPHF/CPKS exchange of RI Hessians.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResponsePrecision {
    /// As `precision.fp32` says: with `off` FP64; else single precision above an exchange work
    /// n^2 o naux of `two_electron::df::response::SINGLE_WORK`, unless `response.tol` asks for
    /// less than `two_electron::df::response::SINGLE_MIN_TOL` (1e-6).
    #[default]
    Auto,
    #[serde(alias = "single")]
    Fp32,
    #[serde(alias = "double")]
    Fp64,
}
setting_enum!(ResponsePrecision: "auto", "fp32", "fp64");

impl ResponsePrecision {
    /// `auto`, `fp32` or `fp64`.
    pub fn name(self) -> &'static str {
        match self {
            ResponsePrecision::Auto => "auto",
            ResponsePrecision::Fp32 => "fp32",
            ResponsePrecision::Fp64 => "fp64",
        }
    }
}

// ---------------------------------------------------------------------------------- precision

/// How the GPU reaches FP64 accuracy (`precision.fp64`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Fp64Mode {
    /// Native on FP64-rich cards (A100, H100/H200) and the CPU, emulated on the others (D27).
    #[default]
    Auto,
    /// The FP64 units.
    #[serde(alias = "double", alias = "fp64")]
    Native,
    /// The density-fitted exchange GEMMs (and the CPKS kernel's large ones) in FP64 emulation
    /// on INT8 tensor cores (55-bit operands: the Ozaki scheme, else cuBLAS's fixed-mantissa
    /// emulation).
    Emulated,
}
setting_enum!(Fp64Mode: "auto", "native", "emulated");

impl Fp64Mode {
    /// `auto`, `native` or `emulated`.
    pub fn name(self) -> &'static str {
        match self {
            Fp64Mode::Auto => "auto",
            Fp64Mode::Native => "native",
            Fp64Mode::Emulated => "emulated",
        }
    }
}

/// Where FP32 arithmetic may run outside the SCF loop (`precision.fp32`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Fp32Mode {
    /// linus's validated uses: the single-precision response exchange of large RI Hessians
    /// (CPU and GPU) with the GPU's CPKS and VV10 kernels in FP32 when it runs, and the GPU's
    /// VV10 pair sums in double-float (FP32 pairs, FP64-equal).
    #[default]
    Auto,
    /// Nowhere: FP64 throughout (reference runs; the VV10 pair sums in FP64 too).
    Off,
    /// Also the GPU's XC gradient, CPKS kernel, Hessian XC and VV10 gradient products, and the
    /// stored response exchange operator where it pays.
    On,
}
setting_enum!(Fp32Mode: "auto", "off", "on");

impl Fp32Mode {
    /// `auto`, `off` or `on`.
    pub fn name(self) -> &'static str {
        match self {
            Fp32Mode::Auto => "auto",
            Fp32Mode::Off => "off",
            Fp32Mode::On => "on",
        }
    }
}

settings_group! {
    /// The arithmetic. Shorthand: `fp64` (synonyms `native`, `double`: FP64 throughout),
    /// `emulated`, `mixed` (FP32 products where validated and opt-in) or `auto`.
    pub struct PrecisionSettings at "precision." {
        /// How the GPU reaches FP64 accuracy in the density-fitted exchange GEMMs: `native`
        /// (the FP64 units) or `emulated` (FP64 emulation on INT8 tensor cores). `auto`: native
        /// on FP64-rich cards (A100, H100/H200) and the CPU, emulated on the others (D27).
        fp64: Fp64Mode = Fp64Mode::Auto, [expert, approximation];
        /// Where FP32 arithmetic may run outside the SCF loop: `off` (nowhere: FP64 throughout,
        /// VV10's pair sums too) or `on` (also the GPU's XC gradient, CPKS kernel, Hessian XC
        /// and VV10 gradient products). `auto`: linus's validated uses (the single-precision
        /// exchange of large RI responses, the GPU's CPKS and VV10 kernels with it, VV10's pair
        /// sums in double-float).
        fp32: Fp32Mode = Fp32Mode::Auto, [expert, approximation];
        /// The arithmetic of the RI response exchange (CPHF/CPKS): `fp32` or `fp64` (synonyms
        /// `single`, `double`). `auto`: as `fp32` says (with `off`, fp64; else single precision in
        /// large RI Hessians unless `response.tol` is below 1e-6; on the GPU, the card's rule,
        /// docs/gpu.md).
        response: ResponsePrecision = ResponsePrecision::Auto, [expert, approximation, needs = Hessian];
    }
}

impl Shorthand for PrecisionSettings {
    fn expand(path: &str, v: Value) -> Result<Map<String, Value>, String> {
        let pair = |k: &str, x: &str| (k.to_string(), json!(x));
        match &v {
            Value::String(s) => match norm_key(s).as_str() {
                "auto" => Ok(Map::new()),
                "fp64" | "native" | "double" => Ok(Map::from_iter([
                    pair("fp64", "native"),
                    pair("fp32", "off"),
                ])),
                "emulated" => Ok(Map::from_iter([pair("fp64", "emulated")])),
                "mixed" => Ok(Map::from_iter([pair("fp32", "on")])),
                _ => Err(format!(
                    "{path} must be fp64 (native, double), emulated, mixed, auto or an object, not {v}"
                )),
            },
            v => Err(format!(
                "{path} must be fp64 (native, double), emulated, mixed, auto or an object, not {v}"
            )),
        }
    }

    fn schema() -> Option<Value> {
        Some(
            json!({"type": "string", "enum": ["auto", "fp64", "native", "double", "emulated", "mixed"]}),
        )
    }
}

impl PrecisionSettings {
    /// The response exchange's arithmetic as set: `precision.response`, and with it `auto`
    /// under `precision.fp32: off` FP64.
    pub fn response_effective(&self) -> ResponsePrecision {
        match (self.response(), self.fp32()) {
            (ResponsePrecision::Auto, Fp32Mode::Off) => ResponsePrecision::Fp64,
            (r, _) => r,
        }
    }

    /// Whether the CPU can run the settings: FP64 emulation and the opt-in FP32 products are
    /// the GPU's.
    pub fn check(&self, kind: BackendKind) -> Result<(), LinusError> {
        if kind != BackendKind::Cpu {
            return Ok(());
        }
        if self.fp64() == Fp64Mode::Emulated {
            return Err(LinusError::Unsupported(
                "backend cpu: precision.fp64 emulated is the GPU's FP64 emulation; the CPU computes in FP64 (precision.fp64 auto or native)".into(),
            ));
        }
        if self.fp32() == Fp32Mode::On {
            return Err(LinusError::Unsupported(
                "backend cpu: precision.fp32 on is the GPU's FP32 XC products; the CPU's FP32 use is the response exchange (precision.fp32 auto, or precision.response fp32)".into(),
            ));
        }
        Ok(())
    }
}

/// The auxiliary basis of the cheaper J/K of the CPKS iterations of large RI Hessians
/// ("two-level" responses; `response.iteration_aux`). Serialized as its string.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub enum IterationAux {
    /// `two_electron::df::response::ITER_AUX_DEFAULT` when the response runs in single
    /// precision and is large, else none.
    #[default]
    Auto,
    /// No separate iteration J/K.
    None,
    /// The main auxiliary basis.
    Main,
    /// A basis set by name (every element of the molecule must be in it).
    Named(String),
}

impl IterationAux {
    /// `auto`, `none`, `main` or the name.
    pub fn as_str(&self) -> &str {
        match self {
            IterationAux::Auto => "auto",
            IterationAux::None => "none",
            IterationAux::Main => "main",
            IterationAux::Named(n) => n,
        }
    }
}

impl From<&str> for IterationAux {
    fn from(s: &str) -> Self {
        match norm_key(s).as_str() {
            "auto" => IterationAux::Auto,
            "none" => IterationAux::None,
            "main" => IterationAux::Main,
            _ => IterationAux::Named(s.trim().to_string()),
        }
    }
}

impl Serialize for IterationAux {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(self.as_str())
    }
}

impl<'de> Deserialize<'de> for IterationAux {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let s = String::deserialize(d)?;
        if s.trim().is_empty() {
            return Err(D::Error::custom(
                "iteration_aux: auto, none, main or a basis set name",
            ));
        }
        Ok(IterationAux::from(s.as_str()))
    }
}

impl SettingType for IterationAux {
    fn schema() -> Value {
        json!({"type": "string", "x-type": "auto | none | main | basis set"})
    }
}

/// Directories searched first for basis set files (`basis_dir`): a string (one) or a list.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize)]
pub struct BasisDirs(pub Vec<PathBuf>);

impl<'de> Deserialize<'de> for BasisDirs {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        match Value::deserialize(d)? {
            Value::Null => Ok(BasisDirs::default()),
            Value::String(s) => Ok(BasisDirs(vec![PathBuf::from(s)])),
            Value::Array(a) => a
                .into_iter()
                .map(|x| match x {
                    Value::String(s) => Ok(PathBuf::from(s)),
                    other => Err(D::Error::custom(format!(
                        "a directory must be a string, not {other}"
                    ))),
                })
                .collect::<Result<_, _>>()
                .map(BasisDirs),
            other => Err(D::Error::custom(format!(
                "a directory or a list of directories, not {other}"
            ))),
        }
    }
}

impl SettingType for BasisDirs {
    fn schema() -> Value {
        json!({"anyOf": [{"type": "string"}, {"type": "array", "items": {"type": "string"}}]})
    }
}

settings_group! {
    /// The grids of DFT: the XC quadrature, VV10's and the CPKS kernel's. Shorthand: a grid,
    /// the XC grid (`"linus:3"`, or a PySCF level).
    pub struct GridSettings at "grid." {
        /// The XC quadrature of the SCF, gradients and Hessians: a PySCF level `N`/`pyscf:N`
        /// (0-9), `linus:N` (1-4), `orca:N` (ORCA 6.0.0's DefGrid 1-3), `psi4`, or a grid's full
        /// specification `{"kind": "spec", ...}`. `auto`: pyscf:3 for energies and gradients,
        /// linus:3 for Hessians.
        xc: GridChoice = GridChoice::Auto, [common, approximation];
        /// The grid of VV10 nonlocal correlation (-V functionals). `auto`: by the XC grid's
        /// family: linus:1 with the default and linus grids, pyscf:3 with PySCF grids, ORCA's NL
        /// grid with `orca:N`, Psi4's VV10 grid with `psi4`.
        nlc: GridChoice = GridChoice::Auto, [expert, approximation];
        /// The grid of the XC kernel in the CPKS equations of Hessians. `auto`: by the XC grid's
        /// family: linus:1 (meta-GGAs linus:2) with the default and linus grids, the XC grid with
        /// PySCF, ORCA and Psi4 grids.
        kernel: GridChoice = GridChoice::Auto, [expert, approximation, needs = Hessian];
        /// The grids move with the atoms in gradients, Hessians and dipole derivatives (the XC,
        /// VV10 and seminumerical-exchange grids' weight derivatives and point motion, as ORCA);
        /// false: fixed grids, PySCF's and Psi4's convention, for comparisons with them.
        response: bool = true, [expert, approximation, needs = Gradient];
    }
}

impl Shorthand for GridSettings {
    fn expand(_: &str, v: Value) -> Result<Map<String, Value>, String> {
        Ok(Map::from_iter([("xc".to_string(), v)]))
    }

    fn schema() -> Option<Value> {
        Some(json!({"type": ["string", "integer"], "x-type": "grid"}))
    }
}

settings_group! {
    /// The SCF: convergence, iterations, guess, the convergence accelerators and the stability
    /// analysis of UHF/UKS determinants.
    pub struct ScfSettings at "scf." {
        /// A preset of `e_tol` and `g_tol`: `loose` (1e-5, 1e-4), `normal` (1e-6, 1e-5), `tight`
        /// (1e-8, 5e-7), `verytight` (1e-9, 1e-8), `extreme` (1e-12, 1e-10) or `pyscf` (PySCF's
        /// defaults, on PySCF's gradient norm). `auto`: tight for energies, 1e-9 and 1e-7 for
        /// gradients, 1e-9 and 1e-9 for Hessians. The presets are absolute: one looser than the
        /// calculation's default in a threshold warns (`convergence_looser_than_default`; `tight`
        /// loosens a gradient, only `extreme` tightens a Hessian).
        convergence: Auto<ConvergencePreset> = Auto::Auto, [common, convergence];
        /// The energy change between iterations below which the SCF may stop (overrides the
        /// preset's). `auto`: the preset's.
        e_tol: Auto<f64> = Auto::Auto, [common, convergence, unit = "Eh"];
        /// The orbital gradient max|F_ai| below which the SCF may stop (PySCF's norm with the
        /// `pyscf` preset; overrides the preset's). `auto`: the preset's.
        g_tol: Auto<f64> = Auto::Auto, [common, convergence];
        /// Iterations at most: the SCF's whole budget, the escalation's included.
        max_iter: usize = 288, [common, convergence];
        /// The initial guess: `sad` (superposition of atomic densities) or `core`.
        guess: Guess = Guess::Sad, [common, algorithm];
        /// The stability of UHF/UKS determinants: `auto` (the probes below, and a full analysis
        /// with descent when one fires or the SCF looks like a saddle point), `off` (no probe or
        /// analysis) or `always` (a full analysis of every converged determinant).
        stability: Stability = Stability::Auto, [common, algorithm];
        /// Davidson products of the open-shell probe of a converged UHF/UKS determinant (0: off,
        /// else at least 2; a negative mode starts the full analysis). `auto`: 2 at a
        /// session's first geometry, none at later ones unless the gap trigger fires.
        stability_probe: Auto<usize> = Auto::Auto, [expert, algorithm];
        /// Davidson products of the closed-shell probe of a converged RHF/RKS point whose HOMO-LUMO
        /// gap is below `stability_closed_gap` (towards a broken-symmetry UHF/UKS solution; a
        /// warning only, the result is unchanged; 0: off, else at least 2). `auto`: 2 for
        /// Kohn-Sham DFT at a session's first geometry, none for Hartree-Fock.
        stability_closed_probe: Auto<usize> = Auto::Auto, [expert, algorithm];
        /// The HOMO-LUMO gap below which the closed-shell probe runs.
        stability_closed_gap: f64 = linus_scf::CLOSED_SHELL_GAP, [expert, algorithm, unit = "Eh"];
        /// Full stability analyses per SCF at most.
        stability_max: usize = 2, [expert, algorithm];
        /// A converged UHF/UKS energy this far above the lowest one the SCF visited triggers a
        /// stability analysis (0: off).
        stability_saddle_check: f64 = 1e-9, [expert, algorithm, unit = "Eh"];
        /// Davidson products per irrep of the check of the lowest orbital-Hessian eigenvalue
        /// before an open-shell Hessian: the stability probe of every converged point of its SCF
        /// runs with at least these (a negative mode is followed, as the probe's is; a degenerate
        /// solution, or a negative mode no step lowers, is flagged with a warning; 0: off, else at
        /// least 2).
        stability_hessian_check: usize = 6, [expert, algorithm, needs = Hessian];
        /// The size of the DIIS subspace.
        diis_space: usize = 10, [expert, algorithm];
        /// The first iteration whose Fock matrix enters DIIS.
        diis_start: usize = 1, [expert, algorithm];
        /// Which entry leaves a full DIIS subspace: `largest_error` or `oldest`.
        diis_drop: DiisDrop = DiisDrop::LargestError, [expert, algorithm];
        /// With `largest_error`: once a new entry's error exceeds this times the smallest, the
        /// subspace drops its oldest entries from then on (0: never).
        diis_unpin: f64 = 10.0, [dev, algorithm];
        /// ADIIS while max|[F, D]| is above `adiis_until`, then DIIS.
        adiis: bool = false, [expert, algorithm];
        /// The orbital gradient below which ADIIS hands over to DIIS.
        adiis_until: f64 = 1e-2, [expert, algorithm];
        /// The level shift of the virtual orbitals while max|F_ai| is above `level_shift_until`.
        level_shift: f64 = 0.25, [expert, algorithm, unit = "Eh"];
        /// The orbital gradient below which the level shift stops.
        level_shift_until: f64 = 1e-2, [expert, algorithm];
        /// Fock damping F = (1 - d) F + d F_previous while max|F_ai| is above `damping_until`
        /// (in [0, 1)).
        damping: f64 = 0.0, [expert, algorithm];
        /// The orbital gradient below which damping stops.
        damping_until: f64 = 0.0, [expert, algorithm];
        /// Quasi-Newton orbital rotations (SOSCF) instead of DIIS once max|F_ai| is below
        /// `soscf_from`.
        soscf: bool = false, [expert, algorithm];
        /// The orbital gradient from which SOSCF takes over.
        soscf_from: f64 = 3.3e-3, [expert, algorithm];
        /// When the energy cycles between two determinants, quasi-Newton steps replace DIIS
        /// (once per SCF).
        two_cycle: bool = true, [expert, algorithm];
        /// An SCF whose energy and orbital gradient stayed on a plateau over this many iterations
        /// when its escalation comes due, or at `max_iter`, is accepted with the warning
        /// `loosely_converged`; under accuracy targets only when the estimates of its final orbital
        /// gradient meet them (0: off).
        plateau_window: usize = 50, [expert, algorithm];
        /// An SCF whose DIIS stagnates (`stagnation_window`), or that has spent 2/3 of
        /// `max_iter`, restarts once from the lowest-energy orbitals it visited, with
        /// quasi-Newton steps, for the rest of `max_iter`.
        escalate: bool = true, [expert, algorithm];
        /// Iterations without a new lowest energy after which the escalation starts (0: only at
        /// 2/3 of `max_iter`).
        stagnation_window: usize = linus_scf::ScfAlgorithm::STAGNATION_WINDOW, [expert, algorithm];
        /// Canonical orthogonalization: overlap eigenvalues at or below this (absolute) drop their
        /// functions; fewer orbitals left than occupied ones is an input error (in (0, 1)).
        lindep: f64 = 1e-8, [expert, algorithm];
    }
}

impl Shorthand for ScfSettings {}

settings_group! {
    /// The response (CPHF/CPKS) equations of Hessians, IR intensities and polarizabilities.
    pub struct ResponseSettings at "response." {
        /// The max-abs residual of the Hessian's nuclear CPHF/CPKS (the Hessian's error is
        /// quadratic in it). `auto`: 1e-6, 1e-5 in two-level responses.
        tol: Auto<f64> = Auto::Auto, [common, convergence, needs = Hessian];
        /// The max-abs residual of the electric-field CPHF/CPKS of IR intensities and the
        /// polarizability.
        field_tol: f64 = 1e-6, [expert, convergence, needs = Ir];
        /// The auxiliary basis of a cheaper J/K for the CPKS iterations of large RI Hessians:
        /// `none`, `main` (the main auxiliary basis) or a basis set. `auto`: two levels for a
        /// large single-precision response.
        iteration_aux: IterationAux = IterationAux::Auto, [expert, approximation, needs = Hessian];
    }
}

impl Shorthand for ResponseSettings {}

settings_group! {
    /// Accuracy targets (qc*'s shared keyword `accuracy`, an object of targets; decision 22):
    /// the error from convergence the caller can tolerate, absolute, in atomic units. linus
    /// turns them into its SCF and response thresholds, tightening given ones but never
    /// loosening them (`crate::driver::accuracy`; docs/usage.md "Accuracy targets"), and
    /// records them under `record.accuracy`. Approximations (basis set, grids, density fitting)
    /// are not covered.
    pub struct AccuracySettings at "accuracy." {
        /// The energy's (Eh).
        energy: Option<f64> = None, [common, convergence, unit = "Eh"];
        /// The largest gradient component's.
        gradient: Option<f64> = None, [common, convergence, unit = "Eh/bohr", needs = Gradient];
        /// The largest Hessian element's.
        hessian: Option<f64> = None, [common, convergence, unit = "Eh/bohr^2", needs = Hessian];
    }
}

impl Shorthand for AccuracySettings {}

impl AccuracySettings {
    /// The targets (`crate::driver::accuracy::Targets`).
    pub fn targets(&self) -> crate::driver::accuracy::Targets {
        crate::driver::accuracy::Targets {
            energy: self.energy(),
            gradient: self.gradient(),
            hessian: self.hessian(),
        }
    }

    /// The group of `targets`.
    pub fn of(t: crate::driver::accuracy::Targets) -> AccuracySettings {
        AccuracySettings {
            energy: t.energy.map(Some),
            gradient: t.gradient.map(Some),
            hessian: t.hessian.map(Some),
        }
    }
}

settings_group! {
    /// Optional outputs and progress.
    pub struct OutputSettings at "output." {
        /// The SCF's progress lines: on stderr from the command line, in the output's `logs` in
        /// process (Python), which never prints.
        verbose: bool = false, [common, output];
        /// The MO coefficients in the result (`mo_coefficients_alpha`/`_beta`, nbf x nmo).
        orbitals: bool = false, [common, output];
        /// The Fock (Kohn-Sham) matrices of the SCF's last build in the result
        /// (`hamiltonian_alpha`/`_beta`, nbf x nbf), with the overlap: the operator of one full
        /// build at the last density, whose generalized eigenvectors are the returned orbitals
        /// (H C = S C diag(eps)).
        hamiltonian: bool = false, [expert, output];
        /// The overlap matrix S of the basis functions in the result (`overlap`, nbf x nbf).
        overlap: bool = false, [expert, output];
        /// The core Hamiltonian in the result (`core_hamiltonian`, nbf x nbf): the kinetic
        /// energy, the nuclei's attraction with their effective charges and the ECPs, whose
        /// trace with the total density is the energy's `one_electron` component.
        core_hamiltonian: bool = false, [expert, output];
        /// A Molden file of the orbitals in the result (`files["orbitals.molden"]`; the CLI's
        /// `--molden FILE` writes it): `spherical` or `cartesian` functions, or `off`.
        molden: MoldenForm = MoldenForm::Off, [common, output];
    }
}

impl Shorthand for OutputSettings {}

/// The Molden file of the orbitals (`output.molden`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MoldenForm {
    #[default]
    Off,
    /// Spherical d, f and g functions (`[5D7F]`, `[9G]`): a spherical basis set's own.
    Spherical,
    /// Cartesian functions: a Cartesian basis set's own, or a spherical one's expanded in its
    /// Cartesian components (the same orbitals).
    Cartesian,
}
setting_enum!(MoldenForm: "off", "spherical", "cartesian");

// ---------------------------------------------------------------------------------- frequencies

/// The harmonic analysis of a Hessian (`freq.analysis`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FreqAnalysis {
    /// Frequencies, normal modes, reduced masses, force constants, rotational constants and the
    /// point group (and IR intensities with `freq.ir`).
    #[default]
    Harmonic,
    /// The Hessian alone.
    Off,
}
setting_enum!(FreqAnalysis: "harmonic", "off");

/// The thermochemistry of a Hessian's harmonic analysis (`freq.thermo`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum Thermo {
    /// Rigid rotor, harmonic oscillator for every mode.
    #[default]
    #[serde(rename = "rrho")]
    Rrho,
    /// Grimme's quasi-RRHO entropy (Chem. Eur. J. 18, 9955 (2012)): low modes interpolated to
    /// free rotors around `freq.rotor_cutoff`.
    #[serde(rename = "qrrho")]
    Qrrho,
    /// Grimme's entropy and Li and Head-Gordon's quasi-RRHO energy and enthalpy (J. Phys.
    /// Chem. C 119, 1840 (2015)), with the same interpolation.
    #[serde(rename = "qrrho-hg")]
    QrrhoHg,
    /// Truhlar's quasiharmonic treatment (J. Phys. Chem. B 115, 14556 (2011)): real frequencies
    /// below `freq.raise_below` raised to it.
    #[serde(rename = "quasiharmonic")]
    Quasiharmonic,
    /// No thermochemistry.
    #[serde(rename = "off")]
    Off,
}
setting_enum!(Thermo: "rrho", "qrrho", "qrrho-hg", "quasiharmonic", "off");

/// One temperature or a list of them (`freq.temperature`).
#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(untagged)]
pub enum Temperatures {
    One(f64),
    List(Vec<f64>),
}

impl Temperatures {
    /// The temperatures, in input order.
    pub fn values(&self) -> Vec<f64> {
        match self {
            Temperatures::One(t) => vec![*t],
            Temperatures::List(v) => v.clone(),
        }
    }
}

impl<'de> Deserialize<'de> for Temperatures {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        match Value::deserialize(d)? {
            Value::Array(a) if !a.is_empty() => a
                .into_iter()
                .map(serde_json::from_value::<f64>)
                .collect::<Result<Vec<_>, _>>()
                .map(Temperatures::List)
                .map_err(D::Error::custom),
            v => serde_json::from_value::<f64>(v)
                .map(Temperatures::One)
                .map_err(|_| D::Error::custom("a temperature (K) or a non-empty list of them")),
        }
    }
}

impl SettingType for Temperatures {
    fn schema() -> Value {
        json!({"anyOf": [{"type": "number"}, {"type": "array", "items": {"type": "number"}, "minItems": 1}]})
    }
}

impl SettingType for Vec<f64> {
    fn schema() -> Value {
        json!({"type": "array", "items": {"type": "number"}})
    }
}

impl SettingType for Vec<(usize, f64)> {
    fn schema() -> Value {
        json!({"type": "array", "items": {"type": "array", "prefixItems": [{"type": "integer", "minimum": 0}, {"type": "number"}], "items": false}, "x-type": "[[atom, mass]]", "x-python": "list[tuple[int, float]]"})
    }
}

setting_enum!(linus_vib::MassTable: "isotope", "average");
setting_enum!(linus_vib::Projection: "translations_rotations", "translations", "none");
setting_enum!(linus_vib::Codata: "codata2022", "codata2014");

settings_group! {
    /// The harmonic analysis and thermochemistry a Hessian carries (and its IR intensities).
    pub struct FreqSettings at "freq." {
        /// The harmonic analysis: `harmonic` or `off` (the Hessian alone).
        analysis: FreqAnalysis = FreqAnalysis::Harmonic, [common, output, needs = Hessian];
        /// The thermochemistry: `rrho` (harmonic), `qrrho` (Grimme 2012's quasi-RRHO entropy),
        /// `qrrho-hg` (that and Li and Head-Gordon 2015's quasi-RRHO enthalpy), `quasiharmonic`
        /// (Truhlar 2011: real frequencies below `raise_below` raised to it) or `off`.
        thermo: Thermo = Thermo::Rrho, [common, model, needs = Hessian];
        /// Dipole derivatives and IR intensities (from the Hessian's CPHF solutions) and the
        /// static polarizability (an electric-field CPHF).
        ir: bool = false, [common, output, needs = Hessian];
        /// The temperature of the thermochemistry, or a list of them (one thermochemistry each,
        /// in order).
        temperature: Temperatures = Temperatures::One(298.15), [common, model, unit = "K", needs = Hessian];
        /// The pressure of the ideal-gas standard state.
        pressure: f64 = 101_325.0, [common, model, unit = "Pa", needs = Hessian];
        /// A concentration standard state instead of the pressure's (1 for 1 M).
        concentration: Option<f64> = None, [expert, model, unit = "mol/L", needs = Hessian];
        /// The factor the frequencies are scaled by in the thermochemistry.
        scale: f64 = 1.0, [common, model, needs = Hessian];
        /// The free-rotor interpolation's crossover of the quasi-RRHO models (`qrrho`, `qrrho-hg`).
        rotor_cutoff: f64 = 100.0, [expert, model, unit = "cm^-1", needs = Hessian];
        /// Quasiharmonic: real frequencies below this are raised to it.
        raise_below: f64 = 100.0, [expert, model, unit = "cm^-1", needs = Hessian];
        /// The exponent of Grimme's interpolation weight (quasi-RRHO models).
        qrrho_alpha: f64 = 4.0, [expert, model, needs = Hessian];
        /// Grimme's average moment of inertia B_av (quasi-RRHO models).
        qrrho_bav: f64 = 1e-44, [expert, model, unit = "kg m^2", needs = Hessian];
        /// Imaginary modes smaller than this are treated as real modes of |nu| in the
        /// thermochemistry (0: none).
        imaginary_as_real: f64 = 0.0, [expert, model, unit = "cm^-1", needs = Hessian];
        /// The electronic degeneracy. `auto`: the spin multiplicity.
        electronic_degeneracy: Auto<f64> = Auto::Auto, [expert, model, needs = Hessian];
        /// The table of the default masses: `isotope` (the most abundant isotope) or `average`
        /// (standard atomic weights).
        mass_table: linus_vib::MassTable = linus_vib::MassTable::Isotope, [expert, model, needs = Hessian];
        /// The masses of all atoms, replacing the table's.
        masses: Option<Vec<f64>> = None, [expert, model, unit = "u", needs = Hessian];
        /// Single atoms' masses, applied last: [[atom index, mass], ...] (deuterium 2.01410177812).
        isotopes: Vec<(usize, f64)> = Vec::new(), [expert, model, unit = "u", needs = Hessian];
        /// What is projected out of the mass-weighted Hessian: `translations_rotations`,
        /// `translations` or `none` (the translation- and rotation-like modes then flagged
        /// `external`).
        projection: linus_vib::Projection = linus_vib::Projection::TranslationsRotations, [expert, model, needs = Hessian];
        /// The rotational symmetry number. `auto`: from the detected point group.
        symmetry_number: Auto<u32> = Auto::Auto, [expert, model, needs = Hessian];
        /// The tolerance of the point-group detection and of the test for linear molecules.
        symmetry_tol: f64 = linus_vib::symmetry::DEFAULT_TOLERANCE, [expert, model, unit = "bohr", needs = Hessian];
        /// The physical constants: `codata2022` or `codata2014` (as Psi4 1.11 and PySCF 2.15).
        codata: linus_vib::Codata = linus_vib::Codata::Codata2022, [expert, model, needs = Hessian];
    }
}

impl Shorthand for FreqSettings {}

impl Shorthand for SolventSettings {
    fn expand(path: &str, v: Value) -> Result<Map<String, Value>, String> {
        match v {
            Value::String(s) => Ok(Map::from_iter([("name".to_string(), Value::String(s))])),
            Value::Number(n) => Ok(Map::from_iter([("eps".to_string(), Value::Number(n))])),
            v => Err(format!(
                "{path} must be a solvent name, a permittivity or an object, not {v}"
            )),
        }
    }

    fn schema() -> Option<Value> {
        Some(json!({"type": ["string", "number"], "x-type": "solvent name | permittivity"}))
    }
}

settings_group! {
    /// The settings of a calculation (module docs; docs/cli.md lists every key).
    pub struct CalcOptions at "" {
        /// `hf` or a functional (`--list-functionals`; libxc names, mixtures such as
        /// `"0.5*b3lyp + 0.5*hf"`). A dispersion suffix (`b3lyp-d3bj`, `-d3(bj)`, `-d3bj-atm`,
        /// `-d4`) sets `dispersion`, unless linus knows the whole name (`wb97x-d3bj`).
        method: String = "hf".to_string(), [common, model];
        /// The orbital basis set (`--list-basis`); the def2 sets carry ECPs for Rb-Rn.
        basis: String = "def2-svp".to_string(), [common, model];
        /// Empirical dispersion: `d3bj`, `d3bj-atm`, `d4` or `none`. `auto`: implied by the
        /// method's name, else none. Bare `d3` is ambiguous (zero damping in Psi4 and qcdata,
        /// Becke-Johnson in ORCA) and rejected.
        dispersion: DispersionSetting = DispersionSetting::Auto, [common, model];
        /// The determinant: `restricted` (synonyms `rhf`, `rks`) or `unrestricted` (`uhf`,
        /// `uks`). `auto`: restricted for closed shells, unrestricted for open shells (ROHF is
        /// not implemented).
        reference: Reference = Reference::Auto, [common, model];
        /// Cartesian d and higher functions (false: spherical).
        cartesian: bool = false, [expert, model];
        /// Directories of BSE JSON basis set files searched first (a string or a list), then
        /// `$LINUS_BASIS_DIR`, a development build's source tree and the installation's sets.
        basis_dir: BasisDirs = BasisDirs::default(), [expert, resource];
        /// Threads of the calculation's own pool.
        threads: usize = 1, [common, resource];
        /// The memory the calculation may use; every host budget derives from it. `auto`: the
        /// total RAM, or the cgroup limit if lower.
        memory_gib: Auto<f64> = Auto::Auto, [common, resource, unit = "GiB"];
        /// The device memory a GPU backend may use; every device budget derives from it. `auto`:
        /// `$LINUS_GPU_MEMORY_GIB`, else 95% of the memory free when the device opens (at most
        /// that either way).
        device_memory_gib: Auto<f64> = Auto::Auto, [expert, resource, unit = "GiB"];
    }
    groups {
        /// The J/K approximation.
        jk: JkSettings,
        /// The grids of DFT.
        grid: GridSettings,
        /// The arithmetic.
        precision: PrecisionSettings,
        /// The implicit solvent (absent: gas phase).
        solvent: Option<SolventSettings>,
        /// The SCF.
        scf: ScfSettings,
        /// The response equations of Hessians.
        response: ResponseSettings,
        /// The analysis and thermochemistry a Hessian carries.
        freq: FreqSettings,
        /// The hardware.
        backend: BackendSettings,
        /// Optional outputs.
        output: OutputSettings,
        /// Accuracy targets.
        accuracy: AccuracySettings,
    }
}

impl Shorthand for CalcOptions {}

/// Auxiliary basis of `jk.kind: df` when `jk.aux` is "auto" (the default), by purpose:
/// Weigend's def2-universal-jkfit (ORCA's def2/JK) for energies and gradients; for Hessians
/// cc-pVTZ-jkfit, with def2-universal-jkfit for the elements it lacks: its RI error in the
/// frequencies of the lactide benchmarks (B3LYP/def2-SVP, against exact J/K) is 0.18-0.26
/// cm^-1 instead of 0.61-0.78 cm^-1 (bench/results/2026-11-07-hess-accuracy).
pub const AUX_AUTO: &str = "auto";
pub const AUX_ENERGY_DEFAULT: &str = "def2-universal-jkfit";
pub const AUX_HESSIAN_DEFAULT: &str = "cc-pvtz-jkfit|def2-universal-jkfit";

/// Highest angular momentum of auxiliary basis functions (`linus_ints::eri3c::MAX_L_AUX`).
pub const MAX_L_AUX: usize = linus_ints::eri3c::MAX_L_AUX;
/// Highest orbital angular momentum of the seminumerical exchange (jk rij_snk; the Rys
/// potential-integral kernels, `linus_ints::points::grid_pot::RYS_LMAX`: g functions).
pub const MAX_L_COSX: usize = linus_ints::points::grid_pot::RYS_LMAX;

/// The roles a grid plays in a calculation; each resolves `auto` its own way
/// ([`CalcOptions::grid_for`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GridRole {
    /// The XC quadrature of the SCF, gradients and the explicit Hessian terms (`grid.xc`).
    Xc,
    /// The XC kernel of the CPKS equations of Hessians (`grid.kernel`).
    Response,
    /// VV10 nonlocal correlation of -V functionals (`grid.nlc`).
    Nlc,
    /// The seminumerical exchange of rij_snk (`jk.k_grid`).
    Exchange,
}

impl CalcOptions {
    /// The settings of a JSON object (keys and enum values normalized, shorthands expanded,
    /// unknown keys an `unknown_key` error), [`normalized`](CalcOptions::normalized).
    pub fn from_json(v: Value) -> Result<CalcOptions, LinusError> {
        let Value::Object(m) = v else {
            return Err(LinusError::Input(
                "the settings must be a JSON object".into(),
            ));
        };
        CalcOptions::from_map(m)?.normalized()
    }

    /// The dotted paths of the settings given (decision D41's checks read them).
    pub fn given(&self) -> Vec<String> {
        let mut out = Vec::new();
        Group::given(self, &mut out);
        out
    }

    /// Every setting, the unset ones at their defaults (`"auto"` where the calculation decides),
    /// as JSON (the base of `resolved.settings`).
    pub fn resolved_tree(&self) -> Value {
        Group::resolved(self)
    }

    /// The settings in canonical form: dispersion suffixes split off the method (unless
    /// linus-xc knows the whole name), the grids checked for their roles.
    pub fn normalized(mut self) -> Result<CalcOptions, LinusError> {
        if let Some(method) = self.method.clone() {
            let method = method.trim().to_string();
            let (base, suffix) = split_dispersion(&method);
            self.method = Some(base);
            if let Some(sfx) = suffix {
                let implied = DispersionSetting::parse(&sfx)?;
                match self.dispersion() {
                    DispersionSetting::Auto => self.dispersion = Some(implied),
                    d if d == implied => {}
                    d => {
                        let given = serde_json::to_value(d).unwrap_or(Value::Null);
                        return Err(LinusError::Input(format!(
                            "method {method:?} means dispersion {sfx:?}, but dispersion is {given}"
                        )));
                    }
                }
            }
        }
        let g = &self.grid;
        for (name, g) in [
            ("grid.xc", g.xc()),
            ("grid.kernel", g.kernel()),
            ("grid.nlc", g.nlc()),
        ] {
            g.validate()
                .map_err(|e| LinusError::Input(format!("{name}: {e}")))?;
            if g.cosx_levels().is_some() {
                return Err(LinusError::Input(format!(
                    "{name} {}: COSX grid schemes are for the seminumerical exchange (jk.k_grid of jk rij_snk), not for the XC terms; use auto, pyscf:N, linus:N, orca:N or psi4",
                    g.label()
                )));
            }
        }
        self.jk
            .k_grid()
            .validate()
            .map_err(|e| LinusError::Input(format!("jk.k_grid: {e}")))?;
        let a = &self.accuracy;
        for (path, v) in [
            ("accuracy.energy", a.energy()),
            ("accuracy.gradient", a.gradient()),
            ("accuracy.hessian", a.hessian()),
        ] {
            if let Some(v) = v.filter(|v| !(v.is_finite() && *v > 0.0)) {
                return Err(LinusError::Input(format!(
                    "{path} {v}: an accuracy target is a positive, finite number"
                )));
            }
        }
        Ok(self)
    }

    /// The J/K method the settings name, with its auxiliary basis and exchange grid as set
    /// (`jk.aux: auto` stays auto for df, whose default depends on the purpose; rij_snk's is
    /// [`AUX_RIJ_DEFAULT`]; `jk.k_grid: auto` is [`K_GRID_DEFAULT`]).
    pub fn jk_method(&self) -> JkMethod {
        let j = &self.jk;
        match j.kind() {
            JkKind::Exact => JkMethod::Exact,
            JkKind::Df => JkMethod::Df { aux: j.aux() },
            JkKind::RijSnk => JkMethod::RijSnk {
                aux: match j.aux() {
                    AuxBasis::Auto => AuxBasis::Named(AUX_RIJ_DEFAULT.into()),
                    a => a,
                },
                k_grid: match j.k_grid() {
                    GridChoice::Auto => K_GRID_DEFAULT,
                    g => g,
                },
            },
        }
    }

    /// The basis set search path: `basis_dir`, then the defaults
    /// ([`linus_ints::basis::search_path`]).
    pub fn basis_path(&self) -> Vec<PathBuf> {
        linus_ints::basis::search_path(&self.basis_dir().0)
    }

    /// Load a basis set (orbital or auxiliary) from the basis set search path
    /// ([`CalcOptions::basis_path`]). `"a|b|..."` is a composite: each element from the first
    /// set that has it ([`BasisDef::load_composite`]). An unknown name is an error with the
    /// closest names on the path.
    pub fn load_basis(&self, name: &str) -> Result<BasisDef, LinusError> {
        let path = self.basis_path();
        BasisDef::load_composite(name, &path).map_err(LinusError::from)
    }

    /// The grid of `role` in a calculation for `purpose`, every `auto` resolved (None: the role
    /// does not apply, e.g. [`GridRole::Exchange`] without rij_snk).
    pub fn grid_for(&self, role: GridRole, purpose: Purpose) -> Option<GridChoice> {
        let xc = self.grid.xc();
        match role {
            GridRole::Xc => Some(xc.for_purpose(purpose)),
            GridRole::Response => Some(kernel_grid_for(
                &self.grid.kernel(),
                &xc.for_purpose(Purpose::Hessian),
                false,
            )),
            GridRole::Nlc => Some(nlc_grid_for(&self.grid.nlc(), &xc)),
            GridRole::Exchange => self.jk_method().k_grid().cloned(),
        }
    }

    /// The CPKS kernel grid of a Hessian whose SCF ran on `scf_grid`, if it differs from it
    /// ([`kernel_grid_for`]; `meta_gga`: the functional is a meta-GGA).
    pub fn response_grid_for(&self, scf_grid: &GridChoice, meta_gga: bool) -> Option<GridChoice> {
        let xc = self.grid.xc().for_purpose(Purpose::Hessian);
        Some(kernel_grid_for(&self.grid.kernel(), &xc, meta_gga)).filter(|g| g != scf_grid)
    }

    /// The NLC grid of VV10 functionals ([`nlc_grid_for`]).
    pub fn nlc_grid_resolved(&self) -> GridChoice {
        nlc_grid_for(&self.grid.nlc(), &self.grid.xc())
    }

    /// The Hessian's CPHF/CPKS tolerance as set: `response.tol`, tightened by a Hessian accuracy
    /// target, which replaces the default (None: the default, `derivatives::hessian::cpks_tol`).
    pub fn response_tol(&self) -> Option<f64> {
        match (
            self.response.tol().value(),
            self.accuracy.targets().response_tol(),
        ) {
            (Some(given), Some(target)) => Some(given.min(target)),
            (given, target) => given.or(target),
        }
    }

    /// SCF thresholds for `purpose`: the preset named by `scf.convergence` (auto: the purpose's
    /// default), with `scf.e_tol` and `scf.g_tol` overriding its values, and the accuracy
    /// targets that apply to `purpose` replacing the default or tightening the given ones.
    pub fn convergence_for(&self, purpose: Purpose) -> Result<Convergence, LinusError> {
        let s = &self.scf;
        let mut c = match s.convergence() {
            Auto::Value(p) => p.thresholds(),
            Auto::Auto => default_convergence(purpose),
        };
        if let Auto::Value(e) = s.e_tol() {
            c.e_tol = e;
        }
        if let Auto::Value(g) = s.g_tol() {
            c.g_tol = g;
        }
        // accuracy targets: they replace the defaults and tighten the given thresholds
        if let Some((e, g)) = self.accuracy.targets().for_purpose(purpose).scf() {
            let preset = s.convergence() != Auto::Auto;
            let given_e = preset || s.e_tol() != Auto::Auto;
            let given_g = preset || s.g_tol() != Auto::Auto;
            c.e_tol = if given_e { c.e_tol.min(e) } else { e };
            c.g_tol = if given_g { c.g_tol.min(g) } else { g };
        }
        if !(c.e_tol > 0.0 && c.g_tol > 0.0) {
            return Err(LinusError::Input(format!(
                "SCF tolerances must be positive (scf.e_tol {}, scf.g_tol {})",
                c.e_tol, c.g_tol
            )));
        }
        Ok(c)
    }

    /// The SCF's algorithm settings ([`linus_scf::ScfAlgorithm`]) from `scf` and the J/K's
    /// incremental builds and grid switching (`jk`); `scf.stability: off` turns every probe
    /// and analysis off.
    pub fn scf_algorithm(&self) -> linus_scf::ScfAlgorithm {
        let s = &self.scf;
        let j = &self.jk;
        let off = s.stability() == Stability::Off;
        linus_scf::ScfAlgorithm {
            diis_space: s.diis_space(),
            diis_start: s.diis_start(),
            diis_drop: s.diis_drop(),
            diis_unpin: s.diis_unpin(),
            level_shift: s.level_shift(),
            level_shift_until: s.level_shift_until(),
            damping: s.damping(),
            damping_until: s.damping_until(),
            adiis: s.adiis(),
            adiis_switch: s.adiis_until(),
            incremental: j.incremental(),
            incremental_reset: j.incremental_reset(),
            soscf: s.soscf(),
            soscf_start: s.soscf_from(),
            grid_switch: j.grid_switch(),
            grid_switch_shift: j.grid_switch_shift(),
            saddle_check: if off { 0.0 } else { s.stability_saddle_check() },
            stability: s.stability() == Stability::Always,
            stability_max: s.stability_max(),
            stability_probe: if off {
                Some(0)
            } else {
                s.stability_probe().value()
            },
            closed_shell_probe: if off {
                Some(0)
            } else {
                s.stability_closed_probe().value()
            },
            closed_shell_gap: s.stability_closed_gap(),
            two_cycle: s.two_cycle(),
            plateau_window: s.plateau_window(),
            escalate: s.escalate(),
            stagnation_window: s.stagnation_window(),
            lindep: s.lindep(),
        }
    }

    /// What a Hessian's analysis computes (`freq`): the dipole derivatives (`freq.ir`), the
    /// harmonic analysis's settings (None: `freq.analysis: off`) and one thermochemistry per
    /// temperature (none with `freq.thermo: off` or without the analysis).
    pub fn freq_options(&self) -> linus_vib::FrequencyOptions {
        let f = &self.freq;
        let analysis = (f.analysis() == FreqAnalysis::Harmonic).then(|| linus_vib::VibOptions {
            mass_table: f.mass_table(),
            masses: f.masses(),
            isotopes: f.isotopes(),
            projection: f.projection(),
            symmetry_number: f.symmetry_number().value(),
            symmetry_tolerance: f.symmetry_tol(),
            codata: f.codata(),
        });
        let thermo = f.thermo();
        let thermo = match (&analysis, thermo) {
            (None, _) | (_, Thermo::Off) => Vec::new(),
            _ => f
                .temperature()
                .values()
                .into_iter()
                .map(|temperature| linus_vib::ThermoOptions {
                    temperature,
                    pressure: f.pressure(),
                    concentration: f.concentration(),
                    qrrho_entropy: matches!(thermo, Thermo::Qrrho | Thermo::QrrhoHg),
                    qrrho_enthalpy: thermo == Thermo::QrrhoHg,
                    qrrho_frequency: f.rotor_cutoff(),
                    qrrho_alpha: f.qrrho_alpha(),
                    qrrho_bav: f.qrrho_bav(),
                    raise_low_frequencies: (thermo == Thermo::Quasiharmonic)
                        .then(|| f.raise_below()),
                    imaginary_as_real: f.imaginary_as_real(),
                    frequency_scale: f.scale(),
                    electronic_degeneracy: f.electronic_degeneracy().value(),
                })
                .collect(),
        };
        linus_vib::FrequencyOptions {
            ir: f.ir(),
            analysis,
            thermo,
        }
    }

    /// The memory limit the budgets derive from.
    pub fn memory(&self) -> MemoryLimit {
        MemoryLimit::from_gib(self.memory_gib().value())
    }

    /// Reject settings the calculation cannot honour, instead of silently computing something
    /// else (before any expensive work).
    pub fn check(&self, nalpha: usize, nbeta: usize) -> Result<(), LinusError> {
        self.convergence_for(Purpose::Energy)?;
        self.backend.check()?;
        self.precision.check(self.backend.kind())?;
        if let Auto::Value(g) = self.device_memory_gib() {
            if !(g > 0.0 && g.is_finite()) {
                return Err(LinusError::Input(format!(
                    "device_memory_gib must be a positive number of GiB, not {g}"
                )));
            }
        }
        if self.threads() == 0 {
            return Err(LinusError::Input("threads must be at least 1".into()));
        }
        if self.reference() == Reference::Restricted && nalpha != nbeta {
            return Err(LinusError::Unsupported(format!(
                "restricted open-shell SCF (ROHF/ROKS) is not implemented ({nalpha} alpha, {nbeta} beta electrons); \
                 leave `reference` auto or unrestricted for UHF/UKS"
            )));
        }
        self.check_scf()?;
        let screening = self.jk.screening();
        if !(screening > 0.0 && screening < 1.0) {
            return Err(LinusError::Input(format!(
                "jk.screening must be in (0, 1), not {screening}"
            )));
        }
        if let Auto::Value(m) = self.memory_gib() {
            if !(m > 0.0 && m.is_finite()) {
                return Err(LinusError::Input(format!(
                    "memory_gib must be a positive number of GiB, not {m}"
                )));
            }
        }
        let r = &self.response;
        for (name, v) in [
            ("response.tol", r.tol().value().unwrap_or(1e-6)),
            ("response.field_tol", r.field_tol()),
        ] {
            if !(v > 0.0 && v < 1.0) {
                return Err(LinusError::Input(format!(
                    "{name} must be in (0, 1), not {v}"
                )));
            }
        }
        if let IterationAux::Named(name) = r.iteration_aux() {
            self.load_basis(&name)
                .map_err(|e| LinusError::Input(format!("response.iteration_aux: {e}")))?;
        }
        if self.scf.stability_hessian_check() == 1 {
            return Err(LinusError::Input(
                "scf.stability_hessian_check must be 0 (off) or at least 2 Hessian-vector products, not 1"
                    .into(),
            ));
        }
        crate::solvent::check(&self.solvent)?;
        Ok(())
    }

    /// The SCF settings in range: iterations, linear-dependence threshold, and the algorithm's
    /// sizes, shifts and switches (values that would silently disable or break a step).
    fn check_scf(&self) -> Result<(), LinusError> {
        let s = &self.scf;
        let j = &self.jk;
        let bad = |what: &str, v: String| Err(LinusError::Input(format!("{what}, not {v}")));
        if s.max_iter() == 0 {
            return bad("scf.max_iter must be at least 1", "0".into());
        }
        if s.diis_space() == 0 {
            return bad("scf.diis_space must be at least 1", "0".into());
        }
        if s.diis_start() == 0 {
            return bad(
                "scf.diis_start must be at least 1 (the first iteration)",
                "0".into(),
            );
        }
        for (name, v) in [
            ("scf.level_shift", s.level_shift()),
            ("scf.level_shift_until", s.level_shift_until()),
            ("scf.damping_until", s.damping_until()),
            ("scf.adiis_until", s.adiis_until()),
            ("scf.soscf_from", s.soscf_from()),
            ("jk.grid_switch", j.grid_switch()),
            ("scf.stability_saddle_check", s.stability_saddle_check()),
            ("scf.diis_unpin", s.diis_unpin()),
            ("scf.stability_closed_gap", s.stability_closed_gap()),
        ] {
            if !(v >= 0.0 && v.is_finite()) {
                return bad(
                    &format!("{name} must be a finite number >= 0"),
                    v.to_string(),
                );
            }
        }
        let d = s.damping();
        if !(0.0..1.0).contains(&d) {
            return bad("scf.damping must be in [0, 1)", d.to_string());
        }
        let l = s.lindep();
        if !(l > 0.0 && l < 1.0) {
            return bad("scf.lindep must be in (0, 1)", l.to_string());
        }
        if j.incremental_reset() == 0 {
            return bad("jk.incremental_reset must be at least 1", "0".into());
        }
        for (name, p) in [
            ("scf.stability_probe", s.stability_probe()),
            ("scf.stability_closed_probe", s.stability_closed_probe()),
        ] {
            if p == Auto::Value(1) {
                return bad(
                    &format!("{name} must be 0 (off) or at least 2 Hessian-vector products"),
                    "1".into(),
                );
            }
        }
        Ok(())
    }
}

/// The settings a group of the JSON form of `v` gives, by their dotted paths, with their
/// values (`scf.level_shift`), the shorthands expanded: the request's view of the given
/// settings for the front ends (the CLI's `--set` refinement).
pub fn expand_group_shorthand(path: &str, v: &Value) -> Result<Map<String, Value>, LinusError> {
    let err = LinusError::Input;
    match (path, v) {
        (_, Value::Object(o)) => Ok(o.clone()),
        (_, Value::Null) => Ok(Map::new()),
        ("jk", v) => JkSettings::expand(path, v.clone()).map_err(err),
        ("grid", v) => GridSettings::expand(path, v.clone()).map_err(err),
        ("solvent", v) => SolventSettings::expand(path, v.clone()).map_err(err),
        ("backend", v) => BackendSettings::expand(path, v.clone()).map_err(err),
        ("precision", v) => PrecisionSettings::expand(path, v.clone()).map_err(err),
        _ => Err(LinusError::Input(format!(
            "{path} is {v}, which has no settings below it"
        ))),
    }
}

// ---------------------------------------------------------------------------------- debugging

/// `LINUS_*` environment variables read by the library (development and study switches; the
/// documented settings are all in [`CalcOptions`]), reported in a result's `resolved.record.debug_env`
/// when set.
pub const DEBUG_ENV: [&str; 8] = [
    "LINUS_BASIS_DIR",
    "LINUS_SOLVENT_TABLE",
    "LINUS_ISA",
    "LINUS_KERNEL",
    "LINUS_NO_GEN",
    "LINUS_ECP_LEVEL",
    "LINUS_COSX_LOG",
    "LINUS_SCF_TRACE",
];

/// The `LINUS_*` variables that are set (every one, not only [`DEBUG_ENV`]: a variable this
/// version no longer reads is worth seeing too).
pub fn debug_env() -> std::collections::BTreeMap<String, String> {
    std::env::vars()
        .filter(|(k, _)| k.starts_with("LINUS_"))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grid_grammar() {
        let p = |s: &str| s.parse::<GridChoice>();
        assert_eq!(p("3").unwrap(), GridChoice::Pyscf { level: 3 });
        assert_eq!(p("pyscf:4").unwrap(), GridChoice::Pyscf { level: 4 });
        assert_eq!(p("PySCF_4").unwrap(), GridChoice::Pyscf { level: 4 });
        assert_eq!(p("linus2").unwrap(), GridChoice::Linus { level: 2 });
        assert_eq!(p("linus-2").unwrap(), GridChoice::Linus { level: 2 });
        assert_eq!(p("auto").unwrap(), GridChoice::Auto);
        assert_eq!(p("DefGrid3").unwrap(), GridChoice::Orca { defgrid: 3 });
        assert_eq!(p("orca:1").unwrap(), GridChoice::Orca { defgrid: 1 });
        assert_eq!(p(" Psi4 ").unwrap(), GridChoice::Psi4);
        assert_eq!(p("cosx:2").unwrap(), GridChoice::Cosx { defgrid: 2 });
        assert_eq!(
            p("gridx:1,2,3").unwrap(),
            GridChoice::CosxLevels { levels: [1, 2, 3] }
        );
        assert_eq!(
            p("gridx2").unwrap(),
            GridChoice::CosxLevels { levels: [2, 2, 2] }
        );
        assert!(p("fine").is_err());
        // levels are checked, not clamped
        assert!(p("linus:9").unwrap_err().contains("1-4"));
        assert!(p("orca:4").is_err());
        assert!(p("10").is_err());
        assert!(p("gridx:1,2,5").is_err());
        // every form deserializes; the serialized form is the string (a spec its object)
        for v in [
            json!("linus:2"),
            json!({"kind": "linus", "level": 2}),
            json!({"kind": "LINUS", "level": 2}),
        ] {
            assert_eq!(
                serde_json::from_value::<GridChoice>(v).unwrap(),
                GridChoice::Linus { level: 2 }
            );
        }
        assert_eq!(
            serde_json::from_value::<GridChoice>(json!(4)).unwrap(),
            GridChoice::Pyscf { level: 4 }
        );
        assert!(
            serde_json::from_value::<GridChoice>(json!({"kind": "linus", "level": 7})).is_err()
        );
        assert!(
            serde_json::from_value::<GridChoice>(json!({"kind": "linus", "levle": 2})).is_err()
        );
        assert_eq!(
            serde_json::to_value(GridChoice::Linus { level: 2 }).unwrap(),
            json!("linus:2")
        );
        let spec = GridChoice::Spec(Box::new(GridSpec::linus(2)));
        let v = serde_json::to_value(&spec).unwrap();
        assert_eq!(v["kind"], "spec");
        assert_eq!(serde_json::from_value::<GridChoice>(v).unwrap(), spec);
    }

    #[test]
    fn jk_grammar() {
        let jk = |v: Value| {
            CalcOptions::from_json(json!({"jk": v}))
                .unwrap()
                .jk_method()
        };
        assert_eq!(
            jk(json!("df")),
            JkMethod::Df {
                aux: AuxBasis::Auto
            }
        );
        assert_eq!(
            jk(json!("RI-JK")),
            JkMethod::Df {
                aux: AuxBasis::Auto
            }
        );
        let snk = JkMethod::RijSnk {
            aux: AUX_RIJ_DEFAULT.into(),
            k_grid: K_GRID_DEFAULT,
        };
        for s in [
            "rij_snk", "rij-snk", "rijcosx", "RIJCOSX", "rij_cosx", "rij-cosx",
        ] {
            assert_eq!(jk(json!(s)), snk, "{s}");
        }
        assert!(CalcOptions::from_json(json!({"jk": "cd"})).is_err());
        assert_eq!(jk(json!({"kind": "rij_snk"})), snk);
        assert_eq!(
            jk(json!({"kind": "rij-cosx", "k_grid": "cosx:3"})),
            JkMethod::RijSnk {
                aux: AUX_RIJ_DEFAULT.into(),
                k_grid: GridChoice::Cosx { defgrid: 3 }
            }
        );
        // incore: exact with the integrals in memory
        for v in [json!("incore"), json!("In-Core")] {
            let o = CalcOptions::from_json(json!({"jk": v})).unwrap();
            assert_eq!(
                (o.jk_method(), o.jk.exact_storage()),
                (JkMethod::Exact, ExactStorage::Incore)
            );
        }
        // the old top-level keys are unknown
        assert!(CalcOptions::from_json(json!({"aux": "x"})).is_err());
        assert!(CalcOptions::from_json(json!({"k_grid": "cosx:2"})).is_err());
    }

    #[test]
    fn dispersion_suffixes() {
        assert_eq!(
            split_dispersion("b3lyp-d3bj"),
            ("b3lyp".into(), Some("d3bj".into()))
        );
        assert_eq!(
            split_dispersion("B3LYP-D3(BJ)"),
            ("B3LYP".into(), Some("d3bj".into()))
        );
        assert_eq!(
            split_dispersion("pbe0-d3bj-atm"),
            ("pbe0".into(), Some("d3bj-atm".into()))
        );
        assert_eq!(
            split_dispersion("r2scan-d4"),
            ("r2scan".into(), Some("d4".into()))
        );
        let e = CalcOptions::from_json(json!({"method": "b3lyp-d3"})).unwrap_err();
        assert!(
            matches!(e, LinusError::Input(_)) && e.to_string() == D3_AMBIGUOUS,
            "{e}"
        );
        // names linus knows whole stay whole
        assert_eq!(split_dispersion("wb97x-d3bj"), ("wb97x-d3bj".into(), None));
        assert_eq!(split_dispersion("pbe"), ("pbe".into(), None));
        let o = CalcOptions::from_json(json!({"method": "b3lyp-d3bj"})).unwrap();
        assert_eq!(
            (o.method().as_str(), o.dispersion()),
            ("b3lyp", DispersionSetting::Model(Dispersion::D3bj))
        );
        let o = CalcOptions::from_json(json!({"method": "b3lyp-d3(bj)", "dispersion": "D3BJ"}))
            .unwrap();
        assert_eq!(o.dispersion(), DispersionSetting::Model(Dispersion::D3bj));
        assert!(
            CalcOptions::from_json(json!({"method": "b3lyp-d3bj", "dispersion": "d4"})).is_err()
        );
        assert!(
            CalcOptions::from_json(json!({"method": "b3lyp-d4", "dispersion": "none"})).is_err()
        );
        // auto is the default and a value; unsupported damping keeps its kind
        assert_eq!(
            CalcOptions::default().resolved_tree()["dispersion"],
            json!("auto")
        );
        let o =
            CalcOptions::from_json(json!({"method": "wb97x-d3bj", "dispersion": "none"})).unwrap();
        assert_eq!(o.dispersion(), DispersionSetting::Off);
        assert!(matches!(
            CalcOptions::from_json(json!({"method": "b3lyp-d3zero"})),
            Err(LinusError::Unsupported(_))
        ));
        assert!(matches!(
            CalcOptions::from_json(json!({"dispersion": "d3(0)"})),
            Err(LinusError::Unsupported(_))
        ));
        assert_eq!(parse_dispersion("D3(BJ)").unwrap(), Some(Dispersion::D3bj));
        assert_eq!(parse_dispersion("None").unwrap(), None);
    }

    /// The groups, the shorthands, presence, and the echo of what was given.
    #[test]
    fn groups_shorthands_and_presence() {
        let o = CalcOptions::from_json(json!({
            "method": "pbe", "jk": {"kind": "rijcosx", "aux": "def2-universal-jkfit", "k_grid": "gridx:2"},
            "grid": 2, "solvent": "water", "memory_gib": 8,
            "scf": {"convergence": "Very_Tight", "e_tol": 1e-9, "max_iter": 64, "guess": "core", "soscf": true},
            "response": {"iteration_aux": "MAIN", "tol": 1e-9},
            "basis_dir": "/somewhere", "backend": "CPU"}))
        .unwrap();
        assert_eq!(
            o.jk_method(),
            JkMethod::RijSnk {
                aux: "def2-universal-jkfit".into(),
                k_grid: GridChoice::CosxLevels { levels: [2, 2, 2] }
            }
        );
        assert_eq!(o.grid.xc(), GridChoice::Pyscf { level: 2 });
        assert_eq!(o.solvent.as_ref().unwrap().name(), Some("water".into()));
        assert_eq!(o.memory().limit, Some(8usize << 30));
        assert_eq!(
            o.scf.convergence(),
            Auto::Value(ConvergencePreset::VeryTight)
        );
        assert_eq!(
            (o.scf.e_tol(), o.scf.max_iter(), o.scf.guess()),
            (Auto::Value(1e-9), 64, Guess::Core)
        );
        assert!(o.scf.soscf());
        assert_eq!(o.response.iteration_aux(), IterationAux::Main);
        assert_eq!(o.basis_dir().0, vec![PathBuf::from("/somewhere")]);
        assert_eq!(o.backend.kind(), BackendKind::Cpu);
        // what was given, in canonical form, and nothing else
        let mut given = o.given();
        given.sort();
        assert!(
            given.contains(&"scf.e_tol".to_string()) && !given.contains(&"scf.g_tol".to_string())
        );
        let v = serde_json::to_value(&o).unwrap();
        assert_eq!(v["scf"]["convergence"], "verytight");
        assert_eq!(v["grid"], json!({"xc": "pyscf:2"}));
        assert_eq!(v["backend"], json!({"kind": "cpu"}));
        assert!(v.get("threads").is_none() && v.get("output").is_none());
        // the echo reads back unchanged
        assert_eq!(CalcOptions::from_json(v).unwrap(), o);
        // the tree with defaults
        let t = CalcOptions::default().resolved_tree();
        assert_eq!(t["scf"]["max_iter"], 288);
        assert_eq!(t["grid"]["xc"], "auto");
        assert!(t.get("solvent").is_none());
        // a solvent permittivity, unknown keys, bad values
        assert_eq!(
            CalcOptions::from_json(json!({"solvent": 78.4}))
                .unwrap()
                .solvent
                .unwrap()
                .eps(),
            Some(78.4)
        );
        assert!(CalcOptions::from_json(json!({"pcm": "cpcm"})).is_err());
        assert!(CalcOptions::from_json(json!({"metod": "pbe"}))
            .unwrap_err()
            .to_string()
            .contains("metod"));
        assert!(CalcOptions::from_json(json!({"grid": "cosx:2"}))
            .unwrap_err()
            .to_string()
            .contains("COSX"));
        let e = CalcOptions::from_json(json!({"jk": {"df_storage": "f16"}})).unwrap_err();
        assert!(e.to_string().contains("jk.df_storage"), "{e}");
        assert!(
            CalcOptions::from_json(json!({"scf": {"convergence": "tightest"}}))
                .unwrap_err()
                .to_string()
                .contains("verytight")
        );
        assert!(CalcOptions::from_json(json!({"scf": {"guess": "huckel"}})).is_err());
        // reference and its synonyms
        for (s, r) in [
            ("UHF", Reference::Unrestricted),
            ("rks", Reference::Restricted),
        ] {
            assert_eq!(
                CalcOptions::from_json(json!({"reference": s}))
                    .unwrap()
                    .reference(),
                r
            );
        }
    }

    #[test]
    fn grids_per_role() {
        let o = CalcOptions::from_json(json!({"jk": "rij_snk", "jk.k_grid": null}));
        assert!(o.is_err(), "a dotted key is not a key");
        let o =
            CalcOptions::from_json(json!({"jk": {"kind": "rij_snk", "k_grid": "auto"}})).unwrap();
        assert_eq!(
            o.grid_for(GridRole::Exchange, Purpose::Energy),
            Some(GridChoice::Cosx { defgrid: 2 })
        );
        let o = CalcOptions::default();
        assert_eq!(
            o.grid_for(GridRole::Xc, Purpose::Hessian),
            Some(GridChoice::HESSIAN_DEFAULT)
        );
        assert_eq!(
            o.grid_for(GridRole::Response, Purpose::Hessian),
            Some(GridChoice::CPKS_DEFAULT)
        );
        assert_eq!(
            o.grid_for(GridRole::Nlc, Purpose::Energy),
            Some(NLC_DEFAULT)
        );
        assert_eq!(o.grid_for(GridRole::Exchange, Purpose::Energy), None);
        let o = CalcOptions::from_json(json!({"grid": "pyscf:4"})).unwrap();
        assert_eq!(
            o.response_grid_for(&GridChoice::Pyscf { level: 4 }, true),
            None
        );
        let d = CalcOptions::default();
        assert_eq!(
            d.response_grid_for(&GridChoice::HESSIAN_DEFAULT, true),
            Some(GridChoice::CPKS_META_GGA)
        );
        assert_eq!(
            d.response_grid_for(&GridChoice::HESSIAN_DEFAULT, false),
            Some(GridChoice::CPKS_DEFAULT)
        );
        // the family rule (F5): the value auto resolves to, given, resolves the kernel as auto does
        let o = CalcOptions::from_json(json!({"grid": "linus:3"})).unwrap();
        assert_eq!(
            o.response_grid_for(&GridChoice::HESSIAN_DEFAULT, false),
            Some(GridChoice::CPKS_DEFAULT)
        );
        let o = CalcOptions::from_json(json!({"grid": "linus:4"})).unwrap();
        assert_eq!(
            o.response_grid_for(&GridChoice::Linus { level: 4 }, true),
            Some(GridChoice::CPKS_META_GGA)
        );
        for g in ["orca:2", "psi4", "5"] {
            let o = CalcOptions::from_json(json!({ "grid": g })).unwrap();
            let xc = o.grid_for(GridRole::Xc, Purpose::Hessian).unwrap();
            assert_eq!(o.response_grid_for(&xc, false), None, "{g}");
            assert_eq!(o.grid_for(GridRole::Response, Purpose::Hessian), Some(xc));
        }
        let o = CalcOptions::from_json(json!({"grid": {"xc": "linus:4", "kernel": "linus:4"}}))
            .unwrap();
        assert_eq!(
            o.response_grid_for(&GridChoice::Linus { level: 4 }, false),
            None
        );
    }

    /// The backend: gpu needs the gpu feature; FP64 emulation and the opt-in FP32 products on
    /// the CPU, several devices and CPU devices are refused; auto runs anywhere.
    #[test]
    fn backend_checks() {
        assert!(BackendSettings::default().check().is_ok());
        for (b, unsupported) in [
            (json!({"backend": {"kind": "gpu", "devices": [0, 1]}}), true),
            (json!({"backend": {"kind": "auto", "devices": []}}), false),
            (json!({"precision": "mixed"}), true),
            (json!({"precision": {"fp64": "emulated"}}), true),
            (json!({"backend": {"kind": "cpu", "devices": 0}}), false),
            (json!({"device_memory_gib": -1.0}), false),
        ] {
            let o = CalcOptions::from_json(b.clone()).unwrap();
            let e = o.check(5, 5).unwrap_err();
            assert_eq!(
                matches!(e, LinusError::Unsupported(_)),
                unsupported,
                "{b}: {e}"
            );
        }
        let gpu = CalcOptions::from_json(json!({"backend": "GPU"})).unwrap();
        assert_eq!(gpu.backend.check().is_ok(), crate::backend::GPU_BUILT);
        let o =
            CalcOptions::from_json(json!({"backend": {"kind": "Auto", "devices": [1]}})).unwrap();
        assert_eq!(o.backend.device_id(), 1);
        // the precision shorthand and its synonyms
        let p = |v: Value| {
            CalcOptions::from_json(json!({"precision": v}))
                .unwrap()
                .precision
        };
        let fp64 = p(json!("FP64"));
        assert_eq!(
            (fp64.fp64(), fp64.fp32()),
            (Fp64Mode::Native, Fp32Mode::Off)
        );
        assert_eq!(p(json!("double")), fp64);
        assert_eq!(p(json!("emulated")).fp64(), Fp64Mode::Emulated);
        let m = p(json!("mixed"));
        assert_eq!((m.fp64(), m.fp32()), (Fp64Mode::Auto, Fp32Mode::On));
        assert!(p(json!("auto")).is_unset());
        assert_eq!(
            p(json!({"response": "single"})).response(),
            ResponsePrecision::Fp32
        );
        assert_eq!(p(json!({"fp64": "double"})).fp64(), Fp64Mode::Native);
        // fp32 off makes auto's response FP64; an explicit response stays
        assert_eq!(
            p(json!({"fp32": "off"})).response_effective(),
            ResponsePrecision::Fp64
        );
        let r = p(json!({"fp32": "off", "response": "fp32"}));
        assert_eq!(r.response_effective(), ResponsePrecision::Fp32);
        assert!(CalcOptions::from_json(json!({"precision": "half"})).is_err());
        // the kernel cache
        let k = |v: Value| {
            CalcOptions::from_json(json!({"backend": {"kernel_cache": v}}))
                .map(|o| o.backend.kernel_cache())
        };
        assert_eq!(k(json!("OFF")).unwrap(), KernelCache::Off);
        assert_eq!(
            k(json!("/tmp/k")).unwrap(),
            KernelCache::Dir("/tmp/k".into())
        );
        assert!(k(json!(3)).is_err());
        let o = CalcOptions::from_json(json!({"backend": {"scf_loop": "device"}})).unwrap();
        assert!(o.backend.check().unwrap_err().is_input());
        assert!(CalcOptions::from_json(json!({"backend": "tpu"})).is_err());
        assert!(CalcOptions::from_json(json!({"backend": {"kind": "cpu", "cores": 4}})).is_err());
        assert!(CalcOptions::from_json(json!({"basis_dir": 3})).is_err());
        // values out of range are refused before any work
        for bad in [
            json!({"threads": 0}),
            json!({"scf": {"max_iter": 0}}),
            json!({"scf": {"damping": 1.0}}),
            json!({"scf": {"level_shift": -0.5}}),
            json!({"scf": {"diis_start": 0}}),
            json!({"scf": {"lindep": 1.0}}),
            json!({"jk": {"screening": 0.0}}),
            json!({"scf": {"stability_probe": 1}}),
        ] {
            let o = CalcOptions::from_json(bad.clone()).unwrap();
            assert!(o.check(5, 5).unwrap_err().is_input(), "{bad}");
        }
    }

    /// `scf.stability: off` turns every probe and analysis off; `always` analyses every
    /// determinant.
    #[test]
    fn stability_modes() {
        let a = |s: &str| {
            CalcOptions::from_json(json!({"scf": {"stability": s}}))
                .unwrap()
                .scf_algorithm()
        };
        let off = a("off");
        assert_eq!(
            (
                off.stability_probe,
                off.closed_shell_probe,
                off.saddle_check,
                off.stability
            ),
            (Some(0), Some(0), 0.0, false)
        );
        let auto = a("auto");
        assert_eq!(
            (
                auto.stability_probe,
                auto.closed_shell_probe,
                auto.stability
            ),
            (None, None, false)
        );
        assert_eq!(auto, CalcOptions::default().scf_algorithm());
        assert!(a("always").stability);
        assert_eq!(
            CalcOptions::default().scf_algorithm(),
            linus_scf::ScfAlgorithm::default(),
            "the settings' defaults are the algorithm's"
        );
    }
}
