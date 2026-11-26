//! The result of a calculation as a typed value, built once here for every front end: the CLI
//! prints a [`Report`] as JSON, the Python binding returns it as a dict (with numpy arrays for
//! the matrices), from which the Python package makes its qcdata frames. A failed calculation is
//! a [`Failure`] (the CLI's failure JSON; the Python exception's `failure`).
//!
//! The JSON keys (docs/cli.md has the full schema; objects come out with sorted keys):
//!
//! - `schema_version` ([`SCHEMA_VERSION`]), `program`, `version`, `calctype`, `success`;
//! - `energy` and its `components` (they sum to `energy`); `gradient` and `gradient_parts`;
//!   `hessian` and `cphf_iterations`; `vibrational_analysis`, `thermochemistry`,
//!   `dipole_derivatives`, `polarizability` (frequencies); `solvation`;
//! - `converged`, `iterations` and the `scf` block (`fock_builds`, `incremental_builds`, the
//!   `convergence` thresholds, the per-iteration `history`, the `stability` analyses);
//! - `restricted`, `natoms`, `nbf`, `nmo`, `nalpha`, `nbeta`, `s2`, orbital energies and
//!   occupations of both spins, `homo_lumo_gap`, `dipole`, `grid` (DFT), `vv10`;
//! - `structure` (symbols, geometry in bohr, charge, multiplicity, ECP core electrons per atom),
//!   `method`, `basis`;
//! - `warnings` (`{code, message, path}`: what the user should know about a calculation that
//!   succeeded);
//! - `timings` (seconds only; `total` is the whole call, `energy` its SCF stage), `resources`
//!   (sizes, counts and peak memory), `options` (every setting, normalized) and `resolved`
//!   ([`Resolved`]: the calctypes, the settings that reproduce the run, why each `auto` took its
//!   value, and the run's record).
//!
//! The SCF's orbitals and Fock matrices ([`Report::wavefunction`]) and the AO matrices a request
//! asks for ([`Report::ao`]: the overlap and the core Hamiltonian) are not part of the serialized
//! report; [`Report::to_json`] and the Python binding add them on request. Asking for them
//! changes no other number: the AO matrices are computed after the calculation, at its geometry,
//! outside every cache.

use crate::derivatives::capabilities::Request;
use crate::derivatives::hessian::ResolvedCphf;
use crate::driver::ground::GroundState;
use crate::driver::molden::{self, MOLDEN_FILE};
use crate::driver::options::MoldenForm;
use crate::driver::settings::{CalcTypes, RunFacts};
use crate::driver::UnconvergedResponse;
use crate::driver::{CalcContext, CalcOptions, GridChoice, Purpose, Session};
use faer::Mat;
use linus_base::convergence::Convergence;
use linus_base::error::LinusError;
use linus_base::error::{ResponseEnd, ResponseFailure};
use linus_core::timing::Timers;
use linus_jk::df::ResolvedDf;
use linus_scf::{
    ClosedShellProbe, Escalation, Plateau, ProbeSkip, ScfIteration, ScfResult, StabilityCheck,
    TriggerRecord,
};
use linus_solvent::SolvationResult;
use linus_vib::{Thermochemistry, VibrationalAnalysis};
use serde::Serialize;
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::sync::Arc;

/// Version of the result and failure JSON (`schema_version`), raised on every breaking change
/// to its keys or their meaning (CHANGELOG.md lists them).
pub const SCHEMA_VERSION: u32 = 1;

/// What a calculation computes: an energy, a gradient, or a Hessian with the analysis the
/// `freq` settings ask for (decision D42: one Hessian calctype carries the vibrational analysis).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CalcType {
    Energy,
    Gradient,
    /// The Hessian, with the harmonic analysis, thermochemistry and IR intensities its `freq`
    /// settings ask for.
    Hessian,
}

impl CalcType {
    pub fn name(&self) -> &'static str {
        match self {
            CalcType::Energy => "energy",
            CalcType::Gradient => "gradient",
            CalcType::Hessian => "hessian",
        }
    }

    /// The purpose its SCF is converged for.
    pub fn purpose(&self) -> Purpose {
        match self {
            CalcType::Energy => Purpose::Energy,
            CalcType::Gradient => Purpose::Gradient,
            CalcType::Hessian => Purpose::Hessian,
        }
    }
}

impl Serialize for CalcType {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(self.name())
    }
}

impl std::str::FromStr for CalcType {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, String> {
        Ok(match s.trim().to_ascii_lowercase().as_str() {
            "energy" => CalcType::Energy,
            "gradient" => CalcType::Gradient,
            "hessian" => CalcType::Hessian,
            _ => {
                return Err(format!(
                    "unknown calctype {s:?}; use energy, gradient or hessian"
                ))
            }
        })
    }
}

/// The energy and its parts (Eh); they sum to the energy.
#[derive(Clone, Debug, Serialize)]
pub struct EnergyComponents {
    pub nuclear_repulsion: f64,
    pub one_electron: f64,
    pub coulomb: f64,
    /// Exact exchange (times the functional's fraction, or with the range-separated operator).
    pub exchange: f64,
    pub xc: f64,
    pub dispersion: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub solvation_electrostatic: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub solvation_cds: Option<f64>,
}

/// The `scf` block: Fock builds (iterations plus the builds of UHF/UKS stability analyses and of
/// the closed-shell probe),
/// incremental builds, the thresholds, the per-iteration history and the stability analyses.
#[derive(Clone, Debug, Serialize)]
pub struct ScfSummary {
    pub converged: bool,
    pub iterations: usize,
    pub fock_builds: usize,
    pub incremental_builds: usize,
    pub convergence: Convergence,
    pub history: Vec<ScfIteration>,
    pub stability: Vec<StabilityCheck>,
}

impl ScfSummary {
    pub fn of(r: &ScfResult) -> ScfSummary {
        ScfSummary {
            converged: r.converged,
            iterations: r.iterations,
            fock_builds: r.iterations
                + r.stability.iter().map(|c| c.fock_builds).sum::<usize>()
                + r.closed_shell_probe
                    .and_then(|p| p.fock_builds)
                    .unwrap_or(0),
            incremental_builds: r.incremental_builds,
            convergence: r.convergence,
            history: r.history.clone(),
            stability: r.stability.clone(),
        }
    }
}

/// The parts of a gradient (natoms x 3 each, Eh/bohr); they sum to the gradient.
#[derive(Clone, Debug, Serialize)]
pub struct GradientPartsOut {
    pub nuclear_repulsion: Vec<[f64; 3]>,
    /// One-electron terms, -W S^x and the ECPs included.
    pub one_electron: Vec<[f64; 3]>,
    pub two_electron: Vec<[f64; 3]>,
    pub xc: Vec<[f64; 3]>,
    pub dispersion: Vec<[f64; 3]>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub solvent: Option<Vec<[f64; 3]>>,
}

/// What a Hessian's analysis adds to it (the `freq` settings).
#[derive(Clone, Debug, Serialize)]
pub struct FrequencyOut {
    /// The harmonic analysis (`freq.analysis`), with qcdata's names.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vibrational_analysis: Option<VibrationalAnalysis>,
    /// One thermochemistry per temperature (`freq.thermo`, `freq.temperature`), with qcdata's
    /// names.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub thermochemistry: Vec<Thermochemistry>,
    /// With `ir`: dmu/dX (3N x 3, e).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dipole_derivatives: Option<Vec<[f64; 3]>>,
    /// With `ir`: the static polarizability (a.u.).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub polarizability: Option<[[f64; 3]; 3]>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub polarizability_cphf_iterations: Option<usize>,
}

/// VV10 nonlocal correlation of -V functionals (its energy is part of `components.xc`).
#[derive(Clone, Debug, Serialize)]
pub struct Vv10Out {
    pub energy: Option<f64>,
    pub nlc_grid: GridChoice,
    pub grid_points: usize,
    pub active_points: usize,
}

/// The structure as computed, as qcdata's `Structure` names it (`geometry` in bohr), with the
/// core electrons each atom's ECP replaces.
#[derive(Clone, Debug, Serialize)]
pub struct StructureOut {
    pub symbols: Vec<&'static str>,
    pub geometry: Vec<[f64; 3]>,
    pub charge: i32,
    pub multiplicity: u32,
    pub ecp_core_electrons: Vec<u32>,
}

/// The result of [`Session::run`]: the whole calculation, typed. Serialized, it is the CLI's
/// `--json` output without the optional matrices ([`Report::to_json`]).
#[derive(Clone, Debug, Serialize)]
pub struct Report {
    pub schema_version: u32,
    pub program: &'static str,
    pub version: &'static str,
    /// [`crate::NUMERICS_EPOCH`]: the arithmetic that produced the numbers.
    pub numerics_epoch: u32,
    pub calctype: CalcType,
    /// The SCF converged.
    pub success: bool,
    pub energy: f64,
    pub components: EnergyComponents,
    /// Gradient (natoms x 3, Eh/bohr) of gradient, Hessian and frequency calculations.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gradient: Option<Vec<[f64; 3]>>,
    /// Its parts (gradient calculations).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gradient_parts: Option<GradientPartsOut>,
    /// The Hessian, 3N x 3N, row-major (index (3a + i) 3N + 3b + j; Eh/bohr^2).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hessian: Option<Vec<f64>>,
    /// Iterations of the Hessian's CPHF/CPKS.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cphf_iterations: Option<usize>,
    #[serde(flatten)]
    pub frequencies: Option<FrequencyOut>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub solvation: Option<SolvationResult>,
    pub converged: bool,
    pub iterations: usize,
    pub scf: ScfSummary,
    pub restricted: bool,
    pub natoms: usize,
    pub nbf: usize,
    pub nmo: usize,
    pub nalpha: usize,
    pub nbeta: usize,
    /// <S^2> of the determinant.
    pub s2: f64,
    pub orbital_energies_alpha: Vec<f64>,
    pub orbital_energies_beta: Vec<f64>,
    pub occupations_alpha: Vec<f64>,
    pub occupations_beta: Vec<f64>,
    pub homo_lumo_gap: Option<f64>,
    /// e bohr, about the coordinate origin.
    pub dipole: [f64; 3],
    /// The SCF's DFT grid.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub grid: Option<GridChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vv10: Option<Vv10Out>,
    pub structure: StructureOut,
    pub method: String,
    pub basis: String,
    /// What the user should know about a calculation that succeeded (`{code, message, path}`;
    /// the CLI prints them on stderr): `backend_fallback`, `df_storage_by_free_memory`,
    /// `closed_shell_unstable`, `non_aufbau`, `mode_check_flagged`, `ir_declined`,
    /// `plan_violation`, `convergence_looser_than_default`.
    pub warnings: Vec<linus_core::Warning>,
    /// Wall-clock seconds per phase; `total` the whole call, `energy` its SCF stage.
    pub timings: BTreeMap<String, f64>,
    /// Sizes, counts and peak memory (not seconds); with a device the calculation's peak device
    /// memory (`mem_device_peak_gib`).
    pub resources: BTreeMap<String, f64>,
    /// Files of the result by name (text): the Molden file of the orbitals
    /// (`orbitals.molden`, `output.molden`). Absent when empty.
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    pub files: BTreeMap<String, String>,
    /// The settings the request gave, normalized (the canonical keys; shorthands expanded).
    pub options: CalcOptions,
    pub resolved: Resolved,
    /// The SCF: orbitals, densities, Fock matrices (not serialized).
    #[serde(skip)]
    pub wavefunction: ScfResult,
    /// The AO matrices the request asked for (not serialized; [`Session::ao_matrices`]).
    #[serde(skip)]
    pub ao: AoMatrices,
}

/// Which AO matrices a request adds to its report beside the wavefunction's orbitals and Fock
/// matrices (`output.overlap`, `output.core_hamiltonian`; `output.hamiltonian` implies the
/// overlap its orbitals are eigenvectors in).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct AoOutputs {
    pub overlap: bool,
    pub core_hamiltonian: bool,
}

impl AoOutputs {
    /// The AO matrices the `output` settings ask for.
    pub fn of(o: &crate::driver::options::OutputSettings) -> AoOutputs {
        AoOutputs {
            overlap: o.overlap() || o.hamiltonian(),
            core_hamiltonian: o.core_hamiltonian(),
        }
    }

    /// Both requests' matrices.
    pub fn or(self, other: AoOutputs) -> AoOutputs {
        AoOutputs {
            overlap: self.overlap || other.overlap,
            core_hamiltonian: self.core_hamiltonian || other.core_hamiltonian,
        }
    }

    /// Whether any matrix is asked for.
    pub fn any(self) -> bool {
        self.overlap || self.core_hamiltonian
    }
}

/// The AO matrices of a report ([`AoOutputs`]), nbf x nbf in the basis set's AO order: the
/// overlap S and the core Hamiltonian h (the kinetic energy, the attraction of the nuclei with
/// their effective charges Z - n_core, and the ECPs: every density-independent one-electron term
/// of the run), as the SCF used them.
#[derive(Clone, Debug, Default)]
pub struct AoMatrices {
    pub overlap: Option<Mat<f64>>,
    pub core_hamiltonian: Option<Mat<f64>>,
}

impl Report {
    /// The whole result as one JSON object (the CLI's `--json` output) with the optional outputs
    /// the settings ask for: the MO coefficients (`output.orbitals`: `mo_coefficients_*`, nbf x
    /// nmo per spin), the Fock matrices of the SCF's last build (`output.hamiltonian`:
    /// `hamiltonian_*`, nbf x nbf per spin), the overlap (`output.overlap`, and with the
    /// Hamiltonian: `overlap`) and the core Hamiltonian (`output.core_hamiltonian`:
    /// `core_hamiltonian`).
    pub fn to_json(&self) -> Value {
        let mut out = serde_json::to_value(self).expect("a report serializes");
        let o = &self.options.output;
        if let Value::Object(m) = &mut out {
            let w = &self.wavefunction;
            if o.hamiltonian() {
                m.insert("hamiltonian_alpha".into(), mat_rows(&w.f_a));
                m.insert("hamiltonian_beta".into(), mat_rows(&w.f_b));
            }
            if o.orbitals() {
                m.insert("mo_coefficients_alpha".into(), mat_rows(&w.c_a));
                m.insert("mo_coefficients_beta".into(), mat_rows(&w.c_b));
            }
            if let Some(s) = &self.ao.overlap {
                m.insert("overlap".into(), mat_rows(s));
            }
            if let Some(h) = &self.ao.core_hamiltonian {
                m.insert("core_hamiltonian".into(), mat_rows(h));
            }
        }
        out
    }
}

/// What a calculation used (`resolved`), generated from its settings and its run record
/// (never reconstructed from the session afterwards): the calctypes it covered, the settings
/// that reproduce it, why each `auto` took its value, and what happened on the way.
#[derive(Clone, Debug, Serialize)]
pub struct Resolved {
    /// The calctypes the run covered: its own and the lower derivatives it computed (`energy`,
    /// `gradient`, `hessian`, and `ir` for a Hessian with IR intensities).
    pub calctypes: Vec<&'static str>,
    /// The calctypes the session declared (`Session::declare`; all by default), which its
    /// given settings were checked against (decision D41).
    pub declared_calctypes: Vec<&'static str>,
    /// The canonical settings of the run (decision D41): only the keys that take effect, every
    /// `auto` as the value it took, grids as their strings (`"linus:3"`). Itself a valid
    /// request: run again with it, the calculation is the same (`crate::driver::settings`).
    pub settings: Value,
    /// Why a setting took its value, by its dotted path (`grid`, `jk.aux`,
    /// `precision.fp64`): the `auto` rules.
    pub reasons: BTreeMap<String, String>,
    /// What happened during the run (diagnostic: its keys may change without a
    /// `schema_version` bump).
    pub record: RunRecordOut,
}

/// `resolved.record`: what the run did and decided.
#[derive(Clone, Debug, Serialize)]
pub struct RunRecordOut {
    /// The SCF started from its lineage's history (a session's earlier orbitals), not from the
    /// input alone, so it may reach another SCF solution than a fresh start (qcdata's shared
    /// record key; `scf.guess` has the detail).
    pub warm_start: bool,
    /// Every given setting whose effect depends on the calctype (it takes effect in a declared
    /// calctype but not in an energy: `freq.*` and `response.*` in a session that may run
    /// Hessians), by its path: whether it took effect in this run. Absent when there is none (a
    /// one-shot run declares its own calctype only).
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    pub settings_applied: BTreeMap<String, bool>,
    /// What the SCF was converged for.
    pub purpose: Purpose,
    /// Kohn-Sham DFT (false: Hartree-Fock).
    pub dft: bool,
    pub scf: ResolvedScf,
    /// The density-fitting storage of the SCF (storage, precision, long-range mode, size,
    /// budget, auxiliary functions).
    pub df: Option<ResolvedDf>,
    pub memory: ResolvedMemory,
    /// The response equations (Hessians and frequencies).
    pub response: Option<ResolvedResponse>,
    pub solvent: Option<ResolvedSolvent>,
    /// What ran each family of kernels.
    pub backend: ResolvedBackend,
    /// The `LINUS_*` environment variables that were set.
    pub debug_env: BTreeMap<String, String>,
    /// With accuracy targets (the request's, else the settings' `accuracy`; qcdata's shared
    /// record key): the `target`, the `thresholds` they set, the `estimate` of each targeted
    /// error from the final orbital gradient and response residual, and whether every estimate
    /// is within its target (`met`; `crate::driver::accuracy`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub accuracy: Option<crate::driver::accuracy::AccuracyRecord>,
}

/// The warning of a converged SCF whose occupied set is not the aufbau set of its final Fock
/// matrix (`non_aufbau`; `resolved.record.scf.non_aufbau`): the result is consistent (the energy,
/// orbitals and derivatives are those of the converged determinant), but a lower aufbau solution
/// may exist.
fn non_aufbau_warning(r: &ScfResult) -> Option<linus_core::Warning> {
    let worst = r
        .non_aufbau
        .iter()
        .max_by(|a, b| a.inversion.total_cmp(&b.inversion))?;
    let channel = match worst.spin {
        "restricted" => String::new(),
        s => format!(" {s}"),
    };
    Some(linus_core::Warning {
        code: "non_aufbau",
        message: format!(
            "the converged SCF state does not follow aufbau: {} empty{channel} orbital(s) lie below \
             occupied ones (the highest occupied orbital energy is {:.1e} Eh above the lowest \
             empty one); it is a stationary point, and a lower aufbau solution may exist \
             (resolved.record.scf.non_aufbau)",
            worst.holes.len(),
            worst.inversion
        ),
        path: "resolved.record.scf.non_aufbau",
    })
}

/// The warning of an SCF accepted on a soft-mode plateau, when its escalation came due or at
/// `max_iter`: qcdata's shared `loosely_converged` (its thresholds missed, accepted under a
/// relaxed rule), with the residual reached and the rule in the message and the threshold missed
/// as its `path`.
fn plateau_warning(r: &ScfResult) -> Option<linus_core::Warning> {
    let p = r.plateau?;
    let c = &r.convergence;
    Some(linus_core::Warning {
        code: "loosely_converged",
        message: format!(
            "the SCF did not meet its thresholds in {} iterations and was accepted by the plateau \
             rule (scf.plateau_window): over the last {} iterations the energy stayed within {:.1e} \
             Eh and the orbital gradient below {:.1e} (last {:.1e}; g_tol {:.0e}); the result is \
             converged more loosely than asked (resolved.record.scf.plateau)",
            r.iterations, p.window, p.energy_spread, p.max_residual, p.residual, c.g_tol
        ),
        path: "scf.g_tol",
    })
}

/// The SCF as it ran.
#[derive(Clone, Debug, Serialize)]
pub struct ResolvedScf {
    /// Where the SCF started: `source` (`sad` or `core`: the configured guess; `lineage`: its
    /// lineage's last orbitals or densities; `given`), its `lineage` and the one whose state
    /// started it (`from_lineage`: its own or a seed), the largest atomic `distance` (bohr) from
    /// that lineage's last positions, whether the orbitals were `orthonormalized` here, and why a
    /// lineage's SCF started fresh (`reason`: `no_history` or `distance`).
    pub guess: crate::driver::lineage::GuessRecord,
    /// Basis functions dropped as linearly dependent (nbf - nmo).
    pub dropped_functions: usize,
    /// Iteration at which a two-cycle was detected and SOSCF steps took over (`scf.two_cycle`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub two_cycle_at: Option<usize>,
    /// Converged by the soft-mode plateau rule (`scf.plateau_window`, with the warning
    /// `loosely_converged`): the window, the energy spread over it and the orbital gradient
    /// left.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub plateau: Option<Plateau>,
    /// The escalation (`scf.escalate`), when it ran: `{"ran": true}`, its `trigger`
    /// (`stagnation`, `blowup` or `two_thirds`), the iteration `at` which it started, the iteration
    /// (`from_iter`) and energy of the lowest-energy orbitals it restarted from, the
    /// `iterations` it ran and whether it `converged`; when the SCF used `max_iter` without it,
    /// `{"ran": false, "reason"}` (`off`, or `max_iter`: too few iterations left for it).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub escalation: Option<Escalation>,
    /// Hessian-vector products of the stability probe the SCF ran at its converged point (0 for
    /// restricted determinants, which it does not check; the gap trigger's when it fired). Its
    /// analyses are in `scf.stability`.
    pub stability_probe: usize,
    /// The closed-shell probe of an RHF/RKS solution (`scf.stability_closed_probe`, decision D36):
    /// `products`, the HOMO-LUMO `gap` and the `threshold`; when it ran the `eigenvalue` (an upper
    /// bound of the lowest UHF/UKS orbital-Hessian eigenvalue), `residual`, `kind` (`external`:
    /// towards a broken-symmetry UHF/UKS solution; `internal`), `unstable` (with a warning),
    /// `fock_builds` and `seconds`; else `skipped`: `gap_above_threshold`, `off`, `hf_default`,
    /// `later_geometry`, `no_virtual_orbitals` or `not_converged`. Absent for UHF/UKS
    /// determinants.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub closed_shell_probe: Option<ClosedShellProbe>,
    /// The gap trigger at an SCF that continues its lineage (`crate::driver::lineage`), where the
    /// unset probes do not run: `last_gap` (the gap at the lineage's last probe), `gap` (this
    /// SCF's: RHF/RKS HOMO-LUMO, UHF/UKS the smaller alpha or beta one) and `trigger:
    /// "gap_dropped"` when the gap fell below `scf.stability_closed_gap` since the last probe, or
    /// 0.02 Eh below its gap there: the probes then ran. Absent where the trigger was not armed.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub probe_trigger: Option<TriggerRecord>,
    /// The spin channels whose converged occupied set is not the aufbau set of the final Fock
    /// matrix (with the warning `non_aufbau`): `spin` (`restricted`, `alpha` or `beta`), the
    /// `holes` (empty orbitals aufbau would occupy) and `particles` (occupied ones it would leave
    /// empty) by their 0-based positions in ascending energy, and the `inversion` (Eh: the highest
    /// occupied orbital energy less the lowest empty one). Absent for aufbau solutions.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub non_aufbau: Vec<linus_scf::NonAufbau>,
}

/// The memory limit and the budgets derived from it (GiB).
#[derive(Clone, Debug, Serialize)]
pub struct ResolvedMemory {
    /// Where the limit came from: `memory_gib`, `cgroup limit` or `total RAM`.
    pub source: &'static str,
    /// The limit.
    pub limit_gib: Option<f64>,
    /// The budget of the SCF's 3-index DF tensors: 75% of the limit less the reservations, with
    /// `memory_gib` unset at most 90% of the free memory less the reservations.
    pub df_tensor_budget_gib: Option<f64>,
    /// What set that budget: the limit's `source`, or `available memory` when the free memory
    /// at the SCF's start lowered it (the guard of an unset `memory_gib`).
    pub df_tensor_budget_source: Option<&'static str>,
    /// The memory free for the calculation when the SCF started (MemAvailable, within a capped
    /// cgroup's headroom), read with `memory_gib` unset.
    pub available_gib: Option<f64>,
    /// The memory reserved for the large allocations alive with the DF tensors (`what`, `gib`:
    /// the PCM's dense matrices while they are built).
    pub reservations: Vec<linus_base::memory::Reservation>,
    /// The H-bar blocks of RI Hessians.
    pub hbar_gib: Option<f64>,
    /// The XC-kernel accumulators.
    pub xc_kernel_gib: f64,
    /// One response buffer slot.
    pub response_slot_gib: f64,
}

/// The response (CPHF/CPKS) of a Hessian or frequency calculation, as it ran.
#[derive(Clone, Debug, Serialize)]
pub struct ResolvedResponse {
    /// The max-abs residual it reached (its tolerance is `settings.response.cphf_tol`).
    pub max_residual: f64,
    /// The response J/K ran in single precision.
    pub single_precision: bool,
    /// The iterations ran with a cheaper J/K (on `settings.response.iteration_aux`).
    pub two_level: bool,
    /// The max-abs residual the electric-field CPHF of the polarizability reached (frequencies
    /// with `ir`).
    pub pol_max_residual: Option<f64>,
    /// With `ir`: the electric-field CPHF/CPKS that stopped above `response.field_tol`, as a
    /// failure's `response` (`what`, `iterations`, `max_residual`, `tol`, `end`), the IR part
    /// declined (warning `ir_declined`); absent when the IR part was computed.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ir_declined: Option<ResponseFailureOut>,
    /// The zero-mode check of open-shell Hessians (`scf.stability_hessian_check`; absent when it did
    /// not run: closed shells, atoms, off).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mode_check: Option<ResolvedModeCheck>,
}

/// The zero-mode check of an open-shell Hessian: the lowest orbital-Hessian eigenvalue of the
/// SCF solution it accepted (an upper bound from a few Davidson steps per irrep, or the full
/// analysis', [`linus_scf::ModeCheck`]; a negative mode a step followed is the SCF's history,
/// `scf.stability`) and the verdict. `flagged`: the solution is degenerate or unstable, so the analytic Hessian and the
/// frequencies are not those of the energy surface (bench/robust/repro/uhf_hessian_zero_mode).
#[derive(Clone, Debug, Serialize)]
pub struct ResolvedModeCheck {
    pub eigenvalue: f64,
    pub residual: f64,
    pub fock_builds: usize,
    pub seconds: f64,
    /// [`linus_scf::MODE_CHECK_THRESHOLD`] (gas-phase HF diatomics: flagged only when negative,
    /// [`linus_scf::ModeCheck::flagged`]).
    pub threshold: f64,
    pub flagged: bool,
    /// The irreps it searched and the irrep of its Ritz value.
    #[serde(flatten)]
    pub irreps: linus_scf::stability::IrrepRecord,
}

/// The implicit solvent as it ran.
#[derive(Clone, Debug, Serialize)]
pub struct ResolvedSolvent {
    /// The permittivity used (the solvent database's for a name).
    pub eps: Option<f64>,
    /// The full CRC table the named solvent's permittivity came from, when linus does not ship
    /// it (`solvent::db::crc_table`; absent otherwise).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub eps_table: Option<String>,
    /// SMD's contact smoothing (Å) of G_CDS.
    pub cds_contact_smoothing: f64,
}

/// What ran each family of kernels (`cpu` or `gpu`; None: the method has none), the device and
/// the fallbacks of a GPU backend ([`crate::driver::backend`]).
#[derive(Clone, Debug, Serialize)]
pub struct ResolvedBackend {
    /// The device a GPU backend opened.
    pub device: Option<crate::driver::backend::DeviceInfo>,
    /// Backend auto without a device: why it runs on the CPU.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub no_device: Option<String>,
    /// The arithmetic of the device: double; emulated (the density-fitted exchange GEMMs in
    /// FP64 emulation); mixed (the XC gradient's and the CPKS kernel's products in FP32). The
    /// CPU computes in double.
    pub precision: &'static str,
    /// With a device: every kernel family's arithmetic (a fallback chain where it has one) and
    /// why, from the device policy (`crate::driver::policy`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub arithmetic: Option<BTreeMap<&'static str, FamilyArith>>,
    /// The overlap, kinetic energy and ECP integrals (and the nuclear attraction unless
    /// `nuclear_attraction` says otherwise).
    pub one_electron: &'static str,
    /// Where the SCF's nuclear attraction V ran (`gpu` or `cpu`) and why, when a device J/K's
    /// setup decided it (large molecules: the device PCM engine's pair x point integrals).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub nuclear_attraction: Option<PlaceRecord>,
    /// Where the last grid built with a device partition (an unscreened partition: PySCF's Becke
    /// grids) took its partition weights (`gpu`, or `cpu` when the device declined) and why;
    /// absent when no grid asked the device (screened partitions are the host's own).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub grid_partition: Option<PlaceRecord>,
    /// The SCF's J/K builds.
    pub jk: &'static str,
    /// The SCF's XC energy and potential (DFT).
    pub xc: Option<&'static str>,
    /// VV10 nonlocal correlation (-V functionals).
    pub vv10: Option<&'static str>,
    /// The two-electron term of gradients.
    pub gradient_jk: &'static str,
    /// The kernels of a device two-electron gradient and why (`{"kernels", "reason"}`:
    /// `hermite_j`, the J-only gradient in Hermite space; `stored_planes`; `fused`); absent
    /// without one.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gradient_jk_kernels: Option<KernelsRecord>,
    /// The XC term of gradients (DFT).
    pub gradient_xc: Option<&'static str>,
    /// The explicit two-electron terms of Hessians (the second derivatives at fixed orbitals and
    /// the derivative Fock matrices of the CPHF right-hand sides).
    pub hessian_jk: &'static str,
    /// The explicit XC terms of Hessians (DFT): the XC potential's derivatives, the XC energy's
    /// second derivative and the grid response.
    pub hessian_xc: Option<&'static str>,
    /// The J/K products of CPHF/CPKS responses (jk exact and df).
    pub response_jk: Option<&'static str>,
    /// The CPKS XC kernel (DFT).
    pub response_xc: Option<&'static str>,
    /// Where the SCF loop's own work ran: host or device.
    pub scf_loop: &'static str,
    /// The SCF's density-fitted J when it ran on the device without a stored three-index
    /// tensor (`{"engine": "direct", "reason", "kernels", "kernels_reason"}`: the tensor does
    /// not fit the device beside the other engines); absent on the DF engine.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub j: Option<JEngine>,
    /// An implicit solvent's surface terms in the SCF and gradients (the surface charges'
    /// integrals and the charge solve).
    pub pcm: Option<&'static str>,
    /// The rest of an implicit solvent: the cavity, SMD's CDS term and the PCM's Hessian terms
    /// (always the CPU).
    pub solvent: Option<&'static str>,
    pub dispersion: Option<&'static str>,
    /// The families a device runs on the CPU, with the reason.
    pub fallbacks: Vec<Fallback>,
    /// The families whose engine drew beyond its workspace grant (the device memory plan),
    /// with the first such draw; absent when every engine kept to its grant.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub plan_violations: Vec<PlanViolationRecord>,
    /// With FP64 emulation: the tier each device family's exchange GEMMs ran (the SCF's `jk`,
    /// the response's `response_jk`, the Hessian's `hessian_jk`), with the reason when a lower
    /// tier than the first ran.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub emulation: Vec<EmulationUsed>,
    /// The arithmetic of the device's CPHF/CPKS products, per response (a Hessian's CPKS, a
    /// polarizability's field CPHF); empty when no response product ran on the device.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub response: Vec<ResponseArithmetic>,
    /// The arithmetic of VV10's pair sums when VV10 ran on the device.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vv10_arithmetic: Option<Vv10Arithmetic>,
    /// Where each device XC family evaluated the functional's derivatives: `gpu` (libxc's code
    /// compiled for the device) or `cpu` (libxc on the host, with the reason); absent without
    /// device XC.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub functional: Vec<FunctionalRecord>,
}

/// Where a device XC family evaluated the functional ([`ResolvedBackend::functional`]).
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct FunctionalRecord {
    pub kernel: &'static str,
    /// `gpu` or `cpu`.
    pub place: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
}

/// A kernel family's arithmetic and why (`resolved.record.backend.arithmetic`).
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct FamilyArith {
    /// The arithmetic (`fp64`, `ozaki (55 bits)`, ...), or the fallback chain the first of whose
    /// tiers that sets up runs.
    pub arith: String,
    pub reason: String,
}

/// The arithmetic of VV10's pair sums on the device (`resolved.record.backend.vv10_arithmetic`):
/// `df64` (double-float: two FP32 numbers per value, each point's sums added in FP64), `fp64`
/// or `fp32`, for the SCF's energy and potential and for the gradient.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Vv10Arithmetic {
    pub scf: &'static str,
    pub gradient: &'static str,
}

/// The arithmetic of one response's device products (`resolved.record.backend.response`).
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ResponseArithmetic {
    /// `hessian` (the CPKS of a Hessian) or `polarizability` (the electric-field CPHF).
    pub purpose: &'static str,
    /// The exchange of the J/K products when `response_jk` ran on the device: `fp64`, `fp32`
    /// (`precision.response` resolved to fp32) or `emulated` (its tier in `emulation`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub jk: Option<&'static str>,
    /// How the device's exchange ran: `operator` (K as a stored (n o)^2 matrix per spin
    /// channel, built once, when the products' perturbations pay for it and it fits) or
    /// `tensor` (through the fitted three-center tensor in every product).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub exchange: Option<&'static str>,
    /// GiB of the stored operator.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub operator_gib: Option<f64>,
    /// Of `operator_gib`, the GiB in page-locked host memory: the chunks that did not fit the
    /// device, streamed to it in every product (overlapped with the products on the rest).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub operator_host_gib: Option<f64>,
    /// Why the exchange stayed on the tensor where the operator would have paid (it did not
    /// fit the device beside the other engines' reserve).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub operator_note: Option<String>,
    /// The device memory granted to the stored operator at the response's setup (the memory
    /// plan's grant; GiB), and the build's draw beyond it (a plan violation, with a
    /// `plan_violation` warning).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub operator_granted_gib: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub plan_violation: Option<String>,
    /// `precision.response: "auto"` on a device with little FP64 throughput (decision D37):
    /// the estimated seconds of the response's exchange on each path it weighed, the cheaper
    /// of which ran (`exchange`; `reason` says so).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub estimate_s: Option<ExchangeEstimate>,
    /// Where the CPHF/CPKS solver's own algebra ran (a Hessian's, with its products on the
    /// device): `gpu` (the subspace, its products and the residuals resident on the device) or
    /// `cpu`, with the reason.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub solver: Option<PlaceRecord>,
    /// Where the second-order assembly's stacked GEMMs ran (a Hessian's, with its products on
    /// the device): `gpu`, or `cpu` with the reason (the device's memory).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub assembly: Option<PlaceRecord>,
    /// The CPKS XC kernel when `response_xc` ran on the device: `fp64`; `emulated` (its two
    /// large GEMMs per block in the Ozaki scheme) with `precision.fp64: "emulated"`; or
    /// `fp32` (products in FP32 per grid batch, summed in FP64) when `precision.response` is
    /// fp32, or auto where the CPU's rule runs single precision, or with
    /// `precision.fp32: "on"`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub xc: Option<&'static str>,
    /// Why this arithmetic: the setting (`precision.response`), `auto`'s choice (the stored
    /// operator in FP64 where it fits, else single precision; double for a small exchange or
    /// a tolerance single precision does not reach), or a polarizability's double precision.
    pub reason: String,
}

/// The estimated seconds of a response's exchange products on the two paths
/// `precision.response: "auto"` weighs on a device with little FP64 throughput
/// (`resolved.record.backend.response[].estimate_s`, decision D37): all of the response's products
/// (a Hessian's 3N perturbations over the CPKS iterations; with IR intensities also the
/// electric-field solve that follows), from the card's rates.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct ExchangeEstimate {
    /// On the stored exchange operator in FP64: its setup and products (on the device, or
    /// streamed from host memory as far as it does not fit).
    pub operator: f64,
    /// In FP32 through the density-fitting tensor in every product.
    pub tensor: f64,
}

/// The FP64 emulation of one family's exchange GEMMs (`resolved.record.backend.emulation`).
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct EmulationUsed {
    pub kernel: &'static str,
    /// `ozaki` (linus's Ozaki scheme on INT8 tensor cores), `cublas` (cuBLAS's fixed-mantissa
    /// emulation) or `fp64` (native: neither set up).
    pub scheme: &'static str,
    /// Mantissa bits per operand value: 8 slices - 1 (ozaki), the fixed mantissa (cublas), 53
    /// (fp64).
    pub bits: u32,
    /// ozaki: the stored tensor was sliced once (else each build slices the blocks it streams).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub presliced: Option<bool>,
    /// Why a lower tier than ozaki ran.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
    /// `jk`: the density-fitting setup's fit B = (mn|P) L^-T, emulated as well.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fit: Option<FitEmulation>,
}

/// The arithmetic of the device's density-fitting fit under `precision: "emulated"`
/// (`resolved.record.backend.emulation[].fit`).
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct FitEmulation {
    /// `ozaki` (the triangular solve blocked, its updates in the Ozaki scheme on INT8 tensor
    /// cores) or `fp64` (the triangular solve in FP64).
    pub scheme: &'static str,
    /// Mantissa bits per operand value of the updates: 8 slices - 1 (ozaki), 53 (fp64).
    pub bits: u32,
    /// Why the fit ran in FP64.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
}

/// The engine of the SCF's density-fitted J ([`ResolvedBackend::j`]) and why: `direct` (no
/// stored tensor) on its `kernels`, `hermite` (the J-engine contractions, FP64-rich cards) or
/// `eri3c` (recomputed three-center integrals), and why those; or `stored` where the direct
/// engine was chosen and its setup `declined` (why).
#[derive(Clone, Debug, Serialize)]
pub struct JEngine {
    pub engine: &'static str,
    pub reason: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub kernels: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub kernels_reason: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub declined: Option<String>,
}

/// The kernels a device family ran, and why ([`ResolvedBackend::gradient_jk_kernels`]).
#[derive(Clone, Debug, Serialize)]
pub struct KernelsRecord {
    pub kernels: &'static str,
    pub reason: String,
}

/// Where a part of a calculation ran, `gpu` or `cpu`, and why (`resolved.record.backend`'s
/// `nuclear_attraction` and `grid_partition`, a response's `solver` and `assembly`).
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct PlaceRecord {
    pub place: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
}

impl PlaceRecord {
    /// On the GPU (no reason).
    pub(crate) fn gpu() -> Self {
        PlaceRecord {
            place: "gpu",
            reason: None,
        }
    }

    /// On the CPU, and why.
    pub(crate) fn cpu(reason: String) -> Self {
        PlaceRecord {
            place: "cpu",
            reason: Some(reason),
        }
    }
}

/// A family of kernels a GPU backend runs on the CPU, and why.
#[derive(Clone, Debug, Serialize)]
pub struct Fallback {
    pub kernel: &'static str,
    pub reason: String,
}

/// A family's draw beyond its workspace grant ([`ResolvedBackend::plan_violations`]).
#[derive(Clone, Debug, Serialize)]
pub struct PlanViolationRecord {
    pub kernel: &'static str,
    pub violation: String,
}

/// A failed calculation (the CLI's failure JSON; the Python exception's `failure`): the error,
/// its kind and, for a calculation that stopped without converging, what it reached.
#[derive(Clone, Debug, Serialize)]
pub struct Failure {
    pub schema_version: u32,
    pub program: &'static str,
    pub version: &'static str,
    pub calctype: CalcType,
    pub success: bool,
    pub error: String,
    pub error_kind: &'static str,
    /// The error's code within its kind ([`LinusError::code`]: a setting's `unknown_key`,
    /// `no_effect` or `not_applicable`; a resource's `host_memory` or `device_memory`; a
    /// convergence failure's `scf_not_converged` or `response_not_converged`; qcdata's shared
    /// codes). Absent when it has none.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error_code: Option<&'static str>,
    /// The error's structured facts ([`error_details`]: qc*'s `ErrorInfo.details`), a memory
    /// failure's `required_gib` and `available_gib`. Absent when it has none.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error_details: Option<Value>,
    /// The last SCF iterate's `scf` block (an SCF that did not converge, failed numerically or
    /// stopped on request).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub scf: Option<ScfSummary>,
    /// Its energy.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub energy: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timings: Option<BTreeMap<String, f64>>,
    /// A CPHF/CPKS that stopped above its tolerance: what it solved, its iterations, residual and
    /// tolerance.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response: Option<ResponseFailureOut>,
    /// The progress lines of `output.verbose` collected for an in-process caller
    /// ([`Session::collect_progress`]; the command line wrote them to stderr).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logs: Option<String>,
    /// The report of what the calculation reached (the unconverged SCF; the converged SCF with
    /// the Hessian of an unconverged response): the Python exception's `result` (not
    /// serialized).
    #[serde(skip)]
    pub result: Option<Box<Report>>,
}

/// A CPHF/CPKS that stopped above its tolerance (`LinusError::ResponseNotConverged`).
#[derive(Clone, Debug, Serialize)]
pub struct ResponseFailureOut {
    /// "analytic Hessian" or "polarizability".
    pub what: &'static str,
    pub iterations: usize,
    pub max_residual: f64,
    pub tol: f64,
    /// How the solve ended: `iterations` (at the limit), `exhausted` (no direction left to add:
    /// a singular operator), `indefinite` (the orbital Hessian has an eigenvalue at or below
    /// zero) or `non_finite` (a NaN residual).
    pub end: ResponseEnd,
}

impl From<&ResponseFailure> for ResponseFailureOut {
    fn from(f: &ResponseFailure) -> Self {
        ResponseFailureOut {
            what: f.what,
            iterations: f.iterations,
            max_residual: f.max_residual,
            tol: f.tol,
            end: f.end,
        }
    }
}

impl Failure {
    /// The failure as JSON (the CLI's failure output).
    pub fn to_json(&self) -> Value {
        serde_json::to_value(self).expect("a failure serializes")
    }
}

/// The structured facts of an error (the failure JSON's `error_details`, qc*'s
/// `ErrorInfo.details`): for a `host_memory` or `device_memory` failure, `required_gib` and
/// `available_gib` ([`linus_base::error::MemoryNeed`]), each when known; None for any other error.
pub fn error_details(err: &LinusError) -> Option<Value> {
    let need = err.memory_need()?;
    let mut m = serde_json::Map::new();
    if let Some(r) = need.required_gib {
        m.insert("required_gib".into(), json!(r));
    }
    if let Some(a) = need.available_gib {
        m.insert("available_gib".into(), json!(a));
    }
    (!m.is_empty()).then_some(Value::Object(m))
}

/// A matrix as a list of rows.
pub fn mat_rows(m: &Mat<f64>) -> Value {
    Value::Array(
        (0..m.nrows())
            .map(|i| json!((0..m.ncols()).map(|j| m[(i, j)]).collect::<Vec<_>>()))
            .collect(),
    )
}

/// Lowest unoccupied minus highest occupied orbital energy over both spins (None without
/// occupied or virtual orbitals).
pub fn homo_lumo_gap(r: &ScfResult) -> Option<f64> {
    let spins = [(&r.eps_a, r.nalpha), (&r.eps_b, r.nbeta)];
    let homo = spins
        .iter()
        .filter(|(_, n)| *n > 0)
        .map(|(e, n)| e[n - 1])
        .fold(None, |a: Option<f64>, e| Some(a.map_or(e, |a| a.max(e))));
    let lumo = spins
        .iter()
        .filter(|(e, n)| *n < e.len())
        .map(|(e, n)| e[*n])
        .fold(None, |a: Option<f64>, e| Some(a.map_or(e, |a| a.min(e))));
    Some(lumo? - homo?)
}

/// What a calculation produced besides its SCF.
#[derive(Default)]
struct Derivatives {
    gradient: Option<Vec<[f64; 3]>>,
    gradient_parts: Option<GradientPartsOut>,
    hessian: Option<Vec<f64>>,
    cphf_iterations: Option<usize>,
    frequencies: Option<FrequencyOut>,
    response: Option<ResolvedCphf>,
    pol_max_residual: Option<f64>,
    /// The electric-field response that missed its tolerance, the IR part declined.
    ir_declined: Option<ResponseFailure>,
}

impl Session {
    /// Run a calculation of `calctype` at the current geometry and report the result (a
    /// Hessian with the analysis of the `freq` settings). The session's timers then hold
    /// every phase of the call, `total` its wall time.
    pub fn run(&mut self, calctype: CalcType) -> Result<Report, LinusError> {
        let freq = self.cx.plan.freq_options();
        self.run_with(calctype, &freq)
    }

    /// [`Session::run`] with a Hessian's analysis as `freq` says (a front end's per-call
    /// `freq` settings, `crate::driver::CalcOptions::freq_options`).
    pub fn run_with(
        &mut self,
        calctype: CalcType,
        freq: &linus_vib::FrequencyOptions,
    ) -> Result<Report, LinusError> {
        let t0 = std::time::Instant::now();
        let ir = calctype == CalcType::Hessian && freq.ir;
        // what the method cannot do fails here, before any work
        self.cx.check_request(Request::of(calctype, ir))?;
        let wants = crate::driver::settings::CalcTypes::of(calctype, ir);
        if !self.cx.calctypes.covers(wants) {
            return Err(LinusError::Input(format!(
                "a {} calculation: the session was declared for {} only",
                calctype.name(),
                self.cx.calctypes.names().join(", ")
            )));
        }
        // the calculation's peak device memory counts from here (resources.mem_device_peak_gib)
        self.cx.backend.reset_device_peak();
        let mut more = Timers::default();
        let mut d = Derivatives::default();
        let gs = match calctype {
            CalcType::Energy => self.scf_stage(Purpose::Energy)?,
            CalcType::Gradient => {
                let g = self.gradient()?;
                let p = &g.parts;
                d.gradient_parts = Some(GradientPartsOut {
                    nuclear_repulsion: p.nuclear_repulsion.clone(),
                    one_electron: p.one_electron.clone(),
                    two_electron: p.two_electron.clone(),
                    xc: p.xc.clone(),
                    dispersion: p.dispersion.clone(),
                    solvent: g.solvent.clone(),
                });
                d.gradient = Some(g.gradient);
                more.merge(&g.timers);
                self.ground_after("gradient")?
            }
            CalcType::Hessian => {
                let f = self.hessian_analysis(freq)?;
                let dd = f.dipole_derivatives.as_ref();
                if let Some(x) = dd {
                    more.merge(&x.timers);
                }
                d.frequencies = Some(FrequencyOut {
                    dipole_derivatives: dd.map(|x| x.apt.clone()),
                    polarizability: dd.map(|x| x.polarizability),
                    polarizability_cphf_iterations: dd.map(|x| x.cphf_iterations),
                    vibrational_analysis: f.vib,
                    thermochemistry: f.thermo,
                });
                d.pol_max_residual = dd
                    .map(|x| x.max_residual)
                    .or(f.ir_declined.as_ref().map(|x| x.max_residual));
                d.ir_declined = f.ir_declined;
                d.gradient = Some(f.hessian.gradient);
                d.cphf_iterations = Some(f.hessian.cphf_iterations);
                d.response = Some(f.hessian.response);
                d.hessian = Some(f.hessian.hessian);
                more.merge(&f.hessian.timers);
                self.ground_after("Hessian")?
            }
        };
        self.timers.merge(&more);
        if let Some(peak) = self.cx.backend.device_peak() {
            self.timers
                .resource("mem_device_peak_gib", peak as f64 / 1073741824.0);
        }
        let mut rep = self.report(&gs, calctype, d);
        self.timers.set_total(t0.elapsed().as_secs_f64());
        rep.timings = timings(&self.timers);
        // the AO matrices the settings ask for, after everything else (outside the timings)
        rep.ao = self.ao_matrices(self.cx.plan.ao_outputs());
        Ok(rep)
    }

    /// The AO matrices `want` asks for at the current geometry (the last calculation's), as its
    /// SCF used them: S and h of a pair list built here and dropped after (never the
    /// calculation's cached one), so computing them changes no other number.
    pub fn ao_matrices(&self, want: AoOutputs) -> AoMatrices {
        if !want.any() {
            return AoMatrices::default();
        }
        let (s, h) = self.cx.pool.install(|| {
            let pairs = linus_ints::pairs::PairSet::build(
                &self.cx.basis,
                linus_ints::pairs::PairOptions::with_order(linus_ints::DerivOrder::Zero),
            );
            self.cx.one_electron(&pairs)
        });
        AoMatrices {
            overlap: want.overlap.then_some(s),
            core_hamiltonian: want.core_hamiltonian.then_some(h),
        }
    }

    /// The ground state of the current geometry after a derivative calculation (a broken
    /// invariant if there is none).
    fn ground_after(&self, after: &str) -> Result<Arc<GroundState>, LinusError> {
        self.ground_here()
            .cloned()
            .ok_or_else(|| LinusError::Internal(format!("no SCF result after the {after}")))
    }

    /// The report of an SCF (its ground state `gs`, converged or not: the last unconverged
    /// one's) without derivatives; times the dipole moment.
    pub(crate) fn scf_report(&mut self, gs: &GroundState, calctype: CalcType) -> Report {
        self.report(gs, calctype, Derivatives::default())
    }

    fn report(&mut self, gs: &GroundState, calctype: CalcType, d: Derivatives) -> Report {
        let r = &gs.scf;
        let t = std::time::Instant::now();
        let dipole = self.cx.dipole_moment(&r.d_a, &r.d_b);
        self.timers.add("dipole", t.elapsed().as_secs_f64());
        let solv = gs.solvation.as_ref();
        let components = EnergyComponents {
            nuclear_repulsion: r.e_nuc,
            one_electron: r.e_one,
            coulomb: r.e_coulomb,
            exchange: r.e_exchange,
            xc: r.e_xc,
            dispersion: r.e_disp,
            solvation_electrostatic: solv.map(|s| s.electrostatic),
            solvation_cds: solv.and_then(|s| s.cds),
        };
        let calctypes = CalcTypes {
            gradient: calctype != CalcType::Energy,
            hessian: calctype == CalcType::Hessian,
            ir: d.pol_max_residual.is_some(),
        };
        let resolved = self.resolved(
            gs,
            calctypes,
            d.response.as_ref(),
            d.pol_max_residual,
            d.ir_declined.as_ref(),
        );
        // what the user should know: the device's main families on the CPU, a DF storage the
        // node's free memory lowered, an unstable closed shell, a flagged open-shell Hessian
        let mut warnings = self.cx.cpu_fallback_warnings();
        warnings.extend(CalcContext::df_guard_warning(gs));
        let df_violation = gs.df.as_ref().and_then(|d| d.plan_violation.clone());
        let grant_violations = self
            .cx
            .backend
            .grant_violations()
            .into_iter()
            .map(|(k, v)| format!("{}: {v}", k.name()));
        warnings.extend(
            df_violation
                .into_iter()
                .chain(self.cx.backend.responses().violations())
                .chain(grant_violations)
                .map(|v| linus_core::Warning {
                    code: "plan_violation",
                    message: format!("the device memory plan was exceeded: {v}"),
                    path: "backend",
                }),
        );
        warnings.extend(d.ir_declined.as_ref().map(|f| linus_core::Warning {
            code: "ir_declined",
            message: format!(
                "the IR intensities, dipole derivatives and polarizability were declined; the \
                 Hessian and frequencies are unaffected (resolved.record.response.ir_declined): {f}"
            ),
            path: "freq.ir",
        }));
        warnings.extend(self.closed_shell_warning(r));
        warnings.extend(non_aufbau_warning(r));
        warnings.extend(plateau_warning(r));
        warnings.extend(self.cx.plan.convergence_warning(gs.purpose()));
        if let Some(m) = resolved
            .record
            .response
            .as_ref()
            .and_then(|x| x.mode_check.as_ref())
            .filter(|m| m.flagged)
        {
            warnings.push(linus_core::Warning {
                code: "mode_check_flagged",
                message: format!(
                    "the UHF/UKS solution has an orbital-Hessian eigenvalue of {:+.1e} Eh (below \
                     {:.0e}: a degenerate or unstable open shell); the analytic Hessian and \
                     frequencies are not well defined at this SCF solution \
                     (resolved.record.response.mode_check)",
                    m.eigenvalue, m.threshold
                ),
                path: "scf.stability_hessian_check",
            });
        }
        let mol = &self.cx.mol;
        let structure = StructureOut {
            symbols: mol
                .z
                .iter()
                .map(|&z| linus_core::elements::z_to_symbol(z).unwrap_or("?"))
                .collect(),
            geometry: mol.positions.clone(),
            charge: mol.charge,
            multiplicity: mol.multiplicity,
            ecp_core_electrons: mol.ecp_core_electrons(),
        };
        let mut files = BTreeMap::new();
        if self.cx.plan.molden() != MoldenForm::Off {
            files.insert(MOLDEN_FILE.to_string(), self.molden(r, &structure));
        }
        Report {
            schema_version: SCHEMA_VERSION,
            program: "linus",
            version: crate::version(),
            numerics_epoch: crate::NUMERICS_EPOCH,
            calctype,
            success: r.converged,
            energy: r.energy,
            components,
            gradient: d.gradient,
            gradient_parts: d.gradient_parts,
            hessian: d.hessian,
            cphf_iterations: d.cphf_iterations,
            frequencies: d.frequencies,
            solvation: solv.cloned(),
            converged: r.converged,
            iterations: r.iterations,
            scf: ScfSummary::of(r),
            restricted: r.restricted,
            natoms: mol.natom(),
            nbf: self.cx.basis.nbf,
            nmo: r.nmo,
            nalpha: r.nalpha,
            nbeta: r.nbeta,
            s2: r.s2,
            orbital_energies_alpha: r.eps_a.clone(),
            orbital_energies_beta: r.eps_b.clone(),
            occupations_alpha: r.occ_a.clone(),
            occupations_beta: r.occ_b.clone(),
            homo_lumo_gap: homo_lumo_gap(r),
            dipole,
            grid: self.is_dft().then(|| gs.grid().clone()),
            vv10: gs.vv10.as_ref().map(|v| Vv10Out {
                energy: v.last_energy,
                nlc_grid: self.cx.plan.nlc_grid(),
                grid_points: v.grid_points,
                active_points: v.active_points,
            }),
            structure,
            method: self.cx.plan.method().to_string(),
            basis: self.cx.plan.basis().to_string(),
            warnings,
            timings: timings(&self.timers),
            resources: self.timers.resources.iter().cloned().collect(),
            files,
            options: self.cx.plan.request().clone(),
            resolved,
            wavefunction: r.clone(),
            ao: AoMatrices::default(),
        }
    }

    /// The Molden file of the SCF's orbitals `r` (`output.molden`).
    fn molden(&self, r: &ScfResult, structure: &StructureOut) -> String {
        let sets = if r.restricted {
            let occ = r.occ_a.iter().zip(&r.occ_b).map(|(a, b)| a + b).collect();
            vec![molden::OrbitalSet {
                spin: "Alpha",
                c: &r.c_a,
                energies: &r.eps_a,
                occupations: occ,
            }]
        } else {
            vec![
                molden::OrbitalSet {
                    spin: "Alpha",
                    c: &r.c_a,
                    energies: &r.eps_a,
                    occupations: r.occ_a.clone(),
                },
                molden::OrbitalSet {
                    spin: "Beta",
                    c: &r.c_b,
                    energies: &r.eps_b,
                    occupations: r.occ_b.clone(),
                },
            ]
        };
        molden::write(
            &structure.symbols,
            &self.cx.mol.z,
            &structure.ecp_core_electrons,
            &structure.geometry,
            &self.cx.basis,
            self.cx.plan.molden(),
            &sets,
        )
    }

    /// The given settings whose effect depends on the calctype (they take effect in a declared
    /// calctype, not in an energy), and whether each took effect in a run of `run`
    /// (`record.settings_applied`); `open_shell`: the determinant is unrestricted.
    fn settings_applied(&self, run: CalcTypes, open_shell: bool) -> BTreeMap<String, bool> {
        use crate::driver::settings::Applies;
        // a run of every declared calctype (a one-shot run) applies every setting it accepted
        if run == self.cx.calctypes {
            return BTreeMap::new();
        }
        let effect = |a: Applies| !matches!(a, Applies::No(_));
        let declared = self.cx.settings_applies(self.cx.calctypes, open_shell);
        let energy = self
            .cx
            .settings_applies(CalcTypes::of(CalcType::Energy, false), open_shell);
        let this = self.cx.settings_applies(run, open_shell);
        self.cx
            .plan
            .given()
            .into_iter()
            .filter(|p| effect(declared(p)) && !effect(energy(p)))
            .map(|p| {
                let applied = effect(this(&p));
                (p, applied)
            })
            .collect()
    }

    /// What the calculation used: the calctypes the run covered, its settings with every `auto`
    /// resolved and why, and its record.
    fn resolved(
        &self,
        gs: &GroundState,
        calctypes: CalcTypes,
        response: Option<&ResolvedCphf>,
        pol_max_residual: Option<f64>,
        ir_declined: Option<&ResponseFailure>,
    ) -> Resolved {
        const GIB: f64 = 1073741824.0;
        let resource = |k: &str| self.timers.resources.iter().find(|n| n.0 == k).map(|n| n.1);
        let r = &gs.scf;
        let dft = self.is_dft();
        let mem = self.cx.plan.memory();
        let df = gs.df.clone();
        let budget = gs.setup.memory.df_budget;
        let (settings, reasons) = self.cx.resolved_settings(&RunFacts {
            calctypes,
            purpose: gs.purpose(),
            scf_grid: gs.grid(),
            restricted: r.restricted,
            convergence: r.convergence,
            probes: self.cx.record.probes(),
            df: df.as_ref(),
            response,
            solvent_scf: gs.solvation.as_ref().map(|s| s.scf),
            df_j: self.cx.record.scf_df_j().map(|c| (c.engine, c.reason)),
            nuclear_attraction: self.cx.record.nuclear_attraction(),
        });
        let targets = self.cx.plan.targets();
        let accuracy = targets.any().then(|| {
            let top = if calctypes.hessian {
                CalcType::Hessian
            } else if calctypes.gradient {
                CalcType::Gradient
            } else {
                CalcType::Energy
            };
            let gn = r.history.last().map_or(f64::NAN, |it| it.gnorm_pyscf);
            targets.record(
                top,
                r.convergence.e_tol,
                r.convergence.g_tol,
                gn,
                response.map(|h| (h.cphf_tol, h.max_residual)),
            )
        });
        let response = response.map(|h| ResolvedResponse {
            max_residual: h.max_residual,
            single_precision: h.single_precision,
            two_level: h.two_level,
            pol_max_residual,
            ir_declined: ir_declined.map(ResponseFailureOut::from),
            mode_check: r.mode_check.map(|m| ResolvedModeCheck {
                eigenvalue: m.eigenvalue,
                residual: m.residual,
                fock_builds: m.fock_builds,
                seconds: m.seconds,
                threshold: linus_scf::MODE_CHECK_THRESHOLD,
                flagged: m.flagged(
                    self.cx.mol.z.len(),
                    !dft && self.cx.plan.solvent().is_none(),
                ),
                irreps: m.irreps,
            }),
        });
        let probes = self.cx.record.probes();
        Resolved {
            calctypes: calctypes.names(),
            declared_calctypes: self.cx.calctypes.names(),
            settings,
            reasons,
            record: RunRecordOut {
                warm_start: gs.guess.warm_start(),
                settings_applied: self.settings_applied(calctypes, !r.restricted),
                purpose: gs.purpose(),
                dft,
                scf: ResolvedScf {
                    guess: gs.guess.clone(),
                    dropped_functions: self.cx.basis.nbf.saturating_sub(r.nmo),
                    two_cycle_at: r.two_cycle_at,
                    plateau: r.plateau,
                    escalation: r.escalation,
                    stability_probe: if r.restricted {
                        0
                    } else if r.probe_trigger.is_some_and(|t| t.fired()) {
                        linus_scf::STABILITY_PROBE_AUTO
                    } else {
                        probes.map_or(0, |p| p.0)
                    },
                    probe_trigger: r.probe_trigger,
                    // the SCF says `off` for no products; why they are none is the driver's to say
                    closed_shell_probe: r.closed_shell_probe.map(|mut p| {
                        if p.skipped == Some(ProbeSkip::Off) {
                            p.skipped = Some(self.cx.closed_shell_probe_off());
                        }
                        p
                    }),
                    non_aufbau: r.non_aufbau.clone(),
                },
                memory: ResolvedMemory {
                    source: mem.source(),
                    limit_gib: mem.bytes().map(|b| b as f64 / GIB),
                    df_tensor_budget_gib: df.as_ref().map(|d| d.budget_bytes as f64 / GIB),
                    df_tensor_budget_source: df.as_ref().map(|_| {
                        if budget.by_available {
                            "available memory"
                        } else {
                            mem.source()
                        }
                    }),
                    available_gib: budget.available.map(|b| b as f64 / GIB),
                    reservations: gs.setup.memory.reserved.clone(),
                    hbar_gib: resource("hess_df_hbar_mb").map(|mb| mb * 1e6 / GIB),
                    xc_kernel_gib: mem.xc_kernel_mb() as f64 / 1024.0,
                    response_slot_gib: mem.response_slot_bytes() as f64 / GIB,
                },
                df,
                response,
                solvent: self.cx.plan.solvent().map(|so| ResolvedSolvent {
                    eps: gs.solvation.as_ref().map(|s| s.eps),
                    eps_table: so
                        .eps()
                        .is_none()
                        .then(|| so.name().as_deref().and_then(linus_solvent::db::lookup))
                        .flatten()
                        .and_then(linus_solvent::db::table_of)
                        .map(|p| p.display().to_string()),
                    // the default of the CDS term itself (crate::solvent::scf)
                    cds_contact_smoothing: crate::solvent::scf::cds_config(
                        so,
                        gs.purpose(),
                        linus_solvent::smd::CdsOrder::Energy,
                    )
                    .contact_smoothing,
                }),
                backend: self.cx.backend_used(),
                debug_env: crate::driver::options::debug_env(),
                accuracy,
            },
        }
    }

    /// A failed calculation: the error, its kind and, for an SCF that stopped without
    /// converging or failed numerically, its last iterate (`scf`, `energy`, `timings`, and the
    /// report of it as `result`); for a calculation stopped on request after its SCF converged,
    /// the report of that SCF with the gradient and Hessian it reached.
    pub fn failure(&mut self, err: &LinusError, calctype: CalcType) -> Failure {
        let mut f = Failure {
            schema_version: SCHEMA_VERSION,
            program: "linus",
            version: crate::version(),
            calctype,
            success: false,
            error: err.to_string(),
            error_kind: err.kind(),
            error_code: err.code(),
            error_details: error_details(err),
            scf: None,
            energy: None,
            timings: None,
            response: None,
            logs: self.take_logs(),
            result: None,
        };
        match err {
            LinusError::NotConverged(_) | LinusError::Numerical(_) => {
                if let Some(gs) = self.unconverged.clone() {
                    f.scf = Some(ScfSummary::of(&gs.scf));
                    f.energy = Some(gs.scf.energy);
                    f.timings = Some(timings(&self.timers));
                    f.result = Some(Box::new(self.scf_report(&gs, calctype)));
                }
            }
            LinusError::ResponseNotConverged(rf) => {
                f.response = Some(ResponseFailureOut::from(&**rf));
                f.timings = Some(timings(&self.timers));
                f.result = self.unconverged_report(calctype).map(Box::new);
            }
            LinusError::Stopped(s) => {
                f.timings = Some(timings(&self.timers));
                // in the SCF's iterations: its last iterate; after it: this request's converged
                // SCF (a stop before the SCF reached nothing)
                let gs = match (s.in_scf(), s.stage) {
                    (true, _) => self.unconverged.clone(),
                    (false, "scf") => None,
                    (false, _) => self.ground_here().cloned(),
                };
                if let Some(gs) = gs {
                    if s.in_scf() {
                        f.scf = Some(ScfSummary::of(&gs.scf));
                    }
                    f.energy = Some(gs.scf.energy);
                    let d = Derivatives {
                        gradient: s.gradient.clone(),
                        hessian: s.hessian.clone(),
                        ..Derivatives::default()
                    };
                    f.result = Some(Box::new(self.report(&gs, calctype, d)));
                }
            }
            _ => {}
        }
        f
    }

    /// The report of a calculation whose CPHF/CPKS stopped above its tolerance: the converged
    /// SCF with what the response reached (the Hessian, or the polarizability's iterations).
    fn unconverged_report(&mut self, calctype: CalcType) -> Option<Report> {
        let gs = self.ground_here()?.clone();
        let mut d = Derivatives::default();
        match self.unconverged_response.clone()? {
            UnconvergedResponse::Hessian(h) => {
                d.gradient = Some(h.gradient);
                d.cphf_iterations = Some(h.cphf_iterations);
                d.response = Some(h.response);
                d.hessian = Some(h.hessian);
            }
            UnconvergedResponse::Polarizability(p) => d.pol_max_residual = Some(p.max_residual),
        }
        Some(self.report(&gs, calctype, d))
    }
}

/// The seconds per phase.
fn timings(t: &Timers) -> BTreeMap<String, f64> {
    t.entries.iter().map(|(n, s, _)| (n.clone(), *s)).collect()
}

/// A grid for people (`pyscf:3`, `linus:3`, ...).
pub fn grid_label(g: &GridChoice) -> String {
    g.label()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Structure;

    fn water_pbe() -> Session {
        let mol = Structure::from_xyz_str(
            "3\n\nO 0 0 0.1173\nH 0 0.7572 -0.4692\nH 0 -0.7572 -0.4692\n",
            0,
            1,
        )
        .unwrap();
        let opts = CalcOptions {
            method: Some("pbe".into()),
            basis: Some("sto-3g".into()),
            ..Default::default()
        };
        Session::new(mol, opts).unwrap()
    }

    /// The typed report serializes to the result JSON's keys; frequency results sit at the top
    /// level; a failure carries its SCF and the report of it.
    #[test]
    fn report_and_failure_json() {
        let mut c = water_pbe();
        let r = c.run(CalcType::Gradient).unwrap();
        let v = r.to_json();
        let keys: Vec<&str> = v.as_object().unwrap().keys().map(|k| k.as_str()).collect();
        assert_eq!(
            keys,
            [
                "basis",
                "calctype",
                "components",
                "converged",
                "dipole",
                "energy",
                "gradient",
                "gradient_parts",
                "grid",
                "homo_lumo_gap",
                "iterations",
                "method",
                "nalpha",
                "natoms",
                "nbeta",
                "nbf",
                "nmo",
                "numerics_epoch",
                "occupations_alpha",
                "occupations_beta",
                "options",
                "orbital_energies_alpha",
                "orbital_energies_beta",
                "program",
                "resolved",
                "resources",
                "restricted",
                "s2",
                "scf",
                "schema_version",
                "structure",
                "success",
                "timings",
                "version",
                "warnings"
            ]
        );
        assert_eq!(v["calctype"], "gradient");
        assert_eq!(v["resolved"]["record"]["scf"]["dropped_functions"], 0);
        assert_eq!(v["numerics_epoch"], crate::NUMERICS_EPOCH);
        // the default J/K: RI-JK with the calculation's auxiliary basis (decision 6)
        assert_eq!(v["resolved"]["settings"]["jk"]["kind"], "df");
        assert_eq!(
            v["resolved"]["settings"]["jk"]["aux"],
            "def2-universal-jkfit"
        );
        assert_eq!(v["resolved"]["record"]["backend"]["xc"], "cpu");
        assert!(v["resolved"]["record"]["response"].is_null() && v["timings"]["total"].is_number());
        assert!(v["resolved"]["settings"].get("response").is_none() && v["warnings"] == json!([]));
        let f = c.run(CalcType::Hessian).unwrap();
        let v = f.to_json();
        for k in [
            "vibrational_analysis",
            "thermochemistry",
            "hessian",
            "cphf_iterations",
        ] {
            assert!(v.get(k).is_some(), "{k}");
        }
        assert!(v.get("gradient_parts").is_none() && v.get("dipole_derivatives").is_none());
        assert_eq!(v["resolved"]["settings"]["grid"]["kernel"], "linus:1");
        assert_eq!(v["resolved"]["settings"]["response"]["tol"], 1e-6);
        // a reason only for a key that takes effect (no stability probe for RKS)
        assert!(v["resolved"]["reasons"].get("response.tol").is_some());
        assert!(v["resolved"]["reasons"]
            .get("scf.stability_probe")
            .is_none());
        assert_eq!(v["resolved"]["settings"]["grid"]["xc"], "linus:3");
        assert_eq!(v["calctype"], "hessian");
        // an SCF that does not converge (max_iter is the whole budget: one iteration, too few for
        // the escalation, which the message says)
        c.cx.plan.request_mut().scf.max_iter = Some(1);
        c.reset_all();
        let e = c.run(CalcType::Energy).unwrap_err();
        assert!(e.to_string().contains("no escalation: max_iter 1"), "{e}");
        let fl = c.failure(&e, CalcType::Energy);
        let fj = fl.to_json();
        assert_eq!(fj["error_kind"], "convergence");
        assert_eq!(fj["error_code"], "scf_not_converged");
        assert_eq!(fj["schema_version"], SCHEMA_VERSION);
        assert_eq!(fj["scf"]["iterations"], 1);
        assert!(fj["energy"].is_number() && fj.get("result").is_none());
        let rep = fl.result.expect("the unconverged SCF's report");
        assert!(!rep.success && !rep.converged);
    }

    /// A CPHF/CPKS that stops above its tolerance (single-precision response J/K against a
    /// tolerance below its noise): a convergence failure with the response block and the report
    /// of the Hessian it reached.
    #[test]
    fn response_failure_json() {
        let mol = Structure::from_xyz_str(
            "3\n\nO 0 0 0.1173\nH 0 0.7572 -0.4692\nH 0 -0.7572 -0.4692\n",
            0,
            1,
        )
        .unwrap();
        let opts = CalcOptions::from_json(json!({"basis": "def2-svp", "jk": "df",
            "response": {"tol": 1e-12}, "precision": {"response": "fp32"}}))
        .unwrap();
        let mut c = Session::new(mol, opts).unwrap();
        let e = c.run(CalcType::Hessian).unwrap_err();
        assert!(matches!(e, LinusError::ResponseNotConverged(_)), "{e}");
        let f = c.failure(&e, CalcType::Hessian);
        let j = f.to_json();
        assert_eq!(j["error_kind"], "convergence");
        assert_eq!(j["error_code"], "response_not_converged");
        assert_eq!(j["response"]["what"], "analytic Hessian");
        assert!(j["response"]["max_residual"].as_f64().unwrap() > 1e-11);
        let rep = f.result.expect("the report of the unconverged response");
        assert!(rep.success && rep.hessian.is_some());
        assert!(rep.resolved.record.response.unwrap().max_residual > 1e-11);
    }

    /// An IR run whose electric-field response has no finite value keeps its Hessian and
    /// frequencies and declines the IR part: the CH radical (UHF/def2-SVP, `extreme`) steps from
    /// its symmetric solution to a lower broken-symmetry one, whose rotation about the axis is a
    /// zero mode the field couples to, so the field CPHF ends `exhausted` or `indefinite`. The
    /// result has the frequencies, no IR intensities, dipole derivatives or polarizability, the
    /// warning `ir_declined` and the failure in `resolved.record.response.ir_declined`.
    #[test]
    fn a_singular_field_response_declines_only_the_ir_part() {
        let mol = Structure::from_xyz_str("2\n\nC 0 0 0\nH 0 0 1.12\n", 0, 2).unwrap();
        let opts = CalcOptions::from_json(json!({"method": "hf", "basis": "def2-svp",
            "jk": "exact", "scf": {"convergence": "extreme"}, "response": {"tol": 1e-9},
            "freq": {"ir": true}, "threads": 4}))
        .unwrap();
        let mut c = Session::new(mol, opts).unwrap();
        let rep = c.run(CalcType::Hessian).unwrap();
        let j = serde_json::to_value(&rep).unwrap();
        assert!(rep.success && rep.hessian.is_some());
        let va = &j["vibrational_analysis"];
        assert_eq!(
            va["frequencies"].as_array().map(|f| f.len()),
            Some(1),
            "{va}"
        );
        assert!(va["ir_intensities"].is_null(), "{va}");
        assert!(j.get("dipole_derivatives").is_none() && j.get("polarizability").is_none());
        let w: Vec<&str> = rep.warnings.iter().map(|w| w.code).collect();
        assert!(w.contains(&"ir_declined"), "{w:?}");
        let d = &j["resolved"]["record"]["response"]["ir_declined"];
        assert_eq!(d["what"], "polarizability");
        assert!(
            d["end"] == "exhausted" || d["end"] == "indefinite",
            "a singular or indefinite response: {d}"
        );
        assert!(d["max_residual"].as_f64().unwrap() > d["tol"].as_f64().unwrap());
        assert!(c.unconverged_response().is_none());
    }
}
