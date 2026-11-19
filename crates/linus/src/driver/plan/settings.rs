//! Which settings a calculation uses, and the settings of a run as `resolved.settings` reports
//! them (decision D41).
//!
//! Every setting, by its dotted path, applies to a calculation, applies only when a run-time
//! condition fires (contingent), or cannot apply, with the reason (`applies`). The calctype
//! half is the settings table's (`crate::driver::schema`, each setting's lowest calctype:
//! `response.*` for Hessians, `grid.response` for gradients); the conditions on the method, the
//! J/K method, the determinant and where the families of kernels run are here. The table is
//! evaluated for a `Scope`.
//!
//! Resource and output settings (`threads`, `memory_gib`, `basis_dir`, `output.*`) always apply:
//! they bound or describe a run rather than change what it computes (the schema's resource and
//! output concerns, which the qc* calculation store also leaves out of its cache key). The
//! model's settings (`method`, `basis`, `cartesian`, `reference`, `dispersion`, `solvent.*`)
//! always apply too: they say what is computed, also where they equal the default.
//!
//! `resolved.settings` (`CalcContext::resolved_settings`) is the canonical settings tree of a
//! run: only the settings that apply, every `auto` as the value it took, grids in their string
//! forms (`"linus:3"`). It is itself a valid request: run again with it, the calculation is the
//! same. `resolved.reasons` says why each `auto` took its value. `auto` stays where it is a mode
//! rather than a choice among the values (`precision.fp32`'s validated uses, `scf.stability`),
//! where the run decides per step (the device response's `precision.response`, D37's costing),
//! with the reason. `auto` settings decided per molecule by the
//! preset or the machine (`memory_gib`, `device_memory_gib`, `backend.kernel_cache`, the
//! solvent's Lebedev grids and radii scale) are left out.

use crate::derivatives::hessian::ResolvedCphf;
use crate::driver::backend::{Family, Placement};
use crate::driver::options::{
    nlc_grid_for, AuxBasis, BackendKind, DfLongRange, DfPrecision, DispersionSetting, ExactStorage,
    Fp32Mode, Fp64Mode, FreqAnalysis, IterationAux, Reference, ResponsePrecision, ScfLoop,
    Stability, Thermo,
};
use crate::driver::report::CalcType;
use crate::driver::schema::{self, Concern, Needs};
use crate::driver::{
    CalcContext, CalcOptions, GridChoice, JkMethod, LinusError, MethodInfo, Purpose,
};
use linus_base::convergence::Convergence;
use linus_jk::df::{DfLongRangeSource, ResolvedDf};
use linus_solvent::PcmScfMode;
use serde_json::{Map, Value};
use std::collections::BTreeMap;

/// The calctypes a scope covers beyond an energy's SCF.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CalcTypes {
    /// Gradients (and the Hessians that compute one).
    pub gradient: bool,
    /// Hessians (with their harmonic analysis).
    pub hessian: bool,
    /// Hessians with IR intensities (the polarizability's electric-field response).
    pub ir: bool,
}

impl CalcTypes {
    /// Every calctype: a session that may run anything.
    pub const ALL: CalcTypes = CalcTypes {
        gradient: true,
        hessian: true,
        ir: true,
    };

    /// What a calculation of `calctype` computes: the calctype and the lower derivatives it
    /// computes on the way; `ir`: a Hessian with IR intensities (`freq.ir`).
    pub fn of(calctype: CalcType, ir: bool) -> CalcTypes {
        CalcTypes {
            gradient: calctype != CalcType::Energy,
            hessian: calctype == CalcType::Hessian,
            ir: calctype == CalcType::Hessian && ir,
        }
    }

    /// Whether these calctypes include every one of `other`'s.
    pub fn covers(self, other: CalcTypes) -> bool {
        (self.gradient || !other.gradient)
            && (self.hessian || !other.hessian)
            && (self.ir || !other.ir)
    }

    /// Whether a setting that needs `needs` can take effect in these calctypes.
    pub fn meet(self, needs: Needs) -> bool {
        match needs {
            Needs::Energy => true,
            Needs::Gradient => self.gradient,
            Needs::Hessian => self.hessian,
            Needs::Ir => self.ir,
        }
    }

    /// The calctypes covered, as `resolved.calctypes` names them.
    pub fn names(self) -> Vec<&'static str> {
        let mut v = vec!["energy"];
        if self.gradient {
            v.push("gradient");
        }
        if self.hessian {
            v.push("hessian");
        }
        if self.ir {
            v.push("ir");
        }
        v
    }
}

/// Whether a setting takes effect.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum Applies {
    /// The calculation reads it.
    Used,
    /// It takes effect when a run-time condition fires (the condition).
    Contingent(&'static str),
    /// It cannot take effect here, and why.
    No(String),
}

impl Applies {
    fn takes_effect(&self) -> bool {
        !matches!(self, Applies::No(_))
    }
}

/// What a calculation's settings are evaluated against.
pub(crate) struct Scope<'a> {
    pub opts: &'a CalcOptions,
    pub method: &'a MethodInfo,
    pub calctypes: CalcTypes,
    /// The determinant is unrestricted (UHF/UKS).
    pub open_shell: bool,
    /// A device is open.
    pub device: bool,
    /// The families of kernels planned on the device.
    pub on_device: &'a dyn Fn(Family) -> bool,
}

/// Why a setting that needs `needs` cannot apply to fewer calctypes.
fn needs_text(path: &str, needs: Needs) -> String {
    match needs {
        Needs::Energy => format!("{path} applies to every calctype"),
        Needs::Gradient => format!("{path} is for gradients and Hessians"),
        Needs::Hessian => format!("{path} is for Hessians"),
        Needs::Ir => format!("{path} is for IR intensities (freq.ir)"),
    }
}

/// Whether the setting at `path` (dotted: `scf.level_shift`, `jk.aux`) takes effect in `s`.
pub(crate) fn applies(path: &str, s: &Scope<'_>) -> Applies {
    use Applies::*;
    let Some(meta) = schema::meta(path) else {
        // an optional group itself (the solvent: on)
        return match path {
            "solvent" => Used,
            p => No(format!("{p} is not a setting")),
        };
    };
    if !s.calctypes.meet(meta.needs) {
        return No(needs_text(path, meta.needs));
    }
    if path == "backend.devices" {
        // a resource, but only of a GPU backend
        return match s.opts.backend.kind() {
            _ if s.device => Used,
            BackendKind::Cpu => No("the cpu backend takes no device ids".into()),
            _ => Contingent("a device opens"),
        };
    }
    if matches!(meta.concern, Concern::Resource | Concern::Output) {
        return Used;
    }
    let o = s.opts;
    let no = |why: &str| No(why.to_string());
    let dft = s.method.is_dft();
    let jk = o.jk_method();
    let jk_df = matches!(jk, JkMethod::Df { .. });
    let jk_snk = matches!(jk, JkMethod::RijSnk { .. });
    let jk_exact = jk == JkMethod::Exact;
    let exact_direct = jk_exact && o.jk.exact_storage().resolved() == ExactStorage::Direct;
    let scf_jk_cpu = !(s.on_device)(Family::Jk);
    let stability_off = o.scf.stability() == Stability::Off;
    let thermo = o.freq.thermo();
    match path {
        // the model
        "method" | "basis" | "cartesian" | "reference" | "dispersion" => Used,
        p if p.starts_with("solvent.") => Used,
        // the J/K method
        "jk.kind" => Used,
        "jk.aux" if jk_df || jk_snk => Used,
        "jk.aux" => no("jk exact has no auxiliary basis"),
        "jk.k_grid" if jk_snk => Used,
        "jk.k_grid" => no("jk.k_grid is the seminumerical exchange's grid (jk rij_snk)"),
        "jk.exact_storage" if jk_exact => Used,
        "jk.exact_storage" => no("jk.exact_storage is for jk exact"),
        "jk.screening" if jk_exact => Used,
        "jk.screening" => no("jk.screening screens the exact four-center integrals (jk exact)"),
        "jk.df_storage" if !jk_df => no("jk.df_storage is the exchange tensor's storage (jk df)"),
        "jk.df_storage" if s.method.exchange.is_none() => {
            no("without exact exchange the fit stores the raw integrals (J only)")
        }
        "jk.df_storage" if scf_jk_cpu => Used,
        "jk.df_storage" => Contingent("the SCF's J/K falls back to the CPU"),
        "jk.df_long_range" if !jk_df => no("jk.df_long_range is for jk df"),
        "jk.df_long_range" if s.method.omega().is_none() => {
            no("jk.df_long_range is for range-separated hybrids")
        }
        "jk.df_long_range" if scf_jk_cpu => Used,
        "jk.df_long_range" => Contingent("the SCF's J/K falls back to the CPU"),
        "jk.incremental" if exact_direct || jk_snk => Used,
        "jk.incremental" => no(
            "incremental builds are for integral-direct exact J/K and the seminumerical exchange",
        ),
        "jk.incremental_reset" if exact_direct => {
            if o.jk.incremental() {
                Used
            } else {
                no("jk.incremental is off")
            }
        }
        "jk.incremental_reset" => no("incremental Fock builds are for integral-direct exact J/K"),
        "jk.grid_switch" | "jk.grid_switch_shift" if jk_snk => {
            Contingent("a COSX grid scheme moves to its next grid")
        }
        "jk.grid_switch" | "jk.grid_switch_shift" => {
            no("grid switching is the seminumerical exchange's (jk rij_snk)")
        }
        // the hardware
        "backend.kind" => Used,
        // the arithmetic
        "precision.fp64" if s.device => Used,
        "precision.fp64" => no("nothing runs on a GPU"),
        "precision.fp32" if s.device => Used,
        // the CPU's FP32: the response exchange, where precision.response leaves it to auto's rule
        "precision.fp32" if jk_df && s.calctypes.hessian => match o.precision.response() {
            ResponsePrecision::Auto => {
                Contingent("auto's rule runs the response exchange single (a large exchange)")
            }
            _ => no("precision.response is set"),
        },
        "precision.fp32" => no(
            "FP32 runs only in a GPU's kernels and the RI response exchange of Hessians (jk df)",
        ),
        "precision.response" if jk_df => Used,
        "precision.response" => no("the response's exact exchange runs in double precision"),
        "backend.scf_loop" if (s.on_device)(Family::Jk) => Used,
        "backend.scf_loop" => no("the SCF's J/K runs on the CPU"),
        // grids
        "grid.xc" | "grid.kernel" if dft => Used,
        "grid.xc" | "grid.kernel" => no("Hartree-Fock has no XC quadrature"),
        "grid.nlc" if s.method.vv10 => Used,
        "grid.nlc" => no("grid.nlc is VV10's grid (-V functionals)"),
        "grid.response" if dft || jk_snk => Used,
        "grid.response" => no("grid.response is for the XC and seminumerical-exchange grids"),
        // the SCF
        "scf.convergence" | "scf.e_tol" | "scf.g_tol" | "scf.max_iter" | "scf.guess"
        | "scf.lindep" | "scf.diis_space" | "scf.diis_start" | "scf.damping" | "scf.adiis"
        | "scf.soscf" => Used,
        "scf.diis_drop" => Contingent("the DIIS space fills"),
        "scf.diis_unpin" => Contingent("an iteration's error rises past diis_unpin x the best"),
        "scf.level_shift" | "scf.level_shift_until" => {
            Contingent("while max |F_ai| is above scf.level_shift_until")
        }
        "scf.damping_until" if o.scf.damping() > 0.0 => {
            Contingent("while max |F_ai| is above scf.damping_until")
        }
        "scf.damping_until" => no("scf.damping is 0"),
        "scf.adiis_until" if o.scf.adiis() => Used,
        "scf.adiis_until" => no("scf.adiis is off"),
        "scf.soscf_from" if o.scf.soscf() => Used,
        "scf.soscf_from" => no("scf.soscf is off"),
        // stability off also turns the closed-shell probe off
        "scf.stability" if s.open_shell || stability_off => Used,
        "scf.stability" => no("the stability analysis is for UHF/UKS determinants"),
        p if p.starts_with("scf.stability_")
            && p != "scf.stability_hessian_check"
            && stability_off =>
        {
            no("scf.stability is off")
        }
        "scf.stability_saddle_check" | "scf.stability_max" if s.open_shell => {
            Contingent("a stability analysis runs")
        }
        "scf.stability_probe" if s.open_shell => Used,
        "scf.stability_saddle_check" | "scf.stability_max" | "scf.stability_probe" => {
            no("the stability analysis is for UHF/UKS determinants")
        }
        "scf.stability_closed_probe" | "scf.stability_closed_gap" if !s.open_shell => Used,
        "scf.stability_closed_probe" => no("the closed-shell probe is for RHF/RKS determinants"),
        "scf.stability_closed_gap" => Contingent("the gap trigger at a later geometry"),
        "scf.stability_hessian_check" if s.open_shell => Used,
        "scf.stability_hessian_check" => no("the zero-mode check is for open-shell Hessians"),
        "scf.two_cycle" => Contingent("the energy cycles between two determinants"),
        "scf.plateau_window" => Contingent("the SCF reaches scf.max_iter"),
        "scf.escalate" => Contingent("DIIS stagnates or the SCF spends 2/3 of scf.max_iter"),
        "scf.stagnation_window" if o.scf.escalate() => Contingent("DIIS stagnates"),
        "scf.stagnation_window" => no("scf.escalate is off"),
        // the response equations (Hessians: the table's calctype half)
        "response.tol" | "response.field_tol" => Used,
        "response.iteration_aux" if jk_df && !(s.on_device)(Family::ResponseJk) => Used,
        "response.iteration_aux" if jk_df => Contingent("the response J/K falls back to the CPU"),
        "response.iteration_aux" => no("two-level responses are for jk df"),
        // the Hessian's analysis (Hessians: the table's calctype half)
        "freq.analysis" | "freq.ir" => Used,
        p if p.starts_with("freq.") && o.freq.analysis() == FreqAnalysis::Off => {
            no("freq.analysis is off")
        }
        "freq.mass_table" if o.freq.masses.is_some() => no("freq.masses replace the table's"),
        "freq.thermo"
        | "freq.mass_table"
        | "freq.masses"
        | "freq.isotopes"
        | "freq.projection"
        | "freq.symmetry_number"
        | "freq.symmetry_tol"
        | "freq.codata" => Used,
        p if p.starts_with("freq.") && thermo == Thermo::Off => no("freq.thermo is off"),
        "freq.pressure" if o.freq.concentration().is_some() => {
            no("freq.concentration sets the standard state")
        }
        "freq.rotor_cutoff" | "freq.qrrho_alpha" | "freq.qrrho_bav"
            if !matches!(thermo, Thermo::Qrrho | Thermo::QrrhoHg) =>
        {
            no("the quasi-RRHO models' (freq.thermo qrrho, qrrho-hg)")
        }
        "freq.raise_below" if thermo != Thermo::Quasiharmonic => {
            no("the quasiharmonic model's (freq.thermo quasiharmonic)")
        }
        p if p.starts_with("freq.") => Used,
        // the accuracy targets set the SCF's (and a Hessian's response's) thresholds
        "accuracy.energy" | "accuracy.gradient" | "accuracy.hessian" => Used,
        p => No(format!("{p} has no applicability rule (a bug)")),
    }
}

/// What a run resolved that `resolved.settings` shows.
pub(crate) struct RunFacts<'a> {
    pub calctypes: CalcTypes,
    pub purpose: Purpose,
    /// The SCF's grid (`grid.xc` resolved for its purpose).
    pub scf_grid: &'a GridChoice,
    /// The SCF's determinant was restricted; its thresholds.
    pub restricted: bool,
    pub convergence: Convergence,
    /// The SCF's probes (stability, closed shell): the products it was given.
    pub probes: Option<(usize, usize)>,
    /// The SCF's density-fitting storage.
    pub df: Option<&'a ResolvedDf>,
    /// A Hessian's response.
    pub response: Option<&'a ResolvedCphf>,
    /// The implicit solvent's SCF mode.
    pub solvent_scf: Option<PcmScfMode>,
}

/// Sets `path` (dotted) in the object `v` to `x`, creating the groups on the way.
fn set(v: &mut Value, path: &str, x: Value) {
    let mut cur = v;
    let mut parts = path.split('.').peekable();
    while let Some(p) = parts.next() {
        let Value::Object(m) = cur else { return };
        if parts.peek().is_none() {
            m.insert(p.to_string(), x);
            return;
        }
        cur = m
            .entry(p.to_string())
            .or_insert_with(|| Value::Object(Map::new()));
    }
}

/// Removes `path` (dotted, at most two levels) from the object `v`.
fn unset(v: &mut Value, path: &str) {
    match path.split_once('.') {
        Some((g, k)) => {
            if let Some(Value::Object(m)) = v.get_mut(g) {
                m.remove(k);
            }
        }
        None => {
            if let Value::Object(m) = v {
                m.remove(path);
            }
        }
    }
}

/// A grid in the request's own string form (`"linus:3"`; a full specification as its object).
fn grid_value(g: &GridChoice) -> Value {
    serde_json::to_value(g).expect("a grid serializes")
}

impl CalcContext {
    /// Every given setting takes effect in a calculation of the declared calctypes (decision
    /// D41): else an input error (`LinusError::Setting`), `no_effect` when it takes effect in no
    /// calctype, `not_applicable` when only in calctypes not declared. Contingent settings pass.
    pub(crate) fn check_settings(&self) -> Result<(), LinusError> {
        let (na, nb) = self.mol.nalpha_nbeta().map_err(LinusError::Input)?;
        let open_shell = !self.plan.restricted().unwrap_or(na == nb);
        // backend gpu: the GPU's kernels must cover the basis sets of a declared calctype's
        // device families (D40)
        if let (Some(why), BackendKind::Gpu) = (self.plan.uncovered(), self.plan.backend().kind()) {
            if self.calctypes.gradient && matches!(self.plan.jk(), JkMethod::Df { .. }) {
                return Err(LinusError::Unsupported(format!(
                    "backend gpu: {why}, which the gradient's and Hessian's J/K need on the GPU; \
                     use backend \"auto\" (they run on the CPU then), a spherical basis set, or \
                     declare energies only"
                )));
            }
        }
        crate::driver::molden::check(&self.basis, self.plan.molden())?;
        let declared = self.settings_applies(self.calctypes, open_shell);
        let anywhere = self.settings_applies(CalcTypes::ALL, open_shell);
        for path in self.plan.given() {
            if let Applies::No(why) = declared(&path) {
                let code = match anywhere(&path) {
                    Applies::No(_) => "no_effect",
                    _ => "not_applicable",
                };
                return Err(LinusError::Setting {
                    code,
                    path,
                    message: format!("{why} (remove the setting)"),
                });
            }
        }
        Ok(())
    }

    /// A Hessian's analysis with the `freq` settings `call` (a front end's per-call keys, as a
    /// JSON object) over the session's own: each given key must take effect (decision D41).
    pub(crate) fn freq_options_with(
        &self,
        call: serde_json::Map<String, Value>,
    ) -> Result<linus_vib::FrequencyOptions, LinusError> {
        let mut merged = match serde_json::to_value(&self.plan.request.freq) {
            Ok(Value::Object(m)) => m,
            _ => Map::new(),
        };
        let keys: Vec<String> = call.keys().map(|k| schema::norm_name(k)).collect();
        merged.extend(call);
        let freq = crate::driver::options::FreqSettings::from_map(merged)?;
        let mut opts = self.plan.request.clone();
        opts.freq = freq;
        let (na, nb) = self.mol.nalpha_nbeta().map_err(LinusError::Input)?;
        let open_shell = !self.plan.restricted().unwrap_or(na == nb);
        let on_device = |k: Family| self.placement(k) == Some(Placement::Device);
        let scope = Scope {
            opts: &opts,
            method: &self.method,
            calctypes: CalcTypes::ALL,
            open_shell,
            device: self.backend.device().is_some(),
            on_device: &on_device,
        };
        for k in keys {
            let path = format!("freq.{k}");
            if let Applies::No(why) = applies(&path, &scope) {
                return Err(LinusError::Setting {
                    code: "no_effect",
                    path,
                    message: format!("{why} (remove the setting)"),
                });
            }
        }
        Ok(opts.freq_options())
    }

    /// The scope of this session's settings for `calctypes` and a determinant that is
    /// unrestricted or not (`open_shell`).
    pub(crate) fn settings_applies(
        &self,
        calctypes: CalcTypes,
        open_shell: bool,
    ) -> impl Fn(&str) -> Applies + '_ {
        move |path| {
            let on_device = |k: Family| self.placement(k) == Some(Placement::Device);
            applies(
                path,
                &Scope {
                    opts: &self.plan.request,
                    method: &self.method,
                    calctypes,
                    open_shell,
                    device: self.backend.device().is_some(),
                    on_device: &on_device,
                },
            )
        }
    }

    /// The settings of a run (`resolved.settings`) and why each `auto` took its value
    /// (`resolved.reasons`) ([module docs](self)).
    pub(crate) fn resolved_settings(
        &self,
        run: &RunFacts<'_>,
    ) -> (Value, BTreeMap<String, String>) {
        let o = &self.plan.request;
        let mut v = o.resolved_tree();
        let mut reasons = BTreeMap::new();
        let mut why = |path: &str, r: String| {
            reasons.insert(path.to_string(), r);
        };
        let purpose = run.purpose.name();
        let jk = self.plan.jk();
        // the J/K method
        if let Some(aux) = jk.aux() {
            set(
                &mut v,
                "jk.aux",
                Value::String(aux.for_purpose(run.purpose).into()),
            );
            if o.jk.aux() == AuxBasis::Auto {
                why(
                    "jk.aux",
                    match jk {
                        JkMethod::RijSnk { .. } => "auto: rij_snk's default".to_string(),
                        _ => format!("auto: the {purpose} default"),
                    },
                );
            }
        }
        if let Some(g) = jk.k_grid() {
            set(&mut v, "jk.k_grid", grid_value(g));
            if o.jk.k_grid() == GridChoice::Auto {
                why(
                    "jk.k_grid",
                    "auto: the ORCA-like COSX grids of DefGrid2".into(),
                );
            }
        }
        if *jk == JkMethod::Exact && o.jk.exact_storage() == ExactStorage::Auto {
            set(&mut v, "jk.exact_storage", Value::String("direct".into()));
            why("jk.exact_storage", "auto: integral-direct".into());
        }
        // the exchange tensor's format: auto is f64, decided before the run (a budget never
        // selects a compressed tensor; one that holds no f64 storage fails), on the CPU or on a
        // device's fallback to it
        if matches!(jk, JkMethod::Df { .. }) && o.jk.df_storage() == DfPrecision::Auto {
            set(&mut v, "jk.df_storage", Value::String("f64".into()));
            why(
                "jk.df_storage",
                "auto: f64 (the compressed formats only when set)".into(),
            );
        }
        if let Some(df) = run.df {
            let lr = match df.long_range {
                DfLongRangeSource::Stored => Some("stored"),
                DfLongRangeSource::OneSided => Some("one_sided"),
                DfLongRangeSource::None => None,
            };
            if let (DfLongRange::Auto, Some(lr)) = (o.jk.df_long_range(), lr) {
                set(&mut v, "jk.df_long_range", Value::String(lr.into()));
                why(
                    "jk.df_long_range",
                    "auto: B and B^omega stored when both fit the budget, else one-sided".into(),
                );
            }
        }
        // the hardware
        let device = self.backend.device();
        if o.backend.kind() == BackendKind::Auto {
            set(
                &mut v,
                "backend.kind",
                Value::String(if device.is_some() { "gpu" } else { "cpu" }.into()),
            );
            why(
                "backend.kind",
                match &self.backend.no_device() {
                    Some(no) => format!("auto: no device ({no})"),
                    None => "auto: a device opened".into(),
                },
            );
        }
        if let Some(d) = device {
            set(&mut v, "backend.devices", serde_json::json!([d.info.id]));
            // the device memory cap from the environment: the setting's default, made explicit
            if let Ok(Some((gib, true))) = self.plan.device_memory_cap() {
                set(&mut v, "device_memory_gib", serde_json::json!(gib));
                why("device_memory_gib", "auto: $LINUS_GPU_MEMORY_GIB".into());
            }
            if let (Fp64Mode::Auto, Some(dp)) = (o.precision.fp64(), self.plan.device()) {
                let p = &dp.policy;
                set(
                    &mut v,
                    "precision.fp64",
                    Value::String(p.precision.setting().into()),
                );
                why("precision.fp64", p.auto_note.clone().unwrap_or_default());
            }
            if o.backend.scf_loop() == ScfLoop::Auto {
                let on = self.placement(Family::ScfLoop) == Some(Placement::Device);
                set(
                    &mut v,
                    "backend.scf_loop",
                    Value::String(if on { "device" } else { "host" }.into()),
                );
                why(
                    "backend.scf_loop",
                    match self.placement(Family::ScfLoop) {
                        Some(Placement::Cpu(Some(r))) => format!("auto: host ({r})"),
                        _ => "auto: the device loop (D27, D31)".into(),
                    },
                );
            }
        }
        // grids
        let xc = o.grid.xc();
        if self.is_dft() {
            set(&mut v, "grid.xc", grid_value(run.scf_grid));
            if xc == GridChoice::Auto {
                why("grid.xc", format!("auto: the {purpose} default"));
            }
        }
        if self.method.vv10 {
            set(
                &mut v,
                "grid.nlc",
                grid_value(&nlc_grid_for(&o.grid.nlc(), &xc)),
            );
            if o.grid.nlc() == GridChoice::Auto {
                why("grid.nlc", "auto: by the XC grid's family".into());
            }
        }
        // the model
        let det = if run.restricted {
            "restricted"
        } else {
            "unrestricted"
        };
        set(&mut v, "reference", Value::String(det.into()));
        if o.reference() == Reference::Auto {
            let shell = if run.restricted { "closed" } else { "open" };
            why("reference", format!("auto: {det} for a {shell} shell"));
        }
        if o.dispersion() == DispersionSetting::Auto {
            let d = self.method.dispersion.map_or("none", |d| d.name());
            set(&mut v, "dispersion", Value::String(d.into()));
            why("dispersion", "auto: implied by the method's name".into());
        }
        // the SCF
        set(
            &mut v,
            "scf.e_tol",
            serde_json::json!(run.convergence.e_tol),
        );
        set(
            &mut v,
            "scf.g_tol",
            serde_json::json!(run.convergence.g_tol),
        );
        let targeted = o.accuracy.targets().for_purpose(run.purpose).any();
        if o.scf.convergence().is_auto() {
            // the tolerances carry it
            unset(&mut v, "scf.convergence");
            if !targeted && (o.scf.e_tol().is_auto() || o.scf.g_tol().is_auto()) {
                why("scf.e_tol", format!("auto: the {purpose} default"));
                why("scf.g_tol", format!("auto: the {purpose} default"));
            }
        }
        if targeted {
            let why_t = "the accuracy targets (record.accuracy), tightening a given value";
            why("scf.e_tol", why_t.into());
            why("scf.g_tol", why_t.into());
        }
        if let Some((stability, closed)) = run.probes {
            if o.scf.stability_probe().is_auto() {
                set(&mut v, "scf.stability_probe", serde_json::json!(stability));
                why(
                    "scf.stability_probe",
                    "auto: at a lineage's first SCF, else off unless the gap trigger fires".into(),
                );
            }
            if o.scf.stability_closed_probe().is_auto() {
                set(
                    &mut v,
                    "scf.stability_closed_probe",
                    serde_json::json!(closed),
                );
                why("scf.stability_closed_probe", "auto: Kohn-Sham DFT at a lineage's first SCF, else off unless the gap trigger fires".into());
            }
        }
        // the response
        if let Some(r) = run.response {
            if o.accuracy.hessian().is_some() {
                set(&mut v, "response.tol", serde_json::json!(r.cphf_tol));
                why(
                    "response.tol",
                    "the accuracy target accuracy.hessian (record.accuracy), tightening a given value"
                        .into(),
                );
            } else if o.response.tol().is_auto() {
                set(&mut v, "response.tol", serde_json::json!(r.cphf_tol));
                why(
                    "response.tol",
                    if r.two_level {
                        "auto: 1e-5 for a two-level response"
                    } else {
                        "auto: 1e-6, the Hessian's error quadratic in it"
                    }
                    .into(),
                );
            }
            if o.precision.response() == ResponsePrecision::Auto
                && self.placement(Family::ResponseJk) != Some(Placement::Device)
            {
                let p = if r.single_precision { "fp32" } else { "fp64" };
                set(&mut v, "precision.response", Value::String(p.into()));
                why(
                    "precision.response",
                    if o.precision.fp32() == Fp32Mode::Off {
                        "auto: fp64 (precision.fp32 off)".into()
                    } else {
                        format!("auto: {p}")
                    },
                );
                // without a device the response's precision was precision.fp32's one effect: with
                // it explicit, fp32 has none left (and given, it would be an input error, D41)
                if device.is_none() {
                    if let Some(Value::Object(m)) = v.get_mut("precision") {
                        m.remove("fp32");
                    }
                }
            }
            if o.response.iteration_aux() == IterationAux::Auto {
                let a = r.iteration_aux.clone().unwrap_or_else(|| "none".into());
                set(&mut v, "response.iteration_aux", Value::String(a));
                why(
                    "response.iteration_aux",
                    "auto: two levels for a large single-precision response".into(),
                );
            }
            if self.is_dft() {
                let g = r.grid.clone().unwrap_or_else(|| run.scf_grid.clone());
                set(&mut v, "grid.kernel", grid_value(&g));
                if o.grid.kernel() == GridChoice::Auto {
                    why(
                        "grid.kernel",
                        "auto: by the XC grid's family (linus grids: linus:1, meta-GGAs linus:2; \
                         PySCF, ORCA and Psi4 grids: the XC grid)"
                            .into(),
                    );
                }
            }
        }
        if let (Some(mode), Some(so)) = (run.solvent_scf, &o.solvent) {
            if mode != PcmScfMode::Df && so.scf_terms().is_auto() {
                set(
                    &mut v,
                    "solvent.scf_terms",
                    Value::String(mode.name().into()),
                );
                why(
                    "solvent.scf_terms",
                    "auto: predicted with density-fitted J, else exact".into(),
                );
            }
        }
        // only the keys that take effect, no unset leaves and no auto decided per molecule
        let applies = self.settings_applies(run.calctypes, !run.restricted);
        prune(&mut v, "", &applies);
        reasons.retain(|k, _| applies(k).takes_effect());
        (v, reasons)
    }
}

impl crate::driver::Session {
    /// A Hessian's analysis with the `freq` settings `call` (a front end's per-call keys, as a
    /// JSON object) over the session's own: each given key must take effect (decision D41).
    pub fn freq_options_with(
        &self,
        call: serde_json::Map<String, Value>,
    ) -> Result<linus_vib::FrequencyOptions, LinusError> {
        self.cx.freq_options_with(call)
    }
}

/// The settings that may stay `auto` in `resolved.settings`: the round trip's documented
/// exceptions ([module docs](self)); and `scf.stability`, whose `auto` is a fixed rule (the
/// probes, and a full analysis when one fires), not a choice the run makes.
const AUTO_KEPT: [&str; 3] = ["precision.fp32", "precision.response", "scf.stability"];

/// Removes from the settings object `v` (at `prefix`) the keys that do not take effect, the
/// unset (null) leaves and the `auto` leaves no rule resolved; groups left empty go too.
fn prune(v: &mut Value, prefix: &str, applies: &dyn Fn(&str) -> Applies) {
    let Value::Object(m) = v else { return };
    m.retain(|k, x| {
        let path = if prefix.is_empty() {
            k.clone()
        } else {
            format!("{prefix}.{k}")
        };
        let group = prefix.is_empty() && schema::table().groups.iter().any(|g| g.path == path);
        if group {
            prune(x, &path, applies);
            return !matches!(x, Value::Object(o) if o.is_empty());
        }
        if x.is_null() || (x.as_str() == Some("auto") && !AUTO_KEPT.contains(&path.as_str())) {
            return false;
        }
        applies(&path).takes_effect()
    });
}
