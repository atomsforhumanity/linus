//! The plan of a session: the one reader of its settings (package PLAN, decision D41).
//!
//! [`Plan`] holds the request (the normalized [`CalcOptions`]) as a private field: nothing
//! outside this module reads a setting. The rest of the crate asks the plan for what it needs,
//! resolved where a rule resolves it (the auxiliary basis and the grids of a purpose, the SCF's
//! thresholds, the exact integrals' storage): the SCF gets its [`ScfConfig`], the J/K builders
//! their configurations, the engines their options. The settings echo of a report
//! (`options`) and [`crate::driver::Session::opts`] are the request as given.
//!
//! [`settings`] holds which settings take effect for a calculation and the run's canonical
//! settings (`resolved.settings`). The test `plan_is_the_only_reader` (in `tests`) keeps the rule:
//! no file outside the plan reads the request, and the functions that take the whole settings are
//! the plan's resolvers, called from here.

pub mod estimate;
pub mod report;
pub mod settings;

use crate::driver::backend::{Device, Family, Placement};
use crate::driver::memory::DeviceMemoryPlan;
use crate::driver::options::{
    Auto, BackendSettings, CalcOptions, DfLongRange, DfPrecision, ExactStorage, JkMethod,
    ResponseSettings, ScfLoop,
};
use crate::driver::policy::{DevicePolicy, ResponseArith};
use crate::driver::{GridChoice, LinusError, MethodInfo, Purpose};
use crate::solvent::SolventSettings;
use crate::structure::Structure;
use crate::two_electron::ResponseUse;
use linus_base::convergence::Convergence;
use linus_base::memory::MemoryLimit;
use linus_ints::basis::BasisDef;
use linus_ints::operator::Operator;
use linus_jk::df::response::sym_op_usable;
use linus_jk::df::response::ITER_AUX_DEFAULT;
use linus_jk::df::response::TWO_LEVEL_WORK;
use linus_scf::{Guess, ScfAlgorithm};
use std::collections::BTreeMap;

/// The plan of a session (module docs).
#[derive(Clone, Debug)]
pub(crate) struct Plan {
    /// The settings as requested, normalized; read only here.
    request: CalcOptions,
    /// What the plan reads often, resolved once: the method's and basis set's names, the J/K
    /// method and the SCF's algorithm settings.
    method: String,
    basis: String,
    jk: JkMethod,
    algo: ScfAlgorithm,
    /// The device's part (None: no device).
    device: Option<DevicePlan>,
    /// Where every family of kernels runs (None: the method has none), before any setup.
    placement: BTreeMap<Family, Option<Placement>>,
    /// Why the GPU's generated integral kernels do not cover the basis sets (None: they do,
    /// or no device): the J/K's derivatives run on the CPU, the SCF's tensor takes the CPU's
    /// integrals (decision D40).
    uncovered: Option<String>,
}

/// What a response's decisions depend on ([`Plan::response_plan`]).
#[derive(Clone, Debug)]
pub(crate) struct ResponseSizes {
    /// Basis functions.
    pub nbf: usize,
    /// Auxiliary functions of the response J/K's fit.
    pub naux: usize,
    /// Occupied orbitals per channel.
    pub occ: Vec<usize>,
    /// The exchange operator of the response J/K's wK term (a range-separated hybrid's long
    /// range, [`linus_base::terms::exchange_parts`]).
    pub op: Option<Operator>,
}

impl ResponseSizes {
    /// The exchange work per density, n^2 o naux (the largest channel's).
    pub(crate) fn work(&self) -> f64 {
        self.occ
            .iter()
            .map(|&o| (self.nbf * self.nbf * o * self.naux) as f64)
            .fold(0.0, f64::max)
    }
}

/// A response's decisions ([`Plan::response_plan`]).
#[derive(Clone, Debug)]
pub(crate) struct ResponsePlan {
    /// The CPU's response J/K in single precision: `precision.response` for a Hessian (auto:
    /// [`linus_jk::df::response::auto_single`]), double for a polarizability.
    pub single: bool,
    /// The CPU's two-level response: the iteration J/K's auxiliary basis ("main": the main
    /// J/K's; None: one level), as `response.iteration_aux` and the size decide it
    /// ([`crate::driver::plan::iteration_aux`]).
    pub iteration_aux: Option<String>,
    /// The device response J/K's arithmetic ([`DevicePolicy::response`]; None: no device).
    pub device: Option<ResponseArith>,
}

/// The device's part of a plan: every kernel family's arithmetic and the engines' reserves and
/// workspaces.
#[derive(Clone, Debug)]
pub(crate) struct DevicePlan {
    /// The card class and every kernel family's arithmetic, from the `precision` settings.
    pub policy: DevicePolicy,
    /// The reserves and workspaces of the device's engines, from its limit.
    pub memory: DeviceMemoryPlan,
}

impl Plan {
    /// The plan of `request` (normalized: [`CalcOptions::normalized`]).
    pub(crate) fn new(request: CalcOptions) -> Result<Plan, LinusError> {
        let request = request.normalized()?;
        Ok(Plan {
            method: request.method(),
            basis: request.basis(),
            jk: request.jk_method(),
            algo: request.scf_algorithm(),
            request,
            device: None,
            placement: BTreeMap::new(),
            uncovered: None,
        })
    }

    /// Decides what depends on the method and the device a session opened (`device`): the
    /// device's policy and memory plan, and where every family of kernels runs.
    pub(crate) fn decide(&mut self, method: &MethodInfo, device: Option<&Device>) {
        self.device = device.map(|d| DevicePlan {
            policy: DevicePolicy::new(
                self.request.precision.fp64(),
                self.request.precision.fp32(),
                d.card.class(),
                &d.info.compute_capability,
                d.card.rates(),
            ),
            memory: DeviceMemoryPlan::for_limit(d.limit, d.response_defaults),
        });
        self.placement = Family::ALL
            .into_iter()
            .map(|k| (k, self.place(k, method, device.is_some())))
            .collect();
    }

    /// The decisions of a response for `purpose` with the sizes `sizes`.
    pub(crate) fn response_plan(
        &self,
        purpose: ResponseUse,
        sizes: &ResponseSizes,
    ) -> ResponsePlan {
        use crate::driver::options::ResponsePrecision as P;
        use crate::driver::plan::iteration_aux;
        use linus_jk::df::response::auto_single;
        let r = &self.request.response;
        let work = sizes.work();
        let (precision, tol) = (self.response_precision(), self.request.response_tol());
        let single = match (purpose, precision) {
            (ResponseUse::Polarizability, _) | (_, P::Fp64) => false,
            (_, P::Fp32) => true,
            (_, P::Auto) => auto_single(work, tol),
        };
        ResponsePlan {
            single,
            iteration_aux: match purpose {
                ResponseUse::Hessian => {
                    iteration_aux(single, work, sizes.op, r.iteration_aux().as_str())
                }
                ResponseUse::Polarizability => None,
            },
            device: self
                .device
                .as_ref()
                .map(|d| d.policy.response(purpose, precision, work, tol)),
        }
    }

    /// With a device: where its kernels do not cover the basis sets (`uncovered`, the device's
    /// provider's answer, `crate::backend::Backend::uncovered`: the GPU's generated integral
    /// kernels), the families that need them are placed on the CPU with the reason (decision
    /// D40's first half; `backend.kind: "gpu"` makes it an unsupported error where a declared
    /// calctype needs them, `CalcContext::check_settings`).
    pub(crate) fn cover(&mut self, method: &MethodInfo, uncovered: Option<String>) {
        if self.device.is_none() {
            return;
        }
        self.uncovered = uncovered;
        if self.uncovered.is_some() {
            let device = true;
            self.placement = Family::ALL
                .into_iter()
                .map(|k| (k, self.place(k, method, device)))
                .collect();
        }
    }

    /// Why the GPU's integral kernels do not cover the basis sets (None: they do).
    pub(crate) fn uncovered(&self) -> Option<&str> {
        self.uncovered.as_deref()
    }

    /// The device's part (None: no device).
    pub(crate) fn device(&self) -> Option<&DevicePlan> {
        self.device.as_ref()
    }

    /// Where the family `k` runs (None: the method has no such kernels), before any setup.
    pub(crate) fn placement(&self, k: Family) -> Option<Placement> {
        self.placement.get(&k).cloned().flatten()
    }

    /// The settings as requested (the report's `options` echo and `Session::opts`).
    pub(in crate::driver) fn request(&self) -> &CalcOptions {
        &self.request
    }

    /// The settings the request gave, by their dotted paths (the settings' presence: the same
    /// for JSON and for settings built in Rust).
    pub(crate) fn given(&self) -> Vec<String> {
        self.request.given()
    }

    /// The request, to change a setting of a session in a test.
    #[cfg(test)]
    pub(crate) fn request_mut(&mut self) -> &mut CalcOptions {
        &mut self.request
    }

    /// Checks the settings against the structure (`nalpha`, `nbeta` electrons) and resolves the
    /// method ([`MethodInfo::resolve`]): what cannot be honoured fails here, before any SCF.
    pub(crate) fn check(
        &self,
        mol: &Structure,
        nalpha: usize,
        nbeta: usize,
    ) -> Result<MethodInfo, LinusError> {
        self.request.check(nalpha, nbeta)?;
        crate::solvent::validate(&self.request, mol)?;
        MethodInfo::resolve(&self.request)
    }

    /// Whether `method` with these settings can do `req`
    /// ([`crate::derivatives::capabilities::check`]).
    pub(crate) fn capabilities(
        &self,
        req: crate::derivatives::capabilities::Request,
        method: &MethodInfo,
    ) -> Result<(), LinusError> {
        crate::derivatives::capabilities::check(req, method, &self.request)
    }

    // ---- the model

    /// The method's name (`method`).
    pub(crate) fn method(&self) -> &str {
        &self.method
    }

    /// The orbital basis set's name (`basis`).
    pub(crate) fn basis(&self) -> &str {
        &self.basis
    }

    /// Spherical (true) or Cartesian functions (`cartesian`).
    pub(crate) fn pure(&self) -> bool {
        !self.request.cartesian()
    }

    /// The determinant (`reference`; None: restricted for a closed shell).
    pub(crate) fn restricted(&self) -> Option<bool> {
        self.request.reference().restricted()
    }

    /// The implicit solvent (`solvent`).
    pub(crate) fn solvent(&self) -> Option<&SolventSettings> {
        self.request.solvent.as_ref()
    }

    // ---- resources and output

    /// Worker threads (`threads`).
    pub(crate) fn threads(&self) -> usize {
        self.request.threads()
    }

    /// The cap on a device's memory limit (`device_memory_gib`, GiB): as given, else
    /// `$LINUS_GPU_MEMORY_GIB`; None: none (the limit is 95% of the memory free when the device
    /// opens). The flag: from the environment.
    pub(crate) fn device_memory_cap(&self) -> Result<Option<(f64, bool)>, LinusError> {
        if let Auto::Value(g) = self.request.device_memory_gib() {
            return Ok(Some((g, false)));
        }
        match std::env::var("LINUS_GPU_MEMORY_GIB") {
            Ok(v) if !v.trim().is_empty() => match v.trim().parse::<f64>() {
                Ok(g) if g > 0.0 && g.is_finite() => Ok(Some((g, true))),
                _ => Err(LinusError::Input(format!(
                    "LINUS_GPU_MEMORY_GIB={v:?} (device_memory_gib's default): a positive number of GiB"
                ))),
            },
            _ => Ok(None),
        }
    }

    /// The response exchange's arithmetic as set (`precision.response`, FP64 under
    /// `precision.fp32: off`).
    pub(crate) fn response_precision(&self) -> crate::driver::options::ResponsePrecision {
        self.request.precision.response_effective()
    }

    /// The SCF's progress lines (`output.verbose`): on stderr, or collected for an in-process
    /// caller's logs (`Session::collect_progress`).
    pub(crate) fn verbose(&self) -> bool {
        self.request.output.verbose()
    }

    /// The Molden file of the orbitals in the result (`output.molden`).
    pub(crate) fn molden(&self) -> crate::driver::options::MoldenForm {
        self.request.output.molden()
    }

    /// The AO matrices the result carries (`output.overlap`, `output.core_hamiltonian`;
    /// `output.hamiltonian` brings the overlap).
    pub(crate) fn ao_outputs(&self) -> crate::driver::report::AoOutputs {
        crate::driver::report::AoOutputs::of(&self.request.output)
    }

    /// The memory limit the budgets derive from (`memory_gib`).
    pub(crate) fn memory(&self) -> MemoryLimit {
        self.request.memory()
    }

    /// A basis set from the search path (`basis_dir`; [`CalcOptions::load_basis`]).
    pub(crate) fn load_basis(&self, name: &str) -> Result<BasisDef, LinusError> {
        self.request.load_basis(name)
    }

    // ---- the J/K method

    /// The J/K method (`jk.kind`, its auxiliary basis and exchange grid).
    pub(crate) fn jk(&self) -> &JkMethod {
        &self.jk
    }

    /// Where jk exact keeps its integrals (`jk.exact_storage`, `auto` resolved).
    pub(crate) fn exact_storage(&self) -> ExactStorage {
        self.request.jk.exact_storage().resolved()
    }

    /// The screening threshold of the exact four-center integrals (`jk.screening`).
    pub(crate) fn int_thresh(&self) -> f64 {
        self.request.jk.screening()
    }

    /// The storage format of the DF exchange tensor (`jk.df_storage`).
    pub(crate) fn df_storage(&self) -> DfPrecision {
        self.request.jk.df_storage()
    }

    /// The long-range tensors of range-separated hybrids with DF (`jk.df_long_range`).
    pub(crate) fn df_long_range(&self) -> DfLongRange {
        self.request.jk.df_long_range()
    }

    // ---- grids

    /// The XC grid of an SCF for `purpose` (`grid.xc`, `auto` resolved).
    pub(crate) fn grid(&self, purpose: Purpose) -> GridChoice {
        self.request.grid.xc().for_purpose(purpose)
    }

    /// VV10's grid (`grid.nlc`, `auto` resolved: [`crate::driver::nlc_grid_for`]).
    pub(crate) fn nlc_grid(&self) -> GridChoice {
        self.request.nlc_grid_resolved()
    }

    /// The grids move with the atoms in derivatives (`grid.response`).
    pub(crate) fn grid_response(&self) -> bool {
        self.request.grid.response()
    }

    /// The CPKS kernel's grid of a Hessian whose SCF ran on `scf_grid`, if it differs
    /// ([`CalcOptions::response_grid_for`]).
    pub(crate) fn response_grid(
        &self,
        scf_grid: &GridChoice,
        meta_gga: bool,
    ) -> Option<GridChoice> {
        self.request.response_grid_for(scf_grid, meta_gga)
    }

    // ---- the SCF

    /// The SCF's thresholds for `purpose` ([`CalcOptions::convergence_for`]).
    pub(crate) fn convergence(&self, purpose: Purpose) -> Result<Convergence, LinusError> {
        self.request.convergence_for(purpose)
    }

    /// A given `scf.convergence` preset that is looser than `purpose`'s default in a threshold
    /// it sets (one `scf.e_tol`/`scf.g_tol` do not override): the warning
    /// `convergence_looser_than_default` (the presets are absolute, the defaults depend on the
    /// calctype: `tight` loosens a gradient's g_tol 5x, only `extreme` tightens a Hessian).
    pub(crate) fn convergence_warning(&self, purpose: Purpose) -> Option<linus_core::Warning> {
        let s = &self.request.scf;
        let Auto::Value(preset) = s.convergence() else {
            return None;
        };
        let (p, d) = (
            preset.thresholds(),
            crate::driver::options::default_convergence(purpose),
        );
        let mut looser = vec![];
        if s.e_tol() == Auto::Auto && p.e_tol > d.e_tol {
            looser.push(format!("e_tol {:.0e} against {:.0e}", p.e_tol, d.e_tol));
        }
        if s.g_tol() == Auto::Auto && p.g_tol > d.g_tol {
            looser.push(format!("g_tol {:.0e} against {:.0e}", p.g_tol, d.g_tol));
        }
        (!looser.is_empty()).then(|| linus_core::Warning {
            code: "convergence_looser_than_default",
            message: format!(
                "scf.convergence \"{}\" is looser than the {} default ({}): the result is less \
                 accurate than linus's default; leave scf.convergence auto, or set scf.e_tol and \
                 scf.g_tol",
                preset.name(),
                purpose.name(),
                looser.join(", ")
            ),
            path: "scf.convergence",
        })
    }

    /// The SCF's algorithm settings (`scf`, and `jk`'s incremental builds and grid switch).
    pub(crate) fn scf_algorithm(&self) -> &ScfAlgorithm {
        &self.algo
    }

    /// The SCF's iterations at most (`scf.max_iter`).
    pub(crate) fn max_iter(&self) -> usize {
        self.request.scf.max_iter()
    }

    /// The SCF's initial guess (`scf.guess`).
    pub(crate) fn guess(&self) -> Guess {
        self.request.scf.guess()
    }

    /// What a Hessian's analysis computes (`freq`; [`CalcOptions::freq_options`]).
    pub(crate) fn freq_options(&self) -> linus_vib::FrequencyOptions {
        self.request.freq_options()
    }

    /// The products of the zero-mode check before open-shell Hessians
    /// (`scf.stability_hessian_check`).
    pub(crate) fn hessian_check(&self) -> usize {
        self.request.scf.stability_hessian_check()
    }

    // ---- responses and the hardware

    /// The response equations' settings (`response`).
    pub(crate) fn response(&self) -> &ResponseSettings {
        &self.request.response
    }

    /// The Hessian's CPHF/CPKS tolerance as set, with the accuracy target's
    /// ([`CalcOptions::response_tol`]; None: the default).
    pub(crate) fn response_tol(&self) -> Option<f64> {
        self.request.response_tol()
    }

    /// The accuracy targets of the requests (`accuracy`; a session request's own replace them
    /// for that request, [`Plan::set_targets`]).
    pub(crate) fn targets(&self) -> crate::driver::accuracy::Targets {
        self.request.accuracy.targets()
    }

    /// Replace the accuracy targets (a session request's), returning the ones they replace.
    pub(crate) fn set_targets(
        &mut self,
        t: crate::driver::accuracy::Targets,
    ) -> crate::driver::accuracy::Targets {
        let old = self.targets();
        self.request.accuracy = crate::driver::options::AccuracySettings::of(t);
        old
    }

    /// The backend (`backend`).
    pub(crate) fn backend(&self) -> &BackendSettings {
        &self.request.backend
    }
}

impl Plan {
    /// Where the kernel family `k` runs for `method` and these settings, with a device open or
    /// not (None: the method has no such kernels, e.g. XC for Hartree-Fock), before any setup:
    /// the rules of `crate::driver::backend`'s module docs.
    fn place(&self, k: Family, method: &MethodInfo, device: bool) -> Option<Placement> {
        let applies = match k {
            Family::Xc | Family::GradientXc | Family::ResponseXc | Family::HessianXc => {
                method.is_dft()
            }
            Family::Vv10 => method.vv10,
            Family::ResponseJk => self.jk().has_second_derivatives(),
            Family::Jk | Family::GradientJk | Family::HessianJk | Family::ScfLoop => true,
            Family::Pcm => self.solvent().is_some(),
        };
        if !applies {
            return None;
        }
        if !device {
            return Some(Placement::Cpu(None));
        }
        let cpu = |why: &str| Some(Placement::Cpu(Some(why.to_string())));
        let no_exact =
            "jk exact: the four-center integrals have no device kernels (jk df has them)";
        let no_snk =
            "jk rij_snk: the seminumerical exchange has no device kernels (jk df has them)";
        match k {
            Family::Jk => match self.jk() {
                JkMethod::Exact => cpu(no_exact),
                JkMethod::RijSnk { .. } => cpu(no_snk),
                JkMethod::Df { .. } if self.solvent_df_pcm() => cpu(
                    "solvent.df: the density-fitted PCM's terms ride on the CPU's density-fitted \
                     Coulomb build (without it the J/K runs on the device)",
                ),
                JkMethod::Df { .. } => Some(Placement::Device),
            },
            Family::GradientJk | Family::HessianJk => match self.jk() {
                JkMethod::Exact => cpu(no_exact),
                JkMethod::RijSnk { .. } => cpu(no_snk),
                JkMethod::Df { .. } => match &self.uncovered {
                    Some(why) => cpu(why),
                    None => Some(Placement::Device),
                },
            },
            Family::Xc
            | Family::GradientXc
            | Family::ResponseXc
            | Family::HessianXc
            | Family::Vv10 => Some(Placement::Device),
            Family::Pcm if self.solvent_df_pcm() => cpu(
                "solvent.df: the density-fitted PCM rides on the CPU's density-fitted \
                     Coulomb build",
            ),
            Family::Pcm => Some(Placement::Device),
            Family::ResponseJk => match self.jk() {
                JkMethod::Df { .. } => Some(Placement::Device),
                _ => cpu(no_exact),
            },
            Family::ScfLoop => {
                let algo = &self.algo;
                if self.backend().scf_loop() == ScfLoop::Host {
                    Some(Placement::Cpu(None))
                } else if self.place(Family::Jk, method, device) != Some(Placement::Device) {
                    cpu("the device loop needs the SCF's J/K on the device")
                } else if algo.adiis || algo.soscf || algo.damping > 0.0 {
                    cpu("scf.adiis, scf.soscf and scf.damping run on the host loop")
                } else {
                    Some(Placement::Device)
                }
            }
        }
    }

    /// The solvent's SCF terms are the density-fitted PCM's (`solvent.df`), which ride on
    /// the CPU's density-fitted Coulomb build.
    fn solvent_df_pcm(&self) -> bool {
        self.solvent().is_some_and(|so| {
            crate::solvent::resolve_scf_mode(so, self.jk())
                .is_ok_and(|m| m == linus_solvent::PcmScfMode::Df)
        })
    }
}

/// The auxiliary basis of a two-level response's iteration J/K ([`IterPlan`]; None: one level)
/// for a main response J/K in single precision or not (`single`), with the exchange work per
/// density `work` and the wK operator `op`, by `response.iteration_aux` (`iter_aux`, module docs
/// of [`iteration_plan`]): "main" or a basis set's name.
pub fn iteration_aux(
    single: bool,
    work: f64,
    op: Option<Operator>,
    iter_aux: &str,
) -> Option<String> {
    // range-separated hybrids in single precision: iterations with the symmetric RI of the
    // exchange operator (half the exchange work of wK)
    let sym = single && op.is_some_and(|op| sym_op_usable(&op));
    let key = crate::driver::options::norm_key(iter_aux);
    let name: String = match key.as_str() {
        "auto" if single && work > TWO_LEVEL_WORK => ITER_AUX_DEFAULT.into(),
        "auto" if sym => "main".into(),
        "auto" => return None,
        "none" | "main" => key.clone(),
        _ => iter_aux.trim().to_string(),
    };
    (name != "none").then_some(name)
}

#[cfg(test)]
mod tests {
    use crate::driver::{CalcOptions, Session};
    use crate::Structure;
    use serde_json::json;
    use std::path::{Path, PathBuf};

    fn mol() -> Structure {
        Structure::from_xyz_str(
            "3\n\nO 0 0 0.1173\nH 0 0.7572 -0.4692\nH 0 -0.7572 -0.4692\n",
            0,
            1,
        )
        .unwrap()
    }

    fn water(o: CalcOptions) -> Session {
        Session::new(mol(), o).unwrap()
    }

    /// IN5's F2: a given preset looser than the calctype's default warns (`tight` on a
    /// gradient: e_tol 1e-8 and g_tol 5e-7 against 1e-9 and 1e-7); one as tight, or a threshold
    /// the user set, does not.
    #[test]
    fn looser_preset_warns() {
        use crate::driver::Purpose;
        let calc = |scf: serde_json::Value| {
            water(CalcOptions::from_json(json!({"basis": "sto-3g", "scf": scf})).unwrap())
        };
        let c = calc(json!({"convergence": "tight"}));
        let w =
            c.cx.plan
                .convergence_warning(Purpose::Gradient)
                .expect("a warning");
        assert_eq!(
            (w.code, w.path),
            ("convergence_looser_than_default", "scf.convergence")
        );
        assert!(w.message.contains("e_tol 1e-8 against 1e-9") && w.message.contains("g_tol 5e-7"));
        assert!(c.cx.plan.convergence_warning(Purpose::Energy).is_none());
        let c = calc(json!({"convergence": "tight", "e_tol": 1e-8, "g_tol": 5e-7}));
        assert!(c.cx.plan.convergence_warning(Purpose::Gradient).is_none());
        let c = calc(json!({"convergence": "verytight"}));
        assert!(c.cx.plan.convergence_warning(Purpose::Gradient).is_none());
        let w =
            c.cx.plan
                .convergence_warning(Purpose::Hessian)
                .expect("a warning");
        assert!(w.message.contains("g_tol 1e-8 against 1e-9") && !w.message.contains("e_tol 1e"));
        assert!(calc(json!({}))
            .cx
            .plan
            .convergence_warning(Purpose::Hessian)
            .is_none());
    }

    /// Every `.rs` file under `dir`.
    fn sources(dir: &Path, out: &mut Vec<PathBuf>) {
        for e in std::fs::read_dir(dir).unwrap() {
            let p = e.unwrap().path();
            if p.is_dir() {
                sources(&p, out);
            } else if p.extension().is_some_and(|x| x == "rs") {
                out.push(p);
            }
        }
    }

    /// The source of a file without its `#[cfg(test)]` module (the tests build settings to
    /// make sessions; they do not read a session's).
    fn non_test(src: &str) -> &str {
        src.find("#[cfg(test)]\nmod tests")
            .map_or(src, |i| &src[..i])
    }

    /// D41's table names every setting: each path of the default settings (and of their object
    /// forms) has a row (none is `not a setting`), so the table cannot fall behind the settings.
    #[test]
    fn the_table_names_every_setting() {
        use crate::driver::options::CalcOptions;
        use crate::driver::settings::CalcTypes;
        let c = water(CalcOptions {
            method: Some("b3lyp".into()),
            basis: Some("sto-3g".into()),
            ..Default::default()
        });
        let applies = c.cx.settings_applies(CalcTypes::ALL, false);
        let mut missing = Vec::new();
        for m in crate::driver::schema::table().settings.iter() {
            let path = m.path.to_string();
            if let super::settings::Applies::No(why) = applies(&path) {
                if why.ends_with("is not a setting") {
                    missing.push(path);
                }
            }
        }
        assert!(
            missing.is_empty(),
            "settings without a row in D41's table: {missing:?}"
        );
    }

    /// D41's errors before any work: a setting no calculation reads (`no_effect`), one only an
    /// undeclared calctype reads (`not_applicable`), and contingent ones passing.
    #[test]
    fn settings_that_cannot_take_effect_are_input_errors() {
        use crate::driver::options::CalcOptions;
        use crate::driver::report::CalcType;
        use crate::driver::settings::CalcTypes;
        use crate::driver::LinusError;
        let req = |v: serde_json::Value| CalcOptions::from_json(v).unwrap();
        let code = |e: LinusError| match e {
            LinusError::Setting { code, path, .. } => (code, path),
            e => panic!("not a setting error: {e}"),
        };
        // the grid of Hartree-Fock, the four-center screening of jk df: no effect anywhere
        let e = Session::new(mol(), req(json!({"method": "hf", "grid": "pyscf:4"})))
            .err()
            .unwrap();
        assert_eq!(code(e), ("no_effect", "grid.xc".to_string()));
        let e = Session::new(
            mol(),
            req(json!({"method": "pbe", "jk": {"kind": "df", "screening": 1e-10}})),
        )
        .err()
        .unwrap();
        assert_eq!(code(e), ("no_effect", "jk.screening".to_string()));
        // the response's settings: fine for a session (all calctypes), not for an energy's
        let mut c = Session::new(
            mol(),
            req(json!({"method": "pbe", "response": {"tol": 1e-8}})),
        )
        .unwrap();
        let e = c
            .declare(CalcTypes::of(CalcType::Energy, false))
            .err()
            .unwrap();
        assert_eq!(code(e), ("not_applicable", "response.tol".to_string()));
        // contingent settings pass (the level shift: while the gradient is large)
        let mut c = Session::new(
            mol(),
            req(json!({"method": "pbe", "scf": {"level_shift": 0.5}})),
        )
        .unwrap();
        c.declare(CalcTypes::of(CalcType::Energy, false)).unwrap();
        // a calculation outside the declared calctypes
        assert!(c.run(CalcType::Gradient).unwrap_err().is_input());
        // Rust settings: given is what differs from the defaults
        let o = CalcOptions {
            method: Some("hf".into()),
            grid: crate::driver::options::GridSettings {
                xc: Some(crate::driver::GridChoice::Pyscf { level: 4 }),
                ..Default::default()
            },
            ..Default::default()
        };
        assert_eq!(code(Session::new(mol(), o).err().unwrap()).1, "grid.xc");
    }

    /// D41's table checked by effect: for each case and every setting the table marks used or
    /// not applicable (resources, outputs and contingent ones aside), an alternative value. Not
    /// applicable: the result and the record are bitwise unchanged (nothing reads it in effect);
    /// used: something changes (energy, gradient or Hessian bits, iterations, Fock builds, the
    /// record). A setting in neither column fails the test, so the table cannot drift from the
    /// code silently.
    #[test]
    fn settings_applicability_checked_by_effect() {
        use super::settings::{Applies, CalcTypes};
        use crate::driver::report::CalcType;
        use serde_json::Value;
        let alternatives: Vec<(&str, Value)> = vec![
            ("grid.xc", json!("pyscf:2")),
            ("grid.nlc", json!("linus:1")),
            ("grid.response", json!(false)),
            ("jk.screening", json!(1e-8)),
            ("jk.aux", json!("def2-svp-rifit|def2-universal-jkfit")),
            ("jk.k_grid", json!("cosx:1")),
            ("jk.exact_storage", json!("incore")),
            ("jk.df_storage", json!("f32")),
            ("jk.df_long_range", json!("one_sided")),
            ("scf.convergence", json!("loose")),
            ("scf.e_tol", json!(1e-5)),
            ("scf.g_tol", json!(1e-3)),
            ("scf.max_iter", json!(5)),
            ("scf.guess", json!("core")),
            ("scf.lindep", json!(0.5)),
            ("scf.diis_space", json!(2)),
            ("scf.diis_start", json!(5)),
            ("scf.diis_drop", json!("oldest")),
            ("scf.diis_unpin", json!(1.0)),
            ("scf.damping", json!(0.4)),
            ("scf.adiis", json!(true)),
            ("scf.soscf", json!(true)),
            ("jk.incremental", json!(false)),
            ("jk.incremental_reset", json!(1)),
            ("scf.stability", json!("always")),
            ("scf.stability_probe", json!(0)),
            ("scf.stability_closed_probe", json!(0)),
            ("response.tol", json!(0.1)),
            ("response.field_tol", json!(1e-2)),
            ("precision.response", json!("fp32")),
            ("response.iteration_aux", json!("main")),
            ("grid.kernel", json!("linus:2")),
            ("scf.stability_hessian_check", json!(0)),
            ("precision.fp64", json!("native")),
            ("precision.fp32", json!("off")),
            ("backend.scf_loop", json!("host")),
        ];
        let water = |charge: i32, mult: u32| {
            Structure::from_xyz_str(
                "3\n\nO 0 0 0.1173\nH 0 0.7572 -0.4692\nH 0 -0.7572 -0.4692\n",
                charge,
                mult,
            )
            .unwrap()
        };
        // what a run computed and recorded (its settings echo and timings aside)
        let summary = |c: &mut Session, calc: CalcType| -> Option<Value> {
            let r = c.run(calc).ok()?;
            let mut v = r.to_json();
            let m = v.as_object_mut()?;
            for k in ["options", "timings", "resources"] {
                m.remove(k);
            }
            if let Some(res) = m.get_mut("resolved").and_then(|x| x.as_object_mut()) {
                res.remove("settings");
                res.remove("reasons");
                res.remove("declared_calctypes");
                // which given settings took effect: the echo of the setting under test
                if let Some(rec) = res.get_mut("record").and_then(|x| x.as_object_mut()) {
                    rec.remove("settings_applied");
                }
            }
            if let Some(scf) = m.get_mut("scf").and_then(|x| x.as_object_mut()) {
                scf.remove("history");
            }
            // wall times (the probes' `seconds`, ...)
            fn strip(v: &mut Value) {
                match v {
                    Value::Object(m) => {
                        m.retain(|k, _| {
                            !(k == "seconds"
                                || k == "available_gib"
                                || k.starts_with("time")
                                || k.starts_with("wall"))
                        });
                        m.values_mut().for_each(strip);
                    }
                    Value::Array(a) => a.iter_mut().for_each(strip),
                    _ => {}
                }
            }
            strip(&mut v);
            Some(v)
        };
        let cases = [
            ("hf", "exact", (0, 1), CalcType::Gradient, false),
            ("pbe", "df", (0, 1), CalcType::Gradient, false),
            ("b3lyp", "df", (1, 2), CalcType::Energy, false),
            ("b3lyp", "df", (0, 1), CalcType::Hessian, true),
            ("wb97x", "rij_snk", (0, 1), CalcType::Energy, false),
        ];
        let mut findings = Vec::new();
        for (method, jk, (charge, mult), calc, ir) in &cases {
            let mut base_req = json!({"method": method, "basis": "sto-3g", "jk": jk, "threads": 2, "memory_gib": 4});
            if *ir {
                base_req["freq"] = json!({"ir": true});
            }
            let mut base_c = Session::new(
                water(*charge, *mult),
                CalcOptions::from_json(base_req.clone()).unwrap(),
            )
            .unwrap();
            let base = summary(&mut base_c, *calc).expect("the base run");
            let open = *mult > 1;
            let applies = base_c.cx.settings_applies(CalcTypes::of(*calc, *ir), open);
            for (path, alt) in &alternatives {
                let a = applies(path);
                if matches!(a, Applies::Contingent(_)) {
                    continue;
                }
                let mut req = base_req.clone();
                match path.split_once('.') {
                    Some((g, k)) => {
                        let m = req.as_object_mut().unwrap();
                        let e = m.entry(g.to_string()).or_insert_with(|| json!({}));
                        if !e.is_object() {
                            *e = json!({"kind": e.clone()});
                        }
                        e.as_object_mut()
                            .unwrap()
                            .insert(k.to_string(), alt.clone());
                    }
                    None => {
                        req[*path] = alt.clone();
                    }
                }
                let Ok(o) = CalcOptions::from_json(req) else {
                    // a setting the J/K method's form cannot even hold (jk.aux of jk exact)
                    if a == Applies::Used {
                        findings.push(format!(
                            "{method} {jk}: {path} = {alt}: not a valid value here"
                        ));
                    }
                    continue;
                };
                let got = Session::new_unchecked(water(*charge, *mult), o)
                    .ok()
                    .and_then(|mut c| summary(&mut c, *calc));
                // a setting the run refuses (jk.exact_storage with jk df) takes no effect either
                let changed = got.as_ref().is_some_and(|g| g != &base);
                let changed = changed || (got.is_none() && a == Applies::Used);
                match (&a, changed) {
                    (Applies::Used, false) => {
                        findings.push(format!("{method} {jk} {}: {path} = {alt}: used, but nothing changed", calc.name()))
                    }
                    (Applies::No(why), true) => findings.push(format!(
                        "{method} {jk} {}: {path} = {alt}: not applicable ({why}), but the result changed",
                        calc.name()
                    )),
                    _ => {}
                }
            }
        }
        assert!(
            findings.is_empty(),
            "D41's table against the code:\n{}",
            findings.join("\n")
        );
    }

    /// The plan is the one reader of a session's settings: no file outside it asks for the
    /// request (but the settings echo of the report and `Session::opts`), and the functions
    /// outside it that take the whole settings are the resolvers the plan calls.
    #[test]
    fn plan_is_the_only_reader() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
        let mut files = Vec::new();
        sources(&root, &mut files);
        // the request's echo, once each: the report's `options` and `Session::opts`
        let echo = ["driver/report.rs", "driver/mod.rs"];
        // the resolvers of the plan, which take the whole settings, and the echo, where they are
        let resolvers = [
            ("driver/mod.rs", "pub fn resolve(opts: &CalcOptions)"),
            ("driver/mod.rs", "pub fn opts(&self) -> &CalcOptions"),
            ("dispersion.rs", "opts: &CalcOptions,"),
            ("solvent/scf.rs", "opts: &crate::driver::CalcOptions,"),
            (
                "derivatives/capabilities.rs",
                "pub fn check(req: Request, method: &MethodInfo, opts: &CalcOptions)",
            ),
        ];
        let mut bad = Vec::new();
        for f in &files {
            let rel = f
                .strip_prefix(&root)
                .unwrap()
                .to_string_lossy()
                .replace('\\', "/");
            // the test-only modules (the method crates' whole-calculation tests, the solvent's)
            // build their settings as any caller does
            if rel.starts_with("driver/plan/")
                || rel == "driver/options.rs"
                || rel.starts_with("tests/")
                || rel == "solvent/tests.rs"
            {
                continue;
            }
            let full = std::fs::read_to_string(f).unwrap();
            let src = non_test(&full);
            let asks = src.matches(".request()").count();
            if asks > usize::from(echo.contains(&rel.as_str())) {
                bad.push(format!("{rel}: asks for the request {asks} times"));
            }
            for (i, line) in src.lines().enumerate() {
                let l = line.trim_start();
                if l.starts_with("//") {
                    continue;
                }
                let takes_all =
                    line.contains("&CalcOptions") || line.contains("&crate::driver::CalcOptions");
                if takes_all
                    && !resolvers
                        .iter()
                        .any(|(r, sig)| *r == rel && line.contains(sig))
                {
                    bad.push(format!("{rel}:{}: takes the whole settings: {l}", i + 1));
                }
            }
            // the resolvers are called from the plan only (and `MethodInfo::resolve` from its tests)
            for call in [
                "MethodInfo::resolve(",
                "solvent::validate(",
                "capabilities::check(",
            ] {
                if src.contains(call) {
                    bad.push(format!("{rel}: calls the plan's resolver {call}"));
                }
            }
        }
        assert!(
            bad.is_empty(),
            "settings read outside the plan:\n{}",
            bad.join("\n")
        );
    }
}
