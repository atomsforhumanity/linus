//! What a calculation will do, decided before it runs (`linus --plan`, [`Session::plan`],
//! Python's `Session.plan`).
//!
//! The report is the plan's: the settings the run will have (`resolved.settings` as far as the
//! plan decides them, with the reasons), where every family of kernels runs, their arithmetic on
//! a device, the memory plan, a Hessian's responses, and what only the run decides, with the
//! rule it decides by (`at_run_time`): a value the run decides stays as requested in
//! `settings`, and an `auto` left so is listed in `deferred` (qcdata's `Resolved.deferred`).

use super::estimate::{EstimateInputs, Exchange, PlannedEstimates};
use super::ResponseSizes;
use crate::derivatives::capabilities::Request;
use crate::derivatives::hessian::{cpks_tol, ResolvedCphf};
use crate::driver::backend::{Family, Placement};
use crate::driver::options::{JkMethod, ResponsePrecision};
use crate::driver::plan::settings::{CalcTypes, RunFacts};
use crate::driver::policy::ResponseMode;
use crate::driver::report::{CalcType, FamilyArith};
use crate::driver::{CalcContext, LinusError, Session};
use crate::two_electron::ResponseUse;
use linus_ints::BasisSet;
use serde::Serialize;
use serde_json::Value;
use std::collections::BTreeMap;

const GIB: f64 = 1073741824.0;

/// What a calculation will do (`linus --plan`; docs/cli.md, "Plan").
#[derive(Clone, Debug, Serialize)]
pub struct PlanReport {
    /// The calculation (`energy`, `gradient` or `hessian`).
    pub calctype: &'static str,
    /// What the settings are checked against (`resolved.calctypes`).
    pub calctypes: Vec<&'static str>,
    /// The settings the run will have (`resolved.settings`), as far as the plan decides them.
    pub settings: Value,
    /// Why each `auto` the plan decides takes its value (`resolved.reasons`).
    pub reasons: BTreeMap<String, String>,
    /// Where every family of kernels the method has runs.
    pub placement: BTreeMap<&'static str, PlannedPlacement>,
    /// Every kernel family's arithmetic on the device, with why (absent without a device).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub arithmetic: Option<BTreeMap<&'static str, FamilyArith>>,
    pub memory: PlannedMemory,
    /// The sizes the calculation scales with.
    pub sizes: PlannedSizes,
    /// What the calculation will need from the machine (peak host and device memory, useful
    /// threads), each an estimate with its error band (`driver/plan/estimate.rs`).
    pub estimates: PlannedEstimates,
    /// The responses of a Hessian or frequency calculation (`hessian`; `polarizability` with
    /// IR intensities).
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    pub response: BTreeMap<&'static str, PlannedResponse>,
    /// What only the run decides, and the rule it decides by.
    pub at_run_time: Vec<RunTimeDecision>,
    /// The settings `settings` leaves `auto` because the run decides them (dotted paths, as
    /// qcdata's `Resolved.deferred`); every other `auto` the plan resolves.
    pub deferred: Vec<String>,
}

/// Where a family of kernels runs: `gpu` or `cpu`, and why on the CPU with a device open.
#[derive(Clone, Debug, Serialize)]
pub struct PlannedPlacement {
    pub on: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub why: Option<String>,
}

/// The memory plan: the host's limit, the smallest `memory_gib` that runs the calculation and
/// what each phase holds, and with a device its reserves and workspaces (GiB).
#[derive(Clone, Debug, Serialize)]
pub struct PlannedMemory {
    /// Where the limit comes from: `memory_gib`, `cgroup limit` or `total RAM`.
    pub source: &'static str,
    pub limit_gib: Option<f64>,
    /// The smallest `memory_gib` that runs the calculation (f64 storage of the DF tensors): the
    /// largest of its phases' requirements, the `details.required_gib` of the `host_memory`
    /// failure below it (`crate::driver::needs`).
    pub required_gib: f64,
    /// What each phase holds at its peak at the limit (`phase`, `parts`: `what` and `gib`) and
    /// its own requirement (`required_gib`).
    pub phases: Vec<linus_base::memory::PhaseNeed>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub device: Option<PlannedDeviceMemory>,
}

/// The sizes a calculation scales with: atoms, basis functions, the auxiliary basis set's
/// functions (density fitting: the J/K's fitting set for the calculation's purpose), occupied
/// orbitals of each spin.
#[derive(Clone, Debug, Serialize)]
pub struct PlannedSizes {
    pub natoms: usize,
    pub nbf: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub naux: Option<usize>,
    pub nalpha: usize,
    pub nbeta: usize,
}

/// The device memory plan (GiB; `crate::driver::memory`).
#[derive(Clone, Debug, Serialize)]
pub struct PlannedDeviceMemory {
    pub limit_gib: f64,
    pub others_reserve_gib: f64,
    pub grid_partition_gib: f64,
    pub response_workspace_gib: f64,
    pub k_operator_reserve_gib: f64,
    pub w_planes_reserve_gib: f64,
    pub ground_cache_leave_gib: f64,
}

/// A response's plan.
#[derive(Clone, Debug, Serialize)]
pub struct PlannedResponse {
    /// The CPU's response J/K: `single` or `double` precision.
    pub precision: &'static str,
    /// The iteration J/K's auxiliary basis of a two-level response (absent: one level).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub iteration_aux: Option<String>,
    /// The max-abs residual the CPHF/CPKS aims at.
    pub cphf_tol: f64,
    /// The device's response J/K (absent: the CPU's).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub device: Option<PlannedDeviceResponse>,
}

/// The device response J/K's mode and exchange arithmetic, and why.
#[derive(Clone, Debug, Serialize)]
pub struct PlannedDeviceResponse {
    /// `double`, `single`, or `auto` (decision D37's costing at the first product).
    pub mode: &'static str,
    pub exchange: String,
    pub why: &'static str,
}

/// A decision only the run makes, and its rule.
#[derive(Clone, Debug, Serialize)]
pub struct RunTimeDecision {
    pub what: &'static str,
    pub rule: &'static str,
    /// The settings it decides (dotted paths), which the plan leaves as requested.
    #[serde(skip)]
    pub settings: &'static [&'static str],
}

/// The dotted paths of the `"auto"` values left in a settings tree.
fn auto_paths(v: &Value, path: &str, out: &mut Vec<String>) {
    match v {
        Value::String(s) if s.eq_ignore_ascii_case("auto") => out.push(path.to_string()),
        Value::Object(m) => {
            for (k, x) in m {
                let p = if path.is_empty() {
                    k.clone()
                } else {
                    format!("{path}.{k}")
                };
                auto_paths(x, &p, out);
            }
        }
        _ => {}
    }
}

impl CalcContext {
    /// The decisions of `calctypes` only the run makes, with their rules.
    fn run_time_decisions(&self, calctypes: CalcTypes) -> Vec<RunTimeDecision> {
        let mut v = Vec::new();
        let mut add = |what, rule, settings| {
            v.push(RunTimeDecision {
                what,
                rule,
                settings,
            })
        };
        let jk = self.plan.jk();
        let device = self.plan.device().is_some();
        let on = |k: Family| self.placement(k) == Some(Placement::Device);
        if matches!(
            jk,
            JkMethod::Df { .. } | JkMethod::RijSnk { .. } | JkMethod::RijCgk { .. }
        ) && !on(Family::Jk)
        {
            add(
                "jk.df_long_range and the storage of the DF tensors",
                "f64 within the DF tensors' budget (75% of the memory limit less the reservations; with \
                 memory_gib auto, at most 90% of the memory free when the SCF starts): B, and B^omega of a \
                 range-separated hybrid when both fit, else its one-sided coefficients; for J alone the raw \
                 integrals, else integral-direct; with exchange and no f64 storage fitting, a host_memory \
                 error (compressed storage only when jk.df_storage sets it)",
                &["jk.df_long_range"],
            );
        }
        if device {
            add(
                "a device setup's fallback",
                "each family's device setup falls back to the CPU when it does not fit the device \
                 memory then (resolved.record.backend.fallbacks)",
                &[],
            );
        }
        if on(Family::ScfLoop) {
            add(
                "backend.scf_loop",
                "the device loop when its state and the SCF's largest J/K build fit the device beside \
                 the engines, else the host loop, else the CPU's J/K",
                &["backend.scf_loop"],
            );
        }
        add(
            "the SCF's probes at a later geometry",
            "auto scf.stability_probe and scf.stability_closed_probe run at a later geometry only when \
             the gap trigger fires (the HOMO-LUMO gap fell below scf.stability_closed_gap, or 0.02 Eh \
             below the last probe's)",
            &[],
        );
        if calctypes.hessian
            && on(Family::ResponseJk)
            && self.plan.response_precision() == ResponsePrecision::Auto
        {
            add(
                "the device response's exchange (precision.response auto)",
                "the stored operator when it pays for the products, fits the device then and its \
                 costing wins (decision D37), else the tensor's",
                &["precision.response"],
            );
        }
        v
    }
}

impl Session {
    /// What a calculation of `calctype` will do, decided before any work (`linus --plan`;
    /// docs/cli.md, "Plan"); what the method cannot do is the same error as
    /// [`Session::run`]'s.
    pub fn plan(&self, calctype: CalcType) -> Result<PlanReport, LinusError> {
        let ir = calctype == CalcType::Hessian && self.cx.plan.freq_options().ir;
        self.cx.check_capabilities(Request::of(calctype, ir))?;
        let calctypes = CalcTypes::of(calctype, ir);
        let purpose = calctype.purpose();
        let (na, nb) = self.cx.mol.nalpha_nbeta().map_err(LinusError::Input)?;
        let restricted = self.cx.plan.restricted_for(na, nb);
        let scf_grid = self.cx.plan.grid(purpose);
        let convergence = self.cx.plan.convergence(purpose)?;
        // the J/K's auxiliary basis set for the calculation's purpose (a Hessian's: its own)
        let naux = match self.cx.plan.jk().aux() {
            Some(a) => {
                let def = self.cx.plan.load_basis(a.for_purpose(purpose))?;
                Some(BasisSet::new(&def, &self.cx.mol.z, &self.cx.mol.positions, true)?.nbf)
            }
            None => None,
        };
        // the responses
        let mut response = BTreeMap::new();
        let mut cphf = None;
        if calctypes.hessian {
            let occ = if restricted { vec![na] } else { vec![na, nb] };
            let sizes = ResponseSizes {
                nbf: self.cx.basis.nbf,
                naux: naux.unwrap_or(0),
                occ,
                op: linus_base::terms::exchange_parts(self.cx.method.exchange).1,
            };
            let df = matches!(self.cx.plan.jk(), JkMethod::Df { .. });
            let on_device = self.cx.placement(Family::ResponseJk) == Some(Placement::Device);
            let mut uses = vec![(ResponseUse::Hessian, "hessian")];
            if calctypes.ir {
                uses.push((ResponseUse::Polarizability, "polarizability"));
            }
            for (purpose, name) in uses {
                let rp = self.cx.plan.response_plan(purpose, &sizes);
                let (single, iteration_aux) = if df {
                    (rp.single, rp.iteration_aux.clone())
                } else {
                    (false, None)
                };
                let tol = match purpose {
                    ResponseUse::Hessian => {
                        cpks_tol(self.cx.plan.response_tol(), iteration_aux.is_some())
                    }
                    ResponseUse::Polarizability => self.cx.plan.response().field_tol(),
                };
                let device = rp
                    .device
                    .filter(|_| df && on_device)
                    .map(|a| PlannedDeviceResponse {
                        mode: match a.mode {
                            ResponseMode::Double => "double",
                            ResponseMode::Single => "single",
                            ResponseMode::AutoOperator => "auto",
                        },
                        exchange: a.k.to_string(),
                        why: a.why,
                    });
                if purpose == ResponseUse::Hessian {
                    let meta = self
                        .cx
                        .method
                        .functional
                        .as_ref()
                        .is_some_and(|f| f.is_meta_gga());
                    cphf = Some(ResolvedCphf {
                        cphf_tol: tol,
                        single_precision: single,
                        two_level: iteration_aux.is_some(),
                        iteration_aux: iteration_aux.clone(),
                        precision_setting: self.cx.plan.response_precision(),
                        max_residual: f64::NAN,
                        end: None,
                        grid: self
                            .is_dft()
                            .then(|| self.cx.plan.response_grid(&scf_grid, meta))
                            .flatten(),
                    });
                }
                response.insert(
                    name,
                    PlannedResponse {
                        precision: if single { "single" } else { "double" },
                        iteration_aux,
                        cphf_tol: tol,
                        device,
                    },
                );
            }
        }
        let solvent_scf = match self.cx.plan.solvent() {
            Some(so) => Some(crate::solvent::resolve_scf_mode(so, self.cx.plan.jk())?),
            None => None,
        };
        // where the SCF's nuclear attraction will run: the plan's rule on this geometry's pairs
        // (a GPU session builds them while its device opens)
        let nuclear_attraction = (self.cx.placement(Family::Jk) == Some(Placement::Device))
            .then(|| {
                let pairs = self
                    .cx
                    .integrals
                    .pairs(&self.cx.basis, linus_ints::DerivOrder::Zero);
                self.cx
                    .nuclear_attraction_place(&pairs, self.cx.mol.point_charges().len())
            })
            .flatten()
            .map(|(gpu, why)| (if gpu { "gpu" } else { "cpu" }, why));
        let (settings, reasons) = self.cx.resolved_settings(&RunFacts {
            calctypes,
            purpose,
            scf_grid: &scf_grid,
            restricted,
            convergence,
            probes: Some((self.stability_probe(), self.closed_shell_probe())),
            df: None,
            response: cphf.as_ref(),
            solvent_scf,
            // decided at the J/K's setup, from the screened tensor's size (not before a run)
            df_j: None,
            nuclear_attraction,
        });
        let placement = Family::ALL
            .into_iter()
            .filter_map(|k| {
                let p = self.cx.placement(k)?;
                Some((
                    k.name(),
                    match p {
                        Placement::Device => PlannedPlacement {
                            on: "gpu",
                            why: None,
                        },
                        Placement::Cpu(why) => PlannedPlacement { on: "cpu", why },
                    },
                ))
            })
            .collect();
        let at_run_time = self.cx.run_time_decisions(calctypes);
        let mut left = Vec::new();
        auto_paths(&settings, "", &mut left);
        let deferred = left
            .into_iter()
            .filter(|p| at_run_time.iter().any(|d| d.settings.contains(&p.as_str())))
            .collect();
        let mem = self.cx.plan.memory();
        let device = self.cx.plan.device();
        let phases = self.cx.memory_needs(Request::of(calctype, ir))?;
        let required = phases.iter().map(|p| p.required()).max().unwrap_or(0);
        let estimates = EstimateInputs {
            calctype,
            natoms: self.cx.mol.natom(),
            nbf: self.cx.basis.nbf,
            naux,
            jk: self.cx.plan.jk(),
            exchange: match self.cx.method.exchange.map(|op| op.global_fraction()) {
                None => Exchange::None,
                Some(Some(_)) => Exchange::Global,
                Some(None) => Exchange::RangeSeparated,
            },
            open_shell: !restricted,
            memory_limit: mem.limit,
            device_limit: device.map(|d| d.memory.limit),
            jk_on_device: self.cx.placement(Family::Jk) == Some(Placement::Device),
            gridk: self.cx.gridk_memory()?,
        }
        .estimates();
        Ok(PlanReport {
            calctype: calctype.name(),
            calctypes: calctypes.names(),
            settings,
            reasons,
            placement,
            arithmetic: device.map(|d| {
                d.policy
                    .table()
                    .into_iter()
                    .map(|(family, arith, reason)| (family, FamilyArith { arith, reason }))
                    .collect()
            }),
            memory: PlannedMemory {
                source: mem.source(),
                limit_gib: mem.bytes().map(|b| b as f64 / GIB),
                required_gib: required as f64 / GIB,
                phases,
                device: device.map(|d| {
                    let m = &d.memory;
                    let g = |b: usize| b as f64 / GIB;
                    PlannedDeviceMemory {
                        limit_gib: g(m.limit),
                        others_reserve_gib: g(m.others_reserve),
                        grid_partition_gib: g(m.grid_partition),
                        response_workspace_gib: g(m.response_workspace),
                        k_operator_reserve_gib: g(m.k_operator_reserve),
                        w_planes_reserve_gib: g(m.w_planes_reserve),
                        ground_cache_leave_gib: g(m.ground_cache_leave),
                    }
                }),
            },
            sizes: PlannedSizes {
                natoms: self.cx.mol.natom(),
                nbf: self.cx.basis.nbf,
                naux,
                nalpha: na,
                nbeta: nb,
            },
            estimates,
            response,
            at_run_time,
            deferred,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::driver::CalcOptions;
    use crate::Structure;

    fn water(opts: CalcOptions) -> Session {
        let mol = Structure::from_xyz_str(
            "3\n\nO 0 0 0.1173\nH 0 0.7572 -0.4692\nH 0 -0.7572 -0.4692\n",
            0,
            1,
        )
        .unwrap();
        Session::new(mol, opts).unwrap()
    }

    /// The plan's settings and reasons are the run's (the DF exchange tensor's format too: auto
    /// is f64, decided before the run); a Hessian's responses are planned; nothing runs.
    #[test]
    fn the_plan_is_what_runs() {
        let opts = CalcOptions {
            method: Some("b3lyp".into()),
            basis: Some("sto-3g".into()),
            jk: CalcOptions::from_json(serde_json::json!({"jk": "df"}))
                .unwrap()
                .jk,
            ..Default::default()
        };
        for calc in [CalcType::Energy, CalcType::Gradient, CalcType::Hessian] {
            // the Hessian with IR intensities (`freq.ir`; the other calctypes take no freq settings)
            let o = if calc == CalcType::Hessian {
                CalcOptions {
                    freq: CalcOptions::from_json(serde_json::json!({"freq": {"ir": true}}))
                        .unwrap()
                        .freq,
                    ..opts.clone()
                }
            } else {
                opts.clone()
            };
            let mut c = water(o);
            let p = c.plan(calc).unwrap();
            assert!(c.last().is_none(), "the plan ran an SCF");
            let r = c.run(calc).unwrap();
            assert_eq!(p.settings, r.resolved.settings, "{}", p.calctype);
            assert_eq!(p.settings["jk"]["df_storage"], "f64");
            assert_eq!(p.reasons, r.resolved.reasons, "{}", p.calctype);
            assert_eq!(p.calctypes, r.resolved.calctypes);
            assert_eq!(p.placement["jk"].on, "cpu");
            assert!(p
                .at_run_time
                .iter()
                .any(|d| d.what.starts_with("jk.df_long_range")));
            // every auto the plan leaves is one the run decides
            let mut left = Vec::new();
            auto_paths(&p.settings, "", &mut left);
            assert_eq!(left, p.deferred, "{}", p.calctype);
            assert!(!p.deferred.contains(&"jk.df_storage".to_string()));
            if p.calctype == "hessian" {
                assert_eq!(p.response["hessian"].precision, "double");
                assert_eq!(p.response["hessian"].cphf_tol, 1e-6);
                assert!(p.response.contains_key("polarizability"));
            } else {
                assert!(p.response.is_empty());
            }
        }
        // what the method cannot do fails the plan as it fails the run
        let snk = CalcOptions {
            jk: CalcOptions::from_json(serde_json::json!({"jk": "rij_snk"}))
                .unwrap()
                .jk,
            ..opts
        };
        assert!(water(snk).plan(CalcType::Hessian).is_err());
    }
}
