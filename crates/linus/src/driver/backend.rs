//! The `backend` settings ([`crate::driver::options::BackendSettings`]) resolved per family of kernels,
//! and what the calculation reports of it (`resolved.record.backend`).
//!
//! A session opens its device once, in `Session::new` (backend `gpu` or `auto`, builds
//! with the `gpu` feature), and keeps it for all its calculations. Each family of kernels
//! ([`Family`]) then runs where `CalcContext::placement` puts it, decided before the work
//! starts:
//!
//! | family | on the device when | otherwise on the CPU |
//! |---|---|---|
//! | `jk` (SCF J/K) | `jk` df, rij_cgk (the J on the DF engine, or without a stored tensor: `backend.df_j`) | exact, rij_snk; `solvent.scf: "df"` (the density-fitted PCM rides on the CPU's J build) |
//! | `xc` (SCF XC) | DFT | - |
//! | `vv10` (VV10's energy, SCF potential and gradient, on the NLC grid) | DFT with VV10 | - |
//! | `gradient_jk` (the two-electron gradient) | `jk` df | exact, rij_snk, rij_cgk |
//! | `gradient_xc` (the XC gradient) | DFT | - |
//! | `hessian_jk` (the Hessian's explicit two-electron terms and their F^x) | `jk` df | exact |
//! | `hessian_xc` (the Hessian's explicit XC terms, the grid response included) | DFT | - |
//! | `response_jk` (CPHF/CPKS J/K) | `jk` df | exact |
//! | `response_xc` (CPKS XC kernel) | DFT | - |
//! | `scf_loop` (the SCF's own work) | `backend.scf_loop: "device"` or `"auto"` (the default; closed and open shells, decisions D27 and D31); the SCF's J/K on the device, no ADIIS, SOSCF or damping | `scf_loop: "host"` |
//! | `pcm` (an implicit solvent's surface terms: integrals, charge solve, SCF and gradient terms) | `solvent.scf_terms` exact or predicted | `solvent.scf: "df"` |
//!
//! The device loop (`linus_gpu::scf`, `crate::backend::gpu::scf_control`) takes an implicit
//! solvent's reaction field on the device (`pcm` on the device; from the CPU otherwise), and hands
//! a UHF/UKS saddle point to the host loop for the stability analysis.
//!
//! What each family computes in is the device policy's (`crate::driver::policy`, decided once
//! when the device opens): `precision.fp64: "auto"` (the default) resolves per card class
//! (decision D27), native FP64 on FP64-rich devices, the Ozaki emulation of `"emulated"` on the
//! others, reported in `resolved.reasons["precision.fp64"]`; `precision.fp32` sets the families
//! with FP32 paths (the XC gradient, the Hessian XC terms, VV10's pair sums).
//!
//! One-electron integrals and ECPs, the cavity, SMD's CDS term, the PCM's Hessian terms and
//! dispersion always run on the CPU (cheap, or no device kernels yet). A device setup that runs out of memory (the density-fitting
//! tensor fits neither the device nor the host budget of its streamed storage) falls back to
//! the CPU for that family, with the reason; any other device error, and every device error
//! after a setup, is a [`LinusError::Backend`]: a calculation never switches backend midway,
//! which keeps it reproducible run to run.

use crate::driver::options::{BackendKind, BackendSettings as Backend, Fp32Mode, KernelCache};
use crate::driver::policy::{DevicePolicy, DevicePrecision};
use crate::driver::record::Decision;
use crate::driver::report::{
    EmulationUsed, Fallback, FamilyArith, PlaceRecord, ResolvedBackend, ResponseArithmetic,
};
use crate::driver::CalcContext;
use linus_base::error::LinusError;
use serde::Serialize;
use std::sync::{Arc, Mutex};

/// A family of kernels the backend resolves on its own.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Family {
    /// The SCF's Coulomb and exchange builds.
    Jk,
    /// The SCF's XC energy and potential (the semilocal part).
    Xc,
    /// VV10 nonlocal correlation of -V functionals.
    Vv10,
    /// The two-electron term of nuclear gradients.
    GradientJk,
    /// The XC term of nuclear gradients (the semilocal part).
    GradientXc,
    /// The explicit two-electron terms of Hessians (second derivatives at fixed orbitals) and
    /// their derivative Fock matrices F^x.
    HessianJk,
    /// The explicit XC terms of Hessians (V_xc^x, E_xc^xy, the grid response).
    HessianXc,
    /// The J/K products of CPHF/CPKS responses (Hessians, polarizabilities).
    ResponseJk,
    /// The XC kernel of CPKS responses.
    ResponseXc,
    /// The SCF loop's own work (`backend.scf_loop`); "device" or "host" in `resolved.record.backend`.
    ScfLoop,
    /// An implicit solvent's surface terms in the SCF and in gradients (the integrals of the
    /// surface charges and the charge solve).
    Pcm,
}

impl Family {
    /// Every family of kernels.
    pub const ALL: [Family; 11] = [
        Family::Jk,
        Family::Xc,
        Family::Vv10,
        Family::GradientJk,
        Family::GradientXc,
        Family::HessianJk,
        Family::HessianXc,
        Family::ResponseJk,
        Family::ResponseXc,
        Family::ScfLoop,
        Family::Pcm,
    ];

    /// The name in `resolved.record.backend` and in device errors.
    pub fn name(&self) -> &'static str {
        match self {
            Family::Jk => "jk",
            Family::Xc => "xc",
            Family::Vv10 => "vv10",
            Family::GradientJk => "gradient_jk",
            Family::GradientXc => "gradient_xc",
            Family::HessianJk => "hessian_jk",
            Family::HessianXc => "hessian_xc",
            Family::ResponseJk => "response_jk",
            Family::ResponseXc => "response_xc",
            Family::ScfLoop => "scf_loop",
            Family::Pcm => "pcm",
        }
    }
}

/// Where a family of kernels runs.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Placement {
    /// On the session's device.
    Device,
    /// On the CPU; with a device open, why (a fallback).
    Cpu(Option<String>),
}

/// The device of a session, as `resolved.record.backend.device` reports it.
#[derive(Clone, Debug, Serialize)]
pub struct DeviceInfo {
    pub id: u32,
    pub name: String,
    /// Device memory (GiB).
    pub memory_gib: f64,
    pub compute_capability: String,
    /// The device memory the session's engines plan against (GiB): 95 % of what was free
    /// when the device opened, at most `memory_cap_gib`.
    pub memory_limit_gib: f64,
    /// The cap `device_memory_gib` (or `$LINUS_GPU_MEMORY_GIB`), when given: processes sharing a
    /// card each plan within theirs.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub memory_cap_gib: Option<f64>,
}

/// The device a session opened.
pub(crate) struct Device {
    pub(crate) info: DeviceInfo,
    /// What the device policy reads of the card (its class and exchange rates,
    /// [`DevicePolicy::new`]).
    pub(crate) card: crate::driver::policy::CardProps,
    /// The device memory limit the engines plan against (bytes; the memory plan's input).
    pub(crate) limit: usize,
    /// The response J/K engine's defaults (the memory plan's input).
    pub(crate) response_defaults: crate::driver::memory::ResponseDefaults,
}

/// The backend of a session: its device (the setups' outcomes are the run record's,
/// `crate::driver::record`).
pub(crate) struct BackendState {
    device: Option<Device>,
    /// The device's provider of engines (`crate::backend`).
    provider: Option<Box<dyn crate::backend::Backend>>,
    /// Backend auto without a device: why.
    no_device: Option<String>,
    /// The arithmetic of each response's products (`resolved.record.backend.response`), shared with
    /// the device's response J/K, which records its stored operator at its first product. A
    /// record only: what a response's kernel runs in is handed to it by its J/K's setup.
    response: ResponseRecords,
}

/// The responses' records of a session, one per purpose (the last response of each).
#[derive(Clone, Debug, Default)]
pub(crate) struct ResponseRecords(Arc<Mutex<Vec<ResponseArithmetic>>>);

impl ResponseRecords {
    pub(crate) fn lock(&self) -> std::sync::MutexGuard<'_, Vec<ResponseArithmetic>> {
        self.0.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Starts the record of a response of `purpose` (replacing an earlier one): its J/K's
    /// device arithmetic (None: the CPU's) and why.
    pub(crate) fn start(&self, purpose: &'static str, jk: Option<&'static str>, reason: String) {
        let mut v = self.lock();
        v.retain(|r| r.purpose != purpose);
        v.push(ResponseArithmetic {
            purpose,
            jk,
            exchange: None,
            operator_gib: None,
            operator_host_gib: None,
            operator_note: None,
            operator_granted_gib: None,
            plan_violation: None,
            estimate_s: None,
            xc: None,
            solver: None,
            assembly: None,
            reason,
        });
    }

    /// Where the CPHF/CPKS solver's algebra of the response of `purpose` ran.
    pub(crate) fn set_solver(&self, purpose: &'static str, at: PlaceRecord) {
        if let Some(r) = self.lock().iter_mut().rev().find(|r| r.purpose == purpose) {
            r.solver = Some(at);
        }
    }

    /// The plan violations of the responses (`plan_violation` warnings).
    pub(crate) fn violations(&self) -> Vec<String> {
        self.lock()
            .iter()
            .filter_map(|r| r.plan_violation.clone())
            .collect()
    }

    fn device_records(&self) -> Vec<ResponseArithmetic> {
        self.lock()
            .iter()
            .filter(|r| r.jk.is_some() || r.xc.is_some())
            .cloned()
            .collect()
    }
}

impl BackendState {
    /// The backend of `b` (checked by [`Backend::check`]): the device of `gpu` (no usable
    /// device is a [`LinusError::Backend`]) or of `auto` (none: the CPU, with the reason); no
    /// device for `cpu`. A device's memory limit is at most `cap_gib` when given, its kernels
    /// cached as `cache` says.
    pub(crate) fn open(
        b: &Backend,
        cap_gib: Option<f64>,
        cache: &KernelCache,
    ) -> Result<Self, LinusError> {
        let mut st = BackendState {
            device: None,
            provider: None,
            no_device: None,
            response: ResponseRecords::default(),
        };
        if b.kind() == BackendKind::Cpu {
            return Ok(st);
        }
        match crate::backend::open_device(b, cap_gib, cache) {
            Ok((d, provider)) => {
                st.device = Some(d);
                st.provider = Some(provider);
            }
            Err(why) if b.kind() == BackendKind::Auto => st.no_device = Some(why),
            Err(why) => {
                return Err(LinusError::Backend {
                    kernel: "device",
                    message: why,
                })
            }
        }
        Ok(st)
    }

    pub(crate) fn device(&self) -> Option<&Device> {
        self.device.as_ref()
    }

    /// The device's provider of engines (None: no device).
    pub(crate) fn provider(&self) -> Option<&dyn crate::backend::Backend> {
        self.provider.as_deref()
    }

    /// The device state the phases of a geometry hand on to each other goes: the atoms moved, or
    /// a new SCF sets up its own ([`crate::backend::Backend::clear_session`]).
    pub(crate) fn clear_session(&self) {
        if let Some(p) = &self.provider {
            p.clear_session();
        }
    }

    /// The derivatives of the last SCF are done with what its setups handed them
    /// ([`crate::backend::Backend::release_handoff`]).
    pub(crate) fn release_handoff(&self) {
        if let Some(p) = &self.provider {
            p.release_handoff();
        }
    }

    /// Backend auto without a device: why.
    pub(crate) fn no_device(&self) -> Option<&String> {
        self.no_device.as_ref()
    }

    /// The responses' records (`resolved.record.backend.response`).
    pub(crate) fn responses(&self) -> &ResponseRecords {
        &self.response
    }

    /// The FP64 emulation the device's last setup of `k` ran
    /// ([`crate::backend::Backend::emulation`]; None without a device).
    pub(crate) fn emulation(&self, k: Family) -> Option<EmulationUsed> {
        self.provider()?.emulation(k)
    }

    /// The families whose last workspace grant the device's engines exceeded, with the first
    /// violation of each ([`crate::backend::Backend::grant_violations`]; none without a device).
    pub(crate) fn grant_violations(&self) -> Vec<(Family, String)> {
        self.provider()
            .map_or_else(Vec::new, |p| p.grant_violations())
    }

    /// Where the device's XC families evaluate the functional
    /// ([`crate::backend::Backend::functional_places`]; none without a device).
    pub(crate) fn functional_places(&self) -> Vec<(Family, Option<String>)> {
        self.provider()
            .map_or_else(Vec::new, |p| p.functional_places())
    }

    /// Where the last grid built with the device partition took its weights, and why
    /// ([`crate::backend::Backend::grid_partition_place`]; None without a device or a grid that
    /// asked it).
    pub(crate) fn grid_partition_place(&self) -> Option<(&'static str, String)> {
        self.provider().and_then(|p| p.grid_partition_place())
    }

    /// Starts the measure of a calculation's peak device memory (nothing without a device).
    pub(crate) fn reset_device_peak(&self) {
        if let Some(p) = &self.provider {
            p.reset_device_peak();
        }
    }

    /// The most device memory held at once since [`BackendState::reset_device_peak`] (bytes;
    /// None without a device).
    pub(crate) fn device_peak(&self) -> Option<usize> {
        self.provider()?.device_peak()
    }
}

impl CalcContext {
    /// The device policy of the session's device (None: no device).
    pub(crate) fn device_policy(&self) -> Option<&DevicePolicy> {
        self.plan.device().map(|d| &d.policy)
    }

    /// Records how the last device setup of `k` went: `Some(reason)` when the CPU took over,
    /// None when it ran on the device.
    pub(crate) fn note_setup(&self, k: Family, fallback: Option<String>) {
        self.record.decide(Decision::Setup {
            kernel: k,
            fallback,
        });
    }

    /// Where the family `k` ran: its planned placement, overridden by its last setup's
    /// fallback (None: the method has no such kernels).
    pub(crate) fn ran(&self, k: Family) -> Option<Placement> {
        match (self.placement(k)?, self.record.setup(k).flatten()) {
            (Placement::Device, Some(why)) => Some(Placement::Cpu(Some(why))),
            (p, _) => Some(p),
        }
    }

    /// With `backend.kind` explicitly `gpu`: a warning (`backend_fallback`, the shared code of decision 23) for each main family
    /// (the SCF's `jk` and `xc`) that ran on the CPU after all, by rule or because its device
    /// setup fell back (the reason is in `resolved.record.backend` as well). A GPU job that
    /// silently runs on the CPU is easy to miss; it stays a warning, not an error.
    pub(crate) fn cpu_fallback_warnings(&self) -> Vec<linus_core::Warning> {
        if self.plan.backend().kind() != BackendKind::Gpu {
            return Vec::new();
        }
        [Family::Jk, Family::Xc]
            .into_iter()
            .filter_map(|k| match self.ran(k) {
                Some(Placement::Cpu(Some(why))) => Some(linus_core::Warning {
                    code: "backend_fallback",
                    message: format!("backend gpu: {} runs on the CPU: {why}", k.name()),
                    path: "backend.kind",
                }),
                _ => None,
            })
            .collect()
    }

    /// Where the kernel family `k` runs for this session's method and settings (None: the
    /// method has no such kernels, e.g. XC for Hartree-Fock), before any setup: the plan's
    /// table ([`crate::driver::plan::Plan::placement`]).
    pub(crate) fn placement(&self, k: Family) -> Option<Placement> {
        self.plan.placement(k)
    }

    /// What ran each family of kernels (`resolved.record.backend`): the placements, overridden
    /// by the fallbacks of the last setups.
    pub fn backend_used(&self) -> ResolvedBackend {
        let mut fallbacks = Vec::new();
        let mut at = |k: Family| -> Option<&'static str> {
            Some(match self.ran(k)? {
                Placement::Device => "gpu",
                Placement::Cpu(why) => {
                    if let Some(reason) = why {
                        fallbacks.push(Fallback {
                            kernel: k.name(),
                            reason,
                        });
                    }
                    "cpu"
                }
            })
        };
        let jk = at(Family::Jk).unwrap_or("cpu");
        let xc = at(Family::Xc);
        let vv10 = at(Family::Vv10);
        let gradient_jk = at(Family::GradientJk).unwrap_or("cpu");
        let gradient_xc = at(Family::GradientXc);
        let hessian_jk = at(Family::HessianJk).unwrap_or("cpu");
        let hessian_xc = at(Family::HessianXc);
        let response_jk = at(Family::ResponseJk);
        let response_xc = at(Family::ResponseXc);
        let scf_loop = match at(Family::ScfLoop) {
            Some("gpu") => "device",
            _ => "host",
        };
        let pcm = at(Family::Pcm);
        let device = self.backend.device().map(|d| d.info.clone());
        // the arithmetic of the device's exchange GEMMs or XC products (the CPU computes in
        // FP64)
        let policy = self.device_policy();
        let precision = policy.map(|p| p.precision);
        let emulated = precision == Some(DevicePrecision::Emulated)
            && (jk == "gpu" || response_jk == Some("gpu") || hessian_jk == "gpu");
        let mixed = policy.is_some_and(|p| p.fp32 == Fp32Mode::On)
            && [gradient_xc, response_xc, hessian_xc, vv10].contains(&Some("gpu"));
        let emulation: Vec<EmulationUsed> = [
            (Family::Jk, jk == "gpu"),
            (Family::ResponseJk, response_jk == Some("gpu")),
            (Family::HessianJk, hessian_jk == "gpu"),
        ]
        .into_iter()
        .filter(|&(_, on)| on && emulated)
        .filter_map(|(k, _)| self.backend.emulation(k))
        .collect();
        let cpu = |on: bool| on.then_some("cpu");
        ResolvedBackend {
            device,
            no_device: self.backend.no_device.clone(),
            precision: if emulated {
                DevicePrecision::Emulated.name()
            } else if mixed {
                "mixed"
            } else {
                DevicePrecision::Double.name()
            },
            arithmetic: policy.map(|p| {
                p.table()
                    .into_iter()
                    .map(|(family, arith, reason)| (family, FamilyArith { arith, reason }))
                    .collect()
            }),
            one_electron: "cpu",
            nuclear_attraction: self.record.nuclear_attraction().map(|(place, reason)| {
                PlaceRecord {
                    place,
                    reason: Some(reason),
                }
            }),
            grid_partition: self.backend.grid_partition_place().map(|(place, reason)| {
                PlaceRecord {
                    place,
                    reason: Some(reason),
                }
            }),
            jk,
            xc,
            vv10,
            gradient_jk,
            gradient_jk_kernels: self
                .record
                .gradient_jk_kernels()
                .map(|(kernels, reason)| crate::driver::report::KernelsRecord { kernels, reason }),
            gradient_xc,
            hessian_jk,
            hessian_xc,
            response_jk,
            response_xc,
            scf_loop,
            // the memory-lean J, or its declined setup (the DF engine's tensor otherwise: absent)
            j: self
                .record
                .scf_df_j()
                .filter(|c| c.engine == "direct" || c.declined.is_some())
                .map(|c| crate::driver::report::JEngine {
                    engine: c.engine,
                    reason: c.reason,
                    kernels: c.kernels.as_ref().map(|k| k.0),
                    kernels_reason: c.kernels.map(|k| k.1),
                    declined: c.declined,
                }),
            pcm,
            solvent: cpu(self.plan.solvent().is_some()),
            dispersion: cpu(self.method.dispersion.is_some()),
            fallbacks,
            plan_violations: self
                .backend
                .grant_violations()
                .into_iter()
                .map(
                    |(k, violation)| crate::driver::report::PlanViolationRecord {
                        kernel: k.name(),
                        violation,
                    },
                )
                .collect(),
            emulation,
            response: self.backend.response.device_records(),
            vv10_arithmetic: policy.filter(|_| vv10 == Some("gpu")).map(|p| {
                crate::driver::report::Vv10Arithmetic {
                    scf: p.vv10_scf.name(),
                    gradient: p.vv10_grad.name(),
                }
            }),
            functional: self
                .backend
                .functional_places()
                .into_iter()
                .map(|(k, why)| crate::driver::report::FunctionalRecord {
                    kernel: k.name(),
                    place: if why.is_none() { "gpu" } else { "cpu" },
                    reason: why,
                })
                .collect(),
        }
    }
}
