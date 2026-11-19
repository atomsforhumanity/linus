//! The GPU's provider (feature `gpu`): linus-gpu's engines on the session's device behind the
//! families' traits, and the device session of a geometry ([`session`]). It is asked only for the
//! families the plan places on the device; a setup that does not fit the device (out of memory)
//! or that its engines do not cover declines with the reason ([`setup`]), and any other device
//! error is the calculation's ([`device_error`]).
//!
//! The adapters of each family set the engines up with the device policy's arithmetic and the
//! device memory plan's reserves (`crate::driver::policy`, `crate::driver::memory`), and record
//! what only the device decides in its setups (the FP64 emulation that ran, how each engine kept
//! to its workspace grant: [`DeviceDecisions`], reported through
//! [`Backend::emulation`] and [`Backend::grant_violations`]). The device loop is
//! [`scf_loop`]'s; its controls stay the SCF's own (`crate::backend::gpu::scf_control`).

pub(crate) mod costing;
pub(crate) mod df;
pub(crate) mod pcm;
pub(crate) mod scf_control;
pub(crate) mod scf_loop;
pub(crate) mod session;
pub(crate) mod xc;

use super::scf::{ScfJkBuilt, ScfRun, ScfTerms, ScfXc};
use super::{Backend, Setup};
use crate::derivatives::terms::{GradientTerm, HessianTerm, TermInputs};
use crate::driver::backend::{Device, DeviceInfo, Family, Placement, ResponseRecords};
use crate::driver::ground::{GroundState, ScfSetup};
use crate::driver::report::EmulationUsed;
use crate::driver::{CalcContext, JkMethod};
use crate::solvent::derivatives::SolventGradient;
use crate::solvent::scf::{PcmSetup, ScfEnvironment};
use crate::solvent::SolventSettings;
use crate::two_electron::{KernelUse, ResponseRequest, ResponseSetup};
use faer::{MatRef, Par};
use linus_base::error::{LinusError, MemoryNeed};
use linus_base::terms::ResponseKernel;
use linus_core::timing::Timers;
use linus_dft::engine::{HessianInput, XcHessianEngine};
use linus_dft::vv10_resp::{Vv10Ground, Vv10PairSweep};
use linus_dft::XcIntegrator;
use linus_gpu::{Gpu, GpuError};
use linus_ints::operator::Operator;
use linus_ints::PairSet;
use linus_scf::ScfLoop;
use session::DeviceSession;
use std::any::Any;
use std::sync::{Arc, Mutex, MutexGuard};

/// The GPU's provider over the session's device (module docs): the device, the device state the
/// phases of the current geometry hand on to each other, and what its setups decided.
pub(crate) struct GpuBackend {
    pub(crate) gpu: Arc<Gpu>,
    pub(crate) session: DeviceSession,
    decisions: Mutex<DeviceDecisions>,
}

/// What only the device decides in its setups, for the session's record (the run record keeps
/// the rest, `crate::driver::record`): per family, the last decision of each kind supersedes the
/// earlier one.
#[derive(Debug, Default)]
pub(crate) struct DeviceDecisions {
    /// The FP64 emulation each family's last device setup ran (None: native FP64 by the
    /// policy).
    emulation: Vec<(Family, Option<EmulationUsed>)>,
    /// Each family's last workspace grant (the device memory plan): its first draw beyond it (a
    /// plan violation), None when the engine stayed within it.
    grants: Vec<(Family, Option<String>)>,
}

/// Records `x` for the family `k` in `v`, superseding its earlier one.
fn supersede<T>(v: &mut Vec<(Family, T)>, k: Family, x: T) {
    v.retain(|e| e.0 != k);
    v.push((k, x));
}

impl GpuBackend {
    /// The GPU's provider of `cx` (None: no device).
    pub(crate) fn of(cx: &CalcContext) -> Option<&GpuBackend> {
        let p: &dyn Any = cx.backend.provider()?;
        p.downcast_ref()
    }

    fn decisions(&self) -> MutexGuard<'_, DeviceDecisions> {
        self.decisions.lock().unwrap_or_else(|e| e.into_inner())
    }
}

/// The device `b.device_id()` of the backend settings `b` and its provider (the reason when it
/// cannot be used): its memory limit at most `cap_gib` when given (`device_memory_gib`), its
/// kernels cached as `cache` says (`backend.kernel_cache`).
pub(crate) fn open(
    b: &crate::driver::options::BackendSettings,
    cap_gib: Option<f64>,
    cache: &crate::driver::options::KernelCache,
) -> Result<(Device, Box<dyn Backend>), String> {
    use crate::driver::options::KernelCache;
    let id = b.device_id();
    let dir = match cache {
        KernelCache::Auto => linus_gpu::nvrtc::default_cache_dir(),
        KernelCache::Off => None,
        KernelCache::Dir(d) => Some(d.clone()),
    };
    let cap = cap_gib.map(|g| (g * 1073741824.0) as usize);
    let gpu = linus_gpu::Gpu::open(id as usize, dir, cap).map_err(|e| e.to_string())?;
    let i = gpu.info();
    let info = DeviceInfo {
        id,
        name: i.name.clone(),
        memory_gib: i.total_mem as f64 / 1073741824.0,
        compute_capability: format!("{}.{}", i.cc.0, i.cc.1),
        memory_limit_gib: gpu.memory().limit as f64 / 1073741824.0,
        memory_cap_gib: cap_gib,
    };
    let card = crate::driver::policy::CardProps {
        fp64_rich: i.fp64_rich(),
        int8_per_fp64: i.int8_per_fp64(),
        cc: i.cc,
        sm_count: i.sm_count,
        clock_khz: i.clock_khz,
    };
    let limit = gpu.budget().limit;
    let d = linus_gpu::response::df::GpuDfResponseOptions::default();
    let device = Device {
        info,
        card,
        limit,
        response_defaults: crate::driver::memory::ResponseDefaults {
            workspace: d.workspace,
            k_operator_reserve: d.k_operator_reserve,
            w_planes_reserve: d.w_planes_reserve,
        },
    };
    let provider = GpuBackend {
        gpu: Arc::new(gpu),
        session: DeviceSession::default(),
        decisions: Mutex::default(),
    };
    Ok((device, Box::new(provider)))
}

/// A device error of the kernel family `k` as the calculation's error, of the kind its device
/// kind maps to (`linus_core::DeviceErrorKind::calculation_kind`): a numerical failure (a
/// matrix not positive definite or singular, a non-finite result, which a CPU run meets too) is
/// [`LinusError::Numerical`], a broken invariant of the device code [`LinusError::Internal`],
/// out of device memory a resource error of code `device_memory` (page-locked host memory that
/// cannot be pinned: `host_memory`), every other one [`LinusError::Backend`].
pub(crate) fn device_error(k: Family, e: GpuError) -> LinusError {
    match e.kind().calculation_kind() {
        linus_core::ErrorKind::Resource => match e {
            GpuError::HostMemory { bytes, .. } => LinusError::host_memory(
                format!(
                    "GPU backend, {}: {e}; give the process more memory (memory_gib, or a larger memory cap) \
                     or run backend \"cpu\"",
                    k.name()
                ),
                MemoryNeed::required_bytes(bytes as f64),
            ),
            e => LinusError::device_memory(
                format!(
                    "GPU backend, {}: {e}; give the process more device memory (device_memory_gib, or a less \
                     busy card) or run backend \"cpu\"",
                    k.name()
                ),
                device_need(&e),
            ),
        },
        linus_core::ErrorKind::Numerical => {
            LinusError::Numerical(format!("GPU backend, {}: {e}", k.name()))
        }
        linus_core::ErrorKind::Internal => {
            LinusError::Internal(format!("GPU backend, {}: {e}", k.name()))
        }
        _ => LinusError::Backend {
            kernel: k.name(),
            message: e.to_string(),
        },
    }
}

/// The outcome of a device setup of the family `k`: the engine; declined out of device memory or
/// in a case the engine does not cover ([`GpuError::falls_back_at_setup`]), with the reason; any
/// other error the device's ([`device_error`]).
pub(crate) fn setup<T>(k: Family, built: Result<T, GpuError>) -> Result<Setup<T>, LinusError> {
    match built {
        Ok(x) => Ok(Setup::Ready(x)),
        Err(e) if e.falls_back_at_setup() => Ok(Setup::Declined(format!("device setup: {e}"))),
        Err(e) => Err(device_error(k, e)),
    }
}

impl CalcContext {
    /// The memory plan of the session's device (its engines' reserves and workspaces).
    pub(crate) fn device_memory(&self) -> Option<crate::driver::memory::DeviceMemoryPlan> {
        self.plan.device().map(|d| d.memory)
    }

    /// The device's handle when the kernel family `k` runs on it.
    pub(crate) fn device_for(&self, k: Family) -> Option<&Arc<Gpu>> {
        match self.placement(k) {
            Some(Placement::Device) => GpuBackend::of(self).map(|d| &d.gpu),
            _ => None,
        }
    }

    /// The device session of the current geometry ([`session`]; None: no device).
    pub(crate) fn device_session(&self) -> Option<&DeviceSession> {
        GpuBackend::of(self).map(|d| &d.session)
    }

    /// The outcome of a device setup of a family's part `k` that the GPU's adapter places itself
    /// (VV10 inside the device's XC terms): the engine, None for a setup that declines ([`setup`];
    /// the CPU's part runs, with the reason recorded for `resolved.record.backend`), or the
    /// device's error.
    pub(crate) fn device_setup<T>(
        &self,
        k: Family,
        built: Result<T, GpuError>,
    ) -> Result<Option<T>, LinusError> {
        Ok(match setup(k, built)? {
            Setup::Ready(x) => {
                self.note_setup(k, None);
                Some(x)
            }
            Setup::Declined(why) => {
                self.note_setup(k, Some(why));
                None
            }
        })
    }

    /// Records the FP64 emulation the last device setup of `k` ran (None: native FP64 by the
    /// policy).
    pub(crate) fn note_emulation(&self, k: Family, used: Option<EmulationUsed>) {
        if let Some(d) = GpuBackend::of(self) {
            supersede(&mut d.decisions().emulation, k, used);
        }
    }

    /// Records how the engine of `k` kept to its workspace grant: its first draw beyond it
    /// (`violations`, [`linus_gpu::pool::Grant::violations`]), if any.
    pub(crate) fn note_grant(&self, k: Family, violations: &[linus_gpu::pool::PlanViolation]) {
        let violation = violations.first().map(|v| {
            format!(
                "{}: {} bytes drawn with {} of the {} granted left",
                v.what, v.requested, v.remaining, v.granted
            )
        });
        if let Some(d) = GpuBackend::of(self) {
            supersede(&mut d.decisions().grants, k, violation);
        }
    }
}

/// The device's records of a session's responses (`resolved.record.backend.response`).
impl ResponseRecords {
    /// The device bytes granted to the last response of `purpose`'s stored operator at its
    /// setup (0: none) and the build's draws beyond them (plan violations).
    pub(crate) fn set_grant(
        &self,
        purpose: &'static str,
        granted: usize,
        violations: &[linus_gpu::pool::PlanViolation],
    ) {
        if let Some(r) = self.lock().iter_mut().rev().find(|r| r.purpose == purpose) {
            r.operator_granted_gib = (granted > 0).then(|| granted as f64 / 1073741824.0);
            r.plan_violation = violations.first().map(|v| {
                format!(
                    "{}: {} bytes drawn with {} of the {} granted left",
                    v.what, v.requested, v.remaining, v.granted
                )
            });
        }
    }

    /// The exchange estimates of `auto`'s choice of path for the last response (decision D37).
    pub(crate) fn set_estimate(&self, estimate: crate::driver::report::ExchangeEstimate) {
        if let Some(r) = self.lock().last_mut() {
            r.estimate_s = Some(estimate);
        }
    }

    /// The device's CPKS kernel arithmetic of the response of `purpose`.
    pub(crate) fn set_xc(&self, purpose: &'static str, xc: &'static str) {
        if let Some(r) = self.lock().iter_mut().rev().find(|r| r.purpose == purpose) {
            r.xc = Some(xc);
        }
    }

    /// The device exchange of the response of `purpose` as it ran: its arithmetic (`auto`
    /// falls back to FP32 when the operator does not fit), on the stored operator (its GiB, and
    /// of them in host memory) or the tensor (with the reason the operator was declined, if
    /// any).
    pub(crate) fn set_exchange(
        &self,
        purpose: &'static str,
        jk: &'static str,
        operator_gib: Option<(f64, f64)>,
        note: Option<String>,
    ) {
        if let Some(r) = self.lock().iter_mut().rev().find(|r| r.purpose == purpose) {
            r.jk = Some(jk);
            r.exchange = Some(if operator_gib.is_some() {
                "operator"
            } else {
                "tensor"
            });
            r.operator_gib = operator_gib.map(|g| g.0);
            r.operator_host_gib = operator_gib.and_then(|g| (g.1 > 0.0).then_some(g.1));
            r.operator_note = note;
        }
    }
}

/// The reason a family the plan places on the device has no device engine for the J/K method
/// (the plan puts only density fitting there).
fn not_df(cx: &CalcContext) -> String {
    format!("jk {}: no device engines", cx.plan.jk().name())
}

impl Backend for GpuBackend {
    fn clear_session(&self) {
        self.session.clear();
    }

    fn release_handoff(&self) {
        self.session.release_df();
    }

    fn emulation(&self, k: Family) -> Option<EmulationUsed> {
        self.decisions()
            .emulation
            .iter()
            .find(|e| e.0 == k)
            .and_then(|e| e.1.clone())
    }

    fn grant_violations(&self) -> Vec<(Family, String)> {
        self.decisions()
            .grants
            .iter()
            .filter_map(|(k, v)| Some((*k, v.clone()?)))
            .collect()
    }

    fn reset_device_peak(&self) {
        self.gpu.reset_peak();
    }

    /// The device pool's high-water mark ([`Gpu::peak`]: its live buffers and grants).
    fn device_peak(&self) -> Option<usize> {
        Some(self.gpu.peak())
    }

    /// The GPU's generated integral kernels' coverage of the basis sets (without an auxiliary
    /// basis: the orbital basis alone).
    fn uncovered(
        &self,
        basis: &linus_ints::BasisSet,
        aux: Option<&linus_ints::BasisSet>,
    ) -> Option<String> {
        let none;
        let aux = match aux {
            Some(a) => a,
            None => {
                none = linus_ints::BasisSet {
                    shells: Vec::new(),
                    ..basis.clone()
                };
                &none
            }
        };
        linus_gpu::eri3c::covers(basis, aux).err().map(|why| {
            format!("the GPU's generated integral kernels do not cover the basis sets ({why})")
        })
    }

    fn scf_jk(
        &self,
        cx: &CalcContext,
        _pairs: &PairSet,
        setup: &ScfSetup,
        timers: &mut Timers,
    ) -> Result<Setup<ScfJkBuilt>, LinusError> {
        let JkMethod::Df { aux } = cx.plan.jk() else {
            return Ok(Setup::Declined(not_df(cx)));
        };
        let aux = cx.aux_set(aux, setup.purpose)?;
        crate::backend::gpu::df::scf_jk(cx, &self.gpu, setup, &aux, timers)
    }

    fn scf_xc(
        &self,
        cx: &CalcContext,
        xc: Arc<XcIntegrator>,
        timers: &mut Timers,
    ) -> Result<Setup<Box<dyn ScfXc>>, LinusError> {
        crate::backend::gpu::xc::scf_potential(cx, &self.gpu, xc, timers)
    }

    fn takes_scf_loop(&self, _cx: &CalcContext, terms: &ScfTerms) -> Result<Setup<()>, LinusError> {
        Ok(scf_loop::takes(terms))
    }

    fn scf_loop<'a>(
        &self,
        _cx: &'a CalcContext,
        terms: &'a mut ScfTerms,
        run: ScfRun<'a>,
    ) -> Box<dyn ScfLoop + 'a> {
        Box::new(scf_loop::DeviceScf::new(terms, run))
    }

    fn scf_environment(
        &self,
        cx: &CalcContext,
        setup: &ScfSetup,
        scf: &mut dyn ScfLoop,
        pcm: &PcmSetup,
    ) -> Result<Setup<Box<dyn ScfEnvironment>>, LinusError> {
        crate::backend::gpu::pcm::GpuScfEnvironment::open(cx, &self.gpu, setup, scf, pcm)
    }

    fn gradient_xc<'a>(
        &self,
        cx: &'a CalcContext,
        _grid_response: bool,
    ) -> Result<Setup<Box<dyn GradientTerm + 'a>>, LinusError> {
        Ok(Setup::Ready(Box::new(
            crate::backend::gpu::xc::GpuXcGradient::new(cx, self.gpu.clone()),
        )))
    }

    fn gradient_jk<'a>(
        &self,
        cx: &'a CalcContext,
        xop: Option<Operator>,
    ) -> Result<Setup<Box<dyn GradientTerm + 'a>>, LinusError> {
        Ok(Setup::Ready(Box::new(
            crate::backend::gpu::df::GpuJkGradient::new(cx, self.gpu.clone(), xop),
        )))
    }

    fn solvent_gradient<'a>(
        &self,
        cx: &'a CalcContext,
        _gs: &GroundState,
        so: &SolventSettings,
        timers: &mut Timers,
    ) -> Result<Setup<Box<dyn SolventGradient + 'a>>, LinusError> {
        Ok(
            match crate::backend::gpu::pcm::GpuSolventGradient::new(
                cx,
                self.gpu.clone(),
                so,
                timers,
            )? {
                Setup::Ready(t) => Setup::Ready(Box::new(t)),
                Setup::Declined(why) => Setup::Declined(why),
            },
        )
    }

    fn hessian_jk<'a>(
        &self,
        cx: &'a CalcContext,
        inputs: &TermInputs<'_>,
        xop: Option<Operator>,
        _pairs: &'a PairSet,
        timers: &mut Timers,
    ) -> Result<Setup<Box<dyn HessianTerm + 'a>>, LinusError> {
        let JkMethod::Df { aux } = cx.plan.jk() else {
            return Ok(Setup::Declined(not_df(cx)));
        };
        let aux = cx.aux_set(aux, inputs.ground.purpose())?;
        let gpu = self.gpu.clone();
        Ok(
            match crate::backend::gpu::df::GpuHessianJk::new(
                cx,
                gpu,
                &inputs.c_occ,
                aux,
                xop,
                timers,
            )? {
                Setup::Ready(t) => Setup::Ready(Box::new(t)),
                Setup::Declined(why) => Setup::Declined(why),
            },
        )
    }

    fn hessian_xc<'a>(
        &self,
        cx: &'a CalcContext,
        xc: &'a XcIntegrator,
        input: &HessianInput<'_>,
        timers: &mut Timers,
    ) -> Result<Setup<Box<dyn XcHessianEngine + 'a>>, LinusError> {
        crate::backend::gpu::xc::hessian_xc(cx, &self.gpu, xc, input, timers)
    }

    fn response_jk(
        &self,
        cx: &CalcContext,
        req: &ResponseRequest<'_>,
        field: bool,
        _pairs: &PairSet,
        _par: Par,
        timers: &mut Timers,
    ) -> Result<Setup<ResponseSetup>, LinusError> {
        let JkMethod::Df { aux } = cx.plan.jk() else {
            return Ok(Setup::Declined(not_df(cx)));
        };
        let aux = cx.aux_set(aux, req.ground.purpose())?;
        let rp = cx.df_response_plan(req, &aux);
        let setup =
            crate::backend::gpu::df::response_setup(cx, &self.gpu, req, field, &aux, &rp, timers)?;
        // the response's setup is the last use of the geometry's metric (crate::context)
        if matches!(setup, Setup::Ready(_)) {
            cx.integrals.release_metrics();
        }
        Ok(setup)
    }

    fn response_kernel<'a>(
        &self,
        cx: &CalcContext,
        xc: &'a XcIntegrator,
        occ: &[MatRef<'_, f64>],
        kernel: KernelUse,
        timers: &mut Timers,
    ) -> Result<Setup<Box<dyn ResponseKernel + 'a>>, LinusError> {
        crate::backend::gpu::xc::response_kernel(cx, &self.gpu, xc, occ, kernel, timers)
    }

    fn vv10_sweeps(
        &self,
        cx: &CalcContext,
        ground: &Vv10Ground,
        prefix: &str,
        timers: &mut Timers,
    ) -> Result<Setup<Option<Box<dyn Vv10PairSweep>>>, LinusError> {
        timers.time(&format!("{prefix}_vv10_device_setup"), || {
            crate::backend::gpu::xc::vv10_sweeps(cx, &self.gpu, ground)
        })
    }
}

/// The numbers of a device's out-of-memory error (qc*'s `ErrorInfo.details` of
/// `device_memory`): what it had allocated plus the request that failed, against its limit (the
/// memory pool's: `device_memory_gib`, else the card's). None for other errors.
pub(crate) fn device_need(e: &GpuError) -> MemoryNeed {
    match e {
        GpuError::OutOfMemory {
            requested,
            in_use,
            limit,
        } => MemoryNeed::bytes((in_use + requested) as f64, *limit as f64),
        _ => MemoryNeed::default(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A setup falls back to the CPU only out of device memory or in a case the engine does not
    /// cover (`device_setup`: [`linus_gpu::GpuError::falls_back_at_setup`], the reason recorded);
    /// otherwise [`device_error`]: a numerical failure is a numerical error and a library fault
    /// a backend error (decision D38; cuSOLVER's "not positive definite" was a silent CPU run).
    #[test]
    fn setup_failures_fall_back_only_out_of_memory_or_not_covered() {
        let oom = GpuError::OutOfMemory {
            requested: 1,
            in_use: 0,
            limit: 0,
        };
        assert!(oom.falls_back_at_setup());
        assert!(GpuError::NotCovered("not covered".into()).falls_back_at_setup());
        let not_pd = GpuError::Numerical {
            op: "cuSOLVER potrf",
            info: 3,
        };
        assert!(!not_pd.falls_back_at_setup());
        match device_error(Family::Jk, not_pd) {
            LinusError::Numerical(m) => assert_eq!(
                m,
                "GPU backend, jk: cuSOLVER potrf: the leading minor of order 3 is not positive definite"
            ),
            other => panic!("{other:?}"),
        }
        let fault = GpuError::Library {
            lib: linus_gpu::error::Lib::Cusolver,
            call: Some("potrf"),
            status: "CUSOLVER_STATUS_INVALID_VALUE".into(),
        };
        assert!(!fault.falls_back_at_setup());
        assert!(matches!(
            device_error(Family::Jk, fault),
            LinusError::Backend { kernel: "jk", .. }
        ));
        // out of device memory after the setup: kind resource, code device_memory (decision 23)
        let e = device_error(Family::Jk, oom);
        assert_eq!(
            (e.kind(), e.code()),
            ("resource", Some("device_memory")),
            "{e}"
        );
        let pinned = GpuError::HostMemory {
            bytes: 1,
            why: "mlock".into(),
        };
        let e = device_error(Family::Jk, pinned);
        assert_eq!(
            (e.kind(), e.code()),
            ("resource", Some("host_memory")),
            "{e}"
        );
        // a broken invariant of the device code is a bug: kind internal
        let bug = GpuError::Contract("eri3c: output buffer too small".into());
        assert!(!bug.falls_back_at_setup());
        match device_error(Family::Jk, bug) {
            LinusError::Internal(m) => {
                assert_eq!(m, "GPU backend, jk: eri3c: output buffer too small")
            }
            other => panic!("{other:?}"),
        }
    }
}
