//! The CPU/GPU boundary: one provider per backend ([`Backend`]), asked for the engine of each
//! method family at each derivative order. A provider answers with the family's engine as a
//! trait object ([`Setup::Ready`]) or declines with the reason ([`Setup::Declined`]).
//!
//! - **Placement picks the provider.** The plan places every family of kernels before any work
//!   (`crate::driver::plan::Plan::placement`, over [`Family`]): the device's provider answers for a
//!   family placed on the device, the CPU's ([`CpuBackend`]) for every other.
//! - **One "device, else CPU".** [`CalcContext::resolve`] asks the placed provider. A device setup
//!   that declines (out of device memory, or a case its engines do not cover) is recorded with its
//!   reason for `resolved.record.backend`, and the CPU's provider answers instead; a device setup on
//!   the device is recorded as such. Any other device error is the calculation's (a `backend`,
//!   `numerical` or `internal` error). The CPU's provider always answers. A family never changes
//!   backend after its setup, so a calculation is reproducible run to run.
//! - **The engines are the families' own traits**, which the derivatives, responses and the SCF
//!   call without knowing the backend: the Hessian's terms
//!   ([`crate::derivatives::terms::HessianTerm`], the XC engine's [`XcHessianEngine`]), the
//!   response's J/K ([`linus_base::terms::ResponseJk`]) and CPKS kernel
//!   ([`linus_base::terms::ResponseKernel`]), VV10's pair sweeps ([`Vv10PairSweep`]).
//! - **The CPU's provider is thin**: it hands out the CPU's engines of each family, which the
//!   families' own modules choose (the J/K method's in `crate::two_electron`, the integrator's in
//!   `linus_dft`). The GPU's provider (module `gpu`, feature `gpu`) wraps linus-gpu's engines in
//!   the same traits; it is the only code here that names linus-gpu.
//! - **The `gpu` feature appears in this file alone** in the facade: the GPU's provider and what
//!   only a device has (its setups' FP64 emulation and workspace grants, its peak memory) are
//!   reached through [`Backend`]'s methods, whose CPU answers are the defaults, and a lost
//!   context through [`device_lost`].

mod cpu;
#[cfg(feature = "gpu")]
pub(crate) mod gpu;
pub(crate) mod scf;

pub(crate) use cpu::CpuBackend;

use crate::derivatives::terms::{GradientTerm, HessianTerm, TermInputs};
use crate::driver::backend::{Device, Family, Placement};
use crate::driver::ground::{GroundState, ScfSetup};
use crate::driver::report::EmulationUsed;
use crate::driver::CalcContext;
use crate::solvent::derivatives::SolventGradient;
use crate::solvent::scf::{PcmSetup, ScfEnvironment};
use crate::solvent::SolventSettings;
use crate::two_electron::{KernelUse, ResponseRequest, ResponseSetup};
use faer::{MatRef, Par};
use linus_base::error::LinusError;
use linus_base::terms::ResponseKernel;
use linus_core::timing::Timers;
use linus_dft::engine::{HessianInput, XcHessianEngine};
use linus_dft::vv10_resp::{Vv10Ground, Vv10PairSweep};
use linus_dft::XcIntegrator;
use linus_ints::operator::Operator;
use linus_ints::PairSet;
use linus_scf::ScfLoop;
use scf::{ScfJkBuilt, ScfRun, ScfTerms, ScfXc};
use std::any::Any;
use std::sync::Arc;

/// Whether this linus was built with GPU support (the `gpu` cargo feature): the GPU's provider
/// exists, and `backend: "gpu"` is a valid setting. Public as [`crate::GPU_BUILD`] (Python's
/// `linus.GPU_BUILD`: true in the `afh-linus-cuda13` distribution).
pub(crate) const GPU_BUILT: bool = cfg!(feature = "gpu");

/// A provider's answer for a family: its engine, or why it does not set one up (the next
/// provider, the CPU's, then answers).
pub(crate) enum Setup<T> {
    Ready(T),
    /// Only a device's provider declines (the CPU's always answers).
    #[cfg_attr(not(feature = "gpu"), allow(dead_code))]
    Declined(String),
}

/// A backend's provider of engines, one factory per method family and derivative order (module
/// docs). A factory sets its engine up; the engine computes. Every factory answers on the CPU.
pub(crate) trait Backend: Any + Send + Sync {
    /// The device state the phases of a geometry hand on to each other goes (the atoms moved, or
    /// a new SCF sets up its own); nothing on the CPU.
    fn clear_session(&self) {}

    /// The derivatives of the last SCF are done with what its setups handed them (the device's
    /// density-fitting data); nothing on the CPU.
    fn release_handoff(&self) {}

    /// The FP64 emulation this backend's last setup of the family `k` ran
    /// (`resolved.record.backend.emulation`; None: native FP64, or no setup). The CPU computes in
    /// FP64.
    fn emulation(&self, _k: Family) -> Option<EmulationUsed> {
        None
    }

    /// The families whose engines drew beyond their last workspace grant (the device memory
    /// plan), each with its first violation, in the order recorded; none on the CPU, which grants
    /// no device memory.
    fn grant_violations(&self) -> Vec<(Family, String)> {
        Vec::new()
    }

    /// Starts the measure of a calculation's peak device memory ([`Backend::device_peak`]);
    /// nothing on the CPU.
    fn reset_device_peak(&self) {}

    /// The most device memory held at once since the last [`Backend::reset_device_peak`]
    /// (bytes; None on the CPU).
    fn device_peak(&self) -> Option<usize> {
        None
    }

    /// Why this backend's kernels do not cover the orbital basis `basis` and the auxiliary basis
    /// `aux` (None: they do, the default), which the plan's placement takes (decision D40).
    fn uncovered(
        &self,
        _basis: &linus_ints::BasisSet,
        _aux: Option<&linus_ints::BasisSet>,
    ) -> Option<String> {
        None
    }

    /// The SCF's J/K builder set up with `setup` (family `jk`; `pairs` the geometry's shell
    /// pairs) and what its setup resolved to (`resolved.record.df`).
    fn scf_jk(
        &self,
        cx: &CalcContext,
        pairs: &PairSet,
        setup: &ScfSetup,
        timers: &mut Timers,
    ) -> Result<Setup<ScfJkBuilt>, LinusError>;

    /// The SCF's XC term over the geometry's integrator `xc` (families `xc` and `vv10`).
    fn scf_xc(
        &self,
        cx: &CalcContext,
        xc: Arc<XcIntegrator>,
        timers: &mut Timers,
    ) -> Result<Setup<Box<dyn ScfXc>>, LinusError>;

    /// Whether the SCF's terms run on this backend's loop (family `scf_loop`); declined: the
    /// host loop, with why.
    fn takes_scf_loop(&self, cx: &CalcContext, terms: &ScfTerms) -> Result<Setup<()>, LinusError>;

    /// The loop over the SCF's terms (taken by [`Backend::takes_scf_loop`]).
    fn scf_loop<'a>(
        &self,
        cx: &'a CalcContext,
        terms: &'a mut ScfTerms,
        run: ScfRun<'a>,
    ) -> Box<dyn ScfLoop + 'a>;

    /// The solvent's term of an SCF set up with `setup` on the loop `scf` (family `pcm`), its
    /// Coulomb hook installed on the loop's J build.
    fn scf_environment(
        &self,
        cx: &CalcContext,
        setup: &ScfSetup,
        scf: &mut dyn ScfLoop,
        pcm: &PcmSetup,
    ) -> Result<Setup<Box<dyn ScfEnvironment>>, LinusError>;

    /// The XC engine's part of a gradient (family `gradient_xc`, VV10 included), with the grid's
    /// response when `grid_response`.
    fn gradient_xc<'a>(
        &self,
        cx: &'a CalcContext,
        grid_response: bool,
    ) -> Result<Setup<Box<dyn GradientTerm + 'a>>, LinusError>;

    /// The two-electron method's part of a gradient (family `gradient_jk`) with the exact exchange
    /// `xop`.
    fn gradient_jk<'a>(
        &self,
        cx: &'a CalcContext,
        xop: Option<Operator>,
    ) -> Result<Setup<Box<dyn GradientTerm + 'a>>, LinusError>;

    /// The solvent `so`'s fixed-density gradient at the ground state `gs` (family `pcm`).
    fn solvent_gradient<'a>(
        &self,
        cx: &'a CalcContext,
        gs: &GroundState,
        so: &SolventSettings,
        timers: &mut Timers,
    ) -> Result<Setup<Box<dyn SolventGradient + 'a>>, LinusError>;

    /// The two-electron method's explicit Hessian terms and their F^x C_s (family
    /// `hessian_jk`) at the ground state of `inputs`, with the exact exchange `xop`; `pairs` are
    /// the geometry's shell pairs of derivative order two.
    fn hessian_jk<'a>(
        &self,
        cx: &'a CalcContext,
        inputs: &TermInputs<'_>,
        xop: Option<Operator>,
        pairs: &'a PairSet,
        timers: &mut Timers,
    ) -> Result<Setup<Box<dyn HessianTerm + 'a>>, LinusError>;

    /// The XC engine's semilocal explicit Hessian terms (family `hessian_xc`) on the integrator
    /// `xc`, for the densities and orbitals of `input`.
    fn hessian_xc<'a>(
        &self,
        cx: &'a CalcContext,
        xc: &'a XcIntegrator,
        input: &HessianInput<'_>,
        timers: &mut Timers,
    ) -> Result<Setup<Box<dyn XcHessianEngine + 'a>>, LinusError>;

    /// The response J/K of the request `req` (family `response_jk`), with the plan of a two-level
    /// response's iteration J/K and the arithmetic the CPKS kernel follows ([`ResponseSetup`]);
    /// `field`: an IR run's electric-field solve follows (the device's exchange costing of
    /// `precision.response: auto` weighs its products too, decision D37; the CPU's J/K does not
    /// depend on it).
    fn response_jk(
        &self,
        cx: &CalcContext,
        req: &ResponseRequest<'_>,
        field: bool,
        pairs: &PairSet,
        par: Par,
        timers: &mut Timers,
    ) -> Result<Setup<ResponseSetup>, LinusError>;

    /// The semilocal CPKS kernel of the integrator `xc` (family `response_xc`) for the occupied
    /// orbitals `occ` (one matrix per spin channel), serving the response `kernel`.
    fn response_kernel<'a>(
        &self,
        cx: &CalcContext,
        xc: &'a XcIntegrator,
        occ: &[MatRef<'_, f64>],
        kernel: KernelUse,
        timers: &mut Timers,
    ) -> Result<Setup<Box<dyn ResponseKernel + 'a>>, LinusError>;

    /// The engine of VV10's response pair sweeps at its ground state `ground` (family `vv10`;
    /// None: the ground state's own sweeps on the CPU), its setup timed as
    /// `<prefix>_vv10_device_setup`.
    fn vv10_sweeps(
        &self,
        cx: &CalcContext,
        ground: &Vv10Ground,
        prefix: &str,
        timers: &mut Timers,
    ) -> Result<Setup<Option<Box<dyn Vv10PairSweep>>>, LinusError>;
}

/// The device of the backend settings `b` (backend `gpu` or `auto`) and its provider, or why it
/// cannot be used (a build without the `gpu` feature opens none); its memory limit at most
/// `cap_gib` (`device_memory_gib`) when given, its kernels cached as `cache` says.
pub(crate) fn open_device(
    b: &crate::driver::options::BackendSettings,
    cap_gib: Option<f64>,
    cache: &crate::driver::options::KernelCache,
) -> Result<(Device, Box<dyn Backend>), String> {
    #[cfg(feature = "gpu")]
    return gpu::open(b, cap_gib, cache);
    #[cfg(not(feature = "gpu"))]
    {
        let _ = (b, cap_gib, cache);
        Err("this linus was built without GPU support (the gpu cargo feature)".into())
    }
}

/// Whether a GPU fault poisoned this process's CUDA context (an illegal address, a launch
/// failure, a device assertion, an uncorrectable ECC error): no device work can succeed in this
/// process any more, so a worker answers its request with the error and exits. Always false
/// without the `gpu` feature.
pub fn device_lost() -> bool {
    #[cfg(feature = "gpu")]
    {
        linus_gpu::error::context_lost()
    }
    #[cfg(not(feature = "gpu"))]
    {
        false
    }
}
impl CalcContext {
    /// The engine of the family `k` from the provider the plan puts it on: the device's, else
    /// the CPU's; `setup` asks a provider for it. A declined device setup is recorded with its
    /// reason and the CPU's provider answers; one that sets up is recorded as on the device
    /// (`resolved.record.backend`; [`CalcContext::ran`] then tells where it ran).
    pub(crate) fn resolve<T>(
        &self,
        k: Family,
        mut setup: impl FnMut(&dyn Backend) -> Result<Setup<T>, LinusError>,
    ) -> Result<T, LinusError> {
        if let (Some(Placement::Device), Some(device)) =
            (self.placement(k), self.backend.provider())
        {
            match setup(device)? {
                Setup::Ready(x) => {
                    self.note_setup(k, None);
                    return Ok(x);
                }
                Setup::Declined(why) => self.note_setup(k, Some(why)),
            }
        }
        match setup(&CpuBackend)? {
            Setup::Ready(x) => Ok(x),
            Setup::Declined(why) => Err(LinusError::Internal(format!(
                "the CPU declined {}: {why}",
                k.name()
            ))),
        }
    }
}
