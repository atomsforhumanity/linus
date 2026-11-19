//! The registry of the hand-written CUDA modules: each declared once, by the engine that owns it
//! ([`KernelModule`]: its name, its source, its entry points), and listed here ([`modules`]), so
//! that a test can compile every one of them with NVRTC without a device (tests/nvrtc.rs) and a
//! manifest can hash them.
//!
//! The sources are files under `crates/linus-gpu/device/` (`include_str!`), assembled by the
//! owning engine (shared headers in front: `common.cuh` (`COMMON_H`), `detsum.h`; compile-time
//! variants as `#define`s or substitutions); the device layer prepends the warp size and
//! `portable.h` ([`crate::Gpu::kernel_source`]). Kernels several engines launch are modules of
//! their own, compiled once: the vector operations ([`crate::linalg::Blas1`]) and the DF
//! derivative engines' common kernels (`df_deriv::DerivKernels`). The
//! generated families keep their own tables: the three-center integrals
//! ([`super::eri3c_gen`], assembled by [`crate::eri3c::module_source`]), the PCM's surface
//! charges ([`super::pcm_gen`]), and the run-time generators ([`super::colloc`],
//! [`super::vgemm`]).

use std::borrow::Cow;

/// The device functions shared by modules (`device/common.cuh`: the packed lower-triangle
/// index), pasted in front of the sources that use them.
pub(crate) const COMMON_H: &str = include_str!("../../device/common.cuh");

/// A hand-written CUDA module: what an engine compiles with [`crate::Gpu::kernel`].
pub struct KernelModule {
    /// The module's name (the kernel cache's and the kernel dump's).
    pub name: &'static str,
    source: fn() -> Cow<'static, str>,
    /// The kernels the engines load from it.
    pub entries: &'static [&'static str],
}

impl KernelModule {
    pub const fn new(
        name: &'static str,
        source: fn() -> Cow<'static, str>,
        entries: &'static [&'static str],
    ) -> KernelModule {
        KernelModule {
            name,
            source,
            entries,
        }
    }

    /// The source as the engine passes it to the device layer.
    pub fn source(&self) -> Cow<'static, str> {
        (self.source)()
    }
}

/// Every hand-written module, once.
pub fn modules() -> Vec<&'static KernelModule> {
    vec![
        &crate::detsum::MODULE,
        &crate::linalg::blas1::MODULE,
        &crate::df_deriv::MODULE,
        &crate::emu::MODULE,
        &crate::df::MODULE,
        &crate::df::FIT_MODULE,
        &crate::df::OZAKI_MODULE,
        &crate::grad::df::MODULE,
        &crate::hess::df::MODULE,
        &crate::response::df::MODULE,
        &crate::xc::MODULE,
        &crate::xc::layout::MODULE,
        &crate::grad::xc::MODULE,
        &crate::hess::xc::MODULE,
        &crate::response::xc::MODULE,
        &crate::response::xc::MODULE_F32,
        &crate::xc::vv10::MODULE,
        &crate::xc::vv10::RESP_MODULE,
        &crate::pcm::MODULE,
        &crate::pcm::solver::MODULE,
        &crate::scf::kernels::MODULE,
    ]
}
