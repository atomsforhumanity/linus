//! What the device SCF loop takes from the GPU backend's SCF J/K builders and gives back: the
//! J engine (the DF engine with its auxiliary basis, or the memory-lean J), and rij_cgk's
//! grid-K term ([`DeviceJk`]), from any builder that has them ([`DeviceScfJk`]: df's
//! `GpuScfJk`, df's memory-lean `GpuJDirectJ`, rij_cgk's `GpuCgkJk`). The loop composes them
//! as linus-gpu's `DeviceTerms`; the builders run the host loop on the same engines once they
//! are back.

use crate::backend::gpu::df::GpuScfJk;
use crate::backend::gpu::gridk::GpuCgkJk;
use crate::backend::gpu::jdirect::GpuJDirectJ;
use crate::backend::scf::ScfJk;
use faer::Par;
use linus_base::error::LinusError;
use linus_gpu::gridk::GridKTerm;
use linus_gpu::scf::DeviceJ;
use linus_gpu::Gpu;
use linus_ints::BasisSet;
use std::any::Any;
use std::sync::Arc;

/// The J/K engines of an SCF on the device loop.
pub(crate) struct DeviceJk {
    pub gpu: Arc<Gpu>,
    pub j: DeviceJ,
    /// The DF engine's auxiliary basis (the predicted PCM's Coulomb fit), with J on it.
    pub aux: Option<BasisSet>,
    /// rij_cgk's exchange (K and E_K); None: the DF engine's K, or no exact exchange.
    pub gridk: Option<GridKTerm>,
}

/// A GPU builder of the SCF's J/K whose engines the device loop takes.
pub(crate) trait DeviceScfJk {
    /// The engines (None once taken: the device loop has them).
    fn take_device(&mut self, par: Par) -> Result<Option<DeviceJk>, LinusError>;

    /// The engines back after the device loop (the host loop continues on them, the builder's
    /// timers read them).
    fn put_device(&mut self, jk: DeviceJk) -> Result<(), LinusError>;

    /// Whether J runs on the DF engine, which hands its fit coefficients to a Coulomb hook.
    fn coulomb_fit(&self) -> bool;
}

/// The device loop's view of the SCF's J/K builder (None: the CPU's).
pub(crate) fn of(jk: &mut dyn ScfJk) -> Option<&mut dyn DeviceScfJk> {
    let jk: &mut dyn Any = jk;
    if jk.is::<GpuScfJk>() {
        return jk
            .downcast_mut::<GpuScfJk>()
            .map(|b| b as &mut dyn DeviceScfJk);
    }
    if jk.is::<GpuCgkJk>() {
        return jk
            .downcast_mut::<GpuCgkJk>()
            .map(|b| b as &mut dyn DeviceScfJk);
    }
    jk.downcast_mut::<GpuJDirectJ>()
        .map(|b| b as &mut dyn DeviceScfJk)
}

/// Whether the SCF's J/K builder is the device's ([`of`]).
pub(crate) fn is_device(jk: &dyn ScfJk) -> bool {
    let jk: &dyn Any = jk;
    jk.is::<GpuScfJk>() || jk.is::<GpuCgkJk>() || jk.is::<GpuJDirectJ>()
}

/// The internal error of engines handed back to a builder of another kind.
pub(crate) fn mismatch(what: &str) -> LinusError {
    LinusError::Internal(format!(
        "the device loop handed {what} back to the wrong J/K builder"
    ))
}

impl DeviceScfJk for GpuScfJk {
    fn take_device(&mut self, _par: Par) -> Result<Option<DeviceJk>, LinusError> {
        Ok(self.take().map(|b| {
            let (gpu, jk, aux) = b.into_parts();
            DeviceJk {
                gpu,
                j: DeviceJ::Df(Box::new(jk)),
                aux: Some(aux),
                gridk: None,
            }
        }))
    }

    fn put_device(&mut self, jk: DeviceJk) -> Result<(), LinusError> {
        match jk {
            DeviceJk {
                gpu,
                j: DeviceJ::Df(eng),
                aux: Some(aux),
                gridk: None,
            } => {
                self.put(crate::backend::gpu::df::GpuDfJkBuilder::from_parts(
                    gpu, *eng, aux,
                ));
                Ok(())
            }
            _ => Err(mismatch("engines other than the DF engine")),
        }
    }

    fn coulomb_fit(&self) -> bool {
        true
    }
}
