//! The device session of one geometry (the GPU provider's): the device state that the phases of a
//! geometry hand on to each other, with one owner and explicit lifetimes (review F9: it was
//! spread over a backend hand-off, the solvation results and the response J/K).
//!
//! - **The density-fitting hand-off** of the SCF's J/K setup ([`linus_gpu::df::DfShared`]: the
//!   kept pairs and their rows, the packed three-center integral data, the Coulomb metric, and
//!   the derivative records once built), taken by the device's DF gradient, its Hessian terms
//!   and the response J/K of a Hessian. The derivative records the derivative will need (first
//!   derivatives; for a Hessian also second) are packed on a thread of their own while the SCF
//!   runs, and copied with the integral records and the metric into page-locked memory, which
//!   the engines' setups upload by DMA ([`CalcContext::hand_off_df`]); the engines wait for that
//!   thread ([`DeviceSession::df`]). Kept only for SCFs converged for a gradient or a Hessian,
//!   and released once the derivative is done ([`DeviceSession::release_df`]).
//! - **The PCM's device term** of the SCF (`crate::backend::gpu::pcm::GpuPcmScf`: its surface
//!   operator and integrals on the device), reused by the PCM gradient of the same SCF.
//!
//! Only state that is a pure function of the basis sets, the geometry and the thresholds is
//! handed over (an engine set up from it is the same, bit for bit, as one that builds its own);
//! the SCF's fitted tensor and the XC screening are not (the gradient's arithmetic and its
//! second-order screening differ from the SCF's). The SCF's CPU XC integrator (its molecular
//! grid) is the geometry's integral context's ([`crate::context::IntegralContext`]), shared on
//! every backend.
//!
//! The session is emptied when the atoms move and when an SCF starts, which drops what the last
//! one left ([`DeviceSession::clear`], the provider's `clear_session`): a new SCF sets up its own,
//! and the old device memory goes before it does. The [`linus_gpu::df::DfShared`] also checks itself against
//! the basis sets an engine is given (`fits`).

use crate::backend::gpu::pcm::GpuPcmScf;
use crate::driver::backend::Family;
use crate::driver::{CalcContext, Purpose};
use linus_gpu::df::DfShared;
use linus_gpu::eri3c::Eri3cKind;
use linus_ints::BasisSet;
use std::sync::{Arc, Mutex, MutexGuard};

/// The SCF's density-fitting data and the thread packing their derivative records.
#[derive(Default)]
struct DfHandoff {
    shared: Option<Arc<DfShared>>,
    records: Option<std::thread::JoinHandle<()>>,
}

/// The device session of one geometry (see the module docs).
#[derive(Default)]
pub(crate) struct DeviceSession {
    df: Mutex<DfHandoff>,
    pcm: Mutex<Option<Arc<Mutex<GpuPcmScf>>>>,
}

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

impl DeviceSession {
    /// What the session holds goes: the atoms moved, or a new SCF of this geometry sets up its
    /// own.
    pub fn clear(&self) {
        self.release_df();
        *lock(&self.pcm) = None;
    }

    /// The SCF's density-fitting data, if it left them, once the derivative records being
    /// packed for them are done (the engines then take those).
    pub fn df(&self) -> Option<Arc<DfShared>> {
        let records = lock(&self.df).records.take();
        if let Some(h) = records {
            // a panic there leaves the records unbuilt: the engine packs its own
            let _ = h.join();
        }
        lock(&self.df).shared.clone()
    }

    /// The derivatives are done with the SCF's density-fitting data: they go (the records
    /// thread finishes on its own).
    pub fn release_df(&self) {
        *lock(&self.df) = DfHandoff::default();
    }

    /// Keeps the SCF's PCM term on the device for the PCM gradient of the same SCF.
    pub fn keep_pcm(&self, pcm: Arc<Mutex<GpuPcmScf>>) {
        *lock(&self.pcm) = Some(pcm);
    }

    /// The SCF's PCM term on the device, if it kept one.
    pub fn pcm(&self) -> Option<Arc<Mutex<GpuPcmScf>>> {
        lock(&self.pcm).clone()
    }
}

impl CalcContext {
    /// Whether an SCF converged for `purpose` keeps its density-fitting hand-off: for a gradient
    /// or a Hessian.
    pub(crate) fn keeps_handoff(purpose: Purpose) -> bool {
        purpose != Purpose::Energy
    }

    /// Keeps the density-fitting data `df` (of the auxiliary basis set `aux`) of an SCF converged
    /// for `purpose` for the derivatives ([`Self::keeps_handoff`]), and starts packing the
    /// derivative records their device engines will take: first derivatives for a gradient (the
    /// two-electron gradient on the device), second derivatives too for a Hessian (its
    /// two-electron terms on the device), on a thread of their own with a pool of the
    /// session's size, while the SCF
    /// runs (the records are a pure function of the basis sets: the same bits wherever packed),
    /// then copied, with the integral records and the metric, into page-locked memory for the
    /// engines' uploads ([`DfShared::stage`]).
    pub(crate) fn hand_off_df(&self, purpose: Purpose, df: Arc<DfShared>, aux: &BasisSet) {
        if !Self::keeps_handoff(purpose) {
            return;
        }
        let mut kinds = vec![];
        if self.device_for(Family::GradientJk).is_some() {
            kinds.push(Eri3cKind::Derivatives);
        }
        if purpose == Purpose::Hessian && self.device_for(Family::HessianJk).is_some() {
            kinds.push(Eri3cKind::Hessian);
            if !kinds.contains(&Eri3cKind::Derivatives) {
                kinds.insert(0, Eri3cKind::Derivatives);
            }
        }
        // the engines' primitive threshold of the derivative records (the gradient's and the
        // Hessian's: one definition)
        let thresh = linus_gpu::cpu::DfThresholds::default().deriv_prim;
        let gpu = self
            .device_for(Family::GradientJk)
            .or_else(|| self.device_for(Family::HessianJk))
            .cloned();
        let records = (!kinds.is_empty())
            .then(|| {
                let (shared, basis, aux) = (df.clone(), self.basis.clone(), aux.clone());
                let threads = self.pool.current_num_threads();
                std::thread::Builder::new()
                    .name("linus-df-records".into())
                    .spawn(move || {
                        let Ok(pool) = rayon::ThreadPoolBuilder::new().num_threads(threads).build()
                        else {
                            return;
                        };
                        pool.install(|| match gpu {
                            // the records, then page-locked copies of them and of the
                            // integrals and the metric for the engines' uploads; an error here
                            // (out of page-locked memory) leaves the plain uploads
                            Some(g) => {
                                let _ = shared.stage(&g, &basis, &aux, &kinds, thresh);
                            }
                            None => {
                                for k in kinds {
                                    // an error here is the engine's too: it reports it there
                                    let _ = shared.derivatives(&basis, &aux, k, thresh);
                                }
                            }
                        });
                    })
                    .ok()
            })
            .flatten();
        if let Some(s) = self.device_session() {
            *lock(&s.df) = DfHandoff {
                shared: Some(df),
                records,
            };
        }
    }
}
