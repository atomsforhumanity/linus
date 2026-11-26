//! The SCF's density-fitted J on the memory-lean device engine ([`linus_gpu::jdirect`]: no
//! stored three-index tensor), for SCFs whose DF tensor would not stay on the device: the J of
//! df's pure functionals ([`j_direct_for`]) and rij_cgk's J (`crate::backend::gpu::gridk`).
//! [`GpuJDirectJ`] is its builder on the host loop; the device loop takes its engine
//! ([`crate::backend::gpu::device_jk`]).

use crate::backend::gpu::device_jk::{mismatch, DeviceJk, DeviceScfJk};
use crate::backend::gpu::df::{device_policy, df_shared, scf_df_budget, scf_df_plan};
use crate::backend::gpu::{device_error, setup};
use crate::backend::scf::ScfJk;
use crate::backend::Setup;
use crate::driver::backend::Family;
use crate::driver::options::DfJEngine;
use crate::driver::record::{Decision, DfJChoice};
use crate::driver::CalcContext;
use faer::{Mat, Par};
use linus_base::error::LinusError;
use linus_base::linalg::mat_to_colmajor;
use linus_base::terms::{JkBuilder, JkOutput, JkRequest};
use linus_core::timing::Timers;
use linus_gpu::df::DfShared;
use linus_gpu::engine::Phases;
use linus_gpu::jdirect::{GpuDfJDirect, JDirectEngine};
use linus_gpu::scf::DeviceJ;
use linus_gpu::{Gpu, GpuError};
use linus_ints::BasisSet;
use std::sync::Arc;

/// The workspace of the eri3c engine's batches of integrals (the engine's default option:
/// one batch of whole shell pairs' rows on the device).
const JDIRECT_WORKSPACE: usize = 2 << 30;

/// The memory-lean J's builder on the host loop.
pub(crate) struct GpuJDirectJ {
    gpu: Arc<Gpu>,
    /// The engine (None while the device loop holds it).
    eng: Option<GpuDfJDirect>,
}

impl GpuJDirectJ {
    /// The engine for `aux` on `shared`'s layout, its kernels the device policy's for the card
    /// (the Hermite J-engine on FP64-rich cards where it covers the basis, else recomputed
    /// three-center integrals), with the kernels and why; its setup's phases timed as
    /// `jk_setup_jdirect_*` (the layout's as `jk_setup_df_*`), the records' size as the resource
    /// `jk_setup_jdirect_records_gib`.
    fn new(
        calc: &CalcContext,
        gpu: Arc<Gpu>,
        aux: &BasisSet,
        shared: Arc<DfShared>,
        timers: &mut Timers,
    ) -> Result<Result<(Self, &'static str, String), GpuError>, LinusError> {
        let (engine, why) = if device_policy(calc)?.j_direct_hermite {
            (
                JDirectEngine::Hermite,
                "the Hermite J-engine on an FP64-rich card".to_string(),
            )
        } else {
            (
                JDirectEngine::Eri3c,
                "recomputed three-center integrals on an FP64-poor card (the Hermite J-engine is \
                 slower there)"
                    .to_string(),
            )
        };
        let layout = shared.build_seconds;
        let eng = match GpuDfJDirect::with_engine(
            gpu.clone(),
            &calc.basis,
            aux,
            Some(shared),
            JDIRECT_WORKSPACE,
            engine,
        ) {
            Ok(e) => e,
            Err(e) => return Ok(Err(e)),
        };
        timers.add("jk_setup_df_layout", layout.layout);
        timers.add("jk_setup_df_pack", layout.pack);
        let st = eng.setup_times();
        for (name, s) in st.phases() {
            timers.add(&format!("jk_setup_jdirect_{name}"), s);
        }
        timers.resource(
            "jk_setup_jdirect_records_gib",
            st.records_bytes as f64 / 1073741824.0,
        );
        let kernels = eng.engine();
        let why = if engine == JDirectEngine::Hermite && kernels == "eri3c" {
            format!(
                "{why}, which does not cover the basis (orbital l > 3 or auxiliary l > 4): \
                 recomputed three-center integrals"
            )
        } else {
            why
        };
        Ok(Ok((
            GpuJDirectJ {
                gpu,
                eng: Some(eng),
            },
            kernels,
            why,
        )))
    }

    fn eng(&mut self) -> Result<&mut GpuDfJDirect, LinusError> {
        self.eng.as_mut().ok_or_else(|| {
            LinusError::Internal("the SCF's memory-lean J while the device loop holds it".into())
        })
    }

    /// The engine's build phases as the SCF's timers `jk_jdirect_*`.
    pub(crate) fn add_timers(&self, timers: &mut Timers) {
        if let Some(e) = &self.eng {
            for (name, s) in e.times().phases() {
                timers.add(&format!("jk_jdirect_{name}"), s);
            }
        }
    }

    /// The engine for the device loop (None once taken).
    pub(crate) fn take_engine(&mut self) -> Option<(Arc<Gpu>, GpuDfJDirect)> {
        Some((self.gpu.clone(), self.eng.take()?))
    }

    /// The builder of an engine the device loop hands back.
    pub(crate) fn from_engine(gpu: Arc<Gpu>, eng: GpuDfJDirect) -> Self {
        GpuJDirectJ {
            gpu,
            eng: Some(eng),
        }
    }
}

impl JkBuilder for GpuJDirectJ {
    fn name(&self) -> String {
        "J: device DF, integral-direct (no stored tensor)".into()
    }

    fn compute(&mut self, req: &JkRequest, _par: Par) -> Result<JkOutput, LinusError> {
        let mut out = JkOutput::default();
        if !req.want_j || req.dens.is_empty() {
            return Ok(out);
        }
        let n = req.dens[0].nrows();
        let nd = req.dens.len();
        let mut flat = Vec::with_capacity(nd * n * n);
        for d in req.dens {
            flat.extend(mat_to_colmajor(d.as_ref()));
        }
        let j = self
            .eng()?
            .coulomb_host(&flat, nd)
            .map_err(|e| device_error(Family::Jk, e))?;
        out.j = (0..nd)
            .map(|s| Mat::from_fn(n, n, |a, b| j[s * n * n + a + b * n]))
            .collect();
        Ok(out)
    }
}

impl ScfJk for GpuJDirectJ {
    fn builder(&mut self) -> &mut dyn JkBuilder {
        self
    }

    fn into_builder(self: Box<Self>) -> Box<dyn JkBuilder> {
        self
    }

    fn timers(&self) -> Option<Timers> {
        let mut t = Timers::default();
        self.add_timers(&mut t);
        Some(t)
    }
}

impl DeviceScfJk for GpuJDirectJ {
    fn take_device(&mut self, _par: Par) -> Result<Option<DeviceJk>, LinusError> {
        Ok(self.take_engine().map(|(gpu, e)| DeviceJk {
            gpu,
            j: DeviceJ::Direct(Box::new(e)),
            aux: None,
            gridk: None,
        }))
    }

    fn put_device(&mut self, jk: DeviceJk) -> Result<(), LinusError> {
        match jk {
            DeviceJk {
                j: DeviceJ::Direct(e),
                gridk: None,
                ..
            } => {
                self.eng = Some(*e);
                Ok(())
            }
            _ => Err(mismatch("engines other than the memory-lean J")),
        }
    }

    fn coulomb_fit(&self) -> bool {
        false
    }
}

/// The engine of an SCF's density-fitted J on the device, as decided ([`decide`]).
pub(crate) enum DfJ {
    /// The memory-lean J, set up.
    Direct(Box<GpuJDirectJ>),
    /// The DF engine's tensor, on the layout the decision built (None: not built).
    Stored(Option<Arc<DfShared>>),
}

/// `backend.df_j`'s engine for an SCF whose J needs no tensor for anything else, on `shared`'s
/// layout, and why: `direct` or `stored` where set; under `auto`, `direct` where the DF
/// engine's storage plan (the rule its setup applies, `linus_gpu::df::plan`, within the memory
/// plan's budget) has no device rung for the tensor, with the plan's refusal as the reason. The
/// DF engine would then stream the tensor from host memory every build or leave J to the CPU,
/// both slower than recomputing the integrals; where the tensor stays on the device, its two
/// passes over device memory are far faster.
fn choose(calc: &CalcContext, shared: &DfShared) -> Result<(&'static str, String), LinusError> {
    Ok(match calc.plan.backend().df_j() {
        DfJEngine::Direct => ("direct", "backend.df_j direct".into()),
        DfJEngine::Stored => ("stored", "backend.df_j stored".into()),
        DfJEngine::Auto => {
            let sizes = linus_gpu::df::plan::DfSizes {
                nbf: calc.basis.nbf,
                naux: shared.layout.naux,
                rows: shared.layout.rows(&calc.basis),
            };
            // the device alone: a streamed tensor is the case the memory-lean J replaces
            match scf_df_plan(calc, sizes, calc.method.omega(), 0)? {
                Ok(p) => (
                    "stored",
                    format!(
                        "the DF engine's plan keeps the tensor on the device ({:.1} GiB with a \
                         build's workspace, within its {:.1} GiB)",
                        p.device_bytes as f64 / 1073741824.0,
                        scf_df_budget(calc)? as f64 / 1073741824.0
                    ),
                ),
                Err(refusal) => (
                    "direct",
                    format!(
                        "the DF engine's plan has no device rung for the tensor: {}",
                        refusal.reason
                    ),
                ),
            }
        }
    })
}

/// Records the engine of the SCF's density-fitted J.
fn record(calc: &CalcContext, choice: DfJChoice) {
    calc.record.decide(Decision::ScfDfJ(choice));
}

/// The density-fitted J of an SCF on `shared`'s layout for `aux` whose J needs no tensor for
/// anything else ([`choose`]): the memory-lean J where chosen, set up (timed as `jk_setup`, its
/// parts as `jk_setup_jdirect_*`); the DF engine's otherwise, and where the memory-lean J's
/// setup does not fit the device; the decision recorded. An implicit solvent keeps the DF
/// engine (its Coulomb term takes the DF fit, which the memory-lean J does not hand over).
pub(crate) fn decide(
    calc: &CalcContext,
    gpu: &Arc<Gpu>,
    aux: &BasisSet,
    shared: Arc<DfShared>,
    timers: &mut Timers,
) -> Result<DfJ, LinusError> {
    let stored = |reason: String, declined: Option<String>| DfJChoice {
        engine: "stored",
        reason,
        kernels: None,
        declined,
    };
    if calc.plan.solvent().is_some() {
        record(
            calc,
            stored(
                "the implicit solvent's Coulomb term takes the DF engine's fit".into(),
                None,
            ),
        );
        return Ok(DfJ::Stored(Some(shared)));
    }
    let (engine, reason) = choose(calc, &shared)?;
    if engine == "stored" {
        record(calc, stored(reason, None));
        return Ok(DfJ::Stored(Some(shared)));
    }
    let t = std::time::Instant::now();
    let mut parts = Timers::default();
    let built = GpuJDirectJ::new(calc, gpu.clone(), aux, shared.clone(), &mut parts)?;
    timers.add("jk_setup", t.elapsed().as_secs_f64());
    timers.merge(&parts);
    Ok(match setup(Family::Jk, built)? {
        Setup::Ready((j, kernels, kernels_reason)) => {
            record(
                calc,
                DfJChoice {
                    engine: "direct",
                    reason,
                    kernels: Some((kernels, kernels_reason)),
                    declined: None,
                },
            );
            DfJ::Direct(Box::new(j))
        }
        Setup::Declined(why) => {
            record(calc, stored(reason, Some(why)));
            DfJ::Stored(Some(shared))
        }
    })
}

/// The density-fitted J of an SCF whose J/K is the DF engine's (jk df, or rij_cgk's J alone
/// without exact exchange; [`decide`]): the DF engine's for exact exchange (K needs the
/// tensor) and an implicit solvent, and under `auto` where the DF engine's storage plan keeps
/// even the tensor's upper bound (every shell pair kept: nbf (nbf + 1) / 2 rows) on the device
/// (decided before any packing); else on the layout built here (handed to the DF engine when
/// it takes the J, else kept for the derivatives).
pub(crate) fn df_j_for(
    calc: &CalcContext,
    gpu: &Arc<Gpu>,
    aux: &BasisSet,
    purpose: crate::driver::Purpose,
    timers: &mut Timers,
) -> Result<DfJ, LinusError> {
    let stored = |reason: &str| {
        record(
            calc,
            DfJChoice {
                engine: "stored",
                reason: reason.into(),
                kernels: None,
                declined: None,
            },
        );
        Ok(DfJ::Stored(None))
    };
    if calc.method.exchange.is_some() {
        return stored("the DF engine's exchange needs the stored tensor");
    }
    if calc.plan.solvent().is_some() {
        return stored("the implicit solvent's Coulomb term takes the DF engine's fit");
    }
    if calc.plan.backend().df_j() == DfJEngine::Auto {
        let n = calc.basis.nbf;
        let dmax = calc
            .basis
            .shells
            .iter()
            .map(|s| s.nfunc())
            .max()
            .unwrap_or(1);
        let upper = linus_gpu::df::plan::DfSizes {
            nbf: n,
            naux: aux.nbf,
            rows: linus_ints::eri3c::DfRows {
                nx: n * (n + 1) / 2,
                np: n * n,
                ncg: n * n,
                max_panel_rows: dmax * n,
                max_pair_rows: dmax * dmax,
            },
        };
        if let Ok(p) = scf_df_plan(calc, upper, calc.method.omega(), 0)? {
            return stored(&format!(
                "the DF engine's plan keeps even the tensor's upper bound on the device ({:.1} \
                 GiB with a build's workspace, within its {:.1} GiB)",
                p.device_bytes as f64 / 1073741824.0,
                scf_df_budget(calc)? as f64 / 1073741824.0
            ));
        }
    }
    let shared = match setup(Family::Jk, timers.time("jk_setup", || df_shared(calc, aux)))? {
        Setup::Ready(s) => s,
        Setup::Declined(_) => return Ok(DfJ::Stored(None)),
    };
    let j = decide(calc, gpu, aux, shared.clone(), timers)?;
    if let DfJ::Direct(_) = &j {
        // the layout, packed integrals and metric, kept for the derivatives (as the DF engine's)
        calc.hand_off_df(purpose, shared, aux);
    }
    Ok(j)
}
