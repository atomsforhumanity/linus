//! The SCF's J/K of `jk: "rij_cgk"` on the device: density-fitted J on the device DF engine
//! (its tensor, no exchange tensor of the operator), or on the memory-lean J
//! (`crate::backend::gpu::jdirect`) where that tensor would not stay on the device, and
//! compensated grid-K on linus-gpu's grid-K engine ([`linus_gpu::gridk::GpuGridK`]: the
//! collocation and the symmetric overlap fit set up on the device, every build there). The
//! exact part of the dSCF partition (the SAD reference D0 and K\[D0\], linus-jk's
//! [`GridKReference`]) is the CPU's, once per geometry and operator. The host loop calls
//! [`GpuCgkJk`] as a [`JkBuilder`]; the device loop takes its engines
//! (`crate::backend::gpu::device_jk`) and gives them back.

use crate::backend::gpu::device_jk::{mismatch, DeviceJk, DeviceScfJk};
use crate::backend::gpu::df::{df_shared, scf_jk_setup, GpuDfJkBuilder};
use crate::backend::gpu::jdirect::{decide, df_j_for, DfJ, GpuJDirectJ};
use crate::backend::gpu::{device_error, setup};
use crate::backend::scf::{ScfJk, ScfJkBuilt};
use crate::backend::Setup;
use crate::driver::backend::Family;
use crate::driver::ground::ScfSetup;
use crate::driver::options::GridChoice;
use crate::driver::CalcContext;
use faer::{Mat, Par};
use linus_base::error::LinusError;
use linus_base::linalg::{ao_view_mut, dot, mat_to_colmajor};
use linus_base::terms::{JkBuilder, JkOutput, JkRequest};
use linus_core::timing::Timers;
use linus_gpu::engine::Phases;
use linus_gpu::gridk::{GpuGridK, GpuGridKOptions, GridKTerm};
use linus_gpu::scf::DeviceJ;
use linus_gpu::{Gpu, GpuError};
use linus_ints::operator::Operator;
use linus_ints::{BasisSet, DerivOrder, OneEPlan};
use linus_jk::df::ResolvedDf;
use linus_jk::gridk::{GridKConfig, GridKReference};
use std::sync::Arc;
use std::time::Instant;

/// Compensated grid-K's builds on the device, with the exact part of the dSCF partition (D0
/// and K\[D0\], linus-jk's [`GridKReference`]) on the CPU.
pub(crate) struct GpuGridKBuilder {
    /// The engine (None while the device loop holds it).
    eng: Option<GpuGridK>,
    basis: BasisSet,
    reference: GridKReference,
    op: Operator,
}

impl GpuGridKBuilder {
    /// The builder on the grid of `k_grid` with the SAD reference `d0` (one spin channel).
    pub(crate) fn new(
        cx: &CalcContext,
        gpu: Arc<Gpu>,
        k_grid: &GridChoice,
        d0: Mat<f64>,
    ) -> Result<Result<Self, GpuError>, LinusError> {
        let basis = &cx.basis;
        let grid = linus_grid::MolecularGrid::build(&cx.mol.z, &cx.mol.positions, &k_grid.spec())
            .map_err(|e| LinusError::Input(format!("jk.k_grid: {e}")))?;
        let n = basis.nbf;
        let mut s = Mat::<f64>::zeros(n, n);
        OneEPlan::new(basis, DerivOrder::Zero).overlap(&mut ao_view_mut(s.as_mut()));
        let cfg = GridKConfig::default();
        let eng = GpuGridK::fitted(
            gpu,
            basis,
            &grid,
            &mat_to_colmajor(s.as_ref()),
            GpuGridKOptions {
                lindep: cx.plan.scf_algorithm().lindep,
                fit: cfg.fit,
                thresh: cfg.thresh,
                prim_cut: cfg.prim_cut,
                style: None,
            },
        );
        // a grid too coarse for the overlap fit is the calculation's input error, as the CPU's
        if let Err(GpuError::Input(m)) = eng {
            return Err(LinusError::Input(m));
        }
        Ok(eng.map(|eng| GpuGridKBuilder {
            eng: Some(eng),
            basis: basis.clone(),
            reference: GridKReference::new(d0, cfg.exact_thresh),
            op: Operator::Coulomb,
        }))
    }

    /// K\[D0\] of the operator `op`, computed elsewhere (a host thread while the device set up).
    fn set_reference(
        &mut self,
        op: Operator,
        k0: Mat<f64>,
        seconds: f64,
    ) -> Result<(), LinusError> {
        self.eng()?.set_operator(&op);
        self.op = op;
        self.reference.set_k0(op, k0, seconds);
        Ok(())
    }

    fn eng(&mut self) -> Result<&mut GpuGridK, LinusError> {
        self.eng.as_mut().ok_or_else(|| {
            LinusError::Internal("the grid-K engine while the device loop holds it".into())
        })
    }

    /// The engine as the device loop's term (K\[D0\] and D0 on the device).
    fn take_term(&mut self, gpu: &Gpu, par: Par) -> Result<GridKTerm, LinusError> {
        let k0 = self.reference.k0(&self.basis, self.op, par)?.clone();
        let e00 = 0.5 * dot(self.reference.d0().as_ref(), k0.as_ref());
        let k0 = mat_to_colmajor(k0.as_ref());
        let d0 = mat_to_colmajor(self.reference.d0().as_ref());
        let eng = self
            .eng
            .take()
            .ok_or_else(|| LinusError::Internal("the grid-K engine taken twice".into()))?;
        GridKTerm::new(gpu, eng, &d0, &k0, e00).map_err(|e| device_error(Family::Jk, e))
    }

    /// The engine back from the device loop's term.
    fn put_term(&mut self, t: GridKTerm) {
        self.eng = Some(t.into_engine());
    }
}

impl JkBuilder for GpuGridKBuilder {
    fn name(&self) -> String {
        "compensated grid-K (device)".into()
    }

    fn compute(&mut self, req: &JkRequest, par: Par) -> Result<JkOutput, LinusError> {
        if !req.want_k {
            return Ok(JkOutput::default());
        }
        let GpuGridKBuilder {
            eng,
            basis,
            reference,
            op,
        } = self;
        let eng = eng.as_mut().ok_or_else(|| {
            LinusError::Internal("the grid-K engine while the device loop holds it".into())
        })?;
        reference.k0(basis, *op, par)?;
        let (Some(k0), d0) = (reference.current_k0(), reference.d0()) else {
            return Err(LinusError::Internal(
                "grid-K: K[D0] before the build".into(),
            ));
        };
        let n = k0.nrows();
        let mut out = JkOutput {
            k_energy: Some(Vec::new()),
            ..Default::default()
        };
        for d in req.dens {
            let dd = d - d0;
            let kb = eng
                .build(&mat_to_colmajor(dd.as_ref()))
                .map_err(|e| device_error(Family::Jk, e))?;
            let kbar = Mat::from_fn(n, n, |i, j| kb[i + j * n]);
            let (k, e_exact) = reference.compensate(k0, &dd, &kbar);
            out.k.push(k);
            if let Some(v) = out.k_energy.as_mut() {
                v.push(e_exact + 0.5 * dot(dd.as_ref(), kbar.as_ref()));
            }
        }
        Ok(out)
    }

    /// K\[D0\] follows the operator (computed on its first use for it).
    fn set_exchange_operator(&mut self, op: Operator) -> bool {
        match self.eng() {
            Ok(e) => e.set_operator(&op),
            Err(_) => return false,
        }
        self.op = op;
        true
    }
}

/// rij_cgk's J: the DF engine, or the memory-lean one where the DF tensor would not stay on the
/// device.
pub(crate) enum ScfJ {
    Df(Box<GpuDfJkBuilder>),
    Direct(Box<GpuJDirectJ>),
}

/// The SCF's device J/K of rij_cgk (module docs): J from the DF engine or the memory-lean one,
/// K from grid-K.
pub(crate) struct GpuCgkJk {
    /// The J engine (None while the device loop holds it).
    j: Option<ScfJ>,
    k: GpuGridKBuilder,
}

impl GpuCgkJk {
    fn j(&mut self) -> Result<&mut dyn JkBuilder, LinusError> {
        match self.j.as_mut() {
            Some(ScfJ::Df(j)) => Ok(&mut **j),
            Some(ScfJ::Direct(j)) => Ok(&mut **j),
            None => Err(LinusError::Internal(
                "the SCF's device J while the device loop holds it".into(),
            )),
        }
    }
}

impl DeviceScfJk for GpuCgkJk {
    fn take_device(&mut self, par: Par) -> Result<Option<DeviceJk>, LinusError> {
        let Some(j) = self.j.take() else {
            return Ok(None);
        };
        let (gpu, j, aux) = match j {
            ScfJ::Df(j) => {
                let (gpu, jk, aux) = (*j).into_parts();
                (gpu, DeviceJ::Df(Box::new(jk)), Some(aux))
            }
            ScfJ::Direct(mut j) => match j.take_engine() {
                Some((gpu, e)) => (gpu, DeviceJ::Direct(Box::new(e)), None),
                None => return Ok(None),
            },
        };
        let gridk = Some(self.k.take_term(&gpu, par)?);
        Ok(Some(DeviceJk { gpu, j, aux, gridk }))
    }

    fn put_device(&mut self, jk: DeviceJk) -> Result<(), LinusError> {
        let DeviceJk { gpu, j, aux, gridk } = jk;
        let Some(k) = gridk else {
            return Err(mismatch("engines without grid-K"));
        };
        self.k.put_term(k);
        self.j = Some(match (j, aux) {
            (DeviceJ::Df(eng), Some(aux)) => {
                ScfJ::Df(Box::new(GpuDfJkBuilder::from_parts(gpu, *eng, aux)))
            }
            (DeviceJ::Direct(e), _) => ScfJ::Direct(Box::new(GpuJDirectJ::from_engine(gpu, *e))),
            (DeviceJ::Df(_), None) => return Err(mismatch("the DF engine without its basis")),
        });
        Ok(())
    }

    fn coulomb_fit(&self) -> bool {
        matches!(self.j, Some(ScfJ::Df(_)))
    }
}

impl JkBuilder for GpuCgkJk {
    fn name(&self) -> String {
        match &self.j {
            Some(ScfJ::Direct(_)) => {
                "J: device DF, integral-direct + K: compensated grid-K (device)"
            }
            _ => "J: device DF + K: compensated grid-K (device)",
        }
        .into()
    }

    fn compute(&mut self, req: &JkRequest, par: Par) -> Result<JkOutput, LinusError> {
        let mut out = JkOutput::default();
        if req.want_j {
            let r = JkRequest {
                dens: req.dens,
                factors: req.factors,
                want_j: true,
                want_k: false,
            };
            out.j = self.j()?.compute(&r, par)?.j;
        }
        if req.want_k {
            let r = JkRequest {
                dens: req.dens,
                factors: req.factors,
                want_j: false,
                want_k: true,
            };
            let k = self.k.compute(&r, par)?;
            (out.k, out.k_energy) = (k.k, k.k_energy);
        }
        Ok(out)
    }

    fn set_exchange_operator(&mut self, op: Operator) -> bool {
        self.k.set_exchange_operator(op)
    }

    /// HOOK (solvent): the density-fitted J takes it.
    fn set_coulomb_hook(
        &mut self,
        hook: Option<Box<dyn linus_base::terms::CoulombHook>>,
    ) -> Result<linus_base::terms::HookOutcome, LinusError> {
        self.j()?.set_coulomb_hook(hook)
    }
}

impl ScfJk for GpuCgkJk {
    fn builder(&mut self) -> &mut dyn JkBuilder {
        self
    }

    fn into_builder(self: Box<Self>) -> Box<dyn JkBuilder> {
        self
    }

    /// The grid-K engine's build phases as `jk_gridk_*`, the memory-lean J's as `jk_jdirect_*`.
    fn timers(&self) -> Option<Timers> {
        let mut t = Timers::default();
        if let Some(e) = &self.k.eng {
            for (name, s) in e.times().phases() {
                t.add(&format!("jk_gridk_{name}"), s);
            }
        }
        if let Some(ScfJ::Direct(j)) = &self.j {
            j.add_timers(&mut t);
        }
        Some(t)
    }
}

/// The SCF's J/K of rij_cgk on the device for the auxiliary basis `aux` and the exchange grid
/// `k_grid`, set up with `scf` (timed as `jk_setup`, the grid-K engine's part as
/// `jk_setup_gridk`); declined (the CPU's) when a setup does not fit. K\[D0\] (exact, the
/// host's long part, `jk_setup_k0`) runs beside the device's setup on the pool the setup runs
/// on, the J/K setup's share of the session's threads; `jk_setup_k0_wait` is how long the setup
/// waited for it.
pub(crate) fn scf_jk(
    calc: &CalcContext,
    gpu: &Arc<Gpu>,
    scf: &ScfSetup,
    aux: &BasisSet,
    k_grid: &GridChoice,
    timers: &mut Timers,
) -> Result<Setup<ScfJkBuilt>, LinusError> {
    // a pure functional's SCF never asks for K: rij_cgk's J alone, as df's (no grid-K engine,
    // no K[D0])
    if calc.method.exchange.is_none() {
        return match df_j_for(calc, gpu, aux, scf.purpose, timers)? {
            DfJ::Direct(j) => Ok(Setup::Ready((j, None))),
            DfJ::Stored(shared) => {
                crate::backend::gpu::df::scf_jk(calc, gpu, scf, aux, shared, timers)
            }
        };
    }
    // K[D0] of the method's operator (a global hybrid's exchange is of 1/r, a_x applied by the
    // SCF; a range-separated hybrid's operator carries its coefficients)
    let op = linus_base::terms::exchange_parts(calc.method.exchange)
        .1
        .unwrap_or(Operator::Coulomb);
    let d0 = calc.sad_reference();
    let thresh = GridKConfig::default().exact_thresh;
    let t0 = Instant::now();
    let mut setup_timers = Timers::default();
    let ((built, t_setup), k0) = rayon::join(
        || {
            let built = device_setup(calc, gpu, scf, aux, k_grid, d0.clone(), &mut setup_timers);
            (built, t0.elapsed())
        },
        || {
            let t = Instant::now();
            let par = Par::rayon(rayon::current_num_threads());
            GridKReference::exact_exchange(&calc.basis, &d0, op, thresh, par)
                .map(|k| (k, t.elapsed().as_secs_f64()))
        },
    );
    timers.merge(&setup_timers);
    let (k0, t_k0) = k0?;
    timers.add("jk_setup_k0", t_k0);
    timers.add("jk_setup_k0_wait", (t0.elapsed() - t_setup).as_secs_f64());
    let (j, resolved, mut k) = match built? {
        Setup::Ready(x) => x,
        Setup::Declined(why) => return Ok(Setup::Declined(why)),
    };
    k.set_reference(op, k0, t_k0)?;
    Ok(Setup::Ready((
        Box::new(GpuCgkJk { j: Some(j), k }),
        resolved,
    )))
}

/// The device part of [`scf_jk`]: the grid-K engine, then J on the DF engine when its tensor
/// fits the device beside it, else on the memory-lean J (recorded).
#[allow(clippy::type_complexity)] // the setup's three results, named at the one caller
fn device_setup(
    calc: &CalcContext,
    gpu: &Arc<Gpu>,
    scf: &ScfSetup,
    aux: &BasisSet,
    k_grid: &GridChoice,
    d0: Mat<f64>,
    timers: &mut Timers,
) -> Result<Setup<(ScfJ, Option<ResolvedDf>, GpuGridKBuilder)>, LinusError> {
    let built = timers.time("jk_setup_gridk", || {
        GpuGridKBuilder::new(calc, gpu.clone(), k_grid, d0)
    })?;
    let k = match setup(Family::Jk, built)? {
        Setup::Ready(k) => k,
        Setup::Declined(why) => return Ok(Setup::Declined(why)),
    };
    // the engine's device memory (its grant, the memory plan's size of it) and its draws beyond
    if let Some(e) = &k.eng {
        timers.resource(
            "jk_setup_gridk_device_gib",
            e.granted_bytes() as f64 / 1073741824.0,
        );
        calc.note_grant(Family::Jk, &e.grant_violations());
    }
    let shared = match setup(Family::Jk, timers.time("jk_setup", || df_shared(calc, aux)))? {
        Setup::Ready(s) => s,
        Setup::Declined(why) => return Ok(Setup::Declined(why)),
    };
    let (j, resolved) = match decide(calc, gpu, aux, shared, timers)? {
        DfJ::Direct(j) => (ScfJ::Direct(j), None),
        DfJ::Stored(shared) => {
            let opts = scf_jk_setup(calc, scf, None)?;
            let built = timers.time("jk_setup", || {
                GpuDfJkBuilder::new(gpu.clone(), &calc.basis, aux, opts, shared.as_deref())
            });
            match setup(Family::Jk, built)? {
                Setup::Ready((j, _)) => {
                    let r = j.resolved();
                    (ScfJ::Df(Box::new(j)), Some(r))
                }
                Setup::Declined(why) => return Ok(Setup::Declined(why)),
            }
        }
    };
    Ok(Setup::Ready((j, resolved, k)))
}
