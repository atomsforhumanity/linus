//! Density-fitted J/K on the GPU (feature `gpu`): linus's builder traits over linus-gpu's
//! engines, for the SCF ([`GpuDfJkBuilder`], a [`JkBuilder`] over `linus_gpu::df::GpuDfJk`)
//! and for CPHF/CPKS responses ([`GpuResponseJk`], a [`ResponseJk`] over
//! `linus_gpu::response::df::GpuDfResponse`), the Hessian's explicit terms ([`GpuHessianJk`]),
//! the two-electron gradient (`linus_gpu::grad::df::GpuDfGrad`, [`gradient`]), and their setups
//! as the GPU's provider asks for them (`crate::backend`: [`scf_jk`], [`response_setup`]).
//!
//! The engines store the CPU builders' screened tensor, row for row, fitted on the device, and
//! give the CPU's J and K to round-off (matrices ~1e-14 relative, energies ~1e-12 Eh);
//! their results are bitwise reproducible run to run on one device and independent of the host
//! thread count, not bitwise equal to the CPU's. What they do not cover resolves to the CPU
//! before the SCF (`crate::driver::backend`): the density-fitted PCM's Coulomb hook
//! (`solvent.scf: "df"`), exact and seminumerical exchange. The `predicted` PCM mode's hook gets the fit coefficients
//! of every J build ([`CoulombFit`]).
//!
//! The device policy's arithmetic becomes the engines' settings here: an FP64 exchange runs on
//! the first tier of its fallback chain that sets up ([`apply_chain`]), and a response J/K's
//! arithmetic (the plan's [`ResponseArith`]) its recipe ([`ResponseRecipe`]: the chain, the
//! stored operator's precision, D37's costing, the CPKS kernel's precision).
//!
//! Matrices cross the boundary column-major (`mat_to_colmajor`, `mat_from_colmajor`).

use crate::backend::gpu::costing::{auto_choice, AutoChoice, ExchangeCosting};
use crate::backend::gpu::device_error;
use crate::backend::gpu::setup;
use crate::backend::scf::{ScfJk, ScfJkBuilt, ScfRun};
use crate::backend::Setup;
use crate::derivatives::terms::{
    ExplicitHessian, GradientInputs, GradientTerm, HessianTerm, Lane, TermInputs,
};
use crate::driver::backend::{Family, ResponseRecords};
use crate::driver::ground::ScfSetup;
use crate::driver::options::Fp32Mode;
use crate::driver::policy::{
    DevicePolicy, DevicePrecision, ResponseArith, ResponseMode, CUBLAS, NATIVE,
};
use crate::driver::report::{EmulationUsed, FitEmulation};
use crate::driver::CalcContext;
use crate::two_electron::{ResponseRequest, ResponseSetup, ResponseUse};
use faer::{Mat, MatRef, Par};
use linus_base::error::LinusError;
use linus_base::linalg::{mat_from_colmajor as from_cm, mat_to_colmajor as to_cm};
use linus_base::terms::{JkBuilder, JkOutput, JkRequest, ResponseJk};
use linus_core::timing::Timers;
use linus_core::Arith;
use linus_gpu::df::{
    DfFit, DfIntegrals, DfShared, DfStorage, Factor, GpuDfJk, GpuDfJkOptions, GpuJkOutput,
    GpuJkRequest, KMath,
};
use linus_gpu::engine::Phases;
use linus_gpu::response::df::{GpuDfResponse, GpuDfResponseOptions, RespMath};
use linus_gpu::{Gpu, GpuError};
use linus_ints::operator::Operator;
use linus_ints::BasisSet;
use linus_jk::df::{factor_symmetric, DfLongRangeSource, DfPlan, ResolvedDf};
use std::sync::Arc;

/// The DF-J/K engine's exchange arithmetic of an [`Arith`].
fn kmath(a: Arith) -> KMath {
    match a {
        Arith::Ozaki { slices } => KMath::Ozaki {
            slices: slices as usize,
        },
        Arith::Fp64Emulated { bits } => KMath::Emulated { bits: bits as i32 },
        _ => KMath::Native,
    }
}

/// The response engine's exchange arithmetic of an [`Arith`].
fn resp_math(a: Arith) -> RespMath {
    match a {
        Arith::Ozaki { slices } => RespMath::Ozaki {
            slices: slices as usize,
        },
        Arith::Fp64Emulated { bits } => RespMath::Emulated { bits: bits as i32 },
        Arith::Fp32 => RespMath::Single,
        _ => RespMath::Native,
    }
}

/// The DF-J/K engine's fit arithmetic of an [`Arith`].
fn df_fit(a: Arith) -> DfFit {
    match a {
        Arith::Ozaki { slices } => DfFit::Ozaki {
            slices: slices as usize,
        },
        _ => DfFit::Native,
    }
}

/// The emulation of a response exchange on the stored operator: its GEMMs in cuBLAS's
/// emulation (the Ozaki scheme's GEMM1 planes would sit unused), else FP64.
pub(crate) const OPERATOR_EMULATION: &[Arith] = &[CUBLAS, Arith::Fp64];

/// The outcome of a fallback chain ([`apply_chain`]): the arithmetic that set up and why each
/// earlier one did not.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct ArithUsed {
    pub arith: Arith,
    /// "tier: error" for each tier that did not set up, in order.
    pub declined: Vec<String>,
}

impl ArithUsed {
    /// The declined tiers' reasons as one sentence (None: the first tier ran).
    pub(crate) fn reason(&self) -> Option<String> {
        (!self.declined.is_empty()).then(|| self.declined.join("; "))
    }
}

/// The name of a tier in a fallback reason.
fn tier_name(a: Arith) -> &'static str {
    match a {
        Arith::Ozaki { .. } => "Ozaki exchange",
        Arith::Fp64Emulated { .. } => "cuBLAS emulation",
        Arith::Fp64 => "FP64",
        Arith::Fp32 => "FP32",
        Arith::DoubleFloat => "double-float",
    }
}

/// Sets up the first arithmetic of `chain` that `set` accepts, each tier that fails recorded
/// with its error; the last is taken whatever `set` returns (native FP64 always sets up on a
/// built engine). The one ladder of the SCF's, the response's and the Hessian's exchange.
pub(crate) fn apply_chain<E: std::fmt::Display>(
    chain: &[Arith],
    mut set: impl FnMut(Arith) -> Result<(), E>,
) -> ArithUsed {
    let mut declined = Vec::new();
    for (i, &a) in chain.iter().enumerate() {
        match set(a) {
            Ok(()) => return ArithUsed { arith: a, declined },
            Err(_) if i + 1 == chain.len() => return ArithUsed { arith: a, declined },
            Err(e) => declined.push(format!("{}: {e}", tier_name(a))),
        }
    }
    ArithUsed {
        arith: Arith::Fp64,
        declined,
    }
}

/// The record of an exchange's fallback chain (`resolved.record.backend.emulation`): the arithmetic
/// that ran and its bits, why a lower tier than the first ran (else `note`), and whether the
/// tensor was sliced once (`presliced`).
fn emulation_used(
    kernel: Family,
    used: &ArithUsed,
    presliced: Option<bool>,
    note: Option<String>,
    fit: Option<FitEmulation>,
) -> EmulationUsed {
    EmulationUsed {
        kernel: kernel.name(),
        scheme: used.arith.name(),
        bits: used.arith.emulated_bits().unwrap_or(53),
        presliced,
        reason: used.reason().or(note),
        fit,
    }
}

/// Puts `eng`'s exchange GEMMs on the first arithmetic of the policy's `chain` that sets up (the
/// Ozaki scheme with the stored tensor sliced once, or per build when its planes do not fit;
/// cuBLAS's fixed-mantissa emulation; native FP64), with the reason for a lower tier.
fn emulate_k(gpu: &Gpu, eng: &mut GpuDfJk, chain: &[Arith]) -> EmulationUsed {
    // the setup's fit (GpuDfJkBuilder::new asks for the policy's)
    let r = eng.resolved();
    let fit = Some(match r.fit {
        DfFit::Ozaki { slices } => FitEmulation {
            scheme: "ozaki",
            bits: 8 * slices as u32 - 1,
            reason: None,
        },
        DfFit::Native => FitEmulation {
            scheme: "fp64",
            bits: 53,
            reason: r.fit_note.clone(),
        },
    });
    let used = apply_chain(chain, |a| eng.set_k_math(gpu, kmath(a)));
    let r = eng.resolved();
    let ozaki = matches!(used.arith, Arith::Ozaki { .. });
    emulation_used(
        Family::Jk,
        &used,
        ozaki.then_some(r.k_planes_bytes > 0),
        r.k_planes_note.clone().filter(|_| ozaki),
        fit,
    )
}

/// [`JkBuilder`] over the device DF-J/K engine: J from the densities, K from the symmetric
/// factors of the request (an SCF's occupied orbitals) or, without them, from the densities
/// split as P P^T - N N^T (as the CPU's `DfJk`).
pub(crate) struct GpuDfJkBuilder {
    gpu: Arc<Gpu>,
    eng: GpuDfJk,
    /// The auxiliary basis (the fit of a Coulomb hook).
    aux: BasisSet,
    /// A Coulomb hook (the `predicted` PCM mode) and the fit it is fed from.
    hook: Option<(CoulombFit, Box<dyn linus_base::terms::CoulombHook>)>,
}

/// The fit coefficients of a device J build, c = V^-1 gamma = L^-T g, from the engine's kept
/// g = L^-1 gamma ([`GpuDfJk::last_coulomb_fit`]) and the metric's Cholesky factor L, built here
/// as the engine builds it (the same integrals and cuSOLVER factorization, so the same bits): a
/// [`linus_base::terms::CoulombHook`] gets the coefficients the CPU's `DfJk` gives it, to
/// round-off.
pub(crate) struct CoulombFit {
    naux: usize,
    /// The metric's Cholesky factor and the triangular solve: on the device, or (when the
    /// device's factorization did not fit: a full card, a per-process cap) on the host.
    solve: FitSolve,
    c: linus_gpu::DeviceBuf<f64>,
}

enum FitSolve {
    Device {
        l: linus_gpu::DeviceBuf<f64>,
        gemm: linus_gpu::linalg::Gemm,
    },
    Host(Mat<f64>),
}

impl CoulombFit {
    pub(crate) fn new(gpu: &Gpu, aux: &BasisSet) -> Result<Self, GpuError> {
        Self::with_solve(gpu, aux, false)
    }

    /// As [`CoulombFit::new`]; `host`: the solve on the host whether or not the device's fits.
    fn with_solve(gpu: &Gpu, aux: &BasisSet, host: bool) -> Result<Self, GpuError> {
        let naux = aux.nbf;
        let v = linus_gpu::cpu::metric(&linus_gpu::cpu::Aux3c::new(aux), 0.0, &Operator::Coulomb);
        let device = || -> Result<FitSolve, GpuError> {
            if host {
                return Err(GpuError::NotCovered("the solve on the host".into()));
            }
            let mut l = gpu.upload(&v)?;
            linus_gpu::solver::Solver::new(gpu)?.potrf_lower(gpu, &mut l, naux)?;
            Ok(FitSolve::Device {
                l,
                gemm: linus_gpu::linalg::Gemm::new(gpu)?,
            })
        };
        let solve = match device() {
            Ok(s) => s,
            // the factor (naux^2) and the libraries' workspaces did not fit: the solve, naux^2
            // per density and SCF iteration, runs on the host
            Err(e) if e.falls_back_at_setup() => {
                let l = linus_jk::df::metric_cholesky(from_cm(naux, naux, &v), Par::Seq).map_err(
                    |e| GpuError::Numerical {
                        op: "the Coulomb metric's host potrf",
                        // the order of the failing leading minor, as cuSOLVER's
                        info: match e {
                            linus_jk::df::DfSetupError::Metric { pivot, .. } => {
                                i32::try_from(pivot + 1).unwrap_or(i32::MAX)
                            }
                            _ => 0,
                        },
                    },
                )?;
                FitSolve::Host(l)
            }
            Err(e) => return Err(e),
        };
        Ok(CoulombFit {
            naux,
            solve,
            c: gpu.alloc_zeros(2 * naux.max(1))?,
        })
    }

    /// Feeds the fit coefficients of the engine's last J build to `hook`: on the device when it
    /// takes them there, else on the host, where it must add nothing to J (the density-fitted PCM,
    /// whose extra coefficients the device J does not expand, runs with the CPU's J:
    /// `crate::driver::backend`). Nothing without a kept fit.
    pub(crate) fn feed(
        &mut self,
        gpu: &Gpu,
        eng: &GpuDfJk,
        hook: &mut dyn linus_base::terms::CoulombHook,
    ) -> Result<(), GpuError> {
        let Some((g, nd)) = eng.last_coulomb_fit() else {
            return Ok(());
        };
        let n = self.naux;
        if self.c.len() < n * nd {
            self.c = gpu.alloc_zeros(n * nd)?;
        }
        match &self.solve {
            FitSolve::Device { l, gemm } => {
                use linus_gpu::blas::{MatMut, MatRef, Op};
                gpu.copy_range(g, 0, &mut self.c, 0, n * nd)?;
                gemm.trsm_lower_left(
                    Op::T,
                    (n, nd),
                    MatRef::new(l, 0, n),
                    MatMut::new(&mut self.c, 0, n),
                )?;
            }
            FitSolve::Host(l) => {
                // c = L^-T g, column by column
                let mut c = from_cm(n, nd, &gpu.download_range(g, 0, n * nd)?);
                linus_base::linalg::solve_upper_in_place(l.transpose(), c.as_mut(), Par::Seq);
                gpu.upload_into(&mut self.c, 0, &to_cm(c.as_ref()))?;
            }
        }
        // a device consumer (the GPU's PCM term) takes them on the device
        if let Some(h) = hook
            .device_form()
            .and_then(|a| a.downcast_mut::<crate::backend::gpu::pcm::GpuPcmHook>())
        {
            h.device_coefficients(gpu, &self.c, nd);
            return Ok(());
        }
        let ch = gpu.download_range(&self.c, 0, n * nd)?;
        let c: Vec<Vec<f64>> = ch.chunks(n).map(|x| x.to_vec()).collect();
        let solved = std::cell::Cell::new(false);
        let extra = hook.extra_coefficients(&c, &|_| solved.set(true));
        if solved.get() || extra.iter().any(|e| e.iter().any(|&x| x != 0.0)) {
            return Err(GpuError::NotCovered(
                "a Coulomb hook that changes J (the density-fitted PCM) ran on the device's J \
                     build"
                    .into(),
            ));
        }
        Ok(())
    }
}

/// A DF-J/K engine for `basis` and `aux` on the rung of its storage plan
/// (`linus_gpu::df::plan::plan`: the cheapest rung of decision D11's ladder that fits
/// `opts.device_budget`; streamed only within `host_budget` bytes, None: the device only). A plan
/// with no rung that fits is an out-of-memory or unsupported error, which makes the caller fall
/// back to the CPU (the host loop and the CPU's J/K are the SCF's next rungs,
/// `crate::backend::scf`). The layout, packed integrals and metric come from `shared` when given
/// ([`GpuDfJk::with_shared`]).
fn df_engine(
    gpu: &Gpu,
    basis: &BasisSet,
    aux: &BasisSet,
    opts: GpuDfJkOptions,
    host_budget: Option<usize>,
    shared: Option<&DfShared>,
) -> Result<GpuDfJk, GpuError> {
    let opts = GpuDfJkOptions {
        host_budget: Some(host_budget.unwrap_or(0)),
        ..opts
    };
    GpuDfJk::with_shared(gpu, basis, aux, &opts, shared)
}

/// The storage rung a device DF-J/K engine set up on ([`df_engine`]) and why, for
/// `resolved.record.df.rung`.
fn df_rung(r: &linus_gpu::df::GpuDfJkResolved) -> String {
    let gib = |b: usize| b as f64 / 1073741824.0;
    let (p, auto) = (&r.plan, r.plan.auto_block);
    let ms = |s: f64| 1e3 * s;
    let shrunk = if p.k_block < auto {
        let costs = if r.storage == DfStorage::Host {
            String::new()
        } else {
            format!(
                " (about {:.0} ms per build, against {:.0} ms for streaming the tensor from host \
                 memory: decision D11's costs)",
                ms(p.extra_seconds),
                ms(p.stream_seconds)
            )
        };
        format!(
            ", with the K block shrunk from {auto} to {} functions to fit beside the other engines' \
             reserve{costs}",
            p.k_block
        )
    } else {
        String::new()
    };
    // the SCF's check before its first build (GpuScfJk::fit)
    let fitted = if r.k_block < p.k_block {
        format!(
            "; the K block stepped down from {} to {} functions before the first build: the other \
             engines' per-call needs (VV10's, the PCM's, the XC batches') beside their reserve \
             are no plan item",
            p.k_block, r.k_block
        )
    } else {
        String::new()
    };
    let shrunk = shrunk + &fitted;
    match r.storage {
        DfStorage::Host => format!(
            "streamed from host memory: the tensor ({:.2} GiB) and a build's workspace do not fit the \
             {:.2} GiB the engine may hold{shrunk}",
            gib(r.tensor_bytes),
            gib(r.device_budget)
        ),
        _ => format!(
            "on the device: the tensor ({:.2} GiB) and a build's workspace fit the {:.2} GiB the \
             engine may hold{shrunk}",
            gib(r.tensor_bytes),
            gib(r.device_budget)
        ),
    }
}

/// What the SCF's device J/K is set up with ([`GpuDfJkBuilder::new`]).
pub(crate) struct ScfJkSetup {
    /// The exchange GEMMs' fallback chain and the fit's arithmetic (the device policy's).
    pub k: &'static [Arith],
    pub fit: Arith,
    /// Native K's panel GEMM (the device policy's, [`panel_kernel`]).
    pub k_panel_kernel: linus_gpu::df::PanelKernel,
    /// The device bytes the engine plans within ([`scf_df_budget`]) and the factor columns of
    /// the SCF's builds ([`scf_k_columns`]).
    pub device_budget: usize,
    pub k_columns: usize,
    /// The range separation of a range-separated hybrid (B^omega prepared at setup).
    pub omega: Option<f64>,
    /// Bytes of host memory for a streamed tensor.
    pub host_budget: usize,
    /// The device memory plan's reserve for the other engines.
    pub others_reserve: usize,
    /// Where the three-center integrals are computed: the CPU where the plan found the
    /// generated kernels do not cover the basis sets (decision D40).
    pub integrals: DfIntegrals,
}

impl GpuDfJkBuilder {
    /// The SCF's engine for `basis` and `aux` ([`df_engine`], from `shared`'s layout,
    /// integrals and metric), K in the first arithmetic of the policy's chain that sets up
    /// ([`emulate_k`], returned unless native) and the setup's fit as the policy says (the
    /// Ozaki scheme's blocked updates, [`DfFit::Ozaki`], FP64 when its workspace does not fit;
    /// reported with K's tier); with `omega` (a range-separated hybrid) the long-range tensor
    /// B^omega is prepared at setup, for [`JkBuilder::set_exchange_operator`].
    pub(crate) fn new(
        gpu: Arc<Gpu>,
        basis: &BasisSet,
        aux: &BasisSet,
        setup: ScfJkSetup,
        shared: Option<&DfShared>,
    ) -> Result<(Self, Option<EmulationUsed>), GpuError> {
        let opts = scf_df_options(&setup);
        let mut eng = df_engine(&gpu, basis, aux, opts, Some(setup.host_budget), shared)?;
        let emu = (setup.k != NATIVE).then(|| emulate_k(&gpu, &mut eng, setup.k));
        Ok((
            GpuDfJkBuilder {
                gpu,
                eng,
                aux: aux.clone(),
                hook: None,
            },
            emu,
        ))
    }

    /// The device, the engine and the auxiliary basis (for the device SCF loop's Fock
    /// terms); a Coulomb hook is dropped.
    pub(crate) fn into_parts(self) -> (Arc<Gpu>, GpuDfJk, BasisSet) {
        (self.gpu, self.eng, self.aux)
    }

    /// The builder of an engine set up on `gpu`.
    pub(crate) fn from_parts(gpu: Arc<Gpu>, eng: GpuDfJk, aux: BasisSet) -> Self {
        GpuDfJkBuilder {
            gpu,
            eng,
            aux,
            hook: None,
        }
    }

    /// What the setup resolved to (and the SCF's check before its first build fitted), as
    /// `resolved.record.df` reports it.
    pub(crate) fn resolved(&self) -> ResolvedDf {
        let r = self.eng.resolved();
        ResolvedDf {
            plan: match r.storage {
                DfStorage::Host => DfPlan::HostStreamed,
                _ => DfPlan::Device,
            },
            long_range: match r.lr_omega {
                Some(_) => DfLongRangeSource::Stored,
                None => DfLongRangeSource::None,
            },
            store_bytes: r.tensor_bytes * if r.lr_omega.is_some() { 2 } else { 1 },
            budget_bytes: r.device_budget,
            naux: r.naux,
            integrals: Some(match r.integrals {
                DfIntegrals::Cpu => "cpu",
                _ => "device",
            }),
            integrals_note: r.integrals_note.clone(),
            k_block: Some(r.k_block),
            rung: Some(df_rung(r)),
            plan_violation: r.tensor_violations.first().map(|v| {
                format!(
                    "{}: {} bytes drawn with {} of the {} granted left",
                    v.what, v.requested, v.remaining, v.granted
                )
            }),
            // the panels' GEMM runs for native FP64 K
            k_kernel: matches!(r.k_math, linus_gpu::df::KMath::Native)
                .then(|| r.k_panel_kernel.name()),
        }
    }

    fn run(
        &mut self,
        dens: &[Vec<f64>],
        facs: &[(Vec<f64>, usize)],
        want_j: bool,
        want_k: bool,
    ) -> Result<GpuJkOutput, LinusError> {
        let d: Vec<&[f64]> = dens.iter().map(|x| x.as_slice()).collect();
        let f: Vec<Factor> = facs.iter().map(|(c, n)| Factor { c, ncol: *n }).collect();
        self.eng
            .compute(
                &self.gpu,
                &GpuJkRequest {
                    dens: &d,
                    factors: &f,
                    want_j,
                    want_k,
                },
            )
            .map_err(|e| scf_build_error(Family::Jk, e))
    }

    /// The least device memory one build of `nd` densities and the factors of `ncols` columns
    /// allocates beyond the engine ([`GpuDfJk::build_bytes`]; `device_loop`: the device SCF
    /// loop's builds, whose inputs and results are the loop's), and the device's free bytes now.
    pub(crate) fn build_fit(
        &self,
        nd: usize,
        ncols: &[usize],
        device_loop: bool,
    ) -> (usize, usize) {
        (
            self.eng.build_bytes(nd, ncols, !device_loop),
            self.gpu.budget().free(),
        )
    }
}

/// An error of an SCF's J/K build (or its device loop) as the calculation's: out of device memory
/// after the setup's check (`GpuScfJk::fit` sizes the builds it knows) is a
/// `resource` error that names what to change, any other [`device_error`].
pub(crate) fn scf_build_error(k: Family, e: GpuError) -> LinusError {
    match e {
        GpuError::OutOfMemory { .. } => LinusError::device_memory(
            format!(
                "GPU backend, {}: an SCF build ran out of device memory after the setup ({e}); give the \
                 process more device memory (device_memory_gib, or a less busy card) or run backend \"cpu\"",
                k.name()
            ),
            crate::backend::gpu::device_need(&e),
        ),
        e => device_error(k, e),
    }
}

impl JkBuilder for GpuDfJkBuilder {
    /// The wK exchange of a range-separated hybrid on the device (B^omega prepared at setup,
    /// [`GpuDfJkBuilder::new`]); false when the device cannot hold it.
    fn set_exchange_operator(&mut self, op: Operator) -> bool {
        self.eng.set_exchange_operator(&self.gpu, op).is_ok()
    }

    /// The `predicted` PCM mode's hook: every J build keeps its fit and feeds the coefficients
    /// to the hook ([`CoulombFit`]). It falls back (with the reason) when the fit's setup does
    /// not fit the device or is not covered, and fails on any other device error.
    fn set_coulomb_hook(
        &mut self,
        hook: Option<Box<dyn linus_base::terms::CoulombHook>>,
    ) -> Result<linus_base::terms::HookOutcome, LinusError> {
        use linus_base::terms::HookOutcome;
        match hook {
            None => {
                self.hook = None;
                self.eng.set_keep_coulomb_fit(false);
                Ok(HookOutcome::Installed)
            }
            Some(h) => match CoulombFit::new(&self.gpu, &self.aux) {
                Ok(fit) => {
                    self.eng.set_keep_coulomb_fit(true);
                    self.hook = Some((fit, h));
                    Ok(HookOutcome::Installed)
                }
                Err(e) if e.falls_back_at_setup() => Ok(HookOutcome::FellBack {
                    reason: e.to_string(),
                }),
                Err(e) => Err(device_error(Family::Jk, e)),
            },
        }
    }

    fn name(&self) -> String {
        let r = self.eng.resolved();
        format!(
            "gpu-df(naux={}, rows={}, storage={:?}, k={:?})",
            r.naux, r.nx, r.storage, r.k_math
        )
    }

    fn compute(&mut self, req: &JkRequest, par: Par) -> Result<JkOutput, LinusError> {
        let n = self.eng.nbf();
        let sq = |v: &Vec<f64>| from_cm(n, n, v);
        let dens: Vec<Vec<f64>> = if req.want_j {
            req.dens.iter().map(|d| to_cm(d.as_ref())).collect()
        } else {
            Vec::new()
        };
        let mut out = JkOutput::default();
        if !req.want_k {
            let r = self.run(&dens, &[], req.want_j, false)?;
            out.j = r.j.iter().map(sq).collect();
            self.feed_hook()?;
            return Ok(out);
        }
        match req.factors.and_then(|f| f.symmetric()) {
            Some(cs) => {
                let facs: Vec<(Vec<f64>, usize)> =
                    cs.iter().map(|c| (to_cm(c.as_ref()), c.ncols())).collect();
                let r = self.run(&dens, &facs, req.want_j, true)?;
                out.j = r.j.iter().map(sq).collect();
                out.k = r.k.iter().map(sq).collect();
            }
            None => {
                // D = P P^T - N N^T: K[D] = K[P] - K[N]
                let split: Vec<(Mat<f64>, Mat<f64>)> = req
                    .dens
                    .iter()
                    .map(|d| factor_symmetric(d.as_ref(), par))
                    .collect();
                let facs: Vec<(Vec<f64>, usize)> = split
                    .iter()
                    .flat_map(|(p, m)| {
                        [
                            (to_cm(p.as_ref()), p.ncols()),
                            (to_cm(m.as_ref()), m.ncols()),
                        ]
                    })
                    .collect();
                let r = self.run(&dens, &facs, req.want_j, true)?;
                out.j = r.j.iter().map(sq).collect();
                out.k = (0..split.len())
                    .map(|i| sq(&r.k[2 * i]) - sq(&r.k[2 * i + 1]))
                    .collect();
            }
        }
        if req.want_j {
            self.feed_hook()?;
        }
        Ok(out)
    }
}

impl GpuDfJkBuilder {
    /// The hook's share of the J build just done (none without a hook or a J build).
    fn feed_hook(&mut self) -> Result<(), LinusError> {
        if let Some((fit, hook)) = self.hook.as_mut() {
            fit.feed(&self.gpu, &self.eng, hook.as_mut())
                .map_err(|e| device_error(Family::Jk, e))?;
        }
        Ok(())
    }
}

/// The SCF's J/K on the device (family `jk`): the builder, which the device loop takes while it
/// runs ([`crate::backend::gpu::scf_loop`]).
pub(crate) struct GpuScfJk {
    b: Option<GpuDfJkBuilder>,
}

impl GpuScfJk {
    /// The builder, for the device loop (None once it has been taken).
    pub(crate) fn take(&mut self) -> Option<GpuDfJkBuilder> {
        self.b.take()
    }

    /// The builder back from the device loop.
    pub(crate) fn put(&mut self, b: GpuDfJkBuilder) {
        self.b = Some(b);
    }

    fn get(&mut self) -> &mut GpuDfJkBuilder {
        self.b
            .as_mut()
            .expect("the SCF's device J/K after the device loop took it")
    }
}

impl ScfJk for GpuScfJk {
    fn builder(&mut self) -> &mut dyn JkBuilder {
        self.get()
    }

    fn into_builder(self: Box<Self>) -> Box<dyn JkBuilder> {
        Box::new(
            self.b
                .expect("the SCF's device J/K after the device loop took it"),
        )
    }

    fn df_record(&mut self) -> Option<ResolvedDf> {
        self.b.as_ref().map(|b| b.resolved())
    }

    /// The device's J/K against the device memory free once every SCF engine is set up, before
    /// the first iteration (decided here, never between iterations): the SCF's J/K builds known
    /// in advance (of the SCF `run`: the iterations', a closed shell's probe builds, the SAD
    /// guess's) and,
    /// where the plan puts the loop on the device, the device loop's state. When the device
    /// loop's builds and state do not fit, the host loop runs (`loop_declined`); when the host
    /// loop's builds do not fit either, declined: the CPU's J/K. A build that still runs out of
    /// device memory (one not known here: a guess density that is not a superposition of atomic
    /// blocks; a PCM set up after this) is a `resource` error.
    fn fit(
        &mut self,
        cx: &CalcContext,
        run: &ScfRun<'_>,
        loop_declined: &mut Option<String>,
    ) -> Result<Setup<()>, LinusError> {
        use crate::driver::backend::Placement;
        let device_loop =
            cx.placement(Family::ScfLoop) == Some(Placement::Device) && loop_declined.is_none();
        let (builds, probe) = scf_builds(cx, run);
        let reserve = device_memory(cx)?.others_reserve;
        let b = self.get();
        let gib = |x: usize| x as f64 / 1073741824.0;
        // the panel path's automatic block stepped down until the largest build fits what the
        // other engines left, their reserve kept free for what they allocate per call. The
        // setup's plan cannot know this yet: it counts the other engines by that reserve alone,
        // and their setups and per-call needs (VV10's, the PCM's, the XC batches') are no plan
        // item (u1an39/def2-SVP wB97M-V UKS with C-PCM under an 8 GB cap ran out of device
        // memory in its second build). Here every engine is set up and nothing runs beside
        // this check, and `free` is the device pool's own accounting under its limit (not the
        // driver's free memory, which other processes on the card move): the same block for
        // one device, cap and calculation. The record says so (`df_rung`).
        loop {
            let (_, free) = b.build_fit(0, &[], false);
            let largest = builds
                .iter()
                .map(|(nd, cols)| b.build_fit(*nd, cols, false).0)
                .max()
                .unwrap_or(0);
            if largest + reserve <= free || !b.eng.shrink_panel_block() {
                break;
            }
        }
        // the closed-shell probe's builds count whether or not its gap gate fires
        let counted = if probe {
            " (the closed-shell probe's two-spin builds counted)"
        } else {
            ""
        };
        let (_, free) = b.build_fit(0, &[], false);
        let need = |device_loop: bool| {
            builds
                .iter()
                .map(|(nd, cols)| b.build_fit(*nd, cols, device_loop).0)
                .max()
                .unwrap_or(0)
        };
        if device_loop {
            let n = need(true) + device_loop_state(cx, run);
            if n <= free {
                return Ok(Setup::Ready(()));
            }
            *loop_declined = Some(format!(
                "device call: the device loop's state and largest J/K build need {:.2} GiB of device \
                 memory{counted}, {:.2} GiB is free beside the engines, so the host loop ran",
                gib(n),
                gib(free)
            ));
        }
        let n = need(false);
        if n <= free {
            return Ok(Setup::Ready(()));
        }
        cx.note_emulation(Family::Jk, None);
        Ok(Setup::Declined(format!(
            "device call: the SCF's largest J/K build needs {:.2} GiB of device memory{counted}, \
             {:.2} GiB is free beside the engines",
            gib(n),
            gib(free)
        )))
    }
}

/// The J/K builds of the SCF `run` known before it starts, as (densities, factor columns),
/// with whether the closed-shell probe's builds are among them: the iterations' (a density and its
/// occupied orbitals per channel), a closed shell's probe builds when the closed-shell probe may
/// run (both spins), and the guess's when the guess density is a superposition of atomic blocks
/// (the SAD guess: its factors' columns counted per block, as `factor_symmetric` keeps them).
/// Without exact exchange, no factors.
fn scf_builds(cx: &CalcContext, run: &ScfRun<'_>) -> (Vec<(usize, Vec<usize>)>, bool) {
    let (na, nb) = (run.inp.nalpha, run.inp.nbeta);
    let opts = run.opts;
    let restricted = opts.restricted.unwrap_or(na == nb);
    let k = linus_base::terms::exchange_parts(cx.method.exchange).0 != 0.0;
    let cols = |c: Vec<usize>| if k { c } else { Vec::new() };
    let mut builds = vec![if restricted {
        (1, cols(vec![na]))
    } else {
        (2, cols(vec![na, nb]))
    }];
    let probe = opts.algo.closed_shell_probe.unwrap_or(0) > 0
        || opts
            .probe_trigger
            .as_ref()
            .is_some_and(|t| t.closed_products > 0);
    if restricted && probe {
        builds.push((2, cols(vec![na, na])));
    }
    if let Some((da, db)) = &run.inp.guess_density {
        let channels: Vec<&Mat<f64>> = if restricted { vec![da] } else { vec![da, db] };
        let counts: Option<Vec<usize>> = channels
            .iter()
            .map(|d| block_factor_columns(d, &cx.basis))
            .collect();
        if let Some(c) = counts {
            builds.push((channels.len(), cols(c)));
        }
    }
    (builds, restricted && probe)
}

/// The device SCF loop's state for the SCF `run`, at least (bytes; its DIIS subspace and
/// eigensolver workspace not counted): the n^2 matrices per channel and shared, and the occupied
/// orbitals.
fn device_loop_state(cx: &CalcContext, run: &ScfRun<'_>) -> usize {
    let (na, nb) = (run.inp.nalpha, run.inp.nbeta);
    let n = cx.basis.nbf;
    let restricted = run.opts.restricted.unwrap_or(na == nb);
    let (nsp, ncol) = if restricted { (1, na) } else { (2, na + nb) };
    let k = usize::from(linus_base::terms::exchange_parts(cx.method.exchange).0 != 0.0);
    8 * ((7 + (8 + k) * nsp) * n * n + n * ncol)
}

/// The columns of the symmetric factors of the density `d` (`two_electron::df::factor_symmetric`:
/// its eigenvectors with eigenvalues above 1e-14 of the largest in magnitude, positive and
/// negative) when `d` is a superposition of atomic blocks of `basis` (None otherwise): counted
/// per atom, from the blocks' eigenvalues, without the whole matrix's decomposition.
fn block_factor_columns(d: &Mat<f64>, basis: &BasisSet) -> Option<usize> {
    let n = d.nrows();
    let mut atom = vec![usize::MAX; n];
    for s in &basis.shells {
        atom[s.ao_start..s.ao_start + s.nfunc()].fill(s.atom);
    }
    let block_diagonal = (0..n).all(|j| (0..n).all(|i| atom[i] == atom[j] || d[(i, j)] == 0.0));
    if !block_diagonal {
        return None;
    }
    let mut w = Vec::with_capacity(n);
    for a in 0..basis.atom_shells.len() {
        let idx: Vec<usize> = (0..n).filter(|&i| atom[i] == a).collect();
        if idx.is_empty() {
            continue;
        }
        let b = Mat::from_fn(idx.len(), idx.len(), |i, j| d[(idx[i], idx[j])]);
        w.extend(linus_base::linalg::eigh(b.as_ref(), Par::Seq).0);
    }
    let cut = 1e-14 * w.iter().fold(0.0f64, |a, &x| a.max(x.abs())).max(1e-300);
    Some(w.iter().filter(|&&x| x > cut || x < -cut).count())
}

/// The DF layout, packed integrals and metric of the SCF's J/K on the device for `aux` (the
/// layout screens the geometry's shell pairs, built while the device opened, with their
/// Schwarz factors: the same layout as from pairs of its own).
pub(crate) fn df_shared(calc: &CalcContext, aux: &BasisSet) -> Result<Arc<DfShared>, GpuError> {
    let pairs = calc
        .integrals
        .pairs(&calc.basis, linus_ints::DerivOrder::Zero);
    DfShared::with_pairs(
        &calc.basis,
        aux,
        &GpuDfJkOptions::default().thresholds,
        Some(&pairs),
    )
    .map(Arc::new)
}

/// What the SCF's device J/K is set up with: the device policy's arithmetic, the memory plan's
/// reserve, `scf`'s host budget, the plan's integrals, and the range separation `omega` of a
/// range-separated exchange on the engine (None: J alone, or K of another engine).
pub(crate) fn scf_jk_setup(
    calc: &CalcContext,
    scf: &ScfSetup,
    omega: Option<f64>,
) -> Result<ScfJkSetup, LinusError> {
    scf_jk_setup_within(calc, scf.memory.df_bytes(), omega)
}

/// [`scf_jk_setup`] with `host_budget` bytes of page-locked host memory for a streamed tensor
/// (zero: the device only), for callers ahead of the SCF's setup ([`scf_df_plan`]).
fn scf_jk_setup_within(
    calc: &CalcContext,
    host_budget: usize,
    omega: Option<f64>,
) -> Result<ScfJkSetup, LinusError> {
    let policy = device_policy(calc)?;
    Ok(ScfJkSetup {
        k: policy.scf_k,
        fit: policy.scf_fit,
        k_panel_kernel: panel_kernel(policy),
        device_budget: scf_df_budget(calc)?,
        k_columns: scf_k_columns(calc)?,
        omega,
        host_budget,
        others_reserve: device_memory(calc)?.others_reserve,
        integrals: match calc.plan.uncovered() {
            Some(_) => DfIntegrals::Cpu,
            None => DfIntegrals::Auto,
        },
    })
}

/// The SCF's J/K on the device for the auxiliary basis `aux`, set up with `scf` (timed as
/// `jk_setup`), with the range-separated exchange operator of the method, on `shared` (the
/// layout, integrals and metric a caller built already) or a [`df_shared`] of its own;
/// declined (the CPU's `DfJk`) when the setup does not fit.
pub(crate) fn scf_jk(
    calc: &CalcContext,
    gpu: &Arc<Gpu>,
    scf: &ScfSetup,
    aux: &BasisSet,
    shared: Option<Arc<DfShared>>,
    timers: &mut Timers,
) -> Result<Setup<ScfJkBuilt>, LinusError> {
    let setup_opts = scf_jk_setup(calc, scf, calc.method.omega())?;
    let built = timers.time("jk_setup", || {
        // the layout, packed integrals and metric, kept for the derivatives
        // (crate::backend::gpu::session) of an SCF converged for them
        let shared = match shared {
            Some(s) => s,
            None => df_shared(calc, aux)?,
        };
        calc.hand_off_df(scf.purpose, shared.clone(), aux);
        GpuDfJkBuilder::new(gpu.clone(), &calc.basis, aux, setup_opts, Some(&shared))
    });
    let (mut b, emu) = match setup(Family::Jk, built)? {
        Setup::Ready(x) => x,
        Setup::Declined(why) => {
            calc.note_emulation(Family::Jk, None);
            return Ok(Setup::Declined(why));
        }
    };
    calc.note_emulation(Family::Jk, emu);
    let resolved = b.resolved();
    timers.resource("df_store_gib", resolved.store_bytes as f64 / 1073741824.0);
    timers.resource("df_bytes_per_elem", 8.0);
    // a range-separated hybrid's operator on the engine (the device loop's K is then a K + b
    // K_LR, exchange fraction 1)
    crate::two_electron::set_exchange_operator(&mut b, calc.method.exchange)?;
    Ok(Setup::Ready((
        Box::new(GpuScfJk { b: Some(b) }),
        Some(resolved),
    )))
}
/// Puts the response engine's exchange GEMMs on the first arithmetic of `chain` that sets up
/// ([`emulate_k`]'s, on the pre-sliced planes of its tensor), or, when the exchange will run on
/// the stored operator (`operator`), its GEMMs in cuBLAS's emulation (else FP64): the Ozaki
/// scheme's GEMM1 planes would sit unused, so they are not built.
fn emulate_response(
    gpu: &Gpu,
    eng: &mut GpuDfResponse,
    chain: &[Arith],
    operator: bool,
) -> EmulationUsed {
    if operator {
        let why =
            "the exchange runs on the stored operator (K as an (n o)^2 matrix): no Ozaki GEMM1 planes";
        let used = apply_chain(OPERATOR_EMULATION, |a| eng.set_k_math(gpu, resp_math(a)));
        let note = Some(match used.reason() {
            Some(r) => format!("{why}; {r}"),
            None => why.to_string(),
        });
        let used = ArithUsed {
            declined: Vec::new(),
            ..used
        };
        return emulation_used(Family::ResponseJk, &used, None, note, None);
    }
    let used = apply_chain(chain, |a| eng.set_k_math(gpu, resp_math(a)));
    let ozaki = matches!(used.arith, Arith::Ozaki { .. });
    emulation_used(Family::ResponseJk, &used, ozaki.then_some(true), None, None)
}

/// The session's device policy (a device is open when a family runs on it).
pub(crate) fn device_policy(calc: &CalcContext) -> Result<&DevicePolicy, LinusError> {
    calc.device_policy()
        .ok_or_else(|| LinusError::Internal("a device setup without a device".into()))
}

/// The engine options of the SCF's device DF-J/K set up with `setup` ([`GpuDfJkBuilder::new`],
/// [`scf_df_plan`]).
fn scf_df_options(setup: &ScfJkSetup) -> GpuDfJkOptions {
    GpuDfJkOptions {
        omega: setup.omega,
        fit: df_fit(setup.fit),
        k_panel_kernel: setup.k_panel_kernel,
        // the plan's budget is net of the other engines' reserve: no second reserve on the
        // shrunk rungs
        device_budget: Some(setup.device_budget),
        host_budget: Some(setup.host_budget),
        shrink_reserve: Some(0),
        k_columns: Some(setup.k_columns),
        others_reserve: setup.others_reserve,
        integrals: setup.integrals,
        // the exchange's arithmetic, set after the setup ([`emulate_k`]): its workspace counts in
        // the storage plan
        k_planned: setup.k.first().map_or(KMath::Native, |&a| kmath(a)),
        ..GpuDfJkOptions::default()
    }
}

/// The storage plan of the SCF's device DF-J/K for `sizes` (`linus_gpu::df::plan::plan`, the
/// rule its setup applies, with the setup's bytes beside the tensors less the device integrals'
/// builder, known only once packed), with B^omega for `omega` and `host_budget` bytes of
/// page-locked host memory for streamed tensors (zero: the device only): for callers that decide
/// ahead of the setup (the placement's memory needs; the memory-lean J's choice).
pub(crate) fn scf_df_plan(
    calc: &CalcContext,
    sizes: linus_gpu::df::plan::DfSizes,
    omega: Option<f64>,
    host_budget: usize,
) -> Result<Result<linus_gpu::df::plan::DfPlan, linus_gpu::df::plan::DfPlanRefusal>, LinusError> {
    let setup = scf_jk_setup_within(calc, host_budget, omega)?;
    let opts = scf_df_options(&setup);
    let request = linus_gpu::df::plan_request(
        &opts,
        sizes,
        setup.device_budget,
        linus_gpu::df::plan::setup_bytes(&sizes),
    );
    Ok(linus_gpu::df::plan::plan(&request))
}

/// The device bytes the SCF's DF-J/K engine plans within (`linus_gpu::df::plan`): the memory
/// plan's device limit less its reserve for the other engines (the XC engines, whose grid
/// partition sets up beside it, PCM, the device loop's state). A share of the plan, not the
/// memory free at the setup, so the storage, the K block and the bits do not depend on what the
/// other engines hold at that moment.
pub(crate) fn scf_df_budget(calc: &CalcContext) -> Result<usize, LinusError> {
    let m = device_memory(calc)?;
    Ok(m.limit.saturating_sub(m.others_reserve))
}

/// The factor columns of the SCF's largest builds, which the panel path's block is sized for:
/// none without exact exchange (J alone); a closed shell's two-spin builds (the closed-shell
/// probe's, twice the occupied orbitals; the SAD guess's per-atom factors stay below them);
/// both spins' for open shells. A build with more columns runs in passes, each reading the
/// tensor again (sized for a closed shell's occupied orbitals alone, ts61/def2-QZVP's J/K took
/// 19 % longer on the H200, the probe's and the guess's builds in two passes or more).
pub(crate) fn scf_k_columns(calc: &CalcContext) -> Result<usize, LinusError> {
    // as the SCF's builds count their factors (scf_builds)
    if linus_base::terms::exchange_parts(calc.method.exchange).0 == 0.0 {
        return Ok(0);
    }
    let (na, nb) = calc.mol.nalpha_nbeta().map_err(LinusError::Input)?;
    Ok(if calc.plan.restricted().unwrap_or(na == nb) {
        2 * na
    } else {
        na + nb
    })
}

/// Native K's panel GEMM of the device policy: the sm_90a tensor-core kernel
/// (`DevicePolicy::k_tensor_cores`), else cuBLAS on the gathered panels.
pub(crate) fn panel_kernel(policy: &DevicePolicy) -> linus_gpu::df::PanelKernel {
    if policy.k_tensor_cores {
        linus_gpu::df::PanelKernel::MmaSm90a
    } else {
        linus_gpu::df::PanelKernel::Strided
    }
}

/// The session's device memory plan (a device is open when a family runs on it).
pub(crate) fn device_memory(
    calc: &CalcContext,
) -> Result<crate::driver::memory::DeviceMemoryPlan, LinusError> {
    calc.device_memory()
        .ok_or_else(|| LinusError::Internal("a device setup without a device".into()))
}

/// The memory a device response J/K is set up with ([`GpuResponseJk::new`]).
pub(crate) struct ResponseMemory {
    /// Bytes of host memory for a streamed tensor.
    pub tensor_host: usize,
    /// Bytes of page-locked host memory for the stored operator's chunks that do not fit the
    /// device (streamed to it in every product).
    pub operator_host: usize,
    /// The device's reserves and workspaces.
    pub device: crate::driver::memory::DeviceMemoryPlan,
}

/// The label of an exchange arithmetic in `resolved.record.backend.response`.
fn jk_label(m: RespMath) -> &'static str {
    match m {
        RespMath::Native => "fp64",
        RespMath::Single => "fp32",
        RespMath::Emulated { .. } | RespMath::Ozaki { .. } => "emulated",
    }
}

/// How the device's response J/K sets up the plan's arithmetic ([`ResponseArith`]: what the
/// response computes in, decisions D23 and D37) under the device policy: the engine's settings
/// that follow from it.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ResponseRecipe {
    /// The exchange: FP64 (on the first arithmetic of `chain` that sets up) or FP32.
    pub k: Arith,
    /// The FP64 exchange's fallback chain through the tensor: the SCF exchange's ([`NATIVE`], or
    /// the policy's emulation chain), native for an FP32 exchange; on the stored operator
    /// [`OPERATOR_EMULATION`] instead.
    pub chain: &'static [Arith],
    /// The stored operator in FP32 where it pays and fits: `precision.fp32: on`, where the
    /// response runs single precision.
    pub fp32_operator: bool,
    /// [`ResponseMode::AutoOperator`]'s cost model (decision D37): where the CPU would run single
    /// precision, on a device with little FP64 throughput, the stored operator in FP64 or the
    /// FP32 exchange through the tensor, whichever is cheaper, the operator's setup emulated as
    /// the other exchange GEMMs are.
    pub costing: Option<ExchangeCosting>,
    /// The CPKS kernel runs as a single-precision response's (every mode but double;
    /// [`DevicePolicy::cpks_xc`]).
    pub single_kernel: bool,
}

impl ResponseRecipe {
    /// The recipe of the arithmetic `arith` under `policy`; `field`: an IR run's electric-field
    /// solve follows (D37's costing weighs its products too).
    pub(crate) fn new(policy: &DevicePolicy, arith: &ResponseArith, field: bool) -> Self {
        let single = arith.mode == ResponseMode::Single;
        ResponseRecipe {
            k: arith.k,
            chain: if single { NATIVE } else { policy.scf_k },
            fp32_operator: single && policy.fp32 == Fp32Mode::On,
            costing: (arith.mode == ResponseMode::AutoOperator).then_some(ExchangeCosting {
                rates: policy.rates,
                setup_emulated: policy.precision == DevicePrecision::Emulated,
                field,
            }),
            single_kernel: arith.mode != ResponseMode::Double,
        }
    }
}

/// [`ResponseJk`] over the device's response J/K: each channel's factors side by side
/// (n x nd o_s, column-major) in one call for all perturbations, the results split back into
/// `[s][d]`. The tensor and the projected Y = B C stay on the device, so `release` and
/// `restore` have nothing to do. The exchange runs on the stored operator when the products
/// pay for it and it fits (`linus_gpu::response::df::KOperator::Auto`), recorded in
/// `resolved.record.backend.response` after each product.
pub(crate) struct GpuResponseJk {
    gpu: Arc<Gpu>,
    eng: GpuDfResponse,
    n: usize,
    occ: Vec<usize>,
    timers: Timers,
    records: ResponseRecords,
    purpose: &'static str,
}

impl GpuResponseJk {
    /// The response J/K of `basis` and `aux` for the occupied orbitals `occ` (one or two
    /// channels), the exchange operator `xop` of a range-separated hybrid (wK; None: global
    /// exchange) and the policy's arithmetic as its `recipe`: the FP64 exchange on the first
    /// arithmetic of its chain that sets up (returned unless native): the Ozaki scheme on the
    /// tensor's pre-sliced planes, else cuBLAS's emulation, else FP64. The tensor goes
    /// on the device when it, the projected Y and the exchange workspace fit, else it is
    /// streamed from host memory ([`df_engine`]; not for range-separated exchange, which needs
    /// it on the device), within `memory` (host bytes for a streamed tensor and for the stored
    /// operator's chunks beyond the device; the device plan's workspace and reserves).
    /// `expected_nd`: the perturbations of the products to come (a
    /// Hessian's 3N), when known: in emulation, an exchange that will run on the stored
    /// operator skips the Ozaki planes ([`emulate_response`]). `auto`
    /// ([`ResponseMode::AutoOperator`] on a device with little FP64 throughput): FP64 on the
    /// operator when it pays, fits (on the device, or streamed from host memory) and is the
    /// cheaper path by the policy's exchange costing, else (and if it later does not fit) the
    /// FP32 exchange through the tensor; returns the choice.
    #[allow(
        clippy::too_many_arguments,
        reason = "the response's inputs: the orbitals, the operator, the recipe, its memory and records"
    )]
    pub(crate) fn new(
        gpu: Arc<Gpu>,
        basis: &BasisSet,
        aux: &BasisSet,
        occ: &[MatRef<'_, f64>],
        xop: Option<Operator>,
        recipe: &ResponseRecipe,
        memory: ResponseMemory,
        expected_nd: Option<usize>,
        (records, purpose): (ResponseRecords, &'static str),
        shared: Option<&DfShared>,
    ) -> Result<(Self, Option<EmulationUsed>, Option<AutoChoice>), GpuError> {
        let n = basis.nbf;
        let plan = memory.device;
        let k_math = resp_math(recipe.k);
        let ropts = GpuDfResponseOptions {
            k_math,
            k_operator_fallback: recipe.costing.is_some().then_some(RespMath::Single),
            // the operator's chunks beyond the device in page-locked host memory (auto: where
            // its costing found the streamed operator the cheaper path; the products are
            // bitwise the same with any split)
            k_operator_host: memory.operator_host,
            // FP32 (`fp32`, and auto where the FP64 operator does not fit): through the
            // tensor; the FP32 operator only with `precision.fp32: on` (`fp32_operator`): its
            // Hessians lie 2-3.5x further from FP64 than the FP32 tensor exchange's (RESP2.4),
            // far below the grid's error
            k_operator_fp32: recipe.fp32_operator,
            k_operator_host32: Some(memory.operator_host),
            // the exchange workspace per call and the operator's reserve: the defaults, at most
            // a quarter of the device's limit each (a capped process, device_memory_gib: an 8
            // GB cap left the tensor a budget of 0 and every response J/K fell back to the CPU,
            // and the 6 GiB reserve sent larger molecules' operators to the FP32 exchange)
            workspace: plan.response_workspace,
            k_operator_reserve: plan.k_operator_reserve,
            w_planes_reserve: plan.w_planes_reserve,
            ..GpuDfResponseOptions::default()
        };
        // leave the device room for Y (and Yw) and the exchange workspace
        let rs = xop.is_some();
        let ybytes: usize = occ
            .iter()
            .map(|c| 8 * n * c.ncols() * aux.nbf)
            .sum::<usize>()
            * if rs { 2 } else { 1 };
        let free = gpu.budget().free();
        let jopts = GpuDfJkOptions {
            device_budget: Some(free.saturating_sub(ybytes + ropts.workspace)),
            others_reserve: plan.others_reserve,
            ..GpuDfJkOptions::default()
        };
        // range-separated exchange: its operator tensor needs the tensor on the device
        let jk = df_engine(
            &gpu,
            basis,
            aux,
            jopts,
            (!rs).then_some(memory.tensor_host),
            shared,
        )?;
        let cm: Vec<Vec<f64>> = occ.iter().map(|c| to_cm(*c)).collect();
        let fs: Vec<Factor> = cm
            .iter()
            .zip(occ)
            .map(|(c, m)| Factor { c, ncol: m.ncols() })
            .collect();
        let mut eng = GpuDfResponse::new(&gpu, jk, basis, aux, &fs, xop, &ropts)?;
        let plan = expected_nd.map(|nd| eng.k_operator_plan(&gpu, nd));
        let mut operator = plan.as_ref().is_some_and(|p| p.pays && p.blocked.is_none());
        let mut choice = recipe
            .costing
            .map(|c| auto_choice(&c, plan.as_ref(), operator, expected_nd));
        // the FP64 operator granted now (the memory plan's grant): built at the first product
        // within the device bytes set aside, whatever the Hessian allocates in between; not
        // granted, the decision is the tensor's for good (and auto's costing weighs again)
        if let (true, false, Some(nd)) = (
            operator && !choice.as_ref().is_some_and(|c| c.single),
            recipe.fp32_operator,
            expected_nd,
        ) {
            match eng.grant_k_operator(&gpu, nd) {
                Ok(_) => {}
                Err(e) if e.falls_back_at_setup() => {
                    operator = false;
                    eng.set_k_operator(linus_gpu::response::df::KOperator::Off);
                    choice = recipe
                        .costing
                        .map(|c| auto_choice(&c, plan.as_ref(), false, expected_nd));
                }
                Err(e) => return Err(e),
            }
        }
        let single = choice.as_ref().is_some_and(|c| c.single);
        if single {
            // auto without the FP64 operator: the FP32 exchange, as the CPU's rule chose
            eng.set_k_math(&gpu, RespMath::Single)?;
            operator = false;
        }
        let emu = (recipe.chain != NATIVE && !single && k_math == RespMath::Native)
            .then(|| emulate_response(&gpu, &mut eng, recipe.chain, operator));
        Ok((
            GpuResponseJk {
                gpu,
                eng,
                n,
                occ: occ.iter().map(|c| c.ncols()).collect(),
                timers: Timers::default(),
                records,
                purpose,
            },
            emu,
            choice,
        ))
    }
}

impl ResponseJk for GpuResponseJk {
    fn name(&self) -> String {
        let r = self.eng.resolved();
        format!(
            "gpu-df-resp(naux={}, storage={:?}, k={:?})",
            r.naux, r.storage, r.k_math
        )
    }

    fn fock_occ(
        &mut self,
        a: &[Vec<Mat<f64>>],
        c_j: f64,
        c_k: f64,
        _par: Par,
    ) -> Result<Vec<Vec<Mat<f64>>>, LinusError> {
        let n = self.n;
        assert_eq!(
            a.len(),
            self.occ.len(),
            "GpuResponseJk: one list of factors per channel"
        );
        let nd = a.first().map_or(0, |v| v.len());
        let cat: Vec<Vec<f64>> = a
            .iter()
            .map(|ad| ad.iter().flat_map(|x| to_cm(x.as_ref())).collect())
            .collect();
        let refs: Vec<&[f64]> = cat.iter().map(|v| v.as_slice()).collect();
        let g = self
            .eng
            .fock_occ(&self.gpu, &refs, nd, c_j, c_k)
            .map_err(|e| device_error(Family::ResponseJk, e))?;
        for (name, s) in self.eng.last.phases() {
            self.timers.add(&format!("resp_gpu_{name}"), s);
        }
        if c_k != 0.0 {
            let r = self.eng.resolved();
            self.records.set_exchange(
                self.purpose,
                jk_label(r.k_math),
                (r.k_operator_bytes > 0).then(|| {
                    (
                        r.k_operator_bytes as f64 / 1073741824.0,
                        r.k_operator_host_bytes as f64 / 1073741824.0,
                    )
                }),
                r.k_operator_note.clone(),
            );
            self.records
                .set_grant(self.purpose, r.k_operator_granted, &r.k_operator_violations);
        }
        Ok(g.iter()
            .zip(&self.occ)
            .map(|(gs, &o)| {
                (0..nd)
                    .map(|d| from_cm(n, o, &gs[d * n * o..(d + 1) * n * o]))
                    .collect()
            })
            .collect())
    }

    fn release(&mut self) {}

    fn restore(&mut self, _par: Par) {}

    fn timers(&self) -> Option<Timers> {
        Some(self.timers.clone())
    }

    /// The device engine serves another response of its occupied orbitals when its exchange
    /// runs in double precision (native FP64, or with `precision.fp64: emulated` its FP64
    /// emulation, as that response's own engine would): an IR run's electric-field products
    /// then run on the Hessian's stored exchange operator, if it built one (else through the
    /// tensor, without a second setup). Not after its exchange went to FP32.
    fn serve(&mut self, purpose: &'static str) -> bool {
        let r = self.eng.resolved();
        if r.k_math == RespMath::Single {
            return false;
        }
        let label = jk_label(r.k_math);
        let from = self.purpose;
        self.records.start(
            purpose,
            Some(label),
            format!("the {from}'s response J/K in double precision, kept for these products"),
        );
        self.purpose = purpose;
        self.timers = Timers::default();
        true
    }
}

/// The response J/K on the device for the occupied orbitals `occ`, the auxiliary basis `aux`
/// and the exchange `xop`, when the session puts the response J/K there and the setup fits
/// (None: the CPU's `DfResponseJk`, with the reason recorded). One level (no iteration J/K);
/// the exchange arithmetic from `precision.response` as on the CPU (fp32: FP32 GEMMs;
/// polarizabilities: double) and `precision.fp64` (emulated: FP64 emulation for double).
pub(crate) fn response_setup(
    calc: &CalcContext,
    gpu: &Arc<Gpu>,
    req: &ResponseRequest<'_>,
    field: bool,
    aux: &BasisSet,
    rp: &crate::driver::plan::ResponsePlan,
    timers: &mut Timers,
) -> Result<Setup<ResponseSetup>, LinusError> {
    let ResponseRequest {
        ground: gs,
        occ,
        xop,
        purpose,
    } = *req;
    // the arithmetic, the plan's (the device policy's: as set, or auto by the CPU's rule with
    // the stored operator preferred to single precision, D37's costing on FP64-poor cards), and
    // the engine's recipe of it
    let policy = device_policy(calc)?;
    let arith = rp.device.ok_or_else(|| {
        LinusError::Internal("a device response J/K without the device's plan".into())
    })?;
    let recipe = ResponseRecipe::new(policy, &arith, field);
    // the perturbations of the products: a Hessian's 3N at first, a polarizability's three
    // field directions
    let expected_nd = Some(match purpose {
        ResponseUse::Hessian => 3 * calc.mol.natom(),
        ResponseUse::Polarizability => 3,
    });
    let timer = match purpose {
        ResponseUse::Hessian => "hess_cphf_setup",
        ResponseUse::Polarizability => "pol_cphf_setup",
    };
    let op = linus_base::terms::exchange_parts(xop).1;
    // host memory for a streamed tensor, and for the stored operator's chunks that do not fit
    // the device; the device's reserves
    let memory = ResponseMemory {
        tensor_host: gs.setup.memory.df_bytes(),
        operator_host: calc.plan.memory().response_operator_host_bytes(),
        device: device_memory(calc)?,
    };
    let built = timers.time(timer, || {
        GpuResponseJk::new(
            gpu.clone(),
            &calc.basis,
            aux,
            occ,
            op,
            &recipe,
            memory,
            expected_nd,
            (calc.backend.responses().clone(), purpose.name()),
            calc.device_session().and_then(|s| s.df()).as_deref(),
        )
    });
    let (jk, emu, auto) = match setup(Family::ResponseJk, built)? {
        Setup::Ready(x) => x,
        Setup::Declined(why) => {
            calc.note_emulation(Family::ResponseJk, None);
            return Ok(Setup::Declined(why));
        }
    };
    // FP32 exchange: as set, or auto's choice
    let single = arith.mode == ResponseMode::Single || auto.as_ref().is_some_and(|a| a.single);
    let label = if single {
        "fp32"
    } else if emu.is_some() {
        "emulated"
    } else {
        "fp64"
    };
    calc.note_emulation(Family::ResponseJk, emu);
    let reason = auto
        .as_ref()
        .map_or_else(|| arith.why.to_string(), |a| a.reason.clone());
    calc.backend
        .responses()
        .start(purpose.name(), Some(label), reason);
    if let Some(e) = auto.and_then(|a| a.estimate) {
        calc.backend.responses().set_estimate(e);
    }
    Ok(Setup::Ready(ResponseSetup {
        jk: Box::new(jk),
        plan: None,
        single_precision: single,
        kernel: crate::two_electron::KernelUse {
            purpose,
            single: recipe.single_kernel,
        },
    }))
}

/// The density-fitted two-electron terms of a Hessian on the device (family `hessian_jk`): F^x
/// C_s, the RI cross terms and the second-derivative integrals on the device
/// (`linus_gpu::hess::df`), the two-center second derivatives on the host, for the occupied
/// orbitals of the setup; added to the Hessian as the CPU's `df_hessian_2e` is. Timed as
/// `hess_2e` (its setup also as `hess_2e_setup`, its phases as `hess_2e_*`).
pub(crate) struct GpuHessianJk<'a> {
    calc: &'a CalcContext,
    gpu: Arc<Gpu>,
    eng: linus_gpu::hess::df::GpuDfHess,
    aux: Arc<BasisSet>,
    /// The channels' weights (closed shell \[2\], open shell \[1, 1\]).
    f: Vec<f64>,
    /// The calls: (a_x, Coulomb terms, operator of the exchange integrals), as `df_hessian_2e`.
    calls: Vec<(f64, bool, Operator)>,
    ncols: Vec<usize>,
}

impl<'a> GpuHessianJk<'a> {
    /// The engine for the occupied orbitals `occ` per channel, the auxiliary basis `aux` and the
    /// method's exact exchange `xop` as [`linus_jk::df::hessian::df_hessian_2e`] takes
    /// it (a global fraction, or a range-separated operator in Psi4's wK scheme: the Coulomb call
    /// and the two polarization calls), its workspace granted and its exchange in the policy's
    /// emulation; declined (the CPU's `df_hessian_2e` then runs) when the setup and its workspace
    /// do not fit.
    pub(crate) fn new(
        calc: &'a CalcContext,
        gpu: Arc<Gpu>,
        occ: &[MatRef<'_, f64>],
        aux: Arc<BasisSet>,
        xop: Option<Operator>,
        timers: &mut Timers,
    ) -> Result<Setup<Self>, LinusError> {
        use linus_gpu::hess::df::{GpuDfHess, GpuDfHessOptions};
        let ncols: Vec<usize> = occ.iter().map(|c| c.ncols()).collect();
        let nch = ncols.len();
        let f: Vec<f64> = if nch == 1 { vec![2.0] } else { vec![1.0; nch] };
        let calls: Vec<(f64, bool, Operator)> = match xop.map(|op| (op, op.global_fraction())) {
            Some((x, None)) => {
                let mut v = vec![(0.0, true, Operator::Coulomb)];
                for (ax, op) in [
                    (
                        0.25,
                        Operator::Attenuated {
                            coulomb: 1.0 + x.coulomb(),
                            long_range: x.long_range(),
                            omega: x.omega(),
                        },
                    ),
                    (
                        -0.25,
                        Operator::Attenuated {
                            coulomb: 1.0 - x.coulomb(),
                            long_range: -x.long_range(),
                            omega: x.omega(),
                        },
                    ),
                ] {
                    if op.coulomb() == 0.0 && !op.has_erf() {
                        continue;
                    }
                    v.push((ax, false, op));
                }
                v
            }
            global => vec![(
                global.and_then(|g| g.1).unwrap_or(0.0),
                true,
                Operator::Coulomb,
            )],
        };
        let t0 = std::time::Instant::now();
        let ot: usize = ncols.iter().sum();
        // the SCF's layout, integrals, metric and derivative records (crate::backend::gpu::session)
        let shared = calc.device_session().and_then(|s| s.df());
        let built = GpuDfHess::with_shared(
            &gpu,
            &calc.basis,
            &aux,
            &GpuDfHessOptions::default(),
            shared.as_deref(),
        )
        .and_then(|mut h| {
            let mut need = 0;
            for (ax, _, op) in &calls {
                h.prepare_operator(&gpu, op)?;
                need = need.max(h.workspace_bytes(&ncols, *ax != 0.0 && ot > 0, op.has_erf()));
            }
            h.grant_workspace(&gpu, need)?;
            Ok(h)
        });
        let t_setup = t0.elapsed().as_secs_f64();
        timers.add("hess_2e_setup", t_setup);
        timers.add("hess_2e", t_setup);
        let mut eng = match setup(Family::HessianJk, built)? {
            Setup::Ready(e) => e,
            Setup::Declined(why) => {
                calc.note_emulation(Family::HessianJk, None);
                return Ok(Setup::Declined(why));
            }
        };
        // the policy's emulation (precision.fp64 emulated): the exchange part of F^x C (its
        // largest GEMM) in the Ozaki scheme, native FP64 when that does not set up (with the reason)
        let chain = device_policy(calc)?.hess_k;
        let emu = (chain != NATIVE).then(|| {
            let used = apply_chain(chain, |a| {
                eng.set_emulation(
                    &gpu,
                    match a {
                        Arith::Ozaki { slices } => Some(slices as usize),
                        _ => None,
                    },
                )
            });
            emulation_used(Family::HessianJk, &used, Some(false), None, None)
        });
        calc.note_emulation(Family::HessianJk, emu);
        Ok(Setup::Ready(GpuHessianJk {
            calc,
            gpu,
            eng,
            aux,
            f,
            calls,
            ncols,
        }))
    }

    /// The terms of the occupied orbitals `occ` (the setup's shapes), summed over the calls.
    pub(crate) fn compute(
        &mut self,
        occ: &[MatRef<'_, f64>],
        timers: &mut Timers,
    ) -> Result<linus_jk::df::hessian::DfHess2e, LinusError> {
        use linus_jk::df::hessian::DfHess2e;
        let basis = &self.calc.basis;
        let n = basis.nbf;
        let ncols = &self.ncols;
        let cm: Vec<Vec<f64>> = occ.iter().map(|c| to_cm(*c)).collect();
        let o: Vec<(&[f64], usize)> = cm
            .iter()
            .zip(ncols)
            .map(|(c, &k)| (c.as_slice(), k))
            .collect();
        let n3 = 3 * basis.atom_shells.len().max(self.aux.atom_shells.len());
        let mut out = DfHess2e {
            fxo: (0..n3)
                .map(|_| ncols.iter().map(|&k| Mat::zeros(n, k)).collect())
                .collect(),
            hess: vec![0.0; n3 * n3],
        };
        for &(ax, want_j, op) in &self.calls {
            let t = std::time::Instant::now();
            let parts = self
                .eng
                .compute(&self.gpu, &o, &self.f, ax, want_j, &op)
                .map_err(|e| device_error(Family::HessianJk, e))?;
            timers.add("hess_2e", t.elapsed().as_secs_f64());
            for (name, v) in self.eng.last.phases() {
                timers.add(&format!("hess_2e_{name}"), v);
            }
            out.hess
                .iter_mut()
                .zip(&parts.hess)
                .for_each(|(u, v)| *u += v);
            for (s, fs) in parts.fxo.iter().enumerate() {
                let k = ncols[s];
                for (xi, fx) in out.fxo.iter_mut().enumerate() {
                    let m = from_cm(n, k, &fs[xi * n * k..(xi + 1) * n * k]);
                    fx[s] += &m;
                }
            }
        }
        self.calc
            .note_grant(Family::HessianJk, &self.eng.workspace_violations());
        Ok(out)
    }
}

impl HessianTerm for GpuHessianJk<'_> {
    fn add_hessian(
        &mut self,
        inputs: &TermInputs<'_>,
        out: &mut ExplicitHessian,
        timers: &mut Timers,
    ) -> Result<(), LinusError> {
        let p2e = self.compute(&inputs.c_occ, timers)?;
        out.h.iter_mut().zip(&p2e.hess).for_each(|(a, b)| *a += b);
        for (p, gp) in p2e.fxo.iter().enumerate() {
            for (fs, gs) in out.fxo.iter_mut().zip(gp) {
                fs[p] += gs;
            }
        }
        linus_base::memory::mem_mark(timers, "explicit_ri");
        Ok(())
    }
}
/// The two-electron gradient on the device (family `gradient_jk`): its engine set up in the
/// gradient's device lane, beside the host's one-electron pass (the setup waits for the SCF's
/// derivative records and uploads them there: `grad_2e_setup_wait`, `grad_2e_setup`), then its
/// pass; declined (the CPU's `df_gradient`, after the overlap) when the setup and its workspace do
/// not fit. Timed as `grad_2e`.
pub(crate) struct GpuJkGradient<'a> {
    calc: &'a CalcContext,
    gpu: Arc<Gpu>,
    xop: Option<Operator>,
}

impl<'a> GpuJkGradient<'a> {
    pub(crate) fn new(calc: &'a CalcContext, gpu: Arc<Gpu>, xop: Option<Operator>) -> Self {
        GpuJkGradient { calc, gpu, xop }
    }
}

impl GradientTerm for GpuJkGradient<'_> {
    fn lane(&self) -> Lane {
        Lane::Device
    }

    fn gradient(
        &mut self,
        inputs: &GradientInputs<'_>,
        timers: &mut Timers,
    ) -> Result<Setup<Vec<[f64; 3]>>, LinusError> {
        let calc = self.calc;
        let crate::driver::JkMethod::Df { aux } = calc.plan.jk() else {
            return Ok(Setup::Declined(format!(
                "jk {}: no device engines",
                calc.plan.jk().name()
            )));
        };
        let auxb = calc.aux_set(aux, inputs.ground.purpose())?;
        let r = &inputs.ground.scf;
        let nch = if r.restricted { 1 } else { 2 };
        let ncols = &[r.nalpha, r.nbeta][..nch];
        let xop = crate::two_electron::gradient_operator(self.xop);
        let g = match gradient_engine(calc, &self.gpu, &auxb, ncols, xop, timers)? {
            Setup::Ready(g) => g,
            Setup::Declined(why) => return Ok(Setup::Declined(why)),
        };
        let occ = &[inputs.ca, inputs.cb][..nch];
        gradient_with(calc, &self.gpu, g, occ, r.restricted, xop, timers).map(Setup::Ready)
    }
}
/// The device engine of the two-electron gradient for the auxiliary basis `aux`, `ncols` occupied
/// orbitals per channel and the exchange operator `xop` (coefficients included: a_x/r for HF and
/// global hybrids, a/r + b erf(omega r)/r for range-separated ones, the wK scheme of the CPU's
/// `df_gradient`), declined when its setup and workspace do not fit; timed as `grad_2e_setup`
/// (and `grad_2e`). Its derivative planes are the device policy's
/// (`DevicePolicy::df_grad_planes`); the kernels it takes, and why, go to the record
/// (`resolved.record.backend.gradient_jk_kernels`).
fn gradient_engine(
    calc: &CalcContext,
    gpu: &Arc<Gpu>,
    aux: &BasisSet,
    ncols: &[usize],
    xop: Operator,
    timers: &mut Timers,
) -> Result<Setup<linus_gpu::grad::df::GpuDfGrad>, LinusError> {
    use linus_gpu::grad::df::{DerivPlanes, GpuDfGrad, GpuDfGradOptions};
    let policy = device_policy(calc)?;
    let opts = GpuDfGradOptions {
        planes: if policy.df_grad_planes {
            DerivPlanes::StoredOrFused
        } else {
            DerivPlanes::Fused
        },
        ..GpuDfGradOptions::default()
    };
    let t0 = std::time::Instant::now();
    // the SCF's layout, integrals, metric and derivative records (crate::backend::gpu::session),
    // once the records packed during the SCF are done
    let shared = calc.device_session().and_then(|s| s.df());
    timers.add("grad_2e_setup_wait", t0.elapsed().as_secs_f64());
    let built = GpuDfGrad::with_shared(gpu, &calc.basis, aux, &opts, shared.as_deref()).and_then(
        |mut g| {
            // the erf integrals of a range-separated operator, then the workspace of the gradient
            // pass on top of the setup's allocations
            g.prepare_operator(gpu, &xop)?;
            g.grant_workspace(gpu, ncols, &xop)?;
            Ok(g)
        },
    );
    let t_setup = t0.elapsed().as_secs_f64();
    timers.add("grad_2e_setup", t_setup);
    timers.add("grad_2e", t_setup);
    if let Ok(g) = &built {
        let k = g.kernels(ncols, &xop);
        let reason = match (k.declined, k.kernels) {
            (Some(why), _) => why,
            (None, "hermite_j") => "no exchange: the rank-1 Gamma contracted in Hermite space (no \
                                   integral block or derivative plane)"
                .to_string(),
            (None, "stored_planes") => "the device policy on FP64-rich cards: the derivative \
                                       planes stored and contracted (the fused kernels' atomics \
                                       bind there)"
                .to_string(),
            (None, _) => "the device policy: Gamma contracted inside the derivative kernels".into(),
        };
        calc.record
            .decide(crate::driver::record::Decision::GradientJkKernels {
                kernels: k.kernels,
                reason,
            });
        let s = &g.setup_times;
        for (name, v) in s.phases() {
            timers.add(&format!("grad_2e_setup_{name}"), v);
        }
        timers.resource(
            "grad_2e_setup_upload_gib",
            s.upload_bytes as f64 / 1073741824.0,
        );
    }
    setup(Family::GradientJk, built)
}

/// The two-electron gradient of the occupied orbitals `occ` (one channel for a closed shell with
/// `restricted`, else alpha and beta) with the device engine `g` ([`gradient_engine`], `xop` as
/// there).
fn gradient_with(
    calc: &CalcContext,
    gpu: &Arc<Gpu>,
    mut g: linus_gpu::grad::df::GpuDfGrad,
    occ: &[&Mat<f64>],
    restricted: bool,
    xop: Operator,
    timers: &mut Timers,
) -> Result<Vec<[f64; 3]>, LinusError> {
    let ncols: Vec<usize> = occ.iter().map(|c| c.ncols()).collect();
    let cm: Vec<Vec<f64>> = occ.iter().map(|c| to_cm(c.as_ref())).collect();
    let o: Vec<(&[f64], usize)> = cm
        .iter()
        .zip(&ncols)
        .map(|(c, &n)| (c.as_slice(), n))
        .collect();
    let out = timers.time("grad_2e", || g.compute(gpu, &o, restricted, xop));
    calc.note_grant(Family::GradientJk, &g.workspace_violations());
    Ok(out
        .map_err(|e| device_error(Family::GradientJk, e))?
        .gradient)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::driver::CalcOptions;
    use crate::driver::Session;
    use crate::Structure;
    use linus_base::memory::MemoryLimit;
    use linus_ints::{DerivOrder, PairOptions, PairSet};
    use serde_json::json;

    const MOL: &str =
        "4\n\nO 0.10 -0.05 0.12\nH 0.85 0.52 -0.03\nH -0.68 0.55 0.05\nC 0.20 -1.30 0.60\n";

    /// One fallback chain for every exchange: the first tier that sets up runs; the last is
    /// taken whatever happens; the declined tiers' errors make the reason.
    #[test]
    fn fallback_chains() {
        use crate::driver::policy::EMULATION;
        let used = apply_chain(EMULATION, |a| match a {
            Arith::Ozaki { .. } => Err("no planes"),
            _ => Ok(()),
        });
        assert_eq!(used.arith, CUBLAS);
        assert_eq!(used.reason().as_deref(), Some("Ozaki exchange: no planes"));
        let used = apply_chain(OPERATOR_EMULATION, |_| Err::<(), _>("x"));
        assert_eq!(used.arith, Arith::Fp64);
        assert_eq!(used.reason().as_deref(), Some("cuBLAS emulation: x"));
        assert_eq!(apply_chain(NATIVE, |_| Ok::<(), String>(())).reason(), None);
    }

    /// The device response's recipe of the plan's arithmetic: auto on an FP64-poor card weighs
    /// the costing (with the field solve when one follows) and emulates the FP64 exchange; fp32
    /// on puts single-precision responses on the FP32 operator; double keeps the CPKS kernel in
    /// FP64.
    #[test]
    fn response_recipes() {
        use crate::driver::options::{Fp64Mode, ResponsePrecision as P};
        use crate::driver::policy::{CardClass, CardProps, EMULATION};
        let pol = |fp64, fp32, rich: bool| {
            let int8 = Some(if rich { 30.0 } else { 500.0 });
            DevicePolicy::new(
                fp64,
                fp32,
                &CardProps::of_class(CardClass::new(rich, int8), (12, 0)),
            )
        };
        let rich = pol(Fp64Mode::Auto, Fp32Mode::Auto, true);
        let poor = pol(Fp64Mode::Auto, Fp32Mode::Auto, false);
        let mixed = pol(Fp64Mode::Native, Fp32Mode::On, false);
        let h = ResponseUse::Hessian;
        let recipe = |p: &DevicePolicy, u, pr, field| {
            ResponseRecipe::new(p, &p.response(u, pr, 1e11, None), field)
        };
        let r = recipe(&rich, h, P::Auto, false);
        assert!(r.costing.is_none() && !r.single_kernel && r.chain == NATIVE);
        let r = recipe(&poor, h, P::Auto, true);
        assert!(r.chain == EMULATION && r.single_kernel && !r.fp32_operator);
        assert!(r.costing.is_some_and(|c| c.field && c.setup_emulated));
        let r = recipe(&mixed, h, P::Auto, false);
        assert!(r.fp32_operator && r.k == Arith::Fp32 && r.chain == NATIVE && r.single_kernel);
        assert!(r.costing.is_none());
        let r = recipe(&poor, ResponseUse::Polarizability, P::Fp32, false);
        assert!(r.chain == EMULATION && !r.single_kernel && r.costing.is_none());
    }

    /// The fit coefficients a Coulomb hook receives from the device's J build, through the
    /// device's triangular solve and through the host's (the fallback when the device's metric
    /// factor does not fit), agree to round-off amplified by the metric's conditioning. Skips
    /// without a device unless LINUS_GPU_REQUIRED=1.
    #[test]
    fn coulomb_fit_on_the_host_matches_the_device() {
        struct Capture(Vec<Vec<f64>>);
        impl linus_base::terms::CoulombHook for Capture {
            fn extra_coefficients(
                &mut self,
                c: &[Vec<f64>],
                _solve: &dyn Fn(&mut [f64]),
            ) -> Vec<Vec<f64>> {
                self.0 = c.to_vec();
                c.iter().map(|x| vec![0.0; x.len()]).collect()
            }
        }
        let gpu = match Gpu::new(0) {
            Ok(g) => g,
            Err(e) if std::env::var("LINUS_GPU_REQUIRED").as_deref() != Ok("1") => {
                eprintln!("no usable GPU, skipping: {e}");
                return;
            }
            Err(e) => panic!("{e}"),
        };
        let mol = Structure::from_xyz_str(MOL, 0, 1).unwrap();
        let o = CalcOptions::from_json(json!({"method": "hf", "basis": "def2-svp", "jk": "df"}))
            .unwrap();
        let c = Session::new(mol, o).unwrap();
        let crate::driver::JkMethod::Df { aux } = c.cx.plan.jk().clone() else {
            unreachable!()
        };
        let auxb = c.aux_set(&aux, c.scf_purpose()).unwrap();
        let n = c.cx.basis.nbf;
        let mut eng = GpuDfJk::new(&gpu, &c.cx.basis, &auxb, &GpuDfJkOptions::default()).unwrap();
        eng.set_keep_coulomb_fit(true);
        // a symmetric density
        let d: Vec<f64> = (0..n * n)
            .map(|k| {
                let (i, j) = (k % n, k / n);
                0.1 / (1.0 + (i as f64 - j as f64).abs())
            })
            .collect();
        eng.compute(
            &gpu,
            &linus_gpu::df::GpuJkRequest {
                dens: &[&d],
                factors: &[],
                want_j: true,
                want_k: false,
            },
        )
        .unwrap();
        let mut got = Vec::new();
        for host in [false, true] {
            let mut fit = CoulombFit::with_solve(&gpu, &auxb, host).unwrap();
            assert_eq!(matches!(fit.solve, FitSolve::Host(_)), host);
            let mut cap = Capture(Vec::new());
            fit.feed(&gpu, &eng, &mut cap).unwrap();
            got.push(cap.0.concat());
        }
        let big = got[0].iter().fold(0.0f64, |m, x| m.max(x.abs()));
        let diff = got[0]
            .iter()
            .zip(&got[1])
            .fold(0.0f64, |m, (a, b)| m.max((a - b).abs()));
        // the metric's conditioning amplifies the two factorizations' round-off (measured:
        // 6e-11 relative on the RTX PRO 6000)
        assert!(
            big > 0.0 && diff <= 1e-9 * big,
            "host vs device fit: {diff:e} of {big:e}"
        );
    }

    /// The device's DF Hessian terms ([`GpuHessianJk`]: F^x C_s and the explicit part with the
    /// second-derivative integrals) equal the CPU's `df_hessian_2e` to round-off,
    /// for HF, a global hybrid, a pure functional and a range-separated hybrid (wK), closed and
    /// open shells; repeated device runs are bitwise identical. Skips without a device unless
    /// LINUS_GPU_REQUIRED=1.
    #[test]
    fn hessian_terms_match_the_cpu() {
        for (charge, mult, method) in [
            (0, 1, "b3lyp"),
            (1, 2, "b3lyp"),
            (0, 1, "hf"),
            (0, 1, "pbe"),
            (0, 1, "wb97x"),
            (1, 2, "wb97x"),
        ] {
            let mol = Structure::from_xyz_str(MOL, charge, mult).unwrap();
            let o = CalcOptions::from_json(json!({
                "method": method, "basis": "def2-svp", "jk": "df", "threads": 4, "backend": "gpu"
            }))
            .unwrap();
            let mut c = match Session::new(mol, o) {
                Ok(c) => c,
                Err(LinusError::Backend { message, .. })
                    if std::env::var("LINUS_GPU_REQUIRED").as_deref() != Ok("1") =>
                {
                    eprintln!("no usable GPU, skipping: {message}");
                    return;
                }
                Err(e) => panic!("{e}"),
            };
            c.energy().unwrap();
            let r = c.last().unwrap().clone();
            let crate::driver::JkMethod::Df { aux } = c.cx.plan.jk().clone() else {
                unreachable!()
            };
            let auxb = c.aux_set(&aux, c.scf_purpose()).unwrap();
            let pairs = PairSet::build(&c.cx.basis, PairOptions::with_order(DerivOrder::Two));
            let (occ, f): (Vec<MatRef<'_, f64>>, Vec<f64>) = if r.restricted {
                (vec![r.c_a.subcols(0, r.nalpha)], vec![2.0])
            } else {
                (
                    vec![r.c_a.subcols(0, r.nalpha), r.c_b.subcols(0, r.nbeta)],
                    vec![1.0, 1.0],
                )
            };
            let xop = c.cx.method.exchange;
            let par = Par::rayon(4);
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(4)
                .build()
                .unwrap();
            pool.install(|| {
                let mut t = Timers::default();
                let cpu = linus_jk::df::hessian::df_hessian_2e(
                    &c.cx.basis,
                    &pairs,
                    &auxb,
                    &linus_jk::df::CoulombMetric::new(&auxb, faer::Par::Seq).unwrap(),
                    &occ,
                    &f,
                    xop,
                    MemoryLimit::from_gib(None),
                    par,
                    &mut t,
                );
                let gpu = c.cx.device_for(Family::HessianJk).unwrap().clone();
                let dev = |t: &mut Timers| {
                    let aux = c.cx.aux_set(&aux, c.scf_purpose()).unwrap();
                    match GpuHessianJk::new(&c.cx, gpu.clone(), &occ, aux, xop, t).unwrap() {
                        Setup::Ready(mut h) => h.compute(&occ, t).unwrap(),
                        Setup::Declined(why) => panic!("the device's terms: {why}"),
                    }
                };
                let g = dev(&mut t);
                let dh = cpu
                    .hess
                    .iter()
                    .zip(&g.hess)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0, f64::max);
                let hmax = cpu.hess.iter().fold(0.0f64, |m, a| m.max(a.abs()));
                let mut df: f64 = 0.0;
                let mut fmax: f64 = 0.0;
                for (a, b) in cpu.fxo.iter().zip(&g.fxo) {
                    for (u, v) in a.iter().zip(b) {
                        df = df.max(linus_base::linalg::max_abs((u - v).as_ref()));
                        fmax = fmax.max(linus_base::linalg::max_abs(u.as_ref()));
                    }
                }
                println!(
                    "{method} {charge} {mult}: |dH| {dh:.2e} (|H| {hmax:.2e}), |dF^x C| {df:.2e} \
                     (|F^x C| {fmax:.2e})"
                );
                assert!(
                    dh < 1e-10 && df < 1e-10,
                    "{method} {charge} {mult}: dH {dh:e} dF {df:e}"
                );
                // bitwise run to run
                let g2 = dev(&mut t);
                assert!(
                    g.hess
                        .iter()
                        .zip(&g2.hess)
                        .all(|(a, b)| a.to_bits() == b.to_bits()),
                    "{method}: device Hessian terms differ run to run"
                );
                let bits = |m: &Mat<f64>| -> Vec<u64> {
                    to_cm(m.as_ref()).iter().map(|x| x.to_bits()).collect()
                };
                for (a, b) in g.fxo.iter().zip(&g2.fxo) {
                    for (u, v) in a.iter().zip(b) {
                        assert_eq!(bits(u), bits(v), "{method}: F^x C differs run to run");
                    }
                }
            });
        }
    }
}
