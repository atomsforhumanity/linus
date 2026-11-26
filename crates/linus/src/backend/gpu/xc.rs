//! The XC terms on the GPU (feature `gpu`), on the grid of the calculation's CPU integrator
//! (built by the one factory, [`XcIntegrator::build`]):
//! - [`GpuXcPotential`]: the SCF's [`XcPotential`] over `linus_gpu::xc::GpuXc`, the density
//!   formed on the device from the occupied orbitals (a guess density without orbitals is split
//!   into its eigenvectors, with the eigenvalues as occupations); VV10 on the device too
//!   (`linus_gpu::xc::vv10::GpuVv10` on the integrator's NLC grid), or the integrator's own
//!   term on the CPU when its setup falls back;
//! - [`GpuResponseKernel`]: the CPKS kernel ([`ResponseKernel`]) over
//!   `linus_gpu::response::xc::GpuXcKernel`, the functional's second derivatives evaluated
//!   once at its setup for the occupied orbitals of the response;
//! - [`gradient_engine`] and [`gradient_with`]: the XC gradient over
//!   `linus_gpu::grad::xc::GpuXcGrad` with the grid
//!   response (Becke weight derivatives on the device), VV10's over
//!   `linus_gpu::xc::vv10::GpuVv10Grad` on the NLC grid (or the CPU's when it falls back);
//! - [`hessian_terms`]: the Hessian's explicit XC terms (the integrator's `hessian_terms` and
//!   `grid_terms` in one pass) over `linus_gpu::hess::xc::GpuXcHess`, LDA and GGA, returned as
//!   V^x C_o and dV^y C_o (within ~5e-13 of the CPU's).
//!
//! Both use the CPU integrator's grid, functional and function screening (`func_thresh`) and
//! agree with it to round-off (V_xc ~1e-13 relative, the kernel ~1e-12; the device screens per
//! block of points, the CPU per batch); they are bitwise reproducible run to run on one
//! device.

use crate::backend::gpu::device_error;
use crate::backend::gpu::df::{device_memory, device_policy};
use crate::backend::gpu::setup;
use crate::backend::scf::ScfXc;
use crate::backend::Setup;
use crate::derivatives::terms::{GradientInputs, GradientTerm, Lane};
use crate::driver::backend::Family;
use crate::driver::options::Fp32Mode;
use crate::driver::policy::{DevicePolicy, DevicePrecision, OZAKI};
use crate::driver::CalcContext;
use crate::two_electron::{KernelUse, ResponseUse};
use faer::{Mat, MatRef, Par};
use linus_base::error::LinusError;
use linus_base::linalg::{mat_from_colmajor as from_cm, mat_to_colmajor as to_cm};
use linus_base::terms::ResponseKernel;
use linus_base::terms::{XcDensity, XcPotential};
use linus_core::timing::Timers;
use linus_core::Arith;
use linus_dft::engine::{HessianInput, XcHessianEngine};
use linus_dft::vv10::Vv10Stats;
use linus_dft::vv10_resp::{Vv10Ground, Vv10PairSweep};
use linus_dft::{XcHessianTerms, XcIntegrator};
use linus_gpu::df::Factor;
use linus_gpu::engine::Phases;
use linus_gpu::response::xc::{GpuXcKernel, GpuXcKernelOptions};
use linus_gpu::xc::vv10::{GpuVv10, GpuVv10Grad, GpuVv10GradOptions, GpuVv10Options, Vv10Math};
use linus_gpu::xc::{GpuXc, GpuXcOptions, ProductEngine, XcChannel, XcMath, XcScreen};
use linus_gpu::Gpu;
use linus_jk::df::factor_symmetric;
use std::sync::{Arc, Mutex};

/// Grid points per device block of the SCF's XC: the collocation's blocks; the products are
/// per grid batch.
const SCF_BLOCK_POINTS: usize = 2048;
/// Grid points per device block of the CPKS kernel (512 to 2048 points cost the same, 256 more).
const KERNEL_BLOCK_POINTS: usize = 512;
/// Grid points per FP64 block of the electric-field CPHF's kernel, three perturbations per call
/// (B3LYP/def2-SVP on an RTX PRO 6000, ms per call with the ground state kept:
/// la18 193 on 512-point blocks, 55 on 2048 with P = F A of every component, 52 on 4096;
/// u1an39 756, 494, 508).
const FIELD_BLOCK_POINTS: usize = 2048;

/// The XC engines' product arithmetic of a family's [`Arith`] (the device policy's): FP32
/// products per grid batch (their sums over at most a batch's points or functions, the batches
/// added in FP64), else FP64. The SCF's potential stays FP64 under every setting: in FP32
/// (densities in FP64) its energies were as accurate, but the SCF needed 1-6 more iterations on
/// TZVP-size molecules, which cost more than the cheaper XC builds saved.
fn xc_math(a: Arith) -> XcMath {
    match a {
        Arith::Fp32 => XcMath::Fp32,
        _ => XcMath::Fp64,
    }
}

/// The semilocal CPKS kernel's arithmetic, the device policy's decision for the device's kernel
/// (the CPU's kernel computes in FP64).
impl DevicePolicy {
    /// The CPKS kernel's arithmetic for a response that runs single precision or not
    /// (`response_single`: the device response's arithmetic other than double, or the CPU's
    /// response J/K in single precision): FP32 products per grid batch with `precision.fp32: on`
    /// or a single-precision response (their frequency errors, at most 5.8e-5 cm^-1, below the
    /// single-precision exchange's); else FP64, its two large GEMMs per block in the Ozaki
    /// scheme with emulated FP64 (1.5-1.6x per product, within 5e-14 of FP64).
    pub(crate) fn cpks_xc(&self, response_single: bool) -> Arith {
        match self.precision {
            _ if self.fp32 == Fp32Mode::On || response_single => Arith::Fp32,
            DevicePrecision::Emulated => OZAKI,
            DevicePrecision::Double => Arith::Fp64,
        }
    }
}

/// The CPKS kernel's screening, products and Ozaki slices for its arithmetic `a`
/// ([`DevicePolicy::cpks_xc`]): FP32 products per grid batch; else FP64 on 512-point blocks
/// screened by the shell bound (its per-batch FP64 products are smaller GEMMs that run slower
/// than the blocks' on cuBLAS), their two large GEMMs per block (P = F A, dV = F^T Z) in the
/// Ozaki scheme for [`Arith::Ozaki`].
fn kernel_plan(a: Arith) -> (XcScreen, XcMath, Option<usize>) {
    match a {
        Arith::Fp32 => (XcScreen::Batch, XcMath::Fp32, None),
        Arith::Ozaki { slices } => (XcScreen::Block, XcMath::Fp64, Some(slices as usize)),
        _ => (XcScreen::Block, XcMath::Fp64, None),
    }
}

/// The engine of the XC engines' FP64 products on the policy's card.
fn products(p: &DevicePolicy) -> ProductEngine {
    if p.xc_grouped {
        ProductEngine::Grouped
    } else {
        ProductEngine::Vgemm
    }
}

/// VV10's pair-sum arithmetic of an [`Arith`] (the device policy's `vv10_scf` or `vv10_grad`).
fn vv10_math(a: Arith) -> Vv10Math {
    match a {
        Arith::Fp64 => Vv10Math::Fp64,
        Arith::Fp32 => Vv10Math::Fp32,
        _ => Vv10Math::DoubleFloat,
    }
}

/// The grid builder's partition on the session's device (linus-grid's `DevicePartition` over
/// linus-gpu's `GpuPartition`), recording where each grid's weights were computed
/// (`resolved.record.backend.grid_partition`): out of device memory, or in a case the kernels
/// do not cover, the host computes them (recorded with the reason); any other device error is
/// the calculation's.
pub(crate) struct GridPartition<'a> {
    pub(crate) backend: &'a super::GpuBackend,
    /// The device bytes the partition may take (the memory plan's `grid_partition`).
    pub(crate) allowance: Option<usize>,
}

impl linus_grid::DevicePartition for GridPartition<'_> {
    fn weights(
        &self,
        tables: &linus_grid::PairTables,
        x: &[f64],
        y: &[f64],
        z: &[f64],
        owner: &[u32],
        vol: &[f64],
    ) -> Result<Option<Vec<f64>>, Box<dyn std::error::Error + Send + Sync>> {
        // within the memory plan's share for it beside the DF-J/K's setup
        let part = linus_gpu::xc::partition::GpuPartition {
            gpu: &self.backend.gpu,
            allowance: self.allowance,
        };
        match part.weights(tables, x, y, z, owner, vol) {
            Ok(w) => {
                self.backend.note_grid_partition(
                    "gpu",
                    "an unscreened partition (Becke's): the device's weights, each point's cell \
                     functions below 2^-100 of their sum left out"
                        .into(),
                );
                Ok(Some(w))
            }
            Err(e) if e.falls_back_at_setup() => {
                self.backend.note_grid_partition(
                    "cpu",
                    format!("the device declined ({e}): the host's weights"),
                );
                Ok(None)
            }
            Err(e) => Err(Box::new(device_error(Family::Xc, e))),
        }
    }
}

/// The device XC engine of `name` on the grid and screening of `xc`, `block_points` per block.
fn engine(
    gpu: &Gpu,
    xc: &XcIntegrator,
    name: &str,
    block_points: usize,
    (screen, math, products): (XcScreen, XcMath, ProductEngine),
) -> Result<GpuXc, linus_gpu::GpuError> {
    GpuXc::new(
        gpu,
        &xc.basis,
        &xc.grid,
        name,
        &GpuXcOptions {
            func_thresh: xc.func_thresh,
            block_points,
            screen,
            math,
            products,
            ..GpuXcOptions::default()
        },
    )
}

/// The SCF's XC term on the device: the semilocal part and, for -V functionals, VV10 (on the
/// device, or the CPU integrator's when its device setup fell back).
pub(crate) struct GpuXcPotential {
    gpu: Arc<Gpu>,
    eng: GpuXc,
    /// The CPU integrator (shared with the geometry's other phases): the grid the engine was
    /// built on, and its VV10 (the fallback).
    cpu: Arc<XcIntegrator>,
    /// VV10 on the device.
    nl: Option<GpuVv10>,
}

/// VV10's statistics in the CPU's form ([`Vv10Stats`]) from the device engine's.
pub(crate) fn vv10_stats_of(nl: &GpuVv10) -> Vv10Stats {
    let t = nl.totals();
    Vv10Stats {
        last_energy: t.last_energy,
        active_points: t.active,
        grid_points: nl.stats().npts,
        t_density: t.density,
        t_pairs: t.pairs,
        t_vmat: t.potential,
        evaluations: t.evaluations,
        ..Vv10Stats::default()
    }
}

impl GpuXcPotential {
    /// VV10's statistics of the evaluations so far (-V functionals): the device's, or the CPU
    /// integrator's.
    pub(crate) fn vv10_stats(&self) -> Option<Vv10Stats> {
        match &self.nl {
            Some(nl) => Some(vv10_stats_of(nl)),
            None => self.cpu.vv10().map(|v| v.stats()),
        }
    }

    /// Whether VV10 (if the functional has it) runs on the device.
    pub(crate) fn vv10_on_device(&self) -> bool {
        self.cpu.vv10().is_none() || self.nl.is_some()
    }

    /// The engines and the CPU integrator (for the device SCF loop's Fock terms).
    pub(crate) fn into_parts(self) -> (GpuXc, Option<GpuVv10>, Arc<XcIntegrator>) {
        (self.eng, self.nl, self.cpu)
    }

    /// The potential of engines set up on `gpu` on the grids of `cpu`.
    pub(crate) fn from_parts(
        gpu: Arc<Gpu>,
        eng: GpuXc,
        nl: Option<GpuVv10>,
        cpu: Arc<XcIntegrator>,
    ) -> Self {
        GpuXcPotential { gpu, eng, cpu, nl }
    }
}

/// A channel's orbitals on the host: C (column-major), its columns and occupations.
type Channel = (Vec<f64>, usize, Vec<f64>);

impl XcPotential for GpuXcPotential {
    fn compute(
        &mut self,
        d: &XcDensity<'_>,
        par: Par,
    ) -> Result<(f64, Mat<f64>, Mat<f64>), LinusError> {
        let n = d.da.nrows();
        let nch = if d.restricted { 1 } else { 2 };
        // (C column-major, columns, occupations) per channel: the orbitals given, or (a guess's
        // densities, for an engine that takes orbitals) D = P P^T - N N^T, occupations +1 and
        // -1, by a factorization on the host (O(nbf^3))
        let factored = || -> Vec<Channel> {
            [d.da, d.db][..nch]
                .iter()
                .map(|m| {
                    let (p, q) = factor_symmetric(m.as_ref(), par);
                    let mut c = to_cm(p.as_ref());
                    c.extend(to_cm(q.as_ref()));
                    let mut occ = vec![1.0; p.ncols()];
                    occ.extend(vec![-1.0; q.ncols()]);
                    (c, p.ncols() + q.ncols(), occ)
                })
                .collect()
        };
        let mut chans: Option<Vec<Channel>> = d.orbitals.map(|o| {
            (0..nch)
                .map(|s| (to_cm(o.c[s].as_ref()), o.c[s].ncols(), o.n[s].clone()))
                .collect()
        });
        fn channels(c: &[Channel]) -> Vec<XcChannel<'_>> {
            c.iter()
                .map(|(c, ncol, occ)| XcChannel {
                    c,
                    ncol: *ncol,
                    occ,
                })
                .collect()
        }
        // a guess's densities go to the device as they are (the per-batch path takes them)
        let from_d = match d.orbitals {
            Some(_) => None,
            None => {
                let dm: Vec<Vec<f64>> = [d.da, d.db][..nch]
                    .iter()
                    .map(|m| to_cm(m.as_ref()))
                    .collect();
                let refs: Vec<&[f64]> = dm.iter().map(|m| m.as_slice()).collect();
                self.eng
                    .compute_densities(&self.gpu, &refs, d.restricted)
                    .map_err(|e| device_error(Family::Xc, e))?
            }
        };
        let r = match from_d {
            Some(r) => r,
            None => {
                let c = chans.get_or_insert_with(factored);
                self.eng
                    .compute(&self.gpu, &channels(c), d.restricted)
                    .map_err(|e| device_error(Family::Xc, e))?
            }
        };
        let mut exc = r.exc;
        let mut va = from_cm(n, n, &r.v[0]);
        let mut vb = if d.restricted {
            va.clone()
        } else {
            from_cm(n, n, &r.v[1])
        };
        // VV10: a functional of the total density, its potential to both spins, as in
        // XcIntegrator::potential; on the device, or on the CPU (crate::xc::vv10)
        if let Some(nl) = self.nl.as_mut() {
            let c = chans.get_or_insert_with(factored);
            let (e, v) = nl
                .compute(&self.gpu, &channels(c), d.restricted)
                .map_err(|e| device_error(Family::Vv10, e))?;
            let v = from_cm(n, n, &v);
            exc += e;
            va += &v;
            vb += &v;
        } else if let Some(nl) = self.cpu.vv10() {
            let (e, v) = nl.energy_potential(&(d.da + d.db));
            exc += e;
            va += &v;
            vb += &v;
        }
        Ok((exc, va, vb))
    }

    /// VV10 (on the device or the CPU) keeps its last energy and statistics.
    fn keeps_state(&self) -> bool {
        self.nl.is_some() || self.cpu.vv10().is_some()
    }
}

impl ScfXc for GpuXcPotential {
    fn vv10_stats(&self) -> Option<Vv10Stats> {
        GpuXcPotential::vv10_stats(self)
    }
}

/// The SCF's XC term on the device over the CPU integrator `cpu` (families `xc` and `vv10`;
/// timed as `xc_setup`, with its part `xc_setup_functional`, the functional compiled and checked
/// for the device), VV10 on the device when its setup fits too; declined (the integrator, VV10
/// with it) when the semilocal setup does not fit.
pub(crate) fn scf_potential(
    calc: &CalcContext,
    gpu: &Arc<Gpu>,
    cpu: Arc<XcIntegrator>,
    timers: &mut Timers,
) -> Result<Setup<Box<dyn ScfXc>>, LinusError> {
    let p = device_policy(calc)?;
    let built = timers.time("xc_setup", || {
        engine(
            gpu,
            &cpu,
            calc.plan.method(),
            SCF_BLOCK_POINTS,
            (XcScreen::Batch, xc_math(p.vxc), products(p)),
        )
        .map(|eng| {
            let t = std::time::Instant::now();
            let place = eng.prepare_functional(gpu, spin_channels(calc));
            (eng, place, t.elapsed().as_secs_f64())
        })
    });
    Ok(match setup(Family::Xc, built)? {
        Setup::Ready((eng, place, t_functional)) => {
            calc.note_functional(Family::Xc, place);
            timers.add("xc_setup_functional", t_functional);
            let nl = vv10_engine(calc, &cpu, timers)?;
            Setup::Ready(Box::new(GpuXcPotential {
                gpu: gpu.clone(),
                eng,
                cpu,
                nl,
            }))
        }
        Setup::Declined(why) => {
            // VV10 goes with the integrator's semilocal term
            if cpu.vv10().is_some() {
                calc.note_setup(Family::Vv10, Some(XC_ON_CPU.into()));
            }
            Setup::Declined(why)
        }
    })
}

/// The spin channels of the calculation's determinant (1 restricted, 2 unrestricted), as the
/// SCF's XC calls will see them.
fn spin_channels(calc: &CalcContext) -> usize {
    let closed = calc.mol.nalpha_nbeta().is_ok_and(|(a, b)| a == b);
    if calc.plan.restricted().unwrap_or(closed) {
        1
    } else {
        2
    }
}

/// The reason of a VV10 that runs on the CPU because the semilocal XC term does.
const XC_ON_CPU: &str = "the XC setup fell back to the CPU, and VV10 with it";

/// VV10 of the SCF on the device, on the NLC grid of the CPU integrator `cpu`, when the
/// session puts it there and the setup fits (timed as `vv10_setup`; None: the integrator's
/// own on the CPU, the reason recorded).
fn vv10_engine(
    calc: &CalcContext,
    cpu: &XcIntegrator,
    timers: &mut Timers,
) -> Result<Option<GpuVv10>, LinusError> {
    let Some(nl) = cpu.vv10() else {
        return Ok(None);
    };
    let Some(gpu) = calc.device_for(Family::Vv10) else {
        return Ok(None);
    };
    let math = vv10_math(device_policy(calc)?.vv10_scf);
    let built = timers.time("vv10_setup", || {
        GpuVv10::new(
            gpu,
            &cpu.basis,
            nl.grid(),
            calc.plan.method(),
            &GpuVv10Options {
                func_thresh: nl.eng.func_thresh,
                rho_thresh: nl.rho_thresh,
                math,
                ..GpuVv10Options::default()
            },
        )
    });
    calc.device_setup(Family::Vv10, built)
}

/// The CPKS kernel on the device for the occupied orbitals fixed at its setup.
pub(crate) struct GpuResponseKernel {
    gpu: Arc<Gpu>,
    /// The engine (calls take it mutably: their timing records).
    k: Mutex<GpuXcKernel>,
    n: usize,
    occ: Vec<usize>,
    /// The calls' device times by phase ([`GpuResponseKernel::timers`]).
    t: Mutex<Timers>,
}

impl GpuResponseKernel {
    /// The device times of the calls so far by phase (seconds, from events), as `xc_dev_*`:
    /// the whole call, the factors' upload and the results' download, the collocation, the
    /// ground state Phi = F C, the perturbations' products in and out and the pointwise kernel,
    /// and the number of calls (`xc_dev_calls`).
    pub(crate) fn timers(&self) -> Timers {
        let mut t = self.t.lock().unwrap_or_else(|e| e.into_inner()).clone();
        let kept = self
            .k
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .ground_cache_bytes();
        if kept > 0 {
            t.resource("xc_dev_ground_gib", kept as f64 / 1073741824.0);
        }
        t
    }
}

impl ResponseKernel for GpuResponseKernel {
    fn timers(&self) -> Option<Timers> {
        Some(GpuResponseKernel::timers(self))
    }

    /// The products at the ground state fixed at the setup (`c_o` must have its shapes).
    fn contract_occ(
        &self,
        c_o: &[MatRef<'_, f64>],
        a: &[Vec<Mat<f64>>],
    ) -> Result<Vec<Vec<Mat<f64>>>, LinusError> {
        let n = self.n;
        assert!(
            c_o.len() == self.occ.len() && c_o.iter().zip(&self.occ).all(|(c, &o)| c.ncols() == o),
            "GpuResponseKernel: the occupied orbitals of its setup"
        );
        let nd = a.first().map_or(0, |v| v.len());
        let cat: Vec<Vec<f64>> = a
            .iter()
            .map(|ad| ad.iter().flat_map(|x| to_cm(x.as_ref())).collect())
            .collect();
        let refs: Vec<&[f64]> = cat.iter().map(|v| v.as_slice()).collect();
        let mut k = self.k.lock().unwrap_or_else(|e| e.into_inner());
        let v = k
            .contract(&self.gpu, &refs, nd)
            .map_err(|e| device_error(Family::ResponseXc, e))?;
        {
            let mut t = self.t.lock().unwrap_or_else(|e| e.into_inner());
            for (name, s) in k.last.phases() {
                t.add(&format!("xc_dev_{name}"), s);
            }
            t.add("xc_dev_calls", 1.0);
        }
        drop(k);
        Ok(v.iter()
            .zip(&self.occ)
            .map(|(vs, &o)| {
                (0..nd)
                    .map(|d| from_cm(n, o, &vs[d * n * o..(d + 1) * n * o]))
                    .collect()
            })
            .collect())
    }
}

/// The CPKS kernel of the CPU integrator `xc` on the device for the occupied orbitals `occ`
/// (one channel: a closed shell, two: the spins of an open shell; family `response_xc`),
/// declined (the CPU integrator's) when the setup does not fit; timed as `cphf_xc_setup`. Its
/// arithmetic is the device policy's for the response it serves (`kernel`: FP32 products for a
/// single-precision response or with `precision.fp32: on`, [`DevicePolicy::cpks_xc`]); recorded in
/// `resolved.record.backend.response`.
pub(crate) fn response_kernel<'a>(
    calc: &CalcContext,
    gpu: &Arc<Gpu>,
    xc: &XcIntegrator,
    occ: &[MatRef<'_, f64>],
    kernel: KernelUse,
    timers: &mut Timers,
) -> Result<Setup<Box<dyn ResponseKernel + 'a>>, LinusError> {
    let cm: Vec<Vec<f64>> = occ.iter().map(|c| to_cm(*c)).collect();
    let fs: Vec<Factor> = cm
        .iter()
        .zip(occ)
        .map(|(c, m)| Factor { c, ncol: m.ncols() })
        .collect();
    let policy = device_policy(calc)?;
    let (screen, math, ozaki) = kernel_plan(policy.cpks_xc(kernel.single));
    // the electric-field CPHF (polarizabilities, IR runs): three
    // perturbations per call over 10-15 calls, which on 512-point blocks leave the device
    // mostly idle (each launch a few thread blocks): FP64 blocks of FIELD_BLOCK_POINTS, P = F A
    // of every component, and the ground state kept across the calls where it fits beside the
    // response J/K's exchange workspace per call and 1 GiB (the products agree with the
    // Hessian kernel's blocks to ~1e-11, relative: the blocks screen other functions)
    let field = kernel.purpose == ResponseUse::Polarizability && screen == XcScreen::Block;
    let (block_points, ground_cache) = if field {
        (
            FIELD_BLOCK_POINTS,
            Some(device_memory(calc)?.ground_cache_leave),
        )
    } else {
        (KERNEL_BLOCK_POINTS, None)
    };
    let built = timers.time("cphf_xc_setup", || {
        engine(
            gpu,
            xc,
            calc.plan.method(),
            block_points,
            (screen, math, products(policy)),
        )
        .and_then(|g| {
            GpuXcKernel::new(
                gpu,
                g,
                &fs,
                &GpuXcKernelOptions {
                    math,
                    ozaki_slices: ozaki,
                    ground_cache,
                    q_route: !field,
                    ..GpuXcKernelOptions::default()
                },
            )
        })
    });
    let k = match setup(Family::ResponseXc, built)? {
        Setup::Ready(k) => k,
        Setup::Declined(why) => return Ok(Setup::Declined(why)),
    };
    calc.backend.responses().set_xc(
        kernel.purpose.name(),
        match (math, ozaki) {
            (XcMath::Fp32, _) => "fp32",
            (_, Some(_)) => "emulated",
            _ => "fp64",
        },
    );
    Ok(Setup::Ready(Box::new(GpuResponseKernel {
        gpu: gpu.clone(),
        k: Mutex::new(k),
        n: xc.basis.nbf,
        occ: occ.iter().map(|c| c.ncols()).collect(),
        t: Mutex::new(Timers::default()),
    })))
}

/// The XC gradient on the device (family `gradient_xc`; VV10's on the device too, or the
/// integrator's): its engine set up before the gradient's overlap (its host work on the whole
/// pool), its pass beside the host's one-electron pass; declined (the integrator's gradient,
/// after the overlap) when the setup or the call's workspaces do not fit.
pub(crate) struct GpuXcGradient<'a> {
    calc: &'a CalcContext,
    gpu: Arc<Gpu>,
    /// The integrator of the SCF grid and the engine set up on it.
    engine: Option<(Arc<XcIntegrator>, linus_gpu::grad::xc::GpuXcGrad)>,
}

impl<'a> GpuXcGradient<'a> {
    pub(crate) fn new(calc: &'a CalcContext, gpu: Arc<Gpu>) -> Self {
        GpuXcGradient {
            calc,
            gpu,
            engine: None,
        }
    }
}

impl GradientTerm for GpuXcGradient<'_> {
    fn lane(&self) -> Lane {
        Lane::Device
    }

    fn prepare(
        &mut self,
        inputs: &GradientInputs<'_>,
        timers: &mut Timers,
    ) -> Result<Setup<()>, LinusError> {
        let grid = inputs.ground.grid();
        let Some(x) = self.calc.xc_on(grid, timers)? else {
            return Ok(Setup::Ready(()));
        };
        Ok(
            match gradient_engine(
                self.calc,
                &self.gpu,
                &x,
                grid,
                if inputs.ground.scf.restricted { 1 } else { 2 },
                timers,
            )? {
                Setup::Ready(e) => {
                    self.engine = Some((x, e));
                    Setup::Ready(())
                }
                Setup::Declined(why) => Setup::Declined(why),
            },
        )
    }

    fn gradient(
        &mut self,
        inputs: &GradientInputs<'_>,
        timers: &mut Timers,
    ) -> Result<Setup<Vec<[f64; 3]>>, LinusError> {
        let Some((x, e)) = self.engine.take() else {
            // no XC term (Hartree-Fock)
            return Ok(Setup::Ready(vec![[0.0; 3]; self.calc.mol.natom()]));
        };
        self.compute(&x, e, inputs, timers)
    }
}

/// The device engine of the XC gradient for the CPU integrator `xc` of the SCF grid `grid`,
/// declined (the integrator's, VV10 with it) when its setup does not fit; timed as
/// `grad_xc_setup` (and `grad_xc`).
fn gradient_engine(
    calc: &CalcContext,
    gpu: &Arc<Gpu>,
    xc: &XcIntegrator,
    grid: &crate::driver::GridChoice,
    nsp: usize,
    timers: &mut Timers,
) -> Result<Setup<linus_gpu::grad::xc::GpuXcGrad>, LinusError> {
    use linus_gpu::grad::xc::{GpuXcGrad, GpuXcGradOptions};
    let spec = grid.spec();
    let t0 = std::time::Instant::now();
    let p = device_policy(calc)?;
    let built = GpuXcGrad::new(
        gpu,
        &xc.basis,
        &xc.grid,
        &spec,
        calc.plan.method(),
        &GpuXcGradOptions {
            screen: XcScreen::Batch,
            math: xc_math(p.grad_xc),
            products: products(p),
            ..GpuXcGradOptions::default()
        },
    );
    if let Ok(g) = &built {
        let t = std::time::Instant::now();
        calc.note_functional(Family::GradientXc, g.prepare_functional(gpu, nsp));
        timers.add("grad_xc_setup_functional", t.elapsed().as_secs_f64());
    }
    let t_setup = t0.elapsed().as_secs_f64();
    timers.add("grad_xc_setup", t_setup);
    timers.add("grad_xc", t_setup);
    if let Ok(g) = &built {
        let s = g.stats();
        timers.add("grad_xc_setup_blocks", s.setup_blocks);
        timers.add("grad_xc_setup_layout", s.setup_layout);
        let p = s.setup_layout_parts;
        timers.add("grad_xc_setup_layout_blocks", p.blocks);
        timers.add("grad_xc_setup_layout_screen", p.screen);
        timers.add("grad_xc_setup_layout_lists", p.lists);
        timers.add("grad_xc_setup_layout_plans", s.setup_plans);
    }
    let g = setup(Family::GradientXc, built)?;
    if matches!(g, Setup::Declined(_)) && xc.vv10().is_some() {
        // the integrator's gradient adds its VV10 on the CPU
        calc.note_setup(Family::Vv10, Some(XC_ON_CPU.into()));
    }
    Ok(g)
}

impl GpuXcGradient<'_> {
    /// The XC gradient at the spin densities of `inputs` with the device engine `g`
    /// ([`gradient_engine`] for the integrator `xc`), VV10's added (on the device, or the
    /// integrator's). Declined when the call's workspaces do not fit the device beside what it holds
    /// (a memory cap): nothing of the device's is used, and the integrator's gradient runs instead.
    fn compute(
        &self,
        xc: &XcIntegrator,
        mut g: linus_gpu::grad::xc::GpuXcGrad,
        inputs: &GradientInputs<'_>,
        timers: &mut Timers,
    ) -> Result<Setup<Vec<[f64; 3]>>, LinusError> {
        let (calc, gpu) = (self.calc, &self.gpu);
        let (da, db) = (inputs.da, inputs.db);
        let restricted = inputs.ground.scf.restricted;
        let dens: Vec<Vec<f64>> = if restricted {
            vec![to_cm((da + db).as_ref())]
        } else {
            vec![to_cm(da.as_ref()), to_cm(db.as_ref())]
        };
        let refs: Vec<&[f64]> = dens.iter().map(|d| d.as_slice()).collect();
        let grid_response = calc.plan.grid_response();
        let nsp = if restricted { 1 } else { 2 };
        let computed = timers.time("grad_xc", || {
            g.grant_workspace(gpu, nsp, grid_response)?;
            g.compute(gpu, &refs, restricted, grid_response)
        });
        calc.note_grant(Family::GradientXc, &g.workspace_violations());
        let mut grad = match computed {
            Ok(grad) => grad,
            Err(e @ linus_gpu::GpuError::OutOfMemory { .. }) => {
                if xc.vv10().is_some() {
                    calc.note_setup(Family::Vv10, Some(XC_ON_CPU.into()));
                }
                return Ok(Setup::Declined(format!("device call: {e}")));
            }
            Err(e) => return Err(device_error(Family::GradientXc, e)),
        };
        if let Some(nl) = xc.vv10() {
            let t = std::time::Instant::now();
            let gv = match vv10_gradient(calc, xc, &(da + db), grid_response)? {
                Some(g) => g,
                None => nl.gradient(&(da + db), grid_response),
            };
            for (a, b) in grad.iter_mut().zip(&gv) {
                for c in 0..3 {
                    a[c] += b[c];
                }
            }
            let dt = t.elapsed().as_secs_f64();
            timers.add("grad_vv10", dt);
            timers.add("grad_xc", dt);
        }
        Ok(Setup::Ready(grad))
    }
}
/// VV10's gradient at the total density `dt` on the device, on the NLC grid of the
/// integrator `xc`, when the session puts VV10 there and the setup fits (None: the CPU's,
/// the reason recorded).
pub(crate) fn vv10_gradient(
    calc: &CalcContext,
    xc: &XcIntegrator,
    dt: &Mat<f64>,
    grid_response: bool,
) -> Result<Option<Vec<[f64; 3]>>, LinusError> {
    let (Some(nl), Some(gpu)) = (xc.vv10(), calc.device_for(Family::Vv10)) else {
        return Ok(None);
    };
    let math = vv10_math(device_policy(calc)?.vv10_grad);
    let spec = calc.plan.nlc_grid().spec();
    let built = GpuVv10Grad::new(
        gpu,
        &xc.basis,
        nl.grid(),
        &spec,
        calc.plan.method(),
        &GpuVv10GradOptions {
            rho_thresh: nl.rho_thresh,
            math,
            ..GpuVv10GradOptions::default()
        },
    );
    let Some(mut g) = calc.device_setup(Family::Vv10, built)? else {
        return Ok(None);
    };
    g.compute(gpu, &to_cm(dt.as_ref()), grid_response)
        .map(Some)
        .map_err(|e| device_error(Family::Vv10, e))
}

/// VV10's pair sweeps of responses on the device (`linus_dft::vv10_resp`): linus-gpu's
/// `GpuVv10Resp` in the arithmetic of the SCF's VV10 (by default FP64 on FP64-rich cards and
/// double-float on the others, FP64 with
/// `precision.fp32: "off"`).
struct GpuSweep {
    gpu: Arc<linus_gpu::Gpu>,
    eng: Mutex<linus_gpu::xc::vv10::GpuVv10Resp>,
    npad: usize,
}

impl Vv10PairSweep for GpuSweep {
    fn sweep(
        &self,
        coef: &[f64],
        ld: usize,
        npp: usize,
        fp32: bool,
    ) -> Result<Vec<f64>, LinusError> {
        self.eng
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .sweep(&self.gpu, coef, ld, npp, fp32)
            .map_err(|e| device_error(Family::Vv10, e))
    }

    fn npad(&self) -> usize {
        self.npad
    }
}

/// The device engine of VV10's response ground state `g` (family `vv10`), declined (the CPU's
/// sweeps) when the setup does not fit.
pub(crate) fn vv10_sweeps(
    calc: &CalcContext,
    gpu: &Arc<Gpu>,
    g: &Vv10Ground,
) -> Result<Setup<Option<Box<dyn Vv10PairSweep>>>, LinusError> {
    let math = vv10_math(device_policy(calc)?.vv10_scf);
    let (pts, n) = g.points();
    let built = linus_gpu::xc::vv10::GpuVv10Resp::new(gpu, pts, n, math);
    Ok(match setup(Family::Vv10, built)? {
        Setup::Ready(e) => {
            let npad = e.npad();
            Setup::Ready(Some(Box::new(GpuSweep {
                gpu: gpu.clone(),
                eng: Mutex::new(e),
                npad,
            }) as Box<dyn Vv10PairSweep>))
        }
        Setup::Declined(why) => Setup::Declined(why),
    })
}

/// The semilocal explicit XC terms of a Hessian on the device (family `hessian_xc`):
/// `XcIntegrator::hessian_terms` and its grid terms in one pass over the integrator's grid, for
/// the densities and orbitals of the setup's shapes; `hess` holds the second derivative with the
/// grid terms. Timed as `hess_xc` (its setup also as `hess_xc_setup`, its phases as `hess_xc_*`).
pub(crate) struct GpuXcHessian<'a> {
    gpu: Arc<Gpu>,
    eng: linus_gpu::hess::xc::GpuXcHess,
    calc: &'a CalcContext,
    n: usize,
    n3: usize,
    ncols: Vec<usize>,
}

/// The device engine of the semilocal XC Hessian terms of the integrator `xc` for the shapes of
/// `input` (its spin channels and occupied orbitals, whether the grid moves), declined (the
/// integrator's) when the setup does not fit.
pub(crate) fn hessian_xc<'a>(
    calc: &'a CalcContext,
    gpu: &Arc<Gpu>,
    xc: &'a XcIntegrator,
    input: &HessianInput<'_>,
    timers: &mut Timers,
) -> Result<Setup<Box<dyn XcHessianEngine + 'a>>, LinusError> {
    use linus_gpu::hess::xc::{GpuXcHess, GpuXcHessOptions};
    let nsp = input.ds.count();
    let ncols: Vec<usize> = input.co.iter().map(|c| c.ncols()).collect();
    let p = device_policy(calc)?;
    let t0 = std::time::Instant::now();
    let built = GpuXcHess::new(
        gpu,
        &xc.basis,
        &xc.grid,
        calc.plan.method(),
        nsp,
        input.grid_response,
        &GpuXcHessOptions {
            func_thresh: xc.func_thresh,
            // the policy's (precision.fp32 "on"): the potential slots' products in FP32
            // (sums over at most a batch's points; the batches added in fixed point), as the FP32
            // XC gradient
            single: p.hess_xc == Arith::Fp32,
            products: products(p),
            ..GpuXcHessOptions::default()
        },
    )
    .and_then(|mut h| {
        // the accumulators sized to the free memory (under a cap: more passes, not the CPU)
        h.fit_accumulators(&ncols, gpu.budget().free());
        h.grant_workspace(gpu, &ncols)?;
        Ok(h)
    });
    let t_setup = t0.elapsed().as_secs_f64();
    timers.add("hess_xc_setup", t_setup);
    timers.add("hess_xc", t_setup);
    Ok(match setup(Family::HessianXc, built)? {
        Setup::Ready(eng) => Setup::Ready(Box::new(GpuXcHessian {
            gpu: gpu.clone(),
            eng,
            calc,
            n: xc.basis.nbf,
            n3: 3 * xc.basis.atom_shells.len(),
            ncols,
        })),
        Setup::Declined(why) => Setup::Declined(why),
    })
}

impl XcHessianEngine for GpuXcHessian<'_> {
    fn terms(
        &mut self,
        input: &HessianInput<'_>,
        timers: &mut Timers,
    ) -> Result<XcHessianTerms, LinusError> {
        let (gpu, n, n3) = (&self.gpu, self.n, self.n3);
        let ds = input.ds.as_slice();
        // the XC's densities: a closed shell's total, or the two spins
        let dens: Vec<Vec<f64>> = if ds.len() == 1 {
            vec![to_cm((ds[0] * 2.0).as_ref())]
        } else {
            ds.iter().map(|d| to_cm(d.as_ref())).collect()
        };
        let drefs: Vec<&[f64]> = dens.iter().map(|d| d.as_slice()).collect();
        let cms: Vec<Vec<f64>> = input.co.iter().map(|c| to_cm(*c)).collect();
        let occ: Vec<(&[f64], usize)> = cms
            .iter()
            .zip(&self.ncols)
            .map(|(c, &o)| (c.as_slice(), o))
            .collect();
        let r = timers.time("hess_xc", || self.eng.compute(gpu, &drefs, &occ));
        self.calc
            .note_grant(Family::HessianXc, &self.eng.workspace_violations());
        let r = r.map_err(|e| device_error(Family::HessianXc, e))?;
        // the device's phases (parts of hess_xc)
        for (name, s) in self.eng.last.phases() {
            timers.add(&format!("hess_xc_{name}"), s);
        }
        let split = |v: &[f64], o: usize| -> Vec<Mat<f64>> {
            (0..n3)
                .map(|x| from_cm(n, o, &v[x * n * o..(x + 1) * n * o]))
                .collect()
        };
        let vfix = r
            .vfix_c
            .iter()
            .zip(&self.ncols)
            .map(|(v, &o)| split(v, o))
            .collect();
        let vgrid = r
            .vgrid_c
            .iter()
            .zip(&self.ncols)
            .map(|(v, &o)| split(v, o))
            .collect();
        let hess = r.hfix.iter().zip(&r.hgrid).map(|(a, b)| a + b).collect();
        Ok(XcHessianTerms {
            vfix,
            vgrid,
            hess,
            hess_grid: Vec::new(),
        })
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::driver::CalcOptions;
    use crate::driver::Session;
    use crate::Structure;
    use serde_json::json;

    const MOL: &str =
        "4\n\nO 0.10 -0.05 0.12\nH 0.85 0.52 -0.03\nH -0.68 0.55 0.05\nC 0.20 -1.30 0.60\n";

    /// The semilocal CPKS kernel's arithmetic: FP32 for single-precision responses and
    /// `precision.fp32: on`; the Ozaki scheme's large GEMMs under emulated FP64.
    #[test]
    fn cpks_kernel_arithmetic() {
        use crate::driver::options::Fp64Mode;
        use crate::driver::policy::{CardClass, CardProps};
        use Arith::{Fp32, Fp64};
        let pol = |fp64, fp32, rich: bool| {
            let int8 = Some(if rich { 30.0 } else { 500.0 });
            DevicePolicy::new(
                fp64,
                fp32,
                &CardProps::of_class(CardClass::new(rich, int8), (12, 0)),
            )
        };
        let e = pol(Fp64Mode::Auto, Fp32Mode::Auto, false);
        assert_eq!((e.cpks_xc(false), e.cpks_xc(true)), (OZAKI, Fp32));
        let d = pol(Fp64Mode::Native, Fp32Mode::Off, false);
        assert_eq!((d.cpks_xc(false), d.cpks_xc(true)), (Fp64, Fp32));
        assert_eq!(
            pol(Fp64Mode::Native, Fp32Mode::On, true).cpks_xc(false),
            Fp32
        );
    }

    /// VV10 on the device ([`GpuVv10`], [`vv10_gradient`]) against the CPU's (`xc::vv10`) at
    /// the same SCF density: the energy, the potential matrix and the gradient with and without
    /// the grid response, for the three -V functionals, closed and open shells, linus (SSF) and
    /// PySCF (Becke) NLC grids; the pair sums in FP64 and double-float at the FP64 tolerance,
    /// FP32 at its own; repeated device runs bitwise identical. Skips without a device unless
    /// LINUS_GPU_REQUIRED=1.
    #[test]
    fn vv10_matches_the_cpu() {
        for (charge, mult, method, nlc) in [
            (0, 1, "wb97m-v", "linus:1"),
            (1, 2, "wb97m-v", "linus:1"),
            (0, 1, "b97m-v", "pyscf:0"),
            (1, 2, "wb97x-v", "pyscf:0"),
        ] {
            let mol = Structure::from_xyz_str(MOL, charge, mult).unwrap();
            let o = CalcOptions::from_json(json!({
                "method": method, "basis": "def2-svp", "jk": "df", "grid": {"xc": "linus:1", "nlc": nlc},
                "threads": 4, "backend": "gpu"
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
            // VV10 ran on the device in the card class's default arithmetic: double-float on
            // FP64-poor cards (where the default precision emulates the exchange), FP64 on
            // FP64-rich ones
            let rb = c.backend_used();
            assert_eq!(rb.vv10, Some("gpu"), "{method}");
            let class = if rb.precision == "emulated" {
                "df64"
            } else {
                "fp64"
            };
            let a = rb.vv10_arithmetic.clone().expect("VV10's arithmetic");
            assert!(
                a.scf == class && a.gradient == class,
                "{a:?} on {}",
                rb.precision
            );
            let table = rb.arithmetic.as_ref().expect("the policy table");
            assert_eq!(table["vv10_scf"].arith, class);
            assert!(c
                .vv10()
                .is_some_and(|v| v.evaluations > 0 && v.last_energy.is_some()));
            let r = c.last().unwrap().clone();
            let co: Vec<Mat<f64>> = if r.restricted {
                vec![r.c_a.subcols(0, r.nalpha).to_owned()]
            } else {
                vec![
                    r.c_a.subcols(0, r.nalpha).to_owned(),
                    r.c_b.subcols(0, r.nbeta).to_owned(),
                ]
            };
            let ds: Vec<Mat<f64>> = co.iter().map(|c| c * c.transpose()).collect();
            let dt = if r.restricted {
                &ds[0] * 2.0
            } else {
                &ds[0] + &ds[1]
            };
            let gpu = c.cx.device_for(Family::Vv10).unwrap().clone();
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(4)
                .build()
                .unwrap();
            pool.install(|| {
                let mut t = Timers::default();
                let grid_choice = c.scf_grid();
                let xc = c.cx.xc_on(&grid_choice, &mut t).unwrap().unwrap();
                let nl = xc.vv10().unwrap();
                let (e_cpu, v_cpu) = nl.energy_potential(&dt);
                let g_cpu = [nl.gradient(&dt, true), nl.gradient(&dt, false)];
                let n = dt.nrows();
                let cms: Vec<(Vec<f64>, usize, Vec<f64>)> = co
                    .iter()
                    .map(|m| (to_cm(m.as_ref()), m.ncols(), vec![1.0; m.ncols()]))
                    .collect();
                let ch: Vec<XcChannel> = cms
                    .iter()
                    .map(|(c, k, o)| XcChannel {
                        c,
                        ncol: *k,
                        occ: o,
                    })
                    .collect();
                let spec = c.cx.plan.nlc_grid().spec();
                let gmax = g_cpu[0]
                    .iter()
                    .flatten()
                    .fold(0.0f64, |m, x| m.max(x.abs()));
                // (arithmetic, relative tolerance of E, absolute of V and of the gradient)
                for (math, te, tv, tg) in [
                    (Vv10Math::Fp64, 1e-12, 1e-12, 1e-12),
                    (Vv10Math::DoubleFloat, 1e-12, 1e-12, 1e-12),
                    (Vv10Math::Fp32, 1e-6, 1e-6, 1e-7),
                ] {
                    let mut eng = GpuVv10::new(
                        &gpu,
                        &xc.basis,
                        nl.grid(),
                        method,
                        &GpuVv10Options {
                            func_thresh: nl.eng.func_thresh,
                            rho_thresh: nl.rho_thresh,
                            math,
                            ..GpuVv10Options::default()
                        },
                    )
                    .unwrap();
                    let (e, v) = eng.compute(&gpu, &ch, r.restricted).unwrap();
                    let (e2, v2) = eng.compute(&gpu, &ch, r.restricted).unwrap();
                    assert!(
                        e.to_bits() == e2.to_bits() && v == v2,
                        "{method}: runs differ"
                    );
                    let dv = linus_base::linalg::max_abs((&from_cm(n, n, &v) - &v_cpu).as_ref());
                    let vmax = linus_base::linalg::max_abs(v_cpu.as_ref());
                    let mut dg = [0.0f64; 2];
                    for (k, gr) in [true, false].into_iter().enumerate() {
                        let mut ge = GpuVv10Grad::new(
                            &gpu,
                            &xc.basis,
                            nl.grid(),
                            &spec,
                            method,
                            &GpuVv10GradOptions {
                                rho_thresh: nl.rho_thresh,
                                math,
                                ..GpuVv10GradOptions::default()
                            },
                        )
                        .unwrap();
                        let g = ge.compute(&gpu, &to_cm(dt.as_ref()), gr).unwrap();
                        let g2 = ge.compute(&gpu, &to_cm(dt.as_ref()), gr).unwrap();
                        assert_eq!(g, g2, "{method}: gradient runs differ");
                        dg[k] = g
                            .iter()
                            .flatten()
                            .zip(g_cpu[k].iter().flatten())
                            .map(|(a, b)| (a - b).abs())
                            .fold(0.0, f64::max);
                    }
                    println!(
                        "{method} {charge} {mult} NLC {nlc} {math:?}: |dE| {:.2e} (E {e_cpu:.6}), \
                         |dV| {dv:.2e} (|V| {vmax:.2e}), |dG| {:.2e} / {:.2e} without grid \
                         response (|G| {gmax:.2e}), active {}",
                        (e - e_cpu).abs(),
                        dg[0],
                        dg[1],
                        eng.last.pairs.active
                    );
                    assert!((e - e_cpu).abs() < te * e_cpu.abs(), "{method} {math:?}: E");
                    assert!(dv < tv, "{method} {math:?}: V {dv:e}");
                    assert!(
                        dg[0] < tg && dg[1] < tg,
                        "{method} {math:?}: gradient {dg:?}"
                    );
                }
            });
        }
    }

    /// The device's explicit XC Hessian terms ([`hessian_terms`]) equal the CPU integrator's
    /// `hessian_terms` and `grid_terms` (times C_o) to round-off: LDA, GGA, meta-GGA (r2SCAN,
    /// M06-2X), global (B3LYP, TPSSh) and range-separated hybrids (wB97X, wB97M), closed and
    /// open shells, SSF (linus) and Becke (pyscf) grids, with and without the grid response,
    /// FP64; repeated device runs are bitwise identical. Skips without a device unless
    /// LINUS_GPU_REQUIRED=1.
    #[test]
    fn hessian_terms_match_the_cpu() {
        for (charge, mult, method, grid, gr) in [
            (0, 1, "b3lyp", "linus:3", true),
            (1, 2, "b3lyp", "linus:3", true),
            (0, 1, "pbe", "pyscf:3", true),
            (0, 1, "svwn", "linus:2", true),
            (1, 2, "svwn", "pyscf:2", true),
            (1, 2, "pbe", "linus:2", false),
            (0, 1, "wb97x", "linus:2", true),
            (0, 1, "r2scan", "linus:3", true),
            (1, 2, "r2scan", "pyscf:2", true),
            (0, 1, "tpssh", "pyscf:3", true),
            (1, 2, "wb97m-d3bj", "linus:2", true),
            (0, 1, "m06-2x", "linus:2", false),
        ] {
            let mol = Structure::from_xyz_str(MOL, charge, mult).unwrap();
            let o = CalcOptions::from_json(json!({
                "method": method, "basis": "def2-svp", "jk": "df", "grid": grid, "threads": 4,
                "backend": "gpu"
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
            let co: Vec<Mat<f64>> = if r.restricted {
                vec![r.c_a.subcols(0, r.nalpha).to_owned()]
            } else {
                vec![
                    r.c_a.subcols(0, r.nalpha).to_owned(),
                    r.c_b.subcols(0, r.nbeta).to_owned(),
                ]
            };
            let ds: Vec<Mat<f64>> = co.iter().map(|c| c * c.transpose()).collect();
            let dsm: Vec<&Mat<f64>> = ds.iter().collect();
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(4)
                .build()
                .unwrap();
            pool.install(|| {
                let mut t = Timers::default();
                let grid_choice = c.scf_grid();
                let xc = c.cx.xc_on(&grid_choice, &mut t).unwrap().unwrap();
                let cor: Vec<MatRef<'_, f64>> = co.iter().map(|c| c.as_ref()).collect();
                let cpu = xc.hessian_terms(
                    linus_core::channels::Channels::from_vec(dsm.clone()),
                    linus_core::channels::Channels::from_vec(cor.clone()),
                    gr,
                    Par::Seq,
                    &mut t,
                );
                let (v, vg) = (&cpu.vfix, &cpu.vgrid);
                let mut h = cpu.hess.clone();
                h.iter_mut().zip(&cpu.hess_grid).for_each(|(a, b)| *a += b);
                let gpu = c.cx.device_for(Family::HessianXc).unwrap().clone();
                let dev = |t: &mut Timers| {
                    let input = HessianInput {
                        ds: linus_core::channels::Channels::from_vec(dsm.clone()),
                        co: linus_core::channels::Channels::from_vec(cor.clone()),
                        dt: &ds[0],
                        grid_response: gr,
                        par: Par::Seq,
                    };
                    match hessian_xc(&c.cx, &gpu, &xc, &input, t).unwrap() {
                        Setup::Ready(mut e) => e.terms(&input, t).unwrap(),
                        Setup::Declined(why) => panic!("the device's terms: {why}"),
                    }
                };
                let d = dev(&mut t);
                let dh = h
                    .iter()
                    .zip(&d.hess)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0, f64::max);
                let hmax = h.iter().fold(0.0f64, |m, a| m.max(a.abs()));
                let cmp = |cpu: &[Vec<Mat<f64>>], dv: &[Vec<Mat<f64>>]| {
                    let (mut e, mut m) = (0.0f64, 0.0f64);
                    assert_eq!(cpu.len(), dv.len());
                    for (a, b) in cpu.iter().zip(dv) {
                        assert_eq!(a.len(), b.len());
                        for (uc, w) in a.iter().zip(b) {
                            e = e.max(linus_base::linalg::max_abs((uc - w).as_ref()));
                            m = m.max(linus_base::linalg::max_abs(uc.as_ref()));
                        }
                    }
                    (e, m)
                };
                let (dv, vmax) = cmp(v, &d.vfix);
                let (dg, gmax) = cmp(vg, &d.vgrid);
                println!(
                    "{method} {charge} {mult} {grid} grid response {gr}: |dH| {dh:.2e} (|H| \
                     {hmax:.2e}), |dV^x C| {dv:.2e} (|V^x C| {vmax:.2e}), |d dV^y C| {dg:.2e} \
                     (|dV^y C| {gmax:.2e})"
                );
                assert!(dh < 1e-11, "{method} {charge} {mult} {grid}: |dH| {dh:e}");
                assert!(
                    dv < 1e-11 && dg < 1e-11,
                    "{method} {charge} {mult} {grid}: V"
                );
                let d2 = dev(&mut t);
                assert!(
                    d.hess == d2.hess && d.vfix == d2.vfix && d.vgrid == d2.vgrid,
                    "{method}: device runs differ"
                );
            });
        }
    }
}
