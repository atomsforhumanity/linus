//! The XC gradient on the device, with the grid response: `GpuXcGrad` (feature `gpu`).
//!
//! linus's CPU `XcIntegrator::gradient` (`xc` in the `linus` crate) at fixed density matrices,
//! on the same grid and functional: per spin s, with X = phi D_s, B = T D_s and
//! T_mu = w vrho phi_mu + sum_k vg_k d_k phi_mu (vg_k = w (2 vsigma grad rho)_k, the spin
//! form for open shells), the derivative for atom A is
//!
//!   -2 sum_{mu in A} sum_g c_i(mu, g),
//!   c_i = d_i phi_mu B_mu + sum_k d_ik phi_mu vg_k X_mu (+ 1/2 w vtau sum_k d_ik phi_mu
//!         (d_k phi D)_mu for meta-GGAs);
//!
//! with the grid response, each point's sum 2 sum_mu c_i(mu, g) goes to the atom that owns the
//! point (the point moves with it), and the weights' derivatives contribute
//! sum_g e_xc(g) dw_g/dR (e_xc = rho zk, the energy density).
//!
//! Work:
//! 1. **Blocks** of grid batches with the shells that reach them ([`crate::xc::blocks`], the
//!    CPU gradient's screening threshold 1e-15 at derivative order 2).
//! 2. **Densities** per block from the density matrices (collocation, X = phi_0 D_sub by GEMM,
//!    the XC engine's density kernel), the functional on the host (linus-xc, order 1), its
//!    derivatives back to the device.
//! 3. **Contraction** per block and spin: X, B and (meta-GGA) Y_k = d_k phi D_sub by GEMMs, then
//!    one warp per function sums c_i over the block's points (fixed-point per function), and,
//!    with the grid response, one thread per point sums it over the functions.
//! 4. **Grid response:** the Becke weight derivatives on the device, one block of threads per
//!    chunk of points and one thread per atom (the cell functions and their pair terms of
//!    linus-grid's `weight_gradient_block`, without its screening of negligible atoms), the
//!    point-motion term added there; atom sums in fixed point. Other partitions (SSF, LKO)
//!    take the energy density to the host for linus-grid's own weight derivatives.
//!
//! LDA, GGA and meta-GGA (tau), closed and open shells. Not covered: Laplacian-dependent
//! meta-GGAs. VV10's gradient (`crate::xc::vv10::GpuVv10Grad`) is this engine on the NLC grid
//! with the pair sums in place of step 2's functional, their explicit point derivatives added
//! to the point-motion term of step 4.

use crate::cpu::{GridSpec, MolecularGrid, PartitionScheme, SizeAdjust};

/// Settings of a `GpuXcGrad` setup.
#[derive(Clone, Debug)]
pub struct GpuXcGradOptions {
    /// Functions whose values and first and second derivatives stay below this over a grid
    /// batch (per-batch screening) or a block (block screening) are skipped (the CPU
    /// gradient's threshold).
    pub func_thresh: f64,
    /// Points per block of collocation (whole grid batches); with block screening also the
    /// points of the block's products.
    pub block_points: usize,
    /// Points per block of threads of the weight-derivative kernel.
    pub becke_chunk: usize,
    /// Threads per block of that kernel (atoms times point slots; a multiple of the warp).
    pub becke_threads: usize,
    /// Per grid batch (the CPU's value test, the default) or per block.
    pub screen: crate::xc::XcScreen,
    /// Device bytes for the per-batch path's workspaces of a chunk of batches.
    pub workspace: usize,
    /// Device bytes the per-batch path may keep X = phi D of the density pass in, for the
    /// contraction pass (else X is recomputed there; the results are the same bits).
    pub x_store: usize,
    /// FP64, or (the precision study) FP32 products with FP64 sums of the batches
    /// ([`crate::xc::XcMath::Fp32`]); emulated FP64 with block screening only.
    pub math: crate::xc::XcMath,
    /// Leave out the Becke pair terms of atoms whose terms are negligible at a point (linus-
    /// grid's CPU criterion, per point: |c_B P_B / min_C s_BC| at most 1e-18 |f w| / N).
    pub becke_screen: bool,
    /// The weight derivatives with each atom pair evaluated once per point (pair values in
    /// shared memory), not once per atom of the pair.
    pub becke_pairs: bool,
    /// What runs the FP64 per-batch products.
    pub products: crate::xc::ProductEngine,
}

impl Default for GpuXcGradOptions {
    fn default() -> Self {
        GpuXcGradOptions {
            func_thresh: 1e-15,
            block_points: 2048,
            becke_chunk: 128,
            becke_threads: 256,
            screen: crate::xc::XcScreen::Batch,
            workspace: 1 << 30,
            x_store: 8 << 30,
            math: crate::xc::XcMath::Fp64,
            becke_screen: true,
            becke_pairs: true,
            products: crate::xc::ProductEngine::Vgemm,
        }
    }
}

/// The Becke partition of a grid as its weight derivatives need it: atom positions, inverse
/// distances 1/R_ij and the size adjustments a_ij (n x n, row-major).
#[derive(Clone, Debug)]
pub struct BeckeParams {
    pub natom: usize,
    pub xyz: Vec<f64>,
    pub inv_r: Vec<f64>,
    pub aij: Vec<f64>,
}

impl BeckeParams {
    /// The parameters of `grid`, built with `spec`, if it is partitioned by Becke's scheme
    /// without screening (None otherwise: SSF and LKO, or a single atom).
    pub fn of(grid: &MolecularGrid, spec: &GridSpec) -> Option<BeckeParams> {
        if spec.partition != PartitionScheme::Becke || !grid.partitioner().has_block_gradient() {
            return None;
        }
        let n = grid.atom_z.len();
        if n < 2 {
            return None;
        }
        // linus-grid's Partitioner: radii (square roots for Treutler), a_ij clipped to 1/2
        // (TreutlerBeyondRow2: only the pairs of H with an atom beyond neon; none at all for a
        // molecule without such a pair)
        let row2 = |z: u32| z <= crate::cpu::ROW2_MAX_Z;
        let xh = |a: u32, b: u32| (a == 1 && !row2(b)) || (b == 1 && !row2(a));
        let (rad, adjust, pair_rule): (Vec<f64>, bool, bool) = match spec.size_adjust {
            SizeAdjust::None => (vec![1.0; n], false, false),
            SizeAdjust::Becke(t) => (
                grid.atom_z.iter().map(|&z| t.radius(z) + 1e-200).collect(),
                true,
                false,
            ),
            SizeAdjust::Treutler(t) => (
                grid.atom_z
                    .iter()
                    .map(|&z| t.radius(z).sqrt() + 1e-200)
                    .collect(),
                true,
                false,
            ),
            SizeAdjust::TreutlerBeyondRow2(t) => {
                if grid.atom_z.iter().all(|&z| row2(z)) || !grid.atom_z.contains(&1) {
                    (vec![1.0; n], false, false)
                } else {
                    (
                        grid.atom_z
                            .iter()
                            .map(|&z| t.radius(z).sqrt() + 1e-200)
                            .collect(),
                        true,
                        true,
                    )
                }
            }
        };
        let inv_rad: Vec<f64> = rad.iter().map(|r| 1.0 / r).collect();
        let mut inv_r = vec![0.0; n * n];
        let mut aij = vec![0.0; n * n];
        let p = &grid.atom_xyz;
        for i in 0..n {
            for j in 0..n {
                if i == j {
                    continue;
                }
                let d = ((p[i][0] - p[j][0]).powi(2)
                    + (p[i][1] - p[j][1]).powi(2)
                    + (p[i][2] - p[j][2]).powi(2))
                .sqrt();
                inv_r[i * n + j] = 1.0 / d;
                if adjust && (!pair_rule || xh(grid.atom_z[i], grid.atom_z[j])) {
                    let a = 0.25 * (rad[j] * inv_rad[i] - rad[i] * inv_rad[j]);
                    aij[i * n + j] = a.clamp(-0.5, 0.5);
                }
            }
        }
        Some(BeckeParams {
            natom: n,
            xyz: p.iter().flat_map(|r| r.iter().copied()).collect(),
            inv_r,
            aij,
        })
    }
}

#[cfg(feature = "gpu")]
pub use device::*;

#[cfg(feature = "gpu")]
mod device {
    use super::*;
    use crate::cpu::BasisSet;
    use crate::detsum::DetAcc;
    use crate::engine::Timeline;
    use crate::error::GpuError;
    use crate::kernels::registry::KernelModule;
    use crate::linalg::Gemm;
    use crate::pool::{Grant, PlanViolation};
    use crate::xc::density::{DensityOps, DensityPass, DensityWork, GridDensities, PassTiming};
    use crate::xc::layout::{self, Collocation, XcLayout, XcLayoutSpec};
    use crate::xc::{
        HostFunctional, PointFn, PointInputs, Prod, ProdKind, XcKind, XcMath, XcScreen,
    };
    use crate::{Arg, DeviceBuf, Gpu, Kernel, Launch};
    use std::borrow::Cow;
    use std::time::Instant;

    const KERNELS: &str = include_str!("../../device/grad/xc.cu");

    /// The XC gradient's module: the fixed-point accumulator, then its kernels.
    pub(crate) const MODULE: KernelModule = KernelModule::new(
        "xcgrad",
        || Cow::Owned(format!("{}\n{}", crate::detsum::DEVICE_SRC, KERNELS)),
        &[
            "xg_weights",
            "xg_contract_funcs",
            "xg_contract_points",
            "xgb_weights",
            "xgb_contract_funcs",
            "xgb_contract_points",
            "xg_energy_density",
            "xg_becke",
            "xg_becke_pairs",
        ],
    );

    /// Times of the last gradient (seconds; device phases from events, host phases wall
    /// clock).
    #[derive(Clone, Debug, Default)]
    pub struct GpuXcGradTimes {
        pub total: f64,
        pub colloc: f64,
        /// Densities on the grid (gathers, GEMMs, kernel).
        pub density: f64,
        /// Densities to the host, libxc, derivatives back (wall clock).
        pub functional: f64,
        /// The contraction GEMMs and weights.
        pub gemm: f64,
        /// The contraction kernels.
        pub contract: f64,
        /// The grid response (weight derivatives and point motion).
        pub grid_response: f64,
    }

    /// Setup statistics.
    #[derive(Clone, Debug)]
    pub struct GpuXcGradStats {
        pub npts: usize,
        pub nblocks: usize,
        pub max_block_funcs: usize,
        /// Function-point values per pass and their share of nbf x npts.
        pub values: usize,
        pub kept_fraction: f64,
        /// The weight derivatives run on the device (Becke) or on the host.
        pub becke_on_device: bool,
        pub setup_seconds: f64,
        /// Of `setup_seconds`: the per-block screening and its uploads (block screening's
        /// path), and the per-batch path's layout, workspaces and product plans.
        pub setup_blocks: f64,
        pub setup_layout: f64,
        /// Of `setup_layout`: the blocks, the per-batch value test on the device, the
        /// batches' lists and descriptors (`XcLayout`), and the product plans.
        pub setup_layout_parts: [f64; 4],
    }

    struct Kernels {
        sigma: Kernel,
        energy_density: Kernel,
        becke: Kernel,
        becke_pairs: Kernel,
    }

    /// The most atoms of the weight-derivative kernel (its shared memory: 12 natom doubles per
    /// point slot); larger molecules take the host path.
    const BECKE_MAX_ATOMS: usize = 1000;

    /// The Becke parameters on the device.
    struct DevBecke {
        natom: usize,
        xyz: DeviceBuf<f64>,
        inv_r: DeviceBuf<f64>,
        aij: DeviceBuf<f64>,
        /// The atom pairs (i > j) in order, two i32 each.
        pairs: DeviceBuf<i32>,
    }

    /// The XC gradient on the device (see the module docs).
    pub struct GpuXcGrad {
        nbf: usize,
        npts: usize,
        natom: usize,
        /// The functional on the host and the ingredients it is evaluated with.
        functional: HostFunctional,
        x: DeviceBuf<f64>,
        y: DeviceBuf<f64>,
        z: DeviceBuf<f64>,
        w: DeviceBuf<f64>,
        vol: DeviceBuf<f64>,
        owner: DeviceBuf<i32>,
        owner_host: Vec<u32>,
        func_atom: Vec<usize>,
        becke: Option<DevBecke>,
        /// The grid itself, for the host weight derivatives of other partitions.
        host_grid: Option<MolecularGrid>,
        becke_chunk: usize,
        becke_threads: usize,
        becke_screen: bool,
        becke_pairs: bool,
        kernels: Kernels,
        /// The block path (block screening; feature `study`).
        #[cfg(feature = "study")]
        blk: Option<study::GradBlocks>,
        /// The per-batch path (per-batch screening).
        bat: Option<GradBatched>,
        /// The memory plan's grant of a call's workspaces on the per-batch path
        /// ([`GpuXcGrad::grant_workspace`]), drawn by their allocations (freed, they give the
        /// bytes back).
        workspace: Option<Grant>,
        stats: GpuXcGradStats,
        pub last: GpuXcGradTimes,
    }

    /// The products of one chunk: X = phi D for the density pass (values to first order) and
    /// for the contraction (second order), Y = grad phi D (tau) for both, and B = W D.
    struct GradPlans {
        x1: Prod,
        y1: Prod,
        x2: Prod,
        y2: Prod,
        b: Prod,
    }

    /// The per-batch path's layout, kernels, workspaces and product plans (the plans hold
    /// the workspaces' addresses; both live as long as the engine).
    struct GradBatched {
        lay: XcLayout,
        weights: Kernel,
        contract_funcs: Kernel,
        contract_points: Kernel,
        /// Collocation kernels of the density pass (first order; values only for LDA) and
        /// its components.
        colloc1: Vec<Kernel>,
        nc1: usize,
        plans: Vec<GradPlans>,
        single: bool,
        phi: DeviceBuf<f64>,
        phic: DeviceBuf<f64>,
        xw: DeviceBuf<f64>,
        ww: DeviceBuf<f64>,
        bw: DeviceBuf<f64>,
        yw: DeviceBuf<f64>,
        dv: DeviceBuf<f64>,
        phic32: DeviceBuf<f32>,
        xw32: DeviceBuf<f32>,
        ww32: DeviceBuf<f32>,
        bw32: DeviceBuf<f32>,
        yw32: DeviceBuf<f32>,
        dv32: DeviceBuf<f32>,
        /// The product plans' GEMMs (FP64 and FP32, native).
        gemm: Gemm,
        /// Keep the density pass's X for the contraction (bytes per spin).
        x_store: usize,
        /// Setup seconds of the product plans.
        plans_seconds: f64,
    }

    impl GradBatched {
        fn new(
            gpu: &Gpu,
            basis: &BasisSet,
            grid: &MolecularGrid,
            kind: XcKind,
            order: usize,
            opts: &GpuXcGradOptions,
        ) -> Result<GradBatched, GpuError> {
            let single = match opts.math {
                XcMath::Fp64 => false,
                XcMath::Fp32 => true,
                #[cfg(feature = "study")]
                m => {
                    return Err(GpuError::NotCovered(format!(
                        "XC gradient: {m:?} with per-batch screening (FP64 or FP32)"
                    )))
                }
            };
            let nc2 = crate::cpu::ncomp(order);
            let mgga = kind.mgga;
            // workspace doubles per compressed value: phi (nc2), X, W, B, Y (3), and the FP32
            // copies
            let per_value = 8 * (nc2 + 6) + if single { 4 * (nc2 + 6) } else { 0 };
            let budget = (opts.workspace / per_value).max(1) * nc2;
            let lay = XcLayout::new(
                gpu,
                basis,
                grid,
                &XcLayoutSpec {
                    screen_order: 2,
                    order,
                    thresh: opts.func_thresh,
                    block_points: opts.block_points,
                    budget,
                },
            )?;
            let tp = Instant::now();
            let ord1 = kind.order();
            let nc1 = crate::cpu::ncomp(ord1);
            let colloc1 = lay.kernels_of_order(gpu, basis, ord1)?;
            let sz = |n: usize, on: bool| if on { n.max(1) } else { 1 };
            let phi = gpu.alloc_zeros::<f64>((lay.max_block_nf * nc2 * lay.max_block_np).max(1))?;
            let phic = gpu.alloc_zeros::<f64>((nc2 * lay.max_x).max(1))?;
            let xw = gpu.alloc_zeros::<f64>(sz(lay.max_x, !single))?;
            let ww = gpu.alloc_zeros::<f64>(sz(lay.max_x, !single))?;
            let bw = gpu.alloc_zeros::<f64>(sz(lay.max_x, !single))?;
            let yw = gpu.alloc_zeros::<f64>(sz(3 * lay.max_x, !single && mgga))?;
            let dv = gpu.alloc_zeros::<f64>(sz(lay.max_d, !single))?;
            let phic32 = gpu.alloc_zeros::<f32>(sz(nc2 * lay.max_x, single))?;
            let xw32 = gpu.alloc_zeros::<f32>(sz(lay.max_x, single))?;
            let ww32 = gpu.alloc_zeros::<f32>(sz(lay.max_x, single))?;
            let bw32 = gpu.alloc_zeros::<f32>(sz(lay.max_x, single))?;
            let yw32 = gpu.alloc_zeros::<f32>(sz(3 * lay.max_x, single && mgga))?;
            let dv32 = gpu.alloc_zeros::<f32>(sz(lay.max_d, single))?;
            let vg = &lay.vg;
            let kind_p = if single {
                ProdKind::Grouped32
            } else {
                ProdKind::fp64(opts.products)
            };
            let a = |b64: &DeviceBuf<f64>, b32: &DeviceBuf<f32>, off: usize| -> u64 {
                if single {
                    b32.at(off).addr()
                } else {
                    b64.at(off).addr()
                }
            };
            let mut plans = Vec::with_capacity(lay.chunks.len());
            for c in &lay.chunks {
                use crate::kernels::vgemm::VgemmProduct as P;
                let xp = |nc: usize| {
                    lay.products(c, |h, [po, xo, dd]| {
                        let (np, nf) = (h.np, h.global.len());
                        Some(P {
                            m: np,
                            n: nf,
                            k: nf,
                            a: a(&phic, &phic32, po),
                            lda: nc * np,
                            b: a(&dv, &dv32, dd),
                            ldb: nf,
                            c: a(&xw, &xw32, xo),
                            ldc: np,
                        })
                    })
                };
                let yp = |nc: usize| {
                    if !mgga {
                        return vec![];
                    }
                    lay.products(c, |h, [po, xo, dd]| {
                        let (np, nf) = (h.np, h.global.len());
                        Some(P {
                            m: 3 * np,
                            n: nf,
                            k: nf,
                            a: a(&phic, &phic32, po + np),
                            lda: nc * np,
                            b: a(&dv, &dv32, dd),
                            ldb: nf,
                            c: a(&yw, &yw32, 3 * xo),
                            ldc: 3 * np,
                        })
                    })
                };
                let bp = lay.products(c, |h, [_, xo, dd]| {
                    let (np, nf) = (h.np, h.global.len());
                    Some(P {
                        m: np,
                        n: nf,
                        k: nf,
                        a: a(&ww, &ww32, xo),
                        lda: np,
                        b: a(&dv, &dv32, dd),
                        ldb: nf,
                        c: a(&bw, &bw32, xo),
                        ldc: np,
                    })
                });
                plans.push(GradPlans {
                    x1: Prod::new(gpu, vg, kind_p, &xp(nc1))?,
                    y1: Prod::new(gpu, vg, kind_p, &yp(nc1))?,
                    x2: Prod::new(gpu, vg, kind_p, &xp(nc2))?,
                    y2: Prod::new(gpu, vg, kind_p, &yp(nc2))?,
                    b: Prod::new(gpu, vg, kind_p, &bp)?,
                });
            }
            let gsrc = MODULE.source();
            let k = |e: &str| gpu.kernel(MODULE.name, &gsrc, e);
            Ok(GradBatched {
                weights: k("xgb_weights")?,
                contract_funcs: k("xgb_contract_funcs")?,
                contract_points: k("xgb_contract_points")?,
                colloc1,
                nc1,
                plans,
                single,
                lay,
                phi,
                phic,
                xw,
                ww,
                bw,
                yw,
                dv,
                phic32,
                xw32,
                ww32,
                bw32,
                yw32,
                dv32,
                gemm: Gemm::new(gpu)?,
                x_store: if single { 0 } else { opts.x_store },
                plans_seconds: tp.elapsed().as_secs_f64(),
            })
        }
    }

    impl GpuXcGrad {
        /// The functional `name` (a linus-xc name) on `grid` (built with `spec`) for `basis`.
        pub fn new(
            gpu: &Gpu,
            basis: &BasisSet,
            grid: &MolecularGrid,
            spec: &GridSpec,
            name: &str,
            opts: &GpuXcGradOptions,
        ) -> Result<GpuXcGrad, GpuError> {
            Self::with_kind(gpu, basis, grid, spec, name, None, opts)
        }

        /// As [`GpuXcGrad::new`], with the semilocal ingredients `kind` in place of the
        /// functional's (crate-internal: VV10's gradient on the NLC grid, a GGA of the total
        /// density).
        pub(crate) fn with_kind(
            gpu: &Gpu,
            basis: &BasisSet,
            grid: &MolecularGrid,
            spec: &GridSpec,
            name: &str,
            kind: Option<XcKind>,
            opts: &GpuXcGradOptions,
        ) -> Result<GpuXcGrad, GpuError> {
            let t0 = Instant::now();
            let functional = HostFunctional::new(name, kind)?;
            if functional.needs_lapl() && kind.is_none() {
                return Err(GpuError::NotCovered(format!(
                    "{name}: Laplacian-dependent meta-GGAs are not implemented"
                )));
            }
            let kind = functional.kind();
            // second derivatives of the functions for GGA and meta-GGA, first for LDA; the
            // screening is the CPU gradient's (order 2) either way
            let order = if kind.gga || kind.mgga { 2 } else { 1 };
            let tb = Instant::now();
            #[cfg(feature = "study")]
            let blk = match opts.screen {
                XcScreen::Block => Some(study::GradBlocks::new(gpu, basis, grid, order, opts)?),
                XcScreen::Batch => None,
            };
            #[cfg(not(feature = "study"))]
            if opts.screen == XcScreen::Block {
                return Err(GpuError::NotCovered(
                    "XC gradient: block screening (XcScreen::Block) is a precision-study \
                     variant: build linus-gpu with the `study` feature"
                        .into(),
                ));
            }
            let setup_blocks = tb.elapsed().as_secs_f64();
            let xc = &crate::xc::MODULE;
            let xsrc = xc.source();
            let gsrc = MODULE.source();
            let k = |e: &str| gpu.kernel(MODULE.name, &gsrc, e);
            let kernels = Kernels {
                sigma: gpu.kernel(xc.name, &xsrc, "xc_sigma")?,
                energy_density: k("xg_energy_density")?,
                becke: k("xg_becke")?,
                becke_pairs: k("xg_becke_pairs")?,
            };
            let natom = grid.atom_z.len();
            let becke = match BeckeParams::of(grid, spec).filter(|p| p.natom <= BECKE_MAX_ATOMS) {
                Some(p) => {
                    let pairs: Vec<i32> = (1..p.natom as i32)
                        .flat_map(|i| (0..i).flat_map(move |j| [i, j]))
                        .collect();
                    Some(DevBecke {
                        natom: p.natom,
                        xyz: gpu.upload(&p.xyz)?,
                        inv_r: gpu.upload(&p.inv_r)?,
                        aij: gpu.upload(&p.aij)?,
                        pairs: gpu.upload(if pairs.is_empty() { &[0, 0] } else { &pairs })?,
                    })
                }
                None => None,
            };
            let host_grid = if becke.is_none() && natom > 1 {
                Some(grid.clone())
            } else {
                None
            };
            let mut func_atom = vec![0; basis.nbf];
            for s in &basis.shells {
                func_atom[s.ao_start..s.ao_start + s.nfunc()].fill(s.atom);
            }
            let owner: Vec<i32> = grid.atom.iter().map(|&a| a as i32).collect();
            let npts = grid.x.len();
            let tl = Instant::now();
            let bat = match opts.screen {
                XcScreen::Batch => Some(GradBatched::new(gpu, basis, grid, kind, order, opts)?),
                XcScreen::Block => None,
            };
            let setup_layout = tl.elapsed().as_secs_f64();
            // (blocks, largest block's functions, values per pass) of the path taken
            let (nblocks, max_block_funcs, values) = match &bat {
                Some(b) => (b.lay.blocks.len(), b.lay.max_block_nf, b.lay.values),
                #[cfg(feature = "study")]
                None => blk.as_ref().map_or((0, 0, 0), |b| b.sizes()),
                #[cfg(not(feature = "study"))]
                None => (0, 0, 0),
            };
            let stats = GpuXcGradStats {
                npts,
                nblocks,
                max_block_funcs,
                values,
                kept_fraction: values as f64 / (basis.nbf * npts).max(1) as f64,
                becke_on_device: becke.is_some(),
                setup_seconds: t0.elapsed().as_secs_f64(),
                setup_blocks,
                setup_layout,
                setup_layout_parts: bat.as_ref().map_or([0.0; 4], |b| {
                    let [x, y, z] = b.lay.setup_times;
                    [x, y, z, b.plans_seconds]
                }),
            };
            Ok(GpuXcGrad {
                nbf: basis.nbf,
                npts,
                natom,
                functional,
                x: gpu.upload(&grid.x)?,
                y: gpu.upload(&grid.y)?,
                z: gpu.upload(&grid.z)?,
                w: gpu.upload(&grid.w)?,
                vol: gpu.upload(&grid.w_atomic)?,
                owner: gpu.upload(if owner.is_empty() { &[0] } else { &owner })?,
                owner_host: grid.atom.clone(),
                func_atom,
                becke,
                host_grid,
                becke_chunk: opts.becke_chunk.max(1),
                becke_threads: opts.becke_threads.max(gpu.warp_size() as usize),
                becke_screen: opts.becke_screen,
                becke_pairs: opts.becke_pairs,
                kernels,
                #[cfg(feature = "study")]
                blk,
                bat,
                workspace: None,
                stats,
                last: GpuXcGradTimes::default(),
            })
        }

        /// Device bytes of a call's workspaces on the per-batch path with `nsp` spin channels
        /// (and the grid response's terms with `grid_response`), besides the setup's and the
        /// densities' and functional's per-point values: X of the density pass kept for the
        /// contraction (`keep_x`), the point-motion and potential terms per point.
        pub fn workspace_bytes(&self, nsp: usize, grid_response: bool, keep_x: bool) -> usize {
            let Some(bat) = &self.bat else {
                return 0;
            };
            let npts = self.npts;
            let total_x: usize = bat.lay.chunks.iter().map(|c| c.x_len).sum();
            let store = if keep_x { nsp * total_x } else { 1 };
            let tp = if grid_response { 3 * npts } else { 1 };
            let vtp = if self.kind().mgga { npts } else { 1 };
            let eps = if grid_response { npts } else { 0 };
            8 * (store + tp + 3 * npts + vtp + eps)
        }

        /// Sets aside a call's workspaces ([`GpuXcGrad::workspace_bytes`]) as the memory plan's
        /// grant, which the call's allocations draw from: with X kept for the contraction when
        /// it fits `x_store` and the device, else recomputed there (the same bits); an
        /// out-of-memory error when not even that fits.
        pub fn grant_workspace(
            &mut self,
            gpu: &Gpu,
            nsp: usize,
            grid_response: bool,
        ) -> Result<(), GpuError> {
            self.workspace = None;
            let Some(bat) = &self.bat else {
                return Ok(());
            };
            let total_x: usize = bat.lay.chunks.iter().map(|c| c.x_len).sum();
            let keep = !bat.single && 8 * total_x <= bat.x_store;
            let what = "the XC gradient's workspace";
            if keep {
                if let Ok(g) = gpu.grant(self.workspace_bytes(nsp, grid_response, true), what) {
                    self.workspace = Some(g);
                    return Ok(());
                }
            }
            let g = gpu.grant(self.workspace_bytes(nsp, grid_response, false), what)?;
            if let Some(bat) = &mut self.bat {
                bat.x_store = 0;
            }
            self.workspace = Some(g);
            Ok(())
        }

        /// The calls' draws beyond the workspace's grant (plan violations).
        pub fn workspace_violations(&self) -> Vec<PlanViolation> {
            self.workspace
                .as_ref()
                .map(Grant::violations)
                .unwrap_or_default()
        }

        pub fn stats(&self) -> &GpuXcGradStats {
            &self.stats
        }

        pub fn kind(&self) -> XcKind {
            self.functional.kind()
        }

        /// The grid's densities `rho` and `grad` as a pointwise functional sees them.
        fn point_inputs<'a>(
            &'a self,
            restricted: bool,
            rho: &'a DeviceBuf<f64>,
            grad: &'a DeviceBuf<f64>,
        ) -> PointInputs<'a> {
            PointInputs {
                npts: self.npts,
                restricted,
                x: &self.x,
                y: &self.y,
                z: &self.z,
                w: &self.w,
                rho,
                grad,
            }
        }

        /// The per-batch path of [`GpuXcGrad::compute_device`] (arguments checked there).
        /// Pass 1: per chunk, each block collocated to first order and compressed, per spin
        /// X = phi D_b (and Y) of all the chunk's batches in one launch, the densities; X kept
        /// for pass 2 when it fits `x_store`. Pass 2: second-order collocation, per spin W,
        /// B = W D_b, X (kept or recomputed), Y, and the contractions per function and per
        /// point.
        fn compute_batched(
            &mut self,
            gpu: &Gpu,
            dens: &DeviceBuf<f64>,
            restricted: bool,
            grid_response: bool,
            func: Option<&mut PointFn<'_>>,
        ) -> Result<Vec<[f64; 3]>, GpuError> {
            let (nbf, npts) = (self.nbf, self.npts);
            let n2 = nbf * nbf;
            let nsp = if restricted { 1 } else { 2 };
            let t_all = Instant::now();
            let mut times = GpuXcGradTimes::default();
            let (gga, mgga) = (self.kind().gga, self.kind().mgga);
            let mut gd = GridDensities::new(gpu, npts, nsp, self.kind())?;
            let mut tl = Timeline::<GpuXcGradTimes>::new(gpu);
            let xyz = [&self.x, &self.y, &self.z];
            let bat = self.bat.as_mut().expect("per-batch path");
            let single = bat.single;
            let (nc1, nc2) = (bat.nc1, bat.lay.ncomp);
            let nchunks = bat.lay.chunks.len();
            let a64 = |b: &DeviceBuf<f64>, on: bool| if on { Arg::buf(b) } else { Arg::Ptr(0) };
            let a32 = |b: &DeviceBuf<f32>, on: bool| if on { Arg::buf(b) } else { Arg::Ptr(0) };
            // X of every chunk and spin kept for pass 2 (chunk offsets: sums of x_len)
            let total_x: usize = bat.lay.chunks.iter().map(|c| c.x_len).sum();
            let keep = !single && 8 * total_x <= bat.x_store;
            let mut store = gpu.alloc_zeros_granted::<f64>(
                self.workspace.as_ref(),
                if keep { nsp * total_x } else { 1 }.max(1),
            )?;
            let mut xbase = Vec::with_capacity(nchunks);
            let mut acc_x = 0;
            for c in &bat.lay.chunks {
                xbase.push(acc_x);
                acc_x += c.x_len;
            }

            // pass 1: densities
            let pass = DensityPass {
                lay: &bat.lay,
                colloc: Collocation::With(&bat.colloc1, nc1),
                xyz,
                d: Some(dens),
                nbf,
                nsp,
                scale: 1.0,
                kind: self.functional.kind(),
                gemm: &bat.gemm,
            };
            let mut work = DensityWork {
                phi: &mut bat.phi,
                phic: &mut bat.phic,
                phic32: if single { Some(&mut bat.phic32) } else { None },
                ops: if single {
                    DensityOps::Fp32 {
                        dv: &mut bat.dv32,
                        xw: &bat.xw32,
                        yw: &bat.yw32,
                    }
                } else {
                    DensityOps::Fp64 {
                        dv: &mut bat.dv,
                        xw: &bat.xw,
                        yw: &bat.yw,
                    }
                },
            };
            let (plans, chunks, xw) = (&bat.plans, &bat.lay.chunks, &bat.xw);
            pass.run(
                gpu,
                &mut work,
                gd.out(),
                Some(PassTiming {
                    tl: &mut tl,
                    colloc: |t| &mut t.colloc,
                    density: |t| &mut t.density,
                }),
                |ci| [&plans[ci].x1, &plans[ci].y1],
                |ci, sp| {
                    if keep {
                        gpu.copy_range(
                            xw,
                            0,
                            &mut store,
                            sp * total_x + xbase[ci],
                            chunks[ci].x_len,
                        )?;
                    }
                    Ok(())
                },
            )?;
            if gga {
                gd.fill_sigma(gpu, &self.kernels.sigma, npts, nsp)?;
            }
            let GridDensities {
                rho,
                grad,
                sigma,
                tau,
            } = gd;
            gpu.synchronize()?;
            let tf = Instant::now();
            let inp = self.point_inputs(restricted, &rho, &grad);
            let (pt, _) = self.functional.pointwise(gpu, &inp, &sigma, &tau, func)?;
            times.functional = tf.elapsed().as_secs_f64();
            let (dzk, dvrho, dvsig, dvtau, force) =
                (pt.zk, pt.vrho, pt.vsigma, pt.vtau, pt.point_force);

            // pass 2: the contraction
            let mut facc = DetAcc::new(gpu, nbf * 3)?;
            // the point-motion term, starting from the functional's explicit point derivatives
            let mut tp = match force {
                Some(f) if grid_response => f,
                _ => gpu.alloc_zeros_granted::<f64>(
                    self.workspace.as_ref(),
                    if grid_response { 3 * npts } else { 1 },
                )?,
            };
            let mut vgp = gpu.alloc_zeros_granted::<f64>(self.workspace.as_ref(), 3 * npts)?;
            let mut vtp = gpu
                .alloc_zeros_granted::<f64>(self.workspace.as_ref(), if mgga { npts } else { 1 })?;
            let xyz = [&self.x, &self.y, &self.z];
            let w = &self.w;
            let bat = self.bat.as_mut().expect("per-batch path");
            let ws = gpu.warp_size() as usize;
            for (ci, &xb) in xbase.iter().enumerate() {
                let c = bat.lay.chunks[ci].clone();
                let nbj = c.j1 - c.j0;
                if nbj == 0 {
                    continue;
                }
                let s = tl.mark()?;
                bat.lay.prepare(
                    gpu,
                    &c,
                    Collocation::Own,
                    xyz,
                    &mut bat.phi,
                    Some(&mut bat.phic),
                    if single { Some(&mut bat.phic32) } else { None },
                )?;
                tl.close(|t| &mut t.colloc, s)?;
                let (maxd, maxx) = bat.lay.chunk_max(&c);
                let maxf = bat.lay.host[c.j0..c.j1]
                    .iter()
                    .map(|h| h.global.len())
                    .max()
                    .unwrap_or(0);
                for sp in 0..nsp {
                    let s = tl.mark()?;
                    // SAFETY: as in pass 1.
                    unsafe {
                        gpu.launch(
                            &bat.lay.k.gather_d,
                            layout::per_batch(maxd, nbj, 256),
                            &[
                                Arg::I64(nbf as i64),
                                Arg::buf(&bat.lay.bd),
                                Arg::I32(c.j0 as i32),
                                Arg::buf(&bat.lay.gf),
                                Arg::from(dens.at(sp * n2)),
                                a64(&bat.dv, !single),
                                a32(&bat.dv32, single),
                            ],
                        )
                    }?;
                    // SAFETY: matches xgb_weights; reads the batches' points of the grid
                    // arrays and compressed values, writes each batch's np nf weights at its
                    // offset and the points' vg (vt).
                    unsafe {
                        gpu.launch(
                            &bat.weights,
                            layout::per_batch(maxx, nbj, 256),
                            &[
                                Arg::I32(nc2 as i32),
                                Arg::I32(nsp as i32),
                                Arg::I32(sp as i32),
                                Arg::I32(gga as i32),
                                Arg::I32(mgga as i32),
                                Arg::buf(&bat.lay.bd),
                                Arg::I32(c.j0 as i32),
                                Arg::buf(w),
                                Arg::buf(&dvrho),
                                Arg::buf(&dvsig),
                                Arg::buf(&dvtau),
                                Arg::buf(&grad),
                                Arg::buf(&bat.phic),
                                a64(&bat.ww, !single),
                                a32(&bat.ww32, single),
                                Arg::buf_mut(&mut vgp),
                                Arg::buf_mut(&mut vtp),
                            ],
                        )
                    }?;
                    // SAFETY: the plans address the engine's workspaces of this chunk.
                    unsafe {
                        bat.plans[ci]
                            .b
                            .run(gpu, &bat.lay.vg, &bat.gemm, false, 0.0)?;
                        if !keep {
                            bat.plans[ci]
                                .x2
                                .run(gpu, &bat.lay.vg, &bat.gemm, false, 0.0)?;
                        }
                        bat.plans[ci]
                            .y2
                            .run(gpu, &bat.lay.vg, &bat.gemm, false, 0.0)?;
                    }
                    let xs = if keep {
                        Arg::from(store.at(sp * total_x + xb))
                    } else {
                        a64(&bat.xw, !single)
                    };
                    let s1 = tl.mark()?;
                    let wpb = 256 / ws;
                    let cfg = layout::per_batch(maxf * ws, nbj, 256);
                    let _ = wpb;
                    // SAFETY: matches xgb_contract_funcs; one warp per kept function of each
                    // batch reads its arrays (np points) and adds to the accumulator of its
                    // global function (< nbf).
                    unsafe {
                        gpu.launch(
                            &bat.contract_funcs,
                            cfg,
                            &[
                                Arg::I32(nc2 as i32),
                                Arg::I32(gga as i32),
                                Arg::I32(mgga as i32),
                                Arg::buf(&bat.lay.bd),
                                Arg::I32(c.j0 as i32),
                                Arg::buf(&bat.lay.gf),
                                Arg::buf(&bat.phic),
                                xs,
                                a64(&bat.bw, !single),
                                a64(&bat.yw, !single && mgga),
                                a32(&bat.xw32, single),
                                a32(&bat.bw32, single),
                                a32(&bat.yw32, single && mgga),
                                Arg::buf(&vgp),
                                Arg::buf(&vtp),
                                facc.arg(),
                                Arg::I64((nbf * 3) as i64),
                            ],
                        )
                    }?;
                    if grid_response {
                        // SAFETY: matches xgb_contract_points; one thread per point of each
                        // batch reads its arrays and adds to tp at the point.
                        unsafe {
                            gpu.launch(
                                &bat.contract_points,
                                layout::per_batch(c.max_np, nbj, 128),
                                &[
                                    Arg::I32(nc2 as i32),
                                    Arg::I32(gga as i32),
                                    Arg::I32(mgga as i32),
                                    Arg::buf(&bat.lay.bd),
                                    Arg::I32(c.j0 as i32),
                                    Arg::buf(&bat.phic),
                                    xs,
                                    a64(&bat.bw, !single),
                                    a64(&bat.yw, !single && mgga),
                                    a32(&bat.xw32, single),
                                    a32(&bat.bw32, single),
                                    a32(&bat.yw32, single && mgga),
                                    Arg::buf(&vgp),
                                    Arg::buf(&vtp),
                                    Arg::buf_mut(&mut tp),
                                ],
                            )
                        }?;
                    }
                    tl.span(|t| &mut t.gemm, s, s1);
                    tl.close(|t| &mut t.contract, s1)?;
                }
            }
            drop(store);
            tl.add_to(&mut times)?;
            let g = self.tail(gpu, nsp, grid_response, &rho, &dzk, &tp, &facc, &mut times)?;
            times.total = t_all.elapsed().as_secs_f64();
            self.last = times;
            Ok(g)
        }

        /// The grid response (the weight derivatives with the point motion `tp`, from the
        /// energy density rho zk) and the atoms' sums of the function terms in `facc`.
        #[allow(clippy::too_many_arguments)]
        fn tail(
            &self,
            gpu: &Gpu,
            nsp: usize,
            grid_response: bool,
            rho: &DeviceBuf<f64>,
            dzk: &DeviceBuf<f64>,
            tp: &DeviceBuf<f64>,
            facc: &DetAcc,
            times: &mut GpuXcGradTimes,
        ) -> Result<Vec<[f64; 3]>, GpuError> {
            let (npts, natom) = (self.npts, self.natom);
            // grid response: weight derivatives and point motion
            let mut g = vec![[0.0; 3]; natom];
            if grid_response {
                let s = gpu.mark()?;
                let mut eps = gpu.alloc_zeros_granted::<f64>(self.workspace.as_ref(), npts)?;
                // SAFETY: matches xg_energy_density; npts points.
                unsafe {
                    gpu.launch(
                        &self.kernels.energy_density,
                        Launch::linear(npts, 256),
                        &[
                            Arg::I64(npts as i64),
                            Arg::I32(nsp as i32),
                            Arg::buf(rho),
                            Arg::buf(dzk),
                            Arg::buf_mut(&mut eps),
                        ],
                    )
                }?;
                match &self.becke {
                    Some(bk) => {
                        let mut acc = DetAcc::new(gpu, bk.natom * 3)?;
                        let ws = gpu.warp_size() as usize;
                        let nat = bk.natom;
                        let becke_screen = self.becke_screen;
                        // the pair kernel while a slot's pair values fit 96 KiB of shared
                        // memory (about 110 atoms)
                        let pair_slot = 8 * (12 * nat + 2 + nat * (nat - 1));
                        let pairs = self.becke_pairs && pair_slot <= 96 << 10;
                        let (kernel, nslot, threads, shared) = if pairs {
                            // up to 4 points per block, 256 threads shared out among them
                            let nslot = ((96 << 10) / pair_slot).clamp(1, 4).min(self.becke_chunk);
                            let tps = (256 / nslot / ws).max(1) * ws;
                            (
                                &self.kernels.becke_pairs,
                                nslot,
                                tps * nslot,
                                nslot * pair_slot,
                            )
                        } else {
                            // several points per block when the atoms fill fewer than 256 threads
                            let nslot = (self.becke_threads / nat).clamp(1, self.becke_chunk);
                            let threads = if nslot > 1 {
                                (nslot * nat).div_ceil(ws) * ws
                            } else {
                                nat.div_ceil(ws).clamp(1, self.becke_threads / ws) * ws
                            };
                            (
                                &self.kernels.becke,
                                nslot,
                                threads,
                                nslot * (12 * nat + 2) * 8,
                            )
                        };
                        if shared > 48 * 1024 {
                            kernel.set_max_dynamic_shared(shared as u32)?;
                        }
                        let ch = self.becke_chunk;
                        let cfg = Launch {
                            grid: (npts.div_ceil(ch) as u32, 1, 1),
                            block: (threads as u32, 1, 1),
                            shared_bytes: shared as u32,
                        };
                        // SAFETY: matches xg_becke (xg_becke_pairs); reads npts points of the
                        // grid arrays, eps and tp, the natom^2 tables; writes the dynamic shared
                        // memory of nslot (12 natom + 2) (12 natom + 2 + natom (natom - 1))
                        // doubles and natom 3 accumulators.
                        unsafe {
                            gpu.launch(
                                kernel,
                                cfg,
                                &[
                                    Arg::I64(npts as i64),
                                    Arg::I32(nat as i32),
                                    Arg::I32(nslot as i32),
                                    Arg::I32(ch as i32),
                                    Arg::buf(&self.x),
                                    Arg::buf(&self.y),
                                    Arg::buf(&self.z),
                                    Arg::buf(&self.owner),
                                    Arg::buf(&self.vol),
                                    Arg::buf(&eps),
                                    Arg::I32(1),
                                    Arg::buf(tp),
                                    Arg::buf(&bk.xyz),
                                    Arg::buf(&bk.inv_r),
                                    Arg::buf(&bk.aij),
                                    acc.arg(),
                                    Arg::I64((bk.natom * 3) as i64),
                                    Arg::I32(becke_screen as i32),
                                    Arg::buf(&bk.pairs),
                                ],
                            )
                        }?;
                        let h = acc.to_f64(gpu)?;
                        for (a, ga) in g.iter_mut().enumerate().take(bk.natom) {
                            for k in 0..3 {
                                ga[k] += h[3 * a + k];
                            }
                        }
                    }
                    None => {
                        let htp = gpu.download(tp)?;
                        let mut gm = vec![[0.0; 3]; natom];
                        for (p, &own) in self.owner_host.iter().enumerate() {
                            for k in 0..3 {
                                gm[own as usize][k] += htp[3 * p + k];
                            }
                        }
                        super::super::add_atoms(&mut g, &gm);
                        if let Some(grid) = &self.host_grid {
                            let heps = gpu.download(&eps)?;
                            super::super::add_atoms(&mut g, &grid.weight_gradient_contract(&heps));
                        }
                    }
                }
                times.grid_response += s.seconds_until(&gpu.mark()?)?;
            }
            let hf = facc.to_f64(gpu)?;
            for (mu, &a) in self.func_atom.iter().enumerate() {
                for k in 0..3 {
                    g[a][k] += hf[3 * mu + k];
                }
            }
            Ok(g)
        }

        /// The gradient of density matrices given on the host (nbf x nbf each; one total
        /// density for a closed shell with `restricted`, else alpha and beta).
        pub fn compute(
            &mut self,
            gpu: &Gpu,
            dens: &[&[f64]],
            restricted: bool,
            grid_response: bool,
        ) -> Result<Vec<[f64; 3]>, GpuError> {
            let n2 = self.nbf * self.nbf;
            let mut flat = Vec::with_capacity(dens.len() * n2);
            for d in dens {
                if d.len() != n2 {
                    return Err(GpuError::Contract(
                        "XC gradient: a density has the wrong size".into(),
                    ));
                }
                flat.extend_from_slice(d);
            }
            let dd = gpu.upload(if flat.is_empty() { &[0.0] } else { &flat })?;
            self.compute_device(gpu, &dd, restricted, grid_response)
        }

        /// The gradient (natom x 3) of density matrices on the device: `dens` holds one total
        /// density (nbf x nbf, symmetric) for a closed shell with `restricted`, else alpha and
        /// beta one after another. With `grid_response` the grid moves with the atoms (the
        /// exact derivative of the quadrature energy).
        pub fn compute_device(
            &mut self,
            gpu: &Gpu,
            dens: &DeviceBuf<f64>,
            restricted: bool,
            grid_response: bool,
        ) -> Result<Vec<[f64; 3]>, GpuError> {
            self.compute_device_with(gpu, dens, restricted, grid_response, None)
        }

        /// As [`GpuXcGrad::compute_device`], with the pointwise functional `func` in place of
        /// libxc when given (crate-internal: VV10); its explicit point derivatives (with
        /// `grid_response`) join the point-motion term.
        pub(crate) fn compute_device_with(
            &mut self,
            gpu: &Gpu,
            dens: &DeviceBuf<f64>,
            restricted: bool,
            grid_response: bool,
            func: Option<&mut PointFn<'_>>,
        ) -> Result<Vec<[f64; 3]>, GpuError> {
            let (nbf, npts, natom) = (self.nbf, self.npts, self.natom);
            let n2 = nbf * nbf;
            let nsp = if restricted { 1 } else { 2 };
            if dens.len() < nsp * n2 {
                return Err(GpuError::Contract(
                    "XC gradient: density buffer too small".into(),
                ));
            }
            if self.kind().none || npts == 0 {
                return Ok(vec![[0.0; 3]; natom]);
            }
            if self.bat.is_some() {
                self.compute_batched(gpu, dens, restricted, grid_response, func)
            } else {
                self.compute_blocks(gpu, dens, restricted, grid_response, func)
            }
        }
    }

    // the precision studies' block path of the gradient
    #[cfg(feature = "study")]
    mod study;

    #[cfg(not(feature = "study"))]
    impl GpuXcGrad {
        /// The block path of the XC gradient is a precision-study variant (feature `study`).
        fn compute_blocks(
            &mut self,
            _gpu: &Gpu,
            _dens: &DeviceBuf<f64>,
            _restricted: bool,
            _grid_response: bool,
            _func: Option<&mut PointFn<'_>>,
        ) -> Result<Vec<[f64; 3]>, GpuError> {
            Err(GpuError::NotCovered(
                "XC gradient: the block path (XcScreen::Block) is a precision-study variant: \
                 build linus-gpu with the `study` feature"
                    .into(),
            ))
        }
    }
}
