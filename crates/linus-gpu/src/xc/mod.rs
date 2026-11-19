//! Exchange-correlation energy and potential on the device: `GpuXc` (feature `gpu`), and the
//! XC family's shared parts.
//!
//! The quadrature of linus's CPU integrator (`XcIntegrator` in the `linus` crate) on the same
//! grid, functional and screening threshold, with the density from the occupied orbitals.
//! The XC engines (V_xc here, the gradient `crate::grad::xc`, the Hessian's XC terms
//! `crate::hess::xc`, the CPKS kernel `crate::response::xc`, VV10 [`vv10`]) are composed on
//! the same parts:
//! - `layout`: the per-batch screening (`XcLayout`): blocks of consecutive grid batches with
//!   about `block_points` points ([`blocks`]: a block keeps the shells whose screening radius,
//!   linus-ints' `Collocator` bound at `func_thresh`, reaches its box), and per batch the
//!   functions the CPU's value test keeps, compressed per chunk of batches;
//! - `density`: the grid densities and the per-batch density pass: per chunk the blocks
//!   collocated (the generated kernels of `kernels::colloc`) and compressed, per spin
//!   X = phi D_b (and Y = grad phi D_b for tau) for every batch in one launch, then rho,
//!   grad rho and tau per point;
//! - `functional`: the functional through linus-xc (libxc on the CPU; the grid's densities
//!   go to the host once and the derivatives come back once), or a device function in its
//!   place (VV10's pair sums).
//!
//! V_xc per batch and spin: W = 1/2 w vrho phi + w vg . grad phi (vg_k = 2 vsigma_ss
//! grad_s,k + vsigma_ab grad_s',k), V_b = phi^T W (+ for meta-GGA grad phi^T (1/4 w vtau
//! grad phi)), V += V_b + V_b^T into a fixed-point accumulator (`detsum`); E_xc = sum w rho zk
//! in fixed point too.
//!
//! Block screening ([`XcScreen::Block`]: the rigorous shell bound per block, one product per
//! block) is the substrate of the FP64 CPKS kernel. Its V_xc and gradient paths, the density
//! routes per block (orbitals or density matrix) and the emulated and FP32-potential products
//! are the precision studies' variants, behind the `study` feature.
//!
//! Deterministic: one writer per output in every kernel, the potential and the energy in fixed
//! point. Restricted and unrestricted; LDA, GGA and meta-GGA
//! (tau). Not covered: Laplacian-dependent meta-GGAs. VV10 is its own engine ([`vv10`]): the
//! same machinery on the NLC grid with the pair sums in place of step 4; `GpuXc::vv10` tells
//! whether the functional has a nonlocal term.

use crate::cpu::{BasisSet, Collocator, Family, Functional, MolecularGrid};

// The per-batch layout's host side (batches, chunks, descriptors) is plain Rust, unit-tested
// without a GPU; only the device side uses it.
#[cfg_attr(not(feature = "gpu"), allow(dead_code))]
pub(crate) mod layout;
// the one host evaluator of the XC engines; without `gpu` only its unit tests use it
#[cfg_attr(not(feature = "gpu"), allow(dead_code))]
pub(crate) mod functional;
// the grid densities and the per-batch density pass of the engines
#[cfg(feature = "gpu")]
pub(crate) mod density;
pub mod vv10;

#[cfg(feature = "gpu")]
pub(crate) use functional::{HostFunctional, PointFn, PointInputs, PointOutputs};

/// Settings of a `GpuXc` setup.
#[derive(Clone, Debug)]
pub struct GpuXcOptions {
    /// Functions whose values and gradients stay below this over a grid batch (per-batch
    /// screening) or a block (block screening) are skipped (the CPU integrator's
    /// `func_thresh`).
    pub func_thresh: f64,
    /// Points per block of collocation (whole grid batches are added until a block reaches
    /// it); with block screening also the points of the block's products.
    pub block_points: usize,
    /// The block path's density route (per-batch screening always takes the density matrix).
    #[cfg(feature = "study")]
    pub route: DensityRoute,
    pub screen: XcScreen,
    /// Device bytes for the per-batch path's workspaces (compressed values and the products
    /// of a chunk of batches; at least one block's).
    pub workspace: usize,
    pub math: XcMath,
    /// What runs the FP64 per-batch products.
    pub products: ProductEngine,
}

impl Default for GpuXcOptions {
    fn default() -> Self {
        GpuXcOptions {
            func_thresh: 1e-11,
            block_points: 2048,
            #[cfg(feature = "study")]
            route: DensityRoute::Auto,
            screen: XcScreen::Batch,
            workspace: 1 << 30,
            math: XcMath::Fp64,
            products: ProductEngine::Vgemm,
        }
    }
}

/// Where the functions are screened (module docs).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum XcScreen {
    /// Per grid batch with the CPU integrator's value test, products per batch (the default).
    Batch,
    /// Per block of batches by the rigorous shell bound, one product per block (the first
    /// implementation; kept for comparisons and the precision study's emulated GEMMs).
    Block,
}

/// The arithmetic of the XC products (densities X = phi D and the potential phi^T W).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum XcMath {
    /// FP64 (the default).
    Fp64,
    /// cuBLAS's fixed-point FP64 emulation with this many mantissa bits for the products
    /// (block screening's GEMMs; per-batch screening's products through cuBLAS's grouped
    /// GEMM). A precision-study variant (feature `study`).
    #[cfg(feature = "study")]
    Emulated { bits: i32 },
    /// FP32 values and products, each batch's sums in FP32 (over at most a batch of points
    /// or its functions), batches and points summed in FP64 (the precision study's variant b).
    Fp32,
    /// FP32 for the potential's products only; densities and the energy in FP64. A
    /// precision-study variant (feature `study`).
    #[cfg(feature = "study")]
    Fp32Potential,
}

impl XcMath {
    /// Whether the FP64 products are emulated (`Emulated`, feature `study`).
    pub fn emulated(self) -> bool {
        #[cfg(feature = "study")]
        if let XcMath::Emulated { .. } = self {
            return true;
        }
        false
    }

    /// Whether the potential's products run in FP32 (`Fp32`; `Fp32Potential`, feature
    /// `study`).
    pub fn fp32_potential(self) -> bool {
        #[cfg(feature = "study")]
        if self == XcMath::Fp32Potential {
            return true;
        }
        self == XcMath::Fp32
    }
}

#[cfg(feature = "gpu")]
impl XcMath {
    /// The arithmetic of the FP64 products' cuBLAS GEMMs: emulated FP64 with its bits, else FP64
    /// (the FP32 products name their own).
    pub fn cublas(self) -> linus_core::Arith {
        match self {
            #[cfg(feature = "study")]
            XcMath::Emulated { bits } => linus_core::Arith::Fp64Emulated {
                bits: bits.clamp(0, u8::MAX as i32) as u8,
            },
            _ => linus_core::Arith::Fp64,
        }
    }
}

/// What runs the per-batch FP64 products of the XC engines (the densities, the potential, the
/// gradient's and the Hessian's): a choice of the caller's device policy (linus decides it per
/// card class, decision D27), never of the engine.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ProductEngine {
    /// The variable-size batched DGEMM kernel (vgemm): the cards with little FP64 throughput.
    #[default]
    Vgemm,
    /// cuBLAS's grouped DGEMM, on the FP64 tensor cores of FP64-rich cards (17.7-18.8 against
    /// vgemm's 9.4-11.0 TFLOP/s on an H200).
    Grouped,
}

/// How a block's densities are formed (the block path's; feature `study`).
#[cfg(feature = "study")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DensityRoute {
    /// Per block, the route with fewer multiply-adds per function-point value: orbitals
    /// (ncomp nocc) or density matrix (nspin nf, times 4 for tau).
    Auto,
    /// psi = phi C_sub for every component, then sums over the occupied orbitals.
    Orbitals,
    /// X = phi_0 D_sub (and Y = grad phi D_sub for tau) per spin, then sums over the block's
    /// functions.
    DensityMatrix,
}

/// The semilocal ingredients of a functional.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct XcKind {
    /// No semilocal part (pure Hartree-Fock exchange).
    pub none: bool,
    pub gga: bool,
    /// Needs tau.
    pub mgga: bool,
}

impl XcKind {
    pub fn of(f: &Functional) -> XcKind {
        let fam = f.family();
        XcKind {
            none: fam == Family::Hf,
            gga: fam >= Family::Gga,
            mgga: fam == Family::Mgga && f.needs_tau(),
        }
    }

    /// Derivative order of the collocation.
    pub fn order(&self) -> usize {
        usize::from(self.gga || self.mgga)
    }
}

/// A block of consecutive grid batches and the shells that reach it.
#[derive(Clone, Debug)]
pub struct XcBlock {
    /// First point and number of points.
    pub p0: usize,
    pub np: usize,
    /// Kept shells (ascending) and the block-local number of each one's first function.
    pub shells: Vec<usize>,
    pub ao_local: Vec<usize>,
    /// The kept functions' global numbers (ascending).
    pub funcs: Vec<usize>,
}

/// Per shell of `basis`: its collocation cutoff radius at derivative `order` and threshold
/// `thresh`, linus-ints' `Collocator::new(basis, order, thresh)`'s `rcut` (which depends only
/// on the shell's angular momentum, form, exponents and coefficients), computed once per
/// distinct shell and in parallel on the current rayon pool: the same values.
pub fn shell_cutoffs(basis: &BasisSet, order: usize, thresh: f64) -> Vec<f64> {
    use rayon::prelude::*;
    use std::collections::HashMap;
    type Key = (usize, bool, Vec<u64>, Vec<u64>);
    let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<u64>>();
    let mut first: Vec<usize> = Vec::new();
    let mut map: HashMap<Key, usize> = HashMap::new();
    let of: Vec<usize> = basis
        .shells
        .iter()
        .enumerate()
        .map(|(i, sh)| {
            *map.entry((sh.l, sh.pure, bits(&sh.exps), bits(&sh.coefs)))
                .or_insert_with(|| {
                    first.push(i);
                    first.len() - 1
                })
        })
        .collect();
    let mut base = basis.clone();
    base.shells.clear();
    base.atom_shells.clear();
    let cut: Vec<f64> = first
        .par_iter()
        .map(|&i| {
            let mut one = base.clone();
            one.nbf = basis.shells[i].nfunc();
            one.shells = vec![basis.shells[i].clone()];
            Collocator::new(&one, order, thresh).shells[0].rcut
        })
        .collect();
    of.iter().map(|&u| cut[u]).collect()
}

/// The blocks of `grid` for `basis`, derivative order `order` and threshold `func_thresh`
/// (see the module docs). The grid's batches must cover its points in order. The blocks'
/// shells are found in parallel on the current rayon pool (each block on its own).
pub fn blocks(
    basis: &BasisSet,
    grid: &MolecularGrid,
    order: usize,
    func_thresh: f64,
    block_points: usize,
) -> Result<Vec<XcBlock>, crate::GpuError> {
    use rayon::prelude::*;
    let rcut = shell_cutoffs(basis, order, func_thresh);
    // the blocks' points: whole batches, at most `block_points` (a larger batch alone)
    let mut ranges: Vec<(usize, usize)> = Vec::new();
    let nb = grid.batches.len();
    let mut next = 0;
    let mut b = 0;
    while b < nb {
        let p0 = grid.batches[b].start;
        if p0 != next {
            return Err(crate::GpuError::Contract(
                "XC: the grid's batches do not cover its points in order".into(),
            ));
        }
        let mut e = b;
        let mut end = p0;
        while e < nb && (end == p0 || end - p0 + grid.batches[e].len() <= block_points) {
            if grid.batches[e].start != end {
                return Err(crate::GpuError::Contract(
                    "XC: the grid's batches do not cover its points in order".into(),
                ));
            }
            end = grid.batches[e].end;
            e += 1;
        }
        ranges.push((p0, end));
        next = end;
        b = e;
    }
    if next != grid.x.len() {
        return Err(crate::GpuError::Contract(
            "XC: the grid's batches do not cover all points".into(),
        ));
    }
    // per block: its bounding box, the shells whose cutoff sphere reaches it
    Ok(ranges
        .par_iter()
        .map(|&(p0, end)| {
            let (mut lo, mut hi) = ([f64::INFINITY; 3], [f64::NEG_INFINITY; 3]);
            for g in p0..end {
                let p = [grid.x[g], grid.y[g], grid.z[g]];
                for k in 0..3 {
                    lo[k] = lo[k].min(p[k]);
                    hi[k] = hi[k].max(p[k]);
                }
            }
            let (mut shells, mut ao_local, mut funcs) = (vec![], vec![], vec![]);
            for (s, sh) in basis.shells.iter().enumerate() {
                if end > p0 && Collocator::box_distance(&sh.center, &lo, &hi) < rcut[s] {
                    shells.push(s);
                    ao_local.push(funcs.len());
                    funcs.extend(sh.ao_start..sh.ao_start + sh.nfunc());
                }
            }
            XcBlock {
                p0,
                np: end - p0,
                shells,
                ao_local,
                funcs,
            }
        })
        .collect())
}

#[cfg(feature = "gpu")]
pub use device::*;

#[cfg(feature = "gpu")]
mod device {
    use super::*;
    use crate::blas::{GemmDims, MatMut, MatRef, Op};
    use crate::detsum::DetAcc;
    use crate::engine::Timeline;
    use crate::error::GpuError;
    use crate::kernels::colloc::{self, ShellTables};
    use crate::kernels::registry::KernelModule;
    use crate::linalg::{Arith, Gemm};
    use crate::xc::density::{DensityOps, DensityPass, DensityWork, GridDensities, PassTiming};
    use crate::xc::layout::Collocation;
    use crate::{Arg, DeviceBuf, Gpu, Kernel, Launch};
    use std::borrow::Cow;

    const KERNELS: &str = include_str!("../../device/xc/xc.cu");

    /// The engine's module: the fixed-point accumulator, then its kernels (also loaded by the XC
    /// gradient, `crate::grad`).
    pub(crate) const MODULE: KernelModule = KernelModule::new(
        "xc",
        || Cow::Owned(format!("{}\n{}", crate::detsum::DEVICE_SRC, KERNELS)),
        &[
            "xc_gather_rows",
            "xc_densities",
            "xc_gather_sub",
            "xc_scale_cols",
            "xc_densities_dm",
            "xc_sigma",
            "xc_energy",
            "xc_weights",
            "xc_weights_tau",
            "xc_scatter",
        ],
    );

    /// Occupied orbitals of one channel: C (nbf x ncol, column-major) and occupations, D =
    /// C diag(occ) C^T.
    #[derive(Clone, Copy)]
    pub struct XcChannel<'a> {
        pub c: &'a [f64],
        pub ncol: usize,
        pub occ: &'a [f64],
    }

    /// E_xc and V_xc per spin (nbf x nbf column-major; one matrix for a closed shell: the
    /// potential of the total density).
    pub struct GpuXcResult {
        pub exc: f64,
        pub v: Vec<Vec<f64>>,
    }

    /// Times of the last evaluation (seconds; device phases from events, the host's
    /// functional evaluation from the wall clock).
    #[derive(Clone, Debug, Default)]
    pub struct GpuXcCallTimes {
        pub total: f64,
        pub colloc: f64,
        /// psi GEMMs, row gathers and the density kernel.
        pub density: f64,
        /// Densities to the host, libxc, derivatives back (wall clock).
        pub functional: f64,
        /// Of which libxc alone (wall clock, all host threads).
        pub libxc: f64,
        /// Weights, V GEMMs and scatters.
        pub potential: f64,
        pub energy: f64,
    }

    /// Setup statistics.
    #[derive(Clone, Debug)]
    pub struct GpuXcStats {
        pub npts: usize,
        pub nblocks: usize,
        /// Function-point values per evaluation pass (sum over blocks of kept functions x
        /// points) and their share of the dense nbf x npts.
        pub values: usize,
        pub kept_fraction: f64,
        pub max_block_funcs: usize,
        pub setup_seconds: f64,
    }

    /// A block on the device (crate-internal: the response engine's XC kernel walks the same
    /// blocks).
    pub(crate) struct DevBlock {
        pub(crate) p0: usize,
        pub(crate) np: usize,
        pub(crate) nf: usize,
        /// The kept functions' global numbers.
        pub(crate) funcs: DeviceBuf<i32>,
        /// (colloc kernel index, shells, offset in the flat lists)
        groups: Vec<(usize, usize, usize)>,
    }

    /// The arithmetic of a product plan ([`Prod::new`]).
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub(crate) enum ProdKind {
        /// FP64 on the variable-size batched DGEMM kernel (vgemm).
        Vgemm,
        /// cuBLAS's grouped DGEMM in this FP64 arithmetic (native or emulated).
        Grouped64(Arith),
        /// cuBLAS's grouped SGEMM: the products' addresses point to f32 arrays.
        Grouped32,
    }

    impl ProdKind {
        /// Whether the products' operands are f32 arrays.
        pub(crate) fn single(self) -> bool {
            self == ProdKind::Grouped32
        }

        /// The FP64 products' kind on `engine`.
        pub(crate) fn fp64(engine: ProductEngine) -> ProdKind {
            match engine {
                ProductEngine::Vgemm => ProdKind::Vgemm,
                ProductEngine::Grouped => ProdKind::Grouped64(Arith::Fp64),
            }
        }
    }

    /// A product kind's plan for one chunk: the variable-size batched DGEMM, or cuBLAS's
    /// grouped GEMM (shapes and the device pointer arrays) in FP64 (native or emulated) or FP32.
    /// The plan holds its arithmetic: its addresses and its GEMM's element type cannot disagree.
    pub(crate) enum Prod {
        V(crate::kernels::vgemm::VgemmPlan),
        G64(Arith, Vec<crate::blas::GemmShape>, DeviceBuf<u64>),
        G32(Vec<crate::blas::GemmShape>, DeviceBuf<u64>),
        None,
    }

    impl Prod {
        /// A plan of `products` in the arithmetic `kind`.
        pub(crate) fn new(
            gpu: &Gpu,
            vg: &crate::kernels::vgemm::Vgemm,
            kind: ProdKind,
            products: &[crate::kernels::vgemm::VgemmProduct],
        ) -> Result<Prod, GpuError> {
            if products.is_empty() {
                return Ok(Prod::None);
            }
            // the grouped GEMM's FP64 arithmetic, or none for FP32
            let fp64 = match kind {
                ProdKind::Vgemm => return Ok(Prod::V(layout::plan(gpu, vg, products)?)),
                ProdKind::Grouped64(a) => Some(a),
                ProdKind::Grouped32 => None,
            };
            let fits = |x: usize| {
                i32::try_from(x).map_err(|_| GpuError::Contract("XC: product too large".into()))
            };
            let mut shapes = Vec::with_capacity(products.len());
            let mut ptr = Vec::with_capacity(3 * products.len());
            for p in products {
                shapes.push(crate::blas::GemmShape {
                    m: fits(p.m)?,
                    n: fits(p.n)?,
                    k: fits(p.k)?,
                    lda: fits(p.lda)?,
                    ldb: fits(p.ldb)?,
                    ldc: fits(p.ldc)?,
                });
            }
            ptr.extend(products.iter().map(|p| p.a));
            ptr.extend(products.iter().map(|p| p.b));
            ptr.extend(products.iter().map(|p| p.c));
            let ptr = gpu.upload(&ptr)?;
            Ok(match fp64 {
                Some(a) => Prod::G64(a, shapes, ptr),
                None => Prod::G32(shapes, ptr),
            })
        }

        /// Queues the products in the plan's arithmetic: C = op(A) op(B) + beta C (op(A) = A^T
        /// with `ta`).
        ///
        /// # Safety
        /// The plan's addresses must point to live arrays of its shapes (the engine's own
        /// workspaces, of the plan's element type).
        pub(crate) unsafe fn run(
            &self,
            gpu: &Gpu,
            vg: &crate::kernels::vgemm::Vgemm,
            gemm: &Gemm,
            ta: bool,
            beta: f64,
        ) -> Result<(), GpuError> {
            let op = if ta { Op::T } else { Op::N };
            // SAFETY: the caller's contract.
            unsafe {
                match self {
                    Prod::V(p) => vg.launch(gpu, ta, false, 1.0, beta, p),
                    Prod::G32(sh, ptr) => gemm.gemm_grouped(
                        Arith::Fp32,
                        (op, Op::N),
                        (1.0f32, beta as f32),
                        sh,
                        (ptr, 0),
                    ),
                    Prod::G64(a, sh, ptr) => {
                        gemm.gemm_grouped(*a, (op, Op::N), (1.0f64, beta), sh, (ptr, 0))
                    }
                    Prod::None => Ok(()),
                }
            }
        }
    }

    /// The products of one chunk: X = phi D, Y = grad phi D (tau), V = phi^T W, and V +=
    /// grad phi^T Wt (tau).
    struct ChunkPlans {
        x: Prod,
        y: Prod,
        v: Prod,
        vt: Prod,
    }

    /// The per-batch path: its layout, workspaces and product plans (the plans hold the
    /// workspaces' addresses; both live as long as the engine).
    pub(crate) struct Batched {
        pub(crate) lay: layout::XcLayout,
        plans: Vec<ChunkPlans>,
        /// Collocation of one block.
        pub(crate) phi: DeviceBuf<f64>,
        /// Compressed values, X/W, Y/Wt, D/V in FP64 and FP32 (as the arithmetic needs).
        pub(crate) phic: DeviceBuf<f64>,
        xw: DeviceBuf<f64>,
        yw: DeviceBuf<f64>,
        dv: DeviceBuf<f64>,
        pub(crate) phic32: DeviceBuf<f32>,
        xw32: DeviceBuf<f32>,
        yw32: DeviceBuf<f32>,
        dv32: DeviceBuf<f32>,
        /// The grouped products' GEMMs.
        gemm: Gemm,
    }

    impl Batched {
        /// Whether the compressed values are kept in FP32 too (an FP32 arithmetic).
        pub(crate) fn has_single(&self) -> bool {
            self.phic32.len() > 1 || self.lay.max_x == 0
        }

        /// The layout of `grid` and its workspaces and plans for `math`; collocation and
        /// screening to `order` (values and gradients for GGA and meta-GGA).
        pub(crate) fn new(
            gpu: &Gpu,
            basis: &BasisSet,
            grid: &MolecularGrid,
            order: usize,
            mgga: bool,
            opts: &GpuXcOptions,
        ) -> Result<Batched, GpuError> {
            let nc = crate::cpu::ncomp(order);
            // the budget in doubles of compressed values: phic and its FP32 copy, X/W, Y/Wt
            let per_value = 8 * nc + 4 * nc + 8 + 4 + if mgga { 36 } else { 0 };
            let budget = (opts.workspace / per_value).max(1) * nc;
            let lay = layout::XcLayout::new(
                gpu,
                basis,
                grid,
                &layout::XcLayoutSpec {
                    screen_order: order,
                    order,
                    thresh: opts.func_thresh,
                    block_points: opts.block_points,
                    budget,
                },
            )?;
            let math = opts.math;
            let (d64, d32) = (!matches!(math, XcMath::Fp32), math.fp32_potential());
            let sz = |n: usize, on: bool| if on { n.max(1) } else { 1 };
            let phi = gpu.alloc_zeros::<f64>((lay.max_block_nf * nc * lay.max_block_np).max(1))?;
            let phic = gpu.alloc_zeros::<f64>((nc * lay.max_x).max(1))?;
            let xw = gpu.alloc_zeros::<f64>(sz(lay.max_x, d64))?;
            let yw = gpu.alloc_zeros::<f64>(sz(3 * lay.max_x, d64 && mgga))?;
            let dv = gpu.alloc_zeros::<f64>(lay.max_d.max(1))?;
            let phic32 = gpu.alloc_zeros::<f32>(sz(nc * lay.max_x, d32))?;
            let xw32 = gpu.alloc_zeros::<f32>(sz(lay.max_x, d32))?;
            let yw32 = gpu.alloc_zeros::<f32>(sz(3 * lay.max_x, d32 && mgga))?;
            let dv32 = gpu.alloc_zeros::<f32>(sz(lay.max_d, d32))?;
            let vg = &lay.vg;
            let mut gemm = Gemm::new(gpu)?;
            if math.emulated() {
                gemm.prepare(gpu, math.cublas())?;
            }
            // the arithmetic of the density and potential products
            let fp64 = ProdKind::fp64(opts.products);
            let (kd, kv) = match math {
                XcMath::Fp64 => (fp64, fp64),
                #[cfg(feature = "study")]
                XcMath::Emulated { .. } => (
                    ProdKind::Grouped64(math.cublas()),
                    ProdKind::Grouped64(math.cublas()),
                ),
                XcMath::Fp32 => (ProdKind::Grouped32, ProdKind::Grouped32),
                #[cfg(feature = "study")]
                XcMath::Fp32Potential => (fp64, ProdKind::Grouped32),
            };
            let a64 = |b: &DeviceBuf<f64>, off: usize| b.at(off).addr();
            let a32 = |b: &DeviceBuf<f32>, off: usize| b.at(off).addr();
            let mut plans = Vec::with_capacity(lay.chunks.len());
            for c in &lay.chunks {
                use crate::kernels::vgemm::VgemmProduct as P;
                // operand addresses in the arithmetic `k`
                let addr = |k: ProdKind, which: u8, off: usize| -> u64 {
                    match (k.single(), which) {
                        (false, 0) => a64(&phic, off),
                        (false, 1) => a64(&xw, off),
                        (false, 2) => a64(&yw, off),
                        (false, _) => a64(&dv, off),
                        (true, 0) => a32(&phic32, off),
                        (true, 1) => a32(&xw32, off),
                        (true, 2) => a32(&yw32, off),
                        (true, _) => a32(&dv32, off),
                    }
                };
                let x = lay.products(c, |h, [po, xo, dd]| {
                    let (np, nf) = (h.np, h.global.len());
                    Some(P {
                        m: np,
                        n: nf,
                        k: nf,
                        a: addr(kd, 0, po),
                        lda: nc * np,
                        b: addr(kd, 3, dd),
                        ldb: nf,
                        c: addr(kd, 1, xo),
                        ldc: np,
                    })
                });
                let y = if mgga {
                    lay.products(c, |h, [po, xo, dd]| {
                        let (np, nf) = (h.np, h.global.len());
                        Some(P {
                            m: 3 * np,
                            n: nf,
                            k: nf,
                            a: addr(kd, 0, po + np),
                            lda: nc * np,
                            b: addr(kd, 3, dd),
                            ldb: nf,
                            c: addr(kd, 2, 3 * xo),
                            ldc: 3 * np,
                        })
                    })
                } else {
                    vec![]
                };
                let v = lay.products(c, |h, [po, xo, dd]| {
                    let (np, nf) = (h.np, h.global.len());
                    Some(P {
                        m: nf,
                        n: nf,
                        k: np,
                        a: addr(kv, 0, po),
                        lda: nc * np,
                        b: addr(kv, 1, xo),
                        ldb: np,
                        c: addr(kv, 3, dd),
                        ldc: nf,
                    })
                });
                let vt = if mgga {
                    lay.products(c, |h, [po, xo, dd]| {
                        let (np, nf) = (h.np, h.global.len());
                        Some(P {
                            m: nf,
                            n: nf,
                            k: 3 * np,
                            a: addr(kv, 0, po + np),
                            lda: nc * np,
                            b: addr(kv, 2, 3 * xo),
                            ldb: 3 * np,
                            c: addr(kv, 3, dd),
                            ldc: nf,
                        })
                    })
                } else {
                    vec![]
                };
                plans.push(ChunkPlans {
                    x: Prod::new(gpu, vg, kd, &x)?,
                    y: Prod::new(gpu, vg, kd, &y)?,
                    v: Prod::new(gpu, vg, kv, &v)?,
                    vt: Prod::new(gpu, vg, kv, &vt)?,
                });
            }
            Ok(Batched {
                lay,
                plans,
                phi,
                phic,
                xw,
                yw,
                dv,
                phic32,
                xw32,
                yw32,
                dv32,
                gemm,
            })
        }
    }

    /// The `xc` module's kernels of the engine (the block path's behind `study`).
    struct Kernels {
        #[cfg(feature = "study")]
        gather: Kernel,
        #[cfg(feature = "study")]
        densities: Kernel,
        #[cfg(feature = "study")]
        gather_sub: Kernel,
        scale_cols: Kernel,
        #[cfg(feature = "study")]
        densities_dm: Kernel,
        sigma: Kernel,
        energy: Kernel,
        #[cfg(feature = "study")]
        weights: Kernel,
        #[cfg(feature = "study")]
        weights_tau: Kernel,
        #[cfg(feature = "study")]
        scatter: Kernel,
    }

    /// XC energy and potential on the device (see the module docs).
    ///
    /// The fields marked `pub(crate)` are read by the response engine's XC kernel
    /// (`crate::response`), which walks the same blocks with the same collocation.
    pub struct GpuXc {
        pub(crate) nbf: usize,
        pub(crate) npts: usize,
        /// The functional on the host and the ingredients it is evaluated with.
        pub(crate) functional: HostFunctional,
        /// Collocation components per function (1, or 4 with gradients).
        pub(crate) ncomp: usize,
        pub(crate) x: DeviceBuf<f64>,
        pub(crate) y: DeviceBuf<f64>,
        pub(crate) z: DeviceBuf<f64>,
        /// Quadrature weights.
        pub(crate) w: DeviceBuf<f64>,
        tables: ShellTables,
        colloc: Vec<Kernel>,
        lists: DeviceBuf<i32>,
        aos: DeviceBuf<i32>,
        pub(crate) blocks: Vec<DevBlock>,
        pub(crate) max_nf: usize,
        pub(crate) max_np: usize,
        kernels: Kernels,
        /// The block path's GEMMs, in `gemm_arith`.
        gemm: Gemm,
        gemm_arith: Arith,
        #[cfg(feature = "study")]
        route: DensityRoute,
        math: XcMath,
        /// The per-batch path (per-batch screening).
        pub(crate) bat: Option<Batched>,
        stats: GpuXcStats,
        pub last: GpuXcCallTimes,
    }

    /// One V_xc evaluation's inputs (checked by [`GpuXc::compute_device`]): the occupied
    /// orbitals of the channels side by side, their counts and occupations, and the
    /// pointwise functional in place of libxc when given.
    pub(crate) struct VxcCall<'a, 'f> {
        pub(crate) cmo: &'a DeviceBuf<f64>,
        pub(crate) ncols: &'a [usize],
        pub(crate) occ: &'a DeviceBuf<f64>,
        pub(crate) restricted: bool,
        pub(crate) func: Option<&'a mut PointFn<'f>>,
    }

    // the precision studies' block path of V_xc
    #[cfg(feature = "study")]
    mod study;

    #[cfg(not(feature = "study"))]
    impl GpuXc {
        /// The block path of V_xc is a precision-study variant (feature `study`); block
        /// screening otherwise serves the CPKS kernel only.
        fn compute_blocks(
            &mut self,
            _gpu: &Gpu,
            _call: VxcCall<'_, '_>,
            _v: &mut DeviceBuf<f64>,
        ) -> Result<f64, GpuError> {
            Err(GpuError::NotCovered(
                "XC: the block path of the energy and potential (XcScreen::Block) is a \
                 precision-study variant: build linus-gpu with the `study` feature"
                    .into(),
            ))
        }
    }

    impl GpuXc {
        /// The functional `name` (a linus-xc name) on `grid` for `basis`.
        pub fn new(
            gpu: &Gpu,
            basis: &BasisSet,
            grid: &MolecularGrid,
            name: &str,
            opts: &GpuXcOptions,
        ) -> Result<GpuXc, GpuError> {
            Self::with_kind(gpu, basis, grid, name, None, opts)
        }

        /// As [`GpuXc::new`], with the semilocal ingredients `kind` in place of the
        /// functional's (crate-internal: VV10's engine on the NLC grid is a GGA of the total
        /// density whatever the functional's own family).
        pub(crate) fn with_kind(
            gpu: &Gpu,
            basis: &BasisSet,
            grid: &MolecularGrid,
            name: &str,
            kind: Option<XcKind>,
            opts: &GpuXcOptions,
        ) -> Result<GpuXc, GpuError> {
            let t0 = std::time::Instant::now();
            let functional = HostFunctional::new(name, kind)?;
            if functional.needs_lapl() && kind.is_none() {
                return Err(GpuError::NotCovered(format!(
                    "{name}: Laplacian-dependent meta-GGAs are not implemented"
                )));
            }
            let kind = functional.kind();
            let order = kind.order();
            let ncomp = crate::cpu::ncomp(order);
            let bl = blocks(
                basis,
                grid,
                order,
                opts.func_thresh,
                opts.block_points.max(1),
            )?;
            // collocation kernels, one per (l, pure) of the basis
            let src = colloc::module_source(basis, order);
            let mut keys: Vec<(usize, bool)> = basis.shells.iter().map(|s| (s.l, s.pure)).collect();
            keys.sort();
            keys.dedup();
            let colloc_k = keys
                .iter()
                .map(|&(l, p)| gpu.kernel("colloc", &src, &colloc::kernel_name(l, p, order)))
                .collect::<Result<Vec<_>, _>>()?;
            let key_of = |s: usize| {
                let sh = &basis.shells[s];
                keys.iter().position(|k| *k == (sh.l, sh.pure)).unwrap()
            };
            // flat shell lists per block and (l, pure)
            let (mut lists, mut aos) = (Vec::<i32>::new(), Vec::<i32>::new());
            let mut dev = Vec::with_capacity(bl.len());
            let (mut max_nf, mut max_np, mut values) = (0, 0, 0usize);
            for b in &bl {
                let mut groups = Vec::new();
                for (ki, _) in keys.iter().enumerate() {
                    let off = lists.len();
                    for (s, &a) in b.shells.iter().zip(&b.ao_local) {
                        if key_of(*s) == ki {
                            lists.push(*s as i32);
                            aos.push(a as i32);
                        }
                    }
                    if lists.len() > off {
                        groups.push((ki, lists.len() - off, off));
                    }
                }
                let f: Vec<i32> = b.funcs.iter().map(|&f| f as i32).collect();
                max_nf = max_nf.max(b.funcs.len());
                max_np = max_np.max(b.np);
                values += b.funcs.len() * b.np;
                dev.push(DevBlock {
                    p0: b.p0,
                    np: b.np,
                    nf: b.funcs.len(),
                    funcs: gpu.upload(if f.is_empty() { &[0] } else { &f })?,
                    groups,
                });
            }
            let src = MODULE.source();
            let k = |e: &str| gpu.kernel(MODULE.name, &src, e);
            let kernels = Kernels {
                #[cfg(feature = "study")]
                gather: k("xc_gather_rows")?,
                #[cfg(feature = "study")]
                densities: k("xc_densities")?,
                #[cfg(feature = "study")]
                gather_sub: k("xc_gather_sub")?,
                scale_cols: k("xc_scale_cols")?,
                #[cfg(feature = "study")]
                densities_dm: k("xc_densities_dm")?,
                sigma: k("xc_sigma")?,
                energy: k("xc_energy")?,
                #[cfg(feature = "study")]
                weights: k("xc_weights")?,
                #[cfg(feature = "study")]
                weights_tau: k("xc_weights_tau")?,
                #[cfg(feature = "study")]
                scatter: k("xc_scatter")?,
            };
            let npts = grid.x.len();
            let bat = match opts.screen {
                XcScreen::Batch => {
                    if opts.math.emulated() {
                        // cuBLAS 13.2 runs grouped GEMMs natively in an emulated mode, and one
                        // emulated GEMM per batch is 10-40x slower than the batched kernel
                        return Err(GpuError::NotCovered(
                            "XC: FP64 emulation needs block screening (cuBLAS does not emulate grouped GEMMs)".into(),
                        ));
                    }
                    Some(Batched::new(gpu, basis, grid, order, kind.mgga, opts)?)
                }
                XcScreen::Block => {
                    if opts.math.fp32_potential() {
                        return Err(GpuError::NotCovered(
                            "XC: FP32 products need per-batch screening".into(),
                        ));
                    }
                    None
                }
            };
            let values = bat.as_ref().map_or(values, |b| b.lay.values);
            let stats = GpuXcStats {
                npts,
                nblocks: dev.len(),
                values,
                kept_fraction: values as f64 / (basis.nbf * npts).max(1) as f64,
                max_block_funcs: max_nf,
                setup_seconds: t0.elapsed().as_secs_f64(),
            };
            // the block path's GEMMs: emulated with block screening and an emulated arithmetic
            let gemm_arith = match (opts.screen, opts.math) {
                (XcScreen::Block, m) if m.emulated() => opts.math.cublas(),
                _ => Arith::Fp64,
            };
            let mut gemm = Gemm::new(gpu)?;
            if gemm_arith != Arith::Fp64 {
                gemm.prepare(gpu, gemm_arith)?;
            }
            Ok(GpuXc {
                nbf: basis.nbf,
                npts,
                functional,
                ncomp,
                x: gpu.upload(&grid.x)?,
                y: gpu.upload(&grid.y)?,
                z: gpu.upload(&grid.z)?,
                w: gpu.upload(&grid.w)?,
                tables: ShellTables::new(gpu, basis)?,
                colloc: colloc_k,
                lists: gpu.upload(if lists.is_empty() { &[0] } else { &lists })?,
                aos: gpu.upload(if aos.is_empty() { &[0] } else { &aos })?,
                blocks: dev,
                max_nf,
                max_np,
                kernels,
                gemm,
                gemm_arith,
                #[cfg(feature = "study")]
                route: opts.route,
                math: opts.math,
                bat,
                stats,
                last: GpuXcCallTimes::default(),
            })
        }

        pub fn stats(&self) -> &GpuXcStats {
            &self.stats
        }

        pub fn kind(&self) -> XcKind {
            self.functional.kind()
        }

        /// The functional's VV10 parameters, if it has a nonlocal term (not evaluated here).
        pub fn vv10(&self) -> Option<linus_xc::Vv10> {
            self.functional.vv10()
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

        /// The block's kept functions on its points into `phi` (function-major:
        /// phi\[(f ncomp + c) np + g\]); crate-internal for the response engine.
        pub(crate) fn collocate(
            &self,
            gpu: &Gpu,
            b: &DevBlock,
            phi: &mut DeviceBuf<f64>,
        ) -> Result<(), GpuError> {
            self.collocate_at(gpu, b, phi, 0)
        }

        /// [`GpuXc::collocate`] into `phi` from element `offset` on (checked to hold the
        /// block's nf ncomp np values).
        pub(crate) fn collocate_at(
            &self,
            gpu: &Gpu,
            b: &DevBlock,
            phi: &mut DeviceBuf<f64>,
            offset: usize,
        ) -> Result<(), GpuError> {
            if offset + b.nf * self.ncomp * b.np > phi.len() {
                return Err(GpuError::Contract("XC collocation: output outside".into()));
            }
            let at = |buf: &DeviceBuf<f64>| buf.at(b.p0);
            let out = phi.at(offset);
            for &(ki, nsh, off) in &b.groups {
                // SAFETY: the block's shells of this kernel's (l, pure) with their local
                // first functions (< nf); the output holds nf ncomp np values from `offset`
                // (checked).
                unsafe {
                    self.tables.launch(
                        gpu,
                        &self.colloc[ki],
                        nsh,
                        self.lists.at(off),
                        self.aos.at(off),
                        [at(&self.x), at(&self.y), at(&self.z)],
                        b.np,
                        self.ncomp * b.np,
                        b.np,
                        out,
                    )
                }?;
            }
            Ok(())
        }

        /// E_xc and V_xc of the occupied orbitals `ch` (one channel for a closed shell with
        /// `restricted`, else alpha and beta), host matrices: the orbitals go to the device,
        /// [`GpuXc::compute_device`] runs, the potential comes back.
        pub fn compute(
            &mut self,
            gpu: &Gpu,
            ch: &[XcChannel],
            restricted: bool,
        ) -> Result<GpuXcResult, GpuError> {
            let nbf = self.nbf;
            let nsp = if restricted { 1 } else { 2 };
            if ch.len() != nsp {
                return Err(GpuError::Contract(format!(
                    "XC: {} channels of orbitals for {nsp} spins",
                    ch.len()
                )));
            }
            for c in ch {
                if c.c.len() != nbf * c.ncol || c.occ.len() != c.ncol {
                    return Err(GpuError::Contract(
                        "XC: orbitals or occupations of the wrong size".into(),
                    ));
                }
            }
            let n2 = nbf * nbf;
            if self.kind().none {
                return Ok(GpuXcResult {
                    exc: 0.0,
                    v: vec![vec![0.0; n2]; nsp],
                });
            }
            let m0 = gpu.mark()?;
            let ntot: usize = ch.iter().map(|c| c.ncol).sum();
            let mut cflat = Vec::with_capacity(nbf * ntot);
            let mut occ = Vec::with_capacity(ntot);
            for c in ch {
                cflat.extend_from_slice(c.c);
                occ.extend_from_slice(c.occ);
            }
            let cdev = gpu.upload(if cflat.is_empty() { &[0.0] } else { &cflat })?;
            let occd = gpu.upload(if occ.is_empty() { &[0.0] } else { &occ })?;
            let ncols: Vec<usize> = ch.iter().map(|c| c.ncol).collect();
            let mut v = gpu.alloc_zeros::<f64>(nsp * n2)?;
            let exc = self.compute_device(gpu, &cdev, &ncols, Some(&occd), restricted, &mut v)?;
            let hv = gpu.download(&v)?;
            self.last.total = m0.seconds_until(&gpu.mark()?)?;
            Ok(GpuXcResult {
                exc,
                v: hv.chunks(n2).map(|c| c.to_vec()).collect(),
            })
        }

        /// E_xc (returned) and V_xc (into `v`: one nbf x nbf column-major matrix per spin,
        /// overwritten; exactly symmetric) of occupied orbitals on the device: `cmo` holds the
        /// channels side by side (nbf x sum(ncols), column-major; one channel for a closed shell
        /// with `restricted`, else alpha and beta), `occ` their occupations (None: all 1). No
        /// nbf-sized host transfer: the grid's densities go to the host for the functional
        /// (linus-xc) and its derivatives come back, and E_xc is read back (one synchronization).
        pub fn compute_device(
            &mut self,
            gpu: &Gpu,
            cmo: &DeviceBuf<f64>,
            ncols: &[usize],
            occ: Option<&DeviceBuf<f64>>,
            restricted: bool,
            v: &mut DeviceBuf<f64>,
        ) -> Result<f64, GpuError> {
            self.compute_device_with(gpu, cmo, ncols, occ, restricted, v, None)
        }

        /// As [`GpuXc::compute_device`], with the pointwise functional `func` in place of
        /// libxc when given (crate-internal: VV10).
        #[allow(clippy::too_many_arguments)]
        pub(crate) fn compute_device_with(
            &mut self,
            gpu: &Gpu,
            cmo: &DeviceBuf<f64>,
            ncols: &[usize],
            occ: Option<&DeviceBuf<f64>>,
            restricted: bool,
            v: &mut DeviceBuf<f64>,
            func: Option<&mut PointFn<'_>>,
        ) -> Result<f64, GpuError> {
            let nbf = self.nbf;
            let n2 = nbf * nbf;
            let nsp = if restricted { 1 } else { 2 };
            if ncols.len() != nsp {
                return Err(GpuError::Contract(format!(
                    "XC: {} channels of orbitals for {nsp} spins",
                    ncols.len()
                )));
            }
            let ntot: usize = ncols.iter().sum();
            if cmo.len() < nbf * ntot || occ.is_some_and(|o| o.len() < ntot) || v.len() < nsp * n2 {
                return Err(GpuError::Contract(
                    "XC: a device buffer is too small".into(),
                ));
            }
            if self.kind().none {
                gpu.zero_range(v, 0, nsp * n2)?;
                return Ok(0.0);
            }
            let ones;
            let occd = match occ {
                Some(o) => o,
                None => {
                    ones = gpu.upload(&vec![1.0; ntot.max(1)])?;
                    &ones
                }
            };
            let call = VxcCall {
                cmo,
                ncols,
                occ: occd,
                restricted,
                func,
            };
            if self.bat.is_some() {
                self.compute_batched(gpu, call, v)
            } else {
                self.compute_blocks(gpu, call, v)
            }
        }

        /// The per-batch path of [`GpuXc::compute_device`] (arguments checked there): per
        /// chunk of batches, each block collocated and its batches' kept functions compressed,
        /// then the products of all the chunk's batches in one launch each; the potential of
        /// every batch into a fixed-point accumulator.
        fn compute_batched(
            &mut self,
            gpu: &Gpu,
            call: VxcCall<'_, '_>,
            v: &mut DeviceBuf<f64>,
        ) -> Result<f64, GpuError> {
            let VxcCall {
                cmo,
                ncols,
                occ: occd,
                restricted,
                func,
            } = call;
            let (nbf, npts) = (self.nbf, self.npts);
            let n2 = nbf * nbf;
            let nsp = ncols.len();
            let ntot: usize = ncols.iter().sum();
            let mut times = GpuXcCallTimes::default();
            let m0 = gpu.mark()?;
            let (gga, mgga) = (self.kind().gga, self.kind().mgga);
            let single_d = matches!(self.math, XcMath::Fp32);
            let single_v = self.math.fp32_potential();
            let mut gd = GridDensities::new(gpu, npts, nsp, self.kind())?;
            // D_s = C_s diag(occ_s) C_s^T
            let mut d = gpu.alloc_zeros::<f64>(nsp * n2)?;
            if ntot > 0 {
                let mut cs = gpu.alloc_zeros::<f64>(nbf * ntot)?;
                // SAFETY: matches xc_scale_cols; nbf ntot values in and out.
                unsafe {
                    gpu.launch(
                        &self.kernels.scale_cols,
                        Launch::linear(nbf * ntot, 256),
                        &[
                            Arg::I64(nbf as i64),
                            Arg::I32(ntot as i32),
                            Arg::buf(cmo),
                            Arg::buf(occd),
                            Arg::buf_mut(&mut cs),
                        ],
                    )
                }?;
                let mut c0 = 0;
                for (sp, &ncol) in ncols.iter().enumerate() {
                    if ncol > 0 {
                        self.gemm.gemm(
                            self.gemm_arith,
                            GemmDims::new(Op::N, Op::T, nbf, nbf, ncol),
                            (1.0, 0.0),
                            MatRef::new(&cs, c0 * nbf, nbf),
                            MatRef::new(cmo, c0 * nbf, nbf),
                            MatMut::new(&mut d, sp * n2, nbf),
                        )?;
                    }
                    c0 += ncol;
                }
            }
            let mut tl = Timeline::<GpuXcCallTimes>::new(gpu);
            let xyz = [&self.x, &self.y, &self.z];
            let w = &self.w;
            let bat = self.bat.as_mut().expect("per-batch path");
            let nc = bat.lay.ncomp;
            let any32 = single_d || single_v;
            let one_chunk = bat.lay.chunks.len() == 1;
            let a64 = |b: &DeviceBuf<f64>, on: bool| if on { Arg::buf(b) } else { Arg::Ptr(0) };
            let a32 = |b: &DeviceBuf<f32>, on: bool| if on { Arg::buf(b) } else { Arg::Ptr(0) };
            let nchunks = bat.lay.chunks.len();

            // phase 1: densities
            let pass = DensityPass {
                lay: &bat.lay,
                colloc: Collocation::Own,
                xyz,
                d: (ntot > 0).then_some(&d),
                nbf,
                nsp,
                scale: if restricted { 2.0 } else { 1.0 },
                kind: self.functional.kind(),
                gemm: &bat.gemm,
            };
            let mut work = DensityWork {
                phi: &mut bat.phi,
                phic: &mut bat.phic,
                phic32: if any32 { Some(&mut bat.phic32) } else { None },
                ops: if single_d {
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
            let plans = &bat.plans;
            pass.run(
                gpu,
                &mut work,
                gd.out(),
                Some(PassTiming {
                    tl: &mut tl,
                    colloc: |t| &mut t.colloc,
                    density: |t| &mut t.density,
                }),
                |ci| [&plans[ci].x, &plans[ci].y],
                |_, _| Ok(()),
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
            let inp = self.point_inputs(restricted, &rho, &grad);
            let (pt, ptimes) = self.functional.pointwise(gpu, &inp, &sigma, &tau, func)?;
            (times.functional, times.libxc) = (ptimes.functional, ptimes.libxc);
            let (dzk, dvrho, dvsig, dvtau) = (pt.zk, pt.vrho, pt.vsigma, pt.vtau);

            // energy (fixed point)
            let me0 = gpu.mark()?;
            let mut eacc = DetAcc::new(gpu, 1)?;
            // SAFETY: matches xc_energy; reads npts points of w, rho, zk; adds to acc[0].
            unsafe {
                gpu.launch(
                    &self.kernels.energy,
                    Launch::linear(npts, 256),
                    &[
                        Arg::I64(npts as i64),
                        Arg::I32(nsp as i32),
                        Arg::buf(&self.w),
                        Arg::buf(&rho),
                        Arg::buf(&dzk),
                        eacc.arg(),
                    ],
                )
            }?;
            let exc = eacc.to_f64(gpu)?[0];
            times.energy = me0.seconds_until(&gpu.mark()?)?;

            // phase 2: the potential, every batch's V_b + V_b^T into a fixed-point accumulator
            let xyz = [&self.x, &self.y, &self.z];
            let bat = self.bat.as_mut().expect("per-batch path");
            let mut acc: Vec<DetAcc> = (0..nsp)
                .map(|_| DetAcc::new(gpu, n2))
                .collect::<Result<_, _>>()?;
            for ci in 0..nchunks {
                let c = bat.lay.chunks[ci].clone();
                let nbj = c.j1 - c.j0;
                if nbj == 0 {
                    continue;
                }
                let s = tl.mark()?;
                if !one_chunk {
                    bat.lay.prepare(
                        gpu,
                        &c,
                        Collocation::Own,
                        xyz,
                        &mut bat.phi,
                        Some(&mut bat.phic),
                        if any32 { Some(&mut bat.phic32) } else { None },
                    )?;
                }
                let s1 = tl.close(|t| &mut t.colloc, s)?;
                let (maxd, maxx) = bat.lay.chunk_max(&c);
                for (sp, acc) in acc.iter_mut().enumerate() {
                    // SAFETY: matches xcb_weights; reads the batches' points of the grid arrays
                    // and their compressed values, writes each batch's np nf weights at its
                    // offset in the chunk's workspace.
                    unsafe {
                        gpu.launch(
                            &bat.lay.k.weights,
                            layout::per_batch(maxx, nbj, 256),
                            &[
                                Arg::I32(nc as i32),
                                Arg::I32(nsp as i32),
                                Arg::I32(sp as i32),
                                Arg::I32(gga as i32),
                                Arg::buf(&bat.lay.bd),
                                Arg::I32(c.j0 as i32),
                                Arg::buf(w),
                                Arg::buf(&dvrho),
                                Arg::buf(&dvsig),
                                Arg::buf(&grad),
                                Arg::buf(&bat.phic),
                                a64(&bat.xw, !single_v),
                                a32(&bat.xw32, single_v),
                            ],
                        )
                    }?;
                    if mgga {
                        // SAFETY: matches xcb_weights_tau; as xcb_weights, 3 np nf values.
                        unsafe {
                            gpu.launch(
                                &bat.lay.k.weights_tau,
                                layout::per_batch(maxx, nbj, 256),
                                &[
                                    Arg::I32(nc as i32),
                                    Arg::I32(nsp as i32),
                                    Arg::I32(sp as i32),
                                    Arg::buf(&bat.lay.bd),
                                    Arg::I32(c.j0 as i32),
                                    Arg::buf(w),
                                    Arg::buf(&dvtau),
                                    Arg::buf(&bat.phic),
                                    a64(&bat.yw, !single_v),
                                    a32(&bat.yw32, single_v),
                                ],
                            )
                        }?;
                    }
                    // SAFETY: the plans address the engine's workspaces of this chunk.
                    unsafe {
                        bat.plans[ci]
                            .v
                            .run(gpu, &bat.lay.vg, &bat.gemm, true, 0.0)?;
                        bat.plans[ci]
                            .vt
                            .run(gpu, &bat.lay.vg, &bat.gemm, true, 1.0)?;
                    }
                    let nacc = acc.len();
                    // SAFETY: matches xcb_vscatter; reads each batch's nf^2 potential, adds to
                    // the accumulator's elements at its functions (< nbf).
                    unsafe {
                        gpu.launch(
                            &bat.lay.k.vscatter,
                            layout::per_batch(maxd, nbj, 256),
                            &[
                                Arg::I64(nbf as i64),
                                Arg::buf(&bat.lay.bd),
                                Arg::I32(c.j0 as i32),
                                Arg::buf(&bat.lay.gf),
                                a64(&bat.dv, !single_v),
                                a32(&bat.dv32, single_v),
                                acc.arg(),
                                Arg::I64(nacc as i64),
                            ],
                        )
                    }?;
                }
                tl.close(|t| &mut t.potential, s1)?;
            }
            let s = tl.mark()?;
            for (sp, acc) in acc.iter_mut().enumerate() {
                // SAFETY: matches xcb_vfinal; reads the nbf^2 accumulator, writes spin sp's
                // nbf^2 potential (v holds nsp n2, checked by the caller).
                unsafe {
                    gpu.launch(
                        &bat.lay.k.vfinal,
                        Launch::linear(n2, 256),
                        &[Arg::I64(nbf as i64), acc.arg(), Arg::from(v.at(sp * n2))],
                    )
                }?;
            }
            tl.close(|t| &mut t.potential, s)?;
            let me = gpu.mark()?;
            tl.add_to(&mut times)?;
            times.total = m0.seconds_until(&me)?;
            self.last = times;
            Ok(exc)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cpu::GridSpec;
    use crate::harness;

    /// The blocks in one serial pass with linus-ints' `Collocator` (the reference [`blocks`]
    /// reproduces with per-shell cutoffs computed once per distinct shell, blocks in parallel).
    fn blocks_serial(
        basis: &BasisSet,
        grid: &MolecularGrid,
        order: usize,
        func_thresh: f64,
        block_points: usize,
    ) -> Vec<XcBlock> {
        let coll = Collocator::new(basis, order, func_thresh);
        let mut out = Vec::new();
        let nb = grid.batches.len();
        let mut b = 0;
        while b < nb {
            let p0 = grid.batches[b].start;
            let mut e = b;
            let mut end = p0;
            while e < nb && (end == p0 || end - p0 + grid.batches[e].len() <= block_points) {
                end = grid.batches[e].end;
                e += 1;
            }
            let (mut lo, mut hi) = ([f64::INFINITY; 3], [f64::NEG_INFINITY; 3]);
            for g in p0..end {
                let p = [grid.x[g], grid.y[g], grid.z[g]];
                for k in 0..3 {
                    lo[k] = lo[k].min(p[k]);
                    hi[k] = hi[k].max(p[k]);
                }
            }
            let (mut shells, mut ao_local, mut funcs) = (vec![], vec![], vec![]);
            for (s, sc) in coll.shells.iter().enumerate() {
                if end > p0 && Collocator::box_distance(&sc.center, &lo, &hi) < sc.rcut {
                    shells.push(s);
                    ao_local.push(funcs.len());
                    funcs.extend(sc.ao_start..sc.ao_start + sc.nf);
                }
            }
            out.push(XcBlock {
                p0,
                np: end - p0,
                shells,
                ao_local,
                funcs,
            });
            b = e;
        }
        out
    }

    /// Per-shell cutoffs equal the `Collocator`'s bit for bit (orders 0 to 3, two thresholds),
    /// and the blocks equal the serial reference's, on a molecule with repeated elements.
    #[test]
    fn blocks_match_the_serial_reference() {
        let mol = harness::structure("la18").unwrap();
        let basis = harness::basis(&mol, "def2-tzvp").unwrap();
        let grid = harness::grid(&mol, &GridSpec::pyscf(1)).unwrap();
        for order in 0..=3 {
            for thresh in [1e-15, 1e-10] {
                let coll = Collocator::new(&basis, order, thresh);
                let cut = shell_cutoffs(&basis, order, thresh);
                assert_eq!(cut.len(), coll.shells.len());
                for (c, sc) in cut.iter().zip(&coll.shells) {
                    assert_eq!(
                        c.to_bits(),
                        sc.rcut.to_bits(),
                        "order {order} thresh {thresh}"
                    );
                }
            }
        }
        for (order, thresh, points) in [(1, 1e-12, 2048), (2, 1e-15, 2048), (2, 1e-15, 512)] {
            let a = blocks(&basis, &grid, order, thresh, points).unwrap();
            let b = blocks_serial(&basis, &grid, order, thresh, points);
            assert_eq!(a.len(), b.len());
            for (x, y) in a.iter().zip(&b) {
                assert_eq!(
                    (x.p0, x.np, &x.shells, &x.ao_local, &x.funcs),
                    (y.p0, y.np, &y.shells, &y.ao_local, &y.funcs)
                );
            }
        }
    }

    /// A product plan runs in the arithmetic it was built for (its operands' element type): the
    /// plan holds it, so a caller cannot run an FP32 plan's f32 workspaces as f64 (gpucore.md F3:
    /// `run` took the precision separately from `new`).
    #[cfg(feature = "gpu")]
    #[test]
    fn a_product_plan_runs_in_its_own_arithmetic() {
        use crate::kernels::vgemm::VgemmProduct;
        use crate::linalg::{Arith, Gemm};
        use crate::Gpu;
        let gpu = match Gpu::new(0) {
            Ok(g) => g,
            Err(e) if std::env::var("LINUS_GPU_REQUIRED").as_deref() != Ok("1") => {
                eprintln!("skipped, no GPU: {e}");
                return;
            }
            Err(e) => panic!("LINUS_GPU_REQUIRED=1 but no GPU: {e}"),
        };
        let (vg, gemm) = (layout::vgemm(&gpu).unwrap(), Gemm::new(&gpu).unwrap());
        // C = A B with A = [[1, 3], [2, 4]] and B = 2 I (column-major): C = 2 A
        let (a, b) = ([1.0, 2.0, 3.0, 4.0], [2.0, 0.0, 0.0, 2.0]);
        let prod = |pa: u64, pb: u64, pc: u64| VgemmProduct {
            m: 2,
            n: 2,
            k: 2,
            a: pa,
            lda: 2,
            b: pb,
            ldb: 2,
            c: pc,
            ldc: 2,
        };
        for kind in [ProdKind::Vgemm, ProdKind::Grouped64(Arith::Fp64)] {
            let (da, db) = (gpu.upload(&a).unwrap(), gpu.upload(&b).unwrap());
            let dc = gpu.alloc_zeros::<f64>(4).unwrap();
            let p = prod(da.ptr().addr(), db.ptr().addr(), dc.ptr().addr());
            let plan = Prod::new(&gpu, &vg, kind, &[p]).unwrap();
            // SAFETY: the product addresses the three live 2 x 2 f64 arrays above.
            unsafe { plan.run(&gpu, &vg, &gemm, false, 0.0) }.unwrap();
            assert_eq!(gpu.download(&dc).unwrap(), [2.0, 4.0, 6.0, 8.0], "{kind:?}");
        }
        let f32s = |v: [f64; 4]| v.map(|x| x as f32);
        let (da, db) = (gpu.upload(&f32s(a)).unwrap(), gpu.upload(&f32s(b)).unwrap());
        let dc = gpu.alloc_zeros::<f32>(4).unwrap();
        let p = prod(da.ptr().addr(), db.ptr().addr(), dc.ptr().addr());
        let plan = Prod::new(&gpu, &vg, ProdKind::Grouped32, &[p]).unwrap();
        assert!(matches!(plan, Prod::G32(..)));
        // SAFETY: the product addresses the three live 2 x 2 f32 arrays above.
        unsafe { plan.run(&gpu, &vg, &gemm, false, 0.0) }.unwrap();
        assert_eq!(gpu.download(&dc).unwrap(), [2.0f32, 4.0, 6.0, 8.0]);
    }
}
