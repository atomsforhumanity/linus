//! The density-fitted two-electron gradient on the device: `GpuDfGrad` (feature `gpu`).
//!
//! The quantity of linus's CPU `df_gradient` (`two_electron::df::gradient` in the `linus`
//! crate) for J and the exchange of the operator a/r + b erf(omega r)/r (a global hybrid's
//! a_x K, or a range-separated hybrid's a K + b K_LR in the wK scheme of the SCF's DF-K:
//! Coulomb integrals on one side, erf ones on the other, Coulomb metric), V = (P|Q), one
//! auxiliary basis:
//!
//!   E^x   = sum_{P,mn} (mn|P)^x G_P,mn + sum_{P,mn} (mn|erf|P)^x G^w_P,mn + sum_PQ (P|Q)^x g_PQ,
//!   G_P   = c_P D - sum_s w_s C_s (a X~^s_P + b/2 Y~^s_P) C_s^T,
//!   G^w_P = -b/2 sum_s w_s C_s X~^s_P C_s^T,
//!   g_PQ  = -1/2 c_P c_Q + sum_s w_s [a/2 <X~^s_P, X~^s_Q> + b/4 (<X~^s_P, Y~^s_Q> + <Y~^s_P, X~^s_Q>)],
//!
//! with gamma_P = (P|mn) D_mn, c = V^-1 gamma, X^s_P,ij = (P|ij) and Y^s_P,ij = (P|erf|ij) over
//! the occupied orbitals C_s of spin s, X~ = V^-1 X, Y~ = V^-1 Y, and w_s = 2 for a closed shell
//! (one set of orbitals for both spins), 1 for open shells. Without an erf part (b = 0) the
//! Y and G^w terms vanish: a global hybrid with a = a_x.
//!
//! Work, in blocks of whole first shells (the kept shell pairs (A >= B) of [`DfLayout`] with A
//! in a range, so their functions m lie in [m0, m1) and n in [0, m1)):
//! 1. **Integrals** (mn|P) of the block's pairs on the device ([`crate::eri3c`]); gamma by a
//!    GEMV with the density rows. With exchange, the rows are unpacked into a dense
//!    Ad\[m\]\[n\]\[P\] (diagonal shell blocks at half weight, so that X = Y + Y^T), and
//!    Y\[i\]\[j\]\[P\] += sum_m C_mi sum_n Ad\[m\]\[n\]\[P\] C_nj by GEMMs (one per m, then one
//!    for the block).
//! 2. **Fit:** X = Y + Y^T over the orbital pairs i >= j (off-diagonal ones scaled by sqrt 2, so
//!    that sums over the pairs are the full sums), then c and X~ by two triangular solves with
//!    the metric's Cholesky factor (cuSOLVER, cuBLAS).
//! 3. **Metric term:** g_PQ by GEMMs over the packed pairs (its lower triangle), to the host, where
//!    linus-ints' two-center derivative integrals contract it (small: naux^2 against nx naux).
//! 4. **Three-center term:** per block, Zd\[m\]\[n\]\[P\] = a_x sum_s w_s (C_s X~^s_P C_s^T)_mn
//!    for n < m1 by GEMMs; then per sub-block of pairs Gamma = f (c D - Zd) in the integral row
//!    layout (f = 2 for the rows of off-diagonal shell pairs), contracted with the derivative
//!    integrals inside [`crate::eri3c`]'s derivative kernels (`Eri3c::gradient`: the pair
//!    centres and, by translational invariance, the auxiliary centre, per atom in fixed
//!    point; no derivative integral stored). The first implementation (`fused: false`) stores
//!    the six derivative planes of a sub-block and contracts them here: one thread per
//!    auxiliary function walks the rows of a chunk of pairs, pair sums to their atoms and
//!    auxiliary sums to their functions, both in fixed point ([`crate::detsum`]).
//!
//! In the code the total density and its rows come first (`density`), then each step is a
//! method of `GpuDfGrad`: `pass1`, `fit`, `metric_term` (the device's GEMMs and the download),
//! `unpack_fit` (X~ in full for step 4) and `metric_contraction` (the host's part, while the
//! device unpacks), `pass2`, and `assemble` (the downloads and the sums per atom). A method
//! takes the call's context `GradCall` (the inputs, the sizes the steps share, the timeline of
//! device events behind `GpuDfGradTimes`) and the buffers earlier steps left (`Pass1`,
//! `Pass2`).
//!
//! Without exchange (a_x = 0 or no occupied orbitals) only gamma, c and the J terms are formed.
//!
//! Range-separated exchange: the same builders compute the erf integrals and derivatives
//! ([`crate::eri3c`]'s erf kernels, compiled at the first such gradient or by
//! `GpuDfGrad::prepare_operator`). Pass 1 forms Y from the erf integrals of each block beside X;
//! the fit solves for Y~ too; the metric term gains the mixed X~/Y~ products (lower triangle,
//! two GEMMs per column block); pass 2 runs twice per block, G with the Coulomb derivative
//! kernels and G^w with the erf ones, through the same Zd buffer.
//!
//! Deterministic: cuBLAS in a fixed configuration, one writer per element in every kernel,
//! blocks in a fixed order on one stream, atom and auxiliary sums in fixed point, the metric
//! term on the host with linus-ints' deterministic reduction.

use crate::cpu::{Aux3c, BasisSet, DfThresholds};
use crate::df::DfLayout;
use std::sync::Arc;

/// Settings of a `GpuDfGrad` setup.
#[derive(Clone, Debug)]
pub struct GpuDfGradOptions {
    /// The screening thresholds (as the CPU's `DfJk` and the DF gradient's MO integrals):
    /// kept shell pairs, the three-center integrals' primitives (Schwarz-bound products) and
    /// the derivative integrals' (derivative-bound products).
    pub thresholds: DfThresholds,
    /// Bytes of a dense block Ad or Zd ((m1 - m0) m1 naux doubles); sets the block size.
    pub block_bytes: usize,
    /// Bytes of the six derivative planes of a sub-block of pairs.
    pub deriv_bytes: usize,
    /// Also report the Coulomb and exchange parts separately (costs a second metric
    /// contraction on the host and, fused, a second derivative pass).
    pub split_terms: bool,
    /// Contract Gamma inside the derivative kernels ([`crate::eri3c`]'s `Eri3c::gradient`):
    /// no derivative integral is stored; `deriv_bytes` then bounds the Gamma rows of a
    /// sub-block. Off: the six derivative planes are stored per sub-block and contracted by a
    /// kernel of this module (the first implementation, kept as a cross-check).
    pub fused: bool,
}

impl Default for GpuDfGradOptions {
    fn default() -> Self {
        GpuDfGradOptions {
            thresholds: DfThresholds::default(),
            block_bytes: 2 << 30,
            deriv_bytes: 2 << 30,
            split_terms: false,
            fused: true,
        }
    }
}

/// Rows of a chunk of pairs: the contraction kernel walks a chunk with one block of threads.
pub const CHUNK_ROWS: usize = 128;

/// A block of the plan: kept pairs `p0..p1` (all pairs of first shells in a range), whose
/// functions m lie in `m0..m1` and n in `0..m1`, rows `x0..x1` of the layout, chunks
/// `c0..c1` of the plan, and sub-blocks (chunk ranges) for the derivative planes.
#[derive(Clone, Debug)]
pub struct GradBlock {
    pub p0: usize,
    pub p1: usize,
    pub m0: usize,
    pub m1: usize,
    pub x0: usize,
    pub x1: usize,
    pub subs: Vec<(usize, usize)>,
}

/// The kept pairs, their rows, blocks, chunks and per-pair data of a DF gradient (no device
/// needed).
pub struct DfGradPlan {
    pub layout: Arc<DfLayout>,
    pub natom: usize,
    pub blocks: Vec<GradBlock>,
    /// Chunks: consecutive pairs (first, end) with at most [`CHUNK_ROWS`] rows (a larger pair
    /// alone), inside one block.
    pub chunks: Vec<(usize, usize)>,
    /// Per pair: atoms of its first and second shells, and the weight of its rows (2 off the
    /// diagonal shell blocks, whose rows stand for (m, n) and (n, m)).
    pub atom_a: Vec<i32>,
    pub atom_b: Vec<i32>,
    pub weight: Vec<f64>,
    /// Atom of each auxiliary function.
    pub aux_atom: Vec<usize>,
}

impl DfGradPlan {
    /// The plan for `basis` and `aux` (see the module docs).
    pub fn new(basis: &BasisSet, aux: &BasisSet, opts: &GpuDfGradOptions) -> DfGradPlan {
        let aux3 = Aux3c::new(aux);
        let layout = DfLayout::new(basis, &aux3, opts.thresholds.pair);
        Self::with_layout(Arc::new(layout), basis, aux, opts)
    }

    /// The plan of the kept pairs of `layout` (built for `basis` and `aux` at
    /// `opts.thresholds.pair`, e.g. a [`crate::df::DfShared`]'s).
    pub fn with_layout(
        layout: Arc<DfLayout>,
        basis: &BasisSet,
        aux: &BasisSet,
        opts: &GpuDfGradOptions,
    ) -> DfGradPlan {
        let naux = layout.naux.max(1);
        let natom = basis.atom_shells.len().max(aux.atom_shells.len());
        let pairs = &layout.pairs;
        let np = pairs.len();
        let ao_end = |s: usize| basis.shells[s].ao_start + basis.shells[s].nfunc();
        // runs of pairs with the same first shell (pairs come in first-shell order)
        let mut runs: Vec<(usize, usize)> = Vec::new();
        let mut p = 0;
        while p < np {
            let mut e = p + 1;
            while e < np && pairs[e].sa == pairs[p].sa {
                e += 1;
            }
            runs.push((p, e));
            p = e;
        }
        let mut blocks = Vec::new();
        let mut r = 0;
        while r < runs.len() {
            let p0 = runs[r].0;
            let m0 = basis.shells[pairs[p0].sa].ao_start;
            let mut e = r + 1;
            let mut m1 = ao_end(pairs[p0].sa);
            while e < runs.len() {
                let m1n = ao_end(pairs[runs[e].0].sa);
                if (m1n - m0) * m1n * naux * 8 > opts.block_bytes {
                    break;
                }
                m1 = m1n;
                e += 1;
            }
            let p1 = runs[e - 1].1;
            let x1 = if p1 < np { layout.x0[p1] } else { layout.nx };
            blocks.push(GradBlock {
                p0,
                p1,
                m0,
                m1,
                x0: layout.x0[p0],
                x1,
                subs: vec![],
            });
            r = e;
        }
        // chunks inside each block, sub-blocks of chunks
        let planes = if opts.fused { 1 } else { 6 };
        let deriv_rows = (opts.deriv_bytes / (planes * 8 * naux)).max(1);
        let mut chunks = Vec::new();
        for b in &mut blocks {
            let c_first = chunks.len();
            let mut p = b.p0;
            while p < b.p1 {
                let mut e = p;
                let mut rows = 0;
                while e < b.p1 && (rows == 0 || rows + pairs[e].nab() <= CHUNK_ROWS) {
                    rows += pairs[e].nab();
                    e += 1;
                }
                chunks.push((p, e));
                p = e;
            }
            let c_end = chunks.len();
            let rows_of = |c: usize| -> usize {
                let (a, e) = chunks[c];
                (a..e).map(|i| pairs[i].nab()).sum()
            };
            let mut c = c_first;
            while c < c_end {
                let mut e = c;
                let mut rows = 0;
                while e < c_end && (rows == 0 || rows + rows_of(e) <= deriv_rows) {
                    rows += rows_of(e);
                    e += 1;
                }
                b.subs.push((c, e));
                c = e;
            }
        }
        let atom_a = pairs
            .iter()
            .map(|p| basis.shells[p.sa].atom as i32)
            .collect();
        let atom_b = pairs
            .iter()
            .map(|p| basis.shells[p.sb].atom as i32)
            .collect();
        let weight = pairs
            .iter()
            .map(|p| if p.sa == p.sb { 1.0 } else { 2.0 })
            .collect();
        let mut aux_atom = vec![0; aux.nbf];
        for s in &aux.shells {
            aux_atom[s.ao_start..s.ao_start + s.nfunc()].fill(s.atom);
        }
        DfGradPlan {
            layout,
            natom,
            blocks,
            chunks,
            atom_a,
            atom_b,
            weight,
            aux_atom,
        }
    }

    /// Rows of the pairs of chunks `c0..c1`.
    pub fn chunk_rows(&self, c0: usize, c1: usize) -> std::ops::Range<usize> {
        let lay = &self.layout;
        if c0 >= c1 {
            return 0..0;
        }
        let (p0, _) = self.chunks[c0];
        let (_, p1) = self.chunks[c1 - 1];
        let x1 = if p1 < lay.pairs.len() {
            lay.x0[p1]
        } else {
            lay.nx
        };
        lay.x0[p0]..x1
    }

    /// The largest block's sizes, which the DF gradient's and Hessian's buffers are sized by.
    #[cfg(feature = "gpu")]
    pub(crate) fn max_block(&self) -> BlockSizes {
        let mut m = BlockSizes::default();
        for b in &self.blocks {
            m.dense = m.dense.max((b.m1 - b.m0) * b.m1);
            m.rows = m.rows.max(b.x1 - b.x0);
            m.mrange = m.mrange.max(b.m1 - b.m0);
            for &(c0, c1) in &b.subs {
                m.sub_rows = m.sub_rows.max(self.chunk_rows(c0, c1).len());
            }
        }
        m
    }
}

/// The largest of a plan's blocks: (m1 - m0) m1 of the dense block, rows, functions m1 - m0;
/// and the most rows of a sub-block.
#[cfg(feature = "gpu")]
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct BlockSizes {
    pub(crate) dense: usize,
    pub(crate) rows: usize,
    pub(crate) mrange: usize,
    pub(crate) sub_rows: usize,
}

#[cfg(feature = "gpu")]
pub use device::*;

#[cfg(feature = "gpu")]
mod device {
    use super::*;
    use crate::blas::{GemmDims, MatMut, MatRef, Op};
    use crate::cpu::Operator;
    use crate::cpu::{AoMatrix, DerivOrder, Eri2DerivOptions, Eri2DerivPlan, PairOptions, PairSet};
    use crate::detsum::DetAcc;
    use crate::df_deriv::{
        upload_orbitals, DerivKernels, DerivSource, GammaJ, GammaTerms, RowBlock, RowTables, Unpack,
    };
    use crate::engine::Timeline;
    use crate::eri3c::{Eri3c, Eri3cKind};
    use crate::error::GpuError;
    use crate::kernels::registry::{KernelModule, COMMON_H};
    use crate::linalg::{Arith, Gemm};
    use crate::pool::{Grant, PlanViolation};
    use crate::{Arg, DeviceBuf, Gpu, Kernel, Launch};
    use std::borrow::Cow;
    use std::time::Instant;

    /// Columns per DGEMM of the metric term's lower triangle, and the most blocks.
    const METRIC_COLUMN_BLOCK: usize = 256;
    const METRIC_BLOCKS_MAX: usize = 8;

    const KERNELS: &str = include_str!("../../device/grad/df.cu");

    /// The DF gradient's module: the fixed-point accumulator and the common device functions,
    /// then its kernels.
    pub(crate) const MODULE: KernelModule = KernelModule::new(
        "dfgrad",
        || Cow::Owned([crate::detsum::DEVICE_SRC, COMMON_H, KERNELS].join("\n")),
        &["gd_pack", "gd_unpack_pairs", "gd_contract"],
    );

    /// The gradient and its parts (natom x 3, Eh/bohr).
    #[derive(Clone, Debug, Default)]
    pub struct GpuDfGradOutput {
        /// The two-electron gradient, J and exchange.
        pub gradient: Vec<[f64; 3]>,
        /// Of which the two-center (metric) term.
        pub metric: Vec<[f64; 3]>,
        /// With `split_terms`: the Coulomb and the exchange parts (each with its metric term).
        pub coulomb: Option<Vec<[f64; 3]>>,
        pub exchange: Option<Vec<[f64; 3]>>,
    }

    /// Setup times (seconds, wall clock).
    #[derive(Clone, Debug, Default)]
    pub struct GpuDfGradSetupTimes {
        /// Pairs, layout and plan (CPU).
        pub plan: f64,
        /// Packing the integral and derivative builders (CPU) and uploading them, kernels.
        pub builders: f64,
        /// Of `builders`: packing the integral builder (CPU).
        pub pack_ints: f64,
        /// Of `builders`: packing the derivative builder (CPU).
        pub pack_derivs: f64,
        /// Of `builders`: uploading the two builders' packed data (and its bytes), getting
        /// their kernels.
        pub upload: f64,
        pub upload_bytes: usize,
        pub kernels: f64,
        /// Metric (CPU), its Cholesky factor (device); auxiliary pair data for the metric term.
        pub metric: f64,
        /// Of `metric`: the metric on the CPU.
        pub metric_host: f64,
        /// Of `metric`: the auxiliary pairs and their derivative data (CPU).
        pub aux_pairs: f64,
        /// The builders' erf kernels (compiled, or loaded from the cache) for range separation.
        pub long_range: f64,
        pub total: f64,
    }

    impl crate::engine::Phases for GpuDfGradSetupTimes {
        fn phases(&self) -> impl IntoIterator<Item = (&'static str, f64)> {
            [
                ("plan", self.plan),
                ("builders", self.builders),
                ("pack_ints", self.pack_ints),
                ("pack_derivs", self.pack_derivs),
                ("metric", self.metric),
                ("metric_host", self.metric_host),
                ("aux_pairs", self.aux_pairs),
                ("upload", self.upload),
                ("kernels", self.kernels),
            ]
        }
    }

    /// Times of the last gradient (seconds; device phases from events, host phases wall
    /// clock).
    #[derive(Clone, Debug, Default)]
    pub struct GpuDfGradTimes {
        pub total: f64,
        /// Density, density rows.
        pub density: f64,
        /// Pass 1: three-center integrals on the device.
        pub ints: f64,
        /// Pass 1: gamma, unpacking and the MO GEMMs.
        pub mo: f64,
        /// Symmetrization and the triangular solves.
        pub fit: f64,
        /// The metric term's GEMMs (device).
        pub metric_gemm: f64,
        /// The metric term's two-center derivative integrals (host, wall clock).
        pub metric_host: f64,
        /// Pass 2: Zd GEMMs.
        pub zd: f64,
        /// Pass 2: derivative integrals (fused: with the contraction).
        pub deriv: f64,
        /// Pass 2: contraction (fused: building Gamma).
        pub contract: f64,
    }

    struct Kernels {
        density_rows: Kernel,
        /// The DF derivative engines' common kernels (the dense blocks' unpack, Gamma).
        deriv: DerivKernels,
        pack: Kernel,
        unpack_pairs: Kernel,
        contract: Kernel,
    }

    /// The DF two-electron gradient on the device (see the module docs).
    pub struct GpuDfGrad {
        nbf: usize,
        naux: usize,
        plan: DfGradPlan,
        aux: BasisSet,
        /// the auxiliary pairs with first-order expansions
        ap: PairSet,
        ints: Eri3c,
        ders: Eri3c,
        chol: DeviceBuf<f64>,
        rows: RowTables,
        px0: DeviceBuf<i64>,
        pnab: DeviceBuf<i32>,
        pat_a: DeviceBuf<i32>,
        pat_b: DeviceBuf<i32>,
        pf: DeviceBuf<f64>,
        cp0: DeviceBuf<i32>,
        cp1: DeviceBuf<i32>,
        gemm: Gemm,
        kernels: Kernels,
        split: bool,
        fused: bool,
        /// The plan's largest block.
        max_block: BlockSizes,
        /// The memory plan's grant of the workspaces ([`GpuDfGrad::grant_workspace`]), drawn
        /// by their allocations in each call (freed, they give the bytes back).
        workspace: Option<Grant>,
        pub setup_times: GpuDfGradSetupTimes,
        pub last: GpuDfGradTimes,
    }

    fn launch_err(what: &str) -> GpuError {
        GpuError::Contract(format!("DF gradient: {what}"))
    }

    impl GpuDfGrad {
        /// Sets up `basis` with the auxiliary basis `aux`: the plan, the device integral and
        /// derivative builders, the metric's Cholesky factor. CPU work runs on the current
        /// rayon pool.
        pub fn new(
            gpu: &Gpu,
            basis: &BasisSet,
            aux: &BasisSet,
            opts: &GpuDfGradOptions,
        ) -> Result<GpuDfGrad, GpuError> {
            Self::with_shared(gpu, basis, aux, opts, None)
        }

        /// As [`GpuDfGrad::new`], with the layout, the packed integrals, the metric and the
        /// derivative records of `shared` (an SCF setup's, [`crate::df::DfShared`]) when it
        /// fits the basis sets and thresholds: the same engine, bit for bit, without
        /// rebuilding them (derivative records it builds stay there for the next engine).
        pub fn with_shared(
            gpu: &Gpu,
            basis: &BasisSet,
            aux: &BasisSet,
            opts: &GpuDfGradOptions,
            shared: Option<&crate::df::DfShared>,
        ) -> Result<GpuDfGrad, GpuError> {
            let t_all = Instant::now();
            let mut times = GpuDfGradSetupTimes::default();
            let t = Instant::now();
            let shared = shared.filter(|s| s.fits(basis, aux, &opts.thresholds));
            let plan = match shared {
                Some(s) => DfGradPlan::with_layout(s.layout.clone(), basis, aux, opts),
                None => DfGradPlan::new(basis, aux, opts),
            };
            let lay = &plan.layout;
            let (nbf, naux, nx) = (lay.nbf, lay.naux, lay.nx);
            if nx == 0 || naux == 0 {
                return Err(launch_err("no shell pair or no auxiliary function"));
            }
            times.plan = t.elapsed().as_secs_f64();
            let t = Instant::now();
            let source = DerivSource {
                basis,
                aux,
                lay,
                shared,
                thresholds: &opts.thresholds,
            };
            let (ints, pack_ints) = source.integrals(gpu)?;
            let (ders, pack_derivs) = source.derivatives(gpu, Eri3cKind::Derivatives)?;
            (times.pack_ints, times.pack_derivs) = (pack_ints, pack_derivs);
            times.upload = ints.setup_upload + ders.setup_upload;
            times.upload_bytes = ints.setup_upload_bytes + ders.setup_upload_bytes;
            times.kernels = ints.setup_kernels + ders.setup_kernels;
            let df = &crate::df::MODULE;
            let src = MODULE.source();
            let k = |e: &str| gpu.kernel(MODULE.name, &src, e);
            let kernels = Kernels {
                density_rows: gpu.kernel(df.name, &df.source(), "df_density_rows")?,
                deriv: DerivKernels::new(gpu)?,
                pack: k("gd_pack")?,
                unpack_pairs: k("gd_unpack_pairs")?,
                contract: k("gd_contract")?,
            };
            let px0: Vec<i64> = lay.x0.iter().map(|&x| x as i64).collect();
            let pnab: Vec<i32> = lay.pairs.iter().map(|p| p.nab() as i32).collect();
            let cp0: Vec<i32> = plan.chunks.iter().map(|c| c.0 as i32).collect();
            let cp1: Vec<i32> = plan.chunks.iter().map(|c| c.1 as i32).collect();
            let rows = RowTables::upload(gpu, lay)?;
            let (px0, pnab) = (gpu.upload(&px0)?, gpu.upload(&pnab)?);
            let (pat_a, pat_b, pf) = (
                gpu.upload(&plan.atom_a)?,
                gpu.upload(&plan.atom_b)?,
                gpu.upload(&plan.weight)?,
            );
            let (cp0, cp1) = (gpu.upload(&cp0)?, gpu.upload(&cp1)?);
            times.builders = t.elapsed().as_secs_f64();
            let t = Instant::now();
            let (chol, metric_host) = source.metric_factor(gpu)?;
            times.metric_host = metric_host;
            let tp = Instant::now();
            let ap = PairSet::aux(aux, PairOptions::with_order(DerivOrder::One));
            times.aux_pairs = tp.elapsed().as_secs_f64();
            gpu.synchronize()?;
            times.metric = t.elapsed().as_secs_f64();
            let max_block = plan.max_block();
            times.total = t_all.elapsed().as_secs_f64();
            Ok(GpuDfGrad {
                nbf,
                naux,
                aux: aux.clone(),
                ap,
                ints,
                ders,
                chol,
                rows,
                px0,
                pnab,
                pat_a,
                pat_b,
                pf,
                cp0,
                cp1,
                gemm: Gemm::new(gpu)?,
                kernels,
                split: opts.split_terms,
                fused: opts.fused,
                max_block,
                plan,
                workspace: None,
                setup_times: times,
                last: GpuDfGradTimes::default(),
            })
        }

        pub fn plan(&self) -> &DfGradPlan {
            &self.plan
        }

        pub fn nbf(&self) -> usize {
            self.nbf
        }

        /// Reports the Coulomb and exchange parts separately from now on (or not).
        pub fn set_split_terms(&mut self, split: bool) {
            self.split = split;
        }

        /// The builders' erf kernels for the exchange operator `xop` (nothing to do without an
        /// erf part); [`GpuDfGrad::compute_device`] calls it when needed.
        pub fn prepare_operator(&mut self, gpu: &Gpu, xop: &Operator) -> Result<(), GpuError> {
            if !xop.has_erf() {
                return Ok(());
            }
            if xop.omega() < 0.0 || !xop.omega().is_finite() {
                return Err(launch_err("the range separation omega must be positive"));
            }
            if self.ints.has_erf() && self.ders.has_erf() {
                return Ok(());
            }
            let t = Instant::now();
            let erf = Operator::LongRange { omega: xop.omega() };
            self.ints.prepare_operator(gpu, &erf)?;
            self.ders.prepare_operator(gpu, &erf)?;
            self.setup_times.long_range += t.elapsed().as_secs_f64();
            Ok(())
        }

        /// Device bytes of the workspaces of a gradient with `ncols` occupied orbitals per
        /// spin and the exchange operator `xop` (see [`GpuDfGrad::compute`]), besides the
        /// setup's builders and factor: a range-separated operator adds Y and Y~ per spin (the
        /// pass-2 buffers are shared).
        pub fn workspace_bytes(&self, ncols: &[usize], xop: &Operator) -> usize {
            let with_k = xop.coulomb() != 0.0 || xop.has_erf();
            let mut b = self.base_workspace_bytes(ncols, with_k);
            if xop.has_erf() {
                b += 8 * ncols
                    .iter()
                    .map(|o| (o * o + o * (o + 1) / 2) * self.naux)
                    .sum::<usize>();
            }
            b
        }

        /// The workspaces' bytes with exchange (`with_k`) or without, before the range
        /// separation's.
        fn base_workspace_bytes(&self, ncols: &[usize], with_k: bool) -> usize {
            let (nbf, naux) = (self.nbf, self.naux);
            let nx = self.plan.layout.nx;
            let omax = ncols.iter().copied().max().unwrap_or(0);
            let m = &self.max_block;
            let mut b = nbf * nbf + nx + naux * 2 + m.rows * naux;
            b += if self.fused { 1 } else { 6 } * m.sub_rows * naux;
            if with_k {
                b += ncols
                    .iter()
                    .map(|o| (o * o + o * (o + 1) / 2) * naux)
                    .sum::<usize>();
                b += m.dense * naux + m.mrange * omax * naux;
            }
            b += if self.split { 2 } else { 1 } * naux * naux;
            8 * b
        }

        /// Sets aside the workspaces of a gradient with `ncols` and `xop`
        /// ([`GpuDfGrad::workspace_bytes`]) as the memory plan's grant, which the calls'
        /// allocations draw from; an out-of-memory error when they do not fit.
        pub fn grant_workspace(
            &mut self,
            gpu: &Gpu,
            ncols: &[usize],
            xop: &Operator,
        ) -> Result<(), GpuError> {
            self.workspace = None;
            let need = self.workspace_bytes(ncols, xop);
            self.workspace = Some(gpu.grant(need, "the two-electron gradient's workspace")?);
            Ok(())
        }

        /// The draws of the calls beyond the workspace's grant (plan violations).
        pub fn workspace_violations(&self) -> Vec<PlanViolation> {
            self.workspace
                .as_ref()
                .map(Grant::violations)
                .unwrap_or_default()
        }

        /// The gradient of occupied orbitals given on the host: per spin channel C (nbf x ncol,
        /// column-major), one channel for a closed shell (`restricted`), else alpha and beta;
        /// `xop` the exchange operator a/r + b erf(omega r)/r (linus's `df_gradient` argument:
        /// `exchange_parts`' operator of a range-separated hybrid, coefficients included; a
        /// global hybrid's a_x/r; 0 for J only).
        pub fn compute(
            &mut self,
            gpu: &Gpu,
            occ: &[(&[f64], usize)],
            restricted: bool,
            xop: Operator,
        ) -> Result<GpuDfGradOutput, GpuError> {
            let (cmo, ncols) = upload_orbitals(gpu, self.nbf, occ, launch_err)?;
            self.compute_device(gpu, &cmo, &ncols, restricted, xop)
        }

        /// As [`GpuDfGrad::compute`] for occupied orbitals on the device: `cmo` holds the
        /// channels side by side (nbf x sum(ncols), column-major; one channel for a closed
        /// shell with `restricted`, else alpha and beta). The phases of the module docs run in
        /// order, the metric term's contraction on the host while the device unpacks X~ for
        /// pass 2.
        pub fn compute_device(
            &mut self,
            gpu: &Gpu,
            cmo: &DeviceBuf<f64>,
            ncols: &[usize],
            restricted: bool,
            xop: Operator,
        ) -> Result<GpuDfGradOutput, GpuError> {
            self.prepare_operator(gpu, &xop)?;
            let (nbf, naux) = (self.nbf, self.naux);
            let nsp = if restricted { 1 } else { 2 };
            if ncols.len() != nsp {
                return Err(launch_err("one orbital channel per spin expected"));
            }
            let ntot: usize = ncols.iter().sum();
            if cmo.len() < nbf * ntot {
                return Err(launch_err("orbital buffer too small"));
            }
            let w_s = if restricted { 2.0 } else { 1.0 };
            // a K + b K_LR(omega): a = the full-range coefficient, b the erf one
            let a_x = xop.coulomb();
            let with_lr = xop.has_erf() && ntot > 0;
            let (b_lr, erf) = if with_lr {
                (xop.long_range(), Operator::LongRange { omega: xop.omega() })
            } else {
                (0.0, Operator::Coulomb)
            };
            let with_k = (a_x != 0.0 || with_lr) && ntot > 0;
            let split = self.split;
            let mut times = GpuDfGradTimes::default();
            let t_all = Instant::now();
            let c0s: Vec<usize> = ncols
                .iter()
                .scan(0, |a, &n| {
                    let v = *a;
                    *a += n;
                    Some(v)
                })
                .collect();
            let mut c = GradCall {
                gpu,
                cmo,
                ncols,
                c0s,
                nsp,
                w_s,
                a_x,
                b_lr,
                erf,
                with_lr,
                with_k,
                nops: if with_lr { 2 } else { 1 },
                split,
                nterm: if split { 2 } else { 1 },
                nbf,
                naux,
                natom: self.plan.natom,
                nx: self.plan.layout.nx,
                tl: Timeline::new(gpu),
            };
            // the density rows and the integral block stay allocated to the end of the call
            let (dens, wrow) = self.density(&mut c)?;
            let mut pass1 = self.pass1(&mut c, &wrow)?;
            let xp = self.fit(&mut c, &mut pass1)?;
            let hgm = self.metric_term(&mut c, &pass1.gamma, &xp)?;
            self.unpack_fit(&c, xp, &mut pass1.y)?;
            let th = Instant::now();
            let g2 = self.metric_contraction(&c, &hgm);
            times.metric_host = th.elapsed().as_secs_f64();
            let acc = self.pass2(&mut c, &dens, &mut pass1)?;
            let out = self.assemble(&c, acc, &g2, &mut times)?;
            times.total = t_all.elapsed().as_secs_f64();
            self.last = times;
            Ok(out)
        }

        /// The total density and its rows (the layout's pair rows of D).
        fn density(&self, c: &mut GradCall) -> Result<(DeviceBuf<f64>, DeviceBuf<f64>), GpuError> {
            let (gpu, cmo, ncols, c0s) = (c.gpu, c.cmo, c.ncols, &c.c0s);
            let (nbf, nx, nsp, w_s) = (c.nbf, c.nx, c.nsp, c.w_s);
            let n2 = nbf * nbf;
            // total density and its rows
            let s0 = c.tl.mark()?;
            let mut dens = gpu.alloc_zeros_granted::<f64>(self.workspace.as_ref(), n2)?;
            for sp in 0..nsp {
                if ncols[sp] > 0 {
                    self.gemm.gemm(
                        Arith::Fp64,
                        GemmDims::new(Op::N, Op::T, nbf, nbf, ncols[sp]),
                        (w_s, if sp == 0 { 0.0 } else { 1.0 }),
                        MatRef::new(cmo, c0s[sp] * nbf, nbf),
                        MatRef::new(cmo, c0s[sp] * nbf, nbf),
                        MatMut::new(&mut dens, 0, nbf),
                    )?;
                }
            }
            let mut wrow = gpu.alloc_zeros_granted::<f64>(self.workspace.as_ref(), nx)?;
            // SAFETY: matches df_density_rows; nx rows of the tables, one nbf^2 density, nx
            // values out.
            unsafe {
                gpu.launch(
                    &self.kernels.density_rows,
                    Launch::linear(nx, 256),
                    &[
                        Arg::I64(nx as i64),
                        Arg::I32(1),
                        Arg::I64(nbf as i64),
                        Arg::buf(&self.rows.m),
                        Arg::buf(&self.rows.n),
                        Arg::buf(&self.rows.kind),
                        Arg::buf(&dens),
                        Arg::buf_mut(&mut wrow),
                    ],
                )
            }?;
            c.tl.close(|ts| &mut ts.density, s0)?;
            Ok((dens, wrow))
        }

        /// Pass 1 (step 1 of the module docs) over the blocks: the integrals, gamma by a GEMV
        /// with the density rows `wrow`, and with exchange Y per spin by the MO GEMMs.
        fn pass1(&self, c: &mut GradCall, wrow: &DeviceBuf<f64>) -> Result<Pass1, GpuError> {
            let (gpu, cmo, ncols, c0s) = (c.gpu, c.cmo, c.ncols, &c.c0s);
            let (nbf, naux, nsp, nops, with_k, erf) =
                (c.nbf, c.naux, c.nsp, c.nops, c.with_k, c.erf);
            // pass 1: gamma and, with exchange, Y per spin (y[op][spin]: op 0 the Coulomb
            // integrals, 1 the erf ones)
            let mut gamma = gpu.alloc_zeros_granted::<f64>(self.workspace.as_ref(), naux)?;
            let mut ablk = gpu.alloc_zeros_granted::<f64>(
                self.workspace.as_ref(),
                self.max_block.rows.max(1) * naux,
            )?;
            let omax = ncols.iter().copied().max().unwrap_or(0);
            let mut y: Vec<Vec<DeviceBuf<f64>>> = (0..nops).map(|_| Vec::new()).collect();
            let (mut ad, mut ut) = if with_k {
                for yo in y.iter_mut() {
                    for &o in ncols {
                        yo.push(gpu.alloc_zeros_granted::<f64>(
                            self.workspace.as_ref(),
                            (o * o * naux).max(1),
                        )?);
                    }
                }
                (
                    Some(gpu.alloc_zeros_granted::<f64>(
                        self.workspace.as_ref(),
                        self.max_block.dense * naux,
                    )?),
                    Some(gpu.alloc_zeros_granted::<f64>(
                        self.workspace.as_ref(),
                        (self.max_block.mrange * omax * naux).max(1),
                    )?),
                )
            } else {
                (None, None)
            };
            for (bi, b) in self.plan.blocks.iter().enumerate() {
                for (op, yo) in y.iter_mut().enumerate().take(if with_k { nops } else { 1 }) {
                    let rows = b.x1 - b.x0;
                    let s = c.tl.mark()?;
                    let o = if op == 0 { &Operator::Coulomb } else { &erf };
                    self.ints.compute(gpu, b.p0..b.p1, o, &mut ablk, 0)?;
                    let s1 = c.tl.mark()?;
                    if op == 0 {
                        self.gemm.gemm(
                            Arith::Fp64,
                            GemmDims::new(Op::N, Op::N, naux, 1, rows),
                            (1.0, if bi == 0 { 0.0 } else { 1.0 }),
                            MatRef::new(&ablk, 0, naux),
                            MatRef::new(wrow, b.x0, rows),
                            MatMut::new(&mut gamma, 0, naux),
                        )?;
                    }
                    if let (Some(ad), Some(u)) = (ad.as_mut(), ut.as_mut()) {
                        let (m0, m1) = (b.m0, b.m1);
                        let nm = m1 - m0;
                        gpu.zero_range(ad, 0, nm * m1 * naux)?;
                        let blk = RowBlock {
                            rows: &self.rows,
                            x0: b.x0,
                            nrows: rows,
                            naux,
                            m0,
                            m1,
                        };
                        // SAFETY: reads rows x0..x1 of the tables and of ablk, writes (m - m0) m1
                        // + n < nm m1 rows of naux (m in [m0, m1), n < m1 for the block's pairs),
                        // inside ad.
                        unsafe {
                            self.kernels
                                .deriv
                                .unpack(gpu, blk, Unpack::Halved, &ablk, ad)
                        }?;
                        for sp in 0..nsp {
                            let o = ncols[sp];
                            if o == 0 {
                                continue;
                            }
                            let cc = c0s[sp] * nbf;
                            // U[m][j][P] = sum_n Ad[m][n][P] C_nj
                            for mi in 0..nm {
                                self.gemm.gemm(
                                    Arith::Fp64,
                                    GemmDims::new(Op::N, Op::N, naux, o, m1),
                                    (1.0, 0.0),
                                    MatRef::new(ad, mi * m1 * naux, naux),
                                    MatRef::new(cmo, cc, nbf),
                                    MatMut::new(u, mi * o * naux, naux),
                                )?;
                            }
                            // Y[i][j][P] += sum_m C_mi U[m][j][P]
                            self.gemm.gemm(
                                Arith::Fp64,
                                GemmDims::new(Op::N, Op::N, o * naux, o, nm),
                                (1.0, if bi == 0 { 0.0 } else { 1.0 }),
                                MatRef::new(u, 0, o * naux),
                                MatRef::new(cmo, cc + m0, nbf),
                                MatMut::new(&mut yo[sp], 0, o * naux),
                            )?;
                        }
                    }
                    let s2 = c.tl.mark()?;
                    c.tl.span(|ts| &mut ts.ints, s, s1);
                    c.tl.span(|ts| &mut ts.mo, s1, s2);
                }
            }
            // pass 2's Zd takes the integral rows' bytes (one dense block in the plan)
            drop(ad);
            Ok(Pass1 {
                gamma,
                y,
                ut,
                _ablk: ablk,
            })
        }

        /// The fit (step 2): c = V^-1 gamma in place, and with exchange X~ (op 0) and Y~ (op 1)
        /// per spin, packed: xp\[op\]\[spin\].
        fn fit(
            &self,
            c: &mut GradCall,
            pass1: &mut Pass1,
        ) -> Result<Vec<Vec<Option<DeviceBuf<f64>>>>, GpuError> {
            let (gpu, ncols, naux, with_k) = (c.gpu, c.ncols, c.naux, c.with_k);
            let y = &pass1.y;
            // fit: c = V^-1 gamma; X = Y + Y^T packed (i >= j, off-diagonal pairs scaled by
            // sqrt 2), X~ = V^-1 X
            let s = c.tl.mark()?;
            let solve = |gemm: &Gemm, b: &mut DeviceBuf<f64>, ncol: usize| {
                gemm.trsm_lower_left(
                    Op::N,
                    (naux, ncol),
                    MatRef::new(&self.chol, 0, naux),
                    MatMut::new(b, 0, naux),
                )?;
                gemm.trsm_lower_left(
                    Op::T,
                    (naux, ncol),
                    MatRef::new(&self.chol, 0, naux),
                    MatMut::new(b, 0, naux),
                )
            };
            solve(&self.gemm, &mut pass1.gamma, 1)?;
            // xp[op][spin]: X~ (op 0) and Y~ (op 1), packed
            let mut xp: Vec<Vec<Option<DeviceBuf<f64>>>> =
                (0..y.len()).map(|_| Vec::new()).collect();
            if with_k {
                for (xo, (sp, yb)) in y
                    .iter()
                    .enumerate()
                    .flat_map(|(op, yo)| yo.iter().enumerate().map(move |e| (op, e)))
                {
                    let xp = &mut xp[xo];
                    let o = ncols[sp];
                    if o == 0 {
                        xp.push(None);
                        continue;
                    }
                    let npair = o * (o + 1) / 2;
                    let mut x =
                        gpu.alloc_zeros_granted::<f64>(self.workspace.as_ref(), npair * naux)?;
                    // SAFETY: matches gd_pack; reads o^2 naux values of y, writes npair naux.
                    unsafe {
                        gpu.launch(
                            &self.kernels.pack,
                            Launch::linear(npair * naux, 256),
                            &[
                                Arg::I32(o as i32),
                                Arg::I32(naux as i32),
                                Arg::buf(yb),
                                Arg::buf_mut(&mut x),
                            ],
                        )
                    }?;
                    solve(&self.gemm, &mut x, npair)?;
                    xp.push(Some(x));
                }
            }
            c.tl.close(|ts| &mut ts.fit, s)?;
            Ok(xp)
        }

        /// The metric term's g_PQ (step 3) on the device (one matrix, or J and K apart),
        /// downloaded, the matrix holding K mirrored on the host.
        fn metric_term(
            &self,
            c: &mut GradCall,
            gamma: &DeviceBuf<f64>,
            xp: &[Vec<Option<DeviceBuf<f64>>>],
        ) -> Result<Vec<f64>, GpuError> {
            let (gpu, ncols, naux, with_k) = (c.gpu, c.ncols, c.naux, c.with_k);
            let (a_x, b_lr, w_s, split, nterm) = (c.a_x, c.b_lr, c.w_s, c.split, c.nterm);
            // metric term: g_PQ (one matrix, or J and K apart); the K part's lower triangle by
            // DGEMMs over column blocks, the upper one mirrored on the host
            let s = c.tl.mark()?;
            let mut gm =
                gpu.alloc_zeros_granted::<f64>(self.workspace.as_ref(), nterm * naux * naux)?;
            self.gemm.gemm(
                Arith::Fp64,
                GemmDims::new(Op::N, Op::T, naux, naux, 1),
                (-0.5, 0.0),
                MatRef::new(gamma, 0, naux),
                MatRef::new(gamma, 0, naux),
                MatMut::new(&mut gm, 0, naux),
            )?;
            if with_k {
                let off = if split { naux * naux } else { 0 };
                let mut first = split;
                let nb = naux
                    .div_ceil(METRIC_COLUMN_BLOCK)
                    .clamp(1, METRIC_BLOCKS_MAX);
                for sp in 0..ncols.len() {
                    let Some(x) = &xp[0][sp] else { continue };
                    let npair = ncols[sp] * (ncols[sp] + 1) / 2;
                    // a/2 w_s X~ X~^T
                    if a_x != 0.0 {
                        for b in 0..nb {
                            let (j0, j1) = (b * naux / nb, (b + 1) * naux / nb);
                            self.gemm.gemm(
                                Arith::Fp64,
                                GemmDims::new(Op::N, Op::T, naux - j0, j1 - j0, npair),
                                (0.5 * a_x * w_s, if first { 0.0 } else { 1.0 }),
                                MatRef::new(x, j0, naux),
                                MatRef::new(x, j0, naux),
                                MatMut::new(&mut gm, off + j0 * naux + j0, naux),
                            )?;
                        }
                        first = false;
                    }
                    // b/4 w_s (X~ Y~^T + Y~ X~^T), the lower triangle by column blocks
                    if let Some(Some(yw)) = xp.get(1).map(|v| &v[sp]) {
                        for b in 0..nb {
                            let (j0, j1) = (b * naux / nb, (b + 1) * naux / nb);
                            for (l, r) in [(x, yw), (yw, x)] {
                                self.gemm.gemm(
                                    Arith::Fp64,
                                    GemmDims::new(Op::N, Op::T, naux - j0, j1 - j0, npair),
                                    (
                                        0.25 * b_lr * w_s,
                                        if first && std::ptr::eq(l, x) {
                                            0.0
                                        } else {
                                            1.0
                                        },
                                    ),
                                    MatRef::new(l, j0, naux),
                                    MatRef::new(r, j0, naux),
                                    MatMut::new(&mut gm, off + j0 * naux + j0, naux),
                                )?;
                            }
                        }
                        first = false;
                    }
                }
            }
            c.tl.close(|ts| &mut ts.metric_gemm, s)?;
            let mut hgm = gpu.download(&gm)?;
            drop(gm);
            if with_k {
                // the matrix holding K is symmetric: its upper triangle from the lower one
                let off = if split { naux * naux } else { 0 };
                for j in 0..naux {
                    for i in 0..j {
                        hgm[off + j * naux + i] = hgm[off + i * naux + j];
                    }
                }
            }
            Ok(hgm)
        }

        /// X~ (and Y~) in full \[i\]\[j\]\[P\] for pass 2's Zd GEMMs, into Y's buffers `y`; frees
        /// the packed `xp`.
        fn unpack_fit(
            &self,
            c: &GradCall,
            xp: Vec<Vec<Option<DeviceBuf<f64>>>>,
            y: &mut [Vec<DeviceBuf<f64>>],
        ) -> Result<(), GpuError> {
            let (gpu, ncols, naux) = (c.gpu, c.ncols, c.naux);
            // X~ (and Y~) in full [i][j][P] for the Zd GEMMs (into Y's buffers)
            for (op, xo) in xp.iter().enumerate() {
                for (sp, x) in xo.iter().enumerate() {
                    let Some(x) = x else { continue };
                    let o = ncols[sp];
                    let npair = o * (o + 1) / 2;
                    // SAFETY: matches gd_unpack_pairs; reads npair naux values, writes o^2 naux.
                    unsafe {
                        gpu.launch(
                            &self.kernels.unpack_pairs,
                            Launch::linear(npair * naux, 256),
                            &[
                                Arg::I32(o as i32),
                                Arg::I32(naux as i32),
                                Arg::buf(x),
                                Arg::buf_mut(&mut y[op][sp]),
                            ],
                        )
                    }?;
                }
            }
            drop(xp);
            Ok(())
        }

        /// The metric term's contraction on the host per term (linus-ints' two-center
        /// derivative integrals against the downloaded g_PQ `hgm`).
        fn metric_contraction(&self, c: &GradCall, hgm: &[f64]) -> Vec<Vec<[f64; 3]>> {
            let (naux, natom, nterm, with_k) = (c.naux, c.natom, c.nterm, c.with_k);
            let n2a = naux * naux;
            let opts = Eri2DerivOptions {
                thresh: DfThresholds::default().metric_deriv,
                op: Operator::Coulomb,
            };
            let plan = Eri2DerivPlan::new(&self.aux, &self.ap, DerivOrder::One, opts)
                .expect("auxiliary pairs of order 1");
            let mut ws2 = plan.workspace();
            (0..nterm)
                .map(|t| {
                    if t == 1 && !with_k {
                        vec![[0.0; 3]; natom]
                    } else {
                        let mut g = vec![[0.0; 3]; self.aux.natom()];
                        let gm = AoMatrix::row_major(&hgm[t * n2a..(t + 1) * n2a], naux);
                        plan.gradient(gm, &mut g, &mut ws2);
                        g
                    }
                })
                .collect()
        }

        /// Pass 2 (step 4, the three-center term) per block: Zd by GEMMs, then per sub-block of
        /// pairs Gamma contracted with the derivative integrals (fused into the derivative
        /// kernels, or the stored planes contracted by `gd_contract`), into fixed-point
        /// accumulators.
        fn pass2(
            &self,
            c: &mut GradCall,
            dens: &DeviceBuf<f64>,
            pass1: &mut Pass1,
        ) -> Result<Pass2, GpuError> {
            let (gpu, cmo, ncols, c0s) = (c.gpu, c.cmo, c.ncols, &c.c0s);
            let (nbf, naux, natom, nsp, nops) = (c.nbf, c.naux, c.natom, c.nsp, c.nops);
            let (a_x, b_lr, w_s, erf) = (c.a_x, c.b_lr, c.w_s, c.erf);
            let (with_k, with_lr, split, nterm) = (c.with_k, c.with_lr, c.split, c.nterm);
            // pass 2: Zd and the derivative contraction per block
            let mut acc_atom = DetAcc::new(gpu, nterm * natom * 3)?;
            let mut acc_aux = DetAcc::new(gpu, nterm * naux * 3)?;
            // fused: Gamma of a sub-block and per-term atom accumulators of the derivative
            // kernels; else the six planes of a sub-block
            let fused = self.fused;
            let natom_e = self.ders.host().natom;
            let mut acc_fused = Vec::new();
            if fused {
                for _ in 0..nterm {
                    acc_fused.push(DetAcc::new(gpu, 3 * natom_e)?);
                }
            }
            let planes = if fused { 1 } else { 6 };
            let mut dbuf = gpu.alloc_zeros_granted::<f64>(
                self.workspace.as_ref(),
                (planes * self.max_block.sub_rows * naux).max(1),
            )?;
            let mut zd = if with_k {
                Some(gpu.alloc_zeros_granted::<f64>(
                    self.workspace.as_ref(),
                    self.max_block.dense * naux,
                )?)
            } else {
                None
            };
            let mut tbuf = pass1.ut.take();
            // zeros in place of c for the erf derivative contraction of the stored planes (no
            // Coulomb term in G^w)
            let zero_c = if with_lr && !fused {
                Some(gpu.alloc_zeros_granted::<f64>(self.workspace.as_ref(), naux)?)
            } else {
                None
            };
            let (gamma, y) = (&pass1.gamma, &pass1.y);
            for b in &self.plan.blocks {
                // op 0: G with the Coulomb derivative integrals; op 1: G^w with the erf ones
                for op in 0..nops {
                    let (m0, m1) = (b.m0, b.m1);
                    let nm = m1 - m0;
                    let s = c.tl.mark()?;
                    if let (Some(zd), Some(t)) = (zd.as_mut(), tbuf.as_mut()) {
                        // Zd = sum_s w_s C_s M_s C_s^T: op 0 M = a X~ + b/2 Y~, op 1 M = b/2 X~
                        let terms: Vec<(usize, f64)> = if op == 0 {
                            let mut v = vec![];
                            if a_x != 0.0 {
                                v.push((0, a_x));
                            }
                            if with_lr {
                                v.push((1, 0.5 * b_lr));
                            }
                            v
                        } else {
                            vec![(0, 0.5 * b_lr)]
                        };
                        let mut first = true;
                        for sp in 0..nsp {
                            let o = ncols[sp];
                            if o == 0 {
                                continue;
                            }
                            let cc = c0s[sp] * nbf;
                            // T[m][j][P] = w_s sum_i C_mi M[i][j][P]
                            for (ti, &(src, f)) in terms.iter().enumerate() {
                                self.gemm.gemm(
                                    Arith::Fp64,
                                    GemmDims::new(Op::N, Op::T, o * naux, nm, o),
                                    (f * w_s, if ti == 0 { 0.0 } else { 1.0 }),
                                    MatRef::new(&y[src][sp], 0, o * naux),
                                    MatRef::new(cmo, cc + m0, nbf),
                                    MatMut::new(t, 0, o * naux),
                                )?;
                            }
                            // Zd[m][n][P] (+)= sum_j T[m][j][P] C_nj, n < m1
                            for mi in 0..nm {
                                self.gemm.gemm(
                                    Arith::Fp64,
                                    GemmDims::new(Op::N, Op::T, naux, m1, o),
                                    (1.0, if first { 0.0 } else { 1.0 }),
                                    MatRef::new(t, mi * o * naux, naux),
                                    MatRef::new(cmo, cc, nbf),
                                    MatMut::new(zd, mi * m1 * naux, naux),
                                )?;
                            }
                            first = false;
                        }
                        if first {
                            gpu.zero_range(zd, 0, nm * m1 * naux)?;
                        }
                    }
                    c.tl.close(|ts| &mut ts.zd, s)?;
                    for &(c0, c1) in &b.subs {
                        let rows = self.plan.chunk_rows(c0, c1);
                        let (p0, _) = self.plan.chunks[c0];
                        let (_, p1) = self.plan.chunks[c1 - 1];
                        if fused {
                            let nr = rows.len();
                            for (t, acc) in acc_fused.iter_mut().enumerate() {
                                // G^w (op 1) has no Coulomb term: exchange only, in the exchange
                                // part with split terms
                                let mode = match (op, split) {
                                    (0, false) => 0,
                                    (0, true) => 1 + t as i32,
                                    (_, false) => 2,
                                    (_, true) if t == 1 => 2,
                                    _ => continue,
                                };
                                if mode == 2 && !with_k {
                                    continue;
                                }
                                let s = c.tl.mark()?;
                                let blk = RowBlock {
                                    rows: &self.rows,
                                    x0: rows.start,
                                    nrows: nr,
                                    naux,
                                    m0,
                                    m1,
                                };
                                // mode 0: both terms, 1: the Coulomb term, 2: the exchange term
                                // (Zd, held with exchange)
                                let terms = GammaTerms {
                                    s: 1.0,
                                    j: (mode != 2).then_some(GammaJ {
                                        cfit: gamma,
                                        dens,
                                        nbf,
                                        symmetrize: true,
                                    }),
                                    zd: if mode != 1 { zd.as_ref() } else { None },
                                };
                                // SAFETY: rows `rows` of the tables, D, and (with exchange) Zd at
                                // ((m - m0) m1 + n) naux + P < nm m1 naux; writes nr naux values
                                // (<= dbuf's max_sub_rows naux).
                                unsafe { self.kernels.deriv.gamma(gpu, blk, terms, &mut dbuf) }?;
                                let s1 = c.tl.mark()?;
                                let o = if op == 0 { &Operator::Coulomb } else { &erf };
                                self.ders.gradient(gpu, p0..p1, o, &dbuf, acc)?;
                                c.tl.span(|ts| &mut ts.contract, s, s1);
                                c.tl.close(|ts| &mut ts.deriv, s1)?;
                            }
                            continue;
                        }
                        let s = c.tl.mark()?;
                        let o = if op == 0 { &Operator::Coulomb } else { &erf };
                        self.ders.compute(gpu, p0..p1, o, &mut dbuf, 0)?;
                        let s1 = c.tl.mark()?;
                        let ps = rows.len() * naux;
                        let zarg = match zd.as_ref() {
                            Some(z) => Arg::buf(z),
                            None => Arg::buf(&dbuf),
                        };
                        let args = [
                            Arg::I32(naux as i32),
                            Arg::I64(rows.start as i64),
                            Arg::I64(ps as i64),
                            Arg::I32(c0 as i32),
                            Arg::buf(&self.cp0),
                            Arg::buf(&self.cp1),
                            Arg::buf(&self.px0),
                            Arg::buf(&self.pnab),
                            Arg::buf(&self.pat_a),
                            Arg::buf(&self.pat_b),
                            Arg::buf(&self.pf),
                            Arg::buf(&self.rows.m),
                            Arg::buf(&self.rows.n),
                            Arg::I64(nbf as i64),
                            Arg::buf(dens),
                            match (op, zero_c.as_ref()) {
                                (0, _) | (_, None) => Arg::buf(gamma),
                                (_, Some(z)) => Arg::buf(z),
                            },
                            Arg::I32(with_k as i32),
                            zarg,
                            Arg::I64(m0 as i64),
                            Arg::I64(m1 as i64),
                            Arg::I32(split as i32),
                            Arg::I32(natom as i32),
                            Arg::buf(&dbuf),
                            acc_atom.arg(),
                            Arg::I64((nterm * natom * 3) as i64),
                            acc_aux.arg(),
                            Arg::I64((nterm * naux * 3) as i64),
                        ];
                        let cfg = Launch {
                            grid: (naux.div_ceil(128) as u32, (c1 - c0) as u32, 1),
                            block: (128, 1, 1),
                            shared_bytes: 0,
                        };
                        // SAFETY: matches gd_contract; chunks c0..c1 cover the pairs p0..p1 whose
                        // derivative planes (rows `rows`, plane stride ps) dbuf holds; rows of
                        // those pairs index the tables, D and (with exchange) Zd at
                        // ((m - m0) m1 + n) naux + P < nm m1 naux; atoms < natom, P < naux.
                        unsafe { gpu.launch(&self.kernels.contract, cfg, &args) }?;
                        c.tl.span(|ts| &mut ts.deriv, s, s1);
                        c.tl.close(|ts| &mut ts.contract, s1)?;
                    }
                }
            }
            Ok(Pass2 {
                atom: acc_atom,
                aux: acc_aux,
                fused: acc_fused,
            })
        }

        /// The downloads and the host assembly: per term the pair sides, the auxiliary sides
        /// (in function order) and the metric term `g2`; the device phases' times into `times`.
        fn assemble(
            &self,
            c: &GradCall,
            acc: Pass2,
            g2: &[Vec<[f64; 3]>],
            times: &mut GpuDfGradTimes,
        ) -> Result<GpuDfGradOutput, GpuError> {
            let (gpu, naux, natom, nterm, split) = (c.gpu, c.naux, c.natom, c.nterm, c.split);
            let fused = self.fused;
            let natom_e = self.ders.host().natom;
            let hatom = acc.atom.to_f64(gpu)?;
            let haux = acc.aux.to_f64(gpu)?;
            let hfused: Vec<Vec<f64>> = acc
                .fused
                .iter()
                .map(|a| a.to_f64(gpu))
                .collect::<Result<_, _>>()?;
            c.tl.add_to(times)?;
            // per term: pair sides, auxiliary sides (in function order), metric
            let mut terms: Vec<Vec<[f64; 3]>> = Vec::with_capacity(nterm);
            for (t, g2t) in g2.iter().enumerate() {
                let mut g = vec![[0.0; 3]; natom.max(natom_e)];
                if fused {
                    for (a, ga) in g.iter_mut().enumerate().take(natom_e) {
                        for k in 0..3 {
                            ga[k] = hfused[t][3 * a + k];
                        }
                    }
                } else {
                    for (a, ga) in g.iter_mut().enumerate().take(natom) {
                        for k in 0..3 {
                            ga[k] = hatom[(t * natom + a) * 3 + k];
                        }
                    }
                    for p in 0..naux {
                        let a = self.plan.aux_atom[p];
                        for k in 0..3 {
                            g[a][k] += haux[(t * naux + p) * 3 + k];
                        }
                    }
                }
                super::super::add_atoms(&mut g, g2t);
                terms.push(g);
            }
            let mut metric_g = vec![[0.0; 3]; natom];
            for g2t in g2 {
                super::super::add_atoms(&mut metric_g, g2t);
            }
            let mut out = GpuDfGradOutput {
                gradient: terms[0].clone(),
                metric: metric_g,
                coulomb: None,
                exchange: None,
            };
            if split {
                super::super::add_atoms(&mut out.gradient, &terms[1]);
                out.coulomb = Some(terms[0].clone());
                out.exchange = Some(terms[1].clone());
            }
            Ok(out)
        }
    }

    /// One call of [`GpuDfGrad::compute_device`]: its inputs and the sizes its phases share,
    /// and the timeline of its phases.
    struct GradCall<'a> {
        gpu: &'a Gpu,
        /// The spin channels' occupied orbitals side by side (nbf x sum(ncols)), their columns,
        /// each channel's first column.
        cmo: &'a DeviceBuf<f64>,
        ncols: &'a [usize],
        c0s: Vec<usize>,
        /// The channels: one for a closed shell (w_s = 2), alpha and beta (w_s = 1).
        nsp: usize,
        w_s: f64,
        /// a K + b K_LR(omega): a, b and the erf operator (without range separation b = 0 and
        /// the Coulomb operator).
        a_x: f64,
        b_lr: f64,
        erf: Operator,
        with_lr: bool,
        with_k: bool,
        /// The operators of passes 1 and 2: the Coulomb integrals, and with range separation
        /// the erf ones.
        nops: usize,
        /// The Coulomb and exchange parts apart (`nterm` 2) or not (1).
        split: bool,
        nterm: usize,
        nbf: usize,
        naux: usize,
        natom: usize,
        /// The layout's rows.
        nx: usize,
        tl: Timeline<'a, GpuDfGradTimes>,
    }

    /// What pass 1 leaves.
    struct Pass1 {
        /// gamma, c = V^-1 gamma after the fit.
        gamma: DeviceBuf<f64>,
        /// With exchange y\[op\]\[spin\]: Y of the Coulomb integrals (op 0) and of the erf ones
        /// (op 1), o^2 naux each; X~ and Y~ in full \[i\]\[j\]\[P\] after the fit's unpacking.
        y: Vec<Vec<DeviceBuf<f64>>>,
        /// The MO GEMMs' U (with exchange), pass 2's T.
        ut: Option<DeviceBuf<f64>>,
        /// The integral block, not read after pass 1 but allocated to the end of the call (the
        /// later phases' allocations count it against the device budget).
        _ablk: DeviceBuf<f64>,
    }

    /// What pass 2 leaves: the fixed-point accumulators.
    struct Pass2 {
        /// The stored planes' contraction: per term the pair sides per atom and the auxiliary
        /// sides per function (3 each).
        atom: DetAcc,
        aux: DetAcc,
        /// Fused: per term the derivative kernels' sums per atom.
        fused: Vec<DetAcc>,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::harness;

    /// The plan covers every pair once, in order; blocks are whole first shells with n < m1;
    /// chunks and sub-blocks tile their blocks.
    #[test]
    fn plan_tiles_the_pairs() {
        let a = 1.0 / linus_core::units::BOHR_ANGSTROM;
        let mol = harness::Structure::new(
            vec![8, 1, 1],
            vec![
                [0.0, 0.0, 0.1173 * a],
                [0.0, 0.7572 * a, -0.4692 * a],
                [0.0, -0.7572 * a, -0.4692 * a],
            ],
            0,
            1,
        )
        .unwrap();
        let basis = harness::basis(&mol, "def2-tzvp").unwrap();
        let aux = harness::basis(&mol, "def2-universal-jkfit").unwrap();
        for block_bytes in [1 << 16, 1 << 20, 2 << 30] {
            let opts = GpuDfGradOptions {
                block_bytes,
                deriv_bytes: 1 << 18,
                ..Default::default()
            };
            let plan = DfGradPlan::new(&basis, &aux, &opts);
            let np = plan.layout.pairs.len();
            let mut next_pair = 0;
            let mut next_chunk = 0;
            for b in &plan.blocks {
                assert_eq!(b.p0, next_pair);
                assert!(b.p1 > b.p0);
                for p in &plan.layout.pairs[b.p0..b.p1] {
                    let a = &basis.shells[p.sa];
                    let bb = &basis.shells[p.sb];
                    assert!(a.ao_start >= b.m0 && a.ao_start + a.nfunc() <= b.m1);
                    assert!(bb.ao_start + bb.nfunc() <= b.m1);
                }
                for &(c0, c1) in &b.subs {
                    assert_eq!(c0, next_chunk);
                    next_chunk = c1;
                }
                next_pair = b.p1;
            }
            assert_eq!(next_pair, np);
            assert_eq!(next_chunk, plan.chunks.len());
            let mut p = 0;
            for &(a, e) in &plan.chunks {
                assert_eq!(a, p);
                let rows: usize = plan.layout.pairs[a..e].iter().map(|x| x.nab()).sum();
                assert!(rows <= CHUNK_ROWS || e == a + 1);
                p = e;
            }
            assert_eq!(p, np);
        }
    }
}
