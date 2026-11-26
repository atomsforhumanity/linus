//! Density-fitted Coulomb and exchange (DF-J/K) on the device: `GpuDfJk` (feature `gpu`).
//!
//! The same quantities as linus's CPU builder (`two_electron::df::DfJk` in the `linus` crate), from the same
//! integrals and the same screening:
//! - The fitted three-center tensor B = (mn|P) L^-T, with V = L L^T the Coulomb metric of the
//!   auxiliary basis. Only the Schwarz-significant shell pairs are stored
//!   ((mn|mn)^1/2 max_P (P|P)^1/2 >= `pair_thresh`), pair-major: one row of naux values per
//!   function pair (m, n) of each kept shell pair A >= B; diagonal shell blocks hold both
//!   triangles ([`DfLayout`]).
//! - J\[D\] = B (B^T w), with w the density rows (D_mn + D_nm off the diagonal shell blocks):
//!   two passes over the tensor, bandwidth-bound.
//! - K\[D\] for D = C C^T, per auxiliary block Q: the half-transformed occupied block
//!   X_Q\[q\]\[m\]\[i\] = sum_n B_Q\[q\]\[m\]\[n\] C_ni and K += X_Q X_Q^T (nbf x nbf, inner
//!   dimension nq nocc; its lower triangle as DGEMMs over column blocks, mirrored at the end).
//!   Native K in FP64 forms X on the shell panels ([`PanelPlan`]): per shell its functions
//!   against its kept partner shells only (the dense planes are 47-60 % zeros at def2-TZVP), X
//!   auxiliary-major (\[q + nq (i + nocc m)\]) so the K update is a transposed DGEMM over
//!   contiguous columns. The panels' GEMM ([`GpuDfJkOptions::k_panel_kernel`], linus's device
//!   policy: the tensor cores on H100/H200) either reads the panels' rows straight from the
//!   stored tensor on FP64 tensor cores ([`PanelKernel::MmaSm90a`], an optional NVIDIA-only
//!   variant: mma.sync f64), or gathers the panels per block and multiplies them by cuBLAS
//!   ([`PanelKernel::Strided`], portable). Emulated K unpacks each block into the dense
//!   A_Q\[q\]\[m\]\[n\] (both triangles) and runs X_Q = A_Q^T C as one GEMM.
//! - Range-separated exchange (`GpuDfJk::set_exchange_operator`, the operator
//!   a/r + b erf(omega r)/r of a range-separated hybrid): the CPU `DfJk`'s wK scheme (Psi4's),
//!   with B^omega = (mn|erf(omega r)/r|P) L^-T, the Coulomb metric, on the same rows, so that
//!   a K + b K_LR = a Z Z^T + b (Z Zw^T + Zw Z^T) / 2 = M + M^T with Z = B^T C,
//!   Zw = (B^omega)^T C and M = Z W^T, W = (a Z + b Zw) / 2. Per auxiliary block: X_Q as above,
//!   Xw_Q from the unpacked block of B^omega, W_Q = (a X_Q + b Xw_Q) / 2, K += X_Q W_Q^T (one
//!   full GEMM); K = M + M^T at the end. Twice the exchange work of a global hybrid, as on the
//!   CPU. B^omega is built as B is, with the erf operator; its storage follows the same budget
//!   rules as B's.
//!
//! Setup, all on the device ([`GpuDfJkOptions::integrals`], [`GpuDfJkOptions::fit`]):
//! - the three-center integrals of B and B^omega from the generated kernels
//!   ([`crate::eri3c`], one builder for both operators), written in place into the tensor's rows
//!   in batches of whole shell pairs. Basis sets they do not cover (Cartesian shells, orbital
//!   l > 4) take linus-ints' CPU kernels (`pair_block`, per batch, uploaded, overlapped with the
//!   device's fit), reported in `GpuDfJkResolved::integrals_note`;
//! - the metric's Cholesky factor (cuSOLVER);
//! - the fit, in place per batch: a triangular solve with the factor (cuBLAS, FP64), or
//!   ([`DfFit::Ozaki`], `precision: "emulated"` in linus) the same solve blocked, every
//!   product an Ozaki-scheme GEMM on INT8 tensor cores, the diagonal blocks refined once
//!   (`OzFit`). The fit is most of the setup (cuBLAS's solve ran at 1.3 TFLOP/s: 28.6 s per
//!   tensor at ts113/TZVP; the Ozaki fit 5.5 s).
//!
//! Storage ([`DfStorage`]): the tensor stays on the device when it and the K workspace fit
//! the budget. Otherwise it is kept in page-locked host memory, in auxiliary blocks, and
//! streamed to the device block by block in every build (J and K share that one pass).
//!
//! A build (`GpuDfJk::compute_device`) is a sequence of phases on the stream, each a method
//! with the call's context (its operands, sizes and timeline): J's density rows; with device
//! storage, J from the whole tensor; the pass over the auxiliary blocks, each block's rows
//! (streamed from host storage), its part of J (host storage) and its exchange (native or in
//! Ozaki slices); the results in the caller's buffers. The phases are timed by events on the
//! stream (`GpuDfJkCallTimes`).
//!
//! Determinism: every kernel has one writer per output, cuBLAS is deterministic for a fixed
//! configuration, the Ozaki products sum in exact integers and recombine in a fixed order, and
//! the auxiliary blocks are summed in a fixed order on one stream, so the results are bitwise
//! reproducible run to run on a device.
//!
//! Precision ([`KMath`]): FP64 throughout, optionally with FP64 emulation on INT8 tensor cores
//! for the two exchange GEMMs: cuBLAS's fixed-point emulation (fixed mantissa), or the Ozaki
//! scheme of [`crate::emu`] with the stored tensor's scales computed once
//! ([`KMath::Ozaki`]): per auxiliary block a gather kernel writes the block's INT8 digits
//! straight from the stored rows (no FP64 unpack, no re-scaling), X = A^T C runs on them
//! against C sliced once per build, and K += X X^T as an Ozaki SYRK (X W^T for range-separated
//! exchange). Fixed slice count: FP64 quality at 7 slices (55 bits), smooth in the density,
//! bitwise reproducible.

use crate::cpu::{
    Aux3c, AuxChunk, BasisSet, DfThresholds, Operator, Pair3c, PairOptions, PairSet, Work3c,
};
use crate::eri3c::{Eri3cHost, Eri3cKind, PairRow};
use crate::error::GpuError;
use rayon::prelude::*;
use std::sync::{Arc, Mutex};

/// Where the fitted tensor lives.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DfStorage {
    /// On the device when the tensor and the K workspace fit the budget, else on the host.
    Auto,
    /// On the device (an error when it does not fit).
    Device,
    /// In page-locked host memory, streamed to the device in auxiliary blocks per build.
    Host,
}

/// Where the setup computes the three-center integrals of B (and B^omega).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DfIntegrals {
    /// On the device when the generated kernels cover both basis sets (spherical shells,
    /// orbital l <= 4, auxiliary l <= 6), else on the CPU, with the reason in
    /// `GpuDfJkResolved::integrals_note`.
    Auto,
    /// On the device ([`crate::eri3c`], written in place in the tensor's rows); an unsupported
    /// error when the kernels do not cover the basis sets.
    Device,
    /// On the CPU (linus-ints' `pair_block`, the current rayon pool), uploaded batch by batch and
    /// overlapped with the device's fit.
    Cpu,
}

/// Arithmetic of the setup's fit, B = (mn|P) L^-T.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DfFit {
    /// A triangular solve with the metric's Cholesky factor L in FP64.
    Native,
    /// The product with L^-1 as an Ozaki-scheme GEMM ([`crate::emu`]) on INT8 tensor cores with
    /// `slices` digits per value.
    Ozaki { slices: usize },
}

/// Arithmetic of the two exchange GEMMs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KMath {
    /// Native FP64.
    Native,
    /// cuBLAS's fixed-point FP64 emulation with a fixed mantissa of `bits` bits (55: FP64
    /// quality on DF data; docs/gpu.md, "Precision").
    Emulated { bits: i32 },
    /// The Ozaki scheme of [`crate::emu`] on INT8 tensor cores with `slices` digits per value
    /// (8 slices - 1 bits; 7: FP64 quality on DF data), the stored tensor's scales computed
    /// once (module docs; docs/gpu.md, "Precision").
    Ozaki { slices: usize },
}

#[cfg(feature = "gpu")]
impl KMath {
    /// The arithmetic of the exchange's cuBLAS GEMMs: emulated FP64 with its bits, else FP64
    /// (the Ozaki scheme's products are its own, [`crate::emu`]; its other GEMMs are FP64).
    pub fn cublas(self) -> linus_core::Arith {
        match self {
            KMath::Emulated { bits } => linus_core::Arith::Fp64Emulated {
                bits: bits.clamp(0, u8::MAX as i32) as u8,
            },
            KMath::Native | KMath::Ozaki { .. } => linus_core::Arith::Fp64,
        }
    }
}

/// The kernel of the shell panels' first GEMM ([`GpuDfJkOptions::k_panel_kernel`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PanelKernel {
    /// cuBLAS: per shell one strided-batched GEMM over its functions, on the panels gathered
    /// per block from the stored rows. Portable.
    Strided,
    /// The FP64 tensor-core kernel (`device/df/panel_mma.cu`: mma.sync f64, a warp-specialized
    /// producer, bulk copies), the panels' rows read straight from the stored rows, compiled
    /// for the device's own architecture: NVIDIA compute capability 9.0 and later, without
    /// sm_90a's register hand-over (tests and benchmarks on other cards).
    Mma,
    /// The same kernel compiled for sm_90a (H100/H200, compute capability 9.0 alone): the
    /// producer warpgroup's registers handed to the consumers (setmaxnreg). The same products
    /// in the same order as [`PanelKernel::Mma`].
    MmaSm90a,
}

impl PanelKernel {
    /// The name in reports: "strided", "mma", "mma_sm90a".
    pub fn name(self) -> &'static str {
        match self {
            PanelKernel::Strided => "strided",
            PanelKernel::Mma => "mma",
            PanelKernel::MmaSm90a => "mma_sm90a",
        }
    }
}

/// Settings of a `GpuDfJk` setup.
#[derive(Clone, Debug)]
pub struct GpuDfJkOptions {
    /// The screening thresholds (as the CPU `DfJk`): kept shell pairs, and the integrals'
    /// primitive pairs and auxiliary batches.
    pub thresholds: DfThresholds,
    pub storage: DfStorage,
    /// Device bytes the engine may hold (tensors and workspaces; linus passes its memory plan's);
    /// None: the device pool's limit minus what is in use at setup (an engine used on its own).
    pub device_budget: Option<usize>,
    /// Page-locked host bytes a streamed tensor may take (None: no limit; zero: the device only):
    /// the storage ladder's host rungs (`plan::plan`).
    pub host_budget: Option<usize>,
    /// The factor columns of the builds the plan sizes (the panel path's X and its block; builds
    /// with more run in passes of these): Some(0) plans J alone (no exchange workspace); None:
    /// nbf / 2.
    pub k_columns: Option<usize>,
    /// Auxiliary functions per exchange block (0: automatic, `plan::plan`: the panel path's
    /// on the device, else A_Q within `k_workspace`; shrunk on the ladder's shrunk rungs). The
    /// block sets the order of the exchange's sums over the auxiliary functions, so K changes
    /// at round-off with it.
    pub k_block: usize,
    /// Bytes of the dense block A_Q that set the automatic block of the unpacked and Ozaki
    /// exchange (and of streamed storage).
    pub k_workspace: usize,
    pub k_math: KMath,
    /// The exchange arithmetic the caller sets after the setup (`GpuDfJk::set_k_math`; with
    /// `k_math` native): its per-build workspace counts in the storage plan (the Ozaki scheme
    /// slices a gathered block, X and W where native K unpacks A_Q).
    pub k_planned: KMath,
    /// Columns of K per DGEMM of the native K update (0: 256; at most 8 blocks).
    pub k_update_block: usize,
    /// Range separation to prepare at setup (a range-separated hybrid's omega): B^omega is
    /// built with B and the storage plan counts both (as the CPU `DfConfig::omega`). None:
    /// `GpuDfJk::set_exchange_operator` builds it when first needed.
    pub omega: Option<f64>,
    /// Where B^omega lives (as the CPU `DfConfig::long_range`): Auto on the device when B is
    /// there and both fit the budget, else in pinned host memory (streamed per build); Device
    /// or Host as requested (an error when it does not fit).
    pub lr_storage: DfStorage,
    /// Ozaki exchange ([`KMath::Ozaki`]): slice the stored tensor (and B^omega) once into
    /// dense GEMM1 planes (slices nbf^2 naux bytes each) when they fit the budget beside a
    /// build's workspace and a reserve for the other engines; else every build gathers and
    /// slices each auxiliary block from the stored rows.
    pub k_presliced: bool,
    /// The kernel of native K's panel GEMM (linus passes its device policy's choice; the
    /// tensor-core kernels need NVIDIA compute capability 9.0 and later, sm_90a 9.0 itself: an
    /// error at setup elsewhere). Default: cuBLAS, portable.
    pub k_panel_kernel: PanelKernel,
    /// Bytes of the gathered panels of one shell group (0: 2 GiB): cuBLAS's panel blocks
    /// ([`PanelKernel::Strided`]) are gathered and multiplied in groups of shells within it.
    pub k_panel_workspace: usize,
    /// The tensor-core kernel's tile: auxiliary functions and orbitals per block (the largest;
    /// a call takes the fewest tiles over its orbitals), consumer warps along each, k-steps in
    /// flight (zeros: the default's, 128, 128, 4, 2, 4; the stages capped to the card's shared
    /// memory). [`PanelKernel::MmaSm90a`] needs whole warpgroups of consumers whose register
    /// count the hand-over raises.
    pub k_panel_mma_tile: [usize; 5],
    /// Device bytes the engine's optional storage (the Ozaki exchange's pre-sliced planes) and,
    /// unless `shrink_reserve` says otherwise, the storage ladder's shrunk rungs leave free for
    /// the other engines of the calculation. linus
    /// passes its memory plan's rule for the device's limit (`driver::memory`); 8 GiB for an
    /// engine used on its own.
    pub others_reserve: usize,
    /// Device bytes the storage ladder's shrunk rungs leave of `device_budget` (None:
    /// `others_reserve`; zero when the budget is already net of the other engines' reserve, as
    /// linus's SCF passes it).
    pub shrink_reserve: Option<usize>,
    /// Where the three-center integrals are computed (B and B^omega).
    pub integrals: DfIntegrals,
    /// The fit's arithmetic (B and B^omega).
    pub fit: DfFit,
    /// Ozaki fit: auxiliary functions per diagonal block (0: 512).
    pub fit_block: usize,
    /// Ozaki fit: integral columns per GEMM (0: 16384, fewer when the setup's memory is short).
    pub fit_chunk: usize,
}

impl Default for GpuDfJkOptions {
    fn default() -> Self {
        GpuDfJkOptions {
            thresholds: DfThresholds::default(),
            storage: DfStorage::Auto,
            device_budget: None,
            host_budget: None,
            k_columns: None,
            k_block: 0,
            k_workspace: 4 << 30,
            k_math: KMath::Native,
            k_planned: KMath::Native,
            k_update_block: 0,
            omega: None,
            lr_storage: DfStorage::Auto,
            k_presliced: true,
            k_panel_kernel: PanelKernel::Strided,
            k_panel_workspace: 0,
            k_panel_mma_tile: [0; 5],
            others_reserve: 8 << 30,
            shrink_reserve: None,
            integrals: DfIntegrals::Auto,
            fit: DfFit::Native,
            fit_block: 0,
            fit_chunk: 0,
        }
    }
}

/// Row kinds of [`DfLayout::row_kind`].
pub const ROW_OFFDIAG: i32 = 0;
/// A row (m, n) of a diagonal shell block with m >= n.
pub const ROW_DIAG_LOWER: i32 = 1;
/// A row (m, n) of a diagonal shell block with m < n.
pub const ROW_DIAG_UPPER: i32 = 2;

/// The kept shell pairs and stored rows of a density-fitting problem, as the CPU `DfJk` stores them
/// (no device needed).
pub struct DfLayout {
    pub nbf: usize,
    pub naux: usize,
    /// Kept shell pairs (A >= B, Schwarz-significant), in linus's pair order.
    pub pairs: Vec<Pair3c>,
    /// First row of each kept pair.
    pub x0: Vec<usize>,
    /// Stored rows (function pairs).
    pub nx: usize,
    /// Shell pairs before screening.
    pub nshell_pairs: usize,
    /// Per row: the functions (m, n) and the kind (`ROW_OFFDIAG`, `ROW_DIAG_LOWER`,
    /// `ROW_DIAG_UPPER`).
    pub row_m: Vec<i32>,
    pub row_n: Vec<i32>,
    pub row_kind: Vec<i32>,
}

impl DfLayout {
    /// The shell pairs of `basis` kept against the auxiliary basis `aux` at `pair_thresh`.
    pub fn new(basis: &BasisSet, aux: &Aux3c, pair_thresh: f64) -> DfLayout {
        Self::with_pairs(basis, None, aux, pair_thresh)
    }

    /// [`DfLayout::new`] screening `pairs` when given (the index-ordered orbital pairs of
    /// `basis` at the default primitive cut, of any derivative order: their pairs, order and
    /// Schwarz factors are those of `PairSet::build(basis, PairOptions::default())`, so the
    /// layout is the same, bit for bit) instead of building them; a set of another kind, size or
    /// primitive cut is not used.
    pub fn with_pairs(
        basis: &BasisSet,
        pairs: Option<&PairSet>,
        aux: &Aux3c,
        pair_thresh: f64,
    ) -> DfLayout {
        let built;
        let pl = match pairs.filter(|p| p.is_default_set_of(basis)) {
            Some(p) => p,
            None => {
                built = PairSet::build(basis, PairOptions::default());
                &built
            }
        };
        let kept = linus_ints::eri3c::df_kept_pairs(pl, aux.qmax, pair_thresh);
        let pairs: Vec<Pair3c> = kept
            .par_iter()
            .map(|&i| Pair3c::from_shell_pair(pl.pair(i)))
            .collect();
        let mut x0 = Vec::with_capacity(pairs.len());
        let mut nx = 0;
        for p in &pairs {
            x0.push(nx);
            nx += p.nab();
        }
        let (mut row_m, mut row_n, mut row_kind) = (
            Vec::with_capacity(nx),
            Vec::with_capacity(nx),
            Vec::with_capacity(nx),
        );
        for p in &pairs {
            let a0 = basis.shells[p.sa].ao_start;
            let b0 = basis.shells[p.sb].ao_start;
            for ia in 0..p.na {
                for ib in 0..p.nb {
                    row_m.push((a0 + ia) as i32);
                    row_n.push((b0 + ib) as i32);
                    row_kind.push(if p.sa != p.sb {
                        ROW_OFFDIAG
                    } else if ia >= ib {
                        ROW_DIAG_LOWER
                    } else {
                        ROW_DIAG_UPPER
                    });
                }
            }
        }
        DfLayout {
            nbf: basis.nbf,
            naux: aux.naux,
            pairs,
            x0,
            nx,
            nshell_pairs: pl.len(),
            row_m,
            row_n,
            row_kind,
        }
    }

    /// Bytes of the fitted tensor in f64.
    /// The layout's sizes (stored rows, shell panels) over the shells of `basis`.
    pub fn rows(&self, basis: &BasisSet) -> linus_ints::eri3c::DfRows {
        linus_ints::eri3c::DfRows::of(
            |s| basis.shells[s].nfunc(),
            self.pairs.iter().map(|p| (p.sa, p.sb)),
        )
    }

    pub fn tensor_bytes(&self) -> usize {
        self.nx * self.naux * 8
    }

    /// The kept pairs with their first rows, for [`Eri3cHost::new`].
    pub fn pair_rows(&self) -> Vec<PairRow> {
        self.pairs
            .iter()
            .zip(&self.x0)
            .map(|(p, &row)| PairRow {
                sa: p.sa,
                sb: p.sb,
                row,
            })
            .collect()
    }

    /// Consecutive runs of whole pairs with at most `max_rows` rows each (a pair larger than
    /// that alone): (first pair, end pair, first row, rows).
    pub fn row_batches(&self, max_rows: usize) -> Vec<(usize, usize, usize, usize)> {
        let mut out = Vec::new();
        let np = self.pairs.len();
        let mut p0 = 0;
        while p0 < np {
            let mut p1 = p0;
            let mut rows = 0;
            while p1 < np && (rows == 0 || rows + self.pairs[p1].nab() <= max_rows) {
                rows += self.pairs[p1].nab();
                p1 += 1;
            }
            out.push((p0, p1, self.x0[p0], rows));
            p0 = p1;
        }
        out
    }

    /// The raw integrals (mn|op|P) of the operator `op` (linus-ints' `pair_block`: 1/r,
    /// erf(omega r)/r, or a combination) of pairs p0..p1 (rows of `self.naux`) into `buf` (rows
    /// x0\[p0\].., overwritten), with the current rayon pool.
    pub fn integrals(
        &self,
        chunks: &[AuxChunk],
        prim_thresh: f64,
        op: &Operator,
        p0: usize,
        p1: usize,
        buf: &mut [f64],
    ) {
        let naux = self.naux;
        let mut slabs: Vec<&mut [f64]> = Vec::with_capacity(p1 - p0);
        let mut rest = buf;
        for p in &self.pairs[p0..p1] {
            let (a, b) = rest.split_at_mut(p.nab() * naux);
            slabs.push(a);
            rest = b;
        }
        self.pairs[p0..p1]
            .par_iter()
            .zip(slabs)
            .with_max_len(1)
            .for_each_init(Work3c::new, |w, (p, slab)| {
                slab.fill(0.0);
                crate::cpu::pair_block(p, chunks, slab, naux, prim_thresh, op, w);
            });
    }
}

/// One shell's panel in a [`PanelPlan`]: its functions f0..f0 + d against the w functions of
/// its kept partner shells (both triangles), its panel rows r0..r0 + d w (row ia + d k: function
/// ia of the shell, partner function k) and its gathered-orbital columns c0..c0 + w.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PanelShell {
    pub f0: usize,
    pub d: usize,
    pub w: usize,
    pub r0: usize,
    pub c0: usize,
}

/// The shell panels of a [`DfLayout`]: the stored rows rearranged per shell so that the exchange's
/// first GEMM, X\[q\]\[m\]\[i\] = sum_n B\[q\]\[m\]\[n\] C\[n\]\[i\], multiplies only the kept pairs
/// (the dense planes are 47-60 % zeros at def2-TZVP), one dense product per shell: for shell s
/// with functions m = f0 + ia and partner functions n_k (its kept partner shells' functions in
/// ascending order, k < w), panel row r0 + ia + d k holds the stored row of the pair (m, n_k).
/// `src[r]` is that row, `cn[c]` the partner function n_k of gathered-orbital column c = c0 + k.
/// Every element of B's dense planes outside the panels is zero, so the panels' products are
/// the dense planes' exactly (summed in another order).
pub struct PanelPlan {
    pub shells: Vec<PanelShell>,
    pub src: Vec<i32>,
    pub cn: Vec<i32>,
}

impl PanelPlan {
    /// The panels of `lay` (the kept pairs of `basis`).
    pub fn new(basis: &BasisSet, lay: &DfLayout) -> PanelPlan {
        let ns = basis.shells.len();
        let dim = |s: usize| basis.shells[s].nfunc();
        // per shell its partners (partner shell, first stored row, whether the shell is the
        // pair's first one), in ascending partner order
        let mut partners: Vec<Vec<(usize, usize, bool)>> = vec![Vec::new(); ns];
        for (p, &x0) in lay.pairs.iter().zip(&lay.x0) {
            partners[p.sa].push((p.sb, x0, true));
            if p.sa != p.sb {
                partners[p.sb].push((p.sa, x0, false));
            }
        }
        let (mut shells, mut src, mut cn) = (Vec::with_capacity(ns), Vec::new(), Vec::new());
        let (mut r0, mut c0) = (0, 0);
        for (s, ps) in partners.iter_mut().enumerate() {
            ps.sort_unstable_by_key(|e| e.0);
            let (f0, d) = (basis.shells[s].ao_start, dim(s));
            let w: usize = ps.iter().map(|e| dim(e.0)).sum();
            for &(t, x0, first) in ps.iter() {
                let dt = dim(t);
                let t0 = basis.shells[t].ao_start;
                for kb in 0..dt {
                    cn.push((t0 + kb) as i32);
                    for ia in 0..d {
                        // (m, n) = (f0 + ia, t0 + kb): the pair (s, t) stores rows ia-major,
                        // the pair (t, s) rows kb-major (diagonal blocks: both triangles)
                        let x = if first {
                            x0 + ia * dt + kb
                        } else {
                            x0 + kb * d + ia
                        };
                        src.push(x as i32);
                    }
                }
            }
            shells.push(PanelShell { f0, d, w, r0, c0 });
            r0 += d * w;
            c0 += w;
        }
        PanelPlan { shells, src, cn }
    }

    /// Panel rows (the elements of a panel block per auxiliary function).
    pub fn np(&self) -> usize {
        self.src.len()
    }

    /// Gathered-orbital columns.
    /// The widest shell's partner columns (the tensor-core kernel's row table).
    pub fn wmax(&self) -> usize {
        self.shells.iter().map(|s| s.w).max().unwrap_or(0)
    }

    pub fn ncg(&self) -> usize {
        self.cn.len()
    }
}

/// The host data of one density-fitting problem (a basis set and an auxiliary basis set at one
/// geometry) that the device engines share: the SCF's J/K (`GpuDfJk::with_shared`), the
/// gradient (`crate::grad::df::GpuDfGrad::with_shared`), the Hessian's terms
/// (`crate::hess::df::GpuDfHess::with_shared`) and the response J/K. The first setup builds it,
/// the others take it instead of rebuilding it. Everything in it is a pure function of the basis
/// sets and the thresholds, so an engine set up from it is the same, bit for bit, as one that
/// builds its own; an engine given a `DfShared` of other basis sets or thresholds
/// ([`DfShared::fits`]) builds its own.
pub struct DfShared {
    /// Fingerprint of the basis sets it was built for ([`DfShared::fingerprint`]).
    key: u64,
    /// The screening thresholds: the kept pairs' Schwarz threshold and the integrals'
    /// primitive threshold (derivative records carry their own, [`DfShared::derivatives`]).
    pub thresholds: DfThresholds,
    /// The kept pairs and their rows.
    pub layout: Arc<DfLayout>,
    /// The device integrals' packed data of the layout's pairs, or why the generated kernels do
    /// not cover the basis sets.
    pub ints: Result<Arc<Eri3cHost>, String>,
    /// The Coulomb metric (P|Q), naux x naux, formed on the CPU on first use
    /// ([`DfShared::metric`]; the memory-lean J forms its own on the device).
    metric: std::sync::OnceLock<Arc<Vec<f64>>>,
    /// The auxiliary basis's shells and Schwarz factors (the metric's input).
    aux3: Aux3c,
    /// Seconds of the build on the host.
    pub build_seconds: DfBuildSeconds,
    /// Derivative records of the layout's pairs, per kind and primitive threshold, once an
    /// engine (or a caller ahead of it, [`DfShared::derivatives`]) built them.
    derivs: Mutex<Vec<(Eri3cKind, f64, Arc<Eri3cHost>)>>,
    /// Page-locked copies of the large arrays, for DMA uploads at the engines' setups
    /// (`DfShared::stage`): the records per kind and primitive threshold (integrals at
    /// `thresholds.prim`), the metric.
    #[cfg(feature = "gpu")]
    staged: Mutex<Staged>,
}

/// Seconds of a [`DfShared`]'s build on the host.
#[derive(Clone, Copy, Debug, Default)]
pub struct DfBuildSeconds {
    /// The layout (the kept pairs and their rows).
    pub layout: f64,
    /// The integrals' packing (the device builder's records).
    pub pack: f64,
}

/// [`DfShared`]'s page-locked copies.
#[cfg(feature = "gpu")]
#[derive(Default)]
struct Staged {
    records: Vec<(Eri3cKind, f64, Arc<crate::eri3c::Eri3cStaged>)>,
    metric: Option<Arc<crate::HostBuf<f64>>>,
}

impl DfShared {
    /// The data of `basis` and `aux` at the screening thresholds `thresholds` (`pair` for the
    /// kept pairs, `prim` for the integrals' primitives); CPU work on the current rayon pool.
    pub fn new(
        basis: &BasisSet,
        aux: &BasisSet,
        thresholds: &DfThresholds,
    ) -> Result<DfShared, GpuError> {
        Self::with_pairs(basis, aux, thresholds, None)
    }

    /// [`DfShared::new`] with its layout screening `pairs` when given ([`DfLayout::with_pairs`]:
    /// the caller's index-ordered pairs of `basis`, the same layout).
    pub fn with_pairs(
        basis: &BasisSet,
        aux: &BasisSet,
        thresholds: &DfThresholds,
        pairs: Option<&PairSet>,
    ) -> Result<DfShared, GpuError> {
        let t0 = std::time::Instant::now();
        let aux3 = Aux3c::new(aux);
        let layout = DfLayout::with_pairs(basis, pairs, &aux3, thresholds.pair);
        let t1 = std::time::Instant::now();
        let prim = thresholds.prim;
        let ints = match Eri3cHost::from_pair3c(basis, aux, &layout.pairs, &layout.x0, prim) {
            Ok(h) => Ok(Arc::new(h)),
            // the packing's only input errors: shells the generated classes do not cover
            Err(GpuError::Contract(why)) => Err(why),
            Err(e) => return Err(e),
        };
        let t2 = std::time::Instant::now();
        Ok(DfShared {
            build_seconds: DfBuildSeconds {
                layout: (t1 - t0).as_secs_f64(),
                pack: (t2 - t1).as_secs_f64(),
            },
            key: Self::fingerprint(basis, aux),
            thresholds: *thresholds,
            layout: Arc::new(layout),
            ints,
            metric: std::sync::OnceLock::new(),
            aux3,
            derivs: Mutex::new(Vec::new()),
            #[cfg(feature = "gpu")]
            staged: Mutex::new(Staged::default()),
        })
    }

    /// The Coulomb metric (P|Q), naux x naux (row-major = column-major): formed on the CPU on
    /// first use (the current rayon pool), the same numbers whenever it is formed.
    pub fn metric(&self) -> &Arc<Vec<f64>> {
        self.metric
            .get_or_init(|| Arc::new(crate::cpu::metric(&self.aux3, 0.0, &Operator::Coulomb)))
    }

    /// A fingerprint of two basis sets: every shell's angular momentum, form, centre, first
    /// function, atom, exponents and coefficients (bits).
    pub fn fingerprint(basis: &BasisSet, aux: &BasisSet) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut h = std::collections::hash_map::DefaultHasher::new();
        for b in [basis, aux] {
            (b.nbf, b.shells.len(), b.atom_shells.len()).hash(&mut h);
            for s in &b.shells {
                (s.l, s.pure, s.ao_start, s.atom).hash(&mut h);
                for x in s.center.iter().chain(&s.exps).chain(&s.coefs) {
                    x.to_bits().hash(&mut h);
                }
            }
        }
        h.finish()
    }

    /// Whether it was built for `basis` and `aux` at the pair and primitive thresholds of
    /// `thresholds` (the derivative records are keyed by their own).
    pub fn fits(&self, basis: &BasisSet, aux: &BasisSet, thresholds: &DfThresholds) -> bool {
        self.thresholds.pair.to_bits() == thresholds.pair.to_bits()
            && self.thresholds.prim.to_bits() == thresholds.prim.to_bits()
            && self.layout.nbf == basis.nbf
            && self.layout.naux == aux.nbf
            && self.key == Self::fingerprint(basis, aux)
    }

    /// The derivative records (`kind`: first or second derivatives) of the layout's pairs at the
    /// primitive threshold `thresh` ([`Eri3cHost::new`] for [`DfLayout::pair_rows`]): built
    /// once (CPU work on the current rayon pool) and kept. A caller that comes while another
    /// builds them waits for that build.
    pub fn derivatives(
        &self,
        basis: &BasisSet,
        aux: &BasisSet,
        kind: Eri3cKind,
        thresh: f64,
    ) -> Result<Arc<Eri3cHost>, GpuError> {
        let mut d = self.derivs.lock().unwrap_or_else(|e| e.into_inner());
        if let Some((_, _, h)) = d
            .iter()
            .find(|(k, t, _)| *k == kind && t.to_bits() == thresh.to_bits())
        {
            return Ok(h.clone());
        }
        let h = Arc::new(Eri3cHost::new(
            basis,
            aux,
            &self.layout.pair_rows(),
            kind,
            thresh,
        )?);
        d.push((kind, thresh, h.clone()));
        Ok(h)
    }

    /// Copies the integral records, the derivative records of `kinds` (built at the primitive
    /// threshold `thresh` when not yet, [`DfShared::derivatives`]) and the metric into
    /// page-locked memory for the engines' uploads ([`DfShared::staged_records`],
    /// [`DfShared::staged_metric`]), and loads the derivative builders' kernels
    /// (`Eri3c::preload`); what is staged already is kept. CPU work on the current rayon pool.
    #[cfg(feature = "gpu")]
    pub fn stage(
        &self,
        gpu: &crate::Gpu,
        basis: &BasisSet,
        aux: &BasisSet,
        kinds: &[Eri3cKind],
        thresh: f64,
    ) -> Result<(), GpuError> {
        let mut todo: Vec<(Eri3cKind, f64, Arc<Eri3cHost>)> = Vec::new();
        if let Ok(h) = &self.ints {
            todo.push((Eri3cKind::Integrals, self.thresholds.prim, h.clone()));
        }
        for &k in kinds {
            todo.push((k, thresh, self.derivatives(basis, aux, k, thresh)?));
        }
        for (k, t, h) in todo {
            // the derivative builders' kernels too (the SCF loaded the integrals')
            if k != Eri3cKind::Integrals {
                crate::eri3c::Eri3c::preload(gpu, &h)?;
            }
            let have = self
                .staged_lock()
                .records
                .iter()
                .any(|(k2, t2, _)| *k2 == k && t2.to_bits() == t.to_bits());
            if !have {
                let s = Arc::new(crate::eri3c::Eri3cStaged::new(gpu, &h)?);
                self.staged_lock().records.push((k, t, s));
            }
        }
        if self.staged_lock().metric.is_none() {
            let metric = self.metric();
            let mut hb = gpu.alloc_host::<f64>(metric.len())?;
            hb.as_mut_slice()?.copy_from_slice(metric);
            self.staged_lock().metric = Some(Arc::new(hb));
        }
        Ok(())
    }

    #[cfg(feature = "gpu")]
    fn staged_lock(&self) -> std::sync::MutexGuard<'_, Staged> {
        self.staged.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// The page-locked copy of the records of `kind` at the primitive threshold `thresh`
    /// (the integrals' at `prim_thresh`), if staged ([`DfShared::stage`]).
    #[cfg(feature = "gpu")]
    pub fn staged_records(
        &self,
        kind: Eri3cKind,
        thresh: f64,
    ) -> Option<Arc<crate::eri3c::Eri3cStaged>> {
        self.staged_lock()
            .records
            .iter()
            .find(|(k, t, _)| *k == kind && t.to_bits() == thresh.to_bits())
            .map(|(_, _, s)| s.clone())
    }

    /// The metric's page-locked copy, if staged ([`DfShared::stage`]).
    #[cfg(feature = "gpu")]
    pub fn staged_metric(&self) -> Option<Arc<crate::HostBuf<f64>>> {
        self.staged_lock().metric.clone()
    }

    /// The metric on the device: from its page-locked copy (DMA) when staged.
    #[cfg(feature = "gpu")]
    pub fn upload_metric(&self, gpu: &crate::Gpu) -> Result<crate::DeviceBuf<f64>, GpuError> {
        let n2 = self.layout.naux * self.layout.naux;
        match self.staged_metric() {
            Some(h) if h.len() == n2 => {
                let mut d = gpu.alloc_zeros::<f64>(h.len())?;
                gpu.upload_from_host(&h, 0, &mut d, 0, h.len())?;
                Ok(d)
            }
            _ => gpu.upload(self.metric()),
        }
    }

    /// Host bytes of the data (the layout's rows and pairs' expansions, the packed integrals and
    /// derivative records, the metric), roughly.
    pub fn host_bytes(&self) -> usize {
        let lay = &self.layout;
        let pairs: usize = lay
            .pairs
            .iter()
            .map(|p| 8 * (p.et.len() + p.p.len() * 6))
            .sum();
        let packed = |h: &Eri3cHost| -> usize {
            h.classes
                .iter()
                .map(|c| 8 * (c.d.len() + c.out.len()) + 4 * (c.p0.len() + c.atom.len()))
                .sum()
        };
        let derivs: usize = self
            .derivs
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .iter()
            .map(|(_, _, h)| packed(h))
            .sum();
        12 * lay.nx
            + pairs
            + self.ints.as_ref().map_or(0, |h| packed(h))
            + derivs
            + 8 * self.layout.naux * self.layout.naux
    }
}

/// Auxiliary blocks (first function, size) of at most `nq` functions.
pub fn aux_blocks(naux: usize, nq: usize) -> Vec<(usize, usize)> {
    let nq = nq.max(1);
    (0..naux.div_ceil(nq))
        .map(|b| (b * nq, nq.min(naux - b * nq)))
        .collect()
}

/// The smallest exchange block the storage plan shrinks the automatic one to (`plan::plan`).
pub const K_BLOCK_MIN: usize = 32;

/// The automatic exchange block size: A_Q (nq nbf^2 doubles) within `workspace` bytes, 16 to
/// 512 functions (a multiple of 8), at most naux.
pub fn auto_k_block(nbf: usize, naux: usize, workspace: usize) -> usize {
    let per = 8 * nbf * nbf;
    let nq = (workspace / per.max(1)).clamp(16, 512) / 8 * 8;
    nq.min(naux).max(1)
}

#[cfg(feature = "gpu")]
pub use device::*;
#[cfg(feature = "gpu")]
pub mod plan;

#[cfg(feature = "gpu")]
mod device {
    use super::*;
    use crate::blas::{GemmDims, MatMut, MatRef, Op};
    use crate::cpu::Operator;
    use crate::df_deriv::{metric_factor, Metric, RowTables};
    use crate::emu::{Order, Ozaki, OzakiWork, Sliced, SlicedRef, Vectors, INT32_TERMS};
    use crate::engine::Timeline;
    use crate::eri3c::{Eri3c, Eri3cHost};
    use crate::error::GpuError;
    use crate::kernels::registry::KernelModule;
    use crate::linalg::{Arith, Blas1, Gemm, Operand};
    use crate::{Arg, Copies, DeviceBuf, Gpu, Kernel, Launch, StreamedStore};
    use std::borrow::Cow;
    use std::time::Instant;

    mod panels;
    use panels::*;
    pub use panels::{panel_mma_source, MmaCopy, MMA_TILE};

    /// Columns per block of the native-FP64 K update (K += X X^T as DGEMMs over column blocks of
    /// K, lower triangle only, then mirrored), and the most blocks: (nb + 1) / (2 nb) of the full
    /// product's flops, in GEMMs large enough to run at the DGEMM rate. The default of
    /// `GpuDfJkOptions::k_update_block`. Emulated K uses one full GEMM: cuBLAS slices both operands
    /// per call, so blocks would repeat that (u1an39/TZVP: 204 ms in 5 blocks, 114 ms in one).
    const K_UPDATE_BLOCK: usize = 256;
    const K_UPDATE_BLOCKS_MAX: usize = 8;
    /// The native K update's column blocks start at multiples of this many functions.
    const K_UPDATE_ALIGN: usize = 32;

    /// The engine's helper kernels (also compiled by the DF gradient, `crate::grad`).
    pub(crate) const KERNELS: &str = include_str!("../../device/df/jk.cu");

    /// The Ozaki exchange's kernels (with [`crate::emu::DEVICE_FNS`] in front).
    const OZ_KERNELS: &str = include_str!("../../device/df/ozaki.cu");

    struct OzKernels {
        fill: Kernel,
        map: Kernel,
        exponents: Kernel,
        slice: Kernel,
    }

    /// The Ozaki exchange's state ([`KMath::Ozaki`]): the map from (m, n) to stored rows, the
    /// scale exponents of B's GEMM1 vectors (q, m) (and of B^omega's for its omega), all
    /// computed once, and the per-build workspaces.
    struct OzState {
        slices: usize,
        ozaki: Ozaki,
        kernels: OzKernels,
        map: DeviceBuf<i32>,
        exps: DeviceBuf<i32>,
        lr_exps: Option<(f64, DeviceBuf<i32>)>,
        /// The pre-sliced GEMM1 planes of B (vectors q nbf + m, all functions) and of
        /// B^omega, when held.
        planes: Option<Sliced>,
        lr_planes: Option<Sliced>,
        work: OzakiWork,
        /// A block's GEMM1 operand, X's rows for GEMM2 and W's (range-separated).
        a: Option<Sliced>,
        x: Option<Sliced>,
        w: Option<Sliced>,
    }

    impl OzState {
        /// Device bytes held between builds.
        fn held_bytes(&self) -> usize {
            4 * (self.map.len() + self.exps.len())
                + self.lr_exps.as_ref().map_or(0, |(_, e)| 4 * e.len())
                + self.planes_bytes()
        }

        /// Device bytes of the pre-sliced planes.
        fn planes_bytes(&self) -> usize {
            self.planes.as_ref().map_or(0, |p| p.bytes())
                + self.lr_planes.as_ref().map_or(0, |p| p.bytes())
        }

        /// Frees the per-build workspaces.
        fn release(&mut self) {
            self.work.clear();
            self.a = None;
            self.x = None;
            self.w = None;
        }
    }

    /// The Ozaki exchange's state taken out of a [`GpuDfJk`] for a call of the response
    /// engine (crate-internal; [`GpuDfJk::take_ozaki`]), which multiplies the pre-sliced planes
    /// while it borrows the engine's storage, and put back after it.
    pub(crate) struct OzTaken(OzState);

    impl OzTaken {
        pub(crate) fn ozaki(&self) -> &Ozaki {
            &self.0.ozaki
        }

        /// B's pre-sliced GEMM1 planes (vectors q nbf + m), when held.
        pub(crate) fn planes(&self) -> Option<&Sliced> {
            self.0.planes.as_ref()
        }

        pub(crate) fn slices(&self) -> usize {
            self.0.slices
        }
    }

    /// `slot` holding a sliced operand of these sizes (reused when it matches).
    fn sliced_slot<'a>(
        gpu: &Gpu,
        slot: &'a mut Option<Sliced>,
        rows: usize,
        k: usize,
        slices: usize,
        order: Order,
    ) -> Result<&'a mut Sliced, GpuError> {
        if slot
            .as_ref()
            .is_none_or(|s| s.rows != rows || s.k != k || s.slices != slices || s.order != order)
        {
            // the old slices freed before the new ones
            *slot = None;
            *slot = Some(Sliced::new(gpu, rows, k, slices, order)?);
        }
        slot.as_mut()
            .ok_or_else(|| GpuError::Contract("DF-J/K: an empty slice slot".into()))
    }

    /// The scale exponents of the GEMM1 vectors (q, m) of a stored tensor (naux x nbf).
    fn oz_exponents(
        gpu: &Gpu,
        kern: &OzKernels,
        map: &DeviceBuf<i32>,
        (nbf, naux, nx): (usize, usize, usize),
        store: &mut Store,
        blocks: &[(usize, usize)],
    ) -> Result<DeviceBuf<i32>, GpuError> {
        let mut e = gpu.alloc_zeros::<i32>(naux * nbf)?;
        let mut run = |src: &DeviceBuf<f64>, ld: usize, q0: usize, nqb: usize| {
            if nqb == 0 {
                return Ok(());
            }
            if nx * ld > src.len() || ld < nqb {
                return Err(GpuError::Contract(
                    "DF Ozaki: block outside the tensor".into(),
                ));
            }
            // SAFETY: matches df_oz_exponents; reads map (nbf^2) and rows x < nx of src at
            // x ld + q (q < nqb <= ld, checked), writes q nbf + m < naux nbf of e.
            unsafe {
                gpu.launch(
                    &kern.exponents,
                    Launch {
                        grid: (nbf as u32, nqb.div_ceil(32) as u32, 1),
                        block: (32, 8, 1),
                        shared_bytes: 0,
                    },
                    &[
                        Arg::I64(nbf as i64),
                        Arg::I32(nqb as i32),
                        Arg::buf(map),
                        Arg::buf(src),
                        Arg::I64(ld as i64),
                        Arg::I64((q0 * nbf) as i64),
                        Arg::buf_mut(&mut e),
                    ],
                )
            }
        };
        match store {
            Store::Device(b) => run(b, naux, 0, naux)?,
            Store::Host(hs) => {
                for &(q0, nqb) in blocks {
                    hs.load(gpu, 0, q0 * nx, nqb * nx)?;
                    let staging = hs.slot(gpu, 0)?;
                    run(staging, nqb, q0, nqb)?;
                }
            }
        }
        Ok(e)
    }

    /// One auxiliary block's sliced GEMM1 operand from its stored rows (`src`: buffer,
    /// offset of the block's first function, leading dimension) into `a`, vectors
    /// r0..r0 + nqb nbf.
    #[allow(
        clippy::too_many_arguments,
        reason = "the device, the gather's tables, and the block's source and destination"
    )]
    fn oz_gather(
        gpu: &Gpu,
        kern: &OzKernels,
        map: &DeviceBuf<i32>,
        exps: &DeviceBuf<i32>,
        (nbf, nx): (usize, usize),
        (q0, nqb): (usize, usize),
        src: (&DeviceBuf<f64>, usize, usize),
        a: &mut Sliced,
        r0: usize,
    ) -> Result<(), GpuError> {
        let (buf, off, ld) = src;
        if r0 + nqb * nbf > a.rows
            || a.k != nbf
            || off + nqb > ld
            || nx * ld > buf.len()
            || (q0 + nqb) * nbf > exps.len()
        {
            return Err(GpuError::Contract(
                "DF Ozaki: block outside its buffers".into(),
            ));
        }
        // SAFETY: matches df_oz_slice; reads map, rows x < nx of the block at off + x ld + q
        // (q < nqb, checked), exponents (q0 + q) nbf + m (checked); writes vectors r0.. r0 +
        // nqb nbf of a (digits and exponents; a holds that many, checked).
        unsafe {
            gpu.launch(
                &kern.slice,
                Launch {
                    grid: (
                        nbf as u32,
                        a.kp.div_ceil(128) as u32,
                        nqb.div_ceil(32) as u32,
                    ),
                    block: (32, 8, 1),
                    shared_bytes: 0,
                },
                &[
                    Arg::I64(nbf as i64),
                    Arg::I32(nqb as i32),
                    Arg::I64(a.kp as i64),
                    Arg::I32(a.slices as i32),
                    Arg::buf(map),
                    Arg::from(buf.at(off)),
                    Arg::I64(ld as i64),
                    Arg::buf(exps),
                    Arg::I64((q0 * nbf) as i64),
                    Arg::from(a.data.at(r0 * a.slices * a.kp)),
                    Arg::from(a.exp.at(r0)),
                ],
            )
        }
    }

    /// All of a stored tensor's GEMM1 vectors in slices (`a`: naux nbf vectors and more for
    /// the padding of the last block's view), block by block.
    #[allow(
        clippy::too_many_arguments,
        reason = "the device, the gather's tables, the tensor and its blocks, and the planes"
    )]
    fn oz_planes(
        gpu: &Gpu,
        kern: &OzKernels,
        map: &DeviceBuf<i32>,
        exps: &DeviceBuf<i32>,
        (nbf, naux, nx): (usize, usize, usize),
        store: &mut Store,
        blocks: &[(usize, usize)],
        a: &mut Sliced,
    ) -> Result<(), GpuError> {
        match store {
            Store::Device(b) => oz_gather(
                gpu,
                kern,
                map,
                exps,
                (nbf, nx),
                (0, naux),
                (b, 0, naux),
                a,
                0,
            ),
            Store::Host(hs) => {
                for &(q0, nqb) in blocks {
                    hs.load(gpu, 0, q0 * nx, nqb * nx)?;
                    let staging = hs.slot(gpu, 0)?;
                    oz_gather(
                        gpu,
                        kern,
                        map,
                        exps,
                        (nbf, nx),
                        (q0, nqb),
                        (staging, 0, nqb),
                        a,
                        q0 * nbf,
                    )?;
                }
                Ok(())
            }
        }
    }

    /// Where a fitted tensor is.
    enum Store {
        /// naux x nx column-major (row x of the tensor contiguous).
        Device(DeviceBuf<f64>),
        /// Page-locked host memory in auxiliary blocks: block (q0, nq) holds nq x nx
        /// column-major at element q0 nx; streamed through one device staging buffer of the
        /// largest block, in order on the compute stream.
        Host(StreamedStore<f64>),
    }

    /// A request: densities for J (nbf x nbf, column-major), their symmetric factors for K
    /// (D_d = C_d C_d^T, C_d nbf x ncol column-major).
    pub struct GpuJkRequest<'a> {
        pub dens: &'a [&'a [f64]],
        pub factors: &'a [Factor<'a>],
        pub want_j: bool,
        pub want_k: bool,
    }

    /// A request with its operands on the device ([`GpuDfJk::compute_device`]).
    pub struct GpuJkDeviceRequest<'a> {
        /// `ndens` densities, nbf x nbf column-major, one after another.
        pub dens: Option<&'a DeviceBuf<f64>>,
        pub ndens: usize,
        /// The symmetric factors side by side, nbf x sum(ncols) column-major.
        pub factors: Option<&'a DeviceBuf<f64>>,
        /// Columns of each factor (one K per entry).
        pub ncols: &'a [usize],
    }

    /// C (nbf x ncol, column-major) of D = C C^T.
    #[derive(Clone, Copy)]
    pub struct Factor<'a> {
        pub c: &'a [f64],
        pub ncol: usize,
    }

    /// J and K per density, nbf x nbf column-major (symmetric).
    #[derive(Default)]
    pub struct GpuJkOutput {
        pub j: Vec<Vec<f64>>,
        pub k: Vec<Vec<f64>>,
    }

    /// The long-range tensor of a range-separated exchange: B^omega and where it is.
    struct LongRange {
        omega: f64,
        store: Store,
    }

    /// What a setup resolved to.
    #[derive(Clone, Debug)]
    pub struct GpuDfJkResolved {
        /// [`DfStorage::Device`] or [`DfStorage::Host`].
        pub storage: DfStorage,
        pub nbf: usize,
        pub naux: usize,
        pub nx: usize,
        pub nshell_pairs: usize,
        pub nshell_pairs_kept: usize,
        /// Bytes of the fitted tensor (f64).
        pub tensor_bytes: usize,
        /// Auxiliary functions per exchange block: the block that orders K's sums (the shell
        /// panels' on device storage with native K) and the streamed blocks.
        pub k_block: usize,
        /// The storage plan the setup took ([`plan::plan`]): its rung of the ladder, the
        /// automatic block and the rung's costs per build.
        pub plan: plan::DfPlan,
        pub k_math: KMath,
        /// Device bytes held between builds, and the budget the plan was made against.
        pub device_bytes: usize,
        pub device_budget: usize,
        /// Page-locked host bytes (host storage).
        pub host_bytes: usize,
        /// The range separation of B^omega, if held, and where B^omega is ([`DfStorage::Device`]
        /// or [`DfStorage::Host`]; its bytes are in `device_bytes` or `host_bytes`).
        pub lr_omega: Option<f64>,
        pub lr_storage: Option<DfStorage>,
        /// The exchange operator K is built for (None: 1/r).
        pub exchange_operator: Option<Operator>,
        /// Ozaki exchange: device bytes of the pre-sliced GEMM1 planes of B and B^omega (in
        /// `device_bytes`; 0: each build slices the blocks it streams).
        pub k_planes_bytes: usize,
        /// The tensor's allocation beyond the bytes granted at the setup's decision (plan
        /// violations; device storage).
        pub tensor_violations: Vec<crate::pool::PlanViolation>,
        /// Ozaki exchange: why a tensor's planes are not held (each build gathers its blocks'
        /// digits from the stored rows instead), if so.
        pub k_planes_note: Option<String>,
        /// Native K on the shell panels: the first GEMM's kernel.
        pub k_panel_kernel: PanelKernel,
        /// Where the three-center integrals were computed: [`DfIntegrals::Device`] or
        /// [`DfIntegrals::Cpu`].
        pub integrals: DfIntegrals,
        /// Why the integrals ran on the CPU under [`DfIntegrals::Auto`] (the device kernels do
        /// not cover the basis sets), if so.
        pub integrals_note: Option<String>,
        /// The fit's arithmetic.
        pub fit: DfFit,
        /// Why the fit ran in native FP64 although [`GpuDfJkOptions::fit`] asked for another,
        /// if so.
        pub fit_note: Option<String>,
    }

    /// Setup times (seconds, wall clock unless noted).
    #[derive(Clone, Debug, Default)]
    pub struct GpuDfJkSetupTimes {
        /// Shell pairs, screening, row tables (CPU).
        pub layout: f64,
        /// Metric integrals (CPU), upload and Cholesky (device).
        pub metric: f64,
        /// The device three-center builder (device integrals): the pairs packed on the CPU,
        /// uploaded, their kernels compiled or loaded from the cache.
        pub prepare: f64,
        /// B's batches queued: CPU integrals computed and uploaded while the device fits the
        /// previous batch, or the device integrals and fits queued.
        pub integrals: f64,
        /// Waiting for the device after the last batch of B (the fit's tail).
        pub fit_tail: f64,
        /// B^omega (range separation): metric and factor when rebuilt, its integrals (the erf
        /// kernels compiled or loaded from the cache), the fit, the copy to host storage.
        pub long_range: f64,
        /// The Ozaki exchange's setup ([`KMath::Ozaki`], here or in a later
        /// [`GpuDfJk::set_k_math`]): the row map, the stored tensors' scale exponents and the
        /// pre-sliced planes, waited for.
        pub ozaki: f64,
        /// The parts of B's build.
        pub b: TensorTimes,
        /// The parts of B^omega's build.
        pub lr: TensorTimes,
        pub total: f64,
    }

    /// The parts of one fitted tensor's build (seconds; "device" from events on the stream).
    #[derive(Clone, Debug, Default)]
    pub struct TensorTimes {
        /// Host time in linus-ints' CPU integrals.
        pub cpu_integrals: f64,
        /// Host time of the CPU integrals' uploads (from pageable memory).
        pub upload: f64,
        /// Device time of the generated three-center kernels.
        pub device_integrals: f64,
        /// Device time of the fit (the triangular solve with the metric's factor, or the Ozaki
        /// GEMM with its inverse).
        pub fit: f64,
        /// Device time of the Ozaki fit's slicing of the integrals.
        pub fit_slicing: f64,
        /// Device time of the copies into host storage.
        pub to_host: f64,
        /// Host time allocating the page-locked storage.
        pub host_alloc: f64,
        /// Wall time of the build, the device's tail included.
        pub wall: f64,
    }

    /// Device times of the last build (seconds, from events).
    #[derive(Clone, Debug, Default)]
    pub struct GpuDfJkCallTimes {
        /// The build: [`GpuDfJk::compute_device`]'s device work, or [`GpuDfJk::compute`]'s whole
        /// call, its uploads and downloads included.
        pub total: f64,
        /// Densities and factors to the device ([`GpuDfJk::compute`]).
        pub upload: f64,
        pub j: f64,
        /// Host storage: tensor blocks to the device.
        pub stream: f64,
        /// Unpacking the blocks (emulated K) or gathering the shell panels (the strided cuBLAS
        /// panels) of B and, range-separated, B^omega.
        pub k_unpack: f64,
        /// X = A^T C (and Xw for B^omega).
        pub k_gemm1: f64,
        /// The K update (range-separated: W = (a X + b Xw) / 2 and K += X W^T).
        pub k_gemm2: f64,
        /// Results to the host ([`GpuDfJk::compute`]).
        pub download: f64,
    }

    struct Kernels {
        density_rows: Kernel,
        rows_to_matrix: Kernel,
        unpack: Kernel,
        mirror: Kernel,
        symmetrize: Kernel,
        panel_gather: Kernel,
        panel_cg: Kernel,
        blas1: Blas1,
    }

    /// Density-fitted J and K on the device (see the module docs).
    pub struct GpuDfJk {
        nbf: usize,
        naux: usize,
        nx: usize,
        rows: RowTables,
        store: Store,
        blocks: Vec<(usize, usize)>,
        /// The GEMMs: the fit's and J's in FP64, the exchange's in `resolved.k_math`'s
        /// arithmetic ([`KMath::cublas`]).
        gemm: Gemm,
        kernels: Kernels,
        resolved: GpuDfJkResolved,
        k_update_block: usize,
        /// The setup's screening thresholds, for tensors of other operators on the same rows
        /// (B^omega, the response engine's range-separated exchange).
        thresholds: DfThresholds,
        /// The basis sets (B^omega built after setup).
        basis: BasisSet,
        aux: BasisSet,
        /// The device budget of the setup and the bytes the engine holds.
        budget: usize,
        /// Rows of a setup batch.
        batch_rows: usize,
        /// Where B^omega goes ([`GpuDfJkOptions::lr_storage`]).
        lr_storage: DfStorage,
        /// [`GpuDfJkOptions::k_presliced`].
        k_presliced: bool,
        /// [`GpuDfJkOptions::others_reserve`].
        others_reserve: usize,
        /// [`GpuDfJkOptions::k_panel_kernel`], and the tensor-core kernel's tile and its
        /// variants (compiled at their first use).
        k_panel_kernel: PanelKernel,
        mma: Option<PanelMma>,
        /// The shell panels (host) and their tables on the device (uploaded at the first panel
        /// build).
        panels: PanelPlan,
        ptabs: Option<PanelTables>,
        /// The panel path's blocks (None: not planned yet) and [`GpuDfJkOptions::k_panel_workspace`].
        pblocks: Option<PanelBlocks>,
        k_panel_workspace: usize,
        /// Whether the caller set the K block ([`GpuDfJkOptions::k_block`]): only an automatic
        /// one is fitted before the first panel build ([`GpuDfJk::shrink_panel_block`]).
        k_block_set: bool,
        /// The plan's sizes, the exchange it sized the builds for, its factor columns and a
        /// build's workspace (bytes) at its block.
        sizes: plan::DfSizes,
        planned: plan::DfExchange,
        k_columns: usize,
        planned_work: usize,
        /// The Ozaki fit's diagonal block and columns per GEMM (a later B^omega's fit).
        fit_shape: (usize, usize),
        lr: Option<LongRange>,
        /// The exchange operator a/r + b erf(omega r)/r (None: K of 1/r).
        xop: Option<Operator>,
        /// The Ozaki exchange's state ([`KMath::Ozaki`]).
        oz: Option<OzState>,
        /// [`GpuDfJk::set_keep_coulomb_fit`]: whether J builds keep their fit, the kept
        /// g = B^T w (naux x nd, column-major; the buffer is reused) and nd of the last build
        /// (0: none yet).
        keep_fit: bool,
        coulomb_fit: Option<DeviceBuf<f64>>,
        coulomb_fit_nd: usize,
        pub setup_times: GpuDfJkSetupTimes,
        pub last: GpuDfJkCallTimes,
    }

    /// One auxiliary block of a stored tensor on the device, for the builds and the crate's
    /// other engines (the response J/K, `crate::response`): functions q0..q0 + nq, element
    /// (q, x) of row x at `off + x ld + q` of `buf`.
    pub(crate) struct TensorBlock<'b> {
        pub q0: usize,
        pub nq: usize,
        pub buf: &'b DeviceBuf<f64>,
        pub off: usize,
        pub ld: usize,
    }

    /// The row tables of a [`GpuDfJk`] and its unpack kernel (crate-internal).
    pub(crate) struct Unpacker<'a> {
        nx: usize,
        nbf: usize,
        rows: &'a RowTables,
        kernel: &'a Kernel,
    }

    impl Unpacker<'_> {
        /// The dense planes a\[q\]\[m\]\[n\] (both triangles; q < nq) of the functions
        /// q_off..q_off + nq of `blk`. Entries of dropped pairs are not written: `a` must hold
        /// zeros there (zeroed once, reused).
        pub(crate) fn unpack(
            &self,
            gpu: &Gpu,
            blk: &TensorBlock<'_>,
            q_off: usize,
            nq: usize,
            a: &mut DeviceBuf<f64>,
        ) -> Result<(), GpuError> {
            self.unpack_at(gpu, blk, q_off, nq, (a, 0))
        }

        /// As [`Unpacker::unpack`], the planes into `a` from its element `a_off`.
        fn unpack_at(
            &self,
            gpu: &Gpu,
            blk: &TensorBlock<'_>,
            q_off: usize,
            nq: usize,
            (a, a_off): (&mut DeviceBuf<f64>, usize),
        ) -> Result<(), GpuError> {
            if q_off + nq > blk.nq
                || blk.off + blk.nq > blk.ld
                || self.nx * blk.ld > blk.buf.len()
                || a_off + nq * self.nbf * self.nbf > a.len()
            {
                return Err(GpuError::Contract("DF unpack: outside the block".into()));
            }
            let cfg = Launch {
                grid: (self.nx.div_ceil(32) as u32, nq.div_ceil(32) as u32, 1),
                block: (32, 8, 1),
                shared_bytes: 0,
            };
            // SAFETY: matches df_unpack; reads rows x < nx at x ld + off + q_off + q (q < nq
            // within the block's functions, which lie within ld) and writes nq planes of nbf^2
            // in a from a_off (all checked above).
            unsafe {
                gpu.launch(
                    self.kernel,
                    cfg,
                    &[
                        Arg::I64(self.nx as i64),
                        Arg::I32(nq as i32),
                        Arg::I64(self.nbf as i64),
                        Arg::buf(&self.rows.m),
                        Arg::buf(&self.rows.n),
                        Arg::buf(&self.rows.kind),
                        Arg::from(blk.buf.at(blk.off + q_off)),
                        Arg::I64(blk.ld as i64),
                        Arg::from(a.at(a_off)),
                    ],
                )
            }
        }
    }

    /// The engine's helper module (also loaded by the DF gradient and Hessian).
    pub(crate) const MODULE: KernelModule = KernelModule::new(
        "dfjk",
        || Cow::Borrowed(KERNELS),
        &[
            "df_density_rows",
            "df_rows_to_matrix",
            "df_unpack",
            "df_mirror_lower",
            "df_symmetrize",
        ],
    );

    /// Native K's shell panels ([`PanelPlan`]): the row gather and the gathered orbitals.
    pub(crate) const PANEL_MODULE: KernelModule = KernelModule::new(
        "dfjk_panels",
        || Cow::Borrowed(include_str!("../../device/df/panels.cu")),
        &["df_panel_gather", "df_panel_cg"],
    );

    /// Native K's shell panels on FP64 tensor cores ([`PanelKernel::Mma`],
    /// [`PanelKernel::MmaSm90a`]; NVIDIA-only), here with the default tile and bulk copies
    /// ([`panel_mma_source`] gives every variant).
    pub(crate) const PANEL_MMA_MODULE: KernelModule = KernelModule::new(
        "dfjk_panel_mma",
        || Cow::Owned(panel_mma_source(MMA_TILE, MmaCopy::Bulk)),
        &["df_panel_mma"],
    );

    /// The Ozaki fit's module.
    pub(crate) const FIT_MODULE: KernelModule = KernelModule::new(
        "dfjk_fit",
        || Cow::Borrowed(OZ_FIT_KERNELS),
        &["df_fit_store"],
    );

    /// The Ozaki exchange's module: the slicing's device functions, then its kernels.
    pub(crate) const OZAKI_MODULE: KernelModule = KernelModule::new(
        "dfjk_ozaki",
        || Cow::Owned(format!("{}{OZ_KERNELS}", crate::emu::DEVICE_FNS)),
        &["df_oz_fill", "df_oz_map", "df_oz_slice", "df_oz_exponents"],
    );

    fn kernels(gpu: &Gpu) -> Result<Kernels, GpuError> {
        let src = MODULE.source();
        let k = |e: &str| gpu.kernel(MODULE.name, &src, e);
        Ok(Kernels {
            density_rows: k("df_density_rows")?,
            rows_to_matrix: k("df_rows_to_matrix")?,
            unpack: k("df_unpack")?,
            mirror: k("df_mirror_lower")?,
            symmetrize: k("df_symmetrize")?,
            panel_gather: gpu.kernel(
                PANEL_MODULE.name,
                &PANEL_MODULE.source(),
                "df_panel_gather",
            )?,
            panel_cg: gpu.kernel(PANEL_MODULE.name, &PANEL_MODULE.source(), "df_panel_cg")?,
            blas1: Blas1::new(gpu)?,
        })
    }

    use super::plan::{
        ozaki_build_bytes as oz_build_bytes_for, unpacked_build_bytes as per_build_bytes,
    };

    /// The exchange the plan sizes for the arithmetic `k_math`: native K on the shell panels
    /// (`kernel`, cuBLAS's panels gathered in groups of at most `workspace` bytes), emulated K
    /// on the unpacked block, or the Ozaki scheme.
    fn exchange_of(k_math: KMath, kernel: PanelKernel, workspace: usize) -> plan::DfExchange {
        match k_math {
            KMath::Native => plan::DfExchange::Panels { kernel, workspace },
            KMath::Emulated { .. } => plan::DfExchange::Unpacked,
            KMath::Ozaki { slices } => plan::DfExchange::Ozaki { slices },
        }
    }

    /// The storage plan's request of a setup with `opts` for `sizes` within `budget` device
    /// bytes, its setup holding `setup_extra` bytes beside the tensors
    /// ([`plan::setup_bytes`], and the device integrals' builder where it packs one): what the
    /// setup plans, and what a caller planning ahead of it asks ([`plan::plan`]).
    pub fn plan_request(
        opts: &GpuDfJkOptions,
        sizes: plan::DfSizes,
        budget: usize,
        setup_extra: usize,
    ) -> plan::DfPlanRequest {
        plan::DfPlanRequest {
            sizes,
            lr: opts.omega.is_some(),
            exchange: if opts.k_columns == Some(0) {
                plan::DfExchange::None
            } else {
                exchange_of(
                    match opts.k_math {
                        KMath::Native => opts.k_planned,
                        k => k,
                    },
                    opts.k_panel_kernel,
                    k_panel_workspace(opts),
                )
            },
            k_columns: k_columns(opts, sizes.nbf),
            budget,
            reserve: opts.shrink_reserve.unwrap_or(opts.others_reserve),
            host_budget: opts.host_budget,
            storage: opts.storage,
            lr_storage: opts.lr_storage,
            k_block: opts.k_block,
            k_workspace: opts.k_workspace,
            setup_extra,
        }
    }

    /// [`GpuDfJkOptions::k_panel_workspace`] (0: [`plan::PANEL_WORKSPACE`]).
    fn k_panel_workspace(opts: &GpuDfJkOptions) -> usize {
        if opts.k_panel_workspace > 0 {
            opts.k_panel_workspace
        } else {
            plan::PANEL_WORKSPACE
        }
    }

    /// [`GpuDfJkOptions::k_columns`] (None: nbf / 2).
    fn k_columns(opts: &GpuDfJkOptions, nbf: usize) -> usize {
        opts.k_columns.unwrap_or_else(|| nbf.div_ceil(2))
    }

    /// The range-separation parameter of an operator with an erf part, checked.
    fn lr_omega(op: &Operator) -> Result<f64, GpuError> {
        if !op.has_erf() || op.omega() < 0.0 || !op.omega().is_finite() {
            return Err(GpuError::Contract(format!(
                "DF-J/K: {op:?} has no long-range part (a global exchange operator a/r is K scaled by a)"
            )));
        }
        Ok(op.omega())
    }

    /// Device and pinned host bytes of a stored tensor.
    fn store_bytes(s: &Store) -> (usize, usize) {
        match s {
            Store::Device(b) => (b.len() * 8, 0),
            Store::Host(hs) => hs.bytes(),
        }
    }

    /// Device bytes of a three-center builder of `h`: the packed pairs and auxiliary
    /// primitives, plus a margin for the Boys table and the kernels' tables.
    fn eri3c_bytes(h: &Eri3cHost) -> usize {
        let classes: usize = h
            .classes
            .iter()
            .filter(|c| !c.pairs.is_empty())
            .map(|c| 4 * c.p0.len() + 8 * c.d.len() + 8 * c.out.len() + 4 * c.atom.len())
            .sum();
        let aux: usize = h
            .aux
            .iter()
            .filter(|a| a.nshell() > 0)
            .map(|a| 4 * a.p0.len() + 8 * a.d.len() + 4 * (a.col.len() + a.atom.len()))
            .sum();
        classes + aux + (16 << 20)
    }

    /// The packed host data of the device integrals of `lay`'s pairs, or why the generated
    /// kernels do not cover the basis sets (`Err(reason)`; other errors propagate).
    fn device_integrals_host(
        basis: &BasisSet,
        aux: &BasisSet,
        lay: &DfLayout,
        prim_thresh: f64,
    ) -> Result<Result<Eri3cHost, String>, GpuError> {
        match Eri3cHost::from_pair3c(basis, aux, &lay.pairs, &lay.x0, prim_thresh) {
            Ok(h) => Ok(Ok(h)),
            // the packing's only input errors: shells the generated classes do not cover
            Err(GpuError::Contract(why)) => Ok(Err(why)),
            Err(e) => Err(e),
        }
    }

    /// Where a setup's three-center integrals come from.
    enum IntSource<'a> {
        /// The generated device kernels, written in place.
        Device(&'a Eri3c),
        /// linus-ints on the CPU (auxiliary chunks, primitive threshold), uploaded per batch.
        Cpu(&'a [AuxChunk], f64),
    }

    /// Auxiliary functions per diagonal block of the Ozaki fit's blocked triangular solve
    /// (sm_120, def2-TZVP with universal-jkfit: 512 fastest from u1an39 to ts113 among
    /// 128-1024; smaller blocks re-slice the solved values more often, larger ones spend more
    /// on the diagonal products).
    const OZ_FIT_BLOCK: usize = 512;

    /// Integral columns (tensor rows) per Ozaki-fit GEMM: the default.
    const OZ_FIT_CHUNK: usize = 16384;

    /// The Ozaki fit's kernel: a block of solved columns stored into the tensor's rows.
    const OZ_FIT_KERNELS: &str = include_str!("../../device/df/fit.cu");

    /// One diagonal block of the Ozaki fit: its functions P0..P1, L's rows P0..P1 over k < P0
    /// (none for the first block), L_ii and its inverse (lower triangular, FP64 from a
    /// triangular solve of the identity), all in slices (ascending).
    struct FitBlock {
        p0: usize,
        p1: usize,
        left: Option<Sliced>,
        l: Sliced,
        linv: Sliced,
    }

    /// The fit B = (mn|P) L^-T, X = L^-1 R per integral column, as a blocked left-looking
    /// triangular solve in the Ozaki scheme ([`DfFit::Ozaki`]). For the diagonal blocks i of L
    /// (functions P0..P1), per chunk of columns:
    /// - R_i -= L_i,<i X_<i (L's rows over k < P0 sliced once, the solved values X_<i sliced
    ///   per block);
    /// - X_i = L_ii^-1 R_i with one step of refinement: X0 = L_ii^-1 R_i, r = R_i - L_ii X0,
    ///   X_i = X0 + L_ii^-1 r (three GEMMs).
    ///
    /// Every product is an Ozaki GEMM. The updates and the residual multiply L with solved
    /// values, both O(1) here, so their fixed-point error is of the size of FP64's: the
    /// solve keeps the triangular solve's backward error. The products with L_ii^-1 alone
    /// would not (an explicit inverse multiplies the error by |L^-1| |R| / |X|: 1e-9 for the
    /// whole L^-1 at u1an39/TZVP); the refinement removes their error to second order. FP64
    /// triangular solves of the diagonal blocks instead (cuBLAS, FP64-bound on this card)
    /// took half the fit's time.
    struct OzFit {
        ozaki: Ozaki,
        slices: usize,
        blocks: Vec<FitBlock>,
        /// A chunk's columns in slices (descending), allocated for `chunk` columns of naux
        /// values; its sizes are set per product.
        cols: Sliced,
        /// The diagonal solve's X0 and X (nb x chunk).
        t: DeviceBuf<f64>,
        store: Kernel,
        /// The diagonal blocks' size asked for, and the columns per GEMM.
        nb: usize,
        chunk: usize,
        work: OzakiWork,
    }

    impl OzFit {
        /// The diagonal blocks (P0, P1) of about `nb` functions, boundaries at multiples of
        /// [`ROW_ALIGN`](crate::emu::ROW_ALIGN).
        fn bounds(naux: usize, nb: usize) -> Vec<(usize, usize)> {
            let nb = nb.max(1).next_multiple_of(crate::emu::ROW_ALIGN);
            (0..naux.div_ceil(nb))
                .map(|j| (j * nb, ((j + 1) * nb).min(naux)))
                .collect()
        }

        /// Device bytes of the fit for `naux` functions: L's blocks in slices, a chunk's
        /// sliced columns and X0, the INT32 partial sums and the maxima.
        fn bytes(naux: usize, slices: usize, nb: usize, chunk: usize) -> usize {
            let bounds = Self::bounds(naux, nb);
            let sl: usize = bounds
                .iter()
                .map(|&(p0, p1)| {
                    let n = p1 - p0;
                    Sliced::bytes_for(n, p0.max(1), slices) + 2 * Sliced::bytes_for(n, n, slices)
                })
                .sum();
            let np: usize = (0..slices)
                .map(|g| crate::emu::groups(&crate::emu::diagonal(g, slices, slices), naux).len())
                .sum();
            let mp = bounds
                .iter()
                .map(|&(p0, p1)| (p1 - p0).next_multiple_of(crate::emu::ROW_ALIGN))
                .max()
                .unwrap_or(0);
            sl + Sliced::bytes_for(chunk, naux, slices)
                + 8 * mp * chunk
                + 4 * np * mp * chunk
                + 8 * chunk
                + 8 * mp * mp
                + (1 << 20)
        }

        /// L's blocks (the metric's factor `chol`, naux x naux, lower triangle) and the
        /// diagonal blocks' inverses in `slices` digits, diagonal blocks of about `nb`
        /// functions, GEMMs over `chunk` columns.
        fn new(
            gpu: &Gpu,
            gemm: &Gemm,
            chol: &DeviceBuf<f64>,
            naux: usize,
            slices: usize,
            nb: usize,
            chunk: usize,
        ) -> Result<OzFit, GpuError> {
            let ozaki = Ozaki::new(gpu)?;
            let store = gpu.kernel(FIT_MODULE.name, &FIT_MODULE.source(), "df_fit_store")?;
            let mut work = OzakiWork::default();
            // the factor's lower triangle (its upper one holds the metric)
            let lh = gpu.download(chol)?;
            fn rows_of(buf: &DeviceBuf<f64>, off: usize, ld: usize) -> Vectors<'_> {
                Vectors {
                    buf,
                    off,
                    ld,
                    contiguous: false,
                }
            }
            let mut blocks = Vec::new();
            for (p0, p1) in Self::bounds(naux, nb) {
                let n = p1 - p0;
                let left = if p0 == 0 {
                    None
                } else {
                    // vector r: row p0 + r of L (column-major), elements k < p0
                    let mut sl = Sliced::new(gpu, n, p0, slices, Order::Ascending)?;
                    ozaki.slice(gpu, rows_of(chol, p0, naux), &mut sl, &mut work)?;
                    Some(sl)
                };
                // L_ii (n x n, column-major, zeros above the diagonal) and L_ii^-1
                let mut lii = vec![0.0f64; n * n];
                for j in 0..n {
                    for i in j..n {
                        lii[j * n + i] = lh[(p0 + j) * naux + p0 + i];
                    }
                }
                let ld = gpu.upload(&lii)?;
                let mut id = vec![0.0f64; n * n];
                for i in 0..n {
                    id[i * n + i] = 1.0;
                }
                let mut li = gpu.upload(&id)?;
                gemm.trsm_lower_left(
                    Op::N,
                    (n, n),
                    MatRef::new(&ld, 0, n),
                    MatMut::new(&mut li, 0, n),
                )?;
                let mut l = Sliced::new(gpu, n, n, slices, Order::Ascending)?;
                ozaki.slice(gpu, rows_of(&ld, 0, n), &mut l, &mut work)?;
                let mut linv = Sliced::new(gpu, n, n, slices, Order::Ascending)?;
                ozaki.slice(gpu, rows_of(&li, 0, n), &mut linv, &mut work)?;
                blocks.push(FitBlock {
                    p0,
                    p1,
                    left,
                    l,
                    linv,
                });
            }
            let chunk = chunk.max(1);
            let cols = Sliced::new(gpu, chunk, naux, slices, Order::Descending)?;
            let mp = blocks.iter().map(|b| b.p1 - b.p0).max().unwrap_or(1);
            let t = gpu.alloc_zeros::<f64>(mp * chunk)?;
            gpu.synchronize()?;
            Ok(OzFit {
                ozaki,
                slices,
                blocks,
                cols,
                t,
                store,
                nb,
                chunk,
                work,
            })
        }

        /// Slices `cn` columns of `k` values (column c's at off + c ld of `buf`) into `cols`.
        #[allow(
            clippy::too_many_arguments,
            reason = "the slicer, its workspace, and the columns' buffer and layout"
        )]
        fn slice_cols(
            ozaki: &Ozaki,
            gpu: &Gpu,
            cols: &mut Sliced,
            work: &mut OzakiWork,
            buf: &DeviceBuf<f64>,
            off: usize,
            ld: usize,
            cn: usize,
            k: usize,
        ) -> Result<(), GpuError> {
            let align = |n: usize, a: usize| n.div_ceil(a).max(1) * a;
            // the cols buffer holds chunk vectors of naux values per slice: any cn and k fit
            cols.rows = cn;
            cols.rows_pad = align(cn, crate::emu::ROW_ALIGN);
            cols.k = k;
            cols.kp = align(k, crate::emu::K_ALIGN);
            ozaki.slice(
                gpu,
                Vectors {
                    buf,
                    off,
                    ld,
                    contiguous: true,
                },
                cols,
                work,
            )
        }

        /// Solves L X = R in place for `rows` integral columns of naux values at `off` of
        /// `buf`.
        fn apply(
            &mut self,
            gpu: &Gpu,
            buf: &mut DeviceBuf<f64>,
            off: usize,
            rows: usize,
            naux: usize,
            tl: &mut Timeline<'_, TensorTimes>,
        ) -> Result<(), GpuError> {
            let (oz, cols, work, t) = (&self.ozaki, &mut self.cols, &mut self.work, &mut self.t);
            let mut c0 = 0;
            while c0 < rows {
                let cn = self.chunk.min(rows - c0);
                let at = off + c0 * naux;
                for b in &self.blocks {
                    let (p0, n) = (b.p0, b.p1 - b.p0);
                    let ri = at + p0;
                    if let Some(left) = &b.left {
                        // R_i -= L_i,<i X_<i
                        let m0 = tl.mark()?;
                        Self::slice_cols(oz, gpu, cols, work, buf, at, naux, cn, p0)?;
                        let (m1, m1b) = (tl.mark()?, tl.mark()?);
                        oz.gemm(
                            gpu,
                            left.view(),
                            cols.view(),
                            -1.0,
                            1.0,
                            (&mut *buf, ri, naux),
                            work,
                        )?;
                        tl.span(|t| &mut t.fit_slicing, m0, m1);
                        tl.close(|t| &mut t.fit, m1b)?;
                    }
                    // X0 = L_ii^-1 R_i into t
                    let m0 = tl.mark()?;
                    Self::slice_cols(oz, gpu, cols, work, buf, ri, naux, cn, n)?;
                    let (m1, m1b) = (tl.mark()?, tl.mark()?);
                    oz.gemm(
                        gpu,
                        b.linv.view(),
                        cols.view(),
                        1.0,
                        0.0,
                        (&mut *t, 0, n),
                        work,
                    )?;
                    tl.span(|t| &mut t.fit_slicing, m0, m1);
                    tl.close(|t| &mut t.fit, m1b)?;
                    // r = R_i - L_ii X0 in place of R_i
                    let m0 = tl.mark()?;
                    Self::slice_cols(oz, gpu, cols, work, t, 0, n, cn, n)?;
                    let (m1, m1b) = (tl.mark()?, tl.mark()?);
                    oz.gemm(
                        gpu,
                        b.l.view(),
                        cols.view(),
                        -1.0,
                        1.0,
                        (&mut *buf, ri, naux),
                        work,
                    )?;
                    tl.span(|t| &mut t.fit_slicing, m0, m1);
                    tl.close(|t| &mut t.fit, m1b)?;
                    // X_i = X0 + L_ii^-1 r, stored over R_i
                    let m0 = tl.mark()?;
                    Self::slice_cols(oz, gpu, cols, work, buf, ri, naux, cn, n)?;
                    let (m1, m1b) = (tl.mark()?, tl.mark()?);
                    oz.gemm(
                        gpu,
                        b.linv.view(),
                        cols.view(),
                        1.0,
                        1.0,
                        (&mut *t, 0, n),
                        work,
                    )?;
                    // SAFETY: matches df_fit_store; reads n x cn of t (ld n, t holds mp chunk
                    // >= that), writes n rows of cn columns of buf from ri with ld naux (the
                    // columns lie in the batch, checked by the GEMM above).
                    unsafe {
                        gpu.launch(
                            &self.store,
                            Launch::linear(n * cn, 256),
                            &[
                                Arg::I64(n as i64),
                                Arg::I64(cn as i64),
                                Arg::buf(t),
                                Arg::I64(n as i64),
                                Arg::from(buf.at(ri)),
                                Arg::I64(naux as i64),
                            ],
                        )?;
                    }
                    tl.span(|t| &mut t.fit_slicing, m0, m1);
                    tl.close(|t| &mut t.fit, m1b)?;
                }
                c0 += cn;
            }
            Ok(())
        }
    }

    /// How a tensor build fits its batches: the triangular solve with the metric's factor L
    /// (`chol`), blocked with Ozaki-scheme updates when `oz` is set.
    struct Fitter<'a> {
        chol: &'a DeviceBuf<f64>,
        oz: Option<&'a mut OzFit>,
    }

    /// One fitted tensor on the rows of `lay`, (mn|op|P) L^-T with V = L L^T the Coulomb metric:
    /// the integrals of `op` (1/r for B, erf(omega r)/r for B^omega) per batch of
    /// whole pairs of at most `batch_rows` rows, fitted in place per batch (`fitter`), into
    /// device storage or into page-locked host memory in the auxiliary blocks
    /// `blocks` (with a staging buffer of `nq` functions for the builds). Batches are queued
    /// on the stream, the CPU's integrals of a batch overlapping the device's fit of the
    /// previous one; the device is waited for at the end. `times` gets the parts.
    #[allow(
        clippy::too_many_arguments,
        reason = "the device, the fit and the integrals' source, the layout and the storage plan"
    )]
    fn build_tensor(
        gpu: &Gpu,
        gemm: &Gemm,
        src: &IntSource<'_>,
        op: &Operator,
        lay: &DfLayout,
        fitter: &mut Fitter<'_>,
        batch_rows: usize,
        storage: DfStorage,
        blocks: &[(usize, usize)],
        nq: usize,
        times: &mut TensorTimes,
        grant: Option<&crate::pool::Grant>,
    ) -> Result<(Store, f64), GpuError> {
        let t_all = Instant::now();
        let (naux, nx) = (lay.naux, lay.nx);
        let batches = lay.row_batches(batch_rows);
        let mut hbuf = match src {
            IntSource::Cpu(..) => vec![0.0f64; batch_rows * naux],
            IntSource::Device(_) => Vec::new(),
        };
        // the device phases by events on the stream, added up once the build is waited for
        let mut tl = Timeline::<TensorTimes>::new(gpu);
        // the integrals of pairs p0..p1 (rows rows) into buf from element off on
        let mut fill = |buf: &mut DeviceBuf<f64>,
                        off: usize,
                        (p0, p1, rows): (usize, usize, usize),
                        tl: &mut Timeline<'_, TensorTimes>,
                        times: &mut TensorTimes|
         -> Result<(), GpuError> {
            match src {
                IntSource::Device(eri) => {
                    let m0 = tl.mark()?;
                    eri.compute(gpu, p0..p1, op, buf, off)?;
                    tl.close(|t| &mut t.device_integrals, m0)?;
                }
                IntSource::Cpu(chunks, thresh) => {
                    let hb = &mut hbuf[..rows * naux];
                    let t = Instant::now();
                    lay.integrals(chunks, *thresh, op, p0, p1, hb);
                    times.cpu_integrals += t.elapsed().as_secs_f64();
                    let t = Instant::now();
                    gpu.upload_into(buf, off, hb)?;
                    times.upload += t.elapsed().as_secs_f64();
                }
            }
            Ok(())
        };
        let mut fit = |buf: &mut DeviceBuf<f64>,
                       off: usize,
                       rows: usize,
                       tl: &mut Timeline<'_, TensorTimes>|
         -> Result<(), GpuError> {
            match fitter.oz.as_deref_mut() {
                None => {
                    let m0 = tl.mark()?;
                    gemm.trsm_lower_left(
                        Op::N,
                        (naux, rows),
                        MatRef::new(fitter.chol, 0, naux),
                        MatMut::new(buf, off, naux),
                    )?;
                    tl.close(|t| &mut t.fit, m0)?;
                }
                Some(f) => f.apply(gpu, buf, off, rows, naux, tl)?,
            }
            Ok(())
        };
        let store = match storage {
            DfStorage::Device => {
                if let Some(g) = grant {
                    g.draw(8 * nx * naux);
                }
                let mut b = gpu.alloc_zeros::<f64>(nx * naux)?;
                for &(p0, p1, r0, rows) in &batches {
                    fill(&mut b, r0 * naux, (p0, p1, rows), &mut tl, times)?;
                    fit(&mut b, r0 * naux, rows, &mut tl)?;
                }
                Store::Device(b)
            }
            _ => {
                let t = Instant::now();
                let mut host = gpu.alloc_host::<f64>(nx * naux)?;
                times.host_alloc += t.elapsed().as_secs_f64();
                let mut stage = gpu.alloc_zeros::<f64>(batch_rows * naux)?;
                for &(p0, p1, r0, rows) in &batches {
                    fill(&mut stage, 0, (p0, p1, rows), &mut tl, times)?;
                    fit(&mut stage, 0, rows, &mut tl)?;
                    // rows r0.. of each block: nq x rows at q0 nx + r0 nq
                    let m0 = tl.mark()?;
                    for &(q0, nqb) in blocks {
                        gpu.download_block_to_host(
                            &stage,
                            q0,
                            naux,
                            &mut host,
                            q0 * nx + r0 * nqb,
                            nqb,
                            nqb,
                            rows,
                        )?;
                    }
                    tl.close(|t| &mut t.to_host, m0)?;
                }
                drop(stage);
                Store::Host(StreamedStore::with_host(
                    gpu,
                    host,
                    (1, nq * nx),
                    Copies::InOrder,
                )?)
            }
        };
        let queued = t_all.elapsed().as_secs_f64();
        gpu.synchronize()?;
        tl.add_to(times)?;
        times.wall += t_all.elapsed().as_secs_f64();
        Ok((store, queued))
    }

    impl GpuDfJk {
        /// Sets up `basis` with the auxiliary basis `aux` (see the module docs). CPU work runs
        /// on the current rayon pool.
        pub fn new(
            gpu: &Gpu,
            basis: &BasisSet,
            aux: &BasisSet,
            opts: &GpuDfJkOptions,
        ) -> Result<GpuDfJk, GpuError> {
            Self::with_shared(gpu, basis, aux, opts, None)
        }

        /// As [`GpuDfJk::new`], with the layout, the packed integrals and the metric of
        /// `shared` when it fits the basis sets and thresholds (the same engine, bit for bit).
        pub fn with_shared(
            gpu: &Gpu,
            basis: &BasisSet,
            aux: &BasisSet,
            opts: &GpuDfJkOptions,
            shared: Option<&DfShared>,
        ) -> Result<GpuDfJk, GpuError> {
            let t_all = Instant::now();
            let mut times = GpuDfJkSetupTimes::default();
            let t = Instant::now();
            let shared = shared.filter(|s| s.fits(basis, aux, &opts.thresholds));
            let aux3 = Aux3c::new(aux);
            let chunks = aux3.chunks(0..aux.shells.len());
            let lay_own;
            let lay: &DfLayout = match shared {
                Some(s) => &s.layout,
                None => {
                    lay_own = DfLayout::new(basis, &aux3, opts.thresholds.pair);
                    &lay_own
                }
            };
            let (nbf, naux, nx) = (lay.nbf, lay.naux, lay.nx);
            if nx == 0 || naux == 0 {
                return Err(GpuError::Contract(
                    "DF-J/K: no shell pair or no auxiliary function".into(),
                ));
            }
            let rows = RowTables::upload(gpu, lay)?;
            // the shell panels on the host; their tables go to the device at the first panel
            // build (native K), not for engines that never run one
            let panels = PanelPlan::new(basis, lay);
            // the tensor-core kernel: the device and the tile checked now, its variants compiled
            // at their first use
            let mma = match opts.k_panel_kernel {
                PanelKernel::Strided => None,
                kind => Some(PanelMma::new(
                    gpu,
                    kind,
                    opts.k_panel_mma_tile,
                    panels.wmax(),
                )?),
            };
            times.layout = t.elapsed().as_secs_f64();

            // the engine's allowance: the caller's `device_budget` bytes, else the device's free
            // memory
            let allowance = match opts.device_budget {
                Some(b) => linus_core::Budget::new(b, 0),
                None => gpu.budget(),
            };
            let budget = allowance.free();
            let sizes = plan::DfSizes {
                nbf,
                naux,
                rows: lay.rows(basis),
            };
            // a setup batch's rows (about 1 GiB)
            let batch_rows = plan::batch_rows(naux, sizes.rows.max_pair_rows);
            let omega = match opts.omega {
                Some(w) => Some(lr_omega(&Operator::LongRange { omega: w })?),
                None => None,
            };
            // the device integrals' packed pairs, unless the CPU computes them
            let t = Instant::now();
            let (eri_host, integrals_note) = match opts.integrals {
                DfIntegrals::Cpu => (None, None),
                want => match match shared {
                    Some(s) => s.ints.clone(),
                    None => {
                        device_integrals_host(basis, aux, lay, opts.thresholds.prim)?.map(Arc::new)
                    }
                } {
                    Ok(h) => (Some(h), None),
                    Err(why) if want == DfIntegrals::Auto => (None, Some(why)),
                    Err(why) => {
                        return Err(GpuError::NotCovered(format!(
                            "DF-J/K: device integrals: {why}"
                        )))
                    }
                },
            };
            times.prepare = t.elapsed().as_secs_f64();
            // at setup: the metric, one batch of rows (host storage's staging) and the device
            // integrals' builder
            let setup_extra =
                plan::setup_bytes(&sizes) + eri_host.as_deref().map_or(0, eri3c_bytes);
            let tensor = lay.tensor_bytes();
            // the storage, the exchange block and a build's workspace: the plan of the sizes and
            // the allowance, for the exchange arithmetic planned (the Ozaki scheme slices a
            // gathered block, X and W where emulated K unpacks A_Q and native K multiplies the
            // shell panels)
            let request = plan_request(opts, sizes, budget, setup_extra);
            let dfplan = plan::plan(&request).map_err(|r| {
                if r.host_limited {
                    GpuError::NotCovered(r.reason)
                } else {
                    GpuError::from(linus_core::Budget::new(r.budget, 0).over(
                        "DF-J/K: the tensor and a build's workspace (on any rung of the storage ladder)",
                        r.need,
                    ))
                }
            })?;
            let (storage, lr_storage, nq) = (dfplan.storage, dfplan.lr_storage, dfplan.k_block);
            let work = dfplan.device_bytes
                - if storage == DfStorage::Device {
                    tensor
                } else {
                    8 * nq * nx
                }
                - match lr_storage {
                    Some(DfStorage::Device) => tensor,
                    Some(_) => 8 * nq * nx,
                    None => 0,
                };
            let blocks = aux_blocks(naux, nq);
            // the fit: the Ozaki scheme, when asked for, within the budget the setup leaves, with
            // fewer columns per GEMM if need be; else the triangular solve
            let oz_fit = match opts.fit {
                DfFit::Native => Ok(None),
                DfFit::Ozaki { slices } if !(1..=8).contains(&slices) => {
                    return Err(GpuError::Contract(format!(
                        "DF-J/K: Ozaki fit with {slices} slices (1 to 8)"
                    )))
                }
                DfFit::Ozaki { slices } => {
                    // the budget beside the stored tensors (or their staging) and what the setup
                    // holds: the metric's factor, the device integrals' builder, a batch's
                    // staging for host storage (the builds' workspace is not allocated yet, so
                    // the plan does not change)
                    let host = storage == DfStorage::Host || lr_storage == Some(DfStorage::Host);
                    let held = 8 * naux * naux
                        + eri_host.as_deref().map_or(0, eri3c_bytes)
                        + if host {
                            8 * batch_rows.min(nx) * naux
                        } else {
                            0
                        };
                    let stored = dfplan.device_bytes - work;
                    let room = budget.saturating_sub(stored + held);
                    let nb = if opts.fit_block > 0 {
                        opts.fit_block
                    } else {
                        OZ_FIT_BLOCK
                    };
                    let mut chunk = if opts.fit_chunk > 0 {
                        opts.fit_chunk
                    } else {
                        OZ_FIT_CHUNK
                    };
                    while chunk > 256 && OzFit::bytes(naux, slices, nb, chunk) > room {
                        chunk /= 2;
                    }
                    let need = OzFit::bytes(naux, slices, nb, chunk);
                    if need <= room {
                        Ok(Some((slices, nb, chunk)))
                    } else {
                        Err(format!(
                            "the Ozaki fit's workspace ({:.2} GiB) does not fit the {:.2} GiB the setup leaves of the \
                             device budget: the fit is a triangular solve in FP64",
                            need as f64 / 1073741824.0,
                            room as f64 / 1073741824.0
                        ))
                    }
                }
            };
            let (oz_fit, fit_note) = match oz_fit {
                Ok(f) => (f, None),
                Err(why) => (None, Some(why)),
            };

            // metric and its Cholesky factor (and, Ozaki fit, its inverse in slices)
            let t = Instant::now();
            let metric = match shared {
                Some(s) => Metric::Staged(s),
                None => Metric::Formed(&aux3),
            };
            let (chol, _) = metric_factor(gpu, metric, naux)?;
            let mut gemm = Gemm::new(gpu)?;
            let mut ozfit = match oz_fit {
                Some((slices, nb, chunk)) => {
                    Some(OzFit::new(gpu, &gemm, &chol, naux, slices, nb, chunk)?)
                }
                None => None,
            };
            times.metric = t.elapsed().as_secs_f64();

            if let KMath::Emulated { .. } = opts.k_math {
                gemm.prepare(gpu, opts.k_math.cublas())?;
            }

            // the device integrals' builder (the kernels compiled or loaded from the cache)
            let t = Instant::now();
            let mut eri = match eri_host {
                Some(h) => Some(Eri3c::new(gpu, h)?),
                None => None,
            };
            times.prepare += t.elapsed().as_secs_f64();
            let src = match &eri {
                Some(e) => IntSource::Device(e),
                None => IntSource::Cpu(&chunks, opts.thresholds.prim),
            };
            let fit = match &ozfit {
                Some(f) => DfFit::Ozaki { slices: f.slices },
                None => DfFit::Native,
            };
            let fit_shape = ozfit
                .as_ref()
                .map_or((OZ_FIT_BLOCK, OZ_FIT_CHUNK), |f| (f.nb, f.chunk));
            let mut fitter = Fitter {
                chol: &chol,
                oz: ozfit.as_mut(),
            };
            // B: integrals per batch of pairs, fitted on the device; on the device, its bytes
            // granted at the decision (the memory plan's grant), drawn by its allocation
            let grant = match storage {
                DfStorage::Device => Some(gpu.grant(tensor, "DF-J/K: the tensor B")?),
                _ => None,
            };
            let (store, queued) = build_tensor(
                gpu,
                &gemm,
                &src,
                &Operator::Coulomb,
                lay,
                &mut fitter,
                batch_rows,
                storage,
                &blocks,
                nq,
                &mut times.b,
                grant.as_ref(),
            )?;
            let tensor_violations = grant.as_ref().map_or_else(Vec::new, |g| g.violations());
            drop(grant);
            times.integrals = queued;
            times.fit_tail = times.b.wall - queued;
            let kernels = kernels(gpu)?;
            let (device_bytes, host_bytes) = store_bytes(&store);
            let mut eng = GpuDfJk {
                nbf,
                naux,
                nx,
                rows,
                store,
                blocks,
                gemm,
                kernels,
                resolved: GpuDfJkResolved {
                    storage,
                    nbf,
                    naux,
                    nx,
                    nshell_pairs: lay.nshell_pairs,
                    nshell_pairs_kept: lay.pairs.len(),
                    tensor_bytes: tensor,
                    k_block: nq,
                    plan: dfplan,
                    k_math: opts.k_math,
                    device_bytes: device_bytes + 12 * nx,
                    device_budget: budget,
                    host_bytes,
                    lr_omega: None,
                    lr_storage: None,
                    exchange_operator: None,
                    k_planes_bytes: 0,
                    tensor_violations,
                    k_planes_note: None,
                    k_panel_kernel: PanelKernel::Strided,
                    integrals: if eri.is_some() {
                        DfIntegrals::Device
                    } else {
                        DfIntegrals::Cpu
                    },
                    integrals_note,
                    fit,
                    fit_note,
                },
                k_update_block: if opts.k_update_block > 0 {
                    opts.k_update_block
                } else {
                    K_UPDATE_BLOCK
                },
                thresholds: opts.thresholds,
                basis: basis.clone(),
                aux: aux.clone(),
                budget,
                batch_rows,
                lr_storage: opts.lr_storage,
                k_presliced: opts.k_presliced,
                others_reserve: opts.others_reserve,
                k_panel_kernel: opts.k_panel_kernel,
                mma,
                panels,
                ptabs: None,
                pblocks: None,
                k_panel_workspace: k_panel_workspace(opts),
                k_block_set: opts.k_block > 0,
                sizes,
                planned: request.exchange,
                k_columns: request.k_columns,
                planned_work: work,
                fit_shape,
                lr: None,
                xop: None,
                oz: None,
                keep_fit: false,
                coulomb_fit: None,
                coulomb_fit_nd: 0,
                setup_times: times,
                last: GpuDfJkCallTimes::default(),
            };
            eng.resolved.k_panel_kernel = opts.k_panel_kernel;
            if let (Some(w), Some(st)) = (omega, lr_storage) {
                let erf = Operator::LongRange { omega: w };
                let src = match eri.as_mut() {
                    Some(e) => {
                        let t = Instant::now();
                        e.prepare_operator(gpu, &erf)?;
                        eng.setup_times.prepare += t.elapsed().as_secs_f64();
                        IntSource::Device(e)
                    }
                    None => IntSource::Cpu(&chunks, opts.thresholds.prim),
                };
                eng.store_long_range(gpu, w, st, lay, &mut fitter, &src, Instant::now())?;
            }
            // the setup's device buffers go before the Ozaki exchange's are made
            drop(ozfit);
            drop(eri);
            drop(chol);
            if let KMath::Ozaki { slices } = opts.k_math {
                eng.prepare_ozaki(gpu, slices)?;
            }
            eng.setup_times.total = t_all.elapsed().as_secs_f64();
            Ok(eng)
        }

        /// Builds B^omega = (mn|erf(omega r)/r|P) L^-T on the rows of B (fitted in place, in
        /// batches of whole pairs), stored as `storage`, after the setup: the layout, the
        /// metric's factor, the fit (the setup's when its workspace fits a build's, not in use
        /// now) and the integrals' source (the setup's: the device kernels or the CPU) rebuilt.
        fn build_long_range(
            &mut self,
            gpu: &Gpu,
            omega: f64,
            storage: DfStorage,
        ) -> Result<(), GpuError> {
            let t = Instant::now();
            let (naux, nx) = (self.naux, self.nx);
            let DfThresholds { pair, prim, .. } = self.thresholds;
            let erf = Operator::LongRange { omega };
            let aux3 = Aux3c::new(&self.aux);
            let lay = DfLayout::new(&self.basis, &aux3, pair);
            let (chol, _) = metric_factor(gpu, Metric::Formed(&aux3), naux)?;
            let room = per_build_bytes(self.nbf, nx, self.resolved.k_block, true);
            let (nb, chunk) = self.fit_shape;
            let mut ozfit = match self.resolved.fit {
                DfFit::Ozaki { slices } if OzFit::bytes(naux, slices, nb, chunk) <= room => {
                    Some(OzFit::new(gpu, &self.gemm, &chol, naux, slices, nb, chunk)?)
                }
                _ => None,
            };
            let mut fitter = Fitter {
                chol: &chol,
                oz: ozfit.as_mut(),
            };
            let tp = Instant::now();
            let eri = match self.resolved.integrals {
                DfIntegrals::Device => {
                    let h =
                        Eri3cHost::from_pair3c(&self.basis, &self.aux, &lay.pairs, &lay.x0, prim)?;
                    let mut e = Eri3c::new(gpu, h)?;
                    e.prepare_operator(gpu, &erf)?;
                    Some(e)
                }
                _ => None,
            };
            self.setup_times.prepare += tp.elapsed().as_secs_f64();
            let chunks = aux3.chunks(0..self.aux.shells.len());
            let src = match &eri {
                Some(e) => IntSource::Device(e),
                None => IntSource::Cpu(&chunks, prim),
            };
            self.store_long_range(gpu, omega, storage, &lay, &mut fitter, &src, t)
        }

        /// B^omega on the setup's rows `lay` with its fit and integrals' source (ready for
        /// erf(omega r)/r), stored as `storage`; `t`: when its build started.
        #[allow(
            clippy::too_many_arguments,
            reason = "the operator, its storage, the setup's layout, fit and integrals' source"
        )]
        fn store_long_range(
            &mut self,
            gpu: &Gpu,
            omega: f64,
            storage: DfStorage,
            lay: &DfLayout,
            fitter: &mut Fitter<'_>,
            src: &IntSource<'_>,
            t: Instant,
        ) -> Result<(), GpuError> {
            let erf = Operator::LongRange { omega };
            if lay.nx != self.nx {
                return Err(GpuError::Contract(
                    "DF-J/K: the long-range tensor's rows differ from the setup's".into(),
                ));
            }
            let (store, _) = build_tensor(
                gpu,
                &self.gemm,
                src,
                &erf,
                lay,
                fitter,
                self.batch_rows,
                storage,
                &self.blocks,
                self.resolved.k_block,
                &mut self.setup_times.lr,
                None,
            )?;
            // the held bytes: drop a previous B^omega's, add this one's
            if let Some(old) = self.lr.take() {
                let (d, h) = store_bytes(&old.store);
                self.resolved.device_bytes -= d;
                self.resolved.host_bytes -= h;
            }
            let (d, h) = store_bytes(&store);
            self.resolved.device_bytes += d;
            self.resolved.host_bytes += h;
            self.resolved.lr_omega = Some(omega);
            self.resolved.lr_storage = Some(storage);
            self.lr = Some(LongRange { omega, store });
            if self.oz.is_some() {
                self.oz_lr_exponents(gpu)?;
            }
            self.setup_times.long_range += t.elapsed().as_secs_f64();
            Ok(())
        }

        /// The exchange operator of a range-separated hybrid, a/r + b erf(omega r)/r with an erf
        /// part (b != 0, omega > 0; linus's `JkBuilder::set_exchange_operator`): from the next
        /// build on, every K is that operator's exchange, coefficients included, in the wK
        /// scheme of the CPU `DfJk` (see the module docs). B^omega is built for omega unless
        /// held ([`GpuDfJkOptions::omega`] prepares it at setup): on the device when it and
        /// the build's extra workspace fit what the budget leaves (B on the device), else in
        /// pinned host memory, streamed per build. An operator without an erf part is an error
        /// (a global fraction scales K).
        pub fn set_exchange_operator(&mut self, gpu: &Gpu, op: Operator) -> Result<(), GpuError> {
            let omega = lr_omega(&op)?;
            if self.lr.as_ref().is_none_or(|l| l.omega != omega) {
                // a B^omega of another omega is freed first; the panel blocks are planned again
                // (for its Xwt)
                self.pblocks = None;
                if let Some(old) = self.lr.take() {
                    let (d, h) = store_bytes(&old.store);
                    self.resolved.device_bytes -= d;
                    self.resolved.host_bytes -= h;
                    self.resolved.lr_omega = None;
                    self.resolved.lr_storage = None;
                }
                let nq = self.resolved.k_block;
                let tensor = self.resolved.tensor_bytes;
                let work = per_build_bytes(self.nbf, self.nx, nq, true);
                let staging = 8 * nq * self.nx;
                let (engine, device) = (self.held(), gpu.budget());
                // Auto: on the device when B is and B^omega with the builds' workspace fits
                // both the setup's budget and the free memory; else streamed from pinned host
                // memory
                let fits = |b: usize| engine.fits(b + work) && device.fits(b + work);
                let storage = match self.lr_storage {
                    DfStorage::Auto
                        if self.resolved.storage == DfStorage::Device && fits(tensor) =>
                    {
                        DfStorage::Device
                    }
                    DfStorage::Auto => DfStorage::Host,
                    s => s,
                };
                let bytes = if storage == DfStorage::Device {
                    tensor
                } else {
                    staging
                };
                let what = "DF-J/K: B^omega and a build's workspace";
                engine.check(what, bytes + work)?;
                device.check(what, bytes + work)?;
                self.build_long_range(gpu, omega, storage)?;
            }
            self.xop = Some(op);
            self.resolved.exchange_operator = Some(op);
            Ok(())
        }

        /// Back to the exchange of 1/r (B^omega stays held for a later operator with its
        /// omega).
        pub fn clear_exchange_operator(&mut self) {
            self.xop = None;
            self.resolved.exchange_operator = None;
        }

        /// The exchange operator of the builds (None: 1/r).
        pub fn exchange_operator(&self) -> Option<Operator> {
            self.xop
        }

        /// Changes the arithmetic of the exchange GEMMs for the following builds.
        pub fn set_k_math(&mut self, gpu: &Gpu, k_math: KMath) -> Result<(), GpuError> {
            match k_math {
                KMath::Ozaki { slices } => self.prepare_ozaki(gpu, slices)?,
                _ => self.drop_ozaki(),
            }
            self.gemm.prepare(gpu, k_math.cublas())?;
            self.resolved.k_math = k_math;
            Ok(())
        }

        /// The Ozaki exchange's per-build device bytes for `ntot` factor columns, with GEMM1's
        /// columns in chunks of at most `cols`: the block's sliced operand, X's and W's slices,
        /// the INT32 partial sums, C's slices.
        fn oz_build_bytes(&self, slices: usize, ntot: usize, cols: usize) -> usize {
            oz_build_bytes_for(self.nbf, self.resolved.k_block, slices, ntot, cols)
        }

        /// Sets up the Ozaki exchange with `slices` digits (the row map and the stored
        /// tensors' scale exponents, computed once); an `OutOfMemory` error when it and a build's
        /// workspace (GEMM1 on at least 64 columns at a time) do not fit the free device memory.
        /// Timed (waited for) as `setup_times.ozaki`.
        fn prepare_ozaki(&mut self, gpu: &Gpu, slices: usize) -> Result<(), GpuError> {
            let t = Instant::now();
            let r = self
                .prepare_ozaki_untimed(gpu, slices)
                .and_then(|()| gpu.synchronize());
            self.setup_times.ozaki += t.elapsed().as_secs_f64();
            r
        }

        fn prepare_ozaki_untimed(&mut self, gpu: &Gpu, slices: usize) -> Result<(), GpuError> {
            if !(1..=8).contains(&slices) {
                return Err(GpuError::Contract(format!(
                    "DF-J/K: Ozaki exchange with {slices} slices (1 to 8)"
                )));
            }
            let (nbf, naux, nx) = (self.nbf, self.naux, self.nx);
            let held = 4 * (nbf * nbf + naux * nbf * if self.lr.is_some() { 2 } else { 1 });
            let need =
                if self.oz.is_some() { 0 } else { held } + self.oz_build_bytes(slices, 64, 64);
            gpu.budget()
                .check("DF-J/K: the Ozaki exchange's workspace", need)?;
            if let Some(oz) = self.oz.as_mut() {
                if oz.slices != slices {
                    let pb = oz.planes_bytes();
                    oz.planes = None;
                    oz.lr_planes = None;
                    oz.slices = slices;
                    self.resolved.device_bytes -= pb;
                    self.resolved.k_planes_bytes = 0;
                    self.resolved.k_planes_note = None;
                    oz.release();
                    self.oz_presliced_planes(gpu, false)?;
                    self.oz_presliced_planes(gpu, true)?;
                }
                return Ok(());
            }
            let src = OZAKI_MODULE.source();
            let k = |e: &str| gpu.kernel(OZAKI_MODULE.name, &src, e);
            let kern = OzKernels {
                fill: k("df_oz_fill")?,
                map: k("df_oz_map")?,
                exponents: k("df_oz_exponents")?,
                slice: k("df_oz_slice")?,
            };
            let mut map = gpu.alloc_zeros::<i32>(nbf * nbf)?;
            // SAFETY: matches df_oz_fill / df_oz_map; nbf^2 ints of map, nx rows of the tables
            // with m, n < nbf.
            unsafe {
                gpu.launch(
                    &kern.fill,
                    Launch::linear(nbf * nbf, 256),
                    &[
                        Arg::I64((nbf * nbf) as i64),
                        Arg::I32(-1),
                        Arg::buf_mut(&mut map),
                    ],
                )?;
                gpu.launch(
                    &kern.map,
                    Launch::linear(nx, 256),
                    &[
                        Arg::I64(nx as i64),
                        Arg::I64(nbf as i64),
                        Arg::buf(&self.rows.m),
                        Arg::buf(&self.rows.n),
                        Arg::buf(&self.rows.kind),
                        Arg::buf_mut(&mut map),
                    ],
                )?;
            }
            let exps = oz_exponents(
                gpu,
                &kern,
                &map,
                (nbf, naux, nx),
                &mut self.store,
                &self.blocks,
            )?;
            let oz = OzState {
                slices,
                ozaki: Ozaki::new(gpu)?,
                kernels: kern,
                map,
                exps,
                lr_exps: None,
                planes: None,
                lr_planes: None,
                work: OzakiWork::default(),
                a: None,
                x: None,
                w: None,
            };
            self.resolved.device_bytes += oz.held_bytes();
            self.oz = Some(oz);
            self.oz_presliced_planes(gpu, false)?;
            if self.lr.is_some() {
                self.oz_lr_exponents(gpu)?;
            }
            Ok(())
        }

        /// Pre-slices B's GEMM1 planes (`lr`: B^omega's) when [`GpuDfJkOptions::k_presliced`]
        /// is on and they fit: within the setup's budget beside what the engine holds and a
        /// build's workspace (for nbf / 4 factor columns), leaving max(8 GiB, a tenth of the
        /// pool) of the free memory to the other engines.
        fn oz_presliced_planes(&mut self, gpu: &Gpu, lr: bool) -> Result<(), GpuError> {
            let Some(s) = self.oz.as_ref().map(|o| o.slices) else {
                return Ok(());
            };
            if !self.k_presliced || (lr && self.lr.is_none()) {
                return Ok(());
            }
            // B^omega's planes are as large as B's: when B's did not fit, its note stands
            if lr && self.oz.as_ref().is_some_and(|o| o.planes.is_none()) {
                return Ok(());
            }
            let (nbf, naux, nx) = (self.nbf, self.naux, self.nx);
            let rows = naux * nbf + crate::emu::ROW_ALIGN;
            let bytes = Sliced::bytes_for(rows, nbf, s);
            let work = self.oz_build_bytes(s, nbf / 4, nbf / 4);
            let (engine, device) = (self.held(), gpu.budget());
            let free = device.free();
            if !engine.fits(bytes + work) || !device.fits(bytes + work + self.others_reserve) {
                let gib = |b: usize| b as f64 / 1073741824.0;
                self.resolved.k_planes_note = Some(format!(
                    "the pre-sliced planes of {} ({:.1} GiB) do not fit beside the engine's {:.1} GiB, a build's \
                     workspace and the reserve for the other engines ({:.1} GiB free): each build slices the blocks it reads",
                    if lr { "B^omega" } else { "the tensor" },
                    gib(bytes),
                    gib(self.resolved.device_bytes),
                    gib(free)
                ));
                return Ok(());
            }
            let mut planes = Sliced::new(gpu, rows, nbf, s, Order::Ascending)?;
            let missing = || GpuError::Contract("DF-J/K: Ozaki planes without their tensor".into());
            let oz = self.oz.as_mut().ok_or_else(missing)?;
            let (store, exps) = if lr {
                let e = &oz.lr_exps.as_ref().ok_or_else(missing)?.1;
                (&mut self.lr.as_mut().ok_or_else(missing)?.store, e)
            } else {
                (&mut self.store, &oz.exps)
            };
            oz_planes(
                gpu,
                &oz.kernels,
                &oz.map,
                exps,
                (nbf, naux, nx),
                store,
                &self.blocks,
                &mut planes,
            )?;
            self.resolved.device_bytes += planes.bytes();
            self.resolved.k_planes_bytes += planes.bytes();
            if lr {
                oz.lr_planes = Some(planes);
            } else {
                oz.planes = Some(planes);
            }
            Ok(())
        }

        /// The scale exponents of the held B^omega's GEMM1 vectors (Ozaki exchange).
        fn oz_lr_exponents(&mut self, gpu: &Gpu) -> Result<(), GpuError> {
            let (Some(oz), Some(lr)) = (self.oz.as_mut(), self.lr.as_mut()) else {
                return Ok(());
            };
            if let Some((_, old)) = oz.lr_exps.take() {
                self.resolved.device_bytes -= 4 * old.len();
            }
            if let Some(old) = oz.lr_planes.take() {
                self.resolved.device_bytes -= old.bytes();
                self.resolved.k_planes_bytes -= old.bytes();
            }
            let e = oz_exponents(
                gpu,
                &oz.kernels,
                &oz.map,
                (self.nbf, self.naux, self.nx),
                &mut lr.store,
                &self.blocks,
            )?;
            self.resolved.device_bytes += 4 * e.len();
            oz.lr_exps = Some((lr.omega, e));
            self.oz_presliced_planes(gpu, true)
        }

        /// Crate-internal (the response engine): the Ozaki GEMM1 planes of another tensor on
        /// this engine's rows (`w`: naux x nx column-major on the device, given back), its
        /// scales computed here with the engine's row map; the Ozaki exchange must be set up.
        pub(crate) fn oz_planes_for(
            &self,
            gpu: &Gpu,
            w: DeviceBuf<f64>,
        ) -> (DeviceBuf<f64>, Result<Sliced, GpuError>) {
            let mut store = Store::Device(w);
            let r = (|| -> Result<Sliced, GpuError> {
                let oz = self.oz.as_ref().ok_or_else(|| {
                    GpuError::Contract("DF-J/K: the Ozaki exchange is not set up".into())
                })?;
                let dims = (self.nbf, self.naux, self.nx);
                let exps = oz_exponents(gpu, &oz.kernels, &oz.map, dims, &mut store, &self.blocks)?;
                let mut planes = Sliced::new(
                    gpu,
                    self.naux * self.nbf + crate::emu::ROW_ALIGN,
                    self.nbf,
                    oz.slices,
                    Order::Ascending,
                )?;
                oz_planes(
                    gpu,
                    &oz.kernels,
                    &oz.map,
                    &exps,
                    dims,
                    &mut store,
                    &self.blocks,
                    &mut planes,
                )?;
                Ok(planes)
            })();
            let Store::Device(w) = store else {
                unreachable!("a device store stays one")
            };
            (w, r)
        }

        /// Crate-internal (the response engine): the Ozaki exchange's state, taken out until
        /// [`GpuDfJk::put_ozaki`] puts it back.
        pub(crate) fn take_ozaki(&mut self) -> Option<OzTaken> {
            self.oz.take().map(OzTaken)
        }

        /// Crate-internal: puts back what [`GpuDfJk::take_ozaki`] took.
        pub(crate) fn put_ozaki(&mut self, t: OzTaken) {
            self.oz = Some(t.0);
        }

        /// Leaves the Ozaki exchange (frees its state).
        fn drop_ozaki(&mut self) {
            if let Some(oz) = self.oz.take() {
                self.resolved.device_bytes -= oz.held_bytes();
                self.resolved.k_planes_bytes = 0;
                self.resolved.k_planes_note = None;
            }
        }

        /// The engine's own budget: the setup's device budget and the bytes it holds.
        fn held(&self) -> linus_core::Budget {
            linus_core::Budget::new(self.budget, self.resolved.device_bytes)
        }

        pub fn resolved(&self) -> &GpuDfJkResolved {
            &self.resolved
        }

        /// Rows r0..r0 + n of the stored B (`long_range`: of B^omega), naux values each, row
        /// after row (row x's fitted values (x, P) at (x - r0) naux + P), from device or host
        /// storage. For tests and benchmarks.
        pub fn tensor_rows(
            &self,
            gpu: &Gpu,
            long_range: bool,
            r0: usize,
            n: usize,
        ) -> Result<Vec<f64>, GpuError> {
            let (naux, nx) = (self.naux, self.nx);
            if r0 + n > nx {
                return Err(GpuError::Contract("DF-J/K: rows outside the tensor".into()));
            }
            let store = if long_range {
                match &self.lr {
                    Some(l) => &l.store,
                    None => return Err(GpuError::Contract("DF-J/K: no B^omega held".into())),
                }
            } else {
                &self.store
            };
            match store {
                Store::Device(b) => gpu.download_range(b, r0 * naux, n * naux),
                Store::Host(hs) => {
                    let h = hs.host().as_slice()?;
                    let mut out = vec![0.0; n * naux];
                    for &(q0, nqb) in &self.blocks {
                        for x in 0..n {
                            let src = q0 * nx + (r0 + x) * nqb;
                            out[x * naux + q0..x * naux + q0 + nqb]
                                .copy_from_slice(&h[src..src + nqb]);
                        }
                    }
                    Ok(out)
                }
            }
        }

        /// Whether the following J builds keep their Coulomb fit for
        /// [`GpuDfJk::last_coulomb_fit`] (off by default; on, one naux x nd device buffer,
        /// reused, counted in `resolved.device_bytes`). Off drops the kept fit.
        pub fn set_keep_coulomb_fit(&mut self, on: bool) {
            self.keep_fit = on;
            if !on {
                if let Some(b) = self.coulomb_fit.take() {
                    self.resolved.device_bytes -= 8 * b.len();
                }
                self.coulomb_fit_nd = 0;
            }
        }

        /// The Coulomb fit of the last J build when [`GpuDfJk::set_keep_coulomb_fit`] is on:
        /// g = B^T w = L^-1 gamma per density (gamma_P = sum_mn (P|mn) D_mn, V = L L^T the
        /// metric), naux x nd column-major, and nd. The fit coefficients are c = L^-T g. None
        /// before the first J build with the setting on.
        pub fn last_coulomb_fit(&self) -> Option<(&DeviceBuf<f64>, usize)> {
            match (&self.coulomb_fit, self.coulomb_fit_nd) {
                (Some(b), nd) if nd > 0 => Some((b, nd)),
                _ => None,
            }
        }

        /// The buffer for a kept Coulomb fit of nd densities (allocated or grown once).
        fn coulomb_fit_buf(&mut self, gpu: &Gpu, nd: usize) -> Result<(), GpuError> {
            let len = self.naux * nd;
            if self.coulomb_fit.as_ref().is_none_or(|b| b.len() < len) {
                if let Some(b) = self.coulomb_fit.take() {
                    self.resolved.device_bytes -= 8 * b.len();
                }
                self.coulomb_fit = Some(gpu.alloc_zeros::<f64>(len)?);
                self.resolved.device_bytes += 8 * len;
            }
            self.coulomb_fit_nd = 0;
            Ok(())
        }

        pub fn nbf(&self) -> usize {
            self.nbf
        }

        /// The least device memory one build ([`Self::compute_device`]) of `nd` densities and
        /// the factors of `ncols` columns allocates beyond what the engine holds
        /// ([`plan::build_bytes`] at the builds' block): the J vectors and the exchange's
        /// workspace of the arithmetic now (the shell panels' X, gathered orbitals and gathered
        /// panels; the unpacked block; the Ozaki scheme's slices and its workspace at the fewest
        /// columns per GEMM, with the margin the build keeps free); with `host`, also
        /// [`Self::compute`]'s device copies of the inputs and results. A caller compares it with
        /// the free memory before a series of builds (an SCF) instead of meeting an
        /// out-of-memory in one of them. Not counted: cuBLAS's own workspaces.
        pub fn build_bytes(&self, nd: usize, ncols: &[usize], host: bool) -> usize {
            let host_mode = matches!(self.store, Store::Host(_));
            let lr = self.xop.is_some() && self.lr.is_some();
            let ntot: usize = ncols.iter().sum();
            let exchange = self.exchange();
            let shape = match exchange {
                plan::DfExchange::Panels { .. } => self.panel_shape(ntot, self.lr.is_some()),
                _ => (self.resolved.k_block, ntot),
            };
            plan::build_bytes(&self.sizes, exchange, shape, nd, ncols, lr, host_mode, host)
        }

        /// The exchange of the builds' arithmetic now ([`GpuDfJk::set_k_math`]).
        fn exchange(&self) -> plan::DfExchange {
            exchange_of(
                self.resolved.k_math,
                self.k_panel_kernel,
                self.k_panel_workspace,
            )
        }

        /// Crate-internal (the response engine): the screening thresholds of the setup.
        pub(crate) fn thresholds(&self) -> &DfThresholds {
            &self.thresholds
        }

        /// Crate-internal (the response engine): the row tables' unpacker, for tensors with
        /// these rows.
        pub(crate) fn unpacker(&self) -> Unpacker<'_> {
            Unpacker {
                nx: self.nx,
                nbf: self.nbf,
                rows: &self.rows,
                kernel: &self.kernels.unpack,
            }
        }

        /// Crate-internal (the response engine): calls `f` with the unpacker and each auxiliary
        /// block of the stored tensor, in order of the functions. Device storage is one block
        /// of all functions; host storage streams its blocks through the staging buffer (`f`
        /// must have queued all its work on the block before it returns).
        pub(crate) fn for_each_block(
            &mut self,
            gpu: &Gpu,
            mut f: impl FnMut(&Unpacker<'_>, &TensorBlock<'_>) -> Result<(), GpuError>,
        ) -> Result<(), GpuError> {
            let up = Unpacker {
                nx: self.nx,
                nbf: self.nbf,
                rows: &self.rows,
                kernel: &self.kernels.unpack,
            };
            match &mut self.store {
                Store::Device(b) => f(
                    &up,
                    &TensorBlock {
                        q0: 0,
                        nq: self.naux,
                        buf: b,
                        off: 0,
                        ld: self.naux,
                    },
                ),
                Store::Host(hs) => {
                    for &(q0, nq) in &self.blocks {
                        hs.load(gpu, 0, q0 * self.nx, nq * self.nx)?;
                        let staging = hs.slot(gpu, 0)?;
                        f(
                            &up,
                            &TensorBlock {
                                q0,
                                nq,
                                buf: staging,
                                off: 0,
                                ld: nq,
                            },
                        )?;
                    }
                    Ok(())
                }
            }
        }

        /// J of `req.dens` and K of `req.factors` (one K per factor), as host matrices: the
        /// inputs go to the device, [`GpuDfJk::compute_device`] runs, the results come back.
        /// The call's record is the device part's with the uploads and downloads, and its total
        /// the whole call's (transfers included) in place of the device part's.
        pub fn compute(&mut self, gpu: &Gpu, req: &GpuJkRequest) -> Result<GpuJkOutput, GpuError> {
            let nbf = self.nbf;
            let n2 = nbf * nbf;
            let want_j = req.want_j && !req.dens.is_empty();
            let want_k = req.want_k && !req.factors.is_empty();
            for d in req.dens {
                if d.len() != n2 {
                    return Err(GpuError::Contract(format!(
                        "DF-J/K: a density has {} elements, not {nbf}^2",
                        d.len()
                    )));
                }
            }
            for f in req.factors {
                if f.c.len() != nbf * f.ncol {
                    return Err(GpuError::Contract(format!(
                        "DF-J/K: a factor has {} elements, not {nbf} x {}",
                        f.c.len(),
                        f.ncol
                    )));
                }
            }
            // the call's own phases: the uploads, the downloads and the whole call
            let mut tl = Timeline::<GpuDfJkCallTimes>::new(gpu);
            let m0 = tl.mark()?;
            let nd = if want_j { req.dens.len() } else { 0 };
            let ncols: Vec<usize> = if want_k {
                req.factors.iter().map(|f| f.ncol).collect()
            } else {
                vec![]
            };
            let ntot: usize = ncols.iter().sum();
            let dens = if nd > 0 {
                let mut flat = Vec::with_capacity(nd * n2);
                for d in req.dens {
                    flat.extend_from_slice(d);
                }
                Some(gpu.upload(&flat)?)
            } else {
                None
            };
            let cmat = if ntot > 0 {
                let mut flat = Vec::with_capacity(nbf * ntot);
                for f in req.factors {
                    flat.extend_from_slice(f.c);
                }
                Some(gpu.upload(&flat)?)
            } else {
                None
            };
            let nk = ncols.len();
            let mut j = if nd > 0 {
                Some(gpu.alloc_zeros::<f64>(nd * n2)?)
            } else {
                None
            };
            let mut k = if nk > 0 {
                Some(gpu.alloc_zeros::<f64>(nk * n2)?)
            } else {
                None
            };
            tl.close(|t| &mut t.upload, m0)?;
            self.compute_device(
                gpu,
                &GpuJkDeviceRequest {
                    dens: dens.as_ref(),
                    ndens: nd,
                    factors: cmat.as_ref(),
                    ncols: &ncols,
                },
                j.as_mut(),
                k.as_mut(),
            )?;
            let md = tl.mark()?;
            let mut out = GpuJkOutput::default();
            if let Some(j) = &j {
                out.j = gpu.download(j)?.chunks(n2).map(|c| c.to_vec()).collect();
            }
            if let Some(k) = &k {
                out.k = gpu.download(k)?.chunks(n2).map(|c| c.to_vec()).collect();
            }
            let me = tl.close(|t| &mut t.download, md)?;
            tl.span(|t| &mut t.total, m0, me);
            // the device part left the transfers at zero; its total gives way to the call's
            self.last.total = 0.0;
            tl.add_to(&mut self.last)?;
            Ok(out)
        }

        /// J and K with every operand on the device, no host transfer (queued on the device's
        /// stream; a download or [`Gpu::synchronize`] waits for it). `req.dens` holds
        /// `req.ndens` densities (nbf x nbf column-major, one after another) and
        /// `req.factors` the symmetric factors side by side (nbf x sum(ncols), column-major;
        /// K\[d\] of the `ncols[d]` columns of factor d). `j` receives ndens matrices and `k`
        /// ncols.len() (nbf x nbf each, overwritten; K exactly symmetric). J is skipped without
        /// `dens` or `j`, K without `k`; a factor without columns gives K = 0. K is the exchange
        /// of the operator set by [`GpuDfJk::set_exchange_operator`] (1/r without one). The
        /// phases of the module docs run in order on the stream.
        pub fn compute_device(
            &mut self,
            gpu: &Gpu,
            req: &GpuJkDeviceRequest,
            j: Option<&mut DeviceBuf<f64>>,
            k: Option<&mut DeviceBuf<f64>>,
        ) -> Result<(), GpuError> {
            let mut c = self.begin(gpu, req, j.as_deref(), k.as_deref())?;
            let m0 = c.tl.mark()?;
            let (nk, n2) = (c.ncols.len(), c.nbf * c.nbf);
            let mut kmat = match k {
                Some(k) if nk > 0 => {
                    gpu.zero_range(k, 0, nk * n2)?;
                    Some(k)
                }
                _ => None,
            };
            let mut jx = self.coulomb_rows(&c)?;
            if let Some(rows) = jx.as_mut() {
                self.coulomb_device(&mut c, rows)?;
            }
            self.block_pass(&mut c, jx.as_mut(), kmat.as_deref_mut())?;
            self.finish(&c, jx.as_ref(), j, kmat)?;
            c.tl.close(|t| &mut t.total, m0)?;
            c.flush()?;
            self.last = c.times;
            Ok(())
        }

        /// A build's call context: the operands checked against the engine's sizes (J taken
        /// only with `j` and densities, K only with `k`), and the coefficients of a
        /// range-separated exchange.
        fn begin<'a>(
            &self,
            gpu: &'a Gpu,
            req: &GpuJkDeviceRequest<'a>,
            j: Option<&DeviceBuf<f64>>,
            k: Option<&DeviceBuf<f64>>,
        ) -> Result<JkCall<'a>, GpuError> {
            let nbf = self.nbf;
            let n2 = nbf * nbf;
            let dens = req.dens.filter(|_| j.is_some() && req.ndens > 0);
            let nd = if dens.is_some() { req.ndens } else { 0 };
            let nk = if k.is_some() { req.ncols.len() } else { 0 };
            let ncols: &'a [usize] = &req.ncols[..nk];
            let ntot: usize = ncols.iter().sum();
            let cmat = req.factors.filter(|_| ntot > 0);
            let small = |what: &str| GpuError::Contract(format!("DF-J/K: {what} buffer too small"));
            if dens.is_some_and(|d| d.len() < nd * n2) {
                return Err(small("density"));
            }
            if j.is_some_and(|j| nd > 0 && j.len() < nd * n2) {
                return Err(small("J"));
            }
            if ntot > 0 && cmat.is_none_or(|c| c.len() < nbf * ntot) {
                return Err(small("factor"));
            }
            if k.is_some_and(|k| k.len() < nk * n2) {
                return Err(small("K"));
            }
            // range-separated exchange: the coefficients (a, b) of a K + b K_LR
            let lr_coefs = match (self.xop, &self.lr) {
                (Some(op), Some(l)) if l.omega == op.omega() => {
                    Some((op.coulomb(), op.long_range()))
                }
                (Some(_), _) => {
                    return Err(GpuError::Contract(
                        "DF-J/K: the exchange operator's long-range tensor is not held".into(),
                    ))
                }
                (None, _) => None,
            };
            Ok(JkCall {
                gpu,
                nbf,
                naux: self.naux,
                nx: self.nx,
                dens,
                nd,
                cmat,
                ncols,
                ntot,
                lr_coefs,
                tl: Timeline::new(gpu),
                times: GpuDfJkCallTimes::default(),
            })
        }

        /// J's vectors over the rows, when J is asked for: the densities' rows w and J's rows.
        fn coulomb_rows(&self, c: &JkCall) -> Result<Option<JRows>, GpuError> {
            let Some(dens) = c.dens else {
                return Ok(None);
            };
            let (gpu, nbf, nd, nx) = (c.gpu, c.nbf, c.nd, c.nx);
            let mut w = gpu.alloc_zeros::<f64>(nx * nd)?;
            // SAFETY: matches df_density_rows; reads nx rows of the tables and nd nbf^2
            // densities, writes nd nx values.
            unsafe {
                gpu.launch(
                    &self.kernels.density_rows,
                    Launch::linear(nx, 256),
                    &[
                        Arg::I64(nx as i64),
                        Arg::I32(nd as i32),
                        Arg::I64(nbf as i64),
                        Arg::buf(&self.rows.m),
                        Arg::buf(&self.rows.n),
                        Arg::buf(&self.rows.kind),
                        Arg::buf(dens),
                        Arg::buf_mut(&mut w),
                    ],
                )
            }?;
            let jr = gpu.alloc_zeros::<f64>(nx * nd)?;
            Ok(Some(JRows { w, jr }))
        }

        /// Device storage: J from the whole tensor in two passes, g = B^T w (naux x nd; kept
        /// when asked for) and J's rows = B g. The device is waited for here (J's time).
        fn coulomb_device(&mut self, c: &mut JkCall, rows: &mut JRows) -> Result<(), GpuError> {
            let Store::Device(b) = &self.store else {
                return Ok(());
            };
            let (gpu, nd, naux, nx) = (c.gpu, c.nd, c.naux, c.nx);
            let s = c.tl.mark()?;
            let mut g = gpu.alloc_zeros::<f64>(naux * nd)?;
            self.gemm.gemm(
                Arith::Fp64,
                GemmDims::new(Op::N, Op::N, naux, nd, nx),
                (1.0, 0.0),
                MatRef::new(b, 0, naux),
                MatRef::new(&rows.w, 0, nx),
                MatMut::new(&mut g, 0, naux),
            )?;
            self.gemm.gemm(
                Arith::Fp64,
                GemmDims::new(Op::T, Op::N, nx, nd, naux),
                (1.0, 0.0),
                MatRef::new(b, 0, naux),
                MatRef::new(&g, 0, naux),
                MatMut::new(&mut rows.jr, 0, nx),
            )?;
            if self.keep_fit {
                self.coulomb_fit_buf(gpu, nd)?;
                let fit = self
                    .coulomb_fit
                    .as_mut()
                    .ok_or_else(|| GpuError::Contract("DF-J/K: no Coulomb fit buffer".into()))?;
                gpu.copy_range(&g, 0, fit, 0, naux * nd)?;
                self.coulomb_fit_nd = nd;
            }
            c.tl.close(|t| &mut t.j, s)?;
            c.flush()
        }

        /// The pass over the auxiliary blocks, when host storage streams the tensor for J or
        /// K is asked for: per block its rows (streamed from host storage), its part of J (host
        /// storage) and its exchange (in Ozaki slices, [`oz_block`], or on the unpacked block in
        /// cuBLAS's emulation, [`UnpackedK`]; native K takes the shell panels, [`GpuDfJk::panel_pass`]).
        /// The device is waited for at the end (the pass's times).
        fn block_pass(
            &mut self,
            c: &mut JkCall,
            mut jx: Option<&mut JRows>,
            mut kmat: Option<&mut DeviceBuf<f64>>,
        ) -> Result<(), GpuError> {
            let host_mode = matches!(self.store, Store::Host(_));
            if !((host_mode && c.nd > 0) || c.ntot > 0) {
                return Ok(());
            }
            if let (true, Some(k)) = (self.panel_path(c), kmat.as_deref_mut()) {
                return self.panel_pass(c, jx, k);
            }
            let (nbf, nx, nq) = (c.nbf, c.nx, self.resolved.k_block);
            let mut w = self.block_work(c, host_mode)?;
            let arith = self.resolved.k_math.cublas();
            let nk = UnpackedK {
                gemm: &self.gemm,
                up: Unpacker {
                    nx,
                    nbf,
                    rows: &self.rows,
                    kernel: &self.kernels.unpack,
                },
                blas1: &self.kernels.blas1,
                arith,
            };
            for bi in 0..self.blocks.len() {
                let (q0, nqb) = self.blocks[bi];
                let beta = if bi == 0 { 0.0 } else { 1.0 };
                let src = self.store.block(c, q0, nqb)?;
                if let (true, Some(rows), Some(g)) = (host_mode, jx.as_deref_mut(), w.gq.as_mut()) {
                    let fit = self.coulomb_fit.as_mut().filter(|_| w.keep_fit);
                    rows.add_block(nk.gemm, c, &src, g, fit, beta)?;
                }
                if let (Some(oz), Some(cs), Some(x), Some(k)) = (
                    self.oz.as_mut(),
                    w.oz_c.as_ref(),
                    w.x.as_mut(),
                    kmat.as_deref_mut(),
                ) {
                    // the block of B^omega (range-separated exchange)
                    let lr_src = match (c.lr_coefs, self.lr.as_mut()) {
                        (Some(coefs), Some(lr)) => {
                            let wsrc = lr.store.block(c, q0, nqb)?;
                            Some(((wsrc.buf, wsrc.off, wsrc.ld), coefs))
                        }
                        _ => None,
                    };
                    let nkb = nbf
                        .div_ceil(self.k_update_block)
                        .clamp(1, K_UPDATE_BLOCKS_MAX);
                    let blk = OzBlock {
                        nbf,
                        nx,
                        nq,
                        q0,
                        nqb,
                        first: bi == 0,
                        nkb,
                        cols: w.oz_cols,
                    };
                    oz_block(
                        c.gpu,
                        oz,
                        &blk,
                        (src.buf, src.off, src.ld),
                        lr_src,
                        cs.view(),
                        c.ncols,
                        (x, w.xw.as_mut()),
                        k,
                        nk.blas1,
                        &mut c.tl,
                    )?;
                    continue;
                }
                if let (Some(x), Some(cm), Some(k)) = (w.x.as_mut(), c.cmat, kmat.as_deref_mut()) {
                    let s2 = nk.gemm1(c, &src, w.a.as_mut(), cm, x)?;
                    if let (Some(coefs), Some(xw)) = (c.lr_coefs, w.xw.as_mut()) {
                        // the block of B^omega: Xw = Aw_Q^T C, W = (a X + b Xw) / 2,
                        // K_d (+)= X_d W_d^T (full), K = M + M^T at the end
                        let lr = self.lr.as_mut().ok_or_else(|| {
                            GpuError::Contract(
                                "DF-J/K: long-range coefficients without B^omega".into(),
                            )
                        })?;
                        let wsrc = lr.store.block(c, q0, nqb)?;
                        let s2 = nk.gemm1(c, &wsrc, w.a.as_mut(), cm, xw)?;
                        nk.update_lr(c, (nqb, beta), coefs, (x, xw), k)?;
                        c.tl.close(|t| &mut t.k_gemm2, s2)?;
                        continue;
                    }
                    nk.update(c, (nqb, beta), x, k)?;
                    c.tl.close(|t| &mut t.k_gemm2, s2)?;
                }
            }
            drop(w.oz_c.take());
            if let Some(oz) = self.oz.as_mut() {
                oz.release();
            }
            if w.keep_fit {
                self.coulomb_fit_nd = c.nd;
            }
            c.flush()
        }

        /// The block pass's workspace: the blocks' buffers, the kept fit's buffer, and for the
        /// Ozaki exchange C in slices (timed as GEMM1) and GEMM1's columns per chunk within the
        /// free memory.
        fn block_work(&mut self, c: &mut JkCall, host_mode: bool) -> Result<BlockWork, GpuError> {
            let (gpu, nbf, nd, ntot) = (c.gpu, c.nbf, c.nd, c.ntot);
            let n2 = nbf * nbf;
            let nq = self.resolved.k_block;
            let ozaki = self.oz.is_some();
            let a = if ntot > 0 && !ozaki {
                Some(gpu.alloc_zeros::<f64>(nq * n2)?)
            } else {
                None
            };
            let x = if ntot > 0 {
                Some(gpu.alloc_zeros::<f64>(nq * nbf * ntot)?)
            } else {
                None
            };
            let xw = if ntot > 0 && c.lr_coefs.is_some() {
                Some(gpu.alloc_zeros::<f64>(nq * nbf * ntot)?)
            } else {
                None
            };
            let gq = if host_mode && nd > 0 {
                Some(gpu.alloc_zeros::<f64>(nq * nd)?)
            } else {
                None
            };
            let keep_fit = self.keep_fit && gq.is_some();
            if keep_fit {
                self.coulomb_fit_buf(gpu, nd)?;
            }
            // Ozaki exchange: C sliced once for every block, GEMM1's column chunks
            let oz_c = match (self.oz.as_mut(), c.cmat) {
                (Some(oz), Some(cm)) if ntot > 0 => {
                    let mut cs = Sliced::new(gpu, ntot, nbf, oz.slices, Order::Descending)?;
                    let s = c.tl.mark()?;
                    let src = Vectors {
                        buf: cm,
                        off: 0,
                        ld: nbf,
                        contiguous: true,
                    };
                    oz.ozaki.slice(gpu, src, &mut cs, &mut oz.work)?;
                    c.tl.close(|t| &mut t.k_gemm1, s)?;
                    Some(cs)
                }
                _ => None,
            };
            let oz_cols = match &self.oz {
                Some(oz) if ntot > 0 => {
                    let device = gpu.budget();
                    let fixed = self.oz_build_bytes(oz.slices, ntot, 0);
                    let per_col =
                        4 * oz.slices * (nq * nbf).next_multiple_of(crate::emu::ROW_ALIGN);
                    let cols = device.free().saturating_sub(fixed + (256 << 20)) / per_col;
                    if cols < 16.min(ntot) {
                        return Err(device
                            .over(
                                "DF-J/K: the Ozaki exchange's sliced orbitals",
                                fixed + 16 * per_col,
                            )
                            .into());
                    }
                    cols.min(ntot)
                }
                _ => ntot,
            };
            Ok(BlockWork {
                a,
                x,
                xw,
                gq,
                keep_fit,
                oz_c,
                oz_cols,
            })
        }

        /// The results in the caller's buffers: J's rows to the matrices (both triangles); K's
        /// upper triangle from the lower, or (range-separated) K = M + M^T.
        fn finish(
            &self,
            c: &JkCall,
            jx: Option<&JRows>,
            j: Option<&mut DeviceBuf<f64>>,
            kmat: Option<&mut DeviceBuf<f64>>,
        ) -> Result<(), GpuError> {
            let (gpu, nbf, nd, nx) = (c.gpu, c.nbf, c.nd, c.nx);
            let n2 = nbf * nbf;
            if let (Some(rows), Some(j)) = (jx, j) {
                gpu.zero_range(j, 0, nd * n2)?;
                // SAFETY: matches df_rows_to_matrix; reads nd nx row values, writes elements
                // (m, n), (n, m) < nbf^2 of nd matrices (j holds nd nbf^2, checked above).
                unsafe {
                    gpu.launch(
                        &self.kernels.rows_to_matrix,
                        Launch::linear(nx, 256),
                        &[
                            Arg::I64(nx as i64),
                            Arg::I32(nd as i32),
                            Arg::I64(nbf as i64),
                            Arg::buf(&self.rows.m),
                            Arg::buf(&self.rows.n),
                            Arg::buf(&self.rows.kind),
                            Arg::buf(&rows.jr),
                            Arg::buf_mut(j),
                        ],
                    )
                }?;
            }
            if let Some(k) = kmat {
                let cfg = Launch {
                    grid: (n2.div_ceil(256) as u32, c.ncols.len() as u32, 1),
                    block: (256, 1, 1),
                    shared_bytes: 0,
                };
                // range-separated: K = M + M^T; else the upper triangle from the lower
                let kernel = if c.lr_coefs.is_some() {
                    &self.kernels.symmetrize
                } else {
                    &self.kernels.mirror
                };
                // SAFETY: matches df_mirror_lower / df_symmetrize; nk matrices of nbf^2
                // (checked above).
                unsafe { gpu.launch(kernel, cfg, &[Arg::I64(nbf as i64), Arg::buf_mut(k)]) }?;
            }
            Ok(())
        }
    }

    /// One call of [`GpuDfJk::compute_device`]: its operands and the sizes its phases share,
    /// the timeline of its phases and the times they add up to.
    struct JkCall<'a> {
        gpu: &'a Gpu,
        nbf: usize,
        naux: usize,
        /// The layout's rows.
        nx: usize,
        /// The densities of J (nd of them; none: no J).
        dens: Option<&'a DeviceBuf<f64>>,
        nd: usize,
        /// The factors of K side by side (ntot columns; none: no K) and each K's columns.
        cmat: Option<&'a DeviceBuf<f64>>,
        ncols: &'a [usize],
        ntot: usize,
        /// Range-separated exchange: the coefficients (a, b) of a K + b K_LR.
        lr_coefs: Option<(f64, f64)>,
        tl: Timeline<'a, GpuDfJkCallTimes>,
        times: GpuDfJkCallTimes,
    }

    impl JkCall<'_> {
        /// Adds the phases timed since the last flush to the call's times (waits for them).
        fn flush(&mut self) -> Result<(), GpuError> {
            self.tl.flush(&mut self.times)
        }
    }

    /// J's vectors over the rows: the densities' rows w and J's rows (nx x nd each).
    struct JRows {
        w: DeviceBuf<f64>,
        jr: DeviceBuf<f64>,
    }

    impl JRows {
        /// Host storage: one block's part of J (timed as J), g = B_Q^T w (nqb x nd; its rows of
        /// the Coulomb fit copied into `fit` when kept) and J's rows (+)= B_Q g (`beta` 0 at the
        /// first block).
        fn add_block(
            &mut self,
            gemm: &Gemm,
            c: &mut JkCall,
            src: &TensorBlock<'_>,
            g: &mut DeviceBuf<f64>,
            fit: Option<&mut DeviceBuf<f64>>,
            beta: f64,
        ) -> Result<(), GpuError> {
            let (gpu, nd, naux, nx) = (c.gpu, c.nd, c.naux, c.nx);
            let (q0, nqb) = (src.q0, src.nq);
            let s = c.tl.mark()?;
            gemm.gemm(
                Arith::Fp64,
                GemmDims::new(Op::N, Op::N, nqb, nd, nx),
                (1.0, 0.0),
                MatRef::new(src.buf, src.off, src.ld),
                MatRef::new(&self.w, 0, nx),
                MatMut::new(g, 0, nqb),
            )?;
            gemm.gemm(
                Arith::Fp64,
                GemmDims::new(Op::T, Op::N, nx, nd, nqb),
                (1.0, beta),
                MatRef::new(src.buf, src.off, src.ld),
                MatRef::new(g, 0, nqb),
                MatMut::new(&mut self.jr, 0, nx),
            )?;
            if let Some(fit) = fit {
                // the block's rows q0..q0 + nqb of g (naux x nd)
                for d in 0..nd {
                    gpu.copy_range(g, d * nqb, fit, d * naux + q0, nqb)?;
                }
            }
            c.tl.close(|t| &mut t.j, s)?;
            Ok(())
        }
    }

    /// The workspace of a pass over the auxiliary blocks ([`GpuDfJk::block_pass`]).
    struct BlockWork {
        /// Emulated K: A_Q, a block unpacked.
        a: Option<DeviceBuf<f64>>,
        /// K: X_Q ((nq nbf) x ntot) and, range-separated, Xw_Q of B^omega.
        x: Option<DeviceBuf<f64>>,
        xw: Option<DeviceBuf<f64>>,
        /// Host storage's J: the block's g (nq x nd), and whether the build keeps the fit.
        gq: Option<DeviceBuf<f64>>,
        keep_fit: bool,
        /// The Ozaki exchange: C sliced once for every block, GEMM1's columns per chunk.
        oz_c: Option<Sliced>,
        oz_cols: usize,
    }

    /// The exchange on one unpacked block (emulated K, cuBLAS's emulation): the engine's GEMMs,
    /// unpacker and kernels, borrowed beside the stored tensors the block pass streams.
    struct UnpackedK<'e> {
        gemm: &'e Gemm,
        up: Unpacker<'e>,
        blas1: &'e Blas1,
        /// The exchange GEMMs' arithmetic ([`KMath::cublas`]).
        arith: Arith,
    }

    impl UnpackedK<'_> {
        /// X = A_Q^T C ((nqb nbf) x ntot) of the block `src`, unpacked into `a`; timed as the
        /// unpack and GEMM1. The mark after GEMM1.
        fn gemm1(
            &self,
            c: &mut JkCall,
            src: &TensorBlock<'_>,
            a: Option<&mut DeviceBuf<f64>>,
            cm: &DeviceBuf<f64>,
            x: &mut DeviceBuf<f64>,
        ) -> Result<usize, GpuError> {
            let (gpu, nbf, ntot) = (c.gpu, c.nbf, c.ntot);
            let nqb = src.nq;
            let s = c.tl.mark()?;
            let a = a.ok_or_else(|| GpuError::Contract("DF-J/K: no unpack workspace".into()))?;
            self.up.unpack(gpu, src, 0, nqb, a)?;
            let s1 = c.tl.mark()?;
            self.gemm.gemm(
                self.arith,
                GemmDims::new(Op::T, Op::N, nqb * nbf, ntot, nbf),
                (1.0, 0.0),
                MatRef::new(&*a, 0, nbf),
                MatRef::new(cm, 0, nbf),
                MatMut::new(x, 0, nqb * nbf),
            )?;
            let s2 = c.tl.mark()?;
            c.tl.span(|t| &mut t.k_unpack, s, s1);
            c.tl.span(|t| &mut t.k_gemm1, s1, s2);
            Ok(s2)
        }

        /// K_d (+)= X_d X_d^T for each factor d, X_d = nbf x (nqb ncol_d) at column c0_d
        /// (`beta` 0 at the first block): one full GEMM per factor (cuBLAS's emulation slices
        /// both operands per call, so column blocks would repeat that work), mirrored at the end.
        fn update(
            &self,
            c: &JkCall,
            (nqb, beta): (usize, f64),
            x: &DeviceBuf<f64>,
            k: &mut DeviceBuf<f64>,
        ) -> Result<(), GpuError> {
            let nbf = c.nbf;
            let n2 = nbf * nbf;
            let mut c0 = 0;
            for (d, &nc) in c.ncols.iter().enumerate() {
                if nc > 0 {
                    let op = KOperand {
                        x,
                        w: None,
                        off: c0 * nqb * nbf,
                        l: nqb * nc,
                        trans: false,
                    };
                    k_update(self.gemm, self.arith, op, (&mut *k, d * n2), nbf, beta, 1)?;
                }
                c0 += nc;
            }
            Ok(())
        }

        /// Range-separated exchange: W = (a X + b Xw) / 2 in place of Xw, then M_d (+)= X_d
        /// W_d^T for each factor d (nbf x nbf, inner dimension nqb ncol_d; `beta` 0 at the first
        /// block); K = M + M^T at the end.
        fn update_lr(
            &self,
            c: &JkCall,
            (nqb, beta): (usize, f64),
            (fa, fb): (f64, f64),
            (x, xw): (&DeviceBuf<f64>, &mut DeviceBuf<f64>),
            k: &mut DeviceBuf<f64>,
        ) -> Result<(), GpuError> {
            let (gpu, nbf, ntot) = (c.gpu, c.nbf, c.ntot);
            let n2 = nbf * nbf;
            // Xw = fma(a/2, X, b/2 Xw)
            self.blas1.axpby(
                gpu,
                nqb * nbf * ntot,
                &[(0.5 * fb, Operand::Out), (0.5 * fa, Operand::Buf(x))],
                xw,
            )?;
            let mut c0 = 0;
            for (d, &nc) in c.ncols.iter().enumerate() {
                if nc > 0 {
                    let op = KOperand {
                        x,
                        w: Some(&*xw),
                        off: c0 * nqb * nbf,
                        l: nqb * nc,
                        trans: false,
                    };
                    k_update(self.gemm, self.arith, op, (&mut *k, d * n2), nbf, beta, 1)?;
                }
                c0 += nc;
            }
            Ok(())
        }
    }

    /// One factor's operands of a K update: X_d and, range-separated, W_d (in place of Xw_d) at
    /// offset `off` of `x` and `w`, nbf x l column-major (ld nbf: the unpacked block's X) or, with
    /// `trans`, transposed, l x nbf (ld l: the shell panels' auxiliary-major X), l = the
    /// factor's columns times the block's auxiliary functions.
    #[derive(Clone, Copy)]
    struct KOperand<'a> {
        x: &'a DeviceBuf<f64>,
        w: Option<&'a DeviceBuf<f64>>,
        off: usize,
        l: usize,
        trans: bool,
    }

    /// The K update of one factor in `nkb` column blocks of K (starts at multiples of
    /// [`K_UPDATE_ALIGN`] functions: the operands and K's block then start 16-byte aligned; an
    /// odd start made cuBLAS take a slow path, u1an39 on an H200 61.8 ms against 15 ms): K_d
    /// (+)= X_d X_d^T over the lower triangle (mirrored at the end), or with W_d, M_d (+)= X_d
    /// W_d^T in full (K = M + M^T at the end). K_d is the nbf x nbf matrix at `k_off` of `k`;
    /// `beta` 0 overwrites it.
    fn k_update(
        gemm: &Gemm,
        arith: Arith,
        op: KOperand<'_>,
        (k, k_off): (&mut DeviceBuf<f64>, usize),
        nbf: usize,
        beta: f64,
        nkb: usize,
    ) -> Result<(), GpuError> {
        let KOperand {
            x,
            w,
            off,
            l,
            trans,
        } = op;
        let edge = |b: usize| {
            if b >= nkb {
                nbf
            } else {
                b * nbf / nkb / K_UPDATE_ALIGN * K_UPDATE_ALIGN
            }
        };
        // the rows (or, transposed, the columns) of X from function j
        let at = |j: usize| if trans { off + l * j } else { off + j };
        let ld = if trans { l } else { nbf };
        let (oa, ob) = if trans {
            (Op::T, Op::N)
        } else {
            (Op::N, Op::T)
        };
        for b in 0..nkb {
            let (j0, j1) = (edge(b), edge(b + 1));
            if j1 <= j0 {
                continue;
            }
            match w {
                // K[j0.., j0..j1] (+)= X[j0.., :] X[j0..j1, :]^T
                None => gemm.gemm(
                    arith,
                    GemmDims::new(oa, ob, nbf - j0, j1 - j0, l),
                    (1.0, beta),
                    MatRef::new(x, at(j0), ld),
                    MatRef::new(x, at(j0), ld),
                    MatMut::new(&mut *k, k_off + j0 * nbf + j0, nbf),
                )?,
                // M[:, j0..j1] (+)= X W[j0..j1, :]^T
                Some(w) => gemm.gemm(
                    arith,
                    GemmDims::new(oa, ob, nbf, j1 - j0, l),
                    (1.0, beta),
                    MatRef::new(x, off, ld),
                    MatRef::new(w, at(j0), ld),
                    MatMut::new(&mut *k, k_off + j0 * nbf, nbf),
                )?,
            }
        }
        Ok(())
    }

    impl Store {
        /// Block q0..q0 + nqb of the tensor for a build: in place (device storage), or loaded
        /// into the staging buffer (host storage; timed as the stream).
        fn block(
            &mut self,
            c: &mut JkCall,
            q0: usize,
            nqb: usize,
        ) -> Result<TensorBlock<'_>, GpuError> {
            Ok(match self {
                Store::Device(b) => TensorBlock {
                    q0,
                    nq: nqb,
                    buf: b,
                    off: q0,
                    ld: c.naux,
                },
                Store::Host(hs) => {
                    let s = c.tl.mark()?;
                    hs.load(c.gpu, 0, q0 * c.nx, nqb * c.nx)?;
                    let staging = hs.slot(c.gpu, 0)?;
                    c.tl.close(|t| &mut t.stream, s)?;
                    TensorBlock {
                        q0,
                        nq: nqb,
                        buf: staging,
                        off: 0,
                        ld: nqb,
                    }
                }
            })
        }
    }

    /// A block of B^omega's stored rows (buffer, offset, leading dimension) and the
    /// coefficients (a, b) of a K + b K_LR.
    type LrBlock<'a> = ((&'a DeviceBuf<f64>, usize, usize), (f64, f64));

    /// Sizes of one auxiliary block of an Ozaki exchange build.
    struct OzBlock {
        nbf: usize,
        nx: usize,
        /// The build's block size (the workspace's) and this block's functions q0..q0 + nqb.
        nq: usize,
        q0: usize,
        nqb: usize,
        /// The first block (K overwritten, not accumulated).
        first: bool,
        /// Column blocks of the K update's lower triangle.
        nkb: usize,
        /// GEMM1's columns per chunk.
        cols: usize,
    }

    /// Columns j = (i, q) of X per sliced chunk of the K update (half the INT32 bound, so a
    /// diagonal's products are summed in at most a few groups).
    pub(super) const OZ_J_CHUNK: usize = INT32_TERMS / 2;

    /// GEMM1 of one block in Ozaki slices: out = A^T C in column chunks, the block's
    /// operand `av` (pre-sliced planes, or gathered for this build).
    #[allow(
        clippy::too_many_arguments,
        reason = "the slicer and its workspace, the block, its operands and the output"
    )]
    fn oz_gemm1(
        gpu: &Gpu,
        ozaki: &Ozaki,
        av: SlicedRef<'_>,
        work: &mut OzakiWork,
        b: &OzBlock,
        cs: SlicedRef<'_>,
        out: &mut DeviceBuf<f64>,
        tl: &mut Timeline<'_, GpuDfJkCallTimes>,
    ) -> Result<(), GpuError> {
        let rows = b.nqb * b.nbf;
        let ntot = cs.rows;
        let t1 = tl.mark()?;
        let mut c0 = 0;
        while c0 < ntot {
            let nc = b.cols.min(ntot - c0);
            let cv = SlicedRef {
                data_off: cs.data_off + c0 * cs.slices * cs.kp,
                exp_off: cs.exp_off + c0,
                rows: nc,
                rows_pad: nc.div_ceil(crate::emu::ROW_ALIGN) * crate::emu::ROW_ALIGN,
                ..cs
            };
            ozaki.gemm(gpu, av, cv, 1.0, 0.0, (&mut *out, c0 * rows, rows), work)?;
            c0 += nc;
        }
        tl.close(|t| &mut t.k_gemm1, t1)?;
        Ok(())
    }

    /// The block's GEMM1 operand: a view of the pre-sliced planes, or gathered from the
    /// stored rows `src` into `a` (timed as the unpack).
    #[allow(
        clippy::too_many_arguments,
        reason = "the gather's kernels and tables, the planes or their buffer, the block's source"
    )]
    fn oz_operand<'s>(
        gpu: &Gpu,
        (kernels, map, exps): (&OzKernels, &DeviceBuf<i32>, &DeviceBuf<i32>),
        planes: Option<&'s Sliced>,
        a: &'s mut Option<Sliced>,
        b: &OzBlock,
        src: (&DeviceBuf<f64>, usize, usize),
        tl: &mut Timeline<'_, GpuDfJkCallTimes>,
    ) -> Result<SlicedRef<'s>, GpuError> {
        let rows = b.nqb * b.nbf;
        if let Some(p) = planes {
            return p.rows_view(b.q0 * b.nbf, rows);
        }
        let a = a
            .as_mut()
            .ok_or_else(|| GpuError::Contract("DF Ozaki: no block workspace".into()))?;
        let t0 = tl.mark()?;
        oz_gather(
            gpu,
            kernels,
            map,
            exps,
            (b.nbf, b.nx),
            (b.q0, b.nqb),
            src,
            a,
            0,
        )?;
        tl.close(|t| &mut t.k_unpack, t0)?;
        let a: &'s Sliced = a;
        a.rows_view(0, rows)
    }

    /// The exchange of one auxiliary block in Ozaki slices (module docs): X = A^T C (and Xw =
    /// Aw^T C from B^omega's block, W = (a X + b Xw) / 2) by [`oz_gemm1`], then per factor
    /// K_d (+)= X_d X_d^T (lower triangle) or X_d W_d^T, the inner dimension j = (i, q) in
    /// chunks the INT32 sums hold. Timed as the unpack (the gather), GEMM1 and the K update.
    #[allow(
        clippy::too_many_arguments,
        reason = "the block's operands, the long-range block, the factors and K"
    )]
    fn oz_block(
        gpu: &Gpu,
        oz: &mut OzState,
        b: &OzBlock,
        src: (&DeviceBuf<f64>, usize, usize),
        lr: Option<LrBlock<'_>>,
        cs: SlicedRef<'_>,
        ncols: &[usize],
        (x, mut xw): (&mut DeviceBuf<f64>, Option<&mut DeviceBuf<f64>>),
        k: &mut DeviceBuf<f64>,
        blas1: &Blas1,
        tl: &mut Timeline<'_, GpuDfJkCallTimes>,
    ) -> Result<(), GpuError> {
        let OzState {
            slices,
            ozaki,
            kernels,
            map,
            exps,
            lr_exps,
            planes,
            lr_planes,
            work,
            a,
            x: xs,
            w: ws,
        } = oz;
        let (nbf, s) = (b.nbf, *slices);
        let n2 = nbf * nbf;
        let rows = b.nqb * nbf;
        let ntot = cs.rows;
        // a gathered block's workspace only when a tensor's planes are not held
        if planes.is_none() || (lr.is_some() && lr_planes.is_none()) {
            sliced_slot(gpu, a, b.nq * nbf, nbf, s, Order::Ascending)?;
        }
        {
            let av = oz_operand(
                gpu,
                (&*kernels, &*map, &*exps),
                planes.as_ref(),
                a,
                b,
                src,
                &mut *tl,
            )?;
            oz_gemm1(gpu, ozaki, av, work, b, cs, x, &mut *tl)?;
        }
        if let (Some((lsrc, (fa, fb))), Some(xw)) = (lr, xw.as_deref_mut()) {
            let (_, le) = lr_exps.as_ref().ok_or_else(|| {
                GpuError::Contract("DF Ozaki: B^omega's exponents are not held".into())
            })?;
            {
                let av = oz_operand(
                    gpu,
                    (&*kernels, &*map, le),
                    lr_planes.as_ref(),
                    a,
                    b,
                    lsrc,
                    &mut *tl,
                )?;
                oz_gemm1(gpu, ozaki, av, work, b, cs, xw, &mut *tl)?;
            }
            // W = (a X + b Xw) / 2 in place of Xw: fma(a/2, X, b/2 Xw)
            let len = rows * ntot;
            blas1.axpby(
                gpu,
                len,
                &[(0.5 * fb, Operand::Out), (0.5 * fa, Operand::Buf(x))],
                xw,
            )?;
        }
        let t0 = tl.mark()?;
        let mut c0 = 0;
        for (d, &nc) in ncols.iter().enumerate() {
            if nc == 0 {
                continue;
            }
            let xo = c0 * rows;
            let jj = b.nqb * nc;
            let mut j0 = 0;
            while j0 < jj {
                let jl = OZ_J_CHUNK.min(jj - j0);
                let beta = if b.first && j0 == 0 { 0.0 } else { 1.0 };
                let sx = sliced_slot(gpu, xs, nbf, jl, s, Order::Ascending)?;
                let xsrc = Vectors {
                    buf: x,
                    off: xo + j0 * nbf,
                    ld: nbf,
                    contiguous: false,
                };
                ozaki.slice(gpu, xsrc, sx, work)?;
                match xw.as_deref() {
                    None => {
                        // X again in descending digit order: each diagonal one GEMM per
                        // column block
                        let sd = sliced_slot(gpu, ws, nbf, jl, s, Order::Descending)?;
                        ozaki.slice(gpu, xsrc, sd, work)?;
                        ozaki.syrk_lower(
                            gpu,
                            sx.view(),
                            sd.view(),
                            1.0,
                            beta,
                            (&mut *k, d * n2, nbf),
                            b.nkb,
                            work,
                        )?
                    }
                    Some(w) => {
                        let sw = sliced_slot(gpu, ws, nbf, jl, s, Order::Descending)?;
                        let wsrc = Vectors {
                            buf: w,
                            off: xo + j0 * nbf,
                            ld: nbf,
                            contiguous: false,
                        };
                        ozaki.slice(gpu, wsrc, sw, work)?;
                        ozaki.gemm(
                            gpu,
                            sx.view(),
                            sw.view(),
                            1.0,
                            beta,
                            (&mut *k, d * n2, nbf),
                            work,
                        )?;
                    }
                }
                j0 += jl;
            }
            c0 += nc;
        }
        tl.close(|t| &mut t.k_gemm2, t0)?;
        Ok(())
    }
}

#[cfg(test)]
mod layout_tests {
    use super::*;
    use crate::harness;
    use linus_ints::DerivOrder;

    fn bits(v: &[f64]) -> Vec<u64> {
        v.iter().map(|x| x.to_bits()).collect()
    }

    /// The two layouts, field by field and bit for bit.
    fn assert_same(a: &DfLayout, b: &DfLayout, what: &str) {
        assert_eq!((a.nbf, a.naux, a.nx), (b.nbf, b.naux, b.nx), "{what}");
        assert_eq!(a.nshell_pairs, b.nshell_pairs, "{what}");
        assert_eq!(a.x0, b.x0, "{what}");
        assert_eq!(
            (&a.row_m, &a.row_n, &a.row_kind),
            (&b.row_m, &b.row_n, &b.row_kind),
            "{what}"
        );
        assert_eq!(a.pairs.len(), b.pairs.len(), "{what}");
        let flat = |v: &[[f64; 3]]| bits(&v.iter().flatten().copied().collect::<Vec<_>>());
        for (p, q) in a.pairs.iter().zip(&b.pairs) {
            assert_eq!(
                (p.sa, p.sb, p.na, p.nb, p.lab),
                (q.sa, q.sb, q.na, q.nb, q.lab),
                "{what}"
            );
            assert_eq!(bits(&p.p), bits(&q.p), "{what}");
            assert_eq!(bits(&p.et), bits(&q.et), "{what}");
            assert_eq!(bits(&p.qprim), bits(&q.qprim), "{what}");
            assert_eq!(flat(&p.pc), flat(&q.pc), "{what}");
        }
    }

    /// A caller's shell pairs (the integral context's: index-ordered at the default primitive
    /// cut, of derivative order zero or higher, their Schwarz factors computed before or not)
    /// give the layout DfLayout::new builds from pairs of its own, bit for bit: the same kept
    /// pairs at DfShared's Schwarz cut, in the same order, the same stored rows and pair records,
    /// so the panel plan and the packed integrals follow unchanged. def2-TZVP and def2-QZVP; the
    /// range-separated case (wB97X's long-range tensor W) is stored on the same layout (the
    /// layout has no operator input), here at the response engine's cut as well as the SCF's.
    #[test]
    fn a_callers_pairs_give_the_same_layout() {
        let cuts = [DfThresholds::default().pair, 1e-14];
        for (name, basis_name) in [("u1an39", "def2-tzvp"), ("la18", "def2-qzvp")] {
            let mol = harness::structure(name).unwrap();
            let basis = harness::basis(&mol, basis_name).unwrap();
            let aux = harness::basis(&mol, "def2-universal-jkfit").unwrap();
            let aux3 = Aux3c::new(&aux);
            let zero = PairSet::build(&basis, PairOptions::with_order(DerivOrder::Zero));
            let _ = zero.schwarz();
            let one = PairSet::build(&basis, PairOptions::with_order(DerivOrder::One));
            for &cut in &cuts {
                let own = DfLayout::new(&basis, &aux3, cut);
                for (set, which) in [(&zero, "order 0, Schwarz cached"), (&one, "order 1")] {
                    let given = DfLayout::with_pairs(&basis, Some(set), &aux3, cut);
                    assert_same(
                        &own,
                        &given,
                        &format!("{name}/{basis_name} {which} cut {cut:e}"),
                    );
                }
            }
        }
    }

    /// The layout's sizes the storage plan reads ([`DfLayout::rows`], linus-ints' `DfRows`, the
    /// rule the memory needs apply ahead of a layout) are the layout's and its shell panels':
    /// the stored rows, the panel rows and gathered-orbital columns of [`PanelPlan`], from the
    /// screening's kept pairs (`df_kept_pairs`) as from the layout's.
    #[test]
    fn the_planned_sizes_are_the_layouts() {
        let mol = harness::structure("u1an39").unwrap();
        let basis = harness::basis(&mol, "def2-tzvp").unwrap();
        let aux = harness::basis(&mol, "def2-universal-jkfit").unwrap();
        let aux3 = Aux3c::new(&aux);
        let cut = DfThresholds::default().pair;
        let lay = DfLayout::new(&basis, &aux3, cut);
        let rows = lay.rows(&basis);
        let panels = PanelPlan::new(&basis, &lay);
        assert_eq!(
            (rows.nx, rows.np, rows.ncg),
            (lay.nx, panels.np(), panels.ncg())
        );
        let max_panel = panels.shells.iter().map(|s| s.d * s.w).max().unwrap_or(0);
        assert_eq!(rows.max_panel_rows, max_panel);
        assert_eq!(
            rows.max_pair_rows,
            lay.pairs.iter().map(|p| p.nab()).max().unwrap_or(0)
        );
        let pairs = PairSet::build(&basis, PairOptions::default());
        let kept = linus_ints::eri3c::df_kept_pairs(&pairs, aux3.qmax, cut);
        assert_eq!(linus_ints::eri3c::DfRows::new(&basis, &pairs, &kept), rows);
    }
}
