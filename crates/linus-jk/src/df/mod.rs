//! Density-fitted Coulomb and exchange (RI-J / RI-K) on the fast Hermite 3-center engine
//! (`linus_ints::eri3c`).
//!
//! Storage: the 3-index tensor is kept "pair-major": one row of naux values for every basis
//! function pair (m, n) of every Schwarz-significant shell pair (A >= B; diagonal shell blocks
//! hold both triangles), so only significant pairs are stored and each pair once. With
//! exchange the rows hold B = (mn|P) L^{-T} (V = L L^T, applied once at setup by a blocked
//! triangular solve); for J only they hold the raw (mn|P) and the fit is solved per call.
//! When the tensor does not fit the memory budget, J is built integral-direct in Hermite space
//! (split-RI-J: the density is contracted into each primitive pair's Hermite basis, gamma and
//! J go through the R tensors only; two integral passes per call, no 3-index storage).
//!
//! Exchange beyond the in-memory limit (chosen by [`DfJk::try_new`] from the budget, most
//! accurate first; [`DfConfig::precision`] and [`DfConfig::long_range`] restrict the choice;
//! [`DfPrecision::Auto`] chooses among the f64 storages only, so a memory budget never selects a
//! compressed tensor: B, then the one-sided coefficients, else a [`DfSetupError::Memory`]):
//! - range-separated hybrids, one-sided ("coef"): only the fitted coefficients
//!   C = (mn|P) V^{-1} are stored (one tensor instead of B and B^omega); the exchange pairs them
//!   with raw 3-center integrals of the other side, computed per auxiliary block in every build
//!   (the 3-center integrals cost a fraction of a K build; the fit, which costs ~8x the
//!   integrals, stays a one-time setup cost):
//!   a K + b K_LR = sym[(ml|P) V^{-1} (P| a/r + b erf(omega r)/r |ns)] D_ls,
//!   the same wK expression as with B and B^omega, equal up to round-off; one integral pass of
//!   the combined operator per build. J: c = C^T D (= V^{-1} gamma), J = C (V c).
//! - compressed storage of B or C, only on request ([`DfConfig::precision`]): an f32 value plus
//!   an integer residual in units of its ulp, F48 (i16, 6 bytes, relative error <= 9.1e-13: SCF
//!   energies and gradients equal the f64 ones to ~1e-11), F40 (i8, 5 bytes, <= 2.3e-10: ~1e-9
//!   Eh), F32 (4 bytes, 1e-7..1e-6 Eh).
//!   Rows are decoded to f64 when gathered; J comes from the integral-direct path unless the
//!   tensor is B in F48 (the Coulomb energy would show the rounding).
//!
//! Storing nothing (fitting the occupied-transformed integrals in every build) would cost an
//! naux^2 nbf nocc triangular solve per build, more than the K build itself; the fit must be
//! paid once, on stored data.
//!
//! J: gamma = B^T D (one streaming pass), J = B gamma (second pass); fixed-size row chunks
//! and ordered reductions, so J does not depend on the thread count.
//!
//! K from density factors (D = C C^T, D = X Y^T + Y X^T for response densities, or
//! non-symmetric D = X Y^T): the auxiliary index is cut into blocks; for each block
//! Z[(m,q), i] = sum_n B^q_mn C_ni is built per shell from a gathered sparse panel (one GEMM per
//! shell over its significant partner functions only), then K += Z Z^T (lower triangle). Blocks
//! are assigned to a fixed number of groups whose partial K matrices are summed in a fixed
//! order, so K does not depend on the thread count either. Each group runs its GEMMs
//! sequentially (faer reaches ~95% of single-core AVX-512 peak on these shapes; its parallel
//! GEMM is far less efficient for the tall-skinny K update). The beyond-memory modes process
//! one auxiliary block at a time with all threads (integrals over pairs, Z over shell panels,
//! the K update in a fixed number of lanes with partial K matrices), so their workspace is one
//! block, not one per group.
//!
//! Range-separated exchange (request `omega`): B^omega = (mn|erf(omega r)/r|P) L^{-T} with the
//! Coulomb metric is built once per omega (Psi4's wK), and K_LR = (Z Zw^T + Zw Z^T)/2 shares Z
//! with K in the same pass.

mod common;
pub mod gradient;
pub mod hessian;
#[cfg(any(test, feature = "reference"))]
pub mod reference;
pub mod response;
pub(crate) mod rij_gradient;

use faer::linalg::matmul::triangular::{self as tri, BlockStructure};
use faer::{Accum, Mat, MatMut, MatRef, Par};
use linus_base::linalg::eigh;
use linus_base::linalg::matmul;
use linus_base::terms::{CoulombHook, HookOutcome};
use linus_base::terms::{JkBuilder, JkOutput, JkRequest};
use linus_ints::eri3c::{self, Aux3c, AuxChunk, DfThresholds, Pair3c, Work3c};
use linus_ints::operator::Operator;
use linus_ints::{BasisSet, PairSet};
use linus_num::det::ordered_chunks;
use rayon::prelude::*;
use std::sync::Arc;
use std::time::Instant;

/// Storage format of the 3-index exchange tensor. The compressed formats keep an f32 value
/// plus an integer residual in units of its ulp (decoded to f64 when rows are gathered):
/// F48 (f32 + i16, 6 bytes, relative error <= 9.1e-13), F40 (f32 + i8, 5 bytes, <= 2.3e-10),
/// F32 (4 bytes, <= 6e-8).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DfPrecision {
    /// F64 (B, else the one-sided coefficients of a range-separated hybrid, within the budget):
    /// the compressed formats only on request, so a memory budget or grant never changes the
    /// arithmetic beyond f64 round-off
    #[default]
    Auto,
    F64,
    F48,
    F40,
    F32,
}

/// Where the long-range (erf) side of range-separated exchange comes from.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DfLongRange {
    /// B and B^omega stored when both fit the budget, else one-sided
    #[default]
    Auto,
    /// B and B^omega stored
    Stored,
    /// only C = (mn|P) V^{-1} stored, raw integrals of the other side computed per build
    #[serde(alias = "one-sided", alias = "onesided")]
    OneSided,
}

#[derive(Clone, Debug)]
pub struct DfConfig {
    /// Bytes allowed for the 3-index tensors. The session always sets it from its memory
    /// limit, less its reservations (the SCF's memory budget, `ScfMemory::df_bytes`); None
    /// (direct users of [`DfJk`]) takes the budget of an unset `memory_gb` with nothing reserved.
    pub mem_budget: Option<usize>,
    /// The screening thresholds: shell pairs with (mn|mn)^1/2 max_P (P|P)^1/2 below `pair` are
    /// dropped; primitive pairs / auxiliary primitive batches with a Schwarz bound below
    /// `prim` are skipped (`deriv_prim` is the gradients' and Hessians').
    pub thresholds: DfThresholds,
    /// Auxiliary functions per exchange block (0: automatic).
    pub k_block: usize,
    /// Number of exchange partial sums (0: automatic, from the problem size only).
    pub k_groups: usize,
    /// Never store the 3-index tensor (J integral-direct; exchange unavailable).
    pub force_direct: bool,
    /// Range-separation parameter to prepare at setup: B^omega when B and B^omega fit the budget
    /// (twice the memory of B), else one-sided storage (any omega, raw integrals per build).
    /// With B stored, other omegas requested later are built on demand.
    pub omega: Option<f64>,
    /// Storage format of the exchange tensor (Auto: f64; the compressed formats only on request).
    pub precision: DfPrecision,
    /// Long-range exchange storage for range-separated hybrids (Auto: one-sided when B and
    /// B^omega do not both fit).
    pub long_range: DfLongRange,
    /// J from an F40 or F32 tensor (default false: J integral-direct, exact, for those
    /// formats; F64 and F48 tensors always give J).
    pub j_from_f32: bool,
    /// The whole process's room for the tensors and the SCF's exchange workspace together
    /// (bytes): the memory limit less everything else the SCF holds (the process, the
    /// reservations, the metric, the matrices and buffers of the measured model). A storage is
    /// chosen only if its tensors fit [`DfConfig::mem_budget`] and, with the exchange workspace
    /// of its builds ([`DfConfig::scf_occ`]), this room. None: no such check (an unset
    /// `memory_gib`, direct users).
    pub room: Option<usize>,
    /// The occupied orbitals per spin of the SCF's exchange builds, sizing each storage's
    /// exchange workspace for [`DfConfig::room`]: both spins, the probes' two-spin builds
    /// included (the largest shape the SCF builds).
    pub scf_occ: [usize; 2],
}

impl Default for DfConfig {
    fn default() -> Self {
        DfConfig {
            mem_budget: None,
            thresholds: DfThresholds::default(),
            k_block: 0,
            k_groups: 0,
            force_direct: false,
            omega: None,
            precision: DfPrecision::Auto,
            long_range: DfLongRange::Auto,
            j_from_f32: false,
            room: None,
            scf_occ: [0, 0],
        }
    }
}

/// Factored density for exchange.
#[derive(Clone, Copy)]
pub enum Factored<'a> {
    /// D = C C^T
    Psd(MatRef<'a, f64>),
    /// D = X Y^T + Y X^T (symmetric, possibly indefinite: response densities)
    SymPair(MatRef<'a, f64>, MatRef<'a, f64>),
    /// D = X Y^T (non-symmetric)
    Pair(MatRef<'a, f64>, MatRef<'a, f64>),
}

#[derive(Clone, Debug)]
pub struct DfStats {
    pub nbf: usize,
    pub naux: usize,
    pub nshell_pairs: usize,
    pub nshell_pairs_kept: usize,
    pub panel_elems: usize,
    pub shell_panel_elems: usize,
    pub distinct_lists: usize,
    /// stored function pairs (rows)
    pub nrows: usize,
    pub bytes: usize,
    /// memory budget used for the 3-index tensors (bytes)
    pub budget: usize,
    /// the storage chosen at setup
    pub plan: DfPlan,
    pub t_schwarz: f64,
    pub t_prep: f64,
    pub t_metric: f64,
    pub t_metric_ints: f64,
    pub t_alloc: f64,
    pub t_ints: f64,
    pub t_fit: f64,
    pub t_lr_setup: f64,
    pub bytes_lr: usize,
    /// last call: (name, seconds)
    pub last: Vec<(String, f64)>,
}

/// Exchange panel: a run of shells s0..s1 (functions f0..f0+nf; currently one shell per panel)
/// and the union of their kept partner shells; entry[(a - s0) * npartner + j] = (pair index,
/// stored transposed) or None (zero-filled).
struct Panel {
    s0: usize,
    s1: usize,
    f0: usize,
    nf: usize,
    partners: Vec<usize>,
    entry: Vec<Option<(u32, bool)>>,
    /// partner basis functions in partner order
    flist: Vec<u32>,
}

/// Storage format of a stored 3-index tensor: f64, an f32 value plus an integer residual in
/// units of its ulp (F48: i16, F40: i8), or f32 alone ([`DfPrecision`] restricts the choice).
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "lowercase")]
pub enum DfFormat {
    F64,
    F48,
    F40,
    F32,
}

impl DfFormat {
    /// Bytes per element.
    pub fn bytes(self) -> usize {
        match self {
            DfFormat::F64 => 8,
            DfFormat::F48 => 6,
            DfFormat::F40 => 5,
            DfFormat::F32 => 4,
        }
    }
    fn suffix(self) -> &'static str {
        match self {
            DfFormat::F64 => "",
            DfFormat::F48 => "48",
            DfFormat::F40 => "40",
            DfFormat::F32 => "32",
        }
    }
}

/// ulp of a normal f32 as f64 (0 for zero and subnormals).
#[inline(always)]
fn ulp32(hi: f32) -> f64 {
    f32::from_bits(hi.to_bits() & 0x7f80_0000) as f64 * (1.0 / 8388608.0)
}

/// Integer residual of the compressed formats: x = hi + q ulp(hi) / (2 QMAX), |q| <= QMAX.
trait Resid: Copy + Send + Sync + Default + 'static {
    const QMAX: f64;
    fn val(self) -> f64;
    fn from_f64(q: f64) -> Self;
}
impl Resid for i16 {
    const QMAX: f64 = 32767.0;
    #[inline(always)]
    fn val(self) -> f64 {
        self as f64
    }
    #[inline(always)]
    fn from_f64(q: f64) -> Self {
        q as i16
    }
}
impl Resid for i8 {
    const QMAX: f64 = 127.0;
    #[inline(always)]
    fn val(self) -> f64 {
        self as f64
    }
    #[inline(always)]
    fn from_f64(q: f64) -> Self {
        q as i8
    }
}

/// f32 rounding of x plus its residual (|x - hi| <= ulp(hi) / 2).
#[inline(always)]
fn encode<Q: Resid>(x: f64) -> (f32, Q) {
    let hi = x as f32;
    let u = ulp32(hi);
    if u == 0.0 {
        return (hi, Q::default());
    }
    let q = ((x - hi as f64) / u * (2.0 * Q::QMAX))
        .round()
        .clamp(-Q::QMAX, Q::QMAX);
    (hi, Q::from_f64(q))
}

#[inline(always)]
fn decode<Q: Resid>(hi: f32, q: Q) -> f64 {
    hi as f64 + q.val() * (ulp32(hi) * (0.5 / Q::QMAX))
}

/// Reads consecutive elements of a stored tensor as f64.
trait RowRead: Sync {
    fn read(&self, start: usize, dst: &mut [f64]);
}
struct R64<'a>(&'a [f64]);
struct R32<'a>(&'a [f32]);
struct RQ<'a, Q: Resid>(&'a [f32], &'a [Q]);
impl RowRead for R64<'_> {
    #[inline(always)]
    fn read(&self, start: usize, dst: &mut [f64]) {
        dst.copy_from_slice(&self.0[start..start + dst.len()]);
    }
}
impl RowRead for R32<'_> {
    #[inline(always)]
    fn read(&self, start: usize, dst: &mut [f64]) {
        let n = dst.len();
        let h = &self.0[start..start + n];
        match linus_num::simd::Isa::detect() {
            #[cfg(target_arch = "x86_64")]
            linus_num::simd::Isa::Avx512 => unsafe { widen_avx512(h, dst) },
            #[cfg(target_arch = "x86_64")]
            linus_num::simd::Isa::Avx2 => unsafe { widen_avx2(h, dst) },
            _ => widen(h, dst),
        }
    }
}
impl<Q: Resid> RowRead for RQ<'_, Q> {
    #[inline(always)]
    fn read(&self, start: usize, dst: &mut [f64]) {
        let n = dst.len();
        let (h, q) = (&self.0[start..start + n], &self.1[start..start + n]);
        match linus_num::simd::Isa::detect() {
            #[cfg(target_arch = "x86_64")]
            linus_num::simd::Isa::Avx512 => unsafe { decode_avx512(h, q, dst) },
            #[cfg(target_arch = "x86_64")]
            linus_num::simd::Isa::Avx2 => unsafe { decode_avx2(h, q, dst) },
            _ => decode_rows(h, q, dst),
        }
    }
}

// decoding loops compiled for the vector ISA (the crate itself targets baseline x86-64)
#[inline(always)]
fn widen(h: &[f32], dst: &mut [f64]) {
    for (d, &v) in dst.iter_mut().zip(h) {
        *d = v as f64;
    }
}
#[inline(always)]
fn decode_rows<Q: Resid>(h: &[f32], q: &[Q], dst: &mut [f64]) {
    for ((d, &h), &q) in dst.iter_mut().zip(h).zip(q) {
        *d = decode(h, q);
    }
}
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512vl,avx512dq,avx2,fma")]
unsafe fn widen_avx512(h: &[f32], dst: &mut [f64]) {
    widen(h, dst)
}
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn widen_avx2(h: &[f32], dst: &mut [f64]) {
    widen(h, dst)
}
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512vl,avx512dq,avx2,fma")]
unsafe fn decode_avx512<Q: Resid>(h: &[f32], q: &[Q], dst: &mut [f64]) {
    decode_rows(h, q, dst)
}
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn decode_avx2<Q: Resid>(h: &[f32], q: &[Q], dst: &mut [f64]) {
    decode_rows(h, q, dst)
}

/// A stored pair-major 3-index tensor.
enum Tensor {
    F64(Vec<f64>),
    F48(Vec<f32>, Vec<i16>),
    F40(Vec<f32>, Vec<i8>),
    F32(Vec<f32>),
}

impl Tensor {
    fn fmt(&self) -> DfFormat {
        match self {
            Tensor::F64(_) => DfFormat::F64,
            Tensor::F48(..) => DfFormat::F48,
            Tensor::F40(..) => DfFormat::F40,
            Tensor::F32(_) => DfFormat::F32,
        }
    }
    fn bytes(&self) -> usize {
        match self {
            Tensor::F64(v) => v.len() * 8,
            Tensor::F48(h, _) => h.len() * 6,
            Tensor::F40(h, _) => h.len() * 5,
            Tensor::F32(v) => v.len() * 4,
        }
    }
}

/// `$body` with `$r` bound to a [`RowRead`] of the tensor (static dispatch per format).
macro_rules! with_rows {
    ($t:expr, |$r:ident| $body:expr) => {
        match $t {
            Tensor::F64(v) => {
                let $r = R64(v);
                $body
            }
            Tensor::F48(h, l) => {
                let $r = RQ(h, l);
                $body
            }
            Tensor::F40(h, l) => {
                let $r = RQ(h, l);
                $body
            }
            Tensor::F32(v) => {
                let $r = R32(v);
                $body
            }
        }
    };
}

enum Store {
    /// B = (mn|P) L^{-T}, pair-major
    B(Tensor),
    /// one-sided: fitted coefficients C = (mn|P) V^{-1}, pair-major (exchange pairs them with
    /// raw integrals computed per auxiliary block)
    Coef(Tensor),
    /// raw (mn|P), pair-major
    A(Vec<f64>),
    /// nothing stored
    Direct,
}

pub struct DfJk {
    nbf: usize,
    naux: usize,
    /// (ao_start, nfunc) per orbital shell
    shells: Vec<(usize, usize)>,
    aux3c: Aux3c,
    chunks: Vec<AuxChunk>,
    pairs: Vec<Pair3c>,
    /// first row of each kept pair
    x0: Vec<usize>,
    nx: usize,
    /// The Coulomb metric's Cholesky factor (the geometry's, shared with the derivatives)
    metric: Arc<CoulombMetric>,
    store: Store,
    /// exchange panels, one per shell
    panels: Vec<Panel>,
    /// long-range B^omega for the last requested omega
    lr: Option<(f64, Tensor)>,
    /// One-pass range-separated exchange a K + b K_LR(omega) ([`JkBuilder::set_exchange_operator`])
    xop: Option<Operator>,
    opts: DfConfig,
    pub stats: DfStats,
    /// HOOK (solvent, the density-fitted PCM): extra fit coefficients for the Coulomb
    /// expansion ([`CoulombHook`]).
    coulomb_hook: Option<Box<dyn CoulombHook>>,
    /// The in-memory exchange's panel scratch not in use, kept from one build to the next
    /// ([`PanelScratch`]).
    k_panels: std::sync::Mutex<Vec<PanelScratch>>,
    /// The block-sequential exchange's scratch, kept from one build to the next
    /// ([`BlockSeqScratch`]).
    kb_scratch: BlockSeqScratch,
}

/// The block-sequential exchange's scratch (`DfJk::k_blockseq`): the Z panels per (source, slot)
/// and the raw integrals of a block per on-the-fly source (about 1 GiB together), kept from one
/// build to the next as [`PanelScratch`] is, with the same rule: written in full before they are
/// read. (The lanes' partial K matrices are allocated per build, as the in-memory exchange's.)
#[derive(Default)]
struct BlockSeqScratch {
    zs: Vec<Vec<Vec<f64>>>,
    fly: Vec<Vec<f64>>,
}

/// A running group's panel scratch in the in-memory exchange (`DfJk::k_groups`): its Z panels per
/// (source, slot), the gathered tensor panel and the gathered factor rows. The builder keeps a
/// pool of them from one build to the next, one per group running at once (never more than were
/// alive at once when each group allocated its own), so the SCF's builds reuse the pages instead
/// of faulting in fresh ones: with the large Z panels, every build otherwise mapped and touched
/// them anew. A build writes every element of them it reads, so what an earlier build left never
/// leaks: the same arithmetic, the same bits. (The groups' partial K matrices are allocated per
/// build: kept, they stayed resident beside the phases after the SCF.)
#[derive(Default)]
struct PanelScratch {
    zs: Vec<Vec<Vec<f64>>>,
    tb: Vec<f64>,
    cbuf: Vec<f64>,
}

impl PanelScratch {
    /// Whether it has the shape [`PanelScratch::prepare`] gives it for `nsrc` sources and Z
    /// panels of lengths `zlen[slot]`.
    fn fits(&self, nsrc: usize, zlen: &[usize]) -> bool {
        self.zs.len() == nsrc
            && self
                .zs
                .iter()
                .all(|zsrc| zsrc.iter().map(Vec::len).eq(zlen.iter().copied()))
    }

    /// Shaped for Z panels of lengths `zlen[slot]` for `nsrc` sources, a tensor panel of `tb`
    /// and factor rows of `cbuf` elements.
    fn prepare(&mut self, nsrc: usize, zlen: &[usize], tb: usize, cbuf: usize) {
        self.zs.resize_with(nsrc, Vec::new);
        for zsrc in &mut self.zs {
            zsrc.resize_with(zlen.len(), Vec::new);
            for (z, &len) in zsrc.iter_mut().zip(zlen) {
                z.resize(len, 0.0);
            }
        }
        self.tb.resize(tb, 0.0);
        self.cbuf.resize(cbuf, 0.0);
    }
}

/// What the exchange's layouts depend on, known at setup before any tensor is stored, so that a
/// storage candidate's exchange workspace is known before it is chosen ([`DfJk::try_new`]): the
/// one place `DfJk::k_groups` and `DfJk::k_blockseq` take their block and group sizes from.
/// Everything here follows the problem and the build's shape alone, never the thread count or
/// the memory (decision D43).
struct KSizes<'a> {
    nbf: usize,
    naux: usize,
    nx: usize,
    /// The largest panel's partner functions.
    maxnl: usize,
    /// The largest panel's elements per auxiliary function (functions x partner functions).
    max_panel: usize,
    aux_shells: &'a [linus_ints::basis::Shell],
    k_block: usize,
    k_groups: usize,
}

/// The in-memory exchange's layout for one build shape (`DfJk::k_groups`).
struct GroupsLayout {
    /// Partial K sums (fixed groups of auxiliary blocks).
    ngroups: usize,
    /// Auxiliary functions per block, and the blocks.
    nq: usize,
    nqb: usize,
    /// A running group's gathered tensor panel, Z panels per slot and factor rows (elements).
    maxpanel: usize,
    zlen: Vec<usize>,
    cbuf_len: usize,
}

/// The block-sequential exchange's layout for one build shape (`DfJk::k_blockseq`).
struct BlockSeqLayout {
    blocks: Vec<common::AuxBlock>,
    nqmax: usize,
}

/// The block-sequential exchange's lanes: partial K matrices, each a share of every block's
/// auxiliary functions.
const NLANE: usize = 8;

impl<'a> KSizes<'a> {
    fn new(
        nbf: usize,
        naux: usize,
        nx: usize,
        panels: &[Panel],
        aux_shells: &'a [linus_ints::basis::Shell],
        opts: &DfConfig,
    ) -> Self {
        KSizes {
            nbf,
            naux,
            nx,
            maxnl: panels.iter().map(|p| p.flist.len()).max().unwrap_or(0),
            max_panel: panels
                .iter()
                .map(|p| p.nf * p.flist.len())
                .max()
                .unwrap_or(0),
            aux_shells,
            k_block: opts.k_block,
            k_groups: opts.k_groups,
        }
    }

    /// The in-memory layout for slots of `r` columns, `nsrc` stored sources and `nacc` K
    /// matrices: a fixed number of groups, at most 64 and ~2 GB of partial matrices; auxiliary
    /// blocks of 16..=128 functions sized so the Z panels stay below ~256 MB per group.
    fn groups(&self, r: &[usize], nsrc: usize, nacc: usize) -> GroupsLayout {
        let (nbf, naux) = (self.nbf, self.naux);
        let rtot: usize = r.iter().sum();
        let gmax = ((2usize << 30) / (8 * nbf * nbf * nacc).max(1)).clamp(4, 64);
        let ngroups = if self.k_groups > 0 {
            self.k_groups
        } else {
            gmax
        }
        .min(naux.div_ceil(16))
        .max(1);
        let nq_target = ((256usize << 20) / (8 * nbf * rtot * nsrc).max(1)).clamp(16, 128);
        let bpg = naux.div_ceil(ngroups * nq_target);
        let nq = if self.k_block > 0 {
            self.k_block
        } else {
            naux.div_ceil(ngroups * bpg)
        }
        .min(naux);
        let nqb = naux.div_ceil(nq);
        GroupsLayout {
            ngroups: ngroups.min(nqb),
            nq,
            nqb,
            maxpanel: self.max_panel * nq,
            zlen: r.iter().map(|&rs| nbf * nq * rs).collect(),
            cbuf_len: self.maxnl * r.iter().copied().max().unwrap_or(0),
        }
    }

    /// The block-sequential layout for slots of `r` columns, `nsrc` sources of which `nfly`
    /// computed per block: Z panels plus raw integrals of a block within ~1 GiB.
    fn blockseq(&self, r: &[usize], nsrc: usize, nfly: usize) -> BlockSeqLayout {
        let rtot: usize = r.iter().sum();
        let per_q = 8 * (self.nbf * rtot * nsrc + self.nx * nfly);
        let maxf = if self.k_block > 0 {
            self.k_block
        } else {
            ((1usize << 30) / per_q.max(1)).clamp(16, 256)
        };
        let blocks = common::aux_blocks(self.aux_shells, maxf, false);
        let nqmax = blocks.iter().map(|b| b.nq).max().unwrap_or(0);
        BlockSeqLayout { blocks, nqmax }
    }

    /// The bytes an exchange build of the SCF holds at its peak with storage `plan` ([`DfPlan`];
    /// range-separated: `omega` set) for occupied orbitals `occ` per density (one per spin
    /// channel), `threads` running: the in-memory exchange's partial K matrices of every group,
    /// the panel scratch of the groups running at once and the summed K matrices; the
    /// block-sequential one's Z panels and raw integrals of a block, its lanes' partial K
    /// matrices and the summed ones. Kept scratch counts once (a build of another shape frees it
    /// first). The SCF's builds take PSD factors (`DfJk::exchange`, `DfJk::exchange_combined`).
    fn scf_exchange_bytes(
        &self,
        plan: DfPlan,
        omega: bool,
        occ: &[usize],
        threads: usize,
    ) -> usize {
        let nd = occ.len();
        let s = 8 * self.nbf * self.nbf;
        // one K matrix per density (the combined a K + b K_LR of a range-separated hybrid)
        let nacc = nd;
        match plan {
            DfPlan::InCore {
                format: DfFormat::F64,
                long_range,
            } => {
                let nsrc = if omega && long_range { 2 } else { 1 };
                let g = self.groups(occ, nsrc, nacc);
                let panel = 8 * (nsrc * g.zlen.iter().sum::<usize>() + g.maxpanel + g.cbuf_len);
                g.ngroups * nacc * s + threads.min(g.ngroups).max(1) * panel + nacc * s
            }
            DfPlan::InCore { .. } | DfPlan::OneSided { .. } => {
                // compressed B (and B^omega), or the one-sided coefficients with the raw
                // integrals of the combined operator computed per block
                let one_sided = matches!(plan, DfPlan::OneSided { .. });
                let long_range = matches!(
                    plan,
                    DfPlan::InCore {
                        long_range: true,
                        ..
                    }
                );
                let (nsrc, nfly) = match (omega, one_sided, long_range) {
                    (true, true, _) => (2, 1),
                    (true, false, true) => (2, 0),
                    _ => (1, 0),
                };
                let b = self.blockseq(occ, nsrc, nfly);
                let rtot: usize = occ.iter().sum();
                8 * b.nqmax * (nsrc * self.nbf * rtot + nfly * self.nx) + (NLANE + 1) * nacc * s
            }
            DfPlan::Raw | DfPlan::Direct | DfPlan::Device | DfPlan::HostStreamed => 0,
        }
    }
}

/// Why a density-fitting setup failed ([`DfJk::try_new`], [`metric_cholesky`]).
#[derive(Clone, Debug, PartialEq)]
pub enum DfSetupError {
    /// No storage of the exchange tensor fits the budget (or, with [`DfConfig::room`], the
    /// whole process's room with its exchange workspace): `need` bytes at least (the most
    /// compact storage allowed: under [`DfPrecision::Auto`] an f64 one), `per_tensor` bytes per
    /// f64 tensor, `budget` bytes; `options` the storages allowed as (tensor bytes, exchange
    /// workspace bytes), for the memory that holds one.
    Memory {
        need: usize,
        per_tensor: usize,
        budget: usize,
        options: Vec<(usize, usize)>,
    },
    /// The auxiliary metric (P|Q) is not positive definite (Cholesky pivot `pivot` of `naux`):
    /// atoms (nearly) on top of each other, or a nearly linearly dependent auxiliary basis.
    Metric { pivot: usize, naux: usize },
    /// Exchange requested from a setup that stores no 3-index tensor
    /// ([`DfConfig::force_direct`]).
    NoTensor,
}

impl std::fmt::Display for DfSetupError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match *self {
            DfSetupError::Memory {
                need,
                per_tensor,
                budget,
                ..
            } => write!(
                f,
                "RI-K needs at least {} for its 3-index exchange tensor ({} per f64 tensor), more than its budget of {} \
                 (or, with its exchange workspace, than the room the process leaves it)",
                size(need),
                size(per_tensor),
                size(budget)
            ),
            DfSetupError::Metric { pivot, naux } => write!(
                f,
                "the auxiliary metric (P|Q) is not positive definite (Cholesky pivot {pivot} of {naux}): \
                 atoms too close to each other, or a nearly linearly dependent auxiliary basis set"
            ),
            DfSetupError::NoTensor => write!(
                f,
                "RI-K needs a stored 3-index tensor (force_direct is set); use seminumerical exchange (RIJ + sn-K)"
            ),
        }
    }
}

impl std::error::Error for DfSetupError {}

impl From<DfSetupError> for linus_base::error::LinusError {
    /// A budget too small is a [`LinusError::Resource`] of code `host_memory` with the ways out
    /// (the session's own conversion also names the `memory_gib` that holds the tensor); a
    /// metric that is not positive definite is [`LinusError::Numerical`].
    ///
    /// [`LinusError::Resource`]: linus_base::error::LinusError::Resource
    /// [`LinusError::Numerical`]: linus_base::error::LinusError::Numerical
    fn from(e: DfSetupError) -> Self {
        use linus_base::error::LinusError;
        match e {
            DfSetupError::Memory { need, budget, .. } => LinusError::host_memory(
                format!(
                    "{e}. Give linus more memory (memory_gib, or --memory on the command line), or use jk=\"rij_snk\" \
                     (--jk rij-snk: RI-J with seminumerical exchange, no 3-index exchange tensor)"
                ),
                linus_base::error::MemoryNeed::bytes(need as f64, budget as f64),
            ),
            DfSetupError::Metric { .. } => LinusError::Numerical(e.to_string()),
            DfSetupError::NoTensor => LinusError::Unsupported(e.to_string()),
        }
    }
}

/// `bytes` in GiB, MiB or KiB with three significant figures (1.23 GiB, 456 MiB).
fn size(bytes: usize) -> String {
    let b = bytes as f64;
    let (x, unit) = if b >= 1073741824.0 {
        (b / 1073741824.0, "GiB")
    } else if b >= 1048576.0 {
        (b / 1048576.0, "MiB")
    } else {
        (b / 1024.0, "KiB")
    };
    format!("{} {unit}", sig3(x))
}

/// `x` with three significant figures (12.3, 1.23, 0.123).
fn sig3(x: f64) -> String {
    let d = if x >= 100.0 {
        0
    } else if x >= 10.0 {
        1
    } else if x >= 1.0 {
        2
    } else {
        3
    };
    format!("{x:.d$}")
}

/// Default 3-index budget ([`DfConfig::mem_budget`] unset): that of an unset `memory_gib` with
/// nothing reserved, 75% of the total RAM or of the cgroup limit if lower, within the free memory
/// ([`linus_base::memory::MemoryLimit::df_budget`]).
fn default_budget() -> usize {
    linus_base::memory::MemoryLimit::from_gib(None).df_tensor_bytes()
}

/// Disjoint mutable slabs of `rows * ld` for consecutive row ranges.
fn split_rows(buf: &mut [f64], rows: impl Iterator<Item = usize>, ld: usize) -> Vec<&mut [f64]> {
    let mut out = vec![];
    let mut rest = buf;
    for r in rows {
        let (a, b) = rest.split_at_mut(r * ld);
        out.push(a);
        rest = b;
    }
    out
}

/// The density-fitting plan: where the 3-index tensor lives and in what form, chosen by
/// [`DfJk::try_new`] from the memory budget (with exchange the most accurate storage that fits;
/// see the module docs), or by the GPU's setup (`Device`, `HostStreamed`). Out-of-core or
/// distributed storage would be further variants.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DfPlan {
    /// In core, for J and K: B = (mn|P) L^{-T} in `format` (compressed below f64); with
    /// `long_range`, B^omega of a range-separated hybrid too (prepared at setup).
    InCore { format: DfFormat, long_range: bool },
    /// One-sided, for J and K: the fitted coefficients C = (mn|P) V^{-1} in `format`; exchange
    /// pairs them with raw integrals computed per auxiliary block (range-separated hybrids
    /// beyond the budget of B and B^omega).
    OneSided { format: DfFormat },
    /// J only, in core: the raw integrals (mn|P), the fit solved per build.
    Raw,
    /// J only, nothing stored: integral-direct (split-RI-J, two integral passes per build).
    Direct,
    /// The GPU's tensor B in device memory, f64 (feature `gpu`).
    Device,
    /// The GPU's tensor B in page-locked host memory, streamed to the device per build.
    HostStreamed,
}

impl DfPlan {
    /// The storage name of `resolved.record.df.storage`: "B", "B48", "B40", "B32" (in core),
    /// "coef", "coef48", "coef40", "coef32" (one-sided), "A" (raw), "direct", or on a GPU
    /// "device" and "host_streamed".
    pub fn name(&self) -> String {
        match self {
            DfPlan::InCore { format, .. } => format!("B{}", format.suffix()),
            DfPlan::OneSided { format } => format!("coef{}", format.suffix()),
            DfPlan::Raw => "A".into(),
            DfPlan::Direct => "direct".into(),
            DfPlan::Device => "device".into(),
            DfPlan::HostStreamed => "host_streamed".into(),
        }
    }

    /// The format of the stored tensor (f64 for the raw integrals, direct J and the GPU's).
    pub fn format(&self) -> DfFormat {
        match self {
            DfPlan::InCore { format, .. } | DfPlan::OneSided { format } => *format,
            DfPlan::Raw | DfPlan::Direct | DfPlan::Device | DfPlan::HostStreamed => DfFormat::F64,
        }
    }
}

/// Where the long-range exchange of a range-separated hybrid comes from in a setup.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DfLongRangeSource {
    /// No long-range tensor (no range separation, or J only).
    None,
    /// B^omega stored.
    Stored,
    /// One-sided coefficients with raw integrals per build ([`DfPlan::OneSided`]).
    OneSided,
}

/// What a density-fitting setup resolved to (`resolved.record.df` of the result): the plan, the
/// long-range source, the bytes stored, the budget (bytes) and the auxiliary basis size.
/// Serialized with the GiB keys of the report: storage, precision, long_range, store_gib,
/// budget_gib, naux, and on a GPU integrals (and integrals_note).
#[derive(Clone, Debug, PartialEq)]
pub struct ResolvedDf {
    pub plan: DfPlan,
    pub long_range: DfLongRangeSource,
    /// Bytes of the stored tensors (B or C, and B^omega).
    pub store_bytes: usize,
    /// The memory budget of the 3-index tensors (bytes).
    pub budget_bytes: usize,
    pub naux: usize,
    /// A GPU setup's three-center integrals: "device" (the generated kernels) or "cpu"; None
    /// on the CPU backend.
    pub integrals: Option<&'static str>,
    /// Why a GPU setup computed the integrals on the CPU (the device kernels do not cover the
    /// basis sets).
    pub integrals_note: Option<String>,
    /// A GPU setup's exchange block (auxiliary functions per K block) and the storage rung it
    /// set up on, with why (`linus::backend::gpu::df`'s `df_engine`); None on the CPU backend.
    pub k_block: Option<usize>,
    pub rung: Option<String>,
    /// A GPU setup's tensor allocation beyond the bytes granted at its decision (a plan
    /// violation, with a `plan_violation` warning).
    pub plan_violation: Option<String>,
}

impl serde::Serialize for ResolvedDf {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeStruct;
        const GIB: f64 = 1073741824.0;
        let mut m = s.serialize_struct("ResolvedDf", 6)?;
        m.serialize_field("storage", &self.plan.name())?;
        m.serialize_field("precision", &self.plan.format())?;
        m.serialize_field("long_range", &self.long_range)?;
        m.serialize_field("store_gib", &(self.store_bytes as f64 / GIB))?;
        m.serialize_field("budget_gib", &(self.budget_bytes as f64 / GIB))?;
        m.serialize_field("naux", &self.naux)?;
        if let Some(i) = self.integrals {
            m.serialize_field("integrals", i)?;
        }
        if let Some(n) = &self.integrals_note {
            m.serialize_field("integrals_note", n)?;
        }
        if let Some(q) = self.k_block {
            m.serialize_field("k_block", &q)?;
        }
        if let Some(r) = &self.rung {
            m.serialize_field("rung", r)?;
        }
        if let Some(v) = &self.plan_violation {
            m.serialize_field("plan_violation", v)?;
        }
        m.end()
    }
}

/// Row source of a gather: stored tensor rows (offset q0 into rows of naux) or a block of raw
/// integrals (rows of nq).
#[derive(Clone, Copy)]
enum Rows<'a> {
    T(&'a Tensor, usize, usize),
    Fly(&'a [f64], usize),
}

/// Source of one Z panel in an exchange build.
#[derive(Clone, Copy, PartialEq, Debug)]
enum Src {
    /// the stored tensor (B or C)
    Main,
    /// stored B^omega
    Lr,
    /// raw integrals of this operator, computed per auxiliary block
    Fly(Operator),
}

#[derive(Clone, Copy, PartialEq, Debug)]
enum Kind {
    /// lower triangle accumulated, mirrored at the end
    Lower,
    /// full, used as is
    Full,
    /// full M, result M + M^T
    FullSym,
}

/// accumulator `acc` += alpha * Z(src sl)[slot l] Z(src sr)[slot r]^T
#[derive(Clone, Copy, Debug)]
struct Term {
    acc: usize,
    l: usize,
    sl: usize,
    r: usize,
    sr: usize,
    alpha: f64,
}

/// The pair-major layout of the 3-index tensor and the exchange's panels, from the screening
/// of the orbital shell pairs against the auxiliary basis: what a storage's size and its
/// exchange workspace follow ([`DfJk::try_new`], [`DfJk::scf_storage_needs`]).
struct Layout {
    p3: Vec<Pair3c>,
    /// first row of each kept pair, and the rows
    x0: Vec<usize>,
    nx: usize,
    /// (ao_start, nfunc) per orbital shell
    shells: Vec<(usize, usize)>,
    panels: Vec<Panel>,
}

impl Layout {
    fn new(
        basis: &BasisSet,
        pairs: &PairSet,
        aux: &Aux3c,
        opts: &DfConfig,
        stats: &mut DfStats,
    ) -> Layout {
        let t = Instant::now();
        let q = pairs.schwarz();
        stats.t_schwarz = t.elapsed().as_secs_f64();
        let t = Instant::now();
        let kept: Vec<usize> = (0..pairs.len())
            .filter(|&i| pairs.pair(i).nprim > 0 && q[i] * aux.qmax >= opts.thresholds.pair)
            .collect();
        let p3: Vec<Pair3c> = kept
            .par_iter()
            .map(|&i| Pair3c::from_shell_pair(pairs.pair(i)))
            .collect();
        let mut x0 = Vec::with_capacity(p3.len());
        let mut nx = 0;
        for p in &p3 {
            x0.push(nx);
            nx += p.nab();
        }
        let shells: Vec<(usize, usize)> = basis
            .shells
            .iter()
            .map(|s| (s.ao_start, s.nfunc()))
            .collect();
        let ns = shells.len();
        let mut partners: Vec<Vec<(usize, usize, bool)>> = vec![vec![]; ns];
        for (i, p) in p3.iter().enumerate() {
            partners[p.sa].push((p.sb, i, false));
            if p.sa != p.sb {
                partners[p.sb].push((p.sa, i, true));
            }
        }
        // one exchange panel per shell (atom-level panels pad too much: +40-70% elements)
        let mut panels = vec![];
        for a in 0..ns {
            let mut ps: Vec<(usize, usize, bool)> = partners[a].clone();
            ps.sort_unstable_by_key(|x| x.0);
            let entry = ps
                .iter()
                .map(|&(_, pi, tr)| Some((pi as u32, tr)))
                .collect();
            let flist = ps
                .iter()
                .flat_map(|&(b, _, _)| (shells[b].0..shells[b].0 + shells[b].1).map(|f| f as u32))
                .collect();
            panels.push(Panel {
                s0: a,
                s1: a + 1,
                f0: shells[a].0,
                nf: shells[a].1,
                partners: ps.iter().map(|x| x.0).collect(),
                entry,
                flist,
            });
        }
        {
            // panel statistics: stored elements (with padding) vs significant pairs, and the
            // number of distinct partner lists per atom
            let pe: usize = panels.iter().map(|p| p.nf * p.flist.len()).sum();
            let se: usize = (0..ns)
                .map(|a| shells[a].1 * partners[a].iter().map(|x| shells[x.0].1).sum::<usize>())
                .sum();
            let mut groups = 0;
            for pan in &panels {
                let mut lists: Vec<Vec<usize>> = (pan.s0..pan.s1)
                    .map(|a| {
                        let mut v: Vec<usize> = partners[a].iter().map(|x| x.0).collect();
                        v.sort_unstable();
                        v
                    })
                    .collect();
                lists.sort();
                lists.dedup();
                groups += lists.len();
            }
            stats.panel_elems = pe;
            stats.shell_panel_elems = se;
            stats.distinct_lists = groups;
        }
        stats.nshell_pairs_kept = p3.len();
        stats.nrows = nx;
        stats.t_prep += t.elapsed().as_secs_f64();
        Layout {
            p3,
            x0,
            nx,
            shells,
            panels,
        }
    }
}

impl DfStats {
    /// The statistics of a setup before it starts.
    fn new(nbf: usize, naux: usize, nshell_pairs: usize) -> Self {
        DfStats {
            nbf,
            naux,
            nshell_pairs,
            nshell_pairs_kept: 0,
            panel_elems: 0,
            shell_panel_elems: 0,
            distinct_lists: 0,
            nrows: 0,
            bytes: 0,
            budget: 0,
            plan: DfPlan::Direct,
            t_schwarz: 0.0,
            t_prep: 0.0,
            t_metric: 0.0,
            t_metric_ints: 0.0,
            t_alloc: 0.0,
            t_ints: 0.0,
            t_fit: 0.0,
            t_lr_setup: 0.0,
            bytes_lr: 0,
            last: vec![],
        }
    }
}

impl DfJk {
    /// [`DfJk::try_new`] with a metric of its own; panics where that returns an error.
    pub fn new(
        basis: &BasisSet,
        pairs: &PairSet,
        auxb: &BasisSet,
        with_k: bool,
        opts: DfConfig,
        par: Par,
    ) -> Self {
        CoulombMetric::new(auxb, par)
            .and_then(|m| Self::try_new(basis, pairs, auxb, Arc::new(m), with_k, opts))
            .unwrap_or_else(|e| panic!("{e}"))
    }

    /// Build the fitting machinery on the Coulomb metric `metric` of `auxb`. `with_k`: store B
    /// for exchange (else raw integrals, J only). Fails if exchange is requested and no storage
    /// mode fits the memory budget (use seminumerical exchange then).
    pub fn try_new(
        basis: &BasisSet,
        pairs: &PairSet,
        auxb: &BasisSet,
        metric: Arc<CoulombMetric>,
        with_k: bool,
        opts: DfConfig,
    ) -> Result<Self, DfSetupError> {
        let mut stats = DfStats::new(basis.nbf, auxb.nbf, pairs.len());
        let t = Instant::now();
        let aux = Aux3c::new(auxb);
        let chunks = aux.chunks(0..auxb.shells.len());
        stats.t_prep += t.elapsed().as_secs_f64();
        let Layout {
            p3,
            x0,
            nx,
            shells,
            panels,
        } = Layout::new(basis, pairs, &aux, &opts, &mut stats);

        let naux = auxb.nbf;
        assert_eq!(
            metric.naux(),
            naux,
            "the metric of another auxiliary basis set"
        );
        // what the metric took, wherever it was built (crate::context)
        stats.t_metric_ints = metric.t_ints;
        stats.t_metric = metric.t_build;

        let budget = opts.mem_budget.unwrap_or_else(default_budget);
        stats.budget = budget;
        let bytes = nx * naux * 8;
        let plan = if with_k {
            let ksz = KSizes::new(basis.nbf, naux, nx, &panels, &aux.shells, &opts);
            let ws = |p: DfPlan| {
                ksz.scf_exchange_bytes(
                    p,
                    opts.omega.is_some(),
                    &opts.scf_occ,
                    rayon::current_num_threads(),
                )
            };
            Some(Self::plan(&opts, bytes, budget, ws)?)
        } else {
            None
        };
        let mut me = DfJk {
            nbf: basis.nbf,
            naux,
            shells,
            aux3c: aux,
            chunks,
            pairs: p3,
            x0,
            nx,
            metric,
            store: Store::Direct,
            panels,
            lr: None,
            xop: None,
            opts,
            stats,
            coulomb_hook: None,
            k_panels: Default::default(),
            kb_scratch: BlockSeqScratch::default(),
        };
        match plan {
            None if opts_fit(&me.opts, bytes, budget)
                && me.opts.room.is_none_or(|r| bytes <= r) =>
            {
                let t = Instant::now();
                let mut buf = vec![0.0; bytes / 8];
                buf.par_chunks_mut(1 << 16).for_each(|c| c.fill(0.0));
                me.stats.t_alloc = t.elapsed().as_secs_f64();
                let t = Instant::now();
                me.compute_ints(&mut buf, &me.chunks, naux, 0.0);
                me.stats.t_ints = t.elapsed().as_secs_f64();
                me.store = Store::A(buf);
                me.stats.plan = DfPlan::Raw;
                me.stats.bytes = bytes;
            }
            None => me.stats.plan = DfPlan::Direct,
            Some(DfPlan::InCore { format, long_range }) => {
                let tens = me.build_tensor(format, false, 0.0);
                me.stats.bytes = tens.bytes();
                me.store = Store::B(tens);
                me.stats.plan = DfPlan::InCore { format, long_range };
                if long_range {
                    me.ensure_long_range(me.opts.omega.unwrap());
                }
            }
            Some(DfPlan::OneSided { format }) => {
                let tens = me.build_tensor(format, true, 0.0);
                me.stats.bytes = tens.bytes();
                me.store = Store::Coef(tens);
                me.stats.plan = DfPlan::OneSided { format };
            }
            Some(p @ (DfPlan::Raw | DfPlan::Direct | DfPlan::Device | DfPlan::HostStreamed)) => {
                unreachable!("exchange plan {p:?}")
            }
        }
        Ok(me)
    }

    /// What the setup resolved to (`resolved.record.df`).
    pub fn resolved(&self) -> ResolvedDf {
        let st = &self.stats;
        ResolvedDf {
            plan: st.plan,
            long_range: if st.bytes_lr > 0 {
                DfLongRangeSource::Stored
            } else if matches!(st.plan, DfPlan::OneSided { .. }) {
                DfLongRangeSource::OneSided
            } else {
                DfLongRangeSource::None
            },
            store_bytes: st.bytes + st.bytes_lr,
            budget_bytes: st.budget,
            naux: st.naux,
            integrals: None,
            integrals_note: None,
            k_block: None,
            rung: None,
            plan_violation: None,
        }
    }

    /// Exchange storage for `bytes` per f64 tensor under `budget` (see the module docs):
    /// [`DfPlan::InCore`] or [`DfPlan::OneSided`].
    /// What each exchange storage `opts` allow needs, before anything is stored: (tensor bytes,
    /// exchange workspace bytes of the SCF's builds at `threads` threads) in the order
    /// [`DfJk::try_new`] tries them, from the same layout and rules (a plan states the memory that
    /// holds one without setting up the fit).
    pub fn scf_storage_needs(
        basis: &BasisSet,
        pairs: &PairSet,
        auxb: &BasisSet,
        opts: &DfConfig,
        threads: usize,
    ) -> Vec<(usize, usize)> {
        let aux = Aux3c::new(auxb);
        let mut stats = DfStats::new(basis.nbf, auxb.nbf, pairs.len());
        let l = Layout::new(basis, pairs, &aux, opts, &mut stats);
        let ksz = KSizes::new(basis.nbf, auxb.nbf, l.nx, &l.panels, &aux.shells, opts);
        Self::candidates(opts, l.nx * auxb.nbf * 8)
            .into_iter()
            .map(|(p, b)| {
                let ws = ksz.scf_exchange_bytes(p, opts.omega.is_some(), &opts.scf_occ, threads);
                (b, ws)
            })
            .collect()
    }

    /// The exchange storages `opts` allow for `bytes` per f64 tensor, in the order of preference,
    /// with their bytes (see the module docs; [`DfJk::plan`] takes the first that fits).
    fn candidates(opts: &DfConfig, bytes: usize) -> Vec<(DfPlan, usize)> {
        // formats allowed, most accurate first. Auto is f64 alone: a budget (a runner's memory
        // grant, the node's free memory) never selects a compressed tensor, so it changes
        // results by f64 round-off at most (B against the one-sided coefficients); F48 and F40
        // move SCF energies by ~1e-11 and ~1e-9 Eh, F32 by 1e-7..1e-6 Eh (CH3SH RHF/def2-TZVP
        // 9e-7 Eh; B3LYP la18/def2-TZVP 7e-8 Eh, u1an39 1.3e-7 Eh): each only on request
        let fmts: Vec<DfFormat> = match opts.precision {
            DfPrecision::Auto => vec![DfFormat::F64],
            DfPrecision::F64 => vec![DfFormat::F64],
            DfPrecision::F48 => vec![DfFormat::F48],
            DfPrecision::F40 => vec![DfFormat::F40],
            DfPrecision::F32 => vec![DfFormat::F32],
        };
        let per = |f: DfFormat| bytes / 8 * f.bytes();
        let mut cands: Vec<(DfPlan, usize)> = vec![];
        match opts.omega {
            None => {
                for &f in &fmts {
                    cands.push((
                        DfPlan::InCore {
                            format: f,
                            long_range: false,
                        },
                        per(f),
                    ));
                }
            }
            Some(_) => {
                let (st, one) = (
                    opts.long_range != DfLongRange::OneSided,
                    opts.long_range != DfLongRange::Stored,
                );
                // B and B^omega (no integrals per build) while both fit in f64, else the
                // one-sided coefficients (half the memory, exact); two compressed tensors
                // only on request
                if st && fmts[0] == DfFormat::F64 {
                    cands.push((
                        DfPlan::InCore {
                            format: DfFormat::F64,
                            long_range: true,
                        },
                        2 * per(DfFormat::F64),
                    ));
                }
                for &f in &fmts {
                    // (f32 coefficients are worse than f32 B: C = A V^-1 carries the metric's
                    // conditioning; u1an39/def2-TZVP wB97X: 3e-7 Eh and 1.2e-7 Eh/bohr)
                    if one {
                        cands.push((DfPlan::OneSided { format: f }, per(f)));
                    }
                    if !one && f != DfFormat::F64 {
                        cands.push((
                            DfPlan::InCore {
                                format: f,
                                long_range: true,
                            },
                            2 * per(f),
                        ));
                    }
                }
            }
        }
        cands
    }

    fn plan(
        opts: &DfConfig,
        bytes: usize,
        budget: usize,
        workspace: impl Fn(DfPlan) -> usize,
    ) -> Result<DfPlan, DfSetupError> {
        if opts.force_direct {
            return Err(DfSetupError::NoTensor);
        }
        let cands = Self::candidates(opts, bytes);
        // the tensors within their budget, and with the exchange workspace of their builds
        // within the whole process's room
        let options: Vec<(usize, usize)> = cands.iter().map(|&(p, b)| (b, workspace(p))).collect();
        for (&(p, b), &(_, ws)) in cands.iter().zip(&options) {
            if b <= budget && opts.room.is_none_or(|r| b + ws <= r) {
                return Ok(p);
            }
        }
        let need = cands.iter().map(|x| x.1).min().unwrap_or(bytes);
        Err(DfSetupError::Memory {
            need,
            per_tensor: bytes,
            budget,
            options,
        })
    }

    /// What the exchange's layouts depend on ([`KSizes`]).
    fn k_sizes(&self) -> KSizes<'_> {
        KSizes::new(
            self.nbf,
            self.naux,
            self.nx,
            &self.panels,
            &self.aux3c.shells,
            &self.opts,
        )
    }

    pub fn nrows(&self) -> usize {
        self.nx
    }

    /// (mn|P) for all kept pairs and the auxiliary `chunks` into `buf` (rows of `ld`).
    fn compute_ints(&self, buf: &mut [f64], chunks: &[AuxChunk], ld: usize, omega: f64) {
        let slabs = split_rows(buf, self.pairs.iter().map(|p| p.nab()), ld);
        let thresh = self.opts.thresholds.prim;
        let op = common::tensor_operator(omega);
        self.pairs
            .par_iter()
            .zip(slabs)
            .with_max_len(1)
            .for_each_init(Work3c::new, |w, (p, slab)| {
                eri3c::pair_block(p, chunks, slab, ld, thresh, &op, w);
            });
    }

    /// Fitted tensor rows L^{-1} a (B) or V^{-1} a (`coef`) of the integrals (mn|op|P) with
    /// op = 1/r (omega = 0) or erf(omega r)/r, stored in format `fmt`.
    fn build_tensor(&mut self, fmt: DfFormat, coef: bool, omega: f64) -> Tensor {
        let naux = self.naux;
        let l = self.metric.factor();
        // B rows: b = L^{-1} a for every stored row (C rows: L^{-T} b); blocks of rows are independent
        const RB: usize = 192;
        let fit = |blk: &mut [f64]| {
            let nr = blk.len() / naux;
            let mut m = MatMut::from_column_major_slice_mut(blk, naux, nr);
            linus_base::linalg::solve_lower_in_place(l, m.as_mut(), Par::Seq);
            if coef {
                linus_base::linalg::solve_upper_in_place(l.transpose(), m, Par::Seq);
            }
        };
        let (mut t_alloc, mut t_ints, mut t_fit) = (0.0, 0.0, 0.0);
        let out = if fmt == DfFormat::F64 {
            let t = Instant::now();
            let mut buf = vec![0.0; self.nx * naux];
            buf.par_chunks_mut(1 << 16).for_each(|c| c.fill(0.0));
            t_alloc += t.elapsed().as_secs_f64();
            let t = Instant::now();
            self.compute_ints(&mut buf, &self.chunks, naux, omega);
            t_ints += t.elapsed().as_secs_f64();
            let t = Instant::now();
            buf.par_chunks_mut(RB * naux).for_each(fit);
            t_fit += t.elapsed().as_secs_f64();
            Tensor::F64(buf)
        } else {
            // batches of whole pairs (~256 MiB of f64 rows), fitted in f64 and stored compressed
            let t = Instant::now();
            let len = self.nx * naux;
            let mut hi = vec![0.0f32; len];
            hi.par_chunks_mut(1 << 16).for_each(|c| c.fill(0.0));
            let mut lo16: Vec<i16> = if fmt == DfFormat::F48 {
                vec![0; len]
            } else {
                vec![]
            };
            let mut lo8: Vec<i8> = if fmt == DfFormat::F40 {
                vec![0; len]
            } else {
                vec![]
            };
            lo16.par_chunks_mut(1 << 16).for_each(|c| c.fill(0));
            lo8.par_chunks_mut(1 << 16).for_each(|c| c.fill(0));
            let batch_rows = ((256usize << 20) / (8 * naux)).max(RB);
            let mut tmp: Vec<f64> = vec![0.0; (batch_rows + 64) * naux];
            t_alloc += t.elapsed().as_secs_f64();
            let thresh = self.opts.thresholds.prim;
            let op = common::tensor_operator(omega);
            let np = self.pairs.len();
            let mut p0 = 0;
            while p0 < np {
                let mut p1 = p0;
                let mut rows = 0;
                while p1 < np && (rows == 0 || rows + self.pairs[p1].nab() <= batch_rows) {
                    rows += self.pairs[p1].nab();
                    p1 += 1;
                }
                let r0 = self.x0[p0];
                let t = Instant::now();
                let tb = &mut tmp[..rows * naux];
                let slabs = split_rows(tb, self.pairs[p0..p1].iter().map(|p| p.nab()), naux);
                self.pairs[p0..p1]
                    .par_iter()
                    .zip(slabs)
                    .with_max_len(1)
                    .for_each_init(Work3c::new, |w, (p, slab)| {
                        slab.fill(0.0);
                        eri3c::pair_block(p, &self.chunks, slab, naux, thresh, &op, w);
                    });
                t_ints += t.elapsed().as_secs_f64();
                let t = Instant::now();
                tb.par_chunks_mut(RB * naux).for_each(fit);
                let rng = r0 * naux..(r0 + rows) * naux;
                const CH: usize = 1 << 14;
                match fmt {
                    DfFormat::F48 => hi[rng.clone()]
                        .par_chunks_mut(CH)
                        .zip(lo16[rng].par_chunks_mut(CH))
                        .zip(tb.par_chunks(CH))
                        .for_each(|((h, l), s)| {
                            for ((h, l), &x) in h.iter_mut().zip(l.iter_mut()).zip(s) {
                                (*h, *l) = encode::<i16>(x);
                            }
                        }),
                    DfFormat::F40 => hi[rng.clone()]
                        .par_chunks_mut(CH)
                        .zip(lo8[rng].par_chunks_mut(CH))
                        .zip(tb.par_chunks(CH))
                        .for_each(|((h, l), s)| {
                            for ((h, l), &x) in h.iter_mut().zip(l.iter_mut()).zip(s) {
                                (*h, *l) = encode::<i8>(x);
                            }
                        }),
                    _ => hi[rng]
                        .par_chunks_mut(CH)
                        .zip(tb.par_chunks(CH))
                        .for_each(|(h, s)| {
                            for (h, &x) in h.iter_mut().zip(s) {
                                *h = x as f32;
                            }
                        }),
                }
                t_fit += t.elapsed().as_secs_f64();
                p0 = p1;
            }
            match fmt {
                DfFormat::F48 => Tensor::F48(hi, lo16),
                DfFormat::F40 => Tensor::F40(hi, lo8),
                _ => Tensor::F32(hi),
            }
        };
        if omega == 0.0 {
            self.stats.t_alloc += t_alloc;
            self.stats.t_ints += t_ints;
            self.stats.t_fit += t_fit;
        }
        out
    }

    /// Weighted density rows: D_mn + D_nm off the diagonal shell blocks, D_mn on them.
    fn density_rows(&self, d: MatRef<'_, f64>) -> Vec<f64> {
        let mut out = vec![0.0; self.nx];
        let slabs = split_rows(&mut out, self.pairs.iter().map(|p| p.nab()), 1);
        self.pairs.par_iter().zip(slabs).for_each(|(p, s)| {
            let (a0, na) = self.shells[p.sa];
            let (b0, nb) = self.shells[p.sb];
            for ia in 0..na {
                for ib in 0..nb {
                    let (m, n) = (a0 + ia, b0 + ib);
                    s[ia * nb + ib] = if p.sa == p.sb {
                        d[(m, n)]
                    } else {
                        d[(m, n)] + d[(n, m)]
                    };
                }
            }
        });
        out
    }

    fn rows_to_matrix(&self, jx: &[f64]) -> Mat<f64> {
        let mut j = Mat::<f64>::zeros(self.nbf, self.nbf);
        for (p, &x0) in self.pairs.iter().zip(&self.x0) {
            let (a0, na) = self.shells[p.sa];
            let (b0, nb) = self.shells[p.sb];
            for ia in 0..na {
                for ib in 0..nb {
                    let v = jx[x0 + ia * nb + ib];
                    j[(a0 + ia, b0 + ib)] = v;
                    j[(b0 + ib, a0 + ia)] = v;
                }
            }
        }
        j
    }

    /// gamma[d][P] = sum_x buf[x][P] w[d][x] over fixed row chunks, ordered reduction.
    fn contract_rows(buf: &[f64], ld: usize, ncol: usize, w: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let nx = w[0].len();
        let nd = w.len();
        const NCH: usize = 192;
        let parts: Vec<Vec<f64>> = ordered_chunks(nx, NCH)
            .map(|rows| {
                let mut g = vec![0.0; nd * ncol];
                for x in rows {
                    let row = &buf[x * ld..x * ld + ncol];
                    for (di, wd) in w.iter().enumerate() {
                        let wv = wd[x];
                        if wv == 0.0 {
                            continue;
                        }
                        let gd = &mut g[di * ncol..(di + 1) * ncol];
                        for (gv, &r) in gd.iter_mut().zip(row) {
                            *gv += wv * r;
                        }
                    }
                }
                g
            })
            .collect();
        let mut g = vec![0.0; nd * ncol];
        // ordered sum, parallel over columns
        g.par_chunks_mut(256).enumerate().for_each(|(ci, gc)| {
            for p in &parts {
                for (k, v) in gc.iter_mut().enumerate() {
                    *v += p[ci * 256 + k];
                }
            }
        });
        (0..nd)
            .map(|d| g[d * ncol..(d + 1) * ncol].to_vec())
            .collect()
    }

    /// out[d][x] (+)= sum_P buf[x][P] c[d][P]
    fn expand_rows(buf: &[f64], ld: usize, ncol: usize, c: &[Vec<f64>], out: &mut [Vec<f64>]) {
        let nd = c.len();
        let nx = out[0].len();
        let mut flat = vec![0.0; nd * nx];
        flat.par_chunks_mut(nd).enumerate().for_each(|(x, o)| {
            let row = &buf[x * ld..x * ld + ncol];
            for d in 0..nd {
                let mut s = 0.0;
                for (&r, &cv) in row.iter().zip(&c[d]) {
                    s += r * cv;
                }
                o[d] = s;
            }
        });
        for d in 0..nd {
            for x in 0..nx {
                out[d][x] += flat[x * nd + d];
            }
        }
    }

    /// [`DfJk::contract_rows`] of a compressed tensor (rows decoded to f64).
    fn contract_rows_r<R: RowRead>(
        buf: &R,
        ld: usize,
        ncol: usize,
        w: &[Vec<f64>],
    ) -> Vec<Vec<f64>> {
        let nx = w[0].len();
        let nd = w.len();
        const NCH: usize = 192;
        let parts: Vec<Vec<f64>> = ordered_chunks(nx, NCH)
            .map(|rows| {
                let mut g = vec![0.0; nd * ncol];
                let mut row = vec![0.0; ncol];
                for x in rows {
                    buf.read(x * ld, &mut row);
                    for (di, wd) in w.iter().enumerate() {
                        let wv = wd[x];
                        if wv == 0.0 {
                            continue;
                        }
                        let gd = &mut g[di * ncol..(di + 1) * ncol];
                        for (gv, &r) in gd.iter_mut().zip(&row) {
                            *gv += wv * r;
                        }
                    }
                }
                g
            })
            .collect();
        let mut g = vec![0.0; nd * ncol];
        g.par_chunks_mut(256).enumerate().for_each(|(ci, gc)| {
            for p in &parts {
                for (k, v) in gc.iter_mut().enumerate() {
                    *v += p[ci * 256 + k];
                }
            }
        });
        (0..nd)
            .map(|d| g[d * ncol..(d + 1) * ncol].to_vec())
            .collect()
    }

    /// [`DfJk::expand_rows`] of a compressed tensor.
    fn expand_rows_r<R: RowRead>(
        buf: &R,
        ld: usize,
        ncol: usize,
        c: &[Vec<f64>],
        out: &mut [Vec<f64>],
    ) {
        let nd = c.len();
        let nx = out[0].len();
        let mut flat = vec![0.0; nd * nx];
        flat.par_chunks_mut(nd * 64).enumerate().for_each_init(
            || vec![0.0; ncol],
            |row, (b, o)| {
                for (k, o) in o.chunks_mut(nd).enumerate() {
                    buf.read((b * 64 + k) * ld, row);
                    for d in 0..nd {
                        let mut s = 0.0;
                        for (&r, &cv) in row.iter().zip(&c[d]) {
                            s += r * cv;
                        }
                        o[d] = s;
                    }
                }
            },
        );
        for d in 0..nd {
            for x in 0..nx {
                out[d][x] += flat[x * nd + d];
            }
        }
    }

    fn solve_metric(&self, g: &mut [f64]) {
        let n = self.naux;
        let mut m = MatMut::from_column_major_slice_mut(g, n, 1);
        linus_base::linalg::solve_lower_in_place(self.metric.factor(), m.as_mut(), Par::Seq);
        linus_base::linalg::solve_upper_in_place(self.metric.factor().transpose(), m, Par::Seq);
    }

    /// c <- V c = L (L^T c).
    fn metric_mul(&self, c: &mut [f64]) {
        let n = self.naux;
        let l = self.metric.factor();
        let y: Vec<f64> = (0..n)
            .into_par_iter()
            .map(|j| {
                let col = l.col(j);
                let mut s = 0.0;
                for i in j..n {
                    s += col[i] * c[i];
                }
                s
            })
            .collect();
        c.par_iter_mut().enumerate().for_each(|(i, ci)| {
            let mut s = 0.0;
            for j in 0..=i {
                s += l[(i, j)] * y[j];
            }
            *ci = s;
        });
    }

    /// Coulomb matrices for symmetric (or symmetrized) densities.
    pub fn coulomb(&mut self, dens: &[MatRef<'_, f64>], _par: Par) -> Vec<Mat<f64>> {
        let naux = self.naux;
        let w: Vec<Vec<f64>> = dens.iter().map(|d| self.density_rows(*d)).collect();
        let mut jx = vec![vec![0.0; self.nx]; dens.len()];
        // J from the stored tensor if it is f64 or B in F48; otherwise integral-direct (the
        // Coulomb energy is large: F40/F32 rounding, and F48 rounding of the coefficients
        // C = A V^-1 amplified by the metric's conditioning (relative J error 2e-11 for water),
        // would show in it)
        let one_sided = matches!(self.store, Store::Coef(_));
        let from_tensor = |t: &Tensor| {
            t.fmt() == DfFormat::F64
                || (t.fmt() == DfFormat::F48 && !one_sided)
                || self.opts.j_from_f32
        };
        // HOOK (solvent, the density-fitted PCM): extra fit coefficients added before the expansion
        let mut hook = self.coulomb_hook.take();
        match &self.store {
            Store::B(Tensor::F64(buf)) => {
                let mut g = Self::contract_rows(buf, naux, naux, &w);
                self.hook_on_lgamma(&mut hook, &mut g);
                Self::expand_rows(buf, naux, naux, &g, &mut jx);
            }
            Store::Coef(Tensor::F64(buf)) => {
                // c = C^T w = V^-1 gamma; J = C (V c)
                let mut g = Self::contract_rows(buf, naux, naux, &w);
                self.hook_on_coefficients(&mut hook, &mut g);
                for gd in &mut g {
                    self.metric_mul(gd);
                }
                Self::expand_rows(buf, naux, naux, &g, &mut jx);
            }
            Store::B(t) | Store::Coef(t) if from_tensor(t) => with_rows!(t, |r| {
                let mut g = Self::contract_rows_r(&r, naux, naux, &w);
                if matches!(self.store, Store::Coef(_)) {
                    self.hook_on_coefficients(&mut hook, &mut g);
                    for gd in &mut g {
                        self.metric_mul(gd);
                    }
                } else {
                    self.hook_on_lgamma(&mut hook, &mut g);
                }
                Self::expand_rows_r(&r, naux, naux, &g, &mut jx);
            }),
            Store::A(buf) => {
                let mut g = Self::contract_rows(buf, naux, naux, &w);
                for gd in &mut g {
                    self.solve_metric(gd);
                }
                self.hook_on_coefficients(&mut hook, &mut g);
                Self::expand_rows(buf, naux, naux, &g, &mut jx);
            }
            _ => {
                let g = self.direct_gamma(&w);
                let mut d = g;
                for gd in &mut d {
                    self.solve_metric(gd);
                }
                self.hook_on_coefficients(&mut hook, &mut d);
                self.direct_expand(&d, &mut jx);
            }
        }
        self.coulomb_hook = hook;
        jx.iter().map(|x| self.rows_to_matrix(x)).collect()
    }

    /// HOOK (solvent, the density-fitted PCM): install or remove the [`CoulombHook`] of the
    /// Coulomb builds.
    pub fn set_coulomb_hook(&mut self, hook: Option<Box<dyn CoulombHook>>) {
        self.coulomb_hook = hook;
    }

    /// HOOK: fit coefficients c in place: c_k += e_k from the hook.
    fn hook_on_coefficients(&self, hook: &mut Option<Box<dyn CoulombHook>>, c: &mut [Vec<f64>]) {
        let Some(h) = hook.as_mut() else { return };
        let e = h.extra_coefficients(c, &|x: &mut [f64]| self.solve_metric(x));
        for (ck, ek) in c.iter_mut().zip(&e) {
            ck.iter_mut().zip(ek).for_each(|(a, b)| *a += b);
        }
    }

    /// HOOK: g = L^-1 gamma (B storage) in place: c = L^-T g to the hook, g_k += L^T e_k.
    fn hook_on_lgamma(&self, hook: &mut Option<Box<dyn CoulombHook>>, g: &mut [Vec<f64>]) {
        let Some(h) = hook.as_mut() else { return };
        let n = self.naux;
        let l = self.metric.factor();
        let c: Vec<Vec<f64>> = g
            .iter()
            .map(|gk| {
                let mut c = gk.clone();
                let m = MatMut::from_column_major_slice_mut(&mut c, n, 1);
                linus_base::linalg::solve_upper_in_place(l.transpose(), m, Par::Seq);
                c
            })
            .collect();
        let e = h.extra_coefficients(&c, &|x: &mut [f64]| self.solve_metric(x));
        for (gk, ek) in g.iter_mut().zip(&e) {
            // (L^T e)_j = sum_{i >= j} L_ij e_i
            let lt: Vec<f64> = (0..n)
                .into_par_iter()
                .map(|j| {
                    let col = l.col(j);
                    let mut s = 0.0;
                    for i in j..n {
                        s += col[i] * ek[i];
                    }
                    s
                })
                .collect();
            gk.iter_mut().zip(&lt).for_each(|(a, b)| *a += b);
        }
    }

    /// Integral-direct gamma_P = sum_mn (P|mn) D_mn in Hermite space (split-RI-J), fixed pair
    /// chunks with an ordered reduction.
    fn direct_gamma(&self, w: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let naux = self.naux;
        let np = self.pairs.len();
        const NCH: usize = 256;
        let thresh = self.opts.thresholds.prim;
        w.iter()
            .map(|wd| {
                let parts: Vec<Vec<f64>> = ordered_chunks(np, NCH)
                    .with_max_len(1)
                    .map_init(Work3c::new, |wk, pairs| {
                        let mut g = vec![0.0; naux];
                        let mut dh = vec![];
                        for i in pairs {
                            let p = &self.pairs[i];
                            let x0 = self.x0[i];
                            dh.resize(p.nprim() * linus_ints::core::hermite::nherm(p.lab), 0.0);
                            eri3c::pair_hermite_density(p, &wd[x0..x0 + p.nab()], &mut dh);
                            eri3c::pair_gamma(p, &self.chunks, &dh, &mut g, thresh, wk);
                        }
                        g
                    })
                    .collect();
                let mut g = vec![0.0; naux];
                g.par_chunks_mut(256).enumerate().for_each(|(ci, gc)| {
                    for p in &parts {
                        for (k, v) in gc.iter_mut().enumerate() {
                            *v += p[ci * 256 + k];
                        }
                    }
                });
                g
            })
            .collect()
    }

    /// Integral-direct J rows: out[x] += sum_P (P|mn) d_P in Hermite space.
    fn direct_expand(&self, d: &[Vec<f64>], out: &mut [Vec<f64>]) {
        let thresh = self.opts.thresholds.prim;
        for (dd, od) in d.iter().zip(out.iter_mut()) {
            let dc: Vec<Vec<eri3c::Lanes>> = self
                .chunks
                .iter()
                .map(|ch| {
                    let mut v = Vec::new();
                    eri3c::chunk_coefficients(ch, dd, &mut v);
                    v
                })
                .collect();
            let cmax: Vec<f64> = dc
                .iter()
                .map(|v| {
                    v.iter()
                        .flat_map(|l| l.iter())
                        .fold(0.0f64, |a, &x| a.max(x.abs()))
                })
                .collect();
            let slabs = split_rows(od, self.pairs.iter().map(|p| p.nab()), 1);
            self.pairs
                .par_iter()
                .zip(slabs)
                .with_max_len(1)
                .for_each_init(
                    || (Work3c::new(), Vec::new()),
                    |(wk, jh), (p, slab)| {
                        jh.resize(p.nprim() * linus_ints::core::hermite::nherm(p.lab), 0.0);
                        eri3c::pair_jh(p, &self.chunks, &dc, &cmax, jh, thresh, wk);
                        eri3c::pair_from_hermite(p, jh, slab);
                    },
                );
        }
    }

    /// Exchange matrices K\[D\] = sum_ls (ml|ns) D_ls for factored densities.
    pub fn exchange(&mut self, dens: &[Factored<'_>], par: Par) -> Vec<Mat<f64>> {
        self.exchange_long_range(dens, true, None, par).0
    }

    /// Build B^omega = (mn|erf(omega r)/r|P) L^{-T} (Coulomb metric, Psi4's wK scheme) unless
    /// it is already held for this omega (one-sided storage: nothing to build, the erf
    /// integrals come per build).
    pub fn ensure_long_range(&mut self, omega: f64) {
        assert!(omega > 0.0);
        if matches!(self.store, Store::Coef(_)) || matches!(&self.lr, Some((w, _)) if *w == omega) {
            return;
        }
        let fmt = match &self.store {
            Store::B(t) => t.fmt(),
            _ => panic!("long-range RI-K needs the B tensor (construct DfJk with with_k = true)"),
        };
        self.lr = None;
        let budget = self.opts.mem_budget.unwrap_or_else(default_budget);
        let per = self.nx * self.naux * fmt.bytes();
        assert!(
            2 * per <= budget,
            "long-range RI-K needs a second 3-index tensor ({:.1} GiB with B, budget {:.1} GiB); set DfConfig::omega at setup (one-sided storage) or raise the budget",
            (2 * per) as f64 / 1073741824.0,
            budget as f64 / 1073741824.0
        );
        let t = Instant::now();
        let buf = self.build_tensor(fmt, false, omega);
        self.stats.t_lr_setup = t.elapsed().as_secs_f64();
        self.stats.bytes_lr = buf.bytes();
        self.lr = Some((omega, buf));
    }

    /// Full-range exchange K\[D\] (if `want_k`) and long-range exchange K^omega\[D\] with the
    /// operator erf(omega r)/r (if `omega` is given), both from one pass over the auxiliary
    /// blocks sharing Z = B^T C:
    ///   K = Z Z^T,   K^omega = (Z Zw^T + Zw Z^T) / 2,   Zw = (B^omega)^T C
    /// (Psi4's wK: Coulomb 3-center integrals on one side, attenuated on the other, Coulomb
    /// metric; symmetrized).
    pub fn exchange_long_range(
        &mut self,
        dens: &[Factored<'_>],
        want_k: bool,
        omega: Option<f64>,
        par: Par,
    ) -> (Vec<Mat<f64>>, Vec<Mat<f64>>) {
        self.exchange_impl(dens, want_k, omega, None, par)
    }

    /// Range-separated exact exchange a K\[D\] + b K^omega\[D\] of PSD factors D = C C^T in one
    /// pass: with Z = B^T C and Zw = (B^omega)^T C, a Z Z^T + b (Z Zw^T + Zw Z^T)/2 = M + M^T for
    /// M = Z W^T, W = (a Z + b Zw)/2, so one full GEMM per block replaces the lower-triangle K
    /// GEMM plus the full K_LR GEMM (2/3 of the K-update flops). One-sided storage: M = Z_C Zc^T
    /// / 2 with Zc from the raw integrals of a/r + b erf(omega r)/r.
    pub fn exchange_combined(
        &mut self,
        dens: &[MatRef<'_, f64>],
        op: Operator,
        par: Par,
    ) -> Vec<Mat<f64>> {
        let fs: Vec<Factored<'_>> = dens.iter().map(|c| Factored::Psd(*c)).collect();
        self.exchange_impl(
            &fs,
            true,
            Some(op.omega()),
            Some((op.coulomb(), op.long_range())),
            par,
        )
        .0
    }

    fn exchange_impl(
        &mut self,
        dens: &[Factored<'_>],
        want_k: bool,
        omega: Option<f64>,
        combine: Option<(f64, f64)>,
        par: Par,
    ) -> (Vec<Mat<f64>>, Vec<Mat<f64>>) {
        if let Some(w) = omega {
            self.ensure_long_range(w);
        }
        let one_sided = matches!(self.store, Store::Coef(_));
        assert!(
            matches!(self.store, Store::B(_) | Store::Coef(_)),
            "RI-K needs the B tensor (construct DfJk with with_k = true)"
        );
        let nbf = self.nbf;
        let mut slots: Vec<MatRef<'_, f64>> = vec![];
        let mut kinds: Vec<Kind> = vec![];
        // terms with source 0 = Coulomb (B), 1 = erf (B^omega), as stored B and B^omega hold them
        let mut terms: Vec<Term> = vec![];
        let nd = dens.len();
        assert!(
            combine.is_none()
                || (omega.is_some() && dens.iter().all(|f| matches!(f, Factored::Psd(_))))
        );
        // combined: K accumulators only, sources Z and W (W overwrites Zw per block)
        let lr = omega.is_some() && combine.is_none();
        // accumulators: K of density d at d (if want_k), K^omega at nk + d
        let nk = if want_k { nd } else { 0 };
        let mut kinds_lr = vec![];
        for (d, f) in dens.iter().enumerate() {
            let (x, y) = match *f {
                Factored::Psd(c) => {
                    slots.push(c);
                    (slots.len() - 1, slots.len() - 1)
                }
                Factored::SymPair(x, y) | Factored::Pair(x, y) => {
                    slots.push(x);
                    slots.push(y);
                    (slots.len() - 2, slots.len() - 1)
                }
            };
            let (kind, kind_lr) = match f {
                Factored::Psd(_) => (Kind::Lower, Kind::FullSym),
                Factored::SymPair(..) => (Kind::FullSym, Kind::FullSym),
                Factored::Pair(..) => (Kind::Full, Kind::Full),
            };
            if combine.is_some() {
                kinds.push(Kind::FullSym);
                terms.push(Term {
                    acc: d,
                    l: x,
                    sl: 0,
                    r: x,
                    sr: 1,
                    alpha: 1.0,
                });
            } else if want_k {
                kinds.push(kind);
                terms.push(Term {
                    acc: d,
                    l: x,
                    sl: 0,
                    r: y,
                    sr: 0,
                    alpha: 1.0,
                });
            }
            if lr {
                kinds_lr.push(kind_lr);
                let a = nk + d;
                match f {
                    // (Z Zw^T + Zw Z^T)/2 -> M = Z Zw^T / 2, result M + M^T
                    Factored::Psd(_) => terms.push(Term {
                        acc: a,
                        l: x,
                        sl: 0,
                        r: x,
                        sr: 1,
                        alpha: 0.5,
                    }),
                    // D = XY^T + YX^T: (Z_X Zw_Y^T + Z_Y Zw_X^T), symmetrized
                    Factored::SymPair(..) => {
                        terms.push(Term {
                            acc: a,
                            l: x,
                            sl: 0,
                            r: y,
                            sr: 1,
                            alpha: 0.5,
                        });
                        terms.push(Term {
                            acc: a,
                            l: y,
                            sl: 0,
                            r: x,
                            sr: 1,
                            alpha: 0.5,
                        });
                    }
                    // D = XY^T: (Z_X Zw_Y^T + Zw_X Z_Y^T)/2
                    Factored::Pair(..) => {
                        terms.push(Term {
                            acc: a,
                            l: x,
                            sl: 0,
                            r: y,
                            sr: 1,
                            alpha: 0.5,
                        });
                        terms.push(Term {
                            acc: a,
                            l: x,
                            sl: 1,
                            r: y,
                            sr: 0,
                            alpha: 0.5,
                        });
                    }
                }
            }
        }
        kinds.extend(kinds_lr);
        let nacc = kinds.len();
        for s in &slots {
            assert_eq!(s.nrows(), nbf);
        }
        let r: Vec<usize> = slots.iter().map(|s| s.ncols()).collect();
        let rtot: usize = r.iter().sum();
        if rtot == 0 || nacc == 0 {
            let z = |n: usize| (0..n).map(|_| Mat::zeros(nbf, nbf)).collect::<Vec<_>>();
            return (z(nk), z(if lr { nd } else { 0 }));
        }
        // sources and the terms in their indices
        let (srcs, terms, w_combine): (Vec<Src>, Vec<Term>, Option<(f64, f64)>) = if !one_sided {
            let srcs = if omega.is_some() {
                vec![Src::Main, Src::Lr]
            } else {
                vec![Src::Main]
            };
            (srcs, terms, combine)
        } else {
            // one-sided: Z from C on one side, raw integrals on the other; a term Z_s Z_t^T
            // (s, t: 0 Coulomb, 1 erf; at least one Coulomb) becomes Z_C Zraw_t^T (s = 0) or
            // Zraw_s Z_C^T (t = 0); symmetric lower-triangle products become M + M^T, M halved
            let w = omega.unwrap_or(0.0);
            let mut srcs = vec![Src::Main];
            let mut src_of = |op: Operator| -> usize {
                let s = Src::Fly(op);
                match srcs.iter().position(|x| *x == s) {
                    Some(i) => i,
                    None => {
                        srcs.push(s);
                        srcs.len() - 1
                    }
                }
            };
            let mut out = vec![];
            if let Some((ca, cb)) = combine {
                let f = src_of(Operator::Attenuated {
                    coulomb: ca,
                    long_range: cb,
                    omega: w,
                });
                for t in &terms {
                    out.push(Term {
                        sl: 0,
                        sr: f,
                        alpha: 0.5,
                        ..*t
                    });
                }
            } else {
                for t in &terms {
                    let op = |s: usize| {
                        if s == 0 {
                            Operator::Coulomb
                        } else {
                            Operator::LongRange { omega: w }
                        }
                    };
                    let mut t2 = if t.sl == 0 {
                        Term {
                            sl: 0,
                            sr: src_of(op(t.sr)),
                            ..*t
                        }
                    } else {
                        Term {
                            sl: src_of(op(t.sl)),
                            sr: 0,
                            ..*t
                        }
                    };
                    if kinds[t.acc] == Kind::Lower {
                        t2.alpha *= 0.5;
                    }
                    out.push(t2);
                }
                for k in kinds.iter_mut() {
                    if *k == Kind::Lower {
                        *k = Kind::FullSym;
                    }
                }
            }
            (srcs, out, None)
        };
        let blockseq = one_sided || matches!(&self.store, Store::B(t) if t.fmt() != DfFormat::F64);
        let (acc, timings) = if blockseq {
            let mut scratch = std::mem::take(&mut self.kb_scratch);
            let out = self.k_blockseq(&slots, &srcs, &kinds, &terms, w_combine, &mut scratch);
            self.kb_scratch = scratch;
            out
        } else {
            self.k_groups(&slots, &srcs, &kinds, &terms, w_combine, par)
        };
        self.stats.last.extend(timings);
        let t = Instant::now();
        let mut out: Vec<Mat<f64>> = Vec::with_capacity(nacc);
        for (kv, kind) in acc.iter().zip(&kinds) {
            let k = Mat::from_fn(nbf, nbf, |i, j| match kind {
                Kind::Lower => {
                    if i >= j {
                        kv[j * nbf + i]
                    } else {
                        kv[i * nbf + j]
                    }
                }
                Kind::FullSym => kv[j * nbf + i] + kv[i * nbf + j],
                Kind::Full => kv[j * nbf + i],
            });
            out.push(k);
        }
        self.stats
            .last
            .push(("k_reduce".into(), t.elapsed().as_secs_f64()));
        let klr = out.split_off(nk);
        (out, klr)
    }

    /// Panel T[(fa, q), n'] (column-major, rows na * nqq) of rows `src` for panel `pan`.
    #[inline]
    fn gather_panel(&self, pan: &Panel, src: Rows<'_>, nqq: usize, tb: &mut [f64]) {
        match src {
            Rows::T(t, ld, off) => {
                with_rows!(t, |r| self.gather_panel_r(pan, &r, ld, off, nqq, tb))
            }
            Rows::Fly(b, ld) => self.gather_panel_r(pan, &R64(b), ld, 0, nqq, tb),
        }
    }

    #[inline(always)]
    fn gather_panel_r<R: RowRead>(
        &self,
        pan: &Panel,
        b: &R,
        ld: usize,
        off: usize,
        nqq: usize,
        tb: &mut [f64],
    ) {
        let mrows = pan.nf * nqq;
        let np = pan.partners.len();
        let mut noff = 0;
        for (j, &bsh) in pan.partners.iter().enumerate() {
            let nb = self.shells[bsh].1;
            for ib in 0..nb {
                let col = &mut tb[(noff + ib) * mrows..(noff + ib + 1) * mrows];
                for a in pan.s0..pan.s1 {
                    let (fa0, nfa) = self.shells[a];
                    let fa0 = fa0 - pan.f0;
                    match pan.entry[(a - pan.s0) * np + j] {
                        Some((pi, tr)) => {
                            let x0 = self.x0[pi as usize];
                            for ia in 0..nfa {
                                let x = if tr {
                                    x0 + ib * nfa + ia
                                } else {
                                    x0 + ia * nb + ib
                                };
                                b.read(
                                    x * ld + off,
                                    &mut col[(fa0 + ia) * nqq..(fa0 + ia + 1) * nqq],
                                );
                            }
                        }
                        None => col[fa0 * nqq..(fa0 + nfa) * nqq].fill(0.0),
                    }
                }
            }
            noff += nb;
        }
    }

    /// Row-major copies of the factors (rows gathered per panel on the fly; C stays in cache).
    fn factor_rows(slots: &[MatRef<'_, f64>]) -> Vec<Vec<f64>> {
        slots
            .iter()
            .map(|c| {
                let (n, ncol) = (c.nrows(), c.ncols());
                let mut v = vec![0.0; n * ncol];
                for i in 0..n {
                    for j in 0..ncol {
                        v[i * ncol + j] = c[(i, j)];
                    }
                }
                v
            })
            .collect()
    }

    /// In-memory exchange: auxiliary blocks in a fixed number of groups, each group sequential
    /// with its own partial K (see the module docs), the groups' panels in the kept pool
    /// ([`PanelScratch`]). Sources: stored tensors only.
    #[allow(clippy::too_many_arguments)]
    fn k_groups(
        &self,
        slots: &[MatRef<'_, f64>],
        srcs: &[Src],
        kinds: &[Kind],
        terms: &[Term],
        combine: Option<(f64, f64)>,
        par: Par,
    ) -> (Vec<Vec<f64>>, Vec<(String, f64)>) {
        let (nbf, naux) = (self.nbf, self.naux);
        let mut timings: Vec<(String, f64)> = vec![];
        let nacc = kinds.len();
        let nslot = slots.len();
        let nsrc = srcs.len();
        let r: Vec<usize> = slots.iter().map(|s| s.ncols()).collect();
        let tens: Vec<&Tensor> = srcs
            .iter()
            .map(|s| match s {
                Src::Main => match &self.store {
                    Store::B(t) | Store::Coef(t) => t,
                    _ => unreachable!(),
                },
                Src::Lr => &self.lr.as_ref().unwrap().1,
                Src::Fly(_) => unreachable!("k_groups takes stored tensors only"),
            })
            .collect();
        let crow = Self::factor_rows(slots);
        // fixed groups (partial K sums) and blocks, from the problem alone (KSizes::groups)
        let GroupsLayout {
            ngroups,
            nq,
            nqb,
            maxpanel,
            zlen,
            cbuf_len,
        } = self.k_sizes().groups(&r, nsrc, nacc);
        let threads = par.degree().max(1);
        let inner = if ngroups >= threads {
            Par::Seq
        } else {
            Par::rayon(threads.div_ceil(ngroups))
        };
        let t = Instant::now();
        let pool = &self.k_panels;
        // a build of another shape (the closed-shell probe's two-spin builds after the
        // iterations') frees the old panels before it allocates its own
        {
            let mut pool = pool.lock().unwrap_or_else(|e| e.into_inner());
            if !pool.iter().all(|ps| ps.fits(nsrc, &zlen)) {
                pool.clear();
            }
        }
        let partials: Vec<(Vec<Mat<f64>>, [f64; 3])> = (0..ngroups)
            .into_par_iter()
            .with_max_len(1)
            .map(|g| {
                let qb0 = g * nqb / ngroups;
                let qb1 = (g + 1) * nqb / ngroups;
                let mut kp: Vec<Mat<f64>> = (0..nacc).map(|_| Mat::zeros(nbf, nbf)).collect();
                // a panel scratch from the pool, returned when the group is done
                let mut ps = pool
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .pop()
                    .unwrap_or_default();
                ps.prepare(nsrc, &zlen, maxpanel, cbuf_len);
                // zs[src][slot]
                let PanelScratch { zs, tb, cbuf } = &mut ps;
                let mut tm3 = [0.0f64; 3];
                for qb in qb0..qb1 {
                    let q0 = qb * nq;
                    let nqq = nq.min(naux - q0);
                    for pan in self.panels.iter() {
                        let (a0, na) = (pan.f0, pan.nf);
                        let nl = pan.flist.len();
                        if nl == 0 {
                            for zsrc in zs.iter_mut() {
                                for s in 0..nslot {
                                    zsrc[s][a0 * nqq * r[s]..(a0 + na) * nqq * r[s]].fill(0.0);
                                }
                            }
                            continue;
                        }
                        let mrows = na * nqq;
                        for (src, zsrc) in zs.iter_mut().enumerate() {
                            let rows = Rows::T(tens[src], naux, q0);
                            let tg = Instant::now();
                            self.gather_panel(pan, rows, nqq, tb);
                            let tm = MatRef::from_column_major_slice(&tb[..mrows * nl], mrows, nl);
                            tm3[0] += tg.elapsed().as_secs_f64();
                            let tg = Instant::now();
                            for s in 0..nslot {
                                let rs = r[s];
                                if rs == 0 {
                                    continue;
                                }
                                for (i, &f) in pan.flist.iter().enumerate() {
                                    cbuf[i * rs..(i + 1) * rs].copy_from_slice(
                                        &crow[s][f as usize * rs..(f as usize + 1) * rs],
                                    );
                                }
                                let cm = MatRef::from_row_major_slice(&cbuf[..nl * rs], nl, rs);
                                let dst = MatMut::from_row_major_slice_mut(
                                    &mut zsrc[s][a0 * nqq * rs..(a0 + na) * nqq * rs],
                                    mrows,
                                    rs,
                                );
                                matmul(dst, Accum::Replace, tm, cm, 1.0, inner);
                            }
                            tm3[1] += tg.elapsed().as_secs_f64();
                        }
                    }
                    if let Some((ca, cb)) = combine {
                        // W = (a Z + b Zw) / 2 in place of Zw
                        let (z0, z1) = zs.split_at_mut(1);
                        for s in 0..nslot {
                            let len = nbf * nqq * r[s];
                            for (w, z) in z1[0][s][..len].iter_mut().zip(&z0[0][s][..len]) {
                                *w = 0.5 * (ca * z + cb * *w);
                            }
                        }
                    }
                    let tg = Instant::now();
                    for tt in terms {
                        let zl = MatRef::from_row_major_slice(
                            &zs[tt.sl][tt.l][..nbf * nqq * r[tt.l]],
                            nbf,
                            nqq * r[tt.l],
                        );
                        let zr = MatRef::from_row_major_slice(
                            &zs[tt.sr][tt.r][..nbf * nqq * r[tt.r]],
                            nbf,
                            nqq * r[tt.r],
                        );
                        if kinds[tt.acc] == Kind::Lower {
                            tri::matmul(
                                kp[tt.acc].as_mut(),
                                BlockStructure::TriangularLower,
                                Accum::Add,
                                zl,
                                BlockStructure::Rectangular,
                                zr.transpose(),
                                BlockStructure::Rectangular,
                                tt.alpha,
                                inner,
                            );
                        } else {
                            matmul(
                                kp[tt.acc].as_mut(),
                                Accum::Add,
                                zl,
                                zr.transpose(),
                                tt.alpha,
                                inner,
                            );
                        }
                    }
                    tm3[2] += tg.elapsed().as_secs_f64();
                }
                pool.lock().unwrap_or_else(|e| e.into_inner()).push(ps);
                (kp, tm3)
            })
            .collect();
        timings.push(("k_blocks".into(), t.elapsed().as_secs_f64()));
        let mut tsum = [0.0; 3];
        for (_, tm3) in &partials {
            for i in 0..3 {
                tsum[i] += tm3[i];
            }
        }
        timings.push(("k_cpu_gather".into(), tsum[0]));
        timings.push(("k_cpu_zgemm".into(), tsum[1]));
        timings.push(("k_cpu_kgemm".into(), tsum[2]));
        timings.push(("k_nq".into(), nq as f64));
        timings.push(("k_groups".into(), ngroups as f64));
        let partials: Vec<Vec<Mat<f64>>> = partials.into_iter().map(|x| x.0).collect();
        let t = Instant::now();
        let mut acc: Vec<Vec<f64>> = Vec::with_capacity(nacc);
        for (ai, kind) in kinds.iter().enumerate() {
            let mut kv = vec![0.0; nbf * nbf];
            // ordered sum over groups, parallel over columns (column-major)
            kv.par_chunks_mut(nbf).enumerate().for_each(|(j, c)| {
                let i0 = if *kind == Kind::Lower { j } else { 0 };
                for p in &partials {
                    let pc = p[ai].col(j);
                    for i in i0..nbf {
                        c[i] += pc[i];
                    }
                }
            });
            acc.push(kv);
        }
        timings.push(("k_sum".into(), t.elapsed().as_secs_f64()));
        (acc, timings)
    }

    /// Exchange one auxiliary block at a time with all threads: raw integrals of the `Fly`
    /// sources (parallel over pairs), Z per shell panel (parallel over panels), then the K
    /// update in a fixed number of lanes (each a share of the block's auxiliary functions, one
    /// sequential GEMM per lane and term into the lane's partial K; lanes summed in order at
    /// the end). Deterministic: nothing depends on the thread count. Workspace: one block of
    /// integrals and Z panels (~1 GiB) and the lanes' partial K matrices.
    #[allow(clippy::too_many_arguments)]
    fn k_blockseq(
        &self,
        slots: &[MatRef<'_, f64>],
        srcs: &[Src],
        kinds: &[Kind],
        terms: &[Term],
        combine: Option<(f64, f64)>,
        scratch: &mut BlockSeqScratch,
    ) -> (Vec<Vec<f64>>, Vec<(String, f64)>) {
        let (nbf, naux, nx) = (self.nbf, self.naux, self.nx);
        let nacc = kinds.len();
        let nslot = slots.len();
        let nsrc = srcs.len();
        let r: Vec<usize> = slots.iter().map(|s| s.ncols()).collect();
        let nfly = srcs.iter().filter(|s| matches!(s, Src::Fly(_))).count();
        let crow = Self::factor_rows(slots);
        // blocks: Z panels plus raw integrals of the block within ~1 GiB (KSizes::blockseq)
        let BlockSeqLayout { blocks, nqmax } = self.k_sizes().blockseq(&r, nsrc, nfly);
        let thresh = self.opts.thresholds.prim;
        // K update in a fixed number of lanes (NLANE), each a share of every block's auxiliary
        // functions with its own partial K (one sequential GEMM per lane and term, as the
        // group scheme's); lanes are summed in order at the end
        let mut lanes: Vec<Vec<Mat<f64>>> = (0..NLANE)
            .map(|_| (0..nacc).map(|_| Mat::zeros(nbf, nbf)).collect())
            .collect();
        // the kept scratch (BlockSeqScratch), shaped for this build; a build of another shape
        // frees the old scratch before it allocates its own
        let BlockSeqScratch { zs, fly } = scratch;
        let fits = zs.len() == nsrc
            && zs.iter().all(|zsrc| {
                zsrc.iter()
                    .map(Vec::len)
                    .eq(r.iter().map(|&rs| nbf * nqmax * rs))
            });
        if !fits {
            zs.clear();
            fly.clear();
        }
        zs.resize_with(nsrc, Vec::new);
        for zsrc in zs.iter_mut() {
            zsrc.resize_with(nslot, Vec::new);
            for (z, &rs) in zsrc.iter_mut().zip(&r) {
                z.resize(nbf * nqmax * rs, 0.0);
            }
        }
        fly.resize_with(nsrc, Vec::new);
        for (b, s) in fly.iter_mut().zip(srcs) {
            b.resize(
                if matches!(s, Src::Fly(_)) {
                    nx * nqmax
                } else {
                    0
                },
                0.0,
            );
        }
        let (mut t_ints, mut t_z, mut t_k) = (0.0, 0.0, 0.0);
        let t_all = Instant::now();
        for crate::df::common::AuxBlock {
            p0: q0,
            nq: nqq,
            shells: sh,
            ..
        } in &blocks
        {
            let (q0, nqq) = (*q0, *nqq);
            // raw integrals of the block
            let t = Instant::now();
            let chunks = if nfly > 0 {
                self.aux3c.chunks(sh.clone())
            } else {
                vec![]
            };
            for (si, s) in srcs.iter().enumerate() {
                if let Src::Fly(op) = s {
                    let fb = &mut fly[si][..nx * nqq];
                    let slabs = split_rows(fb, self.pairs.iter().map(|p| p.nab()), nqq);
                    self.pairs
                        .par_iter()
                        .zip(slabs)
                        .with_max_len(4)
                        .for_each_init(Work3c::new, |w, (p, slab)| {
                            slab.fill(0.0);
                            eri3c::pair_block(p, &chunks, slab, nqq, thresh, op, w);
                        });
                }
            }
            t_ints += t.elapsed().as_secs_f64();
            // Z panels: per (source, slot) buffer, the rows of each panel
            let t = Instant::now();
            {
                let mut per_panel: Vec<Vec<&mut [f64]>> = self
                    .panels
                    .iter()
                    .map(|_| Vec::with_capacity(nsrc * nslot))
                    .collect();
                for zsrc in zs.iter_mut() {
                    for (s, z) in zsrc.iter_mut().enumerate() {
                        let rs = r[s];
                        let parts = split_rows(
                            &mut z[..nbf * nqq * rs],
                            self.panels.iter().map(|p| p.nf),
                            nqq * rs,
                        );
                        for (pp, part) in per_panel.iter_mut().zip(parts) {
                            pp.push(part);
                        }
                    }
                }
                let rows: Vec<Rows<'_>> = srcs
                    .iter()
                    .enumerate()
                    .map(|(si, s)| match s {
                        Src::Main | Src::Lr => {
                            let t = if *s == Src::Main {
                                match &self.store {
                                    Store::B(t) | Store::Coef(t) => t,
                                    _ => unreachable!(),
                                }
                            } else {
                                &self.lr.as_ref().unwrap().1
                            };
                            Rows::T(t, naux, q0)
                        }
                        Src::Fly(_) => Rows::Fly(&fly[si][..nx * nqq], nqq),
                    })
                    .collect();
                self.panels.par_iter().zip(per_panel).for_each_init(
                    || (Vec::<f64>::new(), Vec::<f64>::new()),
                    |(tb, cbuf), (pan, mut dst)| {
                        let nl = pan.flist.len();
                        if nl == 0 {
                            for d in dst.iter_mut() {
                                d.fill(0.0);
                            }
                            return;
                        }
                        let mrows = pan.nf * nqq;
                        tb.resize(mrows * nl, 0.0);
                        for (si, src) in rows.iter().enumerate() {
                            self.gather_panel(pan, *src, nqq, tb);
                            let tm = MatRef::from_column_major_slice(&tb[..mrows * nl], mrows, nl);
                            for s in 0..nslot {
                                let rs = r[s];
                                if rs == 0 {
                                    continue;
                                }
                                cbuf.resize(nl * rs, 0.0);
                                for (i, &f) in pan.flist.iter().enumerate() {
                                    cbuf[i * rs..(i + 1) * rs].copy_from_slice(
                                        &crow[s][f as usize * rs..(f as usize + 1) * rs],
                                    );
                                }
                                let cm = MatRef::from_row_major_slice(&cbuf[..nl * rs], nl, rs);
                                let d = MatMut::from_row_major_slice_mut(
                                    &mut dst[si * nslot + s][..],
                                    mrows,
                                    rs,
                                );
                                matmul(d, Accum::Replace, tm, cm, 1.0, Par::Seq);
                            }
                        }
                    },
                );
            }
            if let Some((ca, cb)) = combine {
                // W = (a Z + b Zw) / 2 in place of Zw
                let (z0, z1) = zs.split_at_mut(1);
                for s in 0..nslot {
                    let len = nbf * nqq * r[s];
                    z1[0][s][..len]
                        .par_chunks_mut(1 << 14)
                        .zip(z0[0][s][..len].par_chunks(1 << 14))
                        .for_each(|(w, z)| {
                            for (w, z) in w.iter_mut().zip(z) {
                                *w = 0.5 * (ca * z + cb * *w);
                            }
                        });
                }
            }
            t_z += t.elapsed().as_secs_f64();
            // K update: lane l takes the block's auxiliary functions [l nqq / NLANE, (l+1) nqq / NLANE)
            let t = Instant::now();
            lanes.par_iter_mut().enumerate().for_each(|(l, kp)| {
                let (qa, qb) = (l * nqq / NLANE, (l + 1) * nqq / NLANE);
                if qa == qb {
                    return;
                }
                for tt in terms {
                    let (rl, rr) = (r[tt.l], r[tt.r]);
                    let zl = MatRef::from_row_major_slice(
                        &zs[tt.sl][tt.l][..nbf * nqq * rl],
                        nbf,
                        nqq * rl,
                    )
                    .subcols(qa * rl, (qb - qa) * rl);
                    let zr = MatRef::from_row_major_slice(
                        &zs[tt.sr][tt.r][..nbf * nqq * rr],
                        nbf,
                        nqq * rr,
                    )
                    .subcols(qa * rr, (qb - qa) * rr);
                    if kinds[tt.acc] == Kind::Lower {
                        tri::matmul(
                            kp[tt.acc].as_mut(),
                            BlockStructure::TriangularLower,
                            Accum::Add,
                            zl,
                            BlockStructure::Rectangular,
                            zr.transpose(),
                            BlockStructure::Rectangular,
                            tt.alpha,
                            Par::Seq,
                        );
                    } else {
                        matmul(
                            kp[tt.acc].as_mut(),
                            Accum::Add,
                            zl,
                            zr.transpose(),
                            tt.alpha,
                            Par::Seq,
                        );
                    }
                }
            });
            t_k += t.elapsed().as_secs_f64();
        }
        // ordered sum over lanes, parallel over columns
        let t = Instant::now();
        let mut acc: Vec<Vec<f64>> = Vec::with_capacity(nacc);
        for (ai, kind) in kinds.iter().enumerate() {
            let mut kv = vec![0.0; nbf * nbf];
            kv.par_chunks_mut(nbf).enumerate().for_each(|(j, c)| {
                let i0 = if *kind == Kind::Lower { j } else { 0 };
                for p in &lanes {
                    let pc = p[ai].col(j);
                    for i in i0..nbf {
                        c[i] += pc[i];
                    }
                }
            });
            acc.push(kv);
        }
        let t_sum = t.elapsed().as_secs_f64();
        let timings = vec![
            ("k_blocks".into(), t_all.elapsed().as_secs_f64()),
            ("k_sum".into(), t_sum),
            ("k_ints".into(), t_ints),
            ("k_z".into(), t_z),
            ("k_kgemm".into(), t_k),
            ("k_nq".into(), nqmax as f64),
            ("k_nblocks".into(), blocks.len() as f64),
        ];
        (acc, timings)
    }
}

/// The lower Cholesky factor of the auxiliary metric of density fitting, in its storage
/// ([`linus_base::linalg::cholesky_in_place`]; [`DfSetupError::Metric`] if it is not positive
/// definite).
pub fn metric_cholesky(a: Mat<f64>, par: Par) -> Result<Mat<f64>, DfSetupError> {
    let naux = a.nrows();
    linus_base::linalg::cholesky_in_place(a, par).map_err(|e| DfSetupError::Metric {
        pivot: e.index,
        naux,
    })
}

/// The Coulomb metric V_PQ = (P|Q) of an auxiliary basis set as its lower Cholesky factor L
/// (V = L L^T): the one place the fits build and factor it. The SCF's fitting ([`DfJk`]), the
/// gradient's ([`gradient::df_gradient`]), the Hessian's ([`hessian::df_hessian_2e`]) and the
/// response's (`response::DfResponseJk`) of one geometry and auxiliary basis set take one
/// (`linus::context::IntegralContext::metric`); the factor has the same bits at any
/// parallelism, so a shared one gives each the bits of its own.
pub struct CoulombMetric {
    l: Mat<f64>,
    /// Seconds spent on the 2-center integrals and on the whole build (integrals and factor).
    t_ints: f64,
    t_build: f64,
}

impl CoulombMetric {
    /// The metric of `aux` (placed on the atoms), factored on up to 16 threads of `par` (the
    /// blocked Cholesky gains little beyond that and degrades badly on a busy node), sequentially
    /// below 1500 functions (faster there); [`DfSetupError::Metric`] if it is not positive
    /// definite.
    pub fn new(aux: &BasisSet, par: Par) -> Result<Self, DfSetupError> {
        let t = Instant::now();
        let naux = aux.nbf;
        let v = eri3c::metric(&Aux3c::new(aux), 0.0, &Operator::Coulomb);
        let t_ints = t.elapsed().as_secs_f64();
        let vm = Mat::from_fn(naux, naux, |i, j| v[i * naux + j]);
        drop(v);
        let fpar = if naux < 1500 {
            Par::Seq
        } else {
            Par::rayon(par.degree().clamp(1, 16))
        };
        let l = metric_cholesky(vm, fpar)?;
        Ok(Self {
            l,
            t_ints,
            t_build: t.elapsed().as_secs_f64(),
        })
    }

    /// L, naux x naux (lower).
    pub fn factor(&self) -> MatRef<'_, f64> {
        self.l.as_ref()
    }

    pub fn naux(&self) -> usize {
        self.l.nrows()
    }
}

fn opts_fit(o: &DfConfig, bytes: usize, budget: usize) -> bool {
    !o.force_direct && bytes <= budget
}

/// Split a symmetric density into PSD factors: D = P P^T - N N^T (eigenvalues below
/// 1e-14 * max |w| dropped).
pub fn factor_symmetric(d: MatRef<'_, f64>, par: Par) -> (Mat<f64>, Mat<f64>) {
    let (w, u) = eigh(d, par);
    let wmax = w.iter().fold(0.0f64, |a, &x| a.max(x.abs()));
    let cut = 1e-14 * wmax.max(1e-300);
    let pos: Vec<usize> = (0..w.len()).filter(|&i| w[i] > cut).collect();
    let neg: Vec<usize> = (0..w.len()).filter(|&i| w[i] < -cut).collect();
    let n = d.nrows();
    let p = Mat::from_fn(n, pos.len(), |i, k| u[(i, pos[k])] * w[pos[k]].sqrt());
    let m = Mat::from_fn(n, neg.len(), |i, k| u[(i, neg[k])] * (-w[neg[k]]).sqrt());
    (p, m)
}

impl JkBuilder for DfJk {
    fn name(&self) -> String {
        format!(
            "ri(naux={}, rows={}, mode={})",
            self.naux,
            self.nx,
            self.stats.plan.name()
        )
    }

    /// HOOK (solvent, the density-fitted PCM): see [`CoulombHook`].
    fn set_coulomb_hook(
        &mut self,
        hook: Option<Box<dyn CoulombHook>>,
    ) -> Result<HookOutcome, linus_base::error::LinusError> {
        DfJk::set_coulomb_hook(self, hook);
        Ok(HookOutcome::Installed)
    }

    /// One pass for a K + b K_LR (see [`DfJk::exchange_combined`]); needs an exchange tensor and
    /// an operator with an erf part.
    fn set_exchange_operator(&mut self, op: Operator) -> bool {
        if !matches!(self.store, Store::B(_) | Store::Coef(_)) || !op.has_erf() || op.omega() < 0.0
        {
            return false;
        }
        self.xop = Some(op);
        true
    }

    fn compute(
        &mut self,
        req: &JkRequest,
        par: Par,
    ) -> Result<JkOutput, linus_base::error::LinusError> {
        self.stats.last.clear();
        let mut out = JkOutput::default();
        if req.want_j {
            let t = Instant::now();
            let d: Vec<MatRef<'_, f64>> = req.dens.iter().map(|d| d.as_ref()).collect();
            out.j = self.coulomb(&d, par);
            self.stats
                .last
                .push(("j".into(), t.elapsed().as_secs_f64()));
        }
        if let (true, Some(op)) = (req.want_k, self.xop) {
            // one-pass range-separated exchange (set_exchange_operator): out.k = a K + b K_LR
            let t = Instant::now();
            out.k = match req.factors.and_then(|f| f.symmetric()) {
                Some(f) => {
                    let cs: Vec<MatRef<'_, f64>> = f.iter().map(|c| c.as_ref()).collect();
                    self.exchange_combined(&cs, op, par)
                }
                None => {
                    let owned: Vec<(Mat<f64>, Mat<f64>)> = req
                        .dens
                        .iter()
                        .map(|d| factor_symmetric(d.as_ref(), par))
                        .collect();
                    let pos: Vec<MatRef<'_, f64>> = owned.iter().map(|(p, _)| p.as_ref()).collect();
                    let mut k = self.exchange_combined(&pos, op, par);
                    for (i, (_, m)) in owned.iter().enumerate() {
                        if m.ncols() > 0 {
                            let kn = self.exchange_combined(&[m.as_ref()], op, par);
                            k[i] = &k[i] - &kn[0];
                        }
                    }
                    k
                }
            };
            self.stats
                .last
                .push(("k".into(), t.elapsed().as_secs_f64()));
            return Ok(out);
        }
        if req.want_k {
            let t = Instant::now();
            let k = match req.factors.and_then(|f| f.symmetric()) {
                Some(f) => {
                    assert_eq!(f.len(), req.dens.len());
                    let fs: Vec<Factored<'_>> =
                        f.iter().map(|c| Factored::Psd(c.as_ref())).collect();
                    self.exchange(&fs, par)
                }
                None => {
                    // D = P P^T - N N^T from the eigendecomposition
                    let owned: Vec<(Mat<f64>, Mat<f64>)> = req
                        .dens
                        .iter()
                        .map(|d| factor_symmetric(d.as_ref(), par))
                        .collect();
                    self.stats
                        .last
                        .push(("k_factor".into(), t.elapsed().as_secs_f64()));
                    let fs: Vec<Factored<'_>> = owned
                        .iter()
                        .map(|(p, _)| Factored::Psd(p.as_ref()))
                        .collect();
                    let mut k = self.exchange(&fs, par);
                    let negs: Vec<(usize, Factored<'_>)> = owned
                        .iter()
                        .enumerate()
                        .filter(|(_, (_, m))| m.ncols() > 0)
                        .map(|(i, (_, m))| (i, Factored::Psd(m.as_ref())))
                        .collect();
                    if !negs.is_empty() {
                        let fs: Vec<Factored<'_>> = negs.iter().map(|x| x.1).collect();
                        let kn = self.exchange(&fs, par);
                        for (j, (i, _)) in negs.iter().enumerate() {
                            k[*i] = &k[*i] - &kn[j];
                        }
                    }
                    k
                }
            };
            out.k = k;
            self.stats
                .last
                .push(("k".into(), t.elapsed().as_secs_f64()));
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::df::reference::DenseDfJk;
    use linus_base::linalg::{max_abs, mm};
    use linus_base::terms::DensityFactors;
    use linus_ints::basis::BasisDef;
    use linus_ints::PairOptions;

    /// Water (bohr), slightly asymmetric, and a closed-shell-like factor C (5 columns).
    fn setup(basis: &str) -> (BasisSet, BasisSet, PairSet, Mat<f64>) {
        let z = vec![8, 1, 1];
        let c = vec![
            [0.0, 0.0, 0.2214],
            [0.0, 1.4309, -0.8857],
            [0.3, -1.4309, -0.8857],
        ];
        let b = BasisSet::new(&BasisDef::load(basis, None).unwrap(), &z, &c, true).unwrap();
        let a = BasisSet::new(
            &BasisDef::load("def2-universal-jkfit", None).unwrap(),
            &z,
            &c,
            true,
        )
        .unwrap();
        let pl = PairSet::build(&b, PairOptions::default());
        let n = b.nbf;
        let cm = Mat::from_fn(n, 5, |i, j| {
            ((i * 7 + j * 13) % 11) as f64 * 0.05 - 0.2 + if i == j { 0.5 } else { 0.0 }
        });
        (b, a, pl, cm)
    }

    fn opts() -> DfConfig {
        DfConfig {
            mem_budget: Some(1 << 30),
            ..Default::default()
        }
    }

    /// The exchange workspace the storage plan counts ([`KSizes::scf_exchange_bytes`]) is what
    /// the builds keep and allocate: the one-sided coefficients' block scratch plus the lanes'
    /// and summed K matrices, the in-memory exchange's panel scratch plus every group's partial
    /// K and the summed ones; and a room too small for the stored B and B^omega with their
    /// workspace takes the one-sided coefficients, with no room at all the error lists both.
    #[test]
    fn exchange_workspace_is_what_the_builds_take() {
        let (b, a, pl, c) = setup("def2-svp");
        let occ = [c.ncols(), c.ncols()];
        let s = 8 * b.nbf * b.nbf;
        let op = Operator::Attenuated {
            coulomb: 0.2,
            long_range: 0.8,
            omega: 0.3,
        };
        for long_range in [DfLongRange::OneSided, DfLongRange::Stored] {
            let o = DfConfig {
                omega: Some(0.3),
                long_range,
                scf_occ: occ,
                ..opts()
            };
            // one thread: one running group, one panel scratch in the pool
            let pool1 = rayon::ThreadPoolBuilder::new()
                .num_threads(1)
                .build()
                .unwrap();
            let mut df = pool1.install(|| DfJk::new(&b, &pl, &a, true, o, Par::Seq));
            let want = df
                .k_sizes()
                .scf_exchange_bytes(df.stats.plan, true, &occ, 1);
            let cs = [c.as_ref(), c.as_ref()];
            pool1.install(|| df.exchange_combined(&cs, op, Par::Seq));
            let kept = match long_range {
                DfLongRange::OneSided => {
                    let sc = &df.kb_scratch;
                    8 * (sc.zs.iter().flatten().map(Vec::len).sum::<usize>()
                        + sc.fly.iter().map(Vec::len).sum::<usize>())
                        + (NLANE + 1) * 2 * s
                }
                _ => {
                    let g = df.k_sizes().groups(&occ, 2, 2);
                    let pool = df.k_panels.lock().unwrap();
                    let panels: usize = pool
                        .iter()
                        .map(|p| {
                            8 * (p.zs.iter().flatten().map(Vec::len).sum::<usize>()
                                + p.tb.len()
                                + p.cbuf.len())
                        })
                        .sum();
                    panels + (g.ngroups + 1) * 2 * s
                }
            };
            assert_eq!(want, kept, "{long_range:?}");
        }
        // the room: B and B^omega with their workspace do not fit, the coefficients do
        let ws = |p: DfPlan| {
            if matches!(p, DfPlan::OneSided { .. }) {
                1
            } else {
                2
            }
        };
        let t = 1usize << 20;
        let rsh = DfConfig {
            omega: Some(0.3),
            room: Some(t + 1),
            ..opts()
        };
        assert!(matches!(
            DfJk::plan(&rsh, t, 4 * t, ws).unwrap(),
            DfPlan::OneSided { .. }
        ));
        let none = DfConfig {
            room: Some(t),
            ..rsh
        };
        match DfJk::plan(&none, t, 4 * t, ws).unwrap_err() {
            DfSetupError::Memory { options, .. } => assert_eq!(options, [(2 * t, 2), (t, 1)]),
            e => panic!("{e:?}"),
        }
    }

    #[test]
    fn matches_reference_dfjk() {
        for basis in ["def2-svp", "def2-tzvp"] {
            let (b, a, pl, c) = setup(basis);
            let d = mm(c.as_ref(), c.transpose(), Par::Seq);
            let mut old = DenseDfJk::new(&b, &pl, &a, Par::Seq);
            let mut df = DfJk::new(&b, &pl, &a, true, opts(), Par::Seq);
            let fs = [c.clone()];
            let r0 = old
                .compute(
                    &JkRequest {
                        dens: std::slice::from_ref(&d),
                        factors: Some(DensityFactors::Symmetric(&fs)),
                        want_j: true,
                        want_k: true,
                    },
                    Par::Seq,
                )
                .unwrap();
            let r1 = df
                .compute(
                    &JkRequest {
                        dens: std::slice::from_ref(&d),
                        factors: Some(DensityFactors::Symmetric(&fs)),
                        want_j: true,
                        want_k: true,
                    },
                    Par::Seq,
                )
                .unwrap();
            let r2 = df
                .compute(
                    &JkRequest::new(std::slice::from_ref(&d), false, true),
                    Par::Seq,
                )
                .unwrap();
            assert!(max_abs((&r1.j[0] - &r0.j[0]).as_ref()) < 1e-12, "{basis} J");
            assert!(max_abs((&r1.k[0] - &r0.k[0]).as_ref()) < 1e-12, "{basis} K");
            assert!(
                max_abs((&r2.k[0] - &r0.k[0]).as_ref()) < 1e-12,
                "{basis} K from the eigen-factors"
            );
        }
    }

    /// HOOK (solvent, the density-fitted PCM): in every storage mode the hook sees the fit
    /// coefficients V^-1 gamma and its extra coefficients e add sum_P (mn|P) e_P to J.
    #[test]
    #[allow(clippy::type_complexity, reason = "a one-off tuple of buffers")]
    fn coulomb_hook_in_every_storage_mode() {
        use std::sync::{Arc, Mutex};
        struct Rec(Arc<Mutex<Vec<Vec<f64>>>>, Vec<f64>);
        impl CoulombHook for Rec {
            fn extra_coefficients(
                &mut self,
                c: &[Vec<f64>],
                _solve: &dyn Fn(&mut [f64]),
            ) -> Vec<Vec<f64>> {
                *self.0.lock().unwrap() = c.to_vec();
                vec![self.1.clone(); c.len()]
            }
        }
        let (b, a, pl, c) = setup("def2-svp");
        let d = mm(c.as_ref(), c.transpose(), Par::Seq);
        let d2 = &d * 0.5;
        let dr = [d.as_ref(), d2.as_ref()];
        let naux = a.nbf;
        let e: Vec<f64> = (0..naux)
            .map(|p| ((p * 13 % 7) as f64 - 3.0) * 0.01)
            .collect();
        let run = |mut df: DfJk, tol: f64, reference: Option<&(Vec<Vec<f64>>, Vec<Mat<f64>>)>| {
            let j0 = df.coulomb(&dr, Par::Seq);
            let seen = Arc::new(Mutex::new(vec![]));
            df.set_coulomb_hook(Some(Box::new(Rec(seen.clone(), e.clone()))));
            let j1 = df.coulomb(&dr, Par::Seq);
            df.set_coulomb_hook(None);
            let dj: Vec<Mat<f64>> = j1.iter().zip(&j0).map(|(x, y)| x - y).collect();
            let cs = seen.lock().unwrap().clone();
            assert_eq!(cs.len(), 2);
            if let Some((cr, djr)) = reference {
                for k in 0..2 {
                    // the fit coefficients carry the metric's conditioning: relative check
                    let ec = cs[k]
                        .iter()
                        .zip(&cr[k])
                        .map(|(x, y)| (x - y).abs())
                        .fold(0.0, f64::max);
                    let cm = cr[k].iter().fold(0.0f64, |m, x| m.max(x.abs()));
                    assert!(
                        ec < 1e-9 * cm.max(1.0),
                        "{}: coefficients {ec:e} (max {cm:e})",
                        df.stats.plan.name()
                    );
                    assert!(
                        max_abs((&dj[k] - &djr[k]).as_ref()) < tol,
                        "{}: extra J",
                        df.stats.plan.name()
                    );
                }
            }
            (cs, dj)
        };
        let reference = run(DfJk::new(&b, &pl, &a, false, opts(), Par::Seq), 0.0, None);
        assert!(reference.1.iter().all(|m| max_abs(m.as_ref()) > 1e-3));
        run(
            DfJk::new(&b, &pl, &a, true, opts(), Par::Seq),
            1e-11,
            Some(&reference),
        );
        run(
            DfJk::new(
                &b,
                &pl,
                &a,
                false,
                DfConfig {
                    force_direct: true,
                    ..opts()
                },
                Par::Seq,
            ),
            1e-11,
            Some(&reference),
        );
        let one = DfConfig {
            omega: Some(0.3),
            long_range: DfLongRange::OneSided,
            ..opts()
        };
        let df = DfJk::new(&b, &pl, &a, true, one, Par::Seq);
        assert!(matches!(df.stats.plan, DfPlan::OneSided { .. }));
        run(df, 1e-10, Some(&reference));
        let df = DfJk::new(
            &b,
            &pl,
            &a,
            true,
            DfConfig {
                precision: DfPrecision::F48,
                ..opts()
            },
            Par::Seq,
        );
        run(df, 1e-9, Some(&reference));
    }

    #[test]
    fn coulomb_storage_modes_agree() {
        let (b, a, pl, c) = setup("def2-svp");
        let d = mm(c.as_ref(), c.transpose(), Par::Seq);
        let dr = [d.as_ref()];
        let jb = DfJk::new(&b, &pl, &a, true, opts(), Par::Seq).coulomb(&dr, Par::Seq);
        let ja = DfJk::new(&b, &pl, &a, false, opts(), Par::Seq).coulomb(&dr, Par::Seq);
        let jd = DfJk::new(
            &b,
            &pl,
            &a,
            false,
            DfConfig {
                force_direct: true,
                ..opts()
            },
            Par::Seq,
        )
        .coulomb(&dr, Par::Seq);
        assert!(max_abs((&jb[0] - &ja[0]).as_ref()) < 1e-12);
        assert!(max_abs((&jd[0] - &ja[0]).as_ref()) < 1e-12);
    }

    #[test]
    fn factored_forms_and_long_range() {
        let (b, a, pl, c) = setup("def2-svp");
        let mut df = DfJk::new(&b, &pl, &a, true, opts(), Par::Seq);
        // D = X Y^T + Y X^T (indefinite) against the eigen-factor path
        let x = c.subcols(0, 2).to_owned();
        let y = c.subcols(2, 2).to_owned();
        let dp = mm(x.as_ref(), y.transpose(), Par::Seq) + mm(y.as_ref(), x.transpose(), Par::Seq);
        let ks = df.exchange(&[Factored::SymPair(x.as_ref(), y.as_ref())], Par::Seq);
        let ke = df
            .compute(
                &JkRequest::new(std::slice::from_ref(&dp), false, true),
                Par::Seq,
            )
            .unwrap();
        assert!(max_abs((&ks[0] - &ke.k[0]).as_ref()) < 1e-12);
        // non-symmetric D = X Y^T: K[XY^T] + K[YX^T] = K[XY^T + YX^T]
        let kxy = df.exchange(
            &[
                Factored::Pair(x.as_ref(), y.as_ref()),
                Factored::Pair(y.as_ref(), x.as_ref()),
            ],
            Par::Seq,
        );
        assert!(max_abs((&(&kxy[0] + &kxy[1]) - &ks[0]).as_ref()) < 1e-12);
        // long range: erf(w r)/r -> 1/r for large w; K_LR symmetric; erf(w r)/r -> 2w/sqrt(pi)
        // for small w, so K_LR vanishes linearly
        let k_and_lr = |df: &mut DfJk, w: f64| {
            df.exchange_long_range(&[Factored::Psd(c.as_ref())], true, Some(w), Par::Seq)
        };
        let (k, klr) = k_and_lr(&mut df, 1e4);
        assert!(max_abs((&klr[0] - &k[0]).as_ref()) < 1e-6 * max_abs(k[0].as_ref()));
        assert_eq!(max_abs((&klr[0] - klr[0].transpose()).as_ref()), 0.0);
        let k1 = k_and_lr(&mut df, 1e-4).1.remove(0);
        let k2 = k_and_lr(&mut df, 2e-4).1.remove(0);
        assert!(max_abs(k1.as_ref()) < 1e-3);
        assert!(max_abs((&k2 - &k1 * 2.0).as_ref()) < 1e-6 * max_abs(k1.as_ref()));
        // SymPair long range against PSD factors: X Y^T + Y X^T = (P P^T - M M^T) / 2 with
        // P = X + Y, M = X - Y
        let (_, kl) = df.exchange_long_range(
            &[Factored::SymPair(x.as_ref(), y.as_ref())],
            false,
            Some(0.3),
            Par::Seq,
        );
        let (pp, mm_) = (&x + &y, &x - &y);
        let (_, kpm) = df.exchange_long_range(
            &[Factored::Psd(pp.as_ref()), Factored::Psd(mm_.as_ref())],
            false,
            Some(0.3),
            Par::Seq,
        );
        assert!(max_abs((&kl[0] - &(&kpm[0] - &kpm[1]) * 0.5).as_ref()) < 1e-12);
    }

    /// One-pass range-separated exchange equals a K + b K_LR (factors and densities).
    #[test]
    fn combined_exchange_operator() {
        let (b, a, pl, c) = setup("def2-svp");
        let mut df = DfJk::new(
            &b,
            &pl,
            &a,
            true,
            DfConfig {
                omega: Some(0.3),
                ..opts()
            },
            Par::Seq,
        );
        let d = mm(c.as_ref(), c.transpose(), Par::Seq);
        let fs = [c.clone()];
        let sep = df
            .compute(
                &JkRequest {
                    dens: std::slice::from_ref(&d),
                    factors: Some(DensityFactors::Symmetric(&fs)),
                    want_j: true,
                    want_k: false,
                },
                Par::Seq,
            )
            .unwrap();
        let (k, klr) =
            df.exchange_long_range(&[Factored::Psd(c.as_ref())], true, Some(0.3), Par::Seq);
        let (wa, wb) = (0.157706, 0.842294);
        let want = &k[0] * wa + &klr[0] * wb;
        assert!(df.set_exchange_operator(Operator::Attenuated {
            coulomb: wa,
            long_range: wb,
            omega: 0.3
        }));
        let one = df
            .compute(
                &JkRequest {
                    dens: std::slice::from_ref(&d),
                    factors: Some(DensityFactors::Symmetric(&fs)),
                    want_j: true,
                    want_k: true,
                },
                Par::Seq,
            )
            .unwrap();
        assert!(max_abs((&one.j[0] - &sep.j[0]).as_ref()) == 0.0);
        assert!(
            max_abs((&one.k[0] - &want).as_ref()) < 1e-13 * max_abs(want.as_ref()).max(1.0),
            "{}",
            max_abs((&one.k[0] - &want).as_ref())
        );
        // density without factors, indefinite (difference densities of incremental builds)
        let x = c.subcols(0, 2).to_owned();
        let dd = &d - mm(x.as_ref(), x.transpose(), Par::Seq) * 1.5;
        let one = df
            .compute(
                &JkRequest::new(std::slice::from_ref(&dd), false, true),
                Par::Seq,
            )
            .unwrap();
        let (k, klr) = df.exchange_long_range(
            &[Factored::Psd(c.as_ref()), Factored::Psd(x.as_ref())],
            true,
            Some(0.3),
            Par::Seq,
        );
        let want = (&k[0] - &k[1] * 1.5) * wa + (&klr[0] - &klr[1] * 1.5) * wb;
        assert!(max_abs((&one.k[0] - &want).as_ref()) < 1e-12 * max_abs(want.as_ref()).max(1.0));
    }

    #[test]
    fn independent_of_thread_count() {
        let (b, a, pl, c) = setup("def2-svp");
        let d = mm(c.as_ref(), c.transpose(), Par::Seq);
        let fs = [c.clone()];
        // in-memory B and B^omega; one-sided f64 and f32 (block-sequential driver, raw
        // integrals per block); B in f32 (block-sequential, stored only)
        let modes = [
            (DfPrecision::F64, DfLongRange::Stored, true),
            (DfPrecision::F64, DfLongRange::OneSided, true),
            (DfPrecision::F48, DfLongRange::OneSided, true),
            (DfPrecision::F32, DfLongRange::OneSided, true),
            (DfPrecision::F40, DfLongRange::Auto, false),
            (DfPrecision::F32, DfLongRange::Stored, true),
        ];
        for (prec, lrm, rsh) in modes {
            let run = |nt: usize| {
                let pool = rayon::ThreadPoolBuilder::new()
                    .num_threads(nt)
                    .build()
                    .unwrap();
                pool.install(|| {
                    let par = if nt > 1 { Par::rayon(nt) } else { Par::Seq };
                    let o = DfConfig {
                        precision: prec,
                        long_range: lrm,
                        omega: rsh.then_some(0.3),
                        k_block: 24,
                        ..opts()
                    };
                    let mut df = DfJk::new(&b, &pl, &a, true, o, par);
                    let mut r = df
                        .compute(
                            &JkRequest {
                                dens: std::slice::from_ref(&d),
                                factors: Some(DensityFactors::Symmetric(&fs)),
                                want_j: true,
                                want_k: true,
                            },
                            par,
                        )
                        .unwrap();
                    if rsh {
                        r.k.extend(
                            df.exchange_long_range(
                                &[Factored::Psd(c.as_ref())],
                                false,
                                Some(0.3),
                                par,
                            )
                            .1,
                        );
                        assert!(df.set_exchange_operator(Operator::Attenuated {
                            coulomb: 0.157706,
                            long_range: 0.842294,
                            omega: 0.3
                        }));
                        let o = df
                            .compute(
                                &JkRequest {
                                    dens: std::slice::from_ref(&d),
                                    factors: Some(DensityFactors::Symmetric(&fs)),
                                    want_j: false,
                                    want_k: true,
                                },
                                par,
                            )
                            .unwrap();
                        r.k.extend(o.k);
                    }
                    r
                })
            };
            let (r1, r3) = (run(1), run(3));
            assert_eq!(
                max_abs((&r1.j[0] - &r3.j[0]).as_ref()),
                0.0,
                "{prec:?} {lrm:?} J"
            );
            for (x, y) in r1.k.iter().zip(&r3.k) {
                assert_eq!(max_abs((x - y).as_ref()), 0.0, "{prec:?} {lrm:?} K");
            }
        }
    }

    fn rel(x: &Mat<f64>, y: &Mat<f64>) -> f64 {
        max_abs((x - y).as_ref()) / max_abs(y.as_ref())
    }

    /// One-sided storage (C = (mn|P) V^-1, raw integrals of the other side per block) equals
    /// stored B and B^omega up to round-off: J, K, K_LR, the one-pass combination, the
    /// eigen-factor path and the SymPair / Pair factor forms.
    #[test]
    fn one_sided_matches_stored() {
        let w = 0.3;
        for basis in ["def2-svp", "def2-tzvp"] {
            let (b, a, pl, c) = setup(basis);
            let d = mm(c.as_ref(), c.transpose(), Par::Seq);
            let fs = [c.clone()];
            let mk = |lrm: DfLongRange| {
                DfJk::new(
                    &b,
                    &pl,
                    &a,
                    true,
                    DfConfig {
                        precision: DfPrecision::F64,
                        long_range: lrm,
                        omega: Some(w),
                        ..opts()
                    },
                    Par::Seq,
                )
            };
            let (mut st, mut os) = (mk(DfLongRange::Stored), mk(DfLongRange::OneSided));
            assert_eq!(
                (st.stats.plan.name().as_str(), os.stats.plan.name().as_str()),
                ("B", "coef")
            );
            let req = JkRequest {
                dens: std::slice::from_ref(&d),
                factors: Some(DensityFactors::Symmetric(&fs)),
                want_j: true,
                want_k: true,
            };
            let (r0, r1) = (
                st.compute(&req, Par::Seq).unwrap(),
                os.compute(&req, Par::Seq).unwrap(),
            );
            assert!(
                rel(&r1.j[0], &r0.j[0]) < 1e-13,
                "{basis} J {:e}",
                rel(&r1.j[0], &r0.j[0])
            );
            assert!(
                rel(&r1.k[0], &r0.k[0]) < 1e-12,
                "{basis} K {:e}",
                rel(&r1.k[0], &r0.k[0])
            );
            let lr = |df: &mut DfJk| {
                df.exchange_long_range(&[Factored::Psd(c.as_ref())], false, Some(w), Par::Seq)
                    .1
                    .remove(0)
            };
            let (l0, l1) = (lr(&mut st), lr(&mut os));
            assert!(rel(&l1, &l0) < 1e-12, "{basis} K_LR {:e}", rel(&l1, &l0));
            // indefinite density without factors (eigen-factor path), K and (operator erf) K_LR
            let x = c.subcols(0, 2).to_owned();
            let y = c.subcols(2, 2).to_owned();
            let dp =
                mm(x.as_ref(), y.transpose(), Par::Seq) + mm(y.as_ref(), x.transpose(), Par::Seq);
            let req = JkRequest::new(std::slice::from_ref(&dp), false, true);
            let (e0, e1) = (
                st.compute(&req, Par::Seq).unwrap(),
                os.compute(&req, Par::Seq).unwrap(),
            );
            assert!(
                st.set_exchange_operator(Operator::LongRange { omega: w })
                    && os.set_exchange_operator(Operator::LongRange { omega: w })
            );
            let (el0, el1) = (
                st.compute(&req, Par::Seq).unwrap(),
                os.compute(&req, Par::Seq).unwrap(),
            );
            (st.xop, os.xop) = (None, None);
            assert!(
                rel(&e1.k[0], &e0.k[0]) < 1e-12 && rel(&el1.k[0], &el0.k[0]) < 1e-12,
                "{basis} eigen path"
            );
            // SymPair and Pair factors
            let f = [
                Factored::SymPair(x.as_ref(), y.as_ref()),
                Factored::Pair(x.as_ref(), y.as_ref()),
            ];
            let (s0, s1) = (
                st.exchange_long_range(&f, true, Some(w), Par::Seq),
                os.exchange_long_range(&f, true, Some(w), Par::Seq),
            );
            for (u, v) in s1.0.iter().zip(&s0.0).chain(s1.1.iter().zip(&s0.1)) {
                assert!(rel(u, v) < 1e-12, "{basis} factor forms {:e}", rel(u, v));
            }
            // one pass a K + b K_LR
            let (wa, wb) = (0.157706, 0.842294);
            let op = Operator::Attenuated {
                coulomb: wa,
                long_range: wb,
                omega: w,
            };
            assert!(st.set_exchange_operator(op) && os.set_exchange_operator(op));
            let req = JkRequest {
                dens: std::slice::from_ref(&d),
                factors: Some(DensityFactors::Symmetric(&fs)),
                want_j: false,
                want_k: true,
            };
            let (c0, c1) = (
                st.compute(&req, Par::Seq).unwrap(),
                os.compute(&req, Par::Seq).unwrap(),
            );
            assert!(
                rel(&c1.k[0], &c0.k[0]) < 1e-12,
                "{basis} combined {:e}",
                rel(&c1.k[0], &c0.k[0])
            );
            let want = &r0.k[0] * wa + &l0 * wb;
            assert!(rel(&c1.k[0], &want) < 1e-12);
        }
    }

    /// Compressed storage: J exact (F48 from the tensor, F40/F32 integral-direct); K and K_LR
    /// carry relative errors of the formats' rounding only (F48 ~1e-12, F40 ~1e-10, F32 ~1e-7).
    #[test]
    fn compressed_storage() {
        let w = 0.3;
        let (b, a, pl, c) = setup("def2-tzvp");
        let d = mm(c.as_ref(), c.transpose(), Par::Seq);
        let fs = [c.clone()];
        let req = JkRequest {
            dens: std::slice::from_ref(&d),
            factors: Some(DensityFactors::Symmetric(&fs)),
            want_j: true,
            want_k: true,
        };
        let lr = |df: &mut DfJk| {
            df.exchange_long_range(&[Factored::Psd(c.as_ref())], false, Some(w), Par::Seq)
                .1
                .remove(0)
        };
        let mut st = DfJk::new(
            &b,
            &pl,
            &a,
            true,
            DfConfig {
                precision: DfPrecision::F64,
                long_range: DfLongRange::Stored,
                omega: Some(w),
                ..opts()
            },
            Par::Seq,
        );
        let (r0, l0) = (st.compute(&req, Par::Seq).unwrap(), lr(&mut st));
        let cases = [
            (
                DfPrecision::F48,
                DfLongRange::OneSided,
                "coef48",
                1e-11,
                1e-13,
            ),
            (
                DfPrecision::F40,
                DfLongRange::OneSided,
                "coef40",
                1e-9,
                1e-13,
            ),
            (
                DfPrecision::F32,
                DfLongRange::OneSided,
                "coef32",
                1e-6,
                1e-13,
            ),
            (DfPrecision::F48, DfLongRange::Stored, "B48", 1e-11, 1e-12),
            (DfPrecision::F32, DfLongRange::Stored, "B32", 1e-6, 1e-13),
        ];
        for (prec, lrm, mode, tol_k, tol_j) in cases {
            let mut df = DfJk::new(
                &b,
                &pl,
                &a,
                true,
                DfConfig {
                    precision: prec,
                    long_range: lrm,
                    omega: Some(w),
                    ..opts()
                },
                Par::Seq,
            );
            assert_eq!(df.stats.plan.name(), mode);
            let (r1, l1) = (df.compute(&req, Par::Seq).unwrap(), lr(&mut df));
            assert!(
                rel(&r1.j[0], &r0.j[0]) < tol_j,
                "{mode} J {:e}",
                rel(&r1.j[0], &r0.j[0])
            );
            let (ek, el) = (rel(&r1.k[0], &r0.k[0]), rel(&l1, &l0));
            assert!(
                ek < tol_k && el < tol_k && ek > 0.0,
                "{mode} K {ek:e} K_LR {el:e}"
            );
        }
        // residual encoding round trip
        for &x in &[1.0, -0.3, 1e-20, 7.123456789e5, -2.0f64.powi(-30) * 1.999] {
            let (h, q) = encode::<i16>(x);
            assert!((decode(h, q) - x).abs() <= x.abs() * 1e-12, "{x}");
            let (h, q) = encode::<i8>(x);
            assert!((decode(h, q) - x).abs() <= x.abs() * 3e-10, "{x}");
        }
    }

    /// Storage plans by budget: in-memory f64 first, then one-sided (range separation); auto
    /// never compresses (a budget too small for f64 is an error), the compressed formats run
    /// when set.
    #[test]
    fn storage_plan_by_budget() {
        let o = |w: Option<f64>| DfConfig {
            omega: w,
            ..Default::default()
        };
        let gb = 1usize << 30;
        let b = |format: DfFormat, long_range: bool| DfPlan::InCore { format, long_range };
        let c = |format: DfFormat| DfPlan::OneSided { format };
        // 10 GiB per f64 tensor
        let t = 10 * gb;
        assert_eq!(
            DfJk::plan(&o(None), t, 12 * gb, |_| 0).unwrap(),
            b(DfFormat::F64, false)
        );
        // auto: f64 or nothing (no compressed tensor for a smaller budget)
        assert!(DfJk::plan(&o(None), t, 8 * gb, |_| 0).is_err());
        let set = |precision: DfPrecision, w: Option<f64>| DfConfig { precision, ..o(w) };
        assert_eq!(
            DfJk::plan(&set(DfPrecision::F48, None), t, 8 * gb, |_| 0).unwrap(),
            b(DfFormat::F48, false)
        );
        assert_eq!(
            DfJk::plan(&set(DfPrecision::F40, None), t, 7 * gb, |_| 0).unwrap(),
            b(DfFormat::F40, false)
        );
        assert!(DfJk::plan(&set(DfPrecision::F48, None), t, 6 * gb, |_| 0).is_err());
        let f32n = DfConfig {
            precision: DfPrecision::F32,
            ..o(None)
        };
        assert_eq!(
            DfJk::plan(&f32n, t, 6 * gb, |_| 0).unwrap(),
            b(DfFormat::F32, false)
        );
        assert_eq!(
            DfJk::plan(&o(Some(0.3)), t, 20 * gb, |_| 0).unwrap(),
            b(DfFormat::F64, true)
        );
        assert_eq!(
            DfJk::plan(&o(Some(0.3)), t, 15 * gb, |_| 0).unwrap(),
            c(DfFormat::F64)
        );
        // auto: the one-sided f64 coefficients or nothing
        assert!(DfJk::plan(&o(Some(0.3)), t, 8 * gb, |_| 0).is_err());
        assert_eq!(
            DfJk::plan(&set(DfPrecision::F48, Some(0.3)), t, 8 * gb, |_| 0).unwrap(),
            c(DfFormat::F48)
        );
        assert_eq!(
            DfJk::plan(&set(DfPrecision::F40, Some(0.3)), t, 7 * gb, |_| 0).unwrap(),
            c(DfFormat::F40)
        );
        assert!(DfJk::plan(&set(DfPrecision::F40, Some(0.3)), t, 6 * gb, |_| 0).is_err());
        let f32 = DfConfig {
            precision: DfPrecision::F32,
            ..o(Some(0.3))
        };
        assert_eq!(
            DfJk::plan(&f32, t, 6 * gb, |_| 0).unwrap(),
            c(DfFormat::F32)
        );
        assert!(DfJk::plan(&f32, t, 4 * gb, |_| 0).is_err());
        let strict = DfConfig {
            precision: DfPrecision::F64,
            ..o(Some(0.3))
        };
        assert!(DfJk::plan(&strict, t, 8 * gb, |_| 0).is_err());
        let stored = DfConfig {
            long_range: DfLongRange::Stored,
            ..o(Some(0.3))
        };
        assert_eq!(
            DfJk::plan(&stored, t, 20 * gb, |_| 0).unwrap(),
            b(DfFormat::F64, true)
        );
        // stored B and B^omega that do not fit in f64: an error under auto, compressed when set
        assert!(DfJk::plan(&stored, t, 15 * gb, |_| 0).is_err());
        let stored48 = DfConfig {
            precision: DfPrecision::F48,
            ..stored.clone()
        };
        assert_eq!(
            DfJk::plan(&stored48, t, 15 * gb, |_| 0).unwrap(),
            b(DfFormat::F48, true)
        );
    }

    /// Setup failures keep their kind: a budget too small is a resource error, a metric that is
    /// not positive definite a numerical one; `resolved.record.df` keeps its keys.
    #[test]
    fn setup_errors_and_resolved() {
        use linus_base::error::LinusError;
        let gb = 1usize << 30;
        // auto: the f64 tensor is what it needs at least
        let e = DfJk::plan(&opts(), 8 * gb, gb, |_| 0).unwrap_err();
        assert_eq!(
            e,
            DfSetupError::Memory {
                need: 8 * gb,
                per_tensor: 8 * gb,
                budget: gb,
                options: vec![(8 * gb, 0)],
            }
        );
        let le = LinusError::from(e);
        assert_eq!((le.kind(), le.code()), ("resource", Some("host_memory")));
        let singular = Mat::from_fn(3, 3, |i, j| {
            if i.max(j) < 2 {
                1.0
            } else {
                (i == j) as u8 as f64
            }
        });
        let e = metric_cholesky(singular, Par::Seq).unwrap_err();
        assert!(
            matches!(e, DfSetupError::Metric { pivot: 1, naux: 3 }),
            "{e:?}"
        );
        assert_eq!(LinusError::from(e).kind(), "numerical");
        let (b, a, pl, _) = setup("sto-3g");
        let df = DfJk::new(&b, &pl, &a, true, opts(), Par::Seq);
        let v = serde_json::to_value(df.resolved()).unwrap();
        let keys: Vec<&str> = v.as_object().unwrap().keys().map(|k| k.as_str()).collect();
        assert_eq!(keys.len(), 6);
        for k in [
            "storage",
            "precision",
            "long_range",
            "store_gib",
            "budget_gib",
            "naux",
        ] {
            assert!(keys.contains(&k), "{k}");
        }
        assert_eq!(
            (
                v["storage"].as_str(),
                v["precision"].as_str(),
                v["long_range"].as_str()
            ),
            (Some("B"), Some("f64"), Some("none"))
        );
    }
}
