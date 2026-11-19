//! Integral-direct four-center J and K ([`JkPlan`]): the J/K digester over the batched
//! four-center engine ([`crate::engine::block`]).
//!
//! J_ab = sum_cd (ab|cd) D_cd and K_ac = sum_bd (ab|cd) D_bd for symmetric densities D, with
//! J of the Coulomb operator and K of any [`Operator`] (a/r + b erf(omega r)/r).
//!
//! Work decomposition: one task per retained bra shell pair i (pairs sorted by class,
//! primitive count and Schwarz factor, see [`crate::eri4c::batch`]); the task visits the ket
//! pairs j <= i in blocks of 8 pairs of one class (one SIMD lane per ket pair), so every
//! symmetry-unique shell quartet is computed once (8-fold permutational symmetry). The bra
//! side has the higher angular momentum class, which is the cheap orientation for the
//! E^T R E contraction (SHARK Scheme 1).
//!
//! Screening (PySCF's semantics, so equal thresholds give equal accuracy): a lane is kept
//! when Q_i Q_j >= thresh and Q_i Q_j max(4|D_ij|, 4|D_kl|, |D_ik|, |D_il|, |D_jk|,
//! |D_jl|) >= thresh over the shell-pair density blocks a quartet contributes through (only the J or
//! K ones when only J or K is requested). Pass difference densities for incremental Fock
//! builds and the screening tightens automatically. Ket pairs are sorted by Q within a
//! group, so a failed bound with the group's density maximum ends the group.
//!
//! Digestion (lane-vectorized, scratch in local fixed-size arrays for shells up to f so the
//! unrolled loops keep partial sums in registers): J_ab += (ab|cd) D_cd accumulates in
//! registers over the whole task, with D_cd loaded from a per-build lane-interleaved copy
//! (one vector load per element, no gather). J is kept in fixed point per ket block and lane
//! (it lives on retained shell pairs only), so J_cd of a block is one lane-wise 128-bit vector
//! add per element and J_ab of a task goes to its own pair's (block, lane). The four exchange
//! contributions K_ac, K_ad, K_bc, K_bd land in a task-local column-major panel (per AO
//! column: the rows of shells a then b, padded to 1, 2, 4 or a multiple of 8), so each lane's
//! values for one ket column are contiguous and are added with one unmasked vector after a
//! lane transpose; the panel is flushed once per task. The densities of those contractions
//! come from a task-local panel of the density rows of a and b in the same layout (filled per
//! ket shell on first use), read per lane and column as one short vector and transposed. Lane
//! addresses (AO offsets, density and accumulator positions, touched shells) are precomputed
//! per ket block, and lanes are screened with vector compares. Lanes that fail screening keep
//! weight 0 and add exact zeros, so no lane loop branches.
//!
//! Determinism: tasks are fixed (independent of the thread count) and computed in a fixed
//! order internally; their contributions are added to per-thread accumulators in 128-bit
//! fixed point (2^-64 resolution), where addition is exact and associative. The result is
//! therefore bitwise identical for any number of threads and any scheduling
//! ([`crate::engine::Reduction::Exact`]).
//!
//! Memory: the plan holds the batched layout; a caller-owned [`JkWorkspace`], made for a
//! request shape and a thread count, holds each thread's accumulators and scratch and the
//! per-build copies of the densities, and is reused build after build ([`JkPlan::workspace_size`]
//! reports it before it is made).

mod digest;

use crate::basis::BasisSet;
use crate::contract::{AoMatrix, AoMatrixMut};
use crate::engine::{schedule, WorkspaceSize};
use crate::eri4c::batch::BatchedPairs;
use crate::error::IntsError;
use crate::operator::Operator;
use crate::pairs::PairSet;
use digest::{tsc, Ctx, JAcc, Slot, WorkerSizes};
use linus_num::det::Fixed128;
use linus_num::simd::{Isa, LANES};

/// Primitive-quartet screening: a bra/ket primitive pair is skipped inside a block when
/// Qp_ip * max_lane Qp_kq * max|D| < prim_factor * thresh, and primitive pairs with
/// Qp * Qmax < prim_factor * thresh are dropped at setup (SHARK's T_cut = 0.01 T_int).
/// Tighter factors changed neither the J/K errors nor the speed measurably (caffeine/def2-SVP,
/// thresh 1e-12: max J error 5.30e-11 for factors 1e-2, 1e-3, 1e-4 and 0).
pub const PRIM_FACTOR: f64 = 1e-2;

/// Quartet screening follows PySCF's `CVHFnrs8_prescreen`, so that equal thresholds give
/// equal accuracy: a quartet is kept when Q_ij Q_kl >= thresh and
/// Q_ij Q_kl max(4 |D_ij|, 4 |D_kl|, |D_ik|, |D_il|, |D_jk|, |D_jl|) >= thresh
/// (with 8-fold symmetry each integral enters J up to four times).
const J_WEIGHT: f64 = 4.0;

const LS: [char; 7] = ['s', 'p', 'd', 'f', 'g', 'h', 'i'];

/// The rdtsc frequency in GHz (profiles).
fn tsc_ghz() -> f64 {
    use std::sync::OnceLock;
    static G: OnceLock<f64> = OnceLock::new();
    *G.get_or_init(|| {
        let (c0, t0) = (tsc(), std::time::Instant::now());
        while t0.elapsed().as_secs_f64() < 0.05 {}
        (tsc() - c0) as f64 / t0.elapsed().as_secs_f64() / 1e9
    })
}

/// Statistics of a build.
#[derive(Clone, Debug, Default, serde::Serialize)]
pub struct JkStats {
    pub tasks: usize,
    pub blocks: u64,
    /// blocks where no lane passes the exchange bound (J only)
    pub blocks_j_only: u64,
    pub prim_quartets_j_only: u64,
    pub lanes: u64,
    pub quartets: u64,
    pub prim_quartets: u64,
    pub pairs: usize,
    pub pairs_total: usize,
    pub prims_kept: usize,
    pub prims_total: usize,
    pub isa: String,
    pub threads: usize,
    pub t_prep: f64,
    pub t_integrals: f64,
    pub t_reduce: f64,
    /// per class: (name, blocks, lane quartets, primitive lane quartets, seconds (CPU),
    /// seconds in the integral kernels, seconds in digestion, primitive quartets evaluated
    /// (vectors of LANES lanes, after primitive screening))
    #[allow(clippy::type_complexity, reason = "a one-off tuple of buffers")]
    pub classes: Vec<(String, u64, u64, u64, f64, f64, f64, u64)>,
    /// with `profile`: CPU seconds summed over threads in the workers (total), in the task
    /// flushes (J_ab, exchange rows to fixed point) and in everything outside the blocks and
    /// flushes (task setup, lane screening)
    pub cpu_total: f64,
    pub cpu_flush: f64,
    pub cpu_other: f64,
}

/// How a [`JkPlan`] is built.
#[derive(Clone, Copy, Debug)]
pub struct JkOptions {
    /// Quartet screening threshold on Q_ab Q_cd max|D| (as PySCF's `direct_scf_tol`); shell
    /// pairs that cannot reach it even with |D| = 100 are dropped from the layout.
    pub thresh: f64,
    /// Primitive pairs and primitive quartets are screened at `prim_factor * thresh`.
    pub prim_factor: f64,
    /// The operator of K (J is always of 1/r): a/r + b erf(omega r)/r, e.g.
    /// `Operator::from_libxc(alpha, beta, omega)` for range-separated hybrids.
    pub k_operator: Operator,
    /// Use the Hermite J-engine where no Coulomb integral block is formed anyway (J-only
    /// requests, J-only blocks, J of a non-Coulomb K operator). Off by default: with the
    /// generic-loop J-engine kernel it measured slower than the generated integral kernels plus
    /// J digestion (caffeine/def2-SVP J only, 8 threads: 0.173 s vs 0.125 s; def2-TZVP even).
    pub jengine: bool,
    /// Time every block by class (rdtsc; small overhead), reported in [`JkStats::classes`].
    pub profile: bool,
}

impl Default for JkOptions {
    fn default() -> Self {
        JkOptions {
            thresh: 1e-12,
            prim_factor: PRIM_FACTOR,
            k_operator: Operator::Coulomb,
            jengine: false,
            profile: false,
        }
    }
}

/// What a [`JkWorkspace`] is made for: the number of densities per build, and whether J, K or
/// both are built.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct JkShape {
    pub ndens: usize,
    pub j: bool,
    pub k: bool,
}

/// The inputs of one build: symmetric densities (one per [`JkShape::ndens`]) and the
/// screening threshold of this build ([`JkOptions::thresh`], or tighter for the difference
/// densities of incremental Fock builds).
pub struct JkArgs<'a> {
    pub dens: &'a [AoMatrix<'a>],
    pub thresh: f64,
}

/// The outputs of one build: J and K of every density (empty when not built), every element
/// written.
pub struct JkOut<'a, 'b> {
    pub j: &'a mut [AoMatrixMut<'b>],
    pub k: &'a mut [AoMatrixMut<'b>],
}

/// The integral-direct four-center J/K plan (see the module docs): the batched layout of a
/// basis's pairs with the per-block lane addresses of the digestion. Immutable; builds run
/// on the caller's rayon pool with a caller-owned [`JkWorkspace`].
pub struct JkPlan {
    basis: BasisSet,
    bp: BatchedPairs,
    opts: JkOptions,
    /// the instruction set of the kernels (the process-wide `Isa::detect`)
    isa: Isa,
    /// offsets of the lower-triangle shell blocks (X >= Y) of the accumulators
    blk_off: Vec<usize>,
    acc_len: usize,
    /// per ket block: lane addresses for digestion (see [`BlockMeta`])
    meta: Vec<BlockMeta>,
    /// touched-shell bits of the blocks (see `BlockMeta::tm`)
    tmask: Vec<(u32, u64)>,
    /// offsets of the shell blocks (x, y) of the shell-blocked density
    dofs: Vec<usize>,
    /// per ket block: first J accumulator element (block-major, ncd elements per block); the
    /// last entry is the total
    joff: Vec<u32>,
    /// per retained pair: (ket block, lane) holding it
    pair_loc: Vec<(u32, u32)>,
}

/// A caller-owned workspace of a [`JkPlan`]: per pool thread its fixed-point accumulators
/// and scratch, and the per-build copies of the densities; made by [`JkPlan::workspace`] for
/// a request shape and a thread count, reused build after build.
pub struct JkWorkspace {
    shape: JkShape,
    /// made for a K operator other than 1/r (its integrals get their own block buffer)
    k_separate: bool,
    slots: Vec<Slot>,
    /// shell-blocked densities, max |D| per shell pair, per ket block lane maxima and
    /// lane-interleaved D_cd (see the digester's context)
    dblk: Vec<Vec<f64>>,
    dmax: Vec<f64>,
    dpair: Vec<f64>,
    dpb: Vec<[f64; LANES]>,
    dcdb: Vec<Vec<f64>>,
    /// the tasks of a build, most expensive first, and the cost prefix sums
    tasks: Vec<(f64, u32)>,
    order: Vec<u32>,
    cum: Vec<f64>,
    /// the reduced accumulators of one density
    tot: Vec<f64>,
}

impl JkWorkspace {
    /// The shape the workspace is made for.
    pub fn shape(&self) -> JkShape {
        self.shape
    }

    /// The thread count the workspace is made for.
    pub fn nthreads(&self) -> usize {
        self.slots.len()
    }
}

impl JkPlan {
    /// The plan for the pairs `pairs` of `basis` (its batched layout orients them for the
    /// kernels, [`BatchedPairs::new`]).
    pub fn new(basis: &BasisSet, pairs: &PairSet, opts: JkOptions) -> Result<Self, IntsError> {
        let thresh = opts.thresh;
        let bp = BatchedPairs::new(basis, pairs, thresh * 1e-2, thresh * opts.prim_factor);
        let ns = basis.shells.len();
        let mut blk_off = vec![usize::MAX; ns * ns];
        let mut off = 0;
        for x in 0..ns {
            for y in 0..=x {
                blk_off[x * ns + y] = off;
                off += basis.shells[x].nfunc() * basis.shells[y].nfunc();
            }
        }
        let mut dofs = vec![0usize; ns * ns];
        let mut doff = 0;
        for x in 0..ns {
            for y in 0..ns {
                dofs[x * ns + y] = doff;
                doff += basis.shells[x].nfunc() * basis.shells[y].nfunc();
            }
        }
        let (meta, tmask) = BlockMeta::build(&bp, basis, &blk_off, &dofs);
        let mut joff = Vec::with_capacity(bp.blocks.len() + 1);
        let mut pair_loc = vec![(0u32, 0u32); bp.pairs.len()];
        let mut jo = 0usize;
        for (bi, b) in bp.blocks.iter().enumerate() {
            joff.push(jo as u32);
            jo += bp.classes[b.class as usize].nab();
            for l in 0..b.nlanes as usize {
                pair_loc[b.first as usize + l] = (bi as u32, l as u32);
            }
        }
        assert!(
            jo < u32::MAX as usize,
            "too many J accumulator elements for u32 offsets"
        );
        joff.push(jo as u32);
        Ok(JkPlan {
            basis: basis.clone(),
            bp,
            opts,
            isa: Isa::detect(),
            blk_off,
            acc_len: off,
            meta,
            tmask,
            dofs,
            joff,
            pair_loc,
        })
    }

    /// The kernels of instruction set `isa` (tests that compare them), at most the CPU's best
    /// ([`Isa::clamped`]): code compiled for instructions the CPU lacks never runs.
    #[cfg(any(test, feature = "reference"))]
    pub fn with_isa(mut self, isa: Isa) -> Self {
        self.isa = isa.clamped();
        self
    }

    /// K of the operator `op` from now on (J is always of 1/r). The layout does not depend on
    /// it; a workspace made for a Coulomb K does not fit a plan whose K is of another
    /// operator ([`Self::fits`]).
    pub fn set_k_operator(&mut self, op: Operator) {
        self.opts.k_operator = op;
    }

    /// Whether `ws` was made for this plan's K operator class (1/r or not).
    pub fn fits(&self, ws: &JkWorkspace) -> bool {
        ws.k_separate == !self.opts.k_operator.is_coulomb()
    }

    /// The options the plan was built with.
    pub fn options(&self) -> &JkOptions {
        &self.opts
    }

    /// The instruction set of the plan's kernels.
    pub fn isa(&self) -> Isa {
        self.isa
    }

    /// The batched layout (statistics and studies).
    pub fn layout(&self) -> &BatchedPairs {
        &self.bp
    }

    fn worker_sizes(&self, shape: &JkShape) -> WorkerSizes {
        let ncl = self.bp.classes.len();
        WorkerSizes {
            lmax: self.basis.max_l(),
            nbf: self.basis.nbf,
            ns: self.basis.shells.len(),
            nd: shape.ndens,
            want_j: shape.j,
            want_k: shape.k,
            k_separate: !self.opts.k_operator.is_coulomb(),
            jlen: *self.joff.last().unwrap() as usize,
            acc_len: self.acc_len,
            maxp: self
                .bp
                .pairs
                .iter()
                .map(|p| p.nprim as usize)
                .max()
                .unwrap_or(1),
            nprof: if self.opts.profile { ncl * ncl } else { 0 },
        }
    }

    /// The memory of a workspace for `shape`: per thread the accumulators and scratch, and
    /// shared the per-build copies of the densities.
    pub fn workspace_size(&self, shape: &JkShape) -> WorkspaceSize {
        let (nbf, ns) = (self.basis.nbf, self.basis.shells.len());
        let (nblk, npair) = (self.bp.blocks.len(), self.bp.pairs.len());
        let jlen = *self.joff.last().unwrap() as usize;
        let f = std::mem::size_of::<f64>();
        let shared = shape.ndens * nbf * nbf * f
            + ns * ns * f
            + npair * (2 * f + 4)
            + nblk * LANES * f
            + shape.j as usize * shape.ndens * jlen * LANES * f
            + self.acc_len * f;
        WorkspaceSize {
            per_thread: digest::worker_bytes(&self.worker_sizes(shape)),
            shared,
        }
    }

    /// A workspace for builds of `shape` on a pool of `nthreads` threads (the size of
    /// [`Self::workspace_size`]).
    pub fn workspace(&self, shape: JkShape, nthreads: usize) -> JkWorkspace {
        let z = self.worker_sizes(&shape);
        let jlen = *self.joff.last().unwrap() as usize;
        let (nbf, ns) = (self.basis.nbf, self.basis.shells.len());
        JkWorkspace {
            shape,
            k_separate: z.k_separate,
            slots: (0..nthreads).map(|_| Slot::new(self.isa, &z)).collect(),
            dblk: (0..shape.ndens).map(|_| vec![0.0; nbf * nbf]).collect(),
            dmax: vec![0.0; ns * ns],
            dpair: vec![0.0; self.bp.pairs.len()],
            dpb: vec![[0.0; LANES]; self.bp.blocks.len()],
            dcdb: if shape.j {
                (0..shape.ndens).map(|_| vec![0.0; jlen * LANES]).collect()
            } else {
                Vec::new()
            },
            tasks: Vec::with_capacity(self.bp.pairs.len()),
            order: Vec::with_capacity(self.bp.pairs.len()),
            cum: vec![0.0; self.bp.pairs.len() + 1],
            tot: vec![0.0; self.acc_len],
        }
    }

    /// J and K of the densities `args.dens` into `out` (see the module docs), on the current
    /// rayon pool. Non-finite densities give NaN matrices (the fixed-point accumulation would
    /// turn them into finite numbers).
    ///
    /// # Panics
    /// When the outputs, the densities or the pool do not match the workspace's shape and
    /// thread count, or a matrix is not of the basis's order.
    pub fn jk(&self, args: &JkArgs<'_>, out: &mut JkOut<'_, '_>, ws: &mut JkWorkspace) -> JkStats {
        let shape = JkShape {
            ndens: args.dens.len(),
            j: !out.j.is_empty(),
            k: !out.k.is_empty(),
        };
        assert_eq!(
            shape, ws.shape,
            "JkPlan::jk: a request of shape {shape:?} with a workspace made for {:?}",
            ws.shape
        );
        assert!(
            (!shape.j || out.j.len() == shape.ndens) && (!shape.k || out.k.len() == shape.ndens),
            "JkPlan::jk: one J and one K output per density"
        );
        assert!(
            self.fits(ws),
            "JkPlan::jk: a workspace made for another class of K operator"
        );
        let nbf = self.basis.nbf;
        assert!(
            args.dens.iter().all(|d| d.n() == nbf)
                && out.j.iter().chain(out.k.iter()).all(|m| m.n() == nbf),
            "JkPlan::jk: matrices of order {nbf} expected"
        );
        if args
            .dens
            .iter()
            .any(|d| (0..nbf).any(|i| (0..nbf).any(|j| !d.get(i, j).is_finite())))
        {
            for m in out.j.iter_mut().chain(out.k.iter_mut()) {
                m.fill(f64::NAN);
            }
            return JkStats::default();
        }
        let t0 = std::time::Instant::now();
        let basis = &self.basis;
        let sh = &basis.shells;
        let ns = sh.len();
        let nd = args.dens.len();
        let bp = &self.bp;
        let thresh = args.thresh;
        let (want_j, want_k) = (shape.j, shape.k);
        // shell-blocked densities and max |D| per shell pair
        let dofs = &self.dofs;
        ws.dmax.fill(0.0);
        for (d, v) in args.dens.iter().zip(ws.dblk.iter_mut()) {
            for x in 0..ns {
                for y in 0..ns {
                    let (x0, y0, nx, ny) =
                        (sh[x].ao_start, sh[y].ao_start, sh[x].nfunc(), sh[y].nfunc());
                    let o = dofs[x * ns + y];
                    debug_assert_eq!(o, x0 * nbf + nx * y0);
                    let mut m: f64 = ws.dmax[x * ns + y];
                    for u in 0..nx {
                        for w in 0..ny {
                            let val = d.get(x0 + u, y0 + w);
                            v[o + u * ny + w] = val;
                            m = m.max(val.abs());
                        }
                    }
                    ws.dmax[x * ns + y] = m;
                }
            }
        }
        let dmax_all = ws.dmax.iter().cloned().fold(0.0, f64::max);
        for (dp, p) in ws.dpair.iter_mut().zip(&bp.pairs) {
            *dp = ws.dmax[p.sa as usize * ns + p.sb as usize];
        }
        for (pb, b) in ws.dpb.iter_mut().zip(&bp.blocks) {
            *pb = std::array::from_fn(|l| {
                if l < b.nlanes as usize {
                    ws.dpair[b.first as usize + l]
                } else {
                    0.0
                }
            });
        }
        if want_j {
            for (db, v) in ws.dblk.iter().zip(ws.dcdb.iter_mut()) {
                for (bi, b) in bp.blocks.iter().enumerate() {
                    let ncd = bp.classes[b.class as usize].nab();
                    let (o, m) = (self.joff[bi] as usize * LANES, &self.meta[bi]);
                    for l in 0..LANES {
                        for x in 0..ncd {
                            v[o + x * LANES + l] = if l < b.nlanes as usize {
                                db[m.cd[l] + x]
                            } else {
                                0.0
                            };
                        }
                    }
                }
            }
        }
        // tasks: bra pairs that can pass the bound, most expensive first
        let k_fac = self.opts.k_operator.schwarz_factor();
        for (j, p) in bp.pairs.iter().enumerate() {
            let ck = &bp.classes[p.class as usize];
            ws.cum[j + 1] = ws.cum[j] + p.nprim as f64 * (ck.nz.len() as f64 + 8.0);
        }
        ws.tasks.clear();
        ws.tasks.extend(
            bp.pairs
                .iter()
                .enumerate()
                .filter(|(_, p)| {
                    p.q * bp.qmax >= thresh
                        && p.q * bp.qmax * dmax_all * k_fac.max(J_WEIGHT) >= thresh
                })
                .map(|(i, p)| {
                    let cb = &bp.classes[p.class as usize];
                    (p.nprim as f64 * cb.nh as f64 * ws.cum[i + 1], i as u32)
                }),
        );
        // every key is distinct (the pair index breaks ties), so the unstable sort gives the
        // stable sort's order without its buffer
        ws.tasks
            .sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap().then(a.1.cmp(&b.1)));
        ws.order.clear();
        ws.order.extend(ws.tasks.iter().map(|t| t.1));
        let ctx = Ctx {
            bp,
            basis,
            ns,
            nbf,
            nd,
            want_j,
            want_k,
            thresh,
            dblk: &ws.dblk,
            dmax: &ws.dmax,
            dmax_all,
            dpb: &ws.dpb,
            dcdb: &ws.dcdb,
            joff: &self.joff,
            pair_loc: &self.pair_loc,
            meta: &self.meta,
            tmask: &self.tmask,
            blk_off: &self.blk_off,
            profile: self.opts.profile,
            xop: self.opts.k_operator,
            k_separate: !self.opts.k_operator.is_coulomb(),
            k_fac,
            no_jengine: !self.opts.jengine,
            prim_factor: self.opts.prim_factor,
        };
        let t_prep = t0.elapsed().as_secs_f64();
        for s in ws.slots.iter_mut() {
            s.reset();
        }
        schedule::exact(&ws.order, &mut ws.slots, |slot, tasks| {
            digest::run(&ctx, slot, tasks)
        });
        let t_int = t0.elapsed().as_secs_f64() - t_prep;
        // reduce (exact, any order) and unpack
        let parts: Vec<_> = ws.slots.iter().map(|s| s.parts()).filter(|p| p.2).collect();
        let tot = &mut ws.tot;
        for di in 0..nd {
            if want_j {
                self.reduce_j(parts.iter().map(|p| &p.0[di]).collect(), tot);
                self.unpack(tot, &mut out.j[di]);
            }
            if want_k {
                self.reduce_k(parts.iter().map(|p| &p.1[di]).collect(), tot);
                self.unpack(tot, &mut out.k[di]);
            }
        }
        let t_red = t0.elapsed().as_secs_f64() - t_prep - t_int;
        self.stats(&ws.slots, ws.order.len(), t_prep, t_int, t_red)
    }

    /// K: per-thread canonical (lower-triangle shell block) accumulators.
    fn reduce_k(&self, parts: Vec<&Vec<Fixed128>>, tot: &mut [f64]) {
        use rayon::prelude::*;
        tot.par_chunks_mut(4096).enumerate().for_each(|(ci, ch)| {
            let base = ci * 4096;
            for (k, x) in ch.iter_mut().enumerate() {
                let mut s = Fixed128::ZERO;
                for p in &parts {
                    s += p[base + k];
                }
                *x = s.to_f64();
            }
        });
    }

    /// J: per-thread accumulators per ket block and lane, summed into the canonical layout
    /// (every canonical element of a retained pair is exactly one (block, lane, element)).
    fn reduce_j(&self, parts: Vec<&Vec<JAcc>>, tot: &mut [f64]) {
        use rayon::prelude::*;
        let bp = &self.bp;
        tot.fill(0.0);
        // every (block, lane, element) is a distinct canonical element: disjoint writes
        struct Out(*mut f64);
        unsafe impl Sync for Out {}
        let out = Out(tot.as_mut_ptr());
        let out = &out;
        (0..bp.blocks.len())
            .into_par_iter()
            .with_min_len(64)
            .for_each(|bi| {
                let b = &bp.blocks[bi];
                let ck = &bp.classes[b.class as usize];
                let (nc, ndd) = (ck.na, ck.nb);
                let (m, jo) = (&self.meta[bi], self.joff[bi] as usize);
                for l in 0..b.nlanes as usize {
                    for ic in 0..nc {
                        for id in 0..ndd {
                            let mut s = Fixed128::ZERO;
                            for p in &parts {
                                s = s.wrapping_add(p[jo + ic * ndd + id].get(l));
                            }
                            let k =
                                (m.jo[l] + ic as u32 * m.jsu[l] + id as u32 * m.jsv[l]) as usize;
                            assert!(k < self.acc_len);
                            // SAFETY: k < acc_len = tot.len(), and no two (block, lane,
                            // element) share a k
                            unsafe { *out.0.add(k) = s.to_f64() };
                        }
                    }
                }
            });
    }

    /// The canonical lower-triangle shell blocks `tot` into a full symmetric matrix.
    fn unpack(&self, tot: &[f64], m: &mut AoMatrixMut<'_>) {
        let sh = &self.basis.shells;
        let ns = sh.len();
        m.fill(0.0);
        for x in 0..ns {
            for y in 0..=x {
                let (x0, y0, nx, ny) =
                    (sh[x].ao_start, sh[y].ao_start, sh[x].nfunc(), sh[y].nfunc());
                let o = self.blk_off[x * ns + y];
                for u in 0..nx {
                    for w in 0..ny {
                        let v = tot[o + u * ny + w];
                        if x == y {
                            *m.at(x0 + u, y0 + w) += v;
                            *m.at(y0 + w, x0 + u) += v;
                        } else {
                            *m.at(x0 + u, y0 + w) = v;
                            *m.at(y0 + w, x0 + u) = v;
                        }
                    }
                }
            }
        }
    }

    /// The statistics of a build from the threads' counters.
    fn stats(&self, slots: &[Slot], ntasks: usize, t_prep: f64, t_int: f64, t_red: f64) -> JkStats {
        let bp = &self.bp;
        let ncl = bp.classes.len();
        let mut st = JkStats {
            tasks: ntasks,
            pairs: bp.pairs.len(),
            pairs_total: bp.n_pairs_total,
            prims_kept: bp.n_prims_kept,
            prims_total: bp.n_prims_total,
            isa: format!("{:?}", self.isa),
            threads: slots.len(),
            t_prep,
            t_integrals: t_int,
            t_reduce: t_red,
            ..Default::default()
        };
        let mut prof = vec![[0u64; 7]; ncl * ncl];
        let (mut cyc_total, mut cyc_flush) = (0u64, 0u64);
        for s in slots {
            let o = s.parts().3;
            cyc_total += o.cyc_total;
            cyc_flush += o.cyc_flush;
            st.blocks += o.blocks;
            st.blocks_j_only += o.blocks_j_only;
            st.prim_quartets_j_only += o.prim_quartets_j_only;
            st.lanes += o.lanes;
            st.prim_quartets += o.prim_quartets;
            for (k, e) in o.prof.iter().enumerate() {
                for x in 0..7 {
                    prof[k][x] += e[x];
                }
            }
        }
        st.quartets = st.lanes;
        if self.opts.profile {
            let ghz = tsc_ghz();
            let mut by_order = vec![0usize; ncl];
            for (ci, c) in bp.classes.iter().enumerate() {
                by_order[c.order] = ci;
            }
            for ob in 0..ncl {
                for ok in 0..ncl {
                    let e = prof[ob * ncl + ok];
                    if e[0] == 0 {
                        continue;
                    }
                    let (b, k) = (&bp.classes[by_order[ob]], &bp.classes[by_order[ok]]);
                    let name = format!("({}{}|{}{})", LS[b.la], LS[b.lb], LS[k.la], LS[k.lb]);
                    let sec = |c: u64| c as f64 / ghz / 1e9;
                    st.classes.push((
                        name,
                        e[0],
                        e[1],
                        e[2],
                        sec(e[3]),
                        sec(e[4]),
                        sec(e[5]),
                        e[6],
                    ));
                }
            }
            st.cpu_total = cyc_total as f64 / ghz / 1e9;
            st.cpu_flush = cyc_flush as f64 / ghz / 1e9;
            st.cpu_other =
                st.cpu_total - st.cpu_flush - st.classes.iter().map(|c| c.4).sum::<f64>();
            st.classes.sort_by(|a, b| b.4.partial_cmp(&a.4).unwrap());
        }
        st
    }
}

/// Per ket block, the lane addresses digestion needs (computed once per basis, so the lanes
/// cost no shell-table lookups per task): AO offsets of the ket shells c and d, the offset of
/// D_cd in the shell-blocked density, the J accumulator position of block (c, d) in canonical
/// orientation, and the symmetry weight of c == d. Padding lanes point at valid addresses
/// with zero weight.
#[derive(Clone, Copy)]
pub(super) struct BlockMeta {
    pub(super) c0: [usize; LANES],
    pub(super) d0: [usize; LANES],
    /// offset of block (c, d) in the shell-blocked density (row-major nc x nd, contiguous)
    pub(super) cd: [usize; LANES],
    /// J accumulator: element (u, v) of block (c, d) at jo + u * jsu + v * jsv
    pub(super) jo: [u32; LANES],
    pub(super) jsu: [u32; LANES],
    pub(super) jsv: [u32; LANES],
    /// 0.5 for c == d, 1 otherwise, 0 for padding lanes
    pub(super) wcd: [f64; LANES],
    /// range in `JkPlan::tmask` of this block's touched-shell bits: (word, bits) over the
    /// shells c and d of its valid lanes, words ascending
    pub(super) tm: [u32; 2],
    /// shells c and d per lane (padding lanes 0), for the lane screening gathers
    pub(super) sai: [usize; LANES],
    pub(super) sbi: [usize; LANES],
}

impl BlockMeta {
    fn build(
        bp: &BatchedPairs,
        basis: &BasisSet,
        blk_off: &[usize],
        dofs: &[usize],
    ) -> (Vec<BlockMeta>, Vec<(u32, u64)>) {
        let sh = &basis.shells;
        let ns = sh.len();
        assert!(
            blk_off
                .iter()
                .filter(|&&o| o != usize::MAX)
                .all(|&o| o < u32::MAX as usize / 2),
            "J accumulator too large for u32 offsets"
        );
        let mut tmask: Vec<(u32, u64)> = Vec::new();
        let meta = bp
            .blocks
            .iter()
            .map(|b| {
                let mut m = BlockMeta {
                    c0: [0; LANES],
                    d0: [0; LANES],
                    cd: [0; LANES],
                    jo: [0; LANES],
                    jsu: [0; LANES],
                    jsv: [0; LANES],
                    wcd: [0.0; LANES],
                    tm: [0; 2],
                    sai: [0; LANES],
                    sbi: [0; LANES],
                };
                let mut words: Vec<(u32, u64)> = Vec::new();
                for l in 0..b.nlanes as usize {
                    for x in [b.sa[l], b.sb[l]] {
                        let (w, bit) = (x / 64, 1u64 << (x % 64));
                        match words.iter_mut().find(|e| e.0 == w) {
                            Some(e) => e.1 |= bit,
                            None => words.push((w, bit)),
                        }
                    }
                }
                words.sort_unstable();
                m.tm = [tmask.len() as u32, (tmask.len() + words.len()) as u32];
                tmask.extend(words);
                for l in 0..b.nlanes as usize {
                    let (c, d) = (b.sa[l] as usize, b.sb[l] as usize);
                    (m.sai[l], m.sbi[l]) = (c, d);
                    m.c0[l] = sh[c].ao_start;
                    m.d0[l] = sh[d].ao_start;
                    m.cd[l] = dofs[c * ns + d];
                    if c >= d {
                        m.jo[l] = blk_off[c * ns + d] as u32;
                        m.jsu[l] = sh[d].nfunc() as u32;
                        m.jsv[l] = 1;
                    } else {
                        m.jo[l] = blk_off[d * ns + c] as u32;
                        m.jsu[l] = 1;
                        m.jsv[l] = sh[c].nfunc() as u32;
                    }
                    m.wcd[l] = if c == d { 0.5 } else { 1.0 };
                }
                m
            })
            .collect();
        (meta, tmask)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::basis::BasisDef;
    use crate::eri4c::eri_full;
    use crate::pairs::PairOptions;

    fn molecule(basis: &str, pure: bool) -> BasisSet {
        let def = BasisDef::load(basis, None).unwrap();
        let z = [8u32, 1, 1, 6];
        let positions = [
            [0.0, 0.0, 0.1],
            [1.4, 0.2, -0.9],
            [-1.5, -0.1, -0.8],
            [0.3, 2.9, 0.7],
        ];
        BasisSet::new(&def, &z, &positions, pure).unwrap()
    }

    fn random_sym(n: usize, seed: u64) -> Vec<f64> {
        let mut s = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let mut m = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..=i {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let v = ((s >> 11) as f64 / (1u64 << 53) as f64 - 0.5) * 0.4;
                m[i * n + j] = v;
                m[j * n + i] = v;
            }
        }
        m
    }

    /// J and K of the plan against the contraction of the full ERI tensor (Coulomb and an
    /// attenuated K operator), and bitwise the same at 1 and 3 threads.
    #[test]
    fn plan_matches_full_tensor_and_thread_counts() {
        let basis = molecule("def2-svp", true);
        let n = basis.nbf;
        let pairs = PairSet::build(&basis, PairOptions::default());
        let d = random_sym(n, 7);
        for op in [
            Operator::Coulomb,
            Operator::Attenuated {
                coulomb: 0.2,
                long_range: 0.6,
                omega: 0.4,
            },
        ] {
            let plan = JkPlan::new(
                &basis,
                &pairs,
                JkOptions {
                    thresh: 1e-16,
                    k_operator: op,
                    ..Default::default()
                },
            )
            .unwrap();
            let run = |nt: usize| -> (Vec<f64>, Vec<f64>) {
                let pool = rayon::ThreadPoolBuilder::new()
                    .num_threads(nt)
                    .build()
                    .unwrap();
                pool.install(|| {
                    let shape = JkShape {
                        ndens: 1,
                        j: true,
                        k: true,
                    };
                    let mut ws = plan.workspace(shape, nt);
                    let (mut j, mut k) = (vec![0.0; n * n], vec![0.0; n * n]);
                    {
                        let dens = [AoMatrix::row_major(&d, n)];
                        let mut jv = [AoMatrixMut::row_major(&mut j, n)];
                        let mut kv = [AoMatrixMut::row_major(&mut k, n)];
                        plan.jk(
                            &JkArgs {
                                dens: &dens,
                                thresh: 1e-16,
                            },
                            &mut JkOut {
                                j: &mut jv,
                                k: &mut kv,
                            },
                            &mut ws,
                        );
                    }
                    (j, k)
                })
            };
            let (j1, k1) = run(1);
            let (j3, k3) = run(3);
            let bits =
                |a: &[f64], b: &[f64]| a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits());
            assert!(
                bits(&j1, &j3) && bits(&k1, &k3),
                "thread-count dependent J/K"
            );
            let g = eri_full(&basis, &pairs, &Operator::Coulomb);
            let gk = eri_full(&basis, &pairs, &op);
            let n2 = n * n;
            let (mut err_j, mut err_k) = (0.0f64, 0.0f64);
            for p in 0..n {
                for q in 0..n {
                    let (mut jr, mut kr) = (0.0, 0.0);
                    for r in 0..n {
                        for s in 0..n {
                            jr += g[(p * n + q) * n2 + r * n + s] * d[r * n + s];
                            kr += gk[(p * n + r) * n2 + q * n + s] * d[r * n + s];
                        }
                    }
                    err_j = err_j.max((jr - j1[p * n + q]).abs());
                    err_k = err_k.max((kr - k1[p * n + q]).abs());
                }
            }
            assert!(
                err_j < 1e-10 && err_k < 1e-10,
                "{op:?}: J {err_j:e} K {err_k:e}"
            );
        }
    }

    /// A non-finite density gives NaN matrices, and the workspace reports its shape.
    #[test]
    fn non_finite_density_gives_nan() {
        let basis = molecule("sto-3g", true);
        let n = basis.nbf;
        let pairs = PairSet::build(&basis, PairOptions::default());
        let plan = JkPlan::new(&basis, &pairs, JkOptions::default()).unwrap();
        let shape = JkShape {
            ndens: 1,
            j: true,
            k: false,
        };
        assert!(plan.workspace_size(&shape).per_thread > 0);
        let mut ws = plan.workspace(shape, rayon::current_num_threads());
        assert_eq!(ws.shape(), shape);
        let mut d = vec![0.0; n * n];
        d[1] = f64::NAN;
        let mut j = vec![0.0; n * n];
        {
            let dens = [AoMatrix::row_major(&d, n)];
            let mut jv = [AoMatrixMut::row_major(&mut j, n)];
            plan.jk(
                &JkArgs {
                    dens: &dens,
                    thresh: 1e-12,
                },
                &mut JkOut {
                    j: &mut jv,
                    k: &mut [],
                },
                &mut ws,
            );
        }
        assert!(j.iter().all(|x| x.is_nan()));
    }
}
