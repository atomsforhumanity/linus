//! Integrals of Gaussian surface charges (polarizable continuum models) and their nuclear
//! derivatives.
//!
//! Surface point g carries the normalized spherical Gaussian charge distribution
//!
//!   rho_g(r) = (zeta_g^2 / pi)^{3/2} exp(-zeta_g^2 |r - r_g|^2),
//!
//! whose potential is erf(zeta_g |r - r_g|) / |r - r_g|. Its Coulomb integral with an AO pair,
//! (ab|g) = <a| erf(zeta_g |r - r_g|) / |r - r_g| |b>, follows from the McMurchie-Davidson form
//! of the point-charge potential: for a primitive pair with exponent sum p and centre P,
//!
//!   (Lambda_tuv | rho_g) = (2 pi / p) sqrt(alpha / p) R_tuv(alpha, P - r_g),
//!   alpha = p zeta^2 / (p + zeta^2),
//!
//! i.e. the Boys argument alpha |P - r_g|^2 and the factors (-2 alpha)^n of the Hermite
//! recursion take the reduced exponent (a point charge is zeta -> infinity, alpha = p). Every
//! zeta may differ (PySCF's int3c2e with `fakemol_for_charges(coords, expnt=zeta^2)`).
//!
//! Three plans, each writing into the caller's buffers over a caller-owned workspace:
//! - [`SurfacePlan`], for SCF iterations and responses: the potentials of densities at the
//!   points, the potential matrices of charge sets (alone or in one fused pass) and the
//!   gradient, on the batched Rys kernels of [`crate::points::grid_pot`] (the points grouped by
//!   exponent; pairs beyond the kernels' angular momenta on the McMurchie-Davidson routines).
//!   It owns what it needs of the basis, pairs and points, so it lives as long as the surface.
//! - [`SurfaceDerivPlan`], a surface plan's nuclear derivatives over a pair set of order one or
//!   two in the McMurchie-Davidson form: the gradient (the potential engine,
//!   [`crate::potential::PotentialPlan`], over [`GaussianCharges`]), the derivatives of the
//!   potentials at the points, the derivative matrices and the Hessian. The pair data are
//!   [`crate::core::hermite::ShellPair`]'s and [`crate::core::deriv::PairDeriv`]'s: the
//!   potential at the points is contracted in Hermite space, v_g = sum_prim sum_r h_r R_r(g)
//!   with the Hermite density h_r = sum_ab E\[r\]\[ab\] D_ab, the potential matrix as
//!   V_ab = sum_prim sum_r E\[r\]\[ab\] (sum_g q_g R_r(g)). Derivatives with respect to the
//!   shell centres use the derivative E-matrices, with respect to the point the raised
//!   Hermite index (d/dC R_t = -R_{t+1}); the point-centre terms are folded onto the atom that
//!   owns the point (its sphere's atom), since a surface point moves rigidly with it.
//! - [`AuxSurfacePlan`], the density-fitted PCM's two-centre integrals (P|g) of auxiliary
//!   functions with the charges in closed form, and their gradient.
//!
//! Screening: a shell pair is skipped when the Cauchy-Schwarz bound |(ab|g)| <=
//! sqrt((ab|ab)) sqrt((g|g)), (g|g) = zeta_g sqrt(2/pi), times the largest density element (or
//! sum of |q_g| sqrt((g|g))) is below the plan's threshold. That bound is infinite for point
//! charges (zeta = infinity), so [`SurfacePlan`] also skips a pair when every primitive pair
//! falls below the kernels' primitive cut (the kernels would compute nothing for it): the same
//! results, without the per-batch work.
//!
//! All reductions are thread-count independent: the contractions fold strided chunks of the
//! pairs (or points) of a fixed, problem-sized count into per-chunk accumulators of the
//! workspace and add them in chunk order ([`crate::engine::schedule`]); matrix blocks belong
//! to one pair each. The one-call forms for tests are in `reference::oneshot` and the plain
//! McMurchie-Davidson references in `reference::pcm` (feature `reference`).

use crate::basis::{ncart, BasisSet, Shell};
use crate::contract::AoMatrixMut;
use crate::core::boys;
use crate::core::deriv::{shift1, PairDeriv};
use crate::core::hermite::{build_pair, nherm, ShellPair, ShellView};
use crate::core::pair_data::PRIM_PAIR_CUT;
use crate::engine::schedule::{ordered_pool, ordered_strided};
use crate::engine::WorkspaceSize;
use crate::error::IntsError;
use crate::one_e::deriv::herm_dens;
use crate::one_e::scatter_sym_ao;
use crate::pairs::{DerivOrder, PairSet};
use crate::points::grid_pot::{PotPair, RYS_LMAX};
use crate::potential::{
    self, coulomb_hermite, gradient_fold, ChargeSlots, GaussianCharges, HermiteCtx, HermiteWork,
    Pairs, PotScratch, PotentialPlan, PotentialWorkspace,
};
use linus_num::det::{add_into, nchunk_for, DEFAULT_BUDGET};
use rayon::prelude::*;
use std::f64::consts::PI;

/// A normalized spherical Gaussian charge at `r` (bohr) with exponent `zeta^2`
/// (`zeta = f64::INFINITY` is a point charge).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GaussPoint {
    pub r: [f64; 3],
    pub zeta: f64,
}

impl GaussPoint {
    /// Coulomb self-interaction (g|g) = zeta sqrt(2/pi).
    pub fn self_coulomb(&self) -> f64 {
        self.zeta * (2.0 / PI).sqrt()
    }
}

/// Largest sqrt((g|g)) over the points.
pub(crate) fn max_sqrt_self(pts: &[GaussPoint]) -> f64 {
    pts.iter()
        .map(|g| g.self_coulomb().sqrt())
        .fold(0.0, f64::max)
}

/// Largest |D_ab| over the block of shells (sa, sb) and the densities.
pub(crate) fn block_max(
    ds: &[&[f64]],
    nbf: usize,
    a0: usize,
    b0: usize,
    na: usize,
    nb: usize,
) -> f64 {
    let mut m: f64 = 0.0;
    for d in ds {
        for x in 0..na {
            for v in &d[(a0 + x) * nbf + b0..(a0 + x) * nbf + b0 + nb] {
                m = m.max(v.abs());
            }
        }
    }
    m
}

/// sum_g |q_g| sqrt((g|g)) of one charge set (the charges' screening bound).
fn charge_bound(q: &[f64], pts: &[GaussPoint]) -> f64 {
    q.iter()
        .zip(pts)
        .map(|(x, g)| x.abs() * g.self_coulomb().sqrt())
        .sum()
}

/// The largest [`charge_bound`] over charge sets.
pub(crate) fn charges_bound(qs: &[&[f64]], pts: &[GaussPoint]) -> f64 {
    qs.iter().map(|q| charge_bound(q, pts)).fold(0.0, f64::max)
}

/// The largest sum_g |q_g| over charge sets (the kernels' primitive cut for matrices).
fn charges_abs(qs: &[&[f64]]) -> f64 {
    qs.iter()
        .map(|q| q.iter().map(|x| x.abs()).sum::<f64>())
        .fold(0.0, f64::max)
}

// ---------------------------------------------------------------------------------------------
// The surface plan: SCF iterations, responses and the gradient on the batched Rys kernels
// ---------------------------------------------------------------------------------------------

/// Points with equal Gaussian exponents (the exponents of a surface take a few values: one per
/// sphere radius and Lebedev weight), as coordinate arrays for the batched kernels, at most
/// [`GROUP`] of them.
#[derive(Clone, Debug)]
struct ZetaGroup {
    zeta: f64,
    idx: Vec<usize>,
    x: Vec<f64>,
    y: Vec<f64>,
    z: Vec<f64>,
    /// the number of points of the groups before this one (its charges' offset)
    first: usize,
}

impl ZetaGroup {
    /// erf(zeta r)/r, or 1/r for point charges (zeta infinite).
    fn op(&self) -> crate::operator::Operator {
        if self.zeta.is_finite() {
            crate::operator::Operator::LongRange { omega: self.zeta }
        } else {
            crate::operator::Operator::Coulomb
        }
    }
}

/// Points per group: the integral block of a pair (functions x points) stays in cache.
const GROUP: usize = 128;

/// The largest chunk count of a pass of matrices alone.
const MATRIX_CHUNKS: usize = 1024;

/// How the integrals of a kept shell pair are computed.
#[allow(
    clippy::large_enum_variant,
    reason = "nearly every pair takes the Rys kernels; the boxed one is the rare fallback"
)]
enum PairKernel {
    /// the batched Rys kernels (both shells within [`RYS_LMAX`])
    Rys(PotPair),
    /// the McMurchie-Davidson routines on the pair's Hermite expansion
    Md(Box<ShellPair>),
}

/// A shell pair a [`SurfacePlan`] keeps: its shells, their first functions and its kernel.
struct SurfacePair {
    /// the pair's index in the pair set the plan was built from (its Schwarz factor)
    k: usize,
    sa: usize,
    sb: usize,
    na: usize,
    nb: usize,
    a0: usize,
    b0: usize,
    kernel: PairKernel,
}

impl SurfacePair {
    fn nab(&self) -> usize {
        self.na * self.nb
    }
}

/// What a [`SurfaceWorkspace`] is made for: the largest number of densities and of charge sets
/// of one call, and whether it serves the gradient. With densities and charge sets it is made
/// for the fused pass; matrices alone (their finer chunks) then grow it on first use.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SurfaceShape {
    pub densities: usize,
    pub charge_sets: usize,
    pub gradient: bool,
}

/// The surface-charge integrals of a basis with the points of a surface, for SCF iterations,
/// responses and the gradient (see the module docs): the points grouped by exponent and each
/// kept shell pair evaluated on up to 128 points at a time by the Rys kernels of
/// [`crate::points::grid_pot`] (operator erf(zeta r)/r with the group's zeta; pairs beyond
/// their angular momenta on the McMurchie-Davidson routines). Shell pairs whose Schwarz bound
/// is far below the threshold are dropped once; per call a pair is skipped when its density
/// block (or the charges) cannot contribute above the threshold, and primitive pairs below
/// the corresponding share of it are skipped inside the kernels.
///
/// Immutable; it owns its copies of the basis, the points and the pair data, so it can live
/// with the surface. Operations write into the caller's buffers over a [`SurfaceWorkspace`]
/// made for their [`SurfaceShape`]; [`SurfacePlan::deriv_plan`] gives the McMurchie-Davidson
/// derivative contractions.
pub struct SurfacePlan {
    basis: BasisSet,
    pts: Vec<GaussPoint>,
    groups: Vec<ZetaGroup>,
    pairs: Vec<SurfacePair>,
    /// the Schwarz factors of every pair of the pair set the plan was built from
    schwarz: Vec<f64>,
    /// max over points of sqrt((g|g))
    gmax: f64,
    thresh: f64,
    /// the gradient's ordered shell pairs: (kept pair, swapped)
    ordered: Vec<(usize, bool)>,
    /// the gradient's raised and lowered shells of every shell, and the basis extended by them
    /// (the raised shell of shell i at index nsh + i, the lowered one at 2 nsh + i)
    dshells: Vec<DerivShells>,
    ext: BasisSet,
    /// the largest function count of a shell and of a kept pair
    max_nf: usize,
    max_nab: usize,
}

/// The scratch of one thread of a [`SurfacePlan`] operation: the integrals of a pair at a
/// group's points, the pair's density blocks, and the gradient's density block and derivative
/// densities.
struct SurfaceScratch {
    out: Vec<f64>,
    dblk: Vec<f64>,
    dab: Vec<f64>,
    dx: Vec<f64>,
}

impl SurfaceScratch {
    fn new([out, dblk, dab, dx]: [usize; 4]) -> Self {
        SurfaceScratch {
            out: vec![0.0; out],
            dblk: vec![0.0; dblk],
            dab: vec![0.0; dab],
            dx: vec![0.0; dx],
        }
    }
}

/// The accumulators of one reduction chunk of a [`SurfacePlan`] operation: the potentials or
/// the gradient, and the matrix blocks of the chunk's pairs in task order (the values, and per
/// block its kept pair and offset).
#[derive(Default)]
struct SurfaceChunk {
    acc: Vec<f64>,
    blocks: Vec<f64>,
    at: Vec<(usize, usize)>,
}

impl SurfaceChunk {
    /// `len` zero accumulators and no blocks.
    fn reset(&mut self, len: usize) {
        self.acc.clear();
        self.acc.resize(len, 0.0);
        self.blocks.clear();
        self.at.clear();
    }

    /// A zero block of `len` values for kept pair `i`, after the chunk's earlier ones.
    fn push_block(&mut self, i: usize, len: usize) -> &mut [f64] {
        let off = self.blocks.len();
        self.at.push((i, off));
        self.blocks.resize(off + len, 0.0);
        &mut self.blocks[off..]
    }
}

/// The caller-owned memory of a [`SurfacePlan`]'s operations, made for a [`SurfaceShape`]: per
/// thread a scratch, per reduction chunk its accumulators and matrix blocks, and the charges
/// ordered by point group. Reusable across calls of its plan.
pub struct SurfaceWorkspace {
    shape: SurfaceShape,
    scratch: Vec<SurfaceScratch>,
    chunks: Vec<SurfaceChunk>,
    qg: Vec<f64>,
    /// the scratch sizes (see [`SurfaceScratch`])
    sizes: [usize; 4],
}

impl SurfaceWorkspace {
    /// The shape the workspace is made for.
    pub fn shape(&self) -> SurfaceShape {
        self.shape
    }

    /// `nchunk` chunks reset to `len` zero accumulators, a scratch for every thread of the
    /// current pool (one more for each thread of a larger pool than the workspace was made in)
    /// and the group-ordered charges.
    fn carve(
        &mut self,
        nchunk: usize,
        len: usize,
    ) -> (&mut [SurfaceScratch], &mut [SurfaceChunk], &mut Vec<f64>) {
        let nt = rayon::current_num_threads();
        let sizes = self.sizes;
        if self.scratch.len() < nt {
            self.scratch.resize_with(nt, || SurfaceScratch::new(sizes));
        }
        if self.chunks.len() < nchunk {
            self.chunks.resize_with(nchunk, SurfaceChunk::default);
        }
        for c in &mut self.chunks[..nchunk] {
            c.reset(len);
        }
        (&mut self.scratch, &mut self.chunks[..nchunk], &mut self.qg)
    }
}

/// Chunk 0's accumulators plus the later chunks', added in chunk order.
fn add_chunks(chunks: &mut [SurfaceChunk]) {
    let (first, rest) = chunks.split_first_mut().expect("at least one chunk");
    for c in rest.iter() {
        add_into(&mut first.acc, &c.acc);
    }
}

/// A group's integrals `out` ([ab][point of the group]) contracted with the pair's density
/// blocks `dblk` ([k][ab]), added to the potentials: acc\[k * np + g\] += sum_ab D^k_ab (ab|g)
/// (zero density elements skipped).
#[inline]
fn add_potentials(
    grp: &ZetaGroup,
    dblk: &[f64],
    out: &[f64],
    nab: usize,
    np: usize,
    acc: &mut [f64],
) {
    let ng = grp.idx.len();
    for (db, a) in dblk.chunks_exact(nab).zip(acc.chunks_exact_mut(np)) {
        let mut vt = [0.0f64; GROUP];
        let vt = &mut vt[..ng];
        for (ab, &dv) in db.iter().enumerate() {
            if dv == 0.0 {
                continue;
            }
            for (t, &x) in vt.iter_mut().zip(&out[ab * ng..(ab + 1) * ng]) {
                *t += dv * x;
            }
        }
        for (&g, &v) in grp.idx.iter().zip(vt.iter()) {
            a[g] += v;
        }
    }
}

/// A group's integrals `out` ([ab][point of the group]) contracted with the group's charges
/// `q` ([k][point of the group]), added to a matrix block: blk\[k\]\[ab\] += sum_g (ab|g) q^k_g.
#[inline]
fn add_matrix_block(q: &[f64], out: &[f64], nab: usize, ng: usize, blk: &mut [f64]) {
    for (qk, b) in q.chunks_exact(ng).zip(blk.chunks_exact_mut(nab)) {
        for (ab, bv) in b.iter_mut().enumerate() {
            let row = &out[ab * ng..(ab + 1) * ng];
            let mut v = 0.0;
            for (x, y) in row.iter().zip(qk) {
                v += x * y;
            }
            *bv += v;
        }
    }
}

impl SurfacePlan {
    /// The plan of the integrals of the charges at `pts` with the pairs `pairs` of `basis`
    /// (any derivative order; screened with their Schwarz factors against `thresh`).
    pub fn new(basis: &BasisSet, pairs: &PairSet, pts: &[GaussPoint], thresh: f64) -> Self {
        let mut groups: Vec<ZetaGroup> = Vec::new();
        for (g, pt) in pts.iter().enumerate() {
            match groups.iter_mut().find(|z| z.zeta == pt.zeta) {
                Some(z) => {
                    z.idx.push(g);
                    z.x.push(pt.r[0]);
                    z.y.push(pt.r[1]);
                    z.z.push(pt.r[2]);
                }
                None => groups.push(ZetaGroup {
                    zeta: pt.zeta,
                    idx: vec![g],
                    x: vec![pt.r[0]],
                    y: vec![pt.r[1]],
                    z: vec![pt.r[2]],
                    first: 0,
                }),
            }
        }
        // batches of at most GROUP points
        let mut groups: Vec<ZetaGroup> = groups
            .into_iter()
            .flat_map(|g| {
                (0..g.idx.len())
                    .step_by(GROUP)
                    .map(|s| {
                        let e = (s + GROUP).min(g.idx.len());
                        ZetaGroup {
                            zeta: g.zeta,
                            idx: g.idx[s..e].to_vec(),
                            x: g.x[s..e].to_vec(),
                            y: g.y[s..e].to_vec(),
                            z: g.z[s..e].to_vec(),
                            first: 0,
                        }
                    })
                    .collect::<Vec<_>>()
            })
            .collect();
        let mut first = 0;
        for g in &mut groups {
            g.first = first;
            first += g.idx.len();
        }
        let schwarz = pairs.schwarz();
        let gmax = max_sqrt_self(pts);
        let kept: Vec<usize> = (0..pairs.len())
            .filter(|&k| schwarz[k] * gmax * 2.0 >= thresh * 1e-3)
            .collect();
        let sh = &basis.shells;
        let sp: Vec<SurfacePair> = kept
            .into_par_iter()
            .map(|k| {
                let pr = pairs.pair(k);
                // the erf kernels need la, lb <= RYS_LMAX; the Coulomb ones fall back to MD
                // themselves
                let kernel = if pr.la <= RYS_LMAX && pr.lb <= RYS_LMAX {
                    PairKernel::Rys(PotPair::new(basis, pr.sa, pr.sb, PRIM_PAIR_CUT))
                } else {
                    PairKernel::Md(Box::new(pr.clone()))
                };
                SurfacePair {
                    k,
                    sa: pr.sa,
                    sb: pr.sb,
                    na: pr.na,
                    nb: pr.nb,
                    a0: sh[pr.sa].ao_start,
                    b0: sh[pr.sb].ao_start,
                    kernel,
                }
            })
            .collect();
        let ordered = sp
            .iter()
            .enumerate()
            .flat_map(|(i, p)| {
                std::iter::once((i, false)).chain((p.sa != p.sb).then_some((i, true)))
            })
            .collect();
        let dshells: Vec<DerivShells> = sh.iter().map(DerivShells::new).collect();
        let mut ext = basis.clone();
        ext.shells.extend(dshells.iter().map(|x| x.up.clone()));
        ext.shells.extend(
            dshells
                .iter()
                .map(|x| x.down.clone().unwrap_or_else(|| x.up.clone())),
        );
        SurfacePlan {
            basis: basis.clone(),
            pts: pts.to_vec(),
            groups,
            max_nf: sh.iter().map(Shell::nfunc).max().unwrap_or(0),
            max_nab: sp.iter().map(SurfacePair::nab).max().unwrap_or(0),
            pairs: sp,
            schwarz: schwarz.to_vec(),
            gmax,
            thresh,
            ordered,
            dshells,
            ext,
        }
    }

    /// The number of point batches (one or more per distinct exponent).
    pub fn ngroups(&self) -> usize {
        self.groups.len()
    }

    /// The points.
    pub fn points(&self) -> &[GaussPoint] {
        &self.pts
    }

    /// The basis.
    pub fn basis(&self) -> &BasisSet {
        &self.basis
    }

    /// The scratch sizes of a workspace for `shape` (see [`SurfaceScratch`]).
    fn scratch_sizes(&self, shape: SurfaceShape) -> [usize; 4] {
        // a raised shell against a shell: the gradient's widest pair
        let grad_nab = if shape.gradient {
            ncart(self.basis.max_l() + 1) * self.max_nf
        } else {
            0
        };
        let nab = if shape.densities + shape.charge_sets > 0 {
            self.max_nab
        } else {
            0
        };
        let g = usize::from(shape.gradient);
        [
            nab.max(grad_nab) * GROUP,
            shape.densities * self.max_nab,
            g * self.max_nf * self.max_nf,
            3 * grad_nab,
        ]
    }

    /// (chunk count, accumulator length) of a pass with `nd` densities: the potentials' chunk
    /// count, sized by the problem so that their sums do not depend on the threads; for
    /// matrices alone (each block belongs to one pair, so any count gives the same bits) up to
    /// [`MATRIX_CHUNKS`], few pairs each, which keeps the threads busy to the end.
    fn pass_chunks(&self, nd: usize) -> (usize, usize) {
        if nd == 0 {
            return (self.pairs.len().clamp(1, MATRIX_CHUNKS), 0);
        }
        let len = nd * self.pts.len();
        (nchunk_for(self.pairs.len(), len, DEFAULT_BUDGET), len)
    }

    /// (chunk count, accumulator length) of the gradient.
    fn gradient_chunks(&self) -> (usize, usize) {
        let len = 3 * (self.basis.natom() + self.pts.len());
        (nchunk_for(self.ordered.len(), len, DEFAULT_BUDGET), len)
    }

    /// Per chunk index, what the operations of `shape` put in it at most: (accumulator values,
    /// matrix block values, matrix blocks). Each operation has its own chunk layout (the
    /// potentials' or the gradient's problem-sized count, or the finer one of matrices alone,
    /// reserved for a shape without densities).
    fn chunk_capacities(&self, shape: SurfaceShape) -> Vec<(usize, usize, usize)> {
        let mut caps: Vec<(usize, usize, usize)> = Vec::new();
        let mut add = |nchunk: usize, len: usize, nq: usize| {
            if caps.len() < nchunk {
                caps.resize(nchunk, (0, 0, 0));
            }
            for (c, cap) in caps.iter_mut().enumerate().take(nchunk) {
                // the blocks of the strided chunk's pairs
                let (nab, n) = if nq > 0 {
                    self.pairs[c.min(self.pairs.len())..]
                        .iter()
                        .step_by(nchunk)
                        .fold((0, 0), |(s, n), p| (s + p.nab(), n + 1))
                } else {
                    (0, 0)
                };
                *cap = (cap.0.max(len), cap.1.max(nq * nab), cap.2.max(n));
            }
        };
        if shape.densities > 0 {
            let (c, l) = self.pass_chunks(shape.densities);
            add(c, l, shape.charge_sets);
        }
        if shape.densities == 0 && shape.charge_sets > 0 {
            add(self.pass_chunks(0).0, 0, shape.charge_sets);
        }
        if shape.gradient {
            let (c, l) = self.gradient_chunks();
            add(c, l, 0);
        }
        caps
    }

    /// The bytes of a workspace for `shape`: the scratch per thread; the chunk accumulators,
    /// matrix blocks and group-ordered charges shared.
    pub fn workspace_size(&self, shape: SurfaceShape) -> WorkspaceSize {
        let s = self.scratch_sizes(shape);
        let caps = self.chunk_capacities(shape);
        let values: usize = caps.iter().map(|c| c.0 + c.1).sum();
        let blocks: usize = caps.iter().map(|c| c.2).sum();
        let nq = shape.charge_sets.max(usize::from(shape.gradient));
        WorkspaceSize {
            per_thread: 8 * s.iter().sum::<usize>(),
            shared: 8 * (values + nq * self.pts.len()) + blocks * size_of::<(usize, usize)>(),
        }
    }

    /// A workspace for the operations of `shape`.
    pub fn workspace(&self, shape: SurfaceShape) -> SurfaceWorkspace {
        let sizes = self.scratch_sizes(shape);
        let nt = rayon::current_num_threads();
        let chunks = self
            .chunk_capacities(shape)
            .into_iter()
            .map(|(len, values, blocks)| SurfaceChunk {
                acc: Vec::with_capacity(len),
                blocks: Vec::with_capacity(values),
                at: Vec::with_capacity(blocks),
            })
            .collect();
        let nq = shape.charge_sets.max(usize::from(shape.gradient));
        SurfaceWorkspace {
            shape,
            scratch: (0..nt).map(|_| SurfaceScratch::new(sizes)).collect(),
            chunks,
            qg: Vec::with_capacity(nq * self.pts.len()),
            sizes,
        }
    }

    /// The pair's density blocks f D_ab of the densities ([k][ab], f = 2 for distinct shells)
    /// into `dblk`.
    fn density_blocks(&self, p: &SurfacePair, ds: &[&[f64]], dblk: &mut [f64]) {
        let nbf = self.basis.nbf;
        let f = if p.sa == p.sb { 1.0 } else { 2.0 };
        let mut i = 0;
        for d in ds {
            for x in 0..p.na {
                for y in 0..p.nb {
                    dblk[i] = f * d[(p.a0 + x) * nbf + p.b0 + y];
                    i += 1;
                }
            }
        }
    }

    /// The charges ordered by point group, [group][k][point of the group], into `qg`.
    fn group_charges(&self, qs: &[&[f64]], qg: &mut Vec<f64>) {
        qg.clear();
        for grp in &self.groups {
            for q in qs {
                qg.extend(grp.idx.iter().map(|&g| q[g]));
            }
        }
    }

    /// The matrix blocks of the chunks into `out` (zeroed first; every element belongs to one
    /// pair's block or its transpose).
    fn scatter_blocks(&self, chunks: &[SurfaceChunk], out: &mut [AoMatrixMut<'_>]) {
        for m in out.iter_mut() {
            m.fill(0.0);
        }
        for c in chunks {
            for &(i, off) in &c.at {
                let p = &self.pairs[i];
                let nab = p.nab();
                for (k, m) in out.iter_mut().enumerate() {
                    let blk = &c.blocks[off + k * nab..off + (k + 1) * nab];
                    scatter_sym_ao(m, (p.a0, p.b0), (p.na, p.nb), blk);
                }
            }
        }
    }

    fn check(&self, ws: &SurfaceWorkspace, densities: usize, charge_sets: usize, gradient: bool) {
        let s = ws.shape;
        assert!(
            densities <= s.densities && charge_sets <= s.charge_sets && (!gradient || s.gradient),
            "a workspace made for {s:?} used for {densities} densities, {charge_sets} charge sets{}",
            if gradient { " and the gradient" } else { "" }
        );
    }

    /// The potentials of densities at the points, out\[k * npts + g\] = sum_ab D^k_ab (ab|g)
    /// (positive operator; D^k symmetric nbf x nbf row-major), overwritten: the pass of
    /// [`Self::potentials_and_matrices`] without charges.
    pub fn potentials(&self, ds: &[&[f64]], out: &mut [f64], ws: &mut SurfaceWorkspace) {
        self.potentials_and_matrices((ds, &[]), out, &mut [], ws);
    }

    /// The potential matrices of charge sets, out\[k\]_ab = sum_g q^k_g (ab|g) (symmetric),
    /// overwritten: the pass of [`Self::potentials_and_matrices`] without densities.
    pub fn matrices(&self, qs: &[&[f64]], out: &mut [AoMatrixMut<'_>], ws: &mut SurfaceWorkspace) {
        self.potentials_and_matrices((&[], qs), &mut [], out, ws);
    }

    /// One pass over the kept pairs: the potentials of the densities `ds` at the points into
    /// `v` (\[k * npts + g\]) and the potential matrices of the charge sets `qs` into `m`, every
    /// integral evaluated once (an SCF iteration's fused pass); both overwritten. A pair is
    /// skipped for the potentials when its density block cannot contribute above the threshold,
    /// for the matrices when the charges cannot, and altogether when every primitive pair falls
    /// below the kernels' cut of what it is evaluated for.
    #[allow(clippy::neg_cmp_op_on_partial_ord, reason = "true for NaN too")]
    pub fn potentials_and_matrices(
        &self,
        (ds, qs): (&[&[f64]], &[&[f64]]),
        v: &mut [f64],
        m: &mut [AoMatrixMut<'_>],
        ws: &mut SurfaceWorkspace,
    ) {
        let nbf = self.basis.nbf;
        let np = self.pts.len();
        let (nd, nq) = (ds.len(), qs.len());
        assert_eq!(v.len(), nd * np, "the potentials are ndens x npoints");
        assert_eq!(m.len(), nq, "one matrix per charge set");
        self.check(ws, nd, nq, false);
        let qsum = charges_bound(qs, &self.pts);
        let qabs = charges_abs(qs);
        let (nchunk, len) = self.pass_chunks(nd);
        let (scratch, chunks, qg) = ws.carve(nchunk, len);
        self.group_charges(qs, qg);
        let qg: &[f64] = qg;
        ordered_strided(self.pairs.len(), scratch, chunks, |s, c, i| {
            let p = &self.pairs[i];
            let dmax = block_max(ds, nbf, p.a0, p.b0, p.na, p.nb);
            // (a pass without densities or charges wants nothing of them)
            let want_v = nd > 0 && !(dmax * self.schwarz[p.k] * self.gmax * 2.0 < self.thresh);
            let want_m = nq > 0 && !(self.schwarz[p.k] * qsum < self.thresh);
            if !want_v && !want_m {
                return;
            }
            let nab = p.nab();
            let pot = match &p.kernel {
                PairKernel::Md(pr) => {
                    if want_v {
                        let f = if p.sa == p.sb { 1.0 } else { 2.0 };
                        md_pair_points(pr, &self.pts, ds, (nbf, p.a0, p.b0), f, &mut c.acc);
                    }
                    if want_m {
                        md_pair_matrix(pr, &self.pts, qs, c.push_block(i, nq * nab));
                    }
                    return;
                }
                PairKernel::Rys(pot) => pot,
            };
            let dblk = &mut s.dblk[..if want_v { nd * nab } else { 0 }];
            if want_v {
                self.density_blocks(p, ds, dblk);
            }
            let cut_v = if want_v {
                self.thresh / (2.0 * dmax * nab as f64)
            } else {
                f64::INFINITY
            };
            let cut_m = if want_m {
                self.thresh / (2.0 * qabs.max(1e-300))
            } else {
                f64::INFINITY
            };
            let cut = cut_v.min(cut_m);
            // every primitive pair below the kernels' cut: nothing to add (the potentials and
            // the matrix block would be zero)
            if pot.nprim_above(cut) == 0 {
                return;
            }
            let SurfaceChunk { acc, blocks, at } = c;
            let blk = if want_m {
                let off = blocks.len();
                at.push((i, off));
                blocks.resize(off + nq * nab, 0.0);
                &mut blocks[off..]
            } else {
                &mut blocks[..0]
            };
            for grp in &self.groups {
                let ng = grp.idx.len();
                let o = &mut s.out[..nab * ng];
                pot.compute(&grp.op(), &grp.x, &grp.y, &grp.z, cut, o);
                if want_v {
                    add_potentials(grp, dblk, o, nab, np, acc);
                }
                if want_m {
                    add_matrix_block(&qg[nq * grp.first..nq * (grp.first + ng)], o, nab, ng, blk);
                }
            }
        });
        add_chunks(chunks);
        v.copy_from_slice(&chunks[0].acc);
        self.scatter_blocks(chunks, m);
    }

    /// The gradient of sum_g q_g sum_ab D_ab (ab|g) on the batched Rys kernels: the part from
    /// the shell centres per atom into `atoms` and the part from the point positions per point
    /// into `points`, overwritten.
    ///
    /// The first-centre derivative of a contracted function is a combination of the raised and
    /// lowered Cartesian shells (`DerivShells`): d/dA_x phi_mu = sum_u RU_x\[mu\]\[u\] U_u +
    /// sum_w RL_x\[mu\]\[w\] L_w, so per ordered shell pair (A, B) and direction the derivative
    /// integrals are potential integrals of the pairs (U_A, B) and (L_A, B) contracted with the
    /// derivative densities RU_x^T D_AB and RL_x^T D_AB. Over ordered pairs the second-centre
    /// terms equal the first-centre ones (D symmetric), so the atom part is 2 sum_ordered dA and
    /// the point part -2 q_g sum_ordered dA (translational invariance). Pairs whose raised
    /// shell is beyond the erf Rys kernels (l > RYS_LMAX) use the McMurchie-Davidson routines.
    pub fn gradient(
        &self,
        q: &[f64],
        d: &[f64],
        atoms: &mut [[f64; 3]],
        points: &mut [[f64; 3]],
        ws: &mut SurfaceWorkspace,
    ) {
        let nbf = self.basis.nbf;
        let natom = self.basis.natom();
        let np = self.pts.len();
        let sh = &self.basis.shells;
        let nsh = sh.len();
        assert_eq!(atoms.len(), natom, "the atoms' gradient has natom rows");
        assert_eq!(points.len(), np, "the points' gradient has npoints rows");
        self.check(ws, 0, 0, true);
        let qsum = charge_bound(q, &self.pts);
        let thresh = self.thresh * 1e-2;
        let (nchunk, len) = self.gradient_chunks();
        let (scratch, chunks, qg) = ws.carve(nchunk, len);
        self.group_charges(&[q], qg);
        let qg: &[f64] = qg;
        ordered_strided(self.ordered.len(), scratch, chunks, |s, c, it| {
            let (i, swap) = self.ordered[it];
            let p = &self.pairs[i];
            let (ia, ib) = if swap { (p.sb, p.sa) } else { (p.sa, p.sb) };
            let (a0, na, b0, nb) = (
                sh[ia].ao_start,
                sh[ia].nfunc(),
                sh[ib].ao_start,
                sh[ib].nfunc(),
            );
            let dmax = block_max(&[d], nbf, a0, b0, na, nb);
            if dmax * self.schwarz[p.k] * qsum * 2.0 < thresh {
                return;
            }
            let dab = &mut s.dab[..na * nb];
            for (x, v) in dab.iter_mut().enumerate() {
                *v = d[(a0 + x / nb) * nbf + b0 + x % nb];
            }
            let dr = &self.dshells[ia];
            let (gpt, gat) = c.acc.split_at_mut(3 * np);
            let gat = &mut gat[..3 * natom];
            let atom = sh[ia].atom;
            for (sidx, nc, r) in [(nsh + ia, dr.nup, &dr.ru), (2 * nsh + ia, dr.ndown, &dr.rl)] {
                if nc == 0 {
                    continue;
                }
                // derivative densities DX[x][u][b] = sum_mu R_x[mu][u] D[mu][b]
                let dx = &mut s.dx[..3 * nc * nb];
                dx.fill(0.0);
                for x in 0..3 {
                    for mu in 0..na {
                        for u in 0..nc {
                            let rv = r[x][mu * nc + u];
                            if rv == 0.0 {
                                continue;
                            }
                            for b in 0..nb {
                                dx[(x * nc + u) * nb + b] += rv * dab[mu * nb + b];
                            }
                        }
                    }
                }
                let dxmax = dx.iter().fold(0.0f64, |m, v| m.max(v.abs()));
                if dxmax == 0.0 {
                    continue;
                }
                let lu = self.ext.shells[sidx].l;
                let nab = nc * nb;
                if lu <= RYS_LMAX && sh[ib].l <= RYS_LMAX {
                    let pot = PotPair::new(&self.ext, sidx, ib, PRIM_PAIR_CUT);
                    let cut = thresh / (2.0 * dxmax * nab as f64);
                    // every primitive pair below the kernels' cut: nothing to add
                    if pot.nprim_above(cut) == 0 {
                        continue;
                    }
                    for grp in &self.groups {
                        let ng = grp.idx.len();
                        let qk = &qg[grp.first..grp.first + ng];
                        let o = &mut s.out[..nab * ng];
                        pot.compute(&grp.op(), &grp.x, &grp.y, &grp.z, cut, o);
                        for x in 0..3 {
                            let dxx = &dx[x * nab..(x + 1) * nab];
                            let mut t = [0.0f64; GROUP];
                            let t = &mut t[..ng];
                            for (ab, &dv) in dxx.iter().enumerate() {
                                if dv == 0.0 {
                                    continue;
                                }
                                for (tv, &ov) in t.iter_mut().zip(&o[ab * ng..(ab + 1) * ng]) {
                                    *tv += dv * ov;
                                }
                            }
                            let mut sx = 0.0;
                            for ((&g, &tv), &qv) in grp.idx.iter().zip(t.iter()).zip(qk) {
                                gpt[3 * g + x] += tv;
                                sx += qv * tv;
                            }
                            gat[3 * atom + x] += sx;
                        }
                    }
                } else {
                    // McMurchie-Davidson: the pair (raised/lowered A, B) with a local density
                    let spair = build_pair(
                        &ShellView::from_shell(&self.ext.shells[sidx]),
                        &ShellView::from_shell(&sh[ib]),
                        sidx,
                        ib,
                    );
                    let nl = nc + nb;
                    let dl: Vec<Vec<f64>> = (0..3)
                        .map(|x| {
                            let mut m = vec![0.0; nl * nl];
                            for u in 0..nc {
                                for b in 0..nb {
                                    m[u * nl + nc + b] = dx[(x * nc + u) * nb + b];
                                }
                            }
                            m
                        })
                        .collect();
                    let refs: Vec<&[f64]> = dl.iter().map(|v| v.as_slice()).collect();
                    let mut v = vec![0.0; 3 * np];
                    md_pair_points(&spair, &self.pts, &refs, (nl, 0, nc), 1.0, &mut v);
                    for x in 0..3 {
                        let mut sx = 0.0;
                        for g in 0..np {
                            gpt[3 * g + x] += v[x * np + g];
                            sx += q[g] * v[x * np + g];
                        }
                        gat[3 * atom + x] += sx;
                    }
                }
            }
        });
        add_chunks(chunks);
        let acc = &chunks[0].acc;
        for (a, o) in atoms.iter_mut().enumerate() {
            for k in 0..3 {
                o[k] = 2.0 * acc[3 * (np + a) + k];
            }
        }
        for (g, o) in points.iter_mut().enumerate() {
            for k in 0..3 {
                o[k] = -2.0 * q[g] * acc[3 * g + k];
            }
        }
    }

    /// The McMurchie-Davidson derivative contractions of the plan's integrals over `pairs` (the
    /// pair set the plan was built from, with derivative expansions of order one or two), the
    /// point g moving with atom `owner[g]` ([`SurfaceDerivPlan`]). Refuses a pair set of order
    /// zero.
    pub fn deriv_plan<'a>(
        &'a self,
        pairs: &'a PairSet,
        owner: &'a [usize],
    ) -> Result<SurfaceDerivPlan<'a>, IntsError> {
        if pairs.deriv_order() == DerivOrder::Zero {
            return Err(IntsError::Order {
                need: DerivOrder::One,
                have: DerivOrder::Zero,
            });
        }
        assert_eq!(
            pairs.len(),
            self.schwarz.len(),
            "the derivative pairs of the pair set the surface plan was built from"
        );
        assert_eq!(owner.len(), self.pts.len(), "an owner atom per point");
        Ok(SurfaceDerivPlan {
            plan: self,
            pairs,
            owner,
        })
    }
}

// ---------------------------------------------------------------------------------------------
// The derivative plan: McMurchie-Davidson derivative contractions
// ---------------------------------------------------------------------------------------------

/// A [`SurfacePlan`]'s nuclear derivatives in the McMurchie-Davidson form over a pair set of
/// order one or two (the pair set the plan was built from, with derivative expansions; see
/// the module docs), the point g moving with atom `owner[g]`: the gradient (the potential
/// engine), the derivatives of the density potentials at the points, the derivative matrices
/// of the charges' potential and (order two) the Hessian, into the caller's buffers. The
/// screening is the surface plan's (its Schwarz factors and threshold, tightened for
/// derivatives, whose integrals are larger by about the inverse extent). Made by
/// [`SurfacePlan::deriv_plan`].
pub struct SurfaceDerivPlan<'a> {
    plan: &'a SurfacePlan,
    pairs: &'a PairSet,
    owner: &'a [usize],
}

impl<'a> SurfaceDerivPlan<'a> {
    /// The plan's derivative order (its pair set's).
    pub fn order(&self) -> DerivOrder {
        self.pairs.deriv_order()
    }

    /// The surface plan whose derivatives these are.
    pub fn surface(&self) -> &SurfacePlan {
        self.plan
    }

    /// The largest angular momentum sum of a pair (the potential scratch's size).
    fn lab(&self) -> usize {
        2 * self.plan.basis.max_l()
    }

    /// The accumulator values of the gradient and, order two, the Hessian (chunk count times
    /// length).
    fn pool_size(&self) -> usize {
        let n = self.pairs.len();
        let natom = self.plan.basis.natom();
        let (c, l) = potential::chunks(n, natom + self.plan.pts.len(), DerivOrder::One);
        if self.order() == DerivOrder::Two {
            let (c2, l2) = potential::chunks(n, natom, DerivOrder::Two);
            (c * l).max(c2 * l2)
        } else {
            c * l
        }
    }

    /// (chunk count, accumulator length) of the point potentials' derivatives.
    fn potentials_deriv_chunks(&self) -> (usize, usize) {
        let len = 3 * self.plan.basis.natom() * self.plan.pts.len();
        (nchunk_for(self.pairs.len(), len, DEFAULT_BUDGET), len)
    }

    /// The bytes of a workspace for the gradient and, order two, the Hessian: the scratch per
    /// thread, the chunk accumulators shared. [`Self::potentials_deriv`] grows the accumulators
    /// to its own, (3 natom) x npoints per chunk ([`Self::potentials_deriv_size`]).
    pub fn workspace_size(&self) -> WorkspaceSize {
        WorkspaceSize {
            per_thread: PotScratch::bytes(self.lab(), self.order()),
            shared: 8 * self.pool_size(),
        }
    }

    /// The shared bytes of the point potentials' derivatives' accumulators.
    pub fn potentials_deriv_size(&self) -> usize {
        let (c, l) = self.potentials_deriv_chunks();
        8 * c * l
    }

    /// A workspace for the contractions of the plan's order (see [`Self::workspace_size`]).
    pub fn workspace(&self) -> PotentialWorkspace {
        PotentialWorkspace::new(self.pool_size(), self.lab(), self.order())
    }

    /// The pairs whose bound (largest density element of the block, Schwarz factor and charge
    /// bound `qsum`) reaches `thresh`, with their derivative expansions.
    fn screened<'d>(
        &'d self,
        d: &'d [f64],
        qsum: f64,
        thresh: f64,
    ) -> Pairs<'d, impl Fn(usize) -> Option<(&'d ShellPair, &'d PairDeriv)> + Sync + 'd> {
        let (basis, pairs, schwarz) = (&self.plan.basis, self.pairs, &self.plan.schwarz);
        let (nbf, sh) = (basis.nbf, &basis.shells);
        Pairs {
            npairs: pairs.len(),
            pair: move |kp| {
                let pr = pairs.pair(kp);
                let (s1, s2) = (&sh[pr.sa], &sh[pr.sb]);
                let bound = block_max(&[d], nbf, s1.ao_start, s2.ao_start, pr.na, pr.nb)
                    * schwarz[kp]
                    * qsum
                    * 2.0;
                if bound < thresh {
                    None
                } else {
                    Some((pr, pairs.deriv(kp)))
                }
            },
        }
    }

    /// The gradient of sum_g q_g sum_ab D_ab (ab|g): the part from the shell centres per atom
    /// into `atoms` and the part from the point positions per point into `points`,
    /// overwritten (the potential engine over the charges, each point a coordinate of its own).
    pub fn gradient(
        &self,
        q: &[f64],
        d: &[f64],
        atoms: &mut [[f64; 3]],
        points: &mut [[f64; 3]],
        ws: &mut PotentialWorkspace,
    ) {
        let (basis, pts) = (&self.plan.basis, &self.plan.pts[..]);
        let natom = basis.natom();
        assert_eq!(atoms.len(), natom, "the atoms' gradient has natom rows");
        assert_eq!(
            points.len(),
            pts.len(),
            "the points' gradient has npoints rows"
        );
        let qsum = charge_bound(q, pts);
        let src = GaussianCharges {
            pts,
            q,
            slots: ChargeSlots::Own { first: natom },
        };
        let screened = self.screened(d, qsum, self.plan.thresh * 1e-2);
        let (nchunk, len) = potential::chunks(self.pairs.len(), natom + pts.len(), DerivOrder::One);
        let (pool, scratch) = ws.carve(nchunk, len);
        gradient_fold(basis, &screened, &src, d, pool, nchunk, scratch);
        atoms.as_flattened_mut().copy_from_slice(&pool[..3 * natom]);
        points
            .as_flattened_mut()
            .copy_from_slice(&pool[3 * natom..len]);
    }

    /// The first derivatives of the density potentials at the points with respect to the
    /// atoms, point g moving with atom owner\[g\]:
    /// out\[(3 A + k) * npts + g\] = d/dX_Ak sum_ab D_ab (ab|g), overwritten.
    pub fn potentials_deriv(&self, d: &[f64], out: &mut [f64], ws: &mut PotentialWorkspace) {
        let basis = &self.plan.basis;
        let (pts, owner, schwarz) = (&self.plan.pts, self.owner, &self.plan.schwarz);
        let (nbf, natom, np) = (basis.nbf, basis.natom(), pts.len());
        let sh = &basis.shells;
        let shf = shift1();
        let pairs = self.pairs;
        let npairs = pairs.len();
        assert_eq!(
            out.len(),
            3 * natom * np,
            "the derivatives are (3 natom) x npoints"
        );
        let gmax = self.plan.gmax;
        let thresh = self.plan.thresh * 1e-2;
        let (nchunk, len) = self.potentials_deriv_chunks();
        ws.reserve(nchunk * len);
        let (pool, scratch) = ws.carve(nchunk, len);
        ordered_pool(npairs, pool, len, nchunk, scratch, |acc, s, kp| {
            let pr = pairs.pair(kp);
            let pd = pairs.deriv(kp);
            let (s1, s2) = (&sh[pr.sa], &sh[pr.sb]);
            if block_max(&[d], nbf, s1.ao_start, s2.ao_start, pr.na, pr.nb)
                * schwarz[kp]
                * gmax
                * 2.0
                < thresh
            {
                return;
            }
            let lab = pr.lab();
            let (nh0, nh1) = (nherm(lab), nherm(lab + 1));
            let f = if pr.sa == pr.sb { 1.0 } else { 2.0 };
            let (hd, hda, r, work) = s.first_order(nh0, nh1);
            let (xa, xb) = (s1.atom, s2.atom);
            for ip in 0..pr.nprim {
                herm_dens(
                    pr.e_prim(ip),
                    nh0,
                    1,
                    d,
                    nbf,
                    s1.ao_start,
                    s2.ao_start,
                    pr.na,
                    pr.nb,
                    f,
                    hd,
                );
                herm_dens(
                    pd.ea_prim(ip),
                    nh1,
                    3,
                    d,
                    nbf,
                    s1.ao_start,
                    s2.ao_start,
                    pr.na,
                    pr.nb,
                    f,
                    hda,
                );
                for (g, pt) in pts.iter().enumerate() {
                    coulomb_hermite(
                        pt.r,
                        pt.zeta,
                        1.0,
                        &mut HermiteCtx {
                            l: lab + 1,
                            p: pr.p[ip],
                            pc: pr.pc[ip],
                            r: &mut *r,
                            work: &mut *work,
                        },
                    );
                    let xc = owner[g];
                    for k in 0..3 {
                        let mut va = 0.0;
                        for rr in 0..nh1 {
                            va += hda[k * nh1 + rr] * r[rr];
                        }
                        let mut vc = 0.0;
                        for rr in 0..nh0 {
                            vc -= hd[rr] * r[shf[rr][k] as usize];
                        }
                        acc[(3 * xa + k) * np + g] += va;
                        acc[(3 * xc + k) * np + g] += vc;
                        acc[(3 * xb + k) * np + g] -= va + vc;
                    }
                }
            }
        });
        out.copy_from_slice(&pool[..len]);
    }

    /// The derivative matrices of the charges' potential matrix sum_g q_g (ab|g) with respect
    /// to the atoms (shell centres and points; point g moves with atom owner\[g\]) into `out`
    /// ((3 natom) x nbf x nbf, row-major, index 3 A + k), overwritten. The pair blocks are
    /// computed in parallel and added pair by pair (every element of a block belongs to one
    /// pair), so the result does not depend on the thread count.
    pub fn matrices_deriv(&self, q: &[f64], out: &mut [f64]) {
        let basis = &self.plan.basis;
        let (pts, owner, schwarz) = (&self.plan.pts[..], self.owner, &self.plan.schwarz);
        let (nbf, natom) = (basis.nbf, basis.natom());
        let n2 = nbf * nbf;
        assert_eq!(
            out.len(),
            3 * natom * n2,
            "the derivative matrices are (3 natom) x nbf^2"
        );
        let sh = &basis.shells;
        let shf = shift1();
        let pairs = self.pairs;
        let qsum = charge_bound(q, pts);
        let thresh = self.plan.thresh * 1e-2;
        let lab_max = self.lab();
        // per pair: [3][nab] for A, [natom][3][nab] for the point positions
        let blocks: Vec<Option<(Vec<f64>, Vec<f64>)>> = pairs
            .pairs()
            .par_iter()
            .enumerate()
            .map_init(
                || MatDerivScratch::new(lab_max, natom),
                |w, (kp, pr)| {
                    if schwarz[kp] * qsum < thresh {
                        return None;
                    }
                    let pd = pairs.deriv(kp);
                    let lab = pr.lab();
                    let (nh0, nh1) = (nherm(lab), nherm(lab + 1));
                    let nab = pr.nab();
                    let mut oa = vec![0.0; 3 * nab];
                    let mut oc = vec![0.0; 3 * natom * nab];
                    let acc1 = &mut w.acc1[..nh1];
                    let accc = &mut w.accc[..3 * natom * nh0];
                    let r = &mut w.r[..nh1];
                    for ip in 0..pr.nprim {
                        acc1.fill(0.0);
                        accc.fill(0.0);
                        for (g, pt) in pts.iter().enumerate() {
                            if q[g] == 0.0 {
                                continue;
                            }
                            coulomb_hermite(
                                pt.r,
                                pt.zeta,
                                q[g],
                                &mut HermiteCtx {
                                    l: lab + 1,
                                    p: pr.p[ip],
                                    pc: pr.pc[ip],
                                    r: &mut *r,
                                    work: &mut w.work,
                                },
                            );
                            for (a, y) in acc1.iter_mut().zip(r.iter()) {
                                *a += y;
                            }
                            let c = &mut accc[3 * owner[g] * nh0..3 * (owner[g] + 1) * nh0];
                            for k in 0..3 {
                                for rr in 0..nh0 {
                                    c[k * nh0 + rr] -= r[shf[rr][k] as usize];
                                }
                            }
                        }
                        let ea = pd.ea_prim(ip);
                        for rr in 0..nh1 {
                            let a = acc1[rr];
                            if a == 0.0 {
                                continue;
                            }
                            for k in 0..3 {
                                let row = &ea[(rr * 3 + k) * nab..(rr * 3 + k + 1) * nab];
                                for (o, &v) in oa[k * nab..(k + 1) * nab].iter_mut().zip(row) {
                                    *o += v * a;
                                }
                            }
                        }
                        let e = pr.e_prim(ip);
                        for j in 0..natom {
                            for k in 0..3 {
                                let c = &accc[(3 * j + k) * nh0..(3 * j + k + 1) * nh0];
                                if c.iter().all(|&x| x == 0.0) {
                                    continue;
                                }
                                let o = &mut oc[(3 * j + k) * nab..(3 * j + k + 1) * nab];
                                for rr in 0..nh0 {
                                    let a = c[rr];
                                    for (x, &v) in o.iter_mut().zip(&e[rr * nab..(rr + 1) * nab]) {
                                        *x += v * a;
                                    }
                                }
                            }
                        }
                    }
                    Some((oa, oc))
                },
            )
            .collect();
        out.fill(0.0);
        let mut ob = Vec::new();
        for (pr, blk) in pairs.pairs().iter().zip(blocks) {
            let Some((oa, oc)) = blk else { continue };
            let (s1, s2) = (&sh[pr.sa], &sh[pr.sb]);
            let same = pr.sa == pr.sb;
            let nab = pr.nab();
            let put = |out: &mut [f64], x: usize, blk: &[f64]| {
                add_pair_block(
                    &mut out[x * n2..(x + 1) * n2],
                    nbf,
                    (s1.ao_start, s2.ao_start),
                    (pr.na, pr.nb),
                    same,
                    blk,
                );
            };
            for k in 0..3 {
                // B = -(A + sum_J C_J)
                ob.clear();
                ob.extend((0..nab).map(|x| {
                    let mut v = oa[k * nab + x];
                    for j in 0..natom {
                        v += oc[(3 * j + k) * nab + x];
                    }
                    -v
                }));
                put(out, 3 * s1.atom + k, &oa[k * nab..(k + 1) * nab]);
                put(out, 3 * s2.atom + k, &ob);
                for j in 0..natom {
                    let c = &oc[(3 * j + k) * nab..(3 * j + k + 1) * nab];
                    if c.iter().all(|&x| x == 0.0) {
                        continue;
                    }
                    put(out, 3 * j + k, c);
                }
            }
        }
    }

    /// The second derivatives of sum_g q_g sum_ab D_ab (ab|g) with respect to the atoms (shell
    /// centres and points, point g moving with atom owner\[g\]) into `out` ((3 natom) x
    /// (3 natom) row-major), overwritten. An order-Two plan.
    pub fn hessian(&self, q: &[f64], d: &[f64], out: &mut [f64], ws: &mut PotentialWorkspace) {
        assert_eq!(
            self.order(),
            DerivOrder::Two,
            "the Hessian needs pairs of derivative order two"
        );
        let (basis, pts) = (&self.plan.basis, &self.plan.pts[..]);
        let qsum = charge_bound(q, pts);
        let src = GaussianCharges {
            pts,
            q,
            slots: ChargeSlots::Owner(self.owner),
        };
        let screened = self.screened(d, qsum, self.plan.thresh * 1e-4);
        let plan = PotentialPlan::new(basis, screened, src, basis.natom(), DerivOrder::Two);
        plan.hessian(d, out, ws);
    }
}

/// The scratch of one task of [`SurfaceDerivPlan::matrices_deriv`]: the Hermite sums of the
/// raised index and of the points' derivatives per atom, a Hermite tensor and its work space.
struct MatDerivScratch {
    acc1: Vec<f64>,
    accc: Vec<f64>,
    r: Vec<f64>,
    work: HermiteWork,
}

impl MatDerivScratch {
    fn new(lab: usize, natom: usize) -> Self {
        MatDerivScratch {
            acc1: vec![0.0; nherm(lab + 1)],
            accc: vec![0.0; 3 * natom * nherm(lab)],
            r: vec![0.0; nherm(lab + 1)],
            work: HermiteWork::new(lab + 1),
        }
    }
}

/// Add a pair block (na x nb) for shells (i, j) into a symmetric nbf x nbf matrix: element
/// (a, b), and (b, a) as well when i != j.
fn add_pair_block(
    m: &mut [f64],
    nbf: usize,
    (a0, b0): (usize, usize),
    (na, nb): (usize, usize),
    same: bool,
    blk: &[f64],
) {
    for x in 0..na {
        for y in 0..nb {
            let v = blk[x * nb + y];
            m[(a0 + x) * nbf + b0 + y] += v;
            if !same {
                m[(b0 + y) * nbf + a0 + x] += v;
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Auxiliary-basis surface integrals (the density-fitted PCM)
// ---------------------------------------------------------------------------------------------
//
// With a density-fitted Coulomb matrix the surface terms can go through the auxiliary basis:
// (ab|g) ~ sum_PQ (ab|P) [V^-1]_PQ (Q|g), so the potential of a density at the points is
// sum_P c_P (P|g) with the fit coefficients c = V^-1 gamma, and the charges' Fock term is
// sum_P (ab|P) d_P with d = V^-1 sum_g (P|g) q_g. The two-centre integrals (P|g) of a solid
// harmonic Gaussian with a Gaussian charge have a closed form (Hobson's theorem: a harmonic
// polynomial S of degree l acting as S(grad_A) on f(|A - C|^2) gives S(A - C) (2 d/du)^l f):
//
//   (P|g) = S_P(A - C) sum_i c_i (2 pi / a_i) sqrt(mu_i / a_i) (-mu_i / a_i)^l F_l(mu_i |A - C|^2),
//   mu_i = a_i zeta^2 / (a_i + zeta^2)   (mu_i = a_i for a point charge),
//
// with S_P the harmonic polynomial of the function (rows of the shell's Cartesian-to-spherical
// transformation; s and p functions are harmonic as they are), c_i, a_i its contraction. No
// Hermite recursion: one Boys function per primitive and point.

/// The two-centre integrals (P|g) of the functions of an auxiliary basis (pure for l >= 2)
/// with the charges of a surface, in closed form (see above), and their gradient contracted
/// with charges and fit coefficients, into the caller's buffers.
pub struct AuxSurfacePlan<'a> {
    aux: &'a BasisSet,
    pts: &'a [GaussPoint],
    polys: Vec<AuxShellPoly<'a>>,
    /// the largest function count of a shell
    maxf: usize,
}

/// The caller-owned memory of an [`AuxSurfacePlan`]'s gradient: the chunk accumulators and a
/// function-derivative buffer per thread.
pub struct AuxSurfaceWorkspace {
    pool: Vec<f64>,
    scratch: Vec<Vec<[f64; 3]>>,
    maxf: usize,
}

impl<'a> AuxSurfacePlan<'a> {
    /// The plan of the integrals of the functions of `aux` with the charges at `pts`. Panics
    /// on a Cartesian shell of l >= 2.
    pub fn new(aux: &'a BasisSet, pts: &'a [GaussPoint]) -> Self {
        AuxSurfacePlan {
            aux,
            pts,
            polys: aux.shells.iter().map(AuxShellPoly::new).collect(),
            maxf: aux.shells.iter().map(Shell::nfunc).max().unwrap_or(0),
        }
    }

    /// (chunk count, accumulator length) of the gradient.
    fn chunks(&self) -> (usize, usize) {
        let np = self.pts.len();
        let len = 3 * (self.aux.natom() + np);
        (nchunk_for(np, len, DEFAULT_BUDGET), len)
    }

    /// The bytes of a gradient workspace: a buffer per thread, the accumulators shared.
    pub fn workspace_size(&self) -> WorkspaceSize {
        let (c, l) = self.chunks();
        WorkspaceSize {
            per_thread: 24 * self.maxf,
            shared: 8 * c * l,
        }
    }

    /// A workspace for the gradient.
    pub fn workspace(&self) -> AuxSurfaceWorkspace {
        let (c, l) = self.chunks();
        let nt = rayon::current_num_threads();
        AuxSurfaceWorkspace {
            pool: vec![0.0; c * l],
            scratch: (0..nt).map(|_| vec![[0.0; 3]; self.maxf]).collect(),
            maxf: self.maxf,
        }
    }

    /// (P|g) of every auxiliary function P with every point g into `out` (npts x naux,
    /// row-major, row g), overwritten. Rows are independent (deterministic).
    pub fn matrix(&self, out: &mut [f64]) {
        let naux = self.aux.nbf;
        assert_eq!(out.len(), self.pts.len() * naux, "(P|g) is npoints x naux");
        if naux == 0 {
            return;
        }
        out.par_chunks_mut(naux)
            .zip(self.pts.par_iter())
            .for_each(|(row, pt)| {
                let mut fm = [0.0; boys::MAX_M + 1];
                for p in &self.polys {
                    let f0 = p.sh.ao_start;
                    p.values(pt, &mut fm, &mut row[f0..f0 + p.sh.nfunc()]);
                }
            });
    }

    /// The gradient of sum_g sum_P q_g c_P (P|g) with respect to the atoms of the auxiliary
    /// shells (per atom, into `atoms`) and to the point positions (per point, into `points`),
    /// overwritten.
    pub fn gradient(
        &self,
        q: &[f64],
        c: &[f64],
        atoms: &mut [[f64; 3]],
        points: &mut [[f64; 3]],
        ws: &mut AuxSurfaceWorkspace,
    ) {
        let natom = self.aux.natom();
        let np = self.pts.len();
        assert_eq!(atoms.len(), natom, "the atoms' gradient has natom rows");
        assert_eq!(points.len(), np, "the points' gradient has npoints rows");
        let (nchunk, len) = self.chunks();
        let nt = rayon::current_num_threads();
        let maxf = ws.maxf;
        if ws.scratch.len() < nt {
            ws.scratch.resize_with(nt, || vec![[0.0; 3]; maxf]);
        }
        ordered_pool(
            np,
            &mut ws.pool,
            len,
            nchunk,
            &mut ws.scratch,
            |acc, dbuf, g| {
                if q[g] == 0.0 {
                    return;
                }
                let pt = &self.pts[g];
                let mut fm = [0.0; boys::MAX_M + 1];
                let mut gp = [0.0; 3];
                for p in &self.polys {
                    let (f0, nf) = (p.sh.ao_start, p.sh.nfunc());
                    let cs = &c[f0..f0 + nf];
                    if cs.iter().all(|&x| x == 0.0) {
                        continue;
                    }
                    let dv = &mut dbuf[..nf];
                    p.grads(pt, &mut fm, dv);
                    let mut ga = [0.0; 3];
                    for (x, &cv) in dv.iter().zip(cs) {
                        for k in 0..3 {
                            ga[k] += cv * x[k];
                        }
                    }
                    let a = p.sh.atom;
                    for k in 0..3 {
                        acc[3 * a + k] += q[g] * ga[k];
                        gp[k] -= q[g] * ga[k];
                    }
                }
                for k in 0..3 {
                    acc[3 * (natom + g) + k] += gp[k];
                }
            },
        );
        let acc = &ws.pool[..len];
        atoms.as_flattened_mut().copy_from_slice(&acc[..3 * natom]);
        points.as_flattened_mut().copy_from_slice(&acc[3 * natom..]);
    }
}

/// Data of one auxiliary shell for the closed-form kernels.
struct AuxShellPoly<'a> {
    sh: &'a crate::basis::Shell,
    /// (nfunc x ncart) harmonic polynomials over the Cartesian monomials `comps`
    t: &'static [f64],
    comps: Vec<[usize; 3]>,
}

impl<'a> AuxShellPoly<'a> {
    fn new(sh: &'a crate::basis::Shell) -> Self {
        assert!(
            sh.pure || sh.l < 2,
            "RI surface integrals need spherical (pure) auxiliary functions for l >= 2"
        );
        AuxShellPoly {
            sh,
            t: BasisSet::shell_transform_ref(sh),
            comps: crate::basis::cart_components(sh.l),
        }
    }

    /// Radial factor h(u) = sum_i c_i (2pi/a) sqrt(mu/a) (-mu/a)^l F_l(mu u) and its first
    /// `nder` derivatives with respect to u = |A - C|^2.
    #[inline]
    fn radial(&self, zeta: f64, u: f64, nder: usize, fm: &mut [f64], h: &mut [f64; 3]) {
        let l = self.sh.l;
        let z2 = zeta * zeta;
        *h = [0.0; 3];
        for (&a, &c) in self.sh.exps.iter().zip(&self.sh.coefs) {
            let mu = if z2.is_finite() { a * z2 / (a + z2) } else { a };
            let r = mu / a;
            let pref = c * 2.0 * PI / a * r.sqrt() * (-r).powi(l as i32);
            boys::boys(l + nder, mu * u, fm);
            h[0] += pref * fm[l];
            if nder >= 1 {
                h[1] -= pref * mu * fm[l + 1];
            }
            if nder >= 2 {
                h[2] += pref * mu * mu * fm[l + 2];
            }
        }
    }

    /// (P|g) for every function P of the shell: out[P].
    #[inline]
    fn values(&self, pt: &GaussPoint, fm: &mut [f64], out: &mut [f64]) {
        let a = self.sh.center;
        let r = [a[0] - pt.r[0], a[1] - pt.r[1], a[2] - pt.r[2]];
        let u = r[0] * r[0] + r[1] * r[1] + r[2] * r[2];
        let mut h = [0.0; 3];
        self.radial(pt.zeta, u, 0, fm, &mut h);
        let l = self.sh.l;
        let (px, py, pz) = (pows(r[0], l), pows(r[1], l), pows(r[2], l));
        let nc = self.comps.len();
        for (f, o) in out.iter_mut().enumerate() {
            let row = &self.t[f * nc..(f + 1) * nc];
            let mut s = 0.0;
            for (k, c) in self.comps.iter().enumerate() {
                if row[k] != 0.0 {
                    s += row[k] * px[c[0]] * py[c[1]] * pz[c[2]];
                }
            }
            *o = h[0] * s;
        }
    }

    /// d/dA (P|g) for every function P (the point's derivative is its negative):
    /// out[P] = [d/dAx, d/dAy, d/dAz].
    #[inline]
    fn grads(&self, pt: &GaussPoint, fm: &mut [f64], out: &mut [[f64; 3]]) {
        let a = self.sh.center;
        let r = [a[0] - pt.r[0], a[1] - pt.r[1], a[2] - pt.r[2]];
        let u = r[0] * r[0] + r[1] * r[1] + r[2] * r[2];
        let mut h = [0.0; 3];
        self.radial(pt.zeta, u, 1, fm, &mut h);
        let l = self.sh.l;
        let (px, py, pz) = (pows(r[0], l), pows(r[1], l), pows(r[2], l));
        let nc = self.comps.len();
        for (f, o) in out.iter_mut().enumerate() {
            let row = &self.t[f * nc..(f + 1) * nc];
            // S and grad S at r
            let (mut s, mut ds) = (0.0, [0.0; 3]);
            for (k, c) in self.comps.iter().enumerate() {
                let t = row[k];
                if t == 0.0 {
                    continue;
                }
                let (x, y, z) = (px[c[0]], py[c[1]], pz[c[2]]);
                s += t * x * y * z;
                if c[0] > 0 {
                    ds[0] += t * c[0] as f64 * px[c[0] - 1] * y * z;
                }
                if c[1] > 0 {
                    ds[1] += t * c[1] as f64 * x * py[c[1] - 1] * z;
                }
                if c[2] > 0 {
                    ds[2] += t * c[2] as f64 * x * y * pz[c[2] - 1];
                }
            }
            for k in 0..3 {
                o[k] = h[0] * ds[k] + 2.0 * r[k] * h[1] * s;
            }
        }
    }
}

/// [1, x, x^2, ..., x^l] (l <= 16).
#[inline]
fn pows(x: f64, l: usize) -> [f64; 17] {
    let mut p = [0.0; 17];
    p[0] = 1.0;
    for i in 1..=l {
        p[i] = p[i - 1] * x;
    }
    p
}

/// Raised (l + 1) and lowered (l - 1) Cartesian shells of a contracted shell for first
/// derivatives with respect to its centre: per primitive d/dA_x x_A^k e^{-a r_A^2} =
/// 2a x_A^{k+1} e^{..} - k x_A^{k-1} e^{..}, so with the raised shell's coefficients 2 a_i c_i
/// and the lowered shell's c_i (Cartesian, the shell's x^l normalization kept),
/// d/dA_x phi_mu = sum_u RU_x[mu][u] U_u + sum_w RL_x[mu][w] L_w (rows mu: the shell's functions).
struct DerivShells {
    up: crate::basis::Shell,
    down: Option<crate::basis::Shell>,
    nup: usize,
    ndown: usize,
    /// per direction: nfunc x nup, nfunc x ndown (row-major)
    ru: [Vec<f64>; 3],
    rl: [Vec<f64>; 3],
}

impl DerivShells {
    fn new(s: &crate::basis::Shell) -> Self {
        use crate::basis::{cart_components, ncart};
        let l = s.l;
        let nf = s.nfunc();
        let t = BasisSet::shell_transform_ref(s);
        let nc = ncart(l);
        let comps = cart_components(l);
        let up_c = cart_components(l + 1);
        let dn_c = if l > 0 {
            cart_components(l - 1)
        } else {
            Vec::new()
        };
        let (nup, ndown) = (up_c.len(), dn_c.len());
        let mut ru = [
            vec![0.0; nf * nup],
            vec![0.0; nf * nup],
            vec![0.0; nf * nup],
        ];
        let mut rl = [
            vec![0.0; nf * ndown],
            vec![0.0; nf * ndown],
            vec![0.0; nf * ndown],
        ];
        for mu in 0..nf {
            for (k, e) in comps.iter().enumerate() {
                let tv = t[mu * nc + k];
                if tv == 0.0 {
                    continue;
                }
                for x in 0..3 {
                    let mut eu = *e;
                    eu[x] += 1;
                    let iu = up_c
                        .iter()
                        .position(|c| *c == eu)
                        .expect("raised component");
                    ru[x][mu * nup + iu] += tv;
                    if e[x] > 0 {
                        let mut ed = *e;
                        ed[x] -= 1;
                        let id = dn_c
                            .iter()
                            .position(|c| *c == ed)
                            .expect("lowered component");
                        rl[x][mu * ndown + id] -= e[x] as f64 * tv;
                    }
                }
            }
        }
        let up = crate::basis::Shell {
            l: l + 1,
            pure: false,
            atom: s.atom,
            center: s.center,
            exps: s.exps.clone(),
            coefs: s
                .exps
                .iter()
                .zip(&s.coefs)
                .map(|(a, c)| 2.0 * a * c)
                .collect(),
            ao_start: 0,
        };
        let down = (l > 0).then(|| crate::basis::Shell {
            l: l - 1,
            pure: false,
            atom: s.atom,
            center: s.center,
            exps: s.exps.clone(),
            coefs: s.coefs.clone(),
            ao_start: 0,
        });
        DerivShells {
            up,
            down,
            nup,
            ndown,
            ru,
            rl,
        }
    }
}

/// McMurchie-Davidson potentials of one pair at all points: acc\[k * np + g\] += f sum_ab
/// D^k_ab (ab|g) for densities of rows of `ld` values with the pair's block at rows a0..,
/// columns b0.. (f: 2 for the two triangles of distinct shells, else 1).
fn md_pair_points(
    pr: &ShellPair,
    pts: &[GaussPoint],
    ds: &[&[f64]],
    (ld, a0, b0): (usize, usize, usize),
    f: f64,
    acc: &mut [f64],
) {
    let np = pts.len();
    let nd = ds.len();
    let lab = pr.lab();
    let nh = nherm(lab);
    let mut hd = vec![0.0; nd * nh];
    let mut r = vec![0.0; nh];
    let mut hw = HermiteWork::new(lab);
    for ip in 0..pr.nprim {
        for (k, d) in ds.iter().enumerate() {
            herm_dens(
                pr.e_prim(ip),
                nh,
                1,
                d,
                ld,
                a0,
                b0,
                pr.na,
                pr.nb,
                f,
                &mut hd[k * nh..(k + 1) * nh],
            );
        }
        for (g, pt) in pts.iter().enumerate() {
            coulomb_hermite(
                pt.r,
                pt.zeta,
                1.0,
                &mut HermiteCtx {
                    l: lab,
                    p: pr.p[ip],
                    pc: pr.pc[ip],
                    r: &mut r,
                    work: &mut hw,
                },
            );
            for k in 0..nd {
                acc[k * np + g] += hd[k * nh..(k + 1) * nh]
                    .iter()
                    .zip(&r)
                    .map(|(x, y)| x * y)
                    .sum::<f64>();
            }
        }
    }
}

/// McMurchie-Davidson potential-matrix block of one pair for charge sets qs, added into `out`
/// ([k][ab], zeroed by the caller).
fn md_pair_matrix(pr: &ShellPair, pts: &[GaussPoint], qs: &[&[f64]], out: &mut [f64]) {
    let nd = qs.len();
    let lab = pr.lab();
    let nh = nherm(lab);
    let nab = pr.nab();
    let mut acc = vec![0.0; nd * nh];
    let mut r = vec![0.0; nh];
    let mut hw = HermiteWork::new(lab);
    for ip in 0..pr.nprim {
        acc.fill(0.0);
        for (g, pt) in pts.iter().enumerate() {
            coulomb_hermite(
                pt.r,
                pt.zeta,
                1.0,
                &mut HermiteCtx {
                    l: lab,
                    p: pr.p[ip],
                    pc: pr.pc[ip],
                    r: &mut r,
                    work: &mut hw,
                },
            );
            for k in 0..nd {
                let q = qs[k][g];
                for (a, &y) in acc[k * nh..(k + 1) * nh].iter_mut().zip(&r) {
                    *a += q * y;
                }
            }
        }
        let e = pr.e_prim(ip);
        for k in 0..nd {
            for rr in 0..nh {
                let a = acc[k * nh + rr];
                for (x, &ev) in out[k * nab..(k + 1) * nab]
                    .iter_mut()
                    .zip(&e[rr * nab..(rr + 1) * nab])
                {
                    *x += ev * a;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::basis::BasisDef;
    use crate::operator::Operator;
    use crate::pairs::{DerivOrder, PairOptions, PairSet};
    use crate::reference::oneshot;
    use crate::reference::pcm::{potential_matrices, potential_points};

    fn water(dx: f64) -> BasisSet {
        let def = BasisDef::load("def2-svp", None).unwrap();
        let z = [8u32, 1, 1];
        let c = [
            [0.0 + dx, 0.0, 0.12],
            [0.0, 1.43, -0.98],
            [0.0, -1.43, -0.98],
        ];
        BasisSet::new(&def, &z, &c, true).unwrap()
    }

    fn points() -> Vec<GaussPoint> {
        vec![
            GaussPoint {
                r: [2.9, 0.4, 0.3],
                zeta: 3.1,
            },
            GaussPoint {
                r: [-0.5, 3.4, -1.0],
                zeta: 5.7,
            },
            GaussPoint {
                r: [0.2, -0.3, 3.3],
                zeta: 1.9,
            },
            GaussPoint {
                r: [0.1, -3.1, -2.5],
                zeta: 4.4,
            },
        ]
    }

    fn density(n: usize) -> Vec<f64> {
        let mut d = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..=i {
                let v = 0.3 * (((i * 7 + j * 3) % 11) as f64 / 11.0 - 0.4)
                    / (1.0 + (i as f64 - j as f64).abs());
                d[i * n + j] = v;
                d[j * n + i] = v;
            }
        }
        d
    }

    /// The Gaussian-charge integrals of the McMurchie-Davidson reference agree with the
    /// independent Rys erf(omega r)/r kernels of grid_pot (omega = zeta), and the matrix and
    /// point contractions are consistent.
    #[test]
    fn matches_rys_erf_kernels() {
        let b = water(0.0);
        let pairs = PairSet::build(&b, PairOptions::default());
        let pts = points();
        let n = b.nbf;
        let d = density(n);
        let v = potential_points(&b, &pairs, &pts, &[&d]);
        let charges = [0.7, -0.4, 1.3, -0.2];
        let m = potential_matrices(&b, &pairs, &pts, &[&charges]);
        // reference from grid_pot, one point at a time
        let mut vref = vec![0.0; pts.len()];
        let mut mref = vec![0.0; n * n];
        for i in 0..b.shells.len() {
            for j in 0..=i {
                let pp = PotPair::new(&b, i, j, 0.0);
                let (a0, b0) = (b.shells[i].ao_start, b.shells[j].ao_start);
                for (g, pt) in pts.iter().enumerate() {
                    let mut out = vec![0.0; pp.nab()];
                    pp.compute(
                        &Operator::LongRange { omega: pt.zeta },
                        &[pt.r[0]],
                        &[pt.r[1]],
                        &[pt.r[2]],
                        0.0,
                        &mut out,
                    );
                    for x in 0..pp.na {
                        for y in 0..pp.nb {
                            let val = out[x * pp.nb + y];
                            let f = if i == j { 1.0 } else { 2.0 };
                            vref[g] += f * d[(a0 + x) * n + b0 + y] * val;
                            mref[(a0 + x) * n + b0 + y] += charges[g] * val;
                            if i != j {
                                mref[(b0 + y) * n + a0 + x] += charges[g] * val;
                            }
                        }
                    }
                }
            }
        }
        for g in 0..pts.len() {
            assert!(
                (v[0][g] - vref[g]).abs() < 1e-12,
                "v[{g}] {} vs {}",
                v[0][g],
                vref[g]
            );
        }
        let err = m[0]
            .iter()
            .zip(&mref)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        assert!(err < 1e-12, "matrix error {err}");
        // sum_g q_g v_g = <D, V(q)>
        let lhs: f64 = v[0].iter().zip(&charges).map(|(a, b)| a * b).sum();
        let rhs: f64 = d.iter().zip(&m[0]).map(|(a, b)| a * b).sum();
        assert!((lhs - rhs).abs() < 1e-12);
    }

    fn maxdiff(a: &[f64], b: &[f64]) -> f64 {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f64::max)
    }

    /// The batched Rys plan agrees with the McMurchie-Davidson reference.
    #[test]
    fn surface_plan_matches_reference() {
        let b = water(0.0);
        let pairs = PairSet::build(&b, PairOptions::default());
        let mut pts = points();
        // repeated exponents (groups) as on a real surface
        pts.push(GaussPoint {
            r: [1.7, 2.2, -0.4],
            zeta: 3.1,
        });
        pts.push(GaussPoint {
            r: [-2.6, -0.4, 1.1],
            zeta: 5.7,
        });
        let n = b.nbf;
        let np = pts.len();
        let d = density(n);
        let d2: Vec<f64> = d.iter().map(|x| 0.5 * x * x - 0.1 * x).collect();
        let charges: Vec<f64> = (0..np).map(|i| 0.3 * i as f64 - 0.7).collect();
        let fast = SurfacePlan::new(&b, &pairs, &pts, 0.0);
        assert_eq!(fast.ngroups(), 4);
        let v_ref = potential_points(&b, &pairs, &pts, &[&d, &d2]);
        let v = oneshot::surface_potentials(&fast, &[&d, &d2]);
        for k in 0..2 {
            assert!(
                maxdiff(&v[k * np..(k + 1) * np], &v_ref[k]) < 1e-12,
                "v {k}"
            );
        }
        let m_ref = potential_matrices(&b, &pairs, &pts, &[&charges]);
        let m = oneshot::surface_matrices(&fast, &[&charges]);
        assert!(maxdiff(&m[0], &m_ref[0]) < 1e-12, "matrix");
        // the fused pass gives both at once, with the same bits as the separate ones
        let (vf, mf) = oneshot::surface_potentials_and_matrices(&fast, &[&d, &d2], &[&charges]);
        assert!(vf.iter().zip(&v).all(|(x, y)| x.to_bits() == y.to_bits()));
        assert!(mf[0]
            .iter()
            .zip(&m[0])
            .all(|(x, y)| x.to_bits() == y.to_bits()));
        // point charges (infinite exponent): the Coulomb kernels against the MD reference
        let ppts: Vec<GaussPoint> = pts
            .iter()
            .map(|p| GaussPoint {
                r: p.r,
                zeta: f64::INFINITY,
            })
            .collect();
        let pf = SurfacePlan::new(&b, &pairs, &ppts, 0.0);
        let vp = oneshot::surface_potentials(&pf, &[&d]);
        let vp_ref = potential_points(&b, &pairs, &ppts, &[&d]);
        assert!(maxdiff(&vp, &vp_ref[0]) < 1e-12, "point-charge v");
    }

    /// The plans' results have the same bits for any thread count and with a reused
    /// workspace, screened (threshold 1e-10) and on a larger basis.
    #[test]
    fn plans_independent_of_threads_and_reuse() {
        let def = BasisDef::load("def2-tzvp", None).unwrap();
        let c = [[0.0, 0.0, 0.12], [0.0, 1.43, -0.98], [0.0, -1.43, -0.98]];
        let b = BasisSet::new(&def, &[8, 1, 1], &c, true).unwrap();
        let pairs = PairSet::build(&b, PairOptions::with_order(DerivOrder::Two));
        let mut pts = points();
        for i in 0..300 {
            let t = i as f64 * 0.37;
            pts.push(GaussPoint {
                r: [3.0 * t.cos(), 3.0 * t.sin(), 2.5 * (0.5 * t).cos()],
                zeta: [3.1, 5.7, 4.4][i % 3],
            });
        }
        let np = pts.len();
        let owner: Vec<usize> = (0..np).map(|g| g % 3).collect();
        let d = density(b.nbf);
        let q: Vec<f64> = (0..np)
            .map(|i| ((i * 13 % 7) as f64 - 3.0) * 0.01)
            .collect();
        let run = || {
            let plan = SurfacePlan::new(&b, &pairs, &pts, 1e-10);
            let shape = SurfaceShape {
                densities: 1,
                charge_sets: 1,
                gradient: true,
            };
            let mut ws = plan.workspace(shape);
            let n = b.nbf;
            let mut out: Vec<Vec<f64>> = Vec::new();
            for _ in 0..2 {
                let mut v = vec![0.0; np];
                plan.potentials(&[&d], &mut v, &mut ws);
                let mut m = vec![0.0; n * n];
                plan.matrices(&[&q], &mut [AoMatrixMut::row_major(&mut m, n)], &mut ws);
                let (mut vf, mut mf) = (vec![0.0; np], vec![0.0; n * n]);
                let mut mv = [AoMatrixMut::row_major(&mut mf, n)];
                plan.potentials_and_matrices((&[&d], &[&q]), &mut vf, &mut mv, &mut ws);
                let (mut ga, mut gp) = (vec![[0.0; 3]; 3], vec![[0.0; 3]; np]);
                plan.gradient(&q, &d, &mut ga, &mut gp, &mut ws);
                let dp = plan.deriv_plan(&pairs, &owner).unwrap();
                let mut dws = dp.workspace();
                let (mut ga2, mut gp2) = (vec![[0.0; 3]; 3], vec![[0.0; 3]; np]);
                dp.gradient(&q, &d, &mut ga2, &mut gp2, &mut dws);
                let mut dv = vec![0.0; 9 * np];
                dp.potentials_deriv(&d, &mut dv, &mut dws);
                let mut dm = vec![0.0; 9 * n * n];
                dp.matrices_deriv(&q, &mut dm);
                let mut h = vec![0.0; 81];
                dp.hessian(&q, &d, &mut h, &mut dws);
                let flat = |g: &[[f64; 3]]| g.iter().flatten().copied().collect::<Vec<f64>>();
                out = vec![
                    v,
                    m,
                    vf,
                    mf,
                    flat(&ga),
                    flat(&gp),
                    flat(&ga2),
                    flat(&gp2),
                    dv,
                    dm,
                    h,
                ];
            }
            out
        };
        let pool = |t: usize| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(t)
                .build()
                .unwrap()
        };
        let r1 = pool(1).install(run);
        assert!(r1[0].iter().any(|&x| x != 0.0) && r1[10].iter().any(|&x| x != 0.0));
        // the Rys and McMurchie-Davidson gradients agree
        assert!(maxdiff(&r1[4], &r1[6]) < 1e-10 && maxdiff(&r1[5], &r1[7]) < 1e-10);
        for t in [3, 8] {
            let rt = pool(t).install(run);
            for (k, (a, b)) in r1.iter().zip(&rt).enumerate() {
                assert!(
                    a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits()),
                    "result {k} at {t} threads"
                );
            }
        }
    }

    /// The Rys gradient (raised/lowered shells) against the McMurchie-Davidson one: Gaussian
    /// and point charges, def2-SVP and def2-QZVP (g functions: the raised h shell takes the
    /// McMurchie-Davidson path).
    #[test]
    fn surface_grad_matches_reference() {
        for basis in ["def2-svp", "def2-qzvp"] {
            let def = BasisDef::load(basis, None).unwrap();
            let c = [[0.0, 0.0, 0.12], [0.0, 1.43, -0.98], [0.0, -1.43, -0.98]];
            let b = BasisSet::new(&def, &[8, 1, 1], &c, true).unwrap();
            let pairs = PairSet::build(&b, PairOptions::with_order(DerivOrder::One));
            let d = density(b.nbf);
            let mut pts = points();
            pts.push(GaussPoint {
                r: [1.7, 2.2, -0.4],
                zeta: 3.1,
            });
            let q: Vec<f64> = (0..pts.len()).map(|i| 0.3 * i as f64 - 0.7).collect();
            let owner = vec![0; pts.len()];
            for set in [
                pts.clone(),
                pts.iter()
                    .map(|p| GaussPoint {
                        r: p.r,
                        zeta: f64::INFINITY,
                    })
                    .collect::<Vec<_>>(),
            ] {
                let si = SurfacePlan::new(&b, &pairs, &set, 0.0);
                let dp = si.deriv_plan(&pairs, &owner).unwrap();
                let (ga0, gc0) = oneshot::surface_deriv_gradient(&dp, &q, &d);
                let (ga, gc) = oneshot::surface_gradient(&si, &q, &d);
                let scale = ga0
                    .iter()
                    .chain(&gc0)
                    .flat_map(|v| v.iter())
                    .fold(0.0f64, |m, x| m.max(x.abs()));
                for (x, y) in ga.iter().zip(&ga0).chain(gc.iter().zip(&gc0)) {
                    for k in 0..3 {
                        assert!(
                            (x[k] - y[k]).abs() < 1e-11 * scale.max(1.0),
                            "{basis}: {} vs {}",
                            x[k],
                            y[k]
                        );
                    }
                }
            }
        }
    }

    /// Auxiliary basis on water (def2-universal-jkfit: up to l = 4 on O) at displacement dx of O.
    fn water_aux(dx: f64) -> BasisSet {
        let def = BasisDef::load("def2-universal-jkfit", None).unwrap();
        let c = [
            [0.0 + dx, 0.0, 0.12],
            [0.0, 1.43, -0.98],
            [0.0, -1.43, -0.98],
        ];
        BasisSet::new(&def, &[8, 1, 1], &c, true).unwrap()
    }

    /// The closed-form (P|g) against generic two-centre ERIs with each Gaussian charge as a
    /// normalized s function; point charges against a very narrow Gaussian.
    #[test]
    fn aux_point_integrals_match_eri_2c() {
        let aux = water_aux(0.0);
        let naux = aux.nbf;
        assert!(aux.max_l() >= 4);
        let mut pts = points();
        pts.push(GaussPoint {
            r: [0.4, 0.3, 0.2],
            zeta: 2.5,
        }); // inside the density
        let point_charges: Vec<GaussPoint> = pts
            .iter()
            .map(|p| GaussPoint {
                r: p.r,
                zeta: f64::INFINITY,
            })
            .collect();
        // point charges against a very narrow Gaussian: the reduced exponent a zeta^2 / (a + zeta^2)
        // differs from a by a / zeta^2 (relative), below 1e-9 for every auxiliary exponent here
        for (set, zref, tol) in [(&pts, None, 1e-11), (&point_charges, Some(1e5), 1e-8)] {
            let m = oneshot::aux_point_matrix(&aux, set);
            let mut shells = aux.shells.clone();
            let mut nbf = naux;
            for p in set.iter() {
                let z = zref.unwrap_or(p.zeta);
                let a = z * z;
                shells.push(crate::basis::Shell {
                    l: 0,
                    pure: false,
                    atom: 0,
                    center: p.r,
                    exps: vec![a],
                    coefs: crate::basis::normalize_shell(0, &[a], &[1.0]),
                    ao_start: nbf,
                });
                nbf += 1;
            }
            let comb = BasisSet {
                name: "x".into(),
                shells,
                nbf,
                atom_shells: aux.atom_shells.clone(),
                pure: true,
            };
            let ap = PairSet::aux(&comb, PairOptions::default());
            let v = crate::reference::eri_2c(&comb, &ap);
            let mut worst: f64 = 0.0;
            for (g, p) in set.iter().enumerate() {
                let a = zref.unwrap_or(p.zeta).powi(2);
                // normalized s function -> unit-charge Gaussian
                let f = (a / PI).powf(1.5) / (2.0 * a / PI).powf(0.75);
                for q in 0..naux {
                    let r = f * v[q * nbf + naux + g];
                    worst = worst.max((m[g * naux + q] - r).abs() / r.abs().max(1e-3));
                }
            }
            assert!(worst < tol, "max relative deviation {worst:e}");
        }
    }

    /// The gradient of sum q c (P|g) against central differences (auxiliary shells and points
    /// moving with their atoms).
    #[test]
    fn aux_point_gradient_by_finite_differences() {
        let aux0 = water_aux(0.0);
        let naux = aux0.nbf;
        let natom = 3;
        let pts0 = points();
        let owner = [0usize, 1, 2, 0];
        let q = [0.7, -0.4, 1.3, -0.2];
        let c: Vec<f64> = (0..naux)
            .map(|i| ((i * 37 % 17) as f64 / 17.0 - 0.45) * 0.3)
            .collect();
        let coords0: Vec<[f64; 3]> = aux0
            .atom_shells
            .iter()
            .map(|s| aux0.shells[s.start].center)
            .collect();
        let at = |x: usize, h: f64| -> (BasisSet, Vec<GaussPoint>) {
            let mut cc = coords0.clone();
            cc[x / 3][x % 3] += h;
            let mut b = aux0.clone();
            b.set_positions(&cc);
            let mut p = pts0.clone();
            for (g, pt) in p.iter_mut().enumerate() {
                if owner[g] == x / 3 {
                    pt.r[x % 3] += h;
                }
            }
            (b, p)
        };
        let e = |b: &BasisSet, p: &[GaussPoint]| -> f64 {
            let m = oneshot::aux_point_matrix(b, p);
            (0..p.len())
                .map(|g| q[g] * (0..naux).map(|k| m[g * naux + k] * c[k]).sum::<f64>())
                .sum()
        };
        let grad = |b: &BasisSet, p: &[GaussPoint]| -> Vec<f64> {
            let (ga, gc) = oneshot::aux_point_grad(b, p, &q, &c);
            let mut g = ga;
            for (i, v) in gc.iter().enumerate() {
                for k in 0..3 {
                    g[owner[i]][k] += v[k];
                }
            }
            g.iter().flat_map(|v| v.iter().copied()).collect()
        };
        let g0 = grad(&aux0, &pts0);
        assert_eq!(g0.len(), 3 * natom);
        let h = 1e-4;
        for (x, &gx) in g0.iter().enumerate() {
            let (bp, pp) = at(x, h);
            let (bm, pm) = at(x, -h);
            let fd = (e(&bp, &pp) - e(&bm, &pm)) / (2.0 * h);
            assert!((fd - gx).abs() < 1e-8, "grad {x}: {fd} vs {gx}");
        }
    }

    /// Gradient, point-potential derivatives, matrix derivatives and Hessian of the derivative
    /// plan against central differences of the McMurchie-Davidson reference.
    #[test]
    fn derivatives_by_finite_differences() {
        let b0 = water(0.0);
        let n = b0.nbf;
        let d = density(n);
        let pts0 = points();
        let owner = [0usize, 1, 2, 0];
        let charges = [0.7, -0.4, 1.3, -0.2];
        let natom = 3;
        let coords0: Vec<[f64; 3]> = b0
            .atom_shells
            .iter()
            .map(|s| b0.shells[s.start].center)
            .collect();
        // energy-like functional at displaced coordinates (points move with owners)
        let at = |x: usize, h: f64| -> (BasisSet, Vec<GaussPoint>) {
            let mut c = coords0.clone();
            c[x / 3][x % 3] += h;
            let mut b = b0.clone();
            b.set_positions(&c);
            let mut p = pts0.clone();
            for (g, pt) in p.iter_mut().enumerate() {
                if owner[g] == x / 3 {
                    pt.r[x % 3] += h;
                }
            }
            (b, p)
        };
        let e = |b: &BasisSet, p: &[GaussPoint]| -> f64 {
            let pairs = PairSet::build(b, PairOptions::default());
            potential_points(b, &pairs, p, &[&d])[0]
                .iter()
                .zip(&charges)
                .map(|(a, c)| a * c)
                .sum()
        };
        let pairs = PairSet::build(&b0, PairOptions::with_order(DerivOrder::Two));
        let sp = SurfacePlan::new(&b0, &pairs, &pts0, 0.0);
        let dp = sp.deriv_plan(&pairs, &owner).unwrap();
        let (ga, gc) = oneshot::surface_deriv_gradient(&dp, &charges, &d);
        let mut g = ga.clone();
        for (i, c) in gc.iter().enumerate() {
            for k in 0..3 {
                g[owner[i]][k] += c[k];
            }
        }
        let dvp = oneshot::surface_potentials_deriv(&dp, &d);
        let dmat = oneshot::surface_matrices_deriv(&dp, &charges);
        let hess = oneshot::surface_hessian(&dp, &charges, &d);
        let h = 1e-4;
        for x in 0..3 * natom {
            let (bp, pp) = at(x, h);
            let (bm, pm) = at(x, -h);
            let fd = (e(&bp, &pp) - e(&bm, &pm)) / (2.0 * h);
            assert!(
                (fd - g[x / 3][x % 3]).abs() < 1e-7,
                "grad {x}: {fd} vs {}",
                g[x / 3][x % 3]
            );
            // point potentials
            let vp = {
                let pr = PairSet::build(&bp, PairOptions::default());
                potential_points(&bp, &pr, &pp, &[&d]).remove(0)
            };
            let vm = {
                let pr = PairSet::build(&bm, PairOptions::default());
                potential_points(&bm, &pr, &pm, &[&d]).remove(0)
            };
            for gi in 0..pts0.len() {
                let fd = (vp[gi] - vm[gi]) / (2.0 * h);
                assert!((fd - dvp[x * pts0.len() + gi]).abs() < 1e-7, "dv {x} {gi}");
            }
            // matrices
            let mp = {
                let pr = PairSet::build(&bp, PairOptions::default());
                potential_matrices(&bp, &pr, &pp, &[&charges]).remove(0)
            };
            let mm = {
                let pr = PairSet::build(&bm, PairOptions::default());
                potential_matrices(&bm, &pr, &pm, &[&charges]).remove(0)
            };
            let err = (0..n * n)
                .map(|i| ((mp[i] - mm[i]) / (2.0 * h) - dmat[x * n * n + i]).abs())
                .fold(0.0, f64::max);
            assert!(err < 1e-7, "dV {x}: {err}");
            // Hessian row from gradient differences
            let grad_at = |b: &BasisSet, p: &[GaussPoint]| -> Vec<f64> {
                let pr = PairSet::build(b, PairOptions::with_order(DerivOrder::One));
                let sp = SurfacePlan::new(b, &pr, p, 0.0);
                let dp = sp.deriv_plan(&pr, &owner).unwrap();
                let (ga, gc) = oneshot::surface_deriv_gradient(&dp, &charges, &d);
                let mut g = ga;
                for (i, c) in gc.iter().enumerate() {
                    for k in 0..3 {
                        g[owner[i]][k] += c[k];
                    }
                }
                g.iter().flat_map(|v| v.iter().copied()).collect()
            };
            let gp = grad_at(&bp, &pp);
            let gm = grad_at(&bm, &pm);
            for y in 0..3 * natom {
                let fd = (gp[y] - gm[y]) / (2.0 * h);
                assert!(
                    (fd - hess[x * 3 * natom + y]).abs() < 1e-6,
                    "hess {x} {y}: {fd} vs {}",
                    hess[x * 3 * natom + y]
                );
            }
        }
    }
}
