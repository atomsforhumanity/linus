//! Potentials of charge distributions between basis functions, sum_C <a| v_C |b>, for point
//! charges (nuclei, external charges), spherical Gaussian charges (the surface charges of
//! continuum solvation) and the local channel of effective core potentials, with their first
//! and second nuclear derivatives.
//!
//! In the McMurchie-Davidson form every such integral is sum_t E_t^{ab} R_t(P - C) over the
//! Hermite expansion E of the shell pair: only the Hermite tensor R of a center depends on the
//! kind of charge. A [`ChargeSource`] supplies it; a [`PotentialPlan`] over a source and the
//! shell pairs it visits ([`Pairs`]) does the rest (the pair loop, the Hermite densities of the
//! pair, the contraction, translational invariance, the thread-count independent reduction),
//! into the caller's buffers over a caller-owned [`PotentialWorkspace`]:
//! - [`PotentialPlan::gradient`]: d/dX sum_ab D_ab V_ab for the shell centers and the charge
//!   centers;
//! - [`PotentialPlan::hessian`]: the second derivatives;
//! - [`PotentialPlan::add_deriv_matrices`]: the derivative matrices d/dX V_ab.
//!
//! The one-electron plan ([`crate::one_e::OneEPlan`]) runs it for point charges (the nuclear
//! attraction), the ECP plan for the local channels, the surface integrals of continuum
//! solvation for Gaussian charges. The raw blocks of one pair are [`matrix_block`] (the
//! potential itself), [`deriv_pair_blocks`] and [`deriv_b_blocks`]. A potential matrix and the
//! d/dA derivative blocks share one summation order: per primitive pair the centers' Hermite
//! tensors are summed first, then contracted once with the pair's expansion.
//!
//! The derivatives with respect to the second shell's center follow from translational
//! invariance (d/dB = -(d/dA + d/dC)), and a center's own derivative raises its Hermite index
//! (d/dC R_t(P - C) = -R_{t+1}). Each center moves with a coordinate slot
//! ([`ChargeSource::slot`]): an atom (a nucleus, an ECP center, a surface charge with its owner
//! atom), or a slot of its own after the atoms (an external point charge).

use crate::basis::BasisSet;
use crate::core::boys;
use crate::core::deriv::{d2_idx, shift1, PairDeriv};
use crate::core::hermite::{nherm, r_tensor, ShellPair};
use crate::engine::schedule::ordered_pool;
use crate::engine::WorkspaceSize;
use crate::one_e::deriv::herm_dens;
use crate::one_e::ecp::{local_r, EcpCenter};
use crate::pairs::{DerivOrder, PairSet};
use crate::points::pcm_ints::GaussPoint;
use linus_num::det::{nchunk_for, DEFAULT_BUDGET};
use rayon::prelude::*;
use std::f64::consts::PI;

/// Scratch of a center's Hermite tensor (Boys values, recursion and partial-sum buffers).
pub struct HermiteWork {
    fm: [f64; boys::MAX_M + 1],
    rbuf: Vec<f64>,
    work: Vec<f64>,
}

impl HermiteWork {
    /// Scratch for Hermite tensors up to degree `l`.
    pub fn new(l: usize) -> Self {
        let nh = nherm(l);
        HermiteWork {
            fm: [0.0; boys::MAX_M + 1],
            rbuf: Vec::new(),
            work: vec![0.0; 2 * nh],
        }
    }

    /// [`HermiteWork::new`] with every buffer at its full length (the ECP channels' too), so
    /// no source grows it.
    pub(crate) fn reserved(l: usize) -> Self {
        HermiteWork {
            rbuf: vec![0.0; nherm(l)],
            ..Self::new(l)
        }
    }
}

/// What a center's Hermite tensor is computed for: a primitive pair of exponent `p` centered
/// at `pc`, up to degree `l`, written into `r` (nherm(l) values).
pub struct HermiteCtx<'a> {
    pub l: usize,
    pub p: f64,
    pub pc: [f64; 3],
    pub r: &'a mut [f64],
    pub work: &'a mut HermiteWork,
}

/// A set of charge centers: what the potential engines need to know about them.
pub trait ChargeSource: Sync {
    /// The number of centers.
    fn len(&self) -> usize;

    /// Whether there are no centers.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The coordinate slot center `c` moves with (see the module docs).
    fn slot(&self, c: usize) -> usize;

    /// R_t(P - C) of center `c`, its charge included, into `ctx.r`; false when the center's
    /// contribution is exactly zero (it is then skipped).
    fn hermite(&self, c: usize, ctx: &mut HermiteCtx<'_>) -> bool;
}

/// R_t of a spherical Gaussian charge of exponent zeta^2 at `c` (zeta = infinity: a point
/// charge) for a primitive pair (p, P), up to degree l, scaled by `scale` (2 pi / p and the
/// attenuation sqrt(alpha / p) of a finite zeta included).
#[inline]
pub(crate) fn coulomb_hermite(c: [f64; 3], zeta: f64, scale: f64, ctx: &mut HermiteCtx<'_>) {
    let (l, p, pc) = (ctx.l, ctx.p, ctx.pc);
    let pq = [pc[0] - c[0], pc[1] - c[1], pc[2] - c[2]];
    let z2 = zeta * zeta;
    let (alpha, f) = if z2.is_finite() {
        let a = p * z2 / (p + z2);
        (a, (a / p).sqrt())
    } else {
        (p, 1.0)
    };
    let t = alpha * (pq[0] * pq[0] + pq[1] * pq[1] + pq[2] * pq[2]);
    let w = &mut *ctx.work;
    boys::boys(l, t, &mut w.fm);
    let nh = nherm(l);
    let pref = scale * 2.0 * PI / p * f;
    r_tensor(l, alpha, pq, &w.fm, pref, ctx.r, &mut w.work[..2 * nh]);
}

/// Point charges q at positions C, as the electrons see them: the potential -q / |r - C|
/// (nuclear attraction). Charge c moves with slot `first_slot + c`.
pub struct PointCharges<'a> {
    pub charges: &'a [(f64, [f64; 3])],
    pub first_slot: usize,
}

impl ChargeSource for PointCharges<'_> {
    fn len(&self) -> usize {
        self.charges.len()
    }

    fn slot(&self, c: usize) -> usize {
        self.first_slot + c
    }

    #[inline]
    fn hermite(&self, c: usize, ctx: &mut HermiteCtx<'_>) -> bool {
        let (q, pos) = self.charges[c];
        coulomb_hermite(pos, f64::INFINITY, -q, ctx);
        true
    }
}

/// How the centers of a [`GaussianCharges`] move.
#[derive(Clone, Copy)]
pub enum ChargeSlots<'a> {
    /// Point g is a coordinate of its own: slot `first + g`.
    Own { first: usize },
    /// Point g moves with atom `owner[g]`.
    Owner(&'a [usize]),
}

/// Spherical Gaussian charges q_g (exponents zeta_g^2) at the points of a surface: the
/// positive potential sum_g q_g (ab|g). Points with q_g = 0 are skipped.
pub struct GaussianCharges<'a> {
    pub pts: &'a [GaussPoint],
    pub q: &'a [f64],
    pub slots: ChargeSlots<'a>,
}

impl ChargeSource for GaussianCharges<'_> {
    fn len(&self) -> usize {
        self.pts.len()
    }

    fn slot(&self, c: usize) -> usize {
        match self.slots {
            ChargeSlots::Own { first } => first + c,
            ChargeSlots::Owner(owner) => owner[c],
        }
    }

    #[inline]
    fn hermite(&self, c: usize, ctx: &mut HermiteCtx<'_>) -> bool {
        if self.q[c] == 0.0 {
            return false;
        }
        let pt = &self.pts[c];
        coulomb_hermite(pt.r, pt.zeta, self.q[c], ctx);
        true
    }
}

/// The local channels of effective core potentials (the centers with one), each moving with
/// its atom. Centers whose tensor vanishes for a primitive pair are skipped.
pub struct EcpLocal<'a> {
    pub centers: &'a [&'a EcpCenter],
}

impl ChargeSource for EcpLocal<'_> {
    fn len(&self) -> usize {
        self.centers.len()
    }

    fn slot(&self, c: usize) -> usize {
        self.centers[c].atom
    }

    #[inline]
    fn hermite(&self, c: usize, ctx: &mut HermiteCtx<'_>) -> bool {
        let center = self.centers[c];
        let nh = nherm(ctx.l);
        ctx.r[..nh].fill(0.0);
        let w = &mut *ctx.work;
        if w.rbuf.len() < nh {
            w.rbuf.resize(nh, 0.0);
        }
        local_r(
            ctx.l,
            ctx.p,
            ctx.pc,
            center.center,
            &center.def.local,
            ctx.r,
            &mut w.rbuf[..nh],
            &mut w.work[..2 * nh],
        );
        !ctx.r.iter().all(|&x| x == 0.0)
    }
}

/// The shell pairs a potential plan visits: `pair(k)` for k < `npairs` gives the pair and its
/// derivative expansion, or None to skip it (screening, a subset of the pairs). The pairs must
/// be distinct.
pub struct Pairs<'a, F>
where
    F: Fn(usize) -> Option<(&'a ShellPair, &'a PairDeriv)> + Sync,
{
    pub npairs: usize,
    pub pair: F,
}

/// Every pair of a pair set (of derivative order one or two), in pair order.
pub fn all_pairs<'a>(
    pairs: &'a PairSet,
) -> Pairs<'a, impl Fn(usize) -> Option<(&'a ShellPair, &'a PairDeriv)> + Sync + 'a> {
    Pairs {
        npairs: pairs.len(),
        pair: move |k| Some((pairs.pair(k), pairs.deriv(k))),
    }
}

/// The scratch of one reduction chunk of a potential contraction, sized for the largest pair
/// of the basis: the Hermite densities of a primitive pair for the density, E (nherm(lab)),
/// E^A (3 x nherm(lab + 1)) and, for Hessians, E^AA (6 x nherm(lab + 2)), with the factor 2
/// of an off-diagonal shell pair; a center's Hermite tensor and its work space.
pub(crate) struct PotScratch {
    hd: Vec<f64>,
    hda: Vec<f64>,
    hdaa: Vec<f64>,
    r: Vec<f64>,
    work: HermiteWork,
}

impl PotScratch {
    /// Scratch for pairs of angular momentum sum at most `lab`, derivatives up to `order`.
    pub(crate) fn new(lab: usize, order: DerivOrder) -> Self {
        let o = order.get();
        PotScratch {
            hd: vec![0.0; nherm(lab)],
            hda: vec![0.0; 3 * nherm(lab + 1)],
            hdaa: if o >= 2 {
                vec![0.0; 6 * nherm(lab + 2)]
            } else {
                Vec::new()
            },
            r: vec![0.0; nherm(lab + o)],
            work: HermiteWork::reserved(lab + o),
        }
    }

    /// The first-order buffers of a pair with nherm(lab) = `nh0` and nherm(lab + 1) = `nh1`:
    /// (Hermite density of E, of E^A, the tensor of degree lab + 1, its work space).
    pub(crate) fn first_order(
        &mut self,
        nh0: usize,
        nh1: usize,
    ) -> (&mut [f64], &mut [f64], &mut [f64], &mut HermiteWork) {
        (
            &mut self.hd[..nh0],
            &mut self.hda[..3 * nh1],
            &mut self.r[..nh1],
            &mut self.work,
        )
    }

    /// The buffers' capacities (tests: a scratch does not grow in a contraction).
    #[cfg(test)]
    pub(crate) fn caps(&self) -> [usize; 6] {
        [
            self.hd.capacity(),
            self.hda.capacity(),
            self.hdaa.capacity(),
            self.r.capacity(),
            self.work.rbuf.capacity(),
            self.work.work.capacity(),
        ]
    }

    /// The bytes of a scratch of [`PotScratch::new`].
    pub(crate) fn bytes(lab: usize, order: DerivOrder) -> usize {
        let o = order.get();
        let nh = nherm(lab + o);
        let hdaa = if o >= 2 { 6 * nherm(lab + 2) } else { 0 };
        8 * (nherm(lab) + 3 * nherm(lab + 1) + hdaa + 4 * nh + boys::MAX_M + 1)
    }
}

/// The Hermite densities of primitive `ip` of the pair (`pr`, `pd`) for the density `d`
/// (row-major nbf x nbf) into `hd`, `hda` and, when given, `hdaa`.
#[inline]
fn prim_densities(
    basis: &BasisSet,
    (pr, pd): (&ShellPair, &PairDeriv),
    ip: usize,
    d: &[f64],
    (hd, hda, hdaa): (&mut [f64], &mut [f64], Option<&mut [f64]>),
) {
    let (s1, s2) = (&basis.shells[pr.sa], &basis.shells[pr.sb]);
    let lab = pr.lab();
    let f = if pr.sa == pr.sb { 1.0 } else { 2.0 };
    let (a0, b0, na, nb, nbf) = (s1.ao_start, s2.ao_start, pr.na, pr.nb, basis.nbf);
    herm_dens(pr.e_prim(ip), nherm(lab), 1, d, nbf, a0, b0, na, nb, f, hd);
    herm_dens(
        pd.ea_prim(ip),
        nherm(lab + 1),
        3,
        d,
        nbf,
        a0,
        b0,
        na,
        nb,
        f,
        hda,
    );
    if let Some(hdaa) = hdaa {
        herm_dens(
            pd.eaa_prim(ip),
            nherm(lab + 2),
            6,
            d,
            nbf,
            a0,
            b0,
            na,
            nb,
            f,
            hdaa,
        );
    }
}

/// The plan of the potential of a set of charge centers `S` between basis functions over the
/// shell pairs `F` visits (see the module docs): the gradient and Hessian contractions with a
/// density, into the caller's buffers over a caller-owned workspace, and the derivative
/// matrices. Generic over the source and the pair selection: monomorphized, the center loop
/// inlined into the pair loop.
///
/// The coordinate slots are the atoms first, then the source's own slots (`nslots` in all).
/// The contractions reduce over a fixed number of strided chunks of the pairs
/// (`linus_num::det::nchunk_for`, from the problem's size only), one accumulator per chunk in
/// the workspace: the results have the same bits for any number of threads.
pub struct PotentialPlan<'a, S, F>
where
    F: Fn(usize) -> Option<(&'a ShellPair, &'a PairDeriv)> + Sync,
{
    basis: &'a BasisSet,
    pairs: Pairs<'a, F>,
    src: S,
    nslots: usize,
    order: DerivOrder,
    /// the largest angular momentum sum of a pair of the basis
    lab: usize,
}

/// The caller-owned memory of a potential contraction: the chunk accumulators (one pool) and
/// one scratch per thread. Reusable across calls of its plan.
pub struct PotentialWorkspace {
    pool: Vec<f64>,
    scratch: Vec<PotScratch>,
    lab: usize,
    order: DerivOrder,
}

impl PotentialWorkspace {
    /// A workspace with room for `values` accumulator values, with a scratch per thread of
    /// the current pool (sized for pairs of angular momentum sum `lab`, derivatives up to
    /// `order`).
    pub(crate) fn new(values: usize, lab: usize, order: DerivOrder) -> Self {
        let nt = rayon::current_num_threads();
        PotentialWorkspace {
            pool: vec![0.0; values],
            scratch: (0..nt).map(|_| PotScratch::new(lab, order)).collect(),
            lab,
            order,
        }
    }

    /// Room for at least `values` accumulator values (the pool grows; its contents are not
    /// kept).
    pub(crate) fn reserve(&mut self, values: usize) {
        if self.pool.len() < values {
            self.pool = vec![0.0; values];
        }
    }

    /// `nchunk` accumulators of `len` values and the threads' scratches (one more for each
    /// thread of a larger pool than the workspace was made in).
    pub(crate) fn carve(&mut self, nchunk: usize, len: usize) -> (&mut [f64], &mut [PotScratch]) {
        assert!(
            self.pool.len() >= nchunk * len,
            "the workspace of another plan"
        );
        let (lab, order) = (self.lab, self.order);
        let nt = rayon::current_num_threads();
        if self.scratch.len() < nt {
            self.scratch.resize_with(nt, || PotScratch::new(lab, order));
        }
        (&mut self.pool[..nchunk * len], &mut self.scratch)
    }
}

/// The chunk count and accumulator length of the gradient (order One) or the Hessian (order
/// Two) over `npairs` pairs and `nslots` coordinate slots.
pub(crate) fn chunks(npairs: usize, nslots: usize, order: DerivOrder) -> (usize, usize) {
    let len = if order == DerivOrder::Two {
        9 * nslots * nslots
    } else {
        3 * nslots
    };
    (nchunk_for(npairs, len, DEFAULT_BUDGET), len)
}

impl<'a, S, F> PotentialPlan<'a, S, F>
where
    S: ChargeSource,
    F: Fn(usize) -> Option<(&'a ShellPair, &'a PairDeriv)> + Sync,
{
    /// The plan of the potential of `src` over `pairs` (pairs of `basis` with derivative
    /// expansions of order `order` or higher), with `nslots` coordinate slots (the atoms, then
    /// the source's own).
    pub fn new(
        basis: &'a BasisSet,
        pairs: Pairs<'a, F>,
        src: S,
        nslots: usize,
        order: DerivOrder,
    ) -> Self {
        assert!(
            order != DerivOrder::Zero,
            "a potential plan is of order One or Two"
        );
        assert!(nslots >= basis.natom(), "the slots start with the atoms");
        PotentialPlan {
            basis,
            pairs,
            src,
            nslots,
            order,
            lab: 2 * basis.max_l(),
        }
    }

    /// The plan's derivative order.
    pub fn order(&self) -> DerivOrder {
        self.order
    }

    /// The number of coordinate slots.
    pub fn nslots(&self) -> usize {
        self.nslots
    }

    /// The bytes of a workspace for the contractions of the plan's order and below; all
    /// shared, as the chunks do not depend on the threads.
    pub fn workspace_size(&self) -> WorkspaceSize {
        let (values, _) = self.pool_size();
        WorkspaceSize {
            per_thread: PotScratch::bytes(self.lab, self.order),
            shared: 8 * values,
        }
    }

    /// A workspace for the contractions of the plan's order and below.
    pub fn workspace(&self) -> PotentialWorkspace {
        let (values, _) = self.pool_size();
        PotentialWorkspace::new(values, self.lab, self.order)
    }

    fn pool_size(&self) -> (usize, usize) {
        let n = self.pairs.npairs;
        let (c1, l1) = chunks(n, self.nslots, DerivOrder::One);
        if self.order == DerivOrder::One {
            (c1 * l1, c1)
        } else {
            let (c2, l2) = chunks(n, self.nslots, DerivOrder::Two);
            ((c1 * l1).max(c2 * l2), c1.max(c2))
        }
    }

    /// The gradient of sum_ab D_ab V_ab (`d` row-major nbf x nbf, symmetric) into `out`: one
    /// 3-vector per coordinate slot, overwritten.
    pub fn gradient(&self, d: &[f64], out: &mut [[f64; 3]], ws: &mut PotentialWorkspace) {
        assert_eq!(out.len(), self.nslots, "the gradient has one row per slot");
        let (nchunk, len) = chunks(self.pairs.npairs, self.nslots, DerivOrder::One);
        let (pool, scratch) = ws.carve(nchunk, len);
        gradient_fold(self.basis, &self.pairs, &self.src, d, pool, nchunk, scratch);
        out.as_flattened_mut().copy_from_slice(&pool[..len]);
    }

    /// The Hessian of sum_ab D_ab V_ab over the coordinate slots into `out`
    /// ((3 nslots) x (3 nslots), row-major, overwritten). An order-Two plan.
    pub fn hessian(&self, d: &[f64], out: &mut [f64], ws: &mut PotentialWorkspace) {
        assert_eq!(
            self.order,
            DerivOrder::Two,
            "the Hessian needs an order-Two plan"
        );
        let (nchunk, len) = chunks(self.pairs.npairs, self.nslots, DerivOrder::Two);
        assert_eq!(out.len(), len, "the Hessian is (3 nslots)^2");
        let (pool, scratch) = ws.carve(nchunk, len);
        hessian_fold(self.basis, &self.pairs, &self.src, d, pool, nchunk, scratch);
        out.copy_from_slice(&pool[..len]);
    }

    /// The derivative matrices d/dX V_ab for every coordinate X of the slots, added into
    /// `out` ((3 nslots) x nbf x nbf, row-major, index 3 slot + k). The pair blocks are
    /// computed in parallel and added pair by pair; every element belongs to one pair, so the
    /// result does not depend on the thread count.
    pub fn add_deriv_matrices(&self, out: &mut [f64]) {
        let nbf = self.basis.nbf;
        assert_eq!(out.len(), 3 * self.nslots * nbf * nbf);
        add_deriv_matrices(self.basis, &self.pairs, &self.src, out);
    }
}

/// The gradient of [`PotentialPlan::gradient`] into `pool[..3 nslots]` (slot-major), folded
/// over `nchunk` chunks of the pool (which holds that many accumulators) on the threads'
/// `scratch`.
pub(crate) fn gradient_fold<'a, S, F>(
    basis: &BasisSet,
    pairs: &Pairs<'a, F>,
    src: &S,
    d: &[f64],
    pool: &mut [f64],
    nchunk: usize,
    scratch: &mut [PotScratch],
) where
    S: ChargeSource,
    F: Fn(usize) -> Option<(&'a ShellPair, &'a PairDeriv)> + Sync,
{
    let sh = &basis.shells;
    let shf = shift1();
    let len = pool.len() / nchunk;
    ordered_pool(pairs.npairs, pool, len, nchunk, scratch, |acc, s, k| {
        let Some((pr, pd)) = (pairs.pair)(k) else {
            return;
        };
        let g = acc.as_chunks_mut::<3>().0;
        let (a, b) = (sh[pr.sa].atom, sh[pr.sb].atom);
        let lab = pr.lab();
        let (nh0, nh1) = (nherm(lab), nherm(lab + 1));
        let PotScratch {
            hd, hda, r, work, ..
        } = s;
        let (hd, hda, r) = (&mut hd[..nh0], &mut hda[..3 * nh1], &mut r[..nh1]);
        for ip in 0..pr.nprim {
            let mut have = false;
            for c in 0..src.len() {
                let mut ctx = HermiteCtx {
                    l: lab + 1,
                    p: pr.p[ip],
                    pc: pr.pc[ip],
                    r: &mut *r,
                    work: &mut *work,
                };
                if !src.hermite(c, &mut ctx) {
                    continue;
                }
                if !have {
                    prim_densities(basis, (pr, pd), ip, d, (&mut *hd, &mut *hda, None));
                    have = true;
                }
                let sc = src.slot(c);
                for k in 0..3 {
                    let mut va = 0.0;
                    for rr in 0..nh1 {
                        va += hda[k * nh1 + rr] * r[rr];
                    }
                    let mut vc = 0.0;
                    for rr in 0..nh0 {
                        vc -= hd[rr] * r[shf[rr][k] as usize];
                    }
                    g[a][k] += va;
                    g[sc][k] += vc;
                    g[b][k] -= va + vc;
                }
            }
        }
    });
}

/// The Hessian of [`PotentialPlan::hessian`] into `pool[..(3 nslots)^2]`, folded over
/// `nchunk` chunks of the pool on the threads' `scratch`. Needs pairs and scratch of
/// derivative order two.
pub(crate) fn hessian_fold<'a, S, F>(
    basis: &BasisSet,
    pairs: &Pairs<'a, F>,
    src: &S,
    d: &[f64],
    pool: &mut [f64],
    nchunk: usize,
    scratch: &mut [PotScratch],
) where
    S: ChargeSource,
    F: Fn(usize) -> Option<(&'a ShellPair, &'a PairDeriv)> + Sync,
{
    let sh = &basis.shells;
    let shf = shift1();
    let len = pool.len() / nchunk;
    let n3 = len.isqrt();
    ordered_pool(pairs.npairs, pool, len, nchunk, scratch, |h, s, k| {
        let Some((pr, pd)) = (pairs.pair)(k) else {
            return;
        };
        let (xa, xb) = (sh[pr.sa].atom, sh[pr.sb].atom);
        let lab = pr.lab();
        let (nh0, nh1, nh2) = (nherm(lab), nherm(lab + 1), nherm(lab + 2));
        let PotScratch {
            hd,
            hda,
            hdaa,
            r,
            work,
        } = s;
        let (hd, hda, hdaa, r) = (
            &mut hd[..nh0],
            &mut hda[..3 * nh1],
            &mut hdaa[..6 * nh2],
            &mut r[..nh2],
        );
        for ip in 0..pr.nprim {
            let mut have = false;
            for c in 0..src.len() {
                let mut ctx = HermiteCtx {
                    l: lab + 2,
                    p: pr.p[ip],
                    pc: pr.pc[ip],
                    r: &mut *r,
                    work: &mut *work,
                };
                if !src.hermite(c, &mut ctx) {
                    continue;
                }
                if !have {
                    let dens = (&mut *hd, &mut *hda, Some(&mut *hdaa));
                    prim_densities(basis, (pr, pd), ip, d, dens);
                    have = true;
                }
                let mut aa = [[0.0; 3]; 3];
                let mut ac = [[0.0; 3]; 3];
                let mut cc = [[0.0; 3]; 3];
                for k in 0..3 {
                    for l in 0..3 {
                        let kl = d2_idx(k, l);
                        let mut v = 0.0;
                        for rr in 0..nh2 {
                            v += hdaa[kl * nh2 + rr] * r[rr];
                        }
                        aa[k][l] = v;
                        let mut v = 0.0;
                        for rr in 0..nh1 {
                            v -= hda[k * nh1 + rr] * r[shf[rr][l] as usize];
                        }
                        ac[k][l] = v;
                        let mut v = 0.0;
                        for rr in 0..nh0 {
                            v += hd[rr] * r[shf[shf[rr][k] as usize][l] as usize];
                        }
                        cc[k][l] = v;
                    }
                }
                let xc = src.slot(c);
                for k in 0..3 {
                    for l in 0..3 {
                        let ab = -(aa[k][l] + ac[k][l]);
                        let bb = aa[k][l] + ac[k][l] + ac[l][k] + cc[k][l];
                        let bc = -(ac[k][l] + cc[k][l]);
                        let mut add =
                            |x: usize, y: usize, v: f64| h[(3 * x + k) * n3 + 3 * y + l] += v;
                        add(xa, xa, aa[k][l]);
                        add(xb, xb, bb);
                        add(xc, xc, cc[k][l]);
                        add(xa, xb, ab);
                        add(xa, xc, ac[k][l]);
                        add(xb, xc, bc);
                        // transposed center pairs: H[Y_l][X_k] = M[k][l]
                        h[(3 * xb + l) * n3 + 3 * xa + k] += ab;
                        h[(3 * xc + l) * n3 + 3 * xa + k] += ac[k][l];
                        h[(3 * xc + l) * n3 + 3 * xb + k] += bc;
                    }
                }
            }
        }
    });
}

/// out\[ab\] += sum_t e\[t stride + ab\] r\[t\] over the Hermite functions t of `r` (in order;
/// vanishing terms skipped): one primitive pair's contraction of an expansion (E, or one
/// component of E^A at `stride` 3 nab) with a Hermite tensor.
#[inline]
fn add_contraction(e: &[f64], stride: usize, r: &[f64], out: &mut [f64]) {
    let nab = out.len();
    for (t, &rv) in r.iter().enumerate() {
        if rv == 0.0 {
            continue;
        }
        let row = &e[t * stride..t * stride + nab];
        for (o, &v) in out.iter_mut().zip(row) {
            *o += v * rv;
        }
    }
}

/// The potential block of one shell pair, sum_C <a| v_C |b> over the centers of `src`, into
/// `out` \[nab\] (overwritten). Per primitive pair the centers' Hermite tensors are summed
/// first and contracted with E once: the summation order of every potential matrix and of the
/// d/dA derivative blocks ([`deriv_pair_blocks`]).
pub fn matrix_block<S: ChargeSource>(pr: &ShellPair, src: &S, out: &mut [f64]) {
    let lab = pr.lab();
    let (nh, nab) = (nherm(lab), pr.nab());
    let out = &mut out[..nab];
    out.fill(0.0);
    let (mut r, mut acc) = (vec![0.0; nh], vec![0.0; nh]);
    let mut work = HermiteWork::new(lab);
    for ip in 0..pr.nprim {
        acc.fill(0.0);
        for c in 0..src.len() {
            let mut ctx = HermiteCtx {
                l: lab,
                p: pr.p[ip],
                pc: pr.pc[ip],
                r: &mut r,
                work: &mut work,
            };
            if src.hermite(c, &mut ctx) {
                acc.iter_mut().zip(&r).for_each(|(a, &v)| *a += v);
            }
        }
        add_contraction(pr.e_prim(ip), nab, &acc, out);
    }
}

/// The first-derivative blocks of one shell pair's potential, accumulated into `oa`
/// \[3\]\[nab\] (d/dA, the first shell's center) and `oc` \[center\]\[3\]\[nab\] (d/dC of each
/// center); d/dB = -(d/dA + sum_C d/dC) by translational invariance ([`deriv_b_blocks`]).
/// d/dA contracts E^A once per primitive pair with the centers' summed tensors, as
/// [`matrix_block`]; d/dC contracts E with each center's own.
pub fn deriv_pair_blocks<S: ChargeSource>(
    pr: &ShellPair,
    pd: &PairDeriv,
    src: &S,
    oa: &mut [f64],
    oc: &mut [f64],
) {
    let lab = pr.lab();
    let nab = pr.nab();
    let (nh0, nh1) = (nherm(lab), nherm(lab + 1));
    let shf = shift1();
    let (mut r, mut acc) = (vec![0.0; nh1], vec![0.0; nh1]);
    let mut work = HermiteWork::new(lab + 1);
    for ip in 0..pr.nprim {
        let e = pr.e_prim(ip);
        let ea = pd.ea_prim(ip);
        acc.fill(0.0);
        for c in 0..src.len() {
            let mut ctx = HermiteCtx {
                l: lab + 1,
                p: pr.p[ip],
                pc: pr.pc[ip],
                r: &mut r,
                work: &mut work,
            };
            if !src.hermite(c, &mut ctx) {
                continue;
            }
            acc.iter_mut().zip(&r).for_each(|(a, &v)| *a += v);
            let o = &mut oc[c * 3 * nab..(c + 1) * 3 * nab];
            for rr in 0..nh0 {
                let row = &e[rr * nab..(rr + 1) * nab];
                for k in 0..3 {
                    let rv = -r[shf[rr][k] as usize];
                    for (x, &v) in o[k * nab..(k + 1) * nab].iter_mut().zip(row) {
                        *x += v * rv;
                    }
                }
            }
        }
        for k in 0..3 {
            add_contraction(
                &ea[k * nab..],
                3 * nab,
                &acc,
                &mut oa[k * nab..(k + 1) * nab],
            );
        }
    }
}

/// A pair's d/dB blocks \[3\]\[nab\] into `ob`: -(d/dA + sum_C d/dC) from
/// [`deriv_pair_blocks`] (translational invariance).
pub fn deriv_b_blocks(oa: &[f64], oc: &[f64], ncenter: usize, nab: usize, ob: &mut [f64]) {
    for (x, o) in ob[..3 * nab].iter_mut().enumerate() {
        let mut v = oa[x];
        for c in 0..ncenter {
            v += oc[c * 3 * nab + x];
        }
        *o = -v;
    }
}

/// A shell pair with its first-derivative blocks d/dA, d/dB and d/dC.
type PairBlocks<'a> = (&'a ShellPair, Vec<f64>, Vec<f64>, Vec<f64>);

/// [`PotentialPlan::add_deriv_matrices`].
pub(crate) fn add_deriv_matrices<'a, S, F>(
    basis: &BasisSet,
    pairs: &Pairs<'a, F>,
    src: &S,
    out: &mut [f64],
) where
    S: ChargeSource,
    F: Fn(usize) -> Option<(&'a ShellPair, &'a PairDeriv)> + Sync,
{
    let nbf = basis.nbf;
    let n2 = nbf * nbf;
    let sh = &basis.shells;
    let nc = src.len();
    let blocks: Vec<Option<PairBlocks>> = (0..pairs.npairs)
        .into_par_iter()
        .map(|k| {
            let (pr, pd) = (pairs.pair)(k)?;
            let nab = pr.nab();
            let mut oa = vec![0.0; 3 * nab];
            let mut ob = vec![0.0; 3 * nab];
            let mut oc = vec![0.0; 3 * nab * nc];
            deriv_pair_blocks(pr, pd, src, &mut oa, &mut oc);
            deriv_b_blocks(&oa, &oc, nc, nab, &mut ob);
            Some((pr, oa, ob, oc))
        })
        .collect();
    for (pr, oa, ob, oc) in blocks.into_iter().flatten() {
        let (s1, s2) = (&sh[pr.sa], &sh[pr.sb]);
        let same = pr.sa == pr.sb;
        let (na, nb) = (pr.na, pr.nb);
        let nab = na * nb;
        for k in 0..3 {
            // the block at (a, b), and at (b, a) for distinct shells
            let mut put = |slot: usize, blk: &[f64]| {
                let x = 3 * slot + k;
                let m = &mut out[x * n2..(x + 1) * n2];
                for a in 0..na {
                    for b in 0..nb {
                        let v = blk[a * nb + b];
                        m[(s1.ao_start + a) * nbf + s2.ao_start + b] += v;
                        if !same {
                            m[(s2.ao_start + b) * nbf + s1.ao_start + a] += v;
                        }
                    }
                }
            };
            put(s1.atom, &oa[k * nab..(k + 1) * nab]);
            put(s2.atom, &ob[k * nab..(k + 1) * nab]);
            for c in 0..nc {
                put(src.slot(c), &oc[(c * 3 + k) * nab..(c * 3 + k + 1) * nab]);
            }
        }
    }
}
