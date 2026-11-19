//! The batch engine of the XC integrator. Every pass over a grid (the potential, the gradient,
//! the CPKS kernel, the Hessian and grid terms, and VV10's passes on its own grid) runs through
//! [`XcIntegrator::for_each_batch`], [`XcIntegrator::for_each_batch_in`] or
//! [`XcIntegrator::for_each_group`], which own what every pass needs and none should write
//! itself:
//! - the scheduling: one rayon task per batch, or per [`BatchGroup`] (consecutive batches with
//!   group-local dense density and potential blocks), on the calling pool;
//! - the per-thread state: the collocation buffers ([`Colloc`]) and the pass's own work (its
//!   scratch and fixed-point accumulators), created lazily per pool thread and returned at the
//!   end for the exact merges (`linus_num::det`);
//! - the collocation of each batch with the pass's derivative order and screening threshold, in
//!   the one function-major layout every pass reads (`ao[(m nc + c) ld + g]`, the SIMD
//!   collocation's own), and the empty and skipped batches (a body sees `nao == 0` and no
//!   functions, without any collocation work);
//! - the outputs, in batch (and group) order.
//!
//! A pass is a pair of closures over its inputs: `init` (its per-thread work) and `body` (one
//! batch, given as a [`BatchCtx`]). Bodies are sequential: each runs while its thread holds its
//! slot of the per-thread state, so a body must not start rayon work (a waiting thread could steal
//! a batch of the same pass and deadlock on its own slot; every product in a body uses
//! `Par::Seq`). Results do not depend on the number of threads or the schedule: a batch's work is
//! the same whichever thread runs it, outputs keep the batch order, and accumulators sum in fixed
//! point.

use crate::profile as xp;
use crate::{sized, BatchGroup, XcIntegrator};
use linus_ints::points::colloc::Collocator;
use linus_num::det::{index_runs, ThreadAccs};
use rayon::prelude::*;

/// How a pass collocates its batches.
pub(crate) struct BatchPlan<'a> {
    /// Derivative order of the basis functions.
    pub(crate) order: usize,
    /// Functions whose value and derivatives stay below this in a batch are left out.
    pub(crate) thresh: f64,
    /// Batches left out before their collocation (their body sees no functions).
    pub(crate) skip: Option<&'a (dyn Fn(usize) -> bool + Sync)>,
}

impl<'a> BatchPlan<'a> {
    /// Function-major collocation to `order` with screening threshold `thresh`, no batch
    /// skipped.
    pub(crate) fn new(order: usize, thresh: f64) -> Self {
        BatchPlan {
            order,
            thresh,
            skip: None,
        }
    }

    /// Leave out the batches for which `skip` is true.
    pub(crate) fn skipping(self, skip: &'a (dyn Fn(usize) -> bool + Sync)) -> Self {
        BatchPlan {
            skip: Some(skip),
            ..self
        }
    }
}

/// Collocation buffers of one thread.
#[derive(Default)]
pub(crate) struct Colloc {
    xs: Vec<f64>,
    ys: Vec<f64>,
    zs: Vec<f64>,
    /// function-major values
    ao: Vec<f64>,
    idx: Vec<usize>,
    owner: Vec<usize>,
    runs: Vec<(usize, usize, usize)>,
    /// group-local position of every basis function (group scheduling), the batch's local
    /// indices and their runs
    lpos: Vec<usize>,
    lidx: Vec<usize>,
    lruns: Vec<(usize, usize, usize)>,
}

/// A collocated batch, as a pass's body sees it.
pub(crate) struct BatchCtx<'a> {
    /// The batch and its first grid point.
    pub(crate) bi: usize,
    pub(crate) start: usize,
    /// Points, and the stride of a function's component in `ao` (`np` padded to a multiple of
    /// 8).
    pub(crate) np: usize,
    pub(crate) ld: usize,
    /// Kept functions and components per function.
    pub(crate) nao: usize,
    pub(crate) nc: usize,
    /// The weights of the batch's points.
    pub(crate) w: &'a [f64],
    /// The kept functions' values and derivatives, component c of function m at
    /// `(m nc + c) ld` (empty when `nao == 0`).
    pub(crate) ao: &'a [f64],
    /// Their global indices (ascending), atoms, and the contiguous runs of `idx`
    /// (local start, global start, length).
    pub(crate) idx: &'a [usize],
    pub(crate) owner: &'a [usize],
    pub(crate) runs: &'a [(usize, usize, usize)],
    /// With group scheduling: the group's number of functions (the order of its local blocks),
    /// the batch functions' positions in the group's function list and the runs of those (0 and
    /// empty otherwise).
    pub(crate) u: usize,
    pub(crate) lidx: &'a [usize],
    pub(crate) lruns: &'a [(usize, usize, usize)],
}

impl XcIntegrator {
    /// Collocates batch `bi` with `coll` into `c` as `plan` says (nothing for an empty or
    /// skipped batch, or one of an inactive group); with `group` (whether the group is active,
    /// its number of functions) the batch's group-local indices too.
    fn collocate<'c>(
        &'c self,
        coll: &Collocator,
        plan: &BatchPlan<'_>,
        bi: usize,
        c: &'c mut Colloc,
        group: Option<(bool, usize)>,
    ) -> BatchCtx<'c> {
        let mut lap = xp::Lap::start();
        let info = &self.batch_info()[bi];
        let grid = &self.grid;
        let b = &grid.batches[bi];
        let np = b.end - b.start;
        let ld = np.div_ceil(8) * 8;
        let nc = coll.ncomp;
        c.idx.clear();
        c.owner.clear();
        c.runs.clear();
        c.lidx.clear();
        c.lruns.clear();
        let skipped = matches!(group, Some((false, _))) || plan.skip.is_some_and(|f| f(bi));
        let mut nao = 0;
        if !skipped && np > 0 && !info.shells.is_empty() {
            for (v, src) in [
                (&mut c.xs, &grid.x),
                (&mut c.ys, &grid.y),
                (&mut c.zs, &grid.z),
            ] {
                v.clear();
                v.extend_from_slice(&src[b.start..b.end]);
                v.resize(ld, src[b.end - 1]);
            }
            let ao = sized(&mut c.ao, info.nfunc * nc * ld);
            nao = coll.eval(
                &info.shells,
                &c.xs,
                &c.ys,
                &c.zs,
                np,
                ld,
                &info.lo,
                &info.hi,
                ao,
                &mut c.idx,
                &mut c.owner,
            );
            index_runs(&c.idx, &mut c.runs);
            if xp::enabled() {
                let neval: usize = info
                    .shells
                    .iter()
                    .filter(|&&sh| {
                        Collocator::box_distance(&coll.shells[sh].center, &info.lo, &info.hi)
                            < coll.shells[sh].rcut
                    })
                    .map(|&sh| coll.shells[sh].nf)
                    .sum();
                xp::batch(np, nao, info.nfunc, neval);
            }
            if group.is_some() {
                let lpos = &c.lpos;
                c.lidx.extend(c.idx.iter().map(|&f| lpos[f]));
                debug_assert!(c.lidx.iter().all(|&l| l != usize::MAX));
                index_runs(&c.lidx, &mut c.lruns);
            }
        }
        let ao = &c.ao[..nao * nc * ld];
        lap.lap(xp::COLLOC);
        BatchCtx {
            bi,
            start: b.start,
            np,
            ld,
            nao,
            nc,
            w: &grid.w[b.start..b.end],
            ao,
            idx: &c.idx,
            owner: &c.owner,
            runs: &c.runs,
            u: group.map_or(0, |g| g.1),
            lidx: &c.lidx,
            lruns: &c.lruns,
        }
    }

    /// One task per item `(batch, payload)` of `items`: `body(work, batch, payload)` for each
    /// (collocated as `plan` says), with this thread's `work` (from `init` on first use).
    /// Returns the outputs in item order and the threads' works.
    pub(crate) fn for_each_batch_in<P: Send, W: Send, O: Send>(
        &self,
        plan: &BatchPlan<'_>,
        items: Vec<(usize, P)>,
        init: impl Fn() -> W + Sync,
        body: impl Fn(&mut W, &BatchCtx<'_>, P) -> O + Sync,
    ) -> (Vec<O>, Vec<W>) {
        let coll = self.collocator(plan.order, plan.thresh);
        // before the parallel loop: its build runs rayon (see the module docs)
        self.batch_info();
        let accs: ThreadAccs<(Colloc, W)> = ThreadAccs::new();
        let outs: Vec<O> = items
            .into_par_iter()
            .map(|(bi, p)| {
                accs.with(
                    || (Colloc::default(), init()),
                    |(c, w)| {
                        let b = self.collocate(&coll, plan, bi, c, None);
                        body(w, &b, p)
                    },
                )
            })
            .collect();
        (outs, accs.into_vec().into_iter().map(|x| x.1).collect())
    }

    /// One task per batch of the grid: [`Self::for_each_batch_in`] without payloads.
    pub(crate) fn for_each_batch<W: Send, O: Send>(
        &self,
        plan: &BatchPlan<'_>,
        init: impl Fn() -> W + Sync,
        body: impl Fn(&mut W, &BatchCtx<'_>) -> O + Sync,
    ) -> (Vec<O>, Vec<W>) {
        let items = (0..self.grid.batches.len()).map(|bi| (bi, ())).collect();
        self.for_each_batch_in(plan, items, init, |w, b, ()| body(w, b))
    }

    /// One task per [`BatchGroup`]: `begin(work, group)` (false: the group is inactive, its
    /// batches are not collocated), `body(work, batch)` for each of its batches (with their
    /// group-local indices), then `end(work, group)`. Returns the batch outputs in batch order,
    /// the group outputs in group order and the threads' works.
    pub(crate) fn for_each_group<W: Send, O: Send, G: Send>(
        &self,
        plan: &BatchPlan<'_>,
        init: impl Fn() -> W + Sync,
        begin: impl Fn(&mut W, &BatchGroup) -> bool + Sync,
        body: impl Fn(&mut W, &BatchCtx<'_>) -> O + Sync,
        end: impl Fn(&mut W, &BatchGroup) -> G + Sync,
    ) -> (Vec<O>, Vec<G>, Vec<W>) {
        let coll = self.collocator(plan.order, plan.thresh);
        let nbf = self.basis.nbf;
        // before the parallel loop: their builds run rayon (see the module docs)
        self.batch_info();
        let groups = self.batch_groups();
        let accs: ThreadAccs<(Colloc, W)> = ThreadAccs::new();
        let parts: Vec<(Vec<O>, G)> = groups
            .par_iter()
            .map(|g| {
                accs.with(
                    || (Colloc::default(), init()),
                    |(c, w)| {
                        if c.lpos.len() != nbf {
                            c.lpos = vec![usize::MAX; nbf];
                        }
                        for (l, &f) in g.funcs.iter().enumerate() {
                            c.lpos[f] = l;
                        }
                        let active = begin(w, g);
                        let outs = g
                            .batches
                            .clone()
                            .map(|bi| {
                                let b = self.collocate(
                                    &coll,
                                    plan,
                                    bi,
                                    c,
                                    Some((active, g.funcs.len())),
                                );
                                body(w, &b)
                            })
                            .collect();
                        (outs, end(w, g))
                    },
                )
            })
            .collect();
        let mut outs = Vec::with_capacity(self.grid.batches.len());
        let mut gouts = Vec::with_capacity(parts.len());
        for (o, g) in parts {
            outs.extend(o);
            gouts.push(g);
        }
        (
            outs,
            gouts,
            accs.into_vec().into_iter().map(|x| x.1).collect(),
        )
    }
}
