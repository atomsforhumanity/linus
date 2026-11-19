//! The plan of the three-center (density-fitting) derivative contractions, on the
//! auxiliary-lane kernels of [`crate::eri3c`].
//!
//! [`Eri3DerivPlan`] borrows an orbital pair set (derivative order one or two), an auxiliary
//! basis and its (P, 1) pairs, and lays out what its operations walk: the auxiliary lane
//! batches, and per family of operations the orbital pairs it keeps, in the layout of its
//! kernels (built once, on the first operation of the family).
//!
//! The split-RI-J terms of a Coulomb fit, for a density D and auxiliary coefficients c
//! (SHARK; [`super::deriv`]), over the pairs that pass [`DfThresholds::pair`] (Schwarz bound
//! times the largest auxiliary one):
//! - [`Eri3DerivPlan::gamma`]: γ_P = sum_mn (P|mn) D_mn, the right-hand side of the fit (no
//!   derivative, but the gradient needs it over the same pairs and screening);
//! - [`Eri3DerivPlan::coulomb_gradient`]: sum_P c_P sum_mn ∂(P|mn)/∂X D_mn, all three centers;
//! - [`Eri3DerivPlan::metric_gradient`]: sum_PQ w_P c_Q ∂(P|Q)/∂X, each auxiliary shell P
//!   standing in for an orbital pair as (P, 1) against the lane batches.
//!
//! These cut their pairs into contiguous chunks fixed by the problem (at most [`NCHUNK`]),
//! fold each chunk in pair order into its own partial and add the partials in chunk order
//! ([`schedule::ordered_ranges`]).
//!
//! The operations with a non-separable three-index density Γ_P\[mn\] (exchange), on a block of
//! auxiliary shells at a time ([`Eri3Block`]: the callers bound their memory by blocks):
//! - [`Eri3DerivPlan::gradient`]: sum_{P, mn} Γ_P\[mn\] ∂(P|mn)/∂X, Γ from the caller's weights
//!   per shell pair ([`super::dgrad`]);
//! - [`Eri3DerivPlan::deriv_block`]: the derivative integrals (∂_k m n|P) themselves
//!   ([`super::dblock`]);
//! - [`Eri3DerivPlan::hessian`]: sum_{P, mn} Γ_P\[mn\] ∂²(P|mn)/∂X∂Y (an order-Two plan).
//!
//! The contractions deal their pairs to strided chunks fixed by the problem and add the
//! chunks' partials in order ([`schedule::ordered_strided`]); the blocks write disjoint
//! memory. Every operation runs its threads on their own scratch, so the results have the same
//! bits for any number of threads, and all memory is in the caller's [`Eri3DerivWorkspace`].

use super::dblock::{deriv_block, hess_block, H3Work, Pair3cD, Pair3cH};
use super::deriv::{pair_hermite_density_deriv, pair_jgrad};
use super::dgrad::{grad_block, GWork, Pair3cG};
use super::{chunk_coefficients, pair_gamma, pair_hermite_density, Aux3c, AuxChunk, DfThresholds};
use super::{Lanes, Pair3c, Work3c, ZERO};
use crate::basis::{ncart, BasisSet};
use crate::contract::AoMatrix;
use crate::core::hermite::{nherm, ShellPair};
use crate::engine::{schedule, WorkspaceSize};
use crate::error::IntsError;
use crate::operator::Operator;
use crate::pairs::{DerivOrder, PairKind, PairSet};
use linus_num::det::{nchunk_for, DEFAULT_BUDGET};
use linus_num::simd::Isa;
use rayon::prelude::*;
use std::ops::Range;
use std::sync::OnceLock;

/// The most chunks of an RI-J operation's reduction.
pub const NCHUNK: usize = 256;

/// The options of an [`Eri3DerivPlan`].
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct Eri3DerivOptions {
    /// The screening thresholds: `pair` drops orbital pairs from the RI-J layout, `prim`
    /// screens the integrals' primitive pairs against lane batches and `deriv_prim` the
    /// derivatives', `deriv_pair` drops orbital pairs from the layouts of
    /// [`Eri3DerivPlan::gradient`] and [`Eri3DerivPlan::deriv_block`].
    pub thresholds: DfThresholds,
}

/// A block of auxiliary shells, the operator and the screening threshold of one operation
/// with a three-index density.
#[derive(Clone, Debug)]
pub struct Eri3Block {
    /// The auxiliary shells (their functions are contiguous).
    pub shells: Range<usize>,
    /// The two-electron operator (a/r + b erf(omega r)/r).
    pub op: Operator,
    /// The kernels' screening threshold (see each operation).
    pub thresh: f64,
}

/// The three-center derivative contractions over an orbital pair set and an auxiliary basis
/// (see the module docs).
pub struct Eri3DerivPlan<'a> {
    basis: &'a BasisSet,
    pairs: &'a PairSet,
    aux: &'a BasisSet,
    apairs: &'a PairSet,
    order: DerivOrder,
    opts: Eri3DerivOptions,
    aux3: Aux3c,
    /// the lane batches of every auxiliary shell
    chunks: Vec<AuxChunk>,
    /// the atom of each auxiliary function, of each orbital shell; (first function, count)
    /// of each orbital shell
    aux_atom: Vec<usize>,
    ao_atom: Vec<usize>,
    ao: Vec<(usize, usize)>,
    natom: usize,
    /// scratch sizes: the most functions, Hermite-density values (order 0 and 1) of a pair
    nab: usize,
    ndh: usize,
    ndha: usize,
    /// the layouts of the operation families, on first use
    rij: OnceLock<Rij>,
    gpairs: OnceLock<Vec<Pair3cG<'a>>>,
    dpairs: OnceLock<(Vec<Pair3cD>, Vec<u32>)>,
    hpairs: OnceLock<Vec<Pair3cH>>,
}

/// The RI-J operations' layout: the kept orbital pairs (positions in the pair set, lane
/// layouts) and the (P, 1) pairs of the auxiliary shells, in shell order.
struct Rij {
    kept: Vec<usize>,
    p3: Vec<Pair3c>,
    q3: Vec<Pair3c>,
}

/// One thread's scratch of an [`Eri3DerivPlan`]'s operations.
pub(crate) struct Eri3Thread {
    pub(crate) work: Work3c,
    /// the auxiliary-center derivatives of a chunk, per auxiliary function
    gaux: Vec<[f64; 3]>,
    /// a pair's weighted density rows and its Hermite densities
    dw: Vec<f64>,
    dh: Vec<f64>,
    dha: Vec<f64>,
    /// the gradient's weights; the derivative blocks' lanes; the Hessian's kernel scratch, a
    /// pair's 21 components per auxiliary function, and the block's function indices
    pub(crate) gw: GWork,
    pub(crate) lanes: Vec<Lanes>,
    pub(crate) h3: H3Work,
    pub(crate) h21: Vec<[f64; 21]>,
    pub(crate) fidx: Vec<usize>,
}

/// What an operation on a block of auxiliary shells passes its kernels' driver.
pub(crate) struct BlockCtx<'p> {
    pub(crate) aux3c: &'p Aux3c,
    /// the block's lane batches
    pub(crate) chunks: Vec<AuxChunk>,
    pub(crate) ao: &'p [(usize, usize)],
    pub(crate) ao_atom: &'p [usize],
    /// the atoms of the block's functions
    pub(crate) aux_atom: &'p [usize],
    pub(crate) nbf: usize,
    pub(crate) nq: usize,
    pub(crate) natom: usize,
    pub(crate) op: Operator,
    pub(crate) thresh: f64,
    pub(crate) isa: Isa,
}

/// The caller-owned memory of an [`Eri3DerivPlan`]'s operations: scratch per pool thread
/// (made for a thread count), a partial result per chunk, and the lane coefficients of c.
/// Reusable across calls of its plan; the partials of an operation are made on its first call
/// (their count depends on the layout, built then), and the scratch grows to the largest
/// block seen.
pub struct Eri3DerivWorkspace {
    threads: Vec<Eri3Thread>,
    part_g: Vec<Vec<[f64; 3]>>,
    part_gamma: Vec<Vec<f64>>,
    part_h: Vec<Vec<f64>>,
    dc: Vec<Vec<Lanes>>,
    cmax: Vec<f64>,
}

impl Eri3DerivWorkspace {
    /// The thread count the workspace was made for.
    pub fn nthreads(&self) -> usize {
        self.threads.len()
    }
}

/// The chunk count of `n` tasks (as [`schedule::ordered_ranges`] cuts them).
fn nchunks(n: usize) -> usize {
    n.div_ceil(n.div_ceil(NCHUNK).max(1))
}

/// At least `n` partials, each made by `make`.
fn ensure<T>(parts: &mut Vec<T>, n: usize, make: impl Fn() -> T) {
    while parts.len() < n {
        parts.push(make());
    }
}

impl<'a> Eri3DerivPlan<'a> {
    /// The plan over `pairs` (orbital pairs of `basis` in shell-index orientation, with
    /// derivative expansions of order `order` or more) and `apairs` (the (P, 1) pairs of
    /// `aux`, [`PairSet::aux`]; for an order-Two plan with second-order expansions too, whose
    /// bounds screen the Hessian). `order` is One (the gradients and derivative blocks) or Two
    /// (the Hessian as well). Refuses pair sets of the wrong kind, orientation or order.
    pub fn new(
        basis: &'a BasisSet,
        pairs: &'a PairSet,
        aux: &'a BasisSet,
        apairs: &'a PairSet,
        order: DerivOrder,
        opts: Eri3DerivOptions,
    ) -> Result<Self, IntsError> {
        assert!(
            order != DerivOrder::Zero,
            "a derivative plan is of order One or Two"
        );
        if pairs.kind() != PairKind::Orbital {
            return Err(IntsError::PairKind {
                need: PairKind::Orbital,
            });
        }
        if apairs.kind() != PairKind::Aux {
            return Err(IntsError::PairKind {
                need: PairKind::Aux,
            });
        }
        for p in [pairs, apairs] {
            // the auxiliary pairs' expansions matter for the Hessian's bounds only
            if p.deriv_order() < order
                && (p.kind() == PairKind::Orbital || order == DerivOrder::Two)
            {
                return Err(IntsError::Order {
                    need: order,
                    have: p.deriv_order(),
                });
            }
        }
        let aux3 = Aux3c::new(aux);
        let chunks = aux3.chunks(0..aux.shells.len());
        let mut aux_atom = vec![0usize; aux.nbf];
        for s in &aux.shells {
            aux_atom[s.ao_start..s.ao_start + s.nfunc()].fill(s.atom);
        }
        // bounds of the RI-J pairs' sizes, from all pairs (the layout is built on first use)
        let (mut nab, mut ndh, mut ndha) = (0, 0, 0);
        for p in pairs.pairs().iter().chain(apairs.pairs()) {
            nab = nab.max(p.nab());
            ndh = ndh.max(p.nprim * nherm(p.lab()));
            ndha = ndha.max(p.nprim * 3 * nherm(p.lab() + 1));
        }
        Ok(Eri3DerivPlan {
            basis,
            pairs,
            aux,
            apairs,
            order,
            opts,
            aux3,
            chunks,
            aux_atom,
            ao_atom: basis.shells.iter().map(|s| s.atom).collect(),
            ao: basis
                .shells
                .iter()
                .map(|s| (s.ao_start, s.nfunc()))
                .collect(),
            natom: basis.natom().max(aux.natom()),
            nab,
            ndh,
            ndha,
            rij: OnceLock::new(),
            gpairs: OnceLock::new(),
            dpairs: OnceLock::new(),
            hpairs: OnceLock::new(),
        })
    }

    /// The plan's derivative order.
    pub fn order(&self) -> DerivOrder {
        self.order
    }

    /// The plan's options.
    pub fn options(&self) -> &Eri3DerivOptions {
        &self.opts
    }

    /// The auxiliary basis prepared for the lane kernels.
    pub fn aux3c(&self) -> &Aux3c {
        &self.aux3
    }

    /// The rows of the gradients: the atoms of the orbital and auxiliary bases.
    pub fn natom(&self) -> usize {
        self.natom
    }

    /// Free the pair layouts built so far (each is built again on its next use): the memory
    /// between the phases of a calculation.
    pub fn free_layouts(&mut self) {
        self.rij = OnceLock::new();
        self.gpairs = OnceLock::new();
        self.dpairs = OnceLock::new();
        self.hpairs = OnceLock::new();
    }

    /// The orbital pairs kept by the RI-J pair threshold: their positions in the pair set.
    pub fn kept_pairs(&self) -> &[usize] {
        &self.rij().kept
    }

    /// The RI-J layout (see [`DfThresholds::pair`]).
    fn rij(&self) -> &Rij {
        self.rij.get_or_init(|| {
            let pairs = self.pairs;
            let q = pairs.schwarz();
            let (qaux, pair_thresh) = (self.aux3.qmax, self.opts.thresholds.pair);
            let kept: Vec<usize> = (0..pairs.len())
                .filter(|&i| pairs.pair(i).nprim > 0 && q[i] * qaux >= pair_thresh)
                .collect();
            let p3: Vec<Pair3c> = kept
                .par_iter()
                .map(|&i| Pair3c::from_shell_pair(pairs.pair(i)))
                .collect();
            let q3 = self
                .apairs
                .pairs()
                .iter()
                .map(Pair3c::from_shell_pair)
                .collect();
            Rij { kept, p3, q3 }
        })
    }

    /// The bytes of a workspace for a pool of threads, at most: the scratch per thread
    /// (without what grows with the blocks), the RI-J partials (at most [`NCHUNK`] of each) and
    /// the lane coefficients shared. The partials of the operations with a three-index density
    /// come on top: at most 64 gradients or Hessians.
    pub fn workspace_size(&self) -> WorkspaceSize {
        let naux = self.aux.nbf;
        let work = std::mem::size_of::<Work3c>() + 64 * work3c_lanes();
        let per_thread = work + 24 * naux + 8 * (self.nab + self.ndh + self.ndha);
        let nlanes: usize = self.chunks.iter().map(|ch| ncart(ch.l)).sum();
        WorkspaceSize {
            per_thread,
            shared: NCHUNK * (24 * self.natom + 8 * naux) + 64 * nlanes + 8 * self.chunks.len(),
        }
    }

    /// A workspace for a pool of `nthreads` threads (the operations must run in a pool of
    /// that size).
    pub fn workspace(&self, nthreads: usize) -> Eri3DerivWorkspace {
        let naux = self.aux.nbf;
        let thread = || Eri3Thread {
            work: Work3c::new(),
            gaux: vec![[0.0; 3]; naux],
            dw: Vec::with_capacity(self.nab),
            dh: Vec::with_capacity(self.ndh),
            dha: Vec::with_capacity(self.ndha),
            gw: GWork::default(),
            lanes: Vec::new(),
            h3: H3Work::default(),
            h21: Vec::new(),
            fidx: Vec::new(),
        };
        Eri3DerivWorkspace {
            threads: (0..nthreads).map(|_| thread()).collect(),
            part_g: Vec::new(),
            part_gamma: Vec::new(),
            part_h: Vec::new(),
            dc: self
                .chunks
                .iter()
                .map(|ch| vec![ZERO; ncart(ch.l)])
                .collect(),
            cmax: vec![0.0; self.chunks.len()],
        }
    }

    fn check_d(&self, d: &AoMatrix<'_>) {
        assert_eq!(d.n(), self.basis.nbf, "the density is nbf x nbf");
    }

    /// The lane coefficients of `c` and their largest magnitude per lane batch.
    fn lane_coefficients(&self, c: &[f64], ws: &mut Eri3DerivWorkspace) {
        assert_eq!(c.len(), self.aux.nbf, "c has naux values");
        for ((ch, dc), m) in self.chunks.iter().zip(&mut ws.dc).zip(&mut ws.cmax) {
            chunk_coefficients(ch, c, dc);
            *m = dc
                .iter()
                .flat_map(|l| l.iter())
                .fold(0.0f64, |m, &x| m.max(x.abs()));
        }
    }

    /// γ_P = sum_mn (P|mn) D_mn of the density `d` into `out` (naux, overwritten).
    pub fn gamma(&self, d: AoMatrix<'_>, out: &mut [f64], ws: &mut Eri3DerivWorkspace) {
        self.check_d(&d);
        assert_eq!(out.len(), self.aux.nbf, "γ has naux values");
        let p3 = &self.rij().p3[..];
        let (chunks, sh) = (&self.chunks[..], &self.basis.shells[..]);
        let thresh = self.opts.thresholds.prim;
        ensure(&mut ws.part_gamma, nchunks(p3.len()), || {
            vec![0.0; out.len()]
        });
        let nc = schedule::ordered_ranges(
            p3.len(),
            NCHUNK,
            &mut ws.threads,
            &mut ws.part_gamma,
            |t, g, r| {
                g.fill(0.0);
                for p in &p3[r] {
                    weighted_rows(&d, sh, p, &mut t.dw);
                    t.dh.resize(p.nprim() * nherm(p.lab), 0.0);
                    pair_hermite_density(p, &t.dw, &mut t.dh);
                    pair_gamma(p, chunks, &t.dh, g, thresh, &mut t.work);
                }
            },
        );
        out.fill(0.0);
        for p in &ws.part_gamma[..nc] {
            linus_num::det::add_into(out, p);
        }
    }

    /// sum_P c_P sum_mn ∂(P|mn)/∂X D_mn (the centers of m, n and P) of the density `d` and
    /// the auxiliary coefficients `c` into `out` ([`Self::natom`] rows, overwritten).
    pub fn coulomb_gradient(
        &self,
        d: AoMatrix<'_>,
        c: &[f64],
        out: &mut [[f64; 3]],
        ws: &mut Eri3DerivWorkspace,
    ) {
        self.check_d(&d);
        assert_eq!(out.len(), self.natom, "the gradient has natom rows");
        self.lane_coefficients(c, ws);
        let rij = self.rij();
        let (p3, kept, chunks) = (&rij.p3[..], &rij.kept[..], &self.chunks[..]);
        ensure(&mut ws.part_g, nchunks(p3.len()), || {
            vec![[0.0; 3]; self.natom]
        });
        let Eri3DerivWorkspace {
            threads,
            part_g,
            dc,
            cmax,
            ..
        } = ws;
        let (dc, cmax) = (&dc[..], &cmax[..]);
        let (drs, sh) = (self.pairs.derivs(), &self.basis.shells[..]);
        let aux_atom = &self.aux_atom[..];
        let thresh = self.opts.thresholds.deriv_prim;
        let nc = schedule::ordered_ranges(p3.len(), NCHUNK, threads, part_g, |t, g, r| {
            g.fill([0.0; 3]);
            t.gaux.fill([0.0; 3]);
            for k in r {
                let p = &p3[k];
                weighted_rows(&d, sh, p, &mut t.dw);
                t.dh.resize(p.nprim() * nherm(p.lab), 0.0);
                pair_hermite_density(p, &t.dw, &mut t.dh);
                t.dha.resize(p.nprim() * 3 * nherm(p.lab + 1), 0.0);
                pair_hermite_density_deriv(&drs[kept[k]], &t.dw, &mut t.dha);
                let (ga, gc) = pair_jgrad(
                    p,
                    chunks,
                    dc,
                    cmax,
                    &t.dh,
                    &t.dha,
                    &mut t.gaux,
                    thresh,
                    &mut t.work,
                );
                let (aa, ab) = (sh[p.sa].atom, sh[p.sb].atom);
                for x in 0..3 {
                    g[aa][x] += ga[x];
                    g[ab][x] -= ga[x] + gc[x];
                }
            }
            fold_aux(&t.gaux, aux_atom, g);
        });
        sum_parts(&part_g[..nc], out);
    }

    /// sum_PQ w_P c_Q ∂(P|Q)/∂X of the auxiliary vectors `w` and `c` into `out`
    /// ([`Self::natom`] rows, overwritten): each auxiliary shell P as the pair (P, 1) with the
    /// density w against the lane batches of c. The pair (P, 1) moves rigidly with P, so its
    /// own derivative is minus the sum of the lane-center ones (translational invariance).
    pub fn metric_gradient(
        &self,
        w: &[f64],
        c: &[f64],
        out: &mut [[f64; 3]],
        ws: &mut Eri3DerivWorkspace,
    ) {
        assert_eq!(w.len(), self.aux.nbf, "w has naux values");
        assert_eq!(out.len(), self.natom, "the gradient has natom rows");
        self.lane_coefficients(c, ws);
        let (q3, chunks) = (&self.rij().q3[..], &self.chunks[..]);
        ensure(&mut ws.part_g, nchunks(q3.len()), || {
            vec![[0.0; 3]; self.natom]
        });
        let Eri3DerivWorkspace {
            threads,
            part_g,
            dc,
            cmax,
            ..
        } = ws;
        let (dc, cmax) = (&dc[..], &cmax[..]);
        let (ash, aux_atom) = (&self.aux.shells[..], &self.aux_atom[..]);
        let thresh = self.opts.thresholds.deriv_prim;
        let nc = schedule::ordered_ranges(q3.len(), NCHUNK, threads, part_g, |t, g, r| {
            g.fill([0.0; 3]);
            t.gaux.fill([0.0; 3]);
            for qs in r {
                let p = &q3[qs];
                let q0 = ash[qs].ao_start;
                t.dw.clear();
                t.dw.extend_from_slice(&w[q0..q0 + p.nab()]);
                t.dh.resize(p.nprim() * nherm(p.lab), 0.0);
                pair_hermite_density(p, &t.dw, &mut t.dh);
                // the A-center part is not needed
                t.dha.clear();
                t.dha.resize(p.nprim() * 3 * nherm(p.lab + 1), 0.0);
                let (_, gc) = pair_jgrad(
                    p,
                    chunks,
                    dc,
                    cmax,
                    &t.dh,
                    &t.dha,
                    &mut t.gaux,
                    thresh,
                    &mut t.work,
                );
                let a = ash[qs].atom;
                for x in 0..3 {
                    g[a][x] -= gc[x];
                }
            }
            fold_aux(&t.gaux, aux_atom, g);
        });
        sum_parts(&part_g[..nc], out);
    }

    /// The context of an operation on `blk`; the block's first function and count.
    fn block_ctx(&self, blk: &Eri3Block) -> BlockCtx<'_> {
        let ash = &self.aux.shells;
        let r = &blk.shells;
        assert!(
            r.start < r.end && r.end <= ash.len(),
            "a block of auxiliary shells"
        );
        let p0 = ash[r.start].ao_start;
        let nq = ash[r.end - 1].ao_start + ash[r.end - 1].nfunc() - p0;
        BlockCtx {
            aux3c: &self.aux3,
            chunks: self.aux3.chunks(r.clone()),
            ao: &self.ao,
            ao_atom: &self.ao_atom,
            aux_atom: &self.aux_atom[p0..p0 + nq],
            nbf: self.basis.nbf,
            nq,
            natom: self.natom,
            op: blk.op,
            thresh: blk.thresh,
            isa: Isa::detect(),
        }
    }

    /// The Γ-contracted gradient's layout: the pairs with primitives, with their derivative
    /// bounds, less those below [`DfThresholds::deriv_pair`].
    fn grad_pairs(&self) -> &[Pair3cG<'a>] {
        self.gpairs.get_or_init(|| {
            let (pairs, at) = (self.pairs, &self.ao_atom);
            let qaux = self.aux3.qmax;
            let idx: Vec<usize> = (0..pairs.len())
                .filter(|&i| pairs.pair(i).nprim > 0)
                .collect();
            idx.par_iter()
                .map(|&i| {
                    let sp = pairs.pair(i);
                    Pair3cG::new(sp, pairs.deriv(i), at[sp.sa] == at[sp.sb])
                })
                .filter(|p| p.qmax * qaux >= self.opts.thresholds.deriv_pair)
                .collect()
        })
    }

    /// sum_{P in the block, mn} Γ_P\[mn\] ∂(P|op|mn)/∂X into `out` ([`Self::natom`] rows,
    /// overwritten), Γ from `weights`: for a shell pair (its first shell's functions a, the
    /// second's b), fill the slice (nq x na nb, \[P\]\[a nb + b\], P relative to the block's
    /// first function) with Γ_P\[ab\], times 2 for distinct shells (the pair stands for (a, b)
    /// and (b, a)), and return the largest |Γ| (0: the pair is skipped). Pairs, lane batches and
    /// primitive pairs whose derivative bound times that largest |Γ| is below `blk.thresh` (the
    /// Coulomb bounds scaled by [`Operator::schwarz_factor`]) are skipped.
    pub fn gradient<F>(
        &self,
        blk: &Eri3Block,
        weights: F,
        out: &mut [[f64; 3]],
        ws: &mut Eri3DerivWorkspace,
    ) where
        F: Fn(&ShellPair, &mut [f64]) -> f64 + Sync,
    {
        assert_eq!(out.len(), self.natom, "the gradient has natom rows");
        let pairs = self.grad_pairs();
        let cx = self.block_ctx(blk);
        let nch = nchunk_for(pairs.len(), 3 * self.natom, DEFAULT_BUDGET);
        ensure(&mut ws.part_g, nch, || vec![[0.0; 3]; self.natom]);
        let parts = &mut ws.part_g[..nch];
        grad_block(pairs, &cx, &weights, &mut ws.threads, parts);
        out.copy_from_slice(&parts[0]);
        for p in &parts[1..] {
            crate::core::util::add_grads_into(out, p);
        }
    }

    /// The derivative blocks' layout and their task order.
    fn block_pairs(&self) -> (&[Pair3cD], &[u32]) {
        let (p, o) = self.dpairs.get_or_init(|| {
            let qaux = self.aux3.qmax;
            let all: Vec<Pair3cD> = self
                .pairs
                .pairs()
                .par_iter()
                .zip(self.pairs.derivs())
                .filter(|(p, _)| p.nprim > 0)
                .map(|(p, d)| Pair3cD::new(p, d))
                .collect();
            let kept: Vec<Pair3cD> = all
                .into_iter()
                .filter(|p| p.qmax() * qaux >= self.opts.thresholds.deriv_pair)
                .collect();
            let order = (0..kept.len() as u32).collect();
            (kept, order)
        });
        (p, o)
    }

    /// The derivative integrals of the block into `out` (at least 3 nbf nq nbf values):
    /// `out[((k * nbf + a) * nq + P) * nbf + b]` = (∂_k a b|op|P), the derivative with respect
    /// to coordinate k of the center of a, for the functions a in `rows` (whole shells) and all
    /// b, P relative to the block's first function. The function pairs of shell pairs below
    /// [`DfThresholds::deriv_pair`] are never written (they stay as they were: zero the
    /// buffer once). `blk.thresh` screens primitive pairs against lane batches (the Coulomb
    /// bounds scaled by [`Operator::schwarz_factor`]).
    pub fn deriv_block(
        &self,
        blk: &Eri3Block,
        rows: Range<usize>,
        out: &mut [f64],
        ws: &mut Eri3DerivWorkspace,
    ) {
        let (pairs, order) = self.block_pairs();
        let cx = self.block_ctx(blk);
        deriv_block(pairs, &cx, rows, out, &mut ws.threads, order);
    }

    /// Build the Hessian's layout now; it is built on the first [`Self::hessian`] otherwise.
    /// Build each layout before the large buffers of the work around it: its bounds take
    /// over twice as long amid them (la18/def2-SVP: 2.2 against 0.8 s at one thread).
    pub fn prepare_hessian(&self) {
        self.hess_pairs();
    }

    /// Build the Gamma-contracted gradient's layout now (see [`Self::prepare_hessian`]).
    pub fn prepare_gradient(&self) {
        self.grad_pairs();
    }

    /// Build the derivative blocks' layout now (see [`Self::prepare_hessian`]).
    pub fn prepare_deriv_blocks(&self) {
        self.block_pairs();
    }

    /// The Hessian's layout: the pairs with primitives, with their screening bounds.
    fn hess_pairs(&self) -> &[Pair3cH] {
        self.hpairs.get_or_init(|| {
            let pairs = self.pairs;
            let q = pairs.schwarz();
            let (q1, q2) = pairs.deriv_bounds();
            let aq = self.apairs.schwarz_max();
            let (aq1, aq2) = self.apairs.deriv_bounds();

            let (aq1, aq2) = (
                aq1.iter().cloned().fold(0.0, f64::max),
                aq2.iter().cloned().fold(0.0, f64::max),
            );
            (0..pairs.len())
                .into_par_iter()
                .filter(|&i| pairs.pair(i).nprim > 0)
                .map(|i| {
                    Pair3cH::new(
                        pairs.pair(i),
                        pairs.deriv(i),
                        q[i],
                        q1[i],
                        q2[i],
                        aq,
                        aq1,
                        aq2,
                    )
                })
                .collect()
        })
    }

    /// sum_{P in the block, mn} ∂²(P|op|mn)/∂X∂Y Γ_P\[mn\] into `out` ((3 natom)^2, row-major,
    /// overwritten) for `gamma` (nq x nbf x nbf, \[P\]\[m\]\[n\], P relative to the block's
    /// first function, symmetric in mn). Pairs whose second-derivative bound times max |Γ| is
    /// below `blk.thresh` are skipped. An order-Two plan.
    pub fn hessian(
        &self,
        blk: &Eri3Block,
        gamma: &[f64],
        out: &mut [f64],
        ws: &mut Eri3DerivWorkspace,
    ) {
        assert_eq!(
            self.order,
            DerivOrder::Two,
            "the Hessian needs an order-Two plan"
        );
        let n3 = 3 * self.natom;
        assert_eq!(out.len(), n3 * n3, "the Hessian is (3 natom)^2");
        let pairs = self.hess_pairs();
        let cx = self.block_ctx(blk);
        assert!(
            gamma.len() >= cx.nq * cx.nbf * cx.nbf,
            "Γ is nq x nbf x nbf"
        );
        let nch = nchunk_for(pairs.len(), n3 * n3, DEFAULT_BUDGET);
        ensure(&mut ws.part_h, nch, || vec![0.0; n3 * n3]);
        let parts = &mut ws.part_h[..nch];
        hess_block(pairs, &cx, gamma, &mut ws.threads, parts);
        out.copy_from_slice(&parts[0]);
        for p in &parts[1..] {
            linus_num::det::add_into(out, p);
        }
    }
}

/// The lanes of [`Work3c`]'s fixed buffers.
fn work3c_lanes() -> usize {
    let n = nherm(super::MAX_L3C);
    2 * (super::MAX_L3C + 1) + 3 * n
}

/// A pair's weighted density rows into `dw`: D_mn for a diagonal shell pair, D_mn + D_nm
/// otherwise (the pair stands for both orders).
#[inline]
fn weighted_rows(d: &AoMatrix<'_>, sh: &[crate::basis::Shell], p: &Pair3c, dw: &mut Vec<f64>) {
    let (a0, b0) = (sh[p.sa].ao_start, sh[p.sb].ao_start);
    dw.clear();
    dw.extend((0..p.nab()).map(|x| {
        let (m, n) = (a0 + x / p.nb, b0 + x % p.nb);
        if p.sa == p.sb {
            d.get(m, n)
        } else {
            d.get(m, n) + d.get(n, m)
        }
    }));
}

/// Add a chunk's auxiliary-center derivatives onto the atoms of their functions.
#[inline]
fn fold_aux(gaux: &[[f64; 3]], aux_atom: &[usize], g: &mut [[f64; 3]]) {
    for (f, v) in gaux.iter().enumerate() {
        if v != &[0.0; 3] {
            let a = aux_atom[f];
            for x in 0..3 {
                g[a][x] += v[x];
            }
        }
    }
}

/// out = 0 + parts\[0\] + parts\[1\] + ..., atom by atom, in chunk order.
fn sum_parts(parts: &[Vec<[f64; 3]>], out: &mut [[f64; 3]]) {
    out.fill([0.0; 3]);
    for p in parts {
        crate::core::util::add_grads_into(out, p);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::basis::BasisDef;
    use crate::operator::Operator;
    use crate::pairs::PairOptions;
    use crate::reference::{eri_3c, eri_grad_3c, oneshot, Dens3c};

    /// gamma, the three-center and the metric gradient against the plain quartet references;
    /// the same bits at 1, 3 and 8 threads and on a reused workspace; the constructor refuses
    /// the wrong pair sets.
    #[test]
    fn rij_operations_match_references() {
        let z = [8, 1, 1];
        let pos = [[0.0, 0.0, 0.2], [1.4, 0.3, -0.9], [-1.5, -0.1, -0.8]];
        let basis =
            BasisSet::new(&BasisDef::load("def2-svp", None).unwrap(), &z, &pos, true).unwrap();
        let aux = BasisSet::new(
            &BasisDef::load("def2-universal-jkfit", None).unwrap(),
            &z,
            &pos,
            true,
        )
        .unwrap();
        let (n, naux) = (basis.nbf, aux.nbf);
        let pairs = PairSet::build(&basis, PairOptions::with_order(DerivOrder::One));
        let ap = PairSet::aux(&aux, PairOptions::with_order(DerivOrder::One));
        let d: Vec<f64> = (0..n * n)
            .map(|x| 0.02 / (1.0 + (x / n).abs_diff(x % n) as f64))
            .collect();
        let c: Vec<f64> = (0..naux).map(|k| 0.1 * ((k as f64) * 0.71).sin()).collect();
        let w: Vec<f64> = (0..naux).map(|k| 0.1 * ((k as f64) * 0.37).cos()).collect();
        let opts = Eri3DerivOptions {
            thresholds: DfThresholds::none(),
        };
        let plan = Eri3DerivPlan::new(&basis, &pairs, &aux, &ap, DerivOrder::One, opts).unwrap();
        let natom = plan.natom();
        type Run = (Vec<f64>, Vec<[f64; 3]>, Vec<[f64; 3]>);
        let run = |nt: usize, ws: &mut Eri3DerivWorkspace| -> Run {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(nt)
                .build()
                .unwrap();
            pool.install(|| {
                let dv = AoMatrix::row_major(&d, n);
                let mut gamma = vec![0.0; naux];
                plan.gamma(dv, &mut gamma, ws);
                let (mut g3, mut g2) = (vec![[0.0; 3]; natom], vec![[0.0; 3]; natom]);
                plan.coulomb_gradient(dv, &c, &mut g3, ws);
                plan.metric_gradient(&w, &c, &mut g2, ws);
                (gamma, g3, g2)
            })
        };
        let mut ws = plan.workspace(1);
        let (gamma, g3, g2) = run(1, &mut ws);
        // references: (P|mn) D_mn, the generic Gamma-contracted kernel, the 2c plan
        let j3 = eri_3c(&basis, &pairs, &aux, &ap);
        let close = |a: f64, b: f64, s: f64| (a - b).abs() <= 1e-11 * s.max(1.0);
        for p in 0..naux {
            let want: f64 = j3[p * n * n..(p + 1) * n * n]
                .iter()
                .zip(&d)
                .map(|(a, b)| a * b)
                .sum();
            assert!(
                close(gamma[p], want, want.abs()),
                "gamma {p}: {} vs {want}",
                gamma[p]
            );
        }
        let dens = Dens3c {
            full: None,
            aux_shells: 0..aux.shells.len(),
            separable: vec![(1.0, &c, &d)],
        };
        let r3 = eri_grad_3c(&basis, &pairs, &aux, &ap, &dens, 0.0, &Operator::Coulomb);
        let wc: Vec<f64> = (0..naux * naux)
            .map(|x| w[x / naux] * c[x % naux])
            .collect();
        let r2 = oneshot::grad_2c(&aux, &ap, &wc, 0.0, &Operator::Coulomb);
        for (got, want) in [(&g3, &r3), (&g2, &r2)] {
            let s = want.iter().flatten().fold(0.0f64, |m, x| m.max(x.abs()));
            assert!(s > 1e-4);
            for (a, b) in got.iter().flatten().zip(want.iter().flatten()) {
                assert!(close(*a, *b, s), "{a} vs {b}");
            }
        }
        // bits across threads; a reused workspace's scratch does not grow
        let bits = |x: &[f64], y: &[f64]| x.iter().zip(y).all(|(a, b)| a.to_bits() == b.to_bits());
        for nt in [3, 8, 8] {
            let mut ws = plan.workspace(nt);
            let caps = |ws: &Eri3DerivWorkspace| -> Vec<usize> {
                ws.threads
                    .iter()
                    .flat_map(|t| [t.dw.capacity(), t.dh.capacity(), t.dha.capacity()])
                    .collect()
            };
            let before = caps(&ws);
            for _ in 0..2 {
                let (ga, a3, a2) = run(nt, &mut ws);
                assert!(bits(&ga, &gamma));
                assert!(bits(a3.as_flattened(), g3.as_flattened()));
                assert!(bits(a2.as_flattened(), g2.as_flattened()));
            }
            assert_eq!(before, caps(&ws), "the scratch grew");
        }
        assert!(matches!(
            Eri3DerivPlan::new(&basis, &ap, &aux, &ap, DerivOrder::One, opts),
            Err(IntsError::PairKind { .. })
        ));
        assert!(matches!(
            Eri3DerivPlan::new(&basis, &pairs, &aux, &pairs, DerivOrder::One, opts),
            Err(IntsError::PairKind { .. })
        ));
        assert!(matches!(
            Eri3DerivPlan::new(&basis, &pairs, &aux, &ap, DerivOrder::Two, opts),
            Err(IntsError::Order { .. })
        ));
    }
}
