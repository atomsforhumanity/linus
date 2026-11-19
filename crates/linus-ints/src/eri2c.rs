//! Two-center integrals (P|Q) of an auxiliary basis (the density-fitting metric): the plan of
//! their nuclear derivatives.
//!
//! A two-center integral is a four-center one with the unit s function (exponent 0) as the
//! second shell of each pair, (P|Q) = (P 1|Q 1): the auxiliary pairs of [`PairSet::aux`] feed
//! the four-center derivative kernels ([`crate::eri4c::deriv`]) unchanged, and the derivatives
//! of the unit functions vanish. [`Eri2DerivPlan`] names the operations:
//! - [`Eri2DerivPlan::gradient`]: sum_PQ ∂(P|Q)/∂X γ_PQ for a naux x naux matrix γ;
//! - [`Eri2DerivPlan::hessian`]: sum_PQ ∂²(P|Q)/∂X∂Y γ_PQ;
//! - [`Eri2DerivPlan::deriv_integrals`]: the first-derivative integrals themselves, (∂_k P|Q)
//!   for every P, Q and direction k, for contractions done by the caller (the metric terms
//!   of density-fitted Hessians).
//!
//! The contractions walk the shell pairs (P, Q), Q <= P, with Schwarz screening of the
//! derivative integrals and reduce in a fixed number of strided chunks over P
//! ([`crate::engine::schedule::ordered`]), one accumulator per chunk in the caller's
//! [`Eri2DerivWorkspace`]: the result has the same bits for any number of threads.

use crate::basis::{BasisSet, Shell};
use crate::contract::AoMatrix;
use crate::core::util::add_grads_into;
use crate::engine::{schedule, WorkspaceSize};
use crate::eri4c::deriv::{
    add_hess_blocks, quartet_grad, quartet_hess, DerivWork, GradAcc, HessAcc, ScratchLens,
    CENTER_A, CENTER_C,
};
use crate::eri4c::jk_deriv::{eri_quartet_deriv_a, DerivAWork};
use crate::error::IntsError;
use crate::operator::Operator;
use crate::pairs::{DerivOrder, PairKind, PairSet};
use linus_num::det::{nchunk_for, DEFAULT_BUDGET};
use rayon::prelude::*;

/// The options of an [`Eri2DerivPlan`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Eri2DerivOptions {
    /// Screening threshold on the Schwarz estimate of a shell pair's derivative integrals.
    pub thresh: f64,
    /// The two-electron operator (a/r + b erf(omega r)/r). The pairs' Schwarz factors stay
    /// the Coulomb ones, scaled by [`Operator::schwarz_factor`].
    pub op: Operator,
}

impl Default for Eri2DerivOptions {
    fn default() -> Self {
        Eri2DerivOptions {
            thresh: 1e-12,
            op: Operator::Coulomb,
        }
    }
}

/// The two-center derivative contractions of one order (One: the gradient, Two: the
/// Hessian) and the first-derivative integrals, for one operator, over the (P, 1) pairs of an
/// auxiliary basis with derivative expansions of at least that order.
pub struct Eri2DerivPlan<'a> {
    aux: &'a BasisSet,
    pairs: &'a PairSet,
    order: DerivOrder,
    opts: Eri2DerivOptions,
    /// chunks of the deterministic reduction: fixed by the problem's size, not the threads
    nchunk: usize,
    /// the length of a chunk's result: 3 natom (gradient) or (3 natom)^2 (Hessian)
    out_len: usize,
    /// the largest pair: angular momentum, functions, primitives
    lab: usize,
    nab: usize,
    kq: usize,
}

/// The caller-owned memory of an [`Eri2DerivPlan`]'s contractions: one accumulator per
/// reduction chunk (the integral scratch, the partial result and a shell pair's γ block),
/// sized for the plan's largest shells, so a contraction allocates nothing. Reusable across
/// calls of its plan.
pub struct Eri2DerivWorkspace {
    order: DerivOrder,
    out_len: usize,
    grad: Vec<GradAcc>,
    hess: Vec<HessAcc>,
}

impl Eri2DerivWorkspace {
    fn nchunk(&self) -> usize {
        self.grad.len() + self.hess.len()
    }
}

impl<'a> Eri2DerivPlan<'a> {
    /// The plan of order `order` (One or Two) over `pairs`, the auxiliary pairs of `aux`
    /// ([`PairSet::aux`]). Refuses orbital pairs and pair sets of a lower derivative order.
    pub fn new(
        aux: &'a BasisSet,
        pairs: &'a PairSet,
        order: DerivOrder,
        opts: Eri2DerivOptions,
    ) -> Result<Self, IntsError> {
        assert!(
            order != DerivOrder::Zero,
            "a derivative plan is of order One or Two"
        );
        if pairs.kind() != PairKind::Aux {
            return Err(IntsError::PairKind {
                need: PairKind::Aux,
            });
        }
        if pairs.deriv_order() < order {
            return Err(IntsError::Order {
                need: order,
                have: pairs.deriv_order(),
            });
        }
        let n3 = 3 * aux.natom();
        let out_len = if order == DerivOrder::One {
            n3
        } else {
            n3 * n3
        };
        let (mut lab, mut nab, mut kq) = (0, 0, 0);
        for p in pairs.pairs() {
            lab = lab.max(p.lab());
            nab = nab.max(p.nab());
            kq = kq.max(p.nprim);
        }
        Ok(Eri2DerivPlan {
            aux,
            pairs,
            order,
            opts,
            nchunk: nchunk_for(pairs.len(), out_len, DEFAULT_BUDGET),
            out_len,
            lab,
            nab,
            kq,
        })
    }

    /// The plan's derivative order.
    pub fn order(&self) -> DerivOrder {
        self.order
    }

    /// The plan's options.
    pub fn options(&self) -> &Eri2DerivOptions {
        &self.opts
    }

    fn scratch(&self) -> ScratchLens {
        ScratchLens::new(self.order.get(), self.lab, self.nab, self.kq, 0)
    }

    /// The bytes of a workspace of the contractions; all shared, as the chunks do not depend
    /// on the threads.
    pub fn workspace_size(&self) -> WorkspaceSize {
        let per_chunk = self.scratch().bytes() + 8 * (self.out_len + self.nab * self.nab);
        WorkspaceSize {
            per_thread: 0,
            shared: self.nchunk * per_chunk,
        }
    }

    /// A workspace of the contractions.
    pub fn workspace(&self) -> Eri2DerivWorkspace {
        let s = self.scratch();
        let work = || {
            let mut w = DerivWork::with_op(self.opts.op);
            w.reserve(&s);
            w
        };
        let gbuf = || Vec::with_capacity(self.nab * self.nab);
        let n = self.nchunk;
        let (grad, hess) = if self.order == DerivOrder::One {
            let natom = self.out_len / 3;
            let g = (0..n).map(|_| (work(), vec![[0.0; 3]; natom], gbuf()));
            (g.collect(), Vec::new())
        } else {
            let h = (0..n).map(|_| (work(), vec![0.0; self.out_len], gbuf()));
            (Vec::new(), h.collect())
        };
        Eri2DerivWorkspace {
            order: self.order,
            out_len: self.out_len,
            grad,
            hess,
        }
    }

    fn check(&self, gamma: &AoMatrix<'_>, ws: &Eri2DerivWorkspace, need: DerivOrder) {
        assert_eq!(self.order, need, "the plan is of order {:?}", self.order);
        assert!(
            ws.order == self.order && ws.out_len == self.out_len && ws.nchunk() == self.nchunk,
            "the workspace of another plan"
        );
        assert_eq!(gamma.n(), self.aux.nbf, "γ is naux x naux");
    }

    /// The gradient sum_PQ ∂(P|Q)/∂X γ_PQ into `out` (natom rows, overwritten). An order-One
    /// plan.
    pub fn gradient(&self, gamma: AoMatrix<'_>, out: &mut [[f64; 3]], ws: &mut Eri2DerivWorkspace) {
        self.check(&gamma, ws, DerivOrder::One);
        assert_eq!(out.len(), self.out_len / 3, "the gradient has natom rows");
        for (w, g, _) in &mut ws.grad {
            w.op = self.opts.op;
            g.fill([0.0; 3]);
        }
        // slices, not the set: their loads hoist out of the loops (a &PairSet is not Freeze)
        let (prs, drs) = (self.pairs.pairs(), self.pairs.derivs());
        let thresh = self.opts.op.screen_threshold(self.opts.thresh);
        let ash = &self.aux.shells;
        let q = self.pairs.schwarz();
        let (qd, _) = self.pairs.deriv_bounds();
        schedule::ordered(
            self.pairs.len(),
            &mut ws.grad,
            |(w, g, gbuf), i| {
                for j in 0..=i {
                    if qd[i] * q[j] + q[i] * qd[j] < thresh {
                        continue;
                    }
                    let (bra, ket) = (&prs[i], &prs[j]);
                    gamma_block(
                        &gamma,
                        ash[i].ao_start,
                        ash[j].ao_start,
                        i == j,
                        bra.na,
                        ket.na,
                        gbuf,
                    );
                    let gq = quartet_grad(bra, &drs[i], ket, &drs[j], Some(gbuf), &[], w);
                    for k in 0..3 {
                        g[ash[i].atom][k] += gq[CENTER_A][k];
                        g[ash[j].atom][k] += gq[CENTER_C][k];
                    }
                }
            },
            |acc, b| add_grads_into(&mut acc.1, &b.1),
        );
        out.copy_from_slice(&ws.grad[0].1);
    }

    /// The Hessian sum_PQ ∂²(P|Q)/∂X∂Y γ_PQ into `out` ((3 natom)^2, row-major, overwritten).
    /// An order-Two plan.
    pub fn hessian(&self, gamma: AoMatrix<'_>, out: &mut [f64], ws: &mut Eri2DerivWorkspace) {
        self.check(&gamma, ws, DerivOrder::Two);
        assert_eq!(out.len(), self.out_len, "the Hessian is (3 natom)^2");
        for (w, h, _) in &mut ws.hess {
            w.op = self.opts.op;
            h.fill(0.0);
        }
        // slices, not the set: their loads hoist out of the loops (a &PairSet is not Freeze)
        let (prs, drs) = (self.pairs.pairs(), self.pairs.derivs());
        let thresh = self.opts.op.screen_threshold(self.opts.thresh);
        let n3 = 3 * self.aux.natom();
        let ash = &self.aux.shells;
        let q = self.pairs.schwarz();
        let (qd, qd2) = self.pairs.deriv_bounds();
        schedule::ordered(
            self.pairs.len(),
            &mut ws.hess,
            |(w, h, gbuf), i| {
                for j in 0..=i {
                    if qd2[i] * q[j] + 2.0 * qd[i] * qd[j] + q[i] * qd2[j] < thresh {
                        continue;
                    }
                    let (bra, ket) = (&prs[i], &prs[j]);
                    gamma_block(
                        &gamma,
                        ash[i].ao_start,
                        ash[j].ao_start,
                        i == j,
                        bra.na,
                        ket.na,
                        gbuf,
                    );
                    let hp = quartet_hess(bra, &drs[i], ket, &drs[j], gbuf, w);
                    let atoms = [ash[i].atom, ash[i].atom, ash[j].atom, ash[j].atom];
                    add_hess_blocks(h, n3, atoms, &hp.center_blocks());
                }
            },
            |acc, b| linus_num::det::add_into(&mut acc.1, &b.1),
        );
        out.copy_from_slice(&ws.hess[0].1);
    }

    /// The first-derivative integrals into `out` (3 naux^2, overwritten):
    /// `out[(k * naux + p) * naux + q]` = (∂_k p|q), the derivative with respect to coordinate
    /// k (x, y, z) of the center of the auxiliary function p, for every p and q. The derivative
    /// of (p|q) with respect to atom A's coordinate k is then
    /// `[atom(p) = A] out[k][p][q] + [atom(q) = A] out[k][q][p]`.
    ///
    /// Each value is one quartet kernel's (no reduction); the shells of p run in parallel, on
    /// scratch per worker of the size of the plan's largest pair.
    pub fn deriv_integrals(&self, out: &mut [f64]) {
        let naux = self.aux.nbf;
        let n2 = naux * naux;
        assert_eq!(out.len(), 3 * n2, "the derivative integrals are 3 naux^2");
        let ash = &self.aux.shells;
        let (prs, drs) = (self.pairs.pairs(), self.pairs.derivs());
        // the rows of each shell of p in the three planes k
        let (px, rest) = out.split_at_mut(n2);
        let (py, pz) = rest.split_at_mut(n2);
        let mut planes: Vec<[&mut [f64]; 3]> = shell_rows(px, ash, naux)
            .zip(shell_rows(py, ash, naux))
            .zip(shell_rows(pz, ash, naux))
            .map(|((x, y), z)| [x, y, z])
            .collect();
        let op = self.opts.op;
        let bmax = 3 * self.nab * self.nab;
        planes.par_iter_mut().enumerate().for_each_init(
            || (DerivAWork::with_op(op), vec![0.0; bmax]),
            |(w, buf), (ps, dst)| {
                let (bra, bd) = (&prs[ps], &drs[ps]);
                let np = bra.na;
                for (qs, ket) in prs.iter().enumerate() {
                    let nq = ket.na;
                    let q0 = ash[qs].ao_start;
                    eri_quartet_deriv_a(bra, bd, ket, buf, w);
                    for (k, plane) in dst.iter_mut().enumerate() {
                        for p in 0..np {
                            let row = &mut plane[p * naux + q0..p * naux + q0 + nq];
                            row.copy_from_slice(&buf[(k * np + p) * nq..(k * np + p + 1) * nq]);
                        }
                    }
                }
            },
        );
    }
}

/// The rows of each shell's functions in a matrix of `ncol` columns, in shell order.
fn shell_rows<'b>(
    m: &'b mut [f64],
    shells: &'b [Shell],
    ncol: usize,
) -> impl Iterator<Item = &'b mut [f64]> {
    let mut rest = m;
    shells.iter().map(move |s| {
        let (r, tail) = std::mem::take(&mut rest).split_at_mut(s.nfunc() * ncol);
        rest = tail;
        r
    })
}

/// The symmetrized γ block of the shells of p (first function `p0`, `np` functions) and q:
/// gbuf[a * nq + b] = f (γ_pq + γ_qp) / 2, f = 2 for distinct shells (the pair stands for both
/// orders).
#[inline]
fn gamma_block(
    gamma: &AoMatrix<'_>,
    p0: usize,
    q0: usize,
    same: bool,
    np: usize,
    nq: usize,
    gbuf: &mut Vec<f64>,
) {
    let f = if same { 1.0 } else { 2.0 };
    gbuf.resize(np * nq, 0.0);
    for a in 0..np {
        for b in 0..nq {
            gbuf[a * nq + b] = 0.5 * f * (gamma.get(p0 + a, q0 + b) + gamma.get(q0 + b, p0 + a));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::basis::BasisDef;
    use crate::pairs::PairOptions;

    fn aux_at(pos: &[[f64; 3]]) -> BasisSet {
        let def = BasisDef::load("def2-universal-jkfit", None).unwrap();
        BasisSet::new(&def, &[8, 1, 1], pos, true).unwrap()
    }

    const POS: [[f64; 3]; 3] = [[0.0, 0.0, 0.2], [1.4, 0.3, -0.9], [-1.5, -0.1, -0.8]];

    fn gamma(n: usize) -> Vec<f64> {
        // not symmetric: the contractions symmetrize
        (0..n * n)
            .map(|x| 0.03 * ((x / n) as f64 * 0.37 + (x % n) as f64 * 0.11).sin())
            .collect()
    }

    fn grad(aux: &BasisSet, g: &[f64]) -> Vec<f64> {
        let ap = PairSet::aux(aux, PairOptions::with_order(DerivOrder::One));
        let opts = Eri2DerivOptions {
            thresh: 0.0,
            ..Default::default()
        };
        let plan = Eri2DerivPlan::new(aux, &ap, DerivOrder::One, opts).unwrap();
        let mut ws = plan.workspace();
        let mut out = vec![[0.0; 3]; aux.natom()];
        plan.gradient(AoMatrix::row_major(g, aux.nbf), &mut out, &mut ws);
        out.iter().flatten().copied().collect()
    }

    /// The gradient is the contraction of the derivative integrals with γ; the Hessian is the
    /// central difference of the gradient; both have the same bits at 1, 3 and 8 threads and
    /// on a reused workspace, whose scratch does not grow; the constructor refuses orbital
    /// pairs and pair sets of a lower order.
    #[test]
    fn contractions_match_integrals_and_differences() {
        let aux = aux_at(&POS);
        let n = aux.nbf;
        let g = gamma(n);
        let ap = PairSet::aux(&aux, PairOptions::with_order(DerivOrder::Two));
        let opts = Eri2DerivOptions {
            thresh: 0.0,
            ..Default::default()
        };
        // the gradient against sum_{p on A} sum_q (γ_pq + γ_qp) (∂_k p|q)
        let p1 = Eri2DerivPlan::new(&aux, &ap, DerivOrder::One, opts).unwrap();
        let mut d = vec![0.0; 3 * n * n];
        p1.deriv_integrals(&mut d);
        let mut want = vec![[0.0; 3]; aux.natom()];
        for s in &aux.shells {
            for p in s.ao_start..s.ao_start + s.nfunc() {
                for q in 0..n {
                    for k in 0..3 {
                        want[s.atom][k] += (g[p * n + q] + g[q * n + p]) * d[(k * n + p) * n + q];
                    }
                }
            }
        }
        let have = grad(&aux, &g);
        let scale = have.iter().fold(0.0f64, |m, x| m.max(x.abs()));
        assert!(scale > 1e-3);
        for (a, b) in have.iter().zip(want.iter().flatten()) {
            assert!((a - b).abs() < 1e-11 * scale, "{a} vs {b}");
        }
        // bits across thread counts and workspace reuse, for both orders
        let n3 = 3 * aux.natom();
        for order in [DerivOrder::One, DerivOrder::Two] {
            let plan = Eri2DerivPlan::new(&aux, &ap, order, opts).unwrap();
            let mut ws = plan.workspace();
            let lens = |ws: &Eri2DerivWorkspace| -> Vec<[usize; 19]> {
                let g = ws.grad.iter().map(|a| a.0.lens());
                g.chain(ws.hess.iter().map(|a| a.0.lens())).collect()
            };
            let before = lens(&ws);
            let mut runs: Vec<Vec<f64>> = Vec::new();
            for nt in [1, 3, 8, 8] {
                let pool = rayon::ThreadPoolBuilder::new()
                    .num_threads(nt)
                    .build()
                    .unwrap();
                runs.push(pool.install(|| {
                    let gm = AoMatrix::row_major(&g, n);
                    if order == DerivOrder::One {
                        let mut o = vec![[0.0; 3]; aux.natom()];
                        plan.gradient(gm, &mut o, &mut ws);
                        o.iter().flatten().copied().collect()
                    } else {
                        let mut h = vec![0.0; n3 * n3];
                        plan.hessian(gm, &mut h, &mut ws);
                        h
                    }
                }));
            }
            assert_eq!(before, lens(&ws), "{order:?}: the scratch grew");
            for r in &runs[1..] {
                assert!(r
                    .iter()
                    .zip(&runs[0])
                    .all(|(x, y)| x.to_bits() == y.to_bits()));
            }
            if order == DerivOrder::Two {
                // central differences of the gradient
                let h = &runs[0];
                let step = 1e-4;
                let mut worst: f64 = 0.0;
                for x in 0..n3 {
                    let at = |s: f64| {
                        let mut pos = POS;
                        pos[x / 3][x % 3] += s;
                        grad(&aux_at(&pos), &g)
                    };
                    let (gp, gm) = (at(step), at(-step));
                    for y in 0..n3 {
                        let fd = (gp[y] - gm[y]) / (2.0 * step);
                        worst = worst.max((h[x * n3 + y] - fd).abs());
                    }
                }
                let hs = h.iter().fold(0.0f64, |m, v| m.max(v.abs()));
                assert!(worst < 1e-6 * hs.max(1.0), "worst {worst:e} (|H| {hs:e})");
            }
        }
        let p0 = PairSet::aux(&aux, PairOptions::default());
        assert!(matches!(
            Eri2DerivPlan::new(&aux, &p0, DerivOrder::One, opts),
            Err(IntsError::Order { .. })
        ));
        let orb = PairSet::build(&aux, PairOptions::with_order(DerivOrder::One));
        assert!(matches!(
            Eri2DerivPlan::new(&aux, &orb, DerivOrder::One, opts),
            Err(IntsError::PairKind { .. })
        ));
    }
}
