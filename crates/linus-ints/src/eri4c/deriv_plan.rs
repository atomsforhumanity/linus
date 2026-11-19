//! The plan of the four-center derivative contractions: the gradient
//! sum_{abcd} ∂(ab|cd)/∂X Γ_abcd and the Hessian sum_{abcd} ∂²(ab|cd)/∂X∂Y Γ_abcd of density
//! terms Γ ([`DensityTerms`]), into the caller's buffers, over a caller-owned workspace.
//!
//! The work walks the bra pairs (the most expensive first) and their kets with 8-fold
//! permutational symmetry and Schwarz x density screening, and reduces in a fixed number of
//! strided chunks (`linus_num::det::par_fold_det_into`), one accumulator per chunk in the
//! workspace: the result has the same bits for any number of threads.

use super::deriv::{grad_4c_into, hess_4c_into, DerivWork, GradAcc, HessAcc, ScratchLens};
use crate::basis::BasisSet;
use crate::contract::DensityTerms;
use crate::engine::WorkspaceSize;
use crate::error::IntsError;
use crate::operator::Operator;
use crate::pairs::{DerivOrder, PairKind, PairSet};
use linus_num::det::{nchunk_for, DEFAULT_BUDGET};

/// The options of an [`Eri4DerivPlan`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Eri4DerivOptions {
    /// Screening threshold on the Schwarz x density estimate of a quartet's contribution.
    pub thresh: f64,
    /// The two-electron operator (a/r + b erf(omega r)/r) of every term. The pairs' Schwarz
    /// factors stay the Coulomb ones: they bound the erf integrals too, scaled by
    /// [`Operator::schwarz_factor`].
    pub op: Operator,
}

impl Default for Eri4DerivOptions {
    fn default() -> Self {
        Eri4DerivOptions {
            thresh: 1e-12,
            op: Operator::Coulomb,
        }
    }
}

/// The four-center derivative contractions of one order (One: the gradient, Two: the
/// Hessian) for one operator, over a pair set in shell-index orientation with derivative
/// expansions of at least that order.
pub struct Eri4DerivPlan<'a> {
    basis: &'a BasisSet,
    pairs: &'a PairSet,
    order: DerivOrder,
    opts: Eri4DerivOptions,
    /// chunks of the deterministic reduction: fixed by the problem's size, not the threads
    nchunk: usize,
    /// the length of a chunk's result: 3 natom (gradient) or (3 natom)^2 (Hessian)
    out_len: usize,
    /// the largest pair: angular momentum sum, functions, primitives
    lab: usize,
    nab: usize,
    kq: usize,
}

/// The caller-owned memory of an [`Eri4DerivPlan`]: one accumulator per reduction chunk
/// (the integral scratch, the partial result and a quartet's Γ block), sized for the plan's
/// largest quartet, so a contraction allocates nothing. Reusable across calls of its plan.
pub struct Eri4DerivWorkspace {
    order: DerivOrder,
    out_len: usize,
    grad: Vec<GradAcc>,
    hess: Vec<HessAcc>,
}

impl Eri4DerivWorkspace {
    fn nchunk(&self) -> usize {
        self.grad.len() + self.hess.len()
    }
}

impl<'a> Eri4DerivPlan<'a> {
    /// The plan of order `order` (One or Two) over `pairs` (built from `basis`). Refuses
    /// auxiliary pairs, a pair set of another orientation than ShellIndex or of a lower
    /// derivative order.
    pub fn new(
        basis: &'a BasisSet,
        pairs: &'a PairSet,
        order: DerivOrder,
        opts: Eri4DerivOptions,
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
        if pairs.deriv_order() < order {
            return Err(IntsError::Order {
                need: order,
                have: pairs.deriv_order(),
            });
        }
        let n3 = 3 * basis.natom();
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
        Ok(Eri4DerivPlan {
            basis,
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
    pub fn options(&self) -> &Eri4DerivOptions {
        &self.opts
    }

    /// The integral scratch of one chunk for `ncoulomb` Coulomb terms (the gradient's
    /// Hermite-density path; the Hessian folds them into the Γ block).
    fn scratch(&self, ncoulomb: usize) -> ScratchLens {
        let nj = if self.order == DerivOrder::One {
            ncoulomb
        } else {
            0
        };
        ScratchLens::new(self.order.get(), self.lab, self.nab, self.kq, nj)
    }

    /// The bytes of a workspace for density terms with up to `ncoulomb` Coulomb terms; all
    /// shared, as the chunks do not depend on the threads.
    pub fn workspace_size(&self, ncoulomb: usize) -> WorkspaceSize {
        let per_chunk = self.scratch(ncoulomb).bytes() + 8 * (self.out_len + self.nab * self.nab);
        WorkspaceSize {
            per_thread: 0,
            shared: self.nchunk * per_chunk,
        }
    }

    /// A workspace for density terms with up to `ncoulomb` Coulomb terms (more grow its
    /// scratch on first use).
    pub fn workspace(&self, ncoulomb: usize) -> Eri4DerivWorkspace {
        let s = self.scratch(ncoulomb);
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
        Eri4DerivWorkspace {
            order: self.order,
            out_len: self.out_len,
            grad,
            hess,
        }
    }

    fn check(&self, ws: &Eri4DerivWorkspace, need: DerivOrder) {
        assert_eq!(self.order, need, "the plan is of order {:?}", self.order);
        assert!(
            ws.order == self.order && ws.out_len == self.out_len && ws.nchunk() == self.nchunk,
            "the workspace of another plan"
        );
    }

    /// The gradient sum_{abcd} ∂(ab|cd)/∂X Γ_abcd of `terms` into `out` (natom rows,
    /// overwritten). An order-One plan.
    pub fn gradient(
        &self,
        terms: &DensityTerms,
        out: &mut [[f64; 3]],
        ws: &mut Eri4DerivWorkspace,
    ) {
        self.check(ws, DerivOrder::One);
        assert_eq!(out.len(), self.out_len / 3, "the gradient has natom rows");
        for (w, g, _) in &mut ws.grad {
            w.op = self.opts.op;
            g.fill([0.0; 3]);
        }
        let (b, p, o) = (self.basis, self.pairs, &self.opts);
        grad_4c_into(b, p, terms, o.thresh, &o.op, &mut ws.grad);
        out.copy_from_slice(&ws.grad[0].1);
    }

    /// The Hessian sum_{abcd} ∂²(ab|cd)/∂X∂Y Γ_abcd of `terms` into `out` ((3 natom)^2,
    /// row-major, overwritten). An order-Two plan.
    pub fn hessian(&self, terms: &DensityTerms, out: &mut [f64], ws: &mut Eri4DerivWorkspace) {
        self.check(ws, DerivOrder::Two);
        assert_eq!(out.len(), self.out_len, "the Hessian is (3 natom)^2");
        for (w, h, _) in &mut ws.hess {
            w.op = self.opts.op;
            h.fill(0.0);
        }
        let (b, p, o) = (self.basis, self.pairs, &self.opts);
        hess_4c_into(b, p, terms, o.thresh, &o.op, &mut ws.hess);
        out.copy_from_slice(&ws.hess[0].1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::basis::BasisDef;
    use crate::pairs::PairOptions;

    fn basis() -> BasisSet {
        let def = BasisDef::load("def2-tzvp", None).unwrap();
        let pos = [[0.0, 0.0, 0.2], [1.4, 0.3, -0.9], [-1.5, -0.1, -0.8]];
        BasisSet::new(&def, &[8, 1, 1], &pos, true).unwrap()
    }

    fn density(n: usize) -> Vec<f64> {
        let f = |i: usize, j: usize| 0.02 / (1.0 + i.abs_diff(j) as f64);
        (0..n * n).map(|x| f(x / n, x % n)).collect()
    }

    /// The reserved scratch is the kernels' high-water mark: a contraction grows nothing; the
    /// results have the same bits at 1, 3 and 8 threads and on a reused workspace; and the
    /// constructor refuses pair sets of a lower order or another orientation.
    #[test]
    fn workspace_bounds_scratch_and_bits_match() {
        let b = basis();
        let d = density(b.nbf);
        let terms = DensityTerms::default()
            .coulomb(0.5, &d, &d)
            .exchange(-0.25, &d, &d);
        let op = Operator::Attenuated {
            coulomb: 0.2,
            long_range: 0.8,
            omega: 0.3,
        };
        let opts = Eri4DerivOptions { thresh: 1e-12, op };
        let pairs = PairSet::build(&b, PairOptions::with_order(DerivOrder::Two));
        let n3 = 3 * b.natom();
        for order in [DerivOrder::One, DerivOrder::Two] {
            let plan = Eri4DerivPlan::new(&b, &pairs, order, opts).unwrap();
            let mut ws = plan.workspace(terms.coulomb.len());
            let lens = |ws: &Eri4DerivWorkspace| -> Vec<[usize; 19]> {
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
                let v = pool.install(|| {
                    if order == DerivOrder::One {
                        let mut g = vec![[0.0; 3]; b.natom()];
                        plan.gradient(&terms, &mut g, &mut ws);
                        g.iter().flatten().copied().collect()
                    } else {
                        let mut h = vec![0.0; n3 * n3];
                        plan.hessian(&terms, &mut h, &mut ws);
                        h
                    }
                });
                runs.push(v);
            }
            assert_eq!(before, lens(&ws), "{order:?}: the scratch grew");
            for r in &runs[1..] {
                assert!(r
                    .iter()
                    .zip(&runs[0])
                    .all(|(x, y)| x.to_bits() == y.to_bits()));
            }
            assert!(runs[0].iter().any(|x| x.abs() > 1e-4));
        }
        let p1 = PairSet::build(&b, PairOptions::with_order(DerivOrder::One));
        assert!(matches!(
            Eri4DerivPlan::new(&b, &p1, DerivOrder::Two, opts),
            Err(IntsError::Order { .. })
        ));
    }
}
