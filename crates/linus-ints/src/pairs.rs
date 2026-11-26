//! Shell pairs: the one owner of the per-geometry pair data every integral engine reads.
//!
//! A [`PairSet`] holds, for every pair of shells of a basis, the Hermite expansion E of the
//! product (contraction coefficients and the transformation to the shells' functions folded
//! in, SHARK's layout) and, up to its [`DerivOrder`], the expansions E^A and E^AA of the
//! derivatives with respect to the first shell's center, all from one primitive loop
//! (`core::pair_data`). Its bounds are computed on first use and kept: the Schwarz factors
//! Q_ab = max |(ab|ab)|^1/2 of the Coulomb operator, the bounds of the derivative integrals,
//! and the primitive Schwarz factors of the batched four-center kernels.
//!
//! The pairs (i, j) with i >= j come in the order k = i (i + 1) / 2 + j ([`PairSet::index`]),
//! the first shell of a pair the one of higher index: one orientation for every engine. The
//! batched four-center kernels, which take the shell of higher angular momentum first, orient
//! their own copy of the expansions (`eri4c::batch::BatchedPairs`, a transposition of the
//! stored coefficients), so one pair set per geometry serves them too.
//!
//! [`PairSet::aux`] gives the one-center "pairs" (P, 1) of an auxiliary basis, with the
//! constant function 1 as the second shell, which turn two- and three-center integrals into
//! four-center ones.

use crate::basis::BasisSet;
use crate::core::deriv::PairDeriv;
use crate::core::hermite::{ShellPair, ShellView};
use crate::core::pair_data::{self, PRIM_PAIR_CUT};
use crate::eri4c::deriv::schwarz_deriv;
use crate::eri4c::{eri_quartet, EriWork};
use crate::operator::Operator;
use rayon::prelude::*;
use std::sync::OnceLock;

/// The derivative order of integrals or of pair data: values, first or second derivatives
/// with respect to the nuclear positions.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum DerivOrder {
    #[default]
    Zero,
    One,
    Two,
}

impl DerivOrder {
    /// The order as a number (0, 1, 2).
    #[inline]
    pub const fn get(self) -> usize {
        self as usize
    }
}

impl TryFrom<usize> for DerivOrder {
    type Error = usize;

    /// The order `n` (0, 1 or 2); larger orders are returned as the error.
    fn try_from(n: usize) -> Result<Self, usize> {
        match n {
            0 => Ok(DerivOrder::Zero),
            1 => Ok(DerivOrder::One),
            2 => Ok(DerivOrder::Two),
            _ => Err(n),
        }
    }
}

/// How a [`PairSet`] is built.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PairOptions {
    /// Primitive pairs with exp(-mu |AB|^2) below this are dropped.
    pub prim_cut: f64,
    /// The highest derivative order of the expansions kept: E (zero), E and E^A (one), E, E^A
    /// and E^AA (two).
    pub deriv_order: DerivOrder,
}

impl Default for PairOptions {
    fn default() -> Self {
        PairOptions {
            prim_cut: PRIM_PAIR_CUT,
            deriv_order: DerivOrder::Zero,
        }
    }
}

impl PairOptions {
    /// Index-ordered pairs with expansions up to derivative order `order`.
    pub fn with_order(order: DerivOrder) -> Self {
        PairOptions {
            deriv_order: order,
            ..Default::default()
        }
    }
}

/// What a [`PairSet`]'s pairs are made of.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PairKind {
    /// The pairs (i, j), i >= j, of the shells of an orbital basis ([`PairSet::build`]).
    Orbital,
    /// The one-center pairs (P, 1) of an auxiliary basis ([`PairSet::aux`]).
    Aux,
}

/// The shell pairs of a basis with their Hermite expansions and bounds (see the module docs).
pub struct PairSet {
    opts: PairOptions,
    kind: PairKind,
    nshell: usize,
    pairs: Vec<ShellPair>,
    derivs: Vec<PairDeriv>,
    schwarz: OnceLock<(Vec<f64>, f64)>,
    deriv_bounds: OnceLock<(Vec<f64>, Vec<f64>)>,
    prim_schwarz: OnceLock<(Vec<f64>, Vec<usize>)>,
}

impl PairSet {
    /// Every pair of shells of `basis`.
    pub fn build(basis: &BasisSet, opts: PairOptions) -> Self {
        let sh = &basis.shells;
        let n = sh.len();
        let views: Vec<ShellView> = sh.iter().map(ShellView::from_shell).collect();
        let order = opts.deriv_order.get();
        let built: Vec<(ShellPair, Option<PairDeriv>)> = (0..n * (n + 1) / 2)
            .into_par_iter()
            .map(|k| {
                let (i, j) = tri_unindex(k);
                pair_data::build(&views[i], &views[j], i, j, order, opts.prim_cut)
            })
            .collect();
        Self::from_built(opts, PairKind::Orbital, n, built)
    }

    /// The pairs (P, 1) of the shells P of an auxiliary basis, in shell order (the second
    /// shell index of each is `usize::MAX`).
    pub fn aux(aux: &BasisSet, opts: PairOptions) -> Self {
        let order = opts.deriv_order.get();
        let built: Vec<(ShellPair, Option<PairDeriv>)> = aux
            .shells
            .par_iter()
            .enumerate()
            .map(|(i, s)| {
                let v = ShellView::from_shell(s);
                let u = ShellView::unit(s.center);
                pair_data::build(&v, &u, i, usize::MAX, order, opts.prim_cut)
            })
            .collect();
        Self::from_built(opts, PairKind::Aux, aux.shells.len(), built)
    }

    fn from_built(
        opts: PairOptions,
        kind: PairKind,
        nshell: usize,
        built: Vec<(ShellPair, Option<PairDeriv>)>,
    ) -> Self {
        let (pairs, derivs): (Vec<ShellPair>, Vec<Option<PairDeriv>>) = built.into_iter().unzip();
        PairSet {
            opts,
            kind,
            nshell,
            pairs,
            derivs: derivs.into_iter().flatten().collect(),
            schwarz: OnceLock::new(),
            deriv_bounds: OnceLock::new(),
            prim_schwarz: OnceLock::new(),
        }
    }

    /// The options the set was built with.
    pub fn options(&self) -> &PairOptions {
        &self.opts
    }

    /// The derivative order of the expansions kept.
    pub fn deriv_order(&self) -> DerivOrder {
        self.opts.deriv_order
    }

    /// Whether these are the (P, 1) pairs of an auxiliary basis.
    pub fn is_aux(&self) -> bool {
        self.kind == PairKind::Aux
    }

    /// What the pairs are made of.
    pub fn kind(&self) -> PairKind {
        self.kind
    }

    /// The primitive-pair cut the set was built with ([`PairOptions::prim_cut`]).
    pub fn prim_cut(&self) -> f64 {
        self.opts.prim_cut
    }

    /// Whether this is an index-ordered orbital set of as many shells as `basis` has, at the
    /// default primitive cut, of any derivative order: a set of `basis` built so has the pairs,
    /// order and Schwarz factors of `PairSet::build(basis, PairOptions::default())` (a pure
    /// function of the basis and the cut; the facade's test `pairs_identical_across_orders`),
    /// so a caller's set can stand in for one built here. A set of another basis with as many
    /// shells is the caller's error.
    pub fn is_default_set_of(&self, basis: &BasisSet) -> bool {
        let ns = basis.shells.len();
        self.kind == PairKind::Orbital
            && self.nshell == ns
            && self.pairs.len() == ns * (ns + 1) / 2
            && self.opts.prim_cut.to_bits() == PairOptions::default().prim_cut.to_bits()
    }

    /// The number of shells of the basis.
    pub fn nshell(&self) -> usize {
        self.nshell
    }

    /// The number of pairs: nshell (nshell + 1) / 2, or nshell for an auxiliary set.
    pub fn len(&self) -> usize {
        self.pairs.len()
    }

    pub fn is_empty(&self) -> bool {
        self.pairs.is_empty()
    }

    /// The position of the pair of shells i and j (either order) in an orbital set.
    #[inline]
    pub fn index(i: usize, j: usize) -> usize {
        if i >= j {
            i * (i + 1) / 2 + j
        } else {
            j * (j + 1) / 2 + i
        }
    }

    /// The shell indices (i, j), i >= j, of pair `k` of an orbital set.
    #[inline]
    pub fn shells(k: usize) -> (usize, usize) {
        tri_unindex(k)
    }

    /// Pair `k`.
    #[inline]
    pub fn pair(&self, k: usize) -> &ShellPair {
        &self.pairs[k]
    }

    /// The pair of shells i and j (either order) of an orbital set.
    #[inline]
    pub fn get(&self, i: usize, j: usize) -> &ShellPair {
        debug_assert_eq!(self.kind, PairKind::Orbital);
        &self.pairs[Self::index(i, j)]
    }

    /// Every pair, in pair order.
    pub fn pairs(&self) -> &[ShellPair] {
        &self.pairs
    }

    /// The derivative expansions of pair `k` (sets of derivative order one or two).
    #[inline]
    pub fn deriv(&self, k: usize) -> &PairDeriv {
        &self.derivs[k]
    }

    /// The derivative expansions of every pair, in pair order (empty at order zero).
    pub fn derivs(&self) -> &[PairDeriv] {
        &self.derivs
    }

    /// Free the derivative expansions and their bounds (the set becomes one of derivative
    /// order zero), e.g. when a calculation still needs E but no longer E^A and E^AA.
    pub fn drop_derivs(&mut self) {
        self.derivs = Vec::new();
        self.deriv_bounds = OnceLock::new();
        self.opts.deriv_order = DerivOrder::Zero;
    }

    /// Schwarz factors Q_k = max_ab |(ab|ab)|^1/2 (Coulomb operator) of every pair, computed
    /// on first use.
    pub fn schwarz(&self) -> &[f64] {
        &self.schwarz_data().0
    }

    /// The largest Schwarz factor.
    pub fn schwarz_max(&self) -> f64 {
        self.schwarz_data().1
    }

    fn schwarz_data(&self) -> &(Vec<f64>, f64) {
        self.schwarz.get_or_init(|| {
            let q: Vec<f64> = self
                .pairs
                .par_iter()
                .map_init(
                    || (EriWork::new(), Vec::new()),
                    |(w, buf), pr| diag_bound(pr, w, buf),
                )
                .collect();
            let qmax = q.iter().cloned().fold(0.0, f64::max);
            (q, qmax)
        })
    }

    /// Bounds of the derivative integrals of every pair, computed on first use (sets of
    /// derivative order one or two): Q1 = max |(∂X_k ab|∂X_k ab)|^1/2 over X in {A, B}, and Q2
    /// the same for the second derivatives (zero at order one). Then
    /// |∂(ab|cd)| <= Q1_ab Q_cd + Q_ab Q1_cd and
    /// |∂²(ab|cd)| <= Q2_ab Q_cd + 2 Q1_ab Q1_cd + Q_ab Q2_cd.
    pub fn deriv_bounds(&self) -> (&[f64], &[f64]) {
        let (q1, q2) = self
            .deriv_bounds
            .get_or_init(|| schwarz_deriv(&self.pairs, &self.derivs));
        (q1, q2)
    }

    /// Primitive Schwarz factors max_ab |(ab|ab)|^1/2 of each retained primitive pair of pair
    /// `k`, computed for every pair on first use.
    pub fn prim_schwarz(&self, k: usize) -> &[f64] {
        let (q, off) = self.prim_schwarz.get_or_init(|| {
            let per: Vec<Vec<f64>> = self
                .pairs
                .par_iter()
                .map_init(
                    || (EriWork::new(), Vec::new()),
                    |(w, buf), pr| {
                        let sz = pr.e.len() / pr.nprim.max(1);
                        (0..pr.nprim)
                            .map(|ip| {
                                let one = ShellPair {
                                    sa: pr.sa,
                                    sb: pr.sb,
                                    la: pr.la,
                                    lb: pr.lb,
                                    na: pr.na,
                                    nb: pr.nb,
                                    nprim: 1,
                                    p: vec![pr.p[ip]],
                                    pc: vec![pr.pc[ip]],
                                    e: pr.e[ip * sz..(ip + 1) * sz].to_vec(),
                                    emax: pr.emax,
                                };
                                diag_bound(&one, w, buf)
                            })
                            .collect()
                    },
                )
                .collect();
            let mut off = Vec::with_capacity(per.len() + 1);
            off.push(0);
            for v in &per {
                off.push(off.last().unwrap() + v.len());
            }
            (per.concat(), off)
        });
        &q[off[k]..off[k + 1]]
    }
}

/// max_ab |(ab|ab)|^1/2 of a pair (Coulomb operator), on a worker's integral scratch and
/// buffer (grown to the largest pair, kept between pairs).
pub(crate) fn diag_bound(pr: &ShellPair, w: &mut EriWork, buf: &mut Vec<f64>) -> f64 {
    let n = pr.nab();
    if buf.len() < n * n {
        buf.resize(n * n, 0.0);
    }
    eri_quartet(pr, pr, &Operator::Coulomb, buf, w);
    (0..n)
        .map(|x| buf[x * n + x].abs())
        .fold(0.0, f64::max)
        .sqrt()
}

/// Inverse of k = i(i+1)/2 + j, j <= i.
#[inline]
pub(crate) fn tri_unindex(k: usize) -> (usize, usize) {
    let mut i = (((8 * k + 1) as f64).sqrt() as usize).saturating_sub(1) / 2;
    while (i + 1) * (i + 2) / 2 <= k {
        i += 1;
    }
    while i * (i + 1) / 2 > k {
        i -= 1;
    }
    (i, k - i * (i + 1) / 2)
}
