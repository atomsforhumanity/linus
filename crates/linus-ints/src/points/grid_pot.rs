//! Potential integrals of shell pairs at batches of points (the analytic half of
//! seminumerical exchange):
//!
//!   A_ab(g) = <a| O(|r - r_g|) |b>,   O(r) = full / r + erf erf(omega r) / r  ([`Operator`]).
//!
//! Rys quadrature (Dupuis, Rys & King 1976): for a primitive pair with exponent sum p, centre
//! P and prefactor K_ab, <a|1/|r-C||b> = (2 pi / p) K_ab sum_i w_i Ix_i Iy_i Iz_i over the
//! n = floor((la+lb)/2)+1 roots x_i = t_i^2 of T = p |PC|^2, with 1D integrals from
//!   I(i+1, 0) = (PA - x PC) I(i, 0) + i (1 - x) / (2p) I(i-1, 0)       (vertical)
//!   I(i, j+1) = I(i+1, j) + AB I(i, j)                                 (horizontal).
//! The erf-attenuated operator uses the roots of T kappa^2 (kappa^2 = omega^2/(omega^2+p)),
//! nodes scaled by kappa^2 and weights by kappa; a mixed operator is one quadrature with both
//! root sets. Points are processed eight at a time (one AVX-512 register per quantity); the
//! kernels are monomorphized per (la, lb), written with explicit lane vectors ([`Lanes`]; plain
//! `[f64; 8]` loops with `mul_add` compiled to scalar FMAs in places) and compiled for AVX-512,
//! AVX2+FMA and baseline x86-64 with runtime dispatch; classes with la + lb > 4 sit in separate
//! functions, because the stack frame of the largest inlined kernel was probed page by page on
//! every call. With [`PotContract::cart`] F and G rows are the Cartesian components of the
//! shells and no spherical transform runs per pair and chunk. Shell pairs with la + lb > 8 fall
//! back to McMurchie-Davidson ([`crate::one_e::potential_pair_points`], Coulomb operator only).
//!
//! Cost per primitive pair and point (ts61 def2-TZVP, GridX3, one thread of 8): (0,0) 1.8 ns,
//! (1,0) 2.6, (1,1) 5.0, (2,0) 7.0, (2,1) 9.5, (3,0) 8.5, (3,1) 18, (2,2) 23, (3,2) 34,
//! (3,3) 71 (before: 2.6, 4.2, 6.5, 11.7, 15.2, 18.6, 31, 45, 58, 101). For the low classes
//! the per-lane evaluation of the fitted Rys roots and weights is a large part (gathers of the
//! coefficient rows are slower: 1.7 ns per 8-lane gather on Zen 5 vs 1.0 ns for 8 loads).

use crate::basis::{ncart, BasisSet, Shell};
use crate::core::hermite::{build_pair, ShellView};
use crate::core::rys::roots_vec;
use crate::operator::Operator;
use linus_num::simd::Isa;
use linus_num::simd::{Lanes, Portable};
use std::f64::consts::PI;

/// One primitive pair of a [`PotPair`] (bra = the shell with the larger l).
#[derive(Clone, Copy, Debug)]
pub struct PotPrim {
    pub p: f64,
    pub pc: [f64; 3],
    /// P - A (bra centre)
    pub pa: [f64; 3],
    pub half_inv_p: f64,
    /// Exponent of the bra primitive (the ket's is p - a1); used by the derivative kernels.
    pub a1: f64,
    /// (2 pi / p) K_ab c_a c_b
    pub coef: f64,
    /// Magnitude estimate of the primitive's integrals (for screening).
    pub est: f64,
}

/// Precomputed data of one shell pair for potential integrals on points.
#[derive(Clone, Debug)]
pub struct PotPair {
    /// Shell indices and properties in the caller's orientation (sa >= sb: ShellIndex).
    pub sa: usize,
    pub sb: usize,
    pub la: usize,
    pub lb: usize,
    pub na: usize,
    pub nb: usize,
    /// Computed as <sb|O|sa> (larger l in the bra) and transposed on output.
    pub(crate) swap: bool,
    pub(crate) l1: usize,
    pub(crate) l2: usize,
    pub(crate) n1: usize,
    pub(crate) n2: usize,
    /// bra centre - ket centre
    pub(crate) ab: [f64; 3],
    /// Primitive pairs sorted by decreasing `est`.
    pub prims: Vec<PotPrim>,
    /// Spherical transforms as nonzero (function, Cartesian component, coefficient) lists,
    /// grouped by function; None for identity (s, p, Cartesian shells).
    pub(crate) t1: Option<Vec<(usize, usize, f64)>>,
    pub(crate) t2: Option<Vec<(usize, usize, f64)>>,
    /// Sum of the primitive estimates.
    pub est: f64,
    /// McMurchie-Davidson data for pairs beyond the Rys kernels.
    md: Option<crate::core::hermite::ShellPair>,
}

/// Largest l handled by the Rys kernels (and la + lb <= 2 * RYS_LMAX).
pub const RYS_LMAX: usize = 4;

fn is_identity(t: &[f64], n: usize, nc: usize) -> bool {
    n == nc && (0..n).all(|i| (0..nc).all(|j| t[i * nc + j] == if i == j { 1.0 } else { 0.0 }))
}

fn sparse_transform(t: &[f64], n: usize, nc: usize) -> Option<Vec<(usize, usize, f64)>> {
    if is_identity(t, n, nc) {
        return None;
    }
    Some(
        (0..n)
            .flat_map(|j| {
                (0..nc)
                    .filter(move |&i| t[j * nc + i] != 0.0)
                    .map(move |i| (j, i, t[j * nc + i]))
            })
            .collect(),
    )
}

impl PotPair {
    /// Pair data for shells `sa`, `sb` of `basis`; primitive pairs with
    /// exp(-mu R^2) < `prim_cut` are dropped.
    pub fn new(basis: &BasisSet, sa: usize, sb: usize, prim_cut: f64) -> Self {
        let (a, b) = (&basis.shells[sa], &basis.shells[sb]);
        let swap = a.l < b.l;
        let (s1, s2): (&Shell, &Shell) = if swap { (b, a) } else { (a, b) };
        let (l1, l2) = (s1.l, s2.l);
        let ab = [
            s1.center[0] - s2.center[0],
            s1.center[1] - s2.center[1],
            s1.center[2] - s2.center[2],
        ];
        let r2 = ab[0] * ab[0] + ab[1] * ab[1] + ab[2] * ab[2];
        let mut prims = Vec::new();
        for (&e1, &c1) in s1.exps.iter().zip(&s1.coefs) {
            for (&e2, &c2) in s2.exps.iter().zip(&s2.coefs) {
                let p = e1 + e2;
                let k = (-e1 * e2 / p * r2).exp();
                if k < prim_cut {
                    continue;
                }
                let pc = [
                    (e1 * s1.center[0] + e2 * s2.center[0]) / p,
                    (e1 * s1.center[1] + e2 * s2.center[1]) / p,
                    (e1 * s1.center[2] + e2 * s2.center[2]) / p,
                ];
                let pa = [
                    pc[0] - s1.center[0],
                    pc[1] - s1.center[1],
                    pc[2] - s1.center[2],
                ];
                let pb = [
                    pc[0] - s2.center[0],
                    pc[1] - s2.center[1],
                    pc[2] - s2.center[2],
                ];
                let coef = 2.0 * PI / p * k * c1 * c2;
                let rp = (0.5 / p).sqrt();
                let na = (pa[0] * pa[0] + pa[1] * pa[1] + pa[2] * pa[2]).sqrt() + rp;
                let nb = (pb[0] * pb[0] + pb[1] * pb[1] + pb[2] * pb[2]).sqrt() + rp;
                let est = coef.abs() * na.powi(l1 as i32) * nb.powi(l2 as i32);
                prims.push(PotPrim {
                    p,
                    pc,
                    pa,
                    half_inv_p: 0.5 / p,
                    a1: e1,
                    coef,
                    est,
                });
            }
        }
        prims.sort_by(|x, y| y.est.partial_cmp(&x.est).unwrap());
        let est = prims.iter().map(|q| q.est).sum();
        let (t1, t2) = (BasisSet::shell_transform(s1), BasisSet::shell_transform(s2));
        let (n1, n2) = (s1.nfunc(), s2.nfunc());
        let t1 = sparse_transform(&t1, n1, ncart(l1));
        let t2 = sparse_transform(&t2, n2, ncart(l2));
        let md = if l1 > RYS_LMAX || l1 + l2 > 2 * RYS_LMAX {
            Some(build_pair(
                &ShellView::from_shell(a),
                &ShellView::from_shell(b),
                sa,
                sb,
            ))
        } else {
            None
        };
        PotPair {
            sa,
            sb,
            la: a.l,
            lb: b.l,
            na: a.nfunc(),
            nb: b.nfunc(),
            swap,
            l1,
            l2,
            n1,
            n2,
            ab,
            prims,
            t1,
            t2,
            est,
            md,
        }
    }

    pub fn nab(&self) -> usize {
        self.na * self.nb
    }

    /// Number of primitive pairs with `est >= cut`.
    pub fn nprim_above(&self, cut: f64) -> usize {
        self.prims.partition_point(|q| q.est >= cut)
    }

    /// out[(ia * nb + ib) * np + g] = <a| O(|r - r_g|) |b> for a in shell `sa`, b in shell `sb`
    /// and the np points (x, y, z). Primitive pairs with `est < cut` are skipped (cut = 0 keeps
    /// all).
    pub fn compute(
        &self,
        op: &Operator,
        x: &[f64],
        y: &[f64],
        z: &[f64],
        cut: f64,
        out: &mut [f64],
    ) {
        let np = x.len();
        assert!(y.len() == np && z.len() == np && out.len() >= self.nab() * np);
        if self.md.is_some() {
            self.compute_md(op, x, y, z, out);
            return;
        }
        self.dispatch(op, x, y, z, cut, &mut Sink::Store(out));
    }

    /// The integrals contracted with F on the fly (seminumerical exchange), without storing
    /// them: for every density k and point g,
    ///   G\[k\]\[a\]\[g\] += sum_b A_ab(g) F\[k\]\[b\]\[g\]   (if `c.do_a`),
    ///   G\[k\]\[b\]\[g\] += sum_a A_ab(g) F\[k\]\[a\]\[g\]   (if `c.do_b`),
    /// with a, b the functions of shells sa, sb (rows `c.a0 + ia`, `c.b0 + ib`). F and G rows
    /// have stride `c.ld`, a multiple of 8 with zero padding in F beyond the np points.
    pub fn contract(
        &self,
        op: &Operator,
        x: &[f64],
        y: &[f64],
        z: &[f64],
        cut: f64,
        c: &mut PotContract,
    ) {
        let np = x.len();
        assert!(y.len() == np && z.len() == np && c.ld.is_multiple_of(LN) && c.ld >= np);
        assert!(c.f.len() >= c.nd * c.dstride && c.g.len() >= c.nd * c.dstride);
        if self.md.is_some() {
            let mut a = vec![0.0; self.nab() * np];
            self.compute_md(op, x, y, z, &mut a);
            for k in 0..c.nd {
                let base = k * c.dstride;
                for ia in 0..self.na {
                    for ib in 0..self.nb {
                        let row = &a[(ia * self.nb + ib) * np..(ia * self.nb + ib + 1) * np];
                        for (g, &v) in row.iter().enumerate() {
                            if c.do_a {
                                c.g[base + (c.a0 + ia) * c.ld + g] +=
                                    v * c.f[base + (c.b0 + ib) * c.ld + g];
                            }
                            if c.do_b {
                                c.g[base + (c.b0 + ib) * c.ld + g] +=
                                    v * c.f[base + (c.a0 + ia) * c.ld + g];
                            }
                        }
                    }
                }
            }
            return;
        }
        self.dispatch(op, x, y, z, cut, &mut Sink::Contract(c));
    }

    fn compute_md(&self, op: &Operator, x: &[f64], y: &[f64], z: &[f64], out: &mut [f64]) {
        let md = self.md.as_ref().unwrap();
        assert!(
            !op.has_erf(),
            "erf-attenuated potential integrals need la, lb <= {RYS_LMAX}"
        );
        let (mut r, mut w) = (Vec::new(), Vec::new());
        crate::one_e::potential_pair_points(md, x, y, z, out, &mut r, &mut w);
        if op.coulomb() != 1.0 {
            out[..self.nab() * x.len()]
                .iter_mut()
                .for_each(|v| *v *= op.coulomb());
        }
    }

    fn dispatch(&self, op: &Operator, x: &[f64], y: &[f64], z: &[f64], cut: f64, sink: &mut Sink) {
        let nprim = self.nprim_above(cut);
        self.dispatch_prims(&self.prims[..nprim], op, x, y, z, sink);
    }

    /// Primitive pairs of this pair that matter for points within `radius` of `center`:
    /// beyond the pair's extent the integrals of a primitive pair are bounded by
    /// est * F_0(p R^2) <= est * sqrt(pi) / (2 sqrt(p) R), R the distance from its centre P to
    /// the sphere. Appends the survivors to `out` and returns the sum of their scaled estimates.
    pub fn screen_prims(
        &self,
        center: [f64; 3],
        radius: f64,
        cut: f64,
        out: &mut Vec<PotPrim>,
    ) -> f64 {
        let mut sum = 0.0;
        for q in &self.prims[..self.nprim_above(cut)] {
            let d = ((q.pc[0] - center[0]).powi(2)
                + (q.pc[1] - center[1]).powi(2)
                + (q.pc[2] - center[2]).powi(2))
            .sqrt();
            let r = d - radius;
            let f = if r > 0.0 {
                (0.886_226_925_452_758 / (q.p.sqrt() * r)).min(1.0)
            } else {
                1.0
            };
            let e = q.est * f;
            if e >= cut {
                out.push(*q);
                sum += e;
            }
        }
        sum
    }

    /// [`PotPair::contract`] over a given list of this pair's primitive pairs (e.g. from
    /// [`PotPair::screen_prims`]).
    pub fn contract_prims(
        &self,
        prims: &[PotPrim],
        op: &Operator,
        x: &[f64],
        y: &[f64],
        z: &[f64],
        c: &mut PotContract,
    ) {
        assert!(
            self.md.is_none(),
            "contract_prims needs a Rys pair (la, lb <= {RYS_LMAX})"
        );
        let np = x.len();
        assert!(y.len() == np && z.len() == np && c.ld.is_multiple_of(LN) && c.ld >= np);
        assert!(c.f.len() >= c.nd * c.dstride && c.g.len() >= c.nd * c.dstride);
        self.dispatch_prims(prims, op, x, y, z, &mut Sink::Contract(c));
    }

    /// True if the pair is evaluated by McMurchie-Davidson (l beyond the Rys kernels).
    pub fn is_md(&self) -> bool {
        self.md.is_some()
    }

    fn dispatch_prims(
        &self,
        prims: &[PotPrim],
        op: &Operator,
        x: &[f64],
        y: &[f64],
        z: &[f64],
        sink: &mut Sink,
    ) {
        // classes with l1 + l2 > 4 in their own functions: the stack frame of the largest
        // inlined kernel is probed page by page on every call
        let big = self.l1 + self.l2 > 4;
        match Isa::detect() {
            #[cfg(target_arch = "x86_64")]
            Isa::Avx512 => unsafe {
                if big {
                    compute_big_avx512(self, prims, op, x, y, z, sink)
                } else {
                    compute_avx512(self, prims, op, x, y, z, sink)
                }
            },
            #[cfg(target_arch = "x86_64")]
            Isa::Avx2 => unsafe {
                if big {
                    compute_big_avx2(self, prims, op, x, y, z, sink)
                } else {
                    compute_avx2(self, prims, op, x, y, z, sink)
                }
            },
            _ => {
                if big {
                    compute_big::<Portable<false>>(self, prims, op, x, y, z, sink)
                } else {
                    compute_small::<Portable<false>>(self, prims, op, x, y, z, sink)
                }
            }
        }
    }
}

/// F and G of [`PotPair::contract`]: nd densities, each a block of `dstride` values holding
/// rows (functions) of stride `ld`.
pub struct PotContract<'a> {
    pub f: &'a [f64],
    pub g: &'a mut [f64],
    pub nd: usize,
    pub dstride: usize,
    pub ld: usize,
    /// First row of shells sa and sb.
    pub a0: usize,
    pub b0: usize,
    pub do_a: bool,
    pub do_b: bool,
    /// Rows are the Cartesian components of the shells (a0, b0 Cartesian offsets): F and G
    /// are used without spherical transforms.
    pub cart: bool,
}

enum Sink<'s, 'a> {
    Store(&'s mut [f64]),
    Contract(&'s mut PotContract<'a>),
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512vl,avx512dq,fma")]
unsafe fn compute_avx512(
    pair: &PotPair,
    prims: &[PotPrim],
    op: &Operator,
    x: &[f64],
    y: &[f64],
    z: &[f64],
    sink: &mut Sink,
) {
    compute_small::<linus_num::simd::Avx512>(pair, prims, op, x, y, z, sink)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512vl,avx512dq,fma")]
unsafe fn compute_big_avx512(
    pair: &PotPair,
    prims: &[PotPrim],
    op: &Operator,
    x: &[f64],
    y: &[f64],
    z: &[f64],
    sink: &mut Sink,
) {
    compute_big::<linus_num::simd::Avx512>(pair, prims, op, x, y, z, sink)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn compute_avx2(
    pair: &PotPair,
    prims: &[PotPrim],
    op: &Operator,
    x: &[f64],
    y: &[f64],
    z: &[f64],
    sink: &mut Sink,
) {
    compute_small::<linus_num::simd::Avx2>(pair, prims, op, x, y, z, sink)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn compute_big_avx2(
    pair: &PotPair,
    prims: &[PotPrim],
    op: &Operator,
    x: &[f64],
    y: &[f64],
    z: &[f64],
    sink: &mut Sink,
) {
    compute_big::<linus_num::simd::Avx2>(pair, prims, op, x, y, z, sink)
}

/// Lanes per chunk of points.
pub(crate) const LN: usize = 8;
pub(crate) type V = [f64; LN];

/// Cartesian exponents of component i of l (lx descending, then ly descending), l <= 4.
pub(crate) const CART: [[[usize; 3]; 15]; 5] = {
    let mut t = [[[0usize; 3]; 15]; 5];
    let mut l = 0;
    while l <= 4 {
        let mut i = 0;
        let mut lx = l as i64;
        while lx >= 0 {
            let mut ly = l as i64 - lx;
            while ly >= 0 {
                t[l][i] = [lx as usize, ly as usize, (l as i64 - lx - ly) as usize];
                i += 1;
                ly -= 1;
            }
            lx -= 1;
        }
        l += 1;
    }
    t
};

// kernel::<L1, L2, NC1, NC2, NR, W, GA = L1 + L2 + 1, GB = L2 + 1, S>
macro_rules! kern {
    ($l1:literal, $l2:literal, $nr:literal, $w:literal, $ga:literal, $gb:literal, $s:ty, $pair:expr, $prims:expr, $op:expr, $x:expr, $y:expr, $z:expr, $sink:expr) => {
        kernel::<$l1, $l2, { ncart($l1) }, { ncart($l2) }, $nr, $w, $ga, $gb, $s>(
            $pair, $prims, $op, $x, $y, $z, $sink,
        )
    };
}

#[inline(always)]
fn compute_small<S: Lanes>(
    pair: &PotPair,
    prims: &[PotPrim],
    op: &Operator,
    x: &[f64],
    y: &[f64],
    z: &[f64],
    sink: &mut Sink,
) {
    match (pair.l1, pair.l2) {
        (0, 0) => kern!(0, 0, 1, 2, 1, 1, S, pair, prims, op, x, y, z, sink),
        (1, 0) => kern!(1, 0, 1, 2, 2, 1, S, pair, prims, op, x, y, z, sink),
        (1, 1) => kern!(1, 1, 2, 4, 3, 2, S, pair, prims, op, x, y, z, sink),
        (2, 0) => kern!(2, 0, 2, 4, 3, 1, S, pair, prims, op, x, y, z, sink),
        (2, 1) => kern!(2, 1, 2, 4, 4, 2, S, pair, prims, op, x, y, z, sink),
        (2, 2) => kern!(2, 2, 3, 8, 5, 3, S, pair, prims, op, x, y, z, sink),
        (3, 0) => kern!(3, 0, 2, 4, 4, 1, S, pair, prims, op, x, y, z, sink),
        (3, 1) => kern!(3, 1, 3, 8, 5, 2, S, pair, prims, op, x, y, z, sink),
        (4, 0) => kern!(4, 0, 3, 8, 5, 1, S, pair, prims, op, x, y, z, sink),
        _ => unreachable!(),
    }
}

#[inline(always)]
fn compute_big<S: Lanes>(
    pair: &PotPair,
    prims: &[PotPrim],
    op: &Operator,
    x: &[f64],
    y: &[f64],
    z: &[f64],
    sink: &mut Sink,
) {
    match (pair.l1, pair.l2) {
        (3, 2) => kern!(3, 2, 3, 8, 6, 3, S, pair, prims, op, x, y, z, sink),
        (3, 3) => kern!(3, 3, 4, 8, 7, 4, S, pair, prims, op, x, y, z, sink),
        (4, 1) => kern!(4, 1, 3, 8, 6, 2, S, pair, prims, op, x, y, z, sink),
        (4, 2) => kern!(4, 2, 4, 8, 7, 3, S, pair, prims, op, x, y, z, sink),
        (4, 3) => kern!(4, 3, 4, 8, 8, 4, S, pair, prims, op, x, y, z, sink),
        (4, 4) => kern!(4, 4, 5, 12, 9, 5, S, pair, prims, op, x, y, z, sink),
        _ => unreachable!(),
    }
}

/// 1D integrals of one Cartesian direction for one root: g[a][b], a + b <= L1 + L2, b <= L2,
/// with the starting value `g00` (the quadrature weight for x, 1 for y and z).
#[inline(always)]
fn one_d<const L1: usize, const L2: usize, const GA: usize, const GB: usize, S: Lanes>(
    g: &mut [[V; GB]; GA],
    g00: S,
    c00: S,
    b00: S,
    ab: f64,
) {
    let lt = L1 + L2;
    g00.store(&mut g[0][0]);
    if lt == 0 {
        return;
    }
    let mut prev = g00;
    let mut cur = c00 * g00;
    cur.store(&mut g[1][0]);
    for k in 1..lt {
        let next = c00.mul_add(cur, S::splat(k as f64) * b00 * prev);
        next.store(&mut g[k + 1][0]);
        prev = cur;
        cur = next;
    }
    let abv = S::splat(ab);
    for b in 1..=L2 {
        for a in 0..=(lt - b) {
            abv.mul_add(S::load8(&g[a][b - 1]), S::load8(&g[a + 1][b - 1]))
                .store(&mut g[a][b]);
        }
    }
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn add_root<
    const L1: usize,
    const L2: usize,
    const NC1: usize,
    const NC2: usize,
    const GA: usize,
    const GB: usize,
    const FIRST: bool,
    S: Lanes,
>(
    acc: &mut [[V; NC2]; NC1],
    u: &V,
    w: &V,
    q: &PotPrim,
    pc: &[S; 3],
    ab: &[f64; 3],
    g: &mut [[[V; GB]; GA]; 3],
) {
    let uv = S::load8(u);
    let b00 = (S::splat(1.0) - uv) * S::splat(q.half_inv_p);
    for d in 0..3 {
        let c00 = (-uv).mul_add(pc[d], S::splat(q.pa[d]));
        one_d::<L1, L2, GA, GB, S>(
            &mut g[d],
            if d == 0 { S::load8(w) } else { S::splat(1.0) },
            c00,
            b00,
            ab[d],
        );
    }
    let c1 = &CART[L1];
    let c2 = &CART[L2];
    for i1 in 0..NC1 {
        let [ax, ay, az] = c1[i1];
        for i2 in 0..NC2 {
            let [bx, by, bz] = c2[i2];
            let xy = S::load8(&g[0][ax][bx]) * S::load8(&g[1][ay][by]);
            let a = &mut acc[i1][i2];
            // (the first root of a chunk stores: no clearing of acc)
            if FIRST {
                (xy * S::load8(&g[2][az][bz])).store(a);
            } else {
                xy.mul_add(S::load8(&g[2][az][bz]), S::load8(a)).store(a);
            }
        }
    }
}

/// The auxiliary integrals [s]^(m), m = 0..=mm (mm <= 3), of a primitive pair for the eight
/// points: coef F_m(T), plus coef c_lr kappa^(2m+1) F_m(kappa^2 T) for the erf part.
#[inline(always)]
pub(crate) fn aux_ss<S: Lanes>(mm: usize, q: &PotPrim, tv: S, op: &Operator, om2: f64) -> [S; 4] {
    let mut sm = [S::zero(); 4];
    let mut f = [S::zero(); 4];
    let cf = q.coef * op.coulomb();
    if cf != 0.0 {
        crate::core::boys::boys_lanes(mm, tv, &mut f);
        for m in 0..=mm {
            sm[m] = f[m] * S::splat(cf);
        }
    }
    if op.has_erf() {
        let k2 = om2 / (om2 + q.p);
        crate::core::boys::boys_lanes(mm, tv * S::splat(k2), &mut f);
        let mut km = k2.sqrt() * q.coef * op.long_range();
        for m in 0..=mm {
            sm[m] = f[m].mul_add(S::splat(km), sm[m]);
            km *= k2;
        }
    }
    sm
}

/// Cartesian component index of d_(i+j) (order of [`CART`]: xx, xy, xz, yy, yz, zz).
pub(crate) const D2: [[usize; 3]; 3] = [[0, 1, 2], [1, 3, 4], [2, 4, 5]];

/// The integrals of one primitive pair of the classes (s|s), (p|s), (p|p) and (d|s) in closed
/// form (Obara-Saika) from [s]^(m) = coef F_m(T), m <= la + lb: (p_i|s) = PA_i [s]^(0) - PC_i
/// [s]^(1), (d_ij|s) = PA_j (p_i|s)^(0) - PC_j (p_i|s)^(1) + delta_ij ([s]^(0) - [s]^(1)) / 2p,
/// (p_i|p_j) = (d_ij|s) + AB_j (p_i|s); stored into acc for the first primitive of a chunk,
/// added after.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::needless_range_loop, reason = "indexes several arrays in step")]
fn closed_ints<const L1: usize, const L2: usize, const NC1: usize, const NC2: usize, S: Lanes>(
    acc: &mut [[V; NC2]; NC1],
    first: bool,
    q: &PotPrim,
    pc: &[S; 3],
    tv: S,
    ab: &[f64; 3],
    op: &Operator,
    om2: f64,
) {
    let sm = aux_ss(L1 + L2, q, tv, op, om2);
    let mut put = |i1: usize, i2: usize, v: S| {
        let a = &mut acc[i1][i2];
        if first {
            v.store(a);
        } else {
            (S::load8(a) + v).store(a);
        }
    };
    if L1 + L2 == 0 {
        put(0, 0, sm[0]);
        return;
    }
    let pa = [S::splat(q.pa[0]), S::splat(q.pa[1]), S::splat(q.pa[2])];
    let p0: [S; 3] = std::array::from_fn(|i| pa[i].mul_add(sm[0], -(pc[i] * sm[1])));
    if L1 + L2 == 1 {
        for i in 0..3 {
            put(i, 0, p0[i]);
        }
        return;
    }
    let p1: [S; 3] = std::array::from_fn(|i| pa[i].mul_add(sm[1], -(pc[i] * sm[2])));
    let h = (sm[0] - sm[1]) * S::splat(q.half_inv_p);
    let d = |i: usize, j: usize| {
        let v = pa[j].mul_add(p0[i], -(pc[j] * p1[i]));
        if i == j {
            v + h
        } else {
            v
        }
    };
    if L1 == 2 {
        for i in 0..3 {
            for j in i..3 {
                put(D2[i][j], 0, d(i, j));
            }
        }
    } else {
        for i in 0..3 {
            for j in 0..3 {
                put(i, j, S::splat(ab[j]).mul_add(p0[i], d(i, j)));
            }
        }
    }
}

/// F rows of a shell (functions r0.., block offset `base`, lanes g0..g0+8 already in `base`)
/// in its Cartesian components (T^T F; plain rows without a transform).
#[inline(always)]
pub(crate) fn cart_rows<const NC: usize, S: Lanes>(
    t: &Option<Vec<(usize, usize, f64)>>,
    f: &[f64],
    base: usize,
    r0: usize,
    ld: usize,
) -> [V; NC] {
    let mut fc = [[0.0; LN]; NC];
    match t {
        Some(t) => {
            for &(j, i, cv) in t {
                let o = base + (r0 + j) * ld;
                let fv: &V = f[o..o + LN].try_into().unwrap();
                let d = &mut fc[i];
                S::splat(cv).mul_add(S::load8(fv), S::load8(d)).store(d);
            }
        }
        None => {
            for (i, fci) in fc.iter_mut().enumerate() {
                let o = base + (r0 + i) * ld;
                *fci = f[o..o + LN].try_into().unwrap();
            }
        }
    }
    fc
}

/// G[rows r0 + j] += T gc (Cartesian G back to the functions; plain adds without a transform).
#[inline(always)]
pub(crate) fn add_g_rows<const NC: usize, S: Lanes>(
    t: &Option<Vec<(usize, usize, f64)>>,
    gc: &[V; NC],
    g: &mut [f64],
    base: usize,
    r0: usize,
    ld: usize,
) {
    match t {
        Some(t) => {
            for &(j, i, cv) in t {
                let o = base + (r0 + j) * ld;
                let gv = &mut g[o..o + LN];
                S::splat(cv)
                    .mul_add(S::load8(&gc[i]), S::load(gv))
                    .store(gv);
            }
        }
        None => {
            for (i, gci) in gc.iter().enumerate() {
                let o = base + (r0 + i) * ld;
                let gv = &mut g[o..o + LN];
                (S::load(gv) + S::load8(gci)).store(gv);
            }
        }
    }
}

/// [`cart_rows`] as lane vectors; without a transform the rows are loaded in place (no copy,
/// no per-row bounds check).
///
/// # Safety
/// Without a transform, `base + (r0 + NC - 1) * ld + LN <= f.len()` (checked by the kernels
/// once per call: [`rows_in_bounds`]).
#[inline(always)]
pub(crate) unsafe fn cart_rows_v<const NC: usize, S: Lanes>(
    t: &Option<Vec<(usize, usize, f64)>>,
    f: &[f64],
    base: usize,
    r0: usize,
    ld: usize,
) -> [S; NC] {
    match t {
        None => {
            let p = f.as_ptr().add(base + r0 * ld);
            std::array::from_fn(|i| S::load8(&*(p.add(i * ld) as *const V)))
        }
        Some(_) => {
            let fc = cart_rows::<NC, S>(t, f, base, r0, ld);
            std::array::from_fn(|i| S::load8(&fc[i]))
        }
    }
}

/// [`add_g_rows`] from lane vectors (in place without a transform).
///
/// # Safety
/// As [`cart_rows_v`] for `g`.
#[inline(always)]
pub(crate) unsafe fn add_g_rows_v<const NC: usize, S: Lanes>(
    t: &Option<Vec<(usize, usize, f64)>>,
    gc: &[S; NC],
    g: &mut [f64],
    base: usize,
    r0: usize,
    ld: usize,
) {
    match t {
        None => {
            let p = g.as_mut_ptr().add(base + r0 * ld);
            for (i, v) in gc.iter().enumerate() {
                let q = &mut *(p.add(i * ld) as *mut V);
                (S::load8(q) + *v).store(q);
            }
        }
        Some(_) => {
            let a: [V; NC] = std::array::from_fn(|i| gc[i].to_array());
            add_g_rows::<NC, S>(t, &a, g, base, r0, ld);
        }
    }
}

/// Whether the untransformed rows r0..r0 + nc of every block (`dstride` values each, rows of
/// `ld`) lie inside blocks of rows: the precondition of [`cart_rows_v`] and [`add_g_rows_v`].
#[inline]
pub(crate) fn rows_in_bounds(
    t: &Option<Vec<(usize, usize, f64)>>,
    r0: usize,
    nc: usize,
    dstride: usize,
    ld: usize,
) -> bool {
    t.is_some() || (r0 + nc) * ld <= dstride
}

/// The contraction of one chunk's Cartesian integrals with F into G (see
/// [`PotPair::contract`]): dir1 G[bra] += A F[ket], dir2 G[ket] += A^T F[bra].
#[inline(always)]
fn contract_chunk<const NC1: usize, const NC2: usize, S: Lanes>(
    pair: &PotPair,
    acc: &[[V; NC2]; NC1],
    c: &mut PotContract,
    g0: usize,
) {
    let (r1, r2) = if pair.swap {
        (c.b0, c.a0)
    } else {
        (c.a0, c.b0)
    };
    let (d1, d2) = if pair.swap {
        (c.do_b, c.do_a)
    } else {
        (c.do_a, c.do_b)
    };
    let (t1, t2) = if c.cart {
        (&None, &None)
    } else {
        (&pair.t1, &pair.t2)
    };
    let ld = c.ld;
    // (bounds of the untransformed rows: checked once per pair in `kernel`)
    for k in 0..c.nd {
        let base = k * c.dstride + g0;
        if d1 {
            // SAFETY: rows_in_bounds(t2, r2, NC2) and k < nd, g0 + LN <= ld (asserted in `kernel`)
            let fc = unsafe { cart_rows_v::<NC2, S>(t2, c.f, base, r2, ld) };
            let gc: [S; NC1] = std::array::from_fn(|i1| {
                let mut v = S::zero();
                for i2 in 0..NC2 {
                    v = S::load8(&acc[i1][i2]).mul_add(fc[i2], v);
                }
                v
            });
            // SAFETY: as above for t1, r1
            unsafe { add_g_rows_v::<NC1, S>(t1, &gc, c.g, base, r1, ld) };
        }
        if d2 {
            // SAFETY: as above
            let fc = unsafe { cart_rows_v::<NC1, S>(t1, c.f, base, r1, ld) };
            let gc: [S; NC2] = std::array::from_fn(|i2| {
                let mut v = S::zero();
                for i1 in 0..NC1 {
                    v = S::load8(&acc[i1][i2]).mul_add(fc[i1], v);
                }
                v
            });
            unsafe { add_g_rows_v::<NC2, S>(t2, &gc, c.g, base, r2, ld) };
        }
    }
}

/// Spherical transform and store of one chunk (the `compute` path: tests, benchmarks, C-PCM
/// surface integrals). Not inlined (its two buffers would enlarge every kernel's stack frame)
/// and so compiled without the kernels' target features: plain multiply-adds, as `f64::mul_add`
/// would be a call into libm's software fma here (u1an39 HF/def2-SVP C-PCM SCF: 9.5% of all
/// samples in `fma` from this function).
#[inline(never)]
#[allow(clippy::needless_range_loop, reason = "indexes several arrays in step")]
fn store_chunk<const NC1: usize, const NC2: usize>(
    pair: &PotPair,
    acc: &[[V; NC2]; NC1],
    out: &mut [f64],
    np: usize,
    g0: usize,
) {
    let (n1, n2, nb) = (pair.n1, pair.n2, pair.nb);
    let nl = LN.min(np - g0);
    let mut tmp = [[[0.0; LN]; NC2]; NC1];
    let mut res_buf = [[[0.0; LN]; NC2]; NC1];
    // Cartesian -> functions: ket first (tmp[i1][j2]), then bra (res[j1][j2])
    if let Some(t2) = &pair.t2 {
        for i1 in 0..NC1 {
            for &(j2, i2, cv) in t2 {
                for l in 0..LN {
                    tmp[i1][j2][l] += cv * acc[i1][i2][l];
                }
            }
        }
    }
    let src: &[[V; NC2]; NC1] = if pair.t2.is_some() { &tmp } else { acc };
    let res: &[[V; NC2]; NC1] = if let Some(t1) = &pair.t1 {
        for &(j1, i1, cv) in t1 {
            for j2 in 0..n2 {
                for l in 0..LN {
                    res_buf[j1][j2][l] += cv * src[i1][j2][l];
                }
            }
        }
        &res_buf
    } else {
        src
    };
    for j1 in 0..n1 {
        for j2 in 0..n2 {
            let (ia, ib) = if pair.swap { (j2, j1) } else { (j1, j2) };
            let o = (ia * nb + ib) * np + g0;
            out[o..o + nl].copy_from_slice(&res[j1][j2][..nl]);
        }
    }
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::needless_range_loop, reason = "indexes several arrays in step")]
fn kernel<
    const L1: usize,
    const L2: usize,
    const NC1: usize,
    const NC2: usize,
    const NR: usize,
    const W: usize,
    const GA: usize,
    const GB: usize,
    S: Lanes,
>(
    pair: &PotPair,
    prims: &[PotPrim],
    op: &Operator,
    x: &[f64],
    y: &[f64],
    z: &[f64],
    sink: &mut Sink,
) {
    let np = x.len();
    let ab = pair.ab;
    let mixed = op.has_erf();
    let om2 = op.omega() * op.omega();
    if let Sink::Contract(c) = sink {
        // the in-place row access of contract_chunk (rows of shells 1, 2 within every block)
        let (r1, r2) = if pair.swap {
            (c.b0, c.a0)
        } else {
            (c.a0, c.b0)
        };
        let (t1, t2) = if c.cart {
            (&None, &None)
        } else {
            (&pair.t1, &pair.t2)
        };
        assert!(
            rows_in_bounds(t1, r1, NC1, c.dstride, c.ld)
                && rows_in_bounds(t2, r2, NC2, c.dstride, c.ld)
        );
    }
    let mut g = [[[[0.0; LN]; GB]; GA]; 3];
    let mut acc = [[[0.0; LN]; NC2]; NC1];
    let mut g0 = 0;
    while g0 < np {
        let nl = LN.min(np - g0);
        let mut c = [[0.0; LN]; 3];
        if nl == LN {
            c[0] = x[g0..g0 + LN].try_into().unwrap();
            c[1] = y[g0..g0 + LN].try_into().unwrap();
            c[2] = z[g0..g0 + LN].try_into().unwrap();
        } else {
            for l in 0..LN {
                let gi = g0 + l.min(nl - 1);
                c[0][l] = x[gi];
                c[1][l] = y[gi];
                c[2][l] = z[gi];
            }
        }
        let cv = [S::load8(&c[0]), S::load8(&c[1]), S::load8(&c[2])];
        let mut first = true;
        for q in prims {
            let pc = [
                S::splat(q.pc[0]) - cv[0],
                S::splat(q.pc[1]) - cv[1],
                S::splat(q.pc[2]) - cv[2],
            ];
            let tv = S::splat(q.p) * pc[0].mul_add(pc[0], pc[1].mul_add(pc[1], pc[2] * pc[2]));
            if L1 + L2 <= 2 {
                // closed forms instead of Rys quadrature (see closed_ints)
                closed_ints::<L1, L2, NC1, NC2, S>(&mut acc, first, q, &pc, tv, &ab, op, om2);
                first = false;
                continue;
            }
            let mut u = [[0.0; LN]; NR];
            let mut w = [[0.0; LN]; NR];
            let cf = q.coef * op.coulomb();
            if cf != 0.0 {
                roots_vec::<NR, W, S>(tv, cf, &mut u, &mut w);
                for i in 0..NR {
                    if first {
                        add_root::<L1, L2, NC1, NC2, GA, GB, true, S>(
                            &mut acc, &u[i], &w[i], q, &pc, &ab, &mut g,
                        );
                        first = false;
                    } else {
                        add_root::<L1, L2, NC1, NC2, GA, GB, false, S>(
                            &mut acc, &u[i], &w[i], q, &pc, &ab, &mut g,
                        );
                    }
                }
            }
            if mixed {
                let k2 = om2 / (om2 + q.p);
                let kc = k2.sqrt() * q.coef * op.long_range();
                roots_vec::<NR, W, S>(tv * S::splat(k2), kc, &mut u, &mut w);
                for ui in u.iter_mut() {
                    (S::load8(ui) * S::splat(k2)).store(ui);
                }
                for i in 0..NR {
                    if first {
                        add_root::<L1, L2, NC1, NC2, GA, GB, true, S>(
                            &mut acc, &u[i], &w[i], q, &pc, &ab, &mut g,
                        );
                        first = false;
                    } else {
                        add_root::<L1, L2, NC1, NC2, GA, GB, false, S>(
                            &mut acc, &u[i], &w[i], q, &pc, &ab, &mut g,
                        );
                    }
                }
            }
        }
        if first {
            // no primitive contributed
            for r in acc.iter_mut() {
                for v in r.iter_mut() {
                    *v = [0.0; LN];
                }
            }
        }
        match sink {
            Sink::Store(out) => store_chunk::<NC1, NC2>(pair, &acc, out, np, g0),
            Sink::Contract(c) => contract_chunk::<NC1, NC2, S>(pair, &acc, c, g0),
        }
        g0 += LN;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::basis::BasisDef;
    use crate::pairs::{PairOptions, PairSet};

    fn test_basis(pure: bool) -> BasisSet {
        // contracted s, p, d and uncontracted f, g on two centres
        let json = r#"{"name":"t","elements":{"8":{"electron_shells":[
          {"function_type":"gto","angular_momentum":[0],"exponents":["130.7","23.81","6.44"],"coefficients":[["0.154","0.535","0.444"]]},
          {"function_type":"gto","angular_momentum":[0],"exponents":["0.38"],"coefficients":[["1.0"]]},
          {"function_type":"gto","angular_momentum":[1],"exponents":["5.03","1.17","0.38"],"coefficients":[["0.156","0.607","0.392"]]},
          {"function_type":"gto","angular_momentum":[2],"exponents":["1.2","0.4"],"coefficients":[["0.6","0.5"]]},
          {"function_type":"gto","angular_momentum":[3],"exponents":["0.9"],"coefficients":[["1.0"]]},
          {"function_type":"gto","angular_momentum":[4],"exponents":["0.7"],"coefficients":[["1.0"]]}]},
          "1":{"electron_shells":[
          {"function_type":"gto","angular_momentum":[0],"exponents":["3.42","0.62"],"coefficients":[["0.15","0.53"]]},
          {"function_type":"gto","angular_momentum":[1],"exponents":["0.8"],"coefficients":[["1.0"]]},
          {"function_type":"gto","angular_momentum":[3],"exponents":["0.5"],"coefficients":[["1.0"]]}]}}}"#;
        let def = BasisDef::from_bse_json(json).unwrap();
        BasisSet::new(&def, &[8, 1], &[[0.1, -0.2, 0.3], [1.2, 0.9, -0.8]], pure).unwrap()
    }

    fn points(n: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        // deterministic pseudo-random points from very close to far away
        let mut s = 12345u64;
        let mut r = || {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (s >> 11) as f64 / (1u64 << 53) as f64
        };
        let (mut x, mut y, mut z) = (vec![], vec![], vec![]);
        for i in 0..n {
            let scale = [0.05, 0.5, 2.0, 6.0, 25.0][i % 5];
            x.push(0.1 + scale * (2.0 * r() - 1.0));
            y.push(-0.2 + scale * (2.0 * r() - 1.0));
            z.push(0.3 + scale * (2.0 * r() - 1.0));
        }
        // a point exactly on a centre
        x[3] = 0.1;
        y[3] = -0.2;
        z[3] = 0.3;
        (x, y, z)
    }

    #[test]
    fn rys_matches_mcmurchie_davidson() {
        for pure in [true, false] {
            let basis = test_basis(pure);
            let pairs = PairSet::build(&basis, PairOptions::default());
            let (x, y, z) = points(37);
            let np = x.len();
            let mut worst: f64 = 0.0;
            for pr in pairs.pairs() {
                let pp = PotPair::new(&basis, pr.sa, pr.sb, 1e-22);
                let mut a = vec![0.0; pp.nab() * np];
                pp.compute(&Operator::Coulomb, &x, &y, &z, 0.0, &mut a);
                let mut b = vec![0.0; pr.nab() * np];
                let (mut r, mut w) = (Vec::new(), Vec::new());
                crate::one_e::potential_pair_points(pr, &x, &y, &z, &mut b, &mut r, &mut w);
                for (u, v) in a.iter().zip(&b) {
                    worst = worst.max((u - v).abs() / v.abs().max(1.0));
                }
            }
            assert!(worst < 1e-13, "pure={pure}: worst deviation {worst:e}");
            eprintln!("pure={pure}: Rys vs MD worst {worst:.2e}");
        }
    }

    #[test]
    fn contract_matches_compute() {
        let basis = test_basis(true);
        let pairs = PairSet::build(&basis, PairOptions::default());
        let (x, y, z) = points(21);
        let np = x.len();
        let ld = 24;
        let nbf = basis.nbf;
        let nd = 2;
        let ds = nbf * ld;
        let mut f = vec![0.0; nd * ds];
        for k in 0..nd {
            for m in 0..nbf {
                for g in 0..np {
                    f[k * ds + m * ld + g] = ((k * 7 + m * 3 + g) as f64 * 0.37).sin();
                }
            }
        }
        let mut worst: f64 = 0.0;
        for pr in pairs.pairs() {
            let pp = PotPair::new(&basis, pr.sa, pr.sb, 1e-22);
            let (a0, b0) = (basis.shells[pr.sa].ao_start, basis.shells[pr.sb].ao_start);
            let mut a = vec![0.0; pp.nab() * np];
            pp.compute(&Operator::Coulomb, &x, &y, &z, 0.0, &mut a);
            let mut want = vec![0.0; nd * ds];
            let do_b = pr.sa != pr.sb;
            for k in 0..nd {
                for ia in 0..pp.na {
                    for ib in 0..pp.nb {
                        for g in 0..np {
                            let v = a[(ia * pp.nb + ib) * np + g];
                            want[k * ds + (a0 + ia) * ld + g] += v * f[k * ds + (b0 + ib) * ld + g];
                            if do_b {
                                want[k * ds + (b0 + ib) * ld + g] +=
                                    v * f[k * ds + (a0 + ia) * ld + g];
                            }
                        }
                    }
                }
            }
            let mut got = vec![0.0; nd * ds];
            let mut c = PotContract {
                f: &f,
                g: &mut got,
                nd,
                dstride: ds,
                ld,
                a0,
                b0,
                do_a: true,
                do_b,
                cart: false,
            };
            pp.contract(&Operator::Coulomb, &x, &y, &z, 0.0, &mut c);
            for (u, v) in got.iter().zip(&want) {
                worst = worst.max((u - v).abs() / v.abs().max(1.0));
            }
        }
        assert!(worst < 1e-13, "worst {worst:e}");
    }

    /// erf(omega r)/r against the identity erf(omega r)/r = 1/r - erfc(omega r)/r evaluated
    /// as a limit: for omega -> infinity the long-range operator becomes 1/r, and the mixed
    /// operator is linear in its coefficients.
    #[test]
    fn erf_operator_limits_and_linearity() {
        let basis = test_basis(true);
        let pairs = PairSet::build(&basis, PairOptions::default());
        let (x, y, z) = points(19);
        let np = x.len();
        let mut worst: f64 = 0.0;
        for pr in pairs.pairs() {
            let pp = PotPair::new(&basis, pr.sa, pr.sb, 1e-22);
            if pp.md.is_some() {
                continue;
            }
            let n = pp.nab() * np;
            let (mut a, mut b, mut c, mut m) =
                (vec![0.0; n], vec![0.0; n], vec![0.0; n], vec![0.0; n]);
            pp.compute(&Operator::Coulomb, &x, &y, &z, 0.0, &mut a);
            pp.compute(&Operator::LongRange { omega: 1e9 }, &x, &y, &z, 0.0, &mut b);
            pp.compute(
                &Operator::LongRange { omega: 0.33 },
                &x,
                &y,
                &z,
                0.0,
                &mut c,
            );
            pp.compute(
                &Operator::from_libxc(0.19, 0.46, 0.33),
                &x,
                &y,
                &z,
                0.0,
                &mut m,
            );
            for i in 0..n {
                worst = worst.max((a[i] - b[i]).abs() / a[i].abs().max(1.0));
                let want = 0.65 * a[i] - 0.46 * c[i];
                worst = worst.max((m[i] - want).abs() / want.abs().max(1.0));
            }
        }
        assert!(worst < 1e-12, "worst {worst:e}");
    }

    /// erf-attenuated s-s integral against the closed form: for two s primitives on one
    /// centre, <s|erf(w r_C)/r_C|s> = (2 pi/p) K kappa F0(T kappa^2).
    #[test]
    fn erf_ss_closed_form() {
        let json = r#"{"name":"t","elements":{"1":{"electron_shells":[
          {"function_type":"gto","angular_momentum":[0],"exponents":["0.9"],"coefficients":[["1.0"]]}]}}}"#;
        let def = BasisDef::from_bse_json(json).unwrap();
        let basis = BasisSet::new(&def, &[1], &[[0.0, 0.0, 0.0]], true).unwrap();
        let pp = PotPair::new(&basis, 0, 0, 1e-22);
        let c = basis.shells[0].coefs[0];
        let p = 1.8;
        for &(d, om) in &[(0.0, 0.4), (0.7, 0.4), (3.0, 1.3), (9.0, 0.1)] {
            let mut a = [0.0];
            pp.compute(
                &Operator::LongRange { omega: om },
                &[d],
                &[0.0],
                &[0.0],
                0.0,
                &mut a,
            );
            let k2 = om * om / (om * om + p);
            let mut f = [0.0];
            crate::core::boys::boys(0, p * d * d * k2, &mut f);
            let want = 2.0 * PI / p * c * c * k2.sqrt() * f[0];
            assert!(
                (a[0] - want).abs() < 1e-15 * want.abs().max(1.0),
                "{d} {om}: {} vs {want}",
                a[0]
            );
        }
    }
}
