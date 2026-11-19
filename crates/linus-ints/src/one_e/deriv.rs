//! The kernels of the nuclear first and second derivatives of one-electron integrals: the
//! derivative blocks of one shell pair for the overlap, the kinetic energy, the dipole and
//! the point-charge potential (basis-function centers and operator centers), and the Hermite
//! densities of a pair. The derivative matrices and the gradient and Hessian contractions
//! with densities over the whole basis are operations of [`crate::one_e::OneEPlan`].
//!
//! Conventions:
//! - Derivative blocks of a shell pair (a, b) are taken with respect to the center A of the
//!   first shell unless stated otherwise; layout \[k\]\[mu\]\[nu\] (k = x, y, z) or \[k\]\[l\]\[mu\]\[nu\].
//! - Derivative matrices are (3 natom) x nbf x nbf, row-major, with perturbation index
//!   3 * atom + k: M\[x\]\[m\]\[n\] = d M_mn / d X_x.
//! - Gradients are natom x [x, y, z]; Hessians are (3 natom) x (3 natom) row-major.
//! - Densities (D, W) are symmetric nbf x nbf row-major matrices.
//!
//! S, T and dipole integrals use the 1D factorization with differentiated basis functions
//! (dG_i/dA = 2a G_{i+1} - i G_{i-1}); the potential uses the Hermite pair data of
//! [`crate::core::deriv`] with the operator-center derivative d/dC R_t(P - C) = -R_{t+1}.

use crate::basis::{cart_components, ncart, Shell};
use crate::core::hermite::{hermite_e_1d, ShellPair};
use crate::core::util::PairTransform;
use crate::potential::{self, PointCharges};
use std::f64::consts::PI;

/// 1D quantities for one primitive pair and one Cartesian direction, i <= la, j <= lb,
/// index i * (lb + 1) + j: overlap s, kinetic t (operator on b), moment m = <i|x - O|j>, and
/// their derivatives with respect to A (suffix A, AA) and B (suffix B); see the K_* indices.
#[derive(Default)]
struct Prim1D {
    v: [Vec<f64>; 10],
}

const K_S: usize = 0;
const K_SA: usize = 1;
const K_SAA: usize = 2;
const K_SB: usize = 3;
const K_T: usize = 4;
const K_TA: usize = 5;
const K_TAA: usize = 6;
const K_M: usize = 7;
const K_MA: usize = 8;
const K_MB: usize = 9;

impl Prim1D {
    #[allow(clippy::too_many_arguments)]
    fn compute(
        &mut self,
        la: usize,
        lb: usize,
        a: f64,
        b: f64,
        xab: f64,
        ao: f64,
        e: &mut Vec<f64>,
    ) {
        let (imax, jmax) = (la + 2, lb + 2);
        let nt = imax + jmax + 1;
        e.resize((imax + 1) * (jmax + 1) * nt, 0.0);
        hermite_e_1d(imax, jmax, a, b, xab, e);
        let sq = (PI / (a + b)).sqrt();
        let w = jmax + 1;
        let s = |i: usize, j: usize| e[(i * w + j) * nt] * sq;
        // kinetic 1D (operator -1/2 d²/dx² on b)
        let t = |i: usize, j: usize| {
            let mut v = -2.0 * b * b * s(i, j + 2) + b * (2 * j + 1) as f64 * s(i, j);
            if j >= 2 {
                v -= 0.5 * (j * (j - 1)) as f64 * s(i, j - 2);
            }
            v
        };
        let m = |i: usize, j: usize| s(i + 1, j) + ao * s(i, j);
        let n = (la + 1) * (lb + 1);
        for v in self.v.iter_mut() {
            v.clear();
            v.resize(n, 0.0);
        }
        let da = |f: &dyn Fn(usize, usize) -> f64, i: usize, j: usize| {
            2.0 * a * f(i + 1, j) - if i >= 1 { i as f64 * f(i - 1, j) } else { 0.0 }
        };
        let db = |f: &dyn Fn(usize, usize) -> f64, i: usize, j: usize| {
            2.0 * b * f(i, j + 1) - if j >= 1 { j as f64 * f(i, j - 1) } else { 0.0 }
        };
        let daa = |f: &dyn Fn(usize, usize) -> f64, i: usize, j: usize| {
            4.0 * a * a * f(i + 2, j) - 2.0 * a * (2 * i + 1) as f64 * f(i, j)
                + if i >= 2 {
                    (i * (i - 1)) as f64 * f(i - 2, j)
                } else {
                    0.0
                }
        };
        for i in 0..=la {
            for j in 0..=lb {
                let k = i * (lb + 1) + j;
                let v = &mut self.v;
                v[K_S][k] = s(i, j);
                v[K_SA][k] = da(&s, i, j);
                v[K_SAA][k] = daa(&s, i, j);
                v[K_SB][k] = db(&s, i, j);
                v[K_T][k] = t(i, j);
                v[K_TA][k] = da(&t, i, j);
                v[K_TAA][k] = daa(&t, i, j);
                v[K_M][k] = m(i, j);
                v[K_MA][k] = da(&m, i, j);
                v[K_MB][k] = db(&m, i, j);
            }
        }
    }
}

/// What to compute in [`one_e_cart`].
#[derive(Clone, Copy, PartialEq, Eq)]
enum OneKind {
    /// S and T first derivatives w.r.t. A: 2 x [3] blocks
    ST1,
    /// S and T second derivatives w.r.t. A: 2 x [9] blocks
    ST2,
    /// dipole first derivatives w.r.t. A and B: 2 x [3 k][3 c] blocks
    Dip1,
}

/// The scratch of the derivative blocks of one shell pair: the Cartesian blocks, the 1D
/// quantities, the Hermite table and the transformation's buffer. Reused across pairs, it
/// allocates nothing once it has reached the largest pair's sizes.
#[derive(Default)]
pub(crate) struct CartScratch {
    cart: Vec<f64>,
    d1: [Prim1D; 3],
    e: Vec<f64>,
    tmp: Vec<f64>,
}

impl CartScratch {
    /// Scratch at its full size for shells of angular momentum at most `lmax` and at most
    /// `nf` functions.
    pub(crate) fn reserved(lmax: usize, nf: usize) -> Self {
        let nc = ncart(lmax);
        let (l2, n1) = (lmax + 3, (lmax + 1) * (lmax + 1));
        let mut sc = CartScratch {
            cart: Vec::with_capacity(18 * nc * nc),
            e: Vec::with_capacity(l2 * l2 * (2 * lmax + 5)),
            tmp: Vec::with_capacity(nf * nc),
            ..Default::default()
        };
        for d in &mut sc.d1 {
            for v in &mut d.v {
                v.reserve(n1);
            }
        }
        sc
    }

    /// The buffers' capacities (tests: a reserved scratch does not grow in a contraction).
    #[cfg(test)]
    pub(crate) fn caps(&self) -> Vec<usize> {
        let d1 = self.d1.iter().flat_map(|d| d.v.iter().map(Vec::capacity));
        let own = [self.cart.capacity(), self.e.capacity(), self.tmp.capacity()];
        own.into_iter().chain(d1).collect()
    }

    /// The bytes of a scratch of [`CartScratch::reserved`].
    pub(crate) fn bytes(lmax: usize, nf: usize) -> usize {
        let nc = ncart(lmax);
        let l2 = lmax + 3;
        8 * (18 * nc * nc + l2 * l2 * (2 * lmax + 5) + nf * nc + 30 * (lmax + 1) * (lmax + 1))
    }
}

/// Cartesian blocks for a shell pair, accumulated over primitive pairs, into `sc.cart`.
/// Returns the number of components; `sc.cart` holds ncomp blocks of nca x ncb.
fn one_e_cart(
    a: &Shell,
    b: &Shell,
    kind: OneKind,
    origin: [f64; 3],
    sc: &mut CartScratch,
) -> usize {
    let (la, lb) = (a.l, b.l);
    let ca = cart_components(la);
    let cb = cart_components(lb);
    let (nca, ncb) = (ca.len(), cb.len());
    let ncomp = match kind {
        OneKind::ST1 => 6,
        OneKind::ST2 => 18,
        OneKind::Dip1 => 18,
    };
    let CartScratch {
        cart: out, d1, e, ..
    } = sc;
    out.clear();
    out.resize(ncomp * nca * ncb, 0.0);
    let w = lb + 1;
    for (&ai, &ci) in a.exps.iter().zip(&a.coefs) {
        for (&bj, &cj) in b.exps.iter().zip(&b.coefs) {
            for d in 0..3 {
                d1[d].compute(
                    la,
                    lb,
                    ai,
                    bj,
                    a.center[d] - b.center[d],
                    a.center[d] - origin[d],
                    e,
                );
            }
            let cc = ci * cj;
            for (ia, pa) in ca.iter().enumerate() {
                for (ib, pb) in cb.iter().enumerate() {
                    let ix = [pa[0] * w + pb[0], pa[1] * w + pb[1], pa[2] * w + pb[2]];
                    let g = |d: usize, kk: usize| d1[d].v[kk][ix[d]];
                    let o = ia * ncb + ib;
                    let blk = nca * ncb;
                    match kind {
                        OneKind::ST1 => {
                            for k in 0..3 {
                                let mut sv = 1.0;
                                for d in 0..3 {
                                    sv *= if d == k { g(d, K_SA) } else { g(d, K_S) };
                                }
                                let mut tv = 0.0;
                                for e_ in 0..3 {
                                    let mut prod = 1.0;
                                    for d in 0..3 {
                                        prod *= match (d == e_, d == k) {
                                            (true, true) => g(d, K_TA),
                                            (true, false) => g(d, K_T),
                                            (false, true) => g(d, K_SA),
                                            (false, false) => g(d, K_S),
                                        };
                                    }
                                    tv += prod;
                                }
                                out[k * blk + o] += cc * sv;
                                out[(3 + k) * blk + o] += cc * tv;
                            }
                        }
                        OneKind::ST2 => {
                            for k in 0..3 {
                                for l in 0..3 {
                                    // order of derivative in each direction
                                    let ord = |d: usize| (d == k) as usize + (d == l) as usize;
                                    let sf = |d: usize| match ord(d) {
                                        0 => g(d, K_S),
                                        1 => g(d, K_SA),
                                        _ => g(d, K_SAA),
                                    };
                                    let tf = |d: usize| match ord(d) {
                                        0 => g(d, K_T),
                                        1 => g(d, K_TA),
                                        _ => g(d, K_TAA),
                                    };
                                    let sv = sf(0) * sf(1) * sf(2);
                                    let tv = tf(0) * sf(1) * sf(2)
                                        + sf(0) * tf(1) * sf(2)
                                        + sf(0) * sf(1) * tf(2);
                                    out[(k * 3 + l) * blk + o] += cc * sv;
                                    out[(9 + k * 3 + l) * blk + o] += cc * tv;
                                }
                            }
                        }
                        OneKind::Dip1 => {
                            for k in 0..3 {
                                for c in 0..3 {
                                    let mut va = 1.0;
                                    let mut vb = 1.0;
                                    for d in 0..3 {
                                        let (fa, fb) = match (d == c, d == k) {
                                            (true, true) => (g(d, K_MA), g(d, K_MB)),
                                            (true, false) => (g(d, K_M), g(d, K_M)),
                                            (false, true) => (g(d, K_SA), g(d, K_SB)),
                                            (false, false) => (g(d, K_S), g(d, K_S)),
                                        };
                                        va *= fa;
                                        vb *= fb;
                                    }
                                    out[(k * 3 + c) * blk + o] += cc * va;
                                    out[(9 + k * 3 + c) * blk + o] += cc * vb;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    ncomp
}

/// First derivatives of the overlap and kinetic blocks with respect to the center A of `a`:
/// s_out, t_out are \[3\]\[na\]\[nb\]. (By translational invariance d/dB = -d/dA.)
pub fn overlap_kinetic_deriv1_block(a: &Shell, b: &Shell, s_out: &mut [f64], t_out: &mut [f64]) {
    st_deriv1(a, b, s_out, t_out, &mut CartScratch::default());
}

/// [`overlap_kinetic_deriv1_block`] with the caller's scratch.
pub(crate) fn st_deriv1(
    a: &Shell,
    b: &Shell,
    s_out: &mut [f64],
    t_out: &mut [f64],
    sc: &mut CartScratch,
) {
    one_e_cart(a, b, OneKind::ST1, [0.0; 3], sc);
    transform_halves(a, b, 3, sc, s_out, t_out);
}

/// The two halves of `sc.cart` (`nrows` Cartesian blocks each) transformed to the shells'
/// functions into `first` and `second`.
fn transform_halves(
    a: &Shell,
    b: &Shell,
    nrows: usize,
    sc: &mut CartScratch,
    first: &mut [f64],
    second: &mut [f64],
) {
    let n = nrows * ncart(a.l) * ncart(b.l);
    let tf = PairTransform::of_shells(a, b);
    let CartScratch { cart, tmp, .. } = sc;
    tf.rows(&cart[..n], nrows, first, tmp);
    tf.rows(&cart[n..], nrows, second, tmp);
}

/// Second derivatives d²/dA_k dA_l of the overlap and kinetic blocks: \[3\]\[3\]\[na\]\[nb\].
/// (d²/dB_k dB_l = d²/dA_k dA_l and d²/dA_k dB_l = -d²/dA_k dA_l.)
pub fn overlap_kinetic_deriv2_block(a: &Shell, b: &Shell, s_out: &mut [f64], t_out: &mut [f64]) {
    st_deriv2(a, b, s_out, t_out, &mut CartScratch::default());
}

/// [`overlap_kinetic_deriv2_block`] with the caller's scratch.
pub(crate) fn st_deriv2(
    a: &Shell,
    b: &Shell,
    s_out: &mut [f64],
    t_out: &mut [f64],
    sc: &mut CartScratch,
) {
    one_e_cart(a, b, OneKind::ST2, [0.0; 3], sc);
    transform_halves(a, b, 9, sc, s_out, t_out);
}

/// First derivatives of the dipole blocks <a|(r - O)_c|b>: out_a\[k\]\[c\]\[na\]\[nb\] = d/dA_k,
/// out_b\[k\]\[c\]\[na\]\[nb\] = d/dB_k. (d/dA_k + d/dB_k = delta_kc S.)
pub fn dipole_deriv1_block(
    a: &Shell,
    b: &Shell,
    origin: [f64; 3],
    out_a: &mut [f64],
    out_b: &mut [f64],
) {
    dip_deriv1(a, b, origin, (out_a, out_b), &mut CartScratch::default());
}

/// [`dipole_deriv1_block`] with the caller's scratch.
pub(crate) fn dip_deriv1(
    a: &Shell,
    b: &Shell,
    origin: [f64; 3],
    (out_a, out_b): (&mut [f64], &mut [f64]),
    sc: &mut CartScratch,
) {
    one_e_cart(a, b, OneKind::Dip1, origin, sc);
    transform_halves(a, b, 9, sc, out_a, out_b);
}

/// Shell pairs (i >= j) as a flat list.
pub(crate) fn shell_pairs(n: usize) -> Vec<(usize, usize)> {
    (0..n).flat_map(|i| (0..=i).map(move |j| (i, j))).collect()
}

/// Add a pair block (na x nb) for shells (i, j) into a symmetric nbf x nbf matrix: element
/// (a, b), and (b, a) as well when i != j.
#[allow(clippy::too_many_arguments, reason = "a kernel's arrays and sizes")]
pub(crate) fn add_pair_block(
    m: &mut [f64],
    nbf: usize,
    a0: usize,
    b0: usize,
    na: usize,
    nb: usize,
    same: bool,
    blk: &[f64],
    f: f64,
) {
    for x in 0..na {
        for y in 0..nb {
            let v = f * blk[x * nb + y];
            m[(a0 + x) * nbf + b0 + y] += v;
            if !same {
                m[(b0 + y) * nbf + a0 + x] += v;
            }
        }
    }
}

/// Weighted sum over a pair block: sum_ab D_ab blk_ab, times 2 for i != j (D symmetric).
#[inline]
pub(crate) fn pair_dot(
    d: &[f64],
    nbf: usize,
    a0: usize,
    b0: usize,
    na: usize,
    nb: usize,
    blk: &[f64],
) -> f64 {
    let mut s = 0.0;
    for x in 0..na {
        let row = &d[(a0 + x) * nbf + b0..(a0 + x) * nbf + b0 + nb];
        for (y, &dv) in row.iter().enumerate() {
            s += dv * blk[x * nb + y];
        }
    }
    s
}

// ---------------------------------------------------------------------------------------
// Point-charge potential
// ---------------------------------------------------------------------------------------

/// First derivatives of the potential block of a shell pair, sum_C -q_C <a|1/|r-C||b>:
/// out_a, out_b: \[3\]\[nab\] (w.r.t. the centers A, B of the shells), out_c: \[ncharge\]\[3\]\[nab\]
/// (w.r.t. the charge positions). `pair` and `pd` must describe the same pair.
pub fn potential_deriv1_block(
    pair: &ShellPair,
    pd: &crate::core::deriv::PairDeriv,
    charges: &[(f64, [f64; 3])],
    out_a: &mut [f64],
    out_b: &mut [f64],
    out_c: &mut [f64],
) {
    let nab = pair.nab();
    let nc = charges.len();
    out_a[..3 * nab].fill(0.0);
    out_c[..3 * nab * nc].fill(0.0);
    let src = PointCharges {
        charges,
        first_slot: 0,
    };
    potential::deriv_pair_blocks(pair, pd, &src, out_a, out_c);
    potential::deriv_b_blocks(out_a, out_c, nc, nab, out_b);
}

/// Hermite density of a pair for one primitive: out[r] = f sum_ab E[r][ab] D_ab.
#[inline]
#[allow(clippy::too_many_arguments, reason = "a kernel's arrays and sizes")]
pub(crate) fn herm_dens(
    e: &[f64],
    nh: usize,
    ncomp: usize,
    d: &[f64],
    nbf: usize,
    a0: usize,
    b0: usize,
    na: usize,
    nb: usize,
    f: f64,
    out: &mut [f64],
) {
    let nab = na * nb;
    for r in 0..nh {
        for c in 0..ncomp {
            let row = &e[(r * ncomp + c) * nab..(r * ncomp + c + 1) * nab];
            let mut s = 0.0;
            for x in 0..na {
                let drow = &d[(a0 + x) * nbf + b0..(a0 + x) * nbf + b0 + nb];
                let erow = &row[x * nb..(x + 1) * nb];
                for (dv, ev) in drow.iter().zip(erow) {
                    s += dv * ev;
                }
            }
            out[c * nh + r] = f * s;
        }
    }
}
