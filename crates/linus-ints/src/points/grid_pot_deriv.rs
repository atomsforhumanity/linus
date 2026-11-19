//! Nuclear derivatives of the grid-point potential integrals of [`crate::points::grid_pot`]
//! (gradients of seminumerical exchange), contracted on the fly.
//!
//! For a primitive pair (bra exponent a, ket exponent b) the derivatives with respect to the
//! shell centres follow from the 1D Rys integrals one order higher:
//!
//!   d/dA_x I(i, j) = 2a I(i+1, j) - i I(i-1, j),   d/dB_x I(i, j) = 2b I(i, j+1) - j I(i, j-1),
//!
//! with I(i, j+1) = I(i+1, j) + AB_x I(i, j), on the floor((la+lb+1)/2)+1 roots that integrate
//! the raised degree exactly (the same roots give the integrals themselves). The integrals
//! depend on the three centres only through their differences, so the derivative with respect
//! to the point is d/dr_g = -(d/dA + d/dB).
//!
//! The derivatives are contracted with a pair density per point,
//!
//!   P_ab(g) = w_g sum_t c_t (F\[i_t\]\[a\](g) F\[j_t\]\[b\](g) + F\[j_t\]\[a\](g) F\[i_t\]\[b\](g)),
//!
//! (a in shell sa, b in shell sb; F blocks as in [`crate::points::grid_pot::PotContract`]), giving the
//! centre gradients sum_g sum_ab P_ab(g) dA_ab(g)/dA, .../dB and the per-point derivatives
//! -sum_ab P_ab(g) (dA_ab(g)/dA + dA_ab(g)/dB). In the same pass the integrals are contracted
//! with F blocks into G exactly as [`crate::points::grid_pot::PotPair::contract`] does. Points are
//! processed eight at a time; kernels are monomorphized per (la, lb) up to (g, g) and compiled
//! for AVX-512, AVX2+FMA and baseline x86-64 with runtime dispatch. Any [`Operator`]
//! (Coulomb, erf-attenuated and their combinations) is supported.
//!
//! The classes with la + lb <= 2 ((s|s), (p|s), (p|p), (d|s): 47% of the derivative-kernel
//! time of ts113 def2-TZVP) skip the Rys quadrature: their derivative integrals, contracted
//! with the pair density, have short closed forms in the Obara-Saika auxiliary integrals
//! \[s\]^(m) = coef F_m(T) (`closed_low`, `closed_l2`; one vectorized Boys evaluation instead
//! of la + lb + 1 roots and their 1D integrals). F and G rows are read and updated in place
//! (`crate::points::grid_pot::cart_rows_v`, bounds checked once per pair).

use crate::core::rys::roots_vec;
use crate::operator::Operator;
use crate::points::grid_pot::{
    add_g_rows_v, aux_ss, cart_rows_v, rows_in_bounds, PotPair, PotPrim, CART, D2, LN, V,
};
use linus_num::simd::{Isa, Lanes, Portable};
use std::sync::OnceLock;

fn isa() -> Isa {
    static I: OnceLock<Isa> = OnceLock::new();
    *I.get_or_init(Isa::detect)
}

/// Inputs and outputs of [`PotPair::contract_deriv`].
pub struct PotDerivContract<'a> {
    /// F blocks: block k holds rows (functions) of stride `ld` at k * dstride (zero beyond the
    /// np points).
    pub f: &'a [f64],
    /// G accumulation for F blocks 0..ng (same layout): G\[k\]\[a\] += A_ab F\[k\]\[b\] (if `do_a`),
    /// G\[k\]\[b\] += A_ab F\[k\]\[a\] (if `do_b`).
    pub g: &'a mut [f64],
    pub ng: usize,
    pub dstride: usize,
    pub ld: usize,
    /// First row of shells sa and sb.
    pub a0: usize,
    pub b0: usize,
    pub do_a: bool,
    pub do_b: bool,
    /// Pair-density terms (block i, block j, c).
    pub terms: &'a [(usize, usize, f64)],
    /// Quadrature weights of the points (at least np entries; zero padding beyond np is not
    /// needed: padded lanes are ignored).
    pub w: &'a [f64],
    /// Per-point derivatives, accumulated: gpt[d * ld + g] (d = x, y, z).
    pub gpt: &'a mut [f64],
    /// Rows of F and G are Cartesian components (a0, b0 Cartesian offsets; see
    /// [`crate::points::grid_pot::PotContract::cart`]).
    pub cart: bool,
}

impl PotPair {
    /// Derivative contraction over the primitive pairs `prims` of this pair (all of
    /// `self.prims`, a screened prefix, or a list from [`PotPair::screen_prims`]) for the np
    /// points (x, y, z): returns the gradients with respect to the centres of shells sa and sb
    /// and accumulates the per-point derivatives and G (see [`PotDerivContract`]).
    #[allow(clippy::too_many_arguments)]
    pub fn contract_deriv(
        &self,
        prims: &[PotPrim],
        op: &Operator,
        x: &[f64],
        y: &[f64],
        z: &[f64],
        c: &mut PotDerivContract,
    ) -> [[f64; 3]; 2] {
        assert!(
            !self.is_md(),
            "derivative kernels need la, lb <= {} (got {}, {})",
            crate::points::grid_pot::RYS_LMAX,
            self.la,
            self.lb
        );
        let np = x.len();
        assert!(
            y.len() == np
                && z.len() == np
                && c.ld.is_multiple_of(LN)
                && c.ld >= np
                && c.w.len() >= np
        );
        assert!(c.gpt.len() >= 3 * c.ld && c.g.len() >= c.ng * c.dstride);
        let nblk = c
            .terms
            .iter()
            .map(|&(i, j, _)| i.max(j) + 1)
            .max()
            .unwrap_or(0)
            .max(c.ng);
        assert!(c.f.len() >= nblk * c.dstride);
        // classes with l1 + l2 > 4 in their own functions (smaller stack frames for the rest)
        let big = self.l1 + self.l2 > 4;
        match isa() {
            #[cfg(target_arch = "x86_64")]
            Isa::Avx512 => unsafe {
                if big {
                    deriv_big_avx512(self, prims, op, x, y, z, c)
                } else {
                    deriv_avx512(self, prims, op, x, y, z, c)
                }
            },
            #[cfg(target_arch = "x86_64")]
            Isa::Avx2 => unsafe {
                if big {
                    deriv_big_avx2(self, prims, op, x, y, z, c)
                } else {
                    deriv_avx2(self, prims, op, x, y, z, c)
                }
            },
            _ => {
                if big {
                    deriv_big::<Portable<false>>(self, prims, op, x, y, z, c)
                } else {
                    deriv_small::<Portable<false>>(self, prims, op, x, y, z, c)
                }
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512vl,avx512dq,fma")]
unsafe fn deriv_avx512(
    pair: &PotPair,
    prims: &[PotPrim],
    op: &Operator,
    x: &[f64],
    y: &[f64],
    z: &[f64],
    c: &mut PotDerivContract,
) -> [[f64; 3]; 2] {
    deriv_small::<linus_num::simd::Avx512>(pair, prims, op, x, y, z, c)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512vl,avx512dq,fma")]
unsafe fn deriv_big_avx512(
    pair: &PotPair,
    prims: &[PotPrim],
    op: &Operator,
    x: &[f64],
    y: &[f64],
    z: &[f64],
    c: &mut PotDerivContract,
) -> [[f64; 3]; 2] {
    deriv_big::<linus_num::simd::Avx512>(pair, prims, op, x, y, z, c)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn deriv_avx2(
    pair: &PotPair,
    prims: &[PotPrim],
    op: &Operator,
    x: &[f64],
    y: &[f64],
    z: &[f64],
    c: &mut PotDerivContract,
) -> [[f64; 3]; 2] {
    deriv_small::<linus_num::simd::Avx2>(pair, prims, op, x, y, z, c)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn deriv_big_avx2(
    pair: &PotPair,
    prims: &[PotPrim],
    op: &Operator,
    x: &[f64],
    y: &[f64],
    z: &[f64],
    c: &mut PotDerivContract,
) -> [[f64; 3]; 2] {
    deriv_big::<linus_num::simd::Avx2>(pair, prims, op, x, y, z, c)
}

// dkernel::<L1, L2, NC1, NC2, NR, W, GA = L1 + L2 + 2, GB = L2 + 2, S>; roots
// floor((l1 + l2 + 1) / 2) + 1, W = the Rys table width of that order
macro_rules! dk {
    ($l1:literal, $l2:literal, $nr:literal, $w:literal, $ga:literal, $gb:literal, $s:ty, $pair:expr, $prims:expr, $op:expr, $x:expr, $y:expr, $z:expr, $c:expr) => {
        dkernel::<
            $l1,
            $l2,
            { crate::basis::ncart($l1) },
            { crate::basis::ncart($l2) },
            $nr,
            $w,
            $ga,
            $gb,
            $s,
        >($pair, $prims, $op, $x, $y, $z, $c)
    };
}

#[inline(always)]
fn deriv_small<S: Lanes>(
    pair: &PotPair,
    prims: &[PotPrim],
    op: &Operator,
    x: &[f64],
    y: &[f64],
    z: &[f64],
    c: &mut PotDerivContract,
) -> [[f64; 3]; 2] {
    match (pair.l1, pair.l2) {
        (0, 0) => dk!(0, 0, 1, 2, 2, 2, S, pair, prims, op, x, y, z, c),
        (1, 0) => dk!(1, 0, 2, 4, 3, 2, S, pair, prims, op, x, y, z, c),
        (1, 1) => dk!(1, 1, 2, 4, 4, 3, S, pair, prims, op, x, y, z, c),
        (2, 0) => dk!(2, 0, 2, 4, 4, 2, S, pair, prims, op, x, y, z, c),
        (2, 1) => dk!(2, 1, 3, 8, 5, 3, S, pair, prims, op, x, y, z, c),
        (2, 2) => dk!(2, 2, 3, 8, 6, 4, S, pair, prims, op, x, y, z, c),
        (3, 0) => dk!(3, 0, 3, 8, 5, 2, S, pair, prims, op, x, y, z, c),
        (3, 1) => dk!(3, 1, 3, 8, 6, 3, S, pair, prims, op, x, y, z, c),
        (4, 0) => dk!(4, 0, 3, 8, 6, 2, S, pair, prims, op, x, y, z, c),
        _ => unreachable!(),
    }
}

#[inline(always)]
fn deriv_big<S: Lanes>(
    pair: &PotPair,
    prims: &[PotPrim],
    op: &Operator,
    x: &[f64],
    y: &[f64],
    z: &[f64],
    c: &mut PotDerivContract,
) -> [[f64; 3]; 2] {
    match (pair.l1, pair.l2) {
        (3, 2) => dk!(3, 2, 4, 8, 7, 4, S, pair, prims, op, x, y, z, c),
        (3, 3) => dk!(3, 3, 4, 8, 8, 5, S, pair, prims, op, x, y, z, c),
        (4, 1) => dk!(4, 1, 4, 8, 7, 3, S, pair, prims, op, x, y, z, c),
        (4, 2) => dk!(4, 2, 4, 8, 8, 4, S, pair, prims, op, x, y, z, c),
        (4, 3) => dk!(4, 3, 5, 12, 9, 5, S, pair, prims, op, x, y, z, c),
        (4, 4) => dk!(4, 4, 5, 12, 10, 6, S, pair, prims, op, x, y, z, c),
        _ => unreachable!(),
    }
}

/// 1D integrals g[a][b] for a + b <= L1 + L2 + 1, b <= L2 + 1 (one root, one direction).
#[inline(always)]
fn one_d_ext<const L1: usize, const L2: usize, const GA: usize, const GB: usize, S: Lanes>(
    g: &mut [[V; GB]; GA],
    g00: S,
    c00: S,
    b00: S,
    ab: f64,
) {
    let lt = L1 + L2 + 1;
    g00.store(&mut g[0][0]);
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
    for b in 1..=L2 + 1 {
        for a in 0..=(lt - b) {
            abv.mul_add(S::load8(&g[a][b - 1]), S::load8(&g[a + 1][b - 1]))
                .store(&mut g[a][b]);
        }
    }
}

/// One root: derivative contraction with the Cartesian pair density p into ga (bra centre)
/// and gb (ket centre), and (if `want_g`) the integrals into acc. Per direction d,
/// Q_d[a][b] = sum over the component pairs with direction-d exponents (a, b) of p times the
/// other two directions' 1D integrals, then contracted with the derivative 1D integrals
/// 2a I(a+1, b) - a I(a-1, b) (bra) and 2b I(a, b+1) - b I(a, b-1) (ket).
/// (Measured alternatives, slower: Q summed per group in registers through a const table of
/// component pairs (not unrolled: index arithmetic dominates); six register accumulators fed
/// by precomputed derivative 1D integrals (5-10% slower).)
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn add_root_deriv<
    const L1: usize,
    const L2: usize,
    const NC1: usize,
    const NC2: usize,
    const GA: usize,
    const GB: usize,
    S: Lanes,
>(
    acc: &mut [[V; NC2]; NC1],
    ga: &mut [S; 3],
    gb: &mut [S; 3],
    p: &[[V; NC2]; NC1],
    u: &V,
    w: &V,
    q: &PotPrim,
    pc: &[S; 3],
    ab: &[f64; 3],
    g: &mut [[[V; GB]; GA]; 3],
    qd: &mut [[[V; GB]; GA]; 3],
    want_g: bool,
) {
    let uv = S::load8(u);
    let b00 = (S::splat(1.0) - uv) * S::splat(q.half_inv_p);
    for d in 0..3 {
        let c00 = (-uv).mul_add(pc[d], S::splat(q.pa[d]));
        one_d_ext::<L1, L2, GA, GB, S>(
            &mut g[d],
            if d == 0 { S::load8(w) } else { S::splat(1.0) },
            c00,
            b00,
            ab[d],
        );
    }
    // (only the entries in use are cleared)
    for qdd in qd.iter_mut() {
        for row in qdd[..=L1].iter_mut() {
            for v in row[..=L2].iter_mut() {
                *v = [0.0; LN];
            }
        }
    }
    let c1 = &CART[L1];
    let c2 = &CART[L2];
    for i1 in 0..NC1 {
        let [ax, ay, az] = c1[i1];
        for i2 in 0..NC2 {
            let [bx, by, bz] = c2[i2];
            let gx = S::load8(&g[0][ax][bx]);
            let gy = S::load8(&g[1][ay][by]);
            let gz = S::load8(&g[2][az][bz]);
            let pv = S::load8(&p[i1][i2]);
            let xy = gx * gy;
            let px = pv * gx;
            if want_g {
                let a = &mut acc[i1][i2];
                xy.mul_add(gz, S::load8(a)).store(a);
            }
            let q2 = &mut qd[2][az][bz];
            pv.mul_add(xy, S::load8(q2)).store(q2);
            let q1 = &mut qd[1][ay][by];
            px.mul_add(gz, S::load8(q1)).store(q1);
            let q0 = &mut qd[0][ax][bx];
            (pv * gy).mul_add(gz, S::load8(q0)).store(q0);
        }
    }
    let a2 = S::splat(2.0 * q.a1);
    let b2 = S::splat(2.0 * (q.p - q.a1));
    for d in 0..3 {
        let gd = &g[d];
        let (mut sa, mut sb) = (ga[d], gb[d]);
        for a in 0..=L1 {
            for b in 0..=L2 {
                let qv = S::load8(&qd[d][a][b]);
                let mut va = a2 * S::load8(&gd[a + 1][b]);
                if a > 0 {
                    va = S::splat(-(a as f64)).mul_add(S::load8(&gd[a - 1][b]), va);
                }
                let mut vb = b2 * S::load8(&gd[a][b + 1]);
                if b > 0 {
                    vb = S::splat(-(b as f64)).mul_add(S::load8(&gd[a][b - 1]), vb);
                }
                sa = qv.mul_add(va, sa);
                sb = qv.mul_add(vb, sb);
            }
        }
        ga[d] = sa;
        gb[d] = sb;
    }
}

/// One primitive pair of the classes (s|s) and (p|s) (bra l = L1 <= 1, ket s) in closed form
/// (Obara-Saika) from the auxiliary integrals [s]^(m) = coef F_m(T) (erf part: kappa^(2m+1)
/// F_m(kappa^2 T)), m <= L1 + 1, instead of Rys quadrature with L1 + 1 roots:
///
///   (p_i|s)^(m) = PA_i [s]^(m) - PC_i [s]^(m+1),
///   d/dA_x (p_i|s) = 2a (d_ix|s) - delta_ix [s],   d/dB_x (p_i|s) = 2b ((d_ix|s) + AB_x (p_i|s)),
///   (d_ix|s) = PA_x (p_i|s)^(0) - PC_x (p_i|s)^(1) + delta_ix ([s]^(0) - [s]^(1)) / 2p,
///
/// contracted with the pair density P_i: W^(m) = sum_i P_i (p_i|s)^(m), S_x = sum_i P_i (d_ix|s)
/// = PA_x W^(0) - PC_x W^(1) + P_x ([s]^(0) - [s]^(1)) / 2p, ga_x = 2a S_x - P_x [s]^(0),
/// gb_x = 2b (S_x + AB_x W^(0)); (s|s): ga_x = 2a P (PA_x [s]^(0) - PC_x [s]^(1)), gb_x likewise
/// with b and PB = PA + AB.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn closed_low<const L1: usize, const NC1: usize, const NC2: usize, S: Lanes>(
    acc: &mut [[V; NC2]; NC1],
    ga: &mut [S; 3],
    gb: &mut [S; 3],
    p: &[[V; NC2]; NC1],
    q: &PotPrim,
    pc: &[S; 3],
    tv: S,
    ab: &[f64; 3],
    op: &Operator,
    om2: f64,
    want_g: bool,
) {
    let sm = aux_ss(L1 + 1, q, tv, op, om2);
    let a2 = 2.0 * q.a1;
    let b2 = 2.0 * (q.p - q.a1);
    if L1 == 0 {
        let pv = S::load8(&p[0][0]);
        let (pa2, pb2) = (pv * S::splat(a2), pv * S::splat(b2));
        for d in 0..3 {
            let t = pc[d] * sm[1];
            ga[d] = pa2.mul_add(S::splat(q.pa[d]).mul_add(sm[0], -t), ga[d]);
            gb[d] = pb2.mul_add(S::splat(q.pa[d] + ab[d]).mul_add(sm[0], -t), gb[d]);
        }
        if want_g {
            let a = &mut acc[0][0];
            (S::load8(a) + sm[0]).store(a);
        }
    } else {
        let pv = [S::load8(&p[0][0]), S::load8(&p[1][0]), S::load8(&p[2][0])];
        let ppa = pv[0].mul_add(
            S::splat(q.pa[0]),
            pv[1].mul_add(S::splat(q.pa[1]), pv[2] * S::splat(q.pa[2])),
        );
        let ppc = pv[0].mul_add(pc[0], pv[1].mul_add(pc[1], pv[2] * pc[2]));
        let w0 = sm[0].mul_add(ppa, -(sm[1] * ppc));
        let w1 = sm[1].mul_add(ppa, -(sm[2] * ppc));
        let h = (sm[0] - sm[1]) * S::splat(q.half_inv_p);
        for d in 0..3 {
            let sx = S::splat(q.pa[d]).mul_add(w0, pv[d].mul_add(h, -(pc[d] * w1)));
            ga[d] = S::splat(a2).mul_add(sx, ga[d] - pv[d] * sm[0]);
            gb[d] = S::splat(b2).mul_add(S::splat(ab[d]).mul_add(w0, sx), gb[d]);
        }
        if want_g {
            for i in 0..3 {
                let a = &mut acc[i][0];
                (S::load8(a) + S::splat(q.pa[i]).mul_add(sm[0], -(pc[i] * sm[1]))).store(a);
            }
        }
    }
}

/// One primitive pair of the classes (d|s) and (p|p) in closed form (Obara-Saika), from
/// [s]^(0..3), (p_i|s)^(0..2) and (d_ij|s)^(0, 1): the derivative integrals need (f|s), whose
/// contraction with the pair density follows from one more vertical step,
/// sum_e P_e (e + 1_x|s) = PA_x V^(0) - PC_x V^(1) + (U_x^(0) - U_x^(1)) / 2p with
/// V^(m) = sum_e P_e (e|s)^(m), U_x^(m) = sum_e P_e n_x(e) (e - 1_x|s)^(m).
///
/// (d|s): ga_x = 2a T_x - U_x^(0), gb_x = 2b (T_x + AB_x V^(0)), T_x the sum above over the d
/// components e.
/// (p|p) ((p_i|p_j) = (d_ij|s) + AB_j (p_i|s)): with e = i + j, Y_i = sum_j P_ij AB_j,
/// R_x = T_x + sum_i Y_i (d_(i+x)|s), Z = V^(0) + sum_i Y_i (p_i|s):
/// ga_x = 2a R_x - sum_j P_xj (p_j|s) - Y_x [s], gb_x = 2b (R_x + AB_x Z) - sum_i P_ix (p_i|s).
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn closed_l2<const L1: usize, const NC1: usize, const NC2: usize, S: Lanes>(
    acc: &mut [[V; NC2]; NC1],
    ga: &mut [S; 3],
    gb: &mut [S; 3],
    p: &[[V; NC2]; NC1],
    q: &PotPrim,
    pc: &[S; 3],
    tv: S,
    ab: &[f64; 3],
    op: &Operator,
    om2: f64,
    want_g: bool,
) {
    let sm = aux_ss(3, q, tv, op, om2);
    let pa = [S::splat(q.pa[0]), S::splat(q.pa[1]), S::splat(q.pa[2])];
    let hp = S::splat(q.half_inv_p);
    // (p_i|s)^(m), m = 0..2
    let pm: [[S; 3]; 3] = std::array::from_fn(|m| {
        std::array::from_fn(|i| pa[i].mul_add(sm[m], -(pc[i] * sm[m + 1])))
    });
    // (d_e|s)^(m), m = 0, 1
    let hm = [(sm[0] - sm[1]) * hp, (sm[1] - sm[2]) * hp];
    let mut dm = [[S::zero(); 6]; 2];
    for (m, dmm) in dm.iter_mut().enumerate() {
        for i in 0..3 {
            for j in i..3 {
                let mut v = pa[j].mul_add(pm[m][i], -(pc[j] * pm[m + 1][i]));
                if i == j {
                    v += hm[m];
                }
                dmm[D2[i][j]] = v;
            }
        }
    }
    let (a2, b2) = (S::splat(2.0 * q.a1), S::splat(2.0 * (q.p - q.a1)));
    if L1 == 2 {
        let pv: [S; 6] = std::array::from_fn(|e| S::load8(&p[e][0]));
        let v: [S; 2] =
            std::array::from_fn(|m| (0..6).fold(S::zero(), |a, e| pv[e].mul_add(dm[m][e], a)));
        for x in 0..3 {
            // U_x^(m) = sum_e P_e n_x(e) (p_(e-x)|s)^(m)
            let u: [S; 2] = std::array::from_fn(|m| {
                let mut t = S::zero();
                for j in 0..3 {
                    let f = if j == x { 2.0 } else { 1.0 };
                    t = (pv[D2[x][j]] * S::splat(f)).mul_add(pm[m][j], t);
                }
                t
            });
            let tx = pa[x].mul_add(v[0], (u[0] - u[1]).mul_add(hp, -(pc[x] * v[1])));
            ga[x] = a2.mul_add(tx, ga[x] - u[0]);
            gb[x] = b2.mul_add(S::splat(ab[x]).mul_add(v[0], tx), gb[x]);
        }
        if want_g {
            for e in 0..6 {
                let a = &mut acc[e][0];
                (S::load8(a) + dm[0][e]).store(a);
            }
        }
    } else {
        // (p|p): P_ij with i (bra), j (ket) in x, y, z
        let pv: [[S; 3]; 3] = std::array::from_fn(|i| std::array::from_fn(|j| S::load8(&p[i][j])));
        let abv = [S::splat(ab[0]), S::splat(ab[1]), S::splat(ab[2])];
        // sum over (i, j) of P_ij (d_(i+j)|s)^(m)
        let v: [S; 2] = std::array::from_fn(|m| {
            let mut t = S::zero();
            for i in 0..3 {
                for j in 0..3 {
                    t = pv[i][j].mul_add(dm[m][D2[i][j]], t);
                }
            }
            t
        });
        let y: [S; 3] = std::array::from_fn(|i| {
            pv[i][0].mul_add(abv[0], pv[i][1].mul_add(abv[1], pv[i][2] * abv[2]))
        });
        // sum_j P_xj p_j^(m) and sum_i P_ix p_i^(m)
        let rowp: [[S; 3]; 2] = std::array::from_fn(|m| {
            std::array::from_fn(|x| {
                pv[x][0].mul_add(pm[m][0], pv[x][1].mul_add(pm[m][1], pv[x][2] * pm[m][2]))
            })
        });
        let colp: [[S; 3]; 2] = std::array::from_fn(|m| {
            std::array::from_fn(|x| {
                pv[0][x].mul_add(pm[m][0], pv[1][x].mul_add(pm[m][1], pv[2][x] * pm[m][2]))
            })
        });
        let z = y[0].mul_add(
            pm[0][0],
            y[1].mul_add(pm[0][1], y[2].mul_add(pm[0][2], v[0])),
        );
        for x in 0..3 {
            // U_x^(m) = sum_j P_xj p_j + sum_i P_ix p_i
            let u = [rowp[0][x] + colp[0][x], rowp[1][x] + colp[1][x]];
            let tx = pa[x].mul_add(v[0], (u[0] - u[1]).mul_add(hp, -(pc[x] * v[1])));
            let rx = y[0].mul_add(
                dm[0][D2[0][x]],
                y[1].mul_add(dm[0][D2[1][x]], y[2].mul_add(dm[0][D2[2][x]], tx)),
            );
            ga[x] = a2.mul_add(rx, ga[x] - rowp[0][x] - y[x] * sm[0]);
            gb[x] = b2.mul_add(abv[x].mul_add(z, rx), gb[x] - colp[0][x]);
        }
        if want_g {
            for i in 0..3 {
                for j in 0..3 {
                    let a = &mut acc[i][j];
                    (S::load8(a) + abv[j].mul_add(pm[0][i], dm[0][D2[i][j]])).store(a);
                }
            }
        }
    }
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::needless_range_loop, reason = "indexes several arrays in step")]
fn dkernel<
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
    c: &mut PotDerivContract,
) -> [[f64; 3]; 2] {
    let np = x.len();
    let ab = pair.ab;
    let mixed = op.has_erf();
    let om2 = op.omega() * op.omega();
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
    let want_g = c.ng > 0 && (d1 || d2);
    let ld = c.ld;
    let ds = c.dstride;
    // in-place row access below: rows of both shells within every block (blocks and the
    // chunk range were checked by contract_deriv: nblk, ng <= blocks of f and g, ld >= np)
    assert!(rows_in_bounds(t1, r1, NC1, ds, ld) && rows_in_bounds(t2, r2, NC2, ds, ld));
    let mut g = [[[[0.0; LN]; GB]; GA]; 3];
    let mut qd = [[[[0.0; LN]; GB]; GA]; 3];
    let mut tot = [[S::zero(); 3]; 2];
    let mut p = [[[0.0; LN]; NC2]; NC1];
    let mut acc = [[[0.0; LN]; NC2]; NC1];
    let mut g0 = 0;
    while g0 < np {
        let nl = LN.min(np - g0);
        let mut cc = [[0.0; LN]; 3];
        if nl == LN {
            cc[0] = x[g0..g0 + LN].try_into().unwrap();
            cc[1] = y[g0..g0 + LN].try_into().unwrap();
            cc[2] = z[g0..g0 + LN].try_into().unwrap();
        } else {
            for l in 0..LN {
                let gi = g0 + l.min(nl - 1);
                cc[0][l] = x[gi];
                cc[1][l] = y[gi];
                cc[2][l] = z[gi];
            }
        }
        let cv = [S::load8(&cc[0]), S::load8(&cc[1]), S::load8(&cc[2])];
        // pair density in the Cartesian components, times the weights (folded into the bra
        // factors; the first term stores)
        let ws = if nl == LN {
            S::load(&c.w[g0..g0 + LN])
        } else {
            let mut wl = [0.0; LN];
            wl[..nl].copy_from_slice(&c.w[g0..g0 + nl]);
            S::load8(&wl)
        };
        for (it, &(bi, bj, cf)) in c.terms.iter().enumerate() {
            // SAFETY: rows checked above; bi, bj < nblk; g0 + LN <= ld
            let (fi1, fj2) = unsafe {
                (
                    cart_rows_v::<NC1, S>(t1, c.f, bi * ds + g0, r1, ld),
                    cart_rows_v::<NC2, S>(t2, c.f, bj * ds + g0, r2, ld),
                )
            };
            // (explicit lane vectors: with plain [f64; 8] loops LLVM vectorizes the component
            // loop instead, with gathers and scatters)
            if bi == bj {
                let cs = S::splat(2.0 * cf) * ws;
                for i1 in 0..NC1 {
                    let a = cs * fi1[i1];
                    for i2 in 0..NC2 {
                        let pv = &mut p[i1][i2];
                        if it == 0 {
                            (a * fj2[i2]).store(pv);
                        } else {
                            a.mul_add(fj2[i2], S::load8(pv)).store(pv);
                        }
                    }
                }
            } else {
                // SAFETY: as above
                let (fj1, fi2) = unsafe {
                    (
                        cart_rows_v::<NC1, S>(t1, c.f, bj * ds + g0, r1, ld),
                        cart_rows_v::<NC2, S>(t2, c.f, bi * ds + g0, r2, ld),
                    )
                };
                let cs = S::splat(cf) * ws;
                for i1 in 0..NC1 {
                    let a = cs * fi1[i1];
                    let b = cs * fj1[i1];
                    for i2 in 0..NC2 {
                        let pv = &mut p[i1][i2];
                        if it == 0 {
                            a.mul_add(fj2[i2], b * fi2[i2]).store(pv);
                        } else {
                            a.mul_add(fj2[i2], b.mul_add(fi2[i2], S::load8(pv)))
                                .store(pv);
                        }
                    }
                }
            }
        }
        if c.terms.is_empty() {
            for r in p.iter_mut() {
                for v in r.iter_mut() {
                    *v = [0.0; LN];
                }
            }
        }
        if want_g {
            for r in acc.iter_mut() {
                for v in r.iter_mut() {
                    *v = [0.0; LN];
                }
            }
        }
        let mut ga = [S::zero(); 3];
        let mut gb = [S::zero(); 3];
        for q in prims {
            let pc = [
                S::splat(q.pc[0]) - cv[0],
                S::splat(q.pc[1]) - cv[1],
                S::splat(q.pc[2]) - cv[2],
            ];
            let tv = S::splat(q.p) * pc[0].mul_add(pc[0], pc[1].mul_add(pc[1], pc[2] * pc[2]));
            if L1 + L2 <= 1 {
                // (s|s) and (p|s): closed forms from the Boys functions (Obara-Saika)
                closed_low::<L1, NC1, NC2, S>(
                    &mut acc, &mut ga, &mut gb, &p, q, &pc, tv, &ab, op, om2, want_g,
                );
                continue;
            }
            if L1 + L2 == 2 {
                // (p|p) and (d|s) likewise
                closed_l2::<L1, NC1, NC2, S>(
                    &mut acc, &mut ga, &mut gb, &p, q, &pc, tv, &ab, op, om2, want_g,
                );
                continue;
            }
            let mut u = [[0.0; LN]; NR];
            let mut w = [[0.0; LN]; NR];
            let cf = q.coef * op.coulomb();
            if cf != 0.0 {
                roots_vec::<NR, W, S>(tv, cf, &mut u, &mut w);
                for i in 0..NR {
                    add_root_deriv::<L1, L2, NC1, NC2, GA, GB, S>(
                        &mut acc, &mut ga, &mut gb, &p, &u[i], &w[i], q, &pc, &ab, &mut g, &mut qd,
                        want_g,
                    );
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
                    add_root_deriv::<L1, L2, NC1, NC2, GA, GB, S>(
                        &mut acc, &mut ga, &mut gb, &p, &u[i], &w[i], q, &pc, &ab, &mut g, &mut qd,
                        want_g,
                    );
                }
            }
        }
        // (padded lanes carry zero weight, so whole vectors can be added)
        for d in 0..3 {
            tot[0][d] += ga[d];
            tot[1][d] += gb[d];
            let gp = &mut c.gpt[d * ld + g0..d * ld + g0 + LN];
            (S::load(gp) - (ga[d] + gb[d])).store(gp);
        }
        if want_g {
            for k in 0..c.ng {
                let base = k * ds + g0;
                if d1 {
                    // SAFETY: rows checked above; k < ng; g0 + LN <= ld
                    let fc = unsafe { cart_rows_v::<NC2, S>(t2, c.f, base, r2, ld) };
                    let gc: [S; NC1] = std::array::from_fn(|i1| {
                        let mut v = S::zero();
                        for i2 in 0..NC2 {
                            v = S::load8(&acc[i1][i2]).mul_add(fc[i2], v);
                        }
                        v
                    });
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
        g0 += LN;
    }
    let tot = tot.map(|t| t.map(|v| v.hsum()));
    if pair.swap {
        [tot[1], tot[0]]
    } else {
        tot
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::basis::{BasisDef, BasisSet};
    use crate::pairs::{PairOptions, PairSet};
    use crate::points::grid_pot::PotContract;

    /// Shells s..g on two centres (contracted s, p, d; uncontracted f, g), spherical or Cartesian.
    fn test_basis(pure: bool, xyz: &[[f64; 3]]) -> BasisSet {
        let json = r#"{"name":"t","elements":{"8":{"electron_shells":[
          {"function_type":"gto","angular_momentum":[0],"exponents":["130.7","23.81","6.44"],"coefficients":[["0.154","0.535","0.444"]]},
          {"function_type":"gto","angular_momentum":[1],"exponents":["5.03","1.17","0.38"],"coefficients":[["0.156","0.607","0.392"]]},
          {"function_type":"gto","angular_momentum":[2],"exponents":["1.2","0.4"],"coefficients":[["0.6","0.5"]]},
          {"function_type":"gto","angular_momentum":[3],"exponents":["0.9"],"coefficients":[["1.0"]]},
          {"function_type":"gto","angular_momentum":[4],"exponents":["0.7"],"coefficients":[["1.0"]]}]},
          "1":{"electron_shells":[
          {"function_type":"gto","angular_momentum":[0],"exponents":["3.42","0.62"],"coefficients":[["0.15","0.53"]]},
          {"function_type":"gto","angular_momentum":[1],"exponents":["0.8"],"coefficients":[["1.0"]]},
          {"function_type":"gto","angular_momentum":[2],"exponents":["0.6"],"coefficients":[["1.0"]]},
          {"function_type":"gto","angular_momentum":[3],"exponents":["0.5"],"coefficients":[["1.0"]]},
          {"function_type":"gto","angular_momentum":[4],"exponents":["0.45"],"coefficients":[["1.0"]]}]}}}"#;
        let def = BasisDef::from_bse_json(json).unwrap();
        BasisSet::new(&def, &[8, 1], xyz, pure).unwrap()
    }

    const XYZ: [[f64; 3]; 2] = [[0.1, -0.2, 0.3], [1.2, 0.9, -0.8]];

    fn points() -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        // near and far from both centres, not on a centre (the integrals are not differentiable
        // in the point position there)
        let p = [
            [0.35, -0.05, 0.42],
            [0.9, 0.4, -0.3],
            [-1.1, 0.7, 1.5],
            [2.5, -1.9, 0.4],
            [0.6, 0.6, -0.1],
            [-4.0, 3.0, 6.0],
            [1.25, 0.95, -0.7],
        ];
        (
            p.iter().map(|q| q[0]).collect(),
            p.iter().map(|q| q[1]).collect(),
            p.iter().map(|q| q[2]).collect(),
        )
    }

    /// Integrals of shell pair (sa, sb) at the points for centres `xyz`: [(ia nb + ib) np + g].
    #[allow(clippy::too_many_arguments, reason = "a kernel's arrays and sizes")]
    fn ints(
        pure: bool,
        xyz: &[[f64; 3]],
        sa: usize,
        sb: usize,
        op: &Operator,
        x: &[f64],
        y: &[f64],
        z: &[f64],
    ) -> Vec<f64> {
        let basis = test_basis(pure, xyz);
        let pp = PotPair::new(&basis, sa, sb, 0.0);
        let mut out = vec![0.0; pp.nab() * x.len()];
        pp.compute(op, x, y, z, 0.0, &mut out);
        out
    }

    /// Sixth-order central difference (Richardson of two fourth-order stencils).
    fn fd(f: impl Fn(f64) -> Vec<f64>, h: f64) -> Vec<f64> {
        let d4 = |h: f64| -> Vec<f64> {
            let (p1, m1, p2, m2) = (f(h), f(-h), f(2.0 * h), f(-2.0 * h));
            (0..p1.len())
                .map(|i| (8.0 * (p1[i] - m1[i]) - (p2[i] - m2[i])) / (12.0 * h))
                .collect()
        };
        let (a, b) = (d4(h), d4(0.5 * h));
        (0..a.len()).map(|i| (16.0 * b[i] - a[i]) / 15.0).collect()
    }

    /// Each derivative integral dA_ab(g)/dX (X: either centre, or the point) of every shell
    /// pair s..g (spherical and Cartesian; Coulomb and erf-attenuated) against finite differences.
    #[test]
    fn derivative_integrals_match_finite_differences() {
        let (x, y, z) = points();
        let np = x.len();
        let ld = 8;
        for pure in [true, false] {
            for op in [Operator::Coulomb, Operator::from_libxc(0.19, 0.46, 0.33)] {
                let basis = test_basis(pure, &XYZ);
                let nbf = basis.nbf;
                let pairs = PairSet::build(&basis, PairOptions::default());
                let mut worst: f64 = 0.0;
                let mut classes = std::collections::BTreeSet::new();
                for pr in pairs.pairs() {
                    let (sa, sb) = (pr.sa, pr.sb);
                    let (aa, ab_) = (basis.shells[sa].atom, basis.shells[sb].atom);
                    let pp = PotPair::new(&basis, sa, sb, 0.0);
                    classes.insert((pp.la.max(pp.lb), pp.la.min(pp.lb)));
                    let (na, nb) = (pp.na, pp.nb);
                    let (a0, b0) = (basis.shells[sa].ao_start, basis.shells[sb].ao_start);
                    // finite differences of all integrals of the pair: [what][dir] (what: atom of
                    // sa, atom of sb (if different), the points)
                    let mut fds: Vec<[Vec<f64>; 3]> = vec![];
                    for what in 0..3 {
                        if what == 1 && aa == ab_ {
                            fds.push([vec![], vec![], vec![]]);
                            continue;
                        }
                        let one = |d: usize| {
                            fd(
                                |h: f64| {
                                    let mut c2 = XYZ;
                                    let (mut px, mut py, mut pz) =
                                        (x.clone(), y.clone(), z.clone());
                                    match what {
                                        0 => c2[aa][d] += h,
                                        1 => c2[ab_][d] += h,
                                        _ => [&mut px, &mut py, &mut pz][d]
                                            .iter_mut()
                                            .for_each(|v| *v += h),
                                    }
                                    ints(pure, &c2, sa, sb, &op, &px, &py, &pz)
                                },
                                1e-3,
                            )
                        };
                        fds.push([one(0), one(1), one(2)]);
                    }
                    // analytic: one point at a time, unit F on function a (block 0) and b (block 1)
                    for ia in 0..na {
                        for ib in 0..nb {
                            for gi in 0..np {
                                let mut f = vec![0.0; 2 * nbf * ld];
                                f[(a0 + ia) * ld] = 1.0;
                                f[nbf * ld + (b0 + ib) * ld] = 1.0;
                                let mut gbuf = vec![0.0; 2 * nbf * ld];
                                let mut gpt = vec![0.0; 3 * ld];
                                // P = F0_a F1_b + F1_a F0_b: 1 on (a, b) for different shells; on
                                // one shell both (ia, ib) and (ib, ia) (or (ia, ia) twice) get 1
                                let terms = [(0usize, 1usize, 1.0)];
                                let mut c = PotDerivContract {
                                    f: &f,
                                    g: &mut gbuf,
                                    ng: 0,
                                    dstride: nbf * ld,
                                    ld,
                                    a0,
                                    b0,
                                    do_a: false,
                                    do_b: false,
                                    terms: &terms,
                                    w: &[1.0],
                                    gpt: &mut gpt,
                                    cart: false,
                                };
                                let gr = pp.contract_deriv(
                                    &pp.prims,
                                    &op,
                                    &x[gi..gi + 1],
                                    &y[gi..gi + 1],
                                    &z[gi..gi + 1],
                                    &mut c,
                                );
                                let sym = if sa == sb { 2.0 } else { 1.0 };
                                let idx = (ia * nb + ib) * np + gi;
                                for d in 0..3 {
                                    let ga = gr[0][d] / sym;
                                    let gb = gr[1][d] / sym;
                                    let mut errs = vec![];
                                    if aa == ab_ {
                                        errs.push((ga + gb, fds[0][d][idx]));
                                    } else {
                                        errs.push((ga, fds[0][d][idx]));
                                        errs.push((gb, fds[1][d][idx]));
                                    }
                                    errs.push((gpt[d * ld] / sym, fds[2][d][idx]));
                                    for (g, w) in errs {
                                        let e = (g - w).abs() / w.abs().max(1.0);
                                        assert!(e < 1e-9, "pure={pure} op={op:?} shells ({sa},{sb}) l=({},{}) fn ({ia},{ib}) point {gi} dir {d}: {g} vs {w}", pp.la, pp.lb);
                                        worst = worst.max(e);
                                    }
                                }
                            }
                        }
                    }
                }
                assert!(
                    classes.contains(&(4, 4))
                        && classes.contains(&(4, 0))
                        && classes.contains(&(3, 2))
                );
                eprintln!(
                    "pure={pure} coulomb={}: {} shell classes, worst relative error {worst:.2e}",
                    !op.has_erf(),
                    classes.len()
                );
            }
        }
    }

    /// The G accumulation of the derivative kernel equals the energy kernel's contraction, and
    /// the derivative contraction with several blocks and weights equals the sum over its terms.
    #[test]
    fn contraction_matches_reference() {
        let basis = test_basis(true, &XYZ);
        let pairs = PairSet::build(&basis, PairOptions::default());
        let (x, y, z) = points();
        let np = x.len();
        let ld = 8;
        let nbf = basis.nbf;
        let nblk = 3;
        let ds = nbf * ld;
        let mut f = vec![0.0; nblk * ds];
        for k in 0..nblk {
            for m in 0..nbf {
                for g in 0..np {
                    f[k * ds + m * ld + g] = ((k * 7 + m * 3 + g) as f64 * 0.37).sin();
                }
            }
        }
        let w: Vec<f64> = (0..np).map(|g| 0.3 + 0.1 * g as f64).collect();
        let op = Operator::Coulomb;
        let mut worst: f64 = 0.0;
        for pr in pairs.pairs() {
            let pp = PotPair::new(&basis, pr.sa, pr.sb, 0.0);
            let (a0, b0) = (basis.shells[pr.sa].ao_start, basis.shells[pr.sb].ao_start);
            let do_b = pr.sa != pr.sb;
            let mut want = vec![0.0; nblk * ds];
            let mut c = PotContract {
                f: &f,
                g: &mut want,
                nd: nblk,
                dstride: ds,
                ld,
                a0,
                b0,
                do_a: true,
                do_b,
                cart: false,
            };
            pp.contract(&op, &x, &y, &z, 0.0, &mut c);
            let terms = [(0usize, 2usize, 0.7), (1, 1, -0.4)];
            let mut got = vec![0.0; nblk * ds];
            let mut gpt = vec![0.0; 3 * ld];
            let mut c = PotDerivContract {
                f: &f,
                g: &mut got,
                ng: nblk,
                dstride: ds,
                ld,
                a0,
                b0,
                do_a: true,
                do_b,
                terms: &terms,
                w: &w,
                gpt: &mut gpt,
                cart: false,
            };
            let gr = pp.contract_deriv(&pp.prims, &op, &x, &y, &z, &mut c);
            for (u, v) in got.iter().zip(&want) {
                worst = worst.max((u - v).abs() / v.abs().max(1.0));
            }
            // one term at a time
            let mut sum = [[0.0; 3]; 2];
            let mut gsum = vec![0.0; 3 * ld];
            for t in &terms {
                let mut gb = vec![0.0; nblk * ds];
                let mut gp = vec![0.0; 3 * ld];
                let mut c = PotDerivContract {
                    f: &f,
                    g: &mut gb,
                    ng: 0,
                    dstride: ds,
                    ld,
                    a0,
                    b0,
                    do_a: false,
                    do_b: false,
                    terms: std::slice::from_ref(t),
                    w: &w,
                    gpt: &mut gp,
                    cart: false,
                };
                let r = pp.contract_deriv(&pp.prims, &op, &x, &y, &z, &mut c);
                for s in 0..2 {
                    for d in 0..3 {
                        sum[s][d] += r[s][d];
                    }
                }
                gsum.iter_mut().zip(&gp).for_each(|(a, b)| *a += b);
            }
            for s in 0..2 {
                for d in 0..3 {
                    worst = worst.max((gr[s][d] - sum[s][d]).abs() / sum[s][d].abs().max(1.0));
                }
            }
            for (a, b) in gpt.iter().zip(&gsum) {
                worst = worst.max((a - b).abs() / b.abs().max(1.0));
            }
            // translational invariance: centres + points = 0
            for d in 0..3 {
                let s: f64 = gr[0][d] + gr[1][d] + gpt[d * ld..d * ld + np].iter().sum::<f64>();
                worst = worst.max(s.abs() / gr[0][d].abs().max(1.0));
            }
        }
        assert!(worst < 1e-12, "worst {worst:e}");
    }
}
