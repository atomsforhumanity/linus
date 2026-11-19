//! The one primitive loop of shell-pair data: for every retained primitive pair of two shells,
//! the Hermite expansion E of the product and, up to the requested derivative order, the
//! expansions E^A and E^AA of its derivatives with respect to the first shell's center
//! ([`crate::core::deriv`] has the formulas). Every pair type of the crate is built here, so
//! the primitive screening, the Cartesian fill and the transformation to the shells' functions
//! exist once.
//!
//! The 1D tables of one primitive pair are computed once per direction, up to l_a + order in
//! the first index: E^{ij}_t for i <= l_a does not depend on how far the recursion runs, so E
//! has the same bits at every order, and so has E^A at orders 1 and 2.

use crate::basis::{cart_components, ncart};
use crate::core::deriv::{PairDeriv, D2_COMPS};
use crate::core::hermite::{herm_idx, hermite_e_1d, nherm, ShellPair, ShellView};
use crate::core::util::PairTransform;

/// Primitive-pair screening: drop pairs with exp(-mu R^2) below this.
pub const PRIM_PAIR_CUT: f64 = 1e-22;

/// The derivative multi-indices of the first derivatives (x, y, z) and, in [`D2_COMPS`]
/// order, of the second derivatives with respect to the first shell's center.
const COMPS1: [[usize; 3]; 3] = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
const COMPS2: [[usize; 3]; 6] = {
    let mut c = [[0; 3]; 6];
    let mut i = 0;
    while i < 6 {
        let (k, l) = D2_COMPS[i];
        c[i][k] += 1;
        c[i][l] += 1;
        i += 1;
    }
    c
};

/// The 1D derivative tables of one primitive pair in one direction from its 1D expansion `e1`
/// (`hermite_e_1d` up to l_a + order): tab[m][(i * (lb+1) + j) * nt + t] = d^m/dA^m of
/// E^{ij}_t, m = 1..=order, i <= la (nt = la + order + lb + 1).
#[allow(clippy::needless_range_loop, reason = "indexes several arrays in step")]
fn deriv_tables_1d(
    la: usize,
    lb: usize,
    order: usize,
    a: f64,
    e1: &[f64],
    tab: &mut [Vec<f64>; 3],
) {
    let lx = la + order;
    let nt = lx + lb + 1;
    let si = (lb + 1) * nt;
    let e = |i: usize, j: usize, t: usize| e1[i * si + j * nt + t];
    for m in 0..=order {
        tab[m][..(la + 1) * si].fill(0.0);
    }
    for i in 0..=la {
        for j in 0..=lb {
            let o = i * si + j * nt;
            for t in 0..nt {
                tab[0][o + t] = e(i, j, t);
                let mut v = 2.0 * a * e(i + 1, j, t);
                if i >= 1 {
                    v -= i as f64 * e(i - 1, j, t);
                }
                tab[1][o + t] = v;
                if order >= 2 {
                    let mut v =
                        4.0 * a * a * e(i + 2, j, t) - 2.0 * a * (2 * i + 1) as f64 * e(i, j, t);
                    if i >= 2 {
                        v += (i * (i - 1)) as f64 * e(i - 2, j, t);
                    }
                    tab[2][o + t] = v;
                }
            }
        }
    }
}

/// The pair data of shells `a` (index `sa`) and `b` (`sb`): E and, for `order` 1 or 2, the
/// derivative expansions of the same retained primitive pairs, in the same order. Primitive
/// pairs with exp(-mu |AB|^2) < `prim_cut` are dropped.
#[allow(clippy::needless_range_loop, reason = "indexes several arrays in step")]
pub fn build(
    a: &ShellView,
    b: &ShellView,
    sa: usize,
    sb: usize,
    order: usize,
    prim_cut: f64,
) -> (ShellPair, Option<PairDeriv>) {
    assert!(order <= 2, "pair data up to second derivatives");
    let (la, lb) = (a.l, b.l);
    let lab = la + lb;
    let nh = nherm(lab);
    let (nca, ncb) = (ncart(la), ncart(lb));
    let (na, nb) = (a.nfunc, b.nfunc);
    let ca = cart_components(la);
    let cb = cart_components(lb);
    let ab = [
        a.center[0] - b.center[0],
        a.center[1] - b.center[1],
        a.center[2] - b.center[2],
    ];
    let r2 = ab[0] * ab[0] + ab[1] * ab[1] + ab[2] * ab[2];
    // 1D tables up to la + order in the first index
    let lx = la + order;
    let nt = lx + lb + 1;
    let e1len = (lx + 1) * (lb + 1) * nt;
    let mut e1: [Vec<f64>; 3] = std::array::from_fn(|_| vec![0.0; e1len]);
    let mut ecart = vec![0.0; nh * nca * ncb];
    let tf = PairTransform::of_views(a, b);
    let mut tmp = Vec::new();
    let mut pp = Vec::new();
    let mut pc = Vec::new();
    let mut e = Vec::new();
    let mut emax: f64 = 0.0;
    // derivative scratch (order >= 1)
    let dlen = (la + 1) * (lb + 1) * nt;
    let mut tabs: [[Vec<f64>; 3]; 3] = Default::default();
    let (nh1, nh2) = (nherm(lab + 1), nherm(lab + 2));
    let (mut c1, mut c2) = (Vec::new(), Vec::new());
    if order >= 1 {
        for t in tabs.iter_mut() {
            for m in t.iter_mut() {
                *m = vec![0.0; dlen];
            }
        }
        c1 = vec![0.0; nh1 * 3 * nca * ncb];
    }
    if order >= 2 {
        c2 = vec![0.0; nh2 * 6 * nca * ncb];
    }
    let (mut ea, mut eaa) = (Vec::new(), Vec::new());
    for (&ai, &ci) in a.exps.iter().zip(a.coefs) {
        for (&bj, &cj) in b.exps.iter().zip(b.coefs) {
            let p = ai + bj;
            let mu = ai * bj / p;
            if (-mu * r2).exp() < prim_cut {
                continue;
            }
            for d in 0..3 {
                hermite_e_1d(lx, lb, ai, bj, ab[d], &mut e1[d]);
            }
            let [ex, ey, ez] = &e1;
            let cc = ci * cj;
            // Cartesian E: ecart[r][ia][ib]
            ecart.fill(0.0);
            for (ia, pa) in ca.iter().enumerate() {
                for (ib, pb) in cb.iter().enumerate() {
                    let ox = (pa[0] * (lb + 1) + pb[0]) * nt;
                    let oy = (pa[1] * (lb + 1) + pb[1]) * nt;
                    let oz = (pa[2] * (lb + 1) + pb[2]) * nt;
                    for t in 0..=pa[0] + pb[0] {
                        let vx = ex[ox + t] * cc;
                        for u in 0..=pa[1] + pb[1] {
                            let vxy = vx * ey[oy + u];
                            for v in 0..=pa[2] + pb[2] {
                                let r = herm_idx(t, u, v);
                                ecart[(r * nca + ia) * ncb + ib] = vxy * ez[oz + v];
                            }
                        }
                    }
                }
            }
            // E[r][mu][nu] = sum Ta[mu][ia] ecart[r][ia][ib] Tb[nu][ib]
            let base = e.len();
            e.resize(base + nh * na * nb, 0.0);
            tf.rows(&ecart, nh, &mut e[base..], &mut tmp);
            emax = e[base..].iter().fold(emax, |m, &s| m.max(s.abs()));
            pp.push(p);
            pc.push([
                (ai * a.center[0] + bj * b.center[0]) / p,
                (ai * a.center[1] + bj * b.center[1]) / p,
                (ai * a.center[2] + bj * b.center[2]) / p,
            ]);
            if order == 0 {
                continue;
            }
            for d in 0..3 {
                deriv_tables_1d(la, lb, order, ai, &e1[d], &mut tabs[d]);
            }
            let fill = |comps: &[[usize; 3]], cart: &mut Vec<f64>| {
                let ncomp = comps.len();
                cart.fill(0.0);
                for (ic, m) in comps.iter().enumerate() {
                    for (ia, pa) in ca.iter().enumerate() {
                        for (ib, pb) in cb.iter().enumerate() {
                            let ox = (pa[0] * (lb + 1) + pb[0]) * nt;
                            let oy = (pa[1] * (lb + 1) + pb[1]) * nt;
                            let oz = (pa[2] * (lb + 1) + pb[2]) * nt;
                            let (tx, ty, tz) = (&tabs[0][m[0]], &tabs[1][m[1]], &tabs[2][m[2]]);
                            for t in 0..=pa[0] + pb[0] + m[0] {
                                let vx = tx[ox + t] * cc;
                                if vx == 0.0 {
                                    continue;
                                }
                                for u in 0..=pa[1] + pb[1] + m[1] {
                                    let vxy = vx * ty[oy + u];
                                    if vxy == 0.0 {
                                        continue;
                                    }
                                    for v in 0..=pa[2] + pb[2] + m[2] {
                                        let r = herm_idx(t, u, v);
                                        cart[((r * ncomp + ic) * nca + ia) * ncb + ib] +=
                                            vxy * tz[oz + v];
                                    }
                                }
                            }
                        }
                    }
                }
            };
            fill(&COMPS1, &mut c1);
            let base = ea.len();
            ea.resize(base + nh1 * 3 * na * nb, 0.0);
            tf.rows(&c1, nh1 * 3, &mut ea[base..], &mut tmp);
            if order >= 2 {
                fill(&COMPS2, &mut c2);
                let base = eaa.len();
                eaa.resize(base + nh2 * 6 * na * nb, 0.0);
                tf.rows(&c2, nh2 * 6, &mut eaa[base..], &mut tmp);
            }
        }
    }
    let nprim = pp.len();
    let pair = ShellPair {
        sa,
        sb,
        la,
        lb,
        na,
        nb,
        nprim,
        p: pp,
        pc,
        e,
        emax,
    };
    if order == 0 {
        return (pair, None);
    }
    // E^A with the component index first: per primitive [k][r][mu][nu]
    let nab = na * nb;
    let mut ea_t = vec![0.0; ea.len()];
    for ip in 0..nprim {
        let o = ip * nh1 * 3 * nab;
        for r in 0..nh1 {
            for k in 0..3 {
                ea_t[o + (k * nh1 + r) * nab..o + (k * nh1 + r + 1) * nab]
                    .copy_from_slice(&ea[o + (r * 3 + k) * nab..o + (r * 3 + k + 1) * nab]);
            }
        }
    }
    let deriv = PairDeriv {
        order,
        nprim,
        la,
        lb,
        na,
        nb,
        ea,
        eaa,
        ea_t,
    };
    (pair, Some(deriv))
}
