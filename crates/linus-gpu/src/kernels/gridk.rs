//! The potential-integral kernels of grid-based exchange ([`crate::gridk`]): for one class of
//! shell pairs (bra l `la` >= ket l `lb`), every point g of a chunk and every pair of the class,
//!
//!   V_ab(g) = <a| O(|r - r_g|) |b> = sum_prim coef sum_tuv Ex^{ab}_t Ey^{ab}_u Ez^{ab}_v R_tuv(P - r_g)
//!
//! (McMurchie-Davidson: Hermite expansions E per dimension from the recurrences in PA and PB,
//! the Hermite integrals R from the Boys function of the operator, coef = (2 pi / p) K_ab c_a c_b),
//! contracted on the fly with the Cartesian rows of Y:
//!
//!   Z\[a\](g) += w_g sum_b V_ab(g) Y\[b\](g),   Z\[b\](g) += w_g sum_a V_ab(g) Y\[a\](g) (off the diagonal).
//!
//! One block per chunk of at most [`THREADS`] points and slice of the class's pairs (blockIdx.y:
//! a contiguous range of the pair list, each slice writing its own copy of Z, `zstride` apart,
//! summed in slice order afterwards: a small molecule's points alone fill few blocks), one thread
//! per point, the slice's pairs in a fixed order (each pair skipped by the whole block when its
//! bound, the pair's estimate times the chunk's max |Y| over the two shells times the chunk's
//! max_p sum_g w |Xbar_gp|, is below the threshold): every Z element has one writer per slice, in
//! a fixed order, so the result is bitwise reproducible. The operator is c / r + l erf(omega r) / r (kernel arguments): the erf part's
//! Hermite integrals use F_n(kappa^2 T) kappa^(2n+1), kappa^2 = omega^2 / (omega^2 + p).
//!
//! The source is generated per class: the Boys values, the R recurrence, the E coefficients and
//! the contraction as straight-line code (the Cartesian components in linus-ints' order, each
//! normalized as x^l, so the spherical transform is the host's).

use super::boys::{BOYS_COLS, BOYS_TMAX};
use crate::cpu::{cart_components, ncart};
use std::fmt::Write;

/// Points per chunk (threads per block).
pub const THREADS: u32 = 128;
/// Doubles per primitive pair: p, coef, P (3), PA (3), PB (3), the magnitude estimate of its
/// integrals (linus-ints' `PotPrim::est`).
pub const PRIM: usize = 12;
/// Integers per shell pair: bra shell, ket shell, bra and ket first Cartesian rows, first and
/// last primitive.
pub const PAIR: usize = 6;
/// Highest orbital l (and la + lb <= 2 L_MAX).
pub const L_MAX: usize = 4;

pub fn kernel_name(la: usize, lb: usize) -> String {
    format!("gridk_v_{la}{lb}")
}

/// Every (t, u, v) with t + u + v <= l.
fn tuvs(l: usize) -> Vec<(usize, usize, usize)> {
    let mut v = Vec::new();
    for n in 0..=l {
        for t in (0..=n).rev() {
            for u in (0..=n - t).rev() {
                v.push((t, u, n - t - u));
            }
        }
    }
    v
}

/// Groups of the bra's Cartesian components with an R recurrence each ([`generate`]): one up to
/// l = la + lb = 4, two for 5 and 6, three beyond.
pub fn groups(l: usize) -> usize {
    match l {
        0..=4 => 1,
        5..=6 => 2,
        _ => 3,
    }
}

/// Hermite coefficients of one dimension of a class: for every i <= la, j <= lb the values
/// E^{ij}_t, t = 0..=i+j.
pub fn ne1(la: usize, lb: usize) -> usize {
    (0..=la)
        .flat_map(|i| (0..=lb).map(move |j| i + j + 1))
        .sum()
}

/// Offset of E^{ij}_0 within a dimension's block.
fn eoff(la: usize, lb: usize, i: usize, j: usize) -> usize {
    let mut o = 0;
    for ii in 0..=la {
        for jj in 0..=lb {
            if (ii, jj) == (i, j) {
                return o;
            }
            o += ii + jj + 1;
        }
    }
    unreachable!("({i}, {j}) beyond ({la}, {lb})")
}

/// The Hermite expansion coefficients of a primitive pair with exponent sum `p`, P - A `pa` and
/// P - B `pb` (without the prefactor K_ab, which the pair's coef carries): x, y and z blocks of
/// [`ne1`] values each (McMurchie-Davidson's recurrences in i, then j). Computed once per
/// geometry, read by every point.
#[allow(clippy::needless_range_loop)] // the recurrences index e[i][j] and e[i][j + 1]
pub fn hermite_e(la: usize, lb: usize, pa: [f64; 3], pb: [f64; 3], p: f64) -> Vec<f64> {
    let n1 = ne1(la, lb);
    let h = 0.5 / p;
    let mut out = vec![0.0; 3 * n1];
    for d in 0..3 {
        // e[i][j][t]
        let mut e = vec![vec![vec![0.0; la + lb + 2]; lb + 1]; la + 1];
        e[0][0][0] = 1.0;
        for i in 0..la {
            for t in 0..=i + 1 {
                let lo = if t > 0 { h * e[i][0][t - 1] } else { 0.0 };
                let mid = if t <= i { pa[d] * e[i][0][t] } else { 0.0 };
                let hi = if t < i {
                    (t + 1) as f64 * e[i][0][t + 1]
                } else {
                    0.0
                };
                e[i + 1][0][t] = lo + mid + hi;
            }
        }
        for j in 0..lb {
            for i in 0..=la {
                for t in 0..=i + j + 1 {
                    let lo = if t > 0 { h * e[i][j][t - 1] } else { 0.0 };
                    let mid = if t <= i + j { pb[d] * e[i][j][t] } else { 0.0 };
                    let hi = if t < i + j {
                        (t + 1) as f64 * e[i][j][t + 1]
                    } else {
                        0.0
                    };
                    e[i][j + 1][t] = lo + mid + hi;
                }
            }
        }
        for i in 0..=la {
            for j in 0..=lb {
                let o = d * n1 + eoff(la, lb, i, j);
                out[o..o + i + j + 1].copy_from_slice(&e[i][j][..i + j + 1]);
            }
        }
    }
    out
}

/// How a class kernel is generated: three switches, decided per class by measurement
/// ([`Style::default_for`]); a study passes others to the engine.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Style {
    /// The Hermite coefficients read from `ecoef` (precomputed per primitive pair) instead of
    /// computed by every thread.
    pub e_global: bool,
    /// The Cartesian block accumulated per primitive straight into the point's Z values of the
    /// two shells (2 (na + nb) accumulators) instead of an na x nb block contracted once per pair.
    pub z_direct: bool,
    /// Primitive pairs screened per chunk by their charge's falloff.
    pub prim_screen: bool,
}

impl Style {
    /// The default of class (la, lb), the fastest per class on the H200 (ts61/def2-TZVP and
    /// QZVP, urea376/def2-SVP): the plain kernel (E computed per thread, the na x nb block, no primitive screening) except Z per
    /// primitive in (3,2), with screening in (4,2), and E from memory with screening in (3,3),
    /// (4,3) and (4,4).
    pub fn default_for(la: usize, lb: usize) -> Style {
        let (e_global, z_direct, prim_screen) = match (la, lb) {
            (3, 2) => (false, true, false),
            (4, 2) => (false, true, true),
            (3, 3) | (4, 3) | (4, 4) => (true, false, true),
            _ => (false, false, false),
        };
        Style {
            e_global,
            z_direct,
            prim_screen,
        }
    }

    /// The style of its letters: e (E global), z (Z direct), s (primitive screening); `-` for
    /// none of them.
    pub fn parse(letters: &str) -> Style {
        Style {
            e_global: letters.contains('e'),
            z_direct: letters.contains('z'),
            prim_screen: letters.contains('s'),
        }
    }
}

/// The CUDA source of the class (la, lb), la >= lb, in style `st`: per thread (point) and pair
/// of the slice, per primitive pair the Boys values and the R recurrence (straight-line), the
/// Cartesian block from the Hermite coefficients (read from `ecoef` or computed), into the
/// point's Z values (straight per primitive, or through an na x nb block per pair).
pub fn generate(la: usize, lb: usize, st: Style) -> String {
    assert!(la >= lb && la <= L_MAX);
    let l = la + lb;
    let (na, nb) = (ncart(la), ncart(lb));
    let n1 = ne1(la, lb);
    let mut s = String::new();
    let w = &mut s;
    writeln!(
        w,
        "// grid-K potential integrals, class ({la}, {lb}), {st:?} (generated by linus-gpu's kernels::gridk)"
    )
    .unwrap();
    writeln!(
        w,
        "extern \"C\" __global__ void __launch_bounds__({THREADS}) {}(",
        kernel_name(la, lb)
    )
    .unwrap();
    w.push_str(
        "    int npair, const int *__restrict__ pinfo, const double *__restrict__ pest,\n    \
         const double *__restrict__ prim, const double *__restrict__ ecoef,\n    \
         const double *__restrict__ pts, long long ld,\n    \
         const double *__restrict__ y, double *__restrict__ z, const int *__restrict__ chunk,\n    \
         const double *__restrict__ cxw, const double *__restrict__ ymax, int nshell, double thresh,\n    \
         const double *__restrict__ btab, double opc, double opl, double omega,\n    \
         long long zstride, const double *__restrict__ cgeo)\n{\n",
    );
    w.push_str(
        "    const int ch = blockIdx.x;\n    const int g = chunk[2 * ch] + (int)threadIdx.x;\n    \
         const bool act = g < chunk[2 * ch + 1];\n    \
         double gx = 0.0, gy = 0.0, gz = 0.0, gw = 0.0;\n    \
         if (act) { gx = pts[4 * g]; gy = pts[4 * g + 1]; gz = pts[4 * g + 2]; gw = pts[4 * g + 3]; }\n    \
         const double xw = cxw[ch];\n    const double *ym = ymax + (long long)ch * nshell;\n    \
         const double om2 = omega * omega;\n    \
         const int ns = (int)gridDim.y, sl = (int)blockIdx.y;\n    \
         const int ip0 = (int)(((long long)npair * sl) / ns);\n    \
         const int ip1 = (int)(((long long)npair * (sl + 1)) / ns);\n    \
         z += (long long)sl * zstride;\n    \
         const double ccx = cgeo[4 * ch], ccy = cgeo[4 * ch + 1], ccz = cgeo[4 * ch + 2];\n    \
         const double crad = cgeo[4 * ch + 3];\n",
    );
    w.push_str("    for (int ip = ip0; ip < ip1; ++ip) {\n");
    w.push_str(&format!(
        "        const int *pi = pinfo + {PAIR} * ip;\n        \
         const double fm = fmax(ym[pi[0]], ym[pi[1]]) * xw;\n        \
         if (pest[ip] * fm < thresh) continue;\n        \
         const double *ybra = y + (long long)pi[2] * ld + g;\n        \
         const double *yket = y + (long long)pi[3] * ld + g;\n"
    ));
    if st.z_direct {
        for a in 0..na {
            writeln!(
                w,
                "        const double ya{a} = act ? ybra[{a} * ld] : 0.0;"
            )
            .unwrap();
        }
        for b in 0..nb {
            writeln!(
                w,
                "        const double yb{b} = act ? yket[{b} * ld] : 0.0;"
            )
            .unwrap();
        }
        for a in 0..na {
            writeln!(w, "        double za{a} = 0.0;").unwrap();
        }
        for b in 0..nb {
            writeln!(w, "        double zb{b} = 0.0;").unwrap();
        }
    } else {
        for i in 0..na * nb {
            writeln!(w, "        double v{i} = 0.0;").unwrap();
        }
    }
    w.push_str(&format!(
        "        for (int k = pi[4]; k < pi[5]; ++k) {{\n            \
         const double *q = prim + {PRIM} * (long long)k;\n            \
         const double p = q[0], coef = q[1];\n"
    ));
    if st.prim_screen {
        w.push_str(
            "            {\n                \
             // beyond the chunk the primitive's integrals fall off as its charge's field:\n                \
             // est * sqrt(pi) / (2 sqrt(p) R) at a distance R from P (linus-ints' screen_prims)\n                \
             const double dx = q[2] - ccx, dy = q[3] - ccy, dz = q[4] - ccz;\n                \
             const double rr = sqrt(dx * dx + dy * dy + dz * dz) - crad;\n                \
             const double fd = rr > 0.0 ? fmin(1.0, 0.886226925452758 / (sqrt(p) * rr)) : 1.0;\n                \
             if (q[11] * fd * fm < thresh) continue;\n            \
             }\n",
        );
    }
    if st.e_global {
        writeln!(
            w,
            "            const double *e = ecoef + {} * (long long)k;",
            3 * n1
        )
        .unwrap();
    }
    w.push_str(
        "            const double pcx = q[2] - gx, pcy = q[3] - gy, pcz = q[4] - gz;\n            \
         const double T = p * (pcx * pcx + pcy * pcy + pcz * pcz);\n",
    );
    // Boys values of the operator: f_n = (-2p)^n [c F_n(T) + l kappa^(2n+1) F_n(kappa^2 T)]
    for n in 0..=l {
        writeln!(w, "            double f{n};").unwrap();
    }
    w.push_str("            {\n                double b[");
    w.push_str(&format!("{}];\n", l + 1));
    boys(w, l, "T", "b", "                ");
    for n in 0..=l {
        writeln!(w, "                f{n} = opc * b[{n}];").unwrap();
    }
    w.push_str("                if (opl != 0.0) {\n");
    w.push_str("                    const double k2 = om2 / (om2 + p);\n");
    w.push_str(&format!("                    double c[{}];\n", l + 1));
    boys(w, l, "k2 * T", "c", "                    ");
    w.push_str("                    double kp = opl * sqrt(k2);\n");
    for n in 0..=l {
        writeln!(w, "                    f{n} += kp * c[{n}];").unwrap();
        if n < l {
            w.push_str("                    kp *= k2;\n");
        }
    }
    w.push_str("                }\n");
    w.push_str("                double m2p = coef;\n");
    w.push_str("                f0 *= coef;\n");
    for n in 1..=l {
        writeln!(w, "                m2p *= -2.0 * p; f{n} *= m2p;").unwrap();
    }
    w.push_str("            }\n");
    // the Hermite coefficients: loads of ecoef, or e{d}_{i}_{j}_{t} computed here
    let ex = |d: usize, i: usize, j: usize, t: usize| -> String {
        if st.e_global {
            format!("e[{}]", d * n1 + eoff(la, lb, i, j) + t)
        } else {
            format!("e{}_{i}_{j}_{t}", ["x", "y", "z"][d])
        }
    };
    if !st.e_global {
        w.push_str("            const double h = 0.5 / p;\n");
        for (d, c) in ["x", "y", "z"].iter().enumerate() {
            let pa = format!("q[{}]", 5 + d);
            let pb = format!("q[{}]", 8 + d);
            let e = |i: usize, j: usize, t: i64| -> Option<String> {
                (t >= 0 && t as usize <= i + j).then(|| format!("e{c}_{i}_{j}_{t}"))
            };
            writeln!(w, "            const double e{c}_0_0_0 = 1.0;").unwrap();
            for i in 0..la {
                for t in 0..=(i + 1) as i64 {
                    let mut terms = vec![];
                    if let Some(x) = e(i, 0, t - 1) {
                        terms.push(format!("h * {x}"));
                    }
                    if let Some(x) = e(i, 0, t) {
                        terms.push(format!("{pa} * {x}"));
                    }
                    if let Some(x) = e(i, 0, t + 1) {
                        terms.push(format!("{}.0 * {x}", t + 1));
                    }
                    let lhs = format!("e{c}_{}_0_{t}", i + 1);
                    writeln!(w, "            const double {lhs} = {};", terms.join(" + ")).unwrap();
                }
            }
            for j in 0..lb {
                for i in 0..=la {
                    for t in 0..=(i + j + 1) as i64 {
                        let mut terms = vec![];
                        if let Some(x) = e(i, j, t - 1) {
                            terms.push(format!("h * {x}"));
                        }
                        if let Some(x) = e(i, j, t) {
                            terms.push(format!("{pb} * {x}"));
                        }
                        if let Some(x) = e(i, j, t + 1) {
                            terms.push(format!("{}.0 * {x}", t + 1));
                        }
                        let lhs = format!("e{c}_{i}_{}_{t}", j + 1);
                        writeln!(w, "            const double {lhs} = {};", terms.join(" + "))
                            .unwrap();
                    }
                }
            }
        }
    }
    // the bra's Cartesian components in groups, each with the R recurrence of its own (the f
    // values shared): the high classes' R tensor and accumulators then fit more of the
    // registers; recomputing R is cheap next to the contraction
    let ca = cart_components(la);
    let cb = cart_components(lb);
    let ng = if st.z_direct { groups(l) } else { 1 };
    let gsize = na.div_ceil(ng);
    for g0 in (0..na).step_by(gsize) {
        w.push_str("            {\n");
        // R recurrence (coef folded into f): r{n}_{t}_{u}_{v}, from level n = l down to 0
        for n in (0..=l).rev() {
            for (t, u, v) in tuvs(l - n) {
                let name = format!("r{n}_{t}_{u}_{v}");
                let rhs = if t > 0 {
                    let a = format!("pcx * r{}_{}_{u}_{v}", n + 1, t - 1);
                    if t > 1 {
                        format!("{a} + {}.0 * r{}_{}_{u}_{v}", t - 1, n + 1, t - 2)
                    } else {
                        a
                    }
                } else if u > 0 {
                    let a = format!("pcy * r{}_{t}_{}_{v}", n + 1, u - 1);
                    if u > 1 {
                        format!("{a} + {}.0 * r{}_{t}_{}_{v}", u - 1, n + 1, u - 2)
                    } else {
                        a
                    }
                } else if v > 0 {
                    let a = format!("pcz * r{}_{t}_{u}_{}", n + 1, v - 1);
                    if v > 1 {
                        format!("{a} + {}.0 * r{}_{t}_{u}_{}", v - 1, n + 1, v - 2)
                    } else {
                        a
                    }
                } else {
                    format!("f{n}")
                };
                writeln!(w, "            const double {name} = {rhs};").unwrap();
            }
        }
        // the contraction: V_ab = sum_t Ex_t (sum_u Ey_u (sum_v Ez_v R_tuv))
        for (ia, a) in ca.iter().enumerate().skip(g0).take(gsize) {
            for (ib, b) in cb.iter().enumerate() {
                let mut sum_t = String::new();
                for t in 0..=a[0] + b[0] {
                    let mut sum_u = String::new();
                    for u in 0..=a[1] + b[1] {
                        let sv: Vec<String> = (0..=a[2] + b[2])
                            .map(|v| format!("{} * r0_{t}_{u}_{v}", ex(2, a[2], b[2], v)))
                            .collect();
                        let term = format!("{} * ({})", ex(1, a[1], b[1], u), sv.join(" + "));
                        sum_u = if sum_u.is_empty() {
                            term
                        } else {
                            format!("{sum_u} + {term}")
                        };
                    }
                    let term = format!("{} * ({sum_u})", ex(0, a[0], b[0], t));
                    sum_t = if sum_t.is_empty() {
                        term
                    } else {
                        format!("{sum_t} + {term}")
                    };
                }
                if st.z_direct {
                    writeln!(
                        w,
                        "            {{\n                const double v = {sum_t};"
                    )
                    .unwrap();
                    writeln!(
                        w,
                        "                za{ia} += v * yb{ib};\n                zb{ib} += v * ya{ia};\n            }}"
                    )
                    .unwrap();
                } else {
                    writeln!(w, "            v{} += {sum_t};", ia * nb + ib).unwrap();
                }
            }
        }
        w.push_str("            }\n");
    }
    w.push_str("        }\n");
    // Z updates
    w.push_str("        if (act) {\n");
    w.push_str("            double *zbra = z + (long long)pi[2] * ld + g;\n");
    if st.z_direct {
        for a in 0..na {
            writeln!(w, "            zbra[{a} * ld] += gw * za{a};").unwrap();
        }
    } else {
        for ia in 0..na {
            let terms: Vec<String> = (0..nb)
                .map(|ib| format!("v{} * yket[{} * ld]", ia * nb + ib, ib))
                .collect();
            writeln!(
                w,
                "            zbra[{ia} * ld] += gw * ({});",
                terms.join(" + ")
            )
            .unwrap();
        }
    }
    w.push_str("            if (pi[0] != pi[1]) {\n");
    w.push_str("                double *zket = z + (long long)pi[3] * ld + g;\n");
    if st.z_direct {
        for b in 0..nb {
            writeln!(w, "                zket[{b} * ld] += gw * zb{b};").unwrap();
        }
    } else {
        for ib in 0..nb {
            let terms: Vec<String> = (0..na)
                .map(|ia| format!("v{} * ybra[{} * ld]", ia * nb + ib, ia))
                .collect();
            writeln!(
                w,
                "                zket[{ib} * ld] += gw * ({});",
                terms.join(" + ")
            )
            .unwrap();
        }
    }
    w.push_str("            }\n        }\n    }\n}\n");
    s
}

/// Boys values F_0..F_l at `t` into `out[0..=l]`, in a block of their own: [`boys_core`] with
/// `t` bound to the expression.
fn boys(w: &mut String, l: usize, t: &str, out: &str, ind: &str) {
    writeln!(w, "{ind}{{").unwrap();
    writeln!(w, "{ind}    const double t = {t};").unwrap();
    w.push_str(&boys_core(l, &format!("{ind}    ")));
    for m in 0..=l {
        writeln!(w, "{ind}    {out}[{m}] = f{m};").unwrap();
    }
    writeln!(w, "{ind}}}").unwrap();
}

/// The Boys stage of the CUDA emitters (tools/intgen/cuda_common.py's `boys_stage`, between `t`
/// and its scaling), `t` in scope, indented by `ind`: `double f0..fl` from the Boys table (0.1
/// apart: an 8-term Taylor expansion of F_l about the nearest row, exp(-t), downward recursion)
/// below [`BOYS_TMAX`], and linus-ints' asymptotic form from it on. Pinned to that emitter's
/// text by `gridk_boys.txt` (the tests), so the two do not drift.
fn boys_core(l: usize, ind: &str) -> String {
    let mut w = String::new();
    let lit = super::lit;
    let fv: Vec<String> = (0..=l).map(|m| format!("f{m}")).collect();
    writeln!(w, "{ind}double {};", fv.join(", ")).unwrap();
    writeln!(w, "{ind}if (t < {}) {{", lit(BOYS_TMAX)).unwrap();
    writeln!(w, "{ind}    const int i = (int)(t * 10.0 + 0.5);").unwrap();
    writeln!(w, "{ind}    const double dt = i * 0.1 - t;").unwrap();
    writeln!(w, "{ind}    const double *row = btab + i * {BOYS_COLS};").unwrap();
    writeln!(w, "{ind}    double acc = row[{}];", l + 7).unwrap();
    for k in (0..7).rev() {
        writeln!(
            w,
            "{ind}    acc = row[{}] + acc * dt * {};",
            l + k,
            lit(1.0 / (k + 1) as f64)
        )
        .unwrap();
    }
    writeln!(w, "{ind}    f{l} = acc;").unwrap();
    // exp(-t) = exp(-t_i) (1 + dt + dt^2 / 2 + ...), nested from the 8th order in
    let mut fact = [1.0f64; 9];
    for k in 1..9 {
        fact[k] = fact[k - 1] * k as f64;
    }
    let mut ser = lit(1.0 / fact[8]);
    for k in (0..8).rev() {
        ser = format!("{} + dt * ({ser})", lit(1.0 / fact[k]));
    }
    writeln!(
        w,
        "{ind}    const double e = row[{}] * ({ser});",
        BOYS_COLS - 1
    )
    .unwrap();
    for m in (0..l).rev() {
        writeln!(
            w,
            "{ind}    f{m} = (2.0 * t * f{} + e) * {};",
            m + 1,
            lit(1.0 / (2 * m + 1) as f64)
        )
        .unwrap();
    }
    writeln!(w, "{ind}}} else {{").unwrap();
    writeln!(w, "{ind}    const double ti = 1.0 / t;").unwrap();
    writeln!(
        w,
        "{ind}    f0 = 0.5 * sqrt({} * ti);",
        lit(std::f64::consts::PI)
    )
    .unwrap();
    for m in 1..=l {
        writeln!(
            w,
            "{ind}    f{m} = f{} * ({} * ti);",
            m - 1,
            lit((2 * m - 1) as f64 * 0.5)
        )
        .unwrap();
    }
    writeln!(w, "{ind}}}").unwrap();
    w
}

/// The engine's hand-written kernels (`device/gridk/gridk.cu`): the setup's row and column
/// scalings and per-chunk screening factor, the per-chunk max |Y|, the sum of Z's slices and
/// the symmetrized K (+ K0).
pub const SOURCE: &str = include_str!("../../device/gridk/gridk.cu");

#[cfg(test)]
mod tests {
    use super::*;

    /// The Boys stage is the CUDA emitters' (tools/intgen/cuda_common.py's `boys_stage`) at
    /// every order the classes take, line for line: `gridk_boys.txt` is that emitter's text,
    /// regenerated by `tools/intgen/intgen.py` (and checked by its `--check`). Literals compare
    /// as their values print (Python writes 2.48015873015873e-05, Rust 2.48015873015873e-5: the
    /// same double).
    #[test]
    fn the_boys_stage_is_the_intgen_emitters() {
        let exponents = |s: &str| {
            s.replace("e-0", "e-")
                .replace("e+0", "e")
                .replace("e+", "e")
        };
        let reference = include_str!("gridk_boys.txt");
        let mut blocks = std::collections::BTreeMap::new();
        let mut cur: Option<(usize, String)> = None;
        for line in reference.lines().filter(|l| !l.starts_with('#')) {
            if let Some(l) = line.strip_prefix("== lt ") {
                if let Some((k, b)) = cur.take() {
                    blocks.insert(k, b);
                }
                cur = Some((l.parse().unwrap(), String::new()));
            } else if let Some((_, b)) = cur.as_mut() {
                b.push_str(line);
                b.push('\n');
            }
        }
        if let Some((k, b)) = cur.take() {
            blocks.insert(k, b);
        }
        assert_eq!(blocks.len(), 2 * L_MAX + 1);
        for (l, want) in &blocks {
            assert_eq!(exponents(&boys_core(*l, "")), exponents(want), "order {l}");
        }
    }

    #[test]
    fn every_class_generates() {
        for la in 0..=L_MAX {
            for lb in 0..=la {
                for bits in 0..8 {
                    let st = Style {
                        e_global: bits & 1 != 0,
                        z_direct: bits & 2 != 0,
                        prim_screen: bits & 4 != 0,
                    };
                    let s = generate(la, lb, st);
                    assert!(s.contains(&kernel_name(la, lb)));
                }
                let s = generate(la, lb, Style::default_for(la, lb));
                assert!(s.contains(&kernel_name(la, lb)));
            }
        }
    }
}
