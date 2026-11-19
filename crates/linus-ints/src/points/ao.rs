//! Basis functions and their Cartesian derivatives on points (for DFT quadrature and
//! seminumerical exchange).
//!
//! Output layout: `out[comp][ao][pt]` (row-major), points contiguous, where the selected
//! shells' functions are concatenated in the order given and `comp` runs over derivative
//! multi-indices: order 0: \[1\]; order 1 adds [x, y, z]; order 2 adds [xx, xy, xz, yy, yz, zz];
//! order 3 adds [xxx, xxy, xxz, xyy, xyz, xzz, yyy, yyz, yzz, zzz].

use crate::basis::{cart_components, ncart, BasisSet, Shell};

/// Number of derivative components up to `order`.
pub const fn ncomp(order: usize) -> usize {
    (order + 1) * (order + 2) * (order + 3) / 6
}

/// Derivative multi-indices up to `order` in output order.
pub fn deriv_indices(order: usize) -> Vec<[usize; 3]> {
    let mut v = Vec::new();
    for n in 0..=order {
        v.extend(cart_components(n));
    }
    v
}

/// Terms of d^p/dx^p [x^l exp(-a x^2)] = sum coef * x^m * (-2a)^k * exp(-a x^2): (coef, m, k).
pub(crate) fn deriv_terms(l: usize, p: usize) -> Vec<(f64, usize, usize)> {
    let mut t = vec![(1.0, l as i64, 0usize)];
    for _ in 0..p {
        let mut n: Vec<(f64, i64, usize)> = Vec::new();
        for &(c, m, k) in &t {
            if m > 0 {
                n.push((c * m as f64, m - 1, k));
            }
            n.push((c, m + 1, k + 1));
        }
        // merge equal (m, k)
        n.sort_by_key(|&(_, m, k)| (m, k));
        let mut merged: Vec<(f64, i64, usize)> = Vec::new();
        for x in n {
            if let Some(last) = merged.last_mut() {
                if last.1 == x.1 && last.2 == x.2 {
                    last.0 += x.0;
                    continue;
                }
            }
            merged.push(x);
        }
        t = merged;
    }
    t.into_iter().map(|(c, m, k)| (c, m as usize, k)).collect()
}

/// Precomputed evaluation recipe for one angular momentum and derivative order:
/// for each (component, cartesian function): list of (coef, mx, my, mz, k).
struct Recipe {
    #[allow(clippy::type_complexity, reason = "a one-off tuple of buffers")]
    terms: Vec<Vec<Vec<(f64, u8, u8, u8, u8)>>>,
}

fn recipe(l: usize, order: usize) -> Recipe {
    let comps = deriv_indices(order);
    let carts = cart_components(l);
    let mut terms = Vec::with_capacity(comps.len());
    for d in &comps {
        let mut per_cart = Vec::with_capacity(carts.len());
        for c in &carts {
            let tx = deriv_terms(c[0], d[0]);
            let ty = deriv_terms(c[1], d[1]);
            let tz = deriv_terms(c[2], d[2]);
            let mut v = Vec::new();
            for &(cx, mx, kx) in &tx {
                for &(cy, my, ky) in &ty {
                    for &(cz, mz, kz) in &tz {
                        v.push((
                            cx * cy * cz,
                            mx as u8,
                            my as u8,
                            mz as u8,
                            (kx + ky + kz) as u8,
                        ));
                    }
                }
            }
            per_cart.push(v);
        }
        terms.push(per_cart);
    }
    Recipe { terms }
}

/// Cached recipes for l <= 10 and derivative order <= 4.
fn recipes(order: usize) -> &'static [Recipe] {
    use std::sync::OnceLock;
    static R: OnceLock<Vec<Vec<Recipe>>> = OnceLock::new();
    &R.get_or_init(|| {
        (0..=4)
            .map(|o| (0..=10).map(|l| recipe(l, o)).collect())
            .collect()
    })[order]
}

/// Radius beyond which every primitive of the shell (times r^l) is below `thresh`.
pub fn shell_extent(sh: &Shell, thresh: f64) -> f64 {
    let mut rmax: f64 = 0.0;
    for (&a, &c) in sh.exps.iter().zip(&sh.coefs) {
        // solve |c| r^l exp(-a r^2) = thresh by bisection-free Newton on log form
        let lc = c.abs().ln() - thresh.ln();
        let mut r = ((lc.max(0.0)) / a).sqrt().max(1e-3);
        for _ in 0..50 {
            let f = lc + sh.l as f64 * r.ln() - a * r * r;
            let df = sh.l as f64 / r - 2.0 * a * r;
            let nr = (r - f / df).max(1e-6);
            if (nr - r).abs() < 1e-10 * r {
                r = nr;
                break;
            }
            r = nr;
        }
        rmax = rmax.max(r);
    }
    rmax
}

/// Evaluate the functions of `shells` (indices into basis.shells) and derivatives up to
/// `order` at points (x, y, z). `out` must hold ncomp(order) * nao_sel * npts values.
/// Returns nao_sel.
pub fn eval_ao(
    basis: &BasisSet,
    shells: &[usize],
    x: &[f64],
    y: &[f64],
    z: &[f64],
    order: usize,
    out: &mut [f64],
) -> usize {
    let npts = x.len();
    let nao: usize = shells.iter().map(|&s| basis.shells[s].nfunc()).sum();
    let nc = ncomp(order);
    assert!(out.len() >= nc * nao * npts);
    let lmax = shells.iter().map(|&s| basis.shells[s].l).max().unwrap_or(0);
    let recipes = recipes(order);
    assert!(
        lmax < recipes.len(),
        "angular momentum too high for AO evaluation"
    );
    let pmax = lmax + order + 1;
    let mut px = vec![0.0; pmax * npts];
    let mut py = vec![0.0; pmax * npts];
    let mut pz = vec![0.0; pmax * npts];
    let mut rk = vec![0.0; (order + 1) * npts];
    let mut cart = vec![0.0; nc * ncart(lmax) * npts];
    let mut ao0 = 0usize;
    for &si in shells {
        let sh = &basis.shells[si];
        let l = sh.l;
        let ncar = ncart(l);
        let pm = l + order + 1;
        // powers and radial sums
        rk.fill(0.0);
        for g in 0..npts {
            let dx = x[g] - sh.center[0];
            let dy = y[g] - sh.center[1];
            let dz = z[g] - sh.center[2];
            px[g] = 1.0;
            py[g] = 1.0;
            pz[g] = 1.0;
            for m in 1..pm {
                px[m * npts + g] = px[(m - 1) * npts + g] * dx;
                py[m * npts + g] = py[(m - 1) * npts + g] * dy;
                pz[m * npts + g] = pz[(m - 1) * npts + g] * dz;
            }
            let r2 = dx * dx + dy * dy + dz * dz;
            for (&a, &c) in sh.exps.iter().zip(&sh.coefs) {
                let e = c * (-a * r2).exp();
                let mut f = e;
                for k in 0..=order {
                    rk[k * npts + g] += f;
                    f *= -2.0 * a;
                }
            }
        }
        // Cartesian values per derivative component
        let rec = &recipes[l];
        for d in 0..nc {
            for ic in 0..ncar {
                let dst = &mut cart[(d * ncar + ic) * npts..(d * ncar + ic + 1) * npts];
                dst.fill(0.0);
                for &(coef, mx, my, mz, k) in &rec.terms[d][ic] {
                    let (mx, my, mz, k) = (mx as usize, my as usize, mz as usize, k as usize);
                    let (ax, ay, az, ar) = (
                        &px[mx * npts..],
                        &py[my * npts..],
                        &pz[mz * npts..],
                        &rk[k * npts..],
                    );
                    for g in 0..npts {
                        dst[g] += coef * ax[g] * ay[g] * az[g] * ar[g];
                    }
                }
            }
        }
        // transform to the shell's functions
        let t = BasisSet::shell_transform_ref(sh);
        let nf = sh.nfunc();
        for d in 0..nc {
            for mu in 0..nf {
                let o = &mut out[(d * nao + ao0 + mu) * npts..(d * nao + ao0 + mu + 1) * npts];
                o.fill(0.0);
                for ic in 0..ncar {
                    let tv = t[mu * ncar + ic];
                    if tv == 0.0 {
                        continue;
                    }
                    let src = &cart[(d * ncar + ic) * npts..(d * ncar + ic + 1) * npts];
                    for g in 0..npts {
                        o[g] += tv * src[g];
                    }
                }
            }
        }
        ao0 += nf;
    }
    nao
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::basis::BasisDef;

    #[test]
    fn derivatives_match_finite_differences() {
        let json = r#"{"name":"t","elements":{"8":{"electron_shells":[
          {"function_type":"gto","angular_momentum":[0],"exponents":["5.0","1.0"],"coefficients":[["0.4","0.7"]]},
          {"function_type":"gto","angular_momentum":[1],"exponents":["2.0","0.5"],"coefficients":[["0.5","0.6"]]},
          {"function_type":"gto","angular_momentum":[2],"exponents":["1.2"],"coefficients":[["1.0"]]},
          {"function_type":"gto","angular_momentum":[3],"exponents":["0.9"],"coefficients":[["1.0"]]}]}}}"#;
        let def = BasisDef::from_bse_json(json).unwrap();
        let b = BasisSet::new(&def, &[8], &[[0.1, -0.2, 0.3]], true).unwrap();
        let shells: Vec<usize> = (0..b.shells.len()).collect();
        let (x, y, z) = (vec![0.4, -0.7], vec![0.2, 0.5], vec![-0.3, 0.9]);
        let order = 3;
        let nc = ncomp(order);
        let nao = b.nbf;
        let mut out = vec![0.0; nc * nao * 2];
        eval_ao(&b, &shells, &x, &y, &z, order, &mut out);
        let idx = deriv_indices(order);
        let h = 1e-4;
        // check every derivative component of order n against FD of order n-1 components
        for (ci, d) in idx.iter().enumerate().skip(1) {
            // find a direction k with d[k] > 0 and the lower component
            let k = (0..3).find(|&k| d[k] > 0).unwrap();
            let mut lower = *d;
            lower[k] -= 1;
            let li = idx.iter().position(|v| *v == lower).unwrap();
            let mut plus = vec![0.0; nc * nao * 2];
            let mut minus = vec![0.0; nc * nao * 2];
            let shift = |v: &Vec<f64>, s: f64, dim: usize, me: usize| -> Vec<f64> {
                if dim == me {
                    v.iter().map(|a| a + s).collect()
                } else {
                    v.clone()
                }
            };
            eval_ao(
                &b,
                &shells,
                &shift(&x, h, k, 0),
                &shift(&y, h, k, 1),
                &shift(&z, h, k, 2),
                order,
                &mut plus,
            );
            eval_ao(
                &b,
                &shells,
                &shift(&x, -h, k, 0),
                &shift(&y, -h, k, 1),
                &shift(&z, -h, k, 2),
                order,
                &mut minus,
            );
            for mu in 0..nao {
                for g in 0..2 {
                    let fd = (plus[(li * nao + mu) * 2 + g] - minus[(li * nao + mu) * 2 + g])
                        / (2.0 * h);
                    let an = out[(ci * nao + mu) * 2 + g];
                    assert!(
                        (fd - an).abs() < 1e-6 * (1.0 + an.abs()),
                        "comp {d:?} mu {mu} g {g}: {an} vs {fd}"
                    );
                }
            }
        }
    }
}
