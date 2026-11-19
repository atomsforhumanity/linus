//! Fast collocation of basis functions (and Cartesian derivatives) on blocks of grid points,
//! for the DFT quadrature: lane-parallel over points (8 at a time, [`linus_num::simd`]), one shell
//! at a time, with screening.
//!
//! Each output (function f, derivative component d) of a shell is a polynomial in the
//! displacement times radial sums, phi_fd = sum_k P_fdk(x, y, z) R_k(r^2), with
//! R_k = sum_p c_p (-2 a_p)^k exp(-a_p r^2). The polynomials (spherical transform folded in,
//! equal monomials merged) are tabulated per (l, pure, order) when the [`Collocator`] is built.
//!
//! Screening: a shell is skipped for a block when the block's bounding box is farther from its
//! centre than `rcut`, the radius beyond which a rigorous bound on |phi_fd| for every output is
//! below the collocator's threshold; within a shell, primitives whose bound is below
//! [`PRIM_CUT`] at that distance are left out. Evaluated functions whose largest |phi_fd| over
//! the block does not exceed the threshold are dropped (as [`crate::points::ao::eval_ao`]-based
//! screening does), so the box test only removes shells that would be dropped anyway.
//!
//! Output layout: function-major, `out[(f * ncomp + d) * ld + g]` for the kept functions in
//! shell order (`ld` >= the number of points, a multiple of 8). Component order as
//! [`crate::points::ao::deriv_indices`]. Lanes past the last point are zero.

use crate::basis::{cart_components, ncart, BasisSet};
use crate::points::ao::{deriv_indices, deriv_terms, ncomp};
use linus_num::simd::{exp_nonpos, Isa, Lanes, Portable, LANES};

/// Primitives whose contribution to every output is below this (absolute) are skipped.
pub const PRIM_CUT: f64 = 1e-18;

/// Polynomial recipe of one (l, pure, order).
#[derive(Clone, Debug)]
struct Recipe {
    nf: usize,
    /// distinct monomials (mx, my, mz)
    monos: Vec<[u8; 3]>,
    /// terms of output o = f * nc + d and radial order k: off[o * nk + k]..off[o * nk + k + 1]
    off: Vec<u32>,
    /// (coefficient, monomial index)
    terms: Vec<(f64, u16)>,
    /// largest monomial degree
    lpow: usize,
    /// per output and k: sum of |coef| r^m as (m, |coef|) pairs for bounds
    bound: Vec<Vec<(usize, f64)>>,
}

fn build_recipe(l: usize, pure: bool, order: usize) -> Recipe {
    let comps = deriv_indices(order);
    let carts = cart_components(l);
    let nc = comps.len();
    let nk = order + 1;
    let ncar = ncart(l);
    let cart_t;
    let t: &[f64] = if pure {
        crate::core::solid_harmonics::cart2sph_cached(l)
    } else {
        cart_t = crate::basis::cart_norm_matrix(l);
        &cart_t
    };
    let nf = if pure { 2 * l + 1 } else { ncar };
    let mut monos: Vec<[u8; 3]> = Vec::new();
    let mut off = vec![0u32];
    let mut terms = Vec::new();
    let mut bound = Vec::new();
    let mut lpow = 0;
    for f in 0..nf {
        for d in &comps {
            // (mono, k) -> coefficient
            let mut acc: Vec<([u8; 3], usize, f64)> = Vec::new();
            for (ic, c) in carts.iter().enumerate() {
                let tv = t[f * ncar + ic];
                if tv == 0.0 {
                    continue;
                }
                let tx = deriv_terms(c[0], d[0]);
                let ty = deriv_terms(c[1], d[1]);
                let tz = deriv_terms(c[2], d[2]);
                for &(cx, mx, kx) in &tx {
                    for &(cy, my, ky) in &ty {
                        for &(cz, mz, kz) in &tz {
                            let key = [mx as u8, my as u8, mz as u8];
                            let k = kx + ky + kz;
                            let v = tv * cx * cy * cz;
                            match acc.iter_mut().find(|e| e.0 == key && e.1 == k) {
                                Some(e) => e.2 += v,
                                None => acc.push((key, k, v)),
                            }
                        }
                    }
                }
            }
            for k in 0..nk {
                let mut b = Vec::new();
                for &(key, kk, v) in &acc {
                    if kk != k || v == 0.0 {
                        continue;
                    }
                    let mi = match monos.iter().position(|m| *m == key) {
                        Some(i) => i,
                        None => {
                            monos.push(key);
                            monos.len() - 1
                        }
                    };
                    let deg = (key[0] + key[1] + key[2]) as usize;
                    lpow = lpow.max(deg);
                    terms.push((v, mi as u16));
                    b.push((deg, v.abs()));
                }
                off.push(terms.len() as u32);
                bound.push(b);
            }
        }
    }
    let _ = nc;
    Recipe {
        nf,
        monos,
        off,
        terms,
        lpow,
        bound,
    }
}

/// Per-shell data of a [`Collocator`].
#[derive(Clone, Debug)]
pub struct ShellColl {
    rec: usize,
    pub center: [f64; 3],
    pub nf: usize,
    pub ao_start: usize,
    pub atom: usize,
    /// exponents and c (-2a)^k for k = 0..=order
    alpha: Vec<f64>,
    ck: Vec<[f64; 4]>,
    /// radius beyond which primitive p is below PRIM_CUT for every output
    prim_rcut: Vec<f64>,
    /// radius beyond which every output is below the threshold
    pub rcut: f64,
}

/// Collocation engine for one basis, derivative order and screening threshold.
#[derive(Clone, Debug)]
pub struct Collocator {
    pub order: usize,
    pub ncomp: usize,
    pub thresh: f64,
    isa: Isa,
    recipes: Vec<Recipe>,
    pub shells: Vec<ShellColl>,
}

/// Bound sum_terms C r^m e^{-a r^2} over (m, C, a) triples (one output).
fn bound_at(terms: &[(usize, f64, f64)], r: f64) -> f64 {
    terms
        .iter()
        .map(|&(m, c, a)| c * r.powi(m as i32) * (-a * r * r).exp())
        .sum()
}

/// Smallest r >= r0 (r0 past the maximum of every term) with bound(r) < thresh, by bisection.
fn cutoff_radius(terms: &[(usize, f64, f64)], thresh: f64) -> f64 {
    if terms.is_empty() {
        return 0.0;
    }
    let r0 = terms
        .iter()
        .map(|&(m, _, a)| (m as f64 / (2.0 * a)).sqrt())
        .fold(0.0, f64::max);
    if bound_at(terms, r0) < thresh {
        return r0;
    }
    let mut hi = r0.max(1.0);
    while bound_at(terms, hi) >= thresh {
        hi *= 1.5;
        if hi > 1e4 {
            return f64::INFINITY;
        }
    }
    let mut lo = r0;
    for _ in 0..60 {
        let mid = 0.5 * (lo + hi);
        if bound_at(terms, mid) >= thresh {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    hi * (1.0 + 1e-12)
}

impl Collocator {
    /// Recipes and per-shell screening data for derivatives up to `order` (<= 3); outputs with
    /// every |value| <= `thresh` over a block are dropped.
    #[allow(clippy::needless_range_loop, reason = "indexes several arrays in step")]
    pub fn new(basis: &BasisSet, order: usize, thresh: f64) -> Self {
        assert!(order <= 3, "collocation up to third derivatives");
        let mut keys: Vec<(usize, bool)> = Vec::new();
        let mut recipes = Vec::new();
        let nk = order + 1;
        let shells = basis
            .shells
            .iter()
            .map(|sh| {
                let key = (sh.l, sh.pure);
                let rec = match keys.iter().position(|k| *k == key) {
                    Some(i) => i,
                    None => {
                        keys.push(key);
                        recipes.push(build_recipe(sh.l, sh.pure, order));
                        recipes.len() - 1
                    }
                };
                let r = &recipes[rec];
                let alpha = sh.exps.clone();
                let ck: Vec<[f64; 4]> = sh
                    .exps
                    .iter()
                    .zip(&sh.coefs)
                    .map(|(&a, &c)| {
                        let mut v = [0.0; 4];
                        let mut f = c;
                        for item in v.iter_mut().take(nk) {
                            *item = f;
                            f *= -2.0 * a;
                        }
                        v
                    })
                    .collect();
                // bound terms per output: sum_k sum_t |coef| r^m |c (2a)^k| e^{-a r^2}
                let nout = r.nf * ncomp(order);
                let mut rcut: f64 = 0.0;
                for o in 0..nout {
                    let mut tt = Vec::new();
                    for k in 0..nk {
                        for &(m, c) in &r.bound[o * nk + k] {
                            for (p, &a) in alpha.iter().enumerate() {
                                tt.push((m, c * ck[p][k].abs(), a));
                            }
                        }
                    }
                    rcut = rcut.max(cutoff_radius(&tt, thresh));
                }
                // primitive cutoffs: the largest bound over outputs, with all monomials
                let prim_rcut = alpha
                    .iter()
                    .enumerate()
                    .map(|(p, &a)| {
                        let mut tt = Vec::new();
                        for o in 0..nout {
                            for k in 0..nk {
                                for &(m, c) in &r.bound[o * nk + k] {
                                    tt.push((m, c * ck[p][k].abs(), a));
                                }
                            }
                        }
                        cutoff_radius(&tt, PRIM_CUT)
                    })
                    .collect();
                ShellColl {
                    rec,
                    center: sh.center,
                    nf: sh.nfunc(),
                    ao_start: sh.ao_start,
                    atom: sh.atom,
                    alpha,
                    ck,
                    prim_rcut,
                    rcut,
                }
            })
            .collect();
        Collocator {
            order,
            ncomp: ncomp(order),
            thresh,
            isa: Isa::detect(),
            recipes,
            shells,
        }
    }

    /// Distance from `c` to the box [lo, hi].
    #[inline]
    pub fn box_distance(c: &[f64; 3], lo: &[f64; 3], hi: &[f64; 3]) -> f64 {
        let mut d2 = 0.0;
        for k in 0..3 {
            let e = if c[k] < lo[k] {
                lo[k] - c[k]
            } else if c[k] > hi[k] {
                c[k] - hi[k]
            } else {
                0.0
            };
            d2 += e * e;
        }
        d2.sqrt()
    }

    /// Evaluates the candidate `shells` on a block of points (x, y, z of length >= `ld`,
    /// padded; `np` real points) with bounding box [lo, hi]. Writes the kept functions to
    /// `out` (layout in the module docs; needs room for every candidate function), their
    /// global indices to `idx` and their shells' atoms to `owner` (both cleared first), and
    /// returns their number.
    #[allow(clippy::too_many_arguments)]
    pub fn eval(
        &self,
        shells: &[usize],
        x: &[f64],
        y: &[f64],
        z: &[f64],
        np: usize,
        ld: usize,
        lo: &[f64; 3],
        hi: &[f64; 3],
        out: &mut [f64],
        idx: &mut Vec<usize>,
        owner: &mut Vec<usize>,
    ) -> usize {
        assert!(
            ld.is_multiple_of(LANES) && ld >= np && x.len() >= ld && y.len() >= ld && z.len() >= ld
        );
        idx.clear();
        owner.clear();
        match self.isa {
            #[cfg(target_arch = "x86_64")]
            Isa::Avx512 => unsafe {
                eval_avx512(self, shells, x, y, z, np, ld, lo, hi, out, idx, owner)
            },
            #[cfg(target_arch = "x86_64")]
            Isa::Avx2 => unsafe {
                eval_avx2(self, shells, x, y, z, np, ld, lo, hi, out, idx, owner)
            },
            _ => eval_generic::<Portable<false>>(
                self, shells, x, y, z, np, ld, lo, hi, out, idx, owner,
            ),
        }
    }

    /// [`Self::eval`] with the portable (baseline) lanes, for tests.
    #[allow(clippy::too_many_arguments)]
    pub fn eval_portable(
        &self,
        shells: &[usize],
        x: &[f64],
        y: &[f64],
        z: &[f64],
        np: usize,
        ld: usize,
        lo: &[f64; 3],
        hi: &[f64; 3],
        out: &mut [f64],
        idx: &mut Vec<usize>,
        owner: &mut Vec<usize>,
    ) -> usize {
        idx.clear();
        owner.clear();
        eval_generic::<Portable<false>>(self, shells, x, y, z, np, ld, lo, hi, out, idx, owner)
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512dq,avx2,fma")]
#[allow(clippy::too_many_arguments)]
unsafe fn eval_avx512(
    c: &Collocator,
    shells: &[usize],
    x: &[f64],
    y: &[f64],
    z: &[f64],
    np: usize,
    ld: usize,
    lo: &[f64; 3],
    hi: &[f64; 3],
    out: &mut [f64],
    idx: &mut Vec<usize>,
    owner: &mut Vec<usize>,
) -> usize {
    eval_generic::<linus_num::simd::Avx512>(c, shells, x, y, z, np, ld, lo, hi, out, idx, owner)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[allow(clippy::too_many_arguments)]
unsafe fn eval_avx2(
    c: &Collocator,
    shells: &[usize],
    x: &[f64],
    y: &[f64],
    z: &[f64],
    np: usize,
    ld: usize,
    lo: &[f64; 3],
    hi: &[f64; 3],
    out: &mut [f64],
    idx: &mut Vec<usize>,
    owner: &mut Vec<usize>,
) -> usize {
    eval_generic::<linus_num::simd::Avx2>(c, shells, x, y, z, np, ld, lo, hi, out, idx, owner)
}

/// Largest nf * ncomp of a shell (l <= 6 pure or l <= 4 Cartesian, order <= 3: 13 x 20).
const MAX_OUT: usize = 15 * 20;
const MAX_MONO: usize = 220;
const MAX_POW: usize = 10;

#[inline(always)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::needless_range_loop, reason = "indexes several arrays in step")]
fn eval_generic<V: Lanes>(
    c: &Collocator,
    shells: &[usize],
    x: &[f64],
    y: &[f64],
    z: &[f64],
    np: usize,
    ld: usize,
    lo: &[f64; 3],
    hi: &[f64; 3],
    out: &mut [f64],
    idx: &mut Vec<usize>,
    owner: &mut Vec<usize>,
) -> usize {
    let nc = c.ncomp;
    let nk = c.order + 1;
    let thresh = c.thresh;
    let mut nao = 0;
    let mut active: Vec<usize> = Vec::with_capacity(16);
    // mask of real points in the last chunk
    let last = (np.max(1) - 1) / LANES * LANES;
    let mask: V = V::from_array(std::array::from_fn(
        |i| if last + i < np { 1.0 } else { 0.0 },
    ));
    let nchunks = np.div_ceil(LANES);
    for &si in shells {
        let sh = &c.shells[si];
        let d = Collocator::box_distance(&sh.center, lo, hi);
        if d >= sh.rcut {
            continue;
        }
        active.clear();
        active.extend((0..sh.alpha.len()).filter(|&p| d < sh.prim_rcut[p]));
        if active.is_empty() {
            continue;
        }
        let rec = &c.recipes[sh.rec];
        let nf = rec.nf;
        let nout = nf * nc;
        assert!(nout <= MAX_OUT && rec.monos.len() <= MAX_MONO && rec.lpow < MAX_POW && nf <= 15);
        let base = nao * nc * ld;
        assert!(out.len() >= base + nout * ld);
        let mut vmax = [V::zero(); 15];
        let (cx, cy, cz) = (
            V::splat(sh.center[0]),
            V::splat(sh.center[1]),
            V::splat(sh.center[2]),
        );
        let lp = rec.lpow;
        for ch in 0..nchunks {
            let g0 = ch * LANES;
            let dx = V::load(&x[g0..]) - cx;
            let dy = V::load(&y[g0..]) - cy;
            let dz = V::load(&z[g0..]) - cz;
            let r2 = dx.mul_add(dx, dy.mul_add(dy, dz * dz));
            let mut rk = [V::zero(); 4];
            for &p in &active {
                let e = exp_nonpos(V::splat(-sh.alpha[p]) * r2);
                let ck = &sh.ck[p];
                for k in 0..nk {
                    rk[k] = V::splat(ck[k]).mul_add(e, rk[k]);
                }
            }
            let mut px = [V::splat(1.0); MAX_POW];
            let mut py = [V::splat(1.0); MAX_POW];
            let mut pz = [V::splat(1.0); MAX_POW];
            for i in 1..=lp {
                px[i] = px[i - 1] * dx;
                py[i] = py[i - 1] * dy;
                pz[i] = pz[i - 1] * dz;
            }
            let mut mono = [V::zero(); MAX_MONO];
            for (i, m) in rec.monos.iter().enumerate() {
                mono[i] = px[m[0] as usize] * py[m[1] as usize] * pz[m[2] as usize];
            }
            let lastc = ch + 1 == nchunks;
            for o in 0..nout {
                let mut v = V::zero();
                for k in 0..nk {
                    let (t0, t1) = (
                        rec.off[o * nk + k] as usize,
                        rec.off[o * nk + k + 1] as usize,
                    );
                    if t0 == t1 {
                        continue;
                    }
                    let mut a = V::zero();
                    for &(coef, mi) in &rec.terms[t0..t1] {
                        a = V::splat(coef).mul_add(mono[mi as usize], a);
                    }
                    v = a.mul_add(rk[k], v);
                }
                if lastc {
                    v *= mask;
                }
                v.store(&mut out[base + o * ld + g0..]);
                let f = o / nc;
                vmax[f] = vmax[f].max(v.abs());
            }
        }
        // keep functions above the threshold, compacting within the shell's block
        let mut dst = nao;
        for (f, vm) in vmax.iter().enumerate().take(nf) {
            let m = vm.to_array().iter().fold(0.0f64, |a, &b| a.max(b));
            if m > thresh {
                let src = nao + f;
                if src != dst {
                    out.copy_within(src * nc * ld..(src + 1) * nc * ld, dst * nc * ld);
                }
                idx.push(sh.ao_start + f);
                owner.push(sh.atom);
                dst += 1;
            }
        }
        nao = dst;
    }
    nao
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::basis::BasisDef;
    use crate::points::ao::eval_ao;

    fn test_basis(pure: bool) -> BasisSet {
        let json = r#"{"name":"t","elements":{"8":{"electron_shells":[
          {"function_type":"gto","angular_momentum":[0],"exponents":["50.0","5.0","1.0"],"coefficients":[["0.2","0.4","0.7"]]},
          {"function_type":"gto","angular_momentum":[1],"exponents":["2.0","0.5"],"coefficients":[["0.5","0.6"]]},
          {"function_type":"gto","angular_momentum":[2],"exponents":["1.2"],"coefficients":[["1.0"]]},
          {"function_type":"gto","angular_momentum":[3],"exponents":["0.9"],"coefficients":[["1.0"]]},
          {"function_type":"gto","angular_momentum":[4],"exponents":["1.1"],"coefficients":[["1.0"]]}]},
          "1":{"electron_shells":[
          {"function_type":"gto","angular_momentum":[0],"exponents":["3.0","0.3"],"coefficients":[["0.5","0.6"]]},
          {"function_type":"gto","angular_momentum":[1],"exponents":["0.8"],"coefficients":[["1.0"]]}]}}}"#;
        let def = BasisDef::from_bse_json(json).unwrap();
        BasisSet::new(&def, &[8, 1], &[[0.1, -0.2, 0.3], [1.5, 0.4, -1.0]], pure).unwrap()
    }

    /// Every component of every function against eval_ao (no screening), all ISAs.
    #[test]
    fn matches_eval_ao() {
        for pure in [true, false] {
            let b = test_basis(pure);
            let shells: Vec<usize> = (0..b.shells.len()).collect();
            let np: usize = 37;
            let ld = np.div_ceil(LANES) * LANES;
            let pts: Vec<[f64; 3]> = (0..ld)
                .map(|i| {
                    let t = i.min(np - 1) as f64;
                    [
                        0.3 * (t * 0.7).sin() * (1.0 + 0.1 * t),
                        0.5 * (t * 1.3).cos(),
                        -0.2 + 0.05 * t,
                    ]
                })
                .collect();
            let (x, y, z): (Vec<f64>, Vec<f64>, Vec<f64>) = (
                pts.iter().map(|p| p[0]).collect(),
                pts.iter().map(|p| p[1]).collect(),
                pts.iter().map(|p| p[2]).collect(),
            );
            for order in 0..=3 {
                let nc = ncomp(order);
                let nao = b.nbf;
                let mut refv = vec![0.0; nc * nao * np];
                eval_ao(&b, &shells, &x[..np], &y[..np], &z[..np], order, &mut refv);
                let c = Collocator::new(&b, order, 0.0);
                let (lo, hi) = ([-1e3; 3], [1e3; 3]);
                for isa in 0..3 {
                    let mut out = vec![f64::NAN; nao * nc * ld];
                    let (mut idx, mut own) = (vec![], vec![]);
                    let n = match isa {
                        0 => c.eval_portable(
                            &shells, &x, &y, &z, np, ld, &lo, &hi, &mut out, &mut idx, &mut own,
                        ),
                        // the AVX2 and AVX-512 kernels exist on x86-64 only
                        #[cfg(not(target_arch = "x86_64"))]
                        _ => continue,
                        #[cfg(target_arch = "x86_64")]
                        1 => {
                            if Isa::detect() == Isa::Generic {
                                continue;
                            }
                            idx.clear();
                            own.clear();
                            unsafe {
                                eval_avx2(
                                    &c, &shells, &x, &y, &z, np, ld, &lo, &hi, &mut out, &mut idx,
                                    &mut own,
                                )
                            }
                        }
                        #[cfg(target_arch = "x86_64")]
                        _ => {
                            if Isa::detect() != Isa::Avx512 {
                                continue;
                            }
                            idx.clear();
                            own.clear();
                            unsafe {
                                eval_avx512(
                                    &c, &shells, &x, &y, &z, np, ld, &lo, &hi, &mut out, &mut idx,
                                    &mut own,
                                )
                            }
                        }
                    };
                    assert_eq!(n, nao);
                    assert_eq!(idx, (0..nao).collect::<Vec<_>>());
                    for f in 0..nao {
                        for d in 0..nc {
                            for g in 0..ld {
                                let v = out[(f * nc + d) * ld + g];
                                if g >= np {
                                    assert_eq!(v, 0.0);
                                    continue;
                                }
                                let r = refv[(d * nao + f) * np + g];
                                assert!((v - r).abs() <= 1e-13 * (1.0 + r.abs()), "pure {pure} order {order} isa {isa} f {f} d {d} g {g}: {v} vs {r}");
                            }
                        }
                    }
                }
            }
        }
    }

    /// With screening: functions kept are exactly those above the threshold, and the shell
    /// cutoff radii bound every output.
    #[test]
    fn screening_is_rigorous() {
        let b = test_basis(true);
        let shells: Vec<usize> = (0..b.shells.len()).collect();
        for order in 0..=2 {
            let thresh = 1e-11;
            let c = Collocator::new(&b, order, thresh);
            let nc = ncomp(order);
            for (si, sh) in c.shells.iter().enumerate() {
                // points just outside rcut in several directions: every output below thresh
                for k in 0..20 {
                    let t = k as f64 * 0.9;
                    let dir = [
                        t.sin() * (1.7 * t).cos(),
                        t.sin() * (1.7 * t).sin(),
                        t.cos(),
                    ];
                    let r = sh.rcut * 1.0001;
                    let p = [
                        sh.center[0] + r * dir[0],
                        sh.center[1] + r * dir[1],
                        sh.center[2] + r * dir[2],
                    ];
                    let nf = sh.nf;
                    let mut v = vec![0.0; nc * nf];
                    eval_ao(&b, &[si], &[p[0]], &[p[1]], &[p[2]], order, &mut v);
                    let m = v.iter().fold(0.0f64, |a, x| a.max(x.abs()));
                    assert!(
                        m <= thresh,
                        "shell {si} order {order}: {m:e} at rcut {}",
                        sh.rcut
                    );
                }
            }
            // a far block: kept set equals the eval_ao-based selection
            let np = 16;
            let (x, y, z): (Vec<f64>, Vec<f64>, Vec<f64>) = (
                (0..np).map(|i| 4.0 + 0.3 * i as f64).collect(),
                (0..np).map(|i| 2.0 - 0.2 * i as f64).collect(),
                vec![1.0; np],
            );
            let lo = [x[0], y[np - 1], 1.0];
            let hi = [x[np - 1], y[0], 1.0];
            let mut out = vec![0.0; b.nbf * nc * np];
            let (mut idx, mut own) = (vec![], vec![]);
            c.eval(
                &shells, &x, &y, &z, np, np, &lo, &hi, &mut out, &mut idx, &mut own,
            );
            let mut refv = vec![0.0; nc * b.nbf * np];
            eval_ao(&b, &shells, &x, &y, &z, order, &mut refv);
            let want: Vec<usize> = (0..b.nbf)
                .filter(|&f| {
                    (0..nc).any(|d| {
                        refv[(d * b.nbf + f) * np..(d * b.nbf + f + 1) * np]
                            .iter()
                            .any(|v| v.abs() > thresh)
                    })
                })
                .collect();
            assert_eq!(idx, want, "order {order}");
        }
    }
}
