//! Semi-local effective core potentials (ECPs): data and the one-electron ECP matrix.
//!
//! An ECP on center C replaces the core electrons of its atom by the operator
//!
//!   U = U_L(r) + sum_{l<L} sum_m |Y_lm> U_l(r) <Y_lm|,     r = |r - C|,
//!
//! with radial channels U(r) = sum_k d_k r^n_k exp(-zeta_k r^2) (all def2 ECPs have n_k = 0).
//! U_L is the local channel (the highest angular momentum in the basis set file) and U_l the
//! semi-local ones, stored as U_l - U_L as in the Basis Set Exchange, NWChem, PySCF and Psi4.
//! Two kinds of integrals arise.
//!
//! Type 1, <a|U_L|b>, is a three-center integral. For Gaussian terms (n = 0) it is a
//! three-center overlap and for n = -1 a Gaussian-damped Coulomb integral; both are exact in the
//! McMurchie-Davidson form used for the nuclear attraction: the Hermite expansion of the pair
//! (E coefficients of [`crate::core::hermite`]) contracted with the Hermite tensor R_tuv(P - C) of
//! the operator, which [`r_tensor`] builds from R^(m)_000 = (-2 mu)^m exp(-mu |P - C|^2)
//! (pi/(p + zeta))^(3/2), mu = p zeta/(p + zeta), for n = 0, and from Boys functions for
//! n = -1. There is no quadrature in type 1 (libcint and libecpint integrate it numerically).
//!
//! Type 2, the semi-local projectors, follows the angular expansion of McMurchie and Davidson
//! (J. Comput. Phys. 44, 289 (1981)) in the form of Flores-Moreno et al. (J. Comput. Chem. 27,
//! 1009 (2006)) used by libcint: a Cartesian Gaussian about A is expanded about C,
//!
//!   phi(r) = sum_ijk B_ijk x^i y^j z^k 4 pi sum_lam F_lam(r) sum_mu Y_lam,mu(r^) Y_lam,mu(A^),
//!   F_lam(r) = sum_p c_p exp(-a_p (r - |A|)^2) i~_lam(2 a_p |A| r),
//!
//! (i~ the exponentially scaled modified spherical Bessel function of the first kind), so its
//! projections f_lm(r) = <Y_lm|phi>(r) on spheres about C follow from angular integrals of a
//! monomial times two real spherical harmonics, tabulated once ([`gaunt`]). libcint
//! integrates every shell pair on its own adaptive radial grid. Here the projections of every
//! basis function are evaluated once per ECP center on one radial grid shared by all
//! functions, so that
//!
//!   <a|U_sl|b> = sum_l sum_g w_g r_g^2 U_l(r_g) sum_m f^a_lm(r_g) f^b_lm(r_g)
//!
//! is a single matrix product over (l, m, g) for all pairs of the center (and the derivative
//! integrals of [`crate::one_e::ecp_deriv`] are products of projections of differentiated functions).
//! The radial grid is the transformed Gauss-Chebyshev rule of Pérez-Jordá et al. with libcint's
//! map r = 1 - log2(1 + x). Its levels (2^k - 1 points) are nested: the matrix on level k is
//! compared with the one on the embedded level k - 1 (no extra work) and the level raised until
//! they agree to [`LEVEL_TOL`]; on def2 ECPs level 7 (127 points, ~80 inside the ECP) suffices
//! and agrees with closed forms of one-center integrals to ~1e-15 relative.

use crate::basis::{cart_components, dfact, ncart, BasisDef, BasisError, BasisSet, Shell};
use crate::contract::AoMatrixMut;
use crate::core::boys;
use crate::core::hermite::{nherm, r_tensor};
use crate::pairs::PairSet;
use crate::potential::{matrix_block, EcpLocal};
use linus_num::simd::{Lanes, Portable};
use rayon::prelude::*;
use std::f64::consts::PI;
use std::sync::OnceLock;

// ---------------------------------------------------------------------------------------
// Data
// ---------------------------------------------------------------------------------------

/// One term d r^rpow exp(-zeta r^2) of a radial ECP channel.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EcpTerm {
    pub rpow: i32,
    pub zeta: f64,
    pub coef: f64,
}

/// The semi-local ECP of one element (see the module docs).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct EcpDef {
    /// Core electrons replaced.
    pub ncore: u32,
    /// Local channel U_L.
    pub local: Vec<EcpTerm>,
    /// Semi-local channels U_l - U_L, l = 0..L.
    pub semilocal: Vec<Vec<EcpTerm>>,
}

/// Value of a radial channel at r.
#[inline]
pub fn channel_value(terms: &[EcpTerm], r: f64) -> f64 {
    let r2 = r * r;
    terms
        .iter()
        .map(|t| t.coef * r.powi(t.rpow) * (-t.zeta * r2).exp())
        .sum()
}

/// An ECP placed on an atom.
#[derive(Clone, Debug)]
pub struct EcpCenter {
    pub atom: usize,
    pub center: [f64; 3],
    pub def: EcpDef,
}

/// The ECPs of a molecule (empty when no atom has one).
#[derive(Clone, Debug, Default)]
pub struct EcpSet {
    pub centers: Vec<EcpCenter>,
}

impl EcpSet {
    /// ECPs of the atoms (atomic numbers `z`, `positions` in bohr) that `def` gives one.
    pub fn new(def: &BasisDef, z: &[u32], positions: &[[f64; 3]]) -> Result<Self, BasisError> {
        let mut centers = Vec::new();
        for (atom, (&zi, &c)) in z.iter().zip(positions).enumerate() {
            let Some(e) = def.ecp.get(&zi) else { continue };
            if let Some(t) = e.local.iter().find(|t| t.rpow != 0 && t.rpow != -1) {
                return Err(BasisError::Unsupported(format!("ECP of Z={zi}: local term r^{} (only r^0 and r^-1 are implemented): LANL-type ECPs are not supported; def2 basis sets carry supported ECPs", t.rpow)));
            }
            if let Some(t) = e.semilocal.iter().flatten().find(|t| t.rpow < -2) {
                return Err(BasisError::Unsupported(format!("ECP of Z={zi}: semi-local term r^{}: LANL-type ECPs are not supported; def2 basis sets carry supported ECPs", t.rpow)));
            }
            if e.semilocal.len() > LMAX_ECP {
                return Err(BasisError::Unsupported(format!(
                    "ECP of Z={zi}: semi-local channels up to l={} (at most {})",
                    e.semilocal.len() - 1,
                    LMAX_ECP - 1
                )));
            }
            centers.push(EcpCenter {
                atom,
                center: c,
                def: e.clone(),
            });
        }
        Ok(EcpSet { centers })
    }

    pub fn is_empty(&self) -> bool {
        self.centers.is_empty()
    }

    /// Move the ECPs with their atoms.
    pub fn set_positions(&mut self, positions: &[[f64; 3]]) {
        for c in &mut self.centers {
            c.center = positions[c.atom];
        }
    }

    /// Core electrons removed per atom (0 for atoms without an ECP).
    pub fn core_electrons(&self, natom: usize) -> Vec<u32> {
        let mut n = vec![0; natom];
        for c in &self.centers {
            n[c.atom] = c.def.ncore;
        }
        n
    }
}

// ---------------------------------------------------------------------------------------
// Radial grid
// ---------------------------------------------------------------------------------------

/// First radial grid level tried (2^k - 1 points).
pub const LEVEL_START: u32 = 7;
/// Highest level (2047 points).
pub const LEVEL_MAX: u32 = 11;
/// Largest |V(level k) - V(level k-1)| element accepted for the semi-local matrix of a center.
/// Convergence is exponential (the error drops by 1e3-1e6 per level at this point), so the
/// accepted level is accurate far below this difference.
pub const LEVEL_TOL: f64 = 1e-12;
/// The first level tried: [`LEVEL_START`], or `$LINUS_ECP_LEVEL` (3-[`LEVEL_MAX`]; a study switch
/// of development builds, `crate::core::switches`).
pub fn start_level() -> u32 {
    static L: OnceLock<u32> = OnceLock::new();
    *L.get_or_init(|| {
        let Some(v) = crate::core::switches::study_switch("LINUS_ECP_LEVEL") else {
            return LEVEL_START;
        };
        match v.trim().parse::<u32>() {
            Ok(l) if (3..=LEVEL_MAX).contains(&l) => l,
            _ => {
                crate::core::switches::ignore_invalid(
                    "LINUS_ECP_LEVEL",
                    &v,
                    &format!("use a level from 3 to {LEVEL_MAX}"),
                );
                LEVEL_START
            }
        }
    })
}

/// Grid points where every |w r^2 U_l(r)| is below this are dropped.
const W_CUT: f64 = 1e-22;

/// Grid of one ECP center: points (those of level k - 1 first), radial weights times
/// r^2 U_l(r) per semi-local channel.
#[derive(Clone, Debug)]
pub struct CenterGrid {
    pub level: u32,
    pub r: Vec<f64>,
    /// w\[l\]\[g\] = w_g r_g^2 U_l(r_g)
    pub w: Vec<Vec<f64>>,
    /// The first `ncoarse` points form level - 1 (with twice their weight there).
    pub ncoarse: usize,
}

impl CenterGrid {
    pub fn ng(&self) -> usize {
        self.r.len()
    }
    pub fn nl(&self) -> usize {
        self.w.len()
    }
    /// Columns of the projection matrices: sum_l (2l + 1) ng.
    pub fn ncol(&self) -> usize {
        (0..self.nl()).map(|l| 2 * l + 1).sum::<usize>() * self.ng()
    }
    /// Columns of the coarse points (they come first, see [`Self::col_index`]).
    pub fn ncol_coarse(&self) -> usize {
        (0..self.nl()).map(|l| 2 * l + 1).sum::<usize>() * self.ncoarse
    }
    /// Column of (l, m, g): coarse points \[l\]\[m\]\[g < nc\] first, then \[l\]\[m\][g - nc].
    #[inline]
    pub fn col_index(&self, l: usize, m: usize, g: usize) -> usize {
        let nc = self.ncoarse;
        let nf = self.ng() - nc;
        let lm = l * l + m; // sum_{l'<l} (2l'+1) = l^2
        if g < nc {
            lm * nc + g
        } else {
            self.ncol_coarse() + lm * nf + (g - nc)
        }
    }
    /// Weight of every column (w\[l\]\[g\] repeated over m).
    pub fn col_weights(&self) -> Vec<f64> {
        let mut out = vec![0.0; self.ncol()];
        for l in 0..self.nl() {
            for m in 0..2 * l + 1 {
                for g in 0..self.ng() {
                    out[self.col_index(l, m, g)] = self.w[l][g];
                }
            }
        }
        out
    }
}

/// Radial points and weights of level k (2^k - 1 points of the transformed Gauss-Chebyshev rule
/// on (-1, 1), mapped by r = 1 - log2(1 + x)), as (r, w, coarse) with coarse = the point belongs
/// to level k - 1.
fn radial_level(level: u32) -> Vec<(f64, f64, bool)> {
    let n = (1usize << level) - 1;
    let np1 = (n + 1) as f64;
    let ln2 = std::f64::consts::LN_2;
    let mut out = Vec::with_capacity(n);
    for i in 1..=n {
        let t = i as f64 * PI / np1;
        let (s, c) = t.sin_cos();
        // 1 + x, x = (n + 1 - 2i)/(n + 1) + (2/pi)(1 + 2/3 sin^2 t) sin t cos t
        let opx = 2.0 * (n + 1 - i) as f64 / np1 + 2.0 / PI * (1.0 + 2.0 / 3.0 * s * s) * s * c;
        if opx < 1e-9 {
            continue; // r > 31 bohr (and 1 + x loses its digits there)
        }
        let w = 16.0 / (3.0 * np1) * s.powi(4);
        out.push((1.0 - opx.ln() / ln2, w / (opx * ln2), i % 2 == 0));
    }
    out
}

/// The grid of ECP center `c` at `level`.
pub fn center_grid(c: &EcpCenter, level: u32) -> CenterGrid {
    let pts = radial_level(level);
    let nl = c.def.semilocal.len();
    let mut keep: Vec<(f64, Vec<f64>, bool)> = Vec::new();
    for (r, w, coarse) in pts {
        let wl: Vec<f64> = c
            .def
            .semilocal
            .iter()
            .map(|t| w * r * r * channel_value(t, r))
            .collect();
        if wl.iter().any(|x| x.abs() > W_CUT) {
            keep.push((r, wl, coarse));
        }
    }
    // coarse points first (stable)
    keep.sort_by_key(|p| !p.2);
    let ncoarse = keep.iter().filter(|p| p.2).count();
    let mut w = vec![Vec::with_capacity(keep.len()); nl];
    let mut r = Vec::with_capacity(keep.len());
    for (ri, wl, _) in keep {
        r.push(ri);
        for l in 0..nl {
            w[l].push(wl[l]);
        }
    }
    CenterGrid {
        level,
        r,
        w,
        ncoarse,
    }
}

// ---------------------------------------------------------------------------------------
// Bessel functions
// ---------------------------------------------------------------------------------------

/// exp(-z) i_l(z), l = 0..=lmax, z >= 0 (i_l the modified spherical Bessel function of the
/// first kind). Power series for the two highest orders and downward recursion for z <= 32
/// (stable); closed forms of i_0, i_1 and upward recursion above (stable there for l <= 16).
/// Relative error ~1e-15.
pub fn sph_ine(lmax: usize, z: f64, out: &mut [f64]) {
    debug_assert!(out.len() > lmax);
    if z == 0.0 {
        out[0] = 1.0;
        out[1..=lmax].iter_mut().for_each(|x| *x = 0.0);
        return;
    }
    if z <= 32.0 {
        let ez = (-z).exp();
        let h = 0.5 * z * z;
        // series for one order l: z^l/(2l+1)!! sum_k (z^2/2)^k / (k! (2l+3)(2l+5)...(2l+2k+1))
        let series = |l: usize| -> f64 {
            let mut pre = 1.0;
            for k in 1..=l {
                pre *= z / (2 * k + 1) as f64;
            }
            let (mut t, mut s) = (1.0f64, 1.0f64);
            let mut k = 0usize;
            loop {
                t *= h / (((k + 1) * (2 * l + 2 * k + 3)) as f64);
                let s2 = s + t;
                k += 1;
                if s2 == s {
                    break;
                }
                s = s2;
            }
            pre * s * ez
        };
        if z < 1e-3 {
            // every order from its own (short) series: the recursion divides by z
            for (l, o) in out[..=lmax].iter_mut().enumerate() {
                *o = series(l);
            }
            return;
        }
        let mut hi = series(lmax + 1);
        let mut cur = series(lmax);
        out[lmax] = cur;
        for l in (1..=lmax).rev() {
            let lo = hi + (2 * l + 1) as f64 / z * cur;
            out[l - 1] = lo;
            hi = cur;
            cur = lo;
        }
    } else {
        let e2 = (-2.0 * z).exp();
        let iz = 1.0 / z;
        out[0] = 0.5 * (1.0 - e2) * iz;
        if lmax >= 1 {
            out[1] = 0.5 * (1.0 + e2) * iz - 0.5 * (1.0 - e2) * iz * iz;
        }
        for l in 1..lmax {
            out[l + 1] = out[l - 1] - (2 * l + 1) as f64 * iz * out[l];
        }
    }
}

// ---------------------------------------------------------------------------------------
// Angular tables
// ---------------------------------------------------------------------------------------

/// Highest monomial degree in the expansion of a (differentiated) basis function about C, which
/// is l + derivative order: basis functions up to l = 9 for energies, 8 for gradients and 7 for
/// Hessians.
pub const NMAX: usize = 9;
/// Semi-local channels l < LMAX_ECP.
pub const LMAX_ECP: usize = 6;
const LYMAX: usize = NMAX + LMAX_ECP;

/// Real spherical harmonics normalized on the unit sphere as coefficients over the Cartesian
/// monomials of degree l: (2l + 1) x ncart(l), row-major (the rows of [`crate::core::solid_harmonics`]
/// scaled by sqrt((2l + 1)/(4 pi))).
fn ylm_cart(l: usize) -> &'static [f64] {
    static T: OnceLock<Vec<Vec<f64>>> = OnceLock::new();
    &T.get_or_init(|| {
        (0..=LYMAX)
            .map(|l| {
                let f = ((2 * l + 1) as f64 / (4.0 * PI)).sqrt();
                crate::core::solid_harmonics::cart2sph(l)
                    .into_iter()
                    .map(|x| x * f)
                    .collect()
            })
            .collect()
    })[l]
}

/// Integral of x^a y^b z^c over the unit sphere.
fn sphere_int(a: usize, b: usize, c: usize) -> f64 {
    if a % 2 == 1 || b % 2 == 1 || c % 2 == 1 {
        return 0.0;
    }
    4.0 * PI * dfact(a as i64 - 1) * dfact(b as i64 - 1) * dfact(c as i64 - 1)
        / dfact((a + b + c) as i64 + 1)
}

/// Index of the Cartesian component (lx, ly, lz) in [`cart_components`] order.
#[inline]
pub(crate) fn cart_index(lx: usize, lz: usize, l: usize) -> usize {
    let k = l - lx;
    k * (k + 1) / 2 + lz
}

/// Angular integrals int x^i y^j z^k Y_lm Y_lam,mu dOmega for the monomials of degree n, one
/// channel l and the lam = n + l, n + l - 2, ... >= 0 that can be nonzero.
pub struct Gaunt {
    /// lam values (descending parity-allowed list, ascending order)
    pub lams: Vec<usize>,
    /// offset of each lam in the mu dimension
    pub off: Vec<usize>,
    /// sum over lams of 2 lam + 1
    pub nmu: usize,
    /// \[ijk\]\[m\][off(lam) + mu]
    pub data: Vec<f64>,
}

/// Cached [`Gaunt`] table of (n, l).
pub fn gaunt(n: usize, l: usize) -> &'static Gaunt {
    assert!(
        n <= NMAX && l < LMAX_ECP,
        "ECP angular table out of range (n={n}, l={l})"
    );
    static T: OnceLock<Vec<OnceLock<Gaunt>>> = OnceLock::new();
    let t = T.get_or_init(|| {
        (0..(NMAX + 1) * LMAX_ECP)
            .map(|_| OnceLock::new())
            .collect()
    });
    t[n * LMAX_ECP + l].get_or_init(|| build_gaunt(n, l))
}

fn build_gaunt(n: usize, l: usize) -> Gaunt {
    let lams: Vec<usize> = ((n + l) % 2..=n + l).step_by(2).collect();
    let mut off = Vec::with_capacity(lams.len());
    let mut nmu = 0;
    for &lam in &lams {
        off.push(nmu);
        nmu += 2 * lam + 1;
    }
    let cn = cart_components(n);
    let cl = cart_components(l);
    let yl = ylm_cart(l);
    let nm = 2 * l + 1;
    let mut data = vec![0.0; cn.len() * nm * nmu];
    for (li, &lam) in lams.iter().enumerate() {
        let clam = cart_components(lam);
        let ylam = ylm_cart(lam);
        let nlam = 2 * lam + 1;
        for (ii, p) in cn.iter().enumerate() {
            for m in 0..nm {
                // z[b] = sum_a Y_lm[a] I(p + a + b)
                let mut zb = vec![0.0; clam.len()];
                for (ia, a) in cl.iter().enumerate() {
                    let y = yl[m * cl.len() + ia];
                    if y == 0.0 {
                        continue;
                    }
                    for (ib, b) in clam.iter().enumerate() {
                        zb[ib] += y * sphere_int(
                            p[0] + a[0] + b[0],
                            p[1] + a[1] + b[1],
                            p[2] + a[2] + b[2],
                        );
                    }
                }
                for mu in 0..nlam {
                    let mut s = 0.0;
                    for (ib, &z) in zb.iter().enumerate() {
                        s += z * ylam[mu * clam.len() + ib];
                    }
                    data[(ii * nm + m) * nmu + off[li] + mu] = s;
                }
            }
        }
    }
    Gaunt {
        lams,
        off,
        nmu,
        data,
    }
}

// ---------------------------------------------------------------------------------------
// Projections of basis functions
// ---------------------------------------------------------------------------------------

/// Scratch space of [`cart_proj`] and [`shell_proj`].
#[derive(Default)]
pub struct ProjWork {
    f: Vec<f64>,
    bes: Vec<f64>,
    ylm: Vec<f64>,
    omega: Vec<f64>,
    ang: Vec<f64>,
    rf: Vec<f64>,
    mono: Vec<f64>,
    bpow: Vec<f64>,
    /// shell_proj's Cartesian derivative projections of one channel
    cart: Vec<f64>,
}

/// The scratch of one thread's shell projections: [`shell_proj`]'s work space and its
/// per-variant Cartesian projections.
#[derive(Default)]
pub struct ProjScratch {
    wk: ProjWork,
    cbuf: Vec<Vec<Vec<f64>>>,
}

/// Cartesian projections onto Y_lm about C of a Cartesian shell of angular momentum `lsh` with
/// primitives (`exps`, `coefs`; coefs as stored, possibly scaled for derivatives) centered at
/// C + `rel`, on the points `r`, for the channels l = 0..nl: out\[l\] = [ncart(lsh)]\[2l+1\]\[ng\]
/// (resized and overwritten).
#[allow(clippy::needless_range_loop, reason = "indexes several arrays in step")]
#[allow(clippy::too_many_arguments, reason = "a kernel's arrays and sizes")]
pub fn cart_proj(
    lsh: usize,
    exps: &[f64],
    coefs: &[f64],
    rel: [f64; 3],
    nl: usize,
    r: &[f64],
    out: &mut [Vec<f64>],
    wk: &mut ProjWork,
) {
    assert!(
        lsh <= NMAX,
        "ECP integrals need basis functions up to l = {} (with derivatives)",
        NMAX
    );
    let ng = r.len();
    let nc = ncart(lsh);
    if nl == 0 {
        return;
    }
    let lammax = lsh + nl - 1;
    let a = (rel[0] * rel[0] + rel[1] * rel[1] + rel[2] * rel[2]).sqrt();
    let on_center = a < 1e-12;
    // radial functions F[lam][g]
    let nlam = if on_center { 1 } else { lammax + 1 };
    wk.f.clear();
    wk.f.resize(nlam * ng, 0.0);
    wk.bes.resize(lammax + 2, 0.0);
    for (&al, &c) in exps.iter().zip(coefs) {
        if c == 0.0 {
            continue;
        }
        let cut = 46.0 + c.abs().ln().max(0.0);
        for (g, &rg) in r.iter().enumerate() {
            let d = rg - a;
            let arg = al * d * d;
            if arg > cut {
                continue;
            }
            let e = c * (-arg).exp();
            if on_center {
                wk.f[g] += e;
            } else {
                sph_ine(lammax, 2.0 * al * a * rg, &mut wk.bes);
                for lam in 0..=lammax {
                    wk.f[lam * ng + g] += e * wk.bes[lam];
                }
            }
        }
    }
    // spherical harmonics at A^ (any direction on the center)
    let u = if on_center {
        [0.0, 0.0, 1.0]
    } else {
        [rel[0] / a, rel[1] / a, rel[2] / a]
    };
    let ylen: usize = (0..nlam).map(|l| 2 * l + 1).sum();
    wk.ylm.clear();
    wk.ylm.resize(ylen, 0.0);
    {
        let mut o = 0;
        for lam in 0..nlam {
            let cl = cart_components(lam);
            let yc = ylm_cart(lam);
            wk.mono.clear();
            wk.mono.extend(
                cl.iter().map(|p| {
                    u[0].powi(p[0] as i32) * u[1].powi(p[1] as i32) * u[2].powi(p[2] as i32)
                }),
            );
            for mu in 0..2 * lam + 1 {
                let mut s = 0.0;
                for (ib, &m) in wk.mono.iter().enumerate() {
                    s += yc[mu * cl.len() + ib] * m;
                }
                wk.ylm[o + mu] = s;
            }
            o += 2 * lam + 1;
        }
    }
    // r^n F[lam] for n <= lsh, lam < nlam: rf[(n * nlam + lam) * ng + g]
    wk.rf.clear();
    wk.rf.resize((lsh + 1) * nlam * ng, 0.0);
    for n in 0..=lsh {
        for lam in 0..nlam {
            let dst = &mut wk.rf[(n * nlam + lam) * ng..(n * nlam + lam + 1) * ng];
            let src = &wk.f[lam * ng..(lam + 1) * ng];
            for g in 0..ng {
                dst[g] = src[g] * r[g].powi(n as i32);
            }
        }
    }
    // powers of -rel and binomials for B_ijk
    wk.bpow.clear();
    wk.bpow.resize(3 * (lsh + 1), 0.0);
    for d in 0..3 {
        let mut p = 1.0;
        for k in 0..=lsh {
            wk.bpow[d * (lsh + 1) + k] = p;
            p *= -rel[d];
        }
    }
    let comps = cart_components(lsh);
    for l in 0..nl {
        let nm = 2 * l + 1;
        // ang[comp][m][n * nlam + lam]
        let nnl = (lsh + 1) * nlam;
        wk.ang.clear();
        wk.ang.resize(nc * nm * nnl, 0.0);
        // omega for each degree n: [ijk][m][lam] = sum_mu gaunt * Y_lam,mu(A^)
        for n in 0..=lsh {
            let gt = gaunt(n, l);
            let cn = ncart(n);
            wk.omega.clear();
            wk.omega.resize(cn * nm * nlam, 0.0);
            for ii in 0..cn {
                for m in 0..nm {
                    let row = &gt.data[(ii * nm + m) * gt.nmu..(ii * nm + m + 1) * gt.nmu];
                    for (li, &lam) in gt.lams.iter().enumerate() {
                        if lam >= nlam {
                            continue;
                        }
                        let yo: usize = lam * lam; // offset of lam in ylm: sum_{l'<lam} (2l'+1)
                        let mut s = 0.0;
                        for mu in 0..2 * lam + 1 {
                            s += row[gt.off[li] + mu] * wk.ylm[yo + mu];
                        }
                        wk.omega[(ii * nm + m) * nlam + lam] = s;
                    }
                }
            }
            // B coefficients of the monomials of degree n in each component
            for (ic, p) in comps.iter().enumerate() {
                let (ax, ay, az) = (p[0], p[1], p[2]);
                for i in 0..=ax.min(n) {
                    for j in 0..=ay.min(n - i) {
                        let k = n - i - j;
                        if k > az {
                            continue;
                        }
                        let b = binom(ax, i)
                            * wk.bpow[ax - i]
                            * binom(ay, j)
                            * wk.bpow[(lsh + 1) + ay - j]
                            * binom(az, k)
                            * wk.bpow[2 * (lsh + 1) + az - k];
                        if b == 0.0 {
                            continue;
                        }
                        let ii = cart_index(i, k, n);
                        for m in 0..nm {
                            let src = &wk.omega[(ii * nm + m) * nlam..(ii * nm + m + 1) * nlam];
                            let dst = &mut wk.ang[(ic * nm + m) * nnl + n * nlam
                                ..(ic * nm + m) * nnl + (n + 1) * nlam];
                            for (d, s) in dst.iter_mut().zip(src) {
                                *d += b * s;
                            }
                        }
                    }
                }
            }
        }
        // out[comp][m][g] = 4 pi sum_(n, lam) ang * rf
        let o = &mut out[l];
        o.clear();
        o.resize(nc * nm * ng, 0.0);
        for cm in 0..nc * nm {
            let dst = &mut o[cm * ng..(cm + 1) * ng];
            for q in 0..nnl {
                let c = 4.0 * PI * wk.ang[cm * nnl + q];
                if c == 0.0 {
                    continue;
                }
                let src = &wk.rf[q * ng..(q + 1) * ng];
                for (d, s) in dst.iter_mut().zip(src) {
                    *d += c * s;
                }
            }
        }
    }
}

#[inline]
fn binom(n: usize, k: usize) -> f64 {
    let mut r = 1.0;
    for i in 0..k {
        r = r * (n - i) as f64 / (i + 1) as f64;
    }
    r
}

/// Number of derivative components of order `order` (value; x, y, z; xx, xy, xz, yy, yz, zz).
pub fn nderiv_comps(order: usize) -> usize {
    match order {
        0 => 1,
        1 => 4,
        _ => 10,
    }
}

/// Projections of the functions of shell `sh` (and, for order >= 1, of their derivatives with
/// respect to the shell center: components 1..4 = d/dA_x,y,z, 4..10 = d2/dA_k dA_l for kl = xx,
/// xy, xz, yy, yz, zz) onto Y_lm about the ECP center, on the grid `grid`: written into
/// `out[d][f * ldo + col]` for the shell's function f (columns as [`CenterGrid::col_index`]).
#[allow(clippy::too_many_arguments, reason = "a kernel's arrays and sizes")]
#[allow(
    clippy::needless_range_loop,
    reason = "the channel indexes every variant's projections and the columns"
)]
pub fn shell_proj(
    sh: &Shell,
    ecp_center: [f64; 3],
    grid: &CenterGrid,
    order: usize,
    out: &mut [&mut [f64]],
    ldo: usize,
    wk: &mut ProjWork,
    cbuf: &mut Vec<Vec<Vec<f64>>>,
) {
    let rel = [
        sh.center[0] - ecp_center[0],
        sh.center[1] - ecp_center[1],
        sh.center[2] - ecp_center[2],
    ];
    let nl = grid.nl();
    let ng = grid.ng();
    let l = sh.l;
    // variants (angular momentum, power k of 2 alpha in the coefficients)
    let variants: Vec<(usize, i32)> = match order {
        0 => vec![(l, 0)],
        1 => {
            let mut v = vec![(l, 0), (l + 1, 1)];
            if l >= 1 {
                v.push((l - 1, 0));
            }
            v
        }
        _ => {
            let mut v = vec![(l, 0), (l + 1, 1), (l + 2, 2), (l, 1)];
            if l >= 1 {
                v.push((l - 1, 0));
            }
            if l >= 2 {
                v.push((l - 2, 0));
            }
            v
        }
    };
    cbuf.resize_with(variants.len(), Vec::new);
    let mut sc = Vec::with_capacity(sh.exps.len());
    for (vi, &(lv, k)) in variants.iter().enumerate() {
        sc.clear();
        sc.extend(
            sh.exps
                .iter()
                .zip(&sh.coefs)
                .map(|(&a, &c)| c * (2.0 * a).powi(k)),
        );
        cbuf[vi].resize_with(nl, Vec::new);
        cart_proj(lv, &sh.exps, &sc, rel, nl, &grid.r, &mut cbuf[vi], wk);
    }
    let find = |lv: usize, k: i32| variants.iter().position(|&v| v == (lv, k));
    let comps = cart_components(l);
    let nc = comps.len();
    let t = BasisSet::shell_transform_ref(sh);
    let nf = sh.nfunc();
    let nd = nderiv_comps(order);
    // Cartesian derivative projections for one l: cart[d][comp][m][g] (the work space's
    // buffer, reused)
    let mut cart = std::mem::take(&mut wk.cart);
    let (nco, ncc) = (grid.ncoarse, grid.ncol_coarse());
    let nfine = ng - nco;
    for lc in 0..nl {
        let nm = 2 * lc + 1;
        let blk = nm * ng;
        cart.clear();
        cart.resize(nd * nc * blk, 0.0);
        let get = |lv: usize, k: i32, lx: usize, ly: usize, lz: usize| -> Option<&[f64]> {
            let vi = find(lv, k)?;
            let i = cart_index(lx, lz, lv);
            debug_assert_eq!(lx + ly + lz, lv);
            Some(&cbuf[vi][lc][i * blk..(i + 1) * blk])
        };
        for (ic, p) in comps.iter().enumerate() {
            let a = [p[0], p[1], p[2]];
            // value
            cart[ic * blk..(ic + 1) * blk].copy_from_slice(get(l, 0, a[0], a[1], a[2]).unwrap());
            if order >= 1 {
                for x in 0..3 {
                    let dst = &mut cart[((1 + x) * nc + ic) * blk..((1 + x) * nc + ic + 1) * blk];
                    let mut up = a;
                    up[x] += 1;
                    let s = get(l + 1, 1, up[0], up[1], up[2]).unwrap();
                    dst.copy_from_slice(s);
                    if a[x] >= 1 {
                        let mut dn = a;
                        dn[x] -= 1;
                        let s = get(l - 1, 0, dn[0], dn[1], dn[2]).unwrap();
                        let f = a[x] as f64;
                        for (d, v) in dst.iter_mut().zip(s) {
                            *d -= f * v;
                        }
                    }
                }
            }
            if order >= 2 {
                let mut kl = 0;
                for x in 0..3 {
                    for y in x..3 {
                        let dst_i = ((4 + kl) * nc + ic) * blk;
                        kl += 1;
                        // terms: (shift vector, variant (lv, k), factor)
                        let mut terms: Vec<([i64; 3], i32, f64)> = Vec::new();
                        let ax = a[x] as f64;
                        let ay = a[y] as f64;
                        if x == y {
                            let mut s = [0i64; 3];
                            s[x] = 2;
                            terms.push((s, 2, 1.0));
                            terms.push(([0; 3], 1, -(2.0 * ax + 1.0)));
                            if a[x] >= 2 {
                                let mut s = [0i64; 3];
                                s[x] = -2;
                                terms.push((s, 0, ax * (ax - 1.0)));
                            }
                        } else {
                            let mut s = [0i64; 3];
                            s[x] = 1;
                            s[y] = 1;
                            terms.push((s, 2, 1.0));
                            if a[y] >= 1 {
                                let mut s = [0i64; 3];
                                s[x] = 1;
                                s[y] = -1;
                                terms.push((s, 1, -ay));
                            }
                            if a[x] >= 1 {
                                let mut s = [0i64; 3];
                                s[x] = -1;
                                s[y] = 1;
                                terms.push((s, 1, -ax));
                            }
                            if a[x] >= 1 && a[y] >= 1 {
                                let mut s = [0i64; 3];
                                s[x] = -1;
                                s[y] = -1;
                                terms.push((s, 0, ax * ay));
                            }
                        }
                        for (s, k, f) in terms {
                            let b = [
                                (a[0] as i64 + s[0]) as usize,
                                (a[1] as i64 + s[1]) as usize,
                                (a[2] as i64 + s[2]) as usize,
                            ];
                            let lv = b[0] + b[1] + b[2];
                            let src = get(lv, k, b[0], b[1], b[2]).unwrap();
                            for (d, v) in cart[dst_i..dst_i + blk].iter_mut().zip(src) {
                                *d += f * v;
                            }
                        }
                    }
                }
            }
        }
        // transform to the shell's functions into the columns (CenterGrid::col_index: per
        // (l, m) the coarse points, then the fine ones, each contiguous): every column is
        // sum_ic t[fi][ic] cart[d][ic][m][g] over the nonzero t in component order, summed
        // from zero point by point
        for d in 0..nd {
            for fi in 0..nf {
                let row = &mut out[d][fi * ldo..(fi + 1) * ldo];
                let trow = &t[fi * nc..(fi + 1) * nc];
                for m in 0..nm {
                    let lm = lc * lc + m;
                    let (row_c, row_f) = row.split_at_mut(ncc);
                    let row_c = &mut row_c[lm * nco..(lm + 1) * nco];
                    let row_f = &mut row_f[lm * nfine..(lm + 1) * nfine];
                    row_c.fill(0.0);
                    row_f.fill(0.0);
                    for (ic, &tv) in trow.iter().enumerate() {
                        if tv == 0.0 {
                            continue;
                        }
                        let src = &cart[(d * nc + ic) * blk + m * ng..][..ng];
                        let (src_c, src_f) = src.split_at(nco);
                        for (x, &c) in row_c.iter_mut().zip(src_c) {
                            *x += tv * c;
                        }
                        for (x, &c) in row_f.iter_mut().zip(src_f) {
                            *x += tv * c;
                        }
                    }
                }
            }
        }
    }
    wk.cart = cart;
}

// ---------------------------------------------------------------------------------------
// Active functions and projection matrices of a center
// ---------------------------------------------------------------------------------------

/// Shells whose functions reach the grid of a center (others project to ~0 there).
pub fn active_shells(basis: &BasisSet, c: &EcpCenter, grid: &CenterGrid) -> Vec<usize> {
    let rmax = grid.r.iter().cloned().fold(0.0, f64::max);
    basis
        .shells
        .iter()
        .enumerate()
        .filter(|(_, s)| {
            let d = ((s.center[0] - c.center[0]).powi(2)
                + (s.center[1] - c.center[1]).powi(2)
                + (s.center[2] - c.center[2]).powi(2))
            .sqrt();
            if d <= rmax {
                return true;
            }
            let gap = d - rmax;
            let poly = (d + rmax).powi(s.l as i32 + 2).max(1.0);
            s.exps
                .iter()
                .zip(&s.coefs)
                .any(|(&a, &cf)| a * gap * gap < 50.0 + (cf.abs() * poly).ln())
        })
        .map(|(i, _)| i)
        .collect()
}

/// Projection matrices of the active functions of a center into `mats` (one nrow x ncol
/// row-major matrix per derivative component, at stride nrow * ncol; every element written):
/// rows = active functions (in the order of `shells`), columns = [`CenterGrid::col_index`]
/// (see [`shell_proj`]). The shells are independent: each writes its own rows, on the scratch
/// of the thread that takes it (`work`, at least one per pool thread). Returns (AO index of
/// each row, atom of each row).
pub fn proj_matrices(
    basis: &BasisSet,
    c: &EcpCenter,
    grid: &CenterGrid,
    shells: &[usize],
    order: usize,
    mats: &mut Vec<f64>,
    work: &mut [ProjScratch],
) -> (Vec<usize>, Vec<usize>) {
    let ncol = grid.ncol();
    let nd = nderiv_comps(order);
    let mut rows = Vec::new();
    let mut atoms = Vec::new();
    for &s in shells {
        let sh = &basis.shells[s];
        for f in 0..sh.nfunc() {
            rows.push(sh.ao_start + f);
            atoms.push(sh.atom);
        }
    }
    let stride = rows.len() * ncol;
    let mats = scratch_len(mats, nd * stride);
    // per shell its rows of every matrix
    let mut parts: Vec<Vec<&mut [f64]>> = shells.iter().map(|_| Vec::with_capacity(nd)).collect();
    for m in mats.chunks_exact_mut(stride.max(1)).take(nd) {
        let mut rest = m;
        for (k, &s) in shells.iter().enumerate() {
            let (mine, tail) = rest.split_at_mut(basis.shells[s].nfunc() * ncol);
            parts[k].push(mine);
            rest = tail;
        }
    }
    if stride > 0 {
        crate::engine::schedule::per_part(work, &mut parts, |w, views, k| {
            let sh = &basis.shells[shells[k]];
            shell_proj(
                sh,
                c.center,
                grid,
                order,
                views,
                ncol,
                &mut w.wk,
                &mut w.cbuf,
            );
        });
    }
    (rows, atoms)
}

/// The first `n` values of `buf`, grown (without copying the old contents) to hold them; the
/// caller overwrites them.
pub(crate) fn scratch_len(buf: &mut Vec<f64>, n: usize) -> &mut [f64] {
    if buf.capacity() < n {
        *buf = vec![0.0; n];
    } else if buf.len() < n {
        buf.resize(n, 0.0);
    }
    &mut buf[..n]
}

// ---------------------------------------------------------------------------------------
// Dense kernels (SIMD, run-time ISA dispatch)
// ---------------------------------------------------------------------------------------

linus_num::multiversion_typed!(
    /// out\[i\]\[j\] = sum_{k in k0..k1} a\[i\]\[k\] * w\[k\] * b\[j\]\[k\] for rows i of `a` (lda = ld) and
    /// rows j of `b` (`nb` rows). `out` is (na x nb) row-major. With `w` None, w = 1.
    pub fn gemm_nt_w(a: &[f64], na: usize, b: &[f64], nb: usize, ld: usize, w: Option<&[f64]>, k0: usize, k1: usize, out: &mut [f64]) -> ()
    = gemm_body::<linus_num::simd::Avx512 | linus_num::simd::Avx2 | Portable<false>>);

/// Rows of `b` per cache block of [`gemm_body`] (32 rows of ~1-2k doubles stay in L2 while every
/// row of `a` passes over them).
const JB: usize = 32;

/// Dot products of rows: element (i, j) = sum_k aw[i][k] b[j][k], accumulated in the same order
/// whatever the blocking (lanes over k, then the lane sum, then the k tail), so results do not
/// depend on the micro-kernel shape or on how rows are split between tasks.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn gemm_body<V: Lanes>(
    a: &[f64],
    na: usize,
    b: &[f64],
    nb: usize,
    ld: usize,
    w: Option<&[f64]>,
    k0: usize,
    k1: usize,
    out: &mut [f64],
) {
    const L: usize = linus_num::simd::LANES;
    let nk = k1 - k0;
    let kv = k0 + nk / L * L;
    // weighted copies of the a rows (w folded in once)
    let mut aw = vec![0.0; na * nk];
    for i in 0..na {
        let ar = &a[i * ld + k0..i * ld + k1];
        let dst = &mut aw[i * nk..(i + 1) * nk];
        match w {
            Some(w) => {
                for (x, (&p, &q)) in dst.iter_mut().zip(ar.iter().zip(&w[k0..k1])) {
                    *x = p * q;
                }
            }
            None => dst.copy_from_slice(ar),
        }
    }
    // one output element from its row of aw and a row of b
    let dot1 = |ar: &[f64], br: &[f64]| -> f64 {
        let mut s = V::zero();
        let mut k = k0;
        while k < kv {
            s = V::load(&ar[k - k0..]).mul_add(V::load(&br[k..]), s);
            k += L;
        }
        let mut t = s.hsum();
        for k in kv..k1 {
            t += ar[k - k0] * br[k];
        }
        t
    };
    let mut j0 = 0;
    while j0 < nb {
        let j1 = (j0 + JB).min(nb);
        let mut i = 0;
        // 2 x 4 micro-kernel: 8 accumulators, 6 loads per 8 FMAs
        while i + 2 <= na {
            let (a0, a1) = (&aw[i * nk..(i + 1) * nk], &aw[(i + 1) * nk..(i + 2) * nk]);
            let mut j = j0;
            while j + 4 <= j1 {
                let bs = [
                    &b[j * ld..],
                    &b[(j + 1) * ld..],
                    &b[(j + 2) * ld..],
                    &b[(j + 3) * ld..],
                ];
                let mut s0 = [V::zero(); 4];
                let mut s1 = [V::zero(); 4];
                let mut k = k0;
                while k < kv {
                    let x0 = V::load(&a0[k - k0..]);
                    let x1 = V::load(&a1[k - k0..]);
                    for c in 0..4 {
                        let y = V::load(&bs[c][k..]);
                        s0[c] = x0.mul_add(y, s0[c]);
                        s1[c] = x1.mul_add(y, s1[c]);
                    }
                    k += L;
                }
                for c in 0..4 {
                    let (mut t0, mut t1) = (s0[c].hsum(), s1[c].hsum());
                    for k in kv..k1 {
                        t0 += a0[k - k0] * bs[c][k];
                        t1 += a1[k - k0] * bs[c][k];
                    }
                    out[i * nb + j + c] = t0;
                    out[(i + 1) * nb + j + c] = t1;
                }
                j += 4;
            }
            while j < j1 {
                out[i * nb + j] = dot1(a0, &b[j * ld..]);
                out[(i + 1) * nb + j] = dot1(a1, &b[j * ld..]);
                j += 1;
            }
            i += 2;
        }
        if i < na {
            let ar = &aw[i * nk..(i + 1) * nk];
            for j in j0..j1 {
                out[i * nb + j] = dot1(ar, &b[j * ld..]);
            }
        }
        j0 = j1;
    }
}

/// Rows per parallel task of the dense products.
const ROW_BLOCK: usize = 16;

/// Symmetric product sum_k a\[i\]\[k\] w\[k\] a\[j\]\[k\] over k0..k1 for all rows (n x n, both
/// triangles), rows computed in parallel (deterministic: each element by one task in a fixed
/// order). Only j <= i is computed and mirrored, so the result is exactly symmetric.
pub fn syrk_w(a: &[f64], n: usize, ld: usize, w: &[f64], k0: usize, k1: usize) -> Vec<f64> {
    let mut out = vec![0.0; n * n];
    if n == 0 {
        return out;
    }
    out.par_chunks_mut(ROW_BLOCK * n)
        .enumerate()
        .for_each(|(blk, chunk)| {
            // rows i0..i0+ni against columns 0..i0+ni (the lower triangle and the diagonal block)
            let i0 = blk * ROW_BLOCK;
            let ni = chunk.len() / n;
            let nc = i0 + ni;
            let mut tmp = vec![0.0; ni * nc];
            gemm_nt_w(&a[i0 * ld..], ni, a, nc, ld, Some(w), k0, k1, &mut tmp);
            for ii in 0..ni {
                let i = i0 + ii;
                chunk[ii * n..ii * n + i + 1].copy_from_slice(&tmp[ii * nc..ii * nc + i + 1]);
            }
        });
    for i in 0..n {
        for j in 0..i {
            out[j * n + i] = out[i * n + j];
        }
    }
    out
}

/// out = (a w) b^T for all rows (na x nb), parallel over row blocks of a (deterministic).
#[allow(clippy::too_many_arguments)]
pub fn gemm_nt_par(
    a: &[f64],
    na: usize,
    b: &[f64],
    nb: usize,
    ld: usize,
    w: Option<&[f64]>,
    k0: usize,
    k1: usize,
) -> Vec<f64> {
    let mut out = vec![0.0; na * nb];
    if nb == 0 {
        return out;
    }
    out.par_chunks_mut(ROW_BLOCK * nb)
        .enumerate()
        .for_each(|(blk, chunk)| {
            let i0 = blk * ROW_BLOCK;
            let ni = chunk.len() / nb;
            gemm_nt_w(&a[i0 * ld..], ni, b, nb, ld, w, k0, k1, chunk);
        });
    out
}

// ---------------------------------------------------------------------------------------
// The semi-local (type 2) matrix of a center
// ---------------------------------------------------------------------------------------

/// Semi-local ECP matrix of one center on the active functions (rows/cols = `aos`), with the
/// grid level chosen as in the module docs. Returns (matrix n x n, AO indices, grid level,
/// largest |V(level) - V(level - 1)|).
pub fn semilocal_center(basis: &BasisSet, c: &EcpCenter) -> (Vec<f64>, Vec<usize>, u32, f64) {
    let mut level = start_level();
    let mut mats = Vec::new();
    let mut work: Vec<ProjScratch> = (0..rayon::current_num_threads())
        .map(|_| ProjScratch::default())
        .collect();
    loop {
        let grid = center_grid(c, level);
        let shells = active_shells(basis, c, &grid);
        let (aos, _) = proj_matrices(basis, c, &grid, &shells, 0, &mut mats, &mut work);
        let p = &mats[..aos.len() * grid.ncol()];
        let n = aos.len();
        let ld = grid.ncol();
        let kc = grid.ncol_coarse();
        let w = grid.col_weights();
        let vc = syrk_w(p, n, ld, &w, 0, kc);
        let vf = syrk_w(p, n, ld, &w, kc, ld);
        // V = Vc + Vf; V(level - 1) = 2 Vc
        let mut err: f64 = 0.0;
        let v: Vec<f64> = vc
            .iter()
            .zip(&vf)
            .map(|(&a, &b)| {
                err = err.max((b - a).abs());
                a + b
            })
            .collect();
        if err <= LEVEL_TOL || level >= LEVEL_MAX || grid.ng() == 0 {
            return (v, aos, level, err);
        }
        level += 1;
    }
}

// ---------------------------------------------------------------------------------------
// The local (type 1) part
// ---------------------------------------------------------------------------------------

/// Hermite tensor (degree <= l) of the local channel of the ECP at `center` for one primitive
/// pair (exponent p, center pc): out\[tuv\] += sum_k R^(k)_tuv(P - C) such that
/// int Lambda_tuv(r; p, P) U_L(|r - C|) dr = R_tuv. `rbuf`, `work` hold nherm(l) and 2 nherm(l).
#[allow(clippy::needless_range_loop, reason = "indexes several arrays in step")]
#[allow(clippy::too_many_arguments, reason = "a kernel's arrays and sizes")]
pub fn local_r(
    l: usize,
    p: f64,
    pc: [f64; 3],
    center: [f64; 3],
    terms: &[EcpTerm],
    out: &mut [f64],
    rbuf: &mut [f64],
    work: &mut [f64],
) {
    let pq = [pc[0] - center[0], pc[1] - center[1], pc[2] - center[2]];
    let r2 = pq[0] * pq[0] + pq[1] * pq[1] + pq[2] * pq[2];
    let nh = nherm(l);
    let mut fm = [0.0; boys::MAX_M + 1];
    for t in terms {
        let q = p + t.zeta;
        let mu = p * t.zeta / q;
        let x = mu * r2;
        if x > 60.0 {
            continue;
        }
        let ex = (-x).exp();
        match t.rpow {
            0 => {
                fm[..=l].iter_mut().for_each(|f| *f = ex);
                r_tensor(l, mu, pq, &fm, t.coef * (PI / q).powf(1.5), rbuf, work);
            }
            -1 => {
                // g(R^2) = 2pi/q exp(-a R^2) F_0(b R^2), a = mu, b = p^2/q; R^(n)_000 =
                // 2pi/q (-2p)^n sum_j C(n,j) (a/p)^(n-j) (b/p)^j F_j(b R^2) exp(-a R^2)
                let b = p * p / q;
                let mut fj = [0.0; boys::MAX_M + 1];
                boys::boys(l, b * r2, &mut fj);
                let (ap, bp) = (mu / p, b / p);
                for n in 0..=l {
                    let mut s = 0.0;
                    for j in 0..=n {
                        s += binom(n, j) * ap.powi((n - j) as i32) * bp.powi(j as i32) * fj[j];
                    }
                    fm[n] = s * ex;
                }
                r_tensor(l, p, pq, &fm, t.coef * 2.0 * PI / q, rbuf, work);
            }
            _ => unreachable!("checked in EcpSet::new"),
        }
        for k in 0..nh {
            out[k] += rbuf[k];
        }
    }
}

// ---------------------------------------------------------------------------------------
// The ECP matrix
// ---------------------------------------------------------------------------------------

/// Statistics of the last ECP matrix build (for logs and tests).
#[derive(Clone, Debug, Default)]
pub struct EcpStats {
    /// Per center: (grid level, grid points kept, active functions, |V(k) - V(k-1)|).
    pub centers: Vec<(u32, usize, usize, f64)>,
}

/// The ECP matrix sum_C <a|U_C|b> into `v` (overwritten), with the build statistics
/// ([`crate::one_e::ecp_deriv::EcpPlan::matrix`]).
pub(crate) fn ecp_matrix_into(
    basis: &BasisSet,
    pairs: &PairSet,
    ecp: &EcpSet,
    v: &mut AoMatrixMut<'_>,
) -> EcpStats {
    v.fill(0.0);
    let mut stats = EcpStats::default();
    if ecp.is_empty() {
        return stats;
    }
    // type 1: per shell pair, the local channels as a charge source
    let local: Vec<&EcpCenter> = ecp
        .centers
        .iter()
        .filter(|c| !c.def.local.is_empty())
        .collect();
    let src = EcpLocal { centers: &local };
    let blocks: Vec<(usize, Vec<f64>)> = pairs
        .pairs()
        .par_iter()
        .enumerate()
        .map(|(k, pr)| {
            let mut out = vec![0.0; pr.nab()];
            matrix_block(pr, &src, &mut out);
            (k, out)
        })
        .collect();
    for (k, blk) in blocks {
        let pr = pairs.pair(k);
        let (a, b) = (&basis.shells[pr.sa], &basis.shells[pr.sb]);
        crate::one_e::scatter_sym_ao(v, (a.ao_start, b.ao_start), (pr.na, pr.nb), &blk);
    }
    // type 2: per center
    for c in &ecp.centers {
        if c.def.semilocal.is_empty() {
            continue;
        }
        let (vc, aos, level, err) = semilocal_center(basis, c);
        let n = aos.len();
        for (i, &a) in aos.iter().enumerate() {
            for (j, &b) in aos.iter().enumerate() {
                *v.at(a, b) += vc[i * n + j];
            }
        }
        stats
            .centers
            .push((level, center_grid(c, level).ng(), n, err));
    }
    stats
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bessel_small_and_large() {
        let mut o = [0.0; 12];
        // i_0(z) e^-z = (1 - e^-2z)/(2z); i_1 = (z cosh z - sinh z)/z^2
        for &z in &[1e-6, 0.3, 2.0, 15.0, 31.9, 32.1, 80.0, 1e4] {
            sph_ine(10, z, &mut o);
            let i0 = -(-2.0 * z).exp_m1() / (2.0 * z);
            let i1 = ((1.0 + (-2.0 * z).exp()) / (2.0 * z)) - i0 / z;
            assert!(((o[0] - i0) / i0).abs() < 1e-14, "z={z} {} {}", o[0], i0);
            if z > 1e-3 {
                assert!(((o[1] - i1) / i1).abs() < 1e-12, "z={z} {} {}", o[1], i1);
            }
            // recurrence i_{l-1} - i_{l+1} = (2l+1)/z i_l
            for l in 1..10 {
                let lhs = o[l - 1] - o[l + 1];
                let rhs = (2 * l + 1) as f64 / z * o[l];
                // the difference cancels for z >> l: compare against the size of its terms
                assert!(
                    (lhs - rhs).abs() <= 1e-14 * (o[l - 1].abs() + o[l + 1].abs()) + 1e-300,
                    "z={z} l={l}"
                );
            }
        }
    }

    #[test]
    fn gaunt_orthonormal() {
        // n = 0: int Y_lm Y_lam,mu = delta (times x^0 = 1)
        for l in 0..4 {
            let g = gaunt(0, l);
            assert_eq!(g.lams, (l % 2..=l).step_by(2).collect::<Vec<_>>());
            let li = g.lams.iter().position(|&x| x == l).unwrap();
            for m in 0..2 * l + 1 {
                for mu in 0..2 * l + 1 {
                    let v = g.data[m * g.nmu + g.off[li] + mu];
                    assert!(
                        (v - if m == mu { 1.0 } else { 0.0 }).abs() < 1e-14,
                        "l={l} {m} {mu} {v}"
                    );
                }
            }
        }
    }

    #[test]
    fn grid_levels_nested() {
        let c = EcpCenter {
            atom: 0,
            center: [0.0; 3],
            def: EcpDef {
                ncore: 0,
                local: vec![],
                semilocal: vec![vec![EcpTerm {
                    rpow: 0,
                    zeta: 1.0,
                    coef: 1.0,
                }]],
            },
        };
        let g = center_grid(&c, 7);
        // int r^2 exp(-r^2) dr = sqrt(pi)/4 on both levels
        let exact = PI.sqrt() / 4.0;
        let fine: f64 = g.w[0].iter().sum();
        let coarse: f64 = 2.0 * g.w[0][..g.ncoarse].iter().sum::<f64>();
        assert!((fine - exact).abs() < 1e-14, "{fine} {exact}");
        assert!((coarse - exact).abs() < 1e-9, "{coarse} {exact}");
    }
}
