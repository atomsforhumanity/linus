//! Radial quadratures on [0, inf).
//!
//! Every scheme returns nodes `r` and weights `w` that already include the volume element
//! `4 pi r^2` (PySCF convention: `w = 4 pi r^2 dr`), so that an atomic grid point has weight
//! `w_rad * w_ang` with Lebedev weights summing to one.  Node ordering follows the program a
//! scheme is modelled on (this matters for index-based pruning such as Treutler's).

use crate::elements::{bragg_radius, psi4_bs_radius, treutler_xi};
use serde::{Deserialize, Serialize};
use std::f64::consts::PI;

/// Per-element length scale applied to a radial mapping.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum RadialScale {
    /// Treutler-Ahlrichs xi (JCP 102, 346 (1995), Table; heavier elements as in Psi4/PySCF).
    TreutlerXi,
    /// `factor` x Bragg-Slater radius (PySCF table, bohr).
    Bragg { factor: f64 },
    /// `factor` x Psi4's Bragg-Slater radius (bohr).
    Psi4BraggSlater { factor: f64 },
    /// Same scale for every element.
    Constant(f64),
    /// ORCA 6.0's optimised per-element xi of the `OptM3 with GC (2021)` radial grid for
    /// `AngularGrid` 3, 4 or 6 (DefGrid1/2/3), from [`crate::orca`]'s tables; elements without a
    /// measurement fall back to the Treutler xi.
    OrcaXi { angular_grid: u8 },
}

impl RadialScale {
    pub fn value(&self, z: u32) -> f64 {
        match *self {
            RadialScale::TreutlerXi => treutler_xi(z),
            RadialScale::Bragg { factor } => factor * bragg_radius(z),
            RadialScale::Psi4BraggSlater { factor } => factor * psi4_bs_radius(z),
            RadialScale::Constant(c) => c,
            RadialScale::OrcaXi { angular_grid } => crate::orca::grids()
                .xi(z, angular_grid)
                .unwrap_or_else(|| treutler_xi(z)),
        }
    }
}

/// Chebyshev quadrature kind on [-1, 1] (unit weight function after the usual rescaling).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ChebyshevKind {
    /// Nodes cos((2i-1) pi / 2n), weights (pi/n) sqrt(1-x^2).
    First,
    /// Nodes cos(i pi/(n+1)), weights (pi/(n+1)) sin(i pi/(n+1)).
    Second,
    /// Perez-Jorda, San-Fabian and Moscardo's modified second-kind rule (PRA 45, 4407 (1992); the
    /// Gauss-Chebyshev grid of Krack and Koster, JCP 108, 3226 (1998)): nodes
    /// (n+1-2i)/(n+1) + (2/pi)(1 + (2/3) sin^2 t_i) cos t_i sin t_i, weights 16/(3(n+1)) sin^4 t_i,
    /// t_i = i pi/(n+1). ORCA 6's radial grid (with the M3 mapping).
    PerezJorda,
}

/// Mapping x in (-1, 1) -> r in (0, inf).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum RadialMapping {
    /// Becke: r = R (1+x)/(1-x).
    Becke,
    /// Treutler-Ahlrichs M3: r = (xi/ln2) ln(2/(1-x)).
    TreutlerM3,
    /// Treutler-Ahlrichs M4: r = (xi/ln2) (1+x)^alpha ln(2/(1-x)).
    TreutlerM4 { alpha: f64 },
}

/// Radial quadrature schemes.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum RadialScheme {
    /// Treutler-Ahlrichs M4 on Chebyshev-2 nodes, PySCF's `radi.treutler` formula and ordering
    /// (ascending r). PySCF default: alpha = 0.6, scale = `TreutlerXi`.
    TreutlerAhlrichs { alpha: f64, scale: RadialScale },
    /// Psi4's `TREUTLER`/`AHLRICHS` radial scheme: Chebyshev-2 nodes, M4 (alpha = 0.6) mapping,
    /// scaled by `bs_alpha` x Psi4's Bragg-Slater radius; Psi4's formula and ordering (descending r).
    Psi4Treutler { bs_alpha: f64 },
    /// Becke (JCP 88, 2547): Chebyshev-2 nodes, r = R (1+x)/(1-x), R = Bragg radius (H) or half of it
    /// (other elements) -- PySCF's `radi.becke`.
    Becke,
    /// Mura-Knowles log3 (JCP 104, 9848): far = 7 for Li, Be, Na, Mg, K, Ca, otherwise `far`
    /// (PySCF uses 5.2, the paper 5.0).
    MuraKnowles { far: f64 },
    /// PySCF's `radi.gauss_chebyshev`: Perez-Jorda/Krack-Koster modified Chebyshev-2 with a
    /// logarithmic map (JCP 108, 3226 (1998)).
    KrackKoster,
    /// PySCF's `radi.delley` (log2 grid, JCP 104, 9848; r_outer = 12 bohr in PySCF).
    Delley { r_outer: f64 },
    /// Generic Gauss-Chebyshev quadrature of either kind with a mapping and per-element scale.
    /// Nodes are returned in ascending r.
    GaussChebyshev {
        kind: ChebyshevKind,
        mapping: RadialMapping,
        scale: RadialScale,
    },
}

impl RadialScheme {
    /// PySCF default (`radi.treutler` with atom-specific xi).
    pub const PYSCF_TREUTLER: RadialScheme = RadialScheme::TreutlerAhlrichs {
        alpha: 0.6,
        scale: RadialScale::TreutlerXi,
    };
    /// PySCF Mura-Knowles.
    pub const PYSCF_MURA_KNOWLES: RadialScheme = RadialScheme::MuraKnowles { far: 5.2 };
    /// PySCF Delley.
    pub const PYSCF_DELLEY: RadialScheme = RadialScheme::Delley { r_outer: 12.0 };
}

/// Radial nodes and weights (weights include 4 pi r^2).
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct RadialGrid {
    pub r: Vec<f64>,
    pub w: Vec<f64>,
}

/// Build a radial grid with `n` points for nuclear charge `z`.
pub fn radial_grid(scheme: &RadialScheme, n: usize, z: u32) -> RadialGrid {
    let (r, dr) = match *scheme {
        RadialScheme::TreutlerAhlrichs { alpha, scale } => {
            treutler_ahlrichs(n, alpha, scale.value(z))
        }
        RadialScheme::Psi4Treutler { bs_alpha } => {
            return psi4_treutler(n, psi4_bs_radius(z) * bs_alpha)
        }
        RadialScheme::Becke => becke(n, z),
        RadialScheme::MuraKnowles { far } => mura_knowles(n, z, far),
        RadialScheme::KrackKoster => krack_koster(n),
        RadialScheme::Delley { r_outer } => delley(n, r_outer),
        RadialScheme::GaussChebyshev {
            kind,
            mapping,
            scale,
        } => gauss_chebyshev(n, kind, mapping, scale.value(z)),
    };
    // PySCF: rad_weight = 4*pi * rad**2 * dr
    let fourpi = 4.0 * PI;
    let w = r
        .iter()
        .zip(&dr)
        .map(|(&ri, &di)| fourpi * (ri * ri) * di)
        .collect();
    RadialGrid { r, w }
}

/// PySCF `radi.treutler_ahlrichs` (ascending r).
fn treutler_ahlrichs(n: usize, alpha: f64, xi: f64) -> (Vec<f64>, Vec<f64>) {
    let mut r = vec![0.0; n];
    let mut dr = vec![0.0; n];
    let step = PI / (n as f64 + 1.0);
    let ln2 = xi / 2f64.ln();
    for i in 0..n {
        let t = (i as f64 + 1.0) * step;
        let x = t.cos();
        let p = (1.0 + x).powf(alpha);
        let l = ((1.0 - x) / 2.0).ln();
        r[i] = -ln2 * p * l;
        dr[i] = step * t.sin() * ln2 * p * (-alpha / (1.0 + x) * l + 1.0 / (1.0 - x));
    }
    r.reverse();
    dr.reverse();
    (r, dr)
}

/// Psi4 `RadialGridMgr::makeRadialGrid` with the `TREUTLER` scheme (descending r).
/// Returns weights including 4 pi r^2.
fn psi4_treutler(n: usize, rparam: f64) -> RadialGrid {
    // Psi4's INVLN2 literal (1.4426950408889634074) rounds to LOG2_E
    const INVLN2: f64 = std::f64::consts::LOG2_E;
    let alpha = 0.6;
    let pi_over = PI / (n as f64 + 1.0);
    let rparam3 = rparam * rparam * rparam;
    let mut r = vec![0.0; n];
    let mut w = vec![0.0; n];
    for i in 1..=n {
        let x = (i as f64 * pi_over).cos();
        let wx = pi_over * (1.0 - x * x).sqrt();
        let ri = -INVLN2 * (1.0 + x).powf(alpha) * ((1.0 - x) / 2.0).ln();
        let drdx = INVLN2
            * (1.0 + x).powf(alpha)
            * (-alpha * ((1.0 - x) / 2.0).ln() / (1.0 + x) + 1.0 / (1.0 - x));
        r[i - 1] = ri * rparam;
        w[i - 1] = 4.0 * PI * (wx * drdx * ri * ri * rparam3);
    }
    RadialGrid { r, w }
}

/// PySCF `radi.becke` (descending r).
fn becke(n: usize, z: u32) -> (Vec<f64>, Vec<f64>) {
    let rm = if z == 1 {
        bragg_radius(z)
    } else {
        bragg_radius(z) * 0.5
    };
    let np1 = n as f64 + 1.0;
    let mut r = vec![0.0; n];
    let mut w = vec![0.0; n];
    for k in 0..n {
        let i = (k + 1) as f64;
        let t = (i * PI / np1).cos();
        let wi = PI / np1 * (i * PI / np1).sin();
        r[k] = (1.0 + t) / (1.0 - t) * rm;
        w[k] = wi * (2.0 / ((1.0 - t) * (1.0 - t)) * rm);
    }
    (r, w)
}

/// PySCF `radi.mura_knowles` (ascending r).
fn mura_knowles(n: usize, z: u32, far_default: f64) -> (Vec<f64>, Vec<f64>) {
    let far = if matches!(z, 3 | 4 | 11 | 12 | 19 | 20) {
        7.0
    } else {
        far_default
    };
    let nf = n as f64;
    let mut r = vec![0.0; n];
    let mut dr = vec![0.0; n];
    for i in 0..n {
        let x = (i as f64 + 0.5) / nf;
        // Python's float ** 3 is C pow(x, 3.0)
        let x3 = x.powf(3.0);
        r[i] = -far * (1.0 - x3).ln();
        dr[i] = far * 3.0 * x * x / ((1.0 - x3) * nf);
    }
    (r, dr)
}

/// PySCF `radi.gauss_chebyshev` (ascending r).
fn krack_koster(n: usize) -> (Vec<f64>, Vec<f64>) {
    let ln2 = 1.0 / 2f64.ln();
    let nf = n as f64;
    let fac = 16.0 / 3.0 / (nf + 1.0);
    let x1: Vec<f64> = (1..=n).map(|i| i as f64 * PI / (nf + 1.0)).collect();
    let xi0: Vec<f64> = (0..n)
        .map(|i| {
            let s = x1[i].sin();
            (nf - 1.0 - (i as f64) * 2.0) / (nf + 1.0)
                + (1.0 + 2.0 / 3.0 * (s * s)) * (2.0 * x1[i]).sin() / PI
        })
        .collect();
    let xi: Vec<f64> = (0..n).map(|i| (xi0[i] - xi0[n - 1 - i]) / 2.0).collect();
    let mut r = vec![0.0; n];
    let mut dr = vec![0.0; n];
    for i in 0..n {
        r[i] = 1.0 - (1.0 + xi[i]).ln() * ln2;
        let s4 = x1[i].sin().powf(4.0);
        dr[i] = fac * s4 * ln2 / (1.0 + xi[i]);
    }
    (r, dr)
}

/// PySCF `radi.delley` (ascending r).
fn delley(n: usize, r_outer: f64) -> (Vec<f64>, Vec<f64>) {
    let step = 1.0 / (n as f64 + 1.0);
    let rfac = r_outer / (1.0 - (n as f64 * step).powi(2)).ln();
    let mut r = vec![0.0; n];
    let mut dr = vec![0.0; n];
    for i in 1..=n {
        let fi = i as f64;
        let q = fi * step;
        r[i - 1] = rfac * (1.0 - q * q).ln();
        dr[i - 1] = rfac * (-2.0 * fi * (step * step)) / (1.0 - q * q);
    }
    (r, dr)
}

/// Generic Gauss-Chebyshev of either kind with a mapping (ascending r).
fn gauss_chebyshev(
    n: usize,
    kind: ChebyshevKind,
    mapping: RadialMapping,
    scale: f64,
) -> (Vec<f64>, Vec<f64>) {
    let nf = n as f64;
    let mut r = Vec::with_capacity(n);
    let mut dr = Vec::with_capacity(n);
    for i in 1..=n {
        let (x, wx) = match kind {
            ChebyshevKind::First => {
                let t = (2.0 * i as f64 - 1.0) * PI / (2.0 * nf);
                (t.cos(), PI / nf * t.sin())
            }
            ChebyshevKind::Second => {
                let t = i as f64 * PI / (nf + 1.0);
                (t.cos(), PI / (nf + 1.0) * t.sin())
            }
            ChebyshevKind::PerezJorda => {
                let t = i as f64 * PI / (nf + 1.0);
                let (s, c) = t.sin_cos();
                let x = (nf + 1.0 - 2.0 * i as f64) / (nf + 1.0)
                    + 2.0 / PI * (1.0 + 2.0 / 3.0 * s * s) * c * s;
                (x, 16.0 / (3.0 * (nf + 1.0)) * (s * s) * (s * s))
            }
        };
        let (ri, dri) = map(mapping, x, scale);
        r.push(ri);
        dr.push(dri * wx);
    }
    // x = cos(...) decreases with i, so r decreases; return ascending.
    r.reverse();
    dr.reverse();
    (r, dr)
}

/// Mapping r(x) and dr/dx.
pub fn map(mapping: RadialMapping, x: f64, scale: f64) -> (f64, f64) {
    match mapping {
        RadialMapping::Becke => {
            let r = scale * (1.0 + x) / (1.0 - x);
            (r, 2.0 * scale / ((1.0 - x) * (1.0 - x)))
        }
        RadialMapping::TreutlerM3 => {
            let c = scale / 2f64.ln();
            let l = (2.0 / (1.0 - x)).ln();
            (c * l, c / (1.0 - x))
        }
        RadialMapping::TreutlerM4 { alpha } => {
            let c = scale / 2f64.ln();
            let l = (2.0 / (1.0 - x)).ln();
            let p = (1.0 + x).powf(alpha);
            (c * p * l, c * (alpha * p / (1.0 + x) * l + p / (1.0 - x)))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// int_0^inf 4 pi r^2 exp(-a r^2) dr = (pi/a)^{3/2}
    fn gauss_norm(g: &RadialGrid, a: f64) -> f64 {
        g.r.iter()
            .zip(&g.w)
            .map(|(&r, &w)| w * (-a * r * r).exp())
            .sum()
    }

    #[test]
    fn schemes_integrate_gaussians() {
        let schemes = [
            RadialScheme::PYSCF_TREUTLER,
            RadialScheme::Psi4Treutler { bs_alpha: 1.0 },
            RadialScheme::Becke,
            RadialScheme::PYSCF_MURA_KNOWLES,
            RadialScheme::KrackKoster,
            RadialScheme::PYSCF_DELLEY,
            RadialScheme::GaussChebyshev {
                kind: ChebyshevKind::First,
                mapping: RadialMapping::TreutlerM4 { alpha: 0.6 },
                scale: RadialScale::TreutlerXi,
            },
            RadialScheme::GaussChebyshev {
                kind: ChebyshevKind::Second,
                mapping: RadialMapping::TreutlerM3,
                scale: RadialScale::TreutlerXi,
            },
            RadialScheme::GaussChebyshev {
                kind: ChebyshevKind::Second,
                mapping: RadialMapping::Becke,
                scale: RadialScale::Bragg { factor: 0.5 },
            },
            RadialScheme::GaussChebyshev {
                kind: ChebyshevKind::PerezJorda,
                mapping: RadialMapping::TreutlerM3,
                scale: RadialScale::TreutlerXi,
            },
            RadialScheme::GaussChebyshev {
                kind: ChebyshevKind::PerezJorda,
                mapping: RadialMapping::TreutlerM3,
                scale: RadialScale::OrcaXi { angular_grid: 4 },
            },
        ];
        for s in schemes.iter() {
            let g = radial_grid(s, 150, 8);
            for &a in &[0.3, 1.0, 10.0, 100.0] {
                let exact = (PI / a).powf(1.5);
                let v = gauss_norm(&g, a);
                assert!(
                    ((v - exact) / exact).abs() < 1e-8,
                    "{s:?} a={a} got {v} exact {exact}"
                );
            }
        }
    }
}
