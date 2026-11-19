//! Smooth discretization of the solute cavity: Lebedev points on atom-centred spheres, each
//! carrying a Gaussian charge, with a switching function that turns points off smoothly as
//! they enter neighbouring spheres.
//!
//! Point i of sphere J (radius R_J, Lebedev direction n_i and weight w_i, sum w = 1):
//!   r_i = X_J + R_J n_i,   W_i = 4 pi w_i,   zeta_i = xi_N / (R_J sqrt(W_i)),
//!   F_i = prod_{K != J} f(|r_i - X_K|; K),   a_i = W_i R_J^2 F_i (area),
//! with xi_N the Gaussian-exponent constant of the N-point Lebedev grid (York & Karplus,
//! J. Phys. Chem. A 103, 11060 (1999); the table PySCF uses). Switching functions (Lange &
//! Herbert, J. Chem. Phys. 133, 244111 (2010)):
//! - SWIG: f = h((r - R_in,K) / R_sw,K), h(x) = x^3 (10 - 15 x + 6 x^2) on [0, 1], with
//!   R_sw,K = R_K sqrt(14 / N), alpha_K = 1/2 + R_K/R_sw - sqrt((R_K/R_sw)^2 - 1/28) and
//!   R_in,K = R_K - alpha_K R_sw,K (C2-continuous);
//! - ISWIG: f = 1 - 1/2 [erf(zeta_i (R_K - r)) + erf(zeta_i (R_K + r))] (smooth).
//!
//! Cut-offs ([`PointCut`]): as in PySCF, factors f < 1e-8 (ISWIG) or arguments x < 1e-8 (SWIG)
//! are set to zero and points with W_i F_i <= 1e-16 are dropped; as in ORCA 6, points with
//! F_i below a cut are dropped (no per-factor cut). Every sphere may have its own Lebedev grid
//! (ORCA picks it from a point density). A point moves rigidly with its sphere's atom (its
//! normal n_i does not change), so derivatives with respect to the atoms act on r_i through the
//! owner and on F_i through all sphere centres.

use super::Discretization;
use linus_ints::points::pcm_ints::GaussPoint;
use std::f64::consts::PI;

/// Gaussian-exponent constants xi_N per Lebedev point count (TABLE II of J. Chem. Phys. 122,
/// 194110 (2005), as in PySCF's `pyscf.solvent.pcm.XI`).
pub const XI: [(usize, f64); 29] = [
    (6, 4.84566077868),
    (14, 4.86458714334),
    (26, 4.85478226219),
    (38, 4.90105812685),
    (50, 4.89250673295),
    (86, 4.89741372580),
    (110, 4.90101060987),
    (146, 4.89825187392),
    (170, 4.90685517725),
    (194, 4.90337644248),
    (302, 4.90498088169),
    (350, 4.86879474832),
    (434, 4.90567349080),
    (590, 4.90624071359),
    (770, 4.90656435779),
    (974, 4.90685167998),
    (1202, 4.90704098216),
    (1454, 4.90721023869),
    (1730, 4.90733270691),
    (2030, 4.90744499142),
    (2354, 4.90753082825),
    (2702, 4.90760972766),
    (3074, 4.90767282394),
    (3470, 4.90773141371),
    (3890, 4.90777965981),
    (4334, 4.90782469526),
    (4802, 4.90749125553),
    (5294, 4.90762073452),
    (5810, 4.90792902522),
];

/// xi_N of an N-point Lebedev grid, if tabulated.
pub fn xi_for(n: usize) -> Option<f64> {
    XI.iter().find(|x| x.0 == n).map(|x| x.1)
}

/// The `orca` PCM preset's constants (ORCA 6.0.0's C-PCM, read off its output and `.cpcm`
/// files; included with permission of FAccTs GmbH): `data/orca/pcm.json` of the source tree,
/// compiled in.
#[derive(Clone, Debug, PartialEq, serde::Deserialize)]
pub struct OrcaPcm {
    /// ORCA's sphere grids (points per sphere) and their exponent constants xi_N.
    #[serde(rename = "lebedev_xi", deserialize_with = "by_points")]
    pub xi: Vec<(usize, f64)>,
    /// Points per Å^2 that pick a sphere's grid.
    pub points_per_angstrom2: f64,
    /// Points with a switching factor F below this are dropped.
    pub cut_swf: f64,
    /// Radius (Å, before scaling) of elements without a Bondi/Mantina radius.
    pub fallback_radius_angstrom: f64,
}

fn by_points<'de, D: serde::Deserializer<'de>>(d: D) -> Result<Vec<(usize, f64)>, D::Error> {
    #[derive(serde::Deserialize)]
    struct T {
        by_points: Vec<(usize, f64)>,
    }
    Ok(<T as serde::Deserialize>::deserialize(d)?.by_points)
}

impl OrcaPcm {
    /// The constants, parsed once per process (the file's literals, to the same bits with
    /// serde_json's `float_roundtrip`).
    pub fn get() -> &'static OrcaPcm {
        static DATA: std::sync::OnceLock<OrcaPcm> = std::sync::OnceLock::new();
        DATA.get_or_init(|| {
            serde_json::from_str(include_str!("../../../data/orca/pcm.json"))
                .unwrap_or_else(|e| panic!("data/orca/pcm.json is malformed: {e}"))
        })
    }

    /// ORCA's xi_N of an N-point grid.
    pub fn xi_for(&self, n: usize) -> Option<f64> {
        self.xi.iter().find(|x| x.0 == n).map(|x| x.1)
    }

    /// ORCA's sphere grid for a sphere of radius `r_angstrom`: the grid of its list nearest to
    /// [`OrcaPcm::points_per_angstrom2`] x 4 pi r^2.
    pub fn lebedev_for(&self, r_angstrom: f64) -> usize {
        let target = self.points_per_angstrom2 * 4.0 * PI * r_angstrom * r_angstrom;
        self.xi
            .iter()
            .map(|x| x.0)
            .min_by(|a, b| {
                (*a as f64 - target)
                    .abs()
                    .partial_cmp(&(*b as f64 - target).abs())
                    .unwrap()
            })
            .unwrap()
    }
}

/// Which points are kept and how small switching factors are cut.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PointCut {
    /// PySCF: factors f < 1e-8 (ISWIG) or SWIG arguments x < 1e-8 set to zero; keep W F > 1e-16.
    Pyscf,
    /// ORCA: keep F >= the cut (`cut_swf`), no per-factor cut.
    Orca(f64),
}

/// The cavity surface: points (with Gaussian exponents), their owners, normals, weights,
/// switching factors and areas, and the per-atom sphere data.
#[derive(Clone, Debug)]
pub struct Surface {
    pub scheme: Discretization,
    pub cut: PointCut,
    /// Lebedev points of each atom's sphere.
    pub nleb: Vec<usize>,
    pub pts: Vec<GaussPoint>,
    /// Atom (sphere) of each point.
    pub owner: Vec<usize>,
    pub normal: Vec<[f64; 3]>,
    /// 4 pi w_i (Lebedev weight on the unit sphere times 4 pi).
    pub weight: Vec<f64>,
    /// Switching factor F_i.
    pub switch: Vec<f64>,
    /// Area a_i = W_i R_J^2 F_i (bohr^2).
    pub area: Vec<f64>,
    /// Sphere radius R_J (bohr) per atom.
    pub atom_radius: Vec<f64>,
    /// SWIG switching width and inner radius per atom (bohr).
    pub r_sw: Vec<f64>,
    pub r_in: Vec<f64>,
    /// First point of each atom (points are grouped by atom), length natom + 1.
    pub atom_start: Vec<usize>,
}

/// SWIG switching polynomial and its derivative.
#[inline]
fn switch_h(x: f64) -> (f64, f64) {
    if x < 0.0 {
        (0.0, 0.0)
    } else if x > 1.0 {
        (1.0, 0.0)
    } else {
        (
            x * x * x * (10.0 - 15.0 * x + 6.0 * x * x),
            30.0 * x * x - 60.0 * x * x * x + 30.0 * x * x * x * x,
        )
    }
}

/// Second derivative of the SWIG polynomial.
#[inline]
pub(crate) fn switch_h2(x: f64) -> f64 {
    if !(0.0..=1.0).contains(&x) {
        0.0
    } else {
        60.0 * x - 180.0 * x * x + 120.0 * x * x * x
    }
}

impl Surface {
    /// Build the surface for atoms at `positions` (bohr) with sphere radii `radii` (bohr), the
    /// sphere of atom J carrying `nleb[J]` Lebedev points with exponent constant `xi[J]`.
    #[allow(clippy::needless_range_loop, reason = "indexes several arrays in step")]
    pub fn build(
        positions: &[[f64; 3]],
        radii: &[f64],
        nleb: &[usize],
        xi: &[f64],
        scheme: Discretization,
        cut: PointCut,
    ) -> Result<Surface, String> {
        let natom = positions.len();
        let r_sw: Vec<f64> = radii
            .iter()
            .zip(nleb)
            .map(|(&r, &n)| r * (14.0 / n as f64).sqrt())
            .collect();
        let r_in: Vec<f64> = radii
            .iter()
            .zip(&r_sw)
            .map(|(&r, &sw)| {
                let alpha = 0.5 + r / sw - ((r / sw).powi(2) - 1.0 / 28.0).sqrt();
                r - alpha * sw
            })
            .collect();
        let mut s = Surface {
            scheme,
            cut,
            nleb: nleb.to_vec(),
            pts: vec![],
            owner: vec![],
            normal: vec![],
            weight: vec![],
            switch: vec![],
            area: vec![],
            atom_radius: radii.to_vec(),
            r_sw,
            r_in,
            atom_start: vec![0],
        };
        for ia in 0..natom {
            let rj = radii[ia];
            let nl = nleb[ia];
            let leb = linus_grid::lebedev::lebedev(nl)
                .ok_or_else(|| format!("no Lebedev grid with {nl} points"))?;
            let xi = xi[ia];
            for k in 0..nl {
                let n = [leb.x[k], leb.y[k], leb.z[k]];
                let r = [
                    rj * n[0] + positions[ia][0],
                    rj * n[1] + positions[ia][1],
                    rj * n[2] + positions[ia][2],
                ];
                let w = leb.w[k] * 4.0 * PI;
                let zeta = xi / (rj * w.sqrt());
                let mut f = 1.0;
                for jb in 0..natom {
                    if jb == ia {
                        continue;
                    }
                    let d = dist(&r, &positions[jb]);
                    let fj = s.factor(jb, d, zeta);
                    f *= fj;
                    if f == 0.0 {
                        break;
                    }
                }
                let keep = match cut {
                    PointCut::Pyscf => w * f > 1e-16,
                    PointCut::Orca(c) => f >= c,
                };
                if keep {
                    s.pts.push(GaussPoint { r, zeta });
                    s.owner.push(ia);
                    s.normal.push(n);
                    s.weight.push(w);
                    s.switch.push(f);
                    s.area.push(w * rj * rj * f);
                }
            }
            s.atom_start.push(s.pts.len());
        }
        Ok(s)
    }

    pub fn len(&self) -> usize {
        self.pts.len()
    }

    pub fn is_empty(&self) -> bool {
        self.pts.is_empty()
    }

    /// Sphere radius of point i's owner.
    pub fn radius_of(&self, i: usize) -> f64 {
        self.atom_radius[self.owner[i]]
    }

    /// Switching factor f of sphere `j` at distance `r` from its centre, for a point with
    /// Gaussian exponent `zeta` (with PySCF's cut-offs).
    pub fn factor(&self, j: usize, r: f64, zeta: f64) -> f64 {
        match self.scheme {
            Discretization::Swig => {
                let mut x = (r - self.r_in[j]) / self.r_sw[j];
                if x < 1e-8 && self.cut == PointCut::Pyscf {
                    x = 0.0;
                }
                switch_h(x).0
            }
            Discretization::Iswig => {
                let rj = self.atom_radius[j];
                let f = 1.0 - 0.5 * (libm_erf(zeta * (rj - r)) + libm_erf(zeta * (rj + r)));
                if f < 1e-8 && self.cut == PointCut::Pyscf {
                    0.0
                } else {
                    f
                }
            }
        }
    }

    /// (f, df/dr, d2f/dr2) of sphere `j` at distance `r` (no cut-offs: used for points that
    /// survived them).
    pub fn factor_derivs(&self, j: usize, r: f64, zeta: f64) -> (f64, f64, f64) {
        match self.scheme {
            Discretization::Swig => {
                let sw = self.r_sw[j];
                let x = (r - self.r_in[j]) / sw;
                let (h, dh) = switch_h(x);
                (h, dh / sw, switch_h2(x) / (sw * sw))
            }
            Discretization::Iswig => {
                let rj = self.atom_radius[j];
                let (am, ap) = (zeta * (rj - r), zeta * (rj + r));
                let f = 1.0 - 0.5 * (libm_erf(am) + libm_erf(ap));
                let c = zeta / PI.sqrt();
                let (em, ep) = ((-am * am).exp(), (-ap * ap).exp());
                let df = c * (em - ep);
                // d/dr exp(-zeta^2 (R - r)^2) = 2 zeta am exp(..); d/dr exp(-zeta^2 (R + r)^2) = -2 zeta ap exp(..)
                let d2f = c * (2.0 * zeta * am * em + 2.0 * zeta * ap * ep);
                (f, df, d2f)
            }
        }
    }

    /// Total area (bohr^2).
    pub fn total_area(&self) -> f64 {
        self.area.iter().sum()
    }

    /// For each point i, the list of (atom K, unit vector (r_i - X_K)/|r_i - X_K|, f, df/dr,
    /// d2f/dr2, distance) over the other spheres K with f < 1 (those whose motion changes F_i).
    pub fn switch_terms(&self, positions: &[[f64; 3]], i: usize) -> Vec<SwitchTerm> {
        let own = self.owner[i];
        let r = self.pts[i].r;
        let mut out = Vec::new();
        for (k, c) in positions.iter().enumerate() {
            if k == own {
                continue;
            }
            let d = dist(&r, c);
            let (f, df, d2f) = self.factor_derivs(k, d, self.pts[i].zeta);
            if df == 0.0 && d2f == 0.0 {
                continue;
            }
            let u = [(r[0] - c[0]) / d, (r[1] - c[1]) / d, (r[2] - c[2]) / d];
            out.push(SwitchTerm {
                atom: k,
                u,
                f,
                df,
                d2f,
                r: d,
            });
        }
        out
    }

    /// sum_i c_i dF_i/dX for all atom coordinates X (natom x 3): dF_i = F_i sum_K (f'_K / f_K)
    /// u_iK . (dr_i - dX_K), with r_i moving with the owner.
    pub fn switch_grad(&self, positions: &[[f64; 3]], c: &[f64]) -> Vec<[f64; 3]> {
        use rayon::prelude::*;
        let per: Vec<Vec<(usize, [f64; 3])>> = (0..self.len())
            .into_par_iter()
            .map(|i| {
                if c[i] == 0.0 {
                    return vec![];
                }
                let mut v = Vec::new();
                for t in self.switch_terms(positions, i) {
                    let s = c[i] * self.switch[i] * t.df / t.f;
                    let g = [s * t.u[0], s * t.u[1], s * t.u[2]];
                    v.push((self.owner[i], g));
                    v.push((t.atom, [-g[0], -g[1], -g[2]]));
                }
                v
            })
            .collect();
        let mut out = vec![[0.0; 3]; positions.len()];
        for v in per {
            for (a, g) in v {
                for k in 0..3 {
                    out[a][k] += g[k];
                }
            }
        }
        out
    }

    /// Point positions and exponents as Gaussian charges.
    pub fn points(&self) -> &[GaussPoint] {
        &self.pts
    }
}

/// One switching factor of a point (see [`Surface::switch_terms`]).
#[derive(Clone, Copy, Debug)]
pub struct SwitchTerm {
    pub atom: usize,
    pub u: [f64; 3],
    pub f: f64,
    pub df: f64,
    pub d2f: f64,
    pub r: f64,
}

#[inline]
pub(crate) fn dist(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}

extern "C" {
    /// C99 libm error function.
    fn erf(x: f64) -> f64;
}

/// The error function (the C library's, accurate to an ulp).
#[inline]
pub fn libm_erf(x: f64) -> f64 {
    // SAFETY: pure libm function without side effects
    unsafe { erf(x) }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The compiled-in `orca` preset constants parse (crates/linus/tests/orca_presets.rs checks
    /// the grids they pick against ORCA's).
    #[test]
    fn orca_pcm_constants_parse() {
        let c = OrcaPcm::get();
        assert_eq!(c.xi.len(), 11);
        assert_eq!(c.xi_for(194), Some(4.903));
        assert_eq!(c.points_per_angstrom2, 5.0);
    }
}
