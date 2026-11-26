//! Atomic partitioning (fuzzy Voronoi cells): Becke, Stratmann-Scuseria-Frisch and
//! Laqua-Kussmann-Ochsenfeld, with optional Becke/Treutler atomic-size adjustment, exact
//! O(N) screening for SSF and LKO, and analytic derivatives of the partition weights with
//! respect to nuclear positions (grid points move rigidly with their owning atom).
//!
//! Notation: for a point p, d_B = |p - R_B|, mu_BC = (d_B - d_C) g(R_BC) with g(R) = 1/R
//! (Becke, SSF) or 1/f(R) (LKO, f a smooth saturation of the distance at r_cut),
//! nu_BC = mu_BC + a_BC (1 - mu_BC^2) (size adjustment), cell function
//! P_B(p) = prod_{C != B} s(nu_BC), and the partition weight of a point owned by A is
//! w_A(p) = P_A(p) / sum_B P_B(p).
//!
//! Floating-point operations follow PySCF (`VXCgen_grid`, `VXCgen_grid_lko`, and the Python
//! `stratmann` path) and products are accumulated in the same order, so PySCF's weights are
//! reproduced to the last bit or ulp. Screening only drops factors that are exactly 1 (or
//! cell functions that are exactly 0), so it does not change any result.

use crate::cells::CellList;
use crate::elements::RadiiTable;
use serde::{Deserialize, Serialize};

/// Partitioning scheme.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum PartitionScheme {
    /// No partitioning: weights are the atomic quadrature weights (single atoms, tests).
    None,
    /// Becke (JCP 88, 2547 (1988)), three iterations of p(x) = 3x/2 - x^3/2. Not screened
    /// (Becke cell functions never vanish exactly): O(N^2) work per point.
    Becke,
    /// Stratmann-Scuseria-Frisch (CPL 257, 213 (1996)), polynomial switching with half-width
    /// `a` (0.64). Exactly screened: O(1) work per point.
    Ssf { a: f64 },
    /// Laqua-Kussmann-Ochsenfeld (JCP 149, 204111 (2018)) as implemented by PySCF
    /// (`becke_lko`): Becke switching on mu = clip((d_B - d_C)/f(R_BC), -1, 1) with
    /// f(R) = r_cut (1 - exp(-sum_{m=1}^{m_cut} (R/r_cut)^m / m)). Exactly screened.
    Lko { r_cut: f64, m_cut: u32 },
}

impl PartitionScheme {
    pub const SSF: PartitionScheme = PartitionScheme::Ssf { a: 0.64 };
    pub const LKO: PartitionScheme = PartitionScheme::Lko {
        r_cut: 5.0,
        m_cut: 12,
    };
}

/// Atomic size adjustment of the cell boundaries (Becke 1988, appendix).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum SizeAdjust {
    None,
    /// chi = R_A / R_B (Becke): PySCF `radi.becke_atomic_radii_adjust`.
    Becke(RadiiTable),
    /// chi = sqrt(R_A / R_B) (Treutler-Ahlrichs): PySCF `radi.treutler_atomic_radii_adjust`
    /// (PySCF default with Bragg radii), Psi4 `DFT_NUCLEAR_SCHEME TREUTLER`.
    Treutler(RadiiTable),
    /// Treutler-Ahlrichs for the pairs of a hydrogen atom with an atom beyond the second row
    /// (Z > 10), none for the other pairs (linus:3). A molecule without such a pair (no H, or
    /// no atom beyond neon) is partitioned exactly as with `None`.
    TreutlerBeyondRow2(RadiiTable),
}

/// Heaviest element of the second row (neon): `SizeAdjust::TreutlerBeyondRow2` adjusts only the
/// pairs with a heavier atom.
pub const ROW2_MAX_Z: u32 = 10;

#[derive(Clone, Copy, Debug, PartialEq)]
enum Kind {
    None,
    Becke,
    Ssf { a: f64 },
    Lko { r_cut: f64, m_cut: u32 },
}

/// Evaluates partition weights and their nuclear derivatives for one molecule.
#[derive(Clone, Debug)]
pub struct Partitioner {
    kind: Kind,
    xyz: Vec<[f64; 3]>,
    /// size-adjustment radii (sqrt'ed for Treutler) and their reciprocals
    rad: Vec<f64>,
    inv_rad: Vec<f64>,
    adjust: bool,
    /// `SizeAdjust::TreutlerBeyondRow2` with an H atom and an atom beyond neon: per atom, 0 for
    /// H, 1 up to neon, 2 beyond (only pairs {0, 2} are adjusted); empty otherwise
    light: Vec<u8>,
    screened: bool,
    cells: Option<CellList>,
    cell_h: f64,
    /// SSF: multiplicative screening factor k; LKO: additive r_cut (both with safety margins)
    k_mul: f64,
    k_add: f64,
    /// per-atom multiplicative screening factors (SSF): factor(B, C) == 1 if
    /// d_C >= k_one[B] d_B; P_B == 0 if d_B >= k_zero[B] d_nearest
    k_one: Vec<f64>,
    k_zero: Vec<f64>,
    k_zero_max: f64,
    /// SSF: points closer than this to their atom have partition weight exactly 1
    fast_r: Vec<f64>,
    /// dense 1/R_ij (or 1/f(R_ij) for LKO) for molecules up to `DENSE_MAX` atoms
    inv_r_mat: Vec<f64>,
    /// NBR nearest neighbours of every atom (screened schemes): likely "killers" whose factor
    /// zeroes a cell function, tested before the full product
    nbr: Vec<[u32; NBR]>,
}

/// The cell factors of an unscreened partition ([`Partitioner::pair_tables`]).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PairScheme {
    /// Becke's s(nu) of the (size-adjusted) nu.
    Becke,
    /// Becke's s(nu) of nu clamped to [-1, 1] before the size adjustment (LKO).
    Lko,
    /// SSF's z(nu / half), -1 and 1 outside (-half, half).
    Ssf { half: f64 },
}

/// The atoms and per-pair factors of an unscreened partition, row-major n x n: `inv_r[i n + j]`
/// = 1/R_ij (1/f(R_ij) for LKO), `a[i n + j]` the size adjustment a_ij (empty: none).
#[derive(Clone, Debug)]
pub struct PairTables {
    pub scheme: PairScheme,
    pub xyz: Vec<[f64; 3]>,
    pub inv_r: Vec<f64>,
    pub a: Vec<f64>,
}

/// Number of nearest neighbours tabulated per atom.
const NBR: usize = 6;

/// Largest molecule for which the inverse interatomic distances are tabulated (32 MB).
const DENSE_MAX: usize = 2048;
/// Screened neighbour lists: the NEAR closest atoms come first, sorted by distance; the rest
/// (all farther than those) follow unsorted.
const NEAR: usize = 24;

/// Per-thread scratch space.
#[derive(Clone, Debug, Default)]
pub struct Scratch {
    blk_d: Vec<f64>,
    blk_p: Vec<f64>,
    list: Vec<(usize, f64)>,
    zl: Vec<(usize, f64)>,
    list2: Vec<(usize, f64)>,
    f: Vec<f64>,
    df: Vec<f64>,
    pre: Vec<f64>,
    dpa: Vec<[f64; 3]>,
    dz: Vec<[f64; 3]>,
    touched: Vec<usize>,
    mark: Vec<bool>,
}

#[inline]
fn dist(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];
    (dx * dx + dy * dy + dz * dz).sqrt()
}

/// Becke's step function s(nu) = f(f(f(nu))) / 2 with f(x) = (3 - x^2) x / 2: the cell factors
/// of a pair are 0.5 - s and 0.5 + s.
#[inline(always)]
fn becke_s(nu: f64) -> f64 {
    let mut t = nu;
    t = (3.0 - t * t) * t * 0.5;
    t = (3.0 - t * t) * t * 0.5;
    ((3.0 - t * t) * t * 0.5) * 0.5
}

/// The SSF polynomial z(m) = (35 m - 35 m^3 + 21 m^5 - 5 m^7) / 16 of m = nu / a, |m| < 1
/// (z = -1 below and 1 above): the cell factors of a pair are (1 - z) / 2 and (1 + z) / 2.
#[inline(always)]
fn ssf_z(m: f64) -> f64 {
    let m2 = m * m;
    (1.0 / 16.0) * (m * (35.0 + m2 * (-35.0 + m2 * (21.0 - 5.0 * m2))))
}

#[inline]
fn lko_sat(r: f64, r_cut: f64, m_cut: u32) -> f64 {
    let x = r / r_cut;
    let mut tot = 0.0;
    let mut xm = 1.0;
    for m in 1..=m_cut {
        xm *= x;
        tot += xm / m as f64;
    }
    r_cut * (1.0 - (-tot).exp())
}

#[inline]
fn lko_sat_deriv(r: f64, r_cut: f64, m_cut: u32) -> f64 {
    let x = r / r_cut;
    let mut tot = 0.0;
    let mut dtot = 0.0;
    let mut xm = 1.0;
    for m in 1..=m_cut {
        dtot += xm;
        xm *= x;
        tot += xm / m as f64;
    }
    (-tot).exp() * dtot
}

impl Partitioner {
    /// `z`: nuclear charges (used for size adjustment); `xyz`: positions in bohr.
    pub fn new(z: &[u32], xyz: &[[f64; 3]], scheme: PartitionScheme, adjust: SizeAdjust) -> Self {
        let n = xyz.len();
        let treutler = |t: RadiiTable| -> Vec<f64> {
            z.iter().map(|&zi| t.radius(zi).sqrt() + 1e-200).collect()
        };
        let mut light = Vec::new();
        let (rad, adjust_on): (Vec<f64>, bool) = match adjust {
            SizeAdjust::None => (vec![1.0; n], false),
            SizeAdjust::Becke(t) => (z.iter().map(|&zi| t.radius(zi) + 1e-200).collect(), true),
            SizeAdjust::Treutler(t) => (treutler(t), true),
            SizeAdjust::TreutlerBeyondRow2(t) => {
                if z.iter().any(|&zi| zi > ROW2_MAX_Z) && z.contains(&1) {
                    light = z
                        .iter()
                        .map(|&zi| match zi {
                            1 => 0,
                            _ if zi <= ROW2_MAX_Z => 1,
                            _ => 2,
                        })
                        .collect();
                    (treutler(t), true)
                } else {
                    (vec![1.0; n], false)
                }
            }
        };
        let inv_rad: Vec<f64> = rad.iter().map(|r| 1.0 / r).collect();
        let adj_on = adjust_on;
        let kind = match scheme {
            PartitionScheme::None => Kind::None,
            PartitionScheme::Becke => Kind::Becke,
            PartitionScheme::Ssf { a } => Kind::Ssf { a },
            PartitionScheme::Lko { r_cut, m_cut } => Kind::Lko { r_cut, m_cut },
        };
        let mut p = Partitioner {
            kind,
            xyz: xyz.to_vec(),
            rad,
            inv_rad,
            adjust: adjust_on,
            light,
            screened: false,
            cells: None,
            cell_h: 4.0,
            k_mul: f64::INFINITY,
            k_add: f64::INFINITY,
            k_one: Vec::new(),
            k_zero: Vec::new(),
            k_zero_max: f64::INFINITY,
            fast_r: vec![0.0; n],
            inv_r_mat: Vec::new(),
            nbr: Vec::new(),
        };
        // largest |a_ij| over the molecule
        let amax = if adjust_on && n > 1 {
            let rmin = p.rad.iter().cloned().fold(f64::INFINITY, f64::min);
            let rmax = p.rad.iter().cloned().fold(0.0, f64::max);
            let q = rmin / rmax;
            (0.25 * (1.0 / q - q)).min(0.5) * (1.0 + 1e-12)
        } else {
            0.0
        };
        match kind {
            Kind::Ssf { a } => {
                // smallest mu with nu(mu) >= a for the worst-case adjustment a_ij = -amax
                let mu_star = if amax > 0.0 {
                    ((1.0 + 4.0 * amax * (amax + a)).sqrt() - 1.0) / (2.0 * amax)
                } else {
                    a
                };
                if mu_star < 1.0 - 1e-6 {
                    p.screened = true;
                    p.k_mul = (1.0 + mu_star) / (1.0 - mu_star) * (1.0 + 1e-9);
                    p.k_add = 0.0;
                    // per-atom factors from the actual pair adjustments (distinct radii only)
                    let kk = |aij: f64| -> f64 {
                        // smallest mu with mu + aij (1 - mu^2) >= a, as a distance ratio
                        let m = if aij.abs() < 1e-14 {
                            a
                        } else {
                            (1.0 - (1.0 - 4.0 * aij * (a - aij)).max(0.0).sqrt()) / (2.0 * aij)
                        };
                        let m = (m * (1.0 + 1e-9) + 1e-12).min(1.0 - 1e-9);
                        (1.0 + m) / (1.0 - m) * (1.0 + 1e-9)
                    };
                    // atom types: (radius, up to neon under TreutlerBeyondRow2)
                    let is_light = |i: usize| p.light.get(i).copied().unwrap_or(3);
                    let mut types: Vec<(f64, u8)> =
                        (0..n).map(|i| (p.rad[i], is_light(i))).collect();
                    types.sort_by(|x, y| x.0.total_cmp(&y.0).then(x.1.cmp(&y.1)));
                    types.dedup();
                    let adj = |(ri, li): (f64, u8), (rj, lj): (f64, u8)| -> f64 {
                        if adj_on && (li == 3 || li + lj == 2 && li != 1) {
                            (0.25 * (rj * (1.0 / ri) - ri * (1.0 / rj))).clamp(-0.5, 0.5)
                        } else {
                            0.0
                        }
                    };
                    p.k_one = vec![0.0; n];
                    p.k_zero = vec![0.0; n];
                    for b in 0..n {
                        let tb = (p.rad[b], is_light(b));
                        for &tc in &types {
                            // factor(B,C) == 1 needs nu_BC <= -a, i.e. nu_CB >= a with a_CB
                            p.k_one[b] = p.k_one[b].max(kk(adj(tc, tb)));
                            // factor(B,N) == 0 needs nu_BN >= a with a_BN
                            p.k_zero[b] = p.k_zero[b].max(kk(adj(tb, tc)));
                        }
                    }
                    p.k_zero_max = p.k_zero.iter().cloned().fold(0.0, f64::max);
                    // filled from the nearest-neighbour distance once the neighbour table exists
                    p.fast_r = vec![0.5 * (1.0 - mu_star) * (1.0 - 1e-9); n];
                }
            }
            Kind::Lko { r_cut, .. } => {
                p.screened = true;
                p.k_mul = 1.0;
                p.k_add = r_cut * (1.0 + 1e-9) + 1e-9;
            }
            _ => {}
        }
        if p.screened {
            p.cells = Some(CellList::new(xyz, p.cell_h));
            p.build_neighbours();
            if let Kind::Ssf { .. } = kind {
                for i in 0..n {
                    let j = p.nbr[i][0];
                    let rmin = if j == u32::MAX {
                        f64::INFINITY
                    } else {
                        dist(&xyz[i], &xyz[j as usize])
                    };
                    p.fast_r[i] *= rmin;
                }
            }
        }
        if n <= DENSE_MAX && kind != Kind::None {
            let mut m = vec![0.0; n * n];
            for i in 0..n {
                for j in 0..n {
                    if i != j {
                        m[i * n + j] = p.inv_r_direct(i, j);
                    }
                }
            }
            p.inv_r_mat = m;
        }
        p
    }

    /// Enables/disables distance screening (SSF and LKO only; Becke is never screened).
    /// Screening drops only exact zeros/ones, so results are identical either way.
    pub fn with_screening(mut self, on: bool) -> Self {
        let can =
            matches!(self.kind, Kind::Ssf { .. } | Kind::Lko { .. }) && self.k_mul.is_finite();
        self.screened = on && can;
        if self.screened && self.cells.is_none() {
            self.cells = Some(CellList::new(&self.xyz, self.cell_h));
        }
        self
    }

    #[allow(clippy::needless_range_loop, reason = "indexes several arrays in step")]
    fn build_neighbours(&mut self) {
        let n = self.xyz.len();
        let cells = self.cells.as_ref().unwrap();
        let mut nbr = vec![[u32::MAX; NBR]; n];
        let mut cand: Vec<(f64, u32)> = Vec::new();
        for i in 0..n {
            let mut rho = self.cell_h;
            loop {
                cand.clear();
                cells.for_each_candidate(&self.xyz[i], rho, |j| {
                    if j != i {
                        let d = dist(&self.xyz[i], &self.xyz[j]);
                        if d <= rho {
                            cand.push((d, j as u32));
                        }
                    }
                });
                if cand.len() >= NBR || cand.len() == n - 1 {
                    break;
                }
                rho *= 2.0;
            }
            cand.sort_unstable_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
            for (k, c) in cand.iter().take(NBR).enumerate() {
                nbr[i][k] = c.1;
            }
        }
        self.nbr = nbr;
    }

    /// True if some tabulated neighbour C of `b` gives an exactly zero factor s(nu_bC) at p.
    #[inline]
    fn killed_by_neighbour(&self, p: &[f64; 3], b: usize, d_b: f64) -> bool {
        for &c in &self.nbr[b] {
            if c == u32::MAX {
                break;
            }
            let c = c as usize;
            let d_c = dist(p, &self.xyz[c]);
            if d_c < d_b && self.factor(b, c, d_b, d_c) == 0.0 {
                return true;
            }
        }
        false
    }

    pub fn natm(&self) -> usize {
        self.xyz.len()
    }

    pub fn scratch(&self) -> Scratch {
        let n = self.natm();
        Scratch {
            dpa: vec![[0.0; 3]; n],
            dz: vec![[0.0; 3]; n],
            mark: vec![false; n],
            ..Default::default()
        }
    }

    /// Whether this partitioner uses exact distance screening.
    pub fn is_screened(&self) -> bool {
        self.screened
    }

    /// What [`Self::weights_block`] computes with when it is not screened and has two atoms or
    /// more (None otherwise): the scheme and the per-pair factors exactly as it reads them, for an
    /// evaluation elsewhere (a device) that repeats it operation for operation.
    pub fn pair_tables(&self) -> Option<PairTables> {
        let n = self.natm();
        if self.screened || n < 2 || self.kind == Kind::None {
            return None;
        }
        let scheme = match self.kind {
            Kind::Becke => PairScheme::Becke,
            Kind::Lko { .. } => PairScheme::Lko,
            Kind::Ssf { a } => PairScheme::Ssf { half: a },
            Kind::None => return None,
        };
        let mut inv_r = vec![0.0; n * n];
        let mut a = vec![0.0; if self.adjust { n * n } else { 0 }];
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    inv_r[i * n + j] = self.inv_r(i, j);
                    if self.adjust {
                        a[i * n + j] = self.a_ij(i, j);
                    }
                }
            }
        }
        Some(PairTables {
            scheme,
            xyz: self.xyz.clone(),
            inv_r,
            a,
        })
    }

    #[inline]
    fn cut(&self, d: f64) -> f64 {
        d * self.k_mul + self.k_add
    }

    /// Distance beyond which factor(b, C) is exactly 1.
    #[inline]
    fn cut_one(&self, b: usize, d_b: f64) -> f64 {
        if self.k_one.is_empty() {
            self.cut(d_b)
        } else {
            d_b * self.k_one[b]
        }
    }

    /// P_b is exactly 0 if d_b >= cut_zero(b, d_nearest).
    #[inline]
    fn cut_zero(&self, b: usize, d_n: f64) -> f64 {
        if self.k_zero.is_empty() {
            self.cut(d_n)
        } else {
            d_n * self.k_zero[b]
        }
    }

    /// Radius of the neighbour list needed for a point owned by `a`.
    #[inline]
    fn list_radius(&self, a: usize, d_a: f64, d_n: f64) -> f64 {
        if self.k_one.is_empty() {
            self.cut(d_a)
        } else {
            self.cut_one(a, d_a).max(d_n * self.k_zero_max)
        }
    }

    #[inline]
    fn a_ij(&self, i: usize, j: usize) -> f64 {
        // TreutlerBeyondRow2: only the pairs of H with an atom beyond neon
        if !self.light.is_empty() && !(self.light[i] + self.light[j] == 2 && self.light[i] != 1) {
            return 0.0;
        }
        // PySCF: rr = rad.reshape(-1,1) * (1./rad); a = .25 * (rr.T - rr); clip to [-.5, .5]
        let a = 0.25 * (self.rad[j] * self.inv_rad[i] - self.rad[i] * self.inv_rad[j]);
        a.clamp(-0.5, 0.5)
    }

    /// 1/R_bc (Becke, SSF) or 1/f(R_bc) (LKO).
    #[inline(always)]
    fn inv_r(&self, b: usize, c: usize) -> f64 {
        if !self.inv_r_mat.is_empty() {
            return self.inv_r_mat[b * self.xyz.len() + c];
        }
        self.inv_r_direct(b, c)
    }

    #[inline]
    fn inv_r_direct(&self, b: usize, c: usize) -> f64 {
        let r = dist(&self.xyz[b], &self.xyz[c]);
        match self.kind {
            Kind::Lko { r_cut, m_cut } => 1.0 / lko_sat(r, r_cut, m_cut),
            _ => 1.0 / r,
        }
    }

    /// Cell-function factor s(nu_bc) for the point with distances d_b, d_c.
    #[inline(always)]
    fn factor(&self, b: usize, c: usize, d_b: f64, d_c: f64) -> f64 {
        let fac = self.inv_r(b, c);
        match self.kind {
            Kind::Becke | Kind::Lko { .. } => {
                let mut g = (d_b - d_c) * fac;
                if let Kind::Lko { .. } = self.kind {
                    g = g.clamp(-1.0, 1.0);
                }
                if self.adjust {
                    g += self.a_ij(b, c) * (1.0 - g * g);
                }
                0.5 - becke_s(g)
            }
            Kind::Ssf { a } => {
                let mut g = fac * (d_b - d_c);
                if self.adjust {
                    g += self.a_ij(b, c) * (1.0 - g * g);
                }
                let z = if g <= -a {
                    -1.0
                } else if g >= a {
                    1.0
                } else {
                    ssf_z(g / a)
                };
                0.5 * (1.0 - z)
            }
            Kind::None => 1.0,
        }
    }

    /// Both factors of a pair (i, j): (s(nu_ij), s(nu_ji)) computed as in PySCF's pair loop.
    #[inline(always)]
    fn pair_factors(&self, i: usize, j: usize, d_i: f64, d_j: f64) -> (f64, f64) {
        let fac = self.inv_r(i, j);
        match self.kind {
            Kind::Becke | Kind::Lko { .. } => {
                let mut g = (d_i - d_j) * fac;
                if let Kind::Lko { .. } = self.kind {
                    g = g.clamp(-1.0, 1.0);
                }
                if self.adjust {
                    g += self.a_ij(i, j) * (1.0 - g * g);
                }
                let s = becke_s(g);
                (0.5 - s, 0.5 + s)
            }
            Kind::Ssf { a } => {
                let mut g = fac * (d_i - d_j);
                if self.adjust {
                    g += self.a_ij(i, j) * (1.0 - g * g);
                }
                let z = if g <= -a {
                    -1.0
                } else if g >= a {
                    1.0
                } else {
                    ssf_z(g / a)
                };
                (0.5 * (1.0 - z), 0.5 * (1.0 + z))
            }
            Kind::None => (1.0, 1.0),
        }
    }

    /// Factor, d factor / d mu, and the pieces needed for d mu / dR.
    /// Returns (s, ds/dmu, fac = g(R), dfac/dR, R).
    #[inline]
    fn factor_d(&self, b: usize, c: usize, d_b: f64, d_c: f64) -> (f64, f64, f64, f64, f64) {
        let r = dist(&self.xyz[b], &self.xyz[c]);
        let (fac, dfac) = match self.kind {
            Kind::Lko { r_cut, m_cut } => {
                let f = lko_sat(r, r_cut, m_cut);
                (1.0 / f, -lko_sat_deriv(r, r_cut, m_cut) / (f * f))
            }
            _ => (1.0 / r, -1.0 / (r * r)),
        };
        let a = if self.adjust { self.a_ij(b, c) } else { 0.0 };
        match self.kind {
            Kind::Becke | Kind::Lko { .. } => {
                let mut mu = (d_b - d_c) * fac;
                let mut dmu = 1.0;
                if let Kind::Lko { .. } = self.kind {
                    if mu <= -1.0 || mu >= 1.0 {
                        dmu = 0.0;
                    }
                    mu = mu.clamp(-1.0, 1.0);
                }
                let nu = if self.adjust {
                    mu + a * (1.0 - mu * mu)
                } else {
                    mu
                };
                let dnu = dmu * (1.0 - 2.0 * a * mu);
                let s0 = nu;
                let s1 = (3.0 - s0 * s0) * s0 * 0.5;
                let s2 = (3.0 - s1 * s1) * s1 * 0.5;
                let s3 = ((3.0 - s2 * s2) * s2 * 0.5) * 0.5;
                let ds = -0.5
                    * (1.5 * (1.0 - s2 * s2))
                    * (1.5 * (1.0 - s1 * s1))
                    * (1.5 * (1.0 - s0 * s0));
                (0.5 - s3, ds * dnu, fac, dfac, r)
            }
            Kind::Ssf { a: half } => {
                let mu = fac * (d_b - d_c);
                let nu = if self.adjust {
                    mu + a * (1.0 - mu * mu)
                } else {
                    mu
                };
                let dnu = 1.0 - 2.0 * a * mu;
                let (z, dz) = if nu <= -half {
                    (-1.0, 0.0)
                } else if nu >= half {
                    (1.0, 0.0)
                } else {
                    let ma = nu / half;
                    let t = 1.0 - ma * ma;
                    (ssf_z(ma), (35.0 / 16.0) * t * t * t / half)
                };
                (0.5 * (1.0 - z), -0.5 * dz * dnu, fac, dfac, r)
            }
            Kind::None => (1.0, 0.0, fac, dfac, r),
        }
    }

    /// All atoms within `rho` of `p` (all atoms, in index order, if unscreened; screened lists
    /// come in cell order).
    fn gather(&self, p: &[f64; 3], rho: f64, out: &mut Vec<(usize, f64)>) {
        out.clear();
        match &self.cells {
            Some(cells) if rho.is_finite() => {
                cells.for_each_candidate(p, rho, |a| {
                    let d = dist(p, &self.xyz[a]);
                    if d <= rho {
                        out.push((a, d));
                    }
                });
                // the NEAR closest atoms are moved to the front and sorted by distance: zero
                // factors (which come from atoms closer than the cell owner) are found first
                if out.len() > NEAR {
                    out.select_nth_unstable_by(NEAR, |x, y| x.1.total_cmp(&y.1));
                    out[..NEAR].sort_unstable_by(|x, y| x.1.total_cmp(&y.1));
                } else {
                    out.sort_unstable_by(|x, y| x.1.total_cmp(&y.1));
                }
            }
            _ => {
                for (a, x) in self.xyz.iter().enumerate() {
                    out.push((a, dist(p, x)));
                }
            }
        }
    }

    /// Nearest atom to p, searching no farther than `d_max` (an atom at distance d_max exists).
    fn nearest(&self, p: &[f64; 3], d_max: f64, fallback: usize) -> (usize, f64) {
        let cells = match &self.cells {
            Some(c) => c,
            None => {
                let mut best = (fallback, d_max);
                for (a, x) in self.xyz.iter().enumerate() {
                    let d = dist(p, x);
                    if d < best.1 {
                        best = (a, d);
                    }
                }
                return best;
            }
        };
        let mut rho = (0.5 * self.cell_h).min(d_max);
        loop {
            let mut best = (usize::MAX, f64::INFINITY);
            cells.for_each_candidate(p, rho, |a| {
                let d = dist(p, &self.xyz[a]);
                if d <= rho && (d < best.1 || (d == best.1 && a < best.0)) {
                    best = (a, d);
                }
            });
            if best.0 != usize::MAX {
                return best;
            }
            if rho >= d_max {
                return (fallback, d_max);
            }
            rho = (rho * 2.0).min(d_max);
        }
    }

    /// Unscreened cell function: product over all atoms in index order (PySCF order).
    #[inline]
    fn cell_product(&self, b: usize, d_b: f64, list: &[(usize, f64)]) -> f64 {
        let mut prod = 1.0;
        for &(c, d_c) in list {
            if c == b {
                continue;
            }
            prod *= self.factor(b, c, d_b, d_c);
            if prod == 0.0 {
                return 0.0;
            }
        }
        prod
    }

    /// Screened cell function: factors from atoms at distance >= cut(d_b) are exactly 1 and
    /// skipped; the factor with the nearest atom `nn` (most likely zero) is applied first.
    /// Returns (P_b, complete) where `complete` is false if the list (all atoms within
    /// `list_rho`) may miss atoms with d < cut(d_b).
    #[inline]
    fn cell_product_screened(
        &self,
        b: usize,
        d_b: f64,
        nn: (usize, f64),
        list: &[(usize, f64)],
        list_rho: f64,
    ) -> (f64, bool) {
        let cut = self.cut_one(b, d_b);
        let mut prod = 1.0;
        if nn.0 != b {
            prod = self.factor(b, nn.0, d_b, nn.1);
            if prod == 0.0 {
                return (0.0, true);
            }
        }
        let ns = list.len().min(NEAR);
        for &(c, d_c) in &list[..ns] {
            if d_c >= cut {
                return (prod, true);
            }
            if c == b || c == nn.0 {
                continue;
            }
            prod *= self.factor(b, c, d_b, d_c);
            if prod == 0.0 {
                return (0.0, true);
            }
        }
        for &(c, d_c) in &list[ns..] {
            if d_c >= cut || c == b || c == nn.0 {
                continue;
            }
            prod *= self.factor(b, c, d_b, d_c);
            if prod == 0.0 {
                return (0.0, true);
            }
        }
        (prod, cut <= list_rho)
    }

    /// Partitioned weight of a point `p` owned by atom `owner` with atomic (unpartitioned)
    /// weight `vol`: `vol * P_owner(p) / sum_B P_B(p)` (PySCF operation order).
    pub fn weight(&self, p: [f64; 3], owner: usize, vol: f64, s: &mut Scratch) -> f64 {
        if self.kind == Kind::None || self.natm() == 1 {
            return vol;
        }
        let d_a = dist(&p, &self.xyz[owner]);
        if !self.screened {
            // PySCF's pair loop: every pair evaluated once, products in the same order
            let n = self.natm();
            self.gather(&p, f64::INFINITY, &mut s.list);
            s.f.clear();
            s.f.resize(n, 1.0);
            for i in 0..n {
                let d_i = s.list[i].1;
                for j in 0..i {
                    let (fi, fj) = self.pair_factors(i, j, d_i, s.list[j].1);
                    s.f[i] *= fi;
                    s.f[j] *= fj;
                }
            }
            let mut z = 0.0;
            for &v in s.f.iter() {
                z += v;
            }
            return vol * s.f[owner] * (1.0 / z);
        }
        // SSF: exactly 1 close to the nucleus
        if d_a <= self.fast_r[owner] {
            // P_A = 1 and Z = 1 exactly: PySCF's vol * 1 * (1/1) is vol
            return vol;
        }
        let (nn, d_n) = self.nearest(&p, d_a, owner);
        if nn != owner && self.factor(owner, nn, d_a, d_n) == 0.0 {
            return vol * 0.0;
        }
        let rho = self.list_radius(owner, d_a, d_n);
        let nnp = (nn, d_n);
        if self.killed_by_neighbour(&p, owner, d_a) {
            return vol * 0.0;
        }
        self.gather(&p, rho, &mut s.list);
        let (pa, _) = self.cell_product_screened(owner, d_a, nnp, &s.list, rho);
        if pa == 0.0 {
            return vol * 0.0;
        }
        // cell functions of the competing atoms, summed in atom-index order like PySCF
        s.zl.clear();
        for i in 0..s.list.len() {
            let (b, d_b) = s.list[i];
            if d_b >= self.cut_zero(b, d_n) {
                continue;
            }
            let pb = if b == owner {
                pa
            } else if b != nn
                && (self.factor(b, nn, d_b, d_n) == 0.0 || self.killed_by_neighbour(&p, b, d_b))
            {
                0.0
            } else {
                let (pb, complete) = self.cell_product_screened(b, d_b, nnp, &s.list, rho);
                if pb != 0.0 && !complete {
                    let mut l2 = std::mem::take(&mut s.list2);
                    let rho2 = self.cut_one(b, d_b);
                    self.gather(&p, rho2, &mut l2);
                    let (v, _) = self.cell_product_screened(b, d_b, nnp, &l2, rho2);
                    s.list2 = l2;
                    v
                } else {
                    pb
                }
            };
            if pb != 0.0 {
                s.zl.push((b, pb));
            }
        }
        s.zl.sort_unstable_by_key(|x| x.0);
        let mut z = 0.0;
        for &(_, pb) in &s.zl {
            z += pb;
        }
        vol * pa * (1.0 / z)
    }

    /// Partitioned weights of a block of points owned by the same atom (`out[k]` for point k).
    /// Unscreened schemes use PySCF's vectorisable pair loop over the block (same operations
    /// and product order as PySCF's VXCgen_grid); screened schemes call [`Self::weight`].
    #[allow(clippy::too_many_arguments, reason = "a kernel's arrays and sizes")]
    pub fn weights_block(
        &self,
        px: &[f64],
        py: &[f64],
        pz: &[f64],
        owner: usize,
        vol: &[f64],
        out: &mut [f64],
        s: &mut Scratch,
    ) {
        let m = px.len();
        if self.kind == Kind::None || self.natm() == 1 {
            out.copy_from_slice(vol);
            return;
        }
        if self.screened {
            for k in 0..m {
                out[k] = self.weight([px[k], py[k], pz[k]], owner, vol[k], s);
            }
            return;
        }
        let n = self.natm();
        s.blk_d.clear();
        s.blk_d.resize(n * m, 0.0);
        s.blk_p.clear();
        s.blk_p.resize(n * m, 1.0);
        for i in 0..n {
            let c = self.xyz[i];
            let d = &mut s.blk_d[i * m..(i + 1) * m];
            for k in 0..m {
                let dx = px[k] - c[0];
                let dy = py[k] - c[1];
                let dz = pz[k] - c[2];
                d[k] = (dx * dx + dy * dy + dz * dz).sqrt();
            }
        }
        for i in 1..n {
            let (lo, hi) = s.blk_p.split_at_mut(i * m);
            let bi = &mut hi[..m];
            let di = &s.blk_d[i * m..(i + 1) * m];
            for j in 0..i {
                let bj = &mut lo[j * m..(j + 1) * m];
                let dj = &s.blk_d[j * m..(j + 1) * m];
                let fac = self.inv_r(i, j);
                let a = if self.adjust { self.a_ij(i, j) } else { 0.0 };
                match self.kind {
                    Kind::Becke | Kind::Lko { .. } => {
                        let lko = matches!(self.kind, Kind::Lko { .. });
                        for k in 0..m {
                            let mut g = (di[k] - dj[k]) * fac;
                            if lko {
                                g = g.clamp(-1.0, 1.0);
                            }
                            if self.adjust {
                                g += a * (1.0 - g * g);
                            }
                            let t = becke_s(g);
                            bi[k] *= 0.5 - t;
                            bj[k] *= 0.5 + t;
                        }
                    }
                    Kind::Ssf { a: half } => {
                        for k in 0..m {
                            let mut g = fac * (di[k] - dj[k]);
                            if self.adjust {
                                g += a * (1.0 - g * g);
                            }
                            let mut z = ssf_z(g / half);
                            if g <= -half {
                                z = -1.0;
                            }
                            if g >= half {
                                z = 1.0;
                            }
                            bi[k] *= 0.5 * (1.0 - z);
                            bj[k] *= 0.5 * (1.0 + z);
                        }
                    }
                    Kind::None => {}
                }
            }
        }
        for k in 0..m {
            let mut z = 0.0;
            for i in 0..n {
                z += s.blk_p[i * m + k];
            }
            out[k] = vol[k] * s.blk_p[owner * m + k] * (1.0 / z);
        }
    }

    /// True if [`Self::weight_gradient_block`] applies (unscreened schemes: Becke).
    pub fn has_block_gradient(&self) -> bool {
        !self.screened && matches!(self.kind, Kind::Becke)
    }

    /// `acc[D] += sum_k f[k] dw_k / dR_D` for a block of points (positions `px, py, pz`, owners
    /// `owner`, atomic weights `vol`), each point moving rigidly with its owner. Unscreened
    /// schemes only ([`Self::has_block_gradient`]): every cell function is a product over all
    /// atoms, so the work is O(N^2) per point, vectorized over the block.
    ///
    /// With P_B = prod_C s(nu_BC), Z = sum_B P_B and w = vol P_A / Z, at a fixed point
    /// dw/dR_X = vol sum_B (delta_BA / Z - P_A / Z^2) dP_B/dR_X, and
    /// dP_B/dR_X = P_B sum_C s'(nu_BC) / s(nu_BC) dnu_BC/dR_X. A pair (B, C) moves only B and C,
    /// and since nu_CB = -nu_BC its two cell factors enter with one coefficient Q:
    /// dR_B gets Q (u_B / R_BC + t e_BC), dR_C gets -Q (u_C / R_BC + t e_BC) (u the unit vectors
    /// from the point to the atoms, e_BC = (R_B - R_C) / R_BC, t = -(d_B - d_C) / R_BC^2). The
    /// owner's derivative is minus the sum over all other atoms (translational invariance).
    /// As in [`Self::weight_gradient`], points with P_A = 0 and cell functions that are exactly
    /// zero contribute nothing.
    #[allow(clippy::too_many_arguments, reason = "a kernel's arrays and sizes")]
    pub fn weight_gradient_block(
        &self,
        px: &[f64],
        py: &[f64],
        pz: &[f64],
        owner: &[u32],
        vol: &[f64],
        f: &[f64],
        acc: &mut [[f64; 3]],
        s: &mut Scratch,
    ) {
        assert!(self.has_block_gradient());
        weight_gradient_block_isa(self, px, py, pz, owner, vol, f, acc, s)
    }

    /// Body of [`Self::weight_gradient_block`] (inlined into the per-ISA copies). Atoms whose
    /// terms are all negligible (|c_B P_B / min_C s_BC| summed over the block below
    /// 1e-18 sum |f vol| / N, which bounds every derivative term of the atom's cell function)
    /// are left out of the pair loop unless their partner is active.
    #[inline(always)]
    #[allow(clippy::too_many_arguments, reason = "a kernel's arrays and sizes")]
    fn wgb_body(
        &self,
        px: &[f64],
        py: &[f64],
        pz: &[f64],
        owner: &[u32],
        vol: &[f64],
        f: &[f64],
        acc: &mut [[f64; 3]],
        s: &mut Scratch,
    ) {
        let n = self.natm();
        let m = px.len();
        if n < 2 || m == 0 {
            return;
        }
        // distances and unit vectors (blk_d: d, then ux, uy, uz), cell functions (blk_p)
        s.blk_d.clear();
        s.blk_d.resize(4 * n * m, 0.0);
        s.blk_p.clear();
        s.blk_p.resize(3 * n * m, 1.0);
        let (dd, uu) = s.blk_d.split_at_mut(n * m);
        for b in 0..n {
            let c = self.xyz[b];
            for k in 0..m {
                let (dx, dy, dz) = (c[0] - px[k], c[1] - py[k], c[2] - pz[k]);
                let d = (dx * dx + dy * dy + dz * dz).sqrt();
                dd[b * m + k] = d;
                let inv = if d > 0.0 { 1.0 / d } else { 0.0 };
                uu[(3 * b) * m + k] = dx * inv;
                uu[(3 * b + 1) * m + k] = dy * inv;
                uu[(3 * b + 2) * m + k] = dz * inv;
            }
        }
        let (pp, rest) = s.blk_p.split_at_mut(n * m);
        let (aa, sm) = rest.split_at_mut(n * m);
        // pass 1: P_B = prod_C s(nu_BC), and the smallest factor of each P_B (sm, starts at 1)
        for i in 1..n {
            let (lo, hi) = pp.split_at_mut(i * m);
            let pi = &mut hi[..m];
            let (slo, shi) = sm.split_at_mut(i * m);
            let si = &mut shi[..m];
            let di = &dd[i * m..(i + 1) * m];
            for j in 0..i {
                let pj = &mut lo[j * m..(j + 1) * m];
                let sj = &mut slo[j * m..(j + 1) * m];
                let dj = &dd[j * m..(j + 1) * m];
                let fac = self.inv_r(i, j);
                let a = if self.adjust { self.a_ij(i, j) } else { 0.0 };
                for k in 0..m {
                    let mut g = (di[k] - dj[k]) * fac;
                    g += a * (1.0 - g * g);
                    let t = becke_s(g);
                    let (fi, fj) = (0.5 - t, 0.5 + t);
                    pi[k] *= fi;
                    pj[k] *= fj;
                    si[k] = si[k].min(fi);
                    sj[k] = sj[k].min(fj);
                }
            }
        }
        // c_B P_B with c_B = f vol (delta_BA / Z - P_A / Z^2); zero where P_A = 0
        let mut cfac = vec![0.0; 2 * m];
        for k in 0..m {
            let mut z = 0.0;
            for b in 0..n {
                z += pp[b * m + k];
            }
            let own = owner[k] as usize;
            let pa = pp[own * m + k];
            if pa == 0.0 || f[k] == 0.0 || z == 0.0 {
                cfac[k] = 0.0;
                cfac[m + k] = 0.0;
            } else {
                let iz = 1.0 / z;
                cfac[k] = -f[k] * vol[k] * pa * iz * iz;
                cfac[m + k] = f[k] * vol[k] * iz;
            }
        }
        let scale: f64 = (0..m).map(|k| (f[k] * vol[k]).abs()).sum::<f64>() * 1e-18 / n as f64;
        let mut active = vec![false; n];
        for b in 0..n {
            let row = &mut pp[b * m..(b + 1) * m];
            let mut bound = 0.0;
            for k in 0..m {
                let c = cfac[k]
                    + if owner[k] as usize == b {
                        cfac[m + k]
                    } else {
                        0.0
                    };
                row[k] *= c;
                let smin = sm[b * m + k];
                if row[k] != 0.0 && smin > 0.0 {
                    bound += (row[k] / smin).abs();
                }
            }
            active[b] = bound > scale;
        }
        // pass 2: pair coefficients Q; A_B(k) collects the u_B terms, the e terms go to acc
        aa.fill(0.0);
        for i in 1..n {
            let (alo, ahi) = aa.split_at_mut(i * m);
            let ai = &mut ahi[..m];
            let di = &dd[i * m..(i + 1) * m];
            let ci = &pp[i * m..(i + 1) * m];
            for j in 0..i {
                if !active[i] && !active[j] {
                    continue;
                }
                let aj = &mut alo[j * m..(j + 1) * m];
                let dj = &dd[j * m..(j + 1) * m];
                let cj = &pp[j * m..(j + 1) * m];
                let r = dist(&self.xyz[i], &self.xyz[j]);
                let fac = 1.0 / r;
                let a = if self.adjust { self.a_ij(i, j) } else { 0.0 };
                let mut qt = 0.0;
                for k in 0..m {
                    let mu = (di[k] - dj[k]) * fac;
                    let nu = mu + a * (1.0 - mu * mu);
                    let dnu = 1.0 - 2.0 * a * mu;
                    let s0 = nu;
                    let s1 = (3.0 - s0 * s0) * s0 * 0.5;
                    let s2 = (3.0 - s1 * s1) * s1 * 0.5;
                    let s3 = ((3.0 - s2 * s2) * s2 * 0.5) * 0.5;
                    let ds = -0.5
                        * (1.5 * (1.0 - s2 * s2))
                        * (1.5 * (1.0 - s1 * s1))
                        * (1.5 * (1.0 - s0 * s0))
                        * dnu;
                    let (sij, sji) = (0.5 - s3, 0.5 + s3);
                    let qi = if ci[k] != 0.0 { ci[k] * ds / sij } else { 0.0 };
                    let qj = if cj[k] != 0.0 { cj[k] * ds / sji } else { 0.0 };
                    let q = qi - qj;
                    ai[k] += q * fac;
                    aj[k] -= q * fac;
                    qt -= q * (di[k] - dj[k]) * fac * fac;
                }
                let (ri, rj) = (self.xyz[i], self.xyz[j]);
                for x in 0..3 {
                    let e = (ri[x] - rj[x]) * fac;
                    acc[i][x] += qt * e;
                    acc[j][x] -= qt * e;
                }
            }
        }
        // u terms; the owner takes minus the sum over all atoms of each point's terms
        let mut tsum = vec![[0.0f64; 3]; m];
        for b in 0..n {
            let ab = &aa[b * m..(b + 1) * m];
            for x in 0..3 {
                let ub = &uu[(3 * b + x) * m..(3 * b + x + 1) * m];
                let mut sx = 0.0;
                for k in 0..m {
                    let v = ab[k] * ub[k];
                    sx += v;
                    tsum[k][x] += v;
                }
                acc[b][x] += sx;
            }
        }
        // the e terms of a pair cancel in the per-point sum (equal and opposite)
        for k in 0..m {
            let own = owner[k] as usize;
            for x in 0..3 {
                acc[own][x] -= tsum[k][x];
            }
        }
    }

    /// Accumulates d P_b / d R_D (point fixed) into `acc` (indexed by atom), marking touched atoms.
    #[allow(
        clippy::too_many_arguments,
        reason = "the point, its atoms and their outputs"
    )]
    fn cell_product_grad(
        &self,
        p: &[f64; 3],
        b: usize,
        d_b: f64,
        list: &[(usize, f64)],
        f: &mut Vec<f64>,
        df: &mut Vec<f64>,
        pre: &mut Vec<f64>,
        acc: &mut [[f64; 3]],
        touched: &mut Vec<usize>,
        mark: &mut [bool],
    ) -> f64 {
        f.clear();
        df.clear();
        for &(c, d_c) in list {
            if c == b {
                continue;
            }
            let (fv, dv, ..) = self.factor_d(b, c, d_b, d_c);
            f.push(fv);
            df.push(dv);
        }
        let m = f.len();
        pre.clear();
        pre.resize(m + 1, 1.0);
        for i in 0..m {
            pre[i + 1] = pre[i] * f[i];
        }
        let prod = pre[m];
        let mut suf = 1.0;
        let mut k = m;
        // iterate backwards to build suffix products on the fly
        let others: Vec<(usize, f64)> = list.iter().copied().filter(|x| x.0 != b).collect();
        let rb = self.xyz[b];
        let ub = if d_b > 0.0 {
            [
                (rb[0] - p[0]) / d_b,
                (rb[1] - p[1]) / d_b,
                (rb[2] - p[2]) / d_b,
            ]
        } else {
            [0.0; 3]
        };
        while k > 0 {
            k -= 1;
            let q = pre[k] * suf;
            suf *= f[k];
            let coef = q * df[k];
            if coef == 0.0 {
                continue;
            }
            let (c, d_c) = others[k];
            let (_, _, fac, dfac, r) = self.factor_d(b, c, d_b, d_c);
            let rc = self.xyz[c];
            let uc = if d_c > 0.0 {
                [
                    (rc[0] - p[0]) / d_c,
                    (rc[1] - p[1]) / d_c,
                    (rc[2] - p[2]) / d_c,
                ]
            } else {
                [0.0; 3]
            };
            let e = [
                (rb[0] - rc[0]) / r,
                (rb[1] - rc[1]) / r,
                (rb[2] - rc[2]) / r,
            ];
            let t = (d_b - d_c) * dfac;
            for x in 0..3 {
                let dmu_db = fac * ub[x] + t * e[x];
                let dmu_dc = -fac * uc[x] - t * e[x];
                acc[b][x] += coef * dmu_db;
                acc[c][x] += coef * dmu_dc;
            }
            for &a in &[b, c] {
                if !mark[a] {
                    mark[a] = true;
                    touched.push(a);
                }
            }
        }
        prod
    }

    /// Partitioned weight and its derivatives with respect to all nuclear positions, the point
    /// moving rigidly with its owning atom. Appends `(atom, dw/dR_atom)` to `out` (cleared first)
    /// for every atom with a possibly nonzero derivative; derivatives sum to zero
    /// (translational invariance).
    pub fn weight_gradient(
        &self,
        p: [f64; 3],
        owner: usize,
        vol: f64,
        s: &mut Scratch,
        out: &mut Vec<(usize, [f64; 3])>,
    ) -> f64 {
        out.clear();
        if self.kind == Kind::None || self.natm() == 1 {
            return vol;
        }
        let d_a = dist(&p, &self.xyz[owner]);
        if self.screened && d_a <= self.fast_r[owner] {
            return vol;
        }
        let (d_n, list_rho) = if self.screened {
            let (nn, d_n) = self.nearest(&p, d_a, owner);
            if nn != owner && self.factor(owner, nn, d_a, d_n) == 0.0 {
                return 0.0;
            }
            (d_n, self.list_radius(owner, d_a, d_n))
        } else {
            (0.0, f64::INFINITY)
        };
        let mut list = std::mem::take(&mut s.list);
        let mut l2 = std::mem::take(&mut s.list2);
        self.gather(&p, list_rho, &mut list);

        // reset accumulators
        for &t in &s.touched {
            s.dpa[t] = [0.0; 3];
            s.dz[t] = [0.0; 3];
            s.mark[t] = false;
        }
        s.touched.clear();
        let mut pa = 0.0;
        let mut z = 0.0;
        for i in 0..list.len() {
            let (b, d_b) = list[i];
            if self.screened && d_b >= self.cut_zero(b, d_n) {
                continue;
            }
            let use_l2 = self.screened && self.cut_one(b, d_b) > list_rho;
            if use_l2 {
                self.gather(&p, self.cut_one(b, d_b), &mut l2);
            }
            let lb: &[(usize, f64)] = if use_l2 { &l2 } else { &list };
            // screened: atoms beyond cut(d_b) contribute factors of exactly 1 (zero derivative)
            let filtered;
            let lb = if self.screened {
                let c = self.cut_one(b, d_b);
                filtered = lb.iter().copied().filter(|x| x.1 < c).collect::<Vec<_>>();
                &filtered[..]
            } else {
                lb
            };
            // value first (cheap, early exit) -- the derivative of a zero cell function is zero
            let pb = self.cell_product(b, d_b, lb);
            if pb == 0.0 {
                continue;
            }
            if b == owner {
                // dP_A accumulates into dpa; also add to dz below
                let v = self.cell_product_grad(
                    &p,
                    b,
                    d_b,
                    lb,
                    &mut s.f,
                    &mut s.df,
                    &mut s.pre,
                    &mut s.dpa,
                    &mut s.touched,
                    &mut s.mark,
                );
                pa = v;
                z += v;
            } else {
                let v = self.cell_product_grad(
                    &p,
                    b,
                    d_b,
                    lb,
                    &mut s.f,
                    &mut s.df,
                    &mut s.pre,
                    &mut s.dz,
                    &mut s.touched,
                    &mut s.mark,
                );
                z += v;
            }
        }
        s.list = list;
        s.list2 = l2;
        if pa == 0.0 {
            // weight is zero, but its derivative may not be (P_A is flat at zero for all schemes)
            for &t in &s.touched {
                s.dpa[t] = [0.0; 3];
                s.dz[t] = [0.0; 3];
                s.mark[t] = false;
            }
            s.touched.clear();
            return 0.0;
        }
        // dZ = dP_A + sum_{B != A} dP_B
        let inv_z = 1.0 / z;
        let mut sum = [0.0; 3];
        for &t in &s.touched {
            if t == owner {
                continue;
            }
            let mut g = [0.0; 3];
            for x in 0..3 {
                let dzt = s.dz[t][x] + s.dpa[t][x];
                g[x] = vol * (s.dpa[t][x] * inv_z - pa * dzt * inv_z * inv_z);
                sum[x] += g[x];
            }
            out.push((t, g));
        }
        out.push((owner, [-sum[0], -sum[1], -sum[2]]));
        for &t in &s.touched {
            s.dpa[t] = [0.0; 3];
            s.dz[t] = [0.0; 3];
            s.mark[t] = false;
        }
        s.touched.clear();
        vol * pa * inv_z
    }
}

/// [`Partitioner::wgb_body`] as a function of the partitioner, compiled per instruction set by
/// `weight_gradient_block_isa` (linus-num's `multiversion!`).
#[inline(always)]
#[allow(clippy::too_many_arguments, reason = "a kernel's arrays and sizes")]
fn wgb_isa_body(
    p: &Partitioner,
    px: &[f64],
    py: &[f64],
    pz: &[f64],
    owner: &[u32],
    vol: &[f64],
    f: &[f64],
    acc: &mut [[f64; 3]],
    s: &mut Scratch,
) {
    p.wgb_body(px, py, pz, owner, vol, f, acc, s)
}

linus_num::multiversion!(fn weight_gradient_block_isa(p: &Partitioner, px: &[f64], py: &[f64], pz: &[f64], owner: &[u32], vol: &[f64], f: &[f64], acc: &mut [[f64; 3]], s: &mut Scratch) -> () = wgb_isa_body);
