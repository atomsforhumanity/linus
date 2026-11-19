//! SMD solvation model (Marenich, Cramer & Truhlar, J. Phys. Chem. B 113, 6378 (2009)):
//! the non-electrostatic CDS term (cavitation, dispersion, solvent structure) with analytic
//! gradient and Hessian, and the SMD intrinsic Coulomb radii for the bulk-electrostatics PCM.
//!
//! # API (for the PCM/SCF wiring)
//!
//! - [`coulomb_radii`]`(z, alpha, scheme)` -> radii in bohr for the PCM cavity (SMD Table 3, the
//!   oxygen radius depends on the solvent's Abraham acidity, eq 16; other elements per
//!   [`RadiiScheme`]). [`coulomb_radii_angstrom`] gives the same in Å.
//! - [`cds`]`(z, coords_bohr, &SmdDescriptors, &CdsConfig)` -> [`CdsResult`]: G_CDS in hartree,
//!   per-atom contributions, SASA and atomic surface tensions, and on request the gradient
//!   (Eh/bohr, natom x 3) and Hessian (Eh/bohr^2, 3N x 3N row-major). Add `energy` to the SCF
//!   energy, `gradient`/`hessian` to the nuclear derivatives; G_CDS does not depend on the
//!   density, so it enters no Fock matrix or response equation.
//! - [`cds_named`] looks the solvent up in [`crate::db`] first.
//!
//! # Model
//!
//! G_CDS = sum_k (sigma_k + sigma_M) A_k, with A_k the solvent-accessible surface area of atom k
//! (exact SASA of spheres with radius r_vdW + 0.4 Å; r_vdW from the CRC 91st-ed. table of
//! Mantina et al. that the Minnesota code uses, Bondi's 1.20 Å for H), sigma_k the atomic surface
//! tension (eq 7, the SM8 COT switching functions T(R) = exp(dR / (R - R0 - dR)) for R < R0 + dR)
//! and sigma_M = 0.35 gamma - 4.19 phi^2 - 6.68 psi^2 the molecular surface tension (eq 9).
//! Water has its own parameter set (Table 4, first column) and sigma_M = 0. The functional forms
//! (which neighbours enter which COT, the N-C term (sum_C T_NC (sum_k T_Ck)^2)^1.3 and the
//! squared C-N term) follow the Minnesota CDS code as distributed with NWChem and PySCF
//! (mnsol.F); the parameters are Table 4 and 5 of the SMD paper. Not every SMD element has a
//! surface tension: P only through O-P, I none (as in the paper).
//!
//! # Conventions
//!
//! [`ParamConvention::Mnsol`] (default) reproduces the Minnesota/NWChem code that PySCF 2.15
//! calls: its radii and surface tensions are Fortran single-precision literals (1.20 becomes
//! 1.2000000476837158) and it converts bohr to Å with 0.52917724924. This reproduces PySCF to
//! ~1e-14 relative. [`ParamConvention::Decimal`] uses the published decimals in double precision
//! and linus's CODATA bohr; it differs by ~1e-8 relative (~1e-10 Eh).
//!
//! # Smoothness
//!
//! The SASA is exact (see `super::smd_sasa`), as in PySCF, NWChem and ORCA. It is continuous,
//! but its gradient jumps where two probe-inflated spheres start to touch (d = R_i + R_j) and its
//! Hessian diverges where two caps on a sphere start to overlap. The COT switching functions are
//! C-infinity. The gradient is exact everywhere else; FD checks must avoid those events.
//! [`CdsConfig::contact_smoothing`] > 0 softens the contact onset (off by default).
//!
//! Recommended use (adopted 2026-11-07): single-point energies keep the exact SASA
//! (`contact_smoothing = 0`, identical to PySCF/ORCA); geometry optimizations and frequency
//! (Hessian) calculations use `contact_smoothing = 0.1` Å, so the gradient has no jumps at
//! exposed contacts (water dimer: 4.3e-4 Eh/Å) and the Hessian is continuous there. That changes
//! G_CDS by at most ~1e-5 Eh, only for pairs within 0.1 Å of contact. The caller (SCF driver)
//! chooses per task; this module does not change its default.
//!
//! Threads: the SASA is computed per sphere with rayon in the caller's pool; all reductions are
//! sequential in atom order, so results are bitwise identical at any thread count.

use super::db::{self, SmdDescriptors};
use super::smd_sasa::{self, outer, sub, M3, V3};
use linus_core::units::{BOHR_ANGSTROM, BOHR_ANGSTROM_PYSCF, HARTREE_KCAL_MOL};

/// Bohr to Å in the Minnesota/NWChem CDS code (mnsol.F TOANGS).
pub const MNSOL_BOHR_ANGSTROM: f64 = 0.52917724924;
/// Hartree to kcal/mol in mnsol.F (TOKCAL) and PySCF's smd.py.
pub const MNSOL_HARTREE_KCAL: f64 = 627.509451;
/// SMD solvent probe radius added to the van der Waals radii for the SASA (Å).
pub const SMD_PROBE_RADIUS: f64 = 0.4;

/// How the SMD constants are represented (see the module docs).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum ParamConvention {
    /// Minnesota/NWChem mnsol.F as called by PySCF: single-precision parameter literals,
    /// bohr = 0.52917724924 Å, 627.509451 kcal/mol per hartree.
    #[default]
    Mnsol,
    /// Published decimals in double precision, linus's CODATA 2022 bohr and hartree.
    Decimal,
}

/// Derivative order of [`cds`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Default)]
pub enum CdsOrder {
    #[default]
    Energy,
    Gradient,
    Hessian,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct CdsConfig {
    pub order: CdsOrder,
    pub convention: ParamConvention,
    /// Overlap window (Å) over which a new sphere contact is smoothed (see
    /// `smd_sasa::contact_distance`); 0 (default) is the exact SASA of SMD/PySCF/ORCA, whose
    /// gradient jumps at sphere contact. 0.1 Å removes the jumps (C2) and changes G_CDS only for
    /// pairs within 0.1 Å of contact. Use 0 for single points and 0.1 for geometry optimizations
    /// and frequencies (see the module docs).
    pub contact_smoothing: f64,
}

/// G_CDS and its parts. Energies in hartree, lengths in bohr unless stated.
#[derive(Clone, Debug)]
pub struct CdsResult {
    /// G_CDS (Eh).
    pub energy: f64,
    /// Per-atom contributions (sigma_k + sigma_M) A_k (Eh); they sum to `energy`.
    pub per_atom: Vec<f64>,
    /// Solvent-accessible surface area per atom (Å^2).
    pub sasa: Vec<f64>,
    /// Atomic surface tensions sigma_k (cal mol^-1 Å^-2).
    pub atomic_surface_tension: Vec<f64>,
    /// Molecular surface tension sigma_M (cal mol^-1 Å^-2).
    pub molecular_surface_tension: f64,
    /// dG/dR (Eh/bohr), natom x 3.
    pub gradient: Option<Vec<[f64; 3]>>,
    /// d2G/dR2 (Eh/bohr^2), 3N x 3N row-major.
    pub hessian: Option<Vec<f64>>,
}

#[derive(Debug, thiserror::Error)]
pub enum SmdError {
    #[error("unknown solvent {0:?}")]
    UnknownSolvent(String),
    #[error("solvent {0:?} has no SMD descriptors")]
    NoDescriptors(String),
    #[error("{0}")]
    Input(String),
}

// ----------------------------------------------------------------------------------------------
// radii

/// Van der Waals radii (Å) of the Minnesota CDS code (mnsol.F VDWRAD) for Z = 1..102: H from
/// Bondi, the rest from Mantina, Valero, Cramer & Truhlar in the CRC Handbook 91st ed.
/// (Bondi's values where Bondi gave one).
const MNSOL_VDW: [f64; 102] = [
    1.20, 1.40, 1.82, 1.53, 1.92, 1.70, 1.55, 1.52, 1.47, 1.54, 2.27, 1.73, 1.84, 2.10, 1.80, 1.80,
    1.75, 1.88, 2.75, 2.31, 2.16, 1.87, 1.79, 1.89, 1.97, 1.94, 1.92, 1.84, 1.86, 2.10, 1.87, 2.11,
    1.85, 1.90, 1.85, 2.02, 3.03, 2.49, 2.19, 1.86, 2.07, 2.09, 2.09, 2.07, 1.95, 2.02, 2.03, 2.30,
    1.93, 2.17, 2.06, 2.06, 1.98, 2.16, 3.43, 2.68, 2.40, 2.35, 2.39, 2.29, 2.36, 2.29, 2.33, 2.37,
    2.21, 2.29, 2.16, 2.35, 2.27, 2.42, 2.21, 2.12, 2.17, 2.10, 2.17, 2.16, 2.02, 2.09, 2.17, 2.09,
    1.96, 2.02, 2.07, 1.97, 2.02, 2.20, 3.48, 2.83, 2.60, 2.37, 2.43, 2.40, 2.21, 2.43, 2.44, 2.45,
    2.44, 2.45, 2.45, 2.45, 2.46, 2.46,
];

/// PySCF 2.15 pyscf.data.radii.VDW in Å for Z = 0..103 (Bondi, with Mantina et al. for some main-
/// group elements; 1.999999 where neither has a value).
const PYSCF_VDW: [f64; 104] = {
    const U: f64 = 1.999999;
    [
        U, 1.20, 1.40, 1.82, 1.53, 1.92, 1.70, 1.55, 1.52, 1.47, 1.54, 2.27, 1.73, 1.84, 2.10,
        1.80, 1.80, 1.75, 1.88, 2.75, 2.31, U, U, U, U, U, U, U, 1.63, 1.40, 1.39, 1.87, 2.11,
        1.85, 1.90, 1.85, 2.02, 3.03, 2.49, U, U, U, U, U, U, U, 1.63, 1.72, 1.58, 1.93, 2.17,
        2.06, 2.06, 1.98, 2.16, 3.43, 2.49, U, U, U, U, U, U, U, U, U, U, U, U, U, U, U, U, U, U,
        U, U, U, 1.75, 1.66, 1.55, 1.96, 2.02, 2.07, 1.97, 2.02, 2.20, 3.48, 2.83, U, U, U, 1.86,
        U, U, U, U, U, U, U, U, U, U, U,
    ]
};

/// Van der Waals radius (Å) used for the SMD SASA (before adding the probe radius): the
/// mnsol.F table for Z <= 102, 2.0 Å beyond (the paper's default); 0 for Z = 0 (no sphere).
pub fn sasa_vdw_radius(z: u32, convention: ParamConvention) -> f64 {
    match z {
        0 => 0.0,
        1..=102 => lit(MNSOL_VDW[z as usize - 1], convention),
        _ => 2.0,
    }
}

/// Which radii to use for elements without an SMD-fitted Coulomb radius (and for Br, I).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum RadiiScheme {
    /// SMD 2009 (Table 3: H C N O F Si P S Cl Br) and, for every other element, the mnsol.F
    /// van der Waals table (Bondi where Bondi has a value, Mantina et al./CRC otherwise), 2.0 Å
    /// beyond Z = 102.
    #[default]
    Smd,
    /// As `Smd`, with the SMD18 radii Br 2.60 and I 2.74 Å (Engelage et al., Chem. Eur. J. 24,
    /// 15983 (2018)).
    Smd18,
    /// PySCF 2.15 `smd.smd_radii`: SMD + SMD18 Br/I, other elements from pyscf.data.radii.VDW
    /// (1.999999 Å if unknown); returned in bohr with PySCF's bohr.
    Pyscf,
}

/// SMD oxygen Coulomb radius (Å), eq 16: 1.52 for alpha >= 0.43, else 1.52 + 1.8 (0.43 - alpha).
pub fn oxygen_radius(alpha: f64) -> f64 {
    if alpha >= 0.43 {
        1.52
    } else {
        1.52 + 1.8 * (0.43 - alpha)
    }
}

/// SMD intrinsic Coulomb radii in Å for atomic numbers `z`; `alpha` is the solvent's Abraham
/// acidity (only the O radius depends on it; water: 0.82).
pub fn coulomb_radii_angstrom(z: &[u32], alpha: f64, scheme: RadiiScheme) -> Vec<f64> {
    z.iter()
        .map(|&z| {
            let fitted = match z {
                1 => Some(1.20),
                6 => Some(1.85),
                7 => Some(1.89),
                8 => Some(oxygen_radius(alpha)),
                9 => Some(1.73),
                14 => Some(2.47),
                15 => Some(2.12),
                16 => Some(2.49),
                17 => Some(2.38),
                35 => Some(if scheme == RadiiScheme::Smd {
                    3.06
                } else {
                    2.60
                }),
                53 if scheme != RadiiScheme::Smd => Some(2.74),
                _ => None,
            };
            fitted.unwrap_or_else(|| match scheme {
                RadiiScheme::Pyscf => PYSCF_VDW.get(z as usize).copied().unwrap_or(1.999999),
                _ => match z {
                    1..=102 => MNSOL_VDW[z as usize - 1],
                    _ => 2.0,
                },
            })
        })
        .collect()
}

/// SMD intrinsic Coulomb radii in bohr (linus's bohr; PySCF's bohr for [`RadiiScheme::Pyscf`]).
pub fn coulomb_radii(z: &[u32], alpha: f64, scheme: RadiiScheme) -> Vec<f64> {
    let b = if scheme == RadiiScheme::Pyscf {
        BOHR_ANGSTROM_PYSCF
    } else {
        BOHR_ANGSTROM
    };
    coulomb_radii_angstrom(z, alpha, scheme)
        .into_iter()
        .map(|r| r / b)
        .collect()
}

// ----------------------------------------------------------------------------------------------
// surface tensions

/// A Fortran REAL literal (mnsol.F DATA statements) or the decimal itself.
#[inline]
fn lit(x: f64, c: ParamConvention) -> f64 {
    match c {
        ParamConvention::Mnsol => x as f32 as f64,
        ParamConvention::Decimal => x,
    }
}

/// Surface tension coefficients for one solvent (cal mol^-1 Å^-2), SMD Tables 4 and 5.
#[derive(Clone, Copy, Debug)]
struct Tensions {
    h: f64,
    c: f64,
    n: f64,
    o: f64,
    f: f64,
    si: f64,
    s: f64,
    cl: f64,
    br: f64,
    hc: f64,
    ho: f64,
    cc: f64,
    oc: f64,
    oo: f64,
    nc: f64,
    cn: f64,
    nc3: f64,
    on: f64,
    op: f64,
    molecular: f64,
}

fn tensions(sd: &SmdDescriptors, conv: ParamConvention) -> Tensions {
    let l = |x| lit(x, conv);
    if sd.water {
        return Tensions {
            h: l(48.69),
            c: l(129.74),
            n: 0.0,
            o: 0.0,
            f: l(38.18),
            si: 0.0,
            s: l(-9.10),
            cl: l(9.82),
            br: l(-8.72),
            hc: l(-60.77),
            ho: 0.0,
            cc: l(-72.95),
            oc: l(68.69),
            oo: 0.0,
            nc: l(-48.22),
            cn: 0.0,
            nc3: l(84.10),
            on: l(121.98),
            op: l(68.85),
            molecular: 0.0,
        };
    }
    let (n, a, b) = (sd.n, sd.alpha, sd.beta);
    // sigma = sigma_n n + sigma_alpha alpha + sigma_beta beta (eq 8), evaluated as in mnsol.F
    let s3 = |sn: f64, sa: f64, sb: f64| l(sn) * n + l(sa) * a + l(sb) * b;
    Tensions {
        h: 0.0,
        c: s3(58.10, 48.10, 32.87),
        n: s3(32.62, 0.0, 0.0),
        o: s3(-17.56, 193.06, -43.79),
        f: 0.0,
        si: s3(-18.04, 0.0, 0.0),
        s: s3(-33.17, 0.0, 0.0),
        cl: s3(-24.31, 0.0, 0.0),
        br: s3(-35.42, 0.0, 0.0),
        hc: s3(-36.37, 0.0, 0.0),
        ho: s3(-19.39, 0.0, 0.0),
        cc: s3(-62.05, 0.0, 0.0),
        oc: s3(-15.70, 95.99, 0.0),
        oo: s3(0.0, 0.0, -128.16),
        nc: s3(0.0, -41.00, 0.0),
        cn: s3(-99.76, 152.20, 0.0),
        nc3: 0.0,
        on: s3(0.0, 0.0, 79.13),
        op: 0.0,
        // eq 9: sigma_gamma gamma/gamma0 + sigma_beta2 beta^2 + sigma_phi2 phi^2 + sigma_psi2 psi^2
        molecular: l(0.35) * sd.gamma
            + 0.0 * b * b
            + l(-4.19) * sd.phi * sd.phi
            + l(-6.68) * sd.psi * sd.psi,
    }
}

/// COT switching function exp(dr / (r - r0 - dr)) for r < r0 + dr, and its first two
/// derivatives with respect to r.
#[inline]
fn cot(r: f64, r0: f64, dr: f64) -> (f64, f64, f64) {
    let t = r - r0 - dr;
    if t >= 0.0 {
        return (0.0, 0.0, 0.0);
    }
    let f = (dr / t).exp();
    let t2 = t * t;
    (
        f,
        -dr / t2 * f,
        (dr * dr / (t2 * t2) + 2.0 * dr / (t2 * t)) * f,
    )
}

/// R0 (Å) of the COT between a carbon and a neighbour of atomic number z inside the N-C term
/// (mnsol.F RKKVAL row "C"; dR = 0.3 Å); elements without an entry do not count.
fn r0_carbon(z: u32) -> Option<f64> {
    match z {
        1 => Some(1.55),
        6..=9 => Some(1.84),
        15 | 16 => Some(2.20),
        17 => Some(2.10),
        35 => Some(2.30),
        53 => Some(2.60),
        _ => None,
    }
}

/// A scalar function of the geometry with sparse first and second derivatives with respect to
/// atomic positions.
#[derive(Clone, Debug, Default)]
struct Sp {
    v: f64,
    g: Vec<(usize, V3)>,
    h: Vec<(usize, usize, M3)>,
}

impl Sp {
    fn constant(v: f64) -> Sp {
        Sp {
            v,
            ..Default::default()
        }
    }

    /// f(|r_i - r_j|) with (f, f', f'')
    fn pair(i: usize, j: usize, pos: &[V3], f: (f64, f64, f64), hess: bool) -> Sp {
        let d = sub(pos[i], pos[j]);
        let r = smd_sasa::dot(d, d).sqrt();
        let u = smd_sasa::scale(d, 1.0 / r);
        let (f0, f1, f2) = f;
        let gi = smd_sasa::scale(u, f1);
        let mut s = Sp {
            v: f0,
            g: vec![(i, gi), (j, smd_sasa::scale(gi, -1.0))],
            h: Vec::new(),
        };
        if hess {
            let uu = outer(u, u);
            let mut q = [[0.0; 3]; 3];
            for x in 0..3 {
                for y in 0..3 {
                    let id = if x == y { 1.0 } else { 0.0 };
                    q[x][y] = f2 * uu[x][y] + f1 * (id - uu[x][y]) / r;
                }
            }
            let mq = q.map(|row| row.map(|v| -v));
            s.h = vec![(i, i, q), (i, j, mq), (j, i, mq), (j, j, q)];
        }
        s
    }

    fn axpy(&mut self, a: f64, x: &Sp) {
        self.v += a * x.v;
        self.g
            .extend(x.g.iter().map(|&(i, g)| (i, smd_sasa::scale(g, a))));
        self.h.extend(
            x.h.iter()
                .map(|&(i, j, m)| (i, j, m.map(|row| row.map(|v| a * v)))),
        );
    }

    #[allow(clippy::needless_range_loop, reason = "indexes several arrays in step")]
    fn compact(&mut self) {
        self.g.sort_by_key(|e| e.0);
        let mut g: Vec<(usize, V3)> = Vec::with_capacity(self.g.len());
        for &(i, v) in &self.g {
            match g.last_mut() {
                Some(last) if last.0 == i => last.1 = smd_sasa::add(last.1, v),
                _ => g.push((i, v)),
            }
        }
        self.g = g;
        self.h.sort_by_key(|e| (e.0, e.1));
        let mut h: Vec<(usize, usize, M3)> = Vec::with_capacity(self.h.len());
        for &(i, j, m) in &self.h {
            match h.last_mut() {
                Some(last) if last.0 == i && last.1 == j => {
                    for x in 0..3 {
                        for y in 0..3 {
                            last.2[x][y] += m[x][y];
                        }
                    }
                }
                _ => h.push((i, j, m)),
            }
        }
        self.h = h;
    }

    /// self * o
    fn mul(&self, o: &Sp, hess: bool) -> Sp {
        let mut s = Sp::constant(self.v * o.v);
        s.g.extend(o.g.iter().map(|&(i, g)| (i, smd_sasa::scale(g, self.v))));
        s.g.extend(self.g.iter().map(|&(i, g)| (i, smd_sasa::scale(g, o.v))));
        if hess {
            s.h.extend(
                o.h.iter()
                    .map(|&(i, j, m)| (i, j, m.map(|r| r.map(|v| self.v * v)))),
            );
            s.h.extend(
                self.h
                    .iter()
                    .map(|&(i, j, m)| (i, j, m.map(|r| r.map(|v| o.v * v)))),
            );
            for &(i, a) in &self.g {
                for &(j, b) in &o.g {
                    s.h.push((i, j, outer(a, b)));
                    s.h.push((j, i, outer(b, a)));
                }
            }
        }
        s.compact();
        s
    }

    /// self^p for self >= 0 (0 maps to 0 with zero derivatives, the limit for p > 1)
    fn powf(&self, p: f64, hess: bool) -> Sp {
        if self.v <= 0.0 {
            return Sp::constant(0.0);
        }
        let c1 = p * self.v.powf(p - 1.0);
        let mut s = Sp::constant(self.v.powf(p));
        s.g.extend(self.g.iter().map(|&(i, g)| (i, smd_sasa::scale(g, c1))));
        if hess {
            let c2 = p * (p - 1.0) * self.v.powf(p - 2.0);
            s.h.extend(
                self.h
                    .iter()
                    .map(|&(i, j, m)| (i, j, m.map(|r| r.map(|v| c1 * v)))),
            );
            for &(i, a) in &self.g {
                for &(j, b) in &self.g {
                    s.h.push((i, j, outer(a, b).map(|r| r.map(|v| c2 * v))));
                }
            }
        }
        s.compact();
        s
    }
}

/// Atomic surface tension sigma_k (eq 7) with derivatives; positions in Å.
#[allow(clippy::needless_range_loop, reason = "indexes several arrays in step")]
fn atomic_tension(k: usize, z: &[u32], pos: &[V3], t: &Tensions, hess: bool) -> Sp {
    let nat = z.len();
    let pair = |j: usize, r0: f64, dr: f64| {
        let d = sub(pos[k], pos[j]);
        Sp::pair(k, j, pos, cot(smd_sasa::dot(d, d).sqrt(), r0, dr), hess)
    };
    // sum over neighbours of element zj (j != k) of COT(r_kj; r0, dr)
    let sum_cot = |zj: u32, r0: f64, dr: f64| {
        let mut s = Sp::default();
        for j in 0..nat {
            if j != k && z[j] == zj {
                s.axpy(1.0, &pair(j, r0, dr));
            }
        }
        s
    };
    let mut s = Sp::constant(match z[k] {
        1 => t.h,
        6 => t.c,
        7 => t.n,
        8 => t.o,
        9 => t.f,
        14 => t.si,
        16 => t.s,
        17 => t.cl,
        35 => t.br,
        _ => 0.0,
    });
    match z[k] {
        1 => {
            if t.hc != 0.0 {
                s.axpy(t.hc, &sum_cot(6, 1.55, 0.3));
            }
            if t.ho != 0.0 {
                s.axpy(t.ho, &sum_cot(8, 1.55, 0.3));
            }
        }
        6 => {
            if t.cc != 0.0 {
                s.axpy(t.cc, &sum_cot(6, 1.84, 0.3));
            }
            if t.cn != 0.0 {
                let x = sum_cot(7, 1.84, 0.3);
                s.axpy(t.cn, &x.mul(&x, hess));
            }
        }
        7 => {
            if t.nc != 0.0 {
                // (sum_{C j} T(r_kj) (sum_{l != k, j} T_C(r_jl))^2)^1.3
                let mut x = Sp::default();
                for j in 0..nat {
                    if j == k || z[j] != 6 {
                        continue;
                    }
                    let tkj = pair(j, 1.84, 0.3);
                    if tkj.v == 0.0 {
                        continue;
                    }
                    let mut tc = Sp::default();
                    for l in 0..nat {
                        if l == k || l == j {
                            continue;
                        }
                        if let Some(r0) = r0_carbon(z[l]) {
                            let d = sub(pos[j], pos[l]);
                            tc.axpy(
                                1.0,
                                &Sp::pair(
                                    j,
                                    l,
                                    pos,
                                    cot(smd_sasa::dot(d, d).sqrt(), r0, 0.3),
                                    hess,
                                ),
                            );
                        }
                    }
                    tc.compact();
                    x.axpy(1.0, &tkj.mul(&tc.mul(&tc, hess), hess));
                }
                x.compact();
                s.axpy(t.nc, &x.powf(1.3, hess));
            }
            if t.nc3 != 0.0 {
                s.axpy(t.nc3, &sum_cot(6, 1.225, 0.065));
            }
        }
        8 => {
            if t.oc != 0.0 {
                s.axpy(t.oc, &sum_cot(6, 1.33, 0.1));
            }
            if t.on != 0.0 {
                s.axpy(t.on, &sum_cot(7, 1.50, 0.3));
            }
            if t.oo != 0.0 {
                s.axpy(t.oo, &sum_cot(8, 1.80, 0.3));
            }
            if t.op != 0.0 {
                s.axpy(t.op, &sum_cot(15, 2.10, 0.3));
            }
        }
        _ => {}
    }
    s.compact();
    s
}

// ----------------------------------------------------------------------------------------------
// G_CDS

/// SMD G_CDS for atoms `z` at `coords_bohr` in the solvent `sd`.
#[allow(clippy::needless_range_loop, reason = "indexes several arrays in step")]
pub fn cds(
    z: &[u32],
    coords_bohr: &[[f64; 3]],
    sd: &SmdDescriptors,
    opts: &CdsConfig,
) -> CdsResult {
    let nat = z.len();
    assert_eq!(
        coords_bohr.len(),
        nat,
        "smd::cds: z and coords differ in length"
    );
    let conv = opts.convention;
    let (to_ang, kcal) = match conv {
        ParamConvention::Mnsol => (MNSOL_BOHR_ANGSTROM, MNSOL_HARTREE_KCAL),
        ParamConvention::Decimal => (BOHR_ANGSTROM, HARTREE_KCAL_MOL),
    };
    let grad_on = opts.order >= CdsOrder::Gradient;
    let hess_on = opts.order >= CdsOrder::Hessian;
    let pos: Vec<V3> = coords_bohr
        .iter()
        .map(|c| [c[0] * to_ang, c[1] * to_ang, c[2] * to_ang])
        .collect();
    let rad: Vec<f64> = z
        .iter()
        .map(|&zk| {
            if zk == 0 {
                0.0
            } else {
                sasa_vdw_radius(zk, conv) + SMD_PROBE_RADIUS
            }
        })
        .collect();
    let t = tensions(sd, conv);
    let sasa = smd_sasa::sasa(&pos, &rad, opts.order as usize, opts.contact_smoothing);

    let mut energy = 0.0;
    let mut per_atom = vec![0.0; nat];
    let mut sigma = vec![0.0; nat];
    let n3 = 3 * nat;
    let mut grad = vec![[0.0; 3]; if grad_on { nat } else { 0 }];
    let mut hess = vec![0.0; if hess_on { n3 * n3 } else { 0 }];
    let hadd = |hess: &mut Vec<f64>, i: usize, j: usize, m: &M3, f: f64| {
        for x in 0..3 {
            for y in 0..3 {
                hess[(3 * i + x) * n3 + 3 * j + y] += f * m[x][y];
            }
        }
    };
    for k in 0..nat {
        let sk = atomic_tension(k, z, &pos, &t, hess_on);
        sigma[k] = sk.v;
        let w = (sk.v + t.molecular) * 0.001;
        let s = &sasa[k];
        let ak = s.area;
        per_atom[k] = ak * w;
        energy += ak * w;
        if !grad_on {
            continue;
        }
        // dA_k as a sparse list over atoms (neighbors, then k itself)
        let m = s.neighbors.len();
        let mut ga: Vec<(usize, V3)> = Vec::with_capacity(m + 1);
        let mut gk = [0.0; 3];
        for (t_, &j) in s.neighbors.iter().enumerate() {
            ga.push((j, s.grad[t_]));
            gk = sub(gk, s.grad[t_]);
        }
        ga.push((k, gk));
        for &(j, g) in &ga {
            for x in 0..3 {
                grad[j][x] += w * g[x];
            }
        }
        for &(j, g) in &sk.g {
            for x in 0..3 {
                grad[j][x] += 0.001 * ak * g[x];
            }
        }
        if !hess_on {
            continue;
        }
        // w * d2A_k: expand the neighbor-variable Hessian (d_t = r_t - r_k) to coordinates
        let dim = 3 * m;
        let blk = |a: usize, b: usize| -> M3 {
            let mut o = [[0.0; 3]; 3];
            for x in 0..3 {
                for y in 0..3 {
                    o[x][y] = s.hess[(3 * a + x) * dim + 3 * b + y];
                }
            }
            o
        };
        let mut row_k = vec![[[0.0; 3]; 3]; m]; // -sum_a H[a][b]
        let mut kk = [[0.0; 3]; 3];
        for a in 0..m {
            let ja = s.neighbors[a];
            let mut col_k = [[0.0; 3]; 3];
            for b in 0..m {
                let hb = blk(a, b);
                hadd(&mut hess, ja, s.neighbors[b], &hb, w);
                for x in 0..3 {
                    for y in 0..3 {
                        col_k[x][y] -= hb[x][y];
                        row_k[b][x][y] -= hb[x][y];
                    }
                }
            }
            hadd(&mut hess, ja, k, &col_k, w);
            for x in 0..3 {
                for y in 0..3 {
                    kk[x][y] -= col_k[x][y];
                }
            }
        }
        for b in 0..m {
            hadd(&mut hess, k, s.neighbors[b], &row_k[b], w);
        }
        hadd(&mut hess, k, k, &kk, w);
        // A_k d2sigma_k and the cross terms
        for &(i, j, ref mm) in &sk.h {
            hadd(&mut hess, i, j, mm, 0.001 * ak);
        }
        for &(i, gs) in &sk.g {
            for &(j, gaj) in &ga {
                hadd(&mut hess, i, j, &outer(gs, gaj), 0.001);
                hadd(&mut hess, j, i, &outer(gaj, gs), 0.001);
            }
        }
    }
    let gfac = to_ang / kcal;
    let hfac = to_ang * to_ang / kcal;
    CdsResult {
        energy: energy / kcal,
        per_atom: per_atom.into_iter().map(|e| e / kcal).collect(),
        sasa: sasa.iter().map(|s| s.area).collect(),
        atomic_surface_tension: sigma,
        molecular_surface_tension: t.molecular,
        gradient: grad_on.then(|| {
            grad.iter()
                .map(|g| [g[0] * gfac, g[1] * gfac, g[2] * gfac])
                .collect()
        }),
        hessian: hess_on.then(|| hess.iter().map(|h| h * hfac).collect()),
    }
}

/// [`cds`] with the solvent looked up by name in [`crate::db`].
pub fn cds_named(
    z: &[u32],
    coords_bohr: &[[f64; 3]],
    solvent: &str,
    opts: &CdsConfig,
) -> Result<CdsResult, SmdError> {
    let s = db::lookup(solvent).ok_or_else(|| SmdError::UnknownSolvent(solvent.to_string()))?;
    let sd = s
        .smd
        .as_ref()
        .ok_or_else(|| SmdError::NoDescriptors(solvent.to_string()))?;
    Ok(cds(z, coords_bohr, sd, opts))
}

#[cfg(test)]
mod tests {
    use super::*;

    const PYSCF_BOHR: f64 = BOHR_ANGSTROM_PYSCF;

    fn bohr(atoms: &[(u32, [f64; 3])]) -> (Vec<u32>, Vec<[f64; 3]>) {
        (
            atoms.iter().map(|a| a.0).collect(),
            atoms.iter().map(|a| a.1.map(|x| x / PYSCF_BOHR)).collect(),
        )
    }

    fn water() -> (Vec<u32>, Vec<[f64; 3]>) {
        bohr(&[
            (8, [0.0, 0.0, 0.117790]),
            (1, [0.0, 0.755453, -0.471161]),
            (1, [0.0, -0.755453, -0.471161]),
        ])
    }

    /// acetonitrile next to formamide and methanol: every SMD pair term is active
    /// (H-C, H-O, C-C, C-N^2, N-C^1.3, N-C(3), O-C, O-N, O-O)
    fn cluster() -> (Vec<u32>, Vec<[f64; 3]>) {
        bohr(&[
            (6, [0.000, 0.000, 0.000]),
            (6, [1.460, 0.010, 0.000]),
            (7, [2.620, 0.020, 0.010]),
            (1, [-0.370, 1.030, 0.000]),
            (1, [-0.370, -0.515, 0.890]),
            (1, [-0.380, -0.510, -0.890]),
            // formamide H2N-CHO
            (6, [4.300, 1.200, 0.300]),
            (8, [4.100, 2.400, 0.250]),
            (7, [5.450, 0.600, 0.400]),
            (1, [3.450, 0.500, 0.250]),
            (1, [5.550, -0.400, 0.450]),
            (1, [6.300, 1.150, 0.420]),
            // methanol
            (8, [3.000, 3.600, 0.800]),
            (6, [2.200, 4.300, 1.700]),
            (1, [3.500, 3.000, 1.300]),
            (1, [1.500, 4.900, 1.100]),
            (1, [2.800, 4.950, 2.300]),
            (1, [1.650, 3.600, 2.300]),
        ])
    }

    fn sd(n: f64, alpha: f64, beta: f64, gamma: f64, phi: f64, psi: f64) -> SmdDescriptors {
        SmdDescriptors {
            n,
            alpha,
            beta,
            gamma,
            phi,
            psi,
            water: false,
        }
    }

    fn water_sd() -> SmdDescriptors {
        *db::smd_descriptors("water").unwrap()
    }

    #[test]
    fn water_in_water_matches_pyscf() {
        // PySCF 2.15 smd.get_cds_legacy (libsolvent / mnsol.F) for tests/molecules/water.xyz
        let (z, x) = water();
        let r = cds(
            &z,
            &x,
            &water_sd(),
            &CdsConfig {
                order: CdsOrder::Gradient,
                ..Default::default()
            },
        );
        assert!(
            (r.energy - 0.002296887841443703).abs() < 1e-15,
            "{}",
            r.energy
        );
        let g = r.gradient.unwrap();
        assert!((g[0][2] - 0.0002902465357855612).abs() < 1e-15);
        assert!((g[1][1] - 0.0004511967579780164).abs() < 1e-15);
        assert!((r.per_atom.iter().sum::<f64>() - r.energy).abs() < 1e-17);
    }

    #[test]
    fn coulomb_radii_rules() {
        let z = [1, 6, 7, 8, 9, 14, 15, 16, 17, 35, 53, 19, 26, 5, 120];
        let a = coulomb_radii_angstrom(&z, 0.0, RadiiScheme::Smd);
        assert_eq!(
            &a[..10],
            &[
                1.20,
                1.85,
                1.89,
                1.52 + 1.8 * 0.43,
                1.73,
                2.47,
                2.12,
                2.49,
                2.38,
                3.06
            ]
        );
        assert_eq!(&a[10..], &[1.98, 2.75, 1.94, 1.92, 2.0]);
        assert_eq!(
            coulomb_radii_angstrom(&[8], 0.82, RadiiScheme::Smd)[0],
            1.52
        );
        let b = coulomb_radii_angstrom(&z, 0.0, RadiiScheme::Smd18);
        assert_eq!((b[9], b[10]), (2.60, 2.74));
        let p = coulomb_radii_angstrom(&z, 0.0, RadiiScheme::Pyscf);
        assert_eq!(
            (p[9], p[10], p[12], p[14]),
            (2.60, 2.74, 1.999999, 1.999999)
        );
        assert_eq!(
            coulomb_radii(&[6], 0.0, RadiiScheme::Pyscf)[0],
            1.85 / BOHR_ANGSTROM_PYSCF
        );
    }

    fn fd_check(z: &[u32], x: &[[f64; 3]], s: &SmdDescriptors, smooth: f64) -> (f64, f64, f64) {
        let opts = |order| CdsConfig {
            order,
            convention: ParamConvention::Mnsol,
            contact_smoothing: smooth,
        };
        let r = cds(z, x, s, &opts(CdsOrder::Hessian));
        let g = r.gradient.unwrap();
        let h = r.hessian.unwrap();
        let n3 = 3 * z.len();
        // fourth-order central differences (errors scale as h^4: 2e-11 for the Hessian here)
        let step = 2.5e-4;
        let (mut eg, mut eh, mut asym) = (0.0f64, 0.0f64, 0.0f64);
        for q in 0..n3 {
            let at = |f: f64| {
                let mut xs = x.to_vec();
                xs[q / 3][q % 3] += f * step;
                cds(z, &xs, s, &opts(CdsOrder::Gradient))
            };
            let (p1, m1, p2, m2) = (at(1.0), at(-1.0), at(2.0), at(-2.0));
            let d4 = |a: f64, b: f64, c: f64, d: f64| (8.0 * (a - b) - (c - d)) / (12.0 * step);
            eg = eg.max((d4(p1.energy, m1.energy, p2.energy, m2.energy) - g[q / 3][q % 3]).abs());
            let (gp1, gm1, gp2, gm2) = (
                p1.gradient.unwrap(),
                m1.gradient.unwrap(),
                p2.gradient.unwrap(),
                m2.gradient.unwrap(),
            );
            for p in 0..n3 {
                let (a, b) = (p / 3, p % 3);
                let fd = d4(gp1[a][b], gm1[a][b], gp2[a][b], gm2[a][b]);
                eh = eh.max((fd - h[q * n3 + p]).abs());
                asym = asym.max((h[q * n3 + p] - h[p * n3 + q]).abs());
            }
        }
        (eg, eh, asym)
    }

    #[test]
    fn gradient_and_hessian_match_finite_differences() {
        let (z, x) = cluster();
        for s in [
            water_sd(),
            sd(1.4050, 0.0, 0.48, 39.44, 0.0, 0.0),
            sd(1.4459, 0.15, 0.02, 38.39, 0.0, 0.75),
            sd(1.3288, 0.43, 0.47, 31.77, 0.0, 0.0),
        ] {
            let (eg, eh, asym) = fd_check(&z, &x, &s, 0.0);
            eprintln!("FD check: gradient {eg:.1e} Hessian {eh:.1e} asym {asym:.1e}");
            assert!(eg < 1e-11, "gradient vs FD {eg:e}");
            assert!(eh < 1e-9, "Hessian vs FD {eh:e}");
            assert!(asym < 1e-12, "Hessian asymmetry {asym:e}");
        }
        let (eg, eh, _) = fd_check(&z, &x, &water_sd(), 0.1);
        assert!(eg < 1e-11 && eh < 1e-9, "smoothed: {eg:e} {eh:e}");
    }

    #[test]
    fn bitwise_identical_across_thread_counts() {
        let (z, x) = cluster();
        let s = sd(1.4050, 0.0, 0.48, 39.44, 0.0, 0.0);
        let run = |t: usize| {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(t)
                .build()
                .unwrap();
            pool.install(|| {
                cds(
                    &z,
                    &x,
                    &s,
                    &CdsConfig {
                        order: CdsOrder::Hessian,
                        ..Default::default()
                    },
                )
            })
        };
        let r1 = run(1);
        for t in [2, 3, 8] {
            let r = run(t);
            assert_eq!(r.energy.to_bits(), r1.energy.to_bits());
            assert_eq!(r.gradient, r1.gradient);
            assert_eq!(r.hessian, r1.hessian);
        }
    }

    #[test]
    fn contact_jump_and_smoothing() {
        // two H atoms approaching: R = 1.2 + 0.4 Å each, contact at 3.2 Å
        let s = water_sd();
        let at = |d: f64, smooth: f64| {
            let x = [[0.0, 0.0, 0.0], [d / PYSCF_BOHR, 0.0, 0.0]];
            let opts = CdsConfig {
                order: CdsOrder::Gradient,
                contact_smoothing: smooth,
                ..Default::default()
            };
            let r = cds(&[1, 1], &x, &s, &opts);
            (r.energy, r.gradient.unwrap()[1][0])
        };
        let touch = 2.0 * (1.2f32 as f64 + 0.4) * PYSCF_BOHR / MNSOL_BOHR_ANGSTROM;
        let eps = 1e-7;
        let (e_out, g_out) = at(touch + eps, 0.0);
        let (e_in, g_in) = at(touch - eps, 0.0);
        assert!((e_out - e_in).abs() < 1e-9, "energy is continuous");
        assert_eq!(g_out, 0.0);
        assert!(
            g_in.abs() > 1e-4,
            "exact SASA: gradient jumps at contact ({g_in:e} Eh/bohr)"
        );
        let (_, g_out) = at(touch + eps, 0.1);
        let (_, g_in) = at(touch - eps, 0.1);
        assert!(
            (g_in - g_out).abs() < 1e-12,
            "smoothed: gradient continuous ({g_in:e})"
        );
        assert_eq!(
            at(3.05, 0.1).0,
            at(3.05, 0.0).0,
            "outside the window nothing changes"
        );
    }

    #[test]
    fn named_lookup_and_conventions() {
        let (z, x) = water();
        let a = cds_named(&z, &x, "THF", &CdsConfig::default()).unwrap();
        let b = cds(
            &z,
            &x,
            db::smd_descriptors("tetrahydrofuran").unwrap(),
            &CdsConfig::default(),
        );
        assert_eq!(a.energy, b.energy);
        let opts = CdsConfig {
            convention: ParamConvention::Decimal,
            ..Default::default()
        };
        let d = cds(&z, &x, db::smd_descriptors("thf").unwrap(), &opts);
        assert!((d.energy - a.energy).abs() < 1e-9 && d.energy != a.energy);
        assert!(cds_named(&z, &x, "no such solvent", &CdsConfig::default()).is_err());
    }
}
