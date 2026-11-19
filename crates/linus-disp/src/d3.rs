// SPDX-License-Identifier: LGPL-3.0-or-later
//! DFT-D3 with rational (Becke-Johnson) damping and the optional Axilrod-Teller-Muto
//! three-body term, for molecules: a port of s-dftd3 1.7.0 (`dftd3_model`,
//! `dftd3_ncoord`, `dftd3_damping_rational`, `dftd3_damping_atm`), with analytic
//! gradients and Hessians.
//!
//! Model (Grimme et al., JCP 132, 154104 (2010); JCC 32, 1456 (2011)):
//!
//! * coordination numbers `CN_i = sum_j 1 / (1 + exp(-16 (R_ij^cov / r_ij - 1)))`, with
//!   `R^cov = 4/3 (Pyykko radii)` and a hard 40 bohr cutoff;
//! * `C6_ij = sum_ab w_ia w_jb C6ref_ab`, Gaussian weights `exp(-4 (CN - CNref)^2)`
//!   normalized per atom (s-dftd3 shifts the exponent by its minimum, which is exact);
//! * `E2 = -sum_{i<j} C6_ij (s6 / (r^6 + R0^6) + s8 rr_ij / (r^8 + R0^8))` with
//!   `rr_ij = 3 sqrt(Q_i) sqrt(Q_j)` (`C8 = rr C6`) and `R0 = a1 sqrt(rr_ij) + a2`; hard
//!   60 bohr cutoff;
//! * `E3 = s9 sum_{i<j<k} sqrt(C6_ij C6_ik C6_jk) (3 cos a cos b cos c + 1) / (r_ij r_ik r_jk)^3
//!   / (1 + 6 (r0_ijk / rbar)^(16/3))` with zero damping on `4/3 r0^vdw` (s-dftd3's ATM for
//!   rational damping; alp = 14 + 2); hard 40 bohr cutoff.
//!
//! Hessians: `E(R, CN(R))` is differentiated as `E_RR + E_RC J + J^T E_CR + J^T E_CC J +
//! sum_k E_C,k d2CN_k`, `J = dCN/dR`. Every row block is computed by the task that owns its
//! atom, so the result does not depend on the number of threads.

use crate::data::d3 as tab;
use crate::par;
use crate::units::{covalent_radius_d3, AATOAU};
use crate::{DispError, DispResult, Order};
use rayon::prelude::*;

/// Maximum number of reference systems per element.
pub const MAX_REF: usize = tab::MAX_REF;
/// Steepness of the D3 counting function.
const KCN: f64 = 16.0;
/// Scaling of the van-der-Waals radii in the ATM zero damping.
const RS9: f64 = 4.0 / 3.0;
/// Gaussian weighting factor of the CN interpolation (s-dftd3 `wf_default`).
pub const WF: f64 = 4.0;

/// Rational-damping D3 parameters with the ATM three-body scaling.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct D3Params {
    pub s6: f64,
    pub s8: f64,
    pub a1: f64,
    pub a2: f64,
    /// ATM three-body scaling: 0 = two-body only (ORCA 6 and the s-dftd3 CLI default),
    /// 1 = ORCA `D3BJ ABC` / s-dftd3 `--atm`.
    pub s9: f64,
    /// s-dftd3's `alp` (14); the ATM zero damping uses the exponent `alp + 2`.
    pub alp: f64,
}

impl D3Params {
    /// Two-body parameters (s9 = 0, alp = 14).
    pub fn new(s6: f64, s8: f64, a1: f64, a2: f64) -> Self {
        Self {
            s6,
            s8,
            a1,
            a2,
            s9: 0.0,
            alp: 14.0,
        }
    }
    /// Same parameters with the ATM three-body term scaled by `s9`.
    pub fn with_atm(mut self, s9: f64) -> Self {
        self.s9 = s9;
        self
    }
}

/// Real-space cutoffs (bohr). All cutoffs are hard, as in s-dftd3.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct D3Cutoffs {
    /// coordination number
    pub cn: f64,
    /// two-body term
    pub disp2: f64,
    /// three-body term
    pub disp3: f64,
}

impl D3Cutoffs {
    /// s-dftd3 1.7.0 defaults (`realspace_cutoff`): CN 40, two-body 60, three-body 40.
    pub const S_DFTD3: Self = Self {
        cn: 40.0,
        disp2: 60.0,
        disp3: 40.0,
    };
}

impl Default for D3Cutoffs {
    fn default() -> Self {
        Self::S_DFTD3
    }
}

/// Options of [`d3bj`].
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct D3Options {
    pub order: Order,
    pub cutoffs: D3Cutoffs,
}

/// D3 model of one molecule (fixed atomic numbers).
#[derive(Clone, Debug)]
pub struct D3Model {
    numbers: Vec<u32>,
    /// species index of every atom
    id: Vec<usize>,
    nsp: usize,
    rcov: Vec<f64>,
    r4r2: Vec<f64>,
    nref: Vec<usize>,
    refcn: Vec<[f64; MAX_REF]>,
    /// `[isp * nsp + jsp][iref][jref]`
    c6ref: Vec<[[f64; MAX_REF]; MAX_REF]>,
    /// pair van-der-Waals radii (bohr), `[isp * nsp + jsp]`
    rvdw: Vec<f64>,
    wf: f64,
}

/// Reference weights of all atoms and their first and second CN derivatives.
pub(crate) struct Weights {
    gw: Vec<[f64; MAX_REF]>,
    d1: Vec<[f64; MAX_REF]>,
    d2: Vec<[f64; MAX_REF]>,
}

/// Pair C6 with its CN derivatives as dense row-major n x n matrices.
#[derive(Clone, Debug, Default)]
pub struct C6Matrix {
    pub n: usize,
    /// `c6[i n + j]`
    pub c6: Vec<f64>,
    /// `d1[i n + j] = dC6_ij / dCN_i`
    pub d1: Vec<f64>,
    /// `d2[i n + j] = d2C6_ij / dCN_i^2` (Hessian only)
    pub d2: Vec<f64>,
    /// `dx[i n + j] = d2C6_ij / dCN_i dCN_j` (Hessian only, symmetric)
    pub dx: Vec<f64>,
}

/// s-dftd3 reference C6 `get_c6(iref, jref, zi, zj)` (heavier element's reference first).
fn reference_c6(iref: usize, jref: usize, zi: usize, zj: usize) -> f64 {
    let nr = |z: usize| tab::NUMBER_OF_REFERENCES[z - 1] as usize;
    if zi > zj {
        let ic = zj + zi * (zi - 1) / 2 - 1;
        tab::REFERENCE_C6[tab::REFERENCE_C6_OFFSET[ic] as usize + jref * nr(zi) + iref]
    } else {
        let ic = zi + zj * (zj - 1) / 2 - 1;
        tab::REFERENCE_C6[tab::REFERENCE_C6_OFFSET[ic] as usize + iref * nr(zj) + jref]
    }
}

/// `sqrt(0.5 * <r4>/<r2> * sqrt(Z))`, s-dftd3 `get_r4r2_val`.
pub fn r4r2(z: u32) -> f64 {
    let z = z as usize;
    if z == 0 || z > tab::R4_OVER_R2.len() {
        return 0.0;
    }
    (0.5 * (tab::R4_OVER_R2[z - 1] * (z as f64).sqrt())).sqrt()
}

/// Pair van-der-Waals radius in bohr (s-dftd3 `get_vdw_rad(zi, zj)`, elements 1-103).
pub fn vdw_rad(zi: u32, zj: u32) -> f64 {
    let (a, b) = if zi > zj {
        (zi as usize, zj as usize)
    } else {
        (zj as usize, zi as usize)
    };
    AATOAU * tab::VDW_RAD_AA[b + a * (a - 1) / 2 - 1]
}

#[inline]
fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

#[inline]
fn norm2(v: [f64; 3]) -> f64 {
    v[0] * v[0] + v[1] * v[1] + v[2] * v[2]
}

/// D3 counting function and its first and second derivatives with respect to r.
#[inline]
fn count3(rc: f64, r: f64) -> (f64, f64, f64) {
    let e = (-KCN * (rc / r - 1.0)).exp();
    let cf = 1.0 / (1.0 + e);
    let r2 = r * r;
    let tmp = cf * (1.0 - cf);
    let d1 = -tmp * KCN * rc / r2;
    let d2 = tmp * (1.0 - 2.0 * cf) * (KCN * rc) * (KCN * rc) / (r2 * r2)
        + 2.0 * tmp * KCN * rc / (r2 * r);
    (cf, d1, d2)
}

/// 3x3 block of the Hessian of a radial pair function with `dE/dr = e1`, `d2E/dr2 = e2`
/// (`u` = unit vector of the pair): `e2 u u^T + e1 / r (I - u u^T)`.
#[inline]
fn radial_block(u: [f64; 3], r: f64, e1: f64, e2: f64) -> [[f64; 3]; 3] {
    let mut b = [[0.0; 3]; 3];
    let er = e1 / r;
    for x in 0..3 {
        for y in 0..3 {
            b[x][y] = (e2 - er) * u[x] * u[y];
        }
        b[x][x] += er;
    }
    b
}

/// Add `blk` to the diagonal block of atom `a` and subtract it from block (a, j), in the
/// three rows `rows` (= rows 3a..3a+3 of an n3-wide matrix).
#[inline]
fn add_pair_rows(rows: &mut [f64], n3: usize, a: usize, j: usize, blk: &[[f64; 3]; 3]) {
    for x in 0..3 {
        for y in 0..3 {
            rows[x * n3 + 3 * a + y] += blk[x][y];
            rows[x * n3 + 3 * j + y] -= blk[x][y];
        }
    }
}

impl D3Model {
    /// D3 model for the given atomic numbers (H-Lr, 1-103).
    pub fn new(numbers: &[u32]) -> Result<Self, DispError> {
        let mut species: Vec<u32> = Vec::new();
        let mut id = Vec::with_capacity(numbers.len());
        for (iat, &z) in numbers.iter().enumerate() {
            if z == 0 || z as usize > tab::MAX_ELEM {
                return Err(DispError::UnsupportedElement {
                    model: "D3",
                    atom: iat,
                    number: z,
                });
            }
            let isp = match species.iter().position(|&s| s == z) {
                Some(p) => p,
                None => {
                    species.push(z);
                    species.len() - 1
                }
            };
            id.push(isp);
        }
        let nsp = species.len();
        let nref: Vec<usize> = species
            .iter()
            .map(|&z| tab::NUMBER_OF_REFERENCES[z as usize - 1] as usize)
            .collect();
        let refcn = species
            .iter()
            .map(|&z| tab::REFERENCE_CN[z as usize - 1])
            .collect();
        let mut c6ref = vec![[[0.0; MAX_REF]; MAX_REF]; nsp * nsp];
        let mut rvdw = vec![0.0; nsp * nsp];
        for isp in 0..nsp {
            for jsp in 0..nsp {
                let (zi, zj) = (species[isp] as usize, species[jsp] as usize);
                for iref in 0..nref[isp] {
                    for jref in 0..nref[jsp] {
                        c6ref[isp * nsp + jsp][iref][jref] = reference_c6(iref, jref, zi, zj);
                    }
                }
                rvdw[isp * nsp + jsp] = vdw_rad(species[isp], species[jsp]);
            }
        }
        Ok(Self {
            numbers: numbers.to_vec(),
            id,
            nsp,
            rcov: species.iter().map(|&z| covalent_radius_d3(z)).collect(),
            r4r2: species.iter().map(|&z| r4r2(z)).collect(),
            nref,
            refcn,
            c6ref,
            rvdw,
            wf: WF,
        })
    }

    pub fn natoms(&self) -> usize {
        self.numbers.len()
    }

    pub fn numbers(&self) -> &[u32] {
        &self.numbers
    }

    #[inline]
    fn sp(&self, i: usize, j: usize) -> usize {
        self.id[i] * self.nsp + self.id[j]
    }

    /// Covalent radius sum of a pair of atoms.
    #[inline]
    fn rc(&self, i: usize, j: usize) -> f64 {
        self.rcov[self.id[i]] + self.rcov[self.id[j]]
    }

    /// D3 coordination numbers (each atom sums its neighbours in index order).
    pub fn coordination_numbers(&self, xyz: &[[f64; 3]], cutoff: f64) -> Vec<f64> {
        let n = self.natoms();
        let cutoff2 = cutoff * cutoff;
        (0..n)
            .into_par_iter()
            .map(|i| {
                let mut cn = 0.0;
                for j in 0..n {
                    let r2 = norm2(sub(xyz[i], xyz[j]));
                    if r2 > cutoff2 || r2 < 1.0e-12 {
                        continue;
                    }
                    let r = r2.sqrt();
                    cn += 1.0 / (1.0 + (-KCN * (self.rc(i, j) / r - 1.0)).exp());
                }
                cn
            })
            .collect()
    }

    /// Jacobian `J[k][3a + x] = dCN_k / dR_ax` (row-major n x 3n).
    fn cn_jacobian(&self, xyz: &[[f64; 3]], cutoff: f64) -> Vec<f64> {
        let n = self.natoms();
        let n3 = 3 * n;
        let cutoff2 = cutoff * cutoff;
        let mut jac = vec![0.0; n * n3];
        jac.par_chunks_mut(n3.max(1))
            .enumerate()
            .for_each(|(k, row)| {
                for l in 0..n {
                    let v = sub(xyz[k], xyz[l]);
                    let r2 = norm2(v);
                    if r2 > cutoff2 || r2 < 1.0e-12 {
                        continue;
                    }
                    let r = r2.sqrt();
                    let (_, d1, _) = count3(self.rc(k, l), r);
                    for x in 0..3 {
                        let g = d1 * v[x] / r;
                        row[3 * k + x] += g;
                        row[3 * l + x] -= g;
                    }
                }
            });
        jac
    }

    /// Reference weights (s-dftd3 1.7.0 `weight_references`, log-sum-exp shifted) and
    /// their CN derivatives up to `order`.
    pub(crate) fn weights(&self, cn: &[f64], order: usize) -> Weights {
        let n = self.natoms();
        let z = [0.0; MAX_REF];
        let mut w = Weights {
            gw: vec![z; n],
            d1: if order >= 1 { vec![z; n] } else { Vec::new() },
            d2: if order >= 2 { vec![z; n] } else { Vec::new() },
        };
        for iat in 0..n {
            let isp = self.id[iat];
            let nref = self.nref[isp];
            let mut dcn = [0.0; MAX_REF];
            let mut gw = [0.0; MAX_REF];
            let mut dmin = f64::INFINITY;
            for r in 0..nref {
                dcn[r] = self.refcn[isp][r] - cn[iat];
                dmin = dmin.min(dcn[r] * dcn[r]);
            }
            let mut norm = 0.0;
            for r in 0..nref {
                gw[r] = (-self.wf * (dcn[r] * dcn[r] - dmin)).exp();
                norm += gw[r];
            }
            for r in 0..nref {
                gw[r] /= norm;
            }
            w.gw[iat] = gw;
            if order >= 1 {
                let mut slope = [0.0; MAX_REF];
                let mut mean = 0.0;
                for r in 0..nref {
                    slope[r] = 2.0 * self.wf * dcn[r];
                    mean += gw[r] * slope[r];
                }
                for r in 0..nref {
                    slope[r] -= mean;
                    w.d1[iat][r] = gw[r] * slope[r];
                }
                if order >= 2 {
                    let mut var = 0.0;
                    for r in 0..nref {
                        var += gw[r] * (slope[r] * slope[r]);
                    }
                    for r in 0..nref {
                        w.d2[iat][r] = gw[r] * (slope[r] * slope[r] - var);
                    }
                }
            }
        }
        w
    }

    /// Pair C6 coefficients (and CN derivatives up to `order`) of all atom pairs.
    pub(crate) fn c6_matrix(&self, w: &Weights, order: usize) -> C6Matrix {
        let n = self.natoms();
        let mut m = C6Matrix {
            n,
            c6: vec![0.0; n * n],
            d1: if order >= 1 {
                vec![0.0; n * n]
            } else {
                Vec::new()
            },
            d2: if order >= 2 {
                vec![0.0; n * n]
            } else {
                Vec::new()
            },
            dx: if order >= 2 {
                vec![0.0; n * n]
            } else {
                Vec::new()
            },
        };
        // every row i computes (i, j) for all j with the loops of s-dftd3's pair (iat = max)
        let rows: Vec<[Vec<f64>; 4]> = (0..n)
            .into_par_iter()
            .map(|i| {
                let mut c6 = vec![0.0; n];
                let mut d1 = vec![0.0; if order >= 1 { n } else { 0 }];
                let mut d2 = vec![0.0; if order >= 2 { n } else { 0 }];
                let mut dx = vec![0.0; if order >= 2 { n } else { 0 }];
                for j in 0..n {
                    // s-dftd3 evaluates the pair with iat >= jat; keep its loop order
                    let (a, b) = if i >= j { (i, j) } else { (j, i) };
                    let blk = &self.c6ref[self.sp(a, b)];
                    let (na, nb) = (self.nref[self.id[a]], self.nref[self.id[b]]);
                    let (wa, wb) = (&w.gw[a], &w.gw[b]);
                    let mut s = 0.0;
                    let (mut sa, mut sb, mut saa, mut sbb, mut sab) = (0.0, 0.0, 0.0, 0.0, 0.0);
                    for ra in 0..na {
                        for rb in 0..nb {
                            let r = blk[ra][rb];
                            s += wa[ra] * wb[rb] * r;
                            if order >= 1 {
                                sa += w.d1[a][ra] * wb[rb] * r;
                                sb += wa[ra] * w.d1[b][rb] * r;
                            }
                            if order >= 2 {
                                saa += w.d2[a][ra] * wb[rb] * r;
                                sbb += wa[ra] * w.d2[b][rb] * r;
                                sab += w.d1[a][ra] * w.d1[b][rb] * r;
                            }
                        }
                    }
                    c6[j] = s;
                    if order >= 1 {
                        // derivative with respect to the CN of atom i
                        d1[j] = if i >= j { sa } else { sb };
                    }
                    if order >= 2 {
                        d2[j] = if i >= j { saa } else { sbb };
                        dx[j] = sab;
                    }
                    // the (i, i) entries are never used: molecules have no self pairs
                }
                [c6, d1, d2, dx]
            })
            .collect();
        for (i, [c6, d1, d2, dx]) in rows.into_iter().enumerate() {
            m.c6[i * n..(i + 1) * n].copy_from_slice(&c6);
            if order >= 1 {
                m.d1[i * n..(i + 1) * n].copy_from_slice(&d1);
            }
            if order >= 2 {
                m.d2[i * n..(i + 1) * n].copy_from_slice(&d2);
                m.dx[i * n..(i + 1) * n].copy_from_slice(&dx);
            }
        }
        m
    }

    /// Coordination numbers and the C6 matrix (no derivatives) at a geometry.
    pub fn c6(&self, xyz: &[[f64; 3]], cutoffs: &D3Cutoffs) -> (Vec<f64>, C6Matrix) {
        let cn = self.coordination_numbers(xyz, cutoffs.cn);
        let w = self.weights(&cn, 0);
        let m = self.c6_matrix(&w, 0);
        (cn, m)
    }

    /// `rr_ij = 3 r4r2_i r4r2_j` and `R0 = a1 sqrt(rr) + a2` of a pair.
    #[inline]
    fn rr_r0(&self, i: usize, j: usize, p: &D3Params) -> (f64, f64) {
        let rr = 3.0 * self.r4r2[self.id[i]] * self.r4r2[self.id[j]];
        (rr, p.a1 * rr.sqrt() + p.a2)
    }

    /// Energy, gradient and `dE/dCN` (two-body and ATM, before the CN chain rule).
    fn energy_gradient(
        &self,
        xyz: &[[f64; 3]],
        p: &D3Params,
        cut: &D3Cutoffs,
        c6: &C6Matrix,
        grad: bool,
    ) -> (f64, Vec<[f64; 3]>, Vec<f64>) {
        let n = self.natoms();
        // two-body: every atom owns its row of pairs (each pair is evaluated twice)
        let cutoff2 = cut.disp2 * cut.disp2;
        let two = p.s6.abs() >= f64::EPSILON || p.s8.abs() >= f64::EPSILON;
        let rows: Vec<(f64, [f64; 3], f64)> = (0..n)
            .into_par_iter()
            .map(|i| {
                let (mut e, mut g, mut f) = (0.0, [0.0; 3], 0.0);
                if !two {
                    return (e, g, f);
                }
                for j in 0..n {
                    let (a, b) = if i >= j { (i, j) } else { (j, i) };
                    let v = sub(xyz[a], xyz[b]);
                    let r2 = norm2(v);
                    if r2 > cutoff2 || r2 < f64::EPSILON {
                        continue;
                    }
                    let (rrij, r0ij) = self.rr_r0(a, b, p);
                    let c6ij = c6.c6[a * n + b];
                    let r0ij2 = r0ij * r0ij;
                    let r0ij6 = r0ij2 * r0ij2 * r0ij2;
                    let r0ij8 = r0ij6 * r0ij2;
                    let r4 = r2 * r2;
                    let t6 = 1.0 / (r4 * r2 + r0ij6);
                    let t8 = 1.0 / (r4 * r4 + r0ij8);
                    let edisp = p.s6 * t6 + p.s8 * rrij * t8;
                    e += -c6ij * edisp * 0.5;
                    if grad {
                        let d6 = -6.0 * r4 * t6 * t6;
                        let d8 = -8.0 * r4 * r2 * t8 * t8;
                        let gdisp = p.s6 * d6 + p.s8 * rrij * d8;
                        // dG of s-dftd3 is the gradient on atom a (= the larger index)
                        let sgn = if i == a { 1.0 } else { -1.0 };
                        for x in 0..3 {
                            g[x] += sgn * (-c6ij * gdisp * v[x]);
                        }
                        f -= c6.d1[i * n + j] * edisp;
                    }
                }
                (e, g, f)
            })
            .collect();
        let mut energy = 0.0;
        let mut gradient = vec![[0.0; 3]; if grad { n } else { 0 }];
        let mut dedcn = vec![0.0; if grad { n } else { 0 }];
        for (i, (e, g, f)) in rows.into_iter().enumerate() {
            energy += e;
            if grad {
                gradient[i] = g;
                dedcn[i] = f;
            }
        }
        if p.s9.abs() >= f64::EPSILON {
            let ranges = par::balanced_ranges(n, |i| (i * i) as f64);
            par::reduce_ranges(
                &ranges,
                |lo, hi| self.atm_range(lo, hi, xyz, p, cut, c6, grad),
                |(e, g, f)| {
                    energy += e;
                    if grad {
                        for i in 0..n {
                            for x in 0..3 {
                                gradient[i][x] += g[i][x];
                            }
                            dedcn[i] += f[i];
                        }
                    }
                },
            );
        }
        (energy, gradient, dedcn)
    }

    /// ATM energy (and gradient / dE/dCN) of the triples i > j > k with i in lo..hi
    /// (s-dftd3 `get_atm_dispersion_derivs`, hard cutoff).
    #[allow(clippy::too_many_arguments)]
    fn atm_range(
        &self,
        lo: usize,
        hi: usize,
        xyz: &[[f64; 3]],
        p: &D3Params,
        cut: &D3Cutoffs,
        c6: &C6Matrix,
        grad: bool,
    ) -> (f64, Vec<[f64; 3]>, Vec<f64>) {
        let n = self.natoms();
        let cutoff2 = cut.disp3 * cut.disp3;
        let alp = p.alp + 2.0;
        let alp3 = alp / 3.0;
        let mut energy = 0.0;
        let mut g = vec![[0.0; 3]; if grad { n } else { 0 }];
        let mut f = vec![0.0; if grad { n } else { 0 }];
        for iat in lo..hi {
            for jat in 0..iat {
                let c6ij = c6.c6[iat * n + jat];
                let r0ij = RS9 * self.rvdw[self.sp(jat, iat)];
                let vij = sub(xyz[jat], xyz[iat]);
                let r2ij = norm2(vij);
                if r2ij > cutoff2 || r2ij < f64::EPSILON {
                    continue;
                }
                for kat in 0..jat {
                    let c6ik = c6.c6[iat * n + kat];
                    let c6jk = c6.c6[jat * n + kat];
                    if grad
                        && (c6ij.abs() < f64::EPSILON
                            || c6ik.abs() < f64::EPSILON
                            || c6jk.abs() < f64::EPSILON)
                    {
                        continue;
                    }
                    let c9 = -p.s9 * (c6ij * c6ik * c6jk).abs().sqrt();
                    let r0ik = RS9 * self.rvdw[self.sp(kat, iat)];
                    let r0jk = RS9 * self.rvdw[self.sp(kat, jat)];
                    let r0 = r0ij * r0ik * r0jk;
                    let vik = sub(xyz[kat], xyz[iat]);
                    let r2ik = norm2(vik);
                    if r2ik > cutoff2 || r2ik < f64::EPSILON {
                        continue;
                    }
                    let vjk = sub(vik, vij);
                    let r2jk = norm2(vjk);
                    if r2jk > cutoff2 || r2jk < f64::EPSILON {
                        continue;
                    }
                    let r2 = r2ij * r2ik * r2jk;
                    let r1 = r2.sqrt();
                    let r3 = r2 * r1;
                    let r5 = r3 * r2;
                    let fdmp = 1.0 / (1.0 + 6.0 * (r0 / r1).powf(alp3));
                    let ang =
                        0.375 * (r2ij + r2jk - r2ik) * (r2ij - r2jk + r2ik) * (-r2ij + r2jk + r2ik)
                            / r5
                            + 1.0 / r3;
                    let rr = ang * fdmp;
                    let de0 = rr * c9;
                    energy -= de0;
                    if !grad {
                        continue;
                    }
                    let r0r1alp3 = (r0 / r1).powf(alp3);
                    let dfdmp = -2.0 * alp * r0r1alp3 * fdmp * fdmp;
                    let dang = |a: f64, b: f64, c: f64| {
                        // d/d(r_a) term of s-dftd3 with a the pair, b and c the others
                        -0.375
                            * (a * a * a
                                + a * a * (b + c)
                                + a * (3.0 * b * b + 2.0 * b * c + 3.0 * c * c)
                                - 5.0 * (b - c) * (b - c) * (b + c))
                            / r5
                    };
                    let dij = dang(r2ij, r2jk, r2ik);
                    let dik = dang(r2ik, r2jk, r2ij);
                    let djk = dang(r2jk, r2ik, r2ij);
                    let mut gij = [0.0; 3];
                    let mut gik = [0.0; 3];
                    let mut gjk = [0.0; 3];
                    for x in 0..3 {
                        gij[x] = c9 * (-dij * fdmp + ang * dfdmp) / r2ij * vij[x];
                        gik[x] = c9 * (-dik * fdmp + ang * dfdmp) / r2ik * vik[x];
                        gjk[x] = c9 * (-djk * fdmp + ang * dfdmp) / r2jk * vjk[x];
                    }
                    for x in 0..3 {
                        g[iat][x] -= gij[x] + gik[x];
                        g[jat][x] += gij[x] - gjk[x];
                        g[kat][x] += gik[x] + gjk[x];
                    }
                    f[iat] -=
                        de0 * 0.5 * (c6.d1[iat * n + jat] / c6ij + c6.d1[iat * n + kat] / c6ik);
                    f[jat] -=
                        de0 * 0.5 * (c6.d1[jat * n + iat] / c6ij + c6.d1[jat * n + kat] / c6jk);
                    f[kat] -=
                        de0 * 0.5 * (c6.d1[kat * n + iat] / c6ik + c6.d1[kat * n + jat] / c6jk);
                }
            }
        }
        (energy, g, f)
    }

    /// Add the CN chain rule `sum_k dE/dCN_k dCN_k/dR` to the gradient.
    fn add_cn_gradient(
        &self,
        xyz: &[[f64; 3]],
        cutoff: f64,
        dedcn: &[f64],
        gradient: &mut [[f64; 3]],
    ) {
        let n = self.natoms();
        let cutoff2 = cutoff * cutoff;
        gradient.par_iter_mut().enumerate().for_each(|(a, ga)| {
            for l in 0..n {
                let v = sub(xyz[a], xyz[l]);
                let r2 = norm2(v);
                if r2 > cutoff2 || r2 < 1.0e-12 {
                    continue;
                }
                let r = r2.sqrt();
                let (_, d1, _) = count3(self.rc(a, l), r);
                let s = d1 * (dedcn[a] + dedcn[l]) / r;
                for x in 0..3 {
                    ga[x] += s * v[x];
                }
            }
        });
    }

    /// Two-body radial function `g(r)` (pair energy `C6 g`) and its r derivatives.
    #[inline]
    fn pair_radial(&self, i: usize, j: usize, p: &D3Params, r: f64) -> (f64, f64, f64) {
        let (rr, r0) = self.rr_r0(i, j, p);
        let r02 = r0 * r0;
        let r06 = r02 * r02 * r02;
        let r08 = r06 * r02;
        let r2 = r * r;
        let r4 = r2 * r2;
        let r6 = r4 * r2;
        let t6 = 1.0 / (r6 + r06);
        let t8 = 1.0 / (r6 * r2 + r08);
        let e = -(p.s6 * t6 + p.s8 * rr * t8);
        let e1 = p.s6 * 6.0 * r4 * r * t6 * t6 + p.s8 * rr * 8.0 * r6 * r * t8 * t8;
        let e2 = p.s6 * (30.0 * r4 * t6 * t6 - 72.0 * r6 * r4 * t6 * t6 * t6)
            + p.s8 * rr * (56.0 * r6 * t8 * t8 - 128.0 * r6 * r6 * r2 * t8 * t8 * t8);
        (e, e1, e2)
    }

    /// Analytic Hessian (row-major 3n x 3n).
    fn hessian(
        &self,
        xyz: &[[f64; 3]],
        p: &D3Params,
        cut: &D3Cutoffs,
        c6: &C6Matrix,
        dedcn: &[f64],
    ) -> Vec<f64> {
        let n = self.natoms();
        let n3 = 3 * n;
        let mut h = vec![0.0; n3 * n3];
        let mut bm = vec![0.0; n3 * n]; // d2E / dR dCN
        let mut mm = vec![0.0; n * n]; // d2E / dCN dCN
        let two = p.s6.abs() >= f64::EPSILON || p.s8.abs() >= f64::EPSILON;
        let atm = p.s9.abs() >= f64::EPSILON;
        let cut2 = cut.disp2 * cut.disp2;
        let cut3 = cut.disp3 * cut.disp3;
        let cutn = cut.cn * cut.cn;
        let alp = p.alp + 2.0;
        let aexp = alp / 6.0;
        h.par_chunks_mut(3 * n3)
            .zip(bm.par_chunks_mut(3 * n))
            .zip(mm.par_chunks_mut(n))
            .enumerate()
            .for_each(|(a, ((hrows, brows), mrow))| {
                for j in 0..n {
                    if j == a {
                        continue;
                    }
                    let v = sub(xyz[a], xyz[j]);
                    let r2 = norm2(v);
                    let r = r2.sqrt();
                    let u = [v[0] / r, v[1] / r, v[2] / r];
                    // two-body pair (a, j)
                    if two && r2 <= cut2 && r2 >= f64::EPSILON {
                        let (g, g1, g2) = self.pair_radial(a, j, p, r);
                        let c = c6.c6[a * n + j];
                        let blk = radial_block(u, r, c * g1, c * g2);
                        add_pair_rows(hrows, n3, a, j, &blk);
                        let (da, dj) = (c6.d1[a * n + j], c6.d1[j * n + a]);
                        for x in 0..3 {
                            brows[x * n + a] += da * g1 * u[x];
                            brows[x * n + j] += dj * g1 * u[x];
                        }
                        mrow[a] += g * c6.d2[a * n + j];
                        mrow[j] += g * c6.dx[a * n + j];
                    }
                    // second derivative of the coordination numbers, weighted by dE/dCN
                    if r2 <= cutn && r2 >= 1.0e-12 {
                        let w = dedcn[a] + dedcn[j];
                        if w != 0.0 {
                            let (_, c1, c2) = count3(self.rc(a, j), r);
                            let blk = radial_block(u, r, w * c1, w * c2);
                            add_pair_rows(hrows, n3, a, j, &blk);
                        }
                    }
                }
                if atm {
                    self.atm_hessian_rows(a, xyz, p, cut3, aexp, c6, hrows, brows, mrow);
                }
            });
        // chain rule through the coordination numbers
        let jac = self.cn_jacobian(xyz, cut.cn);
        let mut x = vec![0.0; n3 * n3];
        par::gemm_acc(n3, n, n3, &bm, &jac, &mut x); // X = B J
        par::add_sym(n3, &x, &mut h); // + X + X^T
        let mut mj = vec![0.0; n * n3];
        par::gemm_acc(n, n, n3, &mm, &jac, &mut mj); // M J
        let jt = par::transpose(n, n3, &jac);
        par::gemm_acc(n3, n, n3, &jt, &mj, &mut h); // + J^T M J
        h
    }

    /// ATM contributions to the Hessian rows of atom `a` (and its rows of d2E/dRdCN and
    /// d2E/dCNdCN): all triples (a, j, k) with j < k.
    #[allow(clippy::too_many_arguments)]
    fn atm_hessian_rows(
        &self,
        a: usize,
        xyz: &[[f64; 3]],
        p: &D3Params,
        cut3: f64,
        aexp: f64,
        c6: &C6Matrix,
        hrows: &mut [f64],
        brows: &mut [f64],
        mrow: &mut [f64],
    ) {
        let n = self.natoms();
        let n3 = 3 * n;
        for j in 0..n {
            if j == a {
                continue;
            }
            let v1 = sub(xyz[j], xyz[a]);
            let u1 = norm2(v1);
            if u1 > cut3 || u1 < f64::EPSILON {
                continue;
            }
            let caj = c6.c6[a * n + j];
            for k in j + 1..n {
                if k == a {
                    continue;
                }
                let v2 = sub(xyz[k], xyz[a]);
                let u2 = norm2(v2);
                if u2 > cut3 || u2 < f64::EPSILON {
                    continue;
                }
                let v3 = sub(xyz[k], xyz[j]);
                let u3 = norm2(v3);
                if u3 > cut3 || u3 < f64::EPSILON {
                    continue;
                }
                let cak = c6.c6[a * n + k];
                let cjk = c6.c6[j * n + k];
                if caj.abs() < f64::EPSILON || cak.abs() < f64::EPSILON || cjk.abs() < f64::EPSILON
                {
                    continue;
                }
                let r0 = RS9
                    * self.rvdw[self.sp(j, a)]
                    * RS9
                    * self.rvdw[self.sp(k, a)]
                    * RS9
                    * self.rvdw[self.sp(k, j)];
                let cval = 6.0 * r0.powf(2.0 * aexp);
                let (gv, gp, gpq) = atm_kernel([u1, u2, u3], cval, aexp);
                let q = (caj * cak * cjk).abs().sqrt();
                let s = p.s9 * q;
                // Cartesian part, rows of atom a: grad_a u1 = -2 v1, grad_a u2 = -2 v2,
                // grad_j u1 = 2 v1, grad_k u2 = 2 v2, grad_j u3 = -2 v3, grad_k u3 = 2 v3
                let ga = [
                    [-2.0 * v1[0], -2.0 * v1[1], -2.0 * v1[2]],
                    [-2.0 * v2[0], -2.0 * v2[1], -2.0 * v2[2]],
                ];
                // gradients of u_q on (a, j, k)
                let gq: [[[f64; 3]; 3]; 3] = [
                    [ga[0], [2.0 * v1[0], 2.0 * v1[1], 2.0 * v1[2]], [0.0; 3]],
                    [ga[1], [0.0; 3], [2.0 * v2[0], 2.0 * v2[1], 2.0 * v2[2]]],
                    [
                        [0.0; 3],
                        [-2.0 * v3[0], -2.0 * v3[1], -2.0 * v3[2]],
                        [2.0 * v3[0], 2.0 * v3[1], 2.0 * v3[2]],
                    ],
                ];
                let atoms = [a, j, k];
                for x in 0..3 {
                    for (b, &atb) in atoms.iter().enumerate() {
                        for y in 0..3 {
                            let mut val = 0.0;
                            for pp in 0..2 {
                                for qq in 0..3 {
                                    val += gpq[pp][qq] * ga[pp][x] * gq[qq][b][y];
                                }
                            }
                            hrows[x * n3 + 3 * atb + y] += s * val;
                        }
                    }
                    // G_p * Hessian of u_p: u1 couples (a, j), u2 couples (a, k)
                    let d1 = 2.0 * s * gp[0];
                    let d2 = 2.0 * s * gp[1];
                    hrows[x * n3 + 3 * a + x] += d1 + d2;
                    hrows[x * n3 + 3 * j + x] -= d1;
                    hrows[x * n3 + 3 * k + x] -= d2;
                }
                // coordination-number couplings: Q = sqrt(C_aj C_ak C_jk), dQ/dCN_m = Q lam_m
                let lam_a = 0.5 * (c6.d1[a * n + j] / caj + c6.d1[a * n + k] / cak);
                let lam_j = 0.5 * (c6.d1[j * n + a] / caj + c6.d1[j * n + k] / cjk);
                let lam_k = 0.5 * (c6.d1[k * n + a] / cak + c6.d1[k * n + j] / cjk);
                // grad_a G
                let mut dga = [0.0; 3];
                for x in 0..3 {
                    dga[x] = gp[0] * ga[0][x] + gp[1] * ga[1][x];
                }
                for x in 0..3 {
                    brows[x * n + a] += s * lam_a * dga[x];
                    brows[x * n + j] += s * lam_j * dga[x];
                    brows[x * n + k] += s * lam_k * dga[x];
                }
                // d lam_a / dCN_n
                let daj = c6.d1[a * n + j] / caj;
                let dak = c6.d1[a * n + k] / cak;
                let dlaa =
                    0.5 * (c6.d2[a * n + j] / caj - daj * daj + c6.d2[a * n + k] / cak - dak * dak);
                let dlaj = 0.5 * (c6.dx[a * n + j] / caj - daj * (c6.d1[j * n + a] / caj));
                let dlak = 0.5 * (c6.dx[a * n + k] / cak - dak * (c6.d1[k * n + a] / cak));
                let e = s * gv;
                mrow[a] += e * (lam_a * lam_a + dlaa);
                mrow[j] += e * (lam_a * lam_j + dlaj);
                mrow[k] += e * (lam_a * lam_k + dlak);
            }
        }
    }
}

/// ATM kernel `G(u) = ang(u) fdmp(u)` of the squared distances `u = (u1, u2, u3)` with its
/// gradient and Hessian in `u` (s-dftd3 `get_atm_dispersion_hessian`).
/// `cval = 6 r0^(alp/3)`, `aexp = alp / 6`.
#[inline]
fn atm_kernel(u: [f64; 3], cval: f64, aexp: f64) -> (f64, [f64; 3], [[f64; 3]; 3]) {
    let w = u[0] * u[1] * u[2];
    let wp = [u[1] * u[2], u[0] * u[2], u[0] * u[1]];
    let mut wpq = [[0.0; 3]; 3];
    wpq[0][1] = u[2];
    wpq[1][0] = u[2];
    wpq[0][2] = u[1];
    wpq[2][0] = u[1];
    wpq[1][2] = u[0];
    wpq[2][1] = u[0];
    // L1 = u1 + u3 - u2, L2 = u1 - u3 + u2, L3 = -u1 + u3 + u2; cl[l][p] = dL_l/du_p
    let l = [u[0] + u[2] - u[1], u[0] - u[2] + u[1], -u[0] + u[2] + u[1]];
    const CL: [[f64; 3]; 3] = [[1.0, -1.0, 1.0], [1.0, 1.0, -1.0], [-1.0, 1.0, 1.0]];
    let nv = l[0] * l[1] * l[2];
    let mut np = [0.0; 3];
    for p in 0..3 {
        np[p] = CL[0][p] * l[1] * l[2] + CL[1][p] * l[0] * l[2] + CL[2][p] * l[0] * l[1];
    }
    let mut npq = [[0.0; 3]; 3];
    for p in 0..3 {
        for q in 0..3 {
            let mut t = 0.0;
            for il in 0..3 {
                for im in 0..3 {
                    if il != im {
                        t += CL[il][p] * CL[im][q] * l[3 - il - im];
                    }
                }
            }
            npq[p][q] = t;
        }
    }
    let ws = w.sqrt();
    let w_m15 = 1.0 / (w * ws);
    let w_m25 = w_m15 / w;
    let w_m35 = w_m25 / w;
    let w_m45 = w_m35 / w;
    let (gv, gd, gdd) = (w_m25, -2.5 * w_m35, 8.75 * w_m45);
    let (hv, hd, hdd) = (w_m15, -1.5 * w_m25, 3.75 * w_m35);
    let ang = 0.375 * nv * gv + hv;
    let mut angp = [0.0; 3];
    for p in 0..3 {
        angp[p] = 0.375 * (np[p] * gv + nv * gd * wp[p]) + hd * wp[p];
    }
    let mut angpq = [[0.0; 3]; 3];
    for p in 0..3 {
        for q in 0..3 {
            angpq[p][q] = 0.375
                * (npq[p][q] * gv
                    + np[p] * gd * wp[q]
                    + np[q] * gd * wp[p]
                    + nv * (gdd * wp[p] * wp[q] + gd * wpq[p][q]))
                + hdd * wp[p] * wp[q]
                + hd * wpq[p][q];
        }
    }
    let t = cval * w.powf(-aexp);
    let mut tp = [0.0; 3];
    for p in 0..3 {
        tp[p] = -aexp * t * wp[p] / w;
    }
    let fd = 1.0 / (1.0 + t);
    let mut fdp = [0.0; 3];
    for p in 0..3 {
        fdp[p] = -tp[p] * fd * fd;
    }
    let mut g2 = [[0.0; 3]; 3];
    let mut g1 = [0.0; 3];
    for p in 0..3 {
        g1[p] = angp[p] * fd + ang * fdp[p];
    }
    for p in 0..3 {
        for q in 0..3 {
            let tpq = aexp * (aexp + 1.0) * t * wp[p] * wp[q] / (w * w) - aexp * t * wpq[p][q] / w;
            let fdpq = -tpq * fd * fd + 2.0 * tp[p] * tp[q] * fd * fd * fd;
            g2[p][q] = angpq[p][q] * fd + angp[p] * fdp[q] + angp[q] * fdp[p] + ang * fdpq;
        }
    }
    (ang * fd, g1, g2)
}

/// DFT-D3(BJ) energy, gradient and Hessian of a molecule.
///
/// `z`: atomic numbers (1-103); `xyz_bohr`: positions in bohr. Runs on the current rayon
/// pool; the result is bitwise identical for any number of threads.
pub fn d3bj(
    z: &[u32],
    xyz_bohr: &[[f64; 3]],
    params: &D3Params,
    opts: &D3Options,
) -> Result<DispResult, DispError> {
    if z.len() != xyz_bohr.len() {
        return Err(DispError::Shape(format!(
            "{} atomic numbers but {} positions",
            z.len(),
            xyz_bohr.len()
        )));
    }
    let model = D3Model::new(z)?;
    Ok(model.compute(xyz_bohr, params, opts))
}

/// Gradient of [`d3bj`] split into its parts (diagnostics).
#[derive(Clone, Debug)]
pub struct D3GradientParts {
    /// two-body term at fixed coordination numbers
    pub two_body: Vec<[f64; 3]>,
    /// ATM term at fixed coordination numbers
    pub three_body: Vec<[f64; 3]>,
    /// `sum_k dE/dCN_k dCN_k/dR` (both terms)
    pub cn_chain: Vec<[f64; 3]>,
    /// `dE/dCN` of the two-body term and of the ATM term
    pub dedcn_two_body: Vec<f64>,
    pub dedcn_three_body: Vec<f64>,
}

impl D3Model {
    /// The parts of the D3 gradient: explicit two-body and ATM terms and the
    /// coordination-number chain rule.
    pub fn gradient_parts(
        &self,
        xyz: &[[f64; 3]],
        params: &D3Params,
        cutoffs: &D3Cutoffs,
    ) -> D3GradientParts {
        let n = self.natoms();
        let cn = self.coordination_numbers(xyz, cutoffs.cn);
        let w = self.weights(&cn, 1);
        let c6 = self.c6_matrix(&w, 1);
        let p2 = D3Params { s9: 0.0, ..*params };
        let (_, g2, f2) = self.energy_gradient(xyz, &p2, cutoffs, &c6, true);
        let (g3, f3) = if params.s9.abs() >= f64::EPSILON {
            let p3 = D3Params {
                s6: 0.0,
                s8: 0.0,
                ..*params
            };
            let (_, g, f) = self.energy_gradient(xyz, &p3, cutoffs, &c6, true);
            (g, f)
        } else {
            (vec![[0.0; 3]; n], vec![0.0; n])
        };
        let f: Vec<f64> = f2.iter().zip(&f3).map(|(a, b)| a + b).collect();
        let mut chain = vec![[0.0; 3]; n];
        self.add_cn_gradient(xyz, cutoffs.cn, &f, &mut chain);
        D3GradientParts {
            two_body: g2,
            three_body: g3,
            cn_chain: chain,
            dedcn_two_body: f2,
            dedcn_three_body: f3,
        }
    }

    /// Energy and the derivatives requested by `opts.order` at one geometry.
    pub fn compute(&self, xyz: &[[f64; 3]], params: &D3Params, opts: &D3Options) -> DispResult {
        let order = match opts.order {
            Order::Energy => 0,
            Order::Gradient => 1,
            Order::Hessian => 2,
        };
        let cut = &opts.cutoffs;
        let cn = self.coordination_numbers(xyz, cut.cn);
        let w = self.weights(&cn, order);
        let c6 = self.c6_matrix(&w, order);
        let (energy, mut gradient, dedcn) = self.energy_gradient(xyz, params, cut, &c6, order >= 1);
        let hessian = if order >= 2 {
            Some(self.hessian(xyz, params, cut, &c6, &dedcn))
        } else {
            None
        };
        if order >= 1 {
            self.add_cn_gradient(xyz, cut.cn, &dedcn, &mut gradient);
        }
        DispResult {
            energy,
            gradient,
            hessian,
        }
    }
}
