//! The plan of the ECP integrals ([`EcpPlan`]) and their nuclear first and second
//! derivatives: gradient contractions sum_ab D_ab dU_ab/dX, derivative matrices dU/dX
//! (3 natom x nbf x nbf, for CPHF right-hand sides) and Hessian contractions
//! sum_ab D_ab d2U_ab/dXdY (3 natom x 3 natom). Conventions as in [`crate::one_e::deriv`]
//! (perturbation index 3 * atom + k, densities symmetric row-major).
//!
//! Every ECP term <a|U_C|b> depends on the centers A, B of the functions and C of the ECP and
//! is invariant under a common translation, so d/dC = -(d/dA + d/dB).
//! - Local part (type 1): as the nuclear attraction in [`crate::one_e::deriv`], with the Hermite
//!   tensor of the operator from [`crate::one_e::ecp::local_r`]: d/dA from the derivative Hermite
//!   expansion of the pair ([`crate::core::deriv`]), d/dC from R_{t+1} (d/dC R(P - C) = -dR/dP),
//!   d/dB by translation.
//! - Semi-local part (type 2): d/dA of a function is a combination of Cartesian functions of
//!   angular momentum l +- 1 (2a G_{i+1} - i G_{i-1}), so its projections are products of the
//!   same projections ([`crate::one_e::ecp::shell_proj`]) and every derivative is a product of
//!   projection matrices over (l, m, g); d/dC by translation. Each contraction checks itself
//!   against the embedded coarser radial grid and raises the grid level until they agree to
//!   [`crate::one_e::ecp::LEVEL_TOL`] (as the matrix does).

use crate::basis::BasisSet;
use crate::basis::Shell;
use crate::contract::AoMatrixMut;
use crate::core::deriv::PairDeriv;
use crate::core::hermite::{ShellPair, ShellView};
use crate::core::pair_data::{self, PRIM_PAIR_CUT};
use crate::core::util::add_grads_into;
use crate::engine::WorkspaceSize;
use crate::one_e::ecp::{
    active_shells, center_grid, ecp_matrix_into, gemm_nt_par, proj_matrices, scratch_len,
    CenterGrid, EcpCenter, EcpSet, EcpStats, ProjScratch, LEVEL_MAX, LEVEL_TOL,
};
use crate::pairs::{DerivOrder, PairSet};
use crate::potential::{EcpLocal, Pairs, PotentialPlan, PotentialWorkspace};
use rayon::prelude::*;

// ---------------------------------------------------------------------------------------
// Local part
// ---------------------------------------------------------------------------------------

fn local_centers(ecp: &EcpSet) -> Vec<&EcpCenter> {
    ecp.centers
        .iter()
        .filter(|c| !c.def.local.is_empty())
        .collect()
}

/// Whether a shell pair can have a non-negligible local-channel integral with one of the
/// centers: the most diffuse primitives (the triple-Gaussian exponent
/// (a b AB^2 + a z AC^2 + b z BC^2)/(a + b + z) grows with every exponent) against e^-60.
fn local_relevant(a: &Shell, b: &Shell, cs: &[&EcpCenter]) -> bool {
    let amin = a.exps.iter().cloned().fold(f64::INFINITY, f64::min);
    let bmin = b.exps.iter().cloned().fold(f64::INFINITY, f64::min);
    let d2 = |x: [f64; 3], y: [f64; 3]| {
        (x[0] - y[0]).powi(2) + (x[1] - y[1]).powi(2) + (x[2] - y[2]).powi(2)
    };
    let ab = d2(a.center, b.center);
    cs.iter().any(|c| {
        let z = c
            .def
            .local
            .iter()
            .map(|t| t.zeta)
            .fold(f64::INFINITY, f64::min);
        let x = (amin * bmin * ab
            + amin * z * d2(a.center, c.center)
            + bmin * z * d2(b.center, c.center))
            / (amin + bmin + z);
        x < 60.0
    })
}

/// The shell pairs (indices into a [`PairSet`]) with non-negligible local-channel integrals
/// and their derivative Hermite expansions of `order` (built only for these pairs: most pairs
/// of a molecule are far from every ECP).
struct LocalPairs {
    idx: Vec<usize>,
    derivs: Vec<PairDeriv>,
}

impl LocalPairs {
    fn new(basis: &BasisSet, pairs: &PairSet, ecp: &EcpSet, order: usize) -> Self {
        let cs = local_centers(ecp);
        let sh = &basis.shells;
        let idx: Vec<usize> = (0..pairs.len())
            .into_par_iter()
            .filter(|&k| {
                let pr = pairs.pair(k);
                local_relevant(&sh[pr.sa], &sh[pr.sb], &cs)
            })
            .collect();
        let derivs = idx
            .par_iter()
            .map(|&k| {
                let pr = pairs.pair(k);
                pair_data::build(
                    &ShellView::from_shell(&sh[pr.sa]),
                    &ShellView::from_shell(&sh[pr.sb]),
                    pr.sa,
                    pr.sb,
                    order,
                    PRIM_PAIR_CUT,
                )
                .1
                .expect("derivative order >= 1")
            })
            .collect();
        LocalPairs { idx, derivs }
    }
}

// ---------------------------------------------------------------------------------------
// Semi-local part
// ---------------------------------------------------------------------------------------

/// The semi-local channels' buffers of an [`EcpWorkspace`]: the projection matrices of a
/// center (one nrow x ncol matrix per derivative component, at stride nrow * ncol), H = D P,
/// and per thread the shell projections' scratch. They grow to the largest grid level a
/// contraction reaches and are reused level after level and call after call (as temporaries
/// they were mapped, faulted in and returned at every level).
#[derive(Default)]
struct SemiWork {
    mats: Vec<f64>,
    h: Vec<f64>,
    proj: Vec<ProjScratch>,
}

/// Projection data of one center at one grid level (its matrices in the [`SemiWork`]).
struct Proj {
    grid: CenterGrid,
    aos: Vec<usize>,
    atoms: Vec<usize>,
    w: Vec<f64>,
}

impl Proj {
    fn new(basis: &BasisSet, c: &EcpCenter, level: u32, order: usize, sw: &mut SemiWork) -> Self {
        let grid = center_grid(c, level);
        let shells = active_shells(basis, c, &grid);
        let nt = rayon::current_num_threads();
        if sw.proj.len() < nt {
            sw.proj.resize_with(nt, ProjScratch::default);
        }
        let (aos, atoms) =
            proj_matrices(basis, c, &grid, &shells, order, &mut sw.mats, &mut sw.proj);
        let w = grid.col_weights();
        Proj {
            grid,
            aos,
            atoms,
            w,
        }
    }
    fn n(&self) -> usize {
        self.aos.len()
    }
    fn ld(&self) -> usize {
        self.grid.ncol()
    }
    /// Projection matrix `d` (a derivative component) in the buffer `mats`.
    fn mat<'w>(&self, mats: &'w [f64], d: usize) -> &'w [f64] {
        let stride = self.n() * self.ld();
        &mats[d * stride..(d + 1) * stride]
    }
    /// The density on the active functions (n x n).
    fn dens(&self, d: &[f64], nbf: usize) -> Vec<f64> {
        let n = self.n();
        let mut out = vec![0.0; n * n];
        for (i, &a) in self.aos.iter().enumerate() {
            for (j, &b) in self.aos.iter().enumerate() {
                out[i * n + j] = d[a * nbf + b];
            }
        }
        out
    }
    /// H = D P (n x ncol) in the buffer `h`, rows in parallel; `p` the projection matrix of the
    /// values.
    fn dp<'h>(&self, dact: &[f64], p: &[f64], h: &'h mut Vec<f64>) -> &'h [f64] {
        let n = self.n();
        let ld = self.ld();
        let h = scratch_len(h, n * ld);
        h.fill(0.0);
        if ld == 0 {
            return h;
        }
        h.par_chunks_mut(ld).enumerate().for_each(|(i, row)| {
            for j in 0..n {
                let dij = dact[i * n + j];
                if dij == 0.0 {
                    continue;
                }
                for (x, &y) in row.iter_mut().zip(&p[j * ld..(j + 1) * ld]) {
                    *x += dij * y;
                }
            }
        });
        h
    }
    /// Per row i: sum_k a[i][k] w[k] h[i][k], split into coarse and fine columns.
    fn row_dots(&self, a: &[f64], h: &[f64]) -> Vec<(f64, f64)> {
        let ld = self.ld();
        let kc = self.grid.ncol_coarse();
        let w = &self.w;
        (0..self.n())
            .into_par_iter()
            .map(|i| {
                let ar = &a[i * ld..(i + 1) * ld];
                let hr = &h[i * ld..(i + 1) * ld];
                let mut sc = 0.0;
                for k in 0..kc {
                    sc += ar[k] * w[k] * hr[k];
                }
                let mut sf = 0.0;
                for k in kc..ld {
                    sf += ar[k] * w[k] * hr[k];
                }
                (sc, sf)
            })
            .collect()
    }
}

/// Run `f` on rising grid levels until its result (fine, coarse estimate) agrees.
fn converge<T>(mut f: impl FnMut(u32) -> (T, f64)) -> T {
    let mut level = crate::one_e::ecp::start_level();
    loop {
        let (t, err) = f(level);
        if err <= LEVEL_TOL || level >= LEVEL_MAX {
            return t;
        }
        level += 1;
    }
}

/// Semi-local gradient of one center (per atom).
fn semilocal_grad_center(
    basis: &BasisSet,
    c: &EcpCenter,
    d: &[f64],
    sw: &mut SemiWork,
) -> Vec<[f64; 3]> {
    let natom = basis.natom();
    converge(|level| {
        let pj = Proj::new(basis, c, level, 1, sw);
        let dact = pj.dens(d, basis.nbf);
        let h = pj.dp(&dact, pj.mat(&sw.mats, 0), &mut sw.h);
        let mut g = vec![[0.0; 3]; natom];
        let mut gc = vec![[0.0; 3]; natom];
        for x in 0..3 {
            let dots = pj.row_dots(pj.mat(&sw.mats, 1 + x), h);
            for (i, &(sc, sf)) in dots.iter().enumerate() {
                let (a, b) = (2.0 * (sc + sf), 4.0 * sc);
                g[pj.atoms[i]][x] += a;
                g[c.atom][x] -= a;
                gc[pj.atoms[i]][x] += b;
                gc[c.atom][x] -= b;
            }
        }
        let err = g
            .iter()
            .zip(&gc)
            .flat_map(|(a, b)| (0..3).map(move |k| (a[k] - b[k]).abs()))
            .fold(0.0, f64::max);
        (g, err)
    })
}

/// Semi-local derivative matrices of one center, added into `dv` ((3 natom) x nbf^2).
fn semilocal_deriv_center(basis: &BasisSet, c: &EcpCenter, dv: &mut [f64], sw: &mut SemiWork) {
    let nbf = basis.nbf;
    let n2 = nbf * nbf;
    let (pj, ys) = converge(|level| {
        let pj = Proj::new(basis, c, level, 1, sw);
        let n = pj.n();
        let ld = pj.ld();
        let kc = pj.grid.ncol_coarse();
        let mut ys = Vec::with_capacity(3);
        let mut err: f64 = 0.0;
        for x in 0..3 {
            // Y_x[i][j] = <d_x a_i|U|a_j>
            let (px, p0) = (pj.mat(&sw.mats, 1 + x), pj.mat(&sw.mats, 0));
            let yc = gemm_nt_par(px, n, p0, n, ld, Some(&pj.w), 0, kc);
            let yf = gemm_nt_par(px, n, p0, n, ld, Some(&pj.w), kc, ld);
            let y: Vec<f64> = yc
                .iter()
                .zip(&yf)
                .map(|(a, b)| {
                    err = err.max((a - b).abs());
                    a + b
                })
                .collect();
            ys.push(y);
        }
        ((pj, ys), err)
    });
    let n = pj.n();
    for (x, y) in ys.iter().enumerate() {
        for i in 0..n {
            let (a, ai) = (pj.aos[i], pj.atoms[i]);
            for j in 0..n {
                let b = pj.aos[j];
                let v = y[i * n + j];
                // d/dA_x of <a|U|b> for a on atom ai: Y[i][j] at (a, b) and (b, a)
                let pa = (3 * ai + x) * n2;
                dv[pa + a * nbf + b] += v;
                dv[pa + b * nbf + a] += v;
                let pc = (3 * c.atom + x) * n2;
                dv[pc + a * nbf + b] -= v;
                dv[pc + b * nbf + a] -= v;
            }
        }
    }
}

/// Semi-local Hessian of one center, added into `hess` ((3 natom)^2).
fn semilocal_hess_center(
    basis: &BasisSet,
    c: &EcpCenter,
    d: &[f64],
    hess: &mut [f64],
    sw: &mut SemiWork,
) {
    let natom = basis.natom();
    let n3 = 3 * natom;
    // basis-center part hb (and the coarse estimate)
    let hb = converge(|level| {
        let pj = Proj::new(basis, c, level, 2, sw);
        let n = pj.n();
        let ld = pj.ld();
        let kc = pj.grid.ncol_coarse();
        let dact = pj.dens(d, basis.nbf);
        let h = pj.dp(&dact, pj.mat(&sw.mats, 0), &mut sw.h);
        let mut hb = vec![0.0; n3 * n3];
        let mut hc = vec![0.0; n3 * n3];
        // second derivatives of one function: 2 sum_i P''_kl[i] W H[i] on the atom of i
        for (kl, &(k, l)) in crate::core::deriv::D2_COMPS.iter().enumerate() {
            let dots = pj.row_dots(pj.mat(&sw.mats, 4 + kl), h);
            for (i, &(sc, sf)) in dots.iter().enumerate() {
                let a = pj.atoms[i];
                for (m, v) in [(&mut hb, 2.0 * (sc + sf)), (&mut hc, 4.0 * sc)] {
                    m[(3 * a + k) * n3 + 3 * a + l] += v;
                    if k != l {
                        m[(3 * a + l) * n3 + 3 * a + k] += v;
                    }
                }
            }
        }
        // first derivatives of both functions: 2 sum_ij D_ij <d_k a_i|U|d_l a_j>; X_lk = X_kl^T,
        // so only k <= l is formed (6 products instead of 9)
        for k in 0..3 {
            for l in k..3 {
                let (pk, pl) = (pj.mat(&sw.mats, 1 + k), pj.mat(&sw.mats, 1 + l));
                let xc = gemm_nt_par(pk, n, pl, n, ld, Some(&pj.w), 0, kc);
                let xf = gemm_nt_par(pk, n, pl, n, ld, Some(&pj.w), kc, ld);
                for i in 0..n {
                    let a = pj.atoms[i];
                    for j in 0..n {
                        let b = pj.atoms[j];
                        let dij = dact[i * n + j];
                        let (c_ij, f_ij) = (xc[i * n + j], xf[i * n + j]);
                        let o = (3 * a + k) * n3 + 3 * b + l;
                        hb[o] += 2.0 * dij * (c_ij + f_ij);
                        hc[o] += 4.0 * dij * c_ij;
                        if k != l {
                            // <d_l a_i|U|d_k a_j> = X_kl[j][i]
                            let (c_ji, f_ji) = (xc[j * n + i], xf[j * n + i]);
                            let o = (3 * a + l) * n3 + 3 * b + k;
                            hb[o] += 2.0 * dij * (c_ji + f_ji);
                            hc[o] += 4.0 * dij * c_ji;
                        }
                    }
                }
            }
        }
        let err = hb
            .iter()
            .zip(&hc)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        (hb, err)
    });
    // fold in the ECP center: d/dC = -sum_A d/dA
    let ca = c.atom;
    let mut colsum = vec![[[0.0; 3]; 3]; natom]; // colsum[B][x][y] = sum_A hb[Ax][By]
    for a in 0..natom {
        for b in 0..natom {
            for x in 0..3 {
                for y in 0..3 {
                    colsum[b][x][y] += hb[(3 * a + x) * n3 + 3 * b + y];
                }
            }
        }
    }
    for (i, v) in hb.iter().enumerate() {
        hess[i] += v;
    }
    let mut tot = [[0.0; 3]; 3];
    for b in 0..natom {
        for x in 0..3 {
            for y in 0..3 {
                let v = colsum[b][x][y];
                hess[(3 * ca + x) * n3 + 3 * b + y] -= v;
                hess[(3 * b + y) * n3 + 3 * ca + x] -= v;
                tot[x][y] += v;
            }
        }
    }
    for x in 0..3 {
        for y in 0..3 {
            hess[(3 * ca + x) * n3 + 3 * ca + y] += tot[x][y];
        }
    }
}

fn semilocal_centers(ecp: &EcpSet) -> impl Iterator<Item = &EcpCenter> {
    ecp.centers.iter().filter(|c| !c.def.semilocal.is_empty())
}

// ---------------------------------------------------------------------------------------
// The plan
// ---------------------------------------------------------------------------------------

/// The caller-owned memory of an [`EcpPlan`]'s contractions: the local channels' potential
/// workspace and the semi-local channels' projection buffers (grown to the largest grid level
/// a contraction reaches, then reused). Reusable across calls of its plan.
pub struct EcpWorkspace {
    local: PotentialWorkspace,
    semi: SemiWork,
}

/// The ECP integrals of a basis and their nuclear derivatives up to an order: the matrix, the
/// derivative matrices and the gradient and Hessian contractions with a density, into the
/// caller's buffers. The local channels run on the potential engine
/// ([`crate::potential::PotentialPlan`]) over the shell pairs near an ECP center, found once
/// per plan with their derivative expansions; the semi-local channels per center.
pub struct EcpPlan<'a> {
    basis: &'a BasisSet,
    pairs: &'a PairSet,
    ecp: &'a EcpSet,
    order: DerivOrder,
    /// the centers with a local channel
    centers: Vec<&'a EcpCenter>,
    /// the pairs near them (orders one and two; none without an ECP)
    local: Option<LocalPairs>,
}

impl<'a> EcpPlan<'a> {
    /// The plan of the ECP integrals of `ecp` over `basis` (whose pairs are `pairs`) and their
    /// derivatives up to `order`.
    pub fn new(
        basis: &'a BasisSet,
        pairs: &'a PairSet,
        ecp: &'a EcpSet,
        order: DerivOrder,
    ) -> Self {
        let local = (order != DerivOrder::Zero && !ecp.is_empty())
            .then(|| LocalPairs::new(basis, pairs, ecp, order.get()));
        EcpPlan {
            basis,
            pairs,
            ecp,
            order,
            centers: local_centers(ecp),
            local,
        }
    }

    /// The plan's derivative order.
    pub fn order(&self) -> DerivOrder {
        self.order
    }

    /// The potential plan of the local channels over the pairs near them.
    #[allow(
        clippy::type_complexity,
        reason = "the closure type of the pair selection"
    )]
    fn local_plan<'s>(
        &'s self,
    ) -> Option<
        PotentialPlan<
            's,
            EcpLocal<'s>,
            impl Fn(usize) -> Option<(&'s ShellPair, &'s PairDeriv)> + Sync + 's,
        >,
    > {
        let lp = self.local.as_ref()?;
        let near = Pairs {
            npairs: lp.idx.len(),
            pair: move |k| Some((self.pairs.pair(lp.idx[k]), &lp.derivs[k])),
        };
        let src = EcpLocal {
            centers: &self.centers,
        };
        let natom = self.basis.natom();
        Some(PotentialPlan::new(self.basis, near, src, natom, self.order))
    }

    /// The bytes of a workspace for the contractions of the plan's order and below: the local
    /// channels' (see [`EcpWorkspace`]: the semi-local buffers grow to the grid level a
    /// contraction reaches, a few MB per ECP center).
    pub fn workspace_size(&self) -> WorkspaceSize {
        self.local_plan()
            .map(|p| p.workspace_size())
            .unwrap_or_default()
    }

    /// A workspace for the contractions of the plan's order and below.
    pub fn workspace(&self) -> EcpWorkspace {
        let local = match self.local_plan() {
            Some(p) => p.workspace(),
            None => PotentialWorkspace::new(0, 0, DerivOrder::One),
        };
        EcpWorkspace {
            local,
            semi: SemiWork::default(),
        }
    }

    /// The ECP matrix sum_C <a|U_C|b> into `v` (overwritten), with the build statistics.
    pub fn matrix(&self, v: &mut AoMatrixMut<'_>) -> EcpStats {
        ecp_matrix_into(self.basis, self.pairs, self.ecp, v)
    }

    /// The gradient sum_ab D_ab d<a|U_ECP|b>/dX into `out` (natom rows, overwritten; D
    /// symmetric). An order-One plan or higher.
    pub fn gradient(&self, d: &[f64], out: &mut [[f64; 3]], ws: &mut EcpWorkspace) {
        assert!(
            self.order >= DerivOrder::One,
            "the gradient needs an order-One plan"
        );
        assert_eq!(out.len(), self.basis.natom(), "the gradient has natom rows");
        match self.local_plan() {
            Some(p) => p.gradient(d, out, &mut ws.local),
            None => return out.fill([0.0; 3]),
        }
        for c in semilocal_centers(self.ecp) {
            add_grads_into(out, &semilocal_grad_center(self.basis, c, d, &mut ws.semi));
        }
    }

    /// The derivative matrices dU_ECP/dX added into `out` ((3 natom) x nbf x nbf; e.g. the
    /// core-Hamiltonian derivatives, with no second copy). An order-One plan or higher.
    pub fn add_deriv_matrices(&self, out: &mut [f64], ws: &mut EcpWorkspace) {
        assert!(
            self.order >= DerivOrder::One,
            "derivative matrices need an order-One plan"
        );
        let nbf = self.basis.nbf;
        assert_eq!(out.len(), 3 * self.basis.natom() * nbf * nbf);
        let Some(p) = self.local_plan() else {
            return;
        };
        p.add_deriv_matrices(out);
        for c in semilocal_centers(self.ecp) {
            semilocal_deriv_center(self.basis, c, out, &mut ws.semi);
        }
    }

    /// The Hessian sum_ab D_ab d2<a|U_ECP|b>/dXdY into `out` ((3 natom)^2 row-major,
    /// overwritten). An order-Two plan.
    pub fn hessian(&self, d: &[f64], out: &mut [f64], ws: &mut EcpWorkspace) {
        assert_eq!(
            self.order,
            DerivOrder::Two,
            "the Hessian needs an order-Two plan"
        );
        let n3 = 3 * self.basis.natom();
        assert_eq!(out.len(), n3 * n3, "the Hessian is (3 natom)^2");
        match self.local_plan() {
            Some(p) => p.hessian(d, out, &mut ws.local),
            None => return out.fill(0.0),
        }
        for c in semilocal_centers(self.ecp) {
            semilocal_hess_center(self.basis, c, d, out, &mut ws.semi);
        }
    }
}
