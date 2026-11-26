//! Compensated grid-based exchange ("compensated Grid-K"; Hohenstein, Alkan, Wu & Parrish,
//! arXiv:2611.16420, 2026), written from the paper's equations (their numbers in brackets).
//!
//! Grid-K integrates one electron of the exchange integrals on a molecular quadrature
//! {r_A, w_A} and the other analytically \[11\]:
//!
//!   (pq|rs)^grid = P_pq|rs sum_A w_A X_Ap X_Aq V^A_rs,   X_Ap = phi_p(r_A),
//!   V^A_rs = <r| O(|r - r_A|) |s>   (the operator of the exchange, [`Operator`]; the potential
//!   integrals of [`linus_ints::points::grid_pot`]).
//!
//! Two compensations, independent of each other ([`GridKConfig::dscf`], [`GridKConfig::fit`]):
//!
//! * **dSCF** \[7-10\]: the density is split into a fixed superposition of spherical atomic
//!   densities D0 (atom-block-diagonal, the same at every geometry: the caller's SAD densities)
//!   and the difference D_d = D - D0. The energy is defined first,
//!
//!     E_K = 1/2 D0 . K\[D0\] + D_d . K\[D0\] + 1/2 D_d . K^grid\[D_d\],
//!
//!   with exact integrals in every term that holds D0 and the grid only in the small quadratic
//!   term; K = dE_K/dD = K\[D0\] + K^grid\[D_d\] is its exact derivative, so the SCF stays
//!   variational. K is affine in D, and E_K is not 1/2 D . K: every build reports E_K
//!   ([`GridKBuild::energy`]).
//! * **The symmetric overlap fit** \[21-25\]: the collocation matrix is dressed with the transfer
//!   matrix M, Xbar = X M^T ([`OverlapFit`]), so that Xbar^T W Xbar reproduces the analytic
//!   overlap on the space the SCF keeps (eigenvalues of S above `scf.lindep`).
//!
//! A build of one density: Ybar = Xbar D_d \[28\], Zbar_A = V^A Ybar_A \[29\] (the potential integrals
//! contracted point by point, screened per batch of points), Kbar = 1/2 (Xbar^T W Zbar + h.c.)
//! \[30\] and E^grid = 1/2 sum_A w_A Ybar_A . Zbar_A \[40\].
//!
//! This is the CPU reference: dense collocation (X and Xbar are npts x nbf), the integrals with
//! the CPU's Rys kernels per batch. Results do not depend on the thread count: Zbar has one
//! writer per point, and the products run through linus-base's linear algebra.

pub mod gradient;

use crate::exact::FastJk;
use faer::{Mat, MatRef, Par};
use linus_base::error::LinusError;
use linus_base::linalg::{ao_view_mut, dot, eigh, mm};
use linus_base::terms::{JkBuilder, JkOutput, JkRequest};
use linus_grid::{Batch, MolecularGrid};
use linus_ints::operator::Operator;
use linus_ints::points::ao::eval_ao;
use linus_ints::points::grid_pot::{PotContract, PotPair, PotPrim};
use linus_ints::{BasisSet, DerivOrder, OneEPlan};
use rayon::prelude::*;
use std::sync::Arc;

/// Points per block of the integral passes (the Rys kernels' lanes need multiples of 8).
pub const CHUNK: usize = 128;

/// The options of a grid-K builder.
#[derive(Clone, Debug)]
pub struct GridKConfig {
    /// The dSCF partition of the exchange energy (needs the SAD densities).
    pub dscf: bool,
    /// The symmetric overlap fit in the canonically orthogonalized basis.
    pub fit: bool,
    /// Eigenvalues of S at or below this are dropped (the SCF's `scf.lindep`, so both use the
    /// same canonical orbitals).
    pub lindep: f64,
    /// Screening threshold of a shell pair's contribution in a batch of points: its integrals'
    /// bound times max |Ybar| over the pair's functions and the batch, times the batch's
    /// max_p sum_A w_A |Xbar_Ap|.
    pub thresh: f64,
    /// Screening threshold of the exact exchange of the SAD densities (K\[D0\]).
    pub exact_thresh: f64,
    /// Primitive pairs whose estimate is below this are dropped at setup.
    pub prim_cut: f64,
    /// The exchange operator (coefficients included for range-separated hybrids).
    pub xop: Operator,
}

impl Default for GridKConfig {
    fn default() -> Self {
        GridKConfig {
            dscf: true,
            fit: true,
            lindep: 1e-8,
            thresh: 1e-12,
            exact_thresh: 1e-12,
            prim_cut: 1e-20,
            xop: Operator::Coulomb,
        }
    }
}

/// The symmetric overlap fit \[21-25\] and what its derivative needs.
///
/// With S = U s U^T (eigenvalues above `lindep` kept): A = U s^-1/2 (the canonical orthogonal
/// orbitals), B = U s^1/2 = S A; Omega = A^T S^grid A (the grid overlap of the orthogonal
/// orbitals, S^grid = X^T W X) = Theta theta Theta^T; T = Omega^-1/2; M = B T A^T. The dressed
/// collocation Xbar = X M^T = (X A) T B^T has Xbar^T W Xbar = B B^T = U s U^T.
#[derive(Clone, Debug)]
pub struct OverlapFit {
    pub a: Mat<f64>,
    pub b: Mat<f64>,
    /// Eigenvalues and eigenvectors of Omega.
    pub theta: Vec<f64>,
    pub vecs: Mat<f64>,
    pub t: Mat<f64>,
    pub m: Mat<f64>,
}

impl OverlapFit {
    /// The fit of the collocation `x` (npts x nbf) with weights `w` to the overlap `s`. An
    /// error when the grid's metric Omega is not positive definite (a grid far too coarse for
    /// the basis, \[25\]).
    pub fn new(
        s: MatRef<'_, f64>,
        x: MatRef<'_, f64>,
        w: &[f64],
        lindep: f64,
        par: Par,
    ) -> Result<Self, String> {
        let n = s.nrows();
        let (sv, u) = eigh(s, par);
        let keep: Vec<usize> = (0..n).filter(|&i| sv[i] > lindep).collect();
        let m = keep.len();
        let a = Mat::from_fn(n, m, |i, k| u[(i, keep[k])] / sv[keep[k]].sqrt());
        let b = Mat::from_fn(n, m, |i, k| u[(i, keep[k])] * sv[keep[k]].sqrt());
        let chi = mm(x, a.as_ref(), par);
        let wchi = Mat::from_fn(chi.nrows(), m, |g, k| w[g] * chi[(g, k)]);
        let mut omega = mm(chi.transpose(), wchi.as_ref(), par);
        sym(&mut omega);
        let (theta, vecs) = eigh(omega.as_ref(), par);
        if let Some(&t0) = theta.first() {
            if t0 <= 0.0 {
                return Err(format!(
                    "the exchange grid is too coarse for the overlap fit: its metric of the \
                     orthogonal orbitals has the eigenvalue {t0:.3e} (use a larger k_grid)"
                ));
            }
        }
        let vt = Mat::from_fn(m, m, |i, k| vecs[(i, k)] / theta[k].sqrt());
        let t = mm(vt.as_ref(), vecs.transpose(), par);
        let bt = mm(b.as_ref(), t.as_ref(), par);
        let mt = mm(bt.as_ref(), a.transpose(), par);
        Ok(OverlapFit {
            a,
            b,
            theta,
            vecs,
            t,
            m: mt,
        })
    }

    /// Orbitals kept by the canonical orthogonalization.
    pub fn nmo(&self) -> usize {
        self.a.ncols()
    }
}

/// (A + A^T) / 2 in place.
pub(crate) fn sym(a: &mut Mat<f64>) {
    let n = a.nrows();
    for j in 0..n {
        for i in j + 1..n {
            let v = 0.5 * (a[(i, j)] + a[(j, i)]);
            a[(i, j)] = v;
            a[(j, i)] = v;
        }
    }
}

/// One build of a list of densities: K and E_K per density, the operator's coefficients
/// included (multiply by the hybrid's a_x as for any K).
#[derive(Clone, Debug, Default)]
pub struct GridKBuild {
    pub k: Vec<Mat<f64>>,
    pub energy: Vec<f64>,
    /// The parts of each energy: 1/2 D0 . K\[D0\] + D_d . K\[D0\] (exact) and the grid's.
    pub e_exact: Vec<f64>,
    pub e_grid: Vec<f64>,
}

/// Counters of a builder.
#[derive(Clone, Debug, Default)]
pub struct GridKStats {
    pub npoints: usize,
    pub npairs: usize,
    pub builds: usize,
    /// Shell-pair x chunk blocks of the last build.
    pub pair_chunks: u64,
}

/// The exact part of the dSCF partition, shared by every grid-K builder (this CPU one and a
/// device's): the SAD reference D0 of one spin channel (atom-block-diagonal, the same at every
/// geometry), its exact exchange K\[D0\] of an operator (four-center integrals, once per
/// geometry and operator), and the compensated K and exact energy terms of a build.
#[derive(Clone, Debug)]
pub struct GridKReference {
    d0: Mat<f64>,
    thresh: f64,
    k0: Option<(Operator, Mat<f64>)>,
    /// Seconds of the last K\[D0\] (computed here, or where [`GridKReference::set_k0`] says).
    pub t_k0: f64,
}

impl GridKReference {
    /// The reference D0 `d0`, its exchange screened at `thresh` ([`GridKConfig::exact_thresh`]).
    pub fn new(d0: Mat<f64>, thresh: f64) -> Self {
        GridKReference {
            d0,
            thresh,
            k0: None,
            t_k0: 0.0,
        }
    }

    pub fn d0(&self) -> &Mat<f64> {
        &self.d0
    }

    /// K\[D0\] of the operator `op` on `basis`, exact (four-center integrals screened at
    /// `thresh`): what [`GridKReference::k0`] computes, for a caller that computes it elsewhere
    /// (a thread beside a device's setup).
    pub fn exact_exchange(
        basis: &BasisSet,
        d0: &Mat<f64>,
        op: Operator,
        thresh: f64,
        par: Par,
    ) -> Result<Mat<f64>, LinusError> {
        let mut jk = FastJk::new(basis, thresh).with_k_operator(op);
        let mut k = jk
            .compute(&JkRequest::new(std::slice::from_ref(d0), false, true), par)?
            .k;
        k.pop()
            .ok_or_else(|| LinusError::Internal("K[D0]: the exact K of one density".into()))
    }

    /// K\[D0\] of `op` on `basis` (computed on its first use for the operator).
    pub fn k0(
        &mut self,
        basis: &BasisSet,
        op: Operator,
        par: Par,
    ) -> Result<&Mat<f64>, LinusError> {
        let k = match self.k0.take() {
            Some((o, k)) if o == op => k,
            _ => {
                let t = std::time::Instant::now();
                let k = Self::exact_exchange(basis, &self.d0, op, self.thresh, par)?;
                self.t_k0 = t.elapsed().as_secs_f64();
                k
            }
        };
        Ok(&self.k0.insert((op, k)).1)
    }

    /// The K\[D0\] last computed (of the last operator [`GridKReference::k0`] was asked for).
    pub fn current_k0(&self) -> Option<&Mat<f64>> {
        self.k0.as_ref().map(|k| &k.1)
    }

    /// K\[D0\] of `op` computed elsewhere ([`GridKReference::exact_exchange`]) in `seconds`.
    pub fn set_k0(&mut self, op: Operator, k0: Mat<f64>, seconds: f64) {
        self.k0 = Some((op, k0));
        self.t_k0 = seconds;
    }

    /// K = K\[D0\] + Kbar and the exact terms of E_K, 1/2 D0 . K\[D0\] + D_d . K\[D0\], of the
    /// difference density `dd` and its grid exchange `kbar` (symmetric), with `k0` =
    /// [`GridKReference::k0`].
    pub fn compensate(&self, k0: &Mat<f64>, dd: &Mat<f64>, kbar: &Mat<f64>) -> (Mat<f64>, f64) {
        (
            k0 + kbar,
            0.5 * dot(self.d0.as_ref(), k0.as_ref()) + dot(dd.as_ref(), k0.as_ref()),
        )
    }
}

/// The grid-K exchange builder (K only; with a J builder through [`crate::CompositeJk`]).
pub struct GridK {
    pub(crate) basis: BasisSet,
    pub(crate) grid: Arc<MolecularGrid>,
    pub(crate) cfg: GridKConfig,
    /// The dressed collocation Xbar = X M^T (npts x nbf; X without the fit).
    pub(crate) xbar: Mat<f64>,
    pub(crate) fit: Option<OverlapFit>,
    pub(crate) pairs: Vec<PotPair>,
    /// The dSCF partition's exact part (None without it).
    pub(crate) reference: Option<GridKReference>,
    pub stats: GridKStats,
}

impl GridK {
    /// The builder on `grid`; `d0` the SAD density of one spin channel (needed with dSCF).
    pub fn new(
        basis: &BasisSet,
        grid: impl Into<Arc<MolecularGrid>>,
        d0: Option<Mat<f64>>,
        cfg: GridKConfig,
        par: Par,
    ) -> Result<Self, String> {
        let grid = grid.into();
        let reference = match (cfg.dscf, d0) {
            (true, Some(d0)) => Some(GridKReference::new(d0, cfg.exact_thresh)),
            (true, None) => {
                return Err("grid-K with the dSCF partition needs the SAD densities".into())
            }
            (false, _) => None,
        };
        let n = basis.nbf;
        let x = collocation(basis, &grid);
        let mut s = Mat::<f64>::zeros(n, n);
        OneEPlan::new(basis, DerivOrder::Zero).overlap(&mut ao_view_mut(s.as_mut()));
        let (fit, xbar) = if cfg.fit {
            let f = OverlapFit::new(s.as_ref(), x.as_ref(), &grid.w, cfg.lindep, par)?;
            let xb = mm(x.as_ref(), f.m.transpose(), par);
            (Some(f), xb)
        } else {
            (None, x)
        };
        let pairs = PotPair::significant(basis, cfg.prim_cut);
        let stats = GridKStats {
            npoints: grid.len(),
            npairs: pairs.len(),
            ..Default::default()
        };
        Ok(GridK {
            basis: basis.clone(),
            grid,
            cfg,
            xbar,
            fit,
            pairs,
            reference,
            stats,
        })
    }

    pub fn fit(&self) -> Option<&OverlapFit> {
        self.fit.as_ref()
    }

    /// The difference densities D - D0 (D without dSCF).
    pub(crate) fn delta(&self, d: &Mat<f64>) -> Mat<f64> {
        match &self.reference {
            Some(r) => d - r.d0(),
            None => d.clone(),
        }
    }

    /// K and E_K of the densities `dens` (one spin channel each).
    pub fn build(&mut self, dens: &[Mat<f64>], par: Par) -> Result<GridKBuild, LinusError> {
        let k0 = match self.reference.as_mut() {
            Some(r) => Some(r.k0(&self.basis, self.cfg.xop, par)?.clone()),
            None => None,
        };
        let mut out = GridKBuild::default();
        let mut chunks = 0;
        for d in dens {
            let dd = self.delta(d);
            let y = mm(self.xbar.as_ref(), dd.as_ref(), par);
            let (z, nc) = self.potential(y.as_ref());
            chunks += nc;
            let w = &self.grid.w;
            let wz = Mat::from_fn(z.nrows(), z.ncols(), |g, k| w[g] * z[(g, k)]);
            let mut kbar = mm(self.xbar.transpose(), wz.as_ref(), par);
            sym(&mut kbar);
            let e_grid = 0.5 * dot(y.as_ref(), wz.as_ref());
            let (k, e_exact) = match (&self.reference, &k0) {
                (Some(r), Some(k0)) => r.compensate(k0, &dd, &kbar),
                _ => (kbar, 0.0),
            };
            out.k.push(k);
            out.energy.push(e_exact + e_grid);
            out.e_exact.push(e_exact);
            out.e_grid.push(e_grid);
        }
        self.stats.builds += 1;
        self.stats.pair_chunks = chunks;
        Ok(out)
    }

    /// Zbar = (V^A Ybar_A)_A for Ybar (npts x nbf), and the pair x chunk blocks computed.
    pub(crate) fn potential(&self, y: MatRef<'_, f64>) -> (Mat<f64>, u64) {
        let grid = &*self.grid;
        let n = self.basis.nbf;
        let chunks = grid.sub_batches(CHUNK);
        let parts: Vec<(Vec<f64>, u64)> = chunks
            .par_iter()
            .map(|c| self.potential_chunk(c, y))
            .collect();
        let mut z = Mat::<f64>::zeros(y.nrows(), n);
        let mut cnt = 0;
        for (c, (g, k)) in chunks.iter().zip(parts) {
            let ld = ld_of(c.len());
            for f in 0..n {
                for p in 0..c.len() {
                    z[(c.start + p, f)] = g[f * ld + p];
                }
            }
            cnt += k;
        }
        (z, cnt)
    }

    /// G (nbf rows of ld points) of one chunk of points.
    fn potential_chunk(&self, c: &Batch, y: MatRef<'_, f64>) -> (Vec<f64>, u64) {
        let grid = &*self.grid;
        let n = self.basis.nbf;
        let np = c.len();
        let ld = ld_of(np);
        let (px, py, pz) = (
            &grid.x[c.start..c.end],
            &grid.y[c.start..c.end],
            &grid.z[c.start..c.end],
        );
        let mut f = vec![0.0; n * ld];
        for k in 0..n {
            for p in 0..np {
                f[k * ld + p] = y[(c.start + p, k)];
            }
        }
        // max |Ybar| per shell over the chunk, and max_p sum_A w_A |Xbar_Ap| of the chunk
        let shells = &self.basis.shells;
        let fmax: Vec<f64> = shells
            .iter()
            .map(|sh| {
                (sh.ao_start..sh.ao_start + sh.nfunc())
                    .flat_map(|k| f[k * ld..k * ld + np].iter())
                    .fold(0.0f64, |m, v| m.max(v.abs()))
            })
            .collect();
        let xw = (0..n)
            .map(|k| {
                (0..np)
                    .map(|p| grid.w[c.start + p] * self.xbar[(c.start + p, k)].abs())
                    .sum::<f64>()
            })
            .fold(0.0f64, f64::max);
        let mut g = vec![0.0; n * ld];
        let mut prims: Vec<PotPrim> = Vec::new();
        let mut blocks = 0;
        let op = &self.cfg.xop;
        for pair in &self.pairs {
            let fm = fmax[pair.sa].max(fmax[pair.sb]) * xw;
            if fm * pair.est < self.cfg.thresh {
                continue;
            }
            let (a0, b0) = (shells[pair.sa].ao_start, shells[pair.sb].ao_start);
            let mut pc = PotContract {
                f: &f,
                g: &mut g,
                nd: 1,
                dstride: n * ld,
                ld,
                a0,
                b0,
                do_a: true,
                do_b: pair.sa != pair.sb,
                cart: false,
            };
            if pair.is_md() {
                pair.contract(op, px, py, pz, 0.0, &mut pc);
            } else {
                prims.clear();
                let cut = self.cfg.thresh / fm;
                let e = pair.screen_prims(c.center, c.radius, cut, &mut prims);
                if e * fm < self.cfg.thresh || prims.is_empty() {
                    continue;
                }
                pair.contract_prims(&prims, op, px, py, pz, &mut pc);
            }
            blocks += 1;
        }
        (g, blocks)
    }
}

/// Host bytes of a grid-K builder on `npts` points for `nbf` functions at its peak: an SCF's
/// builds (Xbar, and per density Ybar, Zbar with its chunks' parts and W Zbar: 4 npts x nbf
/// matrices; the setup's X, X A and W X A are 3) or, with `gradient`, the gradient's (Xbar,
/// the collocation and its derivatives (4) with their chunks' parts while they are assembled
/// (4), then L, and per density Ybar, Zbar and its parts, W Zbar, Lbar and Lbar M: 11).
pub fn host_bytes(npts: usize, nbf: usize, gradient: bool) -> usize {
    let matrices = if gradient { 11 } else { 4 };
    8 * matrices * npts * nbf
}

/// Row stride of a chunk's point blocks (a multiple of 8).
pub(crate) fn ld_of(np: usize) -> usize {
    np.div_ceil(8) * 8
}

/// The basis functions (and with `order` 1 their gradients) on the grid: `order` 0 returns
/// \[X\], order 1 [X, dX/dx, dX/dy, dX/dz] (npts x nbf each).
pub(crate) fn collocation_deriv(
    basis: &BasisSet,
    grid: &MolecularGrid,
    order: usize,
) -> Vec<Mat<f64>> {
    let n = basis.nbf;
    let nc = if order == 0 { 1 } else { 4 };
    let shells: Vec<usize> = (0..basis.shells.len()).collect();
    let chunks = grid.sub_batches(CHUNK);
    let parts: Vec<Vec<f64>> = chunks
        .par_iter()
        .map(|c| {
            let np = c.len();
            let mut out = vec![0.0; nc * n * np];
            eval_ao(
                basis,
                &shells,
                &grid.x[c.start..c.end],
                &grid.y[c.start..c.end],
                &grid.z[c.start..c.end],
                order,
                &mut out,
            );
            out
        })
        .collect();
    let mut xs: Vec<Mat<f64>> = (0..nc).map(|_| Mat::zeros(grid.len(), n)).collect();
    for (c, v) in chunks.iter().zip(parts) {
        let np = c.len();
        for (d, x) in xs.iter_mut().enumerate() {
            for k in 0..n {
                for p in 0..np {
                    x[(c.start + p, k)] = v[(d * n + k) * np + p];
                }
            }
        }
    }
    xs
}

/// The collocation matrix X (npts x nbf).
pub(crate) fn collocation(basis: &BasisSet, grid: &MolecularGrid) -> Mat<f64> {
    collocation_deriv(basis, grid, 0).remove(0)
}

impl JkBuilder for GridK {
    fn name(&self) -> String {
        match (self.cfg.dscf, self.cfg.fit) {
            (true, true) => "compensated grid-K".into(),
            (true, false) => "grid-K (dSCF)".into(),
            (false, true) => "grid-K (overlap fit)".into(),
            (false, false) => "grid-K".into(),
        }
    }

    fn compute(
        &mut self,
        req: &JkRequest,
        par: Par,
    ) -> Result<JkOutput, linus_base::error::LinusError> {
        if !req.want_k {
            return Ok(JkOutput::default());
        }
        let b = self.build(req.dens, par)?;
        Ok(JkOutput {
            j: vec![],
            k: b.k,
            k_energy: Some(b.energy),
        })
    }

    /// K\[D0\] follows the operator ([`GridKReference::k0`]).
    fn set_exchange_operator(&mut self, op: Operator) -> bool {
        self.cfg.xop = op;
        true
    }
}

#[cfg(test)]
mod tests;
