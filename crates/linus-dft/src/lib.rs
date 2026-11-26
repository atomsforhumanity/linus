//! linus-dft: the exchange-correlation quadrature. The functional resolved once
//! ([`XcFunctional`]), the one factory ([`XcIntegrator::build`]) and the integrator: energy,
//! potential and nuclear gradient (here), second-derivative terms (`hessian`), CPKS kernels
//! (`kernel`), each with VV10 ([`vv10`], [`vv10_resp`]) composed at every derivative order
//! (`engine`). It implements linus-base's XC terms (`XcPotential`, `ResponseKernel`); the
//! functionals themselves are linus-xc's.
//!
//! # The quadrature
//!
//! Batch engine (per grid batch, in parallel over batches):
//! 1. collocation of the basis functions that reach the batch ([`Collocator`]: SIMD over
//!    points, shells screened against the batch's bounding box with a rigorous bound,
//!    negligible primitives skipped, functions below `func_thresh` dropped), stored
//!    function-major `ao[(mu * ncomp + c) * ld + g]`;
//! 2. the density sub-block of the kept functions (gathered by contiguous runs), X = phi D and
//!    rho, grad rho, tau from dense products;
//! 3. the functional (libxc) on the batch;
//! 4. V_mn = sum_g w [ vrho phi_m phi_n + (2 vsigma grad rho) . grad(phi_m phi_n) +
//!    1/2 vtau grad phi_m . grad phi_n ] as M = phi^T W, V += M + M^T
//!    (polarized: the alpha potential uses 2 vsigma_aa grad rho_a + vsigma_ab grad rho_b).
//!
//! Reductions are order independent (per-batch energies summed in batch order, matrices in
//! fixed point, see `linus_num::det`), so results do not depend on the number of threads.
//! The elementwise loops are compiled for AVX-512 and AVX2 as well as the baseline and chosen
//! at run time; they do not use FMA contraction, so every ISA gives the same bits for them.

pub(crate) mod batch;
pub mod engine;
pub mod hessian;
pub(crate) mod kernel;
pub(crate) mod points;
pub mod profile;
#[cfg(test)]
mod reference;
pub mod vv10;
pub mod vv10_resp;

pub use hessian::XcHessianTerms;

use crate::batch::{BatchCtx, BatchPlan};
use faer::{Accum, Mat, MatMut, MatRef, Par};
use linus_base::error::LinusError;
use linus_base::linalg::{matmul, matmul_part, Whole};
use linus_base::terms::{XcDensity, XcPotential};
use linus_core::timing::Timers;
use linus_grid::{GridSpec, MolecularGrid};
use linus_ints::operator::Operator;
use linus_ints::points::ao::shell_extent;
use linus_ints::points::colloc::Collocator;
use linus_ints::BasisSet;
use linus_num::det::{merge_all, packed_col, FixedAcc};
use linus_xc::{Family, Functional, XcInput, XcOutput};
use rayon::prelude::*;
use std::sync::{Arc, Mutex, OnceLock};

/// An XC functional resolved once for a calculation, for closed shells (unpolarized) and open
/// shells (polarized); shared by every integrator of the calculation.
#[derive(Clone, Debug)]
pub struct XcFunctional {
    unpolarized: Arc<Functional>,
    polarized: Arc<Functional>,
}

impl XcFunctional {
    /// The functional `name` (a linus alias, libxc name or id, or a mixture).
    pub fn new(name: &str) -> Result<Self, linus_xc::XcError> {
        Ok(XcFunctional {
            unpolarized: Arc::new(Functional::new(name, false)?),
            polarized: Arc::new(Functional::new(name, true)?),
        })
    }

    /// The unpolarized functional (its exact exchange, VV10, family and derivative order are
    /// those of both).
    pub fn functional(&self) -> &Functional {
        &self.unpolarized
    }

    /// A meta-GGA (its energy depends on the kinetic-energy density).
    pub fn is_meta_gga(&self) -> bool {
        let f = self.functional();
        f.family() == Family::Mgga && f.needs_tau()
    }

    /// The exact-exchange operator of the functional (None: semilocal): a global hybrid's
    /// fraction a_x as `Operator::Attenuated { coulomb: a_x, long_range: 0, omega: 0 }`, a
    /// range-separated hybrid's alpha/r + beta erfc(omega r)/r as [`Operator::from_libxc`].
    pub fn exchange(&self) -> Option<Operator> {
        let f = self.functional();
        match f.range_separation() {
            Some((alpha, beta, omega)) => Some(Operator::from_libxc(alpha, beta, omega)),
            None if f.is_hybrid() => Some(Operator::Attenuated {
                coulomb: f.exx_fraction(),
                long_range: 0.0,
                omega: 0.0,
            }),
            None => None,
        }
    }
}

pub struct XcIntegrator {
    pub basis: BasisSet,
    pub grid: MolecularGrid,
    pub(crate) func_u: Arc<Functional>,
    pub(crate) func_p: Arc<Functional>,
    pub(crate) extents: Vec<f64>,
    /// Basis functions whose value and gradient stay below this in a batch are skipped.
    pub func_thresh: f64,
    info: OnceLock<Vec<BatchInfo>>,
    groups: OnceLock<Vec<BatchGroup>>,
    colls: Mutex<Vec<(usize, u64, Arc<Collocator>)>>,
    /// VV10 nonlocal correlation of -V functionals ([`crate::vv10`]), on its own grid; added to
    /// the energy and potential by [`XcIntegrator::potential`].
    pub(crate) vv10: Option<Box<crate::vv10::Vv10>>,
    /// Memory (MB, all threads) for the per-thread accumulators of the XC-kernel contractions
    /// (`dft_kernel`; default 1024, from `memory_gib` in a session).
    pub kernel_budget_mb: usize,
}

/// What a calculation's XC integrator is built with ([`XcIntegrator::build`]).
pub struct XcSetup<'a> {
    pub functional: &'a XcFunctional,
    /// The XC grid.
    pub grid: GridSpec,
    /// The NLC grid of a VV10 functional (None: no VV10 term).
    pub nlc_grid: Option<GridSpec>,
    /// Memory (MB, all threads) for the XC-kernel accumulators of responses.
    pub kernel_budget_mb: usize,
    /// The grids' partition weights on a device (the host's but for terms below 2^-100 of a
    /// point's sum; `linus_grid::DevicePartition`, which records where they ran), else None.
    pub partition: Option<Box<dyn linus_grid::DevicePartition + 'a>>,
}

impl XcIntegrator {
    /// The XC integrator of a calculation, the one factory of the SCF, gradients, Hessians and
    /// polarizabilities: the grids of `setup` for the atoms `z` at `positions` (timed as "grid"
    /// and "nlc_grid"; with a device partition also the XC grid's "grid_partition", its partition
    /// weights (the device's, or the host's where the device declined), and "grid_batches", the
    /// cutoff and batching), the functional and its VV10.
    pub fn build(
        basis: &BasisSet,
        z: &[u32],
        positions: &[[f64; 3]],
        setup: &XcSetup<'_>,
        timers: &mut Timers,
    ) -> Result<Self, LinusError> {
        let part = setup.partition.as_deref();
        let (grid, gt) = timers.time("grid", || {
            MolecularGrid::build_timed(z, positions, &setup.grid, part)
        })?;
        if part.is_some() {
            timers.add("grid_partition", gt.partition);
            timers.add("grid_batches", gt.batches);
        }
        let nlc = match &setup.nlc_grid {
            Some(spec) if setup.functional.functional().vv10().is_some() => {
                Some(timers.time("nlc_grid", || {
                    MolecularGrid::build_with(z, positions, spec, part)
                })?)
            }
            _ => None,
        };
        let mut xc = Self::with_functional(basis, grid, setup.functional, nlc);
        xc.kernel_budget_mb = setup.kernel_budget_mb;
        Ok(xc)
    }
}

/// Candidate shells (bounding sphere within their extent) and bounding box of a batch.
pub(crate) struct BatchInfo {
    pub(crate) shells: Vec<usize>,
    pub(crate) lo: [f64; 3],
    pub(crate) hi: [f64; 3],
    /// number of candidate functions
    pub(crate) nfunc: usize,
}

/// Consecutive batches (a spatially compact subtree of the k-d batching) processed by one task
/// with group-local dense blocks: the density sub-block D[funcs, funcs] is gathered once per
/// group and the potential is summed into a group-local matrix that goes into the global
/// fixed-point accumulator once per group. Groups depend only on the grid, so results do not
/// depend on the number of threads.
pub(crate) struct BatchGroup {
    pub(crate) batches: std::ops::Range<usize>,
    /// union of the candidate functions of the group's batches (ascending global indices)
    pub(crate) funcs: Vec<usize>,
    /// contiguous runs of `funcs`: (local start, global start, length)
    pub(crate) runs: Vec<(usize, usize, usize)>,
}

/// Functions kept by the gradient's screening (no screening before the batch engine; this
/// only drops functions whose value and first and second derivatives are below it everywhere
/// in a batch).
pub(crate) const GRAD_THRESH: f64 = 1e-15;

impl XcIntegrator {
    /// An integrator of the functional `functional` (a name) on `grid`, with its VV10 (if any)
    /// on the same grid (tests and studies; calculations build theirs with [`XcIntegrator::build`]).
    pub fn new(basis: &BasisSet, grid: MolecularGrid, functional: &str) -> Result<Self, String> {
        let f = XcFunctional::new(functional).map_err(|e| e.to_string())?;
        if f.functional().needs_lapl() {
            return Err(format!(
                "{functional}: Laplacian-dependent meta-GGAs are not implemented yet"
            ));
        }
        let nlc = f.functional().vv10().map(|_| grid.clone());
        Ok(Self::with_functional(basis, grid, &f, nlc))
    }

    /// An integrator of `functional` on `grid`; a VV10 functional evaluates its nonlocal
    /// correlation on `nlc_grid` (None: no VV10 term, also for -V functionals).
    pub fn with_functional(
        basis: &BasisSet,
        grid: MolecularGrid,
        functional: &XcFunctional,
        nlc_grid: Option<MolecularGrid>,
    ) -> Self {
        let mut xc = Self::engine(
            basis,
            grid,
            functional.unpolarized.clone(),
            functional.polarized.clone(),
        );
        xc.vv10 = match (functional.functional().vv10(), nlc_grid) {
            (Some(p), Some(g)) => Some(Box::new(crate::vv10::Vv10::new(&xc, g, p))),
            _ => None,
        };
        xc
    }

    /// The batch engine on `grid` without VV10 (also the VV10 engine on the NLC grid).
    pub(crate) fn engine(
        basis: &BasisSet,
        grid: MolecularGrid,
        func_u: Arc<Functional>,
        func_p: Arc<Functional>,
    ) -> Self {
        let extents = basis
            .shells
            .iter()
            .map(|s| shell_extent(s, 1e-12))
            .collect();
        XcIntegrator {
            basis: basis.clone(),
            grid,
            func_u,
            func_p,
            extents,
            func_thresh: 1e-11,
            info: OnceLock::new(),
            groups: OnceLock::new(),
            colls: Mutex::new(Vec::new()),
            vv10: None,
            kernel_budget_mb: 1024,
        }
    }

    pub fn functional(&self) -> &Functional {
        &self.func_u
    }

    /// The functional of `NCH` spin channels: unpolarized for one (a closed shell's total
    /// density), polarized for two (the spin densities).
    pub(crate) fn channel_functional<const NCH: usize>(&self) -> &Functional {
        match NCH {
            1 => &self.func_u,
            2 => &self.func_p,
            _ => unreachable!("{NCH} spin channels"),
        }
    }

    /// Candidate shells and bounding boxes of all batches (computed once). Not `get_or_init`: the
    /// build runs rayon, and the Hessian terms' per-batch loops ask for it per batch
    /// ([`linus_base::get_or_build`]).
    pub(crate) fn batch_info(&self) -> &[BatchInfo] {
        linus_base::get_or_build(&self.info, || {
            let (basis, extents, grid) = (&self.basis, &self.extents, &self.grid);
            grid.batches
                .par_iter()
                .map(|b| {
                    let shells = select_shells(basis, extents, b.center, b.radius);
                    let (lo, hi) = b.bbox(grid);
                    let nfunc = shells.iter().map(|&s| basis.shells[s].nfunc()).sum();
                    BatchInfo {
                        shells,
                        lo,
                        hi,
                        nfunc,
                    }
                })
                .collect()
        })
    }

    /// Batch groups (see [`BatchGroup`]): up to 16 consecutive batches, at least 64 groups
    /// when the grid has enough batches (computed once).
    pub(crate) fn batch_groups(&self) -> &[BatchGroup] {
        linus_base::get_or_build(&self.groups, || {
            let info = self.batch_info();
            let nb = info.len();
            let gsize = (nb / 64).clamp(1, 16);
            let basis = &self.basis;
            (0..nb.div_ceil(gsize))
                .into_par_iter()
                .map(|g| {
                    let r = g * gsize..((g + 1) * gsize).min(nb);
                    let mut shells: Vec<usize> = info[r.clone()]
                        .iter()
                        .flat_map(|b| b.shells.iter().copied())
                        .collect();
                    shells.sort_unstable();
                    shells.dedup();
                    let mut funcs = Vec::new();
                    for &sh in &shells {
                        let x = &basis.shells[sh];
                        funcs.extend(x.ao_start..x.ao_start + x.nfunc());
                    }
                    let mut runs = Vec::new();
                    linus_num::det::index_runs(&funcs, &mut runs);
                    BatchGroup {
                        batches: r,
                        funcs,
                        runs,
                    }
                })
                .collect()
        })
    }

    /// Collocation engine for derivatives up to `order` with screening threshold `thresh`.
    pub(crate) fn collocator(&self, order: usize, thresh: f64) -> Arc<Collocator> {
        let mut v = self.colls.lock().unwrap();
        if let Some(c) = v.iter().find(|c| c.0 == order && c.1 == thresh.to_bits()) {
            return c.2.clone();
        }
        let c = Arc::new(Collocator::new(&self.basis, order, thresh));
        v.push((order, thresh.to_bits(), c.clone()));
        c
    }
}

pub(crate) fn select_shells(basis: &BasisSet, extents: &[f64], c: [f64; 3], r: f64) -> Vec<usize> {
    (0..basis.shells.len())
        .filter(|&s| {
            let a = basis.shells[s].center;
            let d = ((a[0] - c[0]).powi(2) + (a[1] - c[1]).powi(2) + (a[2] - c[2]).powi(2)).sqrt();
            d < extents[s] + r
        })
        .collect()
}

/// Per-thread scratch of the function-major passes (the basis functions themselves are the batch
/// engine's, [`batch`]).
#[derive(Default)]
pub(crate) struct Scratch {
    pub(crate) dsub: Vec<f64>,
    pub(crate) xm: Vec<f64>,
    pub(crate) ym: Vec<f64>,
    pub(crate) rho: Vec<f64>,
    pub(crate) grad: Vec<f64>,
    pub(crate) tau: Vec<f64>,
    pub(crate) rin: Vec<f64>,
    pub(crate) sin: Vec<f64>,
    pub(crate) tin: Vec<f64>,
    pub(crate) out: XcOutput,
    pub(crate) wm: Vec<f64>,
    pub(crate) vsub: Vec<f64>,
    pub(crate) b1: Vec<f64>,
    pub(crate) b2: Vec<f64>,
    pub(crate) b3: Vec<f64>,
    pub(crate) b4: Vec<f64>,
    pub(crate) b5: Vec<f64>,
    /// group-local density blocks (per spin) and potential accumulators
    pub(crate) dloc: Vec<f64>,
    pub(crate) vloc: Vec<f64>,
}

/// D[funcs, funcs] (dense, column-major) of each density into `dloc` (nspin blocks).
pub(crate) fn gather_group(dens: &[Mat<f64>], funcs: &[usize], dloc: &mut Vec<f64>) {
    let u = funcs.len();
    let d = sized(dloc, dens.len() * u * u);
    for (sp, m) in dens.iter().enumerate() {
        gather_sub(m, funcs, &[], &mut d[sp * u * u..(sp + 1) * u * u]);
    }
}

/// Lower triangle (rows >= column) of a batch block m (nao x nao) added to a group-local
/// dense matrix (u x u) at local indices (runs `lruns` of `lidx`).
#[inline(always)]
fn add_lower_local_body(
    vloc: &mut [f64],
    u: usize,
    lidx: &[usize],
    lruns: &[(usize, usize, usize)],
    m: &[f64],
) {
    let nao = lidx.len();
    for j in 0..nao {
        let col = &mut vloc[lidx[j] * u..(lidx[j] + 1) * u];
        let mc = &m[j * nao..(j + 1) * nao];
        for &(ls, gs, len) in lruns {
            let end = ls + len;
            if end <= j {
                continue;
            }
            let i0 = ls.max(j);
            let g0 = gs + (i0 - ls);
            for (o, &x) in col[g0..g0 + (end - i0)].iter_mut().zip(&mc[i0..end]) {
                *o += x;
            }
        }
    }
}
multiversion!(fn add_lower_local(vloc: &mut [f64], u: usize, lidx: &[usize], lruns: &[(usize, usize, usize)], m: &[f64]) -> () = add_lower_local_body);

/// Resize to n without clearing (contents unspecified).
#[inline]
pub(crate) fn sized(v: &mut Vec<f64>, n: usize) -> &mut [f64] {
    if v.len() < n {
        v.resize(n, 0.0);
    }
    &mut v[..n]
}

/// Column-major matrix view with column stride `cs`.
#[inline]
pub(crate) fn mref(v: &[f64], r: usize, c: usize, cs: usize) -> MatRef<'_, f64> {
    if c == 0 || r == 0 {
        return MatRef::from_column_major_slice(&v[..0], r, c);
    }
    MatRef::from_column_major_slice_with_stride(&v[..(c - 1) * cs + r], r, c, cs)
}

#[inline]
pub(crate) fn mmut(v: &mut [f64], r: usize, c: usize, cs: usize) -> MatMut<'_, f64> {
    if c == 0 || r == 0 {
        return MatMut::from_column_major_slice_mut(&mut v[..0], r, c);
    }
    MatMut::from_column_major_slice_with_stride_mut(&mut v[..(c - 1) * cs + r], r, c, cs)
}

/// c (+)= alpha a b (single-threaded) for column-major views.
#[inline]
pub(crate) fn gemm(
    c: MatMut<'_, f64>,
    accum: bool,
    a: MatRef<'_, f64>,
    b: MatRef<'_, f64>,
    alpha: f64,
) {
    matmul(
        c,
        if accum { Accum::Add } else { Accum::Replace },
        a,
        b,
        alpha,
        Par::Seq,
    );
}

/// [`gemm`] for a part of a whole product of `whole` rows and columns, with its kernel
/// (`linus_base::linalg::matmul_part`): blocks of perturbations whose sizes follow a group.
#[inline]
pub(crate) fn gemm_part(
    whole: Whole,
    c: MatMut<'_, f64>,
    accum: bool,
    a: MatRef<'_, f64>,
    b: MatRef<'_, f64>,
    alpha: f64,
) {
    matmul_part(
        whole,
        c,
        if accum { Accum::Add } else { Accum::Replace },
        a,
        b,
        alpha,
        Par::Seq,
    );
}

/// Sum of a slice with eight fixed partial sums (deterministic for any ISA, vectorizable).
#[inline(always)]
pub(crate) fn sum8(a: &[f64]) -> f64 {
    let mut acc = [0.0f64; 8];
    let (c, r) = a.as_chunks::<8>();
    for x in c {
        for l in 0..8 {
            acc[l] += x[l];
        }
    }
    let mut s = ((acc[0] + acc[1]) + (acc[2] + acc[3])) + ((acc[4] + acc[5]) + (acc[6] + acc[7]));
    for x in r {
        s += x;
    }
    s
}

/// Multiversioned elementwise loops ([`linus_num::multiversion!`], dispatched on
/// [`linus_num::simd::Isa::detect`]).
pub(crate) use linus_num::multiversion;

/// dsub (nao x nao, column-major) = D[idx, idx] for a column-major matrix `d` with column
/// stride `cs`.
#[inline(always)]
fn gather_sub_body(d: &[f64], cs: usize, idx: &[usize], dsub: &mut [f64]) {
    let nao = idx.len();
    for j in 0..nao {
        let col = &d[idx[j] * cs..];
        let dc = &mut dsub[j * nao..(j + 1) * nao];
        for (o, &i) in dc.iter_mut().zip(idx) {
            *o = col[i];
        }
    }
}
multiversion!(fn gather_sub_flat(d: &[f64], cs: usize, idx: &[usize], dsub: &mut [f64]) -> () = gather_sub_body);

/// dst[c * idx.len() + m] = src[c * ld + idx[m]] for every column c of a column-major matrix
/// (rows `idx` of all columns).
#[inline(always)]
fn gather_rows_body(src: &[f64], ld: usize, ncols: usize, idx: &[usize], dst: &mut [f64]) {
    let n = idx.len();
    for c in 0..ncols {
        let col = &src[c * ld..];
        for (o, &i) in dst[c * n..(c + 1) * n].iter_mut().zip(idx) {
            *o = col[i];
        }
    }
}
multiversion!(fn gather_rows_flat(src: &[f64], ld: usize, ncols: usize, idx: &[usize], dst: &mut [f64]) -> () = gather_rows_body);

/// Rows `idx` of a column-major matrix into `dst` (idx.len() x ncols, column-major).
pub(crate) fn gather_rows(m: MatRef<'_, f64>, idx: &[usize], dst: &mut [f64]) {
    let (nr, ncol) = (m.nrows(), m.ncols());
    if ncol == 0 || nr == 0 {
        return;
    }
    let cs = m.col_stride();
    assert!(m.row_stride() == 1 && cs >= nr as isize);
    let cs = cs as usize;
    // SAFETY: column-major with unit row stride and column stride cs >= nrows: the columns
    // lie in the (ncols - 1) cs + nrows values from the base pointer
    let flat = unsafe { std::slice::from_raw_parts(m.as_ptr(), (ncol - 1) * cs + nr) };
    gather_rows_flat(flat, cs, ncol, idx, dst)
}

/// dsub (nao x nao, column-major) = D[idx, idx] for a column-major nbf x nbf matrix `d`.
#[inline]
pub(crate) fn gather_sub(
    d: &Mat<f64>,
    idx: &[usize],
    _runs: &[(usize, usize, usize)],
    dsub: &mut [f64],
) {
    let (nr, nc) = (d.nrows(), d.ncols());
    if nr == 0 || nc == 0 {
        return;
    }
    let cs = d.col_stride();
    assert!(d.row_stride() == 1 && cs >= nr as isize);
    let cs = cs as usize;
    // SAFETY: column-major with unit row stride and column stride cs >= nrows: the columns
    // lie in the (ncols - 1) cs + nrows values from the base pointer
    let flat = unsafe { std::slice::from_raw_parts(d.as_ptr(), (nc - 1) * cs + nr) };
    gather_sub_flat(flat, cs, idx, dsub)
}

/// Fixed-point additions of runs: hi/lo[base + gs + r] += src[ls + r] for every run
/// (ls, gs, len) with ls >= `from` (runs starting before `from` are cut there).
#[inline(always)]
fn add_runs_body(
    hi: &mut [i64],
    lo: &mut [i64],
    base: usize,
    from: usize,
    runs: &[(usize, usize, usize)],
    src: &[f64],
) {
    for &(ls, gs, len) in runs {
        let end = ls + len;
        if end <= from {
            continue;
        }
        let i0 = ls.max(from);
        let g0 = base + gs + (i0 - ls);
        let n = end - i0;
        let (h, l, v) = (&mut hi[g0..g0 + n], &mut lo[g0..g0 + n], &src[i0..end]);
        for ((h, l), &x) in h.iter_mut().zip(l.iter_mut()).zip(v) {
            let (a, b) = linus_num::det::split(x);
            *h += a;
            *l += b;
        }
    }
}

/// Lower triangle (rows >= column) of a symmetrized block into a packed accumulator.
#[inline(always)]
fn scatter_lower_body(
    hi: &mut [i64],
    lo: &mut [i64],
    nbf: usize,
    idx: &[usize],
    runs: &[(usize, usize, usize)],
    m: &[f64],
) {
    let nao = idx.len();
    for j in 0..nao {
        let gj = idx[j];
        // packed column of gj, shifted so that global row r sits at base + r
        let base = packed_col(nbf, gj) - gj;
        add_runs_body(hi, lo, base, j, runs, &m[j * nao..(j + 1) * nao]);
    }
}
multiversion!(fn scatter_lower(hi: &mut [i64], lo: &mut [i64], nbf: usize, idx: &[usize], runs: &[(usize, usize, usize)], m: &[f64]) -> () = scatter_lower_body);

/// Columns of a block (nao x ncol, column-major; column c at c * nao) into an accumulator
/// whose column c starts at `col_off[c]`, rows at global indices (by runs).
#[inline(always)]
fn scatter_cols_body(
    hi: &mut [i64],
    lo: &mut [i64],
    col_off: &[usize],
    nao: usize,
    runs: &[(usize, usize, usize)],
    m: &[f64],
) {
    for (c, &off) in col_off.iter().enumerate() {
        add_runs_body(hi, lo, off, 0, runs, &m[c * nao..(c + 1) * nao]);
    }
}
multiversion!(fn scatter_cols_impl(hi: &mut [i64], lo: &mut [i64], col_off: &[usize], nao: usize, runs: &[(usize, usize, usize)], m: &[f64]) -> () = scatter_cols_body);

/// acc[col_off[c] + idx[r]] += m[c * nao + r] (fixed point) for a block of columns.
pub(crate) fn scatter_cols(
    acc: &mut FixedAcc,
    col_off: &[usize],
    nao: usize,
    runs: &[(usize, usize, usize)],
    m: &[f64],
) {
    let (hi, lo) = acc.parts_for(m);
    scatter_cols_impl(hi, lo, col_off, nao, runs, m)
}

/// Lower triangle of m + m^T in place (m nao x nao, column-major; the upper triangle is left
/// as is), in 8 x 8 tiles so that both the columns and the rows are read cache-friendly.
#[inline]
pub(crate) fn symmetrize_lower(m: &mut [f64], nao: usize) {
    const T: usize = 8;
    for jb in (0..nao).step_by(T) {
        let je = (jb + T).min(nao);
        for ib in (jb..nao).step_by(T) {
            let ie = (ib + T).min(nao);
            for j in jb..je {
                for i in ib.max(j)..ie {
                    let u = m[i * nao + j];
                    m[j * nao + i] += u;
                }
            }
        }
    }
}

/// Adds the lower triangle of m + m^T (nao x nao, column-major; m is overwritten) to a packed
/// lower-triangle fixed-point accumulator of the full matrix, at global indices `idx`
/// (ascending).
#[inline]
pub(crate) fn scatter_packed(
    acc: &mut FixedAcc,
    nbf: usize,
    idx: &[usize],
    runs: &[(usize, usize, usize)],
    m: &mut [f64],
) {
    symmetrize_lower(m, idx.len());
    let (hi, lo) = acc.parts_for(m);
    scatter_lower(hi, lo, nbf, idx, runs, m)
}

/// Full symmetric matrix from a packed lower triangle.
pub(crate) fn unpack_lower(v: &[f64], n: usize) -> Mat<f64> {
    let mut m = Mat::<f64>::zeros(n, n);
    for j in 0..n {
        let c = packed_col(n, j);
        for i in j..n {
            let x = v[c + i - j];
            m[(i, j)] = x;
            m[(j, i)] = x;
        }
    }
    m
}

/// Functional settings of one evaluation.
#[derive(Clone, Copy)]
pub(crate) struct Kind {
    pub(crate) gga: bool,
    pub(crate) mgga: bool,
    pub(crate) restricted: bool,
}

impl Kind {
    pub(crate) fn of(func: &Functional, restricted: bool) -> Self {
        let fam = func.family();
        Kind {
            gga: fam >= Family::Gga,
            mgga: fam == Family::Mgga && func.needs_tau(),
            restricted,
        }
    }
    pub(crate) fn nspin(&self) -> usize {
        if self.restricted {
            1
        } else {
            2
        }
    }
}

/// rho, grad rho (if GGA) and tau (if meta-GGA) of one spin density on the batch from
/// X = phi D (and, for tau, Y_k = d_k phi D in `ym`, 3 x ld x nao). ao is function-major
/// with `nc` components.
#[inline(always)]
#[allow(clippy::too_many_arguments, reason = "a kernel's arrays and sizes")]
fn densities_body(
    ao: &[f64],
    nc: usize,
    ld: usize,
    nao: usize,
    gga: bool,
    mgga: bool,
    xm: &[f64],
    ym: &[f64],
    rho: &mut [f64],
    grad: &mut [f64],
    tau: &mut [f64],
) {
    rho.fill(0.0);
    grad.fill(0.0);
    tau.fill(0.0);
    for mu in 0..nao {
        let f = &ao[mu * nc * ld..(mu + 1) * nc * ld];
        let x = &xm[mu * ld..(mu + 1) * ld];
        let p0 = &f[..ld];
        for g in 0..ld {
            rho[g] += p0[g] * x[g];
        }
        if gga {
            let (gx, rest) = grad.split_at_mut(ld);
            let (gy, gz) = rest.split_at_mut(ld);
            let (px, py, pz) = (&f[ld..2 * ld], &f[2 * ld..3 * ld], &f[3 * ld..4 * ld]);
            for g in 0..ld {
                let x2 = 2.0 * x[g];
                gx[g] += px[g] * x2;
                gy[g] += py[g] * x2;
                gz[g] += pz[g] * x2;
            }
        }
        if mgga {
            for k in 0..3 {
                let pk = &f[(1 + k) * ld..(2 + k) * ld];
                let yk = &ym[(k * nao + mu) * ld..(k * nao + mu + 1) * ld];
                for g in 0..ld {
                    tau[g] += 0.5 * pk[g] * yk[g];
                }
            }
        }
    }
}
multiversion!(fn densities(ao: &[f64], nc: usize, ld: usize, nao: usize, gga: bool, mgga: bool, xm: &[f64], ym: &[f64], rho: &mut [f64], grad: &mut [f64], tau: &mut [f64]) -> () = densities_body);

/// W_mu = vr phi_mu + sum_k vg_k d_k phi_mu (ld x nao).
#[inline(always)]
#[allow(clippy::too_many_arguments, reason = "a kernel's arrays and sizes")]
fn wmat_body(
    ao: &[f64],
    nc: usize,
    ld: usize,
    nao: usize,
    gga: bool,
    vr: &[f64],
    vg: &[f64],
    wm: &mut [f64],
) {
    for mu in 0..nao {
        let f = &ao[mu * nc * ld..(mu + 1) * nc * ld];
        let w = &mut wm[mu * ld..(mu + 1) * ld];
        let p0 = &f[..ld];
        if gga {
            let (px, py, pz) = (&f[ld..2 * ld], &f[2 * ld..3 * ld], &f[3 * ld..4 * ld]);
            let (vx, vy, vz) = (&vg[..ld], &vg[ld..2 * ld], &vg[2 * ld..3 * ld]);
            for g in 0..ld {
                w[g] = vr[g] * p0[g] + vx[g] * px[g] + vy[g] * py[g] + vz[g] * pz[g];
            }
        } else {
            for g in 0..ld {
                w[g] = vr[g] * p0[g];
            }
        }
    }
}
multiversion!(fn wmat(ao: &[f64], nc: usize, ld: usize, nao: usize, gga: bool, vr: &[f64], vg: &[f64], wm: &mut [f64]) -> () = wmat_body);

/// W_mu = v d_c phi_mu for component c (ld x nao).
#[inline(always)]
fn wcomp_body(
    ao: &[f64],
    nc: usize,
    ld: usize,
    nao: usize,
    comp: usize,
    v: &[f64],
    wm: &mut [f64],
) {
    for mu in 0..nao {
        let p = &ao[(mu * nc + comp) * ld..(mu * nc + comp + 1) * ld];
        let w = &mut wm[mu * ld..(mu + 1) * ld];
        for g in 0..ld {
            w[g] = v[g] * p[g];
        }
    }
}
multiversion!(fn wcomp(ao: &[f64], nc: usize, ld: usize, nao: usize, comp: usize, v: &[f64], wm: &mut [f64]) -> () = wcomp_body);

/// Per-point weights of one channel's potential-matrix block, zero past the batch's points:
/// `vr` half the weighted v_rho (the symmetrization doubles it), `vg` the weighted GGA vector
/// (3 x ld; read for a GGA), `vt` a quarter of the weighted v_tau (meta-GGA).
pub(crate) struct PointWeights<'a> {
    pub(crate) vr: &'a [f64],
    pub(crate) vg: &'a [f64],
    pub(crate) vt: Option<&'a [f64]>,
}

/// One channel's potential-matrix block of a function-major batch of a group pass, from its
/// per-point weights: V = M + M^T with M = phi^T (vr phi + vg_k d_k phi)
/// (+ sum_k d_k phi^T (vt d_k phi) for a meta-GGA), its lower triangle added to the group-local
/// block `vloc` (u x u). `wm` and `vsub` are scratch.
pub(crate) fn push_vmat(
    b: &BatchCtx<'_>,
    gga: bool,
    pw: &PointWeights<'_>,
    wm: &mut Vec<f64>,
    vsub: &mut Vec<f64>,
    vloc: &mut [f64],
    lap: &mut profile::Lap,
) {
    let (nc, ld, nao) = (b.nc, b.ld, b.nao);
    wmat(
        b.ao,
        nc,
        ld,
        nao,
        gga,
        &pw.vr[..ld],
        &pw.vg[..3 * ld],
        sized(wm, ld * nao),
    );
    lap.lap(profile::WEIGHTS);
    let vs = sized(vsub, nao * nao);
    gemm(
        mmut(vs, nao, nao, nao),
        false,
        mref(b.ao, ld, nao, nc * ld).transpose(),
        mref(wm, ld, nao, ld),
        1.0,
    );
    if let Some(vt) = pw.vt {
        for kk in 0..3 {
            wcomp(b.ao, nc, ld, nao, 1 + kk, &vt[..ld], sized(wm, ld * nao));
            gemm(
                mmut(vs, nao, nao, nao),
                true,
                mref(&b.ao[(1 + kk) * ld..], ld, nao, nc * ld).transpose(),
                mref(wm, ld, nao, ld),
                1.0,
            );
        }
    }
    lap.lap(profile::GEMM_V);
    symmetrize_lower(vs, nao);
    add_lower_local(vloc, b.u, b.lidx, b.lruns, vs);
    lap.lap(profile::SCATTER);
}

/// The batch's density sub-block of spin `sp` into s.dsub, from its group's local dense blocks
/// (`u` x `u` per spin, in Scratch::dloc; the batch's local indices).
pub(crate) fn gather_batch_density(u: usize, sp: usize, b: &BatchCtx<'_>, s: &mut Scratch) {
    let nao = b.nao;
    gather_sub_flat(
        &s.dloc[sp * u * u..(sp + 1) * u * u],
        u,
        b.lidx,
        &mut s.dsub[..nao * nao],
    );
}

/// Ground-state densities of a (function-major) batch for every spin: X_s = phi D_s (in s.xm,
/// nspin x ld x nao), Y_sk = d_k phi D_s (meta-GGA, in s.ym), rho/grad/tau per spin (s.rho:
/// nspin x ld, s.grad: nspin x 3 x ld, s.tau: nspin x ld), and the libxc inputs (s.rin, s.sin,
/// s.tin), with the density sub-blocks from the group's local blocks (`u` x `u` per spin;
/// [`gather_batch_density`]).
pub(crate) fn batch_densities(u: usize, k: Kind, b: &BatchCtx<'_>, s: &mut Scratch) {
    let (nc, np, ld, nao) = (b.nc, b.np, b.ld, b.nao);
    let nspin = k.nspin();
    let ny = if k.mgga { 3 } else { 0 };
    sized(&mut s.dsub, nao * nao);
    sized(&mut s.xm, nspin * ld * nao);
    sized(&mut s.ym, nspin * ny * ld * nao);
    sized(&mut s.rho, nspin * ld);
    sized(&mut s.grad, nspin * 3 * ld);
    sized(&mut s.tau, nspin * ld);
    for sp in 0..nspin {
        gather_batch_density(u, sp, b, s);
        let dm = mref(&s.dsub, nao, nao, nao);
        gemm(
            mmut(&mut s.xm[sp * ld * nao..], ld, nao, ld),
            false,
            mref(b.ao, ld, nao, nc * ld),
            dm,
            1.0,
        );
        for kk in 0..ny {
            let o = (sp * 3 + kk) * ld * nao;
            gemm(
                mmut(&mut s.ym[o..], ld, nao, ld),
                false,
                mref(&b.ao[(1 + kk) * ld..], ld, nao, nc * ld),
                dm,
                1.0,
            );
        }
        let (rho, grad, tau) = (
            &mut s.rho[sp * ld..(sp + 1) * ld],
            &mut s.grad[sp * 3 * ld..(sp + 1) * 3 * ld],
            &mut s.tau[sp * ld..(sp + 1) * ld],
        );
        densities(
            b.ao,
            nc,
            ld,
            nao,
            k.gga,
            k.mgga,
            &s.xm[sp * ld * nao..(sp + 1) * ld * nao],
            &s.ym[sp * ny * ld * nao..(sp + 1) * ny * ld * nao],
            rho,
            grad,
            tau,
        );
    }
    libxc_inputs(k, np, ld, s);
}

/// libxc inputs (s.rin, s.sin, s.tin; np real points) from s.rho, s.grad, s.tau.
pub(crate) fn libxc_inputs(k: Kind, np: usize, ld: usize, s: &mut Scratch) {
    let gr = |sp: usize, kk: usize, g: usize| s.grad[(sp * 3 + kk) * ld + g];
    s.rin.clear();
    s.sin.clear();
    s.tin.clear();
    if k.restricted {
        s.rin.extend_from_slice(&s.rho[..np]);
        if k.gga {
            s.sin.extend((0..np).map(|g| {
                gr(0, 0, g) * gr(0, 0, g) + gr(0, 1, g) * gr(0, 1, g) + gr(0, 2, g) * gr(0, 2, g)
            }));
        }
        if k.mgga {
            s.tin.extend_from_slice(&s.tau[..np]);
        }
    } else {
        for g in 0..np {
            s.rin.push(s.rho[g]);
            s.rin.push(s.rho[ld + g]);
            if k.gga {
                let dot = |a: usize, c: usize| {
                    gr(a, 0, g) * gr(c, 0, g)
                        + gr(a, 1, g) * gr(c, 1, g)
                        + gr(a, 2, g) * gr(c, 2, g)
                };
                s.sin.push(dot(0, 0));
                s.sin.push(dot(0, 1));
                s.sin.push(dot(1, 1));
            }
            if k.mgga {
                s.tin.push(s.tau[g]);
                s.tin.push(s.tau[ld + g]);
            }
        }
    }
}

/// Evaluates the functional to `order` on the batch inputs in `s`.
pub(crate) fn eval_functional(func: &Functional, k: Kind, order: usize, s: &mut Scratch) {
    let input = if k.mgga {
        XcInput::mgga(&s.rin, &s.sin, &s.tin)
    } else if k.gga {
        XcInput::gga(&s.rin, &s.sin)
    } else {
        XcInput::lda(&s.rin)
    };
    func.eval(order, &input, &mut s.out)
        .expect("functional evaluation");
}

/// Per-point potential weights of spin `sp`: vr = half w vrho (ld, zero past np) and, for
/// GGA, vg_k = w (2 vsigma_ss grad_s,k + vsigma_ab grad_o,k) (3 x ld).
#[allow(
    clippy::too_many_arguments,
    reason = "the functional's outputs and the weights' arrays"
)]
pub(crate) fn potential_weights(
    out: &XcOutput,
    k: Kind,
    sp: usize,
    w: &[f64],
    grad: &[f64],
    np: usize,
    ld: usize,
    vr: &mut [f64],
    vg: &mut [f64],
    half: f64,
) {
    let vrho = out.vrho();
    vr.fill(0.0);
    for g in 0..np {
        vr[g] = half
            * w[g]
            * if k.restricted {
                vrho[g]
            } else {
                vrho[2 * g + sp]
            };
    }
    if k.gga {
        vg.fill(0.0);
        let vs = out.vsigma();
        for kk in 0..3 {
            for g in 0..np {
                vg[kk * ld + g] = w[g]
                    * if k.restricted {
                        2.0 * vs[g] * grad[kk * ld + g]
                    } else {
                        2.0 * vs[3 * g + 2 * sp] * grad[(sp * 3 + kk) * ld + g]
                            + vs[3 * g + 1] * grad[((1 - sp) * 3 + kk) * ld + g]
                    };
            }
        }
    }
}

impl XcPotential for XcIntegrator {
    /// From the density matrices (the orbitals are not needed on the CPU).
    fn compute(
        &mut self,
        d: &XcDensity<'_>,
        _par: Par,
    ) -> Result<(f64, Mat<f64>, Mat<f64>), linus_base::error::LinusError> {
        Ok(self.potential(d.da, d.db, d.restricted))
    }

    /// VV10 keeps its last energy and statistics (the report's `vv10`).
    fn keeps_state(&self) -> bool {
        self.vv10().is_some()
    }
}

impl XcIntegrator {
    /// E_xc and the XC potential matrices (alpha, beta) for spin densities D_a, D_b
    /// (restricted: the potential of the total density D_a + D_b, returned twice).
    #[allow(clippy::needless_range_loop, reason = "indexes several arrays in step")]
    pub fn potential(
        &self,
        da: &Mat<f64>,
        db: &Mat<f64>,
        restricted: bool,
    ) -> (f64, Mat<f64>, Mat<f64>) {
        use crate::profile as xp;
        let nbf = self.basis.nbf;
        let func = if restricted {
            &self.func_u
        } else {
            &self.func_p
        };
        let k = Kind::of(func, restricted);
        let order = usize::from(k.gga || k.mgga);
        let nspin = k.nspin();
        let dens: Vec<Mat<f64>> = if restricted {
            vec![da + db]
        } else {
            vec![da.clone(), db.clone()]
        };
        let npack = nbf * (nbf + 1) / 2;
        let init = || {
            let acc: Vec<FixedAcc> = (0..nspin).map(|_| FixedAcc::new(npack)).collect();
            (acc, Scratch::default())
        };
        let (energies, _, works) = self.for_each_group(
            &BatchPlan::new(order, self.func_thresh),
            init,
            // the group's density blocks, and its potential summed group-locally
            |(_, s), grp| {
                let mut lap = xp::Lap::start();
                let u = grp.funcs.len();
                gather_group(&dens, &grp.funcs, &mut s.dloc);
                sized(&mut s.vloc, nspin * u * u).fill(0.0);
                lap.lap(xp::SETUP);
                true
            },
            |(_, s), b| {
                let (np, ld, nao, u, w) = (b.np, b.ld, b.nao, b.u, b.w);
                if nao == 0 {
                    return 0.0;
                }
                let mut lap = xp::Lap::start();
                batch_densities(u, k, b, s);
                lap.lap(xp::DENSITY);
                eval_functional(func, k, 1, s);
                lap.lap(xp::LIBXC);
                // energy: sum_g w rho zk (batch order)
                let zk = s.out.zk();
                let mut e = 0.0;
                for g in 0..np {
                    let rt = if restricted {
                        s.rho[g]
                    } else {
                        s.rho[g] + s.rho[ld + g]
                    };
                    e += w[g] * rt * zk[g];
                }
                for sp in 0..nspin {
                    let (mut vr, mut vg, mut vt) = (
                        std::mem::take(&mut s.b1),
                        std::mem::take(&mut s.b2),
                        std::mem::take(&mut s.b3),
                    );
                    potential_weights(
                        &s.out,
                        k,
                        sp,
                        w,
                        &s.grad,
                        np,
                        ld,
                        sized(&mut vr, ld),
                        sized(&mut vg, 3 * ld),
                        0.5,
                    );
                    if k.mgga {
                        // 1/4 here and the symmetrization give 1/2 vtau grad phi . grad phi
                        let vtau = s.out.vtau();
                        let vt = sized(&mut vt, ld);
                        vt.fill(0.0);
                        for g in 0..np {
                            vt[g] = 0.25
                                * w[g]
                                * if restricted {
                                    vtau[g]
                                } else {
                                    vtau[2 * g + sp]
                                };
                        }
                    }
                    let pw = PointWeights {
                        vr: &vr,
                        vg: &vg,
                        vt: k.mgga.then_some(&vt[..]),
                    };
                    let vloc = &mut s.vloc[sp * u * u..(sp + 1) * u * u];
                    push_vmat(b, k.gga, &pw, &mut s.wm, &mut s.vsub, vloc, &mut lap);
                    (s.b1, s.b2, s.b3) = (vr, vg, vt);
                }
                e
            },
            // the group's potential into the fixed-point accumulators (lower triangle)
            |(acc, s), grp| {
                let mut lap = xp::Lap::start();
                let u = grp.funcs.len();
                for sp in 0..nspin {
                    let (hi, lo) = acc[sp].parts_for(&s.vloc[sp * u * u..(sp + 1) * u * u]);
                    scatter_lower(
                        hi,
                        lo,
                        nbf,
                        &grp.funcs,
                        &grp.runs,
                        &s.vloc[sp * u * u..(sp + 1) * u * u],
                    );
                    acc[sp].tick(1);
                }
                lap.lap(xp::REDUCE);
            },
        );
        let mut lap = xp::Lap::start();
        let exc: f64 = energies.iter().sum();
        let tot = merge_all(works.into_iter().map(|x| x.0).collect());
        let mats: Vec<Mat<f64>> = match tot {
            Some(t) => t.iter().map(|a| unpack_lower(&a.to_f64(), nbf)).collect(),
            None => vec![Mat::zeros(nbf, nbf); nspin],
        };
        lap.lap(xp::REDUCE);
        let mut va = mats[0].clone();
        let mut vb = if restricted {
            va.clone()
        } else {
            mats[1].clone()
        };
        let mut exc = exc;
        // VV10 nonlocal correlation (crate::xc::vv10): a functional of the total density on its own
        // grid; its potential goes to both spins
        if let Some(nl) = &self.vv10 {
            let (e, v) = nl.energy_potential(&(da + db));
            exc += e;
            va += &v;
            vb += &v;
        }
        (exc, va, vb)
    }

    /// E_xc for fixed spin densities (helper for tests and finite differences).
    pub fn energy(&mut self, da: &Mat<f64>, db: &Mat<f64>, restricted: bool) -> f64 {
        self.potential(da, db, restricted).0
    }
}

/// Gradient contraction of one spin for one batch (see [`XcIntegrator::gradient`]): for every
/// function mu (atom `owner[mu]`) and direction i,
/// c_i = d_i phi_mu B_mu + sum_k d_ik phi_mu vg_k X_mu + 1/2 vt sum_k d_ik phi_mu Y_k,mu,
/// gat[atom][i] -= 2 sum_g c_i, tsum[i][g] -= 2 c_i.
#[inline(always)]
#[allow(clippy::too_many_arguments, reason = "a kernel's arrays and sizes")]
fn grad_contract_body(
    ao: &[f64],
    nc: usize,
    ld: usize,
    nao: usize,
    gga: bool,
    mgga: bool,
    owner: &[usize],
    xm: &[f64],
    bm: &[f64],
    ym: &[f64],
    vg: &[f64],
    vt: &[f64],
    gat: &mut [[f64; 3]],
    tsum: &mut [f64],
    cbuf: &mut [f64],
) {
    // second-derivative component of (i, k): xx xy xz yy yz zz at 4..10
    const D2: [[usize; 3]; 3] = [[4, 5, 6], [5, 7, 8], [6, 8, 9]];
    for mu in 0..nao {
        let f = &ao[mu * nc * ld..(mu + 1) * nc * ld];
        let x = &xm[mu * ld..(mu + 1) * ld];
        let bb = &bm[mu * ld..(mu + 1) * ld];
        let a = owner[mu];
        for i in 0..3 {
            let pi = &f[(1 + i) * ld..(2 + i) * ld];
            let c = &mut cbuf[..ld];
            for g in 0..ld {
                c[g] = pi[g] * bb[g];
            }
            if gga {
                let (h0, h1, h2) = (
                    &f[D2[i][0] * ld..(D2[i][0] + 1) * ld],
                    &f[D2[i][1] * ld..(D2[i][1] + 1) * ld],
                    &f[D2[i][2] * ld..(D2[i][2] + 1) * ld],
                );
                let (v0, v1, v2) = (&vg[..ld], &vg[ld..2 * ld], &vg[2 * ld..3 * ld]);
                for g in 0..ld {
                    c[g] += (h0[g] * v0[g] + h1[g] * v1[g] + h2[g] * v2[g]) * x[g];
                }
                if mgga {
                    let (y0, y1, y2) = (
                        &ym[mu * ld..(mu + 1) * ld],
                        &ym[(nao + mu) * ld..(nao + mu + 1) * ld],
                        &ym[(2 * nao + mu) * ld..(2 * nao + mu + 1) * ld],
                    );
                    for g in 0..ld {
                        c[g] += 0.5 * vt[g] * (h0[g] * y0[g] + h1[g] * y1[g] + h2[g] * y2[g]);
                    }
                }
            }
            gat[a][i] -= 2.0 * sum8(c);
            let ts = &mut tsum[i * ld..(i + 1) * ld];
            for g in 0..ld {
                ts[g] -= 2.0 * c[g];
            }
        }
    }
}
multiversion!(fn grad_contract(ao: &[f64], nc: usize, ld: usize, nao: usize, gga: bool, mgga: bool, owner: &[usize], xm: &[f64], bm: &[f64], ym: &[f64], vg: &[f64], vt: &[f64], gat: &mut [[f64; 3]], tsum: &mut [f64], cbuf: &mut [f64]) -> () = grad_contract_body);

/// One channel of a gradient contraction: the functional's family `k`, the channel `sp`, and its
/// per-point weights, zero past the batch's points: `vr` the weighted v_rho, `vg` the weighted
/// GGA vector (3 x ld; GGA), `vt` the weighted v_tau (ld; meta-GGA).
pub(crate) struct GradChannel<'a> {
    pub(crate) k: Kind,
    pub(crate) sp: usize,
    pub(crate) vr: &'a [f64],
    pub(crate) vg: &'a [f64],
    pub(crate) vt: &'a [f64],
}

/// Where a gradient contraction adds: the atoms' derivatives and the per-point sums of the point
/// motion (3 x ld).
pub(crate) struct GradOut<'a> {
    pub(crate) gat: &'a mut [[f64; 3]],
    pub(crate) tsum: &'a mut [f64],
}

/// One channel's nuclear-gradient contraction of a function-major batch of a group pass (the
/// formulas of [`XcIntegrator::gradient`]): T = vr phi + vg_k d_k phi, B = T D_s (the channel's
/// group-local density block), and the contraction with the batch's X_s (and Y_s for a
/// meta-GGA; from [`batch_densities`]) into `out`. `bm` and `cb` are scratch.
pub(crate) fn push_grad(
    b: &BatchCtx<'_>,
    gw: &GradChannel<'_>,
    s: &mut Scratch,
    [bm, cb]: [&mut Vec<f64>; 2],
    out: GradOut<'_>,
    lap: &mut profile::Lap,
) {
    let (k, sp) = (gw.k, gw.sp);
    let (nc, ld, nao) = (b.nc, b.ld, b.nao);
    wmat(
        b.ao,
        nc,
        ld,
        nao,
        k.gga,
        &gw.vr[..ld],
        &gw.vg[..3 * ld],
        sized(&mut s.wm, ld * nao),
    );
    lap.lap(profile::WEIGHTS);
    sized(&mut s.dsub, nao * nao);
    gather_batch_density(b.u, sp, b, s);
    gemm(
        mmut(sized(bm, ld * nao), ld, nao, ld),
        false,
        mref(&s.wm, ld, nao, ld),
        mref(&s.dsub, nao, nao, nao),
        1.0,
    );
    lap.lap(profile::GEMM_D);
    let ny = if k.mgga { 3 } else { 0 };
    grad_contract(
        b.ao,
        nc,
        ld,
        nao,
        k.gga,
        k.mgga,
        b.owner,
        &s.xm[sp * ld * nao..(sp + 1) * ld * nao],
        &bm[..ld * nao],
        &s.ym[sp * ny * ld * nao..(sp + 1) * ny * ld * nao],
        &gw.vg[..3 * ld],
        gw.vt,
        out.gat,
        out.tsum,
        sized(cb, ld),
    );
    lap.lap(profile::OTHER);
}

impl XcIntegrator {
    /// Nuclear gradient of E_xc at fixed AO density matrices (natom x 3), VV10 included (on its
    /// own grid; its time is in [`crate::vv10::Vv10Stats`]). With `grid_response`, the
    /// grids move with the atoms (weight derivatives and point motion), which makes the result
    /// the exact derivative of the quadrature energy.
    ///
    /// Per spin, with X = phi D, B = T D, T_mu = w vrho phi_mu + sum_k vg_k d_k phi_mu
    /// (vg_k = w (2 vsigma grad rho)_k), the derivative for atom A is
    /// -2 sum_{mu in A} sum_g [d_i phi_mu B_mu + sum_k d_ik phi_mu vg_k X_mu
    /// (+ 1/2 w vtau sum_k d_ik phi_mu (d_k phi D)_mu)].
    #[allow(clippy::type_complexity, reason = "a one-off tuple of buffers")]
    pub fn gradient(
        &self,
        da: &Mat<f64>,
        db: &Mat<f64>,
        restricted: bool,
        grid_response: bool,
    ) -> Vec<[f64; 3]> {
        use crate::profile as xp;
        let grid = &self.grid;
        let natom = grid.atom_z.len();
        let func = if restricted {
            &self.func_u
        } else {
            &self.func_p
        };
        let k = Kind::of(func, restricted);
        let nspin = k.nspin();
        let dens: Vec<Mat<f64>> = if restricted {
            vec![da + db]
        } else {
            vec![da.clone(), db.clone()]
        };
        // per batch: atom contributions (point motion folded in) and the energy density
        let (parts, _, _) = self.for_each_group(
            &BatchPlan::new(2, GRAD_THRESH),
            Scratch::default,
            |s, grp| {
                gather_group(&dens, &grp.funcs, &mut s.dloc);
                true
            },
            |s, b| -> (Vec<[f64; 3]>, Vec<f64>) {
                let (np, ld, nao, u, w) = (b.np, b.ld, b.nao, b.u, b.w);
                let mut gat = vec![[0.0; 3]; natom];
                if nao == 0 {
                    return (gat, vec![0.0; np]);
                }
                let mut lap = xp::Lap::start();
                batch_densities(u, k, b, s);
                lap.lap(xp::DENSITY);
                eval_functional(func, k, 1, s);
                lap.lap(xp::LIBXC);
                let mut tsum = vec![0.0; 3 * ld];
                for sp in 0..nspin {
                    let (mut vr, mut vg, mut vt, mut bm, mut cb) = (
                        std::mem::take(&mut s.b1),
                        std::mem::take(&mut s.b2),
                        std::mem::take(&mut s.b3),
                        std::mem::take(&mut s.b4),
                        std::mem::take(&mut s.b5),
                    );
                    // T = w vrho phi + vg . grad phi;  B = T D
                    potential_weights(
                        &s.out,
                        k,
                        sp,
                        w,
                        &s.grad,
                        np,
                        ld,
                        sized(&mut vr, ld),
                        sized(&mut vg, 3 * ld),
                        1.0,
                    );
                    {
                        let vt = sized(&mut vt, ld);
                        vt.fill(0.0);
                        if k.mgga {
                            let vtau = s.out.vtau();
                            for g in 0..np {
                                vt[g] = w[g]
                                    * if restricted {
                                        vtau[g]
                                    } else {
                                        vtau[2 * g + sp]
                                    };
                            }
                        }
                    }
                    let gw = GradChannel {
                        k,
                        sp,
                        vr: &vr,
                        vg: &vg,
                        vt: &vt[..ld],
                    };
                    let out = GradOut {
                        gat: &mut gat,
                        tsum: &mut tsum,
                    };
                    push_grad(b, &gw, s, [&mut bm, &mut cb], out, &mut lap);
                    (s.b1, s.b2, s.b3, s.b4, s.b5) = (vr, vg, vt, bm, cb);
                }
                // point motion: each point's derivative goes to its owner; energy density
                let zk = s.out.zk();
                let mut eps = vec![0.0; np];
                for g in 0..np {
                    let rt: f64 = (0..nspin).map(|sp| s.rho[sp * ld + g]).sum();
                    eps[g] = rt * zk[g];
                    if grid_response {
                        let own = grid.atom[b.start + g] as usize;
                        for i in 0..3 {
                            gat[own][i] -= tsum[i * ld + g];
                        }
                    }
                }
                lap.lap(xp::OTHER);
                (gat, eps)
            },
            |_, _| (),
        );
        let mut lap = xp::Lap::start();
        let mut gtot = vec![[0.0; 3]; natom];
        for (gat, _) in &parts {
            for a in 0..natom {
                for i in 0..3 {
                    gtot[a][i] += gat[a][i];
                }
            }
        }
        lap.lap(xp::REDUCE);
        if grid_response {
            let mut eps = Vec::with_capacity(grid.len());
            for (_, e) in &parts {
                eps.extend_from_slice(e);
            }
            let gw = grid.weight_gradient_contract(&eps);
            for a in 0..natom {
                for i in 0..3 {
                    gtot[a][i] += gw[a][i];
                }
            }
        }
        lap.lap(xp::GRID_RESP);
        // VV10 nonlocal correlation (crate::xc::vv10): the total density on its own grid
        if let Some(nl) = &self.vv10 {
            let gv = nl.gradient(&(da + db), grid_response);
            for (a, b) in gtot.iter_mut().zip(&gv) {
                for c in 0..3 {
                    a[c] += b[c];
                }
            }
        }
        gtot
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use linus_grid::GridSpec;
    use linus_ints::basis::BasisDef;

    pub(crate) fn setup(func: &str, spec: GridSpec) -> (XcIntegrator, Mat<f64>, Mat<f64>) {
        let z = vec![8, 1, 1, 6, 1, 1, 1];
        let xyz = vec![
            [0.02, -0.01, 0.22],
            [0.05, 1.43, -0.89],
            [-0.07, -1.40, -0.93],
            [2.9, 0.1, 1.4],
            [3.1, 2.0, 2.1],
            [4.4, -0.6, 0.2],
            [2.6, -1.1, 3.0],
        ];
        let def = BasisDef::load("def2-svp", None).unwrap();
        let basis = BasisSet::new(&def, &z, &xyz, true).unwrap();
        let (da, db) = linus_scf::guess::sad_guess(&def, &basis, &z, 9, 9, Par::Seq);
        // break the spin symmetry a little for the open-shell checks
        let n = basis.nbf;
        let db = &db
            + Mat::from_fn(n, n, |i, j| {
                1e-3 * (((i + j) % 5) as f64 - 2.0) * (-((i as f64 - j as f64).abs()) / 4.0).exp()
            });
        let g = MolecularGrid::build(&z, &xyz, &spec).unwrap();
        (XcIntegrator::new(&basis, g, func).unwrap(), da, db)
    }

    pub(crate) fn rel(a: &Mat<f64>, b: &Mat<f64>) -> f64 {
        let mut e: f64 = 0.0;
        let mut m: f64 = 0.0;
        for j in 0..a.ncols() {
            for i in 0..a.nrows() {
                e = e.max((a[(i, j)] - b[(i, j)]).abs());
                m = m.max(b[(i, j)].abs());
            }
        }
        e / m
    }

    /// The reference open-shell kernel contraction (which `kernel`'s tests check the production
    /// kernel against) against central differences of the potential, and the same bits for 1
    /// and 4 threads.
    #[test]
    fn open_shell_kernel() {
        let (xc, da, db) = setup("pbe", GridSpec::pyscf(1));
        let n = da.nrows();
        let d1a = Mat::from_fn(n, n, |i, j| {
            1e-2 * ((((i + j) * 7) % 11) as f64 - 5.0) / 5.0
                * (-((i as f64 - j as f64).abs()) / 6.0).exp()
        });
        let d1b = Mat::from_fn(n, n, |i, j| {
            1e-2 * ((((i * j) * 3 + i + j) % 7) as f64 - 3.0) / 3.0
                * (-((i as f64 - j as f64).abs()) / 5.0).exp()
        });
        let run = |t: usize| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(t)
                .build()
                .unwrap()
                .install(|| xc.kernel_contract_ref(&da, &db, false, &[(d1a.clone(), d1b.clone())]))
        };
        let k1 = run(1);
        assert!(run(4) == k1, "open-shell kernel differs with 4 threads");
        let h = 1e-4;
        let at = |sg: f64| xc.potential(&(&da + &d1a * sg), &(&db + &d1b * sg), false);
        let (p, m) = (at(h), at(-h));
        let fa = (&p.1 - &m.1) * (0.5 / h);
        let fb = (&p.2 - &m.2) * (0.5 / h);
        assert!(
            rel(&k1[0].0, &fa) < 1e-6 && rel(&k1[0].1, &fb) < 1e-6,
            "open-shell kernel vs FD: {:e} {:e}",
            rel(&k1[0].0, &fa),
            rel(&k1[0].1, &fb)
        );
    }

    /// Energy, potential and gradient against the reference implementation, restricted and
    /// unrestricted, LDA/GGA/meta-GGA, Becke (ORCA-like) and SSF (linus) grids; and the same
    /// bits for 1, 2 and 5 threads.
    #[test]
    fn matches_reference_and_deterministic() {
        for (func, spec) in [
            ("svwn", GridSpec::pyscf(1)),
            ("b3lyp", GridSpec::orca_defgrid_guess(1)),
            ("pbe", GridSpec::linus(1)),
            ("r2scan", GridSpec::pyscf(1)),
        ] {
            let (mut xc, da, db) = setup(func, spec);
            for restricted in [true, false] {
                let (dab, dbb) = if restricted {
                    (da.clone(), da.clone())
                } else {
                    (da.clone(), db.clone())
                };
                let (e0, va0, vb0) = xc.compute_ref(&dab, &dbb, restricted);
                let g0 = xc.gradient_ref(&dab, &dbb, restricted, true);
                let run = |t: usize| {
                    rayon::ThreadPoolBuilder::new()
                        .num_threads(t)
                        .build()
                        .unwrap()
                        .install(|| {
                            (
                                xc.potential(&dab, &dbb, restricted),
                                xc.gradient(&dab, &dbb, restricted, true),
                            )
                        })
                };
                let ((e1, va1, vb1), g1) = run(2);
                assert!(
                    (e1 - e0).abs() <= 1e-12 * e0.abs(),
                    "{func} r={restricted}: E {e1} vs {e0}"
                );
                assert!(
                    rel(&va1, &va0) <= 1e-12 && rel(&vb1, &vb0) <= 1e-12,
                    "{func} r={restricted}: V {:e} {:e}",
                    rel(&va1, &va0),
                    rel(&vb1, &vb0)
                );
                let gm = g0.iter().flatten().fold(0.0f64, |m, v| m.max(v.abs()));
                let ge = g0
                    .iter()
                    .flatten()
                    .zip(g1.iter().flatten())
                    .fold(0.0f64, |m, (a, b)| m.max((a - b).abs()));
                // rounding of two summation orders, amplified by cancellation: the largest
                // seen is r2SCAN unrestricted, 8.8e-13 (x86-64: generic, AVX2, AVX-512) and
                // 1.02e-12 (aarch64, macOS) of the largest component
                assert!(
                    ge <= 5e-12 * gm,
                    "{func} r={restricted}: gradient {ge:e} (max {gm:e})"
                );
                for t in [1, 5] {
                    let ((e2, va2, vb2), g2) = run(t);
                    assert_eq!(
                        e2.to_bits(),
                        e1.to_bits(),
                        "{func}: energy differs with {t} threads"
                    );
                    assert!(
                        va2 == va1 && vb2 == vb1,
                        "{func}: potential differs with {t} threads"
                    );
                    assert_eq!(g2, g1, "{func}: gradient differs with {t} threads");
                }
            }
            let _ = &mut xc;
        }
    }
}
