//! COSX-style seminumerical exchange (chain-of-spheres: Neese, Wennmohs, Hansen & Becker
//! 2009; overlap fitting: Izsák & Neese 2011; batch shell lists and integral kernels after
//! Helmich-Paris, de Souza, Neese & Izsák 2021):
//!
//!   K'_mn = sum_g X_mg G_ng,   X_mg = w_g phi_m(g),
//!   G_ng  = sum_s A_ns(g) F_sg,  F_sg = sum_l D_sl phi_l(g),  A_ns(g) = <n| O(|r - r_g|) |s>,
//!
//! K = sym(K') or, with overlap fitting, sym(S S_num^-1 K'). Electron 1 is integrated on the
//! grid, electron 2 analytically (Rys quadrature kernels in [`linus_ints::points::grid_pot`]).
//!
//! Per batch of points (64-128, set by [`linus_grid::GridSpec::cosx`]) three shell lists are
//! built, as in ORCA: shells whose functions reach the batch (X, and the rows of K' the batch
//! contributes to), shells with a non-negligible F in the batch, and shell pairs whose
//! integrals, bounded by the pair's primitive estimates, matter for the F and X of the batch;
//! primitive pairs are screened per batch the same way (with their decay with the distance to
//! the batch). F and K' are dense products (GEMM).
//!
//! Inside a batch F and G hold the Cartesian components of the shells (F = (D T)^T phi with
//! the spherical transform T, nbf x ncart), so the integral kernels never transform per shell
//! pair and chunk of points; K' is accumulated as nbf x ncart and transformed once per build.
//!
//! Results do not depend on the number of threads: each thread accumulates the batch blocks
//! of K' in fixed-point integers (exact, order-independent addition, vectorized over runs of
//! consecutive columns) that are converted to double precision once at the end.
//!
//! The final-grid energy ([`final_grid_exchange`]) needs no K: [`CosxK::energy`] sums
//! w_g Ft(g) . G(g) per batch (Ft = (M^T D)^T phi), with each shell pair screened by the
//! bound of its contribution to the energy (sn-LinK's energy criterion, Laqua, Kussmann &
//! Ochsenfeld 2020) at [`CosxConfig::thresh_e`] = 1e-14: on ts61 def2-TZVP 12% faster than
//! the K build at an energy difference of 7e-10 Eh (1e-12 is 25% faster but off by 1e-7: the
//! dropped contributions all have one sign, so their bounds must be summed, not compared).
//!
//! Thresholds: K builds 1e-11 on the last SCF grid; 1e-9 on the grids before it
//! ([`CosxConfig::thresh_coarse`]: their K only steers the first iterations, and the switch
//! to the next grid starts with a full build), which changes the final energy by < 1e-9 Eh.
//!
//! Several densities (UHF, response densities) share the integrals. The operator can be
//! a / r + b erf(omega r) / r (range-separated hybrids, libxc convention via
//! [`Operator::from_libxc`]). Incremental builds (K\[D\] = K\[D_prev\] + K[D - D_prev], as in direct
//! SCF) are optional. Their difference densities are screened at a threshold that shrinks with
//! max |D - D_prev| below [`CosxConfig::delta_ref`]: with the full threshold, a small
//! difference density loses all its contributions below the threshold, each step adds an error
//! of the size of a full build's whole screening error (ts113 wB97X/def2-TZVP: 5e-8 in max |F_ai|
//! per step), the errors add up until the next full build, and the SCF stalls there (between
//! 5e-7 and 1e-7 on ts113, at 2-5e-9 on u1an39 wB97X/def2-SVP). Scaled with the difference
//! density, the error per step stays proportional to the step.
//!
//! Multi-grid SCF builders ([`scf_builder`]) move to the next grid when the SCF asks
//! ([`JkBuilder::next_approximation`], at max |F_ai| < `ScfConfig::grid_switch`); the SCF then
//! carries its DIIS subspace over to the new grid.

pub mod gradient;
pub mod rij_cosx;

use faer::{Accum, Mat, MatRef, Par};
use linus_base::linalg::mat_to_rowmajor;
use linus_base::terms::{JkBuilder, JkOutput, JkRequest};
use linus_grid::{Batch, GridSpec, MolecularGrid};
use linus_ints::operator::Operator;
use linus_ints::points::ao::{eval_ao, shell_extent};
use linus_ints::points::colloc::Collocator;
use linus_ints::points::grid_pot::{PotContract, PotPair, PotPrim};
use linus_ints::BasisSet;
use linus_num::det::FixedAcc;
use rayon::prelude::*;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

#[derive(Clone, Debug)]
pub struct CosxConfig {
    /// Screening threshold on contributions to K (shell pairs and primitive pairs per batch).
    pub thresh: f64,
    /// The threshold on all grids but the last of a multi-grid SCF ([`CosxK::with_next_grid`]):
    /// their K only steers the first iterations, and the switch to the next grid starts with a
    /// full build at `thresh`, so their screening errors do not reach the converged density.
    pub thresh_coarse: f64,
    /// Basis-function value below which a shell does not reach a batch.
    pub bf_cut: f64,
    /// The exact-exchange operator (Coulomb by default).
    pub xop: Operator,
    /// Incremental builds K\[D\] = K\[D_prev\] + K[D - D_prev].
    pub incremental: bool,
    /// Full rebuild after this many incremental builds.
    pub reset_every: usize,
    /// Screening of the difference-density builds of incremental K: threshold times
    /// clamp(max |D - D_prev| / delta_ref, delta_scale, 1).
    pub delta_scale: f64,
    pub delta_ref: f64,
    /// With several grids: move to the next grid once max |D - D_previous| < switch_dmax
    /// (0: only when the SCF asks, [`JkBuilder::next_approximation`]).
    pub switch_dmax: f64,
    /// Scale primitive-pair estimates by their decay with the distance to each batch.
    pub dist_screen: bool,
    /// Time every shell pair's integrals by class (la, lb) into [`CosxStats::classes`].
    pub profile: bool,
    /// Screening threshold of energy-only evaluations ([`CosxK::energy`]) on contributions
    /// to the exchange energy (sn-LinK's energy criterion, Laqua, Kussmann & Ochsenfeld 2020).
    pub thresh_e: f64,
}

impl Default for CosxConfig {
    fn default() -> Self {
        CosxConfig {
            thresh: 1e-11,
            thresh_coarse: 1e-9,
            bf_cut: 1e-11,
            xop: Operator::Coulomb,
            incremental: false,
            reset_every: 8,
            delta_scale: 1.0,
            delta_ref: 0.0,
            switch_dmax: 1e-3,
            dist_screen: true,
            profile: false,
            thresh_e: 1e-14,
        }
    }
}

/// Counters of the last build.
#[derive(Clone, Debug, Default)]
pub struct CosxStats {
    pub builds: usize,
    /// Grid used by the last build and points per build summed over all builds.
    pub grid: usize,
    pub points_total: usize,
    pub incremental_builds: usize,
    /// Wall time of the last build and its parts (summed over threads for the parts).
    pub time: f64,
    pub t_ints: f64,
    pub t_gemm: f64,
    pub t_other: f64,
    /// Shell-pair x batch integral blocks and primitive-pair x point evaluations computed.
    pub pair_batches: u64,
    pub prim_points: u64,
    /// Profiled builds ([`CosxConfig::profile`]): per class (larger l, smaller l) of shell
    /// pairs, the pair x batch blocks, primitive-pair x point evaluations and thread-seconds.
    pub classes: Vec<ClassStat>,
}

/// Counters of one shell-pair class in a profiled build.
#[derive(Clone, Debug, Default)]
pub struct ClassStat {
    pub l1: usize,
    pub l2: usize,
    pub pair_batches: u64,
    pub prim_points: u64,
    pub seconds: f64,
}

/// Per-class profile counters, indexed by l1 * 5 + l2 (l1 >= l2).
pub(crate) struct ClassProfile([[AtomicU64; 3]; 25]);

impl ClassProfile {
    pub(crate) fn new() -> Self {
        ClassProfile(std::array::from_fn(|_| {
            std::array::from_fn(|_| AtomicU64::new(0))
        }))
    }

    pub(crate) fn add(&self, local: &[[u64; 3]; 25]) {
        for (a, l) in self.0.iter().zip(local) {
            if l[0] != 0 {
                for k in 0..3 {
                    a[k].fetch_add(l[k], Ordering::Relaxed);
                }
            }
        }
    }

    pub(crate) fn stats(&self) -> Vec<ClassStat> {
        let mut v = vec![];
        for (i, a) in self.0.iter().enumerate() {
            let n = a[0].load(Ordering::Relaxed);
            if n > 0 {
                v.push(ClassStat {
                    l1: i / 5,
                    l2: i % 5,
                    pair_batches: n,
                    prim_points: a[1].load(Ordering::Relaxed),
                    seconds: a[2].load(Ordering::Relaxed) as f64 * 1e-9,
                });
            }
        }
        v
    }
}

#[inline]
pub(crate) fn class_index(la: usize, lb: usize) -> usize {
    la.max(lb) * 5 + la.min(lb)
}

/// Seminumerical exchange builder (K only; combine with a J builder via
/// [`crate::CompositeJk`]).
pub struct CosxK {
    pub(crate) basis: BasisSet,
    pub(crate) pairs: Vec<PotPair>,
    /// Shells and estimate of each pair, compact (16 bytes): every batch tests every pair,
    /// and the full [`PotPair`]s (hundreds of bytes each) are only read for those that pass.
    pub(crate) pair_scr: Vec<PairScreen>,
    pub(crate) extents: Vec<f64>,
    /// Cartesian layout of F and G inside the batches: first Cartesian row of each shell,
    /// the number of rows, and the spherical transform T (nbf x ncart, G_n = sum_c T_nc Gc_c,
    /// Fc_c = sum_n T_nc F_n; None when the basis has no spherical d or higher shells).
    pub(crate) cart_off: Vec<usize>,
    pub(crate) ncart: usize,
    pub(crate) tsph: Option<Mat<f64>>,
    /// Grids in order of use (e.g. a small one for the first SCF iterations).
    pub(crate) grids: Vec<Arc<MolecularGrid>>,
    active: usize,
    pub opts: CosxConfig,
    /// Overlap-fitting matrices M = S S_num^-1 per grid.
    fits: Vec<Option<Mat<f64>>>,
    /// Densities and K of the last build (incremental builds).
    #[allow(clippy::type_complexity, reason = "a one-off tuple of buffers")]
    prev: Option<(Vec<Mat<f64>>, Vec<Mat<f64>>)>,
    last_dens: Option<Vec<Mat<f64>>>,
    since_reset: usize,
    /// The last build was incremental ([`JkBuilder::built_incrementally`]).
    last_incremental: bool,
    pub stats: CosxStats,
}

/// Maximal runs of consecutive values in an index list (shared with the XC code).
use linus_num::det::index_runs;

/// Per-thread fixed-point accumulators of `n` values ([`FixedAcc`], one per pool thread) and
/// their exact, order-independent sum.
pub(crate) struct ThreadAccs {
    accs: linus_num::det::ThreadAccs<FixedAcc>,
    n: usize,
}

impl ThreadAccs {
    pub(crate) fn new(n: usize) -> Self {
        ThreadAccs {
            accs: linus_num::det::ThreadAccs::new(),
            n,
        }
    }

    pub(crate) fn with<R>(&self, f: impl FnOnce(&mut FixedAcc) -> R) -> R {
        self.accs.with(|| FixedAcc::new(self.n), f)
    }

    pub(crate) fn total(self) -> Vec<f64> {
        let n = self.n;
        match linus_num::det::merge_all(self.accs.into_vec().into_iter().map(|a| vec![a]).collect())
        {
            Some(t) => t[0].to_f64(),
            None => vec![0.0; n],
        }
    }
}

/// First Cartesian row of each shell, the number of Cartesian rows and the spherical
/// transform T (nbf x ncart; None if it is the identity).
pub(crate) fn cart_layout(basis: &BasisSet) -> (Vec<usize>, usize, Option<Mat<f64>>) {
    let mut off = Vec::with_capacity(basis.shells.len());
    let mut nc = 0;
    for sh in &basis.shells {
        off.push(nc);
        nc += linus_ints::basis::ncart(sh.l);
    }
    let mut t = Mat::<f64>::zeros(basis.nbf, nc);
    let mut identity = nc == basis.nbf;
    for (s, sh) in basis.shells.iter().enumerate() {
        let ncs = linus_ints::basis::ncart(sh.l);
        let ts = BasisSet::shell_transform(sh);
        for j in 0..sh.nfunc() {
            for i in 0..ncs {
                let v = ts[j * ncs + i];
                t[(sh.ao_start + j, off[s] + i)] = v;
                if v != if sh.ao_start + j == off[s] + i {
                    1.0
                } else {
                    0.0
                } {
                    identity = false;
                }
            }
        }
    }
    (off, nc, if identity { None } else { Some(t) })
}

/// Row-major nbf x nbf matrix times T: row-major nbf x ncart (a copy without T).
pub(crate) fn times_t(a: &[f64], nbf: usize, t: Option<&Mat<f64>>, par: Par) -> Vec<f64> {
    match t {
        None => a.to_vec(),
        Some(t) => {
            let am = MatRef::from_row_major_slice(a, nbf, nbf);
            let r = linus_base::linalg::mm(am, t.as_ref(), par);
            mat_to_rowmajor(r.as_ref())
        }
    }
}

/// Row-major nbf x ncart matrix times T^T: row-major nbf x nbf.
pub(crate) fn times_tt(a: &[f64], nbf: usize, t: Option<&Mat<f64>>, par: Par) -> Vec<f64> {
    match t {
        None => a.to_vec(),
        Some(t) => {
            let am = MatRef::from_row_major_slice(a, nbf, t.ncols());
            let r = linus_base::linalg::mm(am, t.transpose(), par);
            mat_to_rowmajor(r.as_ref())
        }
    }
}

/// The data of a shell pair needed for the first screening test.
#[derive(Clone, Copy, Debug)]
pub(crate) struct PairScreen {
    pub sa: u32,
    pub sb: u32,
    pub est: f64,
}

/// Per-thread scratch.
#[derive(Default)]
struct Scratch {
    near: Vec<usize>,
    xidx: Vec<usize>,
    phi: Vec<f64>,
    xw: Vec<f64>,
    dxt: Vec<f64>,
    f: Vec<f64>,
    g: Vec<f64>,
    fmax: Vec<f64>,
    ftw: Vec<f64>,
    touched: Vec<bool>,
    tidx: Vec<usize>,
    runs: Vec<(usize, usize, usize)>,
    gc: Vec<f64>,
    kblk: Vec<f64>,
    xp: Vec<f64>,
    yp: Vec<f64>,
    zp: Vec<f64>,
    prims: Vec<PotPrim>,
}

/// Per-thread buffers of [`CosxK::numerical_overlap`].
#[derive(Default)]
struct SnumScratch {
    near: Vec<usize>,
    xp: Vec<f64>,
    yp: Vec<f64>,
    zp: Vec<f64>,
    cbuf: Vec<f64>,
    xidx: Vec<usize>,
    xatom: Vec<usize>,
    xw: Vec<f64>,
    kblk: Vec<f64>,
    runs: Vec<(usize, usize, usize)>,
}

struct Counters {
    t_ints: AtomicU64,
    t_gemm: AtomicU64,
    t_other: AtomicU64,
    pair_batches: AtomicU64,
    prim_points: AtomicU64,
    classes: Option<ClassProfile>,
}

impl CosxK {
    /// Builder on `grid` with the significant shell pairs of `basis`.
    pub fn new(basis: &BasisSet, grid: impl Into<Arc<MolecularGrid>>, opts: CosxConfig) -> Self {
        let ns = basis.shells.len();
        let extents = basis
            .shells
            .iter()
            .map(|s| shell_extent(s, opts.bf_cut))
            .collect();
        // primitive pairs that cannot matter at any F and X are dropped here
        let pair_list: Vec<(usize, usize)> =
            (0..ns).flat_map(|i| (0..=i).map(move |j| (i, j))).collect();
        let cut = opts.thresh * 1e-6;
        let pairs: Vec<PotPair> = pair_list
            .par_iter()
            .map(|&(i, j)| {
                let mut p = PotPair::new(basis, i, j, 1e-30);
                let keep = p.nprim_above(cut);
                p.prims.truncate(keep);
                p.est = p.prims.iter().map(|q| q.est).sum();
                p
            })
            .filter(|p| !p.prims.is_empty())
            .collect();
        let (cart_off, ncart, tsph) = cart_layout(basis);
        CosxK {
            basis: basis.clone(),
            pair_scr: pairs
                .iter()
                .map(|p| PairScreen {
                    sa: p.sa as u32,
                    sb: p.sb as u32,
                    est: p.est,
                })
                .collect(),
            pairs,
            extents,
            cart_off,
            ncart,
            tsph,
            grids: vec![grid.into()],
            active: 0,
            opts,
            fits: vec![None],
            prev: None,
            last_dens: None,
            since_reset: 0,
            last_incremental: false,
            stats: CosxStats::default(),
        }
    }

    /// Adds a grid used after the previous ones (see [`CosxConfig::switch_dmax`]).
    pub fn with_next_grid(mut self, grid: impl Into<Arc<MolecularGrid>>) -> Self {
        self.grids.push(grid.into());
        self.fits.push(None);
        self
    }

    /// Points of the grid in use.
    pub fn npoints(&self) -> usize {
        self.grids[self.active].len()
    }

    pub fn grid_points(&self) -> Vec<usize> {
        self.grids.iter().map(|g| g.len()).collect()
    }

    pub fn npairs(&self) -> usize {
        self.pairs.len()
    }

    /// Primitive pairs of the shell pairs kept.
    pub fn nprims(&self) -> usize {
        self.pairs.iter().map(|p| p.prims.len()).sum()
    }

    /// Shells whose functions reach the batch.
    pub(crate) fn near_shells(&self, b: &Batch, out: &mut Vec<usize>) {
        out.clear();
        for (s, sh) in self.basis.shells.iter().enumerate() {
            let c = sh.center;
            let d2 = (c[0] - b.center[0]).powi(2)
                + (c[1] - b.center[1]).powi(2)
                + (c[2] - b.center[2]).powi(2);
            let r = self.extents[s] + b.radius;
            if d2 < r * r {
                out.push(s);
            }
        }
    }

    /// The overlap matrix S_num = sum_g w_g phi phi^T of `grid`, with the SIMD collocation
    /// (`coll` of order 0) and its screening of the functions per batch (values below its
    /// cut dropped); thread-count independent. The one S_num of overlap fitting: the K builds'
    /// fitting matrices ([`Self::with_overlap_fitting`]) and the exchange gradient's
    /// (`gradient::exchange_gradient`), so the gradient differentiates the energy it reports.
    #[allow(clippy::needless_range_loop, reason = "indexes several arrays in step")]
    pub(crate) fn numerical_overlap(&self, grid: &MolecularGrid, coll: &Collocator) -> Mat<f64> {
        let nbf = self.basis.nbf;
        let accs = ThreadAccs::new(nbf * nbf);
        grid.batches
            .par_iter()
            .for_each_init(SnumScratch::default, |scr, b| {
                let np = b.end - b.start;
                let mut near = std::mem::take(&mut scr.near);
                self.near_shells(b, &mut near);
                let ld = np.div_ceil(8) * 8;
                let (x, y, z) = (
                    &grid.x[b.start..b.end],
                    &grid.y[b.start..b.end],
                    &grid.z[b.start..b.end],
                );
                for (dst, src) in [(&mut scr.xp, x), (&mut scr.yp, y), (&mut scr.zp, z)] {
                    dst.clear();
                    dst.extend_from_slice(src);
                    dst.resize(ld, src[np - 1]);
                }
                let (lo, hi) = b.bbox(grid);
                let ncand: usize = near.iter().map(|&s| coll.shells[s].nf).sum();
                scr.cbuf.resize(ncand * ld, 0.0);
                let nx = coll.eval(
                    &near,
                    &scr.xp,
                    &scr.yp,
                    &scr.zp,
                    np,
                    ld,
                    &lo,
                    &hi,
                    &mut scr.cbuf,
                    &mut scr.xidx,
                    &mut scr.xatom,
                );
                scr.near = near;
                if nx == 0 {
                    return;
                }
                let w = &grid.w[b.start..b.end];
                scr.xw.resize(nx * ld, 0.0);
                for i in 0..nx {
                    for g in 0..np {
                        scr.xw[i * ld + g] = scr.cbuf[i * ld + g] * w[g];
                    }
                    scr.xw[i * ld + np..(i + 1) * ld]
                        .iter_mut()
                        .for_each(|v| *v = 0.0);
                }
                scr.kblk.resize(nx * nx, 0.0);
                {
                    let dst =
                        faer::MatMut::from_row_major_slice_mut(&mut scr.kblk[..nx * nx], nx, nx);
                    let a = MatRef::from_row_major_slice(&scr.xw[..nx * ld], nx, ld);
                    let p = MatRef::from_row_major_slice(&scr.cbuf[..nx * ld], nx, ld);
                    linus_base::linalg::matmul(
                        dst,
                        Accum::Replace,
                        a,
                        p.transpose(),
                        1.0,
                        Par::Seq,
                    );
                }
                index_runs(&scr.xidx, &mut scr.runs);
                accs.with(|acc| {
                    acc.add_block(0, nbf, &scr.xidx, &scr.runs, nx, &scr.kblk[..nx * nx]);
                    acc.batch_done();
                });
            });
        let v = accs.total();
        Mat::from_fn(nbf, nbf, |i, j| 0.5 * (v[i * nbf + j] + v[j * nbf + i]))
    }

    /// Overlap fitting (Izsák & Neese 2011) with the analytic overlap `s`: K = S S_num^-1 K'.
    /// Applies to every grid added so far.
    pub fn with_overlap_fitting(mut self, s: MatRef<'_, f64>) -> Self {
        // the parallelism of the current pool: a Cholesky factorization and a solve with nbf
        // right-hand sides do not depend on the degree (util::linalg)
        let par = Par::rayon(0);
        let coll = Collocator::new(&self.basis, 0, self.opts.bf_cut);
        for gi in 0..self.grids.len() {
            let snum = self.numerical_overlap(&self.grids[gi], &coll);
            let llt = linus_base::linalg::cholesky(snum.as_ref(), par)
                .expect("numerical overlap not positive definite (grid too small?)");
            // M = S S_num^-1, M^T = S_num^-1 S
            let mt = linus_base::linalg::cholesky_solve(llt.as_ref(), s, par);
            self.fits[gi] = Some(mt.transpose().to_owned());
        }
        self
    }

    /// K' (row-major, one per density) of the densities `dens` (row-major nbf x nbf).
    fn build_kprime(
        &self,
        gi: usize,
        dflat: &[Vec<f64>],
        scale: f64,
        par: Par,
    ) -> (Vec<Vec<f64>>, Counters) {
        let nbf = self.basis.nbf;
        let nd = dflat.len();
        let nc = self.ncart;
        // F in Cartesian rows: Fc = (D T)^T[:, X] phi
        let dct: Vec<Vec<f64>> = dflat
            .iter()
            .map(|d| times_t(d, nbf, self.tsph.as_ref(), par))
            .collect();
        let accs = ThreadAccs::new(nd * nbf * nc);
        let cnt = Counters {
            t_ints: AtomicU64::new(0),
            t_gemm: AtomicU64::new(0),
            t_other: AtomicU64::new(0),
            pair_batches: AtomicU64::new(0),
            prim_points: AtomicU64::new(0),
            classes: self.opts.profile.then(ClassProfile::new),
        };
        let grid = &self.grids[gi];
        let thresh = scale
            * if gi + 1 < self.grids.len() {
                self.opts.thresh_coarse.max(self.opts.thresh)
            } else {
                self.opts.thresh
            };
        grid.batches
            .par_iter()
            .for_each_init(Scratch::default, |scr, b| {
                self.batch(grid, b, &dct, thresh, scr, &accs, &cnt);
            });
        let v = accs.total();
        // K' = K'c T^T
        (
            (0..nd)
                .map(|k| {
                    times_tt(
                        &v[k * nbf * nc..(k + 1) * nbf * nc],
                        nbf,
                        self.tsph.as_ref(),
                        par,
                    )
                })
                .collect(),
            cnt,
        )
    }

    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::needless_range_loop, reason = "indexes several arrays in step")]
    fn batch(
        &self,
        grid: &MolecularGrid,
        b: &Batch,
        dct: &[Vec<f64>],
        thresh: f64,
        scr: &mut Scratch,
        accs: &ThreadAccs,
        cnt: &Counters,
    ) {
        let t0 = Instant::now();
        let basis = &self.basis;
        let shells = &basis.shells;
        let nbf = basis.nbf;
        let nc = self.ncart;
        let coff = &self.cart_off;
        let dflat = dct;
        let ns = shells.len();
        let nd = dflat.len();
        let np = b.end - b.start;
        let (x, y, z) = (
            &grid.x[b.start..b.end],
            &grid.y[b.start..b.end],
            &grid.z[b.start..b.end],
        );
        let w = &grid.w[b.start..b.end];
        // shells reaching the batch and their functions on it
        let mut near = std::mem::take(&mut scr.near);
        self.near_shells(b, &mut near);
        let nx: usize = near.iter().map(|&s| shells[s].nfunc()).sum();
        if nx == 0 {
            scr.near = near;
            return;
        }
        scr.xidx.clear();
        for &s in &near {
            scr.xidx
                .extend(shells[s].ao_start..shells[s].ao_start + shells[s].nfunc());
        }
        scr.phi.resize(nx * np, 0.0);
        eval_ao(basis, &near, x, y, z, 0, &mut scr.phi);
        scr.xw.resize(nx * np, 0.0);
        let mut xsum: f64 = 0.0;
        for i in 0..nx {
            let mut s = 0.0;
            for g in 0..np {
                let v = scr.phi[i * np + g] * w[g];
                scr.xw[i * np + g] = v;
                s += v.abs();
            }
            xsum = xsum.max(s);
        }
        // F_k = D_k[:, X] phi  (nbf rows of stride ld; zero beyond the np points)
        let t1 = Instant::now();
        let ld = np.div_ceil(8) * 8;
        // coordinates padded to ld with copies of the last point (F is zero there)
        for (dst, src) in [(&mut scr.xp, x), (&mut scr.yp, y), (&mut scr.zp, z)] {
            dst.clear();
            dst.extend_from_slice(src);
            dst.resize(ld, src[np - 1]);
        }
        let ds = nc * ld;
        scr.f.clear();
        scr.f.resize(nd * ds, 0.0);
        scr.dxt.resize(nx * nc, 0.0);
        for (k, d) in dflat.iter().enumerate() {
            for (i, &m) in scr.xidx.iter().enumerate() {
                scr.dxt[i * nc..(i + 1) * nc].copy_from_slice(&d[m * nc..(m + 1) * nc]);
            }
            // (faer 0.22.6's from_row_major_slice_with_stride_mut builds a column-major view)
            let dst = faer::MatMut::from_column_major_slice_with_stride_mut(
                &mut scr.f[k * ds..(k + 1) * ds],
                np,
                nc,
                ld,
            )
            .transpose_mut();
            let lhs = MatRef::from_row_major_slice(&scr.dxt[..nx * nc], nx, nc).transpose();
            let rhs = MatRef::from_row_major_slice(&scr.phi[..nx * np], nx, np);
            linus_base::linalg::matmul(dst, Accum::Replace, lhs, rhs, 1.0, Par::Seq);
        }
        let t2 = Instant::now();
        // max |F| per shell over points and densities
        scr.fmax.clear();
        scr.fmax.resize(ns, 0.0);
        for (s, sh) in shells.iter().enumerate() {
            let mut m: f64 = 0.0;
            let n = linus_ints::basis::ncart(sh.l);
            for k in 0..nd {
                let blk = &scr.f[k * ds + coff[s] * ld..k * ds + (coff[s] + n) * ld];
                m = blk.iter().fold(m, |a, v| a.max(v.abs()));
            }
            scr.fmax[s] = m;
        }
        // G_k = sum over pairs of A F, contracted inside the integral kernels
        scr.g.clear();
        scr.g.resize(nd * ds, 0.0);
        scr.touched.clear();
        scr.touched.resize(ns, false);
        let mut pair_batches = 0u64;
        let mut prim_points = 0u64;
        let lim = thresh / xsum.max(1e-300);
        let mut cls = [[0u64; 3]; 25];
        let profile = cnt.classes.is_some();
        for (ip, ps) in self.pair_scr.iter().enumerate() {
            let (fa, fb) = (scr.fmax[ps.sa as usize], scr.fmax[ps.sb as usize]);
            let fm = fa.max(fb);
            if ps.est * fm < lim {
                continue;
            }
            let p = &self.pairs[ip];
            let cut = lim / fm;
            // distance-aware primitive screening for this batch (Rys pairs)
            let use_dist = self.opts.dist_screen && !p.is_md();
            let est = if use_dist {
                scr.prims.clear();
                p.screen_prims(b.center, b.radius, cut, &mut scr.prims)
            } else {
                p.est
            };
            let do_a = est * fb >= lim; // G[a] += A F[b]
            let do_b = p.sa != p.sb && est * fa >= lim; // G[b] += A^T F[a]
            if !do_a && !do_b {
                continue;
            }
            scr.touched[p.sa] |= do_a;
            scr.touched[p.sb] |= do_b;
            let mut c = PotContract {
                f: &scr.f,
                g: &mut scr.g,
                nd,
                dstride: ds,
                ld,
                a0: coff[p.sa],
                b0: coff[p.sb],
                do_a,
                do_b,
                cart: true,
            };
            let tp = profile.then(Instant::now);
            let npr = if use_dist {
                p.contract_prims(
                    &scr.prims,
                    &self.opts.xop,
                    &scr.xp,
                    &scr.yp,
                    &scr.zp,
                    &mut c,
                );
                scr.prims.len()
            } else {
                p.contract(&self.opts.xop, &scr.xp, &scr.yp, &scr.zp, cut, &mut c);
                p.nprim_above(cut)
            };
            prim_points += (npr * np) as u64;
            pair_batches += 1;
            if let Some(tp) = tp {
                let e = &mut cls[class_index(p.la, p.lb)];
                e[0] += 1;
                e[1] += (npr * np) as u64;
                e[2] += tp.elapsed().as_nanos() as u64;
            }
        }
        if let Some(cp) = &cnt.classes {
            cp.add(&cls);
        }
        let t3 = Instant::now();
        // K'[X, T] += X G^T over the touched functions T
        scr.tidx.clear();
        for (s, sh) in shells.iter().enumerate() {
            if scr.touched[s] {
                scr.tidx
                    .extend(coff[s]..coff[s] + linus_ints::basis::ncart(sh.l));
            }
        }
        let nt = scr.tidx.len();
        if nt > 0 {
            scr.gc.resize(nt * np, 0.0);
            scr.kblk.resize(nd * nx * nt, 0.0);
            for k in 0..nd {
                for (i, &n) in scr.tidx.iter().enumerate() {
                    scr.gc[i * np..(i + 1) * np]
                        .copy_from_slice(&scr.g[k * ds + n * ld..k * ds + n * ld + np]);
                }
                let dst = faer::MatMut::from_row_major_slice_mut(
                    &mut scr.kblk[k * nx * nt..(k + 1) * nx * nt],
                    nx,
                    nt,
                );
                let lhs = MatRef::from_row_major_slice(&scr.xw[..nx * np], nx, np);
                let rhs = MatRef::from_row_major_slice(&scr.gc[..nt * np], nt, np).transpose();
                linus_base::linalg::matmul(dst, Accum::Replace, lhs, rhs, 1.0, Par::Seq);
            }
        }
        let t4 = Instant::now();
        if nt > 0 {
            index_runs(&scr.tidx, &mut scr.runs);
            accs.with(|acc| {
                for k in 0..nd {
                    acc.add_block(
                        k * nbf * nc,
                        nc,
                        &scr.xidx,
                        &scr.runs,
                        nt,
                        &scr.kblk[k * nx * nt..(k + 1) * nx * nt],
                    );
                }
                acc.batch_done();
            });
        }
        let t5 = Instant::now();
        scr.near = near;
        let ns_ = |d: std::time::Duration| d.as_nanos() as u64;
        cnt.t_ints.fetch_add(ns_(t3 - t2), Ordering::Relaxed);
        cnt.t_gemm
            .fetch_add(ns_(t2 - t1) + ns_(t4 - t3), Ordering::Relaxed);
        cnt.t_other
            .fetch_add(ns_(t1 - t0) + ns_(t5 - t4), Ordering::Relaxed);
        cnt.pair_batches.fetch_add(pair_batches, Ordering::Relaxed);
        cnt.prim_points.fetch_add(prim_points, Ordering::Relaxed);
    }

    /// One batch of [`CosxK::energy`]: sum_k kappa_k sum_g w_g sum_n Ft_kn(g) G_kn(g) with
    /// G = A F, F = D phi, Ft = (M^T D)^T phi (= F without fitting).
    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::needless_range_loop, reason = "indexes several arrays in step")]
    fn batch_energy(
        &self,
        grid: &MolecularGrid,
        b: &Batch,
        dflat: &[Vec<f64>],
        mtd: Option<&[Vec<f64>]>,
        kappa: &[f64],
        scr: &mut Scratch,
        cnt: &Counters,
    ) -> f64 {
        let t0 = Instant::now();
        let basis = &self.basis;
        let shells = &basis.shells;
        let nbf = self.ncart;
        let coff = &self.cart_off;
        let ns = shells.len();
        let nd = dflat.len();
        let np = b.end - b.start;
        let thresh = self.opts.thresh_e;
        let (x, y, z) = (
            &grid.x[b.start..b.end],
            &grid.y[b.start..b.end],
            &grid.z[b.start..b.end],
        );
        let w = &grid.w[b.start..b.end];
        let mut near = std::mem::take(&mut scr.near);
        self.near_shells(b, &mut near);
        let nx: usize = near.iter().map(|&s| shells[s].nfunc()).sum();
        if nx == 0 {
            scr.near = near;
            return 0.0;
        }
        scr.xidx.clear();
        for &s in &near {
            scr.xidx
                .extend(shells[s].ao_start..shells[s].ao_start + shells[s].nfunc());
        }
        scr.phi.resize(nx * np, 0.0);
        eval_ao(basis, &near, x, y, z, 0, &mut scr.phi);
        let t1 = Instant::now();
        let ld = np.div_ceil(8) * 8;
        for (dst, src) in [(&mut scr.xp, x), (&mut scr.yp, y), (&mut scr.zp, z)] {
            dst.clear();
            dst.extend_from_slice(src);
            dst.resize(ld, src[np - 1]);
        }
        // F blocks 0..nd (D phi) and, with fitting, Ft blocks nd..2nd ((M^T D)^T phi)
        let nblk = if mtd.is_some() { 2 * nd } else { nd };
        let ds = nbf * ld;
        scr.f.clear();
        scr.f.resize(nblk * ds, 0.0);
        scr.dxt.resize(nx * nbf, 0.0);
        for k in 0..nblk {
            let src = if k < nd {
                &dflat[k]
            } else {
                &mtd.unwrap()[k - nd]
            };
            for (i, &m) in scr.xidx.iter().enumerate() {
                scr.dxt[i * nbf..(i + 1) * nbf].copy_from_slice(&src[m * nbf..(m + 1) * nbf]);
            }
            let dst = faer::MatMut::from_column_major_slice_with_stride_mut(
                &mut scr.f[k * ds..(k + 1) * ds],
                np,
                nbf,
                ld,
            )
            .transpose_mut();
            let lhs = MatRef::from_row_major_slice(&scr.dxt[..nx * nbf], nx, nbf).transpose();
            let rhs = MatRef::from_row_major_slice(&scr.phi[..nx * np], nx, np);
            linus_base::linalg::matmul(dst, Accum::Replace, lhs, rhs, 1.0, Par::Seq);
        }
        let t2 = Instant::now();
        // per shell: max |F| over the batch and max over functions of sum_g |w_g Ft(g)|
        let ft0 = if mtd.is_some() { nd } else { 0 };
        scr.fmax.clear();
        scr.fmax.resize(ns, 0.0);
        scr.ftw.clear();
        scr.ftw.resize(ns, 0.0);
        for (s, sh) in shells.iter().enumerate() {
            let (mut m, mut t): (f64, f64) = (0.0, 0.0);
            for k in 0..nd {
                for n in coff[s]..coff[s] + linus_ints::basis::ncart(sh.l) {
                    let r = &scr.f[k * ds + n * ld..k * ds + n * ld + np];
                    m = r.iter().fold(m, |a, v| a.max(v.abs()));
                    let rt = &scr.f[(ft0 + k) * ds + n * ld..(ft0 + k) * ds + n * ld + np];
                    t = t.max(rt.iter().zip(w).map(|(v, wg)| (v * wg).abs()).sum());
                }
            }
            scr.fmax[s] = m;
            scr.ftw[s] = t;
        }
        scr.g.clear();
        scr.g.resize(nd * ds, 0.0);
        scr.touched.clear();
        scr.touched.resize(ns, false);
        let mut pair_batches = 0u64;
        let mut prim_points = 0u64;
        let mut cls = [[0u64; 3]; 25];
        let profile = cnt.classes.is_some();
        for (ip, ps) in self.pair_scr.iter().enumerate() {
            // contribution bounds: G[a] += A F[b] meets Ft[a], G[b] += A F[a] meets Ft[b]
            let (sa, sb) = (ps.sa as usize, ps.sb as usize);
            let ba = scr.ftw[sa] * scr.fmax[sb];
            let bb = if sa != sb {
                scr.ftw[sb] * scr.fmax[sa]
            } else {
                0.0
            };
            let fm = ba.max(bb);
            if ps.est * fm < thresh {
                continue;
            }
            let p = &self.pairs[ip];
            let cut = thresh / fm;
            let use_dist = self.opts.dist_screen && !p.is_md();
            let est = if use_dist {
                scr.prims.clear();
                p.screen_prims(b.center, b.radius, cut, &mut scr.prims)
            } else {
                p.est
            };
            let do_a = est * ba >= thresh;
            let do_b = est * bb >= thresh;
            if !do_a && !do_b {
                continue;
            }
            scr.touched[p.sa] |= do_a;
            scr.touched[p.sb] |= do_b;
            let mut c = PotContract {
                f: &scr.f,
                g: &mut scr.g,
                nd,
                dstride: ds,
                ld,
                a0: coff[p.sa],
                b0: coff[p.sb],
                do_a,
                do_b,
                cart: true,
            };
            let tp = profile.then(Instant::now);
            let npr = if use_dist {
                p.contract_prims(
                    &scr.prims,
                    &self.opts.xop,
                    &scr.xp,
                    &scr.yp,
                    &scr.zp,
                    &mut c,
                );
                scr.prims.len()
            } else {
                p.contract(&self.opts.xop, &scr.xp, &scr.yp, &scr.zp, cut, &mut c);
                p.nprim_above(cut)
            };
            prim_points += (npr * np) as u64;
            pair_batches += 1;
            if let Some(tp) = tp {
                let e = &mut cls[class_index(p.la, p.lb)];
                e[0] += 1;
                e[1] += (npr * np) as u64;
                e[2] += tp.elapsed().as_nanos() as u64;
            }
        }
        if let Some(cp) = &cnt.classes {
            cp.add(&cls);
        }
        let t3 = Instant::now();
        let mut e = 0.0;
        for (s, sh) in shells.iter().enumerate() {
            if !scr.touched[s] {
                continue;
            }
            for k in 0..nd {
                let mut ek = 0.0;
                for n in coff[s]..coff[s] + linus_ints::basis::ncart(sh.l) {
                    let (fr, gr) = (
                        &scr.f[(ft0 + k) * ds + n * ld..(ft0 + k) * ds + n * ld + np],
                        &scr.g[k * ds + n * ld..k * ds + n * ld + np],
                    );
                    for g in 0..np {
                        ek += w[g] * fr[g] * gr[g];
                    }
                }
                e += kappa[k] * ek;
            }
        }
        let t4 = Instant::now();
        scr.near = near;
        let ns_ = |d: std::time::Duration| d.as_nanos() as u64;
        cnt.t_ints.fetch_add(ns_(t3 - t2), Ordering::Relaxed);
        cnt.t_gemm.fetch_add(ns_(t2 - t1), Ordering::Relaxed);
        cnt.t_other
            .fetch_add(ns_(t1 - t0) + ns_(t4 - t3), Ordering::Relaxed);
        cnt.pair_batches.fetch_add(pair_batches, Ordering::Relaxed);
        cnt.prim_points.fetch_add(prim_points, Ordering::Relaxed);
        e
    }

    /// The exchange energy sum_k kappa_k D_k . K\[D_k\] on the grid in use (with overlap
    /// fitting if set up: D . sym(M K') = D . M K' for symmetric D) without building K: per
    /// batch sum_g w_g Ft(g) . G(g), screened by the bound of each shell pair's contribution
    /// to the energy (threshold [`CosxConfig::thresh_e`]). Thread-count independent (batch
    /// energies summed in batch order).
    pub fn energy(&mut self, dens: &[Mat<f64>], kappa: &[f64], par: Par) -> f64 {
        assert_eq!(dens.len(), kappa.len());
        let t0 = Instant::now();
        let nbf = self.basis.nbf;
        let gi = self.active;
        let tt = self.tsph.as_ref();
        // F and Ft in Cartesian rows: D T and M^T D T
        let dflat: Vec<Vec<f64>> = dens
            .iter()
            .map(|d| times_t(&mat_to_rowmajor(d.as_ref()), nbf, tt, par))
            .collect();
        let mtd: Option<Vec<Vec<f64>>> = self.fits[gi].as_ref().map(|m| {
            dens.iter()
                .map(|d| {
                    times_t(
                        &mat_to_rowmajor(
                            linus_base::linalg::mm(m.transpose(), d.as_ref(), par).as_ref(),
                        ),
                        nbf,
                        tt,
                        par,
                    )
                })
                .collect()
        });
        let cnt = Counters {
            t_ints: AtomicU64::new(0),
            t_gemm: AtomicU64::new(0),
            t_other: AtomicU64::new(0),
            pair_batches: AtomicU64::new(0),
            prim_points: AtomicU64::new(0),
            classes: self.opts.profile.then(ClassProfile::new),
        };
        let grid = &self.grids[gi];
        let eb: Vec<f64> = grid
            .batches
            .par_iter()
            .map_init(Scratch::default, |scr, b| {
                self.batch_energy(grid, b, &dflat, mtd.as_deref(), kappa, scr, &cnt)
            })
            .collect();
        let e = eb.iter().sum();
        let s = &mut self.stats;
        s.builds += 1;
        s.grid = gi;
        s.points_total += self.grids[gi].len();
        s.time = t0.elapsed().as_secs_f64();
        s.t_ints = cnt.t_ints.load(Ordering::Relaxed) as f64 * 1e-9;
        s.t_gemm = cnt.t_gemm.load(Ordering::Relaxed) as f64 * 1e-9;
        s.t_other = cnt.t_other.load(Ordering::Relaxed) as f64 * 1e-9;
        s.pair_batches = cnt.pair_batches.load(Ordering::Relaxed);
        s.prim_points = cnt.prim_points.load(Ordering::Relaxed);
        s.classes = cnt.classes.as_ref().map(|c| c.stats()).unwrap_or_default();
        e
    }

    /// K for densities `dens` on the grid in use (no incremental logic).
    pub fn build(&mut self, dens: &[Mat<f64>], par: Par) -> Vec<Mat<f64>> {
        self.build_scaled(dens, 1.0, par)
    }

    /// [`CosxK::build`] with the screening threshold times `scale`.
    fn build_scaled(&mut self, dens: &[Mat<f64>], scale: f64, par: Par) -> Vec<Mat<f64>> {
        let t0 = Instant::now();
        let nbf = self.basis.nbf;
        let gi = self.active;
        let dflat: Vec<Vec<f64>> = dens.iter().map(|d| mat_to_rowmajor(d.as_ref())).collect();
        let (kp, cnt) = self.build_kprime(gi, &dflat, scale, par);
        let out = kp
            .iter()
            .map(|t| {
                let kp = Mat::from_fn(nbf, nbf, |i, j| t[i * nbf + j]);
                let kf = match &self.fits[gi] {
                    Some(m) => linus_base::linalg::mm(m.as_ref(), kp.as_ref(), par),
                    None => kp,
                };
                Mat::from_fn(nbf, nbf, |i, j| 0.5 * (kf[(i, j)] + kf[(j, i)]))
            })
            .collect();
        let s = &mut self.stats;
        s.builds += 1;
        s.grid = gi;
        s.points_total += self.grids[gi].len();
        s.time = t0.elapsed().as_secs_f64();
        s.t_ints = cnt.t_ints.load(Ordering::Relaxed) as f64 * 1e-9;
        s.t_gemm = cnt.t_gemm.load(Ordering::Relaxed) as f64 * 1e-9;
        s.t_other = cnt.t_other.load(Ordering::Relaxed) as f64 * 1e-9;
        s.pair_batches = cnt.pair_batches.load(Ordering::Relaxed);
        s.prim_points = cnt.prim_points.load(Ordering::Relaxed);
        s.classes = cnt.classes.as_ref().map(|c| c.stats()).unwrap_or_default();
        out
    }
}

fn max_abs(m: &Mat<f64>) -> f64 {
    let mut v: f64 = 0.0;
    for j in 0..m.ncols() {
        for i in 0..m.nrows() {
            v = v.max(m[(i, j)].abs());
        }
    }
    v
}

impl CosxK {
    /// One line per build on stderr when LINUS_COSX_LOG is set.
    fn log(&self, incremental: bool) {
        if std::env::var_os("LINUS_COSX_LOG").is_some() {
            let s = &self.stats;
            eprintln!(
                "cosx build {:3}: grid {} ({} pts) {} {:.3}s  [ints {:.2} gemm {:.2} other {:.2} thread-s; {} pair-batches, {:.3e} prim-pts]",
                s.builds,
                s.grid,
                self.grids[s.grid].len(),
                if incremental { "incr" } else { "full" },
                s.time,
                s.t_ints,
                s.t_gemm,
                s.t_other,
                s.pair_batches,
                s.prim_points as f64
            );
        }
    }
}

impl JkBuilder for CosxK {
    fn name(&self) -> String {
        format!(
            "cosx-K(points={}, thresh={:e})",
            self.npoints(),
            self.opts.thresh
        )
    }

    fn reset_incremental(&mut self) {
        self.prev = None;
    }

    fn built_incrementally(&self) -> bool {
        self.last_incremental
    }

    /// The next SCF grid (its first build is a full one).
    fn next_approximation(&mut self) -> bool {
        if self.active + 1 >= self.grids.len() {
            return false;
        }
        self.active += 1;
        self.prev = None;
        self.since_reset = 0;
        true
    }

    /// Range-separated exchange in one pass with the combined operator a/r + b erf(w r)/r:
    /// one set of grid-potential integrals per batch (two Boys/Rys stages) instead of a
    /// second pass for K_LR. Caffeine, wB97X, GridX2 with overlap fitting, 8 threads
    /// (linus-dev examples/cosx_op_bench): def2-SVP 0.598 -> 0.451 s, def2-TZVP 2.109 -> 1.559 s per build.
    fn set_exchange_operator(&mut self, op: Operator) -> bool {
        self.opts.xop = op;
        self.prev = None;
        true
    }

    fn compute(
        &mut self,
        req: &JkRequest,
        par: Par,
    ) -> Result<JkOutput, linus_base::error::LinusError> {
        assert!(
            !req.want_j,
            "CosxK builds K only; combine it with a J builder"
        );
        if !req.want_k {
            return Ok(JkOutput::default());
        }
        let nd = req.dens.len();
        // move to the next grid once the density has nearly converged (unless the SCF decides:
        // switch_dmax = 0, see JkBuilder::next_approximation)
        if self.active + 1 < self.grids.len() && self.opts.switch_dmax > 0.0 {
            if let Some(last) = &self.last_dens {
                if last.len() == nd
                    && last
                        .iter()
                        .zip(req.dens)
                        .all(|(a, b)| max_abs(&(b - a)) < self.opts.switch_dmax)
                {
                    self.active += 1;
                    self.prev = None;
                }
            }
        }
        self.last_dens = Some(req.dens.to_vec());
        let usable = self.opts.incremental
            && self.since_reset < self.opts.reset_every
            && matches!(&self.prev, Some((d, _)) if d.len() == nd);
        if usable {
            let (dp, kp) = self.prev.take().unwrap();
            let delta: Vec<Mat<f64>> = req.dens.iter().zip(&dp).map(|(d, o)| d - o).collect();
            let small = delta
                .iter()
                .zip(req.dens)
                .all(|(dd, d)| max_abs(dd) <= 0.5 * max_abs(d));
            if small {
                let ddmax = delta.iter().map(max_abs).fold(0.0, f64::max);
                let scale = if self.opts.delta_ref > 0.0 {
                    (ddmax / self.opts.delta_ref).clamp(self.opts.delta_scale, 1.0)
                } else {
                    self.opts.delta_scale
                };
                let dk = self.build_scaled(&delta, scale, par);
                let k: Vec<Mat<f64>> = kp.iter().zip(&dk).map(|(a, b)| a + b).collect();
                self.since_reset += 1;
                self.stats.incremental_builds += 1;
                self.last_incremental = true;
                self.prev = Some((req.dens.to_vec(), k.clone()));
                self.log(true);
                return Ok(JkOutput { j: vec![], k });
            }
        }
        let k = self.build(req.dens, par);
        self.since_reset = 0;
        self.last_incremental = false;
        if self.opts.incremental {
            self.prev = Some((req.dens.to_vec(), k.clone()));
        }
        self.log(false);
        Ok(JkOutput { j: vec![], k })
    }
}

/// AngularGrid levels of ORCA's DefGrid1/2/3 COSX grids: first SCF iterations, last SCF
/// iterations, final energy.
pub fn defgrid_levels(defgrid: u8) -> [u8; 3] {
    match defgrid {
        0 | 1 => [1, 1, 2],
        2 => [1, 2, 3],
        _ => [2, 3, 4],
    }
}

/// The SCF exchange builder of an ORCA-like COSX scheme: GridX `levels[0]` until the SCF
/// moves it on ([`JkBuilder::next_approximation`]; or, with `switch_dmax` > 0, until the
/// density changes by less than `switch_dmax`), then GridX `levels[1]`; overlap fitting and
/// incremental builds on. The final energy uses [`final_grid_exchange`] on GridX `levels[2]`.
pub fn scf_builder(
    basis: &BasisSet,
    z: &[u32],
    positions: &[[f64; 3]],
    levels: [u8; 3],
    opts: CosxConfig,
) -> Result<CosxK, linus_grid::GridError> {
    let g1 = MolecularGrid::build(z, positions, &GridSpec::cosx(levels[0]))?;
    let mut k = CosxK::new(basis, g1, opts);
    if levels[1] != levels[0] {
        k = k.with_next_grid(MolecularGrid::build(
            z,
            positions,
            &GridSpec::cosx(levels[1]),
        )?);
    }
    let n = basis.nbf;
    let mut s = Mat::zeros(n, n);
    let plan = linus_ints::OneEPlan::new(basis, linus_ints::DerivOrder::Zero);
    plan.overlap(&mut linus_base::linalg::ao_view_mut(s.as_mut()));
    Ok(k.with_overlap_fitting(s.as_ref()))
}

/// Exchange energy -1/2 a_x sum_s D_s . K\[D_s\] of converged spin densities on GridX `level`
/// (with overlap fitting), as ORCA's final COSX energy. `restricted`: D_b = D_a.
#[allow(clippy::too_many_arguments, reason = "a kernel's arrays and sizes")]
pub fn final_grid_exchange(
    basis: &BasisSet,
    z: &[u32],
    positions: &[[f64; 3]],
    level: u8,
    opts: CosxConfig,
    d_a: &Mat<f64>,
    d_b: &Mat<f64>,
    restricted: bool,
    a_x: f64,
    par: Par,
) -> Result<f64, String> {
    let n = basis.nbf;
    let mut s = Mat::zeros(n, n);
    let plan = linus_ints::OneEPlan::new(basis, linus_ints::DerivOrder::Zero);
    plan.overlap(&mut linus_base::linalg::ao_view_mut(s.as_mut()));
    let mut k = CosxK::new(
        basis,
        MolecularGrid::build(z, positions, &GridSpec::cosx(level)).map_err(|e| e.to_string())?,
        CosxConfig {
            incremental: false,
            ..opts
        },
    )
    .with_overlap_fitting(s.as_ref());
    // the energy without K (energy-bound screening, CosxConfig::thresh_e)
    Ok(if restricted {
        k.energy(std::slice::from_ref(d_a), &[-a_x], par)
    } else {
        k.energy(&[d_a.clone(), d_b.clone()], &[-0.5 * a_x, -0.5 * a_x], par)
    })
}

impl CosxConfig {
    /// Settings for SCF iterations: incremental builds on (difference densities below max
    /// |D - D_prev| = 1e-4 screened at threshold x max|D - D_prev| / 1e-4, at least x 1e-3), the
    /// grid switch left to the SCF.
    pub fn scf() -> Self {
        CosxConfig {
            incremental: true,
            delta_ref: 1e-4,
            delta_scale: 1e-3,
            switch_dmax: 0.0,
            ..Default::default()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::IncoreJk;
    use linus_base::terms::{JkBuilder, JkRequest};
    use linus_ints::basis::BasisDef;
    use linus_ints::{PairOptions, PairSet};

    fn water() -> (BasisSet, Vec<u32>, Vec<[f64; 3]>) {
        let json = r#"{"name":"t","elements":{"8":{"electron_shells":[
          {"function_type":"gto","angular_momentum":[0],"exponents":["130.7","23.81","6.44"],"coefficients":[["0.154","0.535","0.444"]]},
          {"function_type":"gto","angular_momentum":[0],"exponents":["0.38"],"coefficients":[["1.0"]]},
          {"function_type":"gto","angular_momentum":[1],"exponents":["5.03","1.17","0.38"],"coefficients":[["0.156","0.607","0.392"]]},
          {"function_type":"gto","angular_momentum":[2],"exponents":["1.2"],"coefficients":[["1.0"]]}]},
          "1":{"electron_shells":[{"function_type":"gto","angular_momentum":[0],"exponents":["3.42","0.62"],"coefficients":[["0.15","0.53"]]},
          {"function_type":"gto","angular_momentum":[1],"exponents":["0.8"],"coefficients":[["1.0"]]}]}}}"#;
        let z = vec![8, 1, 1];
        let xyz = vec![[0.0, 0.0, 0.0], [0.0, 1.43, 1.11], [0.0, -1.43, 1.11]];
        (
            BasisSet::new(&BasisDef::from_bse_json(json).unwrap(), &z, &xyz, true).unwrap(),
            z,
            xyz,
        )
    }

    fn density(n: usize, seed: f64) -> Mat<f64> {
        let c = Mat::from_fn(n, 5, |i, j| {
            ((i * 7 + j * 3) as f64 * seed).sin() / (1.0 + i as f64 * 0.2)
        });
        &c * c.transpose()
    }

    #[test]
    fn deterministic_across_thread_counts_and_densities() {
        let (basis, z, xyz) = water();
        let n = basis.nbf;
        let dens = vec![density(n, 0.37), density(n, 0.91)];
        let run = |threads: usize, d: &[Mat<f64>]| -> Vec<Mat<f64>> {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            pool.install(|| {
                let mut k = CosxK::new(
                    &basis,
                    MolecularGrid::build(&z, &xyz, &GridSpec::cosx(2)).unwrap(),
                    CosxConfig::default(),
                );
                k.build(d, Par::Seq)
            })
        };
        let a = run(1, &dens);
        let b = run(5, &dens);
        for (x, y) in a.iter().zip(&b) {
            for j in 0..n {
                for i in 0..n {
                    assert_eq!(
                        x[(i, j)].to_bits(),
                        y[(i, j)].to_bits(),
                        "thread-count dependence at ({i},{j})"
                    );
                }
            }
        }
        // two densities in one build = two single-density builds
        let c = run(3, &dens[1..]);
        for j in 0..n {
            for i in 0..n {
                assert!((c[0][(i, j)] - a[1][(i, j)]).abs() < 1e-13);
            }
        }
    }

    #[test]
    fn close_to_exact_exchange() {
        let (basis, z, xyz) = water();
        let n = basis.nbf;
        let d = vec![density(n, 0.37)];
        let req = JkRequest {
            dens: &d,
            factors: None,
            want_j: false,
            want_k: true,
        };
        let kex = IncoreJk::new(&basis, &PairSet::build(&basis, PairOptions::default()))
            .compute(&req, Par::Seq)
            .unwrap()
            .k
            .remove(0);
        let (s, _) = linus_ints::reference::oneshot::overlap_kinetic(&basis);
        let s = Mat::from_fn(n, n, |i, j| s[i * n + j]);
        let mut k = CosxK::new(
            &basis,
            MolecularGrid::build(&z, &xyz, &GridSpec::cosx(3)).unwrap(),
            CosxConfig::default(),
        )
        .with_overlap_fitting(s.as_ref());
        let kc = k.compute(&req, Par::Seq).unwrap().k.remove(0);
        let e = linus_base::linalg::dot(d[0].as_ref(), kex.as_ref());
        let de = linus_base::linalg::dot(d[0].as_ref(), kc.as_ref()) - e;
        assert!(de.abs() < 1e-4 * e.abs(), "exchange energy {e} error {de}");
    }

    #[test]
    fn incremental_equals_full() {
        let (basis, z, xyz) = water();
        let n = basis.nbf;
        let g = MolecularGrid::build(&z, &xyz, &GridSpec::cosx(1)).unwrap();
        let mut inc = CosxK::new(
            &basis,
            g.clone(),
            CosxConfig {
                incremental: true,
                ..Default::default()
            },
        );
        let mut full = CosxK::new(&basis, g, CosxConfig::default());
        let (d0, d1) = (density(n, 0.37), density(n, 0.37) * 1.01);
        let k = |b: &mut CosxK, d: &Mat<f64>| {
            b.compute(
                &JkRequest {
                    dens: std::slice::from_ref(d),
                    factors: None,
                    want_j: false,
                    want_k: true,
                },
                Par::Seq,
            )
            .unwrap()
            .k
            .remove(0)
        };
        k(&mut inc, &d0);
        let a = k(&mut inc, &d1);
        assert_eq!(inc.stats.incremental_builds, 1);
        let b = k(&mut full, &d1);
        for j in 0..n {
            for i in 0..n {
                assert!((a[(i, j)] - b[(i, j)]).abs() < 1e-11);
            }
        }
    }
}
