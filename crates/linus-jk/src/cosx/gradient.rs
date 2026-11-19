//! Analytic nuclear gradient of the COSX exchange energy (jk = rij-snk) as [`crate::cosx`]
//! computes it. For spin densities D_s with factors kappa_s (RKS: D_alpha, kappa = -a_x; UKS:
//! D_alpha, D_beta, kappa = -a_x/2 each):
//!
//!   E_K = sum_s kappa_s B(D_s),   B(D) = D . M K'\[D\] = sum_g w_g sum_ns Ft_n(g) A_ns(g) F_s(g),
//!   F = D phi,  Ft = (D M) phi,  A_ns(g) = <n|1/|r - r_g||s>,  M = S S_num^-1 (overlap fitting,
//!   Izsák & Neese 2011; M = 1 without it), S_num = sum_g w_g phi phi^T.
//!
//! At fixed densities, dB/dX splits into six parts, all computed and reported separately
//! ([`CosxGradParts`]):
//!
//! * `centres`: sum_g w_g sum_ns P_ns(g) dA_ns(g)/dX over the centres of n and s, with the pair
//!   density P = Ft F^T + F Ft^T per point (Rys derivative kernels,
//!   [`linus_ints::points::grid_pot_deriv`], which also return G = A F and Gt = A Ft in the same pass);
//! * `ao`: the basis functions on the grid, -sum_g w_g grad phi_k(g) \[(M^T D G)_k + (D Gt)_k\](g)
//!   for k on X (= 2 D G without fitting);
//! * `points`: the grid moving rigidly with the atoms (r_g = R_owner(g) + const): the point
//!   derivative of the integrals (minus the sum over the two centres) and of phi, also for the
//!   numerical overlap;
//! * `weights`: the partition-weight derivatives sum_g dw_g/dX [e_K(g) - e_S(g)] (Becke / SSF
//!   weights of `linus_grid`; e_K the B integrand, e_S that of tr(S_num Z));
//! * `fit_s`: tr(dS Y), Y = S_num^-1 K' D (the analytic overlap in M);
//! * `fit_snum`: -tr(dS_num Z), Z = Y M (the numerical overlap in M, fixed points).
//!
//! Everything is translation invariant when `points` is included. The SCF's -W S^x term
//! accounts for the orbital response of a variational energy; the fitted energy is not
//! variational with respect to the density (B(D1, D2) = D1 . M K'\[D2\] is not symmetric, and
//! the SCF uses K = sym(M K')), so with fitting the gradient misses a response term (measured
//! in the report: small next to the grid error). ORCA's COSX gradient omits the weight
//! derivatives (Izsák & Neese 2011; Neese et al. 2009).
//!
//! Cost: one pass over the grid with the derivative kernels (two densities per spin with
//! fitting), one GEMM per batch for each of G -> H, Gt -> H2 and K', and a cheap second pass
//! (AO values only) for the numerical-overlap and weight terms. Thread-count independent: the
//! per-atom sums and K' are accumulated in fixed point ([`crate::cosx`]'s accumulators), per-point
//! values are written per batch and summed in point order. F, Ft and G are kept in Cartesian
//! rows as in [`crate::cosx`]. The derivative kernel costs 2.7-3x the energy kernel per
//! primitive pair and point (one more Rys root for odd la + lb, the 1D integrals one order
//! higher, the pair density per chunk, G for two F blocks).
//!
//! Screening (default): density-weighted (`DensScreen`, [`CosxGradientConfig::thresh_d`]): each
//! shell pair of a batch runs the derivative kernel only where the bound of its contribution
//! through the pair density passes, and the cheaper energy kernel where only its integrals are
//! needed (basis-function term, weight term, K'). Every term but K' is a product of two
//! density-propagated functions, so this is an energy-type criterion; K' only enters through
//! the fitting-matrix derivative, whose S and S_num parts cancel its screening errors, and keeps
//! the K criterion at a loose [`CosxGradientConfig::thresh_k`]. On ts113 def2-TZVP (B3LYP,
//! GridX3) the derivative kernels evaluate 37% fewer primitive-pair x point blocks than with
//! the K criterion at 1e-9 (2.47e10 instead of 3.93e10, plus 4.0e9 in the energy kernels), at
//! a gradient change of 7.3e-9 Eh/bohr (screening error against a tight reference: 6.8e-9
//! instead of 2.0e-9; the COSX error against exact integrals is ~1e-4). The classes with
//! la + lb <= 2 use closed forms instead of Rys quadrature ([`linus_ints::points::grid_pot_deriv`]).
//! With the faster kernels (bench/results/2026-11-07-cosx-grad-large) the COSX gradient step
//! of ts113 def2-TZVP takes ~70% of the time of the K-criterion code.
//!
//! [`CosxGradientConfig::deriv_fit`] computes the alternative of fitting the derivative functions
//! (Izsák & Neese 2011, eq. 21), measured less accurate (see the option).

use crate::cosx::{
    class_index, index_runs, times_t, times_tt, ClassProfile, ClassStat, CosxConfig, CosxK,
    ThreadAccs,
};
use faer::{Accum, Mat, MatRef, Par};
use linus_base::error::LinusError;
use linus_base::linalg::mat_to_rowmajor;
use linus_core::gradient::Gradient;
use linus_core::timing::Timers;
use linus_grid::{Batch, MolecularGrid};
use linus_ints::operator::Operator;
use linus_ints::points::colloc::Collocator;
use linus_ints::points::grid_pot::{PotContract, PotPrim};
use linus_ints::points::grid_pot_deriv::PotDerivContract;
use linus_ints::BasisSet;
use rayon::prelude::*;
use std::time::Instant;

#[derive(Clone, Debug)]
pub struct CosxGradientConfig {
    /// Screening threshold of the K criterion (as [`CosxConfig::thresh`]) when every term is
    /// screened with it (`thresh_d` = None), applied with the larger of the value and gradient
    /// row sums of the weighted basis functions: 1e-9 leaves screening errors of ~3e-9 Eh/bohr
    /// (u1an39 def2-TZVP: 2.4e-11 at 1e-11, 3.0e-9 at 1e-9, 2.2e-8 at 1e-8), far below the
    /// grid error (~1e-4 against exact integrals).
    pub thresh: f64,
    /// With `thresh_d`: the K-criterion threshold of the integrals that only K' (fitting)
    /// needs. K' enters the gradient through tr(dM S_num Y) (M = S S_num^-1, dM small on a good
    /// grid), so its screening errors largely cancel between the S and S_num terms: on ts113
    /// def2-TZVP 1e-7 changes fit_s and fit_snum by 2.7e-7 each but their sum by < 1e-10.
    pub thresh_k: f64,
    /// Basis-function value below which a shell does not reach a batch.
    pub bf_cut: f64,
    /// The exact-exchange operator (Coulomb by default).
    pub xop: Operator,
    /// Overlap fitting (consistent derivative of K = sym(S S_num^-1 K')).
    pub fitting: bool,
    /// Include the point-motion terms in `gradient` (they are always computed). Use together
    /// with `weight_derivs`: the two largely cancel, and point motion alone is far less
    /// accurate than neither (la18 B3LYP/def2-SVP: 1e-3 vs 3e-5 Eh/bohr against exact).
    pub point_motion: bool,
    /// Compute and include the grid-weight derivatives.
    pub weight_derivs: bool,
    /// Distance-aware primitive screening per batch.
    pub dist_screen: bool,
    /// Time every shell pair's derivative integrals by class into [`CosxGradResult::classes`].
    pub profile: bool,
    /// Also compute [`CosxGradParts::ao_dfit`], the basis-function (grid-side) derivative
    /// integrals with fitted derivative functions (Izsák & Neese 2011, eq. 21; ORCA's COSX
    /// gradient) instead of the derivative of the fitted energy. Measured against exact
    /// integrals (B3LYP/def2-SVP, GridX3, max |dG| Eh/bohr), centres + ao_dfit is less accurate
    /// than the consistent derivative: la18 9.9e-5 vs 3.9e-5, u1an39 4.3e-4 vs 1.1e-4, ts61
    /// 1.5e-3 vs 2.4e-4 (with the net force removed or the grid response added: no better);
    /// on GridX4 both converge (la18 3.5e-5 vs 2.7e-5). Diagnostic only.
    pub deriv_fit: bool,
    /// Density-weighted screening (see `DensScreen`): bounds of each shell pair's
    /// contributions to the gradient terms that contract the integrals with two
    /// density-propagated functions (centres and points, basis functions, weights), at this
    /// threshold; `thresh_k` then screens the integrals that only K' (fitting) needs. None:
    /// every term screened with the K criterion at `thresh`.
    pub thresh_d: Option<f64>,
    /// Bound of |dw_g/dX| / w_g (bohr^-1) used by the density-weighted bound of the
    /// weight-derivative term.
    pub wfac: f64,
    /// With `thresh_d`: the threshold of the integral-only consumers (basis-function and weight
    /// terms) as a multiple of `thresh_d`. Their screening errors are a fraction of those of the
    /// derivative integrals: on ts113 def2-TZVP the default 3 halves the integral-only blocks
    /// (4.0e9 instead of 7.0e9 primitive-pair x point evaluations; 3.5% fewer derivative ones)
    /// and moves the gradient by 2.2e-9 Eh/bohr.
    pub g_factor: f64,
}

/// Default [`CosxGradientConfig::thresh_d`].
pub const DEFAULT_THRESH_D: f64 = 1e-11;

impl Default for CosxGradientConfig {
    fn default() -> Self {
        CosxGradientConfig {
            thresh: 1e-9,
            thresh_k: 1e-7,
            bf_cut: 1e-11,
            xop: Operator::Coulomb,
            fitting: true,
            point_motion: true,
            weight_derivs: true,
            dist_screen: true,
            profile: false,
            deriv_fit: false,
            thresh_d: Some(DEFAULT_THRESH_D),
            wfac: 10.0,
            g_factor: 3.0,
        }
    }
}

/// The parts of dE_K/dX (natom x 3 each); see the module documentation.
#[derive(Clone, Debug, Default)]
pub struct CosxGradParts {
    pub centres: Vec<[f64; 3]>,
    pub ao: Vec<[f64; 3]>,
    pub points: Vec<[f64; 3]>,
    pub weights: Vec<[f64; 3]>,
    pub fit_s: Vec<[f64; 3]>,
    pub fit_snum: Vec<[f64; 3]>,
    /// With [`CosxGradientConfig::deriv_fit`]: the grid-side derivative integrals with fitted
    /// derivative functions, -2 sum_s kappa_s sum_{mu on A} sum_g Q^(d)_mu(g) (D_s G_s)_mu(g),
    /// Q^(d) = M^(d) X^(d), X^(d) = w d_d phi, M^(d) = S^(d) S_num^(d)^-1 with the analytic and
    /// numerical overlaps of the derivative functions d_d phi. ORCA-style gradient:
    /// centres + ao_dfit (no fitting-matrix or grid-response terms).
    pub ao_dfit: Vec<[f64; 3]>,
}

#[derive(Clone, Debug)]
pub struct CosxGradResult {
    /// Sum of the parts selected by the options.
    pub gradient: Vec<[f64; 3]>,
    pub parts: CosxGradParts,
    /// E_K on this grid for the densities (a check against the energy code).
    pub energy: f64,
    pub npoints: usize,
    /// Shell-pair x batch blocks and primitive-pair x point evaluations of the main pass
    /// (derivative kernels).
    pub pair_batches: u64,
    pub prim_points: u64,
    /// The same for the blocks that only need the integrals (energy kernels; density-weighted
    /// screening).
    pub g_pair_batches: u64,
    pub g_prim_points: u64,
    /// Basis functions on the batches and touched Cartesian rows (G), summed over batches.
    pub sum_nx: u64,
    pub sum_nt: u64,
    /// Per shell-pair class counters of the main pass (with [`CosxGradientConfig::profile`]):
    /// derivative kernels, and the integral-only blocks.
    pub classes: Vec<ClassStat>,
    pub classes_g: Vec<ClassStat>,
    pub timers: Timers,
}

fn to_atoms(v: &[f64], natom: usize, part: usize) -> Vec<[f64; 3]> {
    (0..natom)
        .map(|a| {
            [
                v[(part * natom + a) * 3],
                v[(part * natom + a) * 3 + 1],
                v[(part * natom + a) * 3 + 2],
            ]
        })
        .collect()
}

#[derive(Default)]
struct Scr {
    near: Vec<usize>,
    xidx: Vec<usize>,
    xatom: Vec<usize>,
    phi: Vec<f64>,
    xw: Vec<f64>,
    dxt: Vec<f64>,
    f: Vec<f64>,
    g: Vec<f64>,
    fmax: Vec<f64>,
    touched: Vec<bool>,
    tidx: Vec<usize>,
    lhs: Vec<f64>,
    gc: Vec<f64>,
    h: Vec<f64>,
    kblk: Vec<f64>,
    xp: Vec<f64>,
    yp: Vec<f64>,
    zp: Vec<f64>,
    wp: Vec<f64>,
    gpt: Vec<f64>,
    prims: Vec<PotPrim>,
    loc: Vec<f64>,
    cbuf: Vec<f64>,
    runs: Vec<(usize, usize, usize)>,
    xd: Vec<f64>,
    /// Density-weighted screening: per-shell quantities of the batch (see [`DensScreen`]).
    sh: [Vec<f64>; NSH],
    /// The same per shell in one row (fm, ftm, fw, ftw, fmk, r1, r2): the pair loop reads two
    /// rows per pair
    shp: Vec<[f64; 8]>,
    nearsh: Vec<usize>,
    tmp: Vec<f64>,
}

/// Per-shell screening quantities of a batch, indices into [`Scr::sh`]: F side (rows of the
/// shell summed): max |F|, max |Ft|, sum_g w |F|, sum_g w |Ft|; max |F| over rows (K
/// criterion); max sum_g w |grad phi| of the shell's functions; the receiving factors of the
/// basis-function and weight terms.
const SH_FM: usize = 0;
const SH_FTM: usize = 1;
const SH_FW: usize = 2;
const SH_FTW: usize = 3;
const SH_FMK: usize = 4;
const SH_GW: usize = 5;
const SH_R1: usize = 6;
const SH_R2: usize = 7;
const NSH: usize = 8;

/// Data of the density-weighted screening ([`CosxGradientConfig::thresh_d`]).
///
/// Each shell pair (a, b) of a batch has bounds of its contributions to the terms of the
/// gradient (est: the pair's primitive estimates, scaled by their decay with the distance to
/// the batch; kappa = sum_s |kappa_s|):
///
/// * centres and points (derivative integrals contracted with the pair density
///   P = w (Ft_a F_b + F_a Ft_b)): est_d kappa [min(FTW_a FM_b, FTM_a FW_b) + min(FW_a FTM_b,
///   FM_a FTW_b)], est_d = sum_q est_q f_q. The centre derivatives of a primitive pair are
///   bounded by dA = 2 a |PA'| + la / |PA'| (|PA'| = |PA| + (2p)^-1/2; the 1D recurrence of
///   the centre derivatives) and dB likewise, the point derivative by dC = dA + dB and, for a
///   batch at distance r beyond the charge distribution's centre, by (la + lb + 1) / r (its
///   multipole field); with translational invariance (dA + dB + dC = 0),
///   f = min(dA, dB + dC) + min(dB, dA + dC) + dC, and 2 dC for two shells on one atom (the
///   atom's centre term is -dC). Tight primitives (dA ~ (2a)^1/2) thus count with their field
///   at distant batches, not with their width;
/// * the basis-function term through G_a = A_ab F_b and Gt_a = A_ab Ft_b (H = (M^T D) G +
///   D Gt): est kappa (FM_b AOT_a + FTM_b AO_a), AO_a = sum_{n in a} sum_{k in X} |D_kn|
///   sum_g w |grad phi_k| (bounded with shell-block sums of |D| and the shells' maxima of
///   sum_g w |grad phi_k|), AOT_a the same with M^T D;
/// * the weight derivatives through e_K: est kappa FM_b wfac FTW_a (|dw_g/dX| <= wfac w_g);
/// * K' (fitting): est FMK_b max_m sum_g |w phi_m| (the K criterion at `thresh_k`).
///
/// FM, FTM: per shell, the sum over its rows of max_g |F|, max_g |Ft|; FW, FTW: of sum_g w |F|,
/// sum_g w |Ft|; FMK: max over rows and points of |F|. A pair whose derivative bound passes
/// `thresh_d` runs the derivative kernel (with G for the shells whose G bounds pass); a pair
/// that only needs G runs the energy kernel (a third of the cost).
pub(crate) struct DensScreen {
    /// Derivative factors (bra, ket centre) of every primitive pair (in the order of the
    /// pairs' `prims`), offsets per pair, and per pair an upper bound of the per-batch factor
    /// (2 max_q (dA_q + dB_q)), whether both shells sit on one atom, and la + lb.
    dfac: Vec<[f64; 2]>,
    doff: Vec<usize>,
    dfmax: Vec<f64>,
    same_atom: Vec<bool>,
    lsum: Vec<f64>,
    /// Shell of each basis function.
    ao2shell: Vec<usize>,
    /// Shell-block sums of |D_s T| (blocks 0..nd) and |M^T D_s T| (nd..2nd with fitting):
    /// [blk][s * nshell + a] = sum_{m in s} sum_{n in a (Cartesian)} |.|_mn.
    dsum: Vec<Vec<f64>>,
}

impl DensScreen {
    fn new(kb: &CosxK, blocks: &[&Vec<f64>]) -> Self {
        let basis = &kb.basis;
        let shells = &basis.shells;
        let ns = shells.len();
        let nc = kb.ncart;
        let mut dfac = Vec::new();
        let mut doff = Vec::with_capacity(kb.pairs.len() + 1);
        let mut dfmax = Vec::with_capacity(kb.pairs.len());
        let same_atom = kb
            .pairs
            .iter()
            .map(|p| shells[p.sa].atom == shells[p.sb].atom)
            .collect();
        let lsum = kb.pairs.iter().map(|p| (p.la + p.lb) as f64).collect();
        let dist = |a: &[f64; 3], b: &[f64; 3]| {
            ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
        };
        for p in &kb.pairs {
            doff.push(dfac.len());
            // bra: the shell with the larger l (PotPair's orientation; a1 is its exponent)
            let (bra, ket) = if p.la < p.lb {
                (&shells[p.sb], &shells[p.sa])
            } else {
                (&shells[p.sa], &shells[p.sb])
            };
            let mut m: f64 = 0.0;
            for q in &p.prims {
                let r = (0.5 / q.p).sqrt();
                let na = dist(&q.pc, &bra.center) + r;
                let nb = dist(&q.pc, &ket.center) + r;
                let fa = 2.0 * q.a1 * na + bra.l as f64 / na;
                let fb = 2.0 * (q.p - q.a1) * nb + ket.l as f64 / nb;
                dfac.push([fa, fb]);
                m = m.max(2.0 * (fa + fb));
            }
            dfmax.push(m);
        }
        doff.push(dfac.len());
        let mut ao2shell = vec![0; basis.nbf];
        let mut c2shell = vec![0; nc];
        for (s, sh) in shells.iter().enumerate() {
            ao2shell[sh.ao_start..sh.ao_start + sh.nfunc()]
                .iter_mut()
                .for_each(|v| *v = s);
            c2shell[kb.cart_off[s]..kb.cart_off[s] + linus_ints::basis::ncart(sh.l)]
                .iter_mut()
                .for_each(|v| *v = s);
        }
        let dsum = blocks
            .iter()
            .map(|src| {
                let mut o = vec![0.0; ns * ns];
                for m in 0..basis.nbf {
                    let row = &mut o[ao2shell[m] * ns..(ao2shell[m] + 1) * ns];
                    for (n, &v) in src[m * nc..(m + 1) * nc].iter().enumerate() {
                        row[c2shell[n]] += v.abs();
                    }
                }
                o
            })
            .collect();
        DensScreen {
            dfac,
            doff,
            dfmax,
            same_atom,
            lsum,
            ao2shell,
            dsum,
        }
    }
}

/// Values and gradients of the basis functions near a batch (SIMD collocation with screening),
/// laid out as phi[(c * nx + i) * np + g] (c = value, d/dx, d/dy, d/dz); the kept functions'
/// indices and atoms go to `xidx` and `xatom`. `xp`, `yp`, `zp` are the points padded to `ld`.
#[allow(clippy::too_many_arguments)]
fn collocate(
    coll: &Collocator,
    near: &[usize],
    np: usize,
    ld: usize,
    xp: &[f64],
    yp: &[f64],
    zp: &[f64],
    cbuf: &mut Vec<f64>,
    phi: &mut Vec<f64>,
    xidx: &mut Vec<usize>,
    xatom: &mut Vec<usize>,
) -> usize {
    let (lo, hi) = linus_grid::bbox(&xp[..np], &yp[..np], &zp[..np]);
    let ncand: usize = near.iter().map(|&s| coll.shells[s].nf).sum();
    cbuf.resize(ncand * 4 * ld, 0.0);
    let nx = coll.eval(near, xp, yp, zp, np, ld, &lo, &hi, cbuf, xidx, xatom);
    phi.resize(4 * nx * np, 0.0);
    for i in 0..nx {
        for c in 0..4 {
            phi[(c * nx + i) * np..(c * nx + i + 1) * np]
                .copy_from_slice(&cbuf[(i * 4 + c) * ld..(i * 4 + c) * ld + np]);
        }
    }
    nx
}

/// Per-batch outputs of the main pass: per point (d/dr_g x, y, z; e_K).
struct BatchOut {
    gpt: Vec<f64>,
    e: Vec<f64>,
}

struct Ctx<'a> {
    kb: &'a CosxK,
    grid: &'a MolecularGrid,
    nd: usize,
    /// row-major D_s T (nbf x ncart: F in Cartesian rows, see [`CosxK`]'s layout)
    d: Vec<Vec<f64>>,
    /// row-major M^T D_s T (fitting)
    mtd: Option<Vec<Vec<f64>>>,
    kappa: &'a [f64],
    natom: usize,
    opts: &'a CosxGradientConfig,
    coll: &'a Collocator,
    prof: Option<ClassProfile>,
    /// Density-weighted screening data (with [`CosxGradientConfig::thresh_d`]).
    dscr: Option<DensScreen>,
    /// Profile of the integral-only (energy-kernel) blocks.
    prof_g: Option<ClassProfile>,
    /// Thread-nanoseconds of the main pass's phases (see [`PHASES`]).
    phase: [std::sync::atomic::AtomicU64; 6],
}

/// Phases of the main pass, timed per batch (thread-seconds in the result's timers):
/// collocation, F GEMMs, screening bounds, pair loop (screening and integrals), energy density
/// and basis-function term, K'.
const PHASES: [&str; 6] = [
    "cosx_grad_main_coll",
    "cosx_grad_main_fgemm",
    "cosx_grad_main_bounds",
    "cosx_grad_main_pairs",
    "cosx_grad_main_ao",
    "cosx_grad_main_kp",
];

impl Ctx<'_> {
    /// The per-shell quantities of the density-weighted screening ([`DensScreen`]) for a batch
    /// whose F blocks are in `scr.f` and per-shell maxima of sum_g w |grad phi| in
    /// `scr.sh[SH_GW]`; the receiving factors include the thresholds (a bound >= 1 passes).
    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::needless_range_loop, reason = "indexes several arrays in step")]
    fn shell_bounds(
        &self,
        fblk: &[f64],
        sh: &mut [Vec<f64>; NSH],
        nearsh: &mut Vec<usize>,
        tmp: &mut Vec<f64>,
        ld: usize,
        np: usize,
        w: &[f64],
        xsv: f64,
    ) {
        let dsc = self.dscr.as_ref().unwrap();
        let kb = self.kb;
        let shells = &kb.basis.shells;
        let ns = shells.len();
        let nc = kb.ncart;
        let nd = self.nd;
        let fit = self.mtd.is_some();
        let ft0 = if fit { nd } else { 0 };
        let dstr = nc * ld;
        let kap: f64 = self.kappa.iter().map(|k| k.abs()).sum();
        let td = self.opts.thresh_d.unwrap();
        for v in [SH_FM, SH_FTM, SH_FW, SH_FTW, SH_FMK, SH_R1, SH_R2] {
            sh[v].clear();
            sh[v].resize(ns, 0.0);
        }
        // F side
        for (s, shl) in shells.iter().enumerate() {
            let r0 = kb.cart_off[s];
            let nr = linus_ints::basis::ncart(shl.l);
            let mut q = [0.0f64; 5];
            for k in 0..nd {
                let mut t = [0.0f64; 5];
                for n in r0..r0 + nr {
                    let f = &fblk[k * dstr + n * ld..k * dstr + n * ld + np];
                    let ft = &fblk[(ft0 + k) * dstr + n * ld..(ft0 + k) * dstr + n * ld + np];
                    let (mut m1, mut m2, mut s1, mut s2) = (0.0f64, 0.0f64, 0.0, 0.0);
                    for g in 0..np {
                        let (a, b) = (f[g].abs(), ft[g].abs());
                        m1 = m1.max(a);
                        m2 = m2.max(b);
                        s1 += w[g].abs() * a;
                        s2 += w[g].abs() * b;
                    }
                    t[0] += m1;
                    t[1] += m2;
                    t[2] += s1;
                    t[3] += s2;
                    t[4] = t[4].max(m1);
                }
                for i in 0..5 {
                    q[i] = q[i].max(t[i]);
                }
            }
            sh[SH_FM][s] = q[0];
            sh[SH_FTM][s] = q[1];
            sh[SH_FW][s] = q[2];
            sh[SH_FTW][s] = q[3];
            // K' (fitting only): the K criterion at thresh
            sh[SH_FMK][s] = if fit {
                q[4] * xsv / self.opts.thresh_k
            } else {
                0.0
            };
        }
        // basis-function term: AO_a <= sum_{s near} dsum[s][a] max_{k in s} sum_g w |grad phi_k|
        nearsh.clear();
        nearsh.extend((0..ns).filter(|&s| sh[SH_GW][s] > 0.0));
        tmp.resize(ns, 0.0);
        for blk in 0..(if fit { 2 * nd } else { nd }) {
            tmp.iter_mut().for_each(|v| *v = 0.0);
            for &s in nearsh.iter() {
                let gw = sh[SH_GW][s];
                let row = &dsc.dsum[blk][s * ns..(s + 1) * ns];
                for (t, &d) in tmp.iter_mut().zip(row) {
                    *t += d * gw;
                }
            }
            // blocks < nd: D (pairs with Gt: R2); fitting, blocks >= nd: M^T D (pairs with G: R1)
            let dst = if blk < nd { SH_R2 } else { SH_R1 };
            for (a, &t) in tmp.iter().enumerate() {
                sh[dst][a] = sh[dst][a].max(t);
            }
        }
        let wf = if self.opts.weight_derivs {
            self.opts.wfac
        } else {
            0.0
        };
        let c = kap / (td * self.opts.g_factor);
        for a in 0..ns {
            let ao = sh[SH_R2][a];
            let aot = if fit { sh[SH_R1][a] } else { ao };
            sh[SH_R1][a] = c * (aot + wf * sh[SH_FTW][a]);
            sh[SH_R2][a] = c * ao;
        }
    }

    /// Main pass over one batch: centre and AO terms into `gacc` (parts 0 and 1), K' into
    /// `kacc`; returns the per-point derivatives and energy density.
    #[allow(clippy::needless_range_loop, reason = "indexes several arrays in step")]
    fn batch(
        &self,
        b: &Batch,
        scr: &mut Scr,
        gacc: &ThreadAccs,
        kacc: Option<&ThreadAccs>,
        zacc: Option<&ThreadAccs>,
        cnt: &mut [u64; 6],
    ) -> BatchOut {
        let kb = self.kb;
        let basis = &kb.basis;
        let shells = &basis.shells;
        let nbf = basis.nbf;
        let nc = kb.ncart;
        let coff = &kb.cart_off;
        let ns = shells.len();
        let nd = self.nd;
        let fit = self.mtd.is_some();
        let nblk = if fit { 2 * nd } else { nd };
        let natom = self.natom;
        let np = b.end - b.start;
        let grid = self.grid;
        let (x, y, z) = (
            &grid.x[b.start..b.end],
            &grid.y[b.start..b.end],
            &grid.z[b.start..b.end],
        );
        let w = &grid.w[b.start..b.end];
        let mut out = BatchOut {
            gpt: vec![0.0; 3 * np],
            e: vec![0.0; np],
        };
        let mut tp = [Instant::now(); 7];
        let mut near = std::mem::take(&mut scr.near);
        kb.near_shells(b, &mut near);
        let ld = np.div_ceil(8) * 8;
        for (dst, src) in [(&mut scr.xp, x), (&mut scr.yp, y), (&mut scr.zp, z)] {
            dst.clear();
            dst.extend_from_slice(src);
            dst.resize(ld, src[np - 1]);
        }
        let nx = collocate(
            self.coll,
            &near,
            np,
            ld,
            &scr.xp,
            &scr.yp,
            &scr.zp,
            &mut scr.cbuf,
            &mut scr.phi,
            &mut scr.xidx,
            &mut scr.xatom,
        );
        scr.near = near;
        if nx == 0 {
            return out;
        }
        let phi = &scr.phi;
        scr.xw.resize(nx * np, 0.0);
        let mut xs: f64 = 0.0;
        // max_m sum_g |w phi_m| (K criterion of K') and per shell max_k sum_g w |grad phi_k|
        let mut xsv: f64 = 0.0;
        if self.dscr.is_some() {
            scr.sh[SH_GW].clear();
            scr.sh[SH_GW].resize(ns, 0.0);
        }
        for i in 0..nx {
            let (mut sv, mut sg) = (0.0, 0.0);
            for g in 0..np {
                let v = phi[i * np + g] * w[g];
                scr.xw[i * np + g] = v;
                sv += v.abs();
                let gr = phi[(nx + i) * np + g].abs()
                    + phi[(2 * nx + i) * np + g].abs()
                    + phi[(3 * nx + i) * np + g].abs();
                sg += gr * w[g].abs();
            }
            xs = xs.max(sv).max(sg);
            xsv = xsv.max(sv);
            if let Some(dsc) = &self.dscr {
                let s = dsc.ao2shell[scr.xidx[i]];
                scr.sh[SH_GW][s] = scr.sh[SH_GW][s].max(sg);
            }
        }
        tp[1] = Instant::now();
        // F blocks: D_s phi, then (D_s M) phi (rows of (M^T D_s))
        scr.wp.clear();
        scr.wp.extend_from_slice(w);
        scr.wp.resize(ld, 0.0);
        let ds = nc * ld;
        scr.f.clear();
        scr.f.resize(nblk * ds, 0.0);
        // the rows (D_s T)[X, :] and (M^T D_s T)[X, :] (kept for the basis-function term)
        scr.dxt.resize(nblk * nx * nc, 0.0);
        for k in 0..nblk {
            let src = if k < nd {
                &self.d[k]
            } else {
                &self.mtd.as_ref().unwrap()[k - nd]
            };
            let dxt = &mut scr.dxt[k * nx * nc..(k + 1) * nx * nc];
            for (i, &m) in scr.xidx.iter().enumerate() {
                dxt[i * nc..(i + 1) * nc].copy_from_slice(&src[m * nc..(m + 1) * nc]);
            }
            let dst = faer::MatMut::from_column_major_slice_with_stride_mut(
                &mut scr.f[k * ds..(k + 1) * ds],
                np,
                nc,
                ld,
            )
            .transpose_mut();
            let lhs =
                MatRef::from_row_major_slice(&scr.dxt[k * nx * nc..(k + 1) * nx * nc], nx, nc)
                    .transpose();
            let rhs = MatRef::from_row_major_slice(&phi[..nx * np], nx, np);
            linus_base::linalg::matmul(dst, Accum::Replace, lhs, rhs, 1.0, Par::Seq);
        }
        tp[2] = Instant::now();
        if self.dscr.is_none() {
            scr.fmax.clear();
            scr.fmax.resize(ns, 0.0);
            for (s, sh) in shells.iter().enumerate() {
                let mut m: f64 = 0.0;
                let n = linus_ints::basis::ncart(sh.l);
                for k in 0..nblk {
                    let blk = &scr.f[k * ds + coff[s] * ld..k * ds + (coff[s] + n) * ld];
                    m = blk.iter().fold(m, |a, v| a.max(v.abs()));
                }
                scr.fmax[s] = m;
            }
        } else {
            self.shell_bounds(
                &scr.f,
                &mut scr.sh,
                &mut scr.nearsh,
                &mut scr.tmp,
                ld,
                np,
                w,
                xsv,
            );
            let sh = &scr.sh;
            scr.shp.clear();
            scr.shp.extend((0..ns).map(|s| {
                [
                    sh[SH_FM][s],
                    sh[SH_FTM][s],
                    sh[SH_FW][s],
                    sh[SH_FTW][s],
                    sh[SH_FMK][s],
                    sh[SH_R1][s],
                    sh[SH_R2][s],
                    0.0,
                ]
            }));
        }
        tp[3] = Instant::now();
        // pair loop: centre derivatives, point derivatives, G and Gt
        scr.g.clear();
        scr.g.resize(nblk * ds, 0.0);
        scr.touched.clear();
        scr.touched.resize(ns, false);
        scr.gpt.clear();
        scr.gpt.resize(3 * ld, 0.0);
        scr.loc.clear();
        scr.loc.resize(2 * natom * 3, 0.0);
        let terms_off: Vec<(usize, usize, f64)> = (0..nd)
            .map(|s| (if fit { nd + s } else { s }, s, self.kappa[s]))
            .collect();
        let terms_diag: Vec<(usize, usize, f64)> =
            terms_off.iter().map(|&(i, j, c)| (i, j, 0.5 * c)).collect();
        let lim = self.opts.thresh / xs.max(1e-300);
        let op = &self.opts.xop;
        let mut cls = [[0u64; 3]; 25];
        let mut cls_g = [[0u64; 3]; 25];
        // derivative bound factor sum_s |kappa_s| / thresh_d
        let kd = self
            .opts
            .thresh_d
            .map(|t| self.kappa.iter().map(|k| k.abs()).sum::<f64>() / t)
            .unwrap_or(0.0);
        for (ip, ps) in kb.pair_scr.iter().enumerate() {
            let (sa, sb) = (ps.sa as usize, ps.sb as usize);
            // decide: derivative kernel (need_d), G of sa / sb (and whether Gt is needed)
            let (need_d, do_a, do_b, full_g);
            let p;
            match &self.dscr {
                None => {
                    let (fa, fb) = (scr.fmax[sa], scr.fmax[sb]);
                    let fm = fa.max(fb);
                    if ps.est * fm < lim {
                        continue;
                    }
                    p = &kb.pairs[ip];
                    let cut = lim / fm;
                    scr.prims.clear();
                    let est = if self.opts.dist_screen {
                        p.screen_prims(b.center, b.radius, cut, &mut scr.prims)
                    } else {
                        let n = p.nprim_above(cut);
                        scr.prims.extend_from_slice(&p.prims[..n]);
                        p.est
                    };
                    if est * fm < lim || scr.prims.is_empty() {
                        continue;
                    }
                    need_d = true;
                    do_a = est * fb >= lim;
                    do_b = sa != sb && est * fa >= lim;
                    full_g = true;
                }
                Some(dsc) => {
                    // [fm, ftm, fw, ftw, fmk, r1, r2] of both shells
                    let (x, y) = (&scr.shp[sa], &scr.shp[sb]);
                    // pair-density bound per unit derivative estimate (thresholds folded in)
                    let bdp =
                        kd * ((x[3] * y[0]).min(x[1] * y[2]) + (x[2] * y[1]).min(x[0] * y[3]));
                    let ga = y[0] * x[5] + y[1] * x[6];
                    let ka = y[4];
                    let (gb, kb_) = if sa != sb {
                        (x[0] * y[5] + x[1] * y[6], x[4])
                    } else {
                        (0.0, 0.0)
                    };
                    let (fa, fb) = (ga.max(ka), gb.max(kb_));
                    let gmax = fa.max(fb);
                    let top = (dsc.dfmax[ip] * bdp).max(gmax);
                    if ps.est * top < 1.0 {
                        continue;
                    }
                    p = &kb.pairs[ip];
                    let dfs = &dsc.dfac[dsc.doff[ip]..dsc.doff[ip + 1]];
                    let (same, lsum) = (dsc.same_atom[ip], dsc.lsum[ip]);
                    scr.prims.clear();
                    let (mut e, mut ed) = (0.0, 0.0);
                    for (q, &[da, db]) in p.prims.iter().zip(dfs) {
                        if q.est * top < 1.0 {
                            break;
                        }
                        // distance of the batch from the primitive pair's centre (beyond the
                        // batch's sphere; 0 inside)
                        let r = if self.opts.dist_screen {
                            let d = ((q.pc[0] - b.center[0]).powi(2)
                                + (q.pc[1] - b.center[1]).powi(2)
                                + (q.pc[2] - b.center[2]).powi(2))
                            .sqrt();
                            (d - b.radius).max(0.0)
                        } else {
                            0.0
                        };
                        let f = if r > 0.0 {
                            (0.886_226_925_452_758 / (q.p.sqrt() * r)).min(1.0)
                        } else {
                            1.0
                        };
                        // derivative factor: the point derivative dC = -(dA + dB) is bounded by
                        // dA + dB and, beyond the charge distribution, by (la + lb + 1) / r
                        // (the field of its multipoles); dA = -(dB + dC), dB = -(dA + dC); one
                        // atom: the centre and point terms are dC and -dC
                        let dc = if r > 0.0 {
                            (da + db).min((lsum + 1.0) / r)
                        } else {
                            da + db
                        };
                        let df = if same {
                            2.0 * dc
                        } else {
                            da.min(db + dc) + db.min(da + dc) + dc
                        };
                        let eq = q.est * f;
                        if eq * (df * bdp).max(gmax) >= 1.0 {
                            scr.prims.push(*q);
                            e += eq;
                            ed += eq * df;
                        }
                    }
                    need_d = ed * bdp >= 1.0;
                    do_a = e * fa >= 1.0;
                    do_b = e * fb >= 1.0;
                    if !need_d && !do_a && !do_b {
                        continue;
                    }
                    // Gt only for the basis-function term
                    full_g = (do_a && e * ga >= 1.0) || (do_b && e * gb >= 1.0);
                }
            }
            scr.touched[p.sa] |= do_a;
            scr.touched[p.sb] |= do_b;
            let ng = if full_g { nblk } else { nd };
            let tp = (self.prof.is_some() || self.prof_g.is_some()).then(Instant::now);
            if need_d {
                let terms = if p.sa == p.sb {
                    &terms_diag
                } else {
                    &terms_off
                };
                let mut c = PotDerivContract {
                    f: &scr.f,
                    g: &mut scr.g,
                    ng,
                    dstride: ds,
                    ld,
                    a0: coff[p.sa],
                    b0: coff[p.sb],
                    do_a,
                    do_b,
                    terms,
                    w: &scr.wp,
                    gpt: &mut scr.gpt,
                    cart: true,
                };
                let r = p.contract_deriv(&scr.prims, op, &scr.xp, &scr.yp, &scr.zp, &mut c);
                cnt[0] += 1;
                cnt[1] += (scr.prims.len() * np) as u64;
                if let Some(tp) = tp {
                    let e = &mut cls[class_index(p.la, p.lb)];
                    e[0] += 1;
                    e[1] += (scr.prims.len() * np) as u64;
                    e[2] += tp.elapsed().as_nanos() as u64;
                }
                for (s, sh) in [(0, p.sa), (1, p.sb)] {
                    let a = shells[sh].atom;
                    for k in 0..3 {
                        scr.loc[a * 3 + k] += r[s][k];
                    }
                }
            } else {
                let mut c = PotContract {
                    f: &scr.f,
                    g: &mut scr.g,
                    nd: ng,
                    dstride: ds,
                    ld,
                    a0: coff[p.sa],
                    b0: coff[p.sb],
                    do_a,
                    do_b,
                    cart: true,
                };
                p.contract_prims(&scr.prims, op, &scr.xp, &scr.yp, &scr.zp, &mut c);
                cnt[2] += 1;
                cnt[3] += (scr.prims.len() * np) as u64;
                if let Some(tp) = tp {
                    let e = &mut cls_g[class_index(p.la, p.lb)];
                    e[0] += 1;
                    e[1] += (scr.prims.len() * np) as u64;
                    e[2] += tp.elapsed().as_nanos() as u64;
                }
            }
        }
        if let Some(cp) = &self.prof_g {
            cp.add(&cls_g);
        }
        tp[4] = Instant::now();
        tp[5] = tp[4];
        if let Some(cp) = &self.prof {
            cp.add(&cls);
        }
        for g in 0..np {
            for k in 0..3 {
                out.gpt[3 * g + k] = scr.gpt[k * ld + g];
            }
        }
        scr.tidx.clear();
        for (s, sh) in shells.iter().enumerate() {
            if scr.touched[s] {
                scr.tidx
                    .extend(coff[s]..coff[s] + linus_ints::basis::ncart(sh.l));
            }
        }
        let nt = scr.tidx.len();
        cnt[4] += nx as u64;
        cnt[5] += nt as u64;
        if nt > 0 {
            // energy density e_K(g) = sum_s kappa_s sum_n Ft_n G_n
            for s in 0..nd {
                let fb = if fit { nd + s } else { s };
                for &n in &scr.tidx {
                    let (fr, gr) = (
                        &scr.f[fb * ds + n * ld..fb * ds + n * ld + np],
                        &scr.g[s * ds + n * ld..s * ds + n * ld + np],
                    );
                    for g in 0..np {
                        out.e[g] += self.kappa[s] * fr[g] * gr[g];
                    }
                }
            }
            scr.gc.resize(nt * np, 0.0);
            scr.h.resize(nx * np, 0.0);
            // H = (M^T D)[X, :] G (+ D[X, :] Gt) over all Cartesian rows (G is zero on the
            // untouched ones): the rows gathered for F and G in place; AO term
            for s in 0..nd {
                let reps: &[(usize, bool)] = if fit {
                    &[(0, true), (1, false)]
                } else {
                    &[(0, false)]
                };
                for (step, &(_, use_mtd)) in reps.iter().enumerate() {
                    let gblk = if step == 0 { s } else { nd + s };
                    let dblk = if use_mtd { nd + s } else { s };
                    let dst = faer::MatMut::from_row_major_slice_mut(&mut scr.h[..nx * np], nx, np);
                    let a = MatRef::from_row_major_slice(
                        &scr.dxt[dblk * nx * nc..(dblk + 1) * nx * nc],
                        nx,
                        nc,
                    );
                    let bm = MatRef::from_column_major_slice_with_stride(
                        &scr.g[gblk * ds..(gblk + 1) * ds],
                        np,
                        nc,
                        ld,
                    )
                    .transpose();
                    linus_base::linalg::matmul(dst, Accum::Replace, a, bm, 1.0, Par::Seq);
                    let fac = if fit { 1.0 } else { 2.0 } * self.kappa[s];
                    for i in 0..nx {
                        let at = scr.xatom[i];
                        let mut acc = [0.0; 3];
                        for g in 0..np {
                            let c = fac * w[g] * scr.h[i * np + g];
                            for k in 0..3 {
                                let d = c * phi[((k + 1) * nx + i) * np + g];
                                acc[k] += d;
                                out.gpt[3 * g + k] += d;
                            }
                        }
                        for k in 0..3 {
                            scr.loc[(natom + at) * 3 + k] -= acc[k];
                        }
                    }
                }
            }
            tp[5] = Instant::now();
            // derivative-function fitting: Z^(d)[X, T] += (w d_d phi)[X] G[T]^T
            if let Some(zacc) = zacc {
                scr.kblk.resize(nx * nt, 0.0);
                scr.xd.resize(nx * np, 0.0);
                index_runs(&scr.tidx, &mut scr.runs);
                for s in 0..nd {
                    for (j, &n) in scr.tidx.iter().enumerate() {
                        scr.gc[j * np..(j + 1) * np]
                            .copy_from_slice(&scr.g[s * ds + n * ld..s * ds + n * ld + np]);
                    }
                    for d in 0..3 {
                        for i in 0..nx {
                            for g in 0..np {
                                scr.xd[i * np + g] = w[g] * phi[((d + 1) * nx + i) * np + g];
                            }
                        }
                        let dst = faer::MatMut::from_row_major_slice_mut(
                            &mut scr.kblk[..nx * nt],
                            nx,
                            nt,
                        );
                        let a = MatRef::from_row_major_slice(&scr.xd[..nx * np], nx, np);
                        let bm =
                            MatRef::from_row_major_slice(&scr.gc[..nt * np], nt, np).transpose();
                        linus_base::linalg::matmul(dst, Accum::Replace, a, bm, 1.0, Par::Seq);
                        zacc.with(|acc| {
                            acc.add_block(
                                (s * 3 + d) * nbf * nc,
                                nc,
                                &scr.xidx,
                                &scr.runs,
                                nt,
                                &scr.kblk[..nx * nt],
                            )
                        });
                    }
                }
                zacc.with(|acc| acc.batch_done());
            }
            // K'[X, T] += X G[T]^T (G in place, columns T only)
            if let Some(kacc) = kacc {
                scr.kblk.resize(nd * nx * nt, 0.0);
                index_runs(&scr.tidx, &mut scr.runs);
                for s in 0..nd {
                    let gall = MatRef::from_column_major_slice_with_stride(
                        &scr.g[s * ds..(s + 1) * ds],
                        np,
                        nc,
                        ld,
                    );
                    let kb_s = &mut scr.kblk[s * nx * nt..(s + 1) * nx * nt];
                    // one GEMM per run of consecutive touched rows (whole shells)
                    for &(j0, n0, len) in &scr.runs {
                        // (row-major nx x len with row stride nt, as the transpose of a column-major view)
                        let dst = faer::MatMut::from_column_major_slice_with_stride_mut(
                            &mut kb_s[j0..],
                            len,
                            nx,
                            nt,
                        )
                        .transpose_mut();
                        let a = MatRef::from_row_major_slice(&scr.xw[..nx * np], nx, np);
                        let bm = gall.subcols(n0, len);
                        linus_base::linalg::matmul(dst, Accum::Replace, a, bm, 1.0, Par::Seq);
                    }
                }
                kacc.with(|acc| {
                    for s in 0..nd {
                        acc.add_block(
                            s * nbf * nc,
                            nc,
                            &scr.xidx,
                            &scr.runs,
                            nt,
                            &scr.kblk[s * nx * nt..(s + 1) * nx * nt],
                        );
                    }
                    acc.batch_done();
                });
            }
        }
        gacc.with(|acc| {
            for (i, &v) in scr.loc.iter().enumerate() {
                if v != 0.0 {
                    acc.add1(i, v);
                }
            }
            acc.batch_done();
        });
        tp[6] = Instant::now();
        for k in 0..6 {
            self.phase[k].fetch_add(
                (tp[k + 1] - tp[k]).as_nanos() as u64,
                std::sync::atomic::Ordering::Relaxed,
            );
        }
        out
    }
}

/// Gradient of E_K = sum_s kappa_s B(D_s) on the grid `mgrid` (built at the current
/// geometry; it moves with the atoms). `dens`: spin densities (RKS: D_alpha only).
#[allow(clippy::needless_range_loop, reason = "indexes several arrays in step")]
pub fn exchange_gradient(
    basis: &BasisSet,
    mgrid: &MolecularGrid,
    dens: &[Mat<f64>],
    kappa: &[f64],
    opts: &CosxGradientConfig,
    par: Par,
) -> Result<CosxGradResult, LinusError> {
    let t_all = Instant::now();
    let mut timers = Timers::default();
    if let Some(s) = basis
        .shells
        .iter()
        .find(|s| s.l > linus_ints::points::grid_pot::RYS_LMAX)
    {
        return Err(LinusError::Unsupported(format!(
            "COSX gradients support up to l = {} (g functions), not l = {}",
            linus_ints::points::grid_pot::RYS_LMAX,
            s.l
        )));
    }
    assert_eq!(dens.len(), kappa.len());
    let nbf = basis.nbf;
    let nd = dens.len();
    let natom = mgrid
        .atom_z
        .len()
        .max(basis.shells.iter().map(|s| s.atom + 1).max().unwrap_or(0));
    let grid = mgrid;
    let npoints = grid.len();
    // (the pair list keeps the primitive pairs that can pass the smallest threshold)
    let thresh_min = opts.thresh_d.map_or(opts.thresh, |t| t.min(opts.thresh_k));
    let kb = timers.time("cosx_grad_setup", || {
        CosxK::new(
            basis,
            grid.clone(),
            CosxConfig {
                thresh: thresh_min,
                bf_cut: opts.bf_cut,
                xop: opts.xop,
                dist_screen: opts.dist_screen,
                ..Default::default()
            },
        )
    });
    // overlap fitting: M = S S_num^-1
    let fitm = if opts.fitting {
        let t = Instant::now();
        let snum = kb.numerical_overlap(grid, &Collocator::new(basis, 0, opts.bf_cut));
        timers.add("cosx_grad_snum", t.elapsed().as_secs_f64());
        let llt = linus_base::linalg::cholesky(snum.as_ref(), par).map_err(|_| {
            LinusError::Numerical(
                "COSX gradient: numerical overlap not positive definite (grid too small?)".into(),
            )
        })?;
        let mut s = Mat::zeros(nbf, nbf);
        let plan = linus_ints::OneEPlan::new(basis, linus_ints::DerivOrder::Zero);
        plan.overlap(&mut linus_base::linalg::ao_view_mut(s.as_mut()));
        let mt = linus_base::linalg::cholesky_solve(llt.as_ref(), s.as_ref(), par);
        let m = mt.transpose().to_owned();
        timers.add("cosx_grad_fit", t.elapsed().as_secs_f64());
        Some((llt, m))
    } else {
        None
    };
    // F, Ft in Cartesian rows: D T, M^T D T
    let tt = kb.tsph.as_ref();
    let nc = kb.ncart;
    let dflat: Vec<Vec<f64>> = dens
        .iter()
        .map(|d| times_t(&mat_to_rowmajor(d.as_ref()), nbf, tt, par))
        .collect();
    let mtd: Option<Vec<Vec<f64>>> = fitm.as_ref().map(|(_, m)| {
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
    let coll = timers.time("cosx_grad_setup", || Collocator::new(basis, 1, opts.bf_cut));
    let dscr = opts.thresh_d.map(|_| {
        timers.time("cosx_grad_setup", || {
            let mut blocks: Vec<&Vec<f64>> = dflat.iter().collect();
            if let Some(m) = &mtd {
                blocks.extend(m.iter());
            }
            DensScreen::new(&kb, &blocks)
        })
    });
    let ctx = Ctx {
        kb: &kb,
        grid,
        nd,
        d: dflat,
        mtd,
        kappa,
        natom,
        opts,
        coll: &coll,
        prof: opts.profile.then(ClassProfile::new),
        dscr,
        prof_g: opts.profile.then(ClassProfile::new),
        phase: Default::default(),
    };
    // main pass
    let t = Instant::now();
    let gacc = ThreadAccs::new(2 * natom * 3);
    let kacc = opts.fitting.then(|| ThreadAccs::new(nd * nbf * nc));
    let zacc = opts.deriv_fit.then(|| ThreadAccs::new(nd * 3 * nbf * nc));
    let outs: Vec<(BatchOut, [u64; 6])> = grid
        .batches
        .par_iter()
        .map_init(Scr::default, |scr, b| {
            let mut cnt = [0; 6];
            let o = ctx.batch(b, scr, &gacc, kacc.as_ref(), zacc.as_ref(), &mut cnt);
            (o, cnt)
        })
        .collect();
    timers.add("cosx_grad_main", t.elapsed().as_secs_f64());
    for (k, name) in PHASES.iter().enumerate() {
        timers.add(
            name,
            ctx.phase[k].load(std::sync::atomic::Ordering::Relaxed) as f64 * 1e-9,
        );
    }
    let gv = gacc.total();
    let mut parts = CosxGradParts {
        centres: to_atoms(&gv, natom, 0),
        ao: to_atoms(&gv, natom, 1),
        ..Default::default()
    };
    // per-point results in point order
    let mut e_pt = vec![0.0; npoints];
    let mut points = Gradient::zeros(natom);
    let mut cnt = [0u64; 6];
    let mut energy = 0.0;
    for (b, (o, c)) in grid.batches.iter().zip(&outs) {
        for k in 0..6 {
            cnt[k] += c[k];
        }
        for (g, i) in (b.start..b.end).enumerate() {
            e_pt[i] = o.e[g];
            energy += grid.w[i] * o.e[g];
            let a = mgrid.atom[i] as usize;
            for k in 0..3 {
                points[a][k] += o.gpt[3 * g + k];
            }
        }
    }
    drop(outs);
    // fitting: Y = S_num^-1 K' D, W = sum_s kappa_s sym(Y_s) for tr(dS Y), Zs = sum kappa (Z + Z^T)
    let mut zsym: Option<Vec<f64>> = None;
    parts.fit_s = vec![[0.0; 3]; natom];
    if let (Some((llt, m)), Some(kacc)) = (&fitm, kacc) {
        let t = Instant::now();
        let kvc = kacc.total();
        let mut wsum = Mat::<f64>::zeros(nbf, nbf);
        let mut zs = Mat::<f64>::zeros(nbf, nbf);
        for s in 0..nd {
            // K' = K'c T^T
            let kv = times_tt(&kvc[s * nbf * nc..(s + 1) * nbf * nc], nbf, tt, par);
            let kp = Mat::from_fn(nbf, nbf, |i, j| kv[i * nbf + j]);
            let kd = linus_base::linalg::mm(kp.as_ref(), dens[s].as_ref(), par);
            let y = linus_base::linalg::cholesky_solve(llt.as_ref(), kd.as_ref(), par);
            let z = linus_base::linalg::mm(y.as_ref(), m.as_ref(), par);
            let ks = kappa[s];
            wsum += Mat::from_fn(nbf, nbf, |i, j| 0.5 * ks * (y[(i, j)] + y[(j, i)]));
            zs += Mat::from_fn(nbf, nbf, |i, j| ks * (z[(i, j)] + z[(j, i)]));
        }
        let wflat = mat_to_rowmajor(wsum.as_ref());
        // (a deterministic parallel reduction: independent of the thread count)
        let t1 = Instant::now();
        let plan = linus_ints::OneEPlan::new(basis, linus_ints::DerivOrder::One);
        parts.fit_s = vec![[0.0; 3]; basis.natom()];
        plan.overlap_kinetic_grad(Some(&wflat), None, &mut parts.fit_s, &mut plan.workspace());
        timers.add("cosx_grad_sgrad", t1.elapsed().as_secs_f64());
        zsym = Some(mat_to_rowmajor(zs.as_ref()));
        timers.add("cosx_grad_fit", t.elapsed().as_secs_f64());
    }
    // second pass: numerical-overlap terms and weight derivatives
    parts.fit_snum = vec![[0.0; 3]; natom];
    parts.weights = vec![[0.0; 3]; natom];
    if zsym.is_some() || opts.weight_derivs {
        let t = Instant::now();
        // parts: 0 fit_snum, 1 points (S_num); per point the weight-derivative factor
        // e_K - e_S, contracted with the weight derivatives afterwards
        let acc2 = ThreadAccs::new(2 * natom * 3);
        let fpts: Vec<Vec<f64>> = grid
            .batches
            .par_iter()
            .map_init(Scr::default, |scr, b| {
                let np = b.end - b.start;
                let (x, y, z) = (
                    &grid.x[b.start..b.end],
                    &grid.y[b.start..b.end],
                    &grid.z[b.start..b.end],
                );
                let w = &grid.w[b.start..b.end];
                scr.loc.clear();
                scr.loc.resize(2 * natom * 3, 0.0);
                let mut fpt: Vec<f64> = e_pt[b.start..b.end].to_vec();
                if let Some(zf) = &zsym {
                    let mut near = std::mem::take(&mut scr.near);
                    kb.near_shells(b, &mut near);
                    let ld = np.div_ceil(8) * 8;
                    for (dst, src) in [(&mut scr.xp, x), (&mut scr.yp, y), (&mut scr.zp, z)] {
                        dst.clear();
                        dst.extend_from_slice(src);
                        dst.resize(ld, src[np - 1]);
                    }
                    let nx = collocate(
                        &coll,
                        &near,
                        np,
                        ld,
                        &scr.xp,
                        &scr.yp,
                        &scr.zp,
                        &mut scr.cbuf,
                        &mut scr.phi,
                        &mut scr.xidx,
                        &mut scr.xatom,
                    );
                    if nx > 0 {
                        scr.lhs.resize(nx * nx, 0.0);
                        for (i, &m) in scr.xidx.iter().enumerate() {
                            for (j, &n) in scr.xidx.iter().enumerate() {
                                scr.lhs[i * nx + j] = zf[m * nbf + n];
                            }
                        }
                        scr.h.resize(nx * np, 0.0);
                        {
                            let dst = faer::MatMut::from_row_major_slice_mut(
                                &mut scr.h[..nx * np],
                                nx,
                                np,
                            );
                            let a = MatRef::from_row_major_slice(&scr.lhs[..nx * nx], nx, nx);
                            let bm = MatRef::from_row_major_slice(&scr.phi[..nx * np], nx, np);
                            linus_base::linalg::matmul(dst, Accum::Replace, a, bm, 1.0, Par::Seq);
                        }
                        let phi = &scr.phi;
                        for i in 0..nx {
                            let at = scr.xatom[i];
                            let mut acc = [0.0; 3];
                            for g in 0..np {
                                let q = scr.h[i * np + g];
                                fpt[g] -= 0.5 * phi[i * np + g] * q;
                                let c = w[g] * q;
                                let own = mgrid.atom[b.start + g] as usize;
                                for k in 0..3 {
                                    let d = c * phi[((k + 1) * nx + i) * np + g];
                                    acc[k] += d;
                                    // point motion of phi_i in S_num: -tr(dS_num Z) -> -d
                                    scr.loc[(natom + own) * 3 + k] -= d;
                                }
                            }
                            for k in 0..3 {
                                scr.loc[at * 3 + k] += acc[k];
                            }
                        }
                    }
                    scr.near = near;
                }
                acc2.with(|acc| {
                    for (i, &v) in scr.loc.iter().enumerate() {
                        if v != 0.0 {
                            acc.add1(i, v);
                        }
                    }
                    acc.batch_done();
                });
                fpt
            })
            .collect();
        let v = acc2.total();
        parts.fit_snum = to_atoms(&v, natom, 0);
        points += to_atoms(&v, natom, 1);
        if opts.weight_derivs {
            // the partition's weight-derivative contraction (block-vectorized for Becke
            // weights; per-batch sums in batch order: thread-count independent)
            let f: Vec<f64> = fpts.concat();
            parts.weights = mgrid.weight_gradient_contract(&f);
        }
        timers.add("cosx_grad_pass2", t.elapsed().as_secs_f64());
    }
    parts.points = points.into_inner();
    if let Some(zacc) = zacc {
        let t = Instant::now();
        parts.ao_dfit = deriv_fit_ao(&kb, grid, &coll, &zacc.total(), dens, kappa, natom, par)?;
        timers.add("cosx_grad_dfit", t.elapsed().as_secs_f64());
    }
    let mut gradient = Gradient::from(parts.centres.clone());
    gradient += &parts.ao;
    gradient += &parts.fit_s;
    gradient += &parts.fit_snum;
    if opts.point_motion {
        gradient += &parts.points;
    }
    if opts.weight_derivs {
        gradient += &parts.weights;
    }
    timers.add("cosx_grad", t_all.elapsed().as_secs_f64());
    let classes = ctx.prof.as_ref().map(|c| c.stats()).unwrap_or_default();
    let classes_g = ctx.prof_g.as_ref().map(|c| c.stats()).unwrap_or_default();
    Ok(CosxGradResult {
        gradient: gradient.into_inner(),
        parts,
        energy,
        npoints,
        pair_batches: cnt[0],
        prim_points: cnt[1],
        g_pair_batches: cnt[2],
        g_prim_points: cnt[3],
        sum_nx: cnt[4],
        sum_nt: cnt[5],
        classes,
        classes_g,
        timers,
    })
}

/// The analytic overlaps of the derivative functions, S^(d)_ij = <d_d phi_i | d_d phi_j>
/// (= -d^2 S_ij / dA_d^2 of the pair's first centre), row-major nbf x nbf for d = x, y, z.
fn deriv_overlaps(basis: &BasisSet) -> [Vec<f64>; 3] {
    let nbf = basis.nbf;
    let ns = basis.shells.len();
    let blocks: Vec<(usize, usize, Vec<f64>)> = (0..ns)
        .into_par_iter()
        .flat_map_iter(|i| (0..=i).map(move |j| (i, j)))
        .map(|(i, j)| {
            let (a, b) = (&basis.shells[i], &basis.shells[j]);
            let nab = a.nfunc() * b.nfunc();
            let (mut so, mut to) = (vec![0.0; 9 * nab], vec![0.0; 9 * nab]);
            linus_ints::one_e::deriv::overlap_kinetic_deriv2_block(a, b, &mut so, &mut to);
            (i, j, so)
        })
        .collect();
    let mut out = [
        vec![0.0; nbf * nbf],
        vec![0.0; nbf * nbf],
        vec![0.0; nbf * nbf],
    ];
    for (i, j, so) in blocks {
        let (a, b) = (&basis.shells[i], &basis.shells[j]);
        let (na, nb) = (a.nfunc(), b.nfunc());
        for (d, o) in out.iter_mut().enumerate() {
            let blk = &so[(d * 3 + d) * na * nb..(d * 3 + d + 1) * na * nb];
            for ia in 0..na {
                for ib in 0..nb {
                    let v = -blk[ia * nb + ib];
                    let (m, n) = (a.ao_start + ia, b.ao_start + ib);
                    o[m * nbf + n] = v;
                    o[n * nbf + m] = v;
                }
            }
        }
    }
    out
}

/// [`CosxGradParts::ao_dfit`] from Z^(d)_s = (w d_d phi) G_s^T (nbf x ncart per spin and
/// direction, accumulated in the main pass): the numerical overlaps of the derivative
/// functions on the grid, M^(d) = S^(d) S_num^(d)^-1, and per atom A
/// -2 sum_s kappa_s sum_{mu on A} sum_nu (M^(d) Z^(d)_s T^T)_mu,nu D_s,mu,nu.
#[allow(clippy::too_many_arguments)]
fn deriv_fit_ao(
    kb: &CosxK,
    grid: &MolecularGrid,
    coll: &Collocator,
    zv: &[f64],
    dens: &[Mat<f64>],
    kappa: &[f64],
    natom: usize,
    par: Par,
) -> Result<Vec<[f64; 3]>, LinusError> {
    let basis = &kb.basis;
    let nbf = basis.nbf;
    let nc = kb.ncart;
    let nd = dens.len();
    // numerical overlaps of the derivative functions
    let acc = ThreadAccs::new(3 * nbf * nbf);
    grid.batches
        .par_iter()
        .for_each_init(Scr::default, |scr, b| {
            let np = b.end - b.start;
            let (x, y, z) = (
                &grid.x[b.start..b.end],
                &grid.y[b.start..b.end],
                &grid.z[b.start..b.end],
            );
            let w = &grid.w[b.start..b.end];
            let mut near = std::mem::take(&mut scr.near);
            kb.near_shells(b, &mut near);
            let ld = np.div_ceil(8) * 8;
            for (dst, src) in [(&mut scr.xp, x), (&mut scr.yp, y), (&mut scr.zp, z)] {
                dst.clear();
                dst.extend_from_slice(src);
                dst.resize(ld, src[np - 1]);
            }
            let nx = collocate(
                coll,
                &near,
                np,
                ld,
                &scr.xp,
                &scr.yp,
                &scr.zp,
                &mut scr.cbuf,
                &mut scr.phi,
                &mut scr.xidx,
                &mut scr.xatom,
            );
            scr.near = near;
            if nx == 0 {
                return;
            }
            index_runs(&scr.xidx, &mut scr.runs);
            scr.xd.resize(nx * np, 0.0);
            scr.kblk.resize(nx * nx, 0.0);
            for d in 0..3 {
                let dphi = &scr.phi[(d + 1) * nx * np..(d + 2) * nx * np];
                for i in 0..nx {
                    for g in 0..np {
                        scr.xd[i * np + g] = w[g] * dphi[i * np + g];
                    }
                }
                let dst = faer::MatMut::from_row_major_slice_mut(&mut scr.kblk[..nx * nx], nx, nx);
                let a = MatRef::from_row_major_slice(&scr.xd[..nx * np], nx, np);
                let bm = MatRef::from_row_major_slice(dphi, nx, np).transpose();
                linus_base::linalg::matmul(dst, Accum::Replace, a, bm, 1.0, Par::Seq);
                acc.with(|ac| {
                    ac.add_block(
                        d * nbf * nbf,
                        nbf,
                        &scr.xidx,
                        &scr.runs,
                        nx,
                        &scr.kblk[..nx * nx],
                    )
                });
            }
            acc.with(|ac| ac.batch_done());
        });
    let snv = acc.total();
    let sa = deriv_overlaps(basis);
    let mut g = vec![[0.0; 3]; natom];
    for d in 0..3 {
        let snum = Mat::from_fn(nbf, nbf, |i, j| {
            0.5 * (snv[d * nbf * nbf + i * nbf + j] + snv[d * nbf * nbf + j * nbf + i])
        });
        let llt = linus_base::linalg::cholesky(snum.as_ref(), par).map_err(|_| {
            LinusError::Numerical(format!(
                "COSX gradient: numerical overlap of the derivative functions ({}) not positive definite",
                ["x", "y", "z"][d]
            ))
        })?;
        let sd = Mat::from_fn(nbf, nbf, |i, j| sa[d][i * nbf + j]);
        // M^(d) = S^(d) S_num^-1 -> M^T = S_num^-1 S^(d)
        let mt = linus_base::linalg::cholesky_solve(llt.as_ref(), sd.as_ref(), par);
        for s in 0..nd {
            let zf = times_tt(
                &zv[(s * 3 + d) * nbf * nc..(s * 3 + d + 1) * nbf * nc],
                nbf,
                kb.tsph.as_ref(),
                par,
            );
            let zm = MatRef::from_row_major_slice(&zf, nbf, nbf);
            let wd = linus_base::linalg::mm(mt.transpose(), zm, par);
            for sh in &basis.shells {
                let mut v = 0.0;
                for mu in sh.ao_start..sh.ao_start + sh.nfunc() {
                    for nu in 0..nbf {
                        v += wd[(mu, nu)] * dens[s][(mu, nu)];
                    }
                }
                g[sh.atom][d] -= 2.0 * kappa[s] * v;
            }
        }
    }
    Ok(g)
}

#[cfg(test)]
mod tests {
    use super::*;
    use linus_grid::GridSpec;
    use linus_ints::basis::BasisDef;

    fn basis_at(xyz: &[[f64; 3]]) -> BasisSet {
        basis_for(&Z, xyz)
    }

    fn basis_for(z: &[u32], xyz: &[[f64; 3]]) -> BasisSet {
        let json = r#"{"name":"t","elements":{"8":{"electron_shells":[
          {"function_type":"gto","angular_momentum":[0],"exponents":["130.7","23.81","6.44"],"coefficients":[["0.154","0.535","0.444"]]},
          {"function_type":"gto","angular_momentum":[0],"exponents":["0.38"],"coefficients":[["1.0"]]},
          {"function_type":"gto","angular_momentum":[1],"exponents":["5.03","1.17","0.38"],"coefficients":[["0.156","0.607","0.392"]]},
          {"function_type":"gto","angular_momentum":[2],"exponents":["1.2"],"coefficients":[["1.0"]]}]},
          "1":{"electron_shells":[{"function_type":"gto","angular_momentum":[0],"exponents":["3.42","0.62"],"coefficients":[["0.15","0.53"]]},
          {"function_type":"gto","angular_momentum":[1],"exponents":["0.8"],"coefficients":[["1.0"]]}]}}}"#;
        BasisSet::new(&BasisDef::from_bse_json(json).unwrap(), z, xyz, true).unwrap()
    }

    const Z: [u32; 3] = [8, 1, 1];
    const XYZ: [[f64; 3]; 3] = [[0.0, 0.1, 0.0], [0.1, 1.43, 1.11], [0.0, -1.38, 1.05]];

    fn density(n: usize) -> Mat<f64> {
        let c = Mat::from_fn(n, 5, |i, j| {
            ((i * 7 + j * 3) as f64 * 0.37).sin() / (1.0 + i as f64 * 0.2)
        });
        &c * c.transpose()
    }

    /// E_K for fixed densities on a grid built at `xyz` (the grid moves with the atoms).
    fn energy(
        xyz: &[[f64; 3]],
        dens: &[Mat<f64>],
        kappa: &[f64],
        fitting: bool,
        spec: &GridSpec,
    ) -> f64 {
        energy_at_cut(xyz, dens, kappa, fitting, spec, 1e-15)
    }

    /// [`energy`] with the basis-function cut `bf_cut`.
    fn energy_at_cut(
        xyz: &[[f64; 3]],
        dens: &[Mat<f64>],
        kappa: &[f64],
        fitting: bool,
        spec: &GridSpec,
        bf_cut: f64,
    ) -> f64 {
        let basis = basis_at(xyz);
        let n = basis.nbf;
        let g = MolecularGrid::build(&Z, xyz, spec).unwrap();
        let mut k = CosxK::new(
            &basis,
            g,
            CosxConfig {
                thresh: 1e-15,
                bf_cut,
                ..Default::default()
            },
        );
        if fitting {
            let (s, _) = linus_ints::reference::oneshot::overlap_kinetic(&basis);
            k = k.with_overlap_fitting(Mat::from_fn(n, n, |i, j| s[i * n + j]).as_ref());
        }
        let kk = k.build(dens, Par::Seq);
        dens.iter()
            .zip(&kk)
            .zip(kappa)
            .map(|((d, k), c)| c * linus_base::linalg::dot(d.as_ref(), k.as_ref()))
            .sum()
    }

    /// All parts together (moving grid, weight derivatives) are the exact derivative of the
    /// fixed-density COSX energy, with and without overlap fitting, for one and two densities;
    /// and the gradient is translation invariant.
    #[test]
    fn matches_finite_differences_of_fixed_density_energy() {
        let spec = GridSpec::cosx(1);
        let basis = basis_at(&XYZ);
        let n = basis.nbf;
        let d1 = density(n);
        let d2 = Mat::from_fn(n, n, |i, j| {
            0.7 * d1[(i, j)] + 0.01 * ((i + j) as f64).cos()
        });
        for (dens, kappa) in [
            (vec![d1.clone()], vec![-1.0]),
            (vec![d1.clone(), d2.clone()], vec![-0.5, -0.5]),
        ] {
            for (fitting, thresh_d) in [
                (false, None),
                (true, None),
                (false, Some(1e-15)),
                (true, Some(1e-15)),
            ] {
                let mg = MolecularGrid::build(&Z, &XYZ, &spec).unwrap();
                let opts = CosxGradientConfig {
                    thresh: 1e-15,
                    thresh_k: 1e-15,
                    bf_cut: 1e-15,
                    fitting,
                    point_motion: true,
                    weight_derivs: true,
                    thresh_d,
                    ..Default::default()
                };
                let r = exchange_gradient(&basis, &mg, &dens, &kappa, &opts, Par::Seq).unwrap();
                let e0 = energy(&XYZ, &dens, &kappa, fitting, &spec);
                assert!((r.energy - e0).abs() < 1e-10, "energy {} vs {e0}", r.energy);
                let h = 1e-4;
                let mut worst: f64 = 0.0;
                for a in 0..3 {
                    for k in 0..3 {
                        let f = |s: f64| {
                            let mut c = XYZ;
                            c[a][k] += s;
                            energy(&c, &dens, &kappa, fitting, &spec)
                        };
                        let fd = (8.0 * (f(h) - f(-h)) - (f(2.0 * h) - f(-2.0 * h))) / (12.0 * h);
                        let e = (r.gradient[a][k] - fd).abs();
                        worst = worst.max(e);
                        eprintln!("nd={} fit={fitting} atom {a} dir {k}: analytic {:+.10e} fd {:+.10e} diff {e:.1e}  [centres {:+.2e} ao {:+.2e} points {:+.2e} weights {:+.2e} fit_s {:+.2e} fit_snum {:+.2e}]", dens.len(), r.gradient[a][k], fd, r.parts.centres[a][k], r.parts.ao[a][k], r.parts.points[a][k], r.parts.weights[a][k], r.parts.fit_s[a][k], r.parts.fit_snum[a][k]);
                    }
                }
                assert!(
                    worst < 1e-8,
                    "nd={} fitting={fitting}: worst {worst:e}",
                    dens.len()
                );
                for k in 0..3 {
                    let s: f64 = r.gradient.iter().map(|g| g[k]).sum();
                    assert!(s.abs() < 1e-10, "translation: {s:e}");
                }
            }
        }
    }

    /// With overlap fitting at the default basis-function cut (1e-11), where the S_num of the
    /// energy and of the gradient once differed (the energy's unscreened, the gradient's
    /// screened at the cut): the gradient against central differences of the energy at the
    /// same cut. Prints the worst difference (the EPOCH B9 measurement) and holds it to 1e-8.
    #[test]
    fn fitted_gradient_matches_the_energy_at_the_default_cut() {
        let spec = GridSpec::cosx(1);
        let basis = basis_at(&XYZ);
        let dens = vec![density(basis.nbf)];
        let kappa = vec![-1.0];
        let opts = CosxGradientConfig {
            thresh: 1e-15,
            thresh_k: 1e-15,
            fitting: true,
            point_motion: true,
            weight_derivs: true,
            ..Default::default()
        };
        let mg = MolecularGrid::build(&Z, &XYZ, &spec).unwrap();
        let r = exchange_gradient(&basis, &mg, &dens, &kappa, &opts, Par::Seq).unwrap();
        let cut = opts.bf_cut;
        let e0 = energy_at_cut(&XYZ, &dens, &kappa, true, &spec, cut);
        let h = 1e-4;
        let mut worst: f64 = 0.0;
        for a in 0..3 {
            for k in 0..3 {
                let f = |s: f64| {
                    let mut c = XYZ;
                    c[a][k] += s;
                    energy_at_cut(&c, &dens, &kappa, true, &spec, cut)
                };
                let fd = (8.0 * (f(h) - f(-h)) - (f(2.0 * h) - f(-2.0 * h))) / (12.0 * h);
                worst = worst.max((r.gradient[a][k] - fd).abs());
            }
        }
        eprintln!(
            "bf_cut {cut:e}: |E(gradient) - E| {:.1e}, worst |gradient - FD(E)| {worst:.3e}",
            (r.energy - e0).abs()
        );
        assert!(worst < 1e-8, "worst {worst:e}");
    }

    /// The density-weighted screening against the K criterion (four waters 7 bohr apart, so
    /// that the screening drops blocks): the same gradient at tight thresholds; at the defaults
    /// the difference stays far below the grid error.
    #[test]
    fn density_screening_matches_k_screening() {
        let mut z = vec![];
        let mut xyz = vec![];
        for m in 0..4 {
            z.extend_from_slice(&Z);
            xyz.extend(
                XYZ.iter()
                    .map(|c| [c[0] + 7.0 * m as f64, c[1] + 0.3 * m as f64, c[2]]),
            );
        }
        let spec = GridSpec::cosx(1);
        let basis = basis_for(&z, &xyz);
        let n = basis.nbf;
        // a density decaying with the distance between the functions' atoms
        let at: Vec<usize> = (0..n)
            .map(|i| {
                basis
                    .shells
                    .iter()
                    .find(|s| i >= s.ao_start && i < s.ao_start + s.nfunc())
                    .unwrap()
                    .atom
            })
            .collect();
        let c = Mat::from_fn(n, 20, |i, j| {
            ((i * 7 + j * 3) as f64 * 0.37).sin() / (1.0 + i as f64 * 0.05)
                * (-0.5 * (at[i] as f64 / 3.0 - (j / 5) as f64).powi(2)).exp()
        });
        let d1 = &c * c.transpose();
        let d2 = Mat::from_fn(n, n, |i, j| {
            0.7 * d1[(i, j)]
                + 0.01
                    * ((i + j) as f64).cos()
                    * (-((at[i] as f64 - at[j] as f64) / 3.0).powi(2)).exp()
        });
        let mg = MolecularGrid::build(&z, &xyz, &spec).unwrap();
        for (dens, kappa) in [
            (vec![d1.clone()], vec![-0.2]),
            (vec![d1.clone(), d2.clone()], vec![-0.1, -0.1]),
        ] {
            for fitting in [false, true] {
                let run = |thresh: f64, thresh_d: Option<f64>| {
                    exchange_gradient(
                        &basis,
                        &mg,
                        &dens,
                        &kappa,
                        &CosxGradientConfig {
                            thresh,
                            thresh_k: thresh,
                            thresh_d,
                            fitting,
                            ..Default::default()
                        },
                        Par::Seq,
                    )
                    .unwrap()
                };
                let diff = |a: &CosxGradResult, b: &CosxGradResult| {
                    a.gradient
                        .iter()
                        .zip(&b.gradient)
                        .flat_map(|(x, y)| (0..3).map(move |k| (x[k] - y[k]).abs()))
                        .fold(0.0f64, f64::max)
                };
                let reference = run(1e-15, None);
                let tight = run(1e-15, Some(1e-16));
                let dflt = exchange_gradient(
                    &basis,
                    &mg,
                    &dens,
                    &kappa,
                    &CosxGradientConfig {
                        fitting,
                        ..Default::default()
                    },
                    Par::Seq,
                )
                .unwrap();
                let (e1, e2) = (diff(&tight, &reference), diff(&dflt, &reference));
                eprintln!("nd={} fit={fitting}: tight {e1:.1e}, defaults {e2:.1e} ({} derivative + {} integral blocks of {})", dens.len(), dflt.pair_batches, dflt.g_pair_batches, reference.pair_batches);
                assert!(
                    e1 < 1e-12 && e2 < 1e-8,
                    "nd={} fit={fitting}: {e1:e} {e2:e}",
                    dens.len()
                );
                assert!(dflt.pair_batches < reference.pair_batches);
            }
        }
    }

    #[test]
    fn deterministic_across_thread_counts() {
        let spec = GridSpec::cosx(1);
        let basis = basis_at(&XYZ);
        let n = basis.nbf;
        let dens = vec![density(n)];
        let mg = MolecularGrid::build(&Z, &XYZ, &spec).unwrap();
        let run = |t: usize| {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(t)
                .build()
                .unwrap();
            pool.install(|| {
                (
                    exchange_gradient(
                        &basis,
                        &mg,
                        &dens,
                        &[-1.0],
                        &CosxGradientConfig {
                            weight_derivs: true,
                            ..Default::default()
                        },
                        Par::Seq,
                    )
                    .unwrap(),
                    exchange_gradient(
                        &basis,
                        &mg,
                        &dens,
                        &[-1.0],
                        &CosxGradientConfig {
                            thresh_d: None,
                            ..Default::default()
                        },
                        Par::Seq,
                    )
                    .unwrap(),
                )
            })
        };
        let (a, b) = (run(1), run(5));
        for (ra, rb) in [(&a.0, &b.0), (&a.1, &b.1)] {
            for (x, y) in ra.gradient.iter().zip(&rb.gradient) {
                for k in 0..3 {
                    assert_eq!(x[k].to_bits(), y[k].to_bits());
                }
            }
        }
    }
}
