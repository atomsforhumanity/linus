//! VV10 nonlocal correlation in linear responses and analytic Hessians: the CPKS kernel (the
//! second functional derivative of E_nl contracted with a first-order density) and the explicit
//! Hessian terms with the first-order response of the NLC grid, on the NLC grid of
//! [`crate::vv10`] (which has the energy, potential and gradient).
//!
//! **Energy and potential.** With n_i = w_i rho_i on the active points (rho_i >= 1e-8),
//!   E = beta sum_i n_i + 1/2 sum_ij n_i n_j Phi(g_i, g_j),  Phi = -3/2 / (g_i g_j (g_i + g_j)),
//!   g_i = omega_i R_ij^2 + kappa_i  (omega, kappa: functions of rho_i and sigma_i),
//!   v_rho = beta + F + rho kappa' S1 + rho omega_rho S2,  v_sigma = rho omega_sigma S2,
//!   F_i = sum_j n_j Phi_ij,  S1_i = sum_j n_j Phi_gi,  S2_i = sum_j n_j R_ij^2 Phi_gi
//! (Phi_gi = dPhi/dg_i, Phi_gigj = d2Phi/dg_i dg_j, ...; one division per pair gives them all).
//!
//! **First-order potential.** A perturbation changes rho_j and grad rho_j (so sigma_j, omega_j and
//! kappa_j) and, for the grid response, the weights w_j and the positions r_j. With
//! alpha_j = dw_j rho_j + w_j drho_j, b_j = n_j dkappa_j and c_j = n_j domega_j, the nonlocal sums
//!   P1_i = sum_j [alpha_j Phi + (b_j + R^2 c_j) Phi_gj],
//!   P2_i = sum_j [alpha_j Phi_gi + (b_j + R^2 c_j) Phi_gigj],  P3_i = sum_j R^2 [the same]
//! (one pass over the pairs per group of perturbations: the pair functions are computed once per
//! pair and shared by all perturbations, 7 operations per pair and perturbation), the
//! ground-state sums S3..S5 = sum_j n_j (1, R^2, R^4) Phi_gigi and, when points move, the
//! geometric sums G_k = sum_j n_j g_k dR_ij^2 (dR_ij^2 = 2 (r_i - r_j).(dr_i - dr_j);
//! g_1 = omega_j Phi_gj + omega_i Phi_gi, g_2 = omega_j Phi_gigj + omega_i Phi_gigi,
//! g_3 = R^2 g_2 + Phi_gi) give
//!   dF = P1 + dkappa_i S1 + domega_i S2 + G1,  dS1 = P2 + dkappa_i S3 + domega_i S4 + G2,
//!   dS2 = P3 + dkappa_i S4 + domega_i S5 + G3,
//!   dv_rho = dF + d(rho kappa') S1 + rho kappa' dS1 + d(rho omega_rho) S2 + rho omega_rho dS2,
//!   dv_sigma = d(rho omega_sigma) S2 + rho omega_sigma dS2,
//! and the first-order potential in the potential-matrix form of a GGA,
//! t = (dv_rho, 2 dv_sigma grad rho + 2 v_sigma d grad rho), weighted. The matrix
//! V\[t\]_mn = sum_i \[t_rho phi_m phi_n + t_k d_k(phi_m phi_n)\] is only needed projected on the
//! occupied orbitals (V C_s, the layout of `linus_scf::response::operator`).
//!
//! **CPKS kernel** ([`Vv10::kernel_contract_occ`]): the first-order total density of the
//! occupied-projected CPKS form (as `XcIntegrator::kernel_contract_occ`) on the NLC grid, t, and
//! dV C_s for every channel; VV10 is a functional of the total density, so both spins share dV.
//!
//! **Hessian** ([`Vv10::hessian_terms`]), at the total density D on the NLC grid, in linus's
//! scheme-1 convention (the derivative of the gradient without grid response, the grid moving
//! with the atoms; `linus::derivatives::hessian`):
//! - the terms of the potential v itself (basis-function derivatives at fixed v; weight
//!   derivatives and the motion of the basis-function values at the moving points) from the
//!   semilocal Hessian code with v as an external potential
//!   (`hessian_terms_ch`, `grid_terms_ch` with one channel, the total density);
//! - the nonlocal kernel at a fixed grid: t^x for the density derivatives u^x of the basis
//!   functions moving with x gives V^x += V[t^x] and E^xy += sum_i u^x_i . t^y_i;
//! - with the grid response, t^y of the moving grid (weights, the density and its gradient at
//!   the moving points, the moving pair distances) gives dV^y += V[t^y] and
//!   E1^xy += sum_i u^x_i . t^y_i.
//!
//! Every reduction runs in a fixed order (the pair sums over j ascending for each point and
//! perturbation, lane-wise with explicit FMA as in `crate::vv10`) or in fixed point (the
//! matrices and the Hessian), so the results do not depend on the number of threads, on the
//! memory budget's perturbation groups or on the ISA. A group's products are parts of the single
//! group's (`GroupPerts`: blocks of perturbations aligned as in one group, each product with
//! its whole block's kernel, `linus_base::linalg::matmul_part`): faer adds a product's terms
//! differently by its shape, and the groups' sizes follow the memory budget and the thread
//! count.

use crate::batch::{BatchCtx, BatchPlan};
use crate::hessian::{basis_motion, comp_index, local_index, Fns, HBatch, XcHessianTerms, Xs};
use crate::kernel::{occ_out, occ_u1, push_vco, VcoInputs};
use crate::points::ExtPotential;
use crate::vv10::{constants, fuw, local, pair_g, potential, reciprocals, Local, Vv10};
use crate::{gather_rows, gather_sub, gemm, gemm_part, mmut, mref, scatter_cols, sized};
use faer::{Mat, MatRef, Par};
use linus_base::error::LinusError;
use linus_base::linalg::{mm, Whole};
use linus_num::det::{merge_all, FixedAcc};
use rayon::prelude::*;
use std::ops::Range;
use std::sync::Mutex;

/// Points i per block of the pair sweeps (a multiple of every lane count).
const IB: usize = 64;
/// j points per tile of the per-perturbation sweep (its pair functions stay in L1).
const JT: usize = 64;
/// j points per tile of the ground-state and geometric sweeps.
const GT: usize = 512;
/// Perturbations are padded to a multiple of this (the widest perturbation chunk).
const NPAD_P: usize = 4;

/// What the grid response needs besides the ground state of the kernel.
struct GridGround {
    /// The Hessian of rho (xx, xy, xz, yy, yz, zz) at the active points.
    hrho: Vec<[f64; 6]>,
    /// sum_j n_j g_k (r_i - r_j) (with their factors), [k][component].
    geo: Vec<[f64; 9]>,
    /// The weight derivatives of each active point (the point moving with its owner): CSR
    /// offsets and (atom, dw/dR_atom).
    wg_off: Vec<usize>,
    wg: Vec<(usize, [f64; 3])>,
    /// The active points owned by each atom (ascending).
    by_atom: Vec<Vec<usize>>,
}

/// A device engine for the per-perturbation pair sums (the O(n^2) part of the kernel and of the
/// explicit Hessian terms), set on a [`Vv10Ground`] by the backend.
pub trait Vv10PairSweep: Send + Sync {
    /// P1, P2, P3 without their factor 3/2 of `npp` perturbations from their coefficients
    /// `coef[(p 3 + c) ld + j]` (alpha, b, c at the active point j), as `[(c npp + p) npad + i]`
    /// with `npad` = [`Vv10PairSweep::npad`]; with `fp32` in FP32 (the CPKS kernel's products
    /// when the device's semilocal CPKS kernel runs FP32), else in the engine's arithmetic.
    fn sweep(
        &self,
        coef: &[f64],
        ld: usize,
        npp: usize,
        fp32: bool,
    ) -> Result<Vec<f64>, LinusError>;
    /// The padded number of points of the sweep's output.
    fn npad(&self) -> usize;
}

/// The ground state of VV10 for responses at a fixed total density: the active points of the
/// NLC grid and their pair sums ([`Vv10::response_ground`]).
pub struct Vv10Ground {
    /// NLC grid point of each active point (ascending).
    idx: Vec<usize>,
    /// The active positions of each batch of the NLC grid.
    brange: Vec<Range<usize>>,
    /// Padded (a multiple of [`IB`]) SoA of the pair sweeps: coordinates about the active
    /// points' centroid, -2 times them and |r|^2 (the pair primitive's,
    /// [`crate::vv10::pair_g`]), omega, kappa and n = w rho (0 on the padding,
    /// omega = kappa = 1).
    x: Vec<f64>,
    y: Vec<f64>,
    z: Vec<f64>,
    mx: Vec<f64>,
    my: Vec<f64>,
    mz: Vec<f64>,
    nn: Vec<f64>,
    om: Vec<f64>,
    ka: Vec<f64>,
    n: Vec<f64>,
    /// rho, grad rho, the weight and the owner atom of each active point.
    rho: Vec<f64>,
    grad: Vec<[f64; 3]>,
    w: Vec<f64>,
    own: Vec<usize>,
    loc: Vec<Local>,
    /// F, S1, S2, S3, S4, S5 of each active point (with their factors).
    s: Vec<[f64; 6]>,
    /// v_rho and v_sigma (scaled) at every NLC grid point (zero on the inactive ones).
    vrho: Vec<f64>,
    vsig: Vec<f64>,
    grid: Option<GridGround>,
    natom: usize,
    /// The VV10 term's scale factor (the functional's component weight).
    scale: f64,
    /// The pair sweeps on a device (None: the CPU's).
    pub device: Option<Box<dyn Vv10PairSweep>>,
}

impl Vv10Ground {
    /// Active points of the NLC grid.
    pub fn active_points(&self) -> usize {
        self.idx.len()
    }

    fn na(&self) -> usize {
        self.idx.len()
    }

    /// The active points' coordinates (about their centroid), omega and kappa (a device engine's
    /// inputs), and their number.
    pub fn points(&self) -> ([&[f64]; 5], usize) {
        let n = self.na();
        (
            [
                &self.x[..n],
                &self.y[..n],
                &self.z[..n],
                &self.om[..n],
                &self.ka[..n],
            ],
            n,
        )
    }

    fn npad(&self) -> usize {
        self.x.len()
    }
}

/// The total density of the occupied orbitals of a response: 2 C C^T for one channel (a closed
/// shell), C_a C_a^T + C_b C_b^T for two.
pub fn total_density(c_o: &[MatRef<'_, f64>], par: Par) -> Mat<f64> {
    let f = if c_o.len() == 1 { 2.0 } else { 1.0 };
    let mut d = Mat::<f64>::zeros(c_o[0].nrows(), c_o[0].nrows());
    for c in c_o {
        d += mm(*c, c.transpose(), par) * f;
    }
    d
}

/// The kernel's products `[s][p]` and where the time went.
pub type KernelProducts = (Vec<Vec<Mat<f64>>>, Vv10Times);

/// The explicit VV10 terms of a Hessian ([`Vv10::hessian_terms`]).
pub struct Vv10Hessian {
    /// V^x C_s at a fixed grid, with the grid response dV^y C_s, and in `hess` the explicit
    /// second derivative plus (with the grid response) the grid's first-order terms (row =
    /// gradient component, column = displaced coordinate).
    pub terms: XcHessianTerms,
    /// Where the time went.
    pub times: Vv10Times,
}

/// Wall seconds of the VV10 response parts: the potential's own terms (the semilocal code with v
/// as an external potential), the batch passes over the NLC grid (first-order densities, the
/// contractions), the pair sweeps, and the perturbations swept.
#[derive(Clone, Copy, Debug, Default)]
pub struct Vv10Times {
    pub potential_terms: f64,
    pub grid: f64,
    pub pairs: f64,
    pub perturbations: usize,
}

impl Vv10Times {
    fn add(&mut self, o: &Vv10Times) {
        self.potential_terms += o.potential_terms;
        self.grid += o.grid;
        self.pairs += o.pairs;
        self.perturbations += o.perturbations;
    }
}

impl Vv10 {
    /// The ground state of the responses at the total density `dt`; with `grid_terms` also what
    /// the grid response of a Hessian needs (the Hessian of rho, the geometric sums and the
    /// weight derivatives).
    pub fn response_ground(&self, dt: &Mat<f64>, grid_terms: bool) -> Vv10Ground {
        let grid = self.grid();
        let npt = grid.len();
        let natom = grid.atom_z.len();
        let d = self.densities(dt);
        let idx: Vec<usize> = (0..npt).filter(|&i| d.rho[i] >= self.rho_thresh).collect();
        let na = idx.len();
        let npad = na.div_ceil(IB).max(1) * IB;
        let brange = grid
            .batches
            .iter()
            .map(|b| idx.partition_point(|&i| i < b.start)..idx.partition_point(|&i| i < b.end))
            .collect();
        let (beta, kvv) = constants(&self.params);
        let cvv = self.params.c;
        // centroid of the active points (sequential sums: deterministic)
        let mut c = [0.0; 3];
        for &i in &idx {
            c[0] += grid.x[i];
            c[1] += grid.y[i];
            c[2] += grid.z[i];
        }
        let c = c.map(|v| if na > 0 { v / na as f64 } else { 0.0 });
        let mut g = Vv10Ground {
            brange,
            x: vec![0.0; npad],
            y: vec![0.0; npad],
            z: vec![0.0; npad],
            mx: vec![0.0; npad],
            my: vec![0.0; npad],
            mz: vec![0.0; npad],
            nn: vec![0.0; npad],
            om: vec![1.0; npad],
            ka: vec![1.0; npad],
            n: vec![0.0; npad],
            rho: vec![0.0; na],
            grad: vec![[0.0; 3]; na],
            w: vec![0.0; na],
            own: vec![0; na],
            loc: vec![Local::default(); na],
            s: vec![],
            vrho: vec![0.0; npt],
            vsig: vec![0.0; npt],
            grid: None,
            natom,
            scale: self.params.scale,
            device: None,
            idx,
        };
        for p in 0..na {
            let i = g.idx[p];
            let r = d.rho[i];
            let gr = [d.grad[0][i], d.grad[1][i], d.grad[2][i]];
            let s = gr[0] * gr[0] + gr[1] * gr[1] + gr[2] * gr[2];
            let (om, ka, loc) = local(r, s, cvv, kvv);
            g.loc[p] = loc;
            g.x[p] = grid.x[i] - c[0];
            g.y[p] = grid.y[i] - c[1];
            g.z[p] = grid.z[i] - c[2];
            g.mx[p] = -2.0 * g.x[p];
            g.my[p] = -2.0 * g.y[p];
            g.mz[p] = -2.0 * g.z[p];
            g.nn[p] = g.x[p] * g.x[p] + g.y[p] * g.y[p] + g.z[p] * g.z[p];
            g.om[p] = om;
            g.ka[p] = ka;
            g.n[p] = grid.w[i] * r;
            g.rho[p] = r;
            g.grad[p] = gr;
            g.w[p] = grid.w[i];
            g.own[p] = grid.atom[i] as usize;
        }
        let sums = ground_sweep(&g, grid_terms);
        g.s = sums
            .iter()
            .map(|a| {
                [
                    -1.5 * a[0],
                    1.5 * a[1],
                    1.5 * a[2],
                    -1.5 * a[3],
                    -1.5 * a[4],
                    -1.5 * a[5],
                ]
            })
            .collect();
        let sc = g.scale;
        for (p, a) in sums.iter().enumerate() {
            let (l, i) = (&g.loc[p], g.idx[p]);
            (g.vrho[i], g.vsig[i]) = potential(sc, beta, g.s[p][0], a[1], a[2], [l.a1, l.b1, l.c1]);
        }
        if grid_terms {
            let geo = sums
                .iter()
                .map(|a| {
                    let mut o = [0.0; 9];
                    for c in 0..3 {
                        o[c] = 1.5 * a[6 + c];
                        o[3 + c] = -1.5 * a[9 + c];
                        o[6 + c] = 1.5 * a[12 + c];
                    }
                    o
                })
                .collect();
            let hrho = self.rho_hessian(&g, dt);
            let (wg_off, wg) = weight_gradients(grid, &g);
            let mut by_atom = vec![Vec::new(); natom];
            for p in 0..na {
                by_atom[g.own[p]].push(p);
            }
            g.grid = Some(GridGround {
                hrho,
                geo,
                wg_off,
                wg,
                by_atom,
            });
        }
        g
    }

    /// The Hessian of rho at the active points: d_k d_l rho = 2 sum_m (d_kl phi_m X_m +
    /// d_k phi_m X^l_m), X = phi D, X^l = d_l phi D.
    fn rho_hessian(&self, g: &Vv10Ground, dt: &Mat<f64>) -> Vec<[f64; 6]> {
        let eng = &self.eng;
        let ci = comp_index(2);
        let no_points = |bi: usize| g.brange[bi].is_empty();
        let plan = BatchPlan::new(2, eng.func_thresh).skipping(&no_points);
        let (parts, _) = eng.for_each_batch(&plan, BatchWork::default, |wk, b| {
            let r = g.brange[b.bi].clone();
            let mut out = vec![[0.0; 6]; r.len()];
            let (start, ld, nao, nc) = (b.start, b.ld, b.nao, b.nc);
            if nao == 0 {
                return out;
            }
            wk.xmats(dt, b);
            let (ao, x) = (b.ao, &wk.x);
            let pairs = [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)];
            for (q, o) in r.zip(out.iter_mut()) {
                let gp = g.idx[q] - start;
                for (h, &(k, l)) in o.iter_mut().zip(&pairs) {
                    let mut v = 0.0;
                    for m in 0..nao {
                        let f = &ao[m * nc * ld..];
                        v += f[ci.d2[k][l] * ld + gp] * x[m * ld + gp]
                            + f[ci.d1[k] * ld + gp] * x[((1 + l) * nao + m) * ld + gp];
                    }
                    *h = 2.0 * v;
                }
            }
            out
        });
        parts.into_iter().flatten().collect()
    }

    /// The memory of one perturbation's per-point arrays and per-thread accumulators (bytes).
    fn bytes_per_perturbation(&self, g: &Vv10Ground, nocc: &[usize]) -> usize {
        let nthr = rayon::current_num_threads().max(1);
        let nbf = self.eng.basis.nbf;
        10 * 8 * g.npad() + nthr * 16 * nbf * nocc.iter().sum::<usize>().max(1)
    }

    /// Perturbations per group within the memory budget (the XC kernel's, `kernel_budget_mb`).
    fn group_size(&self, g: &Vv10Ground, nocc: &[usize], per_unit: usize) -> usize {
        let per = self.bytes_per_perturbation(g, nocc) * per_unit;
        ((self.eng.kernel_budget_mb << 20) / per.max(1)).max(1)
    }

    /// The CPKS kernel of VV10 for the occupied orbitals `c_o` (one channel: a closed shell's
    /// alpha orbitals; two: the spins) at its ground state `g` (of the total density of `c_o`):
    /// for first-order densities dD_s = A_s C_s^T + C_s A_s^T (`a[s][p]`, nbf x nocc_s), the
    /// first-order VV10 potential of the total first-order density (closed shell 2 dD, open
    /// shell dD_a + dD_b) times C_s, `[s][p]`.
    pub fn kernel_contract_occ(
        &self,
        g: &Vv10Ground,
        c_o: &[MatRef<'_, f64>],
        a: &[Vec<Mat<f64>>],
    ) -> Result<Vec<Vec<Mat<f64>>>, LinusError> {
        Ok(self.kernel_contract_occ_timed(g, c_o, a, false)?.0)
    }

    /// [`Self::kernel_contract_occ`] and where its time went; with `fp32` the pair sums of a
    /// device engine in FP32 (the CPU's are FP64 regardless).
    pub fn kernel_contract_occ_timed(
        &self,
        g: &Vv10Ground,
        c_o: &[MatRef<'_, f64>],
        a: &[Vec<Mat<f64>>],
        fp32: bool,
    ) -> Result<KernelProducts, LinusError> {
        let mut tm = Vv10Times::default();
        let nch = c_o.len();
        let nbf = self.eng.basis.nbf;
        let npert = a[0].len();
        let nocc: Vec<usize> = c_o.iter().map(|c| c.ncols()).collect();
        let mut out: Vec<Vec<Mat<f64>>> = (0..nch).map(|_| Vec::with_capacity(npert)).collect();
        if npert == 0 {
            return Ok((out, tm));
        }
        if g.na() == 0 {
            for (s, o) in out.iter_mut().enumerate() {
                o.extend((0..npert).map(|_| Mat::zeros(nbf, nocc[s])));
            }
            return Ok((out, tm));
        }
        let f_ch = if nch == 1 { 2.0 } else { 1.0 };
        // A^T (nocc x nbf, column-major) for gathering rows of A
        let at: Vec<Vec<Mat<f64>>> = a
            .iter()
            .map(|v| v.par_iter().map(|x| x.transpose().to_owned()).collect())
            .collect();
        let group = self.group_size(g, &nocc, 1);
        for p0 in (0..npert).step_by(group) {
            let r = p0..(p0 + group).min(npert);
            let npp = r.len().div_ceil(NPAD_P) * NPAD_P;
            let t0 = std::time::Instant::now();
            let mut inp = vec![0.0; g.na() * npp * 4];
            let perts = GroupPerts::contiguous(r.clone(), npert);
            self.cpks_inputs(g, c_o, a, &at, f_ch, &perts, npp, &mut inp);
            let coef = coefficients(g, &inp, npp, None);
            let t1 = std::time::Instant::now();
            let outs = pert_sweep(g, &coef, npp, fp32)?;
            let t2 = std::time::Instant::now();
            drop(coef);
            assemble(g, &mut inp, &outs, npp, None);
            drop(outs);
            let (v, _) = self.contract(g, c_o, &inp, npp, &perts, None);
            for (o, vs) in out.iter_mut().zip(v) {
                o.extend(vs);
            }
            tm.grid += (t1 - t0).as_secs_f64() + t2.elapsed().as_secs_f64();
            tm.pairs += (t2 - t1).as_secs_f64();
            tm.perturbations += r.len();
        }
        Ok((out, tm))
    }

    /// The first-order total density (rho1, grad rho1) of the group's perturbations `perts` at
    /// the active points into `out` (point-major: [point][npp][4]).
    #[allow(clippy::too_many_arguments)]
    fn cpks_inputs(
        &self,
        g: &Vv10Ground,
        c_o: &[MatRef<'_, f64>],
        a: &[Vec<Mat<f64>>],
        at: &[Vec<Mat<f64>>],
        f_ch: f64,
        perts: &GroupPerts,
        npp: usize,
        out: &mut [f64],
    ) {
        let eng = &self.eng;
        let np = perts.len();
        let chunks = batch_chunks(out, &g.brange, npp * 4);
        let plan = BatchPlan::new(1, eng.func_thresh);
        eng.for_each_batch_in(&plan, chunks, BatchWork::default, |wk, b, chunk| {
            let (bi, start, ld, nao, nc) = (b.bi, b.start, b.ld, b.nao, b.nc);
            if nao == 0 {
                return;
            }
            let s4 = 4 * ld;
            let ub = sized(&mut wk.u, np * s4);
            ub.fill(0.0);
            for (s, co) in c_o.iter().enumerate() {
                let nk = co.ncols();
                if nk == 0 {
                    continue;
                }
                let ao = b.ao;
                let cs = sized(&mut wk.csub, nao * nk);
                gather_rows(*co, b.idx, cs);
                let f4 = sized(&mut wk.phi4, s4 * nk);
                gemm(
                    mmut(f4, s4, nk, s4),
                    false,
                    mref(ao, s4, nao, nc * ld),
                    mref(cs, nao, nk, nao),
                    1.0,
                );
                let phi0 = mref(ao, ld, nao, nc * ld);
                let phim = mref(f4, ld, nk, s4);
                // perturbations in blocks of ~2^17 doubles, as one group has them
                let per_p = ld * (nk + nao) + 2 * nao * nk;
                let ps = ((1usize << 17) / per_p.max(1)).clamp(1, perts.n_all);
                for (lr, pc_whole) in perts.blocks(ps) {
                    let (b0, pc) = (lr.start, lr.len());
                    let ab = sized(&mut wk.abuf, nao * pc * nk);
                    let atb = sized(&mut wk.atbuf, nk * pc * nao);
                    for pl in 0..pc {
                        let p = perts.global(b0 + pl);
                        gather_rows(
                            a[s][p].as_ref(),
                            b.idx,
                            &mut ab[pl * nk * nao..(pl + 1) * nk * nao],
                        );
                        let atm = &at[s][p];
                        for (m, &r) in b.idx.iter().enumerate() {
                            atb[(pl * nao + m) * nk..(pl * nao + m + 1) * nk]
                                .copy_from_slice(atm.col_as_slice(r));
                        }
                    }
                    let pb = sized(&mut wk.p, ld * pc * nk);
                    let whole = |w: usize| Whole {
                        rows: ld,
                        cols: pc_whole * w,
                    };
                    gemm_part(
                        whole(nk),
                        mmut(pb, ld, pc * nk, ld),
                        false,
                        phi0,
                        mref(ab, nao, pc * nk, nao),
                        1.0,
                    );
                    let qb = sized(&mut wk.q, ld * pc * nao);
                    gemm_part(
                        whole(nao),
                        mmut(qb, ld, pc * nao, ld),
                        false,
                        phim,
                        mref(atb, nk, pc * nao, nk),
                        1.0,
                    );
                    let u4 = sized(&mut wk.t, s4);
                    for pl in 0..pc {
                        occ_u1(
                            ao,
                            nc,
                            ld,
                            nao,
                            nk,
                            true,
                            false,
                            f4,
                            &pb[pl * nk * ld..],
                            ld,
                            &qb[pl * nao * ld..(pl + 1) * nao * ld],
                            &[],
                            0,
                            u4,
                        );
                        let ud = &mut ub[(b0 + pl) * s4..(b0 + pl + 1) * s4];
                        for (d, &v) in ud.iter_mut().zip(u4.iter()) {
                            *d += f_ch * v;
                        }
                    }
                }
            }
            let r = g.brange[bi].clone();
            for (k, q) in r.enumerate() {
                let gp = g.idx[q] - start;
                for p in 0..np {
                    for c in 0..4 {
                        chunk[(k * npp + p) * 4 + c] = ub[p * s4 + c * ld + gp];
                    }
                }
            }
        });
    }

    /// The density derivatives u^x of the basis functions moving with the atoms `atoms` (x = 3
    /// (A - atoms.start) + k, at the fixed density `dt`) at the active points into `out`
    /// (point-major: [point][npp][4]).
    fn basis_motion_inputs(
        &self,
        g: &Vv10Ground,
        dt: &Mat<f64>,
        atoms: Range<usize>,
        npp: usize,
        out: &mut [f64],
    ) {
        let eng = &self.eng;
        let ci = comp_index(2);
        let chunks = batch_chunks(out, &g.brange, npp * 4);
        let plan = BatchPlan::new(2, eng.func_thresh);
        eng.for_each_batch_in(&plan, chunks, BatchWork::default, |wk, b, chunk| {
            let (bi, start, ld) = (b.bi, b.start, b.ld);
            if b.nao == 0 {
                return;
            }
            wk.xmats(dt, b);
            let nat = wk.local_atoms(b);
            wk.basis_motion(&ci, b, nat);
            let (u, la_atoms) = (&wk.u, &wk.atoms);
            for (l, &atom) in la_atoms.iter().enumerate() {
                if !atoms.contains(&atom) {
                    continue;
                }
                for k in 0..3 {
                    let x = 3 * l + k;
                    let p = 3 * (atom - atoms.start) + k;
                    for (kk, q) in g.brange[bi].clone().enumerate() {
                        let gp = g.idx[q] - start;
                        for c in 0..4 {
                            chunk[(kk * npp + p) * 4 + c] = u[(x * 4 + c) * ld + gp];
                        }
                    }
                }
            }
        });
    }

    /// V[t_p] C_s for the weighted first-order potentials t (point-major [point][npp][4]) of
    /// the group's perturbations `perts`, as fixed-point sums `[s][p]`; with `e_dt` (the total
    /// density) also E[x][p] = sum_i u^x_i . t^p_i for every coordinate x ((3N x np), row-major).
    fn contract(
        &self,
        g: &Vv10Ground,
        c_o: &[MatRef<'_, f64>],
        t: &[f64],
        npp: usize,
        perts: &GroupPerts,
        e_dt: Option<&Mat<f64>>,
    ) -> (Vec<Vec<Mat<f64>>>, Vec<f64>) {
        let np = perts.len();
        let eng = &self.eng;
        let nbf = eng.basis.nbf;
        let nch = c_o.len();
        let nocc: Vec<usize> = c_o.iter().map(|c| c.ncols()).collect();
        let n3 = 3 * g.natom;
        let ci = comp_index(2);
        let col_off: Vec<Vec<usize>> = nocc
            .iter()
            .map(|&k| (0..k).map(|i| i * nbf).collect())
            .collect();
        let init = || {
            let acc: Vec<Vec<FixedAcc>> = (0..nch)
                .map(|s| (0..np).map(|_| FixedAcc::new(nbf * nocc[s])).collect())
                .collect();
            let eacc = FixedAcc::new(if e_dt.is_some() { n3 * np } else { 0 });
            (acc, eacc, BatchWork::default())
        };
        let no_points = |bi: usize| g.brange[bi].is_empty();
        let plan = BatchPlan::new(if e_dt.is_some() { 2 } else { 1 }, eng.func_thresh)
            .skipping(&no_points);
        let (_, parts) = eng.for_each_batch(&plan, init, |(acc, eacc, wk), b| {
            let (bi, start, ld, nao, nc) = (b.bi, b.start, b.ld, b.nao, b.nc);
            if nao == 0 {
                return;
            }
            let s4 = 4 * ld;
            // t of the batch (np x 4 x ld), zero at the inactive points
            let tb = sized(&mut wk.tb, np * s4);
            tb.fill(0.0);
            for q in g.brange[bi].clone() {
                let gp = g.idx[q] - start;
                for p in 0..np {
                    for c in 0..4 {
                        tb[p * s4 + c * ld + gp] = t[(q * npp + p) * 4 + c];
                    }
                }
            }
            for s in 0..nch {
                let nk = nocc[s];
                if nk == 0 {
                    continue;
                }
                let ao = b.ao;
                let cs = sized(&mut wk.csub, nao * nk);
                gather_rows(c_o[s], b.idx, cs);
                let f4 = sized(&mut wk.phi4, s4 * nk);
                gemm(
                    mmut(f4, s4, nk, s4),
                    false,
                    mref(ao, s4, nao, nc * ld),
                    mref(cs, nao, nk, nao),
                    1.0,
                );
                let per_p = ld * (nk + nao) + nao * nk;
                let ps = ((1usize << 17) / per_p.max(1)).clamp(1, perts.n_all);
                for (lr, pc_whole) in perts.blocks(ps) {
                    let (b0, pc) = (lr.start, lr.len());
                    let zb = sized(&mut wk.p, ld * pc * nk);
                    let gmb = sized(&mut wk.q, ld * pc * nao);
                    for pl in 0..pc {
                        occ_out(
                            ao,
                            nc,
                            ld,
                            nao,
                            nk,
                            true,
                            false,
                            &tb[(b0 + pl) * s4..(b0 + pl + 1) * s4],
                            f4,
                            &mut zb[pl * nk * ld..(pl + 1) * nk * ld],
                            ld,
                            0,
                            &mut gmb[pl * nao * ld..(pl + 1) * nao * ld],
                            &mut [],
                        );
                    }
                    let nao_o = nao * nk;
                    let x = VcoInputs {
                        nocc: nk,
                        pc,
                        pc_whole,
                        gga: true,
                        mgga: false,
                        phi4: f4,
                        z: zb,
                        gm: gmb,
                        tk: &[],
                    };
                    let ob = push_vco(b, &x, &mut wk.abuf);
                    for pl in 0..pc {
                        scatter_cols(
                            &mut acc[s][b0 + pl],
                            &col_off[s],
                            nao,
                            b.runs,
                            &ob[pl * nao_o..(pl + 1) * nao_o],
                        );
                    }
                }
            }
            for a in acc.iter_mut().flatten() {
                a.tick(1);
            }
            if let Some(dt) = e_dt {
                wk.xmats(dt, b);
                let nat = wk.local_atoms(b);
                wk.basis_motion(&ci, b, nat);
                let nx = 3 * nat;
                let el = sized(&mut wk.el, nx * np);
                // a part of the product for all perturbations (one group)
                let whole = Whole {
                    rows: nx,
                    cols: perts.n_all,
                };
                gemm_part(
                    whole,
                    mmut(el, nx, np, nx),
                    false,
                    mref(&wk.u, s4, nx, s4).transpose(),
                    mref(&wk.tb, s4, np, s4),
                    1.0,
                );
                for (l, &atom) in wk.atoms.iter().enumerate() {
                    for k in 0..3 {
                        let row = 3 * atom + k;
                        for p in 0..np {
                            eacc.add1(row * np + p, el[p * nx + 3 * l + k]);
                        }
                    }
                }
                eacc.tick(1);
            }
        });
        let mut e = vec![0.0; if e_dt.is_some() { n3 * np } else { 0 }];
        let mut mats: Vec<Vec<FixedAcc>> = Vec::new();
        let mut eaccs = Vec::new();
        for (a, ea, _) in parts {
            mats.push(a.into_iter().flatten().collect());
            eaccs.push(vec![ea]);
        }
        let v: Vec<Vec<Mat<f64>>> = match merge_all(mats) {
            Some(m) => {
                let mut it = m.into_iter();
                (0..nch)
                    .map(|s| {
                        (0..np)
                            .map(|_| {
                                let acc = it.next().unwrap();
                                MatRef::from_column_major_slice(&acc.to_f64(), nbf, nocc[s])
                                    .to_owned()
                            })
                            .collect()
                    })
                    .collect()
            }
            None => (0..nch)
                .map(|s| (0..np).map(|_| Mat::zeros(nbf, nocc[s])).collect())
                .collect(),
        };
        if e_dt.is_some() {
            if let Some(m) = merge_all(eaccs) {
                e = m[0].to_f64();
            }
        }
        (v, e)
    }

    /// The explicit VV10 terms of a Hessian at the total density `dt` (its ground state `g`,
    /// with the grid terms when `grid_response`) for the occupied orbitals `c_o` of the channels
    /// (see the module docs): V^x C_s, with the grid response dV^y C_s, and the second-derivative
    /// terms.
    pub fn hessian_terms(
        &self,
        g: &Vv10Ground,
        c_o: &[MatRef<'_, f64>],
        dt: &Mat<f64>,
        grid_response: bool,
        par: Par,
    ) -> Result<Vv10Hessian, LinusError> {
        let grid_response = grid_response && g.grid.is_some();
        let mut times = Vv10Times::default();
        let t0 = std::time::Instant::now();
        let natom = g.natom;
        let n3 = 3 * natom;
        let nocc: Vec<usize> = c_o.iter().map(|c| c.ncols()).collect();
        // the terms of the potential itself: the semilocal Hessian code with v given per point
        let ext = ExtPotential {
            vrho: &g.vrho,
            vsigma: &g.vsig,
        };
        let (vx, mut hess) = self.eng.hessian_terms_ch::<1>([dt], true, true, Some(&ext));
        let mut vfix = vx.project(c_o, par);
        let mut vgrid: Vec<Vec<Mat<f64>>> = Vec::new();
        if grid_response {
            let (vg, hg) = self.eng.grid_terms_ch::<1>([dt], true, true, Some(&ext));
            hess.iter_mut().zip(&hg).for_each(|(a, b)| *a += b);
            vgrid = vg.project(c_o, par);
        }
        times.potential_terms = t0.elapsed().as_secs_f64();
        let done = |vfix, vgrid, hess, times| {
            Ok(Vv10Hessian {
                terms: XcHessianTerms {
                    vfix,
                    vgrid,
                    hess,
                    hess_grid: Vec::new(),
                },
                times,
            })
        };
        if g.na() == 0 {
            return done(vfix, vgrid, hess, times);
        }
        // the nonlocal kernel, in groups of atoms (3 or 6 perturbations each)
        let per_atom = if grid_response { 6 } else { 3 };
        let ga = self.group_size(g, &nocc, per_atom);
        for a0 in (0..natom).step_by(ga) {
            let atoms = a0..(a0 + ga).min(natom);
            let nx = 3 * atoms.len();
            let np = if grid_response { 2 * nx } else { nx };
            let npp = np.div_ceil(NPAD_P) * NPAD_P;
            let t0 = std::time::Instant::now();
            let mut inp = vec![0.0; g.na() * npp * 4];
            self.basis_motion_inputs(g, dt, atoms.clone(), npp, &mut inp);
            let gm = grid_response.then(|| GridPerts {
                atoms: atoms.clone(),
                first: nx,
            });
            if let Some(gp) = &gm {
                grid_inputs(g, gp, npp, &mut inp);
            }
            let coef = coefficients(g, &inp, npp, gm.as_ref());
            let t1 = std::time::Instant::now();
            let mut outs = pert_sweep(g, &coef, npp, false)?;
            drop(coef);
            if let Some(gp) = &gm {
                add_geometric(g, gp, npp, &mut outs);
            }
            let t2 = std::time::Instant::now();
            assemble(g, &mut inp, &outs, npp, gm.as_ref());
            drop(outs);
            // the group's perturbations in the single group's order: [fixed 3N, grid 3N]
            let mut runs = vec![(0, 3 * a0, nx)];
            if grid_response {
                runs.push((nx, n3 + 3 * a0, nx));
            }
            let n_all = if grid_response { 2 * n3 } else { n3 };
            let perts = GroupPerts::new(runs, n_all);
            let (v, e) = self.contract(g, c_o, &inp, npp, &perts, Some(dt));
            times.add(&Vv10Times {
                potential_terms: 0.0,
                grid: (t1 - t0).as_secs_f64() + t2.elapsed().as_secs_f64(),
                pairs: (t2 - t1).as_secs_f64(),
                perturbations: np,
            });
            for (s, vs) in v.into_iter().enumerate() {
                for (p, m) in vs.into_iter().enumerate() {
                    let col = 3 * a0 + p % nx;
                    if p < nx {
                        vfix[s][col] += m;
                    } else {
                        vgrid[s][col] += m;
                    }
                }
            }
            for x in 0..n3 {
                for p in 0..np {
                    hess[x * n3 + 3 * a0 + p % nx] += e[x * np + p];
                }
            }
        }
        done(vfix, vgrid, hess, times)
    }
}

/// The perturbations of a group as a part of the single group (all perturbations in one group,
/// the memory budget permitting): runs `(local, global, len)` of the group's perturbations, its
/// `local..local + len` being the single group's `global..global + len`, of `n_all`. Blocks of
/// perturbations are aligned as in the single group ([`GroupPerts::blocks`]) and every product of
/// a block's part takes the whole block's kernel, so a group gives its perturbations the single
/// group's bits however the memory budget or the thread count groups them.
struct GroupPerts {
    runs: Vec<(usize, usize, usize)>,
    n_all: usize,
}

impl GroupPerts {
    /// The group of these runs, adjacent ones merged (the single group is one run).
    fn new(runs: Vec<(usize, usize, usize)>, n_all: usize) -> Self {
        let mut merged: Vec<(usize, usize, usize)> = Vec::new();
        for r in runs.into_iter().filter(|r| r.2 > 0) {
            match merged.last_mut() {
                Some(m) if m.0 + m.2 == r.0 && m.1 + m.2 == r.1 => m.2 += r.2,
                _ => merged.push(r),
            }
        }
        GroupPerts {
            runs: merged,
            n_all,
        }
    }

    /// The perturbations `range` of `n_all`, in order.
    fn contiguous(range: Range<usize>, n_all: usize) -> Self {
        GroupPerts {
            runs: vec![(0, range.start, range.len())],
            n_all,
        }
    }

    /// The group's number of perturbations.
    fn len(&self) -> usize {
        self.runs.iter().map(|r| r.2).sum()
    }

    /// The single group's index of the group's perturbation `p`.
    fn global(&self, p: usize) -> usize {
        let &(l0, g0, _) = self
            .runs
            .iter()
            .find(|r| (r.0..r.0 + r.2).contains(&p))
            .expect("a perturbation of the group");
        g0 + p - l0
    }

    /// The group's parts of the single group's blocks of `ps` perturbations (at its multiples of
    /// `ps`): the local range of each part and the size of its whole block.
    fn blocks(&self, ps: usize) -> impl Iterator<Item = (Range<usize>, usize)> + '_ {
        let n_all = self.n_all;
        self.runs.iter().flat_map(move |&(l0, g0, len)| {
            (g0 / ps..(g0 + len).div_ceil(ps)).filter_map(move |k| {
                let (b0, b1) = (k * ps, ((k + 1) * ps).min(n_all));
                let (a, b) = (b0.max(g0), b1.min(g0 + len));
                (a < b).then(|| (l0 + a - g0..l0 + b - g0, b1 - b0))
            })
        })
    }
}

/// The grid perturbations of a group: y = 3 (B - atoms.start) + k + `first` for the atoms B.
struct GridPerts {
    atoms: Range<usize>,
    first: usize,
}

/// The density at the points moving with their owners: drho = d_k rho, d grad rho = d_k grad rho
/// on the points of B for y = (B, k), into `out` (point-major).
fn grid_inputs(g: &Vv10Ground, gp: &GridPerts, npp: usize, out: &mut [f64]) {
    let gg = g.grid.as_ref().expect("the grid ground state");
    let sym = |k: usize, l: usize| -> usize {
        let (a, b) = if k <= l { (k, l) } else { (l, k) };
        [[0, 1, 2], [1, 3, 4], [2, 4, 5]][a][b]
    };
    out.par_chunks_mut(npp * 4).enumerate().for_each(|(q, o)| {
        let b = g.own[q];
        if !gp.atoms.contains(&b) {
            return;
        }
        for k in 0..3 {
            let p = gp.first + 3 * (b - gp.atoms.start) + k;
            o[p * 4] = g.grad[q][k];
            for c in 0..3 {
                o[p * 4 + 1 + c] = gg.hrho[q][sym(k, c)];
            }
        }
    });
}

/// The pair-sweep coefficients of the perturbations (layout [p][3][npad]: alpha, b, c) from
/// their first-order densities (point-major `inp`), with the weight derivatives of the grid
/// perturbations `gp`.
fn coefficients(g: &Vv10Ground, inp: &[f64], npp: usize, gp: Option<&GridPerts>) -> Vec<f64> {
    let npad = g.npad();
    let na = g.na();
    let mut cf = vec![0.0; npp * 3 * npad];
    cf.par_chunks_mut(3 * npad).enumerate().for_each(|(p, c)| {
        let (al, rest) = c.split_at_mut(npad);
        let (bs, cs) = rest.split_at_mut(npad);
        for j in 0..na {
            let u = &inp[(j * npp + p) * 4..(j * npp + p) * 4 + 4];
            let (dr, gr, l) = (u[0], &g.grad[j], &g.loc[j]);
            let ds = 2.0 * (gr[0] * u[1] + gr[1] * u[2] + gr[2] * u[3]);
            al[j] = g.w[j] * dr;
            bs[j] = g.n[j] * (l.ka_r * dr);
            cs[j] = g.n[j] * (l.om_r * dr + l.om_s * ds);
        }
    });
    if let Some(gp) = gp {
        let gg = g.grid.as_ref().expect("the grid ground state");
        for j in 0..na {
            for &(atom, dw) in &gg.wg[gg.wg_off[j]..gg.wg_off[j + 1]] {
                if !gp.atoms.contains(&atom) {
                    continue;
                }
                for (k, &d) in dw.iter().enumerate() {
                    let p = gp.first + 3 * (atom - gp.atoms.start) + k;
                    cf[p * 3 * npad + j] += d * g.rho[j];
                }
            }
        }
    }
    cf
}

/// Adds the geometric sums of the grid perturbations (moving pair distances) to the pair sums
/// `outs`: for y = (B, k), 2 [i on B] sum_j n_j g (r_i - r_j)_k - 2 sum_{j on B} n_j g (r_i - r_j)_k.
fn add_geometric(g: &Vv10Ground, gp: &GridPerts, npp: usize, outs: &mut [f64]) {
    let gg = g.grid.as_ref().expect("the grid ground state");
    let ng = gp.atoms.len();
    let bk = bucket_sweep(g, gp.atoms.clone());
    outs.par_chunks_mut(3 * npp * IB)
        .zip(bk.par_chunks(ng * 9 * IB))
        .enumerate()
        .for_each(|(blk, (o, bb))| {
            for l in 0..IB {
                let q = blk * IB + l;
                if q >= g.na() {
                    break;
                }
                for (ba, b) in gp.atoms.clone().enumerate() {
                    let on = g.own[q] == b;
                    for k in 0..3 {
                        let p = gp.first + 3 * ba + k;
                        for kind in 0..3 {
                            let mut v = -2.0 * bb[(ba * 9 + kind * 3 + k) * IB + l];
                            if on {
                                v += 2.0 * gg.geo[q][kind * 3 + k];
                            }
                            o[(kind * npp + p) * IB + l] += v;
                        }
                    }
                }
            }
        });
}

/// The weighted, scaled first-order potentials t = w (dv_rho, 2 dv_sigma grad rho + 2 v_sigma
/// d grad rho) of every perturbation at every active point, in place of its first-order density
/// in `inp` (module docs), from the pair sums `outs` (with the geometric sums of `gp`).
fn assemble(g: &Vv10Ground, inp: &mut [f64], outs: &[f64], npp: usize, gp: Option<&GridPerts>) {
    let _ = gp;
    let sc = g.scale;
    inp.par_chunks_mut(npp * 4).enumerate().for_each(|(q, o)| {
        let (blk, l) = (q / IB, q % IB);
        let base = blk * 3 * npp * IB;
        let (s, lc, gr, w) = (&g.s[q], &g.loc[q], &g.grad[q], g.w[q]);
        let vs = lc.c1 * s[2];
        for p in 0..npp {
            let u = &mut o[p * 4..p * 4 + 4];
            let p1 = outs[base + p * IB + l];
            let p2 = outs[base + (npp + p) * IB + l];
            let p3 = outs[base + (2 * npp + p) * IB + l];
            let dr = u[0];
            let ds = 2.0 * (gr[0] * u[1] + gr[1] * u[2] + gr[2] * u[3]);
            let dom = lc.om_r * dr + lc.om_s * ds;
            let dka = lc.ka_r * dr;
            let df = p1 + dka * s[1] + dom * s[2];
            let ds1 = p2 + dka * s[3] + dom * s[4];
            let ds2 = p3 + dka * s[4] + dom * s[5];
            let dvr = df
                + lc.a2 * dr * s[1]
                + lc.a1 * ds1
                + (lc.b2 * dr + lc.b3 * ds) * s[2]
                + lc.b1 * ds2;
            let dvs = (lc.c2 * dr + lc.c3 * ds) * s[2] + lc.c1 * ds2;
            let f = sc * w;
            let t = [
                f * dvr,
                f * 2.0 * (dvs * gr[0] + vs * u[1]),
                f * 2.0 * (dvs * gr[1] + vs * u[2]),
                f * 2.0 * (dvs * gr[2] + vs * u[3]),
            ];
            u.copy_from_slice(&t);
        }
    });
}

/// The weight derivatives of the active points (CSR offsets and (atom, dw/dR_atom)), the point
/// moving with its owner.
fn weight_gradients(
    grid: &linus_grid::MolecularGrid,
    g: &Vv10Ground,
) -> (Vec<usize>, Vec<(usize, [f64; 3])>) {
    type PointGradients = Vec<Vec<(usize, [f64; 3])>>;
    let parts: Vec<PointGradients> = grid
        .batches
        .par_iter()
        .enumerate()
        .map_init(
            || (grid.partitioner().scratch(), Vec::new()),
            |(scr, out), (bi, _)| {
                g.brange[bi]
                    .clone()
                    .map(|q| {
                        grid.point_weight_gradient(g.idx[q], scr, out);
                        out.clone()
                    })
                    .collect()
            },
        )
        .collect();
    let mut off = vec![0];
    let mut wg = Vec::new();
    for v in parts.into_iter().flatten() {
        wg.extend(v);
        off.push(wg.len());
    }
    (off, wg)
}

/// Splits a point-major array (`stride` values per active point) into the chunks of the
/// batches that have active points: (batch, chunk).
fn batch_chunks<'a>(
    mut buf: &'a mut [f64],
    brange: &[Range<usize>],
    stride: usize,
) -> Vec<(usize, &'a mut [f64])> {
    let mut v = Vec::new();
    for (bi, r) in brange.iter().enumerate() {
        if r.is_empty() {
            continue;
        }
        let (a, b) = std::mem::take(&mut buf).split_at_mut(r.len() * stride);
        v.push((bi, a));
        buf = b;
    }
    v
}

/// Per-thread scratch of the batch passes (the basis functions are the batch engine's).
#[derive(Default)]
struct BatchWork {
    /// X_c = (d_c phi) D, c = value, x, y, z (ld x nao each)
    x: Vec<f64>,
    dsub: Vec<f64>,
    csub: Vec<f64>,
    phi4: Vec<f64>,
    abuf: Vec<f64>,
    atbuf: Vec<f64>,
    p: Vec<f64>,
    q: Vec<f64>,
    t: Vec<f64>,
    tb: Vec<f64>,
    /// u^x of the batch's local atoms (3 nat x 4 x ld)
    u: Vec<f64>,
    el: Vec<f64>,
    /// local atoms of the batch's functions, and the local atom of each function
    atoms: Vec<usize>,
    la: Vec<usize>,
}

impl BatchWork {
    /// X_c = (d_c phi) D_sub (c = 0..4, ld x nao each) of the collocated batch `b`.
    fn xmats(&mut self, dt: &Mat<f64>, b: &BatchCtx<'_>) {
        let (nc, ld, nao) = (b.nc, b.ld, b.nao);
        let ds = sized(&mut self.dsub, nao * nao);
        gather_sub(dt, b.idx, b.runs, ds);
        let x = sized(&mut self.x, 4 * ld * nao);
        for c in 0..4 {
            gemm(
                mmut(&mut x[c * ld * nao..], ld, nao, ld),
                false,
                mref(&b.ao[c * ld..], ld, nao, nc * ld),
                mref(ds, nao, nao, nao),
                1.0,
            );
        }
    }

    /// The local atoms of batch `b` (in order of first appearance among its functions);
    /// returns their number.
    fn local_atoms(&mut self, b: &BatchCtx<'_>) -> usize {
        local_index(b.owner, &mut self.atoms, &mut self.la);
        self.atoms.len()
    }

    /// u^x (3 nat x 4 x ld) of the local atoms (after `xmats` and `local_atoms`): the XC
    /// Hessian's u^x of a GGA ([`crate::hessian::basis_motion`]) with the total density.
    fn basis_motion(&mut self, ci: &crate::hessian::CompIdx, b: &BatchCtx<'_>, nat: usize) {
        let (ld, nao) = (b.ld, b.nao);
        let u = sized(&mut self.u, 3 * nat * 4 * ld);
        u.fill(0.0);
        let hb = HBatch {
            f: Fns::of(b),
            ci,
            gga: true,
            mgga: false,
            ns: 4,
        };
        let xs = Xs {
            x: &self.x,
            nx: 4,
            nao,
            ld,
            np: b.np,
        };
        basis_motion::<1>(hb, xs, &self.la, u);
    }
}

// ------------------------------------------------------------------ pair sweeps

/// Ground-state pair sums of every active point: [F, S1..S5 without their factors, then with
/// `geo` the geometric sums sum_j n_j (e1, e2, e3) (r_i - r_j)] (module docs; the factors 1.5,
/// -1.5 are applied by the caller).
fn ground_sweep(g: &Vv10Ground, geo: bool) -> Vec<[f64; 15]> {
    let npad = g.npad();
    let blocks: Vec<Vec<[f64; 15]>> = (0..npad / IB)
        .into_par_iter()
        .map(|b| {
            let mut out = vec![[0.0; 15]; IB];
            ground_block_dispatch(g, b * IB, geo, &mut out);
            out
        })
        .collect();
    let mut v: Vec<[f64; 15]> = blocks.into_iter().flatten().collect();
    v.truncate(g.na());
    v
}

/// Per-perturbation pair sums of all active points (module docs; with the factor 1.5):
/// layout [block][3][npp][IB] from the coefficients [p][3][npad].
fn pert_sweep(
    g: &Vv10Ground,
    coef: &[f64],
    npp: usize,
    fp32: bool,
) -> Result<Vec<f64>, LinusError> {
    let npad = g.npad();
    let mut out = vec![0.0; npad * 3 * npp];
    if let Some(d) = &g.device {
        // the device's [c][p][i] into the blocks, with the factor 1.5
        let na = g.na();
        let dp = d.npad();
        let s = d.sweep(coef, npad, npp, fp32)?;
        out.par_chunks_mut(3 * npp * IB)
            .enumerate()
            .for_each(|(b, o)| {
                for c in 0..3 {
                    for p in 0..npp {
                        for l in 0..IB {
                            let i = b * IB + l;
                            if i < na {
                                o[(c * npp + p) * IB + l] = 1.5 * s[(c * npp + p) * dp + i];
                            }
                        }
                    }
                }
            });
        return Ok(out);
    }
    out.par_chunks_mut(3 * npp * IB)
        .enumerate()
        .for_each(|(b, o)| {
            pert_block_dispatch(g, b * IB, coef, npp, o);
            for v in o.iter_mut() {
                *v *= 1.5;
            }
        });
    Ok(out)
}

/// Geometric sums per atom B of `atoms`: sum_{j on B} n_j (g1, g2, g3) (r_i - r_j) with their
/// factors, layout [block][B][kind][component][IB].
fn bucket_sweep(g: &Vv10Ground, atoms: Range<usize>) -> Vec<f64> {
    let npad = g.npad();
    let ng = atoms.len();
    let mut out = vec![0.0; npad * ng * 9];
    out.par_chunks_mut(ng * 9 * IB)
        .enumerate()
        .for_each(|(b, o)| {
            bucket_block_dispatch(g, b * IB, atoms.clone(), o);
            for (k, v) in o.iter_mut().enumerate() {
                let kind = (k / IB) % 9 / 3;
                *v *= if kind == 1 { -1.5 } else { 1.5 };
            }
        });
    out
}

#[inline(always)]
fn lanes<const L: usize>(v: &[f64], i: usize) -> [f64; L] {
    v[i..i + L].try_into().unwrap()
}

/// Ground sums of L lanes from point ii against the j points `js`: F, S1, S2 as the SCF sums
/// them ([`fuw`]: the SCF's bits), S3..S5 and the geometric sums from the pair's reciprocals.
#[inline(always)]
fn ground_lanes<const L: usize, const GEO: bool>(
    g: &Vv10Ground,
    ii: usize,
    js: Range<usize>,
    acc: &mut [[f64; L]; 15],
) {
    let (xi, yi, zi, ni, oi, ki) = (
        lanes::<L>(&g.x, ii),
        lanes::<L>(&g.y, ii),
        lanes::<L>(&g.z, ii),
        lanes::<L>(&g.nn, ii),
        lanes::<L>(&g.om, ii),
        lanes::<L>(&g.ka, ii),
    );
    let mut a = *acc;
    for j in js {
        let (mx, my, mz, nnj) = (g.mx[j], g.my[j], g.mz[j], g.nn[j]);
        let (oj, kj, nj) = (g.om[j], g.ka[j], g.n[j]);
        for l in 0..L {
            let (r2, gi, gj) = pair_g(
                xi[l], yi[l], zi[l], ni[l], oi[l], ki[l], mx, my, mz, nnj, oj, kj,
            );
            let (p, q, qi) = fuw(gi, gj, nj);
            a[0][l] = p.mul_add(q, a[0][l]);
            a[1][l] += qi;
            a[2][l] = qi.mul_add(r2, a[2][l]);
            let (t, ig, igj, igt) = reciprocals(gi, gj);
            let tn = nj * t;
            let ai = ig + igt;
            let q2 = ai.mul_add(ai, ig.mul_add(ig, igt * igt));
            let q = tn * q2;
            a[3][l] += q;
            let rq = r2 * q;
            a[4][l] += rq;
            a[5][l] = r2.mul_add(rq, a[5][l]);
            if GEO {
                let d = [xi[l] - g.x[j], yi[l] - g.y[j], zi[l] - g.z[j]];
                let tai = tn * ai;
                let aj = igj + igt;
                let q3 = ai.mul_add(aj, igt * igt);
                let e1 = tn * oj.mul_add(aj, oi[l] * ai);
                let e2 = tn * oj.mul_add(q3, oi[l] * q2);
                let e3 = (-r2).mul_add(e2, tai);
                for c in 0..3 {
                    a[6 + c][l] = e1.mul_add(d[c], a[6 + c][l]);
                    a[9 + c][l] = e2.mul_add(d[c], a[9 + c][l]);
                    a[12 + c][l] = e3.mul_add(d[c], a[12 + c][l]);
                }
            }
        }
    }
    *acc = a;
}

#[inline(always)]
fn ground_block<const L: usize, const GEO: bool>(g: &Vv10Ground, i0: usize, out: &mut [[f64; 15]]) {
    let npad = g.npad();
    let mut acc = vec![[[0.0f64; L]; 15]; IB / L];
    let mut j0 = 0;
    while j0 < npad {
        let j1 = (j0 + GT).min(npad);
        for (lg, a) in acc.iter_mut().enumerate() {
            ground_lanes::<L, GEO>(g, i0 + lg * L, j0..j1, a);
        }
        j0 = j1;
    }
    for (lg, a) in acc.iter().enumerate() {
        for l in 0..L {
            for k in 0..15 {
                out[lg * L + l][k] = a[k][l];
            }
        }
    }
}

/// [`ground_block`] with the geometric sums chosen at run time.
#[inline(always)]
fn ground_block_geo<const L: usize>(g: &Vv10Ground, i0: usize, geo: bool, out: &mut [[f64; 15]]) {
    if geo {
        ground_block::<L, true>(g, i0, out)
    } else {
        ground_block::<L, false>(g, i0, out)
    }
}

linus_num::multiversion_lanes!(fn ground_block_dispatch(g: &Vv10Ground, i0: usize, geo: bool, out: &mut [[f64; 15]]) -> () = ground_block_geo::<8 | 4>);

/// The pair functions of L lanes from point ii against the j points `js`: per j, (-T, T a_j,
/// T a_i, -T (a_i a_j + 1/g_t^2), R^2), a_i = 1/g_i + 1/g_t.
#[inline(always)]
fn kernel_tile<const L: usize>(g: &Vv10Ground, ii: usize, js: Range<usize>, kb: &mut [[f64; L]]) {
    let (xi, yi, zi, ni, oi, ki) = (
        lanes::<L>(&g.x, ii),
        lanes::<L>(&g.y, ii),
        lanes::<L>(&g.z, ii),
        lanes::<L>(&g.nn, ii),
        lanes::<L>(&g.om, ii),
        lanes::<L>(&g.ka, ii),
    );
    for (t, j) in js.enumerate() {
        let (mx, my, mz, nnj, oj, kj) = (g.mx[j], g.my[j], g.mz[j], g.nn[j], g.om[j], g.ka[j]);
        let mut v = [[0.0f64; L]; 5];
        for l in 0..L {
            let (r2, gi, gj) = pair_g(
                xi[l], yi[l], zi[l], ni[l], oi[l], ki[l], mx, my, mz, nnj, oj, kj,
            );
            let (tt, ig, igj, igt) = reciprocals(gi, gj);
            let ai = ig + igt;
            let aj = igj + igt;
            v[0][l] = -tt;
            v[1][l] = tt * aj;
            v[2][l] = tt * ai;
            v[3][l] = -(tt * ai.mul_add(aj, igt * igt));
            v[4][l] = r2;
        }
        kb[5 * t..5 * t + 5].copy_from_slice(&v);
    }
}

/// NP perturbations from p0 of the L lanes at block offset l0 against the j tile from j0 (pair
/// functions `kb`): P1 += alpha K0 + h K1, q = alpha K2 + h K3, P2 += q, P3 += R^2 q,
/// h = b + R^2 c.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn pert_lanes<const L: usize, const NP: usize>(
    kb: &[[f64; L]],
    coef: &[f64],
    npad: usize,
    j0: usize,
    p0: usize,
    npp: usize,
    l0: usize,
    out: &mut [f64],
) {
    let nj = kb.len() / 5;
    let mut acc = [[[0.0f64; L]; 3]; NP];
    for (q, a) in acc.iter_mut().enumerate() {
        for (c, ac) in a.iter_mut().enumerate() {
            *ac = lanes::<L>(out, (c * npp + p0 + q) * IB + l0);
        }
    }
    let cf: [[&[f64]; 3]; NP] = std::array::from_fn(|q| {
        std::array::from_fn(|c| {
            let o = ((p0 + q) * 3 + c) * npad + j0;
            &coef[o..o + nj]
        })
    });
    for j in 0..nj {
        let (k0, k1, k2, k3, r2) = (
            kb[5 * j],
            kb[5 * j + 1],
            kb[5 * j + 2],
            kb[5 * j + 3],
            kb[5 * j + 4],
        );
        for q in 0..NP {
            let (al, b, c) = (cf[q][0][j], cf[q][1][j], cf[q][2][j]);
            let a = &mut acc[q];
            for l in 0..L {
                let h = r2[l].mul_add(c, b);
                let p1 = al.mul_add(k0[l], a[0][l]);
                a[0][l] = h.mul_add(k1[l], p1);
                let qv = h.mul_add(k3[l], al * k2[l]);
                a[1][l] += qv;
                a[2][l] = r2[l].mul_add(qv, a[2][l]);
            }
        }
    }
    for (q, a) in acc.iter().enumerate() {
        for (c, ac) in a.iter().enumerate() {
            let o = (c * npp + p0 + q) * IB + l0;
            out[o..o + L].copy_from_slice(ac);
        }
    }
}

#[inline(always)]
fn pert_block<const L: usize, const NP: usize>(
    g: &Vv10Ground,
    i0: usize,
    coef: &[f64],
    npp: usize,
    out: &mut [f64],
) {
    let npad = g.npad();
    let mut kb = [[0.0f64; L]; 5 * JT];
    let mut j0 = 0;
    while j0 < npad {
        let j1 = (j0 + JT).min(npad);
        let nj = j1 - j0;
        for lg in 0..IB / L {
            kernel_tile::<L>(g, i0 + lg * L, j0..j1, &mut kb[..5 * nj]);
            for p0 in (0..npp).step_by(NP) {
                pert_lanes::<L, NP>(&kb[..5 * nj], coef, npad, j0, p0, npp, lg * L, out);
            }
        }
        j0 = j1;
    }
}

linus_num::multiversion_lanes!(fn pert_block_dispatch(g: &Vv10Ground, i0: usize, coef: &[f64], npp: usize, out: &mut [f64]) -> () = pert_block::<8, 4 | 4, 2>);

/// Geometric sums of L lanes from point ii against the points of each atom of `atoms`.
#[inline(always)]
fn bucket_lanes<const L: usize>(
    g: &Vv10Ground,
    ii: usize,
    atoms: Range<usize>,
    l0: usize,
    out: &mut [f64],
) {
    let gg = g.grid.as_ref().expect("the grid ground state");
    let (xi, yi, zi, ni, oi, ki) = (
        lanes::<L>(&g.x, ii),
        lanes::<L>(&g.y, ii),
        lanes::<L>(&g.z, ii),
        lanes::<L>(&g.nn, ii),
        lanes::<L>(&g.om, ii),
        lanes::<L>(&g.ka, ii),
    );
    for (ba, b) in atoms.enumerate() {
        let mut a = [[0.0f64; L]; 9];
        for &j in &gg.by_atom[b] {
            let (xj, yj, zj, oj, kj, nj) = (g.x[j], g.y[j], g.z[j], g.om[j], g.ka[j], g.n[j]);
            let (mx, my, mz, nnj) = (g.mx[j], g.my[j], g.mz[j], g.nn[j]);
            for l in 0..L {
                let (r2, gi, gj) = pair_g(
                    xi[l], yi[l], zi[l], ni[l], oi[l], ki[l], mx, my, mz, nnj, oj, kj,
                );
                let (t, ig, igj, igt) = reciprocals(gi, gj);
                let d = [xi[l] - xj, yi[l] - yj, zi[l] - zj];
                let tn = nj * t;
                let ai = ig + igt;
                let aj = igj + igt;
                let tai = tn * ai;
                let q2 = ai.mul_add(ai, ig.mul_add(ig, igt * igt));
                let q3 = ai.mul_add(aj, igt * igt);
                let e1 = tn * oj.mul_add(aj, oi[l] * ai);
                let e2 = tn * oj.mul_add(q3, oi[l] * q2);
                let e3 = (-r2).mul_add(e2, tai);
                for c in 0..3 {
                    a[c][l] = e1.mul_add(d[c], a[c][l]);
                    a[3 + c][l] = e2.mul_add(d[c], a[3 + c][l]);
                    a[6 + c][l] = e3.mul_add(d[c], a[6 + c][l]);
                }
            }
        }
        for (k, ak) in a.iter().enumerate() {
            let o = (ba * 9 + k) * IB + l0;
            out[o..o + L].copy_from_slice(ak);
        }
    }
}

#[inline(always)]
fn bucket_block<const L: usize>(g: &Vv10Ground, i0: usize, atoms: Range<usize>, out: &mut [f64]) {
    for lg in 0..IB / L {
        bucket_lanes::<L>(g, i0 + lg * L, atoms.clone(), lg * L, out);
    }
}

linus_num::multiversion_lanes!(fn bucket_block_dispatch(g: &Vv10Ground, i0: usize, atoms: Range<usize>, out: &mut [f64]) -> () = bucket_block::<8 | 4>);

/// The CPKS kernel of VV10 as a [`linus_base::terms::ResponseKernel`]: its ground state fixed at
/// the setup (the occupied orbitals of the response), time in [`Vv10Kernel::seconds`].
pub struct Vv10Kernel<'a> {
    nl: &'a Vv10,
    ground: std::sync::Arc<Vv10Ground>,
    seconds: Mutex<(f64, Vv10Times)>,
    /// The device's pair sums in FP32 (as the device's semilocal CPKS kernel).
    fp32: bool,
}

impl<'a> Vv10Kernel<'a> {
    /// The kernel of `nl` at the ground state `ground` (of the occupied orbitals of the
    /// response).
    pub fn new(nl: &'a Vv10, ground: std::sync::Arc<Vv10Ground>) -> Self {
        Vv10Kernel {
            nl,
            ground,
            seconds: Mutex::new((0.0, Vv10Times::default())),
            fp32: false,
        }
    }

    /// The device's pair sums in FP32 (`fp32`) or in the engine's arithmetic.
    pub fn with_fp32(mut self, fp32: bool) -> Self {
        self.fp32 = fp32;
        self
    }

    /// Whether the products run in FP32 on a device.
    pub fn fp32(&self) -> bool {
        self.fp32 && self.ground.device.is_some()
    }

    /// Wall seconds of the products so far, and where they went.
    pub fn seconds(&self) -> (f64, Vv10Times) {
        *self.seconds.lock().unwrap()
    }
}

impl linus_base::terms::ResponseKernel for Vv10Kernel<'_> {
    fn contract_occ(
        &self,
        c_o: &[MatRef<'_, f64>],
        a: &[Vec<Mat<f64>>],
    ) -> Result<Vec<Vec<Mat<f64>>>, LinusError> {
        let t0 = std::time::Instant::now();
        let (v, tm) = self
            .nl
            .kernel_contract_occ_timed(&self.ground, c_o, a, self.fp32)?;
        let mut s = self.seconds.lock().unwrap();
        s.0 += t0.elapsed().as_secs_f64();
        s.1.add(&tm);
        Ok(v)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{XcFunctional, XcIntegrator};
    use linus_grid::{GridSpec, MolecularGrid};
    use linus_ints::basis::BasisDef;
    use linus_ints::BasisSet;

    const Z: [u32; 7] = [8, 1, 1, 6, 1, 1, 1];
    const XYZ: [[f64; 3]; 7] = [
        [0.02, -0.01, 0.22],
        [0.05, 1.43, -0.89],
        [-0.07, -1.40, -0.93],
        [2.9, 0.1, 1.4],
        [3.1, 2.0, 2.1],
        [4.4, -0.6, 0.2],
        [2.6, -1.1, 3.0],
    ];

    /// The integrator of `func` with its basis at `xyz` and its NLC grid at `grid_at` (PySCF
    /// level 0), the VV10 density threshold 1e-12 (no points cross it in the differences).
    fn integrator(func: &str, xyz: &[[f64; 3]], grid_at: &[[f64; 3]]) -> XcIntegrator {
        let def = BasisDef::load("def2-svp", None).unwrap();
        let basis = BasisSet::new(&def, &Z, xyz, true).unwrap();
        let g = MolecularGrid::build(&Z, grid_at, &GridSpec::pyscf(0)).unwrap();
        let mut xc = XcIntegrator::with_functional(
            &basis,
            g.clone(),
            &XcFunctional::new(func).unwrap(),
            Some(g),
        );
        xc.vv10.as_mut().unwrap().rho_thresh = 1e-12;
        xc
    }

    /// A total density: the SAD guess with a perturbation.
    fn density(xc: &XcIntegrator) -> Mat<f64> {
        let def = BasisDef::load("def2-svp", None).unwrap();
        let (da, db) = linus_scf::guess::sad_guess(&def, &xc.basis, &Z, 9, 9, Par::Seq);
        let n = da.nrows();
        &da + &db
            + Mat::from_fn(n, n, |i, j| {
                1e-3 * (((i + j) % 5) as f64 - 2.0) * (-((i as f64 - j as f64).abs()) / 4.0).exp()
            })
    }

    fn mat(n: usize, k: usize, seed: usize, s: f64) -> Mat<f64> {
        Mat::from_fn(n, k, |i, j| {
            s * ((((i * 7 + j * 13 + seed * 5) % 17) as f64 - 8.0) / 8.0)
                * (-((i as f64 - 3.0 * j as f64).abs()) / 9.0).exp()
        })
    }

    fn max_abs(m: &Mat<f64>) -> f64 {
        let mut x = 0.0f64;
        for j in 0..m.ncols() {
            for i in 0..m.nrows() {
                x = x.max(m[(i, j)].abs());
            }
        }
        x
    }

    /// The kernel against central differences of the VV10 potential along the first-order
    /// densities (closed shell: 2 (A C^T + C A^T); open shell: the sum of the spins').
    #[test]
    fn kernel_matches_potential_differences() {
        for func in ["wb97m-v", "b97m-v"] {
            let xc = integrator(func, &XYZ, &XYZ);
            let nl = xc.vv10().unwrap();
            let dt = density(&xc);
            let n = dt.nrows();
            let g = nl.response_ground(&dt, false);
            let pot = |d: &Mat<f64>| nl.energy_potential(d).1;
            let h = 1e-4;
            // closed shell
            let c = mat(n, 5, 0, 0.3);
            let a: Vec<Mat<f64>> = (0..3).map(|p| mat(n, 5, p + 1, 0.05)).collect();
            let v = nl
                .kernel_contract_occ(&g, &[c.as_ref()], std::slice::from_ref(&a))
                .unwrap();
            let mut err: f64 = 0.0;
            for p in 0..3 {
                let dd = (&a[p] * c.transpose() + &c * a[p].transpose()) * 2.0;
                let fd = (pot(&(&dt + &dd * h)) - pot(&(&dt - &dd * h))) * (0.5 / h) * &c;
                err = err.max(max_abs(&(&fd - &v[0][p])) / max_abs(&fd));
            }
            // open shell
            let cb = mat(n, 4, 7, 0.3);
            let ab: Vec<Mat<f64>> = (0..3).map(|p| mat(n, 4, p + 11, 0.05)).collect();
            let v2 = nl
                .kernel_contract_occ(&g, &[c.as_ref(), cb.as_ref()], &[a.clone(), ab.clone()])
                .unwrap();
            for p in 0..3 {
                let dd = &a[p] * c.transpose()
                    + &c * a[p].transpose()
                    + &ab[p] * cb.transpose()
                    + &cb * ab[p].transpose();
                let dv = (pot(&(&dt + &dd * h)) - pot(&(&dt - &dd * h))) * (0.5 / h);
                for (s, cs) in [&c, &cb].into_iter().enumerate() {
                    let fd = &dv * cs;
                    err = err.max(max_abs(&(&fd - &v2[s][p])) / max_abs(&fd));
                }
            }
            eprintln!(
                "{func}: VV10 kernel vs central differences of the potential: max rel. error {err:e}"
            );
            assert!(err < 1e-7, "{func}: kernel vs FD {err:e}");
        }
    }

    /// The explicit Hessian terms against differences of the VV10 potential matrix and of the
    /// gradient without grid response at a fixed AO density, the basis moving with the atoms and
    /// the NLC grid fixed (`grid_response` false) or moving with them (true).
    #[test]
    fn hessian_terms_match_differences() {
        let func = "wb97m-v";
        let xc = integrator(func, &XYZ, &XYZ);
        let dt = density(&xc);
        let n = dt.nrows();
        let n3 = 3 * Z.len();
        let c = mat(n, 5, 3, 0.3);
        type Pg = (Mat<f64>, Vec<[f64; 3]>);
        for grid_response in [false, true] {
            let nl = xc.vv10().unwrap();
            let g = nl.response_ground(&dt, grid_response);
            let t = nl
                .hessian_terms(&g, &[c.as_ref()], &dt, grid_response, Par::Seq)
                .unwrap();
            // (fourth-order differences: the oxygen 1s functions moving past the fixed grid's
            // innermost points leave ~4e-6 truncation error on the diagonal at h = 2e-3)
            let h = 5e-4;
            let (mut ev, mut eh, mut vmax, mut hmax) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
            for (a, k) in [(0usize, 2usize), (1, 1), (3, 0), (6, 2)] {
                let at = |s: f64| -> Pg {
                    let mut p = XYZ;
                    p[a][k] += s * h;
                    let x = integrator(func, &p, if grid_response { &p } else { &XYZ });
                    let nl = x.vv10().unwrap();
                    (nl.energy_potential(&dt).1, nl.gradient(&dt, false))
                };
                let (p1, m1, p2, m2) = (at(1.0), at(-1.0), at(2.0), at(-2.0));
                let d4 = |f: &dyn Fn(&Pg) -> f64| {
                    (8.0 * (f(&p1) - f(&m1)) - (f(&p2) - f(&m2))) / (12.0 * h)
                };
                let y = 3 * a + k;
                let mut an = t.terms.vfix[0][y].clone();
                if grid_response {
                    an += &t.terms.vgrid[0][y];
                }
                for i in 0..n {
                    for j in 0..c.ncols() {
                        let fd = d4(&|r: &Pg| (0..n).map(|m| r.0[(i, m)] * c[(m, j)]).sum::<f64>());
                        ev = ev.max((fd - an[(i, j)]).abs());
                        vmax = vmax.max(fd.abs());
                    }
                }
                for x in 0..n3 {
                    let fd = d4(&|r: &Pg| r.1[x / 3][x % 3]);
                    eh = eh.max((fd - t.terms.hess[x * n3 + y]).abs());
                    hmax = hmax.max(fd.abs());
                }
            }
            eprintln!(
                "VV10 Hessian terms (grid response {grid_response}) vs fourth-order differences: \
                 V^x C {ev:e} (max {vmax:e}), E^xy {eh:e} (max {hmax:e})"
            );
            assert!(
                ev < 1e-6 * vmax && eh < 1e-6 * hmax,
                "VV10 Hessian terms vs differences (grid response {grid_response})"
            );
        }
    }

    /// One pair primitive and one local block: the responses' ground-state potential is the
    /// SCF's, bit for bit (and so are the energy sweep's and the gradient's pair sums).
    #[test]
    fn ground_potential_is_the_scfs() {
        for func in ["wb97m-v", "b97m-v"] {
            let xc = integrator(func, &XYZ, &XYZ);
            let nl = xc.vv10().unwrap();
            let dt = density(&xc);
            let d = nl.densities(&dt);
            let g = nl.response_ground(&dt, true);
            for forces in [false, true] {
                let ev = nl.evaluate(&d, forces);
                for i in 0..ev.vrho.len() {
                    assert_eq!(ev.vrho[i].to_bits(), g.vrho[i].to_bits(), "{func} vrho {i}");
                    assert_eq!(ev.vsig[i].to_bits(), g.vsig[i].to_bits(), "{func} vsig {i}");
                }
            }
        }
    }

    /// The same bits for 1, 3 and 8 threads and for perturbation groups of any size.
    #[test]
    fn deterministic() {
        let mut xc = integrator("b97m-v", &XYZ, &XYZ);
        let dt = density(&xc);
        let n = dt.nrows();
        let c = mat(n, 5, 0, 0.3);
        let a: Vec<Mat<f64>> = (0..6).map(|p| mat(n, 5, p + 1, 0.05)).collect();
        type Out = (
            Vec<Vec<Mat<f64>>>,
            Vec<Vec<Mat<f64>>>,
            Vec<Vec<Mat<f64>>>,
            Vec<f64>,
        );
        let run = |xc: &XcIntegrator, t: usize| -> Out {
            rayon::ThreadPoolBuilder::new()
                .num_threads(t)
                .build()
                .unwrap()
                .install(|| {
                    let nl = xc.vv10().unwrap();
                    let g = nl.response_ground(&dt, true);
                    let k = nl
                        .kernel_contract_occ(&g, &[c.as_ref()], std::slice::from_ref(&a))
                        .unwrap();
                    let h = nl
                        .hessian_terms(&g, &[c.as_ref()], &dt, true, Par::Seq)
                        .unwrap();
                    (k, h.terms.vfix, h.terms.vgrid, h.terms.hess)
                })
        };
        let r = run(&xc, 3);
        for t in [1, 8] {
            assert!(run(&xc, t) == r, "VV10 response differs with {t} threads");
        }
        // groups of one perturbation (and one atom)
        xc.vv10.as_mut().unwrap().eng.kernel_budget_mb = 0;
        assert!(
            run(&xc, 4) == r,
            "VV10 response differs with the perturbation groups"
        );
    }

    /// Groups of any size give the single group's bits also where a group's products would take
    /// another of faer's kernels than the single group's: a channel with one occupied orbital (a
    /// group of one perturbation makes its products matrix-vector products) and small groups
    /// of the Hessian's atoms (its E = u^T t has one column per perturbation of the group).
    #[test]
    fn groups_of_a_one_orbital_channel_keep_the_bits() {
        let mut xc = integrator("b97m-v", &XYZ, &XYZ);
        let dt = density(&xc);
        let n = dt.nrows();
        let (ca, cb) = (mat(n, 5, 0, 0.3), mat(n, 1, 2, 0.4));
        let a: Vec<Vec<Mat<f64>>> = vec![
            (0..7).map(|p| mat(n, 5, p + 1, 0.05)).collect(),
            (0..7).map(|p| mat(n, 1, p + 11, 0.05)).collect(),
        ];
        type Out = (
            Vec<Vec<Mat<f64>>>,
            Vec<Vec<Mat<f64>>>,
            Vec<Vec<Mat<f64>>>,
            Vec<f64>,
        );
        let run = |xc: &XcIntegrator, t: usize| -> Out {
            rayon::ThreadPoolBuilder::new()
                .num_threads(t)
                .build()
                .unwrap()
                .install(|| {
                    let nl = xc.vv10().unwrap();
                    let g = nl.response_ground(&dt, true);
                    let co = [ca.as_ref(), cb.as_ref()];
                    let k = nl.kernel_contract_occ(&g, &co, &a).unwrap();
                    let h = nl.hessian_terms(&g, &co, &dt, true, Par::Seq).unwrap();
                    (k, h.terms.vfix, h.terms.vgrid, h.terms.hess)
                })
        };
        let single = run(&xc, 3);
        assert!(run(&xc, 8) == single, "8 threads");
        // groups of one perturbation and of one atom
        xc.vv10.as_mut().unwrap().eng.kernel_budget_mb = 0;
        assert!(run(&xc, 3) == single, "groups of one perturbation");
    }
}
