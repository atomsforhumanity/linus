//! The analytic nuclear gradient of compensated grid-K ([`super`]) at fixed densities: the exact
//! derivative of the energy the builds report, every grid and fit dependence included (the
//! paper's eqs. 37-63, derived again here in matrix form).
//!
//! For one spin channel with E_K = 1/2 D0 . K\[D0\] + D_d . K\[D0\] + E^grid(D_d):
//!
//! * **exact**: 1/2 D0 dERI D0 + D0 dERI D_d (four-center derivative integrals; D0 is
//!   atom-block-diagonal and the same at every geometry, so it has no response of its own);
//! * **grid**: E^grid = 1/2 sum_A w_A Ybar_A . Zbar_A depends on the positions through the weights
//!   (Becke partition, points moving with their atoms), the collocation X (functions and points
//!   move), the potential integrals V^A (centres and points) and the transfer matrix M. With
//!   Lbar = W Zbar D_d (dE/dXbar) the pseudo-densities are: weights eps_A = 1/2 Ybar_A . Zbar_A +
//!   (X G_g X^T)_AA; collocation L = Lbar M + 2 W X G_g; integrals Gamma^A_pq = 1/2 w_A Ybar_Ap
//!   Ybar_Aq; overlap G_S (below). The fit's terms come from Pi = Lbar^T X = dE/dM. With
//!   M = B T A^T the pseudo-density of T = Omega^-1/2 is Pt = sym(B^T Pi A); in Omega's
//!   eigenbasis that of Omega is G_O' = Pt' o L(theta), L_ij = (theta_i^-1/2 - theta_j^-1/2) /
//!   (theta_i - theta_j) = -1 / (sqrt(theta_i theta_j) (sqrt(theta_i) + sqrt(theta_j))) (the
//!   divided difference of x^-1/2: the paper's adjoint Sylvester equation, solved in the
//!   eigenbasis); that of the grid overlap S^grid = X^T W X is G_g = A G_O A^T. The analytic
//!   overlap enters through A and B (dA = -1/2 S^+ dS A, dB = (1 - P/2) dS A with S^+ = A A^T,
//!   P = A B^T on the retained space, which is held fixed as in the SCF's own gradient):
//!   G_S = sym[(1 - P/2) Pi A T A^T - 1/2 A T B^T Pi S^+ - S^+ S^grid A G_O A^T].
//!
//! Every term is checked against finite differences of the energy in the tests.

use super::{collocation_deriv, ld_of, sym, GridK, GridKReference, CHUNK};
use faer::{Mat, MatRef, Par};
use linus_base::error::LinusError;
use linus_base::linalg::{mat_to_rowmajor, mm};
use linus_core::gradient::Gradient;
use linus_ints::contract::DensityTerms;
use linus_ints::points::grid_pot::PotPrim;
use linus_ints::points::grid_pot_deriv::PotDerivContract;
use linus_ints::{DerivOrder, Eri4DerivOptions, Eri4DerivPlan, OneEPlan, PairSet};
use rayon::prelude::*;

/// The parts of the gradient (natom rows each), the channels' coefficients applied.
#[derive(Clone, Debug, Default)]
pub struct GridKGradParts {
    /// The exact four-center terms of the SAD densities.
    pub exact: Vec<[f64; 3]>,
    /// The partition-weight derivatives.
    pub weights: Vec<[f64; 3]>,
    /// The basis functions on the grid (their centres and the points).
    pub colloc: Vec<[f64; 3]>,
    /// The potential integrals (their centres and the points).
    pub potential: Vec<[f64; 3]>,
    /// The analytic overlap in the transfer matrix.
    pub overlap: Vec<[f64; 3]>,
}

impl GridKGradParts {
    pub fn total(&self) -> Vec<[f64; 3]> {
        let mut g = Gradient::from(self.exact.clone());
        for p in [&self.weights, &self.colloc, &self.potential, &self.overlap] {
            g += p.clone();
        }
        g.into_inner()
    }
}

impl GridK {
    /// The gradient of sum_s kappa_s E_K\[D_s\] (the densities one spin channel each: a closed
    /// shell's alpha density with kappa = -2 a_x, an open shell's two with -a_x each; the
    /// operator's coefficients are the builder's). `pairs`: index-ordered shell pairs of
    /// derivative order one or more (the exact terms).
    pub fn gradient(
        &mut self,
        dens: &[Mat<f64>],
        kappa: &[f64],
        pairs: &PairSet,
        par: Par,
    ) -> Result<GridKGradParts, LinusError> {
        if dens.len() != kappa.len() {
            return Err(LinusError::Internal(format!(
                "grid-K gradient: {} densities, {} coefficients",
                dens.len(),
                kappa.len()
            )));
        }
        let natom = self.basis.natom();
        let zero = || vec![[0.0; 3]; natom];
        let mut parts = GridKGradParts {
            exact: zero(),
            weights: zero(),
            colloc: zero(),
            potential: zero(),
            overlap: zero(),
        };
        let deltas: Vec<Mat<f64>> = dens.iter().map(|d| self.delta(d)).collect();
        // exact terms: 1/2 D0 dERI D0 + 1/2 (D0 dERI D_d + D_d dERI D0) per channel
        if let Some(d0) = self.reference.as_ref().map(GridKReference::d0) {
            let d0r = mat_to_rowmajor(d0.as_ref());
            let ddr: Vec<Vec<f64>> = deltas.iter().map(|d| mat_to_rowmajor(d.as_ref())).collect();
            let ksum: f64 = kappa.iter().sum();
            let mut terms = DensityTerms::default().exchange(0.5 * ksum, &d0r, &d0r);
            for (dd, &k) in ddr.iter().zip(kappa) {
                terms = terms
                    .exchange(0.5 * k, &d0r, dd)
                    .exchange(0.5 * k, dd, &d0r);
            }
            let opts = Eri4DerivOptions {
                thresh: self.cfg.exact_thresh,
                op: self.cfg.xop,
            };
            let plan = Eri4DerivPlan::new(&self.basis, pairs, DerivOrder::One, opts)
                .map_err(|e| LinusError::Internal(format!("grid-K gradient: {e}")))?;
            let mut ws = plan.workspace(0);
            plan.gradient(&terms, &mut parts.exact, &mut ws);
        }
        // the grid's terms
        let xs = collocation_deriv(&self.basis, &self.grid, 1);
        let (x, gx) = (&xs[0], &xs[1..]);
        let npts = self.grid.len();
        let n = self.basis.nbf;
        let w = &self.grid.w;
        let mut eps = vec![0.0; npts];
        let mut lcol = Mat::<f64>::zeros(npts, n);
        let mut pi_sum = Mat::<f64>::zeros(n, n);
        for (dd, &k) in deltas.iter().zip(kappa) {
            let y = mm(self.xbar.as_ref(), dd.as_ref(), par);
            let (z, _) = self.potential(y.as_ref());
            for g in 0..npts {
                let mut s = 0.0;
                for p in 0..n {
                    s += y[(g, p)] * z[(g, p)];
                }
                eps[g] += k * 0.5 * s;
            }
            let wz = Mat::from_fn(npts, n, |g, p| w[g] * z[(g, p)]);
            // Lbar = W Zbar D_d: dE/dXbar
            let lbar = mm(wz.as_ref(), dd.as_ref(), par);
            match &self.fit {
                Some(f) => {
                    lcol += &(mm(lbar.as_ref(), f.m.as_ref(), par) * k);
                    pi_sum += &(mm(lbar.transpose(), x.as_ref(), par) * k);
                }
                None => lcol += &(&lbar * k),
            }
            let pot = self.potential_gradient(y.as_ref(), 0.5 * k);
            add(&mut parts.potential, &pot);
        }
        if let Some(f) = &self.fit {
            let (gg, gs) = fit_response(f, pi_sum.as_ref(), x.as_ref(), w, par);
            // eps += (X G_g X^T)_AA; L += 2 W X G_g
            let xg = mm(x.as_ref(), gg.as_ref(), par);
            for g in 0..npts {
                let mut s = 0.0;
                for p in 0..n {
                    s += xg[(g, p)] * x[(g, p)];
                }
                eps[g] += s;
                for p in 0..n {
                    lcol[(g, p)] += 2.0 * w[g] * xg[(g, p)];
                }
            }
            let gsr = mat_to_rowmajor(gs.as_ref());
            let plan = OneEPlan::new(&self.basis, DerivOrder::One);
            let mut wk = plan.workspace();
            plan.overlap_kinetic_grad(Some(&gsr), None, &mut parts.overlap, &mut wk);
        }
        parts.weights = self.grid.weight_gradient_contract(&eps);
        parts.colloc = self.colloc_gradient(lcol.as_ref(), gx);
        Ok(parts)
    }

    /// sum_A sum_p L_Ap dX_Ap/dR: the functions move with their atoms (-grad phi), the points
    /// with theirs (+grad phi).
    fn colloc_gradient(&self, l: MatRef<'_, f64>, gx: &[Mat<f64>]) -> Vec<[f64; 3]> {
        let natom = self.basis.natom();
        let n = self.basis.nbf;
        let atom_of: Vec<usize> = self
            .basis
            .shells
            .iter()
            .flat_map(|s| std::iter::repeat_n(s.atom, s.nfunc()))
            .collect();
        let mut g = vec![[0.0; 3]; natom];
        for (c, gxc) in gx.iter().enumerate() {
            for p in 0..n {
                let mut colsum = 0.0;
                for a in 0..l.nrows() {
                    let v = l[(a, p)] * gxc[(a, p)];
                    colsum += v;
                    g[self.grid.atom[a] as usize][c] += v;
                }
                g[atom_of[p]][c] -= colsum;
            }
        }
        g
    }

    /// The potential integrals' derivatives contracted with Gamma^A_pq = c w_A Ybar_Ap Ybar_Aq
    /// (summed over all p, q): the shells' centres and the points (moving with their atoms).
    fn potential_gradient(&self, y: MatRef<'_, f64>, c: f64) -> Vec<[f64; 3]> {
        let grid = &*self.grid;
        let n = self.basis.nbf;
        let natom = self.basis.natom();
        let shells = &self.basis.shells;
        let op = &self.cfg.xop;
        let chunks = grid.sub_batches(CHUNK);
        let parts: Vec<Vec<[f64; 3]>> = chunks
            .par_iter()
            .map(|ch| {
                let np = ch.len();
                let ld = ld_of(np);
                let (px, py, pz) = (
                    &grid.x[ch.start..ch.end],
                    &grid.y[ch.start..ch.end],
                    &grid.z[ch.start..ch.end],
                );
                let mut f = vec![0.0; n * ld];
                for k in 0..n {
                    for p in 0..np {
                        f[k * ld + p] = y[(ch.start + p, k)];
                    }
                }
                let fmax: Vec<f64> = shells
                    .iter()
                    .map(|sh| {
                        (sh.ao_start..sh.ao_start + sh.nfunc())
                            .flat_map(|k| f[k * ld..k * ld + np].iter())
                            .fold(0.0f64, |m, v| m.max(v.abs()))
                    })
                    .collect();
                let wsum: f64 = grid.w[ch.start..ch.end].iter().sum();
                let mut gbuf = vec![0.0; n * ld];
                let mut gpt = vec![0.0; 3 * ld];
                let mut out = vec![[0.0; 3]; natom];
                let mut prims: Vec<PotPrim> = Vec::new();
                for pair in &self.pairs {
                    // the pair's share: 1/2 w y_a y_b for a block on the diagonal, w y_a y_b
                    // (both orders) off it
                    let cp = if pair.sa == pair.sb { 0.5 * c } else { c };
                    let fm = fmax[pair.sa] * fmax[pair.sb] * wsum * cp.abs() * 1e2;
                    if fm * pair.est < self.cfg.thresh {
                        continue;
                    }
                    prims.clear();
                    let e =
                        pair.screen_prims(ch.center, ch.radius, self.cfg.thresh / fm, &mut prims);
                    if e * fm < self.cfg.thresh || prims.is_empty() {
                        continue;
                    }
                    // contract_deriv's pair density is 2 c_t w y_a y_b: c_t = c off the
                    // diagonal (both orders), c / 2 on it
                    let terms = [(0usize, 0usize, cp)];
                    let mut pc = PotDerivContract {
                        f: &f,
                        g: &mut gbuf,
                        ng: 0,
                        dstride: n * ld,
                        ld,
                        a0: shells[pair.sa].ao_start,
                        b0: shells[pair.sb].ao_start,
                        do_a: false,
                        do_b: false,
                        terms: &terms,
                        w: &grid.w[ch.start..ch.end],
                        gpt: &mut gpt,
                        cart: false,
                    };
                    let [ga, gb] = pair.contract_deriv(&prims, op, px, py, pz, &mut pc);
                    let (aa, ab) = (shells[pair.sa].atom, shells[pair.sb].atom);
                    for k in 0..3 {
                        out[aa][k] += ga[k];
                        out[ab][k] += gb[k];
                    }
                }
                for p in 0..np {
                    let a = grid.atom[ch.start + p] as usize;
                    for k in 0..3 {
                        out[a][k] += gpt[k * ld + p];
                    }
                }
                out
            })
            .collect();
        let mut g = vec![[0.0; 3]; natom];
        for p in &parts {
            add(&mut g, p);
        }
        g
    }
}

fn add(g: &mut [[f64; 3]], x: &[[f64; 3]]) {
    for (a, b) in g.iter_mut().zip(x) {
        for k in 0..3 {
            a[k] += b[k];
        }
    }
}

/// The response of the transfer matrix to the pseudo-density `pi` = dE/dM: G_g (the
/// pseudo-density of S^grid) and G_S (of the analytic overlap, symmetric); see the module docs.
fn fit_response(
    f: &super::OverlapFit,
    pi: MatRef<'_, f64>,
    x: MatRef<'_, f64>,
    w: &[f64],
    par: Par,
) -> (Mat<f64>, Mat<f64>) {
    let (a, b, t) = (f.a.as_ref(), f.b.as_ref(), f.t.as_ref());
    let m = f.nmo();
    let n = a.nrows();
    // Pt = sym(B^T Pi A)
    let mut pt = mm(mm(b.transpose(), pi, par).as_ref(), a, par);
    sym(&mut pt);
    // in Omega's eigenbasis: G_O' = Pt' o L(theta)
    let th = &f.theta;
    let v = f.vecs.as_ref();
    let ptp = mm(mm(v.transpose(), pt.as_ref(), par).as_ref(), v, par);
    let go_p = Mat::from_fn(m, m, |i, j| {
        let (si, sj) = (th[i].sqrt(), th[j].sqrt());
        -ptp[(i, j)] / (si * sj * (si + sj))
    });
    let go = mm(mm(v, go_p.as_ref(), par).as_ref(), v.transpose(), par);
    let ago = mm(a, go.as_ref(), par);
    let gg = mm(ago.as_ref(), a.transpose(), par);
    // G_S = (1 - P/2) Pi A T A^T - 1/2 A T B^T Pi S^+ - S^+ S^grid A G_O A^T
    let sp = mm(a, a.transpose(), par);
    let p = mm(a, b.transpose(), par);
    let att = mm(mm(a, t, par).as_ref(), a.transpose(), par);
    let pi_att = mm(pi, att.as_ref(), par);
    let g1 = &pi_att - &(mm(p.as_ref(), pi_att.as_ref(), par) * 0.5);
    let atbt = mm(mm(a, t, par).as_ref(), b.transpose(), par);
    let g3 = mm(mm(atbt.as_ref(), pi, par).as_ref(), sp.as_ref(), par) * -0.5;
    let wx = Mat::from_fn(x.nrows(), n, |g, k| w[g] * x[(g, k)]);
    let sg = mm(x.transpose(), wx.as_ref(), par);
    let g4 = mm(mm(sp.as_ref(), sg.as_ref(), par).as_ref(), gg.as_ref(), par) * -1.0;
    let mut gs = &(&g1 + &g3) + &g4;
    sym(&mut gs);
    (gg, gs)
}

/// The two-electron gradient of jk rij_cgk: RI-J of the total density with auxiliary basis
/// `aux`, plus the compensated grid-K gradient of the builder `k` (None without exact exchange)
/// at the spin densities `da`, `db` (`restricted`: a closed shell, `da` its alpha density) for
/// the exact exchange `xop` (a_x K of 1/r for HF and global hybrids; the operator with its
/// coefficients for range-separated ones, [`linus_base::terms::exchange_parts`]).
#[allow(clippy::too_many_arguments)] // bases, densities, the run's parallelism and timers
pub fn rij_cgk_gradient(
    basis: &linus_ints::BasisSet,
    pairs: &PairSet,
    aux: &linus_ints::BasisSet,
    k: Option<GridK>,
    restricted: bool,
    da: &Mat<f64>,
    db: &Mat<f64>,
    xop: Option<linus_ints::operator::Operator>,
    par: Par,
    timers: &mut linus_core::timing::Timers,
) -> Result<Vec<[f64; 3]>, LinusError> {
    let t = std::time::Instant::now();
    let dt = da + db;
    let mut g = crate::df::rij_gradient::rij_gradient(
        basis,
        pairs,
        aux,
        &dt,
        &linus_ints::eri3c::DfThresholds::default(),
        par,
        timers,
    );
    timers.add("grad_rij", t.elapsed().as_secs_f64());
    let (a_x, op) = linus_base::terms::exchange_parts(xop);
    if let (Some(mut k), true) = (k, a_x != 0.0) {
        let t = std::time::Instant::now();
        use linus_base::terms::JkBuilder;
        k.set_exchange_operator(op.unwrap_or(linus_ints::operator::Operator::Coulomb));
        let (dens, kappa) = if restricted {
            (vec![da.clone()], vec![-2.0 * a_x])
        } else {
            (vec![da.clone(), db.clone()], vec![-a_x; 2])
        };
        let parts = k.gradient(&dens, &kappa, pairs, par)?;
        add(&mut g, &parts.total());
        timers.add("grad_gridk", t.elapsed().as_secs_f64());
    }
    Ok(g)
}
