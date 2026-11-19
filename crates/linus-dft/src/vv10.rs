//! VV10 nonlocal correlation (Vydrov and Van Voorhis, J. Chem. Phys. 133, 244103 (2010)) of
//! the -V functionals (wB97M-V, wB97X-V, B97M-V): energy, SCF potential and analytic nuclear
//! gradient with grid response, evaluated on its own grid (the "NLC grid"). The second
//! derivatives (the CPKS kernel and the explicit Hessian terms) are [`crate::vv10_resp`]'s.
//!
//! On the NLC grid points i with total density rho_i >= `rho_thresh` (1e-8, as PySCF and Psi4):
//!
//!   E_nl = sum_i w_i rho_i [beta + 1/2 F_i],   F_i = sum_j w_j rho_j Phi_ij,
//!   Phi_ij = -3/2 / (g_i g_j (g_i + g_j)),     g_i = omega_i R_ij^2 + kappa_i,
//!   omega = sqrt(C (sigma/rho^2)^2 + 4 pi rho/3),  kappa = b 3 pi/2 (9 pi)^(-1/6) rho^(1/6),
//!   beta = (3/b^2)^(3/4) / 32,
//!
//! with b and C from libxc ([`linus_xc::Functional::vv10`]; 6.0 and 0.01 for the three -V
//! functionals) times the component weight. The potential is GGA-like (a function of rho and
//! sigma at each point, PySCF's `_vv10nlc` formulas):
//!
//!   v_rho = beta + F + 3/2 (U rho dkappa/drho + W rho domega/drho),  v_sigma = 3/2 W rho domega/dsigma,
//!   U_i = sum_j w_j rho_j T_ij (1/g_i + 1/(g_i+g_j)),  W_i = sum_j ... R_ij^2,  T_ij = 1/(g_i g_j (g_i+g_j)),
//!
//! and the matrix V_mn = sum_i w_i [v_rho phi_m phi_n + 2 v_sigma grad rho . grad(phi_m phi_n)]
//! is built by the GGA machinery of [`crate`] on the NLC grid. The nuclear gradient has
//! (i) the GGA orbital term of v_rho and v_sigma (basis functions move with their atoms), and with
//! grid response (ii) the motion of the points through the density (as for any GGA), (iii) the
//! explicit dependence of Phi on the point positions,
//!   dE/dr_i = 3 w_i rho_i sum_j w_j rho_j (omega_i Q_ij + omega_j Q_ji) (r_i - r_j),
//!   Q_ij = T_ij (1/g_i + 1/(g_i+g_j)),
//! summed into the atom that owns point i, and (iv) the weight derivatives with dE/dw_i =
//! rho_i (beta + F_i). Points below the density threshold contribute nothing (as in PySCF), so
//! the analytic gradient is the derivative of the quadrature energy except where points cross the
//! threshold.
//!
//! Pair sums (the O(N^2) part): every active point i sums over all active points j in a fixed
//! order (j ascending), eight or sixteen points i per SIMD register and one j broadcast at a time.
//! One pair primitive (`pair_g`) serves every sweep, here and in [`crate::vv10_resp`]:
//! R^2 = (|r_i|^2 + |r_j|^2) - 2 r_i . r_j over coordinates centred on the active points (the
//! cancellation costs ~1e-15 bohr^2) and g_i, g_j; the ground-state sums F, U, W take one
//! division per pair (`fuw`: all of them from n_j / P^2, P = g_i g_j g_t), the responses'
//! first-order sums another (`reciprocals`: T = 1/P and 1/g_i, 1/g_j, 1/g_t). With one local
//! block (`local1`: omega, kappa and the potential's coefficients; `local` adds the
//! responses' second derivatives) and one assembly of the potential (`potential`), the
//! responses' ground-state potential has the SCF's bits, and so do a gradient's F, U, W. Each point's sums are computed by
//! one task, so the results do not depend on the number of threads; the arithmetic is lane-wise
//! with explicit FMA (`mul_add`: one rounding on every ISA, a software fma where the instruction
//! is missing), so AVX-512, AVX2 and the baseline give the same bits. No distance screening: Phi decays only as R^-6, so a rigorous bound per pair of
//! point batches (sum w rho over each, the smallest g at the closest approach) drops nothing
//! within a molecule at a 1e-10 Eh target; the far pairs are cheap per pair anyway.

use crate::batch::BatchPlan;
use crate::{
    batch_densities, gather_group, push_grad, push_vmat, scatter_lower, sized, unpack_lower,
    GradChannel, GradOut, Kind, PointWeights, Scratch, XcIntegrator,
};
use faer::Mat;
#[cfg(test)]
use linus_grid::GridSpec;
use linus_grid::MolecularGrid;
use linus_num::det::{merge_all, FixedAcc};
use rayon::prelude::*;
use std::sync::Mutex;

/// Points with a total density below this are left out of the VV10 sums (PySCF's `_vv10nlc`
/// threshold and Psi4's `DFT_VV10_RHO_CUTOFF` default).
pub const RHO_THRESH: f64 = 1e-8;

/// VV10 on its own grid: the batch engine of [`XcIntegrator`] on the NLC grid (its functional is
/// not used) and the VV10 parameters.
pub struct Vv10 {
    pub params: linus_xc::Vv10,
    pub eng: XcIntegrator,
    /// Points with total density below this are left out ([`RHO_THRESH`]).
    pub rho_thresh: f64,
    stats: Mutex<Vv10Stats>,
}

/// What the last evaluations did (for timings and the energy component).
#[derive(Clone, Debug, Default)]
pub struct Vv10Stats {
    /// VV10 energy of the last potential evaluation (Eh, scaled; includes beta N).
    pub last_energy: Option<f64>,
    /// Active points (rho >= threshold) of the last evaluation, and all NLC grid points.
    pub active_points: usize,
    pub grid_points: usize,
    /// Accumulated seconds: densities on the NLC grid, pair sums, potential matrix, gradient
    /// contraction (orbital and point-motion terms), grid-weight derivatives.
    pub t_density: f64,
    pub t_pairs: f64,
    pub t_vmat: f64,
    pub t_grad: f64,
    pub t_weights: f64,
    pub evaluations: usize,
}

impl Vv10Stats {
    pub fn total(&self) -> f64 {
        self.t_density + self.t_pairs + self.t_vmat + self.t_grad + self.t_weights
    }

    /// The statistics of the evaluations since `before` (an earlier snapshot of the same
    /// integrator's; None: since it was built): the accumulated seconds and evaluations less
    /// `before`'s, the last evaluation's energy and points as they are.
    pub fn since(&self, before: Option<&Vv10Stats>) -> Vv10Stats {
        let Some(b) = before else {
            return self.clone();
        };
        Vv10Stats {
            t_density: self.t_density - b.t_density,
            t_pairs: self.t_pairs - b.t_pairs,
            t_vmat: self.t_vmat - b.t_vmat,
            t_grad: self.t_grad - b.t_grad,
            t_weights: self.t_weights - b.t_weights,
            evaluations: self.evaluations - b.evaluations,
            ..self.clone()
        }
    }
}

/// beta, and the prefactor of kappa = kvv rho^(1/6).
pub(crate) fn constants(p: &linus_xc::Vv10) -> (f64, f64) {
    let pi = std::f64::consts::PI;
    let kvv = p.b * 1.5 * pi * (9.0 * pi).powf(-1.0 / 6.0);
    let beta = (3.0 / (p.b * p.b)).powf(0.75) / 32.0;
    (beta, kvv)
}

/// rho, grad rho of the total density at every NLC grid point.
pub(crate) struct Dens {
    pub(crate) rho: Vec<f64>,
    pub(crate) grad: [Vec<f64>; 3],
}

/// Active points (rho >= threshold), SoA, coordinates relative to their centroid, padded to a
/// multiple of 16 with points that contribute nothing (rpw = 0, omega = kappa = 1).
struct Active {
    idx: Vec<usize>,
    x: Vec<f64>,
    y: Vec<f64>,
    z: Vec<f64>,
    om: Vec<f64>,
    ka: Vec<f64>,
    rpw: Vec<f64>,
    /// -2 x, -2 y, -2 z and |r|^2 ([`pair_g`])
    mx: Vec<f64>,
    my: Vec<f64>,
    mz: Vec<f64>,
    nn: Vec<f64>,
    /// rho dkappa/drho, rho domega/drho, rho domega/dsigma of the real points ([`local1`])
    loc: Vec<[f64; 3]>,
}

/// The local derivative coefficients of one active point: what the potential and the response
/// need of omega and kappa as functions of rho and sigma ([`local`]).
#[derive(Clone, Copy, Default, Debug)]
pub(crate) struct Local {
    /// domega/drho, domega/dsigma, dkappa/drho
    pub(crate) om_r: f64,
    pub(crate) om_s: f64,
    pub(crate) ka_r: f64,
    /// rho kappa', kappa' + rho kappa''
    pub(crate) a1: f64,
    pub(crate) a2: f64,
    /// rho omega_r, omega_r + rho omega_rr, rho omega_rs
    pub(crate) b1: f64,
    pub(crate) b2: f64,
    pub(crate) b3: f64,
    /// rho omega_s, omega_s + rho omega_rs, rho omega_ss
    pub(crate) c1: f64,
    pub(crate) c2: f64,
    pub(crate) c3: f64,
}

/// omega, kappa and the SCF potential's local coefficients of a point with density `r` and sigma
/// `s`, for the constants C (`cvv`) and kvv: omega = sqrt(X), X = C sigma^2 / rho^4 + 4 pi rho / 3,
/// kappa = kvv rho^(1/6); and rho dkappa/drho, rho domega/drho, rho domega/dsigma (the latter
/// finite at sigma = 0: PySCF's W0tmp rho / (sigma omega)). The SCF's and the responses' alike.
#[inline]
pub(crate) fn local1(r: f64, s: f64, cvv: f64, kvv: f64) -> (f64, f64, [f64; 3]) {
    let pi43 = 4.0 * std::f64::consts::PI / 3.0;
    let t = s / (r * r);
    let wt = cvv * t * t;
    let om = (wt + pi43 * r).sqrt();
    let ka = kvv * r.powf(1.0 / 6.0);
    let dodr = (0.5 * pi43 * r - 2.0 * wt) / om;
    let dods = cvv * s / (r * r * r * om);
    (om, ka, [ka / 6.0, dodr, dods])
}

/// omega, kappa and every local coefficient of the responses ([`Local`]): the SCF's
/// ([`local1`]) and the second derivatives.
pub(crate) fn local(r: f64, s: f64, cvv: f64, kvv: f64) -> (f64, f64, Local) {
    let (om, ka, [a1, b1, c1]) = local1(r, s, cvv, kvv);
    let t = s / (r * r);
    let x_rr = 20.0 * cvv * t * t / (r * r);
    let x_rs = -8.0 * cvv * t / (r * r * r);
    let x_ss = 2.0 * cvv / (r * r * r * r);
    let (om_r, om_s, ka_r) = (b1 / r, c1 / r, a1 / r);
    let om_rr = (x_rr - 2.0 * om_r * om_r) / (2.0 * om);
    let om_rs = (x_rs - 2.0 * om_r * om_s) / (2.0 * om);
    let om_ss = (x_ss - 2.0 * om_s * om_s) / (2.0 * om);
    let ka_rr = -5.0 * ka / (36.0 * r * r);
    let loc = Local {
        om_r,
        om_s,
        ka_r,
        a1,
        a2: ka_r + r * ka_rr,
        b1,
        b2: om_r + r * om_rr,
        b3: r * om_rs,
        c1,
        c2: om_s + r * om_rs,
        c3: r * om_ss,
    };
    (om, ka, loc)
}

/// v_rho and v_sigma of a point (scale `sc`) from its pair sums F (with its -3/2), U, W and its
/// local coefficients a1, b1, c1 ([`local1`]): beta + F + 3/2 (U a1 + W b1) and 3/2 W c1. The
/// SCF's and the responses' ground state alike.
#[inline]
#[allow(clippy::too_many_arguments)]
pub(crate) fn potential(sc: f64, beta: f64, f: f64, u: f64, w: f64, l: [f64; 3]) -> (f64, f64) {
    (
        sc * (beta + f + 1.5 * (u * l[0] + w * l[1])),
        sc * 1.5 * w * l[2],
    )
}

/// The pair primitive of every VV10 sweep (module docs): R^2 of points i (`xi`.. `ni` = |r_i|^2,
/// coordinates centred on the active points) and j (`mxj` = -2 x_j.. `nj` = |r_j|^2), and
/// g_i = omega_i R^2 + kappa_i, g_j.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn pair_g(
    xi: f64,
    yi: f64,
    zi: f64,
    ni: f64,
    oi: f64,
    ki: f64,
    mxj: f64,
    myj: f64,
    mzj: f64,
    nj: f64,
    oj: f64,
    kj: f64,
) -> (f64, f64, f64) {
    let r2 = xi.mul_add(mxj, yi.mul_add(myj, zi.mul_add(mzj, ni + nj)));
    (r2, r2.mul_add(oi, ki), r2.mul_add(oj, kj))
}

/// The ground-state terms of a pair (g_i, g_j; the weight n_j = w_j rho_j): P = g_i g_j g_t and
/// q = n_j / P^2 (one division), q_i = g_j (g_i + g_t) q. The sums F += P q (fused), U += q_i,
/// W += R^2 q_i (fused) are the SCF's and the responses' ground state alike.
#[inline(always)]
pub(crate) fn fuw(gi: f64, gj: f64, nj: f64) -> (f64, f64, f64) {
    let gt = gi + gj;
    let p = gi * gj * gt;
    let q = nj / (p * p);
    (p, q, gj * (gi + gt) * q)
}

/// T = 1/(g_i g_j g_t) and 1/g_i, 1/g_j, 1/g_t of a pair (one division): the responses'
/// first-order pair functions.
#[inline(always)]
pub(crate) fn reciprocals(gi: f64, gj: f64) -> (f64, f64, f64, f64) {
    let gt = gi + gj;
    let gg = gi * gj;
    let t = 1.0 / (gg * gt);
    (t, gj * gt * t, gi * gt * t, gg * t)
}

/// Per active point: F (with its -3/2), U, W, and with forces sum_j (omega_i Q_ij + omega_j Q_ji)
/// w_j rho_j (r_i - r_j).
struct Sums {
    f: Vec<f64>,
    u: Vec<f64>,
    w: Vec<f64>,
    fx: Vec<f64>,
    fy: Vec<f64>,
    fz: Vec<f64>,
}

impl Vv10 {
    /// VV10 with parameters `params` on `grid` (basis functions of `like`).
    pub(crate) fn new(like: &XcIntegrator, grid: MolecularGrid, params: linus_xc::Vv10) -> Self {
        let eng = XcIntegrator::engine(&like.basis, grid, like.func_u.clone(), like.func_p.clone());
        Vv10 {
            params,
            eng,
            rho_thresh: RHO_THRESH,
            stats: Mutex::new(Vv10Stats::default()),
        }
    }

    pub fn grid(&self) -> &MolecularGrid {
        &self.eng.grid
    }

    pub fn stats(&self) -> Vv10Stats {
        self.stats.lock().unwrap().clone()
    }

    fn kind() -> Kind {
        Kind {
            gga: true,
            mgga: false,
            restricted: true,
        }
    }

    /// rho and grad rho of the total density `dt` on the NLC grid.
    pub(crate) fn densities(&self, dt: &Mat<f64>) -> Dens {
        let eng = &self.eng;
        let n = eng.grid.len();
        let k = Self::kind();
        let dens = std::slice::from_ref(dt);
        let (parts, _, _) = eng.for_each_group(
            &BatchPlan::new(1, eng.func_thresh),
            Scratch::default,
            |s, grp| {
                gather_group(dens, &grp.funcs, &mut s.dloc);
                true
            },
            |s, b| {
                let (np, ld) = (b.np, b.ld);
                let mut v = vec![0.0; 4 * np];
                if b.nao > 0 {
                    batch_densities(b.u, k, b, s);
                    v[..np].copy_from_slice(&s.rho[..np]);
                    for c in 0..3 {
                        v[(1 + c) * np..(2 + c) * np].copy_from_slice(&s.grad[c * ld..c * ld + np]);
                    }
                }
                (b.start, v)
            },
            |_, _| (),
        );
        let mut d = Dens {
            rho: vec![0.0; n],
            grad: [vec![0.0; n], vec![0.0; n], vec![0.0; n]],
        };
        for (start, v) in parts {
            let np = v.len() / 4;
            d.rho[start..start + np].copy_from_slice(&v[..np]);
            for c in 0..3 {
                d.grad[c][start..start + np].copy_from_slice(&v[(1 + c) * np..(2 + c) * np]);
            }
        }
        d
    }

    /// The active points and their VV10 quantities.
    fn active(&self, d: &Dens) -> Active {
        let grid = &self.eng.grid;
        let (_, kvv) = constants(&self.params);
        let cvv = self.params.c;
        let idx: Vec<usize> = (0..grid.len())
            .filter(|&i| d.rho[i] >= self.rho_thresh)
            .collect();
        let n = idx.len();
        let npad = n.div_ceil(16) * 16;
        let mut a = Active {
            idx,
            x: vec![0.0; npad],
            y: vec![0.0; npad],
            z: vec![0.0; npad],
            om: vec![1.0; npad],
            ka: vec![1.0; npad],
            rpw: vec![0.0; npad],
            mx: vec![0.0; npad],
            my: vec![0.0; npad],
            mz: vec![0.0; npad],
            nn: vec![0.0; npad],
            loc: vec![[0.0; 3]; n],
        };
        // centroid of the active points (sequential sums: deterministic)
        let mut c = [0.0; 3];
        for &i in &a.idx {
            c[0] += grid.x[i];
            c[1] += grid.y[i];
            c[2] += grid.z[i];
        }
        let c = c.map(|v| if n > 0 { v / n as f64 } else { 0.0 });
        for (p, &i) in a.idx.iter().enumerate() {
            let r = d.rho[i];
            let gr = [d.grad[0][i], d.grad[1][i], d.grad[2][i]];
            let s = gr[0] * gr[0] + gr[1] * gr[1] + gr[2] * gr[2];
            let (om, ka, loc) = local1(r, s, cvv, kvv);
            a.x[p] = grid.x[i] - c[0];
            a.y[p] = grid.y[i] - c[1];
            a.z[p] = grid.z[i] - c[2];
            a.mx[p] = -2.0 * a.x[p];
            a.my[p] = -2.0 * a.y[p];
            a.mz[p] = -2.0 * a.z[p];
            a.nn[p] = a.x[p] * a.x[p] + a.y[p] * a.y[p] + a.z[p] * a.z[p];
            a.om[p] = om;
            a.ka[p] = ka;
            a.rpw[p] = grid.w[i] * r;
            a.loc[p] = loc;
        }
        a
    }

    /// Pair sums over the active points (with `forces`, also the position-derivative sums).
    fn pair_sums(a: &Active, forces: bool) -> Sums {
        const IB: usize = 64;
        let npad = a.x.len();
        let blocks: Vec<[Vec<f64>; 6]> = (0..npad.div_ceil(IB))
            .into_par_iter()
            .map(|b| {
                let r = b * IB..((b + 1) * IB).min(npad);
                let mut out: [Vec<f64>; 6] = std::array::from_fn(|_| vec![0.0; r.len()]);
                pair_block(a, r, forces, &mut out);
                out
            })
            .collect();
        let n = a.idx.len();
        let mut s = Sums {
            f: Vec::with_capacity(npad),
            u: Vec::with_capacity(npad),
            w: Vec::with_capacity(npad),
            fx: vec![],
            fy: vec![],
            fz: vec![],
        };
        for b in &blocks {
            s.f.extend_from_slice(&b[0]);
            s.u.extend_from_slice(&b[1]);
            s.w.extend_from_slice(&b[2]);
            if forces {
                s.fx.extend_from_slice(&b[3]);
                s.fy.extend_from_slice(&b[4]);
                s.fz.extend_from_slice(&b[5]);
            }
        }
        for v in [
            &mut s.f, &mut s.u, &mut s.w, &mut s.fx, &mut s.fy, &mut s.fz,
        ] {
            v.truncate(n);
        }
        for f in &mut s.f {
            *f *= -1.5;
        }
        s
    }

    /// Energy (scaled) and the per-point v_rho, v_sigma (scaled; zero on inactive points), and
    /// with `forces` the position derivative of each active point and dE/dw of every point.
    pub(crate) fn evaluate(&self, d: &Dens, forces: bool) -> Eval {
        let grid = &self.eng.grid;
        let (beta, _) = constants(&self.params);
        let sc = self.params.scale;
        let t0 = std::time::Instant::now();
        let a = self.active(d);
        let s = Self::pair_sums(&a, forces);
        let n = grid.len();
        let mut ev = Eval {
            energy: 0.0,
            vrho: vec![0.0; n],
            vsig: vec![0.0; n],
            dedw: vec![],
            force: vec![],
        };
        if forces {
            ev.dedw = vec![0.0; n];
            ev.force = Vec::with_capacity(a.idx.len());
        }
        for (p, &i) in a.idx.iter().enumerate() {
            let r = d.rho[i];
            ev.energy += grid.w[i] * r * (beta + 0.5 * s.f[p]);
            // the responses' ground state the same way (crate::xc::vv10_resp)
            (ev.vrho[i], ev.vsig[i]) = potential(sc, beta, s.f[p], s.u[p], s.w[p], a.loc[p]);
            if forces {
                ev.dedw[i] = sc * r * (beta + s.f[p]);
                let c = 3.0 * sc * a.rpw[p];
                ev.force.push((i, [c * s.fx[p], c * s.fy[p], c * s.fz[p]]));
            }
        }
        ev.energy *= sc;
        let mut st = self.stats.lock().unwrap();
        st.t_pairs += t0.elapsed().as_secs_f64();
        st.active_points = a.idx.len();
        st.grid_points = n;
        st.evaluations += 1;
        ev
    }

    /// E_nl and its potential matrix for the total density `dt` (the same for both spins).
    pub fn energy_potential(&self, dt: &Mat<f64>) -> (f64, Mat<f64>) {
        let t0 = std::time::Instant::now();
        let d = self.densities(dt);
        self.stats.lock().unwrap().t_density += t0.elapsed().as_secs_f64();
        let ev = self.evaluate(&d, false);
        let t1 = std::time::Instant::now();
        let v = self.vmat(&d, &ev.vrho, &ev.vsig);
        let mut st = self.stats.lock().unwrap();
        st.t_vmat += t1.elapsed().as_secs_f64();
        st.last_energy = Some(ev.energy);
        (ev.energy, v)
    }

    /// E_nl for the total density `dt`.
    pub fn energy(&self, dt: &Mat<f64>) -> f64 {
        let d = self.densities(dt);
        self.evaluate(&d, false).energy
    }

    /// V_mn = sum_i w_i [v_rho phi_m phi_n + 2 v_sigma grad rho . grad(phi_m phi_n)].
    fn vmat(&self, d: &Dens, vrho: &[f64], vsig: &[f64]) -> Mat<f64> {
        let eng = &self.eng;
        let grid = &eng.grid;
        let nbf = eng.basis.nbf;
        let npack = nbf * (nbf + 1) / 2;
        // batches without active points add nothing
        let inactive = |bi: usize| {
            let b = &grid.batches[bi];
            vrho[b.start..b.end].iter().all(|&v| v == 0.0)
                && vsig[b.start..b.end].iter().all(|&v| v == 0.0)
        };
        // per thread: the accumulator, scratch, and whether the group has an active batch
        let init = || (FixedAcc::new(npack), Scratch::default(), false);
        let (_, _, works) = eng.for_each_group(
            &BatchPlan::new(1, eng.func_thresh).skipping(&inactive),
            init,
            |(_, s, any), grp| {
                let u = grp.funcs.len();
                sized(&mut s.vloc, u * u).fill(0.0);
                *any = false;
                true
            },
            |(_, s, any), b| {
                let (np, ld, nao, u) = (b.np, b.ld, b.nao, b.u);
                if nao == 0 {
                    return;
                }
                *any = true;
                let (mut vr, mut vg) = (std::mem::take(&mut s.b1), std::mem::take(&mut s.b2));
                {
                    let (vr, vg) = (sized(&mut vr, ld), sized(&mut vg, 3 * ld));
                    vr.fill(0.0);
                    vg.fill(0.0);
                    for g in 0..np {
                        let i = b.start + g;
                        let w = grid.w[i];
                        vr[g] = 0.5 * w * vrho[i];
                        for c in 0..3 {
                            vg[c * ld + g] = 2.0 * w * vsig[i] * d.grad[c][i];
                        }
                    }
                }
                let pw = PointWeights {
                    vr: &vr,
                    vg: &vg,
                    vt: None,
                };
                let mut lap = crate::profile::Lap::start();
                let vloc = &mut s.vloc[..u * u];
                push_vmat(b, true, &pw, &mut s.wm, &mut s.vsub, vloc, &mut lap);
                s.b1 = vr;
                s.b2 = vg;
            },
            |(acc, s, any), grp| {
                if *any {
                    let u = grp.funcs.len();
                    let (hi, lo) = acc.parts_for(&s.vloc[..u * u]);
                    scatter_lower(hi, lo, nbf, &grp.funcs, &grp.runs, &s.vloc[..u * u]);
                    acc.tick(1);
                }
            },
        );
        match merge_all(works.into_iter().map(|x| vec![x.0]).collect()) {
            Some(t) => unpack_lower(&t[0].to_f64(), nbf),
            None => Mat::zeros(nbf, nbf),
        }
    }

    /// Nuclear gradient of E_nl at a fixed total density `dt` (natom x 3). With `grid_response`
    /// the NLC grid moves with the atoms (point motion and weight derivatives), which makes it the
    /// derivative of the quadrature energy; without, only the basis functions move (PySCF's
    /// default `grid_response = False`).
    pub fn gradient(&self, dt: &Mat<f64>, grid_response: bool) -> Vec<[f64; 3]> {
        let eng = &self.eng;
        let grid = &eng.grid;
        let natom = grid.atom_z.len();
        let t0 = std::time::Instant::now();
        let d = self.densities(dt);
        self.stats.lock().unwrap().t_density += t0.elapsed().as_secs_f64();
        let ev = self.evaluate(&d, grid_response);
        let t1 = std::time::Instant::now();
        let k = Self::kind();
        let dens = std::slice::from_ref(dt);
        let (vrho, vsig) = (&ev.vrho, &ev.vsig);
        let active = |bi: usize| {
            let b = &grid.batches[bi];
            vrho[b.start..b.end].iter().any(|&v| v != 0.0)
                || vsig[b.start..b.end].iter().any(|&v| v != 0.0)
        };
        let inactive = |bi: usize| !active(bi);
        // per thread: scratch and the group's atom contributions
        let init = || (Scratch::default(), vec![[0.0; 3]; natom]);
        let (_, parts, _) = eng.for_each_group(
            &BatchPlan::new(2, crate::GRAD_THRESH).skipping(&inactive),
            init,
            |(s, gat), grp| {
                gat.iter_mut().for_each(|g| *g = [0.0; 3]);
                // a group without active batches adds nothing (its density block is not needed)
                if !grp.batches.clone().any(active) {
                    return false;
                }
                gather_group(dens, &grp.funcs, &mut s.dloc);
                true
            },
            |(s, gat), b| {
                let (np, ld, nao, u) = (b.np, b.ld, b.nao, b.u);
                if nao == 0 {
                    return;
                }
                batch_densities(u, k, b, s);
                let (mut vr, mut vg, mut bm, mut cb) = (
                    std::mem::take(&mut s.b1),
                    std::mem::take(&mut s.b2),
                    std::mem::take(&mut s.b4),
                    std::mem::take(&mut s.b5),
                );
                {
                    let (vr, vg) = (sized(&mut vr, ld), sized(&mut vg, 3 * ld));
                    vr.fill(0.0);
                    vg.fill(0.0);
                    for g in 0..np {
                        let i = b.start + g;
                        let w = grid.w[i];
                        vr[g] = w * vrho[i];
                        for c in 0..3 {
                            vg[c * ld + g] = 2.0 * w * vsig[i] * d.grad[c][i];
                        }
                    }
                }
                // T = w v_rho phi + vg . grad phi;  B = T D, and the contraction
                let mut tsum = vec![0.0; 3 * ld];
                let gw = GradChannel {
                    k,
                    sp: 0,
                    vr: &vr,
                    vg: &vg,
                    vt: &[],
                };
                let out = GradOut {
                    gat: &mut gat[..],
                    tsum: &mut tsum,
                };
                let mut lap = crate::profile::Lap::start();
                push_grad(b, &gw, s, [&mut bm, &mut cb], out, &mut lap);
                if grid_response {
                    // point motion through the density: each point's derivative goes to its owner
                    for g in 0..np {
                        let own = grid.atom[b.start + g] as usize;
                        for c in 0..3 {
                            gat[own][c] -= tsum[c * ld + g];
                        }
                    }
                }
                (s.b1, s.b2, s.b4, s.b5) = (vr, vg, bm, cb);
            },
            |(_, gat), _| gat.clone(),
        );
        let mut gtot = vec![[0.0; 3]; natom];
        for p in &parts {
            for a in 0..natom {
                for c in 0..3 {
                    gtot[a][c] += p[a][c];
                }
            }
        }
        let t2 = std::time::Instant::now();
        if grid_response {
            // explicit position dependence of the kernel, to the owner of each point
            for &(i, f) in &ev.force {
                let own = grid.atom[i] as usize;
                for c in 0..3 {
                    gtot[own][c] += f[c];
                }
            }
            let gw = grid.weight_gradient_contract(&ev.dedw);
            for a in 0..natom {
                for c in 0..3 {
                    gtot[a][c] += gw[a][c];
                }
            }
        }
        let mut st = self.stats.lock().unwrap();
        st.t_grad += t2.duration_since(t1).as_secs_f64();
        st.t_weights += t2.elapsed().as_secs_f64();
        gtot
    }
}

/// Result of one VV10 evaluation on the NLC grid.
pub(crate) struct Eval {
    energy: f64,
    pub(crate) vrho: Vec<f64>,
    pub(crate) vsig: Vec<f64>,
    /// dE/dw_i (with forces)
    dedw: Vec<f64>,
    /// (point, dE/dr_i) of the explicit kernel dependence (with forces)
    force: Vec<(usize, [f64; 3])>,
}

/// Pair sums of the active points `r` (i) against all active points (j), into out[0..3] (F
/// without its -3/2, U, W) and with `forces` out[3..6].
fn pair_block(a: &Active, r: std::ops::Range<usize>, forces: bool, out: &mut [Vec<f64>; 6]) {
    // j in tiles that stay in L1/L2 while the block's lanes sweep them; per lane the sums run
    // over j in ascending order whatever the tiling
    const JT: usize = 1024;
    let n = a.x.len();
    let [o0, o1, o2, o3, o4, o5] = out;
    let mut j0 = 0;
    while j0 < n {
        let j1 = (j0 + JT).min(n);
        let ip = [
            &a.x[r.clone()],
            &a.y[r.clone()],
            &a.z[r.clone()],
            &a.nn[r.clone()],
            &a.om[r.clone()],
            &a.ka[r.clone()],
        ];
        if forces {
            let jp = [
                &a.x[j0..j1],
                &a.y[j0..j1],
                &a.z[j0..j1],
                &a.nn[j0..j1],
                &a.om[j0..j1],
                &a.ka[j0..j1],
                &a.rpw[j0..j1],
            ];
            pair_tile_forces(
                ip,
                jp,
                [
                    &mut o0[..],
                    &mut o1[..],
                    &mut o2[..],
                    &mut o3[..],
                    &mut o4[..],
                    &mut o5[..],
                ],
            );
        } else {
            let jp = [
                &a.mx[j0..j1],
                &a.my[j0..j1],
                &a.mz[j0..j1],
                &a.nn[j0..j1],
                &a.om[j0..j1],
                &a.ka[j0..j1],
                &a.rpw[j0..j1],
            ];
            pair_tile(ip, jp, [&mut o0[..], &mut o1[..], &mut o2[..]]);
        }
        j0 = j1;
    }
}

/// F, U, W of `L` lanes of points i against the j points of a tile (lane-wise): [`pair_g`] and
/// [`fuw`]. FMA via `mul_add`, which rounds once on every ISA (a software fma without the
/// instruction), so AVX-512, AVX2 and the baseline give the same bits. `ip`: x, y, z, |r|^2,
/// omega, kappa of the i points; `jp`: -2 x, -2 y, -2 z, |r|^2, omega, kappa, w rho of the j
/// points ([`tile_lanes_forces`]: x, y, z in place of -2 x, -2 y, -2 z).
#[inline(always)]
#[allow(clippy::needless_range_loop, reason = "indexes several arrays in step")]
fn tile_lanes<const L: usize>(ip: [&[f64]; 6], jp: [&[f64]; 7], acc: [&mut [f64]; 3]) {
    let ld = |v: &[f64]| -> [f64; L] { v[..L].try_into().unwrap() };
    let (xi, yi, zi, ni, oi, ki) = (
        ld(ip[0]),
        ld(ip[1]),
        ld(ip[2]),
        ld(ip[3]),
        ld(ip[4]),
        ld(ip[5]),
    );
    let [af, au, aw] = acc;
    let (mut f, mut u, mut w) = (ld(af), ld(au), ld(aw));
    let nj = jp[0].len();
    let [jmx, jmy, jmz, jnn, jom, jka, jrw] = jp.map(|v| &v[..nj]);
    for j in 0..nj {
        let (mx, my, mz, nnj) = (jmx[j], jmy[j], jmz[j], jnn[j]);
        let (oj, kj, rj) = (jom[j], jka[j], jrw[j]);
        for l in 0..L {
            let (r2, gi, gj) = pair_g(
                xi[l], yi[l], zi[l], ni[l], oi[l], ki[l], mx, my, mz, nnj, oj, kj,
            );
            let (p, q, qi) = fuw(gi, gj, rj);
            f[l] = p.mul_add(q, f[l]);
            u[l] += qi;
            w[l] = qi.mul_add(r2, w[l]);
        }
    }
    af[..L].copy_from_slice(&f);
    au[..L].copy_from_slice(&u);
    aw[..L].copy_from_slice(&w);
}

/// [`tile_lanes`] plus the position-derivative sums
/// sum_j (omega_i Q_ij + omega_j Q_ji) w_j rho_j (r_i - r_j), Q_ij = T (1/g_i + 1/g_t).
#[inline(always)]
#[allow(clippy::needless_range_loop, reason = "indexes several arrays in step")]
fn tile_lanes_forces<const L: usize>(ip: [&[f64]; 6], jp: [&[f64]; 7], acc: [&mut [f64]; 6]) {
    let ld = |v: &[f64]| -> [f64; L] { v[..L].try_into().unwrap() };
    let (xi, yi, zi, ni, oi, ki) = (
        ld(ip[0]),
        ld(ip[1]),
        ld(ip[2]),
        ld(ip[3]),
        ld(ip[4]),
        ld(ip[5]),
    );
    let [af, au, aw, ax, ay, az] = acc;
    let (mut f, mut u, mut w) = (ld(af), ld(au), ld(aw));
    let (mut fx, mut fy, mut fz) = (ld(ax), ld(ay), ld(az));
    let nj = jp[0].len();
    let [jx, jy, jz, jnn, jom, jka, jrw] = jp.map(|v| &v[..nj]);
    for j in 0..nj {
        let (xj, yj, zj, nnj) = (jx[j], jy[j], jz[j], jnn[j]);
        let (oj, kj, rj) = (jom[j], jka[j], jrw[j]);
        // -2 r_j, exact: the energy tile's arrays bit for bit
        let (mx, my, mz) = (-2.0 * xj, -2.0 * yj, -2.0 * zj);
        for l in 0..L {
            let (r2, gi, gj) = pair_g(
                xi[l], yi[l], zi[l], ni[l], oi[l], ki[l], mx, my, mz, nnj, oj, kj,
            );
            let (p, q, qi) = fuw(gi, gj, rj);
            f[l] = p.mul_add(q, f[l]);
            u[l] += qi;
            w[l] = qi.mul_add(r2, w[l]);
            let qj = gi * (gj + (gi + gj)) * q;
            let c = oi[l].mul_add(qi, oj * qj);
            fx[l] = c.mul_add(xi[l] - xj, fx[l]);
            fy[l] = c.mul_add(yi[l] - yj, fy[l]);
            fz[l] = c.mul_add(zi[l] - zj, fz[l]);
        }
    }
    af[..L].copy_from_slice(&f);
    au[..L].copy_from_slice(&u);
    aw[..L].copy_from_slice(&w);
    ax[..L].copy_from_slice(&fx);
    ay[..L].copy_from_slice(&fy);
    az[..L].copy_from_slice(&fz);
}

/// All lanes of a block of points i (length a multiple of 16) against one tile, `L` at a time.
#[inline(always)]
fn tile_body<const L: usize>(ip: [&[f64]; 6], jp: [&[f64]; 7], acc: [&mut [f64]; 3]) {
    let [af, au, aw] = acc;
    let ni = ip[0].len();
    for c in (0..ni).step_by(L) {
        let r = c..c + L;
        tile_lanes::<L>(
            ip.map(|v| &v[r.clone()]),
            jp,
            [&mut af[r.clone()], &mut au[r.clone()], &mut aw[r]],
        );
    }
}

#[inline(always)]
fn tile_forces_body<const L: usize>(ip: [&[f64]; 6], jp: [&[f64]; 7], acc: [&mut [f64]; 6]) {
    let [af, au, aw, ax, ay, az] = acc;
    let ni = ip[0].len();
    for c in (0..ni).step_by(L) {
        let r = c..c + L;
        tile_lanes_forces::<L>(
            ip.map(|v| &v[r.clone()]),
            jp,
            [
                &mut af[r.clone()],
                &mut au[r.clone()],
                &mut aw[r.clone()],
                &mut ax[r.clone()],
                &mut ay[r.clone()],
                &mut az[r],
            ],
        );
    }
}

// Dispatch: 16 lanes (two zmm per quantity) with AVX-512, 8 with AVX2 and the baseline. The
// lane arithmetic is the same, so all give the same bits.
linus_num::multiversion_lanes!(fn pair_tile(ip: [&[f64]; 6], jp: [&[f64]; 7], acc: [&mut [f64]; 3]) -> () = tile_body::<16 | 8>);
linus_num::multiversion_lanes!(fn pair_tile_forces(ip: [&[f64]; 6], jp: [&[f64]; 7], acc: [&mut [f64]; 6]) -> () = tile_forces_body::<16 | 8>);

// ------------------------------------------------------------------ XcIntegrator hooks

impl XcIntegrator {
    /// VV10 of the functional, if it has one, on its NLC grid.
    pub fn vv10(&self) -> Option<&Vv10> {
        self.vv10.as_deref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use faer::Par;
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

    fn setup(func: &str, xyz: &[[f64; 3]], nlc: &GridSpec) -> (XcIntegrator, Mat<f64>, Mat<f64>) {
        let def = BasisDef::load("def2-svp", None).unwrap();
        let basis = BasisSet::new(&def, &Z, xyz, true).unwrap();
        let (da, db) = linus_scf::guess::sad_guess(&def, &basis, &Z, 9, 9, Par::Seq);
        let n = basis.nbf;
        let db = &db
            + Mat::from_fn(n, n, |i, j| {
                1e-3 * (((i + j) % 5) as f64 - 2.0) * (-((i as f64 - j as f64).abs()) / 4.0).exp()
            });
        let g = MolecularGrid::build(&Z, xyz, &GridSpec::pyscf(1)).unwrap();
        let xc = XcIntegrator::with_functional(
            &basis,
            g,
            &crate::XcFunctional::new(func).unwrap(),
            Some(MolecularGrid::build(&Z, xyz, nlc).unwrap()),
        );
        (xc, da, db)
    }

    /// Straightforward double loop (PySCF's `_vv10nlc` in Python form).
    fn reference(nl: &Vv10, dt: &Mat<f64>) -> (f64, Vec<f64>, Vec<f64>) {
        let d = nl.densities(dt);
        let grid = nl.grid();
        let (beta, kvv) = constants(&nl.params);
        let (b, c) = (nl.params.b, nl.params.c);
        let _ = b;
        let pi43 = 4.0 * std::f64::consts::PI / 3.0;
        let act: Vec<usize> = (0..grid.len())
            .filter(|&i| d.rho[i] >= RHO_THRESH)
            .collect();
        let sig = |i: usize| d.grad[0][i].powi(2) + d.grad[1][i].powi(2) + d.grad[2][i].powi(2);
        let om = |i: usize| (c * (sig(i) / d.rho[i].powi(2)).powi(2) + pi43 * d.rho[i]).sqrt();
        let ka = |i: usize| kvv * d.rho[i].powf(1.0 / 6.0);
        let (mut e, mut vr, mut vs) = (0.0, vec![0.0; grid.len()], vec![0.0; grid.len()]);
        for &i in &act {
            let (mut f, mut u, mut w) = (0.0, 0.0, 0.0);
            for &j in &act {
                let r2 = (grid.x[i] - grid.x[j]).powi(2)
                    + (grid.y[i] - grid.y[j]).powi(2)
                    + (grid.z[i] - grid.z[j]).powi(2);
                let gp = r2 * om(j) + ka(j);
                let g = r2 * om(i) + ka(i);
                let gt = g + gp;
                let t = d.rho[j] * grid.w[j] / (g * gp * gt);
                f += t;
                let t = t * (1.0 / g + 1.0 / gt);
                u += t;
                w += t * r2;
            }
            f *= -1.5;
            let r = d.rho[i];
            let wt = c * (sig(i) / (r * r)).powi(2);
            e += grid.w[i] * r * (beta + 0.5 * f);
            vr[i] = beta + f + 1.5 * (u * ka(i) / 6.0 + w * (0.5 * pi43 * r - 2.0 * wt) / om(i));
            vs[i] = 1.5 * w * wt * r / (sig(i) * om(i));
        }
        (e, vr, vs)
    }

    /// Energy and per-point potential against the double loop; the potential matrix against
    /// central differences of the energy; the same bits for 1, 3 and 8 threads.
    #[test]
    fn energy_potential_reference_fd_deterministic() {
        let (xc, da, db) = setup("wb97m-v", &XYZ, &GridSpec::pyscf(0));
        let nl = xc.vv10().unwrap();
        assert_eq!(
            (nl.params.b, nl.params.c, nl.params.scale),
            (6.0, 0.01, 1.0)
        );
        let dt = &da + &db;
        let (e_ref, vr_ref, vs_ref) = reference(nl, &dt);
        let d = nl.densities(&dt);
        let ev = nl.evaluate(&d, false);
        assert!(
            (ev.energy - e_ref).abs() < 1e-13 * e_ref.abs(),
            "E {} vs {}",
            ev.energy,
            e_ref
        );
        let mx = |v: &[f64]| v.iter().fold(0.0f64, |m, x| m.max(x.abs()));
        let dv = vr_ref
            .iter()
            .zip(&ev.vrho)
            .fold(0.0f64, |m, (a, b)| m.max((a - b).abs()));
        let ds = vs_ref
            .iter()
            .zip(&ev.vsig)
            .fold(0.0f64, |m, (a, b)| m.max((a - b).abs()));
        assert!(
            dv < 1e-12 * mx(&vr_ref) && ds < 1e-11 * mx(&vs_ref),
            "v_rho {dv:e} v_sigma {ds:e}"
        );
        let run = |t: usize| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(t)
                .build()
                .unwrap()
                .install(|| nl.energy_potential(&dt))
        };
        let (e1, v1) = run(3);
        assert!((e1 - e_ref).abs() < 1e-13 * e_ref.abs());
        for t in [1, 8] {
            let (e, v) = run(t);
            assert_eq!(e.to_bits(), e1.to_bits(), "energy differs with {t} threads");
            assert!(v == v1, "potential differs with {t} threads");
        }
        // tr(V dD) against central differences of E along a symmetric direction
        let n = dt.nrows();
        let dd = Mat::from_fn(n, n, |i, j| {
            1e-2 * ((((i + j) * 7) % 11) as f64 - 5.0) / 5.0
                * (-((i as f64 - j as f64).abs()) / 6.0).exp()
        });
        let h = 1e-4;
        let fd = (nl.energy(&(&dt + &dd * h)) - nl.energy(&(&dt - &dd * h))) / (2.0 * h);
        let an: f64 = (0..n)
            .flat_map(|i| (0..n).map(move |j| (i, j)))
            .map(|(i, j)| v1[(i, j)] * dd[(i, j)])
            .sum();
        assert!(
            (fd - an).abs() < 1e-7 * an.abs().max(1e-3),
            "tr(V dD) {an} vs FD {fd}"
        );
    }

    /// Analytic gradient at fixed AO density (with grid response) against central differences
    /// of the VV10 energy with the atoms, basis and NLC grid displaced; translational invariance;
    /// the same bits for 1 and 6 threads.
    #[test]
    fn gradient_fd_and_deterministic() {
        let spec = GridSpec::pyscf(0);
        let (xc, da, db) = setup("b97m-v", &XYZ, &spec);
        let dt = &da + &db;
        let nl = xc.vv10().unwrap();
        let run = |t: usize, resp: bool| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(t)
                .build()
                .unwrap()
                .install(|| nl.gradient(&dt, resp))
        };
        let g = run(2, true);
        assert_eq!(run(1, true), g, "gradient differs with 1 thread");
        assert_eq!(run(6, true), g, "gradient differs with 6 threads");
        let sum: Vec<f64> = (0..3)
            .map(|c| g.iter().map(|a| a[c]).sum::<f64>())
            .collect();
        let gmax = g.iter().flatten().fold(0.0f64, |m, x| m.max(x.abs()));
        assert!(
            sum.iter().all(|s| s.abs() < 1e-9 * gmax.max(1.0)),
            "translational invariance {sum:?} (max {gmax:e})"
        );
        let h = 1e-4;
        let e_at = |xyz: &[[f64; 3]]| -> f64 {
            let (x, _, _) = setup("b97m-v", xyz, &spec);
            x.vv10().unwrap().energy(&dt)
        };
        let mut err: f64 = 0.0;
        for (a, c) in [(0usize, 2usize), (1, 1), (3, 0), (6, 2)] {
            let mut p = XYZ;
            p[a][c] += h;
            let mut m = XYZ;
            m[a][c] -= h;
            let fd = (e_at(&p) - e_at(&m)) / (2.0 * h);
            err = err.max((fd - g[a][c]).abs());
        }
        eprintln!(
            "VV10 gradient vs central differences (h = 1e-4, threshold 1e-8): max error {err:e}"
        );
        assert!(
            err < 2e-7,
            "VV10 gradient vs FD: max error {err:e} (max |g| {gmax:e})"
        );
        // tight check: no points cross the density threshold (1e-12 here), fourth-order differences
        let tight = |xyz: &[[f64; 3]]| -> Vv10 {
            let (x, _, _) = setup("b97m-v", xyz, &spec);
            let mut v = Vv10::new(
                &x,
                x.vv10().unwrap().grid().clone(),
                x.vv10().unwrap().params,
            );
            v.rho_thresh = 1e-12;
            v
        };
        let gt = tight(&XYZ).gradient(&dt, true);
        let h = 2e-3;
        let mut err4: f64 = 0.0;
        for (a, c) in [(0usize, 2usize), (1, 1), (2, 0), (3, 0), (6, 2)] {
            let e = |s: f64| {
                let mut p = XYZ;
                p[a][c] += s * h;
                tight(&p).energy(&dt)
            };
            let fd = (8.0 * (e(1.0) - e(-1.0)) - (e(2.0) - e(-2.0))) / (12.0 * h);
            err4 = err4.max((fd - gt[a][c]).abs());
        }
        eprintln!(
            "VV10 gradient vs fourth-order differences (threshold 1e-12): max error {err4:e}"
        );
        assert!(err4 < 1e-9, "VV10 gradient vs fourth-order FD: {err4:e}");
        // without grid response the gradient is the orbital term only
        let g0 = run(3, false);
        assert!(g0.iter().zip(&g).any(|(a, b)| a != b));
    }
}
