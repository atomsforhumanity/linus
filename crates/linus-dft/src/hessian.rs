//! Exchange-correlation pieces of analytic Kohn-Sham Hessians over spin channels, at fixed
//! channel densities: the explicit terms on a fixed grid (no grid-weight or point-motion terms,
//! as in PySCF's Hessians) and the first-order grid terms of a grid moving with the atoms.
//!
//! One channel is a closed shell (the total density D, the unpolarized functional), two are the
//! spins of an open shell (D_a, D_b, the polarized functional). Per channel s the density
//! variables are u_s = (rho_s, grad rho_s, tau_s) (LDA: rho; GGA: + grad rho; meta-GGA: + tau),
//! with the "effective" derivatives v = df/du and F = d2f/du2 of `crate::points` (the
//! chain rule through the sigmas folded in; F couples the channels of an open shell). Every
//! formula below is per channel (D, X, v of channel s) except the kernel term, whose F sums over
//! both. The code is written once, generic over the channel count. Moving atom A along i moves
//! its basis functions, d phi_mu / dA_i = -d_i phi_mu (mu on A), so at fixed D, with X = phi D
//! and X^k = d_k phi D,
//!   rho^{Ai}     = -2 sum_{mu in A} d_i phi_mu X_mu
//!   d_k rho^{Ai} = -2 sum_{mu in A} (d_ki phi_mu X_mu + d_i phi_mu X^k_mu)
//!   tau^{Ai}     =  - sum_{mu in A} sum_k d_ki phi_mu X^k_mu.
//!
//! First derivative of the potential matrix (`want_v` of [`XcIntegrator::hessian_terms`]):
//!   V^{Ai} = Vk[w F u^{Ai}] - G^i_A - (G^i_A)^T,
//! Vk\[t\] the potential-matrix form of a first-order potential t (as in the energy-gradient
//! XC potential), G^i_{mu nu} = sum_g w [v_rho d_i phi_mu phi_nu + v_k (d_ki phi_mu phi_nu
//! + d_i phi_mu d_k phi_nu) + 1/2 v_tau d_ki phi_mu d_k phi_nu], and G^i_A its rows on A.
//!
//! Second derivative of the energy (`want_h`), x = (A, i), y = (B, j):
//!   E^{xy} = sum_g w u^x F u^y                                                  (kernel)
//!          + delta_AB sum_{mu in A} sum_g w [2 v_rho d_ij phi_mu X_mu
//!              + 2 v_k (d_kij phi_mu X_mu + d_ij phi_mu X^k_mu) + v_tau d_kij phi_mu X^k_mu]
//!          + sum_{mu in A, nu in B} D_mu,nu C^{ij}_mu,nu                             (cross)
//!   C^{ij} = sum_g w [2 v_rho d_i phi d_j phi + 2 v_k (d_ki phi d_j phi + d_i phi d_kj phi)
//!              + v_tau d_ki phi d_kj phi].
//!
//! Both are accumulated batch by batch: the per-perturbation work is dense products over the
//! functions that reach a batch, and only the atoms owning those functions are perturbed.

use crate::batch::{BatchCtx, BatchPlan};
use crate::points::{
    dot, mul_add, mul_set, nu_of, point_derivs, polarized_inputs, xc_input, ExtPotential, Points,
};
use crate::XcIntegrator;
use faer::{Accum, Mat, MatMut, MatRef, Par};
use linus_base::linalg::{mat_to_colmajor, matmul, mm};
use linus_core::channels::Channels;
use linus_core::timing::Timers;
use linus_ints::points::ao::deriv_indices;
use linus_num::det::{merge_all, FixedAcc};
use linus_xc::{Family, Functional, XcOutput};
use std::sync::Mutex;

/// Indices of derivative components in `eval_ao`'s output order.
pub(crate) struct CompIdx {
    pub(crate) d1: [usize; 3],
    pub(crate) d2: [[usize; 3]; 3],
    pub(crate) d3: [[[usize; 3]; 3]; 3],
}

pub(crate) fn comp_index(order: usize) -> CompIdx {
    let di = deriv_indices(order);
    let find = |e: [usize; 3]| di.iter().position(|v| *v == e).unwrap_or(usize::MAX);
    let mut c = CompIdx {
        d1: [0; 3],
        d2: [[0; 3]; 3],
        d3: [[[0; 3]; 3]; 3],
    };
    for i in 0..3 {
        let mut e1 = [0; 3];
        e1[i] += 1;
        c.d1[i] = find(e1);
        for j in 0..3 {
            let mut e2 = e1;
            e2[j] += 1;
            c.d2[i][j] = find(e2);
            for k in 0..3 {
                let mut e3 = e2;
                e3[k] += 1;
                c.d3[i][j][k] = find(e3);
            }
        }
    }
    c
}

/// Per-thread scratch buffers of the Hessian passes (reused across batches; the basis functions
/// are the batch engine's, [`crate::batch`]).
#[derive(Default)]
pub(crate) struct Work {
    pub(crate) la: Vec<usize>,
    pub(crate) atoms: Vec<usize>,
    pub(crate) dsub: Vec<f64>,
    pub(crate) x: Vec<f64>,
    pub(crate) rho: Vec<f64>,
    pub(crate) grad: Vec<f64>,
    pub(crate) sigma: Vec<f64>,
    pub(crate) tau: Vec<f64>,
    pub(crate) out: XcOutput,
    pub(crate) wv: Vec<f64>,
    pub(crate) wf: Vec<f64>,
    pub(crate) u: Vec<f64>,
    pub(crate) t: Vec<f64>,
    pub(crate) eloc: Vec<f64>,
    pub(crate) g: Vec<f64>,
    pub(crate) tmp: Vec<f64>,
    pub(crate) ybig: Vec<f64>,
    pub(crate) zbig: Vec<f64>,
    pub(crate) mbig: Vec<f64>,
    pub(crate) lmat: Vec<f64>,
    pub(crate) rmat: Vec<f64>,
    pub(crate) cmat: Vec<f64>,
    /// polarized libxc inputs (interleaved per point)
    pub(crate) rin: Vec<f64>,
    pub(crate) sin: Vec<f64>,
    pub(crate) tin: Vec<f64>,
}

pub(crate) fn zeroed(v: &mut Vec<f64>, n: usize) -> &mut [f64] {
    v.clear();
    v.resize(n, 0.0);
    &mut v[..]
}

pub(crate) fn cm(v: &[f64], r: usize, c: usize) -> MatRef<'_, f64> {
    MatRef::from_column_major_slice(&v[..r * c], r, c)
}

pub(crate) fn cm_mut(v: &mut [f64], r: usize, c: usize) -> MatMut<'_, f64> {
    MatMut::from_column_major_slice_mut(&mut v[..r * c], r, c)
}

/// The basis functions of a batch in the batch engine's function-major layout (every XC pass
/// reads it): component k of function m at `(m nc + k) ld`, its np points first (the rest of
/// the ld padding is not read).
#[derive(Clone, Copy)]
pub(crate) struct Fns<'a> {
    ao: &'a [f64],
    nao: usize,
    nc: usize,
    ld: usize,
    np: usize,
}

impl<'a> Fns<'a> {
    /// The batch's functions (`b.ao`, `b.nc` components each, points at stride `b.ld`).
    pub(crate) fn of(b: &BatchCtx<'a>) -> Self {
        Fns {
            ao: b.ao,
            nao: b.nao,
            nc: b.nc,
            ld: b.ld,
            np: b.np,
        }
    }

    #[inline(always)]
    fn at(&self, k: usize, m: usize) -> &'a [f64] {
        let o = (m * self.nc + k) * self.ld;
        &self.ao[o..o + self.np]
    }

    /// Component k of all functions (np x nao, column-major with column stride nc ld).
    #[inline(always)]
    fn mat(&self, k: usize) -> MatRef<'a, f64> {
        MatRef::from_column_major_slice_with_stride(
            &self.ao[k * self.ld..],
            self.np,
            self.nao,
            self.nc * self.ld,
        )
    }
}

/// The products X_{s,c} = (d_c phi) D_s of a batch: channel s, component c, function m at
/// `((s nx + c) nao + m) ld`, np points each (the Hessian's [`channel_x`] with ld = np; VV10's
/// response with the batch's padded ld).
#[derive(Clone, Copy)]
pub(crate) struct Xs<'a> {
    pub(crate) x: &'a [f64],
    pub(crate) nx: usize,
    pub(crate) nao: usize,
    pub(crate) ld: usize,
    pub(crate) np: usize,
}

impl<'a> Xs<'a> {
    #[inline(always)]
    fn at(&self, s: usize, c: usize, m: usize) -> &'a [f64] {
        let o = ((s * self.nx + c) * self.nao + m) * self.ld;
        &self.x[o..o + self.np]
    }
}

/// Second-derivative pairs (j <= k) in the order of the X_{s,c} components 4..10.
const PAIRS: [(usize, usize); 6] = [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)];

/// Position of the pair (j, k) in [`PAIRS`].
#[inline(always)]
fn sym(j: usize, k: usize) -> usize {
    let (a, b) = if j <= k { (j, k) } else { (k, j) };
    [[0, 1, 2], [1, 3, 4], [2, 4, 5]][a][b]
}

/// A batch of a Hessian pass: its basis functions, the derivative-component indices and the
/// functional's family (`ns` density variables per channel, [`nu_of`]).
#[derive(Clone, Copy)]
pub(crate) struct HBatch<'a> {
    pub(crate) f: Fns<'a>,
    pub(crate) ci: &'a CompIdx,
    pub(crate) gga: bool,
    pub(crate) mgga: bool,
    pub(crate) ns: usize,
}

/// Every channel's D_s[idx, idx] (column-major nao x nao blocks, channel-major, into `dsub`)
/// and X_{s,c} = (d_c phi) D_s for the first `nx` components c (the value, the gradient, then
/// the second derivatives of [`PAIRS`]) into `x` (the layout of [`Xs`]).
fn channel_x<const NCH: usize>(
    hb: HBatch<'_>,
    nx: usize,
    nbf: usize,
    idx: &[usize],
    dflat: &[Vec<f64>; NCH],
    dsub: &mut Vec<f64>,
    x: &mut Vec<f64>,
) {
    let (f, ci) = (hb.f, hb.ci);
    let (nao, np) = (f.nao, f.np);
    let ds = zeroed(dsub, NCH * nao * nao);
    for s in 0..NCH {
        for j in 0..nao {
            let row = &dflat[s][idx[j] * nbf..(idx[j] + 1) * nbf];
            for i in 0..nao {
                ds[s * nao * nao + j * nao + i] = row[idx[i]];
            }
        }
    }
    x.resize(NCH * nx * nao * np, 0.0);
    for s in 0..NCH {
        for c in 0..nx {
            let comp = match c {
                0 => 0,
                1..=3 => ci.d1[c - 1],
                _ => {
                    let (j, k) = PAIRS[c - 4];
                    ci.d2[j][k]
                }
            };
            matmul(
                cm_mut(&mut x[(s * nx + c) * nao * np..], np, nao),
                Accum::Replace,
                f.mat(comp),
                cm(&ds[s * nao * nao..], nao, nao),
                1.0,
                Par::Seq,
            );
        }
    }
}

/// Every channel's rho_s, grad rho_s (with `with_grad`) and tau_s (meta-GGA) from X, at
/// rho[s np + g], grad[(s 3 + k) np + g] and tau[s np + g]; with `du` (the grid terms; zeroed,
/// 3 x NCH ns x np) also grad_j u_s in the same pass over X (which then has the second-derivative
/// components).
fn channel_densities<const NCH: usize>(
    hb: HBatch<'_>,
    xs: Xs<'_>,
    with_grad: bool,
    rho: &mut Vec<f64>,
    grad: &mut Vec<f64>,
    tau: &mut Vec<f64>,
    mut du: Option<&mut [f64]>,
) {
    let (f, ci, ns) = (hb.f, hb.ci, hb.ns);
    let np = f.np;
    let nu = NCH * ns;
    let rho = zeroed(rho, NCH * np);
    let grad = zeroed(grad, NCH * 3 * np);
    let tau = zeroed(tau, NCH * np);
    for s in 0..NCH {
        for m in 0..f.nao {
            let x0 = xs.at(s, 0, m);
            mul_add(&mut rho[s * np..(s + 1) * np], 1.0, f.at(0, m), x0);
            if with_grad {
                for k in 0..3 {
                    let gk = &mut grad[(s * 3 + k) * np..(s * 3 + k + 1) * np];
                    mul_add(gk, 2.0, f.at(ci.d1[k], m), x0);
                }
            }
            if let Some(du) = du.as_deref_mut() {
                for j in 0..3 {
                    if hb.gga {
                        for k in 0..3 {
                            let o = (j * nu + s * ns + 1 + k) * np;
                            let d = &mut du[o..o + np];
                            mul_add(d, 2.0, f.at(ci.d2[j][k], m), x0);
                            mul_add(d, 2.0, f.at(ci.d1[j], m), xs.at(s, 1 + k, m));
                        }
                    }
                    if hb.mgga {
                        let o = (j * nu + s * ns + 4) * np;
                        for k in 0..3 {
                            let d = &mut du[o..o + np];
                            mul_add(d, 1.0, f.at(ci.d2[j][k], m), xs.at(s, 1 + k, m));
                        }
                    }
                }
            }
            if hb.mgga {
                for k in 0..3 {
                    let tr = &mut tau[s * np..(s + 1) * np];
                    mul_add(tr, 0.5, f.at(ci.d1[k], m), xs.at(s, k + 1, m));
                }
            }
        }
        if let Some(du) = du.as_deref_mut() {
            for j in 0..3 {
                let o = (j * nu + s * ns) * np;
                du[o..o + np].copy_from_slice(&grad[(s * 3 + j) * np..(s * 3 + j + 1) * np]);
            }
        }
    }
}

/// The functional's (order-2) output at a batch's np points for its channel variables
/// `dens` = (rho, grad, tau) as [`channel_densities`] gives them: one channel's rho,
/// sigma = |grad rho|^2 (into `sigma`) and tau, or two channels' polarized inputs (into `pol`).
fn eval_channels<const NCH: usize>(
    func: &Functional,
    hb: HBatch<'_>,
    dens: [&[f64]; 3],
    sigma: &mut Vec<f64>,
    pol: [&mut Vec<f64>; 3],
    out: &mut XcOutput,
) {
    let (gga, mgga, np) = (hb.gga, hb.mgga, hb.f.np);
    let [rho, grad, tau] = dens;
    if NCH == 1 {
        let sigma = zeroed(sigma, if gga { np } else { 0 });
        if gga {
            for gi in 0..np {
                sigma[gi] = grad[gi].powi(2) + grad[np + gi].powi(2) + grad[2 * np + gi].powi(2);
            }
        }
        func.eval(2, &xc_input(gga, mgga, rho, sigma, tau), out)
            .expect("functional evaluation");
    } else {
        let [rin, sin, tin] = pol;
        polarized_inputs(gga, mgga, np, rho, grad, tau, rin, sin, tin);
        func.eval(2, &xc_input(gga, mgga, rin, sin, tin), out)
            .expect("functional evaluation");
    }
}

/// u^x of the basis functions moving with their atoms (x = 3 a + i, a the batch-local atom
/// `la[m]` of function m; the module docs' formulas per channel), accumulated into `u`
/// (zeroed; npert x NCH ns rows of `xs.ld`, channel-major within a perturbation, np points
/// each). The one u^x of the XC Hessian and VV10's response.
#[allow(clippy::needless_range_loop, reason = "indexes several arrays in step")]
pub(crate) fn basis_motion<const NCH: usize>(
    hb: HBatch<'_>,
    xs: Xs<'_>,
    la: &[usize],
    u: &mut [f64],
) {
    let (f, ci, ns) = (hb.f, hb.ci, hb.ns);
    let (nao, np, ld) = (f.nao, f.np, xs.ld);
    let nu = NCH * ns;
    for s in 0..NCH {
        for m in 0..nao {
            let a = la[m];
            let x0 = xs.at(s, 0, m);
            for i in 0..3 {
                let base = (3 * a + i) * nu * ld + s * ns * ld;
                let pi = f.at(ci.d1[i], m);
                mul_add(&mut u[base..base + np], -2.0, pi, x0);
                if hb.gga {
                    for k in 0..3 {
                        let (hki, xk) = (f.at(ci.d2[k][i], m), xs.at(s, k + 1, m));
                        let ur = &mut u[base + (1 + k) * ld..base + (1 + k) * ld + np];
                        mul_add(ur, -2.0, hki, x0);
                        mul_add(ur, -2.0, pi, xk);
                    }
                }
                if hb.mgga {
                    for k in 0..3 {
                        let (hki, xk) = (f.at(ci.d2[k][i], m), xs.at(s, k + 1, m));
                        mul_add(&mut u[base + 4 * ld..base + 4 * ld + np], -1.0, hki, xk);
                    }
                }
            }
        }
    }
}

/// t_p = wF u_p (n perturbations of nu x np each) into `t` (zeroed).
fn apply_kernel(nu: usize, np: usize, n: usize, wf: &[f64], u: &[f64], t: &mut [f64]) {
    for p in 0..n {
        for c in 0..nu {
            let tr = &mut t[(p * nu + c) * np..(p * nu + c + 1) * np];
            for dd in 0..nu {
                let (fc, ur) = (
                    &wf[(c * nu + dd) * np..(c * nu + dd + 1) * np],
                    &u[(p * nu + dd) * np..(p * nu + dd + 1) * np],
                );
                mul_add(tr, 1.0, fc, ur);
            }
        }
    }
}

/// G^i (3 x nao x nao, column-major), the potential-matrix derivative with the i-derivative on
/// the first function: A_i^T phi + P_i^T B + sum_k H_ki^T (1/2 w v_tau P_k), with
/// A_i = w v_rho P_i + sum_k w v_k H_ki and B = sum_k w v_k P_k (P = grad phi, H = Hessian), for
/// one channel's weighted derivatives `wv` (ns x np).
fn g_matrices(hb: HBatch<'_>, wv: &[f64], gm: &mut [f64], tmp: &mut Vec<f64>) {
    let (f, ci, gga, mgga) = (hb.f, hb.ci, hb.gga, hb.mgga);
    let (nao, np) = (f.nao, f.np);
    let wvc = |c: usize| &wv[c * np..(c + 1) * np];
    let tm = zeroed(tmp, 2 * nao * np);
    let (am, bm) = tm.split_at_mut(nao * np);
    if gga {
        for m in 0..nao {
            let br = &mut bm[m * np..(m + 1) * np];
            for k in 0..3 {
                mul_add(br, 1.0, wvc(1 + k), f.at(ci.d1[k], m));
            }
        }
    }
    for i in 0..3 {
        for m in 0..nao {
            let ar = &mut am[m * np..(m + 1) * np];
            mul_set(ar, 1.0, wv, f.at(ci.d1[i], m));
            if gga {
                for k in 0..3 {
                    mul_add(ar, 1.0, wvc(1 + k), f.at(ci.d2[k][i], m));
                }
            }
        }
        let gi_m = &mut gm[i * nao * nao..(i + 1) * nao * nao];
        matmul(
            cm_mut(gi_m, nao, nao),
            Accum::Replace,
            cm(am, np, nao).transpose(),
            f.mat(0),
            1.0,
            Par::Seq,
        );
        if gga {
            matmul(
                cm_mut(gi_m, nao, nao),
                Accum::Add,
                f.mat(ci.d1[i]).transpose(),
                cm(bm, np, nao),
                1.0,
                Par::Seq,
            );
        }
    }
    if mgga {
        for k in 0..3 {
            for m in 0..nao {
                let br = &mut bm[m * np..(m + 1) * np];
                mul_set(br, 0.5, wvc(4), f.at(ci.d1[k], m));
            }
            for i in 0..3 {
                let gi_m = &mut gm[i * nao * nao..(i + 1) * nao * nao];
                matmul(
                    cm_mut(gi_m, nao, nao),
                    Accum::Add,
                    f.mat(ci.d2[k][i]).transpose(),
                    cm(bm, np, nao),
                    1.0,
                    Par::Seq,
                );
            }
        }
    }
}

/// Potential-matrix halves for per-point first-order potentials t (nslot x ns x np, weights
/// included): M_s = phi^T Y_s (+ sum_k P_k^T 1/4 t_tau P_k), Y_s = 1/2 t_rho phi + sum_k t_k P_k,
/// so that the potential matrix is M_s + M_s^T. `mb` receives nao x (nslot nao), column-major;
/// `ybig`, `zbig` are scratch.
fn potential_blocks(hb: HBatch<'_>, t: &[f64], scratch: [&mut Vec<f64>; 2], mb: &mut Vec<f64>) {
    let (f, ci, gga, mgga, nu) = (hb.f, hb.ci, hb.gga, hb.mgga, hb.ns);
    let [ybig, zbig] = scratch;
    let (nao, np) = (f.nao, f.np);
    let nslot = t.len() / (nu * np);
    let ncol = nslot * nao;
    let yb = zeroed(ybig, np * ncol);
    for sl in 0..nslot {
        let tp = |c: usize| &t[(sl * nu + c) * np..(sl * nu + c + 1) * np];
        for m in 0..nao {
            let y = &mut yb[(sl * nao + m) * np..(sl * nao + m + 1) * np];
            mul_set(y, 0.5, tp(0), f.at(0, m));
            if gga {
                for k in 0..3 {
                    mul_add(y, 1.0, tp(1 + k), f.at(ci.d1[k], m));
                }
            }
        }
    }
    let mbs = zeroed(mb, nao * ncol);
    matmul(
        cm_mut(mbs, nao, ncol),
        Accum::Replace,
        f.mat(0).transpose(),
        cm(yb, np, ncol),
        1.0,
        Par::Seq,
    );
    if mgga {
        for k in 0..3 {
            let zb = zeroed(zbig, np * ncol);
            for sl in 0..nslot {
                let tt = &t[(sl * nu + 4) * np..(sl * nu + 5) * np];
                for m in 0..nao {
                    mul_set(
                        &mut zb[(sl * nao + m) * np..(sl * nao + m + 1) * np],
                        0.25,
                        tt,
                        f.at(ci.d1[k], m),
                    );
                }
            }
            matmul(
                cm_mut(mbs, nao, ncol),
                Accum::Add,
                f.mat(ci.d1[k]).transpose(),
                cm(zb, np, ncol),
                1.0,
                Par::Seq,
            );
        }
    }
}

/// Target size (doubles) of the per-batch block of perturbed potentials (np x npert x nao):
/// small enough to stay in L2 between building it and multiplying it.
pub(crate) const YBIG_MAX: usize = 1 << 16;

/// True if any |v| > t.
#[cfg(test)]
pub(crate) fn any_above(v: &[f64], t: f64) -> bool {
    v.chunks(16).any(|c| {
        c.iter()
            .fold(0.0f64, |m, x| if x.abs() > m { x.abs() } else { m })
            > t
    })
}

/// v[idx[r], idx[c]] += m[r, c] + m[c, r] for a column-major nao x nao block m.
#[cfg(test)]
pub(crate) fn scatter_sym(v: &mut Mat<f64>, idx: &[usize], m: &[f64]) {
    let nao = idx.len();
    for c in 0..nao {
        let col = v.col_as_slice_mut(idx[c]);
        let mc = &m[c * nao..(c + 1) * nao];
        for r in 0..nao {
            col[idx[r]] += mc[r] + m[r * nao + c];
        }
    }
}

/// Unique values in order of first appearance and each element's position in that list.
pub(crate) fn local_index(v: &[usize], uniq: &mut Vec<usize>, pos: &mut Vec<usize>) {
    uniq.clear();
    pos.clear();
    for &a in v {
        let l = match uniq.iter().position(|&x| x == a) {
            Some(l) => l,
            None => {
                uniq.push(a);
                uniq.len() - 1
            }
        };
        pos.push(l);
    }
}

/// The family flags (GGA, meta-GGA) of an evaluation: the functional's, or a GGA for an
/// external potential.
fn family(func: &Functional, ext: Option<&ExtPotential<'_>>) -> (bool, bool) {
    match ext {
        Some(_) => (true, false),
        None => {
            let fam = func.family();
            (fam >= Family::Gga, fam == Family::Mgga && func.needs_tau())
        }
    }
}

/// The first derivatives V_s^x of the channels' XC potentials as a pass accumulates them: per
/// atom (`[atom][s 3 + i]`), packed lower triangles in fixed point.
pub(crate) struct PotentialDerivs {
    nbf: usize,
    nch: usize,
    acc: Vec<Mutex<Vec<FixedAcc>>>,
}

impl PotentialDerivs {
    /// The full matrices, `[s][3 atom + i]` (checks).
    #[cfg(test)]
    pub(crate) fn unpack(self) -> Vec<Vec<Mat<f64>>> {
        let mut v: Vec<Vec<Mat<f64>>> = (0..self.nch).map(|_| Vec::new()).collect();
        for m in self.acc {
            let a = m.into_inner().unwrap();
            for (s, vs) in v.iter_mut().enumerate() {
                for i in 0..3 {
                    vs.push(crate::unpack_lower(&a[s * 3 + i].to_f64(), self.nbf));
                }
            }
        }
        v
    }

    /// V_s^x C_s for the occupied orbitals `co[s]` of every channel, `[s][3 atom + i]`
    /// (nbf x nocc_s; one accumulated channel is projected on the orbitals of every channel, as
    /// VV10's potential, which is that of the total density). The matrices are unpacked one atom
    /// at a time, so that all of them never exist at once.
    pub(crate) fn project(self, co: &[MatRef<'_, f64>], par: Par) -> Vec<Vec<Mat<f64>>> {
        assert!(self.nch == 1 || self.nch == co.len());
        let natom = self.acc.len();
        let mut out: Vec<Vec<Mat<f64>>> =
            co.iter().map(|_| Vec::with_capacity(3 * natom)).collect();
        for m in self.acc {
            let a = m.into_inner().unwrap();
            for i in 0..3 {
                if self.nch == 1 {
                    let v = crate::unpack_lower(&a[i].to_f64(), self.nbf);
                    for (c, o) in co.iter().zip(out.iter_mut()) {
                        o.push(mm(v.as_ref(), *c, par));
                    }
                } else {
                    for (s, (c, o)) in co.iter().zip(out.iter_mut()).enumerate() {
                        let v = crate::unpack_lower(&a[s * 3 + i].to_f64(), self.nbf);
                        o.push(mm(v.as_ref(), *c, par));
                    }
                }
            }
        }
        out
    }
}

/// The explicit XC terms of a Hessian in the shape every backend gives them (this CPU code, VV10's
/// [`crate::vv10_resp`] and the device's, `linus::backend::gpu::xc`): the channels' potential
/// derivatives projected on their occupied orbitals (all the CPHF right-hand sides need) and the
/// second-derivative terms.
pub struct XcHessianTerms {
    /// V_s^x C_s on a fixed grid, `[s][x]` (nbf x nocc_s).
    pub vfix: Vec<Vec<Mat<f64>>>,
    /// dV_s^y C_s of the grid moving with the atoms, `[s][y]` (empty without the grid response).
    pub vgrid: Vec<Vec<Mat<f64>>>,
    /// The explicit second derivative ((3N)^2 row-major; a backend may include its grid terms).
    pub hess: Vec<f64>,
    /// The grid response's derivative of the fixed-grid gradient when kept apart from `hess`
    /// (row = gradient component, column = displaced coordinate; empty otherwise).
    pub hess_grid: Vec<f64>,
}

impl XcHessianTerms {
    /// Adds the terms to a Hessian's explicit second derivative `h` and its fixed-grid F^x C_s
    /// `fxo`, and the grid response to `gxo` (taken over when empty), in that order (the order
    /// `linus::derivatives::hessian` adds them in).
    pub(crate) fn add_to(
        self,
        h: &mut [f64],
        fxo: &mut [Vec<Mat<f64>>],
        gxo: &mut Vec<Vec<Mat<f64>>>,
    ) {
        for (fs, vs) in fxo.iter_mut().zip(&self.vfix) {
            for (f, v) in fs.iter_mut().zip(vs) {
                *f += v;
            }
        }
        h.iter_mut().zip(&self.hess).for_each(|(a, b)| *a += b);
        h.iter_mut().zip(&self.hess_grid).for_each(|(a, b)| *a += b);
        if gxo.is_empty() {
            *gxo = self.vgrid;
        } else {
            for (gs, vs) in gxo.iter_mut().zip(&self.vgrid) {
                for (g, v) in gs.iter_mut().zip(vs) {
                    *g += v;
                }
            }
        }
    }
}

impl XcIntegrator {
    /// The explicit XC terms of a Hessian over spin channels ([`XcHessianTerms`]): for the
    /// channel densities `ds` (restricted: a closed shell's alpha density, the total density
    /// being 2 D; unrestricted: the spins) and their occupied orbitals `co`, on a fixed grid
    /// the first derivatives V_s^x C_s of the channels' XC potentials and the explicit second
    /// derivative of E_xc (one pass over the grid, timed as `hess_xc`), and with `grid_response`
    /// the first-order grid terms (a second pass, `hess_xc_grid`):
    ///  * dV_s^y C_s, with dV_s^y = sum_g w^y_g P_s(g) + sum_{g on B} w_g d_j P_s(g), P_s(g) the
    ///    potential-matrix integrand and y = (B, j), the grid moving with the atoms (each point
    ///    with its owner; weights from `point_weight_gradient`);
    ///  * the derivative of the fixed-grid XC gradient, E1\[x\]\[y\] = sum_g w^y_g (v.u^x)(g)
    ///    + sum_{g on B} w_g d_j (v.u^x)(g) (not symmetric).
    ///
    /// No second weight derivatives: with the fixed-grid terms, this is the derivative of a
    /// gradient computed without grid response ("scheme 1" of Bykov et al., Mol. Phys. 113,
    /// 1961 (2015), as in ORCA's analytic DFT Hessian). The potential derivatives are
    /// accumulated per atom and projected atom by atom (`par`: the products' parallelism).
    /// LDA, GGA and tau-dependent meta-GGA functionals.
    pub fn hessian_terms(
        &self,
        ds: Channels<&Mat<f64>>,
        co: Channels<MatRef<'_, f64>>,
        grid_response: bool,
        par: Par,
        timers: &mut Timers,
    ) -> XcHessianTerms {
        assert_eq!(ds.count(), co.count(), "a density and orbitals per channel");
        let co = co.as_slice();
        match ds {
            Channels::Restricted(d) => {
                self.terms_ch::<1>([&(d * 2.0)], co, grid_response, par, timers)
            }
            Channels::Unrestricted([a, b]) => {
                self.terms_ch::<2>([a, b], co, grid_response, par, timers)
            }
        }
    }

    /// [`Self::hessian_terms`] for `NCH` channels at the channel densities `dens` (one channel:
    /// the total density).
    fn terms_ch<const NCH: usize>(
        &self,
        dens: [&Mat<f64>; NCH],
        co: &[MatRef<'_, f64>],
        grid_response: bool,
        par: Par,
        timers: &mut Timers,
    ) -> XcHessianTerms {
        let (vfix, hess) = timers.time("hess_xc", || {
            let (v, h) = self.hessian_terms_ch::<NCH>(dens, true, true, None);
            (v.project(co, par), h)
        });
        let (vgrid, hess_grid) = if grid_response {
            timers.time("hess_xc_grid", || {
                let (v, h) = self.grid_terms_ch::<NCH>(dens, true, true, None);
                (v.project(co, par), h)
            })
        } else {
            (Vec::new(), Vec::new())
        };
        XcHessianTerms {
            vfix,
            vgrid,
            hess,
            hess_grid,
        }
    }

    /// The fixed-grid terms for `NCH` channels at the channel densities `dens` (one channel:
    /// the total density): the potential derivatives (`want_v`) and the explicit second
    /// derivative ((3N)^2 row-major; `want_h`), with the functional's derivatives from libxc
    /// (`ext` None) or, for a GGA-like external potential of one channel (`ext`: VV10's v_rho
    /// and v_sigma per grid point), only its potential terms: the kernel terms are left out
    /// (their nonlocal counterparts are [`crate::vv10_resp`]'s). Unrequested parts come
    /// back empty.
    #[allow(clippy::needless_range_loop, reason = "indexes several arrays in step")]
    pub(crate) fn hessian_terms_ch<const NCH: usize>(
        &self,
        dens: [&Mat<f64>; NCH],
        want_v: bool,
        want_h: bool,
        ext: Option<&ExtPotential<'_>>,
    ) -> (PotentialDerivs, Vec<f64>) {
        assert!(
            ext.is_none() || NCH == 1,
            "an external potential has one channel"
        );
        let basis = &self.basis;
        let nbf = basis.nbf;
        let natom = basis.atom_shells.len();
        let n3 = 3 * natom;
        let func = self.channel_functional::<NCH>();
        let (gga, mgga) = family(func, ext);
        let ns = nu_of(gga, mgga);
        let nu = NCH * ns;
        let order = usize::from(gga) + if want_h { 2 } else { 1 };
        let ci = comp_index(order);
        let dflat: [Vec<f64>; NCH] = dens.map(|d| mat_to_colmajor(d.as_ref()));
        let nx = if gga { 4 } else { 1 };
        // fixed-point accumulators (order independent): V_s^x per atom (packed lower triangles
        // s 3 + i, shared, locked per atom) and E^xy per thread
        let npack = nbf * (nbf + 1) / 2;
        let vacc: Vec<Mutex<Vec<FixedAcc>>> = if want_v {
            (0..natom)
                .map(|_| Mutex::new((0..3 * NCH).map(|_| FixedAcc::new(npack)).collect()))
                .collect()
        } else {
            vec![]
        };
        let plan = BatchPlan::new(order, self.func_thresh);
        let init = || {
            (
                FixedAcc::new(if want_h { n3 * n3 } else { 0 }),
                Work::default(),
            )
        };
        let (_, works) = self.for_each_batch(&plan, init, |(hacc, wk), b| {
            let (np, nao, w, idx, runs) = (b.np, b.nao, b.w, b.idx, b.runs);
            if nao == 0 {
                return;
            }
            let Work {
                la,
                atoms,
                dsub,
                x,
                rho,
                grad,
                sigma,
                tau,
                out,
                wv,
                wf,
                u,
                t,
                eloc,
                g: gbuf,
                tmp,
                ybig,
                zbig,
                mbig,
                lmat,
                rmat,
                cmat,
                rin,
                sin,
                tin,
                ..
            } = &mut *wk;
            // the batch-local atoms of the functions
            local_index(b.owner, atoms, la);
            let (la, atoms): (&[usize], &[usize]) = (la, atoms);
            let nat = atoms.len();
            let npert = 3 * nat;
            let f = Fns::of(b);
            let hb = HBatch {
                f,
                ci: &ci,
                gga,
                mgga,
                ns,
            };
            // X_s,c = (d_c phi) D_s for c = value (and gradient components), the densities and
            // the functional's derivatives
            channel_x(hb, nx, nbf, idx, &dflat, dsub, x);
            let dsub: &[f64] = dsub;
            let xs = Xs {
                x,
                nx,
                nao,
                ld: np,
                np,
            };
            channel_densities::<NCH>(hb, xs, gga, rho, grad, tau, None);
            let (rho, grad, tau): (&[f64], &[f64], &[f64]) = (rho, grad, tau);
            match ext {
                None => {
                    eval_channels::<NCH>(func, hb, [rho, grad, tau], sigma, [rin, sin, tin], out);
                    point_derivs::<NCH>(
                        out,
                        &Points {
                            gga,
                            mgga,
                            w,
                            grad,
                            np,
                            ld: np,
                        },
                        zeroed(wv, nu * np),
                        zeroed(wf, nu * nu * np),
                    );
                }
                Some(e) => {
                    e.weighted(b.start, w, grad, zeroed(wv, nu * np));
                    zeroed(wf, nu * nu * np);
                }
            }
            let wv: &[f64] = wv;
            let wf: &[f64] = wf;
            let wvs = |s: usize, c: usize| &wv[(s * ns + c) * np..(s * ns + c + 1) * np];
            // u^{(a,i)} (npert x nu x np) and t = w F u
            let u = zeroed(u, npert * nu * np);
            basis_motion::<NCH>(hb, xs, la, u);
            let u: &[f64] = u;
            let t = zeroed(t, npert * nu * np);
            apply_kernel(nu, np, npert, wf, u, t);
            let t: &[f64] = t;

            if want_h {
                let el = zeroed(eloc, npert * npert);
                // kernel term: E_pq = sum_{c,g} u_p t_q (all channels)
                matmul(
                    cm_mut(el, npert, npert),
                    Accum::Replace,
                    cm(u, nu * np, npert).transpose(),
                    cm(t, nu * np, npert),
                    1.0,
                    Par::Seq,
                );
                for s in 0..NCH {
                    let wv0 = wvs(s, 0);
                    // same-function second derivatives
                    let sc = zeroed(tmp, 4 * np);
                    for m in 0..nao {
                        let a = la[m];
                        let x0 = xs.at(s, 0, m);
                        let (s0, sk) = sc.split_at_mut(np);
                        mul_set(s0, 2.0, wv0, x0);
                        if gga {
                            for k in 0..3 {
                                let (xk, vk) = (xs.at(s, k + 1, m), wvs(s, 1 + k));
                                let skk = &mut sk[k * np..(k + 1) * np];
                                for gi in 0..np {
                                    s0[gi] += 2.0 * vk[gi] * xk[gi];
                                    skk[gi] = 2.0 * vk[gi] * x0[gi];
                                }
                                if mgga {
                                    mul_add(skk, 1.0, wvs(s, 4), xk);
                                }
                            }
                        }
                        for i in 0..3 {
                            for j in i..3 {
                                let mut v: f64 = dot(f.at(ci.d2[i][j], m), s0);
                                if gga {
                                    for k in 0..3 {
                                        v +=
                                            dot(f.at(ci.d3[k][i][j], m), &sk[k * np..(k + 1) * np]);
                                    }
                                }
                                el[(3 * a + j) * npert + 3 * a + i] += v;
                                if i != j {
                                    el[(3 * a + i) * npert + 3 * a + j] += v;
                                }
                            }
                        }
                    }
                    // cross term: C = L^T R (3 nao x 3 nao), L/R column block i stacks
                    // LDA [P_i] / [2 w v_rho P_i]; GGA [P_i; Q_i] / [Q_i; P_i];
                    // meta-GGA additionally [H_ki] / [w v_tau H_ki] (k = x, y, z)
                    let nsl = if mgga {
                        5
                    } else if gga {
                        2
                    } else {
                        1
                    };
                    let lr = nsl * np;
                    let lm = zeroed(lmat, lr * 3 * nao);
                    let rm = zeroed(rmat, lr * 3 * nao);
                    for i in 0..3 {
                        for m in 0..nao {
                            let col = (i * nao + m) * lr;
                            let pi = f.at(ci.d1[i], m);
                            lm[col..col + np].copy_from_slice(pi);
                            if !gga {
                                for gi in 0..np {
                                    rm[col + gi] = 2.0 * wv0[gi] * pi[gi];
                                }
                                continue;
                            }
                            // Q_i = w v_rho P_i + 2 sum_k w v_k H_ki
                            {
                                let q = &mut rm[col..col + np];
                                mul_set(q, 1.0, wv0, pi);
                                for k in 0..3 {
                                    mul_add(q, 2.0, wvs(s, 1 + k), f.at(ci.d2[k][i], m));
                                }
                            }
                            let (rq, lq) = (col..col + np, col + np..col + 2 * np);
                            lm[lq].copy_from_slice(&rm[rq]);
                            rm[col + np..col + 2 * np].copy_from_slice(pi);
                            if mgga {
                                let vt = wvs(s, 4);
                                for k in 0..3 {
                                    let hki = f.at(ci.d2[k][i], m);
                                    let o = col + (2 + k) * np;
                                    lm[o..o + np].copy_from_slice(hki);
                                    for gi in 0..np {
                                        rm[o + gi] = vt[gi] * hki[gi];
                                    }
                                }
                            }
                        }
                    }
                    let n3l = 3 * nao;
                    let cmx = zeroed(cmat, n3l * n3l);
                    matmul(
                        cm_mut(cmx, n3l, n3l),
                        Accum::Replace,
                        cm(lm, lr, n3l).transpose(),
                        cm(rm, lr, n3l),
                        1.0,
                        Par::Seq,
                    );
                    let dss = &dsub[s * nao * nao..(s + 1) * nao * nao];
                    for j in 0..3 {
                        for nv in 0..nao {
                            let q = 3 * la[nv] + j;
                            // D_{mu nu} for all mu (column nv)
                            let drow = &dss[nv * nao..(nv + 1) * nao];
                            for i in 0..3 {
                                let ccol = &cmx[(j * nao + nv) * n3l + i * nao
                                    ..(j * nao + nv) * n3l + (i + 1) * nao];
                                for m in 0..nao {
                                    el[q * npert + 3 * la[m] + i] += drow[m] * ccol[m];
                                }
                            }
                        }
                    }
                }
                // el is column-major (row p = 3 a + i, column q); scatter to the global (row-major)
                for q in 0..npert {
                    let gq = 3 * atoms[q / 3] + q % 3;
                    for p in 0..npert {
                        let gp = 3 * atoms[p / 3] + p % 3;
                        hacc.add1(gp * n3 + gq, el[q * npert + p]);
                    }
                }
                hacc.tick(1);
            }

            if want_v {
                // G_s^i of every channel (NCH x 3 x nao x nao)
                let gm = zeroed(gbuf, NCH * 3 * nao * nao);
                for s in 0..NCH {
                    g_matrices(
                        hb,
                        &wv[s * ns * np..(s + 1) * ns * np],
                        &mut gm[s * 3 * nao * nao..(s + 1) * 3 * nao * nao],
                        tmp,
                    );
                }
                let gm: &[f64] = gm;
                // kernel part per (perturbation, channel) slot, in chunks of atoms:
                // M_p = phi^T Y_p (+ sum_k P_k^T Z_pk), Y_p = 1/2 t_rho phi + sum_k t_k P_k,
                // Z_pk = 1/4 t_tau P_k
                let per_atom = 3 * NCH * np * nao * if mgga { 2 } else { 1 };
                let chunk = (YBIG_MAX / per_atom.max(1)).clamp(1, nat);
                let mut a0 = 0;
                while a0 < nat {
                    let a1 = (a0 + chunk).min(nat);
                    // slots (p, s) with ns components each: t[3 a0 .. 3 a1] is exactly that layout
                    potential_blocks(
                        hb,
                        &t[3 * a0 * nu * np..3 * a1 * nu * np],
                        [&mut *ybig, &mut *zbig],
                        mbig,
                    );
                    let mb = &mut mbig[..];
                    // N_p = M_p - rows of G^i on atom a; V_p = N_p + N_p^T
                    for al in a0..a1 {
                        let rows: Vec<usize> = (0..nao).filter(|&r| la[r] == al).collect();
                        for i in 0..3 {
                            for s in 0..NCH {
                                let sl = (3 * (al - a0) + i) * NCH + s;
                                let nm = &mut mb[sl * nao * nao..(sl + 1) * nao * nao];
                                let gi_m =
                                    &gm[(s * 3 + i) * nao * nao..(s * 3 + i + 1) * nao * nao];
                                for c in 0..nao {
                                    for &r in &rows {
                                        nm[c * nao + r] -= gi_m[c * nao + r];
                                    }
                                }
                            }
                        }
                        let mut vg = vacc[atoms[al]].lock().unwrap();
                        for i in 0..3 {
                            for s in 0..NCH {
                                let sl = (3 * (al - a0) + i) * NCH + s;
                                crate::scatter_packed(
                                    &mut vg[s * 3 + i],
                                    nbf,
                                    idx,
                                    runs,
                                    &mut mb[sl * nao * nao..(sl + 1) * nao * nao],
                                );
                                vg[s * 3 + i].tick(1);
                            }
                        }
                    }
                    a0 = a1;
                }
            }
        });
        let mut h = vec![0.0; if want_h { n3 * n3 } else { 0 }];
        if want_h {
            h = merge_all(works.into_iter().map(|w| vec![w.0]).collect())
                .map(|v| v[0].to_f64())
                .unwrap_or(h);
            for p in 0..n3 {
                for q in 0..p {
                    let s = 0.5 * (h[p * n3 + q] + h[q * n3 + p]);
                    h[p * n3 + q] = s;
                    h[q * n3 + p] = s;
                }
            }
        }
        let v = PotentialDerivs {
            nbf,
            nch: NCH,
            acc: vacc,
        };
        (v, h)
    }

    /// The first-order grid terms for `NCH` channels at the channel densities `dens` (one
    /// channel: the total density; [`Self::hessian_terms`]): dV_s^y (`want_v`) and E1 (`want_h`),
    /// with the functional's derivatives from libxc
    /// (`ext` None) or, for a GGA-like external potential of one channel (`ext`), with its
    /// potential held fixed at the moving points (weight derivatives and the motion of the
    /// basis-function values; the change of the potential itself is
    /// [`crate::vv10_resp`]'s).
    #[allow(clippy::needless_range_loop, reason = "indexes several arrays in step")]
    pub(crate) fn grid_terms_ch<const NCH: usize>(
        &self,
        dens: [&Mat<f64>; NCH],
        want_v: bool,
        want_h: bool,
        ext: Option<&ExtPotential<'_>>,
    ) -> (PotentialDerivs, Vec<f64>) {
        assert!(
            ext.is_none() || NCH == 1,
            "an external potential has one channel"
        );
        let basis = &self.basis;
        let nbf = basis.nbf;
        let grid = &self.grid;
        let natom = basis.atom_shells.len();
        let n3 = 3 * natom;
        let func = self.channel_functional::<NCH>();
        let (gga, mgga) = family(func, ext);
        let ns = nu_of(gga, mgga);
        let nu = NCH * ns;
        let order = usize::from(gga) + if want_h { 2 } else { 1 };
        let ci = comp_index(order);
        let dflat: [Vec<f64>; NCH] = dens.map(|d| mat_to_colmajor(d.as_ref()));
        // per channel: X = phi D_s, X_k = d_k phi D_s, and (GGA second derivatives) X_jk = d_jk phi D_s
        let nx = if gga && want_h { 10 } else { 4 };
        let npack = nbf * (nbf + 1) / 2;
        let vacc: Vec<Mutex<Vec<FixedAcc>>> = if want_v {
            (0..natom)
                .map(|_| Mutex::new((0..3 * NCH).map(|_| FixedAcc::new(npack)).collect()))
                .collect()
        } else {
            vec![]
        };
        // per thread: fixed-point E1 accumulator, a per-batch f64 block, scratch
        let init = || {
            (
                FixedAcc::new(if want_h { n3 * n3 } else { 0 }),
                vec![0.0; if want_h { n3 * n3 } else { 0 }],
                Work::default(),
                grid.partitioner().scratch(),
            )
        };
        let ones = vec![
            1.0;
            grid.batches
                .iter()
                .map(|b| b.end - b.start)
                .max()
                .unwrap_or(0)
        ];
        let plan = BatchPlan::new(order, self.func_thresh);
        let (_, works) = self.for_each_batch(&plan, init, |(hfix, hacc, wk, scr), b| {
            let (np, nao, w, idx, runs) = (b.np, b.nao, b.w, b.idx, b.runs);
            if nao == 0 {
                return;
            }
            // point owners and weight derivatives
            let own: Vec<usize> = grid.atom[b.start..b.start + np]
                .iter()
                .map(|&a| a as usize)
                .collect();
            let (mut owners, mut lo) = (Vec::new(), Vec::new());
            local_index(&own, &mut owners, &mut lo);
            let no = owners.len();
            let mut wout = Vec::new();
            let mut wg: Vec<(usize, usize, [f64; 3])> = Vec::new();
            for gi in 0..np {
                grid.point_weight_gradient(b.start + gi, scr, &mut wout);
                wg.extend(wout.iter().map(|&(c, dw)| (gi, c, dw)));
            }
            let wcs: Vec<usize> = {
                let mut v: Vec<usize> = wg.iter().map(|e| e.1).collect();
                v.sort_unstable();
                v.dedup();
                v
            };
            let Work {
                la,
                atoms,
                dsub,
                x,
                rho,
                grad,
                sigma,
                tau,
                out,
                wv,
                wf,
                u,
                t,
                eloc,
                g: gbuf,
                tmp,
                ybig,
                zbig,
                mbig,
                lmat,
                rmat,
                rin,
                sin,
                tin,
                ..
            } = &mut *wk;
            local_index(b.owner, atoms, la);
            let (la, atoms): (&[usize], &[usize]) = (la, atoms);
            let nat = atoms.len();
            let npert = 3 * nat;
            let f = Fns::of(b);
            let hb = HBatch {
                f,
                ci: &ci,
                gga,
                mgga,
                ns,
            };
            channel_x(hb, nx, nbf, idx, &dflat, dsub, x);
            let xs = Xs {
                x,
                nx,
                nao,
                ld: np,
                np,
            };
            // densities, their gradients (always) and tau; grad_j u (3 x nu x np)
            let du = zeroed(lmat, 3 * nu * np);
            channel_densities::<NCH>(hb, xs, true, rho, grad, tau, Some(&mut *du));
            let (rho, grad, tau): (&[f64], &[f64], &[f64]) = (rho, grad, tau);
            let du: &[f64] = du;
            // unweighted v (wf as scratch), then weighted v and F
            let vu = zeroed(rmat, nu * np);
            match ext {
                None => {
                    eval_channels::<NCH>(func, hb, [rho, grad, tau], sigma, [rin, sin, tin], out);
                    point_derivs::<NCH>(
                        out,
                        &Points {
                            gga,
                            mgga,
                            w: &ones[..np],
                            grad,
                            np,
                            ld: np,
                        },
                        vu,
                        zeroed(wf, nu * nu * np),
                    );
                    point_derivs::<NCH>(
                        out,
                        &Points {
                            gga,
                            mgga,
                            w,
                            grad,
                            np,
                            ld: np,
                        },
                        zeroed(wv, nu * np),
                        zeroed(wf, nu * nu * np),
                    );
                }
                Some(e) => {
                    e.weighted(b.start, &ones[..np], grad, vu);
                    e.weighted(b.start, w, grad, zeroed(wv, nu * np));
                    zeroed(wf, nu * nu * np);
                }
            }
            let vu: &[f64] = vu;
            let wv: &[f64] = wv;
            let wf: &[f64] = wf;
            let wvs = |s: usize, c: usize| &wv[(s * ns + c) * np..(s * ns + c + 1) * np];
            // t_j = w F grad_j u (3 x nu x np): the change of w v at a point moved along j
            let tj = zeroed(t, 3 * nu * np);
            apply_kernel(nu, np, 3, wf, du, tj);
            let tj: &[f64] = tj;

            if want_v {
                let gm = zeroed(gbuf, NCH * 3 * nao * nao);
                let mut wvm = vec![0.0; nu * np];
                let mut tslot = vec![0.0; 3 * nu * np];
                // point motion: per owner B, slots (B, j, s): t_j on B's points, plus G_s^j there
                for ol in 0..no {
                    for c in 0..nu {
                        for gi in 0..np {
                            wvm[c * np + gi] = if lo[gi] == ol { wv[c * np + gi] } else { 0.0 };
                        }
                    }
                    for s in 0..NCH {
                        g_matrices(
                            hb,
                            &wvm[s * ns * np..(s + 1) * ns * np],
                            &mut gm[s * 3 * nao * nao..(s + 1) * 3 * nao * nao],
                            tmp,
                        );
                    }
                    for (o, v) in tslot.iter_mut().enumerate() {
                        let gi = o % np;
                        *v = if lo[gi] == ol { tj[o] } else { 0.0 };
                    }
                    potential_blocks(hb, &tslot, [&mut *ybig, &mut *zbig], mbig);
                    let mb = &mut mbig[..];
                    let mut vg = vacc[owners[ol]].lock().unwrap();
                    for j in 0..3 {
                        for s in 0..NCH {
                            let sl = j * NCH + s;
                            let blk = &mut mb[sl * nao * nao..(sl + 1) * nao * nao];
                            for (a, bb) in blk
                                .iter_mut()
                                .zip(&gm[(s * 3 + j) * nao * nao..(s * 3 + j + 1) * nao * nao])
                            {
                                *a += bb;
                            }
                            crate::scatter_packed(&mut vg[s * 3 + j], nbf, idx, runs, blk);
                            vg[s * 3 + j].tick(1);
                        }
                    }
                }
                // weight derivatives: slots (C, j, s) with t = dw_g/dC_j v(g)
                for &c_at in &wcs {
                    tslot.iter_mut().for_each(|v| *v = 0.0);
                    for &(gi, c, dw) in &wg {
                        if c != c_at {
                            continue;
                        }
                        for j in 0..3 {
                            for c2 in 0..nu {
                                tslot[(j * nu + c2) * np + gi] += dw[j] * vu[c2 * np + gi];
                            }
                        }
                    }
                    potential_blocks(hb, &tslot, [&mut *ybig, &mut *zbig], mbig);
                    let mut vg = vacc[c_at].lock().unwrap();
                    for j in 0..3 {
                        for s in 0..NCH {
                            let sl = j * NCH + s;
                            crate::scatter_packed(
                                &mut vg[s * 3 + j],
                                nbf,
                                idx,
                                runs,
                                &mut mbig[sl * nao * nao..(sl + 1) * nao * nao],
                            );
                            vg[s * 3 + j].tick(1);
                        }
                    }
                }
            }

            if want_h {
                // u^x (npert x nu x np), as in hessian_terms_ch
                let u = zeroed(u, npert * nu * np);
                basis_motion::<NCH>(hb, xs, la, u);
                let u: &[f64] = u;
                // r[x][j][g] = w_g d_j (v.u^x)(g) = (t_j . u^x)(g) + (w v . d_j u^x)(g)
                let r = zeroed(eloc, npert * 3 * np);
                for p in 0..npert {
                    for j in 0..3 {
                        let o = (p * 3 + j) * np;
                        for c in 0..nu {
                            mul_add(
                                &mut r[o..o + np],
                                1.0,
                                &tj[(j * nu + c) * np..(j * nu + c + 1) * np],
                                &u[(p * nu + c) * np..(p * nu + c + 1) * np],
                            );
                        }
                    }
                }
                // d_j u^{Ai} contracted with w v, per channel:
                // -2 [H_ij alpha + P_i beta_j + sum_k T_ijk gamma_k + sum_k H_ik delta_jk],
                // alpha = wv_r X + wv_k X_k, beta_j = wv_r X_j + wv_k X_jk, gamma_k = wv_k X + 1/2 wv_t X_k,
                // delta_jk = wv_k X_j + 1/2 wv_t X_jk
                let sc = zeroed(tmp, 16 * np);
                for s in 0..NCH {
                    for m in 0..nao {
                        let a = la[m];
                        let (alpha, rest) = sc.split_at_mut(np);
                        let (beta, rest) = rest.split_at_mut(3 * np);
                        let (gamma, delta) = rest.split_at_mut(3 * np);
                        mul_set(alpha, 1.0, wvs(s, 0), xs.at(s, 0, m));
                        for j in 0..3 {
                            mul_set(
                                &mut beta[j * np..(j + 1) * np],
                                1.0,
                                wvs(s, 0),
                                xs.at(s, 1 + j, m),
                            );
                        }
                        if gga {
                            for k in 0..3 {
                                mul_add(alpha, 1.0, wvs(s, 1 + k), xs.at(s, 1 + k, m));
                                mul_set(
                                    &mut gamma[k * np..(k + 1) * np],
                                    1.0,
                                    wvs(s, 1 + k),
                                    xs.at(s, 0, m),
                                );
                                if mgga {
                                    mul_add(
                                        &mut gamma[k * np..(k + 1) * np],
                                        0.5,
                                        wvs(s, 4),
                                        xs.at(s, 1 + k, m),
                                    );
                                }
                                for j in 0..3 {
                                    mul_add(
                                        &mut beta[j * np..(j + 1) * np],
                                        1.0,
                                        wvs(s, 1 + k),
                                        xs.at(s, 4 + sym(j, k), m),
                                    );
                                    let dl = &mut delta[(j * 3 + k) * np..(j * 3 + k + 1) * np];
                                    mul_set(dl, 1.0, wvs(s, 1 + k), xs.at(s, 1 + j, m));
                                    if mgga {
                                        mul_add(dl, 0.5, wvs(s, 4), xs.at(s, 4 + sym(j, k), m));
                                    }
                                }
                            }
                        }
                        for i in 0..3 {
                            for j in 0..3 {
                                let o = ((3 * a + i) * 3 + j) * np;
                                let rr = &mut r[o..o + np];
                                mul_add(rr, -2.0, f.at(ci.d2[i][j], m), alpha);
                                mul_add(rr, -2.0, f.at(ci.d1[i], m), &beta[j * np..(j + 1) * np]);
                                if gga {
                                    for k in 0..3 {
                                        mul_add(
                                            rr,
                                            -2.0,
                                            f.at(ci.d3[i][j][k], m),
                                            &gamma[k * np..(k + 1) * np],
                                        );
                                        mul_add(
                                            rr,
                                            -2.0,
                                            f.at(ci.d2[i][k], m),
                                            &delta[(j * 3 + k) * np..(j * 3 + k + 1) * np],
                                        );
                                    }
                                }
                            }
                        }
                    }
                }
                // point motion into the owners' columns; weight derivatives into every atom's
                // (summed in the batch's own order into hacc rows, then added in fixed point)
                for p in 0..npert {
                    let gp = 3 * atoms[p / 3] + p % 3;
                    hacc[gp * n3..(gp + 1) * n3].fill(0.0);
                }
                for p in 0..npert {
                    let gp = 3 * atoms[p / 3] + p % 3;
                    for j in 0..3 {
                        let rr = &r[(p * 3 + j) * np..(p * 3 + j + 1) * np];
                        for gi in 0..np {
                            hacc[gp * n3 + 3 * own[gi] + j] += rr[gi];
                        }
                    }
                    // s^x = v . u^x (unweighted)
                    let mut sx = vec![0.0; np];
                    for c in 0..nu {
                        mul_add(
                            &mut sx,
                            1.0,
                            &vu[c * np..(c + 1) * np],
                            &u[(p * nu + c) * np..(p * nu + c + 1) * np],
                        );
                    }
                    for &(gi, c, dw) in &wg {
                        for j in 0..3 {
                            hacc[gp * n3 + 3 * c + j] += dw[j] * sx[gi];
                        }
                    }
                }
                for p in 0..npert {
                    let gp = 3 * atoms[p / 3] + p % 3;
                    hfix.add(gp * n3, &hacc[gp * n3..(gp + 1) * n3]);
                }
                hfix.tick(1);
            }
        });
        let mut h = vec![0.0; if want_h { n3 * n3 } else { 0 }];
        if want_h {
            h = merge_all(works.into_iter().map(|w| vec![w.0]).collect())
                .map(|v| v[0].to_f64())
                .unwrap_or(h);
        }
        let v = PotentialDerivs {
            nbf,
            nch: NCH,
            acc: vacc,
        };
        (v, h)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use linus_grid::{GridSpec, MolecularGrid};
    use linus_ints::basis::BasisDef;
    use linus_ints::BasisSet;

    /// Distorted water (bohr), def2-SVP, SAD density, a coarse PySCF grid.
    #[allow(clippy::type_complexity, reason = "a one-off tuple of buffers")]
    fn setup() -> (
        Vec<u32>,
        Vec<[f64; 3]>,
        BasisSet,
        Mat<f64>,
        Mat<f64>,
        GridSpec,
    ) {
        let z = vec![8, 1, 1];
        let xyz = vec![
            [0.02, -0.01, 0.22],
            [0.05, 1.43, -0.89],
            [-0.07, -1.40, -0.93],
        ];
        let def = BasisDef::load("def2-svp", None).unwrap();
        let basis = BasisSet::new(&def, &z, &xyz, true).unwrap();
        let (da, db) = linus_scf::guess::sad_guess(&def, &basis, &z, 5, 5, Par::Seq);
        (z, xyz, basis, da, db, GridSpec::pyscf(1))
    }

    /// The unprojected fixed-grid (or, with `grid`, first-order grid) terms of `NCH` channels.
    fn raw<const NCH: usize>(
        xc: &XcIntegrator,
        d: [&Mat<f64>; NCH],
        grid: bool,
    ) -> (Vec<Vec<Mat<f64>>>, Vec<f64>) {
        let (v, h) = if grid {
            xc.grid_terms_ch::<NCH>(d, true, true, None)
        } else {
            xc.hessian_terms_ch::<NCH>(d, true, true, None)
        };
        (v.unpack(), h)
    }

    fn fd5(mut f: impl FnMut(f64) -> Vec<f64>, h: f64) -> Vec<f64> {
        let (p1, m1, p2, m2) = (f(h), f(-h), f(2.0 * h), f(-2.0 * h));
        (0..p1.len())
            .map(|i| (8.0 * (p1[i] - m1[i]) - (p2[i] - m2[i])) / (12.0 * h))
            .collect()
    }

    fn flat(m: &Mat<f64>) -> Vec<f64> {
        mat_to_colmajor(m.as_ref())
    }

    /// V_xc^x and E_xc^xy (fixed grid; or with the first-order grid terms on a moving grid)
    /// against five-point differences of V_xc and of the fixed-grid XC gradient.
    fn check(func: &str, moving: bool) {
        let (z, xyz, basis, da, db, spec) = setup();
        let g0 = MolecularGrid::build(&z, &xyz, &spec).unwrap();
        let xc = XcIntegrator::new(&basis, g0.clone(), func).unwrap();
        let dt: Mat<f64> = &da + &db;
        let (mut v, mut hx) = raw::<1>(&xc, [&dt], false);
        if moving {
            let (vg, hg) = raw::<1>(&xc, [&dt], true);
            v[0].iter_mut().zip(&vg[0]).for_each(|(a, b)| *a += b);
            hx.iter_mut().zip(&hg).for_each(|(a, b)| *a += b);
        }
        let n3 = 9;
        for y in 0..n3 {
            let at = |s: f64| {
                let mut c = xyz.clone();
                c[y / 3][y % 3] += s;
                let mut b = basis.clone();
                b.set_positions(&c);
                let g = if moving {
                    MolecularGrid::build(&z, &c, &spec).unwrap()
                } else {
                    g0.clone()
                };
                XcIntegrator::new(&b, g, func).unwrap()
            };
            let fv = fd5(|s| flat(&at(s).potential(&da, &db, true).1), 1e-4);
            let fg = fd5(|s| at(s).gradient(&da, &db, true, false).concat(), 1e-4);
            let an_v = flat(&v[0][y]);
            let ev = fv
                .iter()
                .zip(&an_v)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0, f64::max);
            let eh = (0..n3)
                .map(|x| (fg[x] - hx[x * n3 + y]).abs())
                .fold(0.0, f64::max);
            assert!(
                ev < 1e-8 && eh < 1e-7,
                "{func} moving={moving} y={y}: V err {ev:.2e}, H err {eh:.2e}"
            );
        }
    }

    #[test]
    fn lda_fixed_grid() {
        check("svwn", false);
    }

    #[test]
    fn gga_fixed_and_moving_grid() {
        check("pbe", false);
        check("b3lyp", true);
    }

    #[test]
    fn meta_gga_moving_grid() {
        check("r2scan", true);
    }

    /// Open-shell V_s^x and E_xc^xy (fixed grid; or with the first-order grid terms on a moving
    /// grid) against five-point differences of the spin potentials and of the fixed-grid
    /// unrestricted XC gradient, at spin densities that differ.
    fn check_uks(func: &str, moving: bool) {
        let (z, xyz, basis, d0, _, spec) = setup();
        let n = basis.nbf;
        // physical (sum of squares) spin densities that differ: the leading eigenvectors of the
        // SAD density, five for alpha and four for beta (meta-GGAs need tau >= tau_W)
        let (w, u) = linus_base::linalg::eigh(d0.as_ref(), Par::Seq);
        let part = |k: usize| {
            Mat::from_fn(n, n, |i, j| {
                (0..k)
                    .map(|m| w[n - 1 - m] * u[(i, n - 1 - m)] * u[(j, n - 1 - m)])
                    .sum::<f64>()
            })
        };
        let (da, db) = (part(5), part(4));
        let g0 = MolecularGrid::build(&z, &xyz, &spec).unwrap();
        let xc = XcIntegrator::new(&basis, g0.clone(), func).unwrap();
        let (mut v, mut hx) = raw::<2>(&xc, [&da, &db], false);
        if moving {
            let (vg, hg) = raw::<2>(&xc, [&da, &db], true);
            for s in 0..2 {
                v[s].iter_mut().zip(&vg[s]).for_each(|(a, b)| *a += b);
            }
            hx.iter_mut().zip(&hg).for_each(|(a, b)| *a += b);
        }
        let n3 = 9;
        for y in 0..n3 {
            let at = |s: f64| {
                let mut c = xyz.clone();
                c[y / 3][y % 3] += s;
                let mut b = basis.clone();
                b.set_positions(&c);
                let g = if moving {
                    MolecularGrid::build(&z, &c, &spec).unwrap()
                } else {
                    g0.clone()
                };
                XcIntegrator::new(&b, g, func).unwrap()
            };
            let fv = fd5(
                |s| {
                    let (_, va, vb) = at(s).potential(&da, &db, false);
                    [flat(&va), flat(&vb)].concat()
                },
                1e-4,
            );
            let fg = fd5(|s| at(s).gradient(&da, &db, false, false).concat(), 1e-4);
            let an_v = [flat(&v[0][y]), flat(&v[1][y])].concat();
            let ev = fv
                .iter()
                .zip(&an_v)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0, f64::max);
            let eh = (0..n3)
                .map(|x| (fg[x] - hx[x * n3 + y]).abs())
                .fold(0.0, f64::max);
            assert!(
                ev < 1e-8 && eh < 1e-7,
                "{func} uks moving={moving} y={y}: V err {ev:.2e}, H err {eh:.2e}"
            );
        }
    }

    #[test]
    fn uks_lda_gga_fixed_grid() {
        check_uks("svwn", false);
        check_uks("pbe", false);
    }

    #[test]
    fn uks_hybrid_meta_gga_moving_grid() {
        check_uks("b3lyp", true);
        check_uks("r2scan", true);
    }

    /// (TPSS is checked against PySCF instead: its iso-orbital indicator is clamped at z = 1
    /// in one-orbital regions, which leaves its derivatives too rough for these differences.)
    #[test]
    fn uks_meta_gga_fixed_grid() {
        check_uks("r2scan", false);
    }

    /// Open-shell terms: the same bits for 1 and 4 threads; with equal spin densities they equal
    /// the closed-shell terms (V_a = V_b = V, same E^xy) to roundoff.
    #[test]
    fn uks_terms_deterministic_and_closed_shell_limit() {
        let (z, xyz, basis, da, _db, spec) = setup();
        let xc = XcIntegrator::new(
            &basis,
            MolecularGrid::build(&z, &xyz, &spec).unwrap(),
            "pbe",
        )
        .unwrap();
        let run = |t: usize| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(t)
                .build()
                .unwrap()
                .install(|| {
                    (
                        raw::<2>(&xc, [&da, &da], false),
                        raw::<2>(&xc, [&da, &da], true),
                    )
                })
        };
        let r1 = run(1);
        let r4 = run(4);
        assert!(
            r4.0 .0 == r1.0 .0 && r4.0 .1 == r1.0 .1 && r4.1 .0 == r1.1 .0 && r4.1 .1 == r1.1 .1,
            "open-shell XC Hessian terms differ with 4 threads"
        );
        let (vr, hr) = raw::<1>(&xc, [&(&da * 2.0)], false);
        let (vgr, hgr) = raw::<1>(&xc, [&(&da * 2.0)], true);
        let md = |a: &[f64], b: &[f64]| {
            a.iter()
                .zip(b)
                .map(|(x, y)| (x - y).abs())
                .fold(0.0f64, f64::max)
        };
        assert!(
            md(&r1.0 .1, &hr) < 1e-10 && md(&r1.1 .1, &hgr) < 1e-10,
            "E^xy: {:e} {:e}",
            md(&r1.0 .1, &hr),
            md(&r1.1 .1, &hgr)
        );
        for s in 0..2 {
            for p in 0..9 {
                assert!(
                    md(&flat(&r1.0 .0[s][p]), &flat(&vr[0][p])) < 1e-10
                        && md(&flat(&r1.1 .0[s][p]), &flat(&vgr[0][p])) < 1e-10,
                    "V spin {s} p {p}"
                );
            }
        }
    }

    /// Explicit XC Hessian terms and first-order grid terms: the same bits for 1, 2 and 5 threads.
    #[test]
    fn hessian_terms_deterministic() {
        let (z, xyz, basis, da, db, spec) = setup();
        let xc = XcIntegrator::new(
            &basis,
            MolecularGrid::build(&z, &xyz, &spec).unwrap(),
            "b3lyp",
        )
        .unwrap();
        let dt: Mat<f64> = &da + &db;
        let run = |t: usize| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(t)
                .build()
                .unwrap()
                .install(|| (raw::<1>(&xc, [&dt], false), raw::<1>(&xc, [&dt], true)))
        };
        let r1 = run(1);
        for t in [2, 5] {
            let r = run(t);
            assert!(
                r.0 .0 == r1.0 .0 && r.0 .1 == r1.0 .1,
                "hessian_terms differs with {t} threads"
            );
            assert!(
                r.1 .0 == r1.1 .0 && r.1 .1 == r1.1 .1,
                "grid_terms differs with {t} threads"
            );
        }
    }

    /// The reference closed-shell kernel contraction (which `kernel`'s tests check the
    /// production kernel against) against differences of V_xc along a perturbed density
    /// (meta-GGA).
    #[test]
    fn kernel_contraction() {
        let (z, xyz, basis, da, db, spec) = setup();
        let n = basis.nbf;
        let xc = XcIntegrator::new(
            &basis,
            MolecularGrid::build(&z, &xyz, &spec).unwrap(),
            "r2scan",
        )
        .unwrap();
        let d1 = Mat::from_fn(n, n, |i, j| {
            1e-2 * (((i * 7 + j * 7) % 11) as f64 - 5.0) / 5.0
                * (-((i as f64 - j as f64).abs()) / 6.0).exp()
        });
        let dt: Mat<f64> = &da + &db;
        let an = xc.kernel_contract_restricted_ref(&dt, std::slice::from_ref(&d1));
        let fd = fd5(
            |s| {
                flat(
                    &xc.potential(&(&da + &d1 * (0.5 * s)), &(&db + &d1 * (0.5 * s)), true)
                        .1,
                )
            },
            1e-3,
        );
        let err = fd
            .iter()
            .zip(&flat(&an[0]))
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        assert!(err < 1e-9, "kernel err {err:.2e}");
    }
}
