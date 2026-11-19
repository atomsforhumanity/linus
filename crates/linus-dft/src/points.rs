//! Pointwise pieces of the XC integrator over spin channels: the functional's inputs at the grid
//! points of a batch and its weighted "effective" derivatives with respect to the density
//! variables of each channel.
//!
//! A batch's density variables per channel s are u_s = (rho_s, grad rho_s, tau_s): rho for an
//! LDA, with grad rho for a GGA, with tau for a meta-GGA; `NS` = [`nu_of`] of them. One channel is a closed shell's total
//! density with the unpolarized functional; two are the spin densities with the polarized one,
//! whose sigmas (sigma_aa, sigma_ab, sigma_bb) couple the channels. Every order of the integrator
//! (kernel, Hessian, grid terms) only sees wv = w df/du (NCH NS x ld) and wF = w d2f/du2
//! ((NCH NS)^2 x ld) from [`point_derivs`], so it is written once for both channel counts: the
//! channel count is a const generic, and the one-channel code is the closed-shell code.

use linus_xc::{XcInput, XcOutput};

/// Density variables per channel: 1 (LDA), 4 (GGA: rho, grad rho) or 5 (meta-GGA: + tau).
#[inline]
pub(crate) fn nu_of(gga: bool, mgga: bool) -> usize {
    if mgga {
        5
    } else if gga {
        4
    } else {
        1
    }
}

/// libxc's input of a batch from rho, sigma and tau (as many points as `rho` has per channel).
pub(crate) fn xc_input<'a>(
    gga: bool,
    mgga: bool,
    rho: &'a [f64],
    sigma: &'a [f64],
    tau: &'a [f64],
) -> XcInput<'a> {
    if mgga {
        XcInput::mgga(rho, sigma, tau)
    } else if gga {
        XcInput::gga(rho, sigma)
    } else {
        XcInput::lda(rho)
    }
}

/// Polarized libxc inputs (rho_a, rho_b; sigma_aa, sigma_ab, sigma_bb; tau_a, tau_b per point)
/// from per-spin rho (s np + g), grad ((s 3 + k) np + g) and tau.
#[allow(clippy::too_many_arguments)]
pub(crate) fn polarized_inputs(
    gga: bool,
    mgga: bool,
    np: usize,
    rho: &[f64],
    grad: &[f64],
    tau: &[f64],
    rin: &mut Vec<f64>,
    sin: &mut Vec<f64>,
    tin: &mut Vec<f64>,
) {
    rin.clear();
    sin.clear();
    tin.clear();
    for g in 0..np {
        rin.push(rho[g]);
        rin.push(rho[np + g]);
        if gga {
            let d = |a: usize, b: usize| {
                (0..3)
                    .map(|k| grad[(a * 3 + k) * np + g] * grad[(b * 3 + k) * np + g])
                    .sum::<f64>()
            };
            sin.push(d(0, 0));
            sin.push(d(0, 1));
            sin.push(d(1, 1));
        }
        if mgga {
            tin.push(tau[g]);
            tin.push(tau[np + g]);
        }
    }
}

/// Weighted effective functional derivatives of a closed-shell batch from libxc's (order-2)
/// output: wv = w df/du (nu x ld) and wF = w d2f/du2 (nu x nu x ld), u = (rho, grad rho, tau),
/// with sigma = |grad rho|^2 folded in by the chain rule; zero past `np`. `grad` holds grad rho
/// (3 x ld).
#[allow(clippy::too_many_arguments)]
pub(crate) fn weighted_derivs(
    out: &XcOutput,
    gga: bool,
    mgga: bool,
    w: &[f64],
    grad: &[f64],
    np: usize,
    ld: usize,
    wv: &mut [f64],
    wf: &mut [f64],
) {
    let nu = nu_of(gga, mgga);
    wv[..nu * ld].fill(0.0);
    wf[..nu * nu * ld].fill(0.0);
    let (vrho, v2rho2) = (out.vrho(), out.v2rho2());
    let fidx = |c: usize, d: usize| (c * nu + d) * ld;
    for g in 0..np {
        let ww = w[g];
        wv[g] = ww * vrho[g];
        wf[fidx(0, 0) + g] = ww * v2rho2[g];
    }
    if gga {
        let (vs, vrs, vss) = (out.vsigma(), out.v2rhosigma(), out.v2sigma2());
        for g in 0..np {
            let ww = w[g];
            let gr = [grad[g], grad[ld + g], grad[2 * ld + g]];
            for k in 0..3 {
                wv[(1 + k) * ld + g] = ww * 2.0 * vs[g] * gr[k];
                let frk = ww * 2.0 * vrs[g] * gr[k];
                wf[fidx(0, 1 + k) + g] = frk;
                wf[fidx(1 + k, 0) + g] = frk;
                for l in 0..3 {
                    wf[fidx(1 + k, 1 + l) + g] = ww
                        * (4.0 * vss[g] * gr[k] * gr[l] + if k == l { 2.0 * vs[g] } else { 0.0 });
                }
            }
        }
    }
    if mgga {
        let (vt, vrt, vst, vtt) = (out.vtau(), out.v2rhotau(), out.v2sigmatau(), out.v2tau2());
        for g in 0..np {
            let ww = w[g];
            wv[4 * ld + g] = ww * vt[g];
            wf[fidx(0, 4) + g] = ww * vrt[g];
            wf[fidx(4, 0) + g] = ww * vrt[g];
            wf[fidx(4, 4) + g] = ww * vtt[g];
            for k in 0..3 {
                let f = ww * 2.0 * vst[g] * grad[k * ld + g];
                wf[fidx(1 + k, 4) + g] = f;
                wf[fidx(4, 1 + k) + g] = f;
            }
        }
    }
}

/// Index of v2sigma2 (aa_aa, aa_ab, aa_bb, ab_ab, ab_bb, bb_bb) for sigma indices t, t2.
const SIG2: [[usize; 3]; 3] = [[0, 1, 2], [1, 3, 4], [2, 4, 5]];

/// Weighted effective functional derivatives of a spin-polarized batch from libxc's (order-2,
/// polarized) output, with the spin variables u = (u_a, u_b), u_s = (rho_s, grad rho_s, tau_s):
/// wv = w df/du (nu x ld) and wF = w d2f/du2 (nu x nu x ld), nu = 2 nu_s, component
/// c = s nu_s + (0: rho, 1..3: grad, 4: tau); zero past `np`. The sigmas
/// (sigma_aa, sigma_ab, sigma_bb) = (g_a.g_a, g_a.g_b, g_b.g_b) are folded in by the chain rule:
/// df/dg_s = sum_t f_t dsigma_t/dg_s, d2f/dg_s dg_s' = sum_tt' f_tt' (dsigma_t/dg_s)(dsigma_t'/dg_s')
/// + sum_t f_t d2sigma_t/dg_s dg_s'. `grad` holds grad rho_s at (s 3 + k) ld + g.
#[allow(clippy::too_many_arguments)]
#[allow(clippy::needless_range_loop, reason = "indexes several arrays in step")]
pub(crate) fn weighted_derivs_spin(
    out: &XcOutput,
    gga: bool,
    mgga: bool,
    w: &[f64],
    grad: &[f64],
    np: usize,
    ld: usize,
    wv: &mut [f64],
    wf: &mut [f64],
) {
    let ns = nu_of(gga, mgga);
    let nu = 2 * ns;
    wv[..nu * ld].fill(0.0);
    wf[..nu * nu * ld].fill(0.0);
    let fidx = |c: usize, d: usize| (c * nu + d) * ld;
    let (vrho, v2rho2) = (out.vrho(), out.v2rho2());
    for g in 0..np {
        let ww = w[g];
        for s in 0..2 {
            wv[s * ns * ld + g] = ww * vrho[2 * g + s];
            for s2 in 0..2 {
                // v2rho2 = (aa, ab, bb): index s + s2
                wf[fidx(s * ns, s2 * ns) + g] = ww * v2rho2[3 * g + s + s2];
            }
        }
    }
    if gga {
        let (vs, vrs, vss) = (out.vsigma(), out.v2rhosigma(), out.v2sigma2());
        for g in 0..np {
            let ww = w[g];
            let gr = |s: usize, k: usize| grad[(s * 3 + k) * ld + g];
            // dsigma_t / d g_{s,k}
            let ds = |t: usize, s: usize, k: usize| -> f64 {
                match (t, s) {
                    (0, 0) => 2.0 * gr(0, k),
                    (1, 0) => gr(1, k),
                    (1, 1) => gr(0, k),
                    (2, 1) => 2.0 * gr(1, k),
                    _ => 0.0,
                }
            };
            let vsig = [vs[3 * g], vs[3 * g + 1], vs[3 * g + 2]];
            // second derivatives of the sigmas: d2 sigma_aa = 2, d2 sigma_ab = 1 (a-b), d2 sigma_bb = 2
            let d2 = [[2.0 * vsig[0], vsig[1]], [vsig[1], 2.0 * vsig[2]]];
            for s in 0..2 {
                for k in 0..3 {
                    let c = s * ns + 1 + k;
                    wv[c * ld + g] = ww * (0..3).map(|t| vsig[t] * ds(t, s, k)).sum::<f64>();
                    for sr in 0..2 {
                        // rho_sr - g_{s,k}: v2rhosigma = (a_aa, a_ab, a_bb, b_aa, b_ab, b_bb)
                        let f = ww
                            * (0..3)
                                .map(|t| vrs[6 * g + 3 * sr + t] * ds(t, s, k))
                                .sum::<f64>();
                        wf[fidx(sr * ns, c) + g] = f;
                        wf[fidx(c, sr * ns) + g] = f;
                    }
                    for s2 in 0..2 {
                        for l in 0..3 {
                            let c2 = s2 * ns + 1 + l;
                            let mut f = 0.0;
                            for t in 0..3 {
                                let a = ds(t, s, k);
                                if a == 0.0 {
                                    continue;
                                }
                                for t2 in 0..3 {
                                    f += vss[6 * g + SIG2[t][t2]] * a * ds(t2, s2, l);
                                }
                            }
                            if k == l {
                                f += d2[s][s2];
                            }
                            wf[fidx(c, c2) + g] = ww * f;
                        }
                    }
                }
            }
        }
    }
    if mgga {
        let (vt, vrt, vst, vtt) = (out.vtau(), out.v2rhotau(), out.v2sigmatau(), out.v2tau2());
        for g in 0..np {
            let ww = w[g];
            let gr = |s: usize, k: usize| grad[(s * 3 + k) * ld + g];
            let ds = |t: usize, s: usize, k: usize| -> f64 {
                match (t, s) {
                    (0, 0) => 2.0 * gr(0, k),
                    (1, 0) => gr(1, k),
                    (1, 1) => gr(0, k),
                    (2, 1) => 2.0 * gr(1, k),
                    _ => 0.0,
                }
            };
            for s in 0..2 {
                let ct = s * ns + 4;
                wv[ct * ld + g] = ww * vt[2 * g + s];
                for s2 in 0..2 {
                    let ct2 = s2 * ns + 4;
                    // v2tau2 = (aa, ab, bb); v2rhotau = (a_a, a_b, b_a, b_b) (rho spin, tau spin)
                    wf[fidx(ct, ct2) + g] = ww * vtt[3 * g + s + s2];
                    let f = ww * vrt[4 * g + 2 * s2 + s];
                    wf[fidx(s2 * ns, ct) + g] = f;
                    wf[fidx(ct, s2 * ns) + g] = f;
                    // v2sigmatau = (aa_a, aa_b, ab_a, ab_b, bb_a, bb_b) (sigma, tau spin)
                    for k in 0..3 {
                        let c = s2 * ns + 1 + k;
                        let f = ww
                            * (0..3)
                                .map(|t| vst[6 * g + 2 * t + s] * ds(t, s2, k))
                                .sum::<f64>();
                        wf[fidx(c, ct) + g] = f;
                        wf[fidx(ct, c) + g] = f;
                    }
                }
            }
        }
    }
}

/// The points of a batch at which the weighted derivatives are formed: the functional's family,
/// the weights, the density gradients (grad rho_s at (s 3 + k) ld + g), np real points of ld.
#[derive(Clone, Copy)]
pub(crate) struct Points<'a> {
    pub(crate) gga: bool,
    pub(crate) mgga: bool,
    pub(crate) w: &'a [f64],
    pub(crate) grad: &'a [f64],
    pub(crate) np: usize,
    pub(crate) ld: usize,
}

/// The weighted effective derivatives of `NCH` channels at the points `pts` ([`weighted_derivs`]
/// for one, [`weighted_derivs_spin`] for two): wv (NCH nu_s x ld) and wF ((NCH nu_s)^2 x ld),
/// zero past np.
#[inline]
pub(crate) fn point_derivs<const NCH: usize>(
    out: &XcOutput,
    pts: &Points<'_>,
    wv: &mut [f64],
    wf: &mut [f64],
) {
    let Points {
        gga,
        mgga,
        w,
        grad,
        np,
        ld,
    } = *pts;
    match NCH {
        1 => weighted_derivs(out, gga, mgga, w, grad, np, ld, wv, wf),
        2 => weighted_derivs_spin(out, gga, mgga, w, grad, np, ld, wv, wf),
        _ => unreachable!("{NCH} spin channels"),
    }
}

/// A GGA-like potential given per grid point instead of by a functional (VV10's, whose kernel
/// is nonlocal): v_rho and v_sigma (closed shell, total density) at every point of the grid.
pub(crate) struct ExtPotential<'a> {
    pub(crate) vrho: &'a [f64],
    pub(crate) vsigma: &'a [f64],
}

impl ExtPotential<'_> {
    /// wv = w (v_rho, 2 v_sigma grad rho) of the np points from `start` (as [`weighted_derivs`]
    /// for a GGA); `grad` holds grad rho (3 x np).
    pub(crate) fn weighted(&self, start: usize, w: &[f64], grad: &[f64], wv: &mut [f64]) {
        let np = w.len();
        for gi in 0..np {
            let ww = w[gi];
            wv[gi] = ww * self.vrho[start + gi];
            let vs = self.vsigma[start + gi];
            for k in 0..3 {
                wv[(1 + k) * np + gi] = ww * 2.0 * vs * grad[k * np + gi];
            }
        }
    }
}

/// y += s a b (elementwise).
#[inline]
pub(crate) fn mul_add(y: &mut [f64], s: f64, a: &[f64], b: &[f64]) {
    for ((y, a), b) in y.iter_mut().zip(a).zip(b) {
        *y += s * a * b;
    }
}

/// y = s a b (elementwise).
#[inline]
pub(crate) fn mul_set(y: &mut [f64], s: f64, a: &[f64], b: &[f64]) {
    for ((y, a), b) in y.iter_mut().zip(a).zip(b) {
        *y = s * a * b;
    }
}

/// sum_i a_i b_i (four partial sums, so it vectorizes).
pub(crate) fn dot(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    let (a, b) = (&a[..n], &b[..n]);
    let mut acc = [0.0f64; 4];
    let ((ca, ra), (cb, rb)) = (a.as_chunks::<4>(), b.as_chunks::<4>());
    for (x, y) in ca.iter().zip(cb) {
        for l in 0..4 {
            acc[l] += x[l] * y[l];
        }
    }
    let mut s = (acc[0] + acc[1]) + (acc[2] + acc[3]);
    for (x, y) in ra.iter().zip(rb) {
        s += x * y;
    }
    s
}
