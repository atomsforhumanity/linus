//! XC kernel contractions (first-order XC potentials) for response equations, over spin
//! channels.
//!
//! At each point the functional is f(u), u = (u_s) with u_s = (rho_s, grad rho_s, tau_s) per
//! channel s, with the weighted "effective" second derivatives wF = w d2f/du2 (the sigmas folded
//! in, see [`crate::points`]; an open shell's wF couples its two spin channels). A
//! first-order density D1 gives u1 on the grid, t = wF u1, and per channel the first-order
//! potential
//!   V1_mn = sum_g [t_rho phi_m phi_n + t_k (d_k phi_m phi_n + phi_m d_k phi_n)
//!           + 1/2 t_tau grad phi_m . grad phi_n].
//!
//! [`XcIntegrator::kernel_contract_occ`] gives V1_s C_s for D1_s = A_s C_s^T + C_s A_s^T, the
//! form of every CPKS density, at the ground state of the occupied orbitals (one channel: a
//! closed shell, D0 = 2 C_o C_o^T; two: the spins, D_s = C_s C_s^T). With Phi = phi C_s,
//! P = phi A_s and Q = Phi A_s^T (per batch and channel),
//!   rho1 = 2 sum_i P_i Phi_i,  d_k rho1 = 2 sum_i P_i d_k Phi_i + 2 sum_m d_k phi_m Q_m,
//!   tau1 = sum_k sum_i d_k P_i d_k Phi_i
//! (halved for a spin channel), and
//!   V1 C_s = phi^T (t_rho Phi + t_k d_k Phi) + (t_k d_k phi)^T Phi +
//!   1/2 sum_k d_k phi^T (t_tau d_k Phi),
//! four (nao x np x nocc) products per perturbation and channel instead of the two
//! (nao x np x nao) of full first-order potentials, no nao^2 gathers or scatters, and an
//! nbf x nocc output. One implementation, generic over the channel count.
//!
//! It runs in parallel over batches with fixed-point accumulation (results independent of the
//! thread count). Perturbations are processed in groups whose per-thread accumulators fit a
//! memory budget (`XcIntegrator::kernel_budget_mb`, default 1024 MB for all threads; set from
//! `memory_gib`); the batch work that does not depend on the perturbations is repeated per
//! group.

use crate::batch::{BatchCtx, BatchPlan};
use crate::points::{nu_of, point_derivs, Points};
use crate::{
    eval_functional, gemm, gemm_part, libxc_inputs, mmut, mref, multiversion, sized, Kind, Scratch,
    XcIntegrator,
};
use faer::{Mat, MatRef};
use linus_base::error::LinusError;
use linus_base::linalg::Whole;
use linus_base::terms::ResponseKernel;
use linus_core::channels::Channels;
use linus_num::det::{merge_all, FixedAcc};
use rayon::prelude::*;

/// Target size (doubles) of the per-batch perturbation blocks (kept in L2/L3).
///
/// The blocks of perturbations a batch processes together (and with them the shapes of its
/// GEMMs, which set the rounding) come from this size and the total number of perturbations
/// only, in the same places for every group: the groups of perturbations that bound the
/// accumulators' memory (sized from the memory budget and the thread count) run whole blocks
/// and keep their own perturbations' results, so the result does not depend on the thread
/// count or the memory budget.
const BLOCK_DOUBLES: usize = 1 << 17;

/// The blocks [p0, p1) of `ps` perturbations (of `n_all`, aligned at multiples of `ps`) that
/// meet the group `range`.
fn blocks(
    range: &std::ops::Range<usize>,
    ps: usize,
    n_all: usize,
) -> impl Iterator<Item = (usize, usize)> {
    let first = range.start / ps * ps;
    let end = range.end;
    (first..end)
        .step_by(ps)
        .map(move |p0| (p0, (p0 + ps).min(n_all)))
}

/// t = wF u (nu x ld) for one perturbation.
#[inline(always)]
fn apply_wf_body(nu: usize, ld: usize, wf: &[f64], u: &[f64], t: &mut [f64]) {
    for c in 0..nu {
        let tc = &mut t[c * ld..(c + 1) * ld];
        let f0 = &wf[(c * nu) * ld..(c * nu + 1) * ld];
        let u0 = &u[..ld];
        for g in 0..ld {
            tc[g] = f0[g] * u0[g];
        }
        for d in 1..nu {
            let fd = &wf[(c * nu + d) * ld..(c * nu + d + 1) * ld];
            let ud = &u[d * ld..(d + 1) * ld];
            for g in 0..ld {
                tc[g] += fd[g] * ud[g];
            }
        }
    }
}

/// Ground-state quantities of a closed-shell batch from the orbitals: Phi4 = [phi; grad phi] C
/// (into `phi4`, 4 ld x nocc), rho, grad rho, tau into s.rho/s.grad/s.tau and libxc inputs.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn occ_ground_body(
    ld: usize,
    nocc: usize,
    gga: bool,
    mgga: bool,
    phi4: &[f64],
    rho: &mut [f64],
    grad: &mut [f64],
    tau: &mut [f64],
) {
    let s4 = 4 * ld;
    rho[..ld].fill(0.0);
    grad[..3 * ld].fill(0.0);
    tau[..ld].fill(0.0);
    for i in 0..nocc {
        let fi = &phi4[i * s4..i * s4 + ld];
        for g in 0..ld {
            rho[g] += fi[g] * fi[g];
        }
        if gga {
            for k in 0..3 {
                let dk = &phi4[i * s4 + (1 + k) * ld..i * s4 + (2 + k) * ld];
                let gk = &mut grad[k * ld..(k + 1) * ld];
                for g in 0..ld {
                    gk[g] += fi[g] * dk[g];
                }
                if mgga {
                    for g in 0..ld {
                        tau[g] += dk[g] * dk[g];
                    }
                }
            }
        }
    }
    for v in rho[..ld].iter_mut() {
        *v *= 2.0;
    }
    for v in grad[..3 * ld].iter_mut() {
        *v *= 4.0;
    }
}
multiversion!(fn occ_ground(ld: usize, nocc: usize, gga: bool, mgga: bool, phi4: &[f64], rho: &mut [f64], grad: &mut [f64], tau: &mut [f64]) -> () = occ_ground_body);

/// Per-channel buffers of the occupied-projected kernel.
#[derive(Default)]
struct ChBufs {
    abuf: Vec<f64>,
    atbuf: Vec<f64>,
    phi4: Vec<f64>,
    csub: Vec<f64>,
    p: Vec<f64>,
    q: Vec<f64>,
    pk: Vec<f64>,
    z: Vec<f64>,
    tk: Vec<f64>,
    gm: Vec<f64>,
    o: Vec<f64>,
}

/// Per-thread buffers of the occupied-projected kernel.
struct OccWork {
    s: Scratch,
    wv: Vec<f64>,
    wf: Vec<f64>,
    u: Vec<f64>,
    t: Vec<f64>,
    ch: Vec<ChBufs>,
}

impl OccWork {
    fn new(nch: usize) -> Self {
        OccWork {
            s: Scratch::default(),
            wv: Vec::new(),
            wf: Vec::new(),
            u: Vec::new(),
            t: Vec::new(),
            ch: (0..nch).map(|_| ChBufs::default()).collect(),
        }
    }
}

impl ResponseKernel for XcIntegrator {
    fn contract_occ(
        &self,
        c_o: &[MatRef<'_, f64>],
        a: &[Vec<Mat<f64>>],
    ) -> Result<Vec<Vec<Mat<f64>>>, LinusError> {
        let c_o = Channels::from_vec(c_o.to_vec());
        let a = Channels::from_vec(a.iter().map(|x| x.as_slice()).collect());
        Ok(self.kernel_contract_occ(c_o, a).into_vec())
    }
}

impl XcIntegrator {
    /// The XC kernel's first-order potentials projected on the occupied orbitals, over spin
    /// channels (the layout of `linus_scf::response::operator`): for first-order densities
    /// dD_s = A_s C_s^T + C_s A_s^T given by their factors `a[s][k]` (nbf x nocc_s) at the ground
    /// state of the occupied orbitals `c_o[s]`, dV_s\[dD\] C_s as `[s][k]`. One channel: a closed
    /// shell (the alpha orbitals; both spins carry dD, so the result is that of the total
    /// first-order density 2 dD); two: the spins of an open shell
    /// (`Self::kernel_contract_occ_ch`).
    pub fn kernel_contract_occ(
        &self,
        c_o: Channels<MatRef<'_, f64>>,
        a: Channels<&[Mat<f64>]>,
    ) -> Channels<Vec<Mat<f64>>> {
        match (c_o, a) {
            (Channels::Restricted(c), Channels::Restricted(a)) => {
                let mut v = self.kernel_contract_occ_ch::<1>([c], [a]).remove(0);
                for m in v.iter_mut() {
                    *m = &*m * 2.0;
                }
                Channels::Restricted(v)
            }
            (Channels::Unrestricted(c), Channels::Unrestricted(a)) => {
                Channels::from_vec(self.kernel_contract_occ_ch::<2>(c, a))
            }
            _ => panic!("orbitals and perturbations of different spin channels"),
        }
    }

    /// The occupied-projected kernel for `NCH` channels (one: a closed shell, at the ground state
    /// D0 = 2 C_o C_o^T, for the first-order total densities A_p C_o^T + C_o A_p^T; two: an open
    /// shell's spins at D_s = C_s C_s^T): for each perturbation p, V1_s[D1] C_s (nbf x nocc_s) as
    /// `[s][p]` (see the module docs for the algorithm; an open shell's first-order potentials
    /// couple the spins through the spin-resolved wF of [`crate::points`]). LDA, GGA and tau
    /// meta-GGA; fixed-point accumulation (independent of the thread count).
    pub(crate) fn kernel_contract_occ_ch<const NCH: usize>(
        &self,
        c_o: [MatRef<'_, f64>; NCH],
        a: [&[Mat<f64>]; NCH],
    ) -> Vec<Vec<Mat<f64>>> {
        let nbf = self.basis.nbf;
        let no: [usize; NCH] = std::array::from_fn(|s| c_o[s].ncols());
        let npert = a[0].len();
        for s in 0..NCH {
            assert!(
                a[s].len() == npert
                    && c_o[s].nrows() == nbf
                    && a[s].iter().all(|x| x.nrows() == nbf && x.ncols() == no[s])
            );
        }
        if npert == 0 || no.iter().all(|&o| o == 0) {
            return (0..NCH)
                .map(|s| (0..npert).map(|_| Mat::zeros(nbf, no[s])).collect())
                .collect();
        }
        let co: [Mat<f64>; NCH] = c_o.map(|c| c.to_owned());
        // A^T (nocc x nbf, column-major) for gathering rows of A
        let at: [Vec<Mat<f64>>; NCH] =
            std::array::from_fn(|s| a[s].par_iter().map(|x| x.transpose().to_owned()).collect());
        let nthr = rayon::current_num_threads().max(1);
        let per = nthr * nbf * no.iter().sum::<usize>().max(1) * 16;
        let group = ((self.kernel_budget_mb << 20) / per.max(1)).clamp(1, npert);
        let mut out: Vec<Vec<Mat<f64>>> = (0..NCH).map(|_| Vec::with_capacity(npert)).collect();
        for g0 in (0..npert).step_by(group) {
            let g1 = (g0 + group).min(npert);
            let accs = self.kernel_occ_group::<NCH>(&co, a, at.each_ref().map(|v| &v[..]), g0..g1);
            let mats: Vec<Mat<f64>> = accs
                .par_iter()
                .enumerate()
                .map(|(x, acc)| {
                    MatRef::from_column_major_slice(&acc.to_f64(), nbf, no[x % NCH]).to_owned()
                })
                .collect();
            for (x, m) in mats.into_iter().enumerate() {
                out[x % NCH].push(m);
            }
        }
        out
    }

    /// One group of perturbations of [`Self::kernel_contract_occ_ch`]: accumulators
    /// (perturbation-major, channel-minor).
    fn kernel_occ_group<const NCH: usize>(
        &self,
        co: &[Mat<f64>; NCH],
        a: [&[Mat<f64>]; NCH],
        at: [&[Mat<f64>]; NCH],
        range: std::ops::Range<usize>,
    ) -> Vec<FixedAcc> {
        use crate::profile as xp;
        let nbf = self.basis.nbf;
        let no: [usize; NCH] = std::array::from_fn(|s| co[s].ncols());
        let func = self.channel_functional::<NCH>();
        let k = Kind::of(func, NCH == 1);
        let (gga, mgga) = (k.gga, k.mgga);
        let ns = nu_of(gga, mgga);
        let nu = NCH * ns;
        let (n_all, npert) = (a[0].len(), range.len());
        let new_accs = || {
            (0..NCH * npert)
                .map(|x| FixedAcc::new(nbf * no[x % NCH]))
                .collect::<Vec<_>>()
        };
        let col_off: [Vec<usize>; NCH] =
            std::array::from_fn(|s| (0..no[s]).map(|i| i * nbf).collect());
        // gradients are needed for the ground-state Phi4 layout even for LDA (cheap)
        let plan = BatchPlan::new(1, self.func_thresh);
        let init = || (new_accs(), OccWork::new(NCH));
        let (_, works) = self.for_each_batch(&plan, init, |(acc, wk), b| {
            let (np, ld, nao, nc, w) = (b.np, b.ld, b.nao, b.nc, b.w);
            if nao == 0 {
                return;
            }
            let mut lap = xp::Lap::start();
            let OccWork {
                s,
                wv,
                wf,
                u,
                t,
                ch,
            } = wk;
            let (idx, runs, ao) = (b.idx, b.runs, b.ao);
            let s4 = 4 * ld;
            // ground state per channel: Phi4_s = [phi; grad phi] C_s and the channel
            // densities
            sized(&mut s.rho, NCH * ld);
            sized(&mut s.grad, NCH * 3 * ld);
            sized(&mut s.tau, NCH * ld);
            for c in 0..NCH {
                let o = no[c];
                let bufs = &mut ch[c];
                let cs = sized(&mut bufs.csub, nao * o);
                crate::gather_rows(co[c].as_ref(), idx, cs);
                let f4 = sized(&mut bufs.phi4, s4 * o);
                if o > 0 {
                    gemm(
                        mmut(f4, s4, o, s4),
                        false,
                        mref(ao, s4, nao, nc * ld),
                        mref(cs, nao, o, nao),
                        1.0,
                    );
                }
                let (rho, grad, tau) = (
                    &mut s.rho[c * ld..(c + 1) * ld],
                    &mut s.grad[c * 3 * ld..(c + 1) * 3 * ld],
                    &mut s.tau[c * ld..(c + 1) * ld],
                );
                occ_ground(ld, o, gga, mgga, f4, rho, grad, tau);
                if NCH == 2 {
                    // occ_ground gives closed-shell (doubly occupied) values: halve
                    for v in rho.iter_mut().chain(grad.iter_mut()).chain(tau.iter_mut()) {
                        *v *= 0.5;
                    }
                }
            }
            libxc_inputs(k, np, ld, s);
            lap.lap(xp::DENSITY);
            eval_functional(func, k, 2, s);
            lap.lap(xp::LIBXC);
            point_derivs::<NCH>(
                &s.out,
                &Points {
                    gga,
                    mgga,
                    w,
                    grad: &s.grad,
                    np,
                    ld,
                },
                sized(wv, nu * ld),
                sized(wf, nu * nu * ld),
            );
            let phi0 = mref(ao, ld, nao, nc * ld);
            // perturbations in blocks sized for the cache
            let per_p: usize = (0..NCH)
                .map(|c| ld * (2 * no[c] + nao + if mgga { 6 * no[c] } else { 0 }) + nao * no[c])
                .sum();
            let ps = (BLOCK_DOUBLES / per_p.max(1)).clamp(1, n_all);
            for (p0, p1) in blocks(&range, ps, n_all) {
                let pc = p1 - p0;
                // per channel: A_sub (nao x pc nocc) and A^T_sub (nocc x pc nao), then
                // P = phi A (ld x pc nocc), Q = Phi A^T (ld x pc nao) and Pk = d_k phi A
                // (one channel at a time, its blocks still in cache)
                for c in 0..NCH {
                    let o = no[c];
                    if o == 0 {
                        continue;
                    }
                    let bufs = &mut ch[c];
                    let ab = sized(&mut bufs.abuf, nao * pc * o);
                    let atb = sized(&mut bufs.atbuf, o * pc * nao);
                    for pl in 0..pc {
                        crate::gather_rows(
                            a[c][p0 + pl].as_ref(),
                            idx,
                            &mut ab[pl * o * nao..(pl + 1) * o * nao],
                        );
                        if gga {
                            let atm = &at[c][p0 + pl];
                            for (m, &r) in idx.iter().enumerate() {
                                atb[(pl * nao + m) * o..(pl * nao + m + 1) * o]
                                    .copy_from_slice(atm.col_as_slice(r));
                            }
                        }
                    }
                    lap.lap(xp::SETUP);
                    let pcn = pc * o;
                    let pb = sized(&mut bufs.p, ld * pcn);
                    gemm(
                        mmut(pb, ld, pcn, ld),
                        false,
                        phi0,
                        mref(&bufs.abuf, nao, pcn, nao),
                        1.0,
                    );
                    let qb = sized(&mut bufs.q, if gga { ld * pc * nao } else { 0 });
                    if gga {
                        gemm(
                            mmut(qb, ld, pc * nao, ld),
                            false,
                            mref(&bufs.phi4, ld, o, s4),
                            mref(&bufs.atbuf, o, pc * nao, o),
                            1.0,
                        );
                    }
                    let pks = ld * pcn;
                    let pkb = sized(&mut bufs.pk, if mgga { 3 * pks } else { 0 });
                    if mgga {
                        for kk in 0..3 {
                            gemm(
                                mmut(&mut pkb[kk * pks..], ld, pcn, ld),
                                false,
                                mref(&ao[(1 + kk) * ld..], ld, nao, nc * ld),
                                mref(&bufs.abuf, nao, pcn, nao),
                                1.0,
                            );
                        }
                    }
                    sized(&mut bufs.z, ld * pcn);
                    sized(&mut bufs.tk, if mgga { 3 * pks } else { 0 });
                    sized(&mut bufs.gm, if gga { ld * pc * nao } else { 0 });
                    lap.lap(xp::GEMM_D);
                }
                // per perturbation: u1, t, then per channel Z (ld x pc nocc), G_p, T_k
                let uu = sized(u, nu * ld);
                let tt = sized(t, nu * ld);
                if gga && !mgga {
                    // register-blocked GGA path (same bits as the streaming one below);
                    // a channel without electrons gets empty slices (its buffers are
                    // unsized)
                    for pl in 0..pc {
                        let span = |len: usize, o: usize, off: usize| {
                            if o == 0 {
                                0..0
                            } else {
                                off..len
                            }
                        };
                        let mut phi4: [&[f64]; NCH] = [&[]; NCH];
                        let mut p: [&[f64]; NCH] = [&[]; NCH];
                        let mut q: [&[f64]; NCH] = [&[]; NCH];
                        let mut z: [&mut [f64]; NCH] = std::array::from_fn(|_| &mut [][..]);
                        let mut gmm: [&mut [f64]; NCH] = std::array::from_fn(|_| &mut [][..]);
                        for (c, bufs) in ch.iter_mut().enumerate() {
                            let o = no[c];
                            let ChBufs {
                                phi4: f4,
                                p: pv,
                                q: qv,
                                z: zv,
                                gm: gv,
                                ..
                            } = bufs;
                            phi4[c] = &f4[..];
                            p[c] = &pv[span(pv.len(), o, pl * o * ld)];
                            q[c] = &qv[span(qv.len(), o, pl * nao * ld)];
                            let rz = span(zv.len(), o, pl * o * ld);
                            let rg = span(gv.len(), o, pl * nao * ld);
                            z[c] = &mut zv[rz];
                            gmm[c] = &mut gv[rg];
                        }
                        let x = GgaInputs {
                            ao,
                            nc,
                            ld,
                            nao,
                            wf,
                            no: &no,
                            phi4: &phi4,
                            p: &p,
                            q: &q,
                        };
                        occ_gga::<NCH>(&x, &mut z, &mut gmm);
                    }
                } else {
                    for pl in 0..pc {
                        for c in 0..NCH {
                            let o = no[c];
                            let us = &mut uu[c * ns * ld..(c + 1) * ns * ld];
                            if o == 0 {
                                us.fill(0.0);
                                continue;
                            }
                            let bufs = &ch[c];
                            let pks = ld * pc * o;
                            let qq = if gga {
                                &bufs.q[pl * nao * ld..(pl + 1) * nao * ld]
                            } else {
                                &bufs.q[..0]
                            };
                            let pkk = if mgga {
                                &bufs.pk[pl * o * ld..]
                            } else {
                                &bufs.pk[..0]
                            };
                            occ_u1(
                                ao,
                                nc,
                                ld,
                                nao,
                                o,
                                gga,
                                mgga,
                                &bufs.phi4,
                                &bufs.p[pl * o * ld..],
                                ld,
                                qq,
                                pkk,
                                pks,
                                us,
                            );
                        }
                        apply_wf(nu, ld, wf, uu, tt);
                        for c in 0..NCH {
                            let o = no[c];
                            if o == 0 {
                                continue;
                            }
                            let pks = ld * pc * o;
                            let ChBufs {
                                phi4, z, tk, gm, ..
                            } = &mut ch[c];
                            let gmm = if gga {
                                &mut gm[pl * nao * ld..(pl + 1) * nao * ld]
                            } else {
                                &mut gm[..0]
                            };
                            let tkk = if mgga {
                                &mut tk[pl * o * ld..]
                            } else {
                                &mut tk[..0]
                            };
                            occ_out(
                                ao,
                                nc,
                                ld,
                                nao,
                                o,
                                gga,
                                mgga,
                                &tt[c * ns * ld..(c + 1) * ns * ld],
                                phi4,
                                &mut z[pl * o * ld..],
                                ld,
                                pks,
                                gmm,
                                tkk,
                            );
                        }
                    }
                }
                lap.lap(xp::WEIGHTS);
                // O_s = phi^T Z_s + G_s^T Phi_s (+ sum_k d_k phi^T T_k,s), into the
                // accumulators while in cache
                for c in 0..NCH {
                    let o = no[c];
                    if o == 0 {
                        continue;
                    }
                    let ChBufs {
                        phi4,
                        z,
                        gm,
                        tk,
                        o: obuf,
                        ..
                    } = &mut ch[c];
                    let nao_o = nao * o;
                    // the groups run whole aligned blocks: each block is its own whole
                    let x = VcoInputs {
                        nocc: o,
                        pc,
                        pc_whole: pc,
                        gga,
                        mgga,
                        phi4,
                        z,
                        gm,
                        tk,
                    };
                    let ob = push_vco(b, &x, obuf);
                    lap.lap(xp::GEMM_V);
                    for pl in (0..pc).filter(|pl| range.contains(&(p0 + pl))) {
                        crate::scatter_cols(
                            &mut acc[NCH * (p0 + pl - range.start) + c],
                            &col_off[c],
                            nao,
                            runs,
                            &ob[pl * nao_o..(pl + 1) * nao_o],
                        );
                    }
                    lap.lap(xp::SCATTER);
                }
            }
            for x in acc.iter_mut() {
                x.tick(1);
            }
        });
        merge_all(works.into_iter().map(|x| x.0).collect()).unwrap_or_else(new_accs)
    }
}

/// What [`push_vco`] contracts for one channel and `pc` perturbations: Z (ld x pc nocc), G (GGA;
/// pc blocks of ld x nao), Phi4 (`phi4`: [Phi; grad Phi] per orbital, column stride 4 ld) and T_k
/// (meta-GGA; 3 blocks of ld x pc nocc, stride ld pc nocc).
pub(crate) struct VcoInputs<'a> {
    pub(crate) nocc: usize,
    pub(crate) pc: usize,
    /// The perturbations of the whole block the `pc` are a part of (`pc` when they are the
    /// block): the products take the whole block's kernels (VV10's groups).
    pub(crate) pc_whole: usize,
    pub(crate) gga: bool,
    pub(crate) mgga: bool,
    pub(crate) phi4: &'a [f64],
    pub(crate) z: &'a [f64],
    pub(crate) gm: &'a [f64],
    pub(crate) tk: &'a [f64],
}

/// The occupied-projected potential products of one channel of a function-major batch:
/// O_p = phi^T Z_p + G_p^T Phi (+ sum_k d_k phi^T T_k,p for a meta-GGA), nao x nocc each, into
/// `ob` (perturbation-major).
pub(crate) fn push_vco<'o>(b: &BatchCtx<'_>, x: &VcoInputs<'_>, ob: &'o mut Vec<f64>) -> &'o [f64] {
    let VcoInputs {
        nocc,
        pc,
        pc_whole,
        gga,
        mgga,
        phi4,
        z,
        gm,
        tk,
    } = *x;
    let (nc, ld, nao) = (b.nc, b.ld, b.nao);
    let pcn = pc * nocc;
    let nao_o = nao * nocc;
    let pks = ld * pcn;
    let ob = sized(ob, pc * nao_o);
    let whole = Whole {
        rows: nao,
        cols: pc_whole * nocc,
    };
    gemm_part(
        whole,
        mmut(ob, nao, pcn, nao),
        false,
        mref(b.ao, ld, nao, nc * ld).transpose(),
        mref(z, ld, pcn, ld),
        1.0,
    );
    if gga {
        for pl in 0..pc {
            gemm(
                mmut(&mut ob[pl * nao_o..], nao, nocc, nao),
                true,
                mref(&gm[pl * nao * ld..], ld, nao, ld).transpose(),
                mref(phi4, ld, nocc, 4 * ld),
                1.0,
            );
        }
    }
    if mgga {
        for kk in 0..3 {
            gemm_part(
                whole,
                mmut(ob, nao, pcn, nao),
                true,
                mref(&b.ao[(1 + kk) * ld..], ld, nao, nc * ld).transpose(),
                mref(&tk[kk * pks..], ld, pcn, ld),
                1.0,
            );
        }
    }
    ob
}

/// First-order density variables of one channel and one perturbation of the occupied kernel
/// (its streaming path's first half; [`occ_gga_chunk`] is the GGA one): u1 = (rho1, grad rho1, tau1) (nu_s x ld) from P = phi A, Phi4, Q = Phi A^T
/// and (meta-GGA) Pk = d_k phi A, for D1 = A C^T + C A^T.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn occ_u1_body(
    ao: &[f64],
    nc: usize,
    ld: usize,
    nao: usize,
    nocc: usize,
    gga: bool,
    mgga: bool,
    phi4: &[f64],
    p: &[f64],
    ps: usize,
    q: &[f64],
    pk: &[f64],
    pks: usize,
    u: &mut [f64],
) {
    let nu = nu_of(gga, mgga);
    let s4 = 4 * ld;
    u[..nu * ld].fill(0.0);
    let (u0, rest) = u.split_at_mut(ld);
    for i in 0..nocc {
        let pi = &p[i * ps..i * ps + ld];
        let fi = &phi4[i * s4..i * s4 + ld];
        if gga {
            let (u1, rest2) = rest.split_at_mut(ld);
            let (u2, rest3) = rest2.split_at_mut(ld);
            let u3 = &mut rest3[..ld];
            let (d1, d2, d3) = (
                &phi4[i * s4 + ld..i * s4 + 2 * ld],
                &phi4[i * s4 + 2 * ld..i * s4 + 3 * ld],
                &phi4[i * s4 + 3 * ld..i * s4 + 4 * ld],
            );
            for g in 0..ld {
                let pg = pi[g];
                u0[g] += pg * fi[g];
                u1[g] += pg * d1[g];
                u2[g] += pg * d2[g];
                u3[g] += pg * d3[g];
            }
        } else {
            for g in 0..ld {
                u0[g] += pi[g] * fi[g];
            }
        }
        if mgga {
            let ut = &mut rest[3 * ld..4 * ld];
            for k in 0..3 {
                let dk = &phi4[i * s4 + (1 + k) * ld..i * s4 + (2 + k) * ld];
                let pki = &pk[k * pks + i * ps..k * pks + i * ps + ld];
                for g in 0..ld {
                    ut[g] += pki[g] * dk[g];
                }
            }
        }
    }
    if gga {
        let (u1, rest2) = rest.split_at_mut(ld);
        let (u2, rest3) = rest2.split_at_mut(ld);
        let u3 = &mut rest3[..ld];
        for m in 0..nao {
            let f = &ao[m * nc * ld..(m + 1) * nc * ld];
            let qm = &q[m * ld..(m + 1) * ld];
            let (p1, p2, p3) = (&f[ld..2 * ld], &f[2 * ld..3 * ld], &f[3 * ld..4 * ld]);
            for g in 0..ld {
                let qg = qm[g];
                u1[g] += p1[g] * qg;
                u2[g] += p2[g] * qg;
                u3[g] += p3[g] * qg;
            }
        }
    }
    for v in u[..4.min(nu) * ld].iter_mut() {
        *v *= 2.0;
    }
}
multiversion!(fn occ_u1(ao: &[f64], nc: usize, ld: usize, nao: usize, nocc: usize, gga: bool, mgga: bool, phi4: &[f64], p: &[f64], ps: usize, q: &[f64], pk: &[f64], pks: usize, u: &mut [f64]) -> () = occ_u1_body);

/// Potential-side products of one channel and one perturbation of the occupied kernel (its
/// streaming path's second half) from that channel's first-order potentials t (nu_s x ld):
/// Z = t_rho Phi + t_k d_k Phi, G = t_k d_k phi and (meta-GGA) T_k = 1/2 t_tau d_k Phi.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn occ_out_body(
    ao: &[f64],
    nc: usize,
    ld: usize,
    nao: usize,
    nocc: usize,
    gga: bool,
    mgga: bool,
    t: &[f64],
    phi4: &[f64],
    z: &mut [f64],
    zs: usize,
    pks: usize,
    gm: &mut [f64],
    tk: &mut [f64],
) {
    let s4 = 4 * ld;
    let t0 = &t[..ld];
    for i in 0..nocc {
        let fi = &phi4[i * s4..i * s4 + ld];
        let zi = &mut z[i * zs..i * zs + ld];
        if gga {
            let (t1, t2, t3) = (&t[ld..2 * ld], &t[2 * ld..3 * ld], &t[3 * ld..4 * ld]);
            let (d1, d2, d3) = (
                &phi4[i * s4 + ld..i * s4 + 2 * ld],
                &phi4[i * s4 + 2 * ld..i * s4 + 3 * ld],
                &phi4[i * s4 + 3 * ld..i * s4 + 4 * ld],
            );
            for g in 0..ld {
                zi[g] = t0[g] * fi[g] + t1[g] * d1[g] + t2[g] * d2[g] + t3[g] * d3[g];
            }
        } else {
            for g in 0..ld {
                zi[g] = t0[g] * fi[g];
            }
        }
        if mgga {
            let tt = &t[4 * ld..5 * ld];
            for k in 0..3 {
                let dk = &phi4[i * s4 + (1 + k) * ld..i * s4 + (2 + k) * ld];
                let o = &mut tk[k * pks + i * zs..k * pks + i * zs + ld];
                for g in 0..ld {
                    o[g] = 0.5 * tt[g] * dk[g];
                }
            }
        }
    }
    if gga {
        let (t1, t2, t3) = (&t[ld..2 * ld], &t[2 * ld..3 * ld], &t[3 * ld..4 * ld]);
        for m in 0..nao {
            let f = &ao[m * nc * ld..(m + 1) * nc * ld];
            let (p1, p2, p3) = (&f[ld..2 * ld], &f[2 * ld..3 * ld], &f[3 * ld..4 * ld]);
            let gmm = &mut gm[m * ld..(m + 1) * ld];
            for g in 0..ld {
                gmm[g] = t1[g] * p1[g] + t2[g] * p2[g] + t3[g] * p3[g];
            }
        }
    }
}
multiversion!(fn occ_out(ao: &[f64], nc: usize, ld: usize, nao: usize, nocc: usize, gga: bool, mgga: bool, t: &[f64], phi4: &[f64], z: &mut [f64], zs: usize, pks: usize, gm: &mut [f64], tk: &mut [f64]) -> () = occ_out_body);

/// The inputs of the register-blocked GGA perturbation of a batch ([`occ_gga`]): the kept
/// functions (function-major, `nc` components, stride `ld`), wF, and per channel its occupied
/// count, Phi4 = [Phi; grad Phi] (column stride 4 ld) and the perturbation's P = phi A and
/// Q = Phi A^T (empty for a channel without electrons).
pub(crate) struct GgaInputs<'a> {
    ao: &'a [f64],
    nc: usize,
    ld: usize,
    nao: usize,
    wf: &'a [f64],
    no: &'a [usize],
    phi4: &'a [&'a [f64]],
    p: &'a [&'a [f64]],
    q: &'a [&'a [f64]],
}

/// The per-channel inputs of [`GgaInputs`] as arrays of the channel count.
struct GgaChannels<'a, const NCH: usize> {
    no: &'a [usize; NCH],
    phi4: &'a [&'a [f64]; NCH],
    p: &'a [&'a [f64]; NCH],
    q: &'a [&'a [f64]; NCH],
}

/// GGA occupied-kernel perturbation for the points g0..g0 + CH, every channel: u1 (`NU` = 4 NCH
/// values per point) and t in registers while the orbital and function columns stream past; the
/// same operations in the same order as [`occ_u1_body`], [`apply_wf_body`] and [`occ_out_body`]
/// (same bits). Outputs per channel: Z (ld x nocc) and G (ld x nao).
#[inline(always)]
#[allow(clippy::needless_range_loop, reason = "indexes several arrays in step")]
fn occ_gga_chunk<const CH: usize, const NCH: usize, const NU: usize>(
    g0: usize,
    x: &GgaInputs<'_>,
    ch: &GgaChannels<'_, NCH>,
    z: &mut [&mut [f64]; NCH],
    gm: &mut [&mut [f64]; NCH],
) {
    let (ao, nc, ld, nao, wf) = (x.ao, x.nc, x.ld, x.nao, x.wf);
    let (no, phi4, p, q) = (ch.no, ch.phi4, ch.p, ch.q);
    let s4 = 4 * ld;
    let ld4 = |v: &[f64], o: usize| -> [f64; CH] { v[o..o + CH].try_into().unwrap() };
    // u and t are indexed directly (constant indices once the channel loops unroll), so that
    // they can stay in registers
    let mut u = [[0.0f64; CH]; NU];
    for s in 0..NCH {
        let c0 = 4 * s;
        for i in 0..no[s] {
            let pi = ld4(p[s], i * ld + g0);
            let b = i * s4 + g0;
            let (f, d1, d2, d3) = (
                ld4(phi4[s], b),
                ld4(phi4[s], b + ld),
                ld4(phi4[s], b + 2 * ld),
                ld4(phi4[s], b + 3 * ld),
            );
            for l in 0..CH {
                u[c0][l] += pi[l] * f[l];
                u[c0 + 1][l] += pi[l] * d1[l];
                u[c0 + 2][l] += pi[l] * d2[l];
                u[c0 + 3][l] += pi[l] * d3[l];
            }
        }
        if no[s] > 0 {
            for m in 0..nao {
                let qm = ld4(q[s], m * ld + g0);
                let b = m * nc * ld + g0;
                let (p1, p2, p3) = (ld4(ao, b + ld), ld4(ao, b + 2 * ld), ld4(ao, b + 3 * ld));
                for l in 0..CH {
                    u[c0 + 1][l] += p1[l] * qm[l];
                    u[c0 + 2][l] += p2[l] * qm[l];
                    u[c0 + 3][l] += p3[l] * qm[l];
                }
            }
        }
        for c in c0..c0 + 4 {
            for l in 0..CH {
                u[c][l] *= 2.0;
            }
        }
    }
    let mut t = [[0.0f64; CH]; NU];
    for c in 0..NU {
        let f0 = ld4(wf, (c * NU) * ld + g0);
        for l in 0..CH {
            t[c][l] = f0[l] * u[0][l];
        }
        for d in 1..NU {
            let fd = ld4(wf, (c * NU + d) * ld + g0);
            for l in 0..CH {
                t[c][l] += fd[l] * u[d][l];
            }
        }
    }
    for s in 0..NCH {
        if no[s] == 0 {
            continue;
        }
        let c0 = 4 * s;
        for i in 0..no[s] {
            let b = i * s4 + g0;
            let (f, d1, d2, d3) = (
                ld4(phi4[s], b),
                ld4(phi4[s], b + ld),
                ld4(phi4[s], b + 2 * ld),
                ld4(phi4[s], b + 3 * ld),
            );
            let zi = &mut z[s][i * ld + g0..i * ld + g0 + CH];
            for l in 0..CH {
                zi[l] = t[c0][l] * f[l]
                    + t[c0 + 1][l] * d1[l]
                    + t[c0 + 2][l] * d2[l]
                    + t[c0 + 3][l] * d3[l];
            }
        }
        for m in 0..nao {
            let b = m * nc * ld + g0;
            let (p1, p2, p3) = (ld4(ao, b + ld), ld4(ao, b + 2 * ld), ld4(ao, b + 3 * ld));
            let o = &mut gm[s][m * ld + g0..m * ld + g0 + CH];
            for l in 0..CH {
                o[l] = t[c0 + 1][l] * p1[l] + t[c0 + 2][l] * p2[l] + t[c0 + 3][l] * p3[l];
            }
        }
    }
}

/// [`occ_gga_chunk`] over all points of a batch (32 at a time, then 8), for `NCH` channels
/// (`NU` = 4 NCH; `z` and `gm` have `NCH` entries).
#[inline(always)]
fn occ_gga_body<const NCH: usize, const NU: usize>(
    x: &GgaInputs<'_>,
    z: &mut [&mut [f64]],
    gm: &mut [&mut [f64]],
) {
    let ch = GgaChannels::<NCH> {
        no: x.no.try_into().unwrap(),
        phi4: x.phi4.try_into().unwrap(),
        p: x.p.try_into().unwrap(),
        q: x.q.try_into().unwrap(),
    };
    let z: &mut [&mut [f64]; NCH] = z.try_into().unwrap();
    let gm: &mut [&mut [f64]; NCH] = gm.try_into().unwrap();
    let ld = x.ld;
    let mut g0 = 0;
    while g0 + 32 <= ld {
        occ_gga_chunk::<32, NCH, NU>(g0, x, &ch, z, gm);
        g0 += 32;
    }
    while g0 < ld {
        occ_gga_chunk::<8, NCH, NU>(g0, x, &ch, z, gm);
        g0 += 8;
    }
}

#[inline(always)]
fn occ_gga1_body(x: &GgaInputs<'_>, z: &mut [&mut [f64]], gm: &mut [&mut [f64]]) {
    occ_gga_body::<1, 4>(x, z, gm)
}
multiversion!(fn occ_gga1(x: &GgaInputs<'_>, z: &mut [&mut [f64]], gm: &mut [&mut [f64]]) -> () = occ_gga1_body);

#[inline(always)]
fn occ_gga2_body(x: &GgaInputs<'_>, z: &mut [&mut [f64]], gm: &mut [&mut [f64]]) {
    occ_gga_body::<2, 8>(x, z, gm)
}
multiversion!(fn occ_gga2(x: &GgaInputs<'_>, z: &mut [&mut [f64]], gm: &mut [&mut [f64]]) -> () = occ_gga2_body);

/// The register-blocked GGA perturbation of `NCH` channels ([`occ_gga_chunk`]): Z and G of
/// every channel (into `z`, `gm`).
fn occ_gga<const NCH: usize>(x: &GgaInputs<'_>, z: &mut [&mut [f64]], gm: &mut [&mut [f64]]) {
    match NCH {
        1 => occ_gga1(x, z, gm),
        2 => occ_gga2(x, z, gm),
        _ => unreachable!("{NCH} spin channels"),
    }
}

#[inline(always)]
fn apply_wf_any_body(nu: usize, ld: usize, wf: &[f64], u: &[f64], t: &mut [f64]) {
    apply_wf_body(nu, ld, wf, u, t)
}
multiversion!(fn apply_wf(nu: usize, ld: usize, wf: &[f64], u: &[f64], t: &mut [f64]) -> () = apply_wf_any_body);

/// The integrator's CPKS kernel ([`XcIntegrator`]'s own [`ResponseKernel`]).
impl ResponseKernel for &XcIntegrator {
    fn contract_occ(
        &self,
        c_o: &[MatRef<'_, f64>],
        a: &[Vec<Mat<f64>>],
    ) -> Result<Vec<Vec<Mat<f64>>>, LinusError> {
        <XcIntegrator as ResponseKernel>::contract_occ(self, c_o, a)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use faer::Par;
    use linus_base::linalg::mm;
    use linus_grid::{GridSpec, MolecularGrid};
    use linus_ints::basis::BasisDef;
    use linus_ints::BasisSet;

    /// Orbitals from the SAD density's eigenvectors (any orthonormal-ish C_o works for the
    /// identities tested here), a small molecule and a coarse grid.
    fn setup(func: &str, spec: GridSpec) -> (XcIntegrator, Mat<f64>, Vec<Mat<f64>>) {
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
        let (da, _) = linus_scf::guess::sad_guess(&def, &basis, &z, 9, 9, Par::Seq);
        let n = basis.nbf;
        let (_, u) = linus_base::linalg::eigh(da.as_ref(), Par::Seq);
        let nocc = 9;
        let co = Mat::from_fn(n, nocc, |i, j| u[(i, n - 1 - j)] * 0.8);
        let a: Vec<Mat<f64>> = (0..5)
            .map(|p| {
                Mat::from_fn(n, nocc, |i, j| {
                    1e-2 * ((((i * 7 + j * 13 + p * 5) % 17) as f64) - 8.0) / 8.0
                })
            })
            .collect();
        let g = MolecularGrid::build(&z, &xyz, &spec).unwrap();
        (XcIntegrator::new(&basis, g, func).unwrap(), co, a)
    }

    fn maxrel(a: &Mat<f64>, b: &Mat<f64>) -> f64 {
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

    /// The occupied-projected kernel against the reference implementation's full first-order
    /// potentials times C_o, LDA/GGA/meta-GGA; identical bits for 1 and 3 threads.
    #[test]
    fn kernels_match_reference() {
        for (func, spec) in [
            ("svwn", GridSpec::pyscf(1)),
            ("b3lyp", GridSpec::orca_defgrid_guess(1)),
            ("r2scan", GridSpec::pyscf(1)),
        ] {
            let (xc, co, a) = setup(func, spec);
            let d0 = mm(co.as_ref(), co.transpose(), Par::Seq) * 2.0;
            let d1: Vec<Mat<f64>> = a
                .iter()
                .map(|x| {
                    let t = mm(x.as_ref(), co.transpose(), Par::Seq);
                    &t + t.transpose()
                })
                .collect();
            let r = xc.kernel_contract_restricted_ref(&d0, &d1);
            let run = |t: usize| {
                rayon::ThreadPoolBuilder::new()
                    .num_threads(t)
                    .build()
                    .unwrap()
                    .install(|| {
                        xc.kernel_contract_occ_ch::<1>([co.as_ref()], [&a])
                            .remove(0)
                    })
            };
            let o1 = run(3);
            for p in 0..d1.len() {
                let ro = mm(r[p].as_ref(), co.as_ref(), Par::Seq);
                let e = maxrel(&o1[p], &ro);
                assert!(e <= 1e-12, "{func} occ p={p}: {e:e}");
            }
            assert!(run(1) == o1, "{func}: thread count changes the kernel");
        }
    }

    /// Open-shell occupied kernel: against the reference open-shell kernel (LDA, GGA) and
    /// central differences of the spin potentials (all, incl. meta-GGA); identical bits for 1
    /// and 3 threads; an empty beta spin.
    #[test]
    fn uks_occ_kernel() {
        for (func, spec) in [
            ("svwn", GridSpec::pyscf(1)),
            ("pbe", GridSpec::orca_defgrid_guess(1)),
            ("r2scan", GridSpec::pyscf(1)),
        ] {
            let (xc, co, a) = setup(func, spec);
            let n = co.nrows();
            // alpha: 9 orbitals; beta: 7 different ones
            let cb = Mat::from_fn(n, 7, |i, j| co[(i, j)] * 0.9 + 0.05 * co[(i, (j + 1) % 9)]);
            let ab: Vec<Mat<f64>> = a
                .iter()
                .map(|x| Mat::from_fn(n, 7, |i, j| x[(i, j + 2)] * 1.3))
                .collect();
            let run = |t: usize| {
                rayon::ThreadPoolBuilder::new()
                    .num_threads(t)
                    .build()
                    .unwrap()
                    .install(|| {
                        xc.kernel_contract_occ_ch::<2>([co.as_ref(), cb.as_ref()], [&a, &ab])
                    })
            };
            let r3 = run(3);
            assert!(
                run(1) == r3,
                "{func}: thread count changes the open-shell kernel"
            );
            let da = mm(co.as_ref(), co.transpose(), Par::Seq);
            let db = mm(cb.as_ref(), cb.transpose(), Par::Seq);
            for p in [0, 3] {
                let d1 = |x: &Mat<f64>, c: &Mat<f64>| {
                    let t = mm(x.as_ref(), c.transpose(), Par::Seq);
                    &t + t.transpose()
                };
                let (d1a, d1b) = (d1(&a[p], &co), d1(&ab[p], &cb));
                let h = 1e-4;
                let at = |sg: f64| xc.potential(&(&da + &d1a * sg), &(&db + &d1b * sg), false);
                let (vp, vm) = (at(h), at(-h));
                let fa = mm(
                    ((&vp.1 - &vm.1) * (0.5 / h)).as_ref(),
                    co.as_ref(),
                    Par::Seq,
                );
                let fb = mm(
                    ((&vp.2 - &vm.2) * (0.5 / h)).as_ref(),
                    cb.as_ref(),
                    Par::Seq,
                );
                let (ea, eb) = (maxrel(&r3[0][p], &fa), maxrel(&r3[1][p], &fb));
                assert!(
                    ea < 1e-6 && eb < 1e-6,
                    "{func} p={p}: open-shell kernel vs FD {ea:e} {eb:e}"
                );
                if func != "r2scan" {
                    let r = xc.kernel_contract_ref(&da, &db, false, &[(d1a.clone(), d1b.clone())]);
                    let (ea, eb) = (
                        maxrel(&r3[0][p], &mm(r[0].0.as_ref(), co.as_ref(), Par::Seq)),
                        maxrel(&r3[1][p], &mm(r[0].1.as_ref(), cb.as_ref(), Par::Seq)),
                    );
                    assert!(
                        ea < 1e-12 && eb < 1e-12,
                        "{func} p={p}: open-shell kernel vs reference {ea:e} {eb:e}"
                    );
                }
            }
            // no beta electrons: alpha part only (not for meta-GGAs, whose libxc second
            // derivatives are not usable at fully spin-polarized points)
            if func == "r2scan" {
                continue;
            }
            let e0 = Mat::<f64>::zeros(n, 0);
            let ab0: Vec<Mat<f64>> = a.iter().map(|_| Mat::zeros(n, 0)).collect();
            let r0 = xc.kernel_contract_occ_ch::<2>([co.as_ref(), e0.as_ref()], [&a, &ab0]);
            let at = |sg: f64| {
                xc.potential(
                    &(&da
                        + &{
                            let t = mm(a[1].as_ref(), co.transpose(), Par::Seq);
                            &t + t.transpose()
                        } * sg),
                    &Mat::zeros(n, n),
                    false,
                )
            };
            let h = 1e-4;
            let (vp, vm) = (at(h), at(-h));
            let fa = mm(
                ((&vp.1 - &vm.1) * (0.5 / h)).as_ref(),
                co.as_ref(),
                Par::Seq,
            );
            assert!(
                r0[1].iter().all(|m| m.ncols() == 0) && maxrel(&r0[0][1], &fa) < 1e-6,
                "{func}: empty beta spin {:e}",
                maxrel(&r0[0][1], &fa)
            );
        }
    }
}
