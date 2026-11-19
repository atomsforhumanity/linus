//! Gamma-contracted three-center gradients and Hessians, sum_{P,mn} ∂(P|mn)/∂X Γ_Pmn and
//! sum_{P,mn} ∂²(P|mn)/∂X∂Y Γ_Pmn, through the generic quartet kernels of
//! [`crate::eri4c::deriv`] (3-center integrals as 4-center ones with a unit s function). The
//! production kernels are [`crate::eri3c::dgrad`] and [`crate::eri3c::dblock`].

use crate::basis::BasisSet;
use crate::core::hermite::nherm;
use crate::core::util::{add_grads_into, shell_pair_dmax};
use crate::eri4c::deriv::{
    add_hess_blocks, pair_herm, quartet_grad, quartet_hess, DerivWork, HermDens, JTerm, PairHerm,
    CENTER_A, CENTER_C, CENTER_D,
};
use crate::operator::Operator;
use crate::pairs::{DerivOrder, PairSet};

/// Three-center density terms for sum_{P,mn} ∂(P|mn)/∂X Γ_Pmn with
/// Γ_Pmn = full\[P\]\[m\]\[n\] + sum_t c_t d_t\[P\] D_t\[m\]\[n\].
/// `full` covers the auxiliary functions of shells `aux_shells` (rows p0..p1 where p0 is the
/// first function of the first shell), each row nbf x nbf symmetric. Batches of auxiliary
/// shells can be contracted separately and the gradients summed.
pub struct Dens3c<'a> {
    pub full: Option<&'a [f64]>,
    pub aux_shells: std::ops::Range<usize>,
    pub separable: Vec<(f64, &'a [f64], &'a [f64])>,
}

/// Gradient sum_{P,mn} ∂(P|mn)/∂X Γ_Pmn (natom x 3; auxiliary shells are attached to atoms by
/// their `atom` field, so `aux` must be placed on the same atoms as `basis`).
/// For the two-electron operator `op` (e.g. erf(omega r)/r: ∂(P|erf|mn)).
#[allow(clippy::too_many_arguments)]
pub fn eri_grad_3c(
    basis: &BasisSet,
    pairs: &PairSet,
    aux: &BasisSet,
    apairs: &PairSet,
    dens: &Dens3c,
    thresh: f64,
    op: &Operator,
) -> Vec<[f64; 3]> {
    let screen = Grad3cScreen::new(pairs, apairs);
    eri_grad_3c_with(basis, pairs, aux, apairs, dens, thresh, op, &screen)
}

/// Schwarz-type bounds of the 3-center gradient contraction, computed once for repeated
/// [`eri_grad_3c_with`] calls (e.g. one per block of auxiliary functions).
pub struct Grad3cScreen {
    qp: Vec<f64>,
    qmn: Vec<f64>,
    qpd: Vec<f64>,
    qmnd: Vec<f64>,
}

impl Grad3cScreen {
    pub fn new(pairs: &PairSet, apairs: &PairSet) -> Self {
        let qp = apairs.schwarz().to_vec();
        let qmn = pairs.schwarz().to_vec();
        let qpd = apairs.deriv_bounds().0.to_vec();
        let qmnd = pairs.deriv_bounds().0.to_vec();
        Grad3cScreen { qp, qmn, qpd, qmnd }
    }
}

/// [`eri_grad_3c`] with precomputed screening data.
#[allow(clippy::too_many_arguments)]
#[allow(clippy::type_complexity, reason = "a one-off tuple of buffers")]
pub fn eri_grad_3c_with(
    basis: &BasisSet,
    pairs: &PairSet,
    aux: &BasisSet,
    apairs: &PairSet,
    dens: &Dens3c,
    thresh: f64,
    op: &Operator,
    screen: &Grad3cScreen,
) -> Vec<[f64; 3]> {
    let thresh = op.screen_threshold(thresh);
    let natom = basis.natom().max(aux.natom());
    let sh = &basis.shells;
    let ash = &aux.shells;
    let nbf = basis.nbf;
    let n2 = nbf * nbf;
    let Grad3cScreen { qp, qmn, qpd, qmnd } = screen;
    // separable terms: Hermite densities of aux pairs (d_P) and AO pairs (D_mn)
    let sep_aux: Vec<(Vec<Vec<f64>>, Vec<Vec<f64>>)> = dens
        .separable
        .iter()
        .map(|&(_, dp, _)| {
            apairs
                .pairs()
                .iter()
                .zip(apairs.derivs())
                .map(|(pr, pd)| {
                    let lab = pr.lab();
                    let (nh0, nh1) = (nherm(lab), nherm(lab + 1));
                    let p0 = ash[pr.sa].ao_start;
                    let mut e = vec![0.0; pr.nprim * nh0];
                    let mut ea = vec![0.0; pr.nprim * 3 * nh1];
                    for ip in 0..pr.nprim {
                        crate::one_e::deriv::herm_dens(
                            pr.e_prim(ip),
                            nh0,
                            1,
                            dp,
                            1,
                            p0,
                            0,
                            pr.na,
                            1,
                            1.0,
                            &mut e[ip * nh0..(ip + 1) * nh0],
                        );
                        crate::one_e::deriv::herm_dens(
                            pd.ea_prim(ip),
                            nh1,
                            3,
                            dp,
                            1,
                            p0,
                            0,
                            pr.na,
                            1,
                            1.0,
                            &mut ea[ip * 3 * nh1..(ip + 1) * 3 * nh1],
                        );
                    }
                    (e, ea)
                })
                .unzip()
        })
        .collect();
    let sep_ao: Vec<PairHerm> = dens
        .separable
        .iter()
        .map(|&(_, _, d)| pair_herm(basis, pairs, d))
        .collect();
    let sep_dmax: Vec<(Vec<f64>, Vec<f64>)> = dens
        .separable
        .iter()
        .map(|&(_, dp, d)| {
            let pm: Vec<f64> = ash
                .iter()
                .map(|s| {
                    (s.ao_start..s.ao_start + s.nfunc())
                        .map(|i| dp[i].abs())
                        .fold(0.0, f64::max)
                })
                .collect();
            (pm, shell_pair_dmax(basis, d))
        })
        .collect();
    let ns = sh.len();
    let p0 = dens
        .aux_shells
        .clone()
        .next()
        .map(|s| ash[s].ao_start)
        .unwrap_or(0);
    // deterministic reduction (independent of the thread count)
    let nas = apairs.len();
    let (_, g, _) = linus_num::det::par_fold_det(
        nas,
        linus_num::det::nchunk_for(nas, 3 * natom, linus_num::det::DEFAULT_BUDGET),
        || {
            (
                DerivWork::with_op(*op),
                vec![[0.0; 3]; natom],
                Vec::<f64>::new(),
            )
        },
        |(w, g, gbuf), ps| {
            let bra = apairs.pair(ps);
            let in_full = dens.full.is_some() && dens.aux_shells.contains(&ps);
            if bra.nprim == 0 || (!in_full && dens.separable.is_empty()) {
                return;
            }
            let bd = apairs.deriv(ps);
            let pa = ash[ps].ao_start;
            let np = bra.na;
            for (kl, ket) in pairs.pairs().iter().enumerate() {
                if ket.nprim == 0 {
                    continue;
                }
                let qq = qpd[ps] * qmn[kl] + qp[ps] * qmnd[kl];
                if qq < thresh {
                    continue;
                }
                let (s3, s4) = (ket.sa, ket.sb);
                let f = if s3 == s4 { 1.0 } else { 2.0 };
                let (c0, d0) = (sh[s3].ao_start, sh[s4].ao_start);
                let ncd = ket.nab();
                let gamma = if in_full {
                    let full = dens.full.unwrap();
                    gbuf.resize(np * ncd, 0.0);
                    let mut gm: f64 = 0.0;
                    for x in 0..np {
                        let row = &full[(pa + x - p0) * n2..(pa + x - p0 + 1) * n2];
                        for c in 0..ket.na {
                            for d in 0..ket.nb {
                                let v = 0.5
                                    * f
                                    * (row[(c0 + c) * nbf + d0 + d] + row[(d0 + d) * nbf + c0 + c]);
                                gbuf[x * ncd + c * ket.nb + d] = v;
                                gm = gm.max(v.abs());
                            }
                        }
                    }
                    if qq * gm >= thresh {
                        Some(&gbuf[..])
                    } else {
                        None
                    }
                } else {
                    None
                };
                let jterms: Vec<JTerm> = dens
                    .separable
                    .iter()
                    .enumerate()
                    .filter(|(t, &(c, _, _))| {
                        qq * c.abs() * sep_dmax[*t].0[ps] * sep_dmax[*t].1[s3 * ns + s4] * f
                            >= thresh
                    })
                    .map(|(t, &(c, _, _))| JTerm {
                        coef: c,
                        bra: HermDens {
                            e: &sep_aux[t].0[ps],
                            ea: &sep_aux[t].1[ps],
                        },
                        ket: HermDens {
                            e: &sep_ao[t].e[kl],
                            ea: &sep_ao[t].ea[kl],
                        },
                    })
                    .collect();
                if gamma.is_none() && jterms.is_empty() {
                    continue;
                }
                let gq = quartet_grad(bra, bd, ket, pairs.deriv(kl), gamma, &jterms, w);
                for (x, at) in [
                    (CENTER_A, ash[ps].atom),
                    (CENTER_C, sh[s3].atom),
                    (CENTER_D, sh[s4].atom),
                ] {
                    for k in 0..3 {
                        g[at][k] += gq[x][k];
                    }
                }
            }
        },
        |acc, b| add_grads_into(&mut acc.1, &b.1),
    );
    g
}

/// Hessian sum_{P,mn} ∂²(P|mn)/∂X∂Y Γ_Pmn, (3 natom)^2. Needs order-2 pair data for both
/// the AO pairs and the auxiliary pairs.
/// For the two-electron operator `op` (∂²(P|op|mn)).
#[allow(clippy::too_many_arguments)]
pub fn eri_hess_3c(
    basis: &BasisSet,
    pairs: &PairSet,
    aux: &BasisSet,
    apairs: &PairSet,
    dens: &Dens3c,
    thresh: f64,
    op: &Operator,
) -> Vec<f64> {
    assert!(pairs.deriv_order() >= DerivOrder::Two);
    let thresh = op.screen_threshold(thresh);
    let natom = basis.natom().max(aux.natom());
    let n3 = 3 * natom;
    let sh = &basis.shells;
    let ash = &aux.shells;
    let nbf = basis.nbf;
    let n2 = nbf * nbf;
    let qp = apairs.schwarz();
    let qmn = pairs.schwarz();
    let (qpd, qpd2) = apairs.deriv_bounds();
    let (qmnd, qmnd2) = pairs.deriv_bounds();
    let p0 = dens
        .aux_shells
        .clone()
        .next()
        .map(|s| ash[s].ao_start)
        .unwrap_or(0);
    // deterministic reduction (independent of the thread count)
    let nas = apairs.len();
    let (_, h, _) = linus_num::det::par_fold_det(
        nas,
        linus_num::det::nchunk_for(nas, n3 * n3, linus_num::det::DEFAULT_BUDGET),
        || {
            (
                DerivWork::with_op(*op),
                vec![0.0; n3 * n3],
                Vec::<f64>::new(),
            )
        },
        |(w, h, gbuf), ps| {
            let bra = apairs.pair(ps);
            let in_full = dens.full.is_some() && dens.aux_shells.contains(&ps);
            if bra.nprim == 0 || (!in_full && dens.separable.is_empty()) {
                return;
            }
            let pa = ash[ps].ao_start;
            let np = bra.na;
            for (kl, ket) in pairs.pairs().iter().enumerate() {
                let qq = qpd2[ps] * qmn[kl] + 2.0 * qpd[ps] * qmnd[kl] + qp[ps] * qmnd2[kl];
                if ket.nprim == 0 || qq < thresh {
                    continue;
                }
                let (s3, s4) = (ket.sa, ket.sb);
                let f = if s3 == s4 { 1.0 } else { 2.0 };
                let (c0, d0) = (sh[s3].ao_start, sh[s4].ao_start);
                let ncd = ket.nab();
                gbuf.resize(np * ncd, 0.0);
                gbuf.fill(0.0);
                let mut gm: f64 = 0.0;
                for x in 0..np {
                    for c in 0..ket.na {
                        for d in 0..ket.nb {
                            let (m, n) = (c0 + c, d0 + d);
                            let mut v = 0.0;
                            if in_full {
                                let row =
                                    &dens.full.unwrap()[(pa + x - p0) * n2..(pa + x - p0 + 1) * n2];
                                v += 0.5 * (row[m * nbf + n] + row[n * nbf + m]);
                            }
                            for &(cf, dp, dm) in &dens.separable {
                                v += cf * dp[pa + x] * dm[m * nbf + n];
                            }
                            v *= f;
                            gbuf[x * ncd + c * ket.nb + d] = v;
                            gm = gm.max(v.abs());
                        }
                    }
                }
                if qq * gm < thresh {
                    continue;
                }
                let hp = quartet_hess(bra, apairs.deriv(ps), ket, pairs.deriv(kl), gbuf, w);
                let atoms = [ash[ps].atom, ash[ps].atom, sh[s3].atom, sh[s4].atom];
                add_hess_blocks(h, n3, atoms, &hp.center_blocks());
            }
        },
        |acc, b| linus_num::det::add_into(&mut acc.1, &b.1),
    );
    h
}
