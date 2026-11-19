//! Nuclear gradient of the split-RI-J term E3 = sum_k d_k sum_mn (k|mn) D_mn, integral-direct
//! in Hermite space (no three-index storage), on the lane kernels of [`crate::eri3c`].
//!
//! With the fitted coefficients of a lane batch in its Cartesian-Hermite basis (dc, see
//! [`crate::eri3c::chunk_coefficients`]), the Hermite potential of a primitive pair ip at the
//! raised degree,
//!
//!   Jh\[t\] = sum_c dc\[c\] R_{c+t}(P_ip - C),   |t| <= lab + 1   (per lane),
//!
//! gives all three centre derivatives of the pair's contribution:
//!
//!   dE3/dA_k = sum_ip sum_t Jh\[t\] DhA_ip\[k\]\[t\],   DhA_ip\[k\]\[t\] = sum_mn EA_ip\[t\]\[k\]\[mn\] D_mn
//!   dE3/dC_x = -sum_ip sum_t' Jh[t' + e_x] Dh_ip[t']           (the auxiliary centre, per lane)
//!   dE3/dB   = -(dE3/dA + dE3/dC)                              (translational invariance)
//!
//! where Dh is the pair's Hermite density ([`crate::eri3c::pair_hermite_density`]) and EA the
//! Hermite expansion of the derivative of the product with respect to the centre A of the
//! pair's first shell ([`crate::core::deriv::PairDeriv::ea`]). Per primitive pair and lane batch the
//! work is one R recursion one degree higher than the energy's, an ncart(l) x nherm(lab+1)
//! contraction and two short dot products.

use crate::basis::ncart;
use crate::core::deriv::PairDeriv;
use crate::core::hermite::{herm_idx, herm_tuv, nherm};
use crate::eri3c::{
    boys_table, htab, prim_hermite, r_prog, AuxChunk, Lanes, Pair3c, Work3c, W, ZERO,
};
use std::sync::OnceLock;

/// shift[x][t'] = herm_idx(t' + e_x) for the Hermite indices t' of degree <= lab.
fn shift_tab(lab: usize) -> &'static [Vec<u16>; 3] {
    static T: OnceLock<Vec<[Vec<u16>; 3]>> = OnceLock::new();
    &T.get_or_init(|| {
        let tuv = herm_tuv();
        (0..crate::eri3c::MAX_L3C)
            .map(|l| {
                let nh = nherm(l);
                let mk = |x: usize| -> Vec<u16> {
                    (0..nh)
                        .map(|t| {
                            let mut c =
                                [tuv[t][0] as usize, tuv[t][1] as usize, tuv[t][2] as usize];
                            c[x] += 1;
                            herm_idx(c[0], c[1], c[2]) as u16
                        })
                        .collect()
                };
                [mk(0), mk(1), mk(2)]
            })
            .collect()
    })[lab]
}

/// Hermite densities of the derivative products: dha[ip * 3 * nh1 + k * nh1 + t] =
/// sum_mn EA_ip\[t\]\[k\]\[mn\] dw\[mn\] (nh1 = nherm(lab + 1)); `pd` must belong to the same shell
/// pair (same primitive order) as the [`Pair3c`].
pub fn pair_hermite_density_deriv(pd: &PairDeriv, dw: &[f64], dha: &mut [f64]) {
    let nh1 = nherm(pd.lab() + 1);
    let nab = pd.nab();
    for ip in 0..pd.nprim {
        let ea = pd.ea_prim(ip);
        let o = &mut dha[ip * 3 * nh1..(ip + 1) * 3 * nh1];
        for t in 0..nh1 {
            for k in 0..3 {
                let row = &ea[(t * 3 + k) * nab..(t * 3 + k + 1) * nab];
                let mut s = 0.0;
                for (&e, &d) in row.iter().zip(&dw[..nab]) {
                    s += e * d;
                }
                o[k * nh1 + t] = s;
            }
        }
    }
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn pair_jgrad_impl(
    pair: &Pair3c,
    chunks: &[AuxChunk],
    dc: &[Vec<Lanes>],
    cmax: &[f64],
    dh: &[f64],
    dha: &[f64],
    gaux: &mut [[f64; 3]],
    thresh: f64,
    w: &mut Work3c,
) -> ([f64; 3], [f64; 3]) {
    let tab = boys_table();
    let prog = r_prog();
    let lab = pair.lab;
    let nh = nherm(lab);
    let nh1 = nherm(lab + 1);
    let nprim = pair.nprim();
    let qpmax = pair.qprim.iter().cloned().fold(0.0, f64::max);
    let sh = shift_tab(lab);
    let Work3c {
        fm, rn, r, rw, acc, ..
    } = w;
    if acc.len() < nh1 {
        acc.resize(nh1, ZERO);
    }
    let mut ga = [ZERO; 3];
    let mut gcs = [0.0f64; 3];
    for (ci, ch) in chunks.iter().enumerate() {
        if qpmax * ch.q < thresh || cmax[ci] == 0.0 {
            continue;
        }
        let l = ch.l;
        let lt = l + lab + 1;
        let ht = htab(l, lab + 1);
        let nc = ncart(l);
        let dcc = &dc[ci];
        let mut gc = [ZERO; 3];
        for ip in 0..nprim {
            if pair.qprim[ip] * ch.q < thresh {
                continue;
            }
            prim_hermite(
                pair.p[ip],
                pair.pc[ip],
                ch,
                lt,
                0.0,
                tab,
                prog,
                fm,
                rn,
                r,
                rw,
            );
            let jl = &mut acc[..nh1];
            jl.fill(ZERO);
            for (c, dcv) in dcc[..nc].iter().enumerate() {
                let ix = &ht.idx[c * nh1..(c + 1) * nh1];
                for (jv, &i) in jl.iter_mut().zip(ix) {
                    let rv = &r[i as usize];
                    for q in 0..W {
                        jv[q] += rv[q] * dcv[q];
                    }
                }
            }
            let da = &dha[ip * 3 * nh1..(ip + 1) * 3 * nh1];
            for k in 0..3 {
                let mut s = ga[k];
                for (jv, &dv) in jl.iter().zip(&da[k * nh1..(k + 1) * nh1]) {
                    for q in 0..W {
                        s[q] += jv[q] * dv;
                    }
                }
                ga[k] = s;
            }
            let d = &dh[ip * nh..(ip + 1) * nh];
            for x in 0..3 {
                let mut s = gc[x];
                for (&i, &dv) in sh[x].iter().zip(d) {
                    let jv = &jl[i as usize];
                    for q in 0..W {
                        s[q] -= jv[q] * dv;
                    }
                }
                gc[x] = s;
            }
        }
        for q in 0..W {
            if ch.col[q] == u32::MAX {
                continue;
            }
            let g = &mut gaux[ch.col[q] as usize];
            for x in 0..3 {
                g[x] += gc[x][q];
                gcs[x] += gc[x][q];
            }
        }
    }
    let mut gat = [0.0; 3];
    for k in 0..3 {
        gat[k] = ga[k].iter().sum();
    }
    (gat, gcs)
}

linus_num::multiversion!(
    /// Gradient of one pair's split-RI-J term sum_k d_k sum_mn (k|mn) D_mn: returns (dE/dA, sum
    /// of dE/dC over the auxiliary lanes), A the centre of the pair's first shell (dE/dB = -(both));
    /// the auxiliary-centre derivatives are accumulated per lane batch column into
    /// gaux[col(lane)] (the first function of the lane's shell, relative to the chunks' range).
    /// `dh`: [`crate::eri3c::pair_hermite_density`]; `dha`: [`pair_hermite_density_deriv`] (both
    /// of the same weighted density rows); `dc`, `cmax` as for [`crate::eri3c::pair_jh`].
    pub fn pair_jgrad(pair: &Pair3c, chunks: &[AuxChunk], dc: &[Vec<Lanes>], cmax: &[f64], dh: &[f64], dha: &[f64], gaux: &mut [[f64; 3]], thresh: f64, w: &mut Work3c) -> ([f64; 3], [f64; 3]) = pair_jgrad_impl);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::basis::{BasisDef, BasisSet};
    use crate::core::hermite::{build_pair, ShellView};
    use crate::eri3c::{chunk_coefficients, pair_gamma, pair_hermite_density, Aux3c};

    fn bases(xyz: &[[f64; 3]]) -> (BasisSet, BasisSet) {
        let json = r#"{"name":"t","elements":{"8":{"electron_shells":[
          {"function_type":"gto","angular_momentum":[0],"exponents":["130.7","23.81","6.44"],"coefficients":[["0.154","0.535","0.444"]]},
          {"function_type":"gto","angular_momentum":[1],"exponents":["5.03","1.17","0.38"],"coefficients":[["0.156","0.607","0.392"]]},
          {"function_type":"gto","angular_momentum":[2],"exponents":["1.2","0.4"],"coefficients":[["0.6","0.5"]]},
          {"function_type":"gto","angular_momentum":[3],"exponents":["0.9"],"coefficients":[["1.0"]]}]},
          "1":{"electron_shells":[
          {"function_type":"gto","angular_momentum":[0],"exponents":["3.42","0.62"],"coefficients":[["0.15","0.53"]]},
          {"function_type":"gto","angular_momentum":[1],"exponents":["0.8"],"coefficients":[["1.0"]]},
          {"function_type":"gto","angular_momentum":[2],"exponents":["0.6"],"coefficients":[["1.0"]]}]}}}"#;
        let ajson = r#"{"name":"a","elements":{"8":{"electron_shells":[
          {"function_type":"gto","angular_momentum":[0],"exponents":["20.0","4.0"],"coefficients":[["0.4","0.7"]]},
          {"function_type":"gto","angular_momentum":[1],"exponents":["1.5"],"coefficients":[["1.0"]]},
          {"function_type":"gto","angular_momentum":[2],"exponents":["1.0"],"coefficients":[["1.0"]]},
          {"function_type":"gto","angular_momentum":[3],"exponents":["0.8"],"coefficients":[["1.0"]]},
          {"function_type":"gto","angular_momentum":[4],"exponents":["0.7"],"coefficients":[["1.0"]]}]},
          "1":{"electron_shells":[{"function_type":"gto","angular_momentum":[0],"exponents":["1.0"],"coefficients":[["1.0"]]},
          {"function_type":"gto","angular_momentum":[2],"exponents":["0.8"],"coefficients":[["1.0"]]}]}}}"#;
        let b = BasisSet::new(
            &BasisDef::from_bse_json(json).unwrap(),
            &[8, 1, 1],
            xyz,
            true,
        )
        .unwrap();
        let a = BasisSet::new(
            &BasisDef::from_bse_json(ajson).unwrap(),
            &[8, 1, 1],
            xyz,
            true,
        )
        .unwrap();
        (b, a)
    }

    /// E3 = sum_k d_k sum_mn (k|mn) D_mn for fixed d and D, through the energy kernels.
    fn e3(xyz: &[[f64; 3]], d: &[f64], dm: &dyn Fn(usize, usize) -> f64) -> f64 {
        let (b, a) = bases(xyz);
        let aux = Aux3c::new(&a);
        let chunks = aux.chunks(0..a.shells.len());
        let mut gamma = vec![0.0; a.nbf];
        let mut w = Work3c::new();
        for i in 0..b.shells.len() {
            for j in 0..=i {
                let sp = build_pair(
                    &ShellView::from_shell(&b.shells[i]),
                    &ShellView::from_shell(&b.shells[j]),
                    i,
                    j,
                );
                let p = Pair3c::from_shell_pair(&sp);
                let (a0, b0) = (b.shells[i].ao_start, b.shells[j].ao_start);
                let dw: Vec<f64> = (0..p.nab())
                    .map(|x| {
                        let (m, n) = (a0 + x / p.nb, b0 + x % p.nb);
                        if i == j {
                            dm(m, n)
                        } else {
                            dm(m, n) + dm(n, m)
                        }
                    })
                    .collect();
                let mut dh = vec![0.0; p.nprim() * nherm(p.lab)];
                pair_hermite_density(&p, &dw, &mut dh);
                pair_gamma(&p, &chunks, &dh, &mut gamma, 0.0, &mut w);
            }
        }
        gamma.iter().zip(d).map(|(g, c)| g * c).sum()
    }

    #[test]
    fn split_rij_gradient_matches_finite_differences() {
        let xyz = [[0.1, -0.2, 0.3], [1.2, 0.9, -0.8], [-0.9, 1.1, 0.5]];
        let (b, a) = bases(&xyz);
        let n = b.nbf;
        let d: Vec<f64> = (0..a.nbf).map(|k| ((k as f64) * 0.71).sin()).collect();
        let dm = |m: usize, nn: usize| {
            (((m + 1) * (nn + 1)) as f64 * 0.37).cos() * 0.3
                + if m == nn { 1.0 } else { 0.0 }
                + 0.1 * ((m + nn) as f64).sin()
        };
        let natom = xyz.len();
        // analytic
        let aux = Aux3c::new(&a);
        let chunks = aux.chunks(0..a.shells.len());
        let dc: Vec<Vec<Lanes>> = chunks
            .iter()
            .map(|ch| {
                let mut v = Vec::new();
                chunk_coefficients(ch, &d, &mut v);
                v
            })
            .collect();
        let cmax: Vec<f64> = dc
            .iter()
            .map(|v| {
                v.iter()
                    .flat_map(|l| l.iter())
                    .fold(0.0f64, |m, &x| m.max(x.abs()))
            })
            .collect();
        let mut g = vec![[0.0; 3]; natom];
        let mut gaux = vec![[0.0; 3]; a.nbf];
        let mut w = Work3c::new();
        for i in 0..b.shells.len() {
            for j in 0..=i {
                let (va, vb) = (
                    ShellView::from_shell(&b.shells[i]),
                    ShellView::from_shell(&b.shells[j]),
                );
                let (sp, pd) = crate::core::pair_data::build(
                    &va,
                    &vb,
                    i,
                    j,
                    1,
                    crate::core::pair_data::PRIM_PAIR_CUT,
                );
                let pd = pd.unwrap();
                let p = Pair3c::from_shell_pair(&sp);
                let (a0, b0) = (b.shells[i].ao_start, b.shells[j].ao_start);
                let dw: Vec<f64> = (0..p.nab())
                    .map(|x| {
                        let (m, nn) = (a0 + x / p.nb, b0 + x % p.nb);
                        if i == j {
                            dm(m, nn)
                        } else {
                            dm(m, nn) + dm(nn, m)
                        }
                    })
                    .collect();
                let mut dh = vec![0.0; p.nprim() * nherm(p.lab)];
                pair_hermite_density(&p, &dw, &mut dh);
                let mut dha = vec![0.0; p.nprim() * 3 * nherm(p.lab + 1)];
                pair_hermite_density_deriv(&pd, &dw, &mut dha);
                let (g_a, g_c) =
                    pair_jgrad(&p, &chunks, &dc, &cmax, &dh, &dha, &mut gaux, 0.0, &mut w);
                for k in 0..3 {
                    g[b.shells[i].atom][k] += g_a[k];
                    g[b.shells[j].atom][k] -= g_a[k] + g_c[k];
                }
            }
        }
        for s in &a.shells {
            for k in 0..3 {
                g[s.atom][k] += gaux[s.ao_start][k];
            }
        }
        let _ = n;
        let h = 1e-4;
        let mut worst: f64 = 0.0;
        for at in 0..natom {
            for k in 0..3 {
                let f = |s: f64| {
                    let mut c = xyz;
                    c[at][k] += s;
                    e3(&c, &d, &dm)
                };
                let fd = (8.0 * (f(h) - f(-h)) - (f(2.0 * h) - f(-2.0 * h))) / (12.0 * h);
                let e = (g[at][k] - fd).abs();
                eprintln!(
                    "atom {at} dir {k}: analytic {:.12} fd {:.12} diff {e:.2e}",
                    g[at][k], fd
                );
                worst = worst.max(e / fd.abs().max(1.0));
            }
        }
        assert!(worst < 1e-8, "worst {worst:e}");
    }
}
