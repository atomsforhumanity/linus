//! The device gradients against dense CPU references built from linus-ints, linus-grid and
//! linus-xc: the DF two-electron term (J, global-hybrid K, range-separated K in the wK scheme,
//! closed and open shells) against
//! linus-ints' three- and two-center derivative contractions, the XC term (LDA, GGA,
//! meta-GGA, closed and open shells, with and without the grid response, Becke and SSF
//! partitions) against a quadrature over every function at every point; and run to run.
#![cfg(feature = "gpu")]

mod common;

use common::Rng;
use linus_gpu::cpu::{
    eval_ao, BasisSet, DfThresholds, Functional, GridSpec, MolecularGrid, Operator, XcInput,
    XcOutput,
};
use linus_gpu::grad::df::{DerivPlanes, GpuDfGrad, GpuDfGradOptions};
use linus_gpu::grad::xc::{GpuXcGrad, GpuXcGradOptions};
use linus_gpu::grad::{add_atoms, bitwise_equal, max_abs_diff};
use linus_gpu::harness::{self, Structure};
use linus_gpu::xc::XcKind;
use linus_ints::eri4c;
use linus_ints::{DerivOrder, PairOptions, PairSet};

/// A water molecule a little off its symmetric geometry (every gradient component nonzero).
fn water() -> Structure {
    let a = 1.0 / linus_core::units::BOHR_ANGSTROM;
    Structure::new(
        vec![8, 1, 1],
        vec![
            [0.10 * a, -0.05 * a, 0.12 * a],
            [0.85 * a, 0.52 * a, -0.03 * a],
            [-0.68 * a, 0.55 * a, 0.05 * a],
        ],
        0,
        1,
    )
    .unwrap()
}

/// The first `n` atoms of u1an39.
fn fragment(n: usize) -> Structure {
    let mut m = harness::structure("u1an39").unwrap();
    m.z.truncate(n);
    m.positions.truncate(n);
    m
}

/// Orbital-like columns (nbf x nc, column-major).
fn orbitals(nbf: usize, nc: usize, seed: u64) -> Vec<f64> {
    let mut rng = Rng(seed);
    (0..nbf * nc).map(|_| (rng.uniform() - 0.5) * 0.5).collect()
}

/// Lower Cholesky factor (row-major) and solves with it.
struct Chol {
    n: usize,
    l: Vec<f64>,
}

impl Chol {
    fn new(v: &[f64], n: usize) -> Chol {
        let mut l = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..=i {
                let mut s = v[i * n + j];
                for k in 0..j {
                    s -= l[i * n + k] * l[j * n + k];
                }
                l[i * n + j] = if i == j { s.sqrt() } else { s / l[j * n + j] };
            }
        }
        Chol { n, l }
    }

    /// x = V^-1 b in place.
    fn solve(&self, b: &mut [f64]) {
        let (n, l) = (self.n, &self.l);
        for i in 0..n {
            let mut s = b[i];
            for k in 0..i {
                s -= l[i * n + k] * b[k];
            }
            b[i] = s / l[i * n + i];
        }
        for i in (0..n).rev() {
            let mut s = b[i];
            for k in i + 1..n {
                s -= l[k * n + i] * b[k];
            }
            b[i] = s / l[i * n + i];
        }
    }
}

/// The dense DF gradient (J with the total density, a_x K from the occupied orbitals; the
/// code linus's blocked CPU gradient replaced), and its J part alone.
fn df_reference(
    basis: &BasisSet,
    aux: &BasisSet,
    occ: &[(Vec<f64>, usize)],
    restricted: bool,
    a_x: f64,
) -> (Vec<[f64; 3]>, Vec<[f64; 3]>) {
    let n = basis.nbf;
    let n2 = n * n;
    let naux = aux.nbf;
    let pairs = PairSet::build(basis, PairOptions::with_order(DerivOrder::One));
    let ap = PairSet::aux(aux, PairOptions::with_order(DerivOrder::One));
    let v = linus_ints::reference::eri_2c(aux, &ap);
    let j3 = linus_ints::reference::eri_3c(basis, &pairs, aux, &ap);
    let chol = Chol::new(&v, naux);
    let w = if restricted { 2.0 } else { 1.0 };
    let mut d = vec![0.0; n2];
    for (c, o) in occ {
        for i in 0..n {
            for j in 0..n {
                d[i * n + j] += w * (0..*o).map(|k| c[k * n + i] * c[k * n + j]).sum::<f64>();
            }
        }
    }
    let mut cfit: Vec<f64> = (0..naux)
        .map(|p| {
            j3[p * n2..(p + 1) * n2]
                .iter()
                .zip(&d)
                .map(|(a, b)| a * b)
                .sum()
        })
        .collect();
    chol.solve(&mut cfit);
    let mut gam = vec![0.0; naux * naux];
    for p in 0..naux {
        for q in 0..naux {
            gam[p * naux + q] = -0.5 * cfit[p] * cfit[q];
        }
    }
    let shells = 0..aux.shells.len();
    let j_only = {
        let dens = linus_ints::reference::Dens3c {
            full: None,
            aux_shells: shells.clone(),
            separable: vec![(1.0, &cfit, &d)],
        };
        let mut g = linus_ints::reference::eri_grad_3c(
            basis,
            &pairs,
            aux,
            &ap,
            &dens,
            0.0,
            &linus_ints::operator::Operator::Coulomb,
        );
        add_atoms(
            &mut g,
            &linus_ints::reference::oneshot::grad_2c(
                aux,
                &ap,
                &gam,
                0.0,
                &linus_ints::operator::Operator::Coulomb,
            ),
        );
        g
    };
    let mut gk = vec![0.0; naux * n2];
    if a_x != 0.0 {
        // B~ = V^-1 (P|mn), column by column
        let mut bt = j3.clone();
        let mut col = vec![0.0; naux];
        for mn in 0..n2 {
            for p in 0..naux {
                col[p] = j3[p * n2 + mn];
            }
            chol.solve(&mut col);
            for p in 0..naux {
                bt[p * n2 + mn] = col[p];
            }
        }
        for (c, o) in occ {
            let o = *o;
            // X_P = C^T B~_P C (o x o), G_P -= a w C X_P C^T
            let mut xs = vec![0.0; naux * o * o];
            for p in 0..naux {
                let b = &bt[p * n2..(p + 1) * n2];
                let mut t = vec![0.0; n * o];
                for m in 0..n {
                    for j in 0..o {
                        t[m * o + j] = (0..n).map(|nn| b[m * n + nn] * c[j * n + nn]).sum();
                    }
                }
                for i in 0..o {
                    for j in 0..o {
                        xs[(p * o + i) * o + j] = (0..n).map(|m| c[i * n + m] * t[m * o + j]).sum();
                    }
                }
                let x = &xs[p * o * o..(p + 1) * o * o];
                let mut u = vec![0.0; n * o];
                for m in 0..n {
                    for j in 0..o {
                        u[m * o + j] = (0..o).map(|i| c[i * n + m] * x[i * o + j]).sum();
                    }
                }
                for m in 0..n {
                    for nn in 0..n {
                        let v: f64 = (0..o).map(|j| u[m * o + j] * c[j * n + nn]).sum();
                        gk[p * n2 + m * n + nn] -= a_x * w * v;
                    }
                }
            }
            for p in 0..naux {
                for q in 0..naux {
                    let s: f64 = (0..o * o)
                        .map(|k| xs[p * o * o + k] * xs[q * o * o + k])
                        .sum();
                    gam[p * naux + q] += 0.5 * a_x * w * s;
                }
            }
        }
    }
    let dens = linus_ints::reference::Dens3c {
        full: (a_x != 0.0).then_some(&gk[..]),
        aux_shells: shells,
        separable: vec![(1.0, &cfit, &d)],
    };
    let mut g = linus_ints::reference::eri_grad_3c(
        basis,
        &pairs,
        aux,
        &ap,
        &dens,
        0.0,
        &linus_ints::operator::Operator::Coulomb,
    );
    add_atoms(
        &mut g,
        &linus_ints::reference::oneshot::grad_2c(
            aux,
            &ap,
            &gam,
            0.0,
            &linus_ints::operator::Operator::Coulomb,
        ),
    );
    (g, j_only)
}

fn check_df(mol: &Structure, basis_name: &str, restricted: bool, a_x: f64, small_blocks: bool) {
    let Some(gpu) = common::gpu() else { return };
    let basis = harness::basis(mol, basis_name).unwrap();
    let aux = harness::basis(mol, "def2-universal-jkfit").unwrap();
    let nbf = basis.nbf;
    let occ: Vec<(Vec<f64>, usize)> = if restricted {
        vec![(orbitals(nbf, 5, 3), 5)]
    } else {
        vec![(orbitals(nbf, 5, 3), 5), (orbitals(nbf, 4, 9), 4)]
    };
    let (g_ref, gj_ref) = df_reference(&basis, &aux, &occ, restricted, a_x);
    let scale = g_ref
        .iter()
        .flat_map(|x| x.iter())
        .fold(0.0f64, |a, b| a.max(b.abs()));
    let occ_ref: Vec<(&[f64], usize)> = occ.iter().map(|(c, o)| (c.as_slice(), *o)).collect();
    let xop = Operator::Attenuated {
        coulomb: a_x,
        long_range: 0.0,
        omega: 0.0,
    };
    // the derivative integrals contracted inside the kernels (fused), and stored then
    // contracted (the first implementation); without exchange also in Hermite space
    let mut both = vec![];
    let mut variants = vec![(true, false), (false, false)];
    if a_x == 0.0 {
        variants.push((true, true));
    }
    for (fused, hermite_j) in variants {
        let opts = GpuDfGradOptions {
            split_terms: true,
            // several blocks and sub-blocks even for water
            block_bytes: if small_blocks { 1 << 15 } else { 2 << 30 },
            deriv_bytes: if small_blocks { 1 << 16 } else { 2 << 30 },
            thresholds: DfThresholds::none(),
            planes: if fused {
                DerivPlanes::Fused
            } else {
                DerivPlanes::Stored
            },
            hermite_j,
        };
        let mut eng = GpuDfGrad::new(&gpu, &basis, &aux, &opts).unwrap();
        let r = eng.compute(&gpu, &occ_ref, restricted, xop).unwrap();
        let r2 = eng.compute(&gpu, &occ_ref, restricted, xop).unwrap();
        assert!(bitwise_equal(&r.gradient, &r2.gradient), "not reproducible");
        let tag = format!(
            "{} {basis_name} {} a_x {a_x} fused {fused} hermite {hermite_j} ({} blocks, {} chunks)",
            harness::formula(mol),
            if restricted { "R" } else { "U" },
            eng.plan().blocks.len(),
            eng.plan().chunks.len()
        );
        let dj = max_abs_diff(r.coulomb.as_ref().unwrap(), &gj_ref);
        let dt = max_abs_diff(&r.gradient, &g_ref);
        eprintln!("{tag}: max |g| {scale:.3e}, J {dj:.2e}, total {dt:.2e}");
        assert!(dj < 1e-12 * scale.max(1.0), "{tag}: J");
        assert!(dt < 1e-12 * scale.max(1.0), "{tag}: J + K");
        // the unsplit sum agrees to round-off
        eng.set_split_terms(false);
        let r3 = eng.compute(&gpu, &occ_ref, restricted, xop).unwrap();
        assert!(max_abs_diff(&r3.gradient, &r.gradient) < 1e-13 * scale.max(1.0));
        both.push(r3.gradient);
    }
    let d = max_abs_diff(&both[0], &both[1]);
    eprintln!("fused vs stored planes: {d:.2e}");
    assert!(d < 1e-13 * scale.max(1.0));
    if let Some(h) = both.get(2) {
        let d = max_abs_diff(&both[0], h);
        eprintln!("fused vs Hermite (J only): {d:.2e}");
        assert!(d < 1e-12 * scale.max(1.0));
    }
}

#[test]
fn df_water_restricted_hybrid() {
    check_df(&water(), "def2-svp", true, 0.2, false);
    check_df(&water(), "def2-svp", true, 0.2, true);
}

#[test]
fn df_water_unrestricted_and_j_only() {
    check_df(&water(), "def2-svp", false, 1.0, true);
    check_df(&water(), "def2-svp", false, 0.0, false);
}

/// The stored derivative planes give way to the fused kernels when their workspace does not fit
/// (`DerivPlanes::StoredOrFused`, the default on FP64-rich cards): the call reports the fused
/// kernels and why, and its gradient is the fused engine's, bit for bit. The device is filled
/// with a grant up to between the two workspaces.
#[test]
fn df_planes_give_way_to_fused_when_short_of_memory() {
    let Some(gpu) = common::gpu() else { return };
    let mol = fragment(4);
    let basis = harness::basis(&mol, "def2-svp").unwrap();
    let aux = harness::basis(&mol, "def2-universal-jkfit").unwrap();
    let occ = orbitals(basis.nbf, 6, 5);
    let occ_ref: Vec<(&[f64], usize)> = vec![(occ.as_slice(), 6)];
    let xop = Operator::Attenuated {
        coulomb: 0.25,
        long_range: 0.0,
        omega: 0.0,
    };
    let opts = |planes| GpuDfGradOptions {
        // several sub-blocks, so that the planes' workspace is well above the fused one's
        deriv_bytes: 1 << 20,
        planes,
        ..GpuDfGradOptions::default()
    };
    let mut fused = GpuDfGrad::new(&gpu, &basis, &aux, &opts(DerivPlanes::Fused)).unwrap();
    let mut eng = GpuDfGrad::new(&gpu, &basis, &aux, &opts(DerivPlanes::StoredOrFused)).unwrap();
    let (w1, w6) = (
        fused.workspace_bytes(&[6], &xop),
        eng.workspace_bytes(&[6], &xop),
    );
    assert!(w6 > w1 + (1 << 18), "planes {w6} B, fused {w1} B");
    // with room for the fused workspace only, the planes give way
    let room = gpu.budget().free();
    let blocker = gpu
        .grant(room - (w1 + w6) / 2, "the test's filler")
        .unwrap();
    eng.grant_workspace(&gpu, &[6], &xop).unwrap();
    let k = eng.kernels(&[6], &xop);
    assert_eq!(k.kernels, "fused");
    let why = k.declined.unwrap_or_default();
    assert!(why.contains("does not fit"), "{why}");
    let r = eng.compute(&gpu, &occ_ref, true, xop).unwrap();
    drop(blocker);
    fused.grant_workspace(&gpu, &[6], &xop).unwrap();
    let rf = fused.compute(&gpu, &occ_ref, true, xop).unwrap();
    assert!(
        bitwise_equal(&r.gradient, &rf.gradient),
        "against the fused engine"
    );
    // with room, the planes are kept
    let mut kept = GpuDfGrad::new(&gpu, &basis, &aux, &opts(DerivPlanes::StoredOrFused)).unwrap();
    kept.grant_workspace(&gpu, &[6], &xop).unwrap();
    assert_eq!(kept.kernels(&[6], &xop).kernels, "stored_planes");
}

#[test]
fn df_fragment_tzvp() {
    check_df(&fragment(4), "def2-tzvp", true, 0.25, true);
    // f pairs and the fit's g shells in Hermite space
    check_df(&fragment(4), "def2-tzvp", true, 0.0, false);
}

/// Dense (P|op|mn) as [P][m][n] (linus-ints' four-center code with the auxiliary shells as
/// bra), as `eri4c::eri_3c` for 1/r.
fn eri_3c_op(basis: &BasisSet, pairs: &PairSet, aux: &BasisSet, op: &Operator) -> Vec<f64> {
    let (nbf, naux) = (basis.nbf, aux.nbf);
    let ap = PairSet::aux(aux, PairOptions::default());
    let sh = &basis.shells;
    let mut out = vec![0.0; naux * nbf * nbf];
    let mut w = eri4c::EriWork::new();
    for (p, apr) in ap.pairs().iter().enumerate() {
        for pr in pairs.pairs() {
            let nab = pr.nab();
            let mut buf = vec![0.0; apr.na * nab];
            eri4c::eri_quartet(apr, pr, op, &mut buf, &mut w);
            let p0 = aux.shells[p].ao_start;
            let (a0, b0) = (sh[pr.sa].ao_start, sh[pr.sb].ao_start);
            for x in 0..apr.na {
                let row = &mut out[(p0 + x) * nbf * nbf..(p0 + x + 1) * nbf * nbf];
                for ia in 0..pr.na {
                    for ib in 0..pr.nb {
                        let v = buf[x * nab + ia * pr.nb + ib];
                        row[(a0 + ia) * nbf + b0 + ib] = v;
                        row[(b0 + ib) * nbf + a0 + ia] = v;
                    }
                }
            }
        }
    }
    out
}

/// The dense DF gradient of J and the exchange of `xop` = a/r + b erf(omega r)/r in the wK
/// scheme (the module docs of `grad::df`), straight from the formulas: G, G^w, the metric
/// term with the mixed X~/Y~ products; and its J part alone.
fn df_reference_op(
    basis: &BasisSet,
    aux: &BasisSet,
    occ: &[(Vec<f64>, usize)],
    restricted: bool,
    xop: Operator,
) -> (Vec<[f64; 3]>, Vec<[f64; 3]>) {
    let n = basis.nbf;
    let n2 = n * n;
    let naux = aux.nbf;
    let (a, b, omega) = (xop.coulomb(), xop.long_range(), xop.omega());
    let erf = Operator::LongRange { omega };
    let pairs = PairSet::build(basis, PairOptions::with_order(DerivOrder::One));
    let ap = PairSet::aux(aux, PairOptions::with_order(DerivOrder::One));
    let v = linus_ints::reference::eri_2c(aux, &ap);
    let j3 = linus_ints::reference::eri_3c(basis, &pairs, aux, &ap);
    let j3w = eri_3c_op(basis, &pairs, aux, &erf);
    let chol = Chol::new(&v, naux);
    let w = if restricted { 2.0 } else { 1.0 };
    let mut d = vec![0.0; n2];
    for (c, o) in occ {
        for i in 0..n {
            for j in 0..n {
                d[i * n + j] += w * (0..*o).map(|k| c[k * n + i] * c[k * n + j]).sum::<f64>();
            }
        }
    }
    let mut cfit: Vec<f64> = (0..naux)
        .map(|p| {
            j3[p * n2..(p + 1) * n2]
                .iter()
                .zip(&d)
                .map(|(x, y)| x * y)
                .sum()
        })
        .collect();
    chol.solve(&mut cfit);
    let mut gam = vec![0.0; naux * naux];
    for p in 0..naux {
        for q in 0..naux {
            gam[p * naux + q] = -0.5 * cfit[p] * cfit[q];
        }
    }
    let shells = 0..aux.shells.len();
    let j_only = {
        let dens = linus_ints::reference::Dens3c {
            full: None,
            aux_shells: shells.clone(),
            separable: vec![(1.0, &cfit, &d)],
        };
        let mut g = linus_ints::reference::eri_grad_3c(
            basis,
            &pairs,
            aux,
            &ap,
            &dens,
            0.0,
            &linus_ints::operator::Operator::Coulomb,
        );
        add_atoms(
            &mut g,
            &linus_ints::reference::oneshot::grad_2c(
                aux,
                &ap,
                &gam,
                0.0,
                &linus_ints::operator::Operator::Coulomb,
            ),
        );
        g
    };
    // V^-1 (P|op|mn), column by column
    let fit = |t: &[f64]| {
        let mut bt = t.to_vec();
        let mut col = vec![0.0; naux];
        for mn in 0..n2 {
            for p in 0..naux {
                col[p] = t[p * n2 + mn];
            }
            chol.solve(&mut col);
            for p in 0..naux {
                bt[p * n2 + mn] = col[p];
            }
        }
        bt
    };
    let (bt, btw) = (fit(&j3), fit(&j3w));
    // C^T T_P C (o x o) for every P
    let mo = |t: &[f64], c: &[f64], o: usize| {
        let mut xs = vec![0.0; naux * o * o];
        for p in 0..naux {
            let tp = &t[p * n2..(p + 1) * n2];
            let mut u = vec![0.0; n * o];
            for m in 0..n {
                for j in 0..o {
                    u[m * o + j] = (0..n).map(|nn| tp[m * n + nn] * c[j * n + nn]).sum();
                }
            }
            for i in 0..o {
                for j in 0..o {
                    xs[(p * o + i) * o + j] = (0..n).map(|m| c[i * n + m] * u[m * o + j]).sum();
                }
            }
        }
        xs
    };
    // G_P += f C M_P C^T
    let add_g = |g: &mut [f64], c: &[f64], o: usize, m: &[f64], f: f64| {
        for p in 0..naux {
            let x = &m[p * o * o..(p + 1) * o * o];
            let mut u = vec![0.0; n * o];
            for mm in 0..n {
                for j in 0..o {
                    u[mm * o + j] = (0..o).map(|i| c[i * n + mm] * x[i * o + j]).sum();
                }
            }
            for mm in 0..n {
                for nn in 0..n {
                    let v: f64 = (0..o).map(|j| u[mm * o + j] * c[j * n + nn]).sum();
                    g[p * n2 + mm * n + nn] += f * v;
                }
            }
        }
    };
    let dot = |x: &[f64], y: &[f64], o: usize, p: usize, q: usize| -> f64 {
        (0..o * o)
            .map(|k| x[p * o * o + k] * y[q * o * o + k])
            .sum()
    };
    let mut gk = vec![0.0; naux * n2];
    let mut gkw = vec![0.0; naux * n2];
    for (c, o) in occ {
        let o = *o;
        let x = mo(&bt, c, o);
        let y = mo(&btw, c, o);
        add_g(&mut gk, c, o, &x, -a * w);
        add_g(&mut gk, c, o, &y, -0.5 * b * w);
        add_g(&mut gkw, c, o, &x, -0.5 * b * w);
        for p in 0..naux {
            for q in 0..naux {
                gam[p * naux + q] += w
                    * (0.5 * a * dot(&x, &x, o, p, q)
                        + 0.25 * b * (dot(&x, &y, o, p, q) + dot(&y, &x, o, p, q)));
            }
        }
    }
    let dens = linus_ints::reference::Dens3c {
        full: Some(&gk[..]),
        aux_shells: shells.clone(),
        separable: vec![(1.0, &cfit, &d)],
    };
    let mut g = linus_ints::reference::eri_grad_3c(
        basis,
        &pairs,
        aux,
        &ap,
        &dens,
        0.0,
        &linus_ints::operator::Operator::Coulomb,
    );
    let densw = linus_ints::reference::Dens3c {
        full: Some(&gkw[..]),
        aux_shells: shells,
        separable: vec![],
    };
    add_atoms(
        &mut g,
        &linus_ints::reference::eri_grad_3c(basis, &pairs, aux, &ap, &densw, 0.0, &erf),
    );
    add_atoms(
        &mut g,
        &linus_ints::reference::oneshot::grad_2c(
            aux,
            &ap,
            &gam,
            0.0,
            &linus_ints::operator::Operator::Coulomb,
        ),
    );
    (g, j_only)
}

/// The range-separated exchange's gradient (wK) against the dense formulas: J apart, the
/// total, fused and stored derivative planes, and run to run.
fn check_df_op(mol: &Structure, basis_name: &str, restricted: bool, xop: Operator, small: bool) {
    let Some(gpu) = common::gpu() else { return };
    let basis = harness::basis(mol, basis_name).unwrap();
    let aux = harness::basis(mol, "def2-universal-jkfit").unwrap();
    let nbf = basis.nbf;
    let occ: Vec<(Vec<f64>, usize)> = if restricted {
        vec![(orbitals(nbf, 5, 3), 5)]
    } else {
        vec![(orbitals(nbf, 5, 3), 5), (orbitals(nbf, 4, 9), 4)]
    };
    let (g_ref, gj_ref) = df_reference_op(&basis, &aux, &occ, restricted, xop);
    let scale = g_ref
        .iter()
        .flat_map(|x| x.iter())
        .fold(0.0f64, |a, b| a.max(b.abs()));
    let occ_ref: Vec<(&[f64], usize)> = occ.iter().map(|(c, o)| (c.as_slice(), *o)).collect();
    let mut both = vec![];
    for fused in [true, false] {
        let opts = GpuDfGradOptions {
            split_terms: true,
            block_bytes: if small { 1 << 15 } else { 2 << 30 },
            deriv_bytes: if small { 1 << 16 } else { 2 << 30 },
            thresholds: DfThresholds::none(),
            planes: if fused {
                DerivPlanes::Fused
            } else {
                DerivPlanes::Stored
            },
            hermite_j: false,
        };
        let mut eng = GpuDfGrad::new(&gpu, &basis, &aux, &opts).unwrap();
        let r = eng.compute(&gpu, &occ_ref, restricted, xop).unwrap();
        let r2 = eng.compute(&gpu, &occ_ref, restricted, xop).unwrap();
        assert!(bitwise_equal(&r.gradient, &r2.gradient), "not reproducible");
        let tag = format!(
            "{} {basis_name} {} {xop:?} fused {fused} ({} blocks)",
            harness::formula(mol),
            if restricted { "R" } else { "U" },
            eng.plan().blocks.len(),
        );
        let dj = max_abs_diff(r.coulomb.as_ref().unwrap(), &gj_ref);
        let dt = max_abs_diff(&r.gradient, &g_ref);
        eprintln!("{tag}: max |g| {scale:.3e}, J {dj:.2e}, total {dt:.2e}");
        assert!(dj < 1e-12 * scale.max(1.0), "{tag}: J");
        assert!(dt < 1e-12 * scale.max(1.0), "{tag}: J + K");
        eng.set_split_terms(false);
        let r3 = eng.compute(&gpu, &occ_ref, restricted, xop).unwrap();
        assert!(max_abs_diff(&r3.gradient, &r.gradient) < 1e-13 * scale.max(1.0));
        both.push(r3.gradient);
    }
    let d = max_abs_diff(&both[0], &both[1]);
    eprintln!("fused vs stored planes: {d:.2e}");
    assert!(d < 1e-13 * scale.max(1.0));
}

#[test]
fn df_range_separated_water() {
    // wB97X's operator, CAM-B3LYP's (b < 0), erf alone (a = 0)
    let wb97x = Operator::from_libxc(1.0, -0.842294, 0.3);
    let cam = Operator::from_libxc(0.19, 0.46, 0.33);
    check_df_op(&water(), "def2-svp", true, wb97x, false);
    check_df_op(&water(), "def2-svp", false, wb97x, true);
    check_df_op(&water(), "def2-svp", true, cam, true);
    check_df_op(
        &water(),
        "def2-svp",
        false,
        Operator::LongRange { omega: 0.4 },
        false,
    );
}

#[test]
fn df_range_separated_fragment_tzvp() {
    let wb97x = Operator::from_libxc(1.0, -0.842294, 0.3);
    check_df_op(&fragment(4), "def2-tzvp", false, wb97x, true);
}

/// The XC gradient by quadrature over every function at every point (densities column-major
/// nbf x nbf, symmetric).
fn xc_reference(
    basis: &BasisSet,
    grid: &MolecularGrid,
    name: &str,
    dens: &[Vec<f64>],
    restricted: bool,
    grid_response: bool,
) -> Vec<[f64; 3]> {
    let nbf = basis.nbf;
    let np = grid.x.len();
    let nsp = dens.len();
    let f = Functional::new(name, !restricted).unwrap();
    let kind = XcKind::of(&f);
    let shells: Vec<usize> = (0..basis.shells.len()).collect();
    let mut ao = vec![0.0; 10 * nbf * np];
    eval_ao(basis, &shells, &grid.x, &grid.y, &grid.z, 2, &mut ao);
    let phi = |d: usize, m: usize, g: usize| ao[(d * nbf + m) * np + g];
    // X_s and Y_sk: [s][k][m][g] (k = 0: phi_0 D, 1..3: phi_k D)
    let mut xy = vec![0.0; nsp * 4 * nbf * np];
    for s in 0..nsp {
        for k in 0..4 {
            for m in 0..nbf {
                for g in 0..np {
                    xy[((s * 4 + k) * nbf + m) * np + g] =
                        (0..nbf).map(|n| phi(k, n, g) * dens[s][m * nbf + n]).sum();
                }
            }
        }
    }
    let xyv = |s: usize, k: usize, m: usize, g: usize| xy[((s * 4 + k) * nbf + m) * np + g];
    let mut rho = vec![[0.0; 2]; np];
    let mut grad = vec![[[0.0; 3]; 2]; np];
    let mut tau = vec![[0.0; 2]; np];
    for s in 0..nsp {
        for g in 0..np {
            for m in 0..nbf {
                rho[g][s] += phi(0, m, g) * xyv(s, 0, m, g);
                for (k, gk) in grad[g][s].iter_mut().enumerate() {
                    *gk += 2.0 * phi(1 + k, m, g) * xyv(s, 0, m, g);
                    tau[g][s] += 0.5 * phi(1 + k, m, g) * xyv(s, 1 + k, m, g);
                }
            }
        }
    }
    let dot = |a: &[f64; 3], b: &[f64; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    let (mut rin, mut sin, mut tin) = (vec![], vec![], vec![]);
    for g in 0..np {
        for s in 0..nsp {
            rin.push(rho[g][s]);
            tin.push(tau[g][s]);
        }
        if nsp == 1 {
            sin.push(dot(&grad[g][0], &grad[g][0]));
        } else {
            sin.extend([
                dot(&grad[g][0], &grad[g][0]),
                dot(&grad[g][0], &grad[g][1]),
                dot(&grad[g][1], &grad[g][1]),
            ]);
        }
    }
    let input = if kind.mgga {
        XcInput::mgga(&rin, &sin, &tin)
    } else if kind.gga {
        XcInput::gga(&rin, &sin)
    } else {
        XcInput::lda(&rin)
    };
    let mut out = XcOutput::new();
    f.eval(1, &input, &mut out).unwrap();
    let (zk, vr, vs, vt) = (out.zk(), out.vrho(), out.vsigma(), out.vtau());
    let natom = grid.atom_z.len();
    let mut gat = vec![[0.0; 3]; natom];
    let mut tsum = vec![[0.0; 3]; np];
    const D2: [[usize; 3]; 3] = [[4, 5, 6], [5, 7, 8], [6, 8, 9]];
    let mut atom_of = vec![0; nbf];
    for s in &basis.shells {
        atom_of[s.ao_start..s.ao_start + s.nfunc()].fill(s.atom);
    }
    for s in 0..nsp {
        for g in 0..np {
            let w = grid.w[g];
            let mut vg = [0.0; 3];
            if kind.gga {
                for k in 0..3 {
                    vg[k] = w * if nsp == 1 {
                        2.0 * vs[g] * grad[g][0][k]
                    } else {
                        2.0 * vs[3 * g + 2 * s] * grad[g][s][k] + vs[3 * g + 1] * grad[g][1 - s][k]
                    };
                }
            }
            let vrw = w * vr[g * nsp + s];
            let vtw = if kind.mgga { w * vt[g * nsp + s] } else { 0.0 };
            // T_n and B_m = sum_n T_n D_nm
            let t: Vec<f64> = (0..nbf)
                .map(|n| vrw * phi(0, n, g) + (0..3).map(|k| vg[k] * phi(1 + k, n, g)).sum::<f64>())
                .collect();
            for m in 0..nbf {
                let b: f64 = (0..nbf).map(|n| t[n] * dens[s][m * nbf + n]).sum();
                for i in 0..3 {
                    let mut c = phi(1 + i, m, g) * b;
                    if kind.gga {
                        for k in 0..3 {
                            c += phi(D2[i][k], m, g) * vg[k] * xyv(s, 0, m, g);
                            if kind.mgga {
                                c += 0.5 * vtw * phi(D2[i][k], m, g) * xyv(s, 1 + k, m, g);
                            }
                        }
                    }
                    gat[atom_of[m]][i] -= 2.0 * c;
                    tsum[g][i] += 2.0 * c;
                }
            }
        }
    }
    if grid_response {
        let mut eps = vec![0.0; np];
        for g in 0..np {
            let rt: f64 = (0..nsp).map(|s| rho[g][s]).sum();
            eps[g] = rt * zk[g];
            let own = grid.atom[g] as usize;
            for i in 0..3 {
                gat[own][i] += tsum[g][i];
            }
        }
        add_atoms(&mut gat, &grid.weight_gradient_contract(&eps));
    }
    gat
}

/// A symmetric density from orbital-like columns (column-major).
fn density(nbf: usize, nc: usize, seed: u64, scale: f64) -> Vec<f64> {
    let c = orbitals(nbf, nc, seed);
    let mut d = vec![0.0; nbf * nbf];
    for j in 0..nbf {
        for i in 0..nbf {
            d[j * nbf + i] = scale
                * (0..nc)
                    .map(|k| c[k * nbf + i] * c[k * nbf + j])
                    .sum::<f64>();
        }
    }
    d
}

fn check_xc(mol: &Structure, basis_name: &str, spec: GridSpec, name: &str, restricted: bool) {
    let Some(gpu) = common::gpu() else { return };
    let basis = harness::basis(mol, basis_name).unwrap();
    let grid = harness::grid(mol, &spec).unwrap();
    let nbf = basis.nbf;
    let dens = if restricted {
        vec![density(nbf, 5, 3, 2.0)]
    } else {
        vec![density(nbf, 5, 3, 1.0), density(nbf, 4, 9, 1.0)]
    };
    use linus_gpu::xc::{XcMath, XcScreen};
    // (grid response, block points, screening, arithmetic, keep X, workspace bytes)
    #[cfg_attr(not(feature = "study"), allow(unused_mut))]
    let mut runs = vec![
        (
            true,
            2048,
            XcScreen::Batch,
            XcMath::Fp64,
            true,
            1usize << 30,
        ),
        (false, 512, XcScreen::Batch, XcMath::Fp64, false, 1),
        (true, 2048, XcScreen::Batch, XcMath::Fp32, false, 1 << 30),
    ];
    // the precision studies' block path
    #[cfg(feature = "study")]
    runs.extend([
        (true, 333, XcScreen::Block, XcMath::Fp64, true, 1 << 30),
        (false, 2048, XcScreen::Block, XcMath::Fp64, true, 1 << 30),
    ]);
    for (gr, block_points, screen, math, keep, workspace) in runs {
        let g_ref = xc_reference(&basis, &grid, name, &dens, restricted, gr);
        let opts = GpuXcGradOptions {
            block_points,
            becke_chunk: 37,
            screen,
            math,
            workspace,
            x_store: if keep { 8 << 30 } else { 0 },
            ..Default::default()
        };
        let mut eng = GpuXcGrad::new(&gpu, &basis, &grid, &spec, name, &opts).unwrap();
        let dr: Vec<&[f64]> = dens.iter().map(|d| d.as_slice()).collect();
        let g = eng.compute(&gpu, &dr, restricted, gr).unwrap();
        let g2 = eng.compute(&gpu, &dr, restricted, gr).unwrap();
        assert!(bitwise_equal(&g, &g2), "not reproducible");
        let scale = g_ref
            .iter()
            .flat_map(|x| x.iter())
            .fold(0.0f64, |a, b| a.max(b.abs()));
        let d = max_abs_diff(&g, &g_ref);
        let mut sum = [0.0; 3];
        for x in &g {
            for k in 0..3 {
                sum[k] += x[k];
            }
        }
        eprintln!(
            "{} {basis_name} {name} {} grid response {gr} {screen:?} {math:?} keep X {keep} (Becke on device {}): max |g| {scale:.3e}, diff {d:.2e}, sum {sum:?}",
            harness::formula(mol),
            if restricted { "R" } else { "U" },
            eng.stats().becke_on_device
        );
        let tol = if math == XcMath::Fp32 { 1e-5 } else { 1e-11 };
        assert!(d < tol * scale.max(1.0), "XC gradient differs: {d:e}");
    }
}

#[test]
fn xc_water_lda_and_gga() {
    check_xc(&water(), "def2-svp", GridSpec::pyscf(1), "svwn", true);
    check_xc(&water(), "def2-svp", GridSpec::pyscf(1), "pbe", false);
}

#[test]
fn xc_water_hybrid_and_meta_gga() {
    check_xc(&water(), "def2-svp", GridSpec::pyscf(1), "b3lyp", true);
    check_xc(&water(), "def2-svp", GridSpec::pyscf(1), "tpss", false);
}

#[test]
fn xc_fragment_becke_and_ssf() {
    check_xc(&fragment(5), "def2-svp", GridSpec::pyscf(1), "b3lyp", false);
    // SSF partition: the weight derivatives on the host
    check_xc(&fragment(5), "def2-svp", GridSpec::linus(1), "pbe", true);
}

/// The weight-derivative kernels agree to round-off: the pair kernel (the default at 39 atoms)
/// and the warp kernel (forced: no shared memory for the pair kernel), with and without the
/// screening of negligible atoms.
#[test]
fn xc_becke_kernels_agree() {
    let Some(gpu) = common::gpu() else { return };
    let mol = fragment(39);
    let basis = harness::basis(&mol, "def2-svp").unwrap();
    let spec = GridSpec::pyscf(0);
    let grid = harness::grid(&mol, &spec).unwrap();
    let nbf = basis.nbf;
    let d = density(nbf, 12, 5, 2.0);
    let run = |becke_screen: bool, warp: bool| {
        let opts = GpuXcGradOptions {
            becke_screen,
            becke_pair_shared: if warp {
                0
            } else {
                GpuXcGradOptions::default().becke_pair_shared
            },
            ..Default::default()
        };
        let mut eng = GpuXcGrad::new(&gpu, &basis, &grid, &spec, "pbe", &opts).unwrap();
        eng.compute(&gpu, &[&d], true, true).unwrap()
    };
    // the pair kernel, screened (1e-18 relative) and not; the warp kernel, screened and not
    let (a, e) = (run(true, false), run(false, false));
    let (w, w0) = (run(true, true), run(false, true));
    let scale = a
        .iter()
        .flat_map(|x| x.iter())
        .fold(0.0f64, |m, v| m.max(v.abs()));
    let dscreen = max_abs_diff(&a, &e);
    let (dwarp, dwarp0) = (max_abs_diff(&a, &w), max_abs_diff(&e, &w0));
    eprintln!(
        "39 atoms Becke: max |g| {scale:.3e}, screened vs all pairs {dscreen:.2e}, warp vs pairs {dwarp:.2e} (unscreened {dwarp0:.2e})"
    );
    for x in [dscreen, dwarp, dwarp0] {
        assert!(x < 1e-12 * scale.max(1.0));
    }
}

/// The engines set up from a shared [`DfShared`] (the SCF's layout, packed integrals, metric
/// and derivative records) are the ones that build their own, bit for bit: the DF gradient
/// (a global hybrid, a range-separated operator, an open shell), the DF-J/K and the Hessian's
/// two-electron terms; the derivative records are built once and kept; a `DfShared` of other
/// thresholds or basis sets is not taken.
#[test]
fn df_shared_setups_match_their_own() {
    use linus_gpu::df::{DfShared, Factor, GpuDfJk, GpuDfJkOptions, GpuJkRequest};
    use linus_gpu::eri3c::Eri3cKind;
    use linus_gpu::hess::df::{GpuDfHess, GpuDfHessOptions};
    let Some(gpu) = common::gpu() else { return };
    let mol = fragment(6);
    let basis = harness::basis(&mol, "def2-tzvp").unwrap();
    let aux = harness::basis(&mol, "def2-universal-jkfit").unwrap();
    let nbf = basis.nbf;
    let (ca, cb) = (orbitals(nbf, 9, 3), orbitals(nbf, 8, 5));
    let o = GpuDfGradOptions::default();
    let shared = DfShared::new(&basis, &aux, &o.thresholds).unwrap();
    assert!(shared.fits(&basis, &aux, &o.thresholds));
    let looser = DfThresholds {
        pair: 1e-12,
        ..o.thresholds
    };
    assert!(!shared.fits(&basis, &aux, &looser));
    assert!(!shared.fits(&aux, &basis, &o.thresholds));
    let wb97x = Operator::from_libxc(1.0, -0.842294, 0.3);
    let b3lyp = Operator::Attenuated {
        coulomb: 0.2,
        long_range: 0.0,
        omega: 0.0,
    };
    // (orbitals per channel, restricted, exchange operator)
    type Case<'a> = (&'a [(&'a [f64], usize)], bool, Operator);
    let cases: [Case; 3] = [
        (&[(&ca, 9)], true, b3lyp),
        (&[(&ca, 9)], true, wb97x),
        (&[(&ca, 9), (&cb, 8)], false, b3lyp),
    ];
    for (occ, restricted, op) in cases {
        let mut own = GpuDfGrad::new(&gpu, &basis, &aux, &o).unwrap();
        let mut sh = GpuDfGrad::with_shared(&gpu, &basis, &aux, &o, Some(&shared)).unwrap();
        let a = own.compute(&gpu, occ, restricted, op).unwrap();
        let b = sh.compute(&gpu, occ, restricted, op).unwrap();
        assert!(bitwise_equal(&a.gradient, &b.gradient), "gradient {op:?}");
    }
    // the derivative records: built by the first engine, kept for the next
    let d1 = shared
        .derivatives(
            &basis,
            &aux,
            Eri3cKind::Derivatives,
            o.thresholds.deriv_prim,
        )
        .unwrap();
    let d2 = shared
        .derivatives(
            &basis,
            &aux,
            Eri3cKind::Derivatives,
            o.thresholds.deriv_prim,
        )
        .unwrap();
    assert!(std::sync::Arc::ptr_eq(&d1, &d2));
    // DF-J/K
    let jo = GpuDfJkOptions::default();
    let d: Vec<f64> = (0..nbf * nbf)
        .map(|x| {
            let (i, j) = (x / nbf, x % nbf);
            (0..9).map(|k| ca[k * nbf + i] * ca[k * nbf + j]).sum()
        })
        .collect();
    let req = GpuJkRequest {
        dens: &[&d],
        factors: &[Factor { c: &ca, ncol: 9 }],
        want_j: true,
        want_k: true,
    };
    let a = GpuDfJk::new(&gpu, &basis, &aux, &jo)
        .unwrap()
        .compute(&gpu, &req)
        .unwrap();
    let b = GpuDfJk::with_shared(&gpu, &basis, &aux, &jo, Some(&shared))
        .unwrap()
        .compute(&gpu, &req)
        .unwrap();
    let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    assert_eq!(bits(&a.j[0]), bits(&b.j[0]), "J");
    assert_eq!(bits(&a.k[0]), bits(&b.k[0]), "K");
    // the Hessian's two-electron terms
    let ho = GpuDfHessOptions::default();
    let occ: [(&[f64], usize); 1] = [(&ca, 9)];
    let a = GpuDfHess::new(&gpu, &basis, &aux, &ho)
        .unwrap()
        .compute(&gpu, &occ, &[2.0], 0.2, true, &Operator::Coulomb)
        .unwrap();
    let b = GpuDfHess::with_shared(&gpu, &basis, &aux, &ho, Some(&shared))
        .unwrap()
        .compute(&gpu, &occ, &[2.0], 0.2, true, &Operator::Coulomb)
        .unwrap();
    assert_eq!(bits(&a.hess), bits(&b.hess), "Hessian terms");
    for (x, y) in a.fxo.iter().zip(&b.fxo) {
        assert_eq!(bits(x), bits(y), "F^x C");
    }
}
