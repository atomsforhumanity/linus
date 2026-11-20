//! linus-jk's density-fitted Hessian terms against whole calculations (linus_jk::df::hessian's
//! tests that run the Session).

use crate::driver::{CalcOptions, JkMethod, Session};
use crate::Structure;
use faer::{Mat, MatRef, Par};
use linus_base::memory::MemoryLimit;
use linus_core::timing::Timers;
use linus_ints::operator::Operator;
use linus_ints::{BasisSet, DerivOrder, PairOptions, PairSet};
use linus_jk::df::hessian::wk_hessian_2e;
use linus_jk::df::hessian::*;
use linus_jk::df::reference::DfDerivJk;

const MOL: &str =
    "4\n\nO 0.10 -0.05 0.12\nH 0.85 0.52 -0.03\nH -0.68 0.55 0.05\nC 0.20 -1.30 0.60\n";

fn max_diff(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, linus_base::linalg::max_nan)
}

/// The blocked terms equal DfDerivJk's (closed and open shell, hybrid and pure, Hbar in
/// one block and in atom blocks).
#[test]
fn blocked_terms_match_dfderivjk() {
    for (charge, mult, method, a_x) in [
        (0, 1, "b3lyp", 0.2),
        (1, 2, "b3lyp", 0.2),
        (0, 1, "pbe", 0.0),
        (0, 1, "hf", 1.0),
    ] {
        let mol = Structure::from_xyz_str(MOL, charge, mult).unwrap();
        let o = CalcOptions {
            method: Some(method.into()),
            basis: Some("def2-svp".into()),
            jk: JkMethod::Df {
                aux: "def2-universal-jkfit".into(),
            }
            .into(),
            threads: Some(4),
            ..Default::default()
        };
        let mut c = Session::new(mol, o).unwrap();
        c.energy().unwrap();
        let r = c.last().unwrap().clone();
        let b = &c.cx.basis;
        let pairs = PairSet::build(b, PairOptions::with_order(DerivOrder::Two));
        let aux = BasisSet::new(
            &c.cx.plan.load_basis("def2-universal-jkfit").unwrap(),
            &c.cx.mol.z,
            &c.cx.mol.positions,
            true,
        )
        .unwrap();
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap();
        let par = Par::rayon(4);
        pool.install(|| {
            let dfd = DfDerivJk::new(b, &aux, par);
            let mut t = Timers::default();
            let ca = r.c_a.subcols(0, r.nalpha);
            let cb = r.c_b.subcols(0, r.nbeta);
            // Hbar in one block, and every atom its own block (the memory_gb budget)
            for (mb, mem) in [
                (
                    "all",
                    MemoryLimit {
                        limit: Some(usize::MAX / 2),
                    },
                ),
                ("none", MemoryLimit { limit: Some(1) }),
            ] {
                if r.restricted {
                    let old = dfd.hessian_2e_restricted(ca, a_x, par, &mut t);
                    let xop = Some(Operator::Attenuated {
                        coulomb: a_x,
                        long_range: 0.0,
                        omega: 0.0,
                    });
                    let new = df_hessian_2e(
                        b,
                        &pairs,
                        &aux,
                        &linus_jk::df::CoulombMetric::new(&aux, faer::Par::Seq).unwrap(),
                        &[ca],
                        &[2.0],
                        xop,
                        mem,
                        par,
                        &mut t,
                    );
                    let dh = max_diff(&old.hess, &new.hess);
                    let df = old
                        .fxo
                        .iter()
                        .zip(&new.fxo)
                        .map(|(u, v)| linus_base::linalg::max_abs((u - &v[0]).as_ref()))
                        .fold(0.0, linus_base::linalg::max_nan);
                    println!("{method} closed hbar {mb}: dH {dh:.2e} dF {df:.2e}");
                    assert!(dh < 1e-9 && df < 1e-9, "{method}: dH {dh:e} dF {df:e}");
                } else {
                    let old = dfd.hessian_2e_unrestricted([ca, cb], a_x, par, &mut t);
                    let new = df_hessian_2e(
                        b,
                        &pairs,
                        &aux,
                        &linus_jk::df::CoulombMetric::new(&aux, faer::Par::Seq).unwrap(),
                        &[ca, cb],
                        &[1.0, 1.0],
                        Some(Operator::Attenuated {
                            coulomb: a_x,
                            long_range: 0.0,
                            omega: 0.0,
                        }),
                        mem,
                        par,
                        &mut t,
                    );
                    let dh = max_diff(&old.hess, &new.hess);
                    let df = old
                        .fxo
                        .iter()
                        .zip(&new.fxo)
                        .map(|(u, v)| {
                            (0..2)
                                .map(|s| linus_base::linalg::max_abs((&u[s] - &v[s]).as_ref()))
                                .fold(0.0, linus_base::linalg::max_nan)
                        })
                        .fold(0.0, linus_base::linalg::max_nan);
                    println!("{method} open hbar {mb}: dH {dh:.2e} dF {df:.2e}");
                    assert!(dh < 1e-9 && df < 1e-9, "{method} open: dH {dh:e} dF {df:e}");
                }
            }
        });
    }
}

/// Range-separated hybrids (wK, polarization identity): the explicit terms are the second
/// derivatives of the RI energy at fixed orbitals, i.e. central differences of the wK DF
/// gradient's two-electron part (linus_jk::df::gradient::df_gradient, validated against energy
/// differences) at fixed C; F^x C_s are differences of J[D_t] - K_w[D_s] (DfJk's one-pass wK
/// exchange) at fixed D, times C_s. Closed and open shell.
#[test]
#[allow(clippy::needless_range_loop, reason = "indexes several arrays in step")]
fn range_separated_terms_match_finite_differences() {
    use linus_base::terms::{JkBuilder, JkRequest};
    use linus_jk::df::{DfConfig, DfJk};
    let x = Operator::from_libxc(1.0, -0.842294, 0.3);
    for (charge, mult) in [(0, 1), (1, 2)] {
        let mol = Structure::from_xyz_str(MOL, charge, mult).unwrap();
        let o = CalcOptions {
            method: Some("wb97x".into()),
            basis: Some("def2-svp".into()),
            jk: JkMethod::Df {
                aux: "def2-universal-jkfit".into(),
            }
            .into(),
            threads: Some(4),
            ..Default::default()
        };
        let mut c = Session::new(mol.clone(), o).unwrap();
        c.energy().unwrap();
        let r = c.last().unwrap().clone();
        let adef = c.cx.plan.load_basis("def2-universal-jkfit").unwrap();
        let bdef = c.cx.plan.load_basis("def2-svp").unwrap();
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap();
        let par = Par::rayon(4);
        pool.install(|| {
            let ca = r.c_a.subcols(0, r.nalpha).to_owned();
            let cb = r.c_b.subcols(0, r.nbeta).to_owned();
            let restricted = r.restricted;
            let (occ, f): (Vec<&Mat<f64>>, Vec<f64>) = if restricted { (vec![&ca], vec![2.0]) } else { (vec![&ca, &cb], vec![1.0, 1.0]) };
            let n = ca.nrows();
            let dt: Vec<f64> = (0..n * n)
                .map(|t| occ.iter().zip(&f).map(|(cs, fs)| fs * (0..cs.ncols()).map(|i| cs[(t / n, i)] * cs[(t % n, i)]).sum::<f64>()).sum())
                .collect();
            let at = |positions: &[[f64; 3]]| {
                let b = BasisSet::new(&bdef, &mol.z, positions, true).unwrap();
                let a = BasisSet::new(&adef, &mol.z, positions, true).unwrap();
                (b, a)
            };
            // analytic pieces at the reference geometry
            let (b0, a0) = at(&mol.positions);
            let p0 = PairSet::build(&b0, PairOptions::with_order(DerivOrder::Two));
            let occr: Vec<MatRef<'_, f64>> = occ.iter().map(|m| m.as_ref()).collect();
            let mut t = Timers::default();
            let an = df_hessian_2e(&b0, &p0, &a0, &linus_jk::df::CoulombMetric::new(&a0, faer::Par::Seq).unwrap(), &occr, &f, Some(x), MemoryLimit::from_gib(None), par, &mut t);
            let natom = mol.natom();
            let n3 = 3 * natom;
            let h = 1e-3;
            let (mut eh, mut ef, mut hmax, mut fmax) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
            for col in 0..n3 {
                let (g, fx): (Vec<Vec<f64>>, Vec<Vec<Mat<f64>>>) = [h, -h, 2.0 * h, -2.0 * h]
                    .iter()
                    .map(|&d| {
                        let mut cc = mol.positions.clone();
                        cc[col / 3][col % 3] += d;
                        let (b, a) = at(&cc);
                        let p = PairSet::build(&b, PairOptions::with_order(DerivOrder::One));
                        let mut tt = Timers::default();
                        let g = linus_jk::df::gradient::df_gradient(&b, &p, &a, &linus_jk::df::CoulombMetric::new(&a, faer::Par::Seq).unwrap(), &dt, &occ, restricted, x, par, &mut tt);
                        // F C_s = (J[D_t] - K_w[D_s]) C_s at fixed D
                        let mut df = DfJk::new(&b, &p, &a, true, DfConfig { omega: Some(x.omega()), ..DfConfig::default() }, par);
                        assert!(df.set_exchange_operator(x));
                        let dts = linus_base::linalg::mat_from_rowmajor(n, n, &dt);
                        let jt = df.compute(&JkRequest::new(std::slice::from_ref(&dts), true, false), par).unwrap().j.remove(0);
                        let fs: Vec<Mat<f64>> = occ
                            .iter()
                            .map(|cs| {
                                let ds = linus_base::linalg::mm(cs.as_ref(), cs.transpose(), par);
                                let fac = [(*cs).clone()];
                                let k = df.compute(&JkRequest { dens: std::slice::from_ref(&ds), factors: Some(linus_base::terms::DensityFactors::Symmetric(&fac)), want_j: false, want_k: true }, par).unwrap().k.remove(0);
                                linus_base::linalg::mm((&jt - &k).as_ref(), cs.as_ref(), par)
                            })
                            .collect();
                        (g.iter().flat_map(|v| v.iter().cloned()).collect(), fs)
                    })
                    .unzip();
                let rich = |a: f64, b: f64, c: f64, d: f64| (8.0 * (a - b) - (c - d)) / (12.0 * h);
                for i in 0..n3 {
                    let v = rich(g[0][i], g[1][i], g[2][i], g[3][i]);
                    eh = eh.max((v - an.hess[i * n3 + col]).abs());
                    hmax = hmax.max(v.abs());
                }
                for s in 0..occ.len() {
                    let num = Mat::from_fn(n, occ[s].ncols(), |p, q| rich(fx[0][s][(p, q)], fx[1][s][(p, q)], fx[2][s][(p, q)], fx[3][s][(p, q)]));
                    ef = ef.max(linus_base::linalg::max_abs((&num - &an.fxo[col][s]).as_ref()));
                    fmax = fmax.max(linus_base::linalg::max_abs(num.as_ref()));
                }
            }
            println!("wK {charge} {mult}: explicit |dH| {eh:.2e} (|H| {hmax:.2e}), F^x C {ef:.2e} (|F^x C| {fmax:.2e})");
            assert!(eh < 2e-8 && ef < 2e-8, "wK {charge} {mult}: explicit |dH| {eh:e}, F^x C {ef:e}");
        });
    }
}

/// The wK terms reduce to the global-hybrid ones when the long-range part is Coulomb-like
/// (b = 0: a_x = a) — the J and polarization calls against the one-call code.
#[test]
fn range_separated_without_long_range_is_global_hybrid() {
    let mol = Structure::from_xyz_str(MOL, 0, 1).unwrap();
    let o = CalcOptions {
        method: Some("b3lyp".into()),
        basis: Some("def2-svp".into()),
        jk: JkMethod::Df {
            aux: "def2-universal-jkfit".into(),
        }
        .into(),
        threads: Some(4),
        ..Default::default()
    };
    let mut c = Session::new(mol, o).unwrap();
    c.energy().unwrap();
    let r = c.last().unwrap().clone();
    let b = &c.cx.basis;
    let pairs = PairSet::build(b, PairOptions::with_order(DerivOrder::Two));
    let aux = BasisSet::new(
        &c.cx.plan.load_basis("def2-universal-jkfit").unwrap(),
        &c.cx.mol.z,
        &c.cx.mol.positions,
        true,
    )
    .unwrap();
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap();
    let par = Par::rayon(4);
    pool.install(|| {
        let ca = r.c_a.subcols(0, r.nalpha);
        let mut t = Timers::default();
        let x = Operator::Attenuated {
            coulomb: 0.3,
            long_range: 0.0,
            omega: 0.4,
        };
        let wk = wk_hessian_2e(
            b,
            &pairs,
            &aux,
            &linus_jk::df::CoulombMetric::new(&aux, faer::Par::Seq).unwrap(),
            &[ca],
            &[2.0],
            x,
            MemoryLimit::from_gib(None),
            par,
            &mut t,
        );
        let gh = df_hessian_2e(
            b,
            &pairs,
            &aux,
            &linus_jk::df::CoulombMetric::new(&aux, faer::Par::Seq).unwrap(),
            &[ca],
            &[2.0],
            Some(Operator::Attenuated {
                coulomb: 0.3,
                long_range: 0.0,
                omega: 0.0,
            }),
            MemoryLimit::from_gib(None),
            par,
            &mut t,
        );
        let dh = max_diff(&wk.hess, &gh.hess);
        let df = wk
            .fxo
            .iter()
            .zip(&gh.fxo)
            .map(|(u, v)| linus_base::linalg::max_abs((&u[0] - &v[0]).as_ref()))
            .fold(0.0, linus_base::linalg::max_nan);
        assert!(dh < 1e-10 && df < 1e-10, "dH {dh:e} dF {df:e}");
    });
}

/// The exchange cross term is assembled in an orientation fixed by the atoms and each pass's
/// products are parts of the single pass's, so the H-bar atom blocks (one pass per block, their
/// size set by the memory budget) do not change any bit: every limit from one block down to one
/// atom per block (1, 2, 3 and 4 passes over the four atoms) gives the Hessian terms and F^x C_s
/// of the single pass bit for bit (closed and open shell; D43). At 16 geometries (atoms moved by
/// up to 1.5e-3 Å): a product taken by another of faer's kernels changes only last bits, which
/// the final sums round away at most geometries (0.1.0's CI failure was one where they did not).
#[test]
fn hbar_passes_do_not_change_the_bits() {
    let mut failures = Vec::new();
    for shift in 0..16 {
        let d = 1e-4 * shift as f64;
        let xyz = format!(
            "4\n\nO {} -0.05 0.12\nH 0.85 {} -0.03\nH -0.68 0.55 {}\nC 0.20 -1.30 0.60\n",
            0.10 + d,
            0.52 - 0.7 * d,
            0.05 + 0.3 * d
        );
        for (charge, mult) in [(0, 1), (1, 2)] {
            failures.extend(hbar_pass_differences(&xyz, charge, mult));
        }
    }
    assert!(failures.is_empty(), "{failures:#?}");
}

/// [`hbar_passes_do_not_change_the_bits`] for one molecule: the limits whose passes changed
/// bits of the single pass.
fn hbar_pass_differences(xyz: &str, charge: i32, mult: u32) -> Vec<String> {
    let mol = Structure::from_xyz_str(xyz, charge, mult).unwrap();
    let o = CalcOptions {
        method: Some("b3lyp".into()),
        basis: Some("def2-svp".into()),
        jk: JkMethod::Df {
            aux: "def2-universal-jkfit".into(),
        }
        .into(),
        threads: Some(4),
        ..Default::default()
    };
    let mut c = Session::new(mol, o).unwrap();
    c.energy().unwrap();
    let r = c.last().unwrap().clone();
    let b = &c.cx.basis;
    let pairs = PairSet::build(b, PairOptions::with_order(DerivOrder::Two));
    let aux = BasisSet::new(
        &c.cx.plan.load_basis("def2-universal-jkfit").unwrap(),
        &c.cx.mol.z,
        &c.cx.mol.positions,
        true,
    )
    .unwrap();
    let metric = linus_jk::df::CoulombMetric::new(&aux, Par::Seq).unwrap();
    let (ca, cb) = (r.c_a.subcols(0, r.nalpha), r.c_b.subcols(0, r.nbeta));
    let (occ, f): (Vec<MatRef<'_, f64>>, Vec<f64>) = if r.restricted {
        (vec![ca], vec![2.0])
    } else {
        (vec![ca, cb], vec![1.0, 1.0])
    };
    let xop = Some(Operator::Attenuated {
        coulomb: 0.2,
        long_range: 0.0,
        omega: 0.0,
    });
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap();
    let bits = |t: &DfHess2e| -> Vec<u64> {
        let mut v: Vec<u64> = t.hess.iter().map(|x| x.to_bits()).collect();
        for m in t.fxo.iter().flatten() {
            v.extend((0..m.ncols()).flat_map(|j| m.col(j).iter().map(|x| x.to_bits())));
        }
        v
    };
    // limits from the whole H-bar in one block (a few MB) down to every atom in its own block
    let mut single: Option<Vec<u64>> = None;
    let mut seen: Vec<usize> = Vec::new();
    let mut out = Vec::new();
    let mut limit = 1usize << 28;
    while limit > 1 << 12 && seen.len() < 4 {
        let mut t = Timers::default();
        let mem = MemoryLimit { limit: Some(limit) };
        let h = pool.install(|| {
            df_hessian_2e(
                b,
                &pairs,
                &aux,
                &metric,
                &occ,
                &f,
                xop,
                mem,
                Par::rayon(4),
                &mut t,
            )
        });
        let passes = t.resources.iter().find(|e| e.0 == "hess_df_passes");
        let passes = passes.expect("the passes are recorded").1 as usize;
        if !seen.contains(&passes) {
            seen.push(passes);
        }
        let v = bits(&h);
        match &single {
            None => {
                assert_eq!(
                    passes, 1,
                    "the first limit ({limit} B) gave {passes} passes"
                );
                single = Some(v);
            }
            Some(s) => {
                let nd = v.iter().zip(s).filter(|(x, y)| x != y).count();
                if nd > 0 {
                    out.push(format!(
                        "mult {mult}, {xyz:?}: {passes} passes (limit {limit} B) changed {nd} of {} values",
                        v.len()
                    ));
                }
            }
        }
        limit = limit / 5 * 4;
    }
    seen.sort_unstable();
    assert_eq!(
        seen,
        [1, 2, 3, 4],
        "mult {mult}: the limits gave passes {seen:?}"
    );
    out
}
