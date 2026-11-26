//! The device grid-K engine against a dense host reference: Kbar = sym(Xbar^T W V(Xbar D)) with
//! linus-ints' potential integrals point by point (no screening), for every class up to (gg)
//! (def2-QZVP), the Coulomb and a range-separated operator; bitwise run to run.
#![cfg(feature = "gpu")]

mod common;

use common::Rng;
use linus_gpu::cpu::{eval_ao, BasisSet, GridSpec, MolecularGrid, Operator, PotPair};
use linus_gpu::gridk::{GpuGridK, GpuGridKOptions, GridKTerm};
use linus_gpu::harness;
use linus_gpu::kernels::gridk::Style;
use std::sync::Arc;

/// The analytic overlap of `basis` (column-major).
fn overlap(basis: &BasisSet) -> Vec<f64> {
    let n = basis.nbf;
    let mut s = vec![0.0; n * n];
    linus_ints::OneEPlan::new(basis, linus_ints::DerivOrder::Zero)
        .overlap(&mut linus_ints::AoMatrixMut::strided(&mut s, n, 1, n));
    s
}

/// The engine's options: the fit on or off, the screening threshold.
fn opts(fit: bool, thresh: f64) -> GpuGridKOptions {
    GpuGridKOptions {
        lindep: 1e-8,
        fit,
        thresh,
        prim_cut: 1e-20,
        style: None,
    }
}

/// A dense host reference of the build (column-major nbf x nbf).
fn reference(
    basis: &BasisSet,
    grid: &MolecularGrid,
    xbar: &[f64],
    d: &[f64],
    op: &Operator,
) -> Vec<f64> {
    let (n, np) = (basis.nbf, grid.len());
    // Y = Xbar D (np x n)
    let mut y = vec![0.0; np * n];
    for q in 0..n {
        for p in 0..n {
            let dpq = d[p + q * n];
            if dpq == 0.0 {
                continue;
            }
            for g in 0..np {
                y[q * np + g] += xbar[p * np + g] * dpq;
            }
        }
    }
    // Z_a(g) = w_g sum_b V_ab(g) Y_b(g)
    let mut z = vec![0.0; np * n];
    let ns = basis.shells.len();
    let mut buf = Vec::new();
    for i in 0..ns {
        for j in 0..ns {
            let p = PotPair::new(basis, i, j, 1e-30);
            let (si, sj) = (&basis.shells[i], &basis.shells[j]);
            let nab = si.nfunc() * sj.nfunc();
            buf.resize(nab * np, 0.0);
            p.compute(op, &grid.x, &grid.y, &grid.z, 0.0, &mut buf);
            for ia in 0..si.nfunc() {
                for ib in 0..sj.nfunc() {
                    let a = si.ao_start + ia;
                    let b = sj.ao_start + ib;
                    let row = &buf[(ia * sj.nfunc() + ib) * np..(ia * sj.nfunc() + ib + 1) * np];
                    for g in 0..np {
                        z[a * np + g] += grid.w[g] * row[g] * y[b * np + g];
                    }
                }
            }
        }
    }
    // K = sym(Xbar^T Z)
    let mut k = vec![0.0; n * n];
    for q in 0..n {
        for p in 0..n {
            let mut s = 0.0;
            for g in 0..np {
                s += xbar[p * np + g] * z[q * np + g];
            }
            k[p + q * n] = s;
        }
    }
    let mut ks = k.clone();
    for q in 0..n {
        for p in 0..n {
            ks[p + q * n] = 0.5 * (k[p + q * n] + k[q + p * n]);
        }
    }
    ks
}

#[test]
fn builds_match_the_host_reference() {
    let Some(gpu) = common::gpu() else { return };
    let gpu = Arc::new(gpu);
    let a = 1.0 / linus_core::units::BOHR_ANGSTROM;
    let mol = harness::Structure::new(
        vec![8, 1, 1],
        vec![
            [0.0, 0.05 * a, 0.1173 * a],
            [0.03 * a, 0.7572 * a, -0.4692 * a],
            [-0.02 * a, -0.7572 * a, -0.4692 * a],
        ],
        0,
        1,
    )
    .unwrap();
    for basis_name in ["def2-svp", "def2-qzvp"] {
        let basis = harness::basis(&mol, basis_name).unwrap();
        let grid = harness::grid(&mol, &GridSpec::cosx(1)).unwrap();
        let n = basis.nbf;
        let s = overlap(&basis);
        // Xbar = X (no fit), a random symmetric density
        let mut rng = Rng(7);
        let mut d = vec![0.0; n * n];
        for q in 0..n {
            for p in 0..=q {
                let v = (rng.uniform() - 0.5) * 0.2;
                d[p + q * n] = v;
                d[q + p * n] = v;
            }
        }
        for op in [
            Operator::Coulomb,
            Operator::from_libxc(0.157706, 0.842294, 0.3),
        ] {
            let mut eng =
                GpuGridK::fitted(gpu.clone(), &basis, &grid, &s, opts(false, 0.0)).unwrap();
            eng.set_operator(&op);
            let ao = eng.xbar().unwrap();
            let k = eng.build(&d).unwrap();
            let k2 = eng.build(&d).unwrap();
            assert!(k.iter().zip(&k2).all(|(a, b)| a.to_bits() == b.to_bits()));
            let r = reference(&basis, &grid, &ao, &d, &op);
            let scale = r.iter().fold(0.0f64, |m, v| m.max(v.abs()));
            let err = k
                .iter()
                .zip(&r)
                .fold(0.0f64, |m, (a, b)| m.max((a - b).abs()));
            eprintln!(
                "{basis_name} {op:?}: max |K| {scale:.3e}, max |dK| {err:.3e}, {:?}",
                eng.times()
            );
            assert!(err < 1e-11 * scale.max(1.0), "{basis_name} {op:?}: {err:e}");
        }
    }
}

/// The device's overlap fit: Xbar^T W Xbar reproduces the analytic overlap, and without the
/// fit Xbar is the collocation matrix.
#[test]
fn the_device_fit_reproduces_the_overlap() {
    let Some(gpu) = common::gpu() else { return };
    let gpu = Arc::new(gpu);
    let mol = harness::structure("la18").unwrap();
    let basis = harness::basis(&mol, "def2-svp").unwrap();
    let grid = harness::grid(&mol, &GridSpec::cosx(1)).unwrap();
    let (n, np) = (basis.nbf, grid.len());
    let s = overlap(&basis);
    let eng = GpuGridK::fitted(gpu.clone(), &basis, &grid, &s, opts(true, 1e-12)).unwrap();
    eprintln!("fit {:?}", eng.fit());
    let xb = eng.xbar().unwrap();
    let mut err: f64 = 0.0;
    for q in 0..n {
        for p in 0..n {
            let v: f64 = (0..np)
                .map(|g| grid.w[g] * xb[p * np + g] * xb[q * np + g])
                .sum();
            err = err.max((v - s[p + q * n]).abs());
        }
    }
    assert!(err < 1e-10, "{err:e}");
    let raw = GpuGridK::fitted(gpu, &basis, &grid, &s, opts(false, 1e-12)).unwrap();
    let x = raw.xbar().unwrap();
    let shells: Vec<usize> = (0..basis.shells.len()).collect();
    let mut ao = vec![0.0; n * np];
    eval_ao(&basis, &shells, &grid.x, &grid.y, &grid.z, 0, &mut ao);
    let dx = x
        .iter()
        .zip(&ao)
        .fold(0.0f64, |m, (a, b)| m.max((a - b).abs()));
    assert!(dx < 1e-12, "{dx:e}");
}

/// A water molecule (a little off its symmetry).
fn water() -> harness::Structure {
    let a = 1.0 / linus_core::units::BOHR_ANGSTROM;
    harness::Structure::new(
        vec![8, 1, 1],
        vec![
            [0.0, 0.05 * a, 0.1173 * a],
            [0.03 * a, 0.7572 * a, -0.4692 * a],
            [-0.02 * a, -0.7572 * a, -0.4692 * a],
        ],
        0,
        1,
    )
    .unwrap()
}

/// A random symmetric matrix of order n.
fn symmetric(n: usize, seed: u64, scale: f64) -> Vec<f64> {
    let mut rng = Rng(seed);
    let mut d = vec![0.0; n * n];
    for q in 0..n {
        for p in 0..=q {
            let v = (rng.uniform() - 0.5) * scale;
            d[p + q * n] = v;
            d[q + p * n] = v;
        }
    }
    d
}

/// Every kernel style gives the default style's K to round-off, and the screening at the
/// production threshold (1e-12) stays within 1e-10 of the unscreened build.
#[test]
fn styles_and_screening_agree() {
    let Some(gpu) = common::gpu() else { return };
    let gpu = Arc::new(gpu);
    let mol = water();
    let basis = harness::basis(&mol, "def2-qzvp").unwrap();
    let grid = harness::grid(&mol, &GridSpec::cosx(1)).unwrap();
    let s = overlap(&basis);
    let d = symmetric(basis.nbf, 11, 0.2);
    let build = |style: Option<Style>, thresh: f64| {
        let mut o = opts(true, thresh);
        o.style = style;
        let mut eng = GpuGridK::fitted(gpu.clone(), &basis, &grid, &s, o).unwrap();
        eng.build(&d).unwrap()
    };
    let r = build(None, 0.0);
    let scale = r.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    let diff = |k: &[f64]| {
        k.iter()
            .zip(&r)
            .fold(0.0f64, |m, (a, b)| m.max((a - b).abs()))
    };
    for tag in ["-", "e", "z", "s", "ezs"] {
        let e = diff(&build(Some(Style::parse(tag)), 0.0));
        assert!(e < 1e-13 * scale.max(1.0), "style {tag}: {e:e}");
    }
    let e = diff(&build(None, 1e-12));
    assert!(e < 1e-10 * scale.max(1.0), "screened: {e:e}");
}

/// The device loop's term against the host's builds: K = K[D0] + Kbar[D - D0] and its energy
/// 1/2 D0 . K[D0] + D_d . K[D0] + 1/2 D_d . Kbar, for both spin channels of a state.
#[test]
fn the_device_term_matches_the_host_builds() {
    let Some(gpu) = common::gpu() else { return };
    let gpu = Arc::new(gpu);
    let mol = water();
    let basis = harness::basis(&mol, "def2-svp").unwrap();
    let grid = harness::grid(&mol, &GridSpec::cosx(1)).unwrap();
    let n = basis.nbf;
    let s = overlap(&basis);
    let d0 = symmetric(n, 3, 0.1);
    let k0 = symmetric(n, 5, 0.3);
    let dens: Vec<f64> = [symmetric(n, 7, 0.2), symmetric(n, 9, 0.2)].concat();
    let dot = |a: &[f64], b: &[f64]| a.iter().zip(b).map(|(x, y)| x * y).sum::<f64>();
    let e00 = 0.5 * dot(&d0, &k0);
    // the host's builds
    let mut host = GpuGridK::fitted(gpu.clone(), &basis, &grid, &s, opts(true, 1e-12)).unwrap();
    let mut want_k = Vec::new();
    let mut want_e = 0.0;
    for sp in 0..2 {
        let dd: Vec<f64> = dens[sp * n * n..(sp + 1) * n * n]
            .iter()
            .zip(&d0)
            .map(|(a, b)| a - b)
            .collect();
        let kbar = host.build(&dd).unwrap();
        want_e += e00 + dot(&dd, &k0) + 0.5 * dot(&dd, &kbar);
        want_k.extend(kbar.iter().zip(&k0).map(|(a, b)| a + b));
    }
    // the device loop's term
    let eng = GpuGridK::fitted(gpu.clone(), &basis, &grid, &s, opts(true, 1e-12)).unwrap();
    let mut term = GridKTerm::new(&gpu, eng, &d0, &k0, e00).unwrap();
    let dbuf = gpu.upload(&dens).unwrap();
    let mut kbuf = gpu.alloc_zeros::<f64>(2 * n * n).unwrap();
    term.exchange(&dbuf, 2, &mut kbuf).unwrap();
    let k = gpu.download(&kbuf).unwrap();
    let e = term.energy().unwrap();
    let scale = want_k.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    let err = k
        .iter()
        .zip(&want_k)
        .fold(0.0f64, |m, (a, b)| m.max((a - b).abs()));
    assert!(err < 1e-13 * scale, "K: {err:e}");
    assert!(
        (e - want_e).abs() < 1e-12 * want_e.abs().max(1.0),
        "E: {e} {want_e}"
    );
    // bitwise run to run
    term.exchange(&dbuf, 2, &mut kbuf).unwrap();
    let k2 = gpu.download(&kbuf).unwrap();
    assert!(k.iter().zip(&k2).all(|(a, b)| a.to_bits() == b.to_bits()));
    assert_eq!(term.energy().unwrap().to_bits(), e.to_bits());
}
