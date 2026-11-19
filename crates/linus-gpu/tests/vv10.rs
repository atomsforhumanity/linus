//! The device VV10 engines against a dense CPU reference: densities from linus-ints'
//! collocation of every function at every point, the VV10 pair sums as a plain double loop
//! (PySCF's `_vv10nlc` in Rust), the potential matrix by direct quadrature. Closed and open
//! shells (an open shell's total density equals the closed-shell channel of all its orbitals at
//! half occupation), the three pair arithmetics at their own tolerances, run to run; the
//! gradient's translational invariance and its arithmetics against each other.
#![cfg(feature = "gpu")]

mod common;

use common::Rng;
use linus_gpu::cpu::{eval_ao, GridSpec};
use linus_gpu::harness::{self, Structure};
use linus_gpu::xc::vv10::{
    constants, GpuVv10, GpuVv10Grad, GpuVv10GradOptions, GpuVv10Options, GpuVv10Resp, Vv10Math,
};
use linus_gpu::xc::XcChannel;
use linus_gpu::GpuError;

fn water() -> Structure {
    let a = 1.0 / linus_core::units::BOHR_ANGSTROM;
    Structure::new(
        vec![8, 1, 1],
        vec![
            [0.0, 0.0, 0.1173 * a],
            [0.0, 0.7572 * a, -0.4692 * a],
            [0.0, -0.7572 * a, -0.4692 * a],
        ],
        0,
        1,
    )
    .unwrap()
}

/// Orbital-like factors: C with columns of decaying random coefficients, occupations 1.
fn channel(nbf: usize, nc: usize, seed: u64) -> (Vec<f64>, usize, Vec<f64>) {
    let mut rng = Rng(seed);
    let c = (0..nbf * nc)
        .map(|k| (rng.uniform() - 0.3) * (-((k % nbf) as f64) / 12.0).exp())
        .collect();
    (c, nc, vec![1.0; nc])
}

/// (E_nl, V_nl) of the total density sum_s sum_i occ C C^T (times 2 for `scale2`) by direct
/// quadrature.
fn reference(
    mol: &Structure,
    grid: &linus_gpu::cpu::MolecularGrid,
    ch: &[(Vec<f64>, usize, Vec<f64>)],
    scale2: bool,
) -> (f64, Vec<f64>) {
    let basis = harness::basis(mol, "def2-svp").unwrap();
    let nbf = basis.nbf;
    let np = grid.x.len();
    let shells: Vec<usize> = (0..basis.shells.len()).collect();
    let mut ao = vec![0.0; 4 * nbf * np];
    eval_ao(&basis, &shells, &grid.x, &grid.y, &grid.z, 1, &mut ao);
    let phi = |d: usize, m: usize, g: usize| ao[(d * nbf + m) * np + g];
    let sc = if scale2 { 2.0 } else { 1.0 };
    let mut rho = vec![0.0; np];
    let mut grad = vec![[0.0; 3]; np];
    for (c, nc, occ) in ch {
        for g in 0..np {
            for i in 0..*nc {
                let psi: Vec<f64> = (0..4)
                    .map(|d| (0..nbf).map(|m| phi(d, m, g) * c[i * nbf + m]).sum())
                    .collect();
                rho[g] += sc * occ[i] * psi[0] * psi[0];
                for (k, gk) in grad[g].iter_mut().enumerate() {
                    *gk += sc * 2.0 * occ[i] * psi[0] * psi[1 + k];
                }
            }
        }
    }
    let p = linus_xc::Functional::new("wb97m-v", false)
        .unwrap()
        .vv10()
        .unwrap();
    let (beta, kvv) = constants(&p);
    let pi43 = 4.0 * std::f64::consts::PI / 3.0;
    let act: Vec<usize> = (0..np).filter(|&g| rho[g] >= 1e-8).collect();
    let sig = |g: usize| grad[g][0].powi(2) + grad[g][1].powi(2) + grad[g][2].powi(2);
    let om = |g: usize| (p.c * (sig(g) / rho[g].powi(2)).powi(2) + pi43 * rho[g]).sqrt();
    let ka = |g: usize| kvv * rho[g].powf(1.0 / 6.0);
    let (mut e, mut vr, mut vs) = (0.0, vec![0.0; np], vec![0.0; np]);
    for &i in &act {
        let (mut f, mut u, mut w) = (0.0, 0.0, 0.0);
        for &j in &act {
            let r2 = (grid.x[i] - grid.x[j]).powi(2)
                + (grid.y[i] - grid.y[j]).powi(2)
                + (grid.z[i] - grid.z[j]).powi(2);
            let gp = r2 * om(j) + ka(j);
            let g = r2 * om(i) + ka(i);
            let gt = g + gp;
            let t = rho[j] * grid.w[j] / (g * gp * gt);
            f += t;
            let t = t * (1.0 / g + 1.0 / gt);
            u += t;
            w += t * r2;
        }
        f *= -1.5;
        let r = rho[i];
        let wt = p.c * (sig(i) / (r * r)).powi(2);
        e += grid.w[i] * r * (beta + 0.5 * f);
        vr[i] = beta + f + 1.5 * (u * ka(i) / 6.0 + w * (0.5 * pi43 * r - 2.0 * wt) / om(i));
        vs[i] = 1.5 * w * p.c * sig(i) / (r * r * r * om(i));
    }
    let mut v = vec![0.0; nbf * nbf];
    for g in 0..np {
        let wg = grid.w[g];
        for n in 0..nbf {
            for m in 0..nbf {
                let (pm, pn) = (phi(0, m, g), phi(0, n, g));
                let mut x = vr[g] * pm * pn;
                for (k, gk) in grad[g].iter().enumerate() {
                    x += 2.0 * vs[g] * gk * (phi(1 + k, m, g) * pn + pm * phi(1 + k, n, g));
                }
                v[n * nbf + m] += wg * x;
            }
        }
    }
    (e * p.scale, v.iter().map(|x| x * p.scale).collect())
}

#[test]
fn vv10_matches_a_dense_reference() {
    let Some(gpu) = common::gpu() else { return };
    let mol = water();
    let basis = harness::basis(&mol, "def2-svp").unwrap();
    let grid = harness::grid(&mol, &GridSpec::pyscf(0)).unwrap();
    let nbf = basis.nbf;
    let a = channel(nbf, 5, 3);
    let b = channel(nbf, 4, 9);
    // closed shell: the channel at twice its density; open shell: both channels
    let (e_rks, v_rks) = reference(&mol, &grid, std::slice::from_ref(&a), true);
    let (e_uks, v_uks) = reference(&mol, &grid, &[a.clone(), b.clone()], false);
    let chan = |x: &(Vec<f64>, usize, Vec<f64>)| {
        let (c, n, o) = x;
        (c.clone(), *n, o.clone())
    };
    let (ca, cb) = (chan(&a), chan(&b));
    for (math, tol) in [
        (Vv10Math::Fp64, 1e-12),
        (Vv10Math::DoubleFloat, 1e-12),
        (Vv10Math::Fp32, 1e-5),
    ] {
        let mut nl = GpuVv10::new(
            &gpu,
            &basis,
            &grid,
            "wb97m-v",
            &GpuVv10Options {
                func_thresh: 1e-15,
                math,
                ..GpuVv10Options::default()
            },
        )
        .unwrap();
        for (restricted, e_ref, v_ref) in [(true, e_rks, &v_rks), (false, e_uks, &v_uks)] {
            let chs: Vec<XcChannel> = if restricted {
                vec![XcChannel {
                    c: &ca.0,
                    ncol: ca.1,
                    occ: &ca.2,
                }]
            } else {
                vec![
                    XcChannel {
                        c: &ca.0,
                        ncol: ca.1,
                        occ: &ca.2,
                    },
                    XcChannel {
                        c: &cb.0,
                        ncol: cb.1,
                        occ: &cb.2,
                    },
                ]
            };
            let (e, v) = nl.compute(&gpu, &chs, restricted).unwrap();
            let (e2, v2) = nl.compute(&gpu, &chs, restricted).unwrap();
            assert!(
                e.to_bits() == e2.to_bits() && v == v2,
                "{math:?}: runs differ"
            );
            let vmax = v_ref.iter().fold(0.0f64, |m, x| m.max(x.abs()));
            let dv = v
                .iter()
                .zip(v_ref)
                .fold(0.0f64, |m, (x, y)| m.max((x - y).abs()));
            println!(
                "{math:?} restricted {restricted}: E {e:.12} (ref {e_ref:.12}, |dE| {:.1e}), |dV| \
                 {dv:.1e} of {vmax:.1e}, active {} of {}",
                (e - e_ref).abs(),
                nl.last.pairs.active,
                grid.x.len()
            );
            assert!(
                (e - e_ref).abs() <= tol * e_ref.abs(),
                "{math:?}: E {e} vs {e_ref}"
            );
            assert!(dv <= tol * vmax, "{math:?}: V {dv:e}");
        }
    }
}

/// The gradient's arithmetics agree (FP64 and double-float to round-off, FP32 loosely), are
/// translationally invariant with the grid response, and bitwise run to run.
#[test]
fn vv10_gradient_arithmetics_and_invariance() {
    let Some(gpu) = common::gpu() else { return };
    let mol = water();
    let basis = harness::basis(&mol, "def2-svp").unwrap();
    for spec in [GridSpec::pyscf(0), GridSpec::linus(1)] {
        let grid = harness::grid(&mol, &spec).unwrap();
        let nbf = basis.nbf;
        let (c, nc, _) = channel(nbf, 5, 3);
        // D = 2 C C^T (column-major, symmetric)
        let mut d = vec![0.0; nbf * nbf];
        for i in 0..nc {
            for m in 0..nbf {
                for n in 0..nbf {
                    d[m * nbf + n] += 2.0 * c[i * nbf + m] * c[i * nbf + n];
                }
            }
        }
        let run = |math: Vv10Math, gr: bool| {
            let mut g = GpuVv10Grad::new(
                &gpu,
                &basis,
                &grid,
                &spec,
                "b97m-v",
                &GpuVv10GradOptions {
                    math,
                    ..GpuVv10GradOptions::default()
                },
            )
            .unwrap();
            let a = g.compute(&gpu, &d, gr).unwrap();
            let b = g.compute(&gpu, &d, gr).unwrap();
            assert_eq!(a, b, "{math:?}: runs differ");
            a
        };
        let g64 = run(Vv10Math::Fp64, true);
        let gmax = g64.iter().flatten().fold(0.0f64, |m, x| m.max(x.abs()));
        for k in 0..3 {
            let s: f64 = g64.iter().map(|a| a[k]).sum();
            assert!(
                s.abs() < 1e-9 * gmax.max(1.0),
                "translational invariance {k}: {s:e}"
            );
        }
        let diff = |a: &[[f64; 3]], b: &[[f64; 3]]| {
            a.iter()
                .flatten()
                .zip(b.iter().flatten())
                .fold(0.0f64, |m, (x, y)| m.max((x - y).abs()))
        };
        let ddf = diff(&g64, &run(Vv10Math::DoubleFloat, true));
        let d32 = diff(&g64, &run(Vv10Math::Fp32, true));
        let g0 = run(Vv10Math::Fp64, false);
        println!(
            "{:?}: |G| {gmax:.2e}, double-float {ddf:.1e}, FP32 {d32:.1e}, grid response changes \
             it by {:.1e}",
            spec.partition,
            diff(&g64, &g0)
        );
        assert!(ddf < 1e-13, "double-float gradient {ddf:e}");
        assert!(d32 < 1e-7, "FP32 gradient {d32:e}");
        assert!(diff(&g64, &g0) > 0.0);
    }
}

/// The response engine's arithmetic is FP64 or double-float; FP32 is a per-call choice of
/// `sweep`, so an engine asked for FP32 is a contract error (it ran double-float without a word).
#[test]
fn the_response_engine_refuses_fp32_as_its_arithmetic() {
    let Some(gpu) = common::gpu() else { return };
    let v = [0.5; 4];
    let pts = [&v[..], &v[..], &v[..], &v[..], &v[..]];
    assert!(matches!(
        GpuVv10Resp::new(&gpu, pts, 4, Vv10Math::Fp32),
        Err(GpuError::Contract(_))
    ));
    for m in [Vv10Math::Fp64, Vv10Math::DoubleFloat] {
        assert_eq!(GpuVv10Resp::new(&gpu, pts, 4, m).unwrap().math(), m);
    }
}
