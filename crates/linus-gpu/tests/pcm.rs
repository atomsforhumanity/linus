//! The device PCM engine against linus-ints' McMurchie-Davidson references (no screening):
//! potentials at the points, potential matrices and the gradient, for Gaussian and point charges
//! over every Hermite degree (def2-QZVP's g shells give (gg), L = 8); bitwise run to run; the
//! charge solve against a dense host solve of the same K.
#![cfg(feature = "gpu")]

mod common;

use common::Rng;
use linus_gpu::cpu::{BasisSet, DerivOrder, GaussPoint, PairOptions, PairSet};
use linus_gpu::harness::{self, Structure};
use linus_gpu::pcm::solver::{GpuPcmSolver, PcmCharges, PcmKind, PcmSurface};
use linus_gpu::pcm::{GpuPcmInts, PcmIntsHost, PcmPoint};
use linus_ints::points::pcm_ints::SurfacePlan;
use linus_ints::reference::{oneshot, pcm};

fn water() -> Structure {
    let a = 1.0 / linus_core::units::BOHR_ANGSTROM;
    Structure::new(
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

/// `n` points per atom on spheres of 1.2 x 1.8 Å around the atoms (some inside other spheres:
/// the integrals do not care), Gaussian exponents 2..6 bohr^-1 or point charges.
fn points(mol: &Structure, n: usize, point_charges: bool, seed: u64) -> Vec<PcmPoint> {
    let mut rng = Rng(seed);
    let r = 1.2 * 1.8 / linus_core::units::BOHR_ANGSTROM;
    let mut out = Vec::new();
    for c in &mol.positions {
        for _ in 0..n {
            let (u, v) = (
                2.0 * rng.uniform() - 1.0,
                2.0 * std::f64::consts::PI * rng.uniform(),
            );
            let s = (1.0 - u * u).sqrt();
            let zeta = if point_charges {
                f64::INFINITY
            } else {
                2.0 + 4.0 * rng.uniform()
            };
            out.push(PcmPoint {
                r: [c[0] + r * s * v.cos(), c[1] + r * s * v.sin(), c[2] + r * u],
                zeta,
            });
        }
    }
    out
}

/// A symmetric density with entries over several decades.
fn density(n: usize, seed: u64) -> Vec<f64> {
    let mut rng = Rng(seed);
    let mut d = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..=i {
            let v = (rng.uniform() - 0.4) * 0.5 / (1.0 + (i as f64 - j as f64).abs());
            d[i * n + j] = v;
            d[j * n + i] = v;
        }
    }
    d
}

fn maxabs(a: &[f64]) -> f64 {
    a.iter().fold(0.0f64, |m, x| m.max(x.abs()))
}

fn maxdiff(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .fold(0.0f64, |m, (x, y)| m.max((x - y).abs()))
}

fn same_bits(a: &[f64], b: &[f64]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}

fn flat3(v: &[[f64; 3]]) -> Vec<f64> {
    v.iter().flatten().copied().collect()
}

/// Potential, matrix (separate and fused) and gradient against the CPU references.
fn check(mol: &Structure, basis_name: &str, npts: usize, point_charges: bool) {
    let Some(gpu) = common::gpu() else { return };
    let basis: BasisSet = harness::basis(mol, basis_name).unwrap();
    let nbf = basis.nbf;
    let pts = points(mol, npts, point_charges, 7);
    let gp: Vec<GaussPoint> = pts
        .iter()
        .map(|p| GaussPoint {
            r: p.r,
            zeta: p.zeta,
        })
        .collect();
    let np = pts.len();
    let d = density(nbf, 3);
    let mut rng = Rng(11);
    let q: Vec<f64> = (0..np).map(|_| rng.uniform() - 0.5).collect();
    // CPU references, unscreened
    let pairs = PairSet::build(&basis, PairOptions::with_order(DerivOrder::One));
    let v_ref = pcm::potential_points(&basis, &pairs, &gp, &[&d]).remove(0);
    let m_ref = pcm::potential_matrices(&basis, &pairs, &gp, &[&q]).remove(0);
    let surface = SurfacePlan::new(&basis, &pairs, &gp, 0.0);
    // (the gradient's points are coordinates of their own; the owners are not read)
    let owner = vec![0; np];
    let deriv = surface.deriv_plan(&pairs, &owner).unwrap();
    let (ga_ref, gc_ref) = oneshot::surface_deriv_gradient(&deriv, &q, &d);
    // device, every pair kept
    let host = PcmIntsHost::new(&basis, &pts, 0.0, true).unwrap();
    // every pair with a primitive above the cut
    assert_eq!(
        host.npairs_kept,
        pairs.pairs().iter().filter(|p| p.nprim > 0).count()
    );
    let mut eng = GpuPcmInts::new(&gpu, &host).unwrap();
    let dd = gpu.upload(&d).unwrap();
    let qd = gpu.upload(&q).unwrap();
    let mut v = gpu.alloc_zeros::<f64>(np).unwrap();
    let mut m = gpu.alloc_zeros::<f64>(nbf * nbf).unwrap();
    eng.pass(&gpu, Some((&dd, 1.0)), None, Some(&mut v), None, None)
        .unwrap();
    eng.pass(&gpu, None, None, None, Some(&qd), Some((&mut m, 1.0)))
        .unwrap();
    let (vh, mh) = (gpu.download(&v).unwrap(), gpu.download(&m).unwrap());
    let ev = maxdiff(&vh, &v_ref) / maxabs(&v_ref);
    let em = maxdiff(&mh, &m_ref) / maxabs(&m_ref);
    // fused, with a base and a negative scale: phi = base - v[D]; scaled matrix
    let base: Vec<f64> = (0..np).map(|g| 0.25 * g as f64).collect();
    let bd = gpu.upload(&base).unwrap();
    let mut v2 = gpu.alloc_zeros::<f64>(np).unwrap();
    let mut m2 = gpu.alloc_zeros::<f64>(nbf * nbf).unwrap();
    eng.pass(
        &gpu,
        Some((&dd, -1.0)),
        Some(&bd),
        Some(&mut v2),
        Some(&qd),
        Some((&mut m2, -2.0)),
    )
    .unwrap();
    let (v2h, m2h) = (gpu.download(&v2).unwrap(), gpu.download(&m2).unwrap());
    // exactly equal (the elements of pairs without primitives above the cut stay +0.0, where
    // the scaled reference has -0.0)
    let phi_ref: Vec<f64> = base.iter().zip(&vh).map(|(b, x)| b - x).collect();
    let m2_ref: Vec<f64> = mh.iter().map(|x| -2.0 * x).collect();
    assert!(
        v2h == phi_ref,
        "fused potential differs from the separate pass"
    );
    assert!(m2h == m2_ref, "fused matrix differs from the separate pass");
    // gradient
    let (ga, gc) = eng.gradient(&gpu, &dd, 1.0, &q).unwrap();
    let (fa, fc) = (flat3(&ga), flat3(&gc));
    let (fa_ref, fc_ref) = (flat3(&ga_ref), flat3(&gc_ref));
    let ea = maxdiff(&fa, &fa_ref) / maxabs(&fa_ref);
    let ec = maxdiff(&fc, &fc_ref) / maxabs(&fc_ref);
    // translational invariance: atoms + points sum to zero
    let mut tot = [0.0; 3];
    for x in ga.iter().chain(&gc) {
        for k in 0..3 {
            tot[k] += x[k];
        }
    }
    eprintln!(
        "{} {basis_name} {} points ({}): nbf {nbf}, pairs {}; rel. max err v {ev:.1e} V {em:.1e} \
         grad atoms {ea:.1e} points {ec:.1e}; sum {:.1e}",
        harness::formula(mol),
        np,
        if point_charges { "point" } else { "Gaussian" },
        host.npairs_kept,
        maxabs(&tot) / maxabs(&fa_ref)
    );
    assert!(ev < 1e-13 && em < 1e-13, "v {ev:e} V {em:e}");
    assert!(ea < 1e-12 && ec < 1e-12, "grad {ea:e} {ec:e}");
    assert!(maxabs(&tot) < 1e-12 * maxabs(&fa_ref));
    // run to run
    let mut v3 = gpu.alloc_zeros::<f64>(np).unwrap();
    let mut m3 = gpu.alloc_zeros::<f64>(nbf * nbf).unwrap();
    eng.pass(
        &gpu,
        Some((&dd, -1.0)),
        Some(&bd),
        Some(&mut v3),
        Some(&qd),
        Some((&mut m3, -2.0)),
    )
    .unwrap();
    assert!(same_bits(&gpu.download(&v3).unwrap(), &v2h));
    assert!(same_bits(&gpu.download(&m3).unwrap(), &m2h));
    let (ga2, gc2) = eng.gradient(&gpu, &dd, 1.0, &q).unwrap();
    assert!(same_bits(&flat3(&ga2), &fa) && same_bits(&flat3(&gc2), &fc));
}

#[test]
fn water_svp_gaussian_charges() {
    check(&water(), "def2-svp", 40, false);
}

#[test]
fn water_qzvp_gaussian_and_point_charges() {
    check(&water(), "def2-qzvp", 30, false);
    check(&water(), "def2-qzvp", 30, true);
}

#[test]
fn fragment_tzvp_gaussian_charges() {
    let mut m = harness::structure("u1an39").unwrap();
    m.z.truncate(6);
    m.positions.truncate(6);
    check(&m, "def2-tzvp", 50, false);
}

/// The dense host solve of K x = b (Gaussian elimination with partial pivoting).
fn host_solve(k: &[f64], n: usize, b: &[f64], trans: bool) -> Vec<f64> {
    let mut a: Vec<f64> = (0..n * n)
        .map(|x| {
            let (i, j) = (x / n, x % n);
            if trans {
                k[i * n + j]
            } else {
                k[j * n + i]
            }
        })
        .collect(); // a[i][j] row-major from column-major k (or its transpose)
    let mut x = b.to_vec();
    for c in 0..n {
        let p = (c..n)
            .max_by(|&i, &j| a[i * n + c].abs().total_cmp(&a[j * n + c].abs()))
            .unwrap();
        if p != c {
            for j in 0..n {
                a.swap(c * n + j, p * n + j);
            }
            x.swap(c, p);
        }
        for i in c + 1..n {
            let f = a[i * n + c] / a[c * n + c];
            for j in c..n {
                a[i * n + j] -= f * a[c * n + j];
            }
            x[i] -= f * x[c];
        }
    }
    for c in (0..n).rev() {
        let mut s = x[c];
        for j in c + 1..n {
            s -= a[c * n + j] * x[j];
        }
        x[c] = s / a[c * n + c];
    }
    x
}

fn erf_r(z: f64, r: f64) -> (f64, f64) {
    // erf(z r)/r and its derivative, through the device's formula's terms
    let x = z * r;
    let c = 2.0 * z / std::f64::consts::PI.sqrt();
    let e = libm_erf(x);
    let g = c * (-x * x).exp();
    (e / r, g / r - e / (r * r))
}

/// erf by its Taylor series (|x| small) or the complementary continued fraction; enough for
/// a 1e-12 reference.
fn libm_erf(x: f64) -> f64 {
    if x < 2.5 {
        let mut s = 0.0f64;
        let mut t = x;
        let mut k = 0.0;
        while t.abs() > 1e-18 * s.abs().max(1e-300) || k < 3.0 {
            s += t / (2.0 * k + 1.0);
            k += 1.0;
            t *= -x * x / k;
        }
        2.0 / std::f64::consts::PI.sqrt() * s
    } else {
        // erfc continued fraction (Lentz)
        let mut f = x;
        let mut c = x;
        let mut d = 0.0;
        for k in 1..200 {
            let a = k as f64 * 0.5;
            d = x + a * d;
            d = 1.0 / d;
            c = x + a / c;
            let del = c * d;
            f *= del;
            if (del - 1.0).abs() < 1e-16 {
                break;
            }
        }
        1.0 - (-x * x).exp() / (std::f64::consts::PI.sqrt() * f)
    }
}

/// The charge solve against the host: K from the same formulas, q_sym of a potential.
#[test]
fn charge_solve_against_a_dense_host_solve() {
    let Some(gpu) = common::gpu() else { return };
    let mol = water();
    let mut rng = Rng(5);
    let n = 180;
    let mut surf = PcmSurface::default();
    for i in 0..n {
        let c = &mol.positions[i % 3];
        let rad = 2.5 + 0.5 * (i % 3) as f64;
        let (u, v) = (
            2.0 * rng.uniform() - 1.0,
            2.0 * std::f64::consts::PI * rng.uniform(),
        );
        let s = (1.0 - u * u).sqrt();
        let nrm = [s * v.cos(), s * v.sin(), u];
        surf.r.push([
            c[0] + rad * nrm[0],
            c[1] + rad * nrm[1],
            c[2] + rad * nrm[2],
        ]);
        surf.normal.push(nrm);
        surf.zeta.push(3.0 + 2.0 * rng.uniform());
        surf.switch.push(0.5 + 0.5 * rng.uniform());
        surf.area.push(0.05 + 0.1 * rng.uniform());
        surf.radius.push(rad);
    }
    let phi: Vec<f64> = (0..n).map(|_| rng.uniform() - 0.3).collect();
    // host matrices (column-major)
    let pi = std::f64::consts::PI;
    let mut s = vec![0.0; n * n];
    let mut d = vec![0.0; n * n];
    for j in 0..n {
        for i in 0..n {
            let (zi, zj) = (surf.zeta[i], surf.zeta[j]);
            if i == j {
                s[j * n + i] = zi * (2.0 / pi).sqrt() / surf.switch[i];
                d[j * n + i] = -zi * (2.0 / pi).sqrt() / (2.0 * surf.radius[i]);
                continue;
            }
            let zij = zi * zj / (zi * zi + zj * zj).sqrt();
            let u: Vec<f64> = (0..3).map(|k| surf.r[i][k] - surf.r[j][k]).collect();
            let r = (u[0] * u[0] + u[1] * u[1] + u[2] * u[2]).sqrt();
            let (sv, s1) = erf_r(zij, r);
            s[j * n + i] = sv;
            let un: f64 = (0..3).map(|k| u[k] * surf.normal[j][k]).sum();
            d[j * n + i] = -s1 * un / r;
        }
    }
    let f = 0.75;
    let matvec = |m: &[f64], x: &[f64], trans: bool| -> Vec<f64> {
        (0..n)
            .map(|i| {
                (0..n)
                    .map(|j| if trans { m[i * n + j] } else { m[j * n + i] } * x[j])
                    .sum()
            })
            .collect()
    };
    for kind in [PcmKind::Conductor, PcmKind::Ief, PcmKind::Ssvpe] {
        let das: Vec<f64> = {
            // (D A S)_ij = sum_k D_ik A_k S_kj
            let mut m = vec![0.0; n * n];
            for j in 0..n {
                for i in 0..n {
                    m[j * n + i] = (0..n)
                        .map(|k| d[k * n + i] * surf.area[k] * s[j * n + k])
                        .sum();
                }
            }
            m
        };
        let k: Vec<f64> = (0..n * n)
            .map(|x| {
                let (j, i) = (x / n, x % n);
                match kind {
                    PcmKind::Conductor => s[x],
                    PcmKind::Ief => s[x] - f / (2.0 * pi) * das[x],
                    PcmKind::Ssvpe => s[x] - f / (4.0 * pi) * (das[x] + das[i * n + j]),
                }
            })
            .collect();
        let y = host_solve(&k, n, &phi, true);
        let qsym: Vec<f64> = match kind {
            PcmKind::Conductor => y.iter().map(|v| -f * v).collect(),
            _ => {
                let a_phi: Vec<f64> = phi.iter().zip(&surf.area).map(|(p, a)| p * a).collect();
                let dap = matvec(&d, &a_phi, false);
                let rphi: Vec<f64> = (0..n)
                    .map(|i| -f * (phi[i] - dap[i] / (2.0 * pi)))
                    .collect();
                let q = host_solve(&k, n, &rphi, false);
                let dty = matvec(&d, &y, true);
                (0..n)
                    .map(|i| 0.5 * (q[i] - f * (y[i] - surf.area[i] * dty[i] / (2.0 * pi))))
                    .collect()
            }
        };
        let mut sol = GpuPcmSolver::new(&gpu, kind, f, &surf).unwrap();
        let pd = gpu.upload(&phi).unwrap();
        let mut ch = PcmCharges::new(&gpu, n).unwrap();
        sol.solve(&gpu, &pd, &mut ch).unwrap();
        let qs = gpu.download(&ch.q_sym).unwrap();
        let e = maxdiff(&qs[..n], &qsym) / maxabs(&qsym);
        eprintln!("{kind:?}: q_sym max rel. err {e:.1e}");
        assert!(e < 1e-11, "{kind:?}: {e:e}");
        // run to run (a new factorization too)
        let mut sol2 = GpuPcmSolver::new(&gpu, kind, f, &surf).unwrap();
        let mut ch2 = PcmCharges::new(&gpu, n).unwrap();
        sol2.solve(&gpu, &pd, &mut ch2).unwrap();
        assert!(same_bits(&gpu.download(&ch2.q_sym).unwrap(), &qs));
    }
}
