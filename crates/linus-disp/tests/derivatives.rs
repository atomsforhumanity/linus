// SPDX-License-Identifier: LGPL-3.0-or-later
//! Analytic gradients and Hessians against 5-point finite differences, and bitwise
//! independence of the thread count.

mod common;
use common::*;
use linus_disp::*;

const CASES: [(&str, &str, bool); 4] = [
    ("d3bj", "b3lyp", false),
    ("d3bj", "b3lyp", true),
    ("d4", "b3lyp", true),
    ("d4", "r2scan-3c", true),
];

#[test]
fn finite_differences() {
    let h = 1e-3;
    let mut worst_g: f64 = 0.0;
    let mut worst_h: f64 = 0.0;
    for name in ["hcn_cl", "cu-nh3-4", "cisplatin", "gdcl3_h2o"] {
        let fx = fixture(name);
        let (z, xyz) = (numbers(&fx), positions(&fx));
        let charge = fx["charge"].as_f64().unwrap();
        let n3 = 3 * z.len();
        for (model, func, atm) in CASES {
            let r = eval(model, func, atm, &z, &xyz, charge, Order::Hessian);
            let hess = r.hessian.unwrap();
            for c in 0..n3 {
                let at = |s: f64| {
                    let mut x = xyz.clone();
                    x[c / 3][c % 3] += s * h;
                    eval(model, func, atm, &z, &x, charge, Order::Gradient)
                };
                let (m2, m1, p1, p2) = (at(-2.0), at(-1.0), at(1.0), at(2.0));
                let fd = |f: &dyn Fn(&DispResult) -> f64| {
                    (f(&m2) - 8.0 * f(&m1) + 8.0 * f(&p1) - f(&p2)) / (12.0 * h)
                };
                worst_g = worst_g.max((fd(&|d| d.energy) - r.gradient[c / 3][c % 3]).abs());
                for row in 0..n3 {
                    let v = fd(&|d| d.gradient[row / 3][row % 3]);
                    worst_h = worst_h.max((v - hess[row * n3 + c]).abs());
                }
            }
        }
    }
    eprintln!("gradient vs FD {worst_g:.1e}, Hessian vs FD {worst_h:.1e}");
    assert!(worst_g < 1e-11);
    assert!(worst_h < 1e-9);
}

#[test]
fn thread_count_independent() {
    let fx = fixture("la18");
    let (z, xyz) = (numbers(&fx), positions(&fx));
    for (model, func, atm) in CASES {
        let mut first: Option<DispResult> = None;
        for threads in [1, 3, 8, 17] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            let r = pool.install(|| eval(model, func, atm, &z, &xyz, 0.0, Order::Hessian));
            match &first {
                None => first = Some(r),
                Some(f) => {
                    assert_eq!(
                        f.energy.to_bits(),
                        r.energy.to_bits(),
                        "{model} {func} energy, {threads} threads"
                    );
                    assert!(
                        flat(&f.gradient)
                            .iter()
                            .zip(flat(&r.gradient))
                            .all(|(a, b)| a.to_bits() == b.to_bits()),
                        "gradient"
                    );
                    let (ha, hb) = (f.hessian.as_ref().unwrap(), r.hessian.as_ref().unwrap());
                    assert!(
                        ha.iter().zip(hb).all(|(a, b)| a.to_bits() == b.to_bits()),
                        "{model} {func} Hessian, {threads} threads"
                    );
                }
            }
        }
    }
}

#[test]
fn hessian_symmetric_and_translation_invariant() {
    let fx = fixture("cisplatin");
    let (z, xyz) = (numbers(&fx), positions(&fx));
    let n3 = 3 * z.len();
    for (model, func, atm) in CASES {
        let r = eval(model, func, atm, &z, &xyz, 0.0, Order::Hessian);
        let h = r.hessian.unwrap();
        let scale = h.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        let mut asym: f64 = 0.0;
        let mut trans: f64 = 0.0;
        for i in 0..n3 {
            for j in 0..n3 {
                asym = asym.max((h[i * n3 + j] - h[j * n3 + i]).abs());
            }
            for x in 0..3 {
                let s: f64 = (0..z.len()).map(|a| h[i * n3 + 3 * a + x]).sum();
                trans = trans.max(s.abs());
            }
        }
        let gsum: [f64; 3] = std::array::from_fn(|x| r.gradient.iter().map(|g| g[x]).sum::<f64>());
        assert!(
            asym < 1e-14 * scale.max(1.0),
            "{model} {func}: asymmetry {asym:e}"
        );
        assert!(
            trans < 1e-13 * scale.max(1.0),
            "{model} {func}: translation {trans:e}"
        );
        assert!(
            gsum.iter().all(|v| v.abs() < 1e-15),
            "{model} {func}: gradient sum {gsum:?}"
        );
    }
}

#[test]
fn errors() {
    assert!(matches!(
        d3bj(
            &[104],
            &[[0.0; 3]],
            &D3Params::new(1.0, 1.0, 0.4, 5.0),
            &D3Options::default()
        ),
        Err(DispError::UnsupportedElement { .. })
    ));
    assert!(matches!(
        d4(
            &[1, 104],
            &[[0.0; 3], [0.0, 0.0, 1.4]],
            0.0,
            &D4Params::new(1.0, 1.0, 0.4, 5.0),
            &D4Options::default()
        ),
        Err(DispError::UnsupportedElement { .. })
    ));
    assert!(matches!(
        d3bj_params("wb97x-v"),
        Err(DispError::NoParameters { .. })
    ));
    assert!(matches!(
        d3bj(
            &[1],
            &[],
            &D3Params::new(1.0, 1.0, 0.4, 5.0),
            &D3Options::default()
        ),
        Err(DispError::Shape(_))
    ));
    // single atom and empty molecule
    let r = d3bj(
        &[6],
        &[[0.0; 3]],
        &d3bj_params("pbe").unwrap(),
        &D3Options {
            order: Order::Hessian,
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(r.energy, 0.0);
    let r = d4(
        &[],
        &[],
        0.0,
        &d4_params("pbe").unwrap(),
        &D4Options {
            order: Order::Hessian,
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(r.energy, 0.0);
}
