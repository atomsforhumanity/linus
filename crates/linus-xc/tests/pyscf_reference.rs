//! Compare linus-xc with PySCF's `eval_xc` (fixtures from gen_reference.py).
//!
//! For every case (functional x spin) we evaluate orders 0, 1 and 2 and compare
//! each array an order produces with PySCF's order-2 output. Element error:
//! `|ours - ref| / max(|ref|, 1e-10 * max|ref|)` (max over the compared points
//! of that array), so entries ten orders below the array's scale count
//! absolutely. NaN must match NaN.
//!
//! * `pyscf_libxc700_reference.json`: PySCF 2.15 running libxc 7.0.0, the same
//!   libxc as ours. Every point, tolerance 1e-13 (observed: bit-identical).
//! * `pyscf_reference.json`: stock PySCF 2.15, which bundles libxc 7.1.2. That
//!   release regenerated all functional code (SymPy backend, cancellation-free
//!   forms, no pow() for rational powers), so values differ by version drift,
//!   not by our wrapper. Checked on bulk points only (every spin density at
//!   least 1e-4, every sigma_ss positive) with 1e-9 (orders 0-1) and 1e-6
//!   (order 2); edge points (empty spin channel, sigma = 0, rho < 1e-4) are
//!   reported, not checked.
//!
//! `LINUS_XC_VALIDATION_TABLE=<dir>` writes both per-case tables there.

use linus_xc::{Family, Functional, Out, XcInput, XcOutput};
use serde::Deserialize;
use std::collections::BTreeMap;

#[derive(Deserialize)]
struct Fixture {
    pyscf_version: String,
    libxc_version: String,
    inputs: BTreeMap<String, Inputs>,
    cases: Vec<Case>,
}

#[derive(Deserialize)]
struct Inputs {
    np: usize,
    rho: Vec<f64>,
    sigma: Vec<f64>,
    tau: Vec<f64>,
}

#[derive(Deserialize)]
struct Case {
    name: String,
    xctype: String,
    spin: String,
    hybrid_coeff: f64,
    rsh_coeff_omega_alpha_beta: [f64; 3],
    nlc_b_c_weight: Option<[f64; 3]>,
    /// null = NaN
    out: BTreeMap<String, Vec<Option<f64>>>,
}

fn load(file: &str) -> Fixture {
    let path = format!("{}/tests/data/{file}", env!("CARGO_MANIFEST_DIR"));
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

impl Inputs {
    fn input(&self) -> XcInput<'_> {
        XcInput {
            rho: &self.rho,
            sigma: Some(&self.sigma),
            lapl: None,
            tau: Some(&self.tau),
        }
    }
    /// Bulk point: every spin density >= 1e-4 and every sigma_ss > 0.
    fn bulk(&self, polarized: bool) -> Vec<bool> {
        let (nr, ns) = if polarized { (2, 3) } else { (1, 1) };
        (0..self.np)
            .map(|i| {
                let r = &self.rho[i * nr..(i + 1) * nr];
                let s = &self.sigma[i * ns..(i + 1) * ns];
                r.iter().all(|&x| x >= 1e-4) && s.iter().step_by(2).all(|&x| x > 0.0)
            })
            .collect()
    }
}

/// Max error over the points where `keep` is true (see module docs).
fn max_rel_err(ours: &[f64], reference: &[Option<f64>], dim: usize, keep: &[bool]) -> f64 {
    assert_eq!(ours.len(), reference.len());
    let pairs = || {
        ours.iter()
            .zip(reference)
            .enumerate()
            .filter(|(i, _)| keep[i / dim])
            .map(|(_, (a, b))| (*a, b.unwrap_or(f64::NAN)))
    };
    let scale = pairs()
        .filter(|(_, b)| b.is_finite())
        .fold(0.0f64, |m, (_, b)| m.max(b.abs()));
    let floor = (1e-10 * scale).max(f64::MIN_POSITIVE);
    pairs()
        .map(|(a, b)| match (a.is_nan(), b.is_nan()) {
            (true, true) => 0.0,
            (false, false) => (a - b).abs() / b.abs().max(floor),
            _ => f64::INFINITY,
        })
        .fold(0.0, f64::max)
}

fn family_name(f: Family) -> &'static str {
    match f {
        Family::Hf => "HF",
        Family::Lda => "LDA",
        Family::Gga => "GGA",
        Family::Mgga => "MGGA",
    }
}

/// Run all cases; returns (markdown table, failures).
fn run(fx: &Fixture, bulk_only: bool, tol: impl Fn(usize) -> f64) -> (String, Vec<String>) {
    let mut rows = vec![format!(
        "linus-xc (libxc 7.0.0) vs PySCF {} eval_xc with libxc {}; {}\n\n\
         | functional | spin | family | order 0 | order 1 | order 2 | worst array |{}\n\
         |---|---|---|---|---|---|---|{}",
        fx.pyscf_version,
        fx.libxc_version,
        if bulk_only {
            "bulk points (checked) and edge points (reported)"
        } else {
            "all points"
        },
        if bulk_only { " edge points |" } else { "" },
        if bulk_only { "---|" } else { "" },
    )];
    let mut failures = Vec::new();
    for case in &fx.cases {
        let polarized = case.spin == "polarized";
        let inp = &fx.inputs[&case.spin];
        let f =
            Functional::new(&case.name, polarized).unwrap_or_else(|e| panic!("{}: {e}", case.name));
        assert_eq!(
            family_name(f.family()),
            case.xctype,
            "{}: family",
            case.name
        );
        check_hybrid(&f, case);
        assert_eq!(f.np(&inp.input()), inp.np);

        let all = vec![true; inp.np];
        let bulk = inp.bulk(polarized);
        let edge: Vec<bool> = bulk.iter().map(|b| !b).collect();
        let keep = if bulk_only { &bulk } else { &all };
        let (mut per_order, mut worst, mut edge_err) = ([0.0f64; 3], (0.0f64, ""), 0.0f64);
        let mut out = XcOutput::new();
        for (order, order_err) in per_order.iter_mut().enumerate() {
            f.eval(order, &inp.input(), &mut out).unwrap();
            for o in Out::ALL.into_iter().filter(|o| o.order() <= order) {
                let ours = out.get(o);
                let Some(r) = case.out.get(o.name()) else {
                    assert!(
                        ours.is_empty(),
                        "{}: {} produced but not in fixture",
                        case.name,
                        o.name()
                    );
                    continue;
                };
                let dim = o.dim(polarized);
                let e = max_rel_err(ours, r, dim, keep);
                *order_err = order_err.max(e);
                if e > worst.0 {
                    worst = (e, o.name());
                }
                if bulk_only {
                    edge_err = edge_err.max(max_rel_err(ours, r, dim, &edge));
                }
                if e > tol(order) {
                    failures.push(format!(
                        "{} {} order {order} {}: {e:.2e} > {:.0e}",
                        case.name,
                        case.spin,
                        o.name(),
                        tol(order)
                    ));
                }
            }
        }
        let e = |x: f64| {
            if x == 0.0 {
                "0".to_string()
            } else {
                format!("{x:.1e}")
            }
        };
        rows.push(format!(
            "| {} | {} | {} | {} | {} | {} | {} |{}",
            case.name,
            case.spin,
            case.xctype,
            e(per_order[0]),
            e(per_order[1]),
            e(per_order[2]),
            if worst.0 > 0.0 { worst.1 } else { "-" },
            if bulk_only {
                format!(" {} |", e(edge_err))
            } else {
                String::new()
            },
        ));
    }
    (rows.join("\n"), failures)
}

fn report(name: &str, table: &str) {
    println!("{table}\n");
    if let Ok(dir) = std::env::var("LINUS_XC_VALIDATION_TABLE") {
        std::fs::write(format!("{dir}/{name}.md"), format!("{table}\n")).unwrap();
    }
}

#[test]
fn matches_pyscf_with_same_libxc_7_0_0() {
    let fx = load("pyscf_libxc700_reference.json");
    assert_eq!(fx.libxc_version, "7.0.0");
    let (table, failures) = run(&fx, false, |_| 1e-13);
    report("validation_libxc700", &table);
    assert!(
        failures.is_empty(),
        "{} failures:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

#[test]
fn drift_vs_stock_pyscf_libxc() {
    let fx = load("pyscf_reference.json");
    let (table, failures) = run(&fx, true, |order| if order < 2 { 1e-9 } else { 1e-6 });
    report("validation_stock_pyscf", &table);
    assert!(
        failures.is_empty(),
        "{} failures:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

fn check_hybrid(f: &Functional, case: &Case) {
    let close = |a: f64, b: f64| (a - b).abs() <= 1e-12 * b.abs().max(1.0);
    let x = f.exact_exchange();
    let [omega, alpha, beta] = case.rsh_coeff_omega_alpha_beta;
    if omega != 0.0 {
        // PySCF: alpha = long-range fraction, beta = SR - LR (libxc convention).
        assert!(
            close(x.omega, omega) && close(x.alpha, alpha) && close(x.beta, beta),
            "{}: ours (a={}, b={}, w={}) pyscf (a={alpha}, b={beta}, w={omega})",
            case.name,
            x.alpha,
            x.beta,
            x.omega
        );
    } else {
        assert!(
            close(x.alpha, case.hybrid_coeff) && x.beta == 0.0,
            "{}: {} vs {}",
            case.name,
            x.alpha,
            case.hybrid_coeff
        );
        assert!(f.range_separation().is_none());
    }
    match (f.vv10(), case.nlc_b_c_weight) {
        (None, None) => {}
        (Some(v), Some([b, c, w])) => {
            assert!(
                close(v.b, b) && close(v.c, c) && close(v.scale, w),
                "{}: VV10",
                case.name
            )
        }
        (a, b) => panic!("{}: VV10 ours {a:?} pyscf {b:?}", case.name),
    }
}

/// A point below every density threshold gives exactly zero everywhere.
#[test]
fn screened_points_are_zero() {
    let fx = load("pyscf_libxc700_reference.json");
    for spin in ["unpolarized", "polarized"] {
        let inp = &fx.inputs[spin];
        let last = inp.np - 1; // rho = 1e-16 (per channel when polarized)
        for name in ["lda", "b3lyp", "r2scan", "wb97m-v"] {
            let f = Functional::new(name, spin == "polarized").unwrap();
            let mut out = XcOutput::new();
            f.eval(2, &inp.input(), &mut out).unwrap();
            for o in Out::ALL {
                let d = o.dim(spin == "polarized");
                let a = out.get(o);
                if !a.is_empty() {
                    assert!(
                        a[last * d..(last + 1) * d].iter().all(|&x| x == 0.0),
                        "{name} {spin} {}",
                        o.name()
                    );
                }
            }
        }
    }
}

/// libxc 7.0.0 quirk kept on purpose (we pin 7.0.0): the polarized fxc of
/// MGGA_C_TPSS / MGGA_C_R2SCAN is NaN when one spin density is exactly zero
/// (fixed in libxc 7.1). If this starts failing, libxc changed; update README.
#[test]
fn libxc_700_empty_channel_nan_is_documented() {
    let fx = load("pyscf_libxc700_reference.json");
    let inp = &fx.inputs["polarized"];
    assert_eq!(inp.rho[2], 0.0); // point 1 is (0, 42.07)
    for name in ["tpss", "r2scan"] {
        let f = Functional::new(name, true).unwrap();
        let mut out = XcOutput::new();
        f.eval(2, &inp.input(), &mut out).unwrap();
        assert!(out.v2rho2()[3..6].iter().all(|x| x.is_nan()), "{name}");
        assert!(out.vrho()[2..4].iter().all(|x| x.is_finite()), "{name}");
    }
}
