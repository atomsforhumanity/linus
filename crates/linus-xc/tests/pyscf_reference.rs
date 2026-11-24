//! Compare linus-xc with PySCF's `eval_xc` (fixtures from gen_reference.py).
//!
//! For every case (functional x spin) we evaluate orders 0, 1 and 2 and compare
//! each array an order produces with PySCF's order-2 output. Element error:
//! `|ours - ref| / max(|ref|, 1e-10 * max|ref|)` (max over the compared points
//! of that array), so entries ten orders below the array's scale count
//! absolutely. NaN must match NaN.
//!
//! * `pyscf_libxc700_reference.json`: PySCF 2.15 running libxc 7.0.0, the same
//!   libxc as ours. Every point, with a tolerance per [`PointClass`]: the
//!   outputs are bitwise PySCF's with Ubuntu 26.04's glibc, and the tolerances
//!   cover other C math libraries (see [`PointClass`]). This checks
//!   that linus-xc calls libxc correctly (layouts, spin, mixing, derivative
//!   order, thresholds), which would show as O(1) errors.
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

/// How far the system's C math library can move libxc's outputs at a point.
///
/// At the fixture's points linus-xc gives PySCF's bits with Ubuntu 26.04's glibc
/// 2.43 (x86-64 build), also with a libxc compiled for x86-64-v3 (build.rs turns
/// floating-point contraction off); glibc 2.36 and 2.39 (Debian 12, Ubuntu 24.04)
/// round `erf` differently and move ωB97X by up to 5.8e-15. Ubuntu 26.04's amd64v3
/// glibc (GitHub's ubuntu-26.04 runner) has a `cbrt` that differs from the x86-64
/// build's by 1-4 ulp on 20% of arguments: cbrt of a minority spin channel,
/// amplified by the B97 u = γx²/(1+γx²) and the ωB97 attenuation, sets every
/// class's maximum (element error, the four environments, 2026-11-20):
///
/// | class | points | max measured | tolerance |
/// |---|---|---|---|
/// | regular | the rest | 7.2e-13 (M06-2X unpolarized v2rho2, ρ = 6.6e-3) | 1e-11 |
/// | polarized | min ρ_s < 1e-3 ρ | 4.4e-10 (ωB97X v2rho2, ρ = (40.9, 3.5e-4)) | 5e-9 |
/// | degenerate | some ρ_s < 1e-6, σ_ss = 0 or τ_s = 0 | 1.1e-3 (ωB97M-V v2rho2, ρ = (0.49, 0)) | 2e-2, and the same screened (all-zero) points |
///
/// Each tolerance is at least 10 times its class's maximum.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum PointClass {
    Regular,
    /// Strongly spin-polarized: the minority channel below 1e-3 of the density.
    Polarized,
    /// An empty, vanishing (< 1e-6) or threshold-clamped spin channel.
    Degenerate,
}

impl PointClass {
    const ALL: [PointClass; 3] = [Self::Regular, Self::Polarized, Self::Degenerate];

    fn tolerance(self) -> f64 {
        match self {
            Self::Regular => 1e-11,
            Self::Polarized => 5e-9,
            Self::Degenerate => 2e-2,
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::Regular => "regular",
            Self::Polarized => "polarized",
            Self::Degenerate => "degenerate",
        }
    }
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

    /// (rho, sigma_ss, tau) of point `i`, per spin channel.
    fn point(&self, i: usize, polarized: bool) -> (&[f64], Vec<f64>, &[f64]) {
        let (nr, ns) = if polarized { (2, 3) } else { (1, 1) };
        let sigma = &self.sigma[i * ns..(i + 1) * ns];
        (
            &self.rho[i * nr..(i + 1) * nr],
            sigma.iter().step_by(2).copied().collect(),
            &self.tau[i * nr..(i + 1) * nr],
        )
    }

    /// Bulk point: every spin density >= 1e-4 and every sigma_ss > 0.
    fn bulk(&self, polarized: bool) -> Vec<bool> {
        (0..self.np)
            .map(|i| {
                let (r, s, _) = self.point(i, polarized);
                r.iter().all(|&x| x >= 1e-4) && s.iter().all(|&x| x > 0.0)
            })
            .collect()
    }

    fn classes(&self, polarized: bool) -> Vec<PointClass> {
        (0..self.np)
            .map(|i| {
                let (r, s, t) = self.point(i, polarized);
                let total: f64 = r.iter().sum();
                if r.iter().any(|&x| x < 1e-6) || s.contains(&0.0) || t.contains(&0.0) {
                    PointClass::Degenerate
                } else if r.iter().any(|&x| x < 1e-3 * total) {
                    PointClass::Polarized
                } else {
                    PointClass::Regular
                }
            })
            .collect()
    }
}

/// The array's scale floor: `1e-10 * max|ref|` over the points where `keep` is true.
fn floor(reference: &[Option<f64>], dim: usize, keep: &[bool]) -> f64 {
    let scale = reference
        .iter()
        .enumerate()
        .filter(|(i, _)| keep[i / dim])
        .filter_map(|(_, b)| b.filter(|b| b.is_finite()))
        .fold(0.0f64, |m, b| m.max(b.abs()));
    (1e-10 * scale).max(f64::MIN_POSITIVE)
}

/// Element error (see module docs); `None` reference is NaN.
fn element_err(a: f64, b: Option<f64>, floor: f64) -> f64 {
    let b = b.unwrap_or(f64::NAN);
    match (a.is_nan(), b.is_nan()) {
        (true, true) => 0.0,
        (false, false) => (a - b).abs() / b.abs().max(floor),
        _ => f64::INFINITY,
    }
}

/// Max error over the points where `keep` is true (see module docs).
fn max_rel_err(ours: &[f64], reference: &[Option<f64>], dim: usize, keep: &[bool]) -> f64 {
    assert_eq!(ours.len(), reference.len());
    let floor = floor(reference, dim, keep);
    ours.iter()
        .zip(reference)
        .enumerate()
        .filter(|(i, _)| keep[i / dim])
        .map(|(_, (a, b))| element_err(*a, *b, floor))
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

/// Evaluate every case at orders 0, 1 and 2 and call `check(case, order, array,
/// ours, reference, dim)` for every array an order produces.
fn for_each_array(
    fx: &Fixture,
    mut check: impl FnMut(&Case, usize, Out, &[f64], &[Option<f64>], usize),
) {
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
        let mut out = XcOutput::new();
        for order in 0..=2 {
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
                assert_eq!(ours.len(), r.len(), "{} {}", case.name, o.name());
                check(case, order, o, ours, r, o.dim(polarized));
            }
        }
    }
}

fn e(x: f64) -> String {
    if x == 0.0 {
        "0".to_string()
    } else {
        format!("{x:.1e}")
    }
}

fn report(name: &str, table: &str) {
    println!("{table}\n");
    if let Ok(dir) = std::env::var("LINUS_XC_VALIDATION_TABLE") {
        std::fs::write(format!("{dir}/{name}.md"), format!("{table}\n")).unwrap();
    }
}

fn assert_no_failures(failures: &[String]) {
    assert!(
        failures.is_empty(),
        "{} failures:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

#[test]
fn matches_pyscf_with_same_libxc_7_0_0() {
    let fx = load("pyscf_libxc700_reference.json");
    assert_eq!(fx.libxc_version, "7.0.0");
    let classes: BTreeMap<&str, Vec<PointClass>> = fx
        .inputs
        .iter()
        .map(|(spin, inp)| (spin.as_str(), inp.classes(spin == "polarized")))
        .collect();
    let mut rows = vec![format!(
        "linus-xc (libxc 7.0.0) vs PySCF {} eval_xc with libxc {}; all points, max error \
         per point class (tolerance)\n\n\
         | functional | spin | family | regular ({:.0e}) | polarized ({:.0e}) | degenerate ({:.0e}) | worst array |\n\
         |---|---|---|---|---|---|---|",
        fx.pyscf_version,
        fx.libxc_version,
        PointClass::Regular.tolerance(),
        PointClass::Polarized.tolerance(),
        PointClass::Degenerate.tolerance(),
    )];
    let mut failures = Vec::new();
    // per case: max error per class, and the worst array
    type Row = (String, [f64; 3], (f64, &'static str));
    let mut row: Option<Row> = None;
    let mut flush = |row: &mut Option<Row>| {
        if let Some((key, m, worst)) = row.take() {
            rows.push(format!(
                "| {key} | {} | {} | {} | {} |",
                e(m[0]),
                e(m[1]),
                e(m[2]),
                if worst.0 > 0.0 { worst.1 } else { "-" }
            ));
        }
    };
    for_each_array(&fx, |case, order, o, ours, r, dim| {
        let key = format!("{} | {} | {}", case.name, case.spin, case.xctype);
        if row.as_ref().is_none_or(|(k, _, _)| *k != key) {
            flush(&mut row);
            row = Some((key, [0.0; 3], (0.0, "")));
        }
        let (_, max, worst) = row.as_mut().unwrap();
        let cls = &classes[case.spin.as_str()];
        let floor = floor(r, dim, &vec![true; cls.len()]);
        let mut class_err = [0.0f64; 3];
        for (k, (&a, &b)) in ours.iter().zip(r).enumerate() {
            let ci = cls[k / dim] as usize;
            class_err[ci] = class_err[ci].max(element_err(a, b, floor));
        }
        // Degenerate points: libxc's screening (a point all zeros) must match exactly; NaN
        // against a number is an infinite error above.
        for (p, _) in cls
            .iter()
            .enumerate()
            .filter(|(_, &c)| c == PointClass::Degenerate)
        {
            let ours_zero = ours[p * dim..(p + 1) * dim].iter().all(|&a| a == 0.0);
            let ref_zero = r[p * dim..(p + 1) * dim].iter().all(|&b| b == Some(0.0));
            if ours_zero != ref_zero {
                failures.push(format!(
                    "{} {} order {order} {} point {p}: screened in {} only",
                    case.name,
                    case.spin,
                    o.name(),
                    if ours_zero { "ours" } else { "the reference" }
                ));
            }
        }
        for c in PointClass::ALL {
            let err = class_err[c as usize];
            max[c as usize] = max[c as usize].max(err);
            if err > worst.0 {
                *worst = (err, o.name());
            }
            if err > c.tolerance() {
                failures.push(format!(
                    "{} {} order {order} {} {} points: {err:.2e} > {:.0e}",
                    case.name,
                    case.spin,
                    o.name(),
                    c.name(),
                    c.tolerance()
                ));
            }
        }
    });
    flush(&mut row);
    report("validation_libxc700", &rows.join("\n"));
    assert_no_failures(&failures);
}

#[test]
fn drift_vs_stock_pyscf_libxc() {
    let fx = load("pyscf_reference.json");
    let tol = |order: usize| if order < 2 { 1e-9 } else { 1e-6 };
    let bulk: BTreeMap<&str, Vec<bool>> = fx
        .inputs
        .iter()
        .map(|(spin, inp)| (spin.as_str(), inp.bulk(spin == "polarized")))
        .collect();
    let mut rows = vec![format!(
        "linus-xc (libxc 7.0.0) vs PySCF {} eval_xc with libxc {}; bulk points (checked) and \
         edge points (reported)\n\n\
         | functional | spin | family | order 0 | order 1 | order 2 | worst array | edge points |\n\
         |---|---|---|---|---|---|---|---|",
        fx.pyscf_version, fx.libxc_version,
    )];
    let mut failures = Vec::new();
    // per case: max error per order at bulk points, the worst array, max error at edge points
    type Row = (String, [f64; 3], (f64, &'static str), f64);
    let mut row: Option<Row> = None;
    let mut flush = |row: &mut Option<Row>| {
        if let Some((key, per_order, worst, edge)) = row.take() {
            rows.push(format!(
                "| {key} | {} | {} | {} | {} | {} |",
                e(per_order[0]),
                e(per_order[1]),
                e(per_order[2]),
                if worst.0 > 0.0 { worst.1 } else { "-" },
                e(edge)
            ));
        }
    };
    for_each_array(&fx, |case, order, o, ours, r, dim| {
        let key = format!("{} | {} | {}", case.name, case.spin, case.xctype);
        if row.as_ref().is_none_or(|(k, ..)| *k != key) {
            flush(&mut row);
            row = Some((key, [0.0; 3], (0.0, ""), 0.0));
        }
        let (_, per_order, worst, edge_err) = row.as_mut().unwrap();
        let bulk = &bulk[case.spin.as_str()];
        let edge: Vec<bool> = bulk.iter().map(|b| !b).collect();
        let err = max_rel_err(ours, r, dim, bulk);
        per_order[order] = per_order[order].max(err);
        if err > worst.0 {
            *worst = (err, o.name());
        }
        *edge_err = edge_err.max(max_rel_err(ours, r, dim, &edge));
        if err > tol(order) {
            failures.push(format!(
                "{} {} order {order} {}: {err:.2e} > {:.0e}",
                case.name,
                case.spin,
                o.name(),
                tol(order)
            ));
        }
    });
    flush(&mut row);
    report("validation_stock_pyscf", &rows.join("\n"));
    assert_no_failures(&failures);
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
