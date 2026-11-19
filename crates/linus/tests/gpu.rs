//! The GPU backend end to end (feature `gpu`): energies, gradients, Hessians and
//! polarizabilities with `backend: "gpu"` against `backend: "cpu"` in the same build, the
//! per-kernel placement and fallbacks in `resolved.backend`, and run-to-run reproducibility on
//! the device. Without a usable device the tests skip, unless `LINUS_GPU_REQUIRED=1`.
#![cfg(feature = "gpu")]

use linus::{CalcOptions, CalcType, LinusError, Session, Structure};
use serde_json::{json, Value};

const WATER: &str = "3\nwater\nO 0.0 0.05 0.11\nH 0.0 0.76 -0.47\nH 0.1 -0.75 -0.46\n";

/// A session with `opts` (JSON settings) and `backend`, or None when the GPU backend has no
/// device (a skip, unless LINUS_GPU_REQUIRED=1).
fn calc(charge: i32, mult: u32, opts: Value, backend: &str) -> Option<Session> {
    calc_with(charge, mult, opts, json!(backend))
}

/// [`calc`] with the backend setting as JSON.
fn calc_with(charge: i32, mult: u32, opts: Value, backend: Value) -> Option<Session> {
    let mol = Structure::from_xyz_str(WATER, charge, mult).unwrap();
    let mut o = opts;
    if o["method"] == "hf" {
        // Hartree-Fock has no XC grid (D41: its grid settings are input errors)
        o.as_object_mut().unwrap().remove("grid");
    }
    o["backend"] = backend;
    o["threads"] = json!(4);
    match Session::new(mol, CalcOptions::from_json(o).unwrap()) {
        Ok(c) => Some(c),
        Err(LinusError::Backend { message, .. })
            if std::env::var("LINUS_GPU_REQUIRED").as_deref() != Ok("1") =>
        {
            eprintln!("no usable GPU, skipping: {message}");
            None
        }
        Err(e) => panic!("{e}"),
    }
}

fn base(method: &str) -> Value {
    json!({"method": method, "basis": "def2-svp", "jk": "df", "grid": "pyscf:2"})
}

fn backend_json(c: &Session) -> Value {
    serde_json::to_value(c.backend_used()).unwrap()
}

fn max_diff(a: &[[f64; 3]], b: &[[f64; 3]]) -> f64 {
    a.iter()
        .flatten()
        .zip(b.iter().flatten())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max)
}

/// SCF energies on the device engines equal the CPU's to round-off with the same iteration
/// counts, bitwise run to run; closed and open shells, HF, B3LYP and range-separated hybrids
/// (the long-range exchange on the device).
#[test]
fn energies_match_the_cpu() {
    for (method, charge, mult, jk_on_gpu) in [
        ("hf", 0, 1, true),
        ("b3lyp", 0, 1, true),
        ("b3lyp", 1, 2, true),
        ("wb97x-d3bj", 0, 1, true),
        ("cam-b3lyp", 1, 2, true),
    ] {
        let Some(mut g) = calc(charge, mult, base(method), "gpu") else {
            return;
        };
        let mut c = calc(charge, mult, base(method), "cpu").unwrap();
        let (rg, rc) = (g.energy().unwrap(), c.energy().unwrap());
        assert!(
            (rg.energy - rc.energy).abs() < 1e-10 && rg.iterations == rc.iterations,
            "{method} {mult}: gpu {} ({} iterations), cpu {} ({})",
            rg.energy,
            rg.iterations,
            rc.energy,
            rc.iterations
        );
        g.reset_all();
        let again = g.energy().unwrap();
        assert_eq!(again.energy.to_bits(), rg.energy.to_bits(), "{method}");
        let b = backend_json(&g);
        assert!(b["device"]["name"].is_string());
        assert_eq!(b["jk"], if jk_on_gpu { "gpu" } else { "cpu" }, "{b}");
        assert_eq!(
            b["xc"],
            if method == "hf" {
                Value::Null
            } else {
                json!("gpu")
            }
        );
        let reasons: Vec<&str> = b["fallbacks"]
            .as_array()
            .unwrap()
            .iter()
            .map(|f| f["kernel"].as_str().unwrap())
            .collect();
        assert_eq!(reasons.contains(&"jk"), !jk_on_gpu, "{b}");
        // scf_loop auto (decisions D27, D31): the device loop, closed and open shells
        assert_eq!(b["scf_loop"], "device", "{b}");
        assert!(!reasons.contains(&"scf_loop"), "{b}");
        // the CPU backend has no device and no fallbacks
        let bc = backend_json(&c);
        assert!(bc["device"].is_null() && bc["fallbacks"].as_array().unwrap().is_empty());
    }
}

/// `auto` takes the device; emulated FP64 exchange (the Ozaki scheme, 7 slices, the tensor
/// pre-sliced) and the Ozaki fit of the density-fitting setup stay within round-off of FP64
/// with the same SCF iterations, for the SCF and the response J/K of a Hessian (its exchange on
/// the stored operator, whose setup GEMMs run in cuBLAS's emulation: no Ozaki planes), and
/// report their tiers.
/// `precision.fp64: auto`'s choice for the card and why (`resolved.reasons`).
fn ra_reason(c: &mut Session) -> String {
    let mut rep = c.run(CalcType::Energy).unwrap();
    rep.resolved
        .reasons
        .remove("precision.fp64")
        .unwrap_or_default()
}

#[test]
fn auto_and_emulated() {
    let Some(mut a) = calc(0, 1, base("b3lyp"), "auto") else {
        return;
    };
    let ra = a.energy().unwrap();
    let b = backend_json(&a);
    assert_eq!(b["jk"], "gpu");
    // precision.fp64 auto (decision D27): emulated on FP64-poor cards, native on FP64-rich
    // ones, with the choice and its reason reported; the closed shell on the device loop
    let note = ra_reason(&mut a);
    let emulates = note.starts_with("auto: emulated (");
    assert!(emulates || note.starts_with("auto: native ("), "{b}");
    // every family's arithmetic with its reason
    assert_eq!(
        b["arithmetic"]["scf_k"]["arith"],
        if emulates {
            "ozaki (55 bits), else cublas (55 bits), else fp64"
        } else {
            "fp64"
        },
        "{b}"
    );
    assert_eq!(
        b["precision"],
        if emulates { "emulated" } else { "double" },
        "{b}"
    );
    assert_eq!(b.get("emulation").is_some(), emulates, "{b}");
    assert_eq!(b["scf_loop"], "device", "{b}");
    let mut o = base("b3lyp");
    o["backend"] = json!("gpu");
    o["precision"] = json!({"fp64": "emulated", "response": "fp64"});
    let mol = Structure::from_xyz_str(WATER, 0, 1).unwrap();
    let mut e = Session::new(mol, CalcOptions::from_json(o).unwrap()).unwrap();
    let re = e.energy().unwrap();
    assert!(
        (ra.energy - re.energy).abs() < 1e-10,
        "{} vs {}",
        ra.energy,
        re.energy
    );
    assert_eq!(ra.iterations, re.iterations);
    let b = backend_json(&e);
    assert_eq!(b["precision"], "emulated");
    // the density-fitting fit in the Ozaki scheme too
    assert_eq!(
        b["emulation"],
        json!([{"kernel": "jk", "scheme": "ozaki", "bits": 55, "presliced": true,
            "fit": {"scheme": "ozaki", "bits": 55}}])
    );
    // the response's J/K and the Hessian's explicit exchange too: a Hessian against the FP64
    // device's
    let mut o = base("b3lyp");
    o["precision"] = json!("fp64");
    let Some(mut d) = calc_with(0, 1, o, json!("gpu")) else {
        return;
    };
    let (he, hd) = (e.hessian().unwrap(), d.hessian().unwrap());
    let dh = he
        .hessian
        .iter()
        .zip(&hd.hessian)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max);
    assert!(dh < 1e-9, "emulated Hessian: max |dH| {dh:e}");
    let emu = backend_json(&e)["emulation"].clone();
    let kernels: Vec<&str> = emu
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x["kernel"].as_str().unwrap())
        .collect();
    assert_eq!(kernels, ["jk", "response_jk", "hessian_jk"], "{emu}");
    assert_eq!(emu[1]["scheme"], "cublas", "{emu}");
    assert!(
        emu[1]["reason"]
            .as_str()
            .unwrap()
            .contains("stored operator"),
        "{emu}"
    );
    assert_eq!(emu[2]["scheme"], "ozaki", "{emu}");
    let resp = backend_json(&e)["response"].clone();
    assert_eq!(resp[0]["purpose"], "hessian", "{resp}");
    assert_eq!(resp[0]["jk"], "emulated", "{resp}");
    assert_eq!(resp[0]["exchange"], "operator", "{resp}");
}

/// Hessians with the explicit two-electron and XC terms and the CPKS products on the device equal
/// the CPU's to 1e-10 Eh/bohr^2 (a GGA and a meta-GGA, whose explicit XC terms' tau parts run on
/// the device too; r2SCAN's response converged to 1e-9: at the default 1e-6 its Hessians follow
/// where the CPHF solver stops, which round-off moves, as in reference/validate_gpu.py);
/// polarizabilities to 1e-8 au.
#[test]
fn hessians_and_polarizabilities_match_the_cpu() {
    for (method, charge, mult) in [("b3lyp", 0, 1), ("hf", 1, 2), ("r2scan", 1, 2)] {
        let mut o = base(method);
        if method == "r2scan" {
            o["response"] = json!({"tol": 1e-9});
        }
        o["precision"] = json!({"response": "fp64"});
        // FP64 on every card (precision.fp64 auto emulates on FP64-poor ones, decision D27)
        let mut og = o.clone();
        og["precision"] = json!("fp64");
        let Some(mut g) = calc_with(charge, mult, og, json!("gpu")) else {
            return;
        };
        let mut c = calc(charge, mult, o, "cpu").unwrap();
        let (hg, hc) = (g.hessian().unwrap(), c.hessian().unwrap());
        let d = hg
            .hessian
            .iter()
            .zip(&hc.hessian)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f64::max);
        assert!(d < 1e-10, "{method}: max |dH| {d:e}");
        assert_eq!(hg.cphf_iterations, hc.cphf_iterations, "{method}");
        let b = backend_json(&g);
        assert_eq!(b["response_jk"], "gpu", "{b}");
        // the explicit two-electron and XC terms on the device
        assert_eq!(b["hessian_jk"], "gpu", "{b}");
        let xc_fallback = b["fallbacks"]
            .as_array()
            .unwrap()
            .iter()
            .any(|f| f["kernel"] == "hessian_xc");
        assert!(!xc_fallback, "{b}");
        if method == "hf" {
            assert!(b["hessian_xc"].is_null(), "{b}");
        } else {
            assert_eq!(b["hessian_xc"], "gpu", "{b}");
        }
        assert_eq!(
            b["response_xc"],
            if method == "hf" {
                Value::Null
            } else {
                json!("gpu")
            }
        );
        // the Hessian's 3N perturbations pay for the stored exchange operator, in FP64 (r2SCAN
        // has no exact exchange)
        let r = &b["response"][0];
        let exchange = if method == "r2scan" {
            Value::Null
        } else {
            json!("operator")
        };
        assert_eq!(
            (&r["purpose"], &r["jk"], &r["exchange"]),
            (&json!("hessian"), &json!("fp64"), &exchange),
            "{b}"
        );
        if method != "hf" {
            assert_eq!(r["xc"], "fp64", "{b}");
        }
        let (pg, pc) = (g.polarizability().unwrap(), c.polarizability().unwrap());
        let d = (0..3)
            .flat_map(|i| (0..3).map(move |j| (i, j)))
            .map(|(i, j)| (pg.alpha[i][j] - pc.alpha[i][j]).abs())
            .fold(0.0, f64::max);
        assert!(d < 1e-8, "{method}: max |d alpha| {d:e}");
        // the polarizability's own record (for water even its three perturbations pay for
        // the operator: o = 5)
        let b = backend_json(&g);
        let r = b["response"]
            .as_array()
            .unwrap()
            .iter()
            .find(|r| r["purpose"] == "polarizability")
            .unwrap()
            .clone();
        assert_eq!(r["jk"], "fp64", "{b}");
    }
}

/// IR runs: the electric-field CPHF runs on the Hessian's response J/K (its
/// stored exchange operator, recorded under "polarizability" with no second setup) and the
/// field kernel's own blocks with its ground state kept; dipole derivatives and polarizability
/// equal the CPU's to 1e-8 au, closed and open shell, and the polarizability equals a standalone
/// one's (its own J/K) to round-off.
#[test]
fn ir_field_response_runs_on_the_hessians_operator() {
    use linus::HessianOptions;
    for (charge, mult) in [(0, 1), (1, 2)] {
        let mut o = base("b3lyp");
        o["precision"] = json!({"response": "fp64"});
        let Some(mut g) = calc(charge, mult, o.clone(), "gpu") else {
            return;
        };
        let mut c = calc(charge, mult, o, "cpu").unwrap();
        let ir = HessianOptions {
            dipole_derivatives: true,
        };
        let (hg, hc) = (g.hessian_with(&ir).unwrap(), c.hessian_with(&ir).unwrap());
        let (dg, dc) = (g.ir_properties(&hg).unwrap(), c.ir_properties(&hc).unwrap());
        let da = max_diff(&dg.polarizability, &dc.polarizability);
        let dd = max_diff(&dg.apt, &dc.apt);
        assert!(
            da < 1e-8 && dd < 1e-8,
            "mult {mult}: |d alpha| {da:e}, |d apt| {dd:e}"
        );
        // the Hessian's J/K arithmetic (FP64, or its emulation where the card's auto resolves
        // to emulated, D27), on its operator; the field kernel FP64-class, never FP32
        let b = backend_json(&g);
        let rec = |p: &str| {
            b["response"]
                .as_array()
                .unwrap()
                .iter()
                .find(|r| r["purpose"] == p)
                .unwrap()
                .clone()
        };
        let (r, h) = (rec("polarizability"), rec("hessian"));
        assert_eq!(
            (&r["jk"], &r["exchange"]),
            (&h["jk"], &json!("operator")),
            "{b}"
        );
        assert!(r["xc"] == "fp64" || r["xc"] == "emulated", "{b}");
        assert!(r["reason"].as_str().unwrap().contains("kept"), "{b}");
        // the operator was granted at the Hessian's setup and built within its grant
        assert!(h["operator_granted_gib"].as_f64().unwrap() > 0.0, "{b}");
        assert!(h.get("plan_violation").is_none(), "{b}");
        let t = &dg.timers;
        assert!(t.get("pol_cphf_setup").is_none(), "a second J/K setup");
        assert!(t.get("pol_cphf_xc_dev_calls").unwrap() >= 1.0);
        // a standalone polarizability: its own J/K, the same field kernel
        let ps = g.polarizability().unwrap();
        let ds = max_diff(&ps.alpha, &dg.polarizability);
        assert!(ds < 1e-11, "mult {mult}: standalone |d alpha| {ds:e}");
        eprintln!("IR mult {mult}: |d alpha| {da:e}, |d apt| {dd:e}, standalone {ds:e}");
    }
}

/// `precision.response: "fp32"`: the device's CPKS kernel runs in FP32 (and the exchange in
/// FP32, as on the CPU); the Hessian stays within the single-precision response's accuracy of
/// the CPU's (whose kernel stays FP64), CPKS iterations within one, and reports both.
#[test]
fn single_precision_response_runs_the_fp32_kernel() {
    let mut o = base("b3lyp");
    o["precision"] = json!({"response": "fp32"});
    let Some(mut g) = calc(0, 1, o.clone(), "gpu") else {
        return;
    };
    let mut c = calc(0, 1, o, "cpu").unwrap();
    let (hg, hc) = (g.hessian().unwrap(), c.hessian().unwrap());
    let d = hg
        .hessian
        .iter()
        .zip(&hc.hessian)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max);
    assert!(d < 1e-6, "max |dH| {d:e}");
    // the FP32 kernel's round-off near the threshold: within one iteration
    assert!(
        hg.cphf_iterations.abs_diff(hc.cphf_iterations) <= 1,
        "{} vs {}",
        hg.cphf_iterations,
        hc.cphf_iterations
    );
    eprintln!(
        "single: max |dH| {d:e}, CPKS iterations {} (CPU {})",
        hg.cphf_iterations, hc.cphf_iterations
    );
    let r = &backend_json(&g)["response"][0];
    assert_eq!(
        (&r["jk"], &r["xc"]),
        (&json!("fp32"), &json!("fp32")),
        "{r}"
    );
    assert_eq!(r["exchange"], "tensor", "{r}");
}

/// Gradients with the device's DF and XC terms equal the CPU's to 1e-9 Eh/bohr.
#[test]
fn gradients_match_the_cpu() {
    for (method, charge, mult) in [
        ("b3lyp", 0, 1),
        ("b3lyp", 1, 2),
        ("hf", 0, 1),
        ("wb97x-d3bj", 0, 1),
        ("cam-b3lyp", 1, 2),
    ] {
        let Some(mut g) = calc(charge, mult, base(method), "gpu") else {
            return;
        };
        let mut c = calc(charge, mult, base(method), "cpu").unwrap();
        let (gg, gc) = (g.gradient().unwrap(), c.gradient().unwrap());
        let d = max_diff(&gg.gradient, &gc.gradient);
        assert!(d < 1e-9, "{method} {mult}: max |dG| {d:e}");
    }
}

/// The device SCF loop gives the CPU's energies to round-off with the same iteration counts,
/// bitwise run to run (range-separated hybrids and VV10 functionals included).
#[test]
fn device_scf_loop() {
    for (method, charge, mult, on_device) in [
        ("b3lyp", 0, 1, true),
        ("hf", 1, 2, true),
        ("wb97x-d3bj", 0, 1, true),
        ("wb97m-v", 0, 1, true),
    ] {
        let dev = json!({"kind": "gpu", "scf_loop": "device"});
        let Some(mut g) = calc_with(charge, mult, base(method), dev) else {
            return;
        };
        let mut c = calc(charge, mult, base(method), "cpu").unwrap();
        let (rg, rc) = (g.energy().unwrap(), c.energy().unwrap());
        assert!(
            (rg.energy - rc.energy).abs() < 1e-10 && rg.iterations == rc.iterations,
            "{method} {mult}: device loop {} ({} iterations), cpu {} ({})",
            rg.energy,
            rg.iterations,
            rc.energy,
            rc.iterations
        );
        g.reset_all();
        assert_eq!(g.energy().unwrap().energy.to_bits(), rg.energy.to_bits());
        let b = backend_json(&g);
        let want = if on_device { "device" } else { "host" };
        assert_eq!(b["scf_loop"], want, "{b}");
        let loop_fallback = b["fallbacks"]
            .as_array()
            .unwrap()
            .iter()
            .any(|f| f["kernel"] == "scf_loop");
        assert_eq!(loop_fallback, !on_device, "{b}");
    }
}

/// The device loop runs the default stability probe as the host loop does: an open shell it
/// finishes reports `resolved.scf.stability_probe` 2 and the probe's record, as the CPU
/// (tests/gpu_open_shell_loop.rs has the hard cases).
#[test]
fn device_loop_runs_the_probe() {
    let dev = json!({"kind": "gpu", "scf_loop": "device"});
    let Some(mut g) = calc_with(1, 2, base("hf"), dev) else {
        return;
    };
    let rg = g.run(CalcType::Energy).unwrap();
    assert_eq!(backend_json(&g)["scf_loop"], "device");
    assert_eq!(rg.resolved.record.scf.stability_probe, 2);
    let rc = calc(1, 2, base("hf"), "cpu")
        .unwrap()
        .run(CalcType::Energy)
        .unwrap();
    assert_eq!(rc.resolved.record.scf.stability_probe, 2);
    let (sg, sc) = (&rg.scf.stability, &rc.scf.stability);
    assert!(
        sg.len() == 1
            && sc.len() == 1
            && sg[0].probe
            && sc[0].probe
            && sg[0].fock_builds == sc[0].fock_builds
            // one-sided products: O(h) apart where the eigensolvers' orbital signs differ
            && (sg[0].eigenvalue - sc[0].eigenvalue).abs() < 1e-3,
        "{sg:?} vs {sc:?}"
    );
}

/// An implicit solvent on the device (the `pcm` family): energies (predicted and exact surface
/// terms, C-PCM and IEF-PCM, host and device loops) and gradients equal the CPU's to round-off
/// with the same iteration counts, bitwise run to run; the density-fitted PCM keeps the PCM and
/// the J/K on the CPU, with the reasons.
#[test]
fn pcm_on_the_device() {
    let thf = |model: &str, scf: Option<&str>| {
        let mut o = base("b3lyp");
        o["solvent"] = json!({"name": "thf", "model": model});
        match scf {
            Some("df") => o["solvent"]["df"] = json!(true),
            Some(s) => o["solvent"]["scf_terms"] = json!(s),
            None => {}
        }
        o
    };
    for (model, scf, charge, mult, dloop) in [
        ("cpcm", None, 0, 1, false),
        ("cpcm", None, 1, 2, false),
        ("iefpcm", Some("exact"), 0, 1, false),
        ("cpcm", None, 0, 1, true),
    ] {
        let dev = json!({"kind": "gpu", "scf_loop": if dloop { "device" } else { "host" }});
        let Some(mut g) = calc_with(charge, mult, thf(model, scf), dev) else {
            return;
        };
        let mut c = calc(charge, mult, thf(model, scf), "cpu").unwrap();
        let (gg, gc) = (g.gradient().unwrap(), c.gradient().unwrap());
        let (rg, rc) = (g.last().unwrap().clone(), c.last().unwrap().clone());
        assert!(
            (gg.energy - gc.energy).abs() < 1e-10 && rg.iterations == rc.iterations,
            "{model} {mult}: gpu {} ({} iterations), cpu {} ({})",
            gg.energy,
            rg.iterations,
            gc.energy,
            rc.iterations
        );
        let d = max_diff(&gg.gradient, &gc.gradient);
        assert!(d < 1e-9, "{model} {mult}: max |dG| {d:e}");
        let b = backend_json(&g);
        assert_eq!(b["pcm"], "gpu", "{b}");
        assert_eq!(b["jk"], "gpu", "{b}");
        assert_eq!(b["scf_loop"], if dloop { "device" } else { "host" }, "{b}");
        assert_eq!(backend_json(&c)["pcm"], "cpu");
        g.reset_all();
        let again = g.gradient().unwrap();
        assert_eq!(again.energy.to_bits(), gg.energy.to_bits());
        assert!(again
            .gradient
            .iter()
            .flatten()
            .zip(gg.gradient.iter().flatten())
            .all(|(x, y)| x.to_bits() == y.to_bits()));
    }
    // the density-fitted PCM (`solvent.df`): the PCM and the J/K on
    // the CPU, with the reasons
    let Some(mut g) = calc(0, 1, thf("cpcm", Some("df")), "gpu") else {
        return;
    };
    let mut c = calc(0, 1, thf("cpcm", Some("df")), "cpu").unwrap();
    let (eg, ec) = (g.energy().unwrap().energy, c.energy().unwrap().energy);
    assert!((eg - ec).abs() < 1e-10, "density-fitted PCM: {eg} vs {ec}");
    let b = backend_json(&g);
    assert!(b["pcm"] == "cpu" && b["jk"] == "cpu", "{b}");
    let reasons: Vec<&str> = b["fallbacks"]
        .as_array()
        .unwrap()
        .iter()
        .map(|f| f["kernel"].as_str().unwrap())
        .collect();
    assert!(reasons.contains(&"pcm") && reasons.contains(&"jk"), "{b}");
}

/// Each calculation reports its peak device memory (`resources.mem_device_peak_gib`), within
/// the device's limit and counted from its own start: an energy after a Hessian reports its
/// SCF's peak, below the Hessian's. The CPU backend reports none.
#[test]
fn calculations_report_their_peak_device_memory() {
    let Some(mut g) = calc(0, 1, base("b3lyp"), "gpu") else {
        return;
    };
    let peak = |r: &linus::Report| r.resources.get("mem_device_peak_gib").copied();
    let hess = g.run(CalcType::Hessian).unwrap();
    let limit = hess
        .resolved
        .record
        .backend
        .device
        .as_ref()
        .unwrap()
        .memory_limit_gib;
    let ph = peak(&hess).unwrap();
    assert!(ph > 0.0 && ph <= limit, "Hessian: {ph} GiB of {limit}");
    // a new SCF for the energy
    g.reset_all();
    let pe = peak(&g.run(CalcType::Energy).unwrap()).unwrap();
    assert!(pe > 0.0 && pe < ph, "energy {pe} GiB, Hessian {ph}");
    let mut c = calc(0, 1, base("b3lyp"), "cpu").unwrap();
    assert_eq!(peak(&c.run(CalcType::Energy).unwrap()), None);
}
