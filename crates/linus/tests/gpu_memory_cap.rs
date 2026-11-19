//! A device memory cap (`device_memory_gib`, or its default `$LINUS_GPU_MEMORY_GIB`; feature
//! `gpu`): it is reported in
//! `resolved.backend.device`, and the setups that do not fit fall back to the CPU with their
//! reasons instead of failing: the SCF's density-fitted J/K, and an implicit solvent in the
//! predicted mode whose device surface terms need the device J build's fit (they then follow
//! the J/K to the CPU), and an XC gradient whose call does not fit beside its setup. Results
//! equal the CPU backend's. A cap the calculation fits in changes
//! nothing: a Hessian under 8 GB keeps its response J/K on the device. Its own test binary: the
//! variable is process-wide and read when a device opens (the tests take [`CAP`] while they set
//! or clear it). The caps are the old tests' GB in GiB, exactly (40 MB is 0.0372529... GiB).
//! Without a usable device the tests skip, unless `LINUS_GPU_REQUIRED=1`.
#![cfg(feature = "gpu")]

use std::sync::Mutex;

use linus::{CalcOptions, LinusError, Session, Structure};
use serde_json::{json, Value};

/// Held while a test sets or clears `LINUS_GPU_MEMORY_GIB` and opens its devices.
static CAP: Mutex<()> = Mutex::new(());

/// A cap of `bytes` in GiB (the old tests' GB caps, the same bytes).
fn gib(bytes: f64) -> f64 {
    bytes / 1073741824.0
}

const LA18: &str = include_str!("../../../bench/molecules/la18.xyz");

fn calc(opts: Value, backend: &str) -> Option<Session> {
    let mol = Structure::from_xyz_str(LA18, 0, 1).unwrap();
    let mut o = opts;
    // the backend's kind (an object's other settings kept)
    match o.get_mut("backend") {
        Some(Value::Object(b)) => {
            b.insert("kind".into(), json!(backend));
        }
        _ => o["backend"] = json!(backend),
    }
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

fn fallback_reason(b: &Value, kernel: &str) -> Option<String> {
    b["fallbacks"]
        .as_array()
        .unwrap()
        .iter()
        .find(|f| f["kernel"] == kernel)
        .map(|f| f["reason"].as_str().unwrap().to_string())
}

#[test]
fn a_memory_cap_falls_back_with_reasons() {
    let _cap = CAP.lock().unwrap_or_else(|e| e.into_inner());
    // 40 MB: la18/def2-SVP's density-fitting setup (tensor and build workspace) does not fit,
    // the PCM's surface operator does
    // the variable, the setting's default
    std::env::set_var("LINUS_GPU_MEMORY_GIB", gib(40e6).to_string());
    let base = json!({"method": "b3lyp", "basis": "def2-svp", "jk": "df", "grid": "pyscf:2"});
    let mut pcm = base.clone();
    pcm["solvent"] = json!({"name": "thf", "model": "cpcm"});
    for (name, opts) in [("gas phase", base), ("C-PCM, predicted", pcm)] {
        let Some(mut g) = calc(opts.clone(), "gpu") else {
            return;
        };
        let mut c = calc(opts, "cpu").unwrap();
        let (eg, ec) = (g.energy().unwrap(), c.energy().unwrap());
        assert!(
            (eg.energy - ec.energy).abs() < 1e-10 && eg.iterations == ec.iterations,
            "{name}: gpu {} ({} iterations), cpu {} ({})",
            eg.energy,
            eg.iterations,
            ec.energy,
            ec.iterations
        );
        let b = serde_json::to_value(g.backend_used()).unwrap();
        assert_eq!(b["device"]["memory_cap_gib"], gib(40e6), "{b}");
        assert!(
            b["device"]["memory_limit_gib"].as_f64().unwrap() <= 0.04 / 1.073741824,
            "{b}"
        );
        assert_eq!(b["jk"], "cpu", "{name}: {b}");
        let why = fallback_reason(&b, "jk").unwrap_or_default();
        assert!(why.contains("device memory"), "{name}: {b}");
        if name.starts_with("C-PCM") {
            assert_eq!(b["pcm"], "cpu", "{b}");
            let why = fallback_reason(&b, "pcm").unwrap_or_default();
            assert!(
                why.contains("fell back") || why.contains("device memory"),
                "{b}"
            );
        }
    }
}

#[test]
fn an_xc_gradient_call_without_room_for_x_recomputes_it() {
    // under a 1 GB cap la18's XC gradient engine sets up, but its call's workspaces with the
    // density pass's X kept (201 MB) do not fit beside the 971 MB it holds: the calculation
    // failed ("gradient_xc: device memory ... rerun with backend cpu"), then the CPU's
    // integrator computed the XC gradient; now the call's grant leaves X out (recomputed in the
    // contraction, the same bits) and the device computes it. A call that does not fit even
    // so falls back to the integrator, with the reason ("device call: device memory ...").
    let _cap = CAP.lock().unwrap_or_else(|e| e.into_inner());
    std::env::remove_var("LINUS_GPU_MEMORY_GIB");
    let opts = json!({"method": "b3lyp", "basis": "def2-svp", "jk": "df", "grid": "linus:3",
        "device_memory_gib": gib(1e9)});
    let Some(mut g) = calc(opts.clone(), "gpu") else {
        return;
    };
    let gg = g.gradient().unwrap();
    let b = serde_json::to_value(g.backend_used()).unwrap();
    assert_eq!(b["gradient_xc"], "gpu", "{b}");
    assert!(fallback_reason(&b, "gradient_xc").is_none(), "{b}");
    assert!(b.get("plan_violations").is_none(), "{b}");
    let gc = calc(opts, "cpu").unwrap().gradient().unwrap();
    let dg = gg
        .gradient
        .iter()
        .flatten()
        .zip(gc.gradient.iter().flatten())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max);
    assert!(dg < 1e-9, "GPU (X recomputed) against CPU: {dg:e}");
}

#[test]
fn scf_builds_that_do_not_fit_are_placed_before_the_first_iteration() {
    // la18 B3LYP/def2-SVP with the Ozaki exchange under a 1.5 GB cap: the J/K and XC engines
    // set up, but an SCF build (the Ozaki workspace, 207 MB, and the 256 MB it leaves free) does
    // not fit beside them. Both loops failed in their first iteration ("GPU backend, jk /
    // scf_loop: device memory ... rerun with backend cpu"); now the device loop gives way to the
    // host loop and the J/K to the CPU's before the first iteration, with their reasons
    let _cap = CAP.lock().unwrap_or_else(|e| e.into_inner());
    std::env::remove_var("LINUS_GPU_MEMORY_GIB");
    let base = json!({"method": "b3lyp", "basis": "def2-svp", "jk": "df"});
    let ec = calc(base.clone(), "cpu").unwrap().energy().unwrap();
    for scf_loop in ["host", "device"] {
        let mut o = base.clone();
        o["backend"] = json!({"scf_loop": scf_loop});
        o["precision"] = json!({"fp64": "emulated"});
        o["device_memory_gib"] = json!(gib(1.5e9));
        let Some(mut g) = calc(o, "gpu") else {
            return;
        };
        let eg = g.energy().unwrap();
        let b = serde_json::to_value(g.backend_used()).unwrap();
        assert_eq!(
            (&b["jk"], &b["scf_loop"]),
            (&json!("cpu"), &json!("host")),
            "{b}"
        );
        // (B3LYP at the first geometry: the closed-shell probe may run, its builds count)
        let why = fallback_reason(&b, "jk").unwrap_or_default();
        assert!(
            why.starts_with("device call: the SCF's largest J/K build needs")
                && why.contains("(the closed-shell probe's two-spin builds counted)"),
            "{scf_loop}: {b}"
        );
        if scf_loop == "device" {
            let why = fallback_reason(&b, "scf_loop").unwrap_or_default();
            assert!(why.starts_with("device call: the device loop's"), "{b}");
        }
        assert!(
            (eg.energy - ec.energy).abs() < 1e-9 && eg.iterations == ec.iterations,
            "{scf_loop}: gpu {} ({} iterations), cpu {} ({})",
            eg.energy,
            eg.iterations,
            ec.energy,
            ec.iterations
        );
    }
}

#[test]
fn a_hessian_under_an_8_gb_cap_keeps_its_response_on_the_device() {
    // the response J/K's exchange workspace (8 GiB by default) once took the whole of an 8 GB
    // cap and left its tensor a device budget of 0: every Hessian's response J/K fell back to
    // the CPU ("device setup: device memory ... over the limit of 0 bytes")
    let _cap = CAP.lock().unwrap_or_else(|e| e.into_inner());
    let opts = json!({"method": "b3lyp", "basis": "def2-svp", "jk": "df", "grid": "pyscf:2"});
    std::env::remove_var("LINUS_GPU_MEMORY_GIB");
    let Some(mut free) = calc(opts.clone(), "gpu") else {
        return;
    };
    let hf = free.hessian().unwrap();
    let bf = serde_json::to_value(free.backend_used()).unwrap();
    drop(free);
    let mut capped = opts.clone();
    capped["device_memory_gib"] = json!(gib(8e9));
    let mut g = calc(capped, "gpu").unwrap();
    let hg = g.hessian().unwrap();
    let b = serde_json::to_value(g.backend_used()).unwrap();
    assert_eq!(b["device"]["memory_cap_gib"], gib(8e9), "{b}");
    for k in ["jk", "response_jk", "response_xc"] {
        assert_eq!(b[k], "gpu", "{k}: {b}");
        assert_eq!(fallback_reason(&b, k), None, "{k}: {b}");
    }
    // the same response path as uncapped (the stored operator's reserve for the CPKS kernel
    // scales with the limit too) in the arithmetic the card's defaults give (FP64 on FP64-rich
    // cards, cuBLAS's emulation on the others: decision D27): the stored operator, or on cards
    // with little FP64 throughput the cheaper of it and the FP32 exchange by estimates that do
    // not depend on the cap (decision D37, `estimate_s`)
    let hessian = |b: &Value| {
        let r = b["response"].as_array().unwrap().iter();
        let h = r
            .clone()
            .find(|r| r["purpose"] == "hessian")
            .unwrap()
            .clone();
        (
            h["exchange"].clone(),
            h["jk"].clone(),
            h["xc"].clone(),
            h["estimate_s"].clone(),
        )
    };
    let (capped, uncapped) = (hessian(&b), hessian(&bf));
    let est = &capped.3;
    let want = if est.is_object() && est["tensor"].as_f64() < est["operator"].as_f64() {
        "tensor"
    } else {
        "operator"
    };
    assert_eq!(capped.0, json!(want), "{b}");
    assert_eq!(capped, uncapped, "capped {b}\nuncapped {bf}");
    let dh = hg
        .hessian
        .iter()
        .zip(&hf.hessian)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max);
    assert!(
        dh < 1e-9,
        "capped against uncapped Hessian: max |dH| {dh:e}"
    );
}
