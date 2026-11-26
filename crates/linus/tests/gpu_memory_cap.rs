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
const U1AN39: &str = include_str!("../../../bench/molecules/u1an39.xyz");

fn calc(opts: Value, backend: &str) -> Option<Session> {
    calc_of(Structure::from_xyz_str(LA18, 0, 1).unwrap(), opts, backend)
}

/// A session of `mol` with `opts` on `backend` (None: no usable GPU, a skip).
fn calc_of(mol: Structure, opts: Value, backend: &str) -> Option<Session> {
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
fn scf_builds_are_planned_or_placed_before_the_first_iteration() {
    // la18 B3LYP/def2-SVP with the Ozaki exchange under tight caps. Both loops once failed in
    // their first iteration ("GPU backend, jk / scf_loop: device memory ... rerun with backend
    // cpu"): the J/K engine took the free memory and its builds (the Ozaki workspace) did not fit
    // beside it. The J/K's storage plan now counts its builds within the memory plan's budget
    // (REVIEW JK-1: at 1.5 GB the J/K stays on the device), and what still does not fit (the
    // device loop's state, a probe's two-spin builds) gives way before the first iteration with
    // its reason: at every cap each family is on the device or on the CPU with a reason, never a
    // failure, and the energy is the CPU's
    let _cap = CAP.lock().unwrap_or_else(|e| e.into_inner());
    std::env::remove_var("LINUS_GPU_MEMORY_GIB");
    let base = json!({"method": "b3lyp", "basis": "def2-svp", "jk": "df"});
    let ec = calc(base.clone(), "cpu").unwrap().energy().unwrap();
    for cap in [1.0e9, 1.25e9, 1.5e9] {
        for scf_loop in ["host", "device"] {
            let mut o = base.clone();
            o["backend"] = json!({"scf_loop": scf_loop});
            o["precision"] = json!({"fp64": "emulated"});
            o["device_memory_gib"] = json!(gib(cap));
            let Some(mut g) = calc(o, "gpu") else {
                return;
            };
            let eg = g.energy().unwrap();
            let b = serde_json::to_value(g.backend_used()).unwrap();
            let tag = format!("{cap:e} B, {scf_loop} loop");
            assert!(b.get("plan_violations").is_none(), "{tag}: {b}");
            if b["jk"] == "cpu" {
                let why = fallback_reason(&b, "jk").unwrap_or_default();
                assert!(
                    why.starts_with("device setup") || why.starts_with("device call"),
                    "{tag}: {b}"
                );
            }
            if scf_loop == "device" && b["scf_loop"] == "host" {
                let why = fallback_reason(&b, "scf_loop").unwrap_or_default();
                assert!(!why.is_empty(), "{tag}: {b}");
            }
            if cap == 1.5e9 {
                assert_eq!(b["jk"], "gpu", "{tag}: {b}");
            }
            eprintln!(
                "{tag}: jk {} scf_loop {} ({} iterations)",
                b["jk"], b["scf_loop"], eg.iterations
            );
            assert!(
                (eg.energy - ec.energy).abs() < 1e-9 && eg.iterations == ec.iterations,
                "{tag}: gpu {} ({} iterations), cpu {} ({})",
                eg.energy,
                eg.iterations,
                ec.energy,
                ec.iterations
            );
        }
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

#[test]
fn the_density_fitting_plan_takes_the_memory_plans_budget() {
    // la18 B3LYP/def2-SVP under a 2 GiB cap: the SCF's DF-J/K plans its storage and K block
    // within the memory plan's share (the limit less the reserve for the other engines), not the
    // memory free when its setup runs (the XC grid's partition sets up beside it), so two sessions
    // at the cap record the same plan and give the same energy, bit for bit
    let _cap = CAP.lock().unwrap_or_else(|e| e.into_inner());
    std::env::remove_var("LINUS_GPU_MEMORY_GIB");
    let opts = json!({"method": "b3lyp", "basis": "def2-svp", "jk": "df", "grid": "pyscf:2",
        "device_memory_gib": 2.0});
    let mut runs = Vec::new();
    for _ in 0..2 {
        let Some(mut g) = calc(opts.clone(), "gpu") else {
            return;
        };
        let rep = g.run(linus::CalcType::Energy).unwrap();
        let r = serde_json::to_value(&rep.resolved).unwrap();
        let e = rep.energy;
        runs.push((e.to_bits(), r["record"]["df"].clone(), r));
    }
    let (b0, df0, r0) = &runs[0];
    let (b1, df1, _) = &runs[1];
    assert_eq!((b0, df0), (b1, df1), "{r0}");
    // the budget: the limit less the memory plan's reserve for the other engines
    // (max(8 GiB, limit / 10), at most limit / 4)
    let limit = r0["record"]["backend"]["device"]["memory_limit_gib"]
        .as_f64()
        .unwrap();
    let reserve = 8f64.max(limit / 10.0).min(limit / 4.0);
    let budget = df0["budget_gib"].as_f64().unwrap();
    assert!(
        (budget - (limit - reserve)).abs() < 1e-9,
        "{df0}, limit {limit}"
    );
    assert!(df0["k_block"].as_u64().unwrap() > 0, "{df0}");
}

#[test]
fn the_k_block_steps_down_for_the_other_engines_per_call_memory() {
    // u1an39 (the neutral doublet) wB97M-V/def2-SVP UKS with C-PCM in FP64 under an 8 GB cap:
    // the J/K's plan sized its panel block within the limit less the other engines' reserve,
    // and VV10's and the PCM's per-call memory beside that reserve ran the second SCF build out
    // of device memory (REVIEW JK-1, found by the H200 validate). The SCF's check before its first
    // build now steps the block down to what the engines left, says so in the record, and two
    // sessions take the same block and give the same energy, bit for bit
    let _cap = CAP.lock().unwrap_or_else(|e| e.into_inner());
    std::env::remove_var("LINUS_GPU_MEMORY_GIB");
    let opts = json!({"method": "wb97m-v", "basis": "def2-svp", "jk": "df",
        "solvent": {"name": "thf", "model": "cpcm"}, "precision": {"fp64": "native"},
        "backend": {"scf_loop": "host"}, "device_memory_gib": gib(8e9)});
    let mut runs = Vec::new();
    for _ in 0..2 {
        let mol = Structure::from_xyz_str(U1AN39, 0, 2).unwrap();
        let Some(mut g) = calc_of(mol, opts.clone(), "gpu") else {
            return;
        };
        let rep = g.run(linus::CalcType::Energy).unwrap();
        let r = serde_json::to_value(&rep.resolved).unwrap();
        let b = &r["record"]["backend"];
        assert_eq!(b["jk"], "gpu", "{r}");
        assert!(b.get("plan_violations").is_none(), "{r}");
        let df = r["record"]["df"].clone();
        let rung = df["rung"].as_str().unwrap_or_default().to_string();
        assert!(rung.contains("stepped down"), "{df}");
        runs.push((rep.energy.to_bits(), df["k_block"].clone(), rung));
    }
    assert_eq!(runs[0], runs[1]);
}
