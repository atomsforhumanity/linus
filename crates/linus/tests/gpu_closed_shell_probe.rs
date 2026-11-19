//! The closed-shell probe (`scf.stability_closed_probe`, decision D36) on the GPU (feature `gpu`): on
//! the device SCF loop (the default under D27/D31) it runs on the device, on the GPU's host loop
//! as on the CPU. The Ritz value agrees CPU = GPU host loop = device loop (the start vector is
//! phase-fixed, D29), the device result repeats bit for bit, and with the probe on the result is
//! the one without it, bit for bit (a VV10 functional and the PCM on the device included). Without
//! a usable device the tests skip, unless `LINUS_GPU_REQUIRED=1`.
#![cfg(feature = "gpu")]

use linus::driver::report::Report;
use linus::{CalcOptions, CalcType, LinusError, Session, Structure};
use linus_scf::{ClosedShellProbe, ProbeKind};
use serde_json::{json, Value};

const A: f64 = 1.0 / linus_core::units::BOHR_ANGSTROM;

/// The session of `mol` with the JSON settings `opts` on `backend` (`cpu`, or the GPU with the
/// loop `host` or `device`); None without a usable device (a skip).
fn calc(mol: &Structure, opts: &Value, backend: &str) -> Option<Session> {
    let mut o = opts.clone();
    o["backend"] = match backend {
        "cpu" => json!("cpu"),
        l => json!({"kind": "gpu", "scf_loop": l}),
    };
    o["threads"] = json!(4);
    if o["method"] == "hf" {
        // Hartree-Fock has no XC grid (D41: its grid settings are input errors)
        o.as_object_mut().unwrap().remove("grid");
    }
    match Session::new(mol.clone(), CalcOptions::from_json(o).unwrap()) {
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

fn run(mol: &Structure, opts: &Value, backend: &str, what: CalcType) -> Option<Report> {
    let mut c = calc(mol, opts, backend)?;
    let r = c.run(what).unwrap_or_else(|e| panic!("{backend}: {e}"));
    if backend != "cpu" {
        let used = serde_json::to_value(c.backend_used()).unwrap()["scf_loop"].clone();
        assert_eq!(used, backend);
    }
    Some(r)
}

/// H2 at `r` Angstrom (RKS gap 0.143 Eh at 2.0 with B3LYP/def2-SVP: probed, unstable).
fn h2(r: f64) -> Structure {
    Structure::new(vec![1, 1], vec![[0.0, 0.0, 0.0], [0.0, 0.0, r * A]], 0, 1).unwrap()
}

/// p-benzyne (singlet diradical; RKS gap 0.067 Eh with B3LYP/def2-SVP, a broken-symmetry UKS
/// solution 4.0e-2 Eh lower; the ROBUST hard set).
fn p_benzyne() -> Structure {
    Structure::from_xyz_str(
        "10\np-benzyne\n\
         C -1.212436 -0.7 0.0\nC -1.212436 0.7 0.0\nC 0.0 -1.4 0.0\nC 0.0 1.4 0.0\n\
         C 1.212436 -0.7 0.0\nC 1.212436 0.7 0.0\nH -2.1477435199225043 1.2399998547929565 0.0\n\
         H 0.0 -2.48 0.0\nH 0.0 2.48 0.0\nH 2.1477435199225043 -1.2399998547929565 0.0\n",
        0,
        1,
    )
    .unwrap()
}

fn settings(method: &str) -> Value {
    json!({"method": method, "basis": "def2-svp", "jk": "df", "grid": "pyscf:3"})
}

/// The JSON values `a` and `b` agree: numbers within `tol` (relative to 1), the rest equal; the
/// keys in `skip` (wall times) are not compared.
fn same_json(a: &Value, b: &Value, tol: f64, skip: &[&str]) -> bool {
    match (a, b) {
        (Value::Object(x), Value::Object(y)) => {
            x.len() == y.len()
                && x.iter().all(|(k, v)| {
                    skip.contains(&k.as_str())
                        || y.get(k).is_some_and(|w| same_json(v, w, tol, skip))
                })
        }
        (Value::Array(x), Value::Array(y)) => {
            x.len() == y.len() && x.iter().zip(y).all(|(v, w)| same_json(v, w, tol, skip))
        }
        (Value::Number(x), Value::Number(y)) => {
            let (x, y) = (x.as_f64().unwrap(), y.as_f64().unwrap());
            (x - y).abs() <= tol * x.abs().max(1.0)
        }
        _ => a == b,
    }
}

fn probe(r: &Report) -> ClosedShellProbe {
    r.resolved
        .record
        .scf
        .closed_shell_probe
        .expect("a closed-shell record")
}

/// CPU, GPU host loop and device loop at the extreme preset: the same record (Ritz value within
/// 1e-9 Eh, kind, products), energies within 1e-9; the device's repeats bit for bit.
#[test]
fn the_probe_agrees_cpu_host_and_device() {
    for (name, mol, method) in [
        ("H2 2.0 b3lyp", h2(2.0), "b3lyp"),
        ("H2 2.0 hf", h2(2.0), "hf"),
        ("p-benzyne b3lyp", p_benzyne(), "b3lyp"),
        ("p-benzyne wb97x-d3bj", p_benzyne(), "wb97x-d3bj"),
    ] {
        // HF and the range-separated hybrid's gaps lie above the default gate: a wider one. The
        // extreme preset: at the default (tight) thresholds the one-sided products also carry
        // the converged point's residual over h (p-benzyne: 3e-6 Eh from the extreme preset's
        // Ritz value on the CPU), which the backends' SCF paths leave differently (2e-6)
        let mut o = settings(method);
        o["scf"] = json!({"stability_closed_gap": 1.0, "convergence": "extreme"});
        if method == "hf" {
            o["scf"]["stability_closed_probe"] = json!(2);
        }
        let mut out = Vec::new();
        for b in ["cpu", "host", "device"] {
            let Some(r) = run(&mol, &o, b, CalcType::Energy) else {
                return;
            };
            out.push(r);
        }
        let p: Vec<ClosedShellProbe> = out.iter().map(probe).collect();
        eprintln!(
            "{name}:\n  cpu    {:?}\n  host   {:?}\n  device {:?}",
            p[0], p[1], p[2]
        );
        for (q, b) in p[1..].iter().zip(["host", "device"]) {
            assert!(
                (q.eigenvalue.unwrap() - p[0].eigenvalue.unwrap()).abs() < 1e-9
                    && q.kind == p[0].kind
                    && q.fock_builds == p[0].fock_builds
                    && q.unstable == Some(true)
                    && q.kind == Some(ProbeKind::External),
                "{name}: {b} {q:?} vs cpu {:?}",
                p[0]
            );
        }
        for r in &out[1..] {
            assert!((r.energy - out[0].energy).abs() < 1e-9, "{name}");
        }
        let again = run(&mol, &o, "device", CalcType::Energy).unwrap();
        assert_eq!(again.energy.to_bits(), out[2].energy.to_bits(), "{name}");
        assert_eq!(
            probe(&again).eigenvalue.unwrap().to_bits(),
            p[2].eigenvalue.unwrap().to_bits(),
            "{name}"
        );
    }
}

/// After the open-shell stability probe and the zero-mode check of open-shell Hessians on the
/// device loop, a -V functional's result reports the converged state's VV10 energy, bitwise that
/// without them (it was the last rotated build's), on both GPU loops.
#[test]
fn vv10_is_reported_at_the_converged_state_after_open_shell_probes() {
    let nh3p = Structure::from_xyz_str(
        "4\nNH3+\nN 0 0 0\nH 1.02 0 0\nH -0.51 0.883 0\nH -0.51 -0.883 0\n",
        1,
        2,
    )
    .unwrap();
    let js = |r: &Report, k: &str| serde_json::to_value(r).unwrap()[k].to_string();
    for lp in ["device", "host"] {
        let mut o = settings("wb97m-v");
        o["scf"] = json!({"stability_probe": 2});
        let Some(probed) = run(&nh3p, &o, lp, CalcType::Energy) else {
            return;
        };
        o["scf"] = json!({"stability_probe": 0});
        let plain = run(&nh3p, &o, lp, CalcType::Energy).unwrap();
        assert!(
            probed.scf.stability.first().is_some_and(|c| c.probe),
            "{lp}"
        );
        assert_eq!(probed.energy.to_bits(), plain.energy.to_bits(), "{lp}");
        assert_eq!(js(&probed, "vv10"), js(&plain, "vv10"), "{lp}: probe");
        // the mode check of a Hessian (the probe off: the check probes the converged point on the
        // device, finds no negative mode, and the result is the one without it)
        let mut h = settings("wb97m-v");
        h["scf"] = json!({"stability_probe": 0, "stability_hessian_check": 6});
        let checked = run(&nh3p, &h, lp, CalcType::Hessian).unwrap();
        h["scf"] = json!({"stability_probe": 0, "stability_hessian_check": 0});
        let unchecked = run(&nh3p, &h, lp, CalcType::Hessian).unwrap();
        assert!(checked
            .resolved
            .record
            .response
            .as_ref()
            .unwrap()
            .mode_check
            .is_some());
        assert_eq!(checked.hessian, unchecked.hessian, "{lp}");
        assert_eq!(
            js(&checked, "vv10"),
            js(&unchecked, "vv10"),
            "{lp}: mode check"
        );
    }
}

/// On the device loop, the result with the probe is the one without it, bit for bit: energy,
/// gradient, dipole, energy components, VV10's energy (wB97M-V) and the solvation summary (C-PCM
/// on the device).
#[test]
fn the_device_result_is_unchanged_bit_for_bit() {
    for (method, solvent) in [("b3lyp", false), ("wb97m-v", false), ("b3lyp", true)] {
        let mk = |n: usize| {
            let mut o = settings(method);
            o["scf"] = json!({"stability_closed_probe": n, "stability_closed_gap": 1.0});
            if solvent {
                o["solvent"] = json!("water");
            }
            o
        };
        let Some(on) = run(&h2(2.0), &mk(2), "device", CalcType::Gradient) else {
            return;
        };
        let off = run(&h2(2.0), &mk(0), "device", CalcType::Gradient).unwrap();
        let case = format!("{method} pcm {solvent}");
        let p = probe(&on);
        eprintln!("{case}: {p:?}");
        assert!(p.is_unstable(), "{case}: {p:?}");
        assert_eq!(on.energy.to_bits(), off.energy.to_bits(), "{case}");
        assert_eq!(on.gradient, off.gradient, "{case}");
        assert_eq!(on.dipole, off.dipole, "{case}");
        let js = |r: &Report, k: &str| serde_json::to_value(r).unwrap()[k].clone();
        assert_eq!(js(&on, "vv10"), js(&off, "vv10"), "{case}");
        // the PCM's restoring pass is incremental from the probe's last rotated state: its
        // summary (and the components' solvation_electrostatic) agree to round-off
        let tol = if solvent { 1e-14 } else { 0.0 };
        for k in ["components", "solvation"] {
            assert!(
                same_json(&js(&on, k), &js(&off, k), tol, &["time"]),
                "{case} {k}: {} vs {}",
                js(&on, k),
                js(&off, k)
            );
        }
    }
}
