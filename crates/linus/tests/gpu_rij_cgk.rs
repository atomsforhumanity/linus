//! rij_cgk's SCF on the GPU (feature `gpu`): the host loop and the device loop (the grid-K term
//! with its E_K hook, the engines taken and given back) against the CPU's rij_cgk, and the
//! memory-lean DF-J (`backend.df_j: "direct"`) under df's pure functionals and rij_cgk, recorded
//! in `resolved.record.backend.j` and resolved in `resolved.settings`. Without a usable device
//! the tests skip, unless `LINUS_GPU_REQUIRED=1`.
#![cfg(feature = "gpu")]

use linus::driver::report::Report;
use linus::{CalcOptions, CalcType, LinusError, Session, Structure};
use serde_json::{json, Value};

const WATER: &str = "3\nwater\nO 0.02 -0.01 0.12\nH 0.03 0.78 -0.47\nH -0.01 -0.74 -0.49\n";

/// The energy of `opts` on `backend` (`cpu`, or the GPU with the loop `host` or `device`), the
/// GPU with `backend.df_j` = `df_j` where given; None without a usable device.
fn energy(
    charge: i32,
    mult: u32,
    opts: &Value,
    backend: &str,
    df_j: Option<&str>,
) -> Option<Report> {
    let mol = Structure::from_xyz_str(WATER, charge, mult).unwrap();
    let mut o = opts.clone();
    o["backend"] = match backend {
        "cpu" => json!("cpu"),
        l => match df_j {
            Some(j) => json!({"kind": "gpu", "scf_loop": l, "df_j": j}),
            None => json!({"kind": "gpu", "scf_loop": l}),
        },
    };
    o["threads"] = json!(4);
    match Session::new(mol, CalcOptions::from_json(o).unwrap()) {
        Ok(mut c) => Some(c.run(CalcType::Energy).unwrap()),
        Err(LinusError::Backend { message, .. })
            if std::env::var("LINUS_GPU_REQUIRED").as_deref() != Ok("1") =>
        {
            eprintln!("no usable GPU, skipping: {message}");
            None
        }
        Err(e) => panic!("{e}"),
    }
}

/// The GPU's host and device loops against the CPU: energies within 1e-10 Eh (docs/gpu.md's
/// bar), the same iterations.
fn against_cpu(
    case: &str,
    charge: i32,
    mult: u32,
    opts: &Value,
    df_j: Option<&str>,
) -> Option<[Report; 3]> {
    let c = energy(charge, mult, opts, "cpu", None).unwrap();
    let h = energy(charge, mult, opts, "host", df_j)?;
    let d = energy(charge, mult, opts, "device", df_j)?;
    for (name, g) in [("host", &h), ("device", &d)] {
        eprintln!(
            "{case} {name}: E {:.12} ({} it) cpu {:.12} ({} it)",
            g.energy, g.iterations, c.energy, c.iterations
        );
        assert!((g.energy - c.energy).abs() < 1e-10, "{case} {name}");
        assert_eq!(g.iterations, c.iterations, "{case} {name}");
    }
    Some([c, h, d])
}

/// The record's J engine and the resolved setting.
fn j_record(r: &Report) -> (Value, Value) {
    let v = serde_json::to_value(r).unwrap();
    (
        v.pointer("/resolved/record/backend/j")
            .cloned()
            .unwrap_or(Value::Null),
        v.pointer("/resolved/settings/backend/df_j")
            .cloned()
            .unwrap_or(Value::Null),
    )
}

#[test]
fn rij_cgk_on_the_gpu_matches_the_cpu() {
    let opts = |method: &str| {
        json!({"method": method, "basis": "def2-svp", "grid": "pyscf:3",
               "jk": {"kind": "rij_cgk", "k_grid": "gridx:1"},
               "scf": {"e_tol": 1e-11, "g_tol": 1e-8}})
    };
    if against_cpu("b3lyp rks", 0, 1, &opts("b3lyp"), None).is_none() {
        return;
    }
    against_cpu("b3lyp uks cation", 1, 2, &opts("b3lyp"), None);
    against_cpu("wb97x-d3bj rks", 0, 1, &opts("wb97x-d3bj"), None);
    // the J on the memory-lean engine (forced at this size), K on grid-K
    let r = against_cpu(
        "b3lyp rks df_j direct",
        0,
        1,
        &opts("b3lyp"),
        Some("direct"),
    )
    .unwrap();
    for g in &r[1..] {
        let (j, setting) = j_record(g);
        assert_eq!(j["engine"], "direct", "{j}");
        assert!(setting.is_null() || setting == "direct", "{setting}");
    }
}

#[test]
fn the_memory_lean_j_of_a_pure_functional() {
    let o = json!({"method": "pbe", "basis": "def2-svp", "grid": "pyscf:3",
                   "scf": {"e_tol": 1e-11, "g_tol": 1e-8}});
    // auto keeps the DF engine at this size: resolved with its reason, nothing in the record
    let Some(d) = energy(0, 1, &o, "device", None) else {
        return;
    };
    let (j, setting) = j_record(&d);
    assert!(j.is_null(), "{j}");
    assert_eq!(setting, "stored");
    let r = against_cpu("pbe df_j direct", 0, 1, &o, Some("direct")).unwrap();
    for g in &r[1..] {
        let (j, _) = j_record(g);
        assert_eq!(j["engine"], "direct", "{j}");
        assert!(j["kernels"] == "hermite" || j["kernels"] == "eri3c", "{j}");
    }
}
