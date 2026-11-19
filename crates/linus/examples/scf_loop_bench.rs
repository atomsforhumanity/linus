//! Decision timing of the SCF loops on the GPU (feature `gpu`): one molecule, method and basis,
//! the loops (`backend.scf_loop`) interleaved in one process, a warm-up round and `reps` timed
//! rounds; one JSON line per run with the SCF's timers, iterations, energy and stability
//! records.
//!
//!   cargo run --release -p linus --features gpu --example scf_loop_bench -- \
//!       mol.xyz charge mult method basis reps [loops, default host,device] [JSON settings]
//!
//! The settings are merged into {method, basis, jk: df, backend: gpu, threads: 8}.

use linus::{CalcOptions, CalcType, Session, Structure};
use serde_json::{json, Value};

fn merge(a: &mut Value, b: &Value) {
    match (a, b) {
        (Value::Object(a), Value::Object(b)) => {
            for (k, v) in b {
                merge(a.entry(k.clone()).or_insert(Value::Null), v);
            }
        }
        (a, b) => *a = b.clone(),
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 7 {
        eprintln!("usage: scf_loop_bench mol.xyz charge mult method basis reps [loops] [json]");
        std::process::exit(2);
    }
    let text = std::fs::read_to_string(&args[1]).expect("the xyz file");
    let charge: i32 = args[2].parse().expect("charge");
    let mult: u32 = args[3].parse().expect("multiplicity");
    let reps: usize = args[6].parse().expect("reps");
    let loops: Vec<String> = args
        .get(7)
        .map_or("host,device", |s| s.as_str())
        .split(',')
        .map(String::from)
        .collect();
    let extra: Value = args.get(8).map_or(json!({}), |s| {
        serde_json::from_str(s).expect("JSON settings")
    });
    let mol = Structure::from_xyz_str(&text, charge, mult).expect("the molecule");
    for rep in 0..=reps {
        for l in &loops {
            let mut o = json!({"method": args[4], "basis": args[5], "jk": "df",
                               "backend": {"kind": "gpu", "scf_loop": l}, "threads": 8});
            merge(&mut o, &extra);
            let opts = CalcOptions::from_json(o).expect("settings");
            let t0 = std::time::Instant::now();
            let mut c = Session::new(mol.clone(), opts).expect("session");
            let r = c.run(CalcType::Energy).expect("the SCF");
            let wall = t0.elapsed().as_secs_f64();
            let b = serde_json::to_value(c.backend_used()).unwrap();
            let line = json!({
                "loop": l, "rep": rep, "wall": wall, "energy": r.energy,
                "iterations": r.iterations, "fock_builds": r.scf.fock_builds,
                "scf_loop": b["scf_loop"], "stability": r.scf.stability,
                "stability_probe": r.resolved.record.scf.stability_probe,
                "timings": r.timings, "nbf": r.nbf,
            });
            println!("{line}");
        }
    }
}
