// SPDX-License-Identifier: LGPL-3.0-or-later
//! JSON driver for validation and timing.
//!
//! ```text
//! cargo run --release -p linus-disp --example disp_json -- input.json [threads]
//! ```
//!
//! Input: `{"numbers": [8, 1, 1], "positions": [[x, y, z], ...] (bohr), "model": "d3bj" |
//! "d4", "functional": "b3lyp", "atm": false (D3 only; true = s9 1), "charge": 0.0 (D4),
//! "order": "energy" | "gradient" | "hessian", "params": {"s6": .., "s8": .., ...}
//! (optional, overrides the functional's), "repeat": 1}`.
//! Output (stdout): `{"energy", "gradient", "hessian", "params", "seconds"}`; `seconds` is
//! the best of `repeat` runs.

use linus_disp::*;
use serde_json::{json, Value};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let inp: Value = serde_json::from_str(&std::fs::read_to_string(&args[1]).expect("read input"))
        .expect("json");
    let threads: usize = args.get(2).map(|s| s.parse().unwrap()).unwrap_or(1);
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .unwrap();
    let z: Vec<u32> = inp["numbers"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as u32)
        .collect();
    let xyz: Vec<[f64; 3]> = inp["positions"]
        .as_array()
        .unwrap()
        .iter()
        .map(|p| {
            let p = p.as_array().unwrap();
            [
                p[0].as_f64().unwrap(),
                p[1].as_f64().unwrap(),
                p[2].as_f64().unwrap(),
            ]
        })
        .collect();
    let order = match inp["order"].as_str().unwrap_or("gradient") {
        "energy" => Order::Energy,
        "gradient" => Order::Gradient,
        "hessian" => Order::Hessian,
        o => panic!("unknown order {o}"),
    };
    let repeat = inp["repeat"].as_u64().unwrap_or(1).max(1);
    let functional = inp["functional"].as_str().unwrap_or("");
    let over = &inp["params"];
    let get = |k: &str, d: f64| over.get(k).and_then(|v| v.as_f64()).unwrap_or(d);
    let model = inp["model"].as_str().unwrap();
    let mut best = f64::INFINITY;
    let mut out = DispResult::default();
    let params_json;
    match model {
        "d3bj" => {
            let mut p = if functional.is_empty() {
                D3Params::new(1.0, 1.0, 0.4, 5.0)
            } else {
                d3bj_params(functional).unwrap()
            };
            if inp["atm"].as_bool().unwrap_or(false) {
                p.s9 = 1.0;
            }
            p = D3Params {
                s6: get("s6", p.s6),
                s8: get("s8", p.s8),
                a1: get("a1", p.a1),
                a2: get("a2", p.a2),
                s9: get("s9", p.s9),
                alp: get("alp", p.alp),
            };
            params_json =
                json!({"s6": p.s6, "s8": p.s8, "a1": p.a1, "a2": p.a2, "s9": p.s9, "alp": p.alp});
            let opts = D3Options {
                order,
                ..Default::default()
            };
            for _ in 0..repeat {
                let t = std::time::Instant::now();
                out = pool.install(|| d3bj(&z, &xyz, &p, &opts)).unwrap();
                best = best.min(t.elapsed().as_secs_f64());
            }
        }
        "d4" => {
            let charge = inp["charge"].as_f64().unwrap_or(0.0);
            let mut p = d4_params(functional).unwrap();
            p = D4Params {
                s6: get("s6", p.s6),
                s8: get("s8", p.s8),
                a1: get("a1", p.a1),
                a2: get("a2", p.a2),
                s9: get("s9", p.s9),
                alp: get("alp", p.alp),
                ga: get("ga", p.ga),
                gc: get("gc", p.gc),
                wf: get("wf", p.wf),
            };
            params_json = json!({"s6": p.s6, "s8": p.s8, "a1": p.a1, "a2": p.a2, "s9": p.s9, "alp": p.alp, "ga": p.ga, "gc": p.gc, "wf": p.wf});
            let opts = D4Options {
                order,
                ..Default::default()
            };
            for _ in 0..repeat {
                let t = std::time::Instant::now();
                out = pool.install(|| d4(&z, &xyz, charge, &p, &opts)).unwrap();
                best = best.min(t.elapsed().as_secs_f64());
            }
        }
        m => panic!("unknown model {m}"),
    }
    let mut res = json!({
        "energy": out.energy,
        "gradient": out.gradient,
        "hessian": out.hessian,
        "params": params_json,
        "seconds": best,
        "threads": threads,
    });
    // optional D3 gradient split into explicit two-body, ATM and CN chain-rule parts
    if model == "d3bj" && inp["parts"].as_bool().unwrap_or(false) {
        let f = |k: &str| params_json[k].as_f64().unwrap();
        let p = D3Params {
            s6: f("s6"),
            s8: f("s8"),
            a1: f("a1"),
            a2: f("a2"),
            s9: f("s9"),
            alp: f("alp"),
        };
        let parts = D3Model::new(&z)
            .unwrap()
            .gradient_parts(&xyz, &p, &D3Cutoffs::default());
        res["parts"] = json!({
            "two_body": parts.two_body,
            "three_body": parts.three_body,
            "cn_chain": parts.cn_chain,
            "dedcn_two_body": parts.dedcn_two_body,
            "dedcn_three_body": parts.dedcn_three_body,
        });
    }
    // optional 5-point finite differences: gradient from energies, Hessian from analytic
    // gradients (step in bohr)
    if let Some(h) = inp["fd_step"].as_f64() {
        let eval = |x: &[[f64; 3]], o: Order| -> DispResult {
            match model {
                "d3bj" => {
                    let p = D3Params {
                        s6: params_json["s6"].as_f64().unwrap(),
                        s8: params_json["s8"].as_f64().unwrap(),
                        a1: params_json["a1"].as_f64().unwrap(),
                        a2: params_json["a2"].as_f64().unwrap(),
                        s9: params_json["s9"].as_f64().unwrap(),
                        alp: params_json["alp"].as_f64().unwrap(),
                    };
                    d3bj(
                        &z,
                        x,
                        &p,
                        &D3Options {
                            order: o,
                            ..Default::default()
                        },
                    )
                    .unwrap()
                }
                _ => {
                    let f = |k: &str| params_json[k].as_f64().unwrap();
                    let p = D4Params {
                        s6: f("s6"),
                        s8: f("s8"),
                        a1: f("a1"),
                        a2: f("a2"),
                        s9: f("s9"),
                        alp: f("alp"),
                        ga: f("ga"),
                        gc: f("gc"),
                        wf: f("wf"),
                    };
                    let charge = inp["charge"].as_f64().unwrap_or(0.0);
                    d4(
                        &z,
                        x,
                        charge,
                        &p,
                        &D4Options {
                            order: o,
                            ..Default::default()
                        },
                    )
                    .unwrap()
                }
            }
        };
        let n3 = 3 * z.len();
        let cols: Vec<(f64, Vec<f64>)> = pool.install(|| {
            use rayon::prelude::*;
            (0..n3)
                .into_par_iter()
                .map(|c| {
                    let disp = |s: f64| {
                        let mut x = xyz.clone();
                        x[c / 3][c % 3] += s * h;
                        eval(&x, Order::Gradient)
                    };
                    let (m2, m1, p1, p2) = (disp(-2.0), disp(-1.0), disp(1.0), disp(2.0));
                    let ge =
                        (m2.energy - 8.0 * m1.energy + 8.0 * p1.energy - p2.energy) / (12.0 * h);
                    let col: Vec<f64> = (0..n3)
                        .map(|r| {
                            let g = |d: &DispResult| d.gradient[r / 3][r % 3];
                            (g(&m2) - 8.0 * g(&m1) + 8.0 * g(&p1) - g(&p2)) / (12.0 * h)
                        })
                        .collect();
                    (ge, col)
                })
                .collect()
        });
        let gfd: Vec<f64> = cols.iter().map(|c| c.0).collect();
        let mut hfd = vec![0.0; n3 * n3];
        for (c, (_, col)) in cols.iter().enumerate() {
            for r in 0..n3 {
                hfd[r * n3 + c] = col[r];
            }
        }
        res["gradient_fd"] = json!(gfd);
        res["hessian_fd"] = json!(hfd);
    }
    println!("{}", serde_json::to_string(&res).unwrap());
}
