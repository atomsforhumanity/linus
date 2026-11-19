//! Evaluate every case of a reference fixture with linus-xc and write our
//! outputs as JSON (same schema as the fixture's `out`), for analysis in
//! Python: `cargo run -p linus-xc --example eval_fixture -- in.json out.json [order]`.

use linus_xc::{Functional, Out, XcInput, XcOutput};
use serde_json::{json, Map, Value};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let fx: Value = serde_json::from_str(&std::fs::read_to_string(&args[1]).unwrap()).unwrap();
    let vec = |v: &Value| -> Vec<f64> {
        v.as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_f64().unwrap_or(f64::NAN))
            .collect()
    };
    let order: usize = args.get(3).map_or(2, |s| s.parse().unwrap());
    let mut cases = Vec::new();
    for case in fx["cases"].as_array().unwrap() {
        let spin = case["spin"].as_str().unwrap();
        let inp = &fx["inputs"][spin];
        let (rho, sigma, tau) = (vec(&inp["rho"]), vec(&inp["sigma"]), vec(&inp["tau"]));
        let f = Functional::new(case["name"].as_str().unwrap(), spin == "polarized").unwrap();
        let mut out = XcOutput::new();
        f.eval(
            order,
            &XcInput {
                rho: &rho,
                sigma: Some(&sigma),
                lapl: None,
                tau: Some(&tau),
            },
            &mut out,
        )
        .unwrap();
        let mut m = Map::new();
        for o in Out::ALL {
            if !out.get(o).is_empty() {
                m.insert(
                    o.name().into(),
                    json!(out
                        .get(o)
                        .iter()
                        .map(|x| x.is_finite().then_some(*x))
                        .collect::<Vec<_>>()),
                );
            }
        }
        cases.push(json!({"name": case["name"], "spin": spin, "out": m}));
    }
    std::fs::write(
        &args[2],
        serde_json::to_string(&json!({ "cases": cases })).unwrap(),
    )
    .unwrap();
}
