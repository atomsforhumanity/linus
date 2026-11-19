//! eval_ao <in.json> <basis> : prints AO values+derivs (order 2) as JSON [comp][ao][pt]
use linus_ints::{basis::BasisDef, points::ao, BasisSet};
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let v: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&a[1]).unwrap()).unwrap();
    let c: Vec<[f64; 3]> = v["coords"]
        .as_array()
        .unwrap()
        .iter()
        .map(|r| {
            let r = r.as_array().unwrap();
            [
                r[0].as_f64().unwrap(),
                r[1].as_f64().unwrap(),
                r[2].as_f64().unwrap(),
            ]
        })
        .collect();
    let pts: Vec<Vec<f64>> = v["pts"]
        .as_array()
        .unwrap()
        .iter()
        .map(|r| {
            r.as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_f64().unwrap())
                .collect()
        })
        .collect();
    let def = BasisDef::load(&a[2], None).unwrap();
    let b = BasisSet::new(&def, &[8, 1, 1], &c, true).unwrap();
    let (x, y, z): (Vec<f64>, Vec<f64>, Vec<f64>) = (
        pts.iter().map(|p| p[0]).collect(),
        pts.iter().map(|p| p[1]).collect(),
        pts.iter().map(|p| p[2]).collect(),
    );
    let shells: Vec<usize> = (0..b.shells.len()).collect();
    let mut out = vec![0.0; ao::ncomp(2) * b.nbf * x.len()];
    ao::eval_ao(&b, &shells, &x, &y, &z, 2, &mut out);
    println!("{}", serde_json::to_string(&out).unwrap());
}
