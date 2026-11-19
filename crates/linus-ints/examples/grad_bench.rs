//! Time an RHF gradient from given occupied orbitals: grad_bench <input.json>
//! input: {"z": [...], "coords": [[x,y,z],...] (bohr), "basis": "def2-svp", "nocc": n,
//!         "c_occ": [nbf*nocc row-major], "e_occ": [...], "thresh": 1e-12, "repeat": 1}
//! Prints {"nbf", "t_setup", "t_1e", "t_2e", "t_total", "grad"} as JSON.
use linus_ints::contract::DensityTerms;
use linus_ints::reference::oneshot;
use linus_ints::{basis::BasisDef, BasisSet, DerivOrder, PairOptions, PairSet};
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let inp: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&args[1]).unwrap()).unwrap();
    let f = |v: &serde_json::Value| -> Vec<f64> {
        v.as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_f64().unwrap())
            .collect()
    };
    let z: Vec<u32> = inp["z"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as u32)
        .collect();
    let positions: Vec<[f64; 3]> = inp["coords"]
        .as_array()
        .unwrap()
        .iter()
        .map(|r| {
            let r = f(r);
            [r[0], r[1], r[2]]
        })
        .collect();
    let nocc = inp["nocc"].as_u64().unwrap() as usize;
    let c = f(&inp["c_occ"]);
    let eps = f(&inp["e_occ"]);
    let thresh = inp["thresh"].as_f64().unwrap_or(1e-12);
    let repeat = inp["repeat"].as_u64().unwrap_or(1) as usize;
    let def = BasisDef::load(inp["basis"].as_str().unwrap(), None).unwrap();
    let basis = BasisSet::new(&def, &z, &positions, true).unwrap();
    let n = basis.nbf;
    let mut d = vec![0.0; n * n];
    let mut w = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            let (mut sd, mut sw) = (0.0, 0.0);
            for k in 0..nocc {
                let x = c[i * nocc + k] * c[j * nocc + k];
                sd += x;
                sw += eps[k] * x;
            }
            d[i * n + j] = 2.0 * sd;
            w[i * n + j] = -2.0 * sw;
        }
    }
    let charges: Vec<(f64, [f64; 3])> = z
        .iter()
        .zip(&positions)
        .map(|(&q, &x)| (q as f64, x))
        .collect();
    let mut best = (f64::MAX, 0.0, 0.0, 0.0);
    let mut grad = Vec::new();
    for _ in 0..repeat {
        let t0 = Instant::now();
        let pairs = PairSet::build(&basis, PairOptions::with_order(DerivOrder::One));
        let t_setup = t0.elapsed().as_secs_f64();
        let t1 = Instant::now();
        let mut g = oneshot::overlap_kinetic_grad(&basis, Some(&w), Some(&d));
        let gv = oneshot::nuclear_attraction_grad(&basis, &pairs, &charges, &d);
        let t_1e = t1.elapsed().as_secs_f64();
        let t2 = Instant::now();
        let g2 = oneshot::grad_4c(
            &basis,
            &pairs,
            &DensityTerms::default()
                .coulomb(0.5, &d, &d)
                .exchange(-0.25 * 1.0, &d, &d),
            thresh,
            &linus_ints::operator::Operator::Coulomb,
        );
        let t_2e = t2.elapsed().as_secs_f64();
        for i in 0..z.len() {
            for k in 0..3 {
                g[i][k] += gv[i][k] + g2[i][k];
            }
        }
        // nuclear repulsion
        for i in 0..z.len() {
            for j in 0..i {
                let r = [
                    positions[i][0] - positions[j][0],
                    positions[i][1] - positions[j][1],
                    positions[i][2] - positions[j][2],
                ];
                let dd = (r[0] * r[0] + r[1] * r[1] + r[2] * r[2]).sqrt();
                let ff = -((z[i] * z[j]) as f64) / (dd * dd * dd);
                for k in 0..3 {
                    g[i][k] += ff * r[k];
                    g[j][k] -= ff * r[k];
                }
            }
        }
        let tot = t0.elapsed().as_secs_f64();
        if tot < best.0 {
            best = (tot, t_setup, t_1e, t_2e);
        }
        grad = g;
    }
    let flat: Vec<f64> = grad.iter().flat_map(|x| x.iter().cloned()).collect();
    let mut extra = serde_json::Map::new();
    if inp["hessian"].as_bool().unwrap_or(false) {
        // second-derivative costs: 4c Hessian contraction and J^x/K^x for all perturbations
        let pairs = PairSet::build(&basis, PairOptions::with_order(DerivOrder::Two));
        let t0 = Instant::now();
        let h = oneshot::hess_4c(
            &basis,
            &pairs,
            &DensityTerms::default()
                .coulomb(0.5, &d, &d)
                .exchange(-0.25 * 1.0, &d, &d),
            thresh,
            &linus_ints::operator::Operator::Coulomb,
        );
        extra.insert("t_hess_4c".into(), t0.elapsed().as_secs_f64().into());
        std::hint::black_box(h);
        let t0 = Instant::now();
        let (jx, kx) = linus_ints::eri4c::jk_deriv::jk_deriv(
            &basis,
            &pairs,
            &[&d],
            true,
            true,
            thresh,
            &linus_ints::operator::Operator::Coulomb,
        );
        extra.insert("t_jk_deriv".into(), t0.elapsed().as_secs_f64().into());
        std::hint::black_box((jx, kx));
    }
    let mut out = serde_json::json!({"nbf": n, "threads": rayon::current_num_threads(), "t_total": best.0, "t_setup": best.1, "t_1e": best.2, "t_2e": best.3, "grad": flat});
    for (k, v) in extra {
        out[k] = v;
    }
    println!("{out}");
}
