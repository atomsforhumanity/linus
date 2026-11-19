//! grid_study: one calculation from a JSON request on stdin, for grid and RI accuracy studies.
//! Request: `xyz` (path, Å), `charge`, `mult`, `calc` (energy | gradient | hessian), `dipole`
//! (Hessian dipole derivatives) and `opts` (the driver's `CalcOptions`, any grid via
//! `{"kind": "spec", ...}`). Prints JSON: grid points (after the weight cutoff and before),
//! energy and its XC part, gradient, Hessian, dipole derivatives, CPHF iterations, timers and
//! wall time (s).
use linus::HessianOptions;
use linus::{CalcOptions, Session};
use serde_json::{json, Value};
use std::io::Read;

fn main() {
    let mut s = String::new();
    std::io::stdin().read_to_string(&mut s).unwrap();
    let req: Value = serde_json::from_str(&s).unwrap();
    // {"spec_of": GridChoice}: print that grid's full specification
    if let Some(g) = req.get("spec_of") {
        let g: linus::GridChoice = serde_json::from_value(g.clone()).unwrap();
        println!("{}", serde_json::to_string(&g.spec()).unwrap());
        return;
    }
    let charge = req["charge"].as_i64().unwrap_or(0) as i32;
    let mult = req["mult"].as_u64().unwrap_or(1) as u32;
    let mol = linus::Structure::from_xyz_str(
        &std::fs::read_to_string(req["xyz"].as_str().unwrap()).unwrap(),
        charge,
        mult,
    )
    .unwrap();
    let opts: CalcOptions = serde_json::from_value(req["opts"].clone()).unwrap();
    let calc_type = req["calc"].as_str().unwrap_or("hessian").to_string();
    let spec = opts.grid.xc().spec();
    let dft = !opts.method().eq_ignore_ascii_case("hf");
    let (np, np0) = if dft {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(opts.threads().max(1))
            .build()
            .unwrap();
        let g = pool
            .install(|| linus_grid::MolecularGrid::build(&mol.z, &mol.positions, &spec).unwrap());
        (g.len(), g.npoints_before_cutoff)
    } else {
        (0, 0)
    };
    let mut out = json!({"npoints": np, "npoints_before_cutoff": np0});
    if calc_type == "count" {
        println!("{out}");
        return;
    }
    let mut calc = Session::new(mol.clone(), opts).unwrap();
    let t = std::time::Instant::now();
    match calc_type.as_str() {
        "energy" => {
            calc.energy().unwrap();
        }
        "gradient" => {
            let g = calc.gradient().unwrap();
            out["gradient"] = json!(g.gradient);
            out["timers"] = g.timers.to_json();
        }
        _ => {
            let want_dip = req["dipole"].as_bool().unwrap_or(false);
            let h = calc
                .hessian_with(&HessianOptions {
                    dipole_derivatives: want_dip,
                })
                .unwrap();
            out["gradient"] = json!(h.gradient);
            out["hessian"] = json!(h.hessian);
            out["cphf_iterations"] = json!(h.cphf_iterations);
            out["dipole_derivatives"] = json!(h.dipole_derivatives);
            out["timers"] = h.timers.to_json();
        }
    }
    let r = calc.last().unwrap();
    out["wall"] = json!(t.elapsed().as_secs_f64());
    out["energy"] = json!(r.energy);
    out["e_xc"] = json!(r.e_xc);
    out["e_exchange"] = json!(r.e_exchange);
    out["scf_iterations"] = json!(r.iterations);
    out["nbf"] = json!(calc.basis().nbf);
    out["scf_timers"] = calc.timers().to_json();
    println!("{out}");
}
