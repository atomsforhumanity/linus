//! smd_cds <input.json> [threads]: SMD G_CDS (energy, gradient, Hessian) for validation scripts.
//!
//! Input JSON: {"z": [..], "coords_bohr": [[x,y,z],..], "order": 0|1|2,
//!              "convention": "mnsol"|"decimal",
//!              "solvent": "<name>"  or  "descriptors": {"n":..,"alpha":..,"beta":..,"gamma":..,
//!              "phi":..,"psi":..,"water":bool}}
//! Output JSON on stdout: energy, per_atom, sasa, atomic_surface_tension,
//! molecular_surface_tension, gradient, hessian, seconds.
use linus_solvent::db::{self, SmdDescriptors};
use linus_solvent::smd::{cds, CdsConfig, CdsOrder, ParamConvention};
// input may also set "contact_smoothing" (Å)
use serde_json::{json, Value};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let input: Value =
        serde_json::from_str(&std::fs::read_to_string(&args[1]).expect("read input"))
            .expect("json");
    let threads: usize = args.get(2).map(|s| s.parse().unwrap()).unwrap_or(1);
    let z: Vec<u32> = input["z"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as u32)
        .collect();
    let positions: Vec<[f64; 3]> = input["coords_bohr"]
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
    let order = match input["order"].as_u64().unwrap_or(0) {
        0 => CdsOrder::Energy,
        1 => CdsOrder::Gradient,
        _ => CdsOrder::Hessian,
    };
    let convention = match input["convention"].as_str().unwrap_or("mnsol") {
        "decimal" => ParamConvention::Decimal,
        _ => ParamConvention::Mnsol,
    };
    let sd: SmdDescriptors = if let Some(name) = input["solvent"].as_str() {
        *db::smd_descriptors(name).unwrap_or_else(|| panic!("no SMD solvent {name}"))
    } else {
        let d = &input["descriptors"];
        let f = |k: &str| d[k].as_f64().unwrap();
        SmdDescriptors {
            n: f("n"),
            alpha: f("alpha"),
            beta: f("beta"),
            gamma: f("gamma"),
            phi: f("phi"),
            psi: f("psi"),
            water: d["water"].as_bool().unwrap_or(false),
        }
    };
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .unwrap();
    let t0 = std::time::Instant::now();
    let contact_smoothing = input["contact_smoothing"].as_f64().unwrap_or(0.0);
    let r = pool.install(|| {
        cds(
            &z,
            &positions,
            &sd,
            &CdsConfig {
                order,
                convention,
                contact_smoothing,
            },
        )
    });
    let secs = t0.elapsed().as_secs_f64();
    let out = json!({
        "energy": r.energy,
        "per_atom": r.per_atom,
        "sasa": r.sasa,
        "atomic_surface_tension": r.atomic_surface_tension,
        "molecular_surface_tension": r.molecular_surface_tension,
        "gradient": r.gradient,
        "hessian": r.hessian,
        "seconds": secs,
    });
    println!("{}", serde_json::to_string(&out).unwrap());
}
