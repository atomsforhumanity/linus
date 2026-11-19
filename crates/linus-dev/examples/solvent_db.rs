//! solvent_db [name...]: print the solvent database as JSON (all solvents, or the named ones
//! resolved through `db::lookup`, with null for names that do not resolve).
use linus_solvent::db::{self, EpsSource, Solvent};
use serde_json::{json, Value};

fn dump(s: &Solvent) -> Value {
    let src = match s.eps_source {
        EpsSource::CrcFit { extrapolated_k } => {
            json!({"crc_fit": {"extrapolated_k": extrapolated_k}})
        }
        EpsSource::CrcTabulated { temperature_k } => {
            json!({"crc_tabulated": {"temperature_k": temperature_k}})
        }
        EpsSource::Minnesota => json!("minnesota"),
    };
    json!({
        "key": s.key,
        "name": s.name,
        "formula": s.formula,
        "eps": s.eps,
        "eps_source": src,
        "smd": s.smd.map(|d| json!({"n": d.n, "alpha": d.alpha, "beta": d.beta, "gamma": d.gamma,
                                     "phi": d.phi, "psi": d.psi, "water": d.water})),
        "n25": s.n25,
        "minnesota_name": s.minnesota_name,
        "eps_minnesota": s.eps_minnesota,
        "eps_pyscf": s.eps_pyscf,
        "eps_orca_smd": s.eps_orca_smd,
        "eps_orca_cpcm": s.eps_orca_cpcm,
        "refrac_orca_cpcm": s.refrac_orca_cpcm,
        "crc_tabulated": s.crc.as_ref().map(|c| json!([c.dielectric, c.dielectric_temperature_k])),
        "aliases": db::aliases(&s.key).iter().map(|(a, src)| json!([a, src])).collect::<Vec<_>>(),
    })
}

fn main() {
    let names: Vec<String> = std::env::args().skip(1).collect();
    let out: Value = if names.is_empty() {
        db::solvents().iter().map(dump).collect()
    } else {
        names
            .iter()
            .map(|n| json!({"query": n, "solvent": db::lookup(n).map(dump)}))
            .collect()
    };
    println!("{}", serde_json::to_string(&out).unwrap());
}
