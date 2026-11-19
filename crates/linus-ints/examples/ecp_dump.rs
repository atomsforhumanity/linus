//! ECP integrals for checks against PySCF: ecp_dump <input.json> [output.bin]
//! input: {"z": [...], "coords": [[x,y,z],...] (bohr), "basis": "def2-svp", "pure": true,
//!         "dens": [nbf*nbf row-major, optional], "what": ["matrix", "grad", "deriv", "hess"],
//!         "repeat": 1}
//! Prints {"nbf", "times": {...}, "stats", "grad", "hess"} as JSON; with an output path the
//! ECP matrix (and the derivative matrices, if asked) are written there as little-endian f64:
//! matrix (nbf^2) then deriv (3 natom nbf^2).
use linus_ints::one_e::ecp::EcpSet;
use linus_ints::reference::oneshot;
use linus_ints::{basis::BasisDef, BasisSet, PairOptions, PairSet};
use std::io::Write;
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
    let pure = inp["pure"].as_bool().unwrap_or(true);
    let repeat = inp["repeat"].as_u64().unwrap_or(1) as usize;
    let what: Vec<String> = inp["what"]
        .as_array()
        .map(|a| a.iter().map(|x| x.as_str().unwrap().to_string()).collect())
        .unwrap_or_else(|| vec!["matrix".into()]);
    let def = BasisDef::load(inp["basis"].as_str().unwrap(), None).unwrap();
    let basis = BasisSet::new(&def, &z, &positions, pure).unwrap();
    let ecps = EcpSet::new(&def, &z, &positions).unwrap();
    let n = basis.nbf;
    let pairs = PairSet::build(&basis, PairOptions::default());
    let mut times = serde_json::Map::new();
    let mut out_bin: Vec<f64> = Vec::new();
    let mut res = serde_json::Map::new();
    if what.iter().any(|w| w == "matrix") {
        let mut best = f64::MAX;
        let mut v = Vec::new();
        let mut stats = Default::default();
        for _ in 0..repeat {
            let t0 = Instant::now();
            let (vv, st) = oneshot::ecp_matrix_stats(&basis, &pairs, &ecps);
            best = best.min(t0.elapsed().as_secs_f64());
            v = vv;
            stats = st;
        }
        times.insert("matrix".into(), best.into());
        res.insert("stats".into(), format!("{:?}", stats).into());
        out_bin.extend_from_slice(&v);
    }
    // "dens": an nbf x nbf array, or "synthetic": D_ij = (cos(i + 2j) + cos(j + 2i)) / nbf
    let dens = inp.get("dens").map(|v| {
        if v.as_str() == Some("synthetic") {
            let mut d = vec![0.0; n * n];
            for i in 0..n {
                for j in 0..n {
                    d[i * n + j] =
                        (((i + 2 * j) as f64).cos() + ((j + 2 * i) as f64).cos()) / n as f64;
                }
            }
            d
        } else {
            f(v)
        }
    });
    if let Some(d) = &dens {
        assert_eq!(d.len(), n * n);
        if what.iter().any(|w| w == "grad") {
            let mut best = f64::MAX;
            let mut g = Vec::new();
            for _ in 0..repeat {
                let t0 = Instant::now();
                g = oneshot::ecp_grad(&basis, &pairs, &ecps, d);
                best = best.min(t0.elapsed().as_secs_f64());
            }
            times.insert("grad".into(), best.into());
            res.insert("grad".into(), serde_json::to_value(&g).unwrap());
        }
        if what.iter().any(|w| w == "hess") {
            let mut best = f64::MAX;
            let mut h = Vec::new();
            for _ in 0..repeat {
                let t0 = Instant::now();
                h = oneshot::ecp_hess(&basis, &pairs, &ecps, d);
                best = best.min(t0.elapsed().as_secs_f64());
            }
            times.insert("hess".into(), best.into());
            res.insert("hess".into(), serde_json::to_value(&h).unwrap());
        }
    }
    if what.iter().any(|w| w == "deriv") {
        let t0 = Instant::now();
        let dv = oneshot::ecp_deriv(&basis, &pairs, &ecps);
        times.insert("deriv".into(), t0.elapsed().as_secs_f64().into());
        out_bin.extend_from_slice(&dv);
    }
    res.insert("nbf".into(), n.into());
    res.insert("times".into(), serde_json::Value::Object(times));
    if let Some(p) = args.get(2) {
        let mut fh = std::fs::File::create(p).unwrap();
        let bytes: Vec<u8> = out_bin.iter().flat_map(|x| x.to_le_bytes()).collect();
        fh.write_all(&bytes).unwrap();
    }
    println!("{}", serde_json::Value::Object(res));
}
