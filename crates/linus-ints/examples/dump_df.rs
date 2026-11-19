//! Dump eri3c density-fitting integrals for validation: dump_df <input.json> <outdir>
//! input: {"z": [...], "coords": [[x,y,z],...] (bohr), "basis": "def2-svp", "aux": "def2-universal-jkfit", "omega": 0.3}
//! writes J3C.bin (naux x nbf^2, (k|mn)), J2C.bin (naux x naux) and, for omega > 0,
//! J3C_LR.bin / J2C_LR.bin with erf(omega r)/r; prints {"nbf", "naux"}.
use linus_ints::eri3c::{self, Aux3c, Pair3c, Work3c};
use linus_ints::operator::Operator;
use linus_ints::{basis::BasisDef, BasisSet, PairOptions, PairSet};
use std::io::Write;

fn write_bin(path: &std::path::Path, v: &[f64]) {
    let mut f = std::fs::File::create(path).unwrap();
    let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
    f.write_all(&bytes).unwrap();
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let inp: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&args[1]).unwrap()).unwrap();
    let out = std::path::Path::new(&args[2]);
    std::fs::create_dir_all(out).unwrap();
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
            let r = r.as_array().unwrap();
            [
                r[0].as_f64().unwrap(),
                r[1].as_f64().unwrap(),
                r[2].as_f64().unwrap(),
            ]
        })
        .collect();
    let b = BasisSet::new(
        &BasisDef::load(inp["basis"].as_str().unwrap(), None).unwrap(),
        &z,
        &positions,
        true,
    )
    .unwrap();
    let a = BasisSet::new(
        &BasisDef::load(inp["aux"].as_str().unwrap(), None).unwrap(),
        &z,
        &positions,
        true,
    )
    .unwrap();
    let omega = inp["omega"].as_f64().unwrap_or(0.0);
    let pl = PairSet::build(&b, PairOptions::default());
    let aux = Aux3c::new(&a);
    let chunks = aux.chunks(0..a.shells.len());
    let (nbf, naux) = (b.nbf, a.nbf);
    let mut w = Work3c::new();
    let mut dump = |om: f64, name3: &str, name2: &str| {
        let mut j3 = vec![0.0; naux * nbf * nbf];
        let op = if om == 0.0 {
            Operator::Coulomb
        } else {
            Operator::LongRange { omega: om }
        };
        for sp in pl.pairs() {
            let p3 = Pair3c::from_shell_pair(sp);
            let mut o = vec![0.0; p3.nab() * naux];
            eri3c::pair_block(&p3, &chunks, &mut o, naux, 0.0, &op, &mut w);
            let (a0, b0) = (b.shells[sp.sa].ao_start, b.shells[sp.sb].ao_start);
            for ia in 0..sp.na {
                for ib in 0..sp.nb {
                    for k in 0..naux {
                        let v = o[(ia * sp.nb + ib) * naux + k];
                        j3[k * nbf * nbf + (a0 + ia) * nbf + b0 + ib] = v;
                        j3[k * nbf * nbf + (b0 + ib) * nbf + a0 + ia] = v;
                    }
                }
            }
        }
        write_bin(&out.join(name3), &j3);
        write_bin(&out.join(name2), &eri3c::metric(&aux, 0.0, &op));
    };
    dump(0.0, "J3C.bin", "J2C.bin");
    if omega > 0.0 {
        dump(omega, "J3C_LR.bin", "J2C_LR.bin");
    }
    println!("{}", serde_json::json!({"nbf": nbf, "naux": naux}));
}
