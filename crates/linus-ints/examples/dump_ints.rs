//! Dump integrals for validation: dump_ints <input.json> <outdir>
//! input: {"z": [...], "coords": [[x,y,z],...] (bohr), "basis": "def2-svp", "aux": null|"def2-universal-jkfit",
//!         "pure": true, "eri": true, "omega": null|0.4}
//! With "omega", also ERI_LR.bin (erf(omega r)/r) and ERI_SR.bin (erfc(omega r)/r).
use linus_ints::reference::oneshot;
use linus_ints::{basis::BasisDef, eri4c, BasisSet, PairOptions, PairSet};
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
    let pure = inp["pure"].as_bool().unwrap_or(true);
    let def = BasisDef::load(inp["basis"].as_str().unwrap(), None).unwrap();
    let basis = BasisSet::new(&def, &z, &positions, pure).unwrap();
    let t0 = std::time::Instant::now();
    let pairs = PairSet::build(&basis, PairOptions::default());
    let (s, t) = oneshot::overlap_kinetic(&basis);
    let charges: Vec<(f64, [f64; 3])> = z
        .iter()
        .zip(&positions)
        .map(|(&zi, &c)| (zi as f64, c))
        .collect();
    let v = oneshot::potential(&basis, &pairs, &charges);
    let dip = oneshot::dipole(&basis, [0.0; 3]);
    let t1 = t0.elapsed().as_secs_f64();
    write_bin(&out.join("S.bin"), &s);
    write_bin(&out.join("T.bin"), &t);
    write_bin(&out.join("V.bin"), &v);
    write_bin(&out.join("DX.bin"), &dip[0]);
    write_bin(&out.join("DY.bin"), &dip[1]);
    write_bin(&out.join("DZ.bin"), &dip[2]);
    let mut t_eri = 0.0;
    if inp["eri"].as_bool().unwrap_or(true) {
        let t0 = std::time::Instant::now();
        let g = eri4c::eri_full(&basis, &pairs, &linus_ints::operator::Operator::Coulomb);
        t_eri = t0.elapsed().as_secs_f64();
        write_bin(&out.join("ERI.bin"), &g);
        if let Some(w) = inp["omega"].as_f64() {
            use linus_ints::operator::Operator;
            write_bin(
                &out.join("ERI_LR.bin"),
                &eri4c::eri_full(&basis, &pairs, &Operator::LongRange { omega: w }),
            );
            write_bin(
                &out.join("ERI_SR.bin"),
                &eri4c::eri_full(
                    &basis,
                    &pairs,
                    &Operator::Attenuated {
                        coulomb: 1.0,
                        long_range: -1.0,
                        omega: w,
                    },
                ),
            );
        }
    }
    let mut naux = 0;
    if let Some(auxname) = inp["aux"].as_str() {
        let adef = BasisDef::load(auxname, None).unwrap();
        let aux = BasisSet::new(&adef, &z, &positions, true).unwrap();
        let ap = PairSet::aux(&aux, PairOptions::default());
        write_bin(
            &out.join("J2C.bin"),
            &linus_ints::reference::eri_2c(&aux, &ap),
        );
        write_bin(
            &out.join("J3C.bin"),
            &linus_ints::reference::eri_3c(&basis, &pairs, &aux, &ap),
        );
        naux = aux.nbf;
    }
    println!(
        "{}",
        serde_json::json!({"nbf": basis.nbf, "naux": naux, "nshell": basis.shells.len(), "t_one": t1, "t_eri": t_eri})
    );
}
