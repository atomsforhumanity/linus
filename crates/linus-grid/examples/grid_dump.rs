//! Dump a molecular grid to a raw binary file for external checks.
//!
//! usage: grid_dump <molecule.xyz (Angstrom)> <preset> <out.bin>
//!   preset: pyscf:<0-9> | psi4 | orca:<1-3> | orca-guess:<1-3> | linus:<1-4> | json:<spec.json>
//! Output: little-endian f64 array of shape (npoints, 6): x, y, z, w, atom, w_atomic (bohr).
//! Coordinates are converted with PySCF's Bohr radius (linus-core's `BOHR_ANGSTROM_PYSCF`).

use linus_core::elements::symbol_to_z;
use linus_core::units::BOHR_ANGSTROM_PYSCF;
use linus_grid::*;
use std::io::Write;

pub fn read_xyz(path: &str) -> (Vec<u32>, Vec<[f64; 3]>) {
    let s = std::fs::read_to_string(path).expect("read xyz");
    let mut lines: Vec<&str> = s.lines().collect();
    // accept both bare "El x y z" lines and the standard xyz header
    if lines
        .first()
        .is_some_and(|l| l.trim().parse::<usize>().is_ok())
    {
        lines = lines[2..].to_vec();
    }
    let mut z = Vec::new();
    let mut xyz = Vec::new();
    for l in lines {
        let t: Vec<&str> = l.split_whitespace().collect();
        if t.len() < 4 {
            continue;
        }
        z.push(symbol_to_z(t[0]).expect("unknown element"));
        let f = |i: usize| t[i].parse::<f64>().unwrap() / BOHR_ANGSTROM_PYSCF;
        xyz.push([f(1), f(2), f(3)]);
    }
    (z, xyz)
}

pub fn preset(p: &str) -> GridSpec {
    let (name, arg) = p.split_once(':').unwrap_or((p, ""));
    match name {
        "pyscf" => GridSpec::pyscf(arg.parse().unwrap_or(3)),
        "psi4" => GridSpec::psi4_default(),
        "orca" => GridSpec::orca_defgrid(arg.parse().unwrap_or(2)),
        "orca-guess" => GridSpec::orca_defgrid_guess(arg.parse().unwrap_or(2)),
        "linus" => GridSpec::linus(arg.parse().unwrap_or(2)),
        "json" => serde_json::from_str(&std::fs::read_to_string(arg).unwrap()).unwrap(),
        _ => panic!("unknown preset {p}"),
    }
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    if a.len() < 4 {
        eprintln!("usage: grid_dump <xyz> <preset> <out.bin>");
        std::process::exit(1);
    }
    let (z, xyz) = read_xyz(&a[1]);
    let spec = preset(&a[2]);
    let t = std::time::Instant::now();
    let g = MolecularGrid::build(&z, &xyz, &spec).unwrap();
    let dt = t.elapsed().as_secs_f64();
    let mut buf = Vec::with_capacity(g.len() * 48);
    for i in 0..g.len() {
        for v in [
            g.x[i],
            g.y[i],
            g.z[i],
            g.w[i],
            g.atom[i] as f64,
            g.w_atomic[i],
        ] {
            buf.extend_from_slice(&v.to_le_bytes());
        }
    }
    std::fs::File::create(&a[3])
        .unwrap()
        .write_all(&buf)
        .unwrap();
    eprintln!(
        "{} points, {} batches, built in {:.3} s",
        g.len(),
        g.batches.len(),
        dt
    );
}
