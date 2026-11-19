//! Grid generation timing for water clusters.
//!
//! usage: grid_bench [n_waters,...] [threads,...] [preset,...] [reps]
//!   e.g. grid_bench 33,333 1,16 linus:2,orca:2,pyscf:3 3
//! Presets as in grid_dump (pyscf:<l>, psi4, orca:<n>, linus:<l>).

use linus_grid::*;

const ANG: f64 = 1.0 / linus_core::units::BOHR_ANGSTROM_PYSCF;

/// Waters on a jittered cubic lattice (3.1 A spacing, ~liquid density), random orientations.
#[allow(clippy::approx_constant)] // arbitrary random-angle scales, not pi
pub fn water_cluster(n: usize) -> (Vec<u32>, Vec<[f64; 3]>) {
    let mut z = Vec::new();
    let mut xyz = Vec::new();
    let m = (n as f64).cbrt().ceil() as usize;
    let mut s: u64 = 12345;
    let mut rnd = || {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((s >> 11) as f64) / ((1u64 << 53) as f64)
    };
    let mut k = 0;
    'outer: for i in 0..m {
        for j in 0..m {
            for l in 0..m {
                if k == n {
                    break 'outer;
                }
                let o = [
                    (i as f64 * 3.1 + 0.3 * rnd()) * ANG,
                    (j as f64 * 3.1 + 0.3 * rnd()) * ANG,
                    (l as f64 * 3.1 + 0.3 * rnd()) * ANG,
                ];
                let th = rnd() * 6.283;
                let ph = rnd() * 3.14;
                let u = [ph.sin() * th.cos(), ph.sin() * th.sin(), ph.cos()];
                let v0 = [-u[1], u[0], 0.0];
                let nv = (v0[0] * v0[0] + v0[1] * v0[1]).sqrt().max(1e-6);
                let v = [v0[0] / nv, v0[1] / nv, 0.0];
                let (c, sn) = ((52.25f64).to_radians().cos(), (52.25f64).to_radians().sin());
                let r = 0.9572 * ANG;
                z.extend([8, 1, 1]);
                xyz.push(o);
                for sg in [1.0, -1.0] {
                    xyz.push([
                        o[0] + r * (c * u[0] + sg * sn * v[0]),
                        o[1] + r * (c * u[1] + sg * sn * v[1]),
                        o[2] + r * (c * u[2] + sg * sn * v[2]),
                    ]);
                }
                k += 1;
            }
        }
    }
    (z, xyz)
}

fn preset(p: &str) -> GridSpec {
    let (name, arg) = p.split_once(':').unwrap_or((p, ""));
    match name {
        "pyscf" => GridSpec::pyscf(arg.parse().unwrap_or(3)),
        "psi4" => GridSpec::psi4_default(),
        "orca" => GridSpec::orca_defgrid(arg.parse().unwrap_or(2)),
        "orca-guess" => GridSpec::orca_defgrid_guess(arg.parse().unwrap_or(2)),
        "linus" => GridSpec::linus(arg.parse().unwrap_or(2)),
        "lko" => {
            let mut s = GridSpec::linus(arg.parse().unwrap_or(2));
            s.partition = PartitionScheme::LKO;
            s
        }
        _ => panic!("unknown preset {p}"),
    }
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let sizes: Vec<usize> = a
        .get(1)
        .map_or("33,333".into(), |s| s.clone())
        .split(',')
        .map(|x| x.parse().unwrap())
        .collect();
    let threads: Vec<usize> = a
        .get(2)
        .map_or("1,16".into(), |s| s.clone())
        .split(',')
        .map(|x| x.parse().unwrap())
        .collect();
    let presets: Vec<String> = a
        .get(3)
        .map_or("linus:2,orca:2".into(), |s| s.clone())
        .split(',')
        .map(String::from)
        .collect();
    let reps: usize = a.get(4).map_or(3, |s| s.parse().unwrap());
    println!(
        "{:>7} {:>6} {:>10} {:>8} {:>11} {:>9} {:>10}",
        "atoms", "thr", "preset", "batches", "points", "time/s", "Mpts/s"
    );
    for &n in &sizes {
        let (z, xyz) = water_cluster(n);
        if let Ok(dir) = std::env::var("LINUS_GRID_WRITE_XYZ") {
            // write the cluster (Angstrom) for comparisons with other programs
            let mut s = format!("{}\nwater cluster {n}\n", z.len());
            for (zi, p) in z.iter().zip(&xyz) {
                let el = if *zi == 8 { "O" } else { "H" };
                s += &format!(
                    "{el} {:.10} {:.10} {:.10}\n",
                    p[0] / ANG,
                    p[1] / ANG,
                    p[2] / ANG
                );
            }
            std::fs::write(format!("{dir}/water{n}.xyz"), s).unwrap();
        }
        for p in &presets {
            let spec = preset(p);
            for &t in &threads {
                let pool = rayon::ThreadPoolBuilder::new()
                    .num_threads(t)
                    .build()
                    .unwrap();
                let mut best = f64::INFINITY;
                let mut np = 0;
                let mut nb = 0;
                for _ in 0..reps {
                    let t0 = std::time::Instant::now();
                    let g = pool.install(|| MolecularGrid::build(&z, &xyz, &spec).unwrap());
                    best = best.min(t0.elapsed().as_secs_f64());
                    np = g.len();
                    nb = g.batches.len();
                }
                println!(
                    "{:7} {:6} {:>10} {:8} {:11} {:9.3} {:10.2}",
                    z.len(),
                    t,
                    p,
                    nb,
                    np,
                    best,
                    np as f64 / best / 1e6
                );
            }
        }
    }
}
