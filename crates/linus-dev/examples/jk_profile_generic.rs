//! Per-class time profile of the generic (reference) J/K path: the same quartet loop,
//! screening and digestion as the SAD guess's builder (`scf::guess`), with each quartet's integral evaluation and
//! digestion timed (rdtsc) and binned by angular-momentum class. On other architectures than
//! x86-64 the counter is the monotonic clock in nanoseconds (`tsc_ghz` is then about 1).
//!
//! jk_profile_generic <xyz> <basis> [--dens file.bin] [--threads n] [--thresh t]
use linus::Structure;
use linus_ints::eri4c::{self, EriWork};
use linus_ints::{basis::BasisDef, BasisSet, PairOptions, PairSet};
use rayon::prelude::*;
use std::collections::BTreeMap;

fn read_bin(path: &str) -> Vec<f64> {
    let b = std::fs::read(path).unwrap();
    b.as_chunks::<8>()
        .0
        .iter()
        .map(|c| f64::from_le_bytes(*c))
        .collect()
}

#[inline]
fn tsc() -> u64 {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        core::arch::x86_64::_rdtsc()
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        static T0: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
        T0.get_or_init(std::time::Instant::now).elapsed().as_nanos() as u64
    }
}

const LS: [char; 6] = ['s', 'p', 'd', 'f', 'g', 'h'];

fn class_name(k: (usize, usize, usize, usize)) -> String {
    format!("({}{}|{}{})", LS[k.0], LS[k.1], LS[k.2], LS[k.3])
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let get = |k: &str| a.iter().position(|x| x == k).map(|i| a[i + 1].clone());
    let threads: usize = get("--threads").map(|s| s.parse().unwrap()).unwrap_or(1);
    let thresh: f64 = get("--thresh").map(|s| s.parse().unwrap()).unwrap_or(1e-12);
    let mol = Structure::from_xyz_str(&std::fs::read_to_string(&a[1]).unwrap(), 0, 1).unwrap();
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .unwrap();
    pool.install(|| {
        let def = BasisDef::load(&a[2], None).unwrap();
        let basis = BasisSet::new(&def, &mol.z, &mol.positions, true).unwrap();
        let nbf = basis.nbf;
        let d = read_bin(&get("--dens").expect("--dens required"));
        let pairs = PairSet::build(&basis, PairOptions::default());
        let q = pairs.schwarz();
        let sh = &basis.shells;
        let ns = sh.len();
        let mut dmax = vec![0.0f64; ns * ns];
        for x in 0..ns {
            for y in 0..ns {
                let mut m: f64 = 0.0;
                for i in sh[x].ao_start..sh[x].ao_start + sh[x].nfunc() {
                    for j in sh[y].ao_start..sh[y].ao_start + sh[y].nfunc() {
                        m = m.max(d[i * nbf + j].abs());
                    }
                }
                dmax[x * ns + y] = m;
            }
        }
        let np = pairs.len();
        let t_wall = std::time::Instant::now();
        // class -> (quartets, primitive quartets, cycles integrals, cycles digestion)
        type Prof = BTreeMap<(usize, usize, usize, usize), (u64, u64, u64, u64)>;
        let prof: Prof = (0..np)
            .into_par_iter()
            .fold(
                || (EriWork::new(), vec![0.0; nbf * nbf], vec![0.0; nbf * nbf], Vec::<f64>::new(), Prof::new()),
                |(mut w, mut jt, mut kt, mut buf, mut pr), ij| {
                    let bra = pairs.pair(ij);
                    if bra.nprim == 0 {
                        return (w, jt, kt, buf, pr);
                    }
                    let (s1, s2) = (bra.sa, bra.sb);
                    for kl in 0..=ij {
                        let ket = pairs.pair(kl);
                        let qq = q[ij] * q[kl];
                        if ket.nprim == 0 || qq < thresh {
                            continue;
                        }
                        let (s3, s4) = (ket.sa, ket.sb);
                        let dm = dmax[s1 * ns + s2]
                            .max(dmax[s3 * ns + s4])
                            .max(dmax[s1 * ns + s3])
                            .max(dmax[s1 * ns + s4])
                            .max(dmax[s2 * ns + s3])
                            .max(dmax[s2 * ns + s4]);
                        if qq * dm < thresh {
                            continue;
                        }
                        let (nab, ncd) = (bra.nab(), ket.nab());
                        buf.resize(nab * ncd, 0.0);
                        let c0 = tsc();
                        eri4c::eri_quartet(bra, ket, &linus_ints::operator::Operator::Coulomb, &mut buf, &mut w);
                        let c1 = tsc();
                        let deg = (if s1 == s2 { 1.0 } else { 2.0 }) * (if s3 == s4 { 1.0 } else { 2.0 }) * (if ij == kl { 1.0 } else { 2.0 });
                        let (p0, q0, r0, t0) = (sh[s1].ao_start, sh[s2].ao_start, sh[s3].ao_start, sh[s4].ao_start);
                        let (na, nb, nc, nd) = (bra.na, bra.nb, ket.na, ket.nb);
                        for ia in 0..na {
                            let p = p0 + ia;
                            for ib in 0..nb {
                                let qi = q0 + ib;
                                let row = &buf[(ia * nb + ib) * ncd..(ia * nb + ib + 1) * ncd];
                                let dpq = d[p * nbf + qi];
                                let mut jpq = 0.0;
                                for ic in 0..nc {
                                    let r = r0 + ic;
                                    for id in 0..nd {
                                        let s = t0 + id;
                                        let v = row[ic * nd + id] * deg;
                                        jpq += d[r * nbf + s] * v;
                                        jt[r * nbf + s] += dpq * v;
                                        kt[p * nbf + r] += d[qi * nbf + s] * v;
                                        kt[qi * nbf + s] += d[p * nbf + r] * v;
                                        kt[p * nbf + s] += d[qi * nbf + r] * v;
                                        kt[qi * nbf + r] += d[p * nbf + s] * v;
                                    }
                                }
                                jt[p * nbf + qi] += jpq;
                            }
                        }
                        let c2 = tsc();
                        // canonical class: higher pair first, each pair (high l, low l)
                        let bc = (bra.la.max(bra.lb), bra.la.min(bra.lb));
                        let kc = (ket.la.max(ket.lb), ket.la.min(ket.lb));
                        let (x, y) = if (bc.0 + bc.1, bc.0) >= (kc.0 + kc.1, kc.0) { (bc, kc) } else { (kc, bc) };
                        let e = pr.entry((x.0, x.1, y.0, y.1)).or_insert((0, 0, 0, 0));
                        e.0 += 1;
                        e.1 += (bra.nprim * ket.nprim) as u64;
                        e.2 += c1 - c0;
                        e.3 += c2 - c1;
                    }
                    (w, jt, kt, buf, pr)
                },
            )
            .map(|(_, _, _, _, pr)| pr)
            .reduce(Prof::new, |mut a, b| {
                for (k, v) in b {
                    let e = a.entry(k).or_insert((0, 0, 0, 0));
                    e.0 += v.0;
                    e.1 += v.1;
                    e.2 += v.2;
                    e.3 += v.3;
                }
                a
            });
        let wall = t_wall.elapsed().as_secs_f64();
        // calibrate TSC frequency
        let (c0, t0) = (tsc(), std::time::Instant::now());
        while t0.elapsed().as_secs_f64() < 0.2 {}
        let ghz = (tsc() - c0) as f64 / t0.elapsed().as_secs_f64() / 1e9;
        let tot: u64 = prof.values().map(|v| v.2 + v.3).sum();
        let mut rows: Vec<_> = prof.iter().collect();
        rows.sort_by_key(|(_, v)| std::cmp::Reverse(v.2 + v.3));
        let classes: Vec<serde_json::Value> = rows
            .iter()
            .map(|(k, v)| {
                serde_json::json!({"class": class_name(**k), "quartets": v.0, "prim_quartets": v.1,
                  "t_int": v.2 as f64 / ghz / 1e9, "t_digest": v.3 as f64 / ghz / 1e9,
                  "frac": (v.2 + v.3) as f64 / tot as f64,
                  "ns_per_quartet": (v.2 + v.3) as f64 / ghz / v.0 as f64})
            })
            .collect();
        println!(
            "{}",
            serde_json::json!({"nbf": nbf, "nshell": ns, "threads": threads, "thresh": thresh, "wall": wall,
              "cpu_total": tot as f64 / ghz / 1e9, "tsc_ghz": ghz, "classes": classes})
        );
    });
}
