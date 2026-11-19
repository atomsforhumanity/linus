//! Time the batched four-center kernels per class pair and kernel path.
//!
//! kernel_bench <xyz> <basis> [--reps n] [--paths default,tiled,loop] [--classes "(fd|fd),(ff|pp)"]
//!              [--isa avx512|avx2|generic] [--cart] [--min-l L] [--iblock-kb n]
//!
//! For every class pair (bra class >= ket class) of the molecule's basis, a sample of up to 256
//! (bra pair, ket block) combinations is evaluated per path (one warm-up round, then the best
//! of three rounds of at least `reps` passes and 20 ms), without primitive screening. Prints one JSON line per class pair and path: time per
//! call, per primitive-quartet vector (LANES ket lanes), and FMAs of the E^T R E contraction
//! (T step nnz_ket * nherm_bra per primitive quartet, I step nnz_bra * n_cd per bra primitive)
//! per ns.
use linus_ints::eri4c::batch::{BatchKernels, BatchedPairs, BlockWork, TILED_MAX_L};
use linus_ints::operator::Operator;
use linus_ints::{basis::BasisDef, BasisSet, PairOptions, PairSet};
use linus_num::simd::{Isa, Portable};

const LS: [char; 7] = ['s', 'p', 'd', 'f', 'g', 'h', 'i'];
/// Atomic numbers and positions (bohr) of an XYZ file (Å).
fn read_xyz(path: &str) -> (Vec<u32>, Vec<[f64; 3]>) {
    linus_core::structure::parse_xyz(&std::fs::read_to_string(path).unwrap()).unwrap()
}

#[derive(Clone, Copy, PartialEq, Debug)]
enum Path {
    Default,
    Tiled,
    Loop,
}

fn select(bp: &mut BatchedPairs, gen0: &[i32], path: Path) {
    let ncl = bp.classes.len();
    for ib in 0..ncl {
        for ik in 0..ncl {
            let k = ib * ncl + ik;
            let l = bp.classes[ib].lab + bp.classes[ik].lab;
            match path {
                Path::Default => {
                    bp.gen_kernel[k] = gen0[k];
                    bp.tiled[k] = l <= TILED_MAX_L;
                }
                Path::Tiled => {
                    bp.gen_kernel[k] = if l <= TILED_MAX_L { -1 } else { gen0[k] };
                    bp.tiled[k] = l <= TILED_MAX_L;
                }
                Path::Loop => {
                    bp.gen_kernel[k] = -1;
                    bp.tiled[k] = false;
                }
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn run<V: BatchKernels>(
    bp: &mut BatchedPairs,
    paths: &[Path],
    reps: usize,
    filter: &Option<Vec<String>>,
    min_l: usize,
    isa: &str,
) {
    let ncl = bp.classes.len();
    let lmax = bp.classes.iter().map(|c| c.la).max().unwrap();
    let mut w = BlockWork::<V>::new(lmax);
    let nf = bp.classes.iter().map(|c| c.na).max().unwrap();
    let mut out = vec![V::zero(); nf.pow(4)];
    let gen0 = bp.gen_kernel.clone();
    let mut order: Vec<usize> = (0..ncl).collect();
    order.sort_by_key(|&c| bp.classes[c].order);
    let op = Operator::Coulomb;
    for (oi, &ib) in order.iter().enumerate() {
        for &ik in &order[..=oi] {
            let (cb, ck) = (bp.classes[ib].clone(), bp.classes[ik].clone());
            let name = format!("({}{}|{}{})", LS[cb.la], LS[cb.lb], LS[ck.la], LS[ck.lb]);
            if cb.lab + ck.lab < min_l {
                continue;
            }
            if let Some(f) = filter {
                if !f.iter().any(|x| x == &name) {
                    continue;
                }
            }
            let bras: Vec<usize> = (0..bp.pairs.len())
                .filter(|&i| bp.pairs[i].class as usize == ib)
                .collect();
            let blks: Vec<usize> = (0..bp.blocks.len())
                .filter(|&i| bp.blocks[i].class as usize == ik)
                .collect();
            if bras.is_empty() || blks.is_empty() {
                continue;
            }
            let pick = |v: &[usize], n: usize| -> Vec<usize> {
                (0..n.min(v.len()))
                    .map(|k| v[k * v.len() / n.min(v.len())])
                    .collect()
            };
            let (bs, ks) = (pick(&bras, 16), pick(&blks, 16));
            let mut combos = Vec::new();
            for &b in &bs {
                for &k in &ks {
                    combos.push((b, k));
                }
            }
            let (nhb, ncd) = (cb.nh, ck.nab());
            let mut vpq = 0u64;
            let mut fma = 0f64;
            for &(b, k) in &combos {
                let (np, nk) = (bp.pairs[b].nprim as u64, bp.blocks[k].nprim as u64);
                vpq += np * nk;
                fma += (np * nk) as f64 * (ck.nz.len() * nhb) as f64
                    + np as f64 * (cb.nz.len() * ncd) as f64;
            }
            for &path in paths {
                select(bp, &gen0, path);
                let kidx = ib * ncl + ik;
                let kind = if bp.gen_kernel[kidx] >= 0 {
                    "generated"
                } else if bp.tiled[kidx] {
                    "tiled"
                } else {
                    "loop"
                };
                // warm-up round, then rounds of at least `reps` passes and 20 ms; best of 3
                let t0 = std::time::Instant::now();
                for &(b, k) in &combos {
                    unsafe { V::block_eri(bp, b, k, &op, 0.0, &mut out, &mut w) };
                }
                let once = t0.elapsed().as_secs_f64().max(1e-7);
                let n = reps.max((0.02 / once).ceil() as usize);
                let mut dt = f64::MAX;
                for _ in 0..3 {
                    let t0 = std::time::Instant::now();
                    for _ in 0..n {
                        for &(b, k) in &combos {
                            unsafe { V::block_eri(bp, b, k, &op, 0.0, &mut out, &mut w) };
                        }
                    }
                    dt = dt.min(t0.elapsed().as_secs_f64() / n as f64);
                }
                println!(
                    "{}",
                    serde_json::json!({"class": name, "L": cb.lab + ck.lab, "path": format!("{path:?}"), "kernel": kind, "isa": isa,
                        "calls": combos.len(), "vpq": vpq, "ns_per_call": dt * 1e9 / combos.len() as f64,
                        "ns_per_vpq": dt * 1e9 / vpq as f64, "fma_per_ns": fma / (dt * 1e9)})
                );
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512dq,avx2,fma")]
unsafe fn run_avx512(
    bp: &mut BatchedPairs,
    paths: &[Path],
    reps: usize,
    filter: &Option<Vec<String>>,
    min_l: usize,
) {
    run::<linus_num::simd::Avx512>(bp, paths, reps, filter, min_l, "avx512")
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn run_avx2(
    bp: &mut BatchedPairs,
    paths: &[Path],
    reps: usize,
    filter: &Option<Vec<String>>,
    min_l: usize,
) {
    run::<linus_num::simd::Avx2>(bp, paths, reps, filter, min_l, "avx2")
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let get = |k: &str| a.iter().position(|x| x == k).map(|i| a[i + 1].clone());
    let reps: usize = get("--reps").map(|s| s.parse().unwrap()).unwrap_or(20);
    let min_l: usize = get("--min-l").map(|s| s.parse().unwrap()).unwrap_or(0);
    if let Some(kb) = get("--iblock-kb") {
        linus_ints::eri4c::batch::TILED_IBLOCK_BYTES.store(
            kb.parse::<usize>().unwrap() * 1024,
            std::sync::atomic::Ordering::Relaxed,
        );
    }
    let paths: Vec<Path> = get("--paths")
        .unwrap_or("default,tiled,loop".into())
        .split(',')
        .map(|p| match p {
            "default" => Path::Default,
            "tiled" => Path::Tiled,
            "loop" => Path::Loop,
            _ => panic!("unknown path {p}"),
        })
        .collect();
    let filter = get("--classes").map(|s| s.split(',').map(|x| x.to_string()).collect::<Vec<_>>());
    let (z, c) = read_xyz(&a[1]);
    let def = BasisDef::load(&a[2], None).unwrap();
    let basis = BasisSet::new(&def, &z, &c, !a.iter().any(|x| x == "--cart")).unwrap();
    let set = PairSet::build(&basis, PairOptions::default());
    let mut bp = BatchedPairs::new(&basis, &set, 1e-14, 1e-14);
    let isa = get("--isa").unwrap_or_else(|| format!("{:?}", Isa::detect()).to_lowercase());
    match isa.as_str() {
        #[cfg(target_arch = "x86_64")]
        "avx512" => unsafe { run_avx512(&mut bp, &paths, reps, &filter, min_l) },
        #[cfg(target_arch = "x86_64")]
        "avx2" => unsafe { run_avx2(&mut bp, &paths, reps, &filter, min_l) },
        _ => run::<Portable<false>>(&mut bp, &paths, reps, &filter, min_l, "generic"),
    }
}
