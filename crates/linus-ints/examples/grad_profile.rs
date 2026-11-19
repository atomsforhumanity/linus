//! Per-class cost profile of the 4c gradient kernel (single thread): grad_profile <input.json>
//! (same input as grad_bench). Prints time, quartet and primitive-quartet counts per
//! (la lb | lc ld) class, sorted by time.
use linus_ints::eri4c::deriv::{quartet_grad, DerivWork};
use linus_ints::{basis::BasisDef, BasisSet, DerivOrder, PairOptions, PairSet};
use std::collections::HashMap;
use std::time::Instant;

#[allow(clippy::type_complexity, reason = "a one-off tuple of buffers")]
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
    let def = BasisDef::load(inp["basis"].as_str().unwrap(), None).unwrap();
    let basis = BasisSet::new(&def, &z, &positions, true).unwrap();
    let pairs = PairSet::build(&basis, PairOptions::with_order(DerivOrder::One));
    let q = pairs.schwarz();
    let sample: usize = args.get(2).map(|s| s.parse().unwrap()).unwrap_or(7);
    let mut w = DerivWork::new();
    let mut stats: HashMap<(usize, usize, usize, usize), (f64, usize, usize)> = HashMap::new();
    let np = pairs.len();
    let mut gbuf = Vec::new();
    for ij in (0..np).step_by(sample) {
        let bra = pairs.pair(ij);
        for kl in 0..=ij {
            let ket = pairs.pair(kl);
            if q[ij] * q[kl] < 1e-11 {
                continue;
            }
            gbuf.resize(bra.nab() * ket.nab(), 0.01);
            let t = Instant::now();
            let g = quartet_grad(
                bra,
                pairs.deriv(ij),
                ket,
                pairs.deriv(kl),
                Some(&gbuf),
                &[],
                &mut w,
            );
            let dt = t.elapsed().as_secs_f64();
            std::hint::black_box(g);
            let e = stats
                .entry((bra.la, bra.lb, ket.la, ket.lb))
                .or_insert((0.0, 0, 0));
            e.0 += dt;
            e.1 += 1;
            e.2 += bra.nprim * ket.nprim;
        }
    }
    let mut v: Vec<_> = stats.into_iter().collect();
    v.sort_by(|a, b| b.1 .0.partial_cmp(&a.1 .0).unwrap());
    let tot: f64 = v.iter().map(|x| x.1 .0).sum();
    println!("total sampled {tot:.3} s (every {sample}th bra pair)");
    for ((a, b, c, d), (t, n, np)) in v.iter().take(25) {
        println!(
            "({a}{b}|{c}{d})  {:6.2}%  t={t:8.3}s  quartets={n:9}  primq={np:10}  ns/primq={:7.1}",
            100.0 * t / tot,
            1e9 * t / *np as f64
        );
    }
}
