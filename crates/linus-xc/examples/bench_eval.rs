//! linus-xc throughput on the batch written by `bench_pyscf.py`.
//!
//! ```text
//! OMP_NUM_THREADS=1 .venv/bin/python crates/linus-xc/examples/bench_pyscf.py /tmp/xcbench
//! cargo run --release -p linus-xc --example bench_eval -- /tmp/xcbench [threads]
//! ```
//! Single thread: the whole batch in grid blocks of 128, 1024 and all points,
//! one reused `XcOutput` (as a grid loop would do). With `threads` > 1, also a
//! rayon run over 1024-point blocks sharing one `Functional`.

use linus_xc::{Functional, XcInput, XcOutput};
use rayon::prelude::*;
use std::time::Instant;

fn read(path: &str) -> Vec<f64> {
    let b = std::fs::read(path).unwrap_or_else(|e| panic!("{path}: {e}"));
    b.as_chunks::<8>()
        .0
        .iter()
        .map(|c| f64::from_le_bytes(*c))
        .collect()
}

fn best_of<F: FnMut()>(reps: usize, mut f: F) -> f64 {
    (0..reps)
        .map(|_| {
            let t = Instant::now();
            f();
            t.elapsed().as_secs_f64()
        })
        .fold(f64::INFINITY, f64::min)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let dir = &args[1];
    let threads: usize = args.get(2).map_or(1, |s| s.parse().unwrap());
    println!(
        "linus-xc (libxc {:?}), release build",
        linus_xc::libxc_version()
    );
    println!("| functional | spin | order | block | threads | Mpoints/s |");
    println!("|---|---|---|---|---|---|");
    for name in ["b3lyp", "r2scan"] {
        for spin in ["unpolarized", "polarized"] {
            let polarized = spin == "polarized";
            let (ns, nsig) = if polarized { (2, 3) } else { (1, 1) };
            let rho = read(&format!("{dir}/{spin}_rho.f64"));
            let sigma = read(&format!("{dir}/{spin}_sigma.f64"));
            let tau = read(&format!("{dir}/{spin}_tau.f64"));
            let np = rho.len() / ns;
            let f = Functional::new(name, polarized).unwrap();
            let block_input = |a: usize, b: usize| XcInput {
                rho: &rho[a * ns..b * ns],
                sigma: Some(&sigma[a * nsig..b * nsig]),
                lapl: None,
                tau: Some(&tau[a * ns..b * ns]),
            };
            for order in [1, 2] {
                for block in [128, 1024, np] {
                    let mut out = XcOutput::new();
                    let t = best_of(5, || {
                        for a in (0..np).step_by(block) {
                            f.eval(order, &block_input(a, (a + block).min(np)), &mut out)
                                .unwrap();
                        }
                    });
                    let label = if block == np {
                        "all".to_string()
                    } else {
                        block.to_string()
                    };
                    println!(
                        "| {name} | {spin} | {order} | {label} | 1 | {:.3} |",
                        np as f64 / t / 1e6
                    );
                }
                if threads > 1 {
                    let pool = rayon::ThreadPoolBuilder::new()
                        .num_threads(threads)
                        .build()
                        .unwrap();
                    let block = 1024;
                    let t = best_of(5, || {
                        pool.install(|| {
                            (0..np.div_ceil(block)).into_par_iter().for_each_init(
                                XcOutput::new,
                                |out, c| {
                                    let a = c * block;
                                    f.eval(order, &block_input(a, (a + block).min(np)), out)
                                        .unwrap();
                                },
                            )
                        })
                    });
                    println!(
                        "| {name} | {spin} | {order} | {block} | {threads} | {:.3} |",
                        np as f64 / t / 1e6
                    );
                }
            }
        }
    }
}
