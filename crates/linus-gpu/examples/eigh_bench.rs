//! The SCF's diagonalization on the device: cuSOLVER's syevd, Xsyevd and syevj
//! ([`EighMethod`]) for matrices of the orders the SCF loop meets, timed with device events
//! (median of the repetitions after a warm-up), with the eigenpair residual and a bitwise
//! repeat check.
//!
//! ```text
//! eigh_bench [n ...] [--reps 7]
//! ```

mod common;
use common::{device_ms, median};
use linus_gpu::solver::{EighMethod, Solver};
use linus_gpu::Gpu;

/// xorshift64*
struct Rng(u64);

impl Rng {
    fn uniform(&mut self) -> f64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        (self.0.wrapping_mul(0x2545_f491_4f6c_dd1d) >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// A Fock-like symmetric matrix: a spread diagonal (orbital energies -20..+5 Eh) plus a
/// decaying random coupling of size `coupling` (0.5: a Fock matrix in a fixed basis; 1e-4: a
/// late SCF iteration's Fock matrix in the previous iteration's orbitals).
fn fock_like(n: usize, seed: u64, coupling: f64) -> Vec<f64> {
    let mut rng = Rng(seed);
    let mut a = vec![0.0; n * n];
    for j in 0..n {
        for i in j..n {
            let v = if i == j {
                -20.0 + 25.0 * (i as f64 / n as f64).powf(0.3)
            } else {
                (rng.uniform() - 0.5) * coupling / (1.0 + 0.05 * (i - j) as f64)
            };
            a[j * n + i] = v;
            a[i * n + j] = v;
        }
    }
    a
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let reps: usize = args
        .iter()
        .position(|a| a == "--reps")
        .map(|i| args[i + 1].parse().unwrap())
        .unwrap_or(7);
    let mut sizes: Vec<usize> = args
        .iter()
        .take_while(|a| !a.starts_with("--"))
        .map(|a| a.parse().unwrap())
        .collect();
    if sizes.is_empty() {
        sizes = vec![200, 450, 1034, 1500, 2155];
    }
    let gpu = Gpu::new(0).expect("GPU");
    let solver = Solver::new(&gpu).unwrap();
    println!("n matrix method ms residual bitwise_repeat");
    for &n in &sizes {
        for (kind, coupling) in [("dense", 0.5), ("near-diagonal", 1e-4)] {
            let a = fock_like(n, 7 + n as u64, coupling);
            for method in [EighMethod::Syevd, EighMethod::Xsyevd, EighMethod::Syevj] {
                let mut plan = solver.eigh_plan(&gpu, n, method, 1).unwrap();
                let mut da = gpu.upload(&a).unwrap();
                let mut dw = gpu.alloc_zeros::<f64>(n).unwrap();
                let mut times = vec![];
                let mut first: Option<(Vec<f64>, Vec<f64>)> = None;
                let mut same = true;
                for r in 0..reps + 1 {
                    gpu.upload_into(&mut da, 0, &a).unwrap();
                    let ms = device_ms(&gpu, || solver.eigh_queued(&mut plan, &mut da, &mut dw, 0))
                        .unwrap();
                    if r > 0 {
                        times.push(ms);
                    }
                    let (x, w) = (gpu.download(&da).unwrap(), gpu.download(&dw).unwrap());
                    assert_eq!(gpu.download(plan.info()).unwrap()[0], 0, "{method:?} n {n}");
                    match &first {
                        None => first = Some((x, w)),
                        Some((x0, w0)) => {
                            same &= x0.iter().zip(&x).all(|(p, q)| p.to_bits() == q.to_bits())
                                && w0.iter().zip(&w).all(|(p, q)| p.to_bits() == q.to_bits())
                        }
                    }
                }
                let (x, w) = first.unwrap();
                // max over a few eigenpairs of |A x - w x|
                let mut res = 0.0f64;
                for k in [0, n / 2, n - 1] {
                    for i in 0..n {
                        let ax: f64 = (0..n).map(|j| a[j * n + i] * x[k * n + j]).sum();
                        res = res.max((ax - w[k] * x[k * n + i]).abs());
                    }
                }
                println!(
                    "{n} {kind} {method:?} {:.2} {res:.1e} {same}",
                    median(times)
                );
            }
        }
    }
}
