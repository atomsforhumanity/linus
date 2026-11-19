//! det_check [n]: are faer's parallel matmul and symmetric eigensolver bitwise independent of
//! the thread count? Prints max differences against the 1-thread result, then the same for
//! the eigensolver at a fixed parallel degree (8) inside pools of 1..32 threads, and timings.
use faer::{Mat, Par};
use linus_base::linalg::{eigh, mm};

fn md(x: &Mat<f64>, y: &Mat<f64>) -> f64 {
    (0..x.nrows())
        .flat_map(|i| (0..x.ncols()).map(move |j| (i, j)))
        .map(|(i, j)| (x[(i, j)] - y[(i, j)]).abs())
        .fold(0.0f64, f64::max)
}

fn main() {
    let n: usize = std::env::args()
        .nth(1)
        .map(|s| s.parse().unwrap())
        .unwrap_or(600);
    let a = Mat::from_fn(n, n, |i, j| {
        ((i * 31 + j * 17) as f64 * 0.013).sin() / (1.0 + (i as f64 - j as f64).abs())
    });
    let s = &a + a.transpose();
    let b = Mat::from_fn(n, n / 3, |i, j| ((i * 7 + j * 3) as f64 * 0.029).cos());
    let run = |t: usize, deg: Option<usize>| {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(t)
            .build()
            .unwrap();
        pool.install(|| {
            let par = match deg {
                Some(d) => Par::rayon(d),
                None if t > 1 => Par::rayon(t),
                None => Par::Seq,
            };
            let c = mm(s.as_ref(), b.as_ref(), par);
            let d = mm(b.as_ref(), b.transpose(), par);
            let t0 = std::time::Instant::now();
            let (w, u) = eigh(s.as_ref(), par);
            (c, d, w, u, t0.elapsed().as_secs_f64())
        })
    };
    let r1 = run(1, None);
    println!("n {n}: eigh 1 thread {:.3}s", r1.4);
    for t in [2, 8, 32] {
        let r = run(t, None);
        let wd =
            r.2.iter()
                .zip(&r1.2)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f64, f64::max);
        println!("threads {t:2}: matmul S*B {:.3e}  B*B^T {:.3e}  eigvals {:.3e}  eigvecs {:.3e}  eigh {:.3}s", md(&r.0, &r1.0), md(&r.1, &r1.1), wd, md(&r.3, &r1.3), r.4);
    }
    let f1 = run(1, Some(8));
    for t in [2, 8, 32] {
        let r = run(t, Some(8));
        let wd =
            r.2.iter()
                .zip(&f1.2)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f64, f64::max);
        println!("Par::rayon(8) in {t:2}-thread pool vs 1-thread pool: eigvals {:.3e} eigvecs {:.3e}  eigh {:.3}s", wd, md(&r.3, &f1.3), r.4);
    }
}
