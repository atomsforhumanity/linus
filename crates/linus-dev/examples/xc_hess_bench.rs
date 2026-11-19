//! xc_hess_bench <xyz> <basis> <functional> [--threads N] [--reps R]: wall time of the XC
//! Hessian terms (V_xc^x C_o and E_xc^xy on a fixed grid, `terms`; with the grid response,
//! `terms_grid`) and one CPKS kernel contraction for 3N first-order densities, at the converged
//! RKS density, as JSON.
use linus::{CalcOptions, Session};
use linus_base::linalg::mm;
use linus_core::channels::Channels;
use std::time::Instant;

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let get = |k: &str| a.iter().position(|x| x == k).map(|i| a[i + 1].clone());
    let threads: usize = get("--threads").map(|s| s.parse().unwrap()).unwrap_or(8);
    let reps: usize = get("--reps").map(|s| s.parse().unwrap()).unwrap_or(3);
    let mol =
        linus::Structure::from_xyz_str(&std::fs::read_to_string(&a[1]).unwrap(), 0, 1).unwrap();
    let o = CalcOptions {
        basis: Some(a[2].clone()),
        method: Some(a[3].clone()),
        threads: Some(threads),
        ..Default::default()
    };
    let mut calc = Session::new(mol, o).unwrap();
    let r = calc.energy().unwrap();
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .unwrap();
    let co = r.c_a.subcols(0, r.nalpha).to_owned();
    let da = mm(co.as_ref(), co.transpose(), faer::Par::Seq);
    let mut t = linus_core::timing::Timers::default();
    let xc = pool.install(|| calc.build_xc(&mut t)).unwrap().unwrap();
    let best = |f: &(dyn Fn() + Sync)| -> f64 {
        (0..reps)
            .map(|_| {
                let t0 = Instant::now();
                pool.install(f);
                t0.elapsed().as_secs_f64()
            })
            .fold(f64::INFINITY, f64::min)
    };
    // the explicit XC terms of a Hessian as it runs them (projected on C_o), on a fixed grid and
    // with the grid response
    let par = faer::Par::rayon(threads);
    let terms = |grid: bool| {
        let mut t = linus_core::timing::Timers::default();
        std::hint::black_box(xc.hessian_terms(
            Channels::Restricted(&da),
            Channels::Restricted(co.as_ref()),
            grid,
            par,
            &mut t,
        ));
    };
    let tb = best(&|| terms(false));
    let tg = best(&|| terms(true));
    // the CPKS kernel contraction for 3N first-order densities A C_o^T + C_o A^T (one CPKS
    // iteration)
    let cv = r.c_a.subcols(r.nalpha, r.nmo - r.nalpha).to_owned();
    let a1: Vec<faer::Mat<f64>> = (0..3 * calc.mol().natom())
        .map(|p| {
            let u = faer::Mat::from_fn(cv.ncols(), co.ncols(), |a, i| {
                (((a * 7 + i * 13 + p * 5) % 17) as f64 - 8.0) * 1e-3
            });
            mm(cv.as_ref(), u.as_ref(), faer::Par::Seq)
        })
        .collect();
    let tk = best(&|| {
        std::hint::black_box(
            xc.kernel_contract_occ(Channels::Restricted(co.as_ref()), Channels::Restricted(&a1)),
        );
    });
    println!(
        "{}",
        serde_json::json!({"nbf": calc.basis().nbf, "npoints": xc.grid.x.len(), "nbatches": xc.grid.batches.len(), "threads": threads,
            "terms": tb, "kernel_3n": tk, "terms_grid": tg})
    );
}
