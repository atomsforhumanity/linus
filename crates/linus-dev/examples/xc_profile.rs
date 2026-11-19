//! xc_profile <xyz> <basis> [--charge Q] [--func F] [--grid orca:2] [--threads N] [--reps R]
//! [--what energy,gradient,kernel_occ,hess_terms] [--npert P] [--batch N] [--cache DIR] [--prof]: wall time (best of R) of
//! the XC energy+potential, the XC gradient (with grid response) and the closed-shell CPKS
//! kernel contraction (3N first-order densities A C_o^T + C_o A^T, A = C_v U) at the converged
//! RKS density, as JSON. The orbitals come from an RI-JK SCF on the same grid, cached in DIR
//! (default: none). With --prof, also the per-phase thread time of each part.
use faer::{Mat, Par};
use linus::{CalcOptions, GridChoice, JkMethod, Session};
use linus_base::linalg::mm;
use linus_core::channels::Channels;
use std::io::{Read, Write};
use std::time::Instant;

fn parse_grid(s: &str) -> GridChoice {
    let (k, n) = s.split_once(':').expect("grid kind:level");
    let n: u8 = n.parse().unwrap();
    match k {
        "orca" => GridChoice::Orca { defgrid: n },
        "linus" => GridChoice::Linus { level: n },
        "pyscf" => GridChoice::Pyscf { level: n },
        _ => panic!("grid {s}"),
    }
}

fn save(path: &std::path::Path, c: &Mat<f64>, eps: &[f64], nocc: usize) {
    let mut v: Vec<f64> = vec![c.nrows() as f64, c.ncols() as f64, nocc as f64];
    for j in 0..c.ncols() {
        for i in 0..c.nrows() {
            v.push(c[(i, j)]);
        }
    }
    v.extend_from_slice(eps);
    let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
    std::fs::File::create(path)
        .unwrap()
        .write_all(&bytes)
        .unwrap();
}

fn load(path: &std::path::Path) -> Option<(Mat<f64>, Vec<f64>, usize)> {
    let mut b = Vec::new();
    std::fs::File::open(path).ok()?.read_to_end(&mut b).ok()?;
    let v: Vec<f64> = b
        .as_chunks::<8>()
        .0
        .iter()
        .map(|c| f64::from_le_bytes(*c))
        .collect();
    let (n, m, o) = (v[0] as usize, v[1] as usize, v[2] as usize);
    let c = Mat::from_fn(n, m, |i, j| v[3 + j * n + i]);
    Some((c, v[3 + n * m..3 + n * m + m].to_vec(), o))
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let get = |k: &str| a.iter().position(|x| x == k).map(|i| a[i + 1].clone());
    let threads: usize = get("--threads").map(|s| s.parse().unwrap()).unwrap_or(8);
    let reps: usize = get("--reps").map(|s| s.parse().unwrap()).unwrap_or(3);
    let charge: i32 = get("--charge").map(|s| s.parse().unwrap()).unwrap_or(0);
    let func = get("--func").unwrap_or("b3lyp".into());
    let grid = get("--grid").unwrap_or("orca:2".into());
    let what = get("--what").unwrap_or("energy,gradient,kernel_occ".into());
    let prof = a.iter().any(|x| x == "--prof");
    let mol = linus::Structure::from_xyz_str(&std::fs::read_to_string(&a[1]).unwrap(), charge, 1)
        .unwrap();
    let mut o = CalcOptions {
        basis: Some(a[2].clone()),
        method: Some(func.clone()),
        threads: Some(threads),
        grid: linus::driver::options::GridSettings {
            xc: Some(parse_grid(&grid)),
            ..Default::default()
        },
        ..Default::default()
    };
    o.jk = JkMethod::Df {
        aux: "def2-universal-jkfit".into(),
    }
    .into();
    o.scf.e_tol = Some(linus::driver::options::Auto::Value(1e-9));
    o.scf.g_tol = Some(linus::driver::options::Auto::Value(1e-7));
    let mut calc = Session::new(mol, o).unwrap();
    let stem = std::path::Path::new(&a[1])
        .file_stem()
        .unwrap()
        .to_string_lossy()
        .to_string();
    let cache = get("--cache").map(|d| {
        std::path::PathBuf::from(d).join(format!(
            "{stem}_{}_{func}_{}.orb",
            a[2],
            grid.replace(':', "")
        ))
    });
    // --sad: orbitals of the SAD density (eigenvectors; no SCF) for systems too big to converge here
    let sad = a.iter().any(|x| x == "--sad");
    let (c, eps, nocc) = match cache.as_ref().and_then(|p| load(p)) {
        Some(x) if !sad => x,
        _ if sad => {
            let (na, nb) = calc.mol().nalpha_nbeta().unwrap();
            let (da, _) = linus_scf::guess::sad_guess(
                calc.basis_def(),
                calc.basis(),
                &calc.mol().z,
                na,
                nb,
                Par::Seq,
            );
            let (ev, u) = linus_base::linalg::eigh(da.as_ref(), Par::Seq);
            let n = ev.len();
            // natural orbitals of D_a, largest occupations first, scaled to reproduce D_a
            let c = Mat::from_fn(n, n, |i, j| {
                u[(i, n - 1 - j)] * ev[n - 1 - j].max(0.0).sqrt()
            });
            (c, vec![0.0; n], na)
        }
        _ => {
            let r = calc.energy().unwrap();
            if let Some(p) = &cache {
                save(p, &r.c_a, &r.eps_a, r.nalpha);
            }
            (r.c_a.clone(), r.eps_a.clone(), r.nalpha)
        }
    };
    let _ = eps;
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .unwrap();
    let co = c.subcols(0, nocc).to_owned();
    let cv = c.subcols(nocc, c.ncols() - nocc).to_owned();
    let da = mm(co.as_ref(), co.transpose(), Par::Seq);
    let mut t = linus_core::timing::Timers::default();
    let t0 = Instant::now();
    // --batch N: grid batches of at most N points (default: the grid preset's)
    let xc = match get("--batch").map(|s| s.parse::<usize>().unwrap()) {
        Some(nb) => {
            let mut spec = calc.opts().grid.xc().spec();
            spec.batching.max_points = nb;
            let g = pool
                .install(|| {
                    linus_grid::MolecularGrid::build(&calc.mol().z, &calc.mol().positions, &spec)
                })
                .unwrap();
            linus_dft::XcIntegrator::new(calc.basis(), g, &func).unwrap()
        }
        None => pool.install(|| calc.build_xc(&mut t)).unwrap().unwrap(),
    };
    let t_grid = t0.elapsed().as_secs_f64();
    let mut out = serde_json::json!({"mol": stem, "basis": a[2], "func": func, "grid": grid, "nbf": calc.basis().nbf, "nocc": nocc,
        "npoints": xc.grid.x.len(), "nbatches": xc.grid.batches.len(), "threads": threads, "t_grid": t_grid});
    let run = |f: &mut (dyn FnMut() + Send)| -> (f64, Vec<(&'static str, f64)>) {
        let mut best = f64::INFINITY;
        let mut prof_best = vec![];
        for _ in 0..reps {
            linus_dft::profile::enable(prof);
            linus_dft::profile::reset();
            let t0 = Instant::now();
            pool.install(&mut *f);
            let el = t0.elapsed().as_secs_f64();
            linus_dft::profile::enable(false);
            if el < best {
                best = el;
                prof_best = linus_dft::profile::report();
                let st = linus_dft::profile::batch_stats();
                for (k, v) in [
                    "nb",
                    "mean_np",
                    "mean_nao",
                    "w_nao",
                    "rms_nao",
                    "mean_cand",
                    "mean_eval",
                ]
                .iter()
                .zip(st)
                {
                    prof_best.push((k, v));
                }
            }
        }
        (best, prof_best)
    };
    let record = |out: &mut serde_json::Value, name: &str, r: (f64, Vec<(&'static str, f64)>)| {
        out[name] = serde_json::json!(r.0);
        if prof {
            out[format!("{name}_prof")] = serde_json::Value::Object(
                r.1.iter()
                    .map(|(k, v)| (k.to_string(), serde_json::json!(v)))
                    .collect(),
            );
        }
    };
    for w in what.split(',') {
        match w {
            "energy" => {
                let mut e = 0.0;
                let r = run(&mut || e = xc.potential(&da, &da, true).0);
                record(&mut out, "energy", r);
                out["exc"] = serde_json::json!(e);
            }
            "gradient" => {
                let r = run(&mut || {
                    std::hint::black_box(xc.gradient(&da, &da, true, true));
                });
                record(&mut out, "gradient", r);
            }
            "kernel_occ" => {
                let npert = get("--npert")
                    .map(|s| s.parse().unwrap())
                    .unwrap_or(3 * calc.mol().natom())
                    .min(3 * calc.mol().natom());
                let a1: Vec<Mat<f64>> = (0..npert)
                    .map(|p| {
                        let u = Mat::from_fn(cv.ncols(), co.ncols(), |a, i| {
                            (((a * 7 + i * 13 + p * 5) % 17) as f64 - 8.0) * 1e-3
                        });
                        mm(cv.as_ref(), u.as_ref(), Par::Seq)
                    })
                    .collect();
                let r = run(&mut || {
                    std::hint::black_box(xc.kernel_contract_occ(
                        Channels::Restricted(co.as_ref()),
                        Channels::Restricted(&a1),
                    ));
                });
                record(&mut out, "kernel_occ", r);
                out["npert"] = serde_json::json!(a1.len());
            }
            "hess_terms" => {
                let r = run(&mut || {
                    let mut t = linus_core::timing::Timers::default();
                    let par = Par::rayon(threads);
                    std::hint::black_box(xc.hessian_terms(
                        Channels::Restricted(&da),
                        Channels::Restricted(co.as_ref()),
                        false,
                        par,
                        &mut t,
                    ));
                });
                record(&mut out, "hess_terms", r);
            }
            _ => panic!("unknown part {w}"),
        }
    }
    println!("{out}");
}
