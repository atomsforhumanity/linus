//! cosx_profile <xyz> <basis> <charge> <cache_dir> [--level 3] [--thresh 1e-11] [--reps 3]
//!   [--threads 8] [--bmax 64] [--profile] [--grad] [--grad-ref] [--no-energy] [--ref-thresh 1e-14]
//!
//! Profile of the COSX exchange build and its gradient for a cached DF-B3LYP density (the
//! `<mol>_<basis>.dens` / `.kex` files written by `cosx_grids`): wall time per build (median of
//! `reps`), the build's phase times, screening counters, per-class integral times (with
//! `--profile`), and the exchange-energy error against exact K (a_x = 0.2, closed shell).
//! `--grad`: the same for the COSX gradient (fitting and grid response on); `--grad-ref` adds a
//! reference gradient at `--ref-thresh` without distance screening and reports the screening
//! error. Prints one JSON object.
use faer::{Mat, Par};
use linus::Structure;
use linus_base::linalg::{dot, mat_from_rowmajor};
use linus_grid::{GridSpec, MolecularGrid};
use linus_ints::{basis::BasisDef, BasisSet};
use linus_jk::cosx::gradient::{exchange_gradient, CosxGradientConfig};
use linus_jk::cosx::{ClassStat, CosxConfig, CosxK};
use serde_json::json;
use std::io::Read;
use std::time::Instant;

fn read_f64s(path: &std::path::Path) -> Vec<f64> {
    let mut f = std::fs::File::open(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let mut b = Vec::new();
    f.read_to_end(&mut b).unwrap();
    b.as_chunks::<8>()
        .0
        .iter()
        .map(|c| f64::from_le_bytes(*c))
        .collect()
}

fn classes_json(c: &[ClassStat]) -> serde_json::Value {
    json!(c.iter().map(|s| json!({"class": format!("{}{}", s.l1, s.l2), "pair_batches": s.pair_batches, "prim_points": s.prim_points, "seconds": s.seconds,
        "ns_per_prim_point": s.seconds * 1e9 / (s.prim_points.max(1) as f64)})).collect::<Vec<_>>())
}

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let get = |k: &str| a.iter().position(|x| x == k).map(|i| a[i + 1].clone());
    let has = |k: &str| a.iter().any(|x| x == k);
    let charge: i32 = a[3].parse().unwrap();
    let cache = std::path::PathBuf::from(&a[4]);
    let level: u8 = get("--level").map(|s| s.parse().unwrap()).unwrap_or(3);
    let thresh: f64 = get("--thresh").map(|s| s.parse().unwrap()).unwrap_or(1e-11);
    let reps: usize = get("--reps").map(|s| s.parse().unwrap()).unwrap_or(3);
    let nt: usize = get("--threads").map(|s| s.parse().unwrap()).unwrap_or(8);
    let profile = has("--profile");
    let mol = Structure::from_xyz_str(&std::fs::read_to_string(&a[1]).unwrap(), charge, 1).unwrap();
    let par = if nt > 1 { Par::rayon(nt) } else { Par::Seq };
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(nt)
        .build()
        .unwrap();
    pool.install(|| {
        let def = BasisDef::load(&a[2], None).unwrap();
        let basis = BasisSet::new(&def, &mol.z, &mol.positions, true).unwrap();
        let n = basis.nbf;
        let molname = std::path::Path::new(&a[1]).file_stem().unwrap().to_string_lossy().to_string();
        let tag = format!("{molname}_{}", a[2]);
        let d = mat_from_rowmajor(n, n, &read_f64s(&cache.join(format!("{tag}.dens"))));
        // exact K (optional: without it the exchange-energy errors are NaN)
        let pk = cache.join(format!("{tag}.kex"));
        let kex = if pk.exists() { mat_from_rowmajor(n, n, &read_f64s(&pk)) } else { Mat::from_fn(n, n, |_, _| f64::NAN) };
        let s = linus_dev::overlap(&basis);
        let s = mat_from_rowmajor(n, n, &s);
        let mut spec = GridSpec::cosx(level);
        if let Some(b) = get("--bmax") {
            spec.batching.max_points = b.parse().unwrap();
        }
        let mg = MolecularGrid::build(&mol.z, &mol.positions, &spec).unwrap();
        let ax = 0.2;
        let ex_exact = -ax * dot(d.as_ref(), kex.as_ref());
        let mut out = json!({"case": tag, "nbf": n, "nshell": basis.shells.len(), "level": level, "thresh": thresh, "threads": nt, "npoints": mg.x.len(), "nbatches": mg.batches.len()});
        if !has("--no-energy") {
            let opts = CosxConfig { thresh, profile, ..Default::default() };
            let t0 = Instant::now();
            let mut k = CosxK::new(&basis, mg.clone(), opts).with_overlap_fitting(s.as_ref());
            let t_setup = t0.elapsed().as_secs_f64();
            let mut times = vec![];
            let mut kk = Mat::<f64>::zeros(n, n);
            for _ in 0..reps {
                let t = Instant::now();
                kk = k.build(std::slice::from_ref(&d), par).remove(0);
                times.push(t.elapsed().as_secs_f64());
            }
            let e = -ax * dot(d.as_ref(), kk.as_ref());
            let mut maxd: f64 = 0.0;
            for j in 0..n {
                for i in 0..n {
                    maxd = maxd.max((kk[(i, j)] - kex[(i, j)]).abs());
                }
            }
            let st = &k.stats;
            out["energy"] = json!({
                "setup": t_setup, "times": times, "median": median(&mut times.clone()), "npairs": k.npairs(), "nprims": k.nprims(),
                "t_ints": st.t_ints, "t_gemm": st.t_gemm, "t_other": st.t_other, "pair_batches": st.pair_batches, "prim_points": st.prim_points,
                "prim_per_point": st.prim_points as f64 / mg.x.len() as f64, "dEx": e - ex_exact, "max_dK": maxd, "classes": classes_json(&st.classes),
            });
            // energy-only evaluations at the thresholds of --ethresh (comma separated)
            if let Some(list) = get("--ethresh") {
                let mut rows = vec![];
                for th in list.split(',').map(|x| x.parse::<f64>().unwrap()) {
                    k.opts.thresh_e = th;
                    let mut times = vec![];
                    let mut ee = 0.0;
                    for _ in 0..reps {
                        let t = Instant::now();
                        ee = k.energy(std::slice::from_ref(&d), &[-ax], par);
                        times.push(t.elapsed().as_secs_f64());
                    }
                    let st = &k.stats;
                    rows.push(json!({"thresh_e": th, "times": times, "median": median(&mut times.clone()), "energy": ee, "dEx": ee - ex_exact, "minus_kpath": ee - e,
                        "t_ints": st.t_ints, "t_gemm": st.t_gemm, "t_other": st.t_other, "pair_batches": st.pair_batches, "prim_points": st.prim_points, "classes": classes_json(&st.classes)}));
                }
                out["energy"]["e_only"] = json!(rows);
            }
        }
        if has("--grad") {
            // variants: --gvar "K:1e-9,D:1e-11:1e-9:10" (K: K criterion at thresh; D: density-weighted
            // screening thresh_d : K' thresh : wfac); default: K at --thresh
            let vars: Vec<CosxGradientConfig> = match get("--gvar") {
                Some(list) => list
                    .split(',')
                    .map(|v| {
                        let f: Vec<&str> = v.split(':').collect();
                        match f[0] {
                            "K" => CosxGradientConfig { thresh: f[1].parse().unwrap(), thresh_d: None, profile, ..Default::default() },
                            _ => CosxGradientConfig {
                                thresh_d: Some(f[1].parse().unwrap()),
                                thresh_k: f[2].parse().unwrap(),
                                wfac: f[3].parse().unwrap(),
                                g_factor: f.get(4).map(|x| x.parse().unwrap()).unwrap_or(1.0),
                                profile,
                                ..Default::default()
                            },
                        }
                    })
                    .collect(),
                None => vec![CosxGradientConfig { profile, ..Default::default() }],
            };
            let parts_of = |r: &linus_jk::cosx::gradient::CosxGradResult| -> Vec<Vec<[f64; 3]>> {
                let p = &r.parts;
                vec![r.gradient.clone(), p.centres.clone(), p.ao.clone(), p.points.clone(), p.weights.clone(), p.fit_s.clone(), p.fit_snum.clone()]
            };
            let names = ["total", "centres", "ao", "points", "weights", "fit_s", "fit_snum"];
            // reference: K criterion at --ref-thresh without distance screening (cached as JSON)
            let reference: Option<(Vec<Vec<[f64; 3]>>, f64)> = has("--grad-ref").then(|| {
                let rt: f64 = get("--ref-thresh").map(|s| s.parse().unwrap()).unwrap_or(1e-14);
                let cache_file = cache.join(format!("{tag}_gref_l{level}_{rt:e}{}.json", if has("--ref-dist") { "_dist" } else { "" }));
                if let Ok(t) = std::fs::read_to_string(&cache_file) {
                    let v: serde_json::Value = serde_json::from_str(&t).unwrap();
                    let parts: Vec<Vec<[f64; 3]>> = serde_json::from_value(v["parts"].clone()).unwrap();
                    (parts, v["energy"].as_f64().unwrap())
                } else {
                    let ro = CosxGradientConfig { thresh: rt, thresh_d: None, dist_screen: has("--ref-dist"), ..Default::default() };
                    let rr = exchange_gradient(&basis, &mg, std::slice::from_ref(&d), &[-ax], &ro, par).unwrap();
                    let parts = parts_of(&rr);
                    std::fs::write(&cache_file, json!({"parts": parts, "energy": rr.energy, "prim_points": rr.prim_points}).to_string()).unwrap();
                    (parts, rr.energy)
                }
            });
            let mut rows = vec![];
            let mut first: Option<Vec<[f64; 3]>> = None;
            for gopts in &vars {
                let mut times = vec![];
                let mut res = None;
                for _ in 0..reps {
                    let t = Instant::now();
                    res = Some(exchange_gradient(&basis, &mg, std::slice::from_ref(&d), &[-ax], gopts, par).unwrap());
                    times.push(t.elapsed().as_secs_f64());
                }
                let r = res.unwrap();
                let mut gj = json!({
                    "thresh": if gopts.thresh_d.is_some() { gopts.thresh_k } else { gopts.thresh }, "thresh_d": gopts.thresh_d, "wfac": gopts.wfac, "g_factor": gopts.g_factor,
                    "times": times, "median": median(&mut times.clone()), "timers": r.timers.to_json(), "pair_batches": r.pair_batches, "prim_points": r.prim_points,
                    "g_pair_batches": r.g_pair_batches, "g_prim_points": r.g_prim_points, "mean_nx": r.sum_nx as f64 / mg.batches.len() as f64, "mean_nt": r.sum_nt as f64 / mg.batches.len() as f64, "nbf": basis.nbf,
                    "prim_per_point": r.prim_points as f64 / mg.x.len() as f64, "energy": r.energy, "classes": classes_json(&r.classes), "classes_g": classes_json(&r.classes_g),
                });
                if let Some((rp, re)) = &reference {
                    let mine = parts_of(&r);
                    let mut errs = serde_json::Map::new();
                    for (i, n) in names.iter().enumerate() {
                        let e = mine[i].iter().zip(&rp[i]).flat_map(|(x, y)| (0..3).map(move |k| (x[k] - y[k]).abs())).fold(0.0f64, f64::max);
                        errs.insert(n.to_string(), json!(e));
                    }
                    gj["screen_err"] = errs["total"].clone();
                    gj["part_err"] = json!(errs);
                    gj["dE"] = json!(r.energy - re);
                }
                // change against the first variant
                match &first {
                    None => first = Some(r.gradient.clone()),
                    Some(f0) => {
                        let e = r.gradient.iter().zip(f0).flat_map(|(x, y)| (0..3).map(move |k| (x[k] - y[k]).abs())).fold(0.0f64, f64::max);
                        gj["vs_first"] = json!(e);
                    }
                }
                if has("--dump") {
                    gj["gradient"] = json!(r.gradient);
                }
                rows.push(gj);
            }
            out["grad"] = if rows.len() == 1 { rows.remove(0) } else { json!(rows) };
        }
        println!("{}", out);
    });
}
