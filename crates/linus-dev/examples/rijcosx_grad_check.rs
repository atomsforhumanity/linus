//! Validation of RIJCOSX (jk = rij-snk) gradients: the analytic gradient, the COSX gradient's
//! parts and variants (grid weights, point motion, grid), errors against exact-integral
//! gradients and central finite differences of the energy. Prints one JSON object.
//!
//! rijcosx_grad_check <xyz> [--basis def2-svp] [--method b3lyp] [--charge Q] [--mult M]
//!   [--threads N] [--k-grid cosx:2 | gridx:2,2,2 | linus:1 ...] [--grid orca:2] [--aux A]
//!   [--uhf] [--g-conv X] [--variants] [--exact] [--fd ATOM,ATOM,... | --fd all] [--h 2e-3]
//!   [--no-grid-response]
use linus::two_electron::gradient_grid;
use linus::{CalcOptions, GridChoice, JkMethod, Session};
use linus_grid::MolecularGrid;
use linus_jk::cosx::gradient::{exchange_gradient, CosxGradResult, CosxGradientConfig};
use serde_json::json;
use std::time::Instant;

fn parse_grid(s: &str) -> GridChoice {
    let (k, v) = s.split_once(':').unwrap_or((s, "2"));
    match k {
        "cosx" => GridChoice::Cosx {
            defgrid: v.parse().unwrap(),
        },
        "gridx" => {
            let l: Vec<u8> = v.split(',').map(|x| x.parse().unwrap()).collect();
            GridChoice::CosxLevels {
                levels: if l.len() == 1 {
                    [l[0]; 3]
                } else {
                    [l[0], l[1], l[2]]
                },
            }
        }
        "linus" => GridChoice::Linus {
            level: v.parse().unwrap(),
        },
        "pyscf" => GridChoice::Pyscf {
            level: v.parse().unwrap(),
        },
        "orca" => GridChoice::Orca {
            defgrid: v.parse().unwrap(),
        },
        _ => panic!("grid {s}"),
    }
}

fn maxabs(g: &[[f64; 3]]) -> f64 {
    g.iter()
        .flat_map(|x| x.iter())
        .fold(0.0f64, |m, v| m.max(v.abs()))
}

fn diff(a: &[[f64; 3]], b: &[[f64; 3]]) -> Vec<[f64; 3]> {
    a.iter()
        .zip(b)
        .map(|(x, y)| [x[0] - y[0], x[1] - y[1], x[2] - y[2]])
        .collect()
}

fn sum(parts: &[&[[f64; 3]]]) -> Vec<[f64; 3]> {
    let mut o = vec![[0.0; 3]; parts[0].len()];
    for p in parts {
        for (x, y) in o.iter_mut().zip(p.iter()) {
            for k in 0..3 {
                x[k] += y[k];
            }
        }
    }
    o
}

#[allow(clippy::needless_range_loop, reason = "indexes several arrays in step")]
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let get = |k: &str| a.iter().position(|x| x == k).map(|i| a[i + 1].clone());
    let has = |k: &str| a.iter().any(|x| x == k);
    let dfit = has("--dfit");
    let mult: u32 = get("--mult").map(|s| s.parse().unwrap()).unwrap_or(1);
    let charge: i32 = get("--charge").map(|s| s.parse().unwrap()).unwrap_or(0);
    let mol =
        linus::Structure::from_xyz_str(&std::fs::read_to_string(&a[1]).unwrap(), charge, mult)
            .unwrap();
    let mut o = CalcOptions {
        basis: Some(get("--basis").unwrap_or("def2-svp".into())),
        method: Some(get("--method").unwrap_or("b3lyp".into())),
        threads: Some(get("--threads").map(|s| s.parse().unwrap()).unwrap_or(8)),
        grid: linus::driver::options::GridSettings {
            xc: Some(parse_grid(&get("--grid").unwrap_or("orca:2".into()))),
            ..Default::default()
        },
        ..Default::default()
    };
    let k_grid = parse_grid(&get("--k-grid").unwrap_or("cosx:2".into()));
    o.jk = JkMethod::RijSnk {
        aux: get("--aux").unwrap_or("def2-universal-jfit".into()).into(),
        k_grid: k_grid.clone(),
    }
    .into();
    o.scf.e_tol = Some(linus::driver::options::Auto::Value(
        get("--e-conv").map(|s| s.parse().unwrap()).unwrap_or(1e-11),
    ));
    o.scf.g_tol = Some(linus::driver::options::Auto::Value(
        get("--g-conv").map(|s| s.parse().unwrap()).unwrap_or(1e-9),
    ));
    o.grid.response = Some(!has("--no-grid-response"));
    if has("--uhf") {
        o.reference = Some(linus::driver::options::Reference::Unrestricted);
    }
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(o.threads())
        .build()
        .unwrap();
    let par = if o.threads() > 1 {
        faer::Par::rayon(o.threads())
    } else {
        faer::Par::Seq
    };
    let mut calc = Session::new(mol.clone(), o.clone()).unwrap();
    let t = Instant::now();
    let g = calc.gradient().unwrap();
    let t_grad = t.elapsed().as_secs_f64();
    let mut out = json!({
        "energy": g.energy, "gradient": g.gradient, "time": t_grad, "timers": g.timers.to_json(), "scf_timers": calc.timers().to_json(),
        "options": serde_json::to_value(&o).unwrap(),
    });
    let r = calc.last().cloned().unwrap();
    if has("--variants") {
        let a_x = calc
            .method()
            .functional
            .as_ref()
            .map_or(1.0, |f| f.functional().exx_fraction());
        let (dens, kappa) = if r.restricted {
            (vec![r.d_a.clone()], vec![-a_x])
        } else {
            (vec![r.d_a.clone(), r.d_b.clone()], vec![-0.5 * a_x; 2])
        };
        let run =
            |spec: &linus_grid::GridSpec, fitting: bool, weights: bool| -> (CosxGradResult, f64) {
                let t = Instant::now();
                let mg = MolecularGrid::build(&mol.z, &mol.positions, spec).unwrap();
                let opts = CosxGradientConfig {
                    fitting,
                    weight_derivs: weights,
                    point_motion: weights,
                    deriv_fit: fitting && dfit,
                    ..Default::default()
                };
                let res = pool.install(|| {
                    exchange_gradient(calc.basis(), &mg, &dens, &kappa, &opts, par).unwrap()
                });
                (res, t.elapsed().as_secs_f64())
            };
        let (spec, fitting) = gradient_grid(&k_grid);
        // the COSX gradient as included in g (same options), then all parts on the final grid
        let (def, _) = run(&spec, fitting, o.grid.response());
        let base = diff(&g.gradient, &def.gradient);
        let (full, t_full) = run(&spec, fitting, true);
        let (_, t_now) = run(&spec, fitting, false);
        let p = &full.parts;
        let all = sum(&[
            &base,
            &p.centres,
            &p.ao,
            &p.fit_s,
            &p.fit_snum,
            &p.points,
            &p.weights,
        ]);
        let mut variants = serde_json::Map::new();
        variants.insert("full".into(), json!(all));
        if dfit && fitting {
            // ORCA-style: centres + grid-side derivative integrals with fitted derivative
            // functions; no fitting-matrix or grid-response terms; and with the net force removed
            let d = sum(&[&base, &p.centres, &p.ao_dfit]);
            let n = d.len() as f64;
            let mut mean = [0.0; 3];
            for x in &p
                .centres
                .iter()
                .zip(&p.ao_dfit)
                .map(|(a, b)| [a[0] + b[0], a[1] + b[1], a[2] + b[2]])
                .collect::<Vec<_>>()
            {
                for k in 0..3 {
                    mean[k] += x[k] / n;
                }
            }
            let dp: Vec<[f64; 3]> = d
                .iter()
                .map(|x| [x[0] - mean[0], x[1] - mean[1], x[2] - mean[2]])
                .collect();
            variants.insert("dfit".into(), json!(d));
            variants.insert("dfit_proj".into(), json!(dp));
            // fitted derivative functions plus the grid response of the consistent scheme
            variants.insert(
                "dfit_grid".into(),
                json!(sum(&[&base, &p.centres, &p.ao_dfit, &p.points, &p.weights])),
            );
        }
        variants.insert("no_weights".into(), json!(diff(&all, &p.weights)));
        variants.insert(
            "fixed_grid".into(),
            json!(diff(&diff(&all, &p.weights), &p.points)),
        );
        variants.insert(
            "no_fit_terms".into(),
            json!(diff(&diff(&diff(&all, &p.weights), &p.fit_s), &p.fit_snum)),
        );
        // the last SCF grid instead of the final one
        if let Some(levels) = k_grid.cosx_levels() {
            let (r2, _) = run(&linus_grid::GridSpec::cosx(levels[1]), true, true);
            let all2 = sum(&[
                &base,
                &r2.parts.centres,
                &r2.parts.ao,
                &r2.parts.fit_s,
                &r2.parts.fit_snum,
                &r2.parts.points,
                &r2.parts.weights,
            ]);
            variants.insert("scf_grid_full".into(), json!(all2));
            variants.insert(
                "scf_grid_no_weights".into(),
                json!(diff(&all2, &r2.parts.weights)),
            );
            // no fitting at all on the final grid
            let (r3, _) = run(&spec, false, true);
            let all3 = sum(&[
                &base,
                &r3.parts.centres,
                &r3.parts.ao,
                &r3.parts.points,
                &r3.parts.weights,
            ]);
            variants.insert("unfitted_full".into(), json!(all3));
        }
        out["cosx"] = json!({
            "energy": full.energy, "npoints": full.npoints, "time_with_weights": t_full, "time_without_weights": t_now,
            "pair_batches": full.pair_batches, "prim_points": full.prim_points, "timers": full.timers.to_json(),
            "maxabs": {"centres": maxabs(&p.centres), "ao": maxabs(&p.ao), "points": maxabs(&p.points), "weights": maxabs(&p.weights),
                        "points_plus_weights": maxabs(&sum(&[&p.points, &p.weights])), "fit_s": maxabs(&p.fit_s), "fit_snum": maxabs(&p.fit_snum),
                        "fit_total": maxabs(&sum(&[&p.fit_s, &p.fit_snum]))},
            "parts": {"centres": p.centres, "ao": p.ao, "points": p.points, "weights": p.weights, "fit_s": p.fit_s, "fit_snum": p.fit_snum},
        });
        out["variants"] = serde_json::Value::Object(variants);
        if has("--thresh-scan") {
            // COSX gradient screening: error against thresh 1e-14 and time
            let mg = MolecularGrid::build(&mol.z, &mol.positions, &spec).unwrap();
            let run_t = |thresh: f64, dist: bool| {
                let opts = CosxGradientConfig {
                    fitting,
                    weight_derivs: o.grid.response(),
                    point_motion: o.grid.response(),
                    thresh,
                    thresh_d: None,
                    dist_screen: dist,
                    ..Default::default()
                };
                let t = Instant::now();
                let r = pool.install(|| {
                    exchange_gradient(calc.basis(), &mg, &dens, &kappa, &opts, par).unwrap()
                });
                (r, t.elapsed().as_secs_f64())
            };
            let (rref, tref) = run_t(1e-14, false);
            let mut scan = vec![
                json!({"thresh": 1e-14, "dist": false, "time": tref, "max_diff": 0.0, "pair_batches": rref.pair_batches, "prim_points": rref.prim_points}),
            ];
            for &th in &[1e-13, 1e-12, 1e-11, 1e-10, 1e-9, 1e-8] {
                let (r, t) = run_t(th, true);
                scan.push(json!({"thresh": th, "dist": true, "time": t, "max_diff": maxabs(&diff(&r.gradient, &rref.gradient)), "pair_batches": r.pair_batches, "prim_points": r.prim_points}));
            }
            out["thresh_scan"] = json!(scan);
        }
    }
    if has("--exact") {
        let mut oe = o.clone();
        oe.jk = JkMethod::Exact.into();
        let mut ce = Session::new(mol.clone(), oe).unwrap();
        if let Some(r) = calc.last() {
            ce.set_guess_density(r.d_a.clone(), r.d_b.clone()).unwrap();
        }
        let t = Instant::now();
        let ge = ce.gradient().unwrap();
        out["exact"] = json!({"energy": ge.energy, "gradient": ge.gradient, "time": t.elapsed().as_secs_f64()});
        let mut errs = serde_json::Map::new();
        errs.insert(
            "default".into(),
            json!(maxabs(&diff(&g.gradient, &ge.gradient))),
        );
        if let Some(v) = out.get("variants").and_then(|v| v.as_object()).cloned() {
            for (k, val) in v {
                let gv: Vec<[f64; 3]> = serde_json::from_value(val).unwrap();
                errs.insert(k, json!(maxabs(&diff(&gv, &ge.gradient))));
            }
        }
        out["max_err_vs_exact"] = serde_json::Value::Object(errs);
    }
    if let Some(sel) = get("--fd") {
        let atoms: Vec<usize> = if sel == "all" {
            (0..mol.natom()).collect()
        } else {
            sel.split(',').map(|x| x.parse().unwrap()).collect()
        };
        let h: f64 = get("--h").map(|s| s.parse().unwrap()).unwrap_or(2e-3);
        let mut fd = vec![];
        let mut worst: f64 = 0.0;
        for &at in &atoms {
            let mut row = [0.0; 3];
            for k in 0..3 {
                let e = |s: f64| {
                    let mut m2 = mol.clone();
                    m2.positions[at][k] += s;
                    let mut c2 = Session::new(m2, o.clone()).unwrap();
                    c2.set_guess_density(r.d_a.clone(), r.d_b.clone()).unwrap();
                    c2.energy_for(linus::Purpose::Gradient).unwrap().energy
                };
                row[k] = (8.0 * (e(h) - e(-h)) - (e(2.0 * h) - e(-2.0 * h))) / (12.0 * h);
                worst = worst.max((row[k] - g.gradient[at][k]).abs());
            }
            fd.push(json!({"atom": at, "fd": row, "analytic": g.gradient[at], "diff": [row[0] - g.gradient[at][0], row[1] - g.gradient[at][1], row[2] - g.gradient[at][2]]}));
        }
        out["fd"] = json!({"h": h, "max_abs_diff": worst, "rows": fd});
    }
    println!("{}", out);
}
