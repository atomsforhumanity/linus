//! vv10_grids <xyz> <basis> <method> [--charge Q] [--mult M] [--grid G] [--jk df|exact] [--threads N]
//!            [--nlc-grids pyscf:1,orca:1,orcanl:2,linus:1,psi4vv10,...] [--no-gradient] [--fixed-too]
//!
//! One SCF (XC grid `--grid`, default orca:3; NLC grid the first of `--nlc-grids`), then at the
//! converged density the VV10 energy and VV10 gradient (with grid response) on every NLC grid of
//! `--nlc-grids`: the NLC grid error at fixed density (first order in the energy). Prints JSON.
use linus::{CalcOptions, GridChoice, JkMethod, Session};
use linus_grid::{GridSpec, MolecularGrid};

fn spec(s: &str) -> GridSpec {
    let (k, n) = s.split_once(':').unwrap_or((s, "0"));
    let n: u8 = n.parse().unwrap_or(0);
    match k {
        "pyscf" => GridSpec::pyscf(n),
        "orca" => GridSpec::orca_defgrid(n),
        "orcanl" => GridSpec::orca_nl(n),
        "linus" => GridSpec::linus(n),
        "psi4vv10" => GridSpec::psi4_vv10(),
        _ => panic!("unknown grid {s}"),
    }
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let get = |k: &str| a.iter().position(|x| x == k).map(|i| a[i + 1].clone());
    let mult: u32 = get("--mult").map(|s| s.parse().unwrap()).unwrap_or(1);
    let charge: i32 = get("--charge").map(|s| s.parse().unwrap()).unwrap_or(0);
    let text = std::fs::read_to_string(&a[1]).unwrap();
    let mol = linus::Structure::from_xyz_str(&text, charge, mult).unwrap();
    let grids: Vec<String> = get("--nlc-grids")
        .unwrap_or("pyscf:1".into())
        .split(',')
        .map(String::from)
        .collect();
    let mut o = CalcOptions {
        basis: Some(a[2].clone()),
        method: Some(a[3].clone()),
        threads: Some(get("--threads").map(|s| s.parse().unwrap()).unwrap_or(8)),
        grid: linus::driver::options::GridSettings {
            xc: Some(GridChoice::Spec(Box::new(spec(
                &get("--grid").unwrap_or("orca:3".into()),
            )))),
            nlc: Some(GridChoice::Spec(Box::new(spec(&grids[0])))),
            ..Default::default()
        },
        ..Default::default()
    };
    o.jk = match get("--jk").as_deref() {
        Some("exact") => JkMethod::Exact,
        _ => JkMethod::Df {
            aux: "def2-universal-jkfit".into(),
        },
    }
    .into();
    if let Some(c) = get("--conv") {
        o.scf.convergence = Some(linus::driver::options::Auto::Value(c.parse().unwrap()));
    }
    let want_grad = !a.iter().any(|x| x == "--no-gradient");
    let mut calc = Session::new(mol.clone(), o.clone()).unwrap();
    let t = std::time::Instant::now();
    let r = calc.energy_for(linus::Purpose::Gradient).unwrap();
    let t_scf = t.elapsed().as_secs_f64();
    let dt = &r.d_a + &r.d_b;
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(o.threads())
        .build()
        .unwrap();
    let mut t0 = linus_core::timing::Timers::default();
    let base = pool.install(|| calc.build_xc(&mut t0)).unwrap().unwrap();
    let functional = calc.method().functional.clone().unwrap();
    let mut out = vec![];
    for g in &grids {
        let grid = MolecularGrid::build(&mol.z, &mol.positions, &spec(g)).unwrap();
        let npts = grid.len();
        let xc = linus_dft::XcIntegrator::with_functional(
            calc.basis(),
            base.grid.clone(),
            &functional,
            Some(grid),
        );
        let nl = xc.vv10().unwrap();
        let t = std::time::Instant::now();
        let e = pool.install(|| nl.energy(&dt));
        let t_e = t.elapsed().as_secs_f64();
        let t = std::time::Instant::now();
        let grad = if want_grad {
            Some(pool.install(|| nl.gradient(&dt, true)))
        } else {
            None
        };
        let t_g = t.elapsed().as_secs_f64();
        // the fixed-grid (orbital-term only) VV10 gradient, for comparisons
        let grad_fixed = if want_grad && a.iter().any(|x| x == "--fixed-too") {
            Some(pool.install(|| nl.gradient(&dt, false)))
        } else {
            None
        };
        let st = nl.stats();
        eprintln!("{g:10} {npts:9} points {:9} active  E_vv10 {e:.12}  {t_e:.2} s  grad {t_g:.2} s  (pairs {:.2} s, densities {:.2} s, grad terms {:.2} s, weights {:.2} s)",
            st.active_points, st.t_pairs, st.t_density, st.t_grad, st.t_weights);
        out.push(serde_json::json!({"nlc_grid": g, "points": npts, "active_points": st.active_points, "e_vv10": e, "gradient_vv10": grad, "gradient_vv10_fixed_grid": grad_fixed,
            "t_energy": t_e, "t_gradient": t_g, "t_pairs": st.t_pairs, "t_density": st.t_density, "t_grad_terms": st.t_grad, "t_weights": st.t_weights}));
    }
    println!(
        "{}",
        serde_json::json!({"xyz": a[1], "basis": a[2], "method": a[3], "energy": r.energy, "scf_nlc_grid": grids[0],
        "t_scf": t_scf, "timers": calc.timers().to_json(), "grids": out})
    );
}
