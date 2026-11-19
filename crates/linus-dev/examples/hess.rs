//! hess <xyz> <basis> [--method M] [--threads N] [--jk exact|df] [--aux A] [--grid L]
//! [--orca-grid G] [--fixed-grid] [--disp D] [--e-conv E] [--g-conv G] [--fd-grad H]
//! [--fd-atoms a,b] [--charge Q] [--mult M] [--uhf]: HF/KS analytic Hessian as JSON
//! (restricted for closed shells, unrestricted otherwise or with --uhf). --grid: PySCF grid
//! level (default 3), linus:N or auto; --orca-grid: ORCA DefGrid instead. --fixed-grid: XC terms on a fixed grid
//! (PySCF's convention) instead of the default first-order grid response (--hess-grid-response,
//! accepted for old scripts). --fd-grad adds the
//! Richardson-extrapolated central-difference Hessian of the analytic gradient (steps H and 2H
//! bohr, SCF converged to 1e-12 in the orbital gradient) and its max deviation from the analytic
//! Hessian (`fd_grad_max_err`, against the symmetrized differences where both orders were
//! differenced; `fd_grad_asymmetry`: the differences' own asymmetry, which for KS with the
//! default convention, gradients without grid response, is a property of that convention and not
//! an error; --fd-full-grid-response: gradients with grid response, the exact Hessian).
use linus::{CalcOptions, GridChoice, JkMethod, Session};
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let get = |k: &str| a.iter().position(|x| x == k).map(|i| a[i + 1].clone());
    let charge: i32 = get("--charge").map(|s| s.parse().unwrap()).unwrap_or(0);
    let mult: u32 = get("--mult").map(|s| s.parse().unwrap()).unwrap_or(1);
    let mol =
        linus::Structure::from_xyz_str(&std::fs::read_to_string(&a[1]).unwrap(), charge, mult)
            .unwrap();
    let mut o = CalcOptions::default();
    if a.iter().any(|x| x == "--uhf") {
        o.reference = Some(linus::driver::options::Reference::Unrestricted);
    }
    o.basis = Some(a[2].clone());
    o.method = Some(get("--method").unwrap_or("hf".into()));
    o.threads = Some(get("--threads").map(|s| s.parse().unwrap()).unwrap_or(16));
    o.scf.e_tol = Some(linus::driver::options::Auto::Value(
        get("--e-conv").map(|s| s.parse().unwrap()).unwrap_or(1e-11),
    ));
    o.scf.g_tol = Some(linus::driver::options::Auto::Value(
        get("--g-conv").map(|s| s.parse().unwrap()).unwrap_or(1e-9),
    ));
    o.scf.max_iter = Some(200);
    // --grid: a PySCF level (default 3), linus:N or auto (the Hessian default)
    o.grid.xc = Some(match (get("--orca-grid"), get("--grid")) {
        (Some(g), _) => GridChoice::Orca {
            defgrid: g.parse().unwrap(),
        },
        (None, Some(g)) if g == "auto" => GridChoice::Auto,
        (None, Some(g)) if g.starts_with("linus:") => GridChoice::Linus {
            level: g[6..].parse().unwrap(),
        },
        (None, g) => GridChoice::Pyscf {
            level: g.map(|s| s.parse().unwrap()).unwrap_or(3),
        },
    });
    o.grid.response = Some(!a.iter().any(|x| x == "--fixed-grid"));
    if o.method() == "hf" {
        // Hartree-Fock has no XC grid (D41: its grid settings would be input errors)
        o.grid = Default::default();
    }
    if let Some(d) = get("--disp") {
        o.dispersion = Some(linus::driver::DispersionSetting::parse(&d).unwrap());
    }
    // --jk exact|df; without it, the library's default (df since numerics epoch 6)
    match get("--jk").as_deref() {
        None => {}
        Some("exact") => o.jk = JkMethod::Exact.into(),
        Some("df") => {
            o.jk = JkMethod::Df {
                aux: get("--aux").unwrap_or("def2-universal-jkfit".into()).into(),
            }
            .into()
        }
        Some(other) => panic!("--jk {other}: exact or df"),
    }
    let mut calc = Session::new(mol.clone(), o.clone()).unwrap();
    let t = std::time::Instant::now();
    let h = calc.hessian().unwrap();
    let time = t.elapsed().as_secs_f64();
    let mut out = serde_json::json!({"energy": h.energy, "hessian": h.hessian, "cphf_iterations": h.cphf_iterations, "time": time, "timers": h.timers.to_json(),
        "scf_builds": calc.last().map(|r| r.iterations), "scf_gmax": calc.last().and_then(|r| r.history.last().map(|x| x.gmax))});
    if let Some(step) = get("--fd-grad").map(|s| s.parse::<f64>().unwrap()) {
        let n3 = 3 * mol.natom();
        let mut fd = vec![0.0; n3 * n3];
        let mut o2 = o.clone();
        if o.method() != "hf" {
            o2.grid.xc = Some(o.grid.xc().for_purpose(linus::Purpose::Hessian));
        }
        o2.scf.e_tol = Some(linus::driver::options::Auto::Value(1e-13));
        o2.scf.g_tol = Some(linus::driver::options::Auto::Value(1e-12));
        o2.scf.max_iter = Some(200);
        // the default Hessian (first-order grid response) is the derivative of the gradient
        // without grid response on a grid that moves with the atoms; --fd-full-grid-response:
        // differences of full-grid-response gradients instead (the energy's exact Hessian)
        if o.method() != "hf" {
            o2.grid.response =
                Some(!o.grid.response() || a.iter().any(|x| x == "--fd-full-grid-response"));
        }
        // --fd-atoms a,b,...: only these columns (others compared as zero-skipped)
        let atoms: Vec<usize> = get("--fd-atoms")
            .map(|s| s.split(',').map(|x| x.parse().unwrap()).collect())
            .unwrap_or((0..mol.natom()).collect());
        let mut cols = Vec::new();
        for &at in &atoms {
            for k in 0..3 {
                cols.push(3 * at + k);
                let mut g = Vec::new();
                for s in [1.0, -1.0, 2.0, -2.0] {
                    let mut m2 = mol.clone();
                    m2.positions[at][k] += s * step;
                    let mut c2 = Session::new(m2, o2.clone()).unwrap();
                    g.push(c2.gradient().unwrap().gradient);
                }
                for b in 0..mol.natom() {
                    for l in 0..3 {
                        fd[(3 * b + l) * n3 + 3 * at + k] = (8.0 * (g[0][b][l] - g[1][b][l])
                            - (g[2][b][l] - g[3][b][l]))
                            / (12.0 * step);
                    }
                }
            }
        }
        // the differences of gradients without grid response (KS default) are not symmetric:
        // that gradient is not an energy derivative, and the analytic Hessian is the symmetrized
        // derivative. Compare with the symmetrized differences where both (i, j) and (j, i) were
        // differenced, and report the asymmetry (a property of the convention, not an error)
        let sym = |i: usize, j: usize| {
            if cols.contains(&i) {
                0.5 * (fd[i * n3 + j] + fd[j * n3 + i])
            } else {
                fd[i * n3 + j]
            }
        };
        let err = (0..n3)
            .flat_map(|i| cols.iter().map(move |&j| (i, j)))
            .map(|(i, j)| (sym(i, j) - h.hessian[i * n3 + j]).abs())
            .fold(0.0f64, f64::max);
        let asym = cols
            .iter()
            .flat_map(|&i| cols.iter().map(move |&j| (i, j)))
            .map(|(i, j)| (fd[i * n3 + j] - fd[j * n3 + i]).abs())
            .fold(0.0f64, f64::max);
        out["fd_grad_max_err"] = serde_json::json!(err);
        out["fd_grad_asymmetry"] = serde_json::json!(asym);
        out["fd_hessian"] = serde_json::json!(fd);
        out["fd_cols"] = serde_json::json!(cols);
    }
    println!("{}", out);
}
