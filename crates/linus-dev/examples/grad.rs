//! grad <xyz> <basis> <method> [--jk exact|df] [--aux A] [--mult M] [--charge Q] [--threads N] [--grid G] [--nlc-grid pyscf:L|orca:N|JSON]
//!      [--no-grid-response] [--fd]
//! Prints energy, gradient, the gradient's parts (nuc, one_e, two_e, xc, disp), e_disp and (VV10 functionals) e_vv10 as JSON.
use linus::{CalcOptions, GridChoice, JkMethod, Session};
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let get = |k: &str| a.iter().position(|x| x == k).map(|i| a[i + 1].clone());
    let mult: u32 = get("--mult").map(|s| s.parse().unwrap()).unwrap_or(1);
    let charge: i32 = get("--charge").map(|s| s.parse().unwrap()).unwrap_or(0);
    let mol =
        linus::Structure::from_xyz_str(&std::fs::read_to_string(&a[1]).unwrap(), charge, mult)
            .unwrap();
    let mut o = CalcOptions {
        basis: Some(a[2].clone()),
        method: Some(a[3].clone()),
        threads: Some(get("--threads").map(|s| s.parse().unwrap()).unwrap_or(16)),
        ..Default::default()
    };
    o.scf.e_tol = Some(linus::driver::options::Auto::Value(1e-11));
    o.scf.g_tol = Some(linus::driver::options::Auto::Value(
        get("--g-conv").map(|s| s.parse().unwrap()).unwrap_or(1e-9),
    ));
    o.grid.xc = Some(GridChoice::Pyscf {
        level: get("--grid").map(|s| s.parse().unwrap()).unwrap_or(3),
    });
    // --scf JSON: SCF algorithm settings (e.g. {"incremental": false})
    if let Some(s) = get("--scf") {
        o.scf = serde_json::from_str(&s).unwrap();
    }
    // --nlc-grid pyscf:L | orca:N (ORCA's NL grid of DefGrid N) | a GridChoice as JSON
    if let Some(s) = get("--nlc-grid") {
        o.grid.nlc = Some(match s.split_once(':') {
            Some(("pyscf", l)) => GridChoice::Pyscf {
                level: l.parse().unwrap(),
            },
            Some(("orca", n)) => linus::driver::options::nlc_grid_for(
                &GridChoice::Auto,
                &GridChoice::Orca {
                    defgrid: n.parse().unwrap(),
                },
            ),
            _ => serde_json::from_str(&s).unwrap(),
        });
    }
    if o.method() == "hf" {
        // Hartree-Fock has no XC grid (D41: its grid settings would be input errors)
        o.grid = Default::default();
    }
    if a.iter().any(|x| x == "--no-grid-response") && o.method() != "hf" {
        o.grid.response = Some(false);
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
    let g = calc.gradient().unwrap();
    let tg = t.elapsed().as_secs_f64();
    let e_disp = calc.last().map(|r| r.e_disp).unwrap_or(0.0);
    let mut out = serde_json::json!({"energy": g.energy, "gradient": g.gradient, "time": tg, "timers": g.timers.to_json(),
        "e_disp": e_disp, "e_vv10": calc.vv10().and_then(|v| v.last_energy), "parts": {"nuc": g.parts.nuclear_repulsion, "one_e": g.parts.one_electron, "two_e": g.parts.two_electron, "xc": g.parts.xc, "disp": g.parts.dispersion}});
    if a.iter().any(|x| x == "--fd") {
        let h = 1e-4;
        let mut fd = vec![[0.0f64; 3]; mol.natom()];
        for i in 0..mol.natom() {
            for k in 0..3 {
                let mut e = [0.0; 2];
                for (s, sg) in [1.0, -1.0].iter().enumerate() {
                    let mut c = mol.positions.clone();
                    c[i][k] += sg * h;
                    let mut m2 = mol.clone();
                    m2.positions = c;
                    let mut cc = Session::new(m2, o.clone()).unwrap();
                    e[s] = cc.energy().unwrap().energy;
                }
                fd[i][k] = (e[0] - e[1]) / (2.0 * h);
            }
        }
        let err = fd
            .iter()
            .zip(&g.gradient)
            .flat_map(|(x, y)| (0..3).map(move |k| (x[k] - y[k]).abs()))
            .fold(0.0f64, f64::max);
        out["fd_max_err"] = serde_json::json!(err);
    }
    println!("{}", out);
}
