//! jk rij_cgk (RI-J + compensated grid-K) through whole calculations: the gradient is the exact
//! derivative of the energy (closed and open shells, a range-separated hybrid), RKS and UKS agree
//! for a closed shell, and the settings that do not apply are input errors.

use crate::driver::{CalcOptions, GridChoice, JkMethod, Session};
use crate::Structure;

const WATER: &str = "3\nwater\nO 0.02 -0.01 0.12\nH 0.03 0.78 -0.47\nH -0.01 -0.74 -0.49\n";

fn calc(
    method: &str,
    charge: i32,
    mult: u32,
    restricted: Option<bool>,
    positions: &[[f64; 3]],
) -> Session {
    let mut mol = Structure::from_xyz_str(WATER, charge, mult).unwrap();
    mol.positions = positions.to_vec();
    let o = CalcOptions {
        method: Some(method.into()),
        basis: Some("def2-svp".into()),
        jk: JkMethod::RijCgk {
            aux: "def2-universal-jkfit".into(),
            k_grid: GridChoice::CosxLevels { levels: [1, 1, 1] },
        }
        .into(),
        grid: crate::driver::options::GridSettings {
            xc: Some(GridChoice::Linus { level: 1 }),
            ..Default::default()
        },
        reference: restricted.map(|r| {
            if r {
                crate::driver::options::Reference::Restricted
            } else {
                crate::driver::options::Reference::Unrestricted
            }
        }),
        threads: Some(2),
        scf: crate::driver::ScfSettings {
            e_tol: Some(crate::driver::options::Auto::Value(1e-12)),
            g_tol: Some(crate::driver::options::Auto::Value(1e-10)),
            ..Default::default()
        },
        ..Default::default()
    };
    Session::new(mol, o).unwrap()
}

#[test]
fn gradients_match_finite_differences() {
    let x0 = Structure::from_xyz_str(WATER, 0, 1).unwrap().positions;
    let g_rks = calc("b3lyp", 0, 1, None, &x0).gradient().unwrap().gradient;
    let g_uks = calc("b3lyp", 0, 1, Some(false), &x0)
        .gradient()
        .unwrap()
        .gradient;
    for (a, b) in g_rks.iter().zip(&g_uks) {
        for k in 0..3 {
            assert!((a[k] - b[k]).abs() < 1e-8, "RKS {} vs UKS {}", a[k], b[k]);
        }
    }
    for (method, charge, mult) in [("b3lyp", 0, 1), ("b3lyp", 1, 2), ("wb97x", 0, 1)] {
        let g = calc(method, charge, mult, None, &x0)
            .gradient()
            .unwrap()
            .gradient;
        for k in 0..3 {
            assert!(
                g.iter().map(|x| x[k]).sum::<f64>().abs() < 1e-9,
                "translational invariance"
            );
        }
        let h = 1e-3;
        for &(at, k) in &[(0usize, 2usize), (1, 1)] {
            let e = |s: f64| {
                let mut x = x0.clone();
                x[at][k] += s;
                calc(method, charge, mult, None, &x)
                    .energy()
                    .unwrap()
                    .energy
            };
            let fd = (8.0 * (e(h) - e(-h)) - (e(2.0 * h) - e(-2.0 * h))) / (12.0 * h);
            assert!(
                (fd - g[at][k]).abs() < 3e-8,
                "{method} charge {charge} atom {at} dir {k}: fd {fd} analytic {}",
                g[at][k]
            );
        }
    }
}

#[test]
fn inapplicable_settings_are_input_errors() {
    use serde_json::json;
    for bad in [
        json!({"jk": {"kind": "rij_cgk", "k_grid": "cosx:2"}}),
        json!({"jk": {"kind": "rij_cgk", "k_grid": "gridx:1,2,3"}}),
    ] {
        assert!(CalcOptions::from_json(bad.clone()).is_err(), "{bad}");
    }
    // D41: a k grid without exact exchange (rij_cgk and rij_snk); backend.df_j where the J/K
    // runs on the CPU
    let mol = Structure::from_xyz_str(WATER, 0, 1).unwrap();
    for (opts, path) in [
        (
            json!({"method": "pbe", "jk": {"kind": "rij_cgk", "k_grid": "gridx:2"}}),
            "jk.k_grid",
        ),
        (
            json!({"method": "pbe", "jk": {"kind": "rij_snk", "k_grid": "cosx:2"}}),
            "jk.k_grid",
        ),
        (
            json!({"method": "pbe", "backend": {"df_j": "stored"}}),
            "backend.df_j",
        ),
    ] {
        match Session::new(mol.clone(), CalcOptions::from_json(opts.clone()).unwrap()) {
            Err(crate::LinusError::Setting { path: p, code, .. }) => {
                assert_eq!((p.as_str(), code), (path, "no_effect"), "{opts}")
            }
            r => panic!("{opts}: {:?}", r.err()),
        }
    }
    let ok =
        CalcOptions::from_json(json!({"jk": {"kind": "rij_cgk", "k_grid": "gridx:2"}})).unwrap();
    assert_eq!(ok.jk_method().name(), "rij_cgk");
    assert_eq!(
        CalcOptions::from_json(json!({"jk": "rij_cgk"}))
            .unwrap()
            .jk_method()
            .k_grid(),
        Some(&crate::driver::options::K_GRID_CGK_DEFAULT)
    );
}

/// The grid-K arrays are in the memory plan: the SCF's reservation on the CPU, the gradient's
/// part, the estimate's term; sized from the grid's points (4 and 11 npts x nbf matrices).
#[test]
fn the_grid_k_arrays_are_in_the_memory_plan() {
    use crate::driver::report::CalcType;
    let x0 = Structure::from_xyz_str(WATER, 0, 1).unwrap().positions;
    let plan = calc("b3lyp", 0, 1, None, &x0)
        .plan(CalcType::Gradient)
        .unwrap();
    let text = serde_json::to_string(&plan).unwrap();
    for part in [
        "reserved (grid-K)",
        "grid-K gradient arrays",
        "+ grid-K arrays",
    ] {
        assert!(text.contains(part), "{part}: {text}");
    }
    // a pure functional has no grid-K
    let o = CalcOptions::from_json(serde_json::json!({"method": "pbe", "basis": "def2-svp",
        "jk": "rij_cgk"}))
    .unwrap();
    let mol = Structure::from_xyz_str(WATER, 0, 1).unwrap();
    let pbe = Session::new(mol, o)
        .unwrap()
        .plan(CalcType::Gradient)
        .unwrap();
    let pbe = serde_json::to_string(&pbe).unwrap();
    assert!(!pbe.contains("grid-K"), "{pbe}");
}
