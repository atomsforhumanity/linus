//! linus-jk's RIJCOSX gradient against whole calculations (linus_jk::cosx::rij_cosx's tests
//! that run the Session).

use crate::driver::{CalcOptions, GridChoice, JkMethod, Session};
use crate::Structure;

const WATER: &str = "3\nwater\nO 0.02 -0.01 0.12\nH 0.03 0.78 -0.47\nH -0.01 -0.74 -0.49\n";

fn calc(charge: i32, mult: u32, restricted: Option<bool>, positions: &[[f64; 3]]) -> Session {
    let mut mol = Structure::from_xyz_str(WATER, charge, mult).unwrap();
    mol.positions = positions.to_vec();
    let o = CalcOptions {
        method: Some("b3lyp".into()),
        basis: Some("def2-svp".into()),
        jk: JkMethod::RijSnk {
            aux: "def2-universal-jfit".into(),
            k_grid: GridChoice::Linus { level: 1 },
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

/// One grid without fitting: the RIJCOSX energy is variational and the gradient (grid
/// response on) its exact derivative; RKS and UKS agree for a closed shell.
#[test]
fn rks_uks_gradients_match_finite_differences() {
    let x0 = Structure::from_xyz_str(WATER, 0, 1).unwrap().positions;
    let g_rks = calc(0, 1, None, &x0).gradient().unwrap().gradient;
    let g_uks = calc(0, 1, Some(false), &x0).gradient().unwrap().gradient;
    for (a, b) in g_rks.iter().zip(&g_uks) {
        for k in 0..3 {
            assert!((a[k] - b[k]).abs() < 1e-8, "RKS {} vs UKS {}", a[k], b[k]);
        }
    }
    let g_cation = calc(1, 2, None, &x0).gradient().unwrap().gradient;
    for (charge, mult, g) in [(0, 1, &g_rks), (1, 2, &g_cation)] {
        for k in 0..3 {
            assert!(
                g.iter().map(|x| x[k]).sum::<f64>().abs() < 1e-9,
                "translational invariance"
            );
        }
        let h = 1e-3;
        for &(at, k) in &[(0usize, 2usize), (1, 1), (2, 0)] {
            let e = |s: f64| {
                let mut x = x0.clone();
                x[at][k] += s;
                calc(charge, mult, None, &x).energy().unwrap().energy
            };
            let fd = (8.0 * (e(h) - e(-h)) - (e(2.0 * h) - e(-2.0 * h))) / (12.0 * h);
            assert!(
                (fd - g[at][k]).abs() < 3e-8,
                "charge {charge} atom {at} dir {k}: fd {fd} analytic {}",
                g[at][k]
            );
        }
    }
}
