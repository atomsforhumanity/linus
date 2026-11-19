//! linus:3 (the Hessian and frequency default grid) on X-H bonds beyond the second row
//! (bench/robust/repro/linus3_heavy_hydrides): with SSF partitioning and no size adjustment at
//! all, the cell boundary sat at the bond midpoint and H's coarse outer shells integrated X's
//! valence region (HCl 2.3e-4, HBr 5.4e-4, H2Se 8.7e-4 Eh/bohr from pyscf:8). The pairs of H with
//! an atom beyond neon now keep the Treutler adjustment.

use linus::Structure;
use linus::{CalcOptions, GridChoice, JkMethod, Session};

fn gradient(xyz: &str, grid: GridChoice) -> Vec<[f64; 3]> {
    let mol = Structure::from_xyz_str(xyz, 0, 1).unwrap();
    let o = CalcOptions {
        method: Some("b3lyp".into()),
        basis: Some("def2-svp".into()),
        jk: JkMethod::Df {
            aux: "def2-universal-jkfit".into(),
        }
        .into(),
        grid: linus::driver::options::GridSettings {
            xc: Some(grid),
            ..Default::default()
        },
        threads: Some(4),
        scf: linus::driver::ScfSettings {
            e_tol: Some(linus::driver::options::Auto::Value(1e-11)),
            g_tol: Some(linus::driver::options::Auto::Value(1e-9)),
            ..Default::default()
        },
        ..Default::default()
    };
    Session::new(mol, o).unwrap().gradient().unwrap().gradient
}

/// HCl and HBr, B3LYP/def2-SVP: linus:3 gradients within 2e-5 Eh/bohr of pyscf:8 (measured 1.0e-5
/// and 2.7e-6; 2.3e-4 and 5.4e-4 before).
#[test]
fn linus3_hydrogen_halide_gradients() {
    for (name, xyz) in [
        ("HCl", "2\n\nCl 0 0 0\nH 0 0 1.27\n"),
        ("HBr", "2\n\nBr 0 0 0\nH 0 0 1.41\n"),
    ] {
        let g3 = gradient(xyz, GridChoice::Linus { level: 3 });
        let g8 = gradient(xyz, GridChoice::Pyscf { level: 8 });
        let d = g3
            .iter()
            .zip(&g8)
            .flat_map(|(a, b)| (0..3).map(move |k| (a[k] - b[k]).abs()))
            .fold(0.0, f64::max);
        assert!(
            d < 2e-5,
            "{name}: linus:3 gradient {d:.1e} Eh/bohr from pyscf:8"
        );
    }
}

/// H2S and CH3SH (an S-H pair beside C-H pairs, which keep no adjustment), B3LYP/def2-SVP:
/// linus:3 gradients within 2e-5 Eh/bohr of pyscf:8 (measured 4.6e-6 and 1.5e-5; linus:4 4.7e-6
/// and 2.4e-6).
#[test]
fn linus3_sulfur_hydride_gradients() {
    for (name, xyz) in [
        ("H2S", "3\n\nS 0 0 0\nH 0 0.961 0.928\nH 0 -0.961 0.928\n"),
        ("CH3SH", include_str!("../../../tests/molecules/ch3sh.xyz")),
    ] {
        let g3 = gradient(xyz, GridChoice::Linus { level: 3 });
        let g8 = gradient(xyz, GridChoice::Pyscf { level: 8 });
        let d = g3
            .iter()
            .zip(&g8)
            .flat_map(|(a, b)| (0..3).map(move |k| (a[k] - b[k]).abs()))
            .fold(0.0, f64::max);
        assert!(
            d < 2e-5,
            "{name}: linus:3 gradient {d:.1e} Eh/bohr from pyscf:8"
        );
    }
}
