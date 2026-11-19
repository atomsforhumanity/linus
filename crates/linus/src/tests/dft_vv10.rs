//! linus-dft's VV10 in whole calculations (from linus_dft::vv10's tests: they run the
//! Session).

use crate::driver::options::{nlc_grid_for, CalcOptions, GridChoice, NLC_DEFAULT};
use linus_base::error::LinusError;
use linus_grid::GridSpec;

/// Driver level: water wB97M-V/def2-SVP (exact J/K, PySCF's grid level 3 and NLC grid level 1, set explicitly)
/// against PySCF 2.15 (energy -76.32547957068539, VV10 energy 0.04266857142046653 Eh); the
/// NLC grid choice; responses accepted.
#[test]
fn driver_matches_pyscf_and_accepts_responses() {
    use crate::driver::Session;
    let xyz = "3\nwater\nO 0.0 0.0 0.117790\nH 0.0 0.755453 -0.471161\nH 0.0 -0.755453 -0.471161\n";
    let mol = crate::Structure::from_xyz_str(xyz, 0, 1).unwrap();
    let o = CalcOptions {
        method: Some("wb97m-v".into()),
        basis: Some("def2-svp".into()),
        // PySCF's reference ran exact J/K
        jk: crate::driver::options::JkMethod::Exact.into(),
        grid: crate::driver::options::GridSettings {
            xc: Some(GridChoice::Pyscf { level: 3 }),
            nlc: Some(GridChoice::Pyscf { level: 1 }),
            ..Default::default()
        },
        threads: Some(4),
        scf: crate::driver::ScfSettings {
            convergence: Some(crate::driver::options::Auto::Value(
                crate::driver::ConvergencePreset::VeryTight,
            )),
            ..Default::default()
        },
        ..Default::default()
    };
    let mut calc = Session::new(mol, o).unwrap();
    let r = calc.energy().unwrap();
    let nl = calc.vv10().cloned().unwrap();
    assert!(
        (r.energy - -76.32547957068539).abs() < 1e-9,
        "E {}",
        r.energy
    );
    assert!(
        (nl.last_energy.unwrap() - 0.04266857142046653).abs() < 1e-10,
        "E_vv10 {:?}",
        nl.last_energy
    );
    // B97M-V and the range-separated -V functionals: responses (Hessians, frequencies,
    // polarizabilities) pass the capability checks (crate::xc::vv10_resp)
    use crate::derivatives::capabilities::Request;
    for method in ["b97m-v", "wb97m-v", "wb97x-v"] {
        let mol = crate::Structure::from_xyz_str(xyz, 0, 1).unwrap();
        let calc = Session::new(
            mol,
            CalcOptions {
                method: Some(method.into()),
                basis: Some("sto-3g".into()),
                ..Default::default()
            },
        )
        .unwrap();
        for req in [
            Request::Hessian {
                dipole_derivatives: true,
            },
            Request::Polarizability,
        ] {
            let r: Result<(), LinusError> = calc.check_request(req);
            assert!(r.is_ok(), "{method}: {r:?}");
        }
    }
    // NLC grids: PySCF XC grids -> PySCF's default nlcgrids level (3); ORCA DefGrid3 -> its NL grid (DefGrid1)
    assert_eq!(
        nlc_grid_for(&GridChoice::Auto, &GridChoice::Pyscf { level: 5 }),
        GridChoice::Pyscf { level: 3 }
    );
    assert_eq!(
        nlc_grid_for(&GridChoice::Auto, &GridChoice::Orca { defgrid: 3 }).spec(),
        GridSpec::orca_defgrid(1)
    );
    assert_eq!(
        nlc_grid_for(&GridChoice::Auto, &GridChoice::Auto),
        NLC_DEFAULT
    );
    assert_eq!(
        nlc_grid_for(
            &GridChoice::Linus { level: 2 },
            &GridChoice::Pyscf { level: 3 }
        ),
        GridChoice::Linus { level: 2 }
    );
}
