//! Analytic Hessians of molecules with effective core potentials combined with density
//! fitting and with open shells, against central differences of analytic gradients (every atom
//! displaced, symmetrized). The exact closed-shell ECP Hessians are validated against PySCF,
//! Psi4 and ORCA in reference/validate.py.

mod common;

use common::{fd_hessian, max_diff, CH2I, HI};
use linus::Structure;
use linus::{CalcOptions, GridChoice, JkMethod, Session};
use std::time::Instant;

fn df() -> JkMethod {
    // explicit: "auto" would fit the Hessian with cc-pVTZ-jkfit and the gradients with
    // def2-universal-jkfit
    JkMethod::Df {
        aux: "def2-universal-jkfit".into(),
    }
}

/// KS runs: first-order grid terms in the Hessian, gradients without grid response on a grid
/// that moves with the atoms (the Hessian is then their exact derivative).
fn opts(method: &str, jk: JkMethod) -> CalcOptions {
    CalcOptions {
        method: Some(method.into()),
        basis: Some("def2-svp".into()),
        jk: jk.into(),
        // an XC grid only for DFT (D41: Hartree-Fock has none)
        grid: linus::driver::options::GridSettings {
            xc: (method != "hf").then_some(GridChoice::Pyscf { level: 1 }),
            ..Default::default()
        },
        threads: Some(4),
        scf: linus::driver::ScfSettings {
            e_tol: Some(linus::driver::options::Auto::Value(1e-11)),
            g_tol: Some(linus::driver::options::Auto::Value(1e-10)),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn check(xyz: &str, charge: i32, mult: u32, method: &str, jk: JkMethod, tol: f64) {
    let t = Instant::now();
    let mol = Structure::from_xyz_str(xyz, charge, mult).unwrap();
    let mut calc = Session::new(mol.clone(), opts(method, jk.clone())).unwrap();
    assert!(
        calc.mol().ecp_core_electrons().iter().any(|&n| n > 0),
        "def2-SVP puts an ECP on I"
    );
    let an = calc.hessian().unwrap().hessian;
    // the Hessian's grid response against gradients without one (one setting): a second session
    let mut o_fd = opts(method, jk.clone());
    if method != "hf" {
        o_fd.grid.response = Some(false);
    }
    let mut calc_fd = Session::new(mol.clone(), o_fd).unwrap();
    let fd = fd_hessian(&mut calc_fd, &mol.positions, 2e-3);
    let e = max_diff(&an, &fd);
    eprintln!(
        "{method} {jk:?} charge {charge} mult {mult}: |H - H_fd| = {e:.2e} ({:.1} s)",
        t.elapsed().as_secs_f64()
    );
    assert!(e < tol, "{method} {jk:?} mult {mult}: |H - H_fd| = {e:e}");
}

/// RI Hessians with an ECP (HI, closed shell): HF and B3LYP. Measured: 5.6e-9, 8.1e-9.
#[test]
fn ecp_ri_hessians_match_finite_differences() {
    check(HI, 0, 1, "hf", df(), 1e-7);
    check(HI, 0, 1, "b3lyp", df(), 1e-7);
}

/// Open-shell Hessians with an ECP (CH2I radical doublet): UHF with exact integrals, UKS
/// (B3LYP) with RI. Measured: 3.1e-8, 8.7e-9.
#[test]
fn ecp_open_shell_hessians_match_finite_differences() {
    check(CH2I, 0, 2, "hf", JkMethod::Exact, 3e-7);
    check(CH2I, 0, 2, "b3lyp", df(), 1e-7);
}
