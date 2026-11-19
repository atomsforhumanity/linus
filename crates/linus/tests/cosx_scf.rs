//! RIJ+COSX SCFs on the ORCA-like grid sequence (GridX1, then GridX2; final energy on GridX3):
//! the SCF moves to the last SCF grid before it converges (`JkBuilder::next_approximation`),
//! reaches Hessian-level thresholds (max |F_ai| < 1e-9) without stalling, ends where an SCF on
//! the last grid alone ends, and does not depend on the number of threads.

use linus::ScfResult;
use linus::Structure;
use linus::{CalcOptions, GridChoice, JkMethod, Session};

const CH3SH: &str = include_str!("../../../tests/molecules/ch3sh.xyz");

fn run(
    method: &str,
    charge: i32,
    mult: u32,
    levels: Option<[u8; 3]>,
    threads: usize,
    g_tol: f64,
) -> ScfResult {
    let mol = Structure::from_xyz_str(CH3SH, charge, mult).unwrap();
    let k_grid = match levels {
        Some(levels) => GridChoice::CosxLevels { levels },
        None => GridChoice::Cosx { defgrid: 2 },
    };
    let o = CalcOptions {
        method: Some(method.into()),
        basis: Some("def2-svp".into()),
        jk: JkMethod::RijSnk {
            aux: "def2-universal-jfit".into(),
            k_grid,
        }
        .into(),
        grid: linus::driver::options::GridSettings {
            xc: Some(GridChoice::Pyscf { level: 3 }),
            ..Default::default()
        },
        threads: Some(threads),
        scf: linus::driver::ScfSettings {
            e_tol: Some(linus::driver::options::Auto::Value(1e-11)),
            g_tol: Some(linus::driver::options::Auto::Value(g_tol)),
            ..Default::default()
        },
        ..Default::default()
    };
    Session::new(mol, o).unwrap().energy().unwrap()
}

/// B3LYP (RKS) and wB97X-D3BJ (UKS doublet cation): the grid switch happens before convergence,
/// the SCF reaches max |F_ai| < 1e-9 and the energy equals that of an SCF on GridX2 alone.
#[test]
fn grid_sequence_converges_tightly_to_the_last_grid_solution() {
    for (method, charge, mult) in [("b3lyp", 0, 1), ("wb97x-d3bj", 1, 2)] {
        let r = run(method, charge, mult, None, 2, 1e-9);
        let h = &r.history;
        let k = h
            .iter()
            .position(|it| it.approx == 1)
            .unwrap_or_else(|| panic!("{method}: no switch to the last SCF grid"));
        assert!(
            k > 0 && k + 2 < h.len(),
            "{method}: switch at iteration {} of {}",
            k + 1,
            h.len()
        );
        let last = h.last().unwrap();
        assert!(
            last.gmax < 1e-9 && last.approx == 1,
            "{method}: final max|F_ai| {:e}",
            last.gmax
        );
        assert!(r.iterations <= 25, "{method}: {} Fock builds", r.iterations);
        // COSX builds K incrementally (its own difference densities), and the SCF still ends
        // on a full build (decision 25's one-build rule): an incremental converged iterate is
        // confirmed by a full K at the same density
        assert!(h.iter().any(|it| it.fock == "incremental"), "{method}");
        assert_eq!(last.fock, "full", "{method}");
        assert_eq!(
            r.incremental_builds,
            h.iter().filter(|it| it.fock == "incremental").count()
        );
        let one = run(method, charge, mult, Some([2, 2, 3]), 2, 1e-9);
        assert!(
            (r.energy - one.energy).abs() < 1e-10,
            "{method}: {} vs GridX2 only {}",
            r.energy,
            one.energy
        );
    }
}

/// The SCF (grid switch, DIIS shift, incremental COSX builds) is bitwise independent of the
/// number of threads.
#[test]
fn grid_sequence_is_thread_count_independent() {
    let a = run("b3lyp", 1, 2, None, 1, 1e-8);
    let b = run("b3lyp", 1, 2, None, 4, 1e-8);
    assert_eq!(a.iterations, b.iterations);
    assert_eq!(
        a.energy.to_bits(),
        b.energy.to_bits(),
        "{} vs {}",
        a.energy,
        b.energy
    );
    for (x, y) in a.history.iter().zip(&b.history) {
        assert_eq!(x.gmax.to_bits(), y.gmax.to_bits(), "iteration {}", x.iter);
    }
}
