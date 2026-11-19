//! Thread-count determinism guard (CONTRIBUTING.md, "Determinism"): no calculation reads
//! faer's global parallelism. faer's matrix operators (`&a * &b`), convenience solvers
//! (`Solve::solve`) and factorizations (`Mat::llt`, `Mat::svd`) run with it, and it is the size
//! of the current thread pool, so a matrix-vector product written with them rounds differently
//! at different thread counts. clippy.toml forbids the methods; operators are out of clippy's
//! reach, so these calculations run with the global parallelism disabled, and any such call on
//! their paths panics. Covered: RKS and UKS, exact, density-fitted and RIJ-COSX J/K (with
//! overlap fitting), a range-separated hybrid, Hessians with their CPHF/CPKS, dipole
//! derivatives and polarizabilities, the harmonic analysis with its symmetry detection, and a
//! PCM. This file is its own test binary (process), so the global setting reaches no other test.

use linus::solvent::SolventSettings;
use linus::{CalcOptions, CalcType, GridChoice, JkMethod, Session, Structure};
use linus_solvent::PcmModel;

/// Water without symmetry (Å).
const WATER: &str = "3\n\nO 0.000 0.000 0.118\nH 0.020 0.755 -0.471\nH -0.010 -0.760 -0.465\n";

fn run(charge: i32, mult: u32, method: &str, jk: JkMethod, calc: CalcType, ir: bool, pcm: bool) {
    let mol = Structure::from_xyz_str(WATER, charge, mult).unwrap();
    let freq = ir.then(|| {
        CalcOptions::from_json(serde_json::json!({"freq": {"ir": true}}))
            .unwrap()
            .freq
    });
    let opts = CalcOptions {
        freq: freq.unwrap_or_default(),
        method: Some(method.into()),
        basis: Some("def2-svp".into()),
        jk: jk.into(),
        // an XC grid only for DFT (D41: Hartree-Fock has none)
        grid: linus::driver::options::GridSettings {
            xc: (method != "hf").then_some(GridChoice::Pyscf { level: 1 }),
            ..Default::default()
        },
        threads: Some(4),
        solvent: pcm.then(|| SolventSettings {
            eps: Some(Some(78.4)),
            model: Some(PcmModel::Cpcm),
            lebedev: Some(linus::driver::options::Auto::Value(110)),
            ..Default::default()
        }),
        ..Default::default()
    };
    let report = Session::new(mol, opts).unwrap().run(calc).unwrap();
    assert!(report.energy.is_finite());
}

#[test]
fn calculations_do_not_read_faer_global_parallelism() {
    faer::disable_global_parallelism();
    let df = || JkMethod::Df {
        aux: "def2-universal-jkfit".into(),
    };
    let ir = || CalcType::Hessian;
    run(0, 1, "b3lyp", df(), ir(), true, false);
    run(
        1,
        2,
        "wb97x",
        JkMethod::Exact,
        CalcType::Hessian,
        false,
        false,
    );
    run(1, 2, "b3lyp", df(), ir(), true, false);
    let rij_snk = JkMethod::RijSnk {
        aux: "def2-universal-jfit".into(),
        k_grid: linus::driver::options::K_GRID_DEFAULT,
    };
    run(0, 1, "b3lyp", rij_snk, CalcType::Gradient, false, false);
    run(0, 1, "hf", df(), ir(), true, true);
}

/// The guard is live: with the global parallelism disabled, a matrix-vector product written
/// with faer's operator panics (a faer version that stopped doing so would void the test above).
#[test]
#[should_panic(expected = "Global parallelism is disabled")]
fn operators_panic_with_global_parallelism_disabled() {
    faer::disable_global_parallelism();
    let a = faer::Mat::<f64>::identity(300, 300);
    let x = faer::Mat::<f64>::zeros(300, 1);
    let _ = &a * &x;
}
