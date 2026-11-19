//! Open-shell SCF convergence (bench/results/2026-11-07-openshell-scf):
//! - a linear radical with a degenerate pi hole (HBr+ UKS) converges to max |F_ai| < 1e-9, which
//!   needs DIIS to drop its oldest entries once the SCF has left the region of its best ones
//!   (`diis_unpin`; without it DIIS freezes at max |F_ai| ~ 7e-7);
//! - a UHF SCF that converges onto a saddle point (L-lactide radical cation) is detected by its
//!   energy lying above the lowest one visited, follows the negative orbital-Hessian mode and
//!   ends on the minimum below it; the path is thread-count independent.

use linus::ScfResult;
use linus::Structure;
use linus::{CalcOptions, GridChoice, JkMethod, LinusError, Session};

const HBR: &str = include_str!("../../../bench/molecules/hbr_146.xyz");
const LA18: &str = include_str!("../../../bench/molecules/la18.xyz");

fn run(
    xyz: &str,
    charge: i32,
    mult: u32,
    method: &str,
    g_tol: f64,
    threads: usize,
    edit: impl Fn(&mut CalcOptions),
) -> Result<ScfResult, LinusError> {
    let mol = Structure::from_xyz_str(xyz, charge, mult).unwrap();
    let mut o = CalcOptions {
        method: Some(method.into()),
        basis: Some("def2-svp".into()),
        jk: JkMethod::Df {
            aux: "def2-universal-jkfit".into(),
        }
        .into(),
        // an XC grid only for DFT (D41: Hartree-Fock has none)
        grid: linus::driver::options::GridSettings {
            xc: (method != "hf").then_some(GridChoice::Pyscf { level: 3 }),
            ..Default::default()
        },
        threads: Some(threads),
        scf: linus::driver::ScfSettings {
            e_tol: Some(linus::driver::options::Auto::Value(1e-9)),
            g_tol: Some(linus::driver::options::Auto::Value(g_tol)),
            max_iter: Some(100),
            ..Default::default()
        },
        ..Default::default()
    };
    edit(&mut o);
    Session::new(mol, o).unwrap().energy()
}

/// HBr+ (r = 1.46 A, 2Pi) B3LYP UKS to max |F_ai| < 1e-9: converges on the doublet with the
/// hole in one pi component (<S^2> 0.7520); with the best DIIS entries kept for good it does not.
#[test]
fn pi_hole_radical_converges_tightly() {
    let r = run(HBR, 1, 2, "b3lyp", 1e-9, 4, |_| {}).expect("HBr+ converges");
    assert!(r.history.last().unwrap().gmax < 1e-9);
    assert!(r.iterations <= 50, "{} SCF iterations", r.iterations);
    assert!((r.s2 - 0.75203).abs() < 1e-4, "<S^2> {}", r.s2);
    // the default stability probe ran on the converged point and found no negative mode
    assert!(
        r.stability.iter().all(|c| c.probe && c.step == 0.0),
        "{:?}",
        r.stability
    );
    // DIIS alone with its best entries kept for good stalls at a fixed point of the pinned
    // extrapolation, max |F_ai| ~7e-7 at a stationary energy, far above the 1e-9 asked for. The
    // plateau rule would accept that point and the escalation restarts from the lowest iterate
    // with quasi-Newton steps, whose success depends on the path (a last-bit change
    // of the SAD guess made it converge in 83 iterations): both are off, so the claim is the
    // pinned subspace's alone.
    let pinned = run(HBR, 1, 2, "b3lyp", 1e-9, 4, |o| {
        o.scf.max_iter = Some(60);
        o.scf.diis_unpin = Some(0.0);
        o.scf.plateau_window = Some(0);
        o.scf.escalate = Some(false);
    });
    match pinned {
        Err(LinusError::NotConverged(f)) => {
            assert!(f.notes == "scf.escalate is off" && f.gmax > 1e-8, "{f}")
        }
        r => panic!(
            "the pinned DIIS subspace converged: {:?}",
            r.map(|r| r.energy)
        ),
    }
}

/// L-lactide radical cation, UHF/def2-SVP: plain DIIS (no saddle check, no stability probe) ends on
/// a spin-contaminated saddle point (<S^2> 0.93); the saddle check finds the negative mode
/// (eigenvalue ~ -0.1) and the SCF ends 52 mEh lower (<S^2> 0.77), the same with 1 and 4 threads.
#[test]
fn saddle_point_is_left_for_the_minimum() {
    let saddle = run(LA18, 1, 2, "hf", 1e-7, 4, |o| {
        o.scf.stability_saddle_check = Some(0.0);
        o.scf.stability_probe = Some(linus::driver::options::Auto::Value(0));
    })
    .unwrap();
    assert!(saddle.s2 > 0.9, "<S^2> {}", saddle.s2);
    let r = run(LA18, 1, 2, "hf", 1e-7, 4, |_| {}).unwrap();
    // the saddle check's analysis and step; then (the default probe) a probe of the minimum
    assert!(
        !r.stability.is_empty() && r.stability[1..].iter().all(|c| c.probe && c.step == 0.0),
        "{:?}",
        r.stability
    );
    let chk = &r.stability[0];
    assert!(
        chk.eigenvalue < -0.05 && chk.step > 0.0 && chk.de < -1e-3,
        "{chk:?}"
    );
    assert!(
        r.energy < saddle.energy - 0.04,
        "{} vs saddle {}",
        r.energy,
        saddle.energy
    );
    assert!(r.s2 < 0.8, "<S^2> {}", r.s2);
    assert!(r.history.last().unwrap().gmax < 1e-7);
    let one = run(LA18, 1, 2, "hf", 1e-7, 1, |_| {}).unwrap();
    assert_eq!(one.iterations, r.iterations);
    assert_eq!(
        one.energy.to_bits(),
        r.energy.to_bits(),
        "{} vs {}",
        one.energy,
        r.energy
    );
    assert_eq!(
        one.stability[0].eigenvalue.to_bits(),
        chk.eigenvalue.to_bits()
    );
}
