//! SCF failure modes the robustness sweep found (docs/internal/robustness.md,
//! bench/robust/repro), each with the behaviour that ends it:
//! - a two-cycle (the energy repeating every other iteration while the SCF does not converge;
//!   the CN radical's pi-hole pair) hands over to quasi-Newton steps (`scf.two_cycle`);
//! - a soft-mode plateau (the F atom in a C-PCM: the energy stationary, max |F_ai| stuck at
//!   ~1e-8) is accepted at `max_iter` with its residual reported and the warning `loosely_converged`
//!   (`scf.plateau_window`);
//! - any other SCF whose DIIS stagnates (or that spends 2/3 of `max_iter`) restarts once from its
//!   lowest-energy orbitals with quasi-Newton steps, within `max_iter` (`scf.escalate`);
//! - a stability analysis whose negative mode has its minimum closer than the first trial
//!   rotation (HF+ r2SCAN, slightly bent: eigenvalue -8.9e-4) takes a shorter step instead of
//!   none;
//! - every stability search runs per irrep of the converged state's symmetry, so whether the SCF
//!   leaves a symmetric saddle (triplet O3) does not depend on the last bits of its path.

mod common;

use linus::driver::ConvergencePreset;
use linus::solvent::SolventSettings;
use linus::ScfResult;
use linus::Structure;
use linus::{CalcOptions, GridChoice, JkMethod, LinusError, Session};
use linus_solvent::PcmModel;

fn run(
    mol: Structure,
    method: &str,
    edit: impl Fn(&mut CalcOptions),
) -> Result<ScfResult, LinusError> {
    session(mol, method, edit).energy()
}

/// The session of the sweep's hard cases: def2-SVP, RI-JK, PySCF grid level 3, the extreme
/// preset, 300 iterations, 4 threads; `edit` changes the options.
fn session(mol: Structure, method: &str, edit: impl Fn(&mut CalcOptions)) -> Session {
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
        threads: Some(4),
        scf: linus::driver::ScfSettings {
            convergence: Some(linus::driver::options::Auto::Value(
                ConvergencePreset::Extreme,
            )),
            max_iter: Some(300),
            ..Default::default()
        },
        ..Default::default()
    };
    edit(&mut o);
    Session::new(mol, o).unwrap()
}

/// CN (1.172 A), UHF/def2-SVP: DIIS stalls with max |F_ai| ~ 1.6e-2 at an energy that stops
/// changing; the two-cycle trigger hands over to SOSCF, which converges to a minimum (stability
/// eigenvalue +0.15) instead of failing at max_iter.
#[test]
fn two_cycle_hands_over_to_soscf() {
    let cn = || {
        Structure::new(
            vec![6, 7],
            vec![[0.0; 3], [0.0, 0.0, 2.2147590195639206]],
            0,
            2,
        )
        .unwrap()
    };
    // the two-cycle reaction alone: the stagnation's escalation would come first
    let r = run(cn(), "hf", |o| o.scf.stagnation_window = Some(0))
        .expect("CN converges after the two-cycle");
    let at = r.two_cycle_at.expect("a two-cycle was detected");
    assert!(
        at > 6 && at < r.iterations,
        "detected at {at} of {}",
        r.iterations
    );
    assert!((r.energy - -92.1287251978).abs() < 1e-8, "E {}", r.energy);
    assert!(r.escalation.is_none());
    let off = run(cn(), "hf", |o| {
        o.scf.two_cycle = Some(false);
        o.scf.escalate = Some(false);
    });
    assert!(
        matches!(off, Err(LinusError::NotConverged(_))),
        "converged without the trigger"
    );
}

/// CN as above with the two-cycle reaction off: DIIS cycles at an energy above its lowest
/// iterate's (iteration 7). After `scf.stagnation_window` iterations without a lower energy the
/// escalation restarts from that iteration's orbitals with quasi-Newton steps and converges to
/// the minimum the two-cycle reaction reaches, within `max_iter`; with the escalation off too,
/// the SCF fails (above).
#[test]
fn escalation_converges_a_cycling_scf() {
    let cn = Structure::new(
        vec![6, 7],
        vec![[0.0; 3], [0.0, 0.0, 2.2147590195639206]],
        0,
        2,
    )
    .unwrap();
    let r = run(cn, "hf", |o| o.scf.two_cycle = Some(false)).expect("converges after escalating");
    let e = *r.escalation.expect("a record").ran().expect("escalated");
    let window = linus_scf::ScfAlgorithm::STAGNATION_WINDOW;
    assert_eq!(e.trigger, "stagnation");
    assert_eq!(e.at, e.from_iter + window, "{e:?}");
    assert!(e.converged && e.iterations < 50, "{e:?}");
    assert_eq!(r.iterations, e.at + e.iterations);
    assert!(e.from_iter > 1 && r.iterations <= 300, "{e:?}");
    assert_eq!(r.history[e.at - 1].step, "escalation");
    assert!(r.history[e.at..].iter().all(|h| h.step != "diis"));
    assert!((r.energy - -92.1287251978).abs() < 1e-8, "E {}", r.energy);
}

/// `max_iter` is the whole budget: with 2 iterations the SCF runs 2, and the escalation, which
/// needs 2 left after its start, does not run; the record and the failure say why.
#[test]
fn max_iter_is_the_whole_budget() {
    let cn = Structure::new(
        vec![6, 7],
        vec![[0.0; 3], [0.0, 0.0, 2.2147590195639206]],
        0,
        2,
    )
    .unwrap();
    let mut s = session(cn, "hf", |o| o.scf.max_iter = Some(2));
    match s.energy() {
        Err(LinusError::NotConverged(f)) => {
            assert_eq!(f.iterations, 2);
            assert!(f.notes.contains("no escalation: max_iter 2"), "{f}");
        }
        r => panic!("{:?}", r.map(|r| r.energy)),
    }
    let r = s.last_unconverged().expect("the unconverged SCF");
    assert_eq!(r.iterations, 2);
    assert_eq!(
        r.escalation,
        Some(linus_scf::Escalation::NotRun { reason: "max_iter" })
    );
    let j = serde_json::to_value(r.escalation).unwrap();
    assert_eq!(j, serde_json::json!({"ran": false, "reason": "max_iter"}));
}

/// F atom (2P), HF/def2-SVP, C-PCM (acetonitrile, 302 points per sphere): from iteration ~10 the
/// energy is stationary to 1e-11 Eh while max |F_ai| stays at 1.7-4.2e-8 (the gas phase converges
/// in 13 iterations). At max_iter the plateau rule accepts it and reports the residual, after the
/// stability probe of the plateau's point (the hole's rotation, a zero mode only the surface keeps
/// from zero: -4.3e-5, within the probe's threshold); with the rule off the SCF fails.
/// Residual risk across ISAs: as for HF+ on the axis below, the p hole starts in a direction of
/// the degenerate p shell that rounding decides, and it stays near it. The generic path and the
/// AVX2/AVX-512 paths take different holes (energies 4.4e-9 Eh apart, residuals 3.4e-8 and
/// 2.9e-8; 2026-11-11), and both reach a plateau. A hole that started on a symmetry axis of the
/// PCM surface could converge outright instead and fail this test.
#[test]
fn soft_mode_plateau_is_accepted_with_its_residual() {
    let f = || Structure::new(vec![9], vec![[0.0; 3]], 0, 2).unwrap();
    let pcm = |o: &mut CalcOptions| {
        o.scf.max_iter = Some(150);
        o.solvent = Some(SolventSettings {
            model: Some(PcmModel::Cpcm),
            name: Some(Some("acetonitrile".into())),
            ..Default::default()
        });
    };
    let r = run(f(), "hf", pcm).expect("accepted on the plateau");
    // accepted when the escalation came due (DIIS stagnated on the plateau), on one more full
    // build, not escalated: the escalation's quasi-Newton steps leave the plateau
    assert!(r.iterations < 150, "{}", r.iterations);
    assert_eq!(r.history.last().unwrap().fock, "full");
    let p = r.plateau.expect("converged by the plateau rule");
    assert_eq!(p.window, 50);
    assert!(p.energy_spread < 1e-10, "{p:?}");
    assert!(p.max_residual < 1e-7 && p.residual > 1e-10, "{p:?}");
    assert_eq!(r.escalation, None);
    assert!(
        r.stability
            .last()
            .is_some_and(|c| c.iter == r.iterations - 1 && c.probe && c.step == 0.0),
        "{:?}",
        r.stability
    );
    // the report warns (loosely_converged) and records the plateau and why nothing escalated
    let rep = session(f(), "hf", pcm)
        .run(linus::CalcType::Energy)
        .expect("accepted");
    let w: Vec<_> = rep
        .warnings
        .iter()
        .filter(|w| w.code == "loosely_converged")
        .collect();
    assert_eq!(w.len(), 1, "{:?}", rep.warnings);
    assert!(
        w[0].message.contains("last 50 iterations"),
        "{}",
        w[0].message
    );
    let j = serde_json::to_value(&rep).unwrap();
    let scf = &j["resolved"]["record"]["scf"];
    assert_eq!(scf["plateau"]["window"], 50);
    assert!(scf.get("escalation").is_none(), "{scf}");
    let off = run(f(), "hf", |o| {
        pcm(o);
        o.scf.plateau_window = Some(0);
        o.scf.escalate = Some(false);
    });
    assert!(
        matches!(off, Err(LinusError::NotConverged(_))),
        "converged without the rule"
    );
    let gas = run(f(), "hf", |_| {}).unwrap();
    assert!(
        gas.plateau.is_none() && gas.iterations < 30,
        "{} iterations",
        gas.iterations
    );
}

/// HF+ (GMTKN55 G21IP/IP_63), r2SCAN/def2-SVP. The r2SCAN energy of the pi hole depends on its
/// orientation about the bond on the (not cylindrical) grid, with saddle points and minima a few
/// degrees apart. Which one DIIS reaches depends on the hole the first iteration picks from the
/// degenerate pi pair, an arbitrary vector of that pair that the ISA's rounding decides: on the
/// z axis (the sweep's geometry, CODATA 2010 bohr) the AVX2 and AVX-512 kernels converge on a
/// saddle (eigenvalue -1.69e-3; 0.25, 0.5 and 1 along the mode raise the energy, 0.125 lowers
/// it by 9.7e-6 Eh) and the generic ones on the minimum directly.
///
/// Bent by 0.04 degrees (H moved by (1e-3, 1e-3, 0) bohr), the pi pair splits and the molecular
/// plane (x = y) is a mirror plane of the grid: every ISA converges on the saddle with the hole
/// fixed by the mirror (-99.7356881247 Eh). The analysis (requested: the saddle check does
/// not fire there) finds -8.9e-4; none of the steps 0.25, 0.5, 1 and 0.125 lowers the energy,
/// 0.0625 lowers it by 1.6e-6 Eh, and the SCF ends 1.79e-6 Eh lower at -99.7356899133 Eh, a
/// minimum, on every platform tried (x86-64 generic, AVX2, AVX-512; aarch64 macOS). Its analysis
/// searches each irrep of the state's group and sees the hole's orientation about the axis, a
/// mode only the grid keeps from zero (+2.9e-7; a search from the lowest pair alone saw +8.6e-4,
/// its own irrep's lowest): positive or within the probe's threshold of zero, no step.
#[test]
fn shallow_negative_mode_takes_a_shorter_step() {
    let z = 0.958622910893333;
    let hf = |h: [f64; 3]| Structure::new(vec![9, 1], vec![[0.0, 0.0, z], h], 1, 2).unwrap();
    let analyse =
        |o: &mut CalcOptions| o.scf.stability = Some(linus::driver::options::Stability::Always);
    let r = run(hf([1e-3, 1e-3, -z]), "r2scan", analyse).unwrap();
    let (chk, last) = (&r.stability[0], r.stability.last().unwrap());
    assert!(chk.eigenvalue < -5e-4, "{chk:?}");
    assert_eq!(chk.step, 0.0625, "{chk:?}");
    assert!(chk.de < -1e-6, "{chk:?}");
    assert!(
        r.stability.len() == 2 && last.step == 0.0 && last.eigenvalue > linus_scf::PROBE_NEGATIVE,
        "{last:?}"
    );
    assert!((r.energy - -99.7356899133).abs() < 1e-9, "E {}", r.energy);
    // on the axis: the saddle and a 0.125 step (AVX2, AVX-512) or the minimum directly (generic),
    // both ending at -99.7356900920; on aarch64 (macOS) the SCF ends on another minimum of the
    // hole's orientation, -99.7356899124. Every path ends below the saddle (-99.7356786955).
    let r = run(hf([0.0, 0.0, -z]), "r2scan", |_| {}).unwrap();
    for chk in &r.stability {
        assert!(chk.eigenvalue > -1e-4 || chk.step > 0.0, "{chk:?}");
    }
    assert!(r.energy < -99.73568, "E {}", r.energy);
}

/// Triplet O3 (bench/robust/molecules/hard.jsonl), the extreme preset's geometry in bohr.
fn o3() -> Structure {
    let ang = 1.0 / linus_core::units::BOHR_ANGSTROM;
    let (x, z) = (1.0648976575756894 * ang, 0.74564936725636 * ang);
    Structure::new(
        vec![8, 8, 8],
        vec![[0.0; 3], [x, 0.0, z], [-x, 0.0, z]],
        0,
        3,
    )
    .unwrap()
}

/// Triplet O3, B3LYP/def2-SVP: the default path converges on a saddle point (lowest
/// orbital-Hessian eigenvalue -3.4e-2) without the saddle check firing, 1.66e-2 Eh above the
/// state the stability analysis reaches (PySCF, Psi4 and ORCA 6.0.0 converge 1.57e-2 below the
/// default). The cheap probe (three one-sided products) finds the negative mode and the SCF ends
/// where the full analysis does.
#[test]
fn stability_probe_finds_the_missed_saddle() {
    let plain = run(o3(), "b3lyp", |o| {
        o.scf.stability_probe = Some(linus::driver::options::Auto::Value(0))
    })
    .unwrap();
    assert!(plain.stability.is_empty(), "{:?}", plain.stability);
    let probed = run(o3(), "b3lyp", |o| {
        o.scf.stability_probe = Some(linus::driver::options::Auto::Value(3))
    })
    .unwrap();
    let full = run(o3(), "b3lyp", |o| {
        o.scf.stability = Some(linus::driver::options::Stability::Always)
    })
    .unwrap();
    assert!(
        probed.energy < plain.energy - 1e-2,
        "{} vs {}",
        probed.energy,
        plain.energy
    );
    assert!(
        (probed.energy - full.energy).abs() < 1e-6,
        "{} vs {}",
        probed.energy,
        full.energy
    );
    let chk = &probed.stability[0];
    assert!(
        !chk.probe && chk.eigenvalue < -1e-2 && chk.step > 0.0,
        "{chk:?}"
    );
}

/// Triplet O3 as above with the default probe: the probe searches each irrep of the C2v state, so
/// the saddle's negative mode is found in its own irrep and the SCF leaves the symmetric saddle for
/// the symmetry-broken state.
#[test]
fn per_irrep_probe_leaves_the_symmetric_saddle() {
    let r = run(o3(), "b3lyp", |_| {}).unwrap();
    assert!((r.energy - -225.1961432).abs() < 1e-6, "E {}", r.energy);
    let chk = &r.stability[0];
    assert!(
        !chk.probe && chk.irreps.symmetry == "C2v" && chk.step > 0.0,
        "{:?}",
        r.stability
    );
}

/// Triplet O3, B3LYP/def2-SVP as above with the default probe, from the SAD guess as it is, from
/// last-bit flips of it and with one coordinate moved by one ulp: every SCF ends at the
/// symmetry-broken minimum. DIIS converges on a C2v saddle (iteration 19) whose negative mode lies
/// in another irrep than the lowest-gap pair; the probe's per-irrep searches find it whatever the
/// rounding. With the single start of the lowest pair the catch rested on round-off breaking the
/// symmetry: one of these perturbations (and a last-bit change of the SAD guess, its atomic SCF
/// on the fast J/K) turned the probe's -3.4e-2 into +0.18 and the SCF stopped on the saddle,
/// -225.1758314 Eh, 20 mEh above the minimum. A round-off change that fails this test is a probe
/// that depends on luck again.
#[test]
fn o3_reaches_the_minimum_whatever_the_last_bits() {
    let mut runs = vec![("as is".to_string(), run(o3(), "b3lyp", |_| {}))];
    for seed in 1..=8u64 {
        let mut c = session(o3(), "b3lyp", |_| {});
        let (da, db) = common::flipped_guess(&o3(), "def2-svp", seed);
        c.set_guess_density(da, db).unwrap();
        runs.push((format!("guess seed {seed}"), c.energy()));
    }
    let mut x = o3().positions().to_vec();
    x[1][0] = f64::from_bits(x[1][0].to_bits() + 1);
    let moved = Structure::new(vec![8, 8, 8], x, 0, 3).unwrap();
    runs.push(("one ulp of x".to_string(), run(moved, "b3lyp", |_| {})));
    for (what, r) in runs {
        let r = r.unwrap_or_else(|e| panic!("{what}: {e}"));
        assert!(
            (r.energy - -225.1961432).abs() < 1e-6,
            "{what}: E {} ({:?})",
            r.energy,
            r.stability
        );
    }
}

/// The C2H5 + H transition state (GMTKN55 BH76), B3LYP/def2-SVP: a doublet whose structure is Cs
/// only within the detection's tolerance (atoms ~1e-3 Å off the mirror), so the irrep projector
/// built from its orbitals holds only nearly. The full analysis searches A' and A'' and reports the
/// lowest eigenvalue PySCF's orbital Hessian has at the same state, +0.106, in A'; with the
/// rotations outside the searched irrep not pushed up, A'' gave a false +8e-8.
#[test]
fn a_nearly_symmetric_structure_has_no_false_zero_mode() {
    let xyz = "7\nBH76 c2h5ts\n\
        C -0.43697488531370227 6.1873954626008e-05 -0.3454053804304669\n\
        C 0.8820416461440614 -2.512611260385203e-05 -0.08451519468295894\n\
        H -1.362982086044648 -0.0004771264253950413 1.405319740338408\n\
        H -0.9707929799941731 0.9206628012791053 -0.535073605410858\n\
        H -0.9711210698647104 -0.9202230034450457 -0.5355572733812952\n\
        H 1.4300311065229006 -0.9223344205151163 0.04731607789381507\n\
        H 1.429798268550261 0.9223350012644242 0.04791563567336141\n";
    let mol = Structure::from_xyz_str(xyz, 0, 2).unwrap();
    let r = run(mol, "b3lyp", |o| {
        o.scf.stability = Some(linus::driver::options::Stability::Always)
    })
    .unwrap();
    let chk = &r.stability[0];
    assert!(
        chk.irreps.symmetry == "Cs"
            && chk.irreps.irreps_searched == 2
            && (chk.eigenvalue - 0.106).abs() < 2e-3
            && chk.step == 0.0,
        "{chk:?}"
    );
}
