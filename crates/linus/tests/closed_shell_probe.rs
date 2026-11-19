//! The closed-shell stability probe (`scf.stability_closed_probe`, decision D36;
//! bench/results/2026-11-11-scfrob): at a converged RKS point whose HOMO-LUMO gap lies below
//! `scf.stability_closed_gap` (0.15 Eh), 2 one-sided products of the UKS orbital Hessian from the
//! external start. A stretched H2 is flagged (an external mode, a lower UKS solution exists), a
//! stable molecule is not, Hartree-Fock is opt-in, later geometries are not probed unless asked,
//! and the result is the one without the probe bit for bit (gas phase and PCM, a VV10
//! functional), with `resolved.scf.closed_shell_probe` saying what ran or why not.

use linus::solvent::SolventSettings;
use linus::{CalcOptions, CalcType, GridChoice, JkMethod, Session, Structure};
use linus_scf::{ProbeKind, ProbeSkip, TriggerReason, GAP_DROP};
use linus_solvent::PcmModel;

const A: f64 = 1.0 / linus_core::units::BOHR_ANGSTROM;

fn opts(method: &str, probe: Option<usize>) -> CalcOptions {
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
        ..Default::default()
    };
    o.scf.stability_closed_probe = probe.map(linus::driver::options::Auto::Value);
    o
}

fn pcm(mut o: CalcOptions) -> CalcOptions {
    o.solvent = Some(SolventSettings {
        model: Some(PcmModel::Cpcm),
        name: Some(Some("water".into())),
        ..Default::default()
    });
    o
}

/// H2 at `r` Angstrom (the ROBUST hard set's H2_r2.0: RKS gap 0.143 Eh with B3LYP/def2-SVP, a
/// broken-symmetry UKS solution 2.0e-2 Eh lower).
fn h2(r: f64) -> Structure {
    Structure::new(vec![1, 1], vec![[0.0, 0.0, 0.0], [0.0, 0.0, r * A]], 0, 1).unwrap()
}

fn water() -> Structure {
    Structure::new(
        vec![8, 1, 1],
        vec![[0.0, 0.0, 0.0], [1.43, 0.0, 1.1], [-1.43, 0.0, 1.1]],
        0,
        1,
    )
    .unwrap()
}

#[test]
fn stretched_h2_is_flagged_and_uks_is_lower() {
    let rep = Session::new(h2(2.0), opts("b3lyp", None))
        .unwrap()
        .run(CalcType::Energy)
        .unwrap();
    let p = rep
        .resolved
        .record
        .scf
        .closed_shell_probe
        .expect("a closed-shell record");
    eprintln!("{p:?}");
    assert_eq!(p.products, 2);
    assert!(p.gap.unwrap() < 0.15 && p.skipped.is_none(), "{p:?}");
    let e = p.eigenvalue.unwrap();
    assert!(e < -1e-2, "{p:?}");
    assert_eq!(p.kind, Some(ProbeKind::External));
    assert_eq!(p.unstable, Some(true));
    assert!(p.fock_builds.unwrap() <= 2);
    assert!(rep.timings.contains_key("closed_shell_probe"));
    // what the warning says: UKS with the stability analysis finds a lower solution
    let mut o = opts("b3lyp", None);
    o.reference = Some(linus::driver::options::Reference::Unrestricted);
    o.scf.stability = Some(linus::driver::options::Stability::Always);
    let uks = Session::new(h2(2.0), o)
        .unwrap()
        .run(CalcType::Energy)
        .unwrap();
    assert!(
        uks.energy < rep.energy - 1e-2,
        "{} vs {}",
        uks.energy,
        rep.energy
    );
    assert!(uks.resolved.record.scf.closed_shell_probe.is_none());
    // the probe's Ritz value bounds the full analysis' lowest eigenvalue from above
    let full = uks.scf.stability[0].eigenvalue;
    assert!(e > full - 1e-3, "probe {e} vs full analysis {full}");
}

#[test]
fn a_stable_closed_shell_is_not_flagged() {
    // water's gap (0.3 Eh) is above the gate
    let rep = Session::new(water(), opts("b3lyp", None))
        .unwrap()
        .run(CalcType::Energy)
        .unwrap();
    let p = rep.resolved.record.scf.closed_shell_probe.unwrap();
    assert_eq!(p.skipped, Some(ProbeSkip::GapAboveThreshold), "{p:?}");
    assert!(p.gap.unwrap() > 0.15 && p.eigenvalue.is_none());
    assert!(!rep.timings.contains_key("closed_shell_probe"));
    // forced through the gate: a positive mode
    let mut o = opts("b3lyp", None);
    o.scf.stability_closed_gap = Some(1.0);
    let forced = Session::new(water(), o)
        .unwrap()
        .run(CalcType::Energy)
        .unwrap();
    let p = forced.resolved.record.scf.closed_shell_probe.unwrap();
    assert!(
        p.eigenvalue.unwrap() > 1e-2 && p.unstable == Some(false),
        "{p:?}"
    );
    assert_eq!(p.kind, Some(ProbeKind::External));
    assert_eq!(forced.energy.to_bits(), rep.energy.to_bits());
}

#[test]
fn hartree_fock_is_opt_in() {
    let rep = Session::new(h2(2.0), opts("hf", None))
        .unwrap()
        .run(CalcType::Energy)
        .unwrap();
    let p = rep.resolved.record.scf.closed_shell_probe.unwrap();
    assert_eq!((p.products, p.skipped), (0, Some(ProbeSkip::HfDefault)));
    let off = Session::new(h2(2.0), opts("b3lyp", Some(0)))
        .unwrap()
        .run(CalcType::Energy)
        .unwrap();
    let q = off.resolved.record.scf.closed_shell_probe.unwrap();
    assert_eq!((q.products, q.skipped), (0, Some(ProbeSkip::Off)));
    // RHF's gap is larger (0.35 Eh here): opt in with a wider gate
    let mut o = opts("hf", Some(2));
    o.scf.stability_closed_gap = Some(1.0);
    let asked = Session::new(h2(2.0), o)
        .unwrap()
        .run(CalcType::Energy)
        .unwrap();
    let p = asked.resolved.record.scf.closed_shell_probe.unwrap();
    assert!(
        p.is_unstable() && p.kind == Some(ProbeKind::External),
        "{p:?}"
    );
    assert_eq!(asked.energy.to_bits(), rep.energy.to_bits());
    // UHF/UKS determinants carry no record
    let mut o = opts("b3lyp", None);
    o.reference = Some(linus::driver::options::Reference::Unrestricted);
    let u = Session::new(water(), o)
        .unwrap()
        .run(CalcType::Energy)
        .unwrap();
    assert!(u.resolved.record.scf.closed_shell_probe.is_none());
}

#[test]
fn later_geometries_are_not_probed_unless_asked() {
    let mut calc = Session::new(h2(2.0), opts("b3lyp", None)).unwrap();
    let first = calc.run(CalcType::Energy).unwrap();
    assert!(first
        .resolved
        .record
        .scf
        .closed_shell_probe
        .unwrap()
        .is_unstable());
    let x = h2(2.05).positions().to_vec();
    calc.set_positions(&x).unwrap();
    let second = calc.run(CalcType::Energy).unwrap();
    let p = second.resolved.record.scf.closed_shell_probe.unwrap();
    assert_eq!((p.products, p.skipped), (0, Some(ProbeSkip::LaterGeometry)));
    let mut calc = Session::new(h2(2.0), opts("b3lyp", Some(2))).unwrap();
    calc.set_positions(&x).unwrap();
    let asked = calc.run(CalcType::Energy).unwrap();
    assert!(asked
        .resolved
        .record
        .scf
        .closed_shell_probe
        .unwrap()
        .is_unstable());
}

/// The JSON values `a` and `b` agree: numbers within `tol` (relative to 1), the rest equal; the
/// keys in `skip` (wall times) are not compared.
fn same_json(a: &serde_json::Value, b: &serde_json::Value, tol: f64, skip: &[&str]) -> bool {
    use serde_json::Value::*;
    match (a, b) {
        (Object(x), Object(y)) => {
            x.len() == y.len()
                && x.iter().all(|(k, v)| {
                    skip.contains(&k.as_str())
                        || y.get(k).is_some_and(|w| same_json(v, w, tol, skip))
                })
        }
        (Array(x), Array(y)) => {
            x.len() == y.len() && x.iter().zip(y).all(|(v, w)| same_json(v, w, tol, skip))
        }
        (Number(x), Number(y)) => {
            let (x, y) = (x.as_f64().unwrap(), y.as_f64().unwrap());
            (x - y).abs() <= tol * x.abs().max(1.0)
        }
        _ => a == b,
    }
}

/// The probe runs (stretched H2) and the result is the one without it, bit for bit: energies,
/// gradients, dipoles, energy components and VV10's energy. The solvation summary (and its
/// electrostatic energy in the components) agrees to round-off: the PCM's incremental pass of
/// the build that restores the converged state starts from the probe's last rotated state (2e-18
/// Eh here), as after the open-shell probe.
#[test]
fn the_result_is_unchanged_bit_for_bit() {
    for (method, solvent) in [
        ("b3lyp", false),
        ("b3lyp", true),
        ("wb97m-v", false),
        ("hf", true),
    ] {
        let mk = |probe| {
            let mut o = opts(method, Some(probe));
            o.scf.stability_closed_gap = Some(1.0);
            if solvent {
                pcm(o)
            } else {
                o
            }
        };
        let on = Session::new(h2(2.0), mk(2))
            .unwrap()
            .run(CalcType::Gradient)
            .unwrap();
        let off = Session::new(h2(2.0), mk(0))
            .unwrap()
            .run(CalcType::Gradient)
            .unwrap();
        let case = format!("{method} pcm {solvent}");
        assert!(
            on.resolved
                .record
                .scf
                .closed_shell_probe
                .unwrap()
                .is_unstable(),
            "{case}"
        );
        assert_eq!(on.energy.to_bits(), off.energy.to_bits(), "{case}");
        assert_eq!(on.gradient, off.gradient, "{case}");
        assert_eq!(on.dipole, off.dipole, "{case}");
        assert_eq!(on.iterations, off.iterations, "{case}");
        let js = |r: &linus::driver::report::Report, k: &str| {
            serde_json::to_value(r).unwrap()[k].clone()
        };
        assert_eq!(js(&on, "vv10"), js(&off, "vv10"), "{case}");
        let tol = if solvent { 1e-14 } else { 0.0 };
        for k in ["components", "solvation"] {
            assert!(
                same_json(&js(&on, k), &js(&off, k), tol, &["time"]),
                "{case} {k}: {} vs {}",
                js(&on, k),
                js(&off, k)
            );
        }
    }
}

#[test]
fn bad_settings_are_input_errors() {
    let new = |scf: serde_json::Value| {
        let o = serde_json::json!({"method": "b3lyp", "basis": "def2-svp", "scf": scf});
        Session::new(water(), CalcOptions::from_json(o).unwrap())
    };
    for bad in [
        serde_json::json!({"stability_closed_probe": 1}),
        serde_json::json!({"stability_closed_gap": -0.1}),
    ] {
        let e = new(bad.clone())
            .err()
            .unwrap_or_else(|| panic!("{bad} accepted"));
        assert!(e.is_input(), "{bad}: {e}");
    }
    let c =
        new(serde_json::json!({"stability_closed_probe": 3, "stability_closed_gap": 0.2})).unwrap();
    assert_eq!(c.closed_shell_probe(), 3);
}

/// H2 (charge `charge`, multiplicity `mult`) at `r` Angstrom about its midpoint, so that a
/// stretch moves each atom by half of it: steps of up to 1 Angstrom stay within the guess
/// distance (`linus::driver::GUESS_DISTANCE`, 1 bohr), and the SCF continues its lineage.
fn h2_centered(r: f64, charge: i32, mult: u32) -> Structure {
    let z = 0.5 * r * A;
    Structure::new(
        vec![1, 1],
        vec![[0.0, 0.0, -z], [0.0, 0.0, z]],
        charge,
        mult,
    )
    .unwrap()
}

/// A later geometry of the lineage: the unset probes are off unless the gap trigger fires, when
/// the gap falls below the gate since the last probe or more than 0.02 Eh below the last probe's
/// gap.
#[test]
fn the_gap_trigger_probes_later_geometries() {
    let h2 = |r: f64| h2_centered(r, 0, 1);
    let mut calc = Session::new(h2(1.0), opts("b3lyp", None)).unwrap();
    let first = calc.run(CalcType::Energy).unwrap();
    let p = first.resolved.record.scf.closed_shell_probe.unwrap();
    assert_eq!(p.skipped, Some(ProbeSkip::GapAboveThreshold));
    assert!(first.resolved.record.scf.probe_trigger.is_none());
    let g1 = p.gap.unwrap();
    let step = |calc: &mut Session, r: f64| {
        calc.set_positions(h2(r).positions()).unwrap();
        calc.run(CalcType::Energy).unwrap()
    };
    // below the gate now, above it at the last probe: probed
    let second = step(&mut calc, 2.0);
    let (p, t) = (
        second.resolved.record.scf.closed_shell_probe.unwrap(),
        second.resolved.record.scf.probe_trigger.unwrap(),
    );
    assert!(
        p.is_unstable() && p.trigger == Some(TriggerReason::GapDropped),
        "{p:?}"
    );
    assert_eq!(
        (t.last_gap, t.gap, t.trigger),
        (Some(g1), p.gap, Some(TriggerReason::GapDropped))
    );
    let g2 = p.gap.unwrap();
    // a small further drop: not probed
    let third = step(&mut calc, 2.05);
    let (p, t) = (
        third.resolved.record.scf.closed_shell_probe.unwrap(),
        third.resolved.record.scf.probe_trigger.unwrap(),
    );
    assert_eq!(p.skipped, Some(ProbeSkip::LaterGeometry), "{p:?}");
    assert!(
        t.trigger.is_none() && t.gap.unwrap() > g2 - GAP_DROP,
        "{t:?}"
    );
    // more than 0.02 Eh below the last probe's gap (still g2's: the third step did not
    // probe): probed
    let fourth = step(&mut calc, 2.6);
    let p = fourth.resolved.record.scf.closed_shell_probe.unwrap();
    assert!(p.is_unstable() && p.trigger.is_some(), "{p:?}");
    let t = fourth.resolved.record.scf.probe_trigger.unwrap();
    assert_eq!(t.last_gap, Some(g2));
    // explicit settings apply at every geometry, without the trigger
    let mut c = Session::new(h2(2.0), opts("b3lyp", Some(2))).unwrap();
    c.run(CalcType::Energy).unwrap();
    let r = step(&mut c, 2.05);
    assert!(r
        .resolved
        .record
        .scf
        .closed_shell_probe
        .unwrap()
        .is_unstable());
}

/// The open-shell stability probe at a later geometry: off, unless the smaller alpha or beta gap
/// fires the trigger (H2+ stretched in steps within the guess distance; the gap falls below the
/// gate at 2 Angstrom and keeps falling more than 0.02 Eh per step).
#[test]
fn the_gap_trigger_probes_open_shells() {
    let h2p = |r: f64| h2_centered(r, 1, 2);
    let mut calc = Session::new(h2p(1.0), opts("b3lyp", None)).unwrap();
    let first = calc.run(CalcType::Energy).unwrap();
    assert_eq!(first.resolved.record.scf.stability_probe, 2);
    assert!(first.resolved.record.scf.closed_shell_probe.is_none());
    calc.set_positions(h2p(1.05).positions()).unwrap();
    let near = calc.run(CalcType::Energy).unwrap();
    assert_eq!(near.resolved.record.scf.stability_probe, 0);
    assert!(near.scf.stability.is_empty());
    let t = near.resolved.record.scf.probe_trigger.unwrap();
    assert!(t.trigger.is_none() && t.gap.unwrap() > 0.15, "{t:?}");
    for r in [2.0, 3.0] {
        calc.set_positions(h2p(r).positions()).unwrap();
        calc.run(CalcType::Energy).unwrap();
    }
    calc.set_positions(h2p(4.0).positions()).unwrap();
    let far = calc.run(CalcType::Energy).unwrap();
    let t = far.resolved.record.scf.probe_trigger.unwrap();
    assert!(t.fired() && t.gap.unwrap() < 0.15, "{t:?}");
    assert_eq!(far.resolved.record.scf.stability_probe, 2);
    assert!(!far.scf.stability.is_empty());
}
