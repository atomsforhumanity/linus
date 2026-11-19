//! The default stability probe (`scf.stability_probe` unset: 2 one-sided products at the
//! session's first geometry, docs/internal/robustness.md, SCF proposal item 4): on a minimum it
//! changes nothing (the energy and gradient bitwise, in the gas phase and in a PCM), closed shells
//! are not probed, later geometries are not probed, and `resolved.scf.stability_probe` says what
//! ran. Its catch on a missed saddle: tests/scf_robustness.rs (triplet O3).

use linus::solvent::SolventSettings;
use linus::{CalcOptions, CalcType, GridChoice, JkMethod, Session, Structure};
use linus_solvent::PcmModel;

const A: f64 = 1.0 / linus_core::units::BOHR_ANGSTROM;

fn opts(probe: Option<usize>, pcm: bool) -> CalcOptions {
    let mut o = CalcOptions {
        method: Some("b3lyp".into()),
        basis: Some("def2-svp".into()),
        jk: JkMethod::Df {
            aux: "def2-universal-jkfit".into(),
        }
        .into(),
        grid: linus::driver::options::GridSettings {
            xc: Some(GridChoice::Pyscf { level: 3 }),
            ..Default::default()
        },
        threads: Some(4),
        ..Default::default()
    };
    o.scf.stability_probe = probe.map(linus::driver::options::Auto::Value);
    if pcm {
        o.solvent = Some(SolventSettings {
            model: Some(PcmModel::Cpcm),
            name: Some(Some("water".into())),
            ..Default::default()
        });
    }
    o
}

/// Pyramidal NH3+ is planar at its minimum; this geometry (N-H 1.02 A, 110 degrees) is a stable
/// doublet: the probe finds a positive mode.
fn nh3p() -> Structure {
    let (r, c, s) = (1.02 * A, 0.3420201433256687, 0.9396926207859084);
    Structure::new(
        vec![7, 1, 1, 1],
        vec![
            [0.0, 0.0, 0.0],
            [r * s, 0.0, r * c],
            [-0.5 * r * s, 0.8660254037844386 * r * s, r * c],
            [-0.5 * r * s, -0.8660254037844386 * r * s, r * c],
        ],
        1,
        2,
    )
    .unwrap()
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
fn default_probe_changes_nothing_on_a_minimum() {
    for pcm in [false, true] {
        let on = Session::new(nh3p(), opts(None, pcm))
            .unwrap()
            .run(CalcType::Gradient)
            .unwrap();
        let off = Session::new(nh3p(), opts(Some(0), pcm))
            .unwrap()
            .run(CalcType::Gradient)
            .unwrap();
        assert_eq!(on.resolved.record.scf.stability_probe, 2, "pcm {pcm}");
        assert_eq!(off.resolved.record.scf.stability_probe, 0, "pcm {pcm}");
        let chk = &on.scf.stability;
        assert!(
            chk.len() == 1 && chk[0].probe && chk[0].eigenvalue > 1e-3,
            "pcm {pcm}: {chk:?}"
        );
        assert!(off.scf.stability.is_empty());
        assert_eq!(on.energy.to_bits(), off.energy.to_bits(), "pcm {pcm}");
        assert_eq!(on.gradient, off.gradient, "pcm {pcm}");
        assert_eq!(on.dipole, off.dipole, "pcm {pcm}");
    }
}

#[test]
fn closed_shells_are_not_probed() {
    let on = Session::new(water(), opts(None, false))
        .unwrap()
        .run(CalcType::Energy)
        .unwrap();
    assert_eq!(on.resolved.record.scf.stability_probe, 0);
    assert!(on.scf.stability.is_empty());
    // setting it for a closed shell is an input error (D41: it cannot take effect)
    match Session::new(water(), opts(Some(0), false)) {
        Err(linus::LinusError::Setting { code, path, .. }) => {
            assert_eq!((code, path.as_str()), ("no_effect", "scf.stability_probe"))
        }
        r => panic!("expected a setting error, got {:?}", r.err()),
    }
}

#[test]
fn later_geometries_are_not_probed_unless_asked() {
    let mut calc = Session::new(nh3p(), opts(None, false)).unwrap();
    let first = calc.run(CalcType::Energy).unwrap();
    assert_eq!(first.resolved.record.scf.stability_probe, 2);
    let mut x = nh3p().positions().to_vec();
    x[1][2] += 0.01;
    calc.set_positions(&x).unwrap();
    let second = calc.run(CalcType::Energy).unwrap();
    assert_eq!(second.resolved.record.scf.stability_probe, 0);
    assert!(second.scf.stability.is_empty());
    // an explicit setting applies at every geometry
    let mut calc = Session::new(nh3p(), opts(Some(2), false)).unwrap();
    calc.set_positions(&x).unwrap();
    let asked = calc.run(CalcType::Energy).unwrap();
    assert_eq!(asked.resolved.record.scf.stability_probe, 2);
    assert_eq!(asked.scf.stability.len(), 1);
}
