//! The zero-mode check of open-shell Hessians (`scf.stability_hessian_check`; docs/internal/robustness.md,
//! bench/robust/repro/uhf_hessian_zero_mode): at a UHF solution with a zero orbital-Hessian mode
//! (C2H2+, linear, a pi hole: a family of degenerate solutions) the analytic Hessian is not the
//! derivative of the gradient along the bend, and the check flags it; an ordinary radical (NH3+)
//! is not flagged and its Hessian is the derivative of its gradient. Where the check finds no
//! negative mode the Hessian with it off is bitwise the same; a negative one takes the stability
//! probe's path (the full analysis, the step, reconvergence), and the Hessian is that of the state
//! the SCF reaches (FeH). A pi-hole diatomic is flagged only where a grid (or a solvent surface)
//! breaks its axial symmetry.

use linus::driver::ConvergencePreset;
use linus::{CalcOptions, CalcType, JkMethod, LinusError, Session, Structure};

const A: f64 = 1.0 / linus_core::units::BOHR_ANGSTROM;

fn opts(mode_check: usize) -> CalcOptions {
    let mut o = CalcOptions {
        method: Some("hf".into()),
        basis: Some("def2-svp".into()),
        jk: linus::driver::options::JkSettings {
            screening: Some(1e-14),
            ..JkMethod::Exact.into()
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
    o.response.tol = Some(linus::driver::options::Auto::Value(1e-9));
    o.scf.stability_hessian_check = Some(mode_check);
    o
}

/// Linear H-C-C-H+ along z (C-C 1.25 A, C-H 1.08 A), doublet.
fn c2h2p() -> Structure {
    let z = [-1.705, -0.625, 0.625, 1.705];
    Structure::new(
        vec![1, 6, 6, 1],
        z.iter().map(|z| [0.0, 0.0, z * A]).collect(),
        1,
        2,
    )
    .unwrap()
}

/// Planar NH3+ (N-H 1.02 A), doublet.
fn nh3p() -> Structure {
    let p = [[0.0, 0.0], [1.02, 0.0], [-0.51, 0.883], [-0.51, -0.883]];
    Structure::new(
        vec![7, 1, 1, 1],
        p.iter().map(|q| [q[0] * A, q[1] * A, 0.0]).collect(),
        1,
        2,
    )
    .unwrap()
}

/// Analytic Hessian (row-major 3N x 3N) and the check's (eigenvalue, flagged).
fn hessian(mol: Structure, mode_check: usize) -> (Vec<f64>, Option<(f64, bool)>) {
    let rep = Session::new(mol, opts(mode_check))
        .unwrap()
        .run(CalcType::Hessian)
        .unwrap();
    let mc = rep
        .resolved
        .record
        .response
        .as_ref()
        .unwrap()
        .mode_check
        .as_ref();
    (rep.hessian.unwrap(), mc.map(|m| (m.eigenvalue, m.flagged)))
}

/// Max |H - H_FD| over the rows of the coordinates `rows` (H_FD: central differences of the
/// analytic gradient, h = 1e-3 bohr).
fn fd_error(mol: Structure, h_an: &[f64], rows: &[usize]) -> f64 {
    let n3 = 3 * mol.positions().len();
    let x0 = mol.positions().to_vec();
    let mut calc = Session::new(mol, opts(0)).unwrap();
    let h = 1e-3;
    let mut err: f64 = 0.0;
    for &i in rows {
        let mut g = [vec![], vec![]];
        for (k, s) in [1.0, -1.0].into_iter().enumerate() {
            let mut x = x0.clone();
            x[i / 3][i % 3] += s * h;
            calc.set_positions(&x).unwrap();
            g[k] = calc.gradient().unwrap().gradient.concat();
        }
        for j in 0..n3 {
            err = err.max((h_an[i * n3 + j] - (g[0][j] - g[1][j]) / (2.0 * h)).abs());
        }
    }
    err
}

#[test]
fn zero_mode_of_a_linear_radical_is_flagged() {
    let (h, mc) = hessian(c2h2p(), 6);
    let (theta, flagged) = mc.expect("the check ran");
    // the exact zero mode, to the six one-sided products' accuracy from the wide start (-2.9e-6)
    assert!(
        flagged && theta.abs() < 1e-5,
        "eigenvalue {theta}, flagged {flagged}"
    );
    // no negative mode, no step: the same Hessian with the check off
    let (h0, mc0) = hessian(c2h2p(), 0);
    assert!(mc0.is_none());
    assert_eq!(h, h0, "the check changed the Hessian");
    // the reason for the warning: along the bend (x of a carbon) the analytic Hessian is not
    // the derivative of the gradient
    let err = fd_error(c2h2p(), &h, &[3]);
    assert!(
        err > 1e-3,
        "analytic vs differences {err:.1e}: expected the degenerate-family error"
    );
}

#[test]
fn ordinary_radical_is_not_flagged() {
    let (h, mc) = hessian(nh3p(), 6);
    let (theta, flagged) = mc.expect("the check ran");
    assert!(
        !flagged && theta > 0.1,
        "eigenvalue {theta}, flagged {flagged}"
    );
    let err = fd_error(nh3p(), &h, &[3, 4]);
    assert!(err < 1e-5, "analytic vs differences {err:.1e}");
}

#[test]
fn mode_check_of_one_product_is_rejected() {
    let e = Session::new(nh3p(), opts(1)).err();
    assert!(matches!(e, Some(LinusError::Input(_))), "{e:?}");
}

/// OH (0.97 A), a pi-hole diatomic: with HF (an exactly axially symmetric functional) the zero
/// mode cannot couple to the stretch and is not flagged; with B3LYP the grid breaks the symmetry
/// and it is.
#[test]
fn diatomic_zero_mode_is_flagged_only_on_a_grid() {
    let oh = || Structure::new(vec![8, 1], vec![[0.0; 3], [0.0, 0.0, 0.97 * A]], 0, 2).unwrap();
    let check = |method: &str| {
        let mut o = opts(6);
        o.method = Some(method.into());
        if method != "hf" {
            o.grid.xc = Some(linus::GridChoice::Pyscf { level: 3 });
        }
        o.scf.convergence = Some(linus::driver::options::Auto::Value(
            ConvergencePreset::Tight,
        ));
        let rep = Session::new(oh(), o)
            .unwrap()
            .run(CalcType::Hessian)
            .unwrap();
        let m = rep.resolved.record.response.unwrap().mode_check.unwrap();
        (m.eigenvalue, m.flagged)
    };
    // the zero mode within the probe's threshold of zero (-1.9e-5 from the wide start), no step
    let (theta, flagged) = check("hf");
    assert!(
        theta.abs() < -linus_scf::PROBE_NEGATIVE && !flagged,
        "HF: {theta} {flagged}"
    );
    let (theta, flagged) = check("b3lyp");
    assert!(theta < 1e-3 && flagged, "B3LYP: {theta} {flagged}");
}

/// NaO (doublet) and CrO (quintet), B3LYP/def2-SVP RI-JK, pyscf:3, the extreme preset (the
/// robustness sweep's hessprobe set): the check searches each irrep of the converged state's group
/// and flags the zero mode of the hole's orientation about the axis, -7.7e-7 and -1.4e-6 (PySCF's
/// UKS stability at the same states: -7.8e-7 and -1.40e-6). It lies in another irrep than the
/// lowest pair's, which a search from that pair alone never reached (+0.065 and +0.030).
#[test]
fn a_zero_mode_in_another_irrep_is_flagged() {
    for (z, at, mult) in [
        (11, [-0.9755791842575424, 0.9755791842575424], 2),
        (24, [-0.01449612013384, 1.63449612013384], 5),
    ] {
        let mol = Structure::new(
            vec![z, 8],
            vec![[0.0, 0.0, at[0] * A], [0.0, 0.0, at[1] * A]],
            0,
            mult,
        )
        .unwrap();
        let mut o = opts(6);
        o.method = Some("b3lyp".into());
        o.jk = JkMethod::Df {
            aux: "def2-universal-jkfit".into(),
        }
        .into();
        o.grid.xc = Some(linus::GridChoice::Pyscf { level: 3 });
        let rep = Session::new(mol, o)
            .unwrap()
            .run(CalcType::Hessian)
            .unwrap();
        let m = rep.resolved.record.response.unwrap().mode_check.unwrap();
        assert!(
            m.flagged && m.eigenvalue.abs() < 1e-5 && m.irreps.symmetry != "C1",
            "Z {z}: {m:?}"
        );
    }
}

/// Quartet FeH, B3LYP/def2-SVP RI-JK (pyscf:3 grid, extreme convergence). From the SAD guess the
/// SCF converges onto a saddle 7.5 mEh above the minimum (the hole's orientation; the state's
/// group C2) whose negative mode, -5.0e-3 in B, the default probe's two products per irrep miss
/// (+1.1e-3 in A, +8.2e-3 in B), and an energy SCF stops there. A Hessian SCF probes every
/// converged point with the zero-mode check's six products per irrep from the irrep's three lowest
/// pairs: it finds the mode, steps, and ends at PySCF's stable UKS solution (-1264.0916459966 Eh:
/// its stability analysis followed to a stable point). On the way it no longer stops at the C2v
/// state 7.2e-5 Eh above (a saddle whose -1.1e-3 mode, in A2, grows from the third pair of its
/// irrep by gap) nor accepts it on the plateau rule. The Hessian is the minimum's, and the check's
/// record at that point is not negative.
#[test]
fn a_hessian_steps_off_a_saddle_the_probe_misses() {
    let b = linus_core::units::BOHR_ANGSTROM;
    let mol = Structure::new(
        vec![26, 1],
        vec![
            [0.0, 0.0, -0.08155152335572 / b],
            [0.0, 0.0, 1.71155152335572 / b],
        ],
        0,
        4,
    )
    .unwrap();
    let mut o = opts(6);
    o.method = Some("b3lyp".into());
    o.jk = JkMethod::Df {
        aux: "def2-universal-jkfit".into(),
    }
    .into();
    o.grid.xc = Some(linus::GridChoice::Pyscf { level: 3 });
    let rep = Session::new(mol, o)
        .unwrap()
        .run(CalcType::Hessian)
        .unwrap();
    assert!(
        (rep.energy - -1264.0916459966).abs() < 1e-6,
        "E {}: the saddle lies at -1264.0841165, the C2v saddle at -1264.0915745; {:?}",
        rep.energy,
        rep.scf.stability
    );
    let m = rep.resolved.record.response.unwrap().mode_check.unwrap();
    assert!(m.eigenvalue > linus_scf::PROBE_NEGATIVE, "{m:?}");
}

/// Triplet CH2 HF/def2-SVP in C-PCM, a Hessian with the default check: the energy components
/// (the solvation summary's electrostatic energy among them) add up to the energy. The check's
/// last rotated build used to stay in the PCM: solvation_electrostatic -0.0016460 instead of
/// -0.0016437 Eh, the components 2.3e-6 Eh off the energy.
#[test]
fn ch2_pcm_hessian_components_add_up() {
    let ch2 = Structure::from_xyz_str(
        "3\nCH2\nC 0.0 0.003 0.102\nH 0.0 0.995 -0.31\nH 0.01 -0.99 -0.302\n",
        0,
        3,
    )
    .unwrap();
    let mut o = CalcOptions {
        method: Some("hf".into()),
        basis: Some("def2-svp".into()),
        threads: Some(4),
        ..Default::default()
    };
    o.solvent = Some(linus::solvent::SolventSettings {
        model: Some(linus_solvent::PcmModel::Cpcm),
        name: Some(Some("water".into())),
        ..Default::default()
    });
    let rep = Session::new(ch2, o)
        .unwrap()
        .run(CalcType::Hessian)
        .unwrap();
    assert!(rep
        .resolved
        .record
        .response
        .as_ref()
        .unwrap()
        .mode_check
        .is_some());
    let c = &rep.components;
    let sum = c.nuclear_repulsion
        + c.one_electron
        + c.coulomb
        + c.exchange
        + c.xc
        + c.dispersion
        + c.solvation_electrostatic.unwrap()
        + c.solvation_cds.unwrap_or(0.0);
    assert!((sum - rep.energy).abs() < 1e-10, "{sum} vs {}", rep.energy);
}

/// The check's rotated builds leave nothing behind: after an open-shell Hessian with it, VV10's
/// energy (a -V functional) and the solvation summary (C-PCM) are the converged state's, bitwise
/// those of the Hessian without the check (they were those of the check's last rotated build).
#[test]
fn the_check_leaves_vv10_and_the_solvent_as_converged() {
    for (method, solvent) in [("wb97m-v", false), ("b3lyp", true)] {
        let run = |mc: usize| {
            let mut o = CalcOptions {
                method: Some(method.into()),
                basis: Some("def2-svp".into()),
                jk: JkMethod::Df {
                    aux: "def2-universal-jkfit".into(),
                }
                .into(),
                threads: Some(4),
                ..Default::default()
            };
            o.scf.stability_hessian_check = Some(mc);
            if solvent {
                o.solvent = Some(linus::solvent::SolventSettings {
                    model: Some(linus_solvent::PcmModel::Cpcm),
                    name: Some(Some("water".into())),
                    ..Default::default()
                });
            }
            Session::new(nh3p(), o)
                .unwrap()
                .run(CalcType::Hessian)
                .unwrap()
        };
        let (on, off) = (run(6), run(0));
        assert!(on
            .resolved
            .record
            .response
            .as_ref()
            .unwrap()
            .mode_check
            .is_some());
        assert_eq!(on.hessian, off.hessian, "{method}");
        let js = |r: &linus::Report, k: &str| serde_json::to_value(r).unwrap()[k].clone();
        assert_eq!(js(&on, "vv10"), js(&off, "vv10"), "{method}");
        // the PCM's restoring pass is incremental from the check's last rotated state: round-off
        let num = |r: &linus::Report, k: &str, f: &str| js(r, k)[f].as_f64().unwrap_or(0.0);
        for (k, f) in [
            ("solvation", "electrostatic"),
            ("solvation", "surface_charge"),
            ("components", "solvation_electrostatic"),
            ("components", "xc"),
            ("components", "coulomb"),
        ] {
            let (a, b) = (num(&on, k, f), num(&off, k, f));
            assert!((a - b).abs() < 1e-14, "{method} {k}.{f}: {a} vs {b}");
        }
    }
}
