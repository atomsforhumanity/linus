//! Open-shell SCFs on the device loop (feature `gpu`, `backend.scf_loop: "device"`) decide as
//! the host loop does: the stability probe and the zero-mode check of open-shell Hessians run
//! on the device; a negative probe, the saddle check's analysis and a two-cycle continue on the
//! host loop from the device's state; the soft-mode plateau rule applies at `max_iter`. Each
//! case runs on the CPU, the GPU's host loop and its device loop: the same stability decisions,
//! iteration counts and energies (the hard cases of tests/scf_robustness.rs). A `scf.lindep` that
//! leaves fewer orbitals than occupied ones is the same input error on every loop. Without a
//! usable device the tests skip, unless `LINUS_GPU_REQUIRED=1`.
#![cfg(feature = "gpu")]

mod common;

use linus::driver::report::Report;
use linus::{CalcOptions, CalcType, LinusError, Session, Structure};
use serde_json::{json, Value};

const A: f64 = 1.0 / linus_core::units::BOHR_ANGSTROM;

/// The session of `mol` with the JSON settings `opts` on `backend` (`cpu`, or the GPU with
/// the loop `host` or `device`); None without a usable device (a skip).
fn calc(mol: &Structure, opts: &Value, backend: &str) -> Option<Session> {
    let mut o = opts.clone();
    o["backend"] = match backend {
        "cpu" => json!("cpu"),
        l => json!({"kind": "gpu", "scf_loop": l}),
    };
    o["threads"] = json!(4);
    if o["method"] == "hf" {
        // Hartree-Fock has no XC grid (D41: its grid settings are input errors)
        o.as_object_mut().unwrap().remove("grid");
    }
    match Session::new(mol.clone(), CalcOptions::from_json(o).unwrap()) {
        Ok(c) => Some(c),
        Err(LinusError::Backend { message, .. })
            if std::env::var("LINUS_GPU_REQUIRED").as_deref() != Ok("1") =>
        {
            eprintln!("no usable GPU, skipping: {message}");
            None
        }
        Err(e) => panic!("{e}"),
    }
}

/// tests/scf_robustness.rs's settings: def2-SVP, density fitting, PySCF grid level 3, the
/// extreme preset.
fn robust(method: &str) -> Value {
    json!({"method": method, "basis": "def2-svp", "jk": "df", "grid": "pyscf:3",
           "scf": {"convergence": "extreme", "max_iter": 300}})
}

/// `calc` on the CPU, the GPU's host loop and its device loop (None: no device).
fn three(mol: &Structure, opts: &Value, calc_type: CalcType) -> Option<[Report; 3]> {
    let mut out = Vec::new();
    for b in ["cpu", "host", "device"] {
        let mut c = calc(mol, opts, b)?;
        let r = c.run(calc_type).unwrap_or_else(|e| panic!("{b}: {e}"));
        let loop_used = serde_json::to_value(c.backend_used()).unwrap()["scf_loop"].clone();
        if b != "cpu" {
            assert_eq!(loop_used, b, "{b}");
        }
        out.push(r);
    }
    out.try_into().ok()
}

/// The probe's Ritz values (one-sided products from the lowest-diagonal unit vector, or from the
/// direction towards the lowest-energy iterate): the O(h) error of a one-sided difference changes
/// sign with the start vector's physical direction, which follows the eigensolver's arbitrary
/// sign of the two orbitals it rotates (the host's faer and the device's cuSOLVER differ for
/// some pairs), and at a point converged to round-off the direction towards the lowest iterate
/// is noise. The decision does not depend on them.
const PROBE_TOL: f64 = 1e-3;

/// What a run decided: the stability analyses (each one's iteration, probe or full, step and
/// Fock builds), the steps of the SCF path, a two-cycle, the plateau.
type Decisions = (
    Vec<(usize, bool, f64, usize)>,
    Vec<&'static str>,
    bool,
    bool,
);

fn decisions(o: &Report) -> Decisions {
    (
        o.scf
            .stability
            .iter()
            .map(|c| (c.iter, c.probe, c.step, c.fock_builds))
            .collect(),
        o.scf.history.iter().map(|i| i.step).collect(),
        o.resolved.record.scf.two_cycle_at.is_some(),
        o.resolved.record.scf.plateau.is_some(),
    )
}

/// The device loop against the GPU's host loop (the same engines): the same decisions and
/// iterations, energies within `tol`, the full analyses' eigenvalues within `etol` (the probe's
/// within [`PROBE_TOL`]). The GPU against the CPU: the same stability decisions, a two-cycle and
/// a plateau where the CPU has one, energies within `cpu_tol` (soft-mode cases end along their
/// mode at the backends' round-off).
fn same_decisions(case: &str, r: &[Report; 3], tol: f64, etol: f64, cpu_tol: f64) {
    device_as_host(case, r, tol, etol);
    let [c, h, _] = r;
    assert!(
        (h.energy - c.energy).abs() < cpu_tol,
        "{case}: gpu E {} vs cpu {}",
        h.energy,
        c.energy
    );
    let (dh, dc) = (decisions(h), decisions(c));
    assert_eq!(
        (&dh.0, dh.2, dh.3),
        (&dc.0, dc.2, dc.3),
        "{case}: gpu vs cpu decisions"
    );
}

/// The device loop against the GPU's host loop (the first part of [`same_decisions`]).
fn device_as_host(case: &str, r: &[Report; 3], tol: f64, etol: f64) {
    let [c, h, d] = r;
    let show = |o: &Report| {
        let st: Vec<_> = o
            .scf
            .stability
            .iter()
            .map(|c| {
                (
                    c.iter,
                    c.probe,
                    c.eigenvalue,
                    c.davidson_iterations,
                    c.step,
                    c.fock_builds,
                )
            })
            .collect();
        format!("E {:.12} ({} iterations) {st:?}", o.energy, o.iterations)
    };
    eprintln!(
        "{case}:\n  cpu    {}\n  host   {}\n  device {}",
        show(c),
        show(h),
        show(d)
    );
    assert!(
        (d.energy - h.energy).abs() < tol,
        "{case}: device loop E {} vs host loop {}",
        d.energy,
        h.energy
    );
    assert_eq!(d.iterations, h.iterations, "{case}: iterations");
    assert_eq!(decisions(d), decisions(h), "{case}: device vs host loop");
    for (a, b) in d.scf.stability.iter().zip(&h.scf.stability) {
        let ok = if a.probe {
            (a.eigenvalue - b.eigenvalue).abs() < PROBE_TOL
        } else {
            (a.eigenvalue - b.eigenvalue).abs() < etol
                && a.davidson_iterations == b.davidson_iterations
        };
        assert!(ok, "{case}: device {a:?} vs host {b:?}");
    }
    assert_eq!(
        d.resolved.record.scf.stability_probe, h.resolved.record.scf.stability_probe,
        "{case}"
    );
}

/// An ordinary radical (H2O+): the default probe (two products per irrep) runs on the device and finds
/// no negative mode, as on the host; the SCF ends on the device with the host's record, and
/// the device result repeats bit for bit.
#[test]
fn probe_on_the_device_decides_as_the_host() {
    let w = Structure::from_xyz_str(
        "3\nwater\nO 0.0 0.05 0.11\nH 0.0 0.76 -0.47\nH 0.1 -0.75 -0.46\n",
        1,
        2,
    )
    .unwrap();
    for method in ["hf", "b3lyp", "wb97x-d3bj"] {
        let o = json!({"method": method, "basis": "def2-svp", "jk": "df", "grid": "pyscf:2"});
        let Some(r) = three(&w, &o, CalcType::Energy) else {
            return;
        };
        same_decisions(method, &r, 1e-10, 1e-7, 1e-9);
        // two products per irrep of the (planar: Cs) state
        let chk = &r[2].scf.stability;
        assert!(
            chk.len() == 1
                && chk[0].probe
                && chk[0].irreps.symmetry == "Cs"
                && chk[0].fock_builds == 2 * chk[0].irreps.irreps_searched,
            "{chk:?}"
        );
        assert_eq!(r[2].resolved.record.scf.stability_probe, 2);
        let again = calc(&w, &o, "device")
            .unwrap()
            .run(CalcType::Energy)
            .unwrap();
        assert_eq!(again.energy.to_bits(), r[2].energy.to_bits(), "{method}");
        assert_eq!(
            again.scf.stability[0].eigenvalue.to_bits(),
            chk[0].eigenvalue.to_bits(),
            "{method}"
        );
    }
}

/// Triplet O3, B3LYP (tests/scf_robustness.rs), the default thresholds: DIIS converges on a
/// saddle point that the probe (three products) finds; the device's probe sees the negative mode
/// and hands over, and the host loop's analysis and step reach the same minimum. The searches run
/// per irrep of the C2v state, so the mode, of another irrep than the lowest pair, is found and
/// its step breaks the molecule's symmetry on every backend; how many
/// iterations that takes depends on the rounding (CPU 58, the GPU's host loop 55, its device
/// loop 54), so the three are compared on the decision at the saddle (the same iteration, the
/// same probe-triggered analysis and step) and on the state they end in, not on the iterations
/// after the step.
#[test]
fn negative_probe_continues_on_the_host_loop() {
    let (x, z) = (1.0648976575756894 * A, 0.74564936725636 * A);
    let o3 = Structure::new(
        vec![8, 8, 8],
        vec![[0.0; 3], [x, 0.0, z], [-x, 0.0, z]],
        0,
        3,
    )
    .unwrap();
    let mut o = robust("b3lyp");
    o["scf"] = json!({"max_iter": 300, "stability_probe": 3});
    let Some(r) = three(&o3, &o, CalcType::Energy) else {
        return;
    };
    let [c, h, d] = &r;
    for (o, b) in [(h, "host"), (d, "device")] {
        assert!(
            (o.energy - c.energy).abs() < 1e-8,
            "{b} E {} vs cpu {}",
            o.energy,
            c.energy
        );
    }
    let first = |o: &Report| {
        let k = &o.scf.stability[0];
        (k.iter, k.probe, k.step, k.fock_builds)
    };
    assert_eq!(
        first(h),
        first(c),
        "O3 triplet: the first analysis, host vs cpu"
    );
    // both GPU loops end in the same state
    assert!(
        (d.energy - h.energy).abs() < 1e-10,
        "O3 triplet: device loop E {} vs host loop {}",
        d.energy,
        h.energy
    );
    // the device loop takes the same decision at the same iteration (exact). Its analysis's
    // Davidson may stop one iteration (its two products, one per spin) earlier or later: the
    // XC functional is evaluated on the device (H2XC), whose round-off differs from libxc's on
    // the host at the 1e-14 level, which moved the device loop's analysis from 16 to 14 products
    // on an RTX PRO 6000 (with libxc on the host both loops have 16). No looser than that.
    let decision = |o: &Report| {
        let k = &o.scf.stability[0];
        (k.iter, k.probe, k.step)
    };
    assert_eq!(
        decision(d),
        decision(h),
        "O3 triplet: the first analysis, device vs host"
    );
    let builds = |o: &Report| o.scf.stability[0].fock_builds as i64;
    assert!(
        (builds(d) - builds(h)).abs() <= 2,
        "O3 triplet: the first analysis's products, device {} vs host {}",
        builds(d),
        builds(h)
    );
    let chk = &r[2].scf.stability[0];
    assert!(
        !chk.probe && chk.eigenvalue < -1e-2 && chk.step > 0.0,
        "{chk:?}"
    );
}

/// Triplet O3, B3LYP/def2-SVP at the default settings on the device loop, from the SAD guess as
/// it is and from four last-bit flips of it (tests/scf_robustness.rs,
/// `o3_reaches_the_minimum_whatever_the_last_bits`, on the CPU): the device's probe searches every
/// irrep of the C2v saddle, so every SCF ends at the symmetry-broken minimum whatever the device's
/// rounding does.
#[test]
fn o3_device_loop_reaches_the_minimum_whatever_the_last_bits() {
    let (x, z) = (1.0648976575756894 * A, 0.74564936725636 * A);
    let o3 = Structure::new(
        vec![8, 8, 8],
        vec![[0.0; 3], [x, 0.0, z], [-x, 0.0, z]],
        0,
        3,
    )
    .unwrap();
    let opts = json!({"method": "b3lyp", "basis": "def2-svp"});
    for seed in 0..=4u64 {
        let Some(mut c) = calc(&o3, &opts, "device") else {
            return;
        };
        if seed > 0 {
            let (da, db) = common::flipped_guess(&o3, "def2-svp", seed);
            c.set_guess_density(da, db).unwrap();
        }
        let r = c
            .run(CalcType::Energy)
            .unwrap_or_else(|e| panic!("seed {seed}: {e}"));
        assert!(
            (r.energy - -225.1961432).abs() < 1e-6,
            "seed {seed}: E {} ({:?})",
            r.energy,
            r.scf.stability
        );
    }
}

/// The L-lactide radical cation, UHF/def2-SVP (tests/openshell_scf.rs): DIIS converges on a
/// saddle point above the lowest energy it visited; the saddle check fires on the device loop,
/// which hands over, and the host loop's analysis (eigenvalue -0.11) and step reach the CPU's
/// minimum, 52 mEh below the saddle: the same decisions at the same iterations on the CPU, the
/// GPU's host loop and its device loop. (HF+ r2SCAN on the axis, the knife edge of
/// tests/scf_robustness.rs, used to stand here: which of its saddles and minima a backend reaches
/// follows its last bits.)
#[test]
fn saddle_check_continues_on_the_host_loop() {
    let la18 =
        Structure::from_xyz_str(include_str!("../../../bench/molecules/la18.xyz"), 1, 2).unwrap();
    let o = json!({"method": "hf", "basis": "def2-svp", "jk": "df",
                   "scf": {"e_tol": 1e-9, "g_tol": 1e-7, "max_iter": 100}});
    let Some(r) = three(&la18, &o, CalcType::Energy) else {
        return;
    };
    same_decisions("la18+ UHF", &r, 1e-9, 1e-6, 1e-9);
    let chk = &r[2].scf.stability[0];
    assert!(
        !chk.probe && chk.e_above_min > 0.0 && chk.eigenvalue < -0.05 && chk.step > 0.0,
        "{chk:?}"
    );
}

/// CN (tests/scf_robustness.rs), UHF: a two-cycle; the device loop stops where the host loop
/// detects it, and the host loop's quasi-Newton steps converge from there (the stagnation's
/// escalation, which would come first, off).
#[test]
fn two_cycle_continues_on_the_host_loop() {
    let cn = Structure::new(
        vec![6, 7],
        vec![[0.0; 3], [0.0, 0.0, 2.2147590195639206]],
        0,
        2,
    )
    .unwrap();
    let mut o = robust("hf");
    o["scf"]["stagnation_window"] = json!(0);
    let Some(r) = three(&cn, &o, CalcType::Energy) else {
        return;
    };
    same_decisions("CN", &r, 1e-8, 1e-6, 1e-6);
    assert!(r[2].resolved.record.scf.two_cycle_at.is_some());
}

/// CN as above with the two-cycle reaction off: DIIS cycles on every loop until it stagnates.
/// The device loop keeps the lowest-energy iterate's occupied orbitals and hands its state after
/// the step the escalation is due at to the host loop, which escalates from them
/// (`scf.escalate`) as a host run does: the same restart iteration, the same escalation
/// iterations and the same minimum.
#[test]
fn escalation_continues_on_the_host_loop() {
    let cn = Structure::new(
        vec![6, 7],
        vec![[0.0; 3], [0.0, 0.0, 2.2147590195639206]],
        0,
        2,
    )
    .unwrap();
    let mut o = robust("hf");
    o["scf"]["two_cycle"] = json!(false);
    let Some(r) = three(&cn, &o, CalcType::Energy) else {
        return;
    };
    same_decisions("CN, two_cycle off", &r, 1e-8, 1e-6, 1e-6);
    let esc: Vec<_> = r
        .iter()
        .map(|x| {
            *x.resolved
                .record
                .scf
                .escalation
                .expect("a record")
                .ran()
                .expect("escalated")
        })
        .collect();
    for e in &esc {
        assert!(
            e.trigger == "stagnation" && e.from_iter > 1 && e.converged,
            "{e:?}"
        );
    }
    assert_eq!(esc[2].at, esc[1].at, "device vs host loop");
    assert_eq!(
        (esc[2].from_iter, esc[2].iterations),
        (esc[1].from_iter, esc[1].iterations),
        "device vs host loop"
    );
    assert_eq!(esc[1].from_iter, esc[0].from_iter, "gpu vs cpu");
}

/// The F atom in C-PCM (tests/scf_robustness.rs), HF: the energy stationary with max |F_ai|
/// stuck near 1e-8; when its escalation comes due the device loop hands the plateau to the host
/// loop, which evaluates the handed-over state, probes it and accepts it, as it would its own
/// iterations. Along the plateau's soft mode the loops' energies differ at ~1e-8, and which
/// iteration holds the lowest energy (and so when the escalation comes due) is round-off's: the
/// loops agree on the decisions, not on that iteration.
#[test]
fn plateau_rule_applies_to_the_device_loop() {
    let f = Structure::new(vec![9], vec![[0.0; 3]], 0, 2).unwrap();
    let mut o = robust("hf");
    o["scf"]["max_iter"] = json!(150);
    o["solvent"] = json!({"name": "acetonitrile", "model": "cpcm"});
    let Some(r) = three(&f, &o, CalcType::Energy) else {
        return;
    };
    for x in &r {
        let scf = &x.resolved.record.scf;
        assert!(
            scf.plateau.is_some() && scf.escalation.is_none() && x.iterations < 150,
            "{:?} {:?} {}",
            scf.plateau,
            scf.escalation,
            x.iterations
        );
        assert!(
            (x.energy - r[0].energy).abs() < 5e-8,
            "{} vs {}",
            x.energy,
            r[0].energy
        );
    }
}

/// The zero-mode check of open-shell Hessians runs on the device loop as on the host: NH3+
/// (planar), HF; the check (the probe of the converged point with the check's products, its
/// record) agrees in eigenvalue and builds, and the Hessians.
#[test]
fn mode_check_on_the_device() {
    let p = [[0.0, 0.0], [1.02, 0.0], [-0.51, 0.883], [-0.51, -0.883]];
    let nh3p = Structure::new(
        vec![7, 1, 1, 1],
        p.iter().map(|q| [q[0] * A, q[1] * A, 0.0]).collect(),
        1,
        2,
    )
    .unwrap();
    let o = json!({"method": "hf", "basis": "def2-svp", "jk": "df",
                   "scf": {"convergence": "extreme"}, "response": {"tol": 1e-9}});
    let Some(r) = three(&nh3p, &o, CalcType::Hessian) else {
        return;
    };
    same_decisions("NH3+ Hessian", &r, 1e-10, 1e-7, 1e-9);
    let mc = |x: &Report| {
        x.resolved
            .record
            .response
            .as_ref()
            .and_then(|r| r.mode_check.as_ref())
            .map(|m| (m.eigenvalue, m.fock_builds, m.flagged))
            .expect("a mode check")
    };
    let (c, d) = (mc(&r[1]), mc(&r[2]));
    assert!(
        (c.0 - d.0).abs() < PROBE_TOL && c.1 == d.1 && c.2 == d.2,
        "host loop {c:?}, device loop {d:?}"
    );
    let dh = r[2]
        .hessian
        .as_ref()
        .unwrap()
        .iter()
        .zip(r[1].hessian.as_ref().unwrap())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max);
    assert!(dh < 1e-7, "max |dH| {dh:e}");
}

/// A `scf.lindep` that leaves fewer orbitals than occupied ones is the same input error, naming
/// the counts, on the CPU, the GPU's host loop and its device loop (the host loops met a faer
/// assertion, the device loop reported an internal error): water STO-3G, whose overlap
/// eigenvalues 0.343, 0.417 and 0.886 are below 0.9, leaves four orbitals for five occupied
/// (the cation's five alpha).
#[test]
fn a_lindep_that_drops_occupied_orbitals_is_an_input_error() {
    let water = "3\n\nO 0.000 0.000 0.118\nH 0.020 0.755 -0.471\nH -0.010 -0.760 -0.465\n";
    let opts = json!({"method": "hf", "basis": "sto-3g", "jk": "df", "scf": {"lindep": 0.9}});
    for (charge, mult) in [(0, 1), (1, 2)] {
        let mol = Structure::from_xyz_str(water, charge, mult).unwrap();
        for b in ["cpu", "host", "device"] {
            let Some(mut c) = calc(&mol, &opts, b) else {
                return;
            };
            match c.energy() {
                Err(LinusError::Input(m)) => assert_eq!(
                    m, "scf.lindep 9e-1 drops the basis to 4 orbitals, fewer than the 5 occupied",
                    "{b}, charge {charge}"
                ),
                other => panic!("{b}, charge {charge}: expected an input error, got {other:?}"),
            }
        }
    }
}
