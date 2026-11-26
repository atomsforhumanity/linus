//! The SCF's hand-off to its consumers (docs/internal/deggrad.md): whichever solver converged it,
//! the final orbitals span the occupied space of the density the energy was evaluated at, so the
//! gradient, the Hessian and the closed-shell probe all describe the converged determinant.
//!
//! A quasi-Newton step keeps the occupied set it starts from. The escalation restarts from an
//! earlier iteration's occupied orbitals with such steps, and can converge to a stationary point
//! whose occupied set is not the aufbau set of its own Fock matrix (C2 singlet PBE: 3 sigma_g empty
//! 3.7e-4 Eh below the occupied pi pair). The final orbitals used to be the aufbau eigenvectors of
//! the last Fock matrix: another determinant, whose gradient was off by up to 0.13 Eh/bohr while
//! the energy was right. The cases are the robustness sweep's (bench/robust: PBE/def2-SVP, RI-JK,
//! ORCA's DefGrid3 with grid response, the extreme preset), with finite differences of fresh SCFs
//! (no warm start) as the reference.

use linus::driver::options::{Auto, Reference, Stability};
use linus::driver::ConvergencePreset;
use linus::{CalcOptions, CalcType, GridChoice, JkMethod, Report, ScfResult, Session, Structure};

const A: f64 = 1.0 / linus_core::units::BOHR_ANGSTROM;

/// The sweep's settings: def2-SVP, RI-JK, DefGrid3 (DFT), the extreme preset, 300 iterations, 4
/// threads; `reference` restricted or unrestricted.
fn opts(method: &str, unrestricted: bool) -> CalcOptions {
    CalcOptions {
        method: Some(method.into()),
        basis: Some("def2-svp".into()),
        jk: JkMethod::Df {
            aux: "def2-universal-jkfit".into(),
        }
        .into(),
        grid: linus::driver::options::GridSettings {
            xc: (method != "hf").then_some(GridChoice::Orca { defgrid: 3 }),
            ..Default::default()
        },
        reference: unrestricted.then_some(Reference::Unrestricted),
        threads: Some(4),
        scf: linus::driver::ScfSettings {
            convergence: Some(Auto::Value(ConvergencePreset::Extreme)),
            max_iter: Some(300),
            ..Default::default()
        },
        ..Default::default()
    }
}

/// A diatomic `z1`-`z2` at `r` Angstrom along z.
fn diatomic(z: [u32; 2], r: f64, charge: i32, mult: u32) -> Structure {
    Structure::new(z.to_vec(), vec![[0.0; 3], [0.0, 0.0, r * A]], charge, mult).unwrap()
}

fn run(mol: Structure, o: CalcOptions, calctype: CalcType) -> Report {
    Session::new(mol, o).unwrap().run(calctype).unwrap()
}

/// dE/dz of the second atom of a diatomic by central differences (step `h` bohr) of fresh SCFs.
fn fd_dz(z: [u32; 2], r: f64, charge: i32, mult: u32, o: &CalcOptions, h: f64) -> f64 {
    let e = |s: f64| {
        let mut mol = diatomic(z, r, charge, mult);
        let mut x = mol.positions().to_vec();
        x[1][2] += s * h;
        mol = Structure::new(z.to_vec(), x, charge, mult).unwrap();
        run(mol, o.clone(), CalcType::Energy).energy
    };
    (e(1.0) - e(-1.0)) / (2.0 * h)
}

/// max |C_occ C_occ^T - D| over the channels: the final orbitals against the density the SCF's
/// energy was evaluated at.
fn handoff_mismatch(r: &ScfResult) -> f64 {
    let mut m: f64 = 0.0;
    for (c, d, no) in [(&r.c_a, &r.d_a, r.nalpha), (&r.c_b, &r.d_b, r.nbeta)] {
        let co = c.subcols(0, no);
        let dd = co * co.transpose();
        for j in 0..d.ncols() {
            for i in 0..d.nrows() {
                m = m.max((dd[(i, j)] - d[(i, j)]).abs());
            }
        }
    }
    m
}

/// C2 singlet at 1.243 A, RKS PBE: DIIS stagnates, the escalation restarts from iteration 46 and
/// converges to the non-aufbau state (ORCA 6.0.0 finds the same energy, to 1.3e-10, and the
/// gradient -0.038282036). The analytic gradient agrees with the finite differences (h 5e-4: 1.2e-7
/// apart; main before the fix: -0.1667), the state is recorded as non-aufbau with the warning, and
/// the closed-shell probe sees the molecule's D2h with a plausible eigenvalue (it was -32 Eh in C2h,
/// the probe of the aufbau determinant).
#[test]
fn c2_singlet_after_the_escalation() {
    let o = opts("pbe", false);
    let rep = run(diatomic([6, 6], 1.243, 0, 1), o.clone(), CalcType::Gradient);
    let r = &rep.wavefunction;
    assert!(r.escalation.and_then(|e| e.ran().copied()).is_some());
    assert!(handoff_mismatch(r) < 1e-10, "{:e}", handoff_mismatch(r));
    let g = rep.gradient.as_ref().unwrap()[1][2];
    let fd = fd_dz([6, 6], 1.243, 0, 1, &o, 5e-4);
    assert!((g - fd).abs() < 1e-6, "analytic {g} FD {fd}");
    assert!((g - -0.038282036).abs() < 1e-6, "analytic {g} vs ORCA");
    let na = &rep.resolved.record.scf.non_aufbau;
    assert_eq!(na.len(), 1, "{na:?}");
    assert_eq!(
        (na[0].spin, &na[0].holes[..], &na[0].particles[..]),
        ("restricted", &[4][..], &[6][..])
    );
    assert!((na[0].inversion - 3.687e-4).abs() < 1e-6, "{na:?}");
    let w = rep
        .warnings
        .iter()
        .find(|w| w.code == "non_aufbau")
        .unwrap();
    assert!(
        w.message.contains("1 empty orbital(s) lie below occupied ones (the highest occupied orbital energy is 3.7e-4 Eh"),
        "{}",
        w.message
    );
    // the gap of the reported orbitals: the lowest empty (3 sigma_g) below the highest occupied
    assert!(rep.homo_lumo_gap.unwrap() < -3e-4);
    let p = rep.resolved.record.scf.closed_shell_probe.unwrap();
    assert!(p.eigenvalue.unwrap() > -1.0, "{p:?}");
    assert_eq!(p.irreps.unwrap().symmetry, "D2h", "{p:?}");
}

/// LiF at 4.0 A, PBE, restricted and unrestricted (the UKS singlet keeps alpha = beta from the
/// guess): DIIS blows up, the escalation restarts from iteration 12 and converges to the
/// non-aufbau state (the hole 1.1e-2 Eh below the occupied pair), in both channels for UKS. The
/// analytic gradients agree with the finite differences (main before the fix: off by 1.4e-2).
#[test]
fn lif_after_the_blowup_restricted_and_unrestricted() {
    for unrestricted in [false, true] {
        let o = opts("pbe", unrestricted);
        let rep = run(diatomic([3, 9], 4.0, 0, 1), o.clone(), CalcType::Gradient);
        let r = &rep.wavefunction;
        let esc = r.escalation.and_then(|e| e.ran().copied()).unwrap();
        assert_eq!(esc.trigger, "blowup");
        assert!(handoff_mismatch(r) < 1e-10, "{:e}", handoff_mismatch(r));
        let g = rep.gradient.as_ref().unwrap()[1][2];
        let fd = fd_dz([3, 9], 4.0, 0, 1, &o, 5e-4);
        assert!(
            (g - fd).abs() < 1e-7,
            "UKS {unrestricted}: analytic {g} FD {fd}"
        );
        let spins: Vec<_> = rep
            .resolved
            .record
            .scf
            .non_aufbau
            .iter()
            .map(|v| v.spin)
            .collect();
        let want: &[&str] = if unrestricted {
            &["alpha", "beta"]
        } else {
            &["restricted"]
        };
        assert_eq!(spins, want);
    }
}

/// The C2 Hessian at the non-aufbau state: the CPHF converges with the hole pair's negative
/// e_a - e_i, and the analytic Hessian is the derivative of the gradient without grid response
/// (the grid moving with the atoms) as for any state: central differences of those gradients agree
/// to 1.2e-6 (C2 B3LYP, an aufbau state, also 1.2e-6). Main before the fix: H_zz +1.09 for 0.915.
#[test]
fn c2_singlet_hessian_after_the_escalation() {
    let o = opts("pbe", false);
    let rep = run(diatomic([6, 6], 1.243, 0, 1), o.clone(), CalcType::Hessian);
    let n3 = 6;
    let h_zz = rep.hessian.as_ref().unwrap()[5 * n3 + 5];
    let mut og = o.clone();
    og.grid.response = Some(false);
    let g = |s: f64| {
        let mol =
            Structure::new(vec![6, 6], vec![[0.0; 3], [0.0, 0.0, 1.243 * A + s]], 0, 1).unwrap();
        run(mol, og.clone(), CalcType::Gradient).gradient.unwrap()[1][2]
    };
    let h = 1e-3;
    let fd = (g(h) - g(-h)) / (2.0 * h);
    assert!((h_zz - fd).abs() < 5e-6, "analytic {h_zz} FD {fd}");
}

/// The two-cycle rule's quasi-Newton steps (CN UHF, the stagnation's escalation off so the
/// two-cycle reaction comes first) and a stability step (H2 at 2.0 A, UKS with `scf.stability:
/// always`: a rotation along the negative mode, then quasi-Newton steps to the broken-symmetry
/// minimum) end on rotated orbitals as the escalation does: the hand-off is the canonical orbitals
/// of the converged occupied space (an aufbau one here, so no record), and the gradients agree with
/// the finite differences.
#[test]
fn two_cycle_and_stability_steps_hand_off_their_occupied_space() {
    let mut cn = opts("hf", true);
    cn.scf.stagnation_window = Some(0);
    let mut h2 = opts("b3lyp", true);
    h2.scf.stability = Some(Stability::Always);
    for (z, r, mult, o, h) in [([6, 7], 1.172, 2, cn, 5e-4), ([1, 1], 2.0, 1, h2, 1e-3)] {
        let rep = run(diatomic(z, r, 0, mult), o.clone(), CalcType::Gradient);
        let w = &rep.wavefunction;
        let rotated = w.two_cycle_at.is_some() || w.stability.iter().any(|s| s.eigenvalue < 0.0);
        assert!(rotated, "{z:?}: neither a two-cycle nor a stability step");
        assert!(
            handoff_mismatch(w) < 1e-10,
            "{z:?}: {:e}",
            handoff_mismatch(w)
        );
        assert!(rep.resolved.record.scf.non_aufbau.is_empty());
        let g = rep.gradient.as_ref().unwrap()[1][2];
        let fd = fd_dz(z, r, 0, mult, &o, h);
        assert!((g - fd).abs() < 1e-6, "{z:?}: analytic {g} FD {fd}");
    }
}
