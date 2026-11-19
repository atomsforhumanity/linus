//! Harmonic analysis and thermochemistry on spring models, where the answers are known in
//! closed form. The comparisons against PySCF, Psi4 and ORCA are in bench/vib/.

use super::*;

/// Hessian of sum over pairs of k/2 (|r_a - r_b| - d0)^2 at d0 = the current distances: a
/// Hessian invariant to translations and rotations, at a stationary point.
fn springs(positions: &[[f64; 3]], pairs: &[(usize, usize, f64)]) -> Vec<f64> {
    let n3 = 3 * positions.len();
    let mut h = vec![0.0; n3 * n3];
    for &(a, b, k) in pairs {
        let e = normalize(sub(positions[a], positions[b]));
        for i in 0..3 {
            for j in 0..3 {
                let v = k * e[i] * e[j];
                h[(3 * a + i) * n3 + 3 * a + j] += v;
                h[(3 * b + i) * n3 + 3 * b + j] += v;
                h[(3 * a + i) * n3 + 3 * b + j] -= v;
                h[(3 * b + i) * n3 + 3 * a + j] -= v;
            }
        }
    }
    h
}

fn water() -> (Vec<u32>, Vec<[f64; 3]>) {
    (
        vec![8, 1, 1],
        vec![[0.1, -0.2, 0.3], [0.1, 1.33, -0.75], [0.1, -1.73, -0.75]],
    )
}

#[test]
fn diatomic_closed_form() {
    let cst = Constants::CODATA2022;
    let (z, c) = (vec![1, 9], vec![[0.0, 0.0, 0.0], [0.3, 0.4, 1.7]]);
    let k = 0.62;
    let h = springs(&c, &[(0, 1, k)]);
    let v = harmonic_analysis(&z, &c, &h, &VibOptions::default(), None).unwrap();
    assert_eq!(v.rotor_type, RotorType::Linear);
    assert_eq!(v.frequencies.len(), 1);
    let m = &v.masses;
    let mu = m[0] * m[1] / (m[0] + m[1]);
    let nu = (k / mu).sqrt() * cst.wavenumber_per_sqrt_au();
    assert!(
        (v.frequencies[0] - nu).abs() < 1e-9 * nu,
        "{} {nu}",
        v.frequencies[0]
    );
    // reduced mass 1/|M^-1/2 L|^2 (Gaussian's, not m1 m2 / M) and force constant μ λ
    let mu_mode = m[0] * m[1] * (m[0] + m[1]) / (m[0] * m[0] + m[1] * m[1]);
    assert!((v.reduced_masses[0] - mu_mode).abs() < 1e-12);
    assert!(
        (v.force_constants[0] - mu_mode * k / mu * cst.mdyn_per_angstrom_per_au()).abs() < 1e-9
    );
    assert_eq!(v.symmetry.point_group, "C*v");
    assert_eq!(v.rotational_constants[0], None);
    let d2 = norm(sub(c[0], c[1])).powi(2);
    let b = cst.rotational_constant_wavenumber(mu * d2);
    assert!((v.rotational_constants[1].unwrap() - b).abs() < 1e-12 * b);
    // unit Cartesian mode along the bond, the heavy atom moving less
    let x = &v.modes_cartesian[0];
    assert!((x.iter().map(|a| a * a).sum::<f64>() - 1.0).abs() < 1e-12);
    // thermochemistry: closed forms
    let t = ThermoOptions::default();
    let th = thermochemistry(&v, -100.0, 2, &t).unwrap();
    let r = cst.kb_hartree();
    let theta = nu * cst.kelvin_per_wavenumber();
    let x = theta / t.temperature;
    assert!((th.zpe - 0.5 * r * theta).abs() < 1e-15);
    assert!(
        (th.vibrational.entropy - r * (x / x.exp_m1() - (1.0 - (-x).exp()).ln())).abs() < 1e-15
    );
    assert!((th.electronic.entropy - r * 2f64.ln()).abs() < 1e-18);
    let ib = mu * d2 * cst.amu * cst.bohr * cst.bohr;
    let q_rot = 8.0 * std::f64::consts::PI.powi(2) * ib * cst.kb * t.temperature / (cst.h * cst.h);
    assert!((th.rotational.entropy - r * (q_rot.ln() + 1.0)).abs() < 1e-15);
    assert!((th.total.enthalpy - th.total.energy - r * t.temperature).abs() < 1e-14);
    assert!(
        (th.total.gibbs - (th.total.enthalpy - t.temperature * th.total.entropy)).abs() < 1e-14
    );
    assert!((th.gibbs_correction - (th.total.gibbs + 100.0)).abs() < 1e-12);
}

#[test]
fn projection_modes() {
    let (z, c) = water();
    let h = springs(&c, &[(0, 1, 0.5), (0, 2, 0.5), (1, 2, 0.05)]);
    let full = harmonic_analysis(&z, &c, &h, &VibOptions::default(), None).unwrap();
    assert_eq!(full.frequencies.len(), 3);
    assert_eq!(full.symmetry.point_group, "C2v");
    assert_eq!(full.symmetry.symmetry_number, 2);
    // at a stationary point projection changes nothing: the unprojected spectrum is the same
    // three vibrations plus six zeros
    for proj in [Projection::None, Projection::Translations] {
        let v = harmonic_analysis(
            &z,
            &c,
            &h,
            &VibOptions {
                projection: proj,
                ..Default::default()
            },
            None,
        )
        .unwrap();
        assert_eq!(
            v.frequencies.len(),
            if proj == Projection::None { 9 } else { 6 }
        );
        let vib = v.vibrational_frequencies();
        assert_eq!(vib.len(), 3);
        for (a, b) in vib.iter().zip(&full.frequencies) {
            assert!((a - b).abs() < 1e-6, "{proj:?} {a} {b}");
        }
        for (f, e) in v.frequencies.iter().zip(&v.external) {
            assert_eq!(*e, f.abs() < 1.0, "{proj:?} {:?}", v.frequencies);
        }
    }
    // modes: orthonormal in mass-weighted coordinates, orthogonal to translations and rotations
    let n3 = 9;
    for i in 0..3 {
        for j in 0..3 {
            let d: f64 = (0..n3)
                .map(|k| full.modes_mass_weighted[i][k] * full.modes_mass_weighted[j][k])
                .sum();
            assert!((d - if i == j { 1.0 } else { 0.0 }).abs() < 1e-12);
        }
        // no net translation: sum_a m_a x_a = 0
        for k in 0..3 {
            let p: f64 = (0..3)
                .map(|a| full.masses[a] * full.modes_cartesian[i][3 * a + k])
                .sum();
            assert!(p.abs() < 1e-12);
        }
    }
    // mass-weighted eigenvector of the Hessian with the reported frequency
    let cst = Constants::CODATA2022;
    for i in 0..3 {
        let l = &full.modes_mass_weighted[i];
        let mut q = 0.0;
        for a in 0..n3 {
            for b in 0..n3 {
                q += l[a] * h[a * n3 + b] * l[b] / (full.masses[a / 3] * full.masses[b / 3]).sqrt();
            }
        }
        let nu = q.signum() * q.abs().sqrt() * cst.wavenumber_per_sqrt_au();
        assert!((nu - full.frequencies[i]).abs() < 1e-8);
    }
}

#[test]
fn imaginary_modes_and_options() {
    // a saddle: a negative spring constant along one bond
    let (z, c) = water();
    let h = springs(&c, &[(0, 1, 0.5), (0, 2, -0.2), (1, 2, 0.05)]);
    let v = harmonic_analysis(&z, &c, &h, &VibOptions::default(), None).unwrap();
    assert_eq!(v.n_imaginary, 1);
    assert!(v.frequencies[0] < 0.0 && v.frequencies[1] > 0.0);
    let th = thermochemistry(&v, 0.0, 1, &ThermoOptions::default()).unwrap();
    assert_eq!(th.excluded_imaginary, vec![v.frequencies[0]]);
    assert!(th.warnings.iter().any(|w| w.contains("imaginary")));
    assert!(v.warnings.is_empty());
    // slightly broken symmetry: C2v within tolerance, with a warning saying so
    let mut c2 = c.clone();
    c2[2][1] += 0.003;
    let v2 = harmonic_analysis(&z, &c2, &h, &VibOptions::default(), None).unwrap();
    assert_eq!(v2.symmetry.point_group, "C2v");
    assert!(
        v2.symmetry.max_deviation > 1e-3 && v2.symmetry.max_deviation < 3e-3,
        "{:?}",
        v2.symmetry
    );
    assert!(v2
        .warnings
        .iter()
        .any(|w| w.contains("symmetric positions")));
    let v3 = harmonic_analysis(
        &z,
        &c2,
        &h,
        &VibOptions {
            symmetry_tolerance: 1e-3,
            ..Default::default()
        },
        None,
    )
    .unwrap();
    assert_eq!(v3.symmetry.point_group, "Cs");
    assert_eq!(
        th.modes
            .iter()
            .filter(|m| m.used_frequency.is_some())
            .count(),
        2
    );
    let th2 = thermochemistry(
        &v,
        0.0,
        1,
        &ThermoOptions {
            imaginary_as_real: 1e5,
            ..Default::default()
        },
    )
    .unwrap();
    assert!(th2.excluded_imaginary.is_empty() && th2.inverted_imaginary.len() == 1);
    // the isotope override breaks C2v and changes the frequencies
    let d = VibOptions {
        isotopes: vec![(1, masses::isotope_mass(1, 2).unwrap())],
        ..Default::default()
    };
    let vd = harmonic_analysis(&z, &c, &h, &d, None).unwrap();
    assert_eq!(
        (
            vd.symmetry.point_group.as_str(),
            vd.symmetry.symmetry_number
        ),
        ("Cs", 1)
    );
    assert!(vd.frequencies[2] < v.frequencies[2]);
    // user symmetry number
    let vs = harmonic_analysis(
        &z,
        &c,
        &h,
        &VibOptions {
            symmetry_number: Some(1),
            ..Default::default()
        },
        None,
    )
    .unwrap();
    assert_eq!(
        (
            vs.symmetry.symmetry_number,
            vs.symmetry.detected_symmetry_number,
            vs.symmetry.source.as_str()
        ),
        (1, 2, "input")
    );
    let s1 = thermochemistry(&vs, 0.0, 1, &ThermoOptions::default())
        .unwrap()
        .rotational
        .entropy;
    let s2 = thermochemistry(&v, 0.0, 1, &ThermoOptions::default())
        .unwrap()
        .rotational
        .entropy;
    assert!((s1 - s2 - Constants::CODATA2022.kb_hartree() * 2f64.ln()).abs() < 1e-15);
    // bad input
    assert!(harmonic_analysis(&z, &c, &h[1..], &VibOptions::default(), None).is_err());
}

#[test]
fn qrrho_limits() {
    // a soft mode: quasi-RRHO entropy below the harmonic one, enthalpy toward RT/2
    let (z, c) = (vec![6, 8], vec![[0.0, 0.0, 0.0], [0.0, 0.0, 2.13]]);
    let h = springs(&c, &[(0, 1, 1e-4)]);
    let v = harmonic_analysis(&z, &c, &h, &VibOptions::default(), None).unwrap();
    assert!(v.frequencies[0] < 30.0, "{:?}", v.frequencies);
    let t = ThermoOptions::default();
    let rrho = thermochemistry(&v, 0.0, 1, &t).unwrap();
    let q = thermochemistry(
        &v,
        0.0,
        1,
        &ThermoOptions {
            qrrho_entropy: true,
            qrrho_enthalpy: true,
            ..t.clone()
        },
    )
    .unwrap();
    let w = q.modes[0].qrrho_weight;
    assert!(w > 0.0 && w < 0.01);
    assert!(q.vibrational.entropy < rrho.vibrational.entropy);
    let rt = Constants::CODATA2022.kb_hartree() * t.temperature;
    assert!(
        (q.vibrational.energy - (w * rrho.vibrational.energy + (1.0 - w) * 0.5 * rt)).abs() < 1e-16
    );
    assert_eq!(q.low_frequencies.len(), 1);
    // Truhlar: as a 100 cm^-1 mode
    let tr = thermochemistry(
        &v,
        0.0,
        1,
        &ThermoOptions {
            raise_low_frequencies: Some(100.0),
            ..t.clone()
        },
    )
    .unwrap();
    assert_eq!(tr.modes[0].used_frequency, Some(100.0));
    // concentration standard state: 1 M is R ln(24.46...) below 1 atm in S_trans at 298.15 K
    let m = thermochemistry(
        &v,
        0.0,
        1,
        &ThermoOptions {
            concentration: Some(1.0),
            ..t.clone()
        },
    )
    .unwrap();
    let cst = Constants::CODATA2022;
    let ratio = (cst.kb * t.temperature / t.pressure) * 1000.0 * cst.na;
    assert!(
        (rrho.translational.entropy - m.translational.entropy - cst.kb_hartree() * ratio.ln())
            .abs()
            < 1e-15
    );
}
