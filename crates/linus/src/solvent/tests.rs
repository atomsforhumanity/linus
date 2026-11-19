//! The implicit solvent in whole calculations (linus-solvent's continuum models through the
//! facade's settings and dispatch).

use super::*;
use faer::Mat;
use linus_ints::basis::BasisDef;
use linus_ints::BasisSet;
use linus_solvent::*;

fn water() -> (Vec<u32>, Vec<[f64; 3]>) {
    (
        vec![8, 1, 1],
        vec![
            [0.0, 0.05, 0.2226],
            [0.03, 1.4276, -0.8904],
            [-0.02, -1.4276, -0.8803],
        ],
    )
}

/// A symmetric, positive "density" (fixed in the AO basis).
fn density(n: usize) -> Vec<f64> {
    let mut d = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..=i {
            let v = if i == j {
                0.6
            } else {
                0.08 * (((i * 7 + j * 3) % 11) as f64 / 11.0 - 0.3)
                    / (1.0 + (i as f64 - j as f64).abs())
            };
            d[i * n + j] = v;
            d[j * n + i] = v;
        }
    }
    d
}

fn energy_grad(
    model: PcmModel,
    disc: Discretization,
    positions: &[[f64; 3]],
    d: &[f64],
) -> (f64, Vec<[f64; 3]>) {
    let so = SolventSettings {
        eps: Some(Some(7.4257)),
        model: Some(model),
        discretization: Some(crate::driver::options::Auto::Value(disc)),
        lebedev: Some(crate::driver::options::Auto::Value(110)),
        ..Default::default()
    };
    energy_grad_with(&so, positions, d)
}

fn energy_grad_with(
    so: &SolventSettings,
    positions: &[[f64; 3]],
    d: &[f64],
) -> (f64, Vec<[f64; 3]>) {
    let (z, _) = water();
    let def = BasisDef::load("def2-svp", None).unwrap();
    let basis = BasisSet::new(&def, &z, positions, true).unwrap();
    let pcm = PcmScf::new(so.resolve(&z).unwrap(), &z, positions, &basis).unwrap();
    pcm.fixed_density_gradient(&z, positions, d)
}

/// d f / d positions[x / 3][x % 3] by Richardson-extrapolated central differences (h, 2h;
/// error O(h^4)).
fn richardson(c0: &[[f64; 3]], x: usize, f: impl Fn(&[[f64; 3]]) -> f64) -> f64 {
    let h = 1e-4;
    let at = |s: f64| {
        let mut c = c0.to_vec();
        c[x / 3][x % 3] += s * h;
        f(&c)
    };
    (8.0 * (at(1.0) - at(-1.0)) - (at(2.0) - at(-2.0))) / (12.0 * h)
}

/// Fixed-density gradients of every model and switching scheme against central differences
/// of the energy (density fixed in the AO basis, which moves with the atoms).
#[test]
fn fixed_density_gradient_by_finite_differences() {
    let (_, c0) = water();
    let n = 24; // def2-SVP water
    let d = density(n);
    for model in [
        PcmModel::Cpcm,
        PcmModel::Cosmo,
        PcmModel::Iefpcm,
        PcmModel::Ssvpe,
    ] {
        for disc in [Discretization::Swig, Discretization::Iswig] {
            let (_, g) = energy_grad(model, disc, &c0, &d);
            let mut err: f64 = 0.0;
            for x in 0..9 {
                let fd = richardson(&c0, x, |c| energy_grad(model, disc, c, &d).0);
                err = err.max((fd - g[x / 3][x % 3]).abs());
            }
            assert!(
                err < 1e-8,
                "{model:?} {disc:?}: max |analytic - fd| = {err:e}"
            );
        }
    }
    // ORCA-like preset: per-sphere grids by point density, point-evaluated potential
    let so = SolventSettings {
        eps: Some(Some(7.4257)),
        preset: Some(PcmPreset::Orca),
        ..Default::default()
    };
    let (_, g) = energy_grad_with(&so, &c0, &d);
    let mut err: f64 = 0.0;
    for x in 0..9 {
        let fd = richardson(&c0, x, |c| energy_grad_with(&so, c, &d).0);
        err = err.max((fd - g[x / 3][x % 3]).abs());
    }
    assert!(err < 1e-8, "orca preset: max |analytic - fd| = {err:e}");
    let st = so.resolve(&[8, 1, 1]).unwrap();
    assert!(st.point_potential);
}

/// Explicit second derivatives and F^x at fixed density against central differences of the
/// fixed-density gradient and of the solvent Fock matrix.
#[test]
fn fixed_density_hessian_by_finite_differences() {
    let (z, c0) = water();
    let n = 24;
    let d = density(n);
    let def = BasisDef::load("def2-svp", None).unwrap();
    for model in [
        PcmModel::Cpcm,
        PcmModel::Cosmo,
        PcmModel::Iefpcm,
        PcmModel::Ssvpe,
    ] {
        for disc in [Discretization::Swig, Discretization::Iswig] {
            let so = SolventSettings {
                eps: Some(Some(7.4257)),
                model: Some(model),
                discretization: Some(crate::driver::options::Auto::Value(disc)),
                lebedev: Some(crate::driver::options::Auto::Value(110)),
                ..Default::default()
            };
            let make = |c: &[[f64; 3]]| {
                let basis = BasisSet::new(&def, &z, c, true).unwrap();
                PcmScf::new(so.resolve(&z).unwrap(), &z, c, &basis).unwrap()
            };
            let t = make(&c0).hessian_terms(&z, &c0, &d);
            let h = 1e-4;
            let (mut eh, mut ef): (f64, f64) = (0.0, 0.0);
            let dm = Mat::from_fn(n, n, |i, j| d[i * n + j]);
            for x in 0..9 {
                // gradient and Fock matrix at displacements -2h, -h, h, 2h
                let at = |s: f64| {
                    let mut c = c0.clone();
                    c[x / 3][x % 3] += s * h;
                    let mut p = make(&c);
                    let g = p.fixed_density_gradient(&z, &c, &d).1;
                    let v = p.energy_fock(&dm).1;
                    (g, v)
                };
                let (m2, m1, p1, p2) = (at(-2.0), at(-1.0), at(1.0), at(2.0));
                let rich =
                    |a2: f64, a1: f64, b1: f64, b2: f64| (8.0 * (b1 - a1) - (b2 - a2)) / (12.0 * h);
                for y in 0..9 {
                    let (k, l) = (y / 3, y % 3);
                    let fd = rich(m2.0[k][l], m1.0[k][l], p1.0[k][l], p2.0[k][l]);
                    eh = eh.max((fd - t.hess[x * 9 + y]).abs());
                }
                for i in 0..n {
                    for j in 0..n {
                        let fd = rich(m2.1[(i, j)], m1.1[(i, j)], p1.1[(i, j)], p2.1[(i, j)]);
                        ef = ef.max((fd - t.fx[x][(i, j)]).abs());
                    }
                }
            }
            assert!(
                eh < 1e-8,
                "{model:?} {disc:?}: Hessian max |analytic - fd| = {eh:e}"
            );
            assert!(
                ef < 1e-8,
                "{model:?} {disc:?}: F^x max |analytic - fd| = {ef:e}"
            );
        }
    }
}

/// PCM Hessians through `Session::hessian` (the hooks in crate::derivatives::hessian) with density
/// fitting, closed and open shell, against central differences of analytic gradients. A
/// merge that drops a hook (explicit terms, F^x or the response J/K) fails here.
#[test]
fn session_hessian_df_by_finite_differences() {
    use crate::driver::{CalcOptions, JkMethod, Session};
    use crate::structure::Structure;
    let (z, c0) = water();
    for (charge, mult, model) in [(0, 1, PcmModel::Cpcm), (1, 2, PcmModel::Iefpcm)] {
        let opts = CalcOptions {
            method: Some("hf".into()),
            basis: Some("def2-svp".into()),
            jk: JkMethod::Df {
                aux: "def2-universal-jkfit".into(),
            }
            .into(),
            threads: Some(4),
            solvent: Some(SolventSettings {
                eps: Some(Some(7.4257)),
                model: Some(model),
                lebedev: Some(crate::driver::options::Auto::Value(110)),
                ..Default::default()
            }),
            scf: crate::driver::ScfSettings {
                convergence: Some(crate::driver::options::Auto::Value(
                    crate::driver::ConvergencePreset::Extreme,
                )),
                ..Default::default()
            },
            ..Default::default()
        };
        let calc_at = |c: &[[f64; 3]]| {
            Session::new(
                Structure::new(z.clone(), c.to_vec(), charge, mult).unwrap(),
                opts.clone(),
            )
            .unwrap()
        };
        let hres = calc_at(&c0).hessian().unwrap();
        let h = 5e-4;
        let mut err: f64 = 0.0;
        for x in 0..9 {
            let mut cp = c0.clone();
            cp[x / 3][x % 3] += h;
            let mut cm = c0.clone();
            cm[x / 3][x % 3] -= h;
            let gp = calc_at(&cp).gradient().unwrap().gradient;
            let gm = calc_at(&cm).gradient().unwrap().gradient;
            for y in 0..9 {
                let fd = (gp[y / 3][y % 3] - gm[y / 3][y % 3]) / (2.0 * h);
                err = err.max((fd - hres.hessian[x * 9 + y]).abs());
            }
        }
        assert!(
            err < 2e-6,
            "{model:?} charge {charge}: max |H - fd(grad)| = {err:e}"
        );
    }
}

/// The SCF modes of the surface terms through `Session` (closed and open shell, C-PCM
/// and IEF-PCM, RI-JK): `predicted` (the default with density-fitted J) converges to the
/// exact PCM solution (energies and gradients equal to `exact`); density-fitted PCM gradients
/// equal central differences of the density-fitted PCM energy, whose error against the exact
/// surface integrals is a few percent of the solvation energy; both give the same bits with 1 and
/// 4 threads; the density-fitted PCM is refused for Hessians.
#[test]
fn scf_modes_energy_gradient() {
    use crate::driver::{CalcOptions, GridChoice, JkMethod, Session};
    use crate::structure::Structure;
    let (z, c0) = water();
    let mk = |c: &[[f64; 3]],
              charge: i32,
              mult: u32,
              model: PcmModel,
              scf: Option<PcmScfMode>,
              threads: usize| {
        let opts = CalcOptions {
            method: Some("hf".into()),
            basis: Some("def2-svp".into()),
            jk: JkMethod::Df {
                aux: "def2-universal-jkfit".into(),
            }
            .into(),
            threads: Some(threads),
            solvent: Some(SolventSettings {
                eps: Some(Some(7.4257)),
                model: Some(model),
                lebedev: Some(crate::driver::options::Auto::Value(110)),
                df: (scf == Some(PcmScfMode::Df)).then_some(true),
                scf_terms: match scf {
                    Some(PcmScfMode::Exact) => Some(crate::driver::options::Auto::Value(
                        crate::solvent::PcmScfTerms::Exact,
                    )),
                    Some(PcmScfMode::Predicted) => Some(crate::driver::options::Auto::Value(
                        crate::solvent::PcmScfTerms::Predicted,
                    )),
                    _ => None,
                },
                ..Default::default()
            }),
            scf: crate::driver::ScfSettings {
                convergence: Some(crate::driver::options::Auto::Value(
                    crate::driver::ConvergencePreset::Extreme,
                )),
                ..Default::default()
            },
            ..Default::default()
        };
        Session::new(
            Structure::new(z.clone(), c.to_vec(), charge, mult).unwrap(),
            opts,
        )
        .unwrap()
    };
    let same_bits = |a: &crate::derivatives::gradient::GradientResult,
                     b: &crate::derivatives::gradient::GradientResult| {
        assert_eq!(a.energy.to_bits(), b.energy.to_bits());
        for (x, y) in a.gradient.iter().zip(&b.gradient) {
            for k in 0..3 {
                assert_eq!(x[k].to_bits(), y[k].to_bits());
            }
        }
    };
    for (charge, mult, model) in [(0, 1, PcmModel::Cpcm), (1, 2, PcmModel::Iefpcm)] {
        let mut exact = mk(&c0, charge, mult, model, Some(PcmScfMode::Exact), 4);
        let gx = exact.gradient().unwrap();
        let sx = exact.solvation().unwrap().electrostatic;
        assert_eq!(exact.solvation().unwrap().scf, PcmScfMode::Exact);
        // predicted (default with RI-JK): the exact solution
        let mut pred = mk(&c0, charge, mult, model, None, 4);
        let gp = pred.gradient().unwrap();
        assert_eq!(pred.solvation().unwrap().scf, PcmScfMode::Predicted);
        // the predicted charges make each build incremental; the converged iterate is confirmed
        // by an exact build at the same density (decision 25's one-build rule), so the SCF ends
        // on the exact PCM operator, as the exact mode's SCF (all of whose builds are full)
        let h = &pred.last().unwrap().history;
        let n = h.len();
        assert_eq!((h[n - 1].fock, h[n - 2].step), ("full", "rebuild"));
        assert!(h.iter().any(|it| it.fock == "incremental"));
        assert!(exact
            .last()
            .unwrap()
            .history
            .iter()
            .all(|it| it.fock == "full"));
        assert!(
            (gp.energy - gx.energy).abs() < 1e-10,
            "predicted E {} exact {}",
            gp.energy,
            gx.energy
        );
        assert!((pred.solvation().unwrap().electrostatic - sx).abs() < 1e-10);
        let dg = gp
            .gradient
            .iter()
            .zip(&gx.gradient)
            .flat_map(|(a, b)| (0..3).map(move |k| (a[k] - b[k]).abs()))
            .fold(0.0, f64::max);
        assert!(dg < 1e-8, "predicted gradient vs exact {dg:e}");
        same_bits(
            &mk(&c0, charge, mult, model, None, 1).gradient().unwrap(),
            &gp,
        );
        // The density-fitted PCM: a few percent of the solvation energy, gradient = derivative
        // of its energy
        let mut ri = mk(&c0, charge, mult, model, Some(PcmScfMode::Df), 4);
        let g = ri.gradient().unwrap();
        let sol = ri.solvation().unwrap();
        assert_eq!(sol.scf, PcmScfMode::Df);
        assert!(
            (sol.electrostatic - sx).abs() < 0.05 * sx.abs(),
            "E_solv RI {} exact {}",
            sol.electrostatic,
            sx
        );
        // energy components: the solvent part is out of the Coulomb and XC terms
        let r = ri.last().unwrap();
        let parts = r.e_one + r.e_coulomb + r.e_exchange + r.e_xc + r.e_nuc + sol.electrostatic;
        assert!(
            (parts - r.energy).abs() < 1e-9,
            "components {parts} vs {}",
            r.energy
        );
        let rx = exact.last().unwrap();
        assert!((r.e_coulomb - rx.e_coulomb).abs() < 1e-2 && (r.e_xc - rx.e_xc).abs() < 1e-9);
        let h = 5e-4;
        let mut err: f64 = 0.0;
        for x in 0..9 {
            let mut cp = c0.clone();
            cp[x / 3][x % 3] += h;
            let mut cm = c0.clone();
            cm[x / 3][x % 3] -= h;
            let ep = mk(&cp, charge, mult, model, Some(PcmScfMode::Df), 4)
                .energy()
                .unwrap()
                .energy;
            let em = mk(&cm, charge, mult, model, Some(PcmScfMode::Df), 4)
                .energy()
                .unwrap()
                .energy;
            err = err.max(((ep - em) / (2.0 * h) - g.gradient[x / 3][x % 3]).abs());
        }
        assert!(
            err < 2e-7,
            "{model:?} charge {charge}: the density-fitted PCM max |G - fd| = {err:e}"
        );
        same_bits(
            &mk(&c0, charge, mult, model, Some(PcmScfMode::Df), 1)
                .gradient()
                .unwrap(),
            &g,
        );
    }
    // RIJ+COSX: predicted by default; Hessians refuse the density-fitted PCM
    let opts = CalcOptions {
        method: Some("hf".into()),
        basis: Some("def2-svp".into()),
        jk: JkMethod::RijSnk {
            aux: "def2-universal-jfit".into(),
            k_grid: GridChoice::Linus { level: 1 },
        }
        .into(),
        threads: Some(2),
        solvent: Some(SolventSettings {
            eps: Some(Some(7.4257)),
            lebedev: Some(crate::driver::options::Auto::Value(110)),
            ..Default::default()
        }),
        ..Default::default()
    };
    let mut c = Session::new(Structure::new(z.clone(), c0.clone(), 0, 1).unwrap(), opts).unwrap();
    c.energy().unwrap();
    assert_eq!(c.solvation().unwrap().scf, PcmScfMode::Predicted);
    // The density-fitted PCM has no response terms: refused before any SCF, as not implemented
    let mut ri = mk(&c0, 0, 1, PcmModel::Cpcm, Some(PcmScfMode::Df), 2);
    let e = ri.hessian().expect_err("an error");
    assert!(matches!(e, crate::LinusError::Unsupported(_)), "{e}");
    assert!(ri.last().is_none());
    // settings that cannot work fail in Session::new: SMD without a solvent name,
    // predicted without density-fitted J
    let new = |so: SolventSettings, jk: JkMethod| {
        let opts = CalcOptions {
            basis: Some("def2-svp".into()),
            jk: jk.into(),
            solvent: Some(so),
            ..Default::default()
        };
        Session::new(Structure::new(z.clone(), c0.clone(), 0, 1).unwrap(), opts)
            .err()
            .expect("an error")
    };
    let smd = SolventSettings {
        eps: Some(Some(78.4)),
        model: Some(PcmModel::Smd),
        ..Default::default()
    };
    assert!(new(smd, JkMethod::Exact).is_input());
    let predicted = SolventSettings {
        eps: Some(Some(78.4)),
        scf_terms: Some(crate::driver::options::Auto::Value(
            crate::solvent::PcmScfTerms::Predicted,
        )),
        ..Default::default()
    };
    assert!(new(predicted, JkMethod::Exact).is_input());
}

/// Static polarizability by finite fields with the PCM's reaction field in every SCF: dipoles
/// at fields ±f along each axis (alpha_cd = d mu_d / d F_c).
fn finite_field_polarizability(
    calc: &crate::driver::Session,
    restricted: bool,
    f: f64,
) -> [[f64; 3]; 3] {
    use linus_base::linalg::{dot, mat_from_rowmajor};
    use linus_base::terms::XcPotential;
    use linus_core::timing::Timers;
    use linus_ints::{PairOptions, PairSet};
    use linus_scf::{HostLoop, ScfConfig, ScfInput};
    let par = calc.par();
    let mut t = Timers::default();
    let pairs = PairSet::build(calc.basis(), PairOptions::default());
    let (s, h) = calc.one_electron(&pairs);
    let mut jk = calc.build_jk(pairs, &mut t).unwrap().0;
    let mut xc = calc.build_xc(&mut t).unwrap();
    let n = calc.basis().nbf;
    let r: Vec<Mat<f64>> = linus_ints::reference::oneshot::dipole(calc.basis(), [0.0; 3])
        .iter()
        .map(|x| mat_from_rowmajor(n, n, x))
        .collect();
    let (na, nb) = calc.mol().nalpha_nbeta().unwrap();
    let mut so = ScfConfig {
        restricted: Some(restricted),
        max_iter: 200,
        ..Default::default()
    };
    so.e_tol = 1e-12;
    so.g_tol = 1e-10;
    let last = calc.last().unwrap();
    let setup = calc.ground.as_ref().unwrap().setup.clone();
    let mut alpha = [[0.0; 3]; 3];
    for c in 0..3 {
        let mut mu = [[0.0; 3]; 2];
        for (k, sg) in [1.0, -1.0].into_iter().enumerate() {
            let hf = &h + &r[c] * (sg * f);
            let inp = ScfInput {
                h: hf.as_ref(),
                s: s.as_ref(),
                e_nuc: calc.mol().nuclear_repulsion(),
                nalpha: na,
                nbeta: nb,
                exchange: calc.method().exchange,
                guess_density: Some((last.d_a.clone(), last.d_b.clone())),
                geometry: None,
            };
            let (rr, solv) = calc
                .cx
                .scf_on(
                    &setup,
                    &mut HostLoop {
                        inp: &inp,
                        jk: jk.as_mut(),
                        xc: xc.as_mut().map(|x| x as &mut dyn XcPotential),
                        opts: &so,
                        par,
                    },
                )
                .unwrap();
            assert!(rr.converged && solv.is_some());
            let dt = &rr.d_a + &rr.d_b;
            for d in 0..3 {
                mu[k][d] = -dot(dt.as_ref(), r[d].as_ref());
            }
        }
        for d in 0..3 {
            alpha[c][d] = (mu[0][d] - mu[1][d]) / (2.0 * f);
        }
    }
    alpha
}

/// Response properties with a PCM through `Session::hessian_analysis` (ir): the polarizability
/// against finite fields, the dipole derivatives against central differences of the SCF dipole
/// moment (the reaction field relaxing at every geometry), and the IR intensities against those
/// dipole derivatives along the normal modes. B3LYP RKS RI with C-PCM (first-order grid terms,
/// as the differences move the grid) and UHF exact (water cation) with IEF-PCM. Measured:
/// polarizability 3.7e-6 / 8.4e-7 a.u. (field 1e-4), dipole derivatives 3.4e-7 / 1.5e-7
/// (step 1e-3 bohr, two-point differences).
#[test]
#[allow(clippy::needless_range_loop, reason = "indexes several arrays in step")]
fn pcm_polarizability_and_ir_intensities() {
    use crate::driver::{CalcOptions, GridChoice, JkMethod, Session};
    use crate::structure::Structure;
    use std::time::Instant;
    // water without symmetry (Å)
    const WATER: &str = "3\n\nO 0.000 0.000 0.118\nH 0.020 0.755 -0.471\nH -0.010 -0.760 -0.465\n";
    let flat = |g: &[[f64; 3]]| -> Vec<f64> { g.iter().flatten().copied().collect() };
    let max_diff = |a: &[f64], b: &[f64]| {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f64::max)
    };
    let df = || JkMethod::Df {
        aux: "def2-universal-jkfit".into(),
    };
    for (charge, mult, method, jk, model) in [
        (0, 1, "b3lyp", df(), PcmModel::Cpcm),
        (1, 2, "hf", JkMethod::Exact, PcmModel::Iefpcm),
    ] {
        let t = Instant::now();
        let mol = Structure::from_xyz_str(WATER, charge, mult).unwrap();
        let o = CalcOptions {
            method: Some(method.into()),
            basis: Some("def2-svp".into()),
            jk: jk.clone().into(),
            // an XC grid only for DFT (D41: Hartree-Fock has none)
            grid: crate::driver::GridSettings {
                xc: (method != "hf").then_some(GridChoice::Pyscf { level: 2 }),
                ..Default::default()
            },
            threads: Some(4),
            solvent: Some(SolventSettings {
                eps: Some(Some(7.4257)),
                model: Some(model),
                lebedev: Some(crate::driver::Auto::Value(110)),
                ..Default::default()
            }),
            scf: crate::driver::ScfSettings {
                e_tol: Some(crate::driver::Auto::Value(1e-12)),
                g_tol: Some(crate::driver::Auto::Value(1e-10)),
                ..Default::default()
            },
            ..Default::default()
        };
        let mut calc = Session::new(mol.clone(), o).unwrap();
        let freq = calc
            .hessian_analysis(&linus_vib::FrequencyOptions {
                ir: true,
                ..Default::default()
            })
            .unwrap();
        let dd = freq.dipole_derivatives.as_ref().unwrap();
        // polarizability
        let ff = finite_field_polarizability(&calc, mult == 1, 1e-4);
        let ea = (0..9)
            .map(|i| (ff[i / 3][i % 3] - dd.polarizability[i / 3][i % 3]).abs())
            .fold(0.0, f64::max);
        // dipole derivatives
        let h = 1e-3;
        let mut fd = vec![[0.0; 3]; 9];
        for (x, row) in fd.iter_mut().enumerate() {
            let mut mu = Vec::new();
            for s in [1.0, -1.0] {
                let mut c = mol.positions.clone();
                c[x / 3][x % 3] += s * h;
                calc.set_positions(&c).unwrap();
                let r = calc.energy().unwrap();
                mu.push(calc.dipole_moment(&r.d_a, &r.d_b));
            }
            for k in 0..3 {
                row[k] = (mu[0][k] - mu[1][k]) / (2.0 * h);
            }
        }
        let ed = max_diff(&flat(&dd.apt), &flat(&fd));
        // IR intensities: proportional to |sum_j dmu/dx_j w_j|^2 (w: Cartesian mode / sqrt(mu))
        let v = freq.vib.as_ref().unwrap();
        let ir = v.ir_intensities.as_ref().unwrap();
        let mut ratios = Vec::new();
        for (k, mode) in v.modes_cartesian.iter().enumerate() {
            let mut dq = [0.0; 3];
            for (j, m) in mode.iter().enumerate() {
                for c in 0..3 {
                    dq[c] += dd.apt[j][c] * m / v.reduced_masses[k].sqrt();
                }
            }
            let q2: f64 = dq.iter().map(|x| x * x).sum();
            assert!(ir[k] >= 0.0 && ir[k].is_finite());
            if q2 > 1e-6 {
                ratios.push(ir[k] / q2);
            }
        }
        let spread = ratios
            .iter()
            .fold(0.0f64, |a, r| a.max((r / ratios[0] - 1.0).abs()));
        eprintln!(
            "{method} {jk:?} {model:?} mult {mult}: |alpha - ff| = {ea:.2e}, |dmu/dx - fd| = {ed:.2e}, IR ratio spread {spread:.1e} ({:.1} s)",
            t.elapsed().as_secs_f64()
        );
        assert!(
            ea < 3e-5,
            "{method} {model:?}: |alpha - finite field| = {ea:e} (alpha {:?})",
            dd.polarizability
        );
        assert!(ed < 3e-6, "{method} {model:?}: |dmu/dx - fd| = {ed:e}");
        assert!(
            !ratios.is_empty() && spread < 1e-10,
            "{method} {model:?}: IR intensities vs dipole derivatives, spread {spread:e}"
        );
    }
}

#[test]
fn options_parse_and_resolve() {
    let o: SolventSettings = serde_json::from_str(
        r#"{"eps": 7.4257, "model": "IEF-PCM", "radii": {"H": 1.2, "8": 1.6}, "radii_scale": 1.0}"#,
    )
    .unwrap();
    assert_eq!(o.model(), PcmModel::Iefpcm);
    let s = o.resolve(&[8, 1, 6]).unwrap();
    assert!((s.radii[0] - 1.6 / linus_core::units::BOHR_ANGSTROM_PYSCF).abs() < 1e-14);
    assert!((s.radii[1] - 1.2 / linus_core::units::BOHR_ANGSTROM_PYSCF).abs() < 1e-14);
    assert!((s.radii[2] - 1.7 / linus_core::units::BOHR_ANGSTROM_PYSCF).abs() < 1e-14);
    assert!(serde_json::from_str::<SolventSettings>(r#"{"eps": 2, "model": "pcmx"}"#).is_err());
    assert!(
        "ss(v)pe".parse::<PcmModel>().unwrap() == PcmModel::Ssvpe
            && "C-PCM".parse::<PcmModel>().unwrap() == PcmModel::Cpcm
    );
    // SMD: IEF-PCM (C-PCM with the orca preset), SMD's Coulomb radii unscaled, the database's eps
    let o: SolventSettings = serde_json::from_str(r#"{"name": "water", "model": "smd"}"#).unwrap();
    let s = o.resolve(&[8, 1, 1]).unwrap();
    let eps = db::lookup("water").unwrap().eps;
    assert!(
        s.model == PcmModel::Iefpcm && s.eps == eps && s.lebedev == vec![590; 3] && s.smd.is_some()
    );
    assert!((s.radii[1] - 1.20 / linus_core::units::BOHR_ANGSTROM_PYSCF).abs() < 1e-14);
    let o: SolventSettings =
        serde_json::from_str(r#"{"name": "water", "model": "smd", "preset": "orca"}"#).unwrap();
    assert_eq!(o.resolve(&[8, 1, 1]).unwrap().model, PcmModel::Cpcm);
    assert!(
        serde_json::from_str::<SolventSettings>(r#"{"eps": 7.0, "model": "smd"}"#)
            .unwrap()
            .resolve(&[8])
            .is_err()
    );
}
