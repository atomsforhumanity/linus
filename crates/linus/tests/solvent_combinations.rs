//! Implicit solvent (PCM) combined with the other features, against finite differences: KS
//! Hessians (B3LYP, wB97X; HF is covered by linus_solvent's own tests), VV10 gradients,
//! RIJ+COSX gradients and ECP gradients (the response properties, polarizability, dipole
//! derivatives and IR intensities, are linus_solvent's own tests: their finite fields run the
//! crate's SCF stage). A merge that drops the solvent term from one of these code paths fails
//! here.

mod common;

use common::{fd_gradient, fd_hessian, flat, max_diff, pcm, HI, WATER};
use linus::Structure;
use linus::{CalcOptions, GridChoice, JkMethod, Session};
use linus_solvent::PcmModel;
use std::time::Instant;

fn df() -> JkMethod {
    JkMethod::Df {
        aux: "def2-universal-jkfit".into(),
    }
}

fn opts(method: &str, jk: JkMethod, model: PcmModel) -> CalcOptions {
    CalcOptions {
        method: Some(method.into()),
        basis: Some("def2-svp".into()),
        jk: jk.into(),
        // an XC grid only for DFT (D41: Hartree-Fock has none)
        grid: linus::driver::options::GridSettings {
            xc: (method != "hf").then_some(GridChoice::Pyscf { level: 1 }),
            ..Default::default()
        },
        threads: Some(4),
        solvent: pcm(model, 7.4257),
        scf: linus::driver::ScfSettings {
            e_tol: Some(linus::driver::options::Auto::Value(1e-11)),
            g_tol: Some(linus::driver::options::Auto::Value(1e-10)),
            ..Default::default()
        },
        ..Default::default()
    }
}

/// KS Hessians with a PCM against central differences of analytic gradients (without grid
/// response, the grid moving with the atoms; first-order grid terms in the Hessian): B3LYP RKS
/// exact with C-PCM, wB97X RKS RI (wK exchange) with IEF-PCM, B3LYP UKS RI (water cation) with
/// C-PCM.
///
/// Step 5e-4 bohr, not the gas-phase tests' 2e-3: the PCM switching functions have large higher
/// derivatives, so the O(h^4) truncation error of the differences dominates at larger steps
/// (B3LYP UKS cation, C-PCM, 110 points: 3.9e-6 at h = 4e-3, 1.9e-6 at 2e-3, 5.2e-7 at 1e-3,
/// 2.6e-9 at 5e-4 and 3.2e-9 at 2.5e-4; the same molecule in the gas phase: 2.0e-8 at 2e-3).
/// The tighter SCF (max |F_ai| < 1e-11) keeps the gradient noise below the tolerance at this step.
/// Measured here: 2.7e-9 (B3LYP), 1.4e-9 (wB97X), 2.6e-9 (B3LYP UKS).
#[test]
fn ks_pcm_hessians_match_finite_differences() {
    for (charge, mult, method, jk, model) in [
        (0, 1, "b3lyp", JkMethod::Exact, PcmModel::Cpcm),
        (0, 1, "wb97x", df(), PcmModel::Iefpcm),
        (1, 2, "b3lyp", df(), PcmModel::Cpcm),
    ] {
        let t = Instant::now();
        let mol = Structure::from_xyz_str(WATER, charge, mult).unwrap();
        let mut o = opts(method, jk.clone(), model);
        o.scf.e_tol = Some(linus::driver::options::Auto::Value(1e-12));
        o.scf.g_tol = Some(linus::driver::options::Auto::Value(1e-11));
        let mut calc = Session::new(mol.clone(), o).unwrap();
        let an = calc.hessian().unwrap().hessian;
        // the Hessian's grid response against gradients without one (one setting): a second session
        let mut o_fd = opts(method, jk.clone(), model);
        o_fd.grid.response = Some(false);
        o_fd.scf.e_tol = Some(linus::driver::options::Auto::Value(1e-12));
        o_fd.scf.g_tol = Some(linus::driver::options::Auto::Value(1e-11));
        let mut calc_fd = Session::new(mol.clone(), o_fd).unwrap();
        let fd = fd_hessian(&mut calc_fd, &mol.positions, 5e-4);
        let e = max_diff(&an, &fd);
        eprintln!(
            "{method} {jk:?} {model:?} mult {mult}: |H - H_fd| = {e:.2e} ({:.1} s)",
            t.elapsed().as_secs_f64()
        );
        assert!(
            e < 5e-8,
            "{method} {jk:?} {model:?} mult {mult}: |H - H_fd| = {e:e}"
        );
    }
}

/// VV10 (wB97X-V, wB97M-V) with C-PCM: the analytic gradient (grid response on, XC and NLC
/// grids moving with the atoms) against differences of SCF energies. Measured: 8.5e-11
/// (wB97X-V), 2.5e-9 (wB97M-V, a meta-GGA on the coarse level-1 grid).
#[test]
fn vv10_pcm_gradient_matches_finite_differences() {
    for method in ["wb97x-v", "wb97m-v"] {
        let t = Instant::now();
        let mol = Structure::from_xyz_str(WATER, 0, 1).unwrap();
        let mut o = opts(method, JkMethod::Exact, PcmModel::Cpcm);
        o.grid.nlc = Some(GridChoice::Pyscf { level: 1 });
        o.scf.e_tol = Some(linus::driver::options::Auto::Value(1e-12));
        let mut calc = Session::new(mol.clone(), o).unwrap();
        let g = flat(&calc.gradient().unwrap().gradient);
        let fd = fd_gradient(&mut calc, &mol.positions, 1e-3);
        let e = max_diff(&g, &fd);
        eprintln!(
            "{method} C-PCM: |g - g_fd| = {e:.2e} ({:.1} s)",
            t.elapsed().as_secs_f64()
        );
        assert!(e < 3e-8, "{method} C-PCM: |g - g_fd| = {e:e}");
    }
}

/// RIJ + seminumerical exchange (the RIJCOSX gradient code) with C-PCM, RKS and UKS: one
/// exchange grid without overlap fitting, so the energy is variational and the gradient its
/// exact derivative (as linus_jk::cosx::rij_cosx's own test, in the gas phase). Measured: 5.9e-11
/// (RKS), 1.2e-10 (UKS cation).
#[test]
fn rijcosx_pcm_gradient_matches_finite_differences() {
    for (charge, mult) in [(0, 1), (1, 2)] {
        let t = Instant::now();
        let mol = Structure::from_xyz_str(WATER, charge, mult).unwrap();
        let mut o = opts(
            "b3lyp",
            JkMethod::RijSnk {
                aux: "def2-universal-jfit".into(),
                k_grid: GridChoice::Linus { level: 1 },
            },
            PcmModel::Cpcm,
        );
        o.grid.xc = Some(GridChoice::Linus { level: 1 });
        o.scf.e_tol = Some(linus::driver::options::Auto::Value(1e-12));
        let mut calc = Session::new(mol.clone(), o).unwrap();
        let g = flat(&calc.gradient().unwrap().gradient);
        let fd = fd_gradient(&mut calc, &mol.positions, 1e-3);
        let e = max_diff(&g, &fd);
        eprintln!(
            "RIJ+COSX B3LYP C-PCM mult {mult}: |g - g_fd| = {e:.2e} ({:.1} s)",
            t.elapsed().as_secs_f64()
        );
        assert!(e < 3e-9, "RIJ+COSX C-PCM mult {mult}: |g - g_fd| = {e:e}");
    }
}

/// An ECP atom in a PCM (HI; the cavity sees the effective nuclear charges): HF exact and
/// B3LYP RI gradients against differences of SCF energies. Measured: 2.8e-10 (HF), 1.9e-10
/// (B3LYP).
#[test]
fn ecp_pcm_gradient_matches_finite_differences() {
    for (method, jk) in [("hf", JkMethod::Exact), ("b3lyp", df())] {
        let t = Instant::now();
        let mol = Structure::from_xyz_str(HI, 0, 1).unwrap();
        let mut o = opts(method, jk.clone(), PcmModel::Cpcm);
        o.scf.e_tol = Some(linus::driver::options::Auto::Value(1e-12));
        let mut calc = Session::new(mol.clone(), o).unwrap();
        assert!(
            calc.mol().ecp_core_electrons().iter().any(|&n| n > 0),
            "HI/def2-SVP has an ECP on I"
        );
        let g = flat(&calc.gradient().unwrap().gradient);
        let fd = fd_gradient(&mut calc, &mol.positions, 1e-3);
        let e = max_diff(&g, &fd);
        eprintln!(
            "HI {method} {jk:?} C-PCM: |g - g_fd| = {e:.2e} ({:.1} s)",
            t.elapsed().as_secs_f64()
        );
        assert!(e < 3e-9, "HI {method} {jk:?} C-PCM: |g - g_fd| = {e:e}");
    }
}

/// The dense PCM matrices count against `memory_gib`: a surface whose matrices cannot be built in
/// it is a typed resource error before anything is allocated, not an out-of-memory kill
/// (bench/robust/repro/pcm_memory_budget: ethanol with 5810 points per sphere was killed at a
/// 3 GB cap with memory_gb 1). Water with 5810 points per sphere: ~1.5e4 points, C-PCM ~3.6 GB
/// (S twice), IEF-PCM ~11 GB, so 1 GB refuses both; 110 points per sphere fit.
#[test]
fn pcm_matrices_count_against_memory_gb() {
    let mol = Structure::from_xyz_str(WATER, 0, 1).unwrap();
    for model in [PcmModel::Cpcm, PcmModel::Iefpcm] {
        let mut o = opts("hf", df(), model);
        o.memory_gib = Some(linus::driver::options::Auto::Value(1.0));
        o.solvent.as_mut().unwrap().lebedev = Some(linus::driver::options::Auto::Value(5810));
        let mut calc = Session::new(mol.clone(), o).unwrap();
        let err = calc.energy().expect_err("a 5810-point surface in 1 GB");
        assert_eq!(err.kind(), "resource", "{model:?}: {err}");
        assert!(
            err.to_string().contains("solvent.lebedev"),
            "{model:?}: {err}"
        );
        let mut o = opts("hf", df(), model);
        o.memory_gib = Some(linus::driver::options::Auto::Value(1.0));
        let mut calc = Session::new(mol.clone(), o).unwrap();
        calc.energy().expect("110 points per sphere fit in 1 GB");
    }
}
