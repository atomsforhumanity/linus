//! linus-dft's VV10 response terms in whole calculations and on the device (from
//! linus_dft::vv10_resp's tests: they run the Session).

use faer::{Mat, MatRef, Par};
use linus_base::linalg::max_abs;
use linus_dft::vv10_resp::*;

/// Driver level: analytic Hessians (default first-order grid response) against fourth-order
/// differences of analytic gradients without grid response on grids moving with the atoms
/// (symmetrized: the scheme-1 differences are not symmetric), and the dipole derivatives
/// (the Hessian's response, VV10 kernel included) against central differences of the SCF
/// dipole: water wB97M-V (RKS, exact J/K) and triplet CH2 B97M-V (UKS, RI).
#[test]
fn driver_hessian_matches_finite_differences() {
    use crate::derivatives::hessian::HessianOptions;
    use crate::driver::{CalcOptions, GridChoice, JkMethod, Session};
    let water = "3\n\nO 0.000 0.000 0.118\nH 0.020 0.755 -0.471\nH -0.010 -0.760 -0.465\n";
    let ch2 = "3\n\nC 0.000 0.003 0.102\nH 0.000 0.995 -0.310\nH 0.010 -0.990 -0.302\n";
    let df = JkMethod::Df {
        aux: "def2-universal-jkfit".into(),
    };
    for (xyz, mult, method, jk) in [
        (water, 1, "wb97m-v", JkMethod::Exact),
        (ch2, 3, "b97m-v", df),
    ] {
        let mol = crate::Structure::from_xyz_str(xyz, 0, mult).unwrap();
        let o = CalcOptions {
            method: Some(method.into()),
            basis: Some("def2-svp".into()),
            jk: jk.into(),
            grid: crate::driver::options::GridSettings {
                xc: Some(GridChoice::Pyscf { level: 1 }),
                nlc: Some(GridChoice::Pyscf { level: 0 }),
                ..Default::default()
            },
            threads: Some(4),
            scf: crate::driver::ScfSettings {
                e_tol: Some(crate::driver::options::Auto::Value(1e-12)),
                g_tol: Some(crate::driver::options::Auto::Value(1e-11)),
                ..Default::default()
            },
            ..Default::default()
        };
        let mut calc = Session::new(mol.clone(), o).unwrap();
        let r = calc
            .hessian_with(&HessianOptions {
                dipole_derivatives: true,
            })
            .unwrap();
        let (an, dip) = (r.hessian, r.dipole_derivatives.unwrap());
        // the Hessian's grid response against gradients without one (one setting now)
        calc.cx.plan.request_mut().grid.response = Some(false);
        let n3 = 9;
        let h = 2e-3;
        let mut fd = vec![0.0; n3 * n3];
        let mut edip: f64 = 0.0;
        for y in 0..n3 {
            let (mut g, mut mu) = (Vec::new(), Vec::new());
            for s in [1.0, -1.0, 2.0, -2.0] {
                let mut c = mol.positions.clone();
                c[y / 3][y % 3] += s * h;
                calc.set_positions(&c).unwrap();
                g.push(calc.gradient().unwrap().gradient);
                let l = calc.last().unwrap();
                mu.push(calc.dipole_moment(&l.d_a, &l.d_b));
            }
            for x in 0..n3 {
                let d = |i: usize| g[i][x / 3][x % 3];
                fd[x * n3 + y] = (8.0 * (d(0) - d(1)) - (d(2) - d(3))) / (12.0 * h);
            }
            for k in 0..3 {
                let f = (8.0 * (mu[0][k] - mu[1][k]) - (mu[2][k] - mu[3][k])) / (12.0 * h);
                edip = edip.max((f - dip[y][k]).abs());
            }
        }
        let mut eh: f64 = 0.0;
        for x in 0..n3 {
            for y in 0..n3 {
                let s = 0.5 * (fd[x * n3 + y] + fd[y * n3 + x]);
                eh = eh.max((an[x * n3 + y] - s).abs());
            }
        }
        eprintln!("{method} mult {mult}: |H - sym FD(grad)| {eh:e}, |dmu/dx - FD| {edip:e}");
        assert!(eh < 2e-7 && edip < 2e-6, "{method}: {eh:e} {edip:e}");
    }
}

/// The device pair sweeps (GPU backend, double-float and FP64) against the CPU's: the CPKS
/// kernel and the explicit Hessian terms (grid response included) of water wB97M-V at its
/// SCF density, closed and open shell; repeated device runs bitwise identical. Skips in a
/// build without GPU support, and without a device unless LINUS_GPU_REQUIRED=1.
#[test]
fn device_sweeps_match_the_cpu() {
    if !crate::backend::GPU_BUILT {
        return;
    }
    use crate::driver::{CalcOptions, Session};
    use linus_base::error::LinusError;
    use linus_core::timing::Timers;
    let water = "3\n\nO 0.000 0.000 0.118\nH 0.020 0.755 -0.471\nH -0.010 -0.760 -0.465\n";
    for (charge, mult, precision) in [(0, 1, "auto"), (1, 2, "auto"), (0, 1, "fp64")] {
        let mol = crate::Structure::from_xyz_str(water, charge, mult).unwrap();
        let o = CalcOptions::from_json(serde_json::json!({
            "method": "wb97m-v", "basis": "def2-svp", "jk": "df", "grid": {"xc": "linus:1", "nlc": "linus:1"},
            "threads": 4,
            "backend": "gpu", "precision": precision
        }))
        .unwrap();
        let mut c = match Session::new(mol, o) {
            Ok(c) => c,
            Err(LinusError::Backend { message, .. })
                if std::env::var("LINUS_GPU_REQUIRED").as_deref() != Ok("1") =>
            {
                eprintln!("no usable GPU, skipping: {message}");
                return;
            }
            Err(e) => panic!("{e}"),
        };
        c.energy().unwrap();
        let r = c.last().unwrap().clone();
        let co: Vec<Mat<f64>> = if r.restricted {
            vec![r.c_a.subcols(0, r.nalpha).to_owned()]
        } else {
            vec![
                r.c_a.subcols(0, r.nalpha).to_owned(),
                r.c_b.subcols(0, r.nbeta).to_owned(),
            ]
        };
        let cor: Vec<MatRef<'_, f64>> = co.iter().map(|x| x.as_ref()).collect();
        let dt = total_density(&cor, Par::Seq);
        let mut t = Timers::default();
        let grid = c.scf_grid();
        let xc = c.cx.xc_on(&grid, &mut t).unwrap().unwrap();
        let nl = xc.vv10().unwrap();
        let cpu = nl.response_ground(&dt, true);
        let dev = c.cx.vv10_ground(nl, &dt, true, "t", &mut t).unwrap();
        assert!(dev.device.is_some(), "the device sweeps were not set up");
        let n = dt.nrows();
        let a: Vec<Vec<Mat<f64>>> = co
            .iter()
            .enumerate()
            .map(|(s, x)| (0..5).map(|p| mat(n, x.ncols(), p + 3 * s, 0.05)).collect())
            .collect();
        let kc = nl.kernel_contract_occ(&cpu, &cor, &a).unwrap();
        let kd = nl.kernel_contract_occ(&dev, &cor, &a).unwrap();
        let kd2 = nl.kernel_contract_occ(&dev, &cor, &a).unwrap();
        assert!(kd == kd2, "repeated device sweeps differ");
        let rel = |x: &Mat<f64>, y: &Mat<f64>| {
            max_abs((x - y).as_ref()) / max_abs(x.as_ref()).max(1e-300)
        };
        let mut ek: f64 = 0.0;
        for (xs, ys) in kc.iter().zip(&kd) {
            for (x, y) in xs.iter().zip(ys) {
                ek = ek.max(rel(x, y));
            }
        }
        // the CPKS products in FP32 (each tile in FP32, the tiles in FP64)
        let kf = nl
            .kernel_contract_occ_timed(&dev, &cor, &a, true)
            .unwrap()
            .0;
        let mut ef: f64 = 0.0;
        for (xs, ys) in kc.iter().zip(&kf) {
            for (x, y) in xs.iter().zip(ys) {
                ef = ef.max(rel(x, y));
            }
        }
        let hc = nl
            .hessian_terms(&cpu, &cor, &dt, true, Par::Seq)
            .unwrap()
            .terms;
        let hd = nl
            .hessian_terms(&dev, &cor, &dt, true, Par::Seq)
            .unwrap()
            .terms;
        let mut ev: f64 = 0.0;
        for (xs, ys) in hc
            .vfix
            .iter()
            .zip(&hd.vfix)
            .chain(hc.vgrid.iter().zip(&hd.vgrid))
        {
            for (x, y) in xs.iter().zip(ys) {
                ev = ev.max(max_abs((x - y).as_ref()));
            }
        }
        let eh = hc
            .hess
            .iter()
            .zip(&hd.hess)
            .fold(0.0f64, |m, (x, y)| m.max((x - y).abs()));
        eprintln!(
            "device VV10 sweeps ({precision}, mult {mult}): kernel rel. {ek:e} (FP32 {ef:e}), V^x C {ev:e}, E^xy {eh:e}"
        );
        assert!(ek < 1e-10 && ef < 1e-5 && ev < 1e-11 && eh < 1e-11);
    }
}

fn mat(n: usize, k: usize, seed: usize, s: f64) -> Mat<f64> {
    Mat::from_fn(n, k, |i, j| {
        s * ((((i * 7 + j * 13 + seed * 5) % 17) as f64 - 8.0) / 8.0)
            * (-((i as f64 - 3.0 * j as f64).abs()) / 9.0).exp()
    })
}
