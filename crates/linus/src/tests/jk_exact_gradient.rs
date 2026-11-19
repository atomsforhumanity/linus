//! linus-jk's exact two-electron gradient against whole calculations
//! (linus_jk::exact::gradient's tests that run the Session).

use crate::driver::{CalcOptions, JkMethod, Session};
use crate::Structure;
use linus_core::gradient::Gradient;
use linus_ints::contract::DensityTerms;
use linus_ints::operator::Operator;
use linus_ints::{BasisSet, DerivOrder, PairOptions, PairSet};
use linus_jk::exact::gradient::{exact_gradient, grad_4c};
use linus_jk::exact::terms::SpinDensities;

const WATER: &str = "3\nwater\nO 0.10 -0.05 0.12\nH 0.85 0.52 -0.03\nH -0.68 0.55 0.05\n";

fn opts(method: &str, jk: JkMethod) -> CalcOptions {
    CalcOptions {
        method: Some(method.into()),
        basis: Some("def2-svp".into()),
        jk: jk.into(),
        threads: Some(2),
        scf: crate::driver::ScfSettings {
            convergence: Some(crate::driver::options::Auto::Value(
                crate::driver::ConvergencePreset::Extreme,
            )),
            ..Default::default()
        },
        ..Default::default()
    }
}

/// Richardson central differences of driver energies for a few coordinates.
fn fd_check(mol: &Structure, o: &CalcOptions, coords: &[(usize, usize)], tol: f64) {
    let g = Session::new(mol.clone(), o.clone())
        .unwrap()
        .gradient()
        .unwrap()
        .gradient;
    let h = 2e-3;
    for &(a, k) in coords {
        let e = |d: f64| {
            let mut m = mol.clone();
            m.positions[a][k] += d;
            Session::new(m, o.clone()).unwrap().energy().unwrap().energy
        };
        let fd = (8.0 * (e(h) - e(-h)) - (e(2.0 * h) - e(-2.0 * h))) / (12.0 * h);
        assert!(
            (fd - g[a][k]).abs() < tol,
            "{} {:?} ({a},{k}): analytic {} fd {fd}",
            o.method(),
            o.jk,
            g[a][k]
        );
    }
}

#[test]
fn exact_and_df_gradients_match_finite_differences() {
    let mol = Structure::from_xyz_str(WATER, 0, 1).unwrap();
    let df = JkMethod::Df {
        aux: "def2-universal-jkfit".into(),
    };
    fd_check(
        &mol,
        &opts("wb97x", JkMethod::Exact),
        &[(0, 0), (1, 2)],
        2e-9,
    );
    fd_check(
        &mol,
        &opts("cam-b3lyp", df.clone()),
        &[(0, 1), (2, 0)],
        2e-9,
    );
    // pure long-range exchange (alpha + beta = 0)
    fd_check(
        &mol,
        &opts("HYB_GGA_XC_LRC_WPBE", df.clone()),
        &[(1, 1)],
        2e-9,
    );
    let trip = Structure::from_xyz_str(WATER, 0, 3).unwrap();
    fd_check(&trip, &opts("wb97x", df), &[(0, 2), (2, 1)], 2e-9);
}

/// RIJ + seminumerical K on one grid without overlap fitting is variational: FD-exact.
#[test]
fn rij_snk_gradient_matches_finite_differences() {
    use crate::driver::GridChoice;
    let mol = Structure::from_xyz_str(WATER, 0, 1).unwrap();
    let jk = JkMethod::RijSnk {
        aux: "def2-universal-jkfit".into(),
        k_grid: GridChoice::Pyscf { level: 2 },
    };
    fd_check(&mol, &opts("wb97x", jk.clone()), &[(0, 2), (1, 0)], 5e-9);
    let trip = Structure::from_xyz_str(WATER, 0, 3).unwrap();
    fd_check(&trip, &opts("HYB_GGA_XC_LRC_WPBE", jk), &[(2, 1)], 5e-9);
}

/// The exact gradient split as J + a K with 1/r and b K_LR with erf(w r)/r (reference for
/// the combined-operator exchange pass of [`exact_gradient`]).
#[allow(clippy::too_many_arguments)]
fn exact_gradient_two_pass(
    basis: &BasisSet,
    pairs: &PairSet,
    dt: &[f64],
    da: &[f64],
    db: &[f64],
    restricted: bool,
    x: Operator,
    thresh: f64,
) -> Vec<[f64; 3]> {
    let spins = SpinDensities::new(restricted, dt, da, db);
    let mut jk = spins.coulomb(DensityTerms::default());
    if x.coulomb() != 0.0 {
        jk = spins.exchange(jk, x.coulomb());
    }
    let mut ws = None;
    let g = grad_4c(
        basis,
        pairs,
        &jk,
        thresh,
        &linus_ints::operator::Operator::Coulomb,
        &mut ws,
    );
    if x.long_range() == 0.0 {
        return g;
    }
    let lr = spins.exchange(DensityTerms::default(), x.long_range());
    let mut g = Gradient::from(g);
    g += grad_4c(
        basis,
        pairs,
        &lr,
        thresh,
        &Operator::LongRange { omega: x.omega() },
        &mut ws,
    );
    g.into_inner()
}

#[test]
fn exact_split_variants_agree() {
    let mol = Structure::from_xyz_str(WATER, 0, 1).unwrap();
    let mut c = Session::new(mol, opts("wb97x", JkMethod::Exact)).unwrap();
    let r = c.energy().unwrap();
    let b = &c.cx.basis;
    let n = b.nbf;
    let pairs = PairSet::build(b, PairOptions::with_order(DerivOrder::One));
    let dt: Vec<f64> = (0..n * n)
        .map(|x| r.d_a[(x / n, x % n)] + r.d_b[(x / n, x % n)])
        .collect();
    let x = Operator::from_libxc(1.0, -0.842294, 0.3);
    let g1 = exact_gradient(b, &pairs, &dt, &dt, &dt, true, x, 1e-14);
    let g2 = exact_gradient_two_pass(b, &pairs, &dt, &dt, &dt, true, x, 1e-14);
    for (u, v) in g1.iter().zip(&g2) {
        for k in 0..3 {
            assert!((u[k] - v[k]).abs() < 1e-12, "{u:?} {v:?}");
        }
    }
}
