use super::*;
use crate::IncoreJk;
use linus_base::linalg::max_abs;
use linus_grid::GridSpec;
use linus_ints::basis::BasisDef;
use linus_ints::{PairOptions, PairSet};

pub(crate) fn water() -> (BasisSet, Vec<u32>, Vec<[f64; 3]>) {
    let json = WATER_BASIS;
    let z = vec![8, 1, 1];
    let xyz = vec![[0.0, 0.0, 0.0], [0.0, 1.43, 1.11], [0.0, -1.43, 1.11]];
    (
        BasisSet::new(&BasisDef::from_bse_json(json).unwrap(), &z, &xyz, true).unwrap(),
        z,
        xyz,
    )
}

/// A symmetric positive semidefinite density.
pub(crate) fn density(n: usize, seed: f64) -> Mat<f64> {
    let c = Mat::from_fn(n, 5, |i, j| {
        ((i * 7 + j * 3) as f64 * seed).sin() / (1.0 + i as f64 * 0.2)
    });
    &c * c.transpose()
}

/// An atom-block-diagonal stand-in for the SAD densities: the blocks of `d` on each atom.
pub(crate) fn blocks(basis: &BasisSet, d: &Mat<f64>) -> Mat<f64> {
    let atom: Vec<usize> = basis
        .shells
        .iter()
        .flat_map(|s| std::iter::repeat_n(s.atom, s.nfunc()))
        .collect();
    Mat::from_fn(d.nrows(), d.ncols(), |i, j| {
        if atom[i] == atom[j] {
            d[(i, j)]
        } else {
            0.0
        }
    })
}

fn exact_k(basis: &BasisSet, d: &Mat<f64>) -> Mat<f64> {
    let pairs = PairSet::build(basis, PairOptions::default());
    IncoreJk::new(basis, &pairs)
        .compute(
            &JkRequest::new(std::slice::from_ref(d), false, true),
            Par::Seq,
        )
        .unwrap()
        .k
        .remove(0)
}

fn builder(level: u8, dscf: bool, fit: bool, d0: &Mat<f64>) -> GridK {
    let (basis, z, xyz) = water();
    let grid = MolecularGrid::build(&z, &xyz, &GridSpec::cosx(level)).unwrap();
    let cfg = GridKConfig {
        dscf,
        fit,
        thresh: 1e-14,
        ..Default::default()
    };
    GridK::new(&basis, grid, Some(d0.clone()), cfg, Par::Seq).unwrap()
}

/// Xbar^T W Xbar reproduces the analytic overlap.
#[test]
fn the_fit_reproduces_the_overlap() {
    let (basis, _, _) = water();
    let d0 = blocks(&basis, &density(basis.nbf, 0.37));
    let k = builder(1, false, true, &d0);
    let w = &k.grid.w;
    let wx = Mat::from_fn(k.xbar.nrows(), k.xbar.ncols(), |g, p| w[g] * k.xbar[(g, p)]);
    let s_fit = mm(k.xbar.transpose(), wx.as_ref(), Par::Seq);
    let n = basis.nbf;
    let mut s = Mat::<f64>::zeros(n, n);
    OneEPlan::new(&basis, DerivOrder::Zero)
        .overlap(&mut linus_base::linalg::ao_view_mut(s.as_mut()));
    let err = max_abs((&s_fit - &s).as_ref());
    assert!(err < 1e-11, "{err:e}");
}

/// The energy's derivative is K (finite differences in the density), and with dSCF the
/// energy at D = D0 is the exact 1/2 D0 . K\[D0\].
#[test]
fn k_is_the_derivative_of_the_energy() {
    let (basis, _, _) = water();
    let n = basis.nbf;
    let d = density(n, 0.37);
    let d0 = blocks(&basis, &density(n, 0.91));
    for (dscf, fit) in [(false, false), (true, false), (false, true), (true, true)] {
        let mut k = builder(1, dscf, fit, &d0);
        let b = k.build(std::slice::from_ref(&d), Par::Seq).unwrap();
        let h = 1e-5;
        for (p, q) in [(0, 0), (3, 7), (10, 2)] {
            let mut e = [0.0; 2];
            for (i, s) in [1.0, -1.0].iter().enumerate() {
                let mut dp = d.clone();
                dp[(p, q)] += s * h;
                if p != q {
                    dp[(q, p)] += s * h;
                }
                e[i] = k.build(std::slice::from_ref(&dp), Par::Seq).unwrap().energy[0];
            }
            let fd = (e[0] - e[1]) / (2.0 * h);
            let an = if p == q {
                b.k[0][(p, q)]
            } else {
                2.0 * b.k[0][(p, q)]
            };
            assert!((fd - an).abs() < 1e-7, "{dscf} {fit} ({p},{q}): {fd} {an}");
        }
        if dscf {
            let b0 = k.build(std::slice::from_ref(&d0), Par::Seq).unwrap();
            let k0 = exact_k(&basis, &d0);
            let e0 = 0.5 * dot(d0.as_ref(), k0.as_ref());
            assert!((b0.energy[0] - e0).abs() < 1e-11, "{} {e0}", b0.energy[0]);
        }
    }
}

/// Every variant converges to the exact exchange with the grid, and the compensations reduce
/// the error of the energy on a coarse grid.
#[test]
fn converges_to_exact_exchange() {
    let (basis, _, _) = water();
    let n = basis.nbf;
    let d = density(n, 0.37);
    let d0 = blocks(&basis, &d);
    let kx = exact_k(&basis, &d);
    let ex = 0.5 * dot(d.as_ref(), kx.as_ref());
    let mut errs = vec![];
    for level in [1u8, 3] {
        for (dscf, fit) in [(false, false), (true, false), (false, true), (true, true)] {
            let mut k = builder(level, dscf, fit, &d0);
            let b = k.build(std::slice::from_ref(&d), Par::Seq).unwrap();
            let de = b.energy[0] - ex;
            let dk = max_abs((&b.k[0] - &kx).as_ref());
            errs.push((level, dscf, fit, de, dk));
        }
    }
    for e in &errs {
        eprintln!("{e:?}");
    }
    let raw1 = errs[0].3.abs();
    let comp1 = errs[3].3.abs();
    assert!(comp1 < raw1, "{errs:?}");
    for e in &errs[4..] {
        assert!(e.3.abs() < 2e-4 && e.4 < 1e-3, "{errs:?}");
    }
}

const WATER_BASIS: &str = r#"{"name":"t","elements":{"8":{"electron_shells":[
  {"function_type":"gto","angular_momentum":[0],"exponents":["130.7","23.81","6.44"],"coefficients":[["0.154","0.535","0.444"]]},
  {"function_type":"gto","angular_momentum":[0],"exponents":["0.38"],"coefficients":[["1.0"]]},
  {"function_type":"gto","angular_momentum":[1],"exponents":["5.03","1.17","0.38"],"coefficients":[["0.156","0.607","0.392"]]},
  {"function_type":"gto","angular_momentum":[2],"exponents":["1.2"],"coefficients":[["1.0"]]}]},
  "1":{"electron_shells":[{"function_type":"gto","angular_momentum":[0],"exponents":["3.42","0.62"],"coefficients":[["0.15","0.53"]]},
  {"function_type":"gto","angular_momentum":[1],"exponents":["0.8"],"coefficients":[["1.0"]]}]}}}"#;

/// The analytic gradient of every variant against central finite differences of the energy
/// (densities fixed, the grid rebuilt at each geometry), for the Coulomb and a range-separated
/// operator.
#[test]
fn gradient_matches_finite_differences() {
    let (basis0, z, xyz) = water();
    let n = basis0.nbf;
    let d = density(n, 0.37);
    let d0 = blocks(&basis0, &density(n, 0.91));
    let def = BasisDef::from_bse_json(WATER_BASIS).unwrap();
    let ops = [
        Operator::Coulomb,
        Operator::from_libxc(0.157706, 0.842294, 0.3),
    ];
    for op in ops {
        for (dscf, fit) in [(false, false), (true, false), (false, true), (true, true)] {
            let make = |xyz: &[[f64; 3]]| {
                let basis = BasisSet::new(&def, &z, xyz, true).unwrap();
                let grid = MolecularGrid::build(&z, xyz, &GridSpec::cosx(1)).unwrap();
                let cfg = GridKConfig {
                    dscf,
                    fit,
                    thresh: 1e-15,
                    exact_thresh: 1e-15,
                    xop: op,
                    ..Default::default()
                };
                GridK::new(&basis, grid, Some(d0.clone()), cfg, Par::Seq).unwrap()
            };
            let mut k = make(&xyz);
            let pairs = PairSet::build(
                &k.basis,
                PairOptions::with_order(linus_ints::DerivOrder::One),
            );
            let kappa = [-1.3];
            let parts = k
                .gradient(std::slice::from_ref(&d), &kappa, &pairs, Par::Seq)
                .unwrap();
            let g = parts.total();
            let h = 1e-4;
            let mut worst: f64 = 0.0;
            for a in 0..3 {
                for c in 0..3 {
                    let mut e = [0.0; 2];
                    for (i, s) in [1.0, -1.0].iter().enumerate() {
                        let mut p = xyz.clone();
                        p[a][c] += s * h;
                        e[i] = kappa[0]
                            * make(&p)
                                .build(std::slice::from_ref(&d), Par::Seq)
                                .unwrap()
                                .energy[0];
                    }
                    let fd = (e[0] - e[1]) / (2.0 * h);
                    worst = worst.max((fd - g[a][c]).abs());
                }
            }
            eprintln!("{op:?} dscf {dscf} fit {fit}: max |fd - analytic| {worst:.2e}; {parts:?}");
            assert!(worst < 2e-7, "{op:?} {dscf} {fit}: {worst:e}");
        }
    }
}
