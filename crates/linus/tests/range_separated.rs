//! Range-separated exact exchange through the J/K builders and the driver: one-pass operators
//! (`JkBuilder::set_exchange_operator`) against K and K_LR built separately (K_LR: the builder
//! with the operator erf(omega r)/r), and driver energies of the J/K methods against each other.

use faer::{Mat, Par};
use linus::JkBuilder;
use linus::Structure;
use linus::{CalcOptions, GridChoice, JkMethod, Session};
use linus_base::terms::JkRequest;
use linus_grid::{GridSpec, MolecularGrid};
use linus_ints::basis::BasisDef;
use linus_ints::operator::Operator;
use linus_ints::BasisSet;
use linus_jk::cosx::{CosxConfig, CosxK};
use linus_jk::exact::FastJk;

const WATER: &str = "3\nwater\nO 0.10 -0.05 0.12\nH 0.85 0.52 -0.03\nH -0.68 0.55 0.05\n";
// wB97X: alpha = 1, beta = -0.842294, omega = 0.3 -> a K + b K_LR
const A: f64 = 0.157706;
const B: f64 = 0.842294;
const W: f64 = 0.3;

fn setup() -> (Structure, BasisSet, Vec<Mat<f64>>) {
    let mol = Structure::from_xyz_str(WATER, 0, 1).unwrap();
    let def = BasisDef::load("def2-svp", None).unwrap();
    let basis = BasisSet::new(&def, &mol.z, &mol.positions, true).unwrap();
    let n = basis.nbf;
    let c = Mat::from_fn(n, 5, |i, j| {
        ((i * 7 + j * 3) as f64 * 0.37).sin() / (1.0 + i as f64 * 0.2)
    });
    let c2 = Mat::from_fn(n, 4, |i, j| {
        ((i * 5 + j * 11) as f64 * 0.21).cos() / (1.0 + i as f64 * 0.3)
    });
    (mol, basis, vec![&c * c.transpose(), &c2 * c2.transpose()])
}

fn max_diff(a: &Mat<f64>, b: &Mat<f64>) -> f64 {
    let mut m: f64 = 0.0;
    for j in 0..a.ncols() {
        for i in 0..a.nrows() {
            m = m.max((a[(i, j)] - b[(i, j)]).abs());
        }
    }
    m
}

#[test]
#[allow(clippy::needless_range_loop, reason = "indexes several arrays in step")]
fn fastjk_one_pass_operator_equals_k_plus_k_lr() {
    let (_, basis, dens) = setup();
    let sep = FastJk::new(&basis, 1e-14)
        .compute(&JkRequest::new(&dens, true, true), Par::Seq)
        .unwrap();
    let mut lr = FastJk::new(&basis, 1e-14);
    assert!(lr.set_exchange_operator(Operator::LongRange { omega: W }));
    let klr = lr
        .compute(&JkRequest::new(&dens, false, true), Par::Seq)
        .unwrap()
        .k;
    let mut one = FastJk::new(&basis, 1e-14);
    assert!(one.set_exchange_operator(Operator::Attenuated {
        coulomb: A,
        long_range: B,
        omega: W
    }));
    let comb = one
        .compute(&JkRequest::new(&dens, true, true), Par::Seq)
        .unwrap();
    for d in 0..dens.len() {
        assert!(max_diff(&sep.j[d], &comb.j[d]) < 1e-12);
        let k = &sep.k[d] * A + &klr[d] * B;
        assert!(
            max_diff(&k, &comb.k[d]) < 1e-12,
            "{}",
            max_diff(&k, &comb.k[d])
        );
    }
}

#[test]
#[allow(clippy::needless_range_loop, reason = "indexes several arrays in step")]
fn rijk_one_pass_operator_equals_k_plus_k_lr() {
    let (mol, basis, dens) = setup();
    let def = BasisDef::load("def2-universal-jkfit", None).unwrap();
    let aux = BasisSet::new(&def, &mol.z, &mol.positions, true).unwrap();
    let pairs = linus_ints::PairSet::build(&basis, linus_ints::PairOptions::default());
    let mk = || {
        linus_jk::df::DfJk::new(
            &basis,
            &pairs,
            &aux,
            true,
            linus_jk::df::DfConfig {
                omega: Some(W),
                ..Default::default()
            },
            Par::Seq,
        )
    };
    let sep = mk()
        .compute(&JkRequest::new(&dens, true, true), Par::Seq)
        .unwrap();
    // K_LR in the two-pass form (stored B and B^omega, DfJk::exchange_long_range) of PSD factors
    let mut lr = mk();
    let fac: Vec<(Mat<f64>, Mat<f64>)> = dens
        .iter()
        .map(|d| linus_jk::df::factor_symmetric(d.as_ref(), Par::Seq))
        .collect();
    let klr: Vec<Mat<f64>> = fac
        .iter()
        .map(|(p, m)| {
            let (_, mut kp) = lr.exchange_long_range(
                &[linus_jk::df::Factored::Psd(p.as_ref())],
                false,
                Some(W),
                Par::Seq,
            );
            let mut k = kp.remove(0);
            if m.ncols() > 0 {
                k -= &lr
                    .exchange_long_range(
                        &[linus_jk::df::Factored::Psd(m.as_ref())],
                        false,
                        Some(W),
                        Par::Seq,
                    )
                    .1[0];
            }
            k
        })
        .collect();
    let mut one = mk();
    assert!(one.set_exchange_operator(Operator::Attenuated {
        coulomb: A,
        long_range: B,
        omega: W
    }));
    let comb = one
        .compute(&JkRequest::new(&dens, true, true), Par::Seq)
        .unwrap();
    for d in 0..dens.len() {
        let k = &sep.k[d] * A + &klr[d] * B;
        assert!(
            max_diff(&k, &comb.k[d]) < 1e-12,
            "{}",
            max_diff(&k, &comb.k[d])
        );
    }
}

#[test]
fn cosx_one_pass_operator_is_linear_in_the_operator() {
    let (mol, basis, dens) = setup();
    let grid = || MolecularGrid::build(&mol.z, &mol.positions, &GridSpec::cosx(2)).unwrap();
    let k_of = |op: Option<(f64, f64, f64)>| {
        let mut k = CosxK::new(
            &basis,
            grid(),
            CosxConfig {
                thresh: 1e-15,
                bf_cut: 1e-15,
                ..CosxConfig::default()
            },
        );
        if let Some((a, b, w)) = op {
            assert!(k.set_exchange_operator(Operator::Attenuated {
                coulomb: a,
                long_range: b,
                omega: w
            }));
        }
        k.compute(&JkRequest::new(&dens, false, true), Par::Seq)
            .unwrap()
            .k
    };
    let kc = k_of(None);
    let kl = k_of(Some((0.0, 1.0, W)));
    let comb = k_of(Some((A, B, W)));
    for d in 0..dens.len() {
        let k = &kc[d] * A + &kl[d] * B;
        assert!(max_diff(&k, &comb[d]) < 1e-12, "{}", max_diff(&k, &comb[d]));
    }
}

/// wB97X energies: exact, RI-JK and RIJ + seminumerical K agree to their approximation errors;
/// UKS of a closed shell equals RKS.
#[test]
fn driver_energies_of_jk_methods_agree() {
    let mol = Structure::from_xyz_str(WATER, 0, 1).unwrap();
    let e = |jk: JkMethod, restricted: Option<bool>| {
        let o = CalcOptions {
            method: Some("wb97x-d3bj".into()),
            basis: Some("def2-svp".into()),
            jk: jk.into(),
            reference: restricted.map(|r| {
                if r {
                    linus::driver::options::Reference::Restricted
                } else {
                    linus::driver::options::Reference::Unrestricted
                }
            }),
            threads: Some(2),
            scf: linus::driver::ScfSettings {
                convergence: Some(linus::driver::options::Auto::Value(
                    linus::driver::ConvergencePreset::VeryTight,
                )),
                ..Default::default()
            },
            ..Default::default()
        };
        Session::new(mol.clone(), o)
            .unwrap()
            .energy()
            .unwrap()
            .energy
    };
    let ex = e(JkMethod::Exact, None);
    let df = e(
        JkMethod::Df {
            aux: "def2-universal-jkfit".into(),
        },
        None,
    );
    let sn = e(
        JkMethod::RijSnk {
            aux: "def2-universal-jkfit".into(),
            k_grid: GridChoice::Pyscf { level: 5 },
        },
        None,
    );
    let cx = e(
        JkMethod::RijSnk {
            aux: "def2-universal-jkfit".into(),
            k_grid: GridChoice::Cosx { defgrid: 3 },
        },
        None,
    );
    let uks = e(JkMethod::Exact, Some(false));
    // jk "incore": exact integrals held in memory (exact.storage)
    let inc = {
        let o = CalcOptions::from_json(
            serde_json::json!({"method": "wb97x-d3bj", "basis": "def2-svp",
            "jk": "incore", "threads": 2, "scf": {"convergence": "verytight"}}),
        )
        .unwrap();
        assert_eq!(
            o.jk.exact_storage,
            Some(linus::driver::ExactStorage::Incore)
        );
        Session::new(mol.clone(), o)
            .unwrap()
            .energy()
            .unwrap()
            .energy
    };
    assert!((ex - inc).abs() < 1e-10, "exact {ex} incore {inc}");
    assert!((ex - df).abs() < 5e-5, "exact {ex} df {df}");
    assert!((df - sn).abs() < 1e-5, "df {df} rij+snk {sn}");
    assert!((df - cx).abs() < 1e-4, "df {df} rij+cosx {cx}");
    assert!((ex - uks).abs() < 1e-9, "rks {ex} uks {uks}");
}

/// Analytic Hessians of range-separated hybrids through the driver (validated in
/// linus::derivatives::hessian, hess_df, hess_resp and tests/python/compare_hess_pyscf.py): exact and RI
/// agree to the RI error, and both are symmetric and translationally invariant (first-order
/// grid response: up to the grid's second-order terms).
#[test]
fn hessians_of_range_separated_hybrids() {
    let mol = Structure::from_xyz_str(WATER, 0, 1).unwrap();
    let run = |jk: JkMethod| {
        let o = CalcOptions {
            method: Some("wb97x".into()),
            basis: Some("def2-svp".into()),
            jk: jk.into(),
            grid: linus::driver::options::GridSettings {
                xc: Some(GridChoice::Pyscf { level: 3 }),
                ..Default::default()
            },
            threads: Some(4),
            ..Default::default()
        };
        Session::new(mol.clone(), o)
            .unwrap()
            .hessian()
            .unwrap()
            .hessian
    };
    let (he, hd) = (
        run(JkMethod::Exact),
        run(JkMethod::Df {
            aux: "def2-universal-jkfit".into(),
        }),
    );
    let n3 = 9;
    for h in [&he, &hd] {
        for i in 0..n3 {
            let mut s = [0.0; 3];
            for j in 0..n3 {
                assert!((h[i * n3 + j] - h[j * n3 + i]).abs() < 1e-12);
                s[j % 3] += h[i * n3 + j];
            }
            // the grid lacks its second-order terms: ~1e-4 on PySCF level 3 (|H| ~ 0.7)
            assert!(
                s.iter().all(|x| x.abs() < 1e-3),
                "translational invariance: {s:?}"
            );
        }
    }
    let d = he
        .iter()
        .zip(&hd)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max);
    assert!(d < 1e-3, "exact vs RI: {d:e}");
}
