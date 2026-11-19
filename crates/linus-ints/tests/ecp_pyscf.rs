//! ECP matrices and gradient contractions against PySCF 2.15 (libcint) reference data from
//! tests/python/gen_ecp_fixtures.py (same basis functions and ECPs: the repository's BSE data).
//!
//! PySCF screens shell pairs by an overlap-type criterion (libcint's check_3c_overlap), which
//! drops semi-local integrals between functions on opposite sides of the ECP atom; elements it
//! returns as exactly zero are only required to be small here.

use linus_ints::basis::BasisDef;
use linus_ints::one_e::ecp::EcpSet;
use linus_ints::reference::oneshot;
use linus_ints::{BasisSet, PairOptions, PairSet};
use serde_json::Value;

fn fixtures() -> Value {
    let p = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/pyscf_ecp.json");
    serde_json::from_str(&std::fs::read_to_string(p).unwrap()).unwrap()
}

fn arr(v: &Value) -> Vec<f64> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap())
        .collect()
}

#[test]
fn ecp_matrix_and_gradient_match_pyscf() {
    let fx = fixtures();
    for (name, c) in fx.as_object().unwrap() {
        let def = BasisDef::load(c["basis"].as_str().unwrap(), None).unwrap();
        let z: Vec<u32> = c["z"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_u64().unwrap() as u32)
            .collect();
        let xyz: Vec<[f64; 3]> = c["coords"]
            .as_array()
            .unwrap()
            .iter()
            .map(|r| {
                let r = arr(r);
                [r[0], r[1], r[2]]
            })
            .collect();
        let b = BasisSet::new(&def, &z, &xyz, true).unwrap();
        let p = PairSet::build(&b, PairOptions::default());
        let e = EcpSet::new(&def, &z, &xyz).unwrap();
        let n = b.nbf;
        assert_eq!(n, c["nbf"].as_u64().unwrap() as usize);
        let v = oneshot::ecp_matrix(&b, &p, &e);
        let r = arr(&c["ecp"]);
        let (mut worst, mut screened): (f64, f64) = (0.0, 0.0);
        for (x, y) in v.iter().zip(&r) {
            if *y == 0.0 {
                screened = screened.max(x.abs());
            } else {
                // PySCF's own error on the large one-center elements is ~1e-14 relative
                worst = worst.max((x - y).abs() / y.abs().max(1.0));
            }
        }
        assert!(worst < 5e-14, "{name}: ECP matrix vs PySCF {worst:e}");
        assert!(
            screened < 1e-8,
            "{name}: elements PySCF screens {screened:e}"
        );
        // gradient contraction with D_ij = (cos(i + 2j) + cos(j + 2i)) / n
        let mut d = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                d[i * n + j] = (((i + 2 * j) as f64).cos() + ((j + 2 * i) as f64).cos()) / n as f64;
            }
        }
        let g = oneshot::ecp_grad(&b, &p, &e, &d);
        let gr: Vec<Vec<f64>> = c["grad"].as_array().unwrap().iter().map(arr).collect();
        for (a, (x, y)) in g.iter().zip(&gr).enumerate() {
            for k in 0..3 {
                assert!(
                    (x[k] - y[k]).abs() < 1e-10,
                    "{name}: gradient atom {a} k {k}: {} vs {}",
                    x[k],
                    y[k]
                );
            }
        }
    }
}
