// SPDX-License-Identifier: LGPL-3.0-or-later
#![allow(dead_code)]

use linus_disp::*;
use serde_json::Value;
use std::path::PathBuf;

pub fn fixture(name: &str) -> Value {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(format!("{name}.json"));
    serde_json::from_str(&std::fs::read_to_string(p).unwrap()).unwrap()
}

pub fn numbers(v: &Value) -> Vec<u32> {
    v["numbers"]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_u64().unwrap() as u32)
        .collect()
}

pub fn positions(v: &Value) -> Vec<[f64; 3]> {
    v["positions"]
        .as_array()
        .unwrap()
        .iter()
        .map(|p| {
            let p = p.as_array().unwrap();
            [
                p[0].as_f64().unwrap(),
                p[1].as_f64().unwrap(),
                p[2].as_f64().unwrap(),
            ]
        })
        .collect()
}

pub fn floats(v: &Value) -> Vec<f64> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap())
        .collect()
}

/// Evaluate one fixture case (model, functional, atm) with linus-disp.
pub fn eval(
    model: &str,
    functional: &str,
    atm: bool,
    z: &[u32],
    xyz: &[[f64; 3]],
    charge: f64,
    order: Order,
) -> DispResult {
    match model {
        "d3bj" => {
            let mut p = d3bj_params(functional).unwrap();
            if atm {
                p = p.with_atm(1.0);
            }
            d3bj(
                z,
                xyz,
                &p,
                &D3Options {
                    order,
                    ..Default::default()
                },
            )
            .unwrap()
        }
        "d4" => d4(
            z,
            xyz,
            charge,
            &d4_params(functional).unwrap(),
            &D4Options {
                order,
                ..Default::default()
            },
        )
        .unwrap(),
        m => panic!("{m}"),
    }
}

pub fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max)
}

pub fn flat(g: &[[f64; 3]]) -> Vec<f64> {
    g.iter().flat_map(|r| r.iter().copied()).collect()
}
