//! Shared helpers for the derivative tests: small test molecules with custom basis sets,
//! deterministic random matrices, Richardson-extrapolated central differences, and JSON
//! fixtures written by tests/python/gen_deriv_fixtures.py.
#![allow(dead_code)]

use linus_ints::basis::{BasisDef, ShellDef};
use linus_ints::BasisSet;
use std::collections::HashMap;

/// Shells as (l, exponents, coefficients of normalized primitives).
pub type ShellSpec = (usize, Vec<f64>, Vec<f64>);

pub fn basis_def(elements: &[(u32, Vec<ShellSpec>)]) -> BasisDef {
    let mut el = HashMap::new();
    for (z, shells) in elements {
        let mut v: Vec<ShellDef> = shells
            .iter()
            .map(|(l, e, c)| ShellDef {
                l: *l,
                exps: e.clone(),
                coefs: c.clone(),
            })
            .collect();
        v.sort_by_key(|s| s.l);
        el.insert(*z, v);
    }
    BasisDef {
        name: "test".into(),
        elements: el,
        ecp_electrons: HashMap::new(),
        ecp: HashMap::new(),
    }
}

/// Three atoms (Z = 8, 1, 7) at low-symmetry positions (bohr) with an s, p, d, f basis on
/// "O", s, p on "H" and s, g on "N" (`with_g`), or s, d on "N".
pub fn test_system(with_g: bool) -> (BasisDef, Vec<u32>, Vec<[f64; 3]>) {
    let o = vec![
        (0, vec![5.0, 1.2], vec![0.4, 0.7]),
        (1, vec![1.1], vec![1.0]),
        (2, vec![0.9], vec![1.0]),
        (3, vec![0.7], vec![1.0]),
    ];
    let h = vec![
        (0, vec![1.5, 0.4], vec![0.5, 0.6]),
        (1, vec![0.8], vec![1.0]),
    ];
    let n = if with_g {
        vec![(0, vec![0.9], vec![1.0]), (4, vec![0.6], vec![1.0])]
    } else {
        vec![
            (0, vec![0.9], vec![1.0]),
            (2, vec![0.6, 1.4], vec![0.5, 0.5]),
        ]
    };
    let def = basis_def(&[(8, o), (1, h), (7, n)]);
    let z = vec![8, 1, 7];
    let positions = vec![[0.1, -0.2, 0.05], [1.3, 0.9, -0.4], [-0.7, 1.1, 1.2]];
    (def, z, positions)
}

pub fn make_basis(def: &BasisDef, z: &[u32], positions: &[[f64; 3]]) -> BasisSet {
    BasisSet::new(def, z, positions, true).unwrap()
}

/// Deterministic pseudo-random numbers in [-1, 1).
pub struct Lcg(u64);
impl Lcg {
    pub fn new(seed: u64) -> Self {
        Lcg(seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407))
    }
    pub fn next(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
    }
}

pub fn random_sym(n: usize, seed: u64) -> Vec<f64> {
    let mut r = Lcg::new(seed);
    let mut m = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..=i {
            let v = r.next();
            m[i * n + j] = v;
            m[j * n + i] = v;
        }
    }
    m
}

pub fn random_vec(n: usize, seed: u64) -> Vec<f64> {
    let mut r = Lcg::new(seed);
    (0..n).map(|_| r.next()).collect()
}

/// Richardson-extrapolated central difference of a vector-valued f at x = 0:
/// [8 (f(h) - f(-h)) - (f(2h) - f(-2h))] / (12 h).
pub fn fd<F: FnMut(f64) -> Vec<f64>>(mut f: F, h: f64) -> Vec<f64> {
    let p1 = f(h);
    let m1 = f(-h);
    let p2 = f(2.0 * h);
    let m2 = f(-2.0 * h);
    (0..p1.len())
        .map(|i| (8.0 * (p1[i] - m1[i]) - (p2[i] - m2[i])) / (12.0 * h))
        .collect()
}

pub const H: f64 = 1e-4;

/// FD step for differences of large summed scalars (two-electron energies with random
/// densities): their roundoff (~1e-13 relative) / h dominates at h = 1e-4, while the
/// Richardson truncation error at h = 1e-3 is ~1e-12. Override with LINUS_FD_H.
pub fn h_step() -> f64 {
    std::env::var("LINUS_FD_H")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1e-3)
}

pub fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max)
}

pub fn max_abs(a: &[f64]) -> f64 {
    a.iter().map(|x| x.abs()).fold(0.0, f64::max)
}

/// Assert max |a - b| <= tol * max(1, max |b|) and print the error.
pub fn check(name: &str, a: &[f64], b: &[f64], tol: f64) -> f64 {
    let e = max_abs_diff(a, b);
    let scale = max_abs(b).max(1.0);
    println!(
        "{name:40} max|diff| = {e:.3e}  (max|ref| = {:.3e})",
        max_abs(b)
    );
    assert!(
        e <= tol * scale,
        "{name}: max |diff| {e:e} > {tol:e} * {scale}"
    );
    e
}

pub fn flat(g: &[[f64; 3]]) -> Vec<f64> {
    g.iter().flat_map(|x| x.iter().cloned()).collect()
}

pub fn displaced(positions: &[[f64; 3]], atom: usize, k: usize, d: f64) -> Vec<[f64; 3]> {
    let mut c = positions.to_vec();
    c[atom][k] += d;
    c
}
