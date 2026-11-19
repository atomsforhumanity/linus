//! Helpers of the feature-combination tests (solvent_combinations.rs, ecp_derivatives.rs):
//! all-atom finite differences of energies and analytic gradients; and of the round-off
//! robustness tests (scf_robustness.rs, gpu_open_shell_loop.rs): last-bit flips of an SCF's
//! start.
#![allow(dead_code)]

use linus::solvent::SolventSettings;
use linus::{Session, Structure};
use linus_solvent::PcmModel;

/// The SAD guess (spin densities) of `mol` in the spherical basis `basis`, as the session
/// builds it.
pub fn sad_guess(mol: &Structure, basis: &str) -> (faer::Mat<f64>, faer::Mat<f64>) {
    let def = linus_ints::BasisDef::load(basis, None).unwrap();
    let set = linus_ints::BasisSet::new(&def, &mol.z, mol.positions(), true).unwrap();
    let (na, nb) = mol.nalpha_nbeta().unwrap();
    linus_scf::guess::sad_guess(&def, &set, &mol.z, na, nb, faer::Par::Seq)
}

/// One-ulp flips of about half the elements of the spin density `d` (each up or down, the
/// matrix kept symmetric), drawn from `seed` by xorshift: a last-bit change of the SCF's start,
/// as a compiler, a BLAS or a GPU's arithmetic makes one.
pub fn flip_last_bits(d: &mut faer::Mat<f64>, seed: u64) {
    let mut x = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    let n = d.nrows();
    for i in 0..n {
        for j in i..n {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            if x & 1 == 0 || d[(i, j)] == 0.0 {
                continue;
            }
            let bits = d[(i, j)].to_bits();
            let v = f64::from_bits(if x & 2 == 0 { bits + 1 } else { bits - 1 });
            d[(i, j)] = v;
            d[(j, i)] = v;
        }
    }
}

/// The SAD guess of `mol` with the last-bit flips of `seed` (the beta density with seed + 100).
pub fn flipped_guess(mol: &Structure, basis: &str, seed: u64) -> (faer::Mat<f64>, faer::Mat<f64>) {
    let (mut da, mut db) = sad_guess(mol, basis);
    flip_last_bits(&mut da, seed);
    flip_last_bits(&mut db, seed + 100);
    (da, db)
}

/// Water without symmetry (Å).
pub const WATER: &str = "3\n\nO 0.000 0.000 0.118\nH 0.020 0.755 -0.471\nH -0.010 -0.760 -0.465\n";
/// HI, slightly off-axis (I: def2 ECP28MWB).
pub const HI: &str = include_str!("../../../../tests/molecules/hi.xyz");
/// CH2I radical, a doublet with a nondegenerate singly occupied orbital (I: def2 ECP28MWB).
pub const CH2I: &str = include_str!("../../../../tests/molecules/ch2i.xyz");

pub fn flat(g: &[[f64; 3]]) -> Vec<f64> {
    g.iter().flatten().copied().collect()
}

pub fn max_diff(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max)
}

/// A PCM of permittivity `eps` with 110 Lebedev points per sphere (PySCF preset otherwise).
pub fn pcm(model: PcmModel, eps: f64) -> Option<SolventSettings> {
    Some(SolventSettings {
        eps: Some(Some(eps)),
        model: Some(model),
        lebedev: Some(linus::driver::options::Auto::Value(110)),
        ..Default::default()
    })
}

/// dE/dx for every Cartesian coordinate by Richardson-extrapolated central differences of SCF
/// energies (steps ±h, ±2h; error O(h^4)); `calc` is left at `x0`.
pub fn fd_gradient(calc: &mut Session, x0: &[[f64; 3]], h: f64) -> Vec<f64> {
    let n3 = 3 * x0.len();
    let mut g = vec![0.0; n3];
    for (x, gx) in g.iter_mut().enumerate() {
        let mut e = [0.0; 4];
        for (i, s) in [1.0, -1.0, 2.0, -2.0].into_iter().enumerate() {
            let mut c = x0.to_vec();
            c[x / 3][x % 3] += s * h;
            calc.set_positions(&c).unwrap();
            e[i] = calc.energy().unwrap().energy;
        }
        *gx = (8.0 * (e[0] - e[1]) - (e[2] - e[3])) / (12.0 * h);
    }
    calc.set_positions(x0).unwrap();
    g
}

/// The Hessian (3N x 3N, row-major) by Richardson-extrapolated central differences of analytic
/// gradients, every atom displaced (steps ±h, ±2h), symmetrized; `calc` is left at `x0`.
pub fn fd_hessian(calc: &mut Session, x0: &[[f64; 3]], h: f64) -> Vec<f64> {
    let n3 = 3 * x0.len();
    let mut fd = vec![0.0; n3 * n3];
    for y in 0..n3 {
        let mut g = Vec::new();
        for s in [1.0, -1.0, 2.0, -2.0] {
            let mut c = x0.to_vec();
            c[y / 3][y % 3] += s * h;
            calc.set_positions(&c).unwrap();
            g.push(flat(&calc.gradient().unwrap().gradient));
        }
        for x in 0..n3 {
            fd[x * n3 + y] = (8.0 * (g[0][x] - g[1][x]) - (g[2][x] - g[3][x])) / (12.0 * h);
        }
    }
    calc.set_positions(x0).unwrap();
    (0..n3 * n3)
        .map(|i| 0.5 * (fd[i] + fd[(i % n3) * n3 + i / n3]))
        .collect()
}
