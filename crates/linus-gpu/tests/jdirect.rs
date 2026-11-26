//! The memory-lean device DF-J against a dense host reference (J = (mn|P) V^-1 (P|ls) D_ls from
//! linus-ints' integrals), one batch and many batches of integrals; bitwise run to run.
#![cfg(feature = "gpu")]

mod common;

use common::Rng;
use linus_gpu::cpu::{metric, Aux3c, Operator};
use linus_gpu::df::DfLayout;
use linus_gpu::harness;
use linus_gpu::jdirect::{GpuDfJDirect, JDirectEngine};
use std::sync::Arc;

fn reference(
    basis: &linus_gpu::cpu::BasisSet,
    aux: &linus_gpu::cpu::BasisSet,
    d: &[f64],
) -> Vec<f64> {
    let aux3 = Aux3c::new(aux);
    let chunks = aux3.chunks(0..aux.shells.len());
    let lay = DfLayout::new(basis, &aux3, 1e-14);
    let (nbf, naux, nx) = (lay.nbf, lay.naux, lay.nx);
    let mut a = vec![0.0; nx * naux];
    lay.integrals(
        &chunks,
        1e-16,
        &Operator::Coulomb,
        0,
        lay.pairs.len(),
        &mut a,
    );
    // the rows of D (D_mn + D_nm off the diagonal shell blocks)
    let w: Vec<f64> = (0..nx)
        .map(|x| {
            let (m, n) = (lay.row_m[x] as usize, lay.row_n[x] as usize);
            let v = d[n * nbf + m];
            if lay.row_kind[x] == linus_gpu::df::ROW_OFFDIAG {
                v + d[m * nbf + n]
            } else {
                v
            }
        })
        .collect();
    let mut g = vec![0.0; naux];
    for x in 0..nx {
        for p in 0..naux {
            g[p] += a[x * naux + p] * w[x];
        }
    }
    // c = V^-1 g (Cholesky)
    let v = metric(&aux3, 0.0, &Operator::Coulomb);
    let mut l = vec![0.0; naux * naux];
    for i in 0..naux {
        for j in 0..=i {
            let mut s = v[i * naux + j];
            for k in 0..j {
                s -= l[i * naux + k] * l[j * naux + k];
            }
            l[i * naux + j] = if i == j {
                s.sqrt()
            } else {
                s / l[j * naux + j]
            };
        }
    }
    for i in 0..naux {
        let s: f64 = (0..i).map(|k| l[i * naux + k] * g[k]).sum();
        g[i] = (g[i] - s) / l[i * naux + i];
    }
    for i in (0..naux).rev() {
        let s: f64 = (i + 1..naux).map(|k| l[k * naux + i] * g[k]).sum();
        g[i] = (g[i] - s) / l[i * naux + i];
    }
    let mut j = vec![0.0; nbf * nbf];
    for x in 0..nx {
        if lay.row_kind[x] == linus_gpu::df::ROW_DIAG_UPPER {
            continue;
        }
        let (m, n) = (lay.row_m[x] as usize, lay.row_n[x] as usize);
        let v: f64 = (0..naux).map(|p| a[x * naux + p] * g[p]).sum();
        j[n * nbf + m] = v;
        j[m * nbf + n] = v;
    }
    j
}

#[test]
fn coulomb_matches_the_host_reference() {
    let Some(gpu) = common::gpu() else { return };
    let gpu = Arc::new(gpu);
    let mol = harness::structure("la18").unwrap();
    let basis = harness::basis(&mol, "def2-svp").unwrap();
    let aux = harness::basis(&mol, "def2-universal-jkfit").unwrap();
    let n = basis.nbf;
    let mut rng = Rng(3);
    let c: Vec<f64> = (0..n * 8).map(|_| (rng.uniform() - 0.5) * 0.4).collect();
    let mut d = vec![0.0; n * n];
    for j in 0..n {
        for i in 0..n {
            d[j * n + i] = (0..8).map(|k| c[k * n + i] * c[k * n + j]).sum();
        }
    }
    let r = reference(&basis, &aux, &d);
    let scale = r.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    // the Hermite engine (Auto at la18/SVP) and the eri3c engine in one batch and in many
    for (engine, ws) in [
        (JDirectEngine::Hermite, 1usize << 30),
        (JDirectEngine::Eri3c, 1 << 30),
        (JDirectEngine::Eri3c, 1 << 20),
    ] {
        let mut eng =
            GpuDfJDirect::with_engine(gpu.clone(), &basis, &aux, None, ws, engine).unwrap();
        if engine == JDirectEngine::Hermite {
            assert_eq!(eng.engine(), "hermite");
        }
        let j1 = eng.coulomb_host(&d, 1).unwrap();
        let j2 = eng.coulomb_host(&d, 1).unwrap();
        assert!(j1.iter().zip(&j2).all(|(a, b)| a.to_bits() == b.to_bits()));
        // two densities in one build (an open shell's): each the one-density build's J (the
        // Hermite engine's bit for bit; the eri3c engine's GEMMs take both columns at once and
        // sum in another order: 3e-14 of the largest element measured)
        let d2: Vec<f64> = d.iter().map(|v| 0.5 * v).collect();
        let j12 = eng
            .coulomb_host(&[d.clone(), d2.clone()].concat(), 2)
            .unwrap();
        let jd2 = eng.coulomb_host(&d2, 1).unwrap();
        let n2 = n * n;
        let diff = |a: &[f64], b: &[f64]| {
            a.iter()
                .zip(b)
                .fold(0.0f64, |m, (x, y)| m.max((x - y).abs()))
        };
        let (d1, d2) = (diff(&j12[..n2], &j1), diff(&j12[n2..], &jd2));
        assert!(
            d1 <= 1e-13 * scale && d2 <= 1e-13 * scale,
            "two densities: {d1:e} {d2:e} (scale {scale:e})"
        );
        let err = j1
            .iter()
            .zip(&r)
            .fold(0.0f64, |m, (a, b)| m.max((a - b).abs()));
        eprintln!(
            "workspace {ws}: {} batches, max |J| {scale:.3e}, max |dJ| {err:.3e}, {:?}",
            eng.nbatches(),
            eng.times()
        );
        // within 1e-12 of the largest element (the DF engine's arithmetic: the same integrals)
        assert!(err < 1e-12 * scale, "{err:e} relative to {scale:e}");
    }
}
