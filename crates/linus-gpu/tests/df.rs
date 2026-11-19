//! The device DF-J/K engine against a dense CPU reference built from linus-ints' integrals
//! (the same screened rows; an independent fit and contractions), in both storage modes, with
//! native and emulated exchange GEMMs, and run to run.
#![cfg(feature = "gpu")]

mod common;

use common::Rng;
use linus_gpu::cpu::{metric, Aux3c, BasisSet, Operator};
use linus_gpu::df::{
    DfFit, DfIntegrals, DfLayout, DfStorage, Factor, GpuDfJk, GpuDfJkOptions, GpuJkDeviceRequest,
    GpuJkRequest, KMath, ROW_DIAG_UPPER, ROW_OFFDIAG,
};
use linus_gpu::harness::{self, compare, Structure};

/// A factor C (nbf x ncol, column-major) and ncol.
type FactorOwned = (Vec<f64>, usize);

/// The dense reference: B = L^-1 A row by row, J = B (B^T w), K = sum_P B_P C C^T B_P.
struct Reference {
    j: Vec<Vec<f64>>,
    k: Vec<Vec<f64>>,
}

fn reference(basis: &BasisSet, aux: &BasisSet, dens: &[Vec<f64>], cs: &[FactorOwned]) -> Reference {
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
    // Cholesky of the metric (row-major lower factor)
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
    // B rows: forward substitution
    for x in 0..nx {
        let r = &mut a[x * naux..(x + 1) * naux];
        for i in 0..naux {
            let mut s = r[i];
            for k in 0..i {
                s -= l[i * naux + k] * r[k];
            }
            r[i] = s / l[i * naux + i];
        }
    }
    let b = a;
    let (rm, rn, rk) = (&lay.row_m, &lay.row_n, &lay.row_kind);
    let j = dens
        .iter()
        .map(|d| {
            let mut g = vec![0.0; naux];
            for x in 0..nx {
                let (m, n) = (rm[x] as usize, rn[x] as usize);
                let w = d[n * nbf + m]
                    + if rk[x] == ROW_OFFDIAG {
                        d[m * nbf + n]
                    } else {
                        0.0
                    };
                for p in 0..naux {
                    g[p] += w * b[x * naux + p];
                }
            }
            let mut jm = vec![0.0; nbf * nbf];
            for x in 0..nx {
                if rk[x] == ROW_DIAG_UPPER {
                    continue;
                }
                let (m, n) = (rm[x] as usize, rn[x] as usize);
                let v: f64 = (0..naux).map(|p| b[x * naux + p] * g[p]).sum();
                jm[n * nbf + m] = v;
                jm[m * nbf + n] = v;
            }
            jm
        })
        .collect();
    let k = cs
        .iter()
        .map(|(c, nc)| {
            let mut km = vec![0.0; nbf * nbf];
            let mut bp = vec![0.0; nbf * nbf];
            let mut z = vec![0.0; nbf * nc];
            for p in 0..naux {
                bp.fill(0.0);
                for x in 0..nx {
                    let (m, n) = (rm[x] as usize, rn[x] as usize);
                    bp[m * nbf + n] = b[x * naux + p];
                    if rk[x] == ROW_OFFDIAG {
                        bp[n * nbf + m] = b[x * naux + p];
                    }
                }
                // z[m, i] = sum_n bp[m][n] c[n, i]
                for m in 0..nbf {
                    for i in 0..*nc {
                        z[i * nbf + m] = (0..nbf).map(|n| bp[m * nbf + n] * c[i * nbf + n]).sum();
                    }
                }
                for q in 0..nbf {
                    for m in 0..nbf {
                        km[q * nbf + m] += (0..*nc)
                            .map(|i| z[i * nbf + m] * z[i * nbf + q])
                            .sum::<f64>();
                    }
                }
            }
            km
        })
        .collect();
    Reference { j, k }
}

fn water() -> Structure {
    let a = 1.0 / linus_core::units::BOHR_ANGSTROM;
    Structure::new(
        vec![8, 1, 1],
        vec![
            [0.0, 0.0, 0.1173 * a],
            [0.0, 0.7572 * a, -0.4692 * a],
            [0.0, -0.7572 * a, -0.4692 * a],
        ],
        0,
        1,
    )
    .unwrap()
}

/// The first `n` atoms of u1an39 (C, F, N, O, H).
fn fragment(n: usize) -> Structure {
    let mut m = harness::structure("u1an39").unwrap();
    m.z.truncate(n);
    m.positions.truncate(n);
    m
}

/// Factors with mixed signs and magnitudes (nbf x nc, column-major) and D = C C^T.
fn factors(nbf: usize, nc: usize, seed: u64) -> (Vec<f64>, Vec<f64>) {
    let mut rng = Rng(seed);
    let c: Vec<f64> = (0..nbf * nc).map(|_| (rng.uniform() - 0.5) * 0.4).collect();
    let mut d = vec![0.0; nbf * nbf];
    for j in 0..nbf {
        for i in 0..nbf {
            d[j * nbf + i] = (0..nc).map(|k| c[k * nbf + i] * c[k * nbf + j]).sum();
        }
    }
    (c, d)
}

fn trace(d: &[f64], m: &[f64]) -> f64 {
    d.iter().zip(m).map(|(a, b)| a * b).sum()
}

fn check(mol: &Structure, basis_name: &str, two_densities: bool) {
    let Some(gpu) = common::gpu() else { return };
    let basis = harness::basis(mol, basis_name).unwrap();
    let aux = harness::basis(mol, "def2-universal-jkfit").unwrap();
    let nbf = basis.nbf;
    let (ca, da) = factors(nbf, 5.min(nbf), 7);
    let (cb, db) = factors(nbf, 3.min(nbf), 11);
    let (dens, cs): (Vec<Vec<f64>>, Vec<FactorOwned>) = if two_densities {
        (vec![da, db], vec![(ca, 5.min(nbf)), (cb, 3.min(nbf))])
    } else {
        (vec![da], vec![(ca, 5.min(nbf))])
    };
    let r = reference(&basis, &aux, &dens, &cs);
    let dref: Vec<&[f64]> = dens.iter().map(|d| d.as_slice()).collect();
    let fs: Vec<Factor> = cs.iter().map(|(c, n)| Factor { c, ncol: *n }).collect();
    let req = GpuJkRequest {
        dens: &dref,
        factors: &fs,
        want_j: true,
        want_k: true,
    };
    let mut first = None;
    let oz = DfFit::Ozaki { slices: 7 };
    // (storage, K arithmetic, SYRK update, auxiliary block, K-update column block, the
    // setup's integrals and fit)
    for (storage, k_math, k_syrk, k_block, k_update_block, integrals, fit) in [
        (
            DfStorage::Device,
            KMath::Native,
            false,
            0,
            0,
            DfIntegrals::Auto,
            DfFit::Native,
        ),
        (
            DfStorage::Device,
            KMath::Native,
            true,
            24,
            0,
            DfIntegrals::Cpu,
            DfFit::Native,
        ),
        (
            DfStorage::Device,
            KMath::Native,
            false,
            24,
            16,
            DfIntegrals::Device,
            oz,
        ),
        (
            DfStorage::Host,
            KMath::Native,
            false,
            40,
            50,
            DfIntegrals::Auto,
            DfFit::Native,
        ),
        (
            DfStorage::Host,
            KMath::Native,
            false,
            40,
            50,
            DfIntegrals::Cpu,
            oz,
        ),
        (
            DfStorage::Device,
            KMath::Emulated { bits: 55 },
            false,
            0,
            32,
            DfIntegrals::Auto,
            DfFit::Native,
        ),
        // Ozaki exchange: one block, several blocks (accumulated), streamed host storage
        (
            DfStorage::Device,
            KMath::Ozaki { slices: 7 },
            false,
            0,
            32,
            DfIntegrals::Auto,
            oz,
        ),
        (
            DfStorage::Device,
            KMath::Ozaki { slices: 7 },
            false,
            24,
            0,
            DfIntegrals::Auto,
            DfFit::Native,
        ),
        (
            DfStorage::Host,
            KMath::Ozaki { slices: 7 },
            false,
            40,
            50,
            DfIntegrals::Device,
            oz,
        ),
    ] {
        let opts = GpuDfJkOptions {
            storage,
            k_math,
            k_syrk,
            k_block,
            k_update_block,
            integrals,
            fit,
            // several diagonal blocks and column chunks in the Ozaki fit
            fit_block: 32,
            fit_chunk: 200,
            ..Default::default()
        };
        let mut eng = GpuDfJk::new(&gpu, &basis, &aux, &opts).unwrap();
        assert_eq!(eng.resolved().storage, storage);
        let want = match integrals {
            DfIntegrals::Cpu => DfIntegrals::Cpu,
            _ => DfIntegrals::Device,
        };
        assert_eq!(eng.resolved().integrals, want);
        assert_eq!(eng.resolved().fit, fit);
        let out = eng.compute(&gpu, &req).unwrap();
        let again = eng.compute(&gpu, &req).unwrap();
        for (a, b) in out
            .j
            .iter()
            .chain(&out.k)
            .zip(again.j.iter().chain(&again.k))
        {
            assert!(
                a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits()),
                "{storage:?} {k_math:?}: not bitwise reproducible"
            );
        }
        let tag = format!(
            "{} {basis_name} {storage:?} {k_math:?} syrk {k_syrk} block {k_block} update {k_update_block} \
             {integrals:?} {fit:?}",
            harness::formula(mol)
        );
        for (d, (g, c)) in out.j.iter().zip(&r.j).enumerate() {
            let st = compare(g, c);
            eprintln!(
                "{tag} J[{d}]: {st}; E_J diff {:.2e}",
                trace(&dens[d], g) - trace(&dens[d], c)
            );
            // the fit amplifies round-off by the metric's conditioning (different Cholesky
            // and triangular-solve orders here and on the device)
            st.check(0.0, 2e-12).unwrap();
        }
        let ktol = 2e-12;
        for (d, (g, c)) in out.k.iter().zip(&r.k).enumerate() {
            let st = compare(g, c);
            eprintln!(
                "{tag} K[{d}]: {st}; E_K diff {:.2e}",
                trace(&dens[d], g) - trace(&dens[d], c)
            );
            st.check(0.0, ktol).unwrap();
            // exactly symmetric
            for i in 0..nbf {
                for j in 0..i {
                    assert_eq!(g[j * nbf + i].to_bits(), g[i * nbf + j].to_bits());
                }
            }
        }
        if first.is_none() {
            first = Some(out);
        } else if let Some(f) = &first {
            // the storage modes and block sizes agree to round-off; another tensor (the CPU's
            // integrals, the Ozaki fit) to the fit's round-off
            let tol = if (integrals, fit) == (DfIntegrals::Auto, DfFit::Native) {
                1e-14
            } else {
                1e-13
            };
            for (a, b) in out.j.iter().zip(&f.j) {
                compare(a, b).check(0.0, tol).unwrap();
            }
        }
    }
}

/// The setup's tensor (B, and B^omega of wB97X's omega) from the device integrals and from the
/// CPU's, fitted by the triangular solve and by the blocked solve with Ozaki updates (diagonal
/// blocks and column chunks of several sizes, host storage): every pair of tensors within
/// round-off of the fit (|dB| <= 1e-11 max |B|), the Ozaki fit bitwise reproducible and
/// independent of the storage; the resolved integrals and fit.
#[test]
fn setup_integrals_and_fit() {
    use linus_gpu::cpu::Operator;
    let Some(gpu) = common::gpu() else { return };
    let mol = fragment(6);
    let basis = harness::basis(&mol, "def2-tzvp").unwrap();
    let aux = harness::basis(&mol, "def2-universal-jkfit").unwrap();
    let oz = DfFit::Ozaki { slices: 7 };
    let mut runs: Vec<(String, Vec<f64>, Vec<f64>)> = Vec::new();
    // (integrals, fit, diagonal block, column chunk, storage)
    for (integrals, fit, fit_block, fit_chunk, storage) in [
        (DfIntegrals::Cpu, DfFit::Native, 0, 0, DfStorage::Device),
        (DfIntegrals::Device, DfFit::Native, 0, 0, DfStorage::Device),
        (DfIntegrals::Device, DfFit::Native, 0, 0, DfStorage::Host),
        (DfIntegrals::Device, oz, 0, 0, DfStorage::Device),
        (DfIntegrals::Device, oz, 0, 0, DfStorage::Device),
        (DfIntegrals::Device, oz, 0, 0, DfStorage::Host),
        (DfIntegrals::Device, oz, 48, 333, DfStorage::Device),
        (DfIntegrals::Cpu, oz, 16, 4096, DfStorage::Host),
    ] {
        let opts = GpuDfJkOptions {
            omega: Some(0.3),
            storage,
            lr_storage: storage,
            integrals,
            fit,
            fit_block,
            fit_chunk,
            ..Default::default()
        };
        let mut eng = GpuDfJk::new(&gpu, &basis, &aux, &opts).unwrap();
        eng.set_exchange_operator(&gpu, Operator::from_libxc(1.0, -0.842294, 0.3))
            .unwrap();
        let r = eng.resolved();
        assert_eq!(r.integrals, integrals);
        assert!(r.integrals_note.is_none());
        assert_eq!((r.fit, r.fit_note.as_deref()), (fit, None));
        let nx = r.nx;
        let b = eng.tensor_rows(&gpu, false, 0, nx).unwrap();
        let bw = eng.tensor_rows(&gpu, true, 0, nx).unwrap();
        let tag = format!("{integrals:?} {fit:?} block {fit_block} chunk {fit_chunk} {storage:?}");
        let st = &eng.setup_times;
        eprintln!(
            "{tag}: fit {:.4} s (slicing {:.4} s), B^w fit {:.4} s",
            st.b.fit, st.b.fit_slicing, st.lr.fit
        );
        runs.push((tag, b, bw));
    }
    let bmax = runs[0].1.iter().fold(0.0f64, |m, x| m.max(x.abs()));
    let wmax = runs[0].2.iter().fold(0.0f64, |m, x| m.max(x.abs()));
    let dmax = |a: &[f64], b: &[f64]| {
        a.iter()
            .zip(b)
            .fold(0.0f64, |m, (x, y)| m.max((x - y).abs()))
    };
    for (tag, b, bw) in &runs[1..] {
        let (db, dw) = (dmax(&runs[0].1, b), dmax(&runs[0].2, bw));
        eprintln!(
            "{tag} against CPU integrals, native fit: B {:.2e}, B^w {:.2e} (relative to max)",
            db / bmax,
            dw / wmax
        );
        assert!(db <= 1e-11 * bmax && dw <= 1e-11 * wmax, "{tag}");
    }
    // the Ozaki fit: run to run and device against host storage, bit for bit
    assert!(same_bits(&runs[3].1, &runs[4].1) && same_bits(&runs[3].2, &runs[4].2));
    assert!(same_bits(&runs[3].1, &runs[5].1) && same_bits(&runs[3].2, &runs[5].2));
    // the native fit: device against host storage, bit for bit
    assert!(same_bits(&runs[1].1, &runs[2].1) && same_bits(&runs[1].2, &runs[2].2));
}

/// Shells the generated kernels do not cover (Cartesian d): the integrals fall back to the CPU
/// with the reason under `DfIntegrals::Auto` (J/K against the reference as usual), an
/// unsupported error under `DfIntegrals::Device`.
#[test]
fn cartesian_basis_takes_the_cpu_integrals() {
    let Some(gpu) = common::gpu() else { return };
    let mol = water();
    let def = linus_gpu::cpu::BasisDef::load("def2-svp", None).unwrap();
    let basis = BasisSet::new(&def, &mol.z, &mol.positions, false).unwrap();
    let aux = harness::basis(&mol, "def2-universal-jkfit").unwrap();
    let nbf = basis.nbf;
    let (ca, da) = factors(nbf, 5, 7);
    let r = reference(&basis, &aux, std::slice::from_ref(&da), &[(ca.clone(), 5)]);
    let mut eng = GpuDfJk::new(&gpu, &basis, &aux, &GpuDfJkOptions::default()).unwrap();
    let res = eng.resolved();
    assert_eq!(res.integrals, DfIntegrals::Cpu);
    let note = res.integrals_note.clone().unwrap();
    assert!(note.contains("Cartesian"), "{note}");
    let out = eng
        .compute(
            &gpu,
            &GpuJkRequest {
                dens: &[&da],
                factors: &[Factor { c: &ca, ncol: 5 }],
                want_j: true,
                want_k: true,
            },
        )
        .unwrap();
    compare(&out.j[0], &r.j[0]).check(0.0, 2e-12).unwrap();
    compare(&out.k[0], &r.k[0]).check(0.0, 2e-12).unwrap();
    let dev = GpuDfJkOptions {
        integrals: DfIntegrals::Device,
        ..Default::default()
    };
    assert!(matches!(
        GpuDfJk::new(&gpu, &basis, &aux, &dev),
        Err(linus_gpu::GpuError::NotCovered(_))
    ));
}

#[test]
fn water_svp_one_density() {
    check(&water(), "def2-svp", false);
}

#[test]
fn fragment_tzvp_two_densities() {
    // f functions in the basis, g in the auxiliary basis
    check(&fragment(6), "def2-tzvp", true);
}

#[test]
fn layout_rows_match_pair_shapes() {
    let mol = fragment(5);
    let basis = harness::basis(&mol, "def2-svp").unwrap();
    let aux = harness::basis(&mol, "def2-universal-jkfit").unwrap();
    let lay = DfLayout::new(&basis, &Aux3c::new(&aux), 1e-14);
    assert_eq!(lay.row_m.len(), lay.nx);
    // every (m, n) with m >= n of a kept pair appears once with kind != upper
    let mut seen = std::collections::HashSet::new();
    for x in 0..lay.nx {
        if lay.row_kind[x] != ROW_DIAG_UPPER {
            let (m, n) = (lay.row_m[x], lay.row_n[x]);
            assert!(m >= n || lay.row_kind[x] == ROW_OFFDIAG);
            assert!(seen.insert((m.max(n), m.min(n))));
        }
    }
    let total: usize = lay.row_batches(100).iter().map(|b| b.3).sum();
    assert_eq!(total, lay.nx);
}

fn same_bits(a: &[f64], b: &[f64]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}

#[test]
fn device_entry_point_matches_the_host_one_bitwise() {
    let Some(gpu) = common::gpu() else { return };
    let mol = fragment(4);
    let basis = harness::basis(&mol, "def2-svp").unwrap();
    let aux = harness::basis(&mol, "def2-universal-jkfit").unwrap();
    let nbf = basis.nbf;
    let (ca, da) = factors(nbf, 6, 5);
    let (cb, db) = factors(nbf, 2, 9);
    let mut eng = GpuDfJk::new(&gpu, &basis, &aux, &GpuDfJkOptions::default()).unwrap();
    let host = eng
        .compute(
            &gpu,
            &GpuJkRequest {
                dens: &[&da, &db],
                factors: &[Factor { c: &ca, ncol: 6 }, Factor { c: &cb, ncol: 2 }],
                want_j: true,
                want_k: true,
            },
        )
        .unwrap();
    let n2 = nbf * nbf;
    let dens = gpu.upload(&[da.clone(), db.clone()].concat()).unwrap();
    let cs = gpu.upload(&[ca.clone(), cb.clone()].concat()).unwrap();
    let mut j = gpu.alloc_zeros::<f64>(2 * n2).unwrap();
    let mut k = gpu.alloc_zeros::<f64>(2 * n2).unwrap();
    let req = GpuJkDeviceRequest {
        dens: Some(&dens),
        ndens: 2,
        factors: Some(&cs),
        ncols: &[6, 2],
    };
    eng.compute_device(&gpu, &req, Some(&mut j), Some(&mut k))
        .unwrap();
    let (j, k) = (gpu.download(&j).unwrap(), gpu.download(&k).unwrap());
    assert!(same_bits(&j, &host.j.concat()) && same_bits(&k, &host.k.concat()));
    // J only and K only, into buffers with stale contents (overwritten)
    let mut j1 = gpu.upload(&vec![7.0; 2 * n2]).unwrap();
    eng.compute_device(&gpu, &req, Some(&mut j1), None).unwrap();
    assert!(same_bits(&gpu.download(&j1).unwrap(), &j));
    let mut k1 = gpu.upload(&vec![7.0; 2 * n2]).unwrap();
    eng.compute_device(&gpu, &req, None, Some(&mut k1)).unwrap();
    assert!(same_bits(&gpu.download(&k1).unwrap(), &k));
}

/// The fitted K block (`k_block_fit`): an allowance the automatic block does not fit gives a
/// smaller block (a multiple of 8, at least K_BLOCK_MIN) that fits, and J/K are bitwise those of
/// that block set explicitly; without the fit the setup is out of memory. (24 atoms of u1an39 in
/// def2-TZVP: the automatic block's workspace is above the setup's 1 GiB batch.)
#[test]
fn k_block_fitted_to_the_allowance() {
    use linus_gpu::df::{auto_k_block, K_BLOCK_MIN};
    use linus_gpu::GpuError;
    let Some(gpu) = common::gpu() else { return };
    let mol = fragment(24);
    let basis = harness::basis(&mol, "def2-tzvp").unwrap();
    let aux = harness::basis(&mol, "def2-universal-jkfit").unwrap();
    let (nbf, naux) = (basis.nbf, aux.nbf);
    let (ca, _) = factors(nbf, 20, 5);
    let fa = [Factor { c: &ca, ncol: 20 }];
    let req = GpuJkRequest {
        dens: &[],
        factors: &fa,
        want_j: true,
        want_k: true,
    };
    let base = GpuDfJkOptions {
        storage: DfStorage::Device,
        others_reserve: 0,
        ..GpuDfJkOptions::default()
    };
    let auto = auto_k_block(nbf, naux, base.k_workspace);
    // the setup's need with the automatic block, from its out-of-memory error
    let need = match GpuDfJk::new(
        &gpu,
        &basis,
        &aux,
        &GpuDfJkOptions {
            device_budget: Some(1),
            ..base.clone()
        },
    ) {
        Err(GpuError::OutOfMemory { requested, .. }) => requested,
        r => panic!("expected out of memory, got {:?}", r.err()),
    };
    let probe = GpuDfJk::new(&gpu, &basis, &aux, &base).unwrap();
    let tensor = probe.resolved().tensor_bytes;
    drop(probe);
    let work = need - tensor;
    assert!(
        work > 2 << 30,
        "{nbf} functions: a build's workspace {work} bytes"
    );
    let capped = GpuDfJkOptions {
        device_budget: Some(tensor + work * 3 / 4),
        ..base.clone()
    };
    assert!(GpuDfJk::new(&gpu, &basis, &aux, &capped).is_err());
    let mut fitted = GpuDfJk::new(
        &gpu,
        &basis,
        &aux,
        &GpuDfJkOptions {
            k_block_fit: true,
            ..capped
        },
    )
    .unwrap();
    let nq = fitted.resolved().k_block;
    assert!(
        (K_BLOCK_MIN..auto).contains(&nq) && nq.is_multiple_of(8),
        "{nq} of {auto}"
    );
    let a = fitted.compute(&gpu, &req).unwrap();
    drop(fitted);
    let mut explicit = GpuDfJk::new(
        &gpu,
        &basis,
        &aux,
        &GpuDfJkOptions {
            k_block: nq,
            ..base
        },
    )
    .unwrap();
    let b = explicit.compute(&gpu, &req).unwrap();
    assert!(same_bits(&a.j.concat(), &b.j.concat()) && same_bits(&a.k.concat(), &b.k.concat()));
}

/// The kept Coulomb fit (`set_keep_coulomb_fit`): J unchanged bit for bit, g = B^T w with
/// |g_d|^2 = tr(D_d J[D_d]), the same g (to round-off) from host-streamed storage, nothing kept
/// when the setting is off.
#[test]
fn kept_coulomb_fit() {
    let Some(gpu) = common::gpu() else { return };
    let mol = fragment(4);
    let basis = harness::basis(&mol, "def2-svp").unwrap();
    let aux = harness::basis(&mol, "def2-universal-jkfit").unwrap();
    let (nbf, naux) = (basis.nbf, aux.nbf);
    let (_, da) = factors(nbf, 6, 5);
    let (_, db) = factors(nbf, 2, 9);
    let req = GpuJkRequest {
        dens: &[&da, &db],
        factors: &[],
        want_j: true,
        want_k: false,
    };
    let mut fits = Vec::new();
    for storage in [DfStorage::Device, DfStorage::Host] {
        let opts = GpuDfJkOptions {
            storage,
            ..GpuDfJkOptions::default()
        };
        let mut eng = GpuDfJk::new(&gpu, &basis, &aux, &opts).unwrap();
        let bytes0 = eng.resolved().device_bytes;
        let off = eng.compute(&gpu, &req).unwrap();
        assert!(eng.last_coulomb_fit().is_none());
        eng.set_keep_coulomb_fit(true);
        let on = eng.compute(&gpu, &req).unwrap();
        assert!(
            same_bits(&on.j.concat(), &off.j.concat()),
            "{storage:?}: J changed"
        );
        let (g, nd) = eng.last_coulomb_fit().unwrap();
        assert_eq!(nd, 2);
        assert_eq!(eng.resolved().device_bytes, bytes0 + 8 * g.len());
        let g = gpu.download(g).unwrap();
        for (d, (dm, jm)) in [&da, &db].iter().zip(&on.j).enumerate() {
            let gd = &g[d * naux..(d + 1) * naux];
            let e = trace(dm, jm);
            let gg: f64 = gd.iter().map(|x| x * x).sum();
            assert!(
                (gg - e).abs() < 1e-12 * e.abs(),
                "{storage:?}: |g|^2 {gg} tr(DJ) {e}"
            );
        }
        fits.push(g[..2 * naux].to_vec());
        eng.set_keep_coulomb_fit(false);
        assert!(eng.last_coulomb_fit().is_none());
        assert_eq!(eng.resolved().device_bytes, bytes0);
    }
    let dmax = fits[0]
        .iter()
        .zip(&fits[1])
        .fold(0.0f64, |m, (a, b)| m.max((a - b).abs()));
    assert!(dmax < 1e-12, "device vs host-streamed g: {dmax:e}");
}

/// The fitted tensor of the operator erf(omega r)/r (omega > 0) or 1/r (omega = 0) on the rows
/// of the layout: B = L^-1 (mn|op|P) row by row (linus-ints' `pair_block`, the Coulomb
/// metric), and the layout.
fn fitted(basis: &BasisSet, aux: &BasisSet, omega: f64) -> (DfLayout, Vec<f64>) {
    use linus_gpu::cpu::{pair_block, Operator, Work3c};
    let aux3 = Aux3c::new(aux);
    let chunks = aux3.chunks(0..aux.shells.len());
    let lay = DfLayout::new(basis, &aux3, 1e-14);
    let (naux, nx) = (lay.naux, lay.nx);
    let op = if omega > 0.0 {
        Operator::LongRange { omega }
    } else {
        Operator::Coulomb
    };
    let mut a = vec![0.0; nx * naux];
    let mut w = Work3c::new();
    for (p, &x0) in lay.pairs.iter().zip(&lay.x0) {
        let slab = &mut a[x0 * naux..(x0 + p.nab()) * naux];
        pair_block(p, &chunks, slab, naux, 1e-16, &op, &mut w);
    }
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
    for x in 0..nx {
        let r = &mut a[x * naux..(x + 1) * naux];
        for i in 0..naux {
            let mut s = r[i];
            for k in 0..i {
                s -= l[i * naux + k] * r[k];
            }
            r[i] = s / l[i * naux + i];
        }
    }
    (lay, a)
}

/// The wK exchange of the operator full/r + erf erf(omega r)/r for D = C C^T, densely:
/// sum_P [full Z_P Z_P^T + erf/2 (Z_P Zw_P^T + Zw_P Z_P^T)], Z_P = B_P C, Zw_P = Bw_P C.
fn reference_wk(
    basis: &BasisSet,
    aux: &BasisSet,
    cs: &[FactorOwned],
    full: f64,
    erf: f64,
    omega: f64,
) -> Vec<Vec<f64>> {
    let (lay, b) = fitted(basis, aux, 0.0);
    let (_, bw) = fitted(basis, aux, omega);
    let (nbf, naux, nx) = (lay.nbf, lay.naux, lay.nx);
    let (rm, rn, rk) = (&lay.row_m, &lay.row_n, &lay.row_kind);
    let plane = |t: &[f64], p: usize, out: &mut [f64]| {
        out.fill(0.0);
        for x in 0..nx {
            let (m, n) = (rm[x] as usize, rn[x] as usize);
            out[m * nbf + n] = t[x * naux + p];
            if rk[x] == ROW_OFFDIAG {
                out[n * nbf + m] = t[x * naux + p];
            }
        }
    };
    cs.iter()
        .map(|(c, nc)| {
            let nc = *nc;
            let mut km = vec![0.0; nbf * nbf];
            let (mut bp, mut bwp) = (vec![0.0; nbf * nbf], vec![0.0; nbf * nbf]);
            let (mut z, mut zw) = (vec![0.0; nbf * nc], vec![0.0; nbf * nc]);
            for p in 0..naux {
                plane(&b, p, &mut bp);
                plane(&bw, p, &mut bwp);
                for m in 0..nbf {
                    for i in 0..nc {
                        z[i * nbf + m] = (0..nbf).map(|n| bp[m * nbf + n] * c[i * nbf + n]).sum();
                        zw[i * nbf + m] = (0..nbf).map(|n| bwp[m * nbf + n] * c[i * nbf + n]).sum();
                    }
                }
                for q in 0..nbf {
                    for m in 0..nbf {
                        km[q * nbf + m] += (0..nc)
                            .map(|i| {
                                let (zm, zq) = (z[i * nbf + m], z[i * nbf + q]);
                                let (wm, wq) = (zw[i * nbf + m], zw[i * nbf + q]);
                                full * zm * zq + 0.5 * erf * (zm * wq + wm * zq)
                            })
                            .sum::<f64>();
                    }
                }
            }
            km
        })
        .collect()
}

/// Range-separated exchange (wK, the CPU DfJk's scheme) against the dense reference: the
/// operators of wB97X (a/r + b erf(0.3 r)/r) and CAM-B3LYP (b < 0), erf alone, every storage
/// combination of B and B^omega, B^omega prepared at setup or by set_exchange_operator,
/// emulated GEMMs (the Ozaki exchange with the Ozaki fit); exactly symmetric, run to run, J
/// unchanged; another omega, and back to 1/r.
fn check_wk(mol: &Structure, basis_name: &str) {
    use linus_gpu::cpu::Operator;
    let Some(gpu) = common::gpu() else { return };
    let basis = harness::basis(mol, basis_name).unwrap();
    let aux = harness::basis(mol, "def2-universal-jkfit").unwrap();
    let nbf = basis.nbf;
    let (ca, da) = factors(nbf, 5.min(nbf), 7);
    let (cb, db) = factors(nbf, 3.min(nbf), 11);
    let dens = vec![da, db];
    let cs: Vec<FactorOwned> = vec![(ca, 5.min(nbf)), (cb, 3.min(nbf))];
    let r = reference(&basis, &aux, &dens, &cs);
    let wb97x = Operator::from_libxc(1.0, -0.842294, 0.3);
    let cam = Operator::from_libxc(0.19, 0.46, 0.33);
    let erf = Operator::LongRange { omega: 0.3 };
    let refs: Vec<(Operator, Vec<Vec<f64>>)> = [wb97x, cam, erf]
        .iter()
        .map(|op| {
            (
                *op,
                reference_wk(&basis, &aux, &cs, op.coulomb(), op.long_range(), op.omega()),
            )
        })
        .collect();
    let dref: Vec<&[f64]> = dens.iter().map(|d| d.as_slice()).collect();
    let fs: Vec<Factor> = cs.iter().map(|(c, n)| Factor { c, ncol: *n }).collect();
    let req = GpuJkRequest {
        dens: &dref,
        factors: &fs,
        want_j: true,
        want_k: true,
    };
    // (B storage, B^omega storage, omega at setup, K arithmetic, auxiliary block)
    for (storage, lr_storage, at_setup, k_math, k_block) in [
        (DfStorage::Device, DfStorage::Auto, true, KMath::Native, 0),
        (DfStorage::Device, DfStorage::Auto, false, KMath::Native, 24),
        (DfStorage::Device, DfStorage::Host, true, KMath::Native, 40),
        (DfStorage::Host, DfStorage::Host, false, KMath::Native, 40),
        (DfStorage::Host, DfStorage::Device, true, KMath::Native, 0),
        (
            DfStorage::Device,
            DfStorage::Auto,
            true,
            KMath::Emulated { bits: 55 },
            0,
        ),
        // Ozaki exchange: B^omega at setup or later, on the device or streamed
        (
            DfStorage::Device,
            DfStorage::Auto,
            true,
            KMath::Ozaki { slices: 7 },
            0,
        ),
        (
            DfStorage::Device,
            DfStorage::Host,
            false,
            KMath::Ozaki { slices: 7 },
            24,
        ),
        (
            DfStorage::Host,
            DfStorage::Host,
            false,
            KMath::Ozaki { slices: 7 },
            40,
        ),
    ] {
        // the Ozaki exchange with the Ozaki fit (B^omega's too, also when built later)
        let fit = match k_math {
            KMath::Ozaki { slices } => DfFit::Ozaki { slices },
            _ => DfFit::Native,
        };
        let opts = GpuDfJkOptions {
            storage,
            lr_storage,
            omega: at_setup.then_some(0.3),
            k_math,
            k_block,
            fit,
            fit_block: 32,
            ..Default::default()
        };
        let mut eng = GpuDfJk::new(&gpu, &basis, &aux, &opts).unwrap();
        assert!(eng
            .set_exchange_operator(
                &gpu,
                Operator::Attenuated {
                    coulomb: 0.25,
                    long_range: 0.0,
                    omega: 0.0
                }
            )
            .is_err());
        let tag = format!(
            "{} {basis_name} B {storage:?} B^w {lr_storage:?} setup {at_setup} {k_math:?} block {k_block}",
            harness::formula(mol)
        );
        for (op, kref) in &refs {
            eng.set_exchange_operator(&gpu, *op).unwrap();
            let res = eng.resolved();
            assert_eq!(res.lr_omega, Some(op.omega()));
            let want_lr = match lr_storage {
                DfStorage::Auto => storage,
                s => s,
            };
            assert_eq!(res.lr_storage, Some(want_lr), "{tag}");
            let out = eng.compute(&gpu, &req).unwrap();
            let again = eng.compute(&gpu, &req).unwrap();
            for (a, b) in out.k.iter().zip(&again.k) {
                assert!(same_bits(a, b), "{tag} {op:?}: not reproducible");
            }
            for (g, c) in out.j.iter().zip(&r.j) {
                compare(g, c).check(0.0, 2e-12).unwrap();
            }
            for (d, (g, c)) in out.k.iter().zip(kref).enumerate() {
                let st = compare(g, c);
                eprintln!(
                    "{tag} {op:?} K[{d}]: {st}; E_K diff {:.2e}",
                    trace(&dens[d], g) - trace(&dens[d], c)
                );
                st.check(0.0, 2e-12).unwrap();
                for i in 0..nbf {
                    for j in 0..i {
                        assert_eq!(g[j * nbf + i].to_bits(), g[i * nbf + j].to_bits());
                    }
                }
            }
        }
        // another omega replaces B^omega; then back to 1/r
        let other = Operator::from_libxc(1.0, -1.0, 0.4);
        eng.set_exchange_operator(&gpu, other).unwrap();
        assert_eq!(eng.resolved().lr_omega, Some(0.4));
        let kref = reference_wk(&basis, &aux, &cs, other.coulomb(), other.long_range(), 0.4);
        let out = eng.compute(&gpu, &req).unwrap();
        for (g, c) in out.k.iter().zip(&kref) {
            compare(g, c).check(0.0, 2e-12).unwrap();
        }
        eng.clear_exchange_operator();
        assert!(eng.exchange_operator().is_none());
        let out = eng.compute(&gpu, &req).unwrap();
        for (g, c) in out.k.iter().zip(&r.k) {
            compare(g, c).check(0.0, 2e-12).unwrap();
        }
    }
}

#[test]
fn wk_water_svp() {
    check_wk(&water(), "def2-svp");
}

#[test]
fn wk_fragment_tzvp() {
    check_wk(&fragment(5), "def2-tzvp");
}

/// The Ozaki exchange switched on and off on a built engine: K within round-off of native at
/// 7 slices, the error growing as slices are removed, native K bit for bit after switching
/// back, the held bytes restored.
#[test]
fn ozaki_mode_switch() {
    let Some(gpu) = common::gpu() else { return };
    let mol = fragment(5);
    let basis = harness::basis(&mol, "def2-tzvp").unwrap();
    let aux = harness::basis(&mol, "def2-universal-jkfit").unwrap();
    let nbf = basis.nbf;
    let (c, _) = factors(nbf, 6.min(nbf), 3);
    let fs = [Factor {
        c: &c,
        ncol: 6.min(nbf),
    }];
    let req = GpuJkRequest {
        dens: &[],
        factors: &fs,
        want_j: false,
        want_k: true,
    };
    let opts = GpuDfJkOptions {
        k_block: 32,
        ..Default::default()
    };
    let mut eng = GpuDfJk::new(&gpu, &basis, &aux, &opts).unwrap();
    let held = eng.resolved().device_bytes;
    let native = eng.compute(&gpu, &req).unwrap().k;
    let kmax = native[0].iter().fold(0.0f64, |m, v| m.max(v.abs()));
    let mut last = 0.0;
    for slices in [7usize, 5, 3] {
        eng.set_k_math(&gpu, KMath::Ozaki { slices }, false)
            .unwrap();
        assert_eq!(eng.resolved().k_math, KMath::Ozaki { slices });
        assert!(eng.resolved().device_bytes > held);
        let k = eng.compute(&gpu, &req).unwrap().k;
        let err = k[0]
            .iter()
            .zip(&native[0])
            .fold(0.0f64, |m, (a, b)| m.max((a - b).abs()))
            / kmax;
        eprintln!("Ozaki {slices} slices: max |dK| / max |K| {err:.2e}");
        let bound = if slices == 7 {
            1e-13
        } else {
            2f64.powi(-(8 * slices as i32 - 1) + 12)
        };
        assert!(err < bound, "{slices} slices: {err:e}");
        assert!(err >= last);
        last = err;
    }
    assert!(eng
        .set_k_math(&gpu, KMath::Ozaki { slices: 9 }, false)
        .is_err());
    eng.set_k_math(&gpu, KMath::Native, false).unwrap();
    assert_eq!(eng.resolved().device_bytes, held);
    let back = eng.compute(&gpu, &req).unwrap().k;
    assert!(same_bits(&back[0], &native[0]));
}

/// The Ozaki exchange from pre-sliced planes and from blocks gathered per build give the same
/// K bit for bit (global and range-separated exchange, device and streamed storage, several
/// auxiliary blocks).
#[test]
fn ozaki_presliced_equals_gathered() {
    use linus_gpu::cpu::Operator;
    let Some(gpu) = common::gpu() else { return };
    let mol = fragment(5);
    let basis = harness::basis(&mol, "def2-svp").unwrap();
    let aux = harness::basis(&mol, "def2-universal-jkfit").unwrap();
    let nbf = basis.nbf;
    let (ca, _) = factors(nbf, 6.min(nbf), 5);
    let (cb, _) = factors(nbf, 4.min(nbf), 9);
    let fs = [
        Factor {
            c: &ca,
            ncol: 6.min(nbf),
        },
        Factor {
            c: &cb,
            ncol: 4.min(nbf),
        },
    ];
    let req = GpuJkRequest {
        dens: &[],
        factors: &fs,
        want_j: false,
        want_k: true,
    };
    let wb97x = Operator::from_libxc(1.0, -0.842294, 0.3);
    for (storage, op) in [
        (DfStorage::Device, None),
        (DfStorage::Host, None),
        (DfStorage::Device, Some(wb97x)),
        (DfStorage::Host, Some(wb97x)),
    ] {
        let mut ks = vec![];
        for presliced in [true, false] {
            let opts = GpuDfJkOptions {
                storage,
                lr_storage: storage,
                k_block: 40,
                k_math: KMath::Ozaki { slices: 7 },
                k_presliced: presliced,
                omega: op.map(|o| o.omega()),
                ..Default::default()
            };
            let mut eng = GpuDfJk::new(&gpu, &basis, &aux, &opts).unwrap();
            if let Some(o) = op {
                eng.set_exchange_operator(&gpu, o).unwrap();
            }
            let planes = eng.resolved().k_planes_bytes;
            assert_eq!(planes > 0, presliced, "{storage:?} {op:?}");
            ks.push(eng.compute(&gpu, &req).unwrap().k);
        }
        for (a, b) in ks[0].iter().zip(&ks[1]) {
            assert!(
                same_bits(a, b),
                "{storage:?} {op:?}: planes and gathered blocks differ"
            );
        }
    }
}
