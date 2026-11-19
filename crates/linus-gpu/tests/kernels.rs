//! Kernels against linus's CPU implementations, the kernel cache, and run-to-run determinism.
#![cfg(feature = "gpu")]

mod common;

use linus_gpu::cpu::{eval_ao, ncomp, GridSpec};
use linus_gpu::harness::{self, compare};
use linus_gpu::kernels::colloc::GpuCollocator;
use linus_gpu::kernels::eri3c::{Class, ClassWork, GpuEri3c, Variant};
use linus_gpu::Gpu;

#[test]
fn kernel_cache_compiles_once_and_reloads_from_disk() {
    let Some(_g) = common::gpu() else { return };
    let dir = std::path::Path::new(env!("CARGO_TARGET_TMPDIR"))
        .join(format!("linus-gpu-cache-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let src = "extern \"C\" __global__ void k(double *x) { x[threadIdx.x] *= 2.0; }";
    let gpu = Gpu::open(0, Some(dir.clone()), None).unwrap();
    gpu.kernel("t", src, "k").unwrap();
    gpu.kernel("t", src, "k").unwrap();
    let s = gpu.kernel_cache().stats();
    assert_eq!((s.compiled, s.memory_hits, s.disk_hits), (1, 1, 0));
    gpu.kernel_cache().clear_memory();
    gpu.kernel("t", src, "k").unwrap();
    assert_eq!(gpu.kernel_cache().stats().disk_hits, 1);
    // a new process-level cache finds the file; a changed source does not
    let gpu2 = Gpu::open(0, Some(dir.clone()), None).unwrap();
    gpu2.kernel("t", src, "k").unwrap();
    gpu2.kernel("t", &format!("{src}\n// changed"), "k")
        .unwrap();
    let s2 = gpu2.kernel_cache().stats();
    assert_eq!((s2.disk_hits, s2.compiled), (1, 1));
    let files = std::fs::read_dir(dir.join(gpu.info().arch()))
        .map(|d| d.count())
        .unwrap_or(0);
    assert_eq!(files, 2, "one cached image per distinct source");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn compile_errors_carry_the_log() {
    let Some(gpu) = common::gpu() else { return };
    let err = gpu
        .kernel(
            "bad",
            "extern \"C\" __global__ void k() { undeclared(); }",
            "k",
        )
        .unwrap_err();
    assert!(err.to_string().contains("undeclared"), "{err}");
}

#[test]
fn collocation_matches_eval_ao_on_u1an39_tzvp() {
    let Some(gpu) = common::gpu() else { return };
    // every 50th point of the grid: all atoms and radial shells are represented
    check_collocation(
        &gpu,
        &harness::structure("u1an39").unwrap(),
        "def2-tzvp",
        50,
    );
}

#[test]
fn collocation_matches_eval_ao_with_g_shells() {
    let Some(gpu) = common::gpu() else { return };
    let mol = small_molecule();
    assert_eq!(harness::basis(&mol, "def2-qzvp").unwrap().max_l(), 4);
    check_collocation(&gpu, &mol, "def2-qzvp", 20);
}

/// GPU collocation (value and gradient) of `basis_name` on every `step`-th point of the PySCF
/// level-3 grid against linus-ints' `eval_ao`.
fn check_collocation(gpu: &Gpu, mol: &harness::Structure, basis_name: &str, step: usize) {
    let basis = harness::basis(mol, basis_name).unwrap();
    let grid = harness::grid(mol, &GridSpec::pyscf(3)).unwrap();
    let pick: Vec<usize> = (0..grid.len()).step_by(step).collect();
    let np = pick.len();
    let (x, y, z): (Vec<f64>, Vec<f64>, Vec<f64>) = (
        pick.iter().map(|&i| grid.x[i]).collect(),
        pick.iter().map(|&i| grid.y[i]).collect(),
        pick.iter().map(|&i| grid.z[i]).collect(),
    );
    let nao = basis.nbf;
    let order = 1;
    let nc = ncomp(order);
    let mut cpu = vec![0.0; nc * nao * np];
    let shells: Vec<usize> = (0..basis.shells.len()).collect();
    eval_ao(&basis, &shells, &x, &y, &z, order, &mut cpu);

    let col = GpuCollocator::new(gpu, &basis, order).unwrap();
    let (dx, dy, dz) = (
        gpu.upload(&x).unwrap(),
        gpu.upload(&y).unwrap(),
        gpu.upload(&z).unwrap(),
    );
    let mut out = gpu.alloc_zeros::<f64>(nc * nao * np).unwrap();
    col.eval(gpu, &dx, &dy, &dz, np, np, &mut out).unwrap();
    let g1 = gpu.download(&out).unwrap();
    col.eval(gpu, &dx, &dy, &dz, np, np, &mut out).unwrap();
    let g2 = gpu.download(&out).unwrap();
    assert!(
        g1.iter().zip(&g2).all(|(a, b)| a.to_bits() == b.to_bits()),
        "collocation not deterministic"
    );

    // per component: relative to the largest value of that component
    for d in 0..nc {
        let r = d * nao * np..(d + 1) * nao * np;
        let st = compare(&g1[r.clone()], &cpu[r]);
        eprintln!("{basis_name} (lmax {}) component {d}: {st}", basis.max_l());
        st.check(0.0, 2e-14).unwrap();
    }
}

fn small_molecule() -> harness::Structure {
    // the first 14 atoms of u1an39 (C, F, N, O, H): enough d shells for every code path
    let mut m = harness::structure("u1an39").unwrap();
    m.z.truncate(14);
    m.positions.truncate(14);
    m
}

#[test]
fn eri3c_dd_d_matches_pair_block() {
    let Some(gpu) = common::gpu() else { return };
    let mol = small_molecule();
    let basis = harness::basis(&mol, "def2-tzvp").unwrap();
    let aux = harness::basis(&mol, "def2-universal-jkfit").unwrap();
    for class in [
        Class {
            la: 2,
            lb: 2,
            lk: 2,
        },
        Class {
            la: 2,
            lb: 1,
            lk: 3,
        },
        Class {
            la: 1,
            lb: 0,
            lk: 0,
        },
    ] {
        let work = ClassWork::new(&basis, &aux, class);
        assert!(
            !work.pairs.is_empty() && !work.aux.shells.is_empty(),
            "{}",
            class.tag()
        );
        let cpu = work.cpu();
        for v in [Variant::H, Variant::G] {
            let k = GpuEri3c::new(&gpu, &work, v).unwrap();
            let mut out = gpu.alloc_zeros::<f64>(k.out_len()).unwrap();
            k.run(&gpu, &mut out).unwrap();
            let g1 = gpu.download(&out).unwrap();
            k.run(&gpu, &mut out).unwrap();
            let g2 = gpu.download(&out).unwrap();
            assert!(
                g1.iter().zip(&g2).all(|(a, b)| a.to_bits() == b.to_bits()),
                "{} {v:?}: not deterministic",
                class.tag()
            );
            let st = compare(&g1, &cpu);
            eprintln!(
                "{} {v:?}: {} pairs, {} aux shells: {st}",
                class.tag(),
                work.pairs.len(),
                work.aux.shells.len()
            );
            st.check(0.0, 1e-13).unwrap();
        }
    }
}
