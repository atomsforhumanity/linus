//! The variable-size batched DGEMM (`kernels::vgemm`) against cuBLAS's DGEMM, product by
//! product, for all four transpositions, odd shapes, beta 0 and 1, and run to run; cuBLAS's
//! grouped GEMM wrapper (`Blas::gemm_grouped`) in FP64 and FP32 on the same products.
#![cfg(feature = "gpu")]

mod common;

use common::Rng;
use linus_gpu::blas::{Blas, GemmDims, GemmShape, MatMut, MatRef, Op};
use linus_gpu::kernels::vgemm::{Vgemm, VgemmProduct, VgemmTile};

#[test]
fn vgemm_matches_cublas() {
    let Some(gpu) = common::gpu() else { return };
    let blas = Blas::new(&gpu).unwrap();
    let mut rng = Rng(11);
    // (m, n, k) with edges that do not divide the tiles
    let shapes = [
        (1, 1, 1),
        (33, 17, 5),
        (64, 64, 16),
        (130, 97, 211),
        (7, 300, 129),
        (256, 31, 0),
    ];
    for tile in [
        VgemmTile::DEFAULT,
        VgemmTile {
            bm: 64,
            bn: 32,
            bk: 8,
            tx: 16,
            ty: 8,
        },
    ] {
        let vg = Vgemm::new(&gpu, tile).unwrap();
        for (ta, tb) in [(false, false), (true, false), (false, true), (true, true)] {
            for beta in [0.0, 1.0] {
                // operands stored with padded leading dimensions
                let mut a_h = vec![];
                let mut b_h = vec![];
                let mut offs = vec![];
                let (mut sa, mut sb, mut sc) = (0, 0, 0);
                for &(m, n, k) in &shapes {
                    let (ar, ac) = if ta { (k, m) } else { (m, k) };
                    let (br, bc) = if tb { (n, k) } else { (k, n) };
                    let (lda, ldb, ldc) = (ar.max(1) + 3, br.max(1) + 1, m + 2);
                    offs.push((sa, lda, sb, ldb, sc, ldc));
                    sa += lda * ac.max(1);
                    sb += ldb * bc.max(1);
                    sc += ldc * n;
                }
                a_h.extend((0..sa).map(|_| rng.uniform() - 0.5));
                b_h.extend((0..sb).map(|_| rng.uniform() - 0.5));
                let c0: Vec<f64> = (0..sc).map(|_| rng.uniform() - 0.5).collect();
                let a = gpu.upload(&a_h).unwrap();
                let b = gpu.upload(&b_h).unwrap();
                let mut c_ref = gpu.upload(&c0).unwrap();
                let c_vg = gpu.upload(&c0).unwrap();
                let opa = if ta { Op::T } else { Op::N };
                let opb = if tb { Op::T } else { Op::N };
                let mut prods = vec![];
                for (&(m, n, k), &(oa, lda, ob, ldb, oc, ldc)) in shapes.iter().zip(&offs) {
                    if k > 0 {
                        blas.gemm(
                            GemmDims::new(opa, opb, m, n, k),
                            (1.0, beta),
                            MatRef::new(&a, oa, lda),
                            MatRef::new(&b, ob, ldb),
                            MatMut::new(&mut c_ref, oc, ldc),
                        )
                        .unwrap();
                    } else if beta == 0.0 {
                        gpu.zero_range(&mut c_ref, oc, ldc * n).unwrap();
                    }
                    prods.push(VgemmProduct {
                        m,
                        n,
                        k,
                        a: a.at(oa).addr(),
                        lda,
                        b: b.at(ob).addr(),
                        ldb,
                        c: c_vg.at(oc).addr(),
                        ldc,
                    });
                }
                // SAFETY: the products address the live buffers above; the C blocks are disjoint.
                let plan = unsafe { vg.plan(&gpu, &prods) }.unwrap();
                unsafe { vg.launch(&gpu, ta, tb, 1.0, beta, &plan) }.unwrap();
                let got = gpu.download(&c_vg).unwrap();
                let want = gpu.download(&c_ref).unwrap();
                // the padding between C blocks is untouched; the products match to round-off
                // (both are FMA chains over k in order)
                let mut err = 0.0f64;
                for (&(m, n, _), &(_, _, _, _, oc, ldc)) in shapes.iter().zip(&offs) {
                    for j in 0..n {
                        for i in 0..ldc {
                            let (x, y) = (got[oc + j * ldc + i], want[oc + j * ldc + i]);
                            if i >= m {
                                assert_eq!(x.to_bits(), c0[oc + j * ldc + i].to_bits());
                            } else {
                                err = err.max((x - y).abs());
                            }
                        }
                    }
                }
                eprintln!("vgemm {tile:?} ta {ta} tb {tb} beta {beta}: max |diff| {err:.2e}");
                assert!(err < 1e-12, "vgemm differs from cuBLAS: {err:e}");
                // run to run
                let c2 = gpu.upload(&c0).unwrap();
                let prods2: Vec<VgemmProduct> = prods
                    .iter()
                    .zip(&offs)
                    .map(|(p, o)| VgemmProduct {
                        c: c2.at(o.4).addr(),
                        ..*p
                    })
                    .collect();
                // SAFETY: as above, into the second output.
                let plan2 = unsafe { vg.plan(&gpu, &prods2) }.unwrap();
                unsafe { vg.launch(&gpu, ta, tb, 1.0, beta, &plan2) }.unwrap();
                let again = gpu.download(&c2).unwrap();
                assert!(got
                    .iter()
                    .zip(&again)
                    .all(|(x, y)| x.to_bits() == y.to_bits()));
            }
        }
    }
}

#[test]
fn grouped_gemm_matches_one_gemm_per_product() {
    let Some(gpu) = common::gpu() else { return };
    let blas = Blas::new(&gpu).unwrap();
    let mut rng = Rng(5);
    let shapes = [(40usize, 30usize, 20usize), (3, 70, 11), (129, 64, 65)];
    let (mut sa, mut sb, mut sc) = (0, 0, 0);
    let mut offs = vec![];
    for &(m, n, k) in &shapes {
        offs.push((sa, sb, sc));
        sa += m * k;
        sb += k * n;
        sc += m * n;
    }
    let ah: Vec<f64> = (0..sa).map(|_| rng.uniform() - 0.5).collect();
    let bh: Vec<f64> = (0..sb).map(|_| rng.uniform() - 0.5).collect();
    let a = gpu.upload(&ah).unwrap();
    let b = gpu.upload(&bh).unwrap();
    let mut c1 = gpu.alloc_zeros::<f64>(sc).unwrap();
    let c2 = gpu.alloc_zeros::<f64>(sc).unwrap();
    let a32 = gpu
        .upload(&ah.iter().map(|&x| x as f32).collect::<Vec<f32>>())
        .unwrap();
    let b32 = gpu
        .upload(&bh.iter().map(|&x| x as f32).collect::<Vec<f32>>())
        .unwrap();
    let c32 = gpu.alloc_zeros::<f32>(sc).unwrap();
    let mut gs = vec![];
    let (mut p64, mut p32) = (vec![vec![]; 3], vec![vec![]; 3]);
    for (&(m, n, k), &(oa, ob, oc)) in shapes.iter().zip(&offs) {
        blas.gemm(
            GemmDims::new(Op::N, Op::N, m, n, k),
            (1.0, 0.0),
            MatRef::new(&a, oa, m),
            MatRef::new(&b, ob, k),
            MatMut::new(&mut c1, oc, m),
        )
        .unwrap();
        gs.push(GemmShape {
            m: m as i32,
            n: n as i32,
            k: k as i32,
            lda: m as i32,
            ldb: k as i32,
            ldc: m as i32,
        });
        p64[0].push(a.at(oa).addr());
        p64[1].push(b.at(ob).addr());
        p64[2].push(c2.at(oc).addr());
        p32[0].push(a32.at(oa).addr());
        p32[1].push(b32.at(ob).addr());
        p32[2].push(c32.at(oc).addr());
    }
    let d64 = gpu.upload(&p64.concat()).unwrap();
    let d32 = gpu.upload(&p32.concat()).unwrap();
    // SAFETY: the pointers address the live buffers above; the outputs are disjoint.
    unsafe {
        blas.gemm_grouped((Op::N, Op::N), (1.0f64, 0.0), &gs, &d64, 0)
            .unwrap();
        blas.gemm_grouped((Op::N, Op::N), (1.0f32, 0.0), &gs, &d32, 0)
            .unwrap();
    }
    let (want, got, got32) = (
        gpu.download(&c1).unwrap(),
        gpu.download(&c2).unwrap(),
        gpu.download(&c32).unwrap(),
    );
    let e64 = want
        .iter()
        .zip(&got)
        .fold(0.0f64, |m, (x, y)| m.max((x - y).abs()));
    let e32 = want
        .iter()
        .zip(&got32)
        .fold(0.0f64, |m, (x, y)| m.max((x - *y as f64).abs()));
    eprintln!("grouped DGEMM {e64:.2e}, grouped SGEMM {e32:.2e}");
    assert!(e64 < 1e-13 && e32 < 1e-4);
}
