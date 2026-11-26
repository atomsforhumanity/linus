//! Grouped batched GEMMs (cuBLAS `cublas{D,S}gemmGroupedBatched`, one group per product) on the
//! shapes of the XC engine's per-batch products, against one GEMM per product and against a
//! single dense GEMM with the same flops: the density product X_b = phi_b D_b (np_b x nf_b x
//! nf_b) and the potential product V_b = phi_b^T W_b (nf_b x nf_b x np_b).
//!
//!   cargo run --release -p linus-gpu --features gpu --example grouped_gemm_bench -- [--count 256]

mod common;
use common::{device_ms, median};
use linus_gpu::blas::{Blas, GemmDims, GemmShape, MatMut, MatRef, Op};
use linus_gpu::kernels::vgemm::{Vgemm, VgemmProduct, VgemmTile};
use linus_gpu::{DeviceBuf, Gpu, GpuError};

type R<T> = Result<T, GpuError>;

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 11) as f64) / ((1u64 << 53) as f64)
    }
}

fn main() -> R<()> {
    let a: Vec<String> = std::env::args().collect();
    let get = |k: &str| a.iter().position(|x| x == k).map(|i| a[i + 1].clone());
    let count: usize = get("--count").map(|s| s.parse().unwrap()).unwrap_or(256);
    let (nf_lo, nf_hi): (usize, usize) = (
        get("--nf-lo").map(|s| s.parse().unwrap()).unwrap_or(150),
        get("--nf-hi").map(|s| s.parse().unwrap()).unwrap_or(450),
    );
    let gpu = Gpu::new(0)?;
    let blas = Blas::new(&gpu)?;
    let mut rng = Lcg(7);
    let shapes: Vec<(usize, usize)> = (0..count)
        .map(|_| {
            let np = 128 + (rng.next() * 129.0) as usize;
            let nf = nf_lo + (rng.next() * (nf_hi - nf_lo + 1) as f64) as usize;
            (np, nf)
        })
        .collect();
    let (mut sa, mut sb, mut sc) = (0, 0, 0);
    let mut offs = vec![];
    for &(np, nf) in &shapes {
        offs.push((sa, sb, sc));
        sa += 4 * np * nf;
        sb += nf * nf;
        sc += (np * nf).max(nf * nf);
    }
    let fill = |n: usize, rng: &mut Lcg| (0..n).map(|_| rng.next() - 0.5).collect::<Vec<f64>>();
    let ha = fill(sa, &mut rng);
    let hb = fill(sb, &mut rng);
    let da: DeviceBuf<f64> = gpu.upload(&ha)?;
    let db: DeviceBuf<f64> = gpu.upload(&hb)?;
    let mut dc: DeviceBuf<f64> = gpu.alloc_zeros(sc + sb)?;
    let fa: Vec<f32> = ha.iter().map(|&x| x as f32).collect();
    let fb: Vec<f32> = hb.iter().map(|&x| x as f32).collect();
    let sa32: DeviceBuf<f32> = gpu.upload(&fa)?;
    let sb32: DeviceBuf<f32> = gpu.upload(&fb)?;
    let sc32: DeviceBuf<f32> = gpu.alloc_zeros(sc + sb)?;
    for (label, density) in [
        ("density X = phi D", true),
        ("potential V = phi^T W", false),
    ] {
        let mut gs = vec![];
        let (mut pa, mut pb, mut pc) = (vec![], vec![], vec![]);
        let (mut pa32, mut pb32, mut pc32) = (vec![], vec![], vec![]);
        let mut flops = 0.0;
        for (&(np, nf), &(oa, ob, oc)) in shapes.iter().zip(&offs) {
            if density {
                gs.push(GemmShape {
                    m: np as i32,
                    n: nf as i32,
                    k: nf as i32,
                    lda: (4 * np) as i32,
                    ldb: nf as i32,
                    ldc: np as i32,
                });
                flops += 2.0 * (np * nf * nf) as f64;
                pb.push(db.at(ob).addr());
                pb32.push(sb32.at(ob).addr());
            } else {
                gs.push(GemmShape {
                    m: nf as i32,
                    n: nf as i32,
                    k: np as i32,
                    lda: (4 * np) as i32,
                    ldb: (4 * np) as i32,
                    ldc: nf as i32,
                });
                flops += 2.0 * (np * nf * nf) as f64;
                pb.push(da.at(oa + np).addr());
                pb32.push(sa32.at(oa + np).addr());
            }
            pa.push(da.at(oa).addr());
            pc.push(dc.at(oc).addr());
            pa32.push(sa32.at(oa).addr());
            pc32.push(sc32.at(oc).addr());
        }
        let ptr: Vec<u64> = pa.iter().chain(&pb).chain(&pc).copied().collect();
        let ptr32: Vec<u64> = pa32.iter().chain(&pb32).chain(&pc32).copied().collect();
        let dptr = gpu.upload(&ptr)?;
        let dptr32 = gpu.upload(&ptr32)?;
        let (ta, tb) = if density {
            (Op::N, Op::N)
        } else {
            (Op::T, Op::N)
        };
        let mut t_grouped = vec![];
        let mut t_loop = vec![];
        let mut t_s = vec![];
        for _ in 0..5 {
            // SAFETY: every pointer addresses a matrix of its shape inside the buffers above;
            // the outputs are disjoint.
            t_grouped.push(device_ms(&gpu, || unsafe {
                blas.gemm_grouped((ta, tb), (1.0f64, 0.0), &gs, &dptr, 0)
            })?);
            t_s.push(device_ms(&gpu, || unsafe {
                blas.gemm_grouped((ta, tb), (1.0f32, 0.0), &gs, &dptr32, 0)
            })?);
            t_loop.push(device_ms(&gpu, || {
                for (i, (&(np, nf), &(oa, ob, oc))) in shapes.iter().zip(&offs).enumerate() {
                    let _ = i;
                    if density {
                        blas.gemm(
                            GemmDims::new(Op::N, Op::N, np, nf, nf),
                            (1.0, 0.0),
                            MatRef::new(&da, oa, 4 * np),
                            MatRef::new(&db, ob, nf),
                            MatMut::new(&mut dc, oc, np),
                        )?;
                    } else {
                        blas.gemm(
                            GemmDims::new(Op::T, Op::N, nf, nf, np),
                            (1.0, 0.0),
                            MatRef::new(&da, oa, 4 * np),
                            MatRef::new(&da, oa + np, 4 * np),
                            MatMut::new(&mut dc, oc, nf),
                        )?;
                    }
                }
                Ok(())
            })?);
        }
        // reference result of the grouped DGEMM
        // SAFETY: as above.
        unsafe { blas.gemm_grouped((ta, tb), (1.0f64, 0.0), &gs, &dptr, 0) }?;
        let refc = gpu.download(&dc)?;
        let prods: Vec<VgemmProduct> = shapes
            .iter()
            .zip(&offs)
            .map(|(&(np, nf), &(oa, ob, oc))| {
                if density {
                    VgemmProduct {
                        m: np,
                        n: nf,
                        k: nf,
                        a: da.at(oa).addr(),
                        lda: 4 * np,
                        b: db.at(ob).addr(),
                        ldb: nf,
                        c: dc.at(oc).addr(),
                        ldc: np,
                    }
                } else {
                    VgemmProduct {
                        m: nf,
                        n: nf,
                        k: np,
                        a: da.at(oa).addr(),
                        lda: 4 * np,
                        b: da.at(oa + np).addr(),
                        ldb: 4 * np,
                        c: dc.at(oc).addr(),
                        ldc: nf,
                    }
                }
            })
            .collect();
        for tile in [
            VgemmTile::DEFAULT,
            VgemmTile {
                bm: 64,
                bn: 64,
                bk: 16,
                tx: 16,
                ty: 16,
            },
            VgemmTile {
                bm: 64,
                bn: 32,
                bk: 16,
                tx: 16,
                ty: 8,
            },
            VgemmTile {
                bm: 32,
                bn: 32,
                bk: 8,
                tx: 8,
                ty: 8,
            },
            VgemmTile {
                bm: 32,
                bn: 64,
                bk: 16,
                tx: 8,
                ty: 16,
            },
        ] {
            let vg = Vgemm::new(&gpu, tile)?;
            // SAFETY: the products' addresses are inside the live buffers above.
            let plan = unsafe { vg.plan(&gpu, &prods) }?;
            gpu.zero(&mut dc)?;
            let mut tv = vec![];
            for _ in 0..5 {
                // SAFETY: as above.
                tv.push(device_ms(&gpu, || unsafe {
                    vg.launch(&gpu, ta == Op::T, false, 1.0, 0.0, &plan)
                })?);
            }
            let got = gpu.download(&dc)?;
            let err = got
                .iter()
                .zip(&refc)
                .fold(0.0f64, |m, (a, b)| m.max((a - b).abs()));
            let v = median(tv);
            println!("  vgemm {tile:?}: {v:.2} ms ({:.2} TFLOP/s), {} tiles, max |diff| vs cuBLAS {err:.1e}", flops / v * 1e-9, plan.ntiles);
        }
        // cuBLAS's fixed-point emulation on the grouped DGEMM: does it emulate?
        let eblas = Blas::with_mode(&gpu, linus_gpu::blas::Fp64Mode::fixed(55))?;
        gpu.zero(&mut dc)?;
        let mut te = vec![];
        for _ in 0..5 {
            // SAFETY: as above.
            te.push(device_ms(&gpu, || unsafe {
                eblas.gemm_grouped((ta, tb), (1.0f64, 0.0), &gs, &dptr, 0)
            })?);
        }
        let bits = eblas.retained_bits(&gpu)?;
        let got = gpu.download(&dc)?;
        let same = got
            .iter()
            .zip(&refc)
            .all(|(a, b)| a.to_bits() == b.to_bits());
        let err = got
            .iter()
            .zip(&refc)
            .fold(0.0f64, |m, (a, b)| m.max((a - b).abs()));
        // and one plain emulated DGEMM of the first product, for comparison
        let (np0, nf0) = shapes[0];
        let _ = eblas.gemm(
            GemmDims::new(Op::N, Op::N, np0, nf0, nf0),
            (1.0, 0.0),
            MatRef::new(&da, 0, 4 * np0),
            MatRef::new(&db, 0, nf0),
            MatMut::new(&mut dc, 0, np0),
        );
        let bits1 = eblas.retained_bits(&gpu)?;
        // one emulated DGEMM per product
        let mut tl = vec![];
        for _ in 0..3 {
            tl.push(device_ms(&gpu, || {
                for (&(np, nf), &(oa, ob, oc)) in shapes.iter().zip(&offs) {
                    if density {
                        eblas.gemm(
                            GemmDims::new(Op::N, Op::N, np, nf, nf),
                            (1.0, 0.0),
                            MatRef::new(&da, oa, 4 * np),
                            MatRef::new(&db, ob, nf),
                            MatMut::new(&mut dc, oc, np),
                        )?;
                    } else {
                        eblas.gemm(
                            GemmDims::new(Op::T, Op::N, nf, nf, np),
                            (1.0, 0.0),
                            MatRef::new(&da, oa, 4 * np),
                            MatRef::new(&da, oa + np, 4 * np),
                            MatMut::new(&mut dc, oc, nf),
                        )?;
                    }
                }
                Ok(())
            })?);
        }
        println!(
            "  one emulated DGEMM (55 bits) per product: {:.2} ms ({:.2} TFLOP/s)",
            median(tl.clone()),
            flops / median(tl) * 1e-9
        );
        println!("  grouped DGEMM in emulated mode (55 bits): {:.2} ms, retained bits {bits:?} (one DGEMM: {bits1:?}), bitwise equal to native {same}, max |diff| {err:.1e}", median(te));
        let (g, l, s) = (median(t_grouped), median(t_loop), median(t_s));
        println!(
            "{label}: {count} products, {:.2} GFLOP: grouped {g:.2} ms ({:.2} TFLOP/s), loop {l:.2} ms ({:.2}), SGEMM grouped {s:.3} ms ({:.1} TFLOP/s)",
            flops * 1e-9,
            flops / g * 1e-9,
            flops / l * 1e-9,
            flops / s * 1e-9
        );
    }
    Ok(())
}
