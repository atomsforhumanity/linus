//! The Ozaki-scheme FP64 emulation (`linus_gpu::emu`) on this card: which INT8 GEMM shapes
//! cuBLAS takes (`probe`), INT8 GEMM throughput (`int8`), and emulated DGEMMs against native
//! DGEMM and cuBLAS's fixed-mantissa emulation on the DF-K shapes (`dfk`).
//!
//! ```text
//! ozaki_bench probe
//! ozaki_bench int8 [m n k]...
//! ozaki_bench dfk [--nbf 2155] [--nq 112] [--nocc 233] [--slices 6,7] [--reps 3]
//! ```

mod common;
use common::{device_ms, median};
use linus_gpu::blas::{Blas, Fp64Mode, GemmDims, MatMut, MatRef, Op};
use linus_gpu::emu::{Order, Ozaki, OzakiWork, Sliced, Vectors};
use linus_gpu::{Gpu, GpuError};

/// The TN INT8 GEMM shapes cuBLAS refuses (`CUBLAS_STATUS_NOT_SUPPORTED`) over a grid of
/// output rows m, columns n and inner dimensions k (multiples of 16, as `Sliced` pads them).
fn probe(gpu: &Gpu) -> Result<(), GpuError> {
    let blas = Blas::new(gpu)?;
    let ms = [
        4usize, 8, 12, 16, 24, 32, 36, 40, 48, 100, 104, 112, 2144, 2152, 2156, 2160, 2176,
    ];
    let ns = [1usize, 3, 4, 8, 20, 23, 24, 64, 65, 120, 233];
    let ks = [16usize, 32, 48, 64, 80, 96, 112, 128, 144, 208, 1008, 30000];
    let a = gpu.alloc_zeros::<i8>(2176 * 30000)?;
    let mut c = gpu.alloc_zeros::<i32>(2176 * 233)?;
    let (mut ok, mut refused) = (0, vec![]);
    for &m in &ms {
        for &k in &ks {
            for &n in &ns {
                match blas.gemm_i8(
                    GemmDims::new(Op::T, Op::N, m, n, k),
                    (1, 0),
                    MatRef::new(&a, 0, k),
                    MatRef::new(&a, 0, k),
                    MatMut::new(&mut c, 0, m),
                ) {
                    Ok(()) => ok += 1,
                    Err(_) => refused.push((m, n, k)),
                }
            }
        }
    }
    gpu.synchronize()?;
    println!("{ok} shapes ran, {} refused", refused.len());
    for m in ms {
        let r: Vec<String> = refused
            .iter()
            .filter(|x| x.0 == m)
            .map(|x| format!("{}x{}", x.1, x.2))
            .collect();
        if !r.is_empty() {
            println!(
                "  m {m} (m % 16 = {}): n x k refused: {}",
                m % 16,
                r.join(" ")
            );
        }
    }
    Ok(())
}

fn int8_rate(gpu: &Gpu, shapes: &[(usize, usize, usize)]) -> Result<(), GpuError> {
    let blas = Blas::new(gpu)?;
    for &(m, n, k) in shapes {
        let a = gpu.alloc_zeros::<i8>(m * k)?;
        let b = gpu.alloc_zeros::<i8>(n * k)?;
        let mut c = gpu.alloc_zeros::<i32>(m * n)?;
        let run = |c: &mut linus_gpu::DeviceBuf<i32>| {
            blas.gemm_i8(
                GemmDims::new(Op::T, Op::N, m, n, k),
                (1, 0),
                MatRef::new(&a, 0, k),
                MatRef::new(&b, 0, k),
                MatMut::new(c, 0, m),
            )
        };
        run(&mut c)?;
        let mut ts = vec![];
        for _ in 0..5 {
            ts.push(device_ms(gpu, || run(&mut c))?);
        }
        ts.sort_by(|x, y| x.partial_cmp(y).unwrap());
        let ms = ts[2];
        println!(
            "int8 m {m} n {n} k {k}: {ms:.3} ms, {:.1} TOPS",
            2.0 * (m * n * k) as f64 / ms / 1e9
        );
    }
    Ok(())
}

/// GEMM1 (X = A^T C: (nq nbf) x nocc, inner nbf) and GEMM2 (K = X X^T: nbf x nbf, inner
/// nq nocc) with smooth random data, native vs cuBLAS emulation vs Ozaki.
fn dfk(
    gpu: &Gpu,
    nbf: usize,
    nq: usize,
    nocc: usize,
    slices: &[usize],
    reps: usize,
) -> Result<(), GpuError> {
    let mut rng = 0x1234_5678_9abc_def0u64;
    let mut uni = || {
        rng ^= rng >> 12;
        rng ^= rng << 25;
        rng ^= rng >> 27;
        (rng.wrapping_mul(0x2545_f491_4f6c_dd1d) >> 11) as f64 / (1u64 << 53) as f64 - 0.5
    };
    let m1 = nq * nbf;
    let a: Vec<f64> = (0..nbf * m1).map(|_| uni()).collect();
    let c: Vec<f64> = (0..nbf * nocc).map(|_| uni()).collect();
    let da = gpu.upload(&a)?;
    drop(a);
    let dc = gpu.upload(&c)?;
    let mut x = gpu.alloc_zeros::<f64>(m1 * nocc)?;
    let mut k = gpu.alloc_zeros::<f64>(nbf * nbf)?;
    let native = Blas::new(gpu)?;
    let emu = Blas::with_mode(gpu, Fp64Mode::fixed(55))?;
    let jj = nq * nocc;
    let time = |f: &mut dyn FnMut() -> Result<(), GpuError>| -> Result<f64, GpuError> {
        f()?;
        let mut ts = vec![];
        for _ in 0..reps {
            ts.push(device_ms(gpu, &mut *f)?);
        }
        Ok(median(ts))
    };
    let g1 = |b: &Blas, x: &mut linus_gpu::DeviceBuf<f64>| {
        b.gemm(
            GemmDims::new(Op::T, Op::N, m1, nocc, nbf),
            (1.0, 0.0),
            MatRef::new(&da, 0, nbf),
            MatRef::new(&dc, 0, nbf),
            MatMut::new(x, 0, m1),
        )
    };
    let t1n = time(&mut || g1(&native, &mut x))?;
    let xn = gpu.download(&x)?;
    let t1e = time(&mut || g1(&emu, &mut x))?;
    let g2 = |b: &Blas, x: &linus_gpu::DeviceBuf<f64>, k: &mut linus_gpu::DeviceBuf<f64>| {
        b.gemm(
            GemmDims::new(Op::N, Op::T, nbf, nbf, jj),
            (1.0, 0.0),
            MatRef::new(x, 0, nbf),
            MatRef::new(x, 0, nbf),
            MatMut::new(k, 0, nbf),
        )
    };
    gpu.upload_into(&mut x, 0, &xn)?;
    let t2n = time(&mut || g2(&native, &x, &mut k))?;
    let kn = gpu.download(&k)?;
    let t2e = time(&mut || g2(&emu, &x, &mut k))?;
    let gf = |m: usize, n: usize, kk: usize, ms: f64| 2.0 * (m * n * kk) as f64 / ms / 1e9;
    println!(
        "nbf {nbf} nq {nq} nocc {nocc}: GEMM1 {m1} x {nocc} x {nbf}, GEMM2 {nbf} x {nbf} x {jj}"
    );
    println!(
        "  native   GEMM1 {t1n:8.2} ms ({:.2} TF)  GEMM2 {t2n:8.2} ms ({:.2} TF)",
        gf(m1, nocc, nbf, t1n),
        gf(nbf, nbf, jj, t2n)
    );
    println!("  cublas55 GEMM1 {t1e:8.2} ms  GEMM2 {t2e:8.2} ms");
    let mut oz = Ozaki::new(gpu)?;
    let mut work = OzakiWork::default();
    let kmax = kn.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    let xmax = xn.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    for &s in slices {
        // the stored operand sliced once (not timed), C per call
        let mut sa = Sliced::new(gpu, m1, nbf, s, Order::Ascending)?;
        let ts = std::time::Instant::now();
        oz.slice(
            gpu,
            Vectors {
                buf: &da,
                off: 0,
                ld: nbf,
                contiguous: true,
            },
            &mut sa,
            &mut work,
        )?;
        gpu.synchronize()?;
        let t_slice_a = ts.elapsed().as_secs_f64() * 1e3;
        let mut sc = Sliced::new(gpu, nocc, nbf, s, Order::Descending)?;
        let t1 = time(&mut || {
            oz.slice(
                gpu,
                Vectors {
                    buf: &dc,
                    off: 0,
                    ld: nbf,
                    contiguous: true,
                },
                &mut sc,
                &mut work,
            )?;
            oz.gemm(
                gpu,
                sa.view(),
                sc.view(),
                1.0,
                0.0,
                (&mut x, 0, m1),
                &mut work,
            )
        })?;
        let xo = gpu.download(&x)?;
        // GEMM1 again without pairing diagonals: the time, and the same bits
        let pair = oz.pair_max_n;
        oz.pair_max_n = 0;
        let t1u = time(&mut || {
            oz.gemm(
                gpu,
                sa.view(),
                sc.view(),
                1.0,
                0.0,
                (&mut x, 0, m1),
                &mut work,
            )
        })?;
        oz.pair_max_n = pair;
        let same = gpu
            .download(&x)?
            .iter()
            .zip(&xo)
            .all(|(p, q)| p.to_bits() == q.to_bits());
        println!("  ozaki{s}   GEMM1 without pairing {t1u:8.2} ms (with pairing {t1:.2} ms incl. slicing C); same bits {same}");
        let ex = xo
            .iter()
            .zip(&xn)
            .fold(0.0f64, |m, (p, q)| m.max((p - q).abs()))
            / xmax;
        gpu.upload_into(&mut x, 0, &xn)?;
        let mut sx = Sliced::new(gpu, nbf, jj, s, Order::Ascending)?;
        let mut sxd = Sliced::new(gpu, nbf, jj, s, Order::Descending)?;
        let t2 = time(&mut || {
            oz.slice(
                gpu,
                Vectors {
                    buf: &x,
                    off: 0,
                    ld: nbf,
                    contiguous: false,
                },
                &mut sx,
                &mut work,
            )?;
            oz.slice(
                gpu,
                Vectors {
                    buf: &x,
                    off: 0,
                    ld: nbf,
                    contiguous: false,
                },
                &mut sxd,
                &mut work,
            )?;
            oz.syrk_lower(
                gpu,
                sx.view(),
                sxd.view(),
                1.0,
                0.0,
                (&mut k, 0, nbf),
                8,
                &mut work,
            )
        })?;
        let ko = gpu.download(&k)?;
        let mut ek = 0.0f64;
        for j in 0..nbf {
            for i in j..nbf {
                ek = ek.max((ko[j * nbf + i] - kn[j * nbf + i]).abs());
            }
        }
        println!(
            "  ozaki{s}   GEMM1 {t1:8.2} ms  GEMM2 {t2:8.2} ms (lower)  slicing A once {t_slice_a:.1} ms  max|dX|/max|X| {ex:.1e}  max|dK|/max|K| {:.1e}  work {:.2} GiB",
            ek / kmax,
            work.bytes() as f64 / 1073741824.0
        );
    }
    Ok(())
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let gpu = Gpu::new(0).expect("GPU");
    let get = |k: &str| a.iter().position(|x| x == k).map(|i| a[i + 1].clone());
    match a.get(1).map(String::as_str) {
        Some("probe") => probe(&gpu).unwrap(),
        Some("int8") => {
            let v: Vec<usize> = a[2..].iter().map(|s| s.parse().unwrap()).collect();
            let shapes: Vec<(usize, usize, usize)> = if v.is_empty() {
                vec![
                    (8192, 8192, 8192),
                    (241360, 233, 2160),
                    (241360, 240, 15120),
                    (2155, 2155, 26096),
                    (2160, 256, 26096),
                ]
            } else {
                v.chunks(3).map(|c| (c[0], c[1], c[2])).collect()
            };
            int8_rate(&gpu, &shapes).unwrap()
        }
        Some("dfk") => {
            let n = |k: &str, d: usize| get(k).map(|s| s.parse().unwrap()).unwrap_or(d);
            let slices: Vec<usize> = get("--slices")
                .unwrap_or_else(|| "6,7".into())
                .split(',')
                .map(|s| s.parse().unwrap())
                .collect();
            dfk(
                &gpu,
                n("--nbf", 2155),
                n("--nq", 112),
                n("--nocc", 233),
                &slices,
                n("--reps", 3),
            )
            .unwrap()
        }
        _ => panic!("usage: ozaki_bench probe|int8|dfk"),
    }
}
