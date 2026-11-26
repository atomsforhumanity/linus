//! cuBLAS FP64 emulation against native DGEMM on the two GEMMs of density-fitted exchange:
//!
//!   GEMM1  B[(Q m), i] = sum_n L[Q, m, n] C[n, i]     (M = nQ nbf, N = nocc, K = nbf)
//!   GEMM2  K[m, n]    += sum_(Q i) B[m, (Q i)] B[n, (Q i)]  (M = N = nbf, K = nQ nocc)
//!
//! B from GEMM1 (column-major, (nQ nbf) x nocc) is already the nbf x (nQ nocc) operand of GEMM2,
//! so the two GEMMs run back to back without a transpose. Exchange energy:
//! E_x = -sum_mn (C C^T)_mn K_mn (closed shell, D = 2 C C^T).
//!
//! Real data (`--data DIR` from crates/linus-gpu/tools/dfk_data.py: PySCF's fitted L and converged C): speed
//! and accuracy against native FP64 and PySCF. Shapes only (`--shapes NAME`: a benchmark
//! molecule's def2-TZVP / def2-universal-jkfit sizes) with the real data's values tiled in: speed
//! and matrix-element accuracy, per block of `--block` auxiliary functions.
//!
//!   cargo run --release -p linus-gpu --features gpu --example gemm_emulation -- \
//!       --data bench/scratch/gpu-spike/dfk-u1an39 --shapes ts113 --block 512 --json out.json

mod common;
use common::{device_ms, median};
use linus_gpu::blas::{Blas, Fp64Mode, GemmDims, Mantissa, MatMut, MatRef, Op, Strategy};
use linus_gpu::harness::{self, compare};
use linus_gpu::{DeviceBuf, Gpu, GpuError};
use std::fmt::Write as _;
use std::path::PathBuf;

type R<T> = Result<T, GpuError>;

fn modes() -> Vec<(String, Fp64Mode)> {
    let mut v = vec![("native".to_string(), Fp64Mode::Native)];
    v.push((
        "performant/dynamic".into(),
        Fp64Mode::Emulated {
            strategy: Strategy::Performant,
            mantissa: Mantissa::Dynamic {
                offset: 0,
                max_bits: 0,
            },
        },
    ));
    for off in [-16, -8, 0, 8] {
        v.push((
            format!("eager/dynamic{off:+}"),
            Fp64Mode::Emulated {
                strategy: Strategy::Eager,
                mantissa: Mantissa::Dynamic {
                    offset: off,
                    max_bits: 0,
                },
            },
        ));
    }
    for bits in [31, 39, 47, 55, 63, 71, 79] {
        v.push((
            format!("eager/fixed{bits}"),
            Fp64Mode::Emulated {
                strategy: Strategy::Eager,
                mantissa: Mantissa::Fixed { bits },
            },
        ));
    }
    v
}

struct Data {
    name: String,
    nbf: usize,
    naux: usize,
    nocc: usize,
    ex_pyscf: f64,
    /// packed L[Q, m >= n]
    lp: Vec<f64>,
    /// column-major nbf x nocc
    c: Vec<f64>,
}

fn read_f64(p: &PathBuf) -> R<Vec<f64>> {
    let b = std::fs::read(p).map_err(|e| GpuError::Input(format!("{}: {e}", p.display())))?;
    Ok(b.as_chunks::<8>()
        .0
        .iter()
        .map(|c| f64::from_le_bytes(*c))
        .collect())
}

fn json_num(text: &str, key: &str) -> R<f64> {
    let i = text
        .find(&format!("\"{key}\""))
        .ok_or_else(|| GpuError::Input(format!("meta.json: no {key}")))?;
    let rest = &text[i + key.len() + 2..];
    let rest = rest[rest.find(':').unwrap() + 1..].trim_start();
    let end = rest.find([',', '\n', '}']).unwrap_or(rest.len());
    rest[..end]
        .trim()
        .parse()
        .map_err(|_| GpuError::Input(format!("meta.json: bad {key}")))
}

fn load(dir: &str) -> R<Data> {
    let d = PathBuf::from(dir);
    let meta = std::fs::read_to_string(d.join("meta.json"))
        .map_err(|e| GpuError::Input(format!("{dir}/meta.json: {e}")))?;
    let (nbf, naux, nocc) = (
        json_num(&meta, "nbf")? as usize,
        json_num(&meta, "naux")? as usize,
        json_num(&meta, "nocc")? as usize,
    );
    let lp = read_f64(&d.join("cderi.f64"))?;
    let c = read_f64(&d.join("c_occ.f64"))?;
    assert_eq!(lp.len(), naux * nbf * (nbf + 1) / 2);
    assert_eq!(c.len(), nbf * nocc);
    let name = d
        .file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default();
    Ok(Data {
        name,
        nbf,
        naux,
        nocc,
        ex_pyscf: json_num(&meta, "e_x_pyscf")?,
        lp,
        c,
    })
}

/// L[q0..q0+nq] unpacked to full nbf x nbf blocks.
fn unpack(d: &Data, q0: usize, nq: usize) -> Vec<f64> {
    let (n, np) = (d.nbf, d.nbf * (d.nbf + 1) / 2);
    let mut out = vec![0.0; nq * n * n];
    for q in 0..nq {
        let src = &d.lp[(q0 + q) * np..(q0 + q + 1) * np];
        let dst = &mut out[q * n * n..(q + 1) * n * n];
        for i in 0..n {
            for j in 0..=i {
                let v = src[i * (i + 1) / 2 + j];
                dst[i * n + j] = v;
                dst[j * n + i] = v;
            }
        }
    }
    out
}

/// -sum_mn P_mn K_mn with P = C C^T.
fn exchange_energy(p: &[f64], k: &[f64]) -> f64 {
    -p.iter().zip(k).map(|(a, b)| a * b).sum::<f64>()
}

struct Run {
    ms_total: f64,
    ms_gemm1: f64,
    ms_gemm2: f64,
    bits1: Option<i32>,
    bits2: Option<i32>,
}

/// One DF-K over `nblocks` blocks of nq auxiliary functions (all blocks use the same device
/// L block when `reuse` is set: shapes-only runs).
#[allow(clippy::too_many_arguments)]
fn dfk(
    gpu: &Gpu,
    blas: &Blas,
    l: &DeviceBuf<f64>,
    c: &DeviceBuf<f64>,
    b: &mut DeviceBuf<f64>,
    k: &mut DeviceBuf<f64>,
    nbf: usize,
    nocc: usize,
    blocks: &[(usize, usize)],
    reuse: bool,
) -> R<Run> {
    let (mut t1, mut t2) = (0.0, 0.0);
    let (mut bits1, mut bits2) = (None, None);
    for (bi, &(q0, nq)) in blocks.iter().enumerate() {
        let off = if reuse { 0 } else { q0 * nbf * nbf };
        t1 += device_ms(gpu, || {
            blas.gemm(
                GemmDims::new(Op::T, Op::N, nq * nbf, nocc, nbf),
                (1.0, 0.0),
                MatRef::new(l, off, nbf),
                MatRef::new(c, 0, nbf),
                MatMut::new(&mut *b, 0, nq * nbf),
            )
        })?;
        if bi == 0 {
            bits1 = blas.retained_bits(gpu)?;
        }
        let beta = if bi == 0 { 0.0 } else { 1.0 };
        t2 += device_ms(gpu, || {
            blas.gemm(
                GemmDims::new(Op::N, Op::T, nbf, nbf, nq * nocc),
                (1.0, beta),
                MatRef::new(&*b, 0, nbf),
                MatRef::new(&*b, 0, nbf),
                MatMut::new(&mut *k, 0, nbf),
            )
        })?;
        if bi == 0 {
            bits2 = blas.retained_bits(gpu)?;
        }
    }
    Ok(Run {
        ms_total: t1 + t2,
        ms_gemm1: t1,
        ms_gemm2: t2,
        bits1,
        bits2,
    })
}

fn fmt_bits(b: Option<i32>) -> String {
    b.map(|x| x.to_string()).unwrap_or_else(|| "-".into())
}

fn real_data(gpu: &Gpu, d: &Data, json: &mut String) -> R<()> {
    let (nbf, naux, nocc) = (d.nbf, d.naux, d.nocc);
    println!("\nreal data {}: nbf {nbf}, naux {naux}, nocc {nocc}; one block of all {naux} auxiliary functions", d.name);
    println!(
        "  GEMM1 {} x {nocc} x {nbf}, GEMM2 {nbf} x {nbf} x {}",
        naux * nbf,
        naux * nocc
    );
    let mut l = gpu.alloc_zeros::<f64>(naux * nbf * nbf)?;
    let step = 128;
    for q0 in (0..naux).step_by(step) {
        let nq = step.min(naux - q0);
        let h = unpack(d, q0, nq);
        gpu.upload_into(&mut l, q0 * nbf * nbf, &h)?;
    }
    let c = gpu.upload(&d.c)?;
    let mut b = gpu.alloc_zeros::<f64>(naux * nbf * nocc)?;
    let mut k = gpu.alloc_zeros::<f64>(nbf * nbf)?;
    // P = C C^T on the host
    let mut p = vec![0.0; nbf * nbf];
    for i in 0..nocc {
        let ci = &d.c[i * nbf..(i + 1) * nbf];
        for m in 0..nbf {
            let cm = ci[m];
            for n in 0..nbf {
                p[m * nbf + n] += cm * ci[n];
            }
        }
    }
    let flops = 2.0 * (naux * nbf) as f64 * nocc as f64 * nbf as f64
        + 2.0 * (nbf * nbf) as f64 * (naux * nocc) as f64;
    let blocks = [(0usize, naux)];
    let (mut b_ref, mut k_ref, mut ex_ref) = (Vec::new(), Vec::new(), 0.0);
    println!(
        "  {:<22} {:>9} {:>9} {:>9} {:>7} {:>6} {:>6} {:>10} {:>10} {:>11} {:>11} {:>9}",
        "mode",
        "total ms",
        "GEMM1 ms",
        "GEMM2 ms",
        "TFLOP/s",
        "bits1",
        "bits2",
        "max|dB|rel",
        "max|dK|rel",
        "dE_x native",
        "dE_x PySCF",
        "same bits"
    );
    for (name, mode) in modes() {
        let blas = Blas::with_mode(gpu, mode)?;
        dfk(
            gpu, &blas, &l, &c, &mut b, &mut k, nbf, nocc, &blocks, false,
        )?;
        let runs: Vec<Run> = (0..3)
            .map(|_| {
                dfk(
                    gpu, &blas, &l, &c, &mut b, &mut k, nbf, nocc, &blocks, false,
                )
            })
            .collect::<R<_>>()?;
        let r = Run {
            ms_total: median(runs.iter().map(|r| r.ms_total).collect()),
            ms_gemm1: median(runs.iter().map(|r| r.ms_gemm1).collect()),
            ms_gemm2: median(runs.iter().map(|r| r.ms_gemm2).collect()),
            bits1: runs[0].bits1,
            bits2: runs[0].bits2,
        };
        let bh = gpu.download(&b)?;
        let kh = gpu.download(&k)?;
        // run to run: one more DF-K must give the same bits
        dfk(
            gpu, &blas, &l, &c, &mut b, &mut k, nbf, nocc, &blocks, false,
        )?;
        let same = gpu
            .download(&k)?
            .iter()
            .zip(&kh)
            .all(|(x, y)| x.to_bits() == y.to_bits());
        let ex = exchange_energy(&p, &kh);
        if name == "native" {
            b_ref = bh.clone();
            k_ref = kh.clone();
            ex_ref = ex;
        }
        let (eb, ek) = (compare(&bh, &b_ref), compare(&kh, &k_ref));
        println!(
            "  {name:<22} {:>9.1} {:>9.1} {:>9.1} {:>7.2} {:>6} {:>6} {:>10.2e} {:>10.2e} {:>11.2e} {:>11.2e} {:>9}",
            r.ms_total,
            r.ms_gemm1,
            r.ms_gemm2,
            flops / r.ms_total / 1e9,
            fmt_bits(r.bits1),
            fmt_bits(r.bits2),
            eb.max_rel,
            ek.max_rel,
            ex - ex_ref,
            ex - d.ex_pyscf,
            same
        );
        let _ = write!(
            json,
            "{{\"data\":\"{}\",\"nbf\":{nbf},\"naux\":{naux},\"nocc\":{nocc},\"mode\":\"{name}\",\"ms_total\":{:.3},\"ms_gemm1\":{:.3},\"ms_gemm2\":{:.3},\"bits1\":{},\"bits2\":{},\"b_max_rel\":{:.3e},\"k_max_rel\":{:.3e},\"k_max_abs\":{:.3e},\"e_x\":{ex:.12},\"de_x_native\":{:.3e},\"de_x_pyscf\":{:.3e},\"run_to_run_identical\":{same}}},",
            d.name,
            r.ms_total,
            r.ms_gemm1,
            r.ms_gemm2,
            r.bits1.unwrap_or(0),
            r.bits2.unwrap_or(0),
            eb.max_rel,
            ek.max_rel,
            ek.max_abs,
            ex - ex_ref,
            ex - d.ex_pyscf
        );
    }
    Ok(())
}

fn shapes(gpu: &Gpu, d: &Data, molname: &str, nq: usize, json: &mut String) -> R<()> {
    let mol = harness::structure(molname)?;
    let nbf = harness::basis(&mol, "def2-tzvp")?.nbf;
    let naux = harness::basis(&mol, "def2-universal-jkfit")?.nbf;
    let nocc = (mol.nelectron() / 2) as usize;
    let nq = nq.min(naux);
    let nblocks = naux.div_ceil(nq);
    println!("\nshapes of {molname}/def2-TZVP: nbf {nbf}, naux {naux}, nocc {nocc}; {nblocks} blocks of {nq} auxiliary functions, values tiled from {}", d.name);
    println!(
        "  per block: GEMM1 {} x {nocc} x {nbf}, GEMM2 {nbf} x {nbf} x {}",
        nq * nbf,
        nq * nocc
    );
    // tile the real L values (first nq blocks, repeated) and C
    let src = unpack(d, 0, nq.min(d.naux));
    let lh: Vec<f64> = (0..nq * nbf * nbf).map(|i| src[i % src.len()]).collect();
    let l = gpu.upload(&lh)?;
    drop(lh);
    let ch: Vec<f64> = (0..nbf * nocc).map(|i| d.c[i % d.c.len()]).collect();
    let c = gpu.upload(&ch)?;
    let mut b = gpu.alloc_zeros::<f64>(nq * nbf * nocc)?;
    let mut k = gpu.alloc_zeros::<f64>(nbf * nbf)?;
    let blocks = [(0usize, nq)];
    let flops = 2.0 * (nq * nbf) as f64 * nocc as f64 * nbf as f64
        + 2.0 * (nbf * nbf) as f64 * (nq * nocc) as f64;
    let mut k_ref = Vec::new();
    println!(
        "  {:<22} {:>12} {:>9} {:>9} {:>7} {:>6} {:>6} {:>10}",
        "mode", "DF-K est. s", "GEMM1 ms", "GEMM2 ms", "TFLOP/s", "bits1", "bits2", "max|dK|rel"
    );
    for (name, mode) in modes() {
        let blas = Blas::with_mode(gpu, mode)?;
        dfk(gpu, &blas, &l, &c, &mut b, &mut k, nbf, nocc, &blocks, true)?;
        let runs: Vec<Run> = (0..3)
            .map(|_| dfk(gpu, &blas, &l, &c, &mut b, &mut k, nbf, nocc, &blocks, true))
            .collect::<R<_>>()?;
        let ms = median(runs.iter().map(|r| r.ms_total).collect());
        let (g1, g2) = (
            median(runs.iter().map(|r| r.ms_gemm1).collect()),
            median(runs.iter().map(|r| r.ms_gemm2).collect()),
        );
        let kh = gpu.download(&k)?;
        if name == "native" {
            k_ref = kh.clone();
        }
        let ek = compare(&kh, &k_ref);
        println!(
            "  {name:<22} {:>12.2} {:>9.1} {:>9.1} {:>7.2} {:>6} {:>6} {:>10.2e}",
            ms * nblocks as f64 / 1e3,
            g1,
            g2,
            flops / ms / 1e9,
            fmt_bits(runs[0].bits1),
            fmt_bits(runs[0].bits2),
            ek.max_rel
        );
        let _ = write!(
            json,
            "{{\"shapes\":\"{molname}\",\"nbf\":{nbf},\"naux\":{naux},\"nocc\":{nocc},\"block\":{nq},\"blocks\":{nblocks},\"mode\":\"{name}\",\"ms_block\":{ms:.3},\"ms_gemm1\":{g1:.3},\"ms_gemm2\":{g2:.3},\"s_total_est\":{:.3},\"bits1\":{},\"bits2\":{},\"k_max_rel\":{:.3e}}},",
            ms * nblocks as f64 / 1e3,
            runs[0].bits1.unwrap_or(0),
            runs[0].bits2.unwrap_or(0),
            ek.max_rel
        );
    }
    Ok(())
}

fn main() -> R<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let get = |k: &str| {
        args.iter()
            .position(|a| a == k)
            .and_then(|i| args.get(i + 1))
            .cloned()
    };
    let data = get("--data").ok_or_else(|| {
        GpuError::Input("--data DIR is required (crates/linus-gpu/tools/dfk_data.py)".into())
    })?;
    let block: usize = get("--block").map(|s| s.parse().unwrap()).unwrap_or(512);
    let gpu = Gpu::new(0)?;
    let blas = Blas::new(&gpu)?;
    println!(
        "{} ({}), cuBLAS {}",
        gpu.info().name,
        gpu.info().arch(),
        blas.version()?
    );
    drop(blas);
    let d = load(&data)?;
    let mut json = String::from("[");
    if !args.iter().any(|a| a == "--no-real") {
        real_data(&gpu, &d, &mut json)?;
    }
    if let Some(s) = get("--shapes") {
        for m in s.split(',') {
            shapes(&gpu, &d, m, block, &mut json)?;
        }
    }
    if json.ends_with(',') {
        json.pop();
    }
    json.push_str("]\n");
    if let Some(p) = get("--json") {
        std::fs::write(&p, json).map_err(|e| GpuError::Input(format!("{p}: {e}")))?;
    }
    Ok(())
}
