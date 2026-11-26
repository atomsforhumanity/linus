//! FP64 tensor cores (DMMA, `mma.sync.m8n8k4.f64`) against plain FP64 FMA on the small matrix
//! products of a SHARK-style integral kernel, to decide whether a data-center kernel variant
//! pays. NVIDIA-only: the DMMA kernel is inline PTX, kept in this file as an isolated variant;
//! the portable kernels stay the reference. Runs on sm_80 / sm_90 (A100, H100: FP64 tensor
//! throughput twice FP64 FMA); elsewhere it skips, unless `--force` (correctness check only:
//! sm_120 has DMMA at a low rate).
//!
//!   cargo run --release -p linus-gpu --features gpu --example dmma_bench -- [--force] [--json out.json]
//!
//! Shapes, from the class sizes our generators use (spherical l >= 2, Cartesian s/p):
//! - SHARK T-step of (ab|cd): T[t][cd] = sum_u R[t][u] E_cd[u][cd]: M = nherm(la+lb),
//!   K = nherm(lc+ld) x (ket primitive pairs), N = n_c n_d;
//! - SHARK I-step: I[ab][cd] = sum_t E_ab[ab][t] T[t][cd]: M = n_a n_b, K = nherm(la+lb) x (bra
//!   primitive pairs), N = n_c n_d;
//! - our three-center G step (eri3c, Variant::G) with 8 auxiliary primitives stacked:
//!   M = 8 ncart(l_k), K = nherm(la+lb), N = n_a n_b.
//!
//! Method: one block per problem; A (M x K) and B (K x N) are staged once in shared memory
//! (zero-padded to M, N multiples of 16 and K of 4), then each warp takes 16 x 16 output tiles
//! and computes C = REPS * (A B) by accumulating the product REPS times (as a primitive sum
//! accumulates), so the on-chip product dominates. Same staging and tiling for both variants:
//! - FMA: each lane owns 8 outputs (rows g, g+8; columns 2t, 2t+1, 8+2t, 9+2t), A by scalar
//!   and B by double2 shared-memory loads;
//! - DMMA: four m8n8k4 products per k-step (two row tiles x two column tiles).
//!
//! Reported: useful TFLOP/s (unpadded 2 M N K REPS) and the fraction of each pipe's measured
//! peak (FMA chains; DMMA chains in registers).

mod common;
use common::{device_ms, median};
use linus_gpu::cpu::{ncart, nherm};
use linus_gpu::{Arg, Gpu, GpuError, Launch};
use std::fmt::Write as _;

type R<T> = Result<T, GpuError>;

const REPS: usize = 32;

fn nfunc(l: usize) -> usize {
    if l >= 2 {
        2 * l + 1
    } else {
        ncart(l)
    }
}

#[derive(Clone, Debug)]
struct Shape {
    name: String,
    m: usize,
    k: usize,
    n: usize,
}

fn shapes() -> Vec<Shape> {
    let mut v = Vec::new();
    // 4-center classes (la lb | lc ld), one and four primitive pairs contracted
    for &(la, lb, lc, ld, p) in &[
        (1, 1, 1, 1, 1),
        (2, 1, 2, 1, 1),
        (2, 2, 2, 2, 1),
        (2, 2, 2, 2, 4),
        (3, 2, 3, 2, 1),
        (3, 3, 3, 3, 1),
    ] {
        let tag = |l: usize| b"spdfg"[l] as char;
        let cls = format!(
            "({}{}|{}{}){}",
            tag(la),
            tag(lb),
            tag(lc),
            tag(ld),
            if p > 1 {
                format!(" x{p} prim")
            } else {
                String::new()
            }
        );
        let (nhab, nhcd) = (nherm(la + lb), nherm(lc + ld));
        let (nab, ncd) = (nfunc(la) * nfunc(lb), nfunc(lc) * nfunc(ld));
        v.push(Shape {
            name: format!("T {cls}"),
            m: nhab,
            k: nhcd * p,
            n: ncd,
        });
        v.push(Shape {
            name: format!("I {cls}"),
            m: nab,
            k: nhab * p,
            n: ncd,
        });
    }
    // three-center G step, 8 auxiliary primitives stacked
    for &(la, lb, lk) in &[(2, 2, 2), (3, 2, 3), (3, 3, 3)] {
        let tag = |l: usize| b"spdfg"[l] as char;
        v.push(Shape {
            name: format!("G ({}{}|{}) x8 aux", tag(la), tag(lb), tag(lk)),
            m: 8 * ncart(lk),
            k: nherm(la + lb),
            n: nfunc(la) * nfunc(lb),
        });
    }
    v
}

fn pad(x: usize, q: usize) -> usize {
    x.div_ceil(q) * q
}

/// The two kernels for one shape (C = REPS * A B, one block per problem).
fn source(s: &Shape, warps: usize) -> String {
    let (mp, kp, np) = (pad(s.m, 16), pad(s.k, 4), pad(s.n, 16));
    format!(
        r#"
#define M_ {m}
#define K_ {k}
#define N_ {n}
#define MP {mp}
#define KP {kp}
#define NP {np}
#define REPS {REPS}
#define WARPS {warps}

__device__ __forceinline__ void stage(const double *__restrict__ A, const double *__restrict__ B, double *As, double *Bs) {{
    const double *Ag = A + (size_t)blockIdx.x * M_ * K_;
    const double *Bg = B + (size_t)blockIdx.x * K_ * N_;
    for (int i = threadIdx.x; i < MP * KP; i += blockDim.x) {{ int r = i / KP, c = i % KP; As[i] = (r < M_ && c < K_) ? Ag[r * K_ + c] : 0.0; }}
    for (int i = threadIdx.x; i < KP * NP; i += blockDim.x) {{ int r = i / NP, c = i % NP; Bs[i] = (r < K_ && c < N_) ? Bg[r * N_ + c] : 0.0; }}
    __syncthreads();
}}

__device__ __forceinline__ void put(double *Cg, int r, int c, double v) {{
    if (r < M_ && c < N_) Cg[r * N_ + c] = v;
}}

// plain FP64 FMA: 8 outputs per lane of a 16 x 16 tile
extern "C" __global__ void __launch_bounds__(WARPS * 32) gemm_fma(const double *__restrict__ A, const double *__restrict__ B, double *__restrict__ C) {{
    extern __shared__ double sm[];
    double *As = sm, *Bs = sm + MP * KP;
    stage(A, B, As, Bs);
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32, g = lane >> 2, t = lane & 3;
    double *Cg = C + (size_t)blockIdx.x * M_ * N_;
    for (int tile = warp; tile < (MP / 16) * (NP / 16); tile += WARPS) {{
        const int m0 = (tile / (NP / 16)) * 16, n0 = (tile % (NP / 16)) * 16;
        double c[8] = {{0, 0, 0, 0, 0, 0, 0, 0}};
        for (int r = 0; r < REPS; ++r) {{
            #pragma unroll 4
            for (int k = 0; k < KP; ++k) {{
                const double a0 = As[(m0 + g) * KP + k], a1 = As[(m0 + 8 + g) * KP + k];
                const double2 b0 = *(const double2 *)&Bs[k * NP + n0 + 2 * t];
                const double2 b1 = *(const double2 *)&Bs[k * NP + n0 + 8 + 2 * t];
                c[0] = fma(a0, b0.x, c[0]); c[1] = fma(a0, b0.y, c[1]); c[2] = fma(a0, b1.x, c[2]); c[3] = fma(a0, b1.y, c[3]);
                c[4] = fma(a1, b0.x, c[4]); c[5] = fma(a1, b0.y, c[5]); c[6] = fma(a1, b1.x, c[6]); c[7] = fma(a1, b1.y, c[7]);
            }}
        }}
        put(Cg, m0 + g, n0 + 2 * t, c[0]); put(Cg, m0 + g, n0 + 2 * t + 1, c[1]); put(Cg, m0 + g, n0 + 8 + 2 * t, c[2]); put(Cg, m0 + g, n0 + 9 + 2 * t, c[3]);
        put(Cg, m0 + 8 + g, n0 + 2 * t, c[4]); put(Cg, m0 + 8 + g, n0 + 2 * t + 1, c[5]); put(Cg, m0 + 8 + g, n0 + 8 + 2 * t, c[6]); put(Cg, m0 + 8 + g, n0 + 9 + 2 * t, c[7]);
    }}
}}

// DMMA m8n8k4 (sm_80+): A fragment row g col t, B fragment row t col g, C fragment row g cols 2t, 2t+1
__device__ __forceinline__ void dmma(double &c0, double &c1, double a, double b) {{
    asm volatile("mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 {{%0, %1}}, {{%2}}, {{%3}}, {{%0, %1}};" : "+d"(c0), "+d"(c1) : "d"(a), "d"(b));
}}

extern "C" __global__ void __launch_bounds__(WARPS * 32) gemm_dmma(const double *__restrict__ A, const double *__restrict__ B, double *__restrict__ C) {{
    extern __shared__ double sm[];
    double *As = sm, *Bs = sm + MP * KP;
    stage(A, B, As, Bs);
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32, g = lane >> 2, t = lane & 3;
    double *Cg = C + (size_t)blockIdx.x * M_ * N_;
    for (int tile = warp; tile < (MP / 16) * (NP / 16); tile += WARPS) {{
        const int m0 = (tile / (NP / 16)) * 16, n0 = (tile % (NP / 16)) * 16;
        double c[8] = {{0, 0, 0, 0, 0, 0, 0, 0}};
        for (int r = 0; r < REPS; ++r) {{
            #pragma unroll 4
            for (int k = 0; k < KP; k += 4) {{
                const double a0 = As[(m0 + g) * KP + k + t], a1 = As[(m0 + 8 + g) * KP + k + t];
                const double b0 = Bs[(k + t) * NP + n0 + g], b1 = Bs[(k + t) * NP + n0 + 8 + g];
                dmma(c[0], c[1], a0, b0); dmma(c[2], c[3], a0, b1);
                dmma(c[4], c[5], a1, b0); dmma(c[6], c[7], a1, b1);
            }}
        }}
        put(Cg, m0 + g, n0 + 2 * t, c[0]); put(Cg, m0 + g, n0 + 2 * t + 1, c[1]); put(Cg, m0 + g, n0 + 8 + 2 * t, c[2]); put(Cg, m0 + g, n0 + 9 + 2 * t, c[3]);
        put(Cg, m0 + 8 + g, n0 + 2 * t, c[4]); put(Cg, m0 + 8 + g, n0 + 2 * t + 1, c[5]); put(Cg, m0 + 8 + g, n0 + 8 + 2 * t, c[6]); put(Cg, m0 + 8 + g, n0 + 9 + 2 * t, c[7]);
    }}
}}
"#,
        m = s.m,
        k = s.k,
        n = s.n
    )
}

/// Peak kernels: eight independent FMA chains per thread, and eight DMMA chains per warp.
const PEAKS: &str = r#"
extern "C" __global__ void fma_peak(double *out, int iters) {
    const double a = 0.9999999, b = 1e-7;
    double x0 = threadIdx.x * 1e-9, x1 = x0 + 1e-9, x2 = x0 + 2e-9, x3 = x0 + 3e-9, x4 = x0 + 4e-9, x5 = x0 + 5e-9, x6 = x0 + 6e-9, x7 = x0 + 7e-9;
    for (int i = 0; i < iters; ++i) {
        x0 = fma(x0, a, b); x1 = fma(x1, a, b); x2 = fma(x2, a, b); x3 = fma(x3, a, b);
        x4 = fma(x4, a, b); x5 = fma(x5, a, b); x6 = fma(x6, a, b); x7 = fma(x7, a, b);
    }
    out[blockIdx.x * blockDim.x + threadIdx.x] = x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7;
}
extern "C" __global__ void dmma_peak(double *out, int iters) {
    double c[16];
    for (int j = 0; j < 16; ++j) c[j] = 0;
    const double a = 1e-3 * threadIdx.x, b = 0.5;
    for (int i = 0; i < iters; ++i)
        #pragma unroll
        for (int j = 0; j < 8; ++j)
            asm volatile("mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 {%0, %1}, {%2}, {%3}, {%0, %1};" : "+d"(c[2 * j]), "+d"(c[2 * j + 1]) : "d"(a), "d"(b));
    double s = 0; for (int j = 0; j < 16; ++j) s += c[j];
    out[blockIdx.x * blockDim.x + threadIdx.x] = s;
}
"#;

fn peaks(gpu: &Gpu) -> R<(f64, f64)> {
    let blocks = gpu.info().sm_count * 8;
    let mut out = gpu.alloc_zeros::<f64>(blocks as usize * 256)?;
    let iters = 2048i32;
    let mut res = [0.0; 2];
    for (i, name) in ["fma_peak", "dmma_peak"].iter().enumerate() {
        let k = gpu.kernel("dmma_peaks", PEAKS, name)?;
        let cfg = Launch {
            grid: (blocks, 1, 1),
            block: (256, 1, 1),
            shared_bytes: 0,
        };
        // SAFETY: matches the peak kernels; one output per thread.
        let mut run = || unsafe { gpu.launch(&k, cfg, &[Arg::buf_mut(&mut out), Arg::I32(iters)]) };
        run()?;
        let ms = median(
            (0..5)
                .map(|_| device_ms(gpu, &mut run))
                .collect::<R<Vec<_>>>()?,
        );
        // fma: 8 chains x 2 flops per thread; dmma: 8 x (8x8x4 x 2 flops) per warp
        let flops = if i == 0 {
            16.0 * 256.0
        } else {
            8.0 * 512.0 * 8.0
        } * blocks as f64
            * iters as f64;
        res[i] = flops / (ms * 1e-3) / 1e12;
    }
    Ok((res[0], res[1]))
}

fn main() -> R<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let force = args.iter().any(|a| a == "--force");
    let json_out = args
        .iter()
        .position(|a| a == "--json")
        .and_then(|i| args.get(i + 1))
        .cloned();
    let gpu = Gpu::new(0)?;
    let info = gpu.info().clone();
    let targeted = matches!(info.cc, (8, 0) | (9, 0));
    println!(
        "{} ({}), opt-in shared memory {} KiB",
        info.name,
        info.arch(),
        info.max_smem_per_block_optin / 1024
    );
    if !targeted && !force {
        println!("skipped: the DMMA microbenchmark targets sm_80 / sm_90 (A100, H100); this is {} (--force runs it as a correctness check)", info.arch());
        return Ok(());
    }
    if info.cc.0 < 8 {
        println!("skipped: DMMA needs sm_80 or newer");
        return Ok(());
    }
    if !targeted {
        println!(
            "note: {} is not a DMMA target; timings below are not representative",
            info.arch()
        );
    }
    let (fma_peak, dmma_peak) = peaks(&gpu)?;
    println!("measured peaks: FP64 FMA {fma_peak:.2} TFLOP/s, DMMA {dmma_peak:.2} TFLOP/s");
    println!(
        "{:<26} {:>4} {:>4} {:>4} {:>8} {:>9} {:>9} {:>7} {:>7} {:>8} {:>9}",
        "shape",
        "M",
        "K",
        "N",
        "padding",
        "FMA TF/s",
        "DMMA TF/s",
        "FMA %pk",
        "DMMA %pk",
        "DMMA/FMA",
        "max err"
    );
    let mut json = format!("{{\"device\":\"{}\",\"arch\":\"{}\",\"fma_peak_tflops\":{fma_peak:.3},\"dmma_peak_tflops\":{dmma_peak:.3},\"shapes\":[", info.name, info.arch());
    let nprob = info.sm_count as usize * 16;
    for s in shapes() {
        let (mp, kp, np) = (pad(s.m, 16), pad(s.k, 4), pad(s.n, 16));
        let smem = ((mp * kp + kp * np) * 8) as u32;
        if smem > info.max_smem_per_block_optin {
            println!(
                "{:<26} {:>4} {:>4} {:>4}  skipped: {} KiB shared memory > the device's {} KiB",
                s.name,
                s.m,
                s.k,
                s.n,
                smem / 1024,
                info.max_smem_per_block_optin / 1024
            );
            continue;
        }
        let tiles = (mp / 16) * (np / 16);
        let warps = tiles.clamp(1, 4);
        let src = source(&s, warps);
        // problem data: values of mixed sign and magnitude
        let a: Vec<f64> = (0..nprob * s.m * s.k)
            .map(|i| ((i * 7919 % 1009) as f64 - 504.0) * 1e-3)
            .collect();
        let b: Vec<f64> = (0..nprob * s.k * s.n)
            .map(|i| ((i * 104729 % 997) as f64 - 498.0) * 1e-3)
            .collect();
        let (da, db) = (gpu.upload(&a)?, gpu.upload(&b)?);
        let mut dc = gpu.alloc_zeros::<f64>(nprob * s.m * s.n)?;
        // host reference for the first and last problems
        let reference = |p: usize| -> Vec<f64> {
            let mut c = vec![0.0; s.m * s.n];
            for i in 0..s.m {
                for j in 0..s.n {
                    let v: f64 = (0..s.k)
                        .map(|q| a[p * s.m * s.k + i * s.k + q] * b[p * s.k * s.n + q * s.n + j])
                        .sum();
                    c[i * s.n + j] = REPS as f64 * v;
                }
            }
            c
        };
        let flops = 2.0 * (s.m * s.n * s.k * REPS * nprob) as f64;
        let mut rate = [0.0; 2];
        let mut err: f64 = 0.0;
        for (vi, entry) in ["gemm_fma", "gemm_dmma"].iter().enumerate() {
            let k = gpu.kernel("dmma_bench", &src, entry)?;
            if smem > 48 * 1024 {
                k.set_max_dynamic_shared(smem)?;
            }
            let cfg = Launch {
                grid: (nprob as u32, 1, 1),
                block: (32 * warps as u32, 1, 1),
                shared_bytes: smem,
            };
            // SAFETY: matches gemm_fma / gemm_dmma; reads nprob A and B problems, writes nprob C.
            let mut run = || unsafe {
                gpu.launch(
                    &k,
                    cfg,
                    &[Arg::buf(&da), Arg::buf(&db), Arg::buf_mut(&mut dc)],
                )
            };
            run()?;
            let ms = median(
                (0..5)
                    .map(|_| device_ms(&gpu, &mut run))
                    .collect::<R<Vec<_>>>()?,
            );
            rate[vi] = flops / (ms * 1e-3) / 1e12;
            let c = gpu.download(&dc)?;
            for p in [0, nprob - 1] {
                let want = reference(p);
                let scale = want.iter().fold(0.0f64, |m, x| m.max(x.abs())).max(1e-300);
                for (x, y) in c[p * s.m * s.n..(p + 1) * s.m * s.n].iter().zip(&want) {
                    err = err.max((x - y).abs() / scale);
                }
            }
        }
        let padding = (mp * kp * np) as f64 / (s.m * s.k * s.n) as f64;
        println!(
            "{:<26} {:>4} {:>4} {:>4} {:>7.2}x {:>9.2} {:>9.2} {:>6.0}% {:>7.0}% {:>8.2} {:>9.1e}",
            s.name,
            s.m,
            s.k,
            s.n,
            padding,
            rate[0],
            rate[1],
            100.0 * rate[0] / fma_peak,
            100.0 * rate[1] / dmma_peak,
            rate[1] / rate[0],
            err
        );
        let _ = write!(json, "{{\"shape\":\"{}\",\"m\":{},\"k\":{},\"n\":{},\"padding\":{padding:.3},\"fma_tflops\":{:.3},\"dmma_tflops\":{:.3},\"max_rel_err\":{err:.2e}}},", s.name, s.m, s.k, s.n, rate[0], rate[1]);
        if err > 1e-12 {
            return Err(GpuError::NotCovered(format!(
                "{}: results differ from the host by {err:.2e}",
                s.name
            )));
        }
    }
    if json.ends_with(',') {
        json.pop();
    }
    json.push_str("]}\n");
    if let Some(p) = json_out {
        std::fs::write(&p, json).map_err(|e| GpuError::Input(format!("{p}: {e}")))?;
    }
    Ok(())
}
