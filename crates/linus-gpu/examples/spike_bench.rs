//! Kernel spike benchmark: the card's roofline numbers (FP64 FMA peak, DGEMM, memory
//! bandwidth), then the collocation and three-center kernels on benchmark molecules, against
//! linus's CPU kernels on the threads of the current affinity mask.
//!
//!   cargo run --release -p linus-gpu --features gpu --example spike_bench -- [molecules...] [--json out.json]
//!
//! Defaults: u1an39 ts61, def2-TZVP, PySCF level-3 grid, (dd|d) with def2-universal-jkfit.

mod common;
use common::device_ms;
use linus_gpu::blas::{Blas, GemmDims, MatMut, MatRef, Op};
use linus_gpu::cpu::{Collocator, GridSpec};
use linus_gpu::harness::{self, compare};
use linus_gpu::kernels::colloc::{self, GpuCollocator};
use linus_gpu::kernels::eri3c::{self, Class, ClassWork, GpuEri3c, Variant};
use linus_gpu::{Arg, Gpu, GpuError, Launch};
use std::fmt::Write as _;
use std::time::Instant;

type R<T> = Result<T, GpuError>;

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

/// FP64 FMA throughput (TFLOP/s) of eight independent FMA chains per thread.
fn fp64_peak(gpu: &Gpu) -> R<f64> {
    let src = r#"extern "C" __global__ void fma_peak(double *out, int iters, double a, double b) {
    double x0 = threadIdx.x * 1e-9, x1 = x0 + 1e-9, x2 = x0 + 2e-9, x3 = x0 + 3e-9, x4 = x0 + 4e-9, x5 = x0 + 5e-9, x6 = x0 + 6e-9, x7 = x0 + 7e-9;
    for (int i = 0; i < iters; ++i) {
        x0 = fma(x0, a, b); x1 = fma(x1, a, b); x2 = fma(x2, a, b); x3 = fma(x3, a, b);
        x4 = fma(x4, a, b); x5 = fma(x5, a, b); x6 = fma(x6, a, b); x7 = fma(x7, a, b);
    }
    out[blockIdx.x * blockDim.x + threadIdx.x] = x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7;
}"#;
    let k = gpu.kernel("fma_peak", src, "fma_peak")?;
    let blocks = gpu.info().sm_count * 8;
    let mut out = gpu.alloc_zeros::<f64>(blocks as usize * 256)?;
    let iters: i32 = 4096;
    let (a, b) = (0.999_999_9f64, 1e-7f64);
    let cfg = Launch {
        grid: (blocks, 1, 1),
        block: (256, 1, 1),
        shared_bytes: 0,
    };
    // SAFETY: matches fma_peak; writes one value per thread of the grid into out.
    let mut run = || unsafe {
        gpu.launch(
            &k,
            cfg,
            &[
                Arg::buf_mut(&mut out),
                Arg::I32(iters),
                Arg::F64(a),
                Arg::F64(b),
            ],
        )
    };
    run()?;
    let ms = median(
        (0..5)
            .map(|_| device_ms(gpu, &mut run))
            .collect::<R<Vec<_>>>()?,
    );
    Ok(2.0 * 8.0 * iters as f64 * blocks as f64 * 256.0 / (ms * 1e-3) / 1e12)
}

/// Device-to-device copy bandwidth (GB/s, read + write) of 2 GiB.
fn bandwidth(gpu: &Gpu) -> R<f64> {
    let n = 1usize << 28;
    let src = gpu.alloc_zeros::<f64>(n)?;
    let mut dst = gpu.alloc_zeros::<f64>(n)?;
    let mut run = || gpu.copy(&src, &mut dst);
    run()?;
    let ms = median(
        (0..5)
            .map(|_| device_ms(gpu, &mut run))
            .collect::<R<Vec<_>>>()?,
    );
    Ok(2.0 * 8.0 * n as f64 / (ms * 1e-3) / 1e9)
}

/// Native DGEMM rate (TFLOP/s) at n x n x n.
fn dgemm_rate(gpu: &Gpu, n: usize) -> R<f64> {
    let blas = Blas::new(gpu)?;
    let a = gpu.upload(&vec![1e-3; n * n])?;
    let b = gpu.upload(&vec![2e-3; n * n])?;
    let mut c = gpu.alloc_zeros::<f64>(n * n)?;
    let mut run = || {
        blas.gemm(
            GemmDims::new(Op::N, Op::N, n, n, n),
            (1.0, 0.0),
            MatRef::new(&a, 0, n),
            MatRef::new(&b, 0, n),
            MatMut::new(&mut c, 0, n),
        )
    };
    run()?;
    let ms = median(
        (0..3)
            .map(|_| device_ms(gpu, &mut run))
            .collect::<R<Vec<_>>>()?,
    );
    Ok(2.0 * (n as f64).powi(3) / (ms * 1e-3) / 1e12)
}

fn ptxas_info(gpu: &Gpu, name: &str, src: &str) -> String {
    match gpu
        .kernel_cache()
        .compile_log(name, src, &["--ptxas-options=-v"])
    {
        Ok(log) => log
            .lines()
            .filter(|l| {
                l.contains("registers") || l.contains("spill") || l.contains("Compiling entry")
            })
            .map(|l| l.trim().to_string())
            .collect::<Vec<_>>()
            .join(" | "),
        Err(e) => format!("(log unavailable: {e})"),
    }
}

fn bench_colloc(gpu: &Gpu, molname: &str, basis_name: &str, json: &mut String) -> R<()> {
    let mol = harness::structure(molname)?;
    let basis = harness::basis(&mol, basis_name)?;
    let grid = harness::grid(&mol, &GridSpec::pyscf(3))?;
    let (npts, nao, order) = (grid.len(), basis.nbf, 1usize);
    let nc = linus_gpu::cpu::ncomp(order);
    // work model: per (shell, point) the polynomial flops of its class, per (primitive, point)
    // one exp and 2 + 2 order flops
    let mut flops = 0.0;
    let mut exps = 0.0;
    for sh in &basis.shells {
        flops += colloc::generate(sh.l, sh.pure, order).1.flops as f64
            + (sh.nprim() * (2 + 2 * order)) as f64;
        exps += sh.nprim() as f64;
    }
    flops *= npts as f64;
    exps *= npts as f64;
    let bytes = (nc * nao * npts * 8) as f64;

    let coll = GpuCollocator::new(gpu, &basis, order)?;
    let chunk = 65536usize.min(npts);
    let mut out = gpu.alloc_zeros::<f64>(nc * nao * chunk)?;
    // time the kernels only: upload the chunks once, then launch
    let mut chunks = Vec::new();
    let mut s = 0;
    while s < npts {
        let n = chunk.min(npts - s);
        chunks.push((
            n,
            gpu.upload(&grid.x[s..s + n])?,
            gpu.upload(&grid.y[s..s + n])?,
            gpu.upload(&grid.z[s..s + n])?,
        ));
        s += n;
    }
    chunks
        .iter()
        .try_for_each(|(n, cx, cy, cz)| coll.eval(gpu, cx, cy, cz, *n, chunk, &mut out))?;
    let gpu_ms = median(
        (0..5)
            .map(|_| {
                device_ms(gpu, || {
                    chunks.iter().try_for_each(|(n, cx, cy, cz)| {
                        coll.eval(gpu, cx, cy, cz, *n, chunk, &mut out)
                    })
                })
            })
            .collect::<R<Vec<_>>>()?,
    );

    // CPU: linus-ints' Collocator over the grid batches, unscreened (threshold 0: every shell
    // at every point, negligible primitives still skipped) and at linus's DFT threshold 1e-11
    let mut cpu = Vec::new();
    for thresh in [0.0, 1e-11] {
        use rayon::prelude::*;
        let cc = Collocator::new(&basis, order, thresh);
        let shells: Vec<usize> = (0..basis.shells.len()).collect();
        let t = (0..3)
            .map(|_| {
                let t0 = Instant::now();
                let kept: usize = grid
                    .batches
                    .par_iter()
                    .map_init(
                        || {
                            (
                                vec![0.0; nc * nao * 264],
                                Vec::new(),
                                Vec::new(),
                                vec![0.0; 264],
                                vec![0.0; 264],
                                vec![0.0; 264],
                            )
                        },
                        |(o, idx, own, xs, ys, zs), b| {
                            let np = b.end - b.start;
                            let ld = np.div_ceil(8) * 8;
                            if o.len() < nc * nao * ld {
                                o.resize(nc * nao * ld, 0.0);
                            }
                            for (v, src) in [
                                (&mut *xs, &grid.x),
                                (&mut *ys, &grid.y),
                                (&mut *zs, &grid.z),
                            ] {
                                v.clear();
                                v.extend_from_slice(&src[b.start..b.end]);
                                v.resize(ld, src[b.end - 1]);
                            }
                            let mut lo = [f64::MAX; 3];
                            let mut hi = [f64::MIN; 3];
                            for i in 0..np {
                                for (k, c) in [xs[i], ys[i], zs[i]].into_iter().enumerate() {
                                    lo[k] = lo[k].min(c);
                                    hi[k] = hi[k].max(c);
                                }
                            }
                            cc.eval(&shells, xs, ys, zs, np, ld, &lo, &hi, o, idx, own) * np
                        },
                    )
                    .sum();
                (t0.elapsed().as_secs_f64() * 1e3, kept)
            })
            .collect::<Vec<_>>();
        let kept = t[0].1;
        cpu.push((thresh, median(t.iter().map(|x| x.0).collect()), kept));
    }
    let src = colloc::module_source(&basis, order);
    let regs = ptxas_info(gpu, "colloc", &src);
    println!("\ncollocation {molname}/{basis_name}, PySCF level 3: {npts} points, {nao} functions, value + gradient");
    println!("  GPU dense: {gpu_ms:.1} ms  ({:.0} GFLOP/s nominal excl. exp, {:.1} Gexp/s, {:.0} GB/s written)", flops / gpu_ms / 1e6, exps / gpu_ms / 1e6, bytes / gpu_ms / 1e6);
    for (th, ms, kept) in &cpu {
        println!(
            "  CPU {} threads, threshold {th:e}: {ms:.1} ms ({:.1}% of function-point values kept)",
            rayon::current_num_threads(),
            100.0 * *kept as f64 / (nao * npts) as f64
        );
    }
    println!("  ptxas: {regs}");
    let _ = write!(
        json,
        "{{\"kernel\":\"colloc\",\"molecule\":\"{molname}\",\"basis\":\"{basis_name}\",\"lmax\":{},\"points\":{npts},\"nao\":{nao},\"gpu_ms\":{gpu_ms:.3},\"nominal_gflop\":{:.3},\"gexp\":{:.3},\"bytes\":{bytes},\"cpu_threads\":{},\"cpu_ms_unscreened\":{:.3},\"cpu_ms_1e-11\":{:.3},\"cpu_kept_fraction_1e-11\":{:.4}}},",
        basis.max_l(),
        flops / 1e9,
        exps / 1e9,
        rayon::current_num_threads(),
        cpu[0].1,
        cpu[1].1,
        cpu[1].2 as f64 / (nao * npts) as f64
    );
    Ok(())
}

fn bench_eri3c(gpu: &Gpu, molname: &str, class: Class, json: &mut String) -> R<()> {
    let mol = harness::structure(molname)?;
    let basis = harness::basis(&mol, "def2-tzvp")?;
    let aux = harness::basis(&mol, "def2-universal-jkfit")?;
    let work = ClassWork::new(&basis, &aux, class);
    let triples = work.prim_triples() as f64;
    let (rows, ld) = work.shape();
    let t0 = Instant::now();
    let cpu = work.cpu();
    let mut cpu_t = vec![t0.elapsed().as_secs_f64() * 1e3];
    for _ in 0..2 {
        let t0 = Instant::now();
        let _ = work.cpu();
        cpu_t.push(t0.elapsed().as_secs_f64() * 1e3);
    }
    let cpu_ms = median(cpu_t);
    println!("\nthree-center {} {molname}/def2-TZVP + def2-universal-jkfit: {} pairs x {} aux shells, {:.3e} primitive triples, output {rows} x {ld}", class.tag(), work.pairs.len(), work.aux.shells.len(), triples);
    println!(
        "  CPU {} threads (pair_block): {cpu_ms:.1} ms",
        rayon::current_num_threads()
    );
    let _ = write!(json, "{{\"kernel\":\"eri3c\",\"class\":\"{}\",\"molecule\":\"{molname}\",\"pairs\":{},\"auxsh\":{},\"triples\":{triples},\"cpu_threads\":{},\"cpu_ms\":{cpu_ms:.3}", class.tag(), work.pairs.len(), work.aux.shells.len(), rayon::current_num_threads());
    for v in [Variant::H, Variant::G] {
        let k = GpuEri3c::new(gpu, &work, v)?;
        let (src, st) = eri3c::generate(class, v);
        let mut out = gpu.alloc_zeros::<f64>(k.out_len())?;
        k.run(gpu, &mut out)?;
        let ms = median(
            (0..5)
                .map(|_| device_ms(gpu, || k.run(gpu, &mut out)))
                .collect::<R<Vec<_>>>()?,
        );
        let err = compare(&gpu.download(&out)?, &cpu);
        let regs = k.kernel().registers().unwrap_or(-1);
        let local = k.kernel().local_bytes().unwrap_or(-1);
        let gf = st.flops as f64 * triples / ms / 1e6;
        println!("  GPU {v:?}: {ms:.2} ms, {gf:.0} GFLOP/s nominal ({} flops/triple), {regs} registers, {local} B local (spills), {} source lines; vs CPU {err}", st.flops, st.lines);
        println!(
            "    ptxas: {}",
            ptxas_info(gpu, &eri3c::kernel_name(class, v), &src)
        );
        let _ = write!(json, ",\"gpu_{v:?}_ms\":{ms:.3},\"gpu_{v:?}_gflops\":{gf:.1},\"gpu_{v:?}_regs\":{regs},\"gpu_{v:?}_local_bytes\":{local},\"gpu_{v:?}_max_rel_err\":{:.3e}", err.max_rel);
    }
    json.push_str("},");
    Ok(())
}

fn main() -> R<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut mols = Vec::new();
    let mut json_out = None;
    let mut it = args.iter();
    while let Some(a) = it.next() {
        if a == "--json" {
            json_out = it.next().cloned();
        } else {
            mols.push(a.clone());
        }
    }
    if mols.is_empty() {
        mols = vec!["u1an39".into(), "ts61".into()];
    }
    let gpu = Gpu::new(0)?;
    let i = gpu.info();
    let (peak, bw, dg) = (fp64_peak(&gpu)?, bandwidth(&gpu)?, dgemm_rate(&gpu, 8192)?);
    println!(
        "{} ({}, {} SMs, {:.0} GiB, driver {}), NVRTC {:?} -> {:?}",
        i.name,
        i.arch(),
        i.sm_count,
        i.total_mem as f64 / (1u64 << 30) as f64,
        i.driver_version,
        gpu.kernel_cache().nvrtc_version(),
        gpu.kernel_cache().target()
    );
    println!("roofline: FP64 FMA {peak:.2} TFLOP/s, DGEMM(8192) {dg:.2} TFLOP/s, copy bandwidth {bw:.0} GB/s; CPU threads {}", rayon::current_num_threads());
    let mut json = format!("{{\"device\":\"{}\",\"arch\":\"{}\",\"fp64_fma_tflops\":{peak:.3},\"dgemm_tflops\":{dg:.3},\"copy_gbs\":{bw:.1},\"runs\":[", i.name, i.arch());
    for m in &mols {
        bench_colloc(&gpu, m, "def2-tzvp", &mut json)?;
        bench_eri3c(
            &gpu,
            m,
            Class {
                la: 2,
                lb: 2,
                lk: 2,
            },
            &mut json,
        )?;
    }
    // g shells (l = 4) for the collocation
    bench_colloc(&gpu, &mols[0], "def2-qzvp", &mut json)?;
    if json.ends_with(',') {
        json.pop();
    }
    json.push_str("]}\n");
    let st = gpu.kernel_cache().stats();
    println!(
        "\nkernel cache ({:?}): {} compiled in {:.2} s, {} loaded from disk, {} from memory",
        gpu.kernel_cache().dir(),
        st.compiled,
        st.compile_seconds,
        st.disk_hits,
        st.memory_hits
    );
    if let Some(p) = json_out {
        std::fs::write(&p, json).map_err(|e| GpuError::Input(format!("{p}: {e}")))?;
    }
    Ok(())
}
