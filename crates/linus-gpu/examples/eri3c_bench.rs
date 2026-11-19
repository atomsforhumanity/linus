//! Benchmark of the generated three-center kernels (`linus_gpu::eri3c`): the card's FP64 peak,
//! the full (mn|P) build of a molecule (every pair against every auxiliary shell, in pair
//! batches that fit `--batch-gb`), per-class times and FP64 rates against the peak, the same
//! build on the CPU (linus-ints' `pair_block` on the threads of the affinity mask), the spike's
//! hand-written (dd|d) kernels on the same pairs, and the first-derivative build. With
//! `--omega w` the full builds (integrals, derivatives, Gamma gradient) are timed for
//! erf(w r)/r too, beside the Coulomb ones.
//!
//!   eri3c_bench [--molecule u1an39] [--basis def2-tzvp] [--aux def2-universal-jkfit]
//!               [--thresh 1e-16] [--batch-gb 4] [--omega 0.3] [--no-cpu] [--no-deriv]

mod common;
use common::device_ms;
use linus_gpu::cpu::{pair_block, Aux3c, Operator, Pair3c, Work3c};
use linus_gpu::eri3c::{all_pairs, Eri3c, Eri3cHost, Eri3cKind, PairRow};
use linus_gpu::harness;
use linus_gpu::kernels::eri3c::{Class, ClassWork, GpuEri3c, Variant};
use linus_gpu::{Arg, Gpu, GpuError, Launch};
use std::time::Instant;

type R<T> = Result<T, GpuError>;

fn arg(args: &[String], name: &str) -> Option<String> {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1).cloned())
}

/// FP64 FMA throughput (GFLOP/s) of eight independent FMA chains per thread.
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
    let cfg = Launch {
        grid: (blocks, 1, 1),
        block: (256, 1, 1),
        shared_bytes: 0,
    };
    let mut best = f64::INFINITY;
    for _ in 0..5 {
        // SAFETY: matches fma_peak; one value per thread of the grid.
        let ms = device_ms(gpu, || unsafe {
            gpu.launch(
                &k,
                cfg,
                &[
                    Arg::buf_mut(&mut out),
                    Arg::I32(iters),
                    Arg::F64(0.999_999_9),
                    Arg::F64(1e-7),
                ],
            )
        })?;
        best = best.min(ms);
    }
    Ok(blocks as f64 * 256.0 * iters as f64 * 8.0 * 2.0 / best * 1e-6)
}

/// Pair batches whose output fits `bytes`.
fn batches(host: &Eri3cHost, bytes: usize) -> Vec<std::ops::Range<usize>> {
    let mut out = Vec::new();
    let n = host.pairs.len();
    let mut p0 = 0;
    while p0 < n {
        let mut p1 = p0 + 1;
        while p1 < n && host.out_len(p0..p1 + 1) * 8 <= bytes {
            p1 += 1;
        }
        out.push(p0..p1);
        p0 = p1;
    }
    out
}

fn main() -> R<()> {
    let args: Vec<String> = std::env::args().collect();
    let molname = arg(&args, "--molecule").unwrap_or_else(|| "u1an39".into());
    let bname = arg(&args, "--basis").unwrap_or_else(|| "def2-tzvp".into());
    let aname = arg(&args, "--aux").unwrap_or_else(|| "def2-universal-jkfit".into());
    let thresh: f64 = arg(&args, "--thresh")
        .map(|s| s.parse().unwrap())
        .unwrap_or(1e-16);
    let batch_bytes = (arg(&args, "--batch-gb")
        .map(|s| s.parse::<f64>().unwrap())
        .unwrap_or(4.0)
        * 1e9) as usize;
    let erf = arg(&args, "--omega").map(|s| Operator::LongRange {
        omega: s.parse().unwrap(),
    });
    let gpu = Gpu::new(0)?;
    let info = gpu.info().clone();
    let peak = fp64_peak(&gpu)?;
    println!(
        "card {} ({} SMs), FP64 FMA peak {:.0} GFLOP/s",
        info.name, info.sm_count, peak
    );
    let mol = harness::structure(&molname)?;
    let basis = harness::basis(&mol, &bname)?;
    let aux = harness::basis(&mol, &aname)?;
    let pairs = all_pairs(&basis);
    let t = Instant::now();
    let host = Eri3cHost::new(&basis, &aux, &pairs, Eri3cKind::Integrals, thresh)?;
    let pack_s = t.elapsed().as_secs_f64();
    let t = Instant::now();
    let mut b = Eri3c::new(&gpu, host)?;
    let setup_s = t.elapsed().as_secs_f64();
    if let Some(op) = &erf {
        let t = Instant::now();
        b.prepare_operator(&gpu, op)?;
        println!(
            "erf kernels: compile (or load) {:.2} s",
            t.elapsed().as_secs_f64()
        );
    }
    let h = b.host();
    let np = pairs.len();
    println!(
        "{molname}/{bname}/{aname}: nbf {}, naux {}, {np} shell pairs, {} rows; pack {pack_s:.2} s, upload + compile {setup_s:.2} s; prim_thresh {thresh:e}",
        basis.nbf,
        aux.nbf,
        h.rows(0..np).len()
    );
    let bat = batches(h, batch_bytes);
    let maxlen = bat.iter().map(|r| h.out_len(r.clone())).max().unwrap();
    let mut out = gpu.alloc_zeros::<f64>(maxlen)?;
    // warm up
    for r in &bat {
        b.compute(&gpu, r.clone(), &Operator::Coulomb, &mut out, 0)?;
    }
    let mut total = f64::INFINITY;
    for _ in 0..3 {
        let mut s = 0.0;
        for r in &bat {
            s += device_ms(&gpu, || {
                b.compute(&gpu, r.clone(), &Operator::Coulomb, &mut out, 0)
            })?;
        }
        total = total.min(s);
    }
    let fl = h.flops(0..np);
    println!(
        "integrals, all classes: {total:.2} ms in {} batches, {:.0} GFLOP/s (model), {:.0} % of the FP64 peak",
        bat.len(),
        fl / total * 1e-6,
        100.0 * fl / total * 1e-6 / peak
    );
    if let Some(op) = &erf {
        for r in &bat {
            b.compute(&gpu, r.clone(), op, &mut out, 0)?;
        }
        let mut te = f64::INFINITY;
        for _ in 0..3 {
            let mut s = 0.0;
            for r in &bat {
                s += device_ms(&gpu, || b.compute(&gpu, r.clone(), op, &mut out, 0))?;
            }
            te = te.min(s);
        }
        println!(
            "integrals of erf({} r)/r, all classes: {te:.2} ms ({:.2}x the Coulomb build)",
            op.omega(),
            te / total
        );
    }
    // per class
    println!("class        schedule        ms  GFLOP/s  %peak  regs  local  share");
    let mut rows = Vec::new();
    for la in 0..=4usize {
        for lb in 0..=la {
            for lk in 0..=6usize {
                let Some((gk, k)) = b.kernel(la, lb, lk) else {
                    continue;
                };
                let mut ms = f64::INFINITY;
                for _ in 0..2 {
                    let mut s = 0.0;
                    for r in &bat {
                        s += device_ms(&gpu, || {
                            b.compute_class(&gpu, r.clone(), &mut out, (la, lb, lk))
                        })?;
                    }
                    ms = ms.min(s);
                }
                let f = h.flops_of(0..np, Some((la, lb, lk)));
                rows.push((
                    la,
                    lb,
                    lk,
                    gk.schedule,
                    ms,
                    f,
                    k.registers().unwrap_or(-1),
                    k.local_bytes().unwrap_or(-1),
                ));
            }
        }
    }
    let sum: f64 = rows.iter().map(|r| r.4).sum();
    rows.sort_by(|a, b| b.4.partial_cmp(&a.4).unwrap());
    for (la, lb, lk, s, ms, f, regs, local) in &rows {
        println!(
            "({la}{lb}|{lk})  {s:>14} {ms:8.3} {:8.0} {:6.1} {regs:5} {local:6} {:5.1}%",
            f / ms * 1e-6,
            100.0 * f / ms * 1e-6 / peak,
            100.0 * ms / sum
        );
    }
    println!("sum of class times {sum:.2} ms (launch overheads included)");

    // the spike's hand-written (dd|d) on the same pairs
    let dd = ClassWork::new(
        &basis,
        &aux,
        Class {
            la: 2,
            lb: 2,
            lk: 2,
        },
    );
    if !dd.pairs.is_empty() {
        let mut o = gpu.alloc_zeros::<f64>(dd.shape().0 * dd.shape().1)?;
        for v in [Variant::G, Variant::H] {
            let k = GpuEri3c::new(&gpu, &dd, v)?;
            k.run(&gpu, &mut o)?;
            let mut ms = f64::INFINITY;
            for _ in 0..3 {
                ms = ms.min(device_ms(&gpu, || k.run(&gpu, &mut o))?);
            }
            println!(
                "spike (dd|d) {v:?}: {ms:.3} ms, {} pairs x {} aux shells, regs {}",
                dd.pairs.len(),
                dd.aux.shells.len(),
                k.kernel().registers().unwrap_or(-1)
            );
        }
        // generated (dd|d) on exactly the spike's pairs and no screening
        let mut row = 0;
        let sel: Vec<PairRow> = dd
            .pairs
            .iter()
            .map(|p| {
                let r = PairRow {
                    sa: p.sa,
                    sb: p.sb,
                    row,
                };
                row += 25;
                r
            })
            .collect();
        let hd = Eri3cHost::new(&basis, &aux, &sel, Eri3cKind::Integrals, 0.0)?;
        let bd = Eri3c::new(&gpu, hd)?;
        let n = sel.len();
        let mut o2 = gpu.alloc_zeros::<f64>(bd.host().out_len(0..n))?;
        bd.compute_class(&gpu, 0..n, &mut o2, (2, 2, 2))?;
        let mut ms = f64::INFINITY;
        for _ in 0..3 {
            ms = ms.min(device_ms(&gpu, || {
                bd.compute_class(&gpu, 0..n, &mut o2, (2, 2, 2))
            })?);
        }
        let f = bd.host().flops_of(0..n, Some((2, 2, 2)));
        println!(
            "generated (dd|d) {}: {ms:.3} ms, {:.0} GFLOP/s (model), {:.0} % of peak",
            bd.kernel(2, 2, 2).unwrap().0.schedule,
            f / ms * 1e-6,
            100.0 * f / ms * 1e-6 / peak
        );
    }

    // CPU: pair_block over every pair (same primitive threshold), rayon on the affinity mask
    if !args.iter().any(|a| a == "--no-cpu") {
        use rayon::prelude::*;
        let aux3 = Aux3c::new(&aux);
        let chunks = aux3.chunks(0..aux.shells.len());
        let p3: Vec<Pair3c> = pairs
            .par_iter()
            .map(|p| {
                Pair3c::from_shell_pair(&linus_gpu::cpu::build_pair(
                    &linus_gpu::cpu::ShellView::from_shell(&basis.shells[p.sa]),
                    &linus_gpu::cpu::ShellView::from_shell(&basis.shells[p.sb]),
                    p.sa,
                    p.sb,
                ))
            })
            .collect();
        let maxnab = p3.iter().map(|p| p.nab()).max().unwrap();
        let t = Instant::now();
        p3.par_iter().for_each_init(
            || (Work3c::new(), vec![0.0; maxnab * aux.nbf]),
            |(w, blk), p| {
                let blk = &mut blk[..p.nab() * aux.nbf];
                blk.fill(0.0);
                pair_block(
                    p,
                    &chunks,
                    blk,
                    aux.nbf,
                    thresh,
                    &linus_ints::operator::Operator::Coulomb,
                    w,
                );
            },
        );
        let cpu_ms = t.elapsed().as_secs_f64() * 1e3;
        println!(
            "CPU pair_block, {} threads: {cpu_ms:.1} ms; GPU speed-up {:.1}x",
            rayon::current_num_threads(),
            cpu_ms / total
        );
    }

    // first derivatives
    if !args.iter().any(|a| a == "--no-deriv") {
        drop(out);
        drop(b);
        let hd = Eri3cHost::new(&basis, &aux, &pairs, Eri3cKind::Derivatives, thresh)?;
        let mut bd = Eri3c::new(&gpu, hd)?;
        if let Some(op) = &erf {
            bd.prepare_operator(&gpu, op)?;
        }
        let h = bd.host();
        let bat = batches(h, batch_bytes);
        let maxlen = bat.iter().map(|r| h.out_len(r.clone())).max().unwrap();
        let mut out = gpu.alloc_zeros::<f64>(maxlen)?;
        for r in &bat {
            bd.compute(&gpu, r.clone(), &Operator::Coulomb, &mut out, 0)?;
        }
        let mut s = 0.0;
        for r in &bat {
            s += device_ms(&gpu, || {
                bd.compute(&gpu, r.clone(), &Operator::Coulomb, &mut out, 0)
            })?;
        }
        let fl = h.flops(0..np);
        println!(
            "first derivatives (6 planes), all classes: {s:.2} ms in {} batches, {:.0} GFLOP/s (model), {:.0} % of peak",
            bat.len(),
            fl / s * 1e-6,
            100.0 * fl / s * 1e-6 / peak
        );
        if let Some(op) = &erf {
            for r in &bat {
                bd.compute(&gpu, r.clone(), op, &mut out, 0)?;
            }
            let mut se = 0.0;
            for r in &bat {
                se += device_ms(&gpu, || bd.compute(&gpu, r.clone(), op, &mut out, 0))?;
            }
            println!(
                "first derivatives of erf({} r)/r: {se:.2} ms ({:.2}x the Coulomb build)",
                op.omega(),
                se / s
            );
        }
        // Gamma-contracted gradient: one Gamma plane per batch, no derivative integral stored
        drop(out);
        let gbat = batches(h, 6 * batch_bytes);
        let glen = gbat.iter().map(|r| h.out_len(r.clone()) / 6).max().unwrap();
        let gam = gpu.upload(
            &(0..glen)
                .map(|i| ((i % 1013) as f64 - 506.0) * 1e-3)
                .collect::<Vec<f64>>(),
        )?;
        let mut acc = linus_gpu::detsum::DetAcc::new(&gpu, 3 * h.natom)?;
        for r in &gbat {
            bd.gradient(&gpu, r.clone(), &Operator::Coulomb, &gam, &mut acc)?;
        }
        let mut s = 0.0;
        for r in &gbat {
            s += device_ms(&gpu, || {
                bd.gradient(&gpu, r.clone(), &Operator::Coulomb, &gam, &mut acc)
            })?;
        }
        println!(
            "Gamma-contracted gradient, all classes: {s:.2} ms in {} batches, {:.0} GFLOP/s (model), {:.0} % of peak",
            gbat.len(),
            fl / s * 1e-6,
            100.0 * fl / s * 1e-6 / peak
        );
        if let Some(op) = &erf {
            for r in &gbat {
                bd.gradient(&gpu, r.clone(), op, &gam, &mut acc)?;
            }
            let mut se = 0.0;
            for r in &gbat {
                se += device_ms(&gpu, || bd.gradient(&gpu, r.clone(), op, &gam, &mut acc))?;
            }
            println!(
                "Gamma-contracted gradient of erf({} r)/r: {se:.2} ms ({:.2}x the Coulomb one)",
                op.omega(),
                se / s
            );
        }
    }
    Ok(())
}
