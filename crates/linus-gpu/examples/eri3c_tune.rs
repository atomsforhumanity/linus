//! Auto-tuner of the generated three-center kernels: times every candidate schedule of a class
//! on a benchmark workload, checks it against the class's current kernel, and appends the best
//! per class to a JSON-lines record that `tools/intgen/intgen.py cuda --record` merges into
//! tools/intgen/eri3c_schedules.json (the schedules the production sources are generated with).
//!
//!   python3 tools/intgen/intgen.py cuda --candidates DIR [--only e222 ...]
//!   eri3c_tune --candidates DIR --molecule u1an39 --basis def2-tzvp [--aux def2-universal-jkfit]
//!              [--kind e|d] [--only e222 ...] [--max-rows N] [--reps 3] [--compile-only]
//!              --out results.jsonl
//!
//! Each class runs on the workload's pairs of that class (up to `--max-rows` function-pair
//! rows) against every auxiliary shell of l_k. Compile with `--compile-only` first (NVRTC fills
//! the disk cache), then time under the GPU lock.

mod common;
use common::device_ms;
use linus_gpu::eri3c::{all_pairs, Eri3c, Eri3cHost, Eri3cKind, PairRow};
use linus_gpu::harness::{self, compare};
use linus_gpu::kernels::eri3c_class::{gen_kernel_for, GenKernel, Mapping};
use linus_gpu::{Gpu, GpuError};
use std::collections::BTreeMap;
use std::io::Write as _;

type R<T> = Result<T, GpuError>;

struct Cand {
    key: String,
    schedule: String,
    entry: String,
    file: String,
    mapping: Mapping,
    threads: u32,
    smem: u32,
    ktile: u32,
    rec: usize,
    nrows: usize,
    hash: u64,
    /// shared R recursion file, or empty
    prelude: String,
}

fn read_manifest(dir: &str) -> Vec<Cand> {
    let text = std::fs::read_to_string(format!("{dir}/manifest.tsv")).expect("manifest.tsv");
    text.lines()
        .skip(1)
        .map(|l| {
            let f: Vec<&str> = l.split('\t').collect();
            Cand {
                key: f[0].into(),
                schedule: f[1].into(),
                entry: f[2].into(),
                file: f[3].into(),
                mapping: if f[4] == "thread" {
                    Mapping::Thread
                } else {
                    Mapping::Block
                },
                threads: f[5].parse().unwrap(),
                smem: f[6].parse().unwrap(),
                ktile: f[7].parse().unwrap(),
                rec: f[8].parse().unwrap(),
                nrows: f[9].parse().unwrap(),
                hash: f[10].parse().unwrap(),
                prelude: f
                    .get(11)
                    .filter(|p| **p != "-")
                    .map(|p| p.to_string())
                    .unwrap_or_default(),
            }
        })
        .collect()
}

fn arg(args: &[String], name: &str) -> Option<String> {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1).cloned())
}

fn args_after(args: &[String], name: &str) -> Vec<String> {
    match args.iter().position(|a| a == name) {
        Some(i) => args[i + 1..]
            .iter()
            .take_while(|a| !a.starts_with("--"))
            .cloned()
            .collect(),
        None => vec![],
    }
}

fn main() -> R<()> {
    let args: Vec<String> = std::env::args().collect();
    let dir = arg(&args, "--candidates").expect("--candidates DIR");
    let molname = arg(&args, "--molecule").unwrap_or_else(|| "u1an39".into());
    let bname = arg(&args, "--basis").unwrap_or_else(|| "def2-tzvp".into());
    let aname = arg(&args, "--aux").unwrap_or_else(|| "def2-universal-jkfit".into());
    let kind = match arg(&args, "--kind").as_deref() {
        Some("d") => Eri3cKind::Derivatives,
        _ => Eri3cKind::Integrals,
    };
    let kc = if kind == Eri3cKind::Integrals {
        'e'
    } else {
        'd'
    };
    let only = args_after(&args, "--only");
    let max_rows: usize = arg(&args, "--max-rows")
        .map(|s| s.parse().unwrap())
        .unwrap_or(60_000);
    let reps: usize = arg(&args, "--reps")
        .map(|s| s.parse().unwrap())
        .unwrap_or(3);
    let compile_only = args.iter().any(|a| a == "--compile-only");
    let out_path = arg(&args, "--out");

    let gpu = Gpu::new(0)?;
    let mol = harness::structure(&molname)?;
    let basis = harness::basis(&mol, &bname)?;
    let aux = harness::basis(&mol, &aname)?;
    let pairs = all_pairs(&basis);
    let mut by_key: BTreeMap<String, Vec<Cand>> = BTreeMap::new();
    for c in read_manifest(&dir) {
        if c.key.starts_with(kc) && (only.is_empty() || only.contains(&c.key)) {
            by_key.entry(c.key.clone()).or_default().push(c);
        }
    }
    let mut out = out_path.map(|p| {
        std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(p)
            .unwrap()
    });
    let mut hosts: BTreeMap<(usize, usize), Option<Eri3c>> = BTreeMap::new();
    for (key, cands) in &by_key {
        let d: Vec<usize> = key[1..]
            .chars()
            .map(|c| c.to_digit(10).unwrap() as usize)
            .collect();
        let (la, lb, lk) = (d[0], d[1], d[2]);
        let b = hosts.entry((la, lb)).or_insert_with(|| {
            // the workload's pairs of class (la lb), consecutive rows, up to max_rows
            let mut sel = Vec::new();
            let mut row = 0;
            for p in &pairs {
                let (a, bb) = (basis.shells[p.sa].l, basis.shells[p.sb].l);
                if (a.max(bb), a.min(bb)) != (la, lb) {
                    continue;
                }
                let nab = basis.shells[p.sa].nfunc() * basis.shells[p.sb].nfunc();
                if row + nab > max_rows {
                    break;
                }
                sel.push(PairRow { row, ..*p });
                row += nab;
            }
            if sel.is_empty() {
                return None;
            }
            let host = Eri3cHost::new(&basis, &aux, &sel, kind, 1e-16).ok()?;
            Eri3c::new(&gpu, host).ok()
        });
        let Some(b) = b.as_mut() else {
            eprintln!("{key}: no pairs of this class in {molname}/{bname}");
            continue;
        };
        if b.kernel(la, lb, lk).is_none() {
            eprintln!("{key}: no auxiliary shells with l = {lk}");
            continue;
        }
        let np = b.host().pairs.len();
        let flops = b.host().flops_of(0..np, Some((la, lb, lk)));
        let prod = gen_kernel_for(gpu.info().cc, kind, la, lb, lk).unwrap();
        // reference: the current kernel of the class (not needed to fill the compile cache)
        let mut buf = gpu.alloc_zeros::<f64>(if compile_only {
            1
        } else {
            b.host().out_len(0..np)
        })?;
        let reference = if compile_only {
            Vec::new()
        } else {
            b.compute_class(&gpu, 0..np, &mut buf, (la, lb, lk))?;
            gpu.download(&buf)?
        };
        let mut times: Vec<(String, f64, i32, i32)> = Vec::new();
        for c in cands {
            assert_eq!(
                (c.rec, c.nrows, c.hash),
                (prod.rec, prod.nrows, prod.hash),
                "{}: candidate shape differs from the production kernel",
                c.entry
            );
            let mut src = std::fs::read_to_string(format!("{dir}/{}", c.file)).unwrap();
            if !c.prelude.is_empty() {
                let pre = std::fs::read_to_string(format!("{dir}/{}", c.prelude)).unwrap();
                src = format!("{pre}\n{src}");
            }
            let gk: &'static GenKernel = Box::leak(Box::new(GenKernel {
                entry: Box::leak(c.entry.clone().into_boxed_str()),
                schedule: Box::leak(c.schedule.clone().into_boxed_str()),
                // the candidate source above is complete
                prelude: "",
                mapping: c.mapping,
                threads: c.threads,
                smem: c.smem,
                ktile: c.ktile,
                ..*prod
            }));
            if let Err(e) = b.set_kernel(&gpu, gk, &src) {
                eprintln!("{key} {}: {e}", c.schedule);
                continue;
            }
            if compile_only {
                continue;
            }
            gpu.zero(&mut buf)?;
            b.compute_class(&gpu, 0..np, &mut buf, (la, lb, lk))?;
            let got = gpu.download(&buf)?;
            let st = compare(&got, &reference);
            if st.check(1e-14, 1e-12).is_err() {
                eprintln!("{key} {}: REJECTED, {st}", c.schedule);
                continue;
            }
            let mut best = f64::INFINITY;
            for _ in 0..reps {
                let ms = device_ms(&gpu, || {
                    b.compute_class(&gpu, 0..np, &mut buf, (la, lb, lk))
                })?;
                best = best.min(ms);
            }
            let (_, k) = b.kernel(la, lb, lk).unwrap();
            let regs = k.registers().unwrap_or(-1);
            let local = k.local_bytes().unwrap_or(-1);
            eprintln!(
                "{key} {:>14}: {best:9.3} ms {:7.1} GFLOP/s regs {regs:3} local {local:5}",
                c.schedule,
                flops / best * 1e-6
            );
            times.push((c.schedule.clone(), best, regs, local));
        }
        // back to the production kernel for the next classes of this pair class
        b.set_kernel(&gpu, prod, prod.source)?;
        if compile_only || times.is_empty() {
            continue;
        }
        let (bs, bt, _, _) = times
            .iter()
            .min_by(|x, y| x.1.partial_cmp(&y.1).unwrap())
            .unwrap()
            .clone();
        let cand_json: Vec<String> = times
            .iter()
            .map(|(s, t, r, l)| format!("\"{s}\": [{t:.4}, {r}, {l}]"))
            .collect();
        let line = format!(
            "{{\"key\": \"{key}\", \"schedule\": \"{bs}\", \"ms\": {bt:.4}, \"gflops\": {:.1}, \"workload\": \"{molname}/{bname}/{aname}\", \"pairs\": {np}, \"card\": \"{}\", \"candidates\": {{{}}}}}",
            flops / bt * 1e-6,
            gpu.info().name,
            cand_json.join(", ")
        );
        println!("{line}");
        if let Some(f) = out.as_mut() {
            writeln!(f, "{line}").unwrap();
        }
    }
    Ok(())
}
