//! Every CUDA module linus compiles, compiled with NVRTC without a device, for the datacenter
//! (sm_90) and workstation (sm_120) architectures: a syntax or type error in a kernel fails here,
//! on any machine, instead of on the first GPU that runs it. The modules: every hand-written one
//! (the registry, `linus_gpu::kernels::registry`), the PCM's and the three-center integrals'
//! generated kernels (each class for 1/r; for erf(omega r)/r one class per kind and derivative
//! order, or all with `LINUS_NVRTC_ERI3C=all`), the collocation kernels of every (l, pure) at
//! every derivative order, and the variable-size batched GEMM's tile. Every module's declared
//! entry points must be kernels of its image.
//!
//! The sources are assembled as the engines assemble them, and the device layer's prelude as
//! `Gpu::kernel_source` does (warp size 32), so `kernel_manifest` lists exactly what NVRTC is
//! given: with `LINUS_KERNEL_MANIFEST=<file>` it writes the SHA-256 of every module's source.
//!
//! NVRTC is loaded at run time; without it the tests skip with a message, unless
//! `LINUS_NVRTC_REQUIRED=1` (CI installs it from NVIDIA's `nvidia-cuda-nvrtc` wheel).
#![cfg(feature = "gpu")]

use linus_gpu::kernels::eri3c_class::GenKernel;
use linus_gpu::kernels::eri3c_gen::KERNELS as ERI3C;
use linus_gpu::kernels::eri3c_gen_arch::ARCH_KERNELS as ERI3C_ARCH;
use linus_gpu::kernels::{colloc, pcm_gen, registry, vgemm};
use linus_gpu::nvrtc::{self, Target};
use rayon::prelude::*;
use sha2::{Digest, Sha256};

/// The warp size of the architectures linus runs on (NVIDIA's).
const WARP: u32 = 32;

/// A module as the compiler gets it (before the device layer's prelude).
struct Module {
    label: String,
    source: String,
    entries: Vec<String>,
}

fn module(label: impl Into<String>, source: impl Into<String>, entries: Vec<String>) -> Module {
    Module {
        label: label.into(),
        source: source.into(),
        entries,
    }
}

/// The three-center classes' modules: every class for 1/r (`erf` false) or, for erf(omega r)/r,
/// every class with `all`, else the first class of each kind and maximal angular momenta.
fn eri3c_modules(erf: bool, all: bool) -> Vec<Module> {
    let gks: Vec<&GenKernel> = ERI3C
        .iter()
        .chain(ERI3C_ARCH.iter().map(|(_, g)| g))
        .collect();
    let pick = |g: &&GenKernel| {
        all || !erf || {
            let same_kind = gks.iter().filter(|h| h.kind == g.kind);
            let first = same_kind.clone().next().map(|h| h.entry) == Some(g.entry);
            let top = same_kind
                .max_by_key(|h| (h.la + h.lb + h.lk, h.lk))
                .map(|h| h.entry)
                == Some(g.entry);
            first || top
        }
    };
    gks.iter()
        .copied()
        .filter(pick)
        .map(|g| {
            let (name, src) = linus_gpu::eri3c::module_source(g, None, erf);
            module(name, src, vec![g.entry.to_string()])
        })
        .collect()
}

/// Every module the engines compile (see the module docs).
fn modules(eri3c_all: bool) -> Vec<Module> {
    let mut out: Vec<Module> = registry::modules()
        .into_iter()
        .map(|m| {
            module(
                m.name,
                m.source(),
                m.entries.iter().map(|e| e.to_string()).collect(),
            )
        })
        .collect();
    for (l, (e, d)) in pcm_gen::POTENTIAL
        .iter()
        .zip(pcm_gen::DERIVATIVE.iter())
        .enumerate()
    {
        out.push(module(format!("pcm_e{l}"), *e, vec![format!("pcm_e{l}")]));
        out.push(module(format!("pcm_d{l}"), *d, vec![format!("pcm_d{l}")]));
    }
    out.extend(eri3c_modules(false, true));
    out.extend(eri3c_modules(true, eri3c_all));
    let keys: Vec<(usize, bool)> = (0..=colloc::MAX_L)
        .flat_map(|l| [(l, false), (l, true)])
        .collect();
    for order in 0..=colloc::MAX_ORDER {
        let entries = keys
            .iter()
            .map(|&(l, p)| colloc::kernel_name(l, p, order))
            .collect();
        out.push(module(
            format!("colloc_order{order}"),
            colloc::module_source_for(&keys, order),
            entries,
        ));
    }
    let tile = vgemm::VgemmTile::DEFAULT;
    let entries = [(false, false), (false, true), (true, false), (true, true)]
        .iter()
        .map(|&(ta, tb)| vgemm::kernel_name(ta, tb, &tile))
        .collect();
    out.push(module("vgemm", vgemm::module_source(&tile), entries));
    out
}

/// Whether NVRTC can be loaded (the device layer panics when the library is missing): skip, or fail when
/// `LINUS_NVRTC_REQUIRED=1`.
fn nvrtc_available() -> bool {
    match std::panic::catch_unwind(nvrtc::nvrtc_version) {
        Ok(Ok(v)) => {
            eprintln!("NVRTC {}.{}", v.0, v.1);
            true
        }
        r => {
            let why = match r {
                Ok(Err(e)) => e.to_string(),
                _ => "libnvrtc not found".to_string(),
            };
            assert!(
                std::env::var("LINUS_NVRTC_REQUIRED").as_deref() != Ok("1"),
                "LINUS_NVRTC_REQUIRED=1 but NVRTC is unusable: {why}"
            );
            eprintln!("skipped, no NVRTC: {why}");
            false
        }
    }
}

/// Whether `entry` is a symbol of the image (ELF string tables hold NUL-terminated names).
fn has_symbol(image: &[u8], entry: &str) -> bool {
    let pat: Vec<u8> = [&[0u8][..], entry.as_bytes(), &[0u8][..]].concat();
    image.windows(pat.len()).any(|w| w == pat.as_slice())
}

#[test]
fn every_module_compiles_for_sm90_and_sm120() {
    if !nvrtc_available() {
        return;
    }
    let all = std::env::var("LINUS_NVRTC_ERI3C").as_deref() == Ok("all");
    let mods = modules(all);
    let archs = [90u32, 120];
    let supported = nvrtc::supported_archs().unwrap();
    for a in archs {
        assert!(supported.contains(&a), "NVRTC cannot compile for sm_{a}");
    }
    let jobs: Vec<(&Module, u32)> = mods
        .iter()
        .flat_map(|m| archs.iter().map(move |&a| (m, a)))
        .collect();
    let t = std::time::Instant::now();
    let failures: Vec<String> = jobs
        .par_iter()
        .filter_map(|&(m, arch)| {
            let src = linus_gpu::kernel_source(WARP, &m.source);
            match nvrtc::compile_uncached(&m.label, &src, &Target::Cubin(arch)) {
                Err(e) => Some(format!("{} (sm_{arch}): {e}", m.label)),
                Ok((image, _)) => {
                    let missing: Vec<&String> = m
                        .entries
                        .iter()
                        .filter(|e| !has_symbol(&image, e))
                        .collect();
                    (!missing.is_empty())
                        .then(|| format!("{} (sm_{arch}): no kernels {missing:?}", m.label))
                }
            }
        })
        .collect();
    eprintln!(
        "compiled {} modules for sm_90 and sm_120 ({} compiles) in {:.1} s",
        mods.len(),
        jobs.len(),
        t.elapsed().as_secs_f64()
    );
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// The registry's modules have distinct names and each declares the kernels the engines load;
/// with `LINUS_KERNEL_MANIFEST=<file>`, every module's source hash (no NVRTC needed).
#[test]
fn kernel_manifest() {
    let regs = registry::modules();
    let mut names: Vec<&str> = regs.iter().map(|m| m.name).collect();
    names.sort();
    let n = names.len();
    names.dedup();
    assert_eq!(names.len(), n, "two registry modules share a name");
    for m in &regs {
        assert!(!m.entries.is_empty(), "{}: no entries", m.name);
    }
    let mods = modules(true);
    if let Ok(path) = std::env::var("LINUS_KERNEL_MANIFEST") {
        let mut lines: Vec<String> = mods
            .iter()
            .map(|m| {
                let full = linus_gpu::kernel_source(WARP, &m.source);
                let h = Sha256::digest(full.as_bytes());
                let hex: String = h.iter().map(|b| format!("{b:02x}")).collect();
                format!("{}\t{hex}\t{}", m.label, full.len())
            })
            .collect();
        lines.sort();
        std::fs::write(&path, lines.join("\n") + "\n").unwrap();
        eprintln!("{} module hashes written to {path}", lines.len());
    }
}
