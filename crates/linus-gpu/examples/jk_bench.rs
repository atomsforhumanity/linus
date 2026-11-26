//! Benchmark of the density-fitted J/K engines on the device, per call: the SCF's J and K
//! (`df::GpuDfJk`, global or range-separated exchange), the two-electron gradient
//! (`grad::df::GpuDfGrad`) and the response J/K (`response::df::GpuDfResponse`), for occupied
//! orbitals from an orthonormalized pseudo-random matrix (no SCF: the work does not depend on the
//! orbitals' values). One JSON line per call with the engine's phase times (device events) and
//! checksums of the results (E_J, E_K and the gradient's norm, with their bits), so two builds or
//! two variants can be compared for speed and agreement. For timing and for Nsight.
//!
//!   jk_bench [--molecule u1an39 | --xyz FILE] [--basis def2-tzvp] [--aux def2-universal-jkfit]
//!            [--omega w --alpha a --beta b] [--reps 3] [--k-block n] [--others-reserve-gib 14]
//!            [--no-scf] [--grad] [--resp nd] [--j-only] [--k-only] [--open] [--dump-k FILE]
//!            [--stats] [--host-setup] [--check] [--grad-unfused | --grad-fused] [--grad-j]
//!            [--grad-classic] [--panel-gemm strided|mma|mma_sm90a]
//!            [--mma-tile bm,bn,wm,wn,st] [--panel-ws-mib m]
//!
//! `--omega w`: the range-separated exchange a/r + b erf(w r)/r (`--alpha a --beta b`; wB97X's
//! 0.167 and 0.833 by default), as the device SCF loop's K (exchange fraction 1). Without it K is
//! the 1/r exchange. `--panel-gemm`: by default linus's choice (the sm_90a tensor-core kernel on
//! compute capability 9.0, else cuBLAS). `--check`: K against the other panel kernel (cuBLAS's
//! against the tensor cores') on a second engine.

mod common;
use common::{arg, flag, orbitals};
use linus_gpu::cpu::Operator;
use linus_gpu::df::{Factor, GpuDfJk, GpuDfJkOptions, GpuJkDeviceRequest, PanelKernel};
use linus_gpu::grad::df::{DerivPlanes, GpuDfGrad, GpuDfGradOptions};
use linus_gpu::harness;
use linus_gpu::response::df::{GpuDfResponse, GpuDfResponseOptions};
use linus_gpu::{Gpu, GpuError};
use std::time::Instant;

/// `len` deterministic pseudo-random values in [-0.5, 0.5).
fn random(len: usize, seed: u64) -> Vec<f64> {
    let mut s = seed;
    (0..len)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 11) as f64 / (1u64 << 53) as f64) - 0.5
        })
        .collect()
}

/// D = sum_s C_s C_s^T (n x n, column-major).
fn density(n: usize, cs: &[(&[f64], usize)]) -> Vec<f64> {
    let mut d = vec![0.0; n * n];
    for &(c, o) in cs {
        for i in 0..o {
            let col = &c[i * n..(i + 1) * n];
            for b in 0..n {
                let cb = col[b];
                for a in 0..n {
                    d[a + b * n] += col[a] * cb;
                }
            }
        }
    }
    d
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    // a fixed order (the same in every run)
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn hex(x: f64) -> String {
    format!("{:016x}", x.to_bits())
}

/// The layout's sparsity (no device): stored rows against nbf^2 / 2, and the elements of the
/// shell panels (each shell's functions against its kept partner shells, both triangles) against
/// nbf^2, with the gathered orbitals' columns (the layout's sizes the storage plan reads).
fn layout_stats(basis: &linus_gpu::cpu::BasisSet, aux: &linus_gpu::cpu::BasisSet) {
    use linus_gpu::df::DfLayout;
    let lay = DfLayout::new(
        basis,
        &linus_gpu::cpu::Aux3c::new(aux),
        linus_gpu::cpu::DfThresholds::default().pair,
    );
    let rows = lay.rows(basis);
    let (nbf, ns) = (basis.nbf, basis.shells.len());
    let gib = |b: usize| b as f64 / 1073741824.0;
    println!(
        "{{\"what\":\"stats\",\"nbf\":{nbf},\"naux\":{},\"nshell\":{ns},\"pairs_kept\":{},\"pairs_all\":{},\"nx\":{},\
         \"rows_frac\":{:.4},\"panel_frac\":{:.4},\"ncg\":{},\"max_panel_rows\":{},\"tensor_gib\":{:.2},\
         \"panels_gib\":{:.2},\"dense_gib\":{:.2}}}",
        aux.nbf,
        lay.pairs.len(),
        lay.nshell_pairs,
        lay.nx,
        lay.nx as f64 / (nbf * (nbf + 1) / 2) as f64,
        rows.np as f64 / (nbf * nbf) as f64,
        rows.ncg,
        rows.max_panel_rows,
        gib(lay.tensor_bytes()),
        gib(rows.np * aux.nbf * 8),
        gib(nbf * nbf * aux.nbf * 8),
    );
}

fn main() -> Result<(), GpuError> {
    let args: Vec<String> = std::env::args().collect();
    let name = arg(&args, "--molecule").unwrap_or_else(|| "u1an39".into());
    let bname = arg(&args, "--basis").unwrap_or_else(|| "def2-tzvp".into());
    let aname = arg(&args, "--aux").unwrap_or_else(|| "def2-universal-jkfit".into());
    let omega: Option<f64> = arg(&args, "--omega").map(|v| v.parse().unwrap());
    let alpha: f64 = arg(&args, "--alpha").map_or(0.167, |v| v.parse().unwrap());
    let beta: f64 = arg(&args, "--beta").map_or(0.833, |v| v.parse().unwrap());
    let reps: usize = arg(&args, "--reps").map_or(3, |v| v.parse().unwrap());
    let k_block: usize = arg(&args, "--k-block").map_or(0, |v| v.parse().unwrap());
    let reserve: f64 = arg(&args, "--others-reserve-gib").map_or(14.0, |v| v.parse().unwrap());
    let open = flag(&args, "--open");
    let (want_j, want_k) = (!flag(&args, "--k-only"), !flag(&args, "--j-only"));
    let mol = match arg(&args, "--xyz") {
        Some(path) => harness::load_xyz(std::path::Path::new(&path))?,
        None => harness::structure(&name)?,
    };
    let basis = harness::basis(&mol, &bname)?;
    let aux = harness::basis(&mol, &aname)?;
    let nel = mol.nelectron() as usize;
    let n = basis.nbf;
    if flag(&args, "--stats") {
        layout_stats(&basis, &aux);
        return Ok(());
    }
    if flag(&args, "--host-setup") {
        // the DF setup's host parts, timed one by one (no device)
        use linus_gpu::cpu::{Aux3c, DfThresholds, PairOptions, PairSet};
        let th = DfThresholds::default();
        let t = Instant::now();
        let pl = PairSet::build(&basis, PairOptions::default());
        let t_pairs = t.elapsed().as_secs_f64();
        let t = Instant::now();
        let _q = pl.schwarz();
        let t_schwarz = t.elapsed().as_secs_f64();
        let t = Instant::now();
        let aux3 = Aux3c::new(&aux);
        let lay = linus_gpu::df::DfLayout::new(&basis, &aux3, th.pair);
        let t_layout = t.elapsed().as_secs_f64();
        let t = Instant::now();
        let _pp = linus_gpu::df::PanelPlan::new(&basis, &lay);
        let t_panels = t.elapsed().as_secs_f64();
        let t = Instant::now();
        let _h =
            linus_gpu::eri3c::Eri3cHost::from_pair3c(&basis, &aux, &lay.pairs, &lay.x0, th.prim);
        let t_pack = t.elapsed().as_secs_f64();
        let t = Instant::now();
        let _m = linus_gpu::cpu::metric(&aux3, 0.0, &Operator::Coulomb);
        let t_metric = t.elapsed().as_secs_f64();
        println!(
            "{{\"what\":\"host_setup\",\"threads\":{},\"pairset\":{t_pairs:.4},\"schwarz\":{t_schwarz:.4},\
             \"layout_total\":{t_layout:.4},\"panels\":{t_panels:.4},\"pack_ints\":{t_pack:.4},\"metric\":{t_metric:.4}}}",
            rayon::current_num_threads()
        );
        return Ok(());
    }
    let gpu = Gpu::new(0)?;
    let chans: Vec<usize> = if open {
        vec![nel.div_ceil(2), nel / 2]
    } else {
        vec![nel / 2]
    };
    let cs: Vec<Vec<f64>> = chans
        .iter()
        .enumerate()
        .map(|(s, &o)| orbitals(n, o, 7 + s as u64))
        .collect();
    let occ: Vec<(&[f64], usize)> = cs.iter().zip(&chans).map(|(c, &o)| (&c[..], o)).collect();
    let xop = match omega {
        Some(w) => Operator::Attenuated {
            coulomb: alpha,
            long_range: beta,
            omega: w,
        },
        None => Operator::Coulomb,
    };
    let opts = GpuDfJkOptions {
        omega,
        k_panel_kernel: match arg(&args, "--panel-gemm").as_deref() {
            Some("strided") => PanelKernel::Strided,
            Some("mma") => PanelKernel::Mma,
            Some("mma_sm90a") => PanelKernel::MmaSm90a,
            _ if gpu.info().cc == (9, 0) => PanelKernel::MmaSm90a,
            _ => PanelKernel::Strided,
        },
        k_panel_mma_tile: arg(&args, "--mma-tile").map_or([0; 5], |v| {
            let t: Vec<usize> = v.split(',').map(|x| x.parse().unwrap()).collect();
            std::array::from_fn(|i| t.get(i).copied().unwrap_or(0))
        }),
        k_panel_workspace: arg(&args, "--panel-ws-mib")
            .map_or(0, |v| v.parse::<usize>().unwrap() << 20),
        k_block,
        // the panel path's block sized for these orbitals
        k_columns: Some(chans.iter().sum()),
        others_reserve: (reserve * 1073741824.0) as usize,
        ..GpuDfJkOptions::default()
    };
    let t = Instant::now();
    let mut eng = GpuDfJk::new(&gpu, &basis, &aux, &opts)?;
    if omega.is_some() {
        eng.set_exchange_operator(&gpu, xop)?;
    }
    gpu.synchronize()?;
    let t_setup = t.elapsed().as_secs_f64();
    let r = eng.resolved().clone();
    println!(
        "{{\"what\":\"setup\",\"molecule\":\"{name}\",\"basis\":\"{bname}\",\"nbf\":{n},\"naux\":{},\"nx\":{},\
         \"nocc\":{:?},\"k_panel_kernel\":\"{}\",\"storage\":\"{:?}\",\"lr_storage\":\"{:?}\",\"k_block\":{},\
         \"tensor_gib\":{:.3},\"wall\":{t_setup:.4},\"times\":\"{:?}\"}}",
        r.naux,
        r.nx,
        chans,
        r.k_panel_kernel.name(),
        r.storage,
        r.lr_storage,
        r.k_block,
        r.tensor_bytes as f64 / 1073741824.0,
        eng.setup_times
    );
    let dens = density(n, &occ);
    if !flag(&args, "--no-scf") {
        // the device loop's operands: the total density and each channel's factor side by side
        let ncols: Vec<usize> = if want_k { chans.clone() } else { vec![] };
        let mut flat = Vec::new();
        for c in &cs {
            flat.extend_from_slice(c);
        }
        let dbuf = gpu.upload(&dens)?;
        let cbuf = gpu.upload(&flat)?;
        let mut jbuf = gpu.alloc_zeros::<f64>(n * n)?;
        let mut kbuf = gpu.alloc_zeros::<f64>(n * n * chans.len())?;
        let dch: Vec<Vec<f64>> = (0..chans.len())
            .map(|s| density(n, &occ[s..s + 1]))
            .collect();
        let mut last_k = Vec::new();
        for rep in 0..=reps {
            gpu.synchronize()?;
            let t = Instant::now();
            eng.compute_device(
                &gpu,
                &GpuJkDeviceRequest {
                    dens: want_j.then_some(&dbuf),
                    ndens: if want_j { 1 } else { 0 },
                    factors: want_k.then_some(&cbuf),
                    ncols: &ncols,
                },
                want_j.then_some(&mut jbuf),
                want_k.then_some(&mut kbuf),
            )?;
            gpu.synchronize()?;
            let wall = t.elapsed().as_secs_f64();
            let jh = gpu.download(&jbuf)?;
            let kh = gpu.download(&kbuf)?;
            let ej = dot(&dens, &jh);
            let ek: f64 = (0..chans.len())
                .map(|s| dot(&dch[s], &kh[s * n * n..(s + 1) * n * n]))
                .sum();
            if rep == reps {
                if let Some(p) = arg(&args, "--dump-k") {
                    let bytes: Vec<u8> =
                        jh.iter().chain(&kh).flat_map(|x| x.to_le_bytes()).collect();
                    std::fs::write(&p, bytes).expect("dump");
                }
            }
            let l = &eng.last;
            println!(
                "{{\"what\":\"jk\",\"rep\":{rep},\"wall\":{wall:.5},\"total\":{:.5},\"j\":{:.5},\"k_unpack\":{:.5},\
                 \"k_gemm1\":{:.5},\"k_gemm2\":{:.5},\"stream\":{:.5},\"e_j\":{ej:.12},\"e_k\":{ek:.12},\
                 \"e_j_hex\":\"{}\",\"e_k_hex\":\"{}\"}}",
                l.total,
                l.j,
                l.k_unpack,
                l.k_gemm1,
                l.k_gemm2,
                l.stream,
                hex(ej),
                hex(ek),
            );
            if rep == reps {
                last_k = jh.iter().chain(&kh).copied().collect();
            }
        }
        if flag(&args, "--check") {
            // the other panel kernel (cuBLAS's against the tensor cores') on a second engine
            drop(eng);
            let other = match r.k_panel_kernel {
                PanelKernel::Strided => PanelKernel::Mma,
                _ => PanelKernel::Strided,
            };
            let o2 = GpuDfJkOptions {
                k_panel_kernel: other,
                ..opts.clone()
            };
            let mut e2 = GpuDfJk::new(&gpu, &basis, &aux, &o2)?;
            if omega.is_some() {
                e2.set_exchange_operator(&gpu, xop)?;
            }
            e2.compute_device(
                &gpu,
                &GpuJkDeviceRequest {
                    dens: want_j.then_some(&dbuf),
                    ndens: if want_j { 1 } else { 0 },
                    factors: want_k.then_some(&cbuf),
                    ncols: &ncols,
                },
                want_j.then_some(&mut jbuf),
                want_k.then_some(&mut kbuf),
            )?;
            let other: Vec<f64> = gpu
                .download(&jbuf)?
                .into_iter()
                .chain(gpu.download(&kbuf)?)
                .collect();
            let (mut dj, mut dk, mut mk) = (0.0f64, 0.0f64, 0.0f64);
            for (i, (a, b)) in last_k.iter().zip(&other).enumerate() {
                if i < n * n {
                    dj = dj.max((a - b).abs());
                } else {
                    dk = dk.max((a - b).abs());
                    mk = mk.max(a.abs());
                }
            }
            let ek: f64 = (0..chans.len())
                .map(|s| dot(&dch[s], &other[(s + 1) * n * n..(s + 2) * n * n]))
                .sum();
            println!(
                "{{\"what\":\"check\",\"other_kernel\":\"{}\",\"max_dj\":{dj:.3e},\"max_dk\":{dk:.3e},\"max_k\":{mk:.3e},\"e_k_other\":{ek:.12}}}",
                e2.resolved().k_panel_kernel.name()
            );
            return Ok(());
        }
    }
    if flag(&args, "--grad") {
        let t = Instant::now();
        let gopts = GpuDfGradOptions {
            // by default linus's choice: the stored planes (or the fused kernels where they do
            // not fit) on FP64-rich cards, else the fused kernels
            planes: if flag(&args, "--grad-unfused") {
                DerivPlanes::Stored
            } else if flag(&args, "--grad-fused") || !gpu.info().fp64_rich() {
                DerivPlanes::Fused
            } else {
                DerivPlanes::StoredOrFused
            },
            hermite_j: !flag(&args, "--grad-classic"),
            ..GpuDfGradOptions::default()
        };
        let mut g = GpuDfGrad::new(&gpu, &basis, &aux, &gopts)?;
        let gop = match omega {
            Some(w) => Operator::Attenuated {
                coulomb: alpha,
                long_range: beta,
                omega: w,
            },
            None => Operator::Attenuated {
                coulomb: if flag(&args, "--grad-j") { 0.0 } else { 0.2 },
                long_range: 0.0,
                omega: 0.0,
            },
        };
        g.prepare_operator(&gpu, &gop)?;
        g.grant_workspace(&gpu, &chans, &gop)?;
        gpu.synchronize()?;
        println!(
            "{{\"what\":\"grad_setup\",\"wall\":{:.4},\"times\":\"{:?}\"}}",
            t.elapsed().as_secs_f64(),
            g.setup_times
        );
        for rep in 0..=reps {
            let t = Instant::now();
            let out = g.compute(&gpu, &occ, !open, gop)?;
            let wall = t.elapsed().as_secs_f64();
            let nrm: f64 = out
                .gradient
                .iter()
                .flat_map(|v| v.iter())
                .map(|x| x * x)
                .sum::<f64>()
                .sqrt();
            let l = &g.last;
            println!(
                "{{\"what\":\"grad\",\"rep\":{rep},\"wall\":{wall:.5},\"total\":{:.5},\"density\":{:.5},\"ints\":{:.5},\
                 \"mo\":{:.5},\"fit\":{:.5},\"metric_gemm\":{:.5},\"metric_host\":{:.5},\"zd\":{:.5},\"deriv\":{:.5},\
                 \"contract\":{:.5},\"hermite_j\":{:.5},\"norm\":{nrm:.14e},\"norm_hex\":\"{}\",\"g0\":{:.14e}}}",
                l.total,
                l.density,
                l.ints,
                l.mo,
                l.fit,
                l.metric_gemm,
                l.metric_host,
                l.zd,
                l.deriv,
                l.contract,
                l.hermite_j,
                hex(nrm),
                out.gradient[0][0],
            );
        }
    }
    if let Some(nd) = arg(&args, "--resp").map(|v| v.parse::<usize>().unwrap()) {
        let t = Instant::now();
        let jk = GpuDfJk::new(&gpu, &basis, &aux, &opts)?;
        let fac: Vec<Factor> = occ.iter().map(|&(c, o)| Factor { c, ncol: o }).collect();
        let rop = omega.map(|_| xop);
        let mut resp = GpuDfResponse::new(
            &gpu,
            jk,
            &basis,
            &aux,
            &fac,
            rop,
            &GpuDfResponseOptions::default(),
        )?;
        gpu.synchronize()?;
        println!(
            "{{\"what\":\"resp_setup\",\"wall\":{:.4},\"nd\":{nd}}}",
            t.elapsed().as_secs_f64()
        );
        let a: Vec<Vec<f64>> = chans
            .iter()
            .enumerate()
            .map(|(s, &o)| random(n * nd * o, 101 + s as u64))
            .collect();
        let ar: Vec<&[f64]> = a.iter().map(|v| &v[..]).collect();
        for rep in 0..=reps {
            let t = Instant::now();
            let out = resp.fock_occ(&gpu, &ar, nd, 1.0, 0.2)?;
            let wall = t.elapsed().as_secs_f64();
            let s: f64 = out.iter().map(|v| dot(v, v)).sum();
            println!(
                "{{\"what\":\"resp\",\"rep\":{rep},\"wall\":{wall:.5},\"times\":\"{:?}\",\"sumsq\":{s:.14e},\"sumsq_hex\":\"{}\"}}",
                resp.last,
                hex(s)
            );
        }
    }
    Ok(())
}
