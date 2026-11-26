//! Benchmark of the Hessian's explicit XC terms on the device (`linus_gpu::hess::xc`): one
//! molecule's `GpuXcHess` setup and calls (V^x C, dV^y C, the XC energy's second derivative
//! and E1) for occupied orbitals from an orthonormalized random matrix (no SCF: the work does
//! not depend on the orbitals' values), with the phase times of each call. For timing and for
//! Nsight Compute (profile one call: `--reps 1`).
//!
//!   xchess_bench [--molecule u1an39] [--basis def2-svp] [--grid linus:3] [--method b3lyp]
//!                [--open] [--no-grid-response] [--reps 2] [--v-gib 24] [--fp32]
//!
//! With `--fp32` each call also runs the FP32 potential products and reports the largest
//! differences from the FP64 call.

mod common;
use common::{arg, density, orbitals};
use linus_gpu::cpu::GridSpec;
use linus_gpu::harness;
use linus_gpu::hess::xc::{GpuXcHess, GpuXcHessOptions};
use linus_gpu::{Gpu, GpuError};
use std::time::Instant;

fn main() -> Result<(), GpuError> {
    let args: Vec<String> = std::env::args().collect();
    let name = arg(&args, "--molecule").unwrap_or_else(|| "u1an39".into());
    let bname = arg(&args, "--basis").unwrap_or_else(|| "def2-svp".into());
    let gname = arg(&args, "--grid").unwrap_or_else(|| "linus:3".into());
    let method = arg(&args, "--method").unwrap_or_else(|| "b3lyp".into());
    let reps: usize = arg(&args, "--reps").map_or(2, |v| v.parse().unwrap());
    let v_gib: usize = arg(&args, "--v-gib").map_or(24, |v| v.parse().unwrap());
    let open = args.iter().any(|a| a == "--open");
    let fp32 = args.iter().any(|a| a == "--fp32");
    let grid_terms = !args.iter().any(|a| a == "--no-grid-response");
    let mol = harness::structure(&name)?;
    let basis = harness::basis(&mol, &bname)?;
    let spec = match gname.split_once(':') {
        Some(("linus", l)) => GridSpec::linus(l.parse().unwrap()),
        Some(("pyscf", l)) => GridSpec::pyscf(l.parse().unwrap()),
        _ => return Err(GpuError::Input(format!("grid {gname}: linus:N or pyscf:N"))),
    };
    let grid = harness::grid(&mol, &spec)?;
    let n = basis.nbf;
    let ne = mol.nelectron() as usize;
    let (na, nb) = if open {
        (ne.div_ceil(2), ne / 2)
    } else {
        (ne / 2, ne / 2)
    };
    let gpu = Gpu::new(0)?;
    let t0 = Instant::now();
    let nsp = if open { 2 } else { 1 };
    let mut eng = GpuXcHess::new(
        &gpu,
        &basis,
        &grid,
        &method,
        nsp,
        grid_terms,
        &GpuXcHessOptions {
            v_bytes: v_gib << 30,
            ..GpuXcHessOptions::default()
        },
    )?;
    let st = eng.stats().clone();
    println!(
        "{name}/{bname} {gname} {method} {}: n {n}, natom {}, {} points, {} batches, {} \
         chunks; slots fixed {} grid {} (points {:.1} %), vslot {:.1} GFLOP; setup {:.2} s \
         (layout {:.2}, weight derivatives {:.2})",
        if open { "open" } else { "closed" },
        mol.z.len(),
        grid.x.len(),
        st.nbatches,
        st.nchunks,
        st.slots_fixed,
        st.slots_grid,
        100.0 * st.grid_points as f64 / st.grid_points_dense.max(1) as f64,
        st.vslot_flops * 1e-9,
        t0.elapsed().as_secs_f64(),
        st.layout_seconds,
        st.weights_seconds
    );
    let ca = orbitals(n, na, 1);
    let cb = orbitals(n, nb, 2);
    let dens: Vec<Vec<f64>> = if open {
        vec![density(&ca, n, na, 1.0), density(&cb, n, nb, 1.0)]
    } else {
        vec![density(&ca, n, na, 2.0)]
    };
    let drefs: Vec<&[f64]> = dens.iter().map(|d| d.as_slice()).collect();
    let occ: Vec<(&[f64], usize)> = if open {
        vec![(&ca[..], na), (&cb[..], nb)]
    } else {
        vec![(&ca[..], na)]
    };
    let mut eng32 = if fp32 {
        Some(GpuXcHess::new(
            &gpu,
            &basis,
            &grid,
            &method,
            nsp,
            grid_terms,
            &GpuXcHessOptions {
                v_bytes: v_gib << 30,
                single: true,
                ..GpuXcHessOptions::default()
            },
        )?)
    } else {
        None
    };
    // --compare-mib M: a second engine with M MiB of per-batch workspace (its own chunks)
    if let Some(mib) = arg(&args, "--compare-mib") {
        let mut e2 = GpuXcHess::new(
            &gpu,
            &basis,
            &grid,
            &method,
            nsp,
            grid_terms,
            &GpuXcHessOptions {
                v_bytes: v_gib << 30,
                workspace: mib.parse::<usize>().unwrap() << 20,
                ..GpuXcHessOptions::default()
            },
        )?;
        let r = eng.compute(&gpu, &drefs, &occ)?;
        let r2 = e2.compute(&gpu, &drefs, &occ)?;
        let md = |a: &[f64], b: &[f64]| {
            a.iter()
                .zip(b)
                .map(|(x, y)| (x - y).abs())
                .fold(0.0f64, f64::max)
        };
        let mx = |v: &[Vec<f64>], w: &[Vec<f64>]| {
            v.iter().zip(w).map(|(a, b)| md(a, b)).fold(0.0, f64::max)
        };
        println!(
            "chunks {} vs {}: max |dHfix| {:.2e} |dHgrid| {:.2e} |dV^x C| {:.2e} |d dV^y C| {:.2e}; \
             max |Hfix| {:.2e}",
            eng.stats().nchunks,
            e2.stats().nchunks,
            md(&r.hfix, &r2.hfix),
            md(&r.hgrid, &r2.hgrid),
            mx(&r.vfix_c, &r2.vfix_c),
            mx(&r.vgrid_c, &r2.vgrid_c),
            r.hfix.iter().fold(0.0f64, |m, v| m.max(v.abs()))
        );
    }
    for rep in 0..reps {
        let r = eng.compute(&gpu, &drefs, &occ)?;
        let l = &eng.last;
        let hn = r
            .hfix
            .iter()
            .chain(&r.hgrid)
            .fold(0.0f64, |m, v| m.max(v.abs()));
        println!(
            "call {rep}: {:.3} s (densities {:.3}, functional {:.3}, main {:.3}, V C {:.3}; {} \
             pass(es)); max |H| {hn:.3e}",
            l.total, l.densities, l.functional, l.main, l.vc, l.passes
        );
        if let Some(e32) = eng32.as_mut() {
            let r32 = e32.compute(&gpu, &drefs, &occ)?;
            let md = |a: &[f64], b: &[f64]| {
                a.iter()
                    .zip(b)
                    .map(|(x, y)| (x - y).abs())
                    .fold(0.0f64, f64::max)
            };
            let dh = md(&r.hfix, &r32.hfix).max(md(&r.hgrid, &r32.hgrid));
            let dv = r
                .vfix_c
                .iter()
                .zip(&r32.vfix_c)
                .map(|(a, b)| md(a, b))
                .fold(0.0, f64::max);
            let dg = r
                .vgrid_c
                .iter()
                .zip(&r32.vgrid_c)
                .map(|(a, b)| md(a, b))
                .fold(0.0, f64::max);
            let l = &e32.last;
            println!(
                "  fp32: {:.3} s (main {:.3}); max |dH| {dh:.2e} (E is FP64 in both), \
                 |dV^x C| {dv:.2e}, |d dV^y C| {dg:.2e}",
                l.total, l.main
            );
        }
    }
    Ok(())
}
