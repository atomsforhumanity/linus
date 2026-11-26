//! The XC grid's build with the device's partition weights, phase by phase, and the XC engine's
//! setup on that grid (the setup of a large system: grid points, partition, batching, blocks,
//! screening, work lists).
//!
//!   grid_bench [--molecule urea376] [--basis def2-svp] [--grid pyscf:3] [--reps 2] [--host]
//!              [--every-pair]
//!
//! --every-pair also times the unscreened kernel (every atom pair at every point).
//! --host also builds the grid with the host's partition (slow above a few hundred atoms) and
//! compares the weights.

mod common;
use common::arg;
use linus_gpu::cpu::{GridSpec, MolecularGrid};
use linus_gpu::harness;
use linus_gpu::xc::partition::GpuPartition;
use linus_gpu::xc::{GpuXc, GpuXcOptions, ProductEngine};
use linus_gpu::{Gpu, GpuError};
use std::time::Instant;

fn main() -> Result<(), GpuError> {
    let args: Vec<String> = std::env::args().collect();
    let name = arg(&args, "--molecule").unwrap_or_else(|| "urea376".into());
    let basis_name = arg(&args, "--basis").unwrap_or_else(|| "def2-svp".into());
    let grid_name = arg(&args, "--grid").unwrap_or_else(|| "pyscf:3".into());
    let reps: usize = arg(&args, "--reps").map_or(2, |s| s.parse().unwrap());
    let spec = match grid_name.split_once(':') {
        Some(("linus", l)) => GridSpec::linus(l.parse().unwrap()),
        Some(("pyscf", l)) => GridSpec::pyscf(l.parse().unwrap()),
        _ => panic!("--grid linus:N or pyscf:N"),
    };
    let gpu = Gpu::new(0)?;
    let mol = harness::structure(&name)?;
    let basis = harness::basis(&mol, &basis_name)?;
    println!(
        "{name}: {} atoms, {basis_name} {} functions, grid {grid_name}, {} host threads",
        mol.z.len(),
        basis.nbf,
        rayon::current_num_threads()
    );
    let part = GpuPartition {
        gpu: &gpu,
        allowance: None,
    };
    let mut grid = None;
    for rep in 0..reps {
        let t = Instant::now();
        let (g, times) = MolecularGrid::build_timed(&mol.z, &mol.positions, &spec, Some(&part))
            .map_err(|e| GpuError::Contract(e.to_string()))?;
        let wall = t.elapsed().as_secs_f64();
        println!(
            "grid (device partition) {rep}: {wall:.3} s = points {:.3} + partition {:.3} + \
             cutoff/batches {:.3}; {} points of {} before the cutoff, {} batches",
            times.points,
            times.partition,
            times.batches,
            g.x.len(),
            g.npoints_before_cutoff,
            g.batches.len()
        );
        grid = Some(g);
    }
    let grid = grid.unwrap();
    // the screened kernel alone on the grid's points: the atoms whose cell functions it
    // computes per point
    if let Some(t) = grid.partitioner().pair_tables() {
        let t0 = Instant::now();
        if let Some((w, kept)) =
            part.screened(&t, &grid.x, &grid.y, &grid.z, &grid.atom, &grid.w_atomic)?
        {
            let same = w
                .iter()
                .zip(&grid.w)
                .all(|(a, b)| a.to_bits() == b.to_bits());
            println!(
                "screened partition: {:.3} s, {kept:.1} of {} atoms per point; the grid's \
                 weights bitwise {same}",
                t0.elapsed().as_secs_f64(),
                t.xyz.len()
            );
        }
        if args.iter().any(|a| a == "--every-pair") {
            let t0 = Instant::now();
            let w = part.every_pair(&t, &grid.x, &grid.y, &grid.z, &grid.atom, &grid.w_atomic)?;
            let same = w
                .iter()
                .zip(&grid.w)
                .all(|(a, b)| a.to_bits() == b.to_bits());
            println!(
                "every-pair partition: {:.3} s; the grid's weights bitwise {same}",
                t0.elapsed().as_secs_f64()
            );
        }
    }
    if args.iter().any(|a| a == "--host") {
        let t = Instant::now();
        let (h, times) = MolecularGrid::build_timed(&mol.z, &mol.positions, &spec, None)
            .map_err(|e| GpuError::Contract(e.to_string()))?;
        let wall = t.elapsed().as_secs_f64();
        let same = h.w.len() == grid.w.len()
            && h.w
                .iter()
                .zip(&grid.w)
                .all(|(a, b)| a.to_bits() == b.to_bits());
        let dmax = if h.w.len() == grid.w.len() {
            h.w.iter()
                .zip(&grid.w)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0, f64::max)
        } else {
            f64::NAN
        };
        println!(
            "grid (host partition): {wall:.3} s = points+partition {:.3} + cutoff/batches {:.3}; \
             {} points; device weights bitwise {same}, max |dw| {dmax:.3e}",
            times.points,
            times.batches,
            h.x.len()
        );
    }
    let opts = GpuXcOptions {
        products: ProductEngine::Grouped,
        ..GpuXcOptions::default()
    };
    for rep in 0..reps {
        let t = Instant::now();
        let e = GpuXc::new(&gpu, &basis, &grid, "b3lyp", &opts)?;
        gpu.synchronize()?;
        let st = e.stats();
        let p = st.setup_layout;
        println!(
            "V_xc setup {rep}: {:.3} s (host blocks {:.3}, layout blocks {:.3}, screen {:.3}, \
             lists {:.3}, plans {:.3}); {} blocks, values {} ({:.2}% of nbf npts), largest block \
             {} functions, {:.1} GFMA per product pass",
            t.elapsed().as_secs_f64(),
            st.setup_blocks,
            p.blocks,
            p.screen,
            p.lists,
            st.setup_plans,
            st.nblocks,
            st.values,
            100.0 * st.kept_fraction,
            st.max_block_funcs,
            st.product_fma * 1e-9
        );
    }
    Ok(())
}
