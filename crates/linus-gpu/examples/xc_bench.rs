//! Benchmark of the XC engines' setup and calls on the device (`linus_gpu::xc::GpuXc`,
//! `linus_gpu::grad::xc::GpuXcGrad`): one molecule's grid build, the engines' setups (twice: the
//! second with the kernels loaded) and V_xc and gradient calls, for occupied orbitals from an
//! orthonormalized random matrix (no SCF: the work does not depend on the orbitals' values),
//! with the phase times of each call. For timing and for Nsight Compute (`--reps 1`).
//!
//!   xc_bench [--molecule u1an39] [--basis def2-tzvp] [--grid pyscf:3] [--method b3lyp]
//!            [--open] [--reps 2] [--no-grad] [--grouped] [--host-functional] [--compare]
//!            [--becke-warp] [--json <file>]
//!
//! `--grouped` runs the FP64 products as cuBLAS grouped GEMMs (the FP64-rich cards' choice);
//! `--becke-warp` the gradient's weight derivatives on the warp kernel at any size (the pair
//! kernel otherwise, where it fits);
//! `--host-functional` evaluates the functional with libxc on the host; `--compare` also runs a
//! V_xc call and a gradient with the other placement and prints the largest differences.

mod common;
use common::{arg, density, median_warm, orbitals};
use linus_gpu::cpu::GridSpec;
use linus_gpu::grad::xc::{GpuXcGrad, GpuXcGradOptions};
use linus_gpu::harness;
use linus_gpu::xc::{FunctionalPlace, GpuXc, GpuXcOptions, ProductEngine, XcChannel};
use linus_gpu::{Gpu, GpuError};
use std::time::Instant;

fn main() -> Result<(), GpuError> {
    let args: Vec<String> = std::env::args().collect();
    let name = arg(&args, "--molecule").unwrap_or_else(|| "u1an39".into());
    let basis_name = arg(&args, "--basis").unwrap_or_else(|| "def2-tzvp".into());
    let grid_name = arg(&args, "--grid").unwrap_or_else(|| "pyscf:3".into());
    let method = arg(&args, "--method").unwrap_or_else(|| "b3lyp".into());
    let reps: usize = arg(&args, "--reps").map_or(2, |s| s.parse().unwrap());
    let open = args.iter().any(|a| a == "--open");
    // --products vgemm|grouped (--grouped: grouped)
    let products = match arg(&args, "--products").as_deref() {
        Some("grouped") => ProductEngine::Grouped,
        Some("vgemm") => ProductEngine::Vgemm,
        _ if args.iter().any(|a| a == "--grouped") => ProductEngine::Grouped,
        _ => ProductEngine::Vgemm,
    };
    let place = if args.iter().any(|a| a == "--host-functional") {
        FunctionalPlace::Host
    } else {
        FunctionalPlace::Device
    };
    let other = match place {
        FunctionalPlace::Host => FunctionalPlace::Device,
        FunctionalPlace::Device => FunctionalPlace::Host,
    };
    let compare = args.iter().any(|a| a == "--compare");
    let spec = match grid_name.split_once(':') {
        Some(("linus", l)) => GridSpec::linus(l.parse().unwrap()),
        Some(("pyscf", l)) => GridSpec::pyscf(l.parse().unwrap()),
        _ => panic!("--grid linus:N or pyscf:N"),
    };
    let gpu = Gpu::new(0)?;
    let mol = harness::structure(&name)?;
    let basis = harness::basis(&mol, &basis_name)?;
    let t = Instant::now();
    let grid = harness::grid(&mol, &spec)?;
    let t_grid = t.elapsed().as_secs_f64();
    let nbf = basis.nbf;
    let nel = mol.nelectron() as usize;
    // open shells: a doublet, or a triplet for an even count
    let (na, nb) = match (open, nel % 2) {
        (false, _) => (nel / 2, nel / 2),
        (true, 1) => (nel / 2 + 1, nel / 2),
        (true, _) => (nel / 2 + 1, nel / 2 - 1),
    };
    let ca = orbitals(nbf, na, 7);
    let cb = orbitals(nbf, nb, 11);
    let (oa, ob) = (vec![1.0; na], vec![1.0; nb]);
    println!(
        "{name} {basis_name} {method} {} ({nbf} functions, {na}+{nb} occupied), grid {grid_name}: \
         {} points in {} batches, built in {:.3} s (host, {} threads)",
        if open { "UKS" } else { "RKS" },
        grid.x.len(),
        grid.batches.len(),
        t_grid,
        rayon::current_num_threads(),
    );
    let mut json = serde_like::Obj::default();
    json.s("molecule", &name);
    json.s("basis", &basis_name);
    json.s("grid", &grid_name);
    json.s("method", &method);
    json.b("open", open);
    json.n("nbf", nbf as f64);
    json.n("npts", grid.x.len() as f64);
    json.n("grid_build_s", t_grid);
    let opts = GpuXcOptions {
        products,
        functional: place,
        ..GpuXcOptions::default()
    };
    // the setup twice: the first loads (or compiles) the kernels
    let mut setups = vec![];
    let mut eng = None;
    for _ in 0..2 {
        let t = Instant::now();
        let e = GpuXc::new(&gpu, &basis, &grid, &method, &opts)?;
        gpu.synchronize()?;
        setups.push(t.elapsed().as_secs_f64());
        eng = Some(e);
    }
    let mut eng = eng.unwrap();
    let t = Instant::now();
    let bl = linus_gpu::xc::blocks(&basis, &grid, eng.kind().order(), 1e-11, 2048)?;
    let t_blocks = t.elapsed().as_secs_f64();
    let st = eng.stats().clone();
    println!(
        "V_xc setup: {:.3} s first, {:.3} s second (blocks() alone {:.3} s, {} blocks); values {} \
         ({:.1}% of nbf npts), largest block {} functions",
        setups[0],
        setups[1],
        t_blocks,
        bl.len(),
        st.values,
        100.0 * st.kept_fraction,
        st.max_block_funcs
    );
    println!(
        "products: {:.1} GFMA per pass, average kept functions per batch point {:.0}",
        st.product_fma * 1e-9,
        st.product_fma / st.values.max(1) as f64
    );
    json.n("product_gfma", st.product_fma * 1e-9);
    json.n("vxc_setup_first_s", setups[0]);
    json.n("vxc_setup_s", setups[1]);
    json.n("blocks_s", t_blocks);
    let ch_a = XcChannel {
        c: &ca,
        ncol: na,
        occ: &oa,
    };
    let ch_b = XcChannel {
        c: &cb,
        ncol: nb,
        occ: &ob,
    };
    let chans: Vec<XcChannel> = if open { vec![ch_a, ch_b] } else { vec![ch_a] };
    let nsp = if open { 2 } else { 1 };
    let status = eng.prepare_functional(&gpu, nsp);
    println!(
        "functional: {:?} {}",
        place,
        match &status {
            Ok(()) => "on the device".to_string(),
            Err(e) => format!("on the host: {e}"),
        }
    );
    json.s("functional", if status.is_ok() { "device" } else { "host" });
    let (mut tot, mut co, mut de, mut fu, mut lx, mut po, mut en) =
        (vec![], vec![], vec![], vec![], vec![], vec![], vec![]);
    let mut exc = 0.0;
    let mut vref = vec![];
    for _ in 0..reps + 1 {
        let t = Instant::now();
        let r = eng.compute(&gpu, &chans, !open)?;
        exc = r.exc;
        vref = r.v;
        let wall = t.elapsed().as_secs_f64();
        let l = &eng.last;
        tot.push(wall * 1e3);
        co.push(l.colloc * 1e3);
        de.push(l.density * 1e3);
        fu.push(l.functional * 1e3);
        lx.push(l.libxc * 1e3);
        po.push(l.potential * 1e3);
        en.push(l.energy * 1e3);
    }
    let m = |v: &[f64]| median_warm(v);
    println!(
        "V_xc call: {:8.2} ms wall = colloc {:7.2} + density {:7.2} + functional {:7.2} (libxc \
         {:7.2}) + energy {:6.2} + potential {:7.2}; first call {:.2} ms; E_xc {exc:.12}",
        m(&tot),
        m(&co),
        m(&de),
        m(&fu),
        m(&lx),
        m(&en),
        m(&po),
        tot[0]
    );
    for (k, v) in [
        ("vxc_call_ms", &tot),
        ("vxc_colloc_ms", &co),
        ("vxc_density_ms", &de),
        ("vxc_functional_ms", &fu),
        ("vxc_libxc_ms", &lx),
        ("vxc_energy_ms", &en),
        ("vxc_potential_ms", &po),
    ] {
        json.n(k, m(v));
    }
    json.n("exc", exc);
    drop(eng);
    if compare {
        let mut e2 = GpuXc::new(
            &gpu,
            &basis,
            &grid,
            &method,
            &GpuXcOptions {
                functional: other,
                ..opts.clone()
            },
        )?;
        let r = e2.compute(&gpu, &chans, !open)?;
        let dv = vref
            .iter()
            .flatten()
            .zip(r.v.iter().flatten())
            .fold(0.0f64, |m, (a, b)| m.max((a - b).abs()));
        let vmax = vref.iter().flatten().fold(0.0f64, |m, a| m.max(a.abs()));
        println!(
            "compare {other:?}: dE_xc {:.3e}, max |dV| {dv:.3e} (max |V| {vmax:.3e})",
            r.exc - exc
        );
        json.n("compare_dexc", r.exc - exc);
        json.n("compare_dv", dv);
    }
    if !args.iter().any(|a| a == "--no-grad") {
        let gopts = GpuXcGradOptions {
            products,
            functional: place,
            becke_pair_shared: if args.iter().any(|a| a == "--becke-warp") {
                0
            } else {
                GpuXcGradOptions::default().becke_pair_shared
            },
            ..GpuXcGradOptions::default()
        };
        let mut setups = vec![];
        let mut g = None;
        for _ in 0..2 {
            let t = Instant::now();
            let e = GpuXcGrad::new(&gpu, &basis, &grid, &spec, &method, &gopts)?;
            gpu.synchronize()?;
            setups.push(t.elapsed().as_secs_f64());
            g = Some(e);
        }
        let mut g = g.unwrap();
        let s = g.stats().clone();
        println!(
            "gradient setup: {:.3} s first, {:.3} s second (layout {:.3}: blocks {:.3}, screen {:.3}, \
             lists {:.3}, plans {:.3})",
            setups[0],
            setups[1],
            s.setup_layout,
            s.setup_layout_parts.blocks,
            s.setup_layout_parts.screen,
            s.setup_layout_parts.lists,
            s.setup_plans
        );
        json.n("grad_setup_first_s", setups[0]);
        json.n("grad_setup_s", setups[1]);
        json.n("grad_setup_blocks_s", s.setup_layout_parts.blocks);
        json.n("grad_setup_screen_s", s.setup_layout_parts.screen);
        json.n("grad_setup_lists_s", s.setup_layout_parts.lists);
        json.n("grad_setup_plans_s", s.setup_plans);
        let dens: Vec<Vec<f64>> = if open {
            vec![density(&ca, nbf, na, 1.0), density(&cb, nbf, nb, 1.0)]
        } else {
            vec![density(&ca, nbf, na, 2.0)]
        };
        let refs: Vec<&[f64]> = dens.iter().map(|d| d.as_slice()).collect();
        let nsp = refs.len();
        g.grant_workspace(&gpu, nsp, true)?;
        let (mut tot, mut co, mut de, mut fu, mut ge, mut ct, mut gr) =
            (vec![], vec![], vec![], vec![], vec![], vec![], vec![]);
        let mut gref = vec![];
        for _ in 0..reps + 1 {
            let t = Instant::now();
            gref = g.compute(&gpu, &refs, !open, true)?;
            let wall = t.elapsed().as_secs_f64();
            let l = &g.last;
            tot.push(wall * 1e3);
            co.push(l.colloc * 1e3);
            de.push(l.density * 1e3);
            fu.push(l.functional * 1e3);
            ge.push(l.gemm * 1e3);
            ct.push(l.contract * 1e3);
            gr.push(l.grid_response * 1e3);
        }
        println!(
            "gradient call: {:8.2} ms wall = colloc {:7.2} + density {:7.2} + functional {:7.2} + \
             gemm {:7.2} + contract {:7.2} + grid response {:7.2}; first call {:.2} ms",
            m(&tot),
            m(&co),
            m(&de),
            m(&fu),
            m(&ge),
            m(&ct),
            m(&gr),
            tot[0]
        );
        if compare {
            let mut g2 = GpuXcGrad::new(
                &gpu,
                &basis,
                &grid,
                &spec,
                &method,
                &GpuXcGradOptions {
                    functional: other,
                    ..gopts.clone()
                },
            )?;
            g2.grant_workspace(&gpu, nsp, true)?;
            let r = g2.compute(&gpu, &refs, !open, true)?;
            let dg = gref
                .iter()
                .flatten()
                .zip(r.iter().flatten())
                .fold(0.0f64, |m, (a, b)| m.max((a - b).abs()));
            println!("compare {other:?}: max |dG| {dg:.3e}");
            json.n("compare_dgrad", dg);
        }
        for (k, v) in [
            ("grad_call_ms", &tot),
            ("grad_colloc_ms", &co),
            ("grad_density_ms", &de),
            ("grad_functional_ms", &fu),
            ("grad_gemm_ms", &ge),
            ("grad_contract_ms", &ct),
            ("grad_grid_response_ms", &gr),
        ] {
            json.n(k, m(v));
        }
    }
    if let Some(path) = arg(&args, "--json") {
        std::fs::write(&path, json.finish() + "\n").expect("--json: writable file");
    }
    Ok(())
}

/// A flat JSON object without a JSON crate (the examples' dependencies are the crate's).
mod serde_like {
    #[derive(Default)]
    pub struct Obj(Vec<String>);

    impl Obj {
        pub fn s(&mut self, k: &str, v: &str) {
            self.0.push(format!("\"{k}\": \"{v}\""));
        }
        pub fn n(&mut self, k: &str, v: f64) {
            self.0.push(format!("\"{k}\": {v:e}"));
        }
        pub fn b(&mut self, k: &str, v: bool) {
            self.0.push(format!("\"{k}\": {v}"));
        }
        pub fn finish(self) -> String {
            format!("{{{}}}", self.0.join(", "))
        }
    }
}
