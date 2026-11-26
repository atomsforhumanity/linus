//! Benchmark of VV10 on the device (`linus_gpu::xc::vv10`): one molecule's `GpuVv10` (energy and
//! potential) and `GpuVv10Grad` calls on its NLC grid in each pair arithmetic, for occupied
//! orbitals from an orthonormalized random matrix (no SCF: the work depends on the number of
//! active points, not on the orbitals' values), with the phase times of each call and the
//! differences of each arithmetic from FP64. For timing and for Nsight Compute (`--reps 1`).
//!
//!   vv10_bench [--molecule u1an39] [--basis def2-svp] [--grid linus:1] [--method wb97m-v]
//!              [--math fp64,df64,fp32] [--reps 3] [--no-grad] [--no-energy]

mod common;
use common::{arg, median_warm, orbitals};
use linus_gpu::cpu::GridSpec;
use linus_gpu::harness;
use linus_gpu::xc::vv10::{GpuVv10, GpuVv10Grad, GpuVv10GradOptions, GpuVv10Options, Vv10Math};
use linus_gpu::xc::XcChannel;
use linus_gpu::{Gpu, GpuError};

fn main() -> Result<(), GpuError> {
    let args: Vec<String> = std::env::args().collect();
    let name = arg(&args, "--molecule").unwrap_or_else(|| "u1an39".into());
    let basis_name = arg(&args, "--basis").unwrap_or_else(|| "def2-svp".into());
    let grid_name = arg(&args, "--grid").unwrap_or_else(|| "linus:1".into());
    let method = arg(&args, "--method").unwrap_or_else(|| "wb97m-v".into());
    let reps: usize = arg(&args, "--reps").map_or(3, |s| s.parse().unwrap());
    let maths: Vec<Vv10Math> = arg(&args, "--math")
        .unwrap_or_else(|| "fp64,df64,fp32".into())
        .split(',')
        .map(|s| Vv10Math::from_name(s).expect("--math: fp64, df64, fp32"))
        .collect();
    let spec = match grid_name.split_once(':') {
        Some(("linus", l)) => GridSpec::linus(l.parse().unwrap()),
        Some(("pyscf", l)) => GridSpec::pyscf(l.parse().unwrap()),
        _ => panic!("--grid linus:N or pyscf:N"),
    };
    let gpu = Gpu::new(0)?;
    let mol = harness::structure(&name)?;
    let basis = harness::basis(&mol, &basis_name)?;
    let grid = harness::grid(&mol, &spec)?;
    let nbf = basis.nbf;
    let nocc = (mol.nelectron() / 2) as usize;
    let c = orbitals(nbf, nocc, 7);
    let occ = vec![1.0; nocc];
    let mut dens = vec![0.0; nbf * nbf];
    for i in 0..nocc {
        for m in 0..nbf {
            for n in 0..nbf {
                dens[m * nbf + n] += 2.0 * c[i * nbf + m] * c[i * nbf + n];
            }
        }
    }
    println!(
        "{name} {basis_name} ({nbf} functions, {nocc} occupied), NLC grid {grid_name}: {} points",
        grid.x.len()
    );
    let mut e64 = None;
    let mut g64: Option<Vec<[f64; 3]>> = None;
    for &math in &maths {
        if !args.iter().any(|a| a == "--no-energy") {
            let mut nl = GpuVv10::new(
                &gpu,
                &basis,
                &grid,
                &method,
                &GpuVv10Options {
                    math,
                    ..GpuVv10Options::default()
                },
            )?;
            let ch = [XcChannel {
                c: &c,
                ncol: nocc,
                occ: &occ,
            }];
            let (mut tot, mut pk, mut pt, mut dn, mut po) =
                (vec![], vec![], vec![], vec![], vec![]);
            let mut e = 0.0;
            for _ in 0..reps + 1 {
                e = nl.compute(&gpu, &ch, true)?.0;
                // the engines' seconds, printed in ms
                let l = &nl.last;
                tot.push(l.total * 1e3);
                pk.push(l.pairs.pairs * 1e3);
                pt.push(l.pairs.total * 1e3);
                dn.push(l.density * 1e3);
                po.push(l.potential * 1e3);
            }
            // the first call compiles and warms up
            let (tot, pk, pt, dn, po) = (
                median_warm(&tot),
                median_warm(&pk),
                median_warm(&pt),
                median_warm(&dn),
                median_warm(&po),
            );
            let n = nl.last.pairs.active as f64;
            let de = e64.map_or(0.0, |r: f64| e - r);
            if math == Vv10Math::Fp64 {
                e64 = Some(e);
            }
            println!(
                "energy {:5}: call {tot:8.2} ms = densities {dn:7.2} + pairs {pt:8.2} (kernel {pk:8.2}, \
                 {:.3} ns/pair... {:.2e} pairs/s) + potential {po:7.2}; active {n}, E {e:.12} (dE {de:.1e})",
                math.name(),
                pk * 1e6 / (n * n),
                n * n / (pk * 1e-3),
            );
        }
        if !args.iter().any(|a| a == "--no-grad") {
            let mut g = GpuVv10Grad::new(
                &gpu,
                &basis,
                &grid,
                &spec,
                &method,
                &GpuVv10GradOptions {
                    math,
                    ..GpuVv10GradOptions::default()
                },
            )?;
            let (mut tot, mut pk) = (vec![], vec![]);
            let mut gr = vec![];
            for _ in 0..reps + 1 {
                gr = g.compute(&gpu, &dens, true)?;
                tot.push(g.last.total * 1e3);
                pk.push(g.last.pairs.pairs * 1e3);
            }
            let (tot, pk) = (median_warm(&tot), median_warm(&pk));
            let dg = g64.as_ref().map_or(0.0, |r| {
                r.iter()
                    .flatten()
                    .zip(gr.iter().flatten())
                    .fold(0.0f64, |m, (a, b)| m.max((a - b).abs()))
            });
            if math == Vv10Math::Fp64 {
                g64 = Some(gr.clone());
            }
            let n = g.last.pairs.active as f64;
            println!(
                "grad   {:5}: call {tot:8.2} ms, pair kernel {pk:8.2} ms ({:.3} ns/pair); engine {:?}; max dG {dg:.1e}",
                math.name(),
                pk * 1e6 / (n * n),
                g.last.engine
            );
        }
    }
    Ok(())
}
