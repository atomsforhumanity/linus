//! Benchmark of the DF Hessian's two-electron terms on the device (`linus_gpu::hess::df`): one
//! molecule's `GpuDfHess` setup and calls (F^x C, the RI cross terms, the second-derivative
//! integrals) for occupied orbitals from an orthonormalized random matrix (no SCF: the work does
//! not depend on the orbitals' values), with the phase times of each call. For timing and for
//! Nsight Compute (profile one call: `--reps 1`).
//!
//!   hess_bench [--molecule u1an39] [--basis def2-svp] [--aux def2-universal-jkfit]
//!              [--a-x 0.2] [--omega 0.3] [--open] [--reps 2] [--emulated] [--dump FILE]

mod common;
use common::{arg, orbitals};
use linus_gpu::cpu::Operator;
use linus_gpu::harness;
use linus_gpu::hess::df::{GpuDfHess, GpuDfHessOptions};
use linus_gpu::{Gpu, GpuError};
use std::time::Instant;

fn main() -> Result<(), GpuError> {
    let args: Vec<String> = std::env::args().collect();
    let name = arg(&args, "--molecule").unwrap_or_else(|| "u1an39".into());
    let bname = arg(&args, "--basis").unwrap_or_else(|| "def2-svp".into());
    let aname = arg(&args, "--aux").unwrap_or_else(|| "def2-universal-jkfit".into());
    let a_x: f64 = arg(&args, "--a-x").map_or(0.2, |v| v.parse().unwrap());
    let omega: Option<f64> = arg(&args, "--omega").map(|v| v.parse().unwrap());
    let reps: usize = arg(&args, "--reps").map_or(2, |v| v.parse().unwrap());
    let open = args.iter().any(|a| a == "--open");
    let mol = harness::structure(&name)?;
    let basis = harness::basis(&mol, &bname)?;
    let aux = harness::basis(&mol, &aname)?;
    let nel = mol.nelectron() as usize;
    let (n, naux) = (basis.nbf, aux.nbf);
    let gpu = Gpu::new(0)?;
    let t = Instant::now();
    let mut h = GpuDfHess::new(&gpu, &basis, &aux, &GpuDfHessOptions::default())?;
    let op = match omega {
        Some(w) => Operator::Attenuated {
            coulomb: 1.0 + a_x,
            long_range: -a_x,
            omega: w,
        },
        None => Operator::Coulomb,
    };
    h.prepare_operator(&gpu, &op)?;
    if args.iter().any(|a| a == "--emulated") {
        // the Ozaki scheme with 7 slices (55 bits), as linus's precision.fp64 = "emulated"
        h.set_emulation(&gpu, Some(7))?;
    }
    println!(
        "{name} {bname}/{aname}: nbf {n}, naux {naux}, {} atoms, {} kept pairs, {} aux blocks; setup {:.2} s",
        h.plan().natom,
        h.plan().layout().pairs.len(),
        h.plan().aux_blocks.len(),
        t.elapsed().as_secs_f64()
    );
    let (chans, f): (Vec<usize>, Vec<f64>) = if open {
        (vec![nel.div_ceil(2), nel / 2], vec![1.0, 1.0])
    } else {
        (vec![nel / 2], vec![2.0])
    };
    let cs: Vec<Vec<f64>> = chans
        .iter()
        .enumerate()
        .map(|(s, &o)| orbitals(n, o, 7 + s as u64))
        .collect();
    let occ: Vec<(&[f64], usize)> = cs.iter().zip(&chans).map(|(c, &o)| (&c[..], o)).collect();
    // a global hybrid's call (J and a_x K), or a range-separated hybrid's polarization call
    let (ax, want_j) = if omega.is_some() {
        (0.25, false)
    } else {
        (a_x, true)
    };
    for rep in 0..reps {
        let t = Instant::now();
        let p = h.compute(&gpu, &occ, &f, ax, want_j, &op)?;
        if let Some(path) = arg(&args, "--dump") {
            let mut v: Vec<f64> = p.hess.clone();
            for x in &p.fxo {
                v.extend_from_slice(x);
            }
            let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
            std::fs::write(&path, bytes).expect("dump");
        }
        let l = &h.last;
        println!(
            "call {rep}: {:.3} s (mo {:.3} fit {:.3} deriv {:.3} pass1 {:.3} solve {:.3} rz {:.3} t2 {:.3} \
             2nd {:.3} 2nd_2c {:.3} host {:.3}); |H| {:.3e}",
            t.elapsed().as_secs_f64(),
            l.mo,
            l.fit,
            l.deriv,
            l.pass1,
            l.solve,
            l.rz,
            l.t2,
            l.d2,
            l.d2_2c,
            l.host,
            p.hess.iter().fold(0.0f64, |m, x| m.max(x.abs()))
        );
    }
    Ok(())
}
