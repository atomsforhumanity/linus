//! The device XC engine against a dense CPU quadrature built from linus-ints' collocation and
//! linus-xc (every function at every point, no screening), closed and open shells, LDA, GGA
//! and meta-GGA, per-batch and block screening, run to run; the FP32 arithmetic of the
//! precision study at its own (loose) tolerance.
#![cfg(feature = "gpu")]

mod common;

use common::Rng;
use linus_gpu::cpu::{eval_ao, Functional, GridSpec, XcInput, XcOutput};
use linus_gpu::harness::{self, compare, Structure};
use linus_gpu::xc::{GpuXc, GpuXcOptions, XcChannel, XcKind, XcMath};

/// (E_xc, V per spin) by direct quadrature.
fn reference(
    mol: &Structure,
    basis_name: &str,
    grid: &linus_gpu::cpu::MolecularGrid,
    name: &str,
    ch: &[(Vec<f64>, usize, Vec<f64>)],
    restricted: bool,
) -> (f64, Vec<Vec<f64>>) {
    let basis = harness::basis(mol, basis_name).unwrap();
    let nbf = basis.nbf;
    let np = grid.x.len();
    let f = Functional::new(name, !restricted).unwrap();
    let kind = XcKind::of(&f);
    let shells: Vec<usize> = (0..basis.shells.len()).collect();
    let mut ao = vec![0.0; 4 * nbf * np];
    eval_ao(&basis, &shells, &grid.x, &grid.y, &grid.z, 1, &mut ao);
    let phi = |d: usize, m: usize, g: usize| ao[(d * nbf + m) * np + g];
    let nsp = ch.len();
    // spin densities (closed shell: the total density, twice the channel)
    let scale = if restricted { 2.0 } else { 1.0 };
    let mut rho = vec![[0.0; 2]; np];
    let mut grad = vec![[[0.0; 3]; 2]; np];
    let mut tau = vec![[0.0; 2]; np];
    for (s, (c, nc, occ)) in ch.iter().enumerate() {
        for g in 0..np {
            for i in 0..*nc {
                let psi: Vec<f64> = (0..4)
                    .map(|d| (0..nbf).map(|m| phi(d, m, g) * c[i * nbf + m]).sum())
                    .collect();
                rho[g][s] += scale * occ[i] * psi[0] * psi[0];
                for k in 0..3 {
                    grad[g][s][k] += scale * 2.0 * occ[i] * psi[0] * psi[1 + k];
                    tau[g][s] += scale * 0.5 * occ[i] * psi[1 + k] * psi[1 + k];
                }
            }
        }
    }
    let dot = |a: &[f64; 3], b: &[f64; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    let (mut rin, mut sin, mut tin) = (vec![], vec![], vec![]);
    for g in 0..np {
        for s in 0..nsp {
            rin.push(rho[g][s]);
            tin.push(tau[g][s]);
        }
        if nsp == 1 {
            sin.push(dot(&grad[g][0], &grad[g][0]));
        } else {
            sin.extend([
                dot(&grad[g][0], &grad[g][0]),
                dot(&grad[g][0], &grad[g][1]),
                dot(&grad[g][1], &grad[g][1]),
            ]);
        }
    }
    let input = if kind.mgga {
        XcInput::mgga(&rin, &sin, &tin)
    } else if kind.gga {
        XcInput::gga(&rin, &sin)
    } else {
        XcInput::lda(&rin)
    };
    let mut out = XcOutput::new();
    f.eval(1, &input, &mut out).unwrap();
    let (zk, vr, vs, vt) = (out.zk(), out.vrho(), out.vsigma(), out.vtau());
    let mut exc = 0.0;
    for g in 0..np {
        let rt: f64 = (0..nsp).map(|s| rho[g][s]).sum();
        exc += grid.w[g] * rt * zk[g];
    }
    let v = (0..nsp)
        .map(|s| {
            let mut vm = vec![0.0; nbf * nbf];
            for g in 0..np {
                let w = grid.w[g];
                let vrho = vr[g * nsp + s];
                let mut vg = [0.0; 3];
                if kind.gga {
                    for k in 0..3 {
                        vg[k] = if nsp == 1 {
                            2.0 * vs[g] * grad[g][0][k]
                        } else {
                            2.0 * vs[3 * g + 2 * s] * grad[g][s][k]
                                + vs[3 * g + 1] * grad[g][1 - s][k]
                        };
                    }
                }
                let vtau = if kind.mgga { vt[g * nsp + s] } else { 0.0 };
                for n in 0..nbf {
                    for m in 0..nbf {
                        let (pm, pn) = (phi(0, m, g), phi(0, n, g));
                        let mut x = vrho * pm * pn;
                        for (k, vgk) in vg.iter().enumerate() {
                            x += vgk * (phi(1 + k, m, g) * pn + pm * phi(1 + k, n, g));
                            x += 0.5 * vtau * phi(1 + k, m, g) * phi(1 + k, n, g);
                        }
                        vm[n * nbf + m] += w * x;
                    }
                }
            }
            vm
        })
        .collect();
    (exc, v)
}

fn water() -> Structure {
    let a = 1.0 / linus_core::units::BOHR_ANGSTROM;
    Structure::new(
        vec![8, 1, 1],
        vec![
            [0.0, 0.0, 0.1173 * a],
            [0.0, 0.7572 * a, -0.4692 * a],
            [0.0, -0.7572 * a, -0.4692 * a],
        ],
        0,
        1,
    )
    .unwrap()
}

/// Orbital-like factors: random C with occupations 1 (and one fractional).
fn channel(nbf: usize, nc: usize, seed: u64) -> (Vec<f64>, usize, Vec<f64>) {
    let mut rng = Rng(seed);
    let c = (0..nbf * nc).map(|_| (rng.uniform() - 0.5) * 0.6).collect();
    let mut occ = vec![1.0; nc];
    if nc > 1 {
        occ[nc - 1] = 0.5;
    }
    (c, nc, occ)
}

fn check(
    mol: &Structure,
    basis_name: &str,
    functional: &str,
    restricted: bool,
    block_points: usize,
) {
    let Some(gpu) = common::gpu() else { return };
    let basis = harness::basis(mol, basis_name).unwrap();
    let grid = harness::grid(mol, &GridSpec::pyscf(1)).unwrap();
    let nbf = basis.nbf;
    let ch: Vec<(Vec<f64>, usize, Vec<f64>)> = if restricted {
        vec![channel(nbf, 5, 3)]
    } else {
        vec![channel(nbf, 5, 3), channel(nbf, 4, 9)]
    };
    let (e_ref, v_ref) = reference(mol, basis_name, &grid, functional, &ch, restricted);
    let base = GpuXcOptions {
        func_thresh: 1e-15,
        block_points,
        ..GpuXcOptions::default()
    };
    // (settings, relative tolerance of E_xc and V)
    #[cfg_attr(not(feature = "study"), allow(unused_mut))]
    let mut runs = vec![
        (base.clone(), 1e-11),
        // a workspace of one block per chunk: several chunks
        (
            GpuXcOptions {
                workspace: 1,
                ..base.clone()
            },
            1e-11,
        ),
        (
            GpuXcOptions {
                math: XcMath::Fp32,
                ..base.clone()
            },
            1e-5,
        ),
    ];
    // the precision studies' variants: the block path by each density route, and FP32 for the
    // potential only
    #[cfg(feature = "study")]
    {
        use linus_gpu::xc::DensityRoute;
        for route in [
            DensityRoute::Orbitals,
            DensityRoute::DensityMatrix,
            DensityRoute::Auto,
        ] {
            let screen = linus_gpu::xc::XcScreen::Block;
            runs.push((
                GpuXcOptions {
                    screen,
                    route,
                    ..base.clone()
                },
                1e-11,
            ));
        }
        runs.push((
            GpuXcOptions {
                math: XcMath::Fp32Potential,
                ..base.clone()
            },
            1e-5,
        ));
    }
    for (opts, tol) in runs {
        let mut xc = GpuXc::new(&gpu, &basis, &grid, functional, &opts).unwrap();
        let chans: Vec<XcChannel> = ch
            .iter()
            .map(|(c, nc, occ)| XcChannel { c, ncol: *nc, occ })
            .collect();
        let r = xc.compute(&gpu, &chans, restricted).unwrap();
        let r2 = xc.compute(&gpu, &chans, restricted).unwrap();
        assert_eq!(r.exc.to_bits(), r2.exc.to_bits(), "E_xc not reproducible");
        for (a, b) in r.v.iter().zip(&r2.v) {
            assert!(
                a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits()),
                "V not reproducible"
            );
        }
        let tag = format!(
            "{} {basis_name} {functional} {} block {block_points} {:?} {:?} workspace {} ({} blocks, {:.0}% of values)",
            harness::formula(mol),
            if restricted { "R" } else { "U" },
            opts.screen,
            opts.math,
            opts.workspace,
            xc.stats().nblocks,
            100.0 * xc.stats().kept_fraction
        );
        eprintln!("{tag}: E_xc {:.12} diff {:.2e}", r.exc, r.exc - e_ref);
        assert!(
            (r.exc - e_ref).abs() < tol * e_ref.abs().max(1.0),
            "{tag}: E_xc"
        );
        for (s, (g, c)) in r.v.iter().zip(&v_ref).enumerate() {
            let st = compare(g, c);
            eprintln!("{tag} V[{s}]: {st}");
            st.check(0.0, tol).unwrap();
            for i in 0..nbf {
                for j in 0..i {
                    assert_eq!(g[j * nbf + i].to_bits(), g[i * nbf + j].to_bits());
                }
            }
        }
    }
}

#[test]
fn water_lda_restricted() {
    check(&water(), "def2-svp", "svwn", true, 512);
}

#[test]
fn water_gga_unrestricted() {
    check(&water(), "def2-svp", "pbe", false, 1024);
}

#[test]
fn water_hybrid_gga_restricted_tzvp() {
    check(&water(), "def2-tzvp", "b3lyp", true, 4096);
}

#[test]
fn water_meta_gga_both() {
    check(&water(), "def2-svp", "tpss", true, 777);
    check(&water(), "def2-svp", "r2scan", false, 2048);
}

#[test]
fn fragment_meta_gga_unrestricted_tzvp() {
    let mut m = harness::structure("u1an39").unwrap();
    m.z.truncate(4);
    m.positions.truncate(4);
    check(&m, "def2-tzvp", "tpss", false, 2048);
}

#[test]
fn device_entry_point_matches_the_host_one_bitwise() {
    let Some(gpu) = common::gpu() else { return };
    let mol = water();
    let basis = harness::basis(&mol, "def2-svp").unwrap();
    let grid = harness::grid(&mol, &GridSpec::pyscf(1)).unwrap();
    let nbf = basis.nbf;
    let (ca, na, _) = channel(nbf, 5, 3);
    let (cb, nb, _) = channel(nbf, 4, 9);
    let ones = |n: usize| vec![1.0; n];
    let mut xc = GpuXc::new(&gpu, &basis, &grid, "pbe", &GpuXcOptions::default()).unwrap();
    let (oa, ob) = (ones(na), ones(nb));
    let host = xc
        .compute(
            &gpu,
            &[
                XcChannel {
                    c: &ca,
                    ncol: na,
                    occ: &oa,
                },
                XcChannel {
                    c: &cb,
                    ncol: nb,
                    occ: &ob,
                },
            ],
            false,
        )
        .unwrap();
    let cmo = gpu.upload(&[ca.clone(), cb.clone()].concat()).unwrap();
    let mut v = gpu.upload(&vec![3.0; 2 * nbf * nbf]).unwrap();
    // occupations defaulted to 1 on the device
    let exc = xc
        .compute_device(&gpu, &cmo, &[na, nb], None, false, &mut v)
        .unwrap();
    let v = gpu.download(&v).unwrap();
    assert_eq!(exc.to_bits(), host.exc.to_bits());
    assert!(v
        .iter()
        .zip(host.v.concat())
        .all(|(x, y)| x.to_bits() == y.to_bits()));
}

/// The density entry point (a guess's densities, no orbitals) against the orbitals' on the
/// per-batch path: closed and open shells, a GGA and a meta-GGA, to round-off (D formed on the
/// host instead of the device); None on the block path.
#[test]
fn density_entry_point_matches_the_orbitals() {
    let Some(gpu) = common::gpu() else { return };
    let mol = water();
    let basis = harness::basis(&mol, "def2-svp").unwrap();
    let grid = harness::grid(&mol, &GridSpec::pyscf(1)).unwrap();
    let nbf = basis.nbf;
    let dm = |c: &[f64], nc: usize, occ: &[f64]| -> Vec<f64> {
        let mut d = vec![0.0; nbf * nbf];
        for j in 0..nbf {
            for i in 0..nbf {
                d[i + j * nbf] = (0..nc)
                    .map(|k| occ[k] * c[i + k * nbf] * c[j + k * nbf])
                    .sum();
            }
        }
        d
    };
    for (name, restricted) in [("pbe", true), ("pbe", false), ("r2scan", false)] {
        let mut xc = GpuXc::new(&gpu, &basis, &grid, name, &GpuXcOptions::default()).unwrap();
        let chans: Vec<(Vec<f64>, usize, Vec<f64>)> = if restricted {
            vec![channel(nbf, 5, 3)]
        } else {
            vec![channel(nbf, 5, 3), channel(nbf, 4, 9)]
        };
        let ch: Vec<XcChannel> = chans
            .iter()
            .map(|(c, ncol, occ)| XcChannel {
                c,
                ncol: *ncol,
                occ,
            })
            .collect();
        let orb = xc.compute(&gpu, &ch, restricted).unwrap();
        let ds: Vec<Vec<f64>> = chans.iter().map(|(c, nc, occ)| dm(c, *nc, occ)).collect();
        let refs: Vec<&[f64]> = ds.iter().map(|d| d.as_slice()).collect();
        let den = xc
            .compute_densities(&gpu, &refs, restricted)
            .unwrap()
            .unwrap();
        let de = (den.exc - orb.exc).abs();
        let vmax = orb.v.concat().iter().fold(0.0f64, |m, x| m.max(x.abs()));
        let dv = den
            .v
            .concat()
            .iter()
            .zip(orb.v.concat())
            .fold(0.0f64, |m, (a, b)| m.max((a - b).abs()));
        eprintln!("{name} restricted {restricted}: dE {de:.2e}, max |dV| {dv:.2e} of {vmax:.2e}");
        assert!(
            de < 1e-11 * orb.exc.abs().max(1.0),
            "{name} {restricted}: dE {de:e}"
        );
        assert!(dv < 1e-11 * vmax.max(1.0), "{name} {restricted}: dV {dv:e}");
    }
    let mut blocks = GpuXc::new(
        &gpu,
        &basis,
        &grid,
        "pbe",
        &GpuXcOptions {
            screen: linus_gpu::xc::XcScreen::Block,
            ..GpuXcOptions::default()
        },
    )
    .unwrap();
    let d = vec![0.0; nbf * nbf];
    assert!(blocks
        .compute_densities(&gpu, &[&d], true)
        .unwrap()
        .is_none());
}

/// The XC Hessian's accumulators sized to a small free memory (a cap): main passes over groups
/// of atoms instead of a CPU fallback, the call within that memory, and the same bits as one pass (the
/// accumulators were a fixed 24 GiB, so under a cap the whole call had to fit or fall back).
#[test]
fn xc_hessian_accumulators_fit_the_free_memory_bitwise() {
    use linus_gpu::hess::xc::{GpuXcHess, GpuXcHessOptions};
    let Some(gpu) = common::gpu() else { return };
    let mol = harness::structure("la18").unwrap();
    let natom = mol.z.len();
    let basis = harness::basis(&mol, "def2-svp").unwrap();
    let grid = harness::grid(&mol, &GridSpec::linus(2)).unwrap();
    let (n, o) = (basis.nbf, 5);
    let mut rng = Rng(7);
    let c: Vec<f64> = (0..n * o).map(|_| 0.3 * (rng.uniform() - 0.5)).collect();
    let mut d = vec![0.0; n * n];
    for j in 0..n {
        for i in 0..n {
            d[i + j * n] = 2.0 * (0..o).map(|k| c[i + k * n] * c[j + k * n]).sum::<f64>();
        }
    }
    let new = || {
        GpuXcHess::new(
            &gpu,
            &basis,
            &grid,
            "b3lyp",
            1,
            true,
            &GpuXcHessOptions::default(),
        )
        .unwrap()
    };
    let mut one = new();
    let r1 = one.compute(&gpu, &[&d], &[(&c, o)]).unwrap();
    assert_eq!(one.last.passes, 1);
    // room for five and a half atoms' accumulators beside the other workspaces: groups of 5
    let mut split = new();
    let all = split.workspace_bytes(&[o]);
    let per_atom = 2 * 3 * (n * (n + 1) / 2) * 24;
    let free = all - natom * per_atom + per_atom * 11 / 2;
    split.fit_accumulators(&[o], free);
    assert!(split.workspace_bytes(&[o]) <= free);
    let r3 = split.compute(&gpu, &[&d], &[(&c, o)]).unwrap();
    assert_eq!(split.last.passes, natom.div_ceil(5));
    let same = |p: &[f64], q: &[f64]| {
        p.len() == q.len() && p.iter().zip(q).all(|(x, y)| x.to_bits() == y.to_bits())
    };
    assert!(same(&r1.hfix, &r3.hfix) && same(&r1.hgrid, &r3.hgrid));
    assert!(same(&r1.vfix_c[0], &r3.vfix_c[0]) && same(&r1.vgrid_c[0], &r3.vgrid_c[0]));
    // with room for every atom (no cap) nothing changes
    let mut roomy = new();
    roomy.fit_accumulators(&[o], usize::MAX);
    assert_eq!(roomy.workspace_bytes(&[o]), all);
}
