//! The device response engines against dense CPU references built here from linus-ints'
//! integrals and collocation and linus-xc (an independent fit, contraction and quadrature):
//! the projected J/K of factored first-order densities (closed and open shells, global and
//! range-separated exchange, both storage modes, perturbation groups and small blocks,
//! emulated exchange GEMMs), and the XC kernel contraction (LDA, GGA, meta-GGA, closed and
//! open shells), the latter also against finite differences of the device XC potential.
//! Every product is checked bitwise run to run.
#![cfg(feature = "gpu")]

mod common;

use common::Rng;
use linus_gpu::cpu::{
    eval_ao, metric, pair_block, Aux3c, BasisSet, Functional, GridSpec, Operator, Work3c, XcInput,
    XcOutput,
};
use linus_gpu::df::{DfLayout, DfStorage, Factor, GpuDfJk, GpuDfJkOptions, ROW_OFFDIAG};
use linus_gpu::harness::{self, compare, Structure};
use linus_gpu::response::df::{GpuDfResponse, GpuDfResponseOptions, KOperator, RespMath};
use linus_gpu::response::xc::{kernel_weights, GpuXcKernel, GpuXcKernelOptions};
use linus_gpu::xc::{GpuXc, GpuXcOptions, XcChannel, XcKind, XcMath, XcScreen};

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

/// The first `n` atoms of u1an39 (C, F, N, O, H).
fn fragment(n: usize) -> Structure {
    let mut m = harness::structure("u1an39").unwrap();
    m.z.truncate(n);
    m.positions.truncate(n);
    m
}

/// Random n x m column-major values in (-scale/2, scale/2).
fn random(n: usize, m: usize, scale: f64, seed: u64) -> Vec<f64> {
    let mut rng = Rng(seed);
    (0..n * m).map(|_| (rng.uniform() - 0.5) * scale).collect()
}

/// C = A B (column-major; A r x k, B k x c).
fn mm(a: &[f64], b: &[f64], r: usize, k: usize, c: usize) -> Vec<f64> {
    let mut o = vec![0.0; r * c];
    for j in 0..c {
        for l in 0..k {
            let bl = b[j * k + l];
            if bl != 0.0 {
                for i in 0..r {
                    o[j * r + i] += a[l * r + i] * bl;
                }
            }
        }
    }
    o
}

/// A^T (r x c column-major).
fn tr(a: &[f64], r: usize, c: usize) -> Vec<f64> {
    let mut o = vec![0.0; r * c];
    for j in 0..c {
        for i in 0..r {
            o[i * c + j] = a[j * r + i];
        }
    }
    o
}

/// The fitted planes L^-1 (P|op|mn) (Coulomb metric), naux dense n x n matrices, on the rows
/// that the device engine keeps.
fn planes(basis: &BasisSet, aux: &BasisSet, op: Operator) -> Vec<Vec<f64>> {
    let aux3 = Aux3c::new(aux);
    let chunks = aux3.chunks(0..aux.shells.len());
    let lay = DfLayout::new(basis, &aux3, 1e-14);
    let (n, naux, nx) = (lay.nbf, lay.naux, lay.nx);
    let mut a = vec![0.0; nx * naux];
    let mut w = Work3c::new();
    let mut off = 0;
    for p in &lay.pairs {
        let slab = &mut a[off..off + p.nab() * naux];
        pair_block(p, &chunks, slab, naux, 1e-16, &op, &mut w);
        off += p.nab() * naux;
    }
    let v = metric(&aux3, 0.0, &Operator::Coulomb);
    let mut l = vec![0.0; naux * naux];
    for i in 0..naux {
        for j in 0..=i {
            let mut s = v[i * naux + j];
            for k in 0..j {
                s -= l[i * naux + k] * l[j * naux + k];
            }
            l[i * naux + j] = if i == j {
                s.sqrt()
            } else {
                s / l[j * naux + j]
            };
        }
    }
    for x in 0..nx {
        let r = &mut a[x * naux..(x + 1) * naux];
        for i in 0..naux {
            let mut s = r[i];
            for k in 0..i {
                s -= l[i * naux + k] * r[k];
            }
            r[i] = s / l[i * naux + i];
        }
    }
    (0..naux)
        .map(|p| {
            let mut b = vec![0.0; n * n];
            for x in 0..nx {
                let (m, k) = (lay.row_m[x] as usize, lay.row_n[x] as usize);
                b[k * n + m] = a[x * naux + p];
                if lay.row_kind[x] == ROW_OFFDIAG {
                    b[m * n + k] = a[x * naux + p];
                }
            }
            b
        })
        .collect()
}

/// G_{s,d} = (c_j J[sum_s dD_s] - c_k K[dD_s]) C_s densely (wK with the operator planes `w`).
#[allow(clippy::too_many_arguments)]
fn reference_jk(
    b: &[Vec<f64>],
    w: Option<&[Vec<f64>]>,
    n: usize,
    c: &[Vec<f64>],
    occ: &[usize],
    a: &[Vec<f64>],
    nd: usize,
    c_j: f64,
    c_k: f64,
) -> Vec<Vec<f64>> {
    let nch = c.len();
    let dd = |s: usize, d: usize| -> Vec<f64> {
        let o = occ[s];
        let ad = &a[s][d * n * o..(d + 1) * n * o];
        let x = mm(ad, &tr(&c[s], n, o), n, o, n);
        let xt = tr(&x, n, n);
        x.iter().zip(&xt).map(|(p, q)| p + q).collect()
    };
    let mut out: Vec<Vec<f64>> = (0..nch).map(|s| vec![0.0; n * nd * occ[s]]).collect();
    for d in 0..nd {
        let dens: Vec<Vec<f64>> = (0..nch).map(|s| dd(s, d)).collect();
        let mut jm = vec![0.0; n * n];
        for bp in b {
            let g: f64 = (0..nch)
                .map(|s| bp.iter().zip(&dens[s]).map(|(x, y)| x * y).sum::<f64>())
                .sum();
            jm.iter_mut().zip(bp).for_each(|(j, x)| *j += g * x);
        }
        for s in 0..nch {
            let o = occ[s];
            let mut km = vec![0.0; n * n];
            for (p, bp) in b.iter().enumerate() {
                let t = mm(bp, &mm(&dens[s], bp, n, n, n), n, n, n);
                match w {
                    None => km.iter_mut().zip(&t).for_each(|(k, x)| *k += x),
                    Some(w) => {
                        let u = mm(bp, &mm(&dens[s], &w[p], n, n, n), n, n, n);
                        let v = mm(&w[p], &mm(&dens[s], bp, n, n, n), n, n, n);
                        km.iter_mut()
                            .zip(u.iter().zip(&v))
                            .for_each(|(k, (x, y))| *k += 0.5 * (x + y));
                    }
                }
            }
            let f: Vec<f64> = jm.iter().zip(&km).map(|(j, k)| c_j * j - c_k * k).collect();
            let g = mm(&f, &c[s], n, n, o);
            out[s][d * n * o..(d + 1) * n * o].copy_from_slice(&g);
        }
    }
    out
}

fn bitwise(a: &[Vec<f64>], b: &[Vec<f64>]) -> bool {
    a.iter()
        .zip(b)
        .all(|(x, y)| x.iter().zip(y).all(|(p, q)| p.to_bits() == q.to_bits()))
}

/// The response J/K of `occ` channels and `nd` perturbations against the dense reference, in
/// several engine configurations.
fn check_jk(mol: &Structure, basis_name: &str, occ: &[usize], nd: usize, xop: Option<Operator>) {
    let Some(gpu) = common::gpu() else { return };
    let basis = harness::basis(mol, basis_name).unwrap();
    let aux = harness::basis(mol, "def2-universal-jkfit").unwrap();
    let n = basis.nbf;
    let c: Vec<Vec<f64>> = occ
        .iter()
        .enumerate()
        .map(|(s, &o)| random(n, o, 0.6, 3 + s as u64))
        .collect();
    let a: Vec<Vec<f64>> = occ
        .iter()
        .enumerate()
        .map(|(s, &o)| random(n, nd * o, 0.4, 17 + s as u64))
        .collect();
    let b = planes(&basis, &aux, Operator::Coulomb);
    let w = xop.map(|op| planes(&basis, &aux, op));
    let (c_j, c_k) = (
        if occ.len() == 1 { 2.0 } else { 1.0 },
        if xop.is_some() { 1.0 } else { 0.25 },
    );
    let fs: Vec<Factor> = c
        .iter()
        .zip(occ)
        .map(|(c, &o)| Factor { c, ncol: o })
        .collect();
    let ar: Vec<&[f64]> = a.iter().map(|x| x.as_slice()).collect();
    let mut first: Option<Vec<Vec<f64>>> = None;
    let storages: &[DfStorage] = if xop.is_some() {
        &[DfStorage::Device]
    } else {
        &[DfStorage::Device, DfStorage::Host]
    };
    use KOperator::{Off, On};
    // the products on the operator stored on the device, per exchange arithmetic: the same
    // operator streamed from host memory must give the same bits
    let mut on_device: Vec<(RespMath, Vec<Vec<f64>>)> = Vec::new();
    for &storage in storages {
        // (exchange GEMMs, workspace bytes, block, stored operator, its fraction on the
        // device): through the tensor, the default plan; a tiny workspace (groups of one
        // perturbation, small blocks); a block size that does not divide naux; then on the
        // stored operator (device storage), on the device, split with host memory, and all
        // in host memory
        for (k_math, workspace, k_block, k_operator, on_dev) in [
            (RespMath::Native, 1usize << 30, 0, Off, None),
            (RespMath::Native, 1, 0, Off, None),
            (RespMath::Native, 1 << 30, 7, Off, None),
            (RespMath::Emulated { bits: 55 }, 1 << 30, 0, Off, None),
            (RespMath::Single, 1 << 30, 0, Off, None),
            (RespMath::Single, 1, 5, Off, None),
            // Ozaki GEMM1 on the pre-sliced planes: the default plan, groups of one, odd blocks
            (RespMath::Ozaki { slices: 7 }, 1 << 30, 0, Off, None),
            (RespMath::Ozaki { slices: 7 }, 1, 0, Off, None),
            (RespMath::Ozaki { slices: 7 }, 1 << 30, 7, Off, None),
            (RespMath::Native, 1 << 30, 0, On, None),
            (RespMath::Native, 1 << 30, 0, On, Some(0.6)),
            (RespMath::Native, 1 << 30, 0, On, Some(0.0)),
            (RespMath::Emulated { bits: 55 }, 1 << 30, 0, On, None),
            (RespMath::Emulated { bits: 55 }, 1 << 30, 0, On, Some(0.3)),
            (RespMath::Ozaki { slices: 7 }, 1 << 30, 0, On, None),
            // the FP32 operator (with the FP32 exchange arithmetic)
            (RespMath::Single, 1 << 30, 0, On, None),
            (RespMath::Single, 1 << 30, 0, On, Some(0.5)),
        ] {
            if k_operator == On && storage != DfStorage::Device {
                continue;
            }
            let dopts = GpuDfJkOptions {
                storage,
                k_block: if storage == DfStorage::Host { 40 } else { 0 },
                ..Default::default()
            };
            let jk = GpuDfJk::new(&gpu, &basis, &aux, &dopts).unwrap();
            let ropts = GpuDfResponseOptions {
                k_math,
                workspace,
                k_block,
                k_operator,
                k_operator_fp32: k_math == RespMath::Single,
                ..Default::default()
            };
            let mut eng = GpuDfResponse::new(&gpu, jk, &basis, &aux, &fs, xop, &ropts).unwrap();
            if let Some(f) = on_dev {
                let bytes = eng.k_operator_plan(&gpu, nd).bytes;
                eng.set_k_operator_storage(Some((bytes as f64 * f) as usize), bytes);
            }
            let tag = format!(
                "{} {basis_name} occ {occ:?} nd {nd} {xop:?} {storage:?} {k_math:?} ws {workspace} block {k_block} {k_operator:?} on device {on_dev:?}",
                harness::formula(mol)
            );
            let mut cases = vec![(c_j, c_k), (c_j, 0.0), (0.0, c_k)];
            if workspace != 1 << 30 || k_block != 0 || storage != DfStorage::Device {
                cases.truncate(1);
            }
            for (cj, ck) in cases {
                let g = eng.fock_occ(&gpu, &ar, nd, cj, ck).unwrap();
                let again = eng.fock_occ(&gpu, &ar, nd, cj, ck).unwrap();
                assert!(bitwise(&g, &again), "{tag}: not bitwise reproducible");
                // the stored operator is built by the first exchange of `On`
                if ck != 0.0 {
                    let r = eng.resolved();
                    assert_eq!(
                        r.k_operator_bytes > 0,
                        k_operator == On,
                        "{tag}: stored operator"
                    );
                    assert_eq!(
                        r.k_operator_host_bytes > 0,
                        on_dev.is_some(),
                        "{tag}: host part"
                    );
                    if on_dev == Some(0.0) {
                        assert_eq!(
                            r.k_operator_host_bytes, r.k_operator_bytes,
                            "{tag}: all host"
                        );
                    }
                }
                // the operator streamed from host memory: bitwise its products on the device
                if k_operator == On && (cj, ck) == (c_j, c_k) {
                    match (on_dev, on_device.iter().find(|x| x.0 == k_math)) {
                        (None, _) => on_device.push((k_math, g.clone())),
                        (Some(_), Some((_, g0))) => {
                            assert!(bitwise(&g, g0), "{tag}: not bitwise the device operator's")
                        }
                        (Some(_), None) => panic!("{tag}: no device run before"),
                    }
                }
                let r = reference_jk(&b, w.as_deref(), n, &c, occ, &a, nd, cj, ck);
                // FP32 exchange: single-precision round-off of K C (J stays FP64)
                let tol = match k_math {
                    RespMath::Single => 1e-6,
                    RespMath::Emulated { .. } | RespMath::Ozaki { .. } => 1e-11,
                    RespMath::Native => 2e-12,
                };
                for s in 0..occ.len() {
                    let st = compare(&g[s], &r[s]);
                    eprintln!(
                        "{tag} c_j {cj} c_k {ck} ch {s}: {st} (group {}, block {})",
                        eng.last.k_group, eng.last.k_block
                    );
                    st.check(0.0, tol).unwrap();
                }
                if (cj, ck) == (c_j, c_k) {
                    match &first {
                        None => first = Some(g),
                        // the configurations agree to round-off
                        Some(f) => {
                            for (x, y) in g.iter().zip(f) {
                                compare(x, y).check(0.0, tol).unwrap();
                            }
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn jk_closed_shell_global_hybrid() {
    check_jk(&water(), "def2-svp", &[5], 4, None);
}

#[test]
fn jk_open_shell_global_hybrid_tzvp() {
    // f functions in the basis, g in the auxiliary basis
    check_jk(&fragment(4), "def2-tzvp", &[9, 7], 3, None);
}

#[test]
fn jk_range_separated_closed_and_open() {
    // omega-B97X-like operator: (alpha + beta)/r - beta erf(omega r)/r
    let op = Operator::from_libxc(0.157706, 0.842294, 0.3);
    check_jk(&water(), "def2-svp", &[5], 3, Some(op));
    check_jk(&fragment(3), "def2-svp", &[6, 5], 2, Some(op));
}

/// dV_s C_s of the factors by dense quadrature: every function at every point, the ground
/// state of C, linus-xc's second derivatives and the weights of [`kernel_weights`].
fn reference_kernel(
    basis: &BasisSet,
    grid: &linus_gpu::cpu::MolecularGrid,
    name: &str,
    c: &[Vec<f64>],
    occ: &[usize],
    a: &[Vec<f64>],
    nd: usize,
) -> Vec<Vec<f64>> {
    let n = basis.nbf;
    let np = grid.x.len();
    let nsp = c.len();
    let f = Functional::new(name, nsp == 2).unwrap();
    let kind = XcKind::of(&f);
    let (gga, mgga) = (kind.gga, kind.mgga);
    let ns = if mgga {
        5
    } else if gga {
        4
    } else {
        1
    };
    let nu = nsp * ns;
    let shells: Vec<usize> = (0..basis.shells.len()).collect();
    let mut ao = vec![0.0; 4 * n * np];
    eval_ao(basis, &shells, &grid.x, &grid.y, &grid.z, 1, &mut ao);
    let phi = |k: usize, m: usize, g: usize| ao[(k * n + m) * np + g];
    // orbitals on the grid: orb(s, i, k, g)
    let on = |x: &[f64], o: usize| -> Vec<f64> {
        let mut v = vec![0.0; o * 4 * np];
        for i in 0..o {
            for k in 0..4 {
                for g in 0..np {
                    v[(i * 4 + k) * np + g] = (0..n).map(|m| phi(k, m, g) * x[i * n + m]).sum();
                }
            }
        }
        v
    };
    let orb: Vec<Vec<f64>> = (0..nsp).map(|s| on(&c[s], occ[s])).collect();
    let scale = if nsp == 1 { 2.0 } else { 1.0 };
    let (mut rin, mut sin, mut tin) = (vec![0.0; np * nsp], vec![], vec![0.0; np * nsp]);
    let mut grad = vec![0.0; nsp * 3 * np];
    for s in 0..nsp {
        for g in 0..np {
            for i in 0..occ[s] {
                let p = |k: usize| orb[s][(i * 4 + k) * np + g];
                rin[g * nsp + s] += scale * p(0) * p(0);
                for k in 0..3 {
                    grad[(s * 3 + k) * np + g] += scale * 2.0 * p(0) * p(1 + k);
                    tin[g * nsp + s] += scale * 0.5 * p(1 + k) * p(1 + k);
                }
            }
        }
    }
    for g in 0..np {
        let gr = |s: usize, k: usize| grad[(s * 3 + k) * np + g];
        let dot = |x: usize, y: usize| (0..3).map(|k| gr(x, k) * gr(y, k)).sum::<f64>();
        if nsp == 1 {
            sin.push(dot(0, 0));
        } else {
            sin.extend([dot(0, 0), dot(0, 1), dot(1, 1)]);
        }
    }
    let input = if mgga {
        XcInput::mgga(&rin, &sin, &tin)
    } else if gga {
        XcInput::gga(&rin, &sin)
    } else {
        XcInput::lda(&rin)
    };
    let mut out = XcOutput::new();
    f.eval(2, &input, &mut out).unwrap();
    let wf = kernel_weights(&out, nsp, gga, mgga, &grid.w, &grad, np);
    let mut res: Vec<Vec<f64>> = (0..nsp).map(|s| vec![0.0; n * nd * occ[s]]).collect();
    for d in 0..nd {
        let pert: Vec<Vec<f64>> = (0..nsp)
            .map(|s| on(&a[s][d * n * occ[s]..(d + 1) * n * occ[s]], occ[s]))
            .collect();
        for g in 0..np {
            let mut u = vec![0.0; nu];
            for s in 0..nsp {
                for i in 0..occ[s] {
                    let p = |k: usize| pert[s][(i * 4 + k) * np + g];
                    let f = |k: usize| orb[s][(i * 4 + k) * np + g];
                    u[s * ns] += 2.0 * scale * p(0) * f(0);
                    if gga {
                        for k in 0..3 {
                            u[s * ns + 1 + k] += 2.0 * scale * (p(1 + k) * f(0) + p(0) * f(1 + k));
                        }
                    }
                    if mgga {
                        for k in 0..3 {
                            u[s * ns + 4] += scale * p(1 + k) * f(1 + k);
                        }
                    }
                }
            }
            let t: Vec<f64> = (0..nu)
                .map(|cc| (0..nu).map(|e| wf[(cc * nu + e) * np + g] * u[e]).sum())
                .collect();
            for s in 0..nsp {
                let ts = &t[s * ns..(s + 1) * ns];
                let o = occ[s];
                for j in 0..o {
                    let f = |k: usize| orb[s][(j * 4 + k) * np + g];
                    for m in 0..n {
                        let mut v = ts[0] * phi(0, m, g) * f(0);
                        if gga {
                            for k in 0..3 {
                                v +=
                                    ts[1 + k] * (phi(1 + k, m, g) * f(0) + phi(0, m, g) * f(1 + k));
                            }
                        }
                        if mgga {
                            for k in 0..3 {
                                v += 0.5 * ts[4] * phi(1 + k, m, g) * f(1 + k);
                            }
                        }
                        res[s][(d * o + j) * n + m] += v;
                    }
                }
            }
        }
    }
    res
}

/// The XC kernel of `name` against the dense quadrature and, at `fd_tol`, against central
/// differences of the device XC potential.
fn check_kernel(
    mol: &Structure,
    basis_name: &str,
    name: &str,
    occ: &[usize],
    nd: usize,
    block_points: usize,
    fd_tol: f64,
) {
    let Some(gpu) = common::gpu() else { return };
    let basis = harness::basis(mol, basis_name).unwrap();
    let grid = harness::grid(mol, &GridSpec::pyscf(1)).unwrap();
    let n = basis.nbf;
    let nsp = occ.len();
    let c: Vec<Vec<f64>> = occ
        .iter()
        .enumerate()
        .map(|(s, &o)| random(n, o, 0.6, 5 + s as u64))
        .collect();
    let a: Vec<Vec<f64>> = occ
        .iter()
        .enumerate()
        .map(|(s, &o)| random(n, nd * o, 0.3, 29 + s as u64))
        .collect();
    let xopts = GpuXcOptions {
        func_thresh: 1e-15,
        block_points,
        ..Default::default()
    };
    let fs: Vec<Factor> = c
        .iter()
        .zip(occ)
        .map(|(c, &o)| Factor { c, ncol: o })
        .collect();
    let ar: Vec<&[f64]> = a.iter().map(|x| x.as_slice()).collect();
    let r = reference_kernel(&basis, &grid, name, &c, occ, &a, nd);
    let tag = format!(
        "{} {basis_name} {name} occ {occ:?} nd {nd} block {block_points}",
        harness::formula(mol)
    );
    let mut first: Option<Vec<Vec<f64>>> = None;
    // (screening, workspace, emulated GEMM bits, Ozaki slices, arithmetic, kept ground state,
    // GGA Q route): per batch; groups of one perturbation; block screening (linus's FP64
    // default) with and without fixed-mantissa emulation or the Ozaki scheme on the large
    // GEMMs, the ground state kept across calls (bitwise the same products), and P = F A of
    // every component for GGAs; per batch in FP32 (the precision study, its own tolerance)
    let keep = Some(0);
    for (screen, workspace, emulated_bits, ozaki_slices, math, ground_cache, q_route) in [
        (
            XcScreen::Batch,
            1usize << 30,
            None,
            None,
            XcMath::Fp64,
            None,
            true,
        ),
        (XcScreen::Batch, 1, None, None, XcMath::Fp64, None, true),
        (
            XcScreen::Block,
            1 << 30,
            None,
            None,
            XcMath::Fp64,
            None,
            true,
        ),
        (
            XcScreen::Block,
            1 << 30,
            None,
            None,
            XcMath::Fp64,
            keep,
            true,
        ),
        (XcScreen::Block, 1, None, None, XcMath::Fp64, keep, true),
        (
            XcScreen::Block,
            1 << 30,
            None,
            None,
            XcMath::Fp64,
            keep,
            false,
        ),
        (
            XcScreen::Block,
            1 << 30,
            Some(55),
            None,
            XcMath::Fp64,
            None,
            true,
        ),
        (
            XcScreen::Block,
            1 << 30,
            None,
            Some(7),
            XcMath::Fp64,
            None,
            true,
        ),
        (XcScreen::Block, 1, None, Some(7), XcMath::Fp64, keep, true),
        (
            XcScreen::Batch,
            1 << 30,
            None,
            None,
            XcMath::Fp32,
            None,
            true,
        ),
    ] {
        let xo = GpuXcOptions {
            screen,
            math,
            ..xopts.clone()
        };
        let xc = GpuXc::new(&gpu, &basis, &grid, name, &xo).unwrap();
        let kopts = GpuXcKernelOptions {
            workspace,
            emulated_bits,
            ozaki_slices,
            math,
            ground_cache,
            q_route,
        };
        let mut eng = GpuXcKernel::new(&gpu, xc, &fs, &kopts).unwrap();
        let v = eng.contract(&gpu, &ar, nd).unwrap();
        let again = eng.contract(&gpu, &ar, nd).unwrap();
        assert!(bitwise(&v, &again), "{tag}: not bitwise reproducible");
        if ground_cache.is_some() {
            // made by the first call, read by the second: the products of a kernel without it
            assert!(eng.ground_cache_bytes() > 0, "{tag}: no kept ground state");
            let xc = GpuXc::new(&gpu, &basis, &grid, name, &xo).unwrap();
            let plain = GpuXcKernelOptions {
                ground_cache: None,
                ..kopts.clone()
            };
            let w = GpuXcKernel::new(&gpu, xc, &fs, &plain)
                .unwrap()
                .contract(&gpu, &ar, nd)
                .unwrap();
            assert!(
                bitwise(&v, &w),
                "{tag}: the kept ground state changes the products"
            );
        }
        let single = math == XcMath::Fp32;
        for s in 0..nsp {
            let st = compare(&v[s], &r[s]);
            eprintln!(
                "{tag} {screen:?} {math:?} ws {workspace} emulated {emulated_bits:?} ozaki {ozaki_slices:?} (group {}) ch {s}: {st}",
                eng.last.group
            );
            st.check(0.0, if single { 1e-5 } else { 1e-11 }).unwrap();
        }
        if single {
            continue;
        }
        match &first {
            None => first = Some(v),
            Some(f) => {
                let tol = if emulated_bits.is_some() || ozaki_slices.is_some() {
                    1e-12
                } else {
                    1e-13
                };
                for (x, y) in v.iter().zip(f) {
                    compare(x, y).check(0.0, tol).unwrap();
                }
            }
        }
    }
    // central differences of the device potential: D_s +- h dD_s through the channel
    // orbitals [C + h A | A] with occupations [1, -h^2] (a closed shell's total density is
    // twice its channel, as the kernel's)
    let v = first.unwrap();
    let mut xc = GpuXc::new(&gpu, &basis, &grid, name, &xopts).unwrap();
    let h = 1e-4;
    for d in 0..nd {
        let pot = |xc: &mut GpuXc, sign: f64| -> Vec<Vec<f64>> {
            let chans: Vec<(Vec<f64>, Vec<f64>)> = (0..nsp)
                .map(|s| {
                    let o = occ[s];
                    let ad = &a[s][d * n * o..(d + 1) * n * o];
                    let mut cc: Vec<f64> =
                        c[s].iter().zip(ad).map(|(x, y)| x + sign * h * y).collect();
                    cc.extend_from_slice(ad);
                    let mut oc = vec![1.0; o];
                    oc.extend(vec![-h * h; o]);
                    (cc, oc)
                })
                .collect();
            let xch: Vec<XcChannel> = chans
                .iter()
                .zip(occ)
                .map(|((cc, oc), &o)| XcChannel {
                    c: cc,
                    ncol: 2 * o,
                    occ: oc,
                })
                .collect();
            xc.compute(&gpu, &xch, nsp == 1).unwrap().v
        };
        let (vp, vm) = (pot(&mut xc, 1.0), pot(&mut xc, -1.0));
        for s in 0..nsp {
            let o = occ[s];
            let dv: Vec<f64> = vp[s]
                .iter()
                .zip(&vm[s])
                .map(|(p, q)| (p - q) / (2.0 * h))
                .collect();
            let fd = mm(&dv, &c[s], n, n, o);
            let st = compare(&v[s][d * n * o..(d + 1) * n * o], &fd);
            eprintln!("{tag} pert {d} ch {s} vs finite differences (h {h}): {st}");
            st.check(0.0, fd_tol).unwrap();
        }
    }
}

#[test]
fn kernel_lda_closed_shell() {
    check_kernel(&water(), "def2-svp", "svwn", &[5], 3, 512, 1e-6);
}

#[test]
fn kernel_gga_open_shell() {
    check_kernel(&water(), "def2-svp", "pbe", &[5, 4], 3, 1024, 1e-6);
}

#[test]
fn kernel_hybrid_gga_closed_shell_tzvp() {
    check_kernel(&water(), "def2-tzvp", "b3lyp", &[5], 2, 4096, 1e-6);
}

#[test]
fn kernel_meta_gga_closed_and_open() {
    check_kernel(&water(), "def2-svp", "tpss", &[5], 2, 777, 1e-5);
    check_kernel(&water(), "def2-svp", "r2scan", &[5, 4], 2, 2048, 1e-5);
}
