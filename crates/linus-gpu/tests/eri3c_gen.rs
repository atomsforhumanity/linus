//! The generated three-center kernels (`linus_gpu::eri3c`) against linus-ints' CPU kernels:
//! every class (l_a l_b | l_k) with l_a, l_b <= 4 and l_k <= 6, integrals and first
//! derivatives, for 1/r and erf(omega r)/r, run-to-run determinism and pair ranges.
#![cfg(feature = "gpu")]

mod common;

use linus_gpu::cpu::{build_pair, BasisSet, Operator, Pair3c, ShellView};
use linus_gpu::eri3c::{all_pairs, Eri3c, Eri3cHost, Eri3cKind};
use linus_gpu::harness::{self, compare, Structure};
use linus_gpu::Gpu;

/// FeO with def2-QZVP (g shells on both atoms) and def2-universal-jkfit (up to i on Fe): every
/// class of both kinds occurs.
fn feo() -> (BasisSet, BasisSet) {
    let mol = Structure::new(vec![26, 8], vec![[0.0, 0.0, 0.0], [0.4, -0.3, 3.05]], 0, 1).unwrap();
    let basis = harness::basis(&mol, "def2-qzvp").unwrap();
    let aux = harness::basis(&mol, "def2-universal-jkfit").unwrap();
    assert_eq!((basis.max_l(), aux.max_l()), (4, 6));
    (basis, aux)
}

/// (class key, largest |gpu - cpu|, largest |cpu|) per class (la lb | lk) of the output.
fn per_class(
    host: &Eri3cHost,
    basis: &BasisSet,
    aux: &BasisSet,
    gpu_out: &[f64],
    cpu_out: &[f64],
) -> Vec<((usize, usize, usize), f64, f64)> {
    let naux = host.naux;
    let rows = host.rows(0..host.pairs.len());
    let plane = rows.len() * naux;
    let mut acc = std::collections::BTreeMap::new();
    for pr in &host.pairs {
        let (la, lb) = (basis.shells[pr.sa].l, basis.shells[pr.sb].l);
        let nab = basis.shells[pr.sa].nfunc() * basis.shells[pr.sb].nfunc();
        for s in &aux.shells {
            let key = (la.max(lb), la.min(lb), s.l);
            let e: &mut (f64, f64) = acc.entry(key).or_insert((0.0, 0.0));
            for p in 0..host.kind.planes() {
                for r in pr.row..pr.row + nab {
                    for c in s.ao_start..s.ao_start + s.nfunc() {
                        let i = p * plane + r * naux + c;
                        e.0 = f64::max(e.0, (gpu_out[i] - cpu_out[i]).abs());
                        e.1 = f64::max(e.1, cpu_out[i].abs());
                    }
                }
            }
        }
    }
    acc.into_iter().map(|(k, (d, m))| (k, d, m)).collect()
}

fn run(gpu: &Gpu, b: &Eri3c, range: std::ops::Range<usize>) -> Vec<f64> {
    run_op(gpu, b, range, &Operator::Coulomb)
}

fn run_op(gpu: &Gpu, b: &Eri3c, range: std::ops::Range<usize>, op: &Operator) -> Vec<f64> {
    let mut out = gpu
        .alloc_zeros::<f64>(b.host().out_len(range.clone()))
        .unwrap();
    b.compute(gpu, range, op, &mut out, 0).unwrap();
    gpu.download(&out).unwrap()
}

fn check_all_classes(kind: Eri3cKind, rel: f64, op: Operator) {
    let Some(gpu) = common::gpu() else { return };
    let (basis, aux) = feo();
    let pairs = all_pairs(&basis);
    let host = Eri3cHost::new(&basis, &aux, &pairs, kind, 0.0).unwrap();
    let cpu = host.cpu_reference(&basis, &aux, 0..pairs.len(), &op);
    let mut b = Eri3c::new(&gpu, host).unwrap();
    b.prepare_operator(&gpu, &op).unwrap();
    let g1 = run_op(&gpu, &b, 0..pairs.len(), &op);
    let g2 = run_op(&gpu, &b, 0..pairs.len(), &op);
    assert!(
        g1.iter().zip(&g2).all(|(x, y)| x.to_bits() == y.to_bits()),
        "{kind:?}: not deterministic"
    );
    let all = compare(&g1, &cpu);
    eprintln!("{kind:?} {op:?} all classes: {all}");
    let mut bad = Vec::new();
    let classes = per_class(b.host(), &basis, &aux, &g1, &cpu);
    for &((la, lb, lk), d, m) in &classes {
        let (gk, k) = b.kernel(la, lb, lk).unwrap();
        eprintln!(
            "  ({la}{lb}|{lk}) {:>14} regs {:>3} local {:>5} B: max |d| {d:.2e}, max |cpu| {m:.2e}, rel {:.2e}",
            gk.schedule,
            k.registers().unwrap_or(-1),
            k.local_bytes().unwrap_or(-1),
            d / m.max(1e-300)
        );
        if d > rel * m + 1e-15 {
            bad.push((la, lb, lk));
        }
    }
    assert_eq!(classes.len(), 15 * 7, "every class occurs");
    assert!(bad.is_empty(), "{kind:?}: classes beyond {rel:e}: {bad:?}");
    // a pair range lands at the start of the buffer, rows relative to its first pair
    let (p0, p1) = (pairs.len() / 3, 2 * pairs.len() / 3);
    let sub = run_op(&gpu, &b, p0..p1, &op);
    let rows = b.host().rows(p0..p1);
    let all_rows = b.host().rows(0..pairs.len()).len();
    let naux = b.host().naux;
    for p in 0..kind.planes() {
        let a = &g1[p * all_rows * naux + rows.start * naux..p * all_rows * naux + rows.end * naux];
        let s = &sub[p * rows.len() * naux..(p + 1) * rows.len() * naux];
        assert!(
            a.iter().zip(s).all(|(x, y)| x.to_bits() == y.to_bits()),
            "pair range differs"
        );
    }
}

#[test]
fn eri3c_every_class_matches_pair_block() {
    check_all_classes(Eri3cKind::Integrals, 1e-13, Operator::Coulomb);
}

#[test]
fn eri3c_derivatives_every_class_match_deriv_block() {
    check_all_classes(Eri3cKind::Derivatives, 1e-13, Operator::Coulomb);
}

/// erf(omega r)/r at the omegas of wB97X-D3BJ (0.3) and CAM-B3LYP (0.33), and a large one
/// (nearly 1/r): every class against linus-ints' `pair_block`.
#[test]
fn eri3c_erf_every_class_matches_pair_block() {
    for w in [0.3, 0.33, 7.0] {
        check_all_classes(
            Eri3cKind::Integrals,
            1e-13,
            Operator::LongRange { omega: w },
        );
    }
}

#[test]
fn eri3c_erf_derivatives_every_class_match_deriv_block_op() {
    for w in [0.3, 3.0] {
        check_all_classes(
            Eri3cKind::Derivatives,
            1e-13,
            Operator::LongRange { omega: w },
        );
    }
}

/// The operator API: 1/r through `compute_op` is `compute`; erf without `prepare_operator`, and
/// operators the kernels do not compute, are errors; an output offset shifts the rows; one set
/// of erf kernels serves every omega; erf(w r)/r tends to 1/r for large w.
#[test]
fn eri3c_operator_api() {
    let Some(gpu) = common::gpu() else { return };
    let mut mol = harness::structure("u1an39").unwrap();
    mol.z.truncate(5);
    mol.positions.truncate(5);
    let basis = harness::basis(&mol, "def2-svp").unwrap();
    let aux = harness::basis(&mol, "def2-universal-jkfit").unwrap();
    let pairs = all_pairs(&basis);
    let n = pairs.len();
    let host = Eri3cHost::new(&basis, &aux, &pairs, Eri3cKind::Integrals, 1e-16).unwrap();
    let len = host.out_len(0..n);
    let mut b = Eri3c::new(&gpu, host).unwrap();
    let mut out = gpu.alloc_zeros::<f64>(len + 37).unwrap();
    assert!(b
        .compute(&gpu, 0..n, &Operator::LongRange { omega: 0.3 }, &mut out, 0)
        .is_err());
    for bad in [
        Operator::Attenuated {
            coulomb: 0.5,
            long_range: 0.5,
            omega: 0.3,
        },
        Operator::Attenuated {
            coulomb: 0.0,
            long_range: 2.0,
            omega: 0.3,
        },
        Operator::LongRange { omega: 0.0 },
    ] {
        assert!(b.prepare_operator(&gpu, &bad).is_err(), "{bad:?}");
    }
    assert!(!b.has_erf());
    b.prepare_operator(&gpu, &Operator::LongRange { omega: 0.3 })
        .unwrap();
    assert!(b.has_erf());
    let coul = run(&gpu, &b, 0..n);
    let via_op = run_op(&gpu, &b, 0..n, &Operator::Coulomb);
    let mut tmp = gpu.alloc_zeros::<f64>(len).unwrap();
    b.compute(&gpu, 0..n, &Operator::Coulomb, &mut tmp, 0)
        .unwrap();
    let via_op2 = gpu.download(&tmp).unwrap();
    assert!(coul
        .iter()
        .zip(&via_op)
        .all(|(x, y)| x.to_bits() == y.to_bits()));
    assert!(coul
        .iter()
        .zip(&via_op2)
        .all(|(x, y)| x.to_bits() == y.to_bits()));
    b.compute(
        &gpu,
        0..n,
        &Operator::LongRange { omega: 0.3 },
        &mut out,
        37,
    )
    .unwrap();
    let shifted = gpu.download(&out).unwrap();
    let erf = run_op(&gpu, &b, 0..n, &Operator::LongRange { omega: 0.3 });
    assert!(shifted[..37].iter().all(|&x| x == 0.0));
    assert!(shifted[37..]
        .iter()
        .zip(&erf)
        .all(|(x, y)| x.to_bits() == y.to_bits()));
    // the same erf kernels for another omega
    let st = compare(
        &run_op(&gpu, &b, 0..n, &Operator::LongRange { omega: 0.11 }),
        &b.host()
            .cpu_reference(&basis, &aux, 0..n, &Operator::LongRange { omega: 0.11 }),
    );
    st.check(1e-14, 1e-13).unwrap();
    // large omega (k = omega^2 / (omega^2 + alpha) -> 1 but not for the tightest primitives):
    // against the CPU, and close to 1/r
    let big = run_op(&gpu, &b, 0..n, &Operator::LongRange { omega: 1e3 });
    let st = compare(
        &big,
        &b.host()
            .cpu_reference(&basis, &aux, 0..n, &Operator::LongRange { omega: 1e3 }),
    );
    st.check(1e-14, 1e-13).unwrap();
    let st = compare(&big, &coul);
    eprintln!("erf(1000 r)/r against 1/r: {st}");
    st.check(0.0, 1e-4).unwrap();
}

#[test]
fn eri3c_screening_stays_below_the_threshold_scale() {
    let Some(gpu) = common::gpu() else { return };
    let mut mol = harness::structure("u1an39").unwrap();
    mol.z.truncate(10);
    mol.positions.truncate(10);
    let basis = harness::basis(&mol, "def2-tzvp").unwrap();
    let aux = harness::basis(&mol, "def2-universal-jkfit").unwrap();
    let pairs = all_pairs(&basis);
    let host = Eri3cHost::new(&basis, &aux, &pairs, Eri3cKind::Integrals, 1e-14).unwrap();
    let cpu = host.cpu_reference(&basis, &aux, 0..pairs.len(), &Operator::Coulomb);
    let b = Eri3c::new(&gpu, host).unwrap();
    let g = run(&gpu, &b, 0..pairs.len());
    let st = compare(&g, &cpu);
    eprintln!("u1an39[..10] TZVP, prim_thresh 1e-14: {st}");
    st.check(1e-12, 1e-13).unwrap();
}

#[test]
fn eri3c_from_linus_ints_pairs_is_the_same_build() {
    // the entry point of a density-fitting engine that already holds linus-ints' Pair3c list
    // (a DF layout's kept pairs and first rows)
    let Some(gpu) = common::gpu() else { return };
    let mut mol = harness::structure("u1an39").unwrap();
    mol.z.truncate(6);
    mol.positions.truncate(6);
    let basis = harness::basis(&mol, "def2-tzvp").unwrap();
    let aux = harness::basis(&mol, "def2-universal-jkfit").unwrap();
    // every other pair, rows renumbered: a screened list
    let mut row = 0;
    let pairs: Vec<_> = all_pairs(&basis)
        .into_iter()
        .step_by(2)
        .map(|mut p| {
            p.row = row;
            row += basis.shells[p.sa].nfunc() * basis.shells[p.sb].nfunc();
            p
        })
        .collect();
    let p3: Vec<Pair3c> = pairs
        .iter()
        .map(|p| {
            Pair3c::from_shell_pair(&build_pair(
                &ShellView::from_shell(&basis.shells[p.sa]),
                &ShellView::from_shell(&basis.shells[p.sb]),
                p.sa,
                p.sb,
            ))
        })
        .collect();
    let rows: Vec<usize> = pairs.iter().map(|p| p.row).collect();
    let a = Eri3c::new(
        &gpu,
        Eri3cHost::new(&basis, &aux, &pairs, Eri3cKind::Integrals, 1e-16).unwrap(),
    )
    .unwrap();
    let b = Eri3c::new(
        &gpu,
        Eri3cHost::from_pair3c(&basis, &aux, &p3, &rows, 1e-16).unwrap(),
    )
    .unwrap();
    let n = pairs.len();
    let (ga, gb) = (run(&gpu, &a, 0..n), run(&gpu, &b, 0..n));
    assert!(ga.iter().zip(&gb).all(|(x, y)| x.to_bits() == y.to_bits()));
    let st = compare(
        &ga,
        &a.host()
            .cpu_reference(&basis, &aux, 0..n, &Operator::Coulomb),
    );
    st.check(1e-14, 1e-13).unwrap();
}

#[test]
fn eri3c_gamma_gradient_matches_contracted_deriv_block() {
    gamma_gradient(Operator::Coulomb);
}

#[test]
fn eri3c_erf_gamma_gradient_matches_contracted_deriv_block_op() {
    gamma_gradient(Operator::LongRange { omega: 0.3 });
}

/// Launches restricted to blocks of auxiliary shells (`compute_block_op`, block-relative
/// columns) give the full build's columns bit for bit, integrals and derivatives, 1/r and erf.
#[test]
fn eri3c_aux_blocks_reproduce_the_full_build() {
    let Some(gpu) = common::gpu() else { return };
    let mol = Structure::new(
        vec![26, 8, 1],
        vec![[0.0, 0.0, 0.0], [0.4, -0.3, 3.05], [0.9, 1.2, 4.1]],
        0,
        1,
    )
    .unwrap();
    let basis = harness::basis(&mol, "def2-tzvp").unwrap();
    let aux = harness::basis(&mol, "def2-universal-jkfit").unwrap();
    let pairs = all_pairs(&basis);
    let np = pairs.len();
    // blocks of whole shells: per atom, the atom's shells split in two
    let mut blocks = Vec::new();
    for r in &aux.atom_shells {
        let mid = r.start + r.len() / 2;
        for (s0, s1) in [(r.start, mid), (mid, r.end)] {
            if s0 < s1 {
                let p0 = aux.shells[s0].ao_start;
                let last = &aux.shells[s1 - 1];
                blocks.push((p0, last.ao_start + last.nfunc() - p0));
            }
        }
    }
    for kind in [Eri3cKind::Integrals, Eri3cKind::Derivatives] {
        let host = Eri3cHost::new(&basis, &aux, &pairs, kind, 1e-16).unwrap();
        let (rows, naux, planes) = (host.rows(0..np).len(), host.naux, kind.planes());
        let mut b = Eri3c::new(&gpu, host).unwrap();
        let blk = b.aux_blocks(&gpu, &blocks).unwrap();
        for op in [Operator::Coulomb, Operator::LongRange { omega: 0.3 }] {
            b.prepare_operator(&gpu, &op).unwrap();
            let full = run_op(&gpu, &b, 0..np, &op);
            for (bi, &(p0, nq)) in blocks.iter().enumerate() {
                let mut out = gpu.alloc_zeros::<f64>(planes * rows * nq).unwrap();
                b.compute_block(&gpu, 0..np, &blk, bi, &op, &mut out, 0)
                    .unwrap();
                let got = gpu.download(&out).unwrap();
                for s in 0..planes {
                    for r in 0..rows {
                        for q in 0..nq {
                            let (g, f) = (
                                got[(s * rows + r) * nq + q],
                                full[(s * rows + r) * naux + p0 + q],
                            );
                            assert_eq!(
                                g.to_bits(),
                                f.to_bits(),
                                "{kind:?} {op:?} block {bi} plane {s} row {r} col {q}: {g} vs {f}"
                            );
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn eri3c_gamma_hessian_matches_hess_3c() {
    gamma_hessian(Operator::Coulomb);
}

#[test]
fn eri3c_erf_gamma_hessian_matches_hess_3c_op() {
    gamma_hessian(Operator::LongRange { omega: 0.3 });
}

/// The second-derivative kernels' Gamma-contracted Hessian (`Eri3c::hessian`, every class:
/// FeOH with def2-QZVP and def2-universal-jkfit) against linus-ints' `Eri3DerivPlan::hessian` for a random
/// symmetric Gamma, and bitwise run to run.
fn gamma_hessian(op: Operator) {
    use linus_ints::{
        DerivOrder, Eri3Block, Eri3DerivOptions, Eri3DerivPlan, PairOptions, PairSet,
    };
    let Some(gpu) = common::gpu() else { return };
    let mol = Structure::new(
        vec![26, 8, 1],
        vec![[0.0, 0.0, 0.0], [0.4, -0.3, 3.05], [0.9, 1.2, 4.1]],
        0,
        1,
    )
    .unwrap();
    let basis = harness::basis(&mol, "def2-qzvp").unwrap();
    let aux = harness::basis(&mol, "def2-universal-jkfit").unwrap();
    let (n, naux) = (basis.nbf, aux.nbf);
    let natom = 3;
    let n3 = 3 * natom;
    // a random symmetric Gamma_P[m][n]
    let mut rng = common::Rng(0x5eed_4321);
    let mut g = vec![0.0; naux * n * n];
    for p in 0..naux {
        for i in 0..n {
            for j in 0..=i {
                let v = 2.0 * rng.uniform() - 1.0;
                g[(p * n + i) * n + j] = v;
                g[(p * n + j) * n + i] = v;
            }
        }
    }
    // CPU: the plan's Hessian over every auxiliary shell, no screening
    let pl = PairSet::build(&basis, PairOptions::with_order(DerivOrder::Two));
    let ap = PairSet::aux(&aux, PairOptions::with_order(DerivOrder::Two));
    let opts = Eri3DerivOptions::default();
    let plan = Eri3DerivPlan::new(&basis, &pl, &aux, &ap, DerivOrder::Two, opts).unwrap();
    let ao: Vec<(usize, usize)> = basis
        .shells
        .iter()
        .map(|s| (s.ao_start, s.nfunc()))
        .collect();
    let mut want = vec![0.0; n3 * n3];
    let blk = Eri3Block {
        shells: 0..aux.shells.len(),
        op,
        thresh: 0.0,
    };
    let mut ws = plan.workspace(rayon::current_num_threads());
    plan.hessian(&blk, &g, &mut want, &mut ws);
    // device: Gamma in the pair-row layout (2 Gamma_mn off the diagonal shell pairs)
    let pairs = all_pairs(&basis);
    let host = Eri3cHost::new(&basis, &aux, &pairs, Eri3cKind::Hessian, 0.0).unwrap();
    let np = pairs.len();
    let rows = host.rows(0..np).len();
    let mut gr = vec![0.0; rows * naux];
    for pr in &pairs {
        let (a0, na) = ao[pr.sa];
        let (b0, nb) = ao[pr.sb];
        let f = if pr.sa == pr.sb { 1.0 } else { 2.0 };
        for ia in 0..na {
            for ib in 0..nb {
                let r = pr.row + ia * nb + ib;
                for p in 0..naux {
                    gr[r * naux + p] = f * g[(p * n + a0 + ia) * n + b0 + ib];
                }
            }
        }
    }
    let mut b = Eri3c::new(&gpu, host).unwrap();
    b.prepare_operator(&gpu, &op).unwrap();
    let gam = gpu.upload(&gr).unwrap();
    let mut got = Vec::new();
    for _ in 0..2 {
        let mut acc = linus_gpu::detsum::DetAcc::new(&gpu, n3 * n3).unwrap();
        b.hessian(&gpu, 0..np, &op, &gam, &mut acc).unwrap();
        let m = acc.to_f64(&gpu).unwrap();
        got.push(
            (0..n3 * n3)
                .map(|e| 0.5 * (m[e] + m[(e % n3) * n3 + e / n3]))
                .collect::<Vec<f64>>(),
        );
    }
    assert!(
        got[0]
            .iter()
            .zip(&got[1])
            .all(|(x, y)| x.to_bits() == y.to_bits()),
        "Hessian not deterministic"
    );
    let st = compare(&got[0], &want);
    eprintln!("FeOH QZVP Gamma Hessian, {op:?}: {st}");
    st.check(0.0, 1e-12).unwrap();
}

fn gamma_gradient(op: Operator) {
    let Some(gpu) = common::gpu() else { return };
    // FeOH: three atoms, so the per-atom sums are not fixed by translational invariance alone
    let mol = Structure::new(
        vec![26, 8, 1],
        vec![[0.0, 0.0, 0.0], [0.4, -0.3, 3.05], [0.9, 1.2, 4.1]],
        0,
        1,
    )
    .unwrap();
    let basis = harness::basis(&mol, "def2-qzvp").unwrap();
    let aux = harness::basis(&mol, "def2-universal-jkfit").unwrap();
    let pairs = all_pairs(&basis);
    let host = Eri3cHost::new(&basis, &aux, &pairs, Eri3cKind::Derivatives, 0.0).unwrap();
    let np = pairs.len();
    let rows = host.rows(0..np).len();
    let naux = host.naux;
    let natom = host.natom;
    let planes = host.cpu_reference(&basis, &aux, 0..np, &op);
    let mut rng = common::Rng(0x5eed_1234);
    let gamma: Vec<f64> = (0..rows * naux)
        .map(|_| 2.0 * rng.uniform() - 1.0)
        .collect();
    // host contraction of the CPU derivative planes
    let mut aux_atom = vec![0usize; naux];
    for s in &aux.shells {
        for a in &mut aux_atom[s.ao_start..s.ao_start + s.nfunc()] {
            *a = s.atom;
        }
    }
    let plane = rows * naux;
    let mut want = vec![0.0; 3 * natom];
    for pr in &pairs {
        let nab = basis.shells[pr.sa].nfunc() * basis.shells[pr.sb].nfunc();
        let (at_a, at_b) = (basis.shells[pr.sa].atom, basis.shells[pr.sb].atom);
        for r in pr.row..pr.row + nab {
            for p in 0..naux {
                let g = gamma[r * naux + p];
                for x in 0..3 {
                    let da = planes[x * plane + r * naux + p];
                    let db = planes[(3 + x) * plane + r * naux + p];
                    want[3 * at_a + x] += g * da;
                    want[3 * at_b + x] += g * db;
                    want[3 * aux_atom[p] + x] -= g * (da + db);
                }
            }
        }
    }
    let mut b = Eri3c::new(&gpu, host).unwrap();
    b.prepare_operator(&gpu, &op).unwrap();
    let gam = gpu.upload(&gamma).unwrap();
    let mut got = Vec::new();
    for _ in 0..2 {
        let mut acc = linus_gpu::detsum::DetAcc::new(&gpu, 3 * natom).unwrap();
        b.gradient(&gpu, 0..np, &op, &gam, &mut acc).unwrap();
        got.push(acc.to_f64(&gpu).unwrap());
    }
    assert!(
        got[0]
            .iter()
            .zip(&got[1])
            .all(|(x, y)| x.to_bits() == y.to_bits()),
        "gradient not deterministic"
    );
    let st = compare(&got[0], &want);
    eprintln!(
        "FeOH QZVP Gamma gradient, {op:?} ({natom} atoms): {st}\n  gpu {:?}\n  cpu {:?}",
        got[0], want
    );
    st.check(0.0, 1e-12).unwrap();
    // translational invariance: the components sum to zero over atoms
    for x in 0..3 {
        let s: f64 = (0..natom).map(|a| got[0][3 * a + x]).sum();
        assert!(s.abs() < 1e-9 * st.ref_max.max(1.0), "sum over atoms {s:e}");
    }
}
