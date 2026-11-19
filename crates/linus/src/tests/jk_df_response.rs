//! linus-jk's density-fitted response J/K against whole calculations (linus_jk::df::response's
//! tests that run the Session).

use crate::driver::{CalcOptions, JkMethod, Session};
use crate::Structure;
use faer::{Mat, MatRef, Par};
use linus_base::terms::{JkBuilder, JkRequest, ResponseJk};
use linus_ints::operator::Operator;
use linus_ints::{BasisSet, PairOptions, PairSet};
use linus_jk::df::response::{
    auto_single, sym_op_usable, DfResponseJk, SINGLE_MIN_TOL, SINGLE_WORK, T1_SLOT_BYTES,
};

/// The closed-shell (one channel) J C and K C of factors `a`.
fn proj1(rj: &mut DfResponseJk, a: &[Mat<f64>], par: Par) -> (Vec<Mat<f64>>, Vec<Mat<f64>>) {
    let (mut j, mut k) = rj.jk_proj(&[a.to_vec()], true, par);
    (j.remove(0), k.remove(0))
}

/// DfResponseJk's projected J C and K C equal the full RI J and K (the reference DenseDfJk) times C for
/// virtual-occupied and occupied-space factors, in double precision; single precision
/// agrees to ~1e-5 relative.
#[test]
fn projected_jk_matches_full_ri_jk() {
    let mol = Structure::from_xyz_str(
        "4\n\nO 0.10 -0.05 0.12\nH 0.85 0.52 -0.03\nH -0.68 0.55 0.05\nC 0.20 -1.30 0.60\n",
        0,
        1,
    )
    .unwrap();
    let o = CalcOptions {
        method: Some("b3lyp".into()),
        basis: Some("def2-svp".into()),
        jk: JkMethod::Df {
            aux: "def2-universal-jkfit".into(),
        }
        .into(),
        threads: Some(4),
        ..Default::default()
    };
    let mut c = Session::new(mol, o).unwrap();
    c.energy().unwrap();
    let r = c.last().unwrap().clone();
    let b = c.cx.basis.clone();
    let n = b.nbf;
    let aux = BasisSet::new(
        &c.cx.plan.load_basis("def2-universal-jkfit").unwrap(),
        &c.cx.mol.z,
        &c.cx.mol.positions,
        true,
    )
    .unwrap();
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap();
    let par = Par::rayon(4);
    pool.install(|| {
        let co = r.c_a.subcols(0, r.nalpha).to_owned();
        let cv = r.c_a.subcols(r.nalpha, r.nmo - r.nalpha).to_owned();
        let s = linus_base::linalg::mat_from_rowmajor(
            n,
            n,
            &linus_ints::reference::oneshot::overlap_kinetic(&b).0,
        );
        // factors: C_v U (general) and C_o M (occupied space)
        let mut a: Vec<Mat<f64>> = (0..3)
            .map(|t| {
                let u = Mat::from_fn(cv.ncols(), co.ncols(), |i, j| {
                    (((i * 7 + j * 3 + t) % 11) as f64 - 5.0) / 50.0
                });
                linus_base::linalg::mm(cv.as_ref(), u.as_ref(), par)
            })
            .collect();
        let m = Mat::from_fn(co.ncols(), co.ncols(), |i, j| {
            ((i + j) % 5) as f64 / 10.0 - 0.2
        });
        a.push(linus_base::linalg::mm(co.as_ref(), m.as_ref(), par));
        let dens: Vec<Mat<f64>> = a
            .iter()
            .map(|x| {
                let t = linus_base::linalg::mm(x.as_ref(), co.transpose(), par);
                &t + t.transpose()
            })
            .collect();
        let mut full = linus_jk::df::reference::DenseDfJk::new(
            &b,
            &PairSet::build(&b, PairOptions::default()),
            &aux,
            par,
        );
        let out = full
            .compute(
                &JkRequest {
                    dens: &dens,
                    factors: None,
                    want_j: true,
                    want_k: true,
                },
                par,
            )
            .unwrap();
        let mut rj = DfResponseJk::new(
            &b,
            &PairSet::build(&b, PairOptions::default()),
            &aux,
            &linus_jk::df::CoulombMetric::new(&aux, faer::Par::Seq).unwrap(),
            &[co.as_ref()],
            s.as_ref(),
            par,
        );
        // one group of densities in exchange term 1, and one density per group
        for (single, slot) in [
            (false, T1_SLOT_BYTES),
            (true, T1_SLOT_BYTES),
            (false, 1),
            (true, 1),
        ] {
            rj.single = single;
            rj.t1_slot_bytes = slot;
            let (jc, kc) = proj1(&mut rj, &a, par);
            for d in 0..a.len() {
                let jr = linus_base::linalg::mm(out.j[d].as_ref(), co.as_ref(), par);
                let kr = linus_base::linalg::mm(out.k[d].as_ref(), co.as_ref(), par);
                let ej = linus_base::linalg::max_abs((&jc[d] - &jr).as_ref())
                    / linus_base::linalg::max_abs(jr.as_ref());
                let ek = linus_base::linalg::max_abs((&kc[d] - &kr).as_ref())
                    / linus_base::linalg::max_abs(kr.as_ref());
                let tol = if single { 1e-5 } else { 1e-12 };
                assert!(
                    ej < tol && ek < tol,
                    "density {d} single {single}: J {ej:e} K {ek:e}"
                );
            }
        }
    });
}

/// The two response J/K implementations agree: DenseResponseJk over the reference RI J/K
/// (DenseDfJk, full densities) and DfResponseJk (projected), closed shell (one channel, c_j = 2) and
/// open shell (two channels, J of the total density).
#[test]
fn dense_and_projected_response_jk_agree() {
    let mol = Structure::from_xyz_str(
        "4\n\nO 0.10 -0.05 0.12\nH 0.85 0.52 -0.03\nH -0.68 0.55 0.05\nC 0.20 -1.30 0.60\n",
        0,
        3,
    )
    .unwrap();
    let o = CalcOptions {
        method: Some("b3lyp".into()),
        basis: Some("def2-svp".into()),
        jk: JkMethod::Df {
            aux: "def2-universal-jkfit".into(),
        }
        .into(),
        threads: Some(4),
        ..Default::default()
    };
    let mut c = Session::new(mol, o).unwrap();
    c.energy().unwrap();
    let r = c.last().unwrap().clone();
    let b = c.cx.basis.clone();
    let n = b.nbf;
    let aux = BasisSet::new(
        &c.cx.plan.load_basis("def2-universal-jkfit").unwrap(),
        &c.cx.mol.z,
        &c.cx.mol.positions,
        true,
    )
    .unwrap();
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap();
    let par = Par::rayon(4);
    pool.install(|| {
        let s = linus_base::linalg::mat_from_rowmajor(
            n,
            n,
            &linus_ints::reference::oneshot::overlap_kinetic(&b).0,
        );
        let pairs = PairSet::build(&b, PairOptions::default());
        let spins = [(r.c_a.clone(), r.nalpha), (r.c_b.clone(), r.nbeta)];
        let occ: Vec<Mat<f64>> = spins
            .iter()
            .map(|(cf, no)| cf.subcols(0, *no).to_owned())
            .collect();
        let a: Vec<Vec<Mat<f64>>> = spins
            .iter()
            .enumerate()
            .map(|(s_, (cf, no))| {
                let cv = cf.subcols(*no, r.nmo - no);
                let mut v: Vec<Mat<f64>> = (0..3)
                    .map(|t| {
                        linus_base::linalg::mm(
                            cv,
                            Mat::from_fn(cv.ncols(), *no, |i, j| {
                                (((i * 5 + j * 3 + t + s_) % 11) as f64 - 5.0) / 50.0
                            })
                            .as_ref(),
                            par,
                        )
                    })
                    .collect();
                v.push(linus_base::linalg::mm(
                    occ[s_].as_ref(),
                    Mat::from_fn(*no, *no, |i, j| ((i + 2 * j + s_) % 5) as f64 / 10.0 - 0.2)
                        .as_ref(),
                    par,
                ));
                v
            })
            .collect();
        for (nch, c_j) in [(1, 2.0), (2, 1.0)] {
            let oc: Vec<MatRef<'_, f64>> = occ[..nch].iter().map(|m| m.as_ref()).collect();
            let mut dense = linus_jk::DenseResponseJk::new(
                Box::new(linus_jk::df::reference::DenseDfJk::new(
                    &b, &pairs, &aux, par,
                )),
                &oc,
            );
            let mut proj = DfResponseJk::new(
                &b,
                &pairs,
                &aux,
                &linus_jk::df::CoulombMetric::new(&aux, faer::Par::Seq).unwrap(),
                &oc,
                s.as_ref(),
                par,
            );
            let (gd, gp) = (
                dense.fock_occ(&a[..nch], c_j, 0.7, par).unwrap(),
                proj.fock_occ(&a[..nch], c_j, 0.7, par).unwrap(),
            );
            for s_ in 0..nch {
                for d in 0..a[s_].len() {
                    let e = linus_base::linalg::max_abs((&gd[s_][d] - &gp[s_][d]).as_ref())
                        / linus_base::linalg::max_abs(gp[s_][d].as_ref());
                    assert!(e < 1e-12, "{nch} channels: channel {s_} density {d}: {e:e}");
                }
            }
        }
    });
}

/// DfResponseJk's setup and projected J C and K C give the same bits on 1, 3 and 8 threads
/// (exchange term 1 deals the auxiliary functions to a fixed number of slots), with one and
/// with several density groups, in double and in single precision, closed and open shell.
#[test]
fn projected_jk_is_thread_count_independent() {
    let mol = Structure::from_xyz_str(
        "4\n\nO 0.10 -0.05 0.12\nH 0.85 0.52 -0.03\nH -0.68 0.55 0.05\nC 0.20 -1.30 0.60\n",
        0,
        3,
    )
    .unwrap();
    let o = CalcOptions {
        method: Some("b3lyp".into()),
        basis: Some("def2-svp".into()),
        jk: JkMethod::Df {
            aux: "def2-universal-jkfit".into(),
        }
        .into(),
        threads: Some(4),
        ..Default::default()
    };
    let mut c = Session::new(mol, o).unwrap();
    c.energy().unwrap();
    let r = c.last().unwrap().clone();
    let b = c.cx.basis.clone();
    let n = b.nbf;
    let aux = BasisSet::new(
        &c.cx.plan.load_basis("def2-universal-jkfit").unwrap(),
        &c.cx.mol.z,
        &c.cx.mol.positions,
        true,
    )
    .unwrap();
    let s = linus_base::linalg::mat_from_rowmajor(
        n,
        n,
        &linus_ints::reference::oneshot::overlap_kinetic(&b).0,
    );
    let spins = [(r.c_a.clone(), r.nalpha), (r.c_b.clone(), r.nbeta)];
    let occ: Vec<Mat<f64>> = spins
        .iter()
        .map(|(cf, no)| cf.subcols(0, *no).to_owned())
        .collect();
    // per channel: 7 virtual-occupied factors and one inside the occupied space
    let a: Vec<Vec<Mat<f64>>> = spins
        .iter()
        .enumerate()
        .map(|(s_, (cf, no))| {
            let cv = cf.subcols(*no, r.nmo - no);
            let mut v: Vec<Mat<f64>> = (0..7)
                .map(|t| {
                    linus_base::linalg::mm(
                        cv,
                        Mat::from_fn(cv.ncols(), *no, |i, j| {
                            (((i * 7 + j * 3 + t + s_) % 11) as f64 - 5.0) / 50.0
                        })
                        .as_ref(),
                        Par::Seq,
                    )
                })
                .collect();
            v.push(linus_base::linalg::mm(
                occ[s_].as_ref(),
                Mat::from_fn(*no, *no, |i, j| ((i + 2 * j + s_) % 5) as f64 / 10.0 - 0.2).as_ref(),
                Par::Seq,
            ));
            v
        })
        .collect();
    let bits = |m: &[Vec<Mat<f64>>]| -> Vec<u64> {
        m.iter()
            .flatten()
            .flat_map(|x| {
                (0..x.ncols()).flat_map(move |j| (0..x.nrows()).map(move |i| x[(i, j)].to_bits()))
            })
            .collect()
    };
    for nch in [1, 2] {
        for (single, slot) in [(false, T1_SLOT_BYTES), (false, 1), (true, T1_SLOT_BYTES)] {
            let run = |threads: usize| {
                let pool = rayon::ThreadPoolBuilder::new()
                    .num_threads(threads)
                    .build()
                    .unwrap();
                let par = if threads > 1 {
                    Par::rayon(threads)
                } else {
                    Par::Seq
                };
                pool.install(|| {
                    let oc: Vec<MatRef<'_, f64>> = occ[..nch].iter().map(|m| m.as_ref()).collect();
                    let mut rj = DfResponseJk::new(
                        &b,
                        &PairSet::build(&b, PairOptions::default()),
                        &aux,
                        &linus_jk::df::CoulombMetric::new(&aux, faer::Par::Seq).unwrap(),
                        &oc,
                        s.as_ref(),
                        par,
                    );
                    rj.single = single;
                    rj.t1_slot_bytes = slot;
                    let (j, k) = rj.jk_proj(&a[..nch], true, par);
                    (bits(&j), bits(&k))
                })
            };
            let r1 = run(1);
            for t in [3, 8] {
                assert!(
                    run(t) == r1,
                    "channels {nch} single {single} slot bytes {slot}: {t} threads differ from 1"
                );
            }
        }
    }
}

/// Range-separated exchange (wK): DfResponseJk's projected K C of the operator a/r + b erf(w r)/r
/// equals the SCF's RI-K of that operator (DfJk, one-pass wK exchange) times C, for
/// virtual-occupied and occupied-space factors, closed shell (one channel) and open shell
/// (two channels); double precision to 1e-12, single to ~1e-5 relative, and the two-level
/// iteration builder (single only) to ~1e-5 relative of its own RI basis.
#[test]
fn projected_wk_exchange_matches_rijk() {
    use linus_jk::df::{DfConfig, DfJk};
    let mol = Structure::from_xyz_str(
        "4\n\nO 0.10 -0.05 0.12\nH 0.85 0.52 -0.03\nH -0.68 0.55 0.05\nC 0.20 -1.30 0.60\n",
        0,
        3,
    )
    .unwrap();
    let o = CalcOptions {
        method: Some("wb97x".into()),
        basis: Some("def2-svp".into()),
        jk: JkMethod::Df {
            aux: "def2-universal-jkfit".into(),
        }
        .into(),
        threads: Some(4),
        ..Default::default()
    };
    let mut c = Session::new(mol, o).unwrap();
    c.energy().unwrap();
    let r = c.last().unwrap().clone();
    let b = c.cx.basis.clone();
    let n = b.nbf;
    let aux = BasisSet::new(
        &c.cx.plan.load_basis("def2-universal-jkfit").unwrap(),
        &c.cx.mol.z,
        &c.cx.mol.positions,
        true,
    )
    .unwrap();
    let (wa, wb, w) = (0.157706, 0.842294, 0.3);
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap();
    let par = Par::rayon(4);
    pool.install(|| {
        let s = linus_base::linalg::mat_from_rowmajor(
            n,
            n,
            &linus_ints::reference::oneshot::overlap_kinetic(&b).0,
        );
        let mut df = DfJk::new(
            &b,
            &PairSet::build(&b, PairOptions::default()),
            &aux,
            true,
            DfConfig {
                omega: Some(w),
                ..DfConfig::default()
            },
            par,
        );
        assert!(df.set_exchange_operator(Operator::Attenuated {
            coulomb: wa,
            long_range: wb,
            omega: w
        }));
        let spins = [(r.c_a.clone(), r.nalpha), (r.c_b.clone(), r.nbeta)];
        // factors per channel: C_v U (general) and C_o M (occupied space)
        let mut facs: Vec<(Mat<f64>, Vec<Mat<f64>>)> = Vec::new();
        for (s_, (cf, no)) in spins.iter().enumerate() {
            let co = cf.subcols(0, *no).to_owned();
            let cv = cf.subcols(*no, r.nmo - no).to_owned();
            let mut a: Vec<Mat<f64>> = (0..3)
                .map(|t| {
                    let u = Mat::from_fn(cv.ncols(), co.ncols(), |i, j| {
                        (((i * 7 + j * 3 + t + s_) % 11) as f64 - 5.0) / 50.0
                    });
                    linus_base::linalg::mm(cv.as_ref(), u.as_ref(), par)
                })
                .collect();
            let m = Mat::from_fn(co.ncols(), co.ncols(), |i, j| {
                ((i + 2 * j + s_) % 5) as f64 / 10.0 - 0.2
            });
            a.push(linus_base::linalg::mm(co.as_ref(), m.as_ref(), par));
            facs.push((co, a));
        }
        let mut reference = |s_: usize| -> Vec<Mat<f64>> {
            let (co, a) = &facs[s_];
            let dens: Vec<Mat<f64>> = a
                .iter()
                .map(|x| {
                    let t = linus_base::linalg::mm(x.as_ref(), co.transpose(), par);
                    &t + t.transpose()
                })
                .collect();
            let out = df
                .compute(
                    &JkRequest {
                        dens: &dens,
                        factors: None,
                        want_j: false,
                        want_k: true,
                    },
                    par,
                )
                .unwrap();
            out.k
                .iter()
                .map(|k| linus_base::linalg::mm(k.as_ref(), co.as_ref(), par))
                .collect()
        };
        let refs = [reference(0), reference(1)];
        let op = Some(linus_ints::operator::Operator::Attenuated {
            coulomb: wa,
            long_range: wb,
            omega: w,
        });
        let rel = |x: &Mat<f64>, y: &Mat<f64>| {
            linus_base::linalg::max_abs((x - y).as_ref()) / linus_base::linalg::max_abs(y.as_ref())
        };
        // closed-shell form: channel 0 alone
        let mut one = DfResponseJk::new_op(
            &b,
            &PairSet::build(&b, PairOptions::default()),
            &aux,
            &linus_jk::df::CoulombMetric::new(&aux, faer::Par::Seq).unwrap(),
            &[facs[0].0.as_ref()],
            s.as_ref(),
            op,
            par,
        );
        assert!(one.exchange_operator() == op);
        for single in [false, true] {
            one.single = single;
            let (_, kc) = proj1(&mut one, &facs[0].1, par);
            for d in 0..kc.len() {
                let e = rel(&kc[d], &refs[0][d]);
                assert!(
                    e < if single { 1e-5 } else { 1e-12 },
                    "closed density {d} single {single}: K {e:e}"
                );
            }
        }
        // open-shell form: both channels
        let occ = [facs[0].0.as_ref(), facs[1].0.as_ref()];
        let mut two = DfResponseJk::new_op(
            &b,
            &PairSet::build(&b, PairOptions::default()),
            &aux,
            &linus_jk::df::CoulombMetric::new(&aux, faer::Par::Seq).unwrap(),
            &occ,
            s.as_ref(),
            op,
            par,
        );
        let (_, kc) = two.jk_proj(&[facs[0].1.clone(), facs[1].1.clone()], true, par);
        for s_ in 0..2 {
            for d in 0..kc[s_].len() {
                let e = rel(&kc[s_][d], &refs[s_][d]);
                assert!(e < 1e-12, "open channel {s_} density {d}: K {e:e}");
            }
        }
        // single-precision only (two-level iteration builder) with the same auxiliary basis,
        // built directly in f32 or converted; and a single-precision main builder released
        // and restored (single precision only) for the final product
        let mut it = DfResponseJk::new_op(
            &b,
            &PairSet::build(&b, PairOptions::default()),
            &aux,
            &linus_jk::df::CoulombMetric::new(&aux, faer::Par::Seq).unwrap(),
            &occ,
            s.as_ref(),
            op,
            par,
        )
        .into_single_only();
        let mut it2 = DfResponseJk::with_mode(
            &b,
            &PairSet::build(&b, PairOptions::default()),
            &aux,
            &linus_jk::df::CoulombMetric::new(&aux, faer::Par::Seq).unwrap(),
            &occ,
            s.as_ref(),
            op,
            false,
            true,
            par,
        );
        let mut rr = DfResponseJk::new_op(
            &b,
            &PairSet::build(&b, PairOptions::default()),
            &aux,
            &linus_jk::df::CoulombMetric::new(&aux, faer::Par::Seq).unwrap(),
            &occ,
            s.as_ref(),
            op,
            par,
        );
        rr.single = true;
        rr.release_storage();
        rr.restore_storage(par);
        for (name, j) in [
            ("single-only", &mut it),
            ("f32-built", &mut it2),
            ("restored", &mut rr),
        ] {
            let (_, kc) = j.jk_proj(&[facs[0].1.clone(), facs[1].1.clone()], true, par);
            for s_ in 0..2 {
                for d in 0..kc[s_].len() {
                    let e = rel(&kc[s_][d], &refs[s_][d]);
                    assert!(e < 1e-5, "{name} channel {s_} density {d}: K {e:e}");
                }
            }
        }
    });
}

/// The symmetric RI of the exchange operator (two-level iterations of range-separated
/// hybrids) approximates the same exact w-exchange as wK: its K C agrees with wK's and with
/// the exact four-center w-exchange (FastJk) to the RI error; J is wK's (Coulomb Bt).
#[test]
fn symmetric_operator_ri_approximates_wk() {
    let mol = Structure::from_xyz_str(
        "4\n\nO 0.10 -0.05 0.12\nH 0.85 0.52 -0.03\nH -0.68 0.55 0.05\nC 0.20 -1.30 0.60\n",
        0,
        1,
    )
    .unwrap();
    let o = CalcOptions {
        method: Some("wb97x".into()),
        basis: Some("def2-svp".into()),
        jk: JkMethod::Df {
            aux: "def2-universal-jkfit".into(),
        }
        .into(),
        threads: Some(4),
        ..Default::default()
    };
    let mut c = Session::new(mol, o).unwrap();
    c.energy().unwrap();
    let r = c.last().unwrap().clone();
    let b = c.cx.basis.clone();
    let n = b.nbf;
    let aux = BasisSet::new(
        &c.cx.plan.load_basis("def2-universal-jkfit").unwrap(),
        &c.cx.mol.z,
        &c.cx.mol.positions,
        true,
    )
    .unwrap();
    let (wa, wb, w) = (0.157706, 0.842294, 0.3);
    let op = linus_ints::operator::Operator::Attenuated {
        coulomb: wa,
        long_range: wb,
        omega: w,
    };
    assert!(
        sym_op_usable(&op)
            && !sym_op_usable(&linus_ints::operator::Operator::LongRange { omega: 0.3 })
    );
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap();
    let par = Par::rayon(4);
    pool.install(|| {
        let s = linus_base::linalg::mat_from_rowmajor(n, n, &linus_ints::reference::oneshot::overlap_kinetic(&b).0);
        let co = r.c_a.subcols(0, r.nalpha).to_owned();
        let cv = r.c_a.subcols(r.nalpha, r.nmo - r.nalpha).to_owned();
        let a: Vec<Mat<f64>> = (0..3)
            .map(|t| {
                let u = Mat::from_fn(cv.ncols(), co.ncols(), |i, j| (((i * 7 + j * 3 + t) % 11) as f64 - 5.0) / 50.0);
                linus_base::linalg::mm(cv.as_ref(), u.as_ref(), par)
            })
            .collect();
        let pairs = PairSet::build(&b, PairOptions::default());
        let mut wk = DfResponseJk::new_op(&b, &pairs, &aux, &linus_jk::df::CoulombMetric::new(&aux, faer::Par::Seq).unwrap(), &[co.as_ref()], s.as_ref(), Some(op), par);
        let mut sy = DfResponseJk::new_sym_op(&b, &pairs, &aux, &linus_jk::df::CoulombMetric::new(&aux, faer::Par::Seq).unwrap(), &[co.as_ref()], s.as_ref(), op, par);
        let (jw, kw) = proj1(&mut wk, &a, par);
        let (js, ks) = proj1(&mut sy, &a, par);
        // exact four-center w-exchange
        let mut fast = linus_jk::exact::FastJk::new(&b, 1e-13);
        assert!(fast.set_exchange_operator(op));
        let dens: Vec<Mat<f64>> = a.iter().map(|x| {
            let t = linus_base::linalg::mm(x.as_ref(), co.transpose(), par);
            &t + t.transpose()
        }).collect();
        let ke = fast.compute(&JkRequest { dens: &dens, factors: None, want_j: false, want_k: true }, par).unwrap().k;
        let rel = |x: &Mat<f64>, y: &Mat<f64>| linus_base::linalg::max_abs((x - y).as_ref()) / linus_base::linalg::max_abs(y.as_ref());
        for d in 0..a.len() {
            let kex = linus_base::linalg::mm(ke[d].as_ref(), co.as_ref(), par);
            let (e_sw, e_we, e_se) = (rel(&ks[d], &kw[d]), rel(&kw[d], &kex), rel(&ks[d], &kex));
            println!("density {d}: sym vs wK {e_sw:.2e}, wK vs exact {e_we:.2e}, sym vs exact {e_se:.2e}");
            assert!(e_sw < 2e-3 && e_se < 2e-3, "density {d}: sym vs wK {e_sw:e}, sym vs exact {e_se:e}");
            assert!(rel(&js[d], &jw[d]) < 1e-12);
        }
        // single precision (the iteration builders are single-only)
        let mut sy32 = DfResponseJk::new_sym_op(&b, &pairs, &aux, &linus_jk::df::CoulombMetric::new(&aux, faer::Par::Seq).unwrap(), &[co.as_ref()], s.as_ref(), op, par).into_single_only();
        let (_, ks32) = proj1(&mut sy32, &a, par);
        for d in 0..a.len() {
            assert!(rel(&ks32[d], &ks[d]) < 1e-5, "single: {:e}", rel(&ks32[d], &ks[d]));
        }
    });
}

/// `precision.response = auto` runs single precision only for a large exchange work and a
/// tolerance single precision reaches (unset: the defaults, which it does).
#[test]
fn auto_precision_follows_work_and_tolerance() {
    let big = 2.0 * SINGLE_WORK;
    for tol in [None, Some(SINGLE_MIN_TOL), Some(1e-6), Some(1e-5)] {
        assert!(auto_single(big, tol), "{tol:?}");
        assert!(!auto_single(SINGLE_WORK, tol), "{tol:?}");
    }
    for tol in [Some(9e-7), Some(1e-9), Some(f64::NAN)] {
        assert!(!auto_single(big, tol), "{tol:?}");
    }
}

/// A CPHF tolerance below the single-precision floor on a system large enough for single
/// precision by size (L-lactide HF/def2-SVP, reference/validate.py's la18.hf.def2-svp.df:
/// in single precision the residual levels off near 3e-8 and the solve fails after 100
/// iterations at 9e-8): `auto` runs double precision and the CPHF converges (10 iterations).
#[test]
fn auto_precision_converges_below_the_single_floor() {
    let mol = Structure::from_xyz_str(include_str!("../../../../bench/molecules/la18.xyz"), 0, 1)
        .unwrap();
    let mut o = CalcOptions {
        method: Some("hf".into()),
        basis: Some("def2-svp".into()),
        jk: JkMethod::Df {
            aux: "def2-universal-jkfit".into(),
        }
        .into(),
        threads: Some(8),
        ..Default::default()
    };
    o.response.tol = Some(crate::driver::options::Auto::Value(1e-9));
    let mut c = Session::new(mol, o).unwrap();
    let aux = BasisSet::new(
        &c.cx.plan.load_basis("def2-universal-jkfit").unwrap(),
        &c.cx.mol.z,
        &c.cx.mol.positions,
        true,
    )
    .unwrap();
    let (n, o) = (c.cx.basis.nbf, c.cx.mol.nalpha_nbeta().unwrap().0);
    let work = (n * n * o * aux.nbf) as f64;
    assert!(work > SINGLE_WORK, "work {work:e}");
    let h = c.hessian().expect("the CPHF converges");
    let r = &h.response;
    assert!(!r.single_precision && !r.two_level);
    assert_eq!(r.cphf_tol, 1e-9);
    assert!(r.max_residual <= 1e-9, "max residual {:e}", r.max_residual);
}
