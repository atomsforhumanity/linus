//! gridk_study <xyz> <basis> <charge> <mult> <method> : the accuracy of grid-based exchange and of
//! density fitting at a fixed density (the study behind compensated grid-K's opt-in status).
//!
//! Runs the SCF with RI-JK (the first of AUXES) for the density D, then at D evaluates the
//! exchange energy E_x = sum_s kappa_s E_K\[D_s\] (kappa: -a_x, a closed shell's alpha density
//! counted twice) and the Coulomb energy E_J = 1/2 D_t . J\[D_t\] with
//! - exact four-center integrals (the reference),
//! - density fitting with each auxiliary basis of AUXES (J and K errors: jk df's error is their
//!   sum; rij_cgk's is the J error plus the grid's K error),
//! - grid-K in its four variants (raw, dSCF, overlap fit, compensated) on every grid of GRIDS,
//!
//! and with GRAD=1 the gradients of E_x and E_J at fixed D the same ways (DF's exchange gradient as
//! the DF gradient with exchange minus the one without). The errors at fixed D are the first-order
//! errors of the SCF energy (it is stationary in D) and of the gradient's J and K terms. Prints
//! one JSON object.
//!
//! env: THREADS (8), GRIDS ("gridx:1,gridx:2,gridx:3"), GRAD (0), VARIANTS ("raw,dscf,fit,cgk"),
//! AUXES ("def2-universal-jkfit;cc-pvqz-jkfit|def2-universal-jkfit", ';'-separated), THRESH
//! (grid-K screening, its default).
use faer::{Mat, Par};
use linus::Structure;
use linus::{CalcOptions, GridChoice, JkMethod, Session};
use linus_base::linalg::{dot, mat_to_rowmajor};
use linus_base::terms::{JkBuilder, JkRequest};
use linus_ints::contract::DensityTerms;
use linus_ints::operator::Operator;
use linus_ints::{DerivOrder, Eri4DerivOptions, Eri4DerivPlan, PairOptions, PairSet};
use linus_jk::exact::FastJk;
use linus_jk::gridk::{GridK, GridKConfig};
use serde_json::json;
use std::time::Instant;

fn env<T: std::str::FromStr>(k: &str, d: T) -> T {
    std::env::var(k)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(d)
}

fn gdiff(a: &[[f64; 3]], b: &[[f64; 3]]) -> (f64, f64) {
    let mut mx: f64 = 0.0;
    let mut ss = 0.0;
    for (x, y) in a.iter().zip(b) {
        for c in 0..3 {
            let d = x[c] - y[c];
            mx = mx.max(d.abs());
            ss += d * d;
        }
    }
    (mx, (ss / (3 * a.len()) as f64).sqrt())
}

fn sub(a: &[[f64; 3]], b: &[[f64; 3]]) -> Vec<[f64; 3]> {
    a.iter()
        .zip(b)
        .map(|(x, y)| [x[0] - y[0], x[1] - y[1], x[2] - y[2]])
        .collect()
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let charge: i32 = a[3].parse().unwrap();
    let mult: u32 = a[4].parse().unwrap();
    let method = a[5].clone();
    let mol =
        Structure::from_xyz_str(&std::fs::read_to_string(&a[1]).unwrap(), charge, mult).unwrap();
    let nt: usize = env("THREADS", 8);
    let par = Par::rayon(nt);
    let auxes: Vec<String> = std::env::var("AUXES")
        .unwrap_or("def2-universal-jkfit;cc-pvqz-jkfit|def2-universal-jkfit".into())
        .split(';')
        .map(String::from)
        .collect();
    let o = CalcOptions {
        method: Some(method.clone()),
        basis: Some(a[2].clone()),
        jk: JkMethod::Df {
            aux: auxes[0].as_str().into(),
        }
        .into(),
        threads: Some(nt),
        ..Default::default()
    };
    let mut calc = Session::new(mol.clone(), o).unwrap();
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(nt)
        .build()
        .unwrap();
    let t0 = Instant::now();
    let r = calc.energy().unwrap();
    let t_scf = t0.elapsed().as_secs_f64();
    let basis = calc.basis().clone();
    let xop = linus_dft::XcFunctional::new(&method).unwrap().exchange();
    let (a_x, op) = linus_base::terms::exchange_parts(xop);
    let kop = op.unwrap_or(Operator::Coulomb);
    let (na, nb) = mol.nalpha_nbeta().unwrap();
    let restricted = r.restricted;
    let dens: Vec<Mat<f64>> = if restricted {
        vec![linus_scf::density(r.c_a.as_ref(), na, par)]
    } else {
        vec![
            linus_scf::density(r.c_a.as_ref(), na, par),
            linus_scf::density(r.c_b.as_ref(), nb, par),
        ]
    };
    let kappa: Vec<f64> = if restricted {
        vec![-2.0 * a_x]
    } else {
        vec![-a_x; 2]
    };
    let want_grad = env("GRAD", 0u8) == 1;
    let grids: Vec<String> = std::env::var("GRIDS")
        .unwrap_or("gridx:1,gridx:2,gridx:3".into())
        .split(',')
        .map(String::from)
        .collect();
    let variants: Vec<String> = std::env::var("VARIANTS")
        .unwrap_or("raw,dscf,fit,cgk".into())
        .split(',')
        .map(String::from)
        .collect();
    let out = pool.install(|| {
        let n = basis.nbf;
        let ex = |e: &[f64]| -> f64 { e.iter().zip(&kappa).map(|(e, k)| e * k).sum() };
        let dt = if restricted {
            &dens[0] * 2.0
        } else {
            &dens[0] + &dens[1]
        };
        // exact J and K
        let t = Instant::now();
        let mut fj = FastJk::new(&basis, 1e-13).with_k_operator(kop);
        let kx = fj
            .compute(&JkRequest::new(&dens, false, true), par)
            .unwrap()
            .k;
        let ek_exact: Vec<f64> = dens
            .iter()
            .zip(&kx)
            .map(|(d, k)| 0.5 * dot(d.as_ref(), k.as_ref()))
            .collect();
        let one = std::slice::from_ref(&dt);
        let mut fjj = FastJk::new(&basis, 1e-13);
        let jx = fjj
            .compute(&JkRequest::new(one, true, false), par)
            .unwrap()
            .j;
        let ej_exact = 0.5 * dot(dt.as_ref(), jx[0].as_ref());
        let t_exact = t.elapsed().as_secs_f64();
        let ex_exact = ex(&ek_exact);
        let gpairs = PairSet::build(&basis, PairOptions::with_order(DerivOrder::One));
        // exact gradients of E_x and E_J
        let (gk_exact, gj_exact) = if want_grad {
            let dr: Vec<Vec<f64>> = dens.iter().map(|d| mat_to_rowmajor(d.as_ref())).collect();
            let dtr = mat_to_rowmajor(dt.as_ref());
            let grad = |terms: &DensityTerms, op: Operator| {
                let plan = Eri4DerivPlan::new(
                    &basis,
                    &gpairs,
                    DerivOrder::One,
                    Eri4DerivOptions { thresh: 1e-13, op },
                )
                .unwrap();
                let mut ws = plan.workspace(terms.coulomb.len());
                let mut g = vec![[0.0; 3]; basis.natom()];
                plan.gradient(terms, &mut g, &mut ws);
                g
            };
            let mut tk = DensityTerms::default();
            for (d, k) in dr.iter().zip(&kappa) {
                tk = tk.exchange(0.5 * k, d, d);
            }
            let tj = DensityTerms::default().coulomb(0.5, &dtr, &dtr);
            (Some(grad(&tk, kop)), Some(grad(&tj, Operator::Coulomb)))
        } else {
            (None, None)
        };
        // density fitting with each auxiliary basis
        let pairs = PairSet::build(&basis, PairOptions::default());
        let occ: Vec<Mat<f64>> = if restricted {
            vec![r.c_a.subcols(0, na).to_owned()]
        } else {
            vec![
                r.c_a.subcols(0, na).to_owned(),
                r.c_b.subcols(0, nb).to_owned(),
            ]
        };
        let mut dfrows = vec![];
        for auxn in &auxes {
            let t = Instant::now();
            let auxdef = calc
                .aux_basis(&auxn.as_str().into(), calc.scf_purpose())
                .unwrap();
            let auxb =
                linus_ints::BasisSet::new(&auxdef, &mol.z, &mol.positions, basis.pure).unwrap();
            let mut df =
                linus_jk::df::DfJk::new(&basis, &pairs, &auxb, true, Default::default(), par);
            if let Some(op) = op {
                df.set_exchange_operator(op);
            }
            let kdf = df
                .compute(&JkRequest::new(&dens, false, true), par)
                .unwrap()
                .k;
            let ek_df: Vec<f64> = dens
                .iter()
                .zip(&kdf)
                .map(|(d, k)| 0.5 * dot(d.as_ref(), k.as_ref()))
                .collect();
            let jdf = df
                .compute(&JkRequest::new(one, true, false), par)
                .unwrap()
                .j;
            let mut row = json!({"aux": auxn, "naux": auxb.nbf,
                "dex": ex(&ek_df) - ex_exact,
                "dej": 0.5 * dot(dt.as_ref(), jdf[0].as_ref()) - ej_exact,
                "t": t.elapsed().as_secs_f64()});
            if let (Some(gkx), Some(gjx)) = (&gk_exact, &gj_exact) {
                let dtf = mat_to_rowmajor(dt.as_ref());
                let metric = linus_jk::df::CoulombMetric::new(&auxb, par).unwrap();
                let occr: Vec<&Mat<f64>> = occ.iter().collect();
                let gop = Operator::Attenuated {
                    coulomb: kop.coulomb() * a_x,
                    long_range: kop.long_range() * a_x,
                    omega: kop.omega(),
                };
                let mut timers = linus_core::timing::Timers::default();
                let mut dfg = |op: Operator| {
                    linus_jk::df::gradient::df_gradient(
                        &basis,
                        &gpairs,
                        &auxb,
                        &metric,
                        &dtf,
                        &occr,
                        restricted,
                        op,
                        par,
                        &mut timers,
                    )
                };
                let gjk = dfg(gop);
                let gj = dfg(Operator::Attenuated {
                    coulomb: 0.0,
                    long_range: 0.0,
                    omega: 0.0,
                });
                let gk = sub(&gjk, &gj);
                let (mk, rk) = gdiff(&gk, gkx);
                let (mj, rj) = gdiff(&gj, gjx);
                // the J + K gradient error (jk df)
                let gx: Vec<[f64; 3]> = gkx
                    .iter()
                    .zip(gjx)
                    .map(|(a, b)| [a[0] + b[0], a[1] + b[1], a[2] + b[2]])
                    .collect();
                let (mt, rt) = gdiff(&gjk, &gx);
                row["dgk_max"] = json!(mk);
                row["dgk_rms"] = json!(rk);
                row["dgj_max"] = json!(mj);
                row["dgj_rms"] = json!(rj);
                row["dgjk_max"] = json!(mt);
                row["dgjk_rms"] = json!(rt);
                // rij_cgk's J gradient with this aux: keep the vectors for the grid rows
                row["gj_err"] = json!(sub(&gj, gjx));
            }
            dfrows.push(row);
        }
        // the SAD reference (one spin channel: half the superposition), as rij_cgk's
        let d0 = &linus_scf::guess::sad_superposition(calc.basis_def(), &basis, &mol.z) * 0.5;
        let mut rows = vec![];
        for g in &grids {
            let gc: GridChoice = g.parse().unwrap();
            let grid = std::sync::Arc::new(
                linus_grid::MolecularGrid::build(&mol.z, &mol.positions, &gc.spec()).unwrap(),
            );
            for v in &variants {
                let (dscf, fit) = match v.as_str() {
                    "raw" => (false, false),
                    "dscf" => (true, false),
                    "fit" => (false, true),
                    _ => (true, true),
                };
                let cfg = GridKConfig {
                    dscf,
                    fit,
                    xop: kop,
                    thresh: env("THRESH", GridKConfig::default().thresh),
                    ..Default::default()
                };
                let t = Instant::now();
                let mut k = GridK::new(&basis, grid.clone(), Some(d0.clone()), cfg, par).unwrap();
                let b = k.build(&dens, par).unwrap();
                let t_build = t.elapsed().as_secs_f64();
                let ex_g = ex(&b.energy);
                let mut row = json!({"grid": g, "variant": v, "npts": grid.len(),
                    "dex": ex_g - ex_exact, "t": t_build, "fit_nmo": k.fit().map(|f| f.nmo()),
                    "theta_min": k.fit().map(|f| f.theta[0]), "pair_chunks": k.stats.pair_chunks,
                    "pairs": k.stats.npairs, "chunks": grid.sub_batches(linus_jk::gridk::CHUNK).len()});
                if let (Some(gkx), Some(_)) = (&gk_exact, &gj_exact) {
                    let t = Instant::now();
                    let gk = k.gradient(&dens, &kappa, &gpairs, par).unwrap().total();
                    let (mx, rms) = gdiff(&gk, gkx);
                    row["dgk_max"] = json!(mx);
                    row["dgk_rms"] = json!(rms);
                    row["t_grad"] = json!(t.elapsed().as_secs_f64());
                    // rij_cgk's whole two-electron gradient error with each aux's J
                    let mut per_aux = vec![];
                    for d in &dfrows {
                        let ej: Vec<[f64; 3]> =
                            serde_json::from_value(d["gj_err"].clone()).unwrap();
                        let ek = sub(&gk, gkx);
                        let tot: Vec<[f64; 3]> = ek
                            .iter()
                            .zip(&ej)
                            .map(|(a, b)| [a[0] + b[0], a[1] + b[1], a[2] + b[2]])
                            .collect();
                        let zero = vec![[0.0; 3]; tot.len()];
                        let (m, r) = gdiff(&tot, &zero);
                        per_aux.push(json!({"aux": d["aux"], "dgjk_max": m, "dgjk_rms": r}));
                    }
                    row["with_aux"] = json!(per_aux);
                }
                rows.push(row);
            }
        }
        for d in dfrows.iter_mut() {
            if let Some(m) = d.as_object_mut() {
                m.remove("gj_err");
            }
        }
        json!({"natom": mol.z.len(), "nbf": n, "method": method, "basis": a[2],
            "a_x": a_x, "op": format!("{kop:?}"), "energy_df": r.energy, "t_scf": t_scf,
            "ex_exact": ex_exact, "ej_exact": ej_exact, "t_exact": t_exact,
            "df": dfrows, "grids": rows})
    });
    println!("{}", serde_json::to_string(&out).unwrap());
}
