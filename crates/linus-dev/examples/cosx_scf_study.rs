//! cosx_scf_study <xyz> <basis> <charge> <mult> <method> : convergence study of RIJ+COSX SCFs
//! (RI-J def2-universal-jfit, COSX grids of DefGrid2, XC grid orca:2, as the driver's rij-snk).
//!
//! Runs the SCF (verbose; set LINUS_COSX_LOG for the COSX build log) and then, at the final
//! density D (from the final orbitals), measures on the last SCF grid (a single-grid builder):
//! - the screening error of K_t(D) against K at REF_THRESH, for each threshold t of THRESHES,
//!   as max |dK| and as max |X^T (dF D S - S D dF) X| (orbital-gradient error, dF = -a_x dK);
//! - the non-linearity of the screened K under small density changes D2 = D(kappa) (orbitals
//!   rotated by kappa of max size s in PERTS): (K_t(D2) - K_t(D)) - (K_ref(D2) - K_ref(D));
//! - incremental vs full: K_t(D2) + K_t(D - D2) - K_t(D).
//!
//! Prints one JSON object (history, noise table).
//!
//! env: THREADS (8), E_CONV (1e-9), G_CONV (1e-7), MAX_ITER (128), THRESH, THRESH_COARSE,
//! INCR (1), RESET (8), SWITCH (builder's |dD| switch; 0: the SCF's), GRID_SWITCH, GRID_SHIFT (1), DELTA_SCALE, DELTA_REF, LEVELS (1,2,3), NOISE (1), THRESHES (1e-11,1e-12,1e-13),
//! REF_THRESH (1e-15), PERTS (1e-3,1e-5,1e-7), SCF_INCR (scf incremental flag, 1).
use faer::{Mat, Par};
use linus::JkBuilder;
use linus::Structure;
use linus::{CalcOptions, GridChoice, JkMethod, Session};
use linus_base::linalg::{canonical_orthogonalizer, mm};
use linus_base::terms::JkRequest;
use linus_base::terms::XcPotential;
use linus_dft::XcIntegrator;
use linus_grid::{GridSpec, MolecularGrid};
use linus_ints::{PairOptions, PairSet};
use linus_jk::cosx::{scf_builder, CosxConfig, CosxK};
use linus_jk::df::{DfConfig, DfJk};
use linus_jk::CompositeJk;
use linus_scf::{density, run_scf, Guess, ScfConfig, ScfInput};
use serde_json::json;
use std::time::Instant;

fn env<T: std::str::FromStr>(k: &str, d: T) -> T {
    std::env::var(k)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(d)
}

fn envlist(k: &str, d: &str) -> Vec<f64> {
    std::env::var(k)
        .unwrap_or(d.into())
        .split(',')
        .map(|x| x.parse().unwrap())
        .collect()
}

fn maxabs(m: &Mat<f64>) -> f64 {
    let mut v: f64 = 0.0;
    for j in 0..m.ncols() {
        for i in 0..m.nrows() {
            v = v.max(m[(i, j)].abs());
        }
    }
    v
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
    let o = CalcOptions {
        method: Some(method.clone()),
        basis: Some(a[2].clone()),
        jk: JkMethod::RijSnk {
            aux: "def2-universal-jfit".into(),
            k_grid: GridChoice::Cosx { defgrid: 2 },
        }
        .into(),
        grid: linus::driver::options::GridSettings {
            xc: Some(GridChoice::Orca { defgrid: 2 }),
            ..Default::default()
        },
        threads: Some(nt),
        ..Default::default()
    };
    let calc = Session::new(mol.clone(), o).unwrap();
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(nt)
        .build()
        .unwrap();
    pool.install(|| {
        let basis = &calc.basis();
        let n = basis.nbf;
        let pairs = PairSet::build(basis, PairOptions::default());
        let (s, h) = calc.one_electron(&pairs);
        let auxb = linus_ints::BasisSet::new(&calc.aux_basis(&"def2-universal-jfit".into(), calc.scf_purpose()).unwrap(), &mol.z, &mol.positions, true).unwrap();
        let levels: Vec<u8> = std::env::var("LEVELS").unwrap_or("1,2,3".into()).split(',').map(|x| x.parse().unwrap()).collect();
        let levels = [levels[0], levels[1], levels[2]];
        let mut copts = CosxConfig::scf();
        copts.thresh = env("THRESH", copts.thresh);
        copts.thresh_coarse = env("THRESH_COARSE", copts.thresh_coarse);
        copts.incremental = env("INCR", 1u8) == 1;
        copts.reset_every = env("RESET", copts.reset_every);
        copts.switch_dmax = env("SWITCH", copts.switch_dmax);
        copts.delta_scale = env("DELTA_SCALE", copts.delta_scale);
        copts.delta_ref = env("DELTA_REF", copts.delta_ref);
        let j = DfJk::new(basis, &pairs, &auxb, false, DfConfig::default(), par);
        let k = scf_builder(basis, &mol.z, &mol.positions, levels, copts.clone()).unwrap();
        let mut jk = CompositeJk { j: Box::new(j), k: Box::new(k) };
        let xcgrid = MolecularGrid::build(&mol.z, &mol.positions, &GridSpec::orca_defgrid(2)).unwrap();
        let mut xc = XcIntegrator::new(basis, xcgrid, &method).unwrap();
        let (na, nb) = mol.nalpha_nbeta().unwrap();
        let guess = linus_scf::guess::sad_guess(calc.basis_def(), basis, &mol.z, na, nb, par);
        let mut opts = ScfConfig { guess: Guess::Sad, progress: linus_base::progress::Progress::stderr(), max_iter: env("MAX_ITER", 128), e_tol: env("E_CONV", 1e-9), g_tol: env("G_CONV", 1e-7), ..Default::default() };
        opts.algo.incremental = env("SCF_INCR", 1u8) == 1;
        opts.algo.grid_switch = env("GRID_SWITCH", opts.algo.grid_switch);
        opts.algo.grid_switch_shift = env("GRID_SHIFT", 1u8) == 1;
        // the exact exchange of the functional; the K builder applies a range-separated operator
        let xop = linus_dft::XcFunctional::new(&method).unwrap().exchange();
        let (a_x, op) = linus_base::terms::exchange_parts(xop);
        if let Some(op) = op {
            assert!(jk.set_exchange_operator(op));
        }
        let inp = ScfInput { h: h.as_ref(), s: s.as_ref(), e_nuc: mol.nuclear_repulsion(), nalpha: na, nbeta: nb, exchange: xop, guess_density: Some(guess), geometry: None };
        let t0 = Instant::now();
        let r = run_scf(&inp, &mut jk, Some(&mut xc as &mut dyn XcPotential), None, &opts, par).unwrap();
        let t_scf = t0.elapsed().as_secs_f64();
        let mut out = json!({"nbf": n, "method": method, "levels": levels, "thresh": copts.thresh, "incr": copts.incremental, "reset": copts.reset_every,
            "switch": copts.switch_dmax, "converged": r.converged, "builds": r.iterations, "energy_scf": r.energy, "t_scf": t_scf, "history": r.history});
        if env("NOISE", 1u8) == 1 {
            let x = canonical_orthogonalizer(s.as_ref(), 1e-8, par);
            let restricted = r.restricted;
            let occ_d = |ca: &Mat<f64>, cb: &Mat<f64>| -> Vec<Mat<f64>> {
                if restricted {
                    vec![density(ca.as_ref(), na, par)]
                } else {
                    vec![density(ca.as_ref(), na, par), density(cb.as_ref(), nb, par)]
                }
            };
            let d0 = occ_d(&r.c_a, &r.c_b);
            // rotated orbitals: C exp(kappa), kappa_ai = s * sin(...) pattern (deterministic)
            let rotate = |c: &Mat<f64>, no: usize, size: f64| -> Mat<f64> {
                let nmo = c.ncols();
                let nv = nmo - no;
                let kap = Mat::from_fn(nv, no, |a, i| size * (((a * 7 + i * 13) as f64) * 0.618).sin());
                // first-order rotation then re-orthonormalize (Löwdin) to stay a projector
                let co = c.subcols(0, no).to_owned() + mm(c.subcols(no, nv), kap.as_ref(), par);
                let sm = mm(co.transpose(), mm(s.as_ref(), co.as_ref(), par).as_ref(), par);
                let (w, u) = linus_base::linalg::eigh(sm.as_ref(), par);
                let inv = Mat::from_fn(no, no, |p, q| (0..no).map(|k| u[(p, k)] * u[(q, k)] / w[k].sqrt()).sum::<f64>());
                let mut cn = c.clone();
                cn.subcols_mut(0, no).copy_from(mm(co.as_ref(), inv.as_ref(), par).as_ref());
                cn
            };
            let grad_of = |dk: &[Mat<f64>], d: &[Mat<f64>]| -> f64 {
                // orbital-gradient change of dF = -a_x dK: max |X^T (dF D S - S D dF) X|
                let mut g: f64 = 0.0;
                for (k, dd) in dk.iter().zip(d) {
                    let df = k * (-a_x);
                    let fds = mm(mm(df.as_ref(), dd.as_ref(), par).as_ref(), s.as_ref(), par);
                    let e = &fds - fds.transpose();
                    g = g.max(maxabs(&mm(x.transpose(), mm(e.as_ref(), x.as_ref(), par).as_ref(), par)));
                }
                g
            };
            let threshes = envlist("THRESHES", "1e-11,1e-12,1e-13");
            let ref_t: f64 = env("REF_THRESH", 1e-15);
            let perts = envlist("PERTS", "1e-3,1e-5,1e-7");
            let fine_grid = std::sync::Arc::new(MolecularGrid::build(&mol.z, &mol.positions, &GridSpec::cosx(levels[1])).unwrap());
            let mk = |t: f64| -> CosxK {
                let mut k = CosxK::new(basis, fine_grid.clone(), CosxConfig { thresh: t, incremental: false, ..CosxConfig::default() }).with_overlap_fitting(s.as_ref());
                if let Some(op) = op {
                    k.set_exchange_operator(op);
                }
                k
            };
            let kb = |k: &mut CosxK, d: &[Mat<f64>]| -> Vec<Mat<f64>> { k.compute(&JkRequest { dens: d, factors: None, want_j: false, want_k: true }, par).unwrap().k };
            let sub = |a: &[Mat<f64>], b: &[Mat<f64>]| -> Vec<Mat<f64>> { a.iter().zip(b).map(|(x, y)| x - y).collect() };
            let add = |a: &[Mat<f64>], b: &[Mat<f64>]| -> Vec<Mat<f64>> { a.iter().zip(b).map(|(x, y)| x + y).collect() };
            let mx = |a: &[Mat<f64>]| a.iter().map(maxabs).fold(0.0, f64::max);
            let mut kref = mk(ref_t);
            let t1 = Instant::now();
            let kr0 = kb(&mut kref, &d0);
            let t_ref = t1.elapsed().as_secs_f64();
            let dperts: Vec<Vec<Mat<f64>>> = perts
                .iter()
                .map(|&p| {
                    let ca = rotate(&r.c_a, na, p);
                    let cb = if restricted { ca.clone() } else { rotate(&r.c_b, nb, p) };
                    occ_d(&ca, &cb)
                })
                .collect();
            let krp: Vec<Vec<Mat<f64>>> = dperts.iter().map(|d| kb(&mut kref, d)).collect();
            let mut rows = vec![];
            for &t in &threshes {
                let mut kt = mk(t);
                let t1 = Instant::now();
                let k0 = kb(&mut kt, &d0);
                let tt = t1.elapsed().as_secs_f64();
                let e0 = sub(&k0, &kr0);
                let mut prow = vec![];
                for (ip, &p) in perts.iter().enumerate() {
                    let k2 = kb(&mut kt, &dperts[ip]);
                    let nonlin = sub(&sub(&k2, &k0), &sub(&krp[ip], &kr0));
                    // incremental: K(D2) + K(D - D2) vs K(D)
                    let dd = sub(&d0, &dperts[ip]);
                    let td = Instant::now();
                    let kdd = kb(&mut kt, &dd);
                    let t_delta = td.elapsed().as_secs_f64();
                    let inc = sub(&add(&k2, &kdd), &k0);
                    let ddmax = mx(&dd);
                    prow.push(json!({"pert": p, "ddmax": ddmax, "nonlin_k": mx(&nonlin), "nonlin_g": grad_of(&nonlin, &d0),
                        "incr_k": mx(&inc), "incr_g": grad_of(&inc, &d0), "t_delta": t_delta, "err2_g": grad_of(&sub(&k2, &krp[ip]), &dperts[ip])}));
                }
                rows.push(json!({"thresh": t, "time": tt, "err_k": mx(&e0), "err_g": grad_of(&e0, &d0), "perts": prow}));
                eprintln!("{}", rows.last().unwrap());
            }
            out["noise"] = json!({"a_x": a_x, "ref_thresh": ref_t, "t_ref": t_ref, "rows": rows});
        }
        println!("{out}");
    });
}
