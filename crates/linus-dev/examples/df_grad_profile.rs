//! Time the blocked DF two-electron gradient (linus_jk::df::gradient) on converged orbitals:
//! df_grad_profile <xyz> <basis> <method> [--charge Q] [--mult M] [--threads N] [--aux A]
//! [--cache file.json] [--repeat R]
//! The SCF result is cached in --cache (orbitals as JSON) so repeated runs skip the SCF.
//! Range-separated functionals include the K_LR terms. Prints the timers as JSON (one entry
//! per repetition).
//! --mult > 1: unrestricted orbitals (alpha and beta cached).
use faer::{Mat, Par};
use linus::{CalcOptions, JkMethod, Session};
use linus_core::timing::Timers;
use linus_ints::operator::Operator;
use linus_ints::{BasisSet, DerivOrder, PairOptions, PairSet};
use linus_jk::df::gradient::df_gradient;

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let get = |k: &str| a.iter().position(|x| x == k).map(|i| a[i + 1].clone());
    let charge: i32 = get("--charge").map(|s| s.parse().unwrap()).unwrap_or(0);
    let mult: u32 = get("--mult").map(|s| s.parse().unwrap()).unwrap_or(1);
    let threads: usize = get("--threads").map(|s| s.parse().unwrap()).unwrap_or(8);
    let aux = get("--aux").unwrap_or("def2-universal-jkfit".into());
    let repeat: usize = get("--repeat").map(|s| s.parse().unwrap()).unwrap_or(1);
    let mol =
        linus::Structure::from_xyz_str(&std::fs::read_to_string(&a[1]).unwrap(), charge, mult)
            .unwrap();
    let mut o = CalcOptions {
        basis: Some(a[2].clone()),
        method: Some(a[3].clone()),
        threads: Some(threads),
        jk: JkMethod::Df { aux: (&aux).into() }.into(),
        scf: linus::driver::ScfSettings {
            convergence: Some(linus::driver::options::Auto::Value(
                linus::driver::ConvergencePreset::Normal,
            )),
            ..Default::default()
        },
        ..Default::default()
    };
    o.grid.xc = Some(linus::GridChoice::Orca { defgrid: 2 });
    let mut calc = Session::new(mol, o).unwrap();
    let n = calc.basis().nbf;
    let cache = get("--cache");
    let to_mat = |v: &serde_json::Value, key: &str, nkey: &str| -> Mat<f64> {
        let no = v[nkey].as_u64().unwrap() as usize;
        let c: Vec<f64> = v[key]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_f64().unwrap())
            .collect();
        Mat::from_fn(n, no, |i, j| c[i * no + j])
    };
    let (ca, cb): (Mat<f64>, Option<Mat<f64>>) =
        match cache.as_ref().and_then(|p| std::fs::read_to_string(p).ok()) {
            Some(s) => {
                let v: serde_json::Value = serde_json::from_str(&s).unwrap();
                let ca = to_mat(&v, "c", "nocc");
                let cb = v.get("c_b").map(|_| to_mat(&v, "c_b", "nocc_b"));
                (ca, cb)
            }
            None => {
                let r = calc.energy().unwrap();
                let ca = r.c_a.subcols(0, r.nalpha).to_owned();
                let cb = (!r.restricted).then(|| r.c_b.subcols(0, r.nbeta).to_owned());
                if let Some(p) = &cache {
                    let flat = |c: &Mat<f64>| -> Vec<f64> {
                        (0..n * c.ncols())
                            .map(|x| c[(x / c.ncols(), x % c.ncols())])
                            .collect()
                    };
                    let mut j = serde_json::json!({"nocc": ca.ncols(), "c": flat(&ca)});
                    if let Some(cb) = &cb {
                        j["nocc_b"] = cb.ncols().into();
                        j["c_b"] = flat(cb).into();
                    }
                    std::fs::write(p, j.to_string()).unwrap();
                }
                (ca, cb)
            }
        };
    let f = linus_xc::Functional::new(&a[3], false).unwrap();
    // the exchange operator with its coefficients (global hybrids: a_x / r)
    let xop = match f.range_separation() {
        Some((al, be, w)) => Operator::from_libxc(al, be, w),
        None => Operator::Attenuated {
            coulomb: f.exx_fraction(),
            long_range: 0.0,
            omega: 0.0,
        },
    };
    let (x, lr) = (
        xop.coulomb(),
        xop.has_erf().then_some((xop.long_range(), xop.omega())),
    );
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .unwrap();
    let par = Par::rayon(threads);
    let out = pool.install(|| {
        let b: &BasisSet = calc.basis();
        let pairs = PairSet::build(b, PairOptions::with_order(DerivOrder::One));
        let adef = calc.opts().load_basis(&aux).unwrap();
        let auxb = BasisSet::new(&adef, &calc.mol().z, &calc.mol().positions, true).unwrap();
        let restricted = cb.is_none();
        let d = match &cb {
            None => &ca * ca.transpose() * 2.0,
            Some(cb) => &ca * ca.transpose() + cb * cb.transpose(),
        };
        let dt: Vec<f64> = (0..n * n).map(|k| d[(k / n, k % n)]).collect();
        let occ: Vec<&Mat<f64>> = match &cb {
            None => vec![&ca],
            Some(cb) => vec![&ca, cb],
        };
        let mut runs = vec![];
        let mut g = vec![];
        for _ in 0..repeat {
            let mut t = Timers::default();
            let t0 = std::time::Instant::now();
            g = df_gradient(b, &pairs, &auxb, &linus_jk::df::CoulombMetric::new(&auxb, faer::Par::Seq).unwrap(), &dt, &occ, restricted, xop, par, &mut t);
            t.add("total", t0.elapsed().as_secs_f64());
            runs.push(t.to_json());
        }
        let gmax = g.iter().flat_map(|p| p.iter().map(|x| x.abs())).fold(0.0, f64::max);
        serde_json::json!({"nbf": n, "naux": auxb.nbf, "nocc": ca.ncols(), "restricted": restricted, "a": x, "lr": lr, "runs": runs,
            "max_abs_g": gmax, "g": g})
    });
    let _ = &mut calc;
    println!("{out}");
}
