//! cosx_op_bench <xyz> <basis> [charge] [--threads N] [--grid G] [--reps R]: CosxK build time for
//! the exchange operator of wB97X done as one pass (a/r + b erf(w r)/r) against two passes
//! (1/r and erf(w r)/r), on a converged DF-HF density with overlap fitting. Prints JSON.
use faer::Par;
use linus::JkBuilder;
use linus::Structure;
use linus_base::linalg::mat_from_rowmajor;
use linus_base::terms::JkRequest;
use linus_grid::{GridSpec, MolecularGrid};
use linus_ints::{basis::BasisDef, BasisSet, PairOptions, PairSet};
use linus_jk::cosx::{CosxConfig, CosxK};
use linus_jk::df::{DfConfig, DfJk};
use linus_scf::{run_scf, ScfConfig, ScfInput};

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let get = |k: &str| a.iter().position(|x| x == k).map(|i| a[i + 1].clone());
    let charge: i32 = a.get(3).and_then(|s| s.parse().ok()).unwrap_or(0);
    let nt: usize = get("--threads").map(|s| s.parse().unwrap()).unwrap_or(8);
    let g: u8 = get("--grid").map(|s| s.parse().unwrap()).unwrap_or(2);
    let reps: usize = get("--reps").map(|s| s.parse().unwrap()).unwrap_or(3);
    let mol = Structure::from_xyz_str(&std::fs::read_to_string(&a[1]).unwrap(), charge, 1).unwrap();
    let par = Par::rayon(nt);
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(nt)
        .build()
        .unwrap();
    let out = pool.install(|| {
        let basis = BasisSet::new(&BasisDef::load(&a[2], None).unwrap(), &mol.z, &mol.positions, true).unwrap();
        let pairs = PairSet::build(&basis, PairOptions::default());
        let n = basis.nbf;
        let (s, t, v) = linus_dev::one_electron(&basis, &pairs, &mol.point_charges());
        let (s, h) = (mat_from_rowmajor(n, n, &s), mat_from_rowmajor(n, n, &t) + mat_from_rowmajor(n, n, &v));
        let aux = BasisSet::new(&BasisDef::load("def2-universal-jkfit", None).unwrap(), &mol.z, &mol.positions, true).unwrap();
        let mut df = DfJk::new(&basis, &pairs, &aux, true, DfConfig::default(), par);
        let (na, nb) = mol.nalpha_nbeta().unwrap();
        let inp = ScfInput { h: h.as_ref(), s: s.as_ref(), e_nuc: mol.nuclear_repulsion(), nalpha: na, nbeta: nb, exchange: Some(linus_ints::operator::Operator::Coulomb), guess_density: None, geometry: None };
        let r = run_scf(&inp, &mut df, None, None, &ScfConfig { e_tol: 1e-7, g_tol: 1e-5, ..Default::default() }, par).unwrap();
        let d = vec![r.d_a.clone()];
        let (wa, wb, w) = (0.157706, 0.842294, 0.3);
        let time = |op: Option<(f64, f64, f64)>| -> (f64, faer::Mat<f64>) {
            let mut k = CosxK::new(&basis, MolecularGrid::build(&mol.z, &mol.positions, &GridSpec::cosx(g)).unwrap(), CosxConfig::default()).with_overlap_fitting(s.as_ref());
            if let Some((x, y, z)) = op {
                assert!(k.set_exchange_operator(linus_ints::operator::Operator::Attenuated { coulomb: x, long_range: y, omega: z }));
            }
            let mut ts = vec![];
            let mut last = None;
            for _ in 0..reps {
                let t0 = std::time::Instant::now();
                last = Some(k.compute(&JkRequest::new(&d, false, true), par).unwrap().k.remove(0));
                ts.push(t0.elapsed().as_secs_f64());
            }
            ts.sort_by(|x, y| x.partial_cmp(y).unwrap());
            (ts[ts.len() / 2], last.unwrap())
        };
        let (t_c, k_c) = time(None);
        let (t_l, k_l) = time(Some((0.0, 1.0, w)));
        let (t_one, k_one) = time(Some((wa, wb, w)));
        let k2 = &k_c * wa + &k_l * wb;
        let diff = (0..n * n).map(|x| (k2[(x / n, x % n)] - k_one[(x / n, x % n)]).abs()).fold(0.0, f64::max);
        serde_json::json!({"nbf": n, "grid": g, "threads": nt, "t_coulomb": t_c, "t_erf": t_l, "t_two_pass": t_c + t_l, "t_one_pass": t_one, "max_diff": diff})
    });
    println!("{out}");
}
