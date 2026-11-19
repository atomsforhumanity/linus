//! scf <xyz> <basis> [--charge q] [--mult m] [--jk incore|fast|ri|ri-j|rij-exactk] [--aux name] [--threads n] [--uhf]
//!     [--thresh t] [--xc functional] [--grid level|pyscfN|linusN|orcaN|json:<GridSpec file>] [--e-conv x] [--g-conv x] [--verbose]
//! fast: FastJk (exact, batched SIMD); ri: DfJk (J and K); ri-j: DfJk J only (pure functionals);
//! rij-exactk: DfJk J (default aux def2-universal-jfit) with FastJk exact K (ORCA's RIJONX).
use faer::Par;
use linus::JkBuilder;
use linus::Structure;
use linus_base::linalg::mat_from_rowmajor;
use linus_ints::operator::Operator;
use linus_ints::{basis::BasisDef, BasisSet, PairOptions, PairSet};
use linus_jk::exact::FastJk;
use linus_jk::IncoreJk;
use linus_scf::{run_scf, ScfConfig, ScfInput};

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let get = |k: &str| a.iter().position(|x| x == k).map(|i| a[i + 1].clone());
    let charge: i32 = get("--charge").map(|s| s.parse().unwrap()).unwrap_or(0);
    let mult: u32 = get("--mult").map(|s| s.parse().unwrap()).unwrap_or(1);
    let threads: usize = get("--threads").map(|s| s.parse().unwrap()).unwrap_or(1);
    let jkname = get("--jk").unwrap_or("fast".into());
    let thresh: f64 = get("--thresh").map(|s| s.parse().unwrap()).unwrap_or(1e-14);
    let mol =
        Structure::from_xyz_str(&std::fs::read_to_string(&a[1]).unwrap(), charge, mult).unwrap();
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
        let t0 = std::time::Instant::now();
        let def = BasisDef::load(&a[2], None).unwrap();
        let basis = BasisSet::new(&def, &mol.z, &mol.positions, true).unwrap();
        let pairs = PairSet::build(&basis, PairOptions::default());
        let (s, t, v) = linus_dev::one_electron(&basis, &pairs, &mol.point_charges());
        let n = basis.nbf;
        let s = mat_from_rowmajor(n, n, &s);
        let h = mat_from_rowmajor(n, n, &t) + mat_from_rowmajor(n, n, &v);
        let t_setup = t0.elapsed().as_secs_f64();
        let t1 = std::time::Instant::now();
        let mut jk: Box<dyn JkBuilder> = match jkname.as_str() {
            "incore" => Box::new(IncoreJk::new(&basis, &pairs)),
            "ri" | "ri-j" => {
                let adef = BasisDef::load(&get("--aux").unwrap_or("def2-universal-jkfit".into()), None).unwrap();
                let aux = BasisSet::new(&adef, &mol.z, &mol.positions, true).unwrap();
                Box::new(linus_jk::df::DfJk::new(&basis, &pairs, &aux, jkname == "ri", linus_jk::df::DfConfig::default(), par))
            }
            // RI-J with exact K (ORCA RIJONX): DfJk for J, FastJk for K
            "rij-exactk" => {
                let adef = BasisDef::load(&get("--aux").unwrap_or("def2-universal-jfit".into()), None).unwrap();
                let aux = BasisSet::new(&adef, &mol.z, &mol.positions, true).unwrap();
                Box::new(linus_jk::CompositeJk {
                    j: Box::new(linus_jk::df::DfJk::new(&basis, &pairs, &aux, false, Default::default(), par)),
                    k: Box::new(FastJk::new(&basis, thresh)),
                })
            }
            _ => Box::new(FastJk::new(&basis, thresh)),
        };
        let t_jk_setup = t1.elapsed().as_secs_f64();
        let (na, nb) = mol.nalpha_nbeta().unwrap();
        let mut opts = ScfConfig {
            progress: if a.iter().any(|x| x == "--verbose") {
                linus_base::progress::Progress::stderr()
            } else {
                linus_base::progress::Progress::off()
            },
            ..Default::default()
        };
        if let Some(c) = get("--conv") {
            // "tight": ORCA TightSCF-equivalent (dE 1e-8, DIIS error 5e-7)
            let (e, g) = match c.as_str() { "tight" => (1e-8, 5e-7), "verytight" => (1e-9, 1e-8), _ => (1e-10, 1e-8) };
            opts.e_tol = e;
            opts.g_tol = g;
        }
        if a.iter().any(|x| x == "--no-adiis") {
            opts.algo.adiis = false;
        }
        if let Some(x) = get("--adiis-switch") {
            opts.algo.adiis_switch = x.parse().unwrap();
        }
        if let Some(v) = get("--e-conv") {
            opts.e_tol = v.parse().unwrap();
        }
        if let Some(v) = get("--g-conv") {
            opts.g_tol = v.parse().unwrap();
        }
        if a.iter().any(|x| x == "--uhf") {
            opts.restricted = Some(false);
        }
        let guess = if a.iter().any(|x| x == "--core-guess") {
            None
        } else {
            Some(linus_scf::guess::sad_guess(&def, &basis, &mol.z, na, nb, par))
        };
        let mut xc = get("--xc").map(|f| {
            let gs = get("--grid").unwrap_or("3".into());
            let spec = if let Some(path) = gs.strip_prefix("json:") {
                serde_json::from_str::<linus_grid::GridSpec>(&std::fs::read_to_string(path).unwrap()).unwrap()
            } else if let Some(n) = gs.strip_prefix("orca") {
                linus_grid::GridSpec::orca_defgrid(n.parse().unwrap())
            } else if let Some(n) = gs.strip_prefix("linus") {
                linus_grid::GridSpec::linus(n.parse().unwrap())
            } else {
                linus_grid::GridSpec::pyscf(gs.trim_start_matches("pyscf").parse().unwrap())
            };
            let g = linus_grid::MolecularGrid::build(&mol.z, &mol.positions, &spec).unwrap();
            let x = linus_dft::XcIntegrator::new(&basis, g, &f).unwrap();
            (x, linus_dft::XcFunctional::new(&f).unwrap().exchange())
        });
        // the exact exchange: the functional's, or --exx alpha,beta,omega (exact exchange only, libxc
        // convention alpha K + beta K_sr(omega)), or HF's
        let exchange = match (&xc, get("--exx")) {
            (Some((_, x)), _) => *x,
            (None, Some(v)) => {
                let x: Vec<f64> = v.split(',').map(|t| t.parse().unwrap()).collect();
                Some(if x[1] != 0.0 && x[2] != 0.0 {
                    Operator::from_libxc(x[0], x[1], x[2])
                } else {
                    Operator::Attenuated { coulomb: x[0], long_range: 0.0, omega: 0.0 }
                })
            }
            (None, None) => Some(Operator::Coulomb),
        };
        if let Some(op) = linus_base::terms::exchange_parts(exchange).1 {
            assert!(jk.set_exchange_operator(op), "{}: no range-separated exchange", jk.name());
        }
        let inp = ScfInput { h: h.as_ref(), s: s.as_ref(), e_nuc: mol.nuclear_repulsion(), nalpha: na, nbeta: nb, exchange, guess_density: guess, geometry: None };
        let xcp: Option<&mut dyn linus_base::terms::XcPotential> = xc.as_mut().map(|x| &mut x.0 as &mut dyn linus_base::terms::XcPotential);
        let r = run_scf(&inp, jk.as_mut(), xcp, None, &opts, par).unwrap();
        println!(
            "{}",
            serde_json::json!({"energy": r.energy, "e_xc": r.e_xc, "converged": r.converged, "iterations": r.iterations, "nbf": n,
              "jk": jk.name(), "t_setup": t_setup, "t_jk_setup": t_jk_setup,
              "e_one": r.e_one, "e_coulomb": r.e_coulomb, "e_exchange": r.e_exchange, "t_total": t0.elapsed().as_secs_f64(), "timers": r.timers.to_json()})
        );
    });
}
