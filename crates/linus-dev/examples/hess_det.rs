//! hess_det <xyz> <basis> [threads a,b,...]: are the pieces of an analytic Hessian bitwise
//! independent of the thread count? Evaluates each derivative-integral contraction of the
//! Hessian (SAD density, no SCF) in rayon pools of the given sizes and prints the max
//! difference against the first pool size per piece (0 = identical bits).
use faer::{Mat, Par};
use linus::JkBuilder;
use linus_ints::basis::BasisDef;
use linus_ints::contract::DensityTerms;
use linus_ints::{
    eri4c::jk_deriv, BasisSet, DerivOrder, Eri4DerivOptions, Eri4DerivPlan, OneEPlan, PairOptions,
    PairSet,
};

fn md(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max)
}

fn flat(m: &Mat<f64>) -> Vec<f64> {
    (0..m.nrows() * m.ncols())
        .map(|x| m[(x / m.ncols(), x % m.ncols())])
        .collect()
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let mol =
        linus::Structure::from_xyz_str(&std::fs::read_to_string(&a[1]).unwrap(), 0, 1).unwrap();
    let threads: Vec<usize> = a
        .get(3)
        .map(|s| s.split(',').map(|x| x.parse().unwrap()).collect())
        .unwrap_or(vec![1, 3, 8]);
    let def = BasisDef::load(&a[2], None).unwrap();
    let basis = BasisSet::new(&def, &mol.z, &mol.positions, true).unwrap();
    let (na, nb) = mol.nalpha_nbeta().unwrap();
    let (da, _) = linus_scf::guess::sad_guess(&def, &basis, &mol.z, na, nb, Par::Seq);
    let n = basis.nbf;
    let daf = flat(&da);
    let dtf: Vec<f64> = daf.iter().map(|x| 2.0 * x).collect();
    let ws: Vec<f64> = daf.iter().map(|x| -0.3 * x).collect();
    let charges = mol.point_charges();
    let auxdef = BasisDef::load("def2-universal-jkfit", None).unwrap();
    let aux = BasisSet::new(&auxdef, &mol.z, &mol.positions, true).unwrap();
    let dens: Vec<Mat<f64>> = (0..4)
        .map(|p| {
            Mat::from_fn(n, n, |i, j| {
                da[(i, j)] * (1.0 + 0.1 * p as f64) * if (i + j + p) % 3 == 0 { -0.5 } else { 1.0 }
            })
        })
        .collect();
    let dens: Vec<Mat<f64>> = dens.iter().map(|d| d + d.transpose()).collect();
    let run = |t: usize| -> Vec<(&'static str, Vec<f64>)> {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(t)
            .build()
            .unwrap();
        let par = if t > 1 { Par::rayon(t) } else { Par::Seq };
        pool.install(|| {
            let pairs = PairSet::build(&basis, PairOptions::with_order(DerivOrder::Two));
            let one_e = OneEPlan::new(&basis, DerivOrder::Two)
                .with_charges(&pairs, &charges)
                .unwrap();
            let mut wk = one_e.workspace();
            let (n3, nx) = (3 * basis.natom(), 3 * basis.natom() * n * n);
            let mut out: Vec<(&'static str, Vec<f64>)> = vec![
                ("overlap_kinetic_hess", {
                    let mut h = vec![0.0; n3 * n3];
                    one_e.overlap_kinetic_hess(Some(&ws), Some(&dtf), &mut h, &mut wk);
                    h
                }),
                ("nuclear_attraction_hess", {
                    let mut h = vec![0.0; n3 * n3];
                    one_e.nuclear_attraction_hess(&dtf, &mut h, &mut wk);
                    h
                }),
                ("overlap_kinetic_deriv", {
                    let mut sx = vec![0.0; nx];
                    one_e.overlap_kinetic_deriv(Some(&mut sx), None);
                    sx
                }),
                ("hcore_deriv", {
                    let mut hx = vec![0.0; nx];
                    one_e.hcore_deriv(&mut hx);
                    hx
                }),
                ("dipole_grad", {
                    let mut g = vec![[[0.0; 3]; 3]; basis.natom()];
                    one_e.dipole_grad([0.0; 3], &dtf, &mut g, &mut wk);
                    g.iter().flat_map(|x| x.iter().flatten().copied()).collect()
                }),
                ("eri_hess_4c", {
                    let terms = DensityTerms::default().coulomb(0.5, &dtf, &dtf).exchange(
                        -0.25 * 1.0,
                        &dtf,
                        &dtf,
                    );
                    let plan = Eri4DerivPlan::new(
                        &basis,
                        &pairs,
                        DerivOrder::Two,
                        Eri4DerivOptions::default(),
                    )
                    .unwrap();
                    let mut ws = plan.workspace(terms.coulomb.len());
                    let n3 = 3 * basis.natom();
                    let mut h = vec![0.0; n3 * n3];
                    plan.hessian(&terms, &mut h, &mut ws);
                    h
                }),
            ];
            let (jx, kx) = jk_deriv::jk_deriv(
                &basis,
                &pairs,
                &[&dtf, &daf],
                true,
                true,
                1e-12,
                &linus_ints::operator::Operator::Coulomb,
            );
            out.push(("jk_deriv J", jx.concat()));
            out.push(("jk_deriv K", kx.concat()));
            let mut fjk = linus_jk::exact::FastJk::new(&basis, 1e-12);
            let o = fjk
                .compute(&linus_base::terms::JkRequest::new(&dens, true, true), par)
                .unwrap();
            out.push(("FastJk J", o.j.iter().flat_map(flat).collect()));
            out.push(("FastJk K", o.k.iter().flat_map(flat).collect()));
            let (_, u) = linus_base::linalg::eigh(da.as_ref(), Par::Seq);
            let co = Mat::from_fn(n, na, |i, j| u[(i, n - 1 - j)]);
            let mut tm = linus_core::timing::Timers::default();
            let p = linus_jk::df::hessian::df_hessian_2e(
                &basis,
                &pairs,
                &aux,
                &linus_jk::df::CoulombMetric::new(&aux, faer::Par::Seq).unwrap(),
                &[co.as_ref()],
                &[2.0],
                Some(linus::linus_ints::operator::Operator::Coulomb),
                linus_base::memory::MemoryLimit::from_gib(None),
                par,
                &mut tm,
            );
            out.push(("df_hessian_2e hess", p.hess));
            out.push((
                "df_hessian_2e fxo",
                p.fxo.iter().flat_map(|f| flat(&f[0])).collect(),
            ));
            out
        })
    };
    let r0 = run(threads[0]);
    for &t in &threads[1..] {
        let r = run(t);
        for ((name, x), (_, y)) in r0.iter().zip(&r) {
            let d = md(x, y);
            println!(
                "{name:28} threads {} vs {t}: {}",
                threads[0],
                if d == 0.0 && x == y {
                    "identical".to_string()
                } else {
                    format!(
                        "max diff {d:.2e} (max |x| {:.2e})",
                        x.iter().fold(0.0f64, |m, v| m.max(v.abs()))
                    )
                }
            );
        }
    }
}
