//! A tour of linus-ints's integral API: a basis, a pair set, the J/K plan with its workspace,
//! a gradient contraction, and the raw path on single blocks, with timings.
//!
//! integrals [xyz-file] [basis]   (default: water in def2-SVP)
use linus_ints::contract::DensityTerms;
use linus_ints::{
    AoMatrix, AoMatrixMut, BasisDef, BasisSet, DerivOrder, Eri4DerivOptions, Eri4DerivPlan,
    Eri4Raw, JkArgs, JkOptions, JkOut, JkPlan, JkShape, OneERaw, Operator, PairOptions, PairSet,
};
use std::time::Instant;

/// Atomic numbers and positions (bohr) from an XYZ file (positions in Angstrom).
fn read_xyz(path: &str) -> (Vec<u32>, Vec<[f64; 3]>) {
    linus_core::structure::parse_xyz(&std::fs::read_to_string(path).unwrap()).unwrap()
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let (z, positions) = match a.get(1) {
        Some(p) => read_xyz(p),
        None => (
            vec![8, 1, 1],
            vec![[0.0, 0.0, 0.22], [1.43, 0.0, -0.89], [-1.43, 0.0, -0.89]],
        ),
    };
    let name = a.get(2).map_or("def2-svp", |s| s.as_str());
    let basis = BasisSet::new(&BasisDef::load(name, None).unwrap(), &z, &positions, true).unwrap();
    let n = basis.nbf;
    println!(
        "{} atoms, {} shells, {} basis functions ({name})",
        z.len(),
        basis.shells.len(),
        n
    );
    // a symmetric model density
    let d: Vec<f64> = (0..n * n)
        .map(|x| 0.01 / (1.0 + (x / n).abs_diff(x % n) as f64))
        .collect();

    // the J/K plan: a pair set, a plan, a workspace
    let t = Instant::now();
    let pairs = PairSet::build(&basis, PairOptions::default());
    let plan = JkPlan::new(&basis, &pairs, JkOptions::default()).unwrap();
    let shape = JkShape {
        ndens: 1,
        j: true,
        k: true,
    };
    let size = plan.workspace_size(&shape);
    let nt = rayon::current_num_threads();
    let mut ws = plan.workspace(shape, nt);
    println!(
        "J/K plan: {:.3} s; workspace {:.1} MiB per thread, {:.1} MiB shared ({nt} threads)",
        t.elapsed().as_secs_f64(),
        size.per_thread as f64 / 1048576.0,
        size.shared as f64 / 1048576.0
    );
    let (mut j, mut k) = (vec![0.0; n * n], vec![0.0; n * n]);
    for build in 0..2 {
        let t = Instant::now();
        let stats = plan.jk(
            &JkArgs {
                dens: &[AoMatrix::row_major(&d, n)],
                thresh: plan.options().thresh,
            },
            &mut JkOut {
                j: &mut [AoMatrixMut::row_major(&mut j, n)],
                k: &mut [AoMatrixMut::row_major(&mut k, n)],
            },
            &mut ws,
        );
        println!(
            "J/K build {build}: {:.3} s, {} blocks, {} quartets",
            t.elapsed().as_secs_f64(),
            stats.blocks,
            stats.quartets
        );
    }
    let dot = |m: &[f64]| m.iter().zip(&d).map(|(x, y)| x * y).sum::<f64>();
    println!("E_J = {:.10}  E_K = {:.10}", 0.5 * dot(&j), -0.25 * dot(&k));

    // the two-electron gradient of the same density
    let t = Instant::now();
    let dpairs = PairSet::build(&basis, PairOptions::with_order(DerivOrder::One));
    let terms = DensityTerms::default()
        .coulomb(0.5, &d, &d)
        .exchange(-0.25, &d, &d);
    let grad = Eri4DerivPlan::new(
        &basis,
        &dpairs,
        DerivOrder::One,
        Eri4DerivOptions::default(),
    )
    .unwrap();
    let size = grad.workspace_size(terms.coulomb.len());
    let mut gws = grad.workspace(terms.coulomb.len());
    let mut g = vec![[0.0; 3]; basis.natom()];
    grad.gradient(&terms, &mut g, &mut gws);
    println!(
        "gradient: {:.3} s; workspace {:.1} MiB",
        t.elapsed().as_secs_f64(),
        size.total(1) as f64 / 1048576.0
    );
    for (i, gi) in g.iter().enumerate() {
        println!(
            "  atom {i}: {:>14.8} {:>14.8} {:>14.8}",
            gi[0], gi[1], gi[2]
        );
    }

    // the raw path: one quartet and its first derivatives, one overlap block
    let sh = &basis.shells;
    let (sa, sb, sc, sd) = (&sh[0], &sh[1], &sh[sh.len() - 1], &sh[2]);
    let mut e = Eri4Raw::new(Operator::Coulomb, DerivOrder::One).unwrap();
    let mut buf = vec![0.0; e.buffer_len(sa, sb, sc, sd)];
    e.compute(sa, sb, sc, sd, &mut buf);
    let nq = buf.len() / 12;
    let sum_x: f64 = (0..4).map(|c| buf[3 * c * nq]).sum();
    println!(
        "raw d(ab|cd)/dX: {} values; sum over the 4 centers of the first: {sum_x:e}",
        buf.len()
    );
    let mut s = vec![0.0; sa.nfunc() * sb.nfunc()];
    let mut tk = s.clone();
    OneERaw::overlap_kinetic(sa, sb, &mut s, &mut tk);
    println!("raw <a|b>, first value: {:.10}", s[0]);
}
