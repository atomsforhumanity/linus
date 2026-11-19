//! eri3c_check <xyz> <basis> <aux> [threads]: compare eri3c (k|mn) and (k|l) with the generic
//! reference routines (eri4c::eri_3c, eri4c::eri_2c); prints max abs/rel differences and timings.
use linus_ints::eri3c::{self, Aux3c, Pair3c, Work3c};
use linus_ints::operator::Operator;
use linus_ints::{basis::BasisDef, BasisSet, PairOptions, PairSet};
use rayon::prelude::*;
use std::time::Instant;

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let threads: usize = a.get(4).map(|s| s.parse().unwrap()).unwrap_or(8);
    let op = Operator::Coulomb;
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build_global()
        .unwrap();
    let (z, c) =
        linus_core::structure::parse_xyz(&std::fs::read_to_string(&a[1]).unwrap()).unwrap();
    let b = BasisSet::new(&BasisDef::load(&a[2], None).unwrap(), &z, &c, true).unwrap();
    let ab = BasisSet::new(&BasisDef::load(&a[3], None).unwrap(), &z, &c, true).unwrap();
    let pl = PairSet::build(&b, PairOptions::default());
    let t = Instant::now();
    let ap = PairSet::aux(&ab, PairOptions::default());
    let refv = linus_ints::reference::eri_3c(&b, &pl, &ab, &ap);
    let t_ref = t.elapsed().as_secs_f64();
    let t = Instant::now();
    let aux = Aux3c::new(&ab);
    let chunks = aux.chunks(0..ab.shells.len());
    let p3: Vec<Pair3c> = pl.pairs().par_iter().map(Pair3c::from_shell_pair).collect();
    let naux = ab.nbf;
    let blocks: Vec<Vec<f64>> = p3
        .par_iter()
        .map_init(Work3c::new, |w, p| {
            let mut o = vec![0.0; p.nab() * naux];
            eri3c::pair_block(p, &chunks, &mut o, naux, 0.0, &op, w);
            o
        })
        .collect();
    let t_new = t.elapsed().as_secs_f64();
    let nbf = b.nbf;
    let mut worst: f64 = 0.0;
    let mut worst_rel: f64 = 0.0;
    for (p, o) in p3.iter().zip(&blocks) {
        let (a0, b0) = (b.shells[p.sa].ao_start, b.shells[p.sb].ao_start);
        for ia in 0..p.na {
            for ib in 0..p.nb {
                for k in 0..naux {
                    let x = o[(ia * p.nb + ib) * naux + k];
                    let y = refv[k * nbf * nbf + (a0 + ia) * nbf + b0 + ib];
                    worst = worst.max((x - y).abs());
                    if y.abs() > 1e-10 {
                        worst_rel = worst_rel.max(((x - y) / y).abs());
                    }
                }
            }
        }
    }
    let v = eri3c::metric(&aux, 0.0, &Operator::Coulomb);
    let vr = linus_ints::reference::eri_2c(&ab, &ap);
    let dv = v
        .iter()
        .zip(&vr)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max);
    println!("nbf {nbf} naux {naux}: 3c max|d| {worst:.3e} max rel (|ref|>1e-10) {worst_rel:.3e}; metric max|d| {dv:.3e}; t_ref {t_ref:.3} s t_new {t_new:.3} s");
}
