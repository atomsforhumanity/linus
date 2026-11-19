//! cosx_grids <xyz> <basis> <charge> <cache_dir> <spec>... : B3LYP exchange-energy error of
//! seminumerical K on candidate grids, for a DF-B3LYP density, against exact K (cached in
//! cache_dir). Each spec is `name:key=value,...` with keys
//!   g (AngularGrid 1..7), acc (IntAcc), b (radial b), h (H angular reduction 0/1),
//!   prune (orca|none|nwchem|r<a0>-<a1>-<a2>-<a3>-<a4> region Lebedev sets),
//!   part (becke|ssf), adj (becke|treutler|none), rad (gc2m3|ta4), bmax (batch size),
//!   wcut (weight cutoff, 0 = none)
//! e.g. `x2:g=2,acc=4.02`. Output: points, dE_x (no fit / fit), max|dK|, build time.
use faer::{Mat, Par};
use linus::JkBuilder;
use linus::Structure;
use linus_base::linalg::{dot, mat_from_rowmajor};
use linus_base::terms::JkRequest;
use linus_base::terms::XcPotential;
use linus_dft::XcIntegrator;
use linus_grid::{
    AtomSizes, ChebyshevKind, GridSpec, MolecularGrid, PartitionScheme, Pruning, RadialMapping,
    RadialScale, RadialScheme, RadiiTable, SizeAdjust,
};
use linus_ints::{basis::BasisDef, BasisSet, PairOptions, PairSet};
use linus_jk::cosx::{CosxConfig, CosxK};
use linus_jk::exact::FastJk;
use linus_scf::{run_scf, ScfConfig, ScfInput};
use std::io::{Read, Write};

fn parse_spec(s: &str) -> (String, GridSpec) {
    let (name, rest) = s.split_once(':').unwrap_or((s, ""));
    let mut g = 2u8;
    let mut acc: Option<f64> = None;
    let mut b = 5.0;
    let mut h = true;
    let mut spec = GridSpec::cosx(2);
    spec.overrides.clear();
    let mut prune = "orca".to_string();
    for kv in rest.split(',').filter(|x| !x.is_empty()) {
        let (k, v) = kv.split_once('=').unwrap();
        match k {
            "g" => g = v.parse().unwrap(),
            "acc" => acc = Some(v.parse().unwrap()),
            "b" => b = v.parse().unwrap(),
            "h" => h = v == "1",
            "prune" => prune = v.to_string(),
            "part" => {
                spec.partition = if v == "ssf" {
                    PartitionScheme::SSF
                } else {
                    PartitionScheme::Becke
                }
            }
            "adj" => {
                spec.size_adjust = match v {
                    "treutler" => SizeAdjust::Treutler(RadiiTable::BraggPyscf),
                    "none" => SizeAdjust::None,
                    _ => SizeAdjust::Becke(RadiiTable::BraggPyscf),
                }
            }
            "rad" => {
                spec.radial = match v {
                    "ta4" => RadialScheme::PYSCF_TREUTLER,
                    _ => RadialScheme::GaussChebyshev {
                        kind: ChebyshevKind::Second,
                        mapping: RadialMapping::TreutlerM3,
                        scale: RadialScale::TreutlerXi,
                    },
                }
            }
            "bmax" => spec.batching.max_points = v.parse().unwrap(),
            "bh" => {
                // radial b for rows >= 3 (applied after the other keys through overrides)
                let bh: f64 = v.parse().unwrap();
                for z in 11u32..=86 {
                    let row = linus_grid::elements::row(z) as f64;
                    let nr = (15.0
                        * acc.unwrap_or(
                            linus_grid::presets::ORCA_COSX_INTACC[(g.min(4) - 1) as usize],
                        )
                        - 40.0
                        + bh * row)
                        .round() as usize;
                    spec.overrides
                        .push((z, nr, linus_grid::prune::orca_angular_max(g as usize)));
                }
            }
            "ov" => {
                // per-element override Z-nrad-nang, several separated by '/'
                for o in v.split('/') {
                    let t: Vec<usize> = o.split('-').map(|x| x.parse().unwrap()).collect();
                    spec.overrides.push((t[0] as u32, t[1], t[2]));
                }
            }
            "wcut" => {
                let w: f64 = v.parse().unwrap();
                spec.weight_cutoff = if w > 0.0 { Some(w) } else { None }
            }
            _ => panic!("unknown key {k}"),
        }
    }
    let int_acc = acc.unwrap_or(linus_grid::presets::ORCA_COSX_INTACC[(g.min(4) - 1) as usize]);
    spec.sizes = AtomSizes::OrcaLike {
        angular_grid: g,
        int_acc,
        b,
        h_reduced: h,
    };
    spec.pruning = match prune.as_str() {
        "orca" => Pruning::OrcaRegions,
        "none" => Pruning::None,
        "nwchem" => Pruning::NWChem,
        r if r.starts_with('r') => {
            let v: Vec<usize> = r[1..].split('-').map(|x| x.parse().unwrap()).collect();
            Pruning::Regions {
                radii: RadiiTable::BraggPyscf,
                alphas: linus_grid::prune::NWCHEM_ALPHAS,
                grids: [v[0], v[1], v[2], v[3], v[4]],
                cap: true,
            }
        }
        _ => panic!("prune {prune}"),
    };
    (name.to_string(), spec)
}

fn write_f64s(path: &std::path::Path, v: &[f64]) {
    let mut f = std::fs::File::create(path).unwrap();
    for x in v {
        f.write_all(&x.to_le_bytes()).unwrap();
    }
}

fn read_f64s(path: &std::path::Path) -> Option<Vec<f64>> {
    let mut f = std::fs::File::open(path).ok()?;
    let mut b = Vec::new();
    f.read_to_end(&mut b).ok()?;
    Some(
        b.as_chunks::<8>()
            .0
            .iter()
            .map(|c| f64::from_le_bytes(*c))
            .collect(),
    )
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let charge: i32 = a[3].parse().unwrap();
    let cache = std::path::PathBuf::from(&a[4]);
    std::fs::create_dir_all(&cache).unwrap();
    let mol = Structure::from_xyz_str(&std::fs::read_to_string(&a[1]).unwrap(), charge, 1).unwrap();
    let nt: usize = std::env::var("THREADS")
        .ok()
        .map(|s| s.parse().unwrap())
        .unwrap_or(8);
    let thresh: f64 = std::env::var("THRESH")
        .ok()
        .map(|s| s.parse().unwrap())
        .unwrap_or(1e-11);
    let par = Par::rayon(nt);
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(nt)
        .build()
        .unwrap();
    pool.install(|| {
        let def = BasisDef::load(&a[2], None).unwrap();
        let basis = BasisSet::new(&def, &mol.z, &mol.positions, true).unwrap();
        let n = basis.nbf;
        let molname = std::path::Path::new(&a[1])
            .file_stem()
            .unwrap()
            .to_string_lossy()
            .to_string();
        let tag = format!("{molname}_{}", a[2]);
        let s = linus_dev::overlap(&basis);
        let s = mat_from_rowmajor(n, n, &s);
        let (pd, pk) = (
            cache.join(format!("{tag}.dens")),
            cache.join(format!("{tag}.kex")),
        );
        let (d, kex) = match (read_f64s(&pd), read_f64s(&pk)) {
            (Some(d), Some(k)) if d.len() == n * n && k.len() == n * n => {
                (mat_from_rowmajor(n, n, &d), mat_from_rowmajor(n, n, &k))
            }
            _ => {
                let pairs = PairSet::build(&basis, PairOptions::default());
                let (_, t, v) = linus_dev::one_electron(&basis, &pairs, &mol.point_charges());
                let h = mat_from_rowmajor(n, n, &t) + mat_from_rowmajor(n, n, &v);
                let aux = BasisSet::new(
                    &BasisDef::load("def2-universal-jfit", None).unwrap(),
                    &mol.z,
                    &mol.positions,
                    true,
                )
                .unwrap();
                let kb = linus_jk::cosx::scf_builder(
                    &basis,
                    &mol.z,
                    &mol.positions,
                    linus_jk::cosx::defgrid_levels(2),
                    CosxConfig::scf(),
                )
                .unwrap();
                let mut df = linus_jk::CompositeJk {
                    j: Box::new(linus_jk::df::DfJk::new(
                        &basis,
                        &pairs,
                        &aux,
                        false,
                        Default::default(),
                        par,
                    )),
                    k: Box::new(kb),
                };
                let xg = MolecularGrid::build(&mol.z, &mol.positions, &GridSpec::orca_defgrid(2))
                    .unwrap();
                let mut xc = XcIntegrator::new(&basis, xg, "b3lyp").unwrap();
                let (na, nb) = mol.nalpha_nbeta().unwrap();
                let guess = linus_scf::guess::sad_guess(&def, &basis, &mol.z, na, nb, par);
                let inp = ScfInput {
                    h: h.as_ref(),
                    s: s.as_ref(),
                    e_nuc: mol.nuclear_repulsion(),
                    nalpha: na,
                    nbeta: nb,
                    exchange: linus_dft::XcFunctional::new("b3lyp").unwrap().exchange(),
                    guess_density: Some(guess),
                    geometry: None,
                };
                let r = run_scf(
                    &inp,
                    &mut df,
                    Some(&mut xc as &mut dyn XcPotential),
                    None,
                    &ScfConfig {
                        e_tol: 1e-8,
                        g_tol: 1e-6,
                        ..Default::default()
                    },
                    par,
                )
                .unwrap();
                let flat =
                    |m: &Mat<f64>| (0..n * n).map(|x| m[(x / n, x % n)]).collect::<Vec<f64>>();
                if std::env::var_os("NOKEX").is_some() {
                    // density only (large molecules: cosx_profile runs without the exact K)
                    write_f64s(&pd, &flat(&r.d_a));
                    eprintln!("{tag}: density written ({} SCF iterations)", r.iterations);
                    return;
                }
                let t0 = std::time::Instant::now();
                let mut ex = FastJk::new(&basis, 1e-13);
                let k = ex
                    .compute(
                        &JkRequest {
                            dens: std::slice::from_ref(&r.d_a),
                            factors: None,
                            want_j: false,
                            want_k: true,
                        },
                        par,
                    )
                    .unwrap()
                    .k
                    .remove(0);
                eprintln!("{tag}: exact K in {:.1}s", t0.elapsed().as_secs_f64());
                write_f64s(&pd, &flat(&r.d_a));
                write_f64s(&pk, &flat(&k));
                (r.d_a, k)
            }
        };
        let ax = 0.2;
        let ex_exact = -ax * dot(d.as_ref(), kex.as_ref());
        println!("# {tag} nbf {n}  E_x(B3LYP HF part, closed shell) = {ex_exact:.8}");
        for sp in &a[5..] {
            let (name, spec) = parse_spec(sp);
            let g = MolecularGrid::build(&mol.z, &mol.positions, &spec).unwrap();
            let mut line = format!("{tag:18} {name:12} {:7} pts", g.x.len());
            for fit in [false, true] {
                let mut k = CosxK::new(
                    &basis,
                    g.clone(),
                    CosxConfig {
                        thresh,
                        dist_screen: std::env::var("DIST").map(|v| v != "0").unwrap_or(true),
                        ..Default::default()
                    },
                );
                if fit {
                    k = k.with_overlap_fitting(s.as_ref());
                }
                let t0 = std::time::Instant::now();
                let kk = k.build(std::slice::from_ref(&d), par).remove(0);
                let tb = t0.elapsed().as_secs_f64();
                let e = -ax * dot(d.as_ref(), kk.as_ref());
                let mut maxd: f64 = 0.0;
                for j in 0..n {
                    for i in 0..n {
                        maxd = maxd.max((kk[(i, j)] - kex[(i, j)]).abs());
                    }
                }
                line += &format!(
                    "  {}: dEx {:+.2e} max|dK| {:.1e} {:.2}s",
                    if fit { "fit" } else { "raw" },
                    e - ex_exact,
                    maxd,
                    tb
                );
            }
            println!("{line}");
        }
    });
}
