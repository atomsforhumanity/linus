//! ORCA 6.0.0's DefGrids (`GridSpec::orca_defgrid`, tables in data/orca/grids.json) against
//! ORCA 6.0.0's own numbers (bench/results/2026-11-07-orca-grid,
//! crates/linus-grid/tools/orca_probe/).

use linus_grid::radial::{radial_grid, ChebyshevKind, RadialMapping, RadialScale, RadialScheme};
use linus_grid::{
    atomic_grid, AtomSizes, Batching, GridSpec, MolecularGrid, Orientation, PartitionScheme,
    Pruning, RadiiTable, SizeAdjust,
};
use std::f64::consts::PI;

/// ORCA 6.0.0, one C atom with a basis of one s primitive (exponent alpha) and two electrons (C4+),
/// RKS B3LYP/G, `%method AngularGrid 4 IntAcc 4.388` (DefGrid2's XC grid): the printed `N(Total)`.
/// The density is spherical, so N(Total) = sum_i W_i rho(r_i) over ORCA's radial grid alone.
const ORCA_C_DEFGRID2: [(f64, f64); 14] = [
    (0.002, 0.305410548746),
    (0.005, 0.878134975700),
    (0.01, 1.527970594064),
    (0.02, 1.952331058397),
    (0.03, 2.002512461551),
    (0.05, 2.000180327751),
    (0.07, 1.999950577432),
    (0.1, 2.000003128381),
    (0.15, 1.999999992147),
    (500.0, 1.999999812631),
    (1000.0, 2.000004669679),
    (1500.0, 1.999988050626),
    (2000.0, 2.000028010692),
    (3000.0, 1.999950568717),
];

fn orca_radial(g: u8) -> RadialScheme {
    RadialScheme::GaussChebyshev {
        kind: ChebyshevKind::PerezJorda,
        mapping: RadialMapping::TreutlerM3,
        scale: RadialScale::OrcaXi { angular_grid: g },
    }
}

#[test]
fn defgrid2_radial_grid_reproduces_orca_electron_counts() {
    let n = AtomSizes::OrcaDefGrid { level: 2 }.sizes(6).unwrap().0;
    assert_eq!(n, 44);
    let rg = radial_grid(&orca_radial(4), n, 6);
    for (alpha, n_orca) in ORCA_C_DEFGRID2 {
        let c = 2.0 * (2.0 * alpha / PI).powf(1.5);
        let n_ours: f64 =
            rg.r.iter()
                .zip(&rg.w)
                .map(|(&r, &w)| w * c * (-2.0 * alpha * r * r).exp())
                .sum();
        // ORCA prints 12 decimals
        assert!(
            (n_ours - n_orca).abs() < 2e-11,
            "alpha {alpha}: ours {n_ours:.12} ORCA {n_orca:.12}"
        );
    }
}

#[test]
fn perez_jorda_m3_unit_scale_is_krack_koster() {
    for n in [10, 35, 44, 75] {
        let a = radial_grid(&RadialScheme::KrackKoster, n, 6);
        let b = radial_grid(
            &RadialScheme::GaussChebyshev {
                kind: ChebyshevKind::PerezJorda,
                mapping: RadialMapping::TreutlerM3,
                scale: RadialScale::Constant(1.0),
            },
            n,
            6,
        );
        // PySCF symmetrizes the nodes ((x - x[::-1]) / 2), which moves the innermost ones (r ~ 1e-6
        // bohr) by ~1e-9 relative
        for i in 0..n {
            assert!(
                (a.r[i] - b.r[i]).abs() <= 1e-8 * a.r[i] + 1e-14,
                "n {n} node {i}: {} vs {}",
                a.r[i],
                b.r[i]
            );
            assert!(
                (a.w[i] - b.w[i]).abs() <= 1e-8 * a.w[i].abs() + 1e-14,
                "n {n} weight {i}: {} vs {}",
                a.w[i],
                b.w[i]
            );
        }
    }
}

#[test]
fn defgrid_radial_shells_by_row() {
    // measured: DefGrid1/2/3 radial shells for H, C, Si, K (rows 1-4)
    for (z, nr) in [
        (1u32, [30, 35, 44]),
        (6, [38, 44, 53]),
        (14, [46, 54, 63]),
        (19, [55, 64, 73]),
    ] {
        for level in 1..=3u8 {
            assert_eq!(
                AtomSizes::OrcaDefGrid { level }.sizes(z).unwrap().0,
                nr[level as usize - 1],
                "Z {z} DefGrid{level}"
            );
        }
    }
    // H/He one AngularGrid lower: DefGrid2 = AngularGrid 4 (302), H 194
    assert_eq!(AtomSizes::OrcaDefGrid { level: 2 }.sizes(1).unwrap().1, 194);
    assert_eq!(AtomSizes::OrcaDefGrid { level: 2 }.sizes(8).unwrap().1, 302);
}

#[test]
fn tables_cover_h_to_kr() {
    for z in 1..=36u32 {
        for g in [3u8, 4, 6] {
            let xi = linus_grid::orca::grids()
                .xi(z, g)
                .unwrap_or_else(|| panic!("Z {z} AngularGrid {g}"));
            assert!((0.2..2.0).contains(&xi), "Z {z} g {g}: xi {xi}");
        }
        for level in 1..=3u8 {
            assert!(linus_grid::orca::grids().defgrid_nr(z, level).is_some());
        }
    }
    // beyond Kr: not measured, the row's shell count
    assert_eq!(linus_grid::orca::grids().xi(46, 4), None);
    assert_eq!(AtomSizes::OrcaDefGrid { level: 2 }.sizes(46).unwrap().0, 73);
}

#[test]
fn isolated_atom_point_counts_are_orcas() {
    // ORCA 6.0.0 '# of grid points (after initial pruning)' of one atom with a one-s-primitive
    // basis (no adaptive change); rows: H, C, F, Si, Cl, K, Fe, Br
    let orca: [(u32, [usize; 3]); 8] = [
        (1, [1512, 3334, 9244]),
        (6, [3352, 6136, 14578]),
        (9, [3352, 6136, 14578]),
        (14, [4256, 7368, 16314]),
        (17, [4256, 7368, 16314]),
        (19, [4922, 9152, 17186]),
        (26, [4922, 9152, 17186]),
        (35, [4922, 9152, 17186]),
    ];
    for (z, counts) in orca {
        for level in 1..=3u8 {
            if z == 1 && level == 3 {
                continue; // H's DefGrid3 region boundaries are not known (9604 points here)
            }
            let n = atomic_grid(&GridSpec::orca_defgrid(level), z)
                .unwrap()
                .npoints();
            assert_eq!(n, counts[level as usize - 1], "Z {z} DefGrid{level}");
        }
    }
}

fn la18() -> (Vec<u32>, Vec<[f64; 3]>) {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../bench/molecules/la18.xyz"
    );
    let text = std::fs::read_to_string(path).unwrap();
    let mut z = vec![];
    let mut xyz = vec![];
    for l in text.lines().skip(2) {
        let t: Vec<&str> = l.split_whitespace().collect();
        if t.len() < 4 {
            continue;
        }
        z.push(match t[0] {
            "H" => 1,
            "C" => 6,
            "O" => 8,
            s => panic!("{s}"),
        });
        let f = |i: usize| t[i].parse::<f64>().unwrap() / linus_core::units::BOHR_ANGSTROM;
        xyz.push([f(1), f(2), f(3)]);
    }
    (z, xyz)
}

#[test]
fn lactide_point_count_is_orcas() {
    // ORCA 6.0.0 DefGrid2, L-lactide (def2-SVP or def2-TZVP: no adaptive change): 85915 points
    let (z, xyz) = la18();
    let g = MolecularGrid::build(&z, &xyz, &GridSpec::orca_defgrid(2)).unwrap();
    assert_eq!(g.w.len(), 85915);
}

#[test]
fn cosx_grids_unchanged() {
    // GridSpec::cosx builds on the pre-measurement ORCA-like grid, unchanged
    for g in 1..=4u8 {
        let int_acc = linus_grid::presets::ORCA_COSX_INTACC[g as usize - 1];
        let n_ang = linus_grid::prune::orca_angular_max(g as usize);
        let want = GridSpec {
            radial: RadialScheme::GaussChebyshev {
                kind: ChebyshevKind::Second,
                mapping: RadialMapping::TreutlerM3,
                scale: RadialScale::TreutlerXi,
            },
            sizes: AtomSizes::OrcaLike {
                angular_grid: g,
                int_acc,
                b: 5.0,
                h_reduced: true,
            },
            overrides: (11u32..=86)
                .map(|z| {
                    (
                        z,
                        (15.0 * int_acc - 40.0 + 11.0 * linus_grid::elements::row(z) as f64).round()
                            as usize,
                        n_ang,
                    )
                })
                .collect(),
            pruning: Pruning::OrcaRegions,
            partition: PartitionScheme::Becke,
            size_adjust: SizeAdjust::Becke(RadiiTable::BraggPyscf),
            weight_cutoff: Some(1e-14),
            partition_cutoff: None,
            orientation: Orientation::Identity,
            batching: Batching {
                max_points: 128,
                per_atom: false,
            },
        };
        assert_eq!(GridSpec::cosx(g), want, "GridX {g}");
    }
}

#[test]
fn spec_round_trips_through_json() {
    let s = GridSpec::orca_defgrid(2);
    let t: GridSpec = serde_json::from_str(&serde_json::to_string(&s).unwrap()).unwrap();
    assert_eq!(s, t);
    // specs written before partition_cutoff existed still load
    let mut v: serde_json::Value = serde_json::to_value(GridSpec::linus(2)).unwrap();
    v.as_object_mut().unwrap().remove("partition_cutoff");
    assert_eq!(
        serde_json::from_value::<GridSpec>(v).unwrap(),
        GridSpec::linus(2)
    );
}

const ANG: f64 = 1.0 / linus_core::units::BOHR_ANGSTROM_PYSCF; // as tests/integration.rs

fn benzene() -> (Vec<u32>, Vec<[f64; 3]>) {
    let c = [
        [0.0, 1.396792, 0.0],
        [1.209657, 0.698396, 0.0],
        [1.209657, -0.698396, 0.0],
        [0.0, -1.396792, 0.0],
        [-1.209657, -0.698396, 0.0],
        [-1.209657, 0.698396, 0.0],
        [0.0, 2.484212, 0.0],
        [2.151390, 1.242106, 0.0],
        [2.151390, -1.242106, 0.0],
        [0.0, -2.484212, 0.0],
        [-2.151390, -1.242106, 0.0],
        [-2.151390, 1.242106, 0.0],
    ];
    (
        vec![6, 6, 6, 6, 6, 6, 1, 1, 1, 1, 1, 1],
        c.iter()
            .map(|p| [p[0] * ANG, p[1] * ANG, p[2] * ANG])
            .collect(),
    )
}

#[test]
fn orca_defgrid_benzene_point_counts() {
    // ORCA 6.0.0 on this benzene: DefGrid1/2/3 = 28216 / 55526 / 138550 points. DefGrid3
    // differs by H's region boundaries, which are not known for AngularGrid 6.
    // (Moved from tests/integration.rs.)
    let (z, xyz) = benzene();
    for (d, n) in [(1u8, 28216usize), (2, 55526)] {
        assert_eq!(
            MolecularGrid::build(&z, &xyz, &GridSpec::orca_defgrid(d))
                .unwrap()
                .len(),
            n,
            "DefGrid{d}"
        );
    }
}

/// ORCA's NL grids of water (bohr geometry of the probe runs): DefGrid3's equals the DefGrid1
/// XC grid (6340 points in ORCA's output); DefGrid1/2 approximations near ORCA's 2466/3858.
/// (Moved from src/nlc.rs.)
#[test]
fn orca_nl_water() {
    let z = [8, 1, 1];
    let xyz = [
        [0.0, 0.0, 0.2217],
        [0.0, 1.4309, -0.8867],
        [0.0, -1.4309, -0.8867],
    ];
    let n = |s: &GridSpec| MolecularGrid::build(&z, &xyz, s).unwrap().len();
    assert_eq!(n(&GridSpec::orca_nl(3)), 6340);
    let (n1, n2) = (n(&GridSpec::orca_nl(1)), n(&GridSpec::orca_nl(2)));
    assert!(
        (n1 as f64 / 2466.0 - 1.0).abs() < 0.15,
        "orca_nl(1): {n1} points, ORCA 2466"
    );
    assert!(
        (n2 as f64 / 3858.0 - 1.0).abs() < 0.15,
        "orca_nl(2): {n2} points, ORCA 3858"
    );
}
