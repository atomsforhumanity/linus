//! Reproduction of Psi4 1.11's DFT grid (fixtures from tests/fixtures/gen_psi4_fixtures.py).

mod common;
use common::*;
use linus_grid::*;

#[test]
fn psi4_grids_reproduced() {
    let fx = load("psi4_grids.json");
    let mut report = String::new();
    let mut failures = Vec::new();
    for case in fx["cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let (z, xyz) = atoms(case);
        let o: Vec<Vec<f64>> = case["orientation"]
            .as_array()
            .unwrap()
            .iter()
            .map(f64s)
            .collect();
        let m = [
            [o[0][0], o[0][1], o[0][2]],
            [o[1][0], o[1][1], o[1][2]],
            [o[2][0], o[2][1], o[2][2]],
        ];
        let mut spec = GridSpec::psi4_default();
        spec.orientation = Orientation::Matrix(m);
        let opts = &case["options"];
        if opts["dft_nuclear_scheme"] == "BECKE" {
            spec.size_adjust = SizeAdjust::Becke(RadiiTable::BraggSlaterPsi4);
        }
        if let Some(n) = opts["dft_radial_points"].as_u64() {
            let na = opts["dft_spherical_points"].as_u64().unwrap() as usize;
            spec.sizes = AtomSizes::Uniform {
                n_rad: n as usize,
                n_ang: na,
            };
        }
        let g = MolecularGrid::build(&z, &xyz, &spec).unwrap();
        let n_ref = case["npoints"].as_u64().unwrap() as usize;
        if g.len() != n_ref {
            failures.push(format!("{name}: {} points, Psi4 {}", g.len(), n_ref));
        }
        let sums = per_atom_sums(&g);
        let total: f64 = sums.iter().map(|s| s[0]).sum();
        let r = case["sum_w"].as_f64().unwrap();
        let max_sum_rel = ((total - r) / r).abs();
        let c = compare_sample(&g, &case["sample"], false);
        if c.unmatched > 0
            || c.max_dx > 1e-12
            || !(c.max_dw_abs <= 1e-12 || c.max_dw_rel <= 1e-12)
            || max_sum_rel > 1e-13
        {
            failures.push(format!(
                "{name}: unmatched {} dx {:e} dw_abs {:e} dw_rel {:e} sum_w rel {:e}",
                c.unmatched, c.max_dx, c.max_dw_abs, c.max_dw_rel, max_sum_rel
            ));
        }
        report +=
            &format!(
            "{name:50} n={:7} (psi4 {:7}) dx={:.1e} dw_abs={:.1e} dw_rel={:.1e} sum_w_rel={:.1e}\n",
            g.len(), n_ref, c.max_dx, c.max_dw_abs, c.max_dw_rel, max_sum_rel
        );
    }
    println!("{report}");
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
