//! Reproduction of pyscf.dft.Grids (fixtures from tests/fixtures/gen_pyscf_fixtures.py).

mod common;
use common::*;
use linus_grid::*;
use serde_json::Value;

fn spec_for(case: &Value) -> GridSpec {
    let s = &case["settings"];
    let level = s["level"].as_u64().unwrap() as u8;
    let mut spec = GridSpec::pyscf(level);
    spec.radial = match s["radi_method"].as_str().unwrap() {
        "treutler" => RadialScheme::PYSCF_TREUTLER,
        "becke" => RadialScheme::Becke,
        "mura_knowles" => RadialScheme::PYSCF_MURA_KNOWLES,
        "gauss_chebyshev" => RadialScheme::KrackKoster,
        "delley" => RadialScheme::PYSCF_DELLEY,
        x => panic!("{x}"),
    };
    spec.pruning = match s["prune"].as_str().unwrap() {
        "nwchem" => Pruning::NWChem,
        "sg1" => Pruning::SG1,
        "treutler" => Pruning::Treutler,
        "none" => Pruning::None,
        x => panic!("{x}"),
    };
    spec.partition = match s["scheme"].as_str().unwrap() {
        "becke" => PartitionScheme::Becke,
        "stratmann" => PartitionScheme::SSF,
        "lko" => PartitionScheme::LKO,
        x => panic!("{x}"),
    };
    spec.size_adjust = match s["adjust"].as_str().unwrap() {
        "treutler" => SizeAdjust::Treutler(RadiiTable::BraggPyscf),
        "becke" => SizeAdjust::Becke(RadiiTable::BraggPyscf),
        "none" => SizeAdjust::None,
        x => panic!("{x}"),
    };
    if let Some(ag) = s["atom_grid"].as_array() {
        spec.sizes = AtomSizes::Uniform {
            n_rad: ag[0].as_u64().unwrap() as usize,
            n_ang: ag[1].as_u64().unwrap() as usize,
        };
    }
    spec
}

#[test]
fn pyscf_grids_reproduced() {
    let fx = load("pyscf_grids.json");
    let mut worst = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    let mut report = String::new();
    let mut failures: Vec<String> = Vec::new();
    for case in fx["cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let spec = spec_for(case);
        let (z, xyz) = atoms(case);

        // atomic grids: radial nodes, weights, angular counts
        let mut max_rad_rel = 0.0f64;
        for (zs, ag) in case["atomic_grids"].as_object().unwrap() {
            let zz: u32 = zs.parse().unwrap();
            let ours = atomic_grid(&spec, zz).unwrap();
            let r = f64s(&ag["r"]);
            let w = f64s(&ag["w"]);
            let angs = usizes(&ag["angs"]);
            assert_eq!(ours.n_ang, angs, "{name}: angular counts for Z={zz}");
            assert_eq!(ours.r.len(), r.len());
            for i in 0..r.len() {
                max_rad_rel = max_rad_rel.max(((ours.r[i] - r[i]) / r[i]).abs());
                max_rad_rel = max_rad_rel.max(((ours.w[i] - w[i]) / w[i]).abs());
            }
        }
        assert!(
            max_rad_rel < 1e-14,
            "{name}: radial grid rel. deviation {max_rad_rel:e}"
        );

        let g = MolecularGrid::build(&z, &xyz, &spec).unwrap();
        assert_eq!(
            g.len(),
            case["npoints"].as_u64().unwrap() as usize,
            "{name}: point count"
        );
        // per-atom counts and moments
        let sums = per_atom_sums(&g);
        let mut max_sum_rel = 0.0f64;
        for (a, pa) in case["per_atom"].as_array().unwrap().iter().enumerate() {
            assert_eq!(
                sums[a][5] as u64,
                pa["n"].as_u64().unwrap(),
                "{name}: points on atom {a}"
            );
            let refs = [
                &pa["sum_w"],
                &pa["sum_wx"],
                &pa["sum_wy"],
                &pa["sum_wz"],
                &pa["sum_wr2"],
            ];
            let scale = pa["sum_w_abs"].as_f64().unwrap().max(1e-300);
            for k in 0..5 {
                let r = refs[k].as_f64().unwrap();
                let s = if k == 4 {
                    scale * 100.0
                } else if k == 0 {
                    scale
                } else {
                    scale * 10.0
                };
                max_sum_rel = max_sum_rel.max((sums[a][k] - r).abs() / s);
            }
        }
        // Becke's radial map puts the outermost heavy-atom shells at ~1000 bohr, where w r^2
        // moments are dominated by ill-conditioned partition values (see weight check below).
        let becke_radial = case["settings"]["radi_method"] == "becke";
        if max_sum_rel >= if becke_radial { 1e-10 } else { 1e-13 } {
            failures.push(format!(
                "{name}: per-atom moments rel. deviation {max_sum_rel:e}"
            ));
        }
        // sampled points
        let c = compare_sample(&g, &case["sample"], true);
        assert_eq!(
            c.unmatched, 0,
            "{name}: {} sample points not found",
            c.unmatched
        );
        assert_eq!(c.atom_mismatch, 0);
        assert!(
            c.max_dx <= 1e-12,
            "{name}: coordinate deviation {:e}",
            c.max_dx
        );
        // Weights must agree to 1e-12 (absolute or relative). The only exception is points
        // ~1000 bohr out (Becke radial map, S/Cl) with atomic weights ~1e8 whose partition
        // value P_A/Z ~ 1e-12 comes from the cancellation 0.5 - s(nu) in Becke's switching
        // function: PySCF's own value is then only defined to ~1e-17 in P_A/Z (one-ulp input
        // differences from numpy's vs libm's cos), so there we require |dw|/w_atomic < 1e-15.
        if !(c.max_dw_abs <= 1e-12 || c.max_dw_rel <= 1e-12 || c.max_dw_part <= 1e-15) {
            failures.push(format!(
                "{name}: weight deviation abs {:e} rel {:e}",
                c.max_dw_abs, c.max_dw_rel
            ));
        }
        report += &format!(
            "{name:55} n={:8} sample={:5} dx={:.1e} dw_abs={:.1e} dw_rel={:.1e} dw/w_at={:.1e} rad={:.1e} moments={:.1e}\n",
            g.len(),
            c.n,
            c.max_dx,
            c.max_dw_abs,
            c.max_dw_rel,
            c.max_dw_part,
            max_rad_rel,
            max_sum_rel
        );
        worst.0 = worst.0.max(c.max_dx);
        worst.1 = worst.1.max(c.max_dw_abs);
        worst.2 = worst.2.max(c.max_dw_rel);
        worst.3 = worst.3.max(max_sum_rel);
    }
    println!("{report}");
    println!(
        "worst: dx {:e} dw_abs {:e} dw_rel {:e} moments {:e}",
        worst.0, worst.1, worst.2, worst.3
    );
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
