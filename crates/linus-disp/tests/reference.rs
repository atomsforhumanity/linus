// SPDX-License-Identifier: LGPL-3.0-or-later
//! linus-disp against s-dftd3 1.7.0 and dftd4 4.3.1 (fixtures from tools/make_fixtures.py).

mod common;
use common::*;
use linus_disp::Order;

#[test]
fn against_sdftd3_and_dftd4() {
    let mut worst = [0.0f64; 5]; // dE, dG, dH (D3, analytic reference), dH (D4, numerical reference), rel dE
    eprintln!(
        "{:<10} {:<20} {:>9} {:>9} {:>9}",
        "molecule", "case", "dE", "dG", "dH"
    );
    for name in [
        "water",
        "hcn_cl",
        "la18",
        "cu-nh3-4",
        "zncl4",
        "cisplatin",
        "sni4",
        "gdcl3_h2o",
        "uf6_xe",
    ] {
        let fx = fixture(name);
        let (z, xyz) = (numbers(&fx), positions(&fx));
        let charge = fx["charge"].as_f64().unwrap();
        for case in fx["cases"].as_array().unwrap() {
            let model = case["model"].as_str().unwrap();
            let r = eval(
                model,
                case["functional"].as_str().unwrap(),
                case["atm"].as_bool().unwrap(),
                &z,
                &xyz,
                charge,
                Order::Hessian,
            );
            let e_ref = case["energy"].as_f64().unwrap();
            let de = (r.energy - e_ref).abs();
            let dg = max_abs_diff(&flat(&r.gradient), &floats(&case["gradient"]));
            let dh = max_abs_diff(r.hessian.as_ref().unwrap(), &floats(&case["hessian"]));
            eprintln!(
                "{:<10} {:<20} {:>9.1e} {:>9.1e} {:>9.1e}",
                name,
                case["label"].as_str().unwrap(),
                de,
                dg,
                dh
            );
            worst[0] = worst[0].max(de);
            worst[1] = worst[1].max(dg);
            if model == "d3bj" {
                worst[2] = worst[2].max(dh);
            } else {
                worst[3] = worst[3].max(dh);
            }
            worst[4] = worst[4].max(de / e_ref.abs());
        }
    }
    eprintln!(
        "worst: dE {:.1e} (rel {:.1e}), dG {:.1e}, dH(D3) {:.1e}, dH(D4 vs dftd4 FD) {:.1e}",
        worst[0], worst[4], worst[1], worst[2], worst[3]
    );
    assert!(worst[0] <= 1e-12, "energy");
    assert!(worst[1] <= 1e-11, "gradient");
    assert!(worst[2] <= 1e-11, "D3 Hessian vs s-dftd3 analytic Hessian");
    // dftd4's two-body Hessian is a central difference with step 1e-4 bohr
    assert!(worst[3] <= 1e-8, "D4 Hessian vs dftd4 numerical Hessian");
}
