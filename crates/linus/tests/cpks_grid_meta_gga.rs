//! The CPKS kernel grid of default Hessians: linus:2 for meta-GGAs, linus:1 otherwise
//! (GridChoice::CPKS_META_GGA, CPKS_DEFAULT; bench/results/2026-11-10-robust/cpksgrid_*).

use linus::{CalcOptions, CalcType, GridChoice, JkMethod, Session, Structure};

fn water() -> Structure {
    Structure::new(
        vec![8, 1, 1],
        vec![[0.0, 0.0, 0.0], [1.43, 0.0, 1.1], [-1.43, 0.0, 1.1]],
        0,
        1,
    )
    .unwrap()
}

fn cpks_grid(method: &str) -> Option<GridChoice> {
    let o = CalcOptions {
        method: Some(method.into()),
        basis: Some("def2-svp".into()),
        jk: JkMethod::Exact.into(),
        threads: Some(4),
        ..Default::default()
    };
    let rep = Session::new(water(), o)
        .unwrap()
        .run(CalcType::Hessian)
        .unwrap();
    // the CPKS kernel's grid as the run's settings report it (the SCF grid's when they agree)
    let g = rep.resolved.settings["grid"]["kernel"]
        .as_str()
        .unwrap()
        .parse::<GridChoice>();
    Some(g.unwrap()).filter(|g| Some(g) != rep.grid.as_ref())
}

#[test]
fn meta_ggas_take_the_finer_cpks_kernel_grid() {
    assert_eq!(cpks_grid("r2scan"), Some(GridChoice::CPKS_META_GGA));
    assert_eq!(cpks_grid("tpss"), Some(GridChoice::CPKS_META_GGA));
    assert_eq!(cpks_grid("b3lyp"), Some(GridChoice::CPKS_DEFAULT));
}
