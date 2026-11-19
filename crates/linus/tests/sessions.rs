//! A session's lineages (`Session::compute`, crate::driver::lineage): each chain of requests
//! continues from its own last success, bit for bit as if it ran alone; a failed request leaves
//! its lineage as it was; a request farther than the guess distance, after a reset or without
//! history starts fresh and probes; seeds start a lineage from another's state; `commit: false`
//! stores nothing. The record says where every SCF started.

use linus::driver::session::Request;
use linus::driver::{GuessSource, LineageKey, GUESS_DISTANCE};
use linus::{CalcOptions, CalcType, GridChoice, Report, Session, Structure};
use linus_scf::ProbeSkip;
use serde_json::json;

/// Water without symmetry (bohr).
fn water() -> Vec<[f64; 3]> {
    vec![
        [0.0, 0.0, 0.223],
        [0.038, 1.427, -0.890],
        [-0.019, -1.436, -0.879],
    ]
}

/// `x` with atom `i` moved by `d` (bohr) along z.
fn moved(x: &[[f64; 3]], i: usize, d: f64) -> Vec<[f64; 3]> {
    let mut y = x.to_vec();
    y[i][2] += d;
    y
}

/// The settings of the tests' sessions: `method` in STO-3G (DFT on a coarse grid).
fn opts(method: &str) -> CalcOptions {
    let mut o = CalcOptions {
        method: Some(method.into()),
        basis: Some("sto-3g".into()),
        ..Default::default()
    };
    if method != "hf" {
        o.grid.xc = Some(GridChoice::Pyscf { level: 1 });
    }
    o
}

fn mol() -> Structure {
    Structure::new(vec![8, 1, 1], water(), 0, 1).unwrap()
}

fn session(method: &str) -> Session {
    Session::new(mol(), opts(method)).unwrap()
}

fn req(x: &[[f64; 3]], lineage: serde_json::Value) -> Request {
    Request {
        positions: Some(x.to_vec()),
        lineage: LineageKey::new(&lineage),
        ..Request::new(CalcType::Gradient)
    }
}

/// The numbers of a result, for bitwise comparisons: the energy, the gradient and the SCF
/// iterations.
fn numbers(r: &Report) -> (u64, Vec<u64>, usize) {
    (
        r.energy.to_bits(),
        r.gradient
            .as_ref()
            .unwrap()
            .iter()
            .flatten()
            .map(|g| g.to_bits())
            .collect(),
        r.iterations,
    )
}

/// Two chains interleaved in one session give the results of each alone, bit for bit, and a
/// session's default lineage gives the direct calls' results (set_positions + run).
#[test]
fn lineages_are_isolated() {
    let x = water();
    let a: Vec<_> = (1..4).map(|k| moved(&x, 0, 0.02 * k as f64)).collect();
    let b: Vec<_> = (1..4).map(|k| moved(&x, 1, -0.03 * k as f64)).collect();
    let alone = |steps: &[Vec<[f64; 3]>], key: serde_json::Value| {
        let mut s = session("b3lyp");
        steps
            .iter()
            .map(|p| numbers(&s.compute(&req(p, key.clone())).unwrap()))
            .collect::<Vec<_>>()
    };
    let (ra, rb) = (alone(&a, json!("a")), alone(&b, json!(["b", 1])));
    let mut s = session("b3lyp");
    for k in 0..3 {
        assert_eq!(
            numbers(&s.compute(&req(&a[k], json!("a"))).unwrap()),
            ra[k],
            "a {k}"
        );
        assert_eq!(
            numbers(&s.compute(&req(&b[k], json!(["b", 1]))).unwrap()),
            rb[k],
            "b {k}"
        );
    }
    assert_eq!(
        s.lineages(),
        vec![
            LineageKey::new(&json!("a")),
            LineageKey::new(&json!(["b", 1]))
        ]
    );
    // the default lineage is the direct calls' chain
    let mut d = session("b3lyp");
    for (k, p) in a.iter().enumerate() {
        d.set_positions(p).unwrap();
        assert_eq!(
            numbers(&d.run(CalcType::Gradient).unwrap()),
            ra[k],
            "direct {k}"
        );
    }
}

/// Where each SCF started, in the record; the probes run at fresh starts only.
#[test]
fn the_record_says_where_the_scf_started() {
    let x = water();
    let mut s = session("b3lyp");
    let first = s.compute(&req(&x, json!(null))).unwrap();
    let (g, rec) = (&first.resolved.record.scf.guess, &first.resolved.record);
    assert!(!rec.warm_start && g.source == GuessSource::Sad && g.lineage.is_null());
    assert_eq!((g.distance, g.from_lineage.clone()), (None, None));
    let skipped = |r: &Report| r.resolved.record.scf.closed_shell_probe.unwrap().skipped;
    assert_eq!(skipped(&first), Some(ProbeSkip::GapAboveThreshold));
    // a step: the orbitals made orthonormal at the new positions, no probes
    let near = s.compute(&req(&moved(&x, 0, 0.05), json!(null))).unwrap();
    let g = &near.resolved.record.scf.guess;
    assert!(near.resolved.record.warm_start && g.source == GuessSource::Lineage);
    assert!(g.orthonormalized && (g.distance.unwrap() - 0.05).abs() < 1e-12);
    assert_eq!(g.from_lineage, Some(json!(null)));
    assert_eq!(skipped(&near), Some(ProbeSkip::LaterGeometry));
    assert!(near.iterations < first.iterations);
    // the same positions again: the last densities, a continuation
    let again = s.compute(&req(&moved(&x, 0, 0.05), json!(null))).unwrap();
    let g = &again.resolved.record.scf.guess;
    assert!(g.source == GuessSource::Lineage && !g.orthonormalized && g.distance == Some(0.0));
    assert_eq!(skipped(&again), Some(ProbeSkip::LaterGeometry));
    // a jump beyond the guess distance: a fresh start, probed
    let far = moved(&x, 2, -(GUESS_DISTANCE + 0.2));
    let jump = s.compute(&req(&far, json!(null))).unwrap();
    let g = &jump.resolved.record.scf.guess;
    assert!(!jump.resolved.record.warm_start && g.source == GuessSource::Sad);
    assert_eq!(
        serde_json::to_value(g).unwrap()["reason"],
        json!("distance")
    );
    assert_eq!(skipped(&jump), Some(ProbeSkip::GapAboveThreshold));
    // a reset: fresh again, the lineage gone
    s.reset(&LineageKey::default());
    assert!(s.lineages().is_empty());
    let reset = s.compute(&req(&far, json!(null))).unwrap();
    assert_eq!(
        serde_json::to_value(&reset.resolved.record.scf.guess).unwrap()["reason"],
        json!("no_history")
    );
    // the record serializes with the shared key at its top
    let v = reset.to_json();
    assert_eq!(v["resolved"]["record"]["warm_start"], json!(false));
    assert_eq!(
        v["resolved"]["record"]["scf"]["guess"]["source"],
        json!("sad")
    );
}

/// A failed request leaves its lineage as it was: a Hessian whose SCF converged but whose
/// response cannot reach its tolerance, and positions that are no geometry.
#[test]
fn a_failed_request_commits_nothing() {
    let x = water();
    let (x1, x2, x3) = (moved(&x, 0, 0.03), moved(&x, 0, 0.06), moved(&x, 1, 0.04));
    let opts = || {
        let mut o = opts("hf");
        o.response.tol = Some(linus::driver::options::Auto::Value(1e-300));
        o
    };
    let mut s = Session::new(mol(), opts()).unwrap();
    s.compute(&req(&x1, json!(null))).unwrap();
    let hess = Request {
        calctype: CalcType::Hessian,
        ..req(&x2, json!(null))
    };
    // its SCF converges (the Hessian's), the response cannot reach 1e-300: an unconverged
    // response, never a non-finite Hessian (its residual once turned NaN and its maximum read 0)
    let failed = s.compute(&hess).unwrap_err();
    assert!(s.last_here().is_some(), "{failed}");
    assert!(
        matches!(failed, linus::LinusError::ResponseNotConverged(_)),
        "{failed:?}"
    );
    let bad = req(&[x[0], x[0], x[2]], json!(null));
    assert!(s.compute(&bad).unwrap_err().is_input());
    let after = s.compute(&req(&x3, json!(null))).unwrap();
    let mut t = Session::new(mol(), opts()).unwrap();
    t.compute(&req(&x1, json!(null))).unwrap();
    let want = t.compute(&req(&x3, json!(null))).unwrap();
    assert_eq!(numbers(&after), numbers(&want));
    assert_eq!(
        after.resolved.record.scf.guess.distance,
        want.resolved.record.scf.guess.distance
    );
}

/// Seeds start a lineage without state from another's (the first with one); a lineage's own
/// state wins; a seed without state is a fresh start.
#[test]
fn seeds_start_a_lineage_from_another() {
    let x = water();
    let ts = moved(&x, 0, 0.02);
    let mut s = session("b3lyp");
    s.compute(&req(&ts, json!("ts"))).unwrap();
    let branch = |s: &mut Session, key: &str, seed: Vec<serde_json::Value>, d: f64| {
        let r = Request {
            seed: seed.iter().map(LineageKey::new).collect(),
            ..req(&moved(&ts, 1, d), json!(key))
        };
        s.compute(&r).unwrap()
    };
    let fwd = branch(&mut s, "fwd", vec![json!("none"), json!("ts")], 0.04);
    let g = &fwd.resolved.record.scf.guess;
    assert!(g.source == GuessSource::Lineage && g.from_lineage == Some(json!("ts")));
    assert_eq!(g.lineage, json!("fwd"));
    // its own state now wins over the seed
    let fwd2 = branch(&mut s, "fwd", vec![json!("ts")], 0.08);
    assert_eq!(
        fwd2.resolved.record.scf.guess.from_lineage,
        Some(json!("fwd"))
    );
    // a seed without state: fresh
    let lone = branch(&mut s, "lone", vec![json!("none")], -0.04);
    assert_eq!(lone.resolved.record.scf.guess.source, GuessSource::Sad);
    // a seeded start is the seed lineage's continuation, bit for bit
    let mut t = session("b3lyp");
    t.compute(&req(&ts, json!("ts"))).unwrap();
    let cont = t.compute(&req(&moved(&ts, 1, 0.04), json!("ts"))).unwrap();
    assert_eq!(numbers(&fwd), numbers(&cont));
}

/// `commit: false` stores nothing: in [A, B (commit false), C, D (commit false)] of one lineage,
/// B starts from A's state and D from C's, each bit for bit as without the uncommitted ones.
#[test]
fn uncommitted_requests_read_the_state_at_their_place() {
    let x = water();
    let p: Vec<_> = (1..=5).map(|k| moved(&x, 0, 0.015 * k as f64)).collect();
    let r = |k: usize, commit: bool| Request {
        commit,
        ..req(&p[k], json!(null))
    };
    let mut s = session("b3lyp");
    let a = s.compute(&r(0, true)).unwrap();
    let b = s.compute(&r(1, false)).unwrap();
    let c = s.compute(&r(2, true)).unwrap();
    let d = s.compute(&r(3, false)).unwrap();
    let e = s.compute(&r(4, true)).unwrap();
    let chain = |ks: &[usize]| {
        let mut t = session("b3lyp");
        ks.iter()
            .map(|&k| numbers(&t.compute(&r(k, true)).unwrap()))
            .collect::<Vec<_>>()
    };
    assert_eq!(chain(&[0, 1]), vec![numbers(&a), numbers(&b)]);
    assert_eq!(
        chain(&[0, 2, 3]),
        vec![numbers(&a), numbers(&c), numbers(&d)]
    );
    assert_eq!(chain(&[0, 2, 4])[2], numbers(&e));
    // the distances of B and D are from A's and C's positions
    for (rep, (to, from)) in [(&b, (1, 0)), (&d, (3, 2))] {
        let want = (p[to][0][2] - p[from][0][2]).abs();
        let got = rep.resolved.record.scf.guess.distance.unwrap();
        assert!((got - want).abs() < 1e-14, "{got} vs {want}");
    }
}

/// A session that may run Hessians accepts the Hessians' settings; each request's record says
/// whether such a setting took effect in it.
#[test]
fn the_record_says_which_settings_applied() {
    let mut o = opts("hf");
    o.freq = linus::driver::options::FreqSettings::from_map(
        serde_json::from_value(json!({"temperature": [350.0]})).unwrap(),
    )
    .unwrap();
    let mut s = Session::new(mol(), o.clone()).unwrap();
    let applied = |r: &Report| r.resolved.record.settings_applied.clone();
    let g = s.compute(&req(&water(), json!(null))).unwrap();
    assert_eq!(
        applied(&g),
        [("freq.temperature".to_string(), false)].into()
    );
    let h = s
        .compute(&Request {
            calctype: CalcType::Hessian,
            ..req(&water(), json!(null))
        })
        .unwrap();
    assert_eq!(applied(&h), [("freq.temperature".to_string(), true)].into());
    // declared for gradients only, the setting is an error; a one-shot lists nothing
    let mut one = Session::new(mol(), o).unwrap();
    let e = one
        .declare(linus::CalcTypes::of(CalcType::Gradient, false))
        .unwrap_err();
    assert!(e.to_string().contains("freq.temperature"), "{e}");
    let mut one = Session::new(mol(), opts("hf")).unwrap();
    one.declare(linus::CalcTypes::of(CalcType::Gradient, false))
        .unwrap();
    let r = one.run(CalcType::Gradient).unwrap();
    assert!(r.resolved.record.settings_applied.is_empty());
    assert!(r.to_json()["resolved"]["record"]
        .get("settings_applied")
        .is_none());
}
