//! df_bench <xyz> <basis> <aux> [--charge q] [--threads n] [--reps r] [--j-only] [--direct]
//!          [--budget-gb x] [--compare] [--omega w] [--precision f64|f48|f40|f32] [--lr stored|one-sided]
//!
//! Times DfJk setup and per-call J/K on core-Hamiltonian occupied orbitals (closed shell). With
//! --omega w: J + K and K_LR (DfJk::exchange_long_range) per call (t_jk_klr) and the one-pass a K + b K_LR of
//! set_exchange_operator with wB97X's a, b (t_jk_combined). --compare: DfJk's factored
//! exchange of an indefinite density X Y^T + Y X^T against its eigendecomposition path. Prints
//! JSON. (The dense reference DenseDfJk, --old, and the seminumerical K, --snk, were removed; the
//! unit tests compare DfJk against the dense code.)
use faer::{Mat, Par};
use linus::JkBuilder;
use linus::Structure;
use linus_base::linalg::{canonical_orthogonalizer, mat_from_rowmajor, max_abs, mm};
use linus_base::terms::JkRequest;
use linus_ints::{basis::BasisDef, BasisSet, PairOptions, PairSet};
use linus_jk::df::{DfConfig, DfJk};
use std::time::Instant;

fn median(mut v: Vec<f64>) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn rss_gb() -> f64 {
    let s = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
    for l in s.lines() {
        if let Some(r) = l.strip_prefix("VmHWM:") {
            return r
                .trim()
                .trim_end_matches("kB")
                .trim()
                .parse::<f64>()
                .unwrap_or(0.0)
                / 1048576.0;
        }
    }
    0.0
}

/// FNV-1a over the bit patterns (bitwise reproducibility checks).
fn hash(m: &Mat<f64>) -> String {
    let mut h: u64 = 0xcbf29ce484222325;
    for j in 0..m.ncols() {
        for i in 0..m.nrows() {
            for byte in m[(i, j)].to_bits().to_le_bytes() {
                h ^= byte as u64;
                h = h.wrapping_mul(0x100000001b3);
            }
        }
    }
    format!("{h:016x}")
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let get = |k: &str| a.iter().position(|x| x == k).map(|i| a[i + 1].clone());
    let has = |k: &str| a.iter().any(|x| x == k);
    let charge: i32 = get("--charge").map(|s| s.parse().unwrap()).unwrap_or(0);
    let threads: usize = get("--threads").map(|s| s.parse().unwrap()).unwrap_or(1);
    let reps: usize = get("--reps").map(|s| s.parse().unwrap()).unwrap_or(3);
    let j_only = has("--j-only");
    let mol = Structure::from_xyz_str(&std::fs::read_to_string(&a[1]).unwrap(), charge, 1).unwrap();
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .unwrap();
    let par = if threads > 1 {
        Par::rayon(threads)
    } else {
        Par::Seq
    };
    let mut opts = DfConfig::default();
    if let Some(b) = get("--budget-gb") {
        opts.mem_budget = Some((b.parse::<f64>().unwrap() * 1073741824.0) as usize);
    }
    if let Some(t) = get("--pair-thresh") {
        opts.thresholds.pair = t.parse().unwrap();
    }
    if let Some(t) = get("--prim-thresh") {
        opts.thresholds.prim = t.parse().unwrap();
    }
    if let Some(t) = get("--k-block") {
        opts.k_block = t.parse().unwrap();
    }
    if let Some(t) = get("--k-groups") {
        opts.k_groups = t.parse().unwrap();
    }
    opts.force_direct = has("--direct");
    match get("--precision").as_deref() {
        Some("f64") => opts.precision = linus_jk::df::DfPrecision::F64,
        Some("f48") => opts.precision = linus_jk::df::DfPrecision::F48,
        Some("f40") => opts.precision = linus_jk::df::DfPrecision::F40,
        Some("f32") => opts.precision = linus_jk::df::DfPrecision::F32,
        _ => {}
    }
    match get("--lr").as_deref() {
        Some("stored") => opts.long_range = linus_jk::df::DfLongRange::Stored,
        Some("one-sided") => opts.long_range = linus_jk::df::DfLongRange::OneSided,
        _ => {}
    }
    if let Some(w) = get("--omega") {
        // prepare the long-range side at setup (storage chosen for range-separated exchange)
        opts.omega = Some(w.parse().unwrap());
    }
    pool.install(|| {
        let def = BasisDef::load(&a[2], None).unwrap();
        let basis = BasisSet::new(&def, &mol.z, &mol.positions, true).unwrap();
        let adef = BasisDef::load(&a[3], None).unwrap();
        let aux = BasisSet::new(&adef, &mol.z, &mol.positions, true).unwrap();
        let n = basis.nbf;
        let pairs = PairSet::build(&basis, PairOptions::default());
        // core-Hamiltonian occupied orbitals
        let (s, t, v) = linus_dev::one_electron(&basis, &pairs, &mol.point_charges());
        let h = mat_from_rowmajor(n, n, &t) + mat_from_rowmajor(n, n, &v);
        let x = canonical_orthogonalizer(mat_from_rowmajor(n, n, &s).as_ref(), 1e-8, par);
        let (_, c) = linus_scf::diagonalize(h.as_ref(), x.as_ref(), par);
        let nocc = (mol.nelectron() / 2) as usize;
        let cocc: Mat<f64> = c.subcols(0, nocc).to_owned();
        let d = mm(cocc.as_ref(), cocc.transpose(), par);
        let mut out = serde_json::json!({"xyz": a[1].rsplit('/').next().unwrap(), "basis": a[2], "aux": a[3], "threads": threads,
            "nbf": n, "naux": aux.nbf, "nocc": nocc, "j_only": j_only});
        let t0 = Instant::now();
        let mut df = DfJk::new(&basis, &pairs, &aux, !j_only, opts.clone(), par);
        let t_setup = t0.elapsed().as_secs_f64();
        let st = df.stats.clone();
        out["setup"] = serde_json::json!({"t_total": t_setup, "t_schwarz": st.t_schwarz, "t_prep": st.t_prep, "t_metric": st.t_metric, "t_metric_ints": st.t_metric_ints,
            "t_alloc": st.t_alloc, "t_ints": st.t_ints, "t_fit": st.t_fit, "mode": st.plan.name(), "rows": st.nrows, "rows_dense_packed": n * (n + 1) / 2,
            "shell_pairs": st.nshell_pairs, "shell_pairs_kept": st.nshell_pairs_kept, "gb": st.bytes as f64 / 1073741824.0, "budget_gb": st.budget as f64 / 1073741824.0, "panel_elems": st.panel_elems, "shell_panel_elems": st.shell_panel_elems, "distinct_lists": st.distinct_lists, "nshell": basis.shells.len(), "natom": mol.natom()});
        let fs = [cocc.clone()];
        let req = JkRequest { dens: std::slice::from_ref(&d), factors: Some(linus_base::terms::DensityFactors::Symmetric(&fs)), want_j: true, want_k: !j_only };
        let mut tj = vec![];
        let mut tk = vec![];
        let mut tt = vec![];
        let mut last = vec![];
        let mut res = None;
        for _ in 0..reps {
            let t = Instant::now();
            let r = df.compute(&req, par).unwrap();
            tt.push(t.elapsed().as_secs_f64());
            for (k, v) in &df.stats.last {
                if k == "j" {
                    tj.push(*v);
                }
                if k == "k" {
                    tk.push(*v);
                }
            }
            last = df.stats.last.clone();
            res = Some(r);
        }
        out["iter"] = serde_json::json!({"t_jk": median(tt), "t_j": median(tj), "t_k": median(tk),
            "last": serde_json::Value::Object(last.iter().map(|(k, v)| (k.clone(), serde_json::json!(v))).collect())});
        let res = res.unwrap();
        // energy traces at the core-Hamiltonian density (AO-ordering independent checks)
        out["trace"] = serde_json::json!({"DJ": linus_base::linalg::dot(d.as_ref(), res.j[0].as_ref())});
        out["hash"] = serde_json::json!({"J": hash(&res.j[0])});
        if !j_only {
            out["hash"]["K"] = serde_json::json!(hash(&res.k[0]));
        }
        if !j_only {
            out["trace"]["DK"] = serde_json::json!(linus_base::linalg::dot(d.as_ref(), res.k[0].as_ref()));
        }
        if let Some(w) = get("--omega") {
            let w: f64 = w.parse().unwrap();
            let t = Instant::now();
            let fac = [linus_jk::df::Factored::Psd(fs[0].as_ref())];
            let (rk, rklr) = df.exchange_long_range(&fac, true, Some(w), par);
            let t_first = t.elapsed().as_secs_f64();
            let mut tl = vec![];
            for _ in 0..reps {
                let t = Instant::now();
                let _ = df.compute(&JkRequest { dens: std::slice::from_ref(&d), factors: Some(linus_base::terms::DensityFactors::Symmetric(&fs)), want_j: true, want_k: false }, par).unwrap();
                let _ = df.exchange_long_range(&fac, true, Some(w), par);
                tl.push(t.elapsed().as_secs_f64());
            }
            out["trace"]["DKlr"] = serde_json::json!(linus_base::linalg::dot(d.as_ref(), rklr[0].as_ref()));
            out["hash"]["Klr"] = serde_json::json!(hash(&rklr[0]));
            out["omega"] = serde_json::json!({"omega": w, "t_first_incl_setup": t_first, "t_lr_setup": df.stats.t_lr_setup, "t_jk_klr": median(tl),
                "gb_lr": df.stats.bytes_lr as f64 / 1073741824.0,
                "max_asym_klr": max_abs((&rklr[0] - rklr[0].transpose()).as_ref())});
            // one-pass a K + b K_LR (wB97X coefficients) through set_exchange_operator
            let (ca, cb) = (0.157706, 0.842294);
            let want = &rk[0] * ca + &rklr[0] * cb;
            assert!(df.set_exchange_operator(linus_ints::operator::Operator::Attenuated { coulomb: ca, long_range: cb, omega: w }));
            let mut tc = vec![];
            let mut kc = None;
            for _ in 0..reps {
                let t = Instant::now();
                let o = df.compute(&JkRequest { dens: std::slice::from_ref(&d), factors: Some(linus_base::terms::DensityFactors::Symmetric(&fs)), want_j: true, want_k: true }, par).unwrap();
                tc.push(t.elapsed().as_secs_f64());
                kc = Some(o.k[0].clone());
            }
            out["omega"]["t_jk_combined"] = serde_json::json!(median(tc));
            out["omega"]["max_diff_combined"] = serde_json::json!(max_abs((&kc.unwrap() - &want).as_ref()));
        }
        if has("--compare") && !j_only {
            // DfJk's factored exchange of an indefinite density D' = X Y^T + Y X^T against its
            // eigendecomposition path
            let xm = cocc.subcols(0, nocc / 2).to_owned();
            let ym = cocc.subcols(nocc / 2, nocc - nocc / 2).to_owned();
            let ym = Mat::from_fn(n, xm.ncols(), |i, j| if j < ym.ncols() { ym[(i, j)] } else { 0.0 });
            let dp = mm(xm.as_ref(), ym.transpose(), par) + mm(ym.as_ref(), xm.transpose(), par);
            let ks = df.exchange(&[linus_jk::df::Factored::SymPair(xm.as_ref(), ym.as_ref())], par);
            let kr = df.compute(&JkRequest { dens: std::slice::from_ref(&dp), factors: None, want_j: false, want_k: true }, par).unwrap();
            out["compare"] = serde_json::json!({"max_dK_sympair_vs_eig": max_abs((&ks[0] - &kr.k[0]).as_ref()), "max_K": max_abs(kr.k[0].as_ref())});
        }
        out["peak_rss_gb"] = serde_json::json!(rss_gb());
        println!("{}", out);
    });
}
