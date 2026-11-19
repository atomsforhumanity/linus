//! Time J/K builds on a fixed density.
//!
//! jk_bench <xyz> <basis> [--dens file.bin] [--builder new|incore] [--threads n]
//!          [--repeat r] [--thresh t] [--no-j] [--no-k] [--out prefix] [--profile]
//!          [--omega w [--cam-alpha a --cam-beta b]] [--jengine] [--prim-factor f]
//! With --omega, K uses alpha/r + beta erfc(w r)/r (default alpha 0, beta 1: short range).
//!
//! The density file holds nbf*nbf little-endian f64 (row-major, PySCF AO order); without it
//! the core-Hamiltonian guess density is used. `--out prefix` writes prefix.J.bin and
//! prefix.K.bin (last repeat). Prints one JSON line.
use faer::{Mat, Par};
use linus::JkBuilder;
use linus::Structure;
use linus_base::linalg::{eigh, mat_from_rowmajor};
use linus_base::terms::JkRequest;
use linus_ints::{basis::BasisDef, BasisSet, PairOptions, PairSet};
use linus_jk::exact::FastJk;
use linus_jk::IncoreJk;

fn read_bin(path: &str) -> Vec<f64> {
    let b = std::fs::read(path).unwrap();
    b.as_chunks::<8>()
        .0
        .iter()
        .map(|c| f64::from_le_bytes(*c))
        .collect()
}

fn write_bin(path: &str, m: &Mat<f64>) {
    let n = m.nrows();
    let mut v = Vec::with_capacity(n * n * 8);
    for i in 0..n {
        for j in 0..n {
            v.extend_from_slice(&m[(i, j)].to_le_bytes());
        }
    }
    std::fs::write(path, v).unwrap();
}

struct Dummy;
impl JkBuilder for Dummy {
    fn name(&self) -> String {
        String::new()
    }
    fn compute(
        &mut self,
        _: &JkRequest,
        _: Par,
    ) -> Result<linus_base::terms::JkOutput, linus::LinusError> {
        unreachable!()
    }
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let get = |k: &str| a.iter().position(|x| x == k).map(|i| a[i + 1].clone());
    let has = |k: &str| a.iter().any(|x| x == k);
    let threads: usize = get("--threads").map(|s| s.parse().unwrap()).unwrap_or(1);
    let repeat: usize = get("--repeat").map(|s| s.parse().unwrap()).unwrap_or(1);
    let thresh: f64 = get("--thresh").map(|s| s.parse().unwrap()).unwrap_or(1e-12);
    let builder = get("--builder").unwrap_or("new".into());
    let mol = Structure::from_xyz_str(&std::fs::read_to_string(&a[1]).unwrap(), 0, 1).unwrap();
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .unwrap();
    let par = if threads > 1 {
        Par::rayon(threads)
    } else {
        Par::Seq
    };
    pool.install(|| {
        let def = BasisDef::load(&a[2], None).unwrap();
        let basis = BasisSet::new(&def, &mol.z, &mol.positions, true).unwrap();
        let n = basis.nbf;
        let d = match get("--dens") {
            Some(f) => mat_from_rowmajor(n, n, &read_bin(&f)),
            None => {
                // core guess: D = C_occ C_occ^T from h in the Loewdin basis
                let pairs = PairSet::build(&basis, PairOptions::default());
                let (s, t, v) = linus_dev::one_electron(&basis, &pairs, &mol.point_charges());
                let s = mat_from_rowmajor(n, n, &s);
                let h = mat_from_rowmajor(n, n, &t) + mat_from_rowmajor(n, n, &v);
                let (sw, su) = eigh(s.as_ref(), par);
                let x = Mat::from_fn(n, n, |i, j| su[(i, j)] / sw[j].sqrt());
                let hp = x.transpose() * &h * &x;
                let (_, cu) = eigh(hp.as_ref(), par);
                let c = &x * &cu;
                let nocc = (mol.nelectron() / 2) as usize;
                Mat::from_fn(n, n, |i, j| (0..nocc).map(|k| c[(i, k)] * c[(j, k)]).sum())
            }
        };
        let t0 = std::time::Instant::now();
        let mut fast: Option<FastJk> = None;
        let mut jk: Box<dyn JkBuilder> = match builder.as_str() {
            "incore" => Box::new(IncoreJk::new(&basis, &PairSet::build(&basis, PairOptions::default()))),
            "new" => {
                let mut b = FastJk::with_options(
                    &basis,
                    linus_ints::JkOptions {
                        thresh,
                        prim_factor: get("--prim-factor")
                            .map_or(linus_ints::eri4c::jk::PRIM_FACTOR, |f| f.parse().unwrap()),
                        profile: has("--profile"),
                        jengine: has("--jengine"),
                        ..Default::default()
                    },
                );
                if let Some(w) = get("--omega") {
                    let al: f64 = get("--cam-alpha").map(|s| s.parse().unwrap()).unwrap_or(0.0);
                    let be: f64 = get("--cam-beta").map(|s| s.parse().unwrap()).unwrap_or(1.0);
                    b = b.with_k_operator(linus_ints::operator::Operator::from_libxc(
                        al,
                        be,
                        w.parse().unwrap(),
                    ));
                }
                fast = Some(b);
                Box::new(Dummy)
            }
            _ => panic!("unknown builder {builder}"),
        };
        let t_setup = t0.elapsed().as_secs_f64();
        let dens = vec![d];
        let req = JkRequest { dens: &dens, factors: None, want_j: !has("--no-j"), want_k: !has("--no-k") };
        let mut times = Vec::new();
        let mut last = None;
        for _ in 0..repeat {
            let t = std::time::Instant::now();
            let out = match fast.as_mut() {
                Some(f) => f.compute(&req, par).unwrap(),
                None => jk.compute(&req, par).unwrap(),
            };
            times.push(t.elapsed().as_secs_f64());
            last = Some(out);
        }
        let out = last.unwrap();
        if let Some(p) = get("--out") {
            if let Some(j) = out.j.first() {
                write_bin(&format!("{p}.J.bin"), j);
            }
            if let Some(k) = out.k.first() {
                write_bin(&format!("{p}.K.bin"), k);
            }
        }
        let mut sorted = times.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let name = fast.as_ref().map(|f| f.name()).unwrap_or_else(|| jk.name());
        let stats = fast.as_ref().map(|f| serde_json::to_value(&f.stats).unwrap());
        println!(
            "{}",
            serde_json::json!({"builder": name, "nbf": n, "nshell": basis.shells.len(), "threads": threads,
              "thresh": thresh, "t_setup": t_setup, "times": times, "median": sorted[sorted.len() / 2], "stats": stats})
        );
    });
}
