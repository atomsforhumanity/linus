//! gemm_bench [threads...]: faer GEMM throughput on the shapes DF-K uses, against OpenBLAS
//! (system libopenblas, loaded at run time if present). Prints GFLOP/s.
// it measures faer's GEMM itself, not linus's wrapper (clippy.toml)
#![allow(clippy::disallowed_methods)]
use faer::{Accum, Mat, Par};
use std::time::Instant;

type DgemmFn = unsafe extern "C" fn(
    i32,
    i32,
    i32,
    i32,
    i32,
    i32,
    f64,
    *const f64,
    i32,
    *const f64,
    i32,
    f64,
    *mut f64,
    i32,
);
type SetThreadsFn = unsafe extern "C" fn(i32);

struct OpenBlas {
    dgemm: DgemmFn,
    set_threads: SetThreadsFn,
}

fn load_openblas() -> Option<OpenBlas> {
    unsafe {
        extern "C" {
            fn dlopen(name: *const std::ffi::c_char, flag: i32) -> *mut std::ffi::c_void;
            fn dlsym(
                h: *mut std::ffi::c_void,
                name: *const std::ffi::c_char,
            ) -> *mut std::ffi::c_void;
        }
        let h = dlopen(c"libopenblas.so.0".as_ptr(), 2);
        let h = if h.is_null() {
            dlopen(c"libopenblas.so".as_ptr(), 2)
        } else {
            h
        };
        if h.is_null() {
            return None;
        }
        let g = dlsym(h, c"cblas_dgemm".as_ptr());
        let t = dlsym(h, c"openblas_set_num_threads".as_ptr());
        if g.is_null() || t.is_null() {
            return None;
        }
        Some(OpenBlas {
            dgemm: std::mem::transmute::<*mut std::ffi::c_void, DgemmFn>(g),
            set_threads: std::mem::transmute::<*mut std::ffi::c_void, SetThreadsFn>(t),
        })
    }
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn main() {
    let threads: Vec<usize> = std::env::args()
        .skip(1)
        .map(|s| s.parse().unwrap())
        .collect();
    let threads = if threads.is_empty() {
        vec![1, 8, 16, 64]
    } else {
        threads
    };
    let ob = load_openblas();
    // (m, n, k, label): C = A B with A m x k, B k x n, all column-major
    let small = std::env::var("SMALL").is_ok();
    let shapes_small = [
        (54usize, 153usize, 600usize, "Z s-shell nq=54"),
        (108, 153, 600, "Z s-shell nq=108"),
        (162, 153, 600, "Z p-shell nq=54"),
        (270, 153, 600, "Z d-shell nq=54"),
        (756, 153, 600, "Z atom nq=54"),
        (666, 666, 54 * 153, "K nq=54"),
        (666, 666, 108 * 153, "K nq=108"),
    ];
    let shapes_big = [
        (666usize, 666usize, 256 * 153usize, "K=ZZ^T ts61 (nq=256)"),
        (1106, 1106, 128 * 233, "K=ZZ^T ts113svp"),
        (256, 153, 400, "Z_m = B_m C (small)"),
        (4096, 3436, 3436, "B = A L^-T block"),
        (2000, 2000, 2000, "square 2000"),
    ];
    let shapes: Vec<_> = if small {
        shapes_small.to_vec()
    } else {
        shapes_big.to_vec()
    };
    for &nt in &threads {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(nt)
            .build()
            .unwrap();
        let par = if nt > 1 { Par::rayon(nt) } else { Par::Seq };
        for &(m, n, k, label) in &shapes {
            let a = Mat::<f64>::from_fn(m, k, |i, j| ((i * 7 + j * 3) % 11) as f64 * 0.1);
            let b = Mat::<f64>::from_fn(k, n, |i, j| ((i * 5 + j * 2) % 13) as f64 * 0.1);
            let mut c = Mat::<f64>::zeros(m, n);
            let fl = 2.0 * m as f64 * n as f64 * k as f64;
            let reps = ((2e11 * nt as f64 / fl) as usize).clamp(2, 50);
            let mut tf = vec![];
            pool.install(|| {
                faer::linalg::matmul::matmul(
                    c.as_mut(),
                    Accum::Replace,
                    a.as_ref(),
                    b.as_ref(),
                    1.0,
                    par,
                );
                for _ in 0..reps {
                    let t = Instant::now();
                    faer::linalg::matmul::matmul(
                        c.as_mut(),
                        Accum::Replace,
                        a.as_ref(),
                        b.as_ref(),
                        1.0,
                        par,
                    );
                    tf.push(t.elapsed().as_secs_f64());
                }
            });
            let gf = fl / median(tf) / 1e9;
            let mut go = f64::NAN;
            if let Some(o) = &ob {
                unsafe { (o.set_threads)(nt as i32) };
                let mut c2 = vec![0.0; m * n];
                let mut to = vec![];
                for r in 0..=reps {
                    let t = Instant::now();
                    unsafe {
                        (o.dgemm)(
                            102,
                            111,
                            111,
                            m as i32,
                            n as i32,
                            k as i32,
                            1.0,
                            a.as_ptr(),
                            a.col_stride() as i32,
                            b.as_ptr(),
                            b.col_stride() as i32,
                            0.0,
                            c2.as_mut_ptr(),
                            m as i32,
                        )
                    };
                    if r > 0 {
                        to.push(t.elapsed().as_secs_f64());
                    }
                }
                go = fl / median(to) / 1e9;
                let d = (0..m * n)
                    .map(|x| (c2[x] - c[(x % m, x / m)]).abs())
                    .fold(0.0, f64::max);
                assert!(d < 1e-6 * k as f64, "mismatch {d}");
            }
            println!("threads {nt:3}  {label:28} m={m} n={n} k={k}: faer {gf:8.1} GF/s  openblas {go:8.1} GF/s  ({:.1} GF/s/thread faer)", gf / nt as f64);
        }
    }
}
