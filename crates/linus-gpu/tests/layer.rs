//! The device layer: vendor calls stay in the backend; BLAS and eigensolver work.

mod common;

/// Only src/cuda/ may name the vendor runtime (cudarc): kernels, detsum, the harness, the
/// examples and the tests use the device layer's items, so a HIP backend can sit beside it.
#[test]
fn vendor_runtime_only_in_the_backend() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut bad = Vec::new();
    let mut stack = vec![root.join("src"), root.join("examples"), root.join("tests")];
    while let Some(d) = stack.pop() {
        for e in std::fs::read_dir(&d).unwrap() {
            let p = e.unwrap().path();
            if p.is_dir() {
                if p != root.join("src/cuda") {
                    stack.push(p);
                }
            } else if p.extension().is_some_and(|x| x == "rs")
                && p.file_name().is_some_and(|f| f != "layer.rs")
            {
                let text = std::fs::read_to_string(&p).unwrap();
                if text.contains("cudarc") {
                    bad.push(p.display().to_string());
                }
            }
        }
    }
    assert!(bad.is_empty(), "cudarc used outside src/cuda: {bad:?}");
}

#[cfg(feature = "gpu")]
mod gpu {
    use super::common::{self, Rng};
    use linus_gpu::blas::{Blas, GemmDims, MatMut, MatRef, Op};
    use linus_gpu::solver::Solver;

    #[test]
    fn dgemm_matches_host() {
        let Some(gpu) = common::gpu() else { return };
        let (m, n, k) = (37, 23, 51);
        let mut rng = Rng(5);
        let a: Vec<f64> = (0..k * m).map(|_| rng.uniform() - 0.5).collect(); // k x m column-major (A^T used)
        let b: Vec<f64> = (0..k * n).map(|_| rng.uniform() - 0.5).collect();
        let blas = Blas::new(&gpu).unwrap();
        let (da, db) = (gpu.upload(&a).unwrap(), gpu.upload(&b).unwrap());
        let mut dc = gpu.alloc_zeros::<f64>(m * n).unwrap();
        blas.gemm(
            GemmDims::new(Op::T, Op::N, m, n, k),
            (1.0, 0.0),
            MatRef::new(&da, 0, k),
            MatRef::new(&db, 0, k),
            MatMut::new(&mut dc, 0, m),
        )
        .unwrap();
        let c = gpu.download(&dc).unwrap();
        for j in 0..n {
            for i in 0..m {
                let want: f64 = (0..k).map(|p| a[i * k + p] * b[j * k + p]).sum();
                assert!((c[j * m + i] - want).abs() < 1e-13, "({i},{j})");
            }
        }
    }

    #[test]
    fn syevd_recovers_a_known_spectrum() {
        let Some(gpu) = common::gpu() else { return };
        let n = 120;
        // A = H diag(lambda) H with the Householder reflection H = I - 2 v v^T / v^T v
        let mut rng = Rng(11);
        let v: Vec<f64> = (0..n).map(|_| rng.uniform() - 0.5).collect();
        let vv: f64 = v.iter().map(|x| x * x).sum();
        let h = |i: usize, j: usize| (if i == j { 1.0 } else { 0.0 }) - 2.0 * v[i] * v[j] / vv;
        let lambda: Vec<f64> = (0..n)
            .map(|i| -3.0 + 0.05 * i as f64 + 1e-3 * (i % 7) as f64)
            .collect();
        let mut a = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                a[j * n + i] = (0..n).map(|p| h(i, p) * lambda[p] * h(p, j)).sum();
            }
        }
        let solver = Solver::new(&gpu).unwrap();
        let mut da = gpu.upload(&a).unwrap();
        let mut dw = gpu.alloc_zeros::<f64>(n).unwrap();
        solver.syevd(&gpu, &mut da, n, &mut dw).unwrap();
        let w = gpu.download(&dw).unwrap();
        let x = gpu.download(&da).unwrap();
        let mut sorted = lambda.clone();
        sorted.sort_by(|p, q| p.partial_cmp(q).unwrap());
        for i in 0..n {
            assert!(
                (w[i] - sorted[i]).abs() < 1e-12,
                "eigenvalue {i}: {} vs {}",
                w[i],
                sorted[i]
            );
            // residual of A x_i = w_i x_i
            let r: f64 = (0..n)
                .map(|row| {
                    ((0..n).map(|c| a[c * n + row] * x[i * n + c]).sum::<f64>()
                        - w[i] * x[i * n + row])
                        .powi(2)
                })
                .sum::<f64>()
                .sqrt();
            assert!(r < 1e-11, "residual {i}: {r}");
        }
    }
}

/// A grant (`Gpu::grant`): its bytes count as in use from the grant on; a draw hands them to the
/// allocation that follows (in use unchanged across draw and allocation); a draw beyond what
/// remains is a recorded violation, not a refusal; what remains returns to the pool on drop.
#[cfg(feature = "gpu")]
#[test]
fn grants_draws_and_violations() {
    let Some(gpu) = common::gpu() else { return };
    let used = || gpu.budget().in_use;
    let u0 = used();
    let mb = 1usize << 20;
    let g = gpu.grant(64 * mb, "test: an engine").unwrap();
    assert_eq!(
        (used(), g.remaining(), g.granted()),
        (u0 + 64 * mb, 64 * mb, 64 * mb)
    );
    g.draw(16 * mb);
    let a = gpu.alloc_zeros::<u8>(16 * mb).unwrap();
    assert_eq!((used(), g.remaining()), (u0 + 64 * mb, 48 * mb));
    assert!(g.violations().is_empty());
    g.draw(56 * mb);
    let b = gpu.alloc_zeros::<u8>(56 * mb).unwrap();
    assert_eq!((used(), g.remaining()), (u0 + 72 * mb, 0));
    let v = g.violations();
    assert_eq!(v.len(), 1);
    assert_eq!(
        (v[0].what, v[0].requested, v[0].remaining),
        ("test: an engine", 56 * mb, 48 * mb)
    );
    drop(g);
    assert_eq!(used(), u0 + 72 * mb);
    drop((a, b));
    assert_eq!(used(), u0);
    let h = gpu.grant(8 * mb, "test: another").unwrap();
    drop(h);
    assert_eq!(used(), u0);
    // more than the limit leaves: out of memory, nothing held
    assert!(gpu.grant(gpu.budget().limit + 1, "test: too much").is_err());
    assert_eq!(used(), u0);
}

/// A granted allocation (`Gpu::alloc_zeros_granted`) gives its bytes back to the grant when it
/// is freed: a workspace reused by the next phase or call draws them again, and a violation is
/// the engine holding more at once than granted (its peak), not the sum of its allocations.
/// Freed after the grant dropped, it returns them to the pool.
#[cfg(feature = "gpu")]
#[test]
fn granted_allocations_give_their_bytes_back() {
    let Some(gpu) = common::gpu() else { return };
    let used = || gpu.budget().in_use;
    let u0 = used();
    let mb = 1usize << 20;
    let g = gpu.grant(32 * mb, "test: a workspace").unwrap();
    // two calls of 24 MB each: within the grant both times
    for _ in 0..2 {
        let a = gpu.alloc_zeros_granted::<u8>(Some(&g), 24 * mb).unwrap();
        assert_eq!((used(), g.remaining()), (u0 + 32 * mb, 8 * mb));
        drop(a);
        assert_eq!((used(), g.remaining()), (u0 + 32 * mb, 32 * mb));
    }
    assert!(g.violations().is_empty());
    // 24 + 16 MB at once: a violation of 16 MB with 8 left
    let a = gpu.alloc_zeros_granted::<u8>(Some(&g), 24 * mb).unwrap();
    let b = gpu.upload_granted(Some(&g), &vec![0u8; 16 * mb]).unwrap();
    assert_eq!((used(), g.remaining()), (u0 + 40 * mb, 0));
    let v = g.violations();
    assert_eq!(
        (v.len(), v[0].requested, v[0].remaining),
        (1, 16 * mb, 8 * mb)
    );
    // b took 8 MB of the grant: freed, they come back (not the 8 MB beyond it)
    drop(b);
    assert_eq!((used(), g.remaining()), (u0 + 32 * mb, 8 * mb));
    drop(g);
    assert_eq!(used(), u0 + 24 * mb);
    drop(a);
    assert_eq!(used(), u0);
    // without a grant: an ordinary allocation
    let c = gpu.alloc_zeros_granted::<u8>(None, mb).unwrap();
    assert_eq!(used(), u0 + mb);
    drop(c);
    assert_eq!(used(), u0);
}
