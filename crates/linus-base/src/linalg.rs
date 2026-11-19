//! Dense linear algebra helpers on faer matrices with explicit parallelism.
//!
//! This module is linus's route to faer's parallel kernels, and it keeps results independent
//! of the thread count (CONTRIBUTING.md, "Determinism"):
//! - products ([`matmul`], [`gemm`], [`mm`]) and triangular solves ([`solve_lower_in_place`],
//!   [`solve_upper_in_place`]) run sequentially when the result or the right-hand side has one row or column
//!   ([`product_par`]): faer splits such a product into `par.degree()` partial sums; products
//!   with less work per thread than their threads' synchronization costs run sequentially too
//!   (`gemm_par`: a choice of speed, the same bits);
//! - the eigensolver runs with a fixed degree ([`eigh`]);
//! - factorizations and solves take an explicit `Par` ([`cholesky`], [`cholesky_solve`],
//!   [`lu_solve`], [`svd`]): faer's convenience methods (`Mat::llt`, `Solve::solve`, `Mat::svd`)
//!   and its matrix operators (`&a * &b`) run with faer's global parallelism, which is the size
//!   of the current thread pool.
//!
//! clippy.toml forbids the faer functions this module wraps everywhere else, and
//! `tests/faer_parallelism.rs` runs calculations with faer's global parallelism disabled.
//!
//! Instruction sets: faer 0.22 (with its `std` feature) sends every product with
//! m n k > 16^3 to private-gemm-x86, which picks its AVX-512 kernels at run time on CPUs with
//! avx512f and its AVX2+FMA kernels otherwise (the `gemm` crate is faer's fallback for other
//! targets only). Both kernel sets add each element's products over k in the same order (k in
//! fixed blocks of 512, independent of the cache sizes), so dense products are bitwise identical
//! on AVX2 and AVX-512 CPUs, at about 2x the AVX2 rate on AVX-512 ones (Zen 5: 112-144 against
//! 65-72 GFLOP/s per core on linus's shapes; bench/results/2026-11-10-cpuperf). Smaller products
//! (nano-gemm) and faer's BLAS2-type code (pulp x86-v3: matrix-vector products, triangular
//! solves, Cholesky panels, the eigensolver's tridiagonalization) run AVX2 on every CPU. faer's
//! `nightly` feature would give them AVX-512, for under 0.2% of a calculation's time, and would
//! change the eigensolver's and the Cholesky factorization's bits.
// The wrapped faer functions are disallowed elsewhere (clippy.toml); this module is their home.
#![allow(clippy::disallowed_methods)]

use faer::dyn_stack::{MemBuffer, MemStack};
use faer::linalg::evd::{self_adjoint_evd, self_adjoint_evd_scratch, ComputeEigenvectors};
use faer::linalg::triangular_solve as tri;
use faer::mat::{AsMatMut, AsMatRef};
use faer::traits::ComplexField;
use faer::{Accum, Mat, MatMut, MatRef, Par};
use linus_ints::{AoMatrix, AoMatrixMut};
use rayon::prelude::*;

/// A faer matrix as linus-ints's strided view (no copy).
pub fn ao_view(m: MatRef<'_, f64>) -> AoMatrix<'_> {
    let n = m.nrows();
    assert_eq!(n, m.ncols());
    let (rs, cs) = (m.row_stride(), m.col_stride());
    assert!(rs >= 0 && cs >= 0, "positive strides");
    let len = if n == 0 {
        0
    } else {
        (n - 1) * (rs as usize + cs as usize) + 1
    };
    // SAFETY: a faer matrix of order n with these strides covers `len` values from its
    // pointer
    let data = unsafe { std::slice::from_raw_parts(m.as_ptr(), len) };
    AoMatrix::strided(data, n, rs as usize, cs as usize)
}

/// A writable faer matrix as linus-ints's strided view (no copy).
pub fn ao_view_mut(m: MatMut<'_, f64>) -> AoMatrixMut<'_> {
    let n = m.nrows();
    assert_eq!(n, m.ncols());
    let (rs, cs) = (m.row_stride(), m.col_stride());
    assert!(rs >= 0 && cs >= 0, "positive strides");
    let len = if n == 0 {
        0
    } else {
        (n - 1) * (rs as usize + cs as usize) + 1
    };
    // SAFETY: as in ao_view; the view borrows the matrix mutably
    let data = unsafe { std::slice::from_raw_parts_mut(m.as_ptr_mut(), len) };
    AoMatrixMut::strided(data, n, rs as usize, cs as usize)
}

/// Row-major nbf x nbf slice to a faer matrix.
pub fn mat_from_rowmajor(n: usize, m: usize, v: &[f64]) -> Mat<f64> {
    Mat::from_fn(n, m, |i, j| v[i * m + j])
}

/// The `nrows` x `ncols` matrix of column-major values `v`.
pub fn mat_from_colmajor(nrows: usize, ncols: usize, v: &[f64]) -> Mat<f64> {
    Mat::from_fn(nrows, ncols, |i, j| v[j * nrows + i])
}

/// A contiguous column-major copy of `m` (faer's columns may be padded).
pub fn mat_to_colmajor(m: MatRef<'_, f64>) -> Vec<f64> {
    let mut v = Vec::with_capacity(m.nrows() * m.ncols());
    for j in 0..m.ncols() {
        v.extend((0..m.nrows()).map(|i| m[(i, j)]));
    }
    v
}

/// A contiguous row-major copy of `m`: element (i, j) at `i * m.ncols() + j`, the layout the
/// integral contractions of `linus_ints` read.
pub fn mat_to_rowmajor(m: MatRef<'_, f64>) -> Vec<f64> {
    let mut v = Vec::with_capacity(m.nrows() * m.ncols());
    for i in 0..m.nrows() {
        v.extend((0..m.ncols()).map(|j| m[(i, j)]));
    }
    v
}

/// Parallel degree of the symmetric eigensolver. faer's result depends on this degree, not on
/// the size of the thread pool it runs in (checked with linus-dev `examples/det_check`: Par::rayon(8)
/// gives identical bits in pools of 1, 2, 8 and 32 threads, while Par::rayon(t) differs with t),
/// so a fixed degree makes every eigendecomposition independent of the thread count. 8 was
/// also faster than 32 (n = 666: 0.018 s vs 0.154 s; n = 1300: 0.07 s vs 0.32 s).
pub const EIGH_DEGREE: usize = 8;

/// Matrices below this order are diagonalized sequentially (the choice depends on n only).
const EIGH_PAR_MIN: usize = 96;

/// Eigenvalues (ascending) and eigenvectors (columns) of a symmetric matrix (lower triangle read).
/// The result is bitwise independent of the thread count: `par` is ignored and the solver runs
/// with degree [`EIGH_DEGREE`] for n >= 96 (sequentially below), in the current rayon pool.
/// Panics if the solver fails (non-finite elements); [`try_eigh`] returns the failure.
pub fn eigh(a: MatRef<'_, f64>, par: Par) -> (Vec<f64>, Mat<f64>) {
    try_eigh(a, par).expect("symmetric eigensolver failed")
}

/// [`eigh`] that returns an error instead of panicking when the matrix has non-finite elements
/// or the solver fails.
pub fn try_eigh(a: MatRef<'_, f64>, _par: Par) -> Result<(Vec<f64>, Mat<f64>), String> {
    let n = a.nrows();
    if !(0..n).all(|j| (j..n).all(|i| a[(i, j)].is_finite())) {
        return Err(format!("the {n} x {n} matrix has non-finite elements"));
    }
    // an empty matrix (the stability analysis's rotation of a spin with no electrons, e.g. the H
    // atom's beta orbitals): faer 0.22's workspace query computes n - 1, which overflows (a
    // panic with overflow checks, the dev profile; release builds wrapped to the same empty
    // result)
    if n == 0 {
        return Ok((Vec::new(), Mat::zeros(0, 0)));
    }
    let par = if n >= EIGH_PAR_MIN {
        Par::rayon(EIGH_DEGREE)
    } else {
        Par::Seq
    };
    let mut u = Mat::<f64>::zeros(n, n);
    let mut s = faer::diag::Diag::<f64>::zeros(n);
    let mut mem = MemBuffer::new(self_adjoint_evd_scratch::<f64>(
        n,
        ComputeEigenvectors::Yes,
        par,
        Default::default(),
    ));
    self_adjoint_evd(
        a,
        s.as_mut(),
        Some(u.as_mut()),
        par,
        MemStack::new(&mut mem),
        Default::default(),
    )
    .map_err(|e| format!("the symmetric eigensolver failed ({e:?})"))?;
    Ok(((0..n).map(|i| s[i]).collect(), u))
}

/// The parallelism of a product with an m x n result: `par`, except for matrix-vector shapes
/// (m = 1 or n = 1), which run sequentially. faer computes a column-major matrix-vector product
/// with parallelism `par` as `par.degree()` partial sums over the inner dimension, added at the
/// end (faer 0.22 `matmul::matvec_colmajor`, from 256^2 matrix elements), so its rounding would
/// follow the thread count of `Session::par`; sequentially it does not, and it is what one
/// thread (and every `Par::Seq` caller) computed before. Other shapes go to faer's GEMM, whose
/// result does not depend on the degree. Triangular solves with one right-hand side subtract
/// such products too: linus runs them with `Par::Seq`, and avoids faer's `Llt::solve` and
/// matrix operators (both take faer's global parallelism, the thread count of the pool) for
/// one right-hand side.
#[inline]
pub fn product_par(m: usize, n: usize, par: Par) -> Par {
    if m == 1 || n == 1 {
        Par::Seq
    } else {
        par
    }
}

/// Multiply-adds per thread below which a product runs sequentially ([`gemm_par`]): about 20 µs
/// of one Zen 5 core's work.
const GEMM_MIN_FMA_PER_THREAD: usize = 1 << 20;

/// The parallelism of an m x k by k x n product: [`product_par`]'s, and sequential when the
/// product has fewer than [`GEMM_MIN_FMA_PER_THREAD`] multiply-adds per thread of `par`.
///
/// faer's parallel GEMM (private-gemm-x86) gathers all `par.degree()` threads at a spin barrier
/// on every call, whatever the size of the product. Inside one CCD that costs a few µs; with the
/// threads spread over several CCDs, tens of µs to milliseconds (sleeping workers wake one after
/// another, a barrier waits for the slowest). The SCF's products of a small molecule are far
/// smaller than that: water/def2-SVP's 24 x 24 products take 0.2 µs on one thread, and at 32
/// threads on four CCDs its diagonalizations and DIIS took 0.4 s instead of 1 ms
/// (bench/results/2026-11-18-smallthr). The rule depends on the shape and the degree only, and
/// it changes no bits: faer's GEMM gives the same result at every degree, `Par::Seq` included
/// (what one thread computes).
#[inline]
fn gemm_par(m: usize, n: usize, k: usize, par: Par) -> Par {
    let fma = m.saturating_mul(n).saturating_mul(k);
    if fma < par.degree().saturating_mul(GEMM_MIN_FMA_PER_THREAD) {
        Par::Seq
    } else {
        product_par(m, n, par)
    }
}

/// faer's `matmul` with the parallelism of `gemm_par`: dst = alpha a b, or dst += alpha a b
/// with [`Accum::Add`]. Every dense product of linus goes through here or [`gemm`]/[`mm`], so
/// none depends on the thread count, and none smaller than its threads' synchronization runs in
/// parallel.
#[inline]
pub fn matmul<T: ComplexField>(
    dst: impl AsMatMut<T = T, Rows = usize, Cols = usize>,
    beta: Accum,
    a: impl AsMatRef<T = T, Rows = usize, Cols = usize>,
    b: impl AsMatRef<T = T, Rows = usize, Cols = usize>,
    alpha: T,
    par: Par,
) {
    let mut dst = dst;
    let dst = dst.as_mat_mut();
    let a = a.as_mat_ref();
    let par = gemm_par(dst.nrows(), dst.ncols(), a.ncols(), par);
    faer::linalg::matmul::matmul(dst, beta, a, b, alpha, par);
}

/// C = alpha * op(A) op(B) (+ C if accumulate), with the parallelism of `gemm_par`.
pub fn gemm(
    c: &mut Mat<f64>,
    a: MatRef<'_, f64>,
    b: MatRef<'_, f64>,
    alpha: f64,
    accumulate: bool,
    par: Par,
) {
    let acc = if accumulate {
        Accum::Add
    } else {
        Accum::Replace
    };
    matmul(c.as_mut(), acc, a, b, alpha, par);
}

/// A B, with the parallelism of `gemm_par`.
pub fn mm(a: MatRef<'_, f64>, b: MatRef<'_, f64>, par: Par) -> Mat<f64> {
    let mut c = Mat::<f64>::zeros(a.nrows(), b.ncols());
    matmul(c.as_mut(), Accum::Replace, a, b, 1.0, par);
    c
}

/// Solve L X = B in place on `x` (B), L lower triangular (`l`'s upper triangle is not read),
/// with the parallelism of [`product_par`] for the shape of `x`.
#[inline]
pub fn solve_lower_in_place(l: MatRef<'_, f64>, x: MatMut<'_, f64>, par: Par) {
    let par = product_par(x.nrows(), x.ncols(), par);
    tri::solve_lower_triangular_in_place(l, x, par);
}

/// Solve U X = B in place on `x` (B), U upper triangular (the lower triangle is not read), with
/// the parallelism of [`product_par`] for the shape of `x`.
#[inline]
pub fn solve_upper_in_place(u: MatRef<'_, f64>, x: MatMut<'_, f64>, par: Par) {
    let par = product_par(x.nrows(), x.ncols(), par);
    tri::solve_upper_triangular_in_place(u, x, par);
}

/// A Cholesky factorization met a non-positive pivot: the matrix is not (numerically) positive
/// definite.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NonPositivePivot {
    /// The column of the pivot.
    pub index: usize,
}

impl std::fmt::Display for NonPositivePivot {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "not positive definite (pivot {})", self.index)
    }
}

impl std::error::Error for NonPositivePivot {}

/// The Cholesky factor L of a symmetric positive definite matrix (the lower triangle of `a` is
/// read; L's upper triangle is zero), with parallelism `par`. faer's `Mat::llt` with an
/// explicit parallelism. [`cholesky_in_place`] factors a matrix the caller owns, without the copy.
pub fn cholesky(a: MatRef<'_, f64>, par: Par) -> Result<Mat<f64>, NonPositivePivot> {
    let n = a.nrows();
    assert_eq!(n, a.ncols(), "cholesky of a non-square matrix");
    let mut l = Mat::<f64>::zeros(n, n);
    l.copy_from_triangular_lower(a);
    cholesky_in_place(l, par)
}

/// [`cholesky`] in the storage of `a`: its lower triangle is read and overwritten by the factor,
/// and its upper triangle is zeroed. No copy of the matrix, for large ones such as a
/// density-fitting metric.
pub fn cholesky_in_place(mut a: Mat<f64>, par: Par) -> Result<Mat<f64>, NonPositivePivot> {
    use faer::linalg::cholesky::llt::factor::{self, LltError};
    let n = a.nrows();
    assert_eq!(n, a.ncols(), "cholesky of a non-square matrix");
    let mut mem = MemBuffer::new(factor::cholesky_in_place_scratch::<f64>(
        n,
        par,
        Default::default(),
    ));
    factor::cholesky_in_place(
        a.as_mut(),
        Default::default(),
        par,
        MemStack::new(&mut mem),
        Default::default(),
    )
    .map_err(|e| match e {
        LltError::NonPositivePivot { index } => NonPositivePivot { index },
    })?;
    // the factorization leaves the upper triangle untouched
    for j in 1..n {
        for i in 0..j {
            a[(i, j)] = 0.0;
        }
    }
    Ok(a)
}

/// A^-1 B for A = L L^T (`l` from [`cholesky`]): two triangular solves with the parallelism of
/// [`product_par`] (faer's `Llt::solve` with an explicit parallelism).
pub fn cholesky_solve(l: MatRef<'_, f64>, b: MatRef<'_, f64>, par: Par) -> Mat<f64> {
    let mut x = b.to_owned();
    solve_lower_in_place(l, x.as_mut(), par);
    solve_upper_in_place(l.transpose(), x.as_mut(), par);
    x
}

/// A^-1 B by LU decomposition with partial pivoting, with parallelism `par` (faer's
/// `partial_piv_lu().solve` with an explicit parallelism).
pub fn lu_solve(a: MatRef<'_, f64>, b: MatRef<'_, f64>, par: Par) -> Mat<f64> {
    use faer::linalg::lu::partial_pivoting::factor::{lu_in_place, lu_in_place_scratch};
    use faer::linalg::lu::partial_pivoting::solve::{solve_in_place, solve_in_place_scratch};
    let n = a.nrows();
    assert_eq!(n, a.ncols(), "LU solve of a non-square matrix");
    let mut lu = a.to_owned();
    let (mut fwd, mut bwd) = (vec![0usize; n], vec![0usize; n]);
    let mut mem = MemBuffer::new(lu_in_place_scratch::<usize, f64>(
        n,
        n,
        par,
        Default::default(),
    ));
    let (_, perm) = lu_in_place(
        lu.as_mut(),
        &mut fwd,
        &mut bwd,
        par,
        MemStack::new(&mut mem),
        Default::default(),
    );
    let mut x = b.to_owned();
    let mut mem = MemBuffer::new(solve_in_place_scratch::<usize, f64>(n, x.ncols(), par));
    let par = product_par(x.nrows(), x.ncols(), par);
    // the unit lower factor and the upper one share `lu` (each solve reads its own triangle)
    solve_in_place(
        lu.as_ref(),
        lu.as_ref(),
        perm,
        x.as_mut(),
        par,
        MemStack::new(&mut mem),
    );
    x
}

/// The singular value decomposition A = U diag(S) V^T (full U and V; singular values
/// non-increasing) with parallelism `par`; None if it fails (faer's `Mat::svd` with an explicit
/// parallelism).
pub fn svd(a: MatRef<'_, f64>, par: Par) -> Option<(Mat<f64>, Vec<f64>, Mat<f64>)> {
    use faer::linalg::svd::{svd as faer_svd, svd_scratch, ComputeSvdVectors};
    let (m, n) = a.shape();
    let mut u = Mat::<f64>::zeros(m, m);
    let mut v = Mat::<f64>::zeros(n, n);
    let mut s = faer::diag::Diag::<f64>::zeros(m.min(n));
    let full = ComputeSvdVectors::Full;
    let mut mem = MemBuffer::new(svd_scratch::<f64>(
        m,
        n,
        full,
        full,
        par,
        Default::default(),
    ));
    faer_svd(
        a,
        s.as_mut(),
        Some(u.as_mut()),
        Some(v.as_mut()),
        par,
        MemStack::new(&mut mem),
        Default::default(),
    )
    .ok()?;
    Some((u, (0..m.min(n)).map(|i| s[i]).collect(), v))
}

/// Canonical orthogonalization: X = U s^{-1/2} over eigenvalues of S above `thresh`.
pub fn canonical_orthogonalizer(s: MatRef<'_, f64>, thresh: f64, par: Par) -> Mat<f64> {
    let (w, u) = eigh(s, par);
    let keep: Vec<usize> = (0..w.len()).filter(|&i| w[i] > thresh).collect();
    let n = s.nrows();
    Mat::from_fn(n, keep.len(), |i, k| u[(i, keep[k])] / w[keep[k]].sqrt())
}

/// Neumaier-compensated sum of a stream of terms.
#[derive(Clone, Copy, Debug, Default)]
pub struct CompensatedSum {
    s: f64,
    c: f64,
}

impl CompensatedSum {
    #[inline]
    pub fn add(&mut self, x: f64) {
        let t = self.s + x;
        if self.s.abs() >= x.abs() {
            self.c += (self.s - t) + x;
        } else {
            self.c += (x - t) + self.s;
        }
        self.s = t;
    }
    pub fn value(&self) -> f64 {
        self.s + self.c
    }
}

/// sum_ij A_ij B_ij, summed with compensation (Neumaier): the result is the correctly rounded
/// sum of the rounded products up to O(n^2 eps^2), so energy traces are smooth functions of
/// the density. Plain summation left ~1e-11..1e-10 Eh of noise in SCF energies of ~1000 Eh
/// (n = 180..666), visible as dE jumping by +-1e-10 after convergence.
pub fn dot(a: MatRef<'_, f64>, b: MatRef<'_, f64>) -> f64 {
    assert_eq!((a.nrows(), a.ncols()), (b.nrows(), b.ncols()));
    let mut acc = CompensatedSum::default();
    for j in 0..a.ncols() {
        match (a.col(j).try_as_col_major(), b.col(j).try_as_col_major()) {
            (Some(x), Some(y)) => {
                for (u, v) in x.as_slice().iter().zip(y.as_slice()) {
                    acc.add(u * v);
                }
            }
            _ => {
                for i in 0..a.nrows() {
                    acc.add(a[(i, j)] * b[(i, j)]);
                }
            }
        }
    }
    acc.value()
}

/// The larger of `a` and `b`, NaN if either is NaN. `f64::max` returns the other operand for a
/// NaN, so a fold with it over residuals or norms hides a non-finite value behind the largest
/// finite one (a response whose residual was NaN once reported a maximum of 0 and passed its
/// convergence test). For finite values it is `f64::max`, bit for bit.
pub fn max_nan(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        f64::NAN
    } else {
        a.max(b)
    }
}

/// The largest absolute value of `x` (0 when empty), NaN if any element is NaN ([`max_nan`]).
pub fn max_abs_slice(x: &[f64]) -> f64 {
    x.iter().fold(0.0, |m, v| max_nan(m, v.abs()))
}

/// The largest absolute element of `a` (0 when empty), NaN if any element is NaN ([`max_nan`]).
pub fn max_abs(a: MatRef<'_, f64>) -> f64 {
    let mut m: f64 = 0.0;
    for j in 0..a.ncols() {
        for i in 0..a.nrows() {
            m = max_nan(m, a[(i, j)].abs());
        }
    }
    m
}

/// Symmetrize in place: A <- (A + A^T)/2
pub fn symmetrize(a: &mut Mat<f64>) {
    let n = a.nrows();
    for j in 0..n {
        for i in 0..j {
            let v = 0.5 * (a[(i, j)] + a[(j, i)]);
            a[(i, j)] = v;
            a[(j, i)] = v;
        }
    }
}

/// Fixed number of chunks of [`abt_split`]: results do
/// not depend on the thread count.
const K_SPLIT: usize = 32;

/// A B^T for A, B (m x k, n x k) with a long contraction k and small m, n: the k range is
/// split into a fixed number of chunks contracted by sequential GEMMs in parallel and summed
/// in order (faer's parallel GEMM does not split k, which leaves such inner-product shapes at
/// a fraction of the machine's throughput). The result does not depend on the thread count.
pub fn abt_split(a: MatRef<'_, f64>, b: MatRef<'_, f64>) -> Mat<f64> {
    let (m, n, k) = (a.nrows(), b.nrows(), a.ncols());
    let nchunk = K_SPLIT.min(k.div_ceil(256)).max(1);
    let len = k.div_ceil(nchunk);
    let parts: Vec<Mat<f64>> = (0..nchunk)
        .into_par_iter()
        .map(|c| {
            let r0 = c * len;
            let r1 = ((c + 1) * len).min(k);
            let mut p = Mat::<f64>::zeros(m, n);
            if r1 > r0 {
                matmul(
                    p.as_mut(),
                    Accum::Replace,
                    a.subcols(r0, r1 - r0),
                    b.subcols(r0, r1 - r0).transpose(),
                    1.0,
                    Par::Seq,
                );
            }
            p
        })
        .collect();
    let mut out = Mat::<f64>::zeros(m, n);
    for p in parts {
        out += p;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A symmetric positive definite n x n matrix and an n x k right-hand side (deterministic).
    fn spd(n: usize, k: usize) -> (Mat<f64>, Mat<f64>) {
        let g = Mat::from_fn(n, n, |i, j| (((i * 7 + j * 13) % 11) as f64 - 5.0) / 7.0);
        let mut a = mm(g.as_ref(), g.transpose(), Par::Seq);
        for i in 0..n {
            a[(i, i)] += n as f64;
        }
        let b = Mat::from_fn(n, k, |i, j| ((i + 3 * j) % 5) as f64 - 2.0);
        (a, b)
    }

    fn max_diff(a: MatRef<'_, f64>, b: MatRef<'_, f64>) -> f64 {
        max_abs((a.to_owned() - b.to_owned()).as_ref())
    }

    /// Products run sequentially below their threads' share of work, and in parallel above it;
    /// either way with the bits of one thread (faer's GEMM does not depend on the degree), in a
    /// pool of the degree's size.
    #[test]
    fn small_products_run_sequentially_with_the_same_bits() {
        let (m, k) = (24, 24);
        assert_eq!(gemm_par(m, m, k, Par::rayon(8)), Par::Seq);
        assert_eq!(gemm_par(512, 512, 512, Par::rayon(8)), Par::rayon(8));
        assert_eq!(gemm_par(512, 512, 512, Par::rayon(256)), Par::Seq);
        assert_eq!(gemm_par(4096, 1, 4096, Par::rayon(2)), Par::Seq);
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(8)
            .build()
            .unwrap();
        for (m, n, k) in [
            (24, 24, 24),
            (100, 90, 120),
            (200, 300, 250),
            (260, 260, 260),
        ] {
            let a = Mat::from_fn(m, k, |i, j| ((i * 31 + j * 17) % 23) as f64 * 0.1 - 1.0);
            let b = Mat::from_fn(k, n, |i, j| ((i * 13 + j * 7) % 19) as f64 * 0.05 - 0.4);
            let seq = mm(a.as_ref(), b.as_ref(), Par::Seq);
            let par = pool.install(|| mm(a.as_ref(), b.as_ref(), Par::rayon(8)));
            assert!(seq == par, "{m} x {k} x {n}");
        }
    }

    /// A NaN anywhere makes the maximum NaN (f64::max would skip it); finite values as
    /// f64::max.
    #[test]
    fn maxima_propagate_nan() {
        assert_eq!(max_nan(1.0, 2.0), 2.0);
        assert!(max_nan(f64::NAN, 2.0).is_nan() && max_nan(2.0, f64::NAN).is_nan());
        assert_eq!(max_abs_slice(&[]), 0.0);
        assert_eq!(max_abs_slice(&[1.0, -3.0, 2.0]), 3.0);
        assert!(max_abs_slice(&[1.0, f64::NAN, 2.0]).is_nan());
        let m = Mat::from_fn(3, 2, |i, j| if (i, j) == (2, 1) { f64::NAN } else { -1.0 });
        assert!(max_abs(m.as_ref()).is_nan());
        assert_eq!(
            max_abs(Mat::from_fn(2, 2, |i, j| (i + j) as f64).as_ref()),
            2.0
        );
    }

    #[test]
    fn cholesky_and_lu_solve_agree() {
        for (n, k) in [(40, 1), (300, 1), (300, 5)] {
            let (a, b) = spd(n, k);
            let l = cholesky(a.as_ref(), Par::rayon(4)).unwrap();
            assert_eq!(
                max_abs(Mat::from_fn(n, n, |i, j| if i < j { l[(i, j)] } else { 0.0 }).as_ref()),
                0.0
            );
            let x = cholesky_solve(l.as_ref(), b.as_ref(), Par::rayon(4));
            let y = lu_solve(a.as_ref(), b.as_ref(), Par::rayon(4));
            assert!(max_diff(mm(a.as_ref(), x.as_ref(), Par::Seq).as_ref(), b.as_ref()) < 1e-10);
            assert!(max_diff(x.as_ref(), y.as_ref()) < 1e-12);
            // one right-hand side runs sequentially: the same bits as Par::Seq
            if k == 1 {
                let xs = cholesky_solve(l.as_ref(), b.as_ref(), Par::Seq);
                assert_eq!(max_diff(x.as_ref(), xs.as_ref()), 0.0);
            }
        }
        let mut a = Mat::<f64>::identity(3, 3);
        a[(0, 0)] = -1.0;
        assert_eq!(
            cholesky(a.as_ref(), Par::Seq).err(),
            Some(NonPositivePivot { index: 0 })
        );
        // in place: the same factor
        let (a, _) = spd(50, 1);
        let l = cholesky(a.as_ref(), Par::Seq).unwrap();
        let li = cholesky_in_place(a, Par::Seq).unwrap();
        assert_eq!(max_diff(l.as_ref(), li.as_ref()), 0.0);
    }

    #[test]
    fn svd_reconstructs() {
        let a = Mat::from_fn(3, 3, |i, j| ((i * 5 + j * 3) % 7) as f64 - 3.0);
        let (u, s, v) = svd(a.as_ref(), Par::Seq).unwrap();
        assert!(s.windows(2).all(|w| w[0] >= w[1]));
        let us = Mat::from_fn(3, 3, |i, j| u[(i, j)] * s[j]);
        let r = mm(us.as_ref(), v.transpose(), Par::Seq);
        assert!(max_diff(r.as_ref(), a.as_ref()) < 1e-12);
    }

    #[test]
    fn eigh_of_an_empty_matrix() {
        let (w, v) = eigh(Mat::<f64>::zeros(0, 0).as_ref(), Par::Seq);
        assert!(w.is_empty() && v.nrows() == 0 && v.ncols() == 0);
    }
}
