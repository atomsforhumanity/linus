//! The solvers of linear response, independent of what the operator is: the block Krylov
//! linear solver of CPHF/CPKS ([`subspace_solve`]: all right-hand sides in one shared, growing
//! subspace) and the Davidson eigensolver of the lowest eigenpair ([`lowest_eigenpair`]: the
//! orbital Hessian's lowest mode in the SCF's stability analysis). Each takes its products
//! from the caller as a closure on flat vectors: the CPHF/CPKS products of
//! [`crate::response::problem::ResponseProblem`], or the stability analysis' finite
//! differences of Fock builds (`crate::stability`, decision 7: they work for every J/K
//! and environment).
//!
//! Both are deterministic. Every product of the Krylov solver is a faer GEMM or a fixed-chunk
//! reduction of linus-num's `det` ([`det::dot`](fn@det::dot), [`det::col_dots`], [`det::sub_cols`]:
//! the same bits at any thread count); the Davidson's dot products are [`det::dot`](fn@det::dot)
//! too, and the rest of its subspace algebra is sequential.

use faer::{Accum, Mat, MatRef, Par};
use linus_base::error::{LinusError, ResponseEnd};
use linus_base::linalg::{matmul, mm, try_eigh};
use linus_base::progress::{Event, Progress};
use linus_num::det;

#[cfg(any(test, feature = "reference"))]
pub mod reference;

/// Iterations, the max-abs residual of the last one over all right-hand sides, and how the
/// solve ended.
#[derive(Clone, Copy, Debug)]
pub struct ResponseStats {
    pub iterations: usize,
    pub max_residual: f64,
    pub end: ResponseEnd,
    /// Wall seconds of [`subspace_solve`]'s own algebra (all but the products), and of them
    /// the orthonormalization of the new directions.
    pub solver_seconds: f64,
    pub orth_seconds: f64,
}

/// When a [`subspace_solve`] stops, and where it reports its iterations.
#[derive(Clone, Debug)]
pub struct SolveControl {
    /// Converged when max |b - A x| <= `tol` for every right-hand side.
    pub tol: f64,
    pub max_iter: usize,
    /// The `response_iteration` event of every iteration, of the solve named `solve`, and the
    /// caller's stop request between iterations (an error `Stopped` in `solve`).
    pub progress: Progress,
    pub solve: &'static str,
}

impl SolveControl {
    /// `tol` and `max_iter`, reporting nothing.
    pub fn quiet(tol: f64, max_iter: usize) -> Self {
        SolveControl {
            tol,
            max_iter,
            progress: Progress::off(),
            solve: "",
        }
    }
}

/// Columns per block of the new directions' Gram-Schmidt ([`SubspaceAlgebra::orthonormalize`]).
pub const GS_BLOCK: usize = 16;

/// A new direction is dropped when its norm falls to this fraction of its norm before the
/// projections (or to 0): it lies in the subspace already ([`SubspaceAlgebra::orthonormalize`]).
pub const DROP_RATIO: f64 = 1e-10;

/// Directions per block of the extras' final combination ([`SubspaceAlgebra::combine_extras`]).
pub const EXTRAS_BLOCK: usize = 64;

/// Shared-subspace solver of A X = B (A symmetric positive definite, B len x nr) for all
/// right-hand sides at once: a block Krylov / Davidson-type linear solver, like PySCF's
/// `krylov`. Each iteration adds the Jacobi-preconditioned residuals (`diag`) of the unconverged
/// right-hand sides (every column above `tol` after the last iteration, one that was below it
/// too), orthonormalized (twice) against the subspace and each other, asks `apply` for A z and an
/// "extra" vector e(z) of each new direction z (one batched call: one J/K build), and solves the
/// projected system V^T A V c = V^T B (`elen`: length of the extras). Sharing the subspace takes
/// fewer iterations than independent conjugate gradients. Returns X = V c, the same combination
/// of the extras, sum_j c_jk e(v_j) (e linear, e.g. dF\[z\] C_o), and the iterations, final max
/// |residual| and how the solve ended. Converged when max |B - A X| <= `tol` for every column.
/// An error of `apply` stops the solve and is returned, as does the caller's stop request
/// (`LinusError::Stopped`, between iterations).
///
/// The host's algebra ([`HostAlgebra`]) under the one driver ([`subspace_solve_with`]): BLAS-3
/// (block classical Gram-Schmidt, twice) and incremental (V, A V and V^T A V grow by the new
/// directions' columns and rows); every product is a faer GEMM or a fixed-chunk reduction
/// ([`det::dot`](fn@det::dot)), so the results do not depend on the number of threads.
pub fn subspace_solve(
    b: &Mat<f64>,
    diag: &[f64],
    elen: usize,
    control: SolveControl,
    par: Par,
    mut apply: impl FnMut(&[Vec<f64>]) -> Result<Products, LinusError>,
) -> Result<Solution, LinusError> {
    let mut host = HostAlgebra::new(b, diag, par);
    subspace_solve_with(&mut host, b.ncols(), elen, control, par, &mut apply).map_err(|e| match e {
        SolveError::Solve(e) => e,
        SolveError::Algebra(never) => match never {},
    })
}

/// The products of new directions: A z and the extra e(z) of each, in order.
pub type Products = Vec<(Vec<f64>, Vec<f64>)>;

/// The products' closure of a solve ([`Products`] of the new directions).
pub type Apply<'a> = dyn FnMut(&[Vec<f64>]) -> Result<Products, LinusError> + 'a;

/// A solve's result: X, the combined extras and the statistics.
pub type Solution = (Mat<f64>, Mat<f64>, ResponseStats);

/// Why [`subspace_solve_with`] stopped early: its algebra's error (a device's), or the solve's
/// own (a product failed, the caller asked to stop).
#[derive(Debug)]
pub enum SolveError<E> {
    Algebra(E),
    Solve(LinusError),
}

/// The data-parallel half of [`subspace_solve`]: where the subspace V (len x m), its products
/// A V, the right-hand sides B (len x nr), the residuals R and the new directions live, and how
/// their products run (the host's: [`HostAlgebra`]; a device's in the facade). The driver,
/// [`subspace_solve_with`], keeps the rest: the active right-hand sides, the projected system
/// and its solve, convergence, the progress events and the stop request. Columns are added in
/// order; `m0` is the subspace's size before an iteration's new directions.
pub trait SubspaceAlgebra {
    type Error;
    /// The max-abs of each residual column (R = B before the first iteration), NaN where a
    /// column holds a non-finite value the algebra reports as NaN.
    fn residual_max(&mut self) -> Result<Vec<f64>, Self::Error>;
    /// The new directions: the residuals of the `active` columns divided by the
    /// preconditioner's diagonal; their norms (before any projection).
    fn precondition(&mut self, active: &[usize]) -> Result<Vec<f64>, Self::Error>;
    /// Projects the subspace's `m0` columns out of the new directions, twice (no-op for an
    /// empty subspace).
    fn project_out(&mut self, m0: usize) -> Result<(), Self::Error>;
    /// Orthonormalizes the new directions among themselves into the subspace's columns from
    /// `m0` on, in order, and returns how many it kept: block classical Gram-Schmidt in blocks
    /// of [`GS_BLOCK`] columns (two projections against the earlier blocks' kept directions,
    /// then each column twice against its block's kept ones), a direction dropped when its norm
    /// falls to [`DROP_RATIO`] of `z0` (its norm before the projections) or 0.
    fn orthonormalize(&mut self, m0: usize, z0: &[f64]) -> Result<usize, Self::Error>;
    /// The `nn` kept new directions (columns m0..m0 + nn) as vectors of the products.
    fn new_directions(&mut self, m0: usize, nn: usize) -> Result<Vec<Vec<f64>>, Self::Error>;
    /// Adds A z of the new directions (columns m0..m0 + az.len()) to A V.
    fn append_products(&mut self, m0: usize, az: Vec<Vec<f64>>) -> Result<(), Self::Error>;
    /// The projected system's new parts: V^T A V_new (m x nn), V_new^T A V_old (nn x m0) and
    /// V_new^T B (nn x nr), with V_new the columns m0..m0 + nn of V and m = m0 + nn.
    #[allow(
        clippy::type_complexity,
        reason = "three named blocks of one projection"
    )]
    fn new_rows(
        &mut self,
        m0: usize,
        nn: usize,
    ) -> Result<(Mat<f64>, Mat<f64>, Mat<f64>), Self::Error>;
    /// R = B - A V c for the coefficients `c` (m x nr).
    fn residuals(&mut self, c: &Mat<f64>) -> Result<(), Self::Error>;
    /// X = V c (len x nr; zeros for an empty subspace).
    fn solution(&mut self, c: &Mat<f64>) -> Result<Mat<f64>, Self::Error>;
    /// The extras `ex` (one of `elen` per subspace column) combined, sum_j c_jk e_j (elen x nr),
    /// in blocks of [`EXTRAS_BLOCK`] columns. The last call: the algebra may free the subspace
    /// first.
    fn combine_extras(
        &mut self,
        ex: &[Vec<f64>],
        elen: usize,
        c: &Mat<f64>,
    ) -> Result<Mat<f64>, Self::Error>;
}

/// [`subspace_solve`] on the algebra `alg` (holding B, len x `nr`, and the preconditioner): the
/// one driver of the shared-subspace solver for every algebra. The projected m x m system is
/// solved on the host ([`solve_projected`]).
pub fn subspace_solve_with<A: SubspaceAlgebra>(
    alg: &mut A,
    nr: usize,
    elen: usize,
    control: SolveControl,
    par: Par,
    apply: &mut Apply<'_>,
) -> Result<Solution, SolveError<A::Error>> {
    use SolveError::{Algebra, Solve};
    let SolveControl {
        tol,
        max_iter,
        progress,
        solve,
    } = control;
    let t_all = std::time::Instant::now();
    let (mut t_apply, mut t_orth) = (0.0, 0.0);
    // the extras, V^T A V and V^T b (m x m, m x nr), the coefficients (m x nr)
    let mut ex: Vec<Vec<f64>> = Vec::new();
    let mut g = Mat::<f64>::zeros(0, 0);
    let mut vb = Mat::<f64>::zeros(0, nr);
    let mut c = Mat::<f64>::zeros(0, nr);
    let mut m = 0;
    let mut it = 0;
    let mut rmax = alg.residual_max().map_err(Algebra)?;
    let mut active = unconverged(&rmax, tol);
    // the last projected matrix was positive definite; the new directions all lay in the subspace
    let (mut definite, mut exhausted) = (true, false);
    while !active.is_empty() && it < max_iter {
        if it > 0 && progress.stop_due() {
            return Err(Solve(LinusError::stopped(solve, Some(it))));
        }
        // new directions: preconditioned residuals of the active right-hand sides,
        // orthonormalized (twice) against the subspace and each other
        let t0 = std::time::Instant::now();
        let m0 = m;
        let z0 = alg.precondition(&active).map_err(Algebra)?;
        alg.project_out(m0).map_err(Algebra)?;
        let nn = alg.orthonormalize(m0, &z0).map_err(Algebra)?;
        t_orth += t0.elapsed().as_secs_f64();
        if nn == 0 {
            exhausted = true;
            break;
        }
        it += 1;
        let zs = alg.new_directions(m0, nn).map_err(Algebra)?;
        let t0 = std::time::Instant::now();
        let products = apply(&zs).map_err(Solve)?;
        t_apply += t0.elapsed().as_secs_f64();
        drop(zs);
        // V and A V grow by the new columns
        m = m0 + nn;
        let mut az = Vec::with_capacity(nn);
        for (a, e) in products {
            az.push(a);
            ex.push(e);
        }
        alg.append_products(m0, az).map_err(Algebra)?;
        // V^T A V (symmetrized) and V^T b by their new columns and rows
        let (x, y, vbn) = alg.new_rows(m0, nn).map_err(Algebra)?;
        let mut gn = Mat::<f64>::zeros(m, m);
        gn.submatrix_mut(0, 0, m0, m0).copy_from(g.as_ref());
        gn.submatrix_mut(0, m0, m, nn).copy_from(x.as_ref());
        gn.submatrix_mut(m0, 0, nn, m0).copy_from(y.as_ref());
        for i in m0..m {
            for j in 0..i {
                let s = 0.5 * (gn[(i, j)] + gn[(j, i)]);
                gn[(i, j)] = s;
                gn[(j, i)] = s;
            }
        }
        g = gn;
        let mut vbm = Mat::<f64>::zeros(m, nr);
        vbm.submatrix_mut(0, 0, m0, nr).copy_from(vb.as_ref());
        vbm.submatrix_mut(m0, 0, nn, nr).copy_from(vbn.as_ref());
        vb = vbm;
        (c, definite) = solve_projected(g.as_ref(), vb.as_ref(), par);
        alg.residuals(&c).map_err(Algebra)?;
        rmax = alg.residual_max().map_err(Algebra)?;
        active = unconverged(&rmax, tol);
        if progress.listens() {
            progress.event(Event::ResponseIteration {
                solve,
                iteration: it,
                residual: rmax.iter().copied().fold(0.0, linus_base::linalg::max_nan),
                unconverged: active.len(),
                total: nr,
            });
        }
    }
    let max_residual = rmax.iter().copied().fold(0.0, linus_base::linalg::max_nan);
    let end = if max_residual.is_nan() {
        ResponseEnd::NonFinite
    } else if active.is_empty() {
        ResponseEnd::Converged
    } else if !definite {
        ResponseEnd::Indefinite
    } else if exhausted {
        ResponseEnd::Exhausted
    } else {
        ResponseEnd::Iterations
    };
    // X = V c, and sum_j c_jk e(v_j) in blocks of basis vectors (bounded scratch)
    let x = alg.solution(&c).map_err(Algebra)?;
    let fx = alg.combine_extras(&ex, elen, &c).map_err(Algebra)?;
    Ok((
        x,
        fx,
        ResponseStats {
            iterations: it,
            max_residual,
            end,
            solver_seconds: t_all.elapsed().as_secs_f64() - t_apply,
            orth_seconds: t_orth,
        },
    ))
}

/// The right-hand sides whose max-abs residual (`rmax`, per column) is above `tol` now: the ones
/// an iteration of [`subspace_solve_with`] gives new directions, and its stopping test. Every
/// iteration asks again of every column, one that was below `tol` too: the new directions change
/// every column's projected solution, so a column's residual can rise above `tol` after it fell
/// below (an IR Hessian's electric-field CPHF ended at 1.2 `tol`, each of its three columns once
/// below it), and the solve ends only with all of them below it.
fn unconverged(rmax: &[f64], tol: f64) -> Vec<usize> {
    (0..rmax.len()).filter(|&k| rmax[k] > tol).collect()
}

/// c = G^-1 vb for the symmetric projected matrix `g` (m x m) and `vb` (m x nr): Cholesky,
/// else LU (a G that is not positive definite: an operator with an eigenvalue at or below zero,
/// to round-off); and whether G was positive definite.
pub fn solve_projected(g: MatRef<'_, f64>, vb: MatRef<'_, f64>, par: Par) -> (Mat<f64>, bool) {
    match linus_base::linalg::cholesky(g, par) {
        Ok(l) => (
            linus_base::linalg::cholesky_solve(l.as_ref(), vb, par),
            true,
        ),
        Err(_) => (linus_base::linalg::lu_solve(g, vb, par), false),
    }
}

/// [`subspace_solve`]'s algebra on the host (faer matrices; the module docs' determinism).
pub struct HostAlgebra<'a> {
    b: &'a Mat<f64>,
    diag: &'a [f64],
    par: Par,
    /// V and A V (len x m)
    vm: Mat<f64>,
    avm: Mat<f64>,
    /// the residuals R (len x nr)
    resid: Mat<f64>,
    /// the iteration's new directions (len x na), then the kept ones orthonormalized
    zm: Mat<f64>,
    qn: Mat<f64>,
}

impl<'a> HostAlgebra<'a> {
    /// The algebra of A X = `b` with the Jacobi preconditioner `diag`, at parallelism `par`.
    pub fn new(b: &'a Mat<f64>, diag: &'a [f64], par: Par) -> Self {
        let (len, nr) = (b.nrows(), b.ncols());
        HostAlgebra {
            b,
            diag,
            par,
            vm: Mat::<f64>::with_capacity(len, 4 * nr),
            avm: Mat::<f64>::with_capacity(len, 4 * nr),
            resid: b.clone(),
            zm: Mat::<f64>::zeros(0, 0),
            qn: Mat::<f64>::zeros(0, 0),
        }
    }
}

impl SubspaceAlgebra for HostAlgebra<'_> {
    type Error = std::convert::Infallible;

    fn residual_max(&mut self) -> Result<Vec<f64>, Self::Error> {
        Ok((0..self.resid.ncols())
            .map(|k| linus_base::linalg::max_abs_slice(self.resid.col_as_slice(k)))
            .collect())
    }

    fn precondition(&mut self, active: &[usize]) -> Result<Vec<f64>, Self::Error> {
        let (resid, diag) = (&self.resid, self.diag);
        self.zm = Mat::from_fn(resid.nrows(), active.len(), |i, j| {
            resid[(i, active[j])] / diag[i]
        });
        let zm = &self.zm;
        Ok(map_cols(zm.ncols(), self.par, |j| {
            det::dot(zm.col_as_slice(j), zm.col_as_slice(j)).sqrt()
        }))
    }

    fn project_out(&mut self, m0: usize) -> Result<(), Self::Error> {
        if m0 > 0 {
            for _ in 0..2 {
                let p = mm(self.vm.transpose(), self.zm.as_ref(), self.par);
                matmul(
                    self.zm.as_mut(),
                    Accum::Add,
                    self.vm.as_ref(),
                    p.as_ref(),
                    -1.0,
                    self.par,
                );
            }
        }
        Ok(())
    }

    fn orthonormalize(&mut self, _m0: usize, z0: &[f64]) -> Result<usize, Self::Error> {
        self.qn = orthonormalize_new(&self.zm, z0, self.par);
        self.zm = Mat::<f64>::zeros(0, 0);
        Ok(self.qn.ncols())
    }

    fn new_directions(&mut self, _m0: usize, nn: usize) -> Result<Vec<Vec<f64>>, Self::Error> {
        Ok((0..nn).map(|j| self.qn.col_as_slice(j).to_vec()).collect())
    }

    fn append_products(&mut self, m0: usize, az: Vec<Vec<f64>>) -> Result<(), Self::Error> {
        let (len, m) = (self.b.nrows(), m0 + az.len());
        let qn = std::mem::replace(&mut self.qn, Mat::<f64>::zeros(0, 0));
        self.vm.resize_with(len, m, |i, j| qn[(i, j - m0)]);
        drop(qn);
        self.avm.resize_with(len, m, |i, j| az[j - m0][i]);
        Ok(())
    }

    fn new_rows(
        &mut self,
        m0: usize,
        nn: usize,
    ) -> Result<(Mat<f64>, Mat<f64>, Mat<f64>), Self::Error> {
        let (vm, avm, par) = (&self.vm, &self.avm, self.par);
        let x = mm(vm.transpose(), avm.subcols(m0, nn), par);
        let y = mm(vm.subcols(m0, nn).transpose(), avm.subcols(0, m0), par);
        let vbn = mm(vm.subcols(m0, nn).transpose(), self.b.as_ref(), par);
        Ok((x, y, vbn))
    }

    fn residuals(&mut self, c: &Mat<f64>) -> Result<(), Self::Error> {
        self.resid = self.b.clone();
        matmul(
            self.resid.as_mut(),
            Accum::Add,
            self.avm.as_ref(),
            c.as_ref(),
            -1.0,
            self.par,
        );
        Ok(())
    }

    fn solution(&mut self, c: &Mat<f64>) -> Result<Mat<f64>, Self::Error> {
        Ok(if self.vm.ncols() > 0 {
            mm(self.vm.as_ref(), c.as_ref(), self.par)
        } else {
            Mat::<f64>::zeros(self.b.nrows(), self.b.ncols())
        })
    }

    fn combine_extras(
        &mut self,
        ex: &[Vec<f64>],
        elen: usize,
        c: &Mat<f64>,
    ) -> Result<Mat<f64>, Self::Error> {
        self.vm = Mat::<f64>::zeros(0, 0);
        self.avm = Mat::<f64>::zeros(0, 0);
        let mut fx = Mat::<f64>::zeros(elen, self.b.ncols());
        for j0 in (0..ex.len()).step_by(EXTRAS_BLOCK) {
            let j1 = (j0 + EXTRAS_BLOCK).min(ex.len());
            let cb = c.subrows(j0, j1 - j0);
            let eb = Mat::from_fn(elen, j1 - j0, |i, j| ex[j0 + j][i]);
            matmul(fx.as_mut(), Accum::Add, eb.as_ref(), cb, 1.0, self.par);
        }
        Ok(fx)
    }
}

/// A [`subspace_solve`] elsewhere (a device's): its own algebra under
/// [`subspace_solve_with`], from the same right-hand sides, preconditioner and products, its
/// results the host's to round-off. A [`crate::response::problem::ResponseProblem`] given one
/// solves with it instead of the host's.
pub trait SubspaceSolver: Send {
    /// As [`subspace_solve`]: X, the combined extras and the statistics.
    fn solve(
        &mut self,
        b: &Mat<f64>,
        diag: &[f64],
        elen: usize,
        control: SolveControl,
        par: Par,
        apply: &mut Apply<'_>,
    ) -> Result<Solution, LinusError>;
}

/// `f(j)` for j < n, in parallel over j when `par` has more than one thread (each value is
/// computed alone, so the result does not depend on the parallelism).
pub(crate) fn map_cols<T: Send>(
    n: usize,
    par: Par,
    f: impl Fn(usize) -> T + Sync + Send,
) -> Vec<T> {
    use rayon::prelude::*;
    if par.degree() > 1 {
        (0..n).into_par_iter().map(f).collect()
    } else {
        (0..n).map(f).collect()
    }
}

/// The new directions of a [`subspace_solve`] iteration (len x kept): the columns of `zm`
/// (already orthogonalized against the subspace) in order, each orthogonalized twice against
/// the directions kept before it, normalized, and dropped when its norm falls to [`DROP_RATIO`]
/// of `z0` (its norm before the projections) or 0. Block classical Gram-Schmidt: per block of
/// [`GS_BLOCK`] columns, two GEMM projections against the earlier blocks' kept directions, then
/// the block's columns in order against the block's kept ones, twice, by fixed-chunk reductions
/// ([`det::col_dots`], [`det::sub_cols`]).
fn orthonormalize_new(zm: &Mat<f64>, z0: &[f64], par: Par) -> Mat<f64> {
    let (len, na) = (zm.nrows(), zm.ncols());
    let mut q = Mat::<f64>::zeros(len, na);
    let mut k = 0;
    for b0 in (0..na).step_by(GS_BLOCK) {
        let nb = GS_BLOCK.min(na - b0);
        let mut blk = zm.subcols(b0, nb).to_owned();
        if k > 0 {
            for _ in 0..2 {
                let p = mm(q.subcols(0, k).transpose(), blk.as_ref(), par);
                matmul(
                    blk.as_mut(),
                    Accum::Add,
                    q.subcols(0, k),
                    p.as_ref(),
                    -1.0,
                    par,
                );
            }
        }
        let kb = k;
        for j in 0..nb {
            let mut z = blk.col_as_slice(j).to_vec();
            if k > kb {
                for _ in 0..2 {
                    let p = det::col_dots(&q, kb..k, &z, par);
                    det::sub_cols(&mut z, &q, kb..k, &p, par);
                }
            }
            let nz = det::dot(&z, &z).sqrt();
            if nz > DROP_RATIO * z0[b0 + j] && nz > 0.0 {
                for (t, s) in q.col_as_slice_mut(k).iter_mut().zip(&z) {
                    *t = s / nz;
                }
                k += 1;
            }
        }
    }
    q.truncate(len, k);
    q
}

/// Where a Davidson search ([`lowest_eigenpair`]) starts.
#[derive(Clone, Debug)]
pub enum Start {
    /// The unit vector on the lowest diagonal element (the first of equal ones) times this
    /// sign.
    LowestDiagonal(f64),
    /// A given vector (normalized by the search).
    Vector(Vec<f64>),
}

/// What a Davidson search ([`lowest_eigenpair`]) works with besides its products: the
/// preconditioner's diagonal (its length is the operator's dimension; for an orbital Hessian
/// the orbital-energy differences), a guess of the eigenvector (the first subspace vector), the
/// start, and the most products it may ask for.
pub struct DavidsonSetup<'a> {
    pub diag: &'a [f64],
    pub guess: Option<Vec<f64>>,
    pub start: Start,
    pub max_products: usize,
}

/// What a [`lowest_eigenpair`] search found: the eigenvalue estimate theta, the norm of its
/// residual A x - theta x, the eigenvector x (normalized), the iterations (Rayleigh-Ritz steps)
/// and the products spent.
#[derive(Clone, Debug)]
pub struct Eigenpair {
    pub value: f64,
    pub residual: f64,
    pub vector: Vec<f64>,
    pub iterations: usize,
    pub products: usize,
}

/// The lowest eigenpair of a symmetric operator A by Davidson iterations, with
/// `product(v)` = A v, one call per subspace vector (at most `setup.max_products`).
///
/// The subspace starts with the guess, then the start, each orthonormalized (twice) against the
/// vectors before it and dropped when its norm falls below 1e-8. Once every vector has its product,
/// an iteration diagonalizes the symmetrized Rayleigh matrix and adds the correction r_p /
/// max(d_p - theta, 2e-2) of the residual r = A x - theta x (Davidson's diagonal preconditioner d,
/// its shift floored). The search stops when `done(theta, |r|)`, when a correction is linearly
/// dependent on the subspace, or at the product budget. A failed product (NaN, as a caller's failed
/// Fock build returns) ends it with a NaN eigenvalue and residual instead of a panic in the
/// eigensolver. Its dot products are the fixed-chunk reduction of the Krylov solver
/// ([`det::dot`](fn@det::dot)) and the rest of its arithmetic is sequential, so the result does not
/// depend on the thread count. An empty operator (no occupied-virtual rotations) gives the zero
/// pair without a product.
pub fn lowest_eigenpair(
    setup: DavidsonSetup<'_>,
    product: &mut dyn FnMut(&[f64]) -> Vec<f64>,
    done: impl Fn(f64, f64) -> bool,
) -> Eigenpair {
    let DavidsonSetup {
        diag,
        guess,
        start,
        max_products,
    } = setup;
    let n = diag.len();
    if n == 0 {
        return Eigenpair {
            value: 0.0,
            residual: 0.0,
            vector: Vec::new(),
            iterations: 0,
            products: 0,
        };
    }
    let mut vs: Vec<Vec<f64>> = Vec::new();
    let mut hvs: Vec<Vec<f64>> = Vec::new();
    // add a vector orthonormalized against the subspace (false if it is linearly dependent)
    let add = |vs: &mut Vec<Vec<f64>>, mut t: Vec<f64>| -> bool {
        for _ in 0..2 {
            for v in vs.iter() {
                let p = det::dot(v, &t);
                t.iter_mut().zip(v).for_each(|(x, y)| *x -= p * y);
            }
        }
        let nt = det::dot(&t, &t).sqrt();
        if nt < 1e-8 {
            return false;
        }
        t.iter_mut().for_each(|x| *x /= nt);
        vs.push(t);
        true
    };
    if let Some(g) = guess {
        add(&mut vs, g);
    }
    let e = match start {
        Start::Vector(v) => v,
        Start::LowestDiagonal(sign) => {
            let imin = (0..n)
                .min_by(|&a, &b| diag[a].total_cmp(&diag[b]))
                .unwrap_or(0);
            let mut e = vec![0.0; n];
            e[imin] = sign;
            e
        }
    };
    add(&mut vs, e);
    let (mut theta, mut rnorm, mut x) = (0.0, f64::INFINITY, vec![0.0; n]);
    let mut iters = 0;
    while hvs.len() < vs.len() && hvs.len() < max_products {
        let hv = product(&vs[hvs.len()]);
        hvs.push(hv);
        if hvs.len() < vs.len() {
            continue;
        }
        iters += 1;
        let k = vs.len();
        let a = Mat::from_fn(k, k, |i, j| {
            0.5 * (det::dot(&vs[i], &hvs[j]) + det::dot(&vs[j], &hvs[i]))
        });
        // a failed product (NaN) ends the search without an eigenpair
        let Ok((w, u)) = try_eigh(a.as_ref(), Par::Seq) else {
            (theta, rnorm) = (f64::NAN, f64::NAN);
            break;
        };
        theta = w[0];
        x = (0..n)
            .map(|p| (0..k).map(|i| u[(i, 0)] * vs[i][p]).sum())
            .collect();
        let hx: Vec<f64> = (0..n)
            .map(|p| (0..k).map(|i| u[(i, 0)] * hvs[i][p]).sum())
            .collect();
        let r: Vec<f64> = hx.iter().zip(&x).map(|(hv, xv)| hv - theta * xv).collect();
        rnorm = det::dot(&r, &r).sqrt();
        if done(theta, rnorm) {
            break;
        }
        let t: Vec<f64> = r
            .iter()
            .zip(diag)
            .map(|(ri, d)| ri / (d - theta).max(2e-2))
            .collect();
        if !add(&mut vs, t) {
            break;
        }
    }
    Eigenpair {
        value: theta,
        residual: rnorm,
        vector: x,
        iterations: iters,
        products: hvs.len(),
    }
}

#[cfg(test)]
mod tests {
    use super::reference::{noise, Problem};
    use super::*;

    /// [`subspace_solve`] as it was before the parallel orthonormalization (main 324bfbe),
    /// verbatim but for the unconverged columns, which follow the driver's rule (`unconverged`:
    /// every column above `tol` after each iteration): the reference of the algebra.
    #[allow(clippy::type_complexity)]
    fn subspace_solve_reference(
        b: &Mat<f64>,
        diag: &[f64],
        elen: usize,
        tol: f64,
        max_iter: usize,
        par: Par,
        mut apply: impl FnMut(&[Vec<f64>]) -> Result<Vec<(Vec<f64>, Vec<f64>)>, LinusError>,
    ) -> Result<(Mat<f64>, Mat<f64>, ResponseStats), LinusError> {
        let (len, nr) = (b.nrows(), b.ncols());
        let mut v: Vec<Vec<f64>> = Vec::new();
        let mut av: Vec<Vec<f64>> = Vec::new();
        let mut ex: Vec<Vec<f64>> = Vec::new();
        let mut c = Mat::<f64>::zeros(0, nr);
        let mut resid = b.clone();
        let mut it = 0;
        let rmax = |m: &Mat<f64>, k: usize| linus_base::linalg::max_abs_slice(m.col_as_slice(k));
        let above = |r: &Mat<f64>| (0..nr).filter(|&k| rmax(r, k) > tol).collect::<Vec<_>>();
        let mut active = above(&resid);
        while !active.is_empty() && it < max_iter {
            let mut zm = Mat::from_fn(len, active.len(), |i, j| resid[(i, active[j])] / diag[i]);
            let z0: Vec<f64> = (0..zm.ncols())
                .map(|j| zm.col_as_slice(j).iter().map(|t| t * t).sum::<f64>().sqrt())
                .collect();
            if !v.is_empty() {
                let vm = Mat::from_fn(len, v.len(), |i, j| v[j][i]);
                for _ in 0..2 {
                    let p = mm(vm.transpose(), zm.as_ref(), par);
                    zm -= mm(vm.as_ref(), p.as_ref(), par);
                }
            }
            let mut new: Vec<Vec<f64>> = Vec::new();
            for (j, &n0) in z0.iter().enumerate() {
                let mut z = zm.col_as_slice(j).to_vec();
                for _ in 0..2 {
                    for q in &new {
                        let p: f64 = q.iter().zip(&z).map(|(a, b)| a * b).sum();
                        z.iter_mut().zip(q).for_each(|(t, s)| *t -= p * s);
                    }
                }
                let nz = z.iter().map(|t| t * t).sum::<f64>().sqrt();
                if nz > 1e-10 * n0 && nz > 0.0 {
                    z.iter_mut().for_each(|t| *t /= nz);
                    new.push(z);
                }
            }
            if new.is_empty() {
                break;
            }
            it += 1;
            for ((az, e), z) in apply(&new)?.into_iter().zip(new) {
                v.push(z);
                av.push(az);
                ex.push(e);
            }
            let m = v.len();
            let vm = Mat::from_fn(len, m, |i, j| v[j][i]);
            let avm = Mat::from_fn(len, m, |i, j| av[j][i]);
            let mut g = mm(vm.transpose(), avm.as_ref(), par);
            for i in 0..m {
                for j in 0..i {
                    let s = 0.5 * (g[(i, j)] + g[(j, i)]);
                    g[(i, j)] = s;
                    g[(j, i)] = s;
                }
            }
            let vb = mm(vm.transpose(), b.as_ref(), par);
            c = match linus_base::linalg::cholesky(g.as_ref(), par) {
                Ok(l) => linus_base::linalg::cholesky_solve(l.as_ref(), vb.as_ref(), par),
                Err(_) => linus_base::linalg::lu_solve(g.as_ref(), vb.as_ref(), par),
            };
            resid = b - mm(avm.as_ref(), c.as_ref(), par);
            active = above(&resid);
        }
        let max_residual = (0..nr)
            .map(|k| rmax(&resid, k))
            .fold(0.0, linus_base::linalg::max_nan);
        let mut x = Mat::<f64>::zeros(len, nr);
        let mut fx = Mat::<f64>::zeros(elen, nr);
        const BLK: usize = 64;
        for j0 in (0..v.len()).step_by(BLK) {
            let j1 = (j0 + BLK).min(v.len());
            let cb = c.subrows(j0, j1 - j0);
            let vb = Mat::from_fn(len, j1 - j0, |i, j| v[j0 + j][i]);
            linus_base::linalg::matmul(x.as_mut(), faer::Accum::Add, vb.as_ref(), cb, 1.0, par);
            let eb = Mat::from_fn(elen, j1 - j0, |i, j| ex[j0 + j][i]);
            linus_base::linalg::matmul(fx.as_mut(), faer::Accum::Add, eb.as_ref(), cb, 1.0, par);
        }
        Ok((
            x,
            fx,
            ResponseStats {
                iterations: it,
                max_residual,
                end: if active.is_empty() {
                    ResponseEnd::Converged
                } else {
                    ResponseEnd::Iterations
                },
                solver_seconds: 0.0,
                orth_seconds: 0.0,
            },
        ))
    }

    /// [`lowest_eigenpair`] on d + U U^T against the dense eigensolver: the lowest eigenvalue and
    /// its vector (up to sign), from the default start and from a guess; a product budget below
    /// the subspace's stops the search with what it has.
    #[test]
    fn davidson_finds_the_lowest_eigenpair() {
        for &(len, r) in &[(1, 1), (12, 2), (90, 5), (400, 8)] {
            let pr = Problem::new(len, r, 1, 0);
            let dense = Mat::from_fn(len, len, |i, j| {
                let uu: f64 = (0..r).map(|k| pr.u[(i, k)] * pr.u[(j, k)]).sum();
                uu + if i == j { pr.d[i] } else { 0.0 }
            });
            let (w, v) = try_eigh(dense.as_ref(), Par::Seq).unwrap();
            let mut product = |z: &[f64]| pr.apply(&[z.to_vec()]).remove(0).0;
            for guess in [None, Some(pr.b.col_as_slice(0).to_vec())] {
                let setup = DavidsonSetup {
                    diag: &pr.d,
                    guess,
                    start: Start::LowestDiagonal(1.0),
                    max_products: len.max(2) + 10,
                };
                let e = lowest_eigenpair(setup, &mut product, |_, rnorm| rnorm < 1e-10);
                assert!(
                    (e.value - w[0]).abs() < 1e-9,
                    "len {len}: {} vs {}",
                    e.value,
                    w[0]
                );
                let ov: f64 = (0..len).map(|i| e.vector[i] * v[(i, 0)]).sum();
                assert!((ov.abs() - 1.0).abs() < 1e-8, "len {len}: overlap {ov}");
                assert!(e.products <= len.max(2) + 10 && e.iterations >= 1);
            }
            let setup = DavidsonSetup {
                diag: &pr.d,
                guess: None,
                start: Start::LowestDiagonal(-1.0),
                max_products: 1,
            };
            let e = lowest_eigenpair(setup, &mut product, |_, _| false);
            assert_eq!((e.products, e.iterations), (1, 1));
        }
        let e = lowest_eigenpair(
            DavidsonSetup {
                diag: &[],
                guess: None,
                start: Start::LowestDiagonal(1.0),
                max_products: 4,
            },
            &mut |_| unreachable!("no product of an empty operator"),
            |_, _| true,
        );
        assert_eq!((e.value, e.products, e.vector.len()), (0.0, 0, 0));
    }

    fn same(a: &Mat<f64>, b: &Mat<f64>) -> bool {
        a.nrows() == b.nrows()
            && a.ncols() == b.ncols()
            && (0..a.ncols()).all(|j| {
                a.col_as_slice(j)
                    .iter()
                    .zip(b.col_as_slice(j))
                    .all(|(x, y)| x.to_bits() == y.to_bits())
            })
    }

    #[test]
    fn subspace_solve_matches_the_sequential_solver_at_any_parallelism() {
        let rel = |a: &Mat<f64>, b: &Mat<f64>| {
            let (mut d, mut m) = (0.0f64, 0.0f64);
            for j in 0..a.ncols() {
                for (x, y) in a.col_as_slice(j).iter().zip(b.col_as_slice(j)) {
                    d = d.max((x - y).abs());
                    m = m.max(x.abs());
                }
            }
            d / m.max(1e-300)
        };
        // (len, rank, right-hand sides, near-duplicates, tolerance, iterations)
        for &(len, r, nr, dup, tol, max_iter) in &[
            (1, 1, 1, 0, 1e-12, 10),
            (37, 3, 5, 0, 1e-10, 20),
            (300, 8, 17, 3, 1e-9, 12),
            (2000, 20, 40, 5, 1e-7, 6),
            (5000, 10, 3, 1, 1e-8, 8),
            (9000, 6, 50, 4, 1e-8, 5),
        ] {
            let pr = Problem::new(len, r, nr, dup);
            let elen = len / 3;
            let mut first: Option<(Mat<f64>, Mat<f64>)> = None;
            for threads in [1, 3, 8] {
                let pool = rayon::ThreadPoolBuilder::new()
                    .num_threads(threads)
                    .build()
                    .unwrap();
                let par = if threads == 1 {
                    Par::Seq
                } else {
                    Par::rayon(threads)
                };
                let (x0, f0, s0) = pool.install(|| {
                    subspace_solve_reference(&pr.b, &pr.d, elen, tol, max_iter, par, |zs| {
                        Ok(pr.apply(zs))
                    })
                    .unwrap()
                });
                let (x1, f1, s1) = pool.install(|| {
                    subspace_solve(
                        &pr.b,
                        &pr.d,
                        elen,
                        SolveControl::quiet(tol, max_iter),
                        par,
                        |zs| Ok(pr.apply(zs)),
                    )
                    .unwrap()
                });
                let tag = format!("len {len} r {r} nr {nr} dup {dup} threads {threads}");
                // the sequential solver's results to round-off, in as many iterations
                assert_eq!(s0.iterations, s1.iterations, "{tag}: iterations");
                assert!(
                    rel(&x0, &x1) < 1e-10,
                    "{tag}: solution {:.1e}",
                    rel(&x0, &x1)
                );
                if elen > 0 {
                    assert!(rel(&f0, &f1) < 1e-10, "{tag}: extras {:.1e}", rel(&f0, &f1));
                }
                // and bitwise the same at any number of threads
                match &first {
                    None => first = Some((x1, f1)),
                    Some((x, f)) => {
                        assert!(same(x, &x1), "{tag}: solution differs from one thread's");
                        assert!(same(f, &f1), "{tag}: extras differ from one thread's");
                    }
                }
            }
        }
    }

    /// A right-hand side that converged can rise above the tolerance again: the shared subspace's
    /// later directions (for the other right-hand sides) change its projected solution. The solve
    /// gives it directions again and ends only with every column below `tol`. The problem,
    /// A = d + W (W a dense symmetric coupling) with three right-hand sides of different sizes, is
    /// one where a column's residual rises above `tol` after it fell below, which the solutions of
    /// the solve stopped after 1, 2, ... iterations show (B - A X of each, computed here); a solve
    /// that stopped once each column had been below `tol` (the rule before the fix) ended at 1.16
    /// `tol`.
    #[test]
    fn a_converged_column_that_rises_again_gets_new_directions() {
        let (len, tol) = (40, 1e-5);
        let d: Vec<f64> = noise(1, len)
            .iter()
            .map(|x| 1.0 + 4.0 * (x + 0.5))
            .collect();
        let wn = noise(2, len * len);
        let scale = 0.5 / (len as f64).sqrt();
        let a = Mat::from_fn(len, len, |i, j| {
            let w = scale * (wn[i * len + j] + wn[j * len + i]);
            if i == j {
                d[i] + w
            } else {
                w
            }
        });
        let bn = noise(3, len * 3);
        let sizes = [1.0, 0.05, 0.2];
        let b = Mat::from_fn(len, 3, |i, k| sizes[k] * bn[k * len + i]);
        let apply = |zs: &[Vec<f64>]| {
            Ok(zs
                .iter()
                .map(|z| {
                    let az = (0..len)
                        .map(|i| (0..len).map(|j| a[(i, j)] * z[j]).sum())
                        .collect();
                    (az, Vec::new())
                })
                .collect())
        };
        let solve = |max_iter: usize| {
            subspace_solve(
                &b,
                &d,
                0,
                SolveControl::quiet(tol, max_iter),
                Par::Seq,
                apply,
            )
            .unwrap()
        };
        // max |B - A X| per column
        let residuals = |x: &Mat<f64>| -> Vec<f64> {
            let r = &b - mm(a.as_ref(), x.as_ref(), Par::Seq);
            (0..3)
                .map(|k| linus_base::linalg::max_abs_slice(r.col_as_slice(k)))
                .collect()
        };
        let (x, _, st) = solve(50);
        assert!(st.iterations < 50 && st.max_residual <= tol, "{st:?}");
        let r = residuals(&x);
        assert!(
            r.iter().all(|&rk| rk <= 1.0001 * tol),
            "max |B - A X| {r:?}"
        );
        // a column below tol after some iteration and above it after a later one
        let mut below = [false; 3];
        let mut rose = false;
        for k in 1..=st.iterations {
            for (c, &rc) in residuals(&solve(k).0).iter().enumerate() {
                rose |= below[c] && rc > tol;
                below[c] |= rc <= tol;
            }
        }
        assert!(
            rose,
            "no column rose above tol again: the problem no longer tests the rule"
        );
    }

    /// A singular operator whose null vector the right-hand side has a part of: the equations
    /// have no solution, and the solve says how it ended (`exhausted` or `indefinite`, never
    /// `converged` or `iterations`), as for a broken-symmetry state that rotates freely. The
    /// operator is Q diag(0, 1, ..., n - 1) Q^T with Q a product of plane rotations; the same
    /// operator with the null eigenvalue raised to 0.5 converges.
    #[test]
    fn a_singular_operator_ends_exhausted_or_indefinite() {
        let n = 12;
        let rotations = |lam0: f64| {
            let mut q = Mat::<f64>::identity(n, n);
            let angles = noise(7, n);
            for k in 0..n - 1 {
                let (c, s) = ((angles[k] * 3.0).cos(), (angles[k] * 3.0).sin());
                for i in 0..n {
                    let (a, b) = (q[(i, k)], q[(i, k + 1)]);
                    q[(i, k)] = c * a - s * b;
                    q[(i, k + 1)] = s * a + c * b;
                }
            }
            let lam: Vec<f64> = (0..n)
                .map(|k| if k == 0 { lam0 } else { k as f64 })
                .collect();
            Mat::from_fn(n, n, |i, j| {
                (0..n).map(|k| q[(i, k)] * lam[k] * q[(j, k)]).sum()
            })
        };
        let b = Mat::from_fn(n, 1, |i, _| 1.0 + 0.1 * i as f64);
        for (lam0, want) in [(0.0, None), (0.5, Some(ResponseEnd::Converged))] {
            let a = rotations(lam0);
            let diag: Vec<f64> = (0..n).map(|i| a[(i, i)]).collect();
            let apply = |zs: &[Vec<f64>]| {
                Ok(zs
                    .iter()
                    .map(|z| {
                        let az = (0..n)
                            .map(|i| (0..n).map(|j| a[(i, j)] * z[j]).sum())
                            .collect();
                        (az, Vec::new())
                    })
                    .collect())
            };
            let control = SolveControl::quiet(1e-10, 4 * n);
            let (_, _, st) = subspace_solve(&b, &diag, 0, control, Par::Seq, apply).unwrap();
            match want {
                Some(end) => assert_eq!(st.end, end, "{st:?}"),
                None => assert!(
                    matches!(st.end, ResponseEnd::Exhausted | ResponseEnd::Indefinite)
                        && st.max_residual > 1e-10,
                    "{st:?}"
                ),
            }
        }
    }

    /// Digests of the solutions' bits on the test problems (8 threads), to compare builds or
    /// machines: the same digests on an AVX2-only GEMM path (`cargo test --release -p linus --lib
    /// subspace_solve_digests -- --ignored --nocapture`).
    #[test]
    #[ignore]
    fn subspace_solve_digests() {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(8)
            .build()
            .unwrap();
        // tight tolerances: the solves run to the iteration limit
        for &(len, r, nr, dup, tol, max_iter) in &[
            (300, 40, 17, 3, 1e-14, 12),
            (2000, 60, 40, 5, 1e-14, 8),
            (9000, 80, 50, 4, 1e-14, 6),
        ] {
            let pr = Problem::new(len, r, nr, dup);
            let (x, f, st) = pool.install(|| {
                subspace_solve(
                    &pr.b,
                    &pr.d,
                    len / 3,
                    SolveControl::quiet(tol, max_iter),
                    Par::rayon(8),
                    |zs| Ok(pr.apply(zs)),
                )
                .unwrap()
            });
            let mut h: u64 = 0xcbf29ce484222325;
            for m in [&x, &f] {
                for j in 0..m.ncols() {
                    for v in m.col_as_slice(j) {
                        h = (h ^ v.to_bits()).wrapping_mul(0x100000001b3);
                    }
                }
            }
            eprintln!(
                "digest len {len} nr {nr}: {h:016x} ({} iterations)",
                st.iterations
            );
        }
    }

    /// The orthonormalization of one iteration's new directions alone (as
    /// [`subspace_solve_timing`]'s sizes), the sequential loop against [`orthonormalize_new`].
    #[test]
    #[ignore]
    fn orthonormalize_timing() {
        let spec = std::env::var("LINUS_SOLVE_SIZE").unwrap_or_else(|_| "182070,183,6".into());
        let v: Vec<usize> = spec.split(',').map(|s| s.parse().unwrap()).collect();
        let (len, nr) = (v[0], v[1]);
        let zn = noise(5, len * nr);
        let zm = Mat::from_fn(len, nr, |i, j| zn[j * len + i]);
        let z0: Vec<f64> = (0..nr)
            .map(|j| zm.col_as_slice(j).iter().map(|t| t * t).sum::<f64>().sqrt())
            .collect();
        let par = Par::rayon(rayon::current_num_threads());
        let t = std::time::Instant::now();
        let mut new: Vec<Vec<f64>> = Vec::new();
        for (j, &n0) in z0.iter().enumerate() {
            let mut z = zm.col_as_slice(j).to_vec();
            for _ in 0..2 {
                for q in &new {
                    let p: f64 = q.iter().zip(&z).map(|(a, b)| a * b).sum();
                    z.iter_mut().zip(q).for_each(|(t, s)| *t -= p * s);
                }
            }
            let nz = z.iter().map(|t| t * t).sum::<f64>().sqrt();
            if nz > 1e-10 * n0 && nz > 0.0 {
                z.iter_mut().for_each(|t| *t /= nz);
                new.push(z);
            }
        }
        let t_ref = t.elapsed().as_secs_f64();
        let t = std::time::Instant::now();
        let new2 = orthonormalize_new(&zm, &z0, par);
        let t_new = t.elapsed().as_secs_f64();
        let dmax = new
            .iter()
            .enumerate()
            .flat_map(|(j, a)| {
                a.iter()
                    .zip(new2.col_as_slice(j))
                    .map(|(x, y)| (x - y).abs())
            })
            .fold(0.0f64, f64::max);
        eprintln!(
            "orthonormalize {len} x {nr}: sequential {t_ref:.2} s, new {t_new:.2} s, max |dq| {dmax:.1e}"
        );
        assert_eq!(new.len(), new2.ncols());
        assert!(dmax < 1e-10);
    }

    /// The solver's own time at a TZVP-size CPKS (no J/K), the sequential reference against
    /// this one: `LINUS_SOLVE_SIZE=len,nr,iterations` (default ts61/def2-TZVP's 182070
    /// unknowns, 183 right-hand sides, 6 iterations); `cargo test --release -p linus --lib
    /// subspace_solve_timing -- --ignored --nocapture` with RAYON_NUM_THREADS.
    #[test]
    #[ignore]
    fn subspace_solve_timing() {
        let spec = std::env::var("LINUS_SOLVE_SIZE").unwrap_or_else(|_| "182070,183,6".into());
        let v: Vec<usize> = spec.split(',').map(|s| s.parse().unwrap()).collect();
        let (len, nr, iters) = (v[0], v[1], v[2]);
        let pr = Problem::new(len, 4, nr, 0);
        let threads = rayon::current_num_threads();
        let par = Par::rayon(threads);
        let mut t_apply = 0.0;
        let mut run = |which: usize| {
            let t = std::time::Instant::now();
            let mut ta = 0.0;
            let f = |zs: &[Vec<f64>]| {
                let t = std::time::Instant::now();
                let r = pr.apply(zs);
                ta += t.elapsed().as_secs_f64();
                Ok(r)
            };
            let out = if which == 0 {
                subspace_solve_reference(&pr.b, &pr.d, 0, 1e-30, iters, par, f).unwrap()
            } else {
                subspace_solve(&pr.b, &pr.d, 0, SolveControl::quiet(1e-30, iters), par, f).unwrap()
            };
            let total = t.elapsed().as_secs_f64();
            t_apply = ta;
            eprintln!(
                "{}: {total:.2} s, of which apply {ta:.2} s; solver {:.2} s ({} iterations)",
                if which == 0 { "reference" } else { "new" },
                total - ta,
                out.2.iterations
            );
            out
        };
        let a = run(1);
        let b = run(0);
        let d = (0..a.0.ncols())
            .flat_map(|j| {
                a.0.col_as_slice(j)
                    .iter()
                    .zip(b.0.col_as_slice(j))
                    .map(|(x, y)| (x - y).abs())
                    .collect::<Vec<_>>()
            })
            .fold(0.0f64, f64::max);
        eprintln!("max |dx| against the sequential solver: {d:.1e}");
        let _ = t_apply;
    }
}
