//! Coupled-perturbed Hartree-Fock / Kohn-Sham for real, static perturbations, closed and open
//! shells in one formulation over spin channels ([`ResponseProblem`]): one channel for a closed
//! shell (the alpha orbitals, occupation 2), two for an open shell (the spins).
//!
//! Per channel s the unknowns are U_s (nvir_s x nocc_s) per perturbation, the first-order
//! density dD_s = C_vs U_s C_os^T + h.c., the induced Fock matrix
//! dF_s = c_j J[sum_s' dD_s'] - a_x K\[dD_s\] + Vxc_s^(1) (+ the environment's response;
//! closed shell: c_j = 2 and Vxc^(1)[2 dD], since both spins carry dD; open shell: c_j = 1,
//! Vxc_s^(1)[dD_a, dD_b]), and the response equations
//! (A + B)_s U = (e_a - e_i)_s U_s + C_vs^T dF_s C_os = -B^x_s. The operator is symmetric in the
//! inner product sum_s U_s . V_s, with all channels in one vector.
//!
//! A problem is the zeroth-order part (the virtual orbitals and the orbital energies of each
//! channel, borrowed from the [`Reference`]) and the first-order part, the induced Fock matrix of
//! the [`ResponseOperator`] composed for this response, which projects dF on the occupied
//! orbitals: dF C_o is all the equations need.
//!
//! The right-hand sides are a perturbation's ([`crate::response::perturbation`]); the problem
//! gives the induced Fock matrices they and the second-order properties take
//! ([`ResponseProblem::induced_oo`], [`ResponseProblem::induced_vo`]) and the final residuals
//! that make those properties stationary ([`ResponseProblem::residuals`],
//! [`crate::response::stationary`]).
//!
//! Solver: [`ResponseProblem::solve`], one shared, growing subspace for all right-hand sides
//! ([`subspace_solve`]: about half the iterations of independent Jacobi-preconditioned
//! conjugate gradients, and it can return dF\[U\] C_o of the solution). Each iteration is one
//! batched induced-Fock build for all active perturbations (the case integral digestion
//! handles best).
//!
//! Matrices of all channels and perturbations are lists `x[s][k]` (channel s, perturbation k),
//! the layout of [`linus_base::terms::ResponseJk::fock_occ`] and
//! [`linus_base::terms::ResponseKernel::contract_occ`].

use crate::reference::Reference;
use crate::response::operator::ResponseOperator;

/// The smallest diagonal element of the response solver's Jacobi preconditioner (Eh), which is
/// |e_a - e_i| floored here: a positive preconditioner however the reference is occupied. At a
/// non-aufbau reference (an empty orbital below an occupied one, `linus_scf::NonAufbau`: the
/// escalation's states) the hole pair's e_a - e_i is negative, and at an exactly degenerate pair
/// it is zero, a division by zero. The products keep the signed e_a - e_i, so the solution and an
/// indefinite orbital Hessian (a saddle-point reference) are the operator's; the floor lies far
/// below the frontier gaps of aufbau references, whose preconditioner and bits it leaves alone.
pub const PRECONDITIONER_FLOOR: f64 = 1e-6;
use crate::response::perturbation::Perturbation;
use crate::response::solver::{
    map_cols, subspace_solve, Products, ResponseStats, SolveControl, SubspaceSolver,
};
use faer::{Mat, MatRef, Par};
use linus_base::error::LinusError;
use linus_base::linalg::mm;
use linus_base::progress::Progress;
use linus_base::terms::ResponseJk;
use linus_core::timing::Timers;

/// The CPHF/CPKS equations of a response over the spin channels of a reference (see the
/// module docs).
pub struct ResponseProblem<'a> {
    c_vir: Vec<MatRef<'a, f64>>,
    eps_occ: Vec<&'a [f64]>,
    eps_vir: Vec<&'a [f64]>,
    operator: ResponseOperator<'a>,
    /// Where the solve reports its iterations, and the solve's name in the events
    /// ([`ResponseProblem::report_to`]; off by default).
    progress: (Progress, &'static str),
    /// The algebra of the solves elsewhere (a device's; None: the host's [`subspace_solve`]).
    solver: Option<Box<dyn SubspaceSolver + 'a>>,
}

/// The solution of [`ResponseProblem::solve`]: U (`[s][k]`, nvir_s x nocc_s), dF\[U\] C_o when
/// the solve kept its products (`[s][k]`, n x nocc_s), and the solver's statistics.
pub struct ResponseSolution {
    pub u: Vec<Vec<Mat<f64>>>,
    pub fu: Option<Vec<Vec<Mat<f64>>>>,
    pub stats: ResponseStats,
}

impl<'a> ResponseProblem<'a> {
    /// The equations of the channels of `reference` (their virtual orbitals and orbital
    /// energies) with the induced Fock matrix of `operator` (linearized at the reference's
    /// occupied orbitals).
    pub fn new(reference: &'a Reference, operator: ResponseOperator<'a>) -> Self {
        assert_eq!(
            reference.nch(),
            operator.channels(),
            "the reference and the operator of different spin channels"
        );
        ResponseProblem {
            c_vir: reference.c_vir(),
            eps_occ: reference.eps_occ(),
            eps_vir: reference.eps_vir(),
            operator,
            progress: (Progress::off(), ""),
            solver: None,
        }
    }

    /// Report the solve's iterations to `progress` as `response_iteration` events of `solve`
    /// (`hessian`, `polarizability`).
    pub fn report_to(&mut self, progress: Progress, solve: &'static str) {
        self.progress = (progress, solve);
    }

    /// Solve with `solver`'s algebra (a device's) instead of the host's [`subspace_solve`]
    /// (None: the host's, the default).
    pub fn set_solver(&mut self, solver: Option<Box<dyn SubspaceSolver + 'a>>) {
        self.solver = solver;
    }

    fn nch(&self) -> usize {
        self.c_vir.len()
    }

    /// The response operator (a two-level response switches its J/K between the solves).
    pub fn operator(&mut self) -> &mut ResponseOperator<'a> {
        &mut self.operator
    }

    /// The main J/K (to serve another response of the same orbitals,
    /// [`ResponseJk::serve`]) and the phases of the products and the solves (`cphf_*`).
    pub fn into_parts(self) -> (Box<dyn ResponseJk>, Timers) {
        self.operator.into_parts()
    }

    /// Length of a flattened vector (all channels' nvir x nocc blocks).
    fn flat_len(&self) -> usize {
        (0..self.nch())
            .map(|s| self.eps_vir[s].len() * self.eps_occ[s].len())
            .sum()
    }

    /// The channels' blocks (column-major) one after the other.
    fn flatten(x: &[&Mat<f64>], out: &mut [f64]) {
        let mut o = 0;
        for m in x {
            for j in 0..m.ncols() {
                for i in 0..m.nrows() {
                    out[o] = m[(i, j)];
                    o += 1;
                }
            }
        }
    }

    /// The channels' blocks (nrows[s] x ncols[s]) of a flattened vector.
    fn unflatten(v: &[f64], shape: &[(usize, usize)]) -> Vec<Mat<f64>> {
        let mut off = 0;
        shape
            .iter()
            .map(|&(r, c)| {
                let m = Mat::from_fn(r, c, |i, j| v[off + j * r + i]);
                off += r * c;
                m
            })
            .collect()
    }

    /// dF[D^k_oo] C_os for every component of `p`: the induced Fock matrix of the occupied
    /// block's first-order density D^k_oo = -C_os s1oo^k C_os^T of a perturbation that moves the
    /// basis ([`crate::response::perturbation`]), one product of the operator
    /// ([`ResponseOperator::induced_fock_occ`]) for all components; None for a perturbation that
    /// leaves the basis. Its right-hand sides ([`Perturbation::rhs_vo`]) and first-order orbital
    /// energies take it.
    pub fn induced_oo(
        &mut self,
        p: &dyn Perturbation,
        par: Par,
    ) -> Result<Option<Vec<Vec<Mat<f64>>>>, LinusError> {
        let Some(overlap) = p.overlap() else {
            return Ok(None);
        };
        let a = overlap.occupied_factors(p.reference(), par);
        self.operator.induced_fock_occ(&a, par).map(Some)
    }

    /// dF[U^k] C_os of a solution `u` (`[s][k]`, nvir_s x nocc_s), one product: a two-level
    /// response's final one, on the main J/K after [`ResponseOperator::restore`] (its solve
    /// cannot keep the main operator's products).
    pub fn induced_vo(
        &mut self,
        u: &[Vec<Mat<f64>>],
        par: Par,
    ) -> Result<Vec<Vec<Mat<f64>>>, LinusError> {
        let a: Vec<Vec<Mat<f64>>> = self
            .c_vir
            .iter()
            .zip(u)
            .map(|(cv, us)| us.iter().map(|x| mm(*cv, x.as_ref(), par)).collect())
            .collect();
        self.operator.induced_fock_occ(&a, par)
    }

    /// The residuals r = rhs - (A + B) U of a solution `u` (`[s][k]`, nvir_s x nocc_s) from its
    /// dF\[U\] C_o `fu` (n x nocc_s; [`Self::solve`]'s, or rebuilt by
    /// [`Self::induced_vo`]): (A + B) U = (e_a - e_i) U + C_v^T dF\[U\] C_o, so they cost one
    /// small GEMM per perturbation and no product. They are what the stationary (Hylleraas)
    /// forms of second-order properties need: U^x . r^y makes a response property's error
    /// quadratic in the residuals.
    pub fn residuals(
        &self,
        rhs: &[Vec<Mat<f64>>],
        u: &[Vec<Mat<f64>>],
        fu: &[Vec<Mat<f64>>],
        par: Par,
    ) -> Vec<Vec<Mat<f64>>> {
        (0..self.nch())
            .map(|s| {
                let (eo, ev) = (self.eps_occ[s], self.eps_vir[s]);
                rhs[s]
                    .iter()
                    .zip(&u[s])
                    .zip(&fu[s])
                    .map(|((b, x), f)| {
                        let g = mm(self.c_vir[s].transpose(), f.as_ref(), par);
                        Mat::from_fn(ev.len(), eo.len(), |a, i| {
                            b[(a, i)] - (ev[a] - eo[i]) * x[(a, i)] - g[(a, i)]
                        })
                    })
                    .collect()
            })
            .collect()
    }

    /// Solve (A + B) U = rhs for all right-hand sides (`rhs[s][k]`, nvir_s x nocc_s) in one
    /// shared, growing subspace ([`subspace_solve`]). With `keep_products` the solver keeps
    /// dF\[z\] C_o of every basis vector and returns dF\[U\] C_o of the solution, assembled from
    /// them (no extra induced-Fock build); a two-level response's iteration J/K does not give
    /// the main operator's products, so its solve does not keep them. Converged when
    /// max |rhs - (A + B) U| <= `tol` for every right-hand side; an error is a failed product.
    /// The solve's own phases join the operator's timers: the MO transforms of the directions
    /// and products (`cphf_vo`), the subspace algebra (`cphf_solver`; of it the
    /// orthonormalization, `cphf_orth`).
    #[allow(clippy::needless_range_loop, reason = "indexes several arrays in step")]
    pub fn solve(
        &mut self,
        rhs: &[Vec<Mat<f64>>],
        tol: f64,
        max_iter: usize,
        keep_products: bool,
        par: Par,
    ) -> Result<ResponseSolution, LinusError> {
        assert!(
            !(keep_products && self.operator.at_iteration_level()),
            "the iteration J/K of a two-level response does not give the main operator's products"
        );
        let nch = self.nch();
        let nr = rhs.first().map_or(0, |v| v.len());
        let len = self.flat_len();
        let nocc: Vec<usize> = self.eps_occ.iter().map(|e| e.len()).collect();
        let nvir: Vec<usize> = self.eps_vir.iter().map(|e| e.len()).collect();
        let n = self.c_vir[0].nrows();
        let ushape: Vec<(usize, usize)> = (0..nch).map(|s| (nvir[s], nocc[s])).collect();
        let fshape: Vec<(usize, usize)> = (0..nch).map(|s| (n, nocc[s])).collect();
        let (eps_occ, eps_vir) = (&self.eps_occ, &self.eps_vir);
        // the Jacobi preconditioner: |e_a - e_i|, at least PRECONDITIONER_FLOOR (the operator
        // keeps the signed differences)
        let diag: Vec<f64> = (0..nch)
            .flat_map(|s| {
                (0..nocc[s]).flat_map(move |i| {
                    (0..eps_vir[s].len()).map(move |a| {
                        (eps_vir[s][a] - eps_occ[s][i])
                            .abs()
                            .max(PRECONDITIONER_FLOOR)
                    })
                })
            })
            .collect();
        let mut b = Mat::<f64>::zeros(len, nr);
        for k in 0..nr {
            let blocks: Vec<&Mat<f64>> = (0..nch).map(|s| &rhs[s][k]).collect();
            Self::flatten(&blocks, b.col_as_slice_mut(k));
        }
        let elen = if keep_products {
            n * nocc.iter().sum::<usize>()
        } else {
            0
        };
        let (mut t_products, mut t_host) = (0.0, 0.0);
        let c_vir = &self.c_vir;
        let operator = &mut self.operator;
        let solver = &mut self.solver;
        let control = SolveControl {
            tol,
            max_iter,
            progress: self.progress.0.clone(),
            solve: self.progress.1,
        };
        let mut apply = |zs: &[Vec<f64>]| -> Result<Products, LinusError> {
            // (A + B) on the new directions (one batched induced-Fock build); dF[z] C_o of all
            // channels flattened (column-major n x nocc_s blocks) as the extra
            let t_apply = std::time::Instant::now();
            // per direction in parallel, each product sequential: faer's GEMM gives the same
            // bits at any parallelism (`util::linalg::product_par`), so this is bitwise the
            // directions one after another with parallel products
            let us: Vec<Vec<Mat<f64>>> =
                map_cols(zs.len(), par, |k| Self::unflatten(&zs[k], &ushape));
            let a: Vec<Vec<Mat<f64>>> = (0..nch)
                .map(|s| map_cols(us.len(), par, |k| mm(c_vir[s], us[k][s].as_ref(), Par::Seq)))
                .collect();
            let t_fock = std::time::Instant::now();
            let fos = operator.induced_fock_occ(&a, par)?;
            t_products += t_fock.elapsed().as_secs_f64();
            drop(a);
            let fos_ref = &fos;
            let out = map_cols(us.len(), par, |k| {
                let u = &us[k];
                let r: Vec<Mat<f64>> = (0..nch)
                    .map(|s| {
                        let mut r = mm(c_vir[s].transpose(), fos_ref[s][k].as_ref(), Par::Seq);
                        for i in 0..nocc[s] {
                            for a in 0..nvir[s] {
                                r[(a, i)] += (eps_vir[s][a] - eps_occ[s][i]) * u[s][(a, i)];
                            }
                        }
                        r
                    })
                    .collect();
                let mut az = vec![0.0; len];
                Self::flatten(&r.iter().collect::<Vec<_>>(), &mut az);
                let mut fz = vec![0.0; elen];
                if elen > 0 {
                    Self::flatten(
                        &(0..nch).map(|s| &fos_ref[s][k]).collect::<Vec<_>>(),
                        &mut fz,
                    );
                }
                (az, fz)
            });
            t_host += t_apply.elapsed().as_secs_f64();
            Ok(out)
        };
        // the host's solver, or the one given (a device's algebra)
        let (x, fx, st) = match solver.as_mut() {
            Some(s) => s.solve(&b, &diag, elen, control, par, &mut apply)?,
            None => subspace_solve(&b, &diag, elen, control, par, &mut apply)?,
        };
        {
            let t = self.operator.timers_mut();
            t.add("cphf_vo", t_host - t_products);
            t.add("cphf_solver", st.solver_seconds);
            t.add("cphf_orth", st.orth_seconds);
        }
        let mut u: Vec<Vec<Mat<f64>>> = vec![Vec::with_capacity(nr); nch];
        let mut fu: Vec<Vec<Mat<f64>>> = vec![Vec::new(); nch];
        for k in 0..nr {
            for (s, m) in Self::unflatten(x.col_as_slice(k), &ushape)
                .into_iter()
                .enumerate()
            {
                u[s].push(m);
            }
            if elen > 0 {
                for (s, m) in Self::unflatten(fx.col_as_slice(k), &fshape)
                    .into_iter()
                    .enumerate()
                {
                    fu[s].push(m);
                }
            }
        }
        Ok(ResponseSolution {
            u,
            fu: keep_products.then_some(fu),
            stats: st,
        })
    }
}
