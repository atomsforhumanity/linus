//! Solvent terms of derivatives: the fixed-density gradient of the PCM energy, its explicit
//! second derivatives and Fock-matrix derivatives for Hessians, the reaction field's linear
//! response for CPHF/CPKS (an [`EnvironmentResponse`] of the response operator), and the hooks
//! the gradient, Hessian and response code call.

use super::pcm::PcmOperator;
use super::{PcmScf, PcmScfMode};
use faer::{Mat, Par};
use linus_base::error::LinusError;
use linus_base::terms::EnvironmentResponse;
use linus_core::timing::Timers;
use linus_ints::points::pcm_ints::SurfaceShape;
use linus_ints::{DerivOrder, PairOptions, PairSet};

impl PcmScf {
    /// Energy and fixed-density nuclear gradient (natom x 3) of the solvent term for the total
    /// density `d` (row-major nbf^2): nuclear-potential, integral (shell centres and points) and
    /// surface/matrix terms.
    pub fn fixed_density_gradient(
        &self,
        z: &[u32],
        positions: &[[f64; 3]],
        d: &[f64],
    ) -> (f64, Vec<[f64; 3]>) {
        self.fixed_density_gradient_timed(z, positions, d, None, &mut Timers::default())
    }

    /// [`PcmScf::fixed_density_gradient`] with its parts timed (the density-fitted PCM: `ri`),
    /// on the shell pairs `pairs` of the basis with first derivatives when given (the
    /// geometry's, `linus::context`; else built here when the integrals need them).
    pub fn fixed_density_gradient_timed(
        &self,
        z: &[u32],
        positions: &[[f64; 3]],
        d: &[f64],
        pairs: Option<&PairSet>,
        timers: &mut Timers,
    ) -> (f64, Vec<[f64; 3]>) {
        if self.mode() == PcmScfMode::Df {
            return self.ri_fixed_density_gradient(z, positions, d, timers);
        }
        let ve = timers.time("grad_solvent_potential", || self.electron_potential(&[d]));
        let phi: Vec<f64> = self.op.v_nuc.iter().zip(&ve).map(|(a, b)| a - b).collect();
        let c = self.op.solve(&phi);
        let zf: Vec<f64> = z.iter().map(|&x| x as f64).collect();
        let (mut g, gs) = timers.time("grad_solvent_surface", || {
            (
                self.op.nuclear_gradient(&zf, positions, &c.q_sym),
                self.op.solver_gradient(positions, &phi, &c),
            )
        });
        let x = self.exact();
        // the batched Rys kernels (raised/lowered shells) in every SCF mode; the
        // McMurchie-Davidson reference only on request (the dev switch rys_gradient)
        let rys = self.resolved.grad_rys.unwrap_or(true);
        let (ga, gc) = timers.time("grad_solvent_ints", || {
            let mut ga = vec![[0.0; 3]; self.basis.natom()];
            let mut gc = vec![[0.0; 3]; x.points().len()];
            if rys {
                let mut ws = x.workspace(SurfaceShape {
                    gradient: true,
                    ..Default::default()
                });
                x.gradient(&c.q_sym, d, &mut ga, &mut gc, &mut ws);
            } else {
                let built;
                let dpairs = match pairs {
                    Some(p) => p,
                    None => {
                        built =
                            PairSet::build(&self.basis, PairOptions::with_order(DerivOrder::One));
                        &built
                    }
                };
                let plan = x
                    .deriv_plan(dpairs, &self.op.surf.owner)
                    .expect("pairs with first derivatives");
                plan.gradient(&c.q_sym, d, &mut ga, &mut gc, &mut plan.workspace());
            }
            (ga, gc)
        });
        for a in 0..g.len() {
            for k in 0..3 {
                g[a][k] += gs[a][k] - ga[a][k];
            }
        }
        for (i, v) in gc.iter().enumerate() {
            let o = self.op.surf.owner[i];
            for k in 0..3 {
                g[o][k] -= v[k];
            }
        }
        (PcmOperator::energy(&phi, &c), g)
    }
}

/// PCM parts of an analytic Hessian at fixed density.
pub struct PcmHessianTerms {
    /// Explicit second derivative of the solvent energy, (3N x 3N) row-major.
    pub hess: Vec<f64>,
    /// Fixed-density derivatives of the solvent Fock matrix, one nbf x nbf matrix per X.
    pub fx: Vec<Mat<f64>>,
}

impl PcmScf {
    /// Solvent Fock contributions of first-order (symmetric) densities D1: the linear response
    /// -sum_g (Q_sym phi1)_g (ab|g) with phi1 = -v_e\[D1\] (no nuclear part).
    pub fn response_fock(&self, ds: &[Mat<f64>]) -> Vec<Mat<f64>> {
        if ds.is_empty() {
            return Vec::new();
        }
        let np = self.op.npoints();
        let flat: Vec<Vec<f64>> = ds
            .iter()
            .map(|d| linus_base::linalg::mat_to_rowmajor(d.as_ref()))
            .collect();
        let refs: Vec<&[f64]> = flat.iter().map(|v| v.as_slice()).collect();
        let ve = self.electron_potential(&refs);
        let phis: Vec<f64> = ve.iter().map(|x| -x).collect();
        let q = self.op.q_sym_many(&phis);
        let qs: Vec<&[f64]> = (0..ds.len()).map(|k| &q[k * np..(k + 1) * np]).collect();
        self.fock(&qs)
    }

    /// Explicit (fixed-density) second derivatives of the solvent energy and the
    /// fixed-density derivatives of its Fock matrix, for total density `d` (row-major nbf^2);
    /// see `pcm_hess`.
    pub fn hessian_terms(&self, z: &[u32], positions: &[[f64; 3]], d: &[f64]) -> PcmHessianTerms {
        let pairs = PairSet::build(&self.basis, PairOptions::with_order(DerivOrder::Two));
        self.hessian_terms_with(z, positions, d, &pairs)
    }

    /// [`PcmScf::hessian_terms`] on the shell pairs `pairs` of the basis with second
    /// derivatives (the geometry's, `linus::context`).
    pub fn hessian_terms_with(
        &self,
        z: &[u32],
        positions: &[[f64; 3]],
        d: &[f64],
        pairs: &PairSet,
    ) -> PcmHessianTerms {
        let nbf = self.basis.nbf;
        let natom = positions.len();
        let n3 = 3 * natom;
        let np = self.op.npoints();
        let ve = self.electron_potential(&[d]);
        let phi: Vec<f64> = self.op.v_nuc.iter().zip(&ve).map(|(a, b)| a - b).collect();
        let c = self.op.solve(&phi);
        let zf: Vec<f64> = z.iter().map(|&x| x as f64).collect();
        let (dvn, hn) = self.op.nuclear_derivs(&zf, positions, &c.q_sym);
        let owner = &self.op.surf.owner;
        let x = self.exact();
        let plan = x
            .deriv_plan(pairs, owner)
            .expect("pairs with second derivatives");
        let mut ws = plan.workspace();
        let mut dve = vec![0.0; n3 * np];
        plan.potentials_deriv(d, &mut dve, &mut ws);
        let mut dpot = vec![0.0; n3 * nbf * nbf];
        plan.matrices_deriv(&c.q_sym, &mut dpot);
        let mut he = vec![0.0; n3 * n3];
        plan.hessian(&c.q_sym, d, &mut he, &mut ws);

        let dphi: Vec<f64> = dvn.iter().zip(&dve).map(|(a, b)| a - b).collect();
        let q1 = self.op.q_sym_many(&dphi);
        let (w, h2) = self.op.solver_second(positions, &phi, &c);
        // [X][Y] = a_X . b_Y
        let dotx = |a: &[f64], b: &[f64]| -> Vec<f64> {
            let am = faer::MatRef::from_column_major_slice(a, np, n3);
            let bm = faer::MatRef::from_column_major_slice(b, np, n3);
            let m = linus_base::linalg::mm(am.transpose(), bm, Par::Seq);
            linus_base::linalg::mat_to_rowmajor(m.as_ref())
        };
        let a1 = dotx(&dphi, &q1);
        let a2 = dotx(&dphi, &w);
        let mut hess = vec![0.0; n3 * n3];
        for x in 0..n3 {
            for y in 0..n3 {
                let k = x * n3 + y;
                hess[k] = a1[k] + a2[k] + a2[y * n3 + x] + hn[k] - he[k] + h2[k];
            }
        }
        // symmetrize (exact up to round-off)
        for x in 0..n3 {
            for y in 0..x {
                let v = 0.5 * (hess[x * n3 + y] + hess[y * n3 + x]);
                hess[x * n3 + y] = v;
                hess[y * n3 + x] = v;
            }
        }
        // F^x = -sum_g (dq_sym/dX)_g (ab|g) - d/dX sum_g q_sym,g (ab|g)
        let dq: Vec<f64> = q1.iter().zip(&w).map(|(a, b)| a + b).collect();
        let qs: Vec<&[f64]> = (0..n3).map(|x| &dq[x * np..(x + 1) * np]).collect();
        let vq = self.fock(&qs);
        let n2 = nbf * nbf;
        let fx = (0..n3)
            .map(|x| Mat::from_fn(nbf, nbf, |i, j| vq[x][(i, j)] - dpot[x * n2 + i * nbf + j]))
            .collect();
        PcmHessianTerms { hess, fx }
    }
}

/// The reaction field's linear response in the response operator
/// (`linus_scf::response::operator`), over spin channels: for first-order densities
/// dD_s = A_s C_s^T + C_s A_s^T given by their factors `a[s][k]` and the occupied orbitals
/// `c_occ[s]`, dV\[dD\] C_s as `[s][k]`, dV the solvent's Fock response to the total
/// first-order density dD = sum_s w_s dD_s (one channel: a closed shell, w = 2; two: the spins,
/// w = 1). Timed as `cphf_solvent`.
impl EnvironmentResponse for PcmScf {
    fn timer(&self) -> &'static str {
        "cphf_solvent"
    }

    fn induced_fock_occ(
        &self,
        c_occ: &[faer::MatRef<'_, f64>],
        a: &[Vec<Mat<f64>>],
        par: Par,
    ) -> Result<Vec<Vec<Mat<f64>>>, LinusError> {
        let mm = linus_base::linalg::mm;
        let co = c_occ;
        let npert = a.first().map_or(0, |v| v.len());
        let ds: Vec<Mat<f64>> = (0..npert)
            .map(|k| {
                if co.len() == 1 {
                    let t = mm(a[0][k].as_ref(), co[0].transpose(), par);
                    (&t + t.transpose()) * 2.0
                } else {
                    let mut t = mm(a[0][k].as_ref(), co[0].transpose(), par);
                    for s in 1..co.len() {
                        t += mm(a[s][k].as_ref(), co[s].transpose(), par);
                    }
                    &t + t.transpose()
                }
            })
            .collect();
        let dv = self.response_fock(&ds);
        Ok(co
            .iter()
            .map(|c| dv.iter().map(|v| mm(v.as_ref(), *c, par)).collect())
            .collect())
    }
}
