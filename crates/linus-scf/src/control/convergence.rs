//! The convergence test: |dE| < e_tol and the orbital gradient, in the norm `g_norm`, below
//! g_tol.

use linus_base::convergence::GradNorm;

/// The SCF's convergence criteria ([`crate::ScfConfig`]'s `e_tol`, `g_tol`, `g_norm`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ConvergenceTest {
    pub e_tol: f64,
    pub g_tol: f64,
    pub g_norm: GradNorm,
}

impl ConvergenceTest {
    /// The orbital gradient in the convergence norm, from max |F_ai| and PySCF's norm.
    pub fn gconv(&self, gmax: f64, gnorm_pyscf: f64) -> f64 {
        match self.g_norm {
            GradNorm::Max => gmax,
            GradNorm::Pyscf => gnorm_pyscf,
        }
    }

    /// The criteria at iteration `iter` (the first after a guess density's is `guess_iter` + 1):
    /// never before iteration 2 (3 after a guess density, whose energy is not one of orbitals),
    /// |dE| < e_tol and `gconv` < g_tol. A full build at an unchanged density (`rebuilt`: the
    /// confirmation of an incremental build, or the first build after a restart) meets the energy
    /// criterion by itself: its dE only measures the incremental builds' error, which cancels in
    /// the dE of incremental energies.
    pub fn met(&self, iter: usize, guess_iter: usize, de: f64, gconv: f64, rebuilt: bool) -> bool {
        iter > 1
            && iter > guess_iter + 1
            && (rebuilt || de.abs() < self.e_tol)
            && gconv < self.g_tol
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_criteria_wait_for_a_density_from_orbitals() {
        let t = ConvergenceTest {
            e_tol: 1e-8,
            g_tol: 1e-6,
            g_norm: GradNorm::Max,
        };
        assert!(!t.met(1, 0, 0.0, 0.0, false), "never at iteration 1");
        assert!(t.met(2, 0, 1e-9, 1e-7, false));
        assert!(
            !t.met(2, 1, 1e-9, 1e-7, false),
            "a guess density: from iteration 3"
        );
        assert!(t.met(3, 1, 1e-9, 1e-7, false));
        assert!(!t.met(3, 0, 1e-7, 1e-7, false), "dE too large");
        assert!(
            t.met(3, 0, 1e-7, 1e-7, true),
            "a rebuild meets the energy criterion"
        );
        assert!(
            !t.met(3, 0, 1e-9, 1e-6, false),
            "the gradient bound is strict"
        );
        let p = ConvergenceTest {
            g_norm: GradNorm::Pyscf,
            ..t
        };
        assert_eq!((t.gconv(1.0, 2.0), p.gconv(1.0, 2.0)), (1.0, 2.0));
    }
}
