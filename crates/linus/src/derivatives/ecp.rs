//! Effective core potentials in gradients and Hessians: the ECP is a one-electron operator, so
//! its derivatives join the core-Hamiltonian derivative terms (the integrals are linus-ints'
//! [`EcpPlan`]). The core Hamiltonian itself gets the ECP in [`Session::one_electron`];
//! nuclear charges are Z - ncore ([`crate::Structure`]).

use crate::driver::CalcContext;
use linus_ints::{DerivOrder, EcpPlan, PairSet};

impl CalcContext {
    /// sum_ab D_ab dU_ECP,ab/dX per atom (zeros without ECPs); `d` the total density, row-major.
    pub fn ecp_gradient(&self, pairs: &PairSet, d: &[f64]) -> Vec<[f64; 3]> {
        if self.ecp.is_empty() {
            return vec![[0.0; 3]; self.mol.natom()];
        }
        let plan = EcpPlan::new(&self.basis, pairs, &self.ecp, DerivOrder::One);
        let mut g = vec![[0.0; 3]; self.basis.natom()];
        plan.gradient(d, &mut g, &mut plan.workspace());
        g
    }

    /// Hessian terms of the ECPs: the explicit second derivative sum_ab D_ab d2U_ab/dXdY added
    /// to `h` ((3 natom)^2) and dU/dX added to the core-Hamiltonian derivatives `hx`.
    pub fn add_ecp_hessian_terms(&self, pairs: &PairSet, d: &[f64], h: &mut [f64], hx: &mut [f64]) {
        if self.ecp.is_empty() {
            return;
        }
        // one plan for both: the pairs near the ECPs and their expansions found once (E^A has
        // the same bits at orders one and two)
        let plan = EcpPlan::new(&self.basis, pairs, &self.ecp, DerivOrder::Two);
        let mut ws = plan.workspace();
        let mut h2 = vec![0.0; h.len()];
        plan.hessian(d, &mut h2, &mut ws);
        h.iter_mut().zip(&h2).for_each(|(a, b)| *a += b);
        plan.add_deriv_matrices(hx, &mut ws);
    }
}
