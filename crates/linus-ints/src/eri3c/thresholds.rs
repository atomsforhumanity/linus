//! Screening thresholds of density fitting: the one owner of every threshold of the three-center
//! integrals and of the derivatives of the three- and two-center integrals, read by the CPU and
//! GPU engines alike (the SCF's J/K, the gradients, the Hessians and the responses).

/// Screening thresholds of the density-fitting integrals (P|mn) and (P|Q) and their derivatives.
/// Each field is one kind of screening; the values differ because the quantities they bound do
/// (a plain Schwarz bound, a bound times a density, a block of integrals kept for later
/// contractions). Where two engines screened the same quantity at different values, the field
/// takes the tighter one.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DfThresholds {
    /// Shell pairs (mn| whose Schwarz bound is below this are skipped.
    pub pair: f64,
    /// Primitive pairs below this are dropped from a shell pair's contraction.
    pub prim: f64,
    /// The primitive threshold of the derivative integrals: primitive pairs against lane
    /// batches (CPU) or auxiliary primitives (GPU) whose derivative bound is below this are
    /// skipped.
    pub deriv_prim: f64,
    /// Orbital pairs dropped from a derivative layout: their derivative bound times the largest
    /// auxiliary bound is below this (the gradient's and the Hessian's derivative blocks).
    pub deriv_pair: f64,
    /// Density-weighted screening of the derivative contractions with a three-index density:
    /// pairs, lane batches and primitive pairs whose derivative (or second-derivative) bound
    /// times the largest |Γ| is below this (the gradient's and the Hessian's three-center terms).
    pub deriv_weighted: f64,
    /// Primitive screening of the derivative-integral blocks a Hessian keeps for later
    /// contractions (no density to weigh them by).
    pub deriv_block: f64,
    /// The two-center derivative integrals and contractions (the metric's first and second
    /// derivatives in gradients and Hessians).
    pub metric_deriv: f64,
}

impl Default for DfThresholds {
    fn default() -> Self {
        Self {
            pair: 1e-14,
            prim: 1e-16,
            deriv_prim: 1e-16,
            deriv_pair: 1e-18,
            deriv_weighted: 1e-14,
            deriv_block: 1e-15,
            metric_deriv: 1e-14,
        }
    }
}

impl DfThresholds {
    /// No screening at all (references and tests).
    pub fn none() -> Self {
        Self {
            pair: 0.0,
            prim: 0.0,
            deriv_prim: 0.0,
            deriv_pair: 0.0,
            deriv_weighted: 0.0,
            deriv_block: 0.0,
            metric_deriv: 0.0,
        }
    }
}
