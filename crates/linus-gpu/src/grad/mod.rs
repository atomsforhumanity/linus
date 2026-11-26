//! Analytic nuclear gradients on the device: the density-fitted two-electron term
//! ([`df`], `GpuDfGrad`) and the XC term with grid response ([`xc`], `GpuXcGrad`), each the
//! device counterpart of a linus CPU term at the same converged orbitals:
//!
//! - [`df`]: linus's `two_electron::df::gradient::df_gradient` for J and a global hybrid's K
//!   (Coulomb metric, one auxiliary basis): the three-center derivative integrals of
//!   [`crate::eri3c`] contracted on the device, the two-center metric derivative on the host;
//! - [`xc`]: linus's `XcIntegrator::gradient` on the same grid and functional, with the grid
//!   response (the Becke weight derivatives on the device; other partitions on the host).
//!
//! Like the energy engines they take plain host or device matrices and know nothing of linus's
//! types; linus would wrap them behind its gradient dispatch (`gradient_2e` and the XC
//! gradient), with the other terms (one-electron, ECP, solvent, dispersion) on the CPU.
//!
//! Atom sums use the deterministic fixed-point accumulator (`detsum`), and every other
//! reduction has a fixed order, so both gradients are bitwise reproducible run to run on a
//! device.

pub mod df;
#[cfg(feature = "gpu")]
pub(crate) mod df_j;
pub mod xc;

/// Adds `b` to `a` atom by atom (`a` grows to `b`'s length).
pub fn add_atoms(a: &mut Vec<[f64; 3]>, b: &[[f64; 3]]) {
    if a.len() < b.len() {
        a.resize(b.len(), [0.0; 3]);
    }
    for (x, y) in a.iter_mut().zip(b) {
        for k in 0..3 {
            x[k] += y[k];
        }
    }
}

/// Largest |a - b| over atoms and directions (a missing atom counts as zero).
pub fn max_abs_diff(a: &[[f64; 3]], b: &[[f64; 3]]) -> f64 {
    let n = a.len().max(b.len());
    let z = [0.0; 3];
    (0..n)
        .map(|i| {
            let (x, y) = (a.get(i).unwrap_or(&z), b.get(i).unwrap_or(&z));
            (0..3).map(|k| (x[k] - y[k]).abs()).fold(0.0, f64::max)
        })
        .fold(0.0, f64::max)
}

/// True if both gradients have the same bits.
pub fn bitwise_equal(a: &[[f64; 3]], b: &[[f64; 3]]) -> bool {
    a.len() == b.len()
        && a.iter()
            .zip(b)
            .all(|(x, y)| (0..3).all(|k| x[k].to_bits() == y[k].to_bits()))
}
