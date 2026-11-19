//! The reference determinant that derivatives and response properties expand around: the
//! converged SCF's orbitals cut into per-spin-channel blocks, with the densities built from
//! them, computed once per SCF result and borrowed by the gradient, the Hessian, the
//! polarizability and their CPHF/CPKS (BAGEL's explicit reference passing).
//!
//! A closed-shell (restricted) reference has one channel standing for both spins, with
//! occupation 2; an open-shell (unrestricted) one has the two spins, occupation 1
//! ([`Channels`]). Per channel the block holds the occupied and virtual orbitals and their
//! energies and the channel density D_s = C_os C_os^T; the reference holds the total density
//! D_t and the energy-weighted density W = sum_s f C_os e_os C_os^T.
//!
//! The arithmetic is fixed here for every consumer, and the results' bits depend on it:
//! D_t = 2 D (restricted, one channel scaled) or D_alpha + D_beta (unrestricted), and
//! W = 2 W_alpha or W_alpha + W_beta likewise, each W_s one GEMM of C_os e_os by C_os^T. The
//! orbital blocks are contiguous copies of the SCF's columns (the layout every product of the
//! derivatives and the response has always taken).

use crate::ScfResult;
use faer::{Mat, MatRef, Par};
use linus_base::linalg::mm;
use linus_core::channels::Channels;

/// One spin channel of a [`Reference`]: its occupied and virtual orbitals (nbf x nocc and
/// nbf x nvir), their energies, and its density D_s = C_o C_o^T.
pub struct OrbitalBlock {
    pub c_occ: Mat<f64>,
    pub c_vir: Mat<f64>,
    pub eps_occ: Vec<f64>,
    pub eps_vir: Vec<f64>,
    pub density: Mat<f64>,
}

impl OrbitalBlock {
    /// The occupied orbitals.
    pub fn nocc(&self) -> usize {
        self.c_occ.ncols()
    }
}

/// The reference of the derivatives and responses of one SCF result (see the module docs).
pub struct Reference {
    /// The channels' orbital blocks: one (restricted) or two (alpha, beta).
    pub channels: Channels<OrbitalBlock>,
    /// D_t = sum_s f D_s.
    pub total_density: Mat<f64>,
    /// W = sum_s f C_os e_os C_os^T.
    pub energy_weighted: Mat<f64>,
}

impl Reference {
    /// The reference of the converged SCF `r` (its products at parallelism `par`; faer's GEMM
    /// gives the same bits at any).
    pub fn new(r: &ScfResult, par: Par) -> Self {
        let n = r.c_a.nrows();
        let nmo = r.nmo;
        let block = |s: usize| {
            let (c, eps, nocc) = if s == 0 {
                (&r.c_a, &r.eps_a, r.nalpha)
            } else {
                (&r.c_b, &r.eps_b, r.nbeta)
            };
            let c_occ = c.subcols(0, nocc).to_owned();
            let density = mm(c_occ.as_ref(), c_occ.transpose(), par);
            OrbitalBlock {
                c_vir: c.subcols(nocc, nmo - nocc).to_owned(),
                eps_occ: eps[..nocc].to_vec(),
                eps_vir: eps[nocc..nmo].to_vec(),
                c_occ,
                density,
            }
        };
        let channels = Channels::from_fn(r.restricted, block);
        // C_os e_os C_os^T
        let weighted = |b: &OrbitalBlock| {
            let ce = Mat::from_fn(n, b.nocc(), |i, k| b.c_occ[(i, k)] * b.eps_occ[k]);
            mm(ce.as_ref(), b.c_occ.transpose(), par)
        };
        let (total_density, energy_weighted) = match &channels {
            Channels::Restricted(b) => (&b.density * 2.0, weighted(b) * 2.0),
            Channels::Unrestricted([a, b]) => (&a.density + &b.density, weighted(a) + weighted(b)),
        };
        Reference {
            channels,
            total_density,
            energy_weighted,
        }
    }

    /// The spin channels: 1 (restricted) or 2.
    pub fn nch(&self) -> usize {
        self.channels.count()
    }

    /// Electrons per occupied orbital: 2 (restricted) or 1.
    pub fn occupation(&self) -> f64 {
        self.channels.occupation()
    }

    /// The occupied orbitals per channel, `[s]` (the layout of the response J/K and kernels).
    pub fn c_occ(&self) -> Vec<MatRef<'_, f64>> {
        self.channels.iter().map(|b| b.c_occ.as_ref()).collect()
    }

    /// The virtual orbitals per channel, `[s]`.
    pub fn c_vir(&self) -> Vec<MatRef<'_, f64>> {
        self.channels.iter().map(|b| b.c_vir.as_ref()).collect()
    }

    /// The occupied orbital energies per channel, `[s]`.
    pub fn eps_occ(&self) -> Vec<&[f64]> {
        self.channels.iter().map(|b| &b.eps_occ[..]).collect()
    }

    /// The virtual orbital energies per channel, `[s]`.
    pub fn eps_vir(&self) -> Vec<&[f64]> {
        self.channels.iter().map(|b| &b.eps_vir[..]).collect()
    }
}
