//! The SCF's state between Fock builds: the orbitals the current density comes from, and the
//! density.

use super::density;
use faer::{Mat, Par};
use linus_base::linalg::mm;

/// The orbitals and spin densities the next Fock build is evaluated at (restricted: one channel,
/// c_b = c_a and D_b = D_a).
pub(crate) struct OrbitalState {
    pub restricted: bool,
    /// Occupied orbitals per spin, (n_alpha, n_beta).
    pub nocc: [usize; 2],
    /// Orbitals per spin (nbf x nmo, occupied first) and the orbital energies of the last
    /// diagonalization.
    pub c_a: Mat<f64>,
    pub c_b: Mat<f64>,
    pub eps_a: Vec<f64>,
    pub eps_b: Vec<f64>,
    pub da: Mat<f64>,
    pub db: Mat<f64>,
    /// The occupied orbitals of the densities per channel, D_s = C_s C_s^T (None: the density is a
    /// guess, not from orbitals).
    pub occ: Option<Vec<Mat<f64>>>,
}

impl OrbitalState {
    /// The state of the guess densities `(da, db)` (None: orbitals of their own).
    pub fn new(restricted: bool, nocc: [usize; 2], nbf: usize, da: Mat<f64>, db: Mat<f64>) -> Self {
        OrbitalState {
            restricted,
            nocc,
            c_a: Mat::zeros(nbf, nbf),
            c_b: Mat::zeros(nbf, nbf),
            eps_a: Vec::new(),
            eps_b: Vec::new(),
            da,
            db,
            occ: None,
        }
    }

    /// The occupied orbitals of `c_a` and `c_b` per channel (one for a closed shell).
    pub fn occupied_of(&self, c_a: &Mat<f64>, c_b: &Mat<f64>) -> Vec<Mat<f64>> {
        let [na, nb] = self.nocc;
        let ca = c_a.subcols(0, na).to_owned();
        if self.restricted {
            vec![ca]
        } else {
            vec![ca, c_b.subcols(0, nb).to_owned()]
        }
    }

    /// The densities and occupied orbitals of the current orbitals `c_a`, `c_b`.
    pub fn refresh(&mut self, par: Par) {
        let [na, nb] = self.nocc;
        self.da = density(self.c_a.as_ref(), na, par);
        self.db = density(self.c_b.as_ref(), nb, par);
        self.occ = Some(self.occupied_of(&self.c_a, &self.c_b));
    }

    /// New orbitals per spin, and their densities.
    pub fn set_orbitals(&mut self, c_a: Mat<f64>, c_b: Mat<f64>, par: Par) {
        (self.c_a, self.c_b) = (c_a, c_b);
        self.refresh(par);
    }

    /// The densities of the occupied orbitals `occ` alone (per channel), the orbitals unchanged:
    /// the escalation's start, which keeps only the lowest-energy iterate's occupied orbitals.
    pub fn set_occupied(&mut self, occ: Vec<Mat<f64>>, par: Par) {
        self.da = mm(occ[0].as_ref(), occ[0].transpose(), par);
        self.db = if self.restricted {
            self.da.clone()
        } else {
            mm(occ[1].as_ref(), occ[1].transpose(), par)
        };
        self.occ = Some(occ);
    }

    /// The densities per channel (one for a closed shell).
    pub fn densities(&self) -> Vec<Mat<f64>> {
        if self.restricted {
            vec![self.da.clone()]
        } else {
            vec![self.da.clone(), self.db.clone()]
        }
    }
}
