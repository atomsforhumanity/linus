//! The lowest energy the SCF visited: the saddle check's reference (a converged UHF/UKS point
//! above it is not a minimum along the path) and the escalation's start at `max_iter`.

/// The lowest energy of a density from orbitals since the last change of the J/K builder's
/// approximation, and its iteration (infinity and 0: none yet).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LowestEnergy {
    pub energy: f64,
    pub iter: usize,
}

impl Default for LowestEnergy {
    fn default() -> Self {
        LowestEnergy::new()
    }
}

impl LowestEnergy {
    pub fn new() -> Self {
        LowestEnergy {
            energy: f64::INFINITY,
            iter: 0,
        }
    }

    /// Forget it (the J/K builder moved to its next approximation: energies of two operators do
    /// not compare).
    pub fn reset(&mut self) {
        *self = LowestEnergy::new();
    }

    /// Iteration `iter`'s energy (the iteration `guess_iter`'s density was a guess and does not
    /// count): whether it is the new lowest.
    pub fn observe(&mut self, iter: usize, guess_iter: usize, energy: f64) -> bool {
        if iter > guess_iter && energy < self.energy {
            (self.energy, self.iter) = (energy, iter);
            true
        } else {
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_guess_density_and_non_finite_energies_never_count() {
        let mut l = LowestEnergy::new();
        assert!(!l.observe(1, 1, -10.0), "the guess density's iteration");
        assert!(l.observe(2, 1, -1.0));
        assert!(!l.observe(3, 1, f64::NAN));
        assert!(!l.observe(4, 1, -0.5));
        assert!(l.observe(5, 1, -2.0));
        assert_eq!((l.energy, l.iter), (-2.0, 5));
        l.reset();
        assert!(l.observe(6, 1, -0.1));
    }
}
