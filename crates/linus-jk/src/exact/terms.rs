//! The density factors of the exact two-electron derivatives, as `linus_ints`' typed
//! [`DensityTerms`]: the spin and hybrid semantics of a calculation (which densities enter J
//! and K, with which coefficients) live here, not in the integrals crate.

use linus_ints::contract::DensityTerms;

/// The densities of a derivative contraction (row-major nbf x nbf): closed shell the total
/// density, open shell the total and the two spin densities.
#[derive(Clone, Copy)]
pub enum SpinDensities<'a> {
    Closed {
        dt: &'a [f64],
    },
    Open {
        dt: &'a [f64],
        da: &'a [f64],
        db: &'a [f64],
    },
}

impl<'a> SpinDensities<'a> {
    /// `restricted`: the total density `dt` (the spin densities are not read); else the spin
    /// densities `da`, `db` too.
    pub fn new(restricted: bool, dt: &'a [f64], da: &'a [f64], db: &'a [f64]) -> Self {
        if restricted {
            SpinDensities::Closed { dt }
        } else {
            SpinDensities::Open { dt, da, db }
        }
    }

    /// `terms` plus J of the total density: 1/2 D_t ⊗ D_t.
    pub fn coulomb(self, terms: DensityTerms<'a>) -> DensityTerms<'a> {
        let dt = match self {
            SpinDensities::Closed { dt } | SpinDensities::Open { dt, .. } => dt,
        };
        terms.coulomb(0.5, dt, dt)
    }

    /// `terms` plus c times the exchange: -c/4 D_t ⊗ D_t closed shell, -c/2 D_s ⊗ D_s per spin
    /// open shell.
    pub fn exchange(self, terms: DensityTerms<'a>, c: f64) -> DensityTerms<'a> {
        match self {
            SpinDensities::Closed { dt } => terms.exchange(-0.25 * c, dt, dt),
            SpinDensities::Open { da, db, .. } => {
                terms.exchange(-0.5 * c, da, da).exchange(-0.5 * c, db, db)
            }
        }
    }

    /// J plus, for a_x != 0, a_x times the exchange (HF and global hybrids).
    pub(crate) fn coulomb_exchange(self, a_x: f64) -> DensityTerms<'a> {
        let terms = self.coulomb(DensityTerms::default());
        if a_x != 0.0 {
            self.exchange(terms, a_x)
        } else {
            terms
        }
    }
}
