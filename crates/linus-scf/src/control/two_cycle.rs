//! The two-cycle detector ([`crate::ScfAlgorithm::two_cycle`]): the energy repeating every
//! other iteration while the SCF is not converging. Occupations flip between two determinants (the
//! CN radical's two pi-hole configurations under DIIS) and DIIS never ends it; the loop answers with
//! quasi-Newton steps from the current orbitals, once per SCF.

/// |E_k - E_k-2| below this (Eh): the energy repeats.
pub(crate) const REPEAT: f64 = 1e-9;
/// |E_k - E_k-1| above this (Eh), or max |F_ai| above [`GRADIENT`]: the SCF is not converging.
pub(crate) const MOVING: f64 = 1e-7;
pub(crate) const GRADIENT: f64 = 1e-4;
/// Iterations in a row that fire the detector.
pub(crate) const RUN: usize = 6;

/// The detector's state: consecutive two-cycle iterations, and the iteration it fired at.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TwoCycle {
    enabled: bool,
    pub run: usize,
    pub at: Option<usize>,
}

impl TwoCycle {
    pub fn new(enabled: bool) -> Self {
        TwoCycle {
            enabled,
            run: 0,
            at: None,
        }
    }

    /// Iteration `iter` (energy `energy`, max |F_ai| `gmax`) after the iterations whose energies
    /// are `previous` (E_k-1, E_k-2; None before the history has them): whether the detector fires
    /// now. Off once it fired, and while the SCF already takes quasi-Newton steps (`descending`,
    /// after a restart); the count then stays as it was.
    pub fn observe(
        &mut self,
        iter: usize,
        energy: f64,
        previous: [Option<f64>; 2],
        gmax: f64,
        descending: bool,
    ) -> bool {
        if !self.enabled || self.at.is_some() || descending {
            return false;
        }
        match previous {
            [Some(e1), Some(e2)]
                if (energy - e2).abs() < REPEAT
                    && ((energy - e1).abs() > MOVING || gmax > GRADIENT) =>
            {
                self.run += 1
            }
            _ => self.run = 0,
        }
        if self.run >= RUN {
            self.at = Some(iter);
        }
        self.at.is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Six alternating iterations in a row fire it once; a converging stretch resets the count.
    #[test]
    fn six_repeats_in_a_row_fire_once() {
        let mut t = TwoCycle::new(true);
        let e = |k: usize| {
            if k.is_multiple_of(2) {
                -1.0
            } else {
                -1.0 + 1e-3
            }
        };
        let mut hist: Vec<f64> = vec![e(0), e(1)];
        let prev = |h: &[f64]| [h.last().copied(), h.len().checked_sub(2).map(|i| h[i])];
        for k in 2..7 {
            assert!(
                !t.observe(k, e(k), prev(&hist), 1e-2, false),
                "iteration {k}"
            );
            hist.push(e(k));
        }
        assert_eq!(t.run, 5);
        assert!(t.observe(7, e(7), prev(&hist), 1e-2, false));
        assert_eq!(t.at, Some(7));
        assert!(
            !t.observe(8, e(8), prev(&hist), 1e-2, false),
            "once per SCF"
        );
        // a step that breaks the pattern resets the count
        let mut u = TwoCycle::new(true);
        u.observe(2, -1.0, [Some(-0.9), Some(-1.0)], 1e-2, false);
        assert_eq!(u.run, 1);
        u.observe(3, -0.5, [Some(-1.0), Some(-0.9)], 1e-2, false);
        assert_eq!(u.run, 0);
        // converging (both differences small, small gradient): no cycle
        u.observe(4, -1.0, [Some(-1.0 + 1e-8), Some(-1.0)], 1e-6, false);
        assert_eq!(u.run, 0);
        // off while descending or disabled: the count stays
        let mut d = TwoCycle::new(true);
        d.run = 3;
        assert!(!d.observe(5, -1.0, [Some(-0.9), Some(-1.0)], 1e-2, true));
        assert_eq!(d.run, 3);
        assert!(!TwoCycle::new(false).observe(9, -1.0, [Some(-0.9), Some(-1.0)], 1.0, false));
    }
}
