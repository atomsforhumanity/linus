//! A DIIS blow-up ([`crate::ScfAlgorithm::blowup_window`]): the energy far above the lowest the
//! SCF visited while the orbital gradient is far above its smallest, iteration after iteration.
//! It starts the escalation (`blowup`) without waiting for the stagnation window: the lowest-energy
//! orbitals it restarts from are the same, and the iterations in between are lost.
//!
//! The thresholds come from the per-iteration histories of the robustness sweep, its hard set and
//! the SCFROB and epoch-10 runs on them (93,785 SCFs; bench/robust/pipeline/blowup_replay.py,
//! docs/internal/robustness.md, "Escalation on a blow-up"):
//! - [`BLOWUP_GRADIENT`]: of the SCFs whose energy stayed more than 1 Eh above their lowest for 4
//!   iterations, 612 that converge on their own have max |F_ai| below 5x its smallest there (the
//!   chaos of the first iterations from a guess; Trp-cage's iterations 3-5 sit 26-84 Eh up at
//!   2.8x); above 20x are the blow-ups (45 SCFs that fail or escalate; Trp-cage PBE/def2-SVP 21.6x
//!   at +11 Eh) and 15 that DIIS recovers from after 60-280 iterations (LiF at 7 Å, Cr2); 6
//!   converged and 3 failing SCFs lie between 5x and 20x. 10x is the middle of that gap.
//! - [`BLOWUP_ENERGY`]: with max |F_ai| above 10x its smallest for 4 iterations, 17 of the 80,987
//!   SCFs that converge without escalating stay more than 1 Eh above their lowest throughout (4
//!   molecules: LiF at 7 Å, Cr2 at 2.5 Å, NiCl4 2- TPSS without the level shift, a cobalt complex's
//!   vertical triplet; slow DIIS recoveries), against 54 at 0.3 Eh and 129 at 0.1 Eh.
//! - the window, 4 iterations: Trp-cage stays 10 iterations at +11 Eh before it blows up further;
//!   3 iterations add a false positive (LiF at 4 Å), 5 remove none.

/// The energy above the lowest visited (Eh) that counts towards a blow-up.
pub const BLOWUP_ENERGY: f64 = 1.0;
/// The factor over the smallest max |F_ai| visited that counts towards a blow-up.
pub const BLOWUP_GRADIENT: f64 = 10.0;

/// The blow-up detector: the smallest orbital gradient visited and the current run of iterations
/// above both thresholds.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BlowUp {
    /// Iterations in a row that make a blow-up (0: off).
    window: usize,
    /// The smallest max |F_ai| of an iteration from orbitals since the last reset (infinity: none).
    smallest: f64,
    /// Iterations in a row above both thresholds.
    run: usize,
}

impl BlowUp {
    pub fn new(window: usize) -> Self {
        BlowUp {
            window,
            smallest: f64::INFINITY,
            run: 0,
        }
    }

    /// Forget the gradients and the run (the J/K builder moved to its next approximation, as
    /// [`super::LowestEnergy::reset`]).
    pub fn reset(&mut self) {
        *self = BlowUp::new(self.window);
    }

    /// Iteration `iter` (the iteration `guess_iter`'s density was a guess and does not count),
    /// its max |F_ai| `gmax`, its energy `e_above` the lowest visited (this iteration included).
    pub fn observe(&mut self, iter: usize, guess_iter: usize, e_above: f64, gmax: f64) {
        if iter <= guess_iter {
            return;
        }
        if gmax < self.smallest {
            self.smallest = gmax;
        }
        // false for non-finite values: a NaN ends the run
        let up = e_above > BLOWUP_ENERGY && gmax > BLOWUP_GRADIENT * self.smallest;
        self.run = if up { self.run + 1 } else { 0 };
    }

    /// The last `window` iterations were all above both thresholds (never when off).
    pub fn blown_up(&self) -> bool {
        self.window > 0 && self.run >= self.window
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_sustained_rise_of_energy_and_gradient() {
        let mut b = BlowUp::new(3);
        b.observe(1, 1, 0.0, 1e-9); // the guess density's iteration: never counts
        b.observe(2, 1, 0.0, 1e-2);
        b.observe(3, 1, 0.0, 1e-3);
        // 11 Eh up at 20x: three in a row
        b.observe(4, 1, 11.0, 2e-2);
        b.observe(5, 1, 11.0, 2e-2);
        assert!(!b.blown_up());
        b.observe(6, 1, 11.0, 2e-2);
        assert!(b.blown_up());
        // one iteration back near the lowest ends the run
        b.observe(7, 1, 0.5, 2e-2);
        assert!(!b.blown_up());
        // the energy alone (the gradient 5x) or the gradient alone (0.5 Eh) is no blow-up
        let mut e = BlowUp::new(1);
        e.observe(2, 0, 0.0, 1e-3);
        e.observe(3, 0, 80.0, 5e-3);
        assert!(!e.blown_up());
        e.observe(4, 0, 0.5, 1.0);
        assert!(!e.blown_up());
        e.observe(5, 0, f64::NAN, 1.0);
        assert!(!e.blown_up());
        e.observe(6, 0, 2.0, 1.0);
        assert!(e.blown_up());
        e.reset();
        assert!(!e.blown_up() && e.smallest.is_infinite());
        // off
        let mut off = BlowUp::new(0);
        for k in 2..10 {
            off.observe(k, 1, 100.0, if k == 2 { 1e-6 } else { 1.0 });
        }
        assert!(!off.blown_up());
    }
}
