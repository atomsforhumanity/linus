//! The soft-mode plateau rule ([`crate::ScfAlgorithm::plateau_window`]): an SCF that sits on a
//! plateau of tiny gradient and stationary energy when its escalation comes due, or at `max_iter`, is
//! accepted, with the warning `loosely_converged` (the driver's), rather than escalated: the escalation's
//! quasi-Newton steps along the soft mode leave the plateau. Turning a hole among degenerate orbitals is a symmetry of
//! the exact problem that a PCM surface or an XC grid breaks slightly: a tiny gradient along a mode
//! of tiny curvature that DIIS and the diagonally preconditioned SOSCF do not remove (the F atom in
//! C-PCM). Under accuracy targets ([`crate::ScfConfig::plateau_limit`]) a plateau is accepted only
//! when its final orbital gradient's error estimates meet them. A plateau of an open shell is a
//! converged point for the stability trigger (`crate::run`): a UHF/UKS SCF can also crawl along a
//! negative mode at that pace (FeH, B3LYP/def2-SVP: 1e-12 Eh per iteration down a -1.1e-3 mode), so
//! the point is analysed before it is accepted, and a negative mode refuses it.

use crate::{Plateau, PlateauRefusal, ScfConfig, ScfIteration};
use linus_base::convergence::GradNorm;

/// The rule's settings: the window, the convergence thresholds it relaxes and the accuracy
/// targets' limit on the final orbital gradient.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PlateauRule {
    window: usize,
    e_tol: f64,
    g_tol: f64,
    g_norm: GradNorm,
    limit: Option<f64>,
}

/// What the rule says at `max_iter`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Verdict {
    /// No plateau: the SCF stops unconverged.
    None,
    /// A plateau the rule accepts.
    Accept(Plateau),
    /// A plateau the accuracy targets refuse.
    Refuse(PlateauRefusal),
}

impl PlateauRule {
    pub fn new(opts: &ScfConfig) -> Self {
        PlateauRule {
            window: opts.algo.plateau_window,
            e_tol: opts.e_tol,
            g_tol: opts.g_tol,
            g_norm: opts.g_norm,
            limit: opts.plateau_limit,
        }
    }

    /// The bound on the orbital gradient (in the convergence norm) over the window:
    /// min(1000 g_tol, 1e-6).
    fn bound(&self) -> f64 {
        (1e3 * self.g_tol).min(1e-6)
    }

    /// The rule's verdict on an SCF that reached `max_iter` with the iterations `history`
    /// (every one, across the escalation's restart): a plateau when over the last `window`
    /// iterations the energy spread stays below 100 e_tol and the orbital gradient (in `g_norm`)
    /// below min(1000 g_tol, 1e-6); refused when the last iteration's orbital gradient in PySCF's
    /// norm is above the accuracy targets' limit.
    pub fn verdict(&self, history: &[ScfIteration]) -> Verdict {
        let w = self.window;
        if w == 0 || history.len() < w {
            return Verdict::None;
        }
        let tail = &history[history.len() - w..];
        let g_of = |r: &ScfIteration| match self.g_norm {
            GradNorm::Max => r.gmax,
            GradNorm::Pyscf => r.gnorm_pyscf,
        };
        let e_lo = tail.iter().map(|r| r.energy).fold(f64::INFINITY, f64::min);
        let e_hi = tail
            .iter()
            .map(|r| r.energy)
            .fold(f64::NEG_INFINITY, f64::max);
        let g_hi = tail.iter().map(g_of).fold(0.0, linus_base::linalg::max_nan);
        if !(e_hi - e_lo < 100.0 * self.e_tol && g_hi < self.bound()) {
            return Verdict::None;
        }
        let last = &tail[w - 1];
        let plateau = Plateau {
            window: w,
            energy_spread: e_hi - e_lo,
            residual: g_of(last),
            max_residual: g_hi,
        };
        match self.limit {
            Some(limit) if last.gnorm_pyscf.is_nan() || last.gnorm_pyscf > limit => {
                Verdict::Refuse(PlateauRefusal {
                    plateau,
                    reason: "accuracy",
                    gnorm_pyscf: last.gnorm_pyscf,
                    limit: Some(limit),
                })
            }
            _ => Verdict::Accept(plateau),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn it(energy: f64, gmax: f64) -> ScfIteration {
        ScfIteration {
            iter: 0,
            energy,
            de: 0.0,
            gmax,
            grms: 0.0,
            gnorm_pyscf: gmax,
            ddmax: 0.0,
            ddrms: 0.0,
            fock: "full",
            approx: 0,
            step: "diis",
            t_jk: 0.0,
            t_xc: 0.0,
            t_step: 0.0,
        }
    }

    #[test]
    fn a_flat_window_is_accepted_a_moving_one_is_not() {
        let mut o = ScfConfig::default();
        o.algo.plateau_window = 3;
        let r = PlateauRule::new(&o);
        let flat = [it(-1.0, 1e-8), it(-1.0 + 1e-9, 2e-8), it(-1.0, 1.5e-8)];
        let Verdict::Accept(p) = r.verdict(&flat) else {
            panic!("a plateau")
        };
        assert_eq!((p.window, p.residual, p.max_residual), (3, 1.5e-8, 2e-8));
        assert_eq!(
            r.verdict(&flat[1..]),
            Verdict::None,
            "shorter than the window"
        );
        let moving = [it(-1.0, 1e-8), it(-1.0 + 1e-7, 1e-8), it(-1.0, 1e-8)];
        assert_eq!(
            r.verdict(&moving),
            Verdict::None,
            "energy spread 1e-7 >= 100 e_tol"
        );
        let steep = [it(-1.0, 1e-8), it(-1.0, 2e-6), it(-1.0, 1e-8)];
        assert_eq!(r.verdict(&steep), Verdict::None, "gradient above 1e-6");
        // accuracy targets: refused above their limit on the final gradient, accepted below
        o.plateau_limit = Some(1e-8);
        match PlateauRule::new(&o).verdict(&flat) {
            Verdict::Refuse(f) => assert_eq!((f.reason, f.gnorm_pyscf), ("accuracy", 1.5e-8)),
            v => panic!("{v:?}"),
        }
        o.plateau_limit = Some(2e-8);
        assert!(matches!(
            PlateauRule::new(&o).verdict(&flat),
            Verdict::Accept(_)
        ));
        o.algo.plateau_window = 0;
        assert_eq!(PlateauRule::new(&o).verdict(&flat), Verdict::None, "off");
    }
}
