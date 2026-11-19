//! Accuracy targets (`accuracy`, qc*'s shared keyword; decisions 22 and 25b): the error from
//! convergence a caller can tolerate in the energy (Eh), the largest gradient component (Eh/bohr)
//! and the largest Hessian element (Eh/bohr^2), turned into linus's own thresholds. Approximations
//! (basis set, grids, density fitting, precision) are not covered: only the SCF's and the
//! response's convergence.
//!
//! The errors follow the SCF's orbital gradient and the CPHF/CPKS residual r. Measured against
//! references converged to max |F_ai| 1e-10 and r 1e-10 (HF water and NH3+, B3LYP water, CH3 UKS,
//! triplet p-benzyne UKS PBE0, caffeine B3LYP and its cation UKS; def2-SVP; sweeps of scf.g_tol
//! 1e-3..1e-8 and of response.tol 1e-1..1e-7), with g the max |F_ai| the SCF tests and gn PySCF's
//! norm of the orbital gradient (`gnorm_pyscf` in the SCF's history):
//!
//! | error | in g (the thresholds) | in gn (the estimates) |
//! |---|---|---|
//! | energy (variational) | dE <= 153 g^2 | dE <= 3.4 gn^2 |
//! | largest gradient component | dG <= 10.5 g | dG <= 0.90 gn |
//! | largest Hessian element, from the SCF | dH <= 11.2 g | dH <= 0.57 gn |
//! | largest Hessian element, from the response | dH <= 41 r^2 (r <= 1e-3), 120 r^2 (r <= 1e-2) | |
//!
//! The factors in g grow with the molecule (3-5 atoms: dE <= 15 g^2, dG <= 0.6 g; caffeine the
//! largest), because the orbital gradient has more elements near its largest; those in gn do
//! not. The SCF tests g, so the thresholds take the bounds in g with a factor 3 of margin:
//! energy E -> e_tol E, g_tol sqrt(E / 600); gradient G -> g_tol G / 36; Hessian H -> g_tol
//! H / 36, response.tol sqrt(H / 360), and below H = 1e-7 a response.tol under 1e-6, which
//! keeps the response exchange in double precision (single precision's noise was 2.3e-8 for
//! caffeine). A request's thresholds are the tightest of its targets that apply to it (an
//! energy's to every request, a gradient's to gradients and Hessians, a Hessian's to Hessians);
//! given `scf.e_tol`, `scf.g_tol`, `scf.convergence` and `response.tol` are floors a target may
//! tighten, never loosen. A target replaces the defaults, so a loose one runs a cheaper SCF.
//!
//! The run records `accuracy` (qcdata's shared record key): the `target`, the `thresholds` used,
//! the `estimate` of each targeted error from the final orbital gradient (the bounds in gn,
//! which hold for larger molecules too) and response residual, and `met`, whether every
//! estimate is within its target. Without targets nothing changes.

use crate::driver::options::Purpose;
use crate::driver::report::CalcType;
use serde::Serialize;
use std::collections::BTreeMap;

/// dE <= C_ENERGY g^2 (g: max |F_ai|; measured 153).
pub(crate) const C_ENERGY: f64 = 200.0;
/// dG <= C_GRADIENT g (measured 10.5).
pub(crate) const C_GRADIENT: f64 = 12.0;
/// dH <= C_HESSIAN g, from the SCF (measured 11.2).
pub(crate) const C_HESSIAN: f64 = 12.0;
/// dH <= C_RESPONSE r^2 (r: the CPHF/CPKS max-abs residual; measured 120 for r <= 1e-2).
pub(crate) const C_RESPONSE: f64 = 120.0;
/// The margin of the thresholds over the measured bounds.
pub(crate) const MARGIN: f64 = 3.0;
/// Below this Hessian target the response exchange runs in double precision: response.tol is
/// kept under `linus_jk::df::response::SINGLE_MIN_TOL`.
pub(crate) const H_DOUBLE: f64 = 1e-7;
/// The estimates' factors in PySCF's norm gn: dE <= E_GN gn^2, dG <= G_GN gn, dH <= H_GN gn
/// (measured 3.4, 0.90, 0.57).
pub(crate) const E_GN: f64 = 4.0;
pub(crate) const G_GN: f64 = 1.0;
pub(crate) const H_GN: f64 = 0.6;

/// The targets of a calculation (absent: none for that quantity).
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize)]
pub struct Targets {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub energy: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gradient: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hessian: Option<f64>,
}

impl Targets {
    /// Whether any target is set.
    pub fn any(&self) -> bool {
        self.energy.is_some() || self.gradient.is_some() || self.hessian.is_some()
    }

    /// The targets that apply to an SCF for `purpose` (the gradient's to gradients and
    /// Hessians, the Hessian's to Hessians).
    pub fn for_purpose(&self, purpose: Purpose) -> Targets {
        Targets {
            energy: self.energy,
            gradient: self.gradient.filter(|_| purpose != Purpose::Energy),
            hessian: self.hessian.filter(|_| purpose == Purpose::Hessian),
        }
    }

    /// The SCF thresholds (e_tol, g_tol) that meet the targets, the tightest of each (None: no
    /// target applies).
    pub fn scf(&self) -> Option<(f64, f64)> {
        let mut e_tol = None::<f64>;
        let mut g_tol = None::<f64>;
        let mut tighten = |e: Option<f64>, g: f64| {
            g_tol = Some(g_tol.map_or(g, |x| x.min(g)));
            if let Some(e) = e {
                e_tol = Some(e_tol.map_or(e, |x| x.min(e)));
            }
        };
        if let Some(e) = self.energy {
            tighten(Some(e), (e / (MARGIN * C_ENERGY)).sqrt());
        }
        if let Some(g) = self.gradient {
            tighten(None, g / (MARGIN * C_GRADIENT));
        }
        if let Some(h) = self.hessian {
            tighten(None, h / (MARGIN * C_HESSIAN));
        }
        // without an energy target, the energy test does not bind beyond the orbital gradient's
        let g = g_tol?;
        Some((e_tol.unwrap_or(C_ENERGY * g * g), g))
    }

    /// The largest final orbital gradient in PySCF's norm whose estimates (as in `record`)
    /// meet the targets that apply to an SCF: the plateau rule's limit (None: no targets). The
    /// Hessian's is the SCF's part of its estimate.
    pub fn plateau_limit(&self) -> Option<f64> {
        [
            self.energy.map(|e| (e / E_GN).sqrt()),
            self.gradient.map(|g| g / G_GN),
            self.hessian.map(|h| h / H_GN),
        ]
        .into_iter()
        .flatten()
        .reduce(f64::min)
    }

    /// The Hessian's CPHF/CPKS tolerance that meets its target (None: no Hessian target).
    pub fn response_tol(&self) -> Option<f64> {
        self.hessian.map(|h| {
            let r = (h / (MARGIN * C_RESPONSE)).sqrt();
            if h < H_DOUBLE {
                r.min(0.5 * linus_jk::df::response::SINGLE_MIN_TOL)
            } else {
                r
            }
        })
    }

    /// The record of a run of `calctype` with these targets: the thresholds used (`scf.e_tol`,
    /// `scf.g_tol`, `response.tol` when it solved one), the estimates from the final orbital
    /// gradient `gn` (PySCF's norm) and the response's (tolerance, residual), and whether they
    /// are within the targets.
    pub fn record(
        &self,
        calctype: CalcType,
        e_tol: f64,
        g_tol: f64,
        gn: f64,
        response: Option<(f64, f64)>,
    ) -> AccuracyRecord {
        let mut thresholds = BTreeMap::from([("scf.e_tol", e_tol), ("scf.g_tol", g_tol)]);
        let mut estimate = BTreeMap::new();
        let mut met = true;
        let mut check = |name: &'static str, target: Option<f64>, est: f64| {
            if let Some(t) = target {
                estimate.insert(name, est);
                met &= est <= t;
            }
        };
        check("energy", self.energy, E_GN * gn * gn);
        if calctype != CalcType::Energy {
            check("gradient", self.gradient, G_GN * gn);
        }
        if calctype == CalcType::Hessian {
            let (tol, residual) = response.unwrap_or((f64::NAN, 0.0));
            if response.is_some() {
                thresholds.insert("response.tol", tol);
            }
            check(
                "hessian",
                self.hessian,
                H_GN * gn + C_RESPONSE * residual * residual,
            );
        }
        AccuracyRecord {
            target: *self,
            thresholds,
            estimate,
            met,
        }
    }
}

/// `resolved.record.accuracy`: the targets, linus's thresholds for them, the error estimates and
/// whether each estimate is within its target.
#[derive(Clone, Debug, Serialize)]
pub struct AccuracyRecord {
    /// The targets of the request (its own, else the input's `accuracy`).
    pub target: Targets,
    /// The thresholds used, by their setting paths.
    pub thresholds: BTreeMap<&'static str, f64>,
    /// The estimated error of each targeted quantity the run computed.
    pub estimate: BTreeMap<&'static str, f64>,
    /// Every estimate is within its target.
    pub met: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn thresholds_of_targets() {
        let t = Targets {
            energy: Some(1e-6),
            gradient: Some(3e-5),
            hessian: Some(1e-4),
        };
        let (e, g) = t.for_purpose(Purpose::Energy).scf().unwrap();
        assert_eq!(e, 1e-6);
        assert!((g - (1e-6f64 / 600.0).sqrt()).abs() < 1e-18);
        // the gradient's target is the tighter one for a gradient
        let (e, g) = t.for_purpose(Purpose::Gradient).scf().unwrap();
        assert!(e == 1e-6 && (g - 3e-5 / 36.0).abs() < 1e-20);
        assert!(t.for_purpose(Purpose::Gradient).response_tol().is_none());
        // the plateau rule's limit on the final gn: the tightest estimate that meets a target
        // (energy sqrt(E/4) = 5e-4, gradient G = 3e-5, the Hessian's SCF part H/0.6)
        assert_eq!(t.for_purpose(Purpose::Energy).plateau_limit(), Some(5e-4));
        assert_eq!(t.for_purpose(Purpose::Gradient).plateau_limit(), Some(3e-5));
        assert_eq!(Targets::default().plateau_limit(), None);
        // a gradient target alone: the energy test does not bind beyond the gradient's
        let g_only = Targets {
            gradient: Some(3.6e-4),
            ..Default::default()
        };
        let (e, g) = g_only.scf().unwrap();
        assert!((g - 1e-5).abs() < 1e-19 && (e - 200.0 * g * g).abs() < 1e-22);
        assert!(g_only.for_purpose(Purpose::Energy).scf().is_none());
        assert!((t.response_tol().unwrap() - (1e-4f64 / 360.0).sqrt()).abs() < 1e-18);
        // a tight Hessian target keeps the response exchange in double precision
        let tight = Targets {
            hessian: Some(1e-8),
            ..Default::default()
        };
        assert!(tight.response_tol().unwrap() < linus_jk::df::response::SINGLE_MIN_TOL);
    }

    #[test]
    fn record_says_whether_estimates_meet_targets() {
        let t = Targets {
            gradient: Some(1e-5),
            ..Default::default()
        };
        let r = t.record(CalcType::Gradient, 1e-9, 3e-7, 2e-6, None);
        assert!(r.met && r.estimate["gradient"] == 2e-6 && !r.estimate.contains_key("energy"));
        let r = t.record(CalcType::Gradient, 1e-9, 3e-7, 2e-5, None);
        assert!(!r.met);
    }
}
