//! SCF settings: the guess, convergence thresholds and their presets ([`Convergence`]), and
//! the algorithm settings ([`ScfAlgorithm`]: DIIS, damping, level shift, ADIIS, incremental
//! Fock builds, SOSCF, grid switching, stability analysis) that make up [`ScfConfig`].

use linus_base::convergence::{Convergence, GradNorm};
use linus_base::progress::Progress;
use serde::{Deserialize, Serialize};

/// The initial guess of an SCF (`scf.guess`: "sad" or "core").
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Guess {
    /// Orbitals of the core Hamiltonian.
    Core,
    /// Superposition of atomic densities (spin-averaged atomic HF in the target basis)
    Sad,
}

/// Which matrix leaves the DIIS subspace when it is full.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DiisDrop {
    Oldest,
    LargestError,
}

impl DiisDrop {
    /// The entry that leaves a subspace of `capacity` entries with the squared error norms
    /// `norms` (in order) for a new one: None while it has room. LargestError takes the last of
    /// equal largest norms (as `Iterator::max_by`). Both SCF loops ask it (the device loop for its
    /// subspace on the device).
    pub fn victim(self, norms: &[f64], capacity: usize) -> Option<usize> {
        if norms.len() < capacity {
            return None;
        }
        Some(match self {
            DiisDrop::Oldest => 0,
            DiisDrop::LargestError => (0..norms.len())
                .max_by(|&a, &b| norms[a].total_cmp(&norms[b]))
                .expect("a full subspace"),
        })
    }
}

#[derive(Clone, Debug)]
pub struct ScfConfig {
    /// None: restricted for closed shells, unrestricted otherwise.
    pub restricted: Option<bool>,
    /// The SCF's whole budget of iterations (Fock builds of the loop), the escalation's included
    /// ([`ScfAlgorithm::escalate`]).
    pub max_iter: usize,
    pub e_tol: f64,
    /// Bound on the orbital gradient in the norm `g_norm`.
    pub g_tol: f64,
    pub g_norm: GradNorm,
    pub guess: Guess,
    /// Where the progress lines go (`output.verbose`; off by default).
    pub progress: Progress,
    /// The algorithm settings (part of the user's `scf` settings, `CalcOptions::scf`).
    pub algo: ScfAlgorithm,
    /// The zero-mode check of a Hessian SCF: every converged UHF/UKS point is probed with at least
    /// this many one-sided Hessian-vector products per irrep (one Fock build each; 0: off); a
    /// negative mode takes the probe's path (the full analysis, the step), and the search at the
    /// accepted point is the record ([`super::ScfResult::mode_check`]). Set by the driver for
    /// open-shell Hessians (`scf.stability_hessian_check`).
    pub mode_check: usize,
    /// The gap trigger of the default probes at a continued SCF (None: not armed; set by the
    /// driver at a later geometry of a lineage).
    pub probe_trigger: Option<super::stability::ProbeTrigger>,
    /// Under accuracy targets: the largest final orbital gradient, in PySCF's norm, whose error
    /// estimates meet them (set by the driver); the plateau rule accepts no SCF above it. None:
    /// no targets.
    pub plateau_limit: Option<f64>,
}

/// Hessian-vector products of the default stability probe (`scf.stability_probe` unset): on the
/// open-shell set of the ROBUST sweep (500 molecules, B3LYP/def2-SVP) 1.13x the Fock builds and
/// 1.08x the wall time of the SCF alone (medians), catching 10 of the 13 saddles the full analysis
/// finds (docs/internal/robustness.md, SCF proposal item 4).
pub const STABILITY_PROBE_AUTO: usize = 2;

/// Hessian-vector products of the closed-shell probe of KS-DFT (`scf.stability_closed_probe` unset,
/// decision D36): 2 one-sided products of the UKS orbital Hessian at a converged RKS point whose
/// HOMO-LUMO gap lies below [`CLOSED_SHELL_GAP`] (bench/results/2026-11-11-scfrob).
pub const CLOSED_SHELL_PROBE_AUTO: usize = 2;

/// The HOMO-LUMO gap (Eh) below which the closed-shell probe runs (`scf.stability_closed_gap`
/// default). Of the 886 closed-shell tier-1 molecules of the ROBUST set with B3LYP/def2-SVP, 30
/// have a lower broken-symmetry UKS solution; 28 of them lie below 0.15 Eh, as do 16% of the
/// molecules (bench/results/2026-11-11-scfrob, section 3).
pub const CLOSED_SHELL_GAP: f64 = 0.15;

/// SCF algorithm settings: DIIS, damping, level shift, ADIIS, incremental Fock builds, SOSCF,
/// grid switching and stability analysis (`CalcOptions::scf`, and part of [`ScfConfig`]).
/// `incremental` also switches the COSX builder's own incremental exchange builds (jk = rij-snk
/// with COSX grids).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ScfAlgorithm {
    pub diis_space: usize,
    /// First iteration whose Fock matrix enters DIIS. With a guess density that does not come
    /// from orbitals (SAD), iteration 1's Fock matrix is never used for DIIS or SOSCF.
    pub diis_start: usize,
    pub diis_drop: DiisDrop,
    /// With [`DiisDrop::LargestError`]: once a new entry's error norm exceeds `diis_unpin` times
    /// the smallest one in the subspace, the subspace drops its oldest entry from then on (0:
    /// never). LargestError keeps the best entries while the errors fall, and then equals
    /// Oldest; but when the SCF leaves their region (an extrapolated step out of a stall) the
    /// kept entries pin the extrapolation there, and DIIS reaches a fixed point with a nonzero
    /// orbital gradient (HBr+, HI+ UKS at max |F_ai| < 1e-9: frozen at 7e-7 to 1e-6 with
    /// |dD| -> 1e-9; bench/results/2026-11-07-openshell-scf).
    pub diis_unpin: f64,
    /// Level shift (Eh) of the virtual orbitals while max |F_ai| > level_shift_until. Default
    /// 0.25 Eh until 1e-2 (start-up only): at the tight preset it saved 0-4 Fock builds in 10
    /// cases (la18, u1an39, ts61, caffeine; HF/PBE/B3LYP; RI and exact; 154 -> 144 builds in
    /// total) and never cost one; shifting until 1e-3 saved as many but cost one in 2 cases.
    pub level_shift: f64,
    pub level_shift_until: f64,
    /// Fock damping F <- (1 - damping) F + damping F_previous while max |F_ai| > damping_until
    /// (DIIS is held back while damping).
    pub damping: f64,
    pub damping_until: f64,
    /// Use ADIIS while max |\[F,D\]| > adiis_switch (then DIIS).
    pub adiis: bool,
    pub adiis_switch: f64,
    /// Incremental J/K builds for builders that support them.
    pub incremental: bool,
    /// Full J/K rebuild after this many incremental builds.
    pub incremental_reset: usize,
    /// Quasi-Newton (L-BFGS, diagonal orbital-energy Hessian) orbital rotations instead of
    /// DIIS once max |F_ai| < soscf_start.
    pub soscf: bool,
    pub soscf_start: f64,
    /// J/K builders with a sequence of approximations
    /// ([`JkBuilder::next_approximation`](linus_base::terms::JkBuilder::next_approximation);
    /// COSX: its SCF grids) move to the next once max |F_ai| < grid_switch (and always before the SCF
    /// may converge). K is then rebuilt at the same density and, with `grid_switch_shift`, the
    /// Fock matrices of the DIIS subspace are shifted by the change of the exchange matrix (their
    /// errors recomputed), so that DIIS continues on the new grid instead of mixing two
    /// operators. Default 1e-3, about the size of the GridX1 -> GridX2 change of max |F_ai|
    /// (3-7e-4): ts61 B3LYP/def2-TZVP needed 13 builds at the energy thresholds with 1e-3 and
    /// 3e-3 (8-10% less COSX work with 1e-3) and 14 with 3e-4; at the gradient thresholds 15
    /// (1e-3) and 14 (3e-3) builds, 8% less COSX work with 1e-3.
    pub grid_switch: f64,
    pub grid_switch_shift: bool,
    /// UHF/UKS saddle-point check (Eh; 0: off). DIIS converges to any stationary point of the
    /// energy. When the converged energy lies more than `saddle_check` above the lowest energy
    /// the SCF visited on the same J/K approximation, the converged point is not a minimum
    /// along the path: the lowest eigenpair of the orbital Hessian is found
    /// (`stability::lowest_hessian_mode`: Davidson, two Fock builds per Hessian-vector product)
    /// and, if it is negative, the SCF restarts (new DIIS subspace) from the orbitals rotated along it.
    /// In converging runs the converged energy is the lowest visited to within 6e-12 Eh (2e-10
    /// for the orientation of a pi hole on a DFT grid); the lactide radical cation (wB97X-D3BJ,
    /// UHF) converged 1.2e-8 and 5.6e-8 Eh above it, on a saddle point 2.3 mEh (wB97X-D3BJ)
    /// above the minimum.
    pub saddle_check: f64,
    /// Analyse the stability of every converged UHF/UKS determinant (not only on the
    /// `saddle_check` trigger).
    pub stability: bool,
    /// At most this many stability analyses per SCF.
    pub stability_max: usize,
    /// Cheap stability probe of converged UHF/UKS determinants (Hessian-vector products; 0:
    /// off): that many Davidson steps for the lowest orbital-Hessian mode, and the full analysis
    /// (`stability`) with descent only when they find a negative one (the Ritz value is an
    /// upper bound, so a negative probe is a negative mode; a positive one can miss a mode the
    /// full analysis would find). At least 2; each product costs one Fock build (a one-sided
    /// difference from the converged gradient). On a minimum the SCF's result is unchanged.
    /// None (the default, `auto`): [`STABILITY_PROBE_AUTO`] products at the session's first
    /// geometry (single points, the first SCF of Hessians and optimizations), off at later
    /// geometries; a number applies at every geometry. The SCF reads the resolved value
    /// (`unwrap_or(0)`).
    pub stability_probe: Option<usize>,
    /// Closed-shell stability probe (decision D36; Hessian-vector products, 0: off): at a
    /// converged RHF/RKS point whose HOMO-LUMO gap lies below `closed_shell_gap`, that many
    /// one-sided products of the UHF/UKS orbital Hessian (UKS-type Fock builds: two spin densities,
    /// a polarized XC evaluation) from the lowest-gap pair with alpha and beta rotating in opposite
    /// senses (the external direction towards a broken-symmetry solution, phase-fixed as D29). A
    /// negative mode is recorded (`ScfResult::closed_shell_probe`) and the driver warns; the SCF
    /// never switches and its result is unchanged. At least 2. None (the default, `auto`):
    /// [`CLOSED_SHELL_PROBE_AUTO`] for KS-DFT at the session's first geometry, 0 for Hartree-Fock
    /// (whose triplet instabilities are ubiquitous and rarely matter for an RHF reference) and at
    /// later geometries; a number applies at every geometry and to HF. The SCF reads the resolved
    /// value (`unwrap_or(0)`).
    pub closed_shell_probe: Option<usize>,
    /// The HOMO-LUMO gap (Eh) below which the closed-shell probe runs ([`CLOSED_SHELL_GAP`]).
    pub closed_shell_gap: f64,
    /// React to a two-cycle: the energy repeating every other iteration (|E_k - E_{k-2}| < 1e-9 Eh)
    /// while the SCF is not converging (|E_k - E_{k-1}| > 1e-7 Eh or max |F_ai| > 1e-4), six
    /// iterations in a row. Occupations flip between two determinants (the CN radical's two
    /// pi-hole configurations under DIIS) and nothing else ends it. From then on quasi-Newton
    /// (SOSCF) steps from the current orbitals replace DIIS, once per SCF; the iteration is
    /// reported in `resolved.record.scf.two_cycle_at`. The trigger fired on none of the 30,192 converged
    /// SCF histories of the robustness sweep (docs/internal/robustness.md), so an SCF that
    /// converges without it takes the same path.
    pub two_cycle: bool,
    /// Soft-mode plateau rule (iterations; 0: off). An SCF unconverged when its escalation comes
    /// due, or at `max_iter`, is accepted as converged, with the warning `loosely_converged`, when over
    /// its last `plateau_window` iterations the energy spread stays below 100 e_tol and the
    /// orbital gradient (in `g_norm`) below min(1000 g_tol, 1e-6); under accuracy targets only
    /// when its final orbital gradient's error estimates meet them ([`ScfConfig::plateau_limit`]).
    /// It ends on a full build: one more iteration when the escalation came due, the budget's
    /// last (always full) at `max_iter`. Turning a hole among degenerate orbitals (open-shell atoms, linear radicals) is a
    /// symmetry of the exact problem that a Lebedev PCM surface or an XC grid breaks slightly:
    /// a tiny gradient along a mode of tiny curvature that DIIS and the diagonally
    /// preconditioned SOSCF do not remove (F atom, HF/def2-SVP, C-PCM: max |F_ai| 1.8e-8 for
    /// hundreds of iterations at an energy stationary to 1e-11 Eh). Only SCFs that would fail
    /// change; the residual is reported in `resolved.record.scf.plateau`.
    pub plateau_window: usize,
    /// Escalation within `max_iter`: an SCF whose DIIS stagnates (the lowest energy not improved
    /// for `stagnation_window` iterations), or that has spent 2/3 of `max_iter` unconverged,
    /// restarts once from the orbitals of the lowest-energy iteration it visited (on the final J/K
    /// approximation), with a new DIIS subspace, no level shift and no damping, and takes
    /// quasi-Newton (SOSCF) steps, which descend in energy, for the rest of the budget (the
    /// plateau rule decides first, and at the end). It needs at least 2 iterations left
    /// (`control::MIN_ESCALATION`). Reported in `resolved.record.scf.escalation`, with the
    /// reason when it did not run in an SCF that used its budget.
    pub escalate: bool,
    /// Iterations without a new lowest energy after which the escalation starts (0: only at 2/3
    /// of `max_iter`). Default 64: of the 69,615 converged SCF histories of the robustness sweep
    /// and its hard set, none converged by iteration 64 and 62 of the 67,049 converged by
    /// iteration 85 would have reached it first (a smaller window escalates more SCFs that
    /// converge on their own: 676 at 32); of the 9,003 that failed, 3,506 reach it by iteration 85.
    pub stagnation_window: usize,
    /// Linear-dependence threshold of the canonical orthogonalization, absolute: eigenvectors of
    /// the overlap matrix with eigenvalues at or below `lindep` are dropped
    /// ([`ScfResult::dropped_functions`](crate::ScfResult::dropped_functions)); fewer
    /// orbitals left than occupied ones is an input error.
    pub lindep: f64,
}

impl ScfAlgorithm {
    /// The default of [`ScfAlgorithm::stagnation_window`].
    pub const STAGNATION_WINDOW: usize = 64;
}

impl Default for ScfAlgorithm {
    fn default() -> Self {
        ScfAlgorithm {
            diis_space: 10,
            diis_start: 1,
            diis_drop: DiisDrop::LargestError,
            diis_unpin: 10.0,
            level_shift: 0.25,
            level_shift_until: 1e-2,
            damping: 0.0,
            damping_until: 0.0,
            adiis: false,
            adiis_switch: 1e-2,
            incremental: true,
            incremental_reset: 20,
            soscf: false,
            soscf_start: 3.3e-3,
            grid_switch: 1e-3,
            grid_switch_shift: true,
            saddle_check: 1e-9,
            stability: false,
            stability_max: 2,
            stability_probe: None,
            closed_shell_probe: None,
            closed_shell_gap: CLOSED_SHELL_GAP,
            two_cycle: true,
            plateau_window: 50,
            escalate: true,
            stagnation_window: ScfAlgorithm::STAGNATION_WINDOW,
            lindep: 1e-8,
        }
    }
}

impl Default for ScfConfig {
    fn default() -> Self {
        ScfConfig {
            restricted: None,
            max_iter: 288,
            e_tol: 1e-10,
            g_tol: 1e-8,
            g_norm: GradNorm::Max,
            guess: Guess::Core,
            progress: Progress::off(),
            algo: ScfAlgorithm::default(),
            mode_check: 0,
            probe_trigger: None,
            plateau_limit: None,
        }
    }
}

impl ScfConfig {
    pub fn set_convergence(&mut self, c: Convergence) {
        self.e_tol = c.e_tol;
        self.g_tol = c.g_tol;
        self.g_norm = c.g_norm;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_drop_rules_victims() {
        assert_eq!(DiisDrop::LargestError.victim(&[4.0, 9.0], 3), None, "room");
        assert_eq!(DiisDrop::LargestError.victim(&[4.0, 9.0, 1.0], 3), Some(1));
        assert_eq!(DiisDrop::Oldest.victim(&[4.0, 9.0, 1.0], 3), Some(0));
        // ties: the last of equal largest norms
        assert_eq!(DiisDrop::LargestError.victim(&[5.0, 5.0], 2), Some(1));
    }
}
