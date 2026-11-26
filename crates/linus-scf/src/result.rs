//! The SCF's records: one iteration ([`ScfIteration`]), the plateau, escalation and mode-check
//! records, and the result ([`ScfResult`]).

use super::fock::EnergyParts;
use super::stability::{ClosedShellProbe, StabilityCheck, TriggerRecord};
use super::ScfConfig;
use faer::Mat;
use linus_base::convergence::Convergence;
use linus_base::error::ScfFailure;
use linus_core::timing::Timers;
use serde::Serialize;

/// One SCF iteration: the Fock build at D_n and the step to the next orbitals.
#[derive(Clone, Debug, Serialize)]
pub struct ScfIteration {
    pub iter: usize,
    pub energy: f64,
    pub de: f64,
    /// max |F_ai| (max element of the orthonormal-basis commutator, over spins)
    pub gmax: f64,
    /// root mean square of the commutator elements
    pub grms: f64,
    /// PySCF's norm_gorb
    pub gnorm_pyscf: f64,
    /// max and RMS of D_n - D_{n-1} (spin densities; 0 in iteration 1)
    pub ddmax: f64,
    pub ddrms: f64,
    /// "full" (of the density alone) or "incremental" (J/K of D_n - D_{n-1} added to the last
    /// build's, or a term built from its history: COSX's incremental K, the PCM's predicted
    /// charges or incremental passes)
    pub fock: &'static str,
    /// Approximations of the J/K builder moved through
    /// ([`linus_base::terms::JkBuilder::next_approximation`]; COSX: 0 on the first SCF grid, 1
    /// after the switch to the next).
    pub approx: usize,
    /// How the next orbitals were made: "diis", "adiis", "soscf", "roothaan", "damped", "rebuild"
    /// (full Fock rebuild at the same density), "stability" (rotated along a negative
    /// orbital-Hessian mode, [`ScfResult::stability`]), "escalation" (the escalation starts:
    /// the next orbitals are those of the lowest-energy iteration, [`ScfResult::escalation`]),
    /// "converged", or "stopped" (the SCF stopped on its caller's request after this iteration,
    /// [`ScfResult::stopped`]).
    pub step: &'static str,
    pub t_jk: f64,
    pub t_xc: f64,
    pub t_step: f64,
}

/// An SCF accepted on a soft-mode plateau ([`super::ScfAlgorithm::plateau_window`]), which the
/// driver reports with qcdata's shared warning `loosely_converged`.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct Plateau {
    /// Iterations checked (the last ones).
    pub window: usize,
    /// Energy spread (max - min) over them (Eh).
    pub energy_spread: f64,
    /// Orbital gradient (in the convergence norm) of the last iteration and the largest over
    /// the window.
    pub residual: f64,
    pub max_residual: f64,
}

impl Plateau {
    /// The progress line of an SCF accepted on the plateau (`output.verbose`).
    pub fn report(&self, progress: &linus_base::progress::Progress) {
        progress.line(format_args!(
            "          plateau: energy within {:.1e} Eh and orbital gradient below {:.1e} over the last {} iterations; accepted",
            self.energy_spread, self.max_residual, self.window
        ));
    }
}

/// A plateau the SCF found at `max_iter` and did not accept: `reason` `accuracy` (its final
/// orbital gradient, in PySCF's norm, is above `limit`, the largest whose error estimates meet the
/// accuracy targets) or `negative_mode` (the stability analysis of its point found a negative
/// orbital-Hessian mode, which the spent budget leaves no iterations to follow).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PlateauRefusal {
    pub plateau: Plateau,
    pub reason: &'static str,
    /// The final orbital gradient in PySCF's norm, and the accuracy targets' limit on it.
    pub gnorm_pyscf: f64,
    pub limit: Option<f64>,
}

impl PlateauRefusal {
    /// The refusal as the parenthesis of a failure message.
    pub fn describe(&self) -> String {
        match (self.reason, self.limit) {
            ("accuracy", Some(limit)) => format!(
                "on a plateau over the last {} iterations, not accepted: its orbital gradient \
                 (PySCF's norm {:.1e}) is above the {limit:.1e} the accuracy targets allow",
                self.plateau.window, self.gnorm_pyscf
            ),
            _ => format!(
                "on a plateau over the last {} iterations, not accepted: the stability analysis \
                 found a negative orbital-Hessian mode at its point (scf.stability)",
                self.plateau.window
            ),
        }
    }
}

/// The escalation of an SCF ([`super::ScfAlgorithm::escalate`]): a restart from the lowest-energy
/// iteration's orbitals with quasi-Newton steps, within `max_iter`. Recorded when it ran, and when
/// the SCF used its whole budget without it (why not); absent for an SCF that converged before it
/// was due. In the record: `{"ran": true, "trigger", "at", "from_iter", "from_energy",
/// "iterations", "converged"}` or `{"ran": false, "reason"}`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Escalation {
    /// It ran.
    Ran(EscalationRun),
    /// It did not run although the SCF used its budget: `reason` `off` (`scf.escalate` false) or
    /// `max_iter` (the budget left it fewer than the two iterations it needs, e.g. max_iter 2).
    NotRun { reason: &'static str },
}

impl Escalation {
    /// The escalation that ran, if one did.
    pub fn ran(&self) -> Option<&EscalationRun> {
        match self {
            Escalation::Ran(r) => Some(r),
            Escalation::NotRun { .. } => None,
        }
    }

    /// What the escalation did in an SCF of `max_iter` iterations, as the parenthesis of a failure
    /// message.
    pub fn describe(&self, max_iter: usize) -> String {
        match self {
            Escalation::Ran(r) => format!(
                "the escalation ({}) restarted after iteration {} from the lowest-energy orbitals \
                 (iteration {}) with {} quasi-Newton iterations, scf.escalate",
                match r.trigger {
                    "stagnation" => "DIIS stagnated",
                    "blowup" => "DIIS blew up",
                    _ => "at 2/3 of scf.max_iter",
                },
                r.at,
                r.from_iter,
                r.iterations
            ),
            Escalation::NotRun { reason: "off" } => "scf.escalate is off".into(),
            Escalation::NotRun { .. } => format!(
                "no escalation: max_iter {max_iter} leaves it fewer than the 2 iterations it \
                 needs, scf.escalate"
            ),
        }
    }
}

impl Serialize for Escalation {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeMap;
        match self {
            Escalation::Ran(r) => {
                let mut m = s.serialize_map(Some(7))?;
                m.serialize_entry("ran", &true)?;
                m.serialize_entry("trigger", r.trigger)?;
                m.serialize_entry("at", &r.at)?;
                m.serialize_entry("from_iter", &r.from_iter)?;
                m.serialize_entry("from_energy", &r.from_energy)?;
                m.serialize_entry("iterations", &r.iterations)?;
                m.serialize_entry("converged", &r.converged)?;
                m.end()
            }
            Escalation::NotRun { reason } => {
                let mut m = s.serialize_map(Some(2))?;
                m.serialize_entry("ran", &false)?;
                m.serialize_entry("reason", reason)?;
                m.end()
            }
        }
    }
}

/// An escalation that ran.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EscalationRun {
    /// Why it started: `stagnation` (the lowest energy had not improved for
    /// `stagnation_window` iterations), `blowup` (DIIS blew up for `blowup_window` iterations) or
    /// `two_thirds` (2/3 of `max_iter` were spent).
    pub trigger: &'static str,
    /// The iteration after which it started (its iterations follow).
    pub at: usize,
    /// The iteration whose orbitals it restarted from (the lowest energy visited on the final
    /// J/K approximation) and that energy (Eh).
    pub from_iter: usize,
    pub from_energy: f64,
    /// Iterations it ran.
    pub iterations: usize,
    /// It converged (by the criteria or the plateau rule).
    pub converged: bool,
}

/// The lowest orbital-Hessian eigenvalue of the converged UHF/UKS solution a Hessian SCF accepted
/// ([`super::ScfConfig::mode_check`]): the search at that point, per irrep (the trigger's probe
/// with the check's one-sided products, or the full analysis that found the point a minimum; the
/// Ritz value bounds the lowest eigenvalue from above). A negative mode that a step follows never
/// reaches the record: the SCF moves on. Near zero, or negative where no step lowers the energy:
/// a degenerate open shell (a family of solutions, e.g. the π hole of a linear radical) or an
/// unstable one, where analytic second derivatives are not those of the energy surface
/// (bench/robust/repro/uhf_hessian_zero_mode).
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct ModeCheck {
    /// The Ritz value (Eh; linus's orbital-Hessian convention, d g_ai / d kappa_bj).
    pub eigenvalue: f64,
    /// Residual norm of the Ritz vector.
    pub residual: f64,
    /// One per product, and where a term keeps state (a solvent, VV10) the build that restores
    /// the SCF's last state afterwards.
    pub fock_builds: usize,
    /// Wall time of the search (s).
    pub seconds: f64,
    /// The irreps it searched and the irrep of its Ritz value.
    #[serde(flatten)]
    pub irreps: crate::stability::IrrepRecord,
}

/// Below this lowest orbital-Hessian eigenvalue (Eh) a converged open shell is treated as
/// degenerate or unstable by the zero-mode check of Hessians ([`ModeCheck`]). Chosen from the
/// open-shell Hessians of the ROBUST sweep's tier 1 (bench/results/2026-11-10-robust,
/// hessprobe): the cases whose analytic Hessians disagree with PySCF's and with differences
/// lie at 1e-8 (C2H2+, linear, π hole; HI+ with DFT), 1e-5 to 2e-4 (BH76 RKT11), 7e-4 (MeTe,
/// a degenerate e hole) and below zero (C4H4+, NO2); of the 611 molecules of 3 or more atoms
/// with a positive eigenvalue and a Hessian that agrees with PySCF's, two lie below 1.4e-2 (MeTe
/// with B3LYP and PBE, 7.4e-4 and 7.6e-4), the rest above.
pub const MODE_CHECK_THRESHOLD: f64 = 1e-3;

impl ModeCheck {
    /// The check's verdict for a molecule of `natoms` atoms: below the threshold, except for a
    /// diatomic whose energy functional is exactly axially symmetric (`isotropic`: HF without a
    /// grid or a solvent surface), which is flagged only when the mode is negative (beyond
    /// -1e-4): its only vibration, the stretch, keeps the axial symmetry and cannot couple to the
    /// zero mode of a pi hole (OH, CH, ClO HF: their Hessians agree with PySCF's). A DFT grid or
    /// a PCM surface breaks that symmetry slightly and the zero mode couples (HI+ wB97X-D3BJ:
    /// Hessian 1.2e-2 from PySCF's). A negative mode is an unstable solution in any molecule.
    pub fn flagged(&self, natoms: usize, isotropic: bool) -> bool {
        if natoms >= 3 || !isotropic {
            self.eigenvalue < MODE_CHECK_THRESHOLD
        } else {
            self.eigenvalue < -1e-4
        }
    }
}

/// A spin channel of a converged SCF whose occupied set is not the aufbau set of its final Fock
/// matrix: empty orbitals lie below occupied ones. A quasi-Newton step keeps the occupied set it
/// starts from, so the escalation (which restarts from an earlier iteration's occupied orbitals),
/// the two-cycle rule and a stability step can converge to such a stationary point (C2 singlet
/// PBE: 3 sigma_g empty 3.7e-4 Eh below the occupied pi pair; docs/internal/deggrad.md). The
/// final orbitals are those of its occupied space all the same (occupied first), so the energy,
/// orbitals and derivatives describe one determinant; a lower aufbau solution may exist.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct NonAufbau {
    /// The channel: `alpha`, `beta`, or `restricted` (a closed shell's one).
    pub spin: &'static str,
    /// The empty orbitals that aufbau would occupy and the occupied ones it would leave empty, by
    /// their 0-based positions in the channel's spectrum in ascending energy.
    pub holes: Vec<usize>,
    pub particles: Vec<usize>,
    /// The highest occupied orbital energy less the lowest empty one (Eh; positive).
    pub inversion: f64,
}

impl NonAufbau {
    /// The violation of the channel `spin` with orbital energies `eps` (its `nocc` occupied first),
    /// when the highest occupied orbital lies more than `tol` (Eh) above the lowest empty one;
    /// closer than that the two are degenerate to within the SCF's resolution.
    pub fn of(spin: &'static str, eps: &[f64], nocc: usize, tol: f64) -> Option<NonAufbau> {
        let homo = eps[..nocc]
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        let lumo = eps[nocc..].iter().copied().fold(f64::INFINITY, f64::min);
        let inversion = homo - lumo;
        if inversion.is_nan() || inversion <= tol {
            return None;
        }
        // the positions in ascending energy (a stable sort: ties keep the result's order)
        let mut order: Vec<usize> = (0..eps.len()).collect();
        order.sort_by(|&a, &b| eps[a].total_cmp(&eps[b]));
        let mut rank = vec![0; eps.len()];
        for (r, &p) in order.iter().enumerate() {
            rank[p] = r;
        }
        let mut holes: Vec<usize> = (nocc..eps.len())
            .map(|p| rank[p])
            .filter(|&r| r < nocc)
            .collect();
        let mut particles: Vec<usize> = (0..nocc).map(|p| rank[p]).filter(|&r| r >= nocc).collect();
        holes.sort_unstable();
        particles.sort_unstable();
        Some(NonAufbau {
            spin,
            holes,
            particles,
            inversion,
        })
    }
}

#[derive(Clone, Debug)]
pub struct ScfResult {
    pub energy: f64,
    pub e_nuc: f64,
    pub e_one: f64,
    pub e_coulomb: f64,
    pub e_exchange: f64,
    pub e_xc: f64,
    /// Energy of the environment term (included in `energy`): the solvent's electrostatic
    /// solvation energy, plus SMD's G_CDS (set by the solvent's SCF); 0 in the gas phase.
    pub e_solvation: f64,
    /// Empirical dispersion energy (included in `energy`; set by the driver).
    pub e_disp: f64,
    pub converged: bool,
    /// SCF iterations, one Fock build each (iteration 1 is the guess density's); the stability
    /// analyses' builds are counted separately ([`ScfResult::stability`]).
    pub iterations: usize,
    /// Stability analyses of converged UHF/UKS points (`ScfConfig::saddle_check`).
    pub stability: Vec<StabilityCheck>,
    /// Iteration at which a two-cycle was detected and SOSCF steps took over
    /// ([`super::ScfAlgorithm::two_cycle`]).
    pub two_cycle_at: Option<usize>,
    /// Converged by the soft-mode plateau rule ([`super::ScfAlgorithm::plateau_window`]).
    pub plateau: Option<Plateau>,
    /// A plateau at `max_iter` that was not accepted (the SCF did not converge).
    pub plateau_refused: Option<PlateauRefusal>,
    /// The escalation ([`super::ScfAlgorithm::escalate`]): it ran, or why not when the SCF used
    /// its whole budget.
    pub escalation: Option<Escalation>,
    /// The iteration budget the SCF ran with (`max_iter`).
    pub max_iter: usize,
    /// The zero-mode check's estimate of the lowest orbital-Hessian eigenvalue at the accepted
    /// point ([`super::ScfConfig::mode_check`]).
    pub mode_check: Option<ModeCheck>,
    /// The closed-shell probe of an RHF/RKS solution ([`super::ScfAlgorithm::closed_shell_probe`]):
    /// its result, or why it did not run; None for UHF/UKS determinants.
    pub closed_shell_probe: Option<ClosedShellProbe>,
    /// The gap trigger's decision at a continued SCF ([`super::ScfConfig::probe_trigger`]; None: not
    /// armed, or the SCF did not reach a converged point).
    pub probe_trigger: Option<TriggerRecord>,
    /// Fock builds that were incremental (the history's `fock` "incremental").
    pub incremental_builds: usize,
    pub restricted: bool,
    pub eps_a: Vec<f64>,
    pub eps_b: Vec<f64>,
    pub c_a: Mat<f64>,
    pub c_b: Mat<f64>,
    pub nalpha: usize,
    pub nbeta: usize,
    /// Number of molecular orbitals (nbf minus functions removed by canonical orthogonalization).
    pub nmo: usize,
    /// Basis functions the canonical orthogonalization dropped as linearly dependent
    /// (`scf.lindep`): nbf - nmo.
    pub dropped_functions: usize,
    /// Occupation numbers per spin (length nmo; 1 for occupied, 0 for virtual).
    pub occ_a: Vec<f64>,
    pub occ_b: Vec<f64>,
    /// The channels of a converged SCF whose occupied set is not the aufbau set of the final Fock
    /// matrix ([`NonAufbau`]); empty otherwise.
    pub non_aufbau: Vec<NonAufbau>,
    /// <S^2> of the determinant (Kohn-Sham: of the KS determinant), S_z (S_z + 1) + n_beta -
    /// tr(D_a S D_b S); restricted closed shells 0.
    pub s2: f64,
    /// Densities that produced the final Fock matrices (they agree with the returned
    /// orbitals to within the convergence threshold).
    pub d_a: Mat<f64>,
    pub d_b: Mat<f64>,
    pub f_a: Mat<f64>,
    pub f_b: Mat<f64>,
    /// Thresholds the SCF ran with.
    pub convergence: Convergence,
    pub history: Vec<ScfIteration>,
    pub timers: Timers,
}

/// What a finished SCF hands its result, on either loop: the energy and its parts, the final
/// orbitals and orbital energies (of the unextrapolated Fock matrices) per spin, the densities and
/// Fock matrices of the last build per spin (a closed shell's twice), <S^2>, and the records.
pub struct Finished {
    pub energy: f64,
    pub e_nuc: f64,
    pub parts: EnergyParts,
    pub converged: bool,
    pub restricted: bool,
    pub nocc: [usize; 2],
    pub eps: [Vec<f64>; 2],
    pub c: [Mat<f64>; 2],
    pub d: [Mat<f64>; 2],
    pub f: [Mat<f64>; 2],
    pub s2: f64,
    pub iterations: usize,
    pub incremental_builds: usize,
    pub history: Vec<ScfIteration>,
    pub stability: Vec<StabilityCheck>,
    pub two_cycle_at: Option<usize>,
    pub plateau: Option<Plateau>,
    pub plateau_refused: Option<PlateauRefusal>,
    pub escalation: Option<Escalation>,
    pub mode_check: Option<ModeCheck>,
    pub closed_shell_probe: Option<ClosedShellProbe>,
    pub probe_trigger: Option<TriggerRecord>,
    pub timers: Timers,
}

impl ScfResult {
    /// Whether the SCF stopped on its caller's request (unconverged; its last iteration's step is
    /// "stopped", on either loop).
    pub fn stopped(&self) -> bool {
        !self.converged && self.history.last().is_some_and(|h| h.step == "stopped")
    }

    /// Where an SCF that ran out of iterations stopped and the thresholds it missed (the failure
    /// a [`LinusError::NotConverged`](linus_base::error::LinusError::NotConverged) carries).
    pub fn failure(&self) -> ScfFailure {
        let last = self.history.last();
        ScfFailure {
            iterations: self.iterations,
            de: last.map_or(f64::NAN, |h| h.de.abs()),
            gmax: last.map_or(f64::NAN, |h| h.gmax),
            gnorm_pyscf: last.map_or(f64::NAN, |h| h.gnorm_pyscf),
            convergence: self.convergence,
            notes: [
                self.escalation.map(|e| e.describe(self.max_iter)),
                self.plateau_refused.map(|p| p.describe()),
            ]
            .into_iter()
            .flatten()
            .collect::<Vec<_>>()
            .join("; "),
        }
    }

    /// The result of the finished SCF `fin` with the thresholds of `opts`: the orbital count, the
    /// functions the orthogonalization dropped and the occupations follow from the final orbitals.
    pub fn finished(fin: Finished, opts: &ScfConfig) -> Self {
        let ([eps_a, eps_b], [c_a, c_b]) = (fin.eps, fin.c);
        let ([d_a, d_b], [f_a, f_b], [nalpha, nbeta]) = (fin.d, fin.f, fin.nocc);
        let (nbf, nmo) = (c_a.nrows(), eps_a.len());
        let occ = |no: usize| (0..nmo).map(|i| if i < no { 1.0 } else { 0.0 }).collect();
        let non_aufbau = if !fin.converged {
            Vec::new()
        } else if fin.restricted {
            NonAufbau::of("restricted", &eps_a, nalpha, opts.g_tol)
                .into_iter()
                .collect()
        } else {
            [("alpha", &eps_a, nalpha), ("beta", &eps_b, nbeta)]
                .into_iter()
                .filter_map(|(spin, e, n)| NonAufbau::of(spin, e, n, opts.g_tol))
                .collect()
        };
        let p = fin.parts;
        ScfResult {
            energy: fin.energy,
            e_nuc: fin.e_nuc,
            e_one: p.one,
            e_coulomb: p.coulomb,
            e_exchange: p.exchange,
            e_xc: p.xc,
            e_solvation: p.environment,
            e_disp: 0.0,
            converged: fin.converged,
            iterations: fin.iterations,
            stability: fin.stability,
            two_cycle_at: fin.two_cycle_at,
            plateau: fin.plateau,
            plateau_refused: fin.plateau_refused,
            escalation: fin.escalation,
            max_iter: opts.max_iter,
            mode_check: fin.mode_check,
            closed_shell_probe: fin.closed_shell_probe,
            probe_trigger: fin.probe_trigger,
            incremental_builds: fin.incremental_builds,
            restricted: fin.restricted,
            nmo,
            dropped_functions: nbf - nmo,
            occ_a: occ(nalpha),
            occ_b: occ(nbeta),
            non_aufbau,
            eps_a,
            eps_b,
            c_a,
            c_b,
            nalpha,
            nbeta,
            s2: fin.s2,
            d_a,
            d_b,
            f_a,
            f_b,
            convergence: Convergence {
                e_tol: opts.e_tol,
                g_tol: opts.g_tol,
                g_norm: opts.g_norm,
            },
            history: fin.history,
            timers: fin.timers,
        }
    }
}
