//! linus-scf: the self-consistent field (restricted and unrestricted Hartree-Fock and Kohn-Sham),
//! its linear response ([`response`]: CPHF/CPKS and the second-order properties' framework) and the
//! reference determinant derivatives and responses expand around ([`reference`](mod@reference)).
//! The SCF and the response reach the method families only through linus-base's terms
//! (`linus_base::terms`): the J/K, XC and environment crates are its siblings, not its
//! dependencies.
//!
//! # The SCF
//!
//! Everything is written in the unrestricted form F_s = h + J[D_a + D_b] - a_x K\[D_s\] + Vxc_s;
//! the restricted case uses one set of orbitals with D_a = D_b.
//!
//! Iteration n builds F(D_n), the energy E(D_n) and the orbital gradient (the commutator
//! X^T (F D S - S D F) X in the orthonormal basis X), stops when |E_n - E_{n-1}| < e_tol and the
//! gradient norm is below g_tol, and otherwise makes new orbitals by diagonalizing a DIIS
//! extrapolated Fock matrix (ADIIS early on if asked; optionally damped or level-shifted while
//! the gradient is large) or, near convergence and if enabled, by a quasi-Newton orbital
//! rotation (SOSCF).
//!
//! Incremental Fock builds: for J/K builders whose cost falls with |D| (integral screening on
//! the density, [`JkBuilder::incremental_capable`]), J and K are built from D_n - D_{n-1} and
//! added to the previous matrices, with a full rebuild every `incremental_reset` builds and once
//! more whenever an incremental Fock matrix signals convergence: the final energy, Fock matrix
//! and orbitals always come from one full build (the returned Fock matrix is the operator of
//! one build, with the returned orbitals as its eigenvectors). The same holds for terms that
//! build from their own history (`built_incrementally`: COSX's incremental K, the PCM's
//! `predicted` charges and incremental passes): such a build counts as incremental, and the
//! confirming build resets every term's history, so it is of the density alone (exact PCM
//! charges, a full K).
//!
//! Builders with a sequence of approximations ([`JkBuilder::next_approximation`]; COSX: a small
//! SCF grid, then a larger one, as ORCA) are moved to the next once max |F_ai| <
//! `grid_switch`, and always before the SCF may converge. K is rebuilt at the same density and
//! the DIIS subspace is carried over: each stored Fock matrix is shifted by the change of the
//! exchange matrix at the current density, F_i + (K_old - K_new)\[D_n\], and its error recomputed
//! with its own density. Without the shift DIIS mixes Fock matrices of two operators, and the
//! orbital gradient stays at the grid-change level (~5e-4) for 2-3 iterations.
//!
//! Open shells (bench/results/2026-11-07-openshell-scf):
//! - DIIS keeps its best entries (drop policy LargestError) only while the SCF stays near them:
//!   once a new error is `diis_unpin` times the smallest stored one, the oldest entries are
//!   dropped instead. Linear radicals with a degenerate pi hole (HBr+, HI+, UKS) stall below
//!   max |F_ai| ~ 5e-9: the orientation of the hole about the axis is a soft mode that only the
//!   DFT grid breaks (the energy varies by ~1e-8 Eh with it). Ill-conditioned DIIS steps then
//!   move the hole, and the kept entries of the old orientation pinned the extrapolation: a
//!   fixed point with max |F_ai| 7e-7 to 1e-6 and |dD| -> 1e-9.
//! - DIIS converges to any stationary point. A UHF/UKS SCF that converges above the lowest
//!   energy it visited (`saddle_check`) is on a saddle point: the lowest orbital-Hessian mode
//!   (`lowest_hessian_mode`, finite differences of the SCF's own Fock builds) is followed
//!   downhill (rotations of 0.25, 0.5 and 1 along it; if none lowers the energy, 0.125 halved
//!   down to 1/64) and the SCF continues with quasi-Newton steps, which descend in energy (DIIS
//!   climbed back to the saddle). L-lactide radical cation, wB97X-D3BJ/def2-TZVP: 2.3 mEh below
//!   the state DIIS, ORCA and PySCF converge to.
//!
//! Closed shells (decision D36, bench/results/2026-11-11-scfrob): an RHF/RKS solution may lie
//! above a broken-symmetry UHF/UKS one (stretched bonds, singlet diradicals; 30 of 886 closed-shell
//! molecules with B3LYP/def2-SVP). At a converged restricted point whose HOMO-LUMO gap lies below
//! `closed_shell_gap`, the closed-shell probe takes `closed_shell_probe` one-sided products of
//! the UHF/UKS orbital Hessian (UKS-type Fock builds) from the external start and records the
//! Ritz value ([`ScfResult::closed_shell_probe`]); the SCF never switches, and a build at the
//! converged state afterwards leaves the J/K, XC and environment terms as the SCF left them, so the
//! result is the one without the probe, bit for bit.
//!
//! Energies are traced with compensated sums ([`linus_base::linalg::dot`]) and the eigensolver runs
//! with a fixed parallel degree, so with thread-count-independent J/K and XC builders the
//! SCF is bitwise independent of the number of threads.
//!
//! # Structure
//!
//! The SCF is one algorithm, evaluate → observe → decide → step ([`run_scf`], `run.rs`):
//! - `fock`: the `FockEvaluator`, the only caller of the J/K builder, the XC
//!   term and the environment term. Accepted evaluations are the iterates (incremental builds, the
//!   J/K approximations); trial evaluations, for the stability analysis, the mode check and the
//!   closed-shell probe, leave no state behind;
//! - `control`: the control policy, pure state machines on each iteration's scalars (convergence,
//!   the lowest energy, two-cycle, plateau, escalation, the stability trigger);
//! - `accel`: the accelerators (DIIS, ADIIS, damping, SOSCF, the level shift) and their restarts;
//! - `analysis` and `stability`: the analyses of converged points (Davidson on finite differences of
//!   the SCF's own Fock builds) and their records;
//! - the device loop's controller and Fock terms (the facade's `backend::gpu::scf_control`,
//!   feature `gpu` there) decide over this loop's own controls and hand over to this loop what
//!   they do not run themselves. They are built from the SCF's units (the control policy, the
//!   accelerators' rules, the analyses): `accel`, `analysis`, `control`, `fock`, `result`,
//!   `stability` and `trace` are public for them, hidden from the documentation.

#[doc(hidden)]
pub mod accel;
#[doc(hidden)]
pub mod analysis;
#[doc(hidden)]
pub mod control;
#[doc(hidden)]
pub mod fock;
pub mod guess;
mod loops;
pub mod reference;
pub mod response;
#[doc(hidden)]
pub mod result;
#[doc(hidden)]
pub mod run;
#[doc(hidden)]
pub mod settings;
#[doc(hidden)]
pub mod stability;
#[doc(hidden)]
pub mod state;
#[doc(hidden)]
pub mod trace;

pub use result::{
    Escalation, EscalationRun, ModeCheck, Plateau, PlateauRefusal, ScfIteration, ScfResult,
    MODE_CHECK_THRESHOLD,
};
pub use settings::{
    DiisDrop, Guess, ScfAlgorithm, ScfConfig, CLOSED_SHELL_GAP, CLOSED_SHELL_PROBE_AUTO,
    STABILITY_PROBE_AUTO,
};
pub use stability::{
    gate_gap, ClosedShellProbe, ProbeKind, ProbeSkip, ProbeTrigger, StabilityCheck, TriggerReason,
    TriggerRecord, GAP_DROP, PROBE_NEGATIVE,
};

pub(crate) use accel::Diis;
use fock::FockTerms;
pub use loops::{HostLoop, ScfLoop};
use state::OrbitalState;

use faer::{Mat, MatRef, Par};
use linus_base::error::LinusError;
use linus_base::linalg::{mm, try_eigh};
use linus_base::terms::JkBuilder;
use linus_base::terms::{EnvironmentTerm, XcPotential};
use linus_ints::operator::Operator;

pub struct ScfInput<'a> {
    pub h: MatRef<'a, f64>,
    pub s: MatRef<'a, f64>,
    pub e_nuc: f64,
    pub nalpha: usize,
    pub nbeta: usize,
    /// The method's exact exchange (`MethodInfo::exchange`;
    /// [`exchange_parts`](linus_base::terms::exchange_parts)); the J/K builder is set up with its
    /// operator.
    pub exchange: Option<Operator>,
    /// Initial spin densities (overrides opts.guess).
    pub guess_density: Option<(Mat<f64>, Mat<f64>)>,
    /// The atoms and basis set: the point group of the stability analyses' searches (None: they
    /// ignore symmetry, as for a molecule without).
    pub geometry: Option<stability::Geometry<'a>>,
}

/// Diagonalize F in the orthonormal basis X: returns (eps, C). Panics on non-finite elements
/// ([`try_diagonalize`] returns None).
pub fn diagonalize(f: MatRef<'_, f64>, x: MatRef<'_, f64>, par: Par) -> (Vec<f64>, Mat<f64>) {
    try_diagonalize(f, x, par).expect("Fock matrix with non-finite elements")
}

/// [`diagonalize`], None when F (in the orthonormal basis) has non-finite elements.
pub fn try_diagonalize(
    f: MatRef<'_, f64>,
    x: MatRef<'_, f64>,
    par: Par,
) -> Option<(Vec<f64>, Mat<f64>)> {
    let fp = mm(x.transpose(), mm(f, x, par).as_ref(), par);
    let (eps, cp) = try_eigh(fp.as_ref(), par).ok()?;
    Some((eps, mm(x, cp.as_ref(), par)))
}

/// All elements finite.
fn all_finite(m: &Mat<f64>) -> bool {
    (0..m.ncols()).all(|j| m.col_as_slice(j).iter().all(|x| x.is_finite()))
}

pub fn density(c: MatRef<'_, f64>, nocc: usize, par: Par) -> Mat<f64> {
    let co = c.subcols(0, nocc);
    mm(co, co.transpose(), par)
}

/// The state of an SCF at the start of iteration `iter`, from which [`run_scf_resumed`] continues
/// as [`run_scf`] would have from there: the device loop's hand-over (the facade's
/// `backend::gpu::scf_control`), which ran iterations 1 to `iter` - 1 with the same algorithm.
pub struct ScfResume {
    pub iter: usize,
    /// The orbitals D_iter comes from, per channel (nbf x nmo, occupied first; one channel for
    /// a closed shell).
    pub c: Vec<Mat<f64>>,
    /// D_iter-1 per channel (None at iteration 1).
    pub prev: Option<Vec<Mat<f64>>>,
    /// E_iter-1.
    pub e_prev: f64,
    /// The controls before iteration `iter` (the lowest energy, the two-cycle count, ...).
    pub controls: control::Controls,
    /// The occupied orbitals of the lowest-energy iterate's density (`Controls::lowest`).
    pub min_occ: Option<Vec<Mat<f64>>>,
    /// 1 when iteration 1's density was a guess, else 0.
    pub guess_iter: usize,
    /// Iterations 1 to `iter` - 1.
    pub history: Vec<ScfIteration>,
}

/// The SCF of `inp` with the J/K builder `jk` (set up with the exchange operator of
/// `inp.exchange`), the XC term `xc` (Kohn-Sham) and an environment term `env` (a solvent).
/// An SCF that stops without converging is a result (`converged` false); an error is a failed
/// J/K or XC build (a device backend's [`LinusError::Backend`]), which stops the SCF.
pub fn run_scf(
    inp: &ScfInput,
    jk: &mut dyn JkBuilder,
    xc: Option<&mut dyn XcPotential>,
    env: Option<&mut dyn EnvironmentTerm>,
    opts: &ScfConfig,
    par: Par,
) -> Result<ScfResult, LinusError> {
    run::run(inp, FockTerms::new(jk, xc, env), opts, par, None)
}

/// [`run_scf`] from the state `resume` (its iterations before `resume.iter` included in the
/// result's history; `inp.guess_density` is not used).
pub fn run_scf_resumed(
    inp: &ScfInput,
    jk: &mut dyn JkBuilder,
    xc: Option<&mut dyn XcPotential>,
    env: Option<&mut dyn EnvironmentTerm>,
    opts: &ScfConfig,
    par: Par,
    resume: Box<ScfResume>,
) -> Result<ScfResult, LinusError> {
    run::run(inp, FockTerms::new(jk, xc, env), opts, par, Some(*resume))
}

#[cfg(test)]
mod tests {
    use super::*;
    use linus_base::terms::{JkOutput, JkRequest, XcDensity};

    /// No electron-electron terms: a two-function model's SCF is its core Hamiltonian's.
    struct NoJk;

    impl JkBuilder for NoJk {
        fn name(&self) -> String {
            "none".into()
        }

        fn compute(&mut self, req: &JkRequest, _par: Par) -> Result<JkOutput, LinusError> {
            let z = |d: &Mat<f64>| Mat::zeros(d.nrows(), d.ncols());
            Ok(JkOutput {
                j: req.dens.iter().map(z).collect(),
                k: req.dens.iter().map(z).collect(),
            })
        }
    }

    /// An environment term that fails at its `fail_at`-th call (the device PCM's failed step).
    struct FailingTerm {
        calls: usize,
        fail_at: usize,
    }

    impl EnvironmentTerm for FailingTerm {
        fn name(&self) -> &'static str {
            "solvent"
        }

        fn compute(
            &mut self,
            d: &XcDensity<'_>,
            _par: Par,
        ) -> Result<(f64, Option<Mat<f64>>), LinusError> {
            self.calls += 1;
            if self.calls == self.fail_at {
                return Err(LinusError::Backend {
                    kernel: "pcm",
                    message: "a failed step".into(),
                });
            }
            let n = d.da.nrows();
            Ok((
                -0.01,
                Some(Mat::from_fn(n, n, |i, j| 1e-3 * (i + j) as f64)),
            ))
        }
    }

    /// An environment term's error stops the SCF with that error (the device PCM's used to be a
    /// NaN energy, an unconverged result, and the device's error held for after the SCF).
    #[test]
    fn an_environment_term_error_stops_the_scf() {
        let h = Mat::from_fn(
            2,
            2,
            |i, j| if i == j { -1.0 + 0.5 * i as f64 } else { -0.2 },
        );
        let s = Mat::<f64>::identity(2, 2);
        let inp = ScfInput {
            h: h.as_ref(),
            s: s.as_ref(),
            e_nuc: 0.0,
            nalpha: 1,
            nbeta: 1,
            exchange: None,
            guess_density: None,
            geometry: None,
        };
        let opts = ScfConfig::default();
        // without the failure the model converges
        let mut ok = FailingTerm {
            calls: 0,
            fail_at: usize::MAX,
        };
        let r = run_scf(&inp, &mut NoJk, None, Some(&mut ok), &opts, Par::Seq).unwrap();
        assert!(r.converged && r.energy.is_finite(), "{}", r.energy);
        let mut bad = FailingTerm {
            calls: 0,
            fail_at: 2,
        };
        match run_scf(&inp, &mut NoJk, None, Some(&mut bad), &opts, Par::Seq) {
            Err(LinusError::Backend {
                kernel: "pcm",
                message,
            }) => assert_eq!(message, "a failed step"),
            other => panic!("expected the term's backend error, got {other:?}"),
        }
        assert_eq!(bad.calls, 2, "the SCF stopped at the failing step");
    }

    /// A `scf.lindep` that leaves fewer orbitals than occupied ones is an input error before the
    /// loop, naming the counts (it was a faer assertion in `density`): four functions in two
    /// nearly linearly dependent pairs (overlap eigenvalues 1.995 and 0.005, twice), three
    /// occupied. As many orbitals as occupied ones (no virtuals) still runs.
    #[test]
    fn a_lindep_that_drops_occupied_orbitals_is_an_input_error() {
        let pair = |i: usize, j: usize| i / 2 == j / 2;
        let s = Mat::from_fn(4, 4, |i, j| match (i == j, pair(i, j)) {
            (true, _) => 1.0,
            (false, true) => 0.995,
            _ => 0.0,
        });
        let h = Mat::from_fn(
            4,
            4,
            |i, j| if i == j { -1.0 + 0.3 * i as f64 } else { -0.1 },
        );
        for (nalpha, nbeta) in [(3, 3), (3, 1)] {
            let inp = ScfInput {
                h: h.as_ref(),
                s: s.as_ref(),
                e_nuc: 0.0,
                nalpha,
                nbeta,
                exchange: None,
                guess_density: None,
                geometry: None,
            };
            let mut opts = ScfConfig::default();
            opts.algo.lindep = 1e-2;
            match run_scf(&inp, &mut NoJk, None, None, &opts, Par::Seq) {
                Err(LinusError::Input(m)) => assert_eq!(
                    m,
                    "scf.lindep 1e-2 drops the basis to 2 orbitals, fewer than the 3 occupied"
                ),
                other => panic!("({nalpha}, {nbeta}): expected an input error, got {other:?}"),
            }
            if nalpha == nbeta {
                continue;
            }
            // two orbitals for two occupied (alpha 2, beta 1): no virtuals, converged
            let inp = ScfInput { nalpha: 2, ..inp };
            let r = run_scf(&inp, &mut NoJk, None, None, &opts, Par::Seq).unwrap();
            assert!(r.converged && r.nmo == 2 && r.dropped_functions == 2);
        }
    }
}
