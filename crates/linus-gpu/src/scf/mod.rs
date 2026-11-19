//! A device-resident SCF loop: restricted and unrestricted Hartree-Fock and Kohn-Sham with
//! the orbitals, densities, Fock matrices and the DIIS subspace on the device.
//!
//! It is the mechanism of linus's SCF (crates/linus/src/scf/) on the device, for the steps the
//! host loop takes by default with a density-fitted J/K builder:
//! - canonical orthogonalization X = U s^-1/2 over the overlap eigenvalues above `lindep`
//!   (cuSOLVER `syevd` of S);
//! - iteration n: the Fock terms at D_n through `FockTerms`; F_s = h + J - a_x K_s + V_s;
//!   the energy E = tr(D h) + 1/2 tr(D J) - a_x/2 sum_s tr(D_s K_s) + E_xc + E_env + E_nuc;
//!   the orbital gradient X^T (F_s D_s S - S D_s F_s) X;
//! - a step: DIIS (Pulay), the level shift F + shift (S - S D S), diagonalization in the basis X
//!   (cuSOLVER `syevd`), and D_s = C_s,occ C_s,occ^T;
//! - after the last iteration, orbitals and orbital energies of the unextrapolated Fock
//!   matrices, and <S^2>.
//!
//! The decisions are not the loop's: after each Fock build it reports the iteration's scalars
//! ([`Observation`]) to its controller ([`ScfDecide`]), which says whether the SCF converged,
//! stops for the caller, or steps, and how (the DIIS entry and the entry it replaces, the
//! extrapolation, the level shift; [`Verdict`]). linus's controller is the host loop's own
//! controls, so both loops decide alike. A build that a term made from its own history (the PCM's
//! predicted charges, `FockTerms::built_incrementally`) and that meets the criteria is
//! confirmed: the loop builds again at the same state, of the densities alone, and asks again
//! ([`Verdict::Confirm`]), so the SCF ends on one build of its density.
//!
//! What crosses PCIe per iteration: a few dozen scalars (energy terms, gradient norms, the new
//! row of the DIIS matrix) and the DIIS coefficients; the small DIIS system is solved on the
//! host ([`diis::DiisBook`]). The Fock terms decide their own traffic: linus-gpu's engines
//! (`DeviceTerms`: `GpuDfJk`, `GpuXc`) read and write device buffers, the CPU fallback
//! (`OnHost`) downloads the densities and uploads its matrices.
//!
//! Not on the device (the host loop's options that are off by default or do not apply to
//! density-fitted J/K): ADIIS, damping, SOSCF, incremental J/K builds and J/K builders with a
//! sequence of approximations (COSX grids). Where the controller leaves the plain DIIS
//! iteration, the loop stops and hands the caller what it needs to continue (`HandOver`, with
//! the `gpu` feature): at a converged point it analyses ([`Stop::StabilityCheck`]; the products of
//! the analysis run on the device through `Rotations`), at a two-cycle ([`Stop::TwoCycle`], before
//! the step), after a step when the controller says so ([`Stop::HandOver`]: linus's escalation, which
//! the host loop runs), and with [`LoopConfig::handover_at_max_iter`] at `max_iter`. The last
//! iteration of the budget is built of the densities alone (`FockTerms::reset_incremental`), so the
//! loop ends on one build of its density wherever it stops.
//!
//! Determinism: cuBLAS and cuSOLVER (deterministic mode) are bitwise reproducible for a fixed
//! configuration, the element-wise kernels have one writer per element, and the reductions
//! run in a fixed tree ([`kernels`]), so the loop is bitwise reproducible run to run on a
//! device given reproducible Fock terms. Against the CPU it agrees to round-off: a different
//! eigensolver and GEMM order (energies to ~1e-12 Eh, the same iteration counts).
//!
//! Layout: matrices are column-major. Per-spin matrices lie back to back in one buffer (one
//! for a closed shell, alpha then beta otherwise), and the occupied orbitals side by side
//! (nbf x (n_alpha + n_beta); nbf x n_alpha for a closed shell), the layout of the device J/K
//! and XC engines.

pub mod diis;
pub mod kernels;

#[cfg(feature = "gpu")]
mod engines;
#[cfg(feature = "gpu")]
mod host;
#[cfg(feature = "gpu")]
mod rotations;
#[cfg(feature = "gpu")]
mod run;

pub use diis::DiisBook;
#[cfg(feature = "gpu")]
pub use engines::DeviceTerms;
#[cfg(feature = "gpu")]
pub use host::{HostFockTerms, HostState, OnHost};
#[cfg(feature = "gpu")]
pub use rotations::{Rotation, Rotations};
#[cfg(feature = "gpu")]
pub use run::run_scf;

/// The device eigensolver of the loop's diagonalizations (cuSOLVER).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Eigensolver {
    /// Divide and conquer (`syevd`).
    DivideConquer,
    /// Divide and conquer through the 64-bit generic API (`Xsyevd`).
    DivideConquer64,
    /// Cyclic Jacobi (`syevj`).
    Jacobi,
}

/// The device loop's own parameters: its mechanism. What it decides (convergence, the DIIS
/// subspace, the level shift, when to stop for the caller) is asked of its [`ScfDecide`].
#[derive(Clone, Debug, PartialEq)]
pub struct LoopConfig {
    /// Iterations at most.
    pub max_iter: usize,
    /// Entries of the DIIS subspace: its device buffers, and the capacity the controller sees.
    pub diis_space: usize,
    /// Overlap eigenvalues at or below this are dropped by the orthogonalization.
    pub lindep: f64,
    pub eigensolver: Eigensolver,
    /// Keep the occupied orbitals of the lowest-energy iterate (the iterations the controller
    /// calls the lowest) for a hand-over.
    pub keep_lowest: bool,
    /// At [`Stop::MaxIterations`], leave the loop's state after the last step in
    /// `ScfResult::handover`, for the caller to analyse (linus's plateau of an open shell).
    pub handover_at_max_iter: bool,
    /// At [`Stop::Converged`], keep the occupied orbitals of the last Fock build's density in
    /// `ScfResult::occupied` (for the caller's builds at the converged state).
    pub keep_occupied: bool,
}

impl Default for LoopConfig {
    fn default() -> Self {
        LoopConfig {
            max_iter: 128,
            diis_space: 10,
            lindep: 1e-8,
            eigensolver: Eigensolver::DivideConquer,
            keep_lowest: false,
            handover_at_max_iter: false,
            keep_occupied: false,
        }
    }
}

/// The decisions of the device loop's SCF, which it asks after each Fock build. linus implements
/// it with the host loop's own controls (`scf::control` and the accelerators' rules), so the two
/// loops decide alike from the same code; the device loop is the mechanism only.
pub trait ScfDecide {
    /// What the loop does after the iteration `o`.
    fn observe(&mut self, o: &Observation<'_>) -> Verdict;
}

/// The scalars of one device-loop iteration.
#[derive(Clone, Copy, Debug)]
pub struct Observation<'a> {
    pub iter: usize,
    /// 1 when iteration 1's density was a guess (not from orbitals), else 0.
    pub guess_iter: usize,
    pub energy: f64,
    /// E_iter - E_iter-1.
    pub de: f64,
    /// max |F_ai| over spins, and PySCF's gradient norm (sqrt(2) ||E|| for a closed shell,
    /// sqrt((||E_a||^2 + ||E_b||^2) / 2) otherwise).
    pub gmax: f64,
    pub gnorm_pyscf: f64,
    /// The energies of the two iterations before (E_iter-1, E_iter-2; None before there were any).
    pub previous: [Option<f64>; 2],
    /// The DIIS subspace: the squared error norms of its entries (in order), its capacity, and
    /// this iteration's squared error norm.
    pub diis_norms: &'a [f64],
    pub diis_capacity: usize,
    pub error_norm: f64,
    /// The build is the one the controller asked for at the same state ([`Verdict::Confirm`]).
    pub rebuilt: bool,
    /// The build was not of the densities alone (`FockTerms::built_incrementally`).
    pub incremental: bool,
}

/// What the device loop does after an iteration.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Verdict {
    /// Converged: stop ([`Stop::Converged`]).
    Converged,
    /// A converged point that the caller analyses: stop with [`Stop::StabilityCheck`] and the
    /// hand-over state.
    Analyse,
    /// A two-cycle: stop before the step with [`Stop::TwoCycle`] and the hand-over state.
    TwoCycle,
    /// Converged on an incremental build: build again at the same state, of the densities alone
    /// (`FockTerms::reset_incremental`), and ask again ([`Step::Rebuild`]).
    Confirm,
    /// A step to new orbitals.
    Step(StepPlan),
    /// The caller asks the SCF to stop: stop unconverged ([`Stop::Requested`], [`Step::Stopped`]).
    Stop,
}

/// How the loop makes the next orbitals.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StepPlan {
    /// The iteration is the lowest-energy one so far (with [`LoopConfig::keep_lowest`] its
    /// occupied orbitals are kept).
    pub lowest: bool,
    /// The Fock matrix enters the DIIS subspace, replacing the entry `diis_victim` (Some exactly
    /// when the subspace is full).
    pub diis_push: bool,
    pub diis_victim: Option<usize>,
    /// Diagonalize the DIIS extrapolation (else the Fock matrix).
    pub extrapolate: bool,
    /// The level shift of the virtual orbitals (Eh), when on.
    pub level_shift: Option<f64>,
    /// After the step, stop with [`Stop::HandOver`] and the loop's state, for the caller to
    /// continue (linus's escalation).
    pub hand_over: bool,
}

/// How an iteration made the next orbitals (linus's `ScfIteration::step`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Step {
    Diis,
    Roothaan,
    Converged,
    NonFinite,
    StabilityCheck,
    TwoCycle,
    /// No step: the next build is at the same state ([`Verdict::Confirm`]).
    Rebuild,
    /// No step: the SCF stopped on its caller's request ([`Verdict::Stop`]).
    Stopped,
}

impl Step {
    /// linus's name of the step.
    pub fn as_str(self) -> &'static str {
        match self {
            Step::Diis => "diis",
            Step::Roothaan => "roothaan",
            Step::Converged => "converged",
            Step::NonFinite => "non_finite",
            Step::StabilityCheck => "stability",
            Step::TwoCycle => "two_cycle",
            Step::Rebuild => "rebuild",
            Step::Stopped => "stopped",
        }
    }
}

/// Why the loop stopped.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stop {
    Converged,
    MaxIterations,
    /// A non-finite energy, Fock matrix or orbital (the energy is NaN).
    NonFinite,
    /// A converged UHF/UKS point that needs the stability analysis (`saddle_check`,
    /// `stability`); `ScfResult::handover` holds the loop's state.
    StabilityCheck,
    /// A two-cycle (`two_cycle`), detected at the last iteration before its step;
    /// `ScfResult::handover` holds the loop's state.
    TwoCycle,
    /// The controller asked for a hand-over after the last iteration's step
    /// ([`StepPlan::hand_over`]); `ScfResult::handover` holds the loop's state after it.
    HandOver,
    /// The controller stopped the SCF for its caller ([`Verdict::Stop`]), unconverged.
    Requested,
}

/// Seconds of one iteration's phases. The Fock terms are wall clock (the term is waited for);
/// the loop's own phases are device time from marks on the stream.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct PhaseTimes {
    pub jk: f64,
    pub xc: f64,
    pub environment: f64,
    /// Fock assembly, energy, orbital gradient and the batched reductions.
    pub evaluate: f64,
    /// The DIIS push and extrapolation (with the small solve on the host).
    pub diis: f64,
    /// Level shift and diagonalization (to the occupied orbitals).
    pub diag: f64,
    /// The new densities and the checks of the step.
    pub density: f64,
    /// The whole iteration (wall clock).
    pub wall: f64,
}

impl PhaseTimes {
    /// The loop's own work: evaluate + diis + diag + density.
    pub fn own(&self) -> f64 {
        self.evaluate + self.diis + self.diag + self.density
    }
}

/// One iteration (linus's `ScfIteration`).
#[derive(Clone, Debug, PartialEq)]
pub struct Iteration {
    pub iter: usize,
    pub energy: f64,
    pub de: f64,
    /// max |F_ai| over spins.
    pub gmax: f64,
    pub grms: f64,
    pub gnorm_pyscf: f64,
    /// max and RMS of D_n - D_n-1 (0 in iteration 1).
    pub ddmax: f64,
    pub ddrms: f64,
    /// The build was not of the densities alone (`FockTerms::built_incrementally`).
    pub incremental: bool,
    pub step: Step,
    pub level_shift: bool,
    /// Entries of the DIIS subspace after this iteration.
    pub diis_size: usize,
    pub times: PhaseTimes,
}

/// Device memory of the loop.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LoopMemory {
    /// Bytes the loop held at its largest (state, scratch, eigensolver workspace, DIIS).
    pub loop_bytes: usize,
    /// Of these, the DIIS subspace.
    pub diis_bytes: usize,
    /// The device pool's high-water mark at the end (all users of the device, since the handle
    /// opened or its last `Gpu::reset_peak`).
    pub pool_peak: usize,
}

#[cfg(feature = "gpu")]
pub use device::*;

#[cfg(feature = "gpu")]
mod device {
    use super::*;
    use crate::error::GpuError;
    use crate::{DeviceBuf, Gpu};

    /// The state the Fock terms are evaluated at.
    pub struct FockState<'a> {
        pub nbf: usize,
        pub restricted: bool,
        /// Spin densities, nbf x nbf each, back to back: D_a (a closed shell's one channel,
        /// D_b = D_a), then D_b.
        pub densities: &'a DeviceBuf<f64>,
        /// The occupied orbitals of the densities, side by side (nbf x sum(nocc)): C_a, then
        /// C_b; D_s = C_s C_s^T. None while the density is a guess.
        pub orbitals: Option<&'a DeviceBuf<f64>>,
        /// Occupied orbitals per channel: `[n_alpha]` for a closed shell, else `[n_alpha, n_beta]`.
        pub nocc: &'a [usize],
        /// Identifies the densities: a new value whenever they change, unique within the
        /// process (for terms that cache something of the state, as `OnHost` its host copies).
        pub generation: u64,
    }

    impl FockState<'_> {
        /// Spin channels: 1 for a closed shell, else 2.
        pub fn nspin(&self) -> usize {
            self.nocc.len()
        }
    }

    /// The density-dependent Fock contributions of an SCF: Coulomb and exchange, the
    /// exchange-correlation term, an environment term (implicit solvent). The core
    /// Hamiltonian is part of [`ScfInput`]. Device engines and the CPU fallback ([`OnHost`])
    /// implement it; a calculation that mixes them implements it on a struct holding both.
    ///
    /// Outputs are column-major nbf x nbf matrices in caller-owned device buffers, per channel
    /// back to back ([`FockState`]). Work may be queued without synchronizing: the loop orders
    /// everything on the device's stream.
    pub trait FockTerms {
        /// J\[D_s\] of each channel's density into `j` and, when `k` is given, K\[D_s\] into `k`
        /// (unscaled; the loop applies the exchange fraction). A closed shell has one
        /// channel, J\[D_a\] and K\[D_a\]: the loop doubles J.
        fn coulomb_exchange(
            &mut self,
            gpu: &Gpu,
            st: &FockState<'_>,
            j: &mut DeviceBuf<f64>,
            k: Option<&mut DeviceBuf<f64>>,
        ) -> Result<(), GpuError>;

        /// E_xc and V_xc per channel into `v` (a closed shell: one matrix, the potential of the
        /// total density); None when there is no XC term (Hartree-Fock).
        fn exchange_correlation(
            &mut self,
            _gpu: &Gpu,
            _st: &FockState<'_>,
            _v: &mut DeviceBuf<f64>,
        ) -> Result<Option<f64>, GpuError> {
            Ok(None)
        }

        /// E_env and whether the term wrote its own Fock matrix into `v` (one nbf x nbf
        /// matrix, the same for both spins); None when there is no environment term.
        fn environment(
            &mut self,
            _gpu: &Gpu,
            _st: &FockState<'_>,
            _v: &mut DeviceBuf<f64>,
        ) -> Result<Option<(f64, bool)>, GpuError> {
            Ok(None)
        }

        /// True when the last build was not of the densities alone: a term built from its own
        /// history (linus's PCM in its `predicted` mode: charges predicted from the last
        /// iteration's potential). The loop reports it ([`Observation::incremental`]), and its
        /// controller confirms a converged iterate with a full build ([`Verdict::Confirm`]).
        fn built_incrementally(&self) -> bool {
            false
        }

        /// Make the next build of the densities alone (before a [`Verdict::Confirm`]'s build).
        fn reset_incremental(&mut self) {}
    }

    /// The fixed input of an SCF (linus's `ScfInput`).
    pub struct ScfInput<'a> {
        pub nbf: usize,
        /// Core Hamiltonian and overlap, column-major nbf x nbf (symmetric).
        pub h: &'a [f64],
        pub s: &'a [f64],
        pub e_nuc: f64,
        pub nalpha: usize,
        pub nbeta: usize,
        /// One channel of orbitals (a closed shell: `nalpha` = `nbeta`), else two.
        pub restricted: bool,
        /// a_x, the fraction of K in the Fock matrix (linus's `exchange_parts`: 1 for HF and
        /// for range-separated hybrids, whose operator the J/K term applies; 0: no K is
        /// requested).
        pub exchange_fraction: f64,
        /// Initial spin densities (D_a, D_b), column-major; a closed shell's must be equal.
        /// None: the core Hamiltonian's orbitals.
        pub guess: Option<(&'a [f64], &'a [f64])>,
    }

    /// The outcome of [`run_scf`] (linus's `ScfResult`): energies and the iteration history on
    /// the host, orbitals, densities and Fock matrices left on the device.
    pub struct ScfResult {
        pub energy: f64,
        pub e_nuc: f64,
        pub e_one: f64,
        pub e_coulomb: f64,
        pub e_exchange: f64,
        pub e_xc: f64,
        pub e_environment: f64,
        pub converged: bool,
        pub stop: Stop,
        /// Fock builds, one per iteration (iteration 1 is the guess density's).
        pub iterations: usize,
        pub restricted: bool,
        pub nalpha: usize,
        pub nbeta: usize,
        pub nmo: usize,
        /// Basis functions the orthogonalization dropped (nbf - nmo).
        pub dropped_functions: usize,
        pub eps_a: Vec<f64>,
        pub eps_b: Vec<f64>,
        /// <S^2> of the determinant.
        pub s2: f64,
        pub history: Vec<Iteration>,
        /// Orbitals per channel, nbf x nmo each (one for a closed shell).
        pub orbitals: Vec<DeviceBuf<f64>>,
        /// The densities and Fock matrices of the last Fock build, per channel back to back.
        pub densities: DeviceBuf<f64>,
        pub fock: DeviceBuf<f64>,
        /// Setup (uploads, orthogonalizer, guess) in seconds, wall clock.
        pub setup_seconds: f64,
        pub memory: LoopMemory,
        /// The loop's state at a [`Stop::StabilityCheck`], [`Stop::TwoCycle`] or
        /// [`Stop::HandOver`] (and with [`LoopConfig::handover_at_max_iter`] at
        /// [`Stop::MaxIterations`]), for the caller to analyse the point or to continue the SCF.
        pub handover: Option<HandOver>,
        /// With [`LoopConfig::keep_occupied`] at [`Stop::Converged`]: the occupied orbitals of
        /// `densities`, side by side (as [`FockState::orbitals`]).
        pub occupied: Option<DeviceBuf<f64>>,
    }

    /// The device loop's state at its last iteration n (the one it stopped at), from which
    /// linus's host loop can redo iteration n and continue as if it had run from the start (the
    /// controller keeps its own state).
    pub struct HandOver {
        /// The occupied orbitals of D_n, side by side (as [`FockState::orbitals`]).
        pub occupied: DeviceBuf<f64>,
        /// All orbitals D_n came from, per channel (nbf x nmo each: the eigenvectors of the
        /// step of iteration n - 1, occupied first).
        pub step_orbitals: Vec<DeviceBuf<f64>>,
        /// D_n-1 per channel back to back (None at iteration 1).
        pub prev_densities: Option<DeviceBuf<f64>>,
        /// E_n-1 (0 at iteration 1).
        pub e_prev: f64,
        /// With [`LoopConfig::keep_lowest`]: the occupied orbitals of the lowest-energy iterate
        /// before iteration n, side by side (one channel for a closed shell).
        pub min_orbitals: Option<DeviceBuf<f64>>,
        /// 1 when iteration 1's density was a guess (not from orbitals), else 0.
        pub guess_iter: usize,
    }

    /// The next [`FockState::generation`] (unique within the process).
    pub fn next_generation() -> u64 {
        GENERATION.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    }

    static GENERATION: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

    impl ScfResult {
        /// Sum of the loop's own device time over the iterations (s).
        pub fn own_seconds(&self) -> f64 {
            self.history.iter().map(|h| h.times.own()).sum()
        }
    }
}
