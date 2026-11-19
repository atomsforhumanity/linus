//! The ground state: one SCF at one geometry, as the SCF stage leaves it, handed explicitly
//! to the derivatives, the properties and the report of that geometry (BAGEL's reference
//! passing).
//!
//! A [`GroundState`] is immutable and shared (`Arc`). It carries what the SCF was set up with
//! ([`ScfSetup`]: its purpose, which resolves the "auto" auxiliary basis, the grids, SMD's
//! contact smoothing and the convergence; its XC grid; the memory its setups sized against)
//! and what it produced beside the wavefunction (VV10's statistics, the solvation results with
//! the PCM state the derivatives reuse, the density-fitting storage). Everything a gradient, a
//! Hessian or a polarizability reads of "its SCF" comes from the ground state it is given, so
//! their results do not depend on which SCF the session ran last.
//!
//! The ground state owns its reference determinant ([`linus_scf::reference::Reference`]: the
//! orbital blocks and densities the derivatives expand around), built once on first use and
//! shared by the gradient, the Hessian, the field response and the polarizability of the SCF.
//!
//! The session keeps the last converged one (`Session::last`), and the last unconverged
//! one for its failure report. Before its next SCF the session releases the kept one's
//! reference ([`GroundState::release_reference`]), so the reference's matrices are not alive
//! through the next SCF.

use crate::driver::lineage::GuessRecord;
use crate::driver::{GridChoice, Purpose};
use faer::Par;
use linus_base::memory::{DfBudget, MemoryLimit, Reservation};
use linus_dft::vv10::Vv10Stats;
use linus_jk::df::ResolvedDf;
use linus_scf::reference::Reference;
use linus_scf::ScfResult;
use linus_solvent::SolvationResult;
use std::sync::OnceLock;

/// What an SCF is set up with, fixed before its setup starts and kept by its ground state for
/// the derivatives of its geometry.
#[derive(Clone, Debug)]
pub(crate) struct ScfSetup {
    /// What the SCF is converged for.
    pub purpose: Purpose,
    /// The DFT grid (the settings' `grid` resolved for the purpose); the XC parts of the
    /// gradient, the Hessian and the response properties of this SCF use the same grid.
    pub grid: GridChoice,
    /// The memory its large tensors are sized against.
    pub memory: ScfMemory,
}

/// The memory of an SCF's large tensors, taken once before its setup: the reservations of the
/// large allocations alive with its density-fitting tensors (the PCM's dense matrices) and the
/// budget of those tensors that leaves (with `memory_gib` unset, the guard reads the free memory
/// here, once per SCF). The derivatives of its geometry size their tensors against the same
/// budget.
#[derive(Clone, Debug)]
pub(crate) struct ScfMemory {
    pub reserved: Vec<Reservation>,
    pub df_budget: DfBudget,
    /// The SCF's need beside its DF tensors and their exchange workspace (bytes: the process,
    /// the reservations, the metric, the measured rest; `crate::driver::needs`).
    pub fixed: usize,
    /// With `memory_gib`: what the whole process leaves the DF tensors and their exchange
    /// workspace, the limit less `fixed` (`linus_jk::df::DfConfig::room`).
    pub room: Option<usize>,
}

impl ScfMemory {
    /// The budget of `limit` less the `reserved` allocations; `fixed` the SCF's other parts.
    pub fn new(limit: MemoryLimit, reserved: Vec<Reservation>, fixed: usize) -> Self {
        let df_budget = limit.df_budget(reserved.iter().map(|r| r.bytes).sum());
        ScfMemory {
            reserved,
            df_budget,
            fixed,
            room: limit.limit.map(|l| l.saturating_sub(fixed)),
        }
    }

    /// Bytes of the 3-index tensors (DF, the RI-J of rij_snk, the in-core exact integrals, a
    /// GPU's host-side tensor).
    pub fn df_bytes(&self) -> usize {
        self.df_budget.bytes
    }
}

/// One SCF at one geometry (see the module docs): converged, or the last unconverged one kept
/// for its failure report.
pub(crate) struct GroundState {
    /// What the SCF was set up with.
    pub setup: ScfSetup,
    /// The SCF result (with the dispersion energy when converged).
    pub scf: ScfResult,
    /// The coordinates it belongs to (bohr).
    pub positions: Vec<[f64; 3]>,
    /// VV10's statistics (-V functionals): its energy (part of the XC energy), grid and timings.
    pub vv10: Option<Vv10Stats>,
    /// The solvation results (with `solvent`), the PCM state the derivatives reuse included.
    pub solvation: Option<SolvationResult>,
    /// The density-fitting storage (jk df and the RI-J of rij_snk).
    pub df: Option<ResolvedDf>,
    /// Where the SCF started (`record.scf.guess`: its lineage's history or a fresh guess).
    pub guess: GuessRecord,
    /// The reference determinant of its derivatives, built on first use.
    reference: OnceLock<Reference>,
}

impl GroundState {
    /// The ground state of the SCF `scf` at `positions`, set up with `setup` and started as
    /// `guess` records, with its side results.
    pub fn new(
        setup: ScfSetup,
        scf: ScfResult,
        positions: Vec<[f64; 3]>,
        guess: GuessRecord,
        vv10: Option<Vv10Stats>,
        solvation: Option<SolvationResult>,
        df: Option<ResolvedDf>,
    ) -> Self {
        GroundState {
            setup,
            scf,
            positions,
            vv10,
            solvation,
            df,
            guess,
            reference: OnceLock::new(),
        }
    }

    /// The reference determinant of the SCF ([`Reference::new`] at parallelism `par` in `pool`;
    /// built once, on first use).
    pub fn reference(&self, pool: &rayon::ThreadPool, par: Par) -> &Reference {
        // not get_or_init across the pool: the build runs rayon and nothing else asks for it
        // meanwhile (one call at a time per session)
        if let Some(r) = self.reference.get() {
            return r;
        }
        let r = pool.install(|| Reference::new(&self.scf, par));
        self.reference.get_or_init(|| r)
    }

    /// Frees the reference determinant (it is built again on first use).
    pub fn release_reference(&mut self) {
        self.reference.take();
    }

    /// What the SCF was converged for.
    pub fn purpose(&self) -> Purpose {
        self.setup.purpose
    }

    /// The DFT grid of the SCF.
    pub fn grid(&self) -> &GridChoice {
        &self.setup.grid
    }

    /// The derivative order of the shell pairs its derivatives need: none for an energy, first
    /// for a gradient, second for a Hessian (crate::context builds one set at that order).
    pub fn pair_order(&self) -> linus_ints::DerivOrder {
        match self.setup.purpose {
            Purpose::Energy => linus_ints::DerivOrder::Zero,
            Purpose::Gradient => linus_ints::DerivOrder::One,
            Purpose::Hessian => linus_ints::DerivOrder::Two,
        }
    }

    /// Whether it belongs to the coordinates `positions`.
    pub fn is_at(&self, positions: &[[f64; 3]]) -> bool {
        self.positions == positions
    }

    /// Whether it serves a response property at `positions` as an SCF of its own would: at these
    /// coordinates and converged for a response (the Hessian purpose: its grid, auxiliary basis
    /// and thresholds).
    pub fn serves_response_at(&self, positions: &[[f64; 3]]) -> bool {
        self.is_at(positions) && self.setup.purpose == Purpose::Hessian
    }
}
