//! The implicit solvent's settings and dispatch; the continuum models (C-PCM, COSMO, IEF-PCM,
//! SS(V)PE, SMD) are linus-solvent's:
//!
//! - the `solvent` settings group ([`SolventSettings`]) and its resolution for one molecule into
//!   linus-solvent's `ResolvedPcm` (`settings`), and the checks before any SCF;
//! - the solvent's part of an SCF on the backend the plan puts the PCM on (`scf`: the
//!   environment term and its Coulomb hook, opened and closed around the loop);
//! - its parts of gradients and Hessians (`derivatives`).
//!
//! The value types the settings take and the solvent database they name are linus-solvent's,
//! re-exported here.

pub(crate) mod derivatives;
pub(crate) mod scf;
pub(crate) mod settings;
#[cfg(test)]
mod tests;

pub(crate) use derivatives::*;
pub use linus_solvent::{db, Discretization, PcmModel, PcmPreset};
pub use scf::*;
pub use settings::*;
