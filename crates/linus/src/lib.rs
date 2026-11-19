//! linus: Hartree-Fock and Kohn-Sham DFT energies, gradients and Hessians.
//!
//! The stable interfaces of linus are its command line, the JSON request and result, and the
//! Python package. This Rust crate is unstable (0.x, not published); its documented facade is
//! re-exported here:
//!
//! - [`Session`] (set up with a [`Structure`] and [`CalcOptions`]) runs energies, gradients,
//!   Hessians, frequencies ([`vib`]) and response properties; [`Session::run`] returns the
//!   typed [`Report`] of a [`CalcType`] (a [`Failure`] when it fails), with what it used in
//!   [`Resolved`];
//! - the results: [`ScfResult`], [`GradientResult`], [`HessianResult`], [`DipoleDerivatives`],
//!   [`Polarizability`], [`vib::FrequencyResult`]; the errors: [`LinusError`];
//! - for backend authors, the J/K interfaces of the SCF ([`JkBuilder`]) and of the CPHF/CPKS
//!   responses ([`ResponseJk`]).
//!
//! The other modules are internals (hidden from the documentation; the front ends and the
//! development crate use them). ARCHITECTURE.md maps the concepts onto the modules.

pub(crate) mod backend;
pub(crate) mod context;
#[doc(hidden)]
pub mod derivatives;
pub(crate) mod dispersion;
#[doc(hidden)]
pub mod driver;
pub(crate) mod properties;
#[doc(hidden)]
pub mod solvent;
#[doc(hidden)]
pub mod structure;
#[doc(hidden)]
pub mod two_electron;
pub mod vib;

#[cfg(test)]
mod tests;

pub use backend::device_lost;
pub use derivatives::gradient::{GradientParts, GradientResult};
pub use derivatives::hessian::{HessianOptions, HessianResult};
pub use driver::options::{CalcOptions, GridChoice, JkMethod, Purpose};
pub use driver::plan::estimate::THREADS_USED_WELL;
pub use driver::plan::report::PlanReport;
pub use driver::report::{AoMatrices, AoOutputs, CalcType, Failure, Report, Resolved};
pub use driver::request::Request;
pub use driver::settings::CalcTypes;
pub use driver::Session;
pub use linus_base::error::{LinusError, ResponseFailure, ScfFailure};
pub use linus_base::progress;
/// What a calculation's memory is checked against: the process kinds ([`memory::Process`],
/// `Session::set_process`) and the phases' needs ([`memory::PhaseNeed`]: the plan's `memory`).
pub mod memory {
    pub use linus_base::memory::{Part, PhaseNeed, Process};
}
pub use linus_base::terms::{JkBuilder, ResponseJk};
pub use linus_scf::ScfResult;
pub use properties::dipole::DipoleDerivatives;
pub use properties::polarizability::Polarizability;
pub use structure::Structure;

#[doc(hidden)]
pub use linus_ints;

/// The numerics epoch: which arithmetic produced linus's results. Within an epoch, a result is
/// bitwise the same on every release (at any thread count, and per device), so the version alone
/// does not tell two results of the same input apart; an epoch's change can move them (the
/// CHANGELOG's "Changed: behaviour" gives the sizes). Bump it in any merge that changes results
/// beyond the bitwise guarantee. Reported next to the version: `linus --version`, the result
/// JSON's `numerics_epoch` and Python's `linus.NUMERICS_EPOCH`.
pub const NUMERICS_EPOCH: u32 = 10;

/// Whether this build has the GPU backend (the `gpu` cargo feature; `backend: "gpu"` is valid).
/// Python's `linus.GPU_BUILD`: true in the GPU distribution `afh-linus-cuda13`, false in
/// `afh-linus`, which have the same version.
pub const GPU_BUILD: bool = backend::GPU_BUILT;

/// linus's version as every front end reports it (the CLI's `--version` and JSON `version`,
/// Python's `linus.__version__`, qccompute's program version): the Python distribution's PEP 440
/// form of the workspace's Cargo version (`0.1.0a1` for `0.1.0-alpha.1`).
pub fn version() -> &'static str {
    static VERSION: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    VERSION.get_or_init(|| pep440(env!("CARGO_PKG_VERSION")))
}

/// A Cargo (SemVer) version in PEP 440's form, as maturin names the wheel and Python's metadata
/// has it: `0.1.0-alpha.1` -> `0.1.0a1`, `-beta.2` -> `b2`, `-rc.1` -> `rc1`, and build metadata
/// as the local version label (`0.1.0-alpha.1+cuda13` -> `0.1.0a1+cuda13`).
pub(crate) fn pep440(cargo: &str) -> String {
    let (version, local) = match cargo.split_once('+') {
        Some((v, l)) => (v, Some(l)),
        None => (cargo, None),
    };
    let mut out = match version.split_once('-') {
        Some((release, pre)) => {
            let (tag, n) = pre.split_once('.').unwrap_or((pre, "0"));
            let tag = match tag {
                "alpha" | "a" => "a",
                "beta" | "b" => "b",
                "rc" | "c" => "rc",
                other => other,
            };
            format!("{release}{tag}{n}")
        }
        None => version.to_string(),
    };
    if let Some(l) = local {
        out = format!("{out}+{l}");
    }
    out
}

#[cfg(test)]
mod version_tests {
    #[test]
    fn pep440_forms() {
        use super::pep440;
        assert_eq!(pep440("0.1.0"), "0.1.0");
        assert_eq!(pep440("0.1.0-alpha.1"), "0.1.0a1");
        assert_eq!(pep440("0.1.0-alpha.1+cuda13"), "0.1.0a1+cuda13");
        assert_eq!(pep440("1.2.0-beta.3"), "1.2.0b3");
        assert_eq!(pep440("1.2.0-rc.1"), "1.2.0rc1");
        assert!(!super::version().contains('-'));
    }
}
