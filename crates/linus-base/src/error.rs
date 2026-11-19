//! Errors of a calculation ([`LinusError`]), the SCF failure they carry ([`ScfFailure`]), where a
//! calculation stopped on request ([`Stopped`]) and the did-you-mean hints of their messages (unknown functionals; the basis set and solvent names
//! use the same [`did_you_mean`] from the basis set library, `linus_ints::basis`).

use crate::convergence::{Convergence, GradNorm};
use linus_core::ErrorKind;
use linus_ints::basis::BasisError;

pub use linus_ints::basis::{did_you_mean, edit_distance};

/// Errors of a calculation, with one vocabulary of kinds across the front ends
/// ([`LinusError::kind`], linus-core's `ErrorKind`: input, basis, convergence, unsupported,
/// resource, backend, numerical, internal, stopped): the CLI's `error_kind` (exit code 2 for input and basis set errors, 1 otherwise),
/// the `.kind` of the Python exceptions (`InputError`, a `ValueError`; `UnsupportedError`, a
/// `NotImplementedError`; `ResourceError`, a `MemoryError`; `BackendError`;
/// `ConvergenceError`; `NumericalError`, an `ArithmeticError`; all `LinusError`) and the qcdata
/// results' extras.
#[derive(Debug, thiserror::Error)]
pub enum LinusError {
    /// Settings or molecule the calculation cannot run as given (fix the input).
    #[error("{0}")]
    Input(String),
    /// A setting that cannot take effect in this calculation (decision D41): `code`
    /// `no_effect` (in no calculation of these settings: `grid` for Hartree-Fock) or
    /// `not_applicable` (not in the calctypes declared: `response.*` for energies), the
    /// setting's dotted `path`, and why. An input error.
    #[error("{path}: {message}")]
    Setting {
        code: &'static str,
        path: String,
        message: String,
    },
    /// A basis set file that cannot be read or parsed.
    #[error("basis set: {0}")]
    Basis(BasisError),
    /// The SCF did not converge (the last iterate is kept as the session's unconverged result).
    #[error("{0}")]
    NotConverged(Box<ScfFailure>),
    /// A CPHF/CPKS solve (analytic Hessian, polarizability) stopped above its tolerance (the
    /// result of the unconverged solution is kept as the session's
    /// `unconverged_response`).
    #[error("{0}")]
    ResponseNotConverged(Box<ResponseFailure>),
    /// A valid request for something linus does not implement.
    #[error("{0}")]
    Unsupported(String),
    /// Not enough memory (or another resource) for the request as set up, with qcdata's shared
    /// code (decision 23) where one names it: `host_memory` (the host's memory: the DF tensors,
    /// the in-core integrals, the solvent's matrices, a Hessian's arrays) or `device_memory` (a
    /// GPU's), each with the memory it needed and could use ([`MemoryNeed`]); None for another
    /// resource (the thread pool could not start).
    #[error("{message}")]
    Resource {
        code: Option<&'static str>,
        message: String,
        /// Boxed: the error stays the size of its largest other variant.
        need: Box<MemoryNeed>,
    },
    /// The GPU backend failed: no usable device for backend `gpu`, or a device error in a
    /// calculation (out of device memory after the setup, a driver or library fault). The
    /// calculation stops (it never switches to the CPU midway); backend `cpu` runs it.
    #[error("GPU backend, {kernel}: {message}; rerun with backend \"cpu\" to use the CPU")]
    Backend {
        /// The kernel family that failed (`device` when no device opened).
        kernel: &'static str,
        message: String,
    },
    /// A numerical failure (non-finite energies, singular matrices).
    #[error("{0}")]
    Numerical(String),
    /// A broken internal invariant: a bug in linus (panics are reported with this kind too).
    #[error("internal error in linus: {0}")]
    Internal(String),
    /// The caller asked the calculation to stop ([`crate::progress::Progress::stop_due`]), and it
    /// stopped at its next step: where, and what it had reached beyond its SCF.
    #[error("{0}")]
    Stopped(Box<Stopped>),
}

impl LinusError {
    /// The kind of failure (linus-core's taxonomy, shared by every layer).
    pub fn error_kind(&self) -> ErrorKind {
        match self {
            LinusError::Input(_) | LinusError::Setting { .. } => ErrorKind::Input,
            LinusError::Basis(_) => ErrorKind::Basis,
            LinusError::NotConverged(_) | LinusError::ResponseNotConverged(_) => {
                ErrorKind::Convergence
            }
            LinusError::Unsupported(_) => ErrorKind::Unsupported,
            LinusError::Resource { .. } => ErrorKind::Resource,
            LinusError::Backend { .. } => ErrorKind::Backend,
            LinusError::Numerical(_) => ErrorKind::Numerical,
            LinusError::Internal(_) => ErrorKind::Internal,
            LinusError::Stopped(_) => ErrorKind::Stopped,
        }
    }

    /// Stopped on request at the start of `stage` (`after`: None) or in its iterations after
    /// iteration `after`.
    pub fn stopped(stage: &'static str, after: Option<usize>) -> LinusError {
        LinusError::Stopped(Box::new(Stopped {
            stage,
            after,
            gradient: None,
            hessian: None,
        }))
    }

    /// The error with what the calculation had reached when it stopped on request (a gradient, a
    /// Hessian), where the stop did not say already; any other error as it is.
    pub fn reached(self, gradient: Option<&[[f64; 3]]>, hessian: Option<&[f64]>) -> LinusError {
        match self {
            LinusError::Stopped(mut s) => {
                if s.gradient.is_none() {
                    s.gradient = gradient.map(<[_]>::to_vec);
                }
                if s.hessian.is_none() {
                    s.hessian = hessian.map(<[_]>::to_vec);
                }
                LinusError::Stopped(s)
            }
            e => e,
        }
    }

    /// The kind as a short name: input, basis, convergence, unsupported, resource, backend,
    /// numerical, internal, stopped.
    pub fn kind(&self) -> &'static str {
        self.error_kind().name()
    }

    /// Errors the user fixes in the input (CLI exit code 2).
    pub fn is_input(&self) -> bool {
        self.error_kind().is_input()
    }

    /// The error's code within its kind, where it has one (qcdata's `ErrorInfo.code`; the CLI's
    /// `error_code`): a setting's (`unknown_key`, `no_effect`, `not_applicable`), a resource's
    /// (`host_memory`, `device_memory`), or a convergence failure's: `scf_not_converged` (the
    /// code every qc* program gives an SCF out of iterations) and `response_not_converged` (a
    /// CPHF/CPKS solve above its tolerance).
    pub fn code(&self) -> Option<&'static str> {
        match self {
            LinusError::Setting { code, .. } => Some(code),
            LinusError::Resource { code, .. } => *code,
            LinusError::NotConverged(_) => Some("scf_not_converged"),
            LinusError::ResponseNotConverged(_) => Some("response_not_converged"),
            _ => None,
        }
    }

    /// Not enough host memory for the request (code `host_memory`): the message says how much
    /// it needs and the ways out, `need` the numbers.
    pub fn host_memory(message: impl Into<String>, need: MemoryNeed) -> LinusError {
        LinusError::Resource {
            code: Some("host_memory"),
            message: message.into(),
            need: Box::new(need),
        }
    }

    /// Not enough device memory for the request (code `device_memory`), `need` the numbers.
    pub fn device_memory(message: impl Into<String>, need: MemoryNeed) -> LinusError {
        LinusError::Resource {
            code: Some("device_memory"),
            message: message.into(),
            need: Box::new(need),
        }
    }

    /// Another resource that could not be had (no shared code: the thread pool did not start).
    pub fn resource(message: impl Into<String>) -> LinusError {
        LinusError::Resource {
            code: None,
            message: message.into(),
            need: Box::default(),
        }
    }

    /// The memory a `host_memory` or `device_memory` failure needed and could use (None for
    /// every other error).
    pub fn memory_need(&self) -> Option<MemoryNeed> {
        match self {
            LinusError::Resource {
                code: Some(_),
                need,
                ..
            } => Some(**need),
            _ => None,
        }
    }
}

/// The numbers of a memory failure (qc*'s `ErrorInfo.details` of `host_memory` and
/// `device_memory`), each given when known, in GiB, so a dispatcher that resubmits with more
/// memory reads them instead of the message: `required_gib` the memory the calculation needed
/// (the memory budget, `memory_gib` or `device_memory_gib`, that holds what failed; or what had
/// been allocated plus the request that failed), `available_gib` the memory it could use (that
/// budget as it was, or what the device had).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct MemoryNeed {
    pub required_gib: Option<f64>,
    pub available_gib: Option<f64>,
}

impl MemoryNeed {
    /// Both numbers, from bytes.
    pub fn bytes(required: f64, available: f64) -> MemoryNeed {
        MemoryNeed {
            required_gib: Some(required / GIB),
            available_gib: Some(available / GIB),
        }
    }

    /// The memory needed alone (bytes), when what was usable is not known.
    pub fn required_bytes(required: f64) -> MemoryNeed {
        MemoryNeed {
            required_gib: Some(required / GIB),
            available_gib: None,
        }
    }
}

const GIB: f64 = 1073741824.0;

/// An SCF that ran out of iterations: where it stopped and the thresholds it missed (from the
/// SCF's result: `ScfResult::failure`).
#[derive(Clone, Debug)]
pub struct ScfFailure {
    pub iterations: usize,
    /// |dE| of the last iteration (Eh).
    pub de: f64,
    /// max |F_ai| of the last iteration.
    pub gmax: f64,
    /// PySCF's orbital-gradient norm of the last iteration.
    pub gnorm_pyscf: f64,
    pub convergence: Convergence,
    /// What the SCF tried before it stopped, for the message (empty: nothing): the escalation
    /// (`scf.escalate`) or why it did not run, a plateau it did not accept.
    pub notes: String,
}

impl std::fmt::Display for ScfFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let c = &self.convergence;
        let (gname, g) = match c.g_norm {
            GradNorm::Max => ("max |F_ai|", self.gmax),
            GradNorm::Pyscf => ("orbital-gradient norm", self.gnorm_pyscf),
        };
        let esc = if self.notes.is_empty() {
            String::new()
        } else {
            format!(" ({})", self.notes)
        };
        write!(
            f,
            "SCF did not converge in {} iterations{esc}: last |dE| {:.2e} Eh (e_tol {:.0e}), {gname} {g:.2e} (g_tol {:.0e}). \
             Remedies: scf.soscf = true (second-order steps near convergence), scf.level_shift (e.g. 0.5 Eh with \
             level_shift_until 1e-3), scf.damping (e.g. 0.5 with damping_until 1e-2), guess \"core\", orbitals of a \
             converged nearby calculation as the guess, or a larger max_iter",
            self.iterations, self.de, c.e_tol, c.g_tol
        )
    }
}

/// Where a calculation stopped on its caller's request ([`LinusError::Stopped`]), and what it had
/// reached beyond its SCF: the stage (`scf`, `gradient`, `hessian`, `polarizability`: a stage of
/// the calculation, or the response solve of one), the iteration of that stage after which it
/// stopped (None: before the stage started), and the gradient and Hessian it had computed.
#[derive(Clone, Debug)]
pub struct Stopped {
    pub stage: &'static str,
    pub after: Option<usize>,
    pub gradient: Option<Vec<[f64; 3]>>,
    pub hessian: Option<Vec<f64>>,
}

impl Stopped {
    /// Whether it stopped within the SCF's iterations (its last iterate is what it reached), not
    /// after a converged SCF.
    pub fn in_scf(&self) -> bool {
        self.stage == "scf" && self.after.is_some()
    }
}

impl std::fmt::Display for Stopped {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let stage = match self.stage {
            "scf" => "the SCF",
            "gradient" => "the gradient",
            "hessian" => "the Hessian",
            s => s,
        };
        match self.after {
            None => write!(f, "stopped on request before {stage}"),
            Some(n) => write!(f, "stopped on request in {stage} after iteration {n}"),
        }
    }
}

/// A CPHF/CPKS solve that stopped above its tolerance: what it was for, where it stopped and
/// the tolerance it missed.
#[derive(Clone, Debug)]
pub struct ResponseFailure {
    /// "analytic Hessian" or "polarizability".
    pub what: &'static str,
    pub iterations: usize,
    /// Max-abs residual of the last iteration.
    pub max_residual: f64,
    /// The tolerance on it (`response.tol`, `response.field_tol`).
    pub tol: f64,
}

impl std::fmt::Display for ResponseFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "the CPHF/CPKS of the {} did not converge in {} iterations: max residual {:.2e} (tolerance {:.0e}). \
             Remedies: precision.response = \"fp64\" and response.iteration_aux = \"none\" (single-precision response \
             J/K stall near a max residual of {:.0e}), a looser tolerance (the Hessian error is quadratic in the \
             residual), or a tighter SCF convergence",
            self.what,
            self.iterations,
            self.max_residual,
            self.tol,
            crate::terms::SINGLE_FLOOR
        )
    }
}

impl From<linus_core::Error> for LinusError {
    fn from(e: linus_core::Error) -> Self {
        match e {
            linus_core::Error::Input(m) => LinusError::Input(m),
        }
    }
}

impl From<linus_xc::XcError> for LinusError {
    fn from(e: linus_xc::XcError) -> Self {
        use linus_xc::XcError as X;
        match e {
            X::UnknownFunctional(name) => LinusError::Input(unknown_functional(&name)),
            X::Parse { .. } => LinusError::Input(format!(
                "{e}; dispersion goes in `dispersion` or a suffix (b3lyp-d3bj)"
            )),
            X::Order { .. } => LinusError::Unsupported(e.to_string()),
            e => LinusError::Input(e.to_string()),
        }
    }
}

impl From<linus_ints::IntsError> for LinusError {
    /// By the error's kind: a request the kernels do not cover is unsupported, a plan built
    /// from the wrong pair set is a bug.
    fn from(e: linus_ints::IntsError) -> Self {
        match e.kind() {
            linus_core::ErrorKind::Unsupported => {
                LinusError::Unsupported(format!("integrals: {e}"))
            }
            _ => LinusError::Internal(format!("integrals: {e}")),
        }
    }
}

impl From<linus_grid::GridError> for LinusError {
    fn from(e: linus_grid::GridError) -> Self {
        match e {
            linus_grid::GridError::UnsupportedElement { .. } => {
                LinusError::Unsupported(format!("grid: {e}"))
            }
            e => LinusError::Input(format!("grid: {e}")),
        }
    }
}

/// The message of an unknown functional: the closest linus names and where to look.
fn unknown_functional(name: &str) -> String {
    let names = available_functionals();
    let key = linus_xc::spec::normalize(name);
    let mut best: Vec<(usize, &str)> = names
        .iter()
        .map(|n| (edit_distance(&key, &linus_xc::spec::normalize(n)), *n))
        .filter(|(d, _)| *d <= 2)
        .collect();
    best.sort();
    let hint = match best.first() {
        Some((_, n)) => format!(" Did you mean '{n}'?"),
        None => String::new(),
    };
    format!(
        "unknown functional '{name}' (not a linus alias, libxc name or libxc id).{hint} Dispersion goes in `dispersion` or as a suffix \
         (b3lyp-d3bj); linus.available_functionals() and `linus --list-functionals` list the names"
    )
}

/// The functional names linus defines (libxc names and mixtures such as "0.5*b3lyp + 0.5*hf"
/// work too).
pub fn available_functionals() -> Vec<&'static str> {
    linus_xc::spec::ALIASES
        .iter()
        .flat_map(|a| a.names.iter().copied())
        .collect()
}
