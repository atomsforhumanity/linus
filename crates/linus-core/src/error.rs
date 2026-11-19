//! The error model: the kinds of failure, one vocabulary for every layer and front end.
//!
//! [`ErrorKind`] classifies every error a calculation can end with. The driver's error type
//! (`linus::LinusError`) carries the details of each kind, and the front ends derive their
//! behaviour from the kind alone: the CLI's `error_kind` and exit code (2 for input and basis
//! set errors, 1 otherwise), the Python exception class, the qcdata result's extras
//! (docs/conventions.md, "Errors").
//!
//! [`Error`] is the error of this crate's checked constructors ([`crate::Structure::new`]).
//!
//! [`DeviceErrorKind`] classifies the failures of the GPU layer, so the policies that react to
//! them are written over typed kinds in one place rather than over a device type's variant
//! names: fall back to the CPU at setup or stop ([`DeviceErrorKind::falls_back_at_setup`]), and
//! the calculation's error kind when it stops ([`DeviceErrorKind::calculation_kind`]).

use std::fmt;

/// The kind of a failed calculation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ErrorKind {
    /// Settings or structure the calculation cannot run as given (fix the input).
    Input,
    /// A basis set file that cannot be read or parsed.
    Basis,
    /// The SCF or a CPHF/CPKS solve did not converge.
    Convergence,
    /// A valid request for something linus does not implement.
    Unsupported,
    /// Not enough memory (or another resource) for the request as set up.
    Resource,
    /// The GPU backend failed (no usable device for backend `gpu`, or a device error).
    Backend,
    /// A numerical failure (non-finite energies, singular matrices).
    Numerical,
    /// A broken internal invariant: a bug in linus.
    Internal,
    /// The caller asked the calculation to stop (an in-process caller's stop request, qc*'s
    /// `ctx.stop_requested()`), and it ended at its next step with what it had reached.
    Stopped,
}

impl ErrorKind {
    /// The short name the front ends report: input, basis, convergence, unsupported,
    /// resource, backend, numerical, internal, stopped.
    pub const fn name(self) -> &'static str {
        match self {
            ErrorKind::Input => "input",
            ErrorKind::Basis => "basis",
            ErrorKind::Convergence => "convergence",
            ErrorKind::Unsupported => "unsupported",
            ErrorKind::Resource => "resource",
            ErrorKind::Backend => "backend",
            ErrorKind::Numerical => "numerical",
            ErrorKind::Internal => "internal",
            ErrorKind::Stopped => "stopped",
        }
    }

    /// Errors the user fixes in the input (CLI exit code 2): input and basis set errors.
    pub const fn is_input(self) -> bool {
        matches!(self, ErrorKind::Input | ErrorKind::Basis)
    }
}

impl fmt::Display for ErrorKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// An error of this crate's checked constructors (a structure that cannot exist). Each
/// variant is a kind of [`ErrorKind`]; the driver's error converts it to its variant of the
/// same kind.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Error {
    /// Input that cannot be used as given (the message says what to fix).
    Input(String),
}

impl Error {
    pub fn kind(&self) -> ErrorKind {
        match self {
            Error::Input(_) => ErrorKind::Input,
        }
    }

    /// An error the user fixes in the input.
    pub fn is_input(&self) -> bool {
        self.kind().is_input()
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Input(m) => f.write_str(m),
        }
    }
}

impl std::error::Error for Error {}

/// The kind of a GPU-layer failure.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DeviceErrorKind {
    /// No usable device, driver or runtime library.
    Unavailable,
    /// Device memory ran out (linus's accounting, or a library's or the driver's allocation).
    OutOfMemory,
    /// Page-locked host memory could not be mapped or registered with the driver.
    HostMemory,
    /// A valid case the device code does not cover (an operator, an angular momentum).
    NotCovered,
    /// A numerical failure (a matrix not positive definite or singular, an eigensolver that
    /// did not converge, a non-finite result): the calculation's numerical error.
    Numerical,
    /// A vendor library's or the driver's own status (not out of memory): a fault or a misuse.
    Library,
    /// A kernel did not compile.
    Compile,
    /// The on-disk kernel cache could not be used.
    Cache,
    /// An internal invariant of the device code was broken: a bug in linus.
    Contract,
    /// An invalid user or environment value reached the device layer.
    Input,
}

impl DeviceErrorKind {
    /// Whether a device setup that failed with this kind runs on the CPU instead (recorded
    /// with its reason in `resolved.backend`) rather than stopping the calculation: out of
    /// device memory, and cases the device code does not cover. A device error after the
    /// setup always stops the calculation (it never switches to the CPU midway).
    pub const fn falls_back_at_setup(self) -> bool {
        matches!(
            self,
            DeviceErrorKind::OutOfMemory | DeviceErrorKind::NotCovered
        )
    }

    /// The kind of the calculation's error when a device error stops it: a numerical failure
    /// is [`ErrorKind::Numerical`] (a CPU run meets it too), a broken invariant
    /// [`ErrorKind::Internal`] (a bug), out of device or page-locked host memory
    /// [`ErrorKind::Resource`] (qcdata's codes `device_memory` and `host_memory`, decision 23),
    /// every other one [`ErrorKind::Backend`].
    pub const fn calculation_kind(self) -> ErrorKind {
        match self {
            DeviceErrorKind::Numerical => ErrorKind::Numerical,
            DeviceErrorKind::Contract => ErrorKind::Internal,
            DeviceErrorKind::OutOfMemory | DeviceErrorKind::HostMemory => ErrorKind::Resource,
            _ => ErrorKind::Backend,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kinds() {
        assert_eq!(ErrorKind::Convergence.to_string(), "convergence");
        assert!(ErrorKind::Basis.is_input() && !ErrorKind::Resource.is_input());
        assert!(DeviceErrorKind::OutOfMemory.falls_back_at_setup());
        assert!(DeviceErrorKind::NotCovered.falls_back_at_setup());
        assert!(!DeviceErrorKind::HostMemory.falls_back_at_setup());
        // decision D38: numerical failures and library faults are the calculation's errors
        assert!(!DeviceErrorKind::Numerical.falls_back_at_setup());
        assert!(!DeviceErrorKind::Library.falls_back_at_setup());
        assert!(!DeviceErrorKind::Contract.falls_back_at_setup());
        assert_eq!(
            DeviceErrorKind::Numerical.calculation_kind(),
            ErrorKind::Numerical
        );
        assert_eq!(
            DeviceErrorKind::Contract.calculation_kind(),
            ErrorKind::Internal
        );
        assert_eq!(
            DeviceErrorKind::Library.calculation_kind(),
            ErrorKind::Backend
        );
        // out of memory is a resource error (decision 23's device_memory, host_memory)
        assert_eq!(
            DeviceErrorKind::OutOfMemory.calculation_kind(),
            ErrorKind::Resource
        );
        assert_eq!(
            DeviceErrorKind::HostMemory.calculation_kind(),
            ErrorKind::Resource
        );
    }
}
