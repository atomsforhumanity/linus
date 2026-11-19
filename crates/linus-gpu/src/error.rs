//! Errors of the GPU layer.
//!
//! Each variant is one kind of failure ([`GpuError::kind`], linus-core's [`DeviceErrorKind`]),
//! and the policies that react to failures read the kind, never the variant or the message:
//! whether a setup falls back to the CPU ([`GpuError::falls_back_at_setup`]), and which error
//! of the calculation a device error becomes (the `linus` crate's `device_error`).
//!
//! | variant | kind | when |
//! |---|---|---|
//! | `Unavailable` | unavailable | no device, driver or vendor library |
//! | `OutOfMemory` | out of memory | device memory: linus's accounting, or a library's or the driver's allocation |
//! | `HostMemory` | host memory | page-locked host memory could not be mapped or registered |
//! | `NotCovered` | not covered | a valid case the device code does not implement |
//! | `Numerical`, `NotFinite` | numerical | a factorization or eigensolver failed on its matrix; a non-finite result |
//! | `Library` | library | a vendor library's or the driver's own status: a fault or a misuse |
//! | `Compile` | compile | NVRTC rejected a kernel source |
//! | `Cache` | cache | the on-disk kernel cache could not be used |
//! | `Contract` | contract | an internal invariant of the device code was broken: a bug |
//! | `Input` | input | a user or environment value the device layer reads itself was invalid |

use linus_core::DeviceErrorKind;
use std::path::PathBuf;

#[derive(Debug, thiserror::Error)]
pub enum GpuError {
    /// No usable device, driver or runtime library (the CPU path should be used).
    #[error("no usable GPU: {0}")]
    Unavailable(String),
    /// Device memory ran out: linus's own accounting (`requested` > 0: the bytes asked for
    /// against the handle's limit), or a CUDA library's or the driver's own allocation
    /// (`requested` 0: `in_use` and `limit` are the device's used and total bytes then).
    #[error("device memory: {}", oom_text(.requested, .in_use, .limit))]
    OutOfMemory {
        requested: usize,
        in_use: usize,
        limit: usize,
    },
    /// `bytes` of page-locked host memory could not be mapped or registered with the driver
    /// (`why`: the system's or the driver's reason).
    #[error("cannot pin {bytes} bytes of host memory: {why}")]
    HostMemory { bytes: usize, why: String },
    /// A valid case the engine does not cover (an operator, an angular momentum, a precision
    /// mode): a setup falls back to the CPU on it.
    #[error("{0}")]
    NotCovered(String),
    /// A numerical failure of a library call (`op`, e.g. `cuSOLVER potrf`; `info` as the library
    /// reports it): a matrix not positive definite (potrf), singular (getrf), an eigensolver that
    /// did not converge (syevd).
    #[error("{}", numerical_text(op, *info))]
    Numerical { op: &'static str, info: i32 },
    /// A result that is not finite (the message says which).
    #[error("{0}")]
    NotFinite(String),
    /// A vendor library's or the driver's own status (not out of memory, not a numerical
    /// result): a fault or a misuse. `call` is the entry point when the caller knows it.
    #[error("{}", library_text(*lib, *call, status))]
    Library {
        lib: Lib,
        call: Option<&'static str>,
        status: String,
    },
    #[error("NVRTC could not compile {name}:\n{log}")]
    Compile { name: String, log: String },
    #[error("kernel cache {}: {msg}", path.display())]
    Cache { path: PathBuf, msg: String },
    /// An internal invariant of the device code was broken (a buffer too small for its
    /// operation, a missing setup step): a bug in linus, not in the input.
    #[error("{0}")]
    Contract(String),
    /// A user or environment value the device layer reads itself was invalid.
    #[error("input: {0}")]
    Input(String),
}

/// Set once a driver call of this process failed with a status that poisons the CUDA context.
static CONTEXT_LOST: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Records that a driver call failed with a status that poisons the CUDA context.
#[cfg(feature = "gpu")]
pub(crate) fn mark_context_lost() {
    CONTEXT_LOST.store(true, std::sync::atomic::Ordering::SeqCst);
}

/// Whether a driver call of this process failed with a status that poisons the CUDA context (an
/// illegal address, a launch failure, a device assertion, an uncorrectable ECC error, ...): no
/// device work can succeed in this process any more, so a worker answers its request with the
/// error and exits.
pub fn context_lost() -> bool {
    CONTEXT_LOST.load(std::sync::atomic::Ordering::SeqCst)
}

impl GpuError {
    /// The kind of failure (linus-core's device error kinds), which the CPU-fallback policy
    /// reads ([`DeviceErrorKind::falls_back_at_setup`]).
    pub fn kind(&self) -> DeviceErrorKind {
        match self {
            GpuError::Unavailable(_) => DeviceErrorKind::Unavailable,
            GpuError::OutOfMemory { .. } => DeviceErrorKind::OutOfMemory,
            GpuError::HostMemory { .. } => DeviceErrorKind::HostMemory,
            GpuError::NotCovered(_) => DeviceErrorKind::NotCovered,
            GpuError::Numerical { .. } | GpuError::NotFinite(_) => DeviceErrorKind::Numerical,
            GpuError::Library { .. } => DeviceErrorKind::Library,
            GpuError::Compile { .. } => DeviceErrorKind::Compile,
            GpuError::Cache { .. } => DeviceErrorKind::Cache,
            GpuError::Contract(_) => DeviceErrorKind::Contract,
            GpuError::Input(_) => DeviceErrorKind::Input,
        }
    }

    /// Whether a setup that fails with this error falls back to the CPU (the reason recorded in
    /// `resolved.backend`): out of device memory and a case the engine does not cover
    /// ([`DeviceErrorKind::falls_back_at_setup`]). Numerical failures and library faults are
    /// errors of the calculation (decision D38).
    pub fn falls_back_at_setup(&self) -> bool {
        self.kind().falls_back_at_setup()
    }
}

/// A request beyond a device [`linus_core::Budget`] is the device's out-of-memory error (the
/// same fields; `what` is the caller's to name in its context).
impl From<linus_core::OverBudget> for GpuError {
    fn from(e: linus_core::OverBudget) -> Self {
        GpuError::OutOfMemory {
            requested: e.requested,
            in_use: e.in_use,
            limit: e.limit,
        }
    }
}

/// The vendor library of a [`GpuError::Library`] status.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Lib {
    Cublas,
    Cusolver,
    Driver,
    Nvrtc,
}

impl std::fmt::Display for Lib {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Lib::Cublas => "cuBLAS",
            Lib::Cusolver => "cuSOLVER",
            Lib::Driver => "CUDA driver",
            Lib::Nvrtc => "NVRTC",
        })
    }
}

fn library_text(lib: Lib, call: Option<&str>, status: &str) -> String {
    match call {
        Some(c) => format!("{lib} {c}: {status}"),
        None => format!("{lib}: {status}"),
    }
}

/// What a nonzero `info` of the library call `op` means.
fn numerical_text(op: &str, info: i32) -> String {
    if op.ends_with("potrf") {
        format!("{op}: the leading minor of order {info} is not positive definite")
    } else if op.ends_with("getrf") {
        format!("{op}: U({info}, {info}) is exactly zero (singular matrix)")
    } else {
        format!("{op}: info = {info}")
    }
}

fn oom_text(requested: &usize, in_use: &usize, limit: &usize) -> String {
    if *requested > 0 {
        format!("{requested} bytes requested with {in_use} in use, over the limit of {limit} bytes")
    } else {
        format!(
            "a CUDA library or the driver could not allocate ({in_use} of {limit} bytes of the device in use)"
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kinds_and_fallbacks() {
        let oom = GpuError::OutOfMemory {
            requested: 8,
            in_use: 0,
            limit: 4,
        };
        assert!(oom.falls_back_at_setup());
        assert!(GpuError::NotCovered("l = 7".into()).falls_back_at_setup());
        for e in [
            GpuError::Unavailable("no driver".into()),
            GpuError::HostMemory {
                bytes: 8,
                why: "mmap".into(),
            },
            GpuError::Numerical {
                op: "cuSOLVER potrf",
                info: 3,
            },
            GpuError::NotFinite("orbitals".into()),
            GpuError::Library {
                lib: Lib::Cusolver,
                call: Some("syevd"),
                status: "CUSOLVER_STATUS_INTERNAL_ERROR".into(),
            },
            GpuError::Contract("buffer too small".into()),
            GpuError::Input("a bad argument".into()),
        ] {
            assert!(!e.falls_back_at_setup(), "{e}");
        }
        assert_eq!(
            GpuError::NotFinite("x".into()).kind(),
            DeviceErrorKind::Numerical
        );
    }

    #[test]
    fn messages() {
        let e = GpuError::Library {
            lib: Lib::Driver,
            call: None,
            status: "CUDA_ERROR_ILLEGAL_ADDRESS".into(),
        };
        assert_eq!(e.to_string(), "CUDA driver: CUDA_ERROR_ILLEGAL_ADDRESS");
        let e = GpuError::Library {
            lib: Lib::Cusolver,
            call: Some("potrf"),
            status: "CUSOLVER_STATUS_INVALID_VALUE".into(),
        };
        assert_eq!(
            e.to_string(),
            "cuSOLVER potrf: CUSOLVER_STATUS_INVALID_VALUE"
        );
        assert_eq!(GpuError::Input("x".into()).to_string(), "input: x");
    }
}
