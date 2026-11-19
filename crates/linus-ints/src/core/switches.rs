//! Environment switches of the integral engine, read the same way everywhere.
//!
//! - `LINUS_ISA` (`generic`, `avx2`, `avx512`) lowers the instruction set of the dispatched
//!   kernels in every build ([`linus_num::simd::Isa::detect`]).
//! - The study switches `LINUS_KERNEL` (`loop`, `tiled`), `LINUS_NO_GEN` (set: no generated
//!   kernels) and `LINUS_ECP_LEVEL` (3-11: the first ECP quadrature level) only act in builds with
//!   the cargo feature `dev-switches` (off by default); otherwise they are ignored with a warning.
//!
//! Invalid values are ignored with one warning on stderr per variable and process:
//! `linus: ignoring NAME="value" (...)`.

use std::collections::HashSet;
use std::sync::{Mutex, OnceLock};

/// Print `msg` on stderr once per process for `name`.
pub fn warn_once(name: &str, msg: &str) {
    static SEEN: OnceLock<Mutex<HashSet<String>>> = OnceLock::new();
    let mut seen = SEEN
        .get_or_init(|| Mutex::new(HashSet::new()))
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    if seen.insert(name.to_string()) {
        eprintln!("{msg}");
    }
}

/// The warning for an invalid value of `name`: `linus: ignoring NAME="value" (expected)`.
pub fn ignore_invalid(name: &str, value: &str, expected: &str) {
    warn_once(
        name,
        &format!("linus: ignoring {name}={value:?} ({expected})"),
    );
}

/// The value of the study switch `name` in development builds (cargo feature `dev-switches`);
/// None if it is unset, or (with a warning) in other builds.
pub fn study_switch(name: &str) -> Option<String> {
    let v = std::env::var(name).ok()?;
    if cfg!(feature = "dev-switches") {
        Some(v)
    } else {
        ignore_invalid(
            name,
            &v,
            "a study switch of development builds: build with the cargo feature dev-switches",
        );
        None
    }
}

/// How the batched ERI kernels are chosen (`LINUS_KERNEL`, a study switch).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KernelChoice {
    /// Generated kernels, else tiled, else generic loops (the default).
    Default,
    /// Generated kernels, else generic loops (`loop`).
    Loop,
    /// Tiled kernels wherever they apply (`tiled`).
    Tiled,
}

/// `LINUS_KERNEL`, read once.
pub fn kernel_choice() -> KernelChoice {
    static K: OnceLock<KernelChoice> = OnceLock::new();
    *K.get_or_init(|| match study_switch("LINUS_KERNEL").as_deref() {
        None | Some("") => KernelChoice::Default,
        Some(v) => match v.trim().to_ascii_lowercase().as_str() {
            "loop" => KernelChoice::Loop,
            "tiled" => KernelChoice::Tiled,
            _ => {
                ignore_invalid("LINUS_KERNEL", v, "use loop or tiled");
                KernelChoice::Default
            }
        },
    })
}

/// `LINUS_NO_GEN` (set: no generated kernels), read once.
pub fn no_generated_kernels() -> bool {
    static G: OnceLock<bool> = OnceLock::new();
    *G.get_or_init(|| study_switch("LINUS_NO_GEN").is_some())
}
