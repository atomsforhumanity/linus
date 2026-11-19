//! The base of linus's method crates: what the J/K (linus-jk), XC (linus-dft), SCF and response
//! (linus-scf), solvent (linus-solvent) and vibrational (linus-vib) crates and the facade (linus)
//! share, and nothing they own:
//!
//! - the errors of a calculation ([`error`]: `LinusError`, the SCF's and the responses' failures,
//!   the did-you-mean hints) and the SCF's convergence thresholds they report ([`convergence`]);
//! - dense linear algebra ([`linalg`]), the only route to faer's products, solves and
//!   factorizations (thread-count determinism; clippy.toml);
//! - memory limits and probes ([`memory`]);
//! - a calculation's progress lines, to stderr or collected for the output's logs ([`progress`]);
//! - the method families' terms ([`terms`]): the traits through which the SCF and the response
//!   call the J/K, XC and environment families, so that the algorithms and the families are
//!   separate crates that do not know each other;
//! - lazily built shared data ([`get_or_build`]).
//!
//! Thread-count independent sums are linus-num's `det`; timers and nuclear gradients as sums of
//! parts are linus-core's `timing` and `gradient`.

mod basis;
pub mod convergence;
pub mod error;
pub mod linalg;
pub mod memory;
pub mod progress;
pub mod symmetry;
pub mod terms;

use std::sync::OnceLock;

/// The value in `cell`, built by `build` on first use.
///
/// Use this instead of [`OnceLock::get_or_init`] whenever `build` runs rayon work. A rayon worker
/// that waits for its own parallel jobs inside `get_or_init` steals other jobs meanwhile; if one
/// of them asks for the same cell (a per-batch loop that calls an accessor per batch), it waits
/// on the cell forever (a re-entrant initialization), and the other workers that ask for it wait
/// with it: the whole pool deadlocks. Here the build runs outside the cell. Concurrent first
/// callers may each build the value (it is deterministic); the first `set` wins.
pub fn get_or_build<T>(cell: &OnceLock<T>, build: impl FnOnce() -> T) -> &T {
    if let Some(v) = cell.get() {
        return v;
    }
    let v = build();
    // a concurrent caller may have set it first: its value is the same
    let _ = cell.set(v);
    cell.get().expect("the cell was set above")
}

#[cfg(test)]
mod tests {
    use super::get_or_build;
    use rayon::prelude::*;
    use std::sync::OnceLock;

    /// A rayon build of a cell that every job of a parallel loop asks for on first use. With
    /// `get_or_init` in place of `get_or_build` this deadlocks within a few hundred rounds on 16
    /// threads (a standalone copy of this loop: every thread in a futex wait).
    #[test]
    fn rayon_build_asked_for_by_every_job_of_a_parallel_loop() {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(16)
            .build()
            .unwrap();
        pool.install(|| {
            for _ in 0..400 {
                let cell: OnceLock<Vec<u64>> = OnceLock::new();
                let build =
                    || -> Vec<u64> { (0..200_000u64).into_par_iter().map(|i| i % 1013).collect() };
                let s: u64 = (0..4096usize)
                    .into_par_iter()
                    .map(|i| get_or_build(&cell, build)[i * 37])
                    .sum();
                assert_eq!(s, (0..4096u64).map(|i| (i * 37) % 1013).sum::<u64>());
            }
        });
    }
}
