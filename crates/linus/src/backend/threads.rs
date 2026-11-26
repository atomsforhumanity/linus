//! The session's threads when host work runs beside a device's: the `threads` setting is the
//! whole calculation's, so work overlapped with a device setup splits it instead of adding
//! threads of its own.
//!
//! - [`split_beside`]: how `threads` divide between host work beside a device setup and that
//!   setup's own host part;
//! - [`on_pool`]: run a closure on a pool of `n` threads (its rayon work stays there).
//!
//! A closure on a pool of its own must not reach back to the session's pool
//! (`CalcContext::pool`): its parallel work would then run on all of the session's threads beside
//! the other part's.

/// The share of `threads` for host work beside a device setup, and the rest for the setup's own
/// host part: (beside, setup), summing to `threads`; None below two threads, where nothing can
/// overlap within the setting (the parts then run one after another). Half each, the beside
/// part rounding down.
pub(crate) fn split_beside(threads: usize) -> Option<(usize, usize)> {
    (threads >= 2).then(|| (threads / 2, threads - threads / 2))
}

/// `f` on a new pool of `n` threads (at least one), its rayon work (and faer's `Par::rayon`)
/// there; inline on the current pool when the pool cannot be started (the same results: they do
/// not depend on the thread count). From a worker of another pool the worker waits for `f`, so
/// only `n` threads compute it.
pub(crate) fn on_pool<T: Send>(n: usize, f: impl FnOnce() -> T + Send) -> T {
    match rayon::ThreadPoolBuilder::new()
        .num_threads(n.max(1))
        .build()
    {
        Ok(pool) => pool.install(f),
        Err(_) => f(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The split keeps to the setting: the two parts sum to it, each at least one thread.
    #[test]
    fn the_split_sums_to_the_threads() {
        assert_eq!(split_beside(0), None);
        assert_eq!(split_beside(1), None);
        for t in 2..=64 {
            let (b, s) = split_beside(t).unwrap();
            assert_eq!(b + s, t);
            assert!(b >= 1 && s >= b);
        }
    }

    /// A closure on its own pool sees that pool's threads, also from inside another pool.
    #[test]
    fn on_pool_runs_on_its_threads() {
        assert_eq!(on_pool(3, rayon::current_num_threads), 3);
        let outer = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap();
        let (a, b) = outer.install(|| {
            rayon::join(
                || on_pool(1, rayon::current_num_threads),
                || on_pool(3, rayon::current_num_threads),
            )
        });
        assert_eq!((a, b), (1, 3));
    }
}
