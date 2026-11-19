//! Phase profiler for the XC quadrature (off by default; enabled by [`enable`]).
//!
//! Each worker marks phase boundaries with [`Lap::lap`]; the time since the previous mark is
//! added to that phase's counter (thread-seconds, summed over workers). Reports are thread
//! time, so on n threads a phase's wall share is roughly its fraction of the total.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Instant;

/// Phase names, by index.
pub const PHASES: [&str; 12] = [
    "setup",       // per-batch shell/function lists, scratch
    "collocation", // basis functions (and derivatives) on the batch points
    "screening",   // function screening and compression
    "density",     // density-matrix gather, X = phi D, rho/grad/tau
    "libxc",       // functional evaluation
    "weights",     // per-point potential (W, t, u) from the functional derivatives
    "gemm_v",      // potential-matrix products
    "scatter",     // accumulation into the global matrices
    "reduce",      // final reductions and symmetrization
    "grid_resp",   // grid-weight derivatives (gradient)
    "gemm_d",      // perturbed-density products (kernel)
    "other",
];

pub const SETUP: usize = 0;
pub const COLLOC: usize = 1;
pub const SCREEN: usize = 2;
pub const DENSITY: usize = 3;
pub const LIBXC: usize = 4;
pub const WEIGHTS: usize = 5;
pub const GEMM_V: usize = 6;
pub const SCATTER: usize = 7;
pub const REDUCE: usize = 8;
pub const GRID_RESP: usize = 9;
pub const GEMM_D: usize = 10;
pub const OTHER: usize = 11;

static ON: AtomicBool = AtomicBool::new(false);
static NS: [AtomicU64; 12] = [const { AtomicU64::new(0) }; 12];

pub fn enable(on: bool) {
    ON.store(on, Ordering::Relaxed);
}

pub fn enabled() -> bool {
    ON.load(Ordering::Relaxed)
}

pub fn reset() {
    for c in NS.iter().chain(&STATS) {
        c.store(0, Ordering::Relaxed);
    }
}

/// Batch statistics: [batches, sum np, sum np nao, sum np nao^2, sum nao, sum candidate
/// functions, sum evaluated functions] (when enabled).
static STATS: [AtomicU64; 7] = [const { AtomicU64::new(0) }; 7];

/// Record one batch's size: np points, nao functions kept, ncand candidates (sphere test),
/// neval evaluated (box test).
#[inline]
pub fn batch(np: usize, nao: usize, ncand: usize, neval: usize) {
    if enabled() {
        let v = [1, np, np * nao, np * nao * nao, nao, ncand, neval];
        for (c, x) in STATS.iter().zip(v) {
            c.fetch_add(x as u64, Ordering::Relaxed);
        }
    }
}

/// (batches, mean np, mean nao, np-weighted mean nao, sqrt of np-weighted mean nao^2,
/// mean candidates, mean evaluated).
pub fn batch_stats() -> [f64; 7] {
    let v: Vec<f64> = STATS
        .iter()
        .map(|c| c.load(Ordering::Relaxed) as f64)
        .collect();
    let nb = v[0].max(1.0);
    [
        v[0],
        v[1] / nb,
        v[4] / nb,
        v[2] / v[1].max(1.0),
        (v[3] / v[1].max(1.0)).sqrt(),
        v[5] / nb,
        v[6] / nb,
    ]
}

/// (phase, thread-seconds) for phases with nonzero time.
pub fn report() -> Vec<(&'static str, f64)> {
    PHASES
        .iter()
        .zip(&NS)
        .map(|(n, c)| (*n, c.load(Ordering::Relaxed) as f64 * 1e-9))
        .filter(|x| x.1 > 0.0)
        .collect()
}

/// Phase stopwatch of one worker.
pub struct Lap(Option<Instant>);

impl Lap {
    #[inline]
    pub fn start() -> Self {
        Lap(if enabled() {
            Some(Instant::now())
        } else {
            None
        })
    }

    /// Charge the time since the last mark to `phase`.
    #[inline]
    pub fn lap(&mut self, phase: usize) {
        if let Some(t) = self.0 {
            let now = Instant::now();
            NS[phase].fetch_add((now - t).as_nanos() as u64, Ordering::Relaxed);
            self.0 = Some(now);
        }
    }
}
