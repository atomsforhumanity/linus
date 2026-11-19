//! `LINUS_SCF_TRACE`: the SCF loops' internal state per iteration as JSON lines, a debugging aid
//! beside `scf.history` (which records the iterations' energies, gradients and steps). With the
//! variable set to a file, the host loop appends for each iteration the DIIS subspace size and
//! drop rule, the ADIIS history size and the SOSCF history size behind its step, and the device
//! loop the DIIS subspace size and whether the level shift was on. Read once per process; without
//! it each iteration costs one check of a cached `None`, and nothing is written.

use super::settings::DiisDrop;
use super::ScfIteration;
use std::io::Write;
use std::sync::{Mutex, OnceLock};

static SINK: OnceLock<Option<Mutex<std::fs::File>>> = OnceLock::new();

/// The trace file, if `LINUS_SCF_TRACE` names one (opened for appending on first use).
fn sink() -> Option<&'static Mutex<std::fs::File>> {
    SINK.get_or_init(|| {
        let path = std::env::var_os("LINUS_SCF_TRACE").filter(|v| !v.is_empty())?;
        std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .ok()
            .map(Mutex::new)
    })
    .as_ref()
}

pub fn enabled() -> bool {
    sink().is_some()
}

pub fn write(line: &str) {
    if let Some(m) = sink() {
        let mut f = m.lock().unwrap_or_else(|e| e.into_inner());
        // best effort: a trace that cannot be written changes nothing else
        let _ = writeln!(f, "{line}");
    }
}

/// The host loop's state behind each history entry from `from` on (normally the one iteration
/// since the last call): returns the new count of traced entries.
pub(super) fn host(
    history: &[ScfIteration],
    from: usize,
    diis_size: usize,
    drop: DiisDrop,
    adiis_size: usize,
    soscf_size: usize,
) -> usize {
    let drop = match drop {
        DiisDrop::Oldest => "oldest",
        DiisDrop::LargestError => "largest_error",
    };
    for h in history.iter().skip(from) {
        write(&format!(
            "{{\"loop\": \"host\", \"iter\": {}, \"step\": \"{}\", \"diis_size\": {diis_size}, \
             \"diis_drop\": \"{drop}\", \"adiis_size\": {adiis_size}, \"soscf_size\": {soscf_size}}}",
            h.iter, h.step
        ));
    }
    history.len().max(from)
}
