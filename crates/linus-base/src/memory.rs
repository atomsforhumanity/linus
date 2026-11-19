//! Memory: the limit every memory budget derives from ([`MemoryLimit`]: `memory_gib`, else the
//! machine), the budget of the SCF's density-fitting tensors ([`MemoryLimit::df_budget`]: less
//! the [`Reservation`]s of the other large allocations alive with them), what a phase of a
//! calculation holds at its peak ([`PhaseNeed`], from the process's baseline ([`Process`]) and
//! the parts its engines size), the probes of the machine (/proc/meminfo, the cgroup limit and
//! headroom) and peak-memory marks in the timers (`mem_mark`).
//!
//! `memory_gib` is a limit on the whole process: the resident memory of a calculation's process
//! stays within it. Each phase of a calculation (the SCF, the gradient, the Hessian, the
//! polarizability) holds the process's baseline, the large arrays and workspaces its engines
//! size from the problem alone (never from the limit: their block partitions, and with them the
//! rounding, do not follow `memory_gib`, decision D43) and the matrices, XC buffers and allocator
//! slack of a measured model; a phase whose need is above the limit fails before any work,
//! naming the `memory_gib` that holds it, and the plan states the largest need of the
//! calculation's phases (`memory.required_gib`).

use linus_core::timing::Timers;

/// Bytes per GB of `memory_gib` (GiB).
const GIB: f64 = 1073741824.0;

/// The kind of process a calculation runs in, for its baseline: the resident memory the process
/// holds before a calculation's arrays (the program, its libraries, the basis set library, the
/// allocator's arenas), which every phase's need starts from ([`PhaseNeed`]). The front end
/// declares it (`linus::Session::set_process`); a device adds [`DEVICE_HOST_BYTES`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Process {
    /// The `linus` command-line program (measured: 0.02 GiB with a small calculation).
    #[default]
    Cli,
    /// A Python process with linus, qcdata, numpy and qccompute imported (measured: 0.056 GiB
    /// after the imports, 0.065 after a calculation; a qccompute worker 0.066).
    Python,
}

impl Process {
    /// The process's baseline (bytes): the measurements above, rounded up.
    pub fn baseline_bytes(self) -> usize {
        match self {
            Process::Cli => 32 << 20,
            Process::Python => 96 << 20,
        }
    }

    /// Its name in a phase's parts (`process (cli)`, `process (python)`).
    pub fn name(self) -> &'static str {
        match self {
            Process::Cli => "process (cli)",
            Process::Python => "process (python)",
        }
    }
}

/// The host memory a process takes beside its baseline once it has opened a device: the CUDA
/// context, the driver's and the libraries' (cuBLAS, cuSOLVER, NVRTC) host allocations.
pub const DEVICE_HOST_BYTES: usize = 640 << 20;

/// One part of a phase's need: what it is and its bytes (serialized in GiB as `gib`).
#[derive(Clone, Debug, PartialEq, serde::Serialize)]
pub struct Part {
    pub what: &'static str,
    #[serde(rename = "gib", serialize_with = "ser_gib")]
    pub bytes: usize,
}

/// What one phase of a calculation holds at its peak: the process's baseline, the large arrays
/// and workspaces its engines size from the problem, and the measured model of the rest (see the
/// module docs). Its total is the smallest `memory_gib` the phase runs in
/// ([`PhaseNeed::required`]), unless a rule of its own asks for more (the DF tensors' 75%).
#[derive(Clone, Debug, PartialEq, serde::Serialize)]
pub struct PhaseNeed {
    /// `scf`, `gradient`, `hessian` or `polarizability`.
    pub phase: &'static str,
    pub parts: Vec<Part>,
    /// The smallest `memory_gib` the phase runs in (bytes; serialized in GiB as
    /// `required_gib`): the parts' sum, or more where a budget rule asks for it.
    #[serde(rename = "required_gib", serialize_with = "ser_gib")]
    pub required: usize,
}

impl PhaseNeed {
    /// A phase's need from its parts; it requires their sum.
    pub fn new(phase: &'static str, parts: Vec<Part>) -> Self {
        let required = parts.iter().fold(0usize, |s, p| s.saturating_add(p.bytes));
        PhaseNeed {
            phase,
            parts,
            required,
        }
    }

    /// The smallest `memory_gib` (bytes) the phase runs in.
    pub fn required(&self) -> usize {
        self.required
    }

    /// The parts as text, largest first ("1.03 GiB DF tensor, 0.38 GiB exchange blocks, ...").
    pub fn describe(&self) -> String {
        let mut parts: Vec<&Part> = self.parts.iter().filter(|p| p.bytes > 0).collect();
        parts.sort_by_key(|p| std::cmp::Reverse(p.bytes));
        parts
            .iter()
            .map(|p| format!("{:.2} GiB {}", p.bytes as f64 / GIB, p.what))
            .collect::<Vec<_>>()
            .join(", ")
    }
}

/// `bytes` as the `memory_gib` that holds them, rounded up to 0.1 GiB (the value a failure and
/// the plan name).
pub fn gib_holding(bytes: usize) -> f64 {
    (bytes as f64 / GIB * 10.0 - 1e-9).ceil() / 10.0
}

/// The DF tensors' budget when the machine's memory is unknown (no /proc/meminfo, no cgroup).
const UNKNOWN_DF_BUDGET: usize = 8 << 30;

/// What the memory budgets of a calculation derive from: `memory_gib` when set, else the
/// machine ([`MemoryLimit::machine`]: total RAM, or the cgroup memory limit of this process if
/// lower). Both are fixed for a process, never the momentary MemAvailable or resident set, so
/// the budgets (and with them storage formats and block sizes) do not depend on how busy the
/// node is. The one exception is the guard of the DF tensors' budget with `memory_gib` unset
/// ([`MemoryLimit::df_budget`]): it keeps the tensors within the memory actually free when the
/// SCF starts, binds only on a node too full for the limit's budget, and is reported then.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MemoryLimit {
    /// `memory_gib` in bytes (None: from the machine).
    pub limit: Option<usize>,
}

/// A large allocation whose size is known before the SCF's density-fitting setup sizes its
/// tensors and which is alive together with them (the PCM's dense surface matrices while they
/// are built). The DF budget leaves room for it ([`MemoryLimit::df_budget`]); the session
/// takes its reservations before each SCF and reports them in `resolved.record.memory`.
#[derive(Clone, Debug, PartialEq, serde::Serialize)]
pub struct Reservation {
    /// What the memory is for ("pcm").
    pub what: &'static str,
    /// Bytes (serialized in GiB as `gib`).
    #[serde(rename = "gib", serialize_with = "ser_gib")]
    pub bytes: usize,
}

fn ser_gib<S: serde::Serializer>(b: &usize, s: S) -> Result<S::Ok, S::Error> {
    s.serialize_f64(*b as f64 / GIB)
}

/// The budget of the SCF's 3-index DF tensors and what set it ([`MemoryLimit::df_budget`]).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DfBudget {
    /// Bytes the tensors may take.
    pub bytes: usize,
    /// The guard of an unset `memory_gib` set the budget (the memory free when it was taken was
    /// smaller than the limit's budget).
    pub by_available: bool,
    /// The memory free for this process when the budget was taken (bytes; [`mem_available`]);
    /// read only with `memory_gib` unset.
    pub available: Option<usize>,
}

impl DfBudget {
    /// The DF budget for a limit: 75% of `limit` (`memory_gib`), else of `machine` (8 GiB
    /// in all if unknown), less the `reserved` bytes; with `limit` unset also at most 90% of
    /// `available` less `reserved` (the guard). Pure: the probes are the caller's.
    pub fn of(
        limit: Option<usize>,
        machine: Option<usize>,
        available: Option<usize>,
        reserved: usize,
    ) -> DfBudget {
        let det = match limit.or(machine) {
            Some(m) => (m / 4 * 3).saturating_sub(reserved),
            None => UNKNOWN_DF_BUDGET.saturating_sub(reserved),
        };
        let available = if limit.is_some() { None } else { available };
        match available.map(|a| (a / 10 * 9).saturating_sub(reserved)) {
            Some(g) if g < det => DfBudget {
                bytes: g,
                by_available: true,
                available,
            },
            _ => DfBudget {
                bytes: det,
                by_available: false,
                available,
            },
        }
    }
}

impl MemoryLimit {
    pub fn from_gib(gib: Option<f64>) -> Self {
        MemoryLimit {
            limit: gib.map(|g| (g * GIB) as usize),
        }
    }

    /// The machine's memory when `memory_gib` is unset and where it comes from: the smaller of
    /// the total RAM (MemTotal) and the cgroup v2 `memory.max` of this process (a systemd scope's
    /// MemoryMax), as (bytes, "total RAM" | "cgroup limit"); (None, "unknown") if neither is
    /// readable. The one place that reads the machine's limit.
    pub fn machine() -> (Option<usize>, &'static str) {
        match (mem_total(), cgroup_memory_max()) {
            (Some(t), Some(c)) if c < t => (Some(c), "cgroup limit"),
            (Some(t), _) => (Some(t), "total RAM"),
            (None, Some(c)) => (Some(c), "cgroup limit"),
            (None, None) => (None, "unknown"),
        }
    }

    /// The memory a calculation may hold now, which its phases' needs are checked against
    /// before any work, with where it comes from: `memory_gib`, else what an unset one gives
    /// ([`run_limit`]: the machine's limit within 90% of the memory free now); None if unknown.
    /// Only checked against: it chooses no storage and no block.
    pub fn run_limit(&self) -> Option<(usize, &'static str)> {
        match self.limit {
            Some(l) => Some((l, "memory_gib")),
            None => run_limit(MemoryLimit::machine().0, mem_available()).map(|l| {
                (
                    l,
                    "memory available (the machine's limit, at most 90% of the free memory)",
                )
            }),
        }
    }

    /// Where the budgets come from: "memory_gb", "cgroup limit", "total RAM" or "unknown".
    pub fn source(&self) -> &'static str {
        match self.limit {
            Some(_) => "memory_gib",
            None => MemoryLimit::machine().1,
        }
    }

    /// The memory the budgets derive from (bytes): `memory_gib`, else [`MemoryLimit::machine`].
    pub fn bytes(&self) -> Option<usize> {
        self.limit.or_else(|| MemoryLimit::machine().0)
    }

    /// The budget of the SCF's 3-index DF tensors: 75% of the limit (`memory_gib`, else the
    /// machine's, [`MemoryLimit::machine`]; 8 GiB in all if unknown) less `reserved`, the bytes
    /// of the other large allocations alive with them ([`Reservation`]). With `memory_gib` unset
    /// it also stays within the memory free now: at most 90% of [`mem_available`] less
    /// `reserved` (the guard: on a node too full for 75% of its limit, a more compact storage
    /// instead of an out-of-memory kill; `by_available` says so). The guard depends on the
    /// node's load, so it binds only when the node lacks the memory, and `memory_gib` turns it
    /// off (the same budget, and storage, on every machine).
    pub fn df_budget(&self, reserved: usize) -> DfBudget {
        match self.limit {
            Some(l) => DfBudget::of(Some(l), None, None, reserved),
            None => DfBudget::of(None, MemoryLimit::machine().0, mem_available(), reserved),
        }
    }

    /// Bytes for the 3-index DF tensors of the SCF with nothing reserved
    /// ([`MemoryLimit::df_budget`]): 75% of `memory_gib`, else of the machine's memory and
    /// within the free memory.
    pub fn df_tensor_bytes(&self) -> usize {
        self.df_budget(0).bytes
    }

    /// MB for the H-bar atom blocks of RI Hessians, for a wish of `want` MB: unset: `want`, at
    /// least 1500 MB, at most a quarter of the machine's memory ([`MemoryLimit::machine`]); with
    /// `memory_gib`: `want` (at least 1500 MB), at most a quarter of `memory_gib`.
    pub fn hbar_mb(&self, want: f64) -> f64 {
        match (self.limit, MemoryLimit::machine().0) {
            (Some(l), _) => want.max(1500.0).min(l as f64 / 4.0 / 1e6),
            (None, Some(m)) => want.min(m as f64 / 4.0 / 1e6).max(1500.0),
            (None, None) => want.max(1500.0),
        }
    }

    /// MB for the per-thread accumulators of an XC-kernel contraction: 1024; with `memory_gib`,
    /// at most an eighth of it.
    pub fn xc_kernel_mb(&self) -> usize {
        match self.limit {
            Some(l) => ((l / 8) >> 20).clamp(1, 1024),
            None => 1024,
        }
    }

    /// Bytes of page-locked host memory for the part of a GPU response's stored exchange
    /// operator that does not fit the device (streamed to it in every product): 60% of
    /// `memory_gib`, else of the machine's memory ([`MemoryLimit::machine`]; none if unknown).
    pub fn response_operator_host_bytes(&self) -> usize {
        self.bytes().map_or(0, |b| b / 5 * 3)
    }

    /// Bytes for the per-thread buffers of the RI response exchange (term 1): 1 GiB; with
    /// `memory_gib`, at most an eighth of it.
    pub fn response_slot_bytes(&self) -> usize {
        let full = 1usize << 30;
        match self.limit {
            Some(l) => (l / 8).clamp(1, full),
            None => full,
        }
    }
}

/// The memory an unset `memory_gib` lets a calculation hold now: the machine's limit (`machine`:
/// total RAM or the cgroup limit, [`MemoryLimit::machine`]) and at most 90% of the memory free
/// (`available`: [`mem_available`], within a capped cgroup's headroom), as the DF tensors' guard
/// takes it; None when neither is known. Pure: the probes are the caller's.
pub fn run_limit(machine: Option<usize>, available: Option<usize>) -> Option<usize> {
    let free = available.map(|a| a / 10 * 9);
    match (machine, free) {
        (Some(m), Some(f)) => Some(m.min(f)),
        (m, f) => m.or(f),
    }
}

/// MemTotal from /proc/meminfo (bytes), if readable.
pub fn mem_total() -> Option<usize> {
    meminfo_kb("MemTotal:").map(|kb| kb * 1024)
}

/// The memory free for this process now (bytes): MemAvailable from /proc/meminfo, and in a
/// memory-capped cgroup at most its headroom ([`cgroup_headroom`]); None if neither is readable.
pub fn mem_available() -> Option<usize> {
    let avail = meminfo_kb("MemAvailable:").map(|kb| kb * 1024);
    match (avail, cgroup_headroom()) {
        (Some(a), Some(h)) => Some(a.min(h)),
        (a, h) => a.or(h),
    }
}

/// The smallest headroom of this process's capped cgroups (it and its ancestors with a numeric
/// `memory.max`): memory.max - (memory.current - inactive_file), the inactive page cache counted
/// as free (the kernel reclaims it first); None without a capped cgroup.
pub fn cgroup_headroom() -> Option<usize> {
    let mut best: Option<usize> = None;
    for dir in cgroup_dirs()? {
        let (Some(max), Some(cur)) = (
            read_usize(&dir.join("memory.max")),
            read_usize(&dir.join("memory.current")),
        ) else {
            continue;
        };
        let inactive = std::fs::read_to_string(dir.join("memory.stat"))
            .ok()
            .and_then(|s| {
                s.lines()
                    .find_map(|l| l.strip_prefix("inactive_file "))
                    .and_then(|v| v.trim().parse::<usize>().ok())
            })
            .unwrap_or(0);
        let h = max.saturating_sub(cur.saturating_sub(inactive));
        best = Some(best.map_or(h, |b| b.min(h)));
    }
    best
}

/// Smallest cgroup v2 `memory.max` of this process's cgroup and its ancestors (bytes), if any.
pub fn cgroup_memory_max() -> Option<usize> {
    cgroup_dirs()?
        .iter()
        .filter_map(|d| read_usize(&d.join("memory.max")))
        .min()
}

/// This process's cgroup v2 directory and its ancestors below /sys/fs/cgroup, innermost first
/// (in a container's cgroup namespace: /sys/fs/cgroup itself).
fn cgroup_dirs() -> Option<Vec<std::path::PathBuf>> {
    let cg = std::fs::read_to_string("/proc/self/cgroup").ok()?;
    let path = cg
        .lines()
        .find_map(|l| l.strip_prefix("0::"))?
        .trim()
        .to_string();
    let mut out = vec![];
    let mut p = std::path::PathBuf::from(format!("/sys/fs/cgroup{path}"));
    loop {
        out.push(p.clone());
        if !p.pop() || p.as_os_str().len() <= "/sys/fs/cgroup".len() {
            break;
        }
    }
    Some(out)
}

/// The number in a cgroup file (None for "max" and unreadable files).
fn read_usize(p: &std::path::Path) -> Option<usize> {
    std::fs::read_to_string(p).ok()?.trim().parse().ok()
}

/// A /proc/meminfo field (`key` with its colon) in kB, if readable.
fn meminfo_kb(key: &str) -> Option<usize> {
    let s = std::fs::read_to_string("/proc/meminfo").ok()?;
    s.lines()
        .find_map(|l| l.strip_prefix(key))?
        .trim()
        .trim_end_matches("kB")
        .trim()
        .parse()
        .ok()
}

/// Record the process's peak resident memory so far (VmHWM, GB) as `mem_hwm_gb_<name>` in the
/// timers (which phase sets a Hessian's memory peak).
pub fn mem_mark(timers: &mut Timers, name: &str) {
    let kb = std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| {
            s.lines().find_map(|l| {
                l.strip_prefix("VmHWM:")
                    .and_then(|r| r.trim().trim_end_matches("kB").trim().parse::<f64>().ok())
            })
        });
    if let Some(kb) = kb {
        timers.resource(&format!("mem_hwm_gb_{name}"), kb / 1e6);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn memory_budgets() {
        let unset = MemoryLimit::from_gib(None);
        assert_eq!(unset.xc_kernel_mb(), 1024);
        assert_eq!(unset.response_slot_bytes(), 1 << 30);
        let m = MemoryLimit::from_gib(Some(4.0));
        assert_eq!(m.df_tensor_bytes(), 3usize << 30);
        assert_eq!(m.xc_kernel_mb(), 512);
        assert_eq!(m.response_slot_bytes(), 512 << 20);
        // a tiny limit leaves no room for H-bar blocks; a huge one gives the wish
        assert!(MemoryLimit { limit: Some(1) }.hbar_mb(4000.0) < 1e-6);
        assert_eq!(
            MemoryLimit {
                limit: Some(usize::MAX / 2)
            }
            .hbar_mb(4000.0),
            4000.0
        );
        assert!(
            matches!(unset.source(), "total RAM" | "cgroup limit" | "unknown")
                && m.source() == "memory_gib"
        );
        assert_eq!(m.bytes(), Some(4usize << 30));
        // memory_gb: reservations come off 75% of it; the free memory is never read
        let b = m.df_budget(1 << 30);
        assert_eq!(
            (b.bytes, b.by_available, b.available),
            (2 << 30, false, None)
        );
        assert_eq!(m.df_budget(8 << 30).bytes, 0);
        // unset: the probes are read (/proc/meminfo and the cgroup: Linux only; elsewhere the
        // free memory is unknown and the budget falls back to UNKNOWN_DF_BUDGET)
        let b = unset.df_budget(0);
        assert!(b.bytes > 0);
        assert_eq!(b.available.is_some(), cfg!(target_os = "linux"));
    }

    /// A phase requires its parts' sum unless a rule sets more; `memory_gib` values round up to
    /// 0.1 GiB.
    #[test]
    fn phase_needs() {
        const M: usize = 1 << 20;
        let n = PhaseNeed::new(
            "scf",
            vec![
                Part {
                    what: "a",
                    bytes: 300 * M,
                },
                Part {
                    what: "b",
                    bytes: 700 * M,
                },
            ],
        );
        assert_eq!(n.required(), 1000 * M);
        assert_eq!(n.describe(), "0.68 GiB b, 0.29 GiB a");
        assert_eq!(gib_holding(1000 * M), 1.0);
        assert_eq!(gib_holding(1 << 30), 1.0);
        assert_eq!(gib_holding((1 << 30) + 1), 1.1);
        assert!(Process::Python.baseline_bytes() > Process::Cli.baseline_bytes());
    }

    /// An unset memory_gib's run limit: the machine's limit within 90% of the free memory.
    #[test]
    fn run_limit_rule() {
        const G: usize = 1 << 30;
        assert_eq!(run_limit(Some(256 * G), Some(200 * G)), Some(180 * G));
        assert_eq!(run_limit(Some(8 * G), Some(200 * G)), Some(8 * G));
        assert_eq!(run_limit(None, Some(10 * G)), Some(9 * G));
        assert_eq!(run_limit(Some(4 * G), None), Some(4 * G));
        assert_eq!(run_limit(None, None), None);
        let m = MemoryLimit::from_gib(Some(4.0));
        assert_eq!(m.run_limit(), Some((4 << 30, "memory_gib")));
    }

    #[test]
    fn df_budget_rule() {
        const G: usize = 1 << 30;
        // 75% of the limit, no fixed ceiling (a 256 GiB pod: 192 GiB, not 64)
        let b = DfBudget::of(None, Some(256 * G), Some(250 * G), 0);
        assert_eq!((b.bytes, b.by_available), (192 * G, false));
        // reservations come off both terms
        let b = DfBudget::of(None, Some(256 * G), Some(250 * G), 2 * G);
        assert_eq!((b.bytes, b.by_available), (190 * G, false));
        // the guard: a node with 100 GiB free gets 90 of it
        let b = DfBudget::of(None, Some(256 * G), Some(100 * G), 0);
        assert_eq!(
            (b.bytes, b.by_available, b.available),
            (90 * G, true, Some(100 * G))
        );
        // memory_gb: no guard, whatever is free
        let b = DfBudget::of(Some(200 * G), Some(256 * G), Some(10 * G), 0);
        assert_eq!(
            (b.bytes, b.by_available, b.available),
            (150 * G, false, None)
        );
        // unknown machine: 8 GiB; nothing left: zero
        assert_eq!(DfBudget::of(None, None, None, 0).bytes, 8 * G);
        assert_eq!(DfBudget::of(None, Some(4 * G), None, 4 * G).bytes, 0);
    }
}
