//! The memory plan: what each engine may take of the device, from one rule per item.
//!
//! A device's engines plan against its limit (`Gpu::budget`: 95 % of the memory free when the
//! device opened, at most `device_memory_gib`), which is fixed for the session. Every
//! reserve and workspace that depends on that limit is decided here, once per device, and
//! handed to the engines as an explicit option; an engine checks a request against the
//! device's [`linus_core::Budget`] and never derives a policy from the free memory itself.
//!
//! | item | rule | taken by |
//! |---|---|---|
//! | `others_reserve` | max(8 GiB, limit / 10), at most limit / 4 | the SCF's DF-J/K plans its storage and workspaces within the limit less it (`linus_gpu::df::plan`); the response DF-J/K's shrunk storage rungs and the Ozaki exchange's pre-sliced planes leave it free for the other engines |
//! | `grid_partition` | `others_reserve` | the device grid partition, which runs beside the SCF's DF-J/K setup: within this share its grant (up to 512 MiB of tile scratch and about 48 B per grid point) fits beside the setup's planned allocations whatever the two threads' timing; a grid that needs more is partitioned on the host |
//! | `response_workspace` | the response J/K's default (8 GiB), at most limit / 4 | the response exchange's workspace per call |
//! | `k_operator_reserve` | the response J/K's default (6 GiB), at most limit / 4 | the stored exchange operator leaves it free (the CPKS kernel's workspace per call) |
//! | `w_planes_reserve` | the response J/K's default (2 GiB) | the Ozaki planes of a range-separated operator tensor W leave it free beside the workspace |
//! | `ground_cache_leave` | `response_workspace` + 1 GiB | the electric-field CPKS kernel keeps its ground state only where this stays free |
//!
//! The quarter-of-the-limit caps let a capped process (an 8 GB cap: 7.45 GiB) keep room for
//! the tensor and its optional storage and take the same paths as an uncapped one; a limit of
//! 32 GiB or more (an uncapped card) keeps the defaults. The fixed workspaces of the other
//! engines (the DF-J/K's exchange block, the gradients', the Hessian's) are their own options'
//! defaults. The host's budgets derive from `memory_gib` ([`linus_base::memory::MemoryLimit`]).

/// The device memory plan of a session ([module docs](self)).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct DeviceMemoryPlan {
    /// The device limit the engines plan against (bytes).
    pub limit: usize,
    pub others_reserve: usize,
    pub grid_partition: usize,
    pub response_workspace: usize,
    pub k_operator_reserve: usize,
    pub w_planes_reserve: usize,
    pub ground_cache_leave: usize,
}

/// The response J/K engine's defaults, as the device's provider gives them
/// ([`crate::driver::backend::Device`]): the exchange workspace per call, the stored operator's
/// reserve, the W planes' reserve (bytes).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ResponseDefaults {
    pub workspace: usize,
    pub k_operator_reserve: usize,
    pub w_planes_reserve: usize,
}

/// The reserve of [`DeviceMemoryPlan::others_reserve`] on an uncapped card (and the floor of
/// its tenth-of-the-limit rule).
const OTHERS_RESERVE: usize = 8 << 30;

/// What the electric-field CPKS kernel's ground cache leaves free beyond the response
/// exchange's workspace per call.
const GROUND_CACHE_EXTRA: usize = 1 << 30;

impl DeviceMemoryPlan {
    /// The plan for a device limit of `limit` bytes and the response J/K engine's defaults
    /// `defaults`.
    pub(crate) fn for_limit(limit: usize, defaults: ResponseDefaults) -> Self {
        let ResponseDefaults {
            workspace,
            k_operator_reserve,
            w_planes_reserve,
        } = defaults;
        let response_workspace = workspace.min(limit / 4);
        let others_reserve = OTHERS_RESERVE.max(limit / 10).min(limit / 4);
        DeviceMemoryPlan {
            limit,
            others_reserve,
            grid_partition: others_reserve,
            response_workspace,
            k_operator_reserve: k_operator_reserve.min(limit / 4),
            w_planes_reserve,
            ground_cache_leave: response_workspace + GROUND_CACHE_EXTRA,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A capped limit takes a quarter where the defaults would not fit; an uncapped card keeps
    /// them (the reserve for the other engines: max(8 GiB, limit / 10)).
    #[test]
    fn the_rules_scale_with_a_capped_limit() {
        const G: usize = 1 << 30;
        // the response J/K engine's defaults (linus-gpu's)
        const D: ResponseDefaults = ResponseDefaults {
            workspace: 8 * G,
            k_operator_reserve: 6 * G,
            w_planes_reserve: 2 * G,
        };
        // an 8 GB cap (7.45 GiB): a quarter, not 8 GiB beyond the whole limit
        let l8 = 8_000_000_000;
        let p = DeviceMemoryPlan::for_limit(l8, D);
        assert_eq!(
            (p.others_reserve, p.response_workspace, p.k_operator_reserve),
            (l8 / 4, l8 / 4, l8 / 4)
        );
        // the grid partition beside the DF-J/K's setup: the share that setup leaves
        assert_eq!(p.grid_partition, p.others_reserve);
        assert_eq!(p.ground_cache_leave, l8 / 4 + G);
        // 24 GiB: the operator's reserve kept, the workspace capped
        let p = DeviceMemoryPlan::for_limit(24 * G, D);
        assert_eq!((p.response_workspace, p.k_operator_reserve), (6 * G, 6 * G));
        // uncapped 32 GiB and more: the defaults, the reserve max(8 GiB, limit / 10)
        for (l, others) in [(32 * G, 8 * G), (90 * G, 9 * G), (134 * G, 134 * G / 10)] {
            let p = DeviceMemoryPlan::for_limit(l, D);
            assert_eq!(p.others_reserve, others);
            assert_eq!(
                (
                    p.response_workspace,
                    p.k_operator_reserve,
                    p.w_planes_reserve
                ),
                (8 * G, 6 * G, 2 * G)
            );
        }
    }
}
