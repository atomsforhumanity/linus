//! The storage and workspace plan of a DF-J/K engine ([`super::GpuDfJk`]): a pure function of
//! the problem's sizes and the device bytes the caller's memory plan gives the engine. It says
//! where B (and B^omega) are stored, the auxiliary block that orders the exchange's sums, the
//! rung of the storage ladder (decision D11) and the bytes a build takes. The engine's setup takes
//! it, its builds keep to it (the shell panels' block included), and linus's memory decisions
//! call it with the same inputs: the SCF's fit check before the first iteration
//! (`GpuDfJk::build_bytes`), the placement's needs and the memory-lean J's choice. Nothing here
//! reads the device's free memory, so the same sizes and budget give the same plan, block and
//! bits.
//!
//! The ladder: the tensor on the device with the automatic block; else, by their costs per build
//! (the measured constants [`DF_K_BLOCK_S`] and [`DF_STREAM_RATE`]), the tensor on the device
//! with the block shrunk to fit (a multiple of 8, at least [`K_BLOCK_MIN`]) and streamed from
//! page-locked host memory with the automatic block (where the host budget holds it), the
//! device-resident one first unless even its smallest block costs more than streaming; last,
//! streamed with the block shrunk.
//!
//! A build's workspace per exchange path: native K on the shell panels (the panel kernel's X and,
//! range-separated, Xw at the block, the gathered orbitals, cuBLAS's gathered panels), emulated K
//! on the unpacked block A_Q, or the Ozaki scheme's slices; with the J vectors, densities and
//! results of [`build_bytes`].

use super::{auto_k_block, DfStorage, PanelKernel, K_BLOCK_MIN};
use crate::emu::Sliced;
use linus_ints::eri3c::DfRows;

/// What one more K block of a DF-J/K build costs (seconds per build): the exchange's GEMMs over
/// smaller blocks run less efficiently and launch more often. Measured (u1an39 B3LYP/def2-TZVP,
/// nbf 1034, naux 2556, the 5.49 GiB tensor on the device of an RTX PRO 6000; 16 builds per SCF):
/// K blocks of 496 (the automatic block, 6 blocks), 136 (19) and 40 (64) functions took 0.1537,
/// 0.1554 and 0.1656 s per build, 0.13 and 0.21 ms per extra block; the larger is taken.
pub const DF_K_BLOCK_S: f64 = 2.1e-4;

/// The rate at which a build streams the tensor from page-locked host memory (bytes per second,
/// the copies partly behind the exchange's GEMMs). Measured (as [`DF_K_BLOCK_S`]): streamed with
/// the automatic block, 0.2404 s per build, 0.0867 s more than on the device for 5.90e9 bytes,
/// 68 GB/s.
pub const DF_STREAM_RATE: f64 = 68e9;

/// The panel path's per-build exchange workspace (X and Xw, the gathered orbitals, cuBLAS's
/// gathered panels) that sets its automatic block: the block hardly changes its speed (ts113 on
/// an H200: 256 to 2048 functions within 5 %), so it is sized for a bounded peak.
pub const PANEL_K_BYTES: usize = 8 << 30;

/// Bytes of cuBLAS's gathered panels of one shell group (the default of
/// `GpuDfJkOptions::k_panel_workspace`).
pub const PANEL_WORKSPACE: usize = 2 << 30;

/// The sizes of a density-fitting problem: functions, auxiliary functions, the layout's rows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DfSizes {
    pub nbf: usize,
    pub naux: usize,
    pub rows: DfRows,
}

impl DfSizes {
    /// Bytes of one stored tensor (B or B^omega).
    pub fn tensor_bytes(&self) -> usize {
        8 * self.rows.nx * self.naux
    }
}

/// The exchange a plan sizes the builds for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DfExchange {
    /// No exchange: J alone.
    None,
    /// Native K in FP64 on the shell panels, the first GEMM on `kernel`; cuBLAS's gathered
    /// panels in groups of at most `workspace` bytes.
    Panels {
        kernel: PanelKernel,
        workspace: usize,
    },
    /// K in cuBLAS's emulation on the unpacked block A_Q.
    Unpacked,
    /// K in the Ozaki scheme with `slices` slices.
    Ozaki { slices: usize },
}

/// The inputs of a plan ([`plan`]).
#[derive(Clone, Copy, Debug)]
pub struct DfPlanRequest {
    pub sizes: DfSizes,
    /// B^omega (a range-separated exchange) planned with B.
    pub lr: bool,
    pub exchange: DfExchange,
    /// The factor columns of the largest build planned (the panel path's X).
    pub k_columns: usize,
    /// Device bytes the engine may hold: the tensors, a build's workspace and its setup's.
    pub budget: usize,
    /// Bytes the shrunk rungs leave of `budget` for the engines set up after this one.
    pub reserve: usize,
    /// Page-locked host bytes for streamed tensors (None: no limit; zero: the device only).
    pub host_budget: Option<usize>,
    /// The storage asked for of B and B^omega (Auto: the ladder).
    pub storage: DfStorage,
    pub lr_storage: DfStorage,
    /// The exchange block asked for (0: automatic, shrunk on the ladder's shrunk rungs).
    pub k_block: usize,
    /// The unpacked block A_Q's bytes that set the automatic block of the other paths.
    pub k_workspace: usize,
    /// Device bytes the setup holds beside the tensors (the metric, a batch of rows, the device
    /// integrals' builder); a build's workspace counts at least this.
    pub setup_extra: usize,
}

/// The rung of the storage ladder a plan took.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DfRung {
    /// On the device with the automatic block.
    Device,
    /// On the device with the block shrunk to fit.
    DeviceShrunk,
    /// Streamed from page-locked host memory with the automatic block.
    Host,
    /// Streamed with the block shrunk.
    HostShrunk,
}

/// A plan ([`plan`]).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DfPlan {
    pub storage: DfStorage,
    /// Where B^omega goes (None: not planned).
    pub lr_storage: Option<DfStorage>,
    /// The auxiliary block of the exchange (and of streamed storage): it orders K's sums.
    pub k_block: usize,
    /// The automatic block of this storage.
    pub auto_block: usize,
    pub rung: DfRung,
    /// Device bytes of the tensors (or their staging) and a build's workspace at the block.
    pub device_bytes: usize,
    /// Seconds per build this rung costs beyond the device-resident tensor with the automatic
    /// block (D11), and what streaming the tensor would cost.
    pub extra_seconds: f64,
    pub stream_seconds: f64,
}

/// Bytes of the rows of a setup batch: about 1 GiB of rows, at least the largest pair's.
pub(crate) fn batch_rows(naux: usize, max_pair_rows: usize) -> usize {
    ((1usize << 30) / (8 * naux.max(1))).max(max_pair_rows.max(1))
}

/// Device bytes a setup holds beside the tensors before its builds: the metric (and its factor)
/// and one batch of rows (about 1 GiB, at least the largest pair's); the device integrals' builder comes on top where the
/// engine packs it ([`DfPlanRequest::setup_extra`]).
pub fn setup_bytes(sizes: &DfSizes) -> usize {
    let naux = sizes.naux;
    8 * (naux * naux + batch_rows(naux, sizes.rows.max_pair_rows) * naux)
}

/// The seconds per build a device-resident DF tensor with K blocks of `nq` auxiliary functions
/// costs beyond the automatic block `nq_auto` (`naux` auxiliary functions).
pub fn shrunk_seconds(naux: usize, nq_auto: usize, nq: usize) -> f64 {
    let extra = naux
        .div_ceil(nq.max(1))
        .saturating_sub(naux.div_ceil(nq_auto.max(1)));
    DF_K_BLOCK_S * extra as f64
}

/// The seconds per build streaming a DF tensor of `bytes` from host memory costs beyond keeping
/// it on the device.
pub fn stream_seconds(bytes: usize) -> f64 {
    bytes as f64 / DF_STREAM_RATE
}

/// The bytes of the unpacked path's per-build buffers: A_Q, X_Q (and Xw_Q range-separated; nocc
/// <= nbf / 2 assumed), densities, J/K, the J vectors.
pub(crate) fn unpacked_build_bytes(nbf: usize, nx: usize, nq: usize, lr: bool) -> usize {
    let x = nq * nbf * nbf.div_ceil(2);
    8 * (nq * nbf * nbf + x * if lr { 2 } else { 1 } + 6 * nbf * nbf) + 8 * 4 * nx
}

/// The Ozaki exchange's per-build device bytes with blocks of `nq` functions, for `ntot` factor
/// columns with GEMM1's columns in chunks of at most `cols`: the block's sliced operand, X's
/// slices and W's (or X's again in descending digit order: two slots either way), the INT32
/// partial sums, C's slices.
pub(crate) fn ozaki_build_bytes(
    nbf: usize,
    nq: usize,
    slices: usize,
    ntot: usize,
    cols: usize,
) -> usize {
    let jj = (nq * ntot.max(1)).min(super::device::OZ_J_CHUNK);
    Sliced::bytes_for(nq * nbf, nbf, slices)
        + 2 * Sliced::bytes_for(nbf, jj, slices)
        + Sliced::bytes_for(ntot, nbf, slices)
        + 4 * slices * (nq * nbf).next_multiple_of(crate::emu::ROW_ALIGN) * cols.min(ntot)
        + 4 * (slices + 2) * nbf.next_multiple_of(crate::emu::ROW_ALIGN) * nbf
}

/// Bytes of cuBLAS's gathered panels at block `nq` in shell groups of at most `workspace`
/// bytes (a larger shell alone; never more than all panel rows).
pub(crate) fn gathered_panel_bytes(rows: &DfRows, nq: usize, workspace: usize) -> usize {
    (8 * rows.max_panel_rows * nq)
        .max(workspace)
        .min(8 * rows.np * nq)
}

/// Device bytes of the shell panels' tables, uploaded at the first panel build and held after
/// it: per panel row its stored row and per gathered column its function (4 B each), and five
/// ints per function of a shell with partners (at most nbf functions).
pub(crate) fn panel_table_bytes(sizes: &DfSizes) -> usize {
    4 * (sizes.rows.np + sizes.rows.ncg + 5 * sizes.nbf)
}

/// The panel path's exchange workspace of one build at block `nq` for `ncols` factor columns
/// (X and, range-separated, Xw of at most `width` columns per pass; the gathered orbitals of all
/// `ncols`, in rows of an even count for the tensor-core kernel; cuBLAS's gathered panels).
pub(crate) fn panel_k_bytes(
    sizes: &DfSizes,
    (kernel, workspace): (PanelKernel, usize),
    nq: usize,
    (ncols, width): (usize, usize),
    lr: bool,
) -> usize {
    let tensor_cores = kernel != PanelKernel::Strided;
    let ldc = if tensor_cores {
        ncols.next_multiple_of(2)
    } else {
        ncols
    };
    let x = 8 * nq * sizes.nbf * width * if lr { 2 } else { 1 };
    let cg = 8 * sizes.rows.ncg * ldc.max(1);
    let pw = if tensor_cores {
        0
    } else {
        gathered_panel_bytes(&sizes.rows, nq, workspace)
    };
    x + cg + pw
}

/// The rounding of the panel path's automatic block where it reaches 256 auxiliary functions:
/// whole tiles of the tensor-core kernel (its `bm`, `df::MMA_TILE`), on both panel kernels so
/// that the block, which orders K's sums, does not depend on the kernel.
const PANEL_BLOCK_TILE: usize = 128;

/// The automatic block of the panel path for `ncols` factor columns: X (and Xw), the gathered
/// orbitals and panels within [`PANEL_K_BYTES`]; whole [`PANEL_BLOCK_TILE`]s where the block
/// reaches two of them, else a multiple of 8; at least `floor` (the other paths' automatic block)
/// and at most naux.
pub(crate) fn panel_auto_block(
    sizes: &DfSizes,
    panels: (PanelKernel, usize),
    ncols: usize,
    lr: bool,
    floor: usize,
) -> usize {
    let per_q = panel_k_bytes(sizes, panels, 1, (ncols, ncols), lr)
        - panel_k_bytes(sizes, panels, 0, (ncols, ncols), lr);
    let fixed = panel_k_bytes(sizes, panels, 0, (ncols, ncols), lr);
    let q = (PANEL_K_BYTES.saturating_sub(fixed) / per_q.max(1)).max(floor);
    if q >= sizes.naux {
        sizes.naux
    } else if q >= 2 * PANEL_BLOCK_TILE {
        q / PANEL_BLOCK_TILE * PANEL_BLOCK_TILE
    } else {
        (q / 8 * 8).max(8)
    }
}

/// A build's device bytes beyond the engine for `nd` densities and the factors of `ncols`
/// columns at block `nq` (the panel path: X of at most `cols` columns per pass) with
/// `exchange`, B^omega's exchange too with `lr`: the J vectors (and, with `host_mode`, the
/// streamed blocks' fitted gamma), the exchange's workspace; with `io`, the caller's inputs and
/// results copied to the device (`GpuDfJk::compute`).
#[allow(
    clippy::too_many_arguments,
    reason = "a build's shape: densities, factors, block, storage, the exchange and its operator"
)]
pub fn build_bytes(
    sizes: &DfSizes,
    exchange: DfExchange,
    (nq, cols): (usize, usize),
    nd: usize,
    ncols: &[usize],
    lr: bool,
    host_mode: bool,
    io: bool,
) -> usize {
    let (nbf, naux, nx) = (sizes.nbf, sizes.naux, sizes.rows.nx);
    let n2 = nbf * nbf;
    let ntot: usize = ncols.iter().sum();
    // the J vectors (w and the J rows), alive through the build; device storage's fitted gamma,
    // freed before K
    let jv = if nd > 0 { 8 * 2 * nx * nd } else { 0 };
    let jg = if nd > 0 && !host_mode {
        8 * naux * nd
    } else {
        0
    };
    let mut kb = 0;
    if ntot > 0 {
        kb += match exchange {
            DfExchange::None => 0,
            DfExchange::Panels { kernel, workspace } => {
                panel_k_bytes(sizes, (kernel, workspace), nq, (ntot, ntot.min(cols)), lr)
                    + panel_table_bytes(sizes)
            }
            // A_Q, X_Q (and Xw_Q)
            DfExchange::Unpacked => 8 * nq * n2 + 8 * nq * nbf * ntot * if lr { 2 } else { 1 },
            // X_Q (and Xw_Q), C's slices, the workspace at the fewest columns per GEMM, a margin
            DfExchange::Ozaki { slices } => {
                8 * nq * nbf * ntot * if lr { 2 } else { 1 }
                    + Sliced::bytes_for(ntot, nbf, slices)
                    + ozaki_build_bytes(nbf, nq, slices, ntot, 16.min(ntot))
                    + (256 << 20)
            }
        };
    }
    if host_mode && nd > 0 {
        kb += 8 * nq * nd;
    }
    let io = if io {
        8 * (2 * nd * n2 + ncols.len() * n2 + nbf * ntot)
    } else {
        0
    };
    io + jv + jg.max(kb)
}

/// The workspace the plan counts for a build at block `q` (the largest build planned:
/// [`DfPlanRequest::k_columns`]; the unpacked and Ozaki paths as their setups always counted
/// them, nocc <= nbf / 2 and up to nbf columns), at least the setup's own.
fn work_at(r: &DfPlanRequest, q: usize) -> usize {
    let (nbf, nx) = (r.sizes.nbf, r.sizes.rows.nx);
    let w = match r.exchange {
        DfExchange::Panels { kernel, workspace } => {
            let cols = r.k_columns.max(1);
            8 * (6 * nbf * nbf + 4 * nx)
                + panel_k_bytes(&r.sizes, (kernel, workspace), q, (cols, cols), r.lr)
                + panel_table_bytes(&r.sizes)
        }
        DfExchange::Ozaki { slices } => {
            unpacked_build_bytes(nbf, nx, q, r.lr) - 8 * q * nbf * nbf
                + Sliced::bytes_for(nbf, nbf, slices)
                + ozaki_build_bytes(nbf, q, slices, nbf, 16.min(nbf))
                + (256 << 20)
        }
        DfExchange::Unpacked => unpacked_build_bytes(nbf, nx, q, r.lr),
        // J alone: the densities, J and the J vectors (no exchange block)
        DfExchange::None => 8 * (6 * nbf * nbf + 4 * nx),
    };
    w.max(r.setup_extra)
}

/// The tensors on `storage` (host storage: a staging buffer of `q` functions per streamed
/// tensor) and a build's workspace at block `q`.
fn need_at(r: &DfPlanRequest, b: DfStorage, w: Option<DfStorage>, q: usize) -> usize {
    let t = |s: DfStorage| {
        if s == DfStorage::Device {
            r.sizes.tensor_bytes()
        } else {
            8 * q * r.sizes.rows.nx
        }
    };
    t(b) + w.map(t).unwrap_or(0) + work_at(r, q)
}

/// The automatic block on `storage`: the panel path's on the device, else the unpacked block
/// A_Q within `k_workspace`.
fn auto_block(r: &DfPlanRequest, storage: DfStorage) -> usize {
    let (nbf, naux) = (r.sizes.nbf, r.sizes.naux);
    if r.k_block > 0 {
        return r.k_block.min(naux);
    }
    let unpacked = auto_k_block(nbf, naux, r.k_workspace);
    match (r.exchange, storage) {
        (DfExchange::Panels { kernel, workspace }, DfStorage::Device) => {
            let mut q = panel_auto_block(
                &r.sizes,
                (kernel, workspace),
                r.k_columns.max(1),
                r.lr,
                unpacked,
            );
            // within the budget too, B^omega on the device unless asked otherwise: the panel
            // path's speed hardly depends on its block, so a smaller block beats streaming a
            // tensor (u1an39/def2-SVP wB97M-V under an 8 GB cap streamed B^omega at block 1328)
            let lr = r.lr.then_some(match r.lr_storage {
                DfStorage::Host => DfStorage::Host,
                _ => DfStorage::Device,
            });
            while q > unpacked && need_at(r, DfStorage::Device, lr, q) > r.budget {
                q = if q > 2 * PANEL_BLOCK_TILE {
                    ((q - 1) / PANEL_BLOCK_TILE * PANEL_BLOCK_TILE).max(2 * PANEL_BLOCK_TILE)
                } else {
                    ((q - 1) / 8 * 8).max(8)
                }
                .max(unpacked);
            }
            q
        }
        _ => unpacked,
    }
}

/// Why no rung fits ([`plan`]): the smallest device bytes a rung needs and the budget, and why
/// streaming is out.
#[derive(Clone, Debug, PartialEq)]
pub struct DfPlanRefusal {
    pub need: usize,
    pub budget: usize,
    /// The host budget of streamed tensors is what rules streaming out.
    pub host_limited: bool,
    pub reason: String,
}

/// The plan of `r` (module docs): the first rung of the ladder that fits `r.budget`.
pub fn plan(r: &DfPlanRequest) -> Result<DfPlan, DfPlanRefusal> {
    let naux = r.sizes.naux;
    let tensor = r.sizes.tensor_bytes() * if r.lr { 2 } else { 1 };
    let host_ok = r.host_budget.is_none_or(|b| tensor <= b);
    let shrinkable = r.k_block == 0;
    // B^omega with B on `b` at block q: on the device when B is and both fit (Auto), or as asked
    let lr_for = |b: DfStorage, q: usize| -> Option<DfStorage> {
        r.lr.then(|| match r.lr_storage {
            DfStorage::Auto
                if b == DfStorage::Device
                    && need_at(r, DfStorage::Device, Some(DfStorage::Device), q) <= r.budget =>
            {
                DfStorage::Device
            }
            DfStorage::Auto => DfStorage::Host,
            s => s,
        })
    };
    let at = |storage: DfStorage, shrink: bool| -> Option<DfPlan> {
        let auto = auto_block(r, storage);
        let lr_storage = lr_for(storage, auto);
        let fits =
            |q: usize, reserve: usize| need_at(r, storage, lr_storage, q) + reserve <= r.budget;
        let (q, rung) = if shrink {
            let mut q = auto;
            while q > K_BLOCK_MIN && !fits(q, r.reserve) {
                q = ((q - 1) / 8 * 8).max(K_BLOCK_MIN.min(auto));
            }
            if !fits(q, r.reserve) {
                return None;
            }
            let rung = if storage == DfStorage::Device {
                DfRung::DeviceShrunk
            } else {
                DfRung::HostShrunk
            };
            (q, rung)
        } else {
            if !fits(auto, 0) {
                return None;
            }
            let rung = if storage == DfStorage::Device {
                DfRung::Device
            } else {
                DfRung::Host
            };
            (auto, rung)
        };
        let shrunk = shrunk_seconds(naux, auto, q);
        Some(DfPlan {
            storage,
            lr_storage,
            k_block: q,
            auto_block: auto,
            rung,
            device_bytes: need_at(r, storage, lr_storage, q),
            extra_seconds: shrunk
                + if storage == DfStorage::Host {
                    stream_seconds(tensor)
                } else {
                    0.0
                },
            stream_seconds: stream_seconds(tensor),
        })
    };
    // the rungs in the order of their costs per build: the device-resident tensor with the
    // automatic block shrunk to fit (an explicit block stays as set), counted at its smallest
    // block, against streaming the tensor with the automatic block
    let device_auto = auto_block(r, DfStorage::Device);
    let shrunk_first =
        shrinkable && shrunk_seconds(naux, device_auto, K_BLOCK_MIN) <= stream_seconds(tensor);
    let (device, host) = match r.storage {
        DfStorage::Auto => (true, host_ok),
        DfStorage::Device => (true, false),
        DfStorage::Host => (false, true),
    };
    let rungs = [
        (device, DfStorage::Device, false),
        (
            device && shrinkable && shrunk_first,
            DfStorage::Device,
            true,
        ),
        (host, DfStorage::Host, false),
        (
            device && shrinkable && !shrunk_first,
            DfStorage::Device,
            true,
        ),
        (host && shrinkable, DfStorage::Host, true),
    ];
    for (_, storage, shrink) in rungs.into_iter().filter(|x| x.0) {
        if let Some(p) = at(storage, shrink) {
            return Ok(p);
        }
    }
    let gib = |b: usize| b as f64 / 1073741824.0;
    let smallest = |s: DfStorage| {
        let q = if shrinkable {
            K_BLOCK_MIN.min(auto_block(r, s))
        } else {
            auto_block(r, s)
        };
        need_at(r, s, lr_for(s, q), q)
    };
    let need = match (device, host) {
        (true, true) => smallest(DfStorage::Device).min(smallest(DfStorage::Host)),
        (true, false) => smallest(DfStorage::Device),
        _ => smallest(DfStorage::Host),
    };
    let host_limited = r.storage == DfStorage::Auto && !host_ok;
    let reason = if host_limited {
        match r.host_budget {
            Some(0) => format!(
                "the density-fitting tensors ({:.2} GiB) and a build's workspace do not fit the {:.2} GiB \
                 the engine may hold on the device (no streaming)",
                gib(tensor),
                gib(r.budget)
            ),
            Some(b) => format!(
                "the density-fitting tensors ({:.2} GiB) fit neither the device ({:.2} GiB the engine may \
                 hold) nor the host budget of their streamed storage ({:.2} GiB)",
                gib(tensor),
                gib(r.budget),
                gib(b)
            ),
            None => unreachable!("an unlimited host budget holds every tensor"),
        }
    } else {
        format!(
            "the density-fitting tensors ({:.2} GiB) or their staging and a build's workspace do not fit \
             the {:.2} GiB the engine may hold on any rung (at least {:.2} GiB)",
            gib(tensor),
            gib(r.budget),
            gib(need)
        )
    };
    Err(DfPlanRefusal {
        need,
        budget: r.budget,
        host_limited,
        reason,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const G: usize = 1 << 30;

    fn sizes() -> DfSizes {
        // u1an39/def2-TZVP-like: nbf 1034, naux 2556, 268k stored rows (5.1 GiB per tensor)
        DfSizes {
            nbf: 1034,
            naux: 2556,
            rows: DfRows {
                nx: 268_000,
                np: 480_000,
                ncg: 140_000,
                max_panel_rows: 6_000,
                max_pair_rows: 196,
            },
        }
    }

    fn request(budget: usize, exchange: DfExchange) -> DfPlanRequest {
        DfPlanRequest {
            sizes: sizes(),
            lr: false,
            exchange,
            k_columns: 105,
            budget,
            reserve: 0,
            host_budget: None,
            storage: DfStorage::Auto,
            lr_storage: DfStorage::Auto,
            k_block: 0,
            k_workspace: 4 * G,
            setup_extra: 0,
        }
    }

    const PANELS: DfExchange = DfExchange::Panels {
        kernel: PanelKernel::MmaSm90a,
        workspace: PANEL_WORKSPACE,
    };

    /// A plan is a function of its inputs: the same request, the same plan; the ladder takes
    /// the device with the automatic block where it fits, then the cheaper of the shrunk block
    /// and streaming, then streamed with the shrunk block, and refuses with the smallest need.
    #[test]
    fn the_ladder_by_budget() {
        let big = plan(&request(120 * G, PANELS)).unwrap();
        assert_eq!(big, plan(&request(120 * G, PANELS)).unwrap());
        assert_eq!((big.storage, big.rung), (DfStorage::Device, DfRung::Device));
        // u1an39's panels: every auxiliary function in one block
        assert_eq!(big.k_block, 2556);
        // the unpacked path's automatic block: A_Q within 4 GiB
        let emu = plan(&request(120 * G, DfExchange::Unpacked)).unwrap();
        assert_eq!(emu.k_block, auto_k_block(1034, 2556, 4 * G));
        // a budget below the tensor and the automatic block's workspace: shrunk or streamed,
        // whichever costs less per build (streaming 5.1 GiB costs 80 ms, 2556 / 32 blocks 16 ms)
        let tight = plan(&request(6 * G, DfExchange::Unpacked)).unwrap();
        assert_eq!(tight.rung, DfRung::DeviceShrunk);
        assert!(tight.k_block < emu.k_block && tight.k_block >= K_BLOCK_MIN);
        assert!(tight.device_bytes <= 6 * G);
        // no room for the tensor: streamed
        let streamed = plan(&request(3 * G, DfExchange::Unpacked)).unwrap();
        assert_eq!(streamed.storage, DfStorage::Host);
        // a host budget below the tensor: refused, with the reason
        let r = DfPlanRequest {
            host_budget: Some(G),
            ..request(3 * G, DfExchange::Unpacked)
        };
        let refusal = plan(&r).unwrap_err();
        assert!(refusal.reason.contains("host budget"), "{}", refusal.reason);
    }

    /// The build bytes the fit check counts are the panel path's: X at the block, the gathered
    /// orbitals, the panels' tables and (cuBLAS) the gathered panels; no A_Q.
    #[test]
    fn panel_builds_count_no_unpacked_block() {
        let s = sizes();
        let mma = build_bytes(&s, PANELS, (512, 105), 1, &[105], false, false, false);
        let unpacked = build_bytes(
            &s,
            DfExchange::Unpacked,
            (512, 105),
            1,
            &[105],
            false,
            false,
            false,
        );
        let tables = 4 * (480_000 + 140_000 + 5 * 1034);
        assert_eq!(panel_table_bytes(&s), tables);
        assert_eq!(
            mma,
            8 * 512 * 1034 * 105 + 8 * 140_000 * 106 + tables + 8 * 2 * 268_000,
            "X, the gathered orbitals (even rows), the tables, the J vectors"
        );
        assert!(unpacked > 8 * 512 * 1034 * 1034);
        let strided = DfExchange::Panels {
            kernel: PanelKernel::Strided,
            workspace: PANEL_WORKSPACE,
        };
        let cublas = build_bytes(&s, strided, (512, 105), 1, &[105], false, false, false);
        // the groups' gathered panels: all panel rows at this block, below the 2 GiB workspace
        assert_eq!(
            cublas,
            8 * 512 * 1034 * 105 + 8 * 140_000 * 105 + 8 * 480_000 * 512 + tables + 8 * 2 * 268_000
        );
    }

    /// J alone counts no exchange block: its workspace is the densities, J and the J vectors,
    /// so a J-only engine keeps its tensor on the device where the unpacked exchange's A_Q
    /// (the old count) would have pushed it to host memory.
    #[test]
    fn j_alone_counts_no_exchange_block() {
        let s = sizes();
        let (nbf, nx) = (s.nbf, s.rows.nx);
        let r = request(120 * G, DfExchange::None);
        assert_eq!(work_at(&r, 512), 8 * (6 * nbf * nbf + 4 * nx));
        assert!(work_at(&r, 512) < work_at(&request(120 * G, DfExchange::Unpacked), 512));
        // within the tensor and a J build: the device, where the unpacked count streams
        let tight = s.tensor_bytes() + work_at(&r, 512) + (1 << 20);
        assert_eq!(
            plan(&request(tight, DfExchange::None)).unwrap().storage,
            DfStorage::Device
        );
    }
}
