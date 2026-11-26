//! Native FP64 K on the shell panels ([`PanelPlan`]): the panels' device tables, the panel
//! blocks planned at the first panel build, the pass over them (the orbitals gathered per shell,
//! GEMM1 on the tensor-core kernel or on cuBLAS's gathered panels, the K update), and the
//! tensor-core kernel's variants. A child of the engine's module: its methods are
//! [`GpuDfJk`]'s.

use super::*;

/// A [`PanelPlan`] on the device: the panel rows' stored rows and the gathered-orbital
/// columns' functions, with the shells on the host.
pub(super) struct PanelTables {
    shells: Vec<PanelShell>,
    src: DeviceBuf<i32>,
    cn: DeviceBuf<i32>,
    /// Per function of a shell with partners: its first panel row r0 + ia, the shell's d, w
    /// and c0, and the function (the tensor-core kernel's table).
    fns: DeviceBuf<i32>,
    nfn: usize,
}

impl PanelTables {
    fn new(gpu: &Gpu, plan: &PanelPlan) -> Result<PanelTables, GpuError> {
        let mut fns = Vec::new();
        for s in plan.shells.iter().filter(|s| s.w > 0) {
            for ia in 0..s.d {
                fns.extend([s.r0 + ia, s.d, s.w, s.c0, s.f0 + ia].map(|v| v as i32));
            }
        }
        let nfn = fns.len() / 5;
        Ok(PanelTables {
            shells: plan.shells.clone(),
            src: gpu.upload(if plan.src.is_empty() { &[0] } else { &plan.src })?,
            cn: gpu.upload(if plan.cn.is_empty() { &[0] } else { &plan.cn })?,
            fns: gpu.upload(if fns.is_empty() { &[0] } else { &fns })?,
            nfn,
        })
    }

    fn ncg(&self) -> usize {
        self.shells.last().map_or(0, |s| s.c0 + s.w)
    }

    fn bytes(&self) -> usize {
        4 * (self.src.len() + self.cn.len() + self.fns.len())
    }
}

/// The auxiliary blocks of native K on the shell panels, planned at the first panel build
/// ([`GpuDfJk::plan_panel_blocks`]): their size, the factor columns per pass it was planned
/// for, the shell groups (shells s0..s1, panel rows r0..r1) whose gathered panels fit the
/// workspace and the largest group's rows.
pub(super) struct PanelBlocks {
    nq: usize,
    blocks: Vec<(usize, usize)>,
    cols: usize,
    groups: Vec<(usize, usize, usize, usize)>,
    pw_rows: usize,
}

/// The gathered panel rows r0..r0 + nrows of a block from its stored rows: p\[p_off +
/// (r - r0) nqb + q\] = row src\[r\] of the block, its functions q < nqb ([`PanelPlan`]).
fn panel_gather(
    gpu: &Gpu,
    kernels: &Kernels,
    tabs: &PanelTables,
    (r0, nrows): (usize, usize),
    blk: &TensorBlock<'_>,
    (p, p_off): (&mut DeviceBuf<f64>, usize),
) -> Result<(), GpuError> {
    if nrows == 0 {
        return Ok(());
    }
    if blk.off + blk.nq > blk.ld || r0 + nrows > tabs.src.len() || p_off + nrows * blk.nq > p.len()
    {
        return Err(GpuError::Contract("DF panels: outside the block".into()));
    }
    // 8 rows per thread block, 32 threads along q each
    let cfg = Launch {
        grid: (nrows.div_ceil(8) as u32, 1, 1),
        block: (32, 8, 1),
        shared_bytes: 0,
    };
    // SAFETY: matches df_panel_gather; reads nrows stored rows src[r0..] < nx (the layout's)
    // at off + src[r] ld + q, q < nq <= ld - off, inside the block's buffer; writes nrows nq
    // values from p_off (checked above).
    unsafe {
        gpu.launch(
            &kernels.panel_gather,
            cfg,
            &[
                Arg::I64(nrows as i64),
                Arg::I32(blk.nq as i32),
                Arg::from(tabs.src.at(r0)),
                Arg::buf(blk.buf),
                Arg::I64(blk.off as i64),
                Arg::I64(blk.ld as i64),
                Arg::from(p.at(p_off)),
            ],
        )
    }
}

/// The panels' first GEMM of one block for one chunk of factor columns on FP64 tensor cores
/// (`device/df/panel_mma.cu`): X\[xo + q + nqb (i + nc m)\] for every function m of a shell
/// with partners, its panel rows read from the block's stored rows.
fn panel_mma(
    gpu: &Gpu,
    pm: &mut PanelMma,
    tabs: &PanelTables,
    blk: &TensorBlock<'_>,
    (cc, nc, ldc, nbf): (usize, usize, usize, usize),
    cg: &DeviceBuf<f64>,
    (xt, xo): (&mut DeviceBuf<f64>, usize),
) -> Result<(), GpuError> {
    let bm = pm.tile[0];
    let nqb = blk.nq;
    if nc == 0 || tabs.nfn == 0 {
        return Ok(());
    }
    if blk.off + nqb > blk.ld
        || cc + nc > ldc
        || ldc * tabs.ncg() > cg.len()
        || xo + nqb * nc * nbf > xt.len()
    {
        return Err(GpuError::Contract(
            "DF panels: tensor-core GEMM operands outside".into(),
        ));
    }
    // every operand row 16-byte aligned: bulk copies when A's rows are an even number of
    // doubles (B's are rounded up within ldc), else two doubles per copy; else one
    let aligned = [blk.off, blk.ld, cc, ldc]
        .iter()
        .all(|v| v.is_multiple_of(2));
    let copy = match (aligned, nqb.is_multiple_of(2)) {
        (true, true) => MmaCopy::Bulk,
        (true, false) => MmaCopy::Pairs,
        (false, _) => MmaCopy::Singles,
    };
    let bn = pm.bn(nc);
    let k = pm.kernel(gpu, bn, copy)?;
    // at most 65535 functions per launch (the grid's third dimension)
    let mut f0 = 0;
    while f0 < tabs.nfn {
        let nf = (tabs.nfn - f0).min(65535);
        let cfg = Launch {
            grid: (nqb.div_ceil(bm) as u32, nc.div_ceil(bn) as u32, nf as u32),
            block: (k.threads as u32, 1, 1),
            shared_bytes: k.smem as u32,
        };
        // SAFETY: matches df_panel_mma of the kernel's module compiled for the tile, the
        // orbital tile bn and the copy mode (PanelMma::kernel: bulk copies only with every
        // row start even and nqb even, 16-byte copies only with every row start even,
        // checked above); reads the functions' table entries f0.., their panel rows' stored
        // rows (src < nx, q < nqb within ld - off of the block's buffer) and cg's columns
        // cc + i (i < nc, bulk copies up to the even column after: < ldc as cc, ldc even) of
        // the gathered-orbital rows (checked above); writes X[xo + q + nqb (i + nc m)] for
        // m < nbf, inside xt (checked above).
        unsafe {
            gpu.launch(
                &k.kernel,
                cfg,
                &[
                    Arg::I32(nqb as i32),
                    Arg::I32(nc as i32),
                    Arg::I32(ldc as i32),
                    Arg::from(tabs.fns.at(5 * f0)),
                    Arg::buf(&tabs.src),
                    Arg::buf(blk.buf),
                    Arg::I64(blk.off as i64),
                    Arg::I64(blk.ld as i64),
                    Arg::buf(cg),
                    Arg::I64(cc as i64),
                    Arg::buf_mut(&mut *xt),
                    Arg::I64(xo as i64),
                ],
            )
        }?;
        f0 += nf;
    }
    Ok(())
}

/// How the tensor-core kernel copies a k-step's operand rows into shared memory (`DFM_COPY`
/// of `device/df/panel_mma.cu`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MmaCopy {
    /// One bulk copy (the TMA unit) per row: rows 16-byte aligned, A's an even number of
    /// doubles.
    Bulk,
    /// cp.async of two doubles per lane: rows 16-byte aligned.
    Pairs,
    /// cp.async of one double per lane.
    Singles,
}

/// The tensor-core kernel's default tile: 128 auxiliary functions and up to 128 orbitals per
/// block, 8 consumer warps (4 x 2, each 32 x 64), 4 k-steps in flight.
pub const MMA_TILE: [usize; 5] = [128, 128, 4, 2, 4];

/// Registers per lane of an SM: 65536 in 4 partitions of 32 lanes each hold 512 per lane.
const SM_REGISTERS: usize = 65536;

const PARTITION_REGISTERS_PER_LANE: usize = 512;

/// The most registers a thread may hold (setmaxnreg's range ends at 256; 248 keeps the
/// count a multiple of 8 below it with the producer's share).
const MAX_THREAD_REGISTERS: usize = 248;

/// The producer warpgroup's registers per lane after the hand-over (`DFM_PREG`: its one
/// copying warp needs few).
const MMA_PRODUCER_REGISTERS: usize = 56;

/// The consumers' registers per lane after the hand-over (`DFM_CREG`) for `cw` consumer
/// warps: an SM partition's 512 per lane shared by its cw / 4 consumer warps and one
/// producer warp, in multiples of 8.
fn mma_consumer_registers(cw: usize) -> usize {
    ((PARTITION_REGISTERS_PER_LANE - MMA_PRODUCER_REGISTERS) / (cw / 4).max(1) / 8 * 8)
        .min(MAX_THREAD_REGISTERS)
}

/// The registers per thread a launch of `threads` threads per block (one block per SM)
/// starts with, in multiples of 8.
fn launch_registers(threads: usize) -> usize {
    SM_REGISTERS / threads.max(1) / 8 * 8
}

/// The source of the tensor-core kernel for `tile` (bm, bn, consumer warps along each,
/// stages) and copy mode `copy`: its parameters defined in front of
/// `device/df/panel_mma.cu`.
pub fn panel_mma_source(tile: [usize; 5], copy: MmaCopy) -> String {
    let [bm, bn, wm, wn, st] = tile;
    let c = match copy {
        MmaCopy::Bulk => 0,
        MmaCopy::Pairs => 1,
        MmaCopy::Singles => 2,
    };
    format!(
        "#define DFM_BM {bm}\n#define DFM_BN {bn}\n#define DFM_WM {wm}\n#define DFM_WN {wn}\n\
         #define DFM_ST {st}\n#define DFM_COPY {c}\n#define DFM_PREG {}\n#define DFM_CREG {}\n{}",
        MMA_PRODUCER_REGISTERS,
        mma_consumer_registers(wm * wn),
        include_str!("../../../device/df/panel_mma.cu")
    )
}

/// The tensor-core kernel ([`PanelKernel::Mma`] or [`PanelKernel::MmaSm90a`]) for one tile
/// (bm, the largest bn, consumer warps along each, k-steps in flight): per orbital tile bn
/// a call uses (the fewest tiles of at most the largest bn over the chunk's orbitals, a
/// multiple of 8 per warp) and copy mode, compiled at first use. The stages are capped to
/// the device's shared memory per block (at least 2).
pub(crate) struct PanelMma {
    kind: PanelKernel,
    tile: [usize; 5],
    wmax: usize,
    variants: Vec<MmaKernel>,
}

/// A compiled tensor-core kernel: its orbital tile, copy mode, threads per block and dynamic
/// shared memory.
struct MmaKernel {
    bn: usize,
    copy: MmaCopy,
    kernel: Kernel,
    threads: usize,
    smem: usize,
}

impl PanelMma {
    /// The kernel `kind` with the tile option `opt` (zeros: [`MMA_TILE`]'s) on `gpu`: an
    /// unsupported card or tile is an error.
    pub(super) fn new(
        gpu: &Gpu,
        kind: PanelKernel,
        opt: [usize; 5],
        wmax: usize,
    ) -> Result<PanelMma, GpuError> {
        let tile: [usize; 5] =
            std::array::from_fn(|i| if opt[i] > 0 { opt[i] } else { MMA_TILE[i] });
        let [bm, bn, wm, wn, st] = tile;
        let cw = wm * wn;
        let threads = 32 * cw + 128;
        if bm % (16 * wm) != 0 || bn % (8 * wn) != 0 || st < 2 || threads > 1024 {
            return Err(GpuError::Contract(format!(
                "DF panels: tensor-core tile {tile:?}"
            )));
        }
        let cc = gpu.info().cc;
        match kind {
            PanelKernel::Mma if cc.0 >= 9 => {}
            PanelKernel::MmaSm90a if cc == (9, 0) => {
                // the hand-over only raises the consumers' registers: whole warpgroups of
                // consumers, starting below what they get
                if !cw.is_multiple_of(4) || launch_registers(threads) > mma_consumer_registers(cw) {
                    return Err(GpuError::Contract(format!(
                        "DF panels: the sm_90a kernel's tile {tile:?} has no register hand-over \
                         (consumer warps in whole warpgroups, fewer registers at launch than after)"
                    )));
                }
            }
            _ => {
                return Err(GpuError::NotCovered(format!(
                    "DF panels: the tensor-core kernel {kind:?} needs NVIDIA compute capability \
                     9.0 or later (sm_90a: 9.0); this device is {}.{}",
                    cc.0, cc.1
                )))
            }
        }
        Ok(PanelMma {
            kind,
            tile,
            wmax,
            variants: Vec::new(),
        })
    }

    /// The orbital tile for `nc` orbitals: the fewest tiles of at most the largest bn, each
    /// a multiple of 8 per warp.
    fn bn(&self, nc: usize) -> usize {
        let [_, bmax, _, wn, _] = self.tile;
        let step = 8 * wn;
        let nt = nc.div_ceil(bmax).max(1);
        nc.div_ceil(nt).div_ceil(step) * step
    }

    /// The kernel for orbital tile bn and copy mode `copy`.
    fn kernel(&mut self, gpu: &Gpu, bn: usize, copy: MmaCopy) -> Result<&MmaKernel, GpuError> {
        if let Some(i) = self
            .variants
            .iter()
            .position(|k| k.bn == bn && k.copy == copy)
        {
            return Ok(&self.variants[i]);
        }
        let [bm, _, wm, wn, st] = self.tile;
        // the operand stages, the stages' full and empty barriers, the row indices; the
        // stages capped to the card's shared memory per block (the RTX's 99 KiB holds 2 of
        // the default tile's 4; the same products in the same order either way)
        let rows = 4 * self.wmax.next_multiple_of(2);
        let smem_of = |s: usize| 8 * s * 16 * (bm + 4 + bn + 4) + 16 * s + rows;
        let cap = gpu.info().max_smem_per_block_optin as usize;
        let st = (2..=st).rev().find(|&s| smem_of(s) <= cap).unwrap_or(2);
        let smem = smem_of(st);
        let src = panel_mma_source([bm, bn, wm, wn, st], copy);
        let name = PANEL_MMA_MODULE.name;
        let kernel = match self.kind {
            PanelKernel::MmaSm90a => gpu.kernel_arch_specific(name, &src, "df_panel_mma", &[])?,
            _ => gpu.kernel_with(name, &src, "df_panel_mma", &[])?,
        };
        kernel.set_max_dynamic_shared(smem as u32)?;
        self.variants.push(MmaKernel {
            bn,
            copy,
            kernel,
            threads: 32 * wm * wn + 128,
            smem,
        });
        Ok(&self.variants[self.variants.len() - 1])
    }
}

/// The panels' first GEMM of one block for one chunk of factor columns (cmat columns cc..cc +
/// nc) in cuBLAS ([`PanelKernel::Strided`]): per shell s of `shells` (panel rows from `r0` at
/// `p_off` of `p`) one strided batch over its functions m = f0_s + ia,
/// X\[xo + q + nqb (i + nc m)\] = sum_k P\[p_off + (r0_s - r0 + ia + d_s k) nqb + q\]
/// Cg\[cc + i + ntot (c0_s + k)\]: the columns m of the (nc nqb) x nbf matrix the K update
/// multiplies. A shell without partners leaves its rows as they are.
#[allow(
    clippy::too_many_arguments,
    reason = "the GEMM's operands: the panels, the gathered orbitals and X, each with its layout"
)]
fn panel_gemm1(
    gemm: &Gemm,
    shells: &[PanelShell],
    (p, p_off, r0): (&DeviceBuf<f64>, usize, usize),
    nqb: usize,
    (cc, nc, ntot): (usize, usize, usize),
    cg: &DeviceBuf<f64>,
    (xt, xo): (&mut DeviceBuf<f64>, usize),
) -> Result<(), GpuError> {
    if nc == 0 {
        return Ok(());
    }
    for s in shells.iter().filter(|s| s.w > 0) {
        gemm.gemm_strided(
            Arith::Fp64,
            GemmDims::new(Op::N, Op::T, nqb, nc, s.w).batch(s.d),
            (1.0, 0.0),
            MatRef::new(p, p_off + nqb * (s.r0 - r0), nqb * s.d).stride(nqb),
            MatRef::new(cg, ntot * s.c0 + cc, ntot).stride(0),
            MatMut::new(&mut *xt, xo + nc * nqb * s.f0, nqb).stride(nqb * nc),
        )?;
    }
    Ok(())
}

impl GpuDfJk {
    /// The shell panels' blocks of a native K build ([`PanelBlocks`]), planned at the first
    /// panel build (`ntot` factor columns, `rs` range-separated) and kept: the setup's plan's
    /// block and factor columns when it planned the panel path (host storage: the streamed
    /// blocks); else (the exchange's arithmetic changed after the setup) the panel path's
    /// automatic block within the workspace the setup planned. cuBLAS's panels are gathered in
    /// shell groups of at most [`GpuDfJkOptions::k_panel_workspace`] bytes.
    pub(super) fn plan_panel_blocks(&mut self, ntot: usize, rs: bool) {
        if self.pblocks.is_some() {
            return;
        }
        let (nq, cols) = self.panel_shape(ntot, rs);
        let blocks = if nq == self.resolved.k_block {
            self.blocks.clone()
        } else {
            aux_blocks(self.naux, nq)
        };
        // shell groups whose gathered panels fit the workspace (a larger shell alone)
        let rows_cap = (self.k_panel_workspace / (8 * nq.max(1))).max(1);
        let mut groups = Vec::new();
        let sh = &self.panels.shells;
        let mut s0 = 0;
        while s0 < sh.len() {
            let r0 = sh[s0].r0;
            let mut s1 = s0 + 1;
            while s1 < sh.len() && sh[s1].r0 + sh[s1].d * sh[s1].w - r0 <= rows_cap {
                s1 += 1;
            }
            let r1 = sh[s1 - 1].r0 + sh[s1 - 1].d * sh[s1 - 1].w;
            groups.push((s0, s1, r0, r1));
            s0 = s1;
        }
        let pw_rows = if self.k_panel_kernel != PanelKernel::Strided {
            0
        } else {
            groups.iter().map(|g| g.3 - g.2).max().unwrap_or(0)
        };
        self.pblocks = Some(PanelBlocks {
            nq,
            blocks,
            cols,
            groups,
            pw_rows,
        });
    }

    /// Shrinks the panel path's automatic block one step before the first panel build (to the
    /// next lower whole tile of 128 functions from 256 up, else of 8, at least 8), and with it
    /// the record's `k_block`: for a caller that fits the builds to the memory the other engines
    /// leave once they are set up (linus's SCF check before its first build). The setup plans
    /// the block within the plan's budget, which counts the other engines by their reserve
    /// alone. False, nothing changed: the block was set by the caller, a tensor is streamed (its
    /// page-locked copy is laid out in blocks of this size), the exchange is not the planned
    /// panel path, a panel build ran, or the block is at 8.
    pub fn shrink_panel_block(&mut self) -> bool {
        let host = matches!(self.store, Store::Host(_))
            || self
                .lr
                .as_ref()
                .is_some_and(|l| matches!(l.store, Store::Host(_)));
        let q = self.resolved.k_block;
        if self.k_block_set
            || host
            || self.pblocks.is_some()
            || self.planned != self.exchange()
            || !matches!(self.planned, plan::DfExchange::Panels { .. })
            || q <= 8
        {
            return false;
        }
        let next = if q > 256 {
            ((q - 1) / 128 * 128).max(256)
        } else {
            ((q - 1) / 8 * 8).max(8)
        };
        self.resolved.k_block = next;
        self.blocks = aux_blocks(self.naux, next);
        true
    }

    /// The panel path's block and factor columns per pass (`ntot` columns at the first panel
    /// build, `rs` range-separated): the plan's when it planned this path or the storage is
    /// streamed, else the panel path's automatic block within the workspace the setup planned.
    pub(super) fn panel_shape(&self, ntot: usize, rs: bool) -> (usize, usize) {
        if let Some(pb) = &self.pblocks {
            return (pb.nq, pb.cols);
        }
        let panels = (self.k_panel_kernel, self.k_panel_workspace);
        let host = matches!(self.store, Store::Host(_))
            || self
                .lr
                .as_ref()
                .is_some_and(|l| matches!(l.store, Store::Host(_)));
        if host || self.planned == self.exchange() {
            return (self.resolved.k_block, self.k_columns);
        }
        let cols = ntot.max(1);
        let fits = |q: usize| {
            plan::panel_k_bytes(&self.sizes, panels, q, (cols, cols), rs) <= self.planned_work
        };
        let mut q = plan::panel_auto_block(&self.sizes, panels, cols, rs, 8);
        while q > 8 && !fits(q) {
            q = ((q - 1) / 8 * 8).max(8);
        }
        (q, cols)
    }

    /// Whether a build's exchange runs on the shell panels: native K in FP64, factors given.
    pub(super) fn panel_path(&self, c: &JkCall) -> bool {
        c.ntot > 0 && self.oz.is_none() && self.resolved.k_math == KMath::Native
    }

    /// The pass over the auxiliary blocks with the exchange on the shell panels
    /// ([`PanelPlan`], [`PanelBlocks`]): the orbitals gathered per shell once (Cg, timed as
    /// GEMM1); per panel block its rows (streamed from host storage) and its part of J (host
    /// storage); then per pass of factor columns (one pass unless a call has more columns
    /// than the blocks were planned for) X auxiliary-major per chunk of a factor's columns
    /// (X\[q + nqb (i + nc m)\]), from the tensor-core kernel reading the stored rows or
    /// from cuBLAS on the panels gathered per shell group (the gathers timed as the
    /// unpack), and the K update ([`k_update`]: X^T X over the lower triangle's column
    /// blocks; range-separated: Xw from B^omega's panels, W = (a X + b Xw) / 2 in place,
    /// K (+)= X^T W in full). The device is waited for at the end.
    pub(super) fn panel_pass(
        &mut self,
        c: &mut JkCall,
        jx: Option<&mut JRows>,
        k: &mut DeviceBuf<f64>,
    ) -> Result<(), GpuError> {
        let host_mode = matches!(self.store, Store::Host(_));
        let (nd, ntot) = (c.nd, c.ntot);
        let cm = c
            .cmat
            .ok_or_else(|| GpuError::Contract("DF-J/K: no factors".into()))?;
        // planned for B^omega's Xwt whenever B^omega is held
        self.plan_panel_blocks(ntot, self.lr.is_some());
        let tabs = match self.ptabs.take() {
            Some(t) => t,
            None => {
                let t = PanelTables::new(c.gpu, &self.panels)?;
                self.resolved.device_bytes += t.bytes();
                t
            }
        };
        let pb = self
            .pblocks
            .take()
            .ok_or_else(|| GpuError::Contract("DF-J/K: no panel blocks".into()))?;
        let mut mma = self.mma.take();
        let r = self.panel_blocks_pass(c, (&pb, &tabs), mma.as_mut(), (jx, k), (host_mode, nd, cm));
        (self.pblocks, self.mma, self.ptabs) = (Some(pb), mma, Some(tabs));
        r?;
        c.flush()
    }

    /// [`GpuDfJk::panel_pass`]'s work with the panel blocks `pb`, the panels' tables and the tensor-core kernel
    /// `mma` (None: cuBLAS on the gathered panels) taken out of the engine.
    fn panel_blocks_pass(
        &mut self,
        c: &mut JkCall,
        (pb, tabs): (&PanelBlocks, &PanelTables),
        mut mma: Option<&mut PanelMma>,
        (mut jx, k): (Option<&mut JRows>, &mut DeviceBuf<f64>),
        (host_mode, nd, cm): (bool, usize, &DeviceBuf<f64>),
    ) -> Result<(), GpuError> {
        let (gpu, nbf, ntot) = (c.gpu, c.nbf, c.ntot);
        let n2 = nbf * nbf;
        let ncg = tabs.ncg();
        let nq = pb.nq;
        // the factors' columns in chunks of at most pb.cols (factor, first cmat column,
        // columns, first chunk of its factor), in passes of at most pb.cols columns
        let mut chunks = Vec::new();
        let mut c0 = 0;
        for (d, &nc) in c.ncols.iter().enumerate() {
            let mut i0 = 0;
            while i0 < nc {
                let w = (nc - i0).min(pb.cols);
                chunks.push((d, c0 + i0, w, i0 == 0));
                i0 += w;
            }
            c0 += nc;
        }
        let mut passes: Vec<Vec<(usize, usize, usize, bool)>> = Vec::new();
        for ch in chunks {
            match passes.last_mut() {
                Some(p) if p.iter().map(|x| x.2).sum::<usize>() + ch.2 <= pb.cols => p.push(ch),
                _ => passes.push(vec![ch]),
            }
        }
        let width = passes
            .iter()
            .map(|p| p.iter().map(|x| x.2).sum::<usize>())
            .max()
            .unwrap_or(0);
        let mut pw = if mma.is_none() {
            Some(gpu.alloc_zeros::<f64>((pb.pw_rows * nq).max(1))?)
        } else {
            None
        };
        let mut xt = gpu.alloc_zeros::<f64>(nq * nbf * width)?;
        let mut xwt = if c.lr_coefs.is_some() {
            Some(gpu.alloc_zeros::<f64>(nq * nbf * width)?)
        } else {
            None
        };
        let mut gq = if host_mode && nd > 0 {
            Some(gpu.alloc_zeros::<f64>(nq * nd)?)
        } else {
            None
        };
        let keep_fit = self.keep_fit && gq.is_some();
        if keep_fit {
            self.coulomb_fit_buf(gpu, nd)?;
        }
        // the orbitals gathered per shell's partner functions, once per build: rows of ldc
        // doubles (the tensor-core kernel's rows start 16-byte aligned: ldc even, the
        // padding zero)
        let ldc = if mma.is_some() {
            ntot.next_multiple_of(2)
        } else {
            ntot
        };
        let s = c.tl.mark()?;
        let mut cg = gpu.alloc_zeros::<f64>((ncg * ldc).max(1))?;
        // SAFETY: matches df_panel_cg; reads ncg entries of cn and nbf x ntot of cm (cn's
        // functions < nbf), writes ncg rows of ntot values of cg at stride ldc >= ntot.
        unsafe {
            gpu.launch(
                &self.kernels.panel_cg,
                Launch::linear(ncg * ntot, 256),
                &[
                    Arg::I64(ncg as i64),
                    Arg::I32(ntot as i32),
                    Arg::I32(ldc as i32),
                    Arg::I64(nbf as i64),
                    Arg::buf(&tabs.cn),
                    Arg::buf(cm),
                    Arg::buf_mut(&mut cg),
                ],
            )
        }?;
        c.tl.close(|t| &mut t.k_gemm1, s)?;
        let nkb = nbf
            .div_ceil(self.k_update_block)
            .clamp(1, K_UPDATE_BLOCKS_MAX);
        for (bi, &(q0, nqb)) in pb.blocks.iter().enumerate() {
            let src = self.store.block(c, q0, nqb)?;
            if let (true, Some(rows), Some(g)) = (host_mode, jx.as_deref_mut(), gq.as_mut()) {
                let fit = self.coulomb_fit.as_mut().filter(|_| keep_fit);
                let beta = if bi == 0 { 0.0 } else { 1.0 };
                rows.add_block(&self.gemm, c, &src, g, fit, beta)?;
            }
            let wsrc = match (c.lr_coefs, self.lr.as_mut()) {
                (Some(_), Some(lr)) => Some(lr.store.block(c, q0, nqb)?),
                _ => None,
            };
            for pass in &passes {
                // GEMM1 of every chunk of the pass, from B's panels into Xt and B^omega's
                // into Xwt
                for (t, blk) in [Some(&src), wsrc.as_ref()].into_iter().enumerate() {
                    let Some(blk) = blk else { continue };
                    let out = match (t, xwt.as_mut()) {
                        (0, _) => &mut xt,
                        (_, Some(xw)) => xw,
                        (_, None) => {
                            return Err(GpuError::Contract(
                                "DF-J/K: B^omega's panels without Xw".into(),
                            ))
                        }
                    };
                    if let Some(kern) = mma.as_deref_mut() {
                        let s1 = c.tl.mark()?;
                        let mut xo = 0;
                        for &(_, cc, nc, _) in pass {
                            panel_mma(
                                gpu,
                                kern,
                                tabs,
                                blk,
                                (cc, nc, ldc, nbf),
                                &cg,
                                (&mut *out, xo),
                            )?;
                            xo += nc * nqb * nbf;
                        }
                        let s2 = c.tl.mark()?;
                        c.tl.span(|t| &mut t.k_gemm1, s1, s2);
                        continue;
                    }
                    let gemm1 = |p: (&DeviceBuf<f64>, usize),
                                 shells: &[PanelShell],
                                 r0: usize,
                                 out: &mut DeviceBuf<f64>|
                     -> Result<(), GpuError> {
                        let mut xo = 0;
                        for &(_, cc, nc, _) in pass {
                            panel_gemm1(
                                &self.gemm,
                                shells,
                                (p.0, p.1, r0),
                                nqb,
                                (cc, nc, ntot),
                                &cg,
                                (&mut *out, xo),
                            )?;
                            xo += nc * nqb * nbf;
                        }
                        Ok(())
                    };
                    let pw = pw
                        .as_mut()
                        .ok_or_else(|| GpuError::Contract("DF-J/K: no panel workspace".into()))?;
                    for &(s0, s1, r0, r1) in &pb.groups {
                        let m0 = c.tl.mark()?;
                        panel_gather(gpu, &self.kernels, tabs, (r0, r1 - r0), blk, (&mut *pw, 0))?;
                        let m1 = c.tl.mark()?;
                        gemm1((&*pw, 0), &tabs.shells[s0..s1], r0, out)?;
                        let m2 = c.tl.mark()?;
                        c.tl.span(|t| &mut t.k_unpack, m0, m1);
                        c.tl.span(|t| &mut t.k_gemm1, m1, m2);
                    }
                }
                let s2 = c.tl.mark()?;
                if let (Some((fa, fb)), Some(xwt)) = (c.lr_coefs, xwt.as_mut()) {
                    // Wt = (a Xt + b Xwt) / 2 in place of Xwt (the rounding of the unpacked
                    // path's W), then K_d (+)= Xt^T Wt in full per chunk
                    let len: usize = pass.iter().map(|x| x.2).sum::<usize>() * nqb * nbf;
                    self.kernels.blas1.axpby(
                        gpu,
                        len,
                        &[(0.5 * fb, Operand::Out), (0.5 * fa, Operand::Buf(&xt))],
                        xwt,
                    )?;
                    // in two column blocks of K (full height): one square product with a long
                    // inner dimension left the H200's SMs idle in its last wave (ts113
                    // GEMM2 319 against 263 ms; two blocks within 2 % of the best of one,
                    // two, four and eight at u1an39, ts61 and ts113)
                    let mut xo = 0;
                    for &(d, _, nc, first) in pass {
                        let beta = if bi == 0 && first { 0.0 } else { 1.0 };
                        let l = nc * nqb;
                        let op = KOperand {
                            x: &xt,
                            w: Some(&*xwt),
                            off: xo,
                            l,
                            trans: true,
                        };
                        k_update(&self.gemm, Arith::Fp64, op, (&mut *k, d * n2), nbf, beta, 2)?;
                        xo += l * nbf;
                    }
                } else {
                    // K_d (+)= X_d^T X_d, the lower triangle over column blocks, mirrored at
                    // the end (one GEMM per block: cuBLAS's grouped GEMM and a split inner
                    // dimension measured slower on the H200)
                    let mut xo = 0;
                    for &(d, _, nc, first) in pass {
                        let beta = if bi == 0 && first { 0.0 } else { 1.0 };
                        let l = nc * nqb;
                        let op = KOperand {
                            x: &xt,
                            w: None,
                            off: xo,
                            l,
                            trans: true,
                        };
                        k_update(
                            &self.gemm,
                            Arith::Fp64,
                            op,
                            (&mut *k, d * n2),
                            nbf,
                            beta,
                            nkb,
                        )?;
                        xo += l * nbf;
                    }
                }
                c.tl.close(|t| &mut t.k_gemm2, s2)?;
            }
        }
        if keep_fit {
            self.coulomb_fit_nd = nd;
        }
        Ok(())
    }
}
