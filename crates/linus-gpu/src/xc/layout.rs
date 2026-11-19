//! Per-batch screening for the XC engines (feature `gpu`): the CPU integrator's function lists
//! per grid batch on the device, and the kernels that work on them.
//!
//! linus's CPU integrator keeps, per grid batch (~128-256 points), the functions whose
//! largest value or derivative over the batch exceeds `func_thresh` (linus-ints' `Collocator`
//! value test). The device engines collocate a block of consecutive batches at once (the
//! shells whose rigorous bound reaches the block's box), so a block-level GEMM multiplies the
//! values of functions that are negligible on most of the block's batches: 1.6-2x the flops of
//! per-batch products on u1an39 and ts61 (def2-TZVP).
//!
//! [`XcLayout`] finds the per-batch lists once per geometry, on the device: each block is
//! collocated and a kernel takes the CPU's value test per batch and function (`xcb_flags`).
//! A call then collocates each block as before, copies every batch's kept functions into a
//! compressed per-batch array (`xcb_compress`), and runs the per-batch products of a chunk of
//! batches in one launch of the variable-size batched GEMM ([`crate::kernels::vgemm`]). The
//! potential of each batch is added to a fixed-point accumulator (`detsum`): exact and
//! independent of the order of the batches.
//!
//! Batch descriptor (`BD` i64 per batch, `bd[BD j + field]`): first grid point, points,
//! kept functions, first point within the block, offset of its function lists, offsets of its
//! compressed values (phi), of its point x function arrays (X, W) and of its function x
//! function arrays (D, V) within its chunk's workspaces.

use crate::cpu::MolecularGrid;
use crate::xc::XcBlock;

/// i64 fields per batch descriptor.
pub(crate) const BD: usize = 9;
pub(crate) const BD_P0: usize = 0;
pub(crate) const BD_NP: usize = 1;
pub(crate) const BD_NF: usize = 2;
pub(crate) const BD_Q0: usize = 3;
pub(crate) const BD_FL: usize = 4;
pub(crate) const BD_PHI: usize = 5;
pub(crate) const BD_X: usize = 6;
pub(crate) const BD_D: usize = 7;

/// A run of consecutive blocks whose compressed arrays fit the workspace.
#[derive(Clone, Debug)]
pub(crate) struct Chunk {
    pub(crate) b0: usize,
    pub(crate) b1: usize,
    /// Its batches (global batch numbers).
    pub(crate) j0: usize,
    pub(crate) j1: usize,
    /// Largest points in one of its batches.
    pub(crate) max_np: usize,
    /// Sizes of its workspaces: sum of nf np, sum of nf^2.
    pub(crate) x_len: usize,
    pub(crate) d_len: usize,
}

/// One batch on the host.
#[derive(Clone, Debug)]
pub(crate) struct HostBatch {
    pub(crate) p0: usize,
    pub(crate) np: usize,
    pub(crate) q0: usize,
    /// Kept functions: block-local numbers and global numbers (ascending).
    pub(crate) local: Vec<i32>,
    pub(crate) global: Vec<i32>,
}

/// Splits the blocks' points into their grid batches: (block, first point in block, points)
/// per batch, in order.
pub(crate) fn split_batches(grid: &MolecularGrid, blocks: &[XcBlock]) -> Vec<Vec<(usize, usize)>> {
    let mut out = Vec::with_capacity(blocks.len());
    let mut bi = 0;
    for b in blocks {
        let mut v = Vec::new();
        while bi < grid.batches.len() && grid.batches[bi].start < b.p0 + b.np {
            let ba = &grid.batches[bi];
            if ba.end > ba.start {
                v.push((ba.start - b.p0, ba.end - ba.start));
            }
            bi += 1;
        }
        out.push(v);
    }
    out
}

/// Chunks of consecutive blocks within `budget` doubles of workspace per point x function
/// array (sum over the chunk of ncomp nf np, the largest array); a block larger than the
/// budget is a chunk of its own.
pub(crate) fn chunks(
    batches: &[HostBatch],
    block_batches: &[(usize, usize)],
    ncomp: usize,
    budget: usize,
) -> Vec<Chunk> {
    let mut out = Vec::new();
    let mut b = 0;
    while b < block_batches.len() {
        let j0 = block_batches[b].0;
        let (mut x_len, mut d_len, mut max_np) = (0, 0, 0);
        let mut e = b;
        while e < block_batches.len() {
            let (bj0, bj1) = block_batches[e];
            let (bx, bd, bnp) = batches[bj0..bj1].iter().fold((0, 0, 0), |a, h| {
                let nf = h.global.len();
                (a.0 + nf * h.np, a.1 + nf * nf, a.2.max(h.np))
            });
            if e > b && ncomp * (x_len + bx) > budget {
                break;
            }
            x_len += bx;
            d_len += bd;
            max_np = max_np.max(bnp);
            e += 1;
        }
        out.push(Chunk {
            b0: b,
            b1: e,
            j0,
            j1: block_batches[e - 1].1,
            max_np,
            x_len,
            d_len,
        });
        b = e;
    }
    out
}

/// The batch descriptors (BD per batch; offsets within each batch's chunk) and the function
/// lists (local, global) of `batches` for `chunks` with `ncomp` components of compressed
/// values.
pub(crate) fn descriptors(
    batches: &[HostBatch],
    chunks: &[Chunk],
    ncomp: usize,
) -> (Vec<i64>, Vec<i32>, Vec<i32>) {
    let mut bd = vec![0i64; BD * batches.len()];
    let (mut lf, mut gf) = (Vec::new(), Vec::new());
    for c in chunks {
        let (mut xo, mut dd) = (0usize, 0usize);
        for j in c.j0..c.j1 {
            let h = &batches[j];
            let nf = h.global.len();
            let d = &mut bd[BD * j..BD * (j + 1)];
            d[BD_P0] = h.p0 as i64;
            d[BD_NP] = h.np as i64;
            d[BD_NF] = nf as i64;
            d[BD_Q0] = h.q0 as i64;
            d[BD_FL] = lf.len() as i64;
            d[BD_PHI] = (ncomp * xo) as i64;
            d[BD_X] = xo as i64;
            d[BD_D] = dd as i64;
            lf.extend_from_slice(&h.local);
            gf.extend_from_slice(&h.global);
            xo += nf * h.np;
            dd += nf * nf;
        }
    }
    (bd, lf, gf)
}

/// Screening statistics of a layout: sum over batches of points x functions (values per
/// pass) and of points x functions^2 (the work of the products).
pub(crate) fn work(batches: &[HostBatch]) -> (usize, f64) {
    batches.iter().fold((0, 0.0), |a, h| {
        let nf = h.global.len();
        (a.0 + nf * h.np, a.1 + (nf * nf * h.np) as f64)
    })
}

#[cfg(feature = "gpu")]
pub(crate) use device::*;

#[cfg(feature = "gpu")]
mod device {
    use super::*;
    use crate::cpu::BasisSet;
    use crate::error::GpuError;
    use crate::kernels::colloc::{self, ShellTables};
    use crate::kernels::registry::KernelModule;
    use crate::kernels::vgemm::{Vgemm, VgemmPlan, VgemmProduct, VgemmTile};
    use crate::{Arg, DevPtr, DeviceBuf, Gpu, Kernel, Launch};
    use std::borrow::Cow;

    const KERNELS: &str = include_str!("../../device/xc/batched.cu");

    /// The per-batch module: the fixed-point accumulator, then the kernels above.
    pub(crate) const MODULE: KernelModule = KernelModule::new(
        "xc_batched",
        || Cow::Owned(format!("{}\n{}", crate::detsum::DEVICE_SRC, KERNELS)),
        &[
            "xcb_flags",
            "xcb_compress",
            "xcb_gather_d",
            "xcb_densities",
            "xcb_weights",
            "xcb_weights_tau",
            "xcb_vscatter",
            "xcb_vfinal",
        ],
    );

    pub(crate) struct Kernels {
        pub(crate) flags: Kernel,
        pub(crate) compress: Kernel,
        pub(crate) gather_d: Kernel,
        pub(crate) densities: Kernel,
        pub(crate) weights: Kernel,
        pub(crate) weights_tau: Kernel,
        pub(crate) vscatter: Kernel,
        pub(crate) vfinal: Kernel,
    }

    impl Kernels {
        pub(crate) fn new(gpu: &Gpu) -> Result<Kernels, GpuError> {
            let src = MODULE.source();
            let k = |e: &str| gpu.kernel(MODULE.name, &src, e);
            Ok(Kernels {
                flags: k("xcb_flags")?,
                compress: k("xcb_compress")?,
                gather_d: k("xcb_gather_d")?,
                densities: k("xcb_densities")?,
                weights: k("xcb_weights")?,
                weights_tau: k("xcb_weights_tau")?,
                vscatter: k("xcb_vscatter")?,
                vfinal: k("xcb_vfinal")?,
            })
        }
    }

    /// A launch over the batches of a chunk: x covers `per` elements of the largest batch
    /// (grid-stride beyond 4096 blocks), y the batches.
    pub(crate) fn per_batch(per: usize, nbatch: usize, threads: u32) -> Launch {
        let bx = per.div_ceil(threads as usize).clamp(1, 4096);
        Launch {
            grid: (bx as u32, nbatch as u32, 1),
            block: (threads, 1, 1),
            shared_bytes: 0,
        }
    }

    /// A block on the device: its collocation work (as `GpuXc`'s blocks) and its batches.
    pub(crate) struct LBlock {
        pub(crate) p0: usize,
        pub(crate) np: usize,
        pub(crate) nf: usize,
        /// (colloc kernel index, shells, offset in the flat lists)
        pub(crate) groups: Vec<(usize, usize, usize)>,
        /// Its batches (global batch numbers).
        pub(crate) j0: usize,
        pub(crate) j1: usize,
    }

    /// What an [`XcLayout`] is built for: the functions kept per batch are those whose values
    /// and derivatives to `screen_order` exceed `thresh` somewhere on it, collocated in blocks
    /// of `block_points` (whole batches, shells by the rigorous bound at `screen_order`) and
    /// compressed to collocation order `order` (<= `screen_order`), in chunks of at most
    /// `budget` doubles of compressed values.
    #[derive(Clone, Copy, Debug)]
    pub(crate) struct XcLayoutSpec {
        pub(crate) screen_order: usize,
        pub(crate) order: usize,
        pub(crate) thresh: f64,
        pub(crate) block_points: usize,
        pub(crate) budget: usize,
    }

    /// How a chunk's blocks are collocated: at the layout's own order, or with the kernels of
    /// another order and their components (from [`XcLayout::kernels_of_order`]; the gradient's
    /// and the Hessian's first-order values on their higher-order layouts).
    #[derive(Clone, Copy)]
    pub(crate) enum Collocation<'a> {
        Own,
        With(&'a [Kernel], usize),
    }

    /// The per-batch screening of a grid for one collocation order and threshold (see the
    /// module docs), with the device tables and the kernels that work on it: the unit the XC
    /// engines (V_xc, the gradient, the Hessian terms, the CPKS kernel) are composed on.
    pub(crate) struct XcLayout {
        /// Components of the compressed values (the collocation order's).
        pub(crate) ncomp: usize,
        pub(crate) blocks: Vec<LBlock>,
        pub(crate) chunks: Vec<Chunk>,
        pub(crate) host: Vec<HostBatch>,
        pub(crate) bd: DeviceBuf<i64>,
        pub(crate) lf: DeviceBuf<i32>,
        pub(crate) gf: DeviceBuf<i32>,
        /// Collocation of a block: kernels per (l, pure), shell lists and local first
        /// functions, the shell tables.
        colloc: Vec<Kernel>,
        /// The basis's (l, pure) classes, in the kernels' order.
        keys: Vec<(usize, bool)>,
        lists: DeviceBuf<i32>,
        aos: DeviceBuf<i32>,
        tables: ShellTables,
        pub(crate) max_block_nf: usize,
        pub(crate) max_block_np: usize,
        /// Largest workspaces over the chunks: sum nf np, sum nf^2.
        pub(crate) max_x: usize,
        pub(crate) max_d: usize,
        /// Sum over batches of points x kept functions (values per pass).
        pub(crate) values: usize,
        /// The per-batch kernels and the variable-size batched GEMM of the engines on it.
        pub(crate) k: Kernels,
        pub(crate) vg: Vgemm,
        /// Setup seconds: the blocks (host), the per-batch value test on the device (tables,
        /// kernels, collocation, flags, their download), the batches' lists, chunks and
        /// descriptors (host) with their uploads.
        pub(crate) setup_times: [f64; 3],
    }

    impl XcLayout {
        /// The layout of `grid` for `basis` by `spec`.
        pub(crate) fn new(
            gpu: &Gpu,
            basis: &BasisSet,
            grid: &MolecularGrid,
            spec: &XcLayoutSpec,
        ) -> Result<XcLayout, GpuError> {
            let XcLayoutSpec {
                screen_order,
                order,
                thresh,
                block_points,
                budget,
            } = *spec;
            let t0 = std::time::Instant::now();
            let bl = crate::xc::blocks(basis, grid, screen_order, thresh, block_points.max(1))?;
            let t_blocks = t0.elapsed().as_secs_f64();
            let split = split_batches(grid, &bl);
            let mut keys: Vec<(usize, bool)> = basis.shells.iter().map(|s| (s.l, s.pure)).collect();
            keys.sort();
            keys.dedup();
            let key_of = |s: usize| {
                let sh = &basis.shells[s];
                keys.iter().position(|k| *k == (sh.l, sh.pure)).unwrap()
            };
            let (mut lists, mut aos) = (Vec::<i32>::new(), Vec::<i32>::new());
            let mut lblocks = Vec::with_capacity(bl.len());
            let (mut max_nf, mut max_np) = (0, 0);
            let mut nbatch = 0;
            for (b, sp) in bl.iter().zip(&split) {
                let mut groups = Vec::new();
                for (ki, _) in keys.iter().enumerate() {
                    let off = lists.len();
                    for (s, &a) in b.shells.iter().zip(&b.ao_local) {
                        if key_of(*s) == ki {
                            lists.push(*s as i32);
                            aos.push(a as i32);
                        }
                    }
                    if lists.len() > off {
                        groups.push((ki, lists.len() - off, off));
                    }
                }
                max_nf = max_nf.max(b.funcs.len());
                max_np = max_np.max(b.np);
                lblocks.push(LBlock {
                    p0: b.p0,
                    np: b.np,
                    nf: b.funcs.len(),
                    groups,
                    j0: nbatch,
                    j1: nbatch + sp.len(),
                });
                nbatch += sp.len();
            }
            let lists_d = gpu.upload(if lists.is_empty() { &[0] } else { &lists })?;
            let aos_d = gpu.upload(if aos.is_empty() { &[0] } else { &aos })?;
            let tables = ShellTables::new(gpu, basis)?;
            // the value test at the screening order, on the device
            let ksrc = colloc::module_source(basis, screen_order);
            let colloc_s = keys
                .iter()
                .map(|&(l, p)| {
                    gpu.kernel("colloc", &ksrc, &colloc::kernel_name(l, p, screen_order))
                })
                .collect::<Result<Vec<_>, _>>()?;
            let ncs = crate::cpu::ncomp(screen_order);
            let kern = Kernels::new(gpu)?;
            // provisional descriptors: points and block offsets only
            let mut bd0 = vec![0i64; BD * nbatch.max(1)];
            for (b, (lb, sp)) in lblocks.iter().zip(&split).enumerate() {
                let _ = b;
                for (i, &(q0, np)) in sp.iter().enumerate() {
                    let d = &mut bd0[BD * (lb.j0 + i)..BD * (lb.j0 + i + 1)];
                    d[BD_P0] = (lb.p0 + q0) as i64;
                    d[BD_NP] = np as i64;
                    d[BD_Q0] = q0 as i64;
                }
            }
            let bd_d0 = gpu.upload(&bd0)?;
            let fstride = max_nf.max(1);
            let mut flags = gpu.alloc_zeros::<u8>((nbatch * fstride).max(1))?;
            let x = gpu.upload(&grid.x)?;
            let y = gpu.upload(&grid.y)?;
            let z = gpu.upload(&grid.z)?;
            let phi = gpu.alloc_zeros::<f64>((max_nf * ncs * max_np).max(1))?;
            let ws = gpu.warp_size() as usize;
            for lb in &lblocks {
                if lb.nf == 0 || lb.j1 == lb.j0 {
                    continue;
                }
                let at = |buf: &DeviceBuf<f64>| buf.at(lb.p0);
                for &(ki, nsh, off) in &lb.groups {
                    // SAFETY: the block's shells of this kernel's (l, pure) with their local
                    // first functions (< nf); the output holds nf ncs np values.
                    unsafe {
                        tables.launch(
                            gpu,
                            &colloc_s[ki],
                            nsh,
                            lists_d.at(off),
                            aos_d.at(off),
                            [at(&x), at(&y), at(&z)],
                            lb.np,
                            ncs * lb.np,
                            lb.np,
                            phi.ptr(),
                        )
                    }?;
                }
                let per_block = 256 / ws;
                let cfg = Launch {
                    grid: (lb.nf.div_ceil(per_block) as u32, (lb.j1 - lb.j0) as u32, 1),
                    block: (256, 1, 1),
                    shared_bytes: 0,
                };
                // SAFETY: matches xcb_flags; reads the block's collocation (nf ncs np) at its
                // batches' points, writes flags of batches j0.. (< nbatch) and functions < nf
                // (<= fstride).
                unsafe {
                    gpu.launch(
                        &kern.flags,
                        cfg,
                        &[
                            Arg::I32(lb.nf as i32),
                            Arg::I32(ncs as i32),
                            Arg::I32(lb.np as i32),
                            Arg::F64(thresh),
                            Arg::buf(&bd_d0),
                            Arg::I32(lb.j0 as i32),
                            Arg::I64(fstride as i64),
                            Arg::buf(&phi),
                            Arg::buf_mut(&mut flags),
                        ],
                    )
                }?;
            }
            let hf = gpu.download(&flags)?;
            let t_screen = t0.elapsed().as_secs_f64();
            drop((phi, flags, bd_d0));
            // the kept functions per batch
            let mut host = Vec::with_capacity(nbatch);
            for (b, (lb, sp)) in bl.iter().zip(lblocks.iter().zip(&split)) {
                for (i, &(q0, np)) in sp.iter().enumerate() {
                    let j = lb.j0 + i;
                    let (mut local, mut global) = (Vec::new(), Vec::new());
                    for (f, &gfun) in b.funcs.iter().enumerate() {
                        if hf[j * fstride + f] != 0 {
                            local.push(f as i32);
                            global.push(gfun as i32);
                        }
                    }
                    host.push(HostBatch {
                        p0: lb.p0 + q0,
                        np,
                        q0,
                        local,
                        global,
                    });
                }
            }
            let ncomp = crate::cpu::ncomp(order);
            let bb: Vec<(usize, usize)> = lblocks.iter().map(|b| (b.j0, b.j1)).collect();
            let ch = chunks(&host, &bb, ncomp, budget);
            let (bd, lf, gf) = descriptors(&host, &ch, ncomp);
            let (values, _) = work(&host);
            let colloc_k = if order == screen_order {
                colloc_s
            } else {
                let src = colloc::module_source(basis, order);
                keys.iter()
                    .map(|&(l, p)| gpu.kernel("colloc", &src, &colloc::kernel_name(l, p, order)))
                    .collect::<Result<Vec<_>, _>>()?
            };
            Ok(XcLayout {
                ncomp,
                max_x: ch.iter().map(|c| c.x_len).max().unwrap_or(0),
                max_d: ch.iter().map(|c| c.d_len).max().unwrap_or(0),
                blocks: lblocks,
                chunks: ch,
                host,
                bd: gpu.upload(if bd.is_empty() { &[0] } else { &bd })?,
                lf: gpu.upload(if lf.is_empty() { &[0] } else { &lf })?,
                gf: gpu.upload(if gf.is_empty() { &[0] } else { &gf })?,
                colloc: colloc_k,
                keys,
                lists: lists_d,
                aos: aos_d,
                tables,
                max_block_nf: max_nf,
                max_block_np: max_np,
                values,
                k: kern,
                vg: vgemm(gpu)?,
                setup_times: [
                    t_blocks,
                    t_screen - t_blocks,
                    t0.elapsed().as_secs_f64() - t_screen,
                ],
            })
        }

        /// Collocation kernels of the basis's (l, pure) classes at derivative `order` (for
        /// [`XcLayout::collocate_with`]).
        pub(crate) fn kernels_of_order(
            &self,
            gpu: &Gpu,
            basis: &BasisSet,
            order: usize,
        ) -> Result<Vec<Kernel>, GpuError> {
            let src = colloc::module_source(basis, order);
            self.keys
                .iter()
                .map(|&(l, p)| gpu.kernel("colloc", &src, &colloc::kernel_name(l, p, order)))
                .collect()
        }

        /// Collocates block `b` (all its functions, the layout's order) into `phi` at the
        /// grid coordinates x, y, z (device buffers of the whole grid).
        pub(crate) fn collocate(
            &self,
            gpu: &Gpu,
            b: &LBlock,
            xyz: [&DeviceBuf<f64>; 3],
            phi: &mut DeviceBuf<f64>,
        ) -> Result<(), GpuError> {
            self.collocate_with(gpu, &self.colloc, self.ncomp, b, xyz, phi)
        }

        /// [`XcLayout::collocate`] with the kernels `kern` of another derivative order
        /// (`ncomp` components; from [`XcLayout::kernels_of_order`]).
        pub(crate) fn collocate_with(
            &self,
            gpu: &Gpu,
            kern: &[Kernel],
            ncomp: usize,
            b: &LBlock,
            xyz: [&DeviceBuf<f64>; 3],
            phi: &mut DeviceBuf<f64>,
        ) -> Result<(), GpuError> {
            let at = |buf: &DeviceBuf<f64>| buf.at(b.p0);
            if ncomp * b.nf * b.np > phi.len() || kern.len() != self.keys.len() {
                return Err(GpuError::Contract(
                    "XC: collocation buffer too small".into(),
                ));
            }
            for &(ki, nsh, off) in &b.groups {
                // SAFETY: the block's shells of this kernel's (l, pure) with their local first
                // functions (< nf); the output holds nf ncomp np values (checked).
                unsafe {
                    self.tables.launch(
                        gpu,
                        &kern[ki],
                        nsh,
                        self.lists.at(off),
                        self.aos.at(off),
                        [at(xyz[0]), at(xyz[1]), at(xyz[2])],
                        b.np,
                        ncomp * b.np,
                        b.np,
                        phi.ptr(),
                    )
                }?;
            }
            Ok(())
        }

        /// Copies block `b`'s batches' kept functions from its collocation `phi` into the
        /// compressed array `phic` (f64) and/or `phic32` (f32) of its chunk.
        pub(crate) fn compress(
            &self,
            gpu: &Gpu,
            b: &LBlock,
            phi: &DeviceBuf<f64>,
            phic: Option<&mut DeviceBuf<f64>>,
            phic32: Option<&mut DeviceBuf<f32>>,
        ) -> Result<(), GpuError> {
            self.compress_nc(gpu, self.ncomp, b, phi, phic, phic32)
        }

        /// [`XcLayout::compress`] of a collocation with `nc` (<= the layout's) components.
        pub(crate) fn compress_nc(
            &self,
            gpu: &Gpu,
            nc: usize,
            b: &LBlock,
            phi: &DeviceBuf<f64>,
            phic: Option<&mut DeviceBuf<f64>>,
            phic32: Option<&mut DeviceBuf<f32>>,
        ) -> Result<(), GpuError> {
            if b.j1 == b.j0 {
                return Ok(());
            }
            assert!(nc <= self.ncomp);
            let per = (b.j0..b.j1)
                .map(|j| nc * self.host[j].global.len() * self.host[j].np)
                .max()
                .unwrap_or(0);
            if per == 0 {
                return Ok(());
            }
            let p64 = phic.map_or(DevPtr::null(), |p| p.ptr());
            let p32 = phic32.map_or(DevPtr::null(), |p| p.ptr());
            // SAFETY: matches xcb_compress; reads the block's collocation (nf ncomp np) at its
            // batches' kept functions (< nf) and points, writes each batch's compressed values
            // at its offset within the chunk's workspaces (sized for the largest chunk).
            unsafe {
                gpu.launch(
                    &self.k.compress,
                    per_batch(per, b.j1 - b.j0, 256),
                    &[
                        Arg::I32(nc as i32),
                        Arg::I32(b.np as i32),
                        Arg::buf(&self.bd),
                        Arg::I32(b.j0 as i32),
                        Arg::buf(&self.lf),
                        Arg::buf(phi),
                        p64.into(),
                        p32.into(),
                    ],
                )
            }
        }

        /// The largest nf^2 and nf np over the batches of chunk `c`.
        pub(crate) fn chunk_max(&self, c: &Chunk) -> (usize, usize) {
            self.host[c.j0..c.j1].iter().fold((0, 0), |a, h| {
                let nf = h.global.len();
                (a.0.max(nf * nf), a.1.max(nf * h.np))
            })
        }

        /// Collocates every block of chunk `c` with functions (`colloc`) into `phi` at the grid
        /// coordinates `xyz`, and compresses each into the chunk's `phic` and/or `phic32`.
        #[allow(clippy::too_many_arguments)]
        pub(crate) fn prepare(
            &self,
            gpu: &Gpu,
            c: &Chunk,
            colloc: Collocation<'_>,
            xyz: [&DeviceBuf<f64>; 3],
            phi: &mut DeviceBuf<f64>,
            mut phic: Option<&mut DeviceBuf<f64>>,
            mut phic32: Option<&mut DeviceBuf<f32>>,
        ) -> Result<(), GpuError> {
            for b in &self.blocks[c.b0..c.b1] {
                if b.nf == 0 {
                    continue;
                }
                let nc = match colloc {
                    Collocation::Own => {
                        self.collocate(gpu, b, xyz, phi)?;
                        self.ncomp
                    }
                    Collocation::With(kern, nc) => {
                        self.collocate_with(gpu, kern, nc, b, xyz, phi)?;
                        nc
                    }
                };
                self.compress_nc(gpu, nc, b, phi, phic.as_deref_mut(), phic32.as_deref_mut())?;
            }
            Ok(())
        }

        /// Gathers each batch's nf x nf block of the density matrix at `d` (nbf x nbf) over
        /// its kept functions into the chunk's `dv` (FP64) or `dv32` (FP32) at its offset
        /// (`xcb_gather_d`).
        pub(crate) fn gather_d(
            &self,
            gpu: &Gpu,
            c: &Chunk,
            nbf: usize,
            d: DevPtr<f64>,
            dv: Option<&mut DeviceBuf<f64>>,
            dv32: Option<&mut DeviceBuf<f32>>,
        ) -> Result<(), GpuError> {
            let (maxd, _) = self.chunk_max(c);
            let p64 = dv.map_or(DevPtr::null(), |p| p.ptr());
            let p32 = dv32.map_or(DevPtr::null(), |p| p.ptr());
            // SAFETY: matches xcb_gather_d; reads D at the chunk's batches' kept functions
            // (< nbf), writes each batch's nf^2 values at its offset in the chunk's workspace
            // (sized for the largest chunk).
            unsafe {
                gpu.launch(
                    &self.k.gather_d,
                    per_batch(maxd, c.j1 - c.j0, 256),
                    &[
                        Arg::I64(nbf as i64),
                        Arg::buf(&self.bd),
                        Arg::I32(c.j0 as i32),
                        Arg::buf(&self.gf),
                        d.into(),
                        p64.into(),
                        p32.into(),
                    ],
                )
            }
        }

        /// Products of chunk `c`, one per batch, from a shape and address rule.
        pub(crate) fn products(
            &self,
            c: &Chunk,
            f: impl Fn(&HostBatch, [usize; 3]) -> Option<VgemmProduct>,
        ) -> Vec<VgemmProduct> {
            let (mut xo, mut dd) = (0usize, 0usize);
            let mut out = Vec::new();
            for h in &self.host[c.j0..c.j1] {
                let nf = h.global.len();
                if let Some(p) = f(h, [self.ncomp * xo, xo, dd]) {
                    if p.m > 0 && p.n > 0 {
                        out.push(p);
                    }
                }
                xo += nf * h.np;
                dd += nf * nf;
            }
            out
        }
    }

    /// Plans of variable-size batched GEMMs with the default tile.
    pub(crate) fn vgemm(gpu: &Gpu) -> Result<Vgemm, GpuError> {
        Vgemm::new(gpu, VgemmTile::DEFAULT)
    }

    /// A plan of `products` (their addresses must stay valid).
    pub(crate) fn plan(
        gpu: &Gpu,
        vg: &Vgemm,
        products: &[VgemmProduct],
    ) -> Result<VgemmPlan, GpuError> {
        // SAFETY: the callers build products from the engine's own workspaces, which live as
        // long as the plans (both are fields of the engine).
        unsafe { vg.plan(gpu, products) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hb(np: usize, nf: usize) -> HostBatch {
        HostBatch {
            p0: 0,
            np,
            q0: 0,
            local: (0..nf as i32).collect(),
            global: (0..nf as i32).collect(),
        }
    }

    #[test]
    fn chunks_respect_the_budget_and_cover_everything() {
        let batches: Vec<HostBatch> = (0..10).map(|i| hb(100 + i, 10 + i)).collect();
        // five blocks of two batches
        let bb: Vec<(usize, usize)> = (0..5).map(|b| (2 * b, 2 * b + 2)).collect();
        let ch = chunks(&batches, &bb, 4, 4 * 5000);
        assert_eq!(ch.first().unwrap().j0, 0);
        assert_eq!(ch.last().unwrap().j1, 10);
        for w in ch.windows(2) {
            assert_eq!(w[0].b1, w[1].b0);
            assert_eq!(w[0].j1, w[1].j0);
        }
        for c in &ch {
            assert!(c.b1 - c.b0 == 1 || 4 * c.x_len <= 4 * 5000);
        }
        let (bd, lf, gf) = descriptors(&batches, &ch, 4);
        assert_eq!(bd.len(), BD * 10);
        assert_eq!(lf.len(), gf.len());
        // offsets restart per chunk
        for c in &ch {
            assert_eq!(bd[BD * c.j0 + BD_X], 0);
            assert_eq!(bd[BD * c.j0 + BD_D], 0);
        }
    }
}
