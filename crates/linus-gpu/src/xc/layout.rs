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
//! [`XcLayout`] finds the per-batch lists once per geometry, on the device: one screening kernel
//! per (l, pure) class takes the CPU's value test per batch and function
//! (`kernels::colloc::generate_screen`, the kept functions as bits). A call then collocates
//! every batch's kept functions straight into its compressed per-batch array (one launch per
//! class and chunk, `kernels::colloc::generate_batched`, [`XcLayout::prepare`]) and runs the
//! per-batch products of a chunk of batches in one launch of the variable-size batched GEMM
//! ([`crate::kernels::vgemm`]) or cuBLAS's grouped GEMM. The potential of each batch is added to
//! a fixed-point accumulator (`detsum`): exact and independent of the order of the batches.
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

/// `parts` one after another, copied in parallel.
pub(crate) fn par_concat(parts: &[&[i32]]) -> Vec<i32> {
    use rayon::prelude::*;
    let total = parts.iter().map(|p| p.len()).sum();
    let mut out = vec![0i32; total];
    let mut dst: Vec<&mut [i32]> = Vec::with_capacity(parts.len());
    let mut rest = &mut out[..];
    for p in parts {
        let (a, b) = rest.split_at_mut(p.len());
        dst.push(a);
        rest = b;
    }
    dst.into_par_iter()
        .zip(parts.par_iter())
        .for_each(|(d, s)| d.copy_from_slice(s));
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
    let mut fl = 0usize;
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
            d[BD_FL] = fl as i64;
            d[BD_PHI] = (ncomp * xo) as i64;
            d[BD_X] = xo as i64;
            d[BD_D] = dd as i64;
            fl += nf;
            xo += nf * h.np;
            dd += nf * nf;
        }
    }
    // the lists in chunk order: the batches in order when the chunks run over them in order
    let in_order = chunks.first().is_none_or(|c| c.j0 == 0)
        && chunks.windows(2).all(|w| w[0].j1 == w[1].j0)
        && chunks.last().is_none_or(|c| c.j1 == batches.len());
    let order: Vec<usize> = if in_order {
        (0..batches.len()).collect()
    } else {
        chunks.iter().flat_map(|c| c.j0..c.j1).collect()
    };
    let locals: Vec<&[i32]> = order.iter().map(|&j| batches[j].local.as_slice()).collect();
    let globals: Vec<&[i32]> = order
        .iter()
        .map(|&j| batches[j].global.as_slice())
        .collect();
    let (lf, gf) = rayon::join(|| par_concat(&locals), || par_concat(&globals));
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
            "xcb_gather_d",
            "xcb_densities",
            "xcb_weights",
            "xcb_weights_tau",
            "xcb_vscatter",
            "xcb_vfinal",
        ],
    );

    pub(crate) struct Kernels {
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
    /// another order (from [`XcLayout::kernels_of_order`]; the gradient's and the Hessian's
    /// first-order values on their higher-order layouts).
    #[derive(Clone, Copy)]
    pub(crate) enum Collocation<'a> {
        Own,
        With(&'a OrderKernels),
    }

    /// The collocation kernels of one derivative order for the basis's (l, pure) classes, per
    /// batch into the compressed values (`colloc::generate_batched`), and their components.
    pub(crate) struct OrderKernels {
        pub(crate) batched: Vec<Kernel>,
        pub(crate) ncomp: usize,
    }

    /// The direct collocation's work list of a chunking: per entry (a batch and one of its
    /// shells that keeps a function) its batch, global shell and the offset of the shell's kept
    /// positions in the layout's `fmap`; the entries of chunk c and class k at `ranges[c][k]`
    /// (offset, count).
    pub(crate) struct DirectList {
        ent: DeviceBuf<i32>,
        ranges: Vec<Vec<(usize, usize)>>,
    }

    /// Per block and (l, pure) class, the direct collocation's entries of the block's batches
    /// (batch, global shell, offset of the shell's functions' kept positions in `fmap`), in
    /// batch order.
    type BlockLists = Vec<Vec<Vec<i32>>>;

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
        /// Collocation of each batch into its compressed values: kernels per (l, pure) and the
        /// shell tables.
        colloc: OrderKernels,
        /// The direct collocation's work list per chunk (rebuilt with the chunks).
        direct: DirectList,
        /// The kept positions of every block's shells' functions (-1: not kept), which the
        /// entries of `direct` point into (the same for any chunking).
        fmap: DeviceBuf<i32>,
        /// Every block's direct collocation entries per class (their functions' kept positions
        /// in the direct list's `fmap`).
        block_lists: BlockLists,
        /// The basis's (l, pure) classes, in the kernels' order.
        keys: Vec<(usize, bool)>,
        tables: ShellTables,
        pub(crate) max_block_nf: usize,
        /// Largest workspaces over the chunks: sum nf np, sum nf^2.
        pub(crate) max_x: usize,
        pub(crate) max_d: usize,
        /// Sum over batches of points x kept functions (values per pass).
        pub(crate) values: usize,
        /// The per-batch kernels and the variable-size batched GEMM of the engines on it.
        pub(crate) k: Kernels,
        pub(crate) vg: Vgemm,
        /// The setup's parts.
        pub(crate) setup_times: crate::xc::XcLayoutTimes,
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
            let mut max_nf = 0;
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
                lblocks.push(LBlock {
                    p0: b.p0,
                    nf: b.funcs.len(),
                    groups,
                    j0: nbatch,
                    j1: nbatch + sp.len(),
                });
                nbatch += sp.len();
            }
            let tables = ShellTables::new(gpu, basis)?;
            // the value test at the screening order, on the device: one launch per (l, pure)
            // class over the (block, shell) entries, each batch's kept functions as bits (the
            // words of its block's functions)
            let ksrc = colloc::module_source(basis, screen_order);
            let screen_k = keys
                .iter()
                .map(|&(l, p)| {
                    gpu.kernel(
                        "colloc",
                        &ksrc,
                        &colloc::screen_kernel_name(l, p, screen_order),
                    )
                })
                .collect::<Result<Vec<_>, _>>()?;
            let kern = Kernels::new(gpu)?;
            // provisional descriptors: points, block offsets and the batches' first flag words
            let mut bd0 = vec![0i64; BD * nbatch.max(1)];
            let mut words = 0usize;
            let wpb: Vec<usize> = lblocks.iter().map(|lb| lb.nf.div_ceil(32)).collect();
            for (lb, (sp, &w)) in lblocks.iter().zip(split.iter().zip(&wpb)) {
                for (i, &(q0, np)) in sp.iter().enumerate() {
                    let d = &mut bd0[BD * (lb.j0 + i)..BD * (lb.j0 + i + 1)];
                    d[BD_P0] = (lb.p0 + q0) as i64;
                    d[BD_NP] = np as i64;
                    d[BD_Q0] = q0 as i64;
                    d[BD_FL] = words as i64;
                    words += w;
                }
            }
            let bd_d0 = gpu.upload(&bd0)?;
            let mut flags = gpu.alloc_zeros::<u32>(words.max(1))?;
            let x = gpu.upload(&grid.x)?;
            let y = gpu.upload(&grid.y)?;
            let z = gpu.upload(&grid.z)?;
            // per class: (first batch, batches, shell, its first function in the block)
            let mut ents: Vec<Vec<i32>> = vec![Vec::new(); keys.len()];
            let mut max_nb = 0;
            for lb in &lblocks {
                if lb.nf == 0 || lb.j1 == lb.j0 {
                    continue;
                }
                max_nb = max_nb.max(lb.j1 - lb.j0);
                for &(ki, nsh, off) in &lb.groups {
                    for q in off..off + nsh {
                        ents[ki].extend([lb.j0 as i32, (lb.j1 - lb.j0) as i32, lists[q], aos[q]]);
                    }
                }
            }
            for (ki, e) in ents.iter().enumerate() {
                let ne = e.len() / 4;
                if ne == 0 {
                    continue;
                }
                let (l, pure) = keys[ki];
                let nf = if pure {
                    2 * l + 1
                } else {
                    (l + 1) * (l + 2) / 2
                };
                let de = gpu.upload(e)?;
                let cfg = Launch {
                    grid: (ne as u32, 1, 1),
                    block: (128, 1, 1),
                    shared_bytes: (max_nb * nf) as u32,
                };
                // SAFETY: matches the screening kernel: entries of this class (blocks' batches
                // < nbatch, shells of the class, first functions + nf <= the block's
                // functions), the grid's coordinates at the blocks' points; the bits of batch j
                // at its word offset (bd0) within `words`; max_nb nf bytes of shared memory.
                unsafe {
                    gpu.launch(
                        &screen_k[ki],
                        cfg,
                        &[
                            Arg::buf(&x),
                            Arg::buf(&y),
                            Arg::buf(&z),
                            Arg::buf(&bd_d0),
                            Arg::buf(&de),
                            Arg::F64(thresh),
                            Arg::buf(&tables.p0),
                            Arg::buf(&tables.center),
                            Arg::buf(&tables.exp),
                            Arg::buf(&tables.coef),
                            Arg::buf_mut(&mut flags),
                        ],
                    )
                }?;
            }
            let hf = gpu.download(&flags)?;
            let t_screen = t0.elapsed().as_secs_f64();
            drop((flags, bd_d0, x, y, z));
            // the kept functions per batch, the blocks in parallel
            let host: Vec<HostBatch> = {
                use rayon::prelude::*;
                bl.par_iter()
                    .zip(lblocks.par_iter())
                    .zip(split.par_iter())
                    .zip(wpb.par_iter())
                    .flat_map_iter(|(((b, lb), sp), &w)| {
                        let (hf, bd0) = (&hf, &bd0);
                        sp.iter().enumerate().map(move |(i, &(q0, np))| {
                            let wo = bd0[BD * (lb.j0 + i) + BD_FL] as usize;
                            let (mut local, mut global) = (Vec::new(), Vec::new());
                            for (k, &word) in hf[wo..wo + w].iter().enumerate() {
                                let mut m = word;
                                while m != 0 {
                                    let f = 32 * k + m.trailing_zeros() as usize;
                                    m &= m - 1;
                                    local.push(f as i32);
                                    global.push(b.funcs[f] as i32);
                                }
                            }
                            HostBatch {
                                p0: lb.p0 + q0,
                                np,
                                q0,
                                local,
                                global,
                            }
                        })
                    })
                    .collect()
            };
            let ncomp = crate::cpu::ncomp(order);
            let bb: Vec<(usize, usize)> = lblocks.iter().map(|b| (b.j0, b.j1)).collect();
            let ch = chunks(&host, &bb, ncomp, budget);
            let (bd, lf, gf) = descriptors(&host, &ch, ncomp);
            let (values, _) = work(&host);
            let colloc_k = order_kernels(gpu, basis, &keys, order)?;
            // per block and class the kept shells of its batches (in batch order) and their
            // functions' kept positions (fmap), the blocks in parallel
            let class: Vec<usize> = (0..basis.shells.len()).map(key_of).collect();
            let nclass = keys.len();
            let mut parts: Vec<(Vec<Vec<i32>>, Vec<i32>)> = {
                use rayon::prelude::*;
                bl.par_iter()
                    .zip(lblocks.par_iter())
                    .map(|(b, lb)| {
                        let mut lists = vec![Vec::new(); nclass];
                        let mut fm = Vec::new();
                        // the block's function -> its shell (in the block's list)
                        let mut fsh = vec![0u32; b.funcs.len()];
                        for (i, (&sh, &a)) in b.shells.iter().zip(&b.ao_local).enumerate() {
                            fsh[a..a + basis.shells[sh].nfunc()].fill(i as u32);
                        }
                        let mut pos = vec![-1i32; b.funcs.len()];
                        for (j, h) in host.iter().enumerate().take(lb.j1).skip(lb.j0) {
                            for (q, &l) in h.local.iter().enumerate() {
                                pos[l as usize] = q as i32;
                            }
                            let mut last = u32::MAX;
                            for &l in &h.local {
                                let i = fsh[l as usize];
                                if i == last {
                                    continue;
                                }
                                last = i;
                                let (sh, a) = (b.shells[i as usize], b.ao_local[i as usize]);
                                lists[class[sh]].extend([j as i32, sh as i32, fm.len() as i32]);
                                fm.extend_from_slice(&pos[a..a + basis.shells[sh].nfunc()]);
                            }
                            for &l in &h.local {
                                pos[l as usize] = -1;
                            }
                        }
                        (lists, fm)
                    })
                    .collect()
            };
            // the blocks' kept positions one after another: their offsets into the entries
            let fmap = {
                use rayon::prelude::*;
                let bases: Vec<i32> = parts
                    .iter()
                    .scan(0i32, |acc, (_, fm)| {
                        let o = *acc;
                        *acc += fm.len() as i32;
                        Some(o)
                    })
                    .collect();
                parts
                    .par_iter_mut()
                    .zip(bases.par_iter())
                    .for_each(|((lists, _), &o)| {
                        for list in lists.iter_mut() {
                            for e in list.as_chunks_mut::<3>().0 {
                                e[2] += o;
                            }
                        }
                    });
                let fms: Vec<&[i32]> = parts.iter().map(|(_, fm)| fm.as_slice()).collect();
                par_concat(&fms)
            };
            let block_lists: BlockLists = parts.into_iter().map(|(lists, _)| lists).collect();
            let direct = direct_list(gpu, &block_lists, &ch, nclass)?;
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
                direct,
                fmap: gpu.upload(if fmap.is_empty() { &[0] } else { &fmap })?,
                block_lists,
                keys,
                tables,
                max_block_nf: max_nf,
                values,
                k: kern,
                vg: vgemm(gpu)?,
                setup_times: crate::xc::XcLayoutTimes {
                    blocks: t_blocks,
                    screen: t_screen - t_blocks,
                    lists: t0.elapsed().as_secs_f64() - t_screen,
                },
            })
        }

        /// Takes `ch` (consecutive runs of whole blocks covering every block, an engine's own
        /// chunking) as the layout's chunks: the batch descriptors, function lists, workspace
        /// sizes and the direct collocation's work list follow them. The batch descriptors (BD
        /// per batch, as uploaded), for the engine's own tables.
        pub(crate) fn set_chunks(
            &mut self,
            gpu: &Gpu,
            ch: Vec<Chunk>,
        ) -> Result<Vec<i64>, GpuError> {
            let (bd, lf, gf) = descriptors(&self.host, &ch, self.ncomp);
            self.max_x = ch.iter().map(|c| c.x_len).max().unwrap_or(0);
            self.max_d = ch.iter().map(|c| c.d_len).max().unwrap_or(0);
            self.bd = gpu.upload(if bd.is_empty() { &[0] } else { &bd })?;
            self.lf = gpu.upload(if lf.is_empty() { &[0] } else { &lf })?;
            self.gf = gpu.upload(if gf.is_empty() { &[0] } else { &gf })?;
            self.direct = direct_list(gpu, &self.block_lists, &ch, self.keys.len())?;
            self.chunks = ch;
            Ok(bd)
        }

        /// Collocation kernels of the basis's (l, pure) classes at derivative `order` (for
        /// [`XcLayout::prepare`] with [`Collocation::With`]).
        pub(crate) fn kernels_of_order(
            &self,
            gpu: &Gpu,
            basis: &BasisSet,
            order: usize,
        ) -> Result<OrderKernels, GpuError> {
            order_kernels(gpu, basis, &self.keys, order)
        }

        /// The largest nf^2 and nf np over the batches of chunk `c`.
        pub(crate) fn chunk_max(&self, c: &Chunk) -> (usize, usize) {
            self.host[c.j0..c.j1].iter().fold((0, 0), |a, h| {
                let nf = h.global.len();
                (a.0.max(nf * nf), a.1.max(nf * h.np))
            })
        }

        /// Collocates every batch of chunk `c` (with the kernels `colloc`) at the grid coordinates
        /// `xyz` straight into the chunk's compressed values `phic` and/or `phic32`: one launch
        /// per (l, pure) class over the chunk's (batch, kept shell) entries, each of the shell's
        /// kept functions written to its place.
        pub(crate) fn prepare(
            &self,
            gpu: &Gpu,
            c: &Chunk,
            colloc: Collocation<'_>,
            xyz: [&DeviceBuf<f64>; 3],
            phic: Option<&mut DeviceBuf<f64>>,
            phic32: Option<&mut DeviceBuf<f32>>,
        ) -> Result<(), GpuError> {
            let ci = self
                .chunks
                .iter()
                .position(|x| x.j0 == c.j0 && x.j1 == c.j1)
                .ok_or_else(|| GpuError::Contract("XC: a chunk of another layout".into()))?;
            let kern = match colloc {
                Collocation::Own => &self.colloc,
                Collocation::With(k) => k,
            };
            if kern.batched.len() != self.keys.len() {
                return Err(GpuError::Contract(
                    "XC: collocation kernels of another basis".into(),
                ));
            }
            let p64 = phic.map_or(DevPtr::null(), |p| p.ptr());
            let p32 = phic32.map_or(DevPtr::null(), |p| p.ptr());
            for (ki, &(off, n)) in self.direct.ranges[ci].iter().enumerate() {
                if n == 0 {
                    continue;
                }
                let cfg = Launch {
                    grid: (n as u32, 1, 1),
                    block: (128, 1, 1),
                    shared_bytes: 0,
                };
                // SAFETY: matches the batched collocation kernel: the entries off.. off + n of
                // this chunk and class (batches of the chunk, shells of the class, kept
                // positions < the batch's functions), the grid's coordinates at the batches'
                // points; each batch's values at its offset within the chunk's workspaces
                // (sized for the largest chunk).
                unsafe {
                    gpu.launch(
                        &kern.batched[ki],
                        cfg,
                        &[
                            Arg::buf(xyz[0]),
                            Arg::buf(xyz[1]),
                            Arg::buf(xyz[2]),
                            Arg::buf(&self.bd),
                            Arg::from(self.direct.ent.at(3 * off)),
                            Arg::buf(&self.fmap),
                            Arg::buf(&self.tables.p0),
                            Arg::buf(&self.tables.center),
                            Arg::buf(&self.tables.exp),
                            Arg::buf(&self.tables.coef),
                            p64.into(),
                            p32.into(),
                        ],
                    )
                }?;
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

    /// The batched collocation kernels of the (l, pure) classes `keys` at `order`.
    fn order_kernels(
        gpu: &Gpu,
        basis: &BasisSet,
        keys: &[(usize, bool)],
        order: usize,
    ) -> Result<OrderKernels, GpuError> {
        let src = colloc::module_source(basis, order);
        let k = |name: String| gpu.kernel("colloc", &src, &name);
        Ok(OrderKernels {
            batched: keys
                .iter()
                .map(|&(l, p)| k(colloc::batched_kernel_name(l, p, order)))
                .collect::<Result<Vec<_>, _>>()?,
            ncomp: crate::cpu::ncomp(order),
        })
    }

    /// The direct collocation's work list of `chunks` over `nclass` (l, pure) classes: each
    /// chunk's entries grouped by class (its blocks' lists in block order).
    fn direct_list(
        gpu: &Gpu,
        lists: &BlockLists,
        chunks: &[Chunk],
        nclass: usize,
    ) -> Result<DirectList, GpuError> {
        let mut parts: Vec<&[i32]> = Vec::new();
        let mut ranges = Vec::with_capacity(chunks.len());
        let mut at = 0;
        for c in chunks {
            let mut per = Vec::with_capacity(nclass);
            for k in 0..nclass {
                let n0 = at;
                for bl in &lists[c.b0..c.b1] {
                    parts.push(&bl[k]);
                    at += bl[k].len() / 3;
                }
                per.push((n0, at - n0));
            }
            ranges.push(per);
        }
        let ent = par_concat(&parts);
        Ok(DirectList {
            ent: gpu.upload(if ent.is_empty() { &[0, 0, 0] } else { &ent })?,
            ranges,
        })
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
