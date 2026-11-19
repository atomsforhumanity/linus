//! The precision studies' block path of the XC gradient (feature `study`): per block of
//! batches, the functions kept by the rigorous shell bound and one product per block. It was
//! the first device implementation; production screens per batch
//! (`GpuXcGrad::compute_batched`). Kept for comparisons (decision K-b).

use super::*;
use crate::blas::{GemmDims, MatMut, MatRef, Op};
use crate::kernels::colloc::{self, ShellTables};
use crate::linalg::Arith;
use crate::xc::blocks;
use crate::Mark;

/// A block on the device: its points, kept functions and the collocation work per kernel.
struct DevBlock {
    p0: usize,
    np: usize,
    nf: usize,
    funcs: DeviceBuf<i32>,
    groups: Vec<(usize, usize, usize)>,
}

/// The block path's screening (per block by the rigorous shell bound at second order),
/// collocation, kernels and GEMMs.
pub(super) struct GradBlocks {
    ncomp: usize,
    tables: ShellTables,
    colloc: Vec<Kernel>,
    lists: DeviceBuf<i32>,
    aos: DeviceBuf<i32>,
    blocks: Vec<DevBlock>,
    max_nf: usize,
    max_np: usize,
    values: usize,
    /// The GEMMs, in `gemm_arith`.
    gemm: Gemm,
    gemm_arith: Arith,
    gather_sub: Kernel,
    densities_dm: Kernel,
    weights: Kernel,
    contract_funcs: Kernel,
    contract_points: Kernel,
}

impl GradBlocks {
    /// The block path of `basis` on `grid`, collocated to `order` (`opts.math` FP64 or emulated).
    pub(super) fn new(
        gpu: &Gpu,
        basis: &BasisSet,
        grid: &MolecularGrid,
        order: usize,
        opts: &GpuXcGradOptions,
    ) -> Result<GradBlocks, GpuError> {
        let ncomp = crate::cpu::ncomp(order);
        // the screening (the CPU gradient's, second order), shell lists and collocation
        let bl = blocks(basis, grid, 2, opts.func_thresh, opts.block_points.max(1))?;
        let mut keys: Vec<(usize, bool)> = basis.shells.iter().map(|s| (s.l, s.pure)).collect();
        keys.sort();
        keys.dedup();
        let src = colloc::module_source(basis, order);
        let colloc_k = keys
            .iter()
            .map(|&(l, p)| gpu.kernel("colloc", &src, &colloc::kernel_name(l, p, order)))
            .collect::<Result<Vec<_>, _>>()?;
        let key_of = |s: usize| {
            let sh = &basis.shells[s];
            keys.iter().position(|k| *k == (sh.l, sh.pure)).unwrap()
        };
        let (mut lists, mut aos) = (Vec::<i32>::new(), Vec::<i32>::new());
        let mut dev = Vec::with_capacity(bl.len());
        let (mut max_nf, mut max_np, mut values) = (0, 0, 0usize);
        for b in &bl {
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
            let f: Vec<i32> = b.funcs.iter().map(|&f| f as i32).collect();
            max_nf = max_nf.max(b.funcs.len());
            max_np = max_np.max(b.np);
            values += b.funcs.len() * b.np;
            dev.push(DevBlock {
                p0: b.p0,
                np: b.np,
                nf: b.funcs.len(),
                funcs: gpu.upload(if f.is_empty() { &[0] } else { &f })?,
                groups,
            });
        }
        let tables = ShellTables::new(gpu, basis)?;
        let xc = &crate::xc::MODULE;
        let xsrc = xc.source();
        let gsrc = MODULE.source();
        let k = |e: &str| gpu.kernel(MODULE.name, &gsrc, e);
        let mut gemm = Gemm::new(gpu)?;
        let gemm_arith = opts.math.cublas();
        match opts.math {
            m if m.emulated() => gemm.prepare(gpu, gemm_arith)?,
            XcMath::Fp64 => {}
            m => {
                return Err(GpuError::NotCovered(format!(
                    "XC gradient: {m:?} needs per-batch screening"
                )))
            }
        }
        Ok(GradBlocks {
            ncomp,
            tables,
            colloc: colloc_k,
            lists: gpu.upload(if lists.is_empty() { &[0] } else { &lists })?,
            aos: gpu.upload(if aos.is_empty() { &[0] } else { &aos })?,
            blocks: dev,
            max_nf,
            max_np,
            values,
            gemm,
            gemm_arith,
            gather_sub: gpu.kernel(xc.name, &xsrc, "xc_gather_sub")?,
            densities_dm: gpu.kernel(xc.name, &xsrc, "xc_densities_dm")?,
            weights: k("xg_weights")?,
            contract_funcs: k("xg_contract_funcs")?,
            contract_points: k("xg_contract_points")?,
        })
    }

    /// (blocks, the largest block's functions, values per pass).
    pub(super) fn sizes(&self) -> (usize, usize, usize) {
        (self.blocks.len(), self.max_nf, self.values)
    }

    /// Collocates block `b` into `phi` at the grid coordinates `xyz`.
    fn collocate(
        &self,
        gpu: &Gpu,
        b: &DevBlock,
        xyz: [&DeviceBuf<f64>; 3],
        phi: &mut DeviceBuf<f64>,
    ) -> Result<(), GpuError> {
        let at = |buf: &DeviceBuf<f64>| buf.at(b.p0);
        for &(ki, nsh, off) in &b.groups {
            // SAFETY: the block's shells of this kernel's (l, pure) with their local first
            // functions (< nf); the output holds nf ncomp np values (<= its size).
            unsafe {
                self.tables.launch(
                    gpu,
                    &self.colloc[ki],
                    nsh,
                    self.lists.at(off),
                    self.aos.at(off),
                    [at(xyz[0]), at(xyz[1]), at(xyz[2])],
                    b.np,
                    self.ncomp * b.np,
                    b.np,
                    phi.ptr(),
                )
            }?;
        }
        Ok(())
    }
}

impl GpuXcGrad {
    /// The block path of [`GpuXcGrad::compute_device`] (arguments checked there).
    pub(super) fn compute_blocks(
        &mut self,
        gpu: &Gpu,
        dens: &DeviceBuf<f64>,
        restricted: bool,
        grid_response: bool,
        func: Option<&mut PointFn<'_>>,
    ) -> Result<Vec<[f64; 3]>, GpuError> {
        let blk = self.blk.as_ref().expect("the block path's setup");
        let (nbf, npts) = (self.nbf, self.npts);
        let n2 = nbf * nbf;
        let nsp = if restricted { 1 } else { 2 };
        let t_all = Instant::now();
        let mut times = GpuXcGradTimes::default();
        let (gga, mgga) = (self.kind().gga, self.kind().mgga);
        let nc = blk.ncomp;
        let nsig = if !gga {
            0
        } else if nsp == 1 {
            1
        } else {
            3
        };
        let mut rho = gpu.alloc_zeros::<f64>(npts * nsp)?;
        let mut grad = gpu.alloc_zeros::<f64>(if gga { npts * nsp * 3 } else { 1 })?;
        let mut sigma = gpu.alloc_zeros::<f64>((npts * nsig).max(1))?;
        let mut tau = gpu.alloc_zeros::<f64>(if mgga { npts * nsp } else { 1 })?;
        let (mnf, mnp) = (blk.max_nf.max(1), blk.max_np.max(1));
        let mut phi = gpu.alloc_zeros::<f64>(mnf * nc * mnp)?;
        let mut dsub = gpu.alloc_zeros::<f64>(mnf * mnf)?;
        let mut xm = gpu.alloc_zeros::<f64>(mnp * mnf)?;
        let mut ym = gpu.alloc_zeros::<f64>(if mgga { 3 * mnp * mnf } else { 1 })?;
        let mut marks: Vec<Mark> = Vec::new();
        let mut spans: Vec<(usize, usize, usize)> = Vec::new();
        let mark = |marks: &mut Vec<Mark>| -> Result<usize, GpuError> {
            marks.push(gpu.mark()?);
            Ok(marks.len() - 1)
        };
        // D_sub of spin sp, X = phi_0 D_sub, and (meta-GGA) Y = grad phi D_sub
        let dm_products = |b: &DevBlock,
                           sp: usize,
                           phi: &DeviceBuf<f64>,
                           dsub: &mut DeviceBuf<f64>,
                           xm: &mut DeviceBuf<f64>,
                           ym: &mut DeviceBuf<f64>|
         -> Result<(), GpuError> {
            // SAFETY: matches xc_gather_sub; reads D_sp at the block's functions, writes
            // nf^2 values.
            unsafe {
                gpu.launch(
                    &blk.gather_sub,
                    Launch::linear(b.nf * b.nf, 256),
                    &[
                        Arg::I32(b.nf as i32),
                        Arg::I64(nbf as i64),
                        Arg::buf(&b.funcs),
                        Arg::from(dens.at(sp * n2)),
                        Arg::buf_mut(dsub),
                    ],
                )
            }?;
            blk.gemm.gemm(
                blk.gemm_arith,
                GemmDims::new(Op::N, Op::N, b.np, b.nf, b.nf),
                (1.0, 0.0),
                MatRef::new(phi, 0, nc * b.np),
                MatRef::new(dsub, 0, b.nf),
                MatMut::new(xm, 0, b.np),
            )?;
            if mgga {
                blk.gemm.gemm(
                    blk.gemm_arith,
                    GemmDims::new(Op::N, Op::N, 3 * b.np, b.nf, b.nf),
                    (1.0, 0.0),
                    MatRef::new(phi, b.np, nc * b.np),
                    MatRef::new(dsub, 0, b.nf),
                    MatMut::new(ym, 0, 3 * b.np),
                )?;
            }
            Ok(())
        };

        // phase 1: densities
        for b in &blk.blocks {
            if b.nf == 0 {
                continue;
            }
            let s = mark(&mut marks)?;
            blk.collocate(gpu, b, [&self.x, &self.y, &self.z], &mut phi)?;
            let s1 = mark(&mut marks)?;
            for sp in 0..nsp {
                dm_products(b, sp, &phi, &mut dsub, &mut xm, &mut ym)?;
                // SAFETY: matches xc_densities_dm; reads the block's phi, X and Y, writes
                // spin sp at the block's points of the grid arrays.
                unsafe {
                    gpu.launch(
                        &blk.densities_dm,
                        Launch::linear(b.np, 128),
                        &[
                            Arg::I32(b.np as i32),
                            Arg::I64(b.p0 as i64),
                            Arg::I32(nc as i32),
                            Arg::I64(b.np as i64),
                            Arg::I32(b.nf as i32),
                            Arg::I32(nsp as i32),
                            Arg::I32(sp as i32),
                            Arg::F64(1.0),
                            Arg::I32(gga as i32),
                            Arg::I32(mgga as i32),
                            Arg::buf(&phi),
                            Arg::buf(&xm),
                            Arg::buf(&ym),
                            Arg::buf_mut(&mut rho),
                            Arg::buf_mut(&mut grad),
                            Arg::buf_mut(&mut tau),
                        ],
                    )
                }?;
            }
            spans.extend([(0, s, s1), (1, s1, mark(&mut marks)?)]);
        }
        if gga {
            // SAFETY: matches xc_sigma; npts points of grad in, of sigma out.
            unsafe {
                gpu.launch(
                    &self.kernels.sigma,
                    Launch::linear(npts, 256),
                    &[
                        Arg::I64(npts as i64),
                        Arg::I32(nsp as i32),
                        Arg::buf(&grad),
                        Arg::buf_mut(&mut sigma),
                    ],
                )
            }?;
        }

        // the functional on the host
        gpu.synchronize()?;
        let tf = Instant::now();
        let inp = self.point_inputs(restricted, &rho, &grad);
        let (pt, _) = self.functional.pointwise(gpu, &inp, &sigma, &tau, func)?;
        times.functional = tf.elapsed().as_secs_f64();
        let (dzk, dvrho, dvsig, dvtau, force) =
            (pt.zk, pt.vrho, pt.vsigma, pt.vtau, pt.point_force);

        // phase 2: the contraction
        let mut facc = DetAcc::new(gpu, nbf * 3)?;
        // the point-motion term, starting from the functional's explicit point derivatives
        let mut tp = match force {
            Some(f) if grid_response => f,
            _ => gpu.alloc_zeros::<f64>(if grid_response { 3 * npts } else { 1 })?,
        };
        let mut wm = gpu.alloc_zeros::<f64>(mnp * mnf)?;
        let mut bm = gpu.alloc_zeros::<f64>(mnp * mnf)?;
        let mut vg = gpu.alloc_zeros::<f64>(3 * mnp)?;
        let mut vt = gpu.alloc_zeros::<f64>(mnp)?;
        for b in &blk.blocks {
            if b.nf == 0 {
                continue;
            }
            let s = mark(&mut marks)?;
            blk.collocate(gpu, b, [&self.x, &self.y, &self.z], &mut phi)?;
            spans.push((0, s, mark(&mut marks)?));
            for sp in 0..nsp {
                let s = mark(&mut marks)?;
                dm_products(b, sp, &phi, &mut dsub, &mut xm, &mut ym)?;
                // SAFETY: matches xg_weights; reads the block's points of the grid arrays
                // and phi, writes nf np weights and np (3 np) per-point values.
                unsafe {
                    gpu.launch(
                        &blk.weights,
                        Launch::linear(b.np * b.nf, 256),
                        &[
                            Arg::I32(b.np as i32),
                            Arg::I64(b.p0 as i64),
                            Arg::I32(b.nf as i32),
                            Arg::I32(nc as i32),
                            Arg::I32(nsp as i32),
                            Arg::I32(sp as i32),
                            Arg::I32(gga as i32),
                            Arg::I32(mgga as i32),
                            Arg::buf(&self.w),
                            Arg::buf(&dvrho),
                            Arg::buf(&dvsig),
                            Arg::buf(&dvtau),
                            Arg::buf(&grad),
                            Arg::buf(&phi),
                            Arg::buf_mut(&mut wm),
                            Arg::buf_mut(&mut vg),
                            Arg::buf_mut(&mut vt),
                        ],
                    )
                }?;
                // B = W D_sub
                blk.gemm.gemm(
                    blk.gemm_arith,
                    GemmDims::new(Op::N, Op::N, b.np, b.nf, b.nf),
                    (1.0, 0.0),
                    MatRef::new(&wm, 0, b.np),
                    MatRef::new(&dsub, 0, b.nf),
                    MatMut::new(&mut bm, 0, b.np),
                )?;
                let s1 = mark(&mut marks)?;
                let wpb = 256 / gpu.warp_size() as usize;
                let common = [
                    Arg::I32(b.np as i32),
                    Arg::I32(b.nf as i32),
                    Arg::I32(nc as i32),
                    Arg::I32(gga as i32),
                    Arg::I32(mgga as i32),
                ];
                let mut args = common.to_vec();
                args.extend([
                    Arg::buf(&b.funcs),
                    Arg::buf(&phi),
                    Arg::buf(&xm),
                    Arg::buf(&bm),
                    Arg::buf(&ym),
                    Arg::buf(&vg),
                    Arg::buf(&vt),
                    facc.arg(),
                    Arg::I64((nbf * 3) as i64),
                ]);
                let cfg = Launch {
                    grid: (b.nf.div_ceil(wpb) as u32, 1, 1),
                    block: (256, 1, 1),
                    shared_bytes: 0,
                };
                // SAFETY: matches xg_contract_funcs; one warp per function f < nf reads the
                // block's arrays (np points, nf functions) and adds to function funcs[f] <
                // nbf of the accumulator.
                unsafe { gpu.launch(&blk.contract_funcs, cfg, &args) }?;
                if grid_response {
                    // SAFETY: matches xg_contract_points; one thread per point g < np
                    // reads the block's arrays and adds to tp at the block's points.
                    unsafe {
                        gpu.launch(
                            &blk.contract_points,
                            Launch::linear(b.np, 128),
                            &[
                                Arg::I32(b.np as i32),
                                Arg::I64(b.p0 as i64),
                                Arg::I32(b.nf as i32),
                                Arg::I32(nc as i32),
                                Arg::I32(gga as i32),
                                Arg::I32(mgga as i32),
                                Arg::buf(&phi),
                                Arg::buf(&xm),
                                Arg::buf(&bm),
                                Arg::buf(&ym),
                                Arg::buf(&vg),
                                Arg::buf(&vt),
                                Arg::buf_mut(&mut tp),
                            ],
                        )
                    }?;
                }
                spans.extend([(2, s, s1), (3, s1, mark(&mut marks)?)]);
            }
        }

        for &(what, s, e) in &spans {
            let dt = marks[s].seconds_until(&marks[e])?;
            match what {
                0 => times.colloc += dt,
                1 => times.density += dt,
                2 => times.gemm += dt,
                3 => times.contract += dt,
                _ => {}
            }
        }
        let g = self.tail(gpu, nsp, grid_response, &rho, &dzk, &tp, &facc, &mut times)?;
        times.total = t_all.elapsed().as_secs_f64();
        self.last = times;
        Ok(g)
    }
}
