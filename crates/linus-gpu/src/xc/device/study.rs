//! The precision studies' block path of the XC energy and potential (feature `study`): per
//! block of batches, the functions kept by the rigorous shell bound, one product per block,
//! and the density route per block ([`super::DensityRoute`]). It was the first device
//! implementation; production screens per batch (`GpuXc::compute_batched`) and keeps block
//! screening for the CPKS kernel only. Kept for comparisons and the precision study's
//! emulated GEMMs (decision K-b).

use super::*;
use crate::Mark;

impl GpuXc {
    /// The block path of [`GpuXc::compute_device`] (arguments checked there).
    pub(super) fn compute_blocks(
        &mut self,
        gpu: &Gpu,
        call: VxcCall<'_, '_>,
        v: &mut DeviceBuf<f64>,
    ) -> Result<f64, GpuError> {
        let VxcCall {
            src,
            restricted,
            func,
        } = call;
        // densities without orbitals go to the per-batch path only (`compute_densities`)
        let VxcSource::Orbitals {
            cmo,
            ncols,
            occ: occd,
        } = src
        else {
            return Err(GpuError::Contract(
                "XC: the block path takes orbitals".into(),
            ));
        };
        let (nbf, npts) = (self.nbf, self.npts);
        let n2 = nbf * nbf;
        let nsp = if restricted { 1 } else { 2 };
        let ntot: usize = ncols.iter().sum();
        let mut times = GpuXcCallTimes::default();
        let m0 = gpu.mark()?;
        let (gga, mgga) = (self.kind().gga, self.kind().mgga);
        let nc = self.ncomp;
        let n0 = ncols[0];
        // grid arrays
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
        // block workspaces
        let mut phi = gpu.alloc_zeros::<f64>((self.max_nf * nc * self.max_np).max(1))?;
        let mut csub = gpu.alloc_zeros::<f64>((self.max_nf * ntot).max(1))?;
        // per block, the cheaper density route (multiply-adds per function-point value):
        // orbitals, psi = phi C_sub (ncomp ntot), or the density matrix, X = phi_0 D_sub and
        // for tau Y = (grad phi) D_sub per spin (nsp nf (1 + 3 mgga))
        let route = self.route;
        let dm_route = |nf: usize| {
            ntot > 0
                && match route {
                    DensityRoute::Auto => nf * nsp * (1 + 3 * usize::from(mgga)) < nc * ntot,
                    DensityRoute::Orbitals => false,
                    DensityRoute::DensityMatrix => true,
                }
        };
        let any_dm = self.blocks.iter().any(|b| b.nf > 0 && dm_route(b.nf));
        let any_orb = self.blocks.iter().any(|b| b.nf > 0 && !dm_route(b.nf));
        let mut psi =
            gpu.alloc_zeros::<f64>(if any_orb { nc * self.max_np * ntot } else { 1 }.max(1))?;
        let (dmat, mut dsub, mut xm, mut ym) = if any_dm {
            // D_s = C_s diag(occ_s) C_s^T
            let mut cs = gpu.alloc_zeros::<f64>(nbf * ntot)?;
            // SAFETY: matches xc_scale_cols; nbf ntot values in and out.
            unsafe {
                gpu.launch(
                    &self.kernels.scale_cols,
                    Launch::linear(nbf * ntot, 256),
                    &[
                        Arg::I64(nbf as i64),
                        Arg::I32(ntot as i32),
                        Arg::buf(cmo),
                        Arg::buf(occd),
                        Arg::buf_mut(&mut cs),
                    ],
                )
            }?;
            let mut d = gpu.alloc_zeros::<f64>(nsp * n2)?;
            let mut c0 = 0;
            for (sp, &ncol) in ncols.iter().enumerate() {
                if ncol > 0 {
                    self.gemm.gemm(
                        self.gemm_arith,
                        GemmDims::new(Op::N, Op::T, nbf, nbf, ncol),
                        (1.0, 0.0),
                        MatRef::new(&cs, c0 * nbf, nbf),
                        MatRef::new(cmo, c0 * nbf, nbf),
                        MatMut::new(&mut d, sp * n2, nbf),
                    )?;
                }
                c0 += ncol;
            }
            (
                Some(d),
                gpu.alloc_zeros::<f64>(self.max_nf * self.max_nf)?,
                gpu.alloc_zeros::<f64>(self.max_np * self.max_nf)?,
                gpu.alloc_zeros::<f64>(if mgga {
                    3 * self.max_np * self.max_nf
                } else {
                    1
                })?,
            )
        } else {
            (
                None,
                gpu.alloc_zeros::<f64>(1)?,
                gpu.alloc_zeros::<f64>(1)?,
                gpu.alloc_zeros::<f64>(1)?,
            )
        };

        // phase 1: densities on the grid; timing marks and (kind, start, end) spans
        let mut marks: Vec<Mark> = Vec::new();
        let mut spans: Vec<(usize, usize, usize)> = Vec::new();
        let mark = |marks: &mut Vec<Mark>| -> Result<usize, GpuError> {
            marks.push(gpu.mark()?);
            Ok(marks.len() - 1)
        };
        for b in &self.blocks {
            if b.nf == 0 || ntot == 0 {
                continue;
            }
            let s = mark(&mut marks)?;
            self.collocate(gpu, b, &mut phi)?;
            let s1 = mark(&mut marks)?;
            if let (true, Some(d)) = (dm_route(b.nf), dmat.as_ref()) {
                for sp in 0..nsp {
                    // SAFETY: matches xc_gather_sub; reads D_sp at the block's functions,
                    // writes nf^2 values.
                    unsafe {
                        gpu.launch(
                            &self.kernels.gather_sub,
                            Launch::linear(b.nf * b.nf, 256),
                            &[
                                Arg::I32(b.nf as i32),
                                Arg::I64(nbf as i64),
                                Arg::buf(&b.funcs),
                                Arg::from(d.at(sp * n2)),
                                Arg::buf_mut(&mut dsub),
                            ],
                        )
                    }?;
                    self.gemm.gemm(
                        self.gemm_arith,
                        GemmDims::new(Op::N, Op::N, b.np, b.nf, b.nf),
                        (1.0, 0.0),
                        MatRef::new(&phi, 0, nc * b.np),
                        MatRef::new(&dsub, 0, b.nf),
                        MatMut::new(&mut xm, 0, b.np),
                    )?;
                    if mgga {
                        self.gemm.gemm(
                            self.gemm_arith,
                            GemmDims::new(Op::N, Op::N, 3 * b.np, b.nf, b.nf),
                            (1.0, 0.0),
                            MatRef::new(&phi, b.np, nc * b.np),
                            MatRef::new(&dsub, 0, b.nf),
                            MatMut::new(&mut ym, 0, 3 * b.np),
                        )?;
                    }
                    // SAFETY: matches xc_densities_dm; reads the block's phi, X and Y,
                    // writes spin sp at the block's points of the grid arrays.
                    unsafe {
                        gpu.launch(
                            &self.kernels.densities_dm,
                            Launch::linear(b.np, 128),
                            &[
                                Arg::I32(b.np as i32),
                                Arg::I64(b.p0 as i64),
                                Arg::I32(nc as i32),
                                Arg::I64(b.np as i64),
                                Arg::I32(b.nf as i32),
                                Arg::I32(nsp as i32),
                                Arg::I32(sp as i32),
                                Arg::F64(if restricted { 2.0 } else { 1.0 }),
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
                let s2 = mark(&mut marks)?;
                spans.extend([(0, s, s1), (1, s1, s2)]);
                continue;
            }
            // SAFETY: matches xc_gather_rows; reads ntot columns of C at the block's
            // functions (< nbf), writes nf ntot values.
            unsafe {
                gpu.launch(
                    &self.kernels.gather,
                    Launch::linear(b.nf * ntot, 256),
                    &[
                        Arg::I32(b.nf as i32),
                        Arg::I32(ntot as i32),
                        Arg::I64(nbf as i64),
                        Arg::buf(&b.funcs),
                        Arg::buf(cmo),
                        Arg::buf_mut(&mut csub),
                    ],
                )
            }?;
            self.gemm.gemm(
                self.gemm_arith,
                GemmDims::new(Op::N, Op::N, nc * b.np, ntot, b.nf),
                (1.0, 0.0),
                MatRef::new(&phi, 0, nc * b.np),
                MatRef::new(&csub, 0, b.nf),
                MatMut::new(&mut psi, 0, nc * b.np),
            )?;
            // SAFETY: matches xc_densities; reads psi of the block, writes the block's
            // points of the grid arrays.
            unsafe {
                gpu.launch(
                    &self.kernels.densities,
                    Launch::linear(b.np, 128),
                    &[
                        Arg::I32(b.np as i32),
                        Arg::I64(b.p0 as i64),
                        Arg::I32(nc as i32),
                        Arg::I64(b.np as i64),
                        Arg::I32(nsp as i32),
                        Arg::I32(if nsp == 1 { ntot } else { n0 } as i32),
                        Arg::I32(ntot as i32),
                        Arg::F64(if restricted { 2.0 } else { 1.0 }),
                        Arg::I32(gga as i32),
                        Arg::I32(mgga as i32),
                        Arg::buf(&psi),
                        Arg::buf(occd),
                        Arg::buf_mut(&mut rho),
                        Arg::buf_mut(&mut grad),
                        Arg::buf_mut(&mut tau),
                    ],
                )
            }?;
            let s2 = mark(&mut marks)?;
            spans.extend([(0, s, s1), (1, s1, s2)]);
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

        // the functional on the host (after the device has finished the densities), or the
        // caller's
        let inp = self.point_inputs(restricted, &rho, &grad);
        let (pt, ptimes) = self.functional.pointwise(gpu, &inp, &sigma, &tau, func)?;
        (times.functional, times.libxc) = (ptimes.functional, ptimes.libxc);
        let (dzk, dvrho, dvsig, dvtau) = (pt.zk, pt.vrho, pt.vsigma, pt.vtau);

        // energy (fixed point)
        let me0 = gpu.mark()?;
        let mut acc = DetAcc::new(gpu, 1)?;
        // SAFETY: matches xc_energy; reads npts points of w, rho, zk; adds to acc[0].
        unsafe {
            gpu.launch(
                &self.kernels.energy,
                Launch::linear(npts, 256),
                &[
                    Arg::I64(npts as i64),
                    Arg::I32(nsp as i32),
                    Arg::buf(&self.w),
                    Arg::buf(&rho),
                    Arg::buf(&dzk),
                    acc.arg(),
                ],
            )
        }?;
        let exc = acc.to_f64(gpu)?[0];
        times.energy = me0.seconds_until(&gpu.mark()?)?;

        // phase 2: the potential
        gpu.zero_range(v, 0, nsp * n2)?;
        let mut wm = gpu.alloc_zeros::<f64>((self.max_nf * self.max_np).max(1))?;
        let mut wt = gpu.alloc_zeros::<f64>(if mgga {
            3 * self.max_nf * self.max_np
        } else {
            1
        })?;
        let mut vsub = gpu.alloc_zeros::<f64>((self.max_nf * self.max_nf).max(1))?;
        for b in &self.blocks {
            if b.nf == 0 {
                continue;
            }
            let s = mark(&mut marks)?;
            self.collocate(gpu, b, &mut phi)?;
            let s1 = mark(&mut marks)?;
            spans.push((0, s, s1));
            for sp in 0..nsp {
                // SAFETY: matches xc_weights; reads the block's points and phi, writes
                // nf np weights.
                unsafe {
                    gpu.launch(
                        &self.kernels.weights,
                        Launch::linear(b.np * b.nf, 256),
                        &[
                            Arg::I32(b.np as i32),
                            Arg::I64(b.p0 as i64),
                            Arg::I32(b.nf as i32),
                            Arg::I32(nc as i32),
                            Arg::I64(b.np as i64),
                            Arg::I32(nsp as i32),
                            Arg::I32(sp as i32),
                            Arg::I32(gga as i32),
                            Arg::buf(&self.w),
                            Arg::buf(&dvrho),
                            Arg::buf(&dvsig),
                            Arg::buf(&grad),
                            Arg::buf(&phi),
                            Arg::buf_mut(&mut wm),
                        ],
                    )
                }?;
                self.gemm.gemm(
                    self.gemm_arith,
                    GemmDims::new(Op::T, Op::N, b.nf, b.nf, b.np),
                    (1.0, 0.0),
                    MatRef::new(&phi, 0, nc * b.np),
                    MatRef::new(&wm, 0, b.np),
                    MatMut::new(&mut vsub, 0, b.nf),
                )?;
                if mgga {
                    // SAFETY: matches xc_weights_tau; writes 3 nf np values.
                    unsafe {
                        gpu.launch(
                            &self.kernels.weights_tau,
                            Launch::linear(b.np * b.nf, 256),
                            &[
                                Arg::I32(b.np as i32),
                                Arg::I64(b.p0 as i64),
                                Arg::I32(b.nf as i32),
                                Arg::I32(nc as i32),
                                Arg::I64(b.np as i64),
                                Arg::I32(nsp as i32),
                                Arg::I32(sp as i32),
                                Arg::buf(&self.w),
                                Arg::buf(&dvtau),
                                Arg::buf(&phi),
                                Arg::buf_mut(&mut wt),
                            ],
                        )
                    }?;
                    self.gemm.gemm(
                        self.gemm_arith,
                        GemmDims::new(Op::T, Op::N, b.nf, b.nf, 3 * b.np),
                        (1.0, 1.0),
                        MatRef::new(&phi, b.np, nc * b.np),
                        MatRef::new(&wt, 0, 3 * b.np),
                        MatMut::new(&mut vsub, 0, b.nf),
                    )?;
                }
                // SAFETY: matches xc_scatter; reads nf^2 values, adds to elements at the
                // block's functions (< nbf) of spin sp.
                unsafe {
                    gpu.launch(
                        &self.kernels.scatter,
                        Launch::linear(b.nf * b.nf, 256),
                        &[
                            Arg::I32(b.nf as i32),
                            Arg::I64(nbf as i64),
                            Arg::buf(&b.funcs),
                            Arg::buf(&vsub),
                            Arg::from(v.at(sp * n2)),
                        ],
                    )
                }?;
            }
            let s2 = mark(&mut marks)?;
            spans.push((2, s1, s2));
        }
        let me = gpu.mark()?;
        for &(what, s, e) in &spans {
            let dt = marks[s].seconds_until(&marks[e])?;
            match what {
                0 => times.colloc += dt,
                1 => times.density += dt,
                _ => times.potential += dt,
            }
        }
        times.total = m0.seconds_until(&me)?;
        self.last = times;
        Ok(exc)
    }
}
