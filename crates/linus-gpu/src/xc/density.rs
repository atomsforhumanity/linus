//! The grid densities of the XC engines (feature `gpu`): the arrays libxc reads (rho, grad
//! rho, sigma, tau in libxc's layout, `crate::xc`'s module docs) and the per-batch density
//! pass that fills them from density matrices, shared by the energy and potential (`GpuXc`),
//! the gradient (`crate::grad::xc`) and the Hessian's XC terms (`crate::hess::xc`).
//!
//! The pass walks an [`XcLayout`]'s chunks. Each chunk's blocks are collocated and compressed
//! (`XcLayout::prepare`); then per spin the batches' blocks of D are gathered
//! (`XcLayout::gather_d`), the engine's planned products X = phi D (and Y = grad phi D for
//! tau) run, and `xcb_densities` writes each batch's rho, grad rho and tau at its points. A
//! point belongs to one batch, so every value has one writer: the pass is deterministic, and
//! the engines' own launches and products are exactly those they issued before it was shared.

use crate::engine::{PhaseField, Timeline};
use crate::error::GpuError;
use crate::xc::layout::{per_batch, Collocation, XcLayout};
use crate::xc::{Prod, XcKind};
use crate::{Arg, DeviceBuf, Gpu, Kernel, Launch};

/// The grid's densities: `rho` (nsp per point), `grad` (3 nsp per point, GGA and meta-GGA),
/// `sigma` (1 or 3 per point, GGA and meta-GGA) and `tau` (nsp per point, meta-GGA); a
/// one-element placeholder for an ingredient the functional does not have.
pub(crate) struct GridDensities {
    pub(crate) rho: DeviceBuf<f64>,
    pub(crate) grad: DeviceBuf<f64>,
    pub(crate) sigma: DeviceBuf<f64>,
    pub(crate) tau: DeviceBuf<f64>,
}

impl GridDensities {
    /// Zeroed arrays for `npts` points, `nsp` spins and the ingredients `kind`.
    pub(crate) fn new(
        gpu: &Gpu,
        npts: usize,
        nsp: usize,
        kind: XcKind,
    ) -> Result<GridDensities, GpuError> {
        let (gga, mgga) = (kind.gga, kind.mgga);
        let nsig = if !gga {
            0
        } else if nsp == 1 {
            1
        } else {
            3
        };
        Ok(GridDensities {
            rho: gpu.alloc_zeros::<f64>(npts * nsp)?,
            grad: gpu.alloc_zeros::<f64>(if gga { npts * nsp * 3 } else { 1 })?,
            sigma: gpu.alloc_zeros::<f64>((npts * nsig).max(1))?,
            tau: gpu.alloc_zeros::<f64>(if mgga { npts * nsp } else { 1 })?,
        })
    }

    /// The writable rho, grad rho and tau (a density pass's output).
    pub(crate) fn out(&mut self) -> DensityOut<'_> {
        DensityOut {
            rho: &mut self.rho,
            grad: &mut self.grad,
            tau: &mut self.tau,
        }
    }

    /// sigma from grad rho (the `xc` module's `xc_sigma`, `kernel`), for GGAs and meta-GGAs.
    pub(crate) fn fill_sigma(
        &mut self,
        gpu: &Gpu,
        kernel: &Kernel,
        npts: usize,
        nsp: usize,
    ) -> Result<(), GpuError> {
        // SAFETY: matches xc_sigma; npts points of grad in, of sigma out.
        unsafe {
            gpu.launch(
                kernel,
                Launch::linear(npts, 256),
                &[
                    Arg::I64(npts as i64),
                    Arg::I32(nsp as i32),
                    Arg::buf(&self.grad),
                    Arg::buf_mut(&mut self.sigma),
                ],
            )
        }
    }
}

/// The arrays a density pass writes: rho, grad rho and tau, laid out as in [`GridDensities`].
pub(crate) struct DensityOut<'a> {
    pub(crate) rho: &'a mut DeviceBuf<f64>,
    pub(crate) grad: &'a mut DeviceBuf<f64>,
    pub(crate) tau: &'a mut DeviceBuf<f64>,
}

/// The products' operands of a density pass in its arithmetic: the gathered blocks of D, and
/// X = phi D and Y = grad phi D (tau), in FP64 or (with the compressed values' FP32 copy)
/// FP32.
pub(crate) enum DensityOps<'a> {
    Fp64 {
        dv: &'a mut DeviceBuf<f64>,
        xw: &'a DeviceBuf<f64>,
        yw: &'a DeviceBuf<f64>,
    },
    Fp32 {
        dv: &'a mut DeviceBuf<f32>,
        xw: &'a DeviceBuf<f32>,
        yw: &'a DeviceBuf<f32>,
    },
}

/// An engine's workspaces for a density pass (sized for its layout's largest block and chunk).
pub(crate) struct DensityWork<'a> {
    /// One block's collocation.
    pub(crate) phi: &'a mut DeviceBuf<f64>,
    /// The chunk's compressed values, and their FP32 copy when the engine has an FP32
    /// arithmetic (the FP32 products read it).
    pub(crate) phic: &'a mut DeviceBuf<f64>,
    pub(crate) phic32: Option<&'a mut DeviceBuf<f32>>,
    pub(crate) ops: DensityOps<'a>,
}

/// A density pass's timing: the call's timeline and the fields its chunks' collocation and
/// densities count to.
pub(crate) struct PassTiming<'t, 'g, T> {
    pub(crate) tl: &'t mut Timeline<'g, T>,
    pub(crate) colloc: PhaseField<T>,
    pub(crate) density: PhaseField<T>,
}

/// One per-batch density pass: the layout and how its blocks are collocated, the density
/// matrices and the spins, and the engine's products.
pub(crate) struct DensityPass<'a> {
    pub(crate) lay: &'a XcLayout,
    pub(crate) colloc: Collocation<'a>,
    /// The grid's coordinates.
    pub(crate) xyz: [&'a DeviceBuf<f64>; 3],
    /// D_s of spin s at element s nbf^2 (nbf x nbf, column-major); None: the blocks are
    /// collocated and compressed only (no occupied orbitals: the densities stay zero).
    pub(crate) d: Option<&'a DeviceBuf<f64>>,
    pub(crate) nbf: usize,
    pub(crate) nsp: usize,
    /// The densities' factor (2 for a closed shell's D of the occupied orbitals, 1 for a total
    /// or spin density).
    pub(crate) scale: f64,
    pub(crate) kind: XcKind,
    /// The cuBLAS GEMMs of grouped products.
    pub(crate) gemm: &'a crate::linalg::Gemm,
}

impl DensityPass<'_> {
    /// Runs the pass into `out` with the workspaces `work`: `plans(ci)` are chunk `ci`'s
    /// products [X = phi D, Y = grad phi D] (`Prod::None` for an absent one), `after(ci, s)`
    /// runs after spin s of chunk ci (an engine keeping X for a second pass), and `timing`
    /// (when given) counts each chunk's collocation and densities.
    pub(crate) fn run<'p, T>(
        &self,
        gpu: &Gpu,
        work: &mut DensityWork<'_>,
        out: DensityOut<'_>,
        mut timing: Option<PassTiming<'_, '_, T>>,
        plans: impl Fn(usize) -> [&'p Prod; 2],
        mut after: impl FnMut(usize, usize) -> Result<(), GpuError>,
    ) -> Result<(), GpuError> {
        let lay = self.lay;
        let nc = match self.colloc {
            Collocation::Own => lay.ncomp,
            Collocation::With(_, nc) => nc,
        };
        let (gga, mgga) = (self.kind.gga, self.kind.mgga);
        let n2 = self.nbf * self.nbf;
        let a64 = |b: &DeviceBuf<f64>, on: bool| if on { Arg::buf(b) } else { Arg::Ptr(0) };
        let a32 = |b: &DeviceBuf<f32>, on: bool| if on { Arg::buf(b) } else { Arg::Ptr(0) };
        for (ci, c) in lay.chunks.iter().enumerate() {
            let nbj = c.j1 - c.j0;
            if nbj == 0 {
                continue;
            }
            let s = timing.as_mut().map(|t| t.tl.mark()).transpose()?;
            lay.prepare(
                gpu,
                c,
                self.colloc,
                self.xyz,
                work.phi,
                Some(&mut *work.phic),
                work.phic32.as_deref_mut(),
            )?;
            let s1 = timing.as_mut().map(|t| t.tl.mark()).transpose()?;
            if let (Some(t), Some(s), Some(s1)) = (timing.as_mut(), s, s1) {
                t.tl.span(t.colloc, s, s1);
            }
            let Some(d) = self.d else {
                continue;
            };
            let [px, py] = plans(ci);
            for sp in 0..self.nsp {
                let dsp = d.at(sp * n2);
                match &mut work.ops {
                    DensityOps::Fp64 { dv, .. } => {
                        lay.gather_d(gpu, c, self.nbf, dsp, Some(&mut **dv), None)?
                    }
                    DensityOps::Fp32 { dv, .. } => {
                        lay.gather_d(gpu, c, self.nbf, dsp, None, Some(&mut **dv))?
                    }
                }
                // SAFETY: the engine's plans address its workspaces of this chunk.
                unsafe {
                    px.run(gpu, &lay.vg, self.gemm, false, 0.0)?;
                    py.run(gpu, &lay.vg, self.gemm, false, 0.0)?;
                }
                let (x64, y64, x32, y32) = match &work.ops {
                    DensityOps::Fp64 { xw, yw, .. } => {
                        (a64(xw, true), a64(yw, mgga), Arg::Ptr(0), Arg::Ptr(0))
                    }
                    DensityOps::Fp32 { xw, yw, .. } => {
                        (Arg::Ptr(0), Arg::Ptr(0), a32(xw, true), a32(yw, mgga))
                    }
                };
                let p32 = match (&work.ops, work.phic32.as_deref()) {
                    (DensityOps::Fp32 { .. }, Some(p)) => Arg::buf(p),
                    _ => Arg::Ptr(0),
                };
                // SAFETY: matches xcb_densities; reads the chunk's compressed values (nc
                // components) and X (Y), writes spin sp at the batches' points of the grid
                // arrays.
                unsafe {
                    gpu.launch(
                        &lay.k.densities,
                        per_batch(c.max_np, nbj, 128),
                        &[
                            Arg::I32(nc as i32),
                            Arg::I32(self.nsp as i32),
                            Arg::I32(sp as i32),
                            Arg::F64(self.scale),
                            Arg::I32(gga as i32),
                            Arg::I32(mgga as i32),
                            Arg::buf(&lay.bd),
                            Arg::I32(c.j0 as i32),
                            Arg::buf(work.phic),
                            x64,
                            y64,
                            p32,
                            x32,
                            y32,
                            Arg::buf_mut(out.rho),
                            Arg::buf_mut(out.grad),
                            Arg::buf_mut(out.tau),
                        ],
                    )
                }?;
                after(ci, sp)?;
            }
            if let (Some(t), Some(s1)) = (timing.as_mut(), s1) {
                t.tl.close(t.density, s1)?;
            }
        }
        Ok(())
    }
}
