//! The XC kernel contraction on the device: `GpuXcKernel` (feature `gpu`; the algorithm is in
//! the [module docs](super)), and the weighted effective second derivatives it keeps per
//! point ([`kernel_weights`], host code, the formulas of linus's `weighted_derivs` and
//! `weighted_derivs_spin`).

use crate::cpu::XcOutput;

/// Index of v2sigma2 (aa_aa, aa_ab, aa_bb, ab_ab, ab_bb, bb_bb) for sigma indices t, t2.
const SIG2: [[usize; 3]; 3] = [[0, 1, 2], [1, 3, 4], [2, 4, 5]];

/// Variables per spin of a functional kind: rho (1), + grad rho (4), + tau (5).
pub fn vars_per_spin(gga: bool, mgga: bool) -> usize {
    if mgga {
        5
    } else if gga {
        4
    } else {
        1
    }
}

/// The weighted effective second derivatives wF = w d2f/du2 of `np` points (wF\[(c nu + e) np +
/// g\], nu = nsp `vars_per_spin`) from libxc's order-2 output `out` (unpolarized for one spin
/// channel, polarized for two), the weights `w` and the density gradients `grad`
/// (grad\[(s 3 + k) np + g\]; for one channel the total density's). The variables are
/// u_s = (rho_s, grad rho_s, tau_s), c = s nu_s + (0: rho, 1..3: grad, 4: tau), with the
/// sigmas folded in by the chain rule, exactly as linus's CPU kernel contraction does.
pub fn kernel_weights(
    out: &XcOutput,
    nsp: usize,
    gga: bool,
    mgga: bool,
    w: &[f64],
    grad: &[f64],
    np: usize,
) -> Vec<f64> {
    if nsp == 1 {
        weights_closed(out, gga, mgga, w, grad, np)
    } else {
        weights_spin(out, gga, mgga, w, grad, np)
    }
}

/// [`kernel_weights`] of a closed shell (linus's `weighted_derivs`).
fn weights_closed(
    out: &XcOutput,
    gga: bool,
    mgga: bool,
    w: &[f64],
    grad: &[f64],
    np: usize,
) -> Vec<f64> {
    let nu = vars_per_spin(gga, mgga);
    let mut wf = vec![0.0; nu * nu * np];
    let fidx = |c: usize, d: usize| (c * nu + d) * np;
    let v2rho2 = out.v2rho2();
    for g in 0..np {
        wf[fidx(0, 0) + g] = w[g] * v2rho2[g];
    }
    if gga {
        let (vs, vrs, vss) = (out.vsigma(), out.v2rhosigma(), out.v2sigma2());
        for g in 0..np {
            let ww = w[g];
            let gr = [grad[g], grad[np + g], grad[2 * np + g]];
            for k in 0..3 {
                let frk = ww * 2.0 * vrs[g] * gr[k];
                wf[fidx(0, 1 + k) + g] = frk;
                wf[fidx(1 + k, 0) + g] = frk;
                for l in 0..3 {
                    wf[fidx(1 + k, 1 + l) + g] = ww
                        * (4.0 * vss[g] * gr[k] * gr[l] + if k == l { 2.0 * vs[g] } else { 0.0 });
                }
            }
        }
    }
    if mgga {
        let (vrt, vst, vtt) = (out.v2rhotau(), out.v2sigmatau(), out.v2tau2());
        for g in 0..np {
            let ww = w[g];
            wf[fidx(0, 4) + g] = ww * vrt[g];
            wf[fidx(4, 0) + g] = ww * vrt[g];
            wf[fidx(4, 4) + g] = ww * vtt[g];
            for k in 0..3 {
                let f = ww * 2.0 * vst[g] * grad[k * np + g];
                wf[fidx(1 + k, 4) + g] = f;
                wf[fidx(4, 1 + k) + g] = f;
            }
        }
    }
    wf
}

/// [`kernel_weights`] of an open shell (linus's `weighted_derivs_spin`).
// the spin and component indices address several arrays at once (the CPU's formulas, index
// for index)
#[allow(clippy::needless_range_loop)]
fn weights_spin(
    out: &XcOutput,
    gga: bool,
    mgga: bool,
    w: &[f64],
    grad: &[f64],
    np: usize,
) -> Vec<f64> {
    let ns = vars_per_spin(gga, mgga);
    let nu = 2 * ns;
    let mut wf = vec![0.0; nu * nu * np];
    let fidx = |c: usize, d: usize| (c * nu + d) * np;
    let v2rho2 = out.v2rho2();
    for g in 0..np {
        let ww = w[g];
        for s in 0..2 {
            for s2 in 0..2 {
                // v2rho2 = (aa, ab, bb): index s + s2
                wf[fidx(s * ns, s2 * ns) + g] = ww * v2rho2[3 * g + s + s2];
            }
        }
    }
    // dsigma_t / d g_{s,k} at point g
    let ds = |g: usize, t: usize, s: usize, k: usize| -> f64 {
        let gr = |s: usize, k: usize| grad[(s * 3 + k) * np + g];
        match (t, s) {
            (0, 0) => 2.0 * gr(0, k),
            (1, 0) => gr(1, k),
            (1, 1) => gr(0, k),
            (2, 1) => 2.0 * gr(1, k),
            _ => 0.0,
        }
    };
    if gga {
        let (vs, vrs, vss) = (out.vsigma(), out.v2rhosigma(), out.v2sigma2());
        for g in 0..np {
            let ww = w[g];
            let vsig = [vs[3 * g], vs[3 * g + 1], vs[3 * g + 2]];
            // second derivatives of the sigmas: d2 sigma_aa = 2, d2 sigma_ab = 1 (a-b),
            // d2 sigma_bb = 2
            let d2 = [[2.0 * vsig[0], vsig[1]], [vsig[1], 2.0 * vsig[2]]];
            for s in 0..2 {
                for k in 0..3 {
                    let c = s * ns + 1 + k;
                    for sr in 0..2 {
                        // rho_sr - g_{s,k}: v2rhosigma = (a_aa, a_ab, a_bb, b_aa, b_ab, b_bb)
                        let f = ww
                            * (0..3)
                                .map(|t| vrs[6 * g + 3 * sr + t] * ds(g, t, s, k))
                                .sum::<f64>();
                        wf[fidx(sr * ns, c) + g] = f;
                        wf[fidx(c, sr * ns) + g] = f;
                    }
                    for s2 in 0..2 {
                        for l in 0..3 {
                            let c2 = s2 * ns + 1 + l;
                            let mut f = 0.0;
                            for t in 0..3 {
                                let a = ds(g, t, s, k);
                                if a == 0.0 {
                                    continue;
                                }
                                for t2 in 0..3 {
                                    f += vss[6 * g + SIG2[t][t2]] * a * ds(g, t2, s2, l);
                                }
                            }
                            if k == l {
                                f += d2[s][s2];
                            }
                            wf[fidx(c, c2) + g] = ww * f;
                        }
                    }
                }
            }
        }
    }
    if mgga {
        let (vrt, vst, vtt) = (out.v2rhotau(), out.v2sigmatau(), out.v2tau2());
        for g in 0..np {
            let ww = w[g];
            for s in 0..2 {
                let ct = s * ns + 4;
                for s2 in 0..2 {
                    let ct2 = s2 * ns + 4;
                    // v2tau2 = (aa, ab, bb); v2rhotau = (a_a, a_b, b_a, b_b) (rho spin, tau
                    // spin)
                    wf[fidx(ct, ct2) + g] = ww * vtt[3 * g + s + s2];
                    let f = ww * vrt[4 * g + 2 * s2 + s];
                    wf[fidx(s2 * ns, ct) + g] = f;
                    wf[fidx(ct, s2 * ns) + g] = f;
                    // v2sigmatau = (aa_a, aa_b, ab_a, ab_b, bb_a, bb_b) (sigma, tau spin)
                    for k in 0..3 {
                        let c = s2 * ns + 1 + k;
                        let f = ww
                            * (0..3)
                                .map(|t| vst[6 * g + 2 * t + s] * ds(g, t, s2, k))
                                .sum::<f64>();
                        wf[fidx(c, ct) + g] = f;
                        wf[fidx(ct, c) + g] = f;
                    }
                }
            }
        }
    }
    wf
}

#[cfg(feature = "gpu")]
pub use device::*;

#[cfg(feature = "gpu")]
mod device {
    use super::{kernel_weights, vars_per_spin};
    use crate::blas::{GemmDims, MatMut, MatPtr, MatRef, Op};
    use crate::df::Factor;
    use crate::emu::{Ozaki, OzakiWork};
    use crate::engine::Timeline;
    use crate::error::GpuError;
    use crate::kernels::registry::KernelModule;
    use crate::linalg::{Arith, Gemm};
    use crate::xc::layout::Collocation;
    use crate::xc::{GpuXc, XcMath};
    use crate::DevPtr;
    use crate::{Arg, DeviceBuf, Gpu, Kernel, Launch};
    use std::borrow::Cow;
    use std::time::Instant;

    const KERNELS: &str = include_str!("../../device/response/xc.cu");

    /// The XC kernel's module in FP64 (`XCK_T` double, the default).
    pub(crate) const MODULE: KernelModule = KernelModule::new(
        "resp_xc",
        || Cow::Borrowed(KERNELS),
        &[
            "xck_gather_rows",
            "xck_scatter_rows",
            "xck_ground",
            "xck_pert",
            "xck_pert_gga",
        ],
    );

    /// The same kernels in FP32 (`XCK_T` float, the entry points suffixed `_f32`).
    pub(crate) const MODULE_F32: KernelModule = KernelModule::new(
        "resp_xc32",
        || {
            Cow::Owned(format!(
                "#define XCK_T float\n#define XCK_SFX(n) n##_f32\n{KERNELS}"
            ))
        },
        &[
            "xck_gather_rows_f32",
            "xck_scatter_rows_f32",
            "xck_pert_f32",
            "xck_pert_gga_f32",
        ],
    );

    /// Settings of a [`GpuXcKernel`].
    #[derive(Clone, Debug)]
    pub struct GpuXcKernelOptions {
        /// Device bytes for the per-block perturbation buffers of a call (sets the
        /// perturbation groups).
        pub workspace: usize,
        /// The two large GEMMs per block (P = F A and dV = F^T Z) with cuBLAS's fixed-point
        /// FP64 emulation at this many mantissa bits (None: native FP64).
        pub emulated_bits: Option<i32>,
        /// The two large GEMMs per block (P = F A and dV = F^T Z) in the Ozaki scheme
        /// ([`crate::emu`]) with this many INT8 slices, both operands sliced per block (FP64
        /// blocks only; takes precedence over `emulated_bits`).
        pub ozaki_slices: Option<usize>,
        /// FP64, or (the precision study) [`XcMath::Fp32`]: the products in FP32 (values,
        /// orbitals and factors rounded to FP32, each sum over at most a batch's points or
        /// functions in FP32), the batches added in FP64. Needs a [`GpuXc`] with per-batch
        /// screening built for FP32 values (`XcMath::Fp32` or `Fp32Potential`).
        pub math: XcMath,
        /// Keep the ground state of every block (its collocation and Phi_s = F C_s, which do
        /// not depend on the perturbations) on the device across contractions, made by the
        /// first contraction when it fits with this many device bytes left free beside it
        /// (None: made again in every contraction). It pays where a call has few
        /// perturbations (an electric-field CPHF's three) and the solve many calls. The
        /// products are the same bits either way. Blocks only (not the per-batch path).
        pub ground_cache: Option<usize>,
        /// GGAs (no tau) on blocks: P = F A of the values only and the gradient terms through
        /// Q = Phi A^T and G = t_k d_k phi (half the GEMM work; default); false: P = F A for
        /// every component, as for meta-GGAs (fewer launches and no pointwise loop over the
        /// functions).
        pub q_route: bool,
    }

    impl Default for GpuXcKernelOptions {
        fn default() -> Self {
            GpuXcKernelOptions {
                workspace: 4 << 30,
                emulated_bits: None,
                ozaki_slices: None,
                math: XcMath::Fp64,
                ground_cache: None,
                q_route: true,
            }
        }
    }

    /// The ground state of every block kept across contractions
    /// ([`GpuXcKernelOptions::ground_cache`]): the collocation (nc np x nf) and Phi_s = F C_s
    /// (nc np x o_s) of each block side by side, each from an offset aligned to 64 elements.
    struct GroundCache {
        phi: DeviceBuf<f64>,
        psi: Vec<DeviceBuf<f64>>,
        /// Per block: the offset of its collocation, and of its Phi_s per channel.
        at: Vec<(usize, Vec<usize>)>,
        /// Made by a contraction (the next ones read it).
        filled: bool,
    }

    impl GroundCache {
        /// The buffers for `xc`'s blocks and the channels' occupied orbitals `occ`, when they
        /// fit with `leave` device bytes free beside them (None: they do not).
        fn new(
            gpu: &Gpu,
            xc: &GpuXc,
            occ: &[usize],
            leave: usize,
        ) -> Result<Option<GroundCache>, GpuError> {
            let al = |x: usize| x.div_ceil(64) * 64;
            let nc = xc.ncomp;
            let (mut po, mut so) = (0usize, vec![0usize; occ.len()]);
            let mut at = Vec::with_capacity(xc.blocks.len());
            for b in &xc.blocks {
                at.push((po, so.clone()));
                if b.nf == 0 {
                    continue;
                }
                po += al(nc * b.np * b.nf);
                for (s, &o) in occ.iter().enumerate() {
                    so[s] += al(nc * b.np * o);
                }
            }
            let bytes = 8 * (po + so.iter().sum::<usize>());
            if !gpu.budget().fits(bytes + leave) {
                return Ok(None);
            }
            Ok(Some(GroundCache {
                phi: gpu.alloc_zeros::<f64>(po.max(1))?,
                psi: so
                    .iter()
                    .map(|&n| gpu.alloc_zeros::<f64>(n.max(1)))
                    .collect::<Result<_, _>>()?,
                at,
                filled: false,
            }))
        }

        fn bytes(&self) -> usize {
            8 * (self.phi.len() + self.psi.iter().map(|p| p.len()).sum::<usize>())
        }
    }

    /// The FP64 arithmetic of `bits` (None: native).
    fn cublas(bits: Option<i32>) -> Arith {
        match bits {
            None => Arith::Fp64,
            Some(bits) => Arith::Fp64Emulated {
                bits: bits.clamp(0, u8::MAX as i32) as u8,
            },
        }
    }

    /// Setup times (seconds, wall clock).
    #[derive(Clone, Debug, Default)]
    pub struct GpuXcKernelSetupTimes {
        /// Ground-state densities on the device.
        pub density: f64,
        /// Densities to the host, libxc (order 2), the weights, and the weights to the device.
        pub functional: f64,
        /// Of which the parallel libxc and weights section (wall clock).
        pub libxc: f64,
        /// Bytes moved between host and device for the functional.
        pub transfer_bytes: usize,
        pub total: f64,
    }

    /// Times of the last contraction (seconds, from events).
    #[derive(Clone, Debug, Default)]
    pub struct GpuXcKernelCallTimes {
        pub total: f64,
        /// Factors to the device.
        pub upload: f64,
        pub colloc: f64,
        /// Row gathers, Phi = F C and P = F A.
        pub gemm_in: f64,
        /// Of `gemm_in`, the ground state's row gathers and Phi = F C.
        pub ground: f64,
        /// The pointwise kernel.
        pub pointwise: f64,
        /// F^T Z and the scatter.
        pub gemm_out: f64,
        /// Results to the host.
        pub download: f64,
        /// Perturbations per group.
        pub group: usize,
    }

    impl crate::engine::Phases for GpuXcKernelCallTimes {
        /// The perturbed products are `gemm_in` without the ground state's part.
        fn phases(&self) -> impl IntoIterator<Item = (&'static str, f64)> {
            [
                ("total", self.total),
                ("upload", self.upload),
                ("colloc", self.colloc),
                ("ground", self.ground),
                ("pert_in", self.gemm_in - self.ground),
                ("pointwise", self.pointwise),
                ("pert_out", self.gemm_out),
                ("download", self.download),
            ]
        }
    }

    struct Kernels {
        gather: Kernel,
        scatter: Kernel,
        ground: Kernel,
        pert: Kernel,
        pert_gga: Kernel,
        /// The single-precision module's (FP32 values and products).
        gather32: Kernel,
        scatter32: Kernel,
        pert32: Kernel,
        pert_gga32: Kernel,
    }

    struct Chan {
        o: usize,
        c: DeviceBuf<f64>,
    }

    /// The XC kernel contraction dV_s\[dD\] C_s of factored first-order densities (see the
    /// module docs) on the grid and blocks of a [`GpuXc`].
    pub struct GpuXcKernel {
        xc: GpuXc,
        chans: Vec<Chan>,
        /// Variables per spin (1, 4, 5).
        ns: usize,
        /// wF\[(c nu + e) npts + P\].
        wf: DeviceBuf<f64>,
        /// The GEMMs: FP64 and FP32, and the large P and dV GEMMs in `p_arith`.
        gemm: Gemm,
        /// The arithmetic of the large P and dV GEMMs (emulated FP64 if set).
        p_arith: Arith,
        /// The Ozaki scheme for the P and dV GEMMs ([`GpuXcKernelOptions::ozaki_slices`]).
        ozaki: Option<(Ozaki, usize)>,
        oz_work: OzakiWork,
        kernels: Kernels,
        opts: GpuXcKernelOptions,
        /// The kept ground state ([`GpuXcKernelOptions::ground_cache`]), and whether a
        /// contraction has tried to make it.
        ground: Option<GroundCache>,
        ground_tried: bool,
        pub setup_times: GpuXcKernelSetupTimes,
        pub last: GpuXcKernelCallTimes,
    }

    impl GpuXcKernel {
        /// The kernel of `xc`'s functional at the ground state of the occupied orbitals `occ`
        /// (one channel: a closed shell, D = 2 C C^T; two: the spins of an open shell): the
        /// second derivatives are evaluated here, once (libxc on the host; current rayon
        /// pool). LDA, GGA and meta-GGA (tau).
        pub fn new(
            gpu: &Gpu,
            xc: GpuXc,
            occ: &[Factor<'_>],
            opts: &GpuXcKernelOptions,
        ) -> Result<GpuXcKernel, GpuError> {
            let t_all = Instant::now();
            let mut times = GpuXcKernelSetupTimes::default();
            let nsp = occ.len();
            if !(1..=2).contains(&nsp) {
                return Err(GpuError::Contract(format!(
                    "XC kernel: {nsp} channels (one or two)"
                )));
            }
            let (nbf, npts) = (xc.nbf, xc.npts);
            for f in occ {
                if f.c.len() != nbf * f.ncol {
                    return Err(GpuError::Contract(
                        "XC kernel: occupied orbitals of the wrong size".into(),
                    ));
                }
            }
            let kind = xc.kind();
            if kind.none {
                return Err(GpuError::Contract(
                    "XC kernel: the functional has no semilocal part".into(),
                ));
            }
            let (gga, mgga) = (kind.gga, kind.mgga);
            let ns = vars_per_spin(gga, mgga);
            let nu = nsp * ns;
            let nc = xc.ncomp;
            let src = MODULE.source();
            let k = |e: &str| gpu.kernel(MODULE.name, &src, e);
            let src32 = MODULE_F32.source();
            let k32 = |e: &str| gpu.kernel(MODULE_F32.name, &src32, e);
            let kernels = Kernels {
                gather: k("xck_gather_rows")?,
                scatter: k("xck_scatter_rows")?,
                ground: k("xck_ground")?,
                pert: k("xck_pert")?,
                pert_gga: k("xck_pert_gga")?,
                gather32: k32("xck_gather_rows_f32")?,
                scatter32: k32("xck_scatter_rows_f32")?,
                pert32: k32("xck_pert_f32")?,
                pert_gga32: k32("xck_pert_gga_f32")?,
            };
            match (opts.math, xc.bat.as_ref()) {
                (XcMath::Fp64, _) => {}
                (XcMath::Fp32, Some(b)) if b.has_single() => {}
                (m, _) => {
                    return Err(GpuError::NotCovered(format!(
                        "XC kernel: {m:?} needs a per-batch XC engine built for FP32 values"
                    )))
                }
            }
            let mut gemm = Gemm::new(gpu)?;
            let p_arith = cublas(opts.emulated_bits);
            gemm.prepare(gpu, p_arith)?;
            let chans: Vec<Chan> = occ
                .iter()
                .map(|f| {
                    Ok(Chan {
                        o: f.ncol,
                        c: gpu.upload(if f.ncol == 0 { &[0.0] } else { f.c })?,
                    })
                })
                .collect::<Result<_, GpuError>>()?;

            // ground-state densities on the device
            let t = Instant::now();
            let mut rho = gpu.alloc_zeros::<f64>(npts * nsp)?;
            let mut grad = gpu.alloc_zeros::<f64>(if gga { npts * nsp * 3 } else { 1 })?;
            let mut tau = gpu.alloc_zeros::<f64>(if mgga { npts * nsp } else { 1 })?;
            let omax = occ.iter().map(|f| f.ncol).max().unwrap_or(0);
            let mut phi = gpu.alloc_zeros::<f64>((xc.max_nf * nc * xc.max_np).max(1))?;
            let mut csub = gpu.alloc_zeros::<f64>((xc.max_nf * omax).max(1))?;
            let mut psi = gpu.alloc_zeros::<f64>((nc * xc.max_np * omax).max(1))?;
            let scale = if nsp == 1 { 2.0 } else { 1.0 };
            for b in &xc.blocks {
                if b.nf == 0 {
                    continue;
                }
                xc.collocate(gpu, b, &mut phi)?;
                for (s, ch) in chans.iter().enumerate() {
                    if ch.o > 0 {
                        gather_rows(
                            gpu,
                            &kernels.gather,
                            b.nf,
                            ch.o,
                            nbf,
                            &b.funcs,
                            &ch.c,
                            &mut csub,
                        )?;
                        gemm.gemm(
                            Arith::Fp64,
                            GemmDims::new(Op::N, Op::N, nc * b.np, ch.o, b.nf),
                            (1.0, 0.0),
                            MatRef::new(&phi, 0, nc * b.np),
                            MatRef::new(&csub, 0, b.nf),
                            MatMut::new(&mut psi, 0, nc * b.np),
                        )?;
                    }
                    // SAFETY: matches xck_ground; reads the block's Phi (nc np x o), writes
                    // spin s at the block's points of the grid arrays.
                    unsafe {
                        gpu.launch(
                            &kernels.ground,
                            Launch::linear(b.np, 128),
                            &[
                                Arg::I32(b.np as i32),
                                Arg::I64(b.p0 as i64),
                                Arg::I32(nc as i32),
                                Arg::I32(nsp as i32),
                                Arg::I32(s as i32),
                                Arg::I32(ch.o as i32),
                                Arg::F64(scale),
                                Arg::I32(gga as i32),
                                Arg::I32(mgga as i32),
                                Arg::buf(&psi),
                                Arg::buf_mut(&mut rho),
                                Arg::buf_mut(&mut grad),
                                Arg::buf_mut(&mut tau),
                            ],
                        )
                    }?;
                }
            }
            drop((phi, csub, psi));
            gpu.synchronize()?;
            times.density = t.elapsed().as_secs_f64();

            // the functional (order 2) and the weights on the host, in chunks of points
            let t = Instant::now();
            let hrho = gpu.download(&rho)?;
            let hgrad = if gga { gpu.download(&grad)? } else { vec![] };
            let htau = if mgga { gpu.download(&tau)? } else { vec![] };
            let hw = gpu.download(&xc.w)?;
            drop((rho, grad, tau));
            let tl = Instant::now();
            let parts = xc
                .functional
                .second(nsp, npts, &hrho, &hgrad, &htau, |pts, out, gr| {
                    let w = kernel_weights(out, nsp, gga, mgga, &hw[pts.clone()], gr, pts.len());
                    (pts, w)
                })?;
            times.libxc = tl.elapsed().as_secs_f64();
            let mut wfh = vec![0.0; nu * nu * npts];
            for (pts, p) in parts {
                let (p0, p1) = (pts.start, pts.end);
                let np = p1 - p0;
                for ce in 0..nu * nu {
                    wfh[ce * npts + p0..ce * npts + p1].copy_from_slice(&p[ce * np..(ce + 1) * np]);
                }
            }
            let wf = gpu.upload(&wfh)?;
            times.transfer_bytes =
                8 * (hrho.len() + hgrad.len() + htau.len() + hw.len() + wfh.len());
            times.functional = t.elapsed().as_secs_f64();
            times.total = t_all.elapsed().as_secs_f64();
            Ok(GpuXcKernel {
                xc,
                chans,
                ns,
                wf,
                gemm,
                p_arith,
                ozaki: match opts.ozaki_slices {
                    Some(sl) => Some((Ozaki::new(gpu)?, sl)),
                    None => None,
                },
                oz_work: OzakiWork::default(),
                kernels,
                opts: opts.clone(),
                ground: None,
                ground_tried: false,
                setup_times: times,
                last: GpuXcKernelCallTimes::default(),
            })
        }

        /// Native (None) or emulated FP64 for the large GEMMs of the following contractions.
        pub fn set_emulation(&mut self, gpu: &Gpu, bits: Option<i32>) -> Result<(), GpuError> {
            self.p_arith = cublas(bits);
            self.gemm.prepare(gpu, self.p_arith)?;
            self.opts.emulated_bits = bits;
            Ok(())
        }

        /// [`GpuXcKernelOptions::ground_cache`] for the following contractions (None: the kept
        /// ground state, if any, is freed).
        pub fn set_ground_cache(&mut self, leave: Option<usize>) {
            self.opts.ground_cache = leave;
            if leave.is_none() {
                self.ground = None;
            }
            self.ground_tried = false;
        }

        /// Device bytes of the kept ground state (0: none).
        pub fn ground_cache_bytes(&self) -> usize {
            self.ground.as_ref().map_or(0, |g| g.bytes())
        }

        /// The XC engine back (its grid and blocks).
        pub fn into_xc(self) -> GpuXc {
            self.xc
        }

        /// dV_s\[dD\] C_s for the factors `a[s]` (n x (nd o_s) column-major, A_{s,d} at columns
        /// d o_s..) of `nd` perturbations, dD_{s,d} = A_{s,d} C_s^T + C_s A_{s,d}^T, in the same
        /// layout. One channel: the potential of the total first-order density 2 dD (both
        /// spins carry dD), as linus's `kernel_contract_occ`.
        pub fn contract(
            &mut self,
            gpu: &Gpu,
            a: &[&[f64]],
            nd: usize,
        ) -> Result<Vec<Vec<f64>>, GpuError> {
            let nbf = self.xc.nbf;
            let nsp = self.chans.len();
            if a.len() != nsp {
                return Err(GpuError::Contract(format!(
                    "XC kernel: {} lists of factors for {nsp} channels",
                    a.len()
                )));
            }
            for (s, ch) in self.chans.iter().enumerate() {
                if a[s].len() != nbf * nd * ch.o {
                    return Err(GpuError::Contract(format!(
                        "XC kernel: channel {s} has {} factor values, not {nbf} x {nd} x {}",
                        a[s].len(),
                        ch.o
                    )));
                }
            }
            let mut times = GpuXcKernelCallTimes::default();
            if nd == 0 {
                self.last = times;
                return Ok(self.chans.iter().map(|_| Vec::new()).collect());
            }
            if self.xc.bat.is_some() {
                return self.contract_batched(gpu, a, nd);
            }
            let m0 = gpu.mark()?;
            let occ: Vec<usize> = self.chans.iter().map(|c| c.o).collect();
            // the kept ground state (made by the first call that fits it), taken out for the
            // call: read when filled, else filled now
            if let (Some(leave), None, false) =
                (self.opts.ground_cache, &self.ground, self.ground_tried)
            {
                self.ground_tried = true;
                self.ground = GroundCache::new(gpu, &self.xc, &occ, leave)?;
            }
            let mut ground = self.ground.take();
            let filled = ground.as_ref().is_some_and(|g| g.filled);
            let xc = &self.xc;
            let nc = xc.ncomp;
            let (gga, mgga) = (xc.kind().gga, xc.kind().mgga);
            let omax = occ.iter().copied().max().unwrap_or(0);
            let acat: Vec<DeviceBuf<f64>> = (0..nsp)
                .map(|s| gpu.upload(if a[s].is_empty() { &[0.0] } else { a[s] }))
                .collect::<Result<_, _>>()?;
            // written by the scatter kernel at device addresses
            let out: Vec<DeviceBuf<f64>> = occ
                .iter()
                .map(|&o| gpu.alloc_zeros::<f64>((nbf * nd * o).max(1)))
                .collect::<Result<_, _>>()?;
            let m_up = gpu.mark()?;
            times.upload = m0.seconds_until(&m_up)?;
            // GGA (no tau): the CPU's form with half the GEMM work (P of the values only, the
            // gradient terms through Q = Phi A^T and G = t_k d_k phi); else P = F A for every
            // component (and for GGAs without `q_route`)
            let qroute = gga && !mgga && self.opts.q_route;
            let nr = if qroute { 1 } else { nc };
            let (np_max, nf_max) = (xc.max_np, xc.max_nf);
            // per perturbation and spin: P/Z (nr np x o), A_sub / V (nf x o), Q/G (np x nf)
            let per: usize = occ
                .iter()
                .filter(|&&o| o > 0)
                .map(|&o| (nr * np_max + nf_max) * o + if qroute { np_max * nf_max } else { 0 })
                .sum();
            let group = crate::response::kernel_group(per, nd, self.opts.workspace);
            times.group = group;
            // workspaces: collocation, per spin C_sub, Phi (not with a kept ground state), P/Z,
            // Q/G and the gathered factors / products (A_sub, then V)
            let ws = |len: usize| if ground.is_some() { 1 } else { len.max(1) };
            let mut phi = gpu.alloc_zeros::<f64>(ws(xc.max_nf * nc * xc.max_np))?;
            let mut csub =
                gpu.alloc_zeros::<f64>(if filled { 1 } else { (xc.max_nf * omax).max(1) })?;
            let mut psi: Vec<DeviceBuf<f64>> = occ
                .iter()
                .map(|&o| gpu.alloc_zeros::<f64>(ws(nc * xc.max_np * o)))
                .collect::<Result<_, _>>()?;
            let mut pz: Vec<DeviceBuf<f64>> = occ
                .iter()
                .map(|&o| gpu.alloc_zeros::<f64>((nr * np_max * group * o).max(1)))
                .collect::<Result<_, _>>()?;
            let mut qg: Vec<DeviceBuf<f64>> = occ
                .iter()
                .map(|&o| {
                    let len = if qroute && o > 0 {
                        np_max * nf_max * group
                    } else {
                        1
                    };
                    gpu.alloc_zeros::<f64>(len)
                })
                .collect::<Result<_, _>>()?;
            let mut sub: Vec<DeviceBuf<f64>> = occ
                .iter()
                .map(|&o| gpu.alloc_zeros::<f64>((xc.max_nf * group * o).max(1)))
                .collect::<Result<_, _>>()?;
            let scale = if nsp == 1 { 2.0 } else { 1.0 };
            let mut tl = Timeline::<GpuXcKernelCallTimes>::new(gpu);
            for (bi, b) in xc.blocks.iter().enumerate() {
                if b.nf == 0 {
                    continue;
                }
                let s0 = tl.mark()?;
                if !filled {
                    match ground.as_mut() {
                        Some(g) => xc.collocate_at(gpu, b, &mut g.phi, g.at[bi].0)?,
                        None => xc.collocate(gpu, b, &mut phi)?,
                    }
                }
                let s1 = tl.mark()?;
                tl.span(|t| &mut t.colloc, s0, s1);
                // ground state of the block: Phi_s = F C_s
                for (s, ch) in self.chans.iter().enumerate() {
                    if ch.o == 0 || filled {
                        continue;
                    }
                    gather_rows(
                        gpu,
                        &self.kernels.gather,
                        b.nf,
                        ch.o,
                        nbf,
                        &b.funcs,
                        &ch.c,
                        &mut csub,
                    )?;
                    let ld = nc * b.np;
                    match ground.as_mut() {
                        Some(g) => self.gemm.gemm(
                            Arith::Fp64,
                            GemmDims::new(Op::N, Op::N, ld, ch.o, b.nf),
                            (1.0, 0.0),
                            MatRef::new(&g.phi, g.at[bi].0, ld),
                            MatRef::new(&csub, 0, b.nf),
                            MatMut::new(&mut g.psi[s], g.at[bi].1[s], ld),
                        )?,
                        None => self.gemm.gemm(
                            Arith::Fp64,
                            GemmDims::new(Op::N, Op::N, ld, ch.o, b.nf),
                            (1.0, 0.0),
                            MatRef::new(&phi, 0, ld),
                            MatRef::new(&csub, 0, b.nf),
                            MatMut::new(&mut psi[s], 0, ld),
                        )?,
                    }
                }
                let s2 = tl.mark()?;
                tl.span(|t| &mut t.gemm_in, s1, s2);
                tl.span(|t| &mut t.ground, s1, s2);
                // the block's collocation and Phi_s as (buffer, offset): kept or just made
                let (phi_v, phi_o) = match &ground {
                    Some(g) => (&g.phi, g.at[bi].0),
                    None => (&phi, 0),
                };
                let psi_v = |s: usize| match &ground {
                    Some(g) => (&g.psi[s], g.at[bi].1[s]),
                    None => (&psi[s], 0),
                };
                let at = |v: (&DeviceBuf<f64>, usize)| v.0.at(v.1).addr();
                for d0 in (0..nd).step_by(group) {
                    let ndg = group.min(nd - d0);
                    let g0 = tl.mark()?;
                    // P_s = F A_s,sub for the group's perturbations
                    for (s, &o) in occ.iter().enumerate() {
                        if o == 0 {
                            continue;
                        }
                        let w = ndg * o;
                        gather_rows_at(
                            gpu,
                            &self.kernels.gather,
                            b.nf,
                            w,
                            nbf,
                            &b.funcs,
                            &acat[s],
                            d0 * o * nbf,
                            &mut sub[s],
                        )?;
                        // P = F A (values only on the GGA route: rows 0..np of F)
                        match &self.ozaki {
                            Some((oz, sl)) => oz.dgemm_at(
                                gpu,
                                *sl,
                                Op::N,
                                Op::N,
                                nr * b.np,
                                w,
                                b.nf,
                                1.0,
                                (phi_v, phi_o, nc * b.np),
                                (&sub[s], 0, b.nf),
                                0.0,
                                (&mut pz[s], 0, nr * b.np),
                                &mut self.oz_work,
                            )?,
                            None => self.gemm.gemm(
                                self.p_arith,
                                GemmDims::new(Op::N, Op::N, nr * b.np, w, b.nf),
                                (1.0, 0.0),
                                MatRef::new(phi_v, phi_o, nc * b.np),
                                MatRef::new(&sub[s], 0, b.nf),
                                MatMut::new(&mut pz[s], 0, nr * b.np),
                            )?,
                        }
                        if qroute {
                            // Q_d = Phi A_d^T (np x nf), batched over the perturbations
                            self.gemm.gemm_strided(
                                self.p_arith,
                                GemmDims::new(Op::N, Op::T, b.np, b.nf, o).batch(ndg),
                                (1.0, 0.0),
                                MatRef::new(psi_v(s).0, psi_v(s).1, nc * b.np),
                                MatRef::new(&sub[s], 0, b.nf).stride(o * b.nf),
                                MatMut::new(&mut qg[s], 0, b.np).stride(b.np * b.nf),
                            )?;
                        }
                    }
                    let g1 = tl.mark()?;
                    tl.span(|t| &mut t.gemm_in, g0, g1);
                    // the pointwise kernel: P -> Z
                    let (o0, o1) = (occ[0], if nsp == 2 { occ[1] } else { 0 });
                    let (pz0_addr, pz1_addr) = (pz[0].ptr().addr(), pz[nsp - 1].ptr().addr());
                    let (qg0_addr, qg1_addr) = (qg[0].ptr().addr(), qg[nsp - 1].ptr().addr());
                    if qroute {
                        // SAFETY: matches xck_pert_gga; reads the block's collocation
                        // (4 np x nf), Phi_s (4 np x o_s), P_s (np x ndg o_s) and Q_s
                        // (np x nf x ndg), overwrites P_s with Z and Q_s with G (one thread per
                        // point and perturbation owns its values; a second spin's buffers are
                        // distinct, an absent one is not read), reads wF at the block's points.
                        unsafe {
                            gpu.launch(
                                &self.kernels.pert_gga,
                                Launch::linear(b.np * ndg, 128),
                                &[
                                    Arg::I32(b.np as i32),
                                    Arg::I64(b.p0 as i64),
                                    Arg::I64(xc.npts as i64),
                                    Arg::I32(nsp as i32),
                                    Arg::F64(scale),
                                    Arg::I32(ndg as i32),
                                    Arg::I32(b.nf as i32),
                                    Arg::Ptr(at((phi_v, phi_o))),
                                    Arg::I32(o0 as i32),
                                    Arg::Ptr(at(psi_v(0))),
                                    Arg::Ptr(pz0_addr),
                                    Arg::Ptr(qg0_addr),
                                    Arg::I32(o1 as i32),
                                    Arg::Ptr(at(psi_v(nsp - 1))),
                                    Arg::Ptr(pz1_addr),
                                    Arg::Ptr(qg1_addr),
                                    Arg::buf(&self.wf),
                                ],
                            )
                        }?;
                    } else {
                        // SAFETY: matches xck_pert; reads Phi_s (nc np x o_s) and P_s
                        // (nc np x ndg o_s) of the block, writes Z over P_s (one thread per point
                        // and perturbation owns its values; a second spin's buffer is distinct, an
                        // absent one is not read), reads wF at the block's points.
                        unsafe {
                            gpu.launch(
                                &self.kernels.pert,
                                Launch::linear(b.np * ndg, 128),
                                &[
                                    Arg::I32(b.np as i32),
                                    Arg::I64(b.p0 as i64),
                                    Arg::I64(xc.npts as i64),
                                    Arg::I32(nc as i32),
                                    Arg::I32(nsp as i32),
                                    Arg::I32(self.ns as i32),
                                    Arg::I32(gga as i32),
                                    Arg::I32(mgga as i32),
                                    Arg::F64(scale),
                                    Arg::I32(ndg as i32),
                                    Arg::I32(o0 as i32),
                                    Arg::Ptr(at(psi_v(0))),
                                    Arg::Ptr(pz0_addr),
                                    Arg::I32(o1 as i32),
                                    Arg::Ptr(at(psi_v(nsp - 1))),
                                    Arg::Ptr(pz1_addr),
                                    Arg::buf(&self.wf),
                                ],
                            )
                        }?;
                    }
                    let g2 = tl.mark()?;
                    tl.span(|t| &mut t.pointwise, g1, g2);
                    // dV C_sub = F^T Z, added to the block's rows
                    for (s, &o) in occ.iter().enumerate() {
                        if o == 0 {
                            continue;
                        }
                        let w = ndg * o;
                        match &self.ozaki {
                            Some((oz, sl)) => oz.dgemm_at(
                                gpu,
                                *sl,
                                Op::T,
                                Op::N,
                                b.nf,
                                w,
                                nr * b.np,
                                1.0,
                                (phi_v, phi_o, nc * b.np),
                                (&pz[s], 0, nr * b.np),
                                0.0,
                                (&mut sub[s], 0, b.nf),
                                &mut self.oz_work,
                            )?,
                            None => self.gemm.gemm(
                                self.p_arith,
                                GemmDims::new(Op::T, Op::N, b.nf, w, nr * b.np),
                                (1.0, 0.0),
                                MatRef::new(phi_v, phi_o, nc * b.np),
                                MatRef::new(&pz[s], 0, nr * b.np),
                                MatMut::new(&mut sub[s], 0, b.nf),
                            )?,
                        }
                        if qroute {
                            // dV_d C_sub += G_d^T Phi, batched over the perturbations
                            self.gemm.gemm_strided(
                                self.p_arith,
                                GemmDims::new(Op::T, Op::N, b.nf, o, b.np).batch(ndg),
                                (1.0, 1.0),
                                MatRef::new(&qg[s], 0, b.np).stride(b.np * b.nf),
                                MatRef::new(psi_v(s).0, psi_v(s).1, nc * b.np),
                                MatMut::new(&mut sub[s], 0, b.nf).stride(o * b.nf),
                            )?;
                        }
                        let len = b.nf * w;
                        if d0 * o * nbf + w * nbf > out[s].len() {
                            return Err(GpuError::Contract("XC kernel: output outside".into()));
                        }
                        // SAFETY: matches xck_scatter_rows; reads nf w values, adds to rows
                        // funcs (< nbf) of the w output columns from column d0 o (checked).
                        unsafe {
                            gpu.launch(
                                &self.kernels.scatter,
                                Launch::linear(len, 256),
                                &[
                                    Arg::I32(b.nf as i32),
                                    Arg::I64(w as i64),
                                    Arg::I64(nbf as i64),
                                    Arg::buf(&b.funcs),
                                    Arg::buf(&sub[s]),
                                    Arg::from(out[s].at(d0 * o * nbf)),
                                ],
                            )
                        }?;
                    }
                    let g3 = tl.mark()?;
                    tl.span(|t| &mut t.gemm_out, g2, g3);
                }
            }
            let md = gpu.mark()?;
            let res: Vec<Vec<f64>> = out
                .iter()
                .zip(&occ)
                .map(|(o, &no)| {
                    if no == 0 {
                        Ok(Vec::new())
                    } else {
                        gpu.download(o)
                    }
                })
                .collect::<Result<_, _>>()?;
            let me = gpu.mark()?;
            tl.add_to(&mut times)?;
            times.download = md.seconds_until(&me)?;
            times.total = m0.seconds_until(&me)?;
            self.last = times;
            if let Some(g) = ground.as_mut() {
                g.filled = true;
            }
            self.ground = ground;
            Ok(res)
        }

        /// The per-batch path of [`GpuXcKernel::contract`] (arguments checked there): per
        /// chunk of the XC engine's batches, each block collocated and its batches' kept
        /// functions compressed, then per batch the block path's products on its kept
        /// functions only (FP64, emulated FP64 or FP32 GEMMs).
        fn contract_batched(
            &mut self,
            gpu: &Gpu,
            a: &[&[f64]],
            nd: usize,
        ) -> Result<Vec<Vec<f64>>, GpuError> {
            let mut times = GpuXcKernelCallTimes::default();
            let m0 = gpu.mark()?;
            let nbf = self.xc.nbf;
            let nsp = self.chans.len();
            let single = self.opts.math == XcMath::Fp32;
            let (gga, mgga) = (self.xc.kind().gga, self.xc.kind().mgga);
            let npts = self.xc.npts;
            let occ: Vec<usize> = self.chans.iter().map(|c| c.o).collect();
            let omax = occ.iter().copied().max().unwrap_or(0);
            let acat: Vec<DeviceBuf<f64>> = (0..nsp)
                .map(|s| gpu.upload(if a[s].is_empty() { &[0.0] } else { a[s] }))
                .collect::<Result<_, _>>()?;
            let out: Vec<DeviceBuf<f64>> = occ
                .iter()
                .map(|&o| gpu.alloc_zeros::<f64>((nbf * nd * o).max(1)))
                .collect::<Result<_, _>>()?;
            times.upload = m0.seconds_until(&gpu.mark()?)?;
            let xyz = [&self.xc.x, &self.xc.y, &self.xc.z];
            let bat = self.xc.bat.as_mut().expect("per-batch path");
            let nc = bat.lay.ncomp;
            let qroute = gga && !mgga;
            let nr = if qroute { 1 } else { nc };
            let (np_max, nf_max) = bat
                .lay
                .host
                .iter()
                .fold((0, 0), |m, h| (m.0.max(h.np), m.1.max(h.global.len())));
            let per: usize = occ
                .iter()
                .filter(|&&o| o > 0)
                .map(|&o| (nr * np_max + nf_max) * o + if qroute { np_max * nf_max } else { 0 })
                .sum();
            let group = crate::response::kernel_group(per, nd, self.opts.workspace);
            times.group = group;
            // workspaces: per spin C_sub, Phi, P/Z, Q/G and the gathered factors / products
            // (A_sub, then V), in FP64 or FP32
            let w64 = |n: usize| gpu.alloc_zeros::<f64>(if single { 1 } else { n.max(1) });
            let w32 = |n: usize| gpu.alloc_zeros::<f32>(if single { n.max(1) } else { 1 });
            let csub = (w64(nf_max * omax)?, w32(nf_max * omax)?);
            let mut psi = Vec::with_capacity(nsp);
            let mut pz = Vec::with_capacity(nsp);
            let mut qg = Vec::with_capacity(nsp);
            let mut sub = Vec::with_capacity(nsp);
            for &o in &occ {
                psi.push((w64(nc * np_max * o)?, w32(nc * np_max * o)?));
                pz.push((w64(nr * np_max * group * o)?, w32(nr * np_max * group * o)?));
                let qn = if qroute && o > 0 {
                    np_max * nf_max * group
                } else {
                    1
                };
                qg.push((w64(qn)?, w32(qn)?));
                sub.push((w64(nf_max * group * o)?, w32(nf_max * group * o)?));
            }
            let addr = |p: &(DeviceBuf<f64>, DeviceBuf<f32>)| {
                if single {
                    p.1.ptr().addr()
                } else {
                    p.0.ptr().addr()
                }
            };
            let scale = if nsp == 1 { 2.0 } else { 1.0 };
            let (kg, ks, kp, kpg) = if single {
                (
                    &self.kernels.gather32,
                    &self.kernels.scatter32,
                    &self.kernels.pert32,
                    &self.kernels.pert_gga32,
                )
            } else {
                (
                    &self.kernels.gather,
                    &self.kernels.scatter,
                    &self.kernels.pert,
                    &self.kernels.pert_gga,
                )
            };
            let gather = |nf: usize, ncol: usize, funcs: u64, src: u64, dst: u64| {
                // SAFETY: matches xck_gather_rows(_f32); reads ncol columns of an nbf-row
                // matrix at rows funcs[f] < nbf, writes nf ncol values into dst (sized for the
                // largest batch and group by the callers).
                unsafe {
                    gpu.launch(
                        kg,
                        Launch::linear(nf * ncol, 256),
                        &[
                            Arg::I32(nf as i32),
                            Arg::I64(ncol as i64),
                            Arg::I64(nbf as i64),
                            Arg::Ptr(funcs),
                            Arg::Ptr(src),
                            Arg::Ptr(dst),
                        ],
                    )
                }
            };
            // GEMM on raw sub-matrices of the arithmetic's element type
            let raw = |g: (&Gemm, Arith),
                       ta: Op,
                       tb: Op,
                       m: usize,
                       n: usize,
                       k: usize,
                       av: (u64, usize),
                       bv: (u64, usize),
                       beta: f64,
                       cv: (u64, usize),
                       batch: usize,
                       strides: (usize, usize, usize)|
             -> Result<(), GpuError> {
                // SAFETY: the operands are the workspaces above or the engine's compressed
                // values, sized for the largest batch and group.
                unsafe { raw_gemm(g, single, ta, tb, m, n, k, av, bv, beta, cv, batch, strides) }
            };
            let mut tl = Timeline::<GpuXcKernelCallTimes>::new(gpu);
            let mut fl = 0usize;
            for ci in 0..bat.lay.chunks.len() {
                let c = bat.lay.chunks[ci].clone();
                let s0 = tl.mark()?;
                bat.lay.prepare(
                    gpu,
                    &c,
                    Collocation::Own,
                    xyz,
                    Some(&mut bat.phic),
                    if single { Some(&mut bat.phic32) } else { None },
                )?;
                tl.close(|t| &mut t.colloc, s0)?;
                let mut xo = 0usize;
                for j in c.j0..c.j1 {
                    let h = &bat.lay.host[j];
                    let (np, nf, p0) = (h.np, h.global.len(), h.p0);
                    let funcs = bat.lay.gf.at(fl).addr();
                    let phi = if single {
                        bat.phic32.at(nc * xo).addr()
                    } else {
                        bat.phic.at(nc * xo).addr()
                    };
                    fl += nf;
                    xo += nf * np;
                    if nf == 0 || np == 0 {
                        continue;
                    }
                    let s1 = tl.mark()?;
                    // ground state of the batch: Phi_s = F C_s
                    for (s, ch) in self.chans.iter().enumerate() {
                        if ch.o == 0 {
                            continue;
                        }
                        gather(nf, ch.o, funcs, ch.c.ptr().addr(), addr(&csub))?;
                        raw(
                            (&self.gemm, Arith::Fp64),
                            Op::N,
                            Op::N,
                            nc * np,
                            ch.o,
                            nf,
                            (phi, nc * np),
                            (addr(&csub), nf),
                            0.0,
                            (addr(&psi[s]), nc * np),
                            1,
                            (0, 0, 0),
                        )?;
                    }
                    let s2 = tl.mark()?;
                    tl.span(|t| &mut t.gemm_in, s1, s2);
                    tl.span(|t| &mut t.ground, s1, s2);
                    for d0 in (0..nd).step_by(group) {
                        let ndg = group.min(nd - d0);
                        let g0 = tl.mark()?;
                        for (s, &o) in occ.iter().enumerate() {
                            if o == 0 {
                                continue;
                            }
                            let w = ndg * o;
                            gather(nf, w, funcs, acat[s].at(d0 * o * nbf).addr(), addr(&sub[s]))?;
                            raw(
                                (&self.gemm, self.p_arith),
                                Op::N,
                                Op::N,
                                nr * np,
                                w,
                                nf,
                                (phi, nc * np),
                                (addr(&sub[s]), nf),
                                0.0,
                                (addr(&pz[s]), nr * np),
                                1,
                                (0, 0, 0),
                            )?;
                            if qroute {
                                // Q_d = Phi A_d^T (np x nf), batched over the perturbations
                                raw(
                                    (&self.gemm, self.p_arith),
                                    Op::N,
                                    Op::T,
                                    np,
                                    nf,
                                    o,
                                    (addr(&psi[s]), nc * np),
                                    (addr(&sub[s]), nf),
                                    0.0,
                                    (addr(&qg[s]), np),
                                    ndg,
                                    (0, o * nf, np * nf),
                                )?;
                            }
                        }
                        let g1 = tl.mark()?;
                        tl.span(|t| &mut t.gemm_in, g0, g1);
                        let (o0, o1) = (occ[0], if nsp == 2 { occ[1] } else { 0 });
                        let (pz0, pz1) = (addr(&pz[0]), addr(&pz[nsp - 1]));
                        let (qg0, qg1) = (addr(&qg[0]), addr(&qg[nsp - 1]));
                        let (ps0, ps1) = (addr(&psi[0]), addr(&psi[nsp - 1]));
                        if qroute {
                            // SAFETY: matches xck_pert_gga(_f32); reads the batch's values
                            // (4 np x nf), Phi_s, P_s and Q_s, overwrites P_s with Z and Q_s with
                            // G (one thread per point and perturbation), reads wF at the batch's
                            // points.
                            unsafe {
                                gpu.launch(
                                    kpg,
                                    Launch::linear(np * ndg, 128),
                                    &[
                                        Arg::I32(np as i32),
                                        Arg::I64(p0 as i64),
                                        Arg::I64(npts as i64),
                                        Arg::I32(nsp as i32),
                                        Arg::F64(scale),
                                        Arg::I32(ndg as i32),
                                        Arg::I32(nf as i32),
                                        Arg::Ptr(phi),
                                        Arg::I32(o0 as i32),
                                        Arg::Ptr(ps0),
                                        Arg::Ptr(pz0),
                                        Arg::Ptr(qg0),
                                        Arg::I32(o1 as i32),
                                        Arg::Ptr(ps1),
                                        Arg::Ptr(pz1),
                                        Arg::Ptr(qg1),
                                        Arg::buf(&self.wf),
                                    ],
                                )
                            }?;
                        } else {
                            // SAFETY: matches xck_pert(_f32); as above with P = F A for every
                            // component.
                            unsafe {
                                gpu.launch(
                                    kp,
                                    Launch::linear(np * ndg, 128),
                                    &[
                                        Arg::I32(np as i32),
                                        Arg::I64(p0 as i64),
                                        Arg::I64(npts as i64),
                                        Arg::I32(nc as i32),
                                        Arg::I32(nsp as i32),
                                        Arg::I32(self.ns as i32),
                                        Arg::I32(gga as i32),
                                        Arg::I32(mgga as i32),
                                        Arg::F64(scale),
                                        Arg::I32(ndg as i32),
                                        Arg::I32(o0 as i32),
                                        Arg::Ptr(ps0),
                                        Arg::Ptr(pz0),
                                        Arg::I32(o1 as i32),
                                        Arg::Ptr(ps1),
                                        Arg::Ptr(pz1),
                                        Arg::buf(&self.wf),
                                    ],
                                )
                            }?;
                        }
                        let g2 = tl.mark()?;
                        tl.span(|t| &mut t.pointwise, g1, g2);
                        for (s, &o) in occ.iter().enumerate() {
                            if o == 0 {
                                continue;
                            }
                            let w = ndg * o;
                            raw(
                                (&self.gemm, self.p_arith),
                                Op::T,
                                Op::N,
                                nf,
                                w,
                                nr * np,
                                (phi, nc * np),
                                (addr(&pz[s]), nr * np),
                                0.0,
                                (addr(&sub[s]), nf),
                                1,
                                (0, 0, 0),
                            )?;
                            if qroute {
                                raw(
                                    (&self.gemm, self.p_arith),
                                    Op::T,
                                    Op::N,
                                    nf,
                                    o,
                                    np,
                                    (addr(&qg[s]), np),
                                    (addr(&psi[s]), nc * np),
                                    1.0,
                                    (addr(&sub[s]), nf),
                                    ndg,
                                    (np * nf, 0, o * nf),
                                )?;
                            }
                            if d0 * o * nbf + w * nbf > out[s].len() {
                                return Err(GpuError::Contract("XC kernel: output outside".into()));
                            }
                            // SAFETY: matches xck_scatter_rows(_f32); reads nf w values, adds
                            // to rows funcs (< nbf) of the w output columns from column d0 o
                            // (checked).
                            unsafe {
                                gpu.launch(
                                    ks,
                                    Launch::linear(nf * w, 256),
                                    &[
                                        Arg::I32(nf as i32),
                                        Arg::I64(w as i64),
                                        Arg::I64(nbf as i64),
                                        Arg::Ptr(funcs),
                                        Arg::Ptr(addr(&sub[s])),
                                        Arg::from(out[s].at(d0 * o * nbf)),
                                    ],
                                )
                            }?;
                        }
                        tl.close(|t| &mut t.gemm_out, g2)?;
                    }
                }
            }
            let md = gpu.mark()?;
            let res: Vec<Vec<f64>> = out
                .iter()
                .zip(&occ)
                .map(|(o, &no)| {
                    if no == 0 {
                        Ok(Vec::new())
                    } else {
                        gpu.download(o)
                    }
                })
                .collect::<Result<_, _>>()?;
            let me = gpu.mark()?;
            tl.add_to(&mut times)?;
            times.download = md.seconds_until(&me)?;
            times.total = m0.seconds_until(&me)?;
            self.last = times;
            Ok(res)
        }
    }

    /// A GEMM (batch 1) or strided-batched GEMM on raw device addresses of FP64 or (`single`)
    /// FP32 matrices: C = op(A) op(B) + beta C, operands (address, leading dimension), strides
    /// (A, B, C) between the products of a batch.
    ///
    /// # Safety
    /// Every product's operands must lie in live device buffers of the element type.
    #[allow(clippy::too_many_arguments)]
    unsafe fn raw_gemm(
        (gemm, arith): (&Gemm, Arith),
        single: bool,
        ta: Op,
        tb: Op,
        m: usize,
        n: usize,
        k: usize,
        a: (u64, usize),
        b: (u64, usize),
        beta: f64,
        c: (u64, usize),
        batch: usize,
        strides: (usize, usize, usize),
    ) -> Result<(), GpuError> {
        let d = GemmDims::new(ta, tb, m, n, k).batch(batch);
        // SAFETY: the caller's contract (addresses of elements of the arithmetic's type).
        unsafe {
            if single {
                let op = |(p, ld): (u64, usize), stride| MatPtr {
                    ptr: DevPtr::<f32>::from_addr(p),
                    ld,
                    stride,
                };
                gemm.gemm_raw(
                    Arith::Fp32,
                    d,
                    (1.0f32, beta as f32),
                    op(a, strides.0),
                    op(b, strides.1),
                    op(c, strides.2),
                )
            } else {
                let op = |(p, ld): (u64, usize), stride| MatPtr {
                    ptr: DevPtr::<f64>::from_addr(p),
                    ld,
                    stride,
                };
                gemm.gemm_raw(
                    arith,
                    d,
                    (1.0, beta),
                    op(a, strides.0),
                    op(b, strides.1),
                    op(c, strides.2),
                )
            }
        }
    }

    /// sub = the rows `funcs` (nf of them) of the n x ncol matrix `src`.
    #[allow(clippy::too_many_arguments)]
    fn gather_rows(
        gpu: &Gpu,
        k: &Kernel,
        nf: usize,
        ncol: usize,
        nbf: usize,
        funcs: &DeviceBuf<i32>,
        src: &DeviceBuf<f64>,
        sub: &mut DeviceBuf<f64>,
    ) -> Result<(), GpuError> {
        gather_rows_at(gpu, k, nf, ncol, nbf, funcs, src, 0, sub)
    }

    /// [`gather_rows`] of the matrix at element `off` of `src`.
    #[allow(clippy::too_many_arguments)]
    fn gather_rows_at(
        gpu: &Gpu,
        k: &Kernel,
        nf: usize,
        ncol: usize,
        nbf: usize,
        funcs: &DeviceBuf<i32>,
        src: &DeviceBuf<f64>,
        off: usize,
        sub: &mut DeviceBuf<f64>,
    ) -> Result<(), GpuError> {
        if off + ncol * nbf > src.len() || nf * ncol > sub.len() || nf > funcs.len() {
            return Err(GpuError::Contract(
                "XC kernel: gather outside its buffers".into(),
            ));
        }
        // SAFETY: matches xck_gather_rows; reads ncol columns of src from `off` at rows
        // funcs[f] < nbf, writes nf ncol values (all checked above).
        unsafe {
            gpu.launch(
                k,
                Launch::linear(nf * ncol, 256),
                &[
                    Arg::I32(nf as i32),
                    Arg::I64(ncol as i64),
                    Arg::I64(nbf as i64),
                    Arg::buf(funcs),
                    Arg::from(src.at(off)),
                    Arg::buf_mut(sub),
                ],
            )
        }
    }
}
