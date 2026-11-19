//! VV10 nonlocal correlation on the device (feature `gpu`): `GpuVv10` (energy and SCF
//! potential) and `GpuVv10Grad` (the analytic nuclear gradient with the grid response), the
//! device counterparts of linus's CPU VV10 (`xc::vv10` in the `linus` crate, whose module docs
//! give the formulas), on the same NLC grid with the same density threshold.
//!
//! The semilocal machinery is the device XC's, on the NLC grid as a GGA of the total density:
//! `GpuXc` forms rho and grad rho with its per-batch products and assembles
//! V_mn = sum_i w_i [v_rho phi_m phi_n + 2 v_sigma grad rho . grad(phi_m phi_n)];
//! `GpuXcGrad` contracts the gradient's orbital and point-motion terms and the weight
//! derivatives. VV10 replaces their pointwise step (libxc on the host) with, on the device:
//! 1. **Active points**: rho_i >= `rho_thresh` (1e-8, the CPU's), compacted in ascending grid
//!    order (a count per block of points, a scan, a scatter: integer work, deterministic).
//! 2. **Per point** (FP64): omega, kappa, rho w and the potential's factors.
//! 3. **Pair sums** F_i, U_i, W_i (for the gradient also sum_j (omega_i Q_ij + omega_j Q_ji)
//!    rho_j w_j (r_i - r_j)): one thread per point i over all active points j in ascending
//!    order, the j points in tiles of [`TILE`] through shared memory, the j range split into
//!    segments whose sums are added in order (as many segments as fill a large card several
//!    times over, chosen from the number of points alone). Each point's sums run in an order
//!    fixed by the input, whatever the device and its number of SMs, so the results are bitwise
//!    run to run and card to card (for one compiler); no atomics.
//! 4. **Per point**: zk, v_rho and v_sigma at the point's place in the grid (zero at inactive
//!    points), and for the gradient dE/dr_i, which joins the point-motion term.
//!
//! The pair arithmetic ([`Vv10Math`]): FP64; double-float (each value the unevaluated sum of
//! two FP32 numbers, ~46-bit significands from the FP32 units), each tile's sums in
//! double-float and the tiles added in FP64; or FP32 pair terms, each tile summed in FP32 and
//! the tiles in FP64. A pair costs one division and ~25 operations in FP64, ~220 FP32
//! operations in double-float (~330 with the gradient's sums), ~30 in FP32; on cards with
//! little FP64 throughput (1/64 of FP32 on the RTX PRO 6000) double-float runs several times
//! faster than FP64. docs/internal/gpu-spike.md, section VV10GPU, has the precision study.

/// Points j per tile of the pair kernels (threads per block: one point i each). The tile's
/// sums are the double-float and FP32 kernels' unit of FP64 accumulation, so it is fixed
/// (never derived from the device) to keep the results card-independent.
pub const TILE: usize = 128;

/// The arithmetic of the VV10 pair sums (module docs).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Vv10Math {
    /// FP64 throughout.
    Fp64,
    /// Double-float pair arithmetic, each tile of [`TILE`] pairs summed in double-float, the
    /// tiles in FP64.
    DoubleFloat,
    /// FP32 pair terms, each tile summed in FP32, the tiles in FP64.
    Fp32,
}

impl Vv10Math {
    /// The name in reports: "fp64", "df64", "fp32".
    pub fn name(&self) -> &'static str {
        match self {
            Vv10Math::Fp64 => "fp64",
            Vv10Math::DoubleFloat => "df64",
            Vv10Math::Fp32 => "fp32",
        }
    }

    /// From [`Vv10Math::name`].
    pub fn from_name(s: &str) -> Option<Vv10Math> {
        match s {
            "fp64" => Some(Vv10Math::Fp64),
            "df64" => Some(Vv10Math::DoubleFloat),
            "fp32" => Some(Vv10Math::Fp32),
            _ => None,
        }
    }
}

/// Settings of a `GpuVv10` setup.
#[derive(Clone, Debug)]
pub struct GpuVv10Options {
    /// The NLC grid's screening of functions (the CPU's NLC engine: 1e-11).
    pub func_thresh: f64,
    /// Points with a total density below this are left out (the CPU's 1e-8).
    pub rho_thresh: f64,
    pub math: Vv10Math,
    /// Points per block of collocation ([`crate::xc::GpuXcOptions::block_points`]).
    pub block_points: usize,
    /// Device bytes of the per-batch workspaces ([`crate::xc::GpuXcOptions::workspace`]).
    pub workspace: usize,
}

impl Default for GpuVv10Options {
    fn default() -> Self {
        GpuVv10Options {
            func_thresh: 1e-11,
            rho_thresh: 1e-8,
            math: Vv10Math::Fp64,
            block_points: 2048,
            workspace: 1 << 30,
        }
    }
}

/// Settings of a `GpuVv10Grad` setup.
#[derive(Clone, Debug)]
pub struct GpuVv10GradOptions {
    pub rho_thresh: f64,
    pub math: Vv10Math,
    /// The XC gradient engine's (screening 1e-15 at second order, as the CPU's VV10 gradient).
    pub grad: crate::grad::xc::GpuXcGradOptions,
}

impl Default for GpuVv10GradOptions {
    fn default() -> Self {
        GpuVv10GradOptions {
            rho_thresh: 1e-8,
            math: Vv10Math::Fp64,
            grad: crate::grad::xc::GpuXcGradOptions::default(),
        }
    }
}

/// beta and the prefactor of kappa = kvv rho^(1/6) (the CPU's `constants`).
pub fn constants(p: &linus_xc::Vv10) -> (f64, f64) {
    let pi = std::f64::consts::PI;
    let kvv = p.b * 1.5 * pi * (9.0 * pi).powf(-1.0 / 6.0);
    let beta = (3.0 / (p.b * p.b)).powf(0.75) / 32.0;
    (beta, kvv)
}

#[cfg(feature = "gpu")]
pub use device::*;

#[cfg(feature = "gpu")]
mod device {
    use super::*;
    use crate::cpu::{BasisSet, GridSpec, MolecularGrid};
    use crate::error::GpuError;
    use crate::grad::xc::GpuXcGrad;
    use crate::kernels::registry::KernelModule;
    use crate::linalg::{Blas1, Operand, Scale};
    use crate::xc::XcScreen;
    use crate::xc::{GpuXc, GpuXcOptions, PointInputs, PointOutputs, XcChannel, XcKind, XcMath};
    use crate::{Arg, DeviceBuf, Gpu, Kernel, Launch};
    use std::borrow::Cow;

    const KERNELS: &str = include_str!("../../device/xc/vv10.cu");

    /// Points per block of the compaction kernels (`VV_CB`).
    const CB: usize = 1024;

    /// The mean of the atoms' positions: the origin of the pair kernels' coordinates (FP32 and
    /// double-float keep the most digits of differences of numbers near zero).
    fn centre(grid: &MolecularGrid) -> [f64; 3] {
        let n = grid.atom_xyz.len().max(1) as f64;
        let mut c = [0.0; 3];
        for a in &grid.atom_xyz {
            for (ck, ak) in c.iter_mut().zip(a) {
                *ck += ak;
            }
        }
        c.map(|v| v / n)
    }

    /// Segments of each point's j range for `nt` tiles of points: enough blocks of threads
    /// (about [`SEGMENT_BLOCKS`]) to fill a large card several times over, at least one tile
    /// per segment. A function of the number of points only, never of the device, so the order
    /// of every sum is fixed by the input.
    pub(crate) fn segments(nt: usize) -> usize {
        SEGMENT_BLOCKS.div_ceil(nt.max(1)).clamp(1, nt.max(1))
    }

    /// The pair kernels' target number of blocks (188 SMs x 9 resident blocks of the
    /// double-float kernel = 1692 on the RTX PRO 6000: about five waves).
    const SEGMENT_BLOCKS: usize = 8192;

    /// The pair evaluation's module: the tile width, then its kernels.
    pub(crate) const MODULE: KernelModule = KernelModule::new(
        "vv10",
        || Cow::Owned(format!("#define VV_TILE {TILE}\n{KERNELS}")),
        &[
            "vv10_count",
            "vv10_scan",
            "vv10_compact",
            "vv10_prep",
            "vv10_split",
            "vv10_post",
            "vv10_pairs_f64",
            "vv10_pairs_f64_forces",
            "vv10_pairs_df64",
            "vv10_pairs_df64_forces",
            "vv10_pairs_f32",
            "vv10_pairs_f32_forces",
        ],
    );

    /// The kernels of the pair evaluation.
    struct Kernels {
        count: Kernel,
        scan: Kernel,
        compact: Kernel,
        prep: Kernel,
        split: Kernel,
        post: Kernel,
        blas1: Blas1,
        /// [arithmetic][forces]: FP64, double-float, FP32
        pairs: [[Kernel; 2]; 3],
    }

    impl Kernels {
        fn new(gpu: &Gpu) -> Result<Kernels, GpuError> {
            let src = MODULE.source();
            let k = |e: &str| gpu.kernel(MODULE.name, &src, e);
            Ok(Kernels {
                count: k("vv10_count")?,
                scan: k("vv10_scan")?,
                compact: k("vv10_compact")?,
                prep: k("vv10_prep")?,
                split: k("vv10_split")?,
                post: k("vv10_post")?,
                blas1: Blas1::new(gpu)?,
                pairs: [
                    [k("vv10_pairs_f64")?, k("vv10_pairs_f64_forces")?],
                    [k("vv10_pairs_df64")?, k("vv10_pairs_df64_forces")?],
                    [k("vv10_pairs_f32")?, k("vv10_pairs_f32_forces")?],
                ],
            })
        }
    }

    /// Device times of the last pair evaluation (seconds, from events).
    #[derive(Clone, Debug, Default)]
    pub struct Vv10PairTimes {
        /// Compaction, per-point quantities, pair sums and per-point outputs.
        pub total: f64,
        /// The pair kernel alone.
        pub pairs: f64,
        /// Active points and their pairs (n^2).
        pub active: usize,
        pub pair_count: f64,
    }

    /// The pair evaluation: its kernels and the functional's VV10 parameters.
    struct Pairs {
        k: Kernels,
        cvv: f64,
        kvv: f64,
        beta: f64,
        scale: f64,
        rho_thresh: f64,
        centre: [f64; 3],
    }

    impl Pairs {
        fn new(
            gpu: &Gpu,
            p: linus_xc::Vv10,
            rho_thresh: f64,
            grid: &MolecularGrid,
        ) -> Result<Pairs, GpuError> {
            let (beta, kvv) = constants(&p);
            Ok(Pairs {
                k: Kernels::new(gpu)?,
                cvv: p.c,
                kvv,
                beta,
                scale: p.scale,
                rho_thresh,
                centre: centre(grid),
            })
        }

        /// zk, vrho and vsigma at every grid point of `inp` (closed-shell layout: the total
        /// density), with `forces` also dE/dr of every point; zk = sc (beta + zf f).
        fn evaluate(
            &self,
            gpu: &Gpu,
            inp: &PointInputs<'_>,
            math: Vv10Math,
            zf: f64,
            forces: bool,
            times: &mut Vv10PairTimes,
        ) -> Result<PointOutputs, GpuError> {
            if !inp.restricted {
                return Err(GpuError::Contract(
                    "VV10: the pair sums take the total density (closed-shell layout)".into(),
                ));
            }
            let npts = inp.npts;
            let m0 = gpu.mark()?;
            let nb = npts.div_ceil(CB).max(1);
            let mut counts = gpu.alloc_zeros::<i32>(nb)?;
            let mut off = gpu.alloc_zeros::<i32>(nb + 1)?;
            let cfg = Launch {
                grid: (nb as u32, 1, 1),
                block: (CB as u32, 1, 1),
                shared_bytes: 0,
            };
            // SAFETY: matches vv10_count; reads npts densities, writes nb counts.
            unsafe {
                gpu.launch(
                    &self.k.count,
                    cfg,
                    &[
                        Arg::I64(npts as i64),
                        Arg::F64(self.rho_thresh),
                        Arg::buf(inp.rho),
                        Arg::buf_mut(&mut counts),
                    ],
                )
            }?;
            // SAFETY: matches vv10_scan; nb counts in, nb + 1 offsets out.
            unsafe {
                gpu.launch(
                    &self.k.scan,
                    Launch::linear(1, 32),
                    &[
                        Arg::I32(nb as i32),
                        Arg::buf(&counts),
                        Arg::buf_mut(&mut off),
                    ],
                )
            }?;
            let n = gpu.download_range(&off, nb, 1)?[0] as usize;
            let mut zk = gpu.alloc_zeros::<f64>(npts.max(1))?;
            let mut vrho = gpu.alloc_zeros::<f64>(npts.max(1))?;
            let mut vsigma = gpu.alloc_zeros::<f64>(npts.max(1))?;
            let mut force = if forces {
                Some(gpu.alloc_zeros::<f64>((3 * npts).max(1))?)
            } else {
                None
            };
            let npad = n.div_ceil(TILE) * TILE;
            times.active = n;
            times.pair_count = (n as f64) * (n as f64);
            if n > 0 {
                let mut idx = gpu.alloc_zeros::<i32>(n)?;
                // SAFETY: matches vv10_compact; reads npts densities and nb offsets, writes the
                // n active points' indices (off[nb] = n).
                unsafe {
                    gpu.launch(
                        &self.k.compact,
                        cfg,
                        &[
                            Arg::I64(npts as i64),
                            Arg::F64(self.rho_thresh),
                            Arg::buf(inp.rho),
                            Arg::buf(&off),
                            Arg::buf_mut(&mut idx),
                        ],
                    )
                }?;
                let mut q = gpu.alloc_zeros::<f64>(9 * npad)?;
                let c = self.centre;
                // SAFETY: matches vv10_prep; reads the n active points of the grid arrays
                // (grad: 3 npts), writes 9 npad values.
                unsafe {
                    gpu.launch(
                        &self.k.prep,
                        Launch::linear(npad, 256),
                        &[
                            Arg::I32(n as i32),
                            Arg::I32(npad as i32),
                            Arg::F64(self.cvv),
                            Arg::F64(self.kvv),
                            Arg::F64(4.0 * std::f64::consts::PI / 3.0),
                            Arg::F64(c[0]),
                            Arg::F64(c[1]),
                            Arg::F64(c[2]),
                            Arg::buf(&idx),
                            Arg::buf(inp.x),
                            Arg::buf(inp.y),
                            Arg::buf(inp.z),
                            Arg::buf(inp.w),
                            Arg::buf(inp.rho),
                            Arg::buf(inp.grad),
                            Arg::buf_mut(&mut q),
                        ],
                    )
                }?;
                let nsum = if forces { 6 } else { 3 };
                let nseg = segments(npad / TILE);
                let mut s = gpu.alloc_zeros::<f64>(nseg * nsum * npad)?;
                let pcfg = Launch {
                    grid: ((npad / TILE) as u32, nseg as u32, 1),
                    block: (TILE as u32, 1, 1),
                    shared_bytes: 0,
                };
                let ai = match math {
                    Vv10Math::Fp64 => 0,
                    Vv10Math::DoubleFloat => 1,
                    Vv10Math::Fp32 => 2,
                };
                let kp = &self.k.pairs[ai][usize::from(forces)];
                let mp0;
                if math == Vv10Math::Fp64 {
                    mp0 = gpu.mark()?;
                    // SAFETY: matches vv10_pairs_f64(_forces); npad / TILE blocks of TILE
                    // threads read 6 npad values of q, write nsum npad sums.
                    unsafe {
                        gpu.launch(
                            kp,
                            pcfg,
                            &[Arg::I32(npad as i32), Arg::buf(&q), Arg::buf_mut(&mut s)],
                        )
                    }?;
                } else {
                    let df = math == Vv10Math::DoubleFloat;
                    let mut qf = gpu.alloc_zeros::<f32>(if df { 12 } else { 6 } * npad)?;
                    // SAFETY: matches vv10_split; reads 6 npad values, writes 12 (6) npad.
                    unsafe {
                        gpu.launch(
                            &self.k.split,
                            Launch::linear(npad, 256),
                            &[
                                Arg::I32(npad as i32),
                                Arg::I32(df as i32),
                                Arg::buf(&q),
                                Arg::buf_mut(&mut qf),
                            ],
                        )
                    }?;
                    mp0 = gpu.mark()?;
                    // SAFETY: matches vv10_pairs_df64 / vv10_pairs_f32 (and _forces): reads
                    // the split inputs, writes nsum npad sums.
                    unsafe {
                        gpu.launch(
                            kp,
                            pcfg,
                            &[Arg::I32(npad as i32), Arg::buf(&qf), Arg::buf_mut(&mut s)],
                        )
                    }?;
                }
                let mp1 = gpu.mark()?;
                let fptr = match force.as_mut() {
                    Some(f) => Arg::buf_mut(f),
                    None => Arg::Ptr(0),
                };
                // SAFETY: matches vv10_post; reads the n active points' sums and quantities,
                // writes their grid points of zk, vrho, vsigma (npts) and force (3 npts).
                unsafe {
                    gpu.launch(
                        &self.k.post,
                        Launch::linear(n, 256),
                        &[
                            Arg::I32(n as i32),
                            Arg::I32(npad as i32),
                            Arg::I32(nseg as i32),
                            Arg::F64(self.beta),
                            Arg::F64(self.scale),
                            Arg::F64(zf),
                            Arg::I32(forces as i32),
                            Arg::buf(&idx),
                            Arg::buf(&q),
                            Arg::buf(&s),
                            Arg::buf_mut(&mut zk),
                            Arg::buf_mut(&mut vrho),
                            Arg::buf_mut(&mut vsigma),
                            fptr,
                        ],
                    )
                }?;
                times.pairs = mp0.seconds_until(&mp1)?;
            }
            times.total = m0.seconds_until(&gpu.mark()?)?;
            Ok(PointOutputs {
                zk,
                vrho,
                vsigma,
                vtau: gpu.alloc_zeros::<f64>(1)?,
                point_force: force,
            })
        }
    }

    /// The NLC grid's semilocal ingredients: a GGA of the total density.
    const GGA: XcKind = XcKind {
        none: false,
        gga: true,
        mgga: false,
    };

    /// The functional's VV10 parameters, or an error for functionals without a nonlocal term.
    fn params(name: &str) -> Result<linus_xc::Vv10, GpuError> {
        let f = linus_xc::Functional::new(name, false)
            .map_err(|e| GpuError::Input(format!("functional {name}: {e}")))?;
        f.vv10()
            .ok_or_else(|| GpuError::Contract(format!("{name} has no VV10 term")))
    }

    /// Times of the last [`GpuVv10`] evaluation (seconds).
    #[derive(Clone, Debug, Default)]
    pub struct GpuVv10Times {
        pub total: f64,
        /// Collocation and densities on the NLC grid.
        pub density: f64,
        /// The pair evaluation (compaction, per-point work, pair sums).
        pub pairs: Vv10PairTimes,
        /// The potential matrix (weights, products, accumulation).
        pub potential: f64,
    }

    /// Setup statistics.
    #[derive(Clone, Debug)]
    pub struct GpuVv10Stats {
        /// NLC grid points.
        pub npts: usize,
        pub setup_seconds: f64,
    }

    /// What the evaluations since the setup did (seconds, summed).
    #[derive(Clone, Debug, Default)]
    pub struct GpuVv10Totals {
        pub evaluations: usize,
        /// E_nl of the last evaluation.
        pub last_energy: Option<f64>,
        /// Active points of the last evaluation.
        pub active: usize,
        pub density: f64,
        pub pairs: f64,
        pub potential: f64,
    }

    /// VV10's energy and SCF potential on the device (module docs).
    pub struct GpuVv10 {
        xc: GpuXc,
        pairs: Pairs,
        math: Vv10Math,
        nbf: usize,
        params: linus_xc::Vv10,
        stats: GpuVv10Stats,
        totals: GpuVv10Totals,
        pub last: GpuVv10Times,
    }

    impl GpuVv10 {
        /// The VV10 term of the functional `name` (a linus-xc name) on the NLC grid `grid`.
        pub fn new(
            gpu: &Gpu,
            basis: &BasisSet,
            grid: &MolecularGrid,
            name: &str,
            opts: &GpuVv10Options,
        ) -> Result<GpuVv10, GpuError> {
            let t0 = std::time::Instant::now();
            let params = params(name)?;
            let xc = GpuXc::with_kind(
                gpu,
                basis,
                grid,
                name,
                Some(GGA),
                &GpuXcOptions {
                    func_thresh: opts.func_thresh,
                    block_points: opts.block_points,
                    screen: XcScreen::Batch,
                    math: XcMath::Fp64,
                    workspace: opts.workspace,
                    ..GpuXcOptions::default()
                },
            )?;
            let pairs = Pairs::new(gpu, params, opts.rho_thresh, grid)?;
            Ok(GpuVv10 {
                xc,
                pairs,
                math: opts.math,
                nbf: basis.nbf,
                params,
                stats: GpuVv10Stats {
                    npts: grid.x.len(),
                    setup_seconds: t0.elapsed().as_secs_f64(),
                },
                totals: GpuVv10Totals::default(),
                last: GpuVv10Times::default(),
            })
        }

        pub fn stats(&self) -> &GpuVv10Stats {
            &self.stats
        }

        pub fn totals(&self) -> &GpuVv10Totals {
            &self.totals
        }

        pub fn params(&self) -> linus_xc::Vv10 {
            self.params
        }

        pub fn math(&self) -> Vv10Math {
            self.math
        }

        /// E_nl (returned; scaled, beta N included, as the CPU's) and V_nl (into `v`: one
        /// nbf x nbf column-major matrix, overwritten; the same for both spins) of the total
        /// density of the occupied orbitals: `cmo` holds the channels side by side (one for a
        /// closed shell with `restricted`, the total density being twice its own; else alpha
        /// and beta), `occ` their occupations (None: all 1).
        pub fn compute_device(
            &mut self,
            gpu: &Gpu,
            cmo: &DeviceBuf<f64>,
            ncols: &[usize],
            occ: Option<&DeviceBuf<f64>>,
            restricted: bool,
            v: &mut DeviceBuf<f64>,
        ) -> Result<f64, GpuError> {
            let ntot: usize = ncols.iter().sum();
            let m0 = gpu.mark()?;
            // an open shell's total density as one closed-shell channel: the columns of both
            // spins at half their occupations (the engine doubles a closed shell's density;
            // halving and doubling are exact)
            let half;
            let one = [ntot];
            let (nc, oc): (&[usize], Option<&DeviceBuf<f64>>) = if restricted {
                (ncols, occ)
            } else {
                half = match occ {
                    Some(o) => {
                        if o.len() < ntot {
                            return Err(GpuError::Contract("VV10: occupations too short".into()));
                        }
                        let mut h = gpu.alloc_zeros::<f64>(ntot.max(1))?;
                        self.pairs.k.blas1.scale(
                            gpu,
                            ntot,
                            Scale::Scalar(0.5),
                            Operand::Buf(o),
                            &mut h,
                        )?;
                        h
                    }
                    None => gpu.upload(&vec![0.5; ntot.max(1)])?,
                };
                (&one, Some(&half))
            };
            let (pairs, math) = (&self.pairs, self.math);
            let mut pt = Vv10PairTimes::default();
            let mut f =
                |g: &Gpu, inp: &PointInputs<'_>| pairs.evaluate(g, inp, math, 0.5, false, &mut pt);
            let e = self
                .xc
                .compute_device_with(gpu, cmo, nc, oc, true, v, Some(&mut f))?;
            let l = &self.xc.last;
            self.last = GpuVv10Times {
                total: m0.seconds_until(&gpu.mark()?)?,
                density: l.colloc + l.density,
                pairs: pt,
                potential: l.potential + l.energy,
            };
            let t = &mut self.totals;
            t.evaluations += 1;
            t.last_energy = Some(e);
            t.active = self.last.pairs.active;
            t.density += self.last.density;
            t.pairs += self.last.pairs.total;
            t.potential += self.last.potential;
            Ok(e)
        }

        /// As [`GpuVv10::compute_device`] with the orbitals on the host: (E_nl, V_nl as one
        /// nbf x nbf column-major matrix).
        pub fn compute(
            &mut self,
            gpu: &Gpu,
            ch: &[XcChannel],
            restricted: bool,
        ) -> Result<(f64, Vec<f64>), GpuError> {
            let nbf = self.nbf;
            if ch.len() != if restricted { 1 } else { 2 } {
                return Err(GpuError::Contract(format!(
                    "VV10: {} channels of orbitals (restricted: {restricted})",
                    ch.len()
                )));
            }
            let ntot: usize = ch.iter().map(|c| c.ncol).sum();
            let mut c = Vec::with_capacity(nbf * ntot);
            let mut occ = Vec::with_capacity(ntot);
            for x in ch {
                if x.c.len() != nbf * x.ncol || x.occ.len() != x.ncol {
                    return Err(GpuError::Contract(
                        "VV10: orbitals or occupations of the wrong size".into(),
                    ));
                }
                c.extend_from_slice(x.c);
                occ.extend_from_slice(x.occ);
            }
            let cd = gpu.upload(if c.is_empty() { &[0.0] } else { &c })?;
            let od = gpu.upload(if occ.is_empty() { &[0.0] } else { &occ })?;
            let ncols: Vec<usize> = ch.iter().map(|x| x.ncol).collect();
            let mut v = gpu.alloc_zeros::<f64>(nbf * nbf)?;
            let e = self.compute_device(gpu, &cd, &ncols, Some(&od), restricted, &mut v)?;
            Ok((e, gpu.download(&v)?))
        }
    }

    /// Times of the last [`GpuVv10Grad`] evaluation (seconds).
    #[derive(Clone, Debug, Default)]
    pub struct GpuVv10GradTimes {
        pub total: f64,
        pub pairs: Vv10PairTimes,
        /// The XC gradient engine's phases on the NLC grid.
        pub engine: crate::grad::xc::GpuXcGradTimes,
    }

    /// VV10's nuclear gradient on the device (module docs).
    pub struct GpuVv10Grad {
        g: GpuXcGrad,
        pairs: Pairs,
        math: Vv10Math,
        nbf: usize,
        pub last: GpuVv10GradTimes,
    }

    impl GpuVv10Grad {
        /// The VV10 term of the functional `name` on the NLC grid `grid` (built with `spec`).
        pub fn new(
            gpu: &Gpu,
            basis: &BasisSet,
            grid: &MolecularGrid,
            spec: &GridSpec,
            name: &str,
            opts: &GpuVv10GradOptions,
        ) -> Result<GpuVv10Grad, GpuError> {
            let p = params(name)?;
            let g = GpuXcGrad::with_kind(gpu, basis, grid, spec, name, Some(GGA), &opts.grad)?;
            Ok(GpuVv10Grad {
                g,
                pairs: Pairs::new(gpu, p, opts.rho_thresh, grid)?,
                math: opts.math,
                nbf: basis.nbf,
                last: GpuVv10GradTimes::default(),
            })
        }

        pub fn math(&self) -> Vv10Math {
            self.math
        }

        /// The gradient of E_nl (natom x 3) at the total density `dens` (nbf x nbf on the
        /// device). With `grid_response` the NLC grid moves with the atoms (point motion,
        /// the kernel's explicit position dependence, the weight derivatives): the derivative
        /// of the quadrature energy, as the CPU's.
        pub fn compute_device(
            &mut self,
            gpu: &Gpu,
            dens: &DeviceBuf<f64>,
            grid_response: bool,
        ) -> Result<Vec<[f64; 3]>, GpuError> {
            let t0 = std::time::Instant::now();
            let (pairs, math) = (&self.pairs, self.math);
            let mut pt = Vv10PairTimes::default();
            let mut f = |g: &Gpu, inp: &PointInputs<'_>| {
                pairs.evaluate(g, inp, math, 1.0, grid_response, &mut pt)
            };
            let r = self
                .g
                .compute_device_with(gpu, dens, true, grid_response, Some(&mut f))?;
            self.last = GpuVv10GradTimes {
                total: t0.elapsed().as_secs_f64(),
                pairs: pt,
                engine: self.g.last.clone(),
            };
            Ok(r)
        }

        /// As [`GpuVv10Grad::compute_device`] with the total density on the host (nbf x nbf,
        /// column-major).
        pub fn compute(
            &mut self,
            gpu: &Gpu,
            dens: &[f64],
            grid_response: bool,
        ) -> Result<Vec<[f64; 3]>, GpuError> {
            if dens.len() != self.nbf * self.nbf {
                return Err(GpuError::Contract(
                    "VV10 gradient: density of the wrong size".into(),
                ));
            }
            let d = gpu.upload(if dens.is_empty() { &[0.0] } else { dens })?;
            self.compute_device(gpu, &d, grid_response)
        }
    }

    // ---------------------------------------------------------------- second derivatives

    /// Perturbations per block of the response pair kernels (`VR_PC`).
    pub const RESP_PERTURBATIONS: usize = 4;

    const RESP_KERNELS: &str = include_str!("../../device/xc/vv10_resp.cu");

    /// The response's module: the tile width, the pair evaluation's kernels, the response's.
    pub(crate) const RESP_MODULE: KernelModule = KernelModule::new(
        "vv10_resp",
        || Cow::Owned(format!("#define VV_TILE {TILE}\n{KERNELS}\n{RESP_KERNELS}")),
        &[
            "vv10_resp_f64",
            "vv10_resp_df64",
            "vv10_resp_f32",
            "vv10_split",
        ],
    );

    /// The per-perturbation pair sums of VV10's second derivatives on the device (the CPKS
    /// kernel and the explicit Hessian terms of linus's `xc::vv10_resp`): the active points'
    /// positions, omega and kappa stay on the device; each call uploads the perturbations'
    /// coefficients and returns their pair sums: FP64 or double-float ([`Vv10Math`]), and per
    /// call optionally FP32 (each tile summed in FP32, the tiles in FP64) for the CPKS kernel's
    /// products when the device's semilocal CPKS kernel runs FP32.
    pub struct GpuVv10Resp {
        k64: Kernel,
        kdf: Kernel,
        k32: Kernel,
        math: Vv10Math,
        n: usize,
        npad: usize,
        q: DeviceBuf<f64>,
        qf: Option<DeviceBuf<f32>>,
        /// x, y, z, omega, kappa in FP32 (vv10_split with df = 0), for FP32 calls
        qs: DeviceBuf<f32>,
        /// Device seconds of the last call's pair kernel, and summed over all calls.
        pub last_seconds: f64,
        pub total_seconds: f64,
    }

    impl GpuVv10Resp {
        /// The engine for the `n` active points with coordinates `x`, `y`, `z` (about a centre
        /// near them: double-float and FP32 keep the most digits of differences of small
        /// numbers), `omega` and `kappa`; `math` FP64 or double-float. FP32 is not an engine
        /// arithmetic but a per-call choice ([`Self::sweep`]'s `fp32`): `Vv10Math::Fp32` is an
        /// input error (it ran double-float without a word).
        pub fn new(
            gpu: &Gpu,
            pts: [&[f64]; 5],
            n: usize,
            math: Vv10Math,
        ) -> Result<GpuVv10Resp, GpuError> {
            if pts.iter().any(|v| v.len() < n) {
                return Err(GpuError::Contract(
                    "VV10 response: fewer point values than points".into(),
                ));
            }
            if math == Vv10Math::Fp32 {
                return Err(GpuError::Contract(
                    "VV10 response: the engine runs fp64 or df64; fp32 is the per-call choice of sweep"
                        .into(),
                ));
            }
            let npad = n.div_ceil(TILE).max(1) * TILE;
            // x, y, z, omega, kappa and a zero sixth array (vv10_split's layout)
            let mut h = vec![0.0; 6 * npad];
            for (k, v) in pts.iter().enumerate() {
                h[k * npad..k * npad + n].copy_from_slice(&v[..n]);
                if k >= 3 {
                    h[k * npad + n..(k + 1) * npad].fill(1.0);
                }
            }
            let src = RESP_MODULE.source();
            let k = |e: &str| gpu.kernel(RESP_MODULE.name, &src, e);
            let (k64, kdf, k32, split) = (
                k("vv10_resp_f64")?,
                k("vv10_resp_df64")?,
                k("vv10_resp_f32")?,
                k("vv10_split")?,
            );
            let q = gpu.upload(&h)?;
            let mut qs = gpu.alloc_zeros::<f32>(6 * npad)?;
            // SAFETY: matches vv10_split; reads 6 npad values, writes 6 npad.
            unsafe {
                gpu.launch(
                    &split,
                    Launch::linear(npad, 256),
                    &[
                        Arg::I32(npad as i32),
                        Arg::I32(0),
                        Arg::buf(&q),
                        Arg::buf_mut(&mut qs),
                    ],
                )
            }?;
            let qf = if math == Vv10Math::DoubleFloat {
                let mut qf = gpu.alloc_zeros::<f32>(12 * npad)?;
                // SAFETY: matches vv10_split; reads 6 npad values, writes 12 npad.
                unsafe {
                    gpu.launch(
                        &split,
                        Launch::linear(npad, 256),
                        &[
                            Arg::I32(npad as i32),
                            Arg::I32(1),
                            Arg::buf(&q),
                            Arg::buf_mut(&mut qf),
                        ],
                    )
                }?;
                Some(qf)
            } else {
                None
            };
            Ok(GpuVv10Resp {
                k64,
                kdf,
                k32,
                math,
                n,
                npad,
                q,
                qf,
                qs,
                last_seconds: 0.0,
                total_seconds: 0.0,
            })
        }

        pub fn math(&self) -> Vv10Math {
            self.math
        }

        /// The device bytes a call with `npp` perturbations allocates (coefficients with `ld`
        /// values per perturbation and component, and the sums).
        pub fn call_bytes(&self, ld: usize, npp: usize) -> usize {
            8 * 3 * npp * (ld + self.npad)
        }

        /// P1, P2, P3 (without their factor 3/2) of `npp` perturbations from their coefficients
        /// `coef[(p 3 + c) ld + j]` (alpha, b, c at the active point j < n): the sums of point i
        /// at `[(c npp + p) npad_out + i]` of the returned vector, `npad_out` = [`Self::npad`];
        /// with `fp32` in FP32 (each tile summed in FP32, the tiles in FP64), else in the
        /// engine's arithmetic.
        pub fn sweep(
            &mut self,
            gpu: &Gpu,
            coef: &[f64],
            ld: usize,
            npp: usize,
            fp32: bool,
        ) -> Result<Vec<f64>, GpuError> {
            if npp == 0 {
                return Ok(Vec::new());
            }
            if ld < self.n || coef.len() < 3 * npp * ld {
                return Err(GpuError::Contract(
                    "VV10 response: coefficients of the wrong size".into(),
                ));
            }
            let npad = self.npad;
            let cd = gpu.upload(&coef[..3 * npp * ld])?;
            let mut out = gpu.alloc_zeros::<f64>(3 * npp * npad)?;
            let cfg = Launch {
                grid: (
                    (npad / TILE) as u32,
                    npp.div_ceil(RESP_PERTURBATIONS) as u32,
                    1,
                ),
                block: (TILE as u32, 1, 1),
                shared_bytes: 0,
            };
            let m0 = gpu.mark()?;
            let (kern, q) = match (&self.qf, fp32) {
                (_, true) => (&self.k32, Arg::buf(&self.qs)),
                (Some(qf), false) => (&self.kdf, Arg::buf(qf)),
                (None, false) => (&self.k64, Arg::buf(&self.q)),
            };
            // SAFETY: matches vv10_resp_f64 / vv10_resp_df64 / vv10_resp_f32: reads 5 npad
            // values (10 npad split, 5 npad FP32) and 3 npp ld coefficients, writes 3 npp npad
            // sums.
            unsafe {
                gpu.launch(
                    kern,
                    cfg,
                    &[
                        Arg::I32(self.n as i32),
                        Arg::I32(npad as i32),
                        Arg::I32(ld as i32),
                        Arg::I32(npp as i32),
                        q,
                        Arg::buf(&cd),
                        Arg::buf_mut(&mut out),
                    ],
                )
            }?;
            self.last_seconds = m0.seconds_until(&gpu.mark()?)?;
            self.total_seconds += self.last_seconds;
            gpu.download(&out)
        }

        /// The padded number of points of [`Self::sweep`]'s output.
        pub fn npad(&self) -> usize {
            self.npad
        }
    }
}
