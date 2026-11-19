//! The PCM's charge solve on the device (`GpuPcmSolver`, feature `gpu`): the device counterpart
//! of linus's CPU `solvent::pcm::PcmOperator` (Lange & Herbert, J. Chem. Phys. 133, 244111
//! (2010), PySCF's conventions).
//!
//! With the solute's potential phi at the points the charges solve K q = R phi:
//!
//!   C-PCM, COSMO   K = S,                               R = -f I
//!   IEF-PCM        K = S - f/(2 pi) D A S,              R = -f (I - D A / (2 pi))
//!   SS(V)PE        K = S - f/(4 pi) (D A S + S A D^T),  R as IEF
//!
//! S_ij = erf(zeta_ij r_ij) / r_ij, zeta_ij = zeta_i zeta_j / sqrt(zeta_i^2 + zeta_j^2),
//! S_ii = zeta_i sqrt(2/pi) / F_i (F the switching factor); D_ij = -S'(r_ij) (r_i - r_j).n_j / r_ij,
//! D_ii = -zeta_i sqrt(2/pi) / (2 R_i); A = diag(areas). The symmetrized charges
//! q_sym = (K^-1 R + R^T K^-T) phi / 2 give the energy 1/2 q_sym . phi.
//!
//! S and D are built by element kernels (one writer per element), D A S by cuBLAS, and K is
//! factored once per geometry by cuSOLVER in its deterministic mode: Cholesky for C-PCM and COSMO,
//! LU with partial pivoting otherwise (as the CPU). Each solve is a pair of triangular solves
//! (potrs, getrs) and, for IEF-PCM and SS(V)PE, two products with D. Bitwise reproducible run to
//! run; against the CPU's dense solve, equal to round-off (different factorization order).
//!
//! For gradients the solver also gives the O(n^2) parts of the derivative contractions of S and
//! D per point (the CPU's `PcmOperator::ds_points` and `dd_points`): one thread per point over
//! a chunk of the other points, the chunks summed in a fixed order.

/// The form of K and R.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PcmKind {
    /// C-PCM and COSMO (they differ only in f).
    Conductor,
    Ief,
    Ssvpe,
}

/// The surface of the cavity (bohr), per point.
#[derive(Clone, Debug, Default)]
pub struct PcmSurface {
    pub r: Vec<[f64; 3]>,
    /// The Gaussian exponent of the point's charge (finite: the charge-charge interactions are
    /// Gaussian in every preset).
    pub zeta: Vec<f64>,
    /// The switching factor F_i.
    pub switch: Vec<f64>,
    /// Outward unit normal (IEF-PCM, SS(V)PE).
    pub normal: Vec<[f64; 3]>,
    pub area: Vec<f64>,
    /// The radius of the point's sphere.
    pub radius: Vec<f64>,
}

impl PcmSurface {
    pub fn len(&self) -> usize {
        self.r.len()
    }

    pub fn is_empty(&self) -> bool {
        self.r.is_empty()
    }
}

#[cfg(feature = "gpu")]
pub(crate) use device::MODULE;
#[cfg(feature = "gpu")]
pub use device::{GpuPcmSolver, PcmCharges};

#[cfg(feature = "gpu")]
mod device {
    use super::*;
    use crate::blas::{GemmDims, MatMut, MatRef, Op};
    use crate::error::GpuError;
    use crate::kernels::registry::KernelModule;
    use crate::linalg::{Arith, Blas1, Gemm, Operand, Scale};
    use crate::pool::PlanViolation;
    use crate::solver::Solver;
    use crate::{Arg, DeviceBuf, Gpu, Kernel, Launch};
    use std::borrow::Cow;

    const KERNELS: &str = include_str!("../../device/pcm/solver.cu");

    /// The charge solve's module.
    pub(crate) const MODULE: KernelModule = KernelModule::new(
        "pcm_solver",
        || Cow::Borrowed(KERNELS),
        &[
            "pcm_s_matrix",
            "pcm_d_matrix",
            "pcm_k_matrix",
            "pcm_ds_points",
            "pcm_dd_points",
            "pcm_sum_parts",
        ],
    );

    struct Kernels {
        s: Kernel,
        d: Kernel,
        k: Kernel,
        blas1: Blas1,
        ds_points: Kernel,
        dd_points: Kernel,
        sum_parts: Kernel,
    }

    /// The charges of one potential: q = K^-1 R phi, y = K^-T phi and q_sym (device buffers of
    /// npts; for C-PCM and COSMO q = q_sym).
    pub struct PcmCharges {
        pub q: DeviceBuf<f64>,
        pub y: DeviceBuf<f64>,
        pub q_sym: DeviceBuf<f64>,
    }

    impl PcmCharges {
        pub fn new(gpu: &Gpu, n: usize) -> Result<Self, GpuError> {
            Ok(PcmCharges {
                q: gpu.alloc_zeros(n.max(1))?,
                y: gpu.alloc_zeros(n.max(1))?,
                q_sym: gpu.alloc_zeros(n.max(1))?,
            })
        }
    }

    /// The factored PCM response at one geometry on the device (see the module docs).
    pub struct GpuPcmSolver {
        n: usize,
        kind: PcmKind,
        f: f64,
        /// S (IEF-PCM, SS(V)PE; C-PCM factors it in place).
        s: Option<DeviceBuf<f64>>,
        /// D (IEF-PCM, SS(V)PE).
        d: Option<DeviceBuf<f64>>,
        /// The factor of K: lower Cholesky, or LU with the pivots.
        k: DeviceBuf<f64>,
        ipiv: Option<DeviceBuf<i32>>,
        area: DeviceBuf<f64>,
        /// The points (n x 3), their exponents and normals (the gradient's contractions).
        r: DeviceBuf<f64>,
        zeta: DeviceBuf<f64>,
        normal: Option<DeviceBuf<f64>>,
        tmp: DeviceBuf<f64>,
        tmp2: DeviceBuf<f64>,
        info: DeviceBuf<i32>,
        solver: Solver,
        gemm: Gemm,
        kernels: Kernels,
        /// Device bytes held.
        pub bytes: usize,
        /// The draws of the matrices beyond their grant ([`GpuPcmSolver::matrix_bytes`]; plan
        /// violations).
        pub plan_violations: Vec<PlanViolation>,
        /// Setup seconds (wall clock, synchronized).
        pub setup_seconds: f64,
    }

    fn launch_n2(n: usize) -> Launch {
        Launch::linear(n * n, 256)
    }

    impl GpuPcmSolver {
        /// Builds and factors K of `surf` for the model `kind` with f = `f_eps`.
        pub fn new(
            gpu: &Gpu,
            kind: PcmKind,
            f_eps: f64,
            surf: &PcmSurface,
        ) -> Result<GpuPcmSolver, GpuError> {
            let t0 = std::time::Instant::now();
            let n = surf.len();
            if surf.zeta.len() != n
                || surf.switch.len() != n
                || surf.area.len() != n
                || (kind != PcmKind::Conductor
                    && (surf.normal.len() != n || surf.radius.len() != n))
            {
                return Err(GpuError::Contract(
                    "PCM solver: surface arrays of unequal length".into(),
                ));
            }
            if surf.zeta.iter().any(|z| !z.is_finite()) {
                return Err(GpuError::Contract(
                    "PCM solver: the charges' interactions need finite Gaussian exponents".into(),
                ));
            }
            let src = MODULE.source();
            let kk = |e: &str| gpu.kernel(MODULE.name, &src, e);
            let kernels = Kernels {
                s: kk("pcm_s_matrix")?,
                d: kk("pcm_d_matrix")?,
                k: kk("pcm_k_matrix")?,
                blas1: Blas1::new(gpu)?,
                ds_points: kk("pcm_ds_points")?,
                dd_points: kk("pcm_dd_points")?,
                sum_parts: kk("pcm_sum_parts")?,
            };
            let solver = Solver::new(gpu)?;
            let gemm = Gemm::new(gpu)?;
            let flat: Vec<f64> = surf.r.iter().flatten().copied().collect();
            let r = gpu.upload(&flat)?;
            let zeta = gpu.upload(&surf.zeta)?;
            let area = gpu.upload(&surf.area)?;
            let n2 = (n * n).max(1);
            // the matrices as the memory plan's grant, all of them before the first
            let grant = gpu.grant(Self::matrix_bytes(kind, n), "the PCM's surface matrices")?;
            let g = Some(&grant);
            let mut s = gpu.alloc_zeros_granted::<f64>(g, n2)?;
            {
                let sw = gpu.upload(&surf.switch)?;
                // SAFETY: matches pcm_s_matrix; n points of 3, n exponents and switches, n^2
                // outputs.
                unsafe {
                    gpu.launch(
                        &kernels.s,
                        launch_n2(n),
                        &[
                            Arg::I64(n as i64),
                            Arg::buf(&r),
                            Arg::buf(&zeta),
                            Arg::buf(&sw),
                            Arg::buf_mut(&mut s),
                        ],
                    )
                }?;
            }
            let mut bytes = 8 * (3 * n + 2 * n + 2 * n);
            let mut normal = None;
            let (s, d, k, ipiv) = match kind {
                PcmKind::Conductor => {
                    solver.potrf_lower(gpu, &mut s, n)?;
                    bytes += 8 * n2;
                    (None, None, s, None)
                }
                PcmKind::Ief | PcmKind::Ssvpe => {
                    let flat_n: Vec<f64> = surf.normal.iter().flatten().copied().collect();
                    let nrm = gpu.upload(&flat_n)?;
                    normal = Some(gpu.upload(&flat_n)?);
                    let rad = gpu.upload(&surf.radius)?;
                    let mut d = gpu.alloc_zeros_granted::<f64>(g, n2)?;
                    let mut da = gpu.alloc_zeros_granted::<f64>(g, n2)?;
                    // SAFETY: matches pcm_d_matrix; n points, exponents, normals, radii, areas;
                    // n^2 outputs each.
                    unsafe {
                        gpu.launch(
                            &kernels.d,
                            launch_n2(n),
                            &[
                                Arg::I64(n as i64),
                                Arg::buf(&r),
                                Arg::buf(&zeta),
                                Arg::buf(&nrm),
                                Arg::buf(&rad),
                                Arg::buf(&area),
                                Arg::buf_mut(&mut d),
                                Arg::buf_mut(&mut da),
                            ],
                        )
                    }?;
                    // M = (D A) S, then K = S - c (M (+ M^T)) into the D A buffer
                    let mut m = gpu.alloc_zeros_granted::<f64>(g, n2)?;
                    gemm.gemm(
                        Arith::Fp64,
                        GemmDims::new(Op::N, Op::N, n, n, n),
                        (1.0, 0.0),
                        MatRef::new(&da, 0, n),
                        MatRef::new(&s, 0, n),
                        MatMut::new(&mut m, 0, n),
                    )?;
                    let (c, sym) = if kind == PcmKind::Ief {
                        (f_eps / (2.0 * std::f64::consts::PI), 0)
                    } else {
                        (f_eps / (4.0 * std::f64::consts::PI), 1)
                    };
                    let mut k = da;
                    // SAFETY: matches pcm_k_matrix; n^2 elements of s, m and k.
                    unsafe {
                        gpu.launch(
                            &kernels.k,
                            launch_n2(n),
                            &[
                                Arg::I64(n as i64),
                                Arg::buf(&s),
                                Arg::F64(c),
                                Arg::I32(sym),
                                Arg::buf(&m),
                                Arg::buf_mut(&mut k),
                            ],
                        )
                    }?;
                    drop(m);
                    let mut ipiv = gpu.alloc_zeros_granted::<i32>(g, n.max(1))?;
                    solver.getrf(gpu, &mut k, n, &mut ipiv)?;
                    bytes += 3 * 8 * n2 + 4 * n;
                    (Some(s), Some(d), k, Some(ipiv))
                }
            };
            gpu.synchronize()?;
            Ok(GpuPcmSolver {
                n,
                kind,
                f: f_eps,
                s,
                d,
                k,
                ipiv,
                area,
                r,
                zeta,
                normal,
                tmp: gpu.alloc_zeros(n.max(1))?,
                tmp2: gpu.alloc_zeros(n.max(1))?,
                info: gpu.alloc_zeros(1)?,
                solver,
                gemm,
                kernels,
                bytes,
                plan_violations: grant.violations(),
                setup_seconds: t0.elapsed().as_secs_f64(),
            })
        }

        /// Device bytes of the surface matrices of `n` points for `kind` at their peak: S
        /// factored in place (conductor), or S, D, DA (factored as K) and the product M beside
        /// them, and the pivots.
        pub fn matrix_bytes(kind: PcmKind, n: usize) -> usize {
            let n2 = (n * n).max(1);
            match kind {
                PcmKind::Conductor => 8 * n2,
                PcmKind::Ief | PcmKind::Ssvpe => 4 * 8 * n2 + 4 * n.max(1),
            }
        }

        pub fn n(&self) -> usize {
            self.n
        }

        pub fn kind(&self) -> PcmKind {
            self.kind
        }

        /// The O(n^2) part of a derivative contraction per point (`dd`: of D, IEF-PCM and
        /// SS(V)PE; else of S) for the coefficient vectors `a` and `b` (host, n each): the CPU's
        /// `PcmOperator::dd_points` or `ds_points`.
        pub fn contraction_points(
            &self,
            gpu: &Gpu,
            dd: bool,
            a: &[f64],
            b: &[f64],
        ) -> Result<Vec<[f64; 3]>, GpuError> {
            let n = self.n;
            if a.len() != n || b.len() != n {
                return Err(GpuError::Contract(
                    "PCM contraction: vectors of n values".into(),
                ));
            }
            if n == 0 {
                return Ok(Vec::new());
            }
            let nrm = if dd {
                Some(self.normal.as_ref().ok_or_else(|| {
                    GpuError::Contract("PCM contraction: D is kept for IEF-PCM and SS(V)PE".into())
                })?)
            } else {
                None
            };
            let ad = gpu.upload(a)?;
            let bd = gpu.upload(b)?;
            // chunks of the other points: about 8 blocks per SM in all
            let nb = n.div_ceil(128);
            let target = 8 * gpu.info().sm_count.max(1) as usize;
            let nch = target.div_ceil(nb).clamp(1, n.div_ceil(64));
            let jch = n.div_ceil(nch);
            let nch = n.div_ceil(jch);
            let mut part = gpu.alloc_zeros::<f64>(3 * n * nch)?;
            let cfg = Launch {
                grid: (nb as u32, nch as u32, 1),
                block: (128, 1, 1),
                shared_bytes: 0,
            };
            match nrm {
                // SAFETY: matches pcm_dd_points; n points (3 values), exponents, normals,
                // coefficients; 3 n nch partials.
                Some(nrm) => unsafe {
                    gpu.launch(
                        &self.kernels.dd_points,
                        cfg,
                        &[
                            Arg::I64(n as i64),
                            Arg::I64(jch as i64),
                            Arg::buf(&self.r),
                            Arg::buf(&self.zeta),
                            Arg::buf(nrm),
                            Arg::buf(&ad),
                            Arg::buf(&bd),
                            Arg::buf_mut(&mut part),
                        ],
                    )
                }?,
                // SAFETY: matches pcm_ds_points; as above without normals.
                None => unsafe {
                    gpu.launch(
                        &self.kernels.ds_points,
                        cfg,
                        &[
                            Arg::I64(n as i64),
                            Arg::I64(jch as i64),
                            Arg::buf(&self.r),
                            Arg::buf(&self.zeta),
                            Arg::buf(&ad),
                            Arg::buf(&bd),
                            Arg::buf_mut(&mut part),
                        ],
                    )
                }?,
            }
            let mut out = gpu.alloc_zeros::<f64>(3 * n)?;
            // SAFETY: matches pcm_sum_parts; nch blocks of 3 n partials, 3 n outputs.
            unsafe {
                gpu.launch(
                    &self.kernels.sum_parts,
                    Launch::linear(3 * n, 256),
                    &[
                        Arg::I64(3 * n as i64),
                        Arg::I64(nch as i64),
                        Arg::buf(&part),
                        Arg::buf_mut(&mut out),
                    ],
                )
            }?;
            let h = gpu.download(&out)?;
            Ok(h.chunks(3).take(n).map(|c| [c[0], c[1], c[2]]).collect())
        }

        /// out = op(M) x for M = S or D (IEF-PCM, SS(V)PE), `trans` for M^T.
        pub fn matvec(
            &self,
            which_d: bool,
            trans: bool,
            x: &DeviceBuf<f64>,
            out: &mut DeviceBuf<f64>,
        ) -> Result<(), GpuError> {
            let m = if which_d { &self.d } else { &self.s }
                .as_ref()
                .ok_or_else(|| {
                    GpuError::Contract(
                        "PCM solver: S and D are kept for IEF-PCM and SS(V)PE only".into(),
                    )
                })?;
            let n = self.n;
            let op = if trans { Op::T } else { Op::N };
            self.gemm.gemm(
                Arith::Fp64,
                GemmDims::new(op, Op::N, n, 1, n),
                (1.0, 0.0),
                MatRef::new(m, 0, n),
                MatRef::new(x, 0, n),
                MatMut::new(out, 0, n),
            )
        }

        /// The charges of the potential `phi` (n values) into `out` (the CPU's
        /// `PcmOperator::solve`). Queued on the device's stream.
        pub fn solve(
            &mut self,
            gpu: &Gpu,
            phi: &DeviceBuf<f64>,
            out: &mut PcmCharges,
        ) -> Result<(), GpuError> {
            let n = self.n;
            if phi.len() < n || out.q.len() < n {
                return Err(GpuError::Contract("PCM solve: buffers too small".into()));
            }
            // y = K^-T phi
            gpu.copy_range(phi, 0, &mut out.y, 0, n)?;
            match self.kind {
                PcmKind::Conductor => {
                    self.solver
                        .potrs_lower(&self.k, n, &mut out.y, 1, &mut self.info)?;
                    // q = q_sym = -f y
                    self.kernels.blas1.scale(
                        gpu,
                        n,
                        Scale::Scalar(-self.f),
                        Operand::Buf(&out.y),
                        &mut out.q,
                    )?;
                    gpu.copy_range(&out.q, 0, &mut out.q_sym, 0, n)?;
                }
                PcmKind::Ief | PcmKind::Ssvpe => {
                    let ipiv = self.ipiv.as_ref().unwrap();
                    self.solver
                        .getrs(&self.k, n, ipiv, true, &mut out.y, 1, &mut self.info)?;
                    // R phi = -f (phi - D (A phi) / 2 pi), q = K^-1 R phi
                    let mut a_phi = std::mem::replace(&mut self.tmp, gpu.alloc_zeros(1)?);
                    // A phi
                    self.kernels.blas1.scale(
                        gpu,
                        n,
                        Scale::Diagonal(&self.area),
                        Operand::Buf(phi),
                        &mut a_phi,
                    )?;
                    let mut dap = std::mem::replace(&mut self.tmp2, gpu.alloc_zeros(1)?);
                    self.matvec(true, false, &a_phi, &mut dap)?;
                    let tp = 2.0 * std::f64::consts::PI;
                    self.kernels.blas1.axpby(
                        gpu,
                        n,
                        &[
                            (-self.f, Operand::Buf(phi)),
                            (self.f / tp, Operand::Buf(&dap)),
                        ],
                        &mut out.q,
                    )?;
                    self.solver
                        .getrs(&self.k, n, ipiv, false, &mut out.q, 1, &mut self.info)?;
                    // R^T y = -f (y - A (D^T y) / 2 pi); q_sym = (q + R^T y) / 2
                    self.matvec(true, true, &out.y, &mut dap)?;
                    // A D^T y
                    self.kernels.blas1.scale(
                        gpu,
                        n,
                        Scale::Diagonal(&self.area),
                        Operand::Buf(&dap),
                        &mut a_phi,
                    )?;
                    // q_sym = q / 2 - f/2 y + f/(4 pi) A D^T y
                    self.kernels.blas1.axpby(
                        gpu,
                        n,
                        &[
                            (0.5, Operand::Buf(&out.q)),
                            (-0.5 * self.f, Operand::Buf(&out.y)),
                            (0.5 * self.f / tp, Operand::Buf(&a_phi)),
                        ],
                        &mut out.q_sym,
                    )?;
                    self.tmp = a_phi;
                    self.tmp2 = dap;
                }
            }
            Ok(())
        }
    }
}
