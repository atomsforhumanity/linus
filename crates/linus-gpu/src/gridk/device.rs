//! The device engine of grid-K builds (see [`super`]).

use super::*;
use crate::blas::{GemmDims, MatMut as DMatMut, MatRef as DMatRef, Op};
use crate::engine::{Phases, Timeline};
use crate::kernels::colloc::GpuCollocator;
use crate::kernels::gridk::{L_MAX, SOURCE};
use crate::kernels::registry::KernelModule;
use crate::linalg::Gemm;
use crate::scf::kernels::{Ops, Task};
use crate::solver::Solver;
use crate::{Arg, DeviceBuf, Gpu, GpuError, Kernel, Launch};
use linus_core::Arith;
use std::borrow::Cow;
use std::sync::Arc;

struct DevClass {
    kernel: Kernel,
    npair: usize,
    info: DeviceBuf<i32>,
    est: DeviceBuf<f64>,
    prim: DeviceBuf<f64>,
    ecoef: DeviceBuf<f64>,
}

/// The device time of the builds so far by phase (seconds, from events), and the builds.
#[derive(Clone, Debug, Default)]
pub struct GpuGridKTimes {
    /// Y = Xbar (D_d T).
    pub gemm_y: f64,
    /// The per-chunk, per-shell max |Y|.
    pub ymax: f64,
    /// The potential-integral kernels and the sum of their slices.
    pub ints: f64,
    /// G = Xbar^T Z and K = G T^T.
    pub gemm_k: f64,
    pub builds: usize,
}

impl Phases for GpuGridKTimes {
    fn phases(&self) -> impl IntoIterator<Item = (&'static str, f64)> {
        [
            ("gemm_y", self.gemm_y),
            ("ymax", self.ymax),
            ("ints", self.ints),
            ("gemm_k", self.gemm_k),
        ]
    }
}

/// The options of a [`GpuGridK`]: linus-jk's `GridKConfig` values of the engine.
#[derive(Clone, Copy, Debug)]
pub struct GpuGridKOptions {
    /// Eigenvalues of S at or below this are dropped by the canonical orthogonalization.
    pub lindep: f64,
    /// The symmetric overlap fit (else Xbar = X).
    pub fit: bool,
    /// The screening threshold of the potential-integral kernels.
    pub thresh: f64,
    /// Primitive pairs whose estimate is below this are dropped at setup.
    pub prim_cut: f64,
    /// The kernel style of every class (a study's); None: [`Style::default_for`] per class.
    pub style: Option<Style>,
}

/// What the device's overlap fit found ([`GpuGridK::fitted`]).
#[derive(Clone, Debug, Default)]
pub struct FitInfo {
    /// Orbitals kept by the canonical orthogonalization.
    pub nmo: usize,
    /// The smallest eigenvalue of the grid's metric Omega of the orthogonal orbitals.
    pub theta_min: f64,
    /// Seconds of the setup (collocation, fit, Xbar).
    pub seconds: f64,
}

/// The engine's hand-written module ([`SOURCE`]).
pub(crate) const MODULE: KernelModule = KernelModule::new(
    "gridk",
    || Cow::Borrowed(SOURCE),
    &[
        "gridk_rowscale",
        "gridk_colscale",
        "gridk_symadd",
        "gridk_slices",
        "gridk_cxw",
        "gridk_ymax",
    ],
);

/// The element kernels of the setup.
struct Aux {
    rowscale: Kernel,
    colscale: Kernel,
    cxw: Kernel,
    symadd: Kernel,
}

impl Aux {
    fn new(gpu: &Gpu) -> Result<Aux, GpuError> {
        let k = |e: &str| gpu.kernel(MODULE.name, SOURCE, e);
        Ok(Aux {
            rowscale: k("gridk_rowscale")?,
            colscale: k("gridk_colscale")?,
            cxw: k("gridk_cxw")?,
            symadd: k("gridk_symadd")?,
        })
    }

    /// out\[c ld + r\] = v\[r or c\] x\[c ld + r\] for an nrow x ncol block (`rows`: v indexes rows).
    #[allow(clippy::too_many_arguments)] // the block's shape and three operands, as the kernel's
    fn scale(
        &self,
        gpu: &Gpu,
        rows: bool,
        nrow: usize,
        ncol: usize,
        ld: usize,
        v: &DeviceBuf<f64>,
        x: crate::DevPtr<f64>,
        out: crate::DevPtr<f64>,
    ) -> Result<(), GpuError> {
        let k = if rows { &self.rowscale } else { &self.colscale };
        // SAFETY: the callers' blocks of nrow x ncol elements at stride ld lie in their buffers,
        // and v holds nrow (rows) or ncol elements
        unsafe {
            gpu.launch(
                k,
                Launch::linear(nrow * ncol, 256),
                &[
                    Arg::I64(nrow as i64),
                    Arg::I64(ncol as i64),
                    Arg::I64(ld as i64),
                    Arg::buf(v),
                    x.into(),
                    out.into(),
                ],
            )
        }
    }
}

/// The device part of grid-K builds (module docs).
pub struct GpuGridK {
    gpu: Arc<Gpu>,
    gemm: Gemm,
    npts: usize,
    nbf: usize,
    ncart: usize,
    nshell: usize,
    nchunk: usize,
    /// Slices of every class's pairs (blocks per chunk), each with its own copy of Z.
    nslice: usize,
    /// The element kernels of the setup and the device loop's builds.
    aux: Aux,
    /// The device loop's differences and energy dot products (the SCF loop's own kernels).
    ops: Ops,
    /// The two dot products of a device build.
    dots: DeviceBuf<f64>,
    slices_k: Kernel,
    t: DeviceBuf<f64>,
    xbar: DeviceBuf<f64>,
    pts: DeviceBuf<f64>,
    chunk: DeviceBuf<i32>,
    cxw: DeviceBuf<f64>,
    /// Each chunk's bounding sphere (x, y, z, r).
    cgeo: DeviceBuf<f64>,
    shoff: DeviceBuf<i32>,
    btab: DeviceBuf<f64>,
    classes: Vec<DevClass>,
    ymax_k: Kernel,
    ymax: DeviceBuf<f64>,
    y: DeviceBuf<f64>,
    z: DeviceBuf<f64>,
    d: DeviceBuf<f64>,
    dt: DeviceBuf<f64>,
    g: DeviceBuf<f64>,
    k: DeviceBuf<f64>,
    op: (f64, f64, f64),
    thresh: f64,
    times: GpuGridKTimes,
    fit: FitInfo,
    /// The grant of the large buffers (X and the fit's during the setup; Xbar, Y, Z).
    grant: crate::pool::Grant,
}

/// What [`GpuGridK::fitted`] decided before its buffers: the Cartesian layout, the chunks of
/// points, the slices and the grant.
struct Prepared {
    layout: crate::cpu::CartLayout,
    ch: Vec<crate::cpu::Batch>,
    nslice: usize,
    grant: crate::pool::Grant,
}

impl GpuGridK {
    /// The engine with the collocation and, with `opts.fit`, the symmetric overlap fit computed
    /// on the device: the canonical orthogonalization of the analytic overlap `s` (nbf x nbf,
    /// column-major; eigenvalues above `opts.lindep` kept), Omega = A^T X^T W X A and its
    /// inverse square root, M = B T A^T and Xbar = X M^T (`linus_jk::gridk::OverlapFit`, the
    /// same steps as GEMMs and cuSOLVER eigendecompositions). A grid whose metric is not
    /// positive definite is a numerical error.
    pub fn fitted(
        gpu: Arc<Gpu>,
        basis: &BasisSet,
        grid: &MolecularGrid,
        s: &[f64],
        opts: GpuGridKOptions,
    ) -> Result<GpuGridK, GpuError> {
        let GpuGridKOptions { lindep, fit, .. } = opts;
        let t0 = std::time::Instant::now();
        let (n, np) = (basis.nbf, grid.len());
        if s.len() != n * n {
            return Err(GpuError::Contract(format!(
                "grid-K: an overlap of {} elements for {n} functions",
                s.len()
            )));
        }
        if let Some(l) = basis
            .shells
            .iter()
            .map(|s| s.l)
            .max()
            .filter(|&l| l > L_MAX)
        {
            return Err(GpuError::NotCovered(format!(
                "grid-K: the potential-integral kernels cover l <= {L_MAX}, the basis has l = {l}"
            )));
        }
        // the engine's large buffers from one grant (the setup's peak or what it keeps)
        let layout = basis.cart_layout();
        let ch = grid.sub_batches(THREADS as usize);
        let nslice = slices(gpu.info().sm_count as usize, ch.len(), np * layout.ncart());
        let grant = gpu.grant(
            device_bytes(np, n, layout.ncart(), nslice),
            "the grid-K engine",
        )?;
        let big = |len: usize| gpu.alloc_zeros_granted::<f64>(Some(&grant), len);
        let mut gemm = Gemm::new(&gpu)?;
        gemm.prepare(&gpu, Arith::Fp64)?;
        let aux = Aux::new(&gpu)?;
        let fp64 = Arith::Fp64;
        // X (np x n, column-major)
        let coll = GpuCollocator::new(&gpu, basis, 0)?;
        let (dx, dy, dz) = (
            gpu.upload(&grid.x)?,
            gpu.upload(&grid.y)?,
            gpu.upload(&grid.z)?,
        );
        let mut x = big(n * np)?;
        coll.eval(&gpu, &dx, &dy, &dz, np, np, &mut x)?;
        drop((dx, dy, dz));
        let mut info = FitInfo {
            nmo: n,
            ..Default::default()
        };
        let xbar = if !fit {
            x
        } else {
            let solver = Solver::new(&gpu)?;
            // S = U s U^T; the kept columns are the last m (ascending eigenvalues)
            let mut u = gpu.upload(s)?;
            let mut sv = gpu.alloc_zeros::<f64>(n)?;
            solver.syevd(&gpu, &mut u, n, &mut sv)?;
            let svh = gpu.download(&sv)?;
            let m = svh.iter().filter(|&&v| v > lindep).count();
            let off = (n - m) * n;
            let va = gpu.upload(
                &svh[n - m..]
                    .iter()
                    .map(|v| 1.0 / v.sqrt())
                    .collect::<Vec<_>>(),
            )?;
            let vb = gpu.upload(&svh[n - m..].iter().map(|v| v.sqrt()).collect::<Vec<_>>())?;
            let a = gpu.alloc_zeros::<f64>(n * m)?;
            let b = gpu.alloc_zeros::<f64>(n * m)?;
            aux.scale(&gpu, false, n, m, n, &va, u.at(off), a.at(0))?;
            aux.scale(&gpu, false, n, m, n, &vb, u.at(off), b.at(0))?;
            drop(u);
            // Omega = (X A)^T W (X A)
            let mut chi = big(np * m)?;
            gemm.gemm(
                fp64,
                GemmDims::new(Op::N, Op::N, np, m, n),
                (1.0, 0.0),
                DMatRef::new(&x, 0, np),
                DMatRef::new(&a, 0, n),
                DMatMut::new(&mut chi, 0, np),
            )?;
            let w = gpu.upload(&grid.w)?;
            let wchi = big(np * m)?;
            aux.scale(&gpu, true, np, m, np, &w, chi.at(0), wchi.at(0))?;
            let mut om = gpu.alloc_zeros::<f64>(m * m)?;
            gemm.gemm(
                fp64,
                GemmDims::new(Op::T, Op::N, m, m, np),
                (1.0, 0.0),
                DMatRef::new(&chi, 0, np),
                DMatRef::new(&wchi, 0, np),
                DMatMut::new(&mut om, 0, m),
            )?;
            drop((chi, wchi));
            // T = Theta theta^-1/2 Theta^T
            let mut th = gpu.alloc_zeros::<f64>(m)?;
            solver.syevd(&gpu, &mut om, m, &mut th)?;
            let thh = gpu.download(&th)?;
            info.nmo = m;
            info.theta_min = thh.first().copied().unwrap_or(1.0);
            if info.theta_min <= 0.0 {
                // an input error, as the CPU's (linus_jk::gridk::OverlapFit::new)
                return Err(GpuError::Input(format!(
                    "the exchange grid is too coarse for the overlap fit: its metric of the \
                     orthogonal orbitals has the eigenvalue {:.3e} (use a larger k_grid)",
                    info.theta_min
                )));
            }
            let vt = gpu.upload(&thh.iter().map(|v| 1.0 / v.sqrt()).collect::<Vec<_>>())?;
            let ts = gpu.alloc_zeros::<f64>(m * m)?;
            aux.scale(&gpu, false, m, m, m, &vt, om.at(0), ts.at(0))?;
            let mut t = gpu.alloc_zeros::<f64>(m * m)?;
            gemm.gemm(
                fp64,
                GemmDims::new(Op::N, Op::T, m, m, m),
                (1.0, 0.0),
                DMatRef::new(&ts, 0, m),
                DMatRef::new(&om, 0, m),
                DMatMut::new(&mut t, 0, m),
            )?;
            // M = (B T) A^T, Xbar = X M^T
            let mut bt = gpu.alloc_zeros::<f64>(n * m)?;
            gemm.gemm(
                fp64,
                GemmDims::new(Op::N, Op::N, n, m, m),
                (1.0, 0.0),
                DMatRef::new(&b, 0, n),
                DMatRef::new(&t, 0, m),
                DMatMut::new(&mut bt, 0, n),
            )?;
            let mut mm = gpu.alloc_zeros::<f64>(n * n)?;
            gemm.gemm(
                fp64,
                GemmDims::new(Op::N, Op::T, n, n, m),
                (1.0, 0.0),
                DMatRef::new(&bt, 0, n),
                DMatRef::new(&a, 0, n),
                DMatMut::new(&mut mm, 0, n),
            )?;
            let mut xbar = big(np * n)?;
            gemm.gemm(
                fp64,
                GemmDims::new(Op::N, Op::T, np, n, n),
                (1.0, 0.0),
                DMatRef::new(&x, 0, np),
                DMatRef::new(&mm, 0, n),
                DMatMut::new(&mut xbar, 0, np),
            )?;
            xbar
        };
        let prepared = Prepared {
            layout,
            ch,
            nslice,
            grant,
        };
        let mut e = Self::with_xbar(gpu, gemm, aux, basis, grid, xbar, &opts, prepared)?;
        info.seconds = t0.elapsed().as_secs_f64();
        e.fit = info;
        Ok(e)
    }

    #[allow(clippy::too_many_arguments)] // the parts of the setup `fitted` made, handed on
    fn with_xbar(
        gpu: Arc<Gpu>,
        gemm: Gemm,
        aux: Aux,
        basis: &BasisSet,
        grid: &MolecularGrid,
        xbar: DeviceBuf<f64>,
        opts: &GpuGridKOptions,
        prepared: Prepared,
    ) -> Result<GpuGridK, GpuError> {
        let Prepared {
            layout,
            ch,
            nslice,
            grant,
        } = prepared;
        let npts = grid.len();
        let nbf = basis.nbf;
        let (coff, ncart) = (&layout.offsets, layout.ncart());
        let nshell = basis.shells.len();
        let chunkv: Vec<i32> = ch
            .iter()
            .flat_map(|b| [b.start as i32, b.end as i32])
            .collect();
        let chunk = gpu.upload(&chunkv)?;
        // max_p sum_g w |Xbar_gp| per chunk
        let w = gpu.upload(&grid.w)?;
        let mut cxw = gpu.alloc_zeros::<f64>(ch.len().max(1))?;
        // SAFETY: xbar holds nbf columns of ld = npts; chunk the chunks' point ranges
        unsafe {
            gpu.launch(
                &aux.cxw,
                Launch {
                    grid: (ch.len() as u32, 1, 1),
                    block: (256, 1, 1),
                    shared_bytes: 0,
                },
                &[
                    Arg::buf(&xbar),
                    Arg::I64(npts as i64),
                    Arg::I32(nbf as i32),
                    Arg::buf(&chunk),
                    Arg::buf(&w),
                    Arg::buf_mut(&mut cxw),
                ],
            )?;
        }
        let pts: Vec<f64> = (0..npts)
            .flat_map(|g| [grid.x[g], grid.y[g], grid.z[g], grid.w[g]])
            .collect();
        let shoff: Vec<i32> = coff.iter().map(|&c| c as i32).collect();
        let mut classes = Vec::new();
        for c in pack_pairs(basis, coff, opts.prim_cut) {
            let src = generate(
                c.la,
                c.lb,
                opts.style.unwrap_or_else(|| Style::default_for(c.la, c.lb)),
            );
            classes.push(DevClass {
                kernel: gpu.kernel(&kernel_name(c.la, c.lb), &src, &kernel_name(c.la, c.lb))?,
                npair: c.npair(),
                info: gpu.upload(&c.info)?,
                est: gpu.upload(&c.est)?,
                prim: gpu.upload(&c.prim)?,
                ecoef: gpu.upload(&c.ecoef)?,
            });
        }
        let zlen = npts * ncart;
        let y = gpu.alloc_zeros_granted(Some(&grant), zlen)?;
        let z = gpu.alloc_zeros_granted(Some(&grant), zlen * nslice)?;
        Ok(GpuGridK {
            ymax_k: gpu.kernel(MODULE.name, SOURCE, "gridk_ymax")?,
            slices_k: gpu.kernel(MODULE.name, SOURCE, "gridk_slices")?,
            nslice,
            aux,
            ops: Ops::new(&gpu, 2, 1)?,
            dots: gpu.alloc_zeros(2)?,
            t: gpu.upload(&layout.t)?,
            xbar,
            pts: gpu.upload(&pts)?,
            chunk,
            cxw,
            cgeo: gpu.upload(
                &ch.iter()
                    .flat_map(|b| [b.center[0], b.center[1], b.center[2], b.radius])
                    .collect::<Vec<_>>(),
            )?,
            shoff: gpu.upload(&shoff)?,
            btab: gpu.upload(&crate::kernels::boys::boys_table())?,
            ymax: gpu.alloc_zeros(ch.len() * nshell)?,
            y,
            z,
            d: gpu.alloc_zeros(nbf * nbf)?,
            dt: gpu.alloc_zeros(nbf * ncart)?,
            g: gpu.alloc_zeros(nbf * ncart)?,
            k: gpu.alloc_zeros(nbf * nbf)?,
            gemm,
            npts,
            nbf,
            ncart,
            nshell,
            nchunk: ch.len(),
            classes,
            op: (1.0, 0.0, 0.0),
            thresh: opts.thresh,
            times: GpuGridKTimes::default(),
            fit: FitInfo {
                nmo: nbf,
                ..Default::default()
            },
            grant,
            gpu,
        })
    }

    /// The exchange operator (coefficients included).
    pub fn set_operator(&mut self, op: &Operator) {
        self.op = operator_args(op);
    }

    /// Points of the grid.
    pub fn npoints(&self) -> usize {
        self.npts
    }

    /// The device time of the builds so far by phase.
    pub fn times(&self) -> &GpuGridKTimes {
        &self.times
    }

    /// What the overlap fit found at the setup.
    pub fn fit(&self) -> &FitInfo {
        &self.fit
    }

    /// The dressed collocation Xbar (npts x nbf, column-major) on the host.
    pub fn xbar(&self) -> Result<Vec<f64>, GpuError> {
        self.gpu.download(&self.xbar)
    }

    /// The device bytes granted to the engine ([`device_bytes`]).
    pub fn granted_bytes(&self) -> usize {
        self.grant.granted()
    }

    /// The engine's draws beyond its grant (none where [`device_bytes`] holds).
    pub fn grant_violations(&self) -> Vec<crate::pool::PlanViolation> {
        self.grant.violations()
    }

    /// Kbar = sym(Xbar^T W V(Xbar D_d)) of the difference density `dd` (nbf x nbf, symmetric),
    /// column-major.
    pub fn build(&mut self, dd: &[f64]) -> Result<Vec<f64>, GpuError> {
        let nbf = self.nbf;
        if dd.len() != nbf * nbf {
            return Err(GpuError::Contract(format!(
                "grid-K: a density of {} elements for {nbf} functions",
                dd.len()
            )));
        }
        self.gpu.clone().upload_into(&mut self.d, 0, dd)?;
        self.run_build()?;
        let mut k = self.gpu.download(&self.k)?;
        for j in 0..nbf {
            for i in j + 1..nbf {
                let v = 0.5 * (k[i + j * nbf] + k[j + i * nbf]);
                k[i + j * nbf] = v;
                k[j + i * nbf] = v;
            }
        }
        Ok(k)
    }

    /// On the device: K = K0 + Kbar of D_d = D - D0 for the density at `d[off..off + nbf^2]`,
    /// into `kout[koff..]`, and the energy's terms (D_d . K0, D_d . Kbar), each a compensated
    /// sum in a fixed order (the SCF loop's reductions). `d0`, `k0`: the SAD reference and its
    /// exact exchange (nbf x nbf on the device).
    #[allow(clippy::too_many_arguments)] // two (buffer, offset) operands and the reference pair
    pub fn build_device(
        &mut self,
        d: &DeviceBuf<f64>,
        off: usize,
        d0: &DeviceBuf<f64>,
        k0: &DeviceBuf<f64>,
        kout: &mut DeviceBuf<f64>,
        koff: usize,
    ) -> Result<(f64, f64), GpuError> {
        let gpu = self.gpu.clone();
        let n = self.nbf;
        let n2 = n * n;
        self.ops
            .add(&gpu, n2, (d, off), (d0, 0), -1.0, (&mut self.d, 0))?;
        self.run_build()?;
        // SAFETY: self.k and k0 hold n2 values, kout n2 at koff
        unsafe {
            gpu.launch(
                &self.aux.symadd,
                Launch::linear(n2, 256),
                &[
                    Arg::I64(n as i64),
                    Arg::buf(&self.k),
                    Arg::buf(k0),
                    kout.at(koff).into(),
                ],
            )?;
        }
        // Kbar alone (K - K0) for its dot: symmetrized into self.dt's first n2 values
        // SAFETY: as above; dt holds nbf x ncart >= n2 values
        unsafe {
            gpu.launch(
                &self.aux.symadd,
                Launch::linear(n2, 256),
                &[
                    Arg::I64(n as i64),
                    Arg::buf(&self.k),
                    Arg::Ptr(0),
                    Arg::buf_mut(&mut self.dt),
                ],
            )?;
        }
        let tasks = [
            Task::dot((&self.d, 0), (k0, 0), n2)?,
            Task::dot((&self.d, 0), (&self.dt, 0), n2)?,
        ];
        self.ops.reduce(&gpu, &tasks, &mut self.dots)?;
        let e = gpu.download(&self.dots)?;
        Ok((e[0], e[1]))
    }

    /// The build of the difference density in self.d: self.k = G T^T (not symmetrized).
    fn run_build(&mut self) -> Result<(), GpuError> {
        let gpu = self.gpu.clone();
        let (npts, nbf, nc) = (self.npts, self.nbf, self.ncart);
        let mut tl = Timeline::new(&gpu);
        let m0 = tl.mark()?;
        let fp64 = Arith::Fp64;
        // D_d T, then Y = Xbar (D_d T)
        self.gemm.gemm(
            fp64,
            GemmDims::new(Op::N, Op::N, nbf, nc, nbf),
            (1.0, 0.0),
            DMatRef::new(&self.d, 0, nbf),
            DMatRef::new(&self.t, 0, nbf),
            DMatMut::new(&mut self.dt, 0, nbf),
        )?;
        self.gemm.gemm(
            fp64,
            GemmDims::new(Op::N, Op::N, npts, nc, nbf),
            (1.0, 0.0),
            DMatRef::new(&self.xbar, 0, npts),
            DMatRef::new(&self.dt, 0, nbf),
            DMatMut::new(&mut self.y, 0, npts),
        )?;
        let m1 = tl.close(|t: &mut GpuGridKTimes| &mut t.gemm_y, m0)?;
        // SAFETY: y holds ncart rows of ld = npts; chunk, shoff and ymax are sized at setup
        unsafe {
            gpu.launch(
                &self.ymax_k,
                Launch {
                    grid: (self.nchunk as u32, 1, 1),
                    block: (128, 1, 1),
                    shared_bytes: 0,
                },
                &[
                    Arg::buf(&self.y),
                    Arg::I64(npts as i64),
                    Arg::buf(&self.chunk),
                    Arg::buf(&self.shoff),
                    Arg::I32(self.nshell as i32),
                    Arg::buf_mut(&mut self.ymax),
                ],
            )?;
        }
        let m2 = tl.close(|t: &mut GpuGridKTimes| &mut t.ymax, m1)?;
        gpu.zero(&mut self.z)?;
        let zlen = npts * nc;
        for c in &self.classes {
            // SAFETY: the class's packed pairs and primitives (pack_pairs); the Cartesian rows
            // they name lie in y and in each of z's nslice copies (ld = npts, zlen apart), every
            // chunk's points in pts
            unsafe {
                gpu.launch(
                    &c.kernel,
                    Launch {
                        grid: (self.nchunk as u32, self.nslice as u32, 1),
                        block: (THREADS, 1, 1),
                        shared_bytes: 0,
                    },
                    &[
                        Arg::I32(c.npair as i32),
                        Arg::buf(&c.info),
                        Arg::buf(&c.est),
                        Arg::buf(&c.prim),
                        Arg::buf(&c.ecoef),
                        Arg::buf(&self.pts),
                        Arg::I64(npts as i64),
                        Arg::buf(&self.y),
                        Arg::buf_mut(&mut self.z),
                        Arg::buf(&self.chunk),
                        Arg::buf(&self.cxw),
                        Arg::buf(&self.ymax),
                        Arg::I32(self.nshell as i32),
                        Arg::F64(self.thresh),
                        Arg::buf(&self.btab),
                        Arg::F64(self.op.0),
                        Arg::F64(self.op.1),
                        Arg::F64(self.op.2),
                        Arg::I64(zlen as i64),
                        Arg::buf(&self.cgeo),
                    ],
                )?;
            }
        }
        if self.nslice > 1 {
            // SAFETY: z holds nslice copies of zlen elements
            unsafe {
                gpu.launch(
                    &self.slices_k,
                    Launch::linear(zlen, 256),
                    &[
                        Arg::I64(zlen as i64),
                        Arg::I32(self.nslice as i32),
                        Arg::I64(zlen as i64),
                        Arg::buf_mut(&mut self.z),
                    ],
                )?;
            }
        }
        let m3 = tl.close(|t: &mut GpuGridKTimes| &mut t.ints, m2)?;
        // G = Xbar^T Z, K = G T^T
        self.gemm.gemm(
            fp64,
            GemmDims::new(Op::T, Op::N, nbf, nc, npts),
            (1.0, 0.0),
            DMatRef::new(&self.xbar, 0, npts),
            DMatRef::new(&self.z, 0, npts),
            DMatMut::new(&mut self.g, 0, nbf),
        )?;
        self.gemm.gemm(
            fp64,
            GemmDims::new(Op::N, Op::T, nbf, nbf, nc),
            (1.0, 0.0),
            DMatRef::new(&self.g, 0, nbf),
            DMatRef::new(&self.t, 0, nbf),
            DMatMut::new(&mut self.k, 0, nbf),
        )?;
        tl.close(|t: &mut GpuGridKTimes| &mut t.gemm_k, m3)?;
        tl.add_to(&mut self.times)?;
        self.times.builds += 1;
        Ok(())
    }
}

/// Compensated grid-K as a term of the device SCF loop: per spin channel
/// K = K\[D0\] + Kbar\[D_s - D0\] into the loop's K buffer, and the exchange energy the loop asks
/// for (`crate::scf::FockTerms::exchange_energy`), with D0 and K\[D0\] (the exact exchange of the
/// SAD densities, computed by the caller) on the device.
pub struct GridKTerm {
    pub eng: GpuGridK,
    d0: DeviceBuf<f64>,
    k0: DeviceBuf<f64>,
    /// 1/2 D0 . K\[D0\].
    e00: f64,
    last: Option<f64>,
}

impl GridKTerm {
    /// The term over `eng` with the SAD reference `d0` and its exchange `k0` (nbf x nbf,
    /// column-major), and `e00` = 1/2 D0 . K\[D0\] (the host's, as its builds sum it).
    pub fn new(
        gpu: &Gpu,
        eng: GpuGridK,
        d0: &[f64],
        k0: &[f64],
        e00: f64,
    ) -> Result<Self, GpuError> {
        Ok(GridKTerm {
            d0: gpu.upload(d0)?,
            k0: gpu.upload(k0)?,
            eng,
            e00,
            last: None,
        })
    }

    /// The engine back (for the host loop that may continue the SCF).
    pub fn into_engine(self) -> GpuGridK {
        self.eng
    }

    /// K of each of the state's `nspin` channels (densities at s n^2 in `dens`) into `k`.
    pub fn exchange(
        &mut self,
        dens: &DeviceBuf<f64>,
        nspin: usize,
        k: &mut DeviceBuf<f64>,
    ) -> Result<(), GpuError> {
        let n2 = self.eng.nbf * self.eng.nbf;
        let mut e = 0.0;
        for s in 0..nspin {
            let (e1, e2) = self
                .eng
                .build_device(dens, s * n2, &self.d0, &self.k0, k, s * n2)?;
            e += self.e00 + e1 + 0.5 * e2;
        }
        self.last = Some(e);
        Ok(())
    }

    /// sum_s E_K\[D_s\] of the last [`GridKTerm::exchange`].
    pub fn energy(&self) -> Option<f64> {
        self.last
    }
}
