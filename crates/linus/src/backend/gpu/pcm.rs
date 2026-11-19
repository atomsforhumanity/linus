//! The implicit solvent's SCF and gradient terms on the device (the GPU provider's `pcm`
//! family, `crate::backend`): the device counterpart of [`linus_solvent::PcmScf`] with
//! linus-gpu's PCM engine (`linus_gpu::pcm`: the surface-charge integrals and the dense charge
//! solve).
//!
//! The cavity (surface, switching, areas) and the nuclear potential at the points are built on
//! the host as for the CPU ([`PcmOperator::surface_only`]); the matrices S and D, K's
//! factorization (cuSOLVER, deterministic) and every per-iteration step run on the device:
//! - `exact` mode: the potential of the density at the points (phi = v_n - v_e), the charges, then
//!   the charges' Fock matrix -sum_g q_g (mn|g); two passes over pairs x points; - `predicted` mode
//!   (the default with density-fitted J, [`linus_solvent::ri`]): the charges of the potential
//!   predicted from the previous iteration's exact potential and the change of the J build's fit
//!   coefficients (phi~ = phi_prev - M (c - c_prev), M = (P|g)), then one fused pass for the exact
//!   potential and the predicted charges' Fock matrix. The fit coefficients come from the SCF's J
//!   builds through the Coulomb hook (on the device when the J build is there,
//!   `crate::backend::gpu::df::CoulombFit`, else from the CPU's `DfJk`).
//!
//! The device passes are full passes (no density screening and no incremental passes: the CPU's
//! `incremental` setting does not apply); every result is bitwise reproducible run to run and
//! agrees with the CPU's to round-off and the CPU's integral screening (~1e-12 Eh). A `predicted`
//! step is incremental (its Fock term is of predicted charges); when the SCF asks for a full
//! build (`EnvironmentTerm::reset_incremental`: its confirmation of a converged iterate), the next
//! step is the exact mode's two passes, so the SCF ends on the exact PCM operator of its density.
//!
//! The SCF sees the term through [`EnvironmentTerm`] on the host loop (the total density up,
//! the Fock matrix down per iteration) and through [`DeviceEnvironment`] on the device loop (no
//! transfers). A device error of an SCF step stops the SCF as the calculation's `backend` error;
//! one of the Coulomb hook (the fit hand-off) is held and returned by the next step.
//!
//! Gradients ([`solvent_gradient`]): the potential and the charges of the final density on the
//! device, the derivative integrals (shell centres and points) by the device's derivative
//! kernels, the O(n^2) parts of the S and D derivative contractions and the products S x and
//! D^T x on the device; the switching factors' and the nuclear potential's terms (O(n natom))
//! on the host. The density-fitted PCM and the Hessians' PCM terms stay on the CPU.

use crate::backend::gpu::device_error;
use crate::backend::gpu::scf_control::DeviceEnvironment;
use crate::backend::gpu::setup as setup_of;
use crate::backend::Setup;
use crate::driver::backend::Family;
use crate::driver::ground::ScfSetup;
use crate::driver::CalcContext;
use crate::solvent::derivatives::SolventGradient;
use crate::solvent::scf::{PcmSetup, ScfEnvironment};
use crate::solvent::SolventSettings;
use faer::{Mat, Par};
use linus_base::error::LinusError;
use linus_base::linalg::{mat_from_colmajor, mat_to_colmajor};
use linus_base::terms::HookOutcome;
use linus_base::terms::{EnvironmentTerm, XcDensity};
use linus_core::timing::Timers;
use linus_gpu::blas::{GemmDims, MatMut, MatRef, Op};
use linus_gpu::linalg::{Arith, Gemm};
use linus_gpu::pcm::solver::{GpuPcmSolver, PcmCharges, PcmKind, PcmSurface};
use linus_gpu::pcm::{GpuPcmInts, PcmIntsHost, PcmPoint};
use linus_gpu::scf::FockState;
use linus_gpu::{DeviceBuf, Gpu, GpuError};
use linus_ints::BasisSet;
use linus_scf::ScfLoop;
use linus_solvent::pcm::{Charges, MatProduct, PcmOperator, SolverTerms};
use linus_solvent::surface::Surface;
use linus_solvent::{PcmModel, PcmScfMode, ResolvedPcm, SolvationResult};
use std::sync::{Arc, Mutex};

/// The `predicted` mode's RI side on the device: M^T = (P|g) as naux x npts (column-major),
/// the J build's coefficients and the previous iteration's (phi, c).
struct DeviceRi {
    naux: usize,
    mt: DeviceBuf<f64>,
    /// The last J build's coefficients per density (naux x nd).
    c: DeviceBuf<f64>,
    nd: usize,
    /// Total coefficients of this and the previous iteration, their difference, M dc, phi_prev.
    c_tot: DeviceBuf<f64>,
    c_prev: DeviceBuf<f64>,
    dc: DeviceBuf<f64>,
    dv: DeviceBuf<f64>,
    phi_prev: DeviceBuf<f64>,
    has_prev: bool,
    /// Coulomb builds seen and steps that consumed them.
    builds: u64,
    consumed: u64,
}

/// The device's PCM term of one SCF (see the module docs).
pub(crate) struct GpuPcmScf {
    pub(crate) resolved: ResolvedPcm,
    /// The surface, f and v_n on the host (K is factored on the device).
    pub(crate) op: PcmOperator,
    mode: PcmScfMode,
    gpu: Arc<Gpu>,
    ints: GpuPcmInts,
    solver: GpuPcmSolver,
    gemm: Gemm,
    nbf: usize,
    npts: usize,
    vnuc: DeviceBuf<f64>,
    phi: DeviceBuf<f64>,
    phi_p: DeviceBuf<f64>,
    ch: PcmCharges,
    chp: PcmCharges,
    /// The total density (open shells, host loop) and the Fock matrix (host loop).
    dtot: Option<DeviceBuf<f64>>,
    fock: Option<DeviceBuf<f64>>,
    ri: Option<DeviceRi>,
    /// The next step is the exact mode's (the SCF asked for a full build: `predicted` mode).
    full_next: bool,
    /// The last step's Fock term was of predicted charges (`predicted` mode).
    last_incremental: bool,
    /// (energy, total surface charge) of the last step.
    pub(crate) last: Option<(f64, f64)>,
    pub(crate) timers: Timers,
    /// A device error of the Coulomb hook (whose trait returns none): returned by the next SCF
    /// step, or after the SCF.
    error: Option<GpuError>,
}

impl std::fmt::Debug for GpuPcmScf {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "GpuPcmScf {{ model: {:?}, npoints: {} }}",
            self.resolved.model, self.npts
        )
    }
}

/// The host side of a PCM term at the geometry: the surface and its operator without K
/// (the CPU's `PcmScf::new` up to the matrices), with the nuclear potential of `z`.
pub(crate) fn surface_operator(
    resolved: &ResolvedPcm,
    z: &[u32],
    positions: &[[f64; 3]],
    timers: &mut Timers,
) -> Result<PcmOperator, LinusError> {
    let surf = timers
        .time("pcm_setup", || {
            Surface::build(
                positions,
                &resolved.radii,
                &resolved.lebedev,
                &resolved.xi,
                resolved.discretization,
                resolved.cut,
            )
        })
        .map_err(LinusError::Input)?;
    let zf: Vec<f64> = z.iter().map(|&x| x as f64).collect();
    Ok(timers.time("pcm_setup", || {
        PcmOperator::surface_only(
            resolved.model,
            resolved.eps,
            resolved.cosmo_x,
            surf,
            resolved.point_potential,
            &zf,
            positions,
        )
    }))
}

impl GpuPcmScf {
    /// The device term for the operator `op` (from [`surface_operator`]) in SCF mode `mode`
    /// (exact or predicted; predicted needs the auxiliary basis `aux` of the density-fitted J),
    /// the integrals packed with their derivatives when `derivatives`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        gpu: Arc<Gpu>,
        resolved: ResolvedPcm,
        op: PcmOperator,
        basis: &BasisSet,
        mode: PcmScfMode,
        aux: Option<&BasisSet>,
        derivatives: bool,
        mut timers: Timers,
    ) -> Result<Self, GpuError> {
        if mode == PcmScfMode::Df {
            return Err(GpuError::NotCovered(
                "the density-fitted PCM runs on the CPU's density-fitted Coulomb build".into(),
            ));
        }
        let t0 = std::time::Instant::now();
        let surf = &op.surf;
        let npts = surf.len();
        let nbf = basis.nbf;
        let pts: Vec<PcmPoint> = surf
            .pts
            .iter()
            .zip(&op.pot_zeta)
            .map(|(p, &z)| PcmPoint { r: p.r, zeta: z })
            .collect();
        // the shell pairs the CPU's SurfacePlan keeps (scf.rs exact_plan)
        let host = PcmIntsHost::new(basis, &pts, resolved.int_thresh * 1e-3, derivatives)?;
        let ints = GpuPcmInts::new(&gpu, &host)?;
        drop(host);
        let kind = match op.model {
            PcmModel::Cpcm | PcmModel::Cosmo => PcmKind::Conductor,
            PcmModel::Ssvpe => PcmKind::Ssvpe,
            PcmModel::Iefpcm | PcmModel::Smd => PcmKind::Ief,
        };
        let ps = PcmSurface {
            r: surf.pts.iter().map(|p| p.r).collect(),
            zeta: surf.pts.iter().map(|p| p.zeta).collect(),
            switch: surf.switch.clone(),
            normal: surf.normal.clone(),
            area: surf.area.clone(),
            radius: (0..npts).map(|i| surf.radius_of(i)).collect(),
        };
        let solver = GpuPcmSolver::new(&gpu, kind, op.f_eps, &ps)?;
        let ri = match (mode, aux) {
            (PcmScfMode::Predicted, Some(aux)) => {
                let gp: Vec<linus_ints::points::pcm_ints::GaussPoint> = pts
                    .iter()
                    .map(|p| linus_ints::points::pcm_ints::GaussPoint {
                        r: p.r,
                        zeta: p.zeta,
                    })
                    .collect();
                // (P|g) row-major npts x naux = M^T column-major naux x npts
                let mut m = vec![0.0; gp.len() * aux.nbf];
                linus_ints::AuxSurfacePlan::new(aux, &gp).matrix(&mut m);
                let naux = aux.nbf;
                Some(DeviceRi {
                    naux,
                    mt: gpu.upload(&m)?,
                    c: gpu.alloc_zeros(2 * naux)?,
                    nd: 0,
                    c_tot: gpu.alloc_zeros(naux)?,
                    c_prev: gpu.alloc_zeros(naux)?,
                    dc: gpu.alloc_zeros(naux)?,
                    dv: gpu.alloc_zeros(npts.max(1))?,
                    phi_prev: gpu.alloc_zeros(npts.max(1))?,
                    has_prev: false,
                    builds: 0,
                    consumed: 0,
                })
            }
            (PcmScfMode::Predicted, None) => {
                return Err(GpuError::Contract(
                    "PCM predicted mode: no auxiliary basis".into(),
                ))
            }
            _ => None,
        };
        let vnuc = gpu.upload(&op.v_nuc)?;
        let s = GpuPcmScf {
            resolved,
            mode,
            gemm: Gemm::new(&gpu)?,
            nbf,
            npts,
            vnuc,
            phi: gpu.alloc_zeros(npts.max(1))?,
            phi_p: gpu.alloc_zeros(npts.max(1))?,
            ch: PcmCharges::new(&gpu, npts)?,
            chp: PcmCharges::new(&gpu, npts)?,
            dtot: None,
            fock: None,
            ri,
            full_next: false,
            last_incremental: false,
            last: None,
            error: None,
            ints,
            solver,
            op,
            gpu,
            timers: {
                timers.add("pcm_setup", t0.elapsed().as_secs_f64());
                timers
            },
        };
        Ok(s)
    }

    /// Device bytes of the term (integrals' data and work, the solver's matrices).
    pub(crate) fn device_bytes(&self) -> usize {
        self.ints.bytes.data + self.ints.bytes.work + self.solver.bytes
    }

    /// 1/2 q_sym . phi of the step's charges (host, in order) and the total charge.
    fn energy(&self) -> Result<(f64, f64), GpuError> {
        let n = self.npts;
        let phi = self.gpu.download_range(&self.phi, 0, n)?;
        let q = self.gpu.download_range(&self.ch.q_sym, 0, n)?;
        let e = 0.5 * q.iter().zip(&phi).map(|(x, y)| x * y).sum::<f64>();
        Ok((e, q.iter().sum()))
    }

    /// One SCF step at the total density s D (`d` nbf x nbf, symmetric): the energy, the Fock
    /// matrix -sum_g q_g (mn|g) into `out` (see the module docs for the modes).
    fn step(
        &mut self,
        d: &DeviceBuf<f64>,
        s: f64,
        out: &mut DeviceBuf<f64>,
    ) -> Result<f64, GpuError> {
        let t0 = std::time::Instant::now();
        let gpu = self.gpu.clone();
        let (n, nbf) = (self.npts, self.nbf);
        gpu.zero_range(out, 0, nbf * nbf)?;
        // the predicted mode's step after the SCF asked for a full build is the exact mode's
        let predicted = self.mode == PcmScfMode::Predicted && !std::mem::take(&mut self.full_next);
        if let Some(ri) = self.ri.as_mut() {
            if ri.builds != ri.consumed + 1 {
                return Err(GpuError::Contract(
                    "PCM: the SCF evaluated the solvent term without a Coulomb build before it"
                        .into(),
                ));
            }
            ri.consumed = ri.builds;
            let naux = ri.naux;
            // total coefficients: 2 c (a closed shell's one channel) or c_a + c_b
            if ri.nd == 1 {
                self.ints
                    .axpby(&gpu, naux, 2.0, &ri.c, 0.0, None, &mut ri.c_tot)?;
            } else {
                let mut cb = gpu.alloc_zeros::<f64>(naux)?;
                gpu.copy_range(&ri.c, naux, &mut cb, 0, naux)?;
                self.ints
                    .axpby(&gpu, naux, 1.0, &ri.c, 1.0, Some(&cb), &mut ri.c_tot)?;
            }
        }
        if predicted {
            let ri = self.ri.as_mut().expect("RI data");
            let naux = ri.naux;
            // phi~ = phi_prev - M (c - c_prev), or v_n - M c
            if ri.has_prev {
                self.ints.axpby(
                    &gpu,
                    naux,
                    1.0,
                    &ri.c_tot,
                    -1.0,
                    Some(&ri.c_prev),
                    &mut ri.dc,
                )?;
                self.gemm.gemm(
                    Arith::Fp64,
                    GemmDims::new(Op::T, Op::N, n, 1, naux),
                    (1.0, 0.0),
                    MatRef::new(&ri.mt, 0, naux),
                    MatRef::new(&ri.dc, 0, naux),
                    MatMut::new(&mut ri.dv, 0, n),
                )?;
                self.ints.axpby(
                    &gpu,
                    n,
                    1.0,
                    &ri.phi_prev,
                    -1.0,
                    Some(&ri.dv),
                    &mut self.phi_p,
                )?;
            } else {
                self.gemm.gemm(
                    Arith::Fp64,
                    GemmDims::new(Op::T, Op::N, n, 1, naux),
                    (1.0, 0.0),
                    MatRef::new(&ri.mt, 0, naux),
                    MatRef::new(&ri.c_tot, 0, naux),
                    MatMut::new(&mut ri.dv, 0, n),
                )?;
                self.ints.axpby(
                    &gpu,
                    n,
                    1.0,
                    &self.vnuc,
                    -1.0,
                    Some(&ri.dv),
                    &mut self.phi_p,
                )?;
            }
            self.solver.solve(&gpu, &self.phi_p, &mut self.chp)?;
            // one fused pass: the exact potential and the predicted charges' Fock matrix
            self.ints.pass(
                &gpu,
                Some((d, -s)),
                Some(&self.vnuc),
                Some(&mut self.phi),
                Some(&self.chp.q_sym),
                Some((out, -1.0)),
            )?;
            self.solver.solve(&gpu, &self.phi, &mut self.ch)?;
        } else {
            self.ints.pass(
                &gpu,
                Some((d, -s)),
                Some(&self.vnuc),
                Some(&mut self.phi),
                None,
                None,
            )?;
            self.solver.solve(&gpu, &self.phi, &mut self.ch)?;
            self.ints.pass(
                &gpu,
                None,
                None,
                None,
                Some(&self.ch.q_sym),
                Some((out, -1.0)),
            )?;
        }
        // the next prediction starts from this step's exact potential and fit
        if let Some(ri) = self.ri.as_mut() {
            gpu.copy_range(&self.phi, 0, &mut ri.phi_prev, 0, n)?;
            gpu.copy_range(&ri.c_tot, 0, &mut ri.c_prev, 0, ri.naux)?;
            ri.has_prev = true;
        }
        self.last_incremental = predicted;
        let (e, qsum) = self.energy()?;
        self.last = Some((e, qsum));
        self.timers.add("pcm_step", t0.elapsed().as_secs_f64());
        Ok(e)
    }

    /// The host loop's step: the spin densities up, the Fock matrix down.
    fn host_step(&mut self, d: &XcDensity<'_>) -> Result<(f64, Mat<f64>), GpuError> {
        let n = self.nbf;
        let dt = d.da + d.db;
        let t = std::time::Instant::now();
        let dh = mat_to_colmajor(dt.as_ref());
        let mut dd = match self.dtot.take() {
            Some(b) => b,
            None => self.gpu.alloc_zeros(n * n)?,
        };
        self.gpu.upload_into(&mut dd, 0, &dh)?;
        let mut f = match self.fock.take() {
            Some(b) => b,
            None => self.gpu.alloc_zeros(n * n)?,
        };
        self.timers.add("pcm_transfer", t.elapsed().as_secs_f64());
        let r = self.step(&dd, 1.0, &mut f);
        let t = std::time::Instant::now();
        let v = r.and_then(|e| Ok((e, self.gpu.download_range(&f, 0, n * n)?)));
        self.timers.add("pcm_transfer", t.elapsed().as_secs_f64());
        self.dtot = Some(dd);
        self.fock = Some(f);
        let (e, v) = v?;
        Ok((e, mat_from_colmajor(n, n, &v)))
    }
}

/// The device term as the SCF's environment term (see the module docs).
pub(crate) struct GpuPcmTerm {
    pub pcm: Arc<Mutex<GpuPcmScf>>,
}

impl EnvironmentTerm for GpuPcmTerm {
    fn name(&self) -> &'static str {
        "solvent"
    }

    fn compute(
        &mut self,
        d: &XcDensity<'_>,
        _par: Par,
    ) -> Result<(f64, Option<Mat<f64>>), LinusError> {
        let mut pcm = self.pcm.lock().expect("PCM state");
        // a failed fit hand-off (the Coulomb hook) first, then this step's
        let r = match pcm.error.take() {
            Some(e) => Err(e),
            None => pcm.host_step(d),
        };
        r.map(|(e, v)| (e, Some(v)))
            .map_err(|e| device_error(Family::Pcm, e))
    }

    fn device_form(&mut self) -> Option<&mut dyn std::any::Any> {
        Some(self)
    }

    /// A `predicted` step's Fock term is of predicted charges (module docs).
    fn built_incrementally(&self) -> bool {
        self.pcm.lock().expect("PCM state").last_incremental
    }

    /// The next `predicted` step is the exact mode's (the exact mode always is).
    fn reset_incremental(&mut self) {
        let mut pcm = self.pcm.lock().expect("PCM state");
        pcm.full_next = pcm.mode == PcmScfMode::Predicted;
    }
}

impl DeviceEnvironment for GpuPcmTerm {
    fn compute_device(
        &mut self,
        gpu: &Gpu,
        st: &FockState<'_>,
        v: &mut DeviceBuf<f64>,
    ) -> Result<f64, GpuError> {
        let mut pcm = self.pcm.lock().expect("PCM state");
        // a failed fit hand-off (the Coulomb hook) stops the loop here
        if let Some(e) = pcm.error.take() {
            return Err(e);
        }
        let n2 = st.nbf * st.nbf;
        if st.nspin() == 1 {
            return pcm.step(st.densities, 2.0, v);
        }
        let mut dd = match pcm.dtot.take() {
            Some(b) => b,
            None => gpu.alloc_zeros(n2)?,
        };
        let mut db = gpu.alloc_zeros::<f64>(n2)?;
        gpu.copy_range(st.densities, n2, &mut db, 0, n2)?;
        pcm.ints
            .axpby(gpu, n2, 1.0, st.densities, 1.0, Some(&db), &mut dd)?;
        let r = pcm.step(&dd, 1.0, v);
        pcm.dtot = Some(dd);
        r
    }
}

/// The `predicted` mode's Coulomb hook: the fit coefficients of every J build, on the device
/// ([`GpuPcmHook::device_coefficients`], from the device's J build,
/// `crate::backend::gpu::df::CoulombFit`) or from the host.
pub(crate) struct GpuPcmHook(Arc<Mutex<GpuPcmScf>>);

impl GpuPcmHook {
    fn take(&self, nd: usize, put: impl FnOnce(&Gpu, &mut DeviceBuf<f64>) -> Result<(), GpuError>) {
        let mut guard = self.0.lock().expect("PCM state");
        let pcm = &mut *guard;
        let gpu = pcm.gpu.clone();
        let ri = pcm.ri.as_mut().expect("RI data");
        let r = (|| {
            if ri.c.len() < ri.naux * nd {
                ri.c = gpu.alloc_zeros(ri.naux * nd)?;
            }
            put(&gpu, &mut ri.c)
        })();
        ri.nd = nd;
        ri.builds += 1;
        if let Err(e) = r {
            pcm.error.get_or_insert(e);
        }
    }

    /// A device J build's fit coefficients c = V^-1 gamma of its `nd` densities (naux x nd,
    /// column-major, on `gpu`), copied on the device.
    pub(crate) fn device_coefficients(&mut self, gpu: &Gpu, c: &DeviceBuf<f64>, nd: usize) {
        let n = self
            .0
            .lock()
            .expect("PCM state")
            .ri
            .as_ref()
            .map_or(0, |r| r.naux);
        self.take(nd, |_, buf| gpu.copy_range(c, 0, buf, 0, n * nd));
    }
}

impl linus_base::terms::CoulombHook for GpuPcmHook {
    fn extra_coefficients(&mut self, c: &[Vec<f64>], _solve: &dyn Fn(&mut [f64])) -> Vec<Vec<f64>> {
        let flat: Vec<f64> = c.iter().flatten().copied().collect();
        self.take(c.len(), |gpu, buf| gpu.upload_into(buf, 0, &flat));
        Vec::new()
    }

    /// It takes a device J build's coefficients on the device
    /// ([`GpuPcmHook::device_coefficients`]).
    fn device_form(&mut self) -> Option<&mut dyn std::any::Any> {
        Some(self)
    }
}

/// The device's PCM in one SCF (family `pcm` of the GPU's provider, a `crate::solvent::scf::ScfEnvironment`): its
/// term, and the SCF mode of its surface terms.
pub(crate) struct GpuScfEnvironment {
    term: GpuPcmTerm,
    mode: PcmScfMode,
}

impl GpuScfEnvironment {
    /// The device's term for an SCF of `calc` on the loop `scf` (the solvent `pcm`: resolved, its
    /// surface terms' mode and the J build's auxiliary basis), its Coulomb hook installed;
    /// declined (the CPU's term) when the device setup does not fit or, in the `predicted` mode,
    /// the J build cannot hand its fit to the device.
    pub(crate) fn open(
        calc: &CalcContext,
        gpu: &Arc<Gpu>,
        setup: &ScfSetup,
        scf: &mut dyn ScfLoop,
        pcm: &PcmSetup,
    ) -> Result<Setup<Box<dyn ScfEnvironment>>, LinusError> {
        let (resolved, mode) = (&pcm.resolved, pcm.mode);
        let mut timers = Timers::default();
        let op = surface_operator(
            resolved,
            &calc.mol.effective_z(),
            &calc.mol.positions,
            &mut timers,
        )?;
        // the gradient after this SCF reuses the term: pack the derivatives now
        let derivatives = setup.purpose == crate::driver::Purpose::Gradient;
        let built = GpuPcmScf::new(
            gpu.clone(),
            resolved.clone(),
            op,
            &calc.basis,
            mode,
            pcm.aux.as_ref(),
            derivatives,
            timers,
        );
        let state = match setup_of(Family::Pcm, built)? {
            Setup::Ready(s) => s,
            Setup::Declined(why) => return Ok(Setup::Declined(why)),
        };
        calc.note_grant(Family::Pcm, &state.solver.plan_violations);
        let shared = Arc::new(Mutex::new(state));
        if mode == PcmScfMode::Predicted {
            let hook = Box::new(GpuPcmHook(shared.clone()));
            if let HookOutcome::FellBack { reason } = scf.set_coulomb_hook(Some(hook))? {
                // the predicted charges come from the J build's fit; a J builder that cannot hand it
                // over leaves the surface terms to the CPU's path, with the reason
                return Ok(Setup::Declined(format!(
                    "solvent.scf = {}: the SCF's J build could not hand its fit to the device's \
                     surface terms ({reason}); they run on the CPU",
                    mode.name()
                )));
            }
        }
        Ok(Setup::Ready(Box::new(GpuScfEnvironment {
            term: GpuPcmTerm { pcm: shared },
            mode,
        })))
    }
}

impl ScfEnvironment for GpuScfEnvironment {
    fn term(&mut self) -> &mut dyn EnvironmentTerm {
        &mut self.term
    }

    /// After the loop's SCF `r`: the Coulomb hook removed (also when the SCF failed, whose error is
    /// then returned), the device's error of a step, the solvation parts of the result and the
    /// summary; the term is kept for the gradient at the same geometry.
    fn close(
        self: Box<Self>,
        calc: &CalcContext,
        setup: &ScfSetup,
        scf: &mut dyn ScfLoop,
        r: Result<linus_scf::ScfResult, LinusError>,
    ) -> Result<(linus_scf::ScfResult, SolvationResult), LinusError> {
        if self.mode == PcmScfMode::Predicted {
            scf.set_coulomb_hook(None)?;
        }
        let mut r = r?;
        let (so, mode, shared) = (
            calc.plan.solvent().expect("a solvent"),
            self.mode,
            self.term.pcm,
        );
        let mut pcm = shared.lock().expect("PCM state");
        if let Some(e) = pcm.error.take() {
            return Err(device_error(Family::Pcm, e));
        }
        let last = pcm.last.unwrap_or((0.0, 0.0));
        let mut res = linus_solvent::scf::summary(
            &pcm.resolved,
            &pcm.op,
            mode,
            last,
            &pcm.timers,
            (so.name(), so.preset(), calc.mol.charge),
        );
        if let Some(d) = &pcm.resolved.smd {
            let c = linus_solvent::smd::cds(
                &calc.mol.z,
                &calc.mol.positions,
                d,
                &crate::solvent::scf::cds_config(
                    so,
                    setup.purpose,
                    linus_solvent::smd::CdsOrder::Energy,
                ),
            );
            r.energy += c.energy;
            r.e_solvation += c.energy;
            res.cds = Some(c.energy);
        }
        r.timers.merge(&pcm.timers);
        r.timers
            .resource("pcm_device_mib", pcm.device_bytes() as f64 / 1048576.0);
        pcm.timers = Timers::default();
        pcm.last = None;
        drop(pcm);
        // the term for the PCM gradient of this SCF (crate::backend::gpu::session)
        if let Some(s) = calc.device_session() {
            s.keep_pcm(shared);
        }
        Ok((r, res))
    }
}

/// The fixed-density nuclear gradient of the PCM energy on the device (the `pcm` family's
/// gradient): the SCF's term when it ran on the device (the device session's), else one set up
/// for the gradient, its derivative integrals packed (see the module docs).
pub(crate) struct GpuSolventGradient<'a> {
    calc: &'a CalcContext,
    gpu: Arc<Gpu>,
    shared: Arc<Mutex<GpuPcmScf>>,
}

impl<'a> GpuSolventGradient<'a> {
    /// The gradient's term of the solvent `so`, declined (the CPU's) when a setup does not fit;
    /// timed as `grad_solvent_setup`.
    pub(crate) fn new(
        calc: &'a CalcContext,
        gpu: Arc<Gpu>,
        so: &SolventSettings,
        timers: &mut Timers,
    ) -> Result<Setup<Self>, LinusError> {
        // the SCF's term, when it ran on the device (crate::backend::gpu::session)
        let shared = match calc.device_session().and_then(|s| s.pcm()) {
            Some(s) => s,
            None => {
                let resolved = so.resolve(&calc.mol.z).map_err(LinusError::Input)?;
                let mut t = Timers::default();
                let op = surface_operator(
                    &resolved,
                    &calc.mol.effective_z(),
                    &calc.mol.positions,
                    &mut t,
                )?;
                let built = timers.time("grad_solvent_setup", || {
                    GpuPcmScf::new(
                        gpu.clone(),
                        resolved,
                        op,
                        &calc.basis,
                        PcmScfMode::Exact,
                        None,
                        true,
                        t,
                    )
                });
                match setup_of(Family::Pcm, built)? {
                    Setup::Ready(p) => {
                        calc.note_grant(Family::Pcm, &p.solver.plan_violations);
                        Arc::new(Mutex::new(p))
                    }
                    Setup::Declined(why) => return Ok(Setup::Declined(why)),
                }
            }
        };
        {
            let mut pcm = shared.lock().expect("PCM state");
            // the SCF's integrals without derivatives: pack them once with
            if !pcm.ints.has_derivatives() {
                let t = std::time::Instant::now();
                let pts: Vec<PcmPoint> = pcm
                    .op
                    .surf
                    .pts
                    .iter()
                    .zip(&pcm.op.pot_zeta)
                    .map(|(p, &z)| PcmPoint { r: p.r, zeta: z })
                    .collect();
                let host =
                    PcmIntsHost::new(&calc.basis, &pts, pcm.resolved.int_thresh * 1e-3, true)
                        .map_err(|e| device_error(Family::Pcm, e))?;
                let built = GpuPcmInts::new(&gpu, &host);
                match setup_of(Family::Pcm, built)? {
                    Setup::Ready(i) => pcm.ints = i,
                    Setup::Declined(why) => return Ok(Setup::Declined(why)),
                }
                timers.add("grad_solvent_setup", t.elapsed().as_secs_f64());
            }
        }
        Ok(Setup::Ready(GpuSolventGradient { calc, gpu, shared }))
    }
}

impl SolventGradient for GpuSolventGradient<'_> {
    /// The potential and the charges of the total density `d` (row-major nbf^2) on the device,
    /// then the derivative contractions.
    fn gradient(&mut self, d: &[f64], timers: &mut Timers) -> Result<Vec<[f64; 3]>, LinusError> {
        let (calc, gpu) = (self.calc, &*self.gpu);
        let mut pcm = self.shared.lock().expect("PCM state");
        let dev = |e| device_error(Family::Pcm, e);
        let np = pcm.npts;
        let t = std::time::Instant::now();
        let dd = gpu.upload(d).map_err(dev)?;
        let pcm = &mut *pcm;
        pcm.ints
            .pass(
                gpu,
                Some((&dd, -1.0)),
                Some(&pcm.vnuc),
                Some(&mut pcm.phi),
                None,
                None,
            )
            .map_err(dev)?;
        pcm.solver.solve(gpu, &pcm.phi, &mut pcm.ch).map_err(dev)?;
        let down = |b: &DeviceBuf<f64>| gpu.download_range(b, 0, np).map_err(dev);
        let phi = down(&pcm.phi)?;
        let c = Charges {
            q: down(&pcm.ch.q)?,
            y: down(&pcm.ch.y)?,
            q_sym: down(&pcm.ch.q_sym)?,
        };
        timers.add("grad_solvent_potential", t.elapsed().as_secs_f64());
        let (ga, gc) = timers
            .time("grad_solvent_ints", || {
                pcm.ints.gradient(gpu, &dd, 1.0, &c.q_sym)
            })
            .map_err(dev)?;
        let positions = &calc.mol.positions;
        let zf: Vec<f64> = calc.mol.effective_z().iter().map(|&x| x as f64).collect();
        let op = &pcm.op;
        let terms = DeviceSolverTerms {
            gpu,
            solver: &pcm.solver,
            np,
            err: std::cell::RefCell::new(None),
        };
        let mut g = timers.time("grad_solvent_surface", || {
            let mut g = op.nuclear_gradient(&zf, positions, &c.q_sym);
            let gs = op.solver_gradient_by(positions, &phi, &c, &terms);
            for (a, b) in g.iter_mut().zip(&gs) {
                for k in 0..3 {
                    a[k] += b[k];
                }
            }
            g
        });
        if let Some(e) = terms.err.into_inner() {
            return Err(dev(e));
        }
        for (a, v) in g.iter_mut().zip(&ga) {
            for k in 0..3 {
                a[k] -= v[k];
            }
        }
        for (i, v) in gc.iter().enumerate() {
            let o = op.surf.owner[i];
            for k in 0..3 {
                g[o][k] -= v[k];
            }
        }
        Ok(g)
    }
}

/// The solver gradient's products and O(n^2) contractions on the device
/// ([`PcmOperator::solver_gradient_by`]); the first device error is kept (zeros returned).
struct DeviceSolverTerms<'a> {
    gpu: &'a Gpu,
    solver: &'a GpuPcmSolver,
    np: usize,
    err: std::cell::RefCell<Option<GpuError>>,
}

impl DeviceSolverTerms<'_> {
    fn keep<T>(&self, r: Result<T, GpuError>, zero: T) -> T {
        r.unwrap_or_else(|e| {
            self.err.borrow_mut().get_or_insert(e);
            zero
        })
    }
}

impl SolverTerms for DeviceSolverTerms<'_> {
    fn product(&self, m: MatProduct, x: &[f64]) -> Vec<f64> {
        let (gpu, np) = (self.gpu, self.np);
        let r = (|| {
            let xd = gpu.upload(x)?;
            let mut out = gpu.alloc_zeros::<f64>(np.max(1))?;
            let dt = m == MatProduct::Dt;
            self.solver.matvec(dt, dt, &xd, &mut out)?;
            gpu.download_range(&out, 0, np)
        })();
        self.keep(r, vec![0.0; np])
    }

    fn ds_points(&self, a: &[f64], b: &[f64]) -> Vec<[f64; 3]> {
        let r = self.solver.contraction_points(self.gpu, false, a, b);
        self.keep(r, vec![[0.0; 3]; self.np])
    }

    fn dd_points(&self, a: &[f64], b: &[f64]) -> Vec<[f64; 3]> {
        let r = self.solver.contraction_points(self.gpu, true, a, b);
        self.keep(r, vec![[0.0; 3]; self.np])
    }
}
