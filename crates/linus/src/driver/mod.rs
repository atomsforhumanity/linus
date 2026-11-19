//! High-level driver: a `Session` owns the molecule, basis, J/K and XC machinery and
//! its own thread pool (no global or environment-controlled threading). The settings and
//! their grammar are in [`crate::driver::options`] (re-exported here); the structured result of a
//! calculation is built by [`crate::driver::report`]; the J/K method's builders and derivatives are
//! chosen in [`crate::two_electron`]; the backend (CPU, or a GPU with per-kernel fallbacks) is
//! resolved in [`crate::driver::backend`].
//!
//! The session's SCF stage (`Session::scf_stage`) leaves a ground state
//! (`ground::GroundState`: the SCF at one geometry and what it was set up with), which it
//! hands explicitly to the gradient, the Hessian, the polarizability and the report. A session's
//! requests ([`session::Request`], `Session::compute`) run in lineages (`lineage`: the warm
//! start and the probes' state from one request of a chain to the next), each committed only
//! when a request succeeds.

pub mod accuracy;
pub mod backend;
pub(crate) mod ground;
pub mod lineage;
pub(crate) mod memory;
pub mod molden;
pub(crate) mod needs;
pub mod options;
pub(crate) mod plan;
pub(crate) mod policy;
pub(crate) mod record;
pub mod report;
pub mod request;
pub mod schema;
pub mod session;
pub use lineage::{GuessRecord, GuessSource, LineageKey, GUESS_DISTANCE};
pub use plan::settings;

use crate::backend::scf::{ScfRun, ScfTerms};
use crate::driver::ground::{ScfMemory, ScfSetup};
use crate::structure::Structure;
use faer::{Mat, MatRef, Par};
use linus_base::linalg::{ao_view_mut, mm};
use linus_core::timing::Timers;
use linus_dft::engine::XcKernel;
use linus_dft::{XcFunctional, XcIntegrator, XcSetup};
use linus_ints::basis::BasisDef;
use linus_ints::one_e::ecp::EcpSet;
use linus_ints::operator::Operator;
use linus_ints::{BasisSet, DerivOrder, EcpPlan, OneEPlan, PairSet};
use linus_scf::{ScfConfig, ScfInput, ScfResult};
use std::sync::Arc;

pub use crate::driver::options::*;
pub use linus_base::error::{available_functionals, LinusError, ResponseFailure, ScfFailure};
pub use linus_ints::basis::available_basis_sets;

/// What the method is, resolved once from the options ([`Session::new`]): the XC functional
/// (parsed and checked here, and shared by every XC integrator of the calculation), the exact
/// exchange, VV10, the XC derivative order and the dispersion model.
#[derive(Clone, Debug)]
pub struct MethodInfo {
    /// The XC functional of Kohn-Sham DFT (None: Hartree-Fock).
    pub functional: Option<XcFunctional>,
    /// The exact-exchange operator (None: none, semilocal functionals): 1/r for HF, a global
    /// hybrid's fraction a_x as `Operator::Attenuated { coulomb: a_x, long_range: 0, omega: 0 }`,
    /// a range-separated hybrid's operator ([`Operator::from_libxc`]).
    /// [`Operator::global_fraction`] tells the global ones apart.
    pub exchange: Option<Operator>,
    /// VV10 nonlocal correlation (-V functionals).
    pub vv10: bool,
    /// Highest XC derivative order libxc provides (2: kernels for responses; HF: unbounded).
    pub max_order: usize,
    /// The empirical dispersion model (requested or implied by the name).
    pub dispersion: Option<Dispersion>,
}

impl MethodInfo {
    /// Resolve `opts.method` (and `opts.dispersion`): what linus does not implement for a
    /// functional (Yukawa range separation, Laplacian meta-GGAs, energy-only functionals) is
    /// [`LinusError::Unsupported`].
    pub fn resolve(opts: &CalcOptions) -> Result<MethodInfo, LinusError> {
        let name = opts.method();
        if name.trim().eq_ignore_ascii_case("hf") {
            return Ok(MethodInfo {
                functional: None,
                exchange: Some(Operator::Coulomb),
                vv10: false,
                max_order: usize::MAX,
                dispersion: crate::dispersion::requested(opts, None)?,
            });
        }
        let xc = XcFunctional::new(&name)?;
        let f = xc.functional();
        let unsupported = if f.range_separation().is_some()
            && f.exact_exchange().kernel != linus_xc::RangeSeparation::Erf
        {
            Some("range separation with a Yukawa (non-erf) kernel")
        } else if f.needs_lapl() {
            Some("Laplacian-dependent meta-GGAs")
        } else if f.max_order() < 1 {
            Some("functionals without an XC potential in libxc (energy-only)")
        } else {
            None
        };
        if let Some(what) = unsupported {
            return Err(LinusError::Unsupported(format!(
                "{}: {what} not implemented",
                name
            )));
        }
        Ok(MethodInfo {
            exchange: xc.exchange(),
            vv10: f.vv10().is_some(),
            max_order: f.max_order(),
            dispersion: crate::dispersion::requested(opts, Some(f))?,
            functional: Some(xc),
        })
    }

    /// Kohn-Sham DFT (false: Hartree-Fock).
    pub fn is_dft(&self) -> bool {
        self.functional.is_some()
    }

    /// The range-separation parameter omega of a range-separated hybrid.
    pub fn omega(&self) -> Option<f64> {
        self.exchange
            .filter(|op| op.global_fraction().is_none())
            .map(|op| op.omega())
    }
}

/// A response calculation whose CPHF/CPKS stopped above its tolerance, with the result of the
/// unconverged solution ([`LinusError::ResponseNotConverged`]).
#[derive(Clone, Debug)]
pub enum UnconvergedResponse {
    Hessian(Box<crate::derivatives::hessian::HessianResult>),
    Polarizability(Box<crate::properties::polarizability::Polarizability>),
}

/// A session: one molecule, model and settings at many geometries (decision D46). It holds what
/// it computes with (its `CalcContext`) and the state it carries from one calculation to the next
/// (the last ground state, the lineages, the failures, the last call's timers). Its methods
/// orchestrate: an SCF stage, then the derivatives and properties of its ground state on the
/// context, then the report.
pub struct Session {
    /// What the session computes with.
    pub(crate) cx: CalcContext,
    /// The last converged SCF ([`ground::GroundState`]: its result, what it was set up with and
    /// its side results), possibly at other coordinates.
    pub(crate) ground: Option<Arc<ground::GroundState>>,
    /// The last SCF that did not converge, for its failure report (not used as a guess).
    pub(crate) unconverged: Option<Arc<ground::GroundState>>,
    /// The session's lineages by key ([`lineage::Lineage`]: the warm start and the probes'
    /// state), each committed when a request of it succeeds.
    pub(crate) lineages: std::collections::BTreeMap<lineage::LineageKey, lineage::Lineage>,
    /// The request being run ([`Session::compute`]; None for the direct calls).
    pub(crate) call: Option<session::Call>,
    /// What `set_guess_orbitals` or `set_guess_density` gave the default lineage's next SCF.
    pub(crate) given: session::Given,
    /// The result of the last CPHF/CPKS that did not converge
    /// ([`LinusError::ResponseNotConverged`]; cleared by the next converged one).
    pub(crate) unconverged_response: Option<UnconvergedResponse>,
    /// Phase timers (and resources) of the last call ([`Session::timers`]); derivative calls
    /// merge theirs in through [`crate::driver::report`].
    pub(crate) timers: Timers,
}

/// What a session computes with, the engine of a session: the structure at the current
/// geometry with its basis sets and ECPs placed there, the method, the plan, the pool, the
/// backend and the run record, and the setups of the current geometry (the integral context and,
/// on a device, the device session). The SCF stage, the derivatives, the properties and the
/// device adapters run on it; they see nothing of the session's state (the last ground state, the
/// lineage, the failures), which only the [`Session`] holds and hands them explicitly.
pub(crate) struct CalcContext {
    pub(crate) mol: Structure,
    /// The plan: the one reader of the settings ([`plan`]).
    pub(crate) plan: plan::Plan,
    /// The calctypes the session declared ([`Session::declare`]; all by default): the given
    /// settings are checked against them (decision D41).
    pub(crate) calctypes: settings::CalcTypes,
    pub(crate) basis_def: BasisDef,
    pub(crate) basis: BasisSet,
    /// Effective core potentials of the basis set (empty without ECP atoms).
    pub(crate) ecp: EcpSet,
    pub(crate) pool: rayon::ThreadPool,
    pub(crate) method: MethodInfo,
    /// The setups of the current geometry the SCF, the derivatives and the properties share
    /// ([`crate::context::IntegralContext`]: pair sets, S, XC integrators), replaced when the
    /// atoms move.
    pub(crate) integrals: crate::context::IntegralContext,
    /// The backend: the device of a GPU backend.
    pub(crate) backend: backend::BackendState,
    /// What the session decided at run time (the device setups and their fallbacks, the SCF's
    /// probes) and the warnings of the current call ([`record::Recorder`]).
    pub(crate) record: record::Recorder,
    /// Where the calculation's progress goes: the SCF's progress lines with `output.verbose`
    /// (stderr, or collected for the caller: [`Session::collect_progress`],
    /// [`Session::take_logs`]) and the events of a listener ([`Session::listen`]): the stages,
    /// the SCF's and the responses' iterations.
    pub(crate) progress: linus_base::progress::Progress,
    /// The kind of process the session runs in, for the baseline of every phase's memory need
    /// ([`Session::set_process`]).
    pub(crate) process: linus_base::memory::Process,
}

impl CalcContext {
    /// The parallelism of the products (the plan's threads).
    pub fn par(&self) -> Par {
        if self.plan.threads() > 1 {
            Par::rayon(self.plan.threads())
        } else {
            Par::Seq
        }
    }

    /// Kohn-Sham DFT (false: Hartree-Fock).
    pub fn is_dft(&self) -> bool {
        self.method.is_dft()
    }
}

impl CalcContext {
    /// Why the closed-shell probe is off for this SCF: `scf.stability_closed_probe` 0, Hartree-Fock
    /// with it unset, or a later geometry with it unset.
    pub(crate) fn closed_shell_probe_off(&self) -> linus_scf::ProbeSkip {
        use linus_scf::ProbeSkip;
        match self.plan.scf_algorithm().closed_shell_probe {
            Some(_) => ProbeSkip::Off,
            None if !self.is_dft() => ProbeSkip::HfDefault,
            None => ProbeSkip::LaterGeometry,
        }
    }

    /// Densities of occupied orbitals made orthonormal in the current overlap metric (None if
    /// they are linearly dependent here).
    fn projected_guess(
        &self,
        c_a: MatRef<'_, f64>,
        c_b: MatRef<'_, f64>,
        restricted: bool,
    ) -> Option<(Mat<f64>, Mat<f64>)> {
        let par = self.par();
        self.pool.install(|| {
            let s = self.integrals.overlap(&self.basis);
            let da = lowdin_density(c_a, s.as_ref(), par)?;
            let db = if restricted {
                da.clone()
            } else {
                lowdin_density(c_b, s.as_ref(), par)?
            };
            Some((da, db))
        })
    }

    /// The auxiliary basis `aux` of `jk` for an SCF converged for `purpose` ("auto" resolved by
    /// the purpose; composites "a|b" merged).
    pub fn aux_basis(&self, aux: &AuxBasis, purpose: Purpose) -> Result<BasisDef, LinusError> {
        self.plan.load_basis(aux.for_purpose(purpose))
    }

    /// The auxiliary basis set `aux` for `purpose` placed on the atoms, checked (every element
    /// present, angular momentum at most [`MAX_L_AUX`]): the geometry's
    /// ([`crate::context::IntegralContext::aux`]), loaded and placed once for all its phases.
    pub fn aux_set(&self, aux: &AuxBasis, purpose: Purpose) -> Result<Arc<BasisSet>, LinusError> {
        let name = aux.for_purpose(purpose);
        self.integrals.aux(name, || {
            let adef = self.plan.load_basis(name)?;
            let b = BasisSet::new(&adef, &self.mol.z, &self.mol.positions, true)?;
            if b.max_l() > MAX_L_AUX {
                return Err(LinusError::Unsupported(format!(
                    "auxiliary basis {}: angular momentum l = {} (at most {MAX_L_AUX} is supported)",
                    adef.name,
                    b.max_l()
                )));
            }
            Ok(b)
        })
    }

    /// The XC integrator of the method on `grid` with VV10 on the NLC grid (None for HF), shared
    /// by the phases of this geometry ([`crate::context::IntegralContext::xc`]): the SCF's, the
    /// gradient's, the Hessian's and the polarizability's on the SCF grid are one integrator,
    /// built once (timed as "grid" and "nlc_grid" where it is built).
    pub(crate) fn xc_on(
        &self,
        grid: &GridChoice,
        timers: &mut Timers,
    ) -> Result<Option<Arc<XcIntegrator>>, LinusError> {
        let Some(setup) = self.xc_setup(grid, true) else {
            return Ok(None);
        };
        self.integrals
            .xc(&setup.grid, setup.nlc_grid.as_ref(), || {
                XcIntegrator::build(
                    &self.basis,
                    &self.mol.z,
                    &self.mol.positions,
                    &setup,
                    timers,
                )
            })
            .map(Some)
    }

    /// A new XC integrator of the method on `grid` (None for HF), from the one XC factory
    /// ([`XcIntegrator::build`]), with VV10 (for -V functionals) only when `nlc`: the integrator
    /// of a Hessian's CPKS kernel on its own grid (which takes VV10 from the main integrator),
    /// alive only through its response.
    pub(crate) fn build_xc_on(
        &self,
        grid: &GridChoice,
        nlc: bool,
        timers: &mut Timers,
    ) -> Result<Option<XcIntegrator>, LinusError> {
        let Some(setup) = self.xc_setup(grid, nlc) else {
            return Ok(None);
        };
        XcIntegrator::build(
            &self.basis,
            &self.mol.z,
            &self.mol.positions,
            &setup,
            timers,
        )
        .map(Some)
    }

    /// What the method's XC integrator on `grid` is built with (None for HF): VV10 (for -V
    /// functionals) on the NLC grid when `nlc`.
    fn xc_setup(&self, grid: &GridChoice, nlc: bool) -> Option<XcSetup<'_>> {
        Some(XcSetup {
            functional: self.method.functional.as_ref()?,
            grid: grid.spec(),
            nlc_grid: (self.method.vv10 && nlc).then(|| self.plan.nlc_grid().spec()),
            kernel_budget_mb: self.plan.memory().xc_kernel_mb(),
        })
    }

    /// The derivatives of the last SCF are done with the density-fitting data its device setup
    /// left them (the backend's [`crate::backend::Backend::release_handoff`]).
    pub(crate) fn release_handoff(&self) {
        self.backend.release_handoff();
    }

    /// The XC kernel of a response for the occupied orbitals `occ` (one matrix per spin
    /// channel) from the CPU integrator `xc` of its grid (None for HF): the semilocal kernel on
    /// the backend the plan puts it on (family `response_xc`: the integrator's own, or the
    /// device's in the arithmetic the device policy gives the response `kernel` serves); plus
    /// VV10's kernel `vv10` (-V functionals, on the CPU).
    pub(crate) fn response_kernel<'a>(
        &self,
        xc: Option<&'a XcIntegrator>,
        vv10: Option<linus_dft::vv10_resp::Vv10Kernel<'a>>,
        occ: &[MatRef<'_, f64>],
        kernel: crate::two_electron::KernelUse,
        timers: &mut Timers,
    ) -> Result<Option<XcKernel<'a>>, LinusError> {
        use crate::driver::backend::{Family, Placement};
        let Some(xc) = xc else {
            return Ok(None);
        };
        let semilocal = self.resolve(Family::ResponseXc, |b| {
            b.response_kernel(self, xc, occ, kernel, timers)
        })?;
        // with the semilocal kernel on the device, VV10's products in FP32 by the device policy
        // (precision.fp32 on, or a single-precision response; never with precision.fp32 off)
        let fp32 = self.ran(Family::ResponseXc) == Some(Placement::Device)
            && self
                .device_policy()
                .is_some_and(|p| p.vv10_cpks_fp32(kernel.single));
        Ok(Some(XcKernel::new(
            semilocal,
            vv10.map(|v| v.with_fp32(fp32)),
        )))
    }

    /// VV10's CPKS kernel of a response (-V functionals; None otherwise) for the occupied
    /// orbitals `occ` from the integrator `xc` with its VV10, at the ground state of their
    /// total density (timed as `<prefix>_vv10_setup`).
    pub(crate) fn vv10_kernel<'a>(
        &self,
        xc: Option<&'a XcIntegrator>,
        occ: &[MatRef<'_, f64>],
        prefix: &str,
        timers: &mut Timers,
    ) -> Result<Option<linus_dft::vv10_resp::Vv10Kernel<'a>>, LinusError> {
        let Some(nl) = xc.and_then(|x| x.vv10()) else {
            return Ok(None);
        };
        let dt = linus_dft::vv10_resp::total_density(occ, self.par());
        let g = self.vv10_ground(nl, &dt, false, prefix, timers)?;
        Ok(Some(linus_dft::vv10_resp::Vv10Kernel::new(
            nl,
            std::sync::Arc::new(g),
        )))
    }

    /// VV10's response ground state at the total density `dt` (with the grid terms when
    /// `grid_terms`; timed as `<prefix>_vv10_setup`), its pair sweeps on the backend the plan
    /// puts VV10 on (family `vv10`: the device's engine, `<prefix>_vv10_device_setup`, else the
    /// ground state's own).
    pub(crate) fn vv10_ground(
        &self,
        nl: &linus_dft::vv10::Vv10,
        dt: &Mat<f64>,
        grid_terms: bool,
        prefix: &str,
        timers: &mut Timers,
    ) -> Result<linus_dft::vv10_resp::Vv10Ground, LinusError> {
        let mut g = timers.time(&format!("{prefix}_vv10_setup"), || {
            nl.response_ground(dt, grid_terms)
        });
        if g.active_points() > 0 {
            g.device = self.resolve(crate::driver::backend::Family::Vv10, |b| {
                b.vv10_sweeps(self, &g, prefix, timers)
            })?;
        }
        Ok(g)
    }

    /// Core Hamiltonian (with the ECPs) and overlap.
    pub fn one_electron(&self, pairs: &PairSet) -> (Mat<f64>, Mat<f64>) {
        let (s, mut h) = self.one_electron_no_ecp(pairs);
        if let Some(u) = self.ecp_matrix(pairs) {
            h += u;
        }
        (s, h)
    }

    /// Overlap and T + V (point nuclei with their effective charges), without the ECPs.
    pub fn one_electron_no_ecp(&self, pairs: &PairSet) -> (Mat<f64>, Mat<f64>) {
        let n = self.basis.nbf;
        let charges = self.mol.point_charges();
        let plan = OneEPlan::new(&self.basis, DerivOrder::Zero)
            .with_charges(pairs, &charges)
            .expect("the potential takes pairs of any order");
        let (mut s, mut t, mut v) = (Mat::zeros(n, n), Mat::zeros(n, n), Mat::zeros(n, n));
        plan.overlap_kinetic(&mut ao_view_mut(s.as_mut()), &mut ao_view_mut(t.as_mut()));
        plan.potential(&mut ao_view_mut(v.as_mut()));
        (s, t + v)
    }

    /// The ECP part of the core Hamiltonian (None without ECPs).
    pub fn ecp_matrix(&self, pairs: &PairSet) -> Option<Mat<f64>> {
        (!self.ecp.is_empty()).then(|| {
            let n = self.basis.nbf;
            let mut u = Mat::zeros(n, n);
            let plan = EcpPlan::new(&self.basis, pairs, &self.ecp, DerivOrder::Zero);
            plan.matrix(&mut ao_view_mut(u.as_mut()));
            u
        })
    }

    /// The SCF on the loop `scf` with this session's environment term (the plan's solvent,
    /// when it has one), set up with `setup`: the solvent's term opened for the SCF, the loop run
    /// with it, the term closed (the solvation parts of the result and the summary). The SCF
    /// stage's composition: the solvent is a term the SCF runs with, not its driver.
    pub(crate) fn scf_on(
        &self,
        setup: &ScfSetup,
        scf: &mut dyn linus_scf::ScfLoop,
    ) -> Result<(ScfResult, Option<linus_solvent::SolvationResult>), LinusError> {
        let Some(mut env) = crate::solvent::scf::open_environment(self, setup, scf)? else {
            return Ok((scf.run(None)?, None));
        };
        let r = scf.run(Some(env.term()));
        let (r, solvation) = env.close(self, setup, scf, r)?;
        Ok((r, Some(solvation)))
    }

    /// What an SCF converged for `purpose` is set up with: the purpose's grid, and the memory
    /// of the large allocations alive with the DF tensors, reserved before their setup sizes
    /// them, and the DF budget that leaves (the guard of an unset memory_gb reads the free memory
    /// here: once per SCF).
    pub(crate) fn scf_setup(&self, purpose: Purpose) -> Result<ScfSetup, LinusError> {
        let reserved = self.scf_memory_reservations()?;
        let fixed = self
            .scf_fixed_parts(purpose, &reserved)?
            .iter()
            .map(|p| p.bytes)
            .sum();
        Ok(ScfSetup {
            purpose,
            grid: self.plan.grid(purpose),
            memory: ScfMemory::new(self.plan.memory(), reserved, fixed),
        })
    }
}

impl Session {
    /// Set up a calculation: the settings are normalized and checked, the basis sets loaded and
    /// the dispersion parameters looked up, so that bad input fails here, before any SCF.
    pub fn new(mol: Structure, opts: CalcOptions) -> Result<Self, LinusError> {
        Self::new_checked(mol, opts, true)
    }

    /// A session whose given settings are not checked (decision D41's effect test sets each
    /// setting where the table says it cannot take effect, to show it does not).
    #[cfg(test)]
    pub(crate) fn new_unchecked(mol: Structure, opts: CalcOptions) -> Result<Self, LinusError> {
        Self::new_checked(mol, opts, false)
    }

    fn new_checked(mut mol: Structure, opts: CalcOptions, check: bool) -> Result<Self, LinusError> {
        let mut plan = plan::Plan::new(opts)?;
        mol.validate().map_err(LinusError::Input)?;
        let basis_def = plan.load_basis(plan.basis())?;
        // ECP atoms: their core electrons leave the electron count and the nuclear charges
        let ecp = EcpSet::new(&basis_def, &mol.z, &mol.positions)?;
        mol.set_ecp_core_electrons(ecp.core_electrons(mol.natom()));
        let (na, nb) = mol.nalpha_nbeta().map_err(LinusError::Input)?;
        let method = plan.check(&mol, na, nb)?;
        let basis = BasisSet::new(&basis_def, &mol.z, &mol.positions, plan.pure())?;
        // every electron of a spin needs an orbital
        if na.max(nb) > basis.nbf {
            return Err(LinusError::Input(format!(
                "{na} alpha and {nb} beta electrons need at least {} orbitals per spin, and {} has {} basis functions here; \
                 check the charge and multiplicity or use a larger basis set",
                na.max(nb),
                plan.basis(),
                basis.nbf
            )));
        }
        // an explicit iteration auxiliary basis must cover every element (an "auto" one may
        // fall back to the single-level response)
        if let IterationAux::Named(name) = &plan.response().iteration_aux() {
            let def = plan.load_basis(name)?;
            BasisSet::new(&def, &mol.z, &mol.positions, true).map_err(|e| {
                LinusError::Input(format!("response.iteration_aux: {}", LinusError::from(e)))
            })?;
        }
        if let Some(lmax) = plan.jk().max_orbital_l().filter(|&l| basis.max_l() > l) {
            return Err(LinusError::Unsupported(format!(
                "jk {}: the seminumerical exchange supports orbital angular momentum up to l = {lmax} (g), and {} has l = {}; use jk \"df\"",
                plan.jk().name(),
                plan.basis(),
                basis.max_l()
            )));
        }
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(plan.threads().max(1))
            .build()
            .map_err(|e| {
                LinusError::resource(format!("cannot start {} threads: {e}", plan.threads()))
            })?;
        // the device of a GPU backend, once per session
        let backend = backend::BackendState::open(
            plan.backend(),
            plan.device_memory_cap()?.map(|c| c.0),
            &plan.backend().kernel_cache(),
        )?;
        // the device's policy and memory plan, and where every family of kernels runs
        plan.decide(&method, backend.device());
        let progress = if plan.verbose() {
            linus_base::progress::Progress::stderr()
        } else {
            linus_base::progress::Progress::off()
        };
        let mut cx = CalcContext {
            mol,
            plan,
            calctypes: settings::CalcTypes::ALL,
            basis_def,
            basis,
            ecp,
            pool,
            method,
            integrals: Default::default(),
            backend,
            record: record::Recorder::default(),
            progress,
            process: Default::default(),
        };
        // the auxiliary basis of the energy (names, elements, angular momentum) and the
        // dispersion parameters and elements, before any SCF
        let aux = match cx.plan.jk().aux() {
            Some(aux) => Some(cx.aux_set(aux, Purpose::Energy)?),
            None => None,
        };
        // the device kernels' coverage of the basis sets, in the placement (D40)
        let uncovered = cx
            .backend
            .provider()
            .and_then(|p| p.uncovered(&cx.basis, aux.as_deref()));
        let method = cx.method.clone();
        cx.plan.cover(&method, uncovered);
        cx.dispersion(0)?;
        // a given setting that cannot take effect is an input error, before any work (D41)
        if check {
            cx.check_settings()?;
        }
        Ok(Session {
            cx,
            ground: None,
            unconverged: None,
            lineages: Default::default(),
            call: None,
            given: Default::default(),
            unconverged_response: None,
            timers: Timers::default(),
        })
    }

    /// Declare the kind of process the session runs in (the command line's by default): its
    /// baseline starts every phase's memory need (`memory_gib` limits the whole process).
    pub fn set_process(&mut self, process: linus_base::memory::Process) {
        self.cx.process = process;
    }

    /// Collect the progress lines of `output.verbose` instead of writing them to stderr, for an
    /// in-process caller, which returns them as the output's logs ([`Session::take_logs`]):
    /// a program does not print in its caller's process.
    pub fn collect_progress(&mut self) {
        self.cx.progress = self.cx.progress.clone().collecting();
    }

    /// The progress lines collected since the last call (None: not collecting).
    pub fn take_logs(&self) -> Option<String> {
        self.cx.progress.take()
    }

    /// Hand every progress event of this session's calculations to `listener`
    /// ([`linus_base::progress::Event`]: the stages, the SCF's and the responses' iterations), as
    /// they happen, on the calculation's threads.
    pub fn listen(&mut self, listener: linus_base::progress::Listener) {
        self.cx.progress = self.cx.progress.clone().with_events(listener);
    }

    /// Ask `check` whether to stop, before each stage and between the SCF's and the responses'
    /// iterations (an in-process caller's `ctx.stop_requested()`): a calculation it stops fails
    /// with kind `stopped` and what it reached ([`LinusError::Stopped`]; [`Session::failure`]).
    pub fn stop_when(&mut self, check: linus_base::progress::StopCheck) {
        self.cx.progress = self.cx.progress.clone().with_stop(check);
    }

    /// Declares the calctypes this session runs (all by default; the CLI declares its one
    /// calculation's): the given settings are checked again against them (a setting only other
    /// calctypes read is then `not_applicable`), and a calculation outside them is an input
    /// error.
    pub fn declare(&mut self, calctypes: settings::CalcTypes) -> Result<(), LinusError> {
        self.cx.calctypes = calctypes;
        self.cx.check_settings()
    }

    /// The molecule (positions in bohr; move atoms with [`Session::set_positions`]), with the
    /// core electrons the basis set's ECPs replace.
    pub fn mol(&self) -> &Structure {
        &self.cx.mol
    }

    /// The auxiliary basis `aux` for `purpose`.
    pub fn aux_basis(&self, aux: &AuxBasis, purpose: Purpose) -> Result<BasisDef, LinusError> {
        self.cx.aux_basis(aux, purpose)
    }

    /// The auxiliary basis set `aux` for `purpose` placed on the atoms, checked (every element
    /// present, angular momentum at most [`MAX_L_AUX`]).
    pub fn aux_set(&self, aux: &AuxBasis, purpose: Purpose) -> Result<BasisSet, LinusError> {
        Ok((*self.cx.aux_set(aux, purpose)?).clone())
    }

    /// Core Hamiltonian (with the ECPs) and overlap.
    pub fn one_electron(&self, pairs: &PairSet) -> (Mat<f64>, Mat<f64>) {
        self.cx.one_electron(pairs)
    }

    /// Overlap and T + V without the ECPs.
    pub fn one_electron_no_ecp(&self, pairs: &PairSet) -> (Mat<f64>, Mat<f64>) {
        self.cx.one_electron_no_ecp(pairs)
    }

    /// The ECP part of the core Hamiltonian.
    pub fn ecp_matrix(&self, pairs: &PairSet) -> Option<Mat<f64>> {
        self.cx.ecp_matrix(pairs)
    }

    /// The electric dipole moment of the spin densities.
    pub fn dipole_moment(&self, d_a: &Mat<f64>, d_b: &Mat<f64>) -> [f64; 3] {
        self.cx.dipole_moment(d_a, d_b)
    }

    /// The density-fitting options for exchange with range separation `omega` and the
    /// tensors' budget `budget`.
    pub fn df_config(&self, omega: Option<f64>, budget: usize) -> linus_jk::df::DfConfig {
        self.cx.df_config(omega, budget)
    }

    /// What ran each family of kernels.
    pub fn backend_used(&self) -> crate::driver::report::ResolvedBackend {
        self.cx.backend_used()
    }

    /// Whether this session can do `req`.
    pub fn check_request(
        &self,
        req: crate::derivatives::capabilities::Request,
    ) -> Result<(), LinusError> {
        self.cx.check_request(req)
    }

    /// Wall-clock seconds per phase (and resources) of the last calculation.
    pub fn timers(&self) -> &Timers {
        &self.timers
    }

    /// The settings in effect (normalized).
    pub fn opts(&self) -> &CalcOptions {
        self.cx.plan.request()
    }

    /// The orbital basis set as placed on the atoms.
    pub fn basis(&self) -> &BasisSet {
        &self.cx.basis
    }

    /// The orbital basis set definition.
    pub fn basis_def(&self) -> &BasisDef {
        &self.cx.basis_def
    }

    /// The effective core potentials of the basis set.
    pub fn ecp(&self) -> &EcpSet {
        &self.cx.ecp
    }

    /// The method as resolved from the options.
    pub fn method(&self) -> &MethodInfo {
        &self.cx.method
    }

    /// The last converged SCF (possibly at other coordinates; see [`Session::last_here`]).
    pub fn last(&self) -> Option<&ScfResult> {
        self.ground.as_ref().map(|g| &g.scf)
    }

    /// The last converged SCF if it belongs to the current coordinates.
    pub fn last_here(&self) -> Option<&ScfResult> {
        self.ground_here().map(|g| &g.scf)
    }

    /// The ground state of the last converged SCF if it belongs to the current coordinates.
    pub(crate) fn ground_here(&self) -> Option<&Arc<ground::GroundState>> {
        self.ground
            .as_ref()
            .filter(|g| g.is_at(&self.cx.mol.positions))
    }

    /// The last SCF that did not converge (cleared by the next converged one).
    pub fn last_unconverged(&self) -> Option<&ScfResult> {
        self.unconverged.as_ref().map(|g| &g.scf)
    }

    /// The result computed from the last CPHF/CPKS solution that missed its tolerance.
    pub fn unconverged_response(&self) -> Option<&UnconvergedResponse> {
        self.unconverged_response.as_ref()
    }

    /// The DFT grid of the last converged SCF (before any SCF: the energy's).
    pub fn scf_grid(&self) -> GridChoice {
        match &self.ground {
            Some(g) => g.grid().clone(),
            None => self.cx.plan.grid(Purpose::Energy),
        }
    }

    /// What the last converged SCF was converged for (before any SCF: the energy).
    pub fn scf_purpose(&self) -> Purpose {
        self.ground
            .as_ref()
            .map_or(Purpose::Energy, |g| g.purpose())
    }

    /// VV10 statistics of the last converged SCF (-V functionals).
    pub fn vv10(&self) -> Option<&linus_dft::vv10::Vv10Stats> {
        self.ground.as_ref().and_then(|g| g.vv10.as_ref())
    }

    /// Solvation results of the last converged SCF.
    pub fn solvation(&self) -> Option<&linus_solvent::SolvationResult> {
        self.ground.as_ref().and_then(|g| g.solvation.as_ref())
    }

    /// The session's thread pool (every computation of the session runs in it).
    pub fn pool(&self) -> &rayon::ThreadPool {
        &self.cx.pool
    }

    pub fn par(&self) -> Par {
        if self.cx.plan.threads() > 1 {
            Par::rayon(self.cx.plan.threads())
        } else {
            Par::Seq
        }
    }

    pub fn is_dft(&self) -> bool {
        self.cx.method.is_dft()
    }

    /// Hessian-vector products of the stability probe the next SCF runs: `scf.stability_probe`
    /// when set, else [`linus_scf::STABILITY_PROBE_AUTO`] at an SCF that starts fresh (a
    /// lineage's first, or after a reset or a jump beyond [`lineage::GUESS_DISTANCE`]) and none
    /// at one that continues its lineage. The SCF runs it for UHF/UKS determinants only.
    pub fn stability_probe(&self) -> usize {
        self.next_start()
            .stability_probe(self.cx.plan.scf_algorithm())
    }

    /// Hessian-vector products of the closed-shell probe the next SCF runs (decision D36):
    /// `scf.stability_closed_probe` when set, else [`linus_scf::CLOSED_SHELL_PROBE_AUTO`] for
    /// KS-DFT at an SCF that starts fresh, none for Hartree-Fock and at one that continues its
    /// lineage. The SCF runs it for RHF/RKS determinants whose HOMO-LUMO gap lies below
    /// `scf.stability_closed_gap`.
    pub fn closed_shell_probe(&self) -> usize {
        self.next_start()
            .closed_shell_probe(self.cx.plan.scf_algorithm(), self.is_dft())
    }

    /// The closed-shell probe's warning (`closed_shell_unstable`) when the probe of the
    /// converged SCF `r` found a negative mode; the record is
    /// `resolved.record.scf.closed_shell_probe`.
    pub(crate) fn closed_shell_warning(&self, r: &ScfResult) -> Option<linus_core::Warning> {
        let p = r.closed_shell_probe.filter(|p| p.is_unstable())?;
        let (rs, us) = if self.is_dft() {
            ("RKS", "UKS")
        } else {
            ("RHF", "UHF")
        };
        let e = p.eigenvalue.unwrap_or(f64::NAN);
        let message = if p.kind == Some(linus_scf::ProbeKind::Internal) {
            format!(
                "{rs} is unstable towards a lower {rs} solution (orbital-Hessian eigenvalue {e:+.1e} \
                 Eh, an internal mode); rerun with reference: unrestricted and scf.stability: always to \
                 descend (resolved.record.scf.closed_shell_probe)"
            )
        } else {
            format!(
                "{rs} is unstable towards a broken-symmetry {us} solution (eigenvalue {e:+.1e} \
                 Eh); rerun with reference: unrestricted (scf.stability: always analyses it) \
                 (resolved.record.scf.closed_shell_probe)"
            )
        };
        Some(linus_core::Warning {
            code: "closed_shell_unstable",
            message,
            path: "scf.stability_closed_probe",
        })
    }

    /// Move the atoms (bohr; one finite position per atom, no two closer than
    /// [`crate::structure::MIN_DISTANCE`]). The next SCF starts from its lineage's state as
    /// [`lineage`] says: the last converged orbitals made orthonormal at the new positions
    /// (Löwdin: C (C^T S C)^-1/2), or the last densities at the same positions or where the
    /// orbitals are linearly dependent.
    pub fn set_positions(&mut self, positions: &[[f64; 3]]) -> Result<(), LinusError> {
        let n = self.cx.mol.natom();
        if positions.len() != n {
            return Err(LinusError::Input(format!(
                "{} positions for {n} atoms",
                positions.len()
            )));
        }
        if positions.iter().flatten().any(|x| !x.is_finite()) {
            return Err(LinusError::Input("positions must be finite numbers".into()));
        }
        crate::structure::check_distances(&self.cx.mol.z, positions).map_err(LinusError::Input)?;
        self.move_atoms(positions);
        Ok(())
    }

    fn move_atoms(&mut self, positions: &[[f64; 3]]) {
        if self.cx.mol.positions != positions {
            // the setups of the old geometry go
            self.cx.integrals = Default::default();
            self.cx.backend.clear_session();
        }
        self.cx.mol.positions = positions.to_vec();
        self.cx.basis.set_positions(positions);
        self.cx.ecp.set_positions(positions);
    }

    /// Start the next SCF of the direct calls from these spin densities (nbf x nbf each; closed
    /// shell: both half the total density). Used once.
    pub fn set_guess_density(&mut self, d_a: Mat<f64>, d_b: Mat<f64>) -> Result<(), LinusError> {
        let n = self.cx.basis.nbf;
        for (name, d) in [("density_alpha", &d_a), ("density_beta", &d_b)] {
            if d.nrows() != n || d.ncols() != n {
                return Err(LinusError::Input(format!(
                    "{name} must be {n} x {n}, not {} x {}",
                    d.nrows(),
                    d.ncols()
                )));
            }
        }
        self.given.guess = Some((d_a, d_b, false));
        Ok(())
    }

    /// Start the next SCF of the direct calls from occupied orbitals of an earlier SCF of this
    /// molecule (nbf x nocc per spin, any geometry), made orthonormal at the current positions
    /// (Löwdin). Used once. `restricted`: the beta orbitals are the alpha ones.
    pub fn set_guess_orbitals(
        &mut self,
        c_a: MatRef<'_, f64>,
        c_b: MatRef<'_, f64>,
        restricted: bool,
    ) -> Result<(), LinusError> {
        let n = self.cx.basis.nbf;
        let (na, nb) = self.cx.mol.nalpha_nbeta().map_err(LinusError::Input)?;
        if c_a.nrows() != n || c_b.nrows() != n || c_a.ncols() != na || c_b.ncols() != nb {
            return Err(LinusError::Input(format!(
                "guess orbitals must be {n} x {na} (alpha) and {n} x {nb} (beta), not {} x {} and {} x {}",
                c_a.nrows(),
                c_a.ncols(),
                c_b.nrows(),
                c_b.ncols()
            )));
        }
        let (d_a, d_b) = self
            .cx
            .projected_guess(c_a, c_b, restricted)
            .ok_or_else(|| {
                LinusError::Input(
                    "the guess orbitals are linearly dependent at this geometry".into(),
                )
            })?;
        self.given.guess = Some((d_a, d_b, true));
        Ok(())
    }

    /// A new XC integrator on the last converged SCF's grid (before any SCF: the energy's; None
    /// for HF).
    pub fn build_xc(&self, timers: &mut Timers) -> Result<Option<XcIntegrator>, LinusError> {
        self.cx.build_xc_on(&self.scf_grid(), true, timers)
    }

    /// SCF energy, converged for single points ([`Purpose::Energy`]).
    pub fn energy(&mut self) -> Result<ScfResult, LinusError> {
        self.energy_for(Purpose::Energy)
    }

    /// SCF energy converged as `purpose` needs (unless `convergence` is set).
    pub fn energy_for(&mut self, purpose: Purpose) -> Result<ScfResult, LinusError> {
        Ok(self.scf_stage(purpose)?.scf.clone())
    }

    /// The SCF stage: an SCF converged as `purpose` needs at the current geometry, as its
    /// ground state ([`ground::GroundState`]), which the derivatives and properties of this
    /// geometry are then given. It starts as its lineage says ([`Session::take_start`]); a
    /// converged SCF becomes the session's last one and gives its lineage's next state
    /// ([`Session::scf_converged`]: a request commits it when it succeeds, a direct call now);
    /// an SCF that fails leaves the lineage as it was (the next one starts the same way) and, when
    /// it stopped unconverged, is kept for its failure report.
    pub(crate) fn scf_stage(
        &mut self,
        purpose: Purpose,
    ) -> Result<Arc<ground::GroundState>, LinusError> {
        // the kept ground state's reference is not needed through this SCF
        if let Some(g) = self.ground.as_mut().and_then(Arc::get_mut) {
            g.release_reference();
        }
        let mut start = self.take_start();
        let guess = start.guess.take();
        // a direct call's given guess starts the next SCF again if this one fails
        let given = (self.call.is_none() && start.record.source == lineage::GuessSource::Given)
            .then(|| guess.clone())
            .flatten();
        let progress = self.cx.progress.clone();
        let out = progress.stage("scf", || self.run_scf_stage(purpose, guess, &start));
        match &out {
            Ok(gs) => self.scf_converged(&start, &gs.scf),
            Err(_) => {
                if let Some((d_a, d_b)) = given {
                    self.given.guess = Some((d_a, d_b, start.record.orthonormalized));
                }
            }
        }
        out
    }

    /// [`Self::scf_stage`] from the guess densities `guess` (None: the configured guess), with
    /// the probes' rule and the record of `start`.
    fn run_scf_stage(
        &mut self,
        purpose: Purpose,
        guess: Option<(Mat<f64>, Mat<f64>)>,
        start: &lineage::ScfStart,
    ) -> Result<Arc<ground::GroundState>, LinusError> {
        let conv = self.cx.plan.convergence(purpose)?;
        if purpose == Purpose::Hessian {
            // an SCF for a response: what the method cannot do fails before it
            self.cx
                .check_request(crate::derivatives::capabilities::Request::Hessian {
                    dipole_derivatives: false,
                })?;
        }
        // a new SCF: what the last one left on the device goes (this one's setups leave their
        // own)
        self.cx.backend.clear_session();
        let mut timers = Timers::default();
        let t0 = std::time::Instant::now();
        let par = self.par();
        // what the SCF is set up with (its purpose, grid and memory)
        let setup = self.cx.scf_setup(purpose)?;
        // the probes this SCF is given (the rule of its start: fresh, or continuing its
        // lineage), for its record
        let (stability_probe, closed_shell_probe, probe_trigger) = {
            let (algo, dft) = (self.cx.plan.scf_algorithm(), self.is_dft());
            (
                start.stability_probe(algo),
                start.closed_shell_probe(algo, dft),
                start.probe_trigger(algo, dft),
            )
        };
        self.cx.record.decide(record::Decision::Probes {
            stability: stability_probe,
            closed_shell: closed_shell_probe,
        });
        let mut vv10 = None;
        let mut df_resolved = None;
        let (res, solvation) = self.cx.pool.install(
            || -> Result<(ScfResult, Option<linus_solvent::SolvationResult>), LinusError> {
                let pairs = timers.time("pairs", || {
                    self.cx.integrals.pairs(&self.cx.basis, DerivOrder::Zero)
                });
                let (s, mut h) =
                    timers.time("one_electron", || self.cx.one_electron_no_ecp(&pairs));
                if let Some(u) = timers.time("ecp", || self.cx.ecp_matrix(&pairs)) {
                    h += u;
                }
                // the SCF's J/K and XC term on the backends the plan puts them on
                // (crate::backend::scf); the pairs are done with once the J/K is set up
                let terms = self.cx.scf_terms(pairs, &setup, &mut timers)?;
                // VV10's statistics so far (an integrator shared with an earlier phase of this
                // geometry has some): the SCF's are the difference
                let vv10_before = terms.xc.as_ref().and_then(|x| x.vv10_stats());
                let (na, nb) = self.cx.mol.nalpha_nbeta().map_err(LinusError::Input)?;
                let guess = match (guess, self.cx.plan.guess()) {
                    (Some(g), _) => Some(g),
                    (None, Guess::Core) => None,
                    (None, Guess::Sad) => Some(timers.time("guess", || {
                        linus_scf::guess::sad_guess(
                            &self.cx.basis_def,
                            &self.cx.basis,
                            &self.cx.mol.z,
                            na,
                            nb,
                            par,
                        )
                    })),
                };
                let algo = self.cx.plan.scf_algorithm().clone();
                let mut opts = ScfConfig {
                    restricted: self.cx.plan.restricted(),
                    max_iter: self.cx.plan.max_iter(),
                    // under accuracy targets the plateau rule accepts only what meets them
                    plateau_limit: self.cx.plan.targets().for_purpose(purpose).plateau_limit(),
                    guess: self.cx.plan.guess(),
                    progress: self.cx.progress.clone(),
                    algo: linus_scf::ScfAlgorithm {
                        stability_probe: Some(stability_probe),
                        closed_shell_probe: Some(closed_shell_probe),
                        ..algo
                    },
                    // the zero-mode check of open-shell Hessians (the probe's products at every
                    // converged point; atoms have a zero Hessian and are skipped)
                    mode_check: if purpose == Purpose::Hessian && self.cx.mol.z.len() >= 2 {
                        self.cx.plan.hessian_check()
                    } else {
                        0
                    },
                    probe_trigger,
                    ..Default::default()
                };
                opts.set_convergence(conv);
                let inp = ScfInput {
                    h: h.as_ref(),
                    s: s.as_ref(),
                    e_nuc: self.cx.mol.nuclear_repulsion(),
                    nalpha: na,
                    nbeta: nb,
                    exchange: self.cx.method.exchange,
                    guess_density: guess,
                    // the point group of the stability analyses' searches
                    geometry: Some(linus_scf::stability::Geometry::new(
                        &self.cx.mol.z,
                        &self.cx.mol.positions,
                        &self.cx.basis,
                    )),
                };
                let run = ScfRun {
                    inp: &inp,
                    opts: &opts,
                    par,
                };
                // the J/K against what it can hold once the other engines are set up, before the
                // first iteration (a device's: the device loop, then the host loop, then the
                // CPU's J/K)
                let mut terms = self.cx.fit_scf_terms(terms, &run, &setup, &mut timers)?;
                df_resolved = terms.df.take();
                // the SCF on the loop of the backend the plan puts it on (the host loop or the
                // device's), with the solvent's reaction field when the plan has a solvent
                // (CalcContext::scf_on)
                let (mut r, solvation) = {
                    let mut scf = self.cx.scf_loop(&mut terms, run)?;
                    self.cx.scf_on(&setup, scf.as_mut())?
                };
                // the J/K is done with; the XC term keeps VV10's statistics
                let ScfTerms { jk, xc, .. } = terms;
                drop(jk);
                vv10 = xc
                    .as_ref()
                    .and_then(|x| x.vv10_stats())
                    .map(|v| v.since(vv10_before.as_ref()));
                // the J/K method's end of a converged SCF (crate::two_electron)
                self.cx.finish_scf_jk(&mut r, par)?;
                Ok((r, solvation))
            },
        )?;
        timers.merge(&res.timers);
        if let Some(v) = &vv10 {
            // part of "xc": all of VV10, its pair sums, densities and potential matrices
            timers.add("vv10", v.total());
            timers.add("vv10_pairs", v.t_pairs);
            timers.add("vv10_density", v.t_density);
            timers.add("vv10_vmat", v.t_vmat);
        }
        // the whole SCF stage (setup, iterations, final energy); `total` is set per call
        timers.add("energy", t0.elapsed().as_secs_f64());
        timers.set_total(t0.elapsed().as_secs_f64());
        self.timers = timers;
        let positions = self.cx.mol.positions.clone();
        let started = start.record.clone();
        let ground = move |scf| {
            ground::GroundState::new(setup, scf, positions, started, vv10, solvation, df_resolved)
        };
        if !res.energy.is_finite() {
            self.unconverged = Some(Arc::new(ground(res)));
            return Err(LinusError::Numerical("the SCF energy is not finite (NaN or infinite); check the geometry for overlapping atoms".into()));
        }
        if !res.converged {
            // stopped on request, or out of iterations: the last iterate is kept for the failure
            let err = if res.stopped() {
                LinusError::stopped("scf", Some(res.iterations))
            } else {
                LinusError::NotConverged(Box::new(res.failure()))
            };
            self.unconverged = Some(Arc::new(ground(res)));
            return Err(err);
        }
        self.unconverged = None;
        let mut res = res;
        if let Some(d) = self.cx.dispersion(0)? {
            res.e_disp = d.energy;
            res.energy += d.energy;
        }
        let gs = Arc::new(ground(res));
        self.ground = Some(gs.clone());
        Ok(gs)
    }
}

/// Density C' C'^T of occupied orbitals `c` made orthonormal in the metric `s`, C' = C (C^T S C)^-1/2
/// (idempotent, with the right number of electrons); None if they are linearly dependent.
#[allow(clippy::neg_cmp_op_on_partial_ord, reason = "true for NaN too")]
pub fn lowdin_density(c: MatRef<'_, f64>, s: MatRef<'_, f64>, par: Par) -> Option<Mat<f64>> {
    let n = c.nrows();
    if c.ncols() == 0 {
        return Some(Mat::zeros(n, n));
    }
    let m = mm(mm(c.transpose(), s, par).as_ref(), c, par);
    let (w, v) = linus_base::linalg::eigh(m.as_ref(), par);
    let (lo, hi) = (w[0], w[w.len() - 1]);
    if !(lo > 1e-8 * hi) {
        return None;
    }
    let k = w.len();
    let x = mm(
        Mat::from_fn(k, k, |i, j| v[(i, j)] / w[j].sqrt()).as_ref(),
        v.transpose(),
        par,
    );
    let c2 = mm(c, x.as_ref(), par);
    Some(mm(c2.as_ref(), c2.transpose(), par))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Composite auxiliary basis sets take each element from the first set that has it; "auto"
    /// and the default grid resolve by purpose.
    #[test]
    fn composite_aux_and_purpose_defaults() {
        let o = CalcOptions::default();
        let path = o.basis_path();
        let tz = o.load_basis("cc-pvtz-jkfit").unwrap();
        let un = o.load_basis("def2-universal-jkfit").unwrap();
        let c = o.load_basis("cc-pvtz-jkfit|def2-universal-jkfit").unwrap();
        assert_eq!(c.elements.len(), un.elements.len());
        assert_eq!(c.elements[&1], tz.elements[&1]);
        assert_eq!(c.elements[&19], un.elements[&19]);
        assert!(!tz.elements.contains_key(&19));
        assert!(BasisDef::load_composite("def2-svp", &path).is_ok());
        assert_eq!(
            AuxBasis::Auto.for_purpose(Purpose::Energy),
            AUX_ENERGY_DEFAULT
        );
        assert_eq!(
            AuxBasis::from("AUTO").for_purpose(Purpose::Hessian),
            AUX_HESSIAN_DEFAULT
        );
        assert_eq!(
            AuxBasis::from("cc-pvqz-jkfit").for_purpose(Purpose::Hessian),
            "cc-pvqz-jkfit"
        );
        assert_eq!(
            GridChoice::Auto.for_purpose(Purpose::Gradient),
            GridChoice::ENERGY_DEFAULT
        );
        assert_eq!(
            GridChoice::Auto.for_purpose(Purpose::Hessian),
            GridChoice::HESSIAN_DEFAULT
        );
        assert_eq!(
            GridChoice::Pyscf { level: 4 }.for_purpose(Purpose::Hessian),
            GridChoice::Pyscf { level: 4 }
        );
    }

    fn water() -> Structure {
        Structure::from_xyz_str(
            "3\n\nO 0 0 0.1173\nH 0 0.7572 -0.4692\nH 0 -0.7572 -0.4692\n",
            0,
            1,
        )
        .unwrap()
    }

    /// Bad input fails in `Session::new` with the right kind and an actionable message.
    #[test]
    fn early_errors() {
        let new = |o: CalcOptions| Session::new(water(), o).err().expect("an error");
        let e = new(CalcOptions {
            basis: Some("def2svp".into()),
            ..Default::default()
        });
        assert!(e.is_input() && e.to_string().contains("def2-svp"), "{e}");
        let e = new(CalcOptions {
            method: Some("b3lpy".into()),
            ..Default::default()
        });
        assert!(
            e.is_input() && e.to_string().contains("b3lyp") && e.to_string().contains("dispersion"),
            "{e}"
        );
        let e = new(CalcOptions {
            method: Some("wb97x-d3".into()),
            ..Default::default()
        });
        assert!(
            matches!(e, LinusError::Unsupported(_)) && e.to_string().contains("wb97x-d3bj"),
            "{e}"
        );
        let e = new(CalcOptions {
            jk: JkMethod::Df {
                aux: "def2-universal-jkfi".into(),
            }
            .into(),
            ..Default::default()
        });
        assert!(
            e.is_input() && e.to_string().contains("def2-universal-jkfit"),
            "{e}"
        );
        let e =
            CalcOptions::from_json(serde_json::json!({"method": "pbe", "dispersion": "d3zero"}))
                .expect_err("an error");
        assert!(matches!(e, LinusError::Unsupported(_)), "{e}");
        let e = new(CalcOptions {
            response: ResponseSettings {
                iteration_aux: Some("nope".into()),
                ..Default::default()
            },
            ..Default::default()
        });
        assert!(e.is_input(), "{e}");
        if !crate::backend::GPU_BUILT {
            let e = new(CalcOptions {
                backend: serde_json::from_value(serde_json::json!("gpu")).unwrap(),
                ..Default::default()
            });
            assert!(
                matches!(e, LinusError::Unsupported(_)) && e.to_string().contains("GPU"),
                "{e}"
            );
        }
        let iodine = Structure::from_xyz_str("2\n\nI 0 0 0\nI 0 0 2.67\n", 0, 1).unwrap();
        let e = Session::new(
            iodine,
            CalcOptions {
                basis: Some("cc-pvdz".into()),
                ..Default::default()
            },
        )
        .err()
        .unwrap();
        assert!(
            e.to_string().contains("element I (Z=53)") && e.to_string().contains("def2"),
            "{e}"
        );
        assert!(Structure::new(vec![0], vec![[0.0; 3]], 0, 1)
            .unwrap_err()
            .is_input());
        // coincident atoms, more electrons per spin than orbitals
        let e = Structure::from_xyz_str("2\n\nHe 0 0 0\nHe 0 0 0.01\n", 0, 1).unwrap_err();
        assert!(e.is_input() && e.to_string().contains("closer than"), "{e}");
        let h = Structure::new(vec![1], vec![[0.0; 3]], -3, 1).unwrap();
        let e = Session::new(
            h,
            CalcOptions {
                basis: Some("sto-3g".into()),
                ..Default::default()
            },
        )
        .err()
        .unwrap();
        assert!(e.is_input() && e.to_string().contains("orbitals"), "{e}");
        // an explicit iteration auxiliary basis must cover every element
        let e = Session::new(
            potassium_water(),
            CalcOptions {
                response: ResponseSettings {
                    iteration_aux: Some("cc-pvtz-jkfit".into()),
                    ..Default::default()
                },
                ..Default::default()
            },
        )
        .err()
        .unwrap();
        assert!(
            e.is_input() && e.to_string().contains("iteration_aux"),
            "{e}"
        );
    }

    fn potassium_water() -> Structure {
        Structure::from_xyz_str(
            "4\n\nO 0 0 0.1173\nH 0 0.7572 -0.4692\nH 0 -0.7572 -0.4692\nK 0 0 3.0\n",
            1,
            1,
        )
        .unwrap()
    }

    /// Positions are checked when atoms move; a failed SCF leaves the guess it started from.
    #[test]
    fn checked_moves_and_failed_scf() {
        let mut c = Session::new(
            water(),
            CalcOptions {
                basis: Some("sto-3g".into()),
                ..Default::default()
            },
        )
        .unwrap();
        let mut xyz = c.mol().positions().to_vec();
        assert!(c.set_positions(&xyz[..2]).unwrap_err().is_input());
        xyz[1] = xyz[0];
        assert!(c
            .set_positions(&xyz)
            .unwrap_err()
            .to_string()
            .contains("closer than"));
        xyz[1] = [f64::NAN; 3];
        assert!(c.set_positions(&xyz).is_err());
        c.energy().unwrap();
        let mut xyz = c.mol().positions().to_vec();
        xyz[0][2] += 0.05;
        c.set_positions(&xyz).unwrap();
        let next_guess = |c: &Session| {
            let key = lineage::LineageKey::default();
            let s = c.start_from(&key, c.lineages.get(&key).map(|l| (&key, l)), true);
            assert!(s.continuation && s.record.orthonormalized);
            s.guess.unwrap()
        };
        let projected = next_guess(&c);
        // an SCF that fails (one iteration) leaves the lineage it started from
        c.cx.plan.request_mut().scf.max_iter = Some(1);
        assert!(matches!(c.energy(), Err(LinusError::NotConverged(_))));
        let again = next_guess(&c);
        assert!(again.0 == projected.0 && again.1 == projected.1);
    }

    /// The guess state: the next SCF after a move starts from the projected orbitals; the last
    /// result is stamped with its coordinates.
    #[test]
    fn guess_state() {
        let mut c = Session::new(
            water(),
            CalcOptions {
                basis: Some("sto-3g".into()),
                ..Default::default()
            },
        )
        .unwrap();
        assert!(c.lineages().is_empty() && c.last_here().is_none());
        let r0 = c.energy().unwrap();
        assert!(!c.lineages().is_empty() && c.last_here().is_some());
        let mut xyz = c.mol().positions.clone();
        xyz[0][2] += 0.05;
        c.set_positions(&xyz).unwrap();
        assert!(c.last_here().is_none() && c.last().is_some());
        let r1 = c.energy().unwrap();
        assert!(
            r1.iterations < r0.iterations,
            "{} vs {}",
            r1.iterations,
            r0.iterations
        );
        c.reset_all();
        assert!(c.lineages().is_empty());
    }
}
