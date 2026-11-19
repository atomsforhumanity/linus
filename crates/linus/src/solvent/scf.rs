//! The solvent's part of an SCF on the backend the plan puts the PCM on: the environment term
//! and its Coulomb hook, opened before the loop and closed after it ([`ScfEnvironment`]; the
//! CPU's over linus-solvent's `PcmScf`), and the checks of the solvent settings before any SCF.

use crate::driver::ground::{GroundState, ScfSetup};
use crate::driver::CalcContext;
use crate::solvent::SolventSettings;
use linus_base::error::{LinusError, MemoryNeed};
use linus_base::terms::{EnvironmentTerm, HookOutcome};
use linus_ints::BasisSet;
use linus_scf::{ScfLoop, ScfResult};
use linus_solvent::scf::{PcmScf, PcmTerm, SolvationResult};
use linus_solvent::surface::Surface;
use linus_solvent::{ri, smd, PcmScfMode, ResolvedPcm};

/// The SCF mode of the surface terms with the J/K method `jk`: the density-fitted PCM with
/// `solvent.df`, else `solvent.scf_terms` (auto: `predicted` with density-fitted J, `exact`
/// otherwise); `predicted` and `df` need density-fitted J.
pub(crate) fn resolve_scf_mode(
    so: &SolventSettings,
    jk: &crate::driver::JkMethod,
) -> Result<PcmScfMode, LinusError> {
    use crate::driver::schema::Auto;
    use crate::solvent::settings::PcmScfTerms;
    let df_j = jk.aux().is_some();
    let need_df_j = |what: &str| {
        Err(LinusError::Input(format!(
            "{what} needs density-fitted J (jk df or rij_snk)"
        )))
    };
    Ok(match (so.df(), so.scf_terms()) {
        (true, _) if !df_j => return need_df_j("solvent.df"),
        (true, _) => PcmScfMode::Df,
        (false, Auto::Value(PcmScfTerms::Exact)) => PcmScfMode::Exact,
        (false, Auto::Value(PcmScfTerms::Predicted)) if !df_j => {
            return need_df_j("solvent.scf_terms predicted")
        }
        (false, Auto::Value(PcmScfTerms::Predicted)) => PcmScfMode::Predicted,
        (false, Auto::Auto) if df_j => PcmScfMode::Predicted,
        (false, Auto::Auto) => PcmScfMode::Exact,
    })
}

/// The density-fitted PCM has no response terms: Hessians, frequencies and polarizabilities are
/// [`LinusError::Unsupported`] (`derivatives::capabilities`).
pub(crate) fn df_pcm_response_error() -> LinusError {
    LinusError::Unsupported(
        "analytic Hessians, frequencies and polarizabilities with the density-fitted PCM (solvent.df) are not implemented; leave \
         solvent.df off"
            .into(),
    )
}

/// Check the solvent settings of a calculation before any SCF (from `Session::new`): they
/// resolve for the atoms of `mol` (solvent name, SMD descriptors, radii, Lebedev grids,
/// permittivity) and the SCF mode of the surface terms suits the J/K method. Nothing to check
/// without `solvent`. Density-fitted PCM response calculations are refused when they are asked for
/// (`derivatives::capabilities`).
pub fn validate(
    opts: &crate::driver::CalcOptions,
    mol: &crate::structure::Structure,
) -> Result<(), LinusError> {
    let Some(so) = &opts.solvent else {
        return Ok(());
    };
    super::check(&opts.solvent)?;
    so.resolve(&mol.z).map_err(LinusError::Input)?;
    resolve_scf_mode(so, &opts.jk_method())?;
    Ok(())
}

/// The SCF mode of the surface terms for this calculation ([`resolve_scf_mode`]) and, for
/// `predicted` and `ri`, the auxiliary basis of the SCF's density-fitted J.
pub(crate) fn scf_mode(
    calc: &CalcContext,
    so: &SolventSettings,
    purpose: crate::driver::Purpose,
) -> Result<(PcmScfMode, Option<BasisSet>), LinusError> {
    // (an SCF for a response refuses the density-fitted PCM before it starts:
    // derivatives::capabilities)
    let mode = resolve_scf_mode(so, calc.plan.jk())?;
    if mode == PcmScfMode::Exact {
        return Ok((mode, None));
    }
    let adef = calc.aux_basis(calc.plan.jk().aux().expect("density-fitted J"), purpose)?;
    Ok((
        mode,
        Some(BasisSet::new(
            &adef,
            &calc.mol.z,
            &calc.mol.positions,
            true,
        )?),
    ))
}

/// The dense surface matrices of the PCM ([`linus_solvent::pcm::dense_peak_bytes`]) must fit in the
/// memory the budgets derive from (`memory_gib`, else the machine's or the cgroup's): a
/// [`LinusError::Resource`] before any allocation, not an out-of-memory kill. Builds the surface
/// (cheap next to the n^2 matrices) to count its points.
pub(crate) fn check_dense_memory(
    calc: &CalcContext,
    resolved: &ResolvedPcm,
    positions: &[[f64; 3]],
) -> Result<(), LinusError> {
    let mem = calc.plan.memory();
    let Some(limit) = mem.bytes() else {
        return Ok(());
    };
    let surf = Surface::build(
        positions,
        &resolved.radii,
        &resolved.lebedev,
        &resolved.xi,
        resolved.discretization,
        resolved.cut,
    )
    .map_err(LinusError::Input)?;
    let n = surf.len();
    let need = linus_solvent::pcm::dense_peak_bytes(resolved.model, n);
    if need <= limit as f64 {
        return Ok(());
    }
    const GIB: f64 = 1073741824.0;
    let per_sphere = resolved.lebedev.iter().max().copied().unwrap_or(0);
    Err(LinusError::host_memory(
        format!(
            "the {} surface has {n} points: its dense matrices need {:.1} GiB while they are built, more than the \
             {:.1} GiB of the {} allow; use fewer points per sphere (solvent.lebedev, now up to {per_sphere}), give \
             linus more memory (memory_gib, or --memory on the command line), or a smaller molecule",
            resolved.model.name(),
            need / GIB,
            limit as f64 / GIB,
            mem.source()
        ),
        MemoryNeed::bytes(need, limit as f64),
    ))
}

/// The solvent's part of one SCF (`calc.plan.solvent()`) as a backend set it up (family `pcm`,
/// `crate::backend`): the reaction field as the SCF's environment term, with the Coulomb hook it
/// puts on the loop's J build (the `predicted` and density-fitted modes: the J build hands the
/// term its fit coefficients); after the SCF, the result's solvation parts and the summary. The
/// driver opens it ([`open_environment`]), runs its loop with [`ScfEnvironment::term`] and closes
/// it.
pub(crate) trait ScfEnvironment {
    /// The environment term the SCF loop runs with.
    fn term(&mut self) -> &mut dyn EnvironmentTerm;

    /// After the loop's SCF `r` (`calc`'s, on `scf`): the Coulomb hook removed (also when the SCF
    /// failed, whose error is then returned), the solvation parts of the result and the summary.
    fn close(
        self: Box<Self>,
        calc: &CalcContext,
        setup: &ScfSetup,
        scf: &mut dyn ScfLoop,
        r: Result<ScfResult, LinusError>,
    ) -> Result<(ScfResult, SolvationResult), LinusError>;
}

/// What the solvent's term of an SCF is set up with: the solvent resolved, the SCF mode of its
/// surface terms, and the J build's auxiliary basis (the `predicted` and density-fitted modes).
pub(crate) struct PcmSetup {
    pub resolved: ResolvedPcm,
    pub mode: PcmScfMode,
    pub aux: Option<BasisSet>,
}

/// The solvent's term for an SCF on the loop `scf` (None in the gas phase), on the backend the
/// plan puts the PCM on, its Coulomb hook installed.
pub(crate) fn open_environment(
    calc: &CalcContext,
    setup: &ScfSetup,
    scf: &mut dyn ScfLoop,
) -> Result<Option<Box<dyn ScfEnvironment>>, LinusError> {
    let Some(so) = calc.plan.solvent() else {
        return Ok(None);
    };
    let resolved = so.resolve(&calc.mol.z).map_err(LinusError::Input)?;
    // the nuclear potential of the effective charges (Z minus ECP core electrons; an ECP has no
    // long-range part); radii come from the elements
    let (mode, aux) = scf_mode(calc, so, setup.purpose)?;
    let pcm = PcmSetup {
        resolved,
        mode,
        aux,
    };
    calc.resolve(crate::driver::backend::Family::Pcm, |b| {
        b.scf_environment(calc, setup, scf, &pcm)
    })
    .map(Some)
}

/// The CPU's term of the solvent `pcm` for an SCF on the loop `scf`, its Coulomb hook installed
/// (the J build of the `predicted` and density-fitted modes hands the term its fit coefficients;
/// a J build that cannot is an unsupported error).
pub(crate) fn cpu_environment(
    calc: &CalcContext,
    _setup: &ScfSetup,
    scf: &mut dyn ScfLoop,
    pcm: &PcmSetup,
) -> Result<Box<dyn ScfEnvironment>, LinusError> {
    let mode = pcm.mode;
    check_dense_memory(calc, &pcm.resolved, &calc.mol.positions)?;
    let state = PcmScf::with_mode(
        pcm.resolved.clone(),
        &calc.mol.effective_z(),
        &calc.mol.positions,
        &calc.basis,
        mode,
        pcm.aux.clone(),
    )
    .map_err(LinusError::Input)?;
    let shared = std::sync::Arc::new(std::sync::Mutex::new(state));
    // predicted and the density-fitted PCM: the J build hands its fit coefficients to the PCM
    // term (the density-fitted PCM: and carries its Fock term), crate::two_electron::df::CoulombHook
    if mode != PcmScfMode::Exact {
        let hook = Box::new(ri::PcmCoulombHook(shared.clone()));
        if let HookOutcome::FellBack { reason } = scf.set_coulomb_hook(Some(hook))? {
            return Err(LinusError::Unsupported(format!(
                "solvent.scf = {}: {reason}",
                mode.name()
            )));
        }
    }
    Ok(Box::new(CpuEnvironment {
        term: PcmTerm { pcm: shared },
        mode,
    }))
}

/// The CPU's term of the solvent in one SCF ([`cpu_environment`]).
struct CpuEnvironment {
    term: PcmTerm,
    mode: PcmScfMode,
}

impl ScfEnvironment for CpuEnvironment {
    fn term(&mut self) -> &mut dyn EnvironmentTerm {
        &mut self.term
    }

    fn close(
        self: Box<Self>,
        calc: &CalcContext,
        setup: &ScfSetup,
        scf: &mut dyn ScfLoop,
        r: Result<ScfResult, LinusError>,
    ) -> Result<(ScfResult, SolvationResult), LinusError> {
        let CpuEnvironment { term, mode } = *self;
        if mode != PcmScfMode::Exact {
            scf.set_coulomb_hook(None)?;
        }
        let mut r = r?;
        let so = calc.plan.solvent().expect("a solvent");
        let mut pcm = std::sync::Arc::try_unwrap(term.pcm)
            .unwrap_or_else(|_| panic!("PCM state still shared"))
            .into_inner()
            .expect("PCM state");
        let mut res = pcm.result(so.name(), so.preset(), calc.mol.charge);
        // The density-fitted PCM: 1/2 q . v_n came through the environment term, -1/2 q . v_e through J
        if let Some(rp) = pcm.ri.as_ref().filter(|rp| rp.mode == PcmScfMode::Df) {
            r.e_coulomb -= rp.parts.1;
            r.e_solvation += rp.parts.1;
        }
        // SMD: G_CDS (independent of the density)
        if let Some(d) = &pcm.resolved.smd {
            let c = smd::cds(
                &calc.mol.z,
                &calc.mol.positions,
                d,
                &cds_config(so, setup.purpose, smd::CdsOrder::Energy),
            );
            r.energy += c.energy;
            r.e_solvation += c.energy;
            res.cds = Some(c.energy);
        }
        // the SCF timed the surface terms as "solvent", except the density-fitted PCM's, which ride on
        // the Coulomb builds ("jk"): book those under their own names
        if mode == PcmScfMode::Df {
            let t_iter: f64 = pcm
                .timers
                .entries
                .iter()
                .filter(|e| e.0 != "pcm_setup")
                .map(|e| e.1)
                .sum();
            if let Some(e) = r.timers.entries.iter_mut().find(|e| e.0 == "jk") {
                e.1 -= t_iter;
            }
        }
        r.timers.merge(&pcm.timers);
        pcm.last = None;
        res.state = Some((std::sync::Arc::new(pcm), calc.mol.positions.clone()));
        Ok((r, res))
    }
}

impl CalcContext {
    /// The PCM term of the ground state `gs` (at the current geometry): its SCF's when it was
    /// built for these coordinates, else a new one.
    pub(crate) fn solvent_state(
        &self,
        gs: &GroundState,
        so: &SolventSettings,
    ) -> Result<std::sync::Arc<PcmScf>, LinusError> {
        if let Some((p, c)) = gs.solvation.as_ref().and_then(|s| s.state.as_ref()) {
            if *c == self.mol.positions {
                return Ok(p.clone());
            }
        }
        let resolved = so.resolve(&self.mol.z).map_err(LinusError::Input)?;
        check_dense_memory(self, &resolved, &self.mol.positions)?;
        // derivatives need the exact surface integrals unless the SCF was the density-fitted PCM
        let p = match scf_mode(self, so, gs.purpose())? {
            (PcmScfMode::Df, aux) => PcmScf::with_mode(
                resolved,
                &self.mol.effective_z(),
                &self.mol.positions,
                &self.basis,
                PcmScfMode::Df,
                aux,
            ),
            _ => PcmScf::new(
                resolved,
                &self.mol.effective_z(),
                &self.mol.positions,
                &self.basis,
            ),
        };
        Ok(std::sync::Arc::new(p.map_err(LinusError::Input)?))
    }
}

/// Options of SMD's G_CDS of an SCF converged for `purpose`: `cds_contact_smoothing`, else 0 for
/// single points and 0.1 Å for gradients and Hessians.
pub(crate) fn cds_config(
    so: &SolventSettings,
    purpose: crate::driver::Purpose,
    order: smd::CdsOrder,
) -> smd::CdsConfig {
    let smoothing = so.cds_contact_smoothing().or_else(|| match purpose {
        crate::driver::Purpose::Energy => 0.0,
        _ => 0.1,
    });
    smd::CdsConfig {
        order,
        contact_smoothing: smoothing,
        ..Default::default()
    }
}
