//! Two-electron dispatch: the J/K method of the settings (`jk`: exact, incore, df, rij_snk)
//! chooses the CPU's builders and derivative routines of every order here and nowhere else (a
//! new approximation adds one arm per order). They are the CPU provider's engines of the
//! two-electron families (`crate::backend`); the backend the plan puts a family on answers for
//! it, the device's (its density-fitted engines) or the CPU's:
//!
//! * energies: the SCF's J/K builder (`CalcContext::cpu_scf_jk`; any backend's:
//!   [`Session::build_jk`]) and the end of a converged SCF (`CalcContext::finish_scf_jk`: the
//!   COSX final-grid exchange energy);
//! * gradients: the two-electron gradient (`CalcContext::cpu_gradient_2e`);
//! * second derivatives: which methods have them (`CalcContext::check_second_derivatives`),
//!   the explicit second derivatives and F^x of Hessians (`CalcContext::cpu_hessian_2e`), and the
//!   response J/K of Hessians and polarizabilities (`CalcContext::cpu_response_jk`; any
//!   backend's: `CalcContext::response_jk`).
//!
//! The exact exchange enters as one operator `xop` (`MethodInfo::exchange`; None: no exact
//! exchange), split for the engines by [`exchange_parts`] into a_x K for HF and global hybrids
//! and the [`Operator`] a/r + b erf(omega r)/r of a range-separated hybrid with a_x = 1 (the
//! coefficients in the operator). The SCF's builder gets the operator here
//! (`CalcContext::cpu_scf_jk`).

use crate::derivatives::terms::GradientInputs;
use crate::driver::ground::{GroundState, ScfMemory, ScfSetup};
use crate::driver::options::GridChoice;
use crate::driver::{AuxBasis, CalcContext, ExactStorage, JkMethod, Purpose, Session};
use faer::{Mat, MatRef, Par};
use linus_base::error::{LinusError, MemoryNeed};
use linus_base::terms::exchange_parts;
use linus_base::terms::{JkBuilder, ResponseJk};
use linus_core::timing::Timers;
use linus_grid::GridSpec;
use linus_ints::operator::Operator;
use linus_ints::BasisSet;
use linus_ints::PairSet;
use linus_jk::cosx::final_grid_exchange;
use linus_jk::cosx::{CosxConfig, CosxK};
use linus_jk::df::response::IterPlan;
use linus_jk::df::{CoulombMetric, DfConfig, DfJk, ResolvedDf};
use linus_jk::{CompositeJk, DenseResponseJk, IncoreJk};
use linus_scf::ScfResult;
use std::sync::Arc;
use std::time::Instant;

/// What a response J/K serves ([`CalcContext::response_jk`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ResponseUse {
    /// The CPHF/CPKS of a Hessian: `precision.response`, and in two-level responses a cheaper
    /// J/K for the iterations.
    Hessian,
    /// The electric-field CPHF/CPKS of polarizabilities: double precision, one J/K.
    Polarizability,
}

impl ResponseUse {
    /// Its name in `resolved.record.backend.response`.
    pub(crate) fn name(self) -> &'static str {
        match self {
            ResponseUse::Hessian => "hessian",
            ResponseUse::Polarizability => "polarizability",
        }
    }
}

/// What a response's CPKS kernel serves: the response's purpose, and whether it runs single
/// precision (its J/K's arithmetic: the device's other than double, or the CPU's in single
/// precision), which sets the kernel's arithmetic by the device policy
/// (`crate::driver::policy::DevicePolicy::cpks_xc`). Handed from the response's J/K setup to its
/// kernel explicitly, never through a shared record.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct KernelUse {
    pub purpose: ResponseUse,
    pub single: bool,
}

/// What a response J/K is set up for ([`CalcContext::response_jk`]).
#[derive(Clone, Copy)]
pub(crate) struct ResponseRequest<'a> {
    /// The ground state the response is of.
    pub ground: &'a GroundState,
    /// The occupied orbitals, one matrix per spin channel.
    pub occ: &'a [MatRef<'a, f64>],
    /// The exact exchange.
    pub xop: Option<Operator>,
    /// What the response serves.
    pub purpose: ResponseUse,
}

impl CalcContext {
    /// The Coulomb metric of the auxiliary basis set `aux` for `purpose` ([`CoulombMetric`]):
    /// the geometry's ([`crate::context::IntegralContext::metric`]), built once for the SCF's
    /// fitting and the derivatives' of one call.
    pub(crate) fn df_metric(
        &self,
        aux: &AuxBasis,
        purpose: Purpose,
    ) -> Result<Arc<CoulombMetric>, LinusError> {
        let auxb = self.aux_set(aux, purpose)?;
        self.integrals.metric(aux.for_purpose(purpose), || {
            Ok(CoulombMetric::new(&auxb, self.par())?)
        })
    }

    /// The density-fitting options of the settings (storage) for exchange with range
    /// separation `omega` and the tensors' memory budget `budget` (bytes; an SCF's
    /// `ScfMemory::df_bytes`).
    pub fn df_config(&self, omega: Option<f64>, budget: usize) -> DfConfig {
        DfConfig {
            omega,
            mem_budget: Some(budget),
            precision: self.plan.df_storage(),
            long_range: self.plan.df_long_range(),
            ..DfConfig::default()
        }
    }

    /// The density-fitting options of the SCF's J/K ([`CalcContext::df_config`]) within the
    /// SCF's memory `memory`: the tensors' budget, and with `memory_gib` the whole process's room
    /// for them and their exchange workspace, sized for the SCF's two-spin builds.
    fn scf_df_config(&self, memory: &ScfMemory) -> Result<DfConfig, LinusError> {
        let (na, nb) = self.mol.nalpha_nbeta().map_err(LinusError::Input)?;
        Ok(DfConfig {
            room: memory.room,
            scf_occ: [na, nb],
            ..self.df_config(self.method.omega(), memory.df_bytes())
        })
    }

    /// The DF setup's error as the calculation's ([`linus_jk::df::DfSetupError`]'s conversion):
    /// a budget too small for every storage allowed (under `jk.df_storage` auto, the f64 ones:
    /// a budget or a runner's grant never selects a compressed tensor) is a `host_memory`
    /// resource error that also names the `memory_gib` whose budget holds the tensor (75% of
    /// it, less the reservations of `setup`).
    fn df_setup_error(&self, e: linus_jk::df::DfSetupError, setup: &ScfSetup) -> LinusError {
        let linus_jk::df::DfSetupError::Memory {
            budget,
            ref options,
            ..
        } = e
        else {
            return e.into();
        };
        let auto = self.plan.df_storage() == linus_jk::df::DfPrecision::Auto;
        let gib = |b: f64| b / 1073741824.0;
        let reserved: usize = setup.memory.reserved.iter().map(|r| r.bytes).sum();
        // the smallest memory_gib that holds a storage allowed (crate::driver::needs, as the plan)
        let required = options
            .iter()
            .map(|&(t, ws)| {
                crate::driver::needs::scf_df_required(t, ws, reserved, setup.memory.fixed)
            })
            .min()
            .unwrap_or(0);
        let limit = gib(required as f64);
        let needed = (limit * 10.0).ceil() / 10.0;
        let mem = self.plan.memory();
        let source = match (mem.source(), setup.memory.df_budget.by_available) {
            ("memory_gib", _) => "memory_gib".to_string(),
            (s, true) => format!("the {s}, lowered to the memory free when the SCF started"),
            (s, false) => format!("the {s}"),
        };
        let reserve = if reserved > 0 {
            let what: Vec<&str> = setup.memory.reserved.iter().map(|r| r.what).collect();
            format!(
                ", less {:.2} GiB reserved ({})",
                gib(reserved as f64),
                what.join(", ")
            )
        } else {
            String::new()
        };
        let tensor = if auto { "the f64 tensor" } else { "the tensor" };
        let compressed = if auto {
            "jk.df_storage \"f48\" (compressed: relative error at most 9e-13; auto never chooses it), "
        } else {
            ""
        };
        LinusError::host_memory(
            format!(
                "{e}: the budget is 75% of {source}{reserve}, and memory_gib {needed:.1} or more holds {tensor} \
                 with its exchange workspace and the rest of the process (--memory on the command line; the plan's \
                 memory.required_gib). Other ways: {compressed}jk \"rij_snk\" (--jk rij-snk: RI-J with \
                 seminumerical exchange, no 3-index exchange tensor)"
            ),
            MemoryNeed {
                required_gib: Some(limit),
                available_gib: Some(gib(limit_holding(budget, reserved))),
            },
        )
    }

    /// The memory to reserve before the SCF's tensors are sized: the large allocations alive
    /// with them whose sizes are known up front. Now the PCM's dense surface matrices at their
    /// peak while they are built (after the DF setup; `linus_solvent::pcm::dense_peak_bytes`:
    /// 2 n^2 f64 for C-PCM and COSMO, 6 n^2 for IEF-PCM, SS(V)PE and SMD, for the n points of
    /// the surface, built here to count them, cheap next to the matrices).
    pub(crate) fn scf_memory_reservations(
        &self,
    ) -> Result<Vec<linus_base::memory::Reservation>, LinusError> {
        let mut out = vec![];
        if let Some(so) = self.plan.solvent() {
            let r = so.resolve(&self.mol.z).map_err(LinusError::Input)?;
            // matrices beyond the whole limit: the solvent's own resource error (it names the
            // ways out), before the DF setup fails on a budget the reservation emptied
            crate::solvent::scf::check_dense_memory(self, &r, &self.mol.positions)?;
            let n = linus_solvent::surface::Surface::build(
                &self.mol.positions,
                &r.radii,
                &r.lebedev,
                &r.xi,
                r.discretization,
                r.cut,
            )
            .map_err(LinusError::Input)?
            .len();
            out.push(linus_base::memory::Reservation {
                what: "pcm",
                bytes: linus_solvent::pcm::dense_peak_bytes(r.model, n) as usize,
            });
        }
        Ok(out)
    }

    /// The warning (`df_storage_by_free_memory`) of an SCF (its ground state `gs`) whose
    /// density-fitting storage is not the first choice (the one-sided coefficients of a
    /// range-separated hybrid instead of B and B^omega, or a compressed format set explicitly)
    /// because the guard of an unset `memory_gib` lowered its budget below 75% of the machine's
    /// limit (`resolved.record` reports both; this says why the storage, and the last bits,
    /// depend on the node's load).
    pub(crate) fn df_guard_warning(gs: &GroundState) -> Option<linus_core::Warning> {
        use linus_jk::df::{DfFormat, DfPlan};
        let resolved = gs.df.as_ref()?;
        let b = gs.setup.memory.df_budget;
        let best = matches!(
            resolved.plan,
            DfPlan::InCore {
                format: DfFormat::F64,
                ..
            } | DfPlan::Raw
        ) && !matches!(
            resolved.long_range,
            linus_jk::df::DfLongRangeSource::OneSided
        );
        let gib = |x: usize| x as f64 / 1073741824.0;
        (b.by_available && !best).then(|| linus_core::Warning {
            code: "df_storage_by_free_memory",
            message: format!(
                "the density-fitting tensors are stored as {} ({:.1} GiB): the free memory ({:.1} GiB) \
                 lowered their budget to {:.1} GiB; set memory_gib for the same storage on every node",
                resolved.plan.name(),
                gib(resolved.store_bytes),
                gib(b.available.unwrap_or(0)),
                gib(b.bytes)
            ),
            path: "memory_gib",
        })
    }

    /// The CPU's J/K builder of the J/K method for an SCF set up with `setup` (family `jk` on the
    /// CPU, `crate::backend`), with the exchange operator of a range-separated hybrid, and with
    /// density fitting what its setup resolved to.
    pub(crate) fn cpu_scf_jk(
        &self,
        pairs: &PairSet,
        setup: &ScfSetup,
        timers: &mut Timers,
    ) -> Result<(Box<dyn JkBuilder>, Option<ResolvedDf>), LinusError> {
        let (mut b, resolved): (Box<dyn JkBuilder>, Option<ResolvedDf>) = match self.plan.jk() {
            // where the exact integrals are kept (`jk.exact_storage`; auto: integral-direct)
            JkMethod::Exact => match self.plan.exact_storage() {
                ExactStorage::Incore => (
                    Box::new(self.incore_jk(pairs, &setup.memory, timers)?),
                    None,
                ),
                _ => (
                    Box::new(timers.time("jk_setup", || {
                        linus_jk::exact::FastJk::new(&self.basis, self.plan.int_thresh())
                    })),
                    None,
                ),
            },
            JkMethod::Df { aux } => {
                let auxb = self.aux_set(aux, setup.purpose)?;
                // exchange tensor only for HF and hybrids; range-separated hybrids: the long-range
                // tensor prepared at setup (fails early if it does not fit)
                let need_k = self.method.exchange.is_some();
                let opts = self.scf_df_config(&setup.memory)?;
                let df = timers.time("jk_setup", || -> Result<DfJk, LinusError> {
                    let metric = self.df_metric(aux, setup.purpose)?;
                    DfJk::try_new(&self.basis, pairs, &auxb, metric, need_k, opts)
                        .map_err(|e| self.df_setup_error(e, setup))
                })?;
                // storage chosen for the 3-index tensors (GiB, bytes per element: 8 f64, 6/5/4 compressed)
                let st = &df.stats;
                let bpe = st.bytes as f64 / (st.nrows * st.naux).max(1) as f64;
                timers.resource(
                    "df_store_gib",
                    (st.bytes + st.bytes_lr) as f64 / 1073741824.0,
                );
                timers.resource("df_bytes_per_elem", bpe);
                timers.resource(
                    "df_one_sided",
                    if matches!(st.plan, linus_jk::df::DfPlan::OneSided { .. }) {
                        1.0
                    } else {
                        0.0
                    },
                );
                let resolved = df.resolved();
                (Box::new(df), Some(resolved))
            }
            JkMethod::RijSnk { aux, k_grid } => {
                let auxb = self.aux_set(aux, setup.purpose)?;
                let opts = self.df_config(None, setup.memory.df_bytes());
                let j = timers.time("jk_setup", || -> Result<DfJk, LinusError> {
                    let metric = self.df_metric(aux, setup.purpose)?;
                    Ok(DfJk::try_new(
                        &self.basis,
                        pairs,
                        &auxb,
                        metric,
                        false,
                        opts,
                    )?)
                })?;
                let resolved = j.resolved();
                let k = match k_grid.cosx_levels() {
                    Some(levels) => linus_jk::cosx::scf_builder(
                        &self.basis,
                        &self.mol.z,
                        &self.mol.positions,
                        levels,
                        CosxConfig {
                            incremental: self.plan.scf_algorithm().incremental,
                            ..CosxConfig::scf()
                        },
                    ),
                    None => linus_grid::MolecularGrid::build(
                        &self.mol.z,
                        &self.mol.positions,
                        &k_grid.spec(),
                    )
                    .map(|g| CosxK::new(&self.basis, g, CosxConfig::default())),
                }?;
                (
                    Box::new(CompositeJk {
                        j: Box::new(j),
                        k: Box::new(k),
                    }),
                    Some(resolved),
                )
            }
        };
        set_exchange_operator(b.as_mut(), self.method.exchange)?;
        Ok((b, resolved))
    }

    /// The in-core J/K builder (the full ERI tensor), if it fits the budget of the large integral
    /// tensors (`memory`'s); a [`LinusError::Resource`] that points to exact or df otherwise,
    /// before anything is allocated.
    fn incore_jk(
        &self,
        pairs: &PairSet,
        memory: &ScfMemory,
        timers: &mut Timers,
    ) -> Result<IncoreJk, LinusError> {
        let nbf = self.basis.nbf;
        let budget = memory.df_bytes();
        let need = IncoreJk::bytes_needed(nbf, self.method.omega().is_some());
        if need > budget {
            let gib = |b: usize| b as f64 / 1073741824.0;
            let reserved: usize = memory.reserved.iter().map(|r| r.bytes).sum();
            let limits = MemoryNeed::bytes(
                limit_holding(need, reserved),
                limit_holding(budget, reserved),
            );
            return Err(LinusError::host_memory(format!(
                "jk.exact_storage \"incore\" (jk \"incore\") stores the full four-index integral tensor: {:.3} GiB for {nbf} \
                 basis functions, more than its budget of {:.3} GiB (75% of memory_gb, else of the machine's memory \
                 and within the free memory). Use jk.exact_storage \"direct\" (the default: the same integrals, \
                 integral-direct) or jk \"df\"",
                gib(need),
                gib(budget)
            ), limits));
        }
        Ok(timers.time("jk_setup", || IncoreJk::new(&self.basis, pairs)))
    }

    /// The end of a converged SCF: with RIJ + COSX and a COSX grid scheme, the exchange energy
    /// on the final grid ([`crate::two_electron::finalize`]); nothing for the other methods.
    pub(crate) fn finish_scf_jk(&self, r: &mut ScfResult, par: Par) -> Result<(), LinusError> {
        match self.plan.jk() {
            JkMethod::RijSnk { k_grid, .. } if r.converged => crate::two_electron::finalize(
                r,
                &self.basis,
                &self.mol.z,
                &self.mol.positions,
                k_grid.cosx_levels(),
                self.method.exchange,
                par,
            )
            .map_err(LinusError::Input),
            _ => Ok(()),
        }
    }

    /// The CPU's two-electron gradient of the J/K method (family `gradient_jk` on the CPU,
    /// `crate::backend`) at the ground state of `inputs` (its spin densities, the total density
    /// row-major and the occupied orbitals) with the exact exchange `xop`, on the derivative pairs
    /// `pairs`.
    pub(crate) fn cpu_gradient_2e(
        &self,
        inputs: &GradientInputs<'_>,
        pairs: &PairSet,
        xop: Option<Operator>,
        timers: &mut Timers,
    ) -> Result<Vec<[f64; 3]>, LinusError> {
        let (gs, dtf, par) = (inputs.ground, inputs.dtf, inputs.par);
        let (da, db, ca, cb) = (inputs.da, inputs.db, inputs.ca, inputs.cb);
        let basis = &self.basis;
        let r = &gs.scf;
        let thresh = self.plan.int_thresh().min(1e-12);
        match self.plan.jk() {
            JkMethod::Exact => Ok(timers.time("grad_2e", || {
                linus_jk::exact::gradient::gradient_2e(
                    basis,
                    pairs,
                    dtf,
                    da,
                    db,
                    r.restricted,
                    xop,
                    thresh,
                )
            })),
            JkMethod::Df { aux } => {
                let auxb = self.aux_set(aux, gs.purpose())?;
                // blocked, memory-lean DF gradient (crate::two_electron::df::gradient); range-separated hybrids: the
                // long-range term in Psi4's wK scheme
                let t0 = std::time::Instant::now();
                let occ = &[ca, cb][..if r.restricted { 1 } else { 2 }];
                // the exchange operator with its coefficients (global hybrids: a_x / r)
                let xop = gradient_operator(xop);
                let metric = self.df_metric(aux, gs.purpose())?;
                let g = linus_jk::df::gradient::df_gradient(
                    basis,
                    pairs,
                    &auxb,
                    &metric,
                    dtf,
                    occ,
                    r.restricted,
                    xop,
                    par,
                    timers,
                );
                timers.add("grad_2e", t0.elapsed().as_secs_f64());
                Ok(g)
            }
            JkMethod::RijSnk { aux, k_grid } => {
                let auxb = self.aux_set(aux, gs.purpose())?;
                // RI-J + COSX exchange (the combined operator for range-separated hybrids)
                let (z, positions, gr) =
                    (&self.mol.z, &self.mol.positions, self.plan.grid_response());
                linus_jk::cosx::rij_cosx::rijcosx_gradient(
                    basis,
                    z,
                    positions,
                    pairs,
                    &auxb,
                    gradient_grid(k_grid),
                    r.restricted,
                    da,
                    db,
                    xop,
                    gr,
                    par,
                    timers,
                )
            }
        }
    }

    /// Analytic second derivatives (Hessians, polarizabilities; `what`) exist for the J/K
    /// method: exact and RI, not RIJ + COSX.
    pub(crate) fn check_second_derivatives(&self, what: &str) -> Result<(), LinusError> {
        if self.plan.jk().has_second_derivatives() {
            Ok(())
        } else {
            Err(no_second_derivatives(what))
        }
    }

    /// The CPU's explicit two-electron terms of a Hessian (family `hessian_jk` on the CPU,
    /// `crate::backend`), added to `h` ((3N)^2 row-major), and the
    /// two-electron part of F_s^x C_s = (J^x[D_t] - a_x K^x[D_s]) C_s, added to `fxo[s][x]`,
    /// for the occupied orbitals `occ` of the spin channels (one: a closed shell with the alpha
    /// density, occupation 2; two: the spins of an open shell) of the ground state `gs`, their
    /// densities `ds` and the total density `dtf` (row-major). Exact integrals: four-center second derivatives and
    /// J^x/K^x; RI: [`linus_jk::df::hessian`] (the second derivatives of the RI energy at fixed
    /// orbitals, blocked over auxiliary functions). Exact exchange `xop`; range-separated
    /// hybrids: the exchange with the operator (J with 1/r).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn cpu_hessian_2e(
        &self,
        gs: &GroundState,
        occ: &[MatRef<'_, f64>],
        ds: &[&[f64]],
        dtf: &[f64],
        xop: Option<Operator>,
        pairs: &PairSet,
        h: &mut [f64],
        fxo: &mut [Vec<Mat<f64>>],
        par: Par,
        timers: &mut Timers,
    ) -> Result<(), LinusError> {
        let basis = &self.basis;
        let nch = occ.len();
        let thresh = self.plan.int_thresh().min(1e-12);
        let add = |h: &mut [f64], x: &[f64]| h.iter_mut().zip(x).for_each(|(a, b)| *a += b);
        match self.plan.jk() {
            JkMethod::Exact => linus_jk::exact::hessian::hessian_2e(
                basis, pairs, occ, ds, dtf, xop, thresh, h, fxo, par, timers,
            ),
            JkMethod::Df { aux } => {
                let auxb = self.aux_set(aux, gs.purpose())?;
                // explicit terms blocked over auxiliary functions; range-separated hybrids: the
                // wK exchange
                let f: Vec<f64> = if nch == 1 { vec![2.0] } else { vec![1.0; nch] };
                let metric = self.df_metric(aux, gs.purpose())?;
                let p2e = linus_jk::df::hessian::df_hessian_2e(
                    basis,
                    pairs,
                    &auxb,
                    &metric,
                    occ,
                    &f,
                    xop,
                    self.plan.memory(),
                    par,
                    timers,
                );
                add(h, &p2e.hess);
                for (p, gp) in p2e.fxo.iter().enumerate() {
                    for (fs, gs) in fxo.iter_mut().zip(gp) {
                        fs[p] += gs;
                    }
                }
                linus_base::memory::mem_mark(timers, "explicit_ri");
            }
            JkMethod::RijSnk { .. } => return Err(no_second_derivatives("analytic Hessians")),
        }
        Ok(())
    }

    /// The response J/K of the request `req` (family `response_jk`) on the backend the plan puts
    /// it on (`crate::backend`): the device's, else [`Self::cpu_response_jk`]; `field`: an IR
    /// run's electric-field solve follows ([`crate::backend::Backend::response_jk`]).
    pub(crate) fn response_jk(
        &self,
        req: &ResponseRequest<'_>,
        field: bool,
        pairs: &PairSet,
        par: Par,
        timers: &mut Timers,
    ) -> Result<ResponseSetup, LinusError> {
        self.resolve(crate::driver::backend::Family::ResponseJk, |b| {
            b.response_jk(self, req, field, pairs, par, timers)
        })
    }

    /// The plan's decisions for a density-fitted response J/K of the request `req` with the
    /// auxiliary basis set `aux` (its precision, levels, and the device's arithmetic).
    pub(crate) fn df_response_plan(
        &self,
        req: &ResponseRequest<'_>,
        aux: &linus_ints::BasisSet,
    ) -> crate::driver::plan::ResponsePlan {
        self.plan.response_plan(
            req.purpose,
            &crate::driver::plan::ResponseSizes {
                nbf: self.basis.nbf,
                naux: aux.nbf,
                occ: req.occ.iter().map(|c| c.ncols()).collect(),
                op: exchange_parts(req.xop).1,
            },
        )
    }

    /// The CPU's response J/K of the request `req` at its ground state: for its occupied
    /// orbitals (one matrix per spin channel) and exact exchange (a range-separated operator is
    /// fixed at setup; the fraction of a global one is the `c_k` of [`ResponseJk::fock_occ`]):
    /// FastJk with exact integrals; with density fitting the RI J/K specialized to densities with
    /// occupied factors and projected on them
    /// ([`linus_jk::df::response::DfResponseJk`], memory-lean; wK exchange), its
    /// tensors within the ground state's budget. For Hessians also the plan of the two-level
    /// iteration J/K (None: one level), and the precision the builder runs in
    /// ([`ResponseSetup`]).
    pub(crate) fn cpu_response_jk(
        &self,
        req: &ResponseRequest<'_>,
        pairs: &PairSet,
        par: Par,
        timers: &mut Timers,
    ) -> Result<ResponseSetup, LinusError> {
        let ResponseRequest {
            ground: gs,
            occ,
            xop,
            purpose,
        } = *req;
        let basis = &self.basis;
        match self.plan.jk() {
            JkMethod::Exact => {
                let mut jk = linus_jk::exact::FastJk::new(basis, self.plan.int_thresh().min(1e-12));
                set_exchange_operator(&mut jk, xop)?;
                self.backend.responses().start(
                    purpose.name(),
                    None,
                    "exact exchange: the CPU, double precision".to_string(),
                );
                Ok(ResponseSetup {
                    jk: Box::new(DenseResponseJk::new(Box::new(jk), occ)),
                    plan: None,
                    single_precision: false,
                    kernel: KernelUse {
                        purpose,
                        single: false,
                    },
                })
            }
            JkMethod::Df { aux } => {
                let auxb = self.aux_set(aux, gs.purpose())?;
                // the plan's decisions for this response (its precision and levels)
                let rp = self.df_response_plan(req, &auxb);
                // S, the geometry's (crate::context)
                let s = self.integrals.overlap(basis).clone();
                let timer = match purpose {
                    ResponseUse::Hessian => "hess_cphf_setup",
                    ResponseUse::Polarizability => "pol_cphf_setup",
                };
                let main = timers.time(timer, || -> Result<_, LinusError> {
                    let metric = self.df_metric(aux, gs.purpose())?;
                    Ok(linus_jk::df::response::DfResponseJk::new_op(
                        basis,
                        pairs,
                        &auxb,
                        &metric,
                        occ,
                        s.as_ref(),
                        exchange_parts(xop).1,
                        par,
                    )
                    .configured(rp.single, self.plan.memory().response_slot_bytes()))
                })?;
                self.integrals.release_metrics();
                let plan = rp.iteration_aux.as_deref().and_then(|name| {
                    linus_jk::df::response::iteration_plan(
                        &main,
                        |a| self.plan.load_basis(a).ok(),
                        &self.mol,
                        name,
                        s,
                    )
                });
                // the CPU's J/K; a device CPKS kernel follows its precision
                self.backend.responses().start(
                    purpose.name(),
                    None,
                    format!(
                        "the CPU's response J/K in {} precision",
                        if main.single { "single" } else { "double" }
                    ),
                );
                Ok(ResponseSetup {
                    single_precision: main.single,
                    kernel: KernelUse {
                        purpose,
                        single: main.single,
                    },
                    jk: Box::new(main),
                    plan,
                })
            }
            JkMethod::RijSnk { .. } => Err(no_second_derivatives("response properties")),
        }
    }
}

impl Session {
    /// An SCF's J/K builder at the current geometry on the backend the plan puts the SCF's J/K
    /// on, set up as the last converged SCF was (its purpose's auxiliary basis, its memory budget;
    /// before any SCF: an energy's), with the exchange operator of a range-separated hybrid, and
    /// with density fitting what its setup resolved to.
    pub fn build_jk(
        &self,
        pairs: PairSet,
        timers: &mut Timers,
    ) -> Result<(Box<dyn JkBuilder>, Option<ResolvedDf>), LinusError> {
        let setup = match &self.ground {
            Some(g) => g.setup.clone(),
            None => self.cx.scf_setup(crate::driver::Purpose::Energy)?,
        };
        let cx = &self.cx;
        let (jk, df) = cx.resolve(crate::driver::backend::Family::Jk, |b| {
            b.scf_jk(cx, &pairs, &setup, timers)
        })?;
        Ok((jk.into_builder(), df))
    }
}
/// The response J/K of a calculation ([`CalcContext::response_jk`]).
pub(crate) struct ResponseSetup {
    /// The builder of the response's products (the main one of a two-level response).
    pub jk: Box<dyn ResponseJk>,
    /// The plan of a cheaper J/K for the iterations of a two-level response (Hessians with
    /// density fitting; None: one level).
    pub plan: Option<IterPlan>,
    /// The builder computes in single precision.
    pub single_precision: bool,
    /// What the response's CPKS kernel serves (its arithmetic follows the J/K's).
    pub kernel: KernelUse,
}

/// Sets the exchange operator `xop` on the J/K builder `b` when it is not a global fraction of
/// 1/r (range-separated hybrids; a no-op otherwise); a builder that cannot apply it is an error
/// (an internal one: every J/K method applies it).
pub(crate) fn set_exchange_operator<B: JkBuilder + ?Sized>(
    b: &mut B,
    xop: Option<Operator>,
) -> Result<(), LinusError> {
    if let (_, Some(op)) = exchange_parts(xop) {
        if !b.set_exchange_operator(op) {
            return Err(LinusError::Unsupported(format!(
                "{}: this J/K builder cannot apply the range-separated exchange operator",
                b.name()
            )));
        }
    }
    Ok(())
}

/// The exchange operator of the DF gradients with its coefficients ([`exchange_parts`]; a global
/// hybrid's a_x / r).
pub(crate) fn gradient_operator(xop: Option<Operator>) -> Operator {
    let (a_x, op) = exchange_parts(xop);
    op.unwrap_or(Operator::Attenuated {
        coulomb: a_x,
        long_range: 0.0,
        omega: 0.0,
    })
}

fn no_second_derivatives(what: &str) -> LinusError {
    LinusError::Unsupported(format!(
        "{what} are implemented for jk exact and df, not rij_snk; use jk=\"df\" (--jk df)"
    ))
}

/// The grid and fitting of the COSX gradient for a `k_grid` choice: the final-energy grid
/// with overlap fitting for the COSX scheme, else the SCF grid without fitting.
pub fn gradient_grid(k_grid: &GridChoice) -> (GridSpec, bool) {
    match k_grid.cosx_levels() {
        Some(levels) => (GridSpec::cosx(levels[2]), true),
        None => (k_grid.spec(), false),
    }
}

/// The end of a converged RIJ + COSX SCF: with a COSX grid scheme (`levels` = the first SCF, last
/// SCF and final energy grid levels, `GridChoice::cosx_levels`) the exchange energy of `r` is
/// replaced by its value on the final grid (as ORCA does after a RIJCOSX SCF;
/// [`final_grid_exchange`]) for the exact exchange `xop` (a_x K, or the range-separated operator
/// alpha/r + beta erfc(omega r)/r with a_x = 1; [`linus_base::terms::exchange_parts`]); the time
/// goes to the "k_final" timer. A single COSX grid (`levels` None) leaves `r` as it is.
pub fn finalize(
    r: &mut linus_scf::ScfResult,
    basis: &BasisSet,
    z: &[u32],
    positions: &[[f64; 3]],
    levels: Option<[u8; 3]>,
    xop: Option<Operator>,
    par: Par,
) -> Result<(), String> {
    let Some(levels) = levels else {
        return Ok(());
    };
    let (a_x, op) = linus_base::terms::exchange_parts(xop);
    let op = op.unwrap_or(Operator::Coulomb);
    if a_x == 0.0 || (op.coulomb() == 0.0 && !op.has_erf()) {
        return Ok(());
    }
    let t0 = Instant::now();
    let ex = final_grid_exchange(
        basis,
        z,
        positions,
        levels[2],
        CosxConfig {
            xop: op,
            ..CosxConfig::default()
        },
        &r.d_a,
        &r.d_b,
        r.restricted,
        a_x,
        par,
    )?;
    r.energy += ex - r.e_exchange;
    r.e_exchange = ex;
    r.timers.add("k_final", t0.elapsed().as_secs_f64());
    Ok(())
}

/// The memory limit (bytes; `memory_gib` x 2^30) whose budget for the large integral tensors holds
/// `bytes` with `reserved` bytes reserved beside them: the budget is 75% of the limit less the
/// reservations ([`linus_base::memory::MemoryLimit::df_budget`]). A memory failure's
/// `required_gib` and `available_gib` (qc*'s `ErrorInfo.details`) are both in these terms, so
/// a dispatcher compares them and resubmits with `memory_gib` at `required_gib`.
fn limit_holding(bytes: usize, reserved: usize) -> f64 {
    (bytes + reserved) as f64 * 4.0 / 3.0
}

#[cfg(test)]
mod tests {
    use crate::driver::options::Auto;
    use crate::driver::options::JkSettings;
    use crate::driver::{CalcOptions, ExactStorage, JkMethod, Session};
    use crate::Structure;
    use linus_base::error::LinusError;

    const WATER: &str = "3\nwater\nO 0.0 0.0 0.12\nH 0.0 0.76 -0.47\nH 0.0 -0.76 -0.47\n";

    /// Exact storage incore refuses a tensor beyond its budget before allocating, with a resource error;
    /// within it, it runs.
    #[test]
    fn incore_checks_its_size() {
        let mol = Structure::from_xyz_str(WATER, 0, 1).unwrap();
        let o = |gb: f64| CalcOptions {
            method: Some("hf".into()),
            basis: Some("sto-3g".into()),
            jk: JkSettings {
                exact_storage: Some(ExactStorage::Incore),
                ..JkMethod::Exact.into()
            },
            threads: Some(1),
            memory_gib: Some(Auto::Value(gb)),
            ..Default::default()
        };
        // sto-3g water: 7 functions, 7^4 * 8 * 1.125 = 21.6 kB
        let e = Session::new(mol.clone(), o(1e-5)).unwrap().energy();
        assert!(
            matches!(&e, Err(LinusError::Resource { code: Some("host_memory"), message: m, .. }) if m.contains("jk.exact_storage \"direct\"")),
            "{e:?}"
        );
        assert!(Session::new(mol, o(1e-3)).unwrap().energy().is_ok());
    }

    /// The PCM's dense matrices are reserved before the DF setup sizes its tensors: the budget
    /// is 75% of memory_gb less the reservation (2 n^2 f64 for C-PCM, 6 n^2 for IEF-PCM), the
    /// free memory is not read with memory_gb set, and without a solvent nothing is reserved.
    #[test]
    fn pcm_reserves_memory_before_the_df_setup() {
        use crate::solvent::SolventSettings;
        use linus_solvent::PcmModel;
        let mol = Structure::from_xyz_str(WATER, 0, 1).unwrap();
        let gb = 2.0;
        let budget75 = (gb * 1073741824.0) as usize / 4 * 3;
        for (model, mats) in [
            (None, 0),
            (Some(PcmModel::Cpcm), 2),
            (Some(PcmModel::Iefpcm), 6),
        ] {
            let o = CalcOptions {
                method: Some("hf".into()),
                basis: Some("def2-svp".into()),
                jk: JkMethod::Df {
                    aux: "def2-universal-jkfit".into(),
                }
                .into(),
                memory_gib: Some(Auto::Value(gb)),
                solvent: model.map(|model| SolventSettings {
                    eps: Some(Some(78.4)),
                    model: Some(model),
                    ..Default::default()
                }),
                ..Default::default()
            };
            let mut calc = Session::new(mol.clone(), o).unwrap();
            calc.energy().unwrap();
            let gs = calc.ground.clone().unwrap();
            let mem = &gs.setup.memory;
            let reserved: usize = mem.reserved.iter().map(|r| r.bytes).sum();
            if mats == 0 {
                assert!(mem.reserved.is_empty());
            } else {
                assert_eq!(mem.reserved.len(), 1);
                assert_eq!(mem.reserved[0].what, "pcm");
                let n = calc.solvation().unwrap().npoints;
                assert_eq!(reserved, mats * n * n * 8, "{model:?}");
            }
            let b = mem.df_budget;
            assert_eq!(
                (b.bytes, b.by_available, b.available),
                (budget75 - reserved, false, None)
            );
            assert_eq!(gs.df.as_ref().unwrap().budget_bytes, budget75 - reserved);
        }
    }

    /// A memory limit too small for the f64 exchange tensor and the rest of the process never
    /// selects a compressed one under `jk.df_storage` auto: it fails with `host_memory` before any
    /// work, naming the `memory_gib` that holds it and the compressed storage as a way out; that
    /// limit runs it in f64, bitwise as a large one. (That a compressed storage set explicitly fits
    /// a smaller budget is the DF setup's test, `linus_jk::df`.)
    #[test]
    fn df_storage_auto_is_f64_or_host_memory() {
        let mol = Structure::from_xyz_str(WATER, 0, 1).unwrap();
        let o = |gib: f64| {
            CalcOptions::from_json(serde_json::json!({
                "method": "hf", "basis": "def2-svp", "threads": 1, "memory_gib": gib,
                "jk": {"kind": "df", "df_storage": "auto"},
            }))
            .unwrap()
        };
        let stored = |c: &Session| c.ground.as_ref().unwrap().df.clone().unwrap().plan.name();
        let mut big = Session::new(mol.clone(), o(1.0)).unwrap();
        let e64 = big.energy().unwrap().energy;
        assert_eq!(stored(&big), "B");
        let e = Session::new(mol.clone(), o(0.05))
            .unwrap()
            .energy()
            .unwrap_err();
        assert_eq!(
            (e.kind(), e.code()),
            ("resource", Some("host_memory")),
            "{e}"
        );
        let msg = e.to_string();
        assert!(msg.contains("jk.df_storage \"f48\""), "{msg}");
        let needed = e.memory_need().unwrap().required_gib.unwrap();
        let mut enough = Session::new(mol, o(needed)).unwrap();
        assert_eq!(enough.energy().unwrap().energy, e64);
        assert_eq!(stored(&enough), "B");
    }
}
