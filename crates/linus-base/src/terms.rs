//! The method families' terms: the traits through which the SCF (linus-scf's `run_scf`) and the
//! CPHF/CPKS responses (its `ResponseOperator`) call the J/K, XC and environment families, and
//! the data they exchange. The algorithms take trait objects and the families implement them
//! (the CPU engines in linus-jk, linus-dft and linus-solvent; the device engines in the facade's
//! `backend::gpu`), so that neither side names the other's crate.
//!
//! - J and K: [`JkBuilder`] (J and K of full densities, for the SCF; a [`JkRequest`] may carry
//!   [`DensityFactors`]) and [`ResponseJk`] (J and K of first-order densities with occupied
//!   factors, projected on the occupied orbitals, for CPHF/CPKS); [`CoulombHook`], the
//!   density-fitted PCM's hook on a density-fitted J build; [`exchange_parts`], the method's
//!   exact exchange as the builders apply it;
//! - XC: [`XcPotential`] (the SCF's E_xc and potential at an [`XcDensity`]) and
//!   [`ResponseKernel`] (the CPKS kernel's products);
//! - the environment (an implicit solvent): [`EnvironmentTerm`] (the SCF's reaction field) and
//!   [`EnvironmentResponse`] (its linear response).
//!
//! Builds are fallible: a device backend reports a failure after its setup (out of device
//! memory, a driver fault) as [`LinusError::Backend`], which stops the calculation; the CPU
//! engines always succeed.
//!
//! Incremental builds: a term may build from its own history instead of from the density alone
//! (COSX's K of D_n - D_{n-1} added to the last K, the PCM's charges predicted from the last
//! iteration's potential). Each such term says so after a build (`built_incrementally`) and makes
//! its next build of the density alone when asked (`reset_incremental`), so the SCF can end on
//! one full build: the returned Fock matrix is then the operator of the returned density
//! (decision 25's one-build rule) and the same however the SCF got there.

use crate::error::LinusError;
use faer::{Mat, MatRef, Par};
use linus_core::timing::Timers;
use linus_ints::operator::Operator;

// ------------------------------------------------------------------ J and K

pub struct JkRequest<'a> {
    /// Symmetric AO density matrices.
    pub dens: &'a [Mat<f64>],
    /// Optional factors of the densities (lets DF-K work in the occupied space); the densities
    /// stay authoritative, so a builder may ignore factors it does not use.
    pub factors: Option<DensityFactors<'a>>,
    pub want_j: bool,
    pub want_k: bool,
}

/// Factors of the densities of a [`JkRequest`], D_k = L_k R_k^T (n x m_k each).
#[derive(Clone, Copy)]
pub enum DensityFactors<'a> {
    /// L_k = R_k = C_k, D_k = C_k C_k^T (an SCF's occupied orbitals): the case the builders
    /// implement.
    Symmetric(&'a [Mat<f64>]),
    /// Different left and right factors (transition or response densities, e.g. for a GPU
    /// backend that builds K from factor pairs). No builder uses them yet: they fall back to
    /// the densities.
    Pair {
        left: &'a [Mat<f64>],
        right: &'a [Mat<f64>],
    },
}

impl<'a> DensityFactors<'a> {
    /// The factors C_k of symmetric factors (None for pairs).
    pub fn symmetric(&self) -> Option<&'a [Mat<f64>]> {
        match *self {
            DensityFactors::Symmetric(c) => Some(c),
            DensityFactors::Pair { .. } => None,
        }
    }
}

impl<'a> JkRequest<'a> {
    /// J and K for `dens`, no factors.
    pub fn new(dens: &'a [Mat<f64>], want_j: bool, want_k: bool) -> Self {
        JkRequest {
            dens,
            factors: None,
            want_j,
            want_k,
        }
    }
}

#[derive(Default)]
pub struct JkOutput {
    pub j: Vec<Mat<f64>>,
    /// Exchange matrices: K of the builder's exchange operator ([`JkBuilder::set_exchange_operator`];
    /// 1/r unless set).
    pub k: Vec<Mat<f64>>,
    /// The exchange energies E_K of the densities (the operator's coefficients included, as in
    /// `k`; the SCF applies a_x and the sign) when they are not 1/2 D_k . K_k: builders whose
    /// energy is defined first and K is its derivative without being linear in D (compensated
    /// grid-K: K = K\[D0\] + K^grid\[D - D0\]). None: 1/2 D_k . K_k.
    pub k_energy: Option<Vec<f64>>,
}

pub trait JkBuilder: Send {
    fn name(&self) -> String;
    /// J and K of the request's densities (a device builder's failure is an error; the CPU
    /// builders always succeed).
    fn compute(&mut self, req: &JkRequest, par: Par) -> Result<JkOutput, LinusError>;
    /// True when the build cost falls with |D| (integral screening on the density), so the SCF
    /// should build J and K incrementally from difference densities (the result is linear in D
    /// and densities without factors are accepted).
    fn incremental_capable(&self) -> bool {
        false
    }
    /// J/K of a difference density D_n - D_{n-1} in an incremental Fock build (builders may
    /// screen differently than in `compute`).
    fn compute_delta(&mut self, req: &JkRequest, par: Par) -> Result<JkOutput, LinusError> {
        self.compute(req, par)
    }
    /// Make the next build a full one (builders with internal incremental state, e.g. COSX).
    fn reset_incremental(&mut self) {}
    /// True when the last build used the builder's own incremental state (COSX's K of a
    /// difference density added to the last K) rather than the density alone. The SCF's own
    /// incremental builds (`incremental_capable`, `compute_delta`) are the SCF's to count.
    fn built_incrementally(&self) -> bool {
        false
    }
    /// Builders with a sequence of approximations (COSX: SCF grids of increasing size) move to
    /// the next one here and return true; false when already on the last (the default). The
    /// SCF calls this once its orbital gradient is small enough, rebuilds K at the same density
    /// and carries its DIIS subspace over to the new approximation.
    fn next_approximation(&mut self) -> bool {
        false
    }
    /// The exchange operator a/r + b erf(omega r)/r of a range-separated hybrid (range-separated
    /// exact exchange in one pass): a builder that applies it returns true, and its `k` is from
    /// then on the exchange matrix of that operator, coefficients included. Default: false (the
    /// builder cannot; the SCF and responses of range-separated hybrids need it).
    fn set_exchange_operator(&mut self, _op: Operator) -> bool {
        false
    }
    /// HOOK (solvent, the density-fitted PCM): install (Some) or remove (None) a
    /// [`CoulombHook`] on a density-fitted Coulomb build: `Installed`
    /// when the builder takes it (RI-J builders), `FellBack` with the reason when it cannot, an
    /// error when its setup for the hook failed otherwise (a device's numerical or library
    /// failure). Default: not density-fitted.
    fn set_coulomb_hook(
        &mut self,
        _hook: Option<Box<dyn CoulombHook>>,
    ) -> Result<HookOutcome, LinusError> {
        Ok(HookOutcome::FellBack {
            reason: "the J builder is not density-fitted".into(),
        })
    }
}

/// J and K of CPHF/CPKS first-order densities projected on the occupied orbitals: all that the
/// response equations of Hessians and polarizabilities need of the two-electron operator. For
/// spin channels s with occupied orbitals C_s (fixed at setup) and densities d given by their
/// factors A_{s,d} (n x o_s), dD_{s,d} = A_{s,d} C_s^T + C_s A_{s,d}^T, the product is
///
///   G_{s,d} = (c_j J[sum_s' dD_{s',d}] - c_k K[dD_{s,d}]) C_s      (n x o_s).
///
/// One channel is a closed shell (dD the alpha density, c_j = 2), two are the spins of an open
/// shell (c_j = 1). Implementations: `DenseResponseJk` over any SCF [`JkBuilder`], and the
/// memory-lean RI `DfResponseJk` (the J/K family's); the facade's `CalcContext::response_jk`
/// picks one. The exchange operator (range-separated hybrids) is fixed at setup.
pub trait ResponseJk: Send {
    fn name(&self) -> String;
    /// G_{s,d} as `[s][d]` for the factors `a[s][d]` (K only when `c_k` is not 0); a device
    /// builder's failure is an error.
    fn fock_occ(
        &mut self,
        a: &[Vec<Mat<f64>>],
        c_j: f64,
        c_k: f64,
        par: Par,
    ) -> Result<Vec<Vec<Mat<f64>>>, LinusError>;
    /// Free stored data that [`ResponseJk::restore`] rebuilds while the builder is idle (the main
    /// J/K of a two-level response during the CPKS iterations).
    fn release(&mut self);
    /// Rebuild what [`ResponseJk::release`] freed.
    fn restore(&mut self, par: Par);
    /// The builder's own timers, if it keeps any (merged into the caller's report).
    fn timers(&self) -> Option<Timers>;
    /// Go on serving the products of another response of the same occupied orbitals and
    /// exchange (an IR run's electric-field CPHF after its Hessian's), recorded in
    /// `resolved.record.backend.response` under `purpose` from here on, with its own timers from zero:
    /// true when the builder can (it computes in double precision, as that response needs),
    /// false when the response needs a builder of its own (the default).
    fn serve(&mut self, _purpose: &'static str) -> bool {
        false
    }
}

/// Max-abs CPHF/CPKS residual at which solves with single-precision response J/K stall: the
/// round-off of the exchange (K C relative error ~4e-6, times the exact-exchange fraction).
/// def2-SVP Hessians of L-lactide, caffeine and u1an39 (RKS and UKS) level off after 6-13
/// iterations at 1-7e-8 with HF and range-separated hybrids (~1e-6 of the largest right-hand side
/// element) and at 3e-9 to 2e-8 with B3LYP, then creep up with further iterations (to ~1e-7 after
/// 25-100); this is the upper end (the density-fitted response J/K's `SINGLE_MIN_TOL` is ten times
/// it).
pub const SINGLE_FLOOR: f64 = 1e-7;

/// HOOK (solvent, the density-fitted PCM, the solvent's `ri`): the density-fitted J build
/// (`DfJk::coulomb`) passes the fit coefficients c_k = V^-1 gamma_k of the densities of each
/// request and expands c_k + e_k with the returned e_k, so J_k = sum_P (mn|P) (c_k + e_k): a
/// Coulomb-like term that depends on the fitted density (the PCM reaction field through the
/// auxiliary basis) rides on the J expansion. `solve` applies V^-1 in place.
pub trait CoulombHook: Send + Sync {
    fn extra_coefficients(&mut self, c: &[Vec<f64>], solve: &dyn Fn(&mut [f64])) -> Vec<Vec<f64>>;

    /// The hook's own form for a backend's J build, which only that backend reads (the GPU's PCM
    /// term takes a device J build's fit coefficients on the device and adds nothing to J,
    /// the facade's `backend`); None (the default): the hook receives them on the host through
    /// [`CoulombHook::extra_coefficients`].
    fn device_form(&mut self) -> Option<&mut dyn std::any::Any> {
        None
    }
}

/// What a J build did with a [`CoulombHook`] it was offered (`JkBuilder::set_coulomb_hook`). A
/// failure that is neither (a numerical or library failure of a device setup) is the call's
/// error instead.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum HookOutcome {
    /// The build took the hook (or, offered none, removed its own).
    Installed,
    /// The build cannot take it: it is not density-fitted, or a device build's setup for it did
    /// not fit or is not covered. `reason` says which, for the record.
    FellBack { reason: String },
}

/// The exact exchange `xop` of a method (`MethodInfo::exchange`; None: none) as the J/K
/// builders apply it, (a_x, operator): K is scaled by a_x and built with the builder's exchange
/// operator. A global operator a/r (HF: a = 1; global hybrids) is a_x = a with plain 1/r
/// (None); a range-separated one a/r + b erf(omega r)/r is a_x = 1 with the operator, its
/// coefficients included ([`JkBuilder::set_exchange_operator`]). No exact exchange: a_x = 0.
pub fn exchange_parts(xop: Option<Operator>) -> (f64, Option<Operator>) {
    match xop {
        None => (0.0, None),
        Some(op) => match op.global_fraction() {
            Some(a) => (a, None),
            None => (1.0, Some(op)),
        },
    }
}

// ------------------------------------------------------------------ XC

/// The exchange-correlation term of the SCF: E_xc and the XC potential matrices.
pub trait XcPotential: Send {
    /// (E_xc, V_a, V_b) at the SCF state `d` (restricted: the potential of the total density,
    /// returned for both spins). A device backend's failure is an error
    /// ([`LinusError::Backend`]); the CPU integrator always succeeds.
    fn compute(
        &mut self,
        d: &XcDensity<'_>,
        par: Par,
    ) -> Result<(f64, Mat<f64>, Mat<f64>), LinusError>;

    /// Whether the term keeps something of the last state it was evaluated at that is read after
    /// the SCF (VV10's last energy and statistics, for the report): a probe's rotated builds are
    /// then followed by one at the converged state. Default false.
    fn keeps_state(&self) -> bool {
        false
    }
}

/// The state the XC term is evaluated at: the spin densities, and the occupied orbitals that
/// give them when the SCF has them. A backend may form the density on its grid from the
/// orbitals instead of the matrices (a GPU: rho = sum_i n_i |phi C_i|^2); the CPU integrator
/// uses the matrices.
pub struct XcDensity<'a> {
    pub da: &'a Mat<f64>,
    pub db: &'a Mat<f64>,
    /// Closed shell: D_a = D_b, one channel of orbitals.
    pub restricted: bool,
    /// The occupied orbitals that give the densities; None when the density does not come from
    /// orbitals (a guess).
    pub orbitals: Option<OccupiedOrbitals<'a>>,
}

/// Occupied orbitals per channel (nbf x nocc; one channel for a closed shell, alpha and beta
/// otherwise) and their occupations, D_s = C_s diag(n_s) C_s^T.
#[derive(Clone, Copy)]
pub struct OccupiedOrbitals<'a> {
    pub c: &'a [Mat<f64>],
    pub n: &'a [Vec<f64>],
}

impl<'a> XcDensity<'a> {
    /// Spin densities without orbitals.
    pub fn of(da: &'a Mat<f64>, db: &'a Mat<f64>, restricted: bool) -> Self {
        XcDensity {
            da,
            db,
            restricted,
            orbitals: None,
        }
    }
}

/// The XC kernel of a CPKS response (Hessians, polarizabilities): dV_s\[dD\] C_s for first-order
/// densities dD_s = A_s C_s^T + C_s A_s^T given by their factors `a[s][k]` (nbf x nocc_s), at the
/// ground state of the occupied orbitals `c_o[s]`, as `[s][k]` (one channel: a closed shell,
/// the potential of the total first-order density 2 dD; two: the spins of an open shell).
/// The response's `ResponseOperator` takes its kernel through this trait: the CPU
/// integrator (`XcIntegrator::kernel_contract_occ`) or a device kernel set up by the backend
/// (the facade's `backend`, family `response_xc`; it fixes the ground state at its setup).
pub trait ResponseKernel: Send + Sync {
    /// The products; a device kernel's failure is an error ([`LinusError::Backend`]).
    fn contract_occ(
        &self,
        c_o: &[MatRef<'_, f64>],
        a: &[Vec<Mat<f64>>],
    ) -> Result<Vec<Vec<Mat<f64>>>, LinusError>;

    /// The kernel's own times of its calls so far by phase (a device kernel's `xc_dev_*`; None:
    /// none beyond the response's).
    fn timers(&self) -> Option<Timers> {
        None
    }
}

// ------------------------------------------------------------------ the environment

/// A density-dependent environment term of the SCF (an implicit solvent's reaction field): an
/// energy and a Fock contribution at every iterate, which the SCF keeps as their own energy
/// component (`ScfResult::e_solvation`) and timer ([`EnvironmentTerm::name`]).
pub trait EnvironmentTerm: Send {
    /// The name of its timer.
    fn name(&self) -> &'static str;
    /// (E_env, V_env) at the spin densities `d`: V_env is the same for both spins (a potential
    /// of the total density); None when the term has no Fock matrix of its own (the density-fitted
    /// PCM, whose Fock term rides on the Coulomb build). An error (the device PCM's `backend`
    /// error) stops the SCF with it.
    fn compute(
        &mut self,
        d: &XcDensity<'_>,
        par: Par,
    ) -> Result<(f64, Option<Mat<f64>>), LinusError>;

    /// True when the last [`EnvironmentTerm::compute`]'s Fock matrix was not of the density
    /// alone (the PCM's `predicted` mode: the charges of a potential predicted from the last
    /// iteration's; its incremental passes). Default: false.
    fn built_incrementally(&self) -> bool {
        false
    }

    /// Make the next build of the density alone as far as the term can at a changed density
    /// (no incremental passes); at the density of its last build, exactly of the density.
    fn reset_incremental(&mut self) {}

    /// The term's own form for a backend's loop, which only that backend reads (the GPU's PCM
    /// term on the device loop's buffers, the facade's `backend`); None (the default): a loop runs
    /// [`EnvironmentTerm::compute`] on host copies.
    fn device_form(&mut self) -> Option<&mut dyn std::any::Any> {
        None
    }
}

/// An environment's linear response in a CPHF/CPKS (implicit solvent: the reaction field of the
/// first-order density): dV\[dD\] C_s for the first-order densities dD_s = A_s C_s^T + C_s A_s^T
/// given by their factors `a[s][k]`, with dV the environment's Fock response to the total
/// first-order density (one channel: a closed shell, dD = 2 dD_alpha; two: the spins summed),
/// as `[s][k]` (n x nocc_s). Implemented by the solvent (the solvent's `PcmScf`).
pub trait EnvironmentResponse: Send + Sync {
    /// The timer of its products in the operator's timers (for example `cphf_solvent`).
    fn timer(&self) -> &'static str;

    /// The products for the occupied orbitals `c_occ[s]` of the channels.
    fn induced_fock_occ(
        &self,
        c_occ: &[MatRef<'_, f64>],
        a: &[Vec<Mat<f64>>],
        par: Par,
    ) -> Result<Vec<Vec<Mat<f64>>>, LinusError>;
}
