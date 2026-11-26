//! Backend-agnostic types: the [`XcKernel`] trait, inputs, outputs, errors.
//!
//! Everything here is independent of libxc so that native Rust or GPU kernels
//! can implement [`XcKernel`] later and be mixed with libxc components inside a
//! [`crate::Functional`].

use std::fmt;

/// Errors from building or evaluating a functional.
#[derive(Debug, thiserror::Error)]
pub enum XcError {
    #[error("unknown functional '{0}' (not a linus alias, libxc name or libxc id)")]
    UnknownFunctional(String),
    #[error("cannot parse functional spec '{spec}': {msg}")]
    Parse { spec: String, msg: String },
    #[error("libxc could not initialize functional {id} ('{name}'): error code {code}")]
    Init { id: i32, name: String, code: i32 },
    #[error("derivative order {order} is not available for '{name}' (max {max})")]
    Order {
        name: String,
        order: usize,
        max: usize,
    },
    #[error("input '{0}' is required by this functional but was not given")]
    MissingInput(&'static str),
    #[error("input '{name}' has length {len}, expected {expected} ({np} points x {dim})")]
    InputLength {
        name: &'static str,
        len: usize,
        expected: usize,
        np: usize,
        dim: usize,
    },
    #[error("output '{0}' is required for this order/functional but was not given")]
    MissingOutput(&'static str),
    #[error("output '{name}' has length {len}, expected {expected}")]
    OutputLength {
        name: &'static str,
        len: usize,
        expected: usize,
    },
    #[error("unsupported: {0}")]
    Unsupported(String),
}

pub type Result<T> = std::result::Result<T, XcError>;

/// Rung of a functional: which density ingredients it needs.
/// Ordered, so the family of a mixture is the max over its components.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Family {
    /// No semilocal part at all (pure Hartree-Fock exchange): nothing to evaluate.
    Hf,
    /// Needs `rho`.
    Lda,
    /// Needs `rho`, `sigma`.
    Gga,
    /// Needs `rho`, `sigma` and `tau` and/or `lapl` (see `needs_tau`/`needs_lapl`).
    Mgga,
}

impl fmt::Display for Family {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Family::Hf => "HF",
            Family::Lda => "LDA",
            Family::Gga => "GGA",
            Family::Mgga => "meta-GGA",
        })
    }
}

/// Exact-exchange content, in libxc's convention:
/// `K_total = alpha * K(full range) + beta * K_sr(omega)`.
/// At short range the exact-exchange fraction is `alpha + beta`, at long
/// range it is `alpha`. Global hybrids have `beta = omega = 0`.
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct ExactExchange {
    pub alpha: f64,
    pub beta: f64,
    pub omega: f64,
    pub kernel: RangeSeparation,
}

impl ExactExchange {
    /// True if any exact exchange is needed.
    pub fn is_hybrid(&self) -> bool {
        self.alpha != 0.0 || self.beta != 0.0
    }
    /// True for range-separated hybrids (short-range operator needed).
    pub fn is_range_separated(&self) -> bool {
        self.beta != 0.0 && self.omega != 0.0
    }
}

/// Operator used for the short-range exact-exchange part.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum RangeSeparation {
    /// Global hybrid or no exact exchange.
    #[default]
    None,
    /// `erfc(omega r)/r` short-range Coulomb (CAM, wB97 family, ...).
    Erf,
    /// `exp(-omega r)/r` Yukawa (rare; e.g. LCY-PBE).
    Yukawa,
}

/// VV10 non-local correlation parameters. libxc does not evaluate VV10; the
/// host code must add it using these parameters (times `scale`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Vv10 {
    pub b: f64,
    pub c: f64,
    /// Weight of the component that carries VV10 (1 for the named functionals).
    pub scale: f64,
}

/// One output array of an evaluation, named as in libxc.
/// Each holds `np * dim` values, interleaved per grid point (libxc layout).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(usize)]
pub enum Out {
    Zk = 0,
    Vrho,
    Vsigma,
    Vlapl,
    Vtau,
    V2rho2,
    V2rhosigma,
    V2rholapl,
    V2rhotau,
    V2sigma2,
    V2sigmalapl,
    V2sigmatau,
    V2lapl2,
    V2lapltau,
    V2tau2,
}

/// Number of [`Out`] variants.
pub const N_OUT: usize = 15;

impl Out {
    pub const ALL: [Out; N_OUT] = [
        Out::Zk,
        Out::Vrho,
        Out::Vsigma,
        Out::Vlapl,
        Out::Vtau,
        Out::V2rho2,
        Out::V2rhosigma,
        Out::V2rholapl,
        Out::V2rhotau,
        Out::V2sigma2,
        Out::V2sigmalapl,
        Out::V2sigmatau,
        Out::V2lapl2,
        Out::V2lapltau,
        Out::V2tau2,
    ];

    /// libxc name of the array.
    pub fn name(self) -> &'static str {
        [
            "zk",
            "vrho",
            "vsigma",
            "vlapl",
            "vtau",
            "v2rho2",
            "v2rhosigma",
            "v2rholapl",
            "v2rhotau",
            "v2sigma2",
            "v2sigmalapl",
            "v2sigmatau",
            "v2lapl2",
            "v2lapltau",
            "v2tau2",
        ][self as usize]
    }

    /// Derivative order (0 for `zk`).
    pub fn order(self) -> usize {
        match self as usize {
            0 => 0,
            1..=4 => 1,
            _ => 2,
        }
    }

    /// Values per grid point (libxc `xc_dimensions`).
    /// Spin-resolved components are ordered as in libxc, e.g.
    /// `vsigma = (aa, ab, bb)`, `v2rho2 = (a_a, a_b, b_b)`,
    /// `v2rhosigma = (a_aa, a_ab, a_bb, b_aa, b_ab, b_bb)`,
    /// `v2rhotau = (a_a, a_b, b_a, b_b)`.
    pub fn dim(self, polarized: bool) -> usize {
        if !polarized {
            return 1;
        }
        match self {
            Out::Zk => 1,
            Out::Vrho | Out::Vlapl | Out::Vtau => 2,
            Out::Vsigma => 3,
            Out::V2rho2 | Out::V2lapl2 | Out::V2tau2 => 3,
            Out::V2rholapl | Out::V2rhotau | Out::V2lapltau => 4,
            Out::V2rhosigma | Out::V2sigma2 | Out::V2sigmalapl | Out::V2sigmatau => 6,
        }
    }

    /// Whether a functional with these properties produces this array.
    pub fn produced_by(self, family: Family, needs_tau: bool, needs_lapl: bool) -> bool {
        let mgga = family == Family::Mgga;
        match self {
            Out::Zk | Out::Vrho | Out::V2rho2 => family >= Family::Lda,
            Out::Vsigma | Out::V2rhosigma | Out::V2sigma2 => family >= Family::Gga,
            Out::Vtau | Out::V2rhotau | Out::V2sigmatau | Out::V2tau2 => mgga && needs_tau,
            Out::Vlapl | Out::V2rholapl | Out::V2sigmalapl | Out::V2lapl2 => mgga && needs_lapl,
            Out::V2lapltau => mgga && needs_tau && needs_lapl,
        }
    }
}

/// Density ingredients for a batch of `np` grid points, in libxc layout:
/// one array per quantity, spin components interleaved per point.
///
/// | array | unpolarized | polarized (per point) |
/// |-------|-------------|------------------------|
/// | `rho`   | `np`  | `2*np`: (a, b) |
/// | `sigma` | `np`  | `3*np`: (aa = ∇a·∇a, ab = ∇a·∇b, bb) |
/// | `lapl`  | `np`  | `2*np`: (a, b) |
/// | `tau`   | `np`  | `2*np`: (a, b), tau = ½ Σ_i |∇φ_i|² (no ½ missing) |
///
/// Unpolarized inputs are *total* quantities (rho = rho_a + rho_b,
/// sigma = |∇rho|², tau = tau_a + tau_b).
#[derive(Clone, Copy, Debug)]
pub struct XcInput<'a> {
    pub rho: &'a [f64],
    pub sigma: Option<&'a [f64]>,
    pub lapl: Option<&'a [f64]>,
    pub tau: Option<&'a [f64]>,
}

impl<'a> XcInput<'a> {
    pub fn lda(rho: &'a [f64]) -> Self {
        XcInput {
            rho,
            sigma: None,
            lapl: None,
            tau: None,
        }
    }
    pub fn gga(rho: &'a [f64], sigma: &'a [f64]) -> Self {
        XcInput {
            rho,
            sigma: Some(sigma),
            lapl: None,
            tau: None,
        }
    }
    pub fn mgga(rho: &'a [f64], sigma: &'a [f64], tau: &'a [f64]) -> Self {
        XcInput {
            rho,
            sigma: Some(sigma),
            lapl: None,
            tau: Some(tau),
        }
    }
    /// Add the density Laplacian (only a few meta-GGAs need it).
    pub fn with_lapl(mut self, lapl: &'a [f64]) -> Self {
        self.lapl = Some(lapl);
        self
    }

    /// Number of grid points, from `rho`.
    pub fn np(&self, polarized: bool) -> usize {
        self.rho.len() / if polarized { 2 } else { 1 }
    }

    /// Check the arrays a functional with these properties needs.
    pub(crate) fn validate(
        &self,
        polarized: bool,
        family: Family,
        needs_tau: bool,
        needs_lapl: bool,
    ) -> Result<usize> {
        let ns = if polarized { 2 } else { 1 };
        let np = self.rho.len() / ns;
        let check =
            |name: &'static str, a: Option<&[f64]>, dim: usize, needed: bool| -> Result<()> {
                match a {
                    None if needed => Err(XcError::MissingInput(name)),
                    Some(a) if needed && a.len() != np * dim => Err(XcError::InputLength {
                        name,
                        len: a.len(),
                        expected: np * dim,
                        np,
                        dim,
                    }),
                    _ => Ok(()),
                }
            };
        if !self.rho.len().is_multiple_of(ns) {
            return Err(XcError::InputLength {
                name: "rho",
                len: self.rho.len(),
                expected: np * ns,
                np,
                dim: ns,
            });
        }
        check(
            "sigma",
            self.sigma,
            if polarized { 3 } else { 1 },
            family >= Family::Gga,
        )?;
        check("tau", self.tau, ns, family == Family::Mgga && needs_tau)?;
        check("lapl", self.lapl, ns, family == Family::Mgga && needs_lapl)?;
        Ok(np)
    }
}

/// Borrowed output arrays for one evaluation, indexed by [`Out`].
///
/// Contract for `eval_into`: every array that the functional produces at the
/// requested order must be present with length `np * out.dim(polarized)`; it
/// is completely overwritten. Arrays the functional does not produce are left
/// untouched and may be absent.
#[derive(Default)]
pub struct XcOutRefs<'a> {
    bufs: [Option<&'a mut [f64]>; N_OUT],
}

impl<'a> XcOutRefs<'a> {
    pub fn new() -> Self {
        Self::default()
    }
    /// Attach an output array.
    pub fn set(&mut self, which: Out, buf: &'a mut [f64]) -> &mut Self {
        self.bufs[which as usize] = Some(buf);
        self
    }
    pub fn get(&self, which: Out) -> Option<&[f64]> {
        self.bufs[which as usize].as_deref()
    }
    pub fn get_mut(&mut self, which: Out) -> Option<&mut [f64]> {
        self.bufs[which as usize].as_deref_mut()
    }
    /// Raw pointer for FFI: null if absent.
    pub(crate) fn ptr(&mut self, which: Out) -> *mut f64 {
        match &mut self.bufs[which as usize] {
            Some(b) => b.as_mut_ptr(),
            None => std::ptr::null_mut(),
        }
    }

    /// Check presence and length of every array a functional with these
    /// properties produces up to `order`.
    pub(crate) fn validate(
        &self,
        order: usize,
        np: usize,
        polarized: bool,
        family: Family,
        needs_tau: bool,
        needs_lapl: bool,
    ) -> Result<()> {
        for o in Out::ALL {
            if o.order() > order || !o.produced_by(family, needs_tau, needs_lapl) {
                continue;
            }
            let expected = np * o.dim(polarized);
            match &self.bufs[o as usize] {
                None => return Err(XcError::MissingOutput(o.name())),
                Some(b) if b.len() != expected => {
                    return Err(XcError::OutputLength {
                        name: o.name(),
                        len: b.len(),
                        expected,
                    })
                }
                _ => {}
            }
        }
        Ok(())
    }
}

/// Owned, reusable output buffers. [`crate::Functional::eval`] resizes only the
/// arrays the functional produces at the requested order; the others are
/// empty. Reuse one per thread across batches to avoid reallocation.
///
/// Energy convention: `zk` is the XC energy **per particle** (libxc `zk`,
/// PySCF `exc`): `E_xc = Σ_i w_i ρ_i zk_i` with ρ the total density
/// (see [`XcOutput::integrate_energy`]). Derivatives are of the energy
/// **per volume** `e = ρ zk` with respect to the inputs (`vrho = ∂e/∂ρ_s`,
/// `vsigma = ∂e/∂σ_ss'`, ...), exactly as libxc returns them.
#[derive(Clone, Debug, Default)]
pub struct XcOutput {
    bufs: [Vec<f64>; N_OUT],
    np: usize,
    polarized: bool,
    order: usize,
}

macro_rules! accessors {
    ($($fn_name:ident => $variant:ident),* $(,)?) => {
        $(
            #[doc = concat!("The `", stringify!($fn_name), "` array (empty if not produced).")]
            pub fn $fn_name(&self) -> &[f64] { &self.bufs[Out::$variant as usize] }
        )*
    };
}

impl XcOutput {
    pub fn new() -> Self {
        Self::default()
    }

    /// Resize the arrays needed for `order` and these functional properties.
    pub fn prepare(
        &mut self,
        order: usize,
        np: usize,
        polarized: bool,
        family: Family,
        needs_tau: bool,
        needs_lapl: bool,
    ) {
        self.np = np;
        self.polarized = polarized;
        self.order = order;
        for o in Out::ALL {
            let b = &mut self.bufs[o as usize];
            if o.order() <= order && o.produced_by(family, needs_tau, needs_lapl) {
                b.resize(np * o.dim(polarized), 0.0);
            } else {
                b.clear();
            }
        }
    }

    /// Borrow all non-empty arrays for `eval_into`.
    pub fn refs_mut(&mut self) -> XcOutRefs<'_> {
        let mut r = XcOutRefs::new();
        for (o, b) in Out::ALL.iter().zip(self.bufs.iter_mut()) {
            if !b.is_empty() {
                r.bufs[*o as usize] = Some(b.as_mut_slice());
            }
        }
        r
    }

    /// Array by name; empty if not produced.
    pub fn get(&self, which: Out) -> &[f64] {
        &self.bufs[which as usize]
    }
    pub fn get_mut(&mut self, which: Out) -> &mut [f64] {
        &mut self.bufs[which as usize]
    }
    pub fn np(&self) -> usize {
        self.np
    }
    pub fn order(&self) -> usize {
        self.order
    }

    accessors!(
        zk => Zk, vrho => Vrho, vsigma => Vsigma, vlapl => Vlapl, vtau => Vtau,
        v2rho2 => V2rho2, v2rhosigma => V2rhosigma, v2rholapl => V2rholapl, v2rhotau => V2rhotau,
        v2sigma2 => V2sigma2, v2sigmalapl => V2sigmalapl, v2sigmatau => V2sigmatau,
        v2lapl2 => V2lapl2, v2lapltau => V2lapltau, v2tau2 => V2tau2,
    );

    /// `Σ_i w_i ρ_i zk_i`: the XC energy of this batch for quadrature weights
    /// `weights` (ρ_i is the total density; `rho` in libxc layout).
    pub fn integrate_energy(&self, rho: &[f64], weights: &[f64]) -> f64 {
        let zk = self.zk();
        assert_eq!(weights.len(), zk.len());
        if self.polarized {
            zk.iter()
                .zip(weights)
                .zip(rho.as_chunks::<2>().0)
                .map(|((z, w), r)| w * (r[0] + r[1]) * z)
                .sum()
        } else {
            zk.iter()
                .zip(weights)
                .zip(rho)
                .map(|((z, w), r)| w * r * z)
                .sum()
        }
    }
}

/// A semilocal XC kernel evaluated on batches of grid points.
///
/// Implemented by [`crate::LibxcFunctional`]; native Rust or GPU kernels can
/// implement it later and be used as components of a [`crate::Functional`].
/// Implementations must be safe to evaluate from several threads at once
/// (`&self`), which is why the trait requires `Send + Sync`.
pub trait XcKernel: Send + Sync + fmt::Debug {
    /// Short identifier, e.g. libxc's `gga_x_b88`.
    fn name(&self) -> &str;
    /// libxc id, if the kernel corresponds to one.
    fn libxc_id(&self) -> Option<i32> {
        None
    }
    fn polarized(&self) -> bool;
    fn family(&self) -> Family;
    fn needs_tau(&self) -> bool;
    fn needs_lapl(&self) -> bool;
    /// Highest derivative order available (0, 1 or 2).
    fn max_order(&self) -> usize;
    /// Exact exchange this kernel expects the host to add.
    fn exact_exchange(&self) -> ExactExchange;
    /// VV10 parameters if the kernel expects the host to add VV10.
    fn vv10(&self) -> Option<Vv10>;
    /// Points with total density below this are skipped (outputs set to zero).
    fn dens_threshold(&self) -> f64;
    fn set_dens_threshold(&mut self, threshold: f64);
    /// Evaluate up to `order`. Must completely overwrite every output it
    /// produces (see [`XcOutRefs`]) and leave the others alone.
    fn eval(&self, order: usize, input: &XcInput<'_>, out: &mut XcOutRefs<'_>) -> Result<()>;
    /// An independent copy (same parameters and thresholds).
    fn try_clone(&self) -> Result<Box<dyn XcKernel>>;
    /// The kernel as `Any`, for callers that need its concrete type (the device plans of
    /// `crate::device`); None for kernels that do not offer it.
    fn as_any(&self) -> Option<&dyn std::any::Any> {
        None
    }
}
