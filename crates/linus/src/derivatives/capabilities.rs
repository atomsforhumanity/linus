//! Which calculations a method can do: one capability check per request ([`check`]), before
//! any SCF, so that an unsupported request fails in seconds with a clear
//! [`LinusError::Unsupported`] instead of after the SCF or the explicit derivative terms.
//!
//! Energies and gradients exist for every method that [`MethodInfo::resolve`] and
//! `Session::new` accept. Responses (Hessians, frequencies, polarizabilities) need:
//! - analytic second derivatives of the J/K method (exact and DF; not RIJ + COSX,
//!   [`crate::driver::JkMethod::has_second_derivatives`]);
//! - the XC kernel: libxc's second derivatives of the functional (VV10's kernel and Hessian
//!   terms are linus's own, `linus_dft::vv10_resp`);
//! - a solvent SCF mode with response terms (not the density-fitted PCM);
//! - memory for their fixed-size arrays (`Session::check_request`).

use crate::driver::report::CalcType;
use crate::driver::{CalcContext, CalcOptions, MethodInfo};
use linus_base::error::LinusError;

/// What a calculation computes, as far as the capability checks are concerned.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Request {
    Energy,
    Gradient,
    /// Analytic Hessians and frequencies; with the dipole derivatives (IR intensities), which
    /// the frequencies' IR analysis follows with a polarizability.
    Hessian {
        dipole_derivatives: bool,
    },
    Polarizability,
}

impl Request {
    /// The request of a calculation type (`ir`: a Hessian with IR intensities, `freq.ir`).
    pub fn of(calctype: CalcType, ir: bool) -> Self {
        match calctype {
            CalcType::Energy => Request::Energy,
            CalcType::Gradient => Request::Gradient,
            CalcType::Hessian => Request::Hessian {
                dipole_derivatives: ir,
            },
        }
    }

    /// The purpose its SCF is converged for (a polarizability's: a Hessian's, the tightest).
    pub(crate) fn purpose(&self) -> crate::driver::Purpose {
        use crate::driver::Purpose;
        match self {
            Request::Energy => Purpose::Energy,
            Request::Gradient => Purpose::Gradient,
            Request::Hessian { .. } | Request::Polarizability => Purpose::Hessian,
        }
    }

    /// A linear-response calculation (CPHF/CPKS).
    pub fn is_response(&self) -> bool {
        matches!(self, Request::Hessian { .. } | Request::Polarizability)
    }

    fn what(&self) -> &'static str {
        match self {
            Request::Polarizability => "polarizabilities",
            _ => "analytic Hessians",
        }
    }
}

/// Whether `method` with the settings `opts` can do `req` (see the module docs; the J/K
/// method's own capabilities are `CalcContext::check_second_derivatives`, which
/// [`crate::driver::Session::check_request`] asks first); the first thing it cannot do is the error.
pub fn check(req: Request, method: &MethodInfo, opts: &CalcOptions) -> Result<(), LinusError> {
    if !req.is_response() {
        return Ok(());
    }
    if method.max_order < 2 {
        return Err(LinusError::Unsupported(format!(
            "{}: libxc provides no second derivatives (XC kernel) for this functional, which analytic Hessians, frequencies \
             and polarizabilities need; use finite differences of analytic gradients",
            opts.method()
        )));
    }
    if let Some(so) = &opts.solvent {
        if crate::solvent::resolve_scf_mode(so, &opts.jk_method())? == linus_solvent::PcmScfMode::Df
        {
            return Err(crate::solvent::df_pcm_response_error());
        }
    }
    Ok(())
}

impl CalcContext {
    /// Whether this session can do `req`, before any SCF: what the method can do
    /// ([`CalcContext::check_capabilities`]) and the memory of every phase within the limit
    /// (`CalcContext::check_memory`).
    pub fn check_request(&self, req: Request) -> Result<(), LinusError> {
        self.check_capabilities(req)?;
        self.check_memory(req)
    }

    /// Whether the method can do `req`: the J/K method's second derivatives (responses) and
    /// [`check`] (a plan asks this alone: it states the memory instead of failing on it).
    pub fn check_capabilities(&self, req: Request) -> Result<(), LinusError> {
        if req.is_response() {
            self.check_second_derivatives(req.what())?;
        }
        self.plan.capabilities(req, &self.method)
    }
}
