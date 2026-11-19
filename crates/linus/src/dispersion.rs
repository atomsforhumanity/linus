//! Empirical dispersion corrections (DFT-D3(BJ), optionally with the ATM three-body term,
//! and DFT-D4) through the optional LGPL-3.0 crate linus-disp (cargo feature `dispersion`).

use crate::driver::options::{parse_dispersion, Dispersion, DispersionSetting};
use crate::driver::{CalcContext, CalcOptions};
use crate::structure::Structure;
use linus_base::error::LinusError;

/// Dispersion energy and derivatives (gradient natom x 3; Hessian 3N x 3N row-major).
#[derive(Clone, Debug)]
pub struct DispTerms {
    pub energy: f64,
    pub gradient: Vec<[f64; 3]>,
    pub hessian: Option<Vec<f64>>,
}

/// The dispersion model requested by the options, or implied by the name of the resolved
/// functional `f` (e.g. ORCA-style `wb97x-d3bj`; None: Hartree-Fock). `dispersion: "none"`
/// turns an implied model off. Models linus does not implement (D3 with zero damping, implied
/// by `wb97x-d3`) are [`LinusError::Unsupported`].
pub fn requested(
    opts: &CalcOptions,
    f: Option<&linus_xc::Functional>,
) -> Result<Option<Dispersion>, LinusError> {
    match opts.dispersion() {
        DispersionSetting::Off => return Ok(None),
        DispersionSetting::Model(d) => return Ok(Some(d)),
        DispersionSetting::Auto => {}
    }
    let Some(f) = f else {
        return Ok(None);
    };
    match f.dispersion() {
        None => Ok(None),
        Some(d) => parse_dispersion(d).map_err(|e| match e {
            LinusError::Unsupported(_) => LinusError::Unsupported(format!(
                "{} needs D3 with zero damping (D3(0)), which is not implemented; use wb97x-d3bj or wb97x-v, or set dispersion \
                 \"none\" for the functional without it",
                opts.method()
            )),
            e => e,
        }),
    }
}

impl CalcContext {
    /// The dispersion terms of the method at the current geometry up to `order` (0 energy, 1
    /// gradient, 2 Hessian); None without dispersion.
    pub fn dispersion(&self, order: usize) -> Result<Option<DispTerms>, LinusError> {
        match self.method.dispersion {
            None => Ok(None),
            // in the session's own thread pool (linus-disp uses rayon)
            Some(kind) => self
                .pool
                .install(|| terms(&self.mol, self.plan.method(), kind, order))
                .map(Some),
        }
    }
}

#[cfg(feature = "dispersion")]
fn terms(
    mol: &Structure,
    method: &str,
    kind: Dispersion,
    order: usize,
) -> Result<DispTerms, LinusError> {
    use linus_disp::{d3bj, d3bj_params, d4, d4_params, D3Options, D4Options, Order};
    let ord = match order {
        0 => Order::Energy,
        1 => Order::Gradient,
        _ => Order::Hessian,
    };
    let r = match kind {
        Dispersion::D3bj | Dispersion::D3bjAtm => {
            let mut p = d3bj_params(method).map_err(disp_error)?;
            if kind == Dispersion::D3bjAtm {
                p = p.with_atm(1.0);
            }
            d3bj(
                &mol.z,
                &mol.positions,
                &p,
                &D3Options {
                    order: ord,
                    ..Default::default()
                },
            )
            .map_err(disp_error)?
        }
        Dispersion::D4 => {
            let p = d4_params(method).map_err(disp_error)?;
            d4(
                &mol.z,
                &mol.positions,
                mol.charge as f64,
                &p,
                &D4Options {
                    order: ord,
                    ..Default::default()
                },
            )
            .map_err(disp_error)?
        }
    };
    Ok(DispTerms {
        energy: r.energy,
        gradient: r.gradient,
        hessian: r.hessian,
    })
}

/// A linus-disp error as a [`LinusError`] (an input error for missing parameters and bad
/// shapes, unsupported for elements outside the model, numerical otherwise). Here rather than a
/// `From` impl in linus-base, so that the LGPL crate stays a dependency of the facade alone.
#[cfg(feature = "dispersion")]
fn disp_error(e: linus_disp::DispError) -> LinusError {
    use linus_disp::DispError as D;
    match e {
        D::NoParameters { .. } => LinusError::Input(format!(
            "dispersion: {e}; use dispersion \"none\" or another model (d3bj, d4)"
        )),
        D::UnsupportedElement { .. } => LinusError::Unsupported(format!("dispersion: {e}")),
        D::Numerical(_) => LinusError::Numerical(format!("dispersion: {e}")),
        D::Shape(_) => LinusError::Input(format!("dispersion: {e}")),
    }
}

#[cfg(not(feature = "dispersion"))]
fn terms(
    _mol: &Structure,
    _method: &str,
    kind: Dispersion,
    _order: usize,
) -> Result<DispTerms, LinusError> {
    Err(LinusError::Unsupported(format!(
        "{} requested, but linus was built without the `dispersion` feature",
        kind.name()
    )))
}
