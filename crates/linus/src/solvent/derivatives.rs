//! The solvent's parts of gradients and Hessians at a ground state, on the backend the plan
//! puts the PCM on ([`SolventGradient`]; the CPU's over linus-solvent's `PcmScf`).

use crate::driver::ground::GroundState;
use crate::driver::CalcContext;
use faer::{Mat, Par};
use linus_base::error::LinusError;
use linus_base::terms::EnvironmentResponse;
use linus_core::timing::Timers;
use linus_ints::DerivOrder;
use linus_solvent::{smd, PcmScf};

impl CalcContext {
    /// SMD: G_CDS with its gradient or Hessian at the ground state `gs` (None without SMD).
    pub(crate) fn solvent_cds(
        &self,
        gs: &GroundState,
        order: smd::CdsOrder,
    ) -> Result<Option<smd::CdsResult>, LinusError> {
        let Some(so) = self.plan.solvent() else {
            return Ok(None);
        };
        let resolved = so.resolve(&self.mol.z).map_err(LinusError::Input)?;
        Ok(resolved.smd.map(|d| {
            smd::cds(
                &self.mol.z,
                &self.mol.positions,
                &d,
                &super::scf::cds_config(so, gs.purpose(), order),
            )
        }))
    }

    /// Fixed-density nuclear derivative of the solvent energy at the ground state `gs` (natom x
    /// 3), or None without solvent (see [`PcmScf::fixed_density_gradient`]).
    pub(crate) fn solvent_gradient(
        &self,
        gs: &GroundState,
        timers: &mut Timers,
    ) -> Result<Option<Vec<[f64; 3]>>, LinusError> {
        let r = &gs.scf;
        let Some(so) = self.plan.solvent() else {
            return Ok(None);
        };
        let t0 = std::time::Instant::now();
        let dt = &r.d_a + &r.d_b;
        let d = linus_base::linalg::mat_to_rowmajor(dt.as_ref());
        // on the backend the plan puts the PCM on (crate::backend)
        let mut g = {
            let mut term = self.resolve(crate::driver::backend::Family::Pcm, |b| {
                b.solvent_gradient(self, gs, so, timers)
            })?;
            term.gradient(&d, timers)?
        };
        if let Some(c) = self.solvent_cds(gs, smd::CdsOrder::Gradient)? {
            for (a, b) in g.iter_mut().zip(c.gradient.as_ref().expect("CDS gradient")) {
                for k in 0..3 {
                    a[k] += b[k];
                }
            }
        }
        timers.add("grad_solvent", t0.elapsed().as_secs_f64());
        Ok(Some(g))
    }
}

impl CalcContext {
    /// Solvent parts of an analytic Hessian at the ground state `gs` for its total density `dt`
    /// (row-major), or None in the gas phase: (explicit second derivatives, fixed-density F^x
    /// per perturbation, the PCM term for the response).
    #[allow(clippy::type_complexity)]
    pub(crate) fn solvent_hessian(
        &self,
        gs: &GroundState,
        dt: &[f64],
        timers: &mut Timers,
    ) -> Result<Option<(Vec<f64>, Vec<Mat<f64>>, std::sync::Arc<PcmScf>)>, LinusError> {
        let Some(so) = self.plan.solvent() else {
            return Ok(None);
        };
        let t0 = std::time::Instant::now();
        // (the density-fitted PCM has no Hessian terms: refused by derivatives::capabilities)
        let pcm = self.solvent_state(gs, so)?;
        // the geometry's pairs with second derivatives (crate::context; the Hessian's)
        let pairs = self.integrals.pairs(&self.basis, DerivOrder::Two);
        let mut t =
            pcm.hessian_terms_with(&self.mol.effective_z(), &self.mol.positions, dt, &pairs);
        if let Some(c) = self.solvent_cds(gs, smd::CdsOrder::Hessian)? {
            t.hess
                .iter_mut()
                .zip(c.hessian.as_ref().expect("CDS Hessian"))
                .for_each(|(a, b)| *a += b);
        }
        timers.add("hess_solvent", t0.elapsed().as_secs_f64());
        Ok(Some((t.hess, t.fx, pcm)))
    }

    /// The reaction field's response at the ground state `gs` for a response operator (the
    /// PCM term's), or None in the gas phase (the density-fitted PCM has no response terms:
    /// refused by derivatives::capabilities).
    pub(crate) fn solvent_response(
        &self,
        gs: &GroundState,
    ) -> Result<Option<std::sync::Arc<dyn EnvironmentResponse>>, LinusError> {
        match self.plan.solvent() {
            Some(so) => Ok(Some(self.solvent_state(gs, so)?)),
            None => Ok(None),
        }
    }
}

/// The solvent's fixed-density nuclear gradient as a backend set it up (the `pcm` family's
/// gradient, `crate::backend`): the CPU's ([`PcmScf::fixed_density_gradient`]) or a device's.
pub(crate) trait SolventGradient {
    /// The gradient (natom x 3) of the total density `d` (row-major nbf^2).
    fn gradient(&mut self, d: &[f64], timers: &mut Timers) -> Result<Vec<[f64; 3]>, LinusError>;
}

/// The CPU's: the ground state's PCM term ([`CalcContext::solvent_state`]) on the geometry's
/// pairs of derivative order one (the gradient's).
pub(crate) struct CpuSolventGradient<'a> {
    pub cx: &'a CalcContext,
    pub pcm: std::sync::Arc<PcmScf>,
}

impl SolventGradient for CpuSolventGradient<'_> {
    fn gradient(&mut self, d: &[f64], timers: &mut Timers) -> Result<Vec<[f64; 3]>, LinusError> {
        let cx = self.cx;
        let pairs = cx.integrals.pairs(&cx.basis, DerivOrder::One);
        Ok(self
            .pcm
            .fixed_density_gradient_timed(
                &cx.mol.effective_z(),
                &cx.mol.positions,
                d,
                Some(&pairs),
                timers,
            )
            .1)
    }
}

/// Hessian hook (crate::derivatives::hessian), over spin channels: adds the solvent's explicit
/// second derivatives to `h` and its fixed-density F^x C_s to `fxo[s][x]` for the occupied
/// orbitals `co[s]` of the channels of the ground state `gs`, and returns the reaction field's
/// response for the response operator; None in the gas phase. `dt`: the total density
/// (row-major).
#[allow(clippy::too_many_arguments)]
pub(crate) fn hessian_terms(
    calc: &CalcContext,
    gs: &GroundState,
    dt: &[f64],
    co: &[faer::MatRef<'_, f64>],
    h: &mut [f64],
    fxo: &mut [Vec<Mat<f64>>],
    par: Par,
    timers: &mut Timers,
) -> Result<Option<std::sync::Arc<dyn EnvironmentResponse>>, LinusError> {
    let Some((hs, fx, pcm)) = calc.solvent_hessian(gs, dt, timers)? else {
        return Ok(None);
    };
    h.iter_mut().zip(&hs).for_each(|(a, b)| *a += b);
    for (fs, c) in fxo.iter_mut().zip(co) {
        for (f, v) in fs.iter_mut().zip(&fx) {
            *f += linus_base::linalg::mm(v.as_ref(), *c, par);
        }
    }
    Ok(Some(pcm))
}
