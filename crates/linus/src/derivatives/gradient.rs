//! Analytic nuclear gradients of HF and KS-DFT energies (restricted and unrestricted;
//! exact four-center or density-fitted J/K).
//!
//! g = dE_nuc + sum D h^x - sum W S^x + E_2e^x + E_xc^x, with W the energy-weighted density.
//! Density fitting (Coulomb metric, one auxiliary basis for J and K):
//!   E_J^x = sum_P c_P (P|mn)^x D_mn - 1/2 c^T V^x c,                c = V^-1 (P|mn) D_mn
//!   E_K^x = sum_P (P|mn)^x G_Pmn + sum_PQ (P|Q)^x g_PQ,
//!   G_P = -a_x sum_s C_s X^s_P C_s^T,  g_PQ = a_x/2 sum_s tr(X^s_P X^s_Q),  X^s_P = C_s^T B~_P C_s,
//! with B~_P = sum_Q V^-1_PQ (Q|mn) and C_s the occupied orbitals of spin s.

use crate::backend::{Backend, CpuBackend, Setup};
use crate::derivatives::terms::{GradientInputs, GradientTerm, Lane};
use crate::driver::backend::Family;
use crate::driver::ground::GroundState;
use crate::driver::{CalcContext, Purpose, Session};
use faer::Par;
use linus_base::error::LinusError;
use linus_base::linalg::mat_to_rowmajor;
use linus_core::timing::Timers;
use linus_ints::{DerivOrder, OneEPlan, PairSet};
use linus_scf::reference::Reference;
use std::sync::Arc;

#[derive(Clone, Debug)]
pub struct GradientResult {
    pub energy: f64,
    /// dE/dR (natom x 3), Eh/bohr
    pub gradient: Vec<[f64; 3]>,
    /// The contributions (each natom x 3; with `solvent` they sum to `gradient`).
    pub parts: GradientParts,
    /// Implicit-solvent part at fixed density (included in `gradient`; None in the gas phase).
    pub solvent: Option<Vec<[f64; 3]>>,
    pub timers: Timers,
}

/// The contributions to an analytic gradient (each natom x 3, Eh/bohr).
#[derive(Clone, Debug)]
pub struct GradientParts {
    pub nuclear_repulsion: Vec<[f64; 3]>,
    /// One-electron terms, the energy-weighted density's -W S^x and the ECPs included.
    pub one_electron: Vec<[f64; 3]>,
    pub two_electron: Vec<[f64; 3]>,
    /// XC, VV10 included.
    pub xc: Vec<[f64; 3]>,
    pub dispersion: Vec<[f64; 3]>,
}

impl GradientParts {
    /// The parts in the order they are summed.
    pub fn all(&self) -> [&Vec<[f64; 3]>; 5] {
        [
            &self.nuclear_repulsion,
            &self.one_electron,
            &self.two_electron,
            &self.xc,
            &self.dispersion,
        ]
    }
}

/// A family's part of a gradient (natom x 3).
type Part = Vec<[f64; 3]>;

fn add(a: &mut [[f64; 3]], b: &[[f64; 3]]) {
    for (x, y) in a.iter_mut().zip(b) {
        for k in 0..3 {
            x[k] += y[k];
        }
    }
}

impl CalcContext {
    /// [`Self::gradient_for`] at the ground state `gs` with its reference (linus_scf::reference),
    /// which a Hessian goes on to use.
    pub(crate) fn gradient_at(
        &self,
        gs: &GroundState,
        reference: &Reference,
    ) -> Result<GradientResult, LinusError> {
        let mut timers = Timers::default();
        let par = self.par();
        let [p0, p1, p2, p3] = self
            .pool
            .install(|| self.gradient_parts(gs, reference, par, &mut timers))?;
        let p4 = match self.dispersion(1)? {
            Some(d) => d.gradient,
            None => vec![[0.0; 3]; self.mol.natom()],
        };
        let parts = GradientParts {
            nuclear_repulsion: p0,
            one_electron: p1,
            two_electron: p2,
            xc: p3,
            dispersion: p4,
        };
        let mut g = vec![[0.0; 3]; self.mol.natom()];
        for p in parts.all() {
            add(&mut g, p);
        }
        // implicit solvent (crate::solvent)
        let solvent = self
            .pool
            .install(|| self.solvent_gradient(gs, &mut timers))?;
        if let Some(g_solvent) = &solvent {
            add(&mut g, g_solvent);
        }
        // the SCF's device hand-off (crate::backend::gpu::session) is done with unless a Hessian
        // follows
        if gs.purpose() == Purpose::Gradient {
            self.release_handoff();
        }
        Ok(GradientResult {
            energy: gs.scf.energy,
            gradient: g,
            parts,
            solvent,
            timers,
        })
    }

    fn gradient_parts(
        &self,
        gs: &GroundState,
        reference: &Reference,
        par: Par,
        timers: &mut Timers,
    ) -> Result<[Vec<[f64; 3]>; 4], LinusError> {
        let basis = &self.basis;
        // the spins' occupied orbitals and densities (a closed shell's one channel for both)
        let (alpha, beta) = (reference.channels.alpha(), reference.channels.beta());
        let (ca, cb) = (&alpha.c_occ, &beta.c_occ);
        let (da, db) = (&alpha.density, &beta.density);
        // energy-weighted density (with the minus sign the 1e routine expects)
        let ws: Vec<f64> = mat_to_rowmajor(reference.energy_weighted.as_ref())
            .into_iter()
            .map(|x| -x)
            .collect();
        let dtf = mat_to_rowmajor(reference.total_density.as_ref());
        let g_nuc = self.mol.nuclear_repulsion_gradient();
        // the one-electron terms (pairs, S, T, V, ECPs; CPU); the pairs are the geometry's
        // (crate::context), at the order the ground state's purpose needs (a Hessian's second
        // derivatives: one set for its gradient and its explicit terms)
        let one_e = |timers: &mut Timers| {
            let pairs = timers.time("grad_pairs", || self.gradient_pairs(gs));
            let charges = self.mol.point_charges();
            let plan = OneEPlan::new(basis, DerivOrder::One)
                .with_charges(&pairs, &charges)
                .expect("pairs of derivative order one");
            let mut wk = plan.workspace();
            let mut g_1e = vec![[0.0; 3]; basis.natom()];
            timers.time("grad_1e", || {
                plan.overlap_kinetic_grad(Some(&ws), Some(&dtf), &mut g_1e, &mut wk)
            });
            let mut gv = vec![[0.0; 3]; basis.natom()];
            timers.time("grad_1e", || {
                plan.nuclear_attraction_grad(&dtf, &mut gv, &mut wk)
            });
            add(&mut g_1e, &gv);
            add(
                &mut g_1e,
                &timers.time("grad_ecp", || self.ecp_gradient(&pairs, &dtf)),
            );
            (pairs, g_1e)
        };
        let inputs = GradientInputs {
            ground: gs,
            ca,
            cb,
            da,
            db,
            dtf: &dtf,
            par,
        };
        // the method families' terms on the backends the plan puts them on
        // (crate::derivatives::terms), in the order the gradient sums them: the XC engine's (VV10
        // included, on its own grid; "grad_vv10" is part of "grad_xc"), then the J/K method's
        // two-electron gradient with the method's exact exchange
        let grid_response = self.plan.grid_response();
        let xop = self.method.exchange;
        let mut terms = [
            (
                Family::GradientXc,
                self.resolve(Family::GradientXc, |b| b.gradient_xc(self, grid_response))?,
            ),
            (
                Family::GradientJk,
                self.resolve(Family::GradientJk, |b| b.gradient_jk(self, xop))?,
            ),
        ];
        let ((_pairs, g_1e), parts) = self.schedule(&mut terms, &inputs, one_e, timers)?;
        let [g_xc, g_2e] = parts;
        Ok([g_nuc, g_1e, g_2e, g_xc])
    }

    /// The shell pairs of a gradient at the ground state `gs`: the geometry's (crate::context),
    /// at the order its purpose needs, at least one.
    pub(crate) fn gradient_pairs(&self, gs: &GroundState) -> Arc<PairSet> {
        self.integrals
            .pairs(&self.basis, gs.pair_order().max(DerivOrder::One))
    }

    /// The gradient's schedule: the method families' terms `terms` (in the order they are summed)
    /// and the host's one-electron pass `one_e`. Each term's host setup runs first, on the whole
    /// pool (a device XC engine's layout); then the device's terms run beside the one-electron
    /// pass (each part computed as on its own: the same bits), then the host's terms. A device
    /// term that declines (its setup or a call does not fit the device) is recorded with the
    /// reason and replaced by the CPU's term, which runs after the overlap. Returns the
    /// one-electron pass's result and the terms' parts in their order.
    fn schedule<'a, P: Send, const N: usize>(
        &'a self,
        terms: &mut [(Family, Box<dyn GradientTerm + 'a>); N],
        inputs: &GradientInputs<'_>,
        one_e: impl FnOnce(&mut Timers) -> P + Send,
        timers: &mut Timers,
    ) -> Result<(P, [Part; N]), LinusError> {
        let mut parts: [Option<Part>; N] = std::array::from_fn(|_| None);
        for (k, term) in terms.iter_mut() {
            if let Setup::Declined(why) = term.prepare(inputs, timers)? {
                self.note_setup(*k, Some(why));
                *term = self.cpu_gradient_term(*k)?;
            }
        }
        let device = terms.iter().any(|(_, t)| t.lane() == Lane::Device);
        let one = if device {
            let (mut t_dev, mut t_host) = (Timers::default(), Timers::default());
            let (dev, one) = rayon::join(
                || -> Result<_, LinusError> {
                    let mut out = Vec::new();
                    for (i, (_, term)) in terms.iter_mut().enumerate() {
                        if term.lane() == Lane::Device {
                            out.push((i, term.gradient(inputs, &mut t_dev)?));
                        }
                    }
                    Ok(out)
                },
                || one_e(&mut t_host),
            );
            timers.merge(&t_host);
            timers.merge(&t_dev);
            for (i, part) in dev? {
                match part {
                    Setup::Ready(g) => parts[i] = Some(g),
                    Setup::Declined(why) => {
                        let k = terms[i].0;
                        self.note_setup(k, Some(why));
                        terms[i].1 = self.cpu_gradient_term(k)?;
                    }
                }
            }
            one
        } else {
            one_e(timers)
        };
        for (i, (k, term)) in terms.iter_mut().enumerate() {
            if parts[i].is_none() {
                parts[i] = Some(match term.gradient(inputs, timers)? {
                    Setup::Ready(g) => g,
                    Setup::Declined(why) => {
                        return Err(LinusError::Internal(format!(
                            "the CPU declined {}: {why}",
                            k.name()
                        )))
                    }
                });
            }
        }
        Ok((one, parts.map(|p| p.expect("every term computed"))))
    }

    /// The CPU's term of the gradient family `k` (after a device's declined).
    fn cpu_gradient_term(&self, k: Family) -> Result<Box<dyn GradientTerm + '_>, LinusError> {
        let term = match k {
            Family::GradientXc => CpuBackend.gradient_xc(self, self.plan.grid_response())?,
            _ => CpuBackend.gradient_jk(self, self.method.exchange)?,
        };
        match term {
            Setup::Ready(t) => Ok(t),
            Setup::Declined(why) => Err(LinusError::Internal(format!(
                "the CPU declined {}: {why}",
                k.name()
            ))),
        }
    }
}

impl Session {
    /// Energy and analytic gradient at the current geometry.
    pub fn gradient(&mut self) -> Result<GradientResult, LinusError> {
        self.gradient_for(Purpose::Gradient)
    }

    /// Energy and analytic gradient with the SCF converged as `purpose` needs.
    pub fn gradient_for(&mut self, purpose: Purpose) -> Result<GradientResult, LinusError> {
        let gs = self.scf_stage(purpose)?;
        let g = self.cx.progress.stage("gradient", || {
            self.cx
                .gradient_at(&gs, gs.reference(&self.cx.pool, self.par()))
        });
        // the gradient is the last phase of the call to use the metrics (crate::context)
        self.cx.integrals.release_metrics();
        g
    }
}
