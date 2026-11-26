//! linus-jk: the Coulomb (J) and exchange (K) builders and their derivatives, one module per
//! approximation:
//!
//! - [`exact`]: four-center integrals, direct (the SCF builder over linus-ints's `JkPlan`) or
//!   kept in memory ([`IncoreJk`]); gradients and Hessian terms;
//! - [`df`]: density fitting (the RI-JK builder `DfJk`, the response J/K `DfResponseJk`, RI-J),
//!   gradients and Hessian terms;
//! - [`cosx`]: seminumerical exchange on COSX grids, its gradient and the RIJCOSX gradient;
//! - here: [`CompositeJk`] (J from one builder, K from another) and [`DenseResponseJk`] (the
//!   [`ResponseJk`] of the CPHF/CPKS responses through any SCF [`JkBuilder`]).
//!
//! J\[D\]_mn = sum_ls (mn|ls) D_ls,  K\[D\]_mn = sum_ls (ml|ns) D_ls  for symmetric D.
//!
//! The builders implement linus-base's family terms (`linus_base::terms`), through which the SCF
//! and the responses call them; a GPU implementation is dropped in behind the same traits (the
//! facade's `backend::gpu::df`). Which approximation a calculation uses, at which derivative
//! order, is the facade's dispatch (`linus::two_electron`).

pub mod cosx;
pub mod df;
pub mod exact;
pub mod gridk;

use faer::{Mat, Par};
use linus_base::error::LinusError;
use linus_base::linalg::mat_from_rowmajor;
use linus_base::terms::{JkBuilder, JkOutput, JkRequest, ResponseJk};
use linus_ints::eri4c;
use linus_ints::operator::Operator;
use linus_ints::{BasisSet, PairOptions, PairSet};
use rayon::prelude::*;

/// [`ResponseJk`] through any SCF [`JkBuilder`] (exact four-center, in-core): the full
/// first-order densities of all channels in one request, then (c_j J - c_k K) C_s.
pub struct DenseResponseJk {
    jk: Box<dyn JkBuilder>,
    occ: Vec<Mat<f64>>,
}

impl DenseResponseJk {
    /// For the occupied orbitals `occ` of each channel.
    pub fn new(jk: Box<dyn JkBuilder>, occ: &[faer::MatRef<'_, f64>]) -> Self {
        DenseResponseJk {
            jk,
            occ: occ.iter().map(|c| c.to_owned()).collect(),
        }
    }
}

impl ResponseJk for DenseResponseJk {
    fn name(&self) -> String {
        self.jk.name()
    }

    fn fock_occ(
        &mut self,
        a: &[Vec<Mat<f64>>],
        c_j: f64,
        c_k: f64,
        par: Par,
    ) -> Result<Vec<Vec<Mat<f64>>>, LinusError> {
        let nch = self.occ.len();
        assert_eq!(
            a.len(),
            nch,
            "DenseResponseJk: one list of factors per channel"
        );
        let nd = a.first().map_or(0, |v| v.len());
        let occ = &self.occ;
        let dens: Vec<Mat<f64>> = (0..nch)
            .flat_map(|s| {
                a[s].iter().map(move |x| {
                    let t = linus_base::linalg::mm(x.as_ref(), occ[s].transpose(), par);
                    &t + t.transpose()
                })
            })
            .collect();
        let want_k = c_k != 0.0;
        let out = self.jk.compute(
            &JkRequest {
                dens: &dens,
                factors: None,
                want_j: true,
                want_k,
            },
            par,
        )?;
        drop(dens);
        // J of the total density of each perturbation (the sum of the channels' J)
        let jt: Vec<Mat<f64>> = match nch {
            1 => out.j,
            _ => (0..nd)
                .map(|d| (1..nch).fold(out.j[d].clone(), |acc, s| acc + &out.j[s * nd + d]))
                .collect(),
        };
        Ok((0..nch)
            .map(|s| {
                (0..nd)
                    .map(|d| {
                        let mut f = &jt[d] * c_j;
                        if want_k {
                            f -= &out.k[s * nd + d] * c_k;
                        }
                        linus_base::linalg::mm(f.as_ref(), occ[s].as_ref(), par)
                    })
                    .collect()
            })
            .collect())
    }

    fn release(&mut self) {}

    fn restore(&mut self, _par: Par) {}

    fn timers(&self) -> Option<linus_core::timing::Timers> {
        None
    }
}

/// Full ERI tensor in memory (testing and very small systems).
pub struct IncoreJk {
    nbf: usize,
    g: Vec<f64>,
    basis: BasisSet,
    /// tensor of the exchange operator a/r + b erf(w r)/r (set_exchange_operator), else K uses g
    gk: Option<Vec<f64>>,
}

impl IncoreJk {
    /// Bytes the builder allocates for `nbf` basis functions: the nbf^4 tensor (a second one
    /// with a range-separated exchange operator) and the integral pass's temporaries (about an
    /// eighth of a tensor).
    pub fn bytes_needed(nbf: usize, range_separated: bool) -> usize {
        let t = nbf.saturating_pow(4).saturating_mul(8);
        t.saturating_mul(if range_separated { 2 } else { 1 })
            .saturating_add(t / 8)
    }

    pub fn new(basis: &BasisSet, pairs: &PairSet) -> Self {
        IncoreJk {
            nbf: basis.nbf,
            g: eri4c::eri_full(basis, pairs, &linus_ints::operator::Operator::Coulomb),
            basis: basis.clone(),
            gk: None,
        }
    }
}

impl JkBuilder for IncoreJk {
    fn name(&self) -> String {
        "incore".into()
    }
    fn set_exchange_operator(&mut self, op: Operator) -> bool {
        self.gk = Some(eri4c::eri_full(
            &self.basis,
            &PairSet::build(&self.basis, PairOptions::default()),
            &op,
        ));
        true
    }

    fn compute(&mut self, req: &JkRequest, _par: Par) -> Result<JkOutput, LinusError> {
        let n = self.nbf;
        let g = &self.g;
        let gk = self.gk.as_ref().unwrap_or(g);
        let mut out = JkOutput::default();
        for d in req.dens {
            let dv = linus_base::linalg::mat_to_rowmajor(d.as_ref());
            if req.want_j {
                let jv: Vec<f64> = (0..n * n)
                    .into_par_iter()
                    .map(|mn| {
                        g[mn * n * n..(mn + 1) * n * n]
                            .iter()
                            .zip(&dv)
                            .map(|(a, b)| a * b)
                            .sum()
                    })
                    .collect();
                out.j.push(mat_from_rowmajor(n, n, &jv));
            }
            if req.want_k {
                let kv: Vec<f64> = (0..n * n)
                    .into_par_iter()
                    .map(|mn| {
                        let (m, nn) = (mn / n, mn % n);
                        let mut s = 0.0;
                        for l in 0..n {
                            let base = ((m * n + l) * n + nn) * n;
                            for sg in 0..n {
                                s += gk[base + sg] * dv[l * n + sg];
                            }
                        }
                        s
                    })
                    .collect();
                out.k.push(mat_from_rowmajor(n, n, &kv));
            }
        }
        Ok(out)
    }
}

/// J from one builder, K from another (RI-J + COSX exchange, RI-J + exact exchange).
pub struct CompositeJk {
    pub j: Box<dyn JkBuilder>,
    pub k: Box<dyn JkBuilder>,
}

impl JkBuilder for CompositeJk {
    fn name(&self) -> String {
        format!("J:{} + K:{}", self.j.name(), self.k.name())
    }
    fn compute(&mut self, req: &JkRequest, par: Par) -> Result<JkOutput, LinusError> {
        let mut out = JkOutput::default();
        if req.want_j {
            let r = JkRequest {
                dens: req.dens,
                factors: req.factors,
                want_j: true,
                want_k: false,
            };
            out.j = self.j.compute(&r, par)?.j;
        }
        if req.want_k {
            let r = JkRequest {
                dens: req.dens,
                factors: req.factors,
                want_j: false,
                want_k: true,
            };
            let k = self.k.compute(&r, par)?;
            (out.k, out.k_energy) = (k.k, k.k_energy);
        }
        Ok(out)
    }
    fn incremental_capable(&self) -> bool {
        self.j.incremental_capable() && self.k.incremental_capable()
    }
    fn reset_incremental(&mut self) {
        self.j.reset_incremental();
        self.k.reset_incremental();
    }
    fn built_incrementally(&self) -> bool {
        self.j.built_incrementally() || self.k.built_incrementally()
    }
    fn set_exchange_operator(&mut self, op: Operator) -> bool {
        self.k.set_exchange_operator(op)
    }
    fn next_approximation(&mut self) -> bool {
        // J first: both parts move on (neither may skip its step)
        let j = self.j.next_approximation();
        self.k.next_approximation() | j
    }
    /// HOOK (solvent, the density-fitted PCM): the Coulomb part takes it.
    fn set_coulomb_hook(
        &mut self,
        hook: Option<Box<dyn linus_base::terms::CoulombHook>>,
    ) -> Result<linus_base::terms::HookOutcome, LinusError> {
        self.j.set_coulomb_hook(hook)
    }
}
