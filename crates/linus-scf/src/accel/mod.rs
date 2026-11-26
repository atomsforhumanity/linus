//! The SCF's accelerators: how an iteration makes the next orbitals from its Fock matrices.
//!
//! | step | when (the [`crate::ScfAlgorithm`] settings) |
//! |---|---|
//! | SOSCF: a quasi-Newton orbital rotation (`Soscf`) | `soscf` once max \|F_ai\| < `soscf_start`, and always while descending (after a restart) |
//! | ADIIS (`Adiis`) | `adiis` while max \|F_ai\| > `adiis_switch` |
//! | damping: F <- (1 - w) F + w F_previous | `damping` while max \|F_ai\| > `damping_until` (not after the escalation) |
//! | DIIS ([`Diis`]) | from iteration `diis_start` with `diis_space` > 0 |
//! | Roothaan: diagonalize F | otherwise |
//!
//! The level shift F + shift (S - S D S) applies to every diagonalization while max |F_ai| >
//! `level_shift_until` (not after the escalation). The Fock matrix of a guess density (not from
//! orbitals) is never used for an extrapolation.
//!
//! `Accelerators::restart` answers the controls' restarts: a new DIIS subspace, no ADIIS history,
//! a new SOSCF history and no damping, and quasi-Newton steps from then on (descent, which lowers the
//! energy where DIIS converges to any stationary point).

mod adiis;
mod diis;
pub mod soscf;

pub(crate) use adiis::Adiis;
pub use diis::{err_norm, Diis, DiisRule};

pub(crate) use soscf::{rotate, Soscf};

use super::settings::{DiisDrop, ScfAlgorithm};
use super::state::OrbitalState;
use super::{trace, try_diagonalize, try_semicanonical, ScfIteration};
use faer::{Mat, MatRef, Par};
use linus_base::linalg::mm;
use linus_base::progress::Progress;
use linus_core::timing::Timers;

/// The accelerators' restarts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Restart {
    /// A stability step or a two-cycle: quasi-Newton steps from the current orbitals.
    Descent,
    /// The escalation at `max_iter`: quasi-Newton steps with the relative curvature test, no
    /// level shift and no damping, from the lowest-energy iterate's occupied orbitals (the first
    /// step completes them).
    Escalation,
}

/// How an iteration made the next orbitals (`ScfIteration::step`).
pub(crate) type StepName = &'static str;

/// The accelerators' settings and state, in the SCF's basis (the overlap S and the orthonormal
/// basis X).
pub(crate) struct Accelerators<'m> {
    algo: ScfAlgorithm,
    s: MatRef<'m, f64>,
    x: MatRef<'m, f64>,
    par: Par,
    progress: Progress,
    diis_rule: DiisRule,
    diis: Diis,
    adiis: Adiis,
    soscf: Soscf,
    /// SOSCF took over (it keeps the steps once it has).
    soscf_on: bool,
    /// Quasi-Newton steps since a restart.
    descent: bool,
    /// The escalation ran: no level shift, no damping.
    escalated: bool,
    /// The previous (damped) Fock matrices.
    f_damp: Option<(Mat<f64>, Mat<f64>)>,
    /// The next SOSCF step first completes the orbitals from the occupied ones (the escalation).
    complete: bool,
}

impl<'m> Accelerators<'m> {
    pub fn new(
        algo: &ScfAlgorithm,
        s: MatRef<'m, f64>,
        x: MatRef<'m, f64>,
        par: Par,
        progress: Progress,
    ) -> Self {
        Accelerators {
            algo: algo.clone(),
            s,
            x,
            par,
            progress,
            diis_rule: DiisRule::new(algo),
            diis: Diis::with_drop(algo.diis_space, algo.diis_drop),
            adiis: Adiis::new(algo.diis_space),
            soscf: Soscf::new(),
            soscf_on: false,
            descent: false,
            escalated: false,
            f_damp: None,
            complete: false,
        }
    }

    /// The steps are quasi-Newton since a restart.
    pub fn descending(&self) -> bool {
        self.descent
    }

    /// The SCF trace of the iterations of `history` from `from` on with the accelerators' state
    /// (`LINUS_SCF_TRACE`): returns the count of traced entries.
    pub fn trace(&self, history: &[ScfIteration], from: usize) -> usize {
        let (diis, adiis) = (&self.diis, self.adiis.len());
        trace::host(
            history,
            from,
            diis.focks.len(),
            diis.drop,
            adiis,
            self.soscf.hist.len(),
        )
    }

    /// A restart (see the module docs).
    pub fn restart(&mut self, r: Restart) {
        self.diis = Diis::with_drop(self.algo.diis_space, self.algo.diis_drop);
        self.adiis.clear();
        self.soscf = match r {
            Restart::Descent => Soscf::new(),
            Restart::Escalation => Soscf::with_relative_curvature(),
        };
        self.soscf_on = false;
        self.descent = true;
        self.f_damp = None;
        if r == Restart::Escalation {
            self.escalated = true;
            self.complete = true;
        }
    }

    /// The Fock operator changed by `df` per channel at the current density (the J/K builder's
    /// next approximation): with `grid_switch_shift`, the stored Fock matrices (DIIS, with their
    /// errors recomputed; ADIIS; the damping's) move with it, so DIIS continues on the new operator
    /// instead of mixing two. SOSCF's history starts again either way.
    pub fn operator_changed(&mut self, df: &[Mat<f64>], timers: &mut Timers) {
        if self.algo.grid_switch_shift {
            let (s, x, par) = (self.s, self.x, self.par);
            timers.time("diis", || self.diis.shift(df, s, x, par));
            self.adiis.shift(df);
            if let Some((pa, pb)) = self.f_damp.as_mut() {
                *pa += &df[0];
                *pb += &df[df.len() - 1];
            }
        }
        self.soscf.hist.clear();
        self.soscf.last = None;
    }

    /// The loop rebuilds at the same density (the confirmation of an incremental build): SOSCF's
    /// next step is not a step from this one.
    pub fn forget_last_step(&mut self) {
        self.soscf.last = None;
    }

    /// The step of an iteration from its evaluation `ev`: the ADIIS history and the DIIS subspace
    /// take its Fock matrices, then SOSCF, or an extrapolation with the level shift and a
    /// diagonalization, makes the new orbitals of `st`. Returns the step's name, or Err with
    /// "non_finite" when the extrapolated Fock matrix could not be diagonalized.
    pub fn step(
        &mut self,
        ev: FockStep<'_>,
        st: &mut OrbitalState,
        timers: &mut Timers,
    ) -> Result<StepName, StepName> {
        let FockStep {
            iter: it,
            usable,
            keep_occ,
            fa,
            fb,
            ea,
            eb,
            gmax,
        } = ev;
        let (restricted, par, s, x) = (st.restricted, self.par, self.s, self.x);
        let a = &self.algo;
        if a.adiis && usable {
            let f = if restricted {
                vec![fa.clone()]
            } else {
                vec![fa.clone(), fb.clone()]
            };
            self.adiis.push(st.densities(), f);
        }
        let damping = a.damping > 0.0 && gmax > a.damping_until && !self.escalated;
        if self.diis_rule.admits(it, usable) {
            // with the occupied orbitals of D (for the shift at a change of approximation)
            let oc = if keep_occ {
                st.occ.clone().unwrap_or_default()
            } else {
                vec![]
            };
            let e_new = if restricted { vec![ea] } else { vec![ea, eb] };
            // the SCF has left the region of the best stored entries: drop the oldest from now on
            let (norm, stored) = (err_norm(&e_new), self.diis.norms());
            if self.diis_rule.unpins(self.diis.drop, norm, &stored) {
                self.diis.drop = DiisDrop::Oldest;
                if self.progress.on() {
                    let emin = stored.iter().copied().fold(f64::INFINITY, f64::min);
                    self.progress.line(format_args!("          DIIS: error {:.1}x the smallest stored; dropping the oldest entries from now on", (norm / emin).sqrt()));
                }
            }
            let f_new = if restricted {
                vec![fa.clone()]
            } else {
                vec![fa.clone(), fb.clone()]
            };
            self.diis.push_occ(f_new, e_new, oc);
        }
        if (a.soscf || self.descent)
            && usable
            && st.occ.is_some()
            && (self.soscf_on || self.descent || gmax < a.soscf_start)
            && !damping
        {
            self.soscf_on = true;
            if std::mem::take(&mut self.complete) {
                // the escalation's start: full orbitals from its occupied ones
                let o = st.occ.as_ref().expect("occupied orbitals");
                let Some((_, c_a)) = try_semicanonical(&o[0], fa, x, s, par) else {
                    return Err("non_finite");
                };
                st.c_b = if restricted {
                    c_a.clone()
                } else {
                    let Some((_, c_b)) = try_semicanonical(&o[1], fb, x, s, par) else {
                        return Err("non_finite");
                    };
                    c_b
                };
                st.c_a = c_a;
            }
            timers.time("soscf", || {
                let (cs, fs) = if restricted {
                    (vec![st.c_a.clone()], vec![fa])
                } else {
                    (vec![st.c_a.clone(), st.c_b.clone()], vec![fa, fb])
                };
                let mut cn = self.soscf.step(&cs, &fs, &st.nocc[..cs.len()], par);
                let c_a = cn.swap_remove(0);
                let c_b = if restricted {
                    c_a.clone()
                } else {
                    cn.swap_remove(0)
                };
                st.set_orbitals(c_a, c_b, par);
            });
            return Ok("soscf");
        }
        let use_adiis = a.adiis && gmax > a.adiis_switch && self.adiis.len() >= 2;
        let (name, fa_x, fb_x) = if use_adiis {
            let fx = self.adiis.extrapolate(if restricted { 2.0 } else { 1.0 });
            let fb_x = fx[fx.len() - 1].clone();
            ("adiis", fx[0].clone(), fb_x)
        } else if damping {
            let (fpa, fpb) = self.f_damp.take().unwrap_or((fa.clone(), fb.clone()));
            let w = a.damping;
            (
                "damped",
                fa * (1.0 - w) + &fpa * w,
                fb * (1.0 - w) + &fpb * w,
            )
        } else if self
            .diis_rule
            .extrapolates(it, usable, self.diis.focks.len())
        {
            let f = timers.time("diis", || self.diis.extrapolate());
            ("diis", f[0].clone(), f[f.len() - 1].clone())
        } else {
            ("roothaan", fa.clone(), fb.clone())
        };
        if damping {
            self.f_damp = Some((fa_x.clone(), fb_x.clone()));
        }
        // F + shift (S - S D S) raises the virtual levels
        let shift = level_shift(a, gmax, self.escalated);
        let shifted = |f: Mat<f64>, d: &Mat<f64>| match shift {
            Some(v) => f + (s.to_owned() - mm(mm(s, d.as_ref(), par).as_ref(), s, par)) * v,
            None => f,
        };
        let ok = timers.time("diag", || {
            let Some((e, c)) = try_diagonalize(shifted(fa_x, &st.da).as_ref(), x, par) else {
                return false;
            };
            (st.eps_a, st.c_a) = (e, c);
            if restricted {
                st.eps_b = st.eps_a.clone();
                st.c_b = st.c_a.clone();
            } else {
                let Some((e, c)) = try_diagonalize(shifted(fb_x, &st.db).as_ref(), x, par) else {
                    return false;
                };
                (st.eps_b, st.c_b) = (e, c);
            }
            st.refresh_aufbau(par);
            true
        });
        if ok {
            Ok(name)
        } else {
            Err("non_finite")
        }
    }
}

/// What a step reads of an iteration: its number, whether its Fock matrix may be extrapolated
/// (not a guess density's), its Fock matrices, their orbital gradients (moved into the DIIS
/// subspace) and max |F_ai|.
pub(crate) struct FockStep<'a> {
    pub iter: usize,
    pub usable: bool,
    /// The DIIS entry keeps the occupied orbitals of its density (the subspace shift of a later
    /// change of the J/K approximation).
    pub keep_occ: bool,
    pub fa: &'a Mat<f64>,
    pub fb: &'a Mat<f64>,
    pub ea: Mat<f64>,
    pub eb: Mat<f64>,
    pub gmax: f64,
}

/// The level shift of the virtual orbitals (Eh) of an iteration with max |F_ai| `gmax`, applied as
/// F + shift (S - S D S) while max |F_ai| > `level_shift_until` (`level_shift`; not after the
/// escalation). Default 0.25 Eh until 1e-2.
pub fn level_shift(a: &ScfAlgorithm, gmax: f64, escalated: bool) -> Option<f64> {
    (a.level_shift != 0.0 && gmax > a.level_shift_until && !escalated).then_some(a.level_shift)
}
