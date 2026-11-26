//! Fock evaluation: the only place the SCF calls its J/K builder, XC term and environment term.
//!
//! A `FockEvaluator` evaluates the Fock matrices F_s = h + J[D_a + D_b] - a_x K\[D_s\] + V_s
//! (V_s = V_xc,s + V_env), the energy and the orbital gradient X^T (F_s D_s S - S D_s F_s) X of a
//! state, in one of two modes:
//! - **accepted** (`FockEvaluator::evaluate`): the SCF's iterate. With a builder whose cost falls
//!   with |D| the build is incremental (J and K of D_n - D_n-1 added to the last build's, a full
//!   build every `incremental_reset` builds or when the loop asks for one); the evaluator keeps what
//!   the next incremental build needs, times the build ("jk", "xc" and the environment term's
//!   name), and moves the J/K builder to its next approximation when the loop asks
//!   (`FockEvaluator::next_approximation`). A build is also incremental when a term built from its
//!   own history (COSX's incremental K, the PCM's predicted charges or incremental passes:
//!   `built_incrementally`); a full build the loop asks for resets every term's history, so the
//!   SCF can end on a build of the density alone (the loop's confirmation, `super::control`);
//! - **trial** (`FockEvaluator::trial`): the UHF/UKS Fock matrices and energy of rotated orbitals,
//!   for the stability analysis, the mode check and the closed-shell probe. Full builds that leave
//!   the accepted state (the incremental builds, the last density) untouched. What the terms keep
//!   of the last state they saw (a solvent's surface charges, the PCM's Coulomb fit, VV10's last
//!   energy, read by the result) is put back before the SCF reads it again: by
//!   `FockEvaluator::settle`, one build at the accepted state, or by the loop's next accepted
//!   evaluation, which after trials is always a full build.

use super::{OrbitalState, ScfAlgorithm, ScfInput};
use faer::{Mat, MatRef, Par};
use linus_base::error::LinusError;
use linus_base::linalg::{dot, max_abs, mm};
use linus_base::terms::{
    exchange_parts, EnvironmentTerm, OccupiedOrbitals, XcDensity, XcPotential,
};
use linus_base::terms::{DensityFactors, JkBuilder, JkOutput, JkRequest};
use linus_core::timing::Timers;
use std::time::Instant;

/// The density-dependent terms of an SCF's Fock matrix: the J/K builder, the XC term
/// (Kohn-Sham) and an environment term (a solvent's reaction field).
pub(crate) struct FockTerms<'a> {
    pub jk: &'a mut dyn JkBuilder,
    pub xc: Option<&'a mut dyn XcPotential>,
    pub env: Option<&'a mut dyn EnvironmentTerm>,
}

impl<'a> FockTerms<'a> {
    /// The terms, borrowed for as long as all three are.
    pub fn new<'j: 'a, 'x: 'a, 'e: 'a>(
        jk: &'j mut (dyn JkBuilder + 'j),
        xc: Option<&'x mut (dyn XcPotential + 'x)>,
        env: Option<&'e mut (dyn EnvironmentTerm + 'e)>,
    ) -> Self {
        FockTerms {
            jk,
            xc: xc.map(|x| x as _),
            env: env.map(|e| e as _),
        }
    }

    /// (E_xc, E_env, V_a, V_b, seconds in XC): E_xc and the potentials V_s = V_xc,s + V_env of the
    /// XC term and the environment term at `d`, timed as "xc" and the term's name when `timers`
    /// are given.
    fn xc_and_environment(
        &mut self,
        d: &XcDensity<'_>,
        par: Par,
        timers: Option<&mut Timers>,
    ) -> Result<XcEnvParts, LinusError> {
        let nbf = d.da.nrows();
        let t0 = Instant::now();
        let (exc, va, vb) = match self.xc.as_deref_mut() {
            Some(x) => x.compute(d, par)?,
            None => (0.0, Mat::zeros(nbf, nbf), Mat::zeros(nbf, nbf)),
        };
        let t_xc = t0.elapsed().as_secs_f64();
        let Some(env) = self.env.as_deref_mut() else {
            if let (Some(t), true) = (timers, self.xc.is_some()) {
                t.add("xc", t_xc);
            }
            return Ok((exc, 0.0, va, vb, t_xc));
        };
        let t1 = Instant::now();
        let (e_env, v_env) = env.compute(d, par)?;
        let (va, vb) = match v_env {
            Some(v) => (&va + &v, &vb + &v),
            None => (va, vb),
        };
        if let Some(t) = timers {
            if self.xc.is_some() {
                t.add("xc", t_xc);
            }
            t.add(env.name(), t1.elapsed().as_secs_f64());
        }
        Ok((exc, e_env, va, vb, t_xc))
    }
}

/// (E_xc, E_env, V_a, V_b, seconds in XC) of [`FockTerms::xc_and_environment`].
type XcEnvParts = (f64, f64, Mat<f64>, Mat<f64>, f64);

/// The energy parts of an evaluation (Eh): one-electron, Coulomb, exchange (exact-exchange
/// fractions included), XC and environment.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct EnergyParts {
    pub one: f64,
    pub coulomb: f64,
    pub exchange: f64,
    pub xc: f64,
    pub environment: f64,
}

/// One accepted evaluation: the Fock matrices, the orbital gradients (the DIIS errors), the
/// energy and the iteration's scalars.
pub(crate) struct FockEval {
    pub fa: Mat<f64>,
    pub fb: Mat<f64>,
    /// X^T (F_s D_s S - S D_s F_s) X per spin (restricted: `eb` is unused, 1 x 1).
    pub ea: Mat<f64>,
    pub eb: Mat<f64>,
    pub parts: EnergyParts,
    pub energy: f64,
    /// max |F_ai| and RMS of the gradient elements, and PySCF's norm.
    pub gmax: f64,
    pub grms: f64,
    pub gnorm_pyscf: f64,
    /// max and RMS of D_n - D_n-1 (0 at the first build).
    pub ddmax: f64,
    pub ddrms: f64,
    /// The build was incremental: the SCF's own (J/K of D_n - D_n-1) or a term's
    /// (`built_incrementally`: COSX's incremental K, the PCM's predicted charges).
    pub incremental: bool,
    pub t_jk: f64,
    pub t_xc: f64,
    /// When the Fock assembly started (the iteration's step time counts from there).
    pub assembled_at: Instant,
    // J, the exchange matrices and V of the evaluation: a change of the J/K approximation
    // reassembles F with the new K
    j: Mat<f64>,
    ka: Mat<f64>,
    kb: Mat<f64>,
    va: Mat<f64>,
    vb: Mat<f64>,
    /// The exchange energy when the builder reports it (`JkOutput::k_energy`), fractions and
    /// sign included; None: -1/2 (D_a . K_a + D_b . K_b).
    ex_energy: Option<f64>,
}

/// The SCF's Fock evaluations (see the module docs).
pub(crate) struct FockEvaluator<'t, 'm> {
    terms: FockTerms<'t>,
    h: MatRef<'m, f64>,
    pub(super) s: MatRef<'m, f64>,
    /// The orthonormal basis X (nbf x nmo) of the orbital gradient.
    pub(super) x: MatRef<'m, f64>,
    e_nuc: f64,
    /// The exact-exchange fraction applied to K (`linus_base::terms::exchange_parts`); 0: no K is
    /// built.
    a_x: f64,
    restricted: bool,
    pub(super) par: Par,
    /// The occupations of the occupied orbitals per channel (all 1), for the XC term.
    occ_n: Vec<Vec<f64>>,
    incremental: bool,
    incremental_reset: usize,
    /// The J/K output of the last accepted build, which an incremental build adds to.
    last_out: Option<JkOutput>,
    /// The densities of the last accepted build (D_n-1 for the next one; a resumed SCF's).
    pub(super) last_dens: Option<Vec<Mat<f64>>>,
    since_full: usize,
    /// Accepted builds that were incremental ([`FockEval::incremental`]).
    pub(super) incremental_builds: usize,
    /// The builder may have further approximations (only K builders have them).
    more_approx: bool,
    /// The J/K approximations moved through (COSX: 0 on the first grid).
    pub(super) approx: usize,
    /// Trial evaluations since the last accepted state was put back.
    trials: bool,
}

impl<'t, 'm> FockEvaluator<'t, 'm> {
    /// The evaluations of the SCF of `inp` in the orthonormal basis `x` (`restricted`: one
    /// channel), with the incremental builds of `algo` when the builder is capable of them.
    pub fn new(
        terms: FockTerms<'t>,
        inp: &ScfInput<'m>,
        x: MatRef<'m, f64>,
        restricted: bool,
        algo: &ScfAlgorithm,
        par: Par,
    ) -> Self {
        let a_x = exchange_parts(inp.exchange).0;
        let (na, nb) = (inp.nalpha, inp.nbeta);
        FockEvaluator {
            incremental: algo.incremental && terms.jk.incremental_capable(),
            terms,
            h: inp.h,
            s: inp.s,
            x,
            e_nuc: inp.e_nuc,
            a_x,
            restricted,
            par,
            occ_n: if restricted {
                vec![vec![1.0; na]]
            } else {
                vec![vec![1.0; na], vec![1.0; nb]]
            },
            incremental_reset: algo.incremental_reset,
            last_out: None,
            last_dens: None,
            since_full: 0,
            incremental_builds: 0,
            more_approx: a_x != 0.0,
            approx: 0,
            trials: false,
        }
    }

    fn want_k(&self) -> bool {
        self.a_x != 0.0
    }

    /// The builder may still move to a next approximation.
    pub fn may_switch(&self) -> bool {
        self.more_approx && self.want_k()
    }

    /// The exchange energy -a_x sum_s E_K,s of a builder's reported energies `e` (one per
    /// channel; `restricted`: one channel's for both); None without them or without exact
    /// exchange.
    fn exchange_energy(&self, e: Option<&Vec<f64>>, restricted: bool) -> Option<f64> {
        let e = e.filter(|_| self.want_k())?;
        Some(if restricted {
            -self.a_x * 2.0 * e[0]
        } else {
            -self.a_x * (e[0] + e[1])
        })
    }

    /// a_x K_s per spin of the exchange matrices `k` (`restricted`: one channel's for both; zero
    /// without exact exchange).
    fn exchange(&self, k: &[Mat<f64>], restricted: bool) -> (Mat<f64>, Mat<f64>) {
        let nbf = self.h.nrows();
        let kx = |s: usize| -> Mat<f64> {
            if !self.want_k() {
                return Mat::zeros(nbf, nbf);
            }
            &k[s] * self.a_x
        };
        if restricted {
            let k = kx(0);
            (k.clone(), k)
        } else {
            (kx(0), kx(1))
        }
    }

    /// The orbital gradient X^T (F D S - S D F) X (the DIIS error) of `f` at the density `d`.
    fn gradient(&self, f: &Mat<f64>, d: &Mat<f64>) -> Mat<f64> {
        let par = self.par;
        let fds = mm(mm(f.as_ref(), d.as_ref(), par).as_ref(), self.s, par);
        let e = &fds - fds.transpose();
        mm(
            self.x.transpose(),
            mm(e.as_ref(), self.x, par).as_ref(),
            par,
        )
    }

    /// The Fock matrices F_s = h + J - a_x K_s + V_s of an evaluation's J, K and V at the state
    /// `st`, the exchange energy, the gradients and their norms, and the energy.
    fn complete(&self, st: &OrbitalState, ev: &mut FockEval) {
        let (h, da, db) = (self.h, &st.da, &st.db);
        ev.fa = h.to_owned() + &ev.j - &ev.ka + &ev.va;
        ev.fb = h.to_owned() + &ev.j - &ev.kb + &ev.vb;
        ev.parts.exchange = match ev.ex_energy {
            Some(e) => e,
            None => -0.5 * (dot(da.as_ref(), ev.ka.as_ref()) + dot(db.as_ref(), ev.kb.as_ref())),
        };
        ev.ea = self.gradient(&ev.fa, da);
        ev.eb = if self.restricted {
            Mat::zeros(1, 1)
        } else {
            self.gradient(&ev.fb, db)
        };
        let (ea, eb) = (&ev.ea, &ev.eb);
        (ev.gmax, ev.grms) = if self.restricted {
            max_rms(&[ea])
        } else {
            max_rms(&[ea, eb])
        };
        // PySCF's norm: RHF ||2 F_vo|| = sqrt(2) ||E||_F; UHF sqrt((||E_a||^2 + ||E_b||^2) / 2)
        ev.gnorm_pyscf = if self.restricted {
            (2.0 * dot(ea.as_ref(), ea.as_ref())).sqrt()
        } else {
            (0.5 * (dot(ea.as_ref(), ea.as_ref()) + dot(eb.as_ref(), eb.as_ref()))).sqrt()
        };
        // the XC and environment energies as one term (the energy of the solvent's former XC
        // slot, bit for bit)
        let p = &ev.parts;
        ev.energy = p.one + p.coulomb + p.exchange + (p.xc + p.environment) + self.e_nuc;
    }

    /// The J/K request of the densities `dens` of the state `st` (its occupied orbitals as factors).
    fn request<'s>(&self, dens: &'s [Mat<f64>], st: &'s OrbitalState) -> JkRequest<'s> {
        JkRequest {
            dens,
            factors: st.occ.as_deref().map(DensityFactors::Symmetric),
            want_j: true,
            want_k: self.want_k(),
        }
    }

    /// The accepted evaluation at the state `st`: incremental when the builder supports it, a full
    /// build when `rebuild` (or every `incremental_reset` builds, or after trial evaluations).
    pub fn evaluate(
        &mut self,
        st: &OrbitalState,
        rebuild: bool,
        timers: &mut Timers,
    ) -> Result<FockEval, LinusError> {
        let par = self.par;
        let dens = st.densities();
        // trials are always followed by a full rebuild (it puts back what they changed)
        let rebuild = rebuild || std::mem::take(&mut self.trials);
        let t_jk0 = Instant::now();
        let use_incr = self.incremental
            && !rebuild
            && self.last_out.is_some()
            && self.since_full < self.incremental_reset;
        // D_n - D_n-1
        let dd: Option<Vec<Mat<f64>>> = self
            .last_dens
            .as_ref()
            .map(|pd| dens.iter().zip(pd).map(|(a, b)| a - b).collect());
        let (ddmax, ddrms) = match &dd {
            Some(dd) => max_rms(&dd.iter().collect::<Vec<_>>()),
            None => (0.0, 0.0),
        };
        let out = if use_incr {
            let dd = dd.as_ref().expect("previous density");
            let req = JkRequest::new(dd, true, self.want_k());
            let delta = self.terms.jk.compute_delta(&req, par)?;
            self.since_full += 1;
            add_out(self.last_out.as_ref().expect("a last build"), delta)
        } else {
            if rebuild {
                self.terms.jk.reset_incremental();
                if let Some(env) = self.terms.env.as_deref_mut() {
                    env.reset_incremental();
                }
            }
            self.since_full = 0;
            let req = self.request(&dens, st);
            self.terms.jk.compute(&req, par)?
        };
        let t_jk = t_jk0.elapsed().as_secs_f64();
        timers.add("jk", t_jk);
        let (ka, kb) = self.exchange(&out.k, self.restricted);
        let ex_energy = self.exchange_energy(out.k_energy.as_ref(), self.restricted);
        let j = if self.restricted {
            &out.j[0] * 2.0
        } else {
            &out.j[0] + &out.j[1]
        };
        let xd = xc_density(st, self.restricted, &self.occ_n);
        let (exc, e_env, va, vb, t_xc) =
            self.terms
                .xc_and_environment(&xd, par, Some(&mut *timers))?;
        let incremental = use_incr
            || self.terms.jk.built_incrementally()
            || self
                .terms
                .env
                .as_deref()
                .is_some_and(|e| e.built_incrementally());
        if incremental {
            self.incremental_builds += 1;
        }
        let dt = &st.da + &st.db;
        let parts = EnergyParts {
            one: dot(dt.as_ref(), self.h),
            coulomb: 0.5 * dot(dt.as_ref(), j.as_ref()),
            exchange: 0.0,
            xc: exc,
            environment: e_env,
        };
        let mut ev = FockEval {
            fa: Mat::new(),
            fb: Mat::new(),
            ea: Mat::new(),
            eb: Mat::new(),
            parts,
            energy: 0.0,
            gmax: 0.0,
            grms: 0.0,
            gnorm_pyscf: 0.0,
            ddmax,
            ddrms,
            incremental,
            t_jk,
            t_xc,
            assembled_at: Instant::now(),
            j,
            ka,
            kb,
            va,
            vb,
            ex_energy,
        };
        self.complete(st, &mut ev);
        timers.add("diis", ev.assembled_at.elapsed().as_secs_f64());
        if self.incremental {
            self.last_out = Some(out);
        }
        self.last_dens = Some(dens);
        Ok(ev)
    }

    /// Moves the J/K builder to its next approximation (COSX: the next SCF grid) and rebuilds K at
    /// the evaluated state `st`: the evaluation's Fock matrices, exchange energy, energy and
    /// gradients become those of the new operator. Returns the change of the Fock matrices,
    /// K_old - K_new per channel (fractions included), for the accelerators' stored ones; None when
    /// the builder has none left (it is not asked again).
    pub fn next_approximation(
        &mut self,
        st: &OrbitalState,
        ev: &mut FockEval,
        timers: &mut Timers,
    ) -> Result<Option<Vec<Mat<f64>>>, LinusError> {
        if !self.terms.jk.next_approximation() {
            self.more_approx = false;
            return Ok(None);
        }
        self.approx += 1;
        let t_k = Instant::now();
        let dens = self.last_dens.as_deref().expect("an evaluated density");
        let req = JkRequest {
            want_j: false,
            ..self.request(dens, st)
        };
        let out = self.terms.jk.compute(&req, self.par)?;
        let k = out.k;
        let dt_k = t_k.elapsed().as_secs_f64();
        ev.t_jk += dt_k;
        timers.add("jk", dt_k);
        let (ka, kb) = self.exchange(&k, self.restricted);
        ev.ex_energy = self.exchange_energy(out.k_energy.as_ref(), self.restricted);
        if let Some(out) = self.last_out.as_mut() {
            out.k = k;
        }
        // F_new - F_old = K_old - K_new
        let dk = if self.restricted {
            vec![&ev.ka - &ka]
        } else {
            vec![&ev.ka - &ka, &ev.kb - &kb]
        };
        (ev.ka, ev.kb) = (ka, kb);
        let t = Instant::now();
        self.complete(st, ev);
        timers.add("diis", t.elapsed().as_secs_f64());
        Ok(Some(dk))
    }

    /// A trial evaluation: the UHF/UKS Fock matrices [F_a, F_b] and the energy of the orbitals `c`
    /// (per spin, nbf x nmo, the first `nocc[s]` occupied). A full build of both spin densities, the
    /// polarized XC term and the environment term at them, untimed; the accepted state is
    /// untouched (see the module docs).
    pub fn trial(
        &mut self,
        c: &[Mat<f64>],
        nocc: &[usize],
    ) -> Result<(Vec<Mat<f64>>, f64), LinusError> {
        self.trials = true;
        let (par, h) = (self.par, self.h);
        let o: Vec<Mat<f64>> = c
            .iter()
            .zip(nocc)
            .map(|(c, &no)| c.subcols(0, no).to_owned())
            .collect();
        let d: Vec<Mat<f64>> = o
            .iter()
            .map(|c| mm(c.as_ref(), c.transpose(), par))
            .collect();
        let req = JkRequest {
            factors: Some(DensityFactors::Symmetric(&o)),
            ..JkRequest::new(&d, true, self.want_k())
        };
        let out = self.terms.jk.compute(&req, par)?;
        let j = &out.j[0] + &out.j[1];
        let (ka, kb) = self.exchange(&out.k, false);
        let ex_energy = self.exchange_energy(out.k_energy.as_ref(), false);
        let occ_n: Vec<Vec<f64>> = nocc.iter().map(|&n| vec![1.0; n]).collect();
        let xd = XcDensity {
            da: &d[0],
            db: &d[1],
            restricted: false,
            orbitals: Some(OccupiedOrbitals { c: &o, n: &occ_n }),
        };
        let (exc, e_env, va, vb, _) = self.terms.xc_and_environment(&xd, par, None)?;
        let dt = &d[0] + &d[1];
        let ex = match ex_energy {
            Some(e) => e,
            None => -0.5 * (dot(d[0].as_ref(), ka.as_ref()) + dot(d[1].as_ref(), kb.as_ref())),
        };
        let e = dot(dt.as_ref(), h)
            + 0.5 * dot(dt.as_ref(), j.as_ref())
            + ex
            + (exc + e_env)
            + self.e_nuc;
        Ok((
            vec![h.to_owned() + &j - &ka + &va, h.to_owned() + &j - &kb + &vb],
            e,
        ))
    }

    /// After trial evaluations, when the SCF reads its terms' state next (the result, not another
    /// iteration): one build at the accepted state `st`, its output unused, puts back what the
    /// terms keep of the last state they saw (a solvent's surface charges and the PCM's Coulomb fit,
    /// VV10's last energy and statistics), of the density alone (the environment term's history
    /// reset first). Only when a term keeps state (an environment term; an XC term with VV10; the
    /// J/K builder's own such state is the PCM's Coulomb fit, set only with an environment term);
    /// the J/K builder's incremental state is reset either way. Returns whether it built.
    pub fn settle(&mut self, st: &OrbitalState) -> Result<bool, LinusError> {
        let keeps =
            self.terms.env.is_some() || self.terms.xc.as_ref().is_some_and(|x| x.keeps_state());
        if let Some(env) = self.terms.env.as_deref_mut() {
            env.reset_incremental();
        }
        if keeps {
            let dens = st.densities();
            let req = self.request(&dens, st);
            drop(self.terms.jk.compute(&req, self.par)?);
            let xd = xc_density(st, self.restricted, &self.occ_n);
            self.terms.xc_and_environment(&xd, self.par, None)?;
        }
        self.terms.jk.reset_incremental();
        self.trials = false;
        Ok(keeps)
    }
}

/// The XC term's view of the state `st` (`restricted`: one channel; `occ_n` the occupations).
fn xc_density<'s>(st: &'s OrbitalState, restricted: bool, occ_n: &'s [Vec<f64>]) -> XcDensity<'s> {
    XcDensity {
        da: &st.da,
        db: &st.db,
        restricted,
        orbitals: st.occ.as_deref().map(|c| OccupiedOrbitals { c, n: occ_n }),
    }
}

/// The incremental build's J/K: the last build's plus that of D_n - D_n-1 (builders that report
/// their own exchange energies are affine in D, not incremental: none here).
fn add_out(prev: &JkOutput, delta: JkOutput) -> JkOutput {
    debug_assert!(prev.k_energy.is_none() && delta.k_energy.is_none());
    let add = |a: &Vec<Mat<f64>>, b: Vec<Mat<f64>>| -> Vec<Mat<f64>> {
        if b.is_empty() {
            return a.clone();
        }
        a.iter().zip(b).map(|(x, y)| x + y).collect()
    };
    JkOutput {
        j: add(&prev.j, delta.j),
        k: add(&prev.k, delta.k),
        k_energy: None,
    }
}

/// max and RMS of the elements of the matrices.
fn max_rms(ms: &[&Mat<f64>]) -> (f64, f64) {
    let mut mx: f64 = 0.0;
    let mut ss = 0.0;
    let mut cnt = 0usize;
    for m in ms {
        mx = mx.max(max_abs(m.as_ref()));
        ss += dot(m.as_ref(), m.as_ref());
        cnt += m.nrows() * m.ncols();
    }
    (mx, (ss / cnt.max(1) as f64).sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// J = D and K = D for every density: a linear, deterministic model builder that counts its
    /// builds and the resets of its incremental state.
    #[derive(Default)]
    struct EchoJk {
        builds: usize,
        resets: usize,
    }

    impl JkBuilder for EchoJk {
        fn name(&self) -> String {
            "echo".into()
        }

        fn compute(&mut self, req: &JkRequest, _par: Par) -> Result<JkOutput, LinusError> {
            self.builds += 1;
            Ok(JkOutput {
                j: req.dens.to_vec(),
                k: req.dens.to_vec(),
                k_energy: None,
            })
        }

        fn reset_incremental(&mut self) {
            self.resets += 1;
        }
    }

    /// An environment term that keeps the trace of the last total density it saw (as a solvent
    /// keeps its surface charges).
    #[derive(Default)]
    struct Remembers {
        last: f64,
    }

    impl EnvironmentTerm for Remembers {
        fn name(&self) -> &'static str {
            "solvent"
        }

        fn compute(
            &mut self,
            d: &XcDensity<'_>,
            _par: Par,
        ) -> Result<(f64, Option<Mat<f64>>), LinusError> {
            let n = d.da.nrows();
            self.last = (0..n).map(|i| d.da[(i, i)] + d.db[(i, i)]).sum();
            Ok((-0.01 * self.last, None))
        }
    }

    /// An environment term that builds from its history (as the PCM's predicted charges) until
    /// the SCF asks for a full build.
    #[derive(Default)]
    struct Predicts {
        full_next: bool,
        last_incremental: bool,
        resets: usize,
        full_builds: usize,
    }

    impl EnvironmentTerm for Predicts {
        fn name(&self) -> &'static str {
            "solvent"
        }

        fn compute(
            &mut self,
            d: &XcDensity<'_>,
            _par: Par,
        ) -> Result<(f64, Option<Mat<f64>>), LinusError> {
            self.last_incremental = !std::mem::take(&mut self.full_next);
            self.full_builds += usize::from(!self.last_incremental);
            let n = d.da.nrows();
            Ok((
                -0.01,
                Some(Mat::from_fn(n, n, |i, j| 1e-3 * (i + j) as f64)),
            ))
        }

        fn built_incrementally(&self) -> bool {
            self.last_incremental
        }

        fn reset_incremental(&mut self) {
            self.resets += 1;
            self.full_next = true;
        }
    }

    /// A term's own incremental build makes the evaluation incremental (and counted); a full
    /// build the loop asks for resets the term first, so it is of the density alone.
    #[test]
    fn a_term_built_from_its_history_is_an_incremental_build() {
        let n = 3;
        let h = Mat::from_fn(n, n, |i, j| if i == j { i as f64 - 1.0 } else { -0.1 });
        let s = Mat::<f64>::identity(n, n);
        let x = Mat::<f64>::identity(n, n);
        let inp = ScfInput {
            h: h.as_ref(),
            s: s.as_ref(),
            e_nuc: 0.0,
            nalpha: 1,
            nbeta: 1,
            exchange: None,
            guess_density: None,
            geometry: None,
        };
        let algo = ScfAlgorithm::default();
        let mut st = OrbitalState::new(true, [1, 1], n, Mat::zeros(n, n), Mat::zeros(n, n));
        st.set_orbitals(orbitals(n, 0.1), orbitals(n, 0.1), Par::Seq);
        let (mut jk, mut env) = (EchoJk::default(), Predicts::default());
        let mut timers = Timers::default();
        {
            let terms = FockTerms::new(&mut jk, None, Some(&mut env));
            let mut fock = FockEvaluator::new(terms, &inp, x.as_ref(), true, &algo, Par::Seq);
            let a = fock.evaluate(&st, false, &mut timers).unwrap();
            assert!(a.incremental && fock.incremental_builds == 1);
            let b = fock.evaluate(&st, true, &mut timers).unwrap();
            assert!(!b.incremental && fock.incremental_builds == 1);
            assert_eq!(a.energy.to_bits(), b.energy.to_bits());
        }
        assert_eq!((env.resets, env.full_builds, jk.resets), (1, 1, 1));
    }

    /// An SCF whose term builds from its history ends on a confirming build of the density
    /// alone: the history's last build is full, after a "rebuild" step at the converged iterate.
    #[test]
    fn the_scf_ends_on_a_build_of_the_density_alone() {
        let h = Mat::from_fn(
            2,
            2,
            |i, j| if i == j { -1.0 + 0.5 * i as f64 } else { -0.2 },
        );
        let s = Mat::<f64>::identity(2, 2);
        let inp = ScfInput {
            h: h.as_ref(),
            s: s.as_ref(),
            e_nuc: 0.0,
            nalpha: 1,
            nbeta: 1,
            exchange: None,
            guess_density: None,
            geometry: None,
        };
        let (mut jk, mut env) = (EchoJk::default(), Predicts::default());
        let opts = crate::ScfConfig::default();
        let r = crate::run_scf(&inp, &mut jk, None, Some(&mut env), &opts, Par::Seq).unwrap();
        assert!(r.converged);
        let n = r.history.len();
        let (last, before) = (&r.history[n - 1], &r.history[n - 2]);
        assert_eq!(
            (last.fock, before.fock, before.step),
            ("full", "incremental", "rebuild")
        );
        assert_eq!((env.resets, env.full_builds), (1, 1));
        assert_eq!(r.incremental_builds, n - 1);
        assert_eq!(last.energy.to_bits(), before.energy.to_bits());
    }

    fn orbitals(n: usize, angle: f64) -> Mat<f64> {
        let mut c = Mat::<f64>::identity(n, n);
        let (s, co) = angle.sin_cos();
        (c[(0, 0)], c[(1, 0)], c[(0, 1)], c[(1, 1)]) = (co, s, -s, co);
        c
    }

    /// Trial evaluations change neither the accepted state's incremental builds nor, once settled,
    /// what the terms keep of the last state; with no term that keeps state, settling builds
    /// nothing.
    #[test]
    fn trial_evaluations_leave_no_state_behind() {
        let n = 3;
        let h = Mat::from_fn(n, n, |i, j| if i == j { i as f64 - 1.0 } else { -0.1 });
        let s = Mat::<f64>::identity(n, n);
        let x = Mat::<f64>::identity(n, n);
        let inp = ScfInput {
            h: h.as_ref(),
            s: s.as_ref(),
            e_nuc: 0.0,
            nalpha: 1,
            nbeta: 1,
            exchange: Some(linus_ints::operator::Operator::Coulomb),
            guess_density: None,
            geometry: None,
        };
        let algo = ScfAlgorithm::default();
        let mut st = OrbitalState::new(false, [1, 1], n, Mat::zeros(n, n), Mat::zeros(n, n));
        st.set_orbitals(orbitals(n, 0.1), orbitals(n, -0.2), Par::Seq);
        let rotated = [orbitals(n, 0.7), orbitals(n, 0.4)];
        let (mut jk, mut env) = (EchoJk::default(), Remembers::default());
        let mut timers = Timers::default();
        {
            let terms = FockTerms::new(&mut jk, None, Some(&mut env));
            let mut fock = FockEvaluator::new(terms, &inp, x.as_ref(), false, &algo, Par::Seq);
            let ev = fock.evaluate(&st, false, &mut timers).unwrap();
            let (f, e) = fock.trial(&rotated, &[1, 1]).unwrap();
            assert!(e.is_finite() && f.len() == 2 && e != ev.energy);
            assert!(fock.last_dens.is_some() && fock.incremental_builds == 0);
            assert!(
                fock.settle(&st).unwrap(),
                "a term keeps state: one build at the accepted state"
            );
        }
        // the accepted build, the trial, the settling build; the incremental state reset once
        assert_eq!((jk.builds, jk.resets), (3, 1));
        let d = &st.da + &st.db;
        assert_eq!(env.last, (0..n).map(|i| d[(i, i)]).sum::<f64>());
        // nothing keeps state: no build
        let mut jk = EchoJk::default();
        {
            let terms = FockTerms::new(&mut jk, None, None);
            let mut fock = FockEvaluator::new(terms, &inp, x.as_ref(), false, &algo, Par::Seq);
            fock.trial(&rotated, &[1, 1]).unwrap();
            assert!(!fock.settle(&st).unwrap());
        }
        assert_eq!((jk.builds, jk.resets), (1, 1));
    }
}
