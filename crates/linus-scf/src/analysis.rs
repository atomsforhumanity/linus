//! The analyses of converged points: the host loop's stability analysis of a converged UHF/UKS
//! point and its step along a negative mode (`analyse`; for a Hessian SCF its search at the
//! accepted point is the zero-mode check's record, [`point_check`]), and, for both loops, the
//! zero-mode check where no search ran at the accepted point ([`mode_check`]) and the closed-shell
//! probe of RHF/RKS points (D36, [`closed_shell_probe`]). The Davidson searches are `super::stability`'s; they run on the SCF's
//! own Fock builds through [`RotatedGradient`]: the host loop's trial evaluations
//! (`HostRotations`) or the device loop's rotations (`linus::backend::gpu::scf_control`).

use super::accel::rotate;
use super::control::Analysis;
use super::fock::FockEvaluator;
use super::stability::{
    closed_shell_mode, lowest_hessian_mode, orbital_gradient, rotated_gradient, rotation_towards,
    unflatten, ClosedShellProbe, RotatedGradient, Search, StabilityCheck, Starts, PROBE_NEGATIVE,
};
use super::state::OrbitalState;
use super::{diagonalize, ModeCheck, ScfConfig};
use faer::Mat;
use linus_base::error::LinusError;
use linus_base::progress::Progress;
use linus_base::symmetry::ao::AoGroup;
use linus_core::timing::Timers;
use std::time::Instant;

/// The host loop's rotations of the reference orbitals `c` (per spin, `nocc` occupied): trial
/// evaluations of the Fock evaluator; a failed build gives NaN and keeps its error. Settling puts
/// back the accepted state `st`.
pub(crate) struct HostRotations<'f, 't, 'm> {
    fock: &'f mut FockEvaluator<'t, 'm>,
    st: &'f OrbitalState,
    c: Vec<Mat<f64>>,
    nocc: [usize; 2],
    failed: Option<LinusError>,
}

impl<'f, 't, 'm> HostRotations<'f, 't, 'm> {
    pub fn new(
        fock: &'f mut FockEvaluator<'t, 'm>,
        st: &'f OrbitalState,
        c: Vec<Mat<f64>>,
        nocc: [usize; 2],
    ) -> Self {
        HostRotations {
            fock,
            st,
            c,
            nocc,
            failed: None,
        }
    }

    /// The UHF/UKS Fock matrices and energy of the orbitals `c` (a trial evaluation; NaN after a
    /// failed build).
    fn evaluate(&mut self, c: &[Mat<f64>]) -> (Vec<Mat<f64>>, f64) {
        trial(self.fock, &mut self.failed, c, &self.nocc)
    }
}

/// A trial evaluation of `fock` at the orbitals `c` (`nocc` occupied), NaN after a failed build,
/// whose error goes to `failed` (the first one stays).
fn trial(
    fock: &mut FockEvaluator<'_, '_>,
    failed: &mut Option<LinusError>,
    c: &[Mat<f64>],
    nocc: &[usize],
) -> (Vec<Mat<f64>>, f64) {
    match fock.trial(c, nocc) {
        Ok(f) => f,
        Err(e) => {
            failed.get_or_insert(e);
            let n = c[0].nrows();
            let nan = Mat::from_fn(n, n, |_, _| f64::NAN);
            (vec![nan.clone(), nan], f64::NAN)
        }
    }
}

impl RotatedGradient for HostRotations<'_, '_, '_> {
    type Error = LinusError;

    fn nmo(&self) -> usize {
        self.c[0].ncols()
    }

    fn nocc(&self) -> [usize; 2] {
        self.nocc
    }

    fn orbital(&mut self, s: usize, p: usize) -> Vec<f64> {
        (0..self.c[s].nrows()).map(|r| self.c[s][(r, p)]).collect()
    }

    fn orbitals(&mut self, s: usize) -> Mat<f64> {
        self.c[s].clone()
    }

    fn gradient(&mut self, v: &[f64], h: f64) -> Vec<f64> {
        let HostRotations {
            fock,
            c,
            nocc,
            failed,
            ..
        } = self;
        let par = fock.par;
        let mut fock_at = |cr: &[Mat<f64>]| trial(fock, failed, cr, &nocc[..]);
        rotated_gradient(&mut fock_at, c, &nocc[..], v, h, par)
    }

    fn failure(&mut self) -> Option<LinusError> {
        self.failed.take()
    }

    fn settle(&mut self) -> Result<bool, LinusError> {
        self.fock.settle(self.st)
    }
}

/// What a stability analysis found.
pub(crate) enum Outcome {
    /// The probe found no negative mode: the converged point stands (the terms were put back at
    /// it, counted in the record's builds).
    ProbeMinimum,
    /// No step along the mode lowers the energy: a minimum.
    Minimum,
    /// The orbitals rotated along the negative mode, which lowered the energy.
    Step(Vec<Mat<f64>>),
}

/// What an analysis reads of the SCF.
pub(crate) struct AnalysisInput<'a> {
    /// The converged iteration and its energy.
    pub iter: usize,
    pub energy: f64,
    /// The converged Fock matrices.
    pub fa: &'a Mat<f64>,
    pub fb: &'a Mat<f64>,
    /// The occupied orbitals of the lowest-energy iterate (the saddle check's first direction).
    pub lowest: Option<&'a Vec<Mat<f64>>>,
    /// The molecule's point group on its basis (None: the searches ignore symmetry).
    pub group: Option<&'a AoGroup>,
    pub progress: &'a Progress,
}

/// The stability analysis `a` of a converged UHF/UKS point (`st`, the converged state): the lowest
/// orbital-Hessian mode by Davidson iterations at the canonical orbitals of the converged Fock
/// matrices (the probe's one-sided products first, then the full analysis' central differences if
/// it finds a negative mode), and along a negative mode the lowest energy of the rotations by 0.25,
/// 0.5 and 1, else the first lower one of 0.125 halved down to 1/64.
pub(crate) fn analyse(
    fock: &mut FockEvaluator<'_, '_>,
    st: &OrbitalState,
    a: &Analysis,
    inp: &AnalysisInput<'_>,
) -> Result<(Outcome, StabilityCheck), LinusError> {
    let (par, s) = (fock.par, fock.s);
    let (eps_ca, cca) = diagonalize(inp.fa.as_ref(), fock.x, par);
    let (eps_cb, ccb) = diagonalize(inp.fb.as_ref(), fock.x, par);
    let nocc = st.nocc;
    let cs = vec![cca, ccb];
    // (the lowest iterate is kept for the escalation too: the direction only with the saddle check)
    let guess = inp
        .lowest
        .filter(|_| a.towards_lowest)
        .map(|lo| rotation_towards(&cs, lo, &nocc, s, par));
    let e_here = inp.energy;
    let eps_c: [&[f64]; 2] = [&eps_ca, &eps_cb];
    // the probe's products are one-sided, from the converged point's gradient
    let g0 = a
        .probing
        .then(|| orbital_gradient(&cs, &[inp.fa, inp.fb], &nocc, par));
    let mut rot = HostRotations::new(fock, st, cs, nocc);
    // one search per irrep of the converged state's symmetry (the single lowest pair without)
    let starts = Starts::new(&mut rot, &eps_c, inp.group, s);
    let first = lowest_hessian_mode(
        &mut rot,
        &starts,
        Search {
            products: a.products,
            g0: g0.as_deref(),
            guess,
            guess_first: a.saddle_fired,
            irrep: None,
            stop_below: PROBE_NEGATIVE,
        },
    );
    let searched = first.searched;
    let (mut theta, mut rnorm, mut mode, mut irrep) =
        (first.theta, first.residual, first.vector, first.irrep);
    let (mut dav_it, mut nfb) = (first.iterations, first.builds);
    // the Ritz value bounds the lowest eigenvalue from above: a negative probe is a negative mode;
    // converge it with the full analysis from the probe's vector, in the probe's irrep
    let probe_only = a.probing && theta >= PROBE_NEGATIVE;
    if a.probing && !probe_only {
        let full = lowest_hessian_mode(
            &mut rot,
            &starts,
            Search {
                products: super::control::FULL_PRODUCTS,
                g0: None,
                guess: Some(mode),
                guess_first: true,
                irrep: Some(irrep),
                stop_below: PROBE_NEGATIVE,
            },
        );
        (theta, rnorm, mode, irrep) = (full.theta, full.residual, full.vector, full.irrep);
        dav_it += full.iterations;
        nfb += full.builds;
    }
    let mut chk = StabilityCheck {
        iter: inp.iter,
        e_above_min: a.e_above,
        eigenvalue: theta,
        residual: rnorm,
        davidson_iterations: dav_it,
        fock_builds: 0,
        step: 0.0,
        de: 0.0,
        probe: probe_only,
        irreps: starts.record(irrep, searched),
    };
    // a negative mode: the lowest energy of three steps along it, else a shorter one
    // the reference rotated by t along the mode
    let rotated = |c: &[Mat<f64>], t: f64| -> Vec<Mat<f64>> {
        let k = unflatten(&mode, c[0].ncols(), &nocc, t);
        c.iter()
            .zip(&k)
            .zip(nocc)
            .map(|((c, ks), no)| rotate(c, ks, no, par))
            .collect()
    };
    let mut best: Option<(f64, f64, Vec<Mat<f64>>)> = None;
    if theta < PROBE_NEGATIVE {
        for t in [0.25, 0.5, 1.0] {
            let cr = rotated(&rot.c, t);
            let (_, e_t) = rot.evaluate(&cr);
            nfb += 1;
            if e_t < best.as_ref().map_or(e_here, |b| b.1) {
                best = Some((t, e_t, cr));
            }
        }
        // none of them lowers the energy: the minimum along the mode lies closer than 0.25 (a
        // shallow well whose quartic term already wins there; HF+ r2SCAN: eigenvalue -1.7e-3,
        // minimum 1.1e-5 Eh down near 0.1). Halve the step down to 1/64 and take the first that
        // lowers it.
        if !best.as_ref().is_some_and(|b| b.1 < e_here - 1e-9) {
            let mut t = 0.125;
            while t > 0.01 {
                let cr = rotated(&rot.c, t);
                let (_, e_t) = rot.evaluate(&cr);
                nfb += 1;
                if e_t < e_here - 1e-9 {
                    best = Some((t, e_t, cr));
                    break;
                }
                t *= 0.5;
            }
        }
    }
    chk.fock_builds = nfb;
    if let Some(e) = rot.failure() {
        return Err(e);
    }
    let e_above = a.e_above;
    inp.progress.line(format_args!("          stability: lowest Hessian eigenvalue {theta:+.3e} (residual {rnorm:.1e}, {dav_it} Davidson iterations, {nfb} Fock builds), converged {e_above:.2e} Eh above the lowest energy visited"));
    if probe_only {
        // the probe found no negative mode: the SCF ends as it would have without it (the
        // converged Fock matrices, energy and density are untouched); its builds left the terms at
        // a rotated density, which one build at the converged state puts back
        if rot.settle()? {
            chk.fock_builds += 1;
        }
        return Ok((Outcome::ProbeMinimum, chk));
    }
    match best.filter(|b| b.1 < e_here - 1e-9) {
        Some((t, e_t, cr)) => {
            (chk.step, chk.de) = (t, e_t - e_here);
            inp.progress.line(format_args!(
                "          stability: step {t} along the mode, dE {:+.3e}; restarting",
                e_t - e_here
            ));
            Ok((Outcome::Step(cr), chk))
        }
        None => Ok((Outcome::Minimum, chk)),
    }
}

/// The zero-mode check's record of a Hessian SCF from the search at the point it accepted, `chk`
/// (the trigger's probe with the check's products, or the full analysis that found the point a
/// minimum), which took `seconds`.
pub fn point_check(chk: &StabilityCheck, seconds: f64) -> ModeCheck {
    ModeCheck {
        eigenvalue: chk.eigenvalue,
        residual: chk.residual,
        fock_builds: chk.fock_builds,
        seconds,
        irreps: chk.irreps,
    }
}

/// The zero-mode check of a converged UHF/UKS SCF where no search ran at the accepted point (the
/// analyses' budget spent, a plateau accepted), on either loop: `opts.mode_check` one-sided
/// Davidson products of `rot` per irrep of `starts` (at the final orbitals; `g0` their gradient)
/// for the lowest orbital-Hessian eigenvalue's estimate, the terms put back afterwards; timed from
/// `t0`. A record: the SCF has ended.
pub fn mode_check<R: RotatedGradient + ?Sized>(
    rot: &mut R,
    starts: &Starts,
    g0: &[f64],
    opts: &ScfConfig,
    t0: Instant,
    timers: &mut Timers,
) -> Result<ModeCheck, R::Error> {
    let m = lowest_hessian_mode(
        rot,
        starts,
        Search {
            products: opts.mode_check,
            g0: Some(g0),
            guess: None,
            guess_first: false,
            irrep: None,
            stop_below: super::MODE_CHECK_THRESHOLD,
        },
    );
    let (theta, rnorm, nfb) = (m.theta, m.residual, m.builds);
    if let Some(e) = rot.failure() {
        return Err(e);
    }
    // the check's builds left the terms at a rotated density: one build at the SCF's last state
    // puts back what they keep, counted in its builds
    let nfb = nfb + usize::from(rot.settle()?);
    let secs = t0.elapsed().as_secs_f64();
    timers.add("mode_check", secs);
    opts.progress.line(format_args!("          mode check: lowest orbital-Hessian eigenvalue <= {theta:+.3e} (residual {rnorm:.1e}, {nfb} Fock builds, {secs:.2} s)"));
    Ok(ModeCheck {
        eigenvalue: theta,
        residual: rnorm,
        fock_builds: nfb,
        seconds: secs,
        irreps: starts.record(m.irrep, m.searched),
    })
}

/// The closed-shell probe (D36) of the RHF/RKS record `rec` that passed its gate, on either loop:
/// one-sided products of the UHF/UKS orbital Hessian with `rot` (at the final orbitals, alpha =
/// beta; `g0` the gradient) from the external start of each irrep of `starts` (both spins'
/// energies the restricted ones), the terms put back afterwards; timed from `t0`. A record only.
pub fn closed_shell_probe<R: RotatedGradient + ?Sized>(
    rot: &mut R,
    rec: &mut ClosedShellProbe,
    starts: &Starts,
    g0: &[f64],
    opts: &ScfConfig,
    t0: Instant,
    timers: &mut Timers,
) -> Result<(), R::Error> {
    let m = closed_shell_mode(rot, starts, rec.products, g0);
    let (theta, rnorm, mode, nfb) = (m.theta, m.residual, m.vector, m.builds);
    rec.irreps = Some(starts.record(m.irrep, m.searched));
    if let Some(e) = rot.failure() {
        return Err(e);
    }
    // the probe's builds left the terms at a rotated density: one build at the SCF's last state
    // (restricted) puts back what they keep, counted
    let nfb = nfb + usize::from(rot.settle()?);
    let secs = t0.elapsed().as_secs_f64();
    timers.add("closed_shell_probe", secs);
    rec.ran(theta, rnorm, &mode, nfb, secs);
    opts.progress.line(format_args!(
        "          closed-shell probe: HOMO-LUMO gap {:.3e}, lowest UHF/UKS orbital-Hessian eigenvalue <= {theta:+.3e} ({:?}, residual {rnorm:.1e}, {nfb} Fock builds, {secs:.2} s)",
        rec.gap.unwrap_or(f64::NAN),
        rec.kind
    ));
    Ok(())
}

/// The host loop's rotations at the final orbitals `c` (per spin, `nocc` occupied) and their
/// gradient for the converged Fock matrices `f`: the mode check's and the closed-shell probe's.
pub(crate) fn host_rotations<'f, 't, 'm>(
    fock: &'f mut FockEvaluator<'t, 'm>,
    st: &'f OrbitalState,
    c: Vec<Mat<f64>>,
    f: [&Mat<f64>; 2],
    nocc: [usize; 2],
) -> (HostRotations<'f, 't, 'm>, Vec<f64>) {
    let g0 = orbital_gradient(&c, &f, &nocc, fock.par);
    (HostRotations::new(fock, st, c, nocc), g0)
}
