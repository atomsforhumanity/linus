//! Stability analysis of converged UHF/UKS determinants: the lowest orbital-Hessian mode by
//! Davidson iterations (the response's eigensolver, [`crate::response::solver`]) on finite
//! differences of Fock builds ([`lowest_hessian_mode`]), and the rotations along it. The
//! closed-shell probe of RHF/RKS determinants (`closed_shell_mode`, [`ClosedShellProbe`], decision
//! D36) runs the same iterations on the UHF/UKS orbital Hessian at the restricted point.
//!
//! Every search runs per irreducible representation of the converged state's point group
//! ([`symmetry`]): one Davidson per irrep from its lowest-gap pair, its products projected onto
//! the irrep ([`Starts`]), so a negative mode of any irrep with a low pair is found whatever the
//! rounding does. Without symmetry it is the single search from the lowest pair. A search of
//! [`WIDE_START_PRODUCTS`] or more products starts from the sum of the [`WIDE_PAIRS`] lowest pairs
//! instead (per irrep, or overall without symmetry).

pub mod symmetry;

use super::accel::rotate;
use crate::response::solver::{lowest_eigenpair, DavidsonSetup, Start};
use faer::{Mat, MatRef, Par};
use linus_base::linalg::mm;
use linus_base::symmetry::ao::AoGroup;
use serde::Serialize;
pub use symmetry::{Geometry, IrrepStart, PairSymmetry};

/// One stability analysis of a converged UHF/UKS determinant (a saddle-point check).
#[derive(Clone, Debug, Serialize)]
pub struct StabilityCheck {
    /// SCF iteration at which the converged point was analysed.
    pub iter: usize,
    /// Converged energy minus the lowest energy the SCF had visited (the trigger; Eh).
    pub e_above_min: f64,
    /// Lowest eigenvalue of the orbital Hessian d g_ai / d kappa_bj (g_ai = F_ai, half the
    /// energy Hessian), its residual norm and the Davidson iterations.
    pub eigenvalue: f64,
    pub residual: f64,
    pub davidson_iterations: usize,
    /// Fock builds spent (two per Hessian-vector product, one-sided: one; one per trial step;
    /// and the probe's build that restores the converged state where a term keeps state, a
    /// solvent or VV10).
    pub fock_builds: usize,
    /// Norm of the rotation taken along the mode (0: none, the point is a minimum) and the
    /// energy change it gave.
    pub step: f64,
    pub de: f64,
    /// Only the cheap probe ran (`scf.stability_probe` Davidson steps, no negative mode found):
    /// the eigenvalue is an upper bound of the lowest one.
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub probe: bool,
    /// The searches' symmetry: the converged state's abelian point group (C1 without symmetry),
    /// the irrep of the eigenvalue's search (its Mulliken symbol and its index: the bits of the
    /// generators under which it is antisymmetric) and the irreps searched.
    #[serde(flatten)]
    pub irreps: IrrepRecord,
}

/// Which irreps a search covered ([`StabilityCheck`], [`ClosedShellProbe`], the mode check).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct IrrepRecord {
    /// The converged state's abelian point group (C1 without symmetry).
    pub symmetry: &'static str,
    /// The irrep of the reported Ritz value's search and its index.
    pub irrep: &'static str,
    pub irrep_index: usize,
    /// Irreps searched.
    pub irreps_searched: usize,
}

/// The orbitals a stability search rotates and the orbital gradient of their rotations, from the
/// SCF's own Fock builds: the host loop's trial evaluations (`super::analysis::HostRotations`) or
/// the device loop's rotations (`linus::backend::gpu::scf_control`). The searches
/// ([`lowest_hessian_mode`], `closed_shell_mode`) take the reference orbitals' phases (D29) and the
/// gradients from it, so both loops start and step alike.
pub trait RotatedGradient {
    /// The error of a failed Fock build.
    type Error;
    /// Orbitals of the reference (per spin).
    fn nmo(&self) -> usize;
    /// Occupied orbitals of the reference per spin (a closed shell's probe: both its own).
    fn nocc(&self) -> [usize; 2];
    /// The coefficients of the reference orbital `p` of spin `s`.
    fn orbital(&mut self, s: usize, p: usize) -> Vec<f64>;
    /// The reference orbitals of spin `s` (nbf x nmo): the symmetry of the searches' starts.
    fn orbitals(&mut self, s: usize) -> Mat<f64> {
        let cols: Vec<Vec<f64>> = (0..self.nmo()).map(|p| self.orbital(s, p)).collect();
        let n = cols.first().map_or(0, |c| c.len());
        Mat::from_fn(n, cols.len(), |i, j| cols[j][i])
    }
    /// g(h v), the orbital gradient g_s = C_s(k)_v^T F_s[D(k)] C_s(k)_o of the reference rotated
    /// by h v (v flattened per spin as (a, i) blocks), flattened as v; NaN from a failed build on
    /// (the search then ends without a mode).
    fn gradient(&mut self, v: &[f64], h: f64) -> Vec<f64>;
    /// The first failed build's error.
    fn failure(&mut self) -> Option<Self::Error>;
    /// After the rotated builds: puts back what the Fock terms keep of the converged state (one
    /// build at it when a term keeps state, a solvent or VV10); whether it built.
    fn settle(&mut self) -> Result<bool, Self::Error>;
}

/// The phase sign of the occupied-virtual pair (spin `s`, occupied `i`, virtual `a` counted from
/// the first virtual) of `rot`'s reference (decision D29): the product of its two orbitals'
/// [`phase`]s.
fn pair_sign<R: RotatedGradient + ?Sized>(rot: &mut R, (s, i, a): (usize, usize, usize)) -> f64 {
    let no = rot.nocc()[s];
    let (ci, ca) = (rot.orbital(s, i), rot.orbital(s, no + a));
    phase(ci.iter().copied()) * phase(ca.iter().copied())
}

/// The pairs a wide start sums: an irrep's (or, without symmetry, the whole space's) lowest-gap
/// pairs. Three: the orbital-energy gap can misrank a d-shell pair's curvature by 0.2 Eh (FeH's
/// -1.1e-3 mode grows from the third pair of its irrep; TiO's likewise). On the robustness sweep's
/// 542 open shells the full analysis from three misses 2 lower states instead of 5 and loses none,
/// and the zero-mode check of 377 open-shell Hessians reaches every lowest state (numerics epoch
/// 8, docs/internal/robustness.md).
pub const WIDE_PAIRS: usize = 3;

/// Searches of this many products or more (the full analysis, the zero-mode check of a Hessian
/// SCF) start wide ([`WIDE_PAIRS`]) and end early on a negative Ritz value only once its residual
/// is below a tenth of it (a wide start's first negative Ritz value can be a blend of modes);
/// fewer (the two-product probe, the closed-shell probe) start from the lowest pair alone, whose
/// two products a sum would dilute (an RC21 radical: -4.4e-3 from its lowest pair, +3.0e-3 from
/// the sum of three).
pub const WIDE_START_PRODUCTS: usize = 6;

/// Below this fraction of a guess's weight in an irrep, the irrep's search starts without it.
const GUESS_WEIGHT: f64 = 1e-2;

/// The curvature (Eh) an irrep's search gives the rotations outside the irrep: far above anything
/// the searches decide on (a negative mode, the zero-mode check's 1e-3), so none of them is ever
/// the lowest. Without it, at a structure symmetric only within the detection's tolerance (GMTKN55's
/// transition states, ReCl6 2-), a correction's small part outside the irrep gave a Ritz value near
/// zero (+1e-7) where the irrep's lowest eigenvalue is +0.11.
const OUTSIDE_IRREP: f64 = 1.0;

/// The irreps the searches at a converged point cover: those whose lowest pair's gap lies within
/// this of the lowest gap (Eh). Measured on the robustness sets (B3LYP/def2-SVP RI-JK, extreme
/// preset): 0.1 catches what 0.3 and every irrep catch (triplet O3's B2 and A'' saddles, the
/// closed set's three broken-symmetry instabilities; the open set alike) at a third less extra
/// cost (open set: Fock builds against the single start median 1.00x, p90 1.32x, against 1.06x and
/// 1.38x at 0.3); the narrower rule of searching only irreps whose gap lies below the running Ritz
/// value misses all of those.
pub const IRREP_WINDOW: f64 = 0.1;

/// Where the searches at a converged point start ([`symmetry`]): one start per irrep of the
/// state's abelian point group that has a pair, in the order of their gaps; without symmetry (no
/// geometry, or a C1 state) none, and the searches start from the single lowest pair. With the
/// orbital energies (per spin), the searches' preconditioner.
pub struct Starts {
    eps: Vec<Vec<f64>>,
    sym: Option<PairSymmetry>,
    irreps: Vec<IrrepStart>,
    /// The leading irreps within [`IRREP_WINDOW`] of the lowest gap.
    gated: usize,
}

impl Starts {
    /// No symmetry: the single lowest pair of the orbitals with energies `eps` (per spin).
    pub fn without_symmetry(eps: &[&[f64]]) -> Starts {
        Starts {
            eps: eps.iter().map(|e| e.to_vec()).collect(),
            sym: None,
            irreps: Vec::new(),
            gated: 0,
        }
    }

    /// The starts at the reference orbitals of `rot` (energies `eps`, overlap `s`) for the
    /// molecule's group `group` (None: no symmetry).
    pub fn new<R: RotatedGradient + ?Sized>(
        rot: &mut R,
        eps: &[&[f64]],
        group: Option<&AoGroup>,
        s: MatRef<'_, f64>,
    ) -> Starts {
        let Some(g) = group.filter(|g| !g.group.trivial()) else {
            return Starts::without_symmetry(eps);
        };
        let nocc = rot.nocc();
        let c: Vec<Mat<f64>> = (0..nocc.len()).map(|sp| rot.orbitals(sp)).collect();
        let cr: Vec<MatRef<'_, f64>> = c.iter().map(|m| m.as_ref()).collect();
        let sym = PairSymmetry::at(g, &cr, &nocc, s);
        if sym.group().trivial() {
            return Starts::without_symmetry(eps);
        }
        let irreps = sym.starts(&cr, eps);
        let lowest = irreps.first().map_or(0.0, |s| s.gap);
        let gated = irreps
            .iter()
            .take_while(|s| s.gap <= lowest + IRREP_WINDOW)
            .count();
        Starts {
            eps: eps.iter().map(|e| e.to_vec()).collect(),
            sym: Some(sym),
            irreps,
            gated,
        }
    }

    /// The record of a search over these starts that reported irrep `b` after `searched`.
    pub fn record(&self, b: usize, searched: usize) -> IrrepRecord {
        let (symmetry, irrep) = match &self.sym {
            Some(sym) => (sym.group().name, sym.group().irrep_name(b)),
            None => ("C1", "A"),
        };
        IrrepRecord {
            symmetry,
            irrep,
            irrep_index: b,
            irreps_searched: searched,
        }
    }
}

/// What a search is asked for ([`lowest_hessian_mode`]).
pub struct Search<'g> {
    /// Davidson products per irrep.
    pub products: usize,
    /// The gradient at the reference: one-sided products from it (else central).
    pub g0: Option<&'g [f64]>,
    /// A first trial direction (the saddle check's towards the lowest iterate, a probe's mode),
    /// projected onto each irrep searched where it has weight.
    pub guess: Option<Vec<f64>>,
    /// The guess is the path's direction (the saddle check fired): the irreps it has weight in are
    /// searched first, most first, also beyond the window; else the irreps go by gap.
    pub guess_first: bool,
    /// Only this irrep (an analysis continuing a probe's search).
    pub irrep: Option<usize>,
    /// Stop at the first irrep whose Ritz value lies below this.
    pub stop_below: f64,
}

/// What a search found: the lowest Ritz value of the irreps searched (or the first below
/// [`Search::stop_below`]), its residual norm and vector (flattened per spin as (a, i) blocks), the
/// Davidson iterations and Fock builds of all of them, the irrep of the reported value and the
/// irreps searched.
#[derive(Clone, Debug)]
pub struct Mode {
    pub theta: f64,
    pub residual: f64,
    pub vector: Vec<f64>,
    pub iterations: usize,
    pub builds: usize,
    pub irrep: usize,
    pub searched: usize,
}

/// Lowest eigenpair of the orbital Hessian of an unrestricted determinant at the canonical
/// orbitals of `rot` (per spin; `eps` their energies), by
/// Davidson iterations with the preconditioner eps_a - eps_i. Hessian-vector products are
/// central differences of the orbital gradient in the rotated frame,
///   H v = [g(h v) - g(-h v)] / 2h,  g_s(k) = C_s(k)_v^T F_s[D(k)] C_s(k)_o,  C_s(k) = C_s exp(K_s),
/// whose derivative at k = 0 is F_vv v - v F_oo + C_v^T F^(1)\[v\] C_o (the Hessian including
/// every response term the Fock build has: J/K of any builder, XC, solvent). With `g0` (the
/// gradient at k = 0, from the converged Fock matrices) the products are one-sided,
/// H v = [g(h v) - g0] / h: one Fock build each instead of two, an O(h) error that a sign test (the
/// cheap probe) tolerates.
/// With symmetry (`starts`) one Davidson per irrep (`search.products` products each, from the
/// irrep's start and the guess projected onto the irrep where it has weight there; by gap, or the
/// irreps the guess has weight in first, [`Search::guess_first`]), every gradient projected onto
/// the irrep.
pub fn lowest_hessian_mode<R: RotatedGradient + ?Sized>(
    rot: &mut R,
    starts: &Starts,
    search: Search<'_>,
) -> Mode {
    search_irreps(rot, starts, search, false)
}

/// The searches of [`lowest_hessian_mode`] and the closed-shell probe (`external`: each start is
/// the external rotation of its alpha part, alpha and beta in opposite senses).
fn search_irreps<R: RotatedGradient + ?Sized>(
    rot: &mut R,
    starts: &Starts,
    search: Search<'_>,
    external: bool,
) -> Mode {
    let (nmo, nocc) = (rot.nmo(), rot.nocc());
    let eps: Vec<&[f64]> = starts.eps.iter().map(|e| e.as_slice()).collect();
    let eps = &eps[..];
    let Some(sym) = &starts.sym else {
        // no symmetry: the single lowest pair, D29-signed for one-sided products
        let start = match (search.g0, start_pair(eps, &nocc, nmo)) {
            (Some(_), Some(pair)) if external => {
                let (_, i, a) = pair;
                Some(external_start(nmo, nocc[0], i, a, pair_sign(rot, pair)))
            }
            _ => None,
        };
        let sign = match (search.g0, start_pair(eps, &nocc, nmo)) {
            (Some(_), Some(pair)) if !external => pair_sign(rot, pair),
            _ => 1.0,
        };
        // a search of many products: the wide start, the lowest pairs D29-signed and summed
        let start = match start {
            None if !external && search.products >= WIDE_START_PRODUCTS => {
                wide_start(rot, eps, &nocc, nmo, search.g0.is_some())
            }
            s => s,
        };
        let mut grad = |v: &[f64], h: f64| rot.gradient(v, h);
        let (theta, residual, vector, iterations, builds) = davidson(
            &mut grad,
            nmo,
            eps,
            &nocc,
            search.guess,
            search.products,
            search.g0,
            sign,
            start,
        );
        return Mode {
            theta,
            residual,
            vector,
            iterations,
            builds,
            irrep: 0,
            searched: 1,
        };
    };
    // the guess's part in each irrep where it has weight
    let norm2 = |v: &[f64]| v.iter().map(|x| x * x).sum::<f64>();
    let mut weighted: Vec<(f64, usize, Vec<f64>)> = Vec::new();
    if let Some(g) = &search.guess {
        let total = norm2(g);
        for st in &starts.irreps {
            if search.irrep.is_some_and(|b| b != st.irrep) {
                continue;
            }
            let mut gb = g.clone();
            sym.project(st.irrep, &mut gb);
            let w = norm2(&gb);
            if w >= GUESS_WEIGHT * total {
                weighted.push((w, st.irrep, gb));
            }
        }
        weighted.sort_by(|x, y| y.0.total_cmp(&x.0));
    }
    // the irreps in order: by gap within the window (each with its part of the guess), or with a
    // path's guess the irreps it has weight in first, most first
    let mut order: Vec<(usize, Option<Vec<f64>>)> = Vec::new();
    if search.guess_first {
        order.extend(weighted.iter().map(|(_, b, gb)| (*b, Some(gb.clone()))));
    }
    for st in starts.irreps.iter().take(starts.gated) {
        if search.irrep.is_none_or(|b| b == st.irrep) && order.iter().all(|o| o.0 != st.irrep) {
            let gb = weighted
                .iter()
                .find(|w| w.1 == st.irrep)
                .map(|w| w.2.clone());
            order.push((st.irrep, gb));
        }
    }
    if let Some(b) = search.irrep {
        if order.is_empty() {
            order.push((b, None));
        }
    }
    let mut best: Option<Mode> = None;
    let (mut iterations, mut builds, mut searched) = (0, 0, 0);
    for (b, guess) in order {
        let Some(st) = starts.irreps.iter().find(|s| s.irrep == b) else {
            continue;
        };
        let mut start = if !external && search.products >= WIDE_START_PRODUCTS {
            st.wide.clone()
        } else {
            st.vector.clone()
        };
        if external {
            let half = start.len() / 2;
            for k in 0..half {
                start[half + k] = -start[k];
            }
        }
        let g0 = search.g0.map(|g| {
            let mut g = g.to_vec();
            sym.project(b, &mut g);
            g
        });
        // the operator P H P + c (1 - P): the irrep's block, and everything outside it at the
        // curvature c (a symmetry that holds only to the geometry's tolerance projects only
        // nearly, and a direction outside the irrep would otherwise see P H z ~ 0: a false zero)
        let mut grad = |v: &[f64], h: f64| {
            let mut pv = v.to_vec();
            sym.project(b, &mut pv);
            let mut g = rot.gradient(&pv, h);
            sym.project(b, &mut g);
            for ((gi, vi), pi) in g.iter_mut().zip(v).zip(&pv) {
                *gi += OUTSIDE_IRREP * h * (vi - pi);
            }
            g
        };
        let (theta, residual, vector, it, nfb) = davidson(
            &mut grad,
            nmo,
            eps,
            &nocc,
            guess,
            search.products,
            g0.as_deref(),
            1.0,
            Some(start),
        );
        iterations += it;
        builds += nfb;
        searched += 1;
        let stop = theta.is_nan() || theta < search.stop_below;
        if best
            .as_ref()
            .is_none_or(|m| theta.is_nan() || theta < m.theta)
        {
            best = Some(Mode {
                theta,
                residual,
                vector,
                iterations: 0,
                builds: 0,
                irrep: b,
                searched: 0,
            });
        }
        if stop {
            break;
        }
    }
    let mut m = best.unwrap_or(Mode {
        theta: f64::NAN,
        residual: f64::NAN,
        vector: Vec::new(),
        iterations: 0,
        builds: 0,
        irrep: 0,
        searched: 0,
    });
    (m.iterations, m.builds, m.searched) = (iterations, builds, searched);
    m
}

/// The orbital gradient g_s(k) = C_s(k)_v^T F_s[D(k)] C_s(k)_o of the orbitals `c` (per spin,
/// `nocc` occupied) rotated by h v (v flattened per spin as (a, i) blocks), C_s(k) = C_s exp(K_s),
/// with the Fock matrices of `fock_at`: the function whose finite differences are
/// [`lowest_hessian_mode`]'s Hessian-vector products on the host loop. Flattened as v.
#[allow(clippy::type_complexity)]
pub(super) fn rotated_gradient(
    fock_at: &mut dyn FnMut(&[Mat<f64>]) -> (Vec<Mat<f64>>, f64),
    c: &[Mat<f64>],
    nocc: &[usize],
    v: &[f64],
    h: f64,
    par: Par,
) -> Vec<f64> {
    let nmo = c[0].ncols();
    let n: usize = nocc.iter().map(|&no| (nmo - no) * no).sum();
    let k = unflatten(v, nmo, nocc, h);
    let cr: Vec<Mat<f64>> = c
        .iter()
        .zip(&k)
        .zip(nocc)
        .map(|((cs, ks), &no)| rotate(cs, ks, no, par))
        .collect();
    let (f, _) = fock_at(&cr);
    let mut g = Vec::with_capacity(n);
    for ((cs, fs), &no) in cr.iter().zip(&f).zip(nocc) {
        let gs = mm(
            cs.subcols(no, nmo - no).transpose(),
            mm(fs.as_ref(), cs.subcols(0, no), par).as_ref(),
            par,
        );
        for i in 0..no {
            for a in 0..nmo - no {
                g.push(gs[(a, i)]);
            }
        }
    }
    g
}

/// Below this Ritz value (Eh) a stability probe has found a negative orbital-Hessian mode (the
/// Ritz value bounds the lowest eigenvalue from above; above it, the one-sided products' O(h)
/// error could make a zero mode look negative).
pub const PROBE_NEGATIVE: f64 = -1e-4;

/// The direction of a closed-shell probe's mode ([`ClosedShellProbe::kind`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ProbeKind {
    /// Alpha and beta orbitals rotate in opposite senses (triplet-like): towards a
    /// broken-symmetry UHF/UKS solution.
    External,
    /// Alpha and beta rotate in the same sense (singlet-like): towards a lower RHF/RKS solution.
    Internal,
}

impl ProbeKind {
    /// The kind of a UHF/UKS rotation `x` at a closed-shell point (flattened as the Davidson
    /// vectors: the alpha block, then the beta block of the same length): external when the two
    /// blocks point in opposite senses (a negative overlap), else internal.
    pub(super) fn of(x: &[f64]) -> ProbeKind {
        let half = x.len() / 2;
        let ab: f64 = x[..half].iter().zip(&x[half..]).map(|(a, b)| a * b).sum();
        if ab < 0.0 {
            ProbeKind::External
        } else {
            ProbeKind::Internal
        }
    }
}

/// Why the closed-shell probe did not run ([`ClosedShellProbe::skipped`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ProbeSkip {
    /// The HOMO-LUMO gap is at or above `scf.stability_closed_gap`.
    GapAboveThreshold,
    /// `scf.stability_closed_probe` is 0.
    Off,
    /// Hartree-Fock with `scf.stability_closed_probe` unset (opt-in for HF).
    HfDefault,
    /// `scf.stability_closed_probe` unset at a later geometry of the session (it runs at the first).
    LaterGeometry,
    /// No occupied or no virtual orbitals: nothing to rotate.
    NoVirtualOrbitals,
    /// The SCF did not converge.
    NotConverged,
}

/// The closed-shell stability probe of an RHF/RKS solution (`scf.stability_closed_probe`, decision
/// D36): at a converged point whose HOMO-LUMO gap lies below the threshold, a few one-sided
/// products of the UHF/UKS orbital Hessian from the external start (`external_start`). Warning
/// only: nothing of the SCF result depends on it. Without `eigenvalue`, `skipped` says why it
/// did not run.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct ClosedShellProbe {
    /// Hessian-vector products allowed (`scf.stability_closed_probe` resolved; 0: off).
    pub products: usize,
    /// HOMO-LUMO gap of the converged orbitals (Eh); None without occupied or virtual orbitals.
    pub gap: Option<f64>,
    /// The gate, `scf.stability_closed_gap` (Eh).
    pub threshold: f64,
    /// Why it did not run.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub skipped: Option<ProbeSkip>,
    /// The Ritz value (Eh; linus's orbital-Hessian convention, d g_ai / d kappa_bj): an upper
    /// bound of the lowest UHF/UKS orbital-Hessian eigenvalue within the start's symmetry block.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub eigenvalue: Option<f64>,
    /// Residual norm of the Ritz vector.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub residual: Option<f64>,
    /// The direction of the Ritz vector.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub kind: Option<ProbeKind>,
    /// The eigenvalue lies below [`PROBE_NEGATIVE`]: a lower UHF/UKS solution exists (external),
    /// or a lower RHF/RKS one (internal).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub unstable: Option<bool>,
    /// Fock builds: one UKS-type build per product, and where a term keeps state (a solvent,
    /// VV10) the build that restores the SCF's last state afterwards.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fock_builds: Option<usize>,
    /// Wall time (s), with that restoring build.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seconds: Option<f64>,
    /// It ran at a continued SCF because the gap trigger fired ([`ProbeTrigger`]).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub trigger: Option<TriggerReason>,
    /// The irreps it searched and the irrep of its Ritz value.
    #[serde(flatten, skip_serializing_if = "Option::is_none")]
    pub irreps: Option<IrrepRecord>,
}

impl ClosedShellProbe {
    /// The record before the probe: the gap of the orbital energies `eps` with `nocc` occupied,
    /// the gate `threshold`, the `products` allowed, and why it will not run (None: it runs).
    pub(super) fn gate(
        converged: bool,
        eps: &[f64],
        nocc: usize,
        threshold: f64,
        products: usize,
    ) -> ClosedShellProbe {
        let gap = (nocc > 0 && nocc < eps.len())
            .then(|| eps[nocc] - eps[nocc - 1])
            .filter(|g| g.is_finite());
        let skipped = if !converged {
            Some(ProbeSkip::NotConverged)
        } else if products == 0 {
            Some(ProbeSkip::Off)
        } else if gap.is_none() {
            Some(ProbeSkip::NoVirtualOrbitals)
        } else if gap.is_some_and(|g| g >= threshold) {
            Some(ProbeSkip::GapAboveThreshold)
        } else {
            None
        };
        ClosedShellProbe {
            products,
            gap,
            threshold,
            skipped,
            eigenvalue: None,
            residual: None,
            kind: None,
            unstable: None,
            fock_builds: None,
            seconds: None,
            trigger: None,
            irreps: None,
        }
    }

    /// `ClosedShellProbe::gate` with the gap trigger of a continued SCF: when the probe has no
    /// products and the trigger is armed for closed shells, it runs with the trigger's products
    /// if the gap fires it. Also returns the trigger's decision (None: not armed, or the SCF did
    /// not converge).
    pub fn gate_triggered(
        converged: bool,
        eps: &[f64],
        nocc: usize,
        threshold: f64,
        products: usize,
        trigger: Option<&ProbeTrigger>,
    ) -> (ClosedShellProbe, Option<TriggerRecord>) {
        let p = Self::gate(converged, eps, nocc, threshold, products);
        let Some(t) = trigger.filter(|t| t.closed_products > 0 && products == 0 && converged)
        else {
            return (p, None);
        };
        let rec = t.decide(p.gap);
        if !rec.fired() {
            return (p, Some(rec));
        }
        let mut p = Self::gate(converged, eps, nocc, threshold, t.closed_products);
        p.trigger = rec.trigger;
        (p, Some(rec))
    }

    /// It passed the gate.
    pub fn runs(&self) -> bool {
        self.skipped.is_none()
    }

    /// The record of the probe that ran: the Ritz value `theta`, its residual and vector `x`,
    /// the products' Fock builds and the wall time.
    pub(super) fn ran(&mut self, theta: f64, residual: f64, x: &[f64], builds: usize, secs: f64) {
        self.eigenvalue = Some(theta);
        self.residual = Some(residual);
        self.kind = (!x.is_empty()).then(|| ProbeKind::of(x));
        self.unstable = Some(theta < PROBE_NEGATIVE);
        self.fock_builds = Some(builds);
        self.seconds = Some(secs);
    }

    /// The probe found a negative mode.
    pub fn is_unstable(&self) -> bool {
        self.unstable == Some(true)
    }
}

/// The gap at which the default probes' gates look (Eh): an RHF/RKS solution's HOMO-LUMO gap,
/// an UHF/UKS one's smaller alpha or beta HOMO-LUMO gap, from the orbital energies `eps_a`,
/// `eps_b` with `na`, `nb` occupied; None without an occupied and a virtual orbital in a spin.
pub fn gate_gap(
    eps_a: &[f64],
    eps_b: &[f64],
    na: usize,
    nb: usize,
    restricted: bool,
) -> Option<f64> {
    let gap = |e: &[f64], n: usize| {
        (n > 0 && n < e.len())
            .then(|| e[n] - e[n - 1])
            .filter(|g| g.is_finite())
    };
    if restricted {
        return gap(eps_a, na);
    }
    match (gap(eps_a, na), gap(eps_b, nb)) {
        (Some(a), Some(b)) => Some(a.min(b)),
        (a, b) => a.or(b),
    }
}

/// How much the gap must fall below the one at the last probe, while below the gate, for the gap
/// trigger to fire again (Eh; [`ProbeTrigger`]).
pub const GAP_DROP: f64 = 0.02;

/// The gap trigger of the default probes at a continued SCF (`ScfConfig::probe_trigger`): a later
/// geometry of the session, or an SCF it was told continues an earlier geometry of the same
/// task (a qcopt optimization, IRC or finite-difference step), where the unset probes do not run.
/// They run there all the same when the gap ([`gate_gap`]) lies below `threshold` and either
/// was not below it at the task's last probe or has fallen more than [`GAP_DROP`] below that
/// probe's gap: a transition-state search or an IRC that reaches a stretched bond or a diradical
/// is probed.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ProbeTrigger {
    /// The gap at the task's last probe (Eh; None: unknown, any gap below the gate fires).
    pub last_gap: Option<f64>,
    /// The gate (`scf.stability_closed_gap`, Eh).
    pub threshold: f64,
    /// Products of the open-shell stability probe when it fires (0: not armed for open shells;
    /// `scf.stability_probe` set explicitly applies as it is).
    pub stability_products: usize,
    /// Products of the closed-shell probe when it fires (0: not armed for closed shells:
    /// `scf.stability_closed_probe` set explicitly, or Hartree-Fock's opt-in default).
    pub closed_products: usize,
}

impl ProbeTrigger {
    /// Whether the gap `gap` fires the trigger.
    pub fn fires(&self, gap: Option<f64>) -> bool {
        let Some(g) = gap else {
            return false;
        };
        g < self.threshold
            && self
                .last_gap
                .is_none_or(|l| l >= self.threshold || g < l - GAP_DROP)
    }

    /// The record of the decision at the gap `gap`.
    pub(super) fn decide(&self, gap: Option<f64>) -> TriggerRecord {
        TriggerRecord {
            last_gap: self.last_gap,
            gap,
            trigger: self.fires(gap).then_some(TriggerReason::GapDropped),
        }
    }
}

/// Why the gap trigger ran the default probes ([`TriggerRecord::trigger`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum TriggerReason {
    /// The gap fell below the gate since the task's last probe, or more than [`GAP_DROP`] below
    /// that probe's gap.
    GapDropped,
}

/// The gap trigger's decision at a continued SCF ([`ProbeTrigger`]; `resolved.record.scf.probe_trigger`).
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct TriggerRecord {
    /// The gap at the task's last probe (Eh).
    pub last_gap: Option<f64>,
    /// The gap of this SCF's converged point (Eh).
    pub gap: Option<f64>,
    /// Set when the trigger fired: the default probes ran.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub trigger: Option<TriggerReason>,
}

impl TriggerRecord {
    /// The trigger fired.
    pub fn fired(&self) -> bool {
        self.trigger.is_some()
    }
}

/// The closed-shell probe's Davidson start (decision D36) for the pair of occupied orbital `i`
/// and virtual orbital `a` (counted from the first virtual; [`start_pair`] of the restricted
/// orbitals: the lowest eps_a - eps_i) of a closed shell with `nmo` orbitals and `nocc` occupied:
/// alpha and beta rotate in opposite senses, the external (triplet-like) direction towards a
/// broken-symmetry UHF/UKS solution, the alpha component `sign` (the product of the two
/// orbitals' [`phase`]s, D29) and the beta component -`sign`. Flattened as the UHF/UKS Davidson
/// vectors (alpha block, then beta).
pub(super) fn external_start(nmo: usize, nocc: usize, i: usize, a: usize, sign: f64) -> Vec<f64> {
    let nv = nmo - nocc;
    let mut v = vec![0.0; 2 * nv * nocc];
    v[i * nv + a] = sign;
    v[nv * nocc + i * nv + a] = -sign;
    v
}

/// The closed-shell probe at the converged RHF/RKS orbitals of `rot` (its reference: alpha = beta;
/// `starts` of both spins alike): `max_products` one-sided products of the UHF/UKS orbital
/// Hessian from [`external_start`] per irrep of `starts` (or of the lowest pair without
/// symmetry), with `g0` the gradient at the converged point (of the converged Fock matrix for both
/// spins).
pub(super) fn closed_shell_mode<R: RotatedGradient + ?Sized>(
    rot: &mut R,
    starts: &Starts,
    max_products: usize,
    g0: &[f64],
) -> Mode {
    search_irreps(
        rot,
        starts,
        Search {
            products: max_products,
            g0: Some(g0),
            guess: None,
            guess_first: false,
            irrep: None,
            stop_below: PROBE_NEGATIVE,
        },
        true,
    )
}

/// The pair of the Davidson iterations' first unit vector: the occupied-virtual rotation
/// (spin, occupied i, virtual a counted from the first virtual) with the lowest
/// eps_a - eps_i (the first of equal ones, in the vectors' order); None without one.
pub(super) fn start_pair(
    eps: &[&[f64]],
    nocc: &[usize],
    nmo: usize,
) -> Option<(usize, usize, usize)> {
    let mut best: Option<(f64, (usize, usize, usize))> = None;
    for (s, &no) in nocc.iter().enumerate() {
        for i in 0..no {
            for a in 0..nmo - no {
                let d = eps[s][no + a] - eps[s][i];
                if best.is_none_or(|(b, _)| d.total_cmp(&b).is_lt()) {
                    best = Some((d, (s, i, a)));
                }
            }
        }
    }
    best.map(|(_, p)| p)
}

/// The wide start of a search without symmetry: the [`WIDE_PAIRS`] lowest pairs of
/// [`start_pair`]'s order, each a unit vector signed by D29 (`signed`: one-sided products; central
/// ones do not depend on the sign), summed and normalized; None without a pair.
fn wide_start<R: RotatedGradient + ?Sized>(
    rot: &mut R,
    eps: &[&[f64]],
    nocc: &[usize],
    nmo: usize,
    signed: bool,
) -> Option<Vec<f64>> {
    let mut pairs: Vec<(f64, usize, (usize, usize, usize))> = Vec::new();
    let mut off = 0;
    for (s, &no) in nocc.iter().enumerate() {
        let nv = nmo - no;
        for i in 0..no {
            for a in 0..nv {
                pairs.push((eps[s][no + a] - eps[s][i], off + i * nv + a, (s, i, a)));
            }
        }
        off += nv * no;
    }
    if pairs.is_empty() {
        return None;
    }
    pairs.sort_by(|x, y| x.0.total_cmp(&y.0).then(x.1.cmp(&y.1)));
    let mut v = vec![0.0; off];
    for &(_, p, pair) in pairs.iter().take(WIDE_PAIRS) {
        v[p] = if signed { pair_sign(rot, pair) } else { 1.0 };
    }
    let nrm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    v.iter_mut().for_each(|x| *x /= nrm);
    Some(v)
}

/// The phase of an orbital's coefficients (decision D29): the sign of its largest-magnitude
/// component, the lowest index among those within 1e-8 (relative) of the largest. The
/// eigensolvers' own sign conventions (faer's, cuSOLVER's) are arbitrary and differ; this one
/// is a property of the orbital.
pub(super) fn phase(v: impl Iterator<Item = f64> + Clone) -> f64 {
    let m = v.clone().fold(0.0f64, |m, x| m.max(x.abs()));
    match v.into_iter().find(|x| x.abs() >= m * (1.0 - 1e-8)) {
        Some(x) if x < 0.0 => -1.0,
        _ => 1.0,
    }
}

/// The Davidson iterations of [`lowest_hessian_mode`] on the orbital gradient `grad(v, h)` of
/// the orbitals rotated by h v (v flattened per spin as (a, i) blocks, `nmo` orbitals, `nocc`
/// occupied per spin, `eps` their energies): the response's Davidson eigensolver
/// ([`crate::response::solver::lowest_eigenpair`], preconditioned by eps_a - eps_i) with the
/// Hessian-vector products as finite differences of the gradient (h = 1e-3), one-sided from
/// `g0` when given (one Fock build each), else central (two). The first unit vector (on
/// [`start_pair`]) is `start_sign` times the unit vector: with one-sided products each
/// product's O(h) error changes sign with the direction of the rotation, so the start's
/// direction is fixed by the phases of its two orbitals ([`phase`], decision D29), not by the
/// eigensolver's signs (the probe and the mode check; the full analysis' central differences do
/// not depend on it). `start` (an irrep's, [`Starts`]) replaces that unit vector. A clearly
/// negative mode (theta < -1e-3 with a residual below |theta| / 2, or |theta| / 10 in a search of
/// [`WIDE_START_PRODUCTS`] or more) is enough to leave the saddle; otherwise the residual norm is
/// converged to 1e-3. Same returns.
#[allow(clippy::type_complexity, clippy::too_many_arguments)]
pub(super) fn davidson(
    grad: &mut dyn FnMut(&[f64], f64) -> Vec<f64>,
    nmo: usize,
    eps: &[&[f64]],
    nocc: &[usize],
    guess: Option<Vec<f64>>,
    max_products: usize,
    g0: Option<&[f64]>,
    start_sign: f64,
    start: Option<Vec<f64>>,
) -> (f64, f64, Vec<f64>, usize, usize) {
    let dims: Vec<(usize, usize)> = nocc.iter().map(|&no| (nmo - no, no)).collect();
    let diag: Vec<f64> = dims
        .iter()
        .enumerate()
        .flat_map(|(s, &(nv, no))| (0..no).flat_map(move |i| (0..nv).map(move |a| (s, i, a))))
        .map(|(s, i, a)| eps[s][nocc[s] + a] - eps[s][i])
        .collect();
    let h = 1e-3;
    let mut product = |v: &[f64]| -> Vec<f64> {
        let gp = grad(v, h);
        match g0 {
            Some(g0) => gp.iter().zip(g0).map(|(p, z)| (p - z) / h).collect(),
            None => {
                let gm = grad(v, -h);
                gp.iter()
                    .zip(&gm)
                    .map(|(p, m)| (p - m) / (2.0 * h))
                    .collect()
            }
        }
    };
    let setup = DavidsonSetup {
        diag: &diag,
        guess,
        start: start.map_or(Start::LowestDiagonal(start_sign), Start::Vector),
        max_products,
    };
    // a clearly negative Ritz value ends the search early: residual below half of it, or a tenth
    // in a search of many products, whose wide start's first negative Ritz value can be a blend of
    // modes (the L-lactide cation, UHF: -5.3e-3 at half, the -0.11 mode at a tenth)
    let early = if max_products >= WIDE_START_PRODUCTS {
        0.1
    } else {
        0.5
    };
    let e = lowest_eigenpair(setup, &mut product, |theta, rnorm| {
        (theta < -1e-3 && rnorm < early * theta.abs()) || rnorm < 1e-3
    });
    // Fock builds: one per one-sided product, two per central one
    let builds = e.products * if g0.is_some() { 1 } else { 2 };
    (e.value, e.residual, e.vector, e.iterations, builds)
}

/// Per-spin rotation blocks kappa_s (nv x no) of a vector flattened as (a, i) blocks per spin,
/// scaled by `scale`.
pub fn unflatten(v: &[f64], nmo: usize, nocc: &[usize], scale: f64) -> Vec<Mat<f64>> {
    let mut off = 0;
    nocc.iter()
        .map(|&no| {
            let nv = nmo - no;
            let m = Mat::from_fn(nv, no, |a, i| scale * v[off + i * nv + a]);
            off += nv * no;
            m
        })
        .collect()
}

/// The rotation kappa (flattened per spin as in [`lowest_hessian_mode`]) from the orbitals `c`
/// towards the occupied orbitals `occ` to first order: kappa_s = (C_v^T S O_s)(C_o^T S O_s)^T.
pub(super) fn rotation_towards(
    c: &[Mat<f64>],
    occ: &[Mat<f64>],
    nocc: &[usize],
    s: MatRef<'_, f64>,
    par: Par,
) -> Vec<f64> {
    let mut out = Vec::new();
    for ((cs, os), &no) in c.iter().zip(occ).zip(nocc) {
        let nmo = cs.ncols();
        let p = mm(cs.transpose(), mm(s, os.as_ref(), par).as_ref(), par);
        let k = mm(p.subrows(no, nmo - no), p.subrows(0, no).transpose(), par);
        for i in 0..no {
            for a in 0..nmo - no {
                out.push(k[(a, i)]);
            }
        }
    }
    out
}

/// The orbital gradient g_s = C_v^T F_s C_o of orbitals `c` (per spin) for Fock matrices `f`,
/// flattened as [`lowest_hessian_mode`]'s vectors: the k = 0 gradient of its one-sided products.
pub(super) fn orbital_gradient(
    c: &[Mat<f64>],
    f: &[&Mat<f64>],
    nocc: &[usize],
    par: Par,
) -> Vec<f64> {
    let mut g = Vec::new();
    for ((cs, fs), &no) in c.iter().zip(f).zip(nocc) {
        let nmo = cs.ncols();
        let gs = mm(
            cs.subcols(no, nmo - no).transpose(),
            mm(fs.as_ref(), cs.subcols(0, no), par).as_ref(),
            par,
        );
        for i in 0..no {
            for a in 0..nmo - no {
                g.push(gs[(a, i)]);
            }
        }
    }
    g
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn external_start_rotates_alpha_and_beta_in_opposite_senses() {
        // 4 orbitals, 2 occupied: pair (occupied 1, virtual 0) at index i nv + a in each spin
        let v = external_start(4, 2, 1, 0, -1.0);
        assert_eq!(v, vec![0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0, 0.0]);
        assert_eq!(ProbeKind::of(&v), ProbeKind::External);
        let same: Vec<f64> = v.iter().map(|x| x.abs()).collect();
        assert_eq!(ProbeKind::of(&same), ProbeKind::Internal);
    }

    #[test]
    fn the_gate_says_why_the_probe_does_not_run() {
        let eps = [-0.5, -0.3, -0.2, 0.4];
        let g = |conv, nocc, thr, n| ClosedShellProbe::gate(conv, &eps, nocc, thr, n);
        let p = g(true, 2, 0.15, 2);
        assert!(p.runs() && (p.gap.unwrap() - 0.1).abs() < 1e-15);
        assert_eq!(
            g(true, 2, 0.05, 2).skipped,
            Some(ProbeSkip::GapAboveThreshold)
        );
        assert_eq!(g(true, 2, 0.15, 0).skipped, Some(ProbeSkip::Off));
        assert_eq!(g(false, 2, 0.15, 2).skipped, Some(ProbeSkip::NotConverged));
        let full = g(true, 4, 0.15, 2);
        assert_eq!(
            (full.gap, full.skipped),
            (None, Some(ProbeSkip::NoVirtualOrbitals))
        );
        let mut p = p;
        p.ran(-0.02, 1e-3, &external_start(4, 2, 1, 0, 1.0), 2, 0.1);
        assert!(p.is_unstable() && p.kind == Some(ProbeKind::External));
        p.ran(-5e-5, 1e-3, &[], 2, 0.1);
        assert!(!p.is_unstable() && p.kind.is_none());
    }

    /// A failed Fock build gives NaN products (the caller returns its error): the Davidson search
    /// ends without a mode instead of panicking in the eigensolver (an `internal` error in a GPU
    /// run whose probe builds ran out of device memory).
    #[test]
    fn davidson_ends_without_a_mode_on_failed_products() {
        let eps: &[f64] = &[-1.0, -0.5, 0.5, 1.0];
        let g0 = vec![0.0; 4];
        let mut nan = |v: &[f64], _h: f64| vec![f64::NAN; v.len()];
        let (theta, rnorm, _, _, builds) =
            davidson(&mut nan, 4, &[eps], &[2], None, 4, Some(&g0), 1.0, None);
        assert!(theta.is_nan() && rnorm.is_nan());
        assert_eq!(builds, 1);
    }
}
