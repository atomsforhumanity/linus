//! The solvent in the SCF: the PCM term of one SCF ([`PcmScf`]: surface, operator, integral
//! data), its energy and Fock contribution as the SCF's environment term ([`PcmTerm`]), the
//! term the driver opens for an SCF and closes after it (`ScfEnvironment`, from the backend the
//! plan puts the PCM on, with its Coulomb hook on the loop's J build) and the solvation summary
//! ([`SolvationResult`]). The solvent does not drive the SCF: the driver runs its loop with the
//! term.

use super::pcm::{Charges, PcmOperator};
use super::surface::Surface;
use super::{ri, Discretization, PcmModel, PcmPreset, PcmScfMode, ResolvedPcm};
use faer::{Mat, Par};
use linus_base::error::LinusError;
use linus_base::terms::{EnvironmentTerm, XcDensity};
use linus_core::timing::Timers;
use linus_ints::points::pcm_ints::{GaussPoint, SurfacePlan, SurfaceShape};
use linus_ints::{BasisSet, PairOptions, PairSet};
use serde::Serialize;

/// Solvation results of the last SCF (structured; the CLI and Python report them).
#[derive(Clone, Debug, Serialize)]
pub struct SolvationResult {
    pub model: PcmModel,
    pub eps: f64,
    /// f(eps) of the model.
    pub f_eps: f64,
    pub solvent: Option<String>,
    pub preset: PcmPreset,
    /// Lebedev points of each atom's sphere.
    pub lebedev: Vec<usize>,
    pub discretization: Discretization,
    /// Solute potential at the points (ORCA) rather than over the Gaussian charges (PySCF).
    pub point_potential: bool,
    /// How the SCF evaluated the surface terms ([`PcmScfMode`]).
    pub scf: PcmScfMode,
    /// Sphere radius per atom (Å).
    pub radii_angstrom: Vec<f64>,
    /// Electrostatic solvation energy 1/2 q_sym . phi (Eh), included in the SCF energy.
    pub electrostatic: f64,
    /// Surface points (after switching).
    pub npoints: usize,
    /// Cavity area (Å^2).
    pub area_angstrom2: f64,
    /// Total surface charge sum_g q_sym,g (e).
    pub surface_charge: f64,
    /// Gauss's-law value for the solute charge Q inside the cavity: -(eps - 1)/eps Q.
    pub gauss_charge: f64,
    /// SMD: the non-electrostatic term G_CDS (Eh), included in the energy (None without SMD).
    pub cds: Option<f64>,
    /// Seconds in the PCM terms of the SCF (setup, integrals, solves).
    pub time: f64,
    /// The SCF's PCM term (surface, factorized K, integral data) and the coordinates it was
    /// built for, reused by gradients and Hessians at the same geometry.
    #[serde(skip)]
    pub state: Option<(std::sync::Arc<PcmScf>, Vec<[f64; 3]>)>,
}

/// The PCM term of one SCF: operator at the geometry, integral data and the last state.
pub struct PcmScf {
    pub resolved: ResolvedPcm,
    pub op: PcmOperator,
    /// The points as seen by the solute's potential (exponent infinite for point potentials).
    pub(super) pot_pts: Vec<GaussPoint>,
    pub(super) basis: BasisSet,
    /// The exact (ab|g) plan, built with the term (exact and predicted modes) or on first use
    /// (the density-fitted PCM).
    pub(super) exact: std::sync::OnceLock<SurfacePlan>,
    /// The SCF's density-fitted J side of the `predicted` and `ri` modes (`ri`): auxiliary
    /// basis, (P|g), the state carried between iterations.
    pub ri: Option<ri::RiPcm>,
    /// (phi, charges, energy) of the last density.
    pub last: Option<(Vec<f64>, Charges, f64)>,
    pub timers: Timers,
}

/// The exact (ab|g) plan of the points: the shell pairs and their Schwarz factors, the
/// batched integrals (points grouped by exponent).
fn exact_plan(
    basis: &BasisSet,
    pot_pts: &[GaussPoint],
    thresh: f64,
    timers: Option<&mut Timers>,
) -> SurfacePlan {
    let mut tl = Timers::default();
    let t = timers.unwrap_or(&mut tl);
    let pairs = t.time("pcm_setup", || {
        PairSet::build(basis, PairOptions::default())
    });
    t.time("pcm_setup", || {
        SurfacePlan::new(basis, &pairs, pot_pts, thresh)
    })
}

impl PcmScf {
    pub fn new(
        resolved: ResolvedPcm,
        z: &[u32],
        positions: &[[f64; 3]],
        basis: &BasisSet,
    ) -> Result<Self, String> {
        let mut timers = Timers::default();
        let surf = timers.time("pcm_setup", || {
            Surface::build(
                positions,
                &resolved.radii,
                &resolved.lebedev,
                &resolved.xi,
                resolved.discretization,
                resolved.cut,
            )
        })?;
        let zf: Vec<f64> = z.iter().map(|&x| x as f64).collect();
        let op = timers.time("pcm_setup", || {
            PcmOperator::new(
                resolved.model,
                resolved.eps,
                resolved.cosmo_x,
                surf,
                resolved.point_potential,
                &zf,
                positions,
            )
        })?;
        let pot_pts: Vec<GaussPoint> = op
            .surf
            .pts
            .iter()
            .zip(&op.pot_zeta)
            .map(|(p, &z)| GaussPoint { r: p.r, zeta: z })
            .collect();
        let exact = std::sync::OnceLock::new();
        let _ = exact.set(exact_plan(
            basis,
            &pot_pts,
            resolved.int_thresh,
            Some(&mut timers),
        ));
        Ok(PcmScf {
            resolved,
            op,
            pot_pts,
            basis: basis.clone(),
            exact,
            ri: None,
            last: None,
            timers,
        })
    }

    /// The term for SCF mode `mode` ([`PcmScfMode`]) with the auxiliary basis `aux` of the
    /// density-fitted J (needed by `predicted` and `df`). The density-fitted PCM builds the exact
    /// surface integrals only if they are needed later.
    pub fn with_mode(
        resolved: ResolvedPcm,
        z: &[u32],
        positions: &[[f64; 3]],
        basis: &BasisSet,
        mode: PcmScfMode,
        aux: Option<BasisSet>,
    ) -> Result<Self, String> {
        if mode == PcmScfMode::Exact {
            return Self::new(resolved, z, positions, basis);
        }
        let aux = aux
            .ok_or("the predicted and ri PCM modes need the auxiliary basis of density-fitted J")?;
        let mut timers = Timers::default();
        let surf = timers.time("pcm_setup", || {
            Surface::build(
                positions,
                &resolved.radii,
                &resolved.lebedev,
                &resolved.xi,
                resolved.discretization,
                resolved.cut,
            )
        })?;
        let zf: Vec<f64> = z.iter().map(|&x| x as f64).collect();
        let op = timers.time("pcm_setup", || {
            PcmOperator::new(
                resolved.model,
                resolved.eps,
                resolved.cosmo_x,
                surf,
                resolved.point_potential,
                &zf,
                positions,
            )
        })?;
        let pot_pts: Vec<GaussPoint> = op
            .surf
            .pts
            .iter()
            .zip(&op.pot_zeta)
            .map(|(p, &z)| GaussPoint { r: p.r, zeta: z })
            .collect();
        let exact = std::sync::OnceLock::new();
        if mode == PcmScfMode::Predicted {
            let _ = exact.set(exact_plan(
                basis,
                &pot_pts,
                resolved.int_thresh,
                Some(&mut timers),
            ));
        }
        let ri = timers.time("pcm_setup", || ri::RiPcm::new(aux, &pot_pts, mode));
        Ok(PcmScf {
            resolved,
            op,
            pot_pts,
            basis: basis.clone(),
            exact,
            ri: Some(ri),
            last: None,
            timers,
        })
    }

    /// How this term evaluates the surface terms in the SCF.
    pub fn mode(&self) -> PcmScfMode {
        self.ri
            .as_ref()
            .map(|r| r.mode)
            .unwrap_or(PcmScfMode::Exact)
    }

    /// The exact (ab|g) plan (built on first use in RI mode; the build runs rayon, so not
    /// `get_or_init`: [`linus_base::get_or_build`]).
    pub(super) fn exact(&self) -> &SurfacePlan {
        linus_base::get_or_build(&self.exact, || {
            exact_plan(&self.basis, &self.pot_pts, self.resolved.int_thresh, None)
        })
    }

    /// The points as the solute's potential sees them (Gaussian charges, or point charges with
    /// a point-evaluated potential).
    pub fn points(&self) -> &[GaussPoint] {
        &self.pot_pts
    }

    /// Electronic potentials at the points of densities (row-major nbf^2 each), \[k\]\[point\].
    pub fn electron_potential(&self, ds: &[&[f64]]) -> Vec<f64> {
        let x = self.exact();
        let mut ws = x.workspace(SurfaceShape {
            densities: ds.len(),
            ..Default::default()
        });
        let mut v = vec![0.0; ds.len() * x.points().len()];
        x.potentials(ds, &mut v, &mut ws);
        v
    }

    /// -sum_g q_g (ab|g) for charge sets q.
    pub fn fock(&self, qs: &[&[f64]]) -> Vec<Mat<f64>> {
        let x = self.exact();
        let mut ws = x.workspace(SurfaceShape {
            charge_sets: qs.len(),
            ..Default::default()
        });
        let mut m = surface_mats(self.basis.nbf, qs.len());
        x.matrices(qs, &mut ao_views(&mut m), &mut ws);
        negate(&mut m);
        m
    }

    /// Energy and Fock contribution of total density `dt`.
    pub fn energy_fock(&mut self, dt: &Mat<f64>) -> (f64, Mat<f64>) {
        let t0 = std::time::Instant::now();
        let d = linus_base::linalg::mat_to_rowmajor(dt.as_ref());
        let ve = self.electron_potential(&[&d]);
        let t1 = std::time::Instant::now();
        let phi: Vec<f64> = self.op.v_nuc.iter().zip(&ve).map(|(a, b)| a - b).collect();
        let c = self.op.solve(&phi);
        let e = PcmOperator::energy(&phi, &c);
        let t2 = std::time::Instant::now();
        let v = self.fock(&[&c.q_sym]).remove(0);
        let t3 = std::time::Instant::now();
        self.timers.add("pcm_potential", (t1 - t0).as_secs_f64());
        self.timers.add("pcm_solve", (t2 - t1).as_secs_f64());
        self.timers.add("pcm_fock", (t3 - t2).as_secs_f64());
        self.last = Some((phi, c, e));
        (e, v)
    }

    /// Summary of the last state.
    pub fn result(
        &self,
        solvent: Option<String>,
        preset: PcmPreset,
        solute_charge: i32,
    ) -> SolvationResult {
        let (e, qsum) = match &self.last {
            Some((_, c, e)) => (*e, c.q_sym.iter().sum()),
            None => (0.0, 0.0),
        };
        summary(
            &self.resolved,
            &self.op,
            self.mode(),
            (e, qsum),
            &self.timers,
            (solvent, preset, solute_charge),
        )
    }
}

/// The solvation summary of a PCM term with the operator `op` in SCF mode `mode` after an SCF
/// whose last state had the energy and total surface charge `last`, its `pcm_*` timers summed.
pub fn summary(
    resolved: &ResolvedPcm,
    op: &PcmOperator,
    mode: PcmScfMode,
    last: (f64, f64),
    timers: &Timers,
    (solvent, preset, solute_charge): (Option<String>, PcmPreset, i32),
) -> SolvationResult {
    let (e, qsum) = last;
    let b = match preset {
        PcmPreset::Pyscf => linus_core::units::BOHR_ANGSTROM_PYSCF,
        _ => linus_core::units::BOHR_ANGSTROM,
    };
    let eps = resolved.eps;
    let fg = if eps.is_infinite() {
        1.0
    } else {
        (eps - 1.0) / eps
    };
    SolvationResult {
        model: resolved.model,
        eps,
        f_eps: op.f_eps,
        solvent,
        preset,
        lebedev: resolved.lebedev.clone(),
        point_potential: resolved.point_potential,
        scf: mode,
        discretization: resolved.discretization,
        radii_angstrom: resolved.radii.iter().map(|r| r * b).collect(),
        electrostatic: e,
        npoints: op.npoints(),
        area_angstrom2: op.surf.total_area() * b * b,
        surface_charge: qsum,
        gauss_charge: -fg * solute_charge as f64,
        time: timers
            .entries
            .iter()
            .filter(|e| e.0.starts_with("pcm_"))
            .map(|e| e.1)
            .sum(),
        cds: None,
        state: None,
    }
}

impl std::fmt::Debug for PcmScf {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "PcmScf {{ model: {:?}, npoints: {} }}",
            self.resolved.model,
            self.op.npoints()
        )
    }
}

/// The PCM term as the SCF's environment term: the reaction-field energy 1/2 q_sym . phi and
/// Fock contribution -sum_g q_sym,g (ab|g) of the total density (the density-fitted PCM: only the
/// energy part 1/2 q . v_n; the Fock term is in J, see `ri`).
pub struct PcmTerm {
    pub pcm: std::sync::Arc<std::sync::Mutex<PcmScf>>,
}

impl EnvironmentTerm for PcmTerm {
    fn name(&self) -> &'static str {
        "solvent"
    }

    fn compute(
        &mut self,
        d: &XcDensity<'_>,
        _par: Par,
    ) -> Result<(f64, Option<Mat<f64>>), LinusError> {
        let mut pcm = self.pcm.lock().expect("PCM state");
        let dt = d.da + d.db;
        let (e, v) = match pcm.mode() {
            PcmScfMode::Df => return Ok((pcm.ri_xc_energy(), None)),
            PcmScfMode::Predicted => pcm.predicted_energy_fock(&dt),
            PcmScfMode::Exact => pcm.energy_fock(&dt),
        };
        Ok((e, Some(v)))
    }

    /// The `predicted` mode's steps, but for one after [`EnvironmentTerm::reset_incremental`]
    /// (the `ri` module): their Fock term is of predicted charges.
    fn built_incrementally(&self) -> bool {
        let pcm = self.pcm.lock().expect("PCM state");
        pcm.ri
            .as_ref()
            .is_some_and(|r| r.mode == PcmScfMode::Predicted && r.last_incremental)
    }

    /// The `predicted` mode's next step is exact (the exact mode's two passes); the other modes
    /// always build of the density alone.
    fn reset_incremental(&mut self) {
        let mut pcm = self.pcm.lock().expect("PCM state");
        if let Some(r) = pcm.ri.as_mut().filter(|r| r.mode == PcmScfMode::Predicted) {
            r.full_next = true;
        }
    }
}

/// `count` zero nbf x nbf matrices for surface-integral results.
fn surface_mats(nbf: usize, count: usize) -> Vec<Mat<f64>> {
    (0..count).map(|_| Mat::zeros(nbf, nbf)).collect()
}

/// The matrices as linus-ints views.
fn ao_views(m: &mut [Mat<f64>]) -> Vec<linus_ints::AoMatrixMut<'_>> {
    m.iter_mut()
        .map(|x| linus_base::linalg::ao_view_mut(x.as_mut()))
        .collect()
}

/// Every element negated (the potential's sign of the Fock term).
fn negate(m: &mut [Mat<f64>]) {
    for x in m {
        for j in 0..x.ncols() {
            for v in x.col_as_slice_mut(j) {
                *v = -*v;
            }
        }
    }
}
