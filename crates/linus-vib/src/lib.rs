//! Harmonic vibrational analysis and ideal-gas thermochemistry from a Cartesian Hessian.
//!
//! [`harmonic_analysis`] turns a Hessian (any program's; linus's from the facade's
//! `Session::hessian`) into frequencies, normal modes,
//! reduced masses, force constants, rotational constants and the point group;
//! [`thermo::thermochemistry`] turns that into ZPE, E, H, S, G at T and P (RRHO, optionally
//! quasi-RRHO). A `hessian` calculation (the facade's `Session::run`)
//! does Hessian, dipole derivatives (IR intensities), analysis and thermochemistry in one call.
//!
//! Conventions (atomic units unless stated):
//! - masses: most abundant isotope by default ([`masses`]), per-atom overrides;
//! - the Hessian is symmetrized, mass-weighted, H_m = M^-1/2 H M^-1/2, and by default the
//!   translations and infinitesimal rotations about the centre of mass (Eckart conditions) are
//!   projected out: H_m is diagonalized in the orthogonal complement of the 6 (linear: 5; atom:
//!   3) translation/rotation vectors, giving 3N-6 (3N-5) modes, as PySCF's
//!   `harmonic_analysis`. Psi4 and ORCA project with P H_m P in the full space (same modes, six
//!   extra zeros). `Projection::Translations` and `Projection::None` keep the rotations (or
//!   everything): use them off stationary points to see what projection hides; the 3 (5, 6)
//!   modes closest to rotations/translations are then flagged `external` and left out of
//!   thermochemistry;
//! - frequencies in cm^-1 from the eigenvalues λ of the (projected) mass-weighted Hessian,
//!   ν = sign(λ) sqrt(|λ|) / (2π c): imaginary frequencies are negative numbers, modes are in
//!   ascending order;
//! - `modes_mass_weighted[i]`: the orthonormal eigenvector L_i (3N); `modes_cartesian[i]`:
//!   the Cartesian displacement M^-1/2 L_i scaled to unit length (qcdata's
//!   `normal_modes_cartesian`, ORCA's `$normal_modes`); the reduced mass is
//!   μ_i = 1 / |M^-1/2 L_i|^2 (u), the force constant k_i = μ_i λ_i (mdyn/Å). The sign of each
//!   mode is fixed so that its largest mass-weighted component is positive;
//! - rotational constants A >= B >= C in cm^-1 (and GHz); None for a zero moment of inertia;
//! - IR intensities (km/mol) from dipole derivatives dμ/dX (atomic polar tensors):
//!   I_i = N_A π / (3 c^2 4π ε0) |dμ/dQ_i|^2, reported for imaginary modes too (ORCA leaves
//!   them out);
//! - symmetry number from the detected point group ([`symmetry`], tolerance 0.01 bohr; the
//!   result says which group, how far the structure is from it, and warns when σ > 1 rests on
//!   an approximately symmetric structure), or `symmetry_number` given.
//!
//! Checked against PySCF 2.15 and Psi4 1.11 on the same Hessians (bench/vib/validate.py): with
//! the same masses and CODATA 2014, frequencies agree with PySCF and Psi4 to 5e-6 cm^-1, IR
//! intensities with Psi4 to 1e-10 km/mol, and E, H, S, G (RRHO and both quasi-RRHO variants) to
//! 1e-9 Eh. What differs between programs: masses (Psi4 and Gaussian use isotopes, PySCF's
//! `thermo` standard atomic weights: up to 0.4% in X-H stretches); constants (CODATA 2014 vs
//! 2022: ~1e-8 Eh in G); Psi4 1.11 takes rotational constants about the coordinate origin, so
//! with `no_com` (as qccodec runs it) an off-centre structure gets wrong rotational entropy;
//! tolerances of point-group detection differ (PySCF and Psi4 call la18, C2 to 1.5e-3 bohr,
//! C1).

pub mod freq;
pub mod masses;
pub mod thermo;
pub use linus_base::symmetry;
pub use linus_core::units;

use faer::Mat;
pub use freq::FrequencyOptions;
use linus_base::error::LinusError;
use linus_base::symmetry::v3::*;
pub use masses::MassTable;
use serde::{Deserialize, Serialize};
pub use symmetry::PointGroup;
pub use thermo::{thermochemistry, ThermoOptions, Thermochemistry};
pub use units::{Codata, Constants};

/// What is projected out of the mass-weighted Hessian before diagonalization.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Projection {
    /// Translations and rotations (Eckart): 3N-6 modes (linear 3N-5, atom 0).
    #[default]
    #[serde(alias = "full", alias = "eckart")]
    TranslationsRotations,
    /// Translations only: 3N-3 modes, the 3 (linear: 2) most rotation-like flagged external.
    Translations,
    /// Nothing: 3N modes, the 6 (5, 3) most translation/rotation-like flagged external.
    None,
}

/// Settings of the harmonic analysis (from JSON with [`from_json_strict`], which rejects
/// unknown keys; no `deny_unknown_fields` here so that [`FrequencyOptions`] can flatten it).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct VibOptions {
    /// Table for the default masses.
    pub mass_table: MassTable,
    /// Masses of all atoms (u), replacing the table's.
    pub masses: Option<Vec<f64>>,
    /// (atom index, mass in u) for single atoms, applied last (e.g. deuterium 2.01410177812).
    pub isotopes: Vec<(usize, f64)>,
    pub projection: Projection,
    /// Rotational symmetry number; None: from the detected point group.
    pub symmetry_number: Option<u32>,
    /// Tolerance (bohr) of the point-group detection and of the test for linear molecules.
    pub symmetry_tolerance: f64,
    /// Physical constants.
    pub codata: Codata,
}

impl Default for VibOptions {
    fn default() -> Self {
        VibOptions {
            mass_table: MassTable::Isotope,
            masses: None,
            isotopes: Vec::new(),
            projection: Projection::TranslationsRotations,
            symmetry_number: None,
            symmetry_tolerance: symmetry::DEFAULT_TOLERANCE,
            codata: Codata::Codata2022,
        }
    }
}

impl VibOptions {
    /// The values that do not depend on the atoms in range: a symmetry number of at least 1,
    /// a positive symmetry tolerance.
    pub fn check(&self) -> Result<(), LinusError> {
        if self.symmetry_number == Some(0) {
            return Err(LinusError::Input(
                "the symmetry number must be at least 1".into(),
            ));
        }
        if !(self.symmetry_tolerance > 0.0 && self.symmetry_tolerance.is_finite()) {
            return Err(LinusError::Input(format!(
                "symmetry_tolerance must be a positive number of bohr, not {}",
                self.symmetry_tolerance
            )));
        }
        Ok(())
    }

    /// Per-atom masses (u) for atomic numbers `z`.
    pub fn masses_for(&self, z: &[u32]) -> Result<Vec<f64>, LinusError> {
        let mut m = match &self.masses {
            Some(m) if m.len() != z.len() => {
                return Err(LinusError::Input(format!(
                    "{} masses for {} atoms",
                    m.len(),
                    z.len()
                )))
            }
            Some(m) => m.clone(),
            None => masses::masses(z, self.mass_table, &[]).map_err(LinusError::Input)?,
        };
        if m.iter().any(|x| !(*x > 0.0 && x.is_finite())) {
            return Err(LinusError::Input("masses must be positive".into()));
        }
        for &(i, mi) in &self.isotopes {
            if i >= m.len() || !(mi > 0.0 && mi.is_finite()) {
                return Err(LinusError::Input(format!(
                    "bad isotope mass {mi} for atom {i} ({} atoms)",
                    m.len()
                )));
            }
            m[i] = mi;
        }
        Ok(m)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RotorType {
    Atom,
    Linear,
    Nonlinear,
}

/// Point group and the rotational symmetry number used (flattened into the analysis's JSON, with
/// qcdata's names `symmetry_number_source` and linus's `symmetry_tol`).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Symmetry {
    /// Detected point group (Schoenflies; "C*v"/"D*h" linear, "Kh" atom).
    pub point_group: String,
    /// σ used for the rotational entropy.
    pub symmetry_number: u32,
    /// σ of the detected point group.
    pub detected_symmetry_number: u32,
    /// "detected" or "input".
    #[serde(rename = "symmetry_number_source")]
    pub source: String,
    /// Tolerance of the detection (bohr) and the largest deviation of an atom from its image.
    #[serde(rename = "symmetry_tol")]
    pub tolerance: f64,
    pub max_deviation: f64,
}

/// Result of a harmonic analysis. Modes are in ascending order of frequency.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct VibrationalAnalysis {
    pub natoms: usize,
    pub numbers: Vec<u32>,
    /// Masses used (u).
    pub masses: Vec<f64>,
    pub projection: Projection,
    pub codata: Codata,
    pub rotor_type: RotorType,
    /// Centre of mass (bohr).
    pub center_of_mass: [f64; 3],
    /// Principal moments of inertia (u bohr^2), ascending.
    pub moments_of_inertia: [f64; 3],
    /// Principal axes (unit vectors), in the order of the moments.
    pub principal_axes: [[f64; 3]; 3],
    /// Rotational constants A >= B >= C (cm^-1); None for a zero moment.
    pub rotational_constants: [Option<f64>; 3],
    /// The same in GHz.
    pub rotational_constants_ghz: [Option<f64>; 3],
    #[serde(flatten)]
    pub symmetry: Symmetry,
    /// Harmonic frequencies (cm^-1), imaginary ones negative.
    pub frequencies: Vec<f64>,
    /// Eigenvalues of the (projected) mass-weighted Hessian (Eh / (bohr^2 u)).
    pub eigenvalues: Vec<f64>,
    /// Reduced masses (u).
    pub reduced_masses: Vec<f64>,
    /// Force constants μ λ (mdyn/Å), negative for imaginary modes.
    pub force_constants: Vec<f64>,
    /// Modes that are translations/rotations rather than vibrations (only without full
    /// projection); thermochemistry leaves them out.
    pub external: Vec<bool>,
    /// Number of imaginary vibrational (non-external) modes.
    pub n_imaginary: usize,
    /// Unit-length Cartesian displacements (3N each, atom-major: x0 y0 z0 x1 ...; qcdata's
    /// `normal_modes_cartesian`, which reshapes them to n_modes x n_atoms x 3).
    #[serde(rename = "normal_modes_cartesian")]
    pub modes_cartesian: Vec<Vec<f64>>,
    /// Orthonormal eigenvectors of the mass-weighted Hessian (3N each).
    #[serde(rename = "normal_modes_mass_weighted")]
    pub modes_mass_weighted: Vec<Vec<f64>>,
    /// IR intensities (km/mol), with dipole derivatives.
    pub ir_intensities: Option<Vec<f64>>,
    /// max |H - H^T| of the input Hessian (Eh/bohr^2), before symmetrization.
    pub hessian_asymmetry: f64,
    /// Things to check (approximate symmetry, an asymmetric Hessian); the data above is complete
    /// without them.
    #[serde(default)]
    pub warnings: Vec<String>,
}

impl VibrationalAnalysis {
    /// Frequencies of the vibrations (external modes left out), ascending.
    pub fn vibrational_frequencies(&self) -> Vec<f64> {
        self.frequencies
            .iter()
            .zip(&self.external)
            .filter(|(_, e)| !**e)
            .map(|(f, _)| *f)
            .collect()
    }

    /// Imaginary frequencies (negative numbers, cm^-1) of the vibrations.
    pub fn imaginary_frequencies(&self) -> Vec<f64> {
        self.vibrational_frequencies()
            .into_iter()
            .filter(|f| *f < 0.0)
            .collect()
    }

    /// Total mass (u).
    pub fn total_mass(&self) -> f64 {
        self.masses.iter().sum()
    }
}

/// Harmonic analysis of the Cartesian Hessian `hessian` ((3N)^2 row-major, Eh/bohr^2, index
/// 3 * atom + k) of atoms `z` at `positions` (bohr). `dipole_derivatives`, if given, are
/// dμ_c/dX for each of the 3N coordinates (rows 3 * atom + k, columns c = x, y, z; e = atomic
/// units of dipole per bohr) and give IR intensities.
pub fn harmonic_analysis(
    z: &[u32],
    positions: &[[f64; 3]],
    hessian: &[f64],
    opts: &VibOptions,
    dipole_derivatives: Option<&[[f64; 3]]>,
) -> Result<VibrationalAnalysis, LinusError> {
    let natom = z.len();
    let n3 = 3 * natom;
    if natom == 0 || positions.len() != natom {
        return Err(LinusError::Input(format!(
            "{} atomic numbers but {} positions",
            natom,
            positions.len()
        )));
    }
    if hessian.len() != n3 * n3 {
        return Err(LinusError::Input(format!(
            "the Hessian of {natom} atoms needs {} elements, not {}",
            n3 * n3,
            hessian.len()
        )));
    }
    if hessian
        .iter()
        .chain(positions.iter().flatten())
        .any(|x| !x.is_finite())
    {
        return Err(LinusError::Input(
            "the Hessian and positions must be finite".into(),
        ));
    }
    if let Some(d) = dipole_derivatives {
        if d.len() != n3 {
            return Err(LinusError::Input(format!(
                "dipole derivatives need {n3} rows, not {}",
                d.len()
            )));
        }
    }
    let cst = Constants::of(opts.codata);
    let m = opts.masses_for(z)?;
    let mtot: f64 = m.iter().sum();
    let mut com = [0.0; 3];
    for (mi, r) in m.iter().zip(positions) {
        for k in 0..3 {
            com[k] += mi * r[k] / mtot;
        }
    }
    let pos: Vec<V3> = positions.iter().map(|r| sub(*r, com)).collect();
    let (moments, axes) = symmetry::inertia(&m, &pos);
    let tol = opts.symmetry_tolerance;
    let rotor_type = if natom == 1 {
        RotorType::Atom
    } else if pos
        .iter()
        .map(|r| norm(cross(*r, axes[0])))
        .fold(0.0, f64::max)
        < tol
    {
        RotorType::Linear
    } else {
        RotorType::Nonlinear
    };
    // point group and symmetry number
    let pg = symmetry::detect(z, &m, positions, tol);
    opts.check()?;
    let sigma = opts.symmetry_number.unwrap_or(pg.symmetry_number);
    let symmetry = Symmetry {
        point_group: pg.name.clone(),
        symmetry_number: sigma,
        detected_symmetry_number: pg.symmetry_number,
        source: if opts.symmetry_number.is_some() {
            "input"
        } else {
            "detected"
        }
        .into(),
        tolerance: tol,
        max_deviation: pg.max_deviation,
    };
    // rotational constants, A >= B >= C (smallest moment first)
    let rot = |f: &dyn Fn(f64) -> f64| -> [Option<f64>; 3] {
        let zero = |i: usize| match rotor_type {
            RotorType::Atom => true,
            RotorType::Linear => i == 0,
            RotorType::Nonlinear => false,
        };
        [0, 1, 2].map(|i| (!zero(i) && moments[i] > 0.0).then(|| f(moments[i])))
    };
    let rotational_constants = rot(&|i| cst.rotational_constant_wavenumber(i));
    let rotational_constants_ghz =
        rot(&|i| cst.rotational_constant_wavenumber(i) * cst.c * 100.0 * 1e-9);
    // symmetrized, mass-weighted Hessian
    let mut asym: f64 = 0.0;
    let sm: Vec<f64> = (0..n3).map(|i| m[i / 3].sqrt()).collect();
    let hm = Mat::from_fn(n3, n3, |i, j| {
        asym = asym.max((hessian[i * n3 + j] - hessian[j * n3 + i]).abs());
        0.5 * (hessian[i * n3 + j] + hessian[j * n3 + i]) / (sm[i] * sm[j])
    });
    // translation and rotation vectors in mass-weighted coordinates
    let trans: Vec<Vec<f64>> = (0..3)
        .map(|k| {
            (0..n3)
                .map(|i| if i % 3 == k { sm[i] } else { 0.0 })
                .collect()
        })
        .collect();
    let rot_axes: Vec<V3> = match rotor_type {
        RotorType::Atom => vec![],
        RotorType::Linear => vec![axes[1], axes[2]],
        RotorType::Nonlinear => axes.to_vec(),
    };
    let rots: Vec<Vec<f64>> = rot_axes
        .iter()
        .map(|e| {
            let mut v = vec![0.0; n3];
            for a in 0..natom {
                let d = cross(*e, pos[a]);
                for k in 0..3 {
                    v[3 * a + k] = sm[3 * a + k] * d[k];
                }
            }
            v
        })
        .collect();
    let tr_all: Vec<Vec<f64>> = orthonormalize(trans.iter().chain(&rots).cloned().collect());
    let n_ext = tr_all.len();
    let projected: Vec<Vec<f64>> = match opts.projection {
        Projection::TranslationsRotations => tr_all.clone(),
        Projection::Translations => orthonormalize(trans.clone()),
        Projection::None => Vec::new(),
    };
    // diagonalize in the complement of the projected space
    let (lam, modes) = if projected.is_empty() {
        linus_base::linalg::eigh(hm.as_ref(), faer::Par::Seq)
    } else {
        let p = Mat::from_fn(n3, n3, |i, j| if i == j { 1.0 } else { 0.0 } - projected.iter().map(|q| q[i] * q[j]).sum::<f64>());
        let (w, v) = linus_base::linalg::eigh(p.as_ref(), faer::Par::Seq);
        let keep: Vec<usize> = (0..n3).filter(|&i| w[i] > 0.5).collect();
        let b = Mat::from_fn(n3, keep.len(), |i, k| v[(i, keep[k])]);
        let hp = linus_base::linalg::mm(
            linus_base::linalg::mm(b.transpose(), hm.as_ref(), faer::Par::Seq).as_ref(),
            b.as_ref(),
            faer::Par::Seq,
        );
        let hp = Mat::from_fn(hp.nrows(), hp.ncols(), |i, j| {
            0.5 * (hp[(i, j)] + hp[(j, i)])
        });
        let (lam, u) = linus_base::linalg::eigh(hp.as_ref(), faer::Par::Seq);
        (
            lam,
            linus_base::linalg::mm(b.as_ref(), u.as_ref(), faer::Par::Seq),
        )
    };
    let nmode = lam.len();
    // flag the most translation/rotation-like modes when they were not projected out
    let n_flag = n_ext - projected.len();
    let mut external = vec![false; nmode];
    if n_flag > 0 {
        let overlap: Vec<f64> = (0..nmode)
            .map(|i| {
                tr_all
                    .iter()
                    .map(|q| (0..n3).map(|j| q[j] * modes[(j, i)]).sum::<f64>().powi(2))
                    .sum()
            })
            .collect();
        let mut idx: Vec<usize> = (0..nmode).collect();
        idx.sort_by(|&a, &b| overlap[b].total_cmp(&overlap[a]).then(a.cmp(&b)));
        for &i in &idx[..n_flag.min(nmode)] {
            external[i] = true;
        }
    }
    let conv = cst.wavenumber_per_sqrt_au();
    let kconv = cst.mdyn_per_angstrom_per_au();
    let mut out = VibrationalAnalysis {
        natoms: natom,
        numbers: z.to_vec(),
        masses: m.clone(),
        projection: opts.projection,
        codata: opts.codata,
        rotor_type,
        center_of_mass: com,
        moments_of_inertia: moments,
        principal_axes: axes,
        rotational_constants,
        rotational_constants_ghz,
        symmetry,
        frequencies: Vec::with_capacity(nmode),
        eigenvalues: lam.clone(),
        reduced_masses: Vec::with_capacity(nmode),
        force_constants: Vec::with_capacity(nmode),
        external: external.clone(),
        n_imaginary: 0,
        modes_cartesian: Vec::with_capacity(nmode),
        modes_mass_weighted: Vec::with_capacity(nmode),
        ir_intensities: None,
        hessian_asymmetry: asym,
        warnings: Vec::new(),
    };
    if opts.symmetry_number.is_none() && pg.symmetry_number > 1 && pg.max_deviation > 1e-4 {
        out.warnings.push(format!(
            "point group {} found with atoms up to {:.1e} bohr from symmetric positions (tolerance {tol} bohr); its \
             symmetry number {} assumes the structure is symmetric: set symmetry_number (or symmetry_tolerance) to override",
            pg.name, pg.max_deviation, pg.symmetry_number
        ));
    }
    if asym > 1e-6 {
        out.warnings.push(format!(
            "the Hessian was asymmetric by up to {asym:.1e} Eh/bohr^2 and was symmetrized"
        ));
    }
    let mut ir = Vec::new();
    for i in 0..nmode {
        let mut l: Vec<f64> = (0..n3).map(|j| modes[(j, i)]).collect();
        // phase: the largest component (first of near-equal ones) positive
        let big = l.iter().fold(0.0_f64, |a, x| a.max(x.abs()));
        let lead = l.iter().position(|x| x.abs() >= big - 1e-8).unwrap_or(0);
        if l[lead] < 0.0 {
            l.iter_mut().for_each(|x| *x = -*x);
        }
        let w: Vec<f64> = (0..n3).map(|j| l[j] / sm[j]).collect();
        let wn2: f64 = w.iter().map(|x| x * x).sum();
        let mu = 1.0 / wn2;
        let lam_i = lam[i];
        out.frequencies
            .push(lam_i.signum() * lam_i.abs().sqrt() * conv);
        out.reduced_masses.push(mu);
        out.force_constants.push(mu * lam_i * kconv);
        if let Some(d) = dipole_derivatives {
            let mut dq = [0.0; 3];
            for j in 0..n3 {
                for c in 0..3 {
                    dq[c] += d[j][c] * w[j];
                }
            }
            ir.push(cst.ir_km_per_mol_per_au() * dot(dq, dq));
        }
        out.modes_cartesian
            .push(w.iter().map(|x| x * mu.sqrt()).collect());
        out.modes_mass_weighted.push(l);
    }
    out.n_imaginary = (0..nmode).filter(|&i| !external[i] && lam[i] < 0.0).count();
    if dipole_derivatives.is_some() {
        out.ir_intensities = Some(ir);
    }
    Ok(out)
}

/// Settings from a JSON object, unknown keys an error: the keys must be those `T::default()`
/// serializes to (every field of these option structs is always serialized).
pub fn from_json_strict<T: Default + Serialize + serde::de::DeserializeOwned>(
    v: serde_json::Value,
    what: &str,
) -> Result<T, LinusError> {
    let known = serde_json::to_value(T::default()).map_err(|e| LinusError::Input(e.to_string()))?;
    let (Some(obj), Some(known)) = (v.as_object(), known.as_object()) else {
        return Err(LinusError::Input(format!("{what} must be a JSON object")));
    };
    let unknown: Vec<&String> = obj.keys().filter(|k| !known.contains_key(*k)).collect();
    if !unknown.is_empty() {
        let mut names: Vec<&String> = known.keys().collect();
        names.sort();
        return Err(LinusError::Input(format!(
            "unknown {what} {unknown:?}; known: {names:?}"
        )));
    }
    serde_json::from_value(v).map_err(|e| LinusError::Input(format!("bad {what}: {e}")))
}

/// Gram-Schmidt (twice) on a list of vectors, dropping those that become (near) zero.
fn orthonormalize(vs: Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    let mut out: Vec<Vec<f64>> = Vec::new();
    for mut v in vs {
        let n0 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        for _ in 0..2 {
            for q in &out {
                let d: f64 = q.iter().zip(&v).map(|(a, b)| a * b).sum();
                v.iter_mut().zip(q).for_each(|(x, y)| *x -= d * y);
            }
        }
        let n = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        if n > 1e-6 * n0.max(1e-300) && n > 1e-10 {
            out.push(v.into_iter().map(|x| x / n).collect());
        }
    }
    out
}

#[cfg(test)]
mod tests;
