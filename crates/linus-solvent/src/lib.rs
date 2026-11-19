//! linus-solvent: implicit solvation with polarizable continuum models (C-PCM, COSMO, IEF-PCM
//! and SS(V)PE) on a smooth surface of Gaussian charges (switching/Gaussian discretization), and
//! SMD. It implements linus-base's environment terms (`EnvironmentTerm`, `EnvironmentResponse`);
//! the `solvent` settings group, its resolution into a [`ResolvedPcm`] and the solvent's dispatch
//! are the facade's (`linus::solvent`).
//!
//! - `surface`: the cavity (atom-centred spheres of Lebedev points, SWIG/ISWIG switching);
//! - `pcm`: the matrices, the solve for the surface charges, the energy and the explicit
//!   derivative terms;
//! - `radii`: radius tables; per-element radii, a scale and eps can also come from outside
//!   (SMD = IEF-PCM or C-PCM electrostatics with SMD radii, plus CDS).
//! - [`db`]: solvent database (dielectric constants from the CRC Handbook via qcref, SMD
//!   descriptors from the Minnesota Solvent Descriptor Database, names and aliases).
//! - [`smd`]: SMD non-electrostatic term G_CDS (energy, gradient, Hessian) and SMD Coulomb radii.
//! - `smd_sasa`: exact solvent-accessible surface area with analytic derivatives.
//! - `ri`: the SCF modes that use the density-fitted J ([`PcmScfMode`]): `predicted` (one
//!   fused exact pass per iteration, the default with density-fitted J) and `df` (the
//!   density-fitted PCM).
//!
//! The SCF sees the solvent as a density-dependent environment term ([`PcmTerm`]): the
//! reaction-field energy E_pcm = 1/2 q_sym . phi and Fock contribution -sum_g q_sym,g (ab|g),
//! with their own energy component (`ScfResult::e_solvation`) and timer. Gradients add
//! the fixed-density derivative of E_pcm (nuclear, integral and matrix/surface terms) to the
//! usual SCF gradient (whose energy-weighted density already contains the solvent Fock term).
//!
//! Presets ([`PcmPreset`]): `pyscf` reproduces PySCF 2.15's `pyscf.solvent.pcm` (modified
//! Bondi radii x 1.2, 302-point Lebedev spheres, SWIG, its exponent table and cut-offs); the
//! `lebedev` and `discretization` options change the grid exactly as PySCF's
//! `lebedev_order` and `surface_discretization_method` do. `orca` follows ORCA 6.0.0's default
//! C-PCM (Lebedev grids by point density, ISWIG, ORCA's exponent constants and cut-off, the
//! solute potential taken at the points).
//!
//! Solvent names: `SolventSettings::name` takes its permittivity from [`db`] (qcref/CRC at
//! 298.15 K); `eps` overrides it (give the other program's own value for exact comparisons).
//!
//! ECPs: the solute's nuclear potential on the surface uses the effective charges Z - n_core (an
//! ECP has no long-range part); the cavity radii are those of the elements.

pub mod db;
pub(crate) mod derivatives;
pub mod model;
pub mod pcm;
pub(crate) mod pcm_hess;
pub mod radii;
pub mod ri;
pub mod scf;
pub mod smd;
pub(crate) mod smd_sasa;
pub mod surface;

pub use derivatives::*;
pub use model::*;
pub use scf::*;
