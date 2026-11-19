//! linus-ints: Gaussian integrals over contracted spherical and Cartesian basis functions,
//! with a McMurchie-Davidson Hermite engine in the factorized (E^T R E) form of SHARK, and
//! bitwise reproducible results at any thread count.
//!
//! # The shape of the API
//!
//! 1. **Immutable inputs.** A [`BasisSet`] (basis functions placed on atoms, from Basis Set
//!    Exchange JSON: [`BasisDef`]) and an [`Operator`] (1/r, erf(omega r)/r and their linear
//!    combinations).
//! 2. **A pair set.** [`PairSet::build`] computes, once per basis and geometry, the Hermite
//!    expansions E of every shell pair and, up to a [`DerivOrder`], their derivatives E^A and
//!    E^AA, all from one primitive loop; the Schwarz and derivative bounds are computed on
//!    first use and kept.
//! 3. **Plans.** A plan per quantity and derivative order borrows the pair set, precomputes
//!    its layout, and reports the memory of its caller-owned workspace before anything runs
//!    ([`WorkspaceSize`]); e.g. [`JkPlan`] for integral-direct J and K. Plans check their
//!    inputs at construction ([`IntsError`]).
//! 4. **Named operations.** A plan's operations write into the caller's buffers through
//!    strided views ([`AoMatrix`], [`AoMatrixMut`]), with no allocation in the hot loops:
//!    [`JkPlan::jk`]; the derivative contractions
//!    ([`Eri4DerivPlan::gradient`], [`Eri4DerivPlan::hessian`], [`eri4c::jk_deriv`]), those
//!    of the density-fitting metric ([`Eri2DerivPlan`]) and the three-center ones
//!    ([`Eri3DerivPlan`]); the three-center kernels ([`eri3c`]); the one-electron integrals,
//!    their derivative matrices and contractions ([`OneEPlan`]); the potential of any set of
//!    charge centers ([`PotentialPlan`]); effective core potentials ([`EcpPlan`]); the
//!    Gaussian surface charges of continuum solvation ([`SurfacePlan`], [`SurfaceDerivPlan`],
//!    [`AuxSurfacePlan`]).
//! 5. **The raw path** ([`raw`]): one shell pair, triplet or quartet (and first
//!    derivatives) into a buffer, libint-style, for tests and external users.
//!
//! Inside, every plan runs a batch engine that walks its batches in a fixed order and hands
//! them to a monomorphized digester ([`engine`]). A digester declares how it adds its
//! contributions up, exactly in fixed point or in floating point
//! ([`engine::Reduction`]), and the one scheduler ([`engine::schedule`]) takes the order of
//! the work from that: results have the same bits for any number of threads.
//!
//! # Conventions
//!
//! - Atomic orbitals: shells in the order of the basis file per atom, atoms in input order;
//!   spherical functions m = -l..l (p shells x, y, z), Cartesian components in lexicographic
//!   order; unit-normalized (PySCF's order and normalization; docs/conventions.md).
//! - Units: bohr and Hartree; integrals in chemists' notation (ab|cd).
//! - Matrices over the basis functions are nbf x nbf; derivative matrices are indexed by the
//!   perturbation x = 3 atom + k (k = x, y, z) first; gradients are per atom ([f64; 3]).
//! - [`raw`] states the layouts of single blocks and their derivatives.
//!
//! # A worked example
//!
//! J and K of a density, then the two-electron gradient of the same density:
//!
//! ```
//! use linus_ints::contract::DensityTerms;
//! use linus_ints::{
//!     AoMatrix, AoMatrixMut, BasisDef, BasisSet, DerivOrder, Eri4DerivOptions, Eri4DerivPlan,
//!     JkArgs, JkOptions, JkOut, JkPlan, JkShape, PairOptions, PairSet,
//! };
//!
//! // water (bohr) in def2-SVP
//! let def = BasisDef::load("def2-svp", None).unwrap();
//! let positions = [[0.0, 0.0, 0.22], [1.43, 0.0, -0.89], [-1.43, 0.0, -0.89]];
//! let basis = BasisSet::new(&def, &[8, 1, 1], &positions, true).unwrap();
//! let n = basis.nbf;
//! // a symmetric density (an SCF's would come from its orbitals)
//! let d: Vec<f64> = (0..n * n).map(|x| 0.01 / (1.0 + (x / n).abs_diff(x % n) as f64)).collect();
//!
//! // one pair set with first-derivative expansions serves J/K and the gradient
//! let pairs = PairSet::build(&basis, PairOptions::with_order(DerivOrder::One));
//!
//! // J and K
//! let plan = JkPlan::new(&basis, &pairs, JkOptions { thresh: 1e-12, ..Default::default() })
//!     .unwrap();
//! let shape = JkShape { ndens: 1, j: true, k: true };
//! println!("workspace: {} bytes per thread", plan.workspace_size(&shape).per_thread);
//! let mut ws = plan.workspace(shape, rayon::current_num_threads());
//! let (mut j, mut k) = (vec![0.0; n * n], vec![0.0; n * n]);
//! plan.jk(
//!     &JkArgs { dens: &[AoMatrix::row_major(&d, n)], thresh: 1e-12 },
//!     &mut JkOut {
//!         j: &mut [AoMatrixMut::row_major(&mut j, n)],
//!         k: &mut [AoMatrixMut::row_major(&mut k, n)],
//!     },
//!     &mut ws,
//! );
//! let e_j: f64 = 0.5 * j.iter().zip(&d).map(|(a, b)| a * b).sum::<f64>();
//! assert!(e_j > 0.0);
//!
//! // the gradient of 1/2 sum D_ab (ab|cd) D_cd - 1/4 sum D_ab (ac|bd) D_cd
//! let terms = DensityTerms::default().coulomb(0.5, &d, &d).exchange(-0.25, &d, &d);
//! let opts = Eri4DerivOptions::default(); // 1/r, screening at 1e-12
//! let grad = Eri4DerivPlan::new(&basis, &pairs, DerivOrder::One, opts).unwrap();
//! let mut gws = grad.workspace(terms.coulomb.len());
//! let mut g = vec![[0.0; 3]; basis.natom()];
//! grad.gradient(&terms, &mut g, &mut gws);
//! // translational invariance: the gradient sums to zero
//! for k in 0..3 {
//!     assert!(g.iter().map(|x| x[k]).sum::<f64>().abs() < 1e-8);
//! }
//! ```
//!
//! `examples/integrals.rs` runs the same steps with timings, and the raw path on single
//! blocks.

pub mod basis;
pub mod contract;
pub mod core;
pub mod engine;
pub mod eri2c;
pub mod eri3c;
pub mod eri4c;
pub mod error;
// the generated kernels, with tables and variants not every caller uses; rustfmt skips them
// through cfg_attr (a plain #[rustfmt::skip] on the module makes its #[macro_export] macros
// macro-expanded, which the crate then cannot name by path)
#[allow(dead_code, clippy::deprecated_cfg_attr)]
#[cfg_attr(rustfmt, rustfmt::skip)]
pub(crate) mod generated;
pub mod one_e;
pub mod operator;
pub mod pairs;
pub mod points;
pub mod potential;
pub mod raw;
// the reference integrals and contractions the tests check the production code against
#[cfg(any(test, feature = "reference"))]
pub mod reference;

pub use basis::{BasisDef, BasisSet, Shell};
pub use contract::{AoMatrix, AoMatrixMut};
pub use engine::WorkspaceSize;
pub use eri2c::{Eri2DerivOptions, Eri2DerivPlan, Eri2DerivWorkspace};
pub use eri3c::deriv_plan::{Eri3Block, Eri3DerivOptions, Eri3DerivPlan, Eri3DerivWorkspace};
pub use eri4c::deriv_plan::{Eri4DerivOptions, Eri4DerivPlan, Eri4DerivWorkspace};
pub use eri4c::jk::{JkArgs, JkOptions, JkOut, JkPlan, JkShape, JkStats, JkWorkspace};
pub use error::IntsError;
pub use one_e::ecp_deriv::{EcpPlan, EcpWorkspace};
pub use one_e::{OneEPlan, OneEWorkspace};
pub use operator::Operator;
pub use pairs::{DerivOrder, PairKind, PairOptions, PairSet};
pub use points::pcm_ints::{
    AuxSurfacePlan, AuxSurfaceWorkspace, GaussPoint, SurfaceDerivPlan, SurfacePlan, SurfaceShape,
    SurfaceWorkspace,
};
pub use potential::{PotentialPlan, PotentialWorkspace};
pub use raw::{Eri3Raw, Eri4Raw, OneERaw};
