//! Linear response: the coupled-perturbed HF/KS (CPHF/CPKS) equations of Hessians, IR
//! intensities and polarizabilities over spin channels, and the second-order properties
//! assembled from their solutions.
//!
//! A second-order property is a perturbation pair over one response framework:
//! - [`perturbation`]: what a static perturbation hands the response and the properties
//!   ([`perturbation::Perturbation`]: F^k C_o at fixed density, the overlap derivative's
//!   occupied projections, the right-hand sides; the nuclear displacements here, the electric
//!   field in `linus::properties::dipole`);
//! - [`operator`]: the response operator, the induced Fock matrix dF\[dD\] C_o composed once per
//!   response from the method families' linear responses (the J/K, the XC kernel, the
//!   environment's `operator::EnvironmentResponse`) and applied in a fixed order;
//! - [`problem`]: the CPHF/CPKS equations (A + B) U = rhs ([`problem::ResponseProblem`]: the
//!   orbitals' zeroth-order part and the operator), their solve, residuals and induced Fock
//!   matrices;
//! - [`solver`]: the solvers, independent of the operator: the block Krylov linear solver and
//!   the Davidson eigensolver that the SCF's stability analysis shares;
//! - [`second_order`]: the response part of a second-order property E^{ba} from perturbation
//!   b's first-order matrices and perturbation a's first-order orbitals, and the reductions the
//!   assemblies use;
//! - [`stationary`]: the stationary (Hylleraas) form of second-order properties, the correction
//!   U^a . r^b that makes their error quadratic in the residuals.
//!
//! The assemblers: the Hessian (`linus::derivatives::hessian`: nuclear displacements twice,
//! with the explicit second derivatives), the dipole derivatives (`linus::properties::dipole`:
//! nuclear displacements and the field) and the polarizability
//! (`linus::properties::polarizability`: the field twice).

pub mod operator;
pub mod perturbation;
pub mod problem;
pub mod second_order;
pub mod solver;
pub mod stationary;
