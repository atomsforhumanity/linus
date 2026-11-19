//! Reference implementations of density fitting (tests only): the dense DF J/K ([`DenseDfJk`]) that
//! the SCF's and the response's RI J/K are checked against, and the in-memory derivative terms
//! ([`DfDerivJk`]) that the blocked RI Hessian is checked against.

mod deriv;
mod jk;

pub use deriv::DfDerivJk;
pub use jk::DenseDfJk;
