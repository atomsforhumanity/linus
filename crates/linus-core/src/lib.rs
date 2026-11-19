//! linus-core: the vocabulary every linus crate shares.
//!
//! The crates of linus are layered: linus-num and linus-core at the bottom, the method
//! libraries (integrals, grids, functionals, dispersion) and the device engines (linus-gpu)
//! above them, the driver (`linus`) above those, and the front ends on top. A concept that
//! more than one layer needs is defined here once, so no crate keeps a copy of another's type:
//!
//! - [`units`]: the CODATA table and the Bohr radius ([`units::BOHR_ANGSTROM`]);
//! - [`elements`]: the elements by atomic number and symbol;
//! - [`structure`]: the atoms, charge and spin of a calculation ([`Structure`]) and the XYZ reader;
//! - [`channels`]: one value per spin channel ([`Channels`]);
//! - [`gradient`]: a nuclear gradient as a sum of parts ([`Gradient`]);
//! - [`error`]: the kinds of failure ([`ErrorKind`], [`DeviceErrorKind`]) and the errors of
//!   this crate ([`Error`]);
//! - [`memory`]: what a piece of work needs and may use ([`MemoryPlan`], [`Budget`]);
//! - [`arith`]: the arithmetic a kernel family runs in ([`Arith`]);
//! - [`timing`]: phase times and their names ([`Timers`], [`Span`]);
//! - [`record`]: what a calculation decided, as it decided it ([`RunRecord`]).
//!
//! Policy does not live here: this crate names the choices, the driver makes them.

pub mod arith;
pub mod channels;
pub mod elements;
pub mod error;
pub mod gradient;
pub mod memory;
pub mod record;
pub mod structure;
pub mod timing;
pub mod units;

pub use arith::Arith;
pub use channels::Channels;
pub use error::{DeviceErrorKind, Error, ErrorKind};
pub use gradient::Gradient;
pub use memory::{Budget, MemoryPlan, OptionalStorage, OverBudget};
pub use record::{RunRecord, Warning};
pub use structure::Structure;
pub use timing::{Span, Timers};
