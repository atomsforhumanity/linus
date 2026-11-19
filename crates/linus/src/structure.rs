//! Structures: linus-core's [`Structure`] (atoms, charge, spin; nuclear repulsion), its XYZ
//! reader, the element table and the Bohr radius, under the paths of linus's facade
//! (`linus::Structure`, `linus::structure::*`). The [`crate::Session`] adds the core electrons
//! its basis set's effective core potentials replace ([`Structure::set_ecp_core_electrons`]).

pub use linus_core::elements::{symbol_to_z, z_to_symbol, MAX_Z, SYMBOLS};
pub use linus_core::structure::{check_distances, parse_xyz, Structure, MIN_DISTANCE};
pub use linus_core::units::BOHR_ANGSTROM;
