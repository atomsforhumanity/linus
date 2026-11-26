//! What linus-gpu takes from the CPU crates, in one place: basis shells and their conventions,
//! the reference implementations the kernels are checked against, grid points, and the
//! functionals. Keeping the import surface in one small module keeps reorganizations of the
//! leaf crates cheap here.

// pair derivative expansions and their CPU reference (crate::eri3c)
pub use linus_ints::core::boys::boys;
pub use linus_ints::core::deriv::{eb_prim, pair_with_deriv};
pub use linus_ints::core::hermite::r_tensor;
pub use linus_ints::core::pair_data::{self, PRIM_PAIR_CUT};
pub use linus_ints::{Eri3Block, Eri3DerivOptions, Eri3DerivPlan};

// the two-center (metric) derivative terms of the DF gradient and Hessian, and the metric's
// derivative integrals (crate::grad::df, crate::hess::df)
pub use linus_ints::core::deriv::PairDeriv;
pub use linus_ints::core::hermite::ShellPair;
pub use linus_ints::{AoMatrix, Eri2DerivOptions, Eri2DerivPlan};

pub use linus_grid::{
    build_grid, Batch, DevicePartition, GridSpec, MolecularGrid, PairScheme, PairTables,
    Partitioner,
};
// grid-weight derivatives (crate::grad::xc)
pub use linus_grid::{PartitionScheme, RadiiTable, SizeAdjust, ROW2_MAX_Z};
pub use linus_ints::basis::{
    cart_components, ncart, search_path, BasisDef, BasisSet, CartLayout, Shell,
};
pub use linus_ints::core::boys::boys_series;
pub use linus_ints::core::hermite::{build_pair, herm_idx, herm_tuv, nherm, ShellView};
pub use linus_ints::core::solid_harmonics::cart2sph;
pub use linus_ints::eri3c::{metric, pair_block, Aux3c, AuxChunk, DfThresholds, Pair3c, Work3c};
pub use linus_ints::eri4c::ERI_PREF;
pub use linus_ints::operator::Operator;
pub use linus_ints::points::ao::{deriv_indices, eval_ao, ncomp};
pub use linus_ints::points::colloc::Collocator;
pub use linus_ints::points::grid_pot::PotPair;
pub use linus_ints::{DerivOrder, PairOptions, PairSet};
// the PCM surface integrals' CPU reference (crate::pcm, its tests)
pub use linus_ints::points::pcm_ints::{self, GaussPoint};
pub use linus_xc::{Family, Functional, XcInput, XcOutput};
// the functionals' libxc trees for the device (crate::xc::devfunc)
#[cfg(feature = "gpu")]
pub use linus_xc::device::{DeviceLeaf, DeviceNode, DevicePlan};
