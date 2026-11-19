//! The engine every integral family shares: the Boys function and Rys quadrature, the
//! McMurchie-Davidson Hermite machinery of shell pairs ([`hermite`], [`pair_data`]) and their
//! derivative expansions ([`deriv`]), solid harmonics, the lane-parallel kernels (`kernels`, on
//! linus-num's `simd` layer; thread-count independent reductions are linus-num's `det`) and
//! small shared helpers (`util`: shell-pair transforms, density maxima, gradient sums).

pub mod boys;
pub mod deriv;
pub mod hermite;
pub(crate) mod kernels;
pub mod pair_data;
pub mod rys;
pub mod solid_harmonics;
pub mod switches;
pub(crate) mod util;
