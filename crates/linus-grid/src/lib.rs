//! linus-grid: molecular integration grids for DFT.
//!
//! * radial quadratures ([`radial`]): Treutler-Ahlrichs M3/M4, Becke, Mura-Knowles,
//!   Gauss-Chebyshev (first and second kind, Perez-Jorda; any mapping), Krack-Koster, Delley;
//! * Lebedev-Laikov angular grids up to 5810 points ([`lebedev`]);
//! * pruning ([`prune`]): none, NWChem, SG-1, Treutler, ORCA-like regions, ORCA 6.0.0 measured regions, custom regions;
//! * atomic partitioning ([`partition`]): Becke, Stratmann-Scuseria-Frisch,
//!   Laqua-Kussmann-Ochsenfeld, Becke/Treutler size adjustment, exact O(N) screening, and
//!   analytic weight derivatives with respect to nuclear positions;
//! * spatial batching with bounding spheres ([`batch`]);
//! * presets reproducing PySCF and Psi4 grids and ORCA 6.0.0 DefGrids ([`presets`]; the ORCA
//!   ones from tables determined from ORCA's printed output, [`orca`]).
//!
//! Units: bohr. Weights of an atomic grid point are `w_rad * w_ang`, with radial weights
//! including `4 pi r^2` and Lebedev weights summing to one.

#[allow(clippy::excessive_precision, reason = "the tables' digits as published")]
#[rustfmt::skip]
pub(crate) mod element_data;
#[allow(clippy::excessive_precision, reason = "the tables' digits as published")]
#[rustfmt::skip]
pub(crate) mod lebedev_data;

pub mod batch;
pub(crate) mod cells;
pub mod elements;
pub mod grid;
pub mod lebedev;
pub mod nlc;
pub mod orca;
pub mod orient;
pub mod partition;
pub mod presets;
pub mod prune;
pub mod radial;

pub use batch::{bbox, Batch, Batching};
pub use elements::RadiiTable;
pub use grid::{
    atomic_grid, AtomSizes, AtomicGrid, DevicePartition, GridSpec, GridTimes, MolecularGrid,
};
pub use orient::Orientation;
pub use partition::{
    PairScheme, PairTables, PartitionScheme, Partitioner, Scratch, SizeAdjust, ROW2_MAX_Z,
};
pub use prune::Pruning;
pub use radial::{ChebyshevKind, RadialGrid, RadialMapping, RadialScale, RadialScheme};

/// Errors from grid construction.
#[derive(Debug)]
pub enum GridError {
    /// No Lebedev set with this number of points.
    UnsupportedAngular(usize),
    /// A scheme does not support an element.
    UnsupportedElement {
        z: u32,
        what: &'static str,
    },
    InvalidSpec(String),
    /// A device partition failed ([`DevicePartition::weights`]: the device's error, for the
    /// caller to report as its own).
    Device(Box<dyn std::error::Error + Send + Sync>),
}

impl std::fmt::Display for GridError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GridError::UnsupportedAngular(n) => write!(f, "no Lebedev grid with {n} points"),
            GridError::UnsupportedElement { z, what } => {
                write!(f, "{what} not available for Z={z}")
            }
            GridError::InvalidSpec(s) => write!(f, "invalid grid spec: {s}"),
            GridError::Device(e) => write!(f, "device partition: {e}"),
        }
    }
}

impl std::error::Error for GridError {}

/// Builds a molecular grid: `z` nuclear charges, `xyz` positions in bohr.
pub fn build_grid(
    z: &[u32],
    xyz: &[[f64; 3]],
    spec: &GridSpec,
) -> Result<MolecularGrid, GridError> {
    MolecularGrid::build(z, xyz, spec)
}
