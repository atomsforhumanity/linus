//! Grid presets: PySCF, Psi4, ORCA (DefGrid1-3, from tables determined from ORCA's output) and
//! linus.

use crate::batch::Batching;
use crate::elements::RadiiTable;
use crate::grid::{AtomSizes, GridSpec, ORCA_XC_INTACC};
use crate::orient::Orientation;
use crate::partition::{PartitionScheme, SizeAdjust};
use crate::prune::Pruning;
use crate::radial::{ChebyshevKind, RadialMapping, RadialScale, RadialScheme};

impl GridSpec {
    /// `pyscf.dft.Grids(mol)` with default settings at `level` (0-9): Treutler-Ahlrichs M4
    /// radial grid with atom-specific xi, NWChem pruning, Becke partitioning with Treutler
    /// size adjustment on Bragg radii, no weight cutoff. Reproduces PySCF 2.x exactly (points
    /// and weights; PySCF additionally appends zero-weight padding points, which are omitted).
    pub fn pyscf(level: u8) -> GridSpec {
        GridSpec {
            radial: RadialScheme::PYSCF_TREUTLER,
            sizes: AtomSizes::PyscfLevel(level),
            overrides: vec![],
            pruning: Pruning::NWChem,
            partition: PartitionScheme::Becke,
            size_adjust: SizeAdjust::Treutler(RadiiTable::BraggPyscf),
            weight_cutoff: None,
            partition_cutoff: None,
            orientation: Orientation::Identity,
            batching: Batching::default(),
        }
    }

    /// Psi4 1.11 default DFT grid: 75 Treutler radial points (Chebyshev-2, M4 mapping,
    /// Bragg-Slater radius scaling), 302-point Lebedev, no pruning, Becke partitioning with
    /// Treutler size adjustment (Psi4's Bragg-Slater radii), weights <= 1e-15 dropped.
    ///
    /// Psi4 rotates the angular grids into its own "standard orientation"; pass that matrix
    /// (`psi4.core.DFTGrid.orientation()`) via `orientation = Orientation::Matrix(..)` to
    /// reproduce Psi4 exactly. Psi4 additionally removes points beyond the basis-function
    /// extents (`DFT_REMOVE_DISTANT_POINTS`, basis dependent), which is not done here.
    pub fn psi4_default() -> GridSpec {
        GridSpec {
            radial: RadialScheme::Psi4Treutler { bs_alpha: 1.0 },
            sizes: AtomSizes::Uniform {
                n_rad: 75,
                n_ang: 302,
            },
            overrides: vec![],
            pruning: Pruning::None,
            partition: PartitionScheme::Becke,
            size_adjust: SizeAdjust::Treutler(RadiiTable::BraggSlaterPsi4),
            weight_cutoff: Some(1e-15),
            partition_cutoff: None,
            orientation: Orientation::Standard,
            batching: Batching::default(),
        }
    }

    /// ORCA 6.0.0's `DefGrid1/2/3` XC grid (ORCA publishes only its outline, manual section 7.3),
    /// from tables determined from ORCA's printed output ([`crate::orca`]):
    ///
    /// * radial: the Perez-Jorda (Krack-Koster) Gauss-Chebyshev rule with the Treutler-Ahlrichs
    ///   M3 map, ORCA's per-element xi for AngularGrid 3/4/6 and its number of shells;
    /// * angular: the Lebedev sets of AngularGrid 3/4/6 (H/He one lower) in five regions whose
    ///   boundaries are multiples of xi per row and AngularGrid;
    /// * partitioning: Becke with Treutler-Ahlrichs size adjustment on Bragg radii, and ORCA's
    ///   `WeightCut` acting on the partition weight (`partition_cutoff`).
    ///
    /// Not reproduced: ORCA's basis-set-adaptive changes (more core points for core-polarized
    /// basis sets, a finer outer region for diffuse ones, more radial shells for steep s
    /// functions); elements beyond Kr use Treutler's xi and their row's shell count.
    pub fn orca_defgrid(n: u8) -> GridSpec {
        let level = n.clamp(1, 3);
        let g = [3u8, 4, 6][level as usize - 1];
        GridSpec {
            radial: RadialScheme::GaussChebyshev {
                kind: ChebyshevKind::PerezJorda,
                mapping: RadialMapping::TreutlerM3,
                scale: RadialScale::OrcaXi { angular_grid: g },
            },
            sizes: AtomSizes::OrcaDefGrid { level },
            overrides: vec![],
            pruning: Pruning::OrcaMeasured { angular_grid: g },
            partition: PartitionScheme::Becke,
            size_adjust: SizeAdjust::Treutler(RadiiTable::BraggPyscf),
            weight_cutoff: None,
            partition_cutoff: Some(1e-14),
            orientation: Orientation::Identity,
            batching: Batching::default(),
        }
    }

    /// An ORCA-like grid built from the outline in the ORCA 6.0 manual (section 7.3), which
    /// `orca:N` was until 2026-11-07: AngularGrid 3/4/6 with five-region Lebedev pruning (Table
    /// 7.1), XC IntAcc 4.159/4.388/4.959 (Table 7.2), n_r = 15 IntAcc - 40 + 5 ROW,
    /// second-kind Gauss-Chebyshev radial quadrature with the M3 mapping and Treutler-Ahlrichs xi,
    /// H/He one AngularGrid lower, NWChem's Bragg-radius fractions as region boundaries, Becke
    /// partitioning with Becke size adjustment, weight cutoff 1e-14. The base of the COSX grids
    /// ([`GridSpec::cosx`]). Not ORCA's DefGrid, and far less accurate than [`GridSpec::orca_defgrid`]
    /// (B3LYP energies against the converged grid: +9e-6 Eh for L-lactide/def2-SVP, -2.1e-4 for
    /// def2-TZVP, up to +1.9e-3 Eh on molecules with K, P or Cl).
    pub fn orca_defgrid_guess(n: u8) -> GridSpec {
        let g: u8 = match n {
            1 => 3,
            2 => 4,
            _ => 6,
        };
        GridSpec {
            radial: RadialScheme::GaussChebyshev {
                kind: ChebyshevKind::Second,
                mapping: RadialMapping::TreutlerM3,
                scale: RadialScale::TreutlerXi,
            },
            sizes: AtomSizes::OrcaLike {
                angular_grid: g,
                int_acc: ORCA_XC_INTACC[g as usize - 1],
                b: 5.0,
                h_reduced: true,
            },
            overrides: vec![],
            pruning: Pruning::OrcaRegions,
            partition: PartitionScheme::Becke,
            size_adjust: SizeAdjust::Becke(RadiiTable::BraggPyscf),
            weight_cutoff: Some(1e-14),
            partition_cutoff: None,
            orientation: Orientation::Identity,
            batching: Batching::default(),
        }
    }
}

/// IntAcc of ORCA 6.0's COSX grids by AngularGrid 1..=4 (printed by ORCA for GRIDX1-3 of
/// DefGrid1/2/3: AngularGrid 1, 2, 3, 4 come with IntAcc 3.816, 4.020, 4.338, 4.871).
pub const ORCA_COSX_INTACC: [f64; 4] = [3.816, 4.020, 4.338, 4.871];

impl GridSpec {
    /// Seminumerical-exchange (COSX-like) grid with ORCA `AngularGrid` g = 1..=4 (max Lebedev
    /// 50, 110, 194, 302): the [`GridSpec::orca_defgrid_guess`] construction with ORCA 6's COSX
    /// IntAcc for that angular grid ([`ORCA_COSX_INTACC`]), H/He one angular grid lower, and
    /// batches of at most 128 points (64-128, ~90 on average): per
    /// batch and shell pair the integral kernels have a fixed cost that larger batches amortize
    /// against looser screening (u1an39 def2-TZVP RIJCOSX gradient, 8 threads: 56.9 s with 64,
    /// 53.0 s with 128, 52.4 s with 256, 53.0 s with 384). ORCA's DefGrid1/2/3 use g = (1, 1, 2) / (1, 2, 3) /
    /// (2, 3, 4) for GRIDX1 (early SCF iterations), GRIDX2 (final SCF iterations) and GRIDX3
    /// (final energy).
    ///
    /// Rows 3 and below get a radial increment of 11 per row instead of 5 (n_r = 15 IntAcc -
    /// 40 + 11 ROW): with b = 5 the K atom of the ts61 benchmark gave a 1e-4 Eh exchange-energy error
    /// on GridX3, with b = 11 (70 radial shells) 2e-6 Eh.
    pub fn cosx(g: u8) -> GridSpec {
        let g = g.clamp(1, 4);
        let int_acc = ORCA_COSX_INTACC[g as usize - 1];
        let mut s = GridSpec::orca_defgrid_guess(2);
        s.sizes = AtomSizes::OrcaLike {
            angular_grid: g,
            int_acc,
            b: 5.0,
            h_reduced: true,
        };
        let n_ang = crate::prune::orca_angular_max(g as usize);
        s.overrides = (11u32..=86)
            .map(|z| {
                (
                    z,
                    (15.0 * int_acc - 40.0 + 11.0 * crate::elements::row(z) as f64).round()
                        as usize,
                    n_ang,
                )
            })
            .collect();
        s.batching = Batching {
            max_points: 128,
            per_atom: false,
        };
        s
    }
}

/// Radial shells and maximum Lebedev set per period (H-He, Li-Ne, Na-Ar, K-Kr, Rb-Xe, Cs-Rn,
/// Fr-) for the linus production levels 1..=4.
pub const LINUS_SIZES: [[(usize, usize); 7]; 4] = [
    [
        (35, 110),
        (40, 194),
        (45, 194),
        (55, 194),
        (60, 194),
        (65, 194),
        (65, 194),
    ],
    [
        (40, 194),
        (50, 302),
        (55, 302),
        (65, 302),
        (70, 302),
        (75, 302),
        (75, 302),
    ],
    [
        (50, 302),
        (65, 434),
        (70, 434),
        (80, 434),
        (85, 434),
        (90, 434),
        (90, 434),
    ],
    [
        (60, 434),
        (80, 590),
        (90, 590),
        (100, 590),
        (105, 590),
        (110, 590),
        (110, 590),
    ],
];

/// Lebedev sets of the five pruning regions for the linus levels 1..=4 (capped at the
/// element's maximum set); region boundaries are the NWChem Bragg-radius fractions. Level 3
/// (the Hessian/frequency default) prunes the two inner regions to 26/110: 8% fewer points than
/// 50/194 with the same Hessians (lactide, u1an39 frequencies equal to 0.002 cm^-1).
pub const LINUS_REGIONS: [[usize; 5]; 4] = [
    [14, 50, 110, 194, 110],
    [26, 86, 194, 302, 194],
    [26, 110, 302, 434, 302],
    [86, 302, 434, 590, 434],
];

impl GridSpec {
    /// linus production grids, levels 1 (coarse) to 4 (fine). Treutler-Ahlrichs M4 radial grid
    /// with atom-specific xi; five-region pruning at the NWChem Bragg-radius fractions
    /// ([`LINUS_REGIONS`]); Stratmann-Scuseria-Frisch partitioning (exactly screened, O(N)) with
    /// Treutler size adjustment (at level 3 only for the pairs of H with an atom beyond neon); weight
    /// cutoff 1e-15. Level 2 was tuned on
    /// B3LYP/def2-SVP XC energies against ORCA DefGrid2 (see README). Level 3 is the default of
    /// analytic Hessians and frequencies (the driver's `GridChoice::Auto`): B3LYP/def2-SVP
    /// frequencies within 0.15 cm^-1 of the converged grid on the lactide benchmarks and 0.39 on
    /// u1an39 (linus:2 0.9, PySCF level 3 1.7, the ORCA-like DefGrid2 guess 3.1 cm^-1 on the lactides),
    /// and level 1 serves as the CPKS kernel grid (bench/results/2026-11-07-hess-accuracy).
    pub fn linus(level: u8) -> GridSpec {
        let l = (level.clamp(1, 4) - 1) as usize;
        GridSpec {
            radial: RadialScheme::PYSCF_TREUTLER,
            sizes: AtomSizes::PerRow(LINUS_SIZES[l].to_vec()),
            overrides: vec![],
            pruning: Pruning::Regions {
                radii: RadiiTable::BraggPyscf,
                alphas: crate::prune::NWCHEM_ALPHAS,
                grids: LINUS_REGIONS[l],
                cap: true,
            },
            partition: PartitionScheme::SSF,
            // level 3 (the Hessian default): cell boundaries at the bond midpoints. Without the
            // size adjustment each H keeps the density around it on its own grid, which follows
            // it rigidly, instead of on the heavier neighbour's sparser outer shells: soft-mode
            // frequency errors drop from 0.74 to 0.39 cm^-1 (u1an39), energies and gradients are
            // as accurate (bench/results/2026-11-07-hess-accuracy). The pairs of H with an atom
            // beyond neon keep the adjustment: at the midpoint of an X-H bond H's grid integrates
            // X's valence region, and gradients were 1e-4 to 9e-4 Eh/bohr off (PH3, AsH3, H2S,
            // H2Se, HCl, HBr, GaH3; bench/robust/repro/linus3_heavy_hydrides). Adjusting every
            // pair with such an atom also fixed C-I and C-Br pairs but worsened I-F, Be-Cl and
            // Al-Cl ones (IF5 frequencies 0.36 -> 1.06 cm^-1, BeCl2 gradients 3e-6 -> 1.1e-5).
            size_adjust: if level == 3 {
                SizeAdjust::TreutlerBeyondRow2(RadiiTable::BraggPyscf)
            } else {
                SizeAdjust::Treutler(RadiiTable::BraggPyscf)
            },
            weight_cutoff: Some(1e-15),
            partition_cutoff: None,
            orientation: Orientation::Identity,
            batching: Batching::default(),
        }
    }
}
