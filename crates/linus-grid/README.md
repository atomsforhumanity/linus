# linus-grid

Molecular integration grids for linus DFT: radial and Lebedev angular quadratures, pruning,
Becke / Stratmann-Scuseria-Frisch / Laqua-Kussmann-Ochsenfeld partitioning with analytic
nuclear derivatives of the weights, spatial batching, and presets that reproduce PySCF and
Psi4 grids exactly. Pure Rust (depends on `rayon`, `serde` and linus-num, the workspace's SIMD
layer). Units: bohr.

## API

```rust
use linus_grid::*;

let spec = GridSpec::linus(2);            // or ::pyscf(level), ::psi4_default(), ::orca_defgrid(n)
let grid = MolecularGrid::build(&z, &xyz, &spec)?;   // z: &[u32], xyz: &[[f64; 3]] (bohr)

grid.x, grid.y, grid.z, grid.w            // Vec<f64>, SoA, ordered by batch
grid.w_atomic                             // unpartitioned weight (radial x angular) per point
grid.atom                                 // Vec<u32>: owning atom per point
grid.batches                              // Vec<Batch { start, end, center: [f64; 3], radius }>
grid.atomic_grids, grid.atom_grid         // per-element radial r, w (incl. 4 pi r^2), n_ang per shell
grid.weight_gradient_contract(&f)         // sum_i f_i dw_i/dR_A for all atoms -> Vec<[f64; 3]>
grid.point_weight_gradient(i, &mut scratch, &mut out)  // (atom, dw_i/dR_atom) list for one point
```

Lower-level pieces are public too: `radial::radial_grid(&RadialScheme, n, z)`,
`lebedev::lebedev(npoints)`, `prune::angular_counts(&Pruning, z, &r, n_ang)`,
`atomic_grid(&spec, z)`, and `Partitioner::new(&z, &xyz, PartitionScheme, SizeAdjust)` with
`weight(p, owner, vol, &mut scratch)` and `weight_gradient(p, owner, vol, &mut scratch, &mut out)`.
`GridSpec` is serde-serialisable (see `examples/spec_json.rs`).

Conventions: an atomic grid point has weight `w_rad * w_ang`, with radial weights including
`4 pi r^2` and Lebedev weights summing to one (PySCF's convention). For a GPU backend (cuEST
style: user radial nodes/weights, Lebedev order per shell, Becke/SSF partitioning) everything
needed is in `AtomicGrid` plus the partition scheme; `w_atomic` and `atom` are what grid-weight
gradients need. Weight derivatives assume the points move rigidly with their owning atom; they
sum to zero over atoms (translational invariance).

### Components

| | options |
|---|---|
| radial (`RadialScheme`) | `TreutlerAhlrichs{alpha, scale}` (M4, PySCF formula), `Psi4Treutler{bs_alpha}`, `Becke`, `MuraKnowles{far}`, `KrackKoster` (PySCF `gauss_chebyshev`), `Delley{r_outer}`, `GaussChebyshev{kind: First/Second, mapping: Becke/TreutlerM3/TreutlerM4, scale}`; scales: Treutler xi table, Bragg (PySCF), Psi4 Bragg-Slater, constant |
| angular | Lebedev-Laikov, all 32 sets 6 ... 5810 points (degree 3 ... 131) |
| sizes (`AtomSizes`) | PySCF level tables, uniform, ORCA-like (IntAcc, AngularGrid, b, H reduction), per-row table, per-element overrides |
| pruning (`Pruning`) | `None`, `NWChem` and `SG1` and `Treutler` (exactly as PySCF), `OrcaRegions` (ORCA Table 7.1 sets), `Regions{radii, alphas, grids, cap}` |
| partitioning | `Becke`, `Ssf{a}` (Stratmann et al. 1996), `Lko{r_cut, m_cut}` (Laqua, Kussmann, Ochsenfeld, JCP 149, 204111 (2018), as in PySCF's `becke_lko`); `SizeAdjust::{None, Becke(table), Treutler(table)}` |
| screening | SSF and LKO are screened exactly (only factors that are exactly 0 or 1 are skipped; nearest-atom rejection, SSF w = 1 core radius, neighbour pre-checks): O(1) work per point. Becke cell functions never vanish exactly, so Becke is unscreened (O(N^2) pair loop per point, vectorised over 128-point blocks exactly like PySCF's C code) |
| batching | k-d median bisection, batches of max_points/2..max_points (default 256), optional per-atom batches; bounding sphere = bbox centre + max distance |
| orientation | identity, explicit matrix, or principal axes of the nuclear charge (rotationally invariant grids) |

## Presets and what they reproduce

| preset | reproduces | how verified | max deviation |
|---|---|---|---|
| `GridSpec::pyscf(level)` | `pyscf.dft.Grids` defaults, PySCF 2.15, levels 0-9 (Treutler M4 + atom xi, NWChem pruning, Becke + Treutler/Bragg adjust) | every point, water / benzene / CH3SO2Cl / HBr, all levels (`tools/compare_pyscf_full.py`); sampled points + per-atom moments in `cargo test` | identical point counts (per atom); max abs dx 3.6e-15 bohr; max abs dw 8.5e-14 (water and benzene bit-identical at most levels); dw / w_atomic below 3.3e-15 |
| PySCF options | `radi_method` becke / mura_knowles / gauss_chebyshev / delley; `prune` None / sg1 / treutler; `becke_scheme` stratmann / becke_lko; `radii_adjust` becke / None; `atom_grid` | `cargo test` fixtures (water, CH3SO2Cl, benzene) | below 1e-12 in w except the Becke radial map on S/Cl, whose outermost shells sit about 1000 bohr out: there abs dw is 4e-9 on w around 1e-4 (dw / w_atomic 2e-17). PySCF's own value is ill-conditioned there (0.5 - s(nu) cancellation; one-ulp cos differences between numpy and libm) |
| `GridSpec::psi4_default()` | Psi4 1.11 default DFT grid (75 Treutler/Chebyshev-2 radial on Bragg-Slater radii, 302 Lebedev, unpruned, Becke + Treutler adjust on Psi4's radii, weights <= 1e-15 dropped) **given Psi4's orientation matrix** (`Orientation::Matrix(DFTGrid.orientation())`) | water, benzene, CH3SO2Cl; also `DFT_NUCLEAR_SCHEME BECKE` and 50/194 | identical point counts and sum of w; abs dw 2.4e-14; dx 3.6e-15 |
| `GridSpec::orca_defgrid(1/2/3)` | ORCA 6.0.0 DefGrid1/2/3 XC grids, reproduced from ORCA's printed output (radial grid, pruning regions, partitioning, WeightCut) | ORCA's integrated electron counts of one-s-primitive atoms; ORCA's point counts of isolated atoms (H, C, F, Si, Cl, K, Fe, Br), benzene and L-lactide; B3LYP energies against ORCA 6.0.0 B3LYP/G single points (def2-SVP, def2-TZVP; VeryTightSCF; same bohr geometry) (`tests/orca_defgrid.rs`) | radial: 1e-11 electrons; points: identical except H in DefGrid3; B3LYP energies: water 2e-9, L-lactide 5e-8 (DefGrid1/2/3 1e-7), u1an39 1e-8, ts61 2e-7 Eh (ORCA's basis-adaptive changes not made) |
| `GridSpec::orca_nl(1/2/3)` (`nlc.rs`) | ORCA 6.0.0's VV10 ("NL") grids. DefGrid3: AngularGrid 3 at IntAcc 4.159 = `orca_defgrid(1)` (exact). DefGrid1/2: AngularGrid 1/2 at IntAcc 4.004, approximated (DefGrid1 radial grid, shell counts fitted to ORCA's single-atom NL point counts) | ORCA's NL point counts (water 6340 / 2466 / 3858) | DefGrid3: identical points; DefGrid1/2: counts within ~15% |
| `GridSpec::psi4_vv10()` (`nlc.rs`) | Psi4 1.11's VV10 grid: `psi4_default()` with 50 radial x 146 Lebedev points | VV10 energies vs Psi4 | 1e-8 Eh (model level) |
| `GridSpec::orca_defgrid_guess(1/2/3)` | the ORCA-like grid that `orca:N` was until 2026-11-07 (from the manual's outline); base of the COSX grids | -- | not ORCA's grid (see below) |
| `GridSpec::linus(1..4)` | linus production grids (Treutler M4 + xi, 5-region pruning, SSF + Treutler adjust, weight cutoff 1e-15) | accuracy benchmark below | -- |

Psi4 caveats: Psi4's default "standard orientation" depends on LAPACK eigenvector signs and is
not re-derived here (`psi4_default()` uses our own principal-axis orientation unless given
the matrix). Psi4 also drops points beyond basis-function extents (`DFT_REMOVE_DISTANT_POINTS`,
and octree blocks without significant functions); that is basis dependent and not done here
(fixtures use `DFT_REMOVE_DISTANT_POINTS false`, `DFT_BASIS_TOLERANCE 1e-40`).

### ORCA DefGrids

Section 7.3 of the ORCA 6.0 manual gives only an outline: the five-region angular schemes (Table
7.1); the XC IntAcc per AngularGrid (Table 7.2: 4.004, 4.004, 4.159, 4.388, 4.629, 4.959,
4.959); DefGrid1/2/3 = AngularGrid 3/4/6; n_r = 15 IntAcc - 40 + b ROW; a Gauss-Chebyshev radial
quadrature with the Treutler-Ahlrichs M3 map; H/He one AngularGrid lower; WeightCut 1e-14. The
optimised xi, b and region boundaries, the "mBecke (2022)" partitioning and the basis-adaptive
pruning are not published. `orca_defgrid_guess` filled them with guesses (b = 5, Treutler xi,
NWChem regions, Becke + Becke adjustment); it has about ORCA's point count (benzene 30532 /
58816 / 158952 vs 28216 / 55526 / 138550) but errs by up to 2e-3 Eh (K, P, Cl; C and O with
def2-TZVP).

`orca_defgrid` is ORCA 6.0.0's grid, determined from ORCA's printed output (probes in
`tools/orca_probe/`):

* **Radial grid**: one atom with a one-s-primitive basis has a spherical density, so ORCA's
  printed `N(Total)` is a sum over its radial grid alone. Fitting 18 exponents per element and
  DefGrid gives an exact fit (< 1e-11 electrons) only for the Perez-Jorda (Krack-Koster)
  Gauss-Chebyshev rule with the M3 map (`ChebyshevKind::PerezJorda`), with compact per-element xi
  that differ between AngularGrids (`RadialScale::OrcaXi`; C: 0.440 / 0.391 / 0.394 bohr for
  DefGrid1/2/3, Treutler's 1.3) and 9-10 shells per row (`AtomSizes::OrcaDefGrid`; DefGrid2: H 35,
  Li-Ne 44, Na-Ar 54, K-Kr 64). Rb-Xe (def2-ECP on the probe atom) were not determinable:
  Treutler's xi and the row's shell count are used.
* **Pruning** (`Pruning::OrcaMeasured`): the Lebedev sets of Table 7.1 in five regions with
  boundaries at fixed multiples of xi per row and AngularGrid, solved exactly (one solution each)
  from ORCA's point counts of single atoms over IntAcc 3-8. All elements of a row have the same
  counts at the DefGrid settings. H with AngularGrid 6 (DefGrid3) had no exact solution and uses
  the AngularGrid 4 boundaries (+4% points).
* **Partitioning**: Becke with Treutler-Ahlrichs size adjustment on Bragg radii; of Becke without,
  with Becke and with Treutler adjustment and SSF, only the last matches ORCA's energies (1e-8 Eh;
  the others 2e-6 to 7e-5 Eh off). ORCA's WeightCut acts on the partition weight
  (`GridSpec::partition_cutoff`): an isolated atom keeps all its points, and L-lactide has exactly
  ORCA's 85915.
* **Not made**: ORCA's basis-set-adaptive changes ("core-polarized" basis: more core points, e.g.
  +2184 for K in def2-SVP; "diffuse": the outermost region one AngularGrid up; "steep s": twice the
  radial shells). The detection rules are not published; on ts61 (core-polarized K) the energy
  still agrees to 2e-7 Eh.

The measured tables (per-element xi and shell counts for H-Kr, row shell counts, region
boundaries, and the shell counts of the `orca_nl` grids) are `data/orca/grids.json` of the
repository (generator `tools/gen_orca_tables.py`), compiled into the crate and read by
`src/orca.rs`.

## Lebedev tables: provenance and licence

`src/lebedev_data.rs` is generated by `tools/gen_lebedev_table.py` from SciPy 1.18.1's
`scipy/integrate/_lebedev.py` (BSD-3-Clause), which is Robert Parrish's `getLebedevSphere`
(BSD-2-Clause, (c) 2010, licence text reproduced in the generated file), translated from
C. van Wuellen's Fortran version of D. Laikov's C code. The values are the published
Lebedev-Laikov generators (Lebedev and Laikov, Doklady Mathematics 59, 477 (1999), and
earlier papers). Only orbit generators are stored; points are generated from the Oh orbits
with PySCF's ordering and formulas. The script cross-checks every generator against PySCF's
`LebedevGrid.py` (0 differences). `tests/lebedev_exactness.rs` checks every set integrates
all spherical harmonics to its degree (worst |int Y_lm| = 1e-14) and fails at degree + 1.
Element data (`src/element_data.rs`, from `tools/gen_element_tables.py`) are published radii
(Bragg-Slater, covalent, SG-1, Treutler xi; Psi4's Bragg-Slater numbers).

## Validation (`cargo test -p linus-grid`, no Python needed)

* `pyscf_reference`: 52 PySCF cases (fixtures from `tests/fixtures/gen_pyscf_fixtures.py`):
  atomic radial nodes/weights to 1e-14 relative, angular counts per shell, total and per-atom
  point counts, per-atom compensated moments (sum w, w x, w y, w z, w r^2) to 1e-13, and
  sampled points (coordinates to 1e-12, weights to 1e-12).
* `psi4_reference`: 5 Psi4 cases (`tests/fixtures/gen_psi4_fixtures.py`).
* `lebedev_exactness`: all 32 sets.
* `weight_derivatives`: analytic dw/dR against central finite differences (h = 1e-5) for Becke,
  SSF, LKO with and without size adjustment (water and CH3SO2Cl, 440 points): max error
  2e-10 (FD noise); sum over atoms below 3e-16; grid-level contraction consistency.
* `integration`: promolecular (Slater) densities give the exact electron count to the
  expected accuracy for all presets/levels; screened vs unscreened SSF/LKO weights agree to
  1e-14 (values) and 1e-13 (derivatives); batches cover the grid with valid bounding spheres;
  standard orientation gives rotation-invariant integrals.

## Production grids: accuracy vs points

`tools/bench_accuracy.py`: fixed B3LYP/def2-SVP densities for water, water dimer, glycine,
benzene, CH3SO2Cl, HBr. Error is E_xc(grid) - E_xc(PySCF level 9), in micro-Hartree, as mean
and max abs over the six molecules; points are summed over all six. ORCA 6.0.0 rows are SCF
energy differences against an ORCA reference grid (AngularGrid 7, IntAcc 7.5, unpruned), so
the two columns measure slightly different things.

| grid | points | B3LYP mean / max | r2SCAN mean / max |
|---|---|---|---|
| linus(1) | 117,462 | 31.0 / 50.0 | 678 / 1521 |
| ORCA 6.0.0 DefGrid1 (ORCA itself) | 100,557 | 21.6 / 53.2 | 1227 / 3083 |
| linus(2) | 238,395 | 2.54 / 5.89 | 286 / 958 |
| ORCA 6.0.0 DefGrid2 (ORCA itself) | 197,661 | 2.32 / 6.35 | 168 / 323 |
| linus(3) | 482,937 | 0.57 / 1.73 | 80 / 215 |
| ORCA 6.0.0 DefGrid3 (ORCA itself) | 492,748 | 0.75 / 1.25 | 121 / 291 |
| linus(4) | 852,136 | 0.15 / 0.44 | 13.7 / 31.6 |
| pyscf(3) (PySCF default) | 503,690 | 1.75 / 7.20 | 30.6 / 74.9 |
| psi4 default | 890,626 | 7.7 / 29.8 | 391 / 2173 |
| orca_defgrid(2) as it was (now `orca_defgrid_guess(2)`) | 202,700 | 1152 / 6003 | 1406 / 6561 |

SSF with Treutler size adjustment was more accurate than Becke or LKO on the same atomic
grids (B3LYP mean 2.1 vs 2.7 / 2.6 µEh), and it drops more zero-weight points.
Meta-GGAs (r2SCAN) need linus(3) or more.

## Performance

`examples/grid_bench.rs`: water clusters, 3.1 Å lattice, about liquid density, AMD
Threadripper PRO 9985WX (64 cores / 128 threads), release build, best of N runs:

| system | preset | points | 1 thread | 16 threads |
|---|---|---|---|---|
| benzene (12 atoms) | linus(2) | 67,648 | 0.053 s | 0.007 s |
| benzene | pyscf(3) | 143,556 | 0.036 s (PySCF 2.15: 0.10 s) | 0.007 s (PySCF: 0.09 s) |
| 33 H2O (99 atoms) | linus(1) | 251,530 | 1.6 s | 0.15 s |
| 33 H2O | linus(2) | 506,462 | 3.2 s | 0.21-0.27 s |
| 33 H2O | linus(3) | 1,027,432 | 6.2 s | 0.45 s |
| 33 H2O | linus(2) with LKO | 549,944 | 3.9 s | 0.27 s |
| 33 H2O | pyscf(3) (Becke, unscreened) | 1,112,034 | 7.6 s (PySCF 2.15: 12.6 s) | 0.56 s (PySCF: 2.3 s) |
| 33 H2O | orca_defgrid(2) as it was (Becke; now `orca_defgrid_guess(2)`) | 436,548 | 3.1 s | 0.23 s |
| 33 H2O | psi4_default (Becke) | 2,059,446 | 19.1 s | 1.1 s |
| 333 H2O (999 atoms) | linus(1) | 2,435,610 | 81 s | 6.1 s |
| 333 H2O | linus(2) | 4,893,728 | 172 s | 10.5-11.2 s |
| 333 H2O | linus(2) with LKO | 5,289,789 | 71 s | 5.0 s |

The cost is the partitioning. SSF/LKO work per point is bounded, so cost grows linearly
with system size, but in condensed phase about 45-60 atoms per interstitial point fall inside
SSF's worst-case screening radius (k d_nearest, k = 4.6-5.9 with size adjustment). LKO's
additive cutoff (d + 5 bohr) is about 2x cheaper at 1000 atoms. Both are small next to one
SCF iteration at these sizes. Becke presets (PySCF/Psi4/ORCA-like) are O(N^2) per point and
meant for validation or small molecules.

## Regenerating fixtures

```
.venv/bin/python crates/linus-grid/tests/fixtures/gen_pyscf_fixtures.py
.refenv/bin/python crates/linus-grid/tests/fixtures/gen_psi4_fixtures.py
cargo build --release -p linus-grid --examples
.venv/bin/python crates/linus-grid/tools/compare_pyscf_full.py target/release/examples/grid_dump
.venv/bin/python crates/linus-grid/tools/bench_accuracy.py target/release/examples/grid_dump linus:2 pyscf:3
```

## Known gaps

* ORCA DefGrids cannot be reproduced exactly from the documentation (see above).
* Psi4: orientation must be supplied; `SBECKE` (Psi4's hard-cutoff LKO variant) and basis-extent
  point removal are not implemented.
* Becke partitioning is not screened (O(N^2) per point); use SSF or LKO for large systems.
* Points dropped by the weight cutoff are not included in `weight_gradient_contract` (their
  weights are below 1e-15; for SSF most are exactly zero with zero derivative).
* `Orientation::Standard` is not unique for degenerate charge moments (symmetric tops).
* The citation given for LKO in the task (JCP 148, 214116 (2018)) is the LKO seminumerical
  local-hybrid paper; the partitioning paper (and PySCF's reference) is JCP 149, 204111 (2018).
