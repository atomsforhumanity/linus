# Choosing an integration grid

linus picks the DFT integration grid by what the calculation computes (`--grid auto`, the
default; Python `grid="auto"`):

| Calculation | Default grid | Why |
|---|---|---|
| Energies, gradients | **PySCF level 3** (`pyscf:3`) | More accurate than the faster `linus:2` (about half its gradient error, and up to 7× smaller energy error on larger molecules), and point for point identical to PySCF's default grid, which the PySCF comparisons in `tests/` and `reference/` rely on |
| Hessians, frequencies, IR | **linus level 3** (`linus:3`) | Built for second derivatives (SSF partitioning without atomic size adjustment, except Treutler's for the pairs of H with an atom beyond neon; 434-point valence shells): frequencies within 0.15 cm⁻¹ of the converged grid on the lactide benchmarks, where PySCF level 3 is off by up to 1.7 cm⁻¹ |
| CPKS kernel inside Hessians | `linus:1` (meta-GGAs: `linus:2`) | The response kernel of GGAs and hybrids tolerates coarse grids (< 0.005 cm⁻¹ change); a meta-GGA's does not (r2SCAN on linus:1: frequencies up to 14 cm⁻¹ from linus:4, on linus:2 up to 9); with PySCF, ORCA and Psi4 XC grids the kernel uses the XC grid, as those programs do (`grid.kernel: auto` follows the XC grid's family); set with `grid.kernel` |
| VV10 of wB97M-V, wB97X-V, B97M-V | `linus:1` | Sub-µEh VV10 errors at the point count of ORCA 6.0.0's DefGrid3 NL grid; see [VV10 grids](#vv10-nonlocal-correlation-grids); set with `grid.nlc` |

The energy/gradient default was reviewed on 2026-11-07 and kept at PySCF level 3 (decision
recorded in the "linus engineering log" artifact, which is AFH-internal and not part of this
repository; the measurements are in the tables below).

## When to switch to `linus:2` for speed

`linus:2` has about half the points of PySCF level 3. That halves the XC time and saves 5–30%
of a whole job (most on small molecules, least on large def2-TZVP jobs, where exchange
dominates). The price is about twice the gradient error and a larger energy error on bigger
molecules: on L-lactide both grids are within 10 µEh, but on a 39-atom anion `linus:2` is
38–44 µEh (0.03 kcal/mol) off the converged grid against 5–16 µEh for PySCF level 3. Use it
when throughput matters more than the last few hundredths of a kcal/mol: screening many
conformers, single points that only feed relative energies, or loose optimizations.

```
linus mol.xyz --method b3lyp --basis def2-tzvp --grid linus:2          # CLI
ProgramInput(..., model={"method": "b3lyp", "basis": "def2-tzvp"}, keywords={"grid": "linus:2"})   # Python
```

**Accuracy.** L-lactide (18 atoms), error against PySCF level 7, B3LYP:

| Grid | Points | Energy error SVP / TZVP (µEh) | max abs gradient error SVP / TZVP (Eh/bohr) |
|---|---|---|---|
| `pyscf:3` (default) | 220k | +6.2 / +5.8 | 3.7e-5 / 4.7e-5 |
| `linus:2` | 101k | +4.0 / +9.8 | 7.7e-5 / 7.2e-5 |
| `linus:3` | 186k | −9.1 / −11.8 | 2.9e-5 / 3.2e-5 |
| `linus:4` | 362k | +0.5 / +0.1 | 6.3e-6 / 1.1e-5 |
| Psi4 default (`psi4`) | — | −5.4 / +2.9 | 4.1e-5 / 4.6e-5 |
| `orca:2` (ORCA 6.0.0 DefGrid2) | 86k | +3.9 / +10.9 | — |

39-atom anion (`bench/molecules/u1an39.xyz`, charge −1), error against PySCF level 7, B3LYP,
RI-JK:

| Grid | Energy error SVP / TZVP (µEh) | max abs gradient error SVP / TZVP (Eh/bohr) |
|---|---|---|
| `pyscf:3` (default) | +15.8 / +5.2 | 1.1e-4 / 1.3e-4 |
| `linus:2` | +44.4 / +37.5 | 2.1e-4 / 1.9e-4 |

**Speed.** Wall time of whole CLI jobs, B3LYP, RI-JK (def2-universal-jkfit), 8 threads on the
physical cores of one CCD (not in the benchmark lock; faster of two interleaved runs):

| Case | Job | `pyscf:3` (s) | `linus:2` (s) | Change | XC time `pyscf:3` → `linus:2` (s) |
|---|---|---|---|---|---|
| L-lactide SVP | energy | 1.15 | 0.79 | −31% | 0.73 → 0.38 |
| L-lactide SVP | energy + gradient | 1.93 | 1.51 | −22% | 1.00 → 0.56 |
| L-lactide TZVP | energy | 3.56 | 2.66 | −25% | 1.93 → 1.06 |
| L-lactide TZVP | energy + gradient | 5.89 | 4.96 | −16% | 2.50 → 1.44 |
| 39-atom anion SVP | energy | 10.02 | 8.72 | −13% | 2.79 → 1.54 |
| 39-atom anion SVP | energy + gradient | 17.95 | 16.29 | −9% | 3.81 → 2.36 |
| 39-atom anion TZVP | energy | 75.87 | 69.75 | −8% | 17.24 → 9.44 |
| 39-atom anion TZVP | energy + gradient | 118.13 | 111.71 | −5% | 22.10 → 12.48 |

For tight geometry optimizations, prefer `linus:3` or `linus:4`, which have the smallest gradient errors.

## VV10 (nonlocal correlation) grids

The -V functionals (wB97M-V, wB97X-V, B97M-V) add VV10 nonlocal correlation, a double sum over
grid points that costs the square of the point count. Like PySCF, ORCA and Psi4, linus evaluates
it on its own, coarser grid, chosen by `grid.nlc` (CLI `--set grid.nlc=...`, Python `grid={"nlc": ...}`). The
default (`auto`) follows the XC grid, so that a reference program's grid comes with that
program's VV10 grid:

| XC grid | NLC grid (`auto`) | Note |
|---|---|---|
| `auto` (default), `linus:N`, others | **`linus:1`** | production default (below) |
| `pyscf:N` | `pyscf:3` | PySCF 2.15's default `nlcgrids` (level 3 whatever the XC level); PySCF energies reproduced to 1e-12 Eh |
| `orca:3` | ORCA's NL grid of DefGrid3 (`GridSpec::orca_nl`), the `orca:1` XC grid | ORCA 6.0.0's NL grid for DefGrid3 is its DefGrid1 XC grid (same point count); OMol25's level |
| `orca:1`, `orca:2` | ORCA's NL grid of DefGrid1/2, approximated (`GridSpec::orca_nl`) | AngularGrid 1/2 at IntAcc 4.004; point counts fitted, not measured |
| `psi4` | Psi4's VV10 grid | 50 radial x 146 Lebedev points (`DFT_VV10_RADIAL_POINTS`, `DFT_VV10_SPHERICAL_POINTS`) |

**Accuracy** of the VV10 energy and the VV10 part of the gradient at a fixed wB97M-V/def2-TZVPD
density, against PySCF level 6 (L-lactide) or 4 (39-atom anion):

| NLC grid | points L-lactide / anion | dE_vv10 L-lactide / anion (µEh) | max abs dG_vv10 (Eh/bohr) |
|---|---|---|---|
| `linus:1` (default) | 49650 / 119348 | −0.8 / +0.5 | 7.3e-6 / 1.0e-5 |
| `orca:1` (ORCA 6.0.0 DefGrid3 NL) | 43962 / 114574 | +1.6 / +7.7 | 6.1e-6 / 1.1e-5 |
| `pyscf:1` | 71576 / 184384 | −0.6 / +1.3 | 7.6e-6 / 1.1e-5 |
| `pyscf:3` (PySCF 2.15 default) | 219500 / 521692 | −0.1 / −0.4 | 7.7e-7 / 3.3e-6 |
| `pyscf:0` | 15820 / 39044 | +129 / −311 | 5.1e-5 / 8.8e-5 |
| ORCA 6.0.0 DefGrid2 NL (approx.) | 27943 / 74010 | +17 / −19 | 1.5e-5 / 4.2e-5 |
| Psi4 default | 124399 / 266107 | −0.6 / −17 | 2.1e-5 / 4.2e-5 |
| `linus:2` | 101234 / 244663 | +0.3 / +0.6 | 1.3e-6 / 3.3e-6 |

`linus:1` is the default: errors below 1 µEh on both molecules, smaller than those of the XC grid
(PySCF level 3: 2–16 µEh, 3e-5–1e-4 Eh/bohr), at the point count of ORCA's DefGrid3 NL grid and
0.65x PySCF's level 1 (the pair sums cost 0.42x; PySCF 2.15's default, level 3, has 4.4x the
points). ORCA's own DefGrid3 NL grid (`orca:1`) is nearly as cheap but 8 µEh off on the
fluorinated anion; ORCA's DefGrid2 NL grid and Psi4's are 17–19 µEh off there. `linus:2` costs
four times as much and gains little.

**Meta-GGAs (wB97M-V/-D3BJ, B97M-V) on XC grids.** From ORCA's DefGrid2 (`orca:2`) up, wB97M's
grid errors are those of wB97X on the same grid (def2-TZVP, L-lactide and the 39-atom anion: 1–42
µEh, gradients 1e-5–9e-5 Eh/bohr for both); only DefGrid1 (`orca:1`) breaks for the meta-GGA
(−172 µEh on the anion, wB97X +22). The meta-GGA costs more per point (tau), not more
points.

## Other presets

- `pyscf:N` (N = 0–9): PySCF 2.15's `dft.Grids` levels, reproduced point for point.
- `psi4`: Psi4's default grid.
- `linus:N` (N = 1–4): linus's own grids (SSF partitioning, Treutler–Ahlrichs radial grids).
- `orca:N`: ORCA 6.0.0's DefGrid1–3, reproduced from ORCA's printed output (same points; B3LYP
  energies within 2e-7 Eh of ORCA 6.0.0's B3LYP/G single points with def2-SVP and def2-TZVP,
  VeryTightSCF, on the same bohr geometries), apart from ORCA's
  basis-set-adaptive changes. Use it for comparisons with ORCA on the same grid. Until
  2026-11-07 `orca:N` was a guessed ORCA-like grid (old L-lactide row: 90k points, +8.8 / −206.5
  µEh, gradients 1.8e-4 / 1.7e-4 Eh/bohr; up to 1.9e-3 Eh off on molecules with K, P or Cl).
  The DefGrid tables and the ORCA comparisons here are included with permission of FAccTs GmbH
  ([THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md#orca-faccts-gmbh)).
- COSX grids (`cosx:N`, `gridx:A,B,C`) apply to seminumerical exchange (`jk.k_grid`), not to XC.

See `crates/linus-grid/README.md` for how each preset is built and validated.
