# Validation matrix: linus vs PySCF 2.15, Psi4 1.11, ORCA 6.0.0

Energies, dipoles, analytic gradients and analytic Hessians of linus, each compared with three
independent programs, and the three programs with each other, so that every difference can be
pinned on linus or on a reference. Rerunnable with one command; every number below comes from
`results.json` in this directory.

```bash
cargo build --release                                     # linus CLI (target/release/linus)
.refenv/bin/python reference/validate.py --quick          # after every merge: linus only, stored references (4 min);
                                                          # writes only to ../out/validation/quick/
.refenv/bin/python reference/validate.py                  # the whole matrix
.refenv/bin/python reference/validation_summary.py        # print it (--compact, --extras, --problems, --markdown)
.refenv/bin/python reference/validate.py --list           # cases and their runs
```

`validate.py` options: `--cases 'water.*' '*.df'` (fnmatch on case ids), `--skip PATTERN`,
`--core` (one case per feature), `--quick` (the per-merge subset; its linus runs and
`results.json` go to `../out/validation/quick/`, nothing in this directory changes), `--store`
(with `--quick`: write here as a full run does), `--refs missing|all|none` (default `missing`:
rerun a reference only when its stored result is absent or its input specification changed;
`--quick` implies `none`), `--linus run|reuse`, `--linus-bin PATH`, `--programs`, `--lanes`
(default `build,ccd2,ccd3`: one calculation per CPU set at a time; `slice` names or
`cpus:LIST`, e.g. `--lanes build,ccd2,ccd3,ccd4,ccd5,ccd6,cpus:56-63,120-125` when CCDs 4-7
are free, leaving hardware threads 126-127 alone), `--threads 4`, `--dry-run`, `--adopt
CASE:RUN` (store a reference result an interrupted run left in `../out/validation`). Every
calculation runs in its own `systemd-run --user --scope -p MemoryMax=...` (4-7 GB), so an
out-of-memory kill stays inside it. The summary exits 1 when a cell fails without
explanation. The full matrix took ~5 h of reference runs on three shared lanes (2026-11-07);
the linus side alone ~9 min.

## What is stored

| Path | Content |
|---|---|
| `runs/<case>/<run>.json` | one program run: energy, dipole, gradient, Hessian (Eh, e bohr, Eh/bohr, Eh/bohr², 3N x 3N), the program's settings and native input, version, `spec` and `spec_hash` (the input that decides whether a stored reference is still valid), date. linus runs also carry the git commit (`linus_git`) |
| `results.json` | every comparison (pair, quantity, max abs difference, tolerance, verdict, explanation), the linus commits, program versions, tolerances, settings |
| `../out/validation/` | raw outputs (git-ignored): `orca.out`, `psi4.out`, `pyscf.log`, linus requests and stderr |
| `../out/validation/quick/` | what `--quick` writes (git-ignored): `runs/<case>/<run>.json` of its linus runs, read in place of the stored ones, and `results.json` of the quick cases |

Reference results are committed because they are expensive (PySCF Hessians on level-9 grids
take minutes each) and need licensed or slow software; linus is rerun against them. Host names
and absolute paths are stripped before writing.

## Validation map

Where each feature is checked, and the latest committed evidence. `validate.py` patterns are
fnmatch patterns on case ids (`--cases`); deep checks are the per-feature scripts in
`tests/python/` (run with `.refenv/bin/python`) and `reference/checks/`; Rust tests run with
`cargo test --release --workspace`. Commits are the last ones that touched the results file
(short hashes on main). "Matrix" means the PySCF/Psi4/ORCA matrix of this directory, which is
regenerated at the definitive commit; the per-merge `--quick` subset reruns linus on 22 of its
cases against the stored references.

| feature | `validate.py` cases | deep checks | Rust tests | benchmarks, studies | latest results (commit) | status |
|---|---|---|---|---|---|---|
| HF and KS energies, dipoles, gradients, Hessians, closed shell (LDA, GGA, hybrid, meta-GGA) | `water.*` `nh3.*` `ch3sh.*` `clcnh+.*` `la18.*` | `compare_grad_pyscf.py`, `compare_hess_pyscf.py`, `fd_grad_linus.py`, `fd_hess_linus.py` | `linus` unit tests (derivatives, xc) | the benchmark matrix (bench/README.md) | `reference/validation/results.json` (2590114; linus 5b96ef4) | matrix: 0 unexplained cells; quick per merge |
| open shells (UHF/UKS), incl. Hessians | `nh3+.*` `ch2i.*` `au2+.*` `ch3sh+.*` | `compare_hess_pyscf.py --mult` | `derivatives/hessian.rs` `open_shell_matches_finite_differences`, `crates/linus/tests/openshell_scf.rs` | `bench/uks_hess_bench.py`, `bench/openshell_scf.py` |  | energies, gradients and Hessians in the matrix; Hessians also vs PySCF (uhf-hess `validation_pyscf.json`) |
| RI-JK (`jk df`) | `*.df` | `compare_df_pyscf.py`, `compare_df_psi4.py`, `compare_hess_pyscf.py ... df` | `two_electron/df/gradient.rs`, `two_electron/df/hessian.rs` | the benchmark matrix (bench/README.md), `bench/ri_hess_scaling.py` |  | matrix (PySCF exact; Psi4 DF gradients explained) |
| RIJ + COSX (`jk rij_snk`) | none (the programs' seminumerical K grids differ) | | `crates/linus/tests/cosx_scf.rs`, `two_electron/cosx/gradient.rs`, `two_electron/cosx/rij_cosx.rs`, `crates/linus/tests/solvent_combinations.rs` (with PCM) | `bench/cosx_scf_bench.py` |  | finite differences and RI-JK agreement; ORCA RIJCOSX comparison in the bench results |
| range-separated hybrids, exact (wB97X, wB97X-D3BJ, CAM-B3LYP, wB97M) | `*.wb97x.*.exact*` `*.wb97x-d3bj.*.exact` `*.cam-b3lyp.*.exact` `nh3+.wb97x.*` | `compare_rsh_pyscf.py`, `compare_hess_pyscf.py` | `crates/linus/tests/range_separated.rs`, `derivatives/hessian.rs` `range_separated_matches_finite_differences` | `bench/rsh_bench.py`, `bench/rsh_hess_bench.sh` |  | vs PySCF exact and in the matrix; PySCF-only `nh3+.wb97x` |
| range-separated hybrids, RI (wK long-range fit) | `ch3sh.wb97x.def2-svp.df.wk`; `*.wb97x*.df` `*.cam-b3lyp.*.df` (other fits: `rsh-ri-fitting`) | `compare_rsh_pyscf.py --df` | `two_electron/df/hessian.rs` `range_separated_terms_match_finite_differences`, `crates/linus/tests/range_separated.rs` | `bench/rsh_hess_bench.sh` |  | exact vs PySCF patched to wK (PySCF-only case); other programs' fits differ by design |
| VV10 (wB97M-V, wB97X-V, B97M-V) | `*.wb97m-v.*` | `compare_vv10_pyscf.py`, `compare_vv10_psi4.py`, `reference/checks/pyscf_vv10_grad_fd.py` | `xc/vv10.rs`, `crates/linus/tests/solvent_combinations.rs` (with PCM) | `bench/vv10_bench.py` |  | energies and gradients vs PySCF and Psi4 (PySCF-only matrix cases); RKS Hessians vs PySCF (water wB97M-V, B97M-V: PySCF's VV10 Fock derivative patched to the fixed grid of its other terms, `run_ref.pyscf_vv10_hessian_fixed_grid`; PySCF has no UKS NLC Hessian); the CPKS kernel and Hessian terms against finite differences in `xc/vv10_resp.rs` |
| effective core potentials (def2, Rb-Rn) | `hi.*` `ch3i.*` `ch2i.*` `agcl.*` `ptcl4--.*` `au2*` `cscl.*` `pdcl2ph3.*` | `ecp_pyscf_check.py`, `ecp_psi4_check.py`, `ecp_scf_check.py`, `ecp_fd_grad.py` | `linus-ints` ECP tests, `crates/linus/tests/ecp_derivatives.rs` (RI and open-shell Hessians) | the benchmark matrix (pdcl2pme2ph2; bench/README.md) |  | matrix (Psi4 and ORCA ECP integral accuracy explained); RI Hessian vs PySCF (`ch3i.hf.def2-svp.df`) |
| PCM: C-PCM, IEF-PCM, COSMO, SS(V)PE (`pyscf` and `orca` presets) | `*.cpcm-*` `*.iefpcm-*` | `compare_pcm_pyscf.py` (all four models, `--grad`, `--hess`, `--df`), `compare_pcm_ecp_pyscf.py` | `solvent` unit tests, `crates/linus/tests/solvent_combinations.rs` | `bench/pcm_validate.py`, `bench/pcm_orca_compare.py`, `bench/pcm_psi4_compare.py`, `bench/pcm_bench.py` |  | vs PySCF exact (all models; COSMO and SS(V)PE in the deep checks only); `orca` preset vs ORCA 3e-8 Eh; Psi4 at the model level |
| SMD (CDS term, exact SASA) | `*.smd-*` | `compare_pcm_pyscf.py --model smd`, `reference/checks/pyscf_cds_grad_fd.py` | `solvent/smd.rs`, `solvent/smd_sasa.rs` |  |  | vs PySCF and ORCA; PySCF's CDS gradient is wrong for CH3SH (that README) |
| D3(BJ), D4 | `*-d3bj.*` `*-d4.*` `*.wb97x-d3bj.*` | | `crates/linus-disp/tests` (`reference.rs`, `derivatives.rs`) | | `reference/validation/results.json` (2590114) | matrix: dispersion energies vs s-dftd3/dftd4 2e-14 Eh; ORCA's D3 gradient explained |
| dipole derivatives, IR intensities, polarizability | (frequencies only) | | `derivatives/hessian.rs` `dipole_derivatives_match_finite_differences` and `dipole_derivatives_and_polarizability_are_stationary` (error quadratic in the CPHF residual), `properties/polarizability.rs` `polarizability_vs_finite_field`, `properties/dipole.rs` `apt_sum_rule`, `xc/vv10_resp.rs` (VV10, against finite differences of the dipole), `crates/linus/tests/solvent_combinations.rs` (with PCM) | | | finite differences and finite fields only (no reference program in the matrix) |
| harmonic analysis and thermochemistry (`linus::vib`) | the frequency rows (F) of every Hessian case | `crates/linus-py/tests/test_vib.py` | `vib/tests.rs` | `bench/vib/validate.py` |  | frequencies vs each program's own analysis of the same Hessian (orca_vib's constant factor explained); thermochemistry vs PySCF, Psi4 and ORCA (`bench/vib/validate.py`) |
| default grids by purpose (`linus:3` for Hessians with Treutler's adjustment for H-X pairs beyond neon, D26; the meta-GGA CPKS kernel on `linus:2`, D33) | none (the matrix shares PySCF's and Psi4's grids) | | `crates/linus/tests/grid_heavy_hydrides.rs` (HCl, HBr, H2S, CH3SH gradients on `linus:3` against `pyscf:8`), `crates/linus/tests/cpks_grid_meta_gga.rs` | | | Rust tests; grid errors in [docs/grids.md](../../docs/grids.md) |
| SCF robustness: two-cycle reaction, plateau rule, stability probe (D25, D29), multi-pair stability start (D32), escalation at `max_iter` (D30), zero-mode check of open-shell Hessians | none | | `crates/linus/tests/scf_robustness.rs`, `stability_probe_default.rs`, `hessian_mode_check.rs`, `openshell_scf.rs`; on the GPU `gpu_open_shell_loop.rs` (the device loop's decisions against the host loop's and the CPU's) | | | Rust tests and the robustness sets (bench/robust) |
| the GPU backend (`backend: "gpu"`) against the CPU backend | none | `reference/validate_gpu.py` (291 cases: energies, gradients, Hessians, IR, polarizabilities; HF, global, range-separated, meta-GGA and VV10 functionals; closed and open shells; PCM; the device loop; FP64, emulated and the defaults; [docs/gpu.md](../../docs/gpu.md#validating)) | `crates/linus/tests/gpu.rs`, `gpu_open_shell_loop.rs`, `gpu_memory_cap.rs`, the `linus-gpu` tests (need a device: `LINUS_GPU_REQUIRED=1`) | | | GPU vs CPU to round-off at the docs/gpu.md tolerances |

## The cases

Molecules (`tests/molecules`, `bench/molecules`, Å, converted once to bohr with CODATA 2022 and
passed to every program in bohr): water, NH3 (distorted), CH3SH (S), ClCNH+ (Cl; `hcn_cl.xyz`
has 31 electrons, so its closed shell is the cation), NH3+ doublet (open shell), and la18
(lactide, 18 atoms) as the medium case.

Effective core potentials (def2-ECP on Rb-Rn, Stuttgart-type semi-local, from the basis set
name in every program; PySCF `mol.ecp = basis`, Psi4 and ORCA by default): HI and CH3I (I,
ECP28MWB), the CH2I radical (open shell), AgCl, PtCl4²⁻ (ECP60MWB), Au2 and Au2+ (open shell),
CsCl (ECP46MWB) and trans-PdCl2(PH3)2 (11 atoms), HF/PBE/B3LYP, def2-SVP/TZVP, exact and RI.
Open shells with a degenerate singly occupied pi orbital (HI+, AgCl+) are not in the matrix:
on a DFT grid the hole's orientation about the axis is a soft mode, and no program converges it
quickly at tight thresholds (linus's DIIS froze there until the open-shell SCF work; it now
converges HBr+ and HI+ to max |F_ai| < 1e-9 in 35-59 builds, while PySCF stalls near 1e-8). Nuclear repulsion, the dipole's nuclear part and the electron count use
Z - ncore in every program.

Methods: HF; SVWN-RPA (Slater + VWN_RPA: the only SVWN Psi4 has, PySCF `slater,vwn3`, ORCA
`VWN3`, linus `svwn-rpa`); PBE (ORCA with libxc's mu and beta); B3LYP (VWN_RPA, ORCA
`B3LYP/G`); TPSS and r2SCAN (meta-GGA); B3LYP-D3(BJ) (two-body) and B3LYP-D4; the range-separated
hybrids wB97X, wB97X-D3BJ and CAM-B3LYP; wB97M-V (VV10 nonlocal correlation, PySCF-only cases
below). Closed shells RHF/RKS, NH3+ UHF/UKS. Basis sets def2-SVP and
def2-TZVP (spherical) and 6-31G* (Cartesian; ORCA has no Cartesian functions). J/K exact or RI
with def2-universal-jkfit in every program (ORCA `def2/JK`, the same functions).

Each case is run at its highest derivative: Hessians and harmonic frequencies for most, the
gradient for the def2-TZVP DFT cases other than CH3SH B3LYP, ClCNH+ with dispersion, r2SCAN
CH3SH and the la18 DFT cases (the reference DFT Hessians are the expensive part). Open-shell
(UHF/UKS) Hessians and frequencies are compared like the closed-shell ones. The case id is
`<molecule>.<method>[-<disp>].<basis>[-cart].<exact|df>[.<pcm model>-<solvent>][.wk]`.

**PySCF-only cases** (`refs="pyscf"` in `validate.py`; Psi4 and ORCA cells read `n/a`): features
whose Psi4 and ORCA counterparts this harness does not set up, and feature combinations. Every
cell is linus vs PySCF on the same grid at the exact tolerances.

| case | what it covers |
|---|---|
| `water.b3lyp.def2-svp.exact.cpcm-water` | C-PCM with a hybrid functional: energy, gradient (level 9), fixed-grid Hessian and frequencies (level 5) |
| `water.wb97x.def2-svp.exact.cpcm-water` | C-PCM with a range-separated hybrid, up to the Hessian |
| `ch3sh+.b3lyp.def2-svp.exact.iefpcm-water` | IEF-PCM, UKS doublet cation, gradient |
| `water.b3lyp.def2-svp.exact.smd-water` | SMD (electrostatics plus the CDS term on the exact SASA), gradient |
| `ch3i.hf.def2-svp.exact.cpcm-water` | C-PCM with an ECP atom (I, ECP28MWB), gradient |
| `nh3.wb97m-v.def2-svp.exact`, `nh3+.wb97m-v.def2-svp.exact` | VV10 (wB97M-V), RKS and UKS, gradients with grid response; VV10 on PySCF's level-3 `nlcgrids` in both |
| `nh3+.wb97x.def2-svp.exact` | range-separated UKS Hessian and frequencies |
| `ch3sh.wb97x.def2-svp.df.wk` | RI range-separated exchange against PySCF patched to the wK fit linus uses (`run_ref.py --wk`; energy and dipole: PySCF's DF gradient fits the long-range exchange with the erf metric) |
| `ch3i.hf.def2-svp.df` | ECP with an RI Hessian (PySCF's DF Hessian with full auxiliary-basis response) |

Implicit solvent is linus's `pyscf` PCM preset against PySCF's `pyscf.solvent.pcm`/`smd` (modified
Bondi radii x 1.2, SWIG, 302 Lebedev points per sphere, SMD 590 and the exact SASA: linus
`cds_contact_smoothing 0`), water's permittivity 78.3553 (PySCF's SMD table) in both. linus
against ORCA's and Psi4's solvent models: docs/solvation.md.

Frequencies: linus Hessian runs use `calctype: frequencies`; every reference Hessian goes
through its own program's harmonic analysis (PySCF `thermo.harmonic_analysis`, Psi4's qcdb
`vibanal`, ORCA `orca_vib`, via `bench/vib/refs.py`), all with the same masses (most abundant
isotopes) and translations and rotations projected, so frequency differences come from the
Hessians (plus orca_vib's constant factor, below). The geometries are not stationary points;
the projection is the same everywhere, which is what matters here.

### Runs per case

| run | what |
|---|---|
| `linus`, `pyscf`, `psi4`, `orca` | HF: no grid, same settings everywhere |
| `linus@pyscf:9`, `pyscf@pyscf:9` | DFT energies, dipoles, gradients on PySCF's level-9 grid (Treutler radii, NWChem pruning, Becke partitioning, 200 radial shells x 1454 Lebedev points before pruning), which linus reproduces point for point; PySCF `small_rho_cutoff 0`. Gradients with grid response |
| `linus@pyscf:5`, `pyscf@pyscf:5` | DFT Hessians on PySCF's level-5 grid (fixed grid: PySCF has no Hessian grid response). PySCF's XC Hessian takes ~3 min for water and over 15 min for CH3SH on level 9, so the Hessians use level 5 (5x fewer points) |
| `linus@psi4`, `psi4@psi4` | DFT on Psi4's default grid (75 x 302, unpruned), which linus reproduces including Psi4's standard orientation; Psi4 with `dft_remove_distant_points false`. Gradients without grid response (Psi4 has none). Hessians: Psi4 analytic for HF and LDA (fixed grid); finite differences of gradients for GGA and meta-GGA |
| `linus-fd@psi4` | the Hessian Psi4 reports for GGAs, made with linus: 5-point differences (0.005 bohr) of linus gradients on Psi4's grid, symmetrized, translations projected as Psi4 does (see below) |
| `psi4@fine` | Psi4 on 150 x 974 (unpruned), energies and gradients: the third program on a large grid |
| `orca@fine` | ORCA on its largest grid (`IntAcc 8`, Lebedev-770, unpruned, H not reduced); analytic Hessian with `DefGrid3` for the CP-SCF grid (exact J/K), `NumFreq` with RI-JK |
| `orca-numfreq@fine` | ORCA `NumFreq` (central differences of analytic gradients, 0.002 bohr) |
| `orca-numgrad@fine` | ORCA `NumGrad` (central differences of energies, 0.001 bohr), D3(BJ) only |
| `linus-hgr@pyscf:5` | linus Hessian with `grid.response: true` (the run option `hess_grid_response`; first-order grid terms: ORCA's analytic convention, Bykov et al. 2015) |

Settings: SCF converged to dE < 1e-12 Eh and orbital gradient 1e-10 everywhere (linus preset
`extreme`; PySCF `conv_tol 1e-12`, `conv_tol_grad 1e-9`; Psi4 `e/d_convergence 1e-12/1e-10`;
ORCA `TolE 1e-12`, `TolRMSP 1e-10` and more, see `../README.md`); integral thresholds 1e-14;
CPHF: PySCF `conv_tol_cpscf 1e-10` (x atoms per block; `Hessian.conv_tol` is not the CPHF
tolerance), Psi4 `solver_convergence 1e-10`, ORCA `Z_Tol 1e-10`, linus `response.tol 1e-9` (the environment variable `LINUS_CPHF_TOL` before a0f30e4).

## The matrix columns and tolerances

| column | pair | DFT grid |
|---|---|---|
| L-P | linus - PySCF | same grid |
| L-S | linus - Psi4 | same grid |
| L-O | linus - ORCA | different grids: passes when \|L-O\| <= tol + \|P-O\| ("grid" column) |
| P-S, P-O, S-O | references among themselves | PySCF level 9, Psi4 150 x 974, ORCA finest |

Max abs difference over all components; units Eh, e bohr (dipole about the origin), Eh/bohr,
Eh/bohr², cm⁻¹. For DFT Hessians the ORCA column is `NumFreq`. Tolerances (`validate.py` `TOL`;
"grid" tolerances and all energy tolerances scale with the number of atoms / 3):

| kind | E | dipole | gradient | Hessian | frequency | used for |
|---|---|---|---|---|---|---|
| exact | 1e-9 | 3e-8 | 3e-8 | 1e-7 | 0.01 | same model, same integrals, same grid |
| fd5 | 1e-9 | 3e-8 | 3e-8 | 1e-6 | 0.05 | Psi4 FD Hessian vs the same differences of linus gradients |
| fd2 | 1e-9 | 3e-8 | 3e-6 | 2e-5 | 0.5 | ORCA `NumGrad` (0.001 bohr) / `NumFreq` (0.002 bohr: 1.4e-6 from ORCA's analytic HF Hessian; ORCA's default 0.005 gives 8e-6) |
| grid | 1e-7 | 1e-6 | 3e-6 | 1e-4 | 2 | the same model on different grids (references among themselves) |

## Results (2026-11-09, after the release-readiness work)

The committed `results.json`: linus 4f3be67 (feature-complete, after the design audit, the release-readiness restructure and fixes); PySCF
2.15.0 (libxc 7.1.2), Psi4 1.11, ORCA 6.0.0; 89 cases.
**1960 cells within tolerance, 309
explained, 0 unexplained, 262 not available** (the PySCF-only cases have no Psi4 or ORCA cells;
ORCA has no Cartesian basis sets and no range-separated RIJK gradient; Psi4's GGA frequencies are
a finite-difference convention). Every linus feature is compared now, including the
range-separated hybrids, VV10, implicit solvent and the open-shell Hessians that were `n/i` in
the 2026-11-07 run. Of the explained cells that involve linus, all but one group are reference-side
(Psi4's and ORCA's ECP integrals, basis-data digits, ORCA's frequency factor, Psi4's
finite-difference and DF-gradient conventions, ORCA converging to another state); the exception is
`rsh-ri-fitting` (15 cells), where each program fits range-separated exchange its own way by
design. No linus bug was found: on matched grids linus agrees with PySCF and Psi4 to the SCF and
CPHF thresholds, and with ORCA within the grid residual that PySCF shows too.

The worst cell per method, J/K and shell (number of cases in brackets; `*` explained, `n/a` the
program cannot do it here, `-` not computed); `reference/validation_summary.py` prints every
case (`--compact --markdown` prints this table):

| method | J/K | shell | q | L-P | L-S | L-O | P-S | P-O | S-O |
|---|---|---|---|---|---|---|---|---|---|
| hf (15) | exact |  | E | 4.5e-12 | 3.7e-07* | 3.6e-08* | 3.7e-07* | 3.6e-08* | 3.8e-07* |
|  |  |  | D | 9.5e-10 | 8.8e-07* | 1.5e-08 | 8.8e-07* | 1.5e-08 | 8.7e-07* |
|  |  |  | G | 2.2e-10 | 1.0e-05* | 2.2e-08 | 1.0e-05* | 2.2e-08 | 1.0e-05* |
|  |  |  | H | 5.0e-09 | 2.7e-07* | 7.1e-06* | 2.7e-07* | 7.1e-06* | 6.9e-06* |
|  |  |  | F | 4.9e-06 | 4.9e-04 | 7.4e-02* | 4.9e-04 | 7.4e-02* | 7.4e-02* |
| hf (2) | exact cart |  | E | 1.3e-07* | 1.3e-07* | - | 8.5e-13 | - | - |
|  |  |  | D | 1.6e-07* | 1.6e-07* | - | 1.6e-10 | - | - |
|  |  |  | G | 1.1e-08 | 1.1e-08 | - | 6.2e-11 | - | - |
|  |  |  | H | 1.2e-08 | 1.2e-08 | - | 1.7e-10 | - | - |
|  |  |  | F | 9.0e-05 | 9.0e-05 | - | 4.4e-06 | - | - |
| hf (3) | exact | UKS/UHF | E | 2.0e-12 | 3.3e-07* | 7.7e-02* | 3.3e-07* | 7.7e-02* | 7.7e-02* |
|  |  |  | D | 3.3e-10 | 1.6e-10 | 1.1e-09 | 4.9e-10 | 7.4e-10 | 1.2e-09 |
|  |  |  | G | 1.6e-10 | 1.0e-05* | 6.3e-02* | 1.0e-05* | 6.3e-02* | 6.3e-02* |
|  |  |  | H | 3.8e-09 | 1.4e-10 | 6.7e-10 | 3.8e-09 | 3.8e-09 | 6.6e-10 |
|  |  |  | F | 8.3e-06 | 9.7e-07 | 6.7e-02* | 8.6e-06 | 6.7e-02* | 6.7e-02* |
| hf (5) | df |  | E | 4.0e-12 | 1.0e-09 | 7.8e-10 | 1.0e-09 | 7.8e-10 | 1.8e-09 |
|  |  |  | D | 3.9e-10 | 1.6e-09 | 1.5e-08 | 1.2e-09 | 1.5e-08 | 1.5e-08 |
|  |  |  | G | 2.2e-10 | 2.9e-07* | 2.1e-09 | 2.9e-07* | 2.1e-09 | 2.9e-07* |
|  |  |  | H | 1.1e-08 | 4.4e-07* | 1.0e-05 | 4.4e-07* | 1.0e-05 | 1.0e-05 |
|  |  |  | F | 1.4e-05 | 1.6e-03 | 7.2e-02 | 1.6e-03 | 7.2e-02 | 7.2e-02 |
| hf (1) | df | UKS/UHF | E | 4.3e-14 | 1.1e-11 | 8.9e-11 | 1.1e-11 | 8.8e-11 | 1.0e-10 |
|  |  |  | D | 3.0e-10 | 1.4e-10 | 5.2e-10 | 1.6e-10 | 3.7e-10 | 3.8e-10 |
|  |  |  | G | 1.6e-10 | 8.0e-10 | 7.8e-10 | 8.2e-10 | 7.8e-10 | 1.3e-09 |
|  |  |  | H | 1.8e-09 | 1.1e-09 | 1.9e-06 | 1.8e-09 | 1.9e-06 | 1.9e-06 |
|  |  |  | F | 8.3e-06 | 4.8e-06 | 6.5e-02 | 4.1e-06 | 6.5e-02 | 6.5e-02 |
| svwn-rpa (4) | exact |  | E | 1.3e-12 | 2.3e-13 | 2.8e-08 | 8.6e-09 | 2.8e-08 | 2.9e-08 |
|  |  |  | D | 1.9e-09 | 2.9e-09 | 3.2e-07 | 1.5e-08 | 3.2e-07 | 3.1e-07 |
|  |  |  | G | 9.6e-10 | 6.4e-10 | 1.6e-06 | 1.5e-07 | 1.6e-06 | 1.5e-06 |
|  |  |  | H | 6.0e-10 | 4.6e-10 | 4.0e-05 | - | 4.0e-05 | - |
|  |  |  | F | 4.2e-06 | 6.8e-07 | 9.2e-02 | - | 9.2e-02 | - |
| pbe (7) | exact |  | E | 3.2e-12 | 4.6e-08* | 2.3e-06 | 6.9e-07* | 2.3e-06* | 3.0e-06* |
|  |  |  | D | 4.3e-09 | 7.2e-07* | 5.4e-07 | 8.8e-07 | 5.4e-07 | 8.5e-07 |
|  |  |  | G | 9.6e-10 | 3.5e-08* | 2.1e-06 | 1.1e-05* | 2.1e-06 | 1.1e-05* |
|  |  |  | H | 5.6e-10 | 6.5e-04* | 4.4e-05 | - | 4.4e-05 | - |
|  |  |  | F | 4.7e-06 | - | 1.2e-01 | - | 1.2e-01 | - |
| b3lyp (13) | exact |  | E | 6.1e-12 | 4.1e-08* | 1.2e-06 | 1.7e-07* | 1.2e-06* | 1.0e-06* |
|  |  |  | D | 4.4e-09 | 8.4e-07* | 1.4e-05 | 2.4e-06* | 1.4e-05* | 1.7e-05* |
|  |  |  | G | 5.5e-10 | 4.2e-08* | 3.4e-06 | 4.1e-06* | 3.4e-06 | 4.1e-06* |
|  |  |  | H | 5.3e-10 | 6.2e-04* | 3.1e-02 | - | 3.1e-02* | - |
|  |  |  | F | 4.6e-06 | - | 6.3e-01 | - | 6.3e-01 | - |
| tpss (5) | exact |  | E | 1.9e-12 | 8.5e-13 | 2.7e-06 | 4.6e-08 | 2.7e-06* | 2.7e-06* |
|  |  |  | D | 1.1e-08 | 1.8e-09 | 1.7e-06 | 1.3e-06 | 1.6e-06* | 1.4e-06* |
|  |  |  | G | 1.2e-10 | 8.7e-10 | 2.0e-06 | 1.6e-06 | 2.0e-06 | 2.5e-06 |
|  |  |  | H | 5.3e-10 | 8.6e-04* | 8.0e-04 | - | 8.0e-04* | - |
|  |  |  | F | 1.3e-05 | - | 4.4e+01 | - | 4.4e+01* | - |
| r2scan (2) | exact |  | E | 1.4e-12 | 1.1e-13 | 1.3e-06 | 1.1e-06* | 1.3e-06* | 1.7e-07 |
|  |  |  | D | 1.8e-09 | 1.0e-09 | 3.7e-06 | 1.2e-05* | 3.7e-06* | 1.4e-05* |
|  |  |  | G | 5.8e-11 | 4.8e-11 | 2.1e-06 | 5.9e-05* | 2.1e-06 | 5.6e-05* |
|  |  |  | H | 5.0e-11 | 3.8e-08 | 2.4e-02 | - | 2.4e-02* | - |
|  |  |  | F | 4.1e-06 | - | 4.4e-01 | - | 4.4e-01 | - |
| b3lyp (2) | exact cart |  | E | 1.6e-06* | 1.6e-06* | - | 7.4e-09 | - | - |
|  |  |  | D | 2.0e-07* | 2.0e-07* | - | 1.9e-08 | - | - |
|  |  |  | G | 5.0e-09 | 5.1e-09 | - | 3.8e-08 | - | - |
|  |  |  | H | 1.1e-08 | 1.2e-08 | - | - | - | - |
|  |  |  | F | 8.2e-05 | - | - | - | - | - |
| svwn-rpa (1) | exact | UKS/UHF | E | 7.1e-14 | 1.6e-13 | 2.4e-09 | 3.1e-09 | 2.4e-09 | 6.9e-10 |
|  |  |  | D | 6.6e-10 | 9.6e-11 | 5.2e-09 | 3.6e-09 | 5.2e-09 | 7.1e-09 |
|  |  |  | G | 1.3e-10 | 7.4e-12 | 5.6e-08 | 1.4e-08 | 5.6e-08 | 5.2e-08 |
|  |  |  | H | 6.2e-11 | 1.9e-11 | 1.6e-05 | - | 1.6e-05 | - |
|  |  |  | F | 3.6e-06 | 2.3e-07 | 6.6e-02 | - | 6.6e-02 | - |
| pbe (1) | exact | UKS/UHF | E | 2.3e-13 | 1.8e-13 | 2.5e-08 | 3.1e-08 | 2.5e-08 | 6.3e-09 |
|  |  |  | D | 8.3e-10 | 3.2e-10 | 1.9e-07 | 1.8e-07 | 2.0e-07 | 1.3e-08 |
|  |  |  | G | 2.3e-10 | 1.3e-11 | 1.5e-07 | 4.1e-08 | 1.5e-07 | 1.6e-07 |
|  |  |  | H | 1.1e-10 | 5.5e-04* | 3.0e-05 | - | 3.0e-05 | - |
|  |  |  | F | 4.1e-06 | - | 6.4e-02 | - | 6.4e-02 | - |
| b3lyp (4) | exact | UKS/UHF | E | 2.7e-12 | 3.0e-07* | 2.1e-02 | 2.6e-07* | 2.1e-02* | 2.1e-02* |
|  |  |  | D | 3.1e-09 | 6.1e-07* | 6.8e-07 | 4.2e-07 | 6.8e-07 | 2.6e-07 |
|  |  |  | G | 2.2e-10 | 9.8e-06* | 4.2e-02 | 9.9e-06* | 4.2e-02* | 4.2e-02* |
|  |  |  | H | 6.9e-10 | 5.2e-04* | 2.4e-05 | - | 2.4e-05 | - |
|  |  |  | F | 7.3e-06 | - | 6.4e-02 | - | 6.4e-02 | - |
| tpss (1) | exact | UKS/UHF | E | 7.1e-14 | 1.7e-13 | 1.0e-06 | 2.6e-07* | 1.0e-06* | 7.4e-07* |
|  |  |  | D | 2.9e-10 | 3.9e-10 | 3.7e-06 | 3.0e-06* | 3.7e-06* | 6.5e-07 |
|  |  |  | G | 2.8e-11 | 3.2e-11 | 1.9e-06 | 7.8e-07 | 1.9e-06 | 1.1e-06 |
|  |  |  | H | 1.8e-11 | 2.9e-04* | 1.4e-04 | - | 1.4e-04* | - |
|  |  |  | F | 3.6e-06 | - | 5.8e-01 | - | 5.8e-01 | - |
| b3lyp (6) | df |  | E | 2.9e-12 | 1.1e-07* | 1.1e-06 | 9.9e-07* | 1.1e-06* | 2.1e-06* |
|  |  |  | D | 2.1e-09 | 7.4e-07* | 4.5e-06 | 9.6e-07 | 4.5e-06 | 4.7e-06 |
|  |  |  | G | 4.1e-10 | 2.5e-07* | 3.5e-06 | 2.9e-06 | 3.5e-06 | 3.4e-06 |
|  |  |  | H | 2.3e-09 | 6.4e-06* | 2.7e-05 | - | 2.7e-05 | - |
|  |  |  | F | 4.6e-06 | - | 1.4e-01 | - | 1.4e-01 | - |
| pbe (1) | df |  | E | 2.3e-13 | 1.1e-13 | 1.3e-08 | 1.7e-09 | 1.3e-08 | 1.4e-08 |
|  |  |  | D | 1.6e-09 | 1.6e-09 | 5.4e-07 | 2.7e-08 | 5.4e-07 | 5.2e-07 |
|  |  |  | G | 1.2e-10 | 3.1e-09 | 1.7e-06 | 1.2e-07 | 1.7e-06 | 1.8e-06 |
|  |  |  | H | 1.7e-09 | 9.0e-06* | 1.3e-05 | - | 1.3e-05 | - |
|  |  |  | F | 3.3e-06 | - | 1.2e-01 | - | 1.2e-01 | - |
| b3lyp (1) | df | UKS/UHF | E | 1.4e-13 | 1.2e-11 | 1.5e-08 | 1.2e-08 | 1.5e-08 | 3.0e-09 |
|  |  |  | D | 7.5e-11 | 6.9e-11 | 7.6e-08 | 6.7e-08 | 7.6e-08 | 1.1e-08 |
|  |  |  | G | 1.2e-10 | 3.7e-09 | 4.3e-08 | 2.4e-08 | 4.3e-08 | 5.0e-08 |
|  |  |  | H | 6.8e-10 | 5.2e-04* | 2.4e-05 | - | 2.4e-05 | - |
|  |  |  | F | 7.2e-06 | - | 6.4e-02 | - | 6.4e-02 | - |
| b3lyp-d3bj (2) | exact |  | E | 1.9e-12 | 8.0e-13 | 6.2e-08 | 1.1e-08 | 6.2e-08 | 5.2e-08 |
|  |  |  | D | 4.0e-09 | 2.5e-09 | 2.5e-07 | 1.7e-08 | 2.4e-07 | 2.3e-07 |
|  |  |  | G | 5.5e-10 | 1.5e-10 | 1.4e-06 | 1.4e-07 | 1.4e-06 | 1.3e-06 |
|  |  |  | H | 5.0e-10 | 2.7e-09 | 6.1e-05 | - | 6.1e-05 | - |
|  |  |  | F | 3.2e-06 | - | 1.4e-01 | - | 1.4e-01 | - |
| b3lyp-d4 (2) | exact |  | E | 2.6e-12 | 9.1e-13 | 6.2e-08 | 1.1e-08 | 6.2e-08 | 5.2e-08 |
|  |  |  | D | 4.0e-09 | 2.5e-09 | 2.5e-07 | 1.7e-08 | 2.4e-07 | 2.3e-07 |
|  |  |  | G | 5.5e-10 | 1.5e-10 | 1.3e-06 | 1.4e-07 | 1.3e-06 | 1.2e-06 |
|  |  |  | H | 5.2e-10 | 5.2e-09 | 1.2e-05 | - | 1.2e-05 | - |
|  |  |  | F | 3.2e-06 | - | 1.4e-01 | - | 1.4e-01 | - |
| wb97x (2) | exact |  | E | 1.1e-13 | 8.5e-14 | 3.0e-06 | 1.0e-06* | 3.0e-06* | 4.0e-06* |
|  |  |  | D | 2.0e-10 | 1.3e-12 | 2.7e-06 | 1.4e-06* | 2.7e-06* | 1.3e-06* |
|  |  |  | G | 9.1e-11 | 1.4e-11 | 9.7e-06 | 8.0e-06* | 9.7e-06* | 2.9e-06 |
|  |  |  | H | 1.2e-10 | 7.9e-09 | 8.5e-04 | - | 8.5e-04* | - |
|  |  |  | F | 3.9e-06 | - | 5.6e+00 | - | 5.6e+00* | - |
| wb97x (2) | df |  | E | 8.1e-06* | 9.5e-09* | 2.2e-05 | 8.1e-06* | 3.0e-05* | 2.2e-05* |
|  |  |  | D | 1.3e-05* | 7.5e-10 | 1.3e-05 | 1.4e-05* | 2.7e-05* | 1.3e-05* |
|  |  |  | G | 6.9e-06* | 1.0e-08 | 2.1e-05 | 6.9e-06* | 2.2e-05* | 2.1e-05* |
|  |  |  | H | 8.5e-05* | 9.5e-06* | - | - | - | - |
|  |  |  | F | 2.6e-01* | - | - | - | - | - |
| wb97x-d3bj (1) | exact |  | E | 9.9e-14 | 1.4e-14 | 3.6e-06 | 5.0e-09 | 3.6e-06* | 3.6e-06* |
|  |  |  | D | 1.0e-10 | 8.9e-11 | 2.2e-07 | 5.8e-09 | 2.2e-07 | 2.2e-07 |
|  |  |  | G | 4.6e-11 | 4.7e-12 | 7.2e-08 | 2.1e-08 | 7.2e-08 | 9.3e-08 |
| wb97x-d3bj (1) | df |  | E | 7.9e-06* | 9.4e-09* | 1.9e-05 | 7.9e-06* | 2.7e-05* | 1.9e-05* |
|  |  |  | D | 1.3e-05* | 1.2e-09 | 1.2e-05 | 1.3e-05* | 2.6e-05* | 1.2e-05* |
|  |  |  | G | 6.8e-06* | 1.2e-08 | 1.6e-05 | 6.8e-06* | 1.8e-05* | 1.6e-05* |
| cam-b3lyp (1) | exact |  | E | 2.8e-14 | 1.1e-13 | 2.5e-09 | 1.9e-09 | 2.5e-09 | 4.4e-09 |
|  |  |  | D | 9.1e-11 | 8.6e-12 | 3.6e-09 | 6.2e-10 | 3.7e-09 | 3.1e-09 |
|  |  |  | G | 1.7e-11 | 3.0e-11 | 1.4e-07 | 5.4e-09 | 1.4e-07 | 1.4e-07 |
| cam-b3lyp (1) | df |  | E | 1.9e-06* | 5.6e-09* | 1.7e-06 | 1.9e-06* | 3.6e-06* | 1.7e-06* |
|  |  |  | D | 5.2e-06* | 5.8e-10 | 5.7e-06 | 5.2e-06* | 1.1e-05* | 5.7e-06* |
|  |  |  | G | 3.2e-06* | 1.4e-08 | 2.5e-05 | 3.1e-06 | 2.6e-05* | 2.5e-05* |
| wb97m-v (1) | exact |  | E | 8.5e-14 | n/a | n/a | n/a | n/a | n/a |
|  |  |  | D | 1.1e-10 | n/a | n/a | n/a | n/a | n/a |
|  |  |  | G | 6.2e-10 | n/a | n/a | n/a | n/a | n/a |
| wb97m-v (1) | exact | UKS/UHF | E | 2.0e-13 | n/a | n/a | n/a | n/a | n/a |
|  |  |  | D | 4.4e-11 | n/a | n/a | n/a | n/a | n/a |
|  |  |  | G | 6.9e-10 | n/a | n/a | n/a | n/a | n/a |
| wb97x (1) | exact | UKS/UHF | E | 4.3e-14 | n/a | n/a | n/a | n/a | n/a |
|  |  |  | D | 9.6e-11 | n/a | n/a | n/a | n/a | n/a |
|  |  |  | G | 1.9e-10 | n/a | n/a | n/a | n/a | n/a |
|  |  |  | H | 3.0e-09 | n/a | n/a | - | n/a | - |
|  |  |  | F | 3.4e-05 | n/a | n/a | - | n/a | - |

## Explained differences

Each is detected by `explain()` in `validate.py` (results marked `*`), with the evidence below.
None of them is a linus difference.

1. **ORCA's analytic D3(BJ) gradient** is not the derivative of its D3 energy (known; linus-disp
   README). CH3SH B3LYP-D3(BJ): linus vs ORCA's analytic gradient 6.0e-6 Eh/bohr, vs ORCA
   `NumGrad` within the grid residual. The matrix column uses `NumGrad`; ORCA's `NumFreq` and
   analytic Hessians inherit the error. The dispersion energies themselves agree: linus vs
   PySCF and Psi4 (s-dftd3 / dftd4 libraries) 2e-14 Eh, vs ORCA 1.2e-10 Eh.
2. **ORCA's analytic Hessian of non-hybrid functionals with exact Coulomb** (`NoRI`) is wrong
   by ~5e-2 Eh/bohr² (water PBE/def2-SVP: bend 1702 vs 1528 cm⁻¹ from `NumFreq`; SVWN, VWN5,
   HFS, BLYP, TPSS alike; hybrids and RI-J are fine, 6e-6). The matrix uses `NumFreq`; the
   extra row "linus(hess_grid_response) vs ORCA analytic" shows it.
3. **Psi4's finite-difference Hessians** (GGA, meta-GGA): Psi4 projects translations out in
   mass-weighted coordinates even with `fd_project false`, and its grid turns with the
   molecule's standard orientation at every displaced geometry. linus's analytic Hessian with
   `hess_grid_response` (grid moving with the atoms by translation only) differs from Psi4's by
   7.5e-5 Eh/bohr² for water B3LYP on Psi4's default grid; the same finite differences of linus
   gradients, projected the same way (`linus-fd@psi4`), reproduce Psi4 to 4e-9. The L-S column
   compares the latter (it tests linus's gradients on Psi4's grid at every displaced geometry);
   linus's analytic Hessians are tested exactly against PySCF (L-P).
4. **6-31G* basis data**: PySCF's and Psi4's built-in 6-31G* coefficients have 7-8 digits, the
   Basis Set Exchange data linus reads 10 (O 1s: 0.0018311 vs 0.00183107443). Water HF: 7.1e-9
   Eh, dipole 4e-8 e bohr. PySCF given linus's basis data (`pyscf-bse`) agrees with linus to
   1e-13 Eh.
5. **wB97X is grid sensitive, and ORCA's differs.** On the large grids the references differ
   by 1e-6 to 4e-6 Eh (PySCF-Psi4 1.0e-6, PySCF-ORCA 3.0e-6, Psi4-ORCA 4.0e-6; gradients up
   to 1e-5 Eh/bohr). PySCF vs Psi4 is the grid: PySCF's pruned level-7 and level-9 grids agree
   with each other to 4e-9 but sit 1.0e-6 from unpruned 150x974, on which PySCF and Psi4
   agree to 6e-10 (wB97: 7.5e-7 and 8e-10; B3LYP: 2e-9 on any of these grids; CAM-B3LYP
   2e-12, so not the range-separated exchange). The B97-type semilocal part is unusually
   sensitive to the NWChem angular pruning. ORCA's own wB97X is 4.0e-6 Eh above libxc's on
   unpruned grids, unchanged at IntAcc 8, 10 and 12 (B3LYP/G 8e-9): an implementation
   difference. Hessians follow: PySCF's fixed-grid wB97X Hessian vs central differences of
   its own gradients is 8.6e-4 Eh/bohr² on the pruned level-5 grid but 8.3e-5 on unpruned
   100x590 (B3LYP 2.8e-5 and 2.5e-5), i.e. large grid-response terms, and Psi4 FD / ORCA
   NumFreq differ from it by 8.5e-4. The same-grid columns (L-P, L-S) of linus's wB97X stay
   exact tests (`explain()` never applies this to a pair with linus); the cross-grid ones
   need the explanation.
6. **ORCA's TPSS** is 8.5e-7 Eh above libxc's (water: ORCA vs Psi4 150x974 +8.51e-7 at IntAcc 8,
   +8.46e-7 at IntAcc 10; `CBetaPBE`/`XMuePBE` do not change it), while PySCF and Psi4 agree
   to 4e-9: an ORCA implementation difference. It also shows in the level-5 Hessian vs
   `NumFreq` (1.1e-4).
7. **orca_vib's wavenumbers are a constant factor below the others'** for the same Hessian
   and masses: 0.99998181 (found by bench/vib; 0.07 cm⁻¹ at 4000 cm⁻¹). Water HF: linus,
   PySCF and Psi4 vs ORCA 7.4e-2 cm⁻¹, 1e-6 cm⁻¹ after removing the factor. The comparison
   records the factor (`orca_frequency_ratio`) and the remainder (`max_abs_after_ratio`).
8. **Fixed-grid vs grid-response Hessians** (`grid-response-convention`): the matrix's ORCA
   Hessian for DFT is `NumFreq` (differences of gradients with weight derivatives), PySCF's and
   linus's default analytic Hessians keep the grid fixed. The difference is the grid response,
   small for B3LYP on these grids (2.7e-5 on level 5) but large for meta-GGAs: r2SCAN water on
   the pruned level-5 grid, fixed grid vs NumFreq 2.4e-2 Eh/bohr², while linus with
   `hess_grid_response` is 1.1e-4 from NumFreq. The explanation is applied only when the
   case's own "linus(hess_grid_response) vs ORCA NumFreq" row is within the grid tolerance.
   (Relevant to making grid response the default for DFT Hessians: with a fixed grid, r2SCAN
   Hessians on modest grids are far off.)
9. **Meta-GGA references on different grids** (`metagga-grid-sensitivity`): TPSS and especially
   r2SCAN are sensitive to PySCF's angular pruning (r2SCAN water gradients PySCF level 9 vs Psi4
   150x974: 1.5e-5 Eh/bohr; CH3SH 5.9e-5). linus agrees with each reference on its own grid.
10. **Psi4 DF gradients** (`psi4-df-gradient`): Psi4 1.11's density-fitted gradients differ
    from PySCF's, ORCA's and linus's (which agree with each other) by up to ~2e-8 Eh/bohr for
    CH3SH and ~3e-7 for la18, while DF energies agree to 1e-11; its FD Hessians amplify this to
    ~1e-5 Eh/bohr².
11. **NH3 FD Hessians on Psi4's grid** (`psi4-fd-near-symmetric-top`): NH3 is nearly a
    symmetric top, so the standard orientation that Psi4's grid follows is ill-conditioned and
    the gradient is not smooth in the geometry: 5-point differences of linus gradients with
    steps 0.005, 0.0025 and 0.01 bohr differ by 6.2e-4 and 2.4e-4 Eh/bohr² (water: 1e-8). linus
    and Psi4 gradients agree to 6e-11 at the reference and at displaced geometries; Psi4
    displaces along mass-weighted SALCs, linus-fd along Cartesians, so the FD Hessians differ
    by 6-9e-4.
12. **Range-separated hybrids with RI-JK** (`rsh-ri-fitting`): the three programs fit the
    long-range exchange differently with the same auxiliary basis. CAM-B3LYP CH3SH/def2-SVP:
    PySCF-Psi4 1.9e-6 Eh on the same unpruned grid (exact integrals: 6e-10), PySCF-ORCA 3.6e-6,
    Psi4-ORCA 1.7e-6 (exact: 2e-9, 4e-9). A linus RI result can match one program's fit at most.
    ORCA 6.0.0 also refuses range-separated RIJK gradients (energy and `NumGrad` instead).
13. **Psi4's ECP integrals** (`psi4-ecp-integrals`): Psi4 1.11 (libecpint) is less accurate
    than libcint (PySCF) and linus, which agree with each other to ~1e-14 relative. The
    generalized eigenvalues of (U_ECP, S) for the same def2 basis (invariant to function order,
    sign and normalization; `tests/python/ecp_psi4_check.py`) differ from both by 4e-9 (RbCl)
    to 1e-5 (Au2, PdCl2(PH3)2/def2-TZVP), relative ~4e-8, with identical ECP data (Psi4's def2
    gbs files match the BSE ones digit for digit). Psi4 energies of ECP molecules are therefore
    1e-8-4e-7 Eh from the others (HI HF/def2-SVP +2.3e-8), dipoles ~8e-7, gradients up to 1e-5
    (Au2); an analytic HF Hessian of Au2 from Psi4 was 1.6e-1 Eh/bohr² from PySCF's, ORCA's and linus's.
14. **ORCA's ECP accuracy** (`orca-ecp-integrals`): ORCA 6.0.0 differs from PySCF and linus by
    the same amount in both columns: HF energies HI 5e-11, CH3I/def2-TZVP 2e-10, Au2 1.3e-8,
    PdCl2(PH3)2 1.3e-8, PtCl4²⁻ 3.6e-8 Eh; analytic HF Hessians up to 1.4e-6 Eh/bohr². Explained
    up to 1e-7 Eh, 1e-6 (dipole, gradient), 1e-5 Eh/bohr² (Hessian).
15. **Au2+ in ORCA** (`orca-other-state`): ORCA's SCF from its own guess lands 0.077 Eh (UHF) and
    0.021 Eh (UKS B3LYP) above the state linus, PySCF and Psi4 agree on (2e-12 Eh, L-P).
16. **Heavy-atom DFT grids** (`ecp-heavy-atom-grids`): the references on different grids differ
    by up to ~2e-6 Eh, 1.4e-5 e bohr and 6e-6 Eh/bohr for CsCl, PtCl4²⁻ and PdCl2(PH3)2 (above
    the light-atom grid tolerance); linus matches each program on its own grid. Reference pairs
    only, up to 30x the grid tolerance.

## Findings about the reference programs (new here)

`reference/checks/orca_hessian_checks.py` reproduces the two ORCA Hessian findings (water,
~10 min):

```
A. ORCA 6.0.0 analytic Hessian vs NumFreq (0.002 bohr), run_ref settings, def2-SVP
method   J/K           max|dH| Eh/bohr^2   ORCA frequencies (cm-1), analytic | NumFreq
pbe      NoRI                   4.91e-02   [1702.22, 3988.56, 4053.14] | [1539.03, 3905.82, 4020.79]
pbe      RI-J def2/J            5.84e-06   [1539.05, 3904.06, 4018.73] | [1539.04, 3904.05, 4018.72]
svwn-rpa NoRI                   4.41e-02   [1655.53, 4008.44, 4066.76] | [1503.9, 3930.64, 4043.96]
tpss     NoRI                   4.90e-02   [1740.65, 3978.26, 4039.22] | [1581.93, 3895.47, 4007.28]
pbe0     NoRI NoCOSX            4.62e-06   [1618.5, 3927.06, 4034.57] | [1618.48, 3927.05, 4034.56]
b3lyp    NoRI NoCOSX            2.09e-06   [1603.22, 3905.88, 4009.09] | [1603.21, 3905.89, 4009.09]
hf       NoRI NoCOSX            1.36e-06   [1794.81, 3972.01, 4058.68] | [1794.8, 3972.02, 4058.69]
B. B3LYP/G: analytic vs NumFreq on the same main grid (IntAcc 8, Lebedev-770, unpruned, 164468 points)
  + DefGrid3 2.09e-06; no DefGrid word 1.78e-04; + DefGrid1 1.78e-04; + DefGrid2 1.78e-04
linus, PySCF level-9 grid:
  hess_grid_response vs 5-point differences of no-grid-response gradients: 9.32e-09
  linus fixed grid          vs ORCA NumFreq 4.69e-06, vs ORCA analytic+DefGrid3 5.67e-06, fine grid alone 1.74e-04
  linus hess_grid_response  vs ORCA NumFreq 4.20e-06, vs ORCA analytic+DefGrid3 5.18e-06, fine grid alone 1.74e-04
```

* **ORCA's analytic DFT Hessian solves the CP-SCF equations on a separate grid** that follows
  the `DefGrid` level and ignores `IntAcc`/`AngularGrid`: with the finest explicit grid
  settings alone, analytic vs `NumFreq` differ by 1.8e-4 Eh/bohr² (water B3LYP; DefGrid1/2
  the same), with `DefGrid3` added 2.1e-6. The earlier explanation, that ORCA keeps only
  first-order grid-weight derivatives (Bykov et al. 2015), does not account for it: linus's
  Hessian with exactly that approximation agrees with ORCA `NumFreq` to 4e-6 on PySCF's level-9
  grid. `run_ref.py` adds `DefGrid3` to ORCA analytic DFT Hessians.
* ORCA has no analytic Hessian with `RIJK` (`NumFreq` instead); its analytic `Freq` writes no
  gradient without `EnGrad`; its dipole is about the centre of mass (recovered exactly from
  `orca.property.txt`).
* ORCA `VWN3` = libxc VWN_RPA also for UKS (NH3+ SVWN-RPA, fine grids: 1.2e-9 Eh from PySCF).
* Psi4 1.11 analytic Hessians: HF and LDA only (RHF/UHF/RKS/UKS), also with `mem_df`.
* **PySCF 2.15's DFT gradient with grid response is wrong for ECP atoms**: `grad.rks.
  grids_response_becke` looks up the Bragg radii of the Becke-Treutler size adjustment with
  `mol.atom_charges()` (Z - ncore for ECP atoms), the energy grid
  (`radi.treutler_atomic_radii_adjust`) with the proton numbers, so the gradient is not the
  derivative of the energy: CsF B3LYP/def2-TZVP 4.9e-5 Eh/bohr from central differences of the
  energies (linus 3e-8, the difference truncation), PdCl2(PH3) B3LYP/def2-SVP 1.8e-5, IrH3 PBE 2.5e-6 (`tests/python/ecp_fd_grad.py`, `ecp_scf_check.py --grad [--fix-pyscf]`). With
  the proton numbers passed in (`run_ref.pyscf_ecp_grid_response_fix`, applied to every PySCF
  ECP gradient here) PySCF agrees with linus to 2e-10. Without grid response it always did.
* **PySCF (libcint) screens semi-local ECP integrals by overlap**: `check_3c_overlap` drops a
  shell pair when exp(-(a b AB² + a c AC² + b c BC²)/(a+b+c)) < e^-39, with the AB term, but a
  semi-local (projector) integral does not need a and b to overlap, only each to reach the core:
  functions on opposite sides of the ECP atom get exact zeros where the integral is up to 2.2e-7
  (HgH2/def2-QZVP H d-d across Hg; PtCl4²⁻/def2-QZVP 7e-9, PdCl2(PH3)/def2-TZVP 9e-11; an
  independent quadrature agrees with linus to 1e-20 there). Energies move by < 1e-11 Eh here.
  It is far larger with ligands on opposite sides of the ECP atom: linear XeF2 (bench/robust)
  loses elements up to 6.1e-5 (def2-SVP) and 2.9e-5 (def2-TZVP), and PySCF's HF energy moves by
  2.9e-9 and -1.6e-6 Eh, its gradients by 2-3e-6 Eh/bohr; FI···FMe/def2-TZVP (iodine) by -1.7e-9
  Eh. With linus's ECP matrix in place of its own, PySCF agrees with linus to 5e-13 Eh
  (`bench/robust/repro/pyscf_ecp_screening/check.py`). The cutoff is fixed in libcint
  (`PTR_EXPCUTOFF` does not change it).
  On the elements it keeps, PySCF's one-center integrals are 1e-12-3e-11 from closed forms
  (linus ~1e-13).
* PySCF: the CPHF tolerance is `mf.conv_tol_cpscf` (x atoms per block); at 1e-12 its Krylov
  solver stalls (la18: true residual 1.5e-7, slow per-root fallback) or hits a singular
  subspace (water SVWN), so the matrix uses 1e-10. `small_rho_cutoff` must be 0 to reproduce a
  grid. Its D3 Hessian is central differences of D3 gradients.

## How to add a row

1. Add a `Case(...)` to `_cases()` in `validate.py` (molecule from `MOLECULES`, method from
   `run_ref.FUNCTIONALS`, basis, `calc`, `jk`, `cart`, `disp`, `solvent`, `wk`, `refs`;
   `quick=True` puts it in the per-merge subset). A new molecule goes into `MOLECULES` (xyz path, charge, multiplicity).
   A new functional needs its PySCF / Psi4 / ORCA names in `run_ref.FUNCTIONALS`, and a
   check that they mean the same functional (compare the references with each other first).
2. `.refenv/bin/python reference/validate.py --cases '<id>'` runs the references once and
   linus; commit `runs/<id>/` and `results.json`.
3. If a cell is above tolerance, find out whose difference it is from the reference columns.
   A reference quirk goes into `EXPLAINED` and `explain()` in `validate.py`, with the evidence
   here and in `../README.md`; a linus difference is a bug report, never a tolerance change.

A feature the harness runs in PySCF only (see "PySCF-only cases") takes `refs="pyscf"` on its
`Case`; new model fields go into the case id and `spec_of()` only when set, so the stored runs of
the other cases keep their spec hashes.

**Former hooks, now filled.** The range-separated rows (wB97X, wB97X-D3BJ, CAM-B3LYP; PySCF
`wb97x-d3bj`/`camb3lyp`, Psi4 `wb97x-d3bj`/`cam-b3lyp`, ORCA `wB97X-D3BJ`/`CAM-B3LYP`, ORCA's
D3(BJ) gradient through `NumGrad`) and the open-shell Hessian rows (NH3+ UHF/UKS: PySCF analytic,
Psi4 analytic HF/LDA or FD, ORCA analytic and `NumFreq`) ran their references before linus had
these features; linus now computes them, and the committed `results.json` (2026-11-08) compares
them. Psi4's wK carries ~1e-9 Eh of fitting round-off and libcint's erfc integrals a
~1e-12 floor, so expect L-S near 1e-9 Eh for the range-separated rows; wB97X's grid sensitivity
(explanation 5) only affects the cross-grid columns. `run_linus` still falls back to the gradient
when linus answers "not implemented" for a Hessian.

**Per-merge regression check:** `.refenv/bin/python reference/validate.py --quick
[--linus-bin PATH]` reruns linus on the quick cases (22, every feature, ECPs and C-PCM included) against the
stored references (`--refs none` is implied), writes its linus runs and the quick cases'
`results.json` to `reference/out/validation/quick/` (git-ignored; nothing tracked changes, so it
is safe in any checkout), prints the matrix and exits 1 if any cell fails without explanation:
keep the exit code and read the `cells:` line. `--quick --store` writes into this directory
instead, as before (then `git diff reference/validation` shows what changed).

## Reference environment

`.refenv` (`../VERSIONS.md`) plus a pip `--target` overlay for dispersion, which PySCF and Psi4
do not ship: PySCF's `mf.disp` needs `pyscf-dispersion`, Psi4's `-d3bj`/`-d4` the `dftd3` and
`dftd4` Python packages (qcengine harnesses). The overlay is found next to `.refenv`
(`LINUS_REFENV_DISP` overrides) and is added to `sys.path`/`PYSCF_EXT_PATH` (PySCF) and
`PYTHONPATH` (the Psi4 subprocess) only for `--disp` runs:

```bash
.refenv/bin/python -m pip download --no-deps --only-binary=:all: -d /tmp/w \
    pyscf-dispersion==1.7.0 dftd3==1.7.0 dftd4==4.3.1 cffi==2.1.1 pycparser==3.12
.refenv/bin/python -m pip install --no-deps --no-index --target .refenv-disp /tmp/w/*.whl
```
