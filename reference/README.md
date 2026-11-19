# Reference programs for linus

PySCF, Psi4 and ORCA reference energies, gradients and Hessians for validating linus,
plus MIRP for arbitrary-precision integrals (`MIRP.md`). Versions: `VERSIONS.md`. The ORCA
results (ORCA 6.0.0) are included with permission of FAccTs GmbH
([THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md#orca-faccts-gmbh)).

```
reference/
  run_ref.py        run PySCF / Psi4 / ORCA on one molecule, one JSON per program
  compare_ref.py    pairwise differences between the programs' JSONs
  crosscheck.sh     the water/NH3 cross-check below
  validate.py       the validation matrix: linus vs all three (validation/README.md);
                    --quick (per merge) writes only to out/validation/quick/
  validation_summary.py   prints the matrix from validation/results.json
  validation/       stored reference and linus results, results.json, README
  molecules/        water.xyz, nh3.xyz (angstrom; asymmetric, off-centre, off-axis on purpose)
  checks/           small experiments behind the quirks below
  env/              lock files for .refenv
  out/              raw outputs (git-ignored)      src/  MIRP sources + data (git-ignored)
```

## Setup (once per machine)

```bash
# micromamba (static) and the env: VERSIONS.md has the exact commands
export MAMBA_ROOT_PREFIX=$HOME/micromamba
micromamba create -y -p .refenv --file reference/env/refenv-explicit.txt
.refenv/bin/python -m pip install -r reference/env/refenv-pip.txt
# ORCA 6.0.0 (licensed; outside the repo, never commit it): install it as ~/opt/orca-6.0.0
# or point LINUS_ORCA_DIR at it
```

`run_ref.py` finds them at `.refenv` and `~/opt/orca-6.0.0`; override with
`LINUS_REFENV` / `LINUS_ORCA_DIR`.

## Running the reference runner

Always with the env's Python (it needs PySCF, qccompute, qcref):

```bash
.refenv/bin/python reference/run_ref.py reference/molecules/water.xyz \
    --method hf --basis def2-svp --calc gradient --threads 8              # all three programs
.refenv/bin/python reference/run_ref.py reference/molecules/nh3.xyz \
    --method b3lyp --basis def2-svp --calc hessian --grid fine --programs pyscf,orca
.refenv/bin/python reference/run_ref.py mol.xyz --charge 1 --mult 2 --method pbe0 \
    --basis cc-pvtz --calc energy --approx df --aux cc-pvtz/jk --threads 16 --memory-gb 32
.refenv/bin/python reference/compare_ref.py reference/out/water/hf_def2-svp_gradient_exact
```

| Option | Meaning |
|---|---|
| `xyz` | geometry in Å; converted once to bohr with CODATA 2022 (qcref), and every program gets those bohr coordinates |
| `--charge`, `--mult` | closed shell: RHF/RKS; `mult > 1`: UHF/UKS in all three |
| `--method` | `hf`, `b3lyp` (VWN_RPA, = ORCA `B3LYP/G`), `b3lyp5` (VWN5, = ORCA `B3LYP`), `svwn-rpa`, `pbe`, `pbe0`, `blyp`, `tpss`, `r2scan`, `wb97x`, `wb97x-d3bj`, `cam-b3lyp`, and the VV10 functionals `wb97m-v`, `wb97x-v`, `b97m-v` (`run_ref.FUNCTIONALS`); the three programs' names are cross-checked against each other for hf, svwn-rpa, pbe, b3lyp and tpss (`VERIFIED_FUNCTIONALS`; the range-separated ones in `validation/README.md`) |
| `--basis` | name in each program's built-in library (`def2-svp`, `def2-tzvp`, `cc-pvtz`, ...) |
| `--calc` | `energy`, `gradient` or `hessian` |
| `--approx` | `exact` (default: no RI/DF/COSX), `df` (RI-JK; ORCA `RIJK`, Psi4 `MEM_DF`, PySCF `density_fit`), `rij` (RI-J + exact K; ORCA `RIJONX`/`RI`, Psi4 `DFDIRJ+LINK`, PySCF `only_dfj`), `rijcosx` (ORCA `RIJCOSX`, Psi4 `DFDIRJ+COSX`, PySCF sgX; the seminumerical K grids differ, so these never agree tightly) |
| `--aux` | `def2/j`, `def2/jk`, `cc-pv{d,t,q}z/jk`; default `def2/j` for rij/rijcosx, `def2/jk` (or `cc-pvXz/jk` for cc bases) for df |
| `--grid` | DFT only. `fine` (default): PySCF and Psi4 150 radial x 974 Lebedev, unpruned, Treutler radii/Becke-type partition; ORCA `IntAcc 8.0, AngularGrid 7`, `GridPruning 0`, `HGridReduced false`, `BFCut 1e-15` (AngularGrid 7 is Lebedev-770, ORCA 6.0.0's largest, but with `GridPruning 0` ORCA puts Lebedev-590 on every shell). `default`: each program's default grid. `defgrid3`: ORCA `DefGrid3`. `R,A`: PySCF/Psi4 unpruned R x A (ORCA falls back to `fine`). `pyscf:N`: PySCF's level-N grid with `small_rho_cutoff 0`, point for point what linus `--grid pyscf:N` builds (PySCF only). `psi4`: Psi4's default grid (75 x 302, unpruned) with `dft_remove_distant_points false`, what linus `--grid psi4` builds (Psi4 only) |
| `--cart` | Cartesian basis functions: PySCF `cart=True`, Psi4 `puream false`; ORCA has none (refused) |
| `--disp` | `d3bj` (two-body: PySCF `mf.disp="d3bj"`, Psi4 `-d3bj`, ORCA `D3BJ`) or `d4` (with ATM: PySCF `d4`, Psi4 `-d4`, ORCA `D4`). PySCF and Psi4 need the `.refenv-disp` overlay (`validation/README.md`) |
| `--threads`, `--memory-gb` | PySCF `lib.num_threads`; Psi4 `-n`; ORCA `%pal nprocs` with `%maxcore = memory / threads` |
| `--no-grid-response` | PySCF DFT gradients without grid-weight derivatives (default: with them; Psi4 never has them, ORCA's are translation-invariant) |
| `--psi4-scf-type` | exact-integral algorithm in Psi4: `pk` (default, in-core) or `direct` |
| `--psi4-hessian` | `auto` (Psi4's choice: analytic for HF and LDA, finite differences of gradients otherwise), `analytic`, `fd` |
| `--orca-native-pbe` | keep ORCA's own PBE constants (default: libxc's, see quirks) |
| `--orca-hessian` | `analytic` or `numfreq` (default: analytic for HF, `NumFreq` for DFT; RI-JK always `NumFreq`, quirk 19). `--orca-analytic-dft-hessian` is the old spelling of `analytic` |
| `--orca-fd-step`, `--orca-numgrad` | ORCA `NumFreq` increment / `NumGrad` step in bohr (default 0.005, ORCA's); gradient by `NumGrad` |
| `--pyscf-max-memory-mb` | PySCF `max_memory` (default `--memory-gb`) |
| `--pcm`, `--solvent`, `--eps`, `--pcm-lebedev` | implicit solvent, PySCF only (Psi4 and ORCA refuse): `cpcm`, `iefpcm`, `cosmo`, `ssvpe` with `--eps`, or `smd` with `--solvent` (PySCF's descriptors and eps); set up as linus's `pyscf` PCM preset (SWIG, 302 points per sphere, SMD 590) |
| `--nlc-level` | PySCF `nlcgrids` level of the VV10 functionals `wb97m-v`, `wb97x-v`, `b97m-v` (default 3, PySCF's own; linus `grid.nlc: pyscf:3`) |
| `--wk` | with `--approx df`, PySCF only, energies: range-separated exchange fitted by Psi4's wK scheme (linus's) instead of PySCF's erf metric (`wk_get_jk` in `tests/python/common.py`) |
| `--tag`, `--outdir`, `--rundir` | output directory label / location; `--rundir` writes straight into that directory |

Output: `reference/out/<molecule>/<method>_<basis>_<calc>_<approx>[-<aux>][_grid-<grid>][_<tag>]/`
with `pyscf.json`, `psi4.json`, `orca.json` and each program's raw files in `<program>_run/`
(`pyscf.log`; Psi4 `psi4.out`, `job.json`, `results.json`; ORCA `orca.inp`, `orca.out`,
`geometry.xyz`, `orca.engrad`, `orca.hess`; `qcoutput.json` = the qccompute ProgramOutput).
A program that fails gets `"success": false` with the traceback, and the others still run.

JSON fields: `energy` (Eh), `gradient` (natoms x 3, Eh/bohr), `hessian` (3N x 3N, Eh/bohr²,
full Cartesian, not projected), `nuclear_repulsion` (and `nuclear_repulsion_minus_runner`,
the check that the program used exactly our geometry), `scf_iterations`, `wall_time_s`
(around the whole program call, not benchmark quality), `nbasis`, `program_version`,
`settings` (every setting the runner chose, and for ORCA the values it printed back),
`native_input`, `hessian_method`, `gradient_sum` (translational invariance), `molecule`
(symbols, Å and bohr geometry, conversion factor, charge, multiplicity), `approximation`,
`dft_grid`, `functional_mapping`, `host`, `date`, `runner`.

### What each program is told (correctness defaults)

| | PySCF 2.15.0 (in-process) | Psi4 1.11 (qccodec `psi4`) | ORCA 6.0.0 (qccodec `orca`) |
|---|---|---|---|
| Geometry | `unit="Bohr"`, `symmetry=False` | codec driver: `units bohr`, `no_com`, `no_reorient`, `symmetry c1` | `! Bohrs` + bohr `geometry.xyz` (ORCA never reorients) |
| Basis | `cart=False` | `puream true` | spherical only |
| SCF convergence | `conv_tol 1e-12`, `conv_tol_grad 1e-9` (norm of the whole orbital gradient, about 10x the RMS) | `e_convergence 1e-12`, `d_convergence 1e-10`, `fail_on_maxiter` | `VeryTightSCF` plus `%scf TolE 1e-12 TolRMSP 1e-10 TolMaxP 1e-9 TolErr 1e-10 TolG 1e-9 ConvCheckMode 0 ConvForced true` |
| Integrals | `direct_scf_tol 1e-14` (ERIs in core when they fit) | `scf_type pk`, `ints_tolerance 1e-14` | `NoRI NoCOSX`, `%scf Thresh 1e-14 TCut 1e-15 DirectResetFreq 1` |
| Hessian | analytic, CPHF `mf.conv_tol_cpscf 1e-10` (x atoms per block; 1e-12 stalls its Krylov solver; `Hessian.conv_tol` is not the CPHF tolerance), no grid response for DFT | HF and LDA: analytic (`solver_convergence 1e-10`); GGA, meta-GGA: finite differences of gradients, `points 5`, `fd_project false` (translations are still projected, quirk 11) | HF: analytic (`Z_Tol 1e-10`, plus `EnGrad` for the gradient); DFT: `NumFreq` (central differences of analytic gradients) or analytic with `DefGrid3` (quirk 4) |
| Dipole | `mf.dip_moment(origin=0)` | `SCF DIPOLE` (origin 0; after an FD Hessian from a separate single point) | `orca.property.txt`, shifted from ORCA's centre-of-mass origin to 0 |

### Running the programs by hand

```bash
# PySCF: plain Python with the env
.refenv/bin/python -c "from pyscf import gto, scf; m = gto.M(atom='reference/molecules/water.xyz', basis='def2-svp'); print(scf.RHF(m).run().e_tot)"
# Psi4: the env's psi4 (the runner goes through qccompute -> `psi4 input.py -k -o stdout -n N --memory XGB`)
.refenv/bin/psi4 -n 8 input.dat
# ORCA: always by its full path (needed for %pal), with the env's Open MPI 4.1.6
ORCA=${LINUS_ORCA_DIR:-$HOME/opt/orca-6.0.0}
export PATH=$ORCA:$PWD/.refenv/bin:$PATH \
       LD_LIBRARY_PATH=$ORCA/lib:$PWD/.refenv/lib \
       OMPI_MCA_btl=self,vader OMPI_MCA_btl_vader_single_copy_mechanism=none \
       OMPI_MCA_hwloc_base_binding_policy=none OMPI_MCA_rmaps_base_oversubscribe=1
$ORCA/orca orca.inp > orca.out
grep -q 'ORCA TERMINATED NORMALLY' orca.out    # ORCA can exit 0 after an error termination
```

PySCF has no qc* codec, so the runner calls it in-process: a qc-suite gap.

## Cross-check: water and NH3

Run with `reference/crosscheck.sh 8` (PySCF 2.15.0, Psi4 1.11, ORCA 6.0.0; 8 threads or `%pal`
processes each, node load ~20; 2026-11-06). Exact integrals, tight SCF, input frame; DFT on the
`fine` grids (`--grid` above). Each cell is max |ΔE| (Eh) / max |ΔG| (Eh/bohr) /
max |ΔH| (Eh/bohr²) between the two programs. "-": ORCA's analytic HF `Freq` writes no
gradient. Wall times include program start-up and are not benchmarks.

| Molecule | Case | E (PySCF), Eh | pyscf−psi4 dE / dG / dH | pyscf−orca dE / dG / dH | psi4−orca dE / dG / dH | wall s (PySCF / Psi4 / ORCA) |
|---|---|---|---|---|---|---|
| water | HF/def2-SVP grad | -75.9594924060 | 1e-13 / 4e-11 | 9e-11 / 6e-10 | 9e-11 / 6e-10 | 0.3 / 0.8 / 1.3 |
| water | HF/cc-pVTZ grad | -76.0554713652 | 5e-13 / 1e-10 | 1e-10 / 8e-10 | 1e-10 / 8e-10 | 0.5 / 0.9 / 1.6 |
| water | HF/def2-TZVP grad | -76.0574284872 | 3e-13 / 1e-10 | 1e-10 / 1e-9 | 1e-10 / 1e-9 | 0.3 / 0.8 / 1.4 |
| water | HF/def2-SVP Hessian | -75.9594924060 | 0e+00 / 4e-11 / 8e-10 | 9e-11 / - / 2e-9 | 9e-11 / - / 2e-9 | 0.4 / 1.0 / 1.6 |
| water | B3LYP/def2-SVP grad, fine grid | -76.3577491424 | 4e-10 / 3e-8 | 1e-8 / 2e-7 | 1e-8 / 2e-7 | 2.0 / 1.4 / 8.0 |
| water | PBE/def2-SVP grad, fine grid (ORCA with libxc μ, β) | -76.2719438697 | 4e-10 / 1e-7 | 2e-8 / 4e-7 | 2e-8 / 4e-7 | 2.1 / 1.2 / 6.1 |
| water | B3LYP/def2-SVP Hessian, fine grid | -76.3577491424 | 4e-10 / 3e-8 / 3e-6 | 1e-8 / 2e-7 / 7e-6 | 1e-8 / 2e-7 / 6e-6 | 33.1 / 18.8 / 30.1 |
| nh3 | HF/def2-SVP grad | -56.1483921178 | 8e-14 / 1e-10 | 7e-11 / 9e-10 | 7e-11 / 9e-10 | 0.3 / 0.8 / 1.2 |
| nh3 | HF/cc-pVTZ grad | -56.2173879537 | 2e-13 / 2e-10 | 8e-11 / 1e-9 | 8e-11 / 1e-9 | 0.7 / 1.1 / 1.6 |
| nh3 | HF/def2-TZVP grad | -56.2178420413 | 2e-13 / 1e-10 | 7e-11 / 6e-10 | 7e-11 / 6e-10 | 0.5 / 0.9 / 1.3 |
| nh3 | HF/def2-SVP Hessian | -56.1483921178 | 4e-14 / 1e-10 / 6e-10 | 7e-11 / - / 1e-9 | 7e-11 / - / 8e-10 | 0.7 / 1.3 / 1.5 |
| nh3 | B3LYP/def2-SVP grad, fine grid | -56.5090237960 | 1e-9 / 7e-9 | 7e-9 / 1e-7 | 8e-9 / 1e-7 | 2.3 / 1.6 / 7.3 |
| nh3 | PBE/def2-SVP grad, fine grid (ORCA with libxc μ, β) | -56.4321853354 | 1e-9 / 3e-8 | 1e-8 / 2e-7 | 9e-9 / 2e-7 | 2.8 / 1.5 / 7.4 |
| nh3 | B3LYP/def2-SVP Hessian, fine grid | -56.5090237960 | 1e-9 / 7e-9 / 9e-7 | 7e-9 / 1e-7 / 7e-6 | 8e-9 / 1e-7 / 6e-6 | 8.7 / 29.8 / 37.9 |

**HF with exact integrals passes with a wide margin**: max |ΔE| =
1.2e-10 Eh (target 1e-8) and max |ΔG| = 1.1e-9 Eh/bohr (target 1e-7), cc-pVTZ f functions
included, and over 8 cases in all. PySCF and Psi4 agree to ~1e-13 Eh / 1e-10 Eh/bohr. ORCA
sits a steady 7e-11 to 1.2e-10 Eh above them (cause not isolated: not its SCF stopping point,
since the energy error is quadratic in a ~1e-9 density error). Its ~1e-9 Eh/bohr gradient
offset fits that density error (quirk 7). HF Hessians agree to 2e-9.

**DFT on the finest grids**: PySCF and Psi4 (both 150x974 unpruned) agree to ≤1.5e-9 Eh and
≤1.3e-7 Eh/bohr. Most of the gradient gap is Psi4's missing grid-weight derivatives: PySCF
without them matches Psi4 to 6e-8 (water) and 5e-9 (NH3). ORCA (Lebedev-770 max) is ~1e-8 Eh
and ~1-4e-7 Eh/bohr from both, at every IntAcc tried (quirk 5). DFT Hessians agree to 3e-6
(PySCF analytic vs Psi4 5-point FD) and 7e-6 (ORCA `NumFreq`). ORCA's analytic DFT Hessian
was 1.6e-4 off (quirk 4). For comparison, ORCA `DefGrid3` is 6e-7 Eh / 2e-6 Eh/bohr (water)
and 1e-7 / 3e-7 (NH3) from ORCA's fine grid. For linus, read DFT reference tolerances as
~1e-8 Eh and ~1e-7 Eh/bohr against PySCF on the same grid, not tighter.

**Approximations** (water HF/def2-SVP gradient, `--approx`): `df` with def2/JK, all three
agree to 1.1e-10 Eh / 7e-10 Eh/bohr (identical aux basis). `rij` with def2/J: energies agree
to 1e-10 Eh, PySCF/ORCA gradients to 5e-10, but Psi4's gradient is 1.7e-4 off (quirk 10, now
refused). `rijcosx`: 3-9e-6 Eh apart, as expected from different COSX/sgX grids.
UHF (H2O+ doublet, HF/def2-SVP): 1.0e-10 Eh / 1.4e-9 Eh/bohr.

## Quirks found while setting this up (with evidence)

Numbers are from this node, ORCA 6.0.0 / Psi4 1.11 / PySCF 2.15.0, water and NH3 in
`molecules/`, unless stated.

**ORCA**

1. **ORCA converts Å to bohr with CODATA 1998 (0.5291772083 Å/bohr), not CODATA 2022
   (0.529177210544).** Given the same Å geometry, ORCA's water Vnn is 9.07449214329 Eh
   vs 9.07449218177 Eh: 3.8e-8 Eh, which alone breaks a 1e-8 Vnn comparison. With
   `! Bohrs` (it also applies to `* xyzfile`) and bohr coordinates, ORCA's Vnn matches
   to 1e-14. The qccodec ORCA codec always writes Å, so the runner hands it a structure
   pre-divided by qcref's factor plus `Bohrs`. Side effect: the codec's optional
   normal-mode parser then refuses Hessian runs ("coordinates differ from the input
   structure"), and the runner reads the Hessian with the codec's `parse_hessian`
   instead. (A qccodec gap: the codec could write bohr + `! Bohrs`.)
2. **ORCA's `PBE` is not libxc's PBE**: it prints `XMuePBE 0.219520`, `CBetaPBE
   0.066725` (the 1996 paper's rounded constants); libxc uses μ = 0.2195149727645171,
   β = 0.06672455060314922. Water PBE/def2-SVP: ORCA -76.271957981892 vs PySCF
   -76.271943869666 (1.4e-5 Eh; NH3 1.2e-5). Recomputing the XC energy with ORCA's two
   constants on the PySCF density predicts the gap to 2.0e-8 Eh (`checks/orca_pbe_params.py`).
   Setting `%method XMuePBE 0.2195149727645171 CBetaPBE 0.06672455060314922 end`
   brings ORCA to -76.271943890072 (2.0e-8 Eh from PySCF). The runner does this for
   `pbe`/`pbe0` unless `--orca-native-pbe`. ORCA's PBE0 presumably shares these constants
   (not checked).
3. **ORCA `B3LYP/G` = libxc `B3LYP` (VWN_RPA)** to 1e-8 Eh (the VWN5 `B3LYP` is
   0.037 Eh off for water, per the AFH orca skill). ORCA prints `LDAOpt VWN-3`, but that
   is libxc's VWN_RPA (id 8), not libxc's `LDA_C_VWN_3` (id 30). Swapping in
   `LDA_C_VWN_3` would move water B3LYP by 3.7e-2 Eh (first order), while ORCA `B3LYP/G`
   is 1.1e-8 Eh from libxc B3LYP. PySCF's alias `VWN3` also means VWN_RPA, which matters
   if linus composes B3LYP from parts.
4. **ORCA 6.0.0's analytic DFT Hessian solves the CP-SCF equations on a separate grid
   whose size follows the `DefGrid` level and ignores `IntAcc`/`AngularGrid`.** Water
   B3LYP/def2-SVP with the fine grid settings alone: analytic `Freq` vs ORCA's own `NumFreq`
   (central differences of analytic gradients, 0.002 bohr) differ by 1.8e-4 Eh/bohr² (also
   with `DefGrid1` or `DefGrid2` added); adding `DefGrid3` (main grid unchanged, 164468
   points) brings it to 2.1e-6 (`validation/README.md`). It is not the first-order
   grid-response approximation of Bykov et al. (2015): linus's Hessian with exactly that
   approximation (`grid.response`) agrees with ORCA `NumFreq` to 4e-6 on PySCF's
   level-9 grid. The runner adds `DefGrid3` to analytic DFT Hessians on the fine grid and
   uses `NumFreq` by default. HF analytic Hessians are fine (2e-9). **Separately, analytic
   Hessians of non-hybrid functionals with exact Coulomb (`NoRI`) are wrong** (quirk 18).
5. **ORCA's largest angular grid is Lebedev-770** (`AngularGrid 7`; `8` aborts with
   "unknown parameters for the chosen AngularGrid"). By default H/He get one level
   less (`HGridReduced false` turns that off) and grids are pruned (`GridPruning 0`).
   With the finest settings ORCA DFT stays ~1e-8 Eh and ~2e-7 Eh/bohr from PySCF/Psi4
   (B3LYP water: IntAcc 8/10/12 all give 1.0-1.1e-8 Eh, 2.2-2.4e-7 Eh/bohr, while PySCF
   moves only 1e-9 Eh / 4e-8 Eh/bohr between (150,770) and (250,1454)). Cause not
   isolated; treat ~2e-7 Eh/bohr as the ORCA DFT gradient floor.
6. **ORCA can exit 0 after an error termination** ("ORCA finished by error termination in
   Startup" with exit code 0 for `AngularGrid 8`). Check for `ORCA TERMINATED NORMALLY`
   (qccodec does; the runner also does).
7. **`ConvCheckMode 0` ("All-Criteria") isn't strict**: ORCA still stops on "Energy
   convergence in two consecutive cycles with 'reasonable' density convergence", here at
   RMSDP 8e-10 and MaxDP 6e-9 against 1e-10/1e-9 requested. Energies and gradients still
   agree with PySCF/Psi4 to ~1e-10 Eh and ~1e-9 Eh/bohr.
8. ORCA 6: `Thresh`/`TCut` belong in `%scf` (in `%method` they stop the run: "Unknown
   identifier in METHOD block"). ORCA's DFT gradients are translation-invariant to 1e-12
   (consistent with grid-weight derivatives being included; not confirmed in its output). ORCA defaults to RI-J for GGAs and RIJCOSX for
   hybrids, so exact runs need `NoRI NoCOSX`.
9. Open MPI 4.1.6 from conda-forge prints openib (`librdmacm.so.1` missing) and CMA
   (`ptrace_scope=1`) warnings for every rank. `OMPI_MCA_btl=self,vader` and
   `OMPI_MCA_btl_vader_single_copy_mechanism=none` silence them. Serial and `%pal 8` give
   identical energies.

**Psi4**

10. **Psi4 1.11 RI-J / COSX gradients don't match their energies.** `scf_type
    DFDIRJ+LINK` (RI-J, exact K) and `DFDIRJ+COSX` compute the energy with that
    composite JK, but the gradient runs `DFJKGrad` with K density-fitted as well. Water
    HF/def2-SVP, def2/J: energies agree with PySCF and ORCA to 4e-13 and 1e-10, gradients
    are 1.7e-4 Eh/bohr off. The runner refuses Psi4 `rij`/`rijcosx` gradients and Hessians.
11. **Psi4 finite-difference Hessians project rotations out by default (`fd_project
    true`)**, which is wrong away from a stationary point. Water HF/def2-SVP, FD of
    gradients vs PySCF analytic: 2.7e-2 Eh/bohr² with the default, 9.7e-10 with
    `fd_project false` (`checks/psi4_fd_project.py`). Psi4's GGA and meta-GGA Hessians are
    FD of gradients, so set `fd_project false` (the runner does). **Even with `fd_project
    false`, Psi4 projects translations** ("Translations projected? 1. Rotations projected?
    0."), in mass-weighted SALC coordinates: H = Q J Q^T with Q = M^1/2 (1 - T_m T_m^T)
    M^-1/2. That is harmless for translation-invariant gradients (HF), but Psi4's DFT
    gradients have no grid-weight derivatives (quirk 12), so their Jacobian is not
    symmetric. Psi4's DFT grid also turns with the molecule's standard orientation at every
    displaced geometry. Together these make Psi4's FD DFT Hessian differ from an analytic
    Hessian whose grid follows the atoms by translation only (linus `grid.response`):
    7.5e-5 Eh/bohr² for water B3LYP on the default grid. The same finite differences of
    linus gradients on the same grid, projected the same way, reproduce Psi4's Hessian to
    4e-9 (`validation/README.md`).
12. Psi4 DFT gradients have no grid-weight derivatives: |Σ_atoms g| = 2.7e-8 (B3LYP) and
    1.1e-7 Eh/bohr (PBE) for water at 150x974. PySCF's grid response changes the water
    B3LYP gradient by 6.5e-8; PySCF without it matches Psi4 to 6e-8.
13. `import psi4` sets OpenMP threads to 1 in that process (PySCF's `lib.num_threads()`
    then reports 1). The codec runs Psi4 in a subprocess, so the runner never imports it.
14. The qccodec Psi4 driver runs `psi4 input.py -k -o stdout`: the output goes to the log,
    not to a file. The runner saves it as `psi4_run/psi4.out`.

**PySCF**

15. conda-forge has pyscf 2.14.0 while PyPI has 2.15.0 (2026-11-05); the env uses the
    PyPI wheel (bundled libcint 6.2.0, libxc 7.1.2).
16. `conv_tol_grad` is the norm of the whole orbital gradient, not an RMS. At 1e-10 PySCF's
    DIIS crawls (42 iterations for water, 0.85x per cycle below 1e-9), so the runner
    uses 1e-9 (an RMS below 1e-10).
17. `dft.libxc.register_custom_functional_` works with `numint.nr_rks`, but PySCF 2.15.0
    `RKS.kernel()` rejects the custom name (`do_nlc` lookup); `checks/orca_pbe_params.py`
    works around this with first-order perturbation.

**Found while building the validation matrix (`validation/`, 2026-11-07)**

18. **ORCA 6.0.0's analytic Hessian of non-hybrid functionals with exact Coulomb (`NoRI`) is
    wrong by ~5e-2 Eh/bohr².** Water def2-SVP, analytic `Freq` vs `NumFreq` (0.002 bohr):
    PBE `NoRI` 4.9e-2 (bend 1702 vs 1528 cm⁻¹, stretches 3989/4053 vs 3910/4015), TPSS
    4.9e-2, VWN3, VWN5 and HFS 4.4-5.5e-2, BLYP 5.0e-2, on every grid; with RI-J (`RI
    def2/J`, ORCA's default for these functionals) 5.8e-6, and hybrids (B3LYP, PBE0) agree.
    PySCF and Psi4 analytic Hessians and linus agree with ORCA `NumFreq`, not with the
    analytic one. Use RI-J or `NumFreq` for non-hybrid ORCA Hessians.
19. ORCA 6.0.0 has no analytic Hessian with `RIJK` ("Analytical Hessian not available with
    RIJK approximation", the run aborts); the runner switches to `NumFreq`. Its analytic
    `Freq` writes no gradient unless `EnGrad` is also given (the runner adds it).
20. ORCA computes the dipole about the centre of mass and prints the origin with 6
    decimals; `orca.property.txt` has the nuclear and electronic parts in full precision, from
    which the runner recovers the origin exactly and shifts to (0, 0, 0) (matters for ions:
    NH3+ agrees with PySCF/Psi4 to 7e-9 e bohr after the shift).
21. ORCA's `VWN3` (Slater + VWN-3) is libxc's VWN_RPA also for open shells: NH3+ UKS
    SVWN-RPA/def2-SVP on the fine grids, ORCA vs PySCF 1.2e-9 Eh, vs Psi4 6.9e-10.
22. Psi4 1.11's analytic Hessians cover HF and LDA only (RHF/UHF/RKS/UKS; `proc_table.py`
    registers analytic Hessians only for functionals that are not GGAs); GGA and meta-GGA
    Hessians are finite differences of gradients. Analytic DF (`mem_df`) HF Hessians and UKS
    meta-GGA gradients work. Psi4's only SVWN is Slater + VWN_RPA.
23. PySCF's `small_rho_cutoff` (default 1e-7) drops low-density grid points after the first
    SCF cycle; to reproduce a grid point for point (linus `pyscf:N`) set it to 0.
24. PySCF 2.15 has no dispersion of its own: `mf.disp` needs `pyscf-dispersion` (its
    `d3bj` is two-body, `d3bjatm` adds ATM; its D3 Hessian is central differences of D3
    gradients, step 1e-5). Psi4's `-d3bj` (two-body) and `-d4` go through qcengine's
    harnesses and need the `dftd3` and `dftd4` Python packages.
25. `tests/molecules/hcn_cl.xyz` has 31 electrons; the matrix uses it as the closed-shell
    cation ClCNH+ (charge +1), the neutral is a doublet.
26. ORCA 6.0.0's own TPSS is 8.5e-7 Eh above libxc's for water/def2-SVP (converged in IntAcc;
    `CBetaPBE`/`XMuePBE` do not affect it), and its own wB97X 4.0e-6 Eh above libxc's
    (IntAcc 8, 10, 12 alike). PySCF and Psi4 (libxc) agree to 4e-9 and 6e-10 on matching
    grids. wB97X/wB97 are unusually sensitive to angular pruning: PySCF's pruned level-7/9
    grids sit 1e-6 Eh from unpruned 150x974 (B3LYP 2e-9).
27. orca_vib's wavenumbers are a constant factor 0.99998181 below PySCF's, Psi4's and linus's
    for the same Hessian and masses (0.07 cm⁻¹ at 4000 cm⁻¹).

28. ORCA 6.0.0 refuses range-separated RIJK gradients: "Error (SHARK/RIJK/Gradient/RangeSep):
    Sorry, the range sep RIJK gradient has an unresolved bug - please use another
    approximation" (wB97X/def2-SVP RIJK, `EnGrad` and `NumFreq`). The matrix takes ORCA's
    RIJK energy and a `NumGrad` gradient for range-separated hybrids.
29. ORCA 6.0.0 `RIJK` with a non-hybrid functional stops with "no AuxJ basis set in
    ORCA_GTOInt" unless `AuxJ` is given too; the runner sets it to the same def2/JK.

Quirks 4 and 18 are reproduced by `checks/orca_hessian_checks.py`.

**Basis sets and DF**: the built-in def2-SVP, def2-TZVP and cc-pVTZ libraries of the three
programs agree (HF energies to ≤1.2e-10 Eh). ORCA `def2/JK` = BSE `def2-universal-jkfit`
in PySCF and Psi4: HF/def2-SVP RI-JK energies agree to 1e-10 Eh and gradients to 7e-10.
`rijcosx` can't agree across programs (PySCF sgX, Psi4 COSX and ORCA COSX use different
grids): water HF/def2-SVP spreads over 9e-6 Eh.

**Range-separated hybrids** (linus `wb97x`, `wb97x-d3bj`, `cam-b3lyp`, `wb97m-d3bj`, libxc names;
`tests/python/compare_rsh_pyscf.py`, `tests/python/fd_grad_linus.py`):

- ORCA's `wB97X-D3BJ` is PySCF `xc = "wb97x-v"` with `mf.nlc = False` (VV10 off) plus D3(BJ);
  linus's `wb97x-d3bj` alias is the same (`dispersion: "none"` drops the D3 part).
- **PySCF's DF fits K_LR with the erf metric** (`with_range_coulomb` on mol and auxmol:
  (mn|erf|P) (P|erf|Q)^-1 (Q|erf|ls)); Psi4 and linus use the wK form (Coulomb metric,
  sym[(mn|P) V^-1 (P|erf|ls)]). The two differ by 1e-7 to 3e-5 Eh (water to CH3SH,
  def2-SVP, def2-universal-jkfit). `compare_rsh_pyscf.py --df` rebuilds wK from PySCF
  integrals for a like-for-like reference (agreement <= 2e-12 Eh).
- **libxc `HYB_GGA_XC_LC_WPBE` (HJS hole) is not self-consistent at ~1e-5**: linus and PySCF
  analytic gradients agree (2e-10 Eh/bohr), but both differ from finite differences of the
  energy by up to 2e-5 Eh/bohr (NH3/def2-SVP), so its potential is not the exact derivative
  of its energy. Its energies also differ between libxc 7.0.0 (linus) and 7.1.2 (PySCF) by
  1.5e-10 to 7e-10 Eh. `HYB_GGA_XC_LRC_WPBE` (wPBEh hole, also pure long-range exchange)
  matches FD to 4e-11 Eh/bohr and PySCF to <= 8e-13 Eh on water, NH3 (RKS, UKS) and CH3SH+
  (UKS), but on neutral CH3SH (RKS, def2-SVP) it is 1.8e-9 Eh / 1.1e-7 Eh/bohr from PySCF and
  1.4e-8 Eh/bohr from its own FD, while wB97X, CAM-B3LYP and wB97M there (same K_LR code)
  agree to 1e-12 Eh / 3e-10 Eh/bohr: libxc's wPBEh numerics, not the exchange integrals.
- **ORCA prints a `DISPERSION GRADIENT` block** with `EnGrad`; subtracting it from
  `orca.engrad` gives the dispersion-free gradient (ORCA 6.0.0's analytic D3(BJ) gradient is
  not the derivative of its D3 energy, up to 4e-4 Eh/bohr, so compare gradients without
  dispersion). ORCA's D3(BJ) energy for wB97X-D3BJ equals linus's (la18: both -0.030739726).
- **linus `orca:2` was ORCA-like, not ORCA's DefGrid2, until 2026-11-07**: exact-integral la18
  energies differed by 1.2e-5 Eh (def2-SVP) and 2.1e-4 Eh (def2-TZVP) for wB97X-D3BJ, and by
  2.2e-4 Eh for B3LYP/G at def2-TZVP (ts61 9.7e-4 from its K atom, PdCl2(PMe2Ph)2 1.9e-3 from Cl
  and P): the guessed grid's own error; RI-J, basis sets and the functional agree with ORCA. Since
  then `orca:N` is ORCA 6.0.0's DefGrid N, measured from ORCA's output: B3LYP energies agree with
  ORCA's to 2e-9 (water) - 2e-7 Eh (ts61), ORCA's basis-adaptive grid changes aside. The old grid is `GridSpec::orca_defgrid_guess`.
