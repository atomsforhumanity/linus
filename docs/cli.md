# The `linus` command line

`linus` runs one calculation: on an XYZ file (Å) with command-line options, or on a JSON
request (a file, or stdin). It prints a summary for people, or with `--json` the whole result as
one JSON object. The settings and their grammar are the library's
(`linus::CalcOptions`), so the CLI and the Python package (qcdata's `ProgramInput.keywords`,
[python.md](python.md)) accept the same names and values.

```sh
linus mol.xyz --method b3lyp-d3bj --basis def2-tzvp --jk df --grad --threads 8
linus mol.xyz --method wb97x-d3bj --ir --thermo qrrho -T 298.15 --json > result.json
linus request.json --threads 8 --json
linus --json < request.json
linus --help          # the flags, each with its key
linus --list-settings # every setting, with its type, default and meaning
```

Exit codes: `0` success; `1` a failed calculation (an SCF that does not converge, not enough
memory, a GPU failure, a numerical failure, something linus does not implement, an internal error); `2` bad
input (command line, request, settings, unknown basis set or functional). With `--json`, a
failure also prints one JSON object on stdout:
`{"schema_version": 1, "program": "linus", "version": ..., "success": false, "error": ..., "error_kind": ...}`,
plus `calctype` when the calculation itself failed (after setup), and `scf`, `energy` and
`timings` (the last SCF iterate and the SCF history) when the SCF ran without converging or
failed numerically, and `response` (`what`, `iterations`, `max_residual`, `tol`) when a
Hessian's or polarizability's CPHF/CPKS stopped above its tolerance, and `error_code` when the
error has a code within its kind (qcdata's shared codes): a setting's `unknown_key`, `no_effect`
or `not_applicable`, a resource's `host_memory` (not enough host memory: the DF tensors, the
in-core integrals, the solvent's matrices, a Hessian's arrays; the message says how much is
needed) or `device_memory`, a convergence failure's `scf_not_converged` (the SCF ran out of
iterations) or `response_not_converged` (a CPHF/CPKS solve stopped above its tolerance).
`error_kind` is one of:

| `error_kind` | exit code | meaning |
|---|---|---|
| `input` | 2 | bad command line, request or settings; unknown basis set, functional or element |
| `basis` | 2 | a basis set file that cannot be read or parsed |
| `unsupported` | 1 | a valid request for something linus does not implement (ROHF, `rij_snk` Hessians, `wb97x-d3`, ...) |
| `resource` | 1 | not enough memory (`memory_gib`) or threads for the request |
| `backend` | 1 | the GPU backend failed: no usable device for `backend: "gpu"`, or a device error during the calculation ([gpu.md](gpu.md)) |
| `convergence` | 1 | the SCF, or the CPHF/CPKS of a Hessian or polarizability, did not converge |
| `numerical` | 1 | a numerical failure (non-finite energies, singular matrices) |
| `internal` | 1 | a bug in linus (a panic or a broken internal invariant); the message says where |

The kind does not depend on when the error is found (while reading the request, setting up the
calculation or running it). Python raises the same kinds (the exceptions' `kind`; [conventions.md](conventions.md#errors));
a calculation's exception also carries this failure JSON as `failure` and, when the SCF ran
without converging, the result it reached as `result`. qcdata results carry the kind as
`extras["error_kind"]`.

### Basis sets

`--basis NAME` reads `<dir>/<name>.json`, a Basis Set Exchange JSON file, with the name lower-cased
and `*`, `+`, `/`, `(`, `)`, `,` written as `_st_`, `_pl_`, `_sl_`, `_lp_`, `_rp_`, `_cm_`
(`6-311+G(2d,p)` is `6-311_pl_g_lp_2d_cm_p_rp_.json`), from the first directory of the basis set
search path that has it: `--basis-dir DIR` (repeatable; the setting `basis_dir`), then
`$LINUS_BASIS_DIR` (directories separated as in `PATH`), then the source tree's `data/basis` in
a development build, then the installation's sets: `<prefix>/share/linus/basis` for a binary
installed as `<prefix>/bin/linus`, or the Python package's own copy. Release builds (made with
`LINUS_DIST` set) carry no path of the build machine, so a release archive's binary reads the
`share/linus/basis` beside it, or `LINUS_BASIS_DIR`/`--basis-dir`. `--list-basis` lists the
whole search path.

## Settings

Every setting has one key, used alike by the CLI (`--set`), a JSON request and Python
(qcdata's `ProgramInput.keywords`): a top-level key
(`method`, `threads`) or `group.key` in one of the groups below (`scf.max_iter`), never deeper.
A command-line flag is a shortcut for one key (`--help` names it); `--set KEY=VALUE` sets any
key of the request, VALUE parsed as JSON, else taken as a string. A request file or `--options
FILE` is the base; flags, `--options` and `--set` then apply in command-line order, the last
one winning. A dotted path below a group's shorthand refines it: `--jk df --set
jk.aux=def2-universal-jkfit`, `--precision mixed --set precision.fp64=native`, `--solvent thf
--set solvent.model=smd`.

```sh
linus mol.xyz --set scf.soscf=true --set jk.df_storage=f48 --set response.tol=1e-9
linus mol.xyz --options settings.json --set threads=16
linus --list-settings scf          # every setting of a group, with type, default and meaning
linus schema settings              # the JSON Schema (draft 2020-12) of the settings
```

The rules:

- Keys and enum values ignore case, and `-` equals `_` (`rij_snk`, `rij-snk`, `RIJCOSX`).
  Other programs' names are accepted as synonyms (`rijcosx`, `ri`, `rhf`/`uhf`, `d3(bj)`,
  `defgrid2`) and never echoed. Levels are checked, not clamped (`linus:9` is an error).
- `"auto"` means linus decides; null or absent means unset (the default). `resolved.settings`
  says what each `auto` became, `resolved.reasons` why.
- Value shorthands stand for a group: `jk: "df"` (`jk.kind`; `"incore"` is exact with
  `jk.exact_storage: "incore"`), `grid: "linus:3"` or a PySCF level (`grid.xc`), `solvent:
  "thf"` (`solvent.name`) or a number (`solvent.eps`), `backend: "gpu"` (`backend.kind`).
- An unknown key is an input error (code `unknown_key`) with the closest key as a suggestion.
  Every setting given must take effect in the calculation (decision D41): a setting that
  cannot, such as `grid.xc` for Hartree-Fock, `jk.screening` with `jk: "df"` or `response.*` for
  an energy, is an input error naming it (code `no_effect`, or `not_applicable` when only
  another calctype reads it). Settings that take effect when a run-time condition fires (the
  level shift, the plateau rule, `jk.df_storage` of a GPU run that may fall back to the CPU) are
  accepted.

The tables below are generated from the library's settings table (`linus --list-settings
--all` also lists the development settings; `x-tier`, `x-concern`, `x-unit` and `x-applies-to`
are in the JSON Schema).

<!-- settings:begin -->
**Top level**

| key | values | default | meaning |
|---|---|---|---|
| `method` | string | `hf` | `hf` or a functional (`--list-functionals`; libxc names, mixtures such as `"0.5*b3lyp + 0.5*hf"`). A dispersion suffix (`b3lyp-d3bj`, `-d3(bj)`, `-d3bj-atm`, `-d4`) sets `dispersion`, unless linus knows the whole name (`wb97x-d3bj`). |
| `basis` | string | `def2-svp` | The orbital basis set (`--list-basis`); the def2 sets carry ECPs for Rb-Rn. |
| `dispersion` | auto \| none \| d3bj \| d3bj-atm \| d4 | `auto` | Empirical dispersion: `d3bj`, `d3bj-atm`, `d4` or `none`. `auto`: implied by the method's name, else none. Bare `d3` is ambiguous (zero damping in Psi4 and qcdata, Becke-Johnson in ORCA) and rejected. |
| `reference` | auto \| restricted \| unrestricted | `auto` | The determinant: `restricted` (synonyms `rhf`, `rks`) or `unrestricted` (`uhf`, `uks`). `auto`: restricted for closed shells, unrestricted for open shells (ROHF is not implemented). |
| `cartesian` | boolean | `false` | Cartesian d and higher functions (false: spherical). *(expert)* |
| `basis_dir` | string \| [string] | `[]` | Directories of BSE JSON basis set files searched first (a string or a list), then `$LINUS_BASIS_DIR`, a development build's source tree and the installation's sets. *(expert)* |
| `threads` | integer | `1` | Threads of the calculation's own pool. |
| `memory_gib` | auto \| number (GiB) | `auto` | The memory the calculation may use; every host budget derives from it. `auto`: the total RAM, or the cgroup limit if lower. |
| `device_memory_gib` | auto \| number (GiB) | `auto` | The device memory a GPU backend may use; every device budget derives from it. `auto`: `$LINUS_GPU_MEMORY_GIB`, else 95% of the memory free when the device opens (at most that either way). *(expert)* |

**`jk`**: The J/K approximation.

| key | values | default | meaning |
|---|---|---|---|
| `jk.kind` | exact \| df \| rij_snk | `df` | The approximation: `df` (density fitting of J and K with the calculation's auxiliary basis, `aux`; synonyms `ri`, `rijk`), `exact` four-center integrals, or `rij_snk` (density-fitted J with seminumerical K, ORCA's RIJCOSX; synonym `rijcosx`). |
| `jk.aux` | auto \| basis set ("a\|b": a composite) | `auto` | The auxiliary basis of `df` and of `rij_snk`'s J. `auto`: def2-universal-jkfit for energies and gradients, `cc-pvtz-jkfit\|def2-universal-jkfit` for Hessians (df); def2-universal-jfit (rij_snk). `"a\|b"` takes each element from the first set that has it. |
| `jk.k_grid` | grid | `auto` | The grid of `rij_snk`'s seminumerical exchange: `cosx:N` (ORCA-like COSX grids of DefGrid N), `gridx:A,B,C` (COSX grid levels of the first and last SCF iterations and the final energy) or any grid. `auto`: `cosx:2`. *(expert)* |
| `jk.screening` | number | `1e-12` | Screening threshold of the exact four-center integrals (derivatives use at most 1e-12). *(expert)* |
| `jk.exact_storage` | auto \| direct \| incore | `auto` | Where `exact` keeps its integrals: `direct` (recomputed in every Fock build) or `incore` (all held in memory: tiny systems and tests). `auto`: direct. *(expert)* |
| `jk.df_storage` | auto \| f64 \| f48 \| f40 \| f32 | `auto` | Storage format of `df`'s three-index exchange tensor: `f64`, or the compressed `f48` (relative error at most 9e-13), `f40` (2e-10), `f32` (6e-8). `auto`: f64; a memory budget too small for it is a `host_memory` error that names the `memory_gib` needed (the compressed formats only when set, so memory never changes the arithmetic). *(expert)* |
| `jk.df_long_range` | auto \| stored \| one_sided | `auto` | Long-range exchange of range-separated hybrids with `df`: `stored` (B and B^omega in memory) or `one_sided` (raw integrals of one side per build). `auto`: stored when both fit the budget. *(expert)* |
| `jk.incremental` | boolean | `true` | Incremental Fock builds (from the change of the density) for integral-direct exact J/K and the seminumerical exchange. *(expert)* |
| `jk.incremental_reset` | integer | `20` | A full rebuild after this many incremental builds. *(expert)* |
| `jk.grid_switch` | number | `0.001` | The seminumerical exchange moves to its next (finer) grid once max\|F_ai\| is below this. *(expert)* |

**`grid`**: The grids of DFT.

| key | values | default | meaning |
|---|---|---|---|
| `grid.xc` | grid | `auto` | The XC quadrature of the SCF, gradients and Hessians: a PySCF level `N`/`pyscf:N` (0-9), `linus:N` (1-4), `orca:N` (ORCA 6.0.0's DefGrid 1-3), `psi4`, or a grid's full specification `{"kind": "spec", ...}`. `auto`: pyscf:3 for energies and gradients, linus:3 for Hessians. |
| `grid.nlc` | grid | `auto` | The grid of VV10 nonlocal correlation (-V functionals). `auto`: by the XC grid's family: linus:1 with the default and linus grids, pyscf:3 with PySCF grids, ORCA's NL grid with `orca:N`, Psi4's VV10 grid with `psi4`. *(expert)* |
| `grid.kernel` | grid | `auto` | The grid of the XC kernel in the CPKS equations of Hessians. `auto`: by the XC grid's family: linus:1 (meta-GGAs linus:2) with the default and linus grids, the XC grid with PySCF, ORCA and Psi4 grids. *(expert)* |
| `grid.response` | boolean | `true` | The grids move with the atoms in gradients, Hessians and dipole derivatives (the XC, VV10 and seminumerical-exchange grids' weight derivatives and point motion, as ORCA); false: fixed grids, PySCF's and Psi4's convention, for comparisons with them. *(expert)* |

**`precision`**: The arithmetic.

| key | values | default | meaning |
|---|---|---|---|
| `precision.fp64` | auto \| native \| emulated | `auto` | How the GPU reaches FP64 accuracy in the density-fitted exchange GEMMs: `native` (the FP64 units) or `emulated` (FP64 emulation on INT8 tensor cores). `auto`: native on FP64-rich cards (A100, H100/H200) and the CPU, emulated on the others (D27). *(expert)* |
| `precision.fp32` | auto \| off \| on | `auto` | Where FP32 arithmetic may run outside the SCF loop: `off` (nowhere: FP64 throughout, VV10's pair sums too) or `on` (also the GPU's XC gradient, CPKS kernel, Hessian XC and VV10 gradient products). `auto`: linus's validated uses (the single-precision exchange of large RI responses, the GPU's CPKS and VV10 kernels with it, VV10's pair sums in double-float). *(expert)* |
| `precision.response` | auto \| fp32 \| fp64 | `auto` | The arithmetic of the RI response exchange (CPHF/CPKS): `fp32` or `fp64` (synonyms `single`, `double`). `auto`: as `fp32` says (with `off`, fp64; else single precision in large RI Hessians unless `response.tol` is below 1e-6; on the GPU, the card's rule, docs/gpu.md). *(expert)* |

**`solvent`**: The implicit solvent (absent: gas phase).

| key | values | default | meaning |
|---|---|---|---|
| `solvent.name` | string | `none` | The solvent, by name (`--list-solvents`; the rest of the CRC table through `LINUS_SOLVENT_TABLE`): its permittivity and, for SMD, its descriptors. |
| `solvent.eps` | number | `none` | The static permittivity (overrides the named solvent's; at least 1). |
| `solvent.model` | cpcm \| cosmo \| iefpcm \| ssvpe \| smd | `cpcm` | The continuum model: `cpcm`, `cosmo`, `iefpcm`, `ssvpe` or `smd` (needs a solvent name with SMD descriptors). |
| `solvent.preset` | pyscf \| orca | `pyscf` | The cavity and discretization defaults: `pyscf` (PySCF 2.15's pyscf.solvent.pcm) or `orca` (ORCA 6.0.0's C-PCM). *(expert)* |
| `solvent.df` | boolean | `false` | The density-fitted PCM: the surface terms through the auxiliary basis of the density-fitted J (about up to 4% of the electrostatic solvation energy; no Hessians or response properties). *(expert)* |
| `solvent.lebedev` | auto \| integer | `auto` | Lebedev points per sphere. `auto`: 302 (SMD 590) with the pyscf preset, by point density with orca. *(expert)* |
| `solvent.discretization` | auto \| swig \| iswig | `auto` | The switching of the surface: `swig` or `iswig`. `auto`: by the preset (pyscf swig, orca iswig). *(expert)* |
| `solvent.radii` | object (Å) | `{}` | Per-element radii before scaling, by symbol or atomic number (`{"H": 1.2}`); they override the preset's table. *(expert)* |
| `solvent.radii_scale` | auto \| number | `auto` | The scale of the radii. `auto`: 1.2 (SMD 1.0). *(expert)* |
| `solvent.r_probe` | number (Å) | `0.0` | A probe radius added to the scaled radii. *(expert)* |
| `solvent.cosmo_x` | number | `0.5` | COSMO's x in f = (eps - 1) / (eps + x). *(expert)* |
| `solvent.cds_contact_smoothing` | auto \| number (Å) | `auto` | SMD: the window over which new sphere contacts of the CDS surface area are smoothed. `auto`: 0 (the exact area) for single points, 0.1 for gradients and Hessians. *(expert)* |
| `solvent.scf_terms` | auto \| exact \| predicted | `auto` | How the SCF evaluates the surface terms: `exact` (two passes of exact integrals per iteration) or `predicted` (one fused pass with predicted charges; the same converged result). `auto`: predicted with density-fitted J, else exact. *(expert)* |
| `solvent.incremental` | boolean | `true` | The predicted mode's passes on the changes of the density and the charges. *(expert)* |
| `solvent.screening` | number | `1e-14` | The screening threshold of the surface-charge integrals. *(expert)* |

**`scf`**: The SCF.

| key | values | default | meaning |
|---|---|---|---|
| `scf.convergence` | auto \| loose \| normal \| tight \| verytight \| extreme \| pyscf | `auto` | A preset of `e_tol` and `g_tol`: `loose` (1e-5, 1e-4), `normal` (1e-6, 1e-5), `tight` (1e-8, 5e-7), `verytight` (1e-9, 1e-8), `extreme` (1e-12, 1e-10) or `pyscf` (PySCF's defaults, on PySCF's gradient norm). `auto`: tight for energies, 1e-9 and 1e-7 for gradients, 1e-9 and 1e-9 for Hessians. The presets are absolute: one looser than the calculation's default in a threshold warns (`convergence_looser_than_default`; `tight` loosens a gradient, only `extreme` tightens a Hessian). |
| `scf.e_tol` | auto \| number (Eh) | `auto` | The energy change between iterations below which the SCF may stop (overrides the preset's). `auto`: the preset's. |
| `scf.g_tol` | auto \| number | `auto` | The orbital gradient max\|F_ai\| below which the SCF may stop (PySCF's norm with the `pyscf` preset; overrides the preset's). `auto`: the preset's. |
| `scf.max_iter` | integer | `288` | Iterations at most: the SCF's whole budget, the escalation's included. |
| `scf.guess` | sad \| core | `sad` | The initial guess: `sad` (superposition of atomic densities) or `core`. |
| `scf.stability` | auto \| off \| always | `auto` | The stability of UHF/UKS determinants: `auto` (the probes below, and a full analysis with descent when one fires or the SCF looks like a saddle point), `off` (no probe or analysis) or `always` (a full analysis of every converged determinant). |
| `scf.stability_probe` | auto \| integer | `auto` | Davidson products of the open-shell probe of a converged UHF/UKS determinant (0: off, else at least 2; a negative mode starts the full analysis). `auto`: 2 at a session's first geometry, none at later ones unless the gap trigger fires. *(expert)* |
| `scf.stability_closed_probe` | auto \| integer | `auto` | Davidson products of the closed-shell probe of a converged RHF/RKS point whose HOMO-LUMO gap is below `stability_closed_gap` (towards a broken-symmetry UHF/UKS solution; a warning only, the result is unchanged; 0: off, else at least 2). `auto`: 2 for Kohn-Sham DFT at a session's first geometry, none for Hartree-Fock. *(expert)* |
| `scf.stability_closed_gap` | number (Eh) | `0.15` | The HOMO-LUMO gap below which the closed-shell probe runs. *(expert)* |
| `scf.stability_max` | integer | `2` | Full stability analyses per SCF at most. *(expert)* |
| `scf.stability_saddle_check` | number (Eh) | `1e-9` | A converged UHF/UKS energy this far above the lowest one the SCF visited triggers a stability analysis (0: off). *(expert)* |
| `scf.stability_hessian_check` | integer | `6` | Davidson products per irrep of the check of the lowest orbital-Hessian eigenvalue before an open-shell Hessian: the stability probe of every converged point of its SCF runs with at least these (a negative mode is followed, as the probe's is; a degenerate solution, or a negative mode no step lowers, is flagged with a warning; 0: off, else at least 2). *(expert)* |
| `scf.diis_space` | integer | `10` | The size of the DIIS subspace. *(expert)* |
| `scf.diis_start` | integer | `1` | The first iteration whose Fock matrix enters DIIS. *(expert)* |
| `scf.diis_drop` | largest_error \| oldest | `largest_error` | Which entry leaves a full DIIS subspace: `largest_error` or `oldest`. *(expert)* |
| `scf.adiis` | boolean | `false` | ADIIS while max\|[F, D]\| is above `adiis_until`, then DIIS. *(expert)* |
| `scf.adiis_until` | number | `0.01` | The orbital gradient below which ADIIS hands over to DIIS. *(expert)* |
| `scf.level_shift` | number (Eh) | `0.25` | The level shift of the virtual orbitals while max\|F_ai\| is above `level_shift_until`. *(expert)* |
| `scf.level_shift_until` | number | `0.01` | The orbital gradient below which the level shift stops. *(expert)* |
| `scf.damping` | number | `0.0` | Fock damping F = (1 - d) F + d F_previous while max\|F_ai\| is above `damping_until` (in [0, 1)). *(expert)* |
| `scf.damping_until` | number | `0.0` | The orbital gradient below which damping stops. *(expert)* |
| `scf.soscf` | boolean | `false` | Quasi-Newton orbital rotations (SOSCF) instead of DIIS once max\|F_ai\| is below `soscf_from`. *(expert)* |
| `scf.soscf_from` | number | `0.0033` | The orbital gradient from which SOSCF takes over. *(expert)* |
| `scf.two_cycle` | boolean | `true` | When the energy cycles between two determinants, quasi-Newton steps replace DIIS (once per SCF). *(expert)* |
| `scf.plateau_window` | integer | `50` | An SCF whose energy and orbital gradient stayed on a plateau over this many iterations when its escalation comes due, or at `max_iter`, is accepted with the warning `loosely_converged`; under accuracy targets only when the estimates of its final orbital gradient meet them (0: off). *(expert)* |
| `scf.escalate` | boolean | `true` | An SCF whose DIIS stagnates (`stagnation_window`), or that has spent 2/3 of `max_iter`, restarts once from the lowest-energy orbitals it visited, with quasi-Newton steps, for the rest of `max_iter`. *(expert)* |
| `scf.stagnation_window` | integer | `64` | Iterations without a new lowest energy after which the escalation starts (0: only at 2/3 of `max_iter`). *(expert)* |
| `scf.lindep` | number | `1e-8` | Canonical orthogonalization: overlap eigenvalues at or below this (absolute) drop their functions; fewer orbitals left than occupied ones is an input error (in (0, 1)). *(expert)* |

**`response`**: The response equations of Hessians.

| key | values | default | meaning |
|---|---|---|---|
| `response.tol` | auto \| number | `auto` | The max-abs residual of the Hessian's nuclear CPHF/CPKS (the Hessian's error is quadratic in it). `auto`: 1e-6, 1e-5 in two-level responses. |
| `response.field_tol` | number | `1e-6` | The max-abs residual of the electric-field CPHF/CPKS of IR intensities and the polarizability. *(expert)* |
| `response.iteration_aux` | auto \| none \| main \| basis set | `auto` | The auxiliary basis of a cheaper J/K for the CPKS iterations of large RI Hessians: `none`, `main` (the main auxiliary basis) or a basis set. `auto`: two levels for a large single-precision response. *(expert)* |

**`freq`**: The analysis and thermochemistry a Hessian carries.

| key | values | default | meaning |
|---|---|---|---|
| `freq.analysis` | harmonic \| off | `harmonic` | The harmonic analysis: `harmonic` or `off` (the Hessian alone). |
| `freq.thermo` | rrho \| qrrho \| qrrho-hg \| quasiharmonic \| off | `rrho` | The thermochemistry: `rrho` (harmonic), `qrrho` (Grimme 2012's quasi-RRHO entropy), `qrrho-hg` (that and Li and Head-Gordon 2015's quasi-RRHO enthalpy), `quasiharmonic` (Truhlar 2011: real frequencies below `raise_below` raised to it) or `off`. |
| `freq.ir` | boolean | `false` | Dipole derivatives and IR intensities (from the Hessian's CPHF solutions) and the static polarizability (an electric-field CPHF). |
| `freq.temperature` | number \| [number] (K) | `298.15` | The temperature of the thermochemistry, or a list of them (one thermochemistry each, in order). |
| `freq.pressure` | number (Pa) | `101325.0` | The pressure of the ideal-gas standard state. |
| `freq.concentration` | number (mol/L) | `none` | A concentration standard state instead of the pressure's (1 for 1 M). *(expert)* |
| `freq.scale` | number | `1.0` | The factor the frequencies are scaled by in the thermochemistry. |
| `freq.rotor_cutoff` | number (cm^-1) | `100.0` | The free-rotor interpolation's crossover of the quasi-RRHO models (`qrrho`, `qrrho-hg`). *(expert)* |
| `freq.raise_below` | number (cm^-1) | `100.0` | Quasiharmonic: real frequencies below this are raised to it. *(expert)* |
| `freq.qrrho_alpha` | number | `4.0` | The exponent of Grimme's interpolation weight (quasi-RRHO models). *(expert)* |
| `freq.qrrho_bav` | number (kg m^2) | `1e-44` | Grimme's average moment of inertia B_av (quasi-RRHO models). *(expert)* |
| `freq.imaginary_as_real` | number (cm^-1) | `0.0` | Imaginary modes smaller than this are treated as real modes of \|nu\| in the thermochemistry (0: none). *(expert)* |
| `freq.electronic_degeneracy` | auto \| number | `auto` | The electronic degeneracy. `auto`: the spin multiplicity. *(expert)* |
| `freq.mass_table` | isotope \| average | `isotope` | The table of the default masses: `isotope` (the most abundant isotope) or `average` (standard atomic weights). *(expert)* |
| `freq.masses` | [number] (u) | `none` | The masses of all atoms, replacing the table's. *(expert)* |
| `freq.isotopes` | [[atom, mass]] (u) | `[]` | Single atoms' masses, applied last: [[atom index, mass], ...] (deuterium 2.01410177812). *(expert)* |
| `freq.projection` | translations_rotations \| translations \| none | `translations_rotations` | What is projected out of the mass-weighted Hessian: `translations_rotations`, `translations` or `none` (the translation- and rotation-like modes then flagged `external`). *(expert)* |
| `freq.symmetry_number` | auto \| integer | `auto` | The rotational symmetry number. `auto`: from the detected point group. *(expert)* |
| `freq.symmetry_tol` | number (bohr) | `0.01` | The tolerance of the point-group detection and of the test for linear molecules. *(expert)* |
| `freq.codata` | codata2022 \| codata2014 | `codata2022` | The physical constants: `codata2022` or `codata2014` (as Psi4 1.11 and PySCF 2.15). *(expert)* |

**`backend`**: The hardware.

| key | values | default | meaning |
|---|---|---|---|
| `backend.kind` | cpu \| gpu \| auto | `cpu` | `cpu`, `gpu` (every kernel family that has device kernels on an NVIDIA GPU, the others on the CPU, each with its reason; a build with the `gpu` feature) or `auto` (the GPU when the build has it and a device opens, else the CPU). |
| `backend.devices` | integer \| [integer] | `[0]` | The GPU to use: a device id (or a list of one). *(expert)* |
| `backend.scf_loop` | auto \| host \| device | `auto` | Where the SCF loop's own work runs with a GPU: `host` (the CPU drives the device's Fock builds) or `device` (the whole loop on the GPU). `auto`: device, closed and open shells. *(expert)* |
| `backend.kernel_cache` | auto \| off \| directory | `auto` | Where a GPU backend keeps its compiled kernels between runs: a directory or `off` (memory only). `auto`: `$LINUS_GPU_CACHE`, else `$XDG_CACHE_HOME/linus/gpu-kernels`, else `~/.cache/linus/gpu-kernels`. *(expert)* |

**`output`**: Optional outputs.

| key | values | default | meaning |
|---|---|---|---|
| `output.verbose` | boolean | `false` | The SCF's progress lines: on stderr from the command line, in the output's `logs` in process (Python), which never prints. |
| `output.orbitals` | boolean | `false` | The MO coefficients in the result (`mo_coefficients_alpha`/`_beta`, nbf x nmo). |
| `output.hamiltonian` | boolean | `false` | The Fock (Kohn-Sham) matrices of the SCF's last build in the result (`hamiltonian_alpha`/`_beta`, nbf x nbf), with the overlap: the operator of one full build at the last density, whose generalized eigenvectors are the returned orbitals (H C = S C diag(eps)). *(expert)* |
| `output.overlap` | boolean | `false` | The overlap matrix S of the basis functions in the result (`overlap`, nbf x nbf). *(expert)* |
| `output.core_hamiltonian` | boolean | `false` | The core Hamiltonian in the result (`core_hamiltonian`, nbf x nbf): the kinetic energy, the nuclei's attraction with their effective charges and the ECPs, whose trace with the total density is the energy's `one_electron` component. *(expert)* |
| `output.molden` | off \| spherical \| cartesian | `off` | A Molden file of the orbitals in the result (`files["orbitals.molden"]`; the CLI's `--molden FILE` writes it): `spherical` or `cartesian` functions, or `off`. |

**`accuracy`**: Accuracy targets.

| key | values | default | meaning |
|---|---|---|---|
| `accuracy.energy` | number (Eh) | `none` | The energy's (Eh). |
| `accuracy.gradient` | number (Eh/bohr) | `none` | The largest gradient component's. |
| `accuracy.hessian` | number (Eh/bohr^2) | `none` | The largest Hessian element's. |
<!-- settings:end -->

Memory (`memory_gib` = M): the DF tensors of the SCF may take 0.75 M less the memory reserved for
the PCM's dense matrices (built while the tensors are held: 2n² doubles for C-PCM and COSMO,
6n² for IEF-PCM, SS(V)PE and SMD, n surface points); the H-bar blocks of RI Hessians at most
M / 4; the XC-kernel accumulators min(1024 MB, M/8); the response buffers min(1 GiB, M/8).
`auto`, M is the machine's fixed memory, the total RAM or the cgroup limit if lower (a
container's or systemd scope's MemoryMax), with no fixed ceiling: the DF tensors 0.75 M less the
reservations, and also at most 90% of the memory free when the SCF starts less the
reservations (MemAvailable, within a capped cgroup's headroom: the guard, which binds only on a
node too full for 0.75 M); the H-bar blocks 1500 MB to a quarter of M; 1024 MB and 1 GiB for the
others. `resolved.record.memory` reports the limit and its `source` (`memory_gib`, `cgroup limit` or
`total RAM`), the DF budget and what set it (`df_tensor_budget_source`: the limit's source, or
`available memory` when the guard lowered it), the free memory read (`available_gib`) and the
`reservations`; [usage.md](usage.md#memory-and-storage-modes) has the storage modes this
chooses and what they mean for reproducibility.

ORCA presets: `orca:N` grids, ORCA's NL grids and the `orca` PCM preset (`--set
solvent.preset=orca`) take parameters that ORCA 6.0.0 does not publish, determined from its printed
output (`data/orca/` of the source tree, compiled into every build; included with permission of
FAccTs GmbH). No default uses them.

## Environment

| variable | effect |
|---|---|
| `LINUS_BASIS_DIR` | directories of basis set files (separated as in `PATH`), searched after `basis_dir` and before the source tree of a development build and the installation's sets |
| `LINUS_SOLVENT_TABLE` | a full CRC permittivity table (qcref's `data/solvents/crc.csv`) that solvent names outside the shipped table resolve through, with the same rules (the Python package finds qcref's itself); `resolved.record.solvent.eps_table` names it when a solvent came from it |
| `LINUS_ISA` | `generic`, `avx2` or `avx512`: lowers the instruction set of the dispatched kernels (never above what the CPU runs; for tests and comparisons) |
| `LINUS_COSX_LOG` | set: one line per seminumerical-exchange build on stderr |
| `LINUS_GPU_MEMORY_GIB` | GPU backend: the default of `device_memory_gib`, the device memory (GiB) one process plans against, for processes sharing a card (unset: 95 % of what is free when the device opens; [gpu.md](gpu.md#sharing-a-gpu)) |
| `LINUS_GPU_CACHE` | GPU backend: the default of `backend.kernel_cache`, the directory of the compiled-kernel cache (`off`: none; unset: `$XDG_CACHE_HOME/linus/gpu-kernels` or `~/.cache/linus/gpu-kernels`; [gpu.md](gpu.md#building)) |
| `LINUS_GPU_KERNEL_DUMP` | GPU backend, debugging: a directory that receives each compiled kernel module's source (`<key>.cu`) and its name and options (`<key>.json`) |
| `LINUS_SCF_TRACE` | debugging: a file the SCF loops append one JSON line per iteration to, with state `scf.history` does not show (host loop: the DIIS subspace size and drop rule, the ADIIS and SOSCF history sizes behind the step; device loop: the DIIS subspace size and whether the level shift was on) |
| `LINUS_KERNEL`, `LINUS_NO_GEN`, `LINUS_ECP_LEVEL` | study switches of the integral engine; they act only in builds with the cargo feature `dev-switches` (`--features linus/dev-switches`) and are ignored with a warning otherwise |

Invalid values are ignored with one warning on stderr (`linus: ignoring NAME="value" (...)`);
every `LINUS_*` variable that is set appears in `resolved.record.debug_env`.

At build time, `LINUS_DIST` set (any value) makes a distributed build: it carries no path of
the build machine, so the basis sets come only from `basis_dir`, `$LINUS_BASIS_DIR` and the
installation (`<prefix>/share/linus/basis` next to `<prefix>/bin/linus`, or the Python
package's own copy). Development builds also read the source tree's `data/basis`.

## JSON request

`linus request.json [flags]`, or `linus --json < request.json`: one JSON object with the
structure, what to compute and the settings (any key of the tables above).

| key | meaning |
|---|---|
| `structure` | as qcdata's `Structure`: `symbols` (or `numbers`) and `geometry` (n x 3, bohr), or `xyz` (the text of an XYZ file, Å); `charge` (0) and `multiplicity` (1). The CLI's `--charge` and `--mult` set `structure.charge` and `structure.multiplicity`; with an XYZ file the structure is `{"xyz": <the file>}` |
| `calctype` | `energy` (default), `gradient`, `hessian` (with the vibrational analysis and thermochemistry of the `freq` settings) |
| `freq` | the settings group of a Hessian's vibrational analysis and thermochemistry (`hessian` only): `analysis` (`harmonic`, `off`), `thermo` (`rrho` default, `qrrho`, `qrrho-hg`, `quasiharmonic`, `off`), `ir`, `temperature` (a number or a list: one thermochemistry per temperature), `pressure`, `concentration`, `scale`, `rotor_cutoff`, `qrrho_alpha`, `qrrho_bav`, `raise_below`, `imaginary_as_real`, `electronic_degeneracy`, `mass_table`, `masses`, `isotopes`, `projection`, `symmetry_number`, `symmetry_tol`, `codata`; a setting that cannot take effect is an error (`rotor_cutoff`, `qrrho_alpha`, `qrrho_bav` only with `thermo` `qrrho`/`qrrho-hg`, `raise_below` only with `quasiharmonic`, `pressure` not with `concentration`). The CLI has `--ir` (`freq.ir`, implies `--hess`), `--thermo`, `-T/--temperature`, `--pressure`, `--concentration` and `--freq-scale` (`freq.scale`); the rest by `--set freq.<key>=...` |

```json
{"structure": {"symbols": ["O", "H", "H"], "geometry": [[0, 0, 0.22], [0, 1.43, -0.89], [0, -1.43, -0.89]]},
 "calctype": "gradient", "method": "b3lyp-d3bj", "basis": "def2-tzvp", "jk": "df",
 "solvent": {"name": "water", "model": "cpcm"}, "threads": 8, "memory_gib": 16, "scf": {"soscf": true}}
```

## Result

Atomic units throughout (Eh, bohr, e bohr); sizes and memory in the units of the table after
this one. Python and qcdata carry the same values under the names given below the tables.

| key | meaning |
|---|---|
| `schema_version` | version of this result and failure layout: 1; it is raised on every breaking change (CHANGELOG.md) |
| `program`, `version`, `calctype`, `success` | |
| `numerics_epoch` | the arithmetic that produced the numbers: results of one input are bitwise the same within an epoch, and an epoch's change (CHANGELOG.md, "Changed: behaviour") can move them; `linus --version` prints it, Python has it as `linus.NUMERICS_EPOCH` |
| `energy`, `components` | total energy and its parts (`nuclear_repulsion`, `one_electron`, `coulomb`, `exchange`, `xc`, `dispersion`, and `solvation_electrostatic`, `solvation_cds` with a solvent); they sum to `energy` |
| `gradient`, `gradient_parts` | n x 3 (Eh/bohr); parts `nuclear_repulsion`, `one_electron` (incl. -W S^x), `two_electron`, `xc`, `dispersion` (and `solvent`) |
| `hessian`, `cphf_iterations` | the 3N x 3N Hessian (Eh/bohr^2) as one flat row-major list of (3N)^2 values, index (3 a + i) 3N + 3 b + j (Python `results.hessian`: a 3N x 3N array) |
| `vibrational_analysis` | a hessian's harmonic analysis (qcdata's names): `frequencies` (cm^-1, imaginary negative), `normal_modes_cartesian`, `normal_modes_mass_weighted`, reduced masses, force constants, rotational constants, `point_group`, `symmetry_number`, `detected_symmetry_number`, `symmetry_number_source`, `symmetry_tol`, `max_deviation`, IR intensities |
| `thermochemistry` | a list, one entry per `freq.temperature`: `temperature`, `pressure` or `concentration`, `zero_point_energy`, `frequency_scale_factor`, `entropy_model`, `enthalpy_model`, `qrrho_frequency`; E, H, S, Cv, Cp, G per term and in total, corrections to E_el, `energy_0k` |
| `dipole_derivatives`, `polarizability`, `polarizability_cphf_iterations` | with `ir`: 3N x 3 (dmu/dX) and 3 x 3 (a.u.), and the iterations of the electric-field CPHF/CPKS |
| `solvation` | PCM model, eps, f(eps), surface points, area, surface charge and its Gauss's-law value, SMD G_CDS |
| `converged`, `iterations` | |
| `scf` | `converged`, `iterations`, `fock_builds` (iterations plus the builds of stability analyses and of the closed-shell probe), `incremental_builds`, `convergence` (`e_tol`, `g_tol`, `g_norm`: the tolerances used), `history` (per iteration: `iter`, `energy`, `de`, `gmax` (max\|F_ai\|), `grms` (its RMS), `gnorm_pyscf` (PySCF's norm), `ddmax`, `ddrms` (the density change), `fock` (`full`, or `incremental`: of the change of the density, COSX's incremental K or the PCM's predicted charges; a converged SCF ends on a `full` one, and the last iteration of `scf.max_iter` is always `full`), `approx` (the J/K builder's approximation level: COSX's grid switch), `step` (how the next orbitals were made: `roothaan`, `diis`, `adiis`, `soscf`, `damped`, `rebuild`, `stability`, `escalation`, `converged`), `t_jk`, `t_xc`, `t_step` (seconds)), `stability` (UHF/UKS stability analyses: `iter`, `eigenvalue`, `residual`, `davidson_iterations`, `fock_builds`, `step` (the rotation taken, 0 for none), `de`, `e_above_min`, `probe` (a probe alone), `symmetry` (the converged state's abelian point group the searches ran in, `C1` without symmetry), `irrep` and `irrep_index` (the irrep of the eigenvalue's search), `irreps_searched`) |
| `restricted`, `natoms`, `nbf`, `nmo`, `nalpha`, `nbeta`, `s2` | `s2`: <S^2> of the determinant |
| `orbital_energies_alpha`/`_beta`, `occupations_alpha`/`_beta`, `homo_lumo_gap` | per spin |
| `dipole` | e bohr, about the coordinate origin |
| `grid`, `vv10` | the SCF's DFT grid; VV10 energy (in `components.xc`), NLC grid and points |
| `structure` | as qcdata's `Structure`: `symbols`, `geometry` (bohr), `charge`, `multiplicity`; and `ecp_core_electrons` (per atom) |
| `method`, `basis` | as normalized (`b3lyp-d3bj` reads as `b3lyp` with dispersion `d3bj`) |
| `timings` | seconds per phase; `total` is the whole call, `energy` its SCF stage (setup, iterations, final energy) |
| `resources` | sizes, counts and peak memory, not seconds (units below): `df_store_gib`, `df_bytes_per_elem`, `df_one_sided`, `hess_df_hbar_mb`, `hess_df_passes`, `mem_hwm_gb_*` (the process's peak resident memory at named checkpoints), `mem_device_peak_gib` (with a GPU: the most device memory the calculation's engines held at once, their buffers and grants, not the driver's own), block counts |
| `options` | the settings the request gave, in their canonical form (shorthands expanded, `b3lyp-d3bj` as `b3lyp` with `dispersion: "d3bj"`); `resolved.settings` has every setting the calculation used |
| `warnings` | what the user should know about a calculation that succeeded, as `{code, message, path}` (the CLI also prints each on stderr): `backend_fallback` (with `backend: "gpu"`, the SCF's J/K or XC ran on the CPU, and why), `df_storage_by_free_memory` (with `memory_gib` auto, the node's free memory lowered the density-fitting storage to a range-separated hybrid's one-sided coefficients), `closed_shell_unstable` (the closed-shell probe found a lower UHF/UKS or RHF/RKS solution), `loosely_converged` (qcdata's shared warning: the SCF missed its thresholds and was accepted by the plateau rule, `scf.plateau_window`; the message gives the residual and the rule, `path` is `scf.g_tol`), `mode_check_flagged` (an open-shell Hessian at a degenerate or unstable solution), `convergence_looser_than_default` (a given `scf.convergence` preset looser than the calctype's default in `e_tol` or `g_tol`, one `scf.e_tol`/`scf.g_tol` do not override), `plan_violation` (a GPU engine allocated beyond the device memory granted to it at its decision: a bug, please report it); `path` names the setting it concerns |
| `resolved` | what the calculation used, from its settings and its run record: `calctypes` (what the run covered: `energy`, and `gradient`, `hessian`, `ir` as it computed them); `settings`, the canonical settings of the run (decision D41): only the keys that take effect, every `auto` as the value it took (`grid.xc: "linus:3"`, `jk.aux`, `jk.exact_storage`, `jk.df_storage`, `jk.df_long_range`, `backend.kind`, `backend.scf_loop`, `reference`, `scf.e_tol`, `scf.g_tol`, `scf.stability_probe`, `scf.stability_closed_probe`, `dispersion`, `precision.fp64`, `precision.response`, `response.tol`, `response.iteration_aux`, `grid.kernel`, `solvent.scf_terms`, `device_memory_gib` when `$LINUS_GPU_MEMORY_GIB` set it), grids as their strings; itself a valid request: `--options` with it runs the same calculation (`auto` stays where it is a mode rather than a choice among the values, `precision.fp32` and `scf.stability`, and where the run decides per step: the device response's `precision.response`, by D37's costing); `reasons`, why each `auto` took its value, by the setting's path; and `record`, what happened (diagnostic: its keys may change without a `schema_version` change): `warm_start` (the SCF started from its lineage's history, a session's earlier orbitals, not from the input alone; false for a one-shot run), `purpose` (what the SCF was converged for), `dft`, `scf` (`guess`: where the SCF started: `source` (`sad` or `core`, the configured guess; `lineage`, its lineage's last orbitals or densities; `given`), `lineage` (the request's, null the default) and `from_lineage` (the one whose state started it: its own or a seed), `distance` (the largest atomic displacement from that lineage's last positions, bohr), `orthonormalized` (its orbitals made orthonormal at these positions) and `reason` for a fresh start (`no_history`, `distance`: beyond 1 bohr); `dropped_functions`: basis functions dropped as linearly dependent; `two_cycle_at`: the iteration at which a two-cycle handed over to quasi-Newton steps; `plateau`: `window`, `energy_spread`, `residual`, `max_residual` of an SCF accepted by the plateau rule when its escalation came due or at `max_iter` (with the warning `loosely_converged`); `escalation`: when it ran, `ran: true`, its `trigger` (`stagnation`: the lowest energy not improved for `scf.stagnation_window` iterations; `two_thirds`: 2/3 of `scf.max_iter` spent), the iteration `at` which it started, `from_iter` and `from_energy` (the lowest-energy iteration it restarted from), its `iterations` and whether it `converged`; when the SCF used `scf.max_iter` without it, `ran: false` and the `reason` (`off`, or `max_iter`: fewer than 2 iterations were left for it); `stability_probe`: the products of the stability probe the SCF ran, 0 for restricted determinants; `closed_shell_probe` of RHF/RKS determinants: `products`, the HOMO-LUMO `gap` and the `threshold`, and when it ran the `eigenvalue` (an upper bound of the lowest UHF/UKS orbital-Hessian eigenvalue; negative: a lower solution exists, with the `closed_shell_unstable` warning), `residual`, `kind` (`external`: towards a broken-symmetry UHF/UKS solution; `internal`: a lower restricted one), `unstable`, `fock_builds`, `symmetry`, `irrep`, `irrep_index`, `irreps_searched` (as in `scf.stability`) and `seconds`, else `skipped`: `gap_above_threshold`, `off`, `hf_default`, `later_geometry`, `no_virtual_orbitals` or `not_converged`, and `trigger: "gap_dropped"` when the gap trigger ran it; `probe_trigger` at an SCF that continues its lineage, where the unset probes do not run: `last_gap`, `gap` and `trigger: "gap_dropped"` when the gap fell below `scf.stability_closed_gap` since that probe or more than 0.02 Eh below its gap, and the probes ran), `df` (storage, precision, long-range mode, size, budget, auxiliary functions; on the GPU the exchange's `k_block` and the storage `rung` it set up on, with why), `memory` (`source`, `limit_gib`, the budgets, `df_tensor_budget_source`, `available_gib`, the `reservations`), `response` (Hessians: the `max_residual` reached, `single_precision`, `two_level`, `pol_max_residual` with `ir`, `mode_check` of open-shell Hessians: `eigenvalue`, `residual`, `fock_builds`, `seconds`, `symmetry`, `irrep`, `irrep_index`, `irreps_searched` (as in `scf.stability`), `threshold`, `flagged`, with the `mode_check_flagged` warning), `solvent` (`eps`, `eps_table`, SMD's `cds_contact_smoothing`), `backend` (the `device` opened, `no_device` (why `auto` found none), `precision` (the device's arithmetic: `double`, `emulated` or `mixed`), `arithmetic` (with a device: every kernel family's arithmetic, a fallback chain where it has one, and why: `scf_k`, `scf_fit`, `hess_k`, `vxc`, `grad_xc`, `hess_xc`, `vv10_scf`, `vv10_grad`), where each family of kernels ran: `jk`, `xc`, `vv10`, `gradient_jk`, `gradient_xc`, `hessian_jk`, `hessian_xc`, `response_jk`, `response_xc`, `one_electron`, `pcm`, `solvent`, `dispersion` as `cpu` or `gpu` (null when the calculation has no such part), `scf_loop` as `host` or `device`, the `fallbacks` of a GPU backend with their reasons, `emulation` (the FP64-emulation tier per family), `response` (per CPHF/CPKS response on the GPU: its J/K and kernel arithmetic, the stored exchange operator with the device memory granted to it at the setup, `operator_granted_gib`, and any `plan_violation`, and `estimate_s`: the seconds `auto` estimated for each exchange path where it weighed them) and `vv10_arithmetic` when VV10 ran on the GPU; [gpu.md](gpu.md)), `debug_env` (any `LINUS_*` environment variables that were set) |
| `mo_coefficients_alpha`/`_beta`, `hamiltonian_alpha`/`_beta`, `overlap`, `core_hamiltonian` | with `output.orbitals` (nbf x nmo) / `output.hamiltonian` (the Fock matrices of the SCF's last, full build, whose generalized eigenvectors in `overlap` are the orbitals; with `overlap`) / `output.overlap` / `output.core_hamiltonian` (nbf x nbf); PySCF's AO order. Computed after the calculation: asking for them changes no other number |
| `files` | files of the result by name, as text (absent when empty): `orbitals.molden`, the Molden file of the orbitals with `output.molden` (`spherical`: the `[5D7F]`/`[9G]` functions of a spherical basis set; `cartesian`: Cartesian functions, a spherical basis set's expanded in its Cartesian components; up to g functions; ECP atoms carry Z - n_core; restricted orbitals once with total occupations, unrestricted ones alpha then beta). `--molden FILE` writes it to FILE (output.molden as given, else the basis set's own functions) |

Units of sizes and memory (names keep their historical suffixes):

| key | unit |
|---|---|
| `memory_gib`, `device_memory_gib` (settings), `LINUS_GPU_MEMORY_GIB` | GiB (2^30 bytes) |
| `resolved.record.memory.limit_gib`, `df_tensor_budget_gib`, `available_gib`, `reservations[].gib`, `hbar_gib`, `xc_kernel_gib`, `response_slot_gib`; `resolved.record.df.store_gib`, `budget_gib`; `resources.df_store_gib`; `resolved.record.backend.device.memory_gib`, `memory_limit_gib`, `memory_cap_gib`, `resolved.record.backend.response[].operator_gib`, `operator_host_gib`; `resources.mem_device_peak_gib` | GiB |
| `resources.hess_df_hbar_mb` | MB (10^6 bytes) |
| `resources.mem_hwm_gb_*` | the kernel's VmHWM in KiB divided by 10^6 (1.024 x 10^9 bytes) |
| `resources.df_bytes_per_elem` | bytes per stored tensor element |

In Python ([python.md](python.md)) a result is a qcdata `Frame` (a session request) or
`ProgramOutput` (`linus.compute`): `results` has qcdata's fields (`energy`, `energy_components` for
`components`, `gradient`, `hessian` as a 3N x 3N array, `scf_iterations` for `iterations`,
`dipole_moment`, `scf_s_squared`, `wavefunction` (a `qcdata.Wavefunction`: its `orbitals` always
with the orbital energies, the occupations per spin and the core electrons of ECP atoms; with the
output `orbitals` the coefficients; `hamiltonian_a`/`_b`, `overlap` and `core_hamiltonian` on
request; with any of them its `basis`, the basis set they share, with its ECPs and the core
electrons each replaces), `vibrational_analysis` and `thermochemistry` as qcdata's models,
`polarizability`, `files`), `extras` the fields qcdata has no slot for yet (`gradient_components` for
`gradient_parts`), `record` the run record (`resolved.record`, with the SCF's history and
stability analyses, `sizes` for `resources`, `numerics_epoch`, `homo_lumo_gap`, `vv10`,
`cphf_iterations`, `solvation`, and the harmonic analysis's and thermochemistry's fields beyond
qcdata's models, and `accuracy` with accuracy targets), `execution` the `timings`, threads, peak
memory and linus's part of the run's environment (the GPU its backend opened in `devices`,
`peak_device_memory_gib`; a runner adds the machine's `cpu_model` and `libc`),
`logs` the progress lines of `output.verbose` (in Python they never go to stderr), and
`ProgramOutput.resolved` the `settings` and `reasons`.

## Plan

`linus mol.xyz ... --plan` prints what the calculation will do, as JSON, without running it
(Python: `Session.plan(calctype)`, this dict as the plan's `record`, with the settings in
`resolved`): the plan decides every setting it can
before any work, and the run follows it.

| key | content |
|---|---|
| `calctype`, `calctypes` | the calculation, and what its settings are checked against (as `resolved.calctypes`) |
| `settings`, `reasons` | the settings the run will have and why each `auto` takes its value, as `resolved.settings` and `resolved.reasons` report them after the run; a value only the run decides stays as requested (see `at_run_time`) |
| `placement` | where each family of kernels the method has runs: `{"on": "cpu"\|"gpu", "why": ...}`, with the reason a family stays on the CPU with a device open |
| `arithmetic` | with a device: every kernel family's arithmetic and why (as `resolved.record.backend.arithmetic`) |
| `memory` | the memory limit (`source`, `limit_gib`); `required_gib`, the smallest `memory_gib` that runs the calculation (f64 DF storage; the `required_gib` of the `host_memory` failure below it); `phases`, each phase's (`phase`: `scf`, `gradient`, `hessian`, `polarizability`) `parts` at the limit (`what`, `gib`) and its own `required_gib` ([usage.md](usage.md#memory-and-storage-modes)); with a device its plan in GiB: `limit_gib`, `others_reserve_gib`, `response_workspace_gib`, `k_operator_reserve_gib`, `w_planes_reserve_gib`, `ground_cache_leave_gib` |
| `sizes` | what the calculation scales with: `natoms`, `nbf`, `naux` (density fitting: the fitting set's functions for the calculation's purpose), `nalpha`, `nbeta` |
| `estimates` | what the calculation will need, each `{value, band, cases, model, estimate: true}`: `memory_gib` (the process's peak resident memory, at most `memory_gib` when set), `device_memory_gib` (with a device: its memory with the CUDA context, within the device's limit), `threads` (the threads it uses well: one CCD's 8 cores). `band` is the largest relative error over the `cases` the model was fitted on (at least 0.25 for three cases or fewer): an estimate for scheduling, not a bound |
| `response` | Hessians: per response (`hessian`, `polarizability` with `ir`), the CPU J/K's `precision`, a two-level response's `iteration_aux`, the `response.tol` it aims at, and the device's response J/K (`mode`, `exchange`, `why`) |
| `at_run_time` | what only the run decides, as `{what, rule}`: the storage of the DF tensors (the free memory with `memory_gib` auto), a device setup's fallback, the SCF loop, the probes at a later geometry, the device response's exchange under `precision.response: auto` |
| `deferred` | the settings `settings` leaves `auto` because the run decides them, as dotted paths (`jk.df_long_range`, `backend.scf_loop`, `precision.response`; qcdata's `Resolved.deferred`); every other `auto` the plan resolves |

Changes to the request and the result are listed in [CHANGELOG.md](../CHANGELOG.md).
