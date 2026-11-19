# Benchmarks

Two kinds of scripts live here: the **definitive harness** (one matrix, fair settings, run on a
quiet node for the published comparison) and the **campaign scripts** of the development work,
each with a results directory under `results/` that the engineering log cites (directories are
never renamed or merged). Every results directory has a README (purpose, commit, script,
conditions, status) or a `conditions.json`.

Shared rules: timed runs hold the bench lock (`bench_paths.py`: `<main checkout>/bench/scratch/bench.lock`,
the same file from every worktree; `BENCH_LOCK` overrides it), run pinned to CCD0's physical cores
(`slice bench`) unless a layout says otherwise, each in its own memory-capped systemd scope; the
reference environment, the dispersion overlay and ORCA come from `LINUS_REFENV`,
`LINUS_REFENV_DISP` and `LINUS_ORCA_DIR` (defaults in `bench_paths.py`). No host names are written.
ORCA comparisons state the ORCA version (6.0.0) and the settings they ran with (AFH's ORCA
licence from FAccTs; THIRD_PARTY_NOTICES.md).

## The definitive matrix

| File | Role |
|---|---|
| `quiet_node_run.sh` | the phases: `preflight`, `smoke`, `estimate`, `core`, `extended`, `full`, `vv10` (optional), `summary` |
| `run_bench.py` | the harness (`run`, `refs`, `blas`, `plan`): fair per-program settings, accuracy references, provenance per run |
| `summarize_bench.py` | the tables (timings, ratios, integral and grid errors, frequency errors) |
| `jobs/` | `measure.py` (in-scope wall time, peak memory), `pyscf_job.py`, `psi4_job.py`, `build_record.py` |
| `build_linus.sh`, `setup_fastblas.sh`, `blas_probe.py` | the benchmark's own linus binary with a build record; the fair-BLAS bundle for PySCF/Psi4; BLAS checks |

Levels (`run_bench.py` `LEVELS`; approaches per program in brackets):

| Level | Method / basis | Approaches | Notes |
|---|---|---|---|
| `b3lyp-svp`, `b3lyp-tzvp` | B3LYP (VWN-RPA, ORCA `B3LYP/G`) / def2-SVP, def2-TZVP | RI-JK, RIJ+COSX | |
| `pbe-svp`, `pbe-tzvp` | PBE (libxc constants in ORCA) | RI-J | |
| `hf-svp-exact` | HF / def2-SVP | exact integrals | |
| `wb97x-d3bj-svp`, `wb97x-d3bj-tzvp` | ωB97X-D3BJ (ωB97X-V semilocal, VV10 off, two-body D3(BJ)) | RI-JK, RIJ+COSX | PySCF and Psi4 `wb97x-d3bj` through the dispersion overlay; SVP: la18 Hessians (core), u1an39 (extended) |
| `wb97m-v-tzvp` (optional) | ωB97M-V / def2-TZVP | RI-JK, RIJ+COSX | linus and ORCA only; `quiet_node_run.sh vv10` |

Molecules (`molecules/`): `la18` (L-lactide, 18 atoms), `u1an39` (39-atom anion, charge -1),
`ts61` and `ts113` (transition states), `pdcl2pme2ph2` (trans-PdCl2(PMe2Ph)2, 43 atoms, Pd ECP;
B3LYP/def2-SVP energy and gradient, def2-TZVP energy). linus variants: `linus` (grid orca:2),
`linus-pg3` (pyscf:3), `linus-lg2` (linus:2), `linus-auto` (its own defaults: linus:3 and aux
auto for Hessians; the Hessian rows). Layouts: pinned, unpinned, spread, scale16/32/64.

Not in the matrix (the harness has no solvent option and runs closed shells only); cite the
campaign results: implicit solvent in `results/2026-11-07-pcm` (C-PCM/IEF-PCM/COSMO/SS(V)PE,
PySCF and Psi4/ORCA model comparisons, timings) and `results/2026-11-07-smd-cds` (SMD); open shells
in `results/2026-11-07-openshell-scf` (SCF) and `results/2026-11-07-uhf-hess` (UKS Hessians).

Status: pre-flights done (`results/preflight-20261107`, `results/preflight-20261107b`); the
definitive quiet-node run is pending (`bench/quiet_node_run.sh full TAG` once the node is free).

## Campaign scripts and their results

| Campaign | Scripts | Results (`results/`) | Status |
|---|---|---|---|
| Exact and DF J/K builds | `run_jk_bench.py`, `run_df_bench.py`, `summarize_df_bench.py`, `orca_jk.py`, `pyscf_jk.py`, `pyscf_df_bench.py`, `pyscf_exact_rhf.py`, `energy_check.py`, `energy_terms.py`, `class_profile.py`, `compare_mats.py`, `screening_error.py`, `psi4_jk_traces.py` | `jk_*.json`, `df_jk_*.json`, `energy_check*.json`, `class_profiles.json`, `2026-11-07-fastjk-digest`, `2026-11-07-fastjk-highl-profiles.json` | historical |
| SCF convergence | `scf_compare.py`, `scf_compare_table.py`, `scf_runs.py`, `scf_set_compare.py`, `scf_hist.py`, `scf_trace_pyscf.py`, `pyscf_hist.py`, `orca_scf_trace.py`, `conv_accuracy.py`, `hess_conv_accuracy.py`, `scf_change_validate.py`, `det_compare.py` | `2026-11-07-scf-conv`, `-scf-conv-headline`, `-scf-conv-unpinned` | historical; source of the SCF thresholds |
| Open-shell SCF | `openshell_scf.py`, `openshell_pyscf.py`, `openshell_summary.py` | `2026-11-07-openshell-scf` | current |
| Early matrix runs | `run_bench.py` (earlier format), `quiet_node_run.sh` | `2026-11-07-svp-8t`, `-svp-8t-unpinned`, `-unpinned-b` | historical; seed `estimate` |
| RIJCOSX | `grad_bench.py`, `cosx_scf_bench.py`, `cosx_scf_summary.py` | `2026-11-07-rijcosx-grad`, `-cosx-perf`, `-cosx-perf-unpinned`, `-cosx-perf-spread`, `-cosx-scf`, `-cosx-grad-large` | historical (cosx-perf: latest COSX timings) |
| RI-JK gradients, RI-K beyond memory | `rik_bench.py`, `rik_summary.py` | `2026-11-07-df-grad`, `-rik-scale`, `-rik-scale-ref` | historical |
| Range-separated hybrids | `rsh_bench.py`, `rsh_summary.py`, `rsh_hess_bench.sh`, `rsh_hess_summary.py`, `hess_compare.py` | `2026-11-07-wb97x-d3bj`, `-wb97x-d3bj-ref`, `-rsh-hess` | rsh-hess current |
| Hessians | `ri_hess_scaling.py`, `uks_hess_bench.py`, `hess_accuracy/`, `bench_xc_hess_pyscf.py`, `bench_grad.py` | `2026-11-07-ri-hess-scaling`, `-uhf-hess`, `-hess-accuracy` | hess-accuracy current (Hessian defaults) |
| XC and grids | (examples `xc_profile`, `grid_study`; docs/grids.md) | `2026-11-07-xc`, `-grid-speed`, `-orca-grid` | grid-speed, orca-grid current |
| Vibrational analysis | `vib/validate.py`, `vib/refs.py`, `vib/summarize.py` | `2026-11-07-vib` | current |
| Implicit solvent | `pcm_bench.py`, `pcm_lebedev.py`, `pcm_scan.py`, `pcm_validate.py`, `pcm_orca.py`, `pcm_orca_compare.py`, `pcm_psi4.py`, `pcm_psi4_compare.py`, `pcm_summary.py` | `2026-11-07-pcm`, `-smd-cds` | current (see docs/solvation.md) |
| VV10 | `vv10_bench.py`, `vv10_summary.py`, `vv10_pyscf_matrix.py`, `vv10_runs.sh`, `vv10_queue.sh`, `vv10_psi4_runs.sh`, `xc_grid_study.sh`, `cosx_tzvpd_control.py` | `2026-11-07-vv10` | current |
| ECPs | `run_bench.py`, `molecules/build_pd_complex.py` | `ecp-pd-20261107`, `ecp-pd-20261107-final` | final: latest |
| End-to-end workflows | `workflows/` (`linus_workflows.py`, `orca_workflows.py`, `guess_reuse.py`, `overhead.py`, `screen.py`, `summarize.py`) | `2026-11-07-workflows` | current |
| Definitive harness pre-flights | `quiet_node_run.sh preflight`/`smoke` | `preflight-20261107`, `preflight-20261107b` | superseded by the definitive run |

"historical": the result describes the code of its commit; later work changed that code, so
rerun before comparing with main. Campaign scripts are kept as they ran (only their lock and
path defaults follow `bench_paths.py`); several call examples or environment hooks that are
being removed and will not run unchanged on main.
