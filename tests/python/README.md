# tests/python: deep checks against PySCF and Psi4

Per-feature comparison scripts: each runs linus (the `linus --json` CLI or an example binary of
this tree) and a reference program on the same molecule, basis data and grid, and prints the
differences. They are the detailed checks behind the validation matrix
(`reference/validate.py`, `reference/validation/README.md`) and the pytest suite
(`crates/linus-py/tests`); run them when you change the code they cover.

Run them with the reference environment's Python (PySCF 2.15, Psi4 1.11) from the repository
root, after building this tree (`cargo build --release --workspace --bins --lib --examples`),
inside a memory-capped scope on a shared node:

```
systemd-run --user --scope -q -p MemoryMax=16G -p MemorySwapMax=0 \
    .refenv/bin/python tests/python/compare_pcm_pyscf.py tests/molecules/water.xyz def2-svp b3lyp --grad
```

`.refenv` lives in the main checkout; in a worktree use its absolute path.

## Merge gate

```
export LINUS_PYSCF_CACHE=<dir>        # optional: reuse PySCF results between runs
export LINUS_EXAMPLES=target/test-fast/examples   # the test build's examples (default: target/release/examples)
.refenv/bin/python tests/python/compare_grad_pyscf.py tests/molecules/ch3sh.xyz def2-tzvp hf --tol-e 1e-11 --tol-g 2e-9
.refenv/bin/python tests/python/compare_hess_pyscf.py tests/molecules/nh3.xyz def2-tzvp hf 8 --tol-e 1e-11 --tol-h 3e-9
```

Both compare exact J/K (linus's examples with `--jk exact`; linus's default is df) with PySCF's
and print one line (keep its format: the gate shows it). Measured: `dE` 9.7e-13 and `max|dG|`
6.5e-10; `dE` 1.4e-14 and `max|dH|` 1.3e-9. With the `--tol-*` flags a script exits 1 and ends
its line with `FAIL: ...` when a difference is above its tolerance (`ok (...)` otherwise), which
fails the gate's step A4. The rest of the per-merge check is
`cargo test --profile test-fast --workspace`, pytest (pinned with `PYTHONPATH=<tree>/crates/linus-py/python`)
and `reference/validate.py --quick`.

## The checks

Expected agreement is the measured worst case of the validation runs. PySCF columns use linus's basis data, so the
differences are code differences.

| Script | Compares | Typical command | Expected |
|---|---|---|---|
| `compare_ints_pyscf.py` | one-electron, ERI, 3c/2c DF integrals, erf/erfc versions (`dump_ints` example) with libcint | `compare_ints_pyscf.py water.xyz def2-svp def2-universal-jkfit [--omega 0.3]` | 1e-13 |
| `compare_df_pyscf.py` | `ints`: DF integrals incl. erf (`dump_df`); `scf`: RI-HF energies (`scf --jk ri`) | `compare_df_pyscf.py scf water.xyz def2-svp def2-universal-jkfit 8` | 1e-13 |
| `compare_df_psi4.py` | RI-HF and range-separated exact exchange (wK) energies with Psi4 DF-SCF | `compare_df_psi4.py water.xyz def2-svp def2-universal-jkfit 8` | 4e-12 (HF), 1e-9 (wK) |
| `compare_grad_pyscf.py` | HF/KS analytic gradients, exact or DF (`grad` example), PySCF level-3 grid with grid response | `compare_grad_pyscf.py ch3sh.xyz def2-tzvp hf [mult] [df]` | dE 1e-12, dG 1e-9 |
| `compare_hess_pyscf.py` | analytic Hessians (R/U HF and KS, RSH, DF; `hess` example, fixed grid) | `compare_hess_pyscf.py nh3.xyz def2-tzvp hf 8 [df]`; `--mult 2 --charge 1` | HF 1.3e-9; DF + RSH informational (two RI schemes) |
| `compare_rsh_pyscf.py` | wB97X, wB97X-D3BJ, wB97M-D3BJ, CAM-B3LYP, LC-wPBE energies and gradients; `--df` against PySCF with linus's wK exchange | `compare_rsh_pyscf.py water.xyz def2-svp wb97x-d3bj [--df]` | dE 1e-13, dG 5e-10 |
| `compare_vv10_pyscf.py` | wB97M-V, wB97X-V, B97M-V energies, VV10 energy and gradients | `compare_vv10_pyscf.py water.xyz def2-svp wb97m-v [--df] [--no-grid-response]` | dE 1e-11, dG 3e-9 (PySCF's own VV10 grid response) |
| `compare_vv10_psi4.py` | VV10 energies at the model level (each program's own grids) | `compare_vv10_psi4.py water.xyz def2-svp wb97m-v` | ~1e-6 (grid) |
| `compare_pcm_pyscf.py` | C-PCM, IEF-PCM, COSMO, SS(V)PE and SMD: energies, `--grad`, `--hess`, `--df` | `compare_pcm_pyscf.py water.xyz def2-svp b3lyp --model cpcm --grad`; `--model smd --solvent water` | dE 2e-11, dG 8e-10, dH 6e-9; SMD open-shell dG 1e-3 is PySCF's CDS gradient (`reference/checks/pyscf_cds_grad_fd.py`) |
| `compare_pcm_ecp_pyscf.py` | PCM with ECP atoms (Z - n_core on the surface) | `compare_pcm_ecp_pyscf.py --cases CH3I,HI,AgCl` | dE 1e-11 |
| `ecp_pyscf_check.py` | ECP matrices, derivatives, gradient and Hessian contractions (`ecp_dump`) | `ecp_pyscf_check.py --cases HI,CH3I --basis def2-svp --what matrix,grad` | 1e-12 |
| `ecp_psi4_check.py` | ECP matrices against Psi4 (libecpint), through generalized eigenvalues of (U, S) | `ecp_psi4_check.py --cases HI --basis def2-svp` | linus-PySCF 1e-12; Psi4 differs by up to 1e-5 (its integrals) |
| `ecp_scf_check.py` | SCF energies and gradients of ECP molecules (HF, PBE, B3LYP; closed and open shells) | `ecp_scf_check.py --grad --fix-pyscf` | dE 1e-12, dG 2e-10 with `--fix-pyscf` (PySCF's ECP grid-response radii) |
| `ecp_fd_grad.py` | one `ecp_scf_check.py` case: central differences of linus energies vs linus and PySCF gradients | `ecp_fd_grad.py CsF.b3lyp.def2-tzvp` | linus-FD 3e-8 (step truncation) |
| `fd_grad_linus.py` | linus gradients vs Richardson differences of linus energies (any method, jk, solvent) | `fd_grad_linus.py water.xyz def2-svp wb97m-v --h 5e-4` | 1e-10 |
| `fd_hess_linus.py` | linus Hessians vs Richardson differences of linus gradients (KS: all atoms, symmetrized) | `fd_hess_linus.py water.xyz def2-svp b3lyp --solvent '{"eps": 7.4257}'` | 1e-9 (HF); KS see its docstring |
| `gen_deriv_fixtures.py`, `gen_ecp_fixtures.py` | write the PySCF fixtures of the linus-ints derivative and ECP tests | `gen_deriv_fixtures.py [erf]` | (fixtures) |

Molecules: `tests/molecules/*.xyz` (Å). Every script converts with linus's bohr constant
(CODATA 2022, `common.BOHR`) and gives PySCF bohr, so both programs see the same nuclei.

## Shared helpers (`common.py`)

`read_xyz` / `read_xyz_bohr`, `pyscf_basis` / `pyscf_ecp` / `pyscf_mol` (PySCF given linus's
`data/basis` files and their def2 ECPs), `pyscf_xc` (wb97x-d3bj -> PySCF wb97x-v with VV10 off,
wb97m-d3bj -> wb97m-v; linus reports the D3(BJ) part separately), `wk_get_jk` (PySCF density
fitting with linus's wK long-range exchange instead of PySCF's erf-metric fit), `linus_json`
(run `linus --json` on a request), `mol_key` and `cached` (the PySCF cache), and `psi4_cwd`.
`compare_ints_pyscf.pyscf_basis` and `compare_rsh_pyscf.wk_get_jk` stay importable for
`reference/run_ref.py` and the bench scripts. The module is imported as `common` with
tests/python on `sys.path`; don't combine it in one process with `bench/workflows/common.py`.

## PySCF reference cache

With `LINUS_PYSCF_CACHE=<dir>` the scripts that use `common.cached` (or `pyscf_ref_cache.py`)
store each PySCF result as an `.npz` under a key of everything that determines it: the molecule
(atoms, bohr positions, basis and ECP data as PySCF holds them, charge, spin), the method and
options, the auxiliary basis, a hash of the calling script's source and of `common.py` (editing a
script, a tolerance or a PySCF setting invalidates its entries), and PySCF's version. A hit skips
PySCF; the JSON output says `"pyscf_cached": true` (`compare_hess_pyscf.py` appends `(cached)`).
Unset: no caching. Cached: `compare_grad_pyscf.py`, `compare_hess_pyscf.py`, `compare_rsh_pyscf.py`,
`compare_vv10_pyscf.py`, `compare_pcm_pyscf.py`, `compare_pcm_ecp_pyscf.py`, `ecp_scf_check.py`
(and through it `ecp_fd_grad.py`). Delete the directory to start over.

## Psi4

Psi4 writes `timer.dat` (which names the host) and scratch files into the working directory, at
exit too. The Psi4 scripts (`compare_df_psi4.py`, `compare_vv10_psi4.py`, `ecp_psi4_check.py`,
`bench/psi4_jk_traces.py`) therefore move the process into a temporary directory before
importing psi4 (`common.psi4_cwd`), which is removed at exit; their path arguments are resolved
first.

## Elsewhere

Timing scripts moved to `bench/` (`bench_grad.py`, `bench_xc_hess_pyscf.py`,
`psi4_jk_traces.py`); checks of the reference programs themselves to `reference/checks/`
(`pyscf_cds_grad_fd.py`: PySCF's SMD CDS gradient vs its own finite differences;
`pyscf_vv10_grad_fd.py`: PySCF's VV10 grid-response gradient vs finite differences).
