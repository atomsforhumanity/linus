#!/usr/bin/env bash
# Cross-check PySCF, Psi4 and ORCA on water and NH3 (reference/README.md has the table).
# Usage: reference/crosscheck.sh [threads]     (outputs under reference/out/, git-ignored)
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.refenv/bin/python
T=${1:-8}
for mol in water nh3; do
  xyz=reference/molecules/$mol.xyz
  # HF, exact integrals: gradients in three basis sets, Hessian in def2-SVP
  for basis in def2-svp cc-pvtz def2-tzvp; do
    $PY reference/run_ref.py $xyz --method hf --basis $basis --calc gradient --threads $T
  done
  $PY reference/run_ref.py $xyz --method hf --basis def2-svp --calc hessian --threads $T
  $PY reference/run_ref.py $xyz --method b3lyp --basis def2-svp --calc hessian --threads $T --grid fine
  # DFT, exact J/K, finest grids (PySCF/Psi4 150x974 unpruned; ORCA IntAcc 8, Lebedev-770, unpruned)
  for method in b3lyp pbe; do
    $PY reference/run_ref.py $xyz --method $method --basis def2-svp --calc gradient --threads $T --grid fine
  done
  # same without PySCF grid-weight derivatives (Psi4 has none), and ORCA DefGrid3
  $PY reference/run_ref.py $xyz --method b3lyp --basis def2-svp --calc gradient --threads $T --grid fine --no-grid-response --programs pyscf --tag nogridresp
  $PY reference/run_ref.py $xyz --method b3lyp --basis def2-svp --calc gradient --threads $T --grid defgrid3 --programs orca
done
$PY reference/compare_ref.py reference/out/{water,nh3}/hf_*exact reference/out/{water,nh3}/{b3lyp,pbe}_def2-svp_*_exact_grid-fine
