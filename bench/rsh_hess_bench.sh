#!/bin/bash
# Range-separated hybrid Hessians (frequencies) vs ORCA 6.0.0: wB97X-D3BJ, linus RI-JK at its
# Hessian defaults (aux auto, grid auto = linus:3, first-order grid response) and ORCA
# `wB97X-D3BJ <basis> def2/J RIJCOSX DefGrid2 TightSCF Freq`, 8 threads / 8 MPI ranks on
# `slice bench`, wall time inside the shared bench lock (bench/ri_hess_scaling.py does the
# runs, frequencies and errors against the exact-J/K reference Hessians REF_DIR/<mol>_<basis>.json).
#
#   bench/rsh_hess_bench.sh OUTDIR REF_DIR LOCK [case ...]     case = mol:basis:charge:mult:prog
#
# The lock is taken inside ri_hess_scaling.py (--lock) for each run separately.
set -eu
OUT=$1; REF=$2; LOCK=$3; shift 3
ROOT=$(cd "$(dirname "$0")/.." && pwd)
MAIN=${LINUS_MAIN:-$(dirname "$(git -C "$ROOT" rev-parse --path-format=absolute --git-common-dir)")}  # the main checkout (bench/bench_paths.py)
PY=${PY:-${LINUS_REFENV:-$MAIN/.refenv}/bin/python}
mkdir -p "$OUT"
for c in "$@"; do
  IFS=: read -r mol basis charge mult prog <<< "$c"
  ref=()
  [ -f "$REF/${mol}_${basis}.json" ] && ref=(--ref "$REF/${mol}_${basis}.json")
  n=1; while [ -f "$OUT/${mol}_${basis}_${prog}_$n.json" ]; do n=$((n + 1)); done
  "$PY" "$ROOT/bench/ri_hess_scaling.py" "$prog" ${LINUS_BIN:+--linus-bin "$LINUS_BIN"} "$ROOT/bench/molecules/$mol.xyz" --out "$OUT/${mol}_${basis}_${prog}_$n.json" \
    --method wb97x-d3bj --basis "$basis" --charge "$charge" --mult "$mult" --threads 8 --slice bench --lock "$LOCK" --mem 30 "${ref[@]}"
done
