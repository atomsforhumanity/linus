#!/usr/bin/env bash
# XC grid sensitivity of wB97X-D3BJ vs wB97M-D3BJ (wB97M-V's semilocal part), def2-TZVP, RI-JK:
# energy and gradient on ORCA DefGrid1-3, linus:2/3, pyscf:3, against pyscf:7.
# usage: bench/xc_grid_study.sh <slice> <mol> [<mol> ...]   (results: bench/results/2026-11-07-vv10/grid-study)
set -u
here="$(cd "$(dirname "$0")/.." && pwd)"
MAIN=${LINUS_MAIN:-$(dirname "$(git -C "$here" rev-parse --path-format=absolute --git-common-dir)")}  # the main checkout (bench/bench_paths.py)
py=${LINUS_REFENV:-$MAIN/.refenv}/bin/python
slc="$1"; shift
for mol in "$@"; do
  for m in wb97x-d3bj wb97m-d3bj; do
    for g in orca:1 orca:2 orca:3 linus:2 linus:3 pyscf:3 pyscf:7; do
      "$py" "$here/bench/vv10_bench.py" --tag 2026-11-07-vv10/grid-study --cases "$mol:def2-tzvp" \
        --methods "$m" --calcs gradient --programs linus-ri --grid "$g" --label "@${g/:/}" --slice "$slc" --mem 40
    done
  done
done
