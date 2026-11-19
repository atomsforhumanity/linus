#!/usr/bin/env bash
# Psi4 1.11 model-level VV10 checks (tests/python/compare_vv10_psi4.py) -> bench/results/2026-11-07-vv10/psi4_validation.jsonl
# usage: bench/vv10_psi4_runs.sh <slice>
set -u
here="$(cd "$(dirname "$0")/.." && pwd)"
MAIN=${LINUS_MAIN:-$(dirname "$(git -C "$here" rev-parse --path-format=absolute --git-common-dir)")}  # the main checkout (bench/bench_paths.py)
py=${LINUS_REFENV:-$MAIN/.refenv}/bin/python
out="$here/bench/results/2026-11-07-vv10/psi4_validation.jsonl"
cmp() { systemd-run --user --scope -q -p MemoryMax=24G -p MemorySwapMax=0 -p OOMPolicy=continue ~/.local/bin/slice "$1" \
          "$py" "$here/tests/python/compare_vv10_psi4.py" "${@:2}" --threads 8 | tail -1 >> "$out"; }
m="$here/tests/molecules"
cmp "$1" "$m/water.xyz" def2-svp wb97m-v
cmp "$1" "$m/water.xyz" def2-tzvpd wb97m-v
cmp "$1" "$m/ch3sh.xyz" def2-svp wb97x-v
cmp "$1" "$m/ch3sh.xyz" def2-svp b97m-v --charge 1 --mult 2
cmp "$1" "$here/bench/results/2026-11-07-smd-cds/molecules/acetate.xyz" def2-svp wb97m-v --charge -1
cmp "$1" "$here/bench/molecules/la18.xyz" def2-svp wb97m-v
