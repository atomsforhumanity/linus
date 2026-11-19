#!/usr/bin/env bash
# The VV10 benchmark runs (bench/results/2026-11-07-vv10), development timings on one CCD:
#   bench/vv10_runs.sh omol25 <slice>     wB97M-V/def2-TZVPD at OMol25's settings (DefGrid3), linus RI-JK and
#                                         RIJ+COSX vs ORCA 6.0.0 RIJCOSX, energy and gradient
#   bench/vv10_runs.sh functionals <slice> wB97X-D3BJ, wB97M-D3BJ, wB97M-V (def2-TZVP; wB97M-V also def2-TZVPD),
#                                         linus RI-JK and RIJ+COSX, energy and gradient, DefGrid2 and DefGrid3
set -u
here="$(cd "$(dirname "$0")/.." && pwd)"
MAIN=${LINUS_MAIN:-$(dirname "$(git -C "$here" rev-parse --path-format=absolute --git-common-dir)")}  # the main checkout (bench/bench_paths.py)
py=${LINUS_REFENV:-$MAIN/.refenv}/bin/python
what="$1"; slc="$2"
# a copy of the CLI, so that rebuilds during the series do not change the binary
bin="$here/bench/scratch/linus-vv10"
[ -x "$bin" ] || { mkdir -p "$here/bench/scratch"; cp "$here/target/release/linus" "$bin"; }
run() { "$py" "$here/bench/vv10_bench.py" --tag 2026-11-07-vv10 --slice "$slc" --mem 48 --linus-bin "$bin" "$@"; }
case "$what" in
  omol25)
    for mol in la18 u1an39 ts61; do
      run --cases "$mol:def2-tzvpd" --methods wb97m-v --calcs energy,gradient --programs linus-ri,linus-cosx,orca --defgrid 3
    done ;;
  functionals)
    for mol in la18 u1an39 ts61; do
      run --cases "$mol:def2-tzvp" --methods wb97x-d3bj,wb97m-d3bj,wb97m-v --calcs energy,gradient --programs linus-cosx,linus-ri --defgrid 3 --label=-dg3
      run --cases "$mol:def2-tzvp" --methods wb97x-d3bj,wb97m-d3bj --calcs energy,gradient --programs linus-cosx,linus-ri --defgrid 2 --label=-dg2
    done ;;
esac
