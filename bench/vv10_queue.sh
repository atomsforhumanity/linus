#!/usr/bin/env bash
# The remaining VV10 runs, one heavy job at a time on one CCD (the node was ~2x oversubscribed):
#   bench/vv10_queue.sh <slice> <step>...   steps: nogr psi4 gridstudy functionals funcu1 funcu1dg2 u1an39 u1grad refs refla18 refu1
#   attrib pyfd big ts61 ts61ri ts61orca ts61cosx ts61g ts61tzvp (labels need --label=-dgN: argparse reads -dgN as a flag)
set -u
here="$(cd "$(dirname "$0")/.." && pwd)"
MAIN=${LINUS_MAIN:-$(dirname "$(git -C "$here" rev-parse --path-format=absolute --git-common-dir)")}  # the main checkout (bench/bench_paths.py)
py=${LINUS_REFENV:-$MAIN/.refenv}/bin/python
slc="$1"; shift
bin="$here/bench/scratch/linus-vv10"
bench() { "$py" "$here/bench/vv10_bench.py" --slice "$slc" --mem 48 --linus-bin "$bin" "$@"; }
for step in "$@"; do
  echo "=== $step $(date +%H:%M:%S)"
  case "$step" in
    nogr) "$py" "$here/bench/vv10_pyscf_matrix.py" --set nogr --slice "$slc" ;;
    psi4) "$here/bench/vv10_psi4_runs.sh" "$slc" ;;
    gridstudy)
      for g in linus:2 linus:3 pyscf:3 pyscf:7; do
        bench --tag 2026-11-07-vv10/grid-study --cases u1an39:def2-tzvp --methods wb97m-d3bj --calcs gradient --programs linus-ri --grid "$g" --label "@${g/:/}"
      done ;;
    functionals)
      for mol in la18 u1an39; do
        bench --tag 2026-11-07-vv10 --cases "$mol:def2-tzvp" --methods wb97x-d3bj,wb97m-d3bj,wb97m-v --calcs energy,gradient --programs linus-cosx,linus-ri --defgrid 3 --label=-dg3
        bench --tag 2026-11-07-vv10 --cases "$mol:def2-tzvp" --methods wb97x-d3bj --calcs energy,gradient --programs linus-cosx,linus-ri --defgrid 2 --label=-dg2
      done ;;
    ts61)
      bench --tag 2026-11-07-vv10 --cases ts61:def2-tzvpd --methods wb97m-v --calcs energy,gradient --programs linus-ri,linus-cosx,orca --defgrid 3 ;;
    refla18) bench --tag 2026-11-07-vv10/ref --cases la18:def2-tzvpd --methods wb97m-v --calcs gradient --programs linus-exact --ref --grid pyscf:7 --nlc-grid pyscf:4 ;;
    refu1) bench --tag 2026-11-07-vv10/ref --cases u1an39:def2-tzvpd --methods wb97m-v --calcs gradient --programs linus-exact --ref --grid pyscf:7 --nlc-grid pyscf:4 ;;
    funcu1)
      bench --tag 2026-11-07-vv10 --cases u1an39:def2-tzvp --methods wb97x-d3bj,wb97m-d3bj,wb97m-v --calcs energy,gradient --programs linus-cosx,linus-ri --defgrid 3 --label=-dg3
      bench --tag 2026-11-07-vv10 --cases u1an39:def2-tzvp --methods wb97x-d3bj --calcs energy,gradient --programs linus-cosx,linus-ri --defgrid 2 --label=-dg2 ;;
    ts61g) bench --tag 2026-11-07-vv10 --cases ts61:def2-tzvpd --methods wb97m-v --calcs gradient --programs linus-ri,linus-cosx,orca --defgrid 3 ;;
    funcu1dg2) bench --tag 2026-11-07-vv10 --cases u1an39:def2-tzvp --methods wb97x-d3bj --calcs energy,gradient --programs linus-cosx,linus-ri --defgrid 2 --label=-dg2 ;;
    ts61ri) bench --tag 2026-11-07-vv10 --cases ts61:def2-tzvpd --methods wb97m-v --calcs energy,gradient --programs linus-ri --defgrid 3 ;;
    ts61orca) bench --tag 2026-11-07-vv10 --cases ts61:def2-tzvpd --methods wb97m-v --calcs energy,gradient --programs orca --defgrid 3 ;;
    u1grad) bench --tag 2026-11-07-vv10 --cases u1an39:def2-tzvpd --methods wb97m-v --calcs gradient --programs linus-ri,orca --defgrid 3 ;;
    ts61cosx) bench --tag 2026-11-07-vv10 --cases ts61:def2-tzvpd --methods wb97m-v --calcs gradient --programs linus-cosx --defgrid 3 ;;
    ts61tzvp)
      bench --tag 2026-11-07-vv10 --cases ts61:def2-tzvp --methods wb97x-d3bj,wb97m-d3bj,wb97m-v --calcs gradient --programs linus-cosx --defgrid 3 --label=-dg3
      bench --tag 2026-11-07-vv10 --cases ts61:def2-tzvp --methods wb97x-d3bj --calcs gradient --programs linus-cosx --defgrid 2 --label=-dg2 ;;
    refs)
      for mol in la18 u1an39; do
        bench --tag 2026-11-07-vv10/ref --cases "$mol:def2-tzvpd" --methods wb97m-v --calcs gradient --programs linus-exact --ref --grid pyscf:7 --nlc-grid pyscf:4
      done ;;
    big) "$py" "$here/bench/vv10_pyscf_matrix.py" --set big --slice "$slc" ;;
    u1an39) bench --tag 2026-11-07-vv10 --cases u1an39:def2-tzvpd --methods wb97m-v --calcs energy,gradient --programs linus-ri,linus-cosx,orca --defgrid 3 ;;
    attrib)
      # exact integrals on the same grids in both programs: separates RI/COSX from grid and VV10 differences
      bench --tag 2026-11-07-vv10/attrib --cases la18:def2-tzvpd --methods wb97m-v --calcs gradient --programs linus-exact,orca-exact --defgrid 3 ;;
    pyfd)
      for x in "tests/molecules/ch3sh.xyz def2-svp b97m-v --charge 1 --mult 2" "bench/results/2026-11-07-smd-cds/molecules/acetate.xyz def2-svp wb97m-v --charge -1"; do
        systemd-run --user --scope -q -p MemoryMax=16G -p MemorySwapMax=0 -p OOMPolicy=continue ~/.local/bin/slice "$slc" "$py" "$here/reference/checks/pyscf_vv10_grad_fd.py" $x \
          >> "$here/bench/results/2026-11-07-vv10/pyscf_vv10_grad_fd.jsonl"
      done ;;
  esac
done
echo "=== done $(date +%H:%M:%S)"
