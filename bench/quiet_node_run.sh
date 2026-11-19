#!/usr/bin/env bash
# The definitive quiet-node comparison of linus with ORCA 6.0.0, PySCF 2.15 and Psi4 1.11
# (bench/run_bench.py; tables with bench/summarize_bench.py).
#
#   bench/quiet_node_run.sh preflight [TAG]   la18 at every level and variant, u1an39 at one level,
#                                             pinned (slice bench) only: proves the harness on a busy node
#   bench/quiet_node_run.sh smoke [TAG]       every la18 level/variant/calculation once, untimed, on SMOKE_LAYOUT
#                                             (default slice:ccd3), no lock: a functional check
#   bench/quiet_node_run.sh estimate [TAG]    wall time of core and extended, from TAG's timings (default: newest preflight)
#   bench/quiet_node_run.sh core [TAG]        the headline subset (needs CCDs 4-7 free: unpinned, spread, scale64)
#   bench/quiet_node_run.sh full [TAG]        core + extended (`extended [TAG]` adds the rest to a core run)
#   bench/quiet_node_run.sh vv10 [TAG]        optional: wB97M-V/def2-TZVP energies and gradients, linus and ORCA only
#   bench/quiet_node_run.sh summary TAG [TAG2 ...]
#
# Not in the matrix (the harness has no solvent option and runs closed shells only): implicit
# solvent and open shells; bench/README.md cites their campaign results instead.
#
# Every phase is resumable (--resume keeps finished entries): rerun the same command after an
# interruption. Environment:
#   MEM_GB     memory cap per job (default 20; the node rule). PySCF/Psi4 DF cases whose 3-index
#              tensor can't fit are skipped and listed (at 20 GB: everything above u1an39/def2-SVP for
#              Psi4 and above ts61/def2-SVP for PySCF); MEM_GB=64 leaves out only ts113/def2-TZVP and
#              the wB97X ts61/ts113 cases for them.
#   EST_FROM   result tags whose times seed `estimate` (default: TAG and two earlier runs); TAG@LAYOUT*F
#              also uses that layout scaled by F, e.g. preflight-20261107b@slice:ccd3*0.4
#   SCALE_MEM_GB memory cap of the scaling runs (default MEM_GB): ORCA needs >= 0.5 GB per rank, so
#              32/64 ranks need 27/54 GB (ORCA is skipped there otherwise, with the reason)
#   LINUS_BIN  linus binary (default: bench/build_linus.sh builds this checkout's commit into
#              bench/scratch/bin/linus-<commit> with a build record, on `slice build`)
#   PROGRAMS   program variants (default linus,linus-pg3,orca,pyscf-fastblas,psi4: linus on orca:2 = ORCA's
#              DefGrid2 and on its default pyscf:3; every other program with its fastest BLAS. The stock-BLAS
#              variants pyscf/psi4-openblas and linus-lg2 are measured once (2026-11-07) and left out to save time)
#   HESS_PROGRAMS program variants of the DFT Hessian rows (default PROGRAMS plus linus-auto: linus at its own
#              Hessian defaults, grid linus:3 and aux auto; orca:2 is too coarse for Hessians)
#   REPS       timed repetitions per entry after the warm-up (default 2; phases that set --reps keep theirs)
#   REF_LAYOUT CPUs for the untimed accuracy references (default cpus:16-31: CCDs 2-3, 16 threads)
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
MAIN=${LINUS_MAIN:-$(dirname "$(git -C "$ROOT" rev-parse --path-format=absolute --git-common-dir)")}  # the main checkout (bench/bench_paths.py)
PY=${LINUS_REFENV:-$MAIN/.refenv}/bin/python
MODE=${1:-}
TAG=${2:-}
MEM=${MEM_GB:-20}
PROGS=${PROGRAMS:-linus,linus-pg3,orca,pyscf-fastblas,psi4}
HPROGS=${HESS_PROGRAMS:-$PROGS,linus-auto}
REPS=${REPS:-2}
LEVELS_ALL=b3lyp-svp,b3lyp-tzvp,pbe-svp,pbe-tzvp,hf-svp-exact,wb97x-d3bj-tzvp
REF_LAYOUT=${REF_LAYOUT:-cpus:16-31}
RB=("$PY" bench/run_bench.py)

setup() {
    if [ -z "${LINUS_BIN:-}" ]; then
        # a binary of its own (bench/scratch/bin/linus-<commit>), built from this checkout's committed
        # sources, with a build record; never the shared target/release/linus another build may replace
        LINUS_BIN=$(bench/build_linus.sh)
    fi
    [ -e "$LINUS_BIN.build.json" ] || echo "warning: $LINUS_BIN has no build record; its commit is read from git at run time" >&2
    [ -e bench/scratch/fastblas/libopenblas-omp-0.3.34.so ] || bench/setup_fastblas.sh
    COMMON=(--tag "$TAG" --mem-gb "$MEM" --linus-bin "$LINUS_BIN")
}

run() { "${RB[@]}" run "${COMMON[@]}" --resume --reps "$REPS" "$@"; }
refs() { "${RB[@]}" refs "${COMMON[@]}" --mem-gb "${REF_MEM_GB:-$MEM}" "$@"; }

# The timed matrix. `core` is what the finding needs first (pinned la18 at every level, u1an39 at
# B3LYP/SVP and the TZVP B3LYP/wB97X energies, la18 Hessians incl. wB97X-D3BJ/SVP (linus RI-JK vs
# ORCA's analytic RIJCOSX Freq), the ECP complex PdCl2(PMe2Ph)2, ts61 B3LYP/SVP, unpinned/spread
# and scaling on u1an39); `extended` adds the rest; `vv10` is optional.
# `phases core|extended|vv10|full run|plan` runs or estimates.
# ts113: ORCA RI-JK (its slow path) and PySCF/Psi4 exact-integral HF are left out (hours each).
BIG_SKIP='ts113:.*:orca:rijk,ts113:hf-svp-exact:.*:(pyscf|psi4).*:exact'
BEST=linus,linus-pg3,orca,pyscf-fastblas,psi4   # each program with its fastest BLAS (Psi4: MKL)
core() {
    local R=$1
    # pinned (slice bench: CCD0, 8 cores): the main tables
    $R --layouts pinned --mols la18 --levels "$LEVELS_ALL" --calcs energy,gradient --programs "$PROGS"
    $R --layouts pinned --mols u1an39 --levels b3lyp-svp --calcs energy,gradient --programs "$PROGS"
    $R --layouts pinned --mols u1an39 --levels b3lyp-tzvp,wb97x-d3bj-tzvp --calcs energy --programs "$PROGS" --reps 2 --warmup 0
    $R --layouts pinned --mols la18 --levels b3lyp-svp,wb97x-d3bj-svp --calcs hessian --programs "$HPROGS"
    $R --layouts pinned --mols la18 --levels hf-svp-exact --calcs hessian --programs "$PROGS"
    # effective core potentials (Pd: def2 ECP28MWB), 43 atoms
    $R --layouts pinned --mols pdcl2pme2ph2 --levels b3lyp-svp --calcs energy,gradient --programs "$PROGS"
    $R --layouts pinned --mols pdcl2pme2ph2 --levels b3lyp-tzvp --calcs energy --programs "$PROGS" --reps 2
    $R --layouts pinned --mols ts61 --levels b3lyp-svp --calcs energy --programs "$PROGS" --reps 2
    # unpinned (OS-scheduled) and spread (one core per CCD): needs CCDs 4-7
    $R --layouts unpinned,spread --allow-all-ccds --mols u1an39 --levels b3lyp-svp --calcs energy --programs "$PROGS"
    # scaling on physical cores 0-15 / 0-31 / 0-63, each program with its fastest BLAS
    $R --layouts scale16,scale32,scale64 --allow-all-ccds --mols u1an39 --levels b3lyp-tzvp --calcs energy --programs "$BEST" --reps 2 --mem-gb "${SCALE_MEM_GB:-$MEM}"
}
extended() {
    local R=$1
    $R --layouts pinned --mols u1an39 --levels pbe-svp,pbe-tzvp,hf-svp-exact --calcs energy,gradient --programs "$PROGS" --reps 2 --warmup 0
    $R --layouts pinned --mols u1an39 --levels b3lyp-tzvp,wb97x-d3bj-tzvp --calcs gradient --programs "$PROGS" --reps 2 --warmup 0
    $R --layouts pinned --mols la18 --levels b3lyp-tzvp,pbe-svp --calcs hessian --programs "$HPROGS"
    $R --layouts pinned --mols u1an39 --levels b3lyp-svp,wb97x-d3bj-svp --calcs hessian --programs "$HPROGS" --reps 2 --warmup 0
    $R --layouts pinned --mols ts61 --levels b3lyp-tzvp,wb97x-d3bj-tzvp,pbe-svp,pbe-tzvp,hf-svp-exact --calcs energy --programs "$PROGS" --reps 2
    $R --layouts pinned --mols ts61 --levels b3lyp-svp,pbe-svp --calcs gradient --programs "$PROGS" --reps 2
    $R --layouts pinned --mols ts113 --levels b3lyp-svp,pbe-svp,hf-svp-exact,b3lyp-tzvp,wb97x-d3bj-tzvp --calcs energy --programs "$PROGS" --reps 2 --warmup 0 --skip-if "$BIG_SKIP"
    $R --layouts unpinned,spread --allow-all-ccds --mols la18,ts61 --levels b3lyp-svp,b3lyp-tzvp --calcs energy --programs "$PROGS"
    $R --layouts unpinned,spread --allow-all-ccds --mols u1an39 --levels b3lyp-tzvp --calcs energy --programs "$PROGS"
    $R --layouts unpinned,spread --allow-all-ccds --mols la18,u1an39 --levels b3lyp-svp --calcs gradient --programs "$PROGS"
    $R --layouts scale16,scale32,scale64 --allow-all-ccds --mols ts61 --levels b3lyp-tzvp --calcs energy --programs "$BEST" --reps 2 --mem-gb "${SCALE_MEM_GB:-$MEM}"
}
vv10() {  # optional: VV10 (wB97M-V), linus and ORCA only (run_bench.py skips PySCF and Psi4 at this level)
    local R=$1
    $R --layouts pinned --mols la18,u1an39 --levels wb97m-v-tzvp --calcs energy,gradient --programs "$PROGS" --reps 2
}
phases() {  # phases core|extended|vv10|full run|plan
    case $1 in core) core "$2" ;; extended) extended "$2" ;; vv10) vv10 "$2" ;; *) core "$2"; extended "$2" ;; esac
}

case "$MODE" in
preflight)
    # busy node: timed jobs capped at 14 GB and the references (run alongside on CCD2) at 6 GB,
    # inside the agent's 20 GB budget
    TAG=${TAG:-preflight-$(date +%Y%m%d-%H%M)}
    MEM=${MEM_GB:-14}
    REF_MEM_GB=${REF_MEM_GB:-6}
    setup
    {
        refs --layout slice:ccd2 --mols la18 --levels "$LEVELS_ALL" --calcs energy,gradient --programs linus,linus-lg2,linus-pg3,orca,pyscf,psi4
        refs --layout slice:ccd2 --mols la18 --levels b3lyp-svp,pbe-svp,hf-svp-exact,wb97x-d3bj-svp --calcs hessian --programs linus
        refs --layout slice:ccd2 --mols u1an39 --levels b3lyp-svp --calcs energy,gradient --programs linus,linus-lg2,linus-pg3,orca,pyscf,psi4
    } > "bench/scratch/refs-$TAG.log" 2>&1 &
    REFS_PID=$!
    "${RB[@]}" blas "${COMMON[@]}" --programs "$PROGS" --layout pinned
    # u1an39 at one level first: it decides the fair-BLAS variants and anchors the size scaling
    # (the rejected pthreads-OpenBLAS variants: add PROGRAMS=...,pyscf-pthreads1,psi4-pthreads)
    run --layouts pinned --mols u1an39 --levels b3lyp-svp --calcs energy,gradient --programs "$PROGS,pyscf-haswell"
    # la18: every level and variant; energies and gradients, Hessians at the SVP levels
    # (no per-case warm-ups for la18: the programs are warm from the u1an39 cases, and the shared
    # lock is contended on a busy node)
    run --layouts pinned --mols la18 --levels "$LEVELS_ALL" --calcs energy,gradient --programs "$PROGS" --reps 2 --warmup 0
    run --layouts pinned --mols la18 --levels b3lyp-svp,pbe-svp,wb97x-d3bj-svp --calcs hessian --programs "$HPROGS" --reps 2 --warmup 0
    run --layouts pinned --mols la18 --levels hf-svp-exact --calcs hessian --programs "$PROGS" --reps 2 --warmup 0
    wait "$REFS_PID" || echo "references: see bench/scratch/refs-$TAG.log"
    "$PY" bench/summarize_bench.py "$TAG"
    ;;
smoke)
    # functional pass (untimed, no lock) of every la18 level, variant and calculation on a non-bench
    # CCD (default slice:ccd3, 8 threads): proves each path and gives the accuracy columns when the
    # bench lock is too contended for a timed pre-flight; results go to TAG/<layout>/
    TAG=${TAG:-preflight-$(date +%Y%m%d-%H%M)}
    MEM=${MEM_GB:-7}
    setup
    S=(--layouts "${SMOKE_LAYOUT:-slice:ccd3}" --no-lock --reps 1 --warmup 0 --programs "$PROGS")
    run "${S[@]}" --mols la18 --levels "$LEVELS_ALL" --calcs energy,gradient
    run "${S[@]}" --mols la18 --levels b3lyp-svp,hf-svp-exact,wb97x-d3bj-svp --calcs hessian --programs "$HPROGS"
    run "${S[@]}" --mols pdcl2pme2ph2 --levels b3lyp-svp --calcs energy,gradient
    "$PY" bench/summarize_bench.py "$TAG"
    ;;
estimate)
    TAG=${TAG:-$(ls -td bench/results/preflight-* 2>/dev/null | head -1 | xargs -r basename)}
    E=(--tag est --mem-gb "$MEM" --estimate-from "${EST_FROM:-$TAG,2026-11-07-svp-8t,2026-11-07-unpinned-b}")
    plan() { "${RB[@]}" plan "${E[@]}" --reps "$REPS" "$@"; }
    for tier in core extended vv10; do
        echo "$tier:$([ $tier = vv10 ] && echo ' (optional)')"
        phases $tier plan | awk '/entries x/ {n=$0} /^estimated wall time/ {print "  " n "; " $0; s += $4; fflush()}
            END {printf "  timed: %.1f h (+ references ~10-20%%; beyond u1an39 extrapolated, +-50%%)\n", s}'
    done
    ;;
vv10)
    TAG=${TAG:-quiet-$(date +%Y%m%d)}
    setup
    phases vv10 run
    refs --layout "$REF_LAYOUT" --mols la18,u1an39 --levels wb97m-v-tzvp --calcs energy,gradient --programs linus,linus-pg3,orca
    "$PY" bench/summarize_bench.py "$TAG"
    ;;
full|core|extended)
    TIER=$MODE
    TAG=${TAG:-quiet-$(date +%Y%m%d)}
    setup
    "${RB[@]}" blas "${COMMON[@]}" --programs "$PROGS" --layout pinned
    phases "$TIER" run
    "$PY" bench/summarize_bench.py "$TAG" >/dev/null
    # accuracy references (untimed, after the timings): linus exact J/K on a fine grid for every
    # case; each program's own exact-integral run on its benchmark grid for la18, u1an39 and the Pd
    # complex (finished references are kept: rerunning a phase only adds what is missing)
    refs --layout "$REF_LAYOUT" --mols la18,u1an39 --levels "$LEVELS_ALL" --calcs energy,gradient --programs linus,linus-lg2,linus-pg3,orca,pyscf,psi4
    refs --layout "$REF_LAYOUT" --mols la18 --levels b3lyp-svp,hf-svp-exact,wb97x-d3bj-svp --calcs hessian --programs linus
    refs --layout "$REF_LAYOUT" --mols pdcl2pme2ph2 --levels b3lyp-svp --calcs energy,gradient --programs linus,linus-pg3,orca,pyscf,psi4
    refs --layout "$REF_LAYOUT" --mols pdcl2pme2ph2 --levels b3lyp-tzvp --calcs energy --programs linus,linus-pg3,orca,pyscf,psi4
    refs --layout "$REF_LAYOUT" --mols ts61 --levels b3lyp-svp --calcs energy --programs linus,linus-lg2,linus-pg3
    if [ "$TIER" != core ]; then
        refs --layout "$REF_LAYOUT" --mols ts61,ts113 --levels "$LEVELS_ALL" --calcs energy,gradient --programs linus,linus-lg2,linus-pg3
        refs --layout "$REF_LAYOUT" --mols la18 --levels b3lyp-tzvp,pbe-svp --calcs hessian --programs linus
        refs --layout "$REF_LAYOUT" --mols u1an39 --levels b3lyp-svp,wb97x-d3bj-svp --calcs hessian --programs linus
    fi
    "$PY" bench/summarize_bench.py "$TAG"
    ;;
summary)
    shift
    "$PY" bench/summarize_bench.py "$@"
    ;;
*)
    sed -n '2,20p' "$0"
    exit 2
    ;;
esac
