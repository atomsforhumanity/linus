#!/usr/bin/env bash
# Build the "fair BLAS" bundle for PySCF and Psi4 in bench/scratch/fastblas (git-ignored).
#
# PySCF's pip wheel calls a bundled single-threaded OpenBLAS 0.3.3 that falls back to its
# "Barcelona" kernels on Zen 5; Psi4 (conda-forge) calls MKL 2025.3, which takes its AVX2 path on
# AMD. The fair alternative is conda-forge's OpenBLAS 0.3.34 *OpenMP build* (Cooperlake/AVX-512
# kernels): unlike the pthreads build in .refenv, it runs one thread when called inside an OpenMP
# parallel region (PySCF's C code and Psi4 call BLAS from inside their parallel loops) and
# OMP_NUM_THREADS threads outside, which is how both programs expect their BLAS to behave.
#
# The bundle is put in front of everything by LD_PRELOAD (run_bench.py variants pyscf-fastblas and
# psi4-openblas): first .refenv's LLVM libomp (Psi4's own OpenMP runtime; PySCF's GOMP_* calls bind
# to its GNU-compatible entry points, so the program and its BLAS share one OpenMP runtime), then
# the OpenMP OpenBLAS, whose libgomp.so.1 and libgfortran.so.5 point back into .refenv so no
# second runtime is loaded.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
MAIN=${LINUS_MAIN:-$(dirname "$(git -C "$ROOT" rev-parse --path-format=absolute --git-common-dir)")}  # the main checkout (bench/bench_paths.py)
REFENV=${LINUS_REFENV:-$MAIN/.refenv}
OUT=$ROOT/bench/scratch/fastblas
PKG='libopenblas=0.3.34=openmp_h156ec52_2'   # conda-forge build used for the AFH benchmark
MAMBA=${MAMBA:-$HOME/.local/bin/micromamba}
export MAMBA_ROOT_PREFIX=${MAMBA_ROOT_PREFIX:-$HOME/micromamba}

mkdir -p "$OUT"
if [ ! -e "$OUT/env/lib/libopenblasp-r0.3.34.so" ]; then
    "$MAMBA" create -y -q -p "$OUT/env" -c conda-forge --override-channels "$PKG"
fi
cp -f "$OUT/env/lib/libopenblasp-r0.3.34.so" "$OUT/libopenblas-omp-0.3.34.so"
ln -sfn "$REFENV/lib/libomp.so" "$OUT/libgomp.so.1"
ln -sfn "$REFENV/lib/libgfortran.so.5" "$OUT/libgfortran.so.5"
echo "LD_PRELOAD=$REFENV/lib/libomp.so:$OUT/libopenblas-omp-0.3.34.so"
