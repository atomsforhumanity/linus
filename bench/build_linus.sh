#!/usr/bin/env bash
# Build the benchmark's linus binary from this checkout's commit into a dedicated path and record
# what it was built from: bench/scratch/bin/linus-<source commit> plus <binary>.build.json
# (bench/jobs/build_record.py; run_bench.py reads it). Prints the binary's path.
#
# The build uses its own target directory (target/bench), so no other build in this checkout, and
# no other agent's, can replace the binary under a running benchmark; an existing binary for the
# same sources is reused. Refuses to build from uncommitted changes in crates/ or Cargo.*.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
if [ -n "$(git status --porcelain --untracked-files=no -- crates Cargo.toml Cargo.lock)" ]; then
    echo "build_linus.sh: uncommitted changes in crates/ or Cargo.*; commit them first" >&2
    exit 1
fi
HEAD=$(git rev-parse HEAD)
SRC=$(git log -1 --format=%H -- crates Cargo.toml Cargo.lock)
BIN=$ROOT/bench/scratch/bin/linus-${SRC:0:12}
if [ ! -x "$BIN" ] || [ ! -e "$BIN.build.json" ]; then
    mkdir -p "$ROOT/bench/scratch/bin"
    CARGO_TARGET_DIR=$ROOT/target/bench systemd-run --user --scope -q -p MemoryMax=${BUILD_MEM_GB:-16}G -p MemorySwapMax=0 \
        "$HOME/.local/bin/slice" build cargo build --release -q -p linus-cli >&2
    cp "$ROOT/target/bench/release/linus" "$BIN.tmp"
    mv "$BIN.tmp" "$BIN"
    python3 bench/jobs/build_record.py "$BIN" "$HEAD" "$SRC" >/dev/null
fi
echo "$BIN"
