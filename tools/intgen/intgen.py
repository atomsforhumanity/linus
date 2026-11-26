#!/usr/bin/env python3
"""Integral kernel generator for linus: McMurchie-Davidson math, an expression IR with schedules,
and emitters for Rust (linus-ints) and CUDA C (linus-gpu).

    python3 tools/intgen/intgen.py            # writes crates/linus-ints/src/generated/*.rs
    python3 tools/intgen/intgen.py --check    # regenerates in memory, fails if a file differs

Three layers (the algorithm/schedule split):
- (a) math, mdmath.py: Hermite and Cartesian index sets, solid harmonics and E-matrix patterns in
  exact rational arithmetic, the Hermite R recursion rules;
- (b) IR, ir.py: an expression DAG with common-subexpression elimination (hash-consing), ordered
  FMA chains for sums, operation counts and register-pressure estimates; and the schedules,
  `RustSchedule` for the CPU kernels and `CudaSchedule` for the GPU kernels (fusion, where values
  live, thread mapping, precision, unrolling);
- (c) emitters:
  - Rust (rtensor_gen.py, classes_gen.py, tiled_gen.py): crates/linus-ints/src/generated/
    - rtensor.rs: Hermite Coulomb integrals R_tuv (t+u+v <= L) for L = 0..R_MAX_L, from
      R^{(n)}_{000}, by the McMurchie-Davidson recursion with every index unrolled;
    - classes.rs: for low angular-momentum pair classes, the structural nonzero pattern of the
      SHARK E-matrices E[r][ab] (spherical functions for l >= 2, Cartesian x, y, z for p) and
      class-specialized contractions over that pattern;
    - tiled.rs: pieces of the tiled kernels for pair classes up to (gg): R in the cube layout and
      per-class T and I steps with compile-time offsets.
    All are generic over `V: Real` (f64, or 8 lanes of f64), so the same code serves the scalar
    quartet path and the batched (8 ket pairs at once) path. The restructure into layers kept
    this output byte for byte (`--check`).
  - CUDA C (cuda_eri3c.py, `cuda` subcommand): fused three-center kernels and their first
    derivatives for linus-gpu (see that module).
  - CUDA C (cuda_pcm.py, `pcm` subcommand): integrals of polarizable-continuum surface charges
    (Gaussian or point charges at the cavity's points) and their first derivatives for
    linus-gpu's PCM engine (see that module).
  - CUDA C (cuda_jengine.py, `jengine` subcommand): the Hermite J-engine of linus-gpu's
    integral-direct Coulomb matrix and its gradient (see that module).
  The three CUDA emitters share cuda_common.py: exact C literals, the C printer of IR nodes and
  the Boys stage (linus-ints' eri3c scheme, with the emitters' differences as its parameters).
  The default run also writes crates/linus-gpu/src/kernels/gridk_boys.txt, that Boys stage at
  the grid-K engine's orders, which pins the grid-K engine's own (Rust) emitter to it.

Only the standard library is needed.
"""
import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "crates/linus-ints/src/generated"


def rust_outputs():
    import classes_gen
    import rtensor_gen
    import tiled_gen
    return {
        "rtensor.rs": rtensor_gen.generate(),
        "classes.rs": classes_gen.generate(),
        "tiled.rs": tiled_gen.generate(),
        "mod.rs": rtensor_gen.HEADER + "\npub mod classes;\npub mod rtensor;\npub mod tiled;\n",
    }


def write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text() == text:
        print(f"unchanged {path} ({text.count(chr(10))} lines)", file=sys.stderr)
        return
    path.write_text(text)
    print(f"wrote {path} ({text.count(chr(10))} lines)", file=sys.stderr)


def check(files):
    bad = [str(p) for p, text in files.items() if not p.exists() or p.read_text() != text]
    for p in bad:
        print(f"differs: {p}", file=sys.stderr)
    if bad:
        print("generated files are out of date: run python3 tools/intgen/intgen.py", file=sys.stderr)
        return 1
    print(f"{len(files)} generated files up to date", file=sys.stderr)
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true", help="compare with the files on disk instead of writing")
    sub = ap.add_subparsers(dest="cmd")
    sub.add_parser("rust", help="the linus-ints kernels (default)")
    try:
        import cuda_eri3c
        cuda_eri3c.add_cli(sub)
    except ImportError:
        cuda_eri3c = None
    import cuda_pcm
    cuda_pcm.add_cli(sub)
    import cuda_jengine
    cuda_jengine.add_cli(sub)
    args = ap.parse_args(argv)
    if args.cmd in (None, "rust"):
        files = {OUT / name: text for name, text in rust_outputs().items()}
        # the grid-K engine's Boys stage pin (its emitter is linus-gpu's Rust)
        import cuda_common
        files[ROOT / "crates/linus-gpu/src/kernels/gridk_boys.txt"] = cuda_common.gridk_boys_reference()
    elif args.cmd == "pcm":
        files = cuda_pcm.run_cli(args, ROOT)
    elif args.cmd == "jengine":
        files = cuda_jengine.run_cli(args, ROOT)
    else:
        files = cuda_eri3c.run_cli(args, ROOT)
        if files is None:
            return 0
    if args.check:
        return check(files)
    for p, text in files.items():
        write(p, text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
