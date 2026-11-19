#!/usr/bin/env python3
"""Turn a build tree's Python package into the GPU distribution `afh-linus-cuda13`.

linus ships two wheels of the same version, both importing `linus` (as CuPy does): `afh-linus`,
the CPU build, and `afh-linus-cuda13`, built with the `gpu` cargo feature for CUDA 13. maturin
takes the distribution name from crates/linus-py/pyproject.toml, so the GPU build renames it in
its own copy of the tree (a `git archive`, a container's build context) before building:

    python3 tools/cuda13_distribution.py [--undo] [ROOT]   # ROOT: the tree (default: this one)
    cd crates/linus-py && maturin build --profile dist --features gpu

`--undo` names it `afh-linus` again (a tree that builds both wheels, as docker/Dockerfile.cpu's
shared build stage does).

The version stays the workspace's (no local label such as +cuda13, which PyPI refuses); at run
time `linus.GPU_BUILD` tells the builds apart. Never commit the renamed pyproject.toml.
Exit status 1 if the file does not have exactly one name and one description line of the build
it renames from.
"""

import sys
from pathlib import Path

DESCRIPTION = "Hartree-Fock and Kohn-Sham DFT energies, analytic gradients, Hessians and frequencies, in-process"
CPU = ('name = "afh-linus"\n', f'description = "{DESCRIPTION}"\n')
GPU = ('name = "afh-linus-cuda13"\n', f'description = "{DESCRIPTION}, on the CPU or an NVIDIA GPU (CUDA 13)"\n')


def main() -> int:
    args = sys.argv[1:]
    undo = "--undo" in args
    args = [a for a in args if a != "--undo"]
    root = Path(args[0]) if args else Path(__file__).resolve().parent.parent
    path = root / "crates" / "linus-py" / "pyproject.toml"
    text = path.read_text()
    for old, new in zip(*((GPU, CPU) if undo else (CPU, GPU))):
        if text.count(old) != 1:
            print(f"{path}: expected exactly one line {old.strip()!r}", file=sys.stderr)
            return 1
        text = text.replace(old, new)
    path.write_text(text)
    print(f"{path}: distribution {'afh-linus' if undo else 'afh-linus-cuda13'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
