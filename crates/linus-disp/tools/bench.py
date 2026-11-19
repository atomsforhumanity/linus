#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Wall times of linus-disp (in-process, best of N) and of the s-dftd3 1.7.0 / dftd4 4.3.1
programs (whole process, best of N, OMP_NUM_THREADS as given)::

    python3 -I crates/linus-disp/tools/bench.py --threads 1,8,64 \
        crates/linus-disp/tests/molecules/la18.xyz ...
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from compare_ref import read_xyz, write_coord  # noqa: E402

CASES = [
    ("D3(BJ)", "d3bj", "b3lyp", {}, ["s-dftd3", "run", "--bj", "b3lyp"]),
    ("D3(BJ)+ATM", "d3bj", "b3lyp", {"atm": True}, ["s-dftd3", "run", "--bj", "b3lyp", "--atm"]),
    ("D4", "d4", "b3lyp", {}, ["dftd4", "run", "--func", "b3lyp"]),
]


def best_of(n, f):
    return min(f() for _ in range(n))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("molecules", nargs="+", type=Path)
    ap.add_argument("--bindir", type=Path, default=Path.home() / "micromamba/envs/linus-disp-ref/bin")
    ap.add_argument("--bin", type=Path, default=Path("target/release/examples/disp_json"))
    ap.add_argument("--threads", default="1,8,64")
    ap.add_argument("--repeat", type=int, default=5)
    ap.add_argument("--no-ref", action="store_true")
    ap.add_argument("--ref-hessian-max-atoms", type=int, default=200)
    args = ap.parse_args()
    threads = [int(t) for t in args.threads.split(",")]
    print("| molecule | N | case | order | " + " | ".join(f"linus {t} thr (s)" for t in threads)
          + " | " + " | ".join(f"ref {t} thr (s)" for t in threads) + " |")
    print("|---|---|---|---|" + "---|" * (2 * len(threads)))
    for mol in args.molecules:
        z, xyz, charge = read_xyz(mol)
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            write_coord(work / "coord", z, xyz)
            for label, model, func, extra, refcmd in CASES:
                for order in ("gradient", "hessian"):
                    inp = {"numbers": z, "positions": xyz, "model": model, "functional": func, "charge": charge,
                           "order": order, "repeat": args.repeat, **extra}
                    (work / "in.json").write_text(json.dumps(inp))
                    mine = []
                    for t in threads:
                        r = subprocess.run([str(args.bin), str(work / "in.json"), str(t)], check=True,
                                           capture_output=True, text=True)
                        mine.append(json.loads(r.stdout)["seconds"])
                    ref = []
                    if not args.no_ref and (order == "gradient" or len(z) <= args.ref_hessian_max_atoms):
                        cmd = [str(args.bindir / refcmd[0]), *refcmd[1:]]
                        if model == "d4":
                            cmd += ["-c", str(int(charge))]
                        # s-dftd3's --grad / --hessian take an optional file name: never put the
                        # input right after them
                        cmd += ["--grad"] + (["--hessian"] if order == "hessian" else []) + ["--noedisp", "-s", "coord"]
                        for t in threads:
                            env = dict(os.environ, OMP_NUM_THREADS=str(t))

                            def run():
                                t0 = time.perf_counter()
                                subprocess.run(cmd, cwd=work, env=env, check=True, capture_output=True)
                                return time.perf_counter() - t0

                            ref.append(best_of(min(args.repeat, 3), run))
                    cells = " | ".join(f"{v:.2e}" for v in mine) + " | " + (
                        " | ".join(f"{v:.2e}" for v in ref) if ref else " | ".join("-" for _ in threads))
                    print(f"| {mol.stem} | {len(z)} | {label} | {order} | {cells} |", flush=True)


if __name__ == "__main__":
    main()
