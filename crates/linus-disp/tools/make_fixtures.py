#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Write the reference fixtures of ``tests/reference.rs`` from the s-dftd3 1.7.0 and dftd4
4.3.1 programs (conda-forge builds)::

    python3 -I crates/linus-disp/tools/make_fixtures.py --bindir ~/micromamba/envs/linus-disp-ref/bin

One JSON file per molecule in ``tests/fixtures``: numbers, positions (bohr, CODATA 2022 from
the xyz in Angstrom), charge, and per case the program's energy, gradient and Hessian
(s-dftd3: analytic; dftd4: two-body by central differences with step 1e-4, ATM analytic).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from compare_ref import CASES, read_xyz, write_coord  # noqa: E402

MOLECULES = ["water", "hcn_cl", "la18", "cu-nh3-4", "zncl4", "cisplatin", "sni4", "gdcl3_h2o", "uf6_xe"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bindir", type=Path, default=Path.home() / "micromamba/envs/linus-disp-ref/bin")
    args = ap.parse_args()
    out_dir = HERE.parent / "tests" / "fixtures"
    out_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, OMP_NUM_THREADS="1")
    for name in MOLECULES:
        z, xyz, charge = read_xyz(HERE.parent / "tests" / "molecules" / f"{name}.xyz")
        rec = {"molecule": name, "numbers": z, "positions": xyz, "charge": charge, "cases": []}
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            write_coord(work / "coord", z, xyz)
            for label, model, func, extra, refargs in CASES:
                out = work / "ref.json"
                if model == "d3bj":
                    cmd = [str(args.bindir / "s-dftd3"), "run", *refargs]
                    version = "s-dftd3 1.7.0"
                else:
                    cmd = [str(args.bindir / "dftd4"), "run", *refargs, "-c", str(int(charge))]
                    version = "dftd4 4.3.1"
                cmd += ["--grad", "--hessian", "--json", str(out), "coord"]
                subprocess.run(cmd, cwd=work, env=env, check=True, capture_output=True)
                ref = json.loads(out.read_text())
                rec["cases"].append({
                    "label": label, "model": model, "functional": func, "atm": extra.get("atm", False),
                    "program": version, "command": " ".join(Path(cmd[0]).name if i == 0 else c
                                                            for i, c in enumerate(cmd)),
                    "energy": ref["energy"], "gradient": ref["gradient"], "hessian": ref["hessian"],
                })
        (out_dir / f"{name}.json").write_text(json.dumps(rec))
        print(f"wrote {name}.json", file=sys.stderr)


if __name__ == "__main__":
    main()
