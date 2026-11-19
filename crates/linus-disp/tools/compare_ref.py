#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Validate linus-disp against the s-dftd3 1.7.0 and dftd4 4.3.1 programs and against
finite differences of its own energies and gradients.

Usage (from the workspace root, after ``cargo build --release -p linus-disp --examples``)::

    python3 -I crates/linus-disp/tools/compare_ref.py \
        --bindir ~/micromamba/envs/linus-disp-ref/bin [--fd-step 1e-3] [--json out.json] \
        crates/linus-disp/tests/molecules/*.xyz

Geometries are read in Angstrom, converted once to bohr with CODATA 2022
(0.529177210544 A/bohr, as reference/run_ref.py does) and given to every program in bohr
(Turbomole ``coord`` files written with 17 significant digits). The molecular charge comes
from ``charge=<q>`` in the xyz comment line (default 0). The reference programs run with
``OMP_NUM_THREADS=1``.

Columns: dE = |E - E_ref|, dG = max |G - G_ref|, dH = max |H - H_ref| (s-dftd3: analytic;
dftd4: two-body by central differences of analytic gradients with step 1e-4, ATM analytic),
dG_fd = max |G - 5-point FD of E|, dH_fd = max |H - 5-point FD of G|.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

BOHR_A = 0.529177210544  # CODATA 2022
SYMBOLS = [
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar",
    "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br",
    "Kr", "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te",
    "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm",
    "Yb", "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn",
    "Fr", "Ra", "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr",
]

# (label, model, linus functional, linus extra, reference command-line arguments)
CASES = [
    ("D3(BJ) B3LYP", "d3bj", "b3lyp", {}, ["--bj", "b3lyp"]),
    ("D3(BJ)+ATM B3LYP", "d3bj", "b3lyp", {"atm": True}, ["--bj", "b3lyp", "--atm"]),
    ("D3(BJ) wB97X-D3BJ", "d3bj", "wb97x-d3bj", {}, ["--bj", "wb97x"]),
    ("D4 B3LYP", "d4", "b3lyp", {}, ["--func", "b3lyp"]),
    ("D4 wB97X-D4", "d4", "wb97x-d4", {}, ["--func", "wb97x"]),
    ("D4 r2SCAN-3c", "d4", "r2scan-3c", {}, ["--func", "r2scan-3c"]),
]


def read_xyz(path: Path):
    lines = path.read_text().splitlines()
    n = int(lines[0])
    m = re.search(r"charge\s*=\s*([-+]?\d+)", lines[1])
    charge = float(m.group(1)) if m else 0.0
    z, xyz = [], []
    for line in lines[2:2 + n]:
        s, x, y, w = line.split()[:4]
        z.append(SYMBOLS.index(s.capitalize()) + 1)
        xyz.append([float(x) / BOHR_A, float(y) / BOHR_A, float(w) / BOHR_A])
    return z, xyz, charge


def write_coord(path: Path, z, xyz):
    with open(path, "w") as f:
        f.write("$coord\n")
        for zi, p in zip(z, xyz):
            f.write(f" {p[0]!r} {p[1]!r} {p[2]!r} {SYMBOLS[zi - 1].lower()}\n")
        f.write("$end\n")


def maxabs(a, b):
    return max((abs(x - y) for x, y in zip(a, b)), default=0.0)


def flat(m):
    return [v for row in m for v in row] if m and isinstance(m[0], list) else list(m)


def run_case(work: Path, args, z, xyz, charge, case):
    label, model, func, extra, refargs = case
    env = dict(os.environ, OMP_NUM_THREADS="1")
    out = work / "ref.json"
    if out.exists():
        out.unlink()
    if model == "d3bj":
        cmd = [str(args.bindir / "s-dftd3"), "run", *refargs, "--grad"]
    else:
        cmd = [str(args.bindir / "dftd4"), "run", *refargs, "-c", str(int(charge)), "--grad"]
    hess = args.order == "hessian"
    cmd += (["--hessian"] if hess else []) + ["--json", str(out), "coord"]
    subprocess.run(cmd, cwd=work, env=env, check=True, capture_output=True, text=True)
    ref = json.loads(out.read_text())
    inp = {"numbers": z, "positions": xyz, "model": model, "functional": func, "charge": charge,
           "order": args.order, **extra}
    if args.fd_step > 0:
        inp["fd_step"] = args.fd_step
    (work / "in.json").write_text(json.dumps(inp))
    r = subprocess.run([str(args.bin), str(work / "in.json"), str(args.threads)], check=True,
                       capture_output=True, text=True)
    mine = json.loads(r.stdout)
    g = flat(mine["gradient"])
    nan = float("nan")
    return {
        "case": label,
        "natoms": len(z),
        "energy": mine["energy"],
        "energy_ref": ref["energy"],
        "dE": abs(mine["energy"] - ref["energy"]),
        "dG": maxabs(g, ref["gradient"]),
        "dH": maxabs(mine["hessian"], ref["hessian"]) if hess else nan,
        "dG_fd": maxabs(g, mine["gradient_fd"]) if "gradient_fd" in mine else nan,
        "dH_fd": maxabs(mine["hessian"], mine["hessian_fd"]) if hess and "hessian_fd" in mine else nan,
        "max_G": max(abs(v) for v in g),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("molecules", nargs="+", type=Path)
    ap.add_argument("--bindir", type=Path, default=Path.home() / "micromamba/envs/linus-disp-ref/bin")
    ap.add_argument("--bin", type=Path, default=Path("target/release/examples/disp_json"))
    ap.add_argument("--fd-step", type=float, default=1e-3, help="0: no finite differences")
    ap.add_argument("--order", choices=("gradient", "hessian"), default="hessian")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--cases", default="", help="comma-separated substrings of case labels")
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()
    cases = [c for c in CASES if not args.cases or any(s in c[0] for s in args.cases.split(","))]
    rows = []
    print("| molecule | N | case | E (Eh) | dE | dG | dH | dG_fd | dH_fd |")
    print("|---|---|---|---|---|---|---|---|---|")
    for mol in args.molecules:
        z, xyz, charge = read_xyz(mol)
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            write_coord(work / "coord", z, xyz)
            for case in cases:
                try:
                    res = run_case(work, args, z, xyz, charge, case)
                except subprocess.CalledProcessError as e:
                    print(f"| {mol.stem} | {len(z)} | {case[0]} | failed: {e.stderr.strip()[-200:]} |||||")
                    continue
                res["molecule"] = mol.stem
                rows.append(res)
                print(f"| {mol.stem} | {len(z)} | {res['case']} | {res['energy']:.12f} | {res['dE']:.1e} | "
                      f"{res['dG']:.1e} | {res['dH']:.1e} | {res['dG_fd']:.1e} | {res['dH_fd']:.1e} |", flush=True)
    if args.json:
        args.json.write_text(json.dumps(rows, indent=1))
    worst = {k: max((r[k] for r in rows if r[k] == r[k]), default=0.0) for k in ("dE", "dG", "dH", "dG_fd", "dH_fd")}
    print("worst:", " ".join(f"{k} {v:.1e}" for k, v in worst.items()), file=sys.stderr)


if __name__ == "__main__":
    main()
