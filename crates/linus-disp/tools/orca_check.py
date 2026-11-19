#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Compare linus-disp with the dispersion energies and gradients ORCA 6.0.0 prints.

Usage (workspace root, after ``cargo build --release -p linus-disp --examples``)::

    python3 -I crates/linus-disp/tools/orca_check.py --workdir /scratch/dir \
        [--orca ~/opt/orca-6.0.0] [--refenv .refenv] \
        [--fd ATOM,XYZ,STEP] crates/linus-disp/tests/molecules/la18.xyz ...

ORCA gets the CODATA 2022 bohr geometry with ``! Bohrs`` (it converts Angstrom with CODATA
1998) and runs a cheap SCF (STO-3G, def2-SVP for elements beyond Kr; r2SCAN-3c brings its
own basis) with ``EnGrad``; only the dispersion lines are compared. ORCA prints the D3
energy with 12 decimals (``Edisp/kcal,au``), the D4 energy with 9 (``Dispersion
correction``) and dispersion gradients with 9 (``DISPERSION GRADIENT``).

``--fd`` also runs ORCA at R +- STEP along one coordinate and prints the central difference
of ORCA's own dispersion energy next to ORCA's printed gradient and linus-disp's.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from compare_ref import BOHR_A, SYMBOLS, read_xyz  # noqa: E402

# label, ORCA keywords (basis appended unless the method fixes it), linus input
CASES = [
    ("D3(BJ) B3LYP", "B3LYP/G D3BJ", {"model": "d3bj", "functional": "b3lyp"}),
    ("D3(BJ) ABC B3LYP", "B3LYP/G D3BJ ABC", {"model": "d3bj", "functional": "b3lyp", "atm": True}),
    ("D3(BJ) wB97X-D3BJ", "wB97X-D3BJ", {"model": "d3bj", "functional": "wb97x-d3bj"}),
    ("D4 B3LYP", "B3LYP/G D4", {"model": "d4", "functional": "b3lyp"}),
    ("D4 wB97X-D4", "wB97X-D4", {"model": "d4", "functional": "wb97x-d4"}),
    ("D4 r2SCAN-3c", "r2SCAN-3c", {"model": "d4", "functional": "r2scan-3c"}),
]


def orca_env(args):
    env = dict(os.environ)
    env["PATH"] = f"{args.orca}:{args.refenv}/bin:" + env.get("PATH", "")
    env["LD_LIBRARY_PATH"] = f"{args.orca}/lib:{args.refenv}/lib"
    env.update(OMPI_MCA_btl="self,vader", OMPI_MCA_btl_vader_single_copy_mechanism="none",
               OMPI_MCA_hwloc_base_binding_policy="none", OMPI_MCA_rmaps_base_oversubscribe="1",
               OMP_NUM_THREADS="1")
    return env


def run_orca(args, wd: Path, tag: str, kw: str, z, xyz, charge, nprocs=4):
    nel = sum(z) - int(charge)
    mult = 1 if nel % 2 == 0 else 2
    inp = wd / f"{tag}.inp"
    with open(inp, "w") as f:
        f.write(f"! {kw} Bohrs EnGrad NoUseSym\n%pal nprocs {nprocs} end\n%maxcore 2000\n"
                f"%scf maxiter 300 end\n* xyz {int(charge)} {mult}\n")
        for zi, p in zip(z, xyz):
            f.write(f"{SYMBOLS[zi - 1]:3s} {p[0]:24.16f} {p[1]:24.16f} {p[2]:24.16f}\n")
        f.write("*\n")
    out = wd / f"{tag}.out"
    with open(out, "w") as fo:
        subprocess.run([f"{args.orca}/orca", inp.name], cwd=wd, env=orca_env(args), stdout=fo,
                       stderr=subprocess.STDOUT)
    txt = out.read_text()
    if "ORCA TERMINATED NORMALLY" not in txt:
        raise RuntimeError(f"ORCA failed: {out}")
    m = re.search(r"Edisp/kcal,au:\s+\S+\s+(\S+)", txt)
    if m:
        e, digits = float(m.group(1)), 12
    else:
        e, digits = float(re.findall(r"Dispersion correction\s+(-?\d+\.\d+)", txt)[-1]), 9
    blk = txt.split("DISPERSION GRADIENT")[1].split("Difference to translation")[0]
    g = [[float(v) for v in l.split()[3:6]] for l in blk.splitlines()
         if len(l.split()) == 6 and l.split()[2] == ":"]
    return e, digits, g


def linus(args, wd: Path, tag: str, z, xyz, charge, extra):
    inp = {"numbers": z, "positions": xyz, "charge": charge, "order": "gradient", **extra}
    p = wd / f"{tag}.linus.json"
    p.write_text(json.dumps(inp))
    r = subprocess.run([str(args.bin), str(p)], check=True, capture_output=True, text=True)
    return json.loads(r.stdout)


def one(args, mol: Path, case):
    label, kw, extra = case
    z, xyz, charge = read_xyz(mol)
    wd = args.workdir / mol.stem
    wd.mkdir(parents=True, exist_ok=True)
    tag = re.sub(r"\W+", "_", label)
    if "3c" not in kw:
        kw = kw + (" def2-SVP" if max(z) > 36 else " STO-3G")
    e_o, digits, g_o = run_orca(args, wd, tag, kw, z, xyz, charge)
    mine = linus(args, wd, tag, z, xyz, charge, extra)
    g_l = mine["gradient"]
    dg = max(abs(a - b) for ra, rb in zip(g_o, g_l) for a, b in zip(ra, rb))
    res = {"molecule": mol.stem, "natoms": len(z), "case": label, "orca_keywords": kw,
           "energy_orca": e_o, "energy_linus": mine["energy"], "digits": digits,
           "dE": abs(e_o - mine["energy"]), "dG": dg,
           "max_G": max(abs(v) for r in g_l for v in r)}
    if args.fd:
        ia, ic, h = args.fd
        es = []
        for s in (1, -1):
            x2 = [list(p) for p in xyz]
            x2[ia][ic] += s * h
            es.append(run_orca(args, wd, f"{tag}_fd{'p' if s > 0 else 'm'}", kw, z, x2, charge)[0])
        res["fd"] = {"atom": ia, "xyz": ic, "step": h, "orca_fd": (es[0] - es[1]) / (2 * h),
                     "orca_printed": g_o[ia][ic], "linus": g_l[ia][ic]}
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("molecules", nargs="+", type=Path)
    ap.add_argument("--workdir", type=Path, required=True)
    ap.add_argument("--orca", default=os.environ.get("LINUS_ORCA_DIR", str(Path.home() / "opt" / "orca-6.0.0")),
                    help="ORCA 6.0.0 directory ($LINUS_ORCA_DIR, else ~/opt/orca-6.0.0)")
    ap.add_argument("--refenv", default=os.environ.get("LINUS_REFENV", ".refenv"),
                    help="reference environment ($LINUS_REFENV, else .refenv of the working directory)")
    ap.add_argument("--bin", type=Path, default=Path("target/release/examples/disp_json").resolve())
    ap.add_argument("--cases", default="", help="comma-separated substrings of case labels")
    ap.add_argument("--fd", type=lambda s: (int(s.split(",")[0]), int(s.split(",")[1]), float(s.split(",")[2])))
    ap.add_argument("--jobs", type=int, default=12)
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()
    args.bin = args.bin.resolve()
    args.workdir = args.workdir.resolve()
    cases = [c for c in CASES if not args.cases or any(s in c[0] for s in args.cases.split(","))]
    jobs = [(m, c) for m in args.molecules for c in cases]
    rows = []
    with cf.ThreadPoolExecutor(args.jobs) as ex:
        futs = {ex.submit(one, args, m, c): (m, c) for m, c in jobs}
        for fu in cf.as_completed(futs):
            m, c = futs[fu]
            try:
                rows.append(fu.result())
            except Exception as e:  # noqa: BLE001
                rows.append({"molecule": m.stem, "case": c[0], "error": str(e)})
    rows.sort(key=lambda r: (r["molecule"], [c[0] for c in CASES].index(r["case"])))
    print("| molecule | N | case | E ORCA (Eh) | E linus (Eh) | dE | max dG | max abs G | FD check |")
    print("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        if "error" in r:
            print(f"| {r['molecule']} | | {r['case']} | {r['error']} ||||||")
            continue
        fd = ""
        if "fd" in r:
            f = r["fd"]
            fd = f"FD(ORCA E) {f['orca_fd']:.9f}, ORCA {f['orca_printed']:.9f}, linus {f['linus']:.9f}"
        print(f"| {r['molecule']} | {r['natoms']} | {r['case']} | {r['energy_orca']:.{r['digits']}f} | "
              f"{r['energy_linus']:.12f} | {r['dE']:.1e} | {r['dG']:.1e} | {r['max_G']:.1e} | {fd} |")
    if args.json:
        args.json.write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
