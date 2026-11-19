#!/usr/bin/env python3
"""Cluster task: GFN2-xTB optimizations of a bundle of molecules (xtb 6.7.1, one thread each).

The task JSON holds {"id": ..., "items": [{"id", "xyz" (text, angstrom), "charge",
"multiplicity", "optimize"}, ...]}. xtb comes from $RUN_DIR/xtb-6.7.1-linux-x86_64.tar.xz (the
GitHub release, sha256 62a8d187...), unpacked into the task's scratch. Each item runs in its own
directory: `xtb in.xyz --opt tight --chrg Q --uhf N --cycles 1000`; an optimization that runs
out of cycles is restarted once from its last frame. Checks after each run (gxtb-xtb skill):
exit code, "GEOMETRY OPTIMIZATION CONVERGED", the GFN2 Hamiltonian line, and whether the bonding
changed (pairs within 1.25 x the covalent-radius sum before and after).

Writes $OUT_DIR/result.json: one record per item with status, energy, cycles, the optimized
geometry and the bond check. Exits 0 unless the harness itself failed.
"""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
import tarfile
import time
from pathlib import Path

RCOV = {"H": 0.32, "He": 0.46, "Li": 1.33, "Be": 1.02, "B": 0.85, "C": 0.75, "N": 0.71, "O": 0.63, "F": 0.64,
        "Ne": 0.67, "Na": 1.55, "Mg": 1.39, "Al": 1.26, "Si": 1.16, "P": 1.11, "S": 1.03, "Cl": 0.99, "Ar": 0.96,
        "K": 1.96, "Ca": 1.71, "Sc": 1.48, "Ti": 1.36, "V": 1.34, "Cr": 1.22, "Mn": 1.19, "Fe": 1.16, "Co": 1.11,
        "Ni": 1.10, "Cu": 1.12, "Zn": 1.18, "Ga": 1.24, "Ge": 1.21, "As": 1.21, "Se": 1.16, "Br": 1.14, "Kr": 1.17,
        "Rb": 2.10, "Sr": 1.85, "Y": 1.63, "Zr": 1.54, "Nb": 1.47, "Mo": 1.38, "Tc": 1.28, "Ru": 1.25, "Rh": 1.25,
        "Pd": 1.20, "Ag": 1.28, "Cd": 1.36, "In": 1.42, "Sn": 1.40, "Sb": 1.40, "Te": 1.36, "I": 1.33, "Xe": 1.31,
        "Cs": 2.32, "Ba": 1.96, "La": 1.80, "Hf": 1.52, "Ta": 1.46, "W": 1.37, "Re": 1.31, "Os": 1.29, "Ir": 1.22,
        "Pt": 1.23, "Au": 1.24, "Hg": 1.33, "Tl": 1.44, "Pb": 1.44, "Bi": 1.51, "Po": 1.45, "At": 1.47, "Rn": 1.42}


def parse_xyz(text: str):
    lines = text.splitlines()
    n = int(lines[0].split()[0])
    syms, pos = [], []
    for ln in lines[2:2 + n]:
        p = ln.split()
        syms.append(p[0][0].upper() + p[0][1:].lower())
        pos.append([float(v) for v in p[1:4]])
    return syms, pos


def bonds(syms, pos):
    out = set()
    for i in range(len(syms)):
        for j in range(i):
            if math.dist(pos[i], pos[j]) < 1.25 * (RCOV.get(syms[i], 1.5) + RCOV.get(syms[j], 1.5)):
                out.add((j, i))
    return out


def run_xtb(xtb: str, wd: Path, xyz: str, q: int, uhf: int, level: str = "tight") -> dict:
    wd.mkdir(parents=True, exist_ok=True)
    (wd / "in.xyz").write_text(xyz)
    cmd = [xtb, "in.xyz", "--opt", level, "--chrg", str(q), "--uhf", str(uhf), "--cycles", "1000"]
    env = {**os.environ, "OMP_NUM_THREADS": "1", "OMP_STACKSIZE": "1G", "MKL_NUM_THREADS": "1"}
    t0 = time.time()
    try:
        p = subprocess.run(cmd, cwd=wd, capture_output=True, text=True, env=env, timeout=1500)
        out, rc = p.stdout + p.stderr, p.returncode
    except subprocess.TimeoutExpired as e:
        out, rc = (e.stdout or b"").decode(errors="replace") if isinstance(e.stdout, bytes) else (e.stdout or ""), -9
    dt = time.time() - t0
    (wd / "xtb.out").write_text(out)
    e = re.findall(r"TOTAL ENERGY\s+(-?\d+\.\d+)\s+Eh", out)
    m = re.search(r"GEOMETRY OPTIMIZATION CONVERGED AFTER\s+(\d+)\s+ITERATIONS", out)
    return {"rc": rc, "seconds": dt, "converged": bool(m), "cycles": int(m.group(1)) if m else None,
            "energy": float(e[-1]) if e else None, "gfn2": "GFN2-xTB" in out,
            "unknown_option": "Unknown option" in out,
            "final_xyz": (wd / "xtbopt.xyz").read_text() if (wd / "xtbopt.xyz").exists() else None,
            "tail": "\n".join(out.splitlines()[-12:]) if rc else None}


def main() -> int:
    task = json.load(open(os.environ["TASK_JSON"]))
    out_dir = Path(os.environ["OUT_DIR"])
    run_dir = Path(os.environ["RUN_DIR"])
    with tarfile.open(run_dir / "xtb-6.7.1-linux-x86_64.tar.xz") as t:
        t.extractall("xtbdist")
    xtb = str(Path("xtbdist/xtb-dist/bin/xtb").resolve())
    results = []
    for it in task["items"]:
        rec = {"id": it["id"], "charge": it["charge"], "multiplicity": it["multiplicity"]}
        syms0, pos0 = parse_xyz(it["xyz"])
        if not it.get("optimize", True) or len(syms0) == 1:
            rec.update({"status": "not_optimized", "xyz": it["xyz"]})
            results.append(rec)
            continue
        wd = Path("work") / it["id"]
        r = run_xtb(xtb, wd, it["xyz"], it["charge"], it["multiplicity"] - 1)
        attempts = [r]
        if r["rc"] == 0 and not r["converged"] and r["final_xyz"]:
            r = run_xtb(xtb, wd.with_name(wd.name + "_restart"), r["final_xyz"], it["charge"], it["multiplicity"] - 1)
            attempts.append(r)
        rec["attempts"] = [{k: v for k, v in a.items() if k != "final_xyz"} for a in attempts]
        if r["rc"] == 0 and r["converged"] and r["final_xyz"] and r["gfn2"] and not r["unknown_option"]:
            syms1, pos1 = parse_xyz(r["final_xyz"])
            b0, b1 = bonds(syms0, pos0), bonds(syms1, pos1)
            rec.update({"status": "ok", "energy_gfn2": r["energy"], "xyz": r["final_xyz"],
                        "bonds_broken": sorted(b0 - b1), "bonds_formed": sorted(b1 - b0),
                        "topology_changed": b0 != b1})
        else:
            rec.update({"status": "failed", "reason": "timeout" if r["rc"] == -9 else
                        ("not converged" if r["rc"] == 0 else f"xtb exit {r['rc']}")})
        results.append(rec)
    tmp = out_dir / "result.json.tmp"
    tmp.write_text(json.dumps({"task": task["id"], "results": results}))
    tmp.rename(out_dir / "result.json")
    print(f"{sum(r['status'] == 'ok' for r in results)}/{len(results)} ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
