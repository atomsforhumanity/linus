"""Usage: LINUS_BIN=... python linus_check.py heavy_built.CsF [water|iefpcm-water]

linus SMD analytic gradient against central differences of its own energy, for a molecule of the sets
(e.g. heavy_built.CsF), B3LYP/def2-SVP RI-JK, pyscf:3 grid, PySCF's SMD preset."""
import json
import os
import subprocess
from pathlib import Path
import sys

import numpy as np

LINUS = os.environ.get("LINUS_BIN", "linus")
MOLS = str(Path(__file__).resolve().parents[2] / "molecules")
mid, solvent = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else "water")
m = None
for f in (f"{MOLS}/inorganic.jsonl", f"{MOLS}/internal/tmqm.jsonl", f"{MOLS}/internal/gmtkn55.jsonl"):
    for ln in open(f):
        r = json.loads(ln)
        if r["id"] == mid:
            m = r
pos0 = np.array(m["geometry_angstrom"]) / 0.529177210544


def run(pos, calc):
    req = {"structure": {"symbols": m["symbols"], "geometry": pos.tolist(), "charge": m["charge"],
                         "multiplicity": m["multiplicity"]},
           "method": "b3lyp", "basis": "def2-svp", "jk": {"kind": "df", "aux": "def2-universal-jkfit"}, "grid": {"xc": "pyscf:3"},
           "calctype": calc, "threads": 4, "scf": {"convergence": "extreme", "max_iter": 300},
           "solvent": ({"model": "iefpcm", "eps": 78.3553, "preset": "pyscf"} if solvent == "iefpcm-water" else {"model": "smd", "name": solvent, "preset": "pyscf"})}
    o = json.loads(subprocess.run([LINUS, "--json"], input=json.dumps(req), capture_output=True, text=True).stdout)
    return o


o = run(pos0, "gradient")
g = np.array(o["gradient"])
parts = o.get("gradient_parts") or {}
h = 1e-3
n = len(m["symbols"])
fd = np.zeros_like(g)
for a in range(n):
    for k in range(3):
        p, q = pos0.copy(), pos0.copy()
        p[a, k] += h
        q[a, k] -= h
        fd[a, k] = (run(p, "energy")["energy"] - run(q, "energy")["energy"]) / (2 * h)
print(mid, solvent, "max |analytic - FD| %.1e" % np.abs(g - fd).max(), "sum of analytic gradient %.1e" % np.abs(g.sum(0)).max())
print("solvent part of the analytic gradient:", np.round(np.array(parts.get("solvent") or [[0, 0, 0]]), 6).tolist())
