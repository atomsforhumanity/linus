#!/usr/bin/env python
"""The F atom (2P) plateau in C-PCM against the surface's points per sphere.

    LINUS_BIN=path/to/linus python lebedev_scan.py [SYMBOL MULTIPLICITY METHOD]   (default F 2 hf)

Per Lebedev set: converged?, iterations, min/max max|F_ai| over the last 20 iterations and the
energy spread over them. A denser (more nearly isotropic) surface leaves a smaller residual.
"""
import json
import os
import subprocess
import sys

LINUS = os.environ.get("LINUS_BIN", "linus")
sym, mult = (sys.argv[1], int(sys.argv[2])) if len(sys.argv) > 2 else ("F", 2)
method = sys.argv[3] if len(sys.argv) > 3 else "hf"
for leb in (50, 110, 302, 590, 1202, 5810):
    # (int_thresh 1e-14 is the four-center screening, which jk df does not use: D41)
    req = {"structure": {"symbols": [sym], "geometry": [[0, 0, 0]], "charge": 0, "multiplicity": mult}, "method": method,
           "basis": "def2-svp", "jk": {"kind": "df", "aux": "def2-universal-jkfit"}, "calctype": "energy",
           "threads": 1, "scf": {"convergence": "extreme", "max_iter": 150},
           "solvent": {"model": "cpcm", "name": "acetonitrile", "lebedev": leb}}
    if method != "hf":
        req["grid"] = {"xc": "pyscf:3"}
    o = json.loads(subprocess.run([LINUS, "--json"], input=json.dumps(req), capture_output=True, text=True).stdout)
    s = o.get("scf") or {}
    h = s.get("history") or []
    tail = [r["energy"] for r in h[-20:]]
    gm = [r["gmax"] for r in h[-20:]]
    print(f"lebedev {leb:5d}: converged {s.get('converged')} it {s.get('iterations') or len(h)} "
          f"max|F_ai| last 20: {min(gm):.1e} .. {max(gm):.1e}  E spread {max(tail) - min(tail):.1e}", flush=True)
