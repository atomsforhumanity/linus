#!/usr/bin/env python
"""Gradient error of the linus:N grids against pyscf:8 on hydrides of heavier main-group elements.

    LINUS_BIN=path/to/linus python grid3_hydrides.py [METHOD]     (default b3lyp)

linus:3 (the Hessian/frequency default grid) has no atomic size adjustment: cell boundaries at the
bond midpoints. Phase 2 found GeH3 gradients 2.4-3.5e-4 Eh/bohr off with linus:3 for every
functional, 10x worse than linus:2. This scans X-H molecules down the groups (idealized
geometries, def2-SVP, RI-JK, tight SCF; every grid at the same geometry). Prints, per grid, the
max-abs gradient difference from pyscf:8 (and on the heavy atom X) and the energy difference.
"""
import json
import os
import subprocess
import sys

import numpy as np

LINUS = os.environ.get("LINUS_BIN", "linus")
ANG = 1 / 0.529177210544


def tetra(x, r):
    d = r / np.sqrt(3)
    return [x, "H", "H", "H", "H"], [[0, 0, 0], [d, d, d], [-d, -d, d], [-d, d, -d], [d, -d, -d]]


def pyramid(x, r, ang):
    # C3v XH3 with H-X-H angle ang (degrees)
    a = np.radians(ang)
    ct = np.sqrt((1 + 2 * np.cos(a)) / 3)
    st = np.sqrt(1 - ct ** 2)
    hs = [[r * st * np.cos(p), r * st * np.sin(p), -r * ct] for p in (0, 2 * np.pi / 3, 4 * np.pi / 3)]
    return [x, "H", "H", "H"], [[0, 0, 0]] + hs


def bent(x, r, ang):
    a = np.radians(ang) / 2
    return [x, "H", "H"], [[0, 0, 0], [r * np.sin(a), 0, r * np.cos(a)], [-r * np.sin(a), 0, r * np.cos(a)]]


MOLS = {
    "CH4": tetra("C", 1.09), "SiH4": tetra("Si", 1.48), "GeH4": tetra("Ge", 1.53), "SnH4": tetra("Sn", 1.71),
    "NH3": pyramid("N", 1.01, 106.7), "PH3": pyramid("P", 1.42, 93.3), "AsH3": pyramid("As", 1.52, 92.0),
    "H2O": bent("O", 0.96, 104.5), "H2S": bent("S", 1.34, 92.1), "H2Se": bent("Se", 1.46, 91.0),
    "HCl": (["Cl", "H"], [[0, 0, 0], [0, 0, 1.27]]), "HBr": (["Br", "H"], [[0, 0, 0], [0, 0, 1.41]]),
    "GaH3": (["Ga", "H", "H", "H"], [[0, 0, 0], [1.56, 0, 0], [-0.78, 1.351, 0], [-0.78, -1.351, 0]]),
}


def request(sym, pos, method, calctype, grid=None, **extra):
    # (int_thresh 1e-14 is not given: the four-center screening is not used by jk df, D41)
    return {"structure": {"symbols": sym, "geometry": (np.array(pos, float) * ANG).tolist(), "charge": 0, "multiplicity": 1},
            "method": method, "basis": "def2-svp", "jk": {"kind": "df", "aux": "def2-universal-jkfit"},
            "calctype": calctype, "threads": 2,
            "scf": {"convergence": "extreme", "max_iter": 300}, **({"grid": {"xc": grid}} if grid else {}), **extra}


def run(req):
    return json.loads(subprocess.run([LINUS, "--json"], input=json.dumps(req), capture_output=True, text=True).stdout)


def main():
    method = sys.argv[1] if len(sys.argv) > 1 else "b3lyp"
    for name, (sym, pos) in MOLS.items():
        g = {}
        for grid in ("pyscf:8", "pyscf:3", "linus:2", "linus:3", "linus:4"):
            o = run(request(sym, pos, method, "gradient", grid=grid))
            g[grid] = (o["energy"], np.array(o["gradient"]))
        e0, g0 = g["pyscf:8"]
        cells = []
        for grid in ("pyscf:3", "linus:2", "linus:3", "linus:4"):
            e, gr = g[grid]
            d = np.abs(gr - g0)
            cells.append(f"{grid} dG {d.max():.1e} (X {d[0].max():.1e}) dE {e - e0:+.1e}")
        print(f"{name:5s} " + " | ".join(cells), flush=True)


if __name__ == "__main__":
    main()
