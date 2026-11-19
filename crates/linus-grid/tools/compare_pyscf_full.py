#!/usr/bin/env python
"""Full (every point) comparison of linus-grid's pyscf preset against pyscf.dft.Grids.

    cargo build --release -p linus-grid --example grid_dump
    .venv/bin/python crates/linus-grid/tools/compare_pyscf_full.py <path/to/grid_dump> [levels]

Points are matched one-to-one per atom by nearest neighbour (k-d tree); reports point
counts, max |dx|, max |dw|, max |dw|/|w| and max |dw|/w_atomic per molecule and level.
"""
import os
import subprocess
import sys
import tempfile

import numpy as np
from scipy.spatial import cKDTree
from pyscf import gto
from pyscf.dft import gen_grid

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tests", "fixtures"))
from molecules import MOLECULES  # noqa: E402


def run(dump, name, level, tmp):
    xyz = os.path.join(tmp, name + ".xyz")
    with open(xyz, "w") as f:
        f.write(MOLECULES[name] + "\n")
    out = os.path.join(tmp, f"{name}_{level}.bin")
    subprocess.run([dump, xyz, f"pyscf:{level}", out], check=True, capture_output=True)
    a = np.fromfile(out, dtype="<f8").reshape(-1, 6)
    mol = gto.M(atom=MOLECULES[name], basis="sto-3g", verbose=0, spin=None)
    g = gen_grid.Grids(mol)
    g.level = level
    g.build()
    keep = g.atm_idx >= 0
    c, w, atm = g.coords[keep], g.weights[keep], g.atm_idx[keep]
    res = dict(n_ref=len(w), n=len(a), dx=0.0, dw=0.0, dwrel=0.0, dwpart=0.0, unmatched=0)
    for ia in range(mol.natm):
        m1 = atm == ia
        m2 = a[:, 4] == ia
        if m1.sum() != m2.sum():
            res["unmatched"] += abs(int(m1.sum()) - int(m2.sum()))
            continue
        t = cKDTree(a[m2, :3])
        d, j = t.query(c[m1])
        if len(set(j.tolist())) != len(j) or d.max() > 1e-9:
            res["unmatched"] += int((d > 1e-9).sum())
        ours = a[m2][j]
        res["dx"] = max(res["dx"], float(np.abs(ours[:, :3] - c[m1]).max()))
        dw = np.abs(ours[:, 3] - w[m1])
        res["dw"] = max(res["dw"], float(dw.max()))
        nz = np.abs(w[m1]) > 0
        if nz.any():
            res["dwrel"] = max(res["dwrel"], float((dw[nz] / np.abs(w[m1][nz])).max()))
        res["dwpart"] = max(res["dwpart"], float((dw / ours[:, 5]).max()))
    return res


def main():
    dump = sys.argv[1]
    levels = [int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else list(range(10))
    with tempfile.TemporaryDirectory() as tmp:
        print(f"{'molecule':10} {'lvl':>3} {'n_pyscf':>9} {'n_linus':>9} {'max|dx|':>9} {'max|dw|':>9} "
              f"{'max|dw|/|w|':>11} {'max|dw|/w_at':>12} unmatched")
        for name in ("water", "benzene", "ch3so2cl", "hbr"):
            for level in levels:
                r = run(dump, name, level, tmp)
                print(f"{name:10} {level:3d} {r['n_ref']:9d} {r['n']:9d} {r['dx']:9.1e} {r['dw']:9.1e} "
                      f"{r['dwrel']:11.1e} {r['dwpart']:12.1e} {r['unmatched']}", flush=True)


if __name__ == "__main__":
    main()
