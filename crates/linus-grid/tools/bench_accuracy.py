#!/usr/bin/env python
"""XC-integration accuracy of grid presets for fixed SCF densities (uses PySCF for the
density and the XC integration; grids come from linus-grid's grid_dump example).

    cargo build --release -p linus-grid --examples
    .venv/bin/python crates/linus-grid/tools/bench_accuracy.py <path/to/grid_dump> [preset ...]

For each molecule, a B3LYP/def2-SVP density is converged once (PySCF level 5); errors are
E_xc(grid) - E_xc(reference) and N(grid) - N for B3LYP and r2SCAN functionals, the reference
being PySCF level 9 (200 radial x 1454 Lebedev). Presets are grid_dump preset strings or
json:<spec.json>.
"""
import json
import os
import subprocess
import sys
import tempfile
import time

import numpy as np
from pyscf import dft, gto, lib

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tests", "fixtures"))
from molecules import MOLECULES  # noqa: E402

EXTRA = {
    "water_dimer": """O -1.551007 -0.114520 0.000000
H -1.934259 0.762503 0.000000
H -0.599677 0.040712 0.000000
O 1.350625 0.111469 0.000000
H 1.680398 -0.373741 -0.758561
H 1.680398 -0.373741 0.758561""",
    "glycine": """N 1.9504 -0.1306 0.0005
C 0.7038 0.6164 -0.0001
C -0.5279 -0.2614 0.0000
O -0.4999 -1.4723 0.0001
O -1.6985 0.4142 -0.0002
H 2.0102 -0.7234 0.8282
H 2.0109 -0.7241 -0.8264
H 0.6767 1.2731 0.8823
H 0.6765 1.2734 -0.8823
H -2.4474 -0.2106 -0.0002""",
}
ALL = dict(MOLECULES)
ALL.update(EXTRA)
XCS = ("b3lyp", "r2scan")


def numint(mol, coords, weights, xc, dm):
    g = dft.gen_grid.Grids(mol)
    g.coords = np.ascontiguousarray(coords)
    g.weights = np.ascontiguousarray(weights)
    g.non0tab = g.make_mask(mol, g.coords)
    g.screen_index = g.non0tab
    ni = dft.numint.NumInt()
    n, exc, _ = ni.nr_rks(mol, g, xc, dm)
    return n, exc


def main():
    dump = sys.argv[1]
    presets = sys.argv[2:] or ["pyscf:2", "pyscf:3", "psi4", "orca:1", "orca:2", "orca:3",
                               "linus:1", "linus:2", "linus:3", "linus:4"]
    lib.num_threads(16)
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for name, geom in ALL.items():
            mol = gto.M(atom=geom, basis="def2-svp", verbose=0, spin=0)
            cache = os.environ.get("LINUS_GRID_ACC_CACHE")
            cfile = os.path.join(cache, name + ".npz") if cache else None
            if cfile and os.path.exists(cfile):
                c = np.load(cfile)
                dm = c["dm"]
                ref = {xc: (None, float(c[xc])) for xc in XCS}
            else:
                mf = dft.RKS(mol, xc="b3lyp")
                mf.grids.level = 5
                mf.conv_tol = 1e-9
                mf.kernel()
                dm = mf.make_rdm1()
                gref = dft.gen_grid.Grids(mol)
                gref.level = 9
                gref.build()
                ref = {xc: numint(mol, gref.coords, gref.weights, xc, dm) for xc in XCS}
                if cfile:
                    np.savez(cfile, dm=dm, **{xc: ref[xc][1] for xc in XCS})
            xyz = os.path.join(tmp, name + ".xyz")
            with open(xyz, "w") as f:
                f.write(geom + "\n")
            for p in presets:
                out = os.path.join(tmp, "g.bin")
                subprocess.run([dump, xyz, p, out], check=True, capture_output=True)
                a = np.fromfile(out, dtype="<f8").reshape(-1, 6)
                row = dict(mol=name, preset=p, npts=len(a))
                for xc in XCS:
                    n, exc = numint(mol, a[:, :3], a[:, 3], xc, dm)
                    row[f"dE_{xc}"] = exc - ref[xc][1]
                    row["dN"] = n - mol.nelectron
                rows.append(row)
                print(f"{name:12} {p:28} {len(a):9d} dN={row['dN']:+.2e} "
                      + " ".join(f"dE({xc})={row['dE_' + xc] * 1e6:+9.2f}uEh" for xc in XCS), flush=True)
    # summary per preset
    print("\nsummary (mean abs / max abs over molecules, micro-Hartree; total points)")
    for p in presets:
        label = os.path.basename(p)
        rs = [r for r in rows if r["preset"] == p]
        npts = sum(r["npts"] for r in rs)
        s = f"{label:28} pts={npts:9d}"
        for xc in XCS:
            e = np.array([abs(r['dE_' + xc]) for r in rs]) * 1e6
            s += f"  {xc}: mean {e.mean():8.2f} max {e.max():8.2f}"
        print(s)
    with open("bench_accuracy.json", "w") as f:
        json.dump(rows, f)


if __name__ == "__main__":
    main()
