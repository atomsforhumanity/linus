#!/usr/bin/env python
"""Generate PySCF reference grids for linus-grid's Rust tests (tests/fixtures/pyscf_*.json).

Run with the dev venv (PySCF 2.15):
    .venv/bin/python crates/linus-grid/tests/fixtures/gen_pyscf_fixtures.py

For every case we store: the molecule (Z, bohr coordinates), the PySCF settings, the exact
radial nodes/weights and angular counts of each element's atomic grid, the total and per-atom
point counts, compensated sums of weights and weighted moments (per atom), and a deterministic
sample of points (all points for small grids) with coordinates, weights and owning atom.
PySCF's zero-weight alignment padding points (atm_idx == -1) are dropped.
"""
import json
import math
import os
import sys

import numpy
from pyscf import gto, __version__ as pyscf_version
from pyscf.dft import gen_grid, radi

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from molecules import MOLECULES  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
MAX_FULL = 1500   # store every point when the grid is at most this large
NSAMPLE = 250     # otherwise store this many points

RADI = {"treutler": radi.treutler, "becke": radi.becke, "mura_knowles": radi.mura_knowles,
        "gauss_chebyshev": radi.gauss_chebyshev, "delley": radi.delley}
PRUNE = {"nwchem": gen_grid.nwchem_prune, "sg1": gen_grid.sg1_prune,
         "treutler": gen_grid.treutler_prune, "none": None}
SCHEME = {"becke": gen_grid.original_becke, "stratmann": gen_grid.stratmann,
          "lko": gen_grid.becke_lko}
ADJUST = {"treutler": radi.treutler_atomic_radii_adjust, "becke": radi.becke_atomic_radii_adjust,
          "none": None}


def fsum(a):
    return math.fsum(float(x) for x in a)


def build(name, level=3, radi_method="treutler", prune="nwchem", scheme="becke",
          adjust="treutler", atom_grid=None):
    mol = gto.M(atom=MOLECULES[name], basis="sto-3g", unit="Angstrom", verbose=0,
                spin=None if name != "hbr" else 0)
    g = gen_grid.Grids(mol)
    g.level = level
    g.radi_method = RADI[radi_method]
    g.prune = PRUNE[prune]
    g.becke_scheme = SCHEME[scheme]
    g.radii_adjust = ADJUST[adjust]
    if atom_grid is not None:
        g.atom_grid = tuple(atom_grid)
    g.build(with_non0tab=False)
    keep = g.atm_idx >= 0
    coords, w, atm = g.coords[keep], g.weights[keep], g.atm_idx[keep]

    natm = mol.natm
    z = [int(mol.atom_charge(i)) for i in range(natm)]
    xyz = mol.atom_coords().tolist()
    case = dict(name=f"{name}_l{level}_{radi_method}_{prune}_{scheme}_{adjust}"
                + ("" if atom_grid is None else f"_ag{atom_grid[0]}x{atom_grid[1]}"),
                molecule=name, z=z, xyz=xyz,
                settings=dict(level=level, radi_method=radi_method, prune=prune, scheme=scheme,
                              adjust=adjust, atom_grid=atom_grid),
                npoints=int(len(w)))
    # atomic grids, reconstructed exactly as gen_atomic_grids does
    atomic = {}
    for ia in range(natm):
        chg = z[ia]
        if str(chg) in atomic:
            continue
        if atom_grid is not None:
            n_rad, n_ang = atom_grid
        else:
            n_rad = gen_grid._default_rad(chg, level)
            n_ang = gen_grid._default_ang(chg, level)
        rad, dr = g.radi_method(n_rad, chg, ia)
        rw = 4 * numpy.pi * rad**2 * dr
        angs = g.prune(chg, rad, n_ang) if callable(g.prune) else [n_ang] * n_rad
        atomic[str(chg)] = dict(n_rad=int(n_rad), n_ang=int(n_ang), r=rad.tolist(), w=rw.tolist(),
                                angs=[int(a) for a in angs])
    case["atomic_grids"] = atomic
    per_atom = []
    for ia in range(natm):
        m = atm == ia
        c, ww = coords[m], w[m]
        d = c - numpy.asarray(xyz[ia])
        per_atom.append(dict(n=int(m.sum()), sum_w=fsum(ww), sum_wx=fsum(ww * c[:, 0]),
                             sum_wy=fsum(ww * c[:, 1]), sum_wz=fsum(ww * c[:, 2]),
                             sum_wr2=fsum(ww * (d * d).sum(1)), sum_w_abs=fsum(abs(ww))))
    case["per_atom"] = per_atom
    case["sum_w"] = fsum(w)
    n = len(w)
    if n <= MAX_FULL:
        idx = numpy.arange(n)
    else:
        rng = numpy.random.default_rng(12345)
        idx = numpy.sort(rng.choice(n, NSAMPLE, replace=False))
    case["sample"] = dict(x=coords[idx, 0].tolist(), y=coords[idx, 1].tolist(),
                          z=coords[idx, 2].tolist(), w=w[idx].tolist(),
                          atom=atm[idx].astype(int).tolist(), full=bool(n <= MAX_FULL))
    return case


def main():
    cases = []
    for name in ("water", "benzene", "ch3so2cl"):
        for level in range(10):
            cases.append(build(name, level=level))
            print(cases[-1]["name"], cases[-1]["npoints"], file=sys.stderr)
    cases.append(build("hbr", level=0))
    cases.append(build("hbr", level=3))
    # other PySCF options
    for name in ("water", "ch3so2cl"):
        for rm in ("becke", "mura_knowles", "gauss_chebyshev", "delley"):
            cases.append(build(name, level=1, radi_method=rm))
        for pr in ("none", "sg1", "treutler"):
            cases.append(build(name, level=1, prune=pr))
        for sc in ("stratmann", "lko"):
            for ad in ("treutler", "none"):
                cases.append(build(name, level=2, scheme=sc, adjust=ad))
        cases.append(build(name, level=1, adjust="becke"))
        cases.append(build(name, level=1, adjust="none"))
        cases.append(build(name, level=1, atom_grid=(40, 194)))
    cases.append(build("benzene", level=3, scheme="stratmann"))
    cases.append(build("benzene", level=3, scheme="lko"))
    for c in cases[-24:]:
        print(c["name"], c["npoints"], file=sys.stderr)
    out = dict(generator="tests/fixtures/gen_pyscf_fixtures.py", pyscf_version=pyscf_version,
               cases=cases)
    path = os.path.join(HERE, "pyscf_grids.json")
    with open(path, "w") as f:
        json.dump(out, f, separators=(",", ":"))
    print(path, os.path.getsize(path), file=sys.stderr)


if __name__ == "__main__":
    main()
