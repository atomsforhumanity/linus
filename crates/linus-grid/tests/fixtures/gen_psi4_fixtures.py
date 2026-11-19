#!/usr/bin/env python
"""Generate Psi4 reference grids for linus-grid's Rust tests (tests/fixtures/psi4_grids.json).

Run with the Psi4 1.11 reference environment:
    .refenv/bin/python crates/linus-grid/tests/fixtures/gen_psi4_fixtures.py

Psi4 settings: defaults (75 Treutler radial / 302 Lebedev, Treutler nuclear scheme, no
pruning, DFT_WEIGHTS_TOLERANCE 1e-15) except DFT_REMOVE_DISTANT_POINTS = false, because the
distant-point removal depends on the basis set, and DFT_BASIS_TOLERANCE = 1e-40 so that
no octree block is dropped for lacking significant basis functions. The grid orientation matrix chosen by Psi4
(DFTGrid.orientation()) is stored; linus reproduces Psi4 given that matrix.
"""
import json
import math
import os
import sys

import numpy as np
import psi4

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from molecules import MOLECULES  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
NSAMPLE = 300
BOHR = 0.52917721067  # Psi4's (CODATA 2014) value, used to convert the Angstrom input


def fsum(a):
    return math.fsum(float(x) for x in a)


def case(name, extra=None):
    psi4.core.clean()
    psi4.core.clean_options()
    psi4.set_memory("2 GB")
    psi4.core.be_quiet()
    geom = MOLECULES[name] + "\nsymmetry c1\nnocom\nnoreorient\n"
    mol = psi4.geometry(geom)
    mol.update_geometry()
    # Default OCTREE blocking keeps every point (ATOMIC blocking drops blocks without basis
    # functions); octree blocks have no parent atom, so points are matched by coordinates only.
    # DFT_BASIS_TOLERANCE is lowered so that no far-away octree block is dropped for having
    # no significant basis function (another basis-dependent point removal).
    opts = {"basis": "sto-3g", "dft_remove_distant_points": False, "dft_basis_tolerance": 1e-40}
    if extra:
        opts.update(extra)
    psi4.set_options(opts)
    wfn = psi4.core.Wavefunction.build(mol, psi4.core.get_global_option("BASIS"))
    sup = psi4.driver.dft.build_superfunctional("svwn", True)[0]
    V = psi4.core.VBase.build(wfn.basisset(), sup, "RV")
    V.initialize()
    g = V.grid()
    x = np.concatenate([np.array(b.x()) for b in g.blocks()])
    y = np.concatenate([np.array(b.y()) for b in g.blocks()])
    z = np.concatenate([np.array(b.z()) for b in g.blocks()])
    w = np.concatenate([np.array(b.w()) for b in g.blocks()])
    atm = np.concatenate([np.full(b.npoints(), b.parent_atom()) for b in g.blocks()])
    geom_b = np.array(mol.geometry())
    zs = [int(mol.Z(i)) for i in range(mol.natom())]
    orient = np.array(g.orientation()).tolist()
    per_atom = []
    n = len(w)
    rng = np.random.default_rng(777)
    idx = np.sort(rng.choice(n, min(NSAMPLE, n), replace=False))
    return dict(name=name + ("" if not extra else "_" + "_".join(f"{k}{v}" for k, v in extra.items())),
                molecule=name, z=zs, xyz=geom_b.tolist(), orientation=orient, options=extra or {},
                npoints=int(n), sum_w=fsum(w), per_atom=per_atom,
                sample=dict(x=x[idx].tolist(), y=y[idx].tolist(), z=z[idx].tolist(), w=w[idx].tolist(),
                            atom=atm[idx].astype(int).tolist(), full=False))


def main():
    cases = [case("water"), case("benzene"), case("ch3so2cl"),
             case("water", {"dft_nuclear_scheme": "BECKE"}),
             case("water", {"dft_radial_points": 50, "dft_spherical_points": 194})]
    for c in cases:
        print(c["name"], c["npoints"], file=sys.stderr)
    out = dict(generator="tests/fixtures/gen_psi4_fixtures.py", psi4_version=psi4.__version__, cases=cases)
    path = os.path.join(HERE, "psi4_grids.json")
    with open(path, "w") as f:
        json.dump(out, f, separators=(",", ":"))
    print(path, os.path.getsize(path), file=sys.stderr)


if __name__ == "__main__":
    main()
