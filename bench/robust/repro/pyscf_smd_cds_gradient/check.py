#!/usr/bin/env python
"""PySCF 2.15's analytic SMD G_CDS gradient against central differences of its own G_CDS energy.

    .refenv/bin/python bench/robust/repro/pyscf_smd_cds_gradient/check.py [ID [SOLVENT]]
    (default: hard.C2H6_rCC1.54 in water: 1.3e-3 Eh/bohr; drug.aspirin, drug.caffeine agree to 1e-9)

G_CDS depends on the geometry alone (no basis), so the check is cheap: the analytic gradient that
`smd.get_cds_legacy` returns (the MNSOL routine) against 5-point central differences (h 1e-4 bohr)
of the energy it returns. linus's SMD gradient agrees with differences of linus's own energy to
3e-10 on the same molecule, and the two programs' G_CDS energies agree to 2e-12.
"""

import sys
from pathlib import Path

import numpy as np
from pyscf import gto, scf
from pyscf.solvent import smd

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "bench" / "robust" / "repro"))
import sets as molecules  # noqa: E402

mid = sys.argv[1] if len(sys.argv) > 1 else "hard.C2H6_rCC1.54"
solvent = sys.argv[2] if len(sys.argv) > 2 else "water"
rec = next(r for r in molecules.load() if r["id"] == mid)
x0 = np.asarray(rec["geometry_angstrom"]) / 0.529177210544


def cds(x):
    mol = gto.M(atom=list(zip(rec["symbols"], x.tolist())), unit="Bohr", basis="sto-3g", charge=rec["charge"],
                spin=rec["multiplicity"] - 1, verbose=0)
    so = scf.RHF(mol).SMD().with_solvent
    so.solvent = solvent
    return smd.get_cds_legacy(so)


e0, g0 = cds(x0)
h = 1e-4
fd = np.zeros_like(x0)
for a in range(len(x0)):
    for k in range(3):
        e = []
        for s in (-2, -1, 1, 2):
            x = x0.copy()
            x[a, k] += s * h
            e.append(cds(x)[0])
        fd[a, k] = (e[0] - 8 * e[1] + 8 * e[2] - e[3]) / (12 * h)
print(f"{mid} ({solvent}): G_CDS {e0:.10f} Eh; max |analytic - finite differences| {np.abs(np.asarray(g0) - fd).max():.2e} "
      f"Eh/bohr (max |gradient| {np.abs(fd).max():.2e})")
