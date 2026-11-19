"""PySCF 2.15's SMD G_CDS gradient (`smd.get_cds_legacy`) against central differences of its own
G_CDS energy (no SCF: G_CDS depends only on the geometry and the solvent).

  .refenv/bin/python -I reference/checks/pyscf_cds_grad_fd.py <xyz (Å)> [--solvent water] [--charge 1 --spin 1]

Prints one JSON line: E_CDS, both gradients (Eh/bohr) and the largest difference. Found with
CH3SH+: PySCF's analytic G_CDS gradient is off by 1.3e-3 Eh/bohr, which is exactly the gradient
difference between linus's SMD (matching finite differences) and PySCF's.
"""
import argparse, json

import numpy as np
from pyscf import gto
from pyscf.solvent import smd

BOHR = 0.529177210544


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("xyz")
    ap.add_argument("--solvent", default="water")
    ap.add_argument("--charge", type=int, default=0)
    ap.add_argument("--spin", type=int, default=0)
    ap.add_argument("--h", type=float, default=1e-4, help="step (Å)")
    a = ap.parse_args()
    lines = open(a.xyz).read().splitlines()
    n = int(lines[0])
    syms = [ln.split()[0] for ln in lines[2:2 + n]]
    x0 = np.array([[float(v) for v in ln.split()[1:4]] for ln in lines[2:2 + n]])

    def cds(x):
        mol = gto.M(atom=list(zip(syms, map(tuple, x))), basis="sto-3g", charge=a.charge, spin=a.spin, verbose=0)
        e, g = smd.get_cds_legacy(smd.SMD(mol, a.solvent))
        return e, np.asarray(g)

    e0, g0 = cds(x0)
    fd = np.zeros_like(g0)
    for i in range(n):
        for k in range(3):
            xp, xm = x0.copy(), x0.copy()
            xp[i, k] += a.h
            xm[i, k] -= a.h
            fd[i, k] = (cds(xp)[0] - cds(xm)[0]) / (2 * a.h / BOHR)
    print(json.dumps({"xyz": a.xyz.split("/")[-1], "solvent": a.solvent, "e_cds": e0, "grad_pyscf": g0.tolist(), "grad_fd": fd.tolist(),
                      "max_abs_pyscf_minus_fd": float(np.abs(g0 - fd).max())}))


if __name__ == "__main__":
    main()
