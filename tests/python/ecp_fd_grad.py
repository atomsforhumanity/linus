#!/usr/bin/env python
"""Central finite differences of linus energies (same moving grid) against linus's analytic
gradient and PySCF's, for one case of ecp_scf_check.py:

    .refenv/bin/python tests/python/ecp_fd_grad.py CsF.b3lyp.def2-tzvp [--step 1e-3] [--atoms 0,1]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ecp_scf_check as sc  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("case")
    ap.add_argument("--step", type=float, default=1e-3)
    ap.add_argument("--atoms", default=None)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--linus", default=str(sc.REPO / "target" / "release" / "linus"))
    a = ap.parse_args()
    name, method, basis = a.case.split(".", 2)
    charge, mult = sc.SPECS[name]
    atoms = sc.CASES[name.rstrip("+")]
    lin = sc.run_linus(a.linus, atoms, charge, mult, method, basis, True, a.threads)
    ref = sc.run_pyscf(atoms, charge, mult, method, basis, True)
    g = np.array(lin["gradient"])
    idx = range(len(atoms)) if a.atoms is None else [int(x) for x in a.atoms.split(",")]
    for i in idx:
        for k in range(3):
            e = []
            for s in (1, -1):
                at = [(sym, tuple(x)) for sym, x in atoms]
                x = list(at[i][1])
                x[k] += s * a.step
                at[i] = (at[i][0], tuple(x))
                e.append(sc.run_linus(a.linus, at, charge, mult, method, basis, False, a.threads)["energy"])
            fd = (e[0] - e[1]) / (2 * a.step)
            print(f"atom {i} {'xyz'[k]}: FD {fd:+.10f}  linus-FD {g[i, k] - fd:+.2e}  PySCF-FD {ref['gradient'][i, k] - fd:+.2e}", flush=True)


if __name__ == "__main__":
    main()
