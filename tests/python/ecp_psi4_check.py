#!/usr/bin/env python
"""ECP matrices of Psi4 (libecpint) against linus and PySCF, through basis-order-free
invariants: the generalized eigenvalues of (U, S) (U the ECP matrix, S the overlap) of the same
basis set, which no reordering, sign or normalization convention of the functions changes.

    .refenv/bin/python tests/python/ecp_psi4_check.py [--cases HI,PtCl4] [--basis def2-svp]

Psi4 uses its own def2 library (same functions and ECPs as the BSE files; checked with
nbasis); linus and PySCF the repository's BSE JSON (ecp_pyscf_check.py). Psi4 runs in a
temporary working directory (common.psi4_cwd).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import scipy.linalg

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import psi4_cwd  # noqa: E402
from ecp_pyscf_check import CASES, REPO, pyscf_mol, run_linus  # noqa: E402


def psi4_mats(atoms, basis):
    import psi4

    psi4.core.be_quiet()
    psi4.set_memory("2 GB")
    geom = "\n".join(f"{s} {x[0]:.12f} {x[1]:.12f} {x[2]:.12f}" for s, x in atoms)
    mol = psi4.geometry(f"units bohr\nno_com\nno_reorient\nsymmetry c1\n{geom}")
    psi4.set_options({"basis": basis, "puream": True})
    wfn = psi4.core.Wavefunction.build(mol, psi4.core.get_global_option("BASIS"))
    mints = psi4.core.MintsHelper(wfn.basisset())
    return np.asarray(mints.ao_ecp()), np.asarray(mints.ao_overlap())


def geig(u, s):
    return np.sort(scipy.linalg.eigh(u, s, eigvals_only=True))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="HI,CH3I,PdCl2PH3,PtCl4,Au2,CsF,RbCl,AgCl")
    ap.add_argument("--basis", default="def2-svp,def2-tzvp")
    ap.add_argument("--bin", default=str(REPO / "target" / "release" / "examples" / "ecp_dump"))
    a = ap.parse_args()
    a.bin = str(Path(a.bin).resolve())
    psi4_cwd()  # before importing psi4 (timer.dat names the host)
    for name in a.cases.split(","):
        atoms = CASES[name]
        for basis in a.basis.split(","):
            mol = pyscf_mol(atoms, basis, False)
            n = mol.nao
            out, raw = run_linus(a.bin, atoms, basis, True, ["matrix"])
            ul = raw[: n * n].reshape(n, n)
            s = mol.intor("int1e_ovlp")
            up = mol.intor("ECPscalar")
            u4, s4 = psi4_mats(atoms, basis)
            if u4.shape[0] != n:
                print(f"{name} {basis}: Psi4 has {u4.shape[0]} functions, linus {n}")
                continue
            el, ep, e4 = geig(ul, s), geig(up, s), geig(u4, s4)
            sc = np.abs(el).max()
            print(f"{name:10s} {basis:10s} nbf={n:4d}  max|eig|={sc:.2e}  linus-PySCF {np.abs(el - ep).max():.2e}  "
                  f"linus-Psi4 {np.abs(el - e4).max():.2e}  PySCF-Psi4 {np.abs(ep - e4).max():.2e}", flush=True)


if __name__ == "__main__":
    main()
