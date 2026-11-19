#!/usr/bin/env python
"""PySCF reference data for crates/linus-ints/tests/ecp_pyscf.rs (ECP matrices and ECP
gradient contractions), written to crates/linus-ints/tests/fixtures/pyscf_ecp.json.

    .refenv/bin/python tests/python/gen_ecp_fixtures.py

PySCF gets linus's basis data and ECPs (repository BSE JSON, linus's shell order); the density
for the gradient is deterministic (see `density`), so the Rust test rebuilds it.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from pyscf import gto

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ecp_pyscf_check import REPO, ecp_deriv_pyscf, pyscf_mol  # noqa: E402

CASES = {
    "HI": ("def2-svp", [("I", (0.0, 0.0, 0.0)), ("H", (0.3, 0.2, 3.04))]),
    "PtH2Cl": ("def2-svp", [("Pt", (0.1, 0.0, -0.1)), ("H", (2.9, 0.4, 0.0)), ("H", (-0.3, 3.0, 0.2)), ("Cl", (-0.2, -0.4, -4.4))]),
    "RbF": ("def2-tzvp", [("Rb", (0.0, 0.0, 0.0)), ("F", (0.2, -0.1, 4.4))]),
}


def density(n: int) -> np.ndarray:
    """D_ij = cos(i + 2j) + cos(j + 2i), scaled by 1/n (symmetric, deterministic)."""
    i = np.arange(n)
    d = np.cos(i[:, None] + 2 * i[None, :])
    return (d + d.T) / n


def main():
    out = {}
    for name, (basis, atoms) in CASES.items():
        mol = pyscf_mol(atoms, basis, False)
        v = mol.intor("ECPscalar")
        d = density(mol.nao)
        g = np.einsum("axij,ij->ax", ecp_deriv_pyscf(mol), d)
        out[name] = {"basis": basis, "z": [gto.charge(s) for s, _ in atoms], "coords": [list(x) for _, x in atoms],
                     "nbf": mol.nao, "ecp": v.ravel().tolist(), "grad": g.tolist()}
    p = REPO / "crates" / "linus-ints" / "tests" / "fixtures" / "pyscf_ecp.json"
    p.write_text(json.dumps(out))
    print(f"wrote {p} ({p.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
