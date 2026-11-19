#!/usr/bin/env python
"""PySCF HF with its own (screened) ECP matrix and with linus's full one, against linus.

    .refenv/bin/python bench/robust/repro/pyscf_ecp_screening/check.py ID BASIS [BASIS ...]
    .refenv/bin/python bench/robust/repro/pyscf_ecp_screening/check.py heavy_built.XeF2 def2-svp def2-tzvp
    METHOD=pbe .refenv/bin/python bench/robust/repro/pyscf_ecp_screening/check.py heavy_built.XeF2 def2-svp

METHOD (default hf) may be a functional PySCF and linus name alike (pbe, tpss, ...): DFT on
PySCF's level-3 grid, which linus reproduces as pyscf:3 (small_rho_cutoff 0, as run_ref.py).

Builds the molecule as tests/python/ecp_pyscf_check.py does (linus's basis data, the same
function order) and takes linus's ECP matrix from crates/linus-ints/examples/ecp_dump
(ECP_DUMP, default target/release/examples/ecp_dump). If E(PySCF, linus V_ECP) equals
E(linus), the PySCF-linus difference is libcint's overlap screening of semi-local ECP integrals
(check_3c_overlap), which drops integrals between functions on opposite sides of the ECP atom.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from pyscf import dft, lib, scf

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "tests" / "python"))
sys.path.insert(0, str(REPO / "bench" / "robust" / "repro"))
import ecp_pyscf_check as e  # noqa: E402
import sets as molecules  # noqa: E402

lib.num_threads(int(os.environ.get("NTHREADS", "2")))
ECP_DUMP = os.environ.get("ECP_DUMP", str(REPO / "target" / "release" / "examples" / "ecp_dump"))
LINUS = os.environ.get("LINUS_BIN", str(REPO / "target" / "release" / "linus"))
B2A = 0.529177210544
METHOD = os.environ.get("METHOD", "hf")


def check(rec: dict, basis: str) -> dict:
    atoms = [(s, tuple(c / B2A for c in x)) for s, x in zip(rec["symbols"], rec["geometry_angstrom"])]
    mol = e.pyscf_mol(atoms, basis, False)
    mol.charge, mol.spin = rec["charge"], rec["multiplicity"] - 1
    mol.build()
    nao = mol.nao
    _, raw = e.run_linus(ECP_DUMP, atoms, basis, True, ["matrix"])
    v_lin = raw[: nao * nao].reshape(nao, nao)
    v_py = mol.intor("ECPscalar")
    res = {"id": rec["id"], "basis": basis, "method": METHOD, "nbf": nao,
           "screened_max_abs": float(np.abs(np.where(v_py == 0.0, v_lin, 0.0)).max()),
           "kept_max_diff": float(np.abs(np.where(v_py == 0.0, 0.0, v_lin - v_py)).max())}
    h0 = mol.intor("int1e_kin") + mol.intor("int1e_nuc")
    for label, v in (("pyscf_v", v_py), ("linus_v", v_lin)):
        if METHOD == "hf":
            mf = scf.RHF(mol) if mol.spin == 0 else scf.UHF(mol)
        else:
            mf = dft.RKS(mol) if mol.spin == 0 else dft.UKS(mol)
            mf.xc, mf.grids.level, mf.small_rho_cutoff = METHOD, 3, 0.0
        mf.get_hcore = lambda *a, h=h0 + v: h
        mf.conv_tol, mf.conv_tol_grad, mf.direct_scf_tol, mf.max_cycle = 1e-12, 1e-9, 1e-14, 300
        mf.kernel()
        res[f"e_{label}"] = float(mf.e_tot)
        res[f"converged_{label}"] = bool(mf.converged)
    req = {"structure": {"symbols": rec["symbols"], "geometry": [[c / B2A for c in x] for x in rec["geometry_angstrom"]],
                         "charge": rec["charge"], "multiplicity": rec["multiplicity"]},
           "method": METHOD, "basis": basis,
           "calctype": "energy", "threads": int(os.environ.get("NTHREADS", "2")),
           **({} if METHOD == "hf" else {"grid": {"xc": "pyscf:3"}}),
           "scf": {"convergence": "extreme", "max_iter": 300},
           "jk": {"kind": "exact", "screening": 1e-14}}  # the default J/K was exact, which int_thresh screens
    out = json.loads(subprocess.run([LINUS, "--json"], input=json.dumps(req), capture_output=True, text=True).stdout)
    res["e_linus"] = out.get("energy")
    if res["e_linus"] is not None:
        res["d_pyscf_v"] = res["e_pyscf_v"] - res["e_linus"]
        res["d_linus_v"] = res["e_linus_v"] - res["e_linus"]
    return res


if __name__ == "__main__":
    rec = next(r for r in molecules.load() if r["id"] == sys.argv[1])
    for basis in sys.argv[2:]:
        print(json.dumps(check(rec, basis)))
