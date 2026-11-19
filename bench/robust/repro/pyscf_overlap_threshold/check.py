#!/usr/bin/env python
"""PySCF 2.15 drops overlap eigenvectors below 1e-6, linus below 1e-8 (`scf.lindep`).

    .refenv/bin/python bench/robust/repro/pyscf_overlap_threshold/check.py [ID] [BASIS]

Default: cubane (gmtkn55.DC13.ISO_C8H8_41), def2-TZVP, HF with RI-JK (def2-universal-jkfit), the
sweep's settings. Prints the smallest overlap eigenvalue, linus's energy, PySCF's with its default
threshold and with `overlap_zero_eigenvalue_threshold` = 1e-8. LINUS_BIN, NTHREADS as elsewhere.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "bench" / "robust" / "repro"))
import sets  # noqa: E402
from pyscf import gto, lib, scf  # noqa: E402
from pyscf.scf import hf  # noqa: E402

LINUS = os.environ.get("LINUS_BIN", str(REPO / "target" / "release" / "linus"))
NT = int(os.environ.get("NTHREADS", "4"))
lib.num_threads(NT)
mid = sys.argv[1] if len(sys.argv) > 1 else "gmtkn55.DC13.ISO_C8H8_41"
basis = sys.argv[2] if len(sys.argv) > 2 else "def2-tzvp"
rec = next(r for r in sets.load() if r["id"] == mid)
mol = gto.M(atom=list(zip(rec["symbols"], rec["geometry_angstrom"])), basis=basis, charge=rec["charge"],
            spin=rec["multiplicity"] - 1, verbose=0)
print("smallest overlap eigenvalues", np.linalg.eigvalsh(mol.intor("int1e_ovlp"))[:3])
req = {"structure": {"symbols": rec["symbols"], "geometry": mol.atom_coords().tolist(), "charge": rec["charge"],
                     "multiplicity": rec["multiplicity"]},
       "method": "hf", "basis": basis, "calctype": "energy", "threads": NT,
       "jk": "df", "scf": {"convergence": "extreme", "max_iter": 300}}  # (int_thresh 1e-14: not for jk df, D41)
out = json.loads(subprocess.run([LINUS, "--json"], input=json.dumps(req), capture_output=True, text=True).stdout)
print("linus", out["energy"], "dropped functions", ((out.get("resolved") or {}).get("scf") or {}).get("dropped_functions"))
for thr in (hf.overlap_zero_eigenvalue_threshold, 1e-8):
    hf.overlap_zero_eigenvalue_threshold = thr
    mf = (scf.RHF(mol) if mol.spin == 0 else scf.UHF(mol)).density_fit(auxbasis="def2-universal-jkfit")
    mf.conv_tol, mf.conv_tol_grad = 1e-12, 1e-9
    e = mf.kernel()
    print(f"PySCF, threshold {thr:g}: {e}  linus - PySCF {out['energy'] - e:.3e}")
