#!/usr/bin/env python
"""PySCF 2.15 SMD G_CDS of molecules with ECP atoms: mol.atom_charges() (Z - ECP core) vs proton numbers.

    .refenv/bin/python bench/robust/repro/pyscf_smd_ecp_cds/check.py [ID ...]   (default: the cases below)

`smd.get_cds_legacy` passes `mol.atom_charges()` to the MNSOL routine as atomic numbers. For an
ECP atom that is Z minus the core electrons, so Xe (54-28) is treated as Fe, Sn as Ti, I as Mn,
Bi as V, Pt as Ar. With the proton numbers PySCF's G_CDS equals linus's to ~2e-12 Eh.
LINUS_BIN (default target/release/linus) runs linus's side.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from pyscf import gto, scf
from pyscf.solvent import smd

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "bench" / "robust" / "repro"))
import sets as molecules  # noqa: E402

LINUS = os.environ.get("LINUS_BIN", str(REPO / "target" / "release" / "linus"))
B2A = 0.529177210544
ids = sys.argv[1:] or ["heavy_built.XeF2", "heavy_built.SnCl4", "heavy_built.I2", "heavy_built.BiCl3",
                       "tm_built.Pt_cl4_2m"]
recs = {r["id"]: r for r in molecules.load() if r["id"] in ids}
for mid in ids:
    rec = recs[mid]
    for solvent in ("water", "toluene"):
        mol = gto.M(atom=list(zip(rec["symbols"], rec["geometry_angstrom"])), basis="def2-svp", ecp="def2-svp",
                    charge=rec["charge"], spin=rec["multiplicity"] - 1, verbose=0)
        so = scf.RHF(mol).SMD().with_solvent
        so.solvent = solvent
        e_ecp = smd.get_cds_legacy(so)[0]
        z = np.asarray([gto.charge(s) for s in mol.elements])
        saved = mol.atom_charges
        mol.atom_charges = lambda: z
        e_z = smd.get_cds_legacy(so)[0]
        mol.atom_charges = saved
        req = {"structure": {"symbols": rec["symbols"], "geometry": [[c / B2A for c in x] for x in rec["geometry_angstrom"]],
                             "charge": rec["charge"], "multiplicity": rec["multiplicity"]},
               "method": "hf", "basis": "def2-svp",
               "jk": "df", "calctype": "energy", "threads": 2,
               "solvent": {"model": "smd", "name": solvent, "eps": float(smd.solvent_db[solvent][5]),
                           "preset": "pyscf", "lebedev": 590, "discretization": "swig", "cds_contact_smoothing": 0.0}}
        out = json.loads(subprocess.run([LINUS, "--json"], input=json.dumps(req), capture_output=True,
                                        text=True).stdout)
        cl = out["solvation"]["cds"]
        print(f"{mid:22s} {solvent:8s} G_CDS linus {cl: .10f}  PySCF(Z-ncore) {e_ecp: .10f} ({e_ecp - cl: .2e})  "
              f"PySCF(Z) {e_z: .10f} ({e_z - cl: .2e})")
