"""Compare linus density fitting with PySCF (libcint), using identical basis definitions.

  .venv/bin/python tests/python/compare_df_pyscf.py ints <molecule.xyz> [basis] [aux] [omega]
      eri3c (k|mn), (k|l) and their erf(omega r)/r versions against int3c2e / int2c2e
      (mol.with_range_coulomb).
  .venv/bin/python tests/python/compare_df_pyscf.py scf <molecule.xyz> [basis] [aux] [threads] [charge] [mult]
      RHF (UHF if mult > 1) energies: linus `scf --jk ri` against PySCF density_fit() with the
      same auxiliary basis.
Geometries are passed in bohr with linus's bohr constant, so both codes see the same nuclei.
"""
import json, pathlib, subprocess, sys, tempfile

import numpy as np
from pyscf import df, gto, scf

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from common import EXAMPLES, elements, pyscf_basis, pyscf_mol  # noqa: E402


def make_mol(xyz, basis, charge=0, mult=1):
    mol = pyscf_mol(xyz, basis, charge, mult)
    return mol, elements(mol)


def ints(xyz, basis, aux, omega):
    mol, syms = make_mol(xyz, basis)
    auxmol = df.make_auxmol(mol, pyscf_basis(aux, syms))
    nbf, naux = mol.nao, auxmol.nao
    with tempfile.TemporaryDirectory() as td:
        inp = pathlib.Path(td) / "in.json"
        inp.write_text(json.dumps({"z": [gto.charge(s) for s in mol.elements], "coords": mol.atom_coords().tolist(), "basis": basis, "aux": aux, "omega": omega}))
        info = json.loads(subprocess.check_output([str(EXAMPLES / "dump_df"), str(inp), td]))
        assert info["nbf"] == nbf and info["naux"] == naux, (info, nbf, naux)
        ld = lambda n, shape: np.fromfile(pathlib.Path(td) / n, dtype="<f8").reshape(shape)
        res = {}
        j3 = df.incore.aux_e2(mol, auxmol, intor="int3c2e").reshape(nbf * nbf, naux).T
        res["J3C"] = np.abs(ld("J3C.bin", (naux, nbf * nbf)) - j3).max()
        res["J2C"] = np.abs(ld("J2C.bin", (naux, naux)) - auxmol.intor("int2c2e")).max()
        res["max_J3C"] = np.abs(j3).max()
        if omega > 0:
            with mol.with_range_coulomb(omega), auxmol.with_range_coulomb(omega):
                j3w = df.incore.aux_e2(mol, auxmol, intor="int3c2e").reshape(nbf * nbf, naux).T
                j2w = auxmol.intor("int2c2e")
            res["J3C_LR"] = np.abs(ld("J3C_LR.bin", (naux, nbf * nbf)) - j3w).max()
            res["J2C_LR"] = np.abs(ld("J2C_LR.bin", (naux, naux)) - j2w).max()
            res["max_J3C_LR"] = np.abs(j3w).max()
    print(json.dumps({"xyz": pathlib.Path(xyz).name, "basis": basis, "aux": aux, "omega": omega, "nbf": nbf, "naux": naux,
                      **{k: float(f"{v:.3e}") for k, v in res.items()}}))


def scf_energy(xyz, basis, aux, threads, charge, mult):
    mol, syms = make_mol(xyz, basis, charge, mult)
    mf = (scf.RHF(mol) if mult == 1 else scf.UHF(mol)).density_fit(auxbasis=pyscf_basis(aux, syms))
    mf.conv_tol = 1e-12
    mf.conv_tol_grad = 1e-9
    mf.max_cycle = 200
    ep = mf.kernel()
    cmd = [str(EXAMPLES / "scf"), str(xyz), basis, "--jk", "ri", "--aux", aux, "--threads", str(threads),
           "--charge", str(charge), "--mult", str(mult), "--e-conv", "1e-12", "--g-conv", "1e-10"] + (["--uhf"] if mult > 1 else [])
    el = json.loads(subprocess.check_output(cmd))
    print(json.dumps({"xyz": pathlib.Path(xyz).name, "basis": basis, "aux": aux, "charge": charge, "mult": mult, "linus": el["energy"],
                      "pyscf": ep, "pyscf_converged": bool(mf.converged), "diff": el["energy"] - ep}))


if __name__ == "__main__":
    mode, xyz = sys.argv[1], sys.argv[2]
    basis = sys.argv[3] if len(sys.argv) > 3 else "def2-svp"
    aux = sys.argv[4] if len(sys.argv) > 4 else "def2-universal-jkfit"
    if mode == "ints":
        ints(xyz, basis, aux, float(sys.argv[5]) if len(sys.argv) > 5 else 0.3)
    else:
        a = sys.argv[5:]
        scf_energy(xyz, basis, aux, int(a[0]) if a else 8, int(a[1]) if len(a) > 1 else 0, int(a[2]) if len(a) > 2 else 1)
