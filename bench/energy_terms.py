"""Attribute an RHF energy difference linus - PySCF to its terms on a fixed density D:
Tr(D dS), Tr(D dT), Tr(D dV), 1/2 Tr(D dJ), -1/4 Tr(D dK), with linus J/K from FastJk at a
tight threshold and PySCF's from libcint (direct, no screening below 1e-16).

python bench/energy_terms.py <xyz> <basis> <dm.bin> <linus_prefix>
  <linus_prefix>.J.bin/.K.bin from `jk_bench ... --thresh 1e-16 --out <prefix>`;
  one-electron matrices from `dump_ints` (run here).
"""
import json
import pathlib
import subprocess
import sys
import tempfile

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from pyscf_jk import make_mol  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]


def main():
    xyz, basis, dmf, pre = sys.argv[1:5]
    from pyscf import scf
    mol = make_mol(xyz, basis)
    n = mol.nao
    dm = np.fromfile(dmf, dtype="<f8").reshape(n, n)
    ld = lambda f: np.fromfile(f, dtype="<f8").reshape(n, n)
    with tempfile.TemporaryDirectory() as td:
        inp = pathlib.Path(td) / "in.json"
        inp.write_text(json.dumps({"z": [int(z) for z in mol.atom_charges()], "coords": mol.atom_coords().tolist(),
                                   "basis": basis, "aux": None, "pure": True, "eri": False}))
        subprocess.run([str(ROOT / "target/release/examples/dump_ints"), str(inp), td], check=True, capture_output=True)
        lin = {k: ld(pathlib.Path(td) / f"{k}.bin") for k in ("S", "T", "V")}
    ref = {"S": mol.intor("int1e_ovlp"), "T": mol.intor("int1e_kin"), "V": mol.intor("int1e_nuc")}
    mf = scf.RHF(mol)
    mf.direct_scf_tol = 1e-16
    vj, vk = scf.hf.get_jk(mol, dm, 1, mf.init_direct_scf(mol))
    out = {k: float(np.einsum("ij,ij", dm, lin[k] - ref[k])) for k in ("T", "V")}
    out["S_maxabs"] = float(np.abs(lin["S"] - ref["S"]).max())
    out["V_maxabs"] = float(np.abs(lin["V"] - ref["V"]).max())
    J, K = ld(pre + ".J.bin"), ld(pre + ".K.bin")
    out["J"] = float(0.5 * np.einsum("ij,ij", dm, J - vj))
    out["K"] = float(-0.25 * np.einsum("ij,ij", dm, K - vk))
    out["J_maxabs"] = float(np.abs(J - vj).max())
    out["K_maxabs"] = float(np.abs(K - vk).max())
    out["sum_energy_terms"] = out["T"] + out["V"] + out["J"] + out["K"]
    print(json.dumps(out))


if __name__ == "__main__":
    main()
