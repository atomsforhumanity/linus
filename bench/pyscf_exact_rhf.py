"""Exact RHF reference with PySCF: in-core 8-fold ERI tensor (no screening, no incremental Fock
builds), tightly converged.

python bench/pyscf_exact_rhf.py <xyz> <basis>   -> one JSON line
"""
import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from pyscf_jk import make_mol  # noqa: E402


def main():
    from pyscf import scf
    mol = make_mol(sys.argv[1], sys.argv[2])
    n = mol.nao
    gb = (n * (n + 1) // 2) ** 2 / 2 * 8 / 1e9
    if gb > 4.0:  # shared node: in-core references only for small validation molecules
        raise SystemExit(f"8-fold ERI tensor would need {gb:.1f} GB (> 4 GB); refusing")
    mol.max_memory = 6000
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-12
    mf.conv_tol_grad = 1e-9
    mf.max_cycle = 200
    mf.init_guess = "minao"
    t0 = time.perf_counter()
    mf._eri = mol.intor("int2e", aosym="s8")
    e = mf.kernel()
    print(json.dumps({"energy": e, "converged": bool(mf.converged), "cycles": int(mf.cycles), "nbf": mol.nao,
                      "wall": time.perf_counter() - t0}))


if __name__ == "__main__":
    main()
