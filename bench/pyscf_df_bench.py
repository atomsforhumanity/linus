"""PySCF density-fitting J/K timing with linus's basis definitions.

  .venv/bin/python bench/pyscf_df_bench.py <xyz> <basis> <aux> <threads> [charge] [reps]

Times the DF tensor build (with_df.build(): 3-center integrals + Cholesky of the metric) and
get_jk for the core-Hamiltonian closed-shell density, with the density tagged with its
occupied orbitals so PySCF's exchange runs in the occupied space (as linus does).
Prints one JSON line.
"""
import json, pathlib, sys, time

import numpy as np
from pyscf import df, gto, lib, scf

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests/python"))
from compare_ints_pyscf import pyscf_basis  # noqa: E402

BOHR = 0.529177210544


def main():
    xyz, basis, aux, threads = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
    charge = int(sys.argv[5]) if len(sys.argv) > 5 else 0
    reps = int(sys.argv[6]) if len(sys.argv) > 6 else 3
    lib.num_threads(threads)
    lines = pathlib.Path(xyz).read_text().splitlines()
    n = int(lines[0])
    atoms = [(ln.split()[0], [float(x) / BOHR for x in ln.split()[1:4]]) for ln in lines[2:2 + n]]
    syms = sorted({a[0] for a in atoms})
    mol = gto.M(atom=atoms, unit="Bohr", basis=pyscf_basis(basis, syms), cart=False, verbose=0, charge=charge)
    mf = scf.RHF(mol).density_fit(auxbasis=pyscf_basis(aux, syms))
    mf.max_memory = 14000
    mf.with_df.max_memory = 14000
    h = mf.get_hcore()
    s = mf.get_ovlp()
    e, c = mf.eig(h, s)
    nocc = mol.nelectron // 2
    occ = np.zeros(c.shape[1])
    occ[:nocc] = 2.0
    dm = lib.tag_array(c[:, :nocc] @ c[:, :nocc].T * 2.0, mo_coeff=c, mo_occ=occ)
    t0 = time.perf_counter()
    mf.with_df.build()
    t_setup = time.perf_counter() - t0
    tjk = []
    for _ in range(reps):
        t0 = time.perf_counter()
        vj, vk = mf.get_jk(mol, dm)
        tjk.append(time.perf_counter() - t0)
    tj = []
    for _ in range(reps):
        t0 = time.perf_counter()
        vj = mf.get_j(mol, dm)
        tj.append(time.perf_counter() - t0)
    cderi = mf.with_df._cderi
    gb = cderi.nbytes / 2**30 if isinstance(cderi, np.ndarray) else None
    print(json.dumps({"program": "pyscf", "version": __import__("pyscf").__version__, "xyz": pathlib.Path(xyz).name, "basis": basis, "aux": aux,
                      "threads": threads, "nbf": mol.nao, "naux": mf.with_df.get_naoaux(), "t_setup": t_setup, "t_jk": float(np.median(tjk)),
                      "t_j": float(np.median(tj)), "incore": isinstance(cderi, np.ndarray), "cderi_gb": gb}))


if __name__ == "__main__":
    main()
