"""Psi4 1.11 DF J/K/wK energy traces at the core-Hamiltonian density, to compare with
`df_bench --omega w` (field "trace"): tr(D J), tr(D K), tr(D wK) are invariant to AO ordering.

  .refenv/bin/python bench/psi4_jk_traces.py <molecule.xyz> [basis] [aux] [omega] [jk_type]
jk_type: MEM_DF (default) or DISK_DF. Psi4 runs in a temporary working directory
(tests/python/common.psi4_cwd).
"""
import json, pathlib, sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tests" / "python"))
from common import psi4_cwd, read_xyz_bohr  # noqa: E402


def main():
    xyz = sys.argv[1]
    basis = sys.argv[2] if len(sys.argv) > 2 else "def2-svp"
    aux = sys.argv[3] if len(sys.argv) > 3 else "def2-universal-jkfit"
    omega = float(sys.argv[4]) if len(sys.argv) > 4 else 0.3
    jk_type = sys.argv[5] if len(sys.argv) > 5 else "MEM_DF"
    syms, pos = read_xyz_bohr(xyz)
    atoms = [(s, *p) for s, p in zip(syms, pos)]
    psi4_cwd()  # before importing psi4 (timer.dat names the host)
    import psi4

    psi4.core.be_quiet()
    psi4.set_memory("16 GB")
    psi4.set_num_threads(8)
    mol = psi4.geometry("0 1\nunits bohr\nno_reorient\nno_com\nsymmetry c1\n" + "\n".join(f"{s} {x:.12f} {y:.12f} {z:.12f}" for s, x, y, z in atoms))
    psi4.set_options({"basis": basis, "df_basis_scf": aux, "puream": True, "ints_tolerance": 0.0, "df_fitting_condition": 1e-15, "scf_type": "df"})
    wfn = psi4.core.Wavefunction.build(mol, psi4.core.get_global_option("basis"))
    bas = wfn.basisset()
    auxb = psi4.core.BasisSet.build(mol, "DF_BASIS_SCF", aux, "JKFIT", aux, puream=True)
    mints = psi4.core.MintsHelper(bas)
    S = mints.ao_overlap().np
    H = mints.ao_kinetic().np + mints.ao_potential().np
    w, U = np.linalg.eigh(S)
    X = U[:, w > 1e-8] / np.sqrt(w[w > 1e-8])
    e, Cp = np.linalg.eigh(X.T @ H @ X)
    C = X @ Cp
    nocc = sum(int(z) for z in [mol.Z(i) for i in range(mol.natom())]) // 2
    Cocc = C[:, :nocc]
    D = Cocc @ Cocc.T
    jk = psi4.core.JK.build(bas, auxb, jk_type)
    jk.set_do_wK(True)
    jk.set_omega(omega)
    jk.set_memory(int(8e9))
    jk.initialize()
    jk.C_left_add(psi4.core.Matrix.from_array(Cocc))
    jk.compute()
    J = jk.J()[0].np
    K = jk.K()[0].np
    wK = jk.wK()[0].np
    print(json.dumps({"xyz": pathlib.Path(xyz).name, "jk": jk_type, "DJ": float(np.sum(D * J)), "DK": float(np.sum(D * K)), "DKlr": float(np.sum(D * wK)),
                      "max_asym_wK": float(np.abs(wK - wK.T).max())}))


if __name__ == "__main__":
    main()
