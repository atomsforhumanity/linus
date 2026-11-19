"""How good is the RI-J fit for PCM surface potentials? An independent check in PySCF 2.15: the C-PCM
energy of a fixed HF/def2-SVP density (PySCF's own PCM SCF) with the exact surface potential
(int3c2e on PySCF's Gaussian surface charges) and with the potential of the Coulomb-metric fitted
density, sum_P c_P (P|g), c = V^-1 (P|ab) D (the density-fitted PCM approximation of
linus's `solvent.df: true`).

  .refenv/bin/python tests/python/pyscf_ri_pcm_check.py bench/molecules/pcm/water.xyz \
      [--aux def2-universal-jfit] [--eps 7.4257]

Prints one JSON line: E_pcm exact and density-fitted (Eh), their difference (kcal/mol) and the
max/rms error of the potential at the points (au).
"""
import argparse, json, pathlib, sys

import numpy as np
from pyscf import df, gto, scf
from pyscf.solvent import pcm  # noqa: F401
from scipy.special import erf

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from compare_ints_pyscf import pyscf_basis  # noqa: E402

BOHR = 0.529177210544
KCAL = 627.5094740631


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("xyz")
    ap.add_argument("--basis", default="def2-svp")
    ap.add_argument("--aux", default="def2-universal-jfit")
    ap.add_argument("--eps", type=float, default=7.4257)
    ap.add_argument("--charge", type=int, default=0)
    a = ap.parse_args()
    atoms = [ln.split() for ln in pathlib.Path(a.xyz).read_text().splitlines()[2:] if ln.strip()]
    syms = sorted({x[0] for x in atoms})
    mol = gto.M(atom=[(x[0], [float(v) / BOHR for v in x[1:4]]) for x in atoms], unit="Bohr", basis=pyscf_basis(a.basis, syms),
                charge=a.charge, verbose=0)
    mf = scf.RHF(mol).PCM()
    mf.with_solvent.method = "C-PCM"
    mf.with_solvent.eps = a.eps
    mf.kernel()
    dm = mf.make_rdm1()
    sol = mf.with_solvent
    coords = sol.surface["grid_coords"]
    zeta = sol.surface["charge_exp"]
    fakemol = gto.fakemol_for_charges(coords, expnt=zeta ** 2)
    v_exact = np.einsum("ijL,ij->L", df.incore.aux_e2(mol, fakemol, intor="int3c2e"), dm)
    auxmol = df.addons.make_auxmol(mol, pyscf_basis(a.aux, syms))
    gamma = np.einsum("ijP,ij->P", df.incore.aux_e2(mol, auxmol, intor="int3c2e"), dm)
    c = np.linalg.solve(auxmol.intor("int2c2e"), gamma)
    v_ri = gto.mole.intor_cross("int2c2e", auxmol, fakemol).T @ c
    nuc = np.zeros(len(coords))
    for ia in range(mol.natm):
        r = np.linalg.norm(coords - mol.atom_coord(ia), axis=1)
        nuc += mol.atom_charge(ia) * erf(zeta * r) / r
    K, R = sol._intermediates["K"], sol._intermediates["R"]
    e = {}
    for name, ve in [("exact", v_exact), ("df", v_ri)]:
        phi = nuc - ve
        e[name] = 0.5 * np.linalg.solve(K, R @ phi) @ phi
    dv = v_ri - v_exact
    print(json.dumps({"xyz": pathlib.Path(a.xyz).name, "basis": a.basis, "aux": a.aux, "naux": auxmol.nao, "npoints": len(coords),
                      "e_pcm_exact": e["exact"], "e_pcm_pyscf": float(sol.e), "e_pcm_df": e["df"], "dE_kcal": (e["df"] - e["exact"]) * KCAL,
                      "relative": (e["df"] - e["exact"]) / abs(e["exact"]), "max_abs_dv": float(np.abs(dv).max()), "rms_dv": float(np.sqrt(np.mean(dv ** 2)))}))


if __name__ == "__main__":
    main()
