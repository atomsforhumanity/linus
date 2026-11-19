"""PySCF 2.15's own VV10 nuclear gradient with grid response against central differences of its
VV10 energy at a fixed AO density matrix (basis and NLC grid move with the atoms).

  .refenv/bin/python reference/checks/pyscf_vv10_grad_fd.py <xyz> <basis> <xc> [--charge Q] [--mult M] [--level L]

Prints max |analytic - FD| for PySCF's full-response gradient (rks_grad.get_nlc_vxc_full_response)
and for its fixed-grid gradient (rks_grad.get_nlc_vxc): the first should vanish up to the FD error
if PySCF's grid response is exact.
"""
import argparse, json, pathlib, sys

import numpy as np
from pyscf import dft, gto, lib
from pyscf.grad import rks as rks_grad

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "tests" / "python"))
from common import BOHR, pyscf_basis  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("xyz")
    ap.add_argument("basis")
    ap.add_argument("xc")
    ap.add_argument("--charge", type=int, default=0)
    ap.add_argument("--mult", type=int, default=1)
    ap.add_argument("--level", type=int, default=1)
    ap.add_argument("--h", type=float, default=1e-4)
    a = ap.parse_args()
    lib.num_threads(8)
    atoms = [ln.split() for ln in pathlib.Path(a.xyz).read_text().splitlines()[2:] if ln.strip()]
    syms = sorted({x[0] for x in atoms})
    xyz = np.array([[float(v) / BOHR for v in x[1:4]] for x in atoms])

    def mol_at(c):
        return gto.M(atom=[(x[0], c[i]) for i, x in enumerate(atoms)], unit="Bohr", basis=pyscf_basis(a.basis, syms),
                     charge=a.charge, spin=a.mult - 1, verbose=0)

    mol = mol_at(xyz)
    mf = dft.RKS(mol) if a.mult == 1 else dft.UKS(mol)
    mf.xc = a.xc
    mf.small_rho_cutoff = 0.0
    mf.nlcgrids.level = a.level
    mf.kernel()
    dm = mf.make_rdm1()
    dm = dm if a.mult == 1 else dm[0] + dm[1]
    ni = mf._numint

    def e_nlc(c):
        m = mol_at(c)
        g = dft.gen_grid.Grids(m)
        g.level = a.level
        g.build(with_non0tab=True)
        return ni.nr_nlc_vxc(m, g, a.xc, dm)[1]

    natm = mol.natm
    aoslices = mol.aoslice_by_atom()

    def total(excsum, vmat):
        g = np.array(excsum, dtype=float) if excsum is not None else np.zeros((natm, 3))
        for i, (_, _, p0, p1) in enumerate(aoslices):
            g[i] += 2 * np.einsum("xij,ij->x", vmat[:, p0:p1], dm[p0:p1])
        return g

    grids = mf.nlcgrids
    ex, vm = rks_grad.get_nlc_vxc_full_response(ni, mol, grids, a.xc, dm)
    g_full = total(ex, vm)
    _, vm0 = rks_grad.get_nlc_vxc(ni, mol, grids, a.xc, dm)
    g_fixed = total(None, vm0)
    fd = np.zeros((natm, 3))
    for i in range(natm):
        for k in range(3):
            cp, cm = xyz.copy(), xyz.copy()
            cp[i, k] += a.h
            cm[i, k] -= a.h
            fd[i, k] = (e_nlc(cp) - e_nlc(cm)) / (2 * a.h)
    print(json.dumps({"xyz": pathlib.Path(a.xyz).name, "basis": a.basis, "xc": a.xc, "level": a.level, "h": a.h,
                      "max_abs_full_minus_fd": float(np.abs(g_full - fd).max()), "max_abs_fixed_minus_fd": float(np.abs(g_fixed - fd).max()),
                      "max_abs_fd": float(np.abs(fd).max()), "full": g_full.tolist(), "fd": fd.tolist()}))


if __name__ == "__main__":
    main()
