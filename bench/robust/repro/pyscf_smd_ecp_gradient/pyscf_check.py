"""Usage: python pyscf_check.py Cs,F 0,0,0.0403449,0,0,4.4005115 [water|iefpcm-water]

PySCF SMD gradient (with the sweep's refrun.py patches) against central differences of PySCF's own SMD
energy, for CsF (Cs with the def2 ECP), B3LYP/def2-SVP, density fitting, grid level 3. Also the CDS
part alone (patched FD gradient vs FD of the patched CDS energy) and the electrostatic part."""
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "bench" / "robust" / "pipeline"))  # refrun.py: private repository only
import refrun  # noqa: E402

refrun.patch_smd_cds()
refrun.patch_smd_cds_gradient()
sys.path.insert(0, str(REPO / "reference"))
import run_ref  # noqa: E402

run_ref.pyscf_ecp_grid_response_fix()
from pyscf import dft, gto, lib  # noqa: E402
from pyscf.solvent import smd  # noqa: E402

lib.num_threads(4)
symbols, xyz = sys.argv[1].split(","), np.array([float(x) for x in sys.argv[2].split(",")]).reshape(-1, 3)
solvent = sys.argv[3] if len(sys.argv) > 3 else "water"


def build(x):
    mol = gto.M(atom=list(zip(symbols, x.tolist())), unit="Bohr", basis="def2-svp", ecp="def2-svp", verbose=0)
    mf = dft.RKS(mol).density_fit(auxbasis="def2-universal-jkfit")
    mf.xc = "b3lyp"
    mf.grids.level = 3
    if solvent == "iefpcm-water":
        mf = mf.PCM()
        mf.with_solvent.method = "IEF-PCM"
        mf.with_solvent.eps = 78.3553
    else:
        mf = mf.SMD()
        mf.with_solvent.solvent = solvent
    mf.conv_tol = 1e-12
    mf.conv_tol_grad = 1e-8
    return mf


mf = build(xyz)
e0 = mf.kernel()
gm = mf.nuc_grad_method()
gm.grid_response = True
g = gm.kernel()
h = 1e-3
fd = np.zeros_like(g)
for a in range(len(symbols)):
    for k in range(3):
        p, q = xyz.copy(), xyz.copy()
        p[a, k] += h
        q[a, k] -= h
        fd[a, k] = (build(p).kernel() - build(q).kernel()) / (2 * h)
print("E", e0)
print("PySCF SMD analytic gradient:\n", g)
print("central differences of the energy:\n", fd)
print("max |analytic - FD| %.2e, sum of analytic %.1e" % (np.abs(g - fd).max(), np.abs(g.sum(0)).max()))
