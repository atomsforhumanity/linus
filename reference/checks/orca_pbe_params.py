"""Why ORCA 6.0.0 PBE differs from libxc PBE (PySCF/Psi4) by ~1.4e-5 Eh on water.

ORCA prints XMuePBE 0.219520, CBetaPBE 0.066725, LDAOpt PW91-LDA. libxc's PBE uses
mu = 0.2195149727645171, beta = 0.06672455060314922 and the PW_MOD LDA correlation.
Recompute water PBE/def2-SVP with PySCF (150x974 unpruned) using ORCA's constants.
Run: .refenv/bin/python reference/checks/orca_pbe_params.py
"""
import json, sys
from pathlib import Path
import numpy as np
from pyscf import gto, dft
from pyscf.dft import libxc

ref = Path(__file__).resolve().parents[1] / "out/water/pbe_def2-svp_gradient_exact_grid-fine"
orca = json.loads((ref / "orca.json").read_text())
pyscf_rec = json.loads((ref / "pyscf.json").read_text())
mol = gto.M(atom=[(s, x) for s, x in zip(orca["molecule"]["symbols"], orca["molecule"]["geometry_bohr"])],
            unit="Bohr", basis="def2-svp", verbose=0)

def run(xc):
    mf = dft.RKS(mol); mf.xc = xc
    mf.grids.atom_grid = (150, 974); mf.grids.prune = None; mf.small_rho_cutoff = 0.0
    mf.conv_tol = 1e-12; mf.conv_tol_grad = 1e-9
    e = mf.kernel(); return e, mf

MU_ORCA, BETA_ORCA = 0.21952, 0.066725
gamma = (1 - np.log(2)) / np.pi**2
variants = {  # name: (custom xc code, libxc ext params)
    "libxc PBE": ("PBE", None),
    "mu=0.21952": ("PBEMUORCA", {101: [0.804, MU_ORCA]}),
    "beta=0.066725": ("PBEBETAORCA", {130: {"_beta": BETA_ORCA}}),
    "mu=0.21952 + beta=0.066725": ("PBEBOTHORCA", {101: [0.804, MU_ORCA], 130: {"_beta": BETA_ORCA}}),
}
print(f"ORCA  E = {orca['energy']:.10f}")
e0, mf0 = run("PBE")
print(f"libxc PBE (PySCF) E = {e0:.10f}  E - E(ORCA) = {e0 - orca['energy']:+.3e}")
# SCF energy is variational: to first order, changing the functional changes E by
# E_xc'[rho0] - E_xc[rho0] on the converged libxc-PBE density rho0.
dm = mf0.make_rdm1(); ni = mf0._numint
exc0 = ni.nr_rks(mol, mf0.grids, "PBE", dm)[1]
for name, (code, ext) in variants.items():
    if ext is None:
        continue
    libxc.register_custom_functional_(code, "PBE", ext_params=ext)
    exc = ni.nr_rks(mol, mf0.grids, code, dm)[1]
    print(f"{name:30s} dE_xc = {exc - exc0:+.3e}  predicted E - E(ORCA) = {e0 + exc - exc0 - orca['energy']:+.3e}")
e_pw = ni.nr_rks(mol, mf0.grids, "LDA_C_PW", dm)[1]
e_pwmod = ni.nr_rks(mol, mf0.grids, "LDA_C_PW_MOD", dm)[1]
print(f"E_c[LDA_C_PW] - E_c[LDA_C_PW_MOD] on the same density = {e_pw - e_pwmod:+.3e}")
