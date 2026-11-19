#!/usr/bin/env python
"""SCF energies and gradients with ECPs: linus (CLI, JSON) against PySCF on the same basis
functions, ECPs and DFT grids.

    .refenv/bin/python tests/python/ecp_scf_check.py [--linus target/release/linus]
        [--cases HI.hf.def2-svp,...] [--grad] [--threads 8]

PySCF gets linus's basis data (the repository's BSE JSON, segmented and in linus's order) and
ECPs, conventional (exact) J/K with 1e-14 screening, conv_tol 1e-12; linus jk=exact,
convergence extreme; DFT on PySCF's level-N grid in both (linus grid pyscf:N, PySCF
small_rho_cutoff 0). PySCF results are cached with LINUS_PYSCF_CACHE=<dir> (common.cached).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from pyscf import dft, gto, scf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import cached, linus_json, mol_key  # noqa: E402
from ecp_pyscf_check import CASES, REPO, pyscf_mol  # noqa: E402

PYSCF_XC = {"pbe": "pbe", "b3lyp": "b3lyp"}
# (charge, multiplicity). Open shells with a nondegenerate singly occupied orbital: in linear
# radicals with a pi hole (HI+, HBr+, AgCl+) the hole's orientation is a soft mode on a DFT grid,
# and PySCF stalls near max |F_ai| 1e-8 at this script's thresholds (linus converges them since
# bench/results/2026-11-07-openshell-scf)
SPECS = {
    "HI": (0, 1), "CH3I": (0, 1), "AgCl": (0, 1), "Au2": (0, 1), "CsF": (0, 1), "RbCl": (0, 1),
    "PdCl2PH3": (0, 1), "PtCl4": (-2, 1), "HgH2": (0, 1), "IrH3": (0, 1),
    "CH2I": (0, 2), "Au2+": (1, 2),
}
DEFAULT = ["HI.hf.def2-svp", "HI.pbe.def2-tzvp", "CH2I.b3lyp.def2-svp", "CH3I.b3lyp.def2-svp", "AgCl.hf.def2-tzvp",
           "Au2.pbe.def2-svp", "Au2+.hf.def2-svp", "CsF.b3lyp.def2-tzvp", "RbCl.hf.def2-svp", "PtCl4.hf.def2-svp",
           "PdCl2PH3.b3lyp.def2-svp", "HgH2.hf.def2-tzvp", "IrH3.pbe.def2-svp"]
GRID_LEVEL = 3


GRID_RESPONSE = True


def patch_pyscf_grid_response():
    """PySCF 2.15's grids_response_becke (DFT gradients with grid response) looks up the Bragg
    radii of the Becke-Treutler size adjustment with mol.atom_charges(), which are Z - ncore
    for ECP atoms, while the energy grid (radi.treutler_atomic_radii_adjust) uses the proton
    numbers: its gradient is then not the derivative of its energy (e.g. 5e-5 Eh/bohr for CsF
    B3LYP/def2-TZVP). Call the original with the proton numbers in mol._atm."""
    import copy

    from pyscf.grad import rks as rks_grad

    orig = rks_grad.grids_response_becke
    if getattr(orig, "_linus_patched", False):
        return

    def fixed(grids):
        m = grids.mol.copy()
        m._atm = m._atm.copy()
        m._atm[:, 0] = [gto.charge(s) for s in m.elements]
        g2 = copy.copy(grids)
        g2.mol = m
        return orig(g2)

    fixed._linus_patched = True
    rks_grad.grids_response_becke = fixed


def run_linus(binary, atoms, charge, mult, method, basis, grad, threads):
    req = {"structure": {"numbers": [gto.charge(s) for s, _ in atoms], "geometry": [list(x) for _, x in atoms],
                         "charge": charge, "multiplicity": mult},
           "method": method, "basis": basis, "jk": {"kind": "exact", "screening": 1e-14},
           "scf": {"convergence": "extreme", "max_iter": 300}, "threads": threads,
           "calctype": "gradient" if grad else "energy"}
    if method != "hf":
        req["grid"] = {"xc": f"pyscf:{GRID_LEVEL}"}
        if grad:  # D41: the grid response is a gradient's setting
            req["grid"]["response"] = GRID_RESPONSE
    if mult != 1:
        req["reference"] = "unrestricted"
    return linus_json(req, binary)


def run_pyscf(atoms, charge, mult, method, basis, grad):
    """PySCF energy (and gradient); cached with LINUS_PYSCF_CACHE=<dir> (common.cached)."""
    mol = pyscf_mol(atoms, basis, False)
    mol.charge = charge
    mol.spin = mult - 1
    mol.build()
    from pyscf.grad import rks as rks_grad

    patched = bool(getattr(rks_grad.grids_response_becke, "_linus_patched", False))
    key = dict(mol=mol_key(mol), method=method, grad=grad, grid=GRID_LEVEL, grid_response=GRID_RESPONSE, fix_pyscf=patched)
    res, _ = cached(__file__, key, lambda: _pyscf(mol, mult, method, grad))
    return res


def _pyscf(mol, mult, method, grad):
    if method == "hf":
        mf = scf.UHF(mol) if mult != 1 else scf.RHF(mol)
    else:
        mf = dft.UKS(mol) if mult != 1 else dft.RKS(mol)
        mf.xc = PYSCF_XC[method]
        mf.grids.level = GRID_LEVEL
        mf.small_rho_cutoff = 0.0
    mf.conv_tol = 1e-12
    mf.conv_tol_grad = 1e-9
    mf.direct_scf_tol = 1e-14
    mf.max_cycle = 300
    t0 = time.perf_counter()
    e = mf.kernel()
    assert mf.converged
    res = {"energy": e, "e_nuc": mol.energy_nuc(), "nelec": mol.nelectron, "_wall": time.perf_counter() - t0}
    if grad:
        g = mf.nuc_grad_method()
        if method != "hf":
            g.grid_response = GRID_RESPONSE
        res["gradient"] = np.asarray(g.kernel())
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--linus", default=str(REPO / "target" / "release" / "linus"))
    ap.add_argument("--cases", default=",".join(DEFAULT))
    ap.add_argument("--grad", action="store_true")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--no-grid-response", action="store_true")
    ap.add_argument("--fix-pyscf", action="store_true", help="patch PySCF's grid-response radii for ECP atoms")
    args = ap.parse_args()
    global GRID_RESPONSE
    GRID_RESPONSE = not args.no_grid_response
    if args.fix_pyscf:
        patch_pyscf_grid_response()
    worst_e = worst_g = 0.0
    for case in args.cases.split(","):
        name, method, basis = case.split(".", 2)
        charge, mult = SPECS[name]
        atoms = CASES[name.rstrip("+")]
        lin = run_linus(args.linus, atoms, charge, mult, method, basis, args.grad, args.threads)
        ref = run_pyscf(atoms, charge, mult, method, basis, args.grad)
        de = lin["energy"] - ref["energy"]
        worst_e = max(worst_e, abs(de))
        row = f"{case:28s} E {lin['energy']:.10f}  dE {de:+.2e}  Enuc {lin['components']['nuclear_repulsion'] - ref['e_nuc']:+.1e}  nel {lin['nalpha'] + lin['nbeta']}/{ref['nelec']}"
        if args.grad:
            g = np.array(lin["gradient"])
            dg = np.abs(g - ref["gradient"]).max()
            worst_g = max(worst_g, dg)
            row += f"  dG {dg:.2e} (max {np.abs(ref['gradient']).max():.1e}, sum {np.abs(g.sum(0)).max():.1e})"
        row += f"  t {lin['_wall']:.1f}/{ref['_wall']:.1f} s"
        print(row, flush=True)
    print(f"worst dE {worst_e:.2e}" + (f"  worst dG {worst_g:.2e}" if args.grad else ""))


if __name__ == "__main__":
    main()
