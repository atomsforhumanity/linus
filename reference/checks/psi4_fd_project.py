"""Psi4 1.11 finite-difference Hessians: fd_project true (default) vs false at a
non-stationary geometry, against PySCF's analytic HF Hessian from the runner.
Run from any directory: .refenv/bin/python reference/checks/psi4_fd_project.py
"""
import json
from pathlib import Path
import numpy as np
from qccompute import compute
from qcdata import CalcType, Model, ProgramInput, Structure

ROOT = Path(__file__).resolve().parents[2]
ref = json.loads((ROOT / "reference/out/water/hf_def2-svp_hessian_exact/pyscf.json").read_text())
s = Structure(symbols=ref["molecule"]["symbols"], geometry=ref["molecule"]["geometry_bohr"])
for proj in (True, False):
    inp = ProgramInput(program="psi4", calctype=CalcType.hessian, structure=s,
                       model=Model(method="hf", basis="def2-svp"),
                       keywords={"scf_type": "pk", "e_convergence": 1e-12, "d_convergence": 1e-10,
                                 "puream": True, "symmetry": "c1", "dertype": 1,  # FD of gradients
                                 "points": 5, "fd_project": proj},
                       cmdline_args=["-n", "8"])
    out = compute(inp, executables={"psi4": str(ROOT / ".refenv/bin/psi4")})
    H = np.asarray(out.results.hessian)
    print(f"fd_project={proj!s:5s}: max|H_psi4FD - H_pyscf_analytic| = {np.abs(H - np.asarray(ref['hessian'])).max():.1e}")
