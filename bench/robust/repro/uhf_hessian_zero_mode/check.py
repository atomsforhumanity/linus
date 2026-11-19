#!/usr/bin/env python
"""Analytic UHF Hessians at a zero orbital-Hessian mode: linus's and PySCF's against differences.

    .refenv/bin/python bench/robust/repro/uhf_hessian_zero_mode/check.py gmtkn55.BH76.RKT14 [METHOD]

The molecule from the sweep's sets (bench/robust/repro/sets.py), def2-SVP, exact J/K, tight SCF
(as the sweep's `hessian` suite). Prints:
- the lowest eigenvalues of the UHF orbital Hessian at PySCF's solution (the full matrix, built
  from PySCF's Hessian-vector product; Davidson from one start vector stays in its symmetry
  block and misses the mode);
- max |H - H_FD| for linus's and PySCF's analytic Hessians, H_FD from central differences of
  linus's analytic gradients (h = 1e-3 bohr; 2.5e-4 gives the same);
- along the direction where they differ most: the second derivative of the energy from
  differences of energies, against v.H.v of each Hessian.
LINUS_BIN (default target/release/linus), NTHREADS (default 4).
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "bench" / "robust" / "repro"))
import sets  # noqa: E402
from pyscf import gto, lib, scf  # noqa: E402
from pyscf.scf import stability  # noqa: E402

LINUS = os.environ.get("LINUS_BIN", str(REPO / "target" / "release" / "linus"))
NT = int(os.environ.get("NTHREADS", "4"))
lib.num_threads(NT)
B2A = 0.529177210544
mid = sys.argv[1]
method = sys.argv[2] if len(sys.argv) > 2 else "hf"
rec = next(r for r in sets.load() if r["id"] == mid)
x0 = np.array(rec["geometry_angstrom"]) / B2A
n = len(x0)


def linus(x, calctype):
    req = {"structure": {"symbols": rec["symbols"], "geometry": x.tolist(), "charge": rec["charge"],
                         "multiplicity": rec["multiplicity"]},
           "method": method, "basis": "def2-svp", "calctype": calctype,
           "threads": NT, "jk": {"kind": "exact", "screening": 1e-14}, "scf": {"convergence": "extreme", "max_iter": 300},
           **({} if method == "hf" else {"grid": {"xc": "pyscf:3"}})}
    return json.loads(subprocess.run([LINUS, "--json"], input=json.dumps(req), capture_output=True, text=True).stdout)


mol = gto.M(atom=list(zip(rec["symbols"], x0.tolist())), unit="Bohr", basis="def2-svp", charge=rec["charge"],
            spin=rec["multiplicity"] - 1, verbose=0)
if method == "hf":
    mf = scf.UHF(mol)
else:
    from pyscf import dft

    mf = dft.UKS(mol)
    mf.xc, mf.grids.level, mf.small_rho_cutoff = method, 3, 0.0
mf.conv_tol, mf.conv_tol_grad, mf.direct_scf_tol, mf.max_cycle = 1e-12, 1e-9, 1e-14, 300
mf.kernel()
g, hop, _ = stability.newton_ah.gen_g_hop_uhf(mf, mf.mo_coeff, mf.mo_occ)
A = np.array([hop(np.eye(g.size)[k]) for k in range(g.size)])
print("lowest UHF orbital-Hessian eigenvalues", np.round(np.linalg.eigvalsh(0.5 * (A + A.T))[:4], 6).tolist())
hs = mf.Hessian()
hs.conv_tol = 1e-10
H_py = hs.kernel().transpose(0, 2, 1, 3).reshape(3 * n, 3 * n)

h0 = linus(x0, "hessian")
H_an = np.asarray(h0["hessian"]).reshape(3 * n, 3 * n)
print("E linus - E pyscf", h0["energy"] - mf.e_tot)
h = 1e-3
H_fd = np.zeros((3 * n, 3 * n))
for i in range(3 * n):
    xp, xm = x0.copy(), x0.copy()
    xp.flat[i] += h
    xm.flat[i] -= h
    H_fd[i] = (np.ravel(linus(xp, "gradient")["gradient"]) - np.ravel(linus(xm, "gradient")["gradient"])) / (2 * h)
H_fd = 0.5 * (H_fd + H_fd.T)
print("max |H_linus - H_FD|", np.abs(H_an - H_fd).max())
print("max |H_pyscf - H_FD|", np.abs(H_py - H_fd).max())
w, U = np.linalg.eigh(H_an - H_fd)
v = U[:, np.argmax(np.abs(w))]
s = 1e-3
e = {k: linus(x0 + k * s * v.reshape(n, 3), "energy")["energy"] for k in (-2, -1, 0, 1, 2)}
d2 = (-e[2] + 16 * e[1] - 30 * e[0] + 16 * e[-1] - e[-2]) / (12 * s**2)
print(f"d2E/dv2 from energies {d2:.6f}; v.H.v linus {v @ H_an @ v:.6f}, PySCF {v @ H_py @ v:.6f}, "
      f"differences of gradients {v @ H_fd @ v:.6f}")
