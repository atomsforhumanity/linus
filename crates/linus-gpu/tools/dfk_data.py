"""Real DF-K inputs for the GPU GEMM-emulation measurement (crates/linus-gpu/examples/gemm_emulation.rs).

Runs a density-fitted RHF with PySCF and writes, to OUTDIR:
  meta.json   nbf, naux, nocc, the PySCF exchange energy E_x = -1/4 tr(D K[D]) at the converged
              density (D = 2 C C^T), and the SCF energy
  cderi.f64   PySCF's Cholesky-fitted 3-index tensor L[Q, mn], lower triangle packed (n <= m),
              naux x nbf(nbf+1)/2, row-major float64
  c_occ.f64   occupied MO coefficients, column-major nbf x nocc (float64)

    OMP_NUM_THREADS=8 python crates/linus-gpu/tools/dfk_data.py bench/molecules/u1an39.xyz OUTDIR [--basis def2-tzvp]

The charge is read from the xyz comment line (charge=N). Memory: about 1.2 x the packed tensor.
"""

import argparse
import json
import os
import re
import time

import numpy as np
from pyscf import gto, lib, scf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("xyz")
    ap.add_argument("outdir")
    ap.add_argument("--basis", default="def2-tzvp")
    ap.add_argument("--aux", default="def2-universal-jkfit")
    ap.add_argument("--memory-mb", type=int, default=40000)
    a = ap.parse_args()
    lines = open(a.xyz).read().splitlines()
    m = re.search(r"charge[= ](-?\d+)", lines[1])
    charge = int(m.group(1)) if m else 0
    mol = gto.M(atom="\n".join(lines[2 : 2 + int(lines[0])]), basis=a.basis, charge=charge, unit="Angstrom", cart=False, verbose=3)
    mol.max_memory = a.memory_mb
    mf = scf.RHF(mol).density_fit(auxbasis=a.aux)
    mf.max_memory = a.memory_mb
    mf.conv_tol = 1e-9
    t0 = time.time()
    e = mf.kernel()
    t_scf = time.time() - t0
    nocc = mol.nelectron // 2
    c = mf.mo_coeff[:, :nocc]
    dm = 2.0 * c @ c.T
    vk = mf.get_k(mol, dm)
    ex = -0.25 * float(np.einsum("ij,ji->", dm, vk))
    cderi = mf.with_df._cderi
    if isinstance(cderi, str):
        raise SystemExit("the 3-index tensor went to disk; raise --memory-mb")
    os.makedirs(a.outdir, exist_ok=True)
    np.ascontiguousarray(cderi, dtype=np.float64).tofile(os.path.join(a.outdir, "cderi.f64"))
    np.ascontiguousarray(c.T, dtype=np.float64).tofile(os.path.join(a.outdir, "c_occ.f64"))
    meta = {
        "molecule": os.path.basename(a.xyz),
        "basis": a.basis,
        "aux": a.aux,
        "nbf": int(mol.nao),
        "naux": int(cderi.shape[0]),
        "nocc": int(nocc),
        "e_scf": float(e),
        "e_x_pyscf": ex,
        "converged": bool(mf.converged),
        "scf_seconds": t_scf,
        "pyscf": __import__("pyscf").__version__,
        "threads": lib.num_threads(),
    }
    json.dump(meta, open(os.path.join(a.outdir, "meta.json"), "w"), indent=1)
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
