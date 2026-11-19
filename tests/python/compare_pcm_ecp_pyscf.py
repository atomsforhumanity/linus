"""PCM with effective core potentials: linus against PySCF 2.15 (C-PCM etc. on molecules with def2
ECPs; the solute's nuclear potential on the surface uses Z - n_core).

  .refenv/bin/python tests/python/compare_pcm_ecp_pyscf.py [--cases CH3I,HI,AgCl] [--method hf]
      [--model cpcm] [--eps 7.4257] [--threads 8] [--linus target/release/linus]

Molecules and basis data as tests/python/ecp_pyscf_check.py (bohr, def2-SVP with its ECPs).
PySCF results are cached with LINUS_PYSCF_CACHE=<dir> (common.cached, pyscf_ref_cache.py).
Prints one JSON line per case (energy, solvation energy and gradient differences).
"""
import argparse, json, pathlib, sys

import numpy as np
from pyscf import dft, lib, scf
from pyscf.solvent import pcm  # noqa: F401

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from common import LINUS_BIN, cached, linus_json, mol_key  # noqa: E402
from ecp_pyscf_check import CASES, pyscf_mol  # noqa: E402

MODELS = {"cpcm": "C-PCM", "cosmo": "COSMO", "iefpcm": "IEF-PCM", "ssvpe": "SS(V)PE"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="CH3I,HI,AgCl")
    ap.add_argument("--basis", default="def2-svp")
    ap.add_argument("--method", default="hf")
    ap.add_argument("--model", default="cpcm", choices=list(MODELS))
    ap.add_argument("--eps", type=float, default=7.4257)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--linus", default=str(LINUS_BIN))
    a = ap.parse_args()
    lib.num_threads(a.threads)
    for name in a.cases.split(","):
        atoms = CASES[name]
        mol = pyscf_mol(atoms, a.basis, cart=False)

        def pyscf_ref():
            if a.method == "hf":
                mf = scf.RHF(mol) if mol.spin == 0 else scf.UHF(mol)
            else:
                mf = dft.RKS(mol) if mol.spin == 0 else dft.UKS(mol)
                mf.xc = a.method
                mf.grids.level = 3
                mf.small_rho_cutoff = 0.0
            mf.conv_tol = 1e-12
            mf.conv_tol_grad = 1e-9
            mf = mf.PCM()
            mf.with_solvent.method = MODELS[a.model]
            mf.with_solvent.eps = a.eps
            e = mf.kernel()
            return {"e": e, "e_solv": float(mf.with_solvent.e), "grad": np.asarray(mf.nuc_grad_method().kernel())}

        # LINUS_PYSCF_CACHE=<dir>: reuse a stored PySCF reference (pyscf_ref_cache.py)
        ref, hit = cached(__file__, dict(mol=mol_key(mol), method=a.method, grid=3, model=a.model, eps=a.eps), pyscf_ref)
        e, g = ref["e"], ref["grad"]
        req = {"structure": {"symbols": [s for s, _ in atoms], "geometry": [list(x) for _, x in atoms],
                             "multiplicity": mol.spin + 1},
               "method": a.method, "basis": a.basis, "threads": a.threads, "scf": {"convergence": "extreme"}, "calctype": "gradient",
               "jk": "exact",  # PySCF's exact J/K (linus's default is df since numerics epoch 6)
               "solvent": {"eps": a.eps, "model": a.model, "preset": "pyscf"}}
        if a.method != "hf":
            req["grid"] = {"xc": "pyscf:3"}
        out = linus_json(req, a.linus)
        print(json.dumps({"case": name, "basis": a.basis, "method": a.method, "model": a.model, "eps": a.eps,
                          "ecp_core": [int(mol.atom_nelec_core(i)) for i in range(mol.natm)],
                          "e_pyscf": e, "dE": out["energy"] - e, "e_solv_pyscf": ref["e_solv"],
                          "dE_solv": out["components"]["solvation_electrostatic"] - ref["e_solv"],
                          "max_abs_dG": float(np.abs(np.array(out["gradient"]) - g).max()), "max_abs_G": float(np.abs(g).max()),
                          "pyscf_cached": hit}), flush=True)


if __name__ == "__main__":
    main()
