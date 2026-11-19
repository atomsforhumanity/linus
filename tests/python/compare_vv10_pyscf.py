"""VV10 functionals (wB97M-V, wB97X-V, B97M-V): linus energies and analytic gradients against
PySCF 2.15.

  .refenv/bin/python tests/python/compare_vv10_pyscf.py <xyz> <basis> <method> \
      [--mult M] [--charge Q] [--df [--aux A]] [--energy-only] [--no-grid-response] \
      [--grid L] [--nlc-level L] [--threads N]

Same basis definitions (data/basis via compare_ints_pyscf.pyscf_basis); PySCF grid level L
(default 3) with small_rho_cutoff = 0 (linus GridChoice::Pyscf{level: L}); PySCF's nlcgrids
level (default 1 here, set in both programs; PySCF 2.15's own default is 3; linus nlc_grid pyscf:L). Grid response in both unless
--no-grid-response (then PySCF's default gradient, and linus grid_response = false).

--df: J and K fitted with one auxiliary basis; for range-separated functionals PySCF's DF get_jk
is replaced by the wK scheme linus uses (compare_rsh_pyscf.wk_get_jk). Gradients with --df are
compared only for functionals without exact exchange (B97M-V), where PySCF's DF gradient is
plain RI-J.

Also reported: the VV10 energy of the converged densities (PySCF nr_nlc_vxc; linus `vv10`).
PySCF results are cached with LINUS_PYSCF_CACHE=<dir> (common.cached, pyscf_ref_cache.py).
Prints one JSON line.
"""
import argparse, json, pathlib, subprocess, sys, time

import numpy as np
from pyscf import dft, lib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from common import EXAMPLES, cached, elements, mol_key, pyscf_basis, pyscf_mol, wk_get_jk  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("xyz")
    ap.add_argument("basis")
    ap.add_argument("method")
    ap.add_argument("--mult", type=int, default=1)
    ap.add_argument("--charge", type=int, default=0)
    ap.add_argument("--df", action="store_true")
    ap.add_argument("--aux", default="def2-universal-jkfit")
    ap.add_argument("--energy-only", action="store_true")
    ap.add_argument("--no-grid-response", action="store_true")
    ap.add_argument("--grid", type=int, default=3)
    ap.add_argument("--nlc-level", type=int, default=1)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--pyscf-gconv", type=float, default=1e-10, help="PySCF conv_tol_grad")
    ap.add_argument("--linus-gconv", type=float, default=1e-9, help="linus g_conv (max |F_ai|)")
    ap.add_argument("--linus-bin", default=str(EXAMPLES / "grad"))
    a = ap.parse_args()
    lib.num_threads(a.threads)
    mol = pyscf_mol(a.xyz, a.basis, a.charge, a.mult, max_memory=16000)
    hybrid = dft.libxc.is_hybrid_xc(a.method)
    auxbasis = pyscf_basis(a.aux, elements(mol)) if a.df else None
    want_grad = not a.energy_only and (not a.df or not hybrid)

    def pyscf_ref():
        mf = dft.RKS(mol) if a.mult == 1 else dft.UKS(mol)
        mf.xc = a.method
        mf.grids.level = a.grid
        mf.nlcgrids.level = a.nlc_level
        mf.small_rho_cutoff = 0.0
        mf.conv_tol = 1e-12
        mf.conv_tol_grad = a.pyscf_gconv
        mf.direct_scf_tol = 1e-14
        mf.max_cycle = 200
        if a.df:
            mf = mf.density_fit(auxbasis=auxbasis)
            if hybrid:
                mf = wk_get_jk(mf, auxbasis)
        t0 = time.perf_counter()
        e = mf.kernel()
        assert mf.converged
        dm = mf.make_rdm1()
        dmt = dm if a.mult == 1 else dm[0] + dm[1]
        r = {"e": e, "enlc": mf._numint.nr_nlc_vxc(mol, mf.nlcgrids, a.method, dmt)[1], "nlc_points": int(mf.nlcgrids.weights.size)}
        if want_grad:
            gm = mf.nuc_grad_method()
            gm.grid_response = not a.no_grid_response
            r["grad"] = np.asarray(gm.kernel())
        r["time"] = time.perf_counter() - t0
        return r

    # LINUS_PYSCF_CACHE=<dir>: reuse a stored PySCF reference (pyscf_ref_cache.py)
    ref, hit = cached(__file__, dict(mol=mol_key(mol), xc=a.method, grid=a.grid, nlc_level=a.nlc_level, gconv=a.pyscf_gconv, df=a.df,
                                     aux=auxbasis, grad=want_grad, grid_response=not a.no_grid_response), pyscf_ref)
    e, enlc = ref["e"], ref["enlc"]
    res = {"xyz": pathlib.Path(a.xyz).name, "basis": a.basis, "method": a.method, "mult": a.mult, "charge": a.charge,
           "jk": "df" if a.df else "exact", "grid": a.grid, "nlc_level": a.nlc_level, "grid_response": not a.no_grid_response,
           "e_pyscf": e, "e_vv10_pyscf": enlc, "nlc_points": ref["nlc_points"]}
    grad = ref.get("grad")
    res["pyscf_time"] = ref["time"]
    res["pyscf_cached"] = hit
    cmd = [a.linus_bin, a.xyz, a.basis, a.method, "--mult", str(a.mult), "--charge", str(a.charge), "--threads", str(a.threads),
           "--grid", str(a.grid), "--nlc-grid", f"pyscf:{a.nlc_level}", "--g-conv", str(a.linus_gconv)]
    if a.no_grid_response:
        cmd += ["--no-grid-response"]
    # the J/K explicitly: linus's default is df (numerics epoch 6)
    cmd += ["--jk", "df", "--aux", a.aux] if a.df else ["--jk", "exact"]
    out = json.loads(subprocess.check_output(cmd))
    e_l = out["energy"] - out["e_disp"]
    res.update({"e_linus": e_l, "dE": e_l - e, "e_vv10_linus": out.get("e_vv10"), "linus_time": out["time"]})
    if out.get("e_vv10") is not None:
        res["dE_vv10"] = out["e_vv10"] - enlc
    if grad is not None:
        g_l = np.array(out["gradient"]) - np.array(out["parts"]["disp"])
        res.update({"max_abs_dG": float(np.abs(g_l - grad).max()), "max_abs_G": float(np.abs(grad).max())})
    print(json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
