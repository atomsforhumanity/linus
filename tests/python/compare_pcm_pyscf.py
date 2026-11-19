"""Implicit solvent (PCM): linus energies, analytic gradients and Hessians against PySCF 2.15's
`pyscf.solvent.pcm` (C-PCM, COSMO, IEF-PCM, SS(V)PE).

  .refenv/bin/python tests/python/compare_pcm_pyscf.py <xyz> <basis> <method> \
      [--model cpcm|cosmo|iefpcm|ssvpe] [--eps 7.4257] [--charge Q] [--mult M] [--df [--aux A]] \
      [--lebedev 302] [--discretization swig|iswig] [--grad] [--hess] [--threads N] [--linus BIN]

linus runs with the `pyscf` PCM preset (modified Bondi x 1.2, SWIG, 302 points unless
--lebedev); PySCF with the same eps, model, `lebedev_order` and `surface_discretization_method`.
Same basis definitions (data/basis via compare_ints_pyscf.pyscf_basis); DFT on PySCF grid level 3
with small_rho_cutoff = 0 (linus GridChoice::Pyscf{level: 3}); gradients with grid response in
both; Hessians on a fixed grid (linus grid.response false, PySCF's convention). wb97x-d3bj ->
PySCF wb97x-v with nlc off, and linus's D3(BJ) energy/gradient removed before comparing.
--df: density fitting in both (one aux basis for J and K; range-separated hybrids use the wK
scheme of compare_rsh_pyscf.py in PySCF).
Molecules go to PySCF in bohr (linus reads Å with CODATA 2022). PySCF results are cached with
LINUS_PYSCF_CACHE=<dir> (common.cached, pyscf_ref_cache.py). Prints one JSON line.
"""
import argparse, json, pathlib, sys, time

import numpy as np
from pyscf import dft, lib, scf
from pyscf.dft import gen_grid
from pyscf.solvent import pcm  # noqa: F401  (registers mf.PCM)
from pyscf.solvent import smd  # noqa: F401  (registers mf.SMD)

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from common import LINUS_BIN, cached, elements, linus_json, mol_key, pyscf_basis, pyscf_mol, pyscf_xc, read_xyz_bohr, wk_get_jk  # noqa: E402

MODELS = {"cpcm": "C-PCM", "cosmo": "COSMO", "iefpcm": "IEF-PCM", "ssvpe": "SS(V)PE", "smd": "SMD"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("xyz")
    ap.add_argument("basis")
    ap.add_argument("method")
    ap.add_argument("--model", default="cpcm", choices=list(MODELS))
    ap.add_argument("--eps", type=float, default=7.4257)
    ap.add_argument("--charge", type=int, default=0)
    ap.add_argument("--mult", type=int, default=1)
    ap.add_argument("--df", action="store_true")
    ap.add_argument("--aux", default="def2-universal-jkfit")
    ap.add_argument("--lebedev", type=int, default=None, help="points per sphere (default: PySCF's, 302; 590 for SMD)")
    ap.add_argument("--solvent", default=None, help="solvent name (SMD: required; PySCF's eps is then used)")
    ap.add_argument("--discretization", default="swig", choices=["swig", "iswig"])
    ap.add_argument("--grad", action="store_true")
    ap.add_argument("--hess", action="store_true")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--linus", default=str(LINUS_BIN))
    a = ap.parse_args()
    lib.num_threads(a.threads)
    mol = pyscf_mol(a.xyz, a.basis, a.charge, a.mult, max_memory=16000)
    hf = a.method.lower() == "hf"
    pxc, no_nlc = pyscf_xc(a.method)
    rsh = (not hf) and dft.libxc.rsh_coeff(pxc)[0] != 0
    auxbasis = pyscf_basis(a.aux, elements(mol)) if a.df else None
    if a.model == "smd":
        a.eps = float(smd.solvent_db[a.solvent][5])  # PySCF fills sol.eps from the descriptors in build()
        a.lebedev = a.lebedev or 590
    else:
        a.lebedev = a.lebedev or 302
    # wk_get_jk patches only get_jk: PySCF's DF gradient would still use its own (erf-metric)
    # long-range exchange, which is not the derivative of the energy compared here
    no_grad = (a.grad or a.hess) and a.df and rsh
    want_grad = (a.grad or a.hess) and not no_grad

    def pyscf_ref():
        if hf:
            mf = scf.RHF(mol) if a.mult == 1 else scf.UHF(mol)
        else:
            mf = dft.RKS(mol) if a.mult == 1 else dft.UKS(mol)
            mf.xc = pxc
            if no_nlc:
                mf.nlc = False
            mf.grids.level = 3
            mf.small_rho_cutoff = 0.0
        mf.conv_tol = 1e-12
        mf.conv_tol_grad = 1e-9
        mf.direct_scf_tol = 1e-14
        mf.max_cycle = 200
        if a.df:
            mf = mf.density_fit(auxbasis=auxbasis)
            if rsh:
                mf = wk_get_jk(mf, auxbasis)
        if a.model == "smd":
            mf = mf.SMD()
            sol = mf.with_solvent
            sol.solvent = a.solvent
        else:
            mf = mf.PCM()
            sol = mf.with_solvent
            sol.method = MODELS[a.model]
            sol.eps = a.eps
        sol.lebedev_order = {v: k for k, v in gen_grid.LEBEDEV_ORDER.items()}[a.lebedev]
        if a.model == "smd":
            sol.sasa_ng = a.lebedev
        sol.surface_discretization_method = a.discretization.upper()
        t0 = time.time()
        r = {"e": mf.kernel()}
        assert mf.converged
        r.update(e_solv=float(sol.e), npoints=int(sol.surface["grid_coords"].shape[0]))
        if a.model == "smd":
            r["e_cds"] = float(sol.e_cds)
        if want_grad:
            gm = mf.nuc_grad_method()
            if not hf:
                gm.grid_response = True  # linus's gradient keeps its grid response under --hess too
            r["grad"] = np.asarray(gm.kernel())
        if a.hess:
            hm = mf.Hessian()
            r["hess"] = np.asarray(hm.kernel()).transpose(0, 2, 1, 3).reshape(3 * mol.natm, 3 * mol.natm)
        r["time"] = time.time() - t0
        return r

    # LINUS_PYSCF_CACHE=<dir>: reuse a stored PySCF reference (pyscf_ref_cache.py)
    ref, hit = cached(__file__, dict(mol=mol_key(mol), method=a.method, pxc=pxc, no_nlc=no_nlc, grid=3, model=a.model, eps=a.eps,
                                     solvent=a.solvent if a.model == "smd" else None, lebedev=a.lebedev,
                                     discretization=a.discretization, df=a.df, aux=auxbasis, grad=want_grad, hess=a.hess), pyscf_ref)
    e = ref["e"]
    res = {"xyz": pathlib.Path(a.xyz).name, "basis": a.basis, "method": a.method, "model": a.model, "eps": a.eps,
           "charge": a.charge, "mult": a.mult, "jk": "df" if a.df else "exact", "lebedev": a.lebedev,
           "discretization": a.discretization, "e_pyscf": e, "e_solv_pyscf": ref["e_solv"],
           "npoints_pyscf": ref["npoints"]}
    if no_grad:
        res["gradient_note"] = "DF + range-separated hybrid: no PySCF gradient (its DF gradient is not of the wK energy)"
    grad, hess = ref.get("grad"), ref.get("hess")
    res["pyscf_time"] = ref["time"]
    res["pyscf_cached"] = hit
    syms, pos = read_xyz_bohr(a.xyz)
    req = {"structure": {"symbols": syms, "geometry": pos, "charge": a.charge, "multiplicity": a.mult},
           "method": a.method, "basis": a.basis, "threads": a.threads, "scf": {"convergence": "extreme"},
           "calctype": "hessian" if a.hess else ("gradient" if a.grad else "energy"),
           "solvent": {"eps": a.eps, "model": a.model, "preset": "pyscf", "lebedev": a.lebedev, "discretization": a.discretization}}
    if a.model == "smd":
        # PySCF's CDS term uses the exact SASA
        req["solvent"].update(name=a.solvent, cds_contact_smoothing=0.0)
    if a.df:
        req["jk"] = {"kind": "df", "aux": a.aux}
    if not hf:
        req["grid"] = {"xc": "pyscf:3"}
        if a.hess:
            # grid.response is one key for the gradient and the Hessian: the Hessian's fixed grid here, the
            # gradient (with its grid response) from a call of its own below
            req["grid"].update(response=False, kernel="pyscf:3")
    t0 = time.time()
    try:
        out = linus_json(req, a.linus)
        if a.hess and not hf and grad is not None:
            out["gradient"] = linus_json({**req, "calctype": "gradient", "grid": {"xc": "pyscf:3", "response": True}}, a.linus)["gradient"]
    except RuntimeError as exc:
        raise SystemExit(f"linus failed: {exc}")
    res["linus_time"] = time.time() - t0
    disp = out["components"].get("dispersion", 0.0)
    e_l = out["energy"] - disp
    res.update({"e_linus": e_l, "dE": e_l - e, "e_solv_linus": out["components"]["solvation_electrostatic"],
                "dE_solv": out["components"]["solvation_electrostatic"] - ref["e_solv"], "npoints_linus": out["solvation"]["npoints"],
                "surface_charge": out["solvation"]["surface_charge"], "gauss_charge": out["solvation"]["gauss_charge"]})
    if a.model == "smd":
        res.update({"solvent": a.solvent, "e_cds_pyscf": ref["e_cds"], "dE_cds": out["components"]["solvation_cds"] - ref["e_cds"]})
    if grad is not None:
        g_l = np.array(out["gradient"])
        if "gradient_parts" in out:
            g_l = g_l - np.array(out["gradient_parts"]["dispersion"])
        res.update({"max_abs_dG": float(np.abs(g_l - grad).max()), "max_abs_G": float(np.abs(grad).max())})
    if hess is not None:
        h_l = np.array(out["hessian"]).reshape(3 * mol.natm, 3 * mol.natm)
        res.update({"max_abs_dH": float(np.abs(h_l - hess).max()), "max_abs_H": float(np.abs(hess).max())})
    print(json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
