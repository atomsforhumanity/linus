"""Range-separated hybrids: linus energies and analytic gradients against PySCF 2.15.

  .refenv/bin/python tests/python/compare_rsh_pyscf.py <xyz> <basis> <linus-method> \
      [--mult M] [--charge Q] [--df [--aux A]] [--energy-only] [--threads N]

Same basis definitions (data/basis via compare_ints_pyscf.pyscf_basis), PySCF grid level 3
with small_rho_cutoff = 0 (linus GridChoice::Pyscf{level: 3}), grid response in both.

linus -> PySCF functional: wb97x -> wb97x; cam-b3lyp -> cam-b3lyp; HYB_GGA_XC_LC_WPBE ->
HYB_GGA_XC_LC_WPBE; wb97x-d3bj -> wb97x-v with nlc off (linus's dispersion energy and
gradient are removed before comparing); wb97m-d3bj -> wb97m-v with nlc off.

--df: PySCF's own DF fits K_LR with the erf metric (P|erf|Q); linus uses Psi4's wK scheme
(Coulomb metric, K_LR = sym[(ml|P) V^-1 (P|erf|ns)] D_ls). The reference therefore replaces
PySCF's DF get_jk for omega requests by wK built from PySCF integrals (int3c2e with and
without mol.with_range_coulomb, int2c2e). Gradients are compared only for exact integrals
(PySCF's DF gradient is the erf-metric scheme); the DF energy line also reports PySCF's
native erf-metric energy for information.
PySCF results are cached with LINUS_PYSCF_CACHE=<dir> (common.cached, pyscf_ref_cache.py).
Prints one JSON line.
"""
import argparse, json, pathlib, subprocess, sys

import numpy as np
from pyscf import dft, lib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from common import EXAMPLES, cached, elements, mol_key, pyscf_basis, pyscf_mol, pyscf_xc, wk_get_jk  # noqa: E402,F401  (wk_get_jk: imported from here by others)


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
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--pyscf-gconv", type=float, default=1e-10, help="PySCF conv_tol_grad")
    a = ap.parse_args()
    lib.num_threads(a.threads)
    mol = pyscf_mol(a.xyz, a.basis, a.charge, a.mult, max_memory=8000)
    pxc, no_nlc = pyscf_xc(a.method)
    auxbasis = pyscf_basis(a.aux, elements(mol)) if a.df else None

    def make(native_df=False):
        mf = dft.RKS(mol) if a.mult == 1 else dft.UKS(mol)
        mf.xc = pxc
        if no_nlc:
            mf.nlc = False
        mf.grids.level = 3
        mf.small_rho_cutoff = 0.0
        mf.conv_tol = 1e-12
        mf.conv_tol_grad = a.pyscf_gconv
        mf.direct_scf_tol = 1e-14
        mf.max_cycle = 200
        if a.df:
            mf = mf.density_fit(auxbasis=auxbasis)
            if not native_df:
                mf = wk_get_jk(mf, auxbasis)
        return mf

    def pyscf_ref():
        mf = make()
        r = {"e": mf.kernel()}
        assert mf.converged
        if a.df:
            r["e_native_df"] = make(native_df=True).kernel()
        if not a.energy_only and not a.df:
            gm = mf.nuc_grad_method()
            gm.grid_response = True
            r["grad"] = np.asarray(gm.kernel())
        return r

    # LINUS_PYSCF_CACHE=<dir>: reuse a stored PySCF reference (pyscf_ref_cache.py)
    ref, hit = cached(__file__, dict(mol=mol_key(mol), pxc=pxc, no_nlc=no_nlc, grid=3, gconv=a.pyscf_gconv, df=a.df, aux=auxbasis,
                                     energy_only=a.energy_only), pyscf_ref)
    e = ref["e"]
    res = {"xyz": pathlib.Path(a.xyz).name, "basis": a.basis, "method": a.method, "pyscf_xc": pxc, "mult": a.mult,
           "jk": "df-wK" if a.df else "exact", "e_pyscf": e}
    if a.df:
        res["e_pyscf_native_df_erf_metric"] = ref["e_native_df"]
    grad = ref.get("grad")
    cmd = [str(EXAMPLES / "grad"), a.xyz, a.basis, a.method, "--mult", str(a.mult), "--charge", str(a.charge), "--threads", str(a.threads)]
    # the J/K explicitly: linus's default is df (numerics epoch 6)
    cmd += ["--jk", "df", "--aux", a.aux] if a.df else ["--jk", "exact"]
    out = json.loads(subprocess.check_output(cmd))
    e_l = out["energy"] - out["e_disp"]
    g_l = np.array(out["gradient"]) - np.array(out["parts"]["disp"])
    res.update({"e_linus_no_disp": e_l, "dE": e_l - e, "e_disp": out["e_disp"], "linus_time": out["time"]})
    if a.df:
        res["dE_vs_native_df"] = e_l - res["e_pyscf_native_df_erf_metric"]
    if grad is not None:
        res.update({"max_abs_dG": float(np.abs(g_l - grad).max()), "max_abs_G": float(np.abs(grad).max())})
    res["pyscf_cached"] = hit
    print(json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
