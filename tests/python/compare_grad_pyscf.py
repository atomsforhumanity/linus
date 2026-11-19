"""Compare linus analytic gradients with PySCF (same basis definitions, same grid).
python tests/python/compare_grad_pyscf.py <xyz> <basis> <method> [mult] [df] [--tol-e E] [--tol-g G]
DFT on PySCF's level-3 grid with grid response in both. Exact J/K in both programs (linus's grad example with
--jk exact), or with df density fitting in both (def2-universal-jkfit). PySCF results are cached with
LINUS_PYSCF_CACHE=<dir> (common.cached, pyscf_ref_cache.py). --tol-e/--tol-g: exit 1, with FAIL on the line,
when |dE| or max|dG| is above them (the merge gate's check; tests/python/README.md). Merge-gate check: keep the
output line's format."""
import json, subprocess, sys, pathlib
import numpy as np
from pyscf import scf, dft
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from common import EXAMPLES, cached, elements, mol_key, pyscf_basis, pyscf_mol, take_tolerance, verdict  # noqa: E402
args = sys.argv[1:]
tol_e, tol_g = take_tolerance(args, "--tol-e"), take_tolerance(args, "--tol-g")
xyz, basis, method = args[:3]
mult = int(args[3]) if len(args) > 3 else 1
use_df = len(args) > 4 and args[4] == "df"
mol = pyscf_mol(xyz, basis, mult=mult)
aux = pyscf_basis("def2-universal-jkfit", elements(mol)) if use_df else None


def pyscf_ref():
    if method == "hf":
        mf = scf.RHF(mol) if mult == 1 else scf.UHF(mol)
    else:
        mf = dft.RKS(mol) if mult == 1 else dft.UKS(mol)
        mf.xc = method
        mf.grids.level = 3
    if use_df:
        mf = mf.density_fit(auxbasis=aux)
    mf.conv_tol = 1e-12
    mf.direct_scf_tol = 1e-14
    mf.conv_tol_grad = 1e-9
    e = mf.kernel()
    gm = mf.nuc_grad_method()
    if method != "hf":
        gm.grid_response = True
    return {"e_tot": e, "gradient": gm.kernel()}


ref, _ = cached(__file__, dict(mol=mol_key(mol), method=method, mult=mult, df=use_df, aux=aux), pyscf_ref)
e, g = ref["e_tot"], np.asarray(ref["gradient"])
# the J/K explicitly: linus's default is df (numerics epoch 6), PySCF's reference exact unless df
cmd = [str(EXAMPLES / "grad"), xyz, basis, method, "--mult", str(mult), "--threads", "32", "--jk", "df" if use_df else "exact"]
out = json.loads(subprocess.check_output(cmd))
gl = np.array(out["gradient"])
de, dg = out["energy"] - e, np.abs(gl - g).max()
suffix, status = verdict({"|dE|": (de, tol_e), "max|dG|": (dg, tol_g)})
print(f"{pathlib.Path(xyz).name:12} {basis:10} {method:7} mult={mult} df={use_df}: dE {de:+.2e}  max|dG| {dg:.2e}  (|G|max {np.abs(g).max():.2e}, linus {out['time']:.2f}s){suffix}")
sys.exit(status)
