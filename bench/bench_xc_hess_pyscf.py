"""Time PySCF's RKS-Hessian XC pieces next to linus's.
python bench/bench_xc_hess_pyscf.py <xyz> <basis> <functional> [threads]
PySCF: _get_vxc_deriv1 (V_xc^x, in make_h1) and _get_vxc_diag + _get_vxc_deriv2 (E_xc^xy, in
partial_hess_elec), at the converged density on the level-3 grid; OMP threads = threads.
Run under `slice bench` for quiet timings."""
import json, os, subprocess, sys, pathlib, time
import numpy as np
from pyscf import dft, lib
from pyscf.hessian import rks as rks_hess
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tests" / "python"))
from common import EXAMPLES, pyscf_mol  # noqa: E402
xyz, basis, xc = sys.argv[1:4]
threads = int(sys.argv[4]) if len(sys.argv) > 4 else 8
lib.num_threads(threads)
mol = pyscf_mol(xyz, basis, max_memory=8000)
mf = dft.RKS(mol); mf.xc = xc; mf.grids.level = 3; mf.conv_tol = 1e-10
mf.kernel()
hobj = mf.Hessian()
mo, occ = mf.mo_coeff, mf.mo_occ
mem = 8000  # MB; run under a capped scope (systemd-run --user --scope -p MemoryMax=...)
def best(f, reps=2):
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); f(); ts.append(time.perf_counter() - t0)
    return min(ts)
tv = best(lambda: rks_hess._get_vxc_deriv1(hobj, mo, occ, mem))
td = best(lambda: rks_hess._get_vxc_diag(hobj, mo, occ, mem))
t2 = best(lambda: rks_hess._get_vxc_deriv2(hobj, mo, occ, mem))
# one CPKS iteration's XC kernel contraction for 3N first-order densities
nao, nmo = mo.shape
nocc = int((occ > 0).sum())
co, cv = mo[:, :nocc], mo[:, nocc:]
ni = mf._numint
dm0 = mf.make_rdm1()
rho0, vxc0, fxc0 = ni.cache_xc_kernel(mol, mf.grids, mf.xc, mo, occ, 0)
a_, i_ = np.meshgrid(np.arange(nmo - nocc), np.arange(nocc), indexing="ij")
dm1 = []
for p in range(3 * mol.natm):
    u = (((a_ * 7 + i_ * 13 + p * 5) % 17) - 8.0) * 1e-3
    x = cv @ u @ co.T
    dm1.append(x + x.T)
dm1 = np.array(dm1)
tk = best(lambda: ni.nr_rks_fxc(mol, mf.grids, mf.xc, dm0, dm1, 0, 1, rho0, vxc0, fxc0))
out = json.loads(subprocess.check_output([str(EXAMPLES / "xc_hess_bench"), xyz, basis, xc, "--threads", str(threads)]))
print(f"{pathlib.Path(xyz).name} {basis} {xc} nbf {mol.nao} grid {mf.grids.weights.size} pts, {threads} threads")
print(f"  V_xc^x + E_xc^xy: PySCF _get_vxc_deriv1+_diag+_deriv2 {tv:.2f}+{td:.2f}+{t2:.2f} s   "
      f"linus hessian_terms (one pass, V^x C_o) {out['terms']:7.2f} s   ({(tv + td + t2) / out['terms']:.1f}x)")
print(f"  kernel 3N: PySCF nr_rks_fxc {tk:7.2f} s   linus kernel_contract_occ (V1 C_o) {out['kernel_3n']:7.2f} s   "
      f"({tk / out['kernel_3n']:.1f}x)")
print(f"  linus first-order grid terms (no PySCF counterpart): {out['terms_grid'] - out['terms']:.2f} s")
