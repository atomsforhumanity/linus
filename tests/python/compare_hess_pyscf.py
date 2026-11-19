"""Compare linus analytic Hessians (closed and open shells) with PySCF.
python compare_hess_pyscf.py <xyz> <basis> [method] [threads] [df [aux]] [--fd-grad H] [--save PREFIX]
    [--charge Q] [--mult M] [--uhf] [--cpscf-tol T] [--tol-e E] [--tol-h H]
method: hf (default) or a functional (RKS on PySCF's level-3 grid; PySCF's RKS Hessian has no
grid-response terms, so linus runs with --fixed-grid, not its default first-order grid response). With df: PySCF's density-fitted Hessian
(pyscf.df.hessian, full auxiliary-basis response) with the same auxiliary basis definition
(default def2-universal-jkfit) and linus's jk=df. --fd-grad H adds linus's own check against
Richardson differences of its analytic gradient (HF). --save writes PREFIX_pyscf.npy and
PREFIX_linus.npy (3N x 3N, Eh/bohr^2). --charge/--mult set the charge and spin multiplicity;
open shells (M > 1, or --uhf for a closed shell) use UHF/UKS in both programs. --cpscf-tol sets
PySCF's conv_tol_cpscf (default 1e-11; its UHF Krylov solver often stalls above that and falls
back to a slow per-root solve, so open shells may use 1e-10).

VV10 functionals (wb97m-v, wb97x-v, b97m-v; RKS): PySCF's VV10 Fock derivative is patched to a
fixed grid (it hard-codes grid response there, but not in the rest of its VV10 Hessian).

Range-separated hybrids (PySCF's RKS/UKS Hessians support them; meta-GGAs too): wb97x,
cam-b3lyp, HYB_GGA_XC_LC_WPBE, ...; wb97x-d3bj -> PySCF wb97x-v and wb97m-d3bj -> wb97m-v, both
with nlc off, and linus with --disp none (as compare_rsh_pyscf.py). With df, PySCF's Hessian fits
the long-range exchange with the erf metric (P|erf|Q) while linus (and its SCF) use Psi4's wK
scheme (Coulomb metric): the DF line then compares two different RI approximations and is
informational (its dE line says so); linus's wK Hessian is checked against finite differences
(hess_df tests, --fd-grad). DFT runs set small_rho_cutoff = 0 (linus's PySCF grids are not
pruned by density). PySCF results are cached with LINUS_PYSCF_CACHE=<dir> (common.cached,
pyscf_ref_cache.py). Without df, exact J/K in both programs (linus's hess example with --jk exact). --tol-e/--tol-h:
exit 1, with FAIL on the line, when |dE| or max|dH| is above them (the merge gate's check; tests/python/README.md).
Merge-gate check: keep the output line's format."""
import json, os, subprocess, sys, pathlib, time
import numpy as np
from pyscf import scf, dft, lib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from common import EXAMPLES, cached, elements, mol_key, pyscf_basis, pyscf_mol, pyscf_xc, take_tolerance, verdict  # noqa: E402
args = list(sys.argv[1:])
tol_e, tol_h = take_tolerance(args, "--tol-e"), take_tolerance(args, "--tol-h")
fd_step = None
if "--fd-grad" in args:
    i = args.index("--fd-grad"); fd_step = args[i + 1]; del args[i:i + 2]
save = None
if "--save" in args:
    i = args.index("--save"); save = args[i + 1]; del args[i:i + 2]
charge, mult = 0, 1
if "--charge" in args:
    i = args.index("--charge"); charge = int(args[i + 1]); del args[i:i + 2]
if "--mult" in args:
    i = args.index("--mult"); mult = int(args[i + 1]); del args[i:i + 2]
cpscf_tol = 1e-11
if "--cpscf-tol" in args:
    i = args.index("--cpscf-tol"); cpscf_tol = float(args[i + 1]); del args[i:i + 2]
force_uhf = "--uhf" in args
if force_uhf:
    args.remove("--uhf")
unrestricted = force_uhf or mult > 1
xyz, basis = args[:2]
method = args[2] if len(args) > 2 else "hf"
# linus name -> (PySCF xc, nlc off): ORCA-style -D3BJ aliases of the VV10 functionals
pxc, no_nlc = pyscf_xc(method)
# VV10 functionals with their nonlocal correlation (wb97m-v, wb97x-v, b97m-v; RKS only: PySCF 2.15
# has no UKS Hessian with NLC). PySCF's VV10 Hessian takes the Fock-matrix derivative with
# grid response (hard-coded `grid_response = True` in hessian.rks._get_vnlc_deriv1) but the
# energy second derivative and the kernel on a fixed grid; the comparison patches the first
# to a fixed grid too, the convention of linus's --fixed-grid. NLC grid: PySCF's default
# nlcgrids (level 3), linus's nlc_grid auto for PySCF grids (pyscf:3).
vv10 = method != "hf" and not no_nlc and dft.libxc.is_nlc(pxc)


def pyscf_vv10_fixed_grid():
    import inspect
    from pyscf.hessian import rks as rks_hess
    src = inspect.getsource(rks_hess._get_vnlc_deriv1)
    assert "    grid_response = True\n" in src, "PySCF's _get_vnlc_deriv1 changed"
    exec(compile(src.replace("    grid_response = True\n", "    grid_response = False\n"), rks_hess.__file__, "exec"),
         rks_hess.__dict__)


threads = args[3] if len(args) > 3 else "16"
use_df = len(args) > 4 and args[4] == "df"
aux = args[5] if len(args) > 5 else "def2-universal-jkfit"
lib.num_threads(int(threads))
# MB; the node is shared: run under a capped scope (systemd-run --user --scope -p MemoryMax=...)
mol = pyscf_mol(xyz, basis, charge, mult, max_memory=8000)
auxbasis = pyscf_basis(aux, elements(mol)) if use_df else None


def make_mf():
    if method == "hf":
        mf = scf.UHF(mol) if unrestricted else scf.RHF(mol)
    else:
        mf = dft.UKS(mol) if unrestricted else dft.RKS(mol)
        mf.xc = pxc
        if no_nlc:
            mf.nlc = False
        mf.grids.level = 3
        mf.small_rho_cutoff = 0.0
    if use_df:
        mf = mf.density_fit(auxbasis=auxbasis)
    mf.conv_tol = 1e-12; mf.conv_tol_grad = 1e-9; mf.direct_scf_tol = 1e-14
    # PySCF's CPHF tolerance is conv_tol_cpscf (default 1e-8, times the atoms per block), not
    # hessobj.conv_tol; tighten it so the reference Hessian is converged to ~1e-9
    mf.conv_tol_cpscf = cpscf_tol
    return mf


def pyscf_ref():
    if vv10:
        pyscf_vv10_fixed_grid()
    mf = make_mf()
    mf.kernel()
    t0 = time.time()
    hp = mf.Hessian(); hp.conv_tol = 1e-10; hp.max_memory = 8000
    H = hp.kernel().transpose(0, 2, 1, 3).reshape(3 * mol.natm, 3 * mol.natm)
    return {"e_tot": mf.e_tot, "hessian": H, "time": time.time() - t0}


# LINUS_PYSCF_CACHE=<dir>: reuse a stored PySCF reference (pyscf_ref_cache.py)
key = dict(mol=mol_key(mol), method=method, pxc=pxc, no_nlc=no_nlc, df=use_df, aux=auxbasis, uhf=force_uhf,
           cpscf_tol=cpscf_tol)
if vv10:
    key["vv10_fixed_grid"] = True
ref, hit = cached(__file__, key, pyscf_ref)
e_ref, H, tp = ref["e_tot"], np.asarray(ref["hessian"]), ref["time"]
# LINUS_HESS_BIN: another hess binary (default: the hess example in common.EXAMPLES)
cmd = [os.environ.get("LINUS_HESS_BIN", str(EXAMPLES / "hess")), xyz, basis, "--method", method, "--threads", threads, "--fixed-grid"]
# the J/K explicitly: linus's default is df (numerics epoch 6), PySCF's reference exact unless df
cmd += (["--jk", "df", "--aux", aux] if use_df else ["--jk", "exact"]) + (["--fd-grad", fd_step] if fd_step else [])
cmd += ["--charge", str(charge), "--mult", str(mult)] + (["--uhf"] if force_uhf else [])
if no_nlc:
    cmd += ["--disp", "none"]
omega = 0.0
if method != "hf":
    omega = dft.libxc.rsh_coeff(pxc)[0]
df_note = " (PySCF DF: erf-metric LR fit, linus: wK; informational)" if use_df and omega != 0 else ""
out = json.loads(subprocess.check_output(cmd))
Hl = np.array(out["hessian"]).reshape(H.shape)
if save:
    np.save(f"{save}_pyscf.npy", H); np.save(f"{save}_linus.npy", Hl)
extra = f"; FD(grad) max|dH| {out['fd_grad_max_err']:.2e} (vs symmetrized FD; FD asymmetry {out.get('fd_grad_asymmetry', 0.0):.1e})" if fd_step else ""
tag = ("U" if unrestricted else "R") + ("HF" if method == "hf" else "KS")
de, dh = out["energy"] - e_ref, np.abs(Hl - H).max()
suffix, status = verdict({"|dE|": (de, tol_e), "max|dH|": (dh, tol_h)})
print(f"{pathlib.Path(xyz).name:14} {basis:10} {method:7}{' DF' if use_df else '   '} {tag} Hessian: dE {de:+.1e} max|dH| {dh:.2e} (|H|max {np.abs(H).max():.2e}); "
      f"CPHF iters {out['cphf_iterations']}; linus {out['time']:.2f}s (incl. SCF+grad), PySCF hessian {tp:.2f}s{" (cached)" if hit else ""}{extra}{df_note}{suffix}")
sys.exit(status)
