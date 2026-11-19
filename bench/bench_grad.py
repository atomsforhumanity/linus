"""Time linus-ints RHF gradient contractions against PySCF's analytic RHF gradient.

    python bench/bench_grad.py <molecule.xyz> <basis> <threads> [workdir]

Runs PySCF RHF (identical basis definitions from data/basis), times
mf.nuc_grad_method().kernel() with `threads` OpenMP threads, then runs
target/release/examples/grad_bench with the same occupied orbitals and RAYON_NUM_THREADS =
threads, and prints both timings and the max gradient difference as JSON.
"""
import json, os, pathlib, subprocess, sys, tempfile, time

threads = int(sys.argv[3])
os.environ["OMP_NUM_THREADS"] = str(threads)
import numpy as np
from pyscf import gto, scf, lib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tests" / "python"))
from gen_deriv_fixtures import ROOT, SYM, bse_shells, read_xyz, make_mol  # noqa: E402


def main():
    xyz, basis = sys.argv[1], sys.argv[2]
    lib.num_threads(threads)
    zs, coords = read_xyz(pathlib.Path(xyz).resolve())
    mol = make_mol(zs, coords, {SYM[z]: bse_shells(basis, z) for z in set(zs)})
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-10
    t0 = time.perf_counter()
    e = mf.kernel()
    t_scf = time.perf_counter() - t0
    assert mf.converged
    gm = mf.nuc_grad_method()
    t0 = time.perf_counter()
    g_ref = gm.kernel()
    t_pyscf = time.perf_counter() - t0
    occ = mf.mo_occ > 0
    inp = {"z": zs, "coords": coords, "basis": basis, "nocc": int(occ.sum()),
           "c_occ": mf.mo_coeff[:, occ].ravel().tolist(), "e_occ": mf.mo_energy[occ].tolist(),
           "thresh": 1e-12, "repeat": 2}
    wd = pathlib.Path(sys.argv[4]) if len(sys.argv) > 4 else pathlib.Path(tempfile.mkdtemp())
    wd.mkdir(parents=True, exist_ok=True)
    (wd / "in.json").write_text(json.dumps(inp))
    env = dict(os.environ, RAYON_NUM_THREADS=str(threads))
    out = json.loads(subprocess.check_output([str(ROOT / "target/release/examples/grad_bench"), str(wd / "in.json")], env=env))
    g = np.array(out["grad"]).reshape(-1, 3)
    print(json.dumps({
        "molecule": pathlib.Path(xyz).name, "natom": mol.natm, "basis": basis, "nbf": mol.nao, "threads": threads,
        "energy": e, "t_scf_pyscf": round(t_scf, 3), "t_grad_pyscf": round(t_pyscf, 3),
        "t_grad_linus": round(out["t_total"], 3), "t_linus_setup": round(out["t_setup"], 3),
        "t_linus_1e": round(out["t_1e"], 3), "t_linus_2e": round(out["t_2e"], 3),
        "max_abs_diff": float(np.abs(g - g_ref).max()), "max_abs_grad": float(np.abs(g_ref).max()),
    }))


if __name__ == "__main__":
    main()
