"""Open-shell (UKS) analytic Hessian benchmark: linus vs ORCA 6.0.0 vs PySCF, with harmonic
frequencies from every Hessian (same projection and masses) compared against a reference.

Closed shells (--mult 1) run RKS in ORCA and restricted in linus.
usage: .refenv/bin/python bench/uks_hess_bench.py <prog> <xyz> --out OUT.json [--charge Q] [--mult M]
           [--threads 8] [--slice bench|none|cpus:LIST] [--lock FILE] [--linus-bin PATH] [--grid orca:2]
           [--ref] [--coretype Haswell] [--fixed-grid]
prog: linus | orca | pyscf. Level: UKS B3LYP/def2-SVP; linus RI-JK (def2-universal-jkfit) on
ORCA's DefGrid2 construction (--grid), ORCA `UKS B3LYP/G def2-SVP def2/J RIJCOSX DefGrid2
TightSCF Freq`, PySCF DF UKS (def2-universal-jkfit, grid level 3). --ref: linus with exact J/K,
PySCF grid level 5 and the first-order grid terms (the accuracy reference). Each program runs
under the slice and, with --lock, inside `flock LOCK` (timed inside the lock). The record holds
the wall time, the Hessian (Eh/bohr^2), frequencies (cm^-1) and `slice list`.
"""
import argparse, json, os, pathlib, re, subprocess, tempfile, time
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
from bench_paths import ORCA, REFENV  # noqa: E402  (env overrides: bench/bench_paths.py)
# the reference environment lives in the main checkout (worktrees have none)
BOHR = 0.529177210544
# most abundant isotopes (amu), as ORCA and PySCF use
MASS = {"H": 1.00782503223, "C": 12.0, "N": 14.00307400443, "O": 15.99491461957, "S": 31.9720711744, "P": 30.97376199842}
AMU_ME = 1822.888486209
HARTREE_CM = 219474.6313705


def read_xyz(p):
    lines = pathlib.Path(p).read_text().splitlines()
    n = int(lines[0])
    return [(l.split()[0], [float(x) for x in l.split()[1:4]]) for l in lines[2:2 + n]]


def frequencies(h, atoms):
    """Harmonic frequencies (cm^-1; imaginary as negative) of the 3N-6 (3N-5 linear) internal
    modes: mass-weighted Hessian with translations and rotations projected out."""
    n = len(atoms)
    m = np.array([MASS[a] for a, _ in atoms]) * AMU_ME
    x = np.array([c for _, c in atoms]) / BOHR
    com = (m[:, None] * x).sum(0) / m.sum()
    x = x - com
    sm = np.sqrt(np.repeat(m, 3))
    hm = h / np.outer(sm, sm)
    vecs = []
    for k in range(3):
        v = np.zeros((n, 3)); v[:, k] = 1.0
        vecs.append((v * np.sqrt(m)[:, None]).ravel())
    for k in range(3):
        e = np.zeros(3); e[k] = 1.0
        v = np.cross(e, x)
        vecs.append((v * np.sqrt(m)[:, None]).ravel())
    q, s, _ = np.linalg.svd(np.array(vecs).T, full_matrices=False)
    q = q[:, s > 1e-6 * s.max()]
    p = np.eye(3 * n) - q @ q.T
    w = np.linalg.eigvalsh(p @ hm @ p)
    w = w[np.argsort(np.abs(w))][q.shape[1]:]
    f = np.sign(w) * np.sqrt(np.abs(w)) * HARTREE_CM
    return np.sort(f).tolist()


def prefix(args):
    pre = []
    if args.lock:
        pre += ["flock", args.lock]
    if args.slice == "none":
        return pre
    if args.slice.startswith("cpus:"):
        return pre + ["taskset", "-c", args.slice[5:]]
    return pre + [os.path.expanduser("~/.local/bin/slice"), args.slice]


def slice_list():
    return subprocess.run([os.path.expanduser("~/.local/bin/slice"), "list"], capture_output=True, text=True).stdout.split("\n")


def timed_sh(cmd, args, cwd, env=None):
    """Run `cmd` (list) under the slice/lock; the time is taken inside the lock by a wrapper."""
    inner = "s=$(date +%s.%N); \"$@\"; rc=$?; e=$(date +%s.%N); echo \"WALL $s $e\" >&2; exit $rc"
    full = prefix(args) + ["bash", "-c", inner, "x"] + cmd
    p = subprocess.run(full, cwd=cwd, env=env, capture_output=True, text=True)
    m = re.search(r"WALL (\S+) (\S+)", p.stderr)
    wall = float(m.group(2)) - float(m.group(1)) if m else None
    return wall, p


def run_linus(atoms, args, wd):
    xyz = wd / "mol.xyz"
    xyz.write_text(f"{len(atoms)}\n\n" + "\n".join(f"{a} {c[0]:.10f} {c[1]:.10f} {c[2]:.10f}" for a, c in atoms) + "\n")
    cmd = [args.linus_bin, str(xyz), "def2-svp", "--method", "b3lyp", "--threads", str(args.threads), "--charge", str(args.charge), "--mult", str(args.mult)]
    if args.ref:
        cmd += ["--grid", "5"]
    else:
        cmd += ["--jk", "df", "--set", f"jk.aux={args.aux}"]
        g = args.grid.split(":")
        cmd += ["--grid", args.grid if g[0] in ("linus", "auto", "orca") else g[-1]]
        if args.fixed_grid:
            cmd += ["--set", "grid.response=false"]
    env = dict(os.environ, **dict(kv.split("=", 1) for kv in args.linus_env)) if args.linus_env else None
    wall, p = timed_sh(cmd, args, wd, env=env)
    if p.returncode != 0:
        return dict(ok=False, wall=wall, error=p.stderr[-3000:])
    d = json.loads(p.stdout)
    n3 = 3 * len(atoms)
    return dict(ok=True, wall=wall, energy=d["energy"], hessian=d["hessian"], cphf_iterations=d["cphf_iterations"], timers=d["timers"], internal_time=d["time"], h=np.array(d["hessian"]).reshape(n3, n3))


def run_orca(atoms, args, wd):
    ks = "UKS " if args.mult > 1 else ""
    inp = [f"! {ks}B3LYP/G def2-SVP def2/J RIJCOSX DefGrid{args.defgrid} TightSCF Freq Bohrs", f"%pal nprocs {args.threads} end", "%maxcore 1500", f"* xyz {args.charge} {args.mult}"]
    inp += [f"{a} {c[0] / BOHR:.12f} {c[1] / BOHR:.12f} {c[2] / BOHR:.12f}" for a, c in atoms] + ["*", ""]
    (wd / "orca.inp").write_text("\n".join(inp))
    env = dict(os.environ, PATH=f"{ORCA}:{REFENV}/bin:" + os.environ["PATH"], LD_LIBRARY_PATH=f"{ORCA}/lib:{REFENV}/lib",
               OMPI_MCA_btl="self,vader", OMPI_MCA_btl_vader_single_copy_mechanism="none", OMPI_MCA_hwloc_base_binding_policy="none")
    wall, p = timed_sh([str(ORCA / "orca"), "orca.inp"], args, wd, env=env)
    out = p.stdout
    (wd / "orca.out").write_text(out)
    ok = "ORCA TERMINATED NORMALLY" in out
    e = [float(l.split()[-1]) for l in out.splitlines() if "FINAL SINGLE POINT ENERGY" in l]
    res = dict(ok=ok and bool(e), wall=wall, energy=e[-1] if e else None)
    # timings of the SCF-Hessian steps as ORCA prints them
    res["orca_timings"] = {m.group(1).strip(): float(m.group(2)) for m in re.finditer(r"^\s*([A-Za-z][^\n.]*?)\s*\.{3,}\s*([0-9.]+)\s*sec", out, re.M)}
    hf = wd / "orca.hess"
    if hf.exists():
        lines = hf.read_text().splitlines()
        i = lines.index("$hessian")
        n3 = int(lines[i + 1])
        h = np.zeros((n3, n3))
        r = i + 2
        c0 = 0
        while c0 < n3:
            cols = [int(x) for x in lines[r].split()]
            for k in range(n3):
                vals = lines[r + 1 + k].split()
                for j, c in enumerate(cols):
                    h[int(vals[0]), c] = float(vals[1 + j])
            c0 += len(cols)
            r += n3 + 1
        res["h"] = h
        res["hessian"] = h.ravel().tolist()
    if not ok:
        res["error"] = out[-3000:] + "\nSTDERR:\n" + p.stderr[-3000:]
    return res


PYSCF_JOB = r'''
import json, os, sys, time
a = json.load(open(sys.argv[1]))
if a.get("coretype"):
    # OPENBLAS_CORETYPE for the PySCF wheel's bundled OpenBLAS only: numpy/scipy (conda
    # OpenBLAS, which segfaults in scipy's Cholesky with a forced core type) load first
    import numpy, scipy.linalg
    scipy.linalg.cholesky(numpy.eye(2))
    os.environ["OPENBLAS_CORETYPE"] = a["coretype"]
from pyscf import gto, dft, lib
lib.num_threads(a["threads"])
mol = gto.M(atom=[(s, [x / 0.529177210544 for x in c]) for s, c in a["atoms"]], unit="Bohr", basis="def2-svp", charge=a["charge"], spin=a["mult"] - 1, verbose=0, max_memory=8000)
mf = dft.UKS(mol).density_fit(auxbasis="def2-universal-jkfit")
mf.xc = "b3lyp"; mf.grids.level = 3; mf.conv_tol = 1e-10; mf.conv_tol_grad = 1e-6
t0 = time.perf_counter(); e = mf.kernel(); t1 = time.perf_counter()
mf.conv_tol_cpscf = 1e-8
hobj = mf.Hessian(); hobj.max_memory = 8000
h = hobj.kernel().transpose(0, 2, 1, 3).reshape(3 * mol.natm, 3 * mol.natm)
t2 = time.perf_counter()
json.dump(dict(energy=e, scf_time=t1 - t0, hess_time=t2 - t1, hessian=h.ravel().tolist()), open(sys.argv[2], "w"))
'''


def run_pyscf(atoms, args, wd):
    (wd / "job.py").write_text(PYSCF_JOB)
    (wd / "in.json").write_text(json.dumps(dict(atoms=atoms, charge=args.charge, mult=args.mult, threads=args.threads, coretype=args.coretype)))
    env = dict(os.environ, OMP_NUM_THREADS=str(args.threads))
    wall, p = timed_sh([str(REFENV / "bin/python"), "-I", "job.py", "in.json", "out.json"], args, wd, env=env)
    if p.returncode != 0:
        return dict(ok=False, wall=wall, error=p.stderr[-3000:])
    d = json.loads((wd / "out.json").read_text())
    n3 = 3 * len(atoms)
    d["h"] = np.array(d["hessian"]).reshape(n3, n3)
    return dict(ok=True, wall=wall, **d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prog")
    ap.add_argument("xyz")
    ap.add_argument("--out", required=True)
    ap.add_argument("--charge", type=int, default=0)
    ap.add_argument("--mult", type=int, default=2)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--slice", default="bench")
    ap.add_argument("--lock", default=None)
    ap.add_argument("--linus-bin", default=str(ROOT / "target/release/examples/hess"))
    ap.add_argument("--grid", default="orca:2", help="linus grid: orca:N, pyscf:N (or N), linus:N or auto")
    ap.add_argument("--aux", default="def2-universal-jkfit", help="linus RI-JK auxiliary basis")
    ap.add_argument("--linus-env", nargs="*", default=[], help="KEY=VALUE environment settings for linus")
    ap.add_argument("--defgrid", type=int, default=2, help="ORCA DefGrid level")
    ap.add_argument("--ref", action="store_true")
    ap.add_argument("--coretype", default=None)
    ap.add_argument("--fixed-grid", action="store_true", help="linus: XC on a fixed grid (PySCF's convention) instead of the default first-order grid terms (as ORCA)")
    args = ap.parse_args()
    atoms = read_xyz(args.xyz)
    with tempfile.TemporaryDirectory() as td:
        wd = pathlib.Path(td)
        before = slice_list()
        res = dict(linus=run_linus, orca=run_orca, pyscf=run_pyscf)[args.prog](atoms, args, wd)
        res["slice_list_before"] = before
        res["slice_list_after"] = slice_list()
    h = res.pop("h", None)
    if h is not None:
        res["frequencies"] = frequencies(0.5 * (h + h.T), atoms)
    res.update(prog=args.prog, mol=pathlib.Path(args.xyz).stem, charge=args.charge, mult=args.mult, threads=args.threads, slice=args.slice,
               ref=args.ref, coretype=args.coretype, hess_grid_response=not args.fixed_grid, run=time.strftime("%Y-%m-%dT%H:%M:%S"))
    json.dump(res, open(args.out, "w"), indent=1)
    print(f"{args.prog}{' (ref)' if args.ref else ''}: ok={res['ok']} wall={res['wall']}")


if __name__ == "__main__":
    main()
