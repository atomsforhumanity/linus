"""RI-JK Hessian scaling benchmark: linus (defaults: RI-JK with aux auto, grid auto, first-order
grid response) vs ORCA 6.0.0 (`B3LYP/G def2-SVP def2/J RIJCOSX DefGrid2 TightSCF Freq`), wall
time and peak memory, harmonic frequencies against a reference Hessian.

usage: python bench/ri_hess_scaling.py PROG XYZ --out OUT.json [--charge Q] [--mult M]
           [--threads 8] [--slice bench|none|cpus:LIST] [--lock FILE] [--linus-bin PATH]
           [--linus-env K=V ...] [--mem GB] [--ref REF.json] [--method M] [--basis B]
--method: linus name (default b3lyp; ORCA keyword from ORCA_METHOD, e.g. wb97x-d3bj ->
wB97X-D3BJ), --basis: default def2-svp.
PROG: linus | orca. Each run is a memory-capped systemd scope (MemoryMax --mem, no swap) on the
CPU slice; with --lock the wall time is taken inside `flock LOCK` (the wait is not timed). Peak
memory: /usr/bin/time's max RSS (linus: the process; ORCA: the largest single MPI process). The
record has the frequencies (cm^-1, isotopic masses, translations/rotations projected), the
Hessian, `slice list` and the load average before the run; with --ref (a JSON with "hessian")
also max/rms |dnu| against it.
"""
import argparse, json, os, pathlib, re, subprocess, tempfile, time
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
from bench_paths import ORCA, REFENV  # noqa: E402  (env overrides: bench/bench_paths.py)
BOHR = 0.529177210544
# most abundant isotopes (amu)
MASS = {"H": 1.00782503223, "C": 12.0, "N": 14.00307400443, "O": 15.99491461957, "F": 18.99840316207, "S": 31.9720711744, "P": 30.97376199842, "K": 38.9637064864}
AMU_ME = 1822.888486209
HARTREE_CM = 219474.6313705


def read_xyz(p):
    lines = pathlib.Path(p).read_text().splitlines()
    n = int(lines[0])
    return [(l.split()[0], [float(x) for x in l.split()[1:4]]) for l in lines[2:2 + n]]


def frequencies(h, atoms):
    n = len(atoms)
    h = np.asarray(h).reshape(3 * n, 3 * n)
    h = 0.5 * (h + h.T)
    m = np.array([MASS[a] for a, _ in atoms]) * AMU_ME
    x = np.array([c for _, c in atoms]) / BOHR
    x = x - (m[:, None] * x).sum(0) / m.sum()
    sm = np.sqrt(np.repeat(m, 3))
    vecs = []
    for k in range(3):
        v = np.zeros((n, 3)); v[:, k] = 1.0
        vecs.append((v * np.sqrt(m)[:, None]).ravel())
        e = np.zeros(3); e[k] = 1.0
        vecs.append((np.cross(e, x) * np.sqrt(m)[:, None]).ravel())
    q, s, _ = np.linalg.svd(np.array(vecs).T, full_matrices=False)
    q = q[:, s > 1e-6 * s.max()]
    p = np.eye(3 * n) - q @ q.T
    w = np.linalg.eigvalsh(p @ (h / np.outer(sm, sm)) @ p)
    w = w[np.argsort(np.abs(w))][q.shape[1]:]
    return np.sort(np.sign(w) * np.sqrt(np.abs(w)) * HARTREE_CM)


def prefix(args):
    pre = ["systemd-run", "--user", "--scope", "-q", "-p", f"MemoryMax={args.mem}G", "-p", "MemorySwapMax=0", "-p", "OOMPolicy=continue"]
    if args.lock:
        pre += ["flock", args.lock]
    if args.slice == "none":
        return pre
    if args.slice.startswith("cpus:"):
        return pre + ["taskset", "-c", args.slice[5:]]
    return pre + [os.path.expanduser("~/.local/bin/slice"), args.slice]


def timed(cmd, args, cwd, env=None):
    """`cmd` under the scope, lock and slice; wall time and max RSS taken inside the lock."""
    inner = "s=$(date +%s.%N); /usr/bin/time -f 'MAXRSS %M' \"$@\"; rc=$?; e=$(date +%s.%N); echo \"WALL $s $e\" >&2; exit $rc"
    p = subprocess.run(prefix(args) + ["bash", "-c", inner, "x"] + cmd, cwd=cwd, env=env, capture_output=True, text=True)
    m = re.search(r"WALL (\S+) (\S+)", p.stderr)
    r = re.search(r"MAXRSS (\d+)", p.stderr)
    return (float(m.group(2)) - float(m.group(1)) if m else None), (int(r.group(1)) / 1e6 if r else None), p


# linus method -> ORCA keyword
ORCA_METHOD = {"b3lyp": "B3LYP/G", "wb97x-d3bj": "wB97X-D3BJ", "wb97x": "wB97X", "cam-b3lyp": "CAM-B3LYP", "wb97m-d3bj": "wB97M-D3BJ"}


def run_linus(atoms, args, wd):
    xyz = wd / "mol.xyz"
    xyz.write_text(f"{len(atoms)}\n\n" + "\n".join(f"{a} {c[0]:.10f} {c[1]:.10f} {c[2]:.10f}" for a, c in atoms) + "\n")
    cmd = [args.linus_bin, str(xyz), args.basis, "--method", args.method, "--threads", str(args.threads), "--charge", str(args.charge), "--mult", str(args.mult), "--jk", "df", "--set", "jk.aux=auto", "--grid", "auto"]
    env = dict(os.environ, **dict(kv.split("=", 1) for kv in args.linus_env))
    wall, rss, p = timed(cmd, args, wd, env=env)
    if p.returncode != 0:
        return dict(ok=False, wall=wall, error=p.stderr[-3000:])
    d = json.loads(p.stdout)
    return dict(ok=True, wall=wall, maxrss_gb=rss, energy=d["energy"], hessian=d["hessian"], cphf_iterations=d["cphf_iterations"], timers=d["timers"], internal_time=d["time"], env=args.linus_env)


def run_orca(atoms, args, wd):
    ks = "UKS " if args.mult > 1 else ""
    inp = [f"! {ks}{ORCA_METHOD[args.method.lower()]} {args.basis} def2/J RIJCOSX DefGrid2 TightSCF Freq Bohrs", f"%pal nprocs {args.threads} end", "%maxcore 1500", f"* xyz {args.charge} {args.mult}"]
    inp += [f"{a} {c[0] / BOHR:.12f} {c[1] / BOHR:.12f} {c[2] / BOHR:.12f}" for a, c in atoms] + ["*", ""]
    (wd / "orca.inp").write_text("\n".join(inp))
    env = dict(os.environ, PATH=f"{ORCA}:{REFENV}/bin:" + os.environ["PATH"], LD_LIBRARY_PATH=f"{ORCA}/lib:{REFENV}/lib",
               OMPI_MCA_btl="self,vader", OMPI_MCA_btl_vader_single_copy_mechanism="none", OMPI_MCA_hwloc_base_binding_policy="none")
    wall, rss, p = timed([str(ORCA / "orca"), "orca.inp"], args, wd, env=env)
    out = p.stdout
    ok = "ORCA TERMINATED NORMALLY" in out
    e = [float(l.split()[-1]) for l in out.splitlines() if "FINAL SINGLE POINT ENERGY" in l]
    res = dict(ok=ok and bool(e), wall=wall, maxrss_gb_largest_process=rss, energy=e[-1] if e else None,
               orca_timings={m.group(1).strip(): float(m.group(2)) for m in re.finditer(r"^\s*=> ([^\n.]*?)\s*\.{3,}\s*done \(\s*([0-9.]+) sec\)", out, re.M)})
    for key, pat in [("scf_s", r"Total SCF time: 0 days (\d+) hours (\d+) min (\d+) sec"), ("cpscf_s", r"CP-SCF equations solved in\s+([0-9.]+) sec")]:
        m = re.search(pat, out)
        if m:
            res[key] = float(m.group(1)) if len(m.groups()) == 1 else 3600 * int(m.group(1)) + 60 * int(m.group(2)) + int(m.group(3))
    hf = wd / "orca.hess"
    if hf.exists():
        lines = hf.read_text().splitlines()
        i = lines.index("$hessian")
        n3 = int(lines[i + 1])
        h = np.zeros((n3, n3))
        r, c0 = i + 2, 0
        while c0 < n3:
            cols = [int(x) for x in lines[r].split()]
            for k in range(n3):
                vals = lines[r + 1 + k].split()
                for j, c in enumerate(cols):
                    h[int(vals[0]), c] = float(vals[1 + j])
            c0 += len(cols)
            r += n3 + 1
        res["hessian"] = h.ravel().tolist()
    if not ok:
        res["error"] = out[-3000:] + "\nSTDERR:\n" + p.stderr[-3000:]
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prog")
    ap.add_argument("xyz")
    ap.add_argument("--out", required=True)
    ap.add_argument("--charge", type=int, default=0)
    ap.add_argument("--mult", type=int, default=1)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--slice", default="bench")
    ap.add_argument("--lock", default=None)
    ap.add_argument("--outer-lock", action="store_true", help="the caller holds the bench lock (recorded as locked)")
    ap.add_argument("--mem", default="20")
    ap.add_argument("--linus-bin", default=str(ROOT / "target/release/examples/hess"))
    ap.add_argument("--linus-env", nargs="*", default=[])
    ap.add_argument("--ref", default=None)
    ap.add_argument("--method", default="b3lyp")
    ap.add_argument("--basis", default="def2-svp")
    args = ap.parse_args()
    atoms = read_xyz(args.xyz)
    sl = subprocess.run([os.path.expanduser("~/.local/bin/slice"), "list"], capture_output=True, text=True).stdout.split("\n")
    load = open("/proc/loadavg").read().split()[:3]
    with tempfile.TemporaryDirectory() as td:
        wd = pathlib.Path(td)
        rec = run_linus(atoms, args, wd) if args.prog == "linus" else run_orca(atoms, args, wd)
    rec.update(prog=args.prog, method=args.method, basis=args.basis, xyz=str(args.xyz), charge=args.charge, mult=args.mult, threads=args.threads, slice=args.slice, locked=bool(args.lock) or args.outer_lock,
               slice_list_before=sl, loadavg_before=load, date=time.strftime("%Y-%m-%dT%H:%M:%S"))
    if rec.get("hessian"):
        f = frequencies(rec["hessian"], atoms)
        rec["frequencies"] = f.tolist()
        if args.ref:
            fr = frequencies(json.loads(pathlib.Path(args.ref).read_text())["hessian"], atoms)
            d = f - fr
            rec.update(ref=args.ref, max_abs_dnu=float(np.abs(d).max()), rms_dnu=float(np.sqrt((d ** 2).mean())), nu_at_max=float(fr[int(np.abs(d).argmax())]))
    pathlib.Path(args.out).write_text(json.dumps(rec))
    print(f"{args.prog} {pathlib.Path(args.xyz).name}: ok {rec.get('ok')} wall {rec.get('wall')} s maxrss {rec.get('maxrss_gb') or rec.get('maxrss_gb_largest_process')} GB max|dnu| {rec.get('max_abs_dnu')}")


if __name__ == "__main__":
    main()
