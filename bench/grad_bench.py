"""RIJCOSX gradient (and energy) benchmark: linus (jk = rij-snk) against ORCA 6.0.0 (RIJCOSX) at matched settings.

Each run is timed inside the shared bench lock (flock) on the chosen CPU slice (default `slice
bench`, CCD0's 8 physical cores; `none` = unpinned), with one warm-up per program and case.
Writes one JSON per case to bench/results/<tag>/ (no host names): wall times of every
repetition, SCF and gradient times (linus: its timers; ORCA: its module times), energies,
gradients, SCF iterations, `slice list` before each run and the settings.

linus: B3LYP/PBE, jk = rij-snk (def2-universal-jfit, COSX grids of DefGrid2: SCF on GridX1/2,
final energy and gradient on GridX3), XC grid orca:2, default gradient convergence.
ORCA:  `! B3LYP/G <basis> def2/J RIJCOSX DefGrid2 TightSCF EnGrad Bohrs` (PBE likewise; without EnGrad
with --calc energy).

usage: .refenv/bin/python bench/grad_bench.py --tag rijcosx-grad --mols la18,u1an39,ts61 \
           --levels b3lyp-svp,b3lyp-tzvp,pbe-svp,pbe-tzvp --programs linus,orca --reps 2
"""
import argparse, contextlib, fcntl, json, os, pathlib, re, subprocess, tempfile, time

ROOT = pathlib.Path(__file__).resolve().parents[1]
MOLS = ROOT / "bench/molecules"
from bench_paths import ORCA, REFENV  # noqa: E402  (env overrides: bench/bench_paths.py)
# the reference environment (mpirun for ORCA) lives in the main checkout, also for worktrees
BOHR = 0.529177210544
CHARGES = {"la18": 0, "u1an39": -1, "ts61": 0, "ts113": 0}
FUNC = {"b3lyp": ("b3lyp", "B3LYP/G"), "pbe": ("pbe", "PBE")}
BASIS = {"svp": "def2-SVP", "tzvp": "def2-TZVP"}


def level_settings(level):
    f, b = level.split("-")
    return dict(linus=dict(method=FUNC[f][0], basis=BASIS[b].lower(), jk=dict(kind="rij-snk", aux="def2-universal-jfit", k_grid="cosx:2"), grid=dict(xc="orca:2")),
                orca=f"! {FUNC[f][1]} {BASIS[b]} def2/J RIJCOSX DefGrid2 TightSCF")


def read_xyz(name):
    lines = (MOLS / f"{name}.xyz").read_text().splitlines()
    n = int(lines[0])
    return [(l.split()[0], [float(x) for x in l.split()[1:4]]) for l in lines[2:2 + n]]


def prefix(slc, mem):
    p = ["systemd-run", "--user", "--scope", "-q", "-p", f"MemoryMax={mem}G", "-p", "MemorySwapMax=0"]
    if slc.startswith("cpus:"):
        p += ["taskset", "-c", slc[5:]]
    elif slc != "none":
        p += [str(pathlib.Path.home() / ".local/bin/slice"), slc]
    return p


@contextlib.contextmanager
def held(lock):
    """The shared bench lock (flock), so that timings start once no other run holds CCD0."""
    if not lock:
        yield
        return
    with open(lock, "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def run(cmd, cwd, lock, env=None, stdin=None):
    with held(lock):
        slist = subprocess.run([str(pathlib.Path.home() / ".local/bin/slice"), "list"], capture_output=True, text=True).stdout
        # the node's load average when the timed run starts (shared node)
        slist += "load average: " + " ".join("%.2f" % x for x in os.getloadavg()) + "\n"
        t = time.perf_counter()
        p = subprocess.run(cmd, cwd=cwd, env=env, input=stdin, capture_output=True, text=True)
        return time.perf_counter() - t, p, slist


def run_linus(atoms, charge, level, threads, slc, lock, wd, binary, calc="gradient"):
    s = level_settings(level)["linus"]
    req = {"structure": {"symbols": [a for a, _ in atoms], "geometry": [[x / BOHR for x in c] for _, c in atoms],
                         "charge": charge, "multiplicity": 1},
           "calctype": calc, "threads": threads, **s}
    wall, p, slist = run(prefix(slc, 16) + [binary, "--json"], wd, lock, stdin=json.dumps(req))
    if p.returncode != 0:
        return dict(ok=False, wall=wall, error=p.stderr[-2000:], slice_list=slist)
    d = json.loads(p.stdout)
    t = d.get("timings", {})
    # timings.energy: the SCF stage, timings.total: the whole call (before: total and calculation)
    scf = t.get("energy", t.get("total"))
    end = t.get("calculation", t.get("total")) if "energy" not in t else t.get("total")
    grad = end - scf if calc == "gradient" and scf is not None and end is not None and ("energy" in t or "calculation" in t) else None
    return dict(ok=True, wall=wall, energy=d["energy"], gradient=d.get("gradient"), iterations=d["iterations"], nbf=d["nbf"],
                scf_time=scf, grad_time=grad, timings=t, slice_list=slist)


def run_orca(atoms, charge, level, threads, slc, lock, wd, maxcore, calc="gradient"):
    kw = level_settings(level)["orca"] + (" EnGrad" if calc == "gradient" else "") + " Bohrs"
    inp = [kw, f"%pal nprocs {threads} end", f"%maxcore {maxcore}", f"* xyz {charge} 1"]
    inp += [f"{a} {c[0] / BOHR:.12f} {c[1] / BOHR:.12f} {c[2] / BOHR:.12f}" for a, c in atoms] + ["*", ""]
    (wd / "orca.inp").write_text("\n".join(inp))
    env = dict(os.environ, PATH=f"{ORCA}:{REFENV}/bin:" + os.environ["PATH"], LD_LIBRARY_PATH=f"{ORCA}/lib:{REFENV}/lib",
               OMPI_MCA_btl="self,vader", OMPI_MCA_btl_vader_single_copy_mechanism="none", OMPI_MCA_hwloc_base_binding_policy="none")
    wall, p, slist = run(prefix(slc, 16) + [str(ORCA / "orca"), "orca.inp"], wd, lock, env=env)
    out = p.stdout
    ok = "ORCA TERMINATED NORMALLY" in out
    e = [float(l.split()[-1]) for l in out.splitlines() if "FINAL SINGLE POINT ENERGY" in l]
    it = [int(x) for x in re.findall(r"SCF CONVERGED AFTER\s+(\d+)", out)]
    res = dict(ok=ok and bool(e), wall=wall, energy=e[-1] if e else None, iterations=it[-1] if it else None, slice_list=slist)
    for key, pat in [("scf_time", r"SCF iterations\s+\.\.\.\s+([\d.]+) sec"), ("grad_time", r"SCF Gradient evaluation\s+\.\.\.\s+([\d.]+) sec"),
                     ("startup_time", r"Startup calculation\s+\.\.\.\s+([\d.]+) sec"), ("sum_modules", r"Sum of individual times\s+\.\.\.\s+([\d.]+) sec")]:
        m = re.search(pat, out)
        res[key] = float(m.group(1)) if m else None
    parts = {}
    for name in ["One electron gradient", "RI-J Coulomb gradient", "COSX gradient", "XC gradient"]:
        m = re.search(re.escape(name) + r"\s+\.+\s+([\d.]+) sec", out)
        if m:
            parts[name] = float(m.group(1))
    res["grad_parts"] = parts
    eg = wd / "orca.engrad"
    if eg.exists():
        data = [l.strip() for l in eg.read_text().splitlines() if l.strip() and not l.strip().startswith("#")]
        n = int(data[0])
        g = [float(x) for x in data[2:2 + 3 * n]]
        res["gradient"] = [g[3 * i:3 * i + 3] for i in range(n)]
    if not res["ok"]:
        res["error"] = out[-2000:] + p.stderr[-1000:]
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--mols", default="la18")
    ap.add_argument("--levels", default="b3lyp-svp")
    ap.add_argument("--programs", default="linus,orca")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--slice", default="bench", help="slice name, 'none' (unpinned) or 'cpus:LIST' (taskset, e.g. 0-31,64-95 = CCDs 0-3)")
    ap.add_argument("--lock", default=os.environ.get("BENCH_LOCK", ""), help="flock file held during each timed run")
    ap.add_argument("--linus", default=str(ROOT / "target/release/linus"))
    ap.add_argument("--maxcore", type=int, default=1500, help="ORCA %%maxcore per rank (MB)")
    ap.add_argument("--calc", default="gradient", choices=["gradient", "energy"])
    a = ap.parse_args()
    outdir = ROOT / "bench/results" / a.tag
    outdir.mkdir(parents=True, exist_ok=True)
    scratch = ROOT / "bench/scratch"
    scratch.mkdir(exist_ok=True)
    progs = a.programs.split(",")
    for mol in a.mols.split(","):
        atoms = read_xyz(mol)
        for level in a.levels.split(","):
            case = f"{mol}_{level}_{a.calc}"
            runs = {p: [] for p in progs}

            def one(prog):
                with tempfile.TemporaryDirectory(dir=scratch) as td:
                    if prog == "linus":
                        return run_linus(atoms, CHARGES.get(mol, 0), level, a.threads, a.slice, a.lock, pathlib.Path(td), a.linus, a.calc)
                    return run_orca(atoms, CHARGES.get(mol, 0), level, a.threads, a.slice, a.lock, pathlib.Path(td), a.maxcore, a.calc)

            for _ in range(a.warmup):
                for prog in progs:
                    one(prog)
            for rep in range(a.reps):
                for prog in progs[rep % len(progs):] + progs[:rep % len(progs)]:
                    r = one(prog)
                    r["rep"] = rep
                    runs[prog].append(r)
                    print(f"{case:28} {prog:6} rep {rep}: {'ok ' if r['ok'] else 'FAIL'} wall {r['wall']:7.2f} s  scf {r.get('scf_time')}  grad {r.get('grad_time')}", flush=True)
            summary = {}
            for prog, rs in runs.items():
                ok = [r for r in rs if r["ok"]]
                if ok:
                    med = lambda k: sorted(r[k] for r in ok if r.get(k) is not None)[len(ok) // 2] if any(r.get(k) is not None for r in ok) else None
                    summary[prog] = dict(wall=med("wall"), scf=med("scf_time"), grad=med("grad_time"), energy=ok[0]["energy"], iterations=ok[0]["iterations"])
            rec = dict(case=case, mol=mol, level=level, calc=a.calc, threads=a.threads, slice=a.slice, settings=level_settings(level), summary=summary,
                       runs=runs, time=time.strftime("%Y-%m-%dT%H:%M:%S"))
            (outdir / f"{case}.json").write_text(json.dumps(rec, indent=1))
            print(case, json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
