"""RI-JK beyond the in-memory limit: bench/rsh_bench.py with a method choice, DF storage
settings for linus and the peak RSS of every run. linus RI-JK under a memory cap vs linus
RIJ + COSX vs ORCA 6.0.0 RIJCOSX, energies and gradients.

usage: .refenv/bin/python bench/rik_bench.py --tag TAG --cases ts61:def2-tzvp --method wb97x-d3bj \
           --calcs energy,gradient --programs linus-ri,linus-cosx [--mem 16] [--linus-env K=V ...] \
           [--label -coef32] [--slice bench|none|ccd2] [--reps 1]
--no-lock times without the shared bench lock (development runs on an idle CCD, e.g. --slice ccd2;
the program key then carries @ccd2). --mem caps each run's memory scope (GB); linus's default 3-index budget is 75% of the cap, so the
cap is the memory the run is held to. --linus-env passes environment variables to linus-ri runs
(the DF storage settings are request keys now, `memory_gib` and `jk.df_storage`; the LINUS_DF_MEM_GB,
LINUS_DF_PRECISION and LINUS_DF_LR variables of older runs are not read). Peak RSS: /usr/bin/time %M of the program
(for ORCA that is its driver process, not the sum over MPI ranks).

Settings (matched production level, AFH's wB97X-D3BJ):
  linus-ri    jk df, def2-universal-jkfit (= ORCA def2/JK), grid orca:2 (DefGrid2), 8 threads
  linus-cosx  jk rij-snk, def2-universal-jfit (= def2/J), COSX grids of DefGrid2 (cosx:2)
  linus-exact jk exact (four-center), grid orca:2
  orca        ! wB97X-D3BJ <basis> def2/J RIJCOSX DefGrid2 TightSCF [EnGrad] Bohrs, 8 MPI ranks
  orca-exact  ! wB97X-D3BJ <basis> NoRI NoCOSX DefGrid2 TightSCF Bohrs
linus uses its per-purpose SCF defaults (energy: tight, |dE| < 1e-8 and max|F_ai| < 5e-7;
gradient: e 1e-9, max|F_ai| < 1e-7); ORCA TightSCF. Coordinates go to both programs in bohr.
Gradients are also stored without dispersion (linus: minus its D3(BJ) part; ORCA: minus its
DISPERSION GRADIENT block), since ORCA 6.0.0's analytic D3(BJ) gradient is not the derivative
of its D3 energy.

Each timed run holds the shared bench lock (wall time starts once it is held) and runs under
`slice` in a memory-capped systemd scope; `slice list` is recorded with every run.
Results: bench/results/TAG/<mol>_<basis>_<calc>.json (one file per case, all programs).
"""
import argparse, fcntl, json, os, pathlib, re, subprocess, tempfile, time

ROOT = pathlib.Path(__file__).resolve().parents[1]
from bench_paths import LOCK, MAIN, ORCA, REFENV  # noqa: E402,F401  (env overrides: bench/bench_paths.py)
SLICE = str(pathlib.Path.home() / ".local/bin/slice")
BOHR = 0.529177210544
CHARGES = {"la18": 0, "u1an39": -1, "ts61": 0, "ts113": 0}
METHOD = "wb97x-d3bj"
TIME = "/usr/bin/time"
# linus method -> ORCA keyword (linus/libxc b3lyp uses VWN-RPA = ORCA B3LYP/G; ORCA "B3LYP" is VWN5)
ORCA_METHOD = {"wb97x-d3bj": "wB97X-D3BJ", "b3lyp": "B3LYP/G"}


def read_xyz(mol):
    lines = (ROOT / "bench/molecules" / f"{mol}.xyz").read_text().splitlines()
    n = int(lines[0])
    return [(l.split()[0], [float(x) / BOHR for x in l.split()[1:4]]) for l in lines[2:2 + n]]


def slice_list():
    return subprocess.run([SLICE, "list"], capture_output=True, text=True).stdout.strip().splitlines()


def wrap(cmd, slc, mem, env=()):
    pre = ["systemd-run", "--user", "--scope", "-q", "-p", f"MemoryMax={mem}G", "-p", "MemorySwapMax=0", "-p", "OOMPolicy=continue"]
    pre += [f"--setenv={e}" for e in env]
    if slc != "none":
        pre += [SLICE, slc] if not slc.startswith("cpus:") else ["taskset", "-c", slc[5:]]
    return pre + [TIME, "-f", "MAXRSS_KB %M"] + cmd


def peak_rss_gb(stderr):
    for l in (stderr or "").splitlines():
        if l.startswith("MAXRSS_KB"):
            return int(l.split()[1]) / 1048576.0
    return None


def timed(cmd, lock, **kw):
    """With `lock`, hold the shared bench lock; the wall time starts once it is held."""
    if not lock:
        t = time.perf_counter()
        p = subprocess.run(cmd, capture_output=True, text=True, **kw)
        return p, time.perf_counter() - t
    with open(LOCK, "a") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        t = time.perf_counter()
        p = subprocess.run(cmd, capture_output=True, text=True, **kw)
        return p, time.perf_counter() - t


def run_linus(prog, mol, basis, calc, a):
    atoms = read_xyz(mol)
    req = {"structure": {"symbols": [s for s, _ in atoms], "geometry": [c for _, c in atoms], "charge": CHARGES.get(mol, 0),
                         "multiplicity": 1},
           "calctype": calc, "method": a.method, "basis": basis, "grid": {"xc": "orca:2"}, "threads": a.threads}
    if prog == "linus-ri":
        req["jk"] = {"kind": "df", "aux": "def2-universal-jkfit"}
    elif prog == "linus-cosx":
        req["jk"] = {"kind": "rij-snk", "aux": "def2-universal-jfit", "k_grid": "cosx:2"}
    else:
        req["jk"] = "exact"
    env = a.linus_env if prog == "linus-ri" else []
    p, wall = timed(wrap([a.linus_bin, "--json"], a.slice, a.mem, env), not (a.ref or a.no_lock), input=json.dumps(req))
    if p.returncode != 0:
        return dict(ok=False, wall=wall, env=env, mem_cap_gb=a.mem, peak_rss_gb=peak_rss_gb(p.stderr), error=(p.stderr or "")[-2000:] or f"exit {p.returncode} (killed?)")
    d = json.loads(p.stdout)
    r = dict(ok=True, wall=wall, energy=d["energy"], iterations=d["iterations"], nbf=d["nbf"], timings={**d["timings"], **d.get("resources", {})},
             e_disp=d["components"].get("dispersion"), components=d["components"], env=env, mem_cap_gb=a.mem, peak_rss_gb=peak_rss_gb(p.stderr))
    if "gradient" in d:
        r["gradient"] = d["gradient"]
        parts = d.get("gradient_parts") or {}
        if "dispersion" in parts:
            r["gradient_no_disp"] = [[g - x for g, x in zip(gr, dr)] for gr, dr in zip(d["gradient"], parts["dispersion"])]
    return r


def orca_input(mol, basis, calc, exact, threads, method):
    approx = "NoRI NoCOSX" if exact else "def2/J RIJCOSX"
    kw = f"! {ORCA_METHOD[method]} {basis} {approx} DefGrid2 TightSCF" + (" EnGrad" if calc == "gradient" else "") + " Bohrs"
    lines = [kw, f"%pal nprocs {threads} end", "%maxcore 2000", f"* xyz {CHARGES.get(mol, 0)} 1"]
    lines += [f"{s} {c[0]:.12f} {c[1]:.12f} {c[2]:.12f}" for s, c in read_xyz(mol)] + ["*", ""]
    return "\n".join(lines)


def parse_orca(out, wd, natom):
    r = {}
    e = re.findall(r"FINAL SINGLE POINT ENERGY\s+(-?\d+\.\d+)", out)
    r["energy"] = float(e[-1]) if e else None
    m = re.search(r"SCF CONVERGED AFTER\s+(\d+)\s+CYCLES", out)
    r["iterations"] = int(m.group(1)) if m else None
    m = re.search(r"Dispersion correction\s+(-?\d+\.\d+)", out)
    r["e_disp"] = float(m.group(1)) if m else None
    for key, pat in [("t_scf", r"Total SCF time:.*?(\d+) days (\d+) hours (\d+) min (\d+) sec"),]:
        pass
    m = re.search(r"Total SCF gradient time\s+\.+\s+(\d+\.\d+) sec", out)
    r["t_gradient"] = float(m.group(1)) if m else None
    m = re.search(r"SCF iterations\s+\.\.\.\s+(\d+\.\d+) sec", out)
    r["t_scf_iterations"] = float(m.group(1)) if m else None
    eg = wd / "orca.engrad"
    if eg.exists():
        data = [l.strip() for l in eg.read_text().splitlines() if l.strip() and not l.startswith("#")]
        n = int(data[0])
        g = [float(x) for x in data[2:2 + 3 * n]]
        r["gradient"] = [g[3 * i:3 * i + 3] for i in range(n)]
        blk = out.split("DISPERSION GRADIENT")
        if len(blk) > 1:
            rows = re.findall(r"^\s*\d+\s+\w+\s*:\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)", blk[-1], re.M)[:natom]
            dg = [[float(x) for x in row] for row in rows]
            r["gradient_disp"] = dg
            r["gradient_no_disp"] = [[a - b for a, b in zip(ga, gb)] for ga, gb in zip(r["gradient"], dg)]
    return r


def run_orca(prog, mol, basis, calc, a):
    env = dict(os.environ, PATH=f"{ORCA}:{REFENV}/bin:" + os.environ["PATH"], LD_LIBRARY_PATH=f"{ORCA}/lib:{REFENV}/lib",
               OMPI_MCA_btl="self,vader", OMPI_MCA_btl_vader_single_copy_mechanism="none", OMPI_MCA_hwloc_base_binding_policy="none")
    (ROOT / "bench/scratch").mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=ROOT / "bench/scratch") as td:
        wd = pathlib.Path(td)
        (wd / "orca.inp").write_text(orca_input(mol, basis, calc, prog == "orca-exact", a.threads, a.method))
        p, wall = timed(wrap([str(ORCA / "orca"), "orca.inp"], a.slice, a.mem), not (a.ref or a.no_lock), cwd=wd, env=env)
        out = p.stdout
        ok = "ORCA TERMINATED NORMALLY" in out
        r = dict(ok=ok, wall=wall, peak_rss_gb=peak_rss_gb(p.stderr), **parse_orca(out, wd, len(read_xyz(mol))))
        if not ok:
            r["error"] = out[-2000:] + (p.stderr or "")[-1000:]
        return r


RUN = {"linus-ri": run_linus, "linus-cosx": run_linus, "linus-exact": run_linus, "orca": run_orca, "orca-exact": run_orca}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--cases", required=True, help="mol:basis,...")
    ap.add_argument("--calcs", default="energy,gradient")
    ap.add_argument("--programs", default="linus-ri,linus-cosx,orca")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--slice", default="bench")
    ap.add_argument("--mem", type=int, default=18, help="GB per process (systemd scope cap)")
    ap.add_argument("--ref", action="store_true", help="reference runs: no lock")
    ap.add_argument("--label", default="", help="suffix of the program key (e.g. a later linus build)")
    ap.add_argument("--method", default=METHOD, choices=sorted(ORCA_METHOD))
    ap.add_argument("--linus-env", action="append", default=[], help="K=V for linus-ri runs (DF storage overrides)")
    ap.add_argument("--no-lock", action="store_true", help="timed without the bench lock (development timings on an idle CCD)")
    ap.add_argument("--linus-bin", default=str(ROOT / "target/release/linus"), help="linus binary (a frozen copy while the tree changes)")
    ap.add_argument("--commit", default="", help="source revision of the linus binary, recorded with every linus run")
    a = ap.parse_args()
    outdir = ROOT / "bench/results" / a.tag
    outdir.mkdir(parents=True, exist_ok=True)
    for case in a.cases.split(","):
        mol, basis = case.split(":")
        for calc in a.calcs.split(","):
            f = outdir / f"{mol}_{basis}_{a.method}_{calc}.json"
            rec = json.loads(f.read_text()) if f.exists() else dict(mol=mol, basis=basis, calc=calc, method=a.method, threads=a.threads, runs={})
            for prog in a.programs.split(","):
                for rep in range(a.reps):
                    r = RUN[prog](prog, mol, basis, calc, a)
                    r.update(rep=rep, slice=a.slice, slice_list=slice_list(), load=open("/proc/loadavg").read().split()[:3],
                             time=time.strftime("%Y-%m-%dT%H:%M:%S"), locked=not (a.ref or a.no_lock))
                    if prog.startswith("linus"):
                        r["commit"] = a.commit
                    key = (prog if a.slice in ("bench",) or a.ref else f"{prog}@{a.slice}") + a.label
                    rec["runs"].setdefault(key, []).append(r)
                    print(f"{mol:7} {basis:9} {calc:8} {key:18} rep {rep}: {'ok ' if r['ok'] else 'FAIL'} {r['wall']:8.2f} s  E={r.get('energy')}"
                          + ("" if r["ok"] else "  " + r.get("error", "")[-300:].replace("\n", " ")), flush=True)
                    f.write_text(json.dumps(rec, indent=1))


if __name__ == "__main__":
    main()
