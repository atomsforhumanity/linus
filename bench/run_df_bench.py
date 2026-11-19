"""Density-fitting J/K benchmark: linus DfJk against ORCA 6 RIJCOSX (Split-RI-J + COSX) and
PySCF DF get_jk, on this node.

  python3 bench/run_df_bench.py [--mols la18,u1an39,ts61,ts113] [--bases def2-svp,def2-tzvp]
      [--threads 8,16,64] [--reps 2] [--programs linus,orca,pyscf] [--out bench/results/df_jk.json]

Runs are interleaved (rep -> configuration -> program) and executed one at a time; `uptime` and
`slice list` are recorded before every run. With --slice NAME every command runs under
`~/.local/bin/slice NAME` (CPU affinity: bench = CCD0's 8 physical cores, work = CCDs 2-7 with
SMT siblings); ORCA then gets OMPI_MCA_hwloc_base_binding_policy=none so MPI keeps the mask.
Results (raw runs and per-configuration medians) go to JSON; no host names are written.

linus (crates/linus-dev/examples/df_bench, core-Hamiltonian occupied orbitals, closed shell):
  jk     def2-universal-jkfit, J + K: setup (3c integrals, metric, B = A L^-T), per-call J and K
  j      def2-universal-jfit, J only with the raw 3c integrals in memory
  jdir   def2-universal-jfit, J only, integral-direct (split-RI-J)
ORCA 6.0.0: ! B3LYP/G <basis> def2/J RIJCOSX DefGrid2 TightSCF; Split-RI-J and COSX times per
  SCF cycle (module totals / cycles).
PySCF: DF build and get_jk / get_j per call (def2-universal-jkfit; occupied-orbital K).
linus-rsh (program): df_bench --omega 0.3 (jkfit): B^omega setup and J + K + K_LR per call.
orca-rsh (program): ORCA ! wB97X-D3BJ <basis> def2/J RIJCOSX DefGrid2 TightSCF.
scf-ri (program): full linus B3LYP SCF (crates/linus-dev/examples/scf, PySCF-level-3 XC grid, SAD
  guess) with DfJk (def2-universal-jkfit); wall time, J/K setup and the summed J/K and XC times.
(The `old` / `scf-old` programs timed the previous dense DfJK, which was removed.)
"""
import argparse, datetime, json, os, pathlib, re, subprocess, time

ROOT = pathlib.Path(__file__).resolve().parents[1]
MOLS = {"la18": 0, "u1an39": -1, "ts61": 0, "ts113": 0}
from bench_paths import MAIN, ORCA as ORCA_DIR, REFENV  # noqa: E402  (env overrides: bench/bench_paths.py)
ORCA = str(ORCA_DIR / "orca")
ORCA_ENV = {
    "PATH": f"{ORCA_DIR}:{REFENV}/bin:" + os.environ.get("PATH", ""),
    "LD_LIBRARY_PATH": f"{ORCA_DIR}/lib:{REFENV}/lib",
    "OMPI_MCA_btl": "self,vader",
    "OMPI_MCA_btl_vader_single_copy_mechanism": "none",
}
PY = str(MAIN / ".venv/bin/python")
SCRATCH = pathlib.Path(os.environ.get("DF_BENCH_SCRATCH", "/tmp/linus_df_bench"))
SLICE_BIN = os.path.expanduser("~/.local/bin/slice")
SLICE = []  # command prefix: memory-capped scope + slice, set in main()
MEM_CAP_GB = 18  # per-job cap (systemd scope, no swap); the node is shared
LINUS_BUDGET_GB = 14  # DfJk 3-index tensor budget inside the cap
PYSCF_MAX_CDERI_GB = 12  # skip PySCF when its in-core DF tensor would exceed this
NBF = {"la18": {"def2-svp": 180, "def2-tzvp": 358}, "u1an39": {"def2-svp": 483, "def2-tzvp": 1034},
       "ts61": {"def2-svp": 666, "def2-tzvp": 1343}, "ts113": {"def2-svp": 1106, "def2-tzvp": 2155}}
NAUX_JK = {"la18": 902, "u1an39": 2556, "ts61": 3436, "ts113": 5520}


def cpu_busy(cpus, dt=1.0):
    """Mean busy fraction of the given logical CPUs over dt seconds (/proc/stat)."""
    def snap():
        out = {}
        for line in open("/proc/stat"):
            if line.startswith("cpu") and line[3].isdigit():
                f = line.split()
                c = int(f[0][3:])
                v = [int(x) for x in f[1:]]
                out[c] = (sum(v), v[3] + v[4])
        return out
    a = snap()
    time.sleep(dt)
    b = snap()
    fr = [1.0 - (b[c][1] - a[c][1]) / max(1, b[c][0] - a[c][0]) for c in cpus]
    return sum(fr) / len(fr)


BENCH_CPUS = list(range(8))


def wait_quiet(thresh=0.15, max_wait=3600.0):
    """Wait until CCD0 (the bench slice) is idle, so concurrent users of the slice do not
    share our cores. Returns (busy fraction, seconds waited)."""
    t0 = time.perf_counter()
    while True:
        b = cpu_busy(BENCH_CPUS, 2.0)
        if b < thresh or time.perf_counter() - t0 > max_wait:
            return b, time.perf_counter() - t0
        time.sleep(15)


def _proc_table():
    """{pid: ppid} and running threads [(pid, cpu)] from /proc."""
    ppid, running = {}, []
    for d in os.listdir("/proc"):
        if not d.isdigit():
            continue
        try:
            st = open(f"/proc/{d}/stat").read()
            post = st[st.rindex(")") + 2:].split()
            ppid[int(d)] = int(post[1])
            for t in os.listdir(f"/proc/{d}/task"):
                ts = open(f"/proc/{d}/task/{t}/stat").read()
                tp = ts[ts.rindex(")") + 2:].split()
                if tp[0] == "R":
                    running.append((int(d), int(tp[36])))
        except (OSError, ValueError, IndexError):
            continue
    return ppid, running


def run_monitored(cmd, cpus=None, **kw):
    """Run cmd (stdout captured). If cpus is given, sample every 2 s the threads of other
    user processes running on those CPUs; returns (stdout, mean number of such threads per
    sample). Unpinned processes of other users roam over all CPUs (about one core's worth on
    CCD0 on this node), so contention is flagged at a mean of 2 or more."""
    proc = subprocess.Popen(cmd, stdout=kw.pop("stdout", subprocess.PIPE), **kw)
    hits, nsamp = 0, 0
    if cpus is not None:
        me = os.getpid()
        while proc.poll() is None:
            time.sleep(2.0)
            ppid, running = _proc_table()
            mine = {me, proc.pid}
            changed = True
            while changed:
                changed = False
                for c, pp in ppid.items():
                    if pp in mine and c not in mine:
                        mine.add(c)
                        changed = True
            foreign = [(pid, c) for pid, c in running if c in cpus and pid not in mine and pid != 2 and ppid.get(pid) != 2]
            hits += len(foreign)
            nsamp += 1
    out, _ = proc.communicate()
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd)
    return out, (hits / nsamp if nsamp else 0.0)


MONITOR = None  # set of CPUs to watch (bench slice)


def slice_list():
    try:
        return subprocess.check_output([SLICE_BIN, "list"]).decode().strip().splitlines()
    except Exception as e:  # slice not installed
        return [f"slice list unavailable: {e}"]


def uptime():
    return subprocess.check_output(["uptime"]).decode().strip().split("load average:")[-1].strip()


def run_linus(mol, basis, threads, mode):
    exe = pathlib.Path(os.environ.get("DF_BENCH_BIN", ROOT / "target/release/examples/df_bench"))
    aux = "def2-universal-jkfit" if mode in ("jk", "rsh") else "def2-universal-jfit"
    cmd = [str(exe), str(ROOT / f"bench/molecules/{mol}.xyz"), basis, aux, "--threads", str(threads), "--charge", str(MOLS[mol]), "--reps", "3",
           "--budget-gb", str(LINUS_BUDGET_GB)]
    if mode in ("j", "jdir"):
        cmd.append("--j-only")
    if mode == "jdir":
        cmd.append("--direct")
    if mode == "rsh":
        cmd += ["--omega", "0.3"]
    t0 = time.perf_counter()
    o, hits = run_monitored(SLICE + cmd, MONITOR)
    d = json.loads(o)
    wall = time.perf_counter() - t0
    r = {"wall": wall, "foreign_threads": hits, "nbf": d["nbf"], "naux": d["naux"], "nocc": d["nocc"], "peak_rss_gb": d["peak_rss_gb"]}
    if mode == "rsh":
        s, it, om = d["setup"], d["iter"], d["omega"]
        r.update({"t_setup": s["t_total"], "t_lr_setup": om["t_lr_setup"], "t_jk": it["t_jk"], "t_k": it["t_k"], "t_jk_klr": om["t_jk_klr"],
                  "store_gb": s["gb"] + om["gb_lr"]})
    else:
        s, it = d["setup"], d["iter"]
        r.update({"t_setup": s["t_total"], "t_ints": s["t_ints"], "t_fit": s["t_fit"], "t_metric": s["t_metric"], "store_gb": s["gb"], "rows": s["rows"],
                  "rows_dense_packed": s["rows_dense_packed"], "storage": s["mode"], "t_j": it["t_j"], "t_k": it["t_k"], "t_jk": it["t_jk"]})
    return r


def run_scf(mol, basis, threads, jk):
    exe = pathlib.Path(os.environ.get("DF_BENCH_SCF_BIN", ROOT / "target/release/examples/scf"))
    cmd = [str(exe), str(ROOT / f"bench/molecules/{mol}.xyz"), basis, "--jk", jk, "--aux", "def2-universal-jkfit", "--xc", "b3lyp", "--grid", "3",
           "--threads", str(threads), "--charge", str(MOLS[mol])]
    t0 = time.perf_counter()
    o, hits = run_monitored(SLICE + cmd, MONITOR)
    d = json.loads(o)
    wall = time.perf_counter() - t0
    tm = d["timers"]
    return {"wall": wall, "foreign_threads": hits, "energy": d["energy"], "converged": d["converged"], "cycles": d["iterations"], "nbf": d["nbf"], "t_jk_setup": d["t_jk_setup"],
            "t_jk_total": tm.get("jk"), "t_xc_total": tm.get("xc"), "t_scf": tm.get("scf_total"), "t_total": d["t_total"],
            "t_jk_per_cycle": tm.get("jk", 0.0) / max(1, d["iterations"])}


def orca_input(mol, basis, threads, method="B3LYP/G"):
    lines = (ROOT / f"bench/molecules/{mol}.xyz").read_text().splitlines()
    n = int(lines[0])
    atoms = "\n".join(" ".join(ln.split()[:4]) for ln in lines[2:2 + n])
    maxcore = min(2000, int(MEM_CAP_GB * 1024 * 0.7 / threads))
    b = {"def2-svp": "def2-SVP", "def2-tzvp": "def2-TZVP"}[basis]
    return f"! {method} {b} def2/J RIJCOSX DefGrid2 TightSCF NoAutoStart\n%pal nprocs {threads} end\n%maxcore {maxcore}\n* xyz {MOLS[mol]} 1\n{atoms}\n*\n"


def run_orca(mol, basis, threads, rep, method="B3LYP/G"):
    d = SCRATCH / f"orca_{method.replace('/', '_')}_{mol}_{basis}_{threads}_{rep}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "orca.inp").write_text(orca_input(mol, basis, threads, method))
    for f in d.glob("orca.gbw"):  # a retry must not restart from the previous attempt's orbitals
        f.unlink()
    env = dict(os.environ, **ORCA_ENV)
    if SLICE:
        env["OMPI_MCA_hwloc_base_binding_policy"] = "none"
    t0 = time.perf_counter()
    with open(d / "orca.out", "w") as f:
        _, hits = run_monitored(SLICE + [ORCA, "orca.inp"], MONITOR, cwd=d, env=env, stdout=f, stderr=subprocess.STDOUT)
    wall = time.perf_counter() - t0
    out = (d / "orca.out").read_text()

    def sec(label):
        m = re.search(rf"^\s*{re.escape(label)}\s*\.+\s*([0-9.]+) sec", out, re.M)
        return float(m.group(1)) if m else None

    cyc = int(re.search(r"SCF CONVERGED AFTER\s+(\d+) CYCLES", out).group(1))
    e = float(re.search(r"FINAL SINGLE POINT ENERGY\s+(-?[0-9.]+)", out).group(1))
    rij, cosx, xc, fock = sec("Split-RI-J"), sec("Chain of spheres X"), sec("XC integration"), sec("Fock matrix formation")
    for f in d.glob("orca*.tmp*"):
        f.unlink()
    return {"wall": wall, "foreign_threads": hits, "cycles": cyc, "energy": e, "t_rij_total": rij, "t_cosx_total": cosx, "t_xc_total": xc, "t_fock_total": fock,
            "t_rij": rij / cyc, "t_cosx": cosx / cyc, "t_xc": xc / cyc if xc else None}


def run_pyscf(mol, basis, threads):
    nbf = NBF[mol][basis]
    gb = NAUX_JK[mol] * nbf * (nbf + 1) / 2 * 8 / 2**30
    if gb > PYSCF_MAX_CDERI_GB:
        raise RuntimeError(f"skipped: PySCF in-core DF tensor {gb:.1f} GiB exceeds the {PYSCF_MAX_CDERI_GB} GiB limit (memory-capped node)")
    cmd = [PY, str(ROOT / "bench/pyscf_df_bench.py"), str(ROOT / f"bench/molecules/{mol}.xyz"), basis, "def2-universal-jkfit", str(threads), str(MOLS[mol]), "3"]
    env = dict(os.environ, OMP_NUM_THREADS=str(threads))
    t0 = time.perf_counter()
    o, hits = run_monitored(SLICE + cmd, MONITOR, env=env)
    d = json.loads(o.decode().strip().splitlines()[-1])
    d["wall"] = time.perf_counter() - t0
    d["foreign_threads"] = hits
    return d


def median(v):
    v = sorted(x for x in v if x is not None)
    if not v:
        return None
    n = len(v)
    return v[n // 2] if n % 2 else 0.5 * (v[n // 2 - 1] + v[n // 2])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mols", default="la18,u1an39,ts61,ts113")
    ap.add_argument("--bases", default="def2-svp,def2-tzvp")
    ap.add_argument("--threads", default="8,16,64")
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--programs", default="linus,orca,pyscf")
    ap.add_argument("--orca-reps-big", type=int, default=1, help="ORCA reps for ts113 and TZVP ts61")
    ap.add_argument("--skip", default="pyscf:ts113:def2-tzvp", help="comma list of program:mol:basis to skip")
    ap.add_argument("--out", default=str(ROOT / "bench/results/df_jk.json"))
    ap.add_argument("--slice", default="", help="run every command under ~/.local/bin/slice NAME (bench, work, ...)")
    ap.add_argument("--resume", action="store_true", help="skip runs already in the output file (ok and not contended)")
    a = ap.parse_args()
    SLICE[:] = ["systemd-run", "--user", "--scope", "-q", "-p", f"MemoryMax={MEM_CAP_GB}G", "-p", "MemorySwapMax=0"]
    if a.slice:
        SLICE.extend([SLICE_BIN, a.slice])
    global MONITOR
    if a.slice == "bench":
        MONITOR = set(BENCH_CPUS)
    mols, bases = a.mols.split(","), a.bases.split(",")
    threads = [int(t) for t in a.threads.split(",")]
    progs = a.programs.split(",")
    skip = set(a.skip.split(",")) if a.skip else set()
    out_path = pathlib.Path(a.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    runs = json.loads(out_path.read_text())["runs"] if out_path.exists() else []
    cpu = [l for l in open("/proc/cpuinfo") if l.startswith("model name")][0].split(":")[1].strip()
    meta = {"cpu": cpu, "date": datetime.date.today().isoformat(), "logical_cpus": os.cpu_count(),
            "git": subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT).decode().strip(),
            "linus_storage_default": "B in memory (jk), raw 3c in memory (j), split-RI-J (jdir)",
            "slice": a.slice or None, "mem_cap_gb": MEM_CAP_GB, "linus_budget_gb": LINUS_BUDGET_GB,
            "notes": "shared node: load average and `slice list` recorded per run; every job in a memory-capped systemd scope"}
    configs = [(m, b, t) for m in mols for b in bases for t in threads]
    for rep in range(a.reps):
        for i, (m, b, t) in enumerate(configs):
            order = progs[i % len(progs):] + progs[:i % len(progs)]
            for p in order:
                if f"{p}:{m}:{b}" in skip or f"{p}:{m}:*" in skip:
                    continue
                modes = {"linus": ["jk", "j", "jdir"], "orca": ["orca"], "pyscf": ["pyscf"], "scf-ri": ["ri"],
                         "linus-rsh": ["rsh"], "orca-rsh": ["orca"]}[p]
                if p == "orca" and rep >= a.orca_reps_big and (m == "ts113" or (m == "ts61" and b == "def2-tzvp")):
                    continue
                for mode in modes:
                    name = p if p != "linus" else f"linus-{mode}"
                    if a.resume and any(r.get("ok") and not r.get("contended") and not r.get("superseded") and (r["program"], r["mol"], r["basis"], r["threads"], r["rep"]) == (name, m, b, t, rep) for r in runs):
                        continue
                    for attempt in range(3):
                        gate = wait_quiet() if a.slice == "bench" else (None, 0.0)
                        load = uptime()
                        slices = slice_list()
                        try:
                            if p in ("linus", "linus-rsh"):
                                r = run_linus(m, b, t, mode)
                            elif p == "orca-rsh":
                                r = run_orca(m, b, t, rep, "wB97X-D3BJ")
                            elif p.startswith("scf-"):
                                r = run_scf(m, b, t, mode)
                            elif p == "orca":
                                r = run_orca(m, b, t, rep)
                            else:
                                r = run_pyscf(m, b, t)
                            r["ok"] = True
                        except Exception as e:  # keep going; record the failure
                            r = {"ok": False, "error": str(e)[:500]}
                        contended = r.get("foreign_threads", 0.0) >= 2.0
                        r.update({"program": p if p != "linus" else f"linus-{mode}", "mol": m, "basis": b, "threads": t, "rep": rep, "load_before": load,
                                  "slice": a.slice or None, "slice_list_before": slices, "ccd0_busy_before": gate[0], "waited_s": gate[1],
                                  "contended": contended, "attempt": attempt})
                        if not (r["ok"] and contended):
                            break
                        runs.append(dict(r, superseded=True))
                    runs.append(r)
                    print(json.dumps(r), flush=True)
                    summarize_and_write(out_path, meta, runs)


def summarize_and_write(path, meta, runs):
    # contended runs (another process on the bench slice right after the run) are kept in
    # `runs` but left out of the medians when an uncontended run of the configuration exists
    ok = [r for r in runs if r.get("ok") and not r.get("superseded")]
    clean_keys = {(r["program"], r["mol"], r["basis"], r["threads"]) for r in ok if not r.get("contended")}
    runs_for_median = [r for r in ok if not r.get("contended") or (r["program"], r["mol"], r["basis"], r["threads"]) not in clean_keys]
    keys = sorted({(r["program"], r["mol"], r["basis"], r["threads"]) for r in ok})
    summary = []
    for k in keys:
        rs = [r for r in runs_for_median if (r["program"], r["mol"], r["basis"], r["threads"]) == k]
        fields = sorted({f for r in rs for f, v in r.items() if isinstance(v, (int, float)) and not isinstance(v, bool) and f not in ("threads", "rep", "waited_s", "ccd0_busy_before", "ccd0_busy_after")})
        summary.append({"program": k[0], "mol": k[1], "basis": k[2], "threads": k[3], "n": len(rs), **{f: median([r.get(f) for r in rs]) for f in fields},
                        "loads": [r["load_before"] for r in rs], "contended_runs": sum(1 for r in rs if r.get("contended"))})
    path.write_text(json.dumps({"meta": meta, "summary": summary, "runs": runs}, indent=1))


if __name__ == "__main__":
    main()
