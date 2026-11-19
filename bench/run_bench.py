"""Benchmark linus against ORCA 6.0.0, Psi4 1.11 and PySCF 2.15: fair settings, accuracy and
provenance recorded with every timing.

  .refenv/bin/python bench/run_bench.py run --tag TAG [--mols la18,u1an39] [--levels b3lyp-svp,...]
      [--calcs energy,gradient,hessian] [--programs linus,orca,pyscf,pyscf-fastblas,psi4,psi4-openblas]
      [--layouts pinned[,unpinned,spread,scale16,...]] [--reps 3] [--warmup 1] [--mem-gb 20]
  .refenv/bin/python bench/run_bench.py refs --tag TAG --mols ... --levels ... --calcs ... [--layout slice:ccd2]
  .refenv/bin/python bench/run_bench.py blas --tag TAG          # BLAS override validation (untimed)
  .refenv/bin/python bench/run_bench.py plan ...                # print the run matrix, skips and memory settings

Levels (bench/README.md has the table): B3LYP and PBE at def2-SVP/def2-TZVP, HF with exact
integrals at def2-SVP, wB97X-D3BJ at def2-SVP and def2-TZVP, and the optional wB97M-V/def2-TZVP
(linus and ORCA only). wB97X-D3BJ is wB97X-V's semilocal part without VV10 plus two-body D3(BJ)
(Najibi and Goerigk 2018) in every program: PySCF `wb97x-d3bj` (= wb97x-v with nlc off + d3bj)
and Psi4 `wb97x-d3bj` need the dispersion overlay (.refenv-disp, LINUS_REFENV_DISP; skipped
without it). Approaches per program: RI-JK (`rijk`:
def2-universal-jkfit = ORCA def2/JK), RIJ+COSX (`rijcosx`: linus rij-snk with ORCA-like COSX
grids, ORCA RIJCOSX def2/J), RI-J for pure functionals (`rij`: def2-universal-jfit = def2/J)
and exact integrals (`exact`, integral-direct everywhere). Every DFT run of linus uses
`--grid` (default orca:2: ORCA 6.0.0's DefGrid2, measured; before 2026-11-07 a guessed ORCA-like
grid), ORCA DefGrid2, PySCF grids level 3,
Psi4 its default grid. Implicit solvent and open shells are not in this matrix (no solvent option,
closed shells only); their campaign results are in bench/README.md.

Program variants: `linus`, `orca`, `pyscf` and `psi4` as installed, `pyscf-fastblas` and
`psi4-openblas` with the conda OpenBLAS 0.3.34 (Cooperlake kernels) put in front of the
bundled OpenBLAS 0.3.3 / MKL by LD_PRELOAD, and `pyscf-haswell` (OPENBLAS_CORETYPE=Haswell).
linus variants by grid: `linus` (--linus-grid, orca:2), `linus-lg2` (linus:2), `linus-pg3`
(pyscf:3) and `linus-auto` (linus's own defaults: grid auto = linus:3 for Hessians with the CPKS
kernel on linus:1, pyscf:3 otherwise; RI-JK aux auto = cc-pVTZ-jkfit|def2-universal-jkfit for
Hessians): the Hessian rows, where orca:2 is too coarse.
Each run records which library the program's dgemm_ references bind to, its version and
the kernel it picked (bench/blas_probe.py).

Layouts: pinned (`slice bench`, CCD0's 8 physical cores), unpinned (8 threads, OS-scheduled on CPUs 0-125),
spread (one core per CCD: cpus 0,8,...,56), scale16/scale32/scale64 (physical cores 0..N-1),
or for development `slice:NAME` / `cpus:LIST`. Layouts that touch CCDs 4-7 (unpinned, spread,
scale64) need --allow-all-ccds. ORCA runs N MPI ranks with one BLAS thread each; linus, PySCF
and Psi4 run N threads; OMP/MKL/OPENBLAS/RAYON thread variables are set per program.

Timing: one unrecorded warm-up per program and case (ORCA's first start reads ~3.5 GB), then
--reps repetitions in rotating program order; the wall time is measured inside the job's own
memory-capped systemd scope (bench/jobs/measure.py) while this script holds the shared bench
lock (flock; waiting for it is not timed). Per run: start/end time, load average, busy fraction
of the layout's CPUs just before the run, `slice list` before and after, peak memory (scope
memory.peak and the largest process's max RSS), CPU time, exact command, environment
overrides and input. No host names are written. Results: bench/results/TAG/LAYOUT/CASE.json;
summarize with bench/summarize_bench.py.
"""
import argparse, datetime, fcntl, hashlib, json, math, os, pathlib, platform, re, shutil, statistics, subprocess, sys, tempfile, time, uuid

ROOT = pathlib.Path(__file__).resolve().parents[1]
BENCH = ROOT / "bench"
JOBS = BENCH / "jobs"
# main checkout, bench lock, reference environment, dispersion overlay, ORCA and scratch: one
# shared default each, overridable by environment variables (bench/bench_paths.py)
from bench_paths import DISP_OVERLAY, LOCK, MAIN, ORCA, REFENV, SCRATCH  # noqa: E402,F401
import refstore  # noqa: E402 - external programs' results, keyed (bench/refstore.py)
REFPY = REFENV / "bin/python"
SLICE = os.path.expanduser("~/.local/bin/slice")
CONDA_OPENBLAS = os.path.realpath(REFENV / "lib/libopenblas.so.0")
SYS_PY = shutil.which("python3") or sys.executable  # runs measure.py (no BLAS of its own)
BOHR = 0.529177210544  # CODATA 2022, Å per bohr
CHARGES = {"u1an39": -1}

LEVELS = {
    "b3lyp-svp": dict(kind="hybrid", basis="def2-svp", linus="b3lyp", orca="B3LYP/G", pyscf="b3lyp", psi4="b3lyp"),
    "b3lyp-tzvp": dict(kind="hybrid", basis="def2-tzvp", linus="b3lyp", orca="B3LYP/G", pyscf="b3lyp", psi4="b3lyp"),
    "pbe-svp": dict(kind="pure", basis="def2-svp", linus="pbe", orca="PBE", pyscf="pbe", psi4="pbe"),
    "pbe-tzvp": dict(kind="pure", basis="def2-tzvp", linus="pbe", orca="PBE", pyscf="pbe", psi4="pbe"),
    "hf-svp-exact": dict(kind="hf", basis="def2-svp", linus="hf", orca="HF", pyscf="hf", psi4="hf"),
    # wB97X-V's semilocal part, VV10 off, + two-body D3(BJ) in every program. PySCF parses
    # `wb97x-d3bj` into wb97x-v, nlc off and d3bj (pyscf-dispersion); Psi4's `wb97x-d3bj` is
    # HYB_GGA_XC_WB97X_V with nlc off and d3bj2b (dftd3): both from the dispersion overlay
    # (DISP_OVERLAY; before 2026-11-08 they ran plain wB97X without dispersion, another functional).
    # The SVP level pairs linus RI-JK Hessians with ORCA's analytic RIJCOSX Freq.
    "wb97x-d3bj-svp": dict(kind="rsh", basis="def2-svp", linus="wb97x-d3bj", orca="wB97X-D3BJ", pyscf="wb97x-d3bj",
                           psi4="wb97x-d3bj", disp_overlay=("pyscf", "psi4")),
    "wb97x-d3bj-tzvp": dict(kind="rsh", basis="def2-tzvp", linus="wb97x-d3bj", orca="wB97X-D3BJ", pyscf="wb97x-d3bj",
                            psi4="wb97x-d3bj", disp_overlay=("pyscf", "psi4")),
    # optional (not in quiet_node_run.sh's default levels): VV10, linus and ORCA only. ORCA evaluates
    # VV10 after the SCF for energies (its default) and self-consistently (SCNL) with EnGrad; linus
    # always self-consistently, on the NL grid that matches --linus-grid (orca:N -> ORCA's NL grid).
    "wb97m-v-tzvp": dict(kind="rsh", basis="def2-tzvp", linus="wb97m-v", orca="wB97M-V", pyscf=None, psi4=None,
                         programs=("linus", "orca"), vv10=True),
}
ORCA_BASIS = {"def2-svp": "def2-SVP", "def2-tzvp": "def2-TZVP"}
APPROACHES = {"hybrid": ["rijk", "rijcosx"], "rsh": ["rijk", "rijcosx"], "pure": ["rij"], "hf": ["exact"]}
AUX = {"rijk": "def2-universal-jkfit", "rij": "def2-universal-jfit", "rijcosx": "def2-universal-jfit"}
# ORCA 6.0.0 PBE uses rounded mu/beta; these are libxc's (reference/README.md quirk 2)
ORCA_PBE_LIBXC = "%method XMuePBE 0.2195149727645171 CBetaPBE 0.06672455060314922 end"

# The fair-BLAS bundle (bench/setup_fastblas.sh): .refenv's LLVM libomp, then conda-forge's OpenMP
# build of OpenBLAS 0.3.34 (one thread inside the program's OpenMP regions, N outside).
FASTBLAS_DIR = pathlib.Path(os.environ.get("BENCH_FASTBLAS", SCRATCH / "fastblas"))
FASTBLAS = f"{REFENV}/lib/libomp.so:{FASTBLAS_DIR}/libopenblas-omp-0.3.34.so"

VARIANTS = {
    "linus": dict(program="linus", blas="faer (pure Rust; no external BLAS)"),
    # linus on its own grid linus:2 (DefGrid2-level accuracy, 238k vs 198k points on the linus-grid
    # test set). orca:2 is ORCA's own DefGrid2 since 2026-11-07 (same grid, same error as ORCA);
    # the guessed ORCA-like grid used before lost accuracy on heavy atoms (crates/linus-grid/README.md)
    "linus-lg2": dict(program="linus", blas="faer (pure Rust; no external BLAS)", grid="linus:2"),
    # linus on PySCF's level-3 grid (reproduced point for point): finer than orca:2, and the same grid
    # as the PySCF runs; for Hessians orca:2 is too coarse (lowest modes off by >1 cm^-1 even with grid
    # response), so the summary pairs programs with the linus grid of matching accuracy
    "linus-pg3": dict(program="linus", blas="faer (pure Rust; no external BLAS)", grid="pyscf:3"),
    # linus at its own defaults, for the Hessian rows: grid auto (Hessians: linus:3, harmonic
    # frequencies within 0.15 cm^-1 of the converged grid, CPKS kernel on linus:1; energies and
    # gradients: pyscf:3) and RI-JK aux auto (Hessians: cc-pVTZ-jkfit|def2-universal-jkfit, RI
    # error 0.2 cm^-1 instead of 0.7; bench/results/2026-11-07-hess-accuracy)
    "linus-auto": dict(program="linus", blas="faer (pure Rust; no external BLAS)", grid="auto", aux_rijk="auto"),
    "orca": dict(program="orca", blas="OpenBLAS 0.3.27, static in each ORCA executable"),
    "pyscf": dict(program="pyscf", blas="as installed: wheel's OpenBLAS 0.3.3 in the C libraries, conda OpenBLAS 0.3.34 in numpy/scipy"),
    "pyscf-fastblas": dict(program="pyscf", blas="fair: conda-forge OpenBLAS 0.3.34 OpenMP build for every BLAS call (LD_PRELOAD bundle)",
                           setenv={"LD_PRELOAD": FASTBLAS}, needs=FASTBLAS_DIR / "libopenblas-omp-0.3.34.so"),
    "psi4": dict(program="psi4", blas="as installed: MKL 2025.3 (libmkl_rt)"),
    "psi4-openblas": dict(program="psi4", blas="fair: conda-forge OpenBLAS 0.3.34 OpenMP build interposed over MKL (LD_PRELOAD bundle)",
                          setenv={"LD_PRELOAD": FASTBLAS}, needs=FASTBLAS_DIR / "libopenblas-omp-0.3.34.so"),
    # tried and rejected (kept for the record; see bench/README.md): .refenv's pthreads OpenBLAS
    # preloaded (8 BLAS threads inside OpenMP regions: slower; 1 thread: numpy single-threaded),
    # and OPENBLAS_CORETYPE=Haswell (PySCF segfaults in its first SCF on this node)
    "pyscf-pthreads1": dict(program="pyscf", blas="conda OpenBLAS 0.3.34 pthreads build preloaded, OPENBLAS_NUM_THREADS=1",
                            setenv={"LD_PRELOAD": CONDA_OPENBLAS, "OPENBLAS_NUM_THREADS": "1"}),
    "pyscf-pthreads": dict(program="pyscf", blas="conda OpenBLAS 0.3.34 pthreads build preloaded, N threads", setenv={"LD_PRELOAD": CONDA_OPENBLAS}),
    # OPENBLAS_CORETYPE=Haswell for the wheel's OpenBLAS 0.3.3 only: set by the job after NumPy/SciPy
    # (conda OpenBLAS 0.3.34) are loaded; set process-wide it makes 0.3.34 segfault under an 8-CPU mask
    "pyscf-haswell": dict(program="pyscf", blas="wheel's OpenBLAS 0.3.3 with OPENBLAS_CORETYPE=Haswell (set after NumPy/SciPy load)",
                          late_coretype="Haswell"),
    "psi4-pthreads": dict(program="psi4", blas="conda OpenBLAS 0.3.34 pthreads build preloaded, N threads", setenv={"LD_PRELOAD": CONDA_OPENBLAS}),
}
FAIR = {"pyscf": "pyscf-fastblas", "psi4": "psi4-openblas"}  # the "fair BLAS" column of the summary

LAYOUTS = {
    "pinned": dict(prefix=[SLICE, "bench"], cpus=list(range(8)), threads=8),
    # OS-scheduled over the node, minus hardware threads 126-127 (kept free for the node's own use)
    "unpinned": dict(prefix=["taskset", "-c", "0-125"], cpus=list(range(126)), threads=8, all_ccds=True),
    "spread": dict(prefix=["taskset", "-c", ",".join(str(8 * k) for k in range(8))], cpus=[8 * k for k in range(8)], threads=8, all_ccds=True),
    "scale16": dict(prefix=["taskset", "-c", "0-15"], cpus=list(range(16)), threads=16),
    "scale32": dict(prefix=["taskset", "-c", "0-31"], cpus=list(range(32)), threads=32),
    "scale64": dict(prefix=["taskset", "-c", "0-63"], cpus=list(range(64)), threads=64, all_ccds=True),
}


def split_layouts(s):
    """'pinned,cpus:0,8,16,scale16' -> ['pinned', 'cpus:0,8,16', 'scale16'] (CPU numbers and ranges
    after a cpus: layout belong to it)."""
    out = []
    for tok in s.split(","):
        if out and out[-1].startswith("cpus:") and re.fullmatch(r"\d+(-\d+)?", tok):
            out[-1] += "," + tok
        else:
            out.append(tok)
    return out


def parse_cpus(s):
    out = []
    for part in s.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b or a) + 1))
    return out


def layout(name):
    if name in LAYOUTS:
        return dict(LAYOUTS[name], name=name)
    if name.startswith("slice:"):
        sl = name[6:]
        k = int(sl[3:]) if sl.startswith("ccd") else {"bench": 0, "build": 1}.get(sl)
        cpus = list(range(8 * k, 8 * k + 8)) if k is not None else None
        return dict(name=name, prefix=[SLICE, sl], cpus=cpus, threads=8, all_ccds=sl == "work")
    if name.startswith("cpus:"):
        cpus = parse_cpus(name[5:])
        return dict(name=name, prefix=["taskset", "-c", name[5:]], cpus=cpus, threads=len(cpus), all_ccds=any(c >= 32 for c in cpus))
    raise SystemExit(f"unknown layout {name}")


# ---------------------------------------------------------------- provenance

def sh(cmd, **kw):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=120, **kw).stdout.strip()
    except Exception as e:  # noqa: BLE001 - provenance must not stop a run
        return f"unavailable: {e}"


def slice_list():
    return sh([SLICE, "list"]).splitlines() if os.path.exists(SLICE) else []


def loadavg():
    return [float(x) for x in open("/proc/loadavg").read().split()[:3]]


def cpu_busy(cpus, dt=0.5):
    """Mean busy fraction of the given logical CPUs over dt seconds (/proc/stat)."""
    def snap():
        out = {}
        for line in open("/proc/stat"):
            if line.startswith("cpu") and line[3].isdigit():
                f = line.split()
                v = [int(x) for x in f[1:]]
                out[int(f[0][3:])] = (sum(v), v[3] + v[4])
        return out
    a = snap()
    time.sleep(dt)
    b = snap()
    cpus = cpus if cpus is not None else sorted(a)
    fr = [1.0 - (b[c][1] - a[c][1]) / max(1, b[c][0] - a[c][0]) for c in cpus]
    return round(sum(fr) / len(fr), 3)


def git_head(repo):
    """Commit of a checkout or worktree, read from its .git files (no git process)."""
    g = pathlib.Path(repo) / ".git"
    if g.is_file():
        g = pathlib.Path(g.read_text().split("gitdir:", 1)[1].strip())
    head = (g / "HEAD").read_text().strip()
    if not head.startswith("ref:"):
        return head, None
    ref = head[4:].strip()
    common = pathlib.Path((g / "commondir").read_text().strip()) if (g / "commondir").exists() else None
    if common is not None and not common.is_absolute():
        common = (g / common).resolve()
    for base in [g] + ([common] if common else []):
        if (base / ref).exists():
            return (base / ref).read_text().strip(), ref
        pr = base / "packed-refs"
        if pr.exists():
            for line in pr.read_text().splitlines():
                if line.endswith(" " + ref):
                    return line.split()[0], ref
    return None, ref


def linus_info(binary):
    """Provenance of a linus binary. A build record next to it (<binary>.build.json, written by
    bench/build_linus.sh when the binary was built) is authoritative if its sha256 matches; otherwise
    the commit is read from the repository the binary sits in, at run time (provenance=git-now)."""
    b = pathlib.Path(binary).resolve()
    info = dict(binary=str(b), sha256=hashlib.sha256(b.read_bytes()).hexdigest() if b.exists() else None,
                mtime=datetime.datetime.fromtimestamp(b.stat().st_mtime).isoformat(timespec="seconds") if b.exists() else None)
    side = pathlib.Path(str(b) + ".build.json")
    if side.exists():
        rec = json.loads(side.read_text())
        if rec.get("sha256") == info["sha256"]:
            info.update({k: v for k, v in rec.items() if k not in ("binary", "sha256", "mtime")}, provenance="build record")
            return info
        info["build_record_mismatch"] = str(side)
    info["provenance"] = "git-now"
    repo = next((p for p in b.parents if (p / ".git").exists()), None)
    if repo is not None:
        info["repo"] = str(repo)
        info["commit"], info["ref"] = git_head(repo)
        if repo.resolve() == ROOT.resolve():
            # the binary is built from crates/ and Cargo.*: name the last commit that changed them
            src = ["--", "crates", "Cargo.toml", "Cargo.lock"]
            info["source_commit"] = sh(["git", "log", "-1", "--format=%H"] + src, cwd=ROOT)
            ct = sh(["git", "log", "-1", "--format=%ct"] + src, cwd=ROOT)
            info["source_commit_time"] = datetime.datetime.fromtimestamp(int(ct)).isoformat(timespec="seconds") if ct.isdigit() else ct
            info["source_dirty"] = bool(sh(["git", "status", "--porcelain", "--untracked-files=no"] + src, cwd=ROOT))
            if ct.isdigit() and b.exists():
                info["binary_newer_than_source"] = b.stat().st_mtime >= int(ct)
    lock = (repo or ROOT) / "Cargo.lock"
    if lock.exists():
        m = re.search(r'name = "faer"\nversion = "([^"]+)"', lock.read_text())
        info["faer"] = m.group(1) if m else None
    return info


_LINUS_IDENT = {}


def linus_ident(binary):
    """Short provenance of a linus binary (cached per path and mtime: hashing it costs ~0.2 s)."""
    st = os.stat(binary)
    key = (binary, st.st_mtime)
    if key not in _LINUS_IDENT:
        li = linus_info(binary)
        _LINUS_IDENT[key] = {k: li.get(k) for k in ("sha256", "mtime", "commit", "source_commit", "source_dirty", "binary_newer_than_source",
                                                    "provenance", "built_at")}
    return _LINUS_IDENT[key]


def static_versions():
    v = {}
    v["refenv_python"] = sh([str(REFPY), "-c", "import sys, pyscf, numpy, scipy; print(sys.version.split()[0], pyscf.__version__, numpy.__version__, scipy.__version__)"])
    with tempfile.TemporaryDirectory() as td:  # psi4 leaves timer.dat in the working directory
        v["psi4"] = sh([str(REFPY), "-c", "import psi4; print(psi4.__version__)"], cwd=td).splitlines()[-1:]
    v["openmpi"] = sh([str(REFENV / "bin/mpirun"), "--version"]).splitlines()[:1]
    v["orca_dir"] = str(ORCA)
    v["conda_openblas"] = os.path.basename(CONDA_OPENBLAS)
    return v


_STATIC, _FAER = {}, {}


def environment(binary):
    if not _STATIC:
        _STATIC.update(static_versions())
    cpu = [ln.split(":", 1)[1].strip() for ln in open("/proc/cpuinfo") if ln.startswith("model name")]
    mem = {ln.split(":")[0]: int(ln.split()[1]) for ln in open("/proc/meminfo") if ln.split(":")[0] in ("MemTotal", "MemAvailable")}
    flags = next((ln.split(":", 1)[1].split() for ln in open("/proc/cpuinfo") if ln.startswith("flags")), [])
    return dict(time=datetime.datetime.now().astimezone().isoformat(timespec="seconds"), cpu_model=cpu[0] if cpu else None,
                logical_cpus=len(cpu), avx512f="avx512f" in flags, kernel=platform.release(), loadavg=loadavg(),
                mem_total_gb=round(mem["MemTotal"] / 2**20, 1), mem_available_gb=round(mem["MemAvailable"] / 2**20, 1),
                slice_list=slice_list(), linus=linus_info(binary), versions=dict(_STATIC))


# ---------------------------------------------------------------- molecules and inputs

def molecule(name):
    path = next(p for d in ("bench/molecules", "tests/molecules") if (p := ROOT / d / f"{name}.xyz").exists())
    lines = path.read_text().splitlines()
    n = int(lines[0])
    sym, xyz = [], []
    for ln in lines[2:2 + n]:
        f = ln.split()
        sym.append(f[0])
        xyz.append([float(x) / BOHR for x in f[1:4]])
    return dict(name=name, symbols=sym, coords=xyz, charge=CHARGES.get(name, 0), natoms=n, file=str(path.relative_to(ROOT)))


# basis and auxiliary-basis sizes (PySCF 2.15, spherical) and doubly occupied orbitals (def2 ECPs
# included: Pd carries 28 core electrons in its ECP)
NBF = {("la18", "def2-svp"): 180, ("la18", "def2-tzvp"): 358, ("u1an39", "def2-svp"): 483, ("u1an39", "def2-tzvp"): 1034,
       ("ts61", "def2-svp"): 666, ("ts61", "def2-tzvp"): 1343, ("ts113", "def2-svp"): 1106, ("ts113", "def2-tzvp"): 2155,
       ("pdcl2pme2ph2", "def2-svp"): 437, ("pdcl2pme2ph2", "def2-tzvp"): 816}
NAUX = {("la18", "def2-universal-jkfit"): 902, ("la18", "def2-universal-jfit"): 578, ("u1an39", "def2-universal-jkfit"): 2556,
        ("u1an39", "def2-universal-jfit"): 1645, ("ts61", "def2-universal-jkfit"): 3436, ("ts61", "def2-universal-jfit"): 2160,
        ("ts113", "def2-universal-jkfit"): 5520, ("ts113", "def2-universal-jfit"): 3492,
        ("pdcl2pme2ph2", "def2-universal-jkfit"): 2363, ("pdcl2pme2ph2", "def2-universal-jfit"): 1319}
NOCC = {"la18": 38, "u1an39": 120, "ts61": 153, "ts113": 233, "pdcl2pme2ph2": 100}


def df_memory_gb(prog, mol, lv, approach):
    """In-core 3-index DF tensor (GB) as Psi4's MemDFJK (nbf^2 naux) and PySCF (nbf(nbf+1)/2 naux)
    store it; x2 for range-separated hybrids (a second, attenuated tensor). None if unknown."""
    L = LEVELS[lv]
    n, q = NBF.get((mol, L["basis"])), NAUX.get((mol, AUX.get(approach, "")))
    if not n or not q or approach not in ("rijk", "rij"):
        return None
    pairs = n * n if prog == "psi4" else n * (n + 1) / 2
    return pairs * q * 8 / 1e9 * (2 if L["kind"] == "rsh" else 1)


def linus_df_grad_memory_gb(mol, lv, approach):
    """Memory (GB) linus's RI-JK gradient (crates/linus/src/two_electron/df/gradient.rs, also run inside Hessians)
    needs on top of the SCF's 3-index store: the packed MO integrals (P|ij) (and (P|erf|ij) for
    range-separated hybrids), naux x o(o+1)/2 doubles each, plus the block buffers (~1 GiB in
    pass 1; the pass-2 G block, nq x nbf^2 doubles, at most 3 GiB) and the rest of the process,
    taken as 0.5 + min(2.5, naux nbf^2 doubles) GiB. The store itself adapts to the job's memory
    limit (by default at most 75% of a cgroup cap: f64/f48/f40/f32 precision, one-sided long range),
    so it is not a fixed requirement. Measured peak RSS minus store minus MO integrals
    (bench/results/2026-11-07-df-grad and 2026-11-07-rik-scale; def2-TZVP, B3LYP and wB97X-D3BJ):
    la18 0.4-0.8 GiB, u1an39 1.4-1.6, ts61 1.7-2.8, ts113 1.6 GiB, so the estimate is 0.2-1.6 GiB
    high. Without exchange (pure functionals, RI-J only) the gradient is integral-direct: ~1 GiB."""
    L = LEVELS[lv]
    n, q, o = NBF.get((mol, L["basis"])), NAUX.get((mol, AUX.get(approach, ""))), NOCC.get(mol)
    if not n or not q or not o or approach not in ("rijk", "rij"):
        return None
    if L["kind"] not in ("hybrid", "rsh"):
        return 1.0 * 2**30 / 1e9
    mo = (2 if L["kind"] == "rsh" else 1) * q * o * (o + 1) / 2 * 8 / 2**30
    return (0.5 + min(2.5, q * n * n * 8 / 2**30) + mo) * 2**30 / 1e9


def skip_reason(variant, approach, lv, calc, mol=None, mem_gb=None, allow_disk=False, threads=8):
    prog = VARIANTS[variant]["program"]
    L = LEVELS[lv]
    kind = L["kind"]
    if approach not in APPROACHES[kind] and approach != "exact":
        return f"{approach} does not apply to {kind} methods"
    if "programs" in L and prog not in L["programs"]:
        return f"{lv}: {', '.join(L['programs'])} only"
    if prog in L.get("disp_overlay", ()) and not DISP_OVERLAY.is_dir():
        return f"{lv} needs the dispersion overlay {DISP_OVERLAY} for {prog}'s D3(BJ) (reference/validation/README.md)"
    if prog == "linus" and approach == "rijcosx" and calc == "hessian":
        return "linus has no COSX Hessians yet"
    if prog == "linus" and L.get("vv10") and calc == "hessian":
        return "linus has no VV10 Hessians yet"
    if prog in ("pyscf", "psi4") and approach == "rijcosx":
        return {"pyscf": "PySCF has no COSX (its sgX uses different grids)",
                "psi4": "Psi4 COSX not benchmarked (1.11 COSX/RI-J gradients don't match their energies, reference quirk 10)"}[prog]
    if prog == "psi4" and calc == "hessian" and kind != "hf":
        return "Psi4 1.11 DFT Hessians are finite differences of gradients (no analytic DFT Hessian)"
    if prog == "orca" and calc == "hessian" and approach == "rijk":
        return "ORCA Hessians run with the production RIJCOSX only"
    if prog == "orca" and kind == "rsh" and approach == "rijk" and calc != "energy":
        return "ORCA 6.0.0 refuses range-separated RIJK gradients ('the range sep RIJK gradient has an unresolved bug')"
    if mol and mem_gb and prog == "linus" and calc in ("gradient", "hessian"):
        # the SCF's 3-index store takes up to 75% of the cap; the gradient's extras must fit the rest
        need = linus_df_grad_memory_gb(mol, lv, approach)
        if need and need > 0.25 * mem_gb:
            return (f"linus RI-JK gradient needs ~{need:.1f} GB beyond the SCF's 3-index store, more than the quarter of the "
                    f"{mem_gb:g} GB job budget the store leaves free; raise --mem-gb")
    if mem_gb and prog == "orca" and mem_settings(mem_gb, threads)["orca_maxcore"] < 500:
        return (f"ORCA %maxcore would be {mem_settings(mem_gb, threads)['orca_maxcore']} MB per rank with {threads} ranks in "
                f"{mem_gb:g} GB (< 500 MB); raise --mem-gb")
    if mol and mem_gb and prog in ("pyscf", "psi4") and not allow_disk:
        need = df_memory_gb(prog, mol, lv, approach)
        m = mem_settings(mem_gb, 1)
        # PySCF keeps the lower triangle in core only below max_memory; Psi4 screens pairs (its tensor is
        # smaller than nbf^2 naux on large molecules), so skip only clear cases and flag disk runs (disk_df)
        have = (m["pyscf_max_memory"] if prog == "pyscf" else m["psi4_memory_mb"]) / 1024 * (1.3 if prog == "psi4" else 0.9)
        if need and need > have:
            return (f"{prog} DF tensor ~{need:.0f} GB does not fit the {mem_gb:g} GB job budget: it would run from disk "
                    f"(Psi4 DiskDFJK / PySCF out-of-core); raise --mem-gb or pass --allow-disk-df")
    return None


def entries(programs, lv, calc, approaches=None, mol=None, mem_gb=None, allow_disk=False, threads=8):
    kind = LEVELS[lv]["kind"]
    out = []
    for v in programs:
        for ap in APPROACHES[kind]:
            if approaches and ap not in approaches:
                continue
            out.append((v, ap, skip_reason(v, ap, lv, calc, mol, mem_gb, allow_disk, threads)))
    return out


def conv_settings(prog, calc, ref):
    """SCF convergence per program: TightSCF-like for energies; each program's own defaults by
    purpose for derivatives (linus's are the tightest); much tighter for references."""
    if prog == "linus":
        return {"scf": {"convergence": "verytight"}} if ref else {}
    if prog == "orca":
        return "VeryTightSCF" if ref else "TightSCF"
    if prog == "pyscf":
        if ref:
            return dict(conv_tol=1e-11, conv_tol_grad=1e-7)
        return dict(conv_tol=1e-8 if calc == "energy" else 1e-9)
    if prog == "psi4":
        if ref:
            return dict(e_convergence=1e-10, d_convergence=1e-9)
        return dict(e_convergence=1e-8, d_convergence=1e-6) if calc == "energy" else {}


def mem_settings(mem_gb, ranks):
    mb = int(mem_gb * 1024)
    return dict(orca_maxcore=min(4000, int(0.6 * mb / ranks)), pyscf_max_memory=int(0.75 * mb), psi4_memory_mb=int(0.7 * mb))


# ---------------------------------------------------------------- running one job

class Job:
    """Inputs and command of one run, built per program; parse() reads the results."""

    def __init__(self, a, variant, approach, mol, lv, calc, lay, ref=False, grid=None):
        self.a, self.variant, self.approach, self.mol, self.lv, self.calc, self.lay, self.ref = a, variant, approach, mol, lv, calc, lay, ref
        self.prog = VARIANTS[variant]["program"]
        self.L = LEVELS[lv]
        self.threads = lay["threads"]
        self.mem = mem_settings(a.mem_gb, self.threads)
        self.grid = grid or VARIANTS[variant].get("grid") or a.linus_grid
        self.env = base_env(self.prog, self.threads)
        self.setenv = dict(VARIANTS[variant].get("setenv", {}))
        self.setenv.update(getattr(a, "_extra_setenv", {}).get(variant, {}))
        self.stdin = "-"

    def prepare(self, wd):
        self.wd = wd
        getattr(self, "prep_" + self.prog)()
        return self

    # linus: JSON request on stdin, JSON result on stdout
    def prep_linus(self):
        m, L = self.mol, self.L
        req = dict(structure=dict(symbols=m["symbols"], geometry=m["coords"], charge=m["charge"], multiplicity=1),
                   method=L["linus"], basis=L["basis"], threads=self.threads, calctype=self.calc)
        if L["kind"] != "hf":
            req["grid"] = dict(xc=self.grid)
        ap = self.approach
        if ap == "rijk":
            req["jk"] = dict(kind="df", aux=VARIANTS[self.variant].get("aux_rijk") or AUX[ap])
        elif ap == "rij":
            req["jk"] = dict(kind="df", aux=AUX[ap])
        elif ap == "rijcosx":
            req["jk"] = dict(kind="rij-snk", aux=AUX[ap], k_grid="cosx:2")
        else:
            req["jk"] = "exact"
        if self.calc == "hessian" and L["kind"] != "hf":
            req["grid"]["response"] = True  # as ORCA's analytic Hessian; PySCF's DF Hessian is fixed-grid
        req.update(conv_settings("linus", self.calc, self.ref))
        (self.wd / "request.json").write_text(json.dumps(req))
        self.stdin = str(self.wd / "request.json")
        self.cmd = [str(self.a.linus_bin), "--json"]
        self.input = {**req, "structure": {k: v for k, v in req["structure"].items() if k not in ("symbols", "geometry")}}

    def parse_linus(self, out, err):
        d = json.loads(out)
        h = d["scf"]["history"]
        r = dict(energy=d["energy"], iterations=d["iterations"], converged=d["converged"], nbf=d["nbf"],
                 final_gmax=h[-1].get("gmax") if h else None, final_de=h[-1].get("de") if h else None,
                 scf_convergence=d["scf"].get("convergence"), incremental_builds=d["scf"].get("incremental_builds"),
                 timings=d.get("timings"), linus_version=d.get("version"), options=d.get("options"),
                 # since a0f30e4: timings.total is the whole call (timings.energy the SCF stage);
                 # sizes and peak memory are in resources; resolved = the settings actually used
                 resources=d.get("resources"), resolved=d.get("resolved"))
        # faer's GEMM (gemm crate) dispatches on the CPU at run time; on this node it runs AVX-512
        # kernels (142 GF/s per core, crates/linus-dev/examples/gemm_bench.rs); linus links no BLAS/LAPACK
        avx512 = "avx512f" in open("/proc/cpuinfo").read().split("flags", 1)[1].split("\n", 1)[0]
        if "faer" not in _FAER:
            lock = next((p / "Cargo.lock" for p in pathlib.Path(self.a.linus_bin).parents if (p / "Cargo.lock").exists()), None)
            m = re.search(r'name = "faer"\nversion = "([^"]+)"', lock.read_text()) if lock else None
            _FAER["faer"] = m.group(1) if m else None
        r["blas"] = [dict(kind="faer", version=_FAER["faer"],
                          core="AVX-512 (run-time dispatch)" if avx512 else "AVX2 (run-time dispatch)", parallel="rayon")]
        for k in ("gradient", "hessian", "cphf_iterations"):
            if k in d:
                r[k] = d[k]
        if "hessian" in d:
            n = int(round(len(d["hessian"]) ** 0.5))
            r["hessian"] = [d["hessian"][i * n:(i + 1) * n] for i in range(n)] if not isinstance(d["hessian"][0], list) else d["hessian"]
        return r

    # ORCA: input file, bohr coordinates
    def prep_orca(self):
        m, L, ap = self.mol, self.L, self.approach
        words = [L["orca"], ORCA_BASIS[L["basis"]]]
        words += {"rijk": ["def2/JK", "RIJK"], "rijcosx": ["def2/J", "RIJCOSX"], "rij": ["def2/J", "RI"], "exact": ["NoRI", "NoCOSX"]}[ap]
        if L["kind"] != "hf":
            words.append("DefGrid2")
        words.append(conv_settings("orca", self.calc, self.ref))
        words += {"energy": [], "gradient": ["EnGrad"], "hessian": ["Freq"]}[self.calc]
        words += ["Bohrs", "NoAutoStart"]
        lines = ["! " + " ".join(words), f"%pal nprocs {self.threads} end", f"%maxcore {self.mem['orca_maxcore']}"]
        if L["orca"] == "PBE":
            lines.append(ORCA_PBE_LIBXC)
        lines.append(f"* xyz {m['charge']} 1")
        lines += [f"{s} {c[0]:.12f} {c[1]:.12f} {c[2]:.12f}" for s, c in zip(m["symbols"], m["coords"])]
        lines += ["*", ""]
        text = "\n".join(lines)
        (self.wd / "orca.inp").write_text(text)
        self.cmd = [str(ORCA / "orca"), "orca.inp"]
        self.input = text.split("* xyz")[0]
        self.setenv["OPENBLAS_VERBOSE"] = "2"  # each ORCA process reports the OpenBLAS kernel it picked

    def parse_orca(self, out, err):
        r = parse_orca_output(out)
        cores = re.findall(r"^Core:\s*(\S+)", out + "\n" + err, re.M)
        r["blas"] = [dict(kind="openblas", version="0.3.27 (static)", core=sorted(set(cores)), processes_reporting=len(cores))]
        if not r.get("terminated_normally"):
            raise RuntimeError("ORCA did not terminate normally: " + out[-1500:])
        eg = self.wd / "orca.engrad"
        if self.calc == "gradient" and eg.exists():
            data = [x.strip() for x in eg.read_text().splitlines() if x.strip() and not x.startswith("#")]
            n = int(data[0])
            g = [float(x) for x in data[2:2 + 3 * n]]
            r["gradient"] = [g[3 * i:3 * i + 3] for i in range(n)]
        hs = self.wd / "orca.hess"
        if self.calc == "hessian" and hs.exists():
            r["hessian"] = parse_orca_hess(hs.read_text())
        return r

    # PySCF: bench/jobs/pyscf_job.py
    def prep_pyscf(self):
        m, L, ap = self.mol, self.L, self.approach
        job = dict(symbols=m["symbols"], coords=m["coords"], charge=m["charge"], xc=L["pyscf"], basis=L["basis"],
                   jk="exact" if ap == "exact" else "df", aux=AUX.get(ap), grid=3, calc=self.calc, max_memory=self.mem["pyscf_max_memory"],
                   threads=self.threads, grid_response=True, auxbasis_response=2)
        if VARIANTS[self.variant].get("late_coretype"):
            job["late_openblas_coretype"] = VARIANTS[self.variant]["late_coretype"]
        if "pyscf" in L.get("disp_overlay", ()):
            job["disp_overlay"] = str(DISP_OVERLAY)  # pyscf-dispersion for the -D3BJ name
        job.update(conv_settings("pyscf", self.calc, self.ref))
        (self.wd / "in.json").write_text(json.dumps(job))
        self.cmd = [str(REFPY), str(JOBS / "pyscf_job.py"), "in.json", "out.json"]
        self.input = {k: v for k, v in job.items() if k not in ("symbols", "coords")}

    def parse_pyscf(self, out, err):
        d = json.loads((self.wd / "out.json").read_text())
        if d.get("error"):
            raise RuntimeError(d["error"])
        return d

    # Psi4: bench/jobs/psi4_job.py (import psi4 in the env's python)
    def prep_psi4(self):
        m, L, ap = self.mol, self.L, self.approach
        job = dict(symbols=m["symbols"], coords=m["coords"], charge=m["charge"], method=L["psi4"], basis=L["basis"],
                   scf_type="direct" if ap == "exact" else "df", df_basis_scf=AUX.get(ap), calc=self.calc,
                   memory_mb=self.mem["psi4_memory_mb"], threads=self.threads, options=conv_settings("psi4", self.calc, self.ref))
        if "psi4" in L.get("disp_overlay", ()):
            job["disp_overlay"] = str(DISP_OVERLAY)  # dftd3 for the -D3BJ name
        (self.wd / "in.json").write_text(json.dumps(job))
        self.cmd = [str(REFPY), str(JOBS / "psi4_job.py"), "in.json", "out.json"]
        self.input = {k: v for k, v in job.items() if k not in ("symbols", "coords")}

    def parse_psi4(self, out, err):
        d = json.loads((self.wd / "out.json").read_text())
        log = (self.wd / "psi4.out").read_text() if (self.wd / "psi4.out").exists() else ""
        it = re.findall(r"@\S*(?:RHF|RKS) iter\s+(\S+):\s+(-?\d+\.\d+)\s+(\S+)\s+(\S+)", log)
        if it:
            d["final_de"], d["final_orbital_gradient_rms"] = float(it[-1][2]), float(it[-1][3])
        d["jk_algorithm"] = sorted(set(re.findall(r"^\s*(?:Algorithm|Mode):\s*(.+)$", log, re.M)))
        d["jk_classes"] = sorted(set(re.findall(r"=>\s*(\w*JK\w*)\s*:", log)))
        d["disk_df"] = "DiskDFJK" in log or bool(re.search(r"Algorithm:\s*Disk", log))
        d["fd_hessian"] = "finite-differences" in log.lower()
        return d

    def parse(self, out, err):
        return getattr(self, "parse_" + self.prog)(out, err)


def base_env(prog, threads):
    """Thread variables set explicitly per program; BLAS overrides only via the variant."""
    env = {k: v for k, v in os.environ.items() if k not in ("LD_PRELOAD", "OPENBLAS_CORETYPE", "MKL_ENABLE_INSTRUCTIONS", "MKL_DEBUG_CPU_TYPE",
                                                             "OMP_PROC_BIND", "OMP_PLACES", "KMP_AFFINITY", "GOMP_CPU_AFFINITY")}
    n = str(threads)
    if prog == "linus":
        env.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", RAYON_NUM_THREADS=n)
    elif prog == "orca":  # N MPI ranks, one BLAS thread each
        env.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
                   PATH=f"{ORCA}:{REFENV}/bin:" + os.environ.get("PATH", ""), LD_LIBRARY_PATH=f"{ORCA}/lib:{REFENV}/lib",
                   OMPI_MCA_btl="self,vader", OMPI_MCA_btl_vader_single_copy_mechanism="none", OMPI_MCA_hwloc_base_binding_policy="none",
                   OMPI_MCA_rmaps_base_oversubscribe="1")
    else:  # PySCF, Psi4: N OpenMP threads; numpy/scipy (and Psi4's) BLAS N threads outside parallel regions
        env.update(OMP_NUM_THREADS=n, OPENBLAS_NUM_THREADS=n, MKL_NUM_THREADS=n)
    return env


def parse_orca_output(out):
    r = dict(terminated_normally="ORCA TERMINATED NORMALLY" in out)
    e = re.findall(r"FINAL SINGLE POINT ENERGY\s+(-?\d+\.\d+)", out)
    r["energy"] = float(e[-1]) if e else None
    m = re.findall(r"SCF CONVERGED AFTER\s+(\d+)\s+CYCLES", out)
    r["iterations"] = int(m[-1]) if m else None
    for key, pat in [("final_de", r"Last Energy change\s+\.\.\.\s+(\S+)"), ("final_orbital_gradient", r"Last Orbital Gradient\s+\.\.\.\s+(\S+)"),
                     ("final_diis_error", r"Last DIIS Error\s+\.\.\.\s+(\S+)"), ("final_max_density_change", r"Last MAX-Density change\s+\.\.\.\s+(\S+)"),
                     ("final_rms_density_change", r"Last RMS-Density change\s+\.\.\.\s+(\S+)")]:
        m = re.findall(pat, out)
        try:
            r[key] = float(m[-1]) if m else None
        except ValueError:
            r[key] = None
    m = re.search(r"Program Version\s+(\S+)", out)
    r["orca_version"] = m.group(1) if m else None
    m = re.search(r"Number of basis functions\s+\.\.\.\s+(\d+)", out)
    r["nbf"] = int(m.group(1)) if m else None
    m = re.search(r"TOTAL RUN TIME:\s+(\d+) days (\d+) hours (\d+) minutes (\d+) seconds (\d+) msec", out)
    if m:
        d, h, mi, s, ms = map(int, m.groups())
        r["orca_total_run_time"] = ((d * 24 + h) * 60 + mi) * 60 + s + ms / 1000
    r["cpscf_iterations"] = len(re.findall(r"^\s*CP-SCF ITERATION\s+\d+", out, re.M)) or None
    return r


def parse_orca_hess(text):
    lines = text.split("$hessian", 1)[1].splitlines()[1:]
    n = int(lines[0].split()[0])
    h = [[0.0] * n for _ in range(n)]
    i = 1
    while i < len(lines):
        cols = [int(x) for x in lines[i].split()]
        if not cols:
            break
        for r in range(n):
            f = lines[i + 1 + r].split()
            for k, c in enumerate(cols):
                h[int(f[0])][c] = float(f[1 + k])
        i += 1 + n
        if max(cols) == n - 1:
            break
    return h


def scrub_host(text):
    import socket
    names = {socket.gethostname(), socket.getfqdn(), platform.node()} - {"", "localhost"}
    for n in sorted(names, key=len, reverse=True):
        text = text.replace(n, "<host>")
        text = text.replace(n.split(".")[0], "<host>")
    return text


def run_job(job, a, timed=True):
    """Run one job in its own memory-capped scope under the layout's CPU mask; returns a record."""
    SCRATCH.mkdir(parents=True, exist_ok=True)
    wd = pathlib.Path(tempfile.mkdtemp(prefix=f"{job.prog}_", dir=SCRATCH))
    rec = dict(variant=job.variant, approach=job.approach, program=job.prog)
    try:
        try:
            job.prepare(wd)
        except Exception as e:  # noqa: BLE001 - e.g. a missing basis or molecule
            rec.update(status="failed", ok=False, error=f"prepare: {type(e).__name__}: {e}")
            return rec
        unit = f"bench-{uuid.uuid4().hex[:12]}"
        scope = ["systemd-run", "--user", "--scope", "-q", f"--unit={unit}", "-p", f"MemoryMax={a.mem_gb}G", "-p", "MemorySwapMax=0",
                 "-p", "OOMPolicy=continue"]  # an OOM kill takes only the program; measure.py records it
        mcmd = [SYS_PY, str(JOBS / "measure.py"), str(wd / "measure.json"), job.stdin, str(wd / "stdout.txt"), str(wd / "stderr.txt")]
        for k, v in job.setenv.items():
            mcmd += ["--setenv", f"{k}={v}"]
        cmd = scope + job.lay["prefix"] + mcmd + ["--"] + job.cmd
        if job.prog == "linus":  # the exact binary and the commit it was built from, in every linus run
            rec["linus_build"] = linus_ident(a.linus_bin)
        rec.update(command=job.cmd, launch=scope + job.lay["prefix"], setenv=job.setenv,
                   thread_env={k: job.env[k] for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "RAYON_NUM_THREADS") if k in job.env},
                   input=job.input, threads=job.threads, memory_cap_gb=a.mem_gb, mem_settings=job.mem)
        lockf = open(LOCK, "a") if (timed and not a.no_lock) else None
        t_wait = time.perf_counter()
        if lockf:
            fcntl.flock(lockf, fcntl.LOCK_EX)
        try:
            rec["lock_wait_s"] = round(time.perf_counter() - t_wait, 3)
            rec["slice_list_before"] = slice_list()
            rec["loadavg_before"] = loadavg()
            rec["cpu_busy_before"] = cpu_busy(job.lay["cpus"])
            rec["start"] = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
            t0 = time.perf_counter()
            _RUNNING[0] = unit  # stopped by the signal handler if this script is terminated
            try:
                p = subprocess.run(cmd, cwd=wd, env=job.env, capture_output=True, text=True, timeout=a.timeout)
                rc = p.returncode
            except subprocess.TimeoutExpired:
                subprocess.run(["systemctl", "--user", "kill", "--signal=KILL", f"{unit}.scope"], capture_output=True)
                rc = "timeout"
            finally:
                _RUNNING[0] = None
            rec["wall_outer"] = time.perf_counter() - t0
            rec["end"] = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
            rec["loadavg_after"] = loadavg()
            rec["slice_list_after"] = slice_list()
        finally:
            if lockf:
                fcntl.flock(lockf, fcntl.LOCK_UN)
                lockf.close()
        meas = json.loads((wd / "measure.json").read_text()) if (wd / "measure.json").exists() else {}
        rec["measure"] = meas
        rec["wall"] = meas.get("wall")
        out = (wd / "stdout.txt").read_text(errors="replace") if (wd / "stdout.txt").exists() else ""
        err = (wd / "stderr.txt").read_text(errors="replace") if (wd / "stderr.txt").exists() else ""
        if rc == "timeout":
            rec.update(status="timeout", ok=False, error=f"killed after {a.timeout} s")
        elif meas.get("oom_kills"):
            rec.update(status="oom", ok=False, error=f"out of memory in its {a.mem_gb} GB scope")
        elif meas.get("returncode", rc) != 0:
            msg = (err or out)[-2000:] if meas else f"launch failed (rc {rc}; -15: systemd stopped the scope, e.g. after an OOM kill): {p.stderr[-1500:]}"
            if job.prog == "linus" and meas.get("returncode") in (1, 2) and re.search(r"not implemented|not supported|unknown|need", msg):
                rec.update(status="rejected", ok=False, error=msg.strip())
            else:
                rec.update(status="failed", ok=False, error=msg)
        else:
            try:
                rec.update(job.parse(out, err))
                rec.update(status="ok", ok=True)
            except Exception as e:  # noqa: BLE001
                rec.update(status="failed", ok=False, error=f"{type(e).__name__}: {e}"[-2000:])
        if rec.get("error"):  # MPI and other error text can name the host: never record it
            rec["error"] = scrub_host(rec["error"])
        if not rec["ok"] and a.keep_failed:
            rec["workdir"] = str(wd)
    finally:
        if getattr(job, "keep_outputs", False):  # a reference for the store (bench/refstore.py): its raw outputs
            job.outputs = refstore.read_outputs(wd)
        if rec.get("ok") or not a.keep_failed:
            shutil.rmtree(wd, ignore_errors=True)
    return rec


# ---------------------------------------------------------------- comparing repetitions

def max_abs_diff(x, y):
    if x is None or y is None:
        return None
    fx, fy = flatten(x), flatten(y)
    return max(abs(p - q) for p, q in zip(fx, fy)) if len(fx) == len(fy) else None


def flatten(x):
    return [v for row in x for v in (flatten(row) if isinstance(row, list) else [row])] if isinstance(x, list) else [x]


def summarize_entry(runs):
    ok = [r for r in runs if r.get("ok") and not r.get("warmup")]
    s = dict(n_ok=len(ok), n_runs=len([r for r in runs if not r.get("warmup")]))
    if not ok:
        st = [r.get("status") for r in runs]
        s["status"] = "rejected" if "rejected" in st else ("oom" if "oom" in st else ("timeout" if "timeout" in st else "failed"))
        s["error"] = next((r.get("error") for r in runs if r.get("error")), None)
        return s
    w = sorted(r["wall"] for r in ok)
    first = ok[0]
    s.update(status="ok", median=statistics.median(w), min=w[0], max=w[-1], spread=w[-1] / w[0] if w[0] else None,
             energy=first.get("energy"), iterations=first.get("iterations"),
             peak_memory_gb=max((r["measure"].get("cgroup_memory_peak_bytes") or 0) for r in ok) / 2**30,
             max_rss_gb=max((r["measure"].get("max_rss_largest_process_kb") or 0) for r in ok) / 2**20,
             cpu_utilization=statistics.median([r["measure"].get("cpu_utilization") or 0 for r in ok]),
             cpu_busy_before_max=max(r.get("cpu_busy_before") or 0 for r in ok),
             energy_spread=max(abs(r["energy"] - first["energy"]) for r in ok) if first.get("energy") is not None else None)
    for k in ("final_gmax", "final_norm_gorb", "final_orbital_gradient", "final_orbital_gradient_rms", "final_de", "nbf", "disk_df",
              "df_incore", "cphf_iterations", "jk_algorithm"):
        if first.get(k) is not None:
            s[k] = first[k]
    for k in ("gradient", "hessian"):
        if first.get(k) is not None:
            s[k + "_rep_spread"] = max([r.get(k + "_diff_vs_first") or 0 for r in ok[1:]] + [0])
    s["blas"] = first.get("blas")
    s["blas_binding"] = first.get("blas_binding")
    s["openmp_binding"] = first.get("openmp_binding")
    return s


def strip_bulky(rec, first):
    """Keep gradients/Hessians only in the first successful repetition; later ones keep their
    max abs difference from it (a determinism check)."""
    if first is not None:
        for k in ("gradient", "hessian"):
            if k in rec:
                rec[k + "_diff_vs_first"] = max_abs_diff(first.get(k), rec.pop(k))
    return rec


# ---------------------------------------------------------------- commands

def cmd_run(a):
    for lname in split_layouts(a.layouts):
        lay = layout(lname)
        if lay.get("all_ccds") and not a.allow_all_ccds:
            print(f"layout {lname} uses CCDs 4-7 (another team's): pass --allow-all-ccds once they are free; skipped", flush=True)
            continue
        # portable directory name: no ":" or "," (they break Windows checkouts)
        outdir = BENCH / "results" / a.tag / lname.replace(":", "-").replace(",", "_")
        outdir.mkdir(parents=True, exist_ok=True)
        for mname in a.mols.split(","):
            mol = molecule(mname)
            for lv in a.levels.split(","):
                for calc in a.calcs.split(","):
                    run_case(a, lay, outdir, mol, lv, calc)


def run_case(a, lay, outdir, mol, lv, calc):
    case = f"{mol['name']}_{lv}_{calc}"
    fn = outdir / f"{case}.json"
    old = json.loads(fn.read_text()) if fn.exists() else None
    ents = entries(a.programs.split(","), lv, calc, a.approaches.split(",") if a.approaches else None, mol["name"], a.mem_gb, a.allow_disk_df, lay["threads"])
    todo = []
    rec = old or dict(case=case, mol=mol["name"], molecule_file=mol["file"], natoms=mol["natoms"], charge=mol["charge"], level=lv,
                      level_def=LEVELS[lv], calc=calc, layout=lay["name"], threads=lay["threads"], cpus=lay["cpus"], tag=a.tag,
                      linus_grid=a.linus_grid, entries={}, history=[])
    for v, ap, why in ents:
        key = f"{v}:{ap}"
        done = rec["entries"].get(key, {}).get("summary", {})
        if a.resume and done.get("status") == "ok" and done.get("n_ok", 0) >= a.reps:  # skips and rejections are re-checked
            continue
        if why:
            rec["entries"][key] = dict(variant=v, approach=ap, program=VARIANTS[v]["program"], summary=dict(status="skipped", reason=why))
            continue
        if a.skip_if and any(re.fullmatch(p, f"{mol['name']}:{lv}:{calc}:{v}:{ap}") for p in a.skip_if.split(",")):
            rec["entries"][key] = dict(variant=v, approach=ap, program=VARIANTS[v]["program"], summary=dict(status="skipped", reason="--skip-if"))
            continue
        todo.append((v, ap))
    if not todo:  # nothing to run: record the skips
        fn.write_text(json.dumps(rec, indent=1))
        print(f"== {lay['name']} {case}: nothing to run ({', '.join(f'{k} {e['summary']['status']}' for k, e in rec['entries'].items())})", flush=True)
        return
    rec.setdefault("history", []).append(dict(environment_start=environment(a.linus_bin), argv=sys.argv[1:]))
    runs = {f"{v}:{ap}": [] for v, ap in todo}
    dead = set()

    def save():  # after every repetition, so an interrupted case keeps what it measured (--resume redoes it)
        for v, ap in todo:
            key = f"{v}:{ap}"
            if runs[key]:
                rec["entries"][key] = dict(variant=v, approach=ap, program=VARIANTS[v]["program"], blas_variant=VARIANTS[v]["blas"],
                                           summary=summarize_entry(runs[key]), runs=runs[key])
        fn.write_text(json.dumps(rec, indent=1))

    def one(v, ap, rep, warm):
        key = f"{v}:{ap}"
        if key in dead:
            return
        job = Job(a, v, ap, mol, lv, calc, lay)
        r = run_job(job, a, timed=True)
        r["rep"], r["warmup"] = rep, warm
        if warm:  # only the wall time and status of warm-ups are kept
            for k in ("gradient", "hessian", "timings", "options", "slice_list_before", "slice_list_after", "input", "blas", "blas_binding"):
                r.pop(k, None)
        elif r.get("ok"):
            first = next((x for x in runs[key] if x.get("ok") and not x.get("warmup")), None)
            strip_bulky(r, first)
        runs[key].append(r)
        tag = "warm" if warm else f"rep {rep}"
        print(f"{lay['name']:9} {case:34} {key:26} {tag:6} {r.get('status', '?'):8} "
              f"{(r.get('wall') or 0):9.2f} s  E={r.get('energy')}  it={r.get('iterations')}  busy={r.get('cpu_busy_before')}", flush=True)
        if r.get("status") in ("rejected", "oom", "timeout") or (r.get("status") == "failed" and warm):
            dead.add(key)  # don't repeat a run that cannot succeed

    for w in range(a.warmup):
        for v, ap in todo:
            one(v, ap, w, True)
    for rep in range(a.reps):
        order = todo[rep % len(todo):] + todo[:rep % len(todo)] if todo else []
        for v, ap in order:
            one(v, ap, rep, False)
        save()
    rec["history"][-1]["environment_end"] = environment(a.linus_bin)
    save()
    line = "  ".join(f"{k} {e['summary'].get('median', e['summary'].get('status')):.2f}" if isinstance(e['summary'].get('median'), float)
                     else f"{k} {e['summary'].get('status')}" for k, e in rec["entries"].items())
    print(f"== {lay['name']} {case}: {line}", flush=True)


def cmd_refs(a):
    """Accuracy references (untimed, no lock): linus exact J/K on a fine grid ('fine'), and each
    program's own exact-integral run on its benchmark grid ('own', for the integral error). An external
    program's reference comes from the reference store when it holds it (bench/refstore.py; --refresh
    recomputes), and a computed one goes into the store with its raw outputs."""
    lay = layout(a.layout)
    if lay.get("all_ccds") and not a.allow_all_ccds:
        raise SystemExit("this layout uses CCDs 4-7; pass --allow-all-ccds when they are free")
    outdir = BENCH / "results" / a.tag / "refs"
    outdir.mkdir(parents=True, exist_ok=True)
    for mname in a.mols.split(","):
        mol = molecule(mname)
        for lv in a.levels.split(","):
            for calc in a.calcs.split(","):
                fn = outdir / f"{mname}_{lv}_{calc}.json"
                rec = json.loads(fn.read_text()) if fn.exists() else dict(mol=mname, level=lv, calc=calc, refs={})
                todo = [("fine", "linus", a.fine_grid)]
                if calc != "hessian" or LEVELS[lv]["kind"] == "hf":
                    todo += [("own", p, None) for p in a.programs.split(",") if VARIANTS[p]["program"] == p or VARIANTS[p].get("grid")]
                for kind, prog, grid in todo:
                    key = f"{kind}:{prog}"
                    if key in rec["refs"] and rec["refs"][key].get("ok"):
                        continue
                    if prog == "psi4" and calc == "hessian" and LEVELS[lv]["kind"] != "hf":
                        continue
                    job = Job(a, prog, "exact", mol, lv, calc, lay, ref=True, grid=grid)
                    gname = grid or ("orca:2/DefGrid2/pyscf3/psi4-default by program" if LEVELS[lv]["kind"] != "hf" else None)
                    r = refstore.lookup_or_run(job, a, run_job, REFPY, ORCA, dict(tag=a.tag, layout=lay["name"], reference=kind, grid=gname))
                    r.update(kind=kind, grid=gname, environment=environment(a.linus_bin) if prog == "linus" else None)
                    rec["refs"][key] = r
                    print(f"ref {mname}_{lv}_{calc} {key:14} {r.get('status')} {r.get('wall') or 0:.1f} s E={r.get('energy')}", flush=True)
                    fn.write_text(json.dumps(rec, indent=1))


def load_times(tags):
    """Median wall times {(mol, level, calc, variant, approach): s} from earlier results (pinned or
    8-thread layouts); also reads the previous harness's format (summary per program)."""
    t = {}
    for spec in filter(None, (tags or "").split(",")):
        # TAG, or TAG@LAYOUT*FACTOR: also use that layout's times scaled by FACTOR (e.g. a functional
        # pass on a loaded CCD: preflight@slice:ccd3*0.3); pinned times always take precedence
        tag, _, extra = spec.partition("@")
        lay_x, _, fac = extra.partition("*")
        for f in sorted((BENCH / "results" / tag).glob("**/*.json")):
            if f.parent.name == "refs" or f.name == "blas_validation.json":
                continue
            try:
                d = json.loads(f.read_text())
            except ValueError:
                continue
            if "entries" in d:
                if d.get("layout") == "pinned":
                    scale, keep = 1.0, False
                elif lay_x and d.get("layout") == lay_x:
                    scale, keep = float(fac or 1), True
                else:
                    continue
                for e in d["entries"].values():
                    k = (d["mol"], d["level"], d["calc"], e["variant"], e["approach"])
                    if e["summary"].get("status") == "ok" and not (keep and k in t):
                        t[k] = e["summary"]["median"] * scale
            elif "summary" in d and "mol" in d:  # earlier run_bench.py: linus df, ORCA RIJCOSX, Psi4/PySCF df
                for prog, s in d["summary"].items():
                    ap = {"orca": "rijcosx"}.get(prog, "rijk" if "hf" not in d["level"] else "exact")
                    t.setdefault((d["mol"], d["level"], d["calc"], prog, ap), s["median"])
    return t


def estimate(t, mol, lv, calc, v, ap, threads):
    """Seconds for one run: measured if available, else scaled from la18 by nbf^p with p fitted
    per program on the la18 -> u1an39 pair (default 3, +1 for Hessians); crude beyond u1an39."""
    base_v = v if any(k[3] == v for k in t) else VARIANTS[v]["program"]
    key = (mol, lv, calc, base_v, ap)
    thr = (8 / threads) ** 0.75 if threads > 8 else 1.0
    if key in t:
        return t[key] * thr, "measured"
    basis = LEVELS[lv]["basis"]
    anchors = [(m, t[(m, lv, calc, base_v, ap)]) for m in ("u1an39", "la18") if (m, lv, calc, base_v, ap) in t]
    if not anchors:  # same program/approach at b3lyp of the same basis
        alt = "b3lyp-svp" if basis == "def2-svp" else "b3lyp-tzvp"
        anchors = [(m, t[(m, alt, calc, base_v, ap if ap in ("rijk", "rijcosx") else "rijk")]) for m in ("u1an39", "la18")
                   if (m, alt, calc, base_v, ap if ap in ("rijk", "rijcosx") else "rijk") in t]
    rough = 1.0
    if not anchors:  # last resort: B3LYP/def2-SVP of this program (energy if need be), with rough factors
        for c in (calc, "energy"):
            anchors = [(m, t[(m, "b3lyp-svp", c, base_v, a2)]) for m in ("u1an39", "la18") for a2 in (ap, "rijk", "rijcosx", "exact")
                       if (m, "b3lyp-svp", c, base_v, a2) in t]
            if anchors:
                rough = {"gradient": 1.5, "hessian": 8.0}.get(calc, 1.0) if c != calc else 1.0
                rough *= {"hf": 1.5, "rsh": 1.5, "pure": 0.8}.get(LEVELS[lv]["kind"], 1.0)
                anchors = [(m, x, "def2-svp") for m, x in anchors[:1]]
                break
    if not anchors:
        return None, "no anchor"
    anchors = [a if len(a) == 3 else (a[0], a[1], basis) for a in anchors]
    p = None
    pts = [(NBF[(m, "def2-svp")], t[(m, "b3lyp-svp", c, base_v, a2)]) for m in ("la18", "u1an39") for c in (calc, "energy")
           for a2 in (ap, "rijk") if (m, "b3lyp-svp", c, base_v, a2) in t]
    small = [x for x in pts if x[0] == NBF[("la18", "def2-svp")]]
    big = [x for x in pts if x[0] != NBF[("la18", "def2-svp")]]
    if small and big:
        p = math.log(big[0][1] / small[0][1]) / math.log(big[0][0] / small[0][0])
    p = max(1.5, p if p is not None else 3.0) + (1.0 if calc == "hessian" else 0.0)
    m0, t0, b0 = anchors[0]
    n0, n1 = NBF[(m0, b0)], NBF.get((mol, basis))
    return t0 * (n1 / n0) ** p * thr * rough, f"scaled from {m0}/{b0} (p={p:.1f}{', rough' if rough != 1.0 or b0 != basis else ''})"


def cmd_plan(a):
    tot, total_s, unknown = 0, 0.0, 0
    t = load_times(a.estimate_from)
    for lname in split_layouts(a.layouts):
        lay = layout(lname)
        m = mem_settings(a.mem_gb, lay["threads"])
        print(f"layout {lname}: threads {lay['threads']}, cpus {lay['cpus']}, prefix {' '.join(lay['prefix']) or '(none)'}"
              f"{'  [needs --allow-all-ccds]' if lay.get('all_ccds') else ''}; memory cap {a.mem_gb} GB -> {m}")
        for mname in a.mols.split(","):
            for lv in a.levels.split(","):
                for calc in a.calcs.split(","):
                    for v, ap, why in entries(a.programs.split(","), lv, calc, a.approaches.split(",") if a.approaches else None, mname, a.mem_gb, a.allow_disk_df, lay["threads"]):
                        if not why and a.skip_if and any(re.fullmatch(p, f"{mname}:{lv}:{calc}:{v}:{ap}") for p in a.skip_if.split(",")):
                            why = "--skip-if"
                        est = ""
                        if not why and t:
                            s, how = estimate(t, mname, lv, calc, v, ap, lay["threads"])
                            if s is None:
                                unknown += 1
                                est = f"  est ? ({how})"
                            else:
                                n = a.warmup + a.reps
                                total_s += n * (s + 3.0)  # + ~3 s per run for systemd-run, slice list, busy check
                                est = f"  est {s:8.1f} s x {n} ({how})"
                        if a.verbose or not t:
                            print(f"  {mname:7} {lv:16} {calc:9} {v:15} {ap:8} {'SKIP: ' + why if why else 'run'}{est}")
                        tot += 0 if why else 1
    print(f"{tot} entries x ({a.warmup} warm-up + {a.reps} reps)")
    if t:
        print(f"estimated wall time: {total_s / 3600:.1f} h ({unknown} entries without an estimate)")


class bench_lock:
    """The shared bench lock (flock), unless --no-lock."""

    def __init__(self, a):
        self.f = None if a.no_lock else open(LOCK, "a")

    def __enter__(self):
        if self.f:
            fcntl.flock(self.f, fcntl.LOCK_EX)
        return self

    def __exit__(self, *exc):
        if self.f:
            fcntl.flock(self.f, fcntl.LOCK_UN)
            self.f.close()


def cmd_blas(a):
    """Validate the BLAS overrides (untimed): which library each variant's dgemm_ binds to,
    seen by the dynamic linker (LD_DEBUG=bindings) on a water DFT job; the kernel each library
    picks; and single- and multi-thread DGEMM GF/s of those libraries on the layout's CPUs."""
    lay = layout(a.layout)
    out = dict(environment=environment(a.linus_bin), variants={}, libraries={})
    water = dict(name="water", symbols=["O", "H", "H"], coords=[[0, 0, 0], [0, 1.43, 1.1], [0, -1.43, 1.1]], charge=0, natoms=3, file=None)
    for v in a.programs.split(","):
        prog = VARIANTS[v]["program"]
        if prog not in ("pyscf", "psi4"):
            continue
        job = Job(a, v, "rijk", water, "b3lyp-svp", "energy", lay)
        job.setenv["LD_DEBUG"] = "bindings"
        job.setenv["LD_DEBUG_OUTPUT"] = "lddebug"
        job.setenv["BENCH_NO_BINDING_PROBE"] = "1"  # only the dynamic linker's own relocations
        SCRATCH.mkdir(parents=True, exist_ok=True)
        wd = pathlib.Path(tempfile.mkdtemp(prefix=f"blas_{v}_", dir=SCRATCH))
        job.prepare(wd)
        mcmd = [SYS_PY, str(JOBS / "measure.py"), str(wd / "measure.json"), job.stdin, str(wd / "stdout.txt"), str(wd / "stderr.txt")]
        for k, val in job.setenv.items():
            mcmd += ["--setenv", f"{k}={val}"]
        with bench_lock(a):
            subprocess.run(["systemd-run", "--user", "--scope", "-q", "-p", f"MemoryMax={a.mem_gb}G", "-p", "MemorySwapMax=0", "-p", "OOMPolicy=continue"] + lay["prefix"] + mcmd + ["--"] + job.cmd,
                           cwd=wd, env=job.env, capture_output=True)
        binds = {}
        for f in wd.glob("lddebug*"):
            for line in f.read_text(errors="replace").splitlines():
                m = re.search(r"binding file (\S+) \[\d+\] to (\S+) \[\d+\]: normal symbol `(dgemm_|dsyevd_|dsyev_|dpotrf_|GOMP_parallel|omp_get_num_threads)'", line)
                if m and re.search(r"(libnp_helper|libdft|libcvhf|libri|core\.cpython)", m.group(1)):
                    binds.setdefault(f"{os.path.basename(m.group(1))}:{m.group(3)}", set()).add(os.path.basename(m.group(2)))
        res = json.loads((wd / "out.json").read_text()) if (wd / "out.json").exists() else {}
        out["variants"][v] = dict(setenv={k: val for k, val in job.setenv.items() if not k.startswith("LD_DEBUG")},
                                  ld_debug_bindings={k: sorted(s) for k, s in binds.items()}, energy=res.get("energy"),
                                  loaded=res.get("blas"), in_process_binding=res.get("blas_binding"))
        print(v, json.dumps(out["variants"][v]["ld_debug_bindings"]), res.get("energy"), flush=True)
        shutil.rmtree(wd, ignore_errors=True)
    pyscf_lib = sh([str(REFPY), "-c", "import pyscf.lib, os; print(os.path.dirname(pyscf.lib.__file__))"]).splitlines()[-1]
    libs = {"pyscf-bundled-openblas-0.3.3": (os.path.join(pyscf_lib, "libopenblas-r0-f650aae0.3.3.so"), {}),
            "pyscf-bundled-openblas-0.3.3+Haswell (set after NumPy)": (os.path.join(pyscf_lib, "libopenblas-r0-f650aae0.3.3.so"), {"BENCH_LATE_CORETYPE": "Haswell"}),
            "conda-openblas-0.3.34": (CONDA_OPENBLAS, {}),
            "conda-openblas-0.3.34+Haswell (process-wide)": (CONDA_OPENBLAS, {"OPENBLAS_CORETYPE": "Haswell"}),
            "conda-forge-openblas-0.3.34-openmp (fair-BLAS bundle)": (str(FASTBLAS_DIR / "libopenblas-omp-0.3.34.so"), {}),
            "mkl-2025.3": (str(REFENV / "lib/libmkl_rt.so.2"), {})}
    for name, (path, env) in libs.items():
        for nt in (1, lay["threads"]):
            with bench_lock(a):
                p = subprocess.run(["systemd-run", "--user", "--scope", "-q", "-p", "MemoryMax=4G", "-p", "MemorySwapMax=0", "-p", "OOMPolicy=continue"] + lay["prefix"]
                                   + [str(REFPY), str(BENCH / "blas_probe.py"), "dgemm", path, str(nt), "2000"],
                                   env=dict(os.environ, OMP_NUM_THREADS=str(nt), **env), capture_output=True, text=True)
            try:
                d = json.loads(p.stdout.strip().splitlines()[-1])
            except Exception:  # noqa: BLE001
                d = dict(error=(p.stderr or p.stdout)[-500:], returncode=p.returncode)
            out["libraries"][f"{name}@{nt}"] = d
            print(f"{name:40} threads {nt:2}: {d.get('gflops', 0):7.1f} GF/s core={d.get('core')} {d.get('error', '')}", flush=True)
    fn = BENCH / "results" / a.tag / "blas_validation.json"
    fn.parent.mkdir(parents=True, exist_ok=True)
    fn.write_text(json.dumps(out, indent=1, default=list))


_RUNNING = [None]  # systemd unit of the job in flight


def _terminate(signum, frame):
    """Don't leave a job running (outside the lock) when the harness itself is stopped."""
    if _RUNNING[0]:
        subprocess.run(["systemctl", "--user", "kill", "--signal=KILL", f"{_RUNNING[0]}.scope"], capture_output=True)
    raise SystemExit(128 + signum)


def main():
    import signal
    signal.signal(signal.SIGTERM, _terminate)
    signal.signal(signal.SIGINT, _terminate)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--tag", required=True)
    common.add_argument("--mols", default="la18")
    common.add_argument("--levels", default="b3lyp-svp")
    common.add_argument("--calcs", default="energy")
    common.add_argument("--programs", default="linus,orca,pyscf,pyscf-fastblas,psi4,psi4-openblas")
    common.add_argument("--approaches", default=None, help="restrict to these approaches (rijk,rijcosx,rij,exact)")
    common.add_argument("--mem-gb", type=float, default=20, help="memory cap of every job (systemd scope, no swap)")
    common.add_argument("--linus-bin", default=str(ROOT / "target/release/linus"))
    common.add_argument("--linus-grid", default="orca:2", help="linus XC grid for DFT runs (orca:2 = ORCA's DefGrid2)")
    common.add_argument("--timeout", type=float, default=4 * 3600)
    common.add_argument("--no-lock", action="store_true", help="development only: don't take the shared bench lock")
    common.add_argument("--keep-failed", action="store_true", help="keep the work directories of failed runs")
    common.add_argument("--allow-all-ccds", action="store_true", help="allow layouts on CCDs 4-7 (only when they are free)")
    common.add_argument("--allow-disk-df", action="store_true", help="run PySCF/Psi4 DF cases whose 3-index tensor exceeds the budget")
    common.add_argument("--setenv", action="append", default=[], metavar="VARIANT:K=V",
                        help="extra environment for one program variant (experiments), e.g. pyscf-fastblas:OPENBLAS_NUM_THREADS=1")
    r = sub.add_parser("run", parents=[common])
    r.add_argument("--layouts", default="pinned")
    r.add_argument("--reps", type=int, default=3)
    r.add_argument("--warmup", type=int, default=1)
    r.add_argument("--resume", action="store_true", help="keep finished entries of existing result files")
    r.add_argument("--skip-if", default=None, help="comma-separated regexes on mol:level:calc:variant:approach to skip")
    f = sub.add_parser("refs", parents=[common])
    f.add_argument("--layout", default="slice:ccd2")
    f.add_argument("--fine-grid", default="linus:4", help="grid of the linus exact-integral reference")
    f.add_argument("--refresh", action="store_true", help="recompute external references the store holds (and replace them)")
    b = sub.add_parser("blas", parents=[common])
    b.add_argument("--layout", default="pinned")
    p = sub.add_parser("plan", parents=[common])
    p.add_argument("--layouts", default="pinned")
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--estimate-from", default=None, help="comma-separated result tags whose median times seed the estimate")
    p.add_argument("--verbose", action="store_true", help="list every entry with its estimate")
    p.add_argument("--skip-if", default=None, help="as for run")
    p.add_argument("--resume", action="store_true", help="accepted (and ignored) so plan takes run's arguments")
    a = ap.parse_args()
    a.linus_bin = str(pathlib.Path(a.linus_bin).resolve())
    a._extra_setenv = {}
    for item in a.setenv:
        v, _, kv = item.partition(":")
        k, _, val = kv.partition("=")
        a._extra_setenv.setdefault(v, {})[k] = val
    for v in a.programs.split(","):
        if v not in VARIANTS:
            raise SystemExit(f"unknown program variant {v}; known: {', '.join(VARIANTS)}")
        if VARIANTS[v].get("needs") and not pathlib.Path(VARIANTS[v]["needs"]).exists():
            raise SystemExit(f"{v} needs {VARIANTS[v]['needs']}: run bench/setup_fastblas.sh first")
    {"run": cmd_run, "refs": cmd_refs, "plan": cmd_plan, "blas": cmd_blas}[a.cmd](a)


if __name__ == "__main__":
    main()
