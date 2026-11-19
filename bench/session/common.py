"""Shared pieces of the session benchmark (bench/session/README.md): the systems, the paths of the
build under test, the locks, the job runner, and the classification of linus's timers.

Every timed job is one process (a replay of one trajectory in one mode) run under a memory cap in its
own systemd scope, pinned to CCD0's eight physical cores (`slice bench`), while the driver holds the
shared bench lock and, for GPU jobs, the node's writer-preferring exclusive GPU lock (its turnstile,
then the lock, as tools/internal/refbase/common.py `gpu_lock_prefix(True)`). The driver sleeps 5 s
after each release of the GPU lock.
"""
from __future__ import annotations

import contextlib
import fcntl
import json
import os
import pathlib
import platform
import shutil
import subprocess
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "bench"))
from bench_paths import LOCK as BENCH_LOCK, MAIN  # noqa: E402

# results/ holds the 2026-11-12 study (the Calculator API); SESSBENCH_RESULTS another run's (results-session-api:
# the rerun through linus.Session, 2026-11-13). The trajectories are always the 2026-11-12 ones.
RESULTS = pathlib.Path(os.environ.get("SESSBENCH_RESULTS", HERE / "results"))
TRAJECTORIES = HERE / "results" / "trajectories"
# work directories (xyz files of mode A, logs) and the GPU kernel cache: on disk, outside the tree
SCRATCH = pathlib.Path(os.environ.get("SESSBENCH_SCRATCH", ROOT / "bench" / "scratch" / "session"))
KCACHE = SCRATCH / "kcache"
GPU_LOCKS = pathlib.Path(os.environ.get("REFBASE_GPU_LOCKS", MAIN / "bench" / "scratch"))
LINUS_BIN = pathlib.Path(os.environ.get("SESSBENCH_LINUS", ROOT / "target" / "release" / "linus"))
PYTHON = os.environ.get("SESSBENCH_PYTHON", str(MAIN / ".venv-py-a7214c5" / "bin" / "python"))
# the Python package under test: the source tree's (a maturin develop build), or SESSBENCH_PACKAGE (an
# unpacked wheel: the package and its dist-info, so its entry points are found)
PYPATH = pathlib.Path(os.environ.get("SESSBENCH_PACKAGE", ROOT / "crates" / "linus-py" / "python"))
# more directories for the jobs' PYTHONPATH (unreleased qcdata and qccompute sources), after the package's
PYPATH_EXTRA = os.environ.get("SESSBENCH_PYPATH_EXTRA", "")
CCD0 = "0-7"  # slice bench: CCD0's physical cores, SMT siblings idle
THREADS = 8
BOHR_ANGSTROM = 0.529177210544  # linus_core::units::BOHR_ANGSTROM (CODATA 2022): what the CLI divides Å by

# the systems: B3LYP, jk df, default grid and SCF settings, backend cpu (CPU) or auto (GPU)
SYSTEMS = {
    "water-svp": dict(xyz="tests/molecules/water.xyz", charge=0, mult=1, basis="def2-svp",
                      platforms=("cpu", "gpu"), mem="16G", seed=11),
    "la18-svp": dict(xyz="bench/molecules/la18.xyz", charge=0, mult=1, basis="def2-svp",
                     platforms=("cpu", "gpu"), mem="24G", seed=12),
    "u1an39-svp": dict(xyz="bench/molecules/u1an39.xyz", charge=-1, mult=1, basis="def2-svp",
                       platforms=("cpu", "gpu"), mem="32G", seed=13),
    "u1an39-tzvp": dict(xyz="bench/molecules/u1an39.xyz", charge=-1, mult=1, basis="def2-tzvp",
                        platforms=("gpu",), mem="48G", seed=14),
    # open shell: the la18 radical cation, UKS (its stability probe runs at a calculator's first geometry)
    "la18cat-svp": dict(xyz="bench/molecules/la18.xyz", charge=1, mult=2, basis="def2-svp",
                        platforms=("cpu", "gpu"), mem="24G", seed=15),
}
METHOD = "b3lyp"
MODES = ("A", "B", "C", "D", "W")
MODE_NAMES = {
    "A": "new process per gradient (CLI)",
    "B": "new linus.Session per gradient (one process)",
    "C": "one Session, reset() each step",
    "D": "one Session, its default lineage (warm start)",
    "W": "one Session in a worker process (qccompute.serve, JSON lines over a pipe)",
}
TRAJ_RMS_BOHR = 0.1   # start: the input structure displaced by Gaussian noise of this RMS (bohr)
TRAJ_GRMS = 3e-4      # stop: gradient RMS below this (Eh/bohr) ...
TRAJ_MAX_STEPS = 30   # ... or after this many gradients
TRAJ_H0 = 0.72        # BFGS initial Hessian (Eh/bohr^2): ASE BFGS's 70 eV/Å^2


def backend(plat: str) -> str:
    return {"cpu": "cpu", "gpu": "auto"}[plat]


def read_xyz(path) -> tuple[list[str], list[list[float]]]:
    lines = pathlib.Path(path).read_text().splitlines()
    n = int(lines[0].split()[0])
    sym, pos = [], []
    for ln in lines[2:2 + n]:
        p = ln.split()
        sym.append(p[0])
        pos.append([float(v) / BOHR_ANGSTROM for v in p[1:4]])
    return sym, pos


def write_xyz(path, symbols, pos_bohr, comment="") -> None:
    rows = [f"{s:2s} {x[0] * BOHR_ANGSTROM:20.12f} {x[1] * BOHR_ANGSTROM:20.12f} {x[2] * BOHR_ANGSTROM:20.12f}"
            for s, x in zip(symbols, pos_bohr)]
    pathlib.Path(path).write_text(f"{len(rows)}\n{comment}\n" + "\n".join(rows) + "\n")


def traj_path(system: str) -> pathlib.Path:
    return TRAJECTORIES / f"{system}.json"


def serve_command() -> list[str]:
    """The worker of mode W, serving linus."""
    return [PYTHON, "-m", "qccompute.serve", "linus"]


def env() -> dict:
    e = dict(os.environ)
    e.update(PYTHONPATH=os.pathsep.join(p for p in (str(PYPATH), PYPATH_EXTRA) if p), LINUS_GPU_CACHE=str(KCACHE), OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
             MKL_NUM_THREADS="1", PYTHONUNBUFFERED="1")
    e.pop("LINUS_GPU_MEMORY_GIB", None)
    return e


def log(msg: str) -> None:
    print(f"[sessbench {time.strftime('%H:%M:%S', time.gmtime())}] {msg}", flush=True)


# ------------------------------------------------------------------------------------------------
# locks

def _flock(path, shared=False, poll=None, max_wait=None):
    f = open(path, "a")
    mode = fcntl.LOCK_SH if shared else fcntl.LOCK_EX
    if poll is None:
        fcntl.flock(f, mode)
        return f
    end = time.time() + (max_wait or 1e9)
    while True:
        try:
            fcntl.flock(f, mode | fcntl.LOCK_NB)
            return f
        except BlockingIOError:
            if time.time() > end:
                f.close()
                return None
            time.sleep(poll)


def wait_gpu_idle(max_wait=60, poll=2) -> float:
    """Wait, holding nothing, until no one holds the GPU lock (at most `max_wait` s), so that waiting for a long
    shared holder does not keep the bench lock or the turnstile (refbase's wait_gpu_idle)."""
    t0 = time.time()
    f = _flock(GPU_LOCKS / "gpu.lock", poll=poll, max_wait=max_wait)
    if f is not None:
        fcntl.flock(f, fcntl.LOCK_UN)
        f.close()
    return time.time() - t0


def _release(*files) -> None:
    for f in files:
        if f is not None:
            fcntl.flock(f, fcntl.LOCK_UN)
            f.close()


@contextlib.contextmanager
def held(gpu: bool):
    """The bench lock, and for GPU jobs the GPU turnstile and exclusive lock, for the duration of one job.

    One driver runs the jobs one after the other (CPU and GPU). A GPU job takes the bench lock (CCD0), then the
    turnstile (new shared GPU runs queue behind it), waits for the shared runs in flight to end and takes the
    exclusive GPU lock (gpu_round2_common.md's protocol for timed runs): it never holds the GPU idle while another
    agent's CPU timing has CCD0."""
    waits = {"bench": 0.0, "gpu": 0.0}
    t = lk = None
    t0 = time.time()
    b = _flock(BENCH_LOCK)
    waits["bench"] = time.time() - t0
    if gpu:
        t0 = time.time()
        t = _flock(GPU_LOCKS / "gpu.turnstile")
        lk = _flock(GPU_LOCKS / "gpu.lock")
        waits["gpu"] = time.time() - t0
    try:
        yield waits
    finally:
        for f in (lk, t, b):
            if f is not None:
                fcntl.flock(f, fcntl.LOCK_UN)
                f.close()
        if gpu:
            time.sleep(5)  # gpu_round2_common.md: >= 5 s after each release of the GPU lock


def capped(mem: str, cmd: list[str]) -> list[str]:
    """`cmd` in its own memory-capped systemd scope, pinned to CCD0's physical cores."""
    pre = []
    if shutil.which("systemd-run") and not os.environ.get("SESSBENCH_NO_CAP"):
        pre = ["systemd-run", "--user", "--scope", "-q", "-p", f"MemoryMax={mem}", "-p", "MemorySwapMax=0",
               "-p", "OOMPolicy=continue"]
    return pre + ["taskset", "-c", CCD0] + cmd


def run_job(system: str, plat: str, cmd: list[str], logfile: pathlib.Path) -> dict:
    """One job under the locks; returns the lock waits, wall time and exit status."""
    return run_jobs(system, plat, [(cmd, logfile)])[0]


def run_jobs(system: str, plat: str, jobs: list) -> list[dict]:
    """Jobs (cmd, logfile) one after the other under one hold of the locks (a GPU system's four modes of one repeat:
    a few minutes, well under the 15 minutes a GPU hold may last); each its own process and memory-capped scope."""
    gpu = plat == "gpu"
    out = []
    with held(gpu) as waits:
        for cmd, logfile in jobs:
            logfile.parent.mkdir(parents=True, exist_ok=True)
            t0 = time.time()
            with open(logfile, "w") as lf:
                rc = subprocess.run(capped(SYSTEMS[system]["mem"], cmd), stdout=lf, stderr=subprocess.STDOUT,
                                    env=env()).returncode
            out.append(dict(rc=rc, wall=time.time() - t0, waits=waits if not out else {}))
    return out


def run_untimed(system: str, cmd: list[str], cpus: str, logfile: pathlib.Path) -> dict:
    """An untimed GPU job (a profile): through the turnstile, then the shared GPU lock for its whole life, pinned to
    `cpus`, in a memory-capped scope; no bench lock."""
    t0 = time.time()
    t = _flock(GPU_LOCKS / "gpu.turnstile")
    _release(t)
    lk = _flock(GPU_LOCKS / "gpu.lock", shared=True)
    wait = time.time() - t0
    try:
        logfile.parent.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        pre = capped(SYSTEMS[system]["mem"], [])[:-3]  # the scope, without CCD0's taskset
        with open(logfile, "w") as lf:
            rc = subprocess.run(pre + ["taskset", "-c", cpus] + cmd, stdout=lf, stderr=subprocess.STDOUT,
                                env=env()).returncode
        return dict(rc=rc, wall=time.time() - t0, waits=dict(gpu_shared=wait), cpus=cpus)
    finally:
        _release(lk)
        time.sleep(5)


def ccd_load(seconds=1.0) -> dict:
    """Busy hardware threads per CCD (physical cores and SMT siblings) over `seconds`, from /proc/stat."""
    def snap():
        d = {}
        for ln in open("/proc/stat"):
            if ln.startswith("cpu") and ln[3].isdigit():
                p = ln.split()
                v = list(map(int, p[1:]))
                d[int(p[0][3:])] = (sum(v), v[3] + v[4])
        return d
    a = snap()
    time.sleep(seconds)
    b = snap()
    busy = {c: 1 - (b[c][1] - a[c][1]) / max(1, b[c][0] - a[c][0]) for c in a}
    n = len(busy) // 2
    return {f"ccd{k}": [round(sum(busy.get(8 * k + i, 0) for i in range(8)), 1),
                        round(sum(busy.get(n + 8 * k + i, 0) for i in range(8)), 1)] for k in range(n // 8)}


def host_facts() -> dict:
    cpu = ""
    for ln in open("/proc/cpuinfo"):
        if ln.startswith("model name"):
            cpu = ln.split(":", 1)[1].strip()
            break
    gpu = ""
    with contextlib.suppress(Exception):
        gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=30).stdout.strip()
    return dict(cpu=cpu, gpu=gpu, kernel=platform.release(), python=platform.python_version())


# ------------------------------------------------------------------------------------------------
# linus's timers, classified (crates/linus/src/driver/mod.rs energy_for, scf/, two_electron/, derivatives/)
#
# `total` is the whole call; `energy` the SCF stage (setup, loop, dispersion); `scf_total` the SCF loop (iterations,
# final energy, probes; on the device loop also its setup `scf_setup`: loop buffers and uploads); the gradient
# stage is total - energy. Setups nest in their parents (grad_2e_setup in grad_2e, grad_xc_setup in grad_xc).
# `grid` sums every molecular-grid build of the call: on the CPU the SCF's and the gradient's (the gradient
# builds its own integrator at the same geometry, the same grid), on the GPU the SCF's only (the gradient takes the
# SCF's integrator, crates/linus/src/driver/handoff.rs); the CPU's half is put in each stage.
SETUP_KEYS = ("pairs", "one_electron", "ecp", "jk_setup", "xc_setup", "grid", "nlc_grid", "guess", "scf_setup",
              "grad_pairs", "grad_2e_setup", "grad_xc_setup", "closed_shell_probe", "stability")


def breakdown(t: dict, plat: str) -> dict:
    """One gradient call's seconds by stage and by kind of setup, from its `timings`.

    Stages (they sum to `total`): scf_setup (the SCF stage before and after its loop, with the device loop's
    setup), scf_iter (the loop's iterations, the probes included), gradient. Kinds of setup (each a part of one
    stage): ints_pairs (shell-pair data and one-electron integrals, SCF and gradient), df (the DF tensor and
    metric: jk_setup and the gradient's grad_2e_setup), grid_xc (molecular grid and the device XC setup, SCF and
    gradient), guess (SAD), dev_loop (the device SCF loop's setup)."""
    g = lambda k: t.get(k, 0.0)  # noqa: E731
    total, energy, loop = g("total"), g("energy"), g("scf_total")
    grid_scf = g("grid") / 2 if plat == "cpu" else g("grid")
    out = dict(total=total, scf_setup=energy - loop + g("scf_setup"), scf_iter=loop - g("scf_setup"),
               gradient=total - energy)
    out["ints_pairs"] = g("pairs") + g("one_electron") + g("ecp") + g("grad_pairs")
    out["df"] = g("jk_setup") + g("grad_2e_setup")
    out["grid_xc"] = g("grid") + g("nlc_grid") + g("xc_setup") + g("grad_xc_setup")
    out["guess"] = g("guess")
    out["dev_loop"] = g("scf_setup")
    out["probes"] = g("closed_shell_probe") + g("stability")
    out["grad_setup"] = g("grad_pairs") + g("grad_2e_setup") + g("grad_xc_setup") + (g("grid") - grid_scf)
    out["scf_setup_parts"] = g("pairs") + g("one_electron") + g("ecp") + g("jk_setup") + grid_scf + g("nlc_grid") \
        + g("xc_setup") + g("guess") + g("scf_setup")
    out["setup"] = out["scf_setup"] + out["grad_setup"]
    for k in SETUP_KEYS:
        out["t_" + k] = g(k)  # the raw timers
    return out


def dump(path, obj) -> None:
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(obj, indent=1))
    tmp.replace(path)
