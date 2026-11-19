"""Exact J+K build and RHF benchmarks: linus (FastJk) vs PySCF (libcint, direct) vs ORCA 6.

python bench/run_jk_bench.py [--cases caffeine:def2-svp ...] [--threads 1 8 16 64]
                          [--repeats 3] [--programs linus pyscf orca] [--thresh 1e-12]
                          [--rhf] [--tag name] [--lock FILE] [--linus-bin NAME=PATH ...]
                          [--slice bench|none] [--orca-maxiter N]

Per case and thread count the programs are run in interleaved order, `--repeats` times, and
medians are reported:
- J+K build on the same converged density (bench/scratch/<mol>.<basis>.dm.bin, from PySCF
  DF-RHF; generated if missing). linus and PySCF time the build in-process (second of two
  builds, so one-time setup is excluded); ORCA's time is "Fock matrix formation" / number of
  Fock builds of an exact RHF with a full rebuild every iteration (DirectResetFreq 1).
- Accuracy: max |J - J_ref|, |K - K_ref| against linus at thresh 1e-16 (and PySCF's against
  the same reference).
- With --rhf: total wall time and energy of a full direct RHF (linus: core guess, DIIS;
  PySCF: minao guess, DIIS; both conv 1e-10 Eh).

On the shared AFH node commands run under `slice` (~/.local/bin/slice): `slice bench`
(CCD0, 8 physical cores, SMT siblings idle) for <= 8 threads and `slice work` (48 cores +
siblings) above, each inside its own memory-capped scope (`systemd-run --user --scope
-p MemoryMax=<--mem-max>`; the node is shared and an OOM kill takes down the whole session).
ORCA runs with %maxcore scaled to an 8 GB total and at most --orca-max-threads ranks. The
JSON in bench/results/ records the CPU model, load averages and the `slice list` output,
never the host name.

--lock FILE runs every timed process under `flock FILE` (a lock shared by the agents timing on
CCD0); --linus-bin NAME=PATH adds further linus jk_bench binaries as programs NAME (e.g. the
build of an older commit); --slice none runs unpinned; --orca-maxiter N caps ORCA's SCF for
large cases (time per build from its iteration table, see bench/orca_jk.py).
"""
import argparse
import datetime
import json
import os
import pathlib
import shutil
import statistics
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
from bench_paths import MAIN, REFENV  # noqa: E402  (env overrides: bench/bench_paths.py)
VENV_PY = os.environ.get("LINUS_PY", str(MAIN / ".venv/bin/python"))
REF_PY = os.environ.get("LINUS_REFENV_PY", str(REFENV / "bin/python"))
SLICE = shutil.which("slice") or os.path.expanduser("~/.local/bin/slice")
SCRATCH = ROOT / "bench/scratch"
JK = ROOT / "target/release/examples/jk_bench"
SCF = ROOT / "target/release/examples/scf"


MEM_MAX = "12G"
AUX_THREADS = 8  # untimed helper runs (shared CCD2)
LOCK = None
SLICE_MODE = "auto"
LINUS_BINS = {}
ORCA_MAXITER = None


def slice_prefix(threads, lock=True):
    pre = ["flock", LOCK] if LOCK and lock else []
    if shutil.which("systemd-run"):
        pre += ["systemd-run", "--user", "--scope", "-q", "-p", f"MemoryMax={MEM_MAX}", "-p", "MemorySwapMax=0"]
    if os.path.exists(SLICE) and SLICE_MODE != "none":
        pre += [SLICE, SLICE_MODE if SLICE_MODE != "auto" else ("bench" if threads <= 8 else "work")]
    return pre


def run_json(cmd, env=None, timeout=None):
    t0 = time.perf_counter()
    p = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=timeout)
    wall = time.perf_counter() - t0
    if p.returncode != 0:
        raise RuntimeError(f"{cmd[:4]} failed: {p.stderr[-2000:]}")
    return json.loads(p.stdout.strip().splitlines()[-1]), wall


def ensure_density(mol, basis):
    d = SCRATCH / f"{mol}.{basis}.dm.bin"
    if not d.exists():
        SCRATCH.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ, OMP_NUM_THREADS=str(AUX_THREADS))
        run_json(aux_prefix() + [VENV_PY, str(ROOT / "bench/pyscf_jk.py"), "density",
                                     str(ROOT / f"tests/molecules/{mol}.xyz"), basis, str(d)], env)
    return d


def max_abs_diff(a, b):
    import numpy as np
    x = np.fromfile(a, dtype="<f8")
    y = np.fromfile(b, dtype="<f8")
    return float(np.abs(x - y).max())


def aux_prefix():
    """Untimed helper runs (densities, accuracy references): CCD2, no lock."""
    pre = ["systemd-run", "--user", "--scope", "-q", "-p", f"MemoryMax={MEM_MAX}", "-p", "MemorySwapMax=0"] if shutil.which("systemd-run") else []
    if os.path.exists(SLICE) and SLICE_MODE != "auto":
        pre += [SLICE, "ccd2"]
    elif os.path.exists(SLICE):
        pre += [SLICE, "work"]
    return pre


def mol_xyz(mol):
    p = ROOT / f"tests/molecules/{mol}.xyz"
    return p if p.exists() else ROOT / f"bench/molecules/{mol}.xyz"


def linus_jk(mol, basis, dens, threads, thresh, out=None, binary=None, aux=False):
    cmd = (aux_prefix() if aux else slice_prefix(threads)) + [str(binary or JK), str(mol_xyz(mol)), basis, "--dens", str(dens),
                                   "--builder", "new", "--threads", str(threads), "--thresh", str(thresh), "--repeat", "2"]
    if out:
        cmd += ["--out", str(out)]
    d, _ = run_json(cmd)
    return {"time": d["times"][-1], "times": d["times"], "nbf": d["nbf"], "isa": d["stats"]["isa"],
            "quartets": d["stats"]["quartets"], "prim_quartets": d["stats"]["prim_quartets"]}


def pyscf_jk(mol, basis, dens, threads, thresh, out=None, aux=False):
    env = dict(os.environ, OMP_NUM_THREADS=str(threads))
    cmd = (aux_prefix() if aux else slice_prefix(threads)) + [VENV_PY, str(ROOT / "bench/pyscf_jk.py"), "jk", str(ROOT / f"tests/molecules/{mol}.xyz"),
                                   basis, str(dens), str(threads), str(thresh), "2"]
    if out:
        cmd.append(str(out))
    d, _ = run_json(cmd, env)
    return {"time": d["times"][-1], "times": d["times"], "version": d["version"]}


def orca_jk(mol, basis, threads, thresh, tag, rep):
    env = dict(os.environ, OMPI_MCA_hwloc_base_binding_policy="none")
    rundir = SCRATCH / f"orca_{tag}_{mol}_{basis}_t{threads}_r{rep}"
    if rundir.exists():
        shutil.rmtree(rundir)
    cmd = slice_prefix(threads) + [REF_PY, str(ROOT / "bench/orca_jk.py"), str(mol_xyz(mol)),
                                   basis, str(threads), str(thresh), str(rundir)]
    if ORCA_MAXITER:
        cmd.append(str(ORCA_MAXITER))
    d, _ = run_json(cmd, env)
    if not d.get("ok"):
        raise RuntimeError(f"ORCA failed in {rundir}")
    return {"time": d["fock_per_build"], "fock_total": d["fock_total"], "cycles": d["cycles"],
            "energy": d["energy"], "scf_wall": d["wall"], "fock_source": d.get("fock_source"),
            "iterations_timed": d.get("iterations_timed")}


def linus_rhf(mol, basis, threads, thresh):
    cmd = slice_prefix(threads) + [str(SCF), str(ROOT / f"tests/molecules/{mol}.xyz"), basis, "--jk", "fast",
                                   "--thresh", str(thresh), "--threads", str(threads)]
    d, wall = run_json(cmd)
    return {"time": wall, "energy": d["energy"], "iterations": d["iterations"], "converged": d["converged"]}


def pyscf_rhf(mol, basis, threads, thresh):
    env = dict(os.environ, OMP_NUM_THREADS=str(threads))
    cmd = slice_prefix(threads) + [VENV_PY, str(ROOT / "bench/pyscf_jk.py"), "rhf", str(ROOT / f"tests/molecules/{mol}.xyz"),
                                   basis, str(threads), str(thresh)]
    d, wall = run_json(cmd, env)
    return {"time": wall, "scf_wall": d["wall"], "energy": d["energy"], "iterations": d["cycles"], "converged": d["converged"]}


def environment():
    info = {"date": datetime.datetime.now().isoformat(timespec="seconds")}
    try:
        cpu = [ln.split(":", 1)[1].strip() for ln in open("/proc/cpuinfo") if ln.startswith("model name")]
        info["cpu"] = cpu[0]
        info["logical_cpus"] = len(cpu)
    except OSError:
        pass
    info["loadavg"] = open("/proc/loadavg").read().split()[:3]
    if os.path.exists(SLICE):
        info["slice_list"] = subprocess.run([SLICE, "list"], capture_output=True, text=True).stdout.strip().splitlines()
    git = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], capture_output=True, text=True)
    info["linus_commit"] = git.stdout.strip()
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", nargs="+", default=["caffeine:def2-svp", "caffeine:def2-tzvp", "cholesterol:def2-svp"])
    ap.add_argument("--threads", nargs="+", type=int, default=[1, 8, 16, 64])
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--programs", nargs="+", default=["linus", "pyscf", "orca"])
    ap.add_argument("--orca-max-cost", type=float, default=3000.0,
                    help="skip ORCA runs whose estimated cost (nbf^4/1e8 / threads) exceeds this")
    ap.add_argument("--orca-max-threads", type=int, default=16)
    ap.add_argument("--mem-max", default="12G", help="memory cap of every benchmark process")
    ap.add_argument("--thresh", type=float, default=1e-12)
    ap.add_argument("--rhf", action="store_true")
    ap.add_argument("--tag", default=datetime.datetime.now().strftime("%Y%m%d-%H%M"))
    ap.add_argument("--lock", default=None)
    ap.add_argument("--linus-bin", nargs="*", default=[])
    ap.add_argument("--slice", default="auto")
    ap.add_argument("--orca-maxiter", type=int, default=None)
    a = ap.parse_args()
    global MEM_MAX, LOCK, LINUS_BINS, SLICE_MODE, ORCA_MAXITER
    MEM_MAX = a.mem_max
    LOCK = a.lock
    LINUS_BINS = dict(x.split("=", 1) for x in a.linus_bin)
    SLICE_MODE = a.slice
    ORCA_MAXITER = a.orca_maxiter
    res = {"tag": a.tag, "thresh": a.thresh, "environment_start": environment(), "cases": []}
    out = ROOT / f"bench/results/jk_{a.tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    for case in a.cases:
        mol, basis = case.split(":")
        dens = ensure_density(mol, basis)
        # references for accuracy: linus at 1e-16 (and both programs at the benchmark threshold)
        ref = SCRATCH / f"ref_{mol}_{basis}"
        if not pathlib.Path(f"{ref}.J.bin").exists():
            linus_jk(mol, basis, dens, AUX_THREADS, 1e-16, out=ref, aux=True)
        acc = {}
        lo = SCRATCH / f"acc_linus_{mol}_{basis}"
        linus_jk(mol, basis, dens, AUX_THREADS, a.thresh, out=lo, aux=True)
        acc["linus"] = {"J": max_abs_diff(f"{ref}.J.bin", f"{lo}.J.bin"), "K": max_abs_diff(f"{ref}.K.bin", f"{lo}.K.bin")}
        for name, binary in LINUS_BINS.items():
            if name in a.programs:
                lb = SCRATCH / f"acc_{name}_{mol}_{basis}"
                linus_jk(mol, basis, dens, AUX_THREADS, a.thresh, out=lb, binary=binary, aux=True)
                acc[name] = {"J": max_abs_diff(f"{ref}.J.bin", f"{lb}.J.bin"), "K": max_abs_diff(f"{ref}.K.bin", f"{lb}.K.bin")}
        if "pyscf" in a.programs:
            po = SCRATCH / f"acc_pyscf_{mol}_{basis}"
            pyscf_jk(mol, basis, dens, AUX_THREADS, a.thresh, out=po, aux=True)
            acc["pyscf"] = {"J": max_abs_diff(f"{ref}.J.bin", f"{po}.J.bin"), "K": max_abs_diff(f"{ref}.K.bin", f"{po}.K.bin")}
        crow = {"molecule": mol, "basis": basis, "accuracy_vs_linus_1e-16": acc, "runs": {}, "rhf": {}}
        print(json.dumps({"case": case, "accuracy": acc}), flush=True)
        nbf = None
        for t in a.threads:
            runs = {p: [] for p in a.programs}
            for rep in range(a.repeats):
                for p in a.programs:
                    try:
                        if p == "linus":
                            r = linus_jk(mol, basis, dens, t, a.thresh)
                            nbf = r["nbf"]
                        elif p in LINUS_BINS:
                            r = linus_jk(mol, basis, dens, t, a.thresh, binary=LINUS_BINS[p])
                        elif p == "pyscf":
                            r = pyscf_jk(mol, basis, dens, t, a.thresh)
                        else:
                            if (nbf and (nbf / 100) ** 4 / t > a.orca_max_cost) or rep > 0 or t > a.orca_max_threads:
                                continue  # one ORCA SCF per thread count (it averages over its builds)
                            r = orca_jk(mol, basis, t, a.thresh, a.tag, rep)
                    except Exception as e:  # keep going; record the failure
                        r = {"error": str(e)[:500]}
                    runs[p].append(r)
                    print(json.dumps({"case": case, "threads": t, "program": p, "rep": rep, **r}), flush=True)
            med = {p: statistics.median([r["time"] for r in rs if "time" in r]) for p, rs in runs.items() if any("time" in r for r in rs)}
            crow["runs"][str(t)] = {"median": med, "runs": runs}
            if a.rhf:
                rr = {}
                for p in [x for x in a.programs if x in ("linus", "pyscf")]:
                    try:
                        rr[p] = linus_rhf(mol, basis, t, a.thresh) if p == "linus" else pyscf_rhf(mol, basis, t, a.thresh)
                    except Exception as e:
                        rr[p] = {"error": str(e)[:500]}
                if "orca" in runs and runs["orca"] and "scf_wall" in runs["orca"][0]:
                    rr["orca"] = {"time": runs["orca"][0]["scf_wall"], "energy": runs["orca"][0]["energy"],
                                  "iterations": runs["orca"][0]["cycles"]}
                crow["rhf"][str(t)] = rr
                print(json.dumps({"case": case, "threads": t, "rhf": rr}), flush=True)
            res["cases"] = [c for c in res["cases"] if c is not crow] + [crow]
            res["environment_end"] = environment()
            out.write_text(json.dumps(res, indent=1))
    print(f"wrote {out}", file=sys.stderr)
    # summary table
    for c in res["cases"]:
        for t, r in c["runs"].items():
            m = r["median"]
            line = f'{c["molecule"]:12s} {c["basis"]:10s} {t:>3s} thr  ' + "  ".join(f"{p} {v:8.3f}s" for p, v in m.items())
            if "linus" in m:
                line += "  " + "  ".join(f"{p}/linus {v / m['linus']:5.1f}x" for p, v in m.items() if p != "linus")
            print(line, file=sys.stderr)


if __name__ == "__main__":
    main()
