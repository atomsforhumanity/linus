"""RIJCOSX SCF convergence benchmark: linus binaries (e.g. before/after an SCF change) on RIJ+COSX
energy and gradient jobs at the default thresholds, plus untimed tightly converged references.

linus: jk rij-snk (def2-universal-jfit, COSX grids of DefGrid2: SCF on GridX1/GridX2, final
energy and gradient on GridX3), XC grid orca:2, 8 threads, default SCF thresholds by calctype
(energy: |dE| < 1e-8, max |F_ai| < 5e-7; gradient: 1e-9, 1e-7) unless --set overrides them.

Timed runs (`--slice bench`) take the shared bench lock per run (flock), record `slice list`
and the load average before each run, and write one JSON per case and binary to
bench/results/<tag>/ (no host names): wall time, SCF and gradient times, Fock builds, energy,
gradient, the SCF history and the request. `--ref` runs the reference instead: incremental
builds off (every COSX build a full one), max |F_ai| < 1e-9 and |dE| < 1e-11, untimed.

usage: python3 bench/cosx_scf_bench.py --tag 2026-11-07-cosx-scf --bins old=PATH,new=PATH \
           --cases ts61:b3lyp:tzvp:energy,ts61:b3lyp:tzvp:gradient [--slice bench] [--ref] [--set jk.grid_switch=1e-3]
"""
import argparse, contextlib, fcntl, json, os, pathlib, subprocess, time

ROOT = pathlib.Path(__file__).resolve().parents[1]
MOLS = ROOT / "bench/molecules"
from bench_paths import LOCK  # noqa: E402  (env override BENCH_LOCK: bench/bench_paths.py)
BOHR = 0.529177210544
CHARGES = {"la18": 0, "u1an39": -1, "ts61": 0, "ts113": 0}
BASIS = {"svp": "def2-svp", "tzvp": "def2-tzvp"}


@contextlib.contextmanager
def held(lock):
    if not lock:
        yield
        return
    with open(lock, "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def setkv(req, kv):
    k, v = kv.split("=", 1)
    try:
        v = json.loads(v)
    except ValueError:
        pass
    *path, last = k.split(".")
    d = req
    for p in path:
        d = d.setdefault(p, {})
    d[last] = v


def request(mol, func, basis, calc, sets, ref):
    lines = (MOLS / f"{mol}.xyz").read_text().splitlines()
    n = int(lines[0])
    atoms = [(l.split()[0], [float(x) for x in l.split()[1:4]]) for l in lines[2:2 + n]]
    req = {"structure": {"symbols": [a for a, _ in atoms], "geometry": [[x / BOHR for x in c] for _, c in atoms],
                         "charge": CHARGES[mol], "multiplicity": 1},
           "calctype": calc, "threads": 8, "method": func, "basis": BASIS[basis],
           "jk": {"kind": "rij-snk", "aux": "def2-universal-jfit", "k_grid": "cosx:2"}, "grid": {"xc": "orca:2"}}
    if ref:
        req["scf"] = {"e_tol": 1e-11, "g_tol": 1e-9}
        req["jk"]["incremental"] = False
    for kv in sets:
        setkv(req, kv)
    return req


def run(binary, req, slc, lock, mem, timeout=None):
    cmd = ["systemd-run", "--user", "--scope", "-q", "-p", f"MemoryMax={mem}G", "-p", "MemorySwapMax=0", "-p", "OOMPolicy=continue"]
    cmd += ["taskset", "-c", slc[5:]] if slc.startswith("cpus:") else ([] if slc == "none" else [str(pathlib.Path.home() / ".local/bin/slice"), slc])
    with held(lock):
        slist = subprocess.run([str(pathlib.Path.home() / ".local/bin/slice"), "list"], capture_output=True, text=True).stdout
        load = os.getloadavg()
        t = time.perf_counter()
        try:
            p = subprocess.run(cmd + [binary, "--json"], input=json.dumps(req), capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            p = None
        wall = time.perf_counter() - t
    res = dict(wall=wall, slice=slc, slice_list=slist.strip().splitlines(), load=list(load), time=time.strftime("%Y-%m-%dT%H:%M:%S"))
    if p is None:
        return dict(ok=False, error=f"stopped after the time cap of {timeout} s", **res)
    if p.returncode != 0:
        return dict(ok=False, error=p.stderr[-2000:], **res)
    d = json.loads(p.stdout)
    t = d.get("timings", {})
    # timings.energy: the SCF stage, timings.total: the whole call (before: total and calculation)
    scf = t.get("energy", t.get("total"))
    end = t.get("calculation", t.get("total")) if "energy" not in t else t.get("total")
    grad = end - scf if req["calctype"] == "gradient" and scf is not None and end is not None and ("energy" in t or "calculation" in t) else None
    h = d["scf"]["history"]
    return dict(ok=True, energy=d["energy"], gradient=d.get("gradient"), builds=d["scf"]["fock_builds"], converged=d.get("converged"),
                final_gmax=h[-1]["gmax"], scf_time=scf, grad_time=grad, timings=t, history=h, **res)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--bins", required=True)
    ap.add_argument("--cases", required=True)
    ap.add_argument("--slice", default="bench")
    ap.add_argument("--mem", type=int, default=48)
    ap.add_argument("--ref", action="store_true")
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--timeout", type=float, default=None, help="time cap per run (s); a capped run is recorded as not ok")
    a = ap.parse_args()
    bins = dict(kv.split("=", 1) for kv in a.bins.split(","))
    out = ROOT / "bench/results" / a.tag
    out.mkdir(parents=True, exist_ok=True)
    lock = LOCK if a.slice == "bench" and not a.ref else None
    for case in a.cases.split(","):
        mol, func, basis, calc = case.split(":")
        req = request(mol, func, basis, calc, a.set, a.ref)
        for name, b in bins.items():
            label = ("ref-" if a.ref else "") + name
            fn = out / f"{mol}_{func}_{basis}_{calc}_{label}.json"
            r = run(b, req, a.slice, lock, a.mem, a.timeout)
            rec = dict(case=case, program=label, binary=b, request={**req, "structure": {k: v for k, v in req["structure"].items() if k not in ("symbols", "geometry")}},
                       timed=lock is not None, conditions=__doc__.strip(), run=r)
            fn.write_text(json.dumps(rec, indent=1))
            print(f"{case:32s} {label:10s} ok {r['ok']} builds {r.get('builds')} wall {r['wall']:.1f} E {r.get('energy', float('nan')):.10f} gmax {r.get('final_gmax', 0):.1e}", flush=True)


if __name__ == "__main__":
    main()
