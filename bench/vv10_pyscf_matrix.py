"""Runs tests/python/compare_vv10_pyscf.py over the VV10 validation matrix and appends one JSON
line per case to bench/results/2026-11-07-vv10/pyscf_validation.jsonl.

  .refenv/bin/python bench/vv10_pyscf_matrix.py [--set svp|tzvpd|df|nogr|big|all] [--slice ccd2] [--jobs 2]

Each case runs in a memory-capped systemd scope under `slice` (8 threads).
"""
import argparse, json, pathlib, subprocess, sys
from concurrent.futures import ThreadPoolExecutor

ROOT = pathlib.Path(__file__).resolve().parents[1]
from bench_paths import MAIN, REFENV  # noqa: E402,F401  (env overrides: bench/bench_paths.py)
PY = str(REFENV / "bin/python")
OUT = ROOT / "bench/results/2026-11-07-vv10/pyscf_validation.jsonl"
MOL = {
    "water": (ROOT / "tests/molecules/water.xyz", 0, 1),
    "nh3": (ROOT / "tests/molecules/nh3.xyz", 0, 1),
    "ch3sh": (ROOT / "tests/molecules/ch3sh.xyz", 0, 1),
    "ch3sh+": (ROOT / "tests/molecules/ch3sh.xyz", 1, 2),
    "acetate": (ROOT / "bench/results/2026-11-07-smd-cds/molecules/acetate.xyz", -1, 1),
    "la18": (ROOT / "bench/molecules/la18.xyz", 0, 1),
    "u1an39": (ROOT / "bench/molecules/u1an39.xyz", -1, 1),
}
FUNCS = ["wb97m-v", "wb97x-v", "b97m-v"]
SMALL = ["water", "nh3", "ch3sh", "ch3sh+", "acetate"]


def cases(which):
    c = []
    if which in ("svp", "all"):
        c += [(m, "def2-svp", f, []) for m in SMALL + ["la18"] for f in FUNCS]
    if which in ("tzvpd", "all"):
        c += [(m, "def2-tzvpd", f, []) for m in SMALL for f in FUNCS]
    if which in ("df", "all"):
        c += [(m, "def2-svp", f, ["--df"]) for m in ["water", "ch3sh", "ch3sh+", "acetate", "la18"] for f in FUNCS]
        c += [(m, "def2-tzvpd", f, ["--df"]) for m in ["water", "acetate"] for f in FUNCS]
    if which in ("nogr", "all"):
        c += [(m, "def2-svp", f, ["--no-grid-response"]) for m in ["water", "ch3sh+", "acetate"] for f in ["wb97m-v", "b97m-v"]]
    if which in ("big", "all"):
        c += [("u1an39", "def2-svp", "wb97m-v", ["--df", "--energy-only"]), ("u1an39", "def2-svp", "b97m-v", ["--df"])]
    return c


def run(case, a):
    mol, basis, func, extra = case
    xyz, charge, mult = MOL[mol]
    cmd = ["systemd-run", "--user", "--scope", "-q", "-p", f"MemoryMax={a.mem}G", "-p", "MemorySwapMax=0", "-p", "OOMPolicy=continue",
           str(pathlib.Path.home() / ".local/bin/slice"), a.slice, PY, str(ROOT / "tests/python/compare_vv10_pyscf.py"), str(xyz), basis, func,
           "--charge", str(charge), "--mult", str(mult), "--threads", "8"] + extra
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        r = {"xyz": xyz.name, "basis": basis, "method": func, "extra": extra, "error": (p.stderr or p.stdout)[-1500:]}
    else:
        r = json.loads(p.stdout.strip().splitlines()[-1])
    r["mol"] = mol
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="all")
    ap.add_argument("--slice", default="ccd2")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--mem", type=int, default=24)
    a = ap.parse_args()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(a.jobs) as ex, open(OUT, "a") as f:
        for r in ex.map(lambda c: run(c, a), cases(a.set)):
            f.write(json.dumps(r) + "\n")
            f.flush()
            if "error" in r:
                print(f"{r['mol']:8} {r['basis']:10} {r['method']:8} {r.get('extra')} ERROR {r['error'][-300:]}", flush=True)
            else:
                print(f"{r['mol']:8} {r['basis']:10} {r['method']:8} {r['jk']:5} gr={r['grid_response']!s:5} dE={r['dE']:+.2e} dEnl={r.get('dE_vv10', float('nan')):+.2e} "
                      f"dG={r.get('max_abs_dG', float('nan')):.2e}", flush=True)


if __name__ == "__main__":
    main()
