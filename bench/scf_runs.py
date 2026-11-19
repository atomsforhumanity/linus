"""Run linus SCF variants (JSON requests) and print iteration counts, energies and phase times.

usage: python bench/scf_runs.py --mol caffeine --method hf --basis def2-svp --jk exact \
           [--grid orca:2] [--threads 8] [--slice ccd3|bench|none] [--mem 8] \
           [--set key=value ...] [--var name:key=value,key=value ...] [--hist] [--out DIR]

Each --var adds one variant (settings merged into the request; keys with a dot go into a
nested object, e.g. jk.incremental=false). `--slice bench` runs under the shared bench lock
(timings); other slices run without the lock. Results (the full linus JSON) go to --out.
"""
import argparse, json, pathlib, subprocess, sys, time

ROOT = pathlib.Path(__file__).resolve().parents[1]
from bench_paths import LOCK  # noqa: E402  (env override BENCH_LOCK: bench/bench_paths.py)
MOLS = {p.stem: p for d in ("bench/molecules", "tests/molecules") for p in (ROOT / d).glob("*.xyz")}
CHARGES = {"u1an39": -1}


def parse_val(v):
    for f in (json.loads,):
        try:
            return f(v)
        except Exception:
            pass
    return v


def setkv(req, kv):
    k, v = kv.split("=", 1)
    *path, last = k.split(".")
    d = req
    for p in path:
        d = d.setdefault(p, {})
    d[last] = parse_val(v)


def run(req, slc, mem, prefix_lock):
    cmd = ["systemd-run", "--user", "--scope", "-q", "-p", f"MemoryMax={mem}G", "-p", "MemorySwapMax=0"]
    if slc != "none":
        cmd += [str(pathlib.Path.home() / ".local/bin/slice"), slc]
    cmd += [BIN[0], "--json"]
    if prefix_lock:
        cmd = ["flock", LOCK] + cmd
    t = time.perf_counter()
    p = subprocess.run(cmd, input=json.dumps(req), capture_output=True, text=True)
    wall = time.perf_counter() - t
    if p.returncode != 0:
        return None, wall, p.stderr[-3000:]
    if SHOW_STDERR and p.stderr:
        print(p.stderr, end="")
    return json.loads(p.stdout), wall, None


SHOW_STDERR = "--stderr" in sys.argv
BIN = [str(ROOT / "target/release/linus")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mol", required=True)
    ap.add_argument("--method", default="hf")
    ap.add_argument("--basis", default="def2-svp")
    ap.add_argument("--jk", default="exact")
    ap.add_argument("--aux", default=None)
    ap.add_argument("--grid", default=None)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--slice", default="ccd3")
    ap.add_argument("--mem", type=int, default=8)
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--var", action="append", default=[])
    ap.add_argument("--hist", action="store_true")
    ap.add_argument("--stderr", action="store_true", help="print linus stderr (verbose trace, diagnostics)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--calctype", default="energy")
    ap.add_argument("--bin", default=None, help="linus binary (default: this checkout's target/release/linus)")
    a = ap.parse_args()
    if a.bin:
        BIN[0] = a.bin
    base = {"structure": {"xyz": MOLS[a.mol].read_text(), "charge": CHARGES.get(a.mol, 0)}, "method": a.method,
            "basis": a.basis, "jk": {"kind": a.jk}, "threads": a.threads, "calctype": a.calctype}
    if a.aux:
        base["jk"]["aux"] = a.aux
    if a.grid:
        base["grid"] = {"xc": a.grid}
    for kv in a.set:
        setkv(base, kv)
    variants = a.var or ["base:"]
    out = pathlib.Path(a.out) if a.out else None
    if out:
        out.mkdir(parents=True, exist_ok=True)
    slist = subprocess.run([str(pathlib.Path.home() / ".local/bin/slice"), "list"], capture_output=True, text=True).stdout.split("\n")
    print("slice list:", "; ".join(x for x in slist if x))
    for v in variants:
        name, _, sets = v.partition(":")
        req = json.loads(json.dumps(base))
        for kv in filter(None, sets.split(",")):
            setkv(req, kv)
        d, wall, err = run(req, a.slice, a.mem, a.slice == "bench")
        if d is None:
            print(f"{name}: FAILED ({wall:.1f}s)\n{err}")
            continue
        if out:
            (out / f"{a.mol}_{a.method}_{a.basis}_{a.jk}_{name}.json").write_text(json.dumps(d))
        t = d["timings"]
        h = d["scf"]["history"]
        if a.hist:
            for r in h:
                print(f"  {r['iter']:3d} {r['energy']:22.12f} {r['de']:10.2e} g {r['gmax']:9.2e} rms {r['grms']:9.2e} "
                      f"dD {r['ddmax']:9.2e} {r['fock'][:4]:>4} {r['step']:>9} jk {r['t_jk']:6.3f} xc {r['t_xc']:6.3f}")
        gl = h[-1]
        print(f"{name:14s} E {d['energy']:.12f} conv {d['converged']} builds {d['scf']['fock_builds']:3d} "
              f"(incr {d['scf']['incremental_builds']:2d}) final gmax {gl['gmax']:.1e} | wall {wall:6.2f} scf {t.get('scf_total', 0):6.2f} "
              f"jk {t.get('jk', 0):6.2f} xc {t.get('xc', 0):6.2f} diag {t.get('diag', 0):5.2f} diis {t.get('diis', 0):5.2f} "
              f"soscf {t.get('soscf', 0):5.2f} setup {t.get('jk_setup', 0):5.2f}", flush=True)


if __name__ == "__main__":
    main()
