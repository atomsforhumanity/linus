"""Analytic Hessian error of linus SCF convergence thresholds against a tightly converged
reference (g_conv 1e-10): which orbital-gradient threshold do Hessians need?

usage: python bench/hess_conv_accuracy.py <xyz> <basis> <method> [--jk df] [--threads 8]
           [--slice ccd3] [--mem 16] [--g 1e-6,1e-7,1e-8,1e-9] [--out results.json]
"""
import argparse, json, pathlib, subprocess, time
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]


def run(a, e, g):
    cmd = ["systemd-run", "--user", "--scope", "-q", "-p", f"MemoryMax={a.mem}G", "-p", "MemorySwapMax=0",
           str(pathlib.Path.home() / ".local/bin/slice"), a.slice, str(ROOT / "target/release/examples/hess"), a.xyz, a.basis,
           "--method", a.method, "--threads", str(a.threads), "--e-conv", str(e), "--g-conv", str(g)]
    if a.jk:
        cmd += ["--jk", a.jk]
    t = time.perf_counter()
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(p.stderr[-2000:])
    d = json.loads(p.stdout)
    d["wall"] = time.perf_counter() - t
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("xyz")
    ap.add_argument("basis")
    ap.add_argument("method")
    ap.add_argument("--jk", default=None)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--slice", default="ccd3")
    ap.add_argument("--mem", type=int, default=16)
    ap.add_argument("--g", default="1e-6,1e-7,3e-8,1e-8,3e-9,1e-9")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    ref = run(a, 1e-9, 1e-10)
    href = np.array(ref["hessian"])
    print(f"{pathlib.Path(a.xyz).stem} {a.method}/{a.basis} jk={a.jk or 'exact'}: reference g_conv 1e-10, |H|max {np.abs(href).max():.2e}", flush=True)
    rows = []
    for g in [float(x) for x in a.g.split(",")]:
        e = 1e-9  # the gradient criterion dominates
        d = run(a, e, g)
        dh = float(np.abs(np.array(d["hessian"]) - href).max())
        rows.append(dict(g_conv=g, e_conv=e, dH=dh, dE=d["energy"] - ref["energy"], cphf=d["cphf_iterations"], wall=d["wall"],
                         scf_builds=d.get("scf_builds"), scf_gmax=d.get("scf_gmax")))
        print(f"  g_conv {g:.0e}: SCF builds {d.get('scf_builds')} final max|F_ai| {d.get('scf_gmax', 0):.1e}  max|dH| {dh:.2e} Eh/bohr^2  dE {d['energy'] - ref['energy']:+.1e}  ({d['wall']:.1f}s)", flush=True)
    if a.out:
        pathlib.Path(a.out).write_text(json.dumps(dict(xyz=a.xyz, basis=a.basis, method=a.method, jk=a.jk, rows=rows), indent=1))


if __name__ == "__main__":
    main()
