"""Gradient (and energy) error of linus SCF convergence settings against a tightly converged
reference: which thresholds do analytic gradients need?

usage: python bench/conv_accuracy.py --mol la18 --method b3lyp --basis def2-svp --jk df \
           [--grid orca:2] [--threads 8] [--slice ccd3] [--mem 12] [--out results.json]

Runs calctype=gradient for every setting in SETTINGS (preset names or e_tol/g_tol pairs) and
for the reference (g_tol 1e-10 on max |F_ai|; e_tol 1e-9, as the energy is then converged far below
the ~1e-11 Eh screening noise of direct J/K). Reports |dE|, max |dG| (Eh/bohr),
Fock builds and the final orbital gradient. Settings run with the default SCF algorithm.
"""
import argparse, json, pathlib, subprocess, time
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
MOLS = {p.stem: p for d in ("bench/molecules", "tests/molecules") for p in (ROOT / d).glob("*.xyz")}
CHARGES = {"u1an39": -1}
SETTINGS = [
    ("normal", {"scf": {"convergence": "normal"}}),
    ("e1e-7/g1e-5", {"scf": {"e_tol": 1e-7, "g_tol": 1e-5}}),
    ("e1e-8/g1e-6", {"scf": {"e_tol": 1e-8, "g_tol": 1e-6}}),
    ("tight", {"scf": {"convergence": "tight"}}),
    ("e1e-9/g1e-7", {"scf": {"e_tol": 1e-9, "g_tol": 1e-7}}),
    ("verytight", {"scf": {"convergence": "verytight"}}),
    ("e1e-10/g1e-8", {"scf": {"e_tol": 1e-10, "g_tol": 1e-8}}),
    ("pyscf", {"scf": {"convergence": "pyscf"}}),
]
REF = ("ref e1e-9/g1e-10", {"scf": {"e_tol": 1e-9, "g_tol": 1e-10, "max_iter": 200}})


def run(req, slc, mem):
    cmd = ["systemd-run", "--user", "--scope", "-q", "-p", f"MemoryMax={mem}G", "-p", "MemorySwapMax=0"]
    if slc != "none":
        cmd += [str(pathlib.Path.home() / ".local/bin/slice"), slc]
    cmd += [str(ROOT / "target/release/linus"), "--json"]
    t = time.perf_counter()
    p = subprocess.run(cmd, input=json.dumps(req), capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(p.stderr[-2000:])
    d = json.loads(p.stdout)
    d["wall"] = time.perf_counter() - t
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mol", required=True)
    ap.add_argument("--method", default="hf")
    ap.add_argument("--basis", default="def2-svp")
    ap.add_argument("--jk", default="exact")
    ap.add_argument("--grid", default=None)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--slice", default="ccd3")
    ap.add_argument("--mem", type=int, default=12)
    ap.add_argument("--out", default=None)
    ap.add_argument("--set", action="append", default=[], help="extra request key=json for every run")
    a = ap.parse_args()
    base = {"structure": {"xyz": MOLS[a.mol].read_text(), "charge": CHARGES.get(a.mol, 0)}, "method": a.method,
            "basis": a.basis, "jk": a.jk, "threads": a.threads, "calctype": "gradient"}
    if a.grid:
        base["grid"] = {"xc": a.grid}
    for kv in a.set:
        k, v = kv.split("=", 1)
        base[k] = json.loads(v)
    ref = run({**base, **REF[1]}, a.slice, a.mem)
    gref = np.array(ref["gradient"])
    rows = [dict(name=REF[0], energy=ref["energy"], builds=ref["scf"]["fock_builds"], gmax=ref["scf"]["history"][-1]["gmax"],
                 dE=0.0, dG=0.0, wall=ref["wall"])]
    print(f"{a.mol} {a.method}/{a.basis} jk={a.jk}: reference E {ref['energy']:.12f}, {rows[0]['builds']} builds, "
          f"|G|max {np.abs(gref).max():.2e}", flush=True)
    for name, s in SETTINGS:
        d = run({**base, **s}, a.slice, a.mem)
        g = np.array(d["gradient"])
        r = dict(name=name, energy=d["energy"], builds=d["scf"]["fock_builds"], gmax=d["scf"]["history"][-1]["gmax"],
                 gpyscf=d["scf"]["history"][-1]["gnorm_pyscf"], dE=d["energy"] - ref["energy"], dG=float(np.abs(g - gref).max()),
                 wall=d["wall"], convergence=d["scf"]["convergence"])
        rows.append(r)
        print(f"  {name:14s} builds {r['builds']:3d}  final max|F_ai| {r['gmax']:.1e}  dE {r['dE']:+.2e}  max|dG| {r['dG']:.2e}", flush=True)
    if a.out:
        pathlib.Path(a.out).write_text(json.dumps(dict(mol=a.mol, method=a.method, basis=a.basis, jk=a.jk, grid=a.grid, rows=rows), indent=1))


if __name__ == "__main__":
    main()
