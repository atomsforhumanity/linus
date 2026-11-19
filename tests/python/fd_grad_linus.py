"""Analytic linus gradients against Richardson-extrapolated central differences of linus's own
energies (h and 2h, error O(h^4)), through the `linus --json` CLI.

  python tests/python/fd_grad_linus.py <xyz> <basis> <method> [--jk exact|df|rij-snk] [--aux A]
      [--charge Q] [--mult M] [--h 2e-3] [--threads N] [--grid pyscf:3] [--k-grid G] [--atoms 0,1]
      [--solvent '{"eps": 7.4257, "model": "cpcm"}']

The SCFs use the `extreme` preset (|dE| < 1e-12, max |F_ai| < 1e-10). Prints one JSON line with
the largest deviation and the analytic gradient's largest element.
"""
import argparse, json, pathlib, subprocess

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
BOHR = 0.529177210544


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("xyz")
    ap.add_argument("basis")
    ap.add_argument("method")
    ap.add_argument("--jk", default="exact")
    ap.add_argument("--aux", default="def2-universal-jkfit")
    ap.add_argument("--charge", type=int, default=0)
    ap.add_argument("--mult", type=int, default=1)
    ap.add_argument("--h", type=float, default=2e-3)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--grid", default="pyscf:3")
    ap.add_argument("--k-grid", default=None, help="rij-snk exchange grid (e.g. cosx:2, pyscf:3)")
    ap.add_argument("--atoms", default=None, help="comma-separated atom indices to displace (default: all)")
    ap.add_argument("--solvent", default=None, help="implicit solvent: the driver's SolventSettings as JSON")
    ap.add_argument("--bin", default=str(ROOT / "target/release/linus"))
    a = ap.parse_args()
    lines = pathlib.Path(a.xyz).read_text().splitlines()
    n = int(lines[0])
    syms = [ln.split()[0] for ln in lines[2:2 + n]]
    pos = np.array([[float(v) / BOHR for v in ln.split()[1:4]] for ln in lines[2:2 + n]])

    def run(p, calc):
        req = {"structure": {"symbols": syms, "geometry": p.tolist(), "charge": a.charge, "multiplicity": a.mult},
               "calctype": calc, "method": a.method, "basis": a.basis, "jk": a.jk,
               "threads": a.threads, "scf": {"convergence": "extreme"}}
        if a.method.lower() != "hf":
            req["grid"] = {"xc": a.grid}
        if a.jk in ("df", "rij-snk"):
            req["jk"] = {"kind": a.jk, "aux": a.aux}
            if a.k_grid:
                req["jk"]["k_grid"] = a.k_grid
        if a.solvent:
            req["solvent"] = json.loads(a.solvent)
        out = subprocess.run([a.bin, "--json"], input=json.dumps(req), capture_output=True, text=True, check=True).stdout
        return json.loads(out)

    g = np.array(run(pos, "gradient")["gradient"])
    atoms = range(n) if a.atoms is None else [int(x) for x in a.atoms.split(",")]
    worst = 0.0
    fd = np.full_like(g, np.nan)
    for i in atoms:
        for k in range(3):
            e = {}
            for s in (-2, -1, 1, 2):
                p = pos.copy()
                p[i, k] += s * a.h
                e[s] = run(p, "energy")["energy"]
            fd[i, k] = (8 * (e[1] - e[-1]) - (e[2] - e[-2])) / (12 * a.h)
            worst = max(worst, abs(fd[i, k] - g[i, k]))
    print(json.dumps({"xyz": pathlib.Path(a.xyz).name, "basis": a.basis, "method": a.method, "jk": a.jk, "mult": a.mult, "h": a.h, "solvent": a.solvent,
                      "max_abs_dG_fd": worst, "max_abs_G": float(np.abs(g).max())}), flush=True)


if __name__ == "__main__":
    main()
