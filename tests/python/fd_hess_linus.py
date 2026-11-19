"""Analytic linus Hessians against Richardson-extrapolated central differences of linus's own
analytic gradients (steps h and 2h, error O(h^4)), through the `linus --json` CLI.

  python tests/python/fd_hess_linus.py <xyz> <basis> <method> [--jk exact|df] [--aux A]
      [--charge Q] [--mult M] [--h 1e-3] [--threads N] [--grid pyscf:3] [--fixed-grid]
      [--atoms 0,1] [--bin target/release/linus] [--solvent '{"eps": 7.4257, "model": "cpcm"}']

KS conventions: by default the Hessian has the first-order grid response (linus's default,
ORCA's analytic convention), which is the derivative of the gradient WITHOUT grid response on a
grid that moves with the atoms, so the differences use such gradients; --fixed-grid compares the
fixed-grid Hessian (PySCF's convention) with differences of full-grid-response gradients (the
two then differ by the grid's second-order terms; informational). All SCFs use the `extreme`
preset (|dE| < 1e-12, max |F_ai| < 1e-10) and full (not incremental) Fock builds. Prints one JSON
line: max |H - H_fd| over the displaced columns (H_fd symmetrized when all atoms are displaced)
and the Hessian's largest element.

KS with --atoms: the gradient without grid response is not the gradient of any energy, so its
finite-difference Jacobian is not symmetric, and the default Hessian is its SYMMETRIZED
derivative. Partial columns cannot be symmetrized, so they differ from H by half the
asymmetric part, a grid-size quantity (CH3SH def2-SVP RI on PySCF level 3, atom 0: wB97X 3e-4,
wB97X-D3BJ 2e-5, against 3e-10 / 5e-10 with all atoms). Such runs report "symmetrized": false
and a warning; use all atoms for KS (HF has no grid: partial columns are exact).
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
    ap.add_argument("--h", type=float, default=1e-3)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--grid", default="pyscf:3")
    ap.add_argument("--fixed-grid", action="store_true")
    ap.add_argument("--atoms", default=None, help="comma-separated atom indices to displace (default: all)")
    ap.add_argument("--bin", default=str(ROOT / "target/release/linus"))
    ap.add_argument("--solvent", default=None, help="implicit solvent: the driver's SolventSettings as JSON")
    a = ap.parse_args()
    lines = pathlib.Path(a.xyz).read_text().splitlines()
    n = int(lines[0])
    syms = [ln.split()[0] for ln in lines[2:2 + n]]
    pos = np.array([[float(v) / BOHR for v in ln.split()[1:4]] for ln in lines[2:2 + n]])
    hgr = not a.fixed_grid

    def run(p, calc):
        req = {"structure": {"symbols": syms, "geometry": p.tolist(), "charge": a.charge, "multiplicity": a.mult},
               "calctype": calc, "method": a.method, "basis": a.basis, "threads": a.threads,
               "jk": {"kind": a.jk, "incremental": False}, "scf": {"convergence": "extreme"}}
        if a.jk == "df":
            req["jk"]["aux"] = a.aux
        # one key, grid.response: the Hessian's grid response for a Hessian, a gradient's (the opposite of the
        # Hessian's, as before) for the finite-difference gradients; the kernel grid is a Hessian's
        if a.method.lower() != "hf":
            req["grid"] = {"xc": a.grid, "response": hgr if calc == "hessian" else not hgr}
            if calc == "hessian":
                req["grid"]["kernel"] = a.grid
        if a.solvent:
            req["solvent"] = json.loads(a.solvent)
        out = subprocess.run([a.bin, "--json"], input=json.dumps(req), capture_output=True, text=True, check=True).stdout
        return json.loads(out)

    r = run(pos, "hessian")
    h = np.array(r["hessian"]).reshape(3 * n, 3 * n)
    atoms = list(range(n)) if a.atoms is None else [int(x) for x in a.atoms.split(",")]
    fd = np.full_like(h, np.nan)
    for i in atoms:
        for k in range(3):
            g = {}
            for s in (-2, -1, 1, 2):
                p = pos.copy()
                p[i, k] += s * a.h
                g[s] = np.array(run(p, "gradient")["gradient"]).ravel()
            fd[:, 3 * i + k] = (8 * (g[1] - g[-1]) - (g[2] - g[-2])) / (12 * a.h)
    cols = [3 * i + k for i in atoms for k in range(3)]
    if len(atoms) == n:
        fd = 0.5 * (fd + fd.T)
    err = float(np.abs(h[:, cols] - fd[:, cols]).max())
    ks = a.method.lower() != "hf"
    extra = {"symmetrized": len(atoms) == n}
    if ks and hgr and len(atoms) < n:
        extra["warning"] = "KS partial columns are unsymmetrized: grid-size asymmetry, not an error; displace all atoms for a precise check"
    print(json.dumps({**extra, "xyz": pathlib.Path(a.xyz).name, "basis": a.basis, "method": a.method, "jk": a.jk, "mult": a.mult, "charge": a.charge, "h": a.h,
                      "grid": a.grid, "hess_grid_response": hgr, "solvent": a.solvent, "max_abs_dH_fd": err, "max_abs_H": float(np.abs(h).max()),
                      "cphf_iterations": r.get("cphf_iterations")}), flush=True)


if __name__ == "__main__":
    main()
