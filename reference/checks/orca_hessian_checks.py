"""Two ORCA 6.0.0 analytic-Hessian findings, water/def2-SVP (tests/molecules/water.xyz), with
run_ref.py's tight settings (bohr geometry, Thresh 1e-14, TolE 1e-12, Z_Tol 1e-10, IntAcc 8,
Lebedev-770, unpruned) and NumFreq increment 0.002 bohr:

  A. analytic Hessians of non-hybrid functionals with exact Coulomb (NoRI) are wrong by ~5e-2
     Eh/bohr^2; with RI-J, and for hybrids, analytic and NumFreq agree.
  B. the analytic DFT Hessian's CP-SCF grid follows DefGrid, not IntAcc/AngularGrid: with
     the fine grid settings alone analytic vs NumFreq differ by 1.8e-4 (DefGrid1/2 too), with
     DefGrid3 2e-6. linus's Hessian with first-order grid response (hess_grid_response, Bykov
     et al. 2015 scheme 1) and without it both agree with NumFreq to ~5e-6 on PySCF's level-9
     grid, so grid-response approximations do not explain the 1.8e-4.

Run with the reference env from the repository (ORCA, slice ccd2, 4 processes; ~10 min):
    .refenv/bin/python reference/checks/orca_hessian_checks.py [--part A|B] [--threads 4]
Writes reference/out/checks/orca_hessian/ (git-ignored) and prints the tables.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "reference"))
import run_ref  # noqa: E402

OUT = ROOT / "reference" / "out" / "checks" / "orca_hessian"
XYZ = ROOT / "tests" / "molecules" / "water.xyz"
PY = str(run_ref.REFENV / "bin" / "python")
LINUS = ROOT / "target" / "release" / "linus"


def orca_run(name: str, method: str, approx: str, mode: str, threads: int) -> dict:
    """run_ref.py ORCA Hessian (analytic or NumFreq)."""
    d = OUT / name
    if not (d / "orca.json").exists():
        cmd = [PY, str(ROOT / "reference/run_ref.py"), str(XYZ), "--method", method, "--basis", "def2-svp",
               "--calc", "hessian", "--programs", "orca", "--threads", str(threads), "--memory-gb", "2",
               "--grid", "fine", "--approx", approx, "--orca-hessian", mode, "--orca-fd-step", "0.002",
               "--rundir", str(d)]
        subprocess.run(cmd, check=True, capture_output=True)
    return json.loads((d / "orca.json").read_text())


def orca_patched(name: str, defgrid: str) -> tuple[np.ndarray, Path]:
    """Rerun the B3LYP analytic input with its DefGrid3 word replaced by `defgrid`."""
    d = OUT / name
    if not (d / "orca.hess").exists():
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)
        srcdir = OUT / "B_b3lyp_analytic_defgrid3" / "orca_run"
        inp = (srcdir / "orca.inp").read_text().replace("DefGrid3", defgrid)
        (d / "orca.inp").write_text(inp)
        shutil.copy(srcdir / "geometry.xyz", d / "geometry.xyz")
        env = {**os.environ, **run_ref.orca_environment()}
        with open(d / "orca.out", "w") as f:
            subprocess.run([str(run_ref.ORCA_DIR / "orca"), "orca.inp"], cwd=d, env=env, stdout=f, stderr=subprocess.STDOUT)
    from qccodec.programs import orca as oc
    return np.asarray(oc.parse_hessian((d / "orca.hess").read_text())), d / "orca.out"


def freqs(out_file: Path) -> list[float]:
    return [float(x) for x in re.findall(r"^\s*\d+:\s+(-?\d+\.\d+) cm\*\*-1", out_file.read_text(), re.M)][-3:]


def part_a(threads: int) -> None:
    print("A. ORCA 6.0.0 analytic Hessian vs NumFreq (central differences of analytic gradients)")
    print(f"{'method':8s} {'J/K':12s} {'max|dH| Eh/bohr^2':>18s}   ORCA frequencies (cm-1), analytic | NumFreq")
    for method, approx, label in (("pbe", "exact", "NoRI"), ("pbe", "rij", "RI-J def2/J"), ("svwn-rpa", "exact", "NoRI"),
                                  ("tpss", "exact", "NoRI"), ("pbe0", "exact", "NoRI NoCOSX"), ("b3lyp", "exact", "NoRI NoCOSX"),
                                  ("hf", "exact", "NoRI NoCOSX")):
        H, fr = {}, {}
        for mode in ("analytic", "numfreq"):
            name = f"A_{method}_{approx}_{mode}"
            r = orca_run(name, method, approx, mode, threads)
            H[mode] = np.asarray(r["hessian"])
            fr[mode] = freqs(OUT / name / "orca_run" / "orca.out")
        print(f"{method:8s} {label:12s} {np.abs(H['analytic'] - H['numfreq']).max():18.2e}   {fr['analytic']} | {fr['numfreq']}")
    print("simple-input line of the PBE NoRI analytic run:",
          [l for l in (OUT / "A_pbe_exact_analytic/orca_run/orca.inp").read_text().splitlines() if l.startswith("!")])


def linus_hessian(grid: str, hgr: bool, threads: int, gradient_only=False, positions=None) -> dict:
    syms, xyz = run_ref.read_xyz(XYZ)[:2]
    from qcref import constants
    pos = (np.asarray(xyz) * float(constants.ANGSTROM_TO_BOHR)).tolist() if positions is None else positions
    # one key, grid.response: off for the gradient (the finite differences below), the Hessian's own for a
    # Hessian (D41: no response settings for a gradient)
    req = {"structure": {"symbols": syms, "geometry": pos}, "calctype": "gradient" if gradient_only else "hessian",
           "method": "b3lyp", "basis": "def2-svp", "threads": threads, "scf": {"convergence": "extreme"},
           "grid": {"xc": grid, "response": False if gradient_only else hgr},
           "jk": {"screening": 1e-14}}
    if not gradient_only:
        req["response"] = {"tol": 1e-9}
    p = subprocess.run([str(LINUS), "--json"], input=json.dumps(req), capture_output=True, text=True,
                       check=True)
    return json.loads(p.stdout)


def part_b(threads: int) -> None:
    print("B. B3LYP/G: ORCA analytic Hessian (main grid IntAcc 8, Lebedev-770, unpruned) vs NumFreq (same grid)")
    an = orca_run("B_b3lyp_analytic_defgrid3", "b3lyp", "exact", "analytic", threads)  # run_ref adds DefGrid3
    nf = np.asarray(orca_run("B_b3lyp_numfreq", "b3lyp", "exact", "numfreq", threads)["hessian"])
    rows = {"DefGrid3 (run_ref)": (np.asarray(an["hessian"]), OUT / "B_b3lyp_analytic_defgrid3/orca_run/orca.out")}
    for dg in ("", "DefGrid1", "DefGrid2"):
        rows[dg or "no DefGrid word"] = orca_patched(f"B_b3lyp_analytic_{dg or 'none'}", dg)
    for k, (hk, out) in rows.items():
        npts = re.findall(r"Total number of grid points\s+\.\.\.\s+(\d+)", out.read_text())
        print(f"  {k:20s} main-grid points {npts[0] if npts else '?':>7s}: max|analytic - NumFreq| = {np.abs(hk - nf).max():.2e}")
    rows = {k: v[0] for k, v in rows.items()}
    print("linus, PySCF level-9 grid (Becke, NWChem pruning), B3LYP/def2-SVP:")
    hfix = np.asarray(linus_hessian("pyscf:9", False, threads)["hessian"])
    hgr = np.asarray(linus_hessian("pyscf:9", True, threads)["hessian"])
    n3 = hfix.shape[0] if hfix.ndim == 2 else int(round(np.sqrt(hfix.size)))
    hfix, hgr = hfix.reshape(n3, n3), hgr.reshape(n3, n3)
    # hess_grid_response = central differences of gradients without grid response (grid moving with the atoms)
    syms, xyz = run_ref.read_xyz(XYZ)[:2]
    from qcref import constants
    pos0 = np.asarray(xyz) * float(constants.ANGSTROM_TO_BOHR)
    h = 0.005
    J = np.zeros((n3, n3))
    for k in range(n3):
        a, c = divmod(k, 3)
        g = []
        for s in (-2, -1, 1, 2):
            p = pos0.copy(); p[a, c] += s * h
            g.append(np.asarray(linus_hessian("pyscf:9", False, threads, True, p.tolist())["gradient"]).ravel())
        J[:, k] = (g[0] - 8 * g[1] + 8 * g[2] - g[3]) / (12 * h)
    print(f"  hess_grid_response vs 5-point differences of no-grid-response gradients: {np.abs(hgr - 0.5 * (J + J.T)).max():.2e}")
    for label, hl in (("fixed grid (PySCF convention)", hfix), ("hess_grid_response (Bykov scheme 1)", hgr)):
        print(f"  linus {label:36s} vs ORCA NumFreq {np.abs(hl - nf).max():.2e}, vs ORCA analytic with DefGrid3 "
              f"{np.abs(hl - rows['DefGrid3 (run_ref)']).max():.2e}, with the fine grid alone {np.abs(hl - rows['no DefGrid word']).max():.2e}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--part", choices=("A", "B", "both"), default="both")
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.part in ("A", "both"):
        part_a(a.threads)
    if a.part in ("B", "both"):
        part_b(a.threads)


if __name__ == "__main__":
    main()
