#!/usr/bin/env python
"""Compare run_ref.py results between programs.

    .refenv/bin/python reference/compare_ref.py reference/out/water/hf_def2-svp_gradient_exact [...]
    .refenv/bin/python reference/compare_ref.py --markdown DIR [DIR ...]

For each result directory, prints every program's energy, iterations and wall time,
then for each pair of programs: |dE|, max |dG| (Eh/bohr), max |dH| (Eh/bohr^2),
|dVnn|, and each program's translational-invariance error max|sum_atoms g|.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

ORDER = ("pyscf", "psi4", "orca")


def load(d: Path) -> dict[str, dict]:
    out = {}
    for prog in ORDER:
        f = d / f"{prog}.json"
        if f.exists():
            rec = json.loads(f.read_text())
            if rec.get("success"):
                out[prog] = rec
            else:
                print(f"  {d.name}: {prog} failed: {rec.get('error')}")
    return out


def maxabs(a, b) -> float | None:
    if a is None or b is None:
        return None
    return float(np.abs(np.asarray(a) - np.asarray(b)).max())


def fmt(x: float | None) -> str:
    return "-" if x is None else f"{x:.1e}"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("dirs", nargs="+", type=Path)
    p.add_argument("--markdown", action="store_true", help="one markdown table row per program pair")
    a = p.parse_args()
    rows = []
    for d in a.dirs:
        recs = load(d)
        if not a.markdown:
            print(f"== {d}")
            for prog, r in recs.items():
                tinv = fmt(float(np.abs(r["gradient_sum"]).max())) if "gradient_sum" in r else "-"
                print(f"  {prog:6s} E={r['energy']:.12f} iters={r.get('scf_iterations')} "
                      f"wall={r['wall_time_s']:.2f}s Vnn-runner={fmt(abs(r.get('nuclear_repulsion_minus_runner') or 0))} "
                      f"|sum g|={tinv} v{r.get('program_version')}")
        for p1, p2 in itertools.combinations([x for x in ORDER if x in recs], 2):
            r1, r2 = recs[p1], recs[p2]
            dE = abs(r1["energy"] - r2["energy"])
            dG = maxabs(r1.get("gradient"), r2.get("gradient"))
            dH = maxabs(r1.get("hessian"), r2.get("hessian"))
            dV = abs((r1.get("nuclear_repulsion") or 0) - (r2.get("nuclear_repulsion") or 0))
            rows.append((d, f"{p1}-{p2}", dE, dG, dH, dV))
            if not a.markdown:
                print(f"  {p1:>5s}-{p2:<5s} |dE|={dE:.1e} max|dG|={fmt(dG)} max|dH|={fmt(dH)} |dVnn|={dV:.1e}")
    if a.markdown:
        print("| run | pair | abs dE (Eh) | max abs dG (Eh/bohr) | max abs dH (Eh/bohr^2) | abs dVnn |")
        print("|---|---|---|---|---|---|")
        for d, pair, dE, dG, dH, dV in rows:
            print(f"| {d.parent.name}/{d.name} | {pair} | {dE:.1e} | {fmt(dG)} | {fmt(dH)} | {dV:.1e} |")


if __name__ == "__main__":
    main()
