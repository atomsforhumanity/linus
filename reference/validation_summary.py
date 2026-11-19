#!/usr/bin/env python
"""Print the validation matrix from reference/validation/results.json.

    python reference/validation_summary.py [results.json] [--cases PATTERN ...] [--markdown]
        [--extras] [--problems]

One row per case and quantity (E energy, D dipole, G gradient, H Hessian, F harmonic frequencies); the columns are
max |difference| between

    L-P  linus and PySCF        (same grid for DFT: PySCF level N = linus pyscf:N; 9, Hessians 5)
    L-S  linus and Psi4         (same grid for DFT: Psi4's default grid = linus psi4)
    L-O  linus and ORCA         (DFT: different grids; "grid" is |PySCF - ORCA|, the residual)
    P-S, P-O, S-O               the references among themselves (DFT: PySCF level N, Psi4
                                150 x 974, ORCA finest grid)

Markers: no mark = within tolerance, "*" = above tolerance with a known explanation
(reference/validation/README.md), "!" = above tolerance, unexplained; "n/i" = not
implemented in linus, "n/a" = the program cannot do it here (ORCA: Cartesian basis sets; PySCF-only cases), "-" =
not computed. Units: Eh, e bohr, Eh/bohr, Eh/bohr^2, cm^-1. Needs only the standard library.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import sys
from pathlib import Path

DEFAULT = Path(__file__).resolve().parent / "validation" / "results.json"
COLS = ("L-P", "L-S", "L-O", "P-S", "P-O", "S-O")
QLAB = {"energy": "E", "dipole": "D", "gradient": "G", "hessian": "H", "frequency": "F"}
MARK = {"pass": "", "explained": "*", "FAIL": "!", "info": ""}


def cell(item: dict | None) -> str:
    if item is None:
        return "-"
    if item["status"] == "n/a":
        r = item.get("reason", "")
        if "not implemented" in r:
            return "n/i"
        if "Cartesian" in r or "spherical" in r or "PySCF-only" in r:
            return "n/a"
        if r.startswith("failed"):
            return "fail"
        return "-"
    return f"{item['max_abs']:.1e}{MARK[item['status']]}"


def main(argv: list[str] | None = None) -> int:
    global COLS
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", nargs="?", type=Path, default=DEFAULT)
    ap.add_argument("--cases", nargs="*", help="fnmatch patterns on case ids")
    ap.add_argument("--markdown", action="store_true", help="markdown table")
    ap.add_argument("--extras", action="store_true", help="also print the extra comparisons of each case")
    ap.add_argument("--problems", action="store_true", help="list only cells above tolerance (explained or not)")
    ap.add_argument("--compact", action="store_true",
                    help="one block per method / J/K / shell: the worst cell over its cases (number of cases in brackets)")
    ap.add_argument("--columns", help="comma-separated matrix columns to show (default all: " + ",".join(COLS) + "); "
                                      "the release cut shows L-P,L-S,P-S")
    a = ap.parse_args(argv)
    if a.columns:
        COLS = tuple(a.columns.split(","))
    nl = sum(1 for c in COLS if c.startswith("L-"))  # linus columns, then the references among themselves
    data = json.loads(a.results.read_text())
    meta, cases = data["meta"], data["cases"]
    if a.cases:
        cases = [c for c in cases if any(fnmatch.fnmatch(c["id"], p) for p in a.cases)]

    print(f"linus {', '.join(meta['linus_git']) or '?'}; generated {meta['generated']}")
    print("versions: " + "; ".join(f"{k} {', '.join(v) if isinstance(v, list) else v}" for k, v in meta["versions"].items()))
    counts = {"pass": 0, "explained": 0, "FAIL": 0, "n/a": 0, "info": 0}
    problems = []
    rows = []
    for c in cases:
        by = {}
        for it in c["comparisons"]:
            counts[it["status"]] = counts.get(it["status"], 0) + 1
            if it["status"] in ("FAIL", "explained"):
                problems.append((c["id"], it))
            if it["column"] in COLS:
                by[(it["column"], it["quantity"])] = it
        for q in ("energy", "dipole", "gradient", "hessian", "frequency"):
            if not any((col, q) in by for col in COLS):
                continue
            lo = by.get(("L-O", q)) if "L-O" in COLS else None
            grid = f"{lo['residual']:.0e}" if lo and "residual" in lo else ""
            rows.append((c["id"], QLAB[q], [cell(by.get((col, q))) for col in COLS], grid))
        if a.extras:
            for it in c["comparisons"]:
                if it["column"] == "x":
                    rows.append((c["id"], QLAB[it["quantity"]], [cell(it), it["label"]], "x"))

    if a.compact:
        # max over the cases of a group (method, J/K, shell), per quantity and column; a
        # group's cell is marked if any of its cells is
        groups: dict[tuple, dict] = {}
        order = []
        for c in cases:
            cs = c["case"]
            shell = "UKS/UHF" if c["molecule"]["mult"] != 1 else ""
            meth = cs["method"] + (f"-{cs['disp']}" if cs.get("disp") else "")
            g = (meth, cs["jk"] + (" cart" if cs["cart"] else ""), shell)
            if g not in groups:
                groups[g] = {}
                order.append(g)
            for it in c["comparisons"]:
                if it["column"] not in COLS:
                    continue
                key = (it["quantity"], it["column"])
                cur = groups[g].get(key)
                rank = {"FAIL": 3, "explained": 2, "pass": 1, "info": 1, "n/a": 0}
                if it["status"] == "n/a":
                    if cur is None:
                        groups[g][key] = it
                    continue
                if cur is None or cur["status"] == "n/a" or rank[it["status"]] > rank[cur["status"]] or \
                        (rank[it["status"]] == rank[cur["status"]] and it["max_abs"] > cur["max_abs"]):
                    groups[g][key] = it
        sep = " | " if a.markdown else "  "
        if a.markdown:
            print("| method | J/K | shell | q | " + " | ".join(COLS) + " |")
            print("|---|---|---|---|" + "---|" * len(COLS))
        else:
            print(f"{'method':14s} {'J/K':10s} {'shell':8s} q  " + "".join(f"{c:>10s}" for c in COLS))
        for g in order:
            ncase = sum(1 for c in cases if (c["case"]["method"] + (f"-{c['case']['disp']}" if c["case"].get("disp") else ""),
                                           c["case"]["jk"] + (" cart" if c["case"]["cart"] else ""),
                                           "UKS/UHF" if c["molecule"]["mult"] != 1 else "") == g)
            first = True
            for q in ("energy", "dipole", "gradient", "hessian", "frequency"):
                if not any((q, col) in groups[g] for col in COLS):
                    continue
                cells = [cell(groups[g].get((q, col))) for col in COLS]
                name = (f"{g[0]} ({ncase})", g[1], g[2]) if first else ("", "", "")
                if a.markdown:
                    print(f"| {name[0]} | {name[1]} | {name[2]} | {QLAB[q]} | " + " | ".join(cells) + " |")
                else:
                    print(f"{name[0]:14s} {name[1]:10s} {name[2]:8s} {QLAB[q]}  " + "".join(f"{x:>10s}" for x in cells))
                first = False
        return 1 if counts["FAIL"] else 0

    if a.problems:
        for cid, it in problems:
            print(f"{cid:42s} {QLAB[it['quantity']]} {it['label']:34s} {it['a']} vs {it['b']}: "
                  f"{it['max_abs']:.2e} > tol {it['tol']:.1e}  {it['status']}"
                  + (f" ({it['explanation']})" if it.get("explanation") else ""))
        return 0

    if a.markdown:
        print("| case | q | " + " | ".join(COLS) + " | grid |")
        print("|---|---|" + "---|" * (len(COLS) + 1))
        prev = None
        for cid, q, cells, grid in rows:
            if grid == "x":
                print(f"| {'' if cid == prev else cid} | {q} | {cells[0]} | {cells[1]} |" + " |" * (len(COLS) - 1))
            else:
                print(f"| {'' if cid == prev else cid} | {q} | " + " | ".join(cells) + f" | {grid} |")
            prev = cid
    else:
        w = max([len(r[0]) for r in rows] + [10])
        print(f"{'case':{w}s} q  " + "".join(f"{c:>10s}" for c in COLS[:nl]) + "  |" + "".join(f"{c:>10s}" for c in COLS[nl:]) + "   grid")
        prev = None
        for cid, q, cells, grid in rows:
            name = "" if cid == prev else cid
            if grid == "x":
                print(f"{'':{w}s} {q}  {cells[0]:>10s}  {cells[1]}")
            else:
                print(f"{name:{w}s} {q}  " + "".join(f"{x:>10s}" for x in cells[:nl]) + "  |" + "".join(f"{x:>10s}" for x in cells[nl:]) + f"  {grid:>6s}")
            prev = cid
    print(f"\ncells: {counts['pass']} within tolerance, {counts['explained']} explained (*), {counts['FAIL']} unexplained (!), "
          f"{counts['n/a']} not available")
    if problems:
        print("\nabove tolerance:")
        for cid, it in problems:
            print(f"  {cid} {QLAB[it['quantity']]} [{it['label']}] {it['a']} vs {it['b']}: {it['max_abs']:.2e} "
                  f"(tol {it['tol']:.1e}) {it['status']}" + (f": {it['explanation']}" if it.get("explanation") else ""))
    return 1 if counts["FAIL"] else 0


if __name__ == "__main__":
    sys.exit(main())
