#!/usr/bin/env python3
"""Summarize compensated grid-K's accuracy study (pulled run dirs of make_study.py runs): the
tables of its record.

usage: analyze_study.py <pulled run dir> [<pulled run dir> ...] [--json out.json]

Every error is against exact four-center J and K at the same density (first-order SCF energy
errors; the gradient's two-electron term). Methods:
  df[aux]           : DF J + DF K with aux (jk df)
  cgk[grid]+J[aux]  : DF J with aux + compensated grid-K on grid (jk rij_cgk)
  K-only rows       : the exchange alone (DF K; grid-K variants raw, dscf, fit, cgk)
Absolute: per molecule E error (Eh) and per atom (uEh/atom). Relative: GMTKN55 reactions
(sum nu_i dE_i, kcal/mol). Gradients: max |dg| over atoms (Eh/bohr)."""
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

KCAL = 627.509474


def load(dirs):
    rows = []
    rx = {}
    for d in dirs:
        d = Path(d)
        if (d / "reactions.json").exists():
            rx.update(json.loads((d / "reactions.json").read_text()))
        for f in (d / "output").glob("*/result.json"):
            r = json.loads(f.read_text())
            if r.get("rc") == 0:
                rows.append(r)
    return rows, rx


def methods(res):
    """{method label: (dE total J+K, dE K only, dg max or None)} of one species' result."""
    out = {}
    dfs = {d["aux"]: d for d in res["df"]}
    short = {a: ("jk" if a == "def2-universal-jkfit" else "qz") for a in dfs}
    for a, d in dfs.items():
        out[f"df[{short[a]}]"] = (d["dej"] + d["dex"], d["dex"], d.get("dgjk_max"), d.get("dgk_max"))
    for g in res["grids"]:
        kg = g["grid"].replace("gridx:", "x")
        out[f"K:{g['variant']}[{kg}]"] = (None, g["dex"], None, g.get("dgk_max"))
        if g["variant"] == "cgk":
            for a, d in dfs.items():
                tot = next((w["dgjk_max"] for w in g.get("with_aux", []) if w["aux"] == a), None)
                out[f"cgk[{kg}]+J[{short[a]}]"] = (d["dej"] + g["dex"], g["dex"], tot, g.get("dgk_max"))
    return out


def stats(v):
    v = [x for x in v if x is not None]
    if not v:
        return None
    a = [abs(x) for x in v]
    return {"n": len(v), "mae": statistics.mean(a), "max": max(a), "mse": statistics.mean(v)}


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    out_json = sys.argv[sys.argv.index("--json") + 1] if "--json" in sys.argv else None
    if out_json:
        args = [a for a in args if a != out_json]
    rows, rx = load(args)
    by = defaultdict(dict)  # (basis) -> species -> methods
    natoms = {}
    for r in rows:
        by[r["basis"]][r["species"]] = methods(r["result"])
        natoms[r["species"]] = r["natoms"]
    report = {}
    for basis in sorted(by):
        sp = by[basis]
        labels = sorted({m for v in sp.values() for m in v})
        rep = {"species": len(sp), "absolute": {}, "relative": {}, "gradient": {}}
        for m in labels:
            tot = [v[m][0] for v in sp.values() if m in v]
            kon = [v[m][1] for v in sp.values() if m in v]
            per_atom = [v[m][0] / natoms[s] * 1e6 for s, v in sp.items() if m in v and v[m][0] is not None]
            kper_atom = [v[m][1] / natoms[s] * 1e6 for s, v in sp.items() if m in v]
            rep["absolute"][m] = {"total_Eh": stats(tot), "K_Eh": stats(kon),
                                  "total_uEh_per_atom": stats(per_atom), "K_uEh_per_atom": stats(kper_atom)}
            rep["gradient"][m] = {"total": stats([v[m][2] for v in sp.values() if m in v]),
                                  "K": stats([v[m][3] for v in sp.values() if m in v])}
            # relative energies over every reaction whose species are all present
            rel, relk = [], []
            for sub, rs in rx.items():
                for r in rs:
                    keys = [f"{sub}.{n}" for n in r["species"]]
                    if not all(k in sp and m in sp[k] for k in keys):
                        continue
                    if all(sp[k][m][0] is not None for k in keys):
                        rel.append(KCAL * sum(c * sp[k][m][0] for c, k in zip(r["coefs"], keys)))
                    relk.append(KCAL * sum(c * sp[k][m][1] for c, k in zip(r["coefs"], keys)))
            rep["relative"][m] = {"total_kcal": stats(rel), "K_kcal": stats(relk)}
        report[basis] = rep
    for basis, rep in report.items():
        print(f"\n== {basis}: {rep['species']} species")
        print(f"{'method':24s} {'|dE| total MAE/max (uEh/atom)':>32s} {'K only':>22s} "
              f"{'rel. total MAE/max kcal':>26s} {'grad total max':>16s} {'grad K max':>12s}")
        for m in sorted(rep["absolute"]):
            a = rep["absolute"][m]
            r = rep["relative"][m]
            g = rep["gradient"][m]
            f = lambda s, k="mae": ("%.2g" % s[k]) if s else "-"  # noqa: E731
            print(f"{m:24s} {f(a['total_uEh_per_atom']):>15s}/{f(a['total_uEh_per_atom'],'max'):<15s} "
                  f"{f(a['K_uEh_per_atom']):>10s}/{f(a['K_uEh_per_atom'],'max'):<10s} "
                  f"{f(r['total_kcal']):>12s}/{f(r['total_kcal'],'max'):<12s} "
                  f"{f(g['total'],'max'):>16s} {f(g['K'],'max'):>12s}")
    if out_json:
        Path(out_json).write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
