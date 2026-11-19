"""Tables for bench/results/2026-11-07-vv10: the XC grid study (grid-study/), the NLC grid study
(nlc-grids/*.json), the PySCF validation (pyscf_validation.jsonl) and the timings.

  python3 bench/vv10_summary.py [grid|nlc|pyscf|all]
"""
import json, pathlib, sys

import numpy as np

R = pathlib.Path(__file__).resolve().parents[1] / "bench/results/2026-11-07-vv10"


def grid_study():
    print("## XC grid errors (RI-JK, def2-TZVP; reference pyscf:7)\n")
    print("| molecule | functional | grid | wall (s) | dE (µEh) | max abs dG (Eh/bohr) |")
    print("|---|---|---|---|---|---|")
    for f in sorted((R / "grid-study").glob("*.json")):
        d = json.loads(f.read_text())
        runs = {k: v[-1] for k, v in d["runs"].items() if v[-1].get("ok")}
        ref = next((v for k, v in runs.items() if k.endswith("@pyscf7")), None)
        if ref is None:
            continue
        g0 = np.array(ref["gradient"])
        for k, v in runs.items():
            if k.endswith("@pyscf7"):
                continue
            dg = np.abs(np.array(v["gradient"]) - g0).max()
            print(f"| {d['mol']} | {d['method']} | {k.split('@')[-1]} | {v['wall']:.1f} | {1e6 * (v['energy'] - ref['energy']):+.1f} | {dg:.1e} |")


def nlc_study():
    print("## NLC grid errors at fixed density (VV10 energy and VV10 gradient; reference: the last grid)\n")
    for f in sorted((R / "nlc-grids").glob("*.json")):
        if not f.read_text().strip():
            continue
        d = json.loads(f.read_text())
        ref = d["grids"][-1]
        g0 = np.array(ref["gradient_vv10"]) if ref.get("gradient_vv10") else None
        print(f"\n{f.stem} (E_vv10 ref {ref['e_vv10']:.10f} on {ref['nlc_grid']}, {ref['points']} points)\n")
        print("| NLC grid | points | active | dE_vv10 (µEh) | max abs dG_vv10 (Eh/bohr) | t energy (s) | t gradient (s) |")
        print("|---|---|---|---|---|---|---|")
        for g in d["grids"]:
            dg = np.abs(np.array(g["gradient_vv10"]) - g0).max() if g0 is not None and g.get("gradient_vv10") else float("nan")
            print(f"| {g['nlc_grid']} | {g['points']} | {g['active_points']} | {1e6 * (g['e_vv10'] - ref['e_vv10']):+.3f} | {dg:.1e} | {g['t_energy']:.2f} | {g['t_gradient']:.2f} |")


def pyscf():
    print("## linus vs PySCF 2.15 (same grids: XC level 3, NLC level 1)\n")
    print("| molecule | basis | functional | J/K | grid response | dE (Eh) | dE_vv10 (Eh) | max abs dG (Eh/bohr) |")
    print("|---|---|---|---|---|---|---|---|")
    for ln in (R / "pyscf_validation.jsonl").read_text().splitlines():
        r = json.loads(ln)
        if "error" in r:
            print(f"| {r['mol']} | {r['basis']} | {r['method']} | ERROR | | | | |")
            continue
        dg = f"{r['max_abs_dG']:.1e}" if "max_abs_dG" in r else "-"
        print(f"| {r['mol']} | {r['basis']} | {r['method']} | {r['jk']} | {'yes' if r['grid_response'] else 'no'} | {r['dE']:+.1e} | {r.get('dE_vv10', float('nan')):+.1e} | {dg} |")


def pyscf_aggregate():
    """Worst case per (J/K, grid response, basis) over molecules and functionals."""
    import collections
    agg = collections.defaultdict(list)
    for ln in (R / "pyscf_validation.jsonl").read_text().splitlines():
        r = json.loads(ln)
        if "error" not in r:
            agg[(r["jk"], r["grid_response"], r["basis"])].append(r)
    print("| J/K | grid response | basis | cases | molecules | max abs dE (Eh) | max abs dE_vv10 (Eh) | max abs dG (Eh/bohr) |")
    print("|---|---|---|---|---|---|---|---|")
    for (jk, gr, basis), v in sorted(agg.items()):
        dg = [r["max_abs_dG"] for r in v if "max_abs_dG" in r]
        mols = ", ".join(sorted({r["mol"] for r in v}))
        print(f"| {jk} | {'yes' if gr else 'no'} | {basis} | {len(v)} | {mols} | {max(abs(r['dE']) for r in v):.1e} | "
              f"{max(abs(r['dE_vv10']) for r in v):.1e} | {max(dg):.1e} ({len(dg)} gradients) |" if dg else
              f"| {jk} | {'yes' if gr else 'no'} | {basis} | {len(v)} | {mols} | {max(abs(r['dE']) for r in v):.1e} | {max(abs(r['dE_vv10']) for r in v):.1e} | - |")


def tget(t, *keys):
    return sum(t.get(k, 0.0) for k in keys)


def timings(base=R):
    """OMol25 benchmark and functional comparison: wall times and their split."""
    print("## Wall times and their split (s): exchange = J/K builds (SCF jk + COSX final energy + 2e gradient),"
          " XC = semilocal XC (SCF xc minus vv10, grad_xc), VV10 = SCF vv10 + grad_vv10 + NLC grid\n")
    print("| molecule | basis | functional | calc | program | wall | iterations | exchange | XC | VV10 | VV10 share | E |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for f in sorted(base.glob("*_*_*_*.json")):
        d = json.loads(f.read_text())
        if "runs" not in d:
            continue
        for k, runs in sorted(d["runs"].items()):
            r = min((x for x in runs if x.get("ok")), key=lambda x: x["wall"], default=None)
            if r is None:
                print(f"| {d['mol']} | {d['basis']} | {d['method']} | {d['calc']} | {k} | FAIL | | | | | | |")
                continue
            if "timings" in r:
                t = r["timings"]
                vv = tget(t, "vv10", "grad_vv10", "nlc_grid")
                ex = tget(t, "jk", "k_final", "grad_2e", "grad_cosx", "grad_rij")
                xc = tget(t, "xc", "grad_xc") - tget(t, "vv10")
                print(f"| {d['mol']} | {d['basis']} | {d['method']} | {d['calc']} | {k} | {r['wall']:.1f} | {r['iterations']} | {ex:.1f} | {xc:.1f} | {vv:.1f} | {100 * vv / r['wall']:.1f}% | {r['energy']:.8f} |")
            else:
                nl = f"NL grad {r['t_nl_gradient']:.1f}" if r.get("t_nl_gradient") else ""
                print(f"| {d['mol']} | {d['basis']} | {d['method']} | {d['calc']} | {k} | {r['wall']:.1f} | {r.get('iterations')} | | | {nl} | | {r['energy']:.8f} |")


def errors(base=R):
    """Energies and gradients against the tight references (ref/) and against ORCA."""
    print("## Errors against the tight reference (linus exact J/K, pyscf:7 / NLC pyscf:4, verytight) and against ORCA\n")
    print("| molecule | basis | functional | program | dE vs ref (µEh) | max abs dG vs ref | dE vs ORCA (µEh) | max abs dG vs ORCA |")
    print("|---|---|---|---|---|---|---|---|")
    for f in sorted(base.glob("*_gradient.json")):
        d = json.loads(f.read_text())
        rf = R / "ref" / f.name
        ref = None
        if rf.exists():
            rr = json.loads(rf.read_text())["runs"]
            ref = next((v[-1] for k, v in rr.items() if v[-1].get("ok")), None)
        orca = next((v[-1] for k, v in d["runs"].items() if k.startswith("orca") and v[-1].get("ok")), None)
        for k, runs in sorted(d["runs"].items()):
            r = runs[-1]
            if not r.get("ok") or "gradient" not in r:
                continue
            g = np.array(r["gradient"])
            cells = []
            for other in (ref, orca):
                if other is None or other is r:
                    cells += ["", ""]
                else:
                    cells += [f"{1e6 * (r['energy'] - other['energy']):+.1f}", f"{np.abs(g - np.array(other['gradient'])).max():.1e}"]
            print(f"| {d['mol']} | {d['basis']} | {d['method']} | {k} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("grid", "all"):
        grid_study()
    if what in ("nlc", "all"):
        nlc_study()
    if what in ("pyscf", "all"):
        pyscf()
    if what in ("pyscfagg", "all"):
        pyscf_aggregate()
    if what in ("errors", "all"):
        errors()
    if what in ("time", "all"):
        timings()
    if what in ("headline", "all"):
        timings(R / "headline")
        errors(R / "headline")
