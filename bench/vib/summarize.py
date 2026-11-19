"""Tables from validate.py outputs: CASE.analysis-*.json, CASE.pyscf.json, CASE.orca.json.

python bench/vib/summarize.py DIR [--json OUT]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def fmt(x, f="{:.1e}"):
    return "-" if x is None else f.format(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--json")
    a = ap.parse_args()
    d = Path(a.dir)
    summary = {"analysis": {}, "pyscf": {}, "orca": {}}
    print("## Same Hessian through each program's analysis (linus minus reference)\n")
    print("Thermochemistry at 298.15 K, 1 atm, at the reference's symmetry number; linus with the reference's "
          "CODATA (2014) for PySCF and Psi4, its default (2022) for ORCA; 'ORCA scaled': linus with ORCA's frequency "
          "factor as freq.scale.\n")
    print("| case | masses | PySCF max dν | PySCF max dG | qRRHO S, S+H dG | Psi4 max dν | Psi4 max dIR | Psi4 max dG | "
          "ORCA ν ratio - 1 | ORCA max dIR | ORCA dG (scaled) | ORCA qRRHO dG (scaled) | σ linus / PySCF / Psi4 / ORCA |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for f in sorted(d.glob("*.analysis-*.json")):
        r = json.loads(f.read_text())
        rows = {(x["program"], x["codata"]): x for x in r["rows"]}
        label = f.name.split(".")[0]
        summary["analysis"][f"{label}/{r['masses']}"] = r["rows"]
        p, q1, q2 = rows[("pyscf", "codata2014")], rows[("pyscf-qS", "codata2014")], rows[("pyscf-qSH", "codata2014")]
        s4 = rows[("psi4", "codata2014")]
        o, os_, oqs = rows[("orca", "codata2022")], rows[("orca-scaled", "codata2022")], rows[("orca-qS-scaled", "codata2022")]
        g = lambda row, k: abs(row["thermo_diff"].get(k, float("nan")))  # noqa: E731
        ratio = o.get("orca_frequency_ratio", [float("nan")] * 2)
        print(f"| {label} | {r['masses']} | {fmt(p['dfreq_max'])} | {fmt(g(p, 'G_tot'))} | {fmt(g(q1, 'G_tot'))}, {fmt(g(q2, 'G_tot'))} | "
              f"{fmt(s4['dfreq_max'])} | {fmt(s4.get('dir_max'))} | {fmt(g(s4, 'G_tot'))} | {ratio[0] - 1:.4e} | "
              f"{fmt(o.get('dir_max'))} | {fmt(g(os_, 'G_corr'))} | {fmt(g(oqs, 'G_corr'))} | "
              f"{p['symmetry_number'][0]} / {p['symmetry_number'][1]} / {s4['symmetry_number'][1]} / {o['symmetry_number'][1]} |")
    print("\n## End to end: each program's own Hessian\n")
    print("| case | PySCF dE | PySCF max dH | PySCF max dν | ORCA dE | ORCA max dH | ORCA max dν (ν ratio - 1) | ORCA max dIR / max IR | "
          "ORCA dG_corr | imaginary linus / PySCF / ORCA | point group linus / ORCA |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    names = sorted({f.name.split(".")[0] for f in [*d.glob("*.pyscf.json"), *d.glob("*.orca.json")]})
    for n in names:
        pf, of = d / f"{n}.pyscf.json", d / f"{n}.orca.json"
        p = json.loads(pf.read_text()) if pf.exists() else None
        o = json.loads(of.read_text()) if of.exists() else None
        if p is None and o is None:
            continue
        summary["pyscf"][n] = p and {k: v for k, v in p.items() if not k.endswith("frequencies")}
        summary["orca"][n] = o and {k: v for k, v in o.items() if not k.endswith(("frequencies", "_ir"))}
        nim_p = sum(1 for x in p["pyscf_frequencies"] if x < 0) if p else None
        nim_l = sum(1 for x in (p or o)["linus_frequencies"] if x < 0)
        print(f"| {n} | {fmt(p and p['linus_energy'] - p['pyscf_energy'])} | {fmt(p and p['dhess_max'])} | "
              f"{fmt(p and p['dfreq_max'])} | {fmt(o and o['dE'])} | {fmt(o and o['dhess_max'])} | "
              f"{fmt(o and o['dfreq_max'])} ({fmt(o and o['orca_frequency_ratio'] - 1, '{:.2e}')}) | "
              f"{fmt(o and o.get('dir_max'))} / {fmt(o and o.get('ir_max'), '{:.0f}')} | {fmt(o and o['dG_corr'])} | "
              f"{nim_l} / {'-' if nim_p is None else nim_p} / {'-' if o is None else o['imaginary'][1]} | "
              f"{'-' if o is None else ' / '.join(o.get('point_group', ['-', '-']))} |")
    if a.json:
        Path(a.json).write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
