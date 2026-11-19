"""Markdown tables of the PCM results in bench/results/2026-11-07-pcm (validation against PySCF,
finite differences, smoothness scans, ORCA and Psi4 model comparisons, Lebedev convergence,
timings), for the README there.

  .refenv/bin/python bench/pcm_summary.py
"""
import glob, json, pathlib

R = pathlib.Path(__file__).resolve().parents[1] / "bench/results/2026-11-07-pcm"
KCAL = 627.5094740631


def jl(p):
    return [json.loads(x) for x in open(p) if x.strip()]


def pyscf_table(name="validation_pyscf.jsonl", title="Against PySCF 2.15 (exact)"):
    p = R / name
    if not p.exists():
        return
    rows = jl(p)
    cds = any("dE_cds" in r for r in rows)
    print(f"### {title}\n")
    print("| set | cases | max abs dE (Eh) | max abs dE_solv (Eh) | max abs dG (Eh/bohr) | max abs dH (Eh/bohr^2) |"
          + (" max abs dE_CDS (Eh) |" if cds else ""))
    print("|---|---|---|---|---|---|" + ("---|" if cds else ""))
    groups = {}
    for r in rows:
        if "error" in r:
            continue
        key = (r["model"] + (f" ({r['solvent']})" if r.get("solvent") else ""), r["jk"], "UHF/UKS" if r.get("mult", 1) > 1 else "RHF/RKS")
        groups.setdefault(key, []).append(r)
    for k in sorted(groups):
        g = groups[k]
        f = lambda key: max((abs(x[key]) for x in g if key in x), default=None)
        fmt = lambda v: "-" if v is None else f"{v:.1e}"
        print(f"| {' '.join(k)} | {len(g)} | {fmt(f('dE'))} | {fmt(f('dE_solv'))} | {fmt(f('max_abs_dG'))} | {fmt(f('max_abs_dH'))} |"
              + (f" {fmt(f('dE_cds'))} |" if cds else ""))
    print()


def fd_table():
    p = R / "fd_gradients.jsonl"
    if not p.exists():
        return
    print("### Analytic gradients against Richardson finite differences (h = 2e-3 bohr)\n")
    print("| molecule | basis | method | J/K | solvent | max abs (G - FD) | max abs G |")
    print("|---|---|---|---|---|---|---|")
    for r in jl(p):
        if "error" in r:
            continue
        print(f"| {r['xyz']} | {r['basis']} | {r['method']} | {r['jk']} | `{r.get('solvent')}` | {r['max_abs_dG_fd']:.1e} | {r['max_abs_G']:.1e} |")
    print()


def fd_hess_table():
    p = R / "fd_hessians.jsonl"
    if not p.exists():
        return
    print("### Analytic Hessians against Richardson finite differences of analytic gradients (def2-SVP; Eh/bohr^2)\n")
    print("| molecule | method | J/K | mult | solvent | h (bohr) | max abs (H - FD) | note |")
    print("|---|---|---|---|---|---|---|---|")
    for r in jl(p):
        if "error" in r:
            print(f"| {r['xyz']} | error | | | `{r.get('solvent')}` | | |")
            continue
        print(f"| {r['xyz']}{' (atoms ' + r['atoms'] + ')' if r.get('atoms') else ''} | {r['method']} | {r['jk']} | {r['mult']} | `{r.get('solvent')}` | {r['h']:g} | {r['max_abs_dH_fd']:.1e} | {r.get('note', '')} |")
    print()


def timing_table():
    p = R / "linus_bench.jsonl"
    o = R / "orca/bench/summary.json"
    if not p.exists():
        return
    best = {}
    for r in jl(p):
        if r.get("calctype", "gradient") != "gradient" or r["best_wall"] is None:
            continue
        ok = [x for x in r["runs"] if x["ok"]]
        b = min(ok, key=lambda x: x["wall"])
        t = b["timings"]
        best[(r["mol"], r["level"], "pcm" if r["solvent"] else "gas")] = dict(
            wall=b["wall"], pcm_scf=sum(v for k, v in t.items() if k.startswith("pcm_")), pcm_grad=t.get("grad_solvent", 0.0),
            iters=b["iterations"], npoints=(b.get("solvation") or {}).get("npoints"))
    orca = json.load(open(o))["cases"] if o.exists() else {}
    print("### Energy + gradient wall times, C-PCM(THF) vs gas phase (s; 8 threads / 8 ranks on `slice bench`, best of 2; linus ts113 one run)\n")
    print("| molecule | level | points linus/ORCA | linus gas | linus C-PCM | PCM in SCF | PCM gradient | linus ratio | ORCA gas | ORCA CPCM | ORCA ratio | ORCA CPCM / linus C-PCM |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for (mol, level, kind) in sorted(best):
        if kind != "pcm":
            continue
        x, g = best[(mol, level, "pcm")], best.get((mol, level, "gas"))
        oc = orca.get(f"{mol}_{level}_engrad", {})
        og, op = (oc.get("gas") or {}).get("best_wall"), (oc.get("cpcm") or {}).get("best_wall")
        f = lambda v, d=1: "-" if v is None else f"{v:.{d}f}"
        print(f"| {mol} | {level} | {x['npoints']}/{(oc.get('cpcm') or {}).get('npoints', '-')} | {f(g and g['wall'])} | {f(x['wall'])} | {f(x['pcm_scf'])} | {f(x['pcm_grad'])} | "
              f"{f(g and x['wall'] / g['wall'], 2)} | {f(og)} | {f(op)} | {f(og and op / og, 2)} | {f(op and op / x['wall'], 2)} |")
    print()


def freq_table():
    p = R / "linus_bench.jsonl"
    if not (p.exists() and (R / "orca/freq/la18_thf.json").exists()):
        return
    rows = [r for r in jl(p) if r.get("calctype") in ("hessian", "frequencies") and r["best_wall"] is not None]
    if not rows:
        return
    orca = {k: json.load(open(R / f"orca/freq/la18_{k}.json")) for k in ("gas", "thf")}
    real = lambda fs: sorted(x for x in fs if abs(x) > 1e-6)
    print("### la18 B3LYP/def2-SVP analytic frequencies, gas and C-PCM(THF) (8 threads / 8 ranks, `slice bench`)\n")
    print("ORCA 6.0.0: `B3LYP/G def2-SVP def2/J RIJCOSX DefGrid2 TightSCF CPCM(THF) Freq`; linus: RI-JK "
          "(def2-universal-jkfit), grid orca:2, `orca` preset at ORCA's eps 7.565. The geometry is not a "
          "minimum at this level (three imaginary modes in both programs).\n")
    print("| program | phase | wall (s) | lowest three (cm-1) | max abs d(freq) vs ORCA, modes > 100 cm-1 | PCM parts (s) |")
    print("|---|---|---|---|---|---|")
    for k, label in (("gas", "gas"), ("thf", "C-PCM")):
        o = orca[k]
        print(f"| ORCA | {label} | {o.get('best_wall', o.get('wall')):.1f} | {', '.join(f'{v:.1f}' for v in real(o['frequencies_cm1'])[:3])} | - | - |")
    for r in rows:
        b = min((x for x in r["runs"] if x["ok"]), key=lambda x: x["wall"])
        fr = real(b.get("frequencies") or [])
        ref = real(orca["thf" if r["solvent"] else "gas"]["frequencies_cm1"])
        dev = max(abs(a - c) for a, c in zip(fr, ref) if c > 100)
        t = b["timings"]
        parts = "-" if not r["solvent"] else (f"SCF {sum(v for k2, v in t.items() if k2.startswith('pcm_')):.2f}, "
                                             f"explicit {t.get('hess_solvent', 0):.2f}, response {t.get('cphf_solvent', 0):.2f}")
        print(f"| linus | {'C-PCM' if r['solvent'] else 'gas'} | {b['wall']:.1f} | {', '.join(f'{v:.1f}' for v in fr[:3])} | {dev:.2f} | {parts} |")
    print()


def orca_table():
    """One row per comparison set and linus preset (largest deviations over its cases; the
    per-case numbers are in the JSON files)."""
    print("### Against ORCA 6.0.0 CPCM (ORCA's eps; largest deviations over each set)\n")
    print("| set | linus preset | cases | max abs d(E_solv - E_gas) (Eh) | max abs d(dielectric) (Eh) | max abs dG (Eh/bohr) | max abs d(solvent part of G) | points linus/ORCA (la18) |")
    print("|---|---|---|---|---|---|---|---|")
    for f in sorted(glob.glob(str(R / "orca_compare_*.json"))):
        d = json.load(open(f))
        for preset in ("orca", "pyscf"):
            xs = [(c, c[preset]) for c in d["cases"] if preset in c]
            if not xs:
                continue
            m = lambda k: max(abs(x[k]) for _, x in xs)
            la = [f"{x['npoints']}/{c['orca']['npoints']}" for c, x in xs if c["case"].startswith("la18")][:1]
            print(f"| {pathlib.Path(f).stem.replace('orca_compare_', '')} | {preset} | {len(xs)} | {m('d_delta_solv_vs_orca'):.1e} | {m('d_dielectric_vs_orca'):.1e} | "
                  f"{m('max_abs_dG_vs_orca'):.1e} | {m('max_abs_d(solvent part of G)_vs_orca'):.1e} | {la[0] if la else '-'} |")
    print()


def psi4_table():
    p = R / "psi4_compare.json"
    if not p.exists():
        return
    print("### Against Psi4 1.11 + PCMSolver (model level, kcal/mol)\n")
    print("| case | eps | Psi4 E_PCM - E_gas | linus pyscf | linus pyscf, H 1.2 Å | linus (H 1.2) - Psi4 |")
    print("|---|---|---|---|---|---|")
    for r in json.load(open(p))["rows"]:
        print(f"| {r['case']} | {r['eps']} | {r['psi4_delta_solv_kcal']:.2f} | {r['linus_pyscf_delta_solv_kcal']:.2f} | {r['linus_pyscf_H1.2_delta_solv_kcal']:.2f} | {r['linus_pyscf_H1.2_minus_psi4_kcal']:+.2f} |")
    print()


def lebedev_table():
    p = R / "lebedev_convergence.json"
    if not p.exists():
        return
    d = json.load(open(p))
    print("### Discretization: Lebedev points per sphere (B3LYP/def2-SVP RIJ+COSX, C-PCM, eps 7.4257)\n")
    print("| molecule | switching | N | points | E_solv - E_solv(974) (kcal/mol) | max abs dG vs 974 | PCM s in SCF | PCM gradient s |")
    print("|---|---|---|---|---|---|---|---|")
    for r in d["results"]:
        for x in r["rows"]:
            print(f"| {r['mol']} | {r['scheme']} | {x['lebedev']} | {x['npoints']} | {x['dE_solv_vs_largest'] * KCAL:+.3f} | {x['max_abs_dG_vs_largest']:.1e} | {x['t_pcm_scf']:.2f} | {x['t_grad_solvent']:.2f} |")
    print()


def scan_table():
    files = sorted(glob.glob(str(R / "scan_*.json")))
    if not files:
        return
    print("### Smoothness scans (energy and gradient at every step)\n")
    print("| scan | coordinate | steps | max abs (integrated gradient - dE) (Eh) | max/median third difference | points (min-max) |")
    print("|---|---|---|---|---|---|")
    for f in files:
        d = json.load(open(f))
        for k in ("bond", "torsion"):
            x = d[k]
            print(f"| {pathlib.Path(f).stem} | {k} | {len(x['values'])} | {x['max_dev_integrated_gradient']:.1e} | {x['jump_ratio']:.1f} | {min(x['npoints'])}-{max(x['npoints'])} |")
    print()


if __name__ == "__main__":
    pyscf_table()
    pyscf_table("validation_pyscf_hessian.jsonl", "Hessians against PySCF 2.15 (exact)")
    pyscf_table("validation_pyscf_smd.jsonl", "SMD against PySCF 2.15 (IEF-PCM, SMD radii, 590 points, CDS with the exact SASA)")
    fd_table()
    fd_hess_table()
    scan_table()
    orca_table()
    psi4_table()
    lebedev_table()
    timing_table()
    freq_table()
