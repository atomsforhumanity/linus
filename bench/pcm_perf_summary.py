"""Markdown tables of bench/results/2026-11-08-pcm-perf (PCM SCF modes, timings against the gas phase,
the exact two-pass PCM and ORCA 6.0.0 CPCM, the production job, and the discretization error of the
presets), for the README there.

  .refenv/bin/python bench/pcm_perf_summary.py
"""
import json, pathlib, subprocess, sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
R = ROOT / "bench/results/2026-11-08-pcm-perf"
OLD = ROOT / "bench/results/2026-11-07-pcm"
KCAL = 627.5094740631


def jl(p):
    return [json.loads(x) for x in open(p) if x.strip()]


def best_runs(path, calctype):
    best = {}
    for r in jl(path):
        if r.get("calctype", "gradient") != calctype or r["best_wall"] is None:
            continue
        ok = [x for x in r["runs"] if x["ok"]]
        b = min(ok, key=lambda x: x["wall"])
        t = b["timings"]
        var = r.get("variant") or ("pcm" if r["solvent"] else "gas")
        best[(r["mol"], r["level"], var)] = dict(
            wall=b["wall"], pcm_scf=sum(v for k, v in t.items() if k.startswith("pcm_")), pcm_grad=t.get("grad_solvent", 0.0),
            iters=b["iterations"], npoints=(b.get("solvation") or {}).get("npoints"), energy=b["energy"], nreps=len(ok),
            load=b.get("load_average"), mode=(b.get("solvation") or {}).get("scf"))
    return best


def f(v, d=1):
    return "-" if v is None else f"{v:.{d}f}"


def timing_table():
    p = R / "linus_bench.jsonl"
    if not p.exists():
        return
    best = best_runs(p, "gradient")
    o = OLD / "orca/bench/summary.json"
    orca = json.load(open(o))["cases"] if o.exists() else {}
    old = best_runs(OLD / "linus_bench.jsonl", "gradient") if (OLD / "linus_bench.jsonl").exists() else {}
    print("### Energy + gradient, C-PCM(THF) (s; 8 threads / 8 ranks on `slice bench`; linus: best of the runs in one lock hold; ORCA: 2026-11-07 runs unless marked same day)\n")
    print("| molecule | level | points | linus gas | linus C-PCM (default: predicted) | PCM in SCF | PCM gradient | overhead | linus C-PCM exact two-pass | PCM in SCF | PCM gradient | overhead | 2026-11-07 overhead | ORCA gas | ORCA CPCM | ORCA overhead | ORCA CPCM / linus C-PCM |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    order = {"la18": 0, "u1an39": 1, "ts61": 2, "ts113": 3}
    keys = sorted({(m, l) for (m, l, v) in best}, key=lambda k: (order.get(k[0], 9), k[1]))
    for mol, level in keys:
        g, x, e = best.get((mol, level, "gas")), best.get((mol, level, "pcm")), best.get((mol, level, "pcm-exact"))
        if x is None:
            continue
        og_, op_ = old.get((mol, level, "gas")), old.get((mol, level, "pcm"))
        oc = orca.get(f"{mol}_{level}_engrad", {})
        og, op = (oc.get("gas") or {}).get("best_wall"), (oc.get("cpcm") or {}).get("best_wall")
        # same-day ORCA reruns, where there are any
        same = R / "orca" / f"{mol}_{level}_engrad.json"
        if same.exists():
            ss = json.load(open(same))["summary"]
            og, op = (ss.get("gas") or {}).get("best_wall", og), (ss.get("cpcm") or {}).get("best_wall", op)
            level = level + " (ORCA same day)"
        ov = lambda a: None if (a is None or g is None) else 100 * (a["wall"] / g["wall"] - 1)
        print(f"| {mol} | {level} | {x['npoints']} | {f(g and g['wall'])} | {f(x['wall'])} | {f(x['pcm_scf'])} | {f(x['pcm_grad'])} | {f(ov(x), 0)}% | "
              f"{f(e and e['wall'])} | {f(e and e['pcm_scf'])} | {f(e and e['pcm_grad'])} | {f(ov(e), 0)}% | "
              f"{f(og_ and op_ and 100 * (op_['wall'] / og_['wall'] - 1), 0)}% | {f(og)} | {f(op)} | {f(og and op and 100 * (op / og - 1), 0)}% | {f(op and op / x['wall'], 2)} |")
    print()


def production_table():
    p = R / "linus_production.jsonl"
    if not p.exists():
        return
    best = best_runs(p, "energy")
    o = R / "orca" / "ts113_wb97x-tzvp_energy.json"
    orca = json.load(open(o))["summary"] if o.exists() else {}
    print("### Production job: ts113 wB97X-D3BJ/def2-TZVP C-PCM(THF) energy (s; 8 threads / 8 ranks, `slice bench`, one run each)\n")
    print("| program | J/K | solvent | SCF mode | wall (s) | PCM in SCF (s) | SCF iterations | energy (Eh) | vs gas |")
    print("|---|---|---|---|---|---|---|---|---|")
    for level, jk in [("wb97x-tzvp", "RIJ+COSX"), ("wb97x-tzvp-rijk", "RI-JK")]:
        g = best.get(("ts113", level, "gas"))
        for var in ["gas", "pcm", "pcm-predicted", "pcm-exact"]:
            x = best.get(("ts113", level, var))
            if x is None:
                continue
            vs = "-" if var == "gas" or g is None else f"{100 * (x['wall'] / g['wall'] - 1):+.0f}%"
            print(f"| linus | {jk} | {'gas' if var == 'gas' else 'C-PCM'} | {x['mode'] or '-'} | {f(x['wall'])} | {f(x['pcm_scf'])} | {x['iters']} | {x['energy']:.8f} | {vs} |")
    for kind in ["gas", "cpcm"]:
        s = orca.get(kind)
        if s:
            vs = "-" if kind == "gas" or not orca.get("gas") else f"+{100 * (s['best_wall'] / orca['gas']['best_wall'] - 1):.0f}%"
            print(f"| ORCA 6.0.0 | RIJCOSX (def2/J) | {'gas' if kind == 'gas' else 'CPCM'} | - | {f(s['best_wall'])} | {f((s.get('timings_best') or {}).get('fock_cpcm_terms'))} | {s.get('scf_iterations')} | {s['energy']:.8f} | {vs} |")
    print()
    oc = (orca.get("cpcm") or {}).get("best_wall")
    for level, jk in [("wb97x-tzvp", "RIJ+COSX"), ("wb97x-tzvp-rijk", "RI-JK")]:
        x = best.get(("ts113", level, "pcm")) or best.get(("ts113", level, "pcm-predicted"))
        if x and oc:
            print(f"ORCA RIJCOSX CPCM / linus C-PCM ({jk}, predicted): {oc / x['wall']:.2f}x. ", end="")
    print("Energies: linus at eps 7.4257 with the `pyscf` surface, ORCA at its own THF eps 7.565 with its surface (not comparable).\n")


def run_table(script, path, *extra):
    if path.exists():
        print(subprocess.run([sys.executable, str(ROOT / script), "table", str(path), *extra], capture_output=True, text=True).stdout)


def main():
    timing_table()
    production_table()
    print("### SCF modes against the exact two-pass SCF (kcal/mol, Eh/bohr; development runs)\n")
    run_table("bench/pcm_scf_modes.py", R / "scf_modes.jsonl")
    print("### Discretization error of the presets (B3LYP/def2-SVP RIJ+COSX, C-PCM eps 7.4257; E_solv = E(C-PCM) - E(gas))\n")
    run_table("bench/pcm_discretization.py", R / "discretization.jsonl")


if __name__ == "__main__":
    main()
