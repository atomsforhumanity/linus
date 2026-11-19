"""Tables (markdown) from bench/results/2026-11-07-cosx-scf: the SCF set before/after, the timed
RIJCOSX runs against ORCA 6.0.0 and the tightly converged references.

usage: python3 bench/cosx_scf_summary.py [--dir bench/results/2026-11-07-cosx-scf]
"""
import argparse, json, pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
ORCA = {  # (dir, file, run) of the ORCA numbers reused from earlier benchmarks (same harness settings)
    ("ts61", "b3lyp", "energy"): ("2026-11-07-cosx-perf", "ts61_b3lyp-tzvp_energy.json", "orca"),
    ("ts61", "b3lyp", "gradient"): ("2026-11-07-cosx-perf", "ts61_b3lyp-tzvp_gradient.json", "orca"),
    ("ts113", "b3lyp", "energy"): ("2026-11-07-cosx-perf", "ts113_b3lyp-tzvp_energy.json", "orca"),
    ("ts113", "b3lyp", "gradient"): ("2026-11-07-cosx-perf", "ts113_b3lyp-tzvp_gradient.json", "orca"),
    ("ts61", "wb97x-d3bj", "energy"): ("2026-11-07-wb97x-d3bj", "ts61_def2-tzvp_energy.json", "orca"),
    ("ts61", "wb97x-d3bj", "gradient"): ("2026-11-07-wb97x-d3bj", "ts61_def2-tzvp_gradient.json", "orca"),
    ("ts113", "wb97x-d3bj", "energy"): ("2026-11-07-wb97x-d3bj", "ts113_def2-tzvp_energy.json", "orca"),
    ("ts113", "wb97x-d3bj", "gradient"): ("2026-11-07-wb97x-d3bj", "ts113_def2-tzvp_gradient.json", "orca-v2"),
}


def orca(mol, func, calc):
    d, f, k = ORCA[(mol, func, calc)]
    j = json.loads((ROOT / "bench/results" / d / f).read_text())
    if "summary" in j:
        s = j["summary"]["orca"]
        return s["wall"], s["iterations"]
    r = j["runs"][k][0]
    return r["wall"], r["iterations"]


def maxdiff(a, b):
    return max(abs(x - y) for ra, rb in zip(a, b) for x, y in zip(ra, rb))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(ROOT / "bench/results/2026-11-07-cosx-scf"))
    a = ap.parse_args()
    d = pathlib.Path(a.dir)
    s = d / "scf_set.json"
    if s.exists():
        res = json.loads(s.read_text())["results"]
        print("| case | builds old | builds new | E new - old (Eh) |\n|---|---|---|---|")
        for case, r in res.items():
            if "old" in r and "new" in r and r["old"].get("ok") and r["new"].get("ok"):
                print(f"| {case} | {r['old']['builds']} | {r['new']['builds']} | {r['new']['energy'] - r['old']['energy']:+.1e} |")
        print()
    runs = {}
    for f in sorted(d.glob("ts*_*_tzvp_*_*.json")):
        j = json.loads(f.read_text())
        if not j.get("timed") and not j["program"].startswith("ref"):
            continue
        mol, func, _, calc = j["case"].split(":")
        runs.setdefault((mol, func, calc), {})[j["program"]] = j["run"]
    print("| case | linus old: builds, wall s | linus new: builds, wall s | old/new | ORCA: cycles, wall s | ORCA/new | new: dE vs ref (Eh) | new: max dG vs ref | old: dE, max dG vs ref |")
    print("|---|---|---|---|---|---|---|---|---|")
    for key in sorted(runs):
        mol, func, calc = key
        r = runs[key]
        ref = runs.get((mol, func, "gradient"), {}).get("ref-new")
        if calc == "gradient" and "ref-new" in r and len(r) == 1:
            continue
        cols = []
        for p in ("old", "new"):
            x = r.get(p)
            cols.append(f"{x['builds']}, {x['wall']:.1f}" if x and x.get("ok") else "-")
        wo, wn = (r.get(p, {}).get("wall") if r.get(p, {}).get("ok") else None for p in ("old", "new"))
        cols.append(f"{wo / wn:.2f}" if wo and wn else "-")
        try:
            ow, oi = orca(mol, func, calc)
            cols.append(f"{oi}, {ow:.1f}")
            cols.append(f"{ow / wn:.2f}" if wn else "-")
        except (KeyError, FileNotFoundError):
            cols += ["-", "-"]
        acc = []
        for p in ("new", "old"):
            x = r.get(p)
            if x and x.get("ok") and ref and ref.get("ok"):
                de = x["energy"] - ref["energy"]
                dg = maxdiff(x["gradient"], ref["gradient"]) if x.get("gradient") else None
                acc.append(f"{de:+.1e}" + (f", {dg:.1e}" if dg is not None else ""))
            else:
                acc.append("-")
        new_de, _, new_dg = acc[0].partition(", ")
        print(f"| {mol} {func} {calc} | {' | '.join(cols)} | {new_de} | {new_dg or '-'} | {acc[1]} |")
    gconv_table(d)


def gconv_table(d):
    """Gradient jobs at max |F_ai| < 1e-7 (default) and < 5e-7 against the references."""
    print("\n| case | g_tol 1e-7: builds, final max F_ai, max dG vs ref | g_tol 5e-7: builds, final max F_ai, max dG vs ref | max dG between the two |")
    print("|---|---|---|---|")
    for f in sorted(d.glob("*_gradient_ref-new.json")):
        ref = json.loads(f.read_text())["run"]
        stem = f.name[: -len("ref-new.json")]
        runs = {}
        for lab in ("g1e-7", "new", "g5e-7"):
            g = d / f"{stem}{lab}.json"
            if g.exists():
                runs.setdefault("g5e-7" if lab == "g5e-7" else "g1e-7", json.loads(g.read_text())["run"])
        if "g1e-7" not in runs and "g5e-7" not in runs:
            continue
        cols = []
        for k in ("g1e-7", "g5e-7"):
            r = runs.get(k)
            cols.append(f"{r['builds']}, {r['final_gmax']:.1e}, {maxdiff(r['gradient'], ref['gradient']):.1e}" if r and r.get("ok") else "-")
        both = maxdiff(runs["g1e-7"]["gradient"], runs["g5e-7"]["gradient"]) if len(runs) == 2 and all(r.get("ok") for r in runs.values()) else None
        print(f"| {stem.rstrip('_')} | {cols[0]} | {cols[1]} | {both:.1e} |" if both is not None else f"| {stem.rstrip('_')} | {cols[0]} | {cols[1]} | - |")


if __name__ == "__main__":
    main()
