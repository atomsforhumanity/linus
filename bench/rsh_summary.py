"""Tables from bench/rsh_bench.py results: wall times (best of the repetitions) and errors
against the exact-integral references.

usage: python3 bench/rsh_summary.py <timed-tag> <reference-tag> [program keys, comma-separated]
"""
import json, pathlib, sys

ROOT = pathlib.Path(__file__).resolve().parents[1]


def load(tag):
    out = {}
    for f in sorted((ROOT / "bench/results" / tag).glob("*.json")):
        d = json.loads(f.read_text())
        if "mol" in d:
            out[(d["mol"], d["basis"], d["calc"])] = d
    return out


def ok_runs(d, prog):
    return [r for r in d["runs"].get(prog, []) if r.get("ok")]


def maxdiff(a, b):
    return max(abs(x - y) for ra, rb in zip(a, b) for x, y in zip(ra, rb))


def main():
    timed, ref = load(sys.argv[1]), load(sys.argv[2])
    progs = sys.argv[3].split(",") if len(sys.argv) > 3 else ["linus-ri", "linus-cosx", "orca", "linus-ri@none", "orca@none"]
    print("| molecule | basis | calc | nbf | " + " | ".join(f"{p} wall (s)" for p in progs) + " |  (best of n runs: n in parentheses)")
    print("|---|---|---|---|" + "---|" * len(progs))
    for (mol, basis, calc), d in sorted(timed.items()):
        cells, nbf = [], ""
        for p in progs:
            rs = ok_runs(d, p)
            fails = [r for r in d["runs"].get(p, []) if not r.get("ok")]
            if rs:
                cells.append(f"{min(r['wall'] for r in rs):.1f}" + (f" ({len(rs)})" if len(rs) > 1 else ""))
                nbf = nbf or rs[0].get("nbf", "")
            elif fails:
                cells.append("fails: " + fails[0].get("error", "")[:60].replace("\n", " "))
            else:
                cells.append("")
        print(f"| {mol} | {basis} | {calc} | {nbf} | " + " | ".join(cells) + " |")
    print()
    print("Energy errors vs exact integrals (same program, same DefGrid2 grid), Eh:")
    print("| molecule | basis | linus exact | linus RI-JK - exact | linus RIJ+COSX - exact | ORCA RIJCOSX - ORCA exact | ORCA exact - linus exact |")
    print("|---|---|---|---|---|---|---|")
    for (mol, basis, calc), d in sorted(timed.items()):
        if calc != "energy":
            continue
        rd = ref.get((mol, basis, "energy"), {"runs": {}})
        le = ok_runs(rd, "linus-exact")
        oe = ok_runs(rd, "orca-exact")
        e = lambda p, src=d: (ok_runs(src, p) or [{}])[0].get("energy")
        lx = le[0]["energy"] if le else None
        ox = oe[0]["energy"] if oe else None
        f = lambda a, b: f"{a - b:+.2e}" if a is not None and b is not None else ""
        print(f"| {mol} | {basis} | {lx if lx is not None else ''} | {f(e('linus-ri'), lx)} | {f(e('linus-cosx'), lx)} | {f(e('orca'), ox)} | {f(ox, lx)} |")
    print()
    print("Gradients without dispersion, max |difference| (Eh/bohr) against linus exact integrals and against ORCA RIJCOSX:")
    gk = ["linus-ri", "linus-ri-v2", "linus-cosx-v2", "orca"]
    print("| molecule | basis | " + " | ".join(f"{k} vs exact" for k in gk) + " | orca vs ORCA exact | linus RI-JK vs ORCA | linus RIJ+COSX vs ORCA | max abs G |")
    print("|---|---|" + "---|" * (len(gk) + 4))
    for (mol, basis, calc), d in sorted(timed.items()):
        if calc != "gradient":
            continue
        rd = ref.get((mol, basis, "gradient"), {"runs": {}})
        g = lambda p, src: (ok_runs(src, p) or [{}])[0].get("gradient_no_disp")
        gx = g("linus-exact", rd)
        f = lambda a, b: f"{maxdiff(a, b):.1e}" if a and b else ""
        go = g("orca", d)
        gri = g("linus-ri-v2", d) or g("linus-ri", d)
        gm = max(abs(x) for r in (gx or gri or go or [[0]]) for x in r)
        cells = [f(g(k, d), gx) for k in gk]
        cells.append(f(go, g("orca-exact", rd)))
        print(f"| {mol} | {basis} | " + " | ".join(cells) + f" | {f(gri, go)} | {f(g('linus-cosx-v2', d), go)} | {gm:.1e} |")


if __name__ == "__main__":
    main()
