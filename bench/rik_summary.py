"""Tables for RI-JK beyond the in-memory limit (bench/rik_bench.py results): wall time, peak RSS,
3-index storage, SCF iterations and errors against exact integrals.

usage: python3 bench/rik_summary.py [timed-tag]
Timed results: bench/results/<tag>/<mol>_<basis>_<method>_<calc>.json (rik_bench.py). Added for
comparison: ORCA wB97X-D3BJ runs from bench/results/2026-11-07-wb97x-d3bj (ORCA unchanged since)
and B3LYP RIJ+COSX / ORCA runs from bench/results/2026-11-07-cosx-perf (main 538f80c COSX code).
Exact references (program linus-exact): 2026-11-07-wb97x-d3bj-ref (wB97X-D3BJ) and <tag>-ref.
"""
import json, pathlib, sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
RES = ROOT / "bench/results"


def load(tag, method=None):
    out = {}
    for f in sorted((RES / tag).glob("*.json")):
        d = json.loads(f.read_text())
        if "mol" not in d or "basis" not in d:
            continue
        out[(d["mol"], d["basis"], d.get("method", method), d["calc"])] = d
    return out


def load_cosx_perf():
    """bench/grad_bench.py results: {(mol, basis, method, calc): {"runs": {prog: [run]}}}."""
    out = {}
    for f in sorted((RES / "2026-11-07-cosx-perf").glob("*.json")):
        d = json.loads(f.read_text())
        method, b = d["level"].split("-", 1)
        runs = {}
        for prog, key in (("linus", "linus-cosx (cosx-perf)"), ("orca", "orca (cosx-perf)")):
            rs = []
            for r in d["runs"].get(prog, []):
                r = dict(r)
                if r.get("gradient") is not None:
                    r["gradient_no_disp"] = r["gradient"]
                rs.append(r)
            runs[key] = rs
        out[(d["mol"], "def2-" + b, method, d["calc"])] = {"runs": runs}
    return out


def best(runs):
    ok = [r for r in runs if r.get("ok")]
    return min(ok, key=lambda r: r["wall"]) if ok else None


def maxdiff(a, b):
    return max(abs(x - y) for ra, rb in zip(a, b) for x, y in zip(ra, rb))


def grad(r):
    return r.get("gradient_no_disp") or r.get("gradient")


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "2026-11-07-rik-scale"
    timed = {k: {"runs": dict(d["runs"])} for k, d in load(tag).items()}
    for k, d in load("2026-11-07-wb97x-d3bj", "wb97x-d3bj").items():
        if any(kk[:2] == k[:2] and kk[2] == "wb97x-d3bj" for kk in timed):
            for p in ("orca", "orca-v2"):
                if p in d["runs"]:
                    timed.setdefault(k, {"runs": {}})["runs"]["orca (wb97x-d3bj round)"] = d["runs"][p]
    for k, d in load_cosx_perf().items():
        if any(kk[:3] == k[:3] for kk in timed):
            timed.setdefault(k, {"runs": {}})["runs"].update(d["runs"])
    refs = {**load("2026-11-07-wb97x-d3bj-ref", "wb97x-d3bj"), **load(tag + "-ref")}
    print("| molecule | method | calc | program | 3-index storage | wall (s) | SCF it | peak RSS (GB) | cap (GB) | E - E(exact) (Eh) | max dG vs exact (Eh/bohr) |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for (mol, basis, method, calc), d in sorted(timed.items()):
        ref = refs.get((mol, basis, method, calc)) or refs.get((mol, basis, method, "energy")) or refs.get((mol, basis, method, "gradient"))
        rx = best(ref["runs"].get("linus-exact", [])) if ref else None
        for prog, runs in sorted(d["runs"].items()):
            r = best(runs)
            if r is None:
                err = next((x.get("error", "") for x in runs if not x.get("ok")), "")
                print(f"| {mol} {basis} | {method} | {calc} | {prog} | | fails: {err[:70]!r} | | | | | |")
                continue
            mode = ""
            t = r.get("timings", {})
            if "linus-ri" in prog and "df_bytes_per_elem" in t:
                bpe = round(t["df_bytes_per_elem"])
                kind = "C (one-sided)" if t.get("df_one_sided") else ("B+B^w" if method.startswith("wb97") else "B")
                mode = f"{kind} {['', '', '', '', 'f32', 'f40', 'f48', '', 'f64'][bpe]}, {t['df_store_gib']:.1f} GiB"
            de = f"{r['energy'] - rx['energy']:+.2e}" if rx and "orca" not in prog else ""
            dg = ""
            if rx and grad(r) and grad(rx) and "orca" not in prog:
                dg = f"{maxdiff(grad(r), grad(rx)):.1e}"
            rss = f"{r['peak_rss_gb']:.1f}" if r.get("peak_rss_gb") else ""
            cap = r.get("mem_cap_gb", "")
            print(f"| {mol} {basis} | {method} | {calc} | {prog} | {mode} | {r['wall']:.1f} | {r.get('iterations', '')} | {rss} | {cap} | {de} | {dg} |")


if __name__ == "__main__":
    main()
