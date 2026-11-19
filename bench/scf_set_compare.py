"""Fock builds of two linus binaries over the SCF benchmark set of bench/results/2026-11-07-scf-conv
(the same requests as bench/scf_compare.py: grid orca:2, tight preset, 8 threads; jk exact | df
(def2-universal-jkfit) | rij-snk (def2-universal-jfit, COSX grids of DefGrid2)), plus the RIJCOSX
cases at the gradient defaults (calctype gradient, SCF only: `--calc energy` with the gradient
thresholds e_tol 1e-9, g_tol 1e-7).

Untimed: runs on a development slice without the bench lock; wall times are recorded but are
not benchmarks. Writes one JSON with every run's builds, energy, final max |F_ai|, wall time and
SCF history.

usage: python3 bench/scf_set_compare.py --bins old=PATH,new=PATH --out FILE [--slice cpus:56-63] [--cases ...]
"""
import argparse, json, pathlib, subprocess, time

ROOT = pathlib.Path(__file__).resolve().parents[1]
MOLS = {p.stem: p for d in ("bench/molecules", "tests/molecules") for p in (ROOT / d).glob("*.xyz")}
CHARGES = {"u1an39": -1}
SET = ["caffeine:hf:def2-svp:exact", "cholesterol:hf:def2-svp:exact", "la18:b3lyp:def2-svp:cosx", "la18:b3lyp:def2-svp:df",
       "la18:b3lyp:def2-svp:exact", "la18:b3lyp:def2-tzvp:df", "la18:hf:def2-svp:df", "la18:hf:def2-svp:exact", "la18:hf:def2-tzvp:exact",
       "la18:pbe:def2-svp:df", "ts61:b3lyp:def2-svp:cosx", "ts61:b3lyp:def2-svp:df", "ts61:hf:def2-svp:df", "ts61:pbe:def2-svp:df",
       "u1an39:b3lyp:def2-svp:cosx", "u1an39:b3lyp:def2-svp:df", "u1an39:b3lyp:def2-tzvp:df", "u1an39:hf:def2-svp:df",
       "u1an39:hf:def2-svp:exact", "u1an39:pbe:def2-svp:df",
       # RIJCOSX at the gradient thresholds
       "la18:b3lyp:def2-svp:cosx:grad", "u1an39:b3lyp:def2-svp:cosx:grad", "ts61:b3lyp:def2-svp:cosx:grad"]


def run(binary, case, slc, mem):
    mol, method, basis, jk, *mode = case.split(":")
    req = {"structure": {"xyz": MOLS[mol].read_text(), "charge": CHARGES.get(mol, 0)}, "method": method, "basis": basis,
           "threads": 8, "grid": {"xc": "orca:2"}, "jk": {"exact": "exact", "df": "df", "cosx": "rij-snk"}[jk]}
    if mode == ["grad"]:
        req["scf"] = {"e_tol": 1e-9, "g_tol": 1e-7}
    else:
        req["scf"] = {"convergence": "tight"}
    cmd = ["systemd-run", "--user", "--scope", "-q", "-p", f"MemoryMax={mem}G", "-p", "MemorySwapMax=0", "-p", "OOMPolicy=continue"]
    cmd += ["taskset", "-c", slc[5:]] if slc.startswith("cpus:") else [str(pathlib.Path.home() / ".local/bin/slice"), slc]
    t = time.perf_counter()
    p = subprocess.run(cmd + [binary, "--json"], input=json.dumps(req), capture_output=True, text=True)
    wall = time.perf_counter() - t
    if p.returncode != 0:
        return dict(ok=False, wall=wall, error=p.stderr[-1500:])
    d = json.loads(p.stdout)
    h = d["scf"]["history"]
    return dict(ok=True, wall=wall, energy=d["energy"], builds=d["scf"]["fock_builds"], final_gmax=h[-1]["gmax"],
                k_time=d["timings"].get("jk"), scf_time=d["timings"].get("scf_total"), history=h)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bins", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--slice", default="cpus:56-63")
    ap.add_argument("--mem", type=int, default=20)
    ap.add_argument("--cases", default=",".join(SET))
    a = ap.parse_args()
    bins = dict(kv.split("=", 1) for kv in a.bins.split(","))
    out = pathlib.Path(a.out)
    res = json.loads(out.read_text()) if out.exists() else {"conditions": __doc__.strip(), "slice": a.slice, "results": {}}
    for case in a.cases.split(","):
        for name, b in bins.items():
            r = run(b, case, a.slice, a.mem)
            r["time"] = time.strftime("%Y-%m-%dT%H:%M:%S")
            res["results"].setdefault(case, {})[name] = r
            out.write_text(json.dumps(res, indent=1))
            print(f"{case:34s} {name:5s} builds {r.get('builds')} E {r.get('energy', float('nan')):.10f} gmax {r.get('final_gmax', 0):.1e} wall {r['wall']:.1f}", flush=True)


if __name__ == "__main__":
    main()
