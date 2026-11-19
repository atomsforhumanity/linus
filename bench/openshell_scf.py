"""Open-shell SCF set: Fock builds, energies, <S^2> and final orbital gradients of linus binaries on
UHF/UKS cases (bench/results/2026-11-07-openshell-scf).

Case id: <molecule>:<method>:<basis>:<jk>:<thresholds>[:<grid>] with jk exact | df
(def2-universal-jkfit) | cosx (rij-snk: def2-universal-jfit, COSX grids of DefGrid2) and thresholds
energy (default for energies: |dE| < 1e-8, max |F_ai| < 5e-7), gradient (1e-9, 1e-7), hessian
(1e-9, 1e-9) or extreme (1e-12, 1e-10). Grid: pyscf:3 unless given (cosx cases: orca:2). 8 threads.

Untimed: runs on a development slice without the bench lock; wall times are recorded but are not
benchmarks. Non-converged runs are recorded with their verbose iteration trace (stderr).

usage: python3 bench/openshell_scf.py --bins old=PATH,new=PATH --out FILE [--slice ccd5] [--cases ...|--set NAME]
"""
import argparse, json, pathlib, re, subprocess, time

ROOT = pathlib.Path(__file__).resolve().parents[1]
MOLS = {
    # linear radicals with a degenerate pi hole (2Pi): HBr+ (no ECP), HI+ (def2 ECP28MWB), SH
    "hbr+": ("bench/molecules/hbr_1414.xyz", 1, 2),
    "hbr+146": ("bench/molecules/hbr_146.xyz", 1, 2),
    "hbr+146t": ("bench/molecules/hbr_146_tilt.xyz", 1, 2),
    "hi+": ("tests/molecules/hi.xyz", 1, 2),
    "sh": ("tests/molecules/sh.xyz", 0, 2),
    # radical cation of L-lactide, lactide radical, small radicals and triplets
    "la18+": ("bench/molecules/la18.xyz", 1, 2),
    "la17r": ("bench/molecules/la17r.xyz", 0, 2),
    "ch2": ("tests/molecules/ch2.xyz", 0, 3),
    "ch3": ("tests/molecules/ch3.xyz", 0, 2),
    "nh3+": ("tests/molecules/nh3_cation.xyz", 1, 2),
    "nh3+v": ("tests/molecules/nh3.xyz", 1, 2),  # reference/validate.py's NH3+
    "ch3sh+": ("tests/molecules/ch3sh.xyz", 1, 2),
    "o2": ("tests/molecules/o2.xyz", 0, 3),
    "ph2": ("tests/molecules/ph2.xyz", 0, 2),
    "ch2i": ("tests/molecules/ch2i.xyz", 0, 2),
    "au2+": ("tests/molecules/au2.xyz", 1, 2),
}
THRESH = {"energy": {}, "gradient": {"e_tol": 1e-9, "g_tol": 1e-7}, "hessian": {"e_tol": 1e-9, "g_tol": 1e-9},
          "extreme": {"convergence": "extreme"}}  # scf settings
SETS = {
    # the reported failures
    "failures": ["hbr+:b3lyp:def2-svp:df:hessian", "hbr+146:b3lyp:def2-svp:df:hessian", "hbr+146:b3lyp:def2-svp:exact:hessian",
                 "hi+:b3lyp:def2-svp:df:hessian", "hi+:b3lyp:def2-svp:exact:hessian",
                 "la18+:wb97x-d3bj:def2-tzvp:exact:energy", "la18+:wb97x-d3bj:def2-tzvp:df:energy", "la18+:wb97x-d3bj:def2-tzvp:cosx:energy"],
    # open shells that converge (must keep their builds and energies)
    "regression": [
        "hbr+:b3lyp:def2-svp:df:energy", "hbr+:b3lyp:def2-svp:df:gradient", "hbr+:hf:def2-svp:exact:hessian", "hbr+:b3lyp:def2-svp:exact:hessian",
        "hbr+146t:b3lyp:def2-svp:df:hessian", "hi+:hf:def2-svp:exact:hessian", "sh:b3lyp:def2-svp:exact:gradient", "sh:hf:def2-svp:df:hessian",
        "la17r:b3lyp:def2-svp:df:energy", "la17r:b3lyp:def2-svp:cosx:gradient", "la17r:wb97x-d3bj:def2-tzvp:df:energy",
        "la18+:b3lyp:def2-svp:df:gradient", "la18+:wb97x-d3bj:def2-svp:cosx:gradient", "la18+:hf:def2-svp:df:energy",
        "ch2:hf:def2-svp:exact:hessian", "ch2:b3lyp:def2-tzvp:df:gradient", "ch3:b3lyp:def2-svp:exact:hessian", "ch3:pbe:def2-svp:df:gradient",
        "nh3+:b3lyp:def2-svp:exact:hessian", "nh3+:wb97x-d3bj:def2-svp:df:gradient", "nh3+v:hf:def2-svp:exact:extreme",
        "nh3+v:b3lyp:def2-svp:exact:extreme", "nh3+v:b3lyp:def2-svp:df:extreme", "nh3+v:tpss:def2-svp:exact:extreme",
        "ch3sh+:hf:def2-svp:exact:gradient", "ch3sh+:b3lyp:def2-svp:df:gradient", "ch3sh+:wb97x-d3bj:def2-svp:cosx:gradient",
        "o2:b3lyp:def2-svp:exact:hessian", "o2:hf:def2-tzvp:df:gradient", "ph2:b3lyp:def2-svp:df:hessian",
        "ch2i:b3lyp:def2-svp:exact:extreme", "au2+:hf:def2-svp:exact:extreme", "au2+:b3lyp:def2-svp:exact:extreme"],
}


def request(case):
    mol, method, basis, jk, thr, *grid = case.split(":")
    path, charge, mult = MOLS[mol]
    req = {"structure": {"xyz": (ROOT / path).read_text(), "charge": charge, "multiplicity": mult}, "method": method,
           "basis": basis, "threads": 8, "jk": {"exact": "exact", "df": "df", "cosx": "rij-snk"}[jk],
           "scf": {"max_iter": 128}, "output": {"verbose": True}}
    if method != "hf":
        req["grid"] = {"xc": grid[0] if grid else ("orca:2" if jk == "cosx" else "pyscf:3")}
    req["scf"].update(THRESH[thr])
    return req


ITER = re.compile(r"^iter\s+(\d+)\s+E = (\S+)\s+dE = (\S+)\s+max\|\[F,D\]\| = (\S+)")


def run(binary, case, slc, mem):
    cmd = ["systemd-run", "--user", "--scope", "-q", "-p", f"MemoryMax={mem}G", "-p", "MemorySwapMax=0", "-p", "OOMPolicy=continue"]
    cmd += ["taskset", "-c", slc[5:]] if slc.startswith("cpus:") else [str(pathlib.Path.home() / ".local/bin/slice"), slc]
    t = time.perf_counter()
    p = subprocess.run(cmd + [binary, "--json"], input=json.dumps(request(case)), capture_output=True, text=True)
    wall = time.perf_counter() - t
    trace = [[int(m.group(1)), float(m.group(2)), float(m.group(4))] for m in map(ITER.match, p.stderr.splitlines()) if m]
    if p.returncode != 0:
        last = p.stderr.strip().splitlines()[-1] if p.stderr.strip() else ""
        return dict(ok=False, converged=False, wall=wall, error=last, builds=len(trace),
                    final_gmax=trace[-1][2] if trace else None, best_gmax=min((t[2] for t in trace), default=None),
                    energy_last=trace[-1][1] if trace else None, trace=trace)
    d = json.loads(p.stdout)
    h = d["scf"]["history"]
    occ = {s: d.get(f"occupations_{s}") for s in ("alpha", "beta")}
    return dict(ok=True, converged=d["converged"], wall=wall, energy=d["energy"], builds=d["scf"]["fock_builds"], iterations=d["iterations"],
                final_gmax=h[-1]["gmax"], s2=d.get("s2"), steps=[x["step"] for x in h], stability_checks=d["scf"].get("stability"),
                frontier={s: d[f"orbital_energies_{s}"][max(0, d[f"n{s}"] - 3):d[f"n{s}"] + 2] for s in ("alpha", "beta")},
                occ_frac=any(o not in (0.0, 1.0) for s in occ.values() if s for o in s), trace=trace)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bins", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--slice", default="ccd5")
    ap.add_argument("--mem", type=int, default=24)
    ap.add_argument("--set", default="regression")
    ap.add_argument("--cases", default=None)
    a = ap.parse_args()
    bins = dict(kv.split("=", 1) for kv in a.bins.split(","))
    out = pathlib.Path(a.out)
    res = json.loads(out.read_text()) if out.exists() else {"conditions": __doc__.strip(), "slice": a.slice, "results": {}}
    cases = a.cases.split(",") if a.cases else SETS[a.set]
    for case in cases:
        for name, b in bins.items():
            r = run(b, case, a.slice, a.mem)
            r["time"] = time.strftime("%Y-%m-%dT%H:%M:%S")
            res["results"].setdefault(case, {})[name] = r
            out.write_text(json.dumps(res, indent=1))
            e = r.get("energy", r.get("energy_last"))
            print(f"{case:44s} {name:5s} conv {str(r['converged']):5s} builds {r.get('builds')} E {e if e is None else f'{e:.10f}'} "
                  f"gmax {r.get('final_gmax') or 0:.1e} s2 {r.get('s2')} wall {r['wall']:.1f}", flush=True)


if __name__ == "__main__":
    main()
