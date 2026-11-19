"""Old vs new linus binary on energies and analytic gradients (SCF-change validation).

Systems: water, CH3SH, CH3SH+ (doublet, UKS/UHF), la18, u1an39 (charge -1); methods HF,
B3LYP, wB97X-D3BJ; J/K: exact, df (def2-universal-jkfit) and rij-snk (def2-universal-jfit,
COSX grids of DefGrid2); def2-SVP, XC grid orca:2, 8 threads, gradient calctype at the default
gradient thresholds (|dE| < 1e-9, max |F_ai| < 1e-7). Records energies, gradients, Fock builds and
the differences new - old. Untimed (development slice).

usage: python3 bench/scf_change_validate.py --bins old=PATH,new=PATH --out FILE [--slice cpus:LIST] [--systems ...]
"""
import argparse, json, pathlib, subprocess, time

ROOT = pathlib.Path(__file__).resolve().parents[1]
SYSTEMS = {"water": ("tests/molecules/water.xyz", 0, 1), "ch3sh": ("tests/molecules/ch3sh.xyz", 0, 1),
           "ch3sh+": ("tests/molecules/ch3sh.xyz", 1, 2), "la18": ("bench/molecules/la18.xyz", 0, 1),
           "u1an39": ("bench/molecules/u1an39.xyz", -1, 1)}
METHODS = ["hf", "b3lyp", "wb97x-d3bj"]
JKS = ["exact", "df", "rij-snk"]


def run(binary, sysname, method, jk, slc, mem):
    path, charge, mult = SYSTEMS[sysname]
    req = {"structure": {"xyz": (ROOT / path).read_text(), "charge": charge, "multiplicity": mult}, "method": method,
           "basis": "def2-svp", "threads": 8, "grid": {"xc": "orca:2"}, "jk": jk, "calctype": "gradient"}
    cmd = ["systemd-run", "--user", "--scope", "-q", "-p", f"MemoryMax={mem}G", "-p", "MemorySwapMax=0", "-p", "OOMPolicy=continue"]
    cmd += ["taskset", "-c", slc[5:]] if slc.startswith("cpus:") else [str(pathlib.Path.home() / ".local/bin/slice"), slc]
    t = time.perf_counter()
    p = subprocess.run(cmd + [binary, "--json"], input=json.dumps(req), capture_output=True, text=True)
    wall = time.perf_counter() - t
    if p.returncode != 0:
        return dict(ok=False, wall=wall, error=p.stderr[-1500:])
    d = json.loads(p.stdout)
    return dict(ok=True, wall=wall, energy=d["energy"], gradient=d["gradient"], builds=d["scf"]["fock_builds"], final_gmax=d["scf"]["history"][-1]["gmax"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bins", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--slice", default="cpus:56-63")
    ap.add_argument("--mem", type=int, default=24)
    ap.add_argument("--systems", default=",".join(SYSTEMS))
    ap.add_argument("--jks", default=",".join(JKS))
    a = ap.parse_args()
    bins = dict(kv.split("=", 1) for kv in a.bins.split(","))
    out = pathlib.Path(a.out)
    res = json.loads(out.read_text()) if out.exists() else {"conditions": __doc__.strip(), "slice": a.slice, "results": {}}
    for s in a.systems.split(","):
        for m in METHODS:
            for jk in a.jks.split(","):
                key = f"{s}:{m}:{jk}"
                rr = {n: run(b, s, m, jk, a.slice, a.mem) for n, b in bins.items()}
                if all(r["ok"] for r in rr.values()) and {"old", "new"} <= rr.keys():
                    o, n = rr["old"], rr["new"]
                    rr["diff"] = dict(de=n["energy"] - o["energy"], max_dg=max(abs(x - y) for go, gn in zip(o["gradient"], n["gradient"]) for x, y in zip(go, gn)),
                                      bitwise=n["energy"] == o["energy"] and n["gradient"] == o["gradient"], builds=(o["builds"], n["builds"]))
                res["results"][key] = rr
                out.write_text(json.dumps(res, indent=1))
                d = rr.get("diff", {})
                runs = [rr[n] for n in bins]
                print(f"{key:28s} builds {d.get('builds')} dE {d.get('de', float('nan')):+.1e} max|dG| {d.get('max_dg', float('nan')):.1e} bitwise {d.get('bitwise')}"
                      + ("" if all(r["ok"] for r in runs) else "  FAILED: " + " | ".join(r.get("error", "")[-200:] for r in runs if not r["ok"])), flush=True)


if __name__ == "__main__":
    main()
