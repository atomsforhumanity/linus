"""Control for the RIJ+COSX SCF that did not converge on u1an39/def2-TZVPD with wB97M-V: the same
SCF with other functionals (B3LYP without VV10, wB97M-D3BJ without VV10, wB97M-V) and with RI-JK,
recording every iteration's energy and max |F_ai|.

  .refenv/bin/python bench/cosx_tzvpd_control.py --linus-bin BIN --slice ccd4 \
      [--runs b3lyp:cosx,wb97m-d3bj:cosx,wb97m-v:cosx,wb97m-v:ri] [--max-iter 60]

Settings as in the benchmark: XC grid orca:3, COSX grids cosx:3 with def2-universal-jfit, or RI-JK
with def2-universal-jkfit; 8 threads; the energy SCF thresholds. One JSON line per run appended to
bench/results/2026-11-07-vv10/cosx_tzvpd_control.jsonl.
"""
import argparse, json, pathlib, subprocess, time

ROOT = pathlib.Path(__file__).resolve().parents[1]
BOHR = 0.529177210544


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--linus-bin", required=True)
    ap.add_argument("--slice", default="ccd4")
    ap.add_argument("--mol", default="u1an39")
    ap.add_argument("--charge", type=int, default=-1)
    ap.add_argument("--basis", default="def2-tzvpd")
    ap.add_argument("--runs", default="b3lyp:cosx,wb97m-d3bj:cosx,wb97m-v:cosx,wb97m-v:ri")
    ap.add_argument("--max-iter", type=int, default=60)
    ap.add_argument("--label", default="", help="which build (recorded)")
    a = ap.parse_args()
    lines = (ROOT / "bench/molecules" / f"{a.mol}.xyz").read_text().splitlines()
    n = int(lines[0])
    atoms = [(l.split()[0], [float(x) / BOHR for x in l.split()[1:4]]) for l in lines[2:2 + n]]
    out = ROOT / "bench/results/2026-11-07-vv10/cosx_tzvpd_control.jsonl"
    for run in a.runs.split(","):
        method, jk = run.split(":")
        req = {"structure": {"symbols": [s for s, _ in atoms], "geometry": [c for _, c in atoms], "charge": a.charge,
                             "multiplicity": 1},
               "calctype": "energy", "method": method, "basis": a.basis, "grid": {"xc": "orca:3"}, "threads": 8,
               "scf": {"max_iter": a.max_iter}, "output": {"verbose": True}}
        if jk == "cosx":
            req["jk"] = {"kind": "rij-snk", "aux": "def2-universal-jfit", "k_grid": "cosx:3"}
        else:
            req["jk"] = {"kind": "df", "aux": "def2-universal-jkfit"}
        cmd = ["systemd-run", "--user", "--scope", "-q", "-p", "MemoryMax=48G", "-p", "MemorySwapMax=0", "-p", "OOMPolicy=continue",
               str(pathlib.Path.home() / ".local/bin/slice"), a.slice, a.linus_bin, "--json"]
        t = time.perf_counter()
        p = subprocess.run(cmd, input=json.dumps(req), capture_output=True, text=True)
        wall = time.perf_counter() - t
        rec = {"mol": a.mol, "basis": a.basis, "method": method, "jk": jk, "max_iter": a.max_iter, "label": a.label, "wall": wall, "ok": p.returncode == 0,
               "load": open("/proc/loadavg").read().split()[:3]}
        if p.returncode == 0:
            d = json.loads(p.stdout)
            rec.update(energy=d["energy"], converged=d["converged"], iterations=d["iterations"],
                       history=[{k: h.get(k) for k in ("iter", "energy", "de", "gmax", "ddmax", "step")} for h in d["scf"]["history"]])
        else:
            # the per-iteration progress lines (verbose) of the unconverged SCF
            rec["error"] = (p.stderr or "")[-12000:]
        with open(out, "a") as f:
            f.write(json.dumps(rec) + "\n")
        last = rec.get("history", [{}])[-1] if rec.get("history") else {}
        print(f"{method:11} {jk:5} ok={rec['ok']} conv={rec.get('converged')} it={rec.get('iterations')} wall={wall:.0f}s "
              f"last gmax={last.get('gmax')} de={last.get('de')} {rec.get('error', '')[-300:]}", flush=True)


if __name__ == "__main__":
    main()
