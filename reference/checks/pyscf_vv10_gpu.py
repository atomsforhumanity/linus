#!/usr/bin/env python3
"""The stored PySCF VV10 comparisons (crates/linus-py/tests/data/pyscf_vv10.json, the per-push
test_vv10_functionals_match_stored_pyscf) on another backend, through the CLI: NH3 and its cation
with wB97M-V, wB97X-V and B97M-V, energies, VV10 energies and grid-response gradients at the test's
settings (def2-SVP, PySCF grid level 3 and its NLC grid, exact J/K, `extreme` SCF convergence, 4
threads) and tolerances (energy 1e-9 Eh, VV10 energy 1e-10 Eh, gradient 1e-8 Eh/bohr).

    python3 reference/checks/pyscf_vv10_gpu.py --linus-bin target/release/linus --backend gpu \\
        [--prefix 'taskset -c 40-47'] [--set precision=fp64]

Prints one line per case and the largest differences; exits with the number of failed cases.
"""
import argparse
import json
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "crates/linus-py/tests/data/pyscf_vv10.json"
BOHR = 0.529177210544
TOL = {"energy": 1e-9, "e_nlc": 1e-10, "gradient": 1e-8}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--linus-bin", default=str(REPO / "target/release/linus"))
    ap.add_argument("--backend", default="gpu")
    ap.add_argument("--prefix", default="")
    ap.add_argument("--set", action="append", default=[], help="more --set KEY=VALUE for every run")
    a = ap.parse_args()
    data = json.loads(DATA.read_text())
    failed = 0
    with tempfile.TemporaryDirectory() as tmp:
        for key, case in sorted(data["cases"].items()):
            inp = case["inputs"]
            xyz = Path(tmp) / "mol.xyz"
            lines = [f"{s} {x * BOHR:.15f} {y * BOHR:.15f} {z * BOHR:.15f}"
                     for s, (x, y, z) in zip(inp["symbols"], inp["geometry_bohr"])]
            xyz.write_text(f"{len(lines)}\n{key}\n" + "\n".join(lines) + "\n")
            cmd = shlex.split(a.prefix) + [
                a.linus_bin, str(xyz), "--method", inp["xc"], "--basis", "def2-svp", "--grid", "pyscf:3",
                "--charge", str(inp["charge"]), "--mult", str(inp["multiplicity"]), "--threads", "4",
                "--backend", a.backend, "--set", "scf.convergence=extreme", "--grad", "--json"]
            for s in a.set:
                cmd += ["--set", s]
            p = subprocess.run(cmd, capture_output=True, text=True)
            try:
                r = json.loads(p.stdout)
            except json.JSONDecodeError:
                r = {"success": False, "error": p.stderr[-500:]}
            if not r.get("success"):
                print(f"FAIL {key}: {r.get('error')}")
                failed += 1
                continue
            d = {"energy": abs(r["energy"] - case["e_tot"]),
                 "e_nlc": abs(r["vv10"]["energy"] - case["e_nlc"]),
                 "gradient": max(abs(x - y) for ra, ca in zip(r["gradient"], case["gradient"])
                                 for x, y in zip(ra, ca))}
            ok = all(d[k] <= TOL[k] for k in TOL)
            failed += not ok
            rec = r["resolved"].get("record") or {}
            vv = rec["backend"].get("vv10") if rec.get("backend") else None
            print(f"{'ok  ' if ok else 'FAIL'} {key:24s} dE {d['energy']:.1e}  dE_vv10 {d['e_nlc']:.1e}  "
                  f"dG {d['gradient']:.1e}  vv10 on {vv}", flush=True)
    print(f"{len(data['cases']) - failed} of {len(data['cases'])} cases within the stored test's tolerances")
    return failed


if __name__ == "__main__":
    sys.exit(main())
