"""RHF energies: linus (FastJk) against PySCF (direct SCF), both tightly converged.

python bench/energy_check.py [--threads 8] [--thresh 1e-14] [--out file.json] [mol:basis ...]

Gate: |E_linus - E_pyscf| <= 1e-11 Eh. Also runs linus with the reference IncoreJk builder for
small systems. Writes bench/results/energy_check.json (no host names).
"""
import argparse
import json
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
PY = sys.executable
SCF = ROOT / "target/release/examples/scf"

DEFAULT = ["water:def2-tzvp", "nh3:def2-tzvp", "water:cc-pvtz", "nh3:def2-qzvp", "caffeine:def2-svp", "caffeine:def2-tzvp"]


CAP = ["systemd-run", "--user", "--scope", "-q", "-p", "MemoryMax=12G"]  # shared node


def run(cmd, env=None):
    out = subprocess.run(CAP + cmd, capture_output=True, text=True, env=env, check=True).stdout
    return json.loads(out.strip().splitlines()[-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cases", nargs="*", default=DEFAULT)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--thresh", type=float, default=1e-14)
    ap.add_argument("--out", default=str(ROOT / "bench/results/energy_check.json"))
    a = ap.parse_args()
    import os
    rows = []
    for case in a.cases:
        mol, basis = case.split(":")
        xyz = str(ROOT / f"tests/molecules/{mol}.xyz")
        # the converged SCF energy of a ~700 Eh molecule scatters by ~1e-11 between SCF paths at
        # e_conv 1e-12 (any J/K builder): converge to 1e-13
        conv = ["--e-conv", "1e-13", "--g-conv", "1e-9"]
        lin = run([str(SCF), xyz, basis, "--jk", "fast", "--thresh", str(a.thresh), "--threads", str(a.threads)] + conv)
        env = dict(os.environ, OMP_NUM_THREADS=str(a.threads))
        ref = run([PY, str(ROOT / "bench/pyscf_jk.py"), "rhf", xyz, basis, str(a.threads), str(a.thresh), "1e-13", "1e-9", "full"], env)
        row = {"molecule": mol, "basis": basis, "nbf": lin["nbf"], "thresh": a.thresh,
               "e_linus_fast": lin["energy"], "e_pyscf": ref["energy"],
               "diff_fast_pyscf": lin["energy"] - ref["energy"],
               "converged": [lin["converged"], ref["converged"]]}
        if lin["nbf"] <= 100:  # IncoreJk holds nbf^4 doubles (0.8 GB at 100)
            inc = run([str(SCF), xyz, basis, "--jk", "incore", "--threads", str(a.threads)] + conv)
            row["e_linus_incore"] = inc["energy"]
            row["diff_fast_incore"] = lin["energy"] - inc["energy"]
        row["pass"] = abs(row["diff_fast_pyscf"]) <= 1e-11
        rows.append(row)
        print(json.dumps(row), flush=True)
    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"pyscf_version": __import__("pyscf").__version__, "rows": rows}, indent=1))
    print("all pass" if all(r["pass"] for r in rows) else "FAILURES", file=sys.stderr)


if __name__ == "__main__":
    main()
