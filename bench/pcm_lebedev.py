"""Convergence of the PCM discretization: solvation energies and gradients against the number of
Lebedev points per sphere, for SWIG and ISWIG switching (linus CLI, C-PCM unless --model).

For each molecule and switching scheme, runs a gradient at every Lebedev size and reports the
electrostatic solvation energy, total energy, gradient and PCM time, with differences from the
largest grid. Development timings (unlocked; the conditions are recorded).

  .refenv/bin/python bench/pcm_lebedev.py --out bench/results/2026-11-07-pcm/lebedev_convergence.json \
      [--mols la18,u1an39] [--sizes 50,86,110,146,194,302,590,974] [--method b3lyp] [--basis def2-svp] \
      [--slice ccd3] [--threads 8]
"""
import argparse, json, os, pathlib, subprocess

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
BOHR = 0.529177210544
CHARGES = {"la18": 0, "u1an39": -1, "ts61": 0, "ts113": 0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--mols", default="la18,u1an39")
    ap.add_argument("--sizes", default="50,86,110,146,194,302,590,974")
    ap.add_argument("--schemes", default="swig,iswig")
    ap.add_argument("--method", default="b3lyp")
    ap.add_argument("--basis", default="def2-svp")
    ap.add_argument("--model", default="cpcm")
    ap.add_argument("--eps", type=float, default=7.4257)
    ap.add_argument("--slice", default="ccd3")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--bin", default=str(ROOT / "target/release/linus"))
    a = ap.parse_args()
    out = {"settings": vars(a), "load_average": os.getloadavg(),
           "slice_list": subprocess.run([str(pathlib.Path.home() / ".local/bin/slice"), "list"], capture_output=True, text=True).stdout,
           "results": []}
    for mol in a.mols.split(","):
        lines = (ROOT / "bench/molecules" / f"{mol}.xyz").read_text().splitlines()
        n = int(lines[0])
        syms = [ln.split()[0] for ln in lines[2:2 + n]]
        pos = [[float(v) / BOHR for v in ln.split()[1:4]] for ln in lines[2:2 + n]]
        for scheme in a.schemes.split(","):
            rows = []
            for nleb in [int(x) for x in a.sizes.split(",")]:
                req = {"structure": {"symbols": syms, "geometry": pos, "charge": CHARGES.get(mol, 0)}, "method": a.method,
                       "basis": a.basis, "jk": "rij-snk",
                       "grid": {"xc": "orca:2"}, "threads": a.threads, "calctype": "gradient",
                       "solvent": {"eps": a.eps, "model": a.model, "lebedev": nleb, "discretization": scheme}}
                cmd = ["systemd-run", "--user", "--scope", "-q", "-p", "MemoryMax=40G", "-p", "MemorySwapMax=0",
                       str(pathlib.Path.home() / ".local/bin/slice"), a.slice, a.bin, "--json"]
                r = json.loads(subprocess.run(cmd, input=json.dumps(req), capture_output=True, text=True, check=True).stdout)
                t = r["timings"]
                rows.append({"lebedev": nleb, "energy": r["energy"], "solvation_electrostatic": r["components"]["solvation_electrostatic"],
                             "npoints": r["solvation"]["npoints"], "area_angstrom2": r["solvation"]["area_angstrom2"],
                             "surface_charge": r["solvation"]["surface_charge"], "gradient": r["gradient"],
                             "iterations": r["iterations"], "t_total": t["total"],
                             "t_pcm_scf": sum(v for k, v in t.items() if k.startswith("pcm_")), "t_grad_solvent": t.get("grad_solvent", 0.0)})
                print(mol, scheme, nleb, rows[-1]["solvation_electrostatic"], rows[-1]["npoints"], round(rows[-1]["t_pcm_scf"], 2), flush=True)
            ref = rows[-1]
            for x in rows:
                x["dE_solv_vs_largest"] = x["solvation_electrostatic"] - ref["solvation_electrostatic"]
                x["max_abs_dG_vs_largest"] = float(np.abs(np.array(x["gradient"]) - np.array(ref["gradient"])).max())
            out["results"].append({"mol": mol, "scheme": scheme, "rows": rows})
            pathlib.Path(a.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
