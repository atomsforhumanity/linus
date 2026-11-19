"""Model-level comparison with Psi4 1.11 + PCMSolver (C-PCM and IEF-PCM, GePol cavity of Bondi
radii x 1.2 with H 1.2 Å, point charges on 0.3 Å^2 tesserae) against linus (Gaussian charges on
Lebedev spheres; `pyscf` preset, and the same with H 1.2 Å to match PCMSolver's radii).

Reads bench/results/2026-11-07-pcm/psi4/*.json (bench/pcm_psi4.py: geometry, energies with and
without PCM, the PCM polarization energy) and runs linus at the same eps with exact J/K (Psi4:
scf_type pk) and HF or B3LYP (VWN-RPA, as Psi4's b3lyp) on PySCF's level-3 grid. Reports the
solvation shift E(PCM) - E(gas) of both and its difference (kcal/mol).

  .refenv/bin/python bench/pcm_psi4_compare.py --out bench/results/2026-11-07-pcm/psi4_compare.json [--slice ccd3]
"""
import argparse, glob, json, pathlib, subprocess

ROOT = pathlib.Path(__file__).resolve().parents[1]
KCAL = 627.5094740631


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--slice", default="ccd3")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--bin", default=str(ROOT / "target/release/linus"))
    a = ap.parse_args()

    def run(req):
        cmd = ["systemd-run", "--user", "--scope", "-q", "-p", "MemoryMax=16G", "-p", "MemorySwapMax=0",
               str(pathlib.Path.home() / ".local/bin/slice"), a.slice, a.bin, "--json"]
        return json.loads(subprocess.run(cmd, input=json.dumps(req), capture_output=True, text=True, check=True).stdout)

    rows = []
    gas_cache = {}
    for f in sorted(glob.glob(str(ROOT / "bench/results/2026-11-07-pcm/psi4/*_*_*_*.json"))):
        ref = json.load(open(f))
        if "energy" not in ref:
            continue
        solver = "cpcm" if "CPCM" in ref["case"] else "iefpcm"
        eps = float(ref["pcm_input"].split("Eps = ")[1].split()[0]) if "Eps = " in ref["pcm_input"] else 78.39
        base = {"structure": {"symbols": [g[0] for g in ref["geometry_bohr"]], "geometry": [g[1:4] for g in ref["geometry_bohr"]],
                              "charge": ref["charge"], "multiplicity": ref["mult"]},
                "method": ref["method"], "basis": ref["basis"].lower(), "threads": a.threads, "jk": "exact",
                "grid": {"xc": "pyscf:3"}, "scf": {"convergence": "tight"}}
        key = (ref["mol"], ref["method"])
        if key not in gas_cache:
            gas_cache[key] = run(base)["energy"]
        e_gas = gas_cache[key]
        row = {"case": ref["case"], "eps": eps, "psi4_delta_solv_kcal": ref["delta_solv"] * KCAL, "psi4_polarization_kcal": ref["pcm_polarization_energy"] * KCAL}
        for tag, extra in [("linus_pyscf", {}), ("linus_pyscf_H1.2", {"radii": {"H": 1.2}})]:
            r = run({**base, "solvent": {"eps": eps, "model": solver, "preset": "pyscf", **extra}})
            ds = (r["energy"] - e_gas) * KCAL
            row[tag + "_delta_solv_kcal"] = ds
            row[tag + "_minus_psi4_kcal"] = ds - ref["delta_solv"] * KCAL
        rows.append(row)
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
        pathlib.Path(a.out).write_text(json.dumps({"rows": rows}, indent=1))


if __name__ == "__main__":
    main()
