"""Model-level comparison with ORCA 6.0.0 C-PCM: linus with the `orca` preset (and the `pyscf`
preset) at ORCA's permittivity and matched electronic-structure settings, against the ORCA
reference runs of bench/pcm_orca.py (bench/results/2026-11-07-pcm/orca/ref/*.json, which hold the
geometry, gas-phase and solvated energies, dielectric energies and gradients).

Matching: method (B3LYP/G = linus b3lyp, wB97X-D3BJ, HF), basis, J/K (NoRI = jk exact; RIJCOSX
def2/J DefGrid2 = rij-snk def2-universal-jfit cosx:2), XC grid DefGrid2 = orca:2. Reported per
case: the solvation shift E(solv) - E(gas) of both programs and its difference, the dielectric
energies, and the largest gradient difference (solvated, and solvated minus gas, which removes
the gas-phase differences of the two programs).

  .refenv/bin/python bench/pcm_orca_compare.py --out bench/results/2026-11-07-pcm/orca_compare.json \
      [--filter hf_svp_exact] [--presets orca,pyscf] [--slice ccd2] [--threads 8]
"""
import argparse, glob, json, os, pathlib, subprocess

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
REF = ROOT / "bench/results/2026-11-07-pcm/orca/ref"
SYM = {1: "H", 6: "C", 7: "N", 8: "O", 16: "S"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--filter", default="", help="comma-separated substrings a case name must contain one of")
    ap.add_argument("--presets", default="orca,pyscf")
    ap.add_argument("--slice", default="ccd2")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--bin", default=str(ROOT / "target/release/linus"))
    a = ap.parse_args()
    out = {"settings": vars(a), "load_average": os.getloadavg(), "cases": []}
    files = sorted(glob.glob(str(REF / "*.json")))
    pats = [p for p in a.filter.split(",") if p]
    for f in files:
        ref = json.load(open(f))
        if not ref.get("ok") or (pats and not any(p in ref["case"] for p in pats)):
            continue
        geom = ref["geometry_bohr"]
        syms = [g[0] if isinstance(g[0], str) else SYM[g[0]] for g in geom]
        pos = [g[1:4] for g in geom]
        base = {"structure": {"symbols": syms, "geometry": pos, "charge": ref["charge"], "multiplicity": ref["mult"]},
                "method": ref["method"], "basis": ref["basis"].lower(), "threads": a.threads, "calctype": "gradient",
                "grid": {"xc": "orca:2"}}
        if ref["jk"] == "exact":
            base["jk"] = "exact"
        else:
            base["jk"] = {"kind": "rij-snk", "aux": "def2-universal-jfit", "k_grid": "cosx:2"}

        def run(req):
            cmd = ["systemd-run", "--user", "--scope", "-q", "-p", "MemoryMax=24G", "-p", "MemorySwapMax=0",
                   str(pathlib.Path.home() / ".local/bin/slice"), a.slice, a.bin, "--json"]
            p = subprocess.run(cmd, input=json.dumps(req), capture_output=True, text=True)
            if p.returncode:
                raise RuntimeError(p.stderr[-1000:])
            return json.loads(p.stdout)

        gas = run(base)
        rec = {"case": ref["case"], "orca": {k: ref.get(k) for k in ("energy", "gas_energy", "delta_solv", "cpcm_dielectric_energy", "npoints", "area_bohr2", "surface_charge", "epsilon")},
               "linus_gas_energy": gas["energy"], "dE_gas": gas["energy"] - ref["gas_energy"]}
        g_ref, g_gas_ref = np.array(ref["gradient"]), np.array(ref["gas_gradient"])
        for preset in a.presets.split(","):
            r = run({**base, "solvent": {"eps": ref["epsilon"], "model": "cpcm", "preset": preset}})
            g = np.array(r["gradient"])
            g_gas = np.array(gas["gradient"])
            ds = r["energy"] - gas["energy"]
            rec[preset] = {"energy": r["energy"], "delta_solv": ds, "d_delta_solv_vs_orca": ds - ref["delta_solv"],
                           "dielectric": r["components"]["solvation_electrostatic"],
                           "d_dielectric_vs_orca": r["components"]["solvation_electrostatic"] - ref["cpcm_dielectric_energy"],
                           "npoints": r["solvation"]["npoints"], "area_bohr2": r["solvation"]["area_angstrom2"] / 0.529177210544 ** 2,
                           "surface_charge": r["solvation"]["surface_charge"],
                           "max_abs_dG_vs_orca": float(np.abs(g - g_ref).max()),
                           "max_abs_d(solvent part of G)_vs_orca": float(np.abs((g - g_gas) - (g_ref - g_gas_ref)).max())}
            print(ref["case"], preset, {k: (round(v, 9) if isinstance(v, float) else v) for k, v in rec[preset].items() if k.startswith(("d_", "max", "npoints"))}, flush=True)
        out["cases"].append(rec)
        pathlib.Path(a.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
