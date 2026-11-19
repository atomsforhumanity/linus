"""Discretization error of the PCM presets: solvation energies and gradients of the `pyscf` preset
(302 Lebedev points per sphere, SWIG, potential smeared over the Gaussians) and the `orca` preset
(ORCA's grids at 5 points per Å^2, ISWIG, potential at the points) against dense surfaces
(`pyscf` preset with N = 974..2030 points per sphere, SWIG and ISWIG), which converge to the same
sharp-cavity limit.

Level: B3LYP/def2-SVP RIJ+COSX (def2-universal-jfit, cosx:2, grid orca:2), C-PCM eps 7.4257 (THF),
energy + gradient; linus's default surface treatment for RIJ+COSX (the density-fitted PCM, the same
for every surface). E_solv = E(C-PCM) - E(gas) in kcal/mol; gradient deviations in Eh/bohr.

  ~/.local/bin/slice ccd2 .refenv/bin/python bench/pcm_discretization.py run --mols la18,u1an39,ts61 \
      --out bench/results/2026-11-08-pcm-perf/discretization.jsonl [--dense 974,2030]
  .refenv/bin/python bench/pcm_discretization.py table bench/results/2026-11-08-pcm-perf/discretization.jsonl
"""
import argparse, json, pathlib, subprocess, sys, time

ROOT = pathlib.Path(__file__).resolve().parents[1]
BOHR = 0.529177210544
HARTREE_KCAL = 627.5094740631
CHARGES = {"la18": 0, "u1an39": -1, "ts61": 0, "ts113": 0}
LEVEL = dict(method="b3lyp", basis="def2-svp", jk=dict(kind="rij-snk", aux="def2-universal-jfit", k_grid="cosx:2"), grid=dict(xc="orca:2"))


def variants(dense):
    out = [("gas", None), ("pyscf", {"preset": "pyscf"}), ("orca", {"preset": "orca"})]
    for n in dense:
        for disc in ["swig", "iswig"]:
            out.append((f"dense{n}-{disc}", {"preset": "pyscf", "lebedev": n, "discretization": disc}))
    return out


def run(a):
    for mol in a.mols.split(","):
        lines = (ROOT / "bench/molecules" / f"{mol}.xyz").read_text().splitlines()
        n = int(lines[0])
        syms = [ln.split()[0] for ln in lines[2:2 + n]]
        pos = [[float(v) / BOHR for v in ln.split()[1:4]] for ln in lines[2:2 + n]]
        done = set()
        if pathlib.Path(a.out).exists():
            for ln in open(a.out):
                r = json.loads(ln)
                done.add((r["mol"], r["variant"]))
        for name, sol in variants([int(x) for x in a.dense.split(",") if x]):
            if (mol, name) in done:
                continue
            req = {"structure": {"symbols": syms, "geometry": pos, "charge": CHARGES.get(mol, 0)}, "calctype": "gradient",
                   "threads": a.threads, "scf": {"convergence": "tight"}, **LEVEL}
            if sol is not None:
                req["solvent"] = {"eps": a.eps, "model": "cpcm", **sol}
            cmd = ["systemd-run", "--user", "--scope", "-q", "-p", f"MemoryMax={a.mem}G", "-p", "MemorySwapMax=0", "timeout", str(a.timeout), a.bin, "--json"]
            t0 = time.perf_counter()
            p = subprocess.run(cmd, input=json.dumps(req), capture_output=True, text=True)
            wall = time.perf_counter() - t0
            if p.returncode:
                print(mol, name, "FAILED", p.stderr[-500:], flush=True)
                continue
            r = json.loads(p.stdout)
            rec = dict(mol=mol, variant=name, solvent=req.get("solvent"), energy=r["energy"], gradient=r["gradient"], solvation=r.get("solvation"),
                       iterations=r["iterations"], wall=wall, time=time.strftime("%Y-%m-%dT%H:%M:%S"))
            with open(a.out, "a") as f:
                f.write(json.dumps(rec) + "\n")
            npt = (r.get("solvation") or {}).get("npoints")
            print(mol, name, f"E {r['energy']:.10f}", "npoints", npt, f"wall {wall:.1f}", flush=True)


def table(a):
    import numpy as np
    recs = {}
    for ln in open(a.file):
        r = json.loads(ln)
        recs[(r["mol"], r["variant"])] = r
    mols = sorted({k[0] for k in recs}, key=lambda m: ["la18", "u1an39", "ts61", "ts113"].index(m) if m in CHARGES else 9)
    rows = []
    for mol in mols:
        gas = recs.get((mol, "gas"))
        if gas is None:
            continue
        dense = sorted([k[1] for k in recs if k[0] == mol and k[1].startswith("dense")], key=lambda v: (int(v[5:].split("-")[0]), v))
        if not dense:
            continue
        # reference: the given surface, else the densest ISWIG one (SWIG and ISWIG converge to
        # the same sharp-cavity limit; their difference at the densest grid is printed below)
        iswig = [v for v in dense if v.endswith("-iswig")]
        ref_name = a.ref if (mol, a.ref) in recs else (iswig[-1] if iswig else dense[-1])
        ref = recs[(mol, ref_name)]
        es_ref = (ref["energy"] - gas["energy"]) * HARTREE_KCAL
        g_ref = np.array(ref["gradient"])
        gs_ref = g_ref - np.array(gas["gradient"])
        for v in ["pyscf", "orca"] + dense:
            r = recs.get((mol, v))
            if r is None:
                continue
            es = (r["energy"] - gas["energy"]) * HARTREE_KCAL
            g = np.array(r["gradient"])
            dg = g - g_ref
            gs = g - np.array(gas["gradient"])
            rows.append((mol, v, r["solvation"]["npoints"], es, es - es_ref, (es - es_ref) / abs(es_ref), np.abs(dg).max(), np.sqrt((dg ** 2).mean()), np.abs(gs).max(), ref_name))
    print("| molecule | surface | points | E_solv (kcal/mol) | E_solv - E_solv(ref) (kcal/mol) | relative | max abs dG vs ref (Eh/bohr) | rms dG (Eh/bohr) | max abs solvent part of G (Eh/bohr) | ref |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for m, v, npt, es, de, rel, gmax, grms, gsmax, rn in rows:
        print(f"| {m} | {v} | {npt} | {es:.3f} | {de:+.3f} | {100 * rel:+.2f}% | {gmax:.1e} | {grms:.1e} | {gsmax:.1e} | {rn} |")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--mols", default="la18")
    r.add_argument("--out", required=True)
    r.add_argument("--dense", default="974,2030")
    r.add_argument("--eps", type=float, default=7.4257)
    r.add_argument("--threads", type=int, default=8)
    r.add_argument("--mem", type=int, default=48)
    r.add_argument("--timeout", type=int, default=7200)
    r.add_argument("--bin", default=str(ROOT / "target/release/linus"))
    t = sub.add_parser("table")
    t.add_argument("file")
    t.add_argument("--ref", default="auto", help="reference surface (default: the densest ISWIG one per molecule)")
    a = ap.parse_args()
    {"run": run, "table": table}[a.cmd](a)


if __name__ == "__main__":
    main()
