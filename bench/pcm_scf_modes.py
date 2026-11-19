"""The SCF modes of the PCM surface terms against the exact two-pass SCF: the same calculation with
`solvent.scf_terms` = exact, predicted (one exact pass per iteration, RI-predicted Fock charges) and df
(the density-fitted PCM); energies, analytic gradients and SCF iterations.

Levels (C-PCM, eps 7.4257 unless --eps): b3lyp-svp and wb97x-tzvp with RIJ+COSX (def2-universal-jfit,
cosx:2, grid orca:2), and the same with RI-JK (def2-universal-jkfit): `-rijk`. Molecules from
bench/molecules (la18, u1an39, ts61, ts113) or bench/molecules/pcm (water, ch3sh, acetate, ...).

  ~/.local/bin/slice ccd2 .refenv/bin/python bench/pcm_scf_modes.py run --mols water,ch3sh,la18 \
      --levels b3lyp-svp --presets pyscf,orca --out bench/results/2026-11-08-pcm-perf/scf_modes.jsonl
  .refenv/bin/python bench/pcm_scf_modes.py table bench/results/2026-11-08-pcm-perf/scf_modes.jsonl
"""
import argparse, json, pathlib, subprocess, time

ROOT = pathlib.Path(__file__).resolve().parents[1]
BOHR = 0.529177210544
HARTREE_KCAL = 627.5094740631
CHARGES = {"la18": 0, "u1an39": -1, "ts61": 0, "ts113": 0, "acetate": -1, "nh4": 1}
LEVELS = {
    "b3lyp-svp": dict(method="b3lyp", basis="def2-svp", jk=dict(kind="rij-snk", aux="def2-universal-jfit", k_grid="cosx:2"), grid=dict(xc="orca:2")),
    "wb97x-tzvp": dict(method="wb97x-d3bj", basis="def2-tzvp", jk=dict(kind="rij-snk", aux="def2-universal-jfit", k_grid="cosx:2"), grid=dict(xc="orca:2")),
    "b3lyp-svp-rijk": dict(method="b3lyp", basis="def2-svp", jk="df", aux="def2-universal-jkfit", grid="orca:2"),
    "wb97x-tzvp-rijk": dict(method="wb97x-d3bj", basis="def2-tzvp", jk="df", aux="def2-universal-jkfit", grid="orca:2"),
    "hf-svp-rijk": dict(method="hf", basis="def2-svp", jk="df", aux="def2-universal-jkfit"),
}


def geometry(mol):
    for d in [ROOT / "bench/molecules", ROOT / "bench/molecules/pcm"]:
        f = d / f"{mol}.xyz"
        if f.exists():
            lines = f.read_text().splitlines()
            n = int(lines[0])
            return [ln.split()[0] for ln in lines[2:2 + n]], [[float(v) / BOHR for v in ln.split()[1:4]] for ln in lines[2:2 + n]]
    raise SystemExit(f"no geometry for {mol}")


def run(a):
    done = set()
    if pathlib.Path(a.out).exists():
        for ln in open(a.out):
            r = json.loads(ln)
            done.add((r["mol"], r["level"], r["preset"], r["scf"], r.get("model", "cpcm")))
    for mol in a.mols.split(","):
        syms, pos = geometry(mol)
        for level in a.levels.split(","):
            for preset in a.presets.split(","):
                for scf in a.modes.split(","):
                    key = (mol, level, preset, scf, a.model)
                    if key in done:
                        continue
                    # the mode: the exact or predicted SCF terms (solvent.scf_terms) or the density-fitted PCM (solvent.df)
                    mode = {"df": True} if scf.lower() in ("df", "ri") else {"scf_terms": scf}
                    req = {"structure": {"symbols": syms, "geometry": pos, "charge": CHARGES.get(mol, 0)}, "calctype": a.calctype,
                           "threads": a.threads, "scf": {"convergence": a.conv}, **LEVELS[level],
                           "solvent": {"eps": a.eps, "model": a.model, "preset": preset, **mode}}
                    cmd = ["systemd-run", "--user", "--scope", "-q", "-p", f"MemoryMax={a.mem}G", "-p", "MemorySwapMax=0", "timeout", str(a.timeout), a.bin, "--json"]
                    t0 = time.perf_counter()
                    p = subprocess.run(cmd, input=json.dumps(req), capture_output=True, text=True)
                    wall = time.perf_counter() - t0
                    if p.returncode:
                        print(mol, level, preset, scf, "FAILED", p.stderr[-500:], flush=True)
                        continue
                    r = json.loads(p.stdout)
                    rec = dict(mol=mol, level=level, preset=preset, scf=scf, model=a.model, eps=a.eps, energy=r["energy"], components=r.get("components"),
                               gradient=r.get("gradient"), gradient_parts=r.get("gradient_parts"), solvation=r.get("solvation"), iterations=r["iterations"],
                               timings=r.get("timings"), wall=wall, time=time.strftime("%Y-%m-%dT%H:%M:%S"))
                    with open(a.out, "a") as f:
                        f.write(json.dumps(rec) + "\n")
                    print(mol, level, preset, scf, f"E {r['energy']:.10f}", f"E_solv {r['solvation']['electrostatic']:.10f}", "iters", r["iterations"], f"wall {wall:.1f}", flush=True)


def table(a):
    import numpy as np
    recs = {}
    for ln in open(a.file):
        r = json.loads(ln)
        recs[(r["mol"], r["level"], r["preset"], r.get("model", "cpcm"), r["scf"])] = r
    print("| molecule | level | preset | mode | points | dielectric term, exact (kcal/mol) | d(E) vs exact (kcal/mol) | d(dielectric term) (kcal/mol) | max abs dG (Eh/bohr) | rms dG (Eh/bohr) | max abs G (Eh/bohr) | SCF iterations (exact) |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for (mol, level, preset, model, scf), r in recs.items():
        if scf == "exact" or (mol, level, preset, model, "exact") not in recs:
            continue
        x = recs[(mol, level, preset, model, "exact")]
        de = (r["energy"] - x["energy"]) * HARTREE_KCAL
        ds = (r["solvation"]["electrostatic"] - x["solvation"]["electrostatic"]) * HARTREE_KCAL
        if r.get("gradient") and x.get("gradient"):
            g, gx = np.array(r["gradient"]), np.array(x["gradient"])
            dg = g - gx
            gcol = f"{np.abs(dg).max():.1e} | {np.sqrt((dg ** 2).mean()):.1e} | {np.abs(gx).max():.1e}"
        else:
            gcol = "- | - | -"
        print(f"| {mol} | {level} | {preset} | {scf} | {x['solvation']['npoints']} | {x['solvation']['electrostatic'] * HARTREE_KCAL:.3f} | {de:+.2e} | {ds:+.2e} | {gcol} | {r['iterations']} ({x['iterations']}) |")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--mols", default="water")
    r.add_argument("--levels", default="b3lyp-svp")
    r.add_argument("--presets", default="pyscf")
    r.add_argument("--model", default="cpcm")
    r.add_argument("--modes", default="exact,predicted,df")
    r.add_argument("--eps", type=float, default=7.4257)
    r.add_argument("--calctype", default="gradient")
    r.add_argument("--conv", default="tight")
    r.add_argument("--out", required=True)
    r.add_argument("--threads", type=int, default=8)
    r.add_argument("--mem", type=int, default=40)
    r.add_argument("--timeout", type=int, default=7200)
    r.add_argument("--bin", default=str(ROOT / "target/release/linus"))
    t = sub.add_parser("table")
    t.add_argument("file")
    a = ap.parse_args()
    {"run": run, "table": table}[a.cmd](a)


if __name__ == "__main__":
    main()
