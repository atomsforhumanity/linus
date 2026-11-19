"""linus timings with and without the C-PCM implicit solvent (energy + gradient), for the PCM
overhead and the comparison with ORCA 6.0.0's `CPCM(THF)` runs of bench/pcm_orca.py.

Each run is timed inside the shared bench lock (flock, held per molecule and level, or per run with
--lock-per run) on the chosen slice (default
`slice bench`: CCD0's 8 physical cores); `slice list` and the load average are recorded with every
run. Levels (RIJ+COSX unless the level says rijk):
  wb97x-tzvp  wB97X-D3BJ/def2-TZVP, jk rij-snk (def2-universal-jfit, cosx:2), grid orca:2
  b3lyp-svp   B3LYP/def2-SVP, same J/K and grid
  wb97x-tzvp-rijk, b3lyp-svp-rijk: RI-JK (def2-universal-jkfit)
Solvent: C-PCM, eps 7.4257 (THF) unless --eps; --preset pyscf|orca. --variants picks the runs held
together in one lock: gas, pcm (solvent.scf_terms unset: linus's default), pcm-exact, pcm-predicted,
pcm-ri (solvent.scf_terms = exact | predicted, or solvent.df = true for ri); default gas,pcm. --calctype hessian times
the analytic Hessian and harmonic analysis (RI-JK levels; Hessians need exact or RI-JK J/K) for
the comparison with ORCA's analytic CPCM `Freq`.

  .refenv/bin/python bench/pcm_bench.py --out bench/results/2026-11-07-pcm/linus_bench.jsonl \
      --mols la18,u1an39,ts61 --levels b3lyp-svp,wb97x-tzvp [--reps 2] [--slice bench] \
      [--lock .../bench.lock] [--preset pyscf] [--timeout 5400]
"""
import argparse, contextlib, fcntl, json, os, pathlib, subprocess, time

ROOT = pathlib.Path(__file__).resolve().parents[1]
BOHR = 0.529177210544
CHARGES = {"la18": 0, "u1an39": -1, "ts61": 0, "ts113": 0}
LEVELS = {
    "wb97x-tzvp": dict(method="wb97x-d3bj", basis="def2-tzvp", jk=dict(kind="rij-snk", aux="def2-universal-jfit", k_grid="cosx:2"), grid=dict(xc="orca:2")),
    "b3lyp-svp": dict(method="b3lyp", basis="def2-svp", jk=dict(kind="rij-snk", aux="def2-universal-jfit", k_grid="cosx:2"), grid=dict(xc="orca:2")),
    "wb97x-tzvp-rijk": dict(method="wb97x-d3bj", basis="def2-tzvp", jk="df", aux="def2-universal-jkfit", grid="orca:2"),
    "b3lyp-svp-rijk": dict(method="b3lyp", basis="def2-svp", jk="df", aux="def2-universal-jkfit", grid="orca:2"),
}


@contextlib.contextmanager
def held(lock):
    if not lock:
        yield
        return
    with open(lock, "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--mols", default="la18")
    ap.add_argument("--levels", default="b3lyp-svp")
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--slice", default="bench")
    ap.add_argument("--lock", default="")
    ap.add_argument("--lock-per", default="pair", choices=["pair", "run"])
    ap.add_argument("--eps", type=float, default=7.4257)
    ap.add_argument("--preset", default="pyscf")
    ap.add_argument("--calctype", default="gradient", choices=["energy", "gradient", "hessian"])
    ap.add_argument("--solvent-only", action="store_true")
    ap.add_argument("--variants", default="", help="comma list of gas, pcm, pcm-exact, pcm-predicted, pcm-ri (default gas,pcm; pcm with --solvent-only)")
    ap.add_argument("--mem", type=int, default=40)
    ap.add_argument("--timeout", type=int, default=5400)
    ap.add_argument("--bin", default=str(ROOT / "target/release/linus"))
    a = ap.parse_args()
    for mol in a.mols.split(","):
        lines = (ROOT / "bench/molecules" / f"{mol}.xyz").read_text().splitlines()
        n = int(lines[0])
        syms = [ln.split()[0] for ln in lines[2:2 + n]]
        pos = [[float(v) / BOHR for v in ln.split()[1:4]] for ln in lines[2:2 + n]]
        for level in a.levels.split(","):
            # the lock is held per (molecule, level): gas and solvent runs and all repetitions
            # together (same conditions for the pair; shorter waits than per run), or per run
            with held(a.lock if a.lock_per == "pair" else ""):
                variants = a.variants.split(",") if a.variants else (["pcm"] if a.solvent_only else ["gas", "pcm"])
                for var in variants:
                    solv = var != "gas"
                    req = {"structure": {"symbols": syms, "geometry": pos, "charge": CHARGES.get(mol, 0)}, "calctype": a.calctype,
                           "threads": a.threads, **LEVELS[level]}
                    if solv:
                        req["solvent"] = {"eps": a.eps, "model": "cpcm", "preset": a.preset}
                        if var.startswith("pcm-"):
                            req["solvent"].update({"df": True} if var[4:] in ("ri", "df") else {"scf_terms": var[4:]})
                    runs = []
                    for rep in range(a.reps):
                        cmd = ["systemd-run", "--user", "--scope", "-q", "-p", f"MemoryMax={a.mem}G", "-p", "MemorySwapMax=0"]
                        if a.slice != "none":
                            cmd += [str(pathlib.Path.home() / ".local/bin/slice"), a.slice]
                        cmd += ["timeout", str(a.timeout), a.bin, "--json"]
                        with held(a.lock if a.lock_per == "run" else ""):
                            slist = subprocess.run([str(pathlib.Path.home() / ".local/bin/slice"), "list"], capture_output=True, text=True).stdout
                            load = os.getloadavg()
                            t0 = time.perf_counter()
                            p = subprocess.run(cmd, input=json.dumps(req), capture_output=True, text=True)
                            wall = time.perf_counter() - t0
                        if p.returncode:
                            runs.append(dict(ok=False, wall=wall, error=p.stderr[-1500:], slice_list=slist, load_average=load))
                            print(mol, level, var, "FAILED", p.stderr[-300:], flush=True)
                            break
                        r = json.loads(p.stdout)
                        t = r["timings"]
                        runs.append(dict(ok=True, wall=wall, energy=r["energy"], iterations=r["iterations"], nbf=r["nbf"], timings=t,
                                         solvation=r.get("solvation"), gradient=r.get("gradient"), slice_list=slist, load_average=load,
                                         frequencies=(r.get("vibrational_analysis") or {}).get("frequencies"), cphf_iterations=r.get("cphf_iterations")))
                        print(mol, level, var, f"wall {wall:.2f}", "pcm", round(sum(v for k, v in t.items() if k.startswith("pcm_")), 2),
                              "grad_solvent", round(t.get("grad_solvent", 0.0), 2), "iters", r["iterations"], flush=True)
                    rec = dict(mol=mol, level=level, variant=var, calctype=a.calctype, settings=LEVELS[level], solvent=req.get("solvent"), threads=a.threads, slice=a.slice,
                               runs=runs, best_wall=min((x["wall"] for x in runs if x["ok"]), default=None), time=time.strftime("%Y-%m-%dT%H:%M:%S"))
                    with open(a.out, "a") as f:
                        f.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
