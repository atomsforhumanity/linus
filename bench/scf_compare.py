"""SCF iteration counts and per-phase times: linus vs ORCA 6.0.0 vs PySCF at matched criteria.

usage: python bench/scf_compare.py --tag TAG --cases la18:b3lyp:def2-svp:df,la18:hf:def2-svp:exact \
           [--programs linus,orca,pyscf] [--slice bench|none] [--threads 8] [--reps 1] [--linus-conv tight]

Case = mol:method:basis:jk with jk in exact (four-center), df (RI-J/K; RI-J only for pure
functionals) or cosx (RIJCOSX). Matched settings:
  linus  jk exact|df (def2-universal-jkfit)|rij-snk (def2-universal-jfit, COSX DefGrid2), grid orca:2,
         convergence preset --linus-conv (default tight: |dE| < 1e-8 and max|F_ai| < 5e-7)
  ORCA   TightSCF; NoRI NoCOSX | RIJK def2/JK (RI def2/J for pure functionals) | RIJCOSX def2/J;
         DefGrid2; B3LYP/G (= linus/PySCF b3lyp, VWN-RPA); ORCA stops on its energy criteria
         (ConvCheckMode 2: dE < 1e-8, dE(1-el) < 1e-5); its final orbital gradient is recorded
  PySCF  direct or density_fit(def2-universal-jkfit), grids level 3, minao guess; run to
         conv_tol 1e-10 / conv_tol_grad 1e-7 with a per-cycle trace, from which the Fock builds
         and time to reach (a) linus's criteria (same formulas) and (b) PySCF's defaults at
         conv_tol 1e-8 (norm_gorb < 1e-4) are taken. Not timed for COSX (PySCF has no COSX).
Each timed run holds the shared bench lock and runs under `slice bench` (CCD0) in a
memory-capped scope; `slice list` is recorded with every run. Results: bench/results/TAG/*.json.
"""
import argparse, fcntl, json, os, pathlib, re, subprocess, tempfile, time

ROOT = pathlib.Path(__file__).resolve().parents[1]
from bench_paths import LOCK, MAIN, ORCA, REFENV  # noqa: E402,F401  (env overrides: bench/bench_paths.py)
SLICE = str(pathlib.Path.home() / ".local/bin/slice")
MOLS = {p.stem: p for d in ("bench/molecules", "tests/molecules") for p in (ROOT / d).glob("*.xyz")}
CHARGES = {"u1an39": -1}
PURE = {"pbe", "blyp", "bp86", "pbe-d3", "tpss", "r2scan"}


def slice_list():
    return subprocess.run([SLICE, "list"], capture_output=True, text=True).stdout.strip().splitlines()


def wrap(cmd, slc, mem):
    pre = ["systemd-run", "--user", "--scope", "-q", "-p", f"MemoryMax={mem}G", "-p", "MemorySwapMax=0"]
    if slc != "none":
        pre += [SLICE, slc]
    return pre + cmd


def timed(cmd, **kw):
    """Run under the shared bench lock (flock(2), as flock(1) takes it); the wall time starts
    once the lock is held, so waiting for other benchmarks is not counted."""
    with open(LOCK, "a") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        t = time.perf_counter()
        p = subprocess.run(cmd, capture_output=True, text=True, **kw)
        return p, time.perf_counter() - t


def run_linus(mol, method, basis, jk, a):
    req = {"structure": {"xyz": MOLS[mol].read_text(), "charge": CHARGES.get(mol, 0)}, "method": method, "basis": basis,
           "threads": a.threads, "grid": {"xc": "orca:2"}, "scf": {"convergence": a.linus_conv}}
    req["jk"] = {"exact": "exact", "df": "df", "cosx": "rij-snk"}[jk]
    p, wall = timed(wrap([str(ROOT / "target/release/linus"), "--json"], a.slice, a.mem), input=json.dumps(req))
    if p.returncode != 0:
        return dict(ok=False, wall=wall, error=p.stderr[-2000:])
    d = json.loads(p.stdout)
    h = d["scf"]["history"]
    tm = d["timings"]
    return dict(ok=True, wall=wall, energy=d["energy"], builds=d["scf"]["fock_builds"], incremental=d["scf"]["incremental_builds"],
                final_gmax=h[-1]["gmax"], final_gnorm_pyscf=h[-1]["gnorm_pyscf"], nbf=d["nbf"],
                phases=dict(scf=tm.get("scf_total"), jk=tm.get("jk"), xc=tm.get("xc", 0.0), diag=tm.get("diag"), diis=tm.get("diis"),
                            setup=tm.get("jk_setup", 0.0) + tm.get("grid", 0.0) + tm.get("guess", 0.0) + tm.get("one_electron", 0.0),
                            k_final=tm.get("k_final", 0.0)),
                timings=tm, history=h)


def run_linus_base(mol, method, basis, jk, a):
    """An older linus build (--base-bin; before convergence presets and SCF history) at its
    defaults (e_tol 1e-10, max |F_ai| < 1e-8) and at the tight thresholds (`base-tight`)."""
    req = {"structure": {"xyz": MOLS[mol].read_text(), "charge": CHARGES.get(mol, 0)}, "method": method, "basis": basis,
           "threads": a.threads, "grid": {"xc": "orca:2"}, "jk": {"exact": "exact", "df": "df", "cosx": "rij-snk"}[jk]}
    if a._base_tight:
        req["scf"] = {"e_tol": 1e-8, "g_tol": 5e-7}
    p, wall = timed(wrap([a.base_bin, "--json"], a.slice, a.mem), input=json.dumps(req))
    if p.returncode != 0:
        return dict(ok=False, wall=wall, error=p.stderr[-2000:])
    d = json.loads(p.stdout)
    tm = d["timings"]
    return dict(ok=True, wall=wall, energy=d["energy"], builds=d["iterations"], nbf=d["nbf"],
                phases=dict(scf=tm.get("scf_total"), jk=tm.get("jk"), xc=tm.get("xc", 0.0), diag=tm.get("diag"), diis=tm.get("diis")), timings=tm)


def orca_input(mol, method, basis, jk, threads):
    m = method.lower()
    fn = "B3LYP/G" if m == "b3lyp" else ("HF" if m == "hf" else method.upper())
    approx = {"exact": "NoRI NoCOSX", "df": ("RI def2/J" if m in PURE else "RIJK def2/JK"), "cosx": "RIJCOSX def2/J"}[jk]
    lines = [f"! {fn} {basis} {approx} DefGrid2 TightSCF", f"%pal nprocs {threads} end", "%maxcore 3000",
             f"* xyz {CHARGES.get(mol, 0)} 1"]
    xyz = MOLS[mol].read_text().splitlines()
    n = int(xyz[0])
    lines += [l.strip() for l in xyz[2:2 + n]] + ["*", ""]
    return "\n".join(lines)


def parse_orca(out):
    r = {}
    m = re.search(r"SCF CONVERGED AFTER\s+(\d+)\s+CYCLES", out)
    r["cycles"] = int(m.group(1)) if m else None
    e = re.findall(r"FINAL SINGLE POINT ENERGY\s+(-?\d+\.\d+)", out)
    r["energy"] = float(e[-1]) if e else None
    for key, pat in [("last_de", r"Last Energy change\s+\.\.\.\s+(\S+)"), ("last_orbital_gradient", r"Last Orbital Gradient\s+\.\.\.\s+(\S+)"),
                     ("last_diis_error", r"Last DIIS Error\s+\.\.\.\s+(\S+)"), ("last_max_density_change", r"Last MAX-Density change\s+\.\.\.\s+(\S+)")]:
        m = re.search(pat, out)
        r[key] = float(m.group(1)) if m else None
    # TIMINGS block of the SCF
    blk = out.split("TIMINGS\n-------")
    phases = {}
    if len(blk) > 1:
        for line in blk[1].splitlines()[:40]:
            m = re.match(r"\s*(.+?)\s+\.\.\.\.\s+(\d+\.\d+) sec", line)
            if m:
                phases[m.group(1).strip()] = float(m.group(2))
    r["phases"] = phases
    m = re.search(r"TOTAL RUN TIME:\s+(\d+) days (\d+) hours (\d+) minutes (\d+) seconds (\d+) msec", out)
    if m:
        d, h, mi, sec, ms = map(int, m.groups())
        r["total_run_time"] = ((d * 24 + h) * 60 + mi) * 60 + sec + ms / 1000
    return r


def run_orca(mol, method, basis, jk, a):
    with tempfile.TemporaryDirectory(dir=ROOT / "bench/scratch") as td:
        wd = pathlib.Path(td)
        (wd / "orca.inp").write_text(orca_input(mol, method, basis, jk, a.threads))
        env = dict(os.environ, PATH=f"{ORCA}:{REFENV}/bin:" + os.environ["PATH"], LD_LIBRARY_PATH=f"{ORCA}/lib:{REFENV}/lib",
                   OMPI_MCA_btl="self,vader", OMPI_MCA_btl_vader_single_copy_mechanism="none", OMPI_MCA_hwloc_base_binding_policy="none")
        p, wall = timed(wrap([str(ORCA / "orca"), "orca.inp"], a.slice, a.mem), cwd=wd, env=env)
        out = p.stdout
        ok = "ORCA TERMINATED NORMALLY" in out
        r = parse_orca(out)
        r.update(ok=ok, wall=wall, input=(wd / "orca.inp").read_text().split("* xyz")[0])
        if not ok:
            r["error"] = out[-2000:]
        return r


def run_pyscf(mol, method, basis, jk, a):
    cmd = [str(REFENV / "bin/python"), str(ROOT / "bench/scf_trace_pyscf.py"), str(MOLS[mol]), method, basis, "--charge", str(CHARGES.get(mol, 0)),
           "--jk", "df" if jk == "df" else "exact", "--grid", "3", "--conv-tol", "1e-10", "--conv-tol-grad", "1e-7", "--threads", str(a.threads),
           "--max-memory", "8000"]
    env = dict(os.environ, OMP_NUM_THREADS=str(a.threads))
    p, wall = timed(wrap(cmd, a.slice, a.mem), env=env)
    if p.returncode != 0:
        return dict(ok=False, wall=wall, error=p.stderr[-2000:])
    d = json.loads(p.stdout)
    tr = d["trace"]

    def first(pred):
        acc = d["setup_time"]
        for r in tr:
            acc += r["time"]
            if pred(r):
                return r["cycle"] + 1, acc, r
        return None, None, None
    e_c, g_c = (1e-8, 5e-7) if a.linus_conv == "tight" else (1e-9, 1e-8)
    b1, t1, r1 = first(lambda r: abs(r["de"]) < e_c and r["gmax"] < g_c)
    b2, t2, r2 = first(lambda r: abs(r["de"]) < 1e-8 and r["norm_gorb"] < 1e-4)
    return dict(ok=True, wall=wall, energy=d["energy"], nbf=d["nbf"], fock_builds_total=d["fock_builds"], phases=d["phase_times"],
                scf_time=d["scf_time"], setup_time=d["setup_time"],
                matched=dict(builds=b1, time=t1, gmax=r1 and r1["gmax"]), pyscf_default=dict(builds=b2, time=t2, gmax=r2 and r2["gmax"]),
                trace=tr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--cases", required=True)
    ap.add_argument("--programs", default="linus,orca,pyscf")
    ap.add_argument("--slice", default="bench")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--mem", type=int, default=20)
    ap.add_argument("--linus-conv", default="tight")
    ap.add_argument("--base-bin", default=None, help="older linus binary for the programs base and base-tight")
    a = ap.parse_args()
    out = ROOT / "bench/results" / a.tag
    out.mkdir(parents=True, exist_ok=True)
    (ROOT / "bench/scratch").mkdir(exist_ok=True)
    for case in a.cases.split(","):
        mol, method, basis, jk = case.split(":")
        res = dict(case=case, mol=mol, method=method, basis=basis, jk=jk, threads=a.threads, slice=a.slice, linus_conv=a.linus_conv, runs={})
        fn = out / f"{mol}_{method}_{basis}_{jk}.json"
        if fn.exists():  # add programs to an existing result file
            res["runs"] = json.loads(fn.read_text())["runs"]
        for prog in a.programs.split(","):
            if prog == "pyscf" and jk == "cosx":
                continue
            runs = []
            for _ in range(a.reps):
                sl = slice_list()
                a._base_tight = prog == "base-tight"
                r = {"linus": run_linus, "orca": run_orca, "pyscf": run_pyscf, "base": run_linus_base,
                     "base-tight": run_linus_base}[prog](mol, method, basis, jk, a)
                r["slice_list"] = sl
                r["time"] = time.strftime("%Y-%m-%dT%H:%M:%S")
                runs.append(r)
                if prog == "linus" and r["ok"]:
                    print(f"{case:28s} linus  builds {r['builds']:3d} (incr {r['incremental']}) wall {r['wall']:7.2f} scf {r['phases']['scf']:7.2f} "
                          f"jk {r['phases']['jk']:6.2f} xc {r['phases']['xc']:6.2f} final max|F_ai| {r['final_gmax']:.1e}", flush=True)
                elif prog == "orca" and r["ok"]:
                    ph = r["phases"]
                    print(f"{case:28s} orca   cycles {r['cycles']} wall {r['wall']:7.2f} scf {ph.get('Total time', 0):7.2f} "
                          f"fock {ph.get('Fock matrix formation', 0):6.2f} last orb grad {r['last_orbital_gradient']}", flush=True)
                elif prog.startswith("base") and r["ok"]:
                    print(f"{case:28s} {prog:10s} builds {r['builds']:3d} wall {r['wall']:7.2f} scf {r['phases']['scf']:7.2f} "
                          f"jk {r['phases']['jk']:6.2f} xc {r['phases']['xc']:6.2f}", flush=True)
                elif prog == "pyscf" and r["ok"]:
                    print(f"{case:28s} pyscf  matched builds {r['matched']['builds']} t {r['matched']['time']}  default builds "
                          f"{r['pyscf_default']['builds']} t {r['pyscf_default']['time']}", flush=True)
                else:
                    print(f"{case:28s} {prog} FAILED: {r.get('error', '')[-500:]}", flush=True)
            res["runs"][prog] = runs
        (out / f"{mol}_{method}_{basis}_{jk}.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
