"""wB97M-V (VV10) benchmark: linus (RI-JK, RIJ + COSX) vs ORCA 6.0.0 (RIJCOSX) at OMol25's settings,
and the functional comparison wB97X-D3BJ / wB97M-D3BJ / wB97M-V, energies and gradients.

usage: .refenv/bin/python bench/vv10_bench.py --tag TAG --cases la18:def2-tzvpd,u1an39:def2-tzvpd \
           --methods wb97m-v --calcs energy,gradient --programs linus-ri,linus-cosx,orca \
           [--defgrid 3] [--slice bench|ccd2|none] [--lock] [--reps 1]
       references (not timed): --programs linus-exact --ref --grid pyscf:7 --nlc-grid pyscf:3

Settings (OMol25: ORCA 6.0.0 wB97M-V/def2-TZVPD, RIJCOSX def2/J, DEFGRID3, TightSCF):
  linus-ri    jk df, def2-universal-jkfit (= ORCA def2/JK), XC grid orca:N (ORCA's DefGrid N),
              VV10 on ORCA's NL grid of DefGrid N (DefGrid3: orca:1), 8 threads
  linus-cosx  jk rij-snk, def2-universal-jfit (= def2/J), COSX grids of DefGrid N (cosx:N)
  linus-exact jk exact (four-center); --grid / --nlc-grid set the grids (references)
  orca        ! <method> <basis> def2/J RIJCOSX DefGridN TightSCF [EnGrad] Bohrs, 8 MPI ranks
  orca-exact  ! <method> <basis> NoRI NoCOSX DefGridN TightSCF [EnGrad] Bohrs (attribution of differences)
              (energy-only wB97M-V runs evaluate VV10 after the SCF, ORCA's default; with EnGrad
              ORCA switches to self-consistent VV10, as OMol25's force runs did)
linus uses its per-purpose SCF defaults. Coordinates go to both programs in bohr. Gradients are
also stored without dispersion (D3BJ methods).

With --lock each run holds the shared bench lock (wall time starts once it is held).
Results: bench/results/TAG/<mol>_<basis>_<method>_<calc>.json (one file per case, all programs).
"""
import argparse, fcntl, json, os, pathlib, re, subprocess, tempfile, time

ROOT = pathlib.Path(__file__).resolve().parents[1]
from bench_paths import LOCK, MAIN, ORCA, REFENV  # noqa: E402,F401  (env overrides: bench/bench_paths.py)
SLICE = str(pathlib.Path.home() / ".local/bin/slice")
BOHR = 0.529177210544
CHARGES = {"la18": 0, "u1an39": -1, "ts61": 0, "ts113": 0}
ORCA_NAME = {"wb97m-v": "wB97M-V", "wb97m-d3bj": "wB97M-D3BJ", "wb97x-d3bj": "wB97X-D3BJ", "wb97x-v": "wB97X-V", "b97m-v": "B97M-V"}


def read_xyz(mol):
    lines = (ROOT / "bench/molecules" / f"{mol}.xyz").read_text().splitlines()
    n = int(lines[0])
    return [(l.split()[0], [float(x) / BOHR for x in l.split()[1:4]]) for l in lines[2:2 + n]]


def slice_list():
    return subprocess.run([SLICE, "list"], capture_output=True, text=True).stdout.strip().splitlines()


def wrap(cmd, slc, mem):
    pre = ["systemd-run", "--user", "--scope", "-q", "-p", f"MemoryMax={mem}G", "-p", "MemorySwapMax=0", "-p", "OOMPolicy=continue"]
    if slc != "none":
        pre += [SLICE, slc] if not slc.startswith("cpus:") else ["taskset", "-c", slc[5:]]
    return pre + cmd


def timed(cmd, lock, **kw):
    """With `lock`, hold the shared bench lock; the wall time starts once it is held."""
    if not lock:
        t = time.perf_counter()
        p = subprocess.run(cmd, capture_output=True, text=True, **kw)
        return p, time.perf_counter() - t
    with open(LOCK, "a") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        t = time.perf_counter()
        p = subprocess.run(cmd, capture_output=True, text=True, **kw)
        return p, time.perf_counter() - t


def run_linus(prog, mol, basis, method, calc, a):
    atoms = read_xyz(mol)
    req = {"structure": {"symbols": [s for s, _ in atoms], "geometry": [c for _, c in atoms], "charge": CHARGES.get(mol, 0),
                         "multiplicity": 1},
           "calctype": calc, "method": method, "basis": basis, "grid": {"xc": a.grid or f"orca:{a.defgrid}"}, "threads": a.threads}
    if a.nlc_grid:
        req["grid"]["nlc"] = a.nlc_grid
    if prog == "linus-ri":
        req["jk"] = {"kind": "df", "aux": "def2-universal-jkfit"}
    elif prog == "linus-cosx":
        req["jk"] = {"kind": "rij-snk", "aux": "def2-universal-jfit", "k_grid": f"cosx:{a.defgrid}"}
    else:
        req["jk"] = "exact"
    if a.ref:
        req["scf"] = {"convergence": "verytight"}
    p, wall = timed(wrap([a.linus_bin, "--json"], a.slice, a.mem), a.lock, input=json.dumps(req))
    if p.returncode != 0:
        return dict(ok=False, wall=wall, error=(p.stderr or "")[-2000:] or f"exit {p.returncode} (killed?)")
    d = json.loads(p.stdout)
    r = dict(ok=True, wall=wall, energy=d["energy"], iterations=d["iterations"], nbf=d["nbf"], timings=d["timings"],
             e_disp=d["components"].get("dispersion"), components=d["components"], vv10=d.get("vv10"), grid=d.get("grid"))
    if "gradient" in d:
        r["gradient"] = d["gradient"]
        parts = d.get("gradient_parts") or {}
        if "dispersion" in parts:
            r["gradient_no_disp"] = [[g - x for g, x in zip(gr, dr)] for gr, dr in zip(d["gradient"], parts["dispersion"])]
    return r


def orca_input(mol, basis, method, calc, threads, defgrid, exact=False):
    approx = "NoRI NoCOSX" if exact else "def2/J RIJCOSX"
    kw = f"! {ORCA_NAME[method]} {basis} {approx} DefGrid{defgrid} TightSCF" + (" EnGrad" if calc == "gradient" else "") + " Bohrs"
    lines = [kw, f"%pal nprocs {threads} end", "%maxcore 3000", f"* xyz {CHARGES.get(mol, 0)} 1"]
    lines += [f"{s} {c[0]:.12f} {c[1]:.12f} {c[2]:.12f}" for s, c in read_xyz(mol)] + ["*", ""]
    return "\n".join(lines)


def parse_orca(out, wd, natom):
    r = {}
    e = re.findall(r"FINAL SINGLE POINT ENERGY\s+(-?\d+\.\d+)", out)
    r["energy"] = float(e[-1]) if e else None
    m = re.search(r"SCF CONVERGED AFTER\s+(\d+)\s+CYCLES", out)
    r["iterations"] = int(m.group(1)) if m else None
    m = re.search(r"Dispersion correction\s+(-?\d+\.\d+)", out)
    r["e_disp"] = float(m.group(1)) if m else None
    m = re.findall(r"NL Energy, E\(C,NL\)\s*:\s*(-?\d+\.\d+)", out)
    r["e_nl"] = float(m[-1]) if m else None
    r["scnl"] = "Switching to SCNL" in out
    for key, pat in [("t_gradient", r"Total SCF gradient time\s+\.+\s+(\d+\.\d+) sec"), ("t_scf_iterations", r"SCF iterations\s+\.\.\.\s+(\d+\.\d+) sec"),
                     ("t_nl_gradient", r"NL correlation gradient\s+\.+\s+(\d+\.\d+) sec"), ("t_xc_gradient", r"XC gradient\s+\.+\s+(\d+\.\d+) sec"),
                     ("t_cosx_gradient", r"COSX gradient\s+\.+\s+(\d+\.\d+) sec"), ("t_rij_gradient", r"RI-J Coulomb gradient\s+\.+\s+(\d+\.\d+) sec")]:
        m = re.search(pat, out)
        r[key] = float(m.group(1)) if m else None
    m = re.search(r"Total time needed\s+\.+\s+(\d+\.\d+) sec\n\s*\n\s*---+\nSCF ITERATIONS", out)
    eg = wd / "orca.engrad"
    if eg.exists():
        data = [l.strip() for l in eg.read_text().splitlines() if l.strip() and not l.startswith("#")]
        n = int(data[0])
        g = [float(x) for x in data[2:2 + 3 * n]]
        r["gradient"] = [g[3 * i:3 * i + 3] for i in range(n)]
        blk = out.split("DISPERSION GRADIENT")
        if len(blk) > 1:
            rows = re.findall(r"^\s*\d+\s+\w+\s*:\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)", blk[-1], re.M)[:natom]
            dg = [[float(x) for x in row] for row in rows]
            r["gradient_disp"] = dg
            r["gradient_no_disp"] = [[a - b for a, b in zip(ga, gb)] for ga, gb in zip(r["gradient"], dg)]
    return r


def run_orca(prog, mol, basis, method, calc, a):
    env = dict(os.environ, PATH=f"{ORCA}:{REFENV}/bin:" + os.environ["PATH"], LD_LIBRARY_PATH=f"{ORCA}/lib:{REFENV}/lib",
               OMPI_MCA_btl="self,vader", OMPI_MCA_btl_vader_single_copy_mechanism="none", OMPI_MCA_hwloc_base_binding_policy="none")
    (ROOT / "bench/scratch").mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=ROOT / "bench/scratch") as td:
        wd = pathlib.Path(td)
        (wd / "orca.inp").write_text(orca_input(mol, basis, method, calc, a.threads, a.defgrid, exact=prog == "orca-exact"))
        p, wall = timed(wrap([str(ORCA / "orca"), "orca.inp"], a.slice, a.mem), a.lock, cwd=wd, env=env)
        out = p.stdout
        ok = "ORCA TERMINATED NORMALLY" in out
        r = dict(ok=ok, wall=wall, **parse_orca(out, wd, len(read_xyz(mol))))
        if not ok:
            r["error"] = out[-2000:] + (p.stderr or "")[-1000:]
        return r


RUN = {"linus-ri": run_linus, "linus-cosx": run_linus, "linus-exact": run_linus, "orca": run_orca, "orca-exact": run_orca}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--cases", required=True, help="mol:basis,...")
    ap.add_argument("--methods", default="wb97m-v")
    ap.add_argument("--calcs", default="energy,gradient")
    ap.add_argument("--programs", default="linus-ri,linus-cosx,orca")
    ap.add_argument("--defgrid", type=int, default=3)
    ap.add_argument("--grid", default=None, help="linus XC grid (default orca:<defgrid>)")
    ap.add_argument("--nlc-grid", default=None, help="linus NLC grid (default auto: ORCA's NL grid of the DefGrid)")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--slice", default="bench")
    ap.add_argument("--lock", action="store_true", help="hold the shared bench lock (headline timings)")
    ap.add_argument("--mem", type=int, default=30, help="GB per process (systemd scope cap)")
    ap.add_argument("--ref", action="store_true", help="reference runs (verytight SCF)")
    ap.add_argument("--label", default="", help="suffix of the program key")
    ap.add_argument("--linus-bin", default=str(ROOT / "target/release/linus"), help="linus CLI (a fixed copy keeps rebuilds out of a series)")
    a = ap.parse_args()
    outdir = ROOT / "bench/results" / a.tag
    outdir.mkdir(parents=True, exist_ok=True)
    for case in a.cases.split(","):
        mol, basis = case.split(":")
        for method in a.methods.split(","):
            for calc in a.calcs.split(","):
                f = outdir / f"{mol}_{basis}_{method}_{calc}.json"
                rec = json.loads(f.read_text()) if f.exists() else dict(mol=mol, basis=basis, method=method, calc=calc, threads=a.threads, runs={})
                for prog in a.programs.split(","):
                    for rep in range(a.reps):
                        r = RUN[prog](prog, mol, basis, method, calc, a)
                        r.update(rep=rep, slice=a.slice, lock=a.lock, defgrid=a.defgrid, slice_list=slice_list(),
                                 load=open("/proc/loadavg").read().split()[:3], time=time.strftime("%Y-%m-%dT%H:%M:%S"))
                        key = prog + ("" if a.lock else f"@{a.slice}") + a.label
                        rec["runs"].setdefault(key, []).append(r)
                        print(f"{mol:7} {basis:10} {method:10} {calc:8} {key:24} rep {rep}: {'ok ' if r['ok'] else 'FAIL'} {r['wall']:8.2f} s  E={r.get('energy')}"
                              + ("" if r["ok"] else "  " + r.get("error", "")[-300:].replace("\n", " ")), flush=True)
                        f.write_text(json.dumps(rec, indent=1))


if __name__ == "__main__":
    main()
