"""ORCA 6.0.0 CPCM reference data for linus's implicit solvation (C-PCM with Gaussian surface charges).

Subcommands (each writes JSON to bench/results/2026-11-07-pcm/orca/, raw ORCA runs stay in --work):

  geoms   build the small-molecule geometries (water, NH3, CH3SH, NH4+, acetate), optimize them in the
          gas phase with ORCA (B3LYP/G def2-SVP TightOpt, exact J/K) and write bench/molecules/pcm/*.xyz (Angstrom)
  facts   model-level facts of ORCA's CPCM: run cavity variants (default, surfacetype vdw_gaussian,
          gepol_ses_gaussian, gepol_ses, gepol_sas, num_leb, fepstype, user radii, epsilon) on small molecules,
          parse <base>.cpcm / .cpcm_corr into JSON and test the reconstructed model (Lebedev order per sphere,
          York-Karplus switching, Gaussian exponents, A matrix, f(eps), dielectric energy, electronic potential)
  ref     energies and gradients (EnGrad) with and without CPCM: HF, B3LYP/G, wB97X-D3BJ; def2-SVP (+TZVP for
          small molecules); THF and water; exact J/K (NoRI) and RIJCOSX (def2/J); TightSCF. Model-level, unlocked.
  freq    L-lactide B3LYP/G def2-SVP RIJCOSX CPCM(THF) Freq: analytic or numerical? (bench lock, slice bench);
          frequencies, Hessian (.hess) and the timed Hessian steps; --solvents gas,thf; --reparse re-reads orca.out
  bench   timings: la18, u1an39, ts61, ts113; wB97X-D3BJ/def2-TZVP and B3LYP/G/def2-SVP RIJCOSX EnGrad, with and
          without CPCM(THF), 8 ranks on `slice bench`, bench lock taken per run
  fdcheck analytic CPCM gradient vs central differences of the energy (HF/def2-SVP, water, NH4+, acetate)
  summary compact tables ref/summary.json and bench/summary.json

Run the analysis subcommands (facts, fdcheck) under `slice` too: numpy/PySCF threads are capped at 4 below.

ORCA runs as in bench/grad_bench.py: full path, PATH/LD_LIBRARY_PATH with ORCA, its lib/ and .refenv (mpirun),
OMPI_MCA_hwloc_base_binding_policy=none, `! Bohrs` with bohr coordinates, each run in its own memory-capped
systemd scope (MemoryMax, MemorySwapMax=0, OOMPolicy=continue) on a CPU slice.

usage: .refenv/bin/python bench/pcm_orca.py geoms --slice ccd3
       ~/.local/bin/slice ccd3 .refenv/bin/python bench/pcm_orca.py facts --slice ccd3 --grad
       .refenv/bin/python bench/pcm_orca.py ref --slices ccd2,ccd3 --nprocs 4
       .refenv/bin/python bench/pcm_orca.py freq --lock $BENCH_LOCK
       .refenv/bin/python bench/pcm_orca.py bench --lock $BENCH_LOCK --mols la18,u1an39,ts61 --reps 2
"""
import argparse, concurrent.futures, contextlib, fcntl, json, math, os, pathlib, re, shutil, subprocess, threading, time, uuid

# the analysis (numpy/PySCF) must not spread over the whole node: few BLAS/OpenMP threads (run this script under `slice`)
for _v in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ.setdefault(_v, "4")

ROOT = pathlib.Path(__file__).resolve().parents[1]
MOLS = ROOT / "bench/molecules"
PCM_MOLS = MOLS / "pcm"
OUT = ROOT / "bench/results/2026-11-07-pcm/orca"
from bench_paths import ORCA, REFENV, SCRATCH  # noqa: E402  (env overrides: bench/bench_paths.py)
SLICE = pathlib.Path.home() / ".local/bin/slice"
WORK = SCRATCH / "orca-pcm"
BOHR = 0.529177210544  # Angstrom per bohr (CODATA 2022)

# charge, multiplicity, geometry file (bench/molecules or bench/molecules/pcm)
SYSTEMS = {
    "water": (0, 1, "pcm/water"), "nh3": (0, 1, "pcm/nh3"), "ch3sh": (0, 1, "pcm/ch3sh"), "nh4": (1, 1, "pcm/nh4"),
    "acetate": (-1, 1, "pcm/acetate"), "ch3sh+": (1, 2, "pcm/ch3sh"), "la18": (0, 1, "la18"), "u1an39": (-1, 1, "u1an39"),
    "ts61": (0, 1, "ts61"), "ts113": (0, 1, "ts113"),
}
METHODS = {"hf": ("HF", "UHF"), "b3lyp": ("B3LYP/G", "UKS B3LYP/G"), "wb97x-d3bj": ("wB97X-D3BJ", "UKS wB97X-D3BJ")}
BASES = {"svp": "def2-SVP", "tzvp": "def2-TZVP"}
JK = {"exact": "NoRI DefGrid2", "rijcosx": "def2/J RIJCOSX DefGrid2"}
SOLVENTS = {"thf": "CPCM(THF)", "water": "CPCM(Water)", "gas": ""}

# Starting geometries (Angstrom) for `geoms`; ORCA optimizes them in the gas phase.
GUESS = {
    "water": [("O", 0.0, 0.0, 0.1173), ("H", 0.0, 0.7572, -0.4692), ("H", 0.0, -0.7572, -0.4692)],
    "nh3": [("N", 0.0, 0.0, 0.1162), ("H", 0.0, 0.9377, -0.2711), ("H", 0.8121, -0.4689, -0.2711), ("H", -0.8121, -0.4689, -0.2711)],
    "ch3sh": [("C", -0.0475, 1.1489, 0.0), ("S", -0.0475, -0.6699, 0.0), ("H", 1.2789, -0.9254, 0.0), ("H", -1.0866, 1.4797, 0.0),
              ("H", 0.4371, 1.5462, 0.8924), ("H", 0.4371, 1.5462, -0.8924)],
    "nh4": [("N", 0.0, 0.0, 0.0), ("H", 0.59, 0.59, 0.59), ("H", -0.59, -0.59, 0.59), ("H", -0.59, 0.59, -0.59), ("H", 0.59, -0.59, -0.59)],
    "acetate": [("C", 0.0, 0.0, 0.0), ("O", 1.1325, 0.0, -0.5524), ("O", -1.1325, 0.0, -0.5524), ("C", 0.0, 0.0, 1.555),
                ("H", 0.0, 1.029, 1.9295), ("H", -0.8911, -0.5145, 1.9295), ("H", 0.8911, -0.5145, 1.9295)],
}


# ----------------------------------------------------------------------------------------------- running
def read_xyz(path):
    lines = pathlib.Path(path).read_text().splitlines()
    n = int(lines[0])
    return [(l.split()[0], [float(x) for x in l.split()[1:4]]) for l in lines[2:2 + n]]


def system(name):
    charge, mult, geo = SYSTEMS[name]
    return read_xyz(MOLS / f"{geo}.xyz"), charge, mult


def orca_input(keywords, atoms, charge, mult, nprocs, maxcore, blocks=""):
    lines = [f"! {keywords} Bohrs", f"%pal nprocs {nprocs} end", f"%maxcore {maxcore}"]
    if blocks:
        lines.append(blocks)
    lines.append(f"* xyz {charge} {mult}")
    lines += [f"{a:2s} {c[0] / BOHR:20.12f} {c[1] / BOHR:20.12f} {c[2] / BOHR:20.12f}" for a, c in atoms]
    return "\n".join(lines + ["*", ""])


@contextlib.contextmanager
def held(lock):
    """The shared bench lock (flock), taken per timed run (as in grad_bench.py)."""
    if not lock:
        yield
        return
    with open(lock, "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def run_orca(wd, inp, slc, mem_gb, lock=None, timeout=None, base="orca"):
    """Write <wd>/<base>.inp, run ORCA in a memory-capped scope on a CPU slice; returns dict with output text etc."""
    wd = pathlib.Path(wd)
    wd.mkdir(parents=True, exist_ok=True)
    (wd / f"{base}.inp").write_text(inp)
    env = dict(os.environ, PATH=f"{ORCA}:{REFENV}/bin:" + os.environ["PATH"], LD_LIBRARY_PATH=f"{ORCA}/lib:{REFENV}/lib",
               OMPI_MCA_btl="self,vader", OMPI_MCA_btl_vader_single_copy_mechanism="none", OMPI_MCA_hwloc_base_binding_policy="none")
    unit = f"pcm-orca-{uuid.uuid4().hex[:10]}"
    cmd = ["systemd-run", "--user", "--scope", "-q", f"--unit={unit}", "-p", f"MemoryMax={mem_gb}G", "-p", "MemorySwapMax=0",
           "-p", "OOMPolicy=continue"]
    if slc and slc != "none":
        cmd += [str(SLICE), slc]
    cmd += [str(ORCA / "orca"), f"{base}.inp"]
    with held(lock):
        slist = subprocess.run([str(SLICE), "list"], capture_output=True, text=True).stdout
        load = pathlib.Path("/proc/loadavg").read_text().split()[:3]
        t0 = time.perf_counter()
        timed_out = False
        with open(wd / f"{base}.out", "w") as fout:
            p = subprocess.Popen(cmd, cwd=wd, env=env, stdout=fout, stderr=subprocess.STDOUT)
            try:
                rc = p.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
                subprocess.run(["systemctl", "--user", "stop", f"{unit}.scope"], capture_output=True)
                rc = p.wait()
        wall = time.perf_counter() - t0
        load_after = pathlib.Path("/proc/loadavg").read_text().split()[:3]
    out = (wd / f"{base}.out").read_text(errors="replace")
    return dict(wall=wall, rc=rc, timed_out=timed_out, out=out, slice_list=slist, loadavg_before=load, loadavg_after=load_after,
                slice=slc, wd=str(wd))


# ----------------------------------------------------------------------------------------------- parsing
def _f(pat, text, last=True, group=1, conv=float):
    m = re.findall(pat, text, flags=re.M)
    if not m:
        return None
    v = m[-1] if last else m[0]
    if isinstance(v, tuple):
        v = v[group - 1]
    return conv(v)


def parse_out(text):
    """Numbers from an ORCA output (energies, CPCM block, timings). Small extracted numbers only."""
    r = dict(ok="ORCA TERMINATED NORMALLY" in text)
    r["version"] = _f(r"Program Version\s+(\S+)", text, conv=str)
    r["final_energy"] = _f(r"FINAL SINGLE POINT ENERGY\s+(-?\d+\.\d+)", text)
    comp = {}
    for key, label in [("total", "Total Energy"), ("nuclear_repulsion", "Nuclear Repulsion"), ("electronic", "Electronic Energy"),
                       ("one_electron", "One Electron Energy"), ("two_electron", "Two Electron Energy"),
                       ("cpcm_dielectric", "CPCM Dielectric")]:
        comp[key] = _f(rf"^{label}\s*:\s+(-?\d+\.\d+) Eh", text)
    comp["exchange_change_gridx3"] = _f(r"Exchange energy change after final integration\s*:\s+(-?\d+\.\d+)", text)
    comp["dispersion"] = _f(r"^Dispersion correction\s+(-?\d+\.\d+)", text)
    r["scf_components"] = comp
    cp = {}
    cp["surface_charge"] = _f(r"^Surface-charge\s*:\s+(-?\d+\.\d+)", text)
    cp["corrected_charge"] = _f(r"^Corrected charge\s*:\s+(-?\d+\.\d+)", text)
    cp["outlying_charge_correction"] = _f(r"^Outlying charge corr\.\s*:\s+(-?\d+\.\d+) Eh", text)
    cp["epsilon"] = _f(r"^\s+Epsilon\s+\.\.\.\s+(\S+)", text, conv=str)
    cp["refrac"] = _f(r"^\s+Refrac\s+\.\.\.\s+(\S+)", text)
    cp["rsolv"] = _f(r"^\s+Rsolv\s+\.\.\.\s+(\S+)", text)
    cp["feps_x"] = _f(r"FEps X parameter\s+\.\.\.\s+(\S+)", text)
    cp["surface_type"] = _f(r"Surface type\s+\.\.\.\s+(.+?)\s*$", text, conv=str)
    cp["discretization"] = _f(r"Discretization scheme\s+\.\.\.\s+(.+?)\s*$", text, conv=str)
    cp["threshold_h"] = _f(r"Threshold for H atoms\s+\.\.\.\s+(\S+)", text)
    cp["threshold_non_h"] = _f(r"Threshold for non-H atoms\s+\.\.\.\s+(\S+)", text)
    cp["epsilon_function"] = _f(r"Epsilon function type\s+\.\.\.\s+(.+?)\s*$", text, conv=str)
    cp["solvent"] = _f(r"^Solvent:\s+\.\.\.\s+(.+?)\s*$", text, conv=str)
    cp["radii_bohr"] = {s: float(b) for s, b, a in re.findall(r"Radius for (\w+)\s+used is\s+(\S+) Bohr \(=\s*(\S+) Ang", text)}
    cp["radii_changed"] = re.findall(r"Radius for (\w+) changed to\s+(\S+) Bohr", text)
    cp["npoints"] = _f(r"Cavity surface points\s+\.\.\.\s+(\d+)", text, conv=int)
    cp["volume_bohr3"] = _f(r"Cavity Volume\s+\.\.\.\s+(\S+)", text)
    cp["area_bohr2"] = _f(r"Cavity Surface-area\s+\.\.\.\s+(\S+)", text)
    cp["init_time"] = _f(r"Overall time for CPCM initialization\s+\.\.\.\s+([\d.]+)s", text)
    if cp["epsilon"] is not None:
        try:
            cp["epsilon"] = float(cp["epsilon"])
        except ValueError:
            pass
    r["cpcm"] = cp if cp["npoints"] is not None else None
    r["scf_iterations"] = _f(r"SCF CONVERGED AFTER\s+(\d+)", text, conv=int)
    t = {}
    for key, pat in [("sum_modules", r"Sum of individual times\s+\.\.\.\s+([\d.]+) sec"), ("startup", r"Startup calculation\s+\.\.\.\s+([\d.]+) sec"),
                     ("scf", r"SCF iterations\s+\.\.\.\s+([\d.]+) sec"), ("gradient", r"SCF Gradient evaluation\s+\.\.\.\s+([\d.]+) sec"),
                     ("properties", r"Property calculations\s+\.\.\.\s+([\d.]+) sec"),
                     ("hessian", r"Analytical frequency calculation\s*\.\.\.\s+([\d.]+) sec"),
                     ("scf_response", r"SCF Response\s+\.\.\.\s+([\d.]+) sec"),
                     ("numfreq", r"Numerical frequency calculation\s+\.\.\.\s+([\d.]+) sec"),
                     ("fock_cpcm_terms", r"^\s+CPCM terms\s+\.+\s+([\d.]+) sec"), ("fock_total", r"^Fock matrix formation\s+\.+\s+([\d.]+) sec"),
                     ("fock_cosx", r"^\s+Chain of spheres X\s+\.+\s+([\d.]+) sec"), ("fock_xc", r"^\s+XC integration\s+\.+\s+([\d.]+) sec"),
                     ("fock_rij", r"^\s+Split-RI-J\s+\.+\s+([\d.]+) sec"),
                     ("grad_total", r"Total SCF gradient time\s+\.+\s+([\d.]+) sec"), ("grad_one_electron", r"One electron gradient\s+\.+\s+([\d.]+) sec"),
                     ("grad_rij", r"RI-J Coulomb gradient\s+\.+\s+([\d.]+) sec"), ("grad_cosx", r"COSX gradient\s+\.+\s+([\d.]+) sec"),
                     ("grad_xc", r"XC gradient\s+\.+\s+([\d.]+) sec"), ("grad_cpcm", r"CPCM gradient\s+\.+\s+([\d.]+) sec"),
                     ("grad_cpcm_amatrix", r"A-Matrix \(El\+Nuc\)\s+\.+\s+([\d.]+) sec"), ("grad_cpcm_potential", r"^\s+Potential\s+\.+\s+([\d.]+) sec"),
                     ("grad_two_electron", r"Two electron gradient\s+\.+\s+([\d.]+) sec"), ("grad_dispersion", r"Dispersion correction\s+\.+\s+([\d.]+) sec")]:
        t[key] = _f(pat, text)
    r["timings"] = {k: v for k, v in t.items() if v is not None}
    r["total_run_time"] = None
    m = re.search(r"TOTAL RUN TIME: (\d+) days (\d+) hours (\d+) minutes (\d+) seconds (\d+) msec", text)
    if m:
        d, h, mi, s, ms = map(int, m.groups())
        r["total_run_time"] = ((d * 24 + h) * 60 + mi) * 60 + s + ms / 1000
    r["warnings"] = sorted(set(l.strip() for l in text.splitlines() if l.startswith("WARNING")))[:20]
    return r


def parse_engrad(path):
    p = pathlib.Path(path)
    if not p.exists():
        return None
    data = [l.strip() for l in p.read_text().splitlines() if l.strip() and not l.strip().startswith("#")]
    n = int(data[0])
    g = [float(x) for x in data[2:2 + 3 * n]]
    return [g[3 * i:3 * i + 3] for i in range(n)]


def parse_cpcm_file(path):
    """<base>.cpcm: header (# comments), atoms (bohr, radius bohr, Z), optional SES spheres, surface points.
    Point columns (vdW Gaussian): X Y Z area potential charge w_leb Switch_F G_width atom (+ sphere for GEPOL Gaussian)."""
    lines = pathlib.Path(path).read_text().splitlines()
    head, i = {}, 0
    while i < len(lines) and not lines[i].startswith("#---"):
        if "#" in lines[i]:
            v, k = lines[i].split("#", 1)
            head[k.strip()] = float(v) if "." in v else int(v)
        i += 1
    sections, cur = {}, None
    for l in lines[i:]:
        if l.startswith("#---"):
            continue
        if l.startswith("#"):
            cur = l.strip("# ").strip()
            sections[cur] = dict(cols=None, rows=[])
            continue
        if not l.strip():
            continue
        if cur and sections[cur]["cols"] is None and re.match(r"^\s*[A-Za-z]", l):
            sections[cur]["cols"] = l.split()
            continue
        if cur:
            sections[cur]["rows"].append([float(x) for x in l.split()])
    atoms = next((s["rows"] for k, s in sections.items() if k.startswith("CARTESIAN")), [])
    spheres = next((s["rows"] for k, s in sections.items() if k.startswith("SES SPHERE")), None)
    pts = next((s for k, s in sections.items() if k.startswith("SURFACE POINTS")), dict(cols=None, rows=[]))
    return dict(header=head, atoms=atoms, ses_spheres=spheres, point_columns=pts["cols"], points=pts["rows"])


def parse_cpcm_corr(path):
    p = pathlib.Path(path)
    if not p.exists():
        return None
    lines = p.read_text().splitlines()
    e = _f(r"Corrected dielectric energy\s+=\s+(\S+)", lines[0])
    q = _f(r"Total C-PCM charge\s+=\s+(\S+)", lines[1])
    return dict(corrected_dielectric_energy=e, total_charge=q, charges=[float(l) for l in lines[3:] if l.strip()])


def parse_property_solvation(path):
    p = pathlib.Path(path)
    if not p.exists():
        return None
    m = re.search(r"\$Solvation_Details(.*?)\$End", p.read_text(), flags=re.S)
    if not m:
        return None
    d = {}
    for k, t, v in re.findall(r"&(\w+) \[&Type \"(\w+)\"\]\s+(.+)", m.group(1)):
        d[k] = v.strip().strip('"') if t == "String" else (int(v) if t == "Integer" else float(v))
    return d


def result_record(run, wd, base="orca"):
    rec = parse_out(run["out"])
    rec["gradient"] = parse_engrad(pathlib.Path(wd) / f"{base}.engrad")
    rec["property_solvation"] = parse_property_solvation(pathlib.Path(wd) / f"{base}.property.txt")
    for k in ["wall", "rc", "timed_out", "slice", "slice_list", "loadavg_before", "loadavg_after"]:
        rec[k] = run[k]
    if not rec["ok"]:
        rec["error_tail"] = "\n".join(run["out"].splitlines()[-25:])
    return rec


def cleanup(wd, keep=(".inp", ".out", ".engrad", ".cpcm", ".cpcm_corr", ".property.txt", ".hess", ".xyz")):
    for f in pathlib.Path(wd).iterdir():
        if f.is_file() and not any(f.name.endswith(k) for k in keep):
            f.unlink()


def write_json(path, obj):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1))


# ----------------------------------------------------------------------------------------------- geoms
def cmd_geoms(a):
    PCM_MOLS.mkdir(parents=True, exist_ok=True)
    charges = {"water": 0, "nh3": 0, "ch3sh": 0, "nh4": 1, "acetate": -1}
    log = {}
    for name, atoms in GUESS.items():
        wd = pathlib.Path(a.work) / "geoms" / name
        atoms = [(s, [x, y, z]) for s, x, y, z in atoms]
        inp = orca_input("B3LYP/G def2-SVP NoRI DefGrid2 TightSCF TightOpt", atoms, charges[name], 1, a.nprocs, 1000)
        r = run_orca(wd, inp, a.slice, 8)
        rec = parse_out(r["out"])
        xyz = wd / "orca.xyz"
        if not rec["ok"] or not xyz.exists() or "OPTIMIZATION RUN DONE" not in r["out"] and "THE OPTIMIZATION HAS CONVERGED" not in r["out"]:
            raise SystemExit(f"{name}: optimization failed, see {wd}")
        opt = read_xyz(xyz)
        lines = [str(len(opt)), f"{name} charge={charges[name]} mult=1; gas-phase B3LYP/G/def2-SVP (exact J/K, DefGrid2, TightOpt) minimum, ORCA 6.0.0; Angstrom"]
        lines += [f"{s:2s} {c[0]:15.8f} {c[1]:15.8f} {c[2]:15.8f}" for s, c in opt]
        (PCM_MOLS / f"{name}.xyz").write_text("\n".join(lines) + "\n")
        log[name] = dict(energy=rec["final_energy"], charge=charges[name])
        print(name, rec["final_energy"], flush=True)
        cleanup(wd)
    write_json(OUT / "geoms.json", dict(level="B3LYP/G def2-SVP NoRI DefGrid2 TightSCF TightOpt (gas phase)", orca="6.0.0", molecules=log))


# ----------------------------------------------------------------------------------------------- facts
# variants: (tag, simple-input solvent keyword, %cpcm block)
FACT_VARIANTS = [
    ("water_default", "CPCM(Water)", ""),
    ("water_vdw_gaussian", "CPCM(Water)", "%cpcm surfacetype vdw_gaussian end"),
    ("water_gepol_ses_gaussian", "CPCM(Water)", "%cpcm surfacetype gepol_ses_gaussian end"),
    ("water_gepol_ses", "CPCM(Water)", "%cpcm surfacetype gepol_ses end"),
    ("water_gepol_sas", "CPCM(Water)", "%cpcm surfacetype gepol_sas end"),
    ("water_numleb302", "CPCM(Water)", "%cpcm num_leb 302 end"),
    ("water_numleb110", "CPCM(Water)", "%cpcm num_leb 110 end"),
    ("water_fepscosmo", "CPCM(Water)", "%cpcm fepstype cosmo end"),
    ("water_xfeps05", "CPCM(Water)", "%cpcm xfeps 0.5 end"),
    ("thf_default", "CPCM(THF)", ""),
    ("eps7.4257", "CPCM", "%cpcm epsilon 7.4257 refrac 1.4050 end"),
    ("water_radiusH1.1", "CPCM(Water)", "%cpcm radius[1] 1.1 end"),
    ("water_scale_gauss1.0", "CPCM(Water)", "%cpcm scale_gauss 1.0 end"),
    ("water_rmin", "CPCM(Water)", "%cpcm rmin 0.5 end"),
    ("water_thresh_h10", "CPCM(Water)", "%cpcm thresh_h 10.0 end"),
    ("water_thresh_noth2", "CPCM(Water)", "%cpcm thresh_noth 2.0 end"),
    ("water_cut_swf1e-4", "CPCM(Water)", "%cpcm cut_swf 1e-4 end"),
]
# radius scan (user radius for O, used unscaled) to pin how ORCA picks the Lebedev grid of a sphere from its area
FACT_VARIANTS += [(f"water_rO{r:.2f}", "CPCM(Water)", f"%cpcm radius[8] {r:.2f} end") for r in
                  [1.40, 1.50, 1.54, 1.55, 1.56, 1.57, 1.60, 1.70, 1.80, 1.90, 1.95, 1.97, 1.98, 1.99, 2.00, 2.02, 2.10, 2.30, 2.40, 2.45, 2.50]]
FACT_MOLS = ["water", "nh3", "ch3sh", "nh4", "acetate"]


def lebedev_tables():
    from pyscf.dft import gen_grid
    tab = {}
    for n in [14, 26, 38, 50, 74, 86, 110, 146, 170, 194, 230, 266, 302, 350, 434, 590, 770, 974, 1202]:
        g = gen_grid.MakeAngularGrid(n)
        tab[n] = g
    return tab


def analyze_cpcm(cp, epsilon, feps_x=0.0, dm_potential=None):
    """Reconstruct ORCA's Gaussian-charge C-PCM from the parsed .cpcm file and report how well each model piece fits."""
    import numpy as np
    from scipy.special import erf, erfc
    cols = cp["point_columns"]
    P = np.array(cp["points"])
    A = np.array(cp["atoms"])
    xyz_a, rad, Z = A[:, :3], A[:, 3], A[:, 4].astype(int)
    res = dict(npoints=len(P), columns=cols)
    if cols is None or "w_leb" not in cols:
        # point-charge (GEPOL) layout: X Y Z area potential charge atom
        X, area, V, q = P[:, :3], P[:, 3], P[:, 4], P[:, 5]
        res["sum_q"] = float(q.sum())
        res["half_qV"] = float(0.5 * q @ V)
        res["area_sum"] = float(area.sum())
        # point-charge A matrix: A_ij = 1/r_ij, A_ii = k sqrt(4 pi / a_i); fit k from A q = -V row by row
        R = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2)
        np.fill_diagonal(R, np.inf)
        k = (-V - (q[None, :] / R).sum(1)) / (q * np.sqrt(4 * np.pi / area))
        res["diag_factor_k_median"] = float(np.median(k))
        res["diag_factor_k_minmax"] = [float(k.min()), float(k.max())]
        return res
    X, area, V, q, w, F, zeta, iat = P[:, :3], P[:, 3], P[:, 4], P[:, 5], P[:, 6], P[:, 7], P[:, 8], P[:, 9].astype(int)
    tab = lebedev_tables()
    # 1. Lebedev order per sphere: the 4pi-normalized weights of the grid must contain the atom's w_leb values
    order = {}
    for ia in range(len(A)):
        ws = np.unique(np.round(w[iat == ia] / (4 * np.pi), 12))
        cand = [n for n, g in tab.items() if all(np.min(np.abs(g[:, 3] - x)) < 1e-9 for x in ws)]
        order[ia] = cand
    res["lebedev_candidates_per_atom"] = {int(k): v for k, v in order.items()}
    # unique choice: the smallest grid that contains all weights and has at least as many points as were kept
    nleb = {ia: next((n for n in c if n >= int((iat == ia).sum())), None) for ia, c in order.items()}
    res["lebedev_order_per_atom"] = {int(k): v for k, v in nleb.items()}
    res["points_kept_per_atom"] = {int(ia): int((iat == ia).sum()) for ia in range(len(A))}
    # sphere area * density threshold (5 per Ang^2) -> which grid ORCA picks
    res["sphere_area_ang2_times_5"] = {int(ia): float(4 * np.pi * (rad[ia] * BOHR) ** 2 * 5.0) for ia in range(len(A))}
    # 2. Gaussian exponent: zeta_i = zeta_N / (R_I sqrt(w_i)), w_i normalized to 4 pi
    zN = zeta * rad[iat] * np.sqrt(w)
    res["zeta_N_per_atom"] = {int(ia): [float(zN[iat == ia].min()), float(zN[iat == ia].max())] for ia in range(len(A))}
    # 3. switching function: York-Karplus F_i = prod_{J != I} [1 - 1/2 (erf(zeta_i (R_J - r_iJ)) + erf(zeta_i (R_J + r_iJ)))]
    Fyk = np.ones(len(P))
    for j in range(len(A)):
        r = np.linalg.norm(X - xyz_a[j], axis=1)
        m = iat != j
        Fyk[m] *= 1 - 0.5 * (erf(zeta[m] * (rad[j] - r[m])) + erf(zeta[m] * (rad[j] + r[m])))
    res["switch_yk_maxabs_dev"] = float(np.max(np.abs(Fyk - F)))
    res["switch_min_kept"] = float(F.min())
    # 4. area = w_i R_I^2 F_i
    res["area_maxrel_dev"] = float(np.max(np.abs(area - w * rad[iat] ** 2 * F) / area))
    res["area_sum"] = float(area.sum())
    # 5. point positions: atom center + R_I * Lebedev direction
    d = (X - xyz_a[iat]) / rad[iat][:, None]
    res["point_radius_maxdev"] = float(np.max(np.abs(np.linalg.norm(d, axis=1) - 1)))
    dirs_ok = True
    for ia in range(len(A)):
        n = nleb[ia]
        if n is None:
            dirs_ok = False
            continue
        g = tab[n][:, :3]
        dd = d[iat == ia]
        dist = np.min(np.linalg.norm(dd[:, None, :] - g[None, :, :], axis=2), axis=1)
        if dist.max() > 1e-6:
            dirs_ok = False
    res["points_on_standard_lebedev_directions"] = dirs_ok
    # 6. A matrix (York-Karplus/Scalmani-Frisch): A_ii = zeta_i sqrt(2/pi) / F_i, A_ij = erf(zeta_ij r_ij) / r_ij
    R = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2)
    zz = zeta[:, None] * zeta[None, :] / np.sqrt(zeta[:, None] ** 2 + zeta[None, :] ** 2)
    with np.errstate(divide="ignore", invalid="ignore"):
        Am = erf(zz * R) / R
    np.fill_diagonal(Am, zeta * np.sqrt(2 / np.pi) / F)
    q0 = -np.linalg.solve(Am, V)  # charges of the conductor (f = 1)
    feps = (epsilon - 1) / (epsilon + feps_x) if epsilon is not None else 1.0
    res["f_eps"] = feps
    res["q_file_over_q_conductor"] = float((q @ q0) / (q0 @ q0))
    res["q_file_vs_q_conductor_maxabs"] = float(np.max(np.abs(q - q0)))
    res["q_file_vs_feps_q_conductor_maxabs"] = float(np.max(np.abs(q - feps * q0)))
    res["sum_q_file"] = float(q.sum())
    res["half_qV_file"] = float(0.5 * q @ V)
    res["half_feps_qV"] = float(0.5 * feps * q @ V)
    res["half_q0V"] = float(0.5 * q0 @ V)
    # 7. nuclear part of the potential: point nuclei vs Gaussian-smeared surface charges
    rn = np.linalg.norm(X[:, None, :] - xyz_a[None, :, :], axis=2)
    res["_vnuc_point"] = (Z[None, :] / rn).sum(1)
    res["_vnuc_gauss"] = (Z[None, :] * erf(zeta[:, None] * rn) / rn).sum(1)
    if dm_potential is not None:
        vel_point, vel_gauss = dm_potential
        for nn in ["point", "gauss"]:
            for ee, vel in [("point", vel_point), ("gauss", vel_gauss)]:
                v = res[f"_vnuc_{nn}"] + vel
                res[f"V_nuc_{nn}_el_{ee}_maxabs_dev"] = float(np.max(np.abs(v - V)))
    else:
        res["V_minus_vnuc_point_mean"] = float(np.mean(V - res["_vnuc_point"]))
    del res["_vnuc_point"], res["_vnuc_gauss"]
    return res


def electronic_potential(wd, base, points, zeta):
    """Electronic potential of ORCA's density (via orca_2mkl molden) at the surface points: point charges and Gaussians."""
    import numpy as np
    from pyscf import gto, scf
    from pyscf.tools import molden
    env = dict(os.environ, LD_LIBRARY_PATH=f"{ORCA}/lib")
    subprocess.run([str(ORCA / "orca_2mkl"), base, "-molden"], cwd=wd, env=env, capture_output=True)
    mol, mo_e, mo_c, mo_occ, _, _ = molden.load(str(pathlib.Path(wd) / f"{base}.molden.input"))
    # orca_2mkl writes primitive coefficients that already include the primitive normalization; PySCF would
    # normalize them again (N_elec 8.87 instead of 10 for water). Undo it and rebuild the basis.
    from pyscf.gto.mole import gto_norm
    nb = {k: [[sh[0]] + [[p[0]] + [c / gto_norm(sh[0], p[0]) for c in p[1:]] for p in sh[1:]] for sh in shells] for k, shells in mol._basis.items()}
    mol = mol.copy()
    mol.basis = nb
    mol.build()
    if isinstance(mo_c, tuple) or (hasattr(mo_c, "ndim") and mo_c.ndim == 3):
        dm = sum(c[:, o > 0] * o[o > 0] @ c[:, o > 0].T for c, o in zip(mo_c, mo_occ))
    else:
        dm = (mo_c * mo_occ) @ mo_c.T
    S = mol.intor("int1e_ovlp")
    nelec = float(np.einsum("ij,ji", dm, S))
    pts = np.asarray(points)
    # point charges: -sum_mn D_mn <m|1/|r-C||n>
    vp = -np.einsum("gij,ij->g", mol.intor("int1e_grids", grids=pts), dm)
    fake = gto.fakemol_for_charges(pts, expnt=np.asarray(zeta) ** 2)
    from pyscf.df import incore
    int3c = incore.aux_e2(mol, fake, intor="int3c2e")  # (nao, nao, npts), fakemol is normalized to unit charge
    vg = -np.einsum("ijg,ij->g", int3c, dm)
    return nelec, mol, dm, vp, vg


def cmd_facts(a):
    out = OUT / "facts"
    summary = json.loads((out / "summary.json").read_text()) if (out / "summary.json").exists() else {}
    for mol in a.mols.split(","):
        atoms, charge, mult = system(mol)
        for tag, solv, block in FACT_VARIANTS:
            if a.only and not any(x in tag for x in a.only.split(",")):
                continue
            if mol != "water" and tag not in ("water_default", "thf_default", "water_gepol_ses_gaussian", "water_gepol_ses", "water_numleb302"):
                continue
            key = f"{mol}_{tag}"
            wd = pathlib.Path(a.work) / "facts" / key
            calc = "EnGrad" if a.grad else ""
            kw = f"HF def2-SVP NoRI TightSCF {solv} {calc}".strip()
            r = run_orca(wd, orca_input(kw, atoms, charge, mult, a.nprocs, 1000, block), a.slice, 8)
            engrad_abort = None
            if "Analytical Gradient is not implemented" in r["out"]:
                # ORCA refuses EnGrad for this surface: keep its message, then rerun as a single point
                engrad_abort = sorted(set(l.strip() for l in r["out"].splitlines() if "not implemented" in l))
                kw = kw.replace(" EnGrad", "")
                r = run_orca(wd, orca_input(kw, atoms, charge, mult, a.nprocs, 1000, block), a.slice, 8)
            rec = result_record(r, wd)
            rec["engrad_abort"] = engrad_abort
            rec.update(case=key, mol=mol, variant=tag, keywords=f"! {kw} Bohrs", block=block, charge=charge, mult=mult)
            cpf = wd / "orca.cpcm"
            if rec["ok"] and cpf.exists():
                cp = parse_cpcm_file(cpf)
                corr = parse_cpcm_corr(wd / "orca.cpcm_corr")
                rec["cpcm_file_header"] = cp["header"]
                rec["cpcm_file_point_columns"] = cp["point_columns"]
                eps = rec["cpcm"]["epsilon"] if rec["cpcm"] else None
                eps = None if isinstance(eps, str) else eps
                dmpot = None
                try:
                    if cp["point_columns"] and "G_width" in cp["point_columns"]:
                        import numpy as np
                        P = np.array(cp["points"])
                        nelec, _, _, vp, vg = electronic_potential(wd, "orca", P[:, :3], P[:, 8])
                        rec["molden_nelec"] = nelec
                        dmpot = (vp, vg)
                except Exception as e:  # analysis is optional
                    rec["potential_check_error"] = repr(e)
                rec["analysis"] = analyze_cpcm(cp, eps, rec["cpcm"].get("feps_x") or 0.0, dmpot) if rec["cpcm"] else None
                if corr:
                    rec["cpcm_corr"] = dict(corrected_dielectric_energy=corr["corrected_dielectric_energy"], total_charge=corr["total_charge"],
                                            n=len(corr["charges"]))
                # the cavity itself (small molecules only): saved as a separate JSON for linus to compare point by point
                if mol in ("water", "ch3sh", "nh4", "acetate") and tag in ("water_default", "thf_default", "water_gepol_ses_gaussian", "water_gepol_ses", "water_numleb302"):
                    write_json(out / "cavities" / f"{key}.json", dict(case=key, keywords=f"! {kw} Bohrs", block=block, units="bohr, bohr^2, Eh/e",
                                                                       header=cp["header"], atoms_xyz_radius_Z=cp["atoms"],
                                                                       ses_spheres=cp["ses_spheres"], point_columns=cp["point_columns"],
                                                                       points=cp["points"], corrected=corr))
            summary[key] = {k: rec.get(k) for k in ["ok", "final_energy"]}
            summary[key]["cpcm"] = rec.get("cpcm")
            summary[key]["dielectric"] = rec["scf_components"]["cpcm_dielectric"] if rec.get("scf_components") else None
            write_json(out / f"{key}.json", rec)
            an = rec.get("analysis") or {}
            print(f"{key:34s} ok={rec['ok']} E={rec['final_energy']} npts={(rec.get('cpcm') or {}).get('npoints')} "
                  f"Ediel={summary[key]['dielectric']} swdev={an.get('switch_yk_maxabs_dev')} qratio={an.get('q_file_over_q_conductor')}", flush=True)
            if not rec["ok"]:
                print(rec.get("error_tail", "")[-800:], flush=True)
    write_json(out / "summary.json", summary)


# ----------------------------------------------------------------------------------------------- ref
def ref_cases(a):
    cases = []
    for mol in a.mols.split(","):
        bases = ["svp", "tzvp"] if mol not in ("la18", "u1an39", "ts61", "ts113") else ["svp"]
        for method in a.methods.split(","):
            for basis in bases:
                if a.bases and basis not in a.bases.split(","):
                    continue
                for jk in a.jk.split(","):
                    cases.append((mol, method, basis, jk))
    return cases


def ref_keywords(mol, method, basis, jk, solv, calc="EnGrad"):
    _, mult, _ = SYSTEMS[mol]
    m = METHODS[method][1 if mult != 1 else 0]
    return " ".join(x for x in [m, BASES[basis], JK[jk], "TightSCF", SOLVENTS[solv], calc] if x)


def cmd_ref(a):
    out = OUT / "ref"
    slices = a.slices.split(",")
    free = list(slices)
    lk = threading.Lock()
    cases = ref_cases(a)

    def do(case):
        mol, method, basis, jk = case
        name = f"{mol}_{method}_{basis}_{jk}"
        target = [out / f"{name}_{s}.json" for s in a.solvents.split(",")]
        if not a.force and all(t.exists() and json.loads(t.read_text()).get("ok") for t in target):
            return name, "cached"
        with lk:
            slc = free.pop(0)
        try:
            atoms, charge, mult = system(mol)
            res = {}
            for solv in ["gas"] + a.solvents.split(","):
                kw = ref_keywords(mol, method, basis, jk, solv)
                wd = pathlib.Path(a.work) / "ref" / f"{name}_{solv}"
                maxcore = a.maxcore
                mem = max(8, int(a.nprocs * maxcore / 1000 * 1.5) + 4)
                r = run_orca(wd, orca_input(kw, atoms, charge, mult, a.nprocs, maxcore), slc, mem, timeout=a.timeout)
                rec = result_record(r, wd)
                rec.update(keywords=f"! {kw} Bohrs", nprocs=a.nprocs, maxcore=maxcore)
                res[solv] = rec
                cleanup(wd)
            gas = res["gas"]
            for solv in a.solvents.split(","):
                s = res[solv]
                cp = s.get("cpcm") or {}
                rec = dict(case=f"{name}_{solv}", mol=mol, charge=charge, mult=mult, method=method, basis=BASES[basis], jk=jk, solvent=solv,
                           orca_version=s["version"], keywords=s["keywords"], gas_keywords=gas["keywords"], nprocs=a.nprocs, maxcore=a.maxcore,
                           geometry_bohr=[[sym] + [x / BOHR for x in c] for sym, c in atoms], units="Eh, bohr, Eh/bohr",
                           ok=bool(s["ok"] and gas["ok"]), energy=s["final_energy"], gas_energy=gas["final_energy"],
                           delta_solv=(s["final_energy"] - gas["final_energy"]) if s["final_energy"] is not None and gas["final_energy"] is not None else None,
                           cpcm_dielectric_energy=s["scf_components"]["cpcm_dielectric"], gradient=s["gradient"], gas_gradient=gas["gradient"],
                           npoints=cp.get("npoints"), area_bohr2=cp.get("area_bohr2"), volume_bohr3=cp.get("volume_bohr3"),
                           surface_charge=cp.get("surface_charge"), corrected_charge=cp.get("corrected_charge"),
                           outlying_charge_correction=cp.get("outlying_charge_correction"), epsilon=cp.get("epsilon"), refrac=cp.get("refrac"),
                           surface_type=cp.get("surface_type"), radii_bohr=cp.get("radii_bohr"), scf_components=s["scf_components"],
                           gas_scf_components=gas["scf_components"], scf_iterations=s["scf_iterations"], gas_scf_iterations=gas["scf_iterations"],
                           wall=s["wall"], gas_wall=gas["wall"], slice=slc, warnings=s["warnings"], time=time.strftime("%Y-%m-%dT%H:%M:%S"))
                if not rec["ok"]:
                    rec["error_tail"] = s.get("error_tail") or gas.get("error_tail")
                write_json(out / f"{name}_{solv}.json", rec)
            return name, " ".join(f"{k}:{'ok' if v['ok'] else 'FAIL'} {v['final_energy']}" for k, v in res.items())
        finally:
            with lk:
                free.append(slc)

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(slices)) as ex:
        for name, msg in ex.map(do, cases):
            print(f"{name:40s} {msg}", flush=True)


# ----------------------------------------------------------------------------------------------- freq
def parse_freqs(text):
    m = re.search(r"VIBRATIONAL FREQUENCIES\s*\n-+\n(.*?)\n\s*\n-+\s*\nNORMAL MODES", text, flags=re.S)
    if not m:
        blk = text.split("VIBRATIONAL FREQUENCIES")[-1] if "VIBRATIONAL FREQUENCIES" in text else ""
    else:
        blk = m.group(1)
    return [float(x) for x in re.findall(r"^\s*\d+:\s+(-?\d+\.\d+) cm\*\*-1", blk, flags=re.M)]


def freq_details(out):
    """Analytic vs numerical Hessian, with the evidence lines and the timed steps ('=> ... done (x sec)')."""
    steps = {}
    for name, t in re.findall(r"=> (.+?)\s+\.\.\. done \(\s*([\d.]+) sec\)", out):
        steps[name.strip()] = steps.get(name.strip(), 0.0) + float(t)
    numerical = "NUMERICAL FREQUENCIES" in out.upper() and "displacement" in out.lower()
    analytic = "SCF HESSIAN" in out and not numerical
    cpcm_terms = "CPCM-Hessian" in out and "CPCM response operators" in out
    evidence = sorted(set(l.strip() for l in out.splitlines()
                          if re.search(r"SCF Hessian\s+\.\.\.|CPCM-Hessian|CPCM derivative integrals|CPCM response operators|CP-SCF equations solved|"
                                       r"NUMERICAL FREQ|not implemented|Analytical Hessian", l)))
    cpscf = _f(r"CP-SCF equations solved in\s+([\d.]+) sec", out)
    mods = {k: _f(rf"^{re.escape(k)}\s+\.\.\.\s+([\d.]+) sec", out) for k in
            ["Sum of individual times", "Startup calculation", "SCF iterations", "Property integrals", "SCF Response", "Property calculations"]}
    return dict(analytic_hessian=analytic, numerical_hessian=numerical, analytic_cpcm_hessian_terms=cpcm_terms, evidence=evidence, steps_sec=steps, cpscf_sec=cpscf,
                module_times_sec={k: v for k, v in mods.items() if v is not None})


def parse_hess(path):
    """$hessian block of ORCA's .hess file (Eh/bohr^2, Cartesian, not mass weighted)."""
    p = pathlib.Path(path)
    if not p.exists():
        return None
    lines = p.read_text().splitlines()
    i = lines.index("$hessian") + 1
    n = int(lines[i].split()[0])
    H = [[0.0] * n for _ in range(n)]
    i += 1
    while i < len(lines) and lines[i].strip() and not lines[i].startswith("$"):
        cols = [int(c) for c in lines[i].split()]
        for r in range(n):
            vals = lines[i + 1 + r].split()
            for c, v in zip(cols, vals[1:]):
                H[int(vals[0])][c] = float(v)
        i += n + 1
    return H


def cmd_freq(a):
    atoms, charge, mult = system("la18")
    for solv in a.solvents.split(","):
        kw = " ".join(x for x in ["B3LYP/G def2-SVP def2/J RIJCOSX DefGrid2 TightSCF", SOLVENTS[solv], "Freq"] if x)
        wd = pathlib.Path(a.work) / "freq" / f"la18_{solv}"
        target = OUT / "freq" / f"la18_{solv}.json"
        if a.reparse:
            rec = json.loads(target.read_text())
            out = (wd / "orca.out").read_text(errors="replace")
        else:
            r = run_orca(wd, orca_input(kw, atoms, charge, mult, 8, a.maxcore), a.slice, a.mem, lock=a.lock, timeout=a.timeout)
            rec = result_record(r, wd)
            out = r["out"]
            rec.update(case=f"la18_b3lyp_svp_rijcosx_{solv}_freq", keywords=f"! {kw} Bohrs", nprocs=8, maxcore=a.maxcore,
                       time=time.strftime("%Y-%m-%dT%H:%M:%S"))
        rec.pop("hessian_markers", None)
        rec.update(frequencies_cm1=parse_freqs(out), hessian=parse_hess(wd / "orca.hess"), **freq_details(out),
                   note="la18 (bench/molecules/la18.xyz) is not a stationary point at this level: 3 imaginary modes, as in the gas phase "
                        "(bench/results/2026-11-07-vib/la18_b3lyp.pyscf.json: -235, -230, -56 cm-1)")
        write_json(target, rec)
        print(solv, rec["ok"], f"wall {rec['wall']:.1f}", "analytic" if rec["analytic_hessian"] else "", "numerical" if rec["numerical_hessian"] else "",
              rec["frequencies_cm1"][6:10], flush=True)
        if not a.reparse:
            cleanup(wd)


# ----------------------------------------------------------------------------------------------- bench
BENCH_LEVELS = {"wb97x-tzvp": "wB97X-D3BJ def2-TZVP def2/J RIJCOSX DefGrid2 TightSCF", "b3lyp-svp": "B3LYP/G def2-SVP def2/J RIJCOSX DefGrid2 TightSCF"}
BENCH_MEM = {"la18": 16, "u1an39": 16, "ts61": 20, "ts113": 24}


def cmd_bench(a):
    out = pathlib.Path(a.outdir) if a.outdir else OUT / "bench"
    out.mkdir(parents=True, exist_ok=True)
    grad = a.calc == "engrad"
    for mol in a.mols.split(","):
        atoms, charge, mult = system(mol)
        reps = 1 if mol == "ts113" else a.reps
        for level in a.levels.split(","):
            kinds = a.kinds.split(",")
            runs = {k: [] for k in kinds}
            target = out / f"{mol}_{level}_{a.calc}.json"
            if target.exists() and not a.force:
                print("cached", target.name)
                continue
            for rep in range(reps):
                order = kinds if rep % 2 == 0 else kinds[::-1]
                for kind in order:
                    kw = BENCH_LEVELS[level] + (" CPCM(THF)" if kind == "cpcm" else "") + (" EnGrad" if grad else "")
                    wd = pathlib.Path(a.work) / "bench" / f"{mol}_{level}_{kind}_r{rep}"
                    r = run_orca(wd, orca_input(kw, atoms, charge, mult, 8, a.maxcore), "bench", BENCH_MEM[mol], lock=a.lock,
                                 timeout=a.timeout_ts113 if mol == "ts113" else a.timeout)
                    rec = result_record(r, wd)
                    rec.update(rep=rep, keywords=f"! {kw} Bohrs", time=time.strftime("%Y-%m-%dT%H:%M:%S"))
                    runs[kind].append(rec)
                    t = rec["timings"]
                    print(f"{mol:7s} {level:11s} {kind:5s} rep {rep}: {'ok ' if rec['ok'] else 'FAIL'} wall {rec['wall']:8.1f} scf {t.get('scf')} "
                          f"grad {t.get('gradient')} cpcm_fock {t.get('fock_cpcm_terms')} cpcm_grad {t.get('grad_cpcm')} E {rec['final_energy']}", flush=True)
                    cleanup(wd)
            summary = {}
            for kind, rs in runs.items():
                ok = [r for r in rs if r["ok"]]
                if ok:
                    best = min(ok, key=lambda r: r["wall"])
                    summary[kind] = dict(best_wall=best["wall"], walls=[r["wall"] for r in rs], energy=best["final_energy"],
                                         cpcm_dielectric=best["scf_components"]["cpcm_dielectric"], timings_best=best["timings"],
                                         scf_iterations=best["scf_iterations"], npoints=(best.get("cpcm") or {}).get("npoints"))
            write_json(target, dict(case=f"{mol}_{level}_{a.calc}", mol=mol, charge=charge, mult=mult, level=level, keywords=BENCH_LEVELS[level],
                                    solvent="CPCM(THF)", nprocs=8, maxcore=a.maxcore, slice="bench", memory_max_gb=BENCH_MEM[mol],
                                    orca_version="6.0.0", reps=reps, summary=summary, runs=runs))


# ----------------------------------------------------------------------------------------------- fdcheck
def cmd_fdcheck(a):
    """Is ORCA's analytic CPCM gradient the derivative of its CPCM energy? Central differences (h bohr) on a few
    Cartesian components, HF/def2-SVP exact J/K, VeryTightSCF, CPCM(Water) vdW Gaussian, and the same in the gas phase."""
    out = {}
    for mol in a.mols.split(","):
        atoms, charge, mult = system(mol)
        n = len(atoms)
        comps = [(i, k) for i in range(n) for k in range(3)][: a.ncomp]
        for solv in ["water", "gas"]:
            kw = " ".join(x for x in [METHODS["hf"][0 if mult == 1 else 1], "def2-SVP NoRI VeryTightSCF", SOLVENTS[solv]] if x)
            wd = pathlib.Path(a.work) / "fdcheck" / f"{mol}_{solv}"
            run_orca(wd / "ref", orca_input(kw + " EnGrad", atoms, charge, mult, a.nprocs, 1000), a.slice, 8)
            g = parse_engrad(wd / "ref" / "orca.engrad")
            rows = []
            for i, k in comps:
                es = []
                for sgn in (1, -1):
                    dat = [(sym, list(c)) for sym, c in atoms]
                    dat[i][1][k] += sgn * a.h * BOHR
                    rr = run_orca(wd / f"d{i}{k}{'p' if sgn > 0 else 'm'}", orca_input(kw, dat, charge, mult, a.nprocs, 1000), a.slice, 8)
                    es.append(parse_out(rr["out"])["final_energy"])
                fd = (es[0] - es[1]) / (2 * a.h)
                rows.append(dict(atom=i, xyz=k, analytic=g[i][k], fd=fd, diff=g[i][k] - fd))
            out[f"{mol}_{solv}"] = dict(keywords=f"! {kw} Bohrs", h_bohr=a.h, rows=rows, max_abs_diff=max(abs(x["diff"]) for x in rows))
            print(mol, solv, "max |analytic - fd| =", out[f"{mol}_{solv}"]["max_abs_diff"], flush=True)
            shutil.rmtree(wd, ignore_errors=True)
    write_json(OUT / "fdcheck.json", dict(orca="6.0.0", note="central differences of FINAL SINGLE POINT ENERGY; Eh/bohr", cases=out))


# ----------------------------------------------------------------------------------------------- summary
def cmd_summary(a):
    """Compact tables of the ref and bench JSONs (ref/summary.json, bench/summary.json)."""
    rows = {}
    for f in sorted((OUT / "ref").glob("*.json")):
        if f.name == "summary.json":
            continue
        d = json.loads(f.read_text())
        g = d["gradient"]
        rows[d["case"]] = dict(ok=d["ok"], energy=d["energy"], gas_energy=d["gas_energy"], delta_solv_kcal=d["delta_solv"] * 627.5094740631,
                               cpcm_dielectric=d["cpcm_dielectric_energy"], npoints=d["npoints"], area_bohr2=d["area_bohr2"], volume_bohr3=d["volume_bohr3"],
                               surface_charge=d["surface_charge"], epsilon=d["epsilon"], max_abs_grad=max(abs(x) for r in g for x in r) if g else None)
    write_json(OUT / "ref" / "summary.json", dict(orca="6.0.0", units="Eh; delta_solv in kcal/mol (E_CPCM - E_gas at the same settings)", cases=rows))
    print(f"ref: {len(rows)} cases, {sum(r['ok'] for r in rows.values())} ok")
    rows = {}
    for f in sorted((OUT / "bench").glob("*_engrad.json")):
        d = json.loads(f.read_text())
        s = d["summary"]
        row = dict(mol=d["mol"], level=d["level"], reps=d["reps"])
        for kind in ["gas", "cpcm"]:
            if kind in s:
                t = s[kind]["timings_best"]
                row[kind] = dict(best_wall=s[kind]["best_wall"], walls=s[kind]["walls"], scf=t.get("scf"), gradient=t.get("gradient"),
                                 startup=t.get("startup"), fock_cpcm_terms=t.get("fock_cpcm_terms"), fock_total=t.get("fock_total"),
                                 grad_cpcm=t.get("grad_cpcm"), grad_total=t.get("grad_total"), scf_iterations=s[kind]["scf_iterations"],
                                 energy=s[kind]["energy"], cpcm_dielectric=s[kind]["cpcm_dielectric"], npoints=s[kind]["npoints"])
        if "gas" in row and "cpcm" in row:
            row["cpcm_over_gas_wall"] = row["cpcm"]["best_wall"] / row["gas"]["best_wall"]
        rows[d["case"]] = row
        print(f"{d['case']:28s} " + "  ".join(f"{k}: {row[k]['best_wall']:8.1f} s (scf {row[k]['scf']}, grad {row[k]['gradient']}, cpcm fock {row[k]['fock_cpcm_terms']}, cpcm grad {row[k]['grad_cpcm']})" for k in ["gas", "cpcm"] if k in row))
    write_json(OUT / "bench" / "summary.json", dict(orca="6.0.0", nprocs=8, slice="bench (CCD0 physical cores 0-7)", units="seconds, Eh", cases=rows))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("geoms")
    g.add_argument("--slice", default="ccd3")
    g.add_argument("--nprocs", type=int, default=4)
    f = sub.add_parser("facts")
    f.add_argument("--slice", default="ccd3")
    f.add_argument("--nprocs", type=int, default=4)
    f.add_argument("--mols", default=",".join(FACT_MOLS))
    f.add_argument("--only", default="")
    f.add_argument("--grad", action="store_true")
    r = sub.add_parser("ref")
    r.add_argument("--slices", default="ccd2,ccd3")
    r.add_argument("--nprocs", type=int, default=4)
    r.add_argument("--maxcore", type=int, default=1500)
    r.add_argument("--mols", default="water,nh3,ch3sh,nh4,acetate,ch3sh+,la18")
    r.add_argument("--methods", default="hf,b3lyp,wb97x-d3bj")
    r.add_argument("--bases", default="")
    r.add_argument("--jk", default="exact,rijcosx")
    r.add_argument("--solvents", default="thf,water")
    r.add_argument("--timeout", type=float, default=3600)
    r.add_argument("--force", action="store_true")
    q = sub.add_parser("freq")
    q.add_argument("--lock", default=os.environ.get("BENCH_LOCK", ""))
    q.add_argument("--slice", default="bench")
    q.add_argument("--maxcore", type=int, default=1500)
    q.add_argument("--mem", type=int, default=24)
    q.add_argument("--solvents", default="thf")
    q.add_argument("--timeout", type=float, default=5400)
    q.add_argument("--reparse", action="store_true", help="re-read the kept orca.out into the JSON (no run)")
    b = sub.add_parser("bench")
    b.add_argument("--lock", default=os.environ.get("BENCH_LOCK", ""))
    b.add_argument("--mols", default="la18,u1an39,ts61,ts113")
    b.add_argument("--levels", default="b3lyp-svp,wb97x-tzvp")
    b.add_argument("--reps", type=int, default=2)
    b.add_argument("--maxcore", type=int, default=1500)
    b.add_argument("--timeout", type=float, default=3600)
    b.add_argument("--timeout-ts113", type=float, default=5400)
    b.add_argument("--force", action="store_true")
    b.add_argument("--calc", default="engrad", choices=["engrad", "energy"])
    b.add_argument("--kinds", default="cpcm,gas", help="cpcm and/or gas runs")
    b.add_argument("--outdir", default="", help="result directory (default: the 2026-11-07 PCM bench dir)")
    fc = sub.add_parser("fdcheck")
    fc.add_argument("--slice", default="ccd3")
    fc.add_argument("--nprocs", type=int, default=4)
    fc.add_argument("--mols", default="water,nh4,acetate")
    fc.add_argument("--ncomp", type=int, default=6)
    fc.add_argument("--h", type=float, default=1e-3)
    sm = sub.add_parser("summary")
    for p in [g, f, r, q, b, fc, sm]:
        p.add_argument("--work", default=str(WORK))
    a = ap.parse_args()
    dict(geoms=cmd_geoms, facts=cmd_facts, ref=cmd_ref, freq=cmd_freq, bench=cmd_bench, fdcheck=cmd_fdcheck, summary=cmd_summary)[a.cmd](a)


if __name__ == "__main__":
    main()
