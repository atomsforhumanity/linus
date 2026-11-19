"""PCM validation matrix against PySCF 2.15: runs tests/python/compare_pcm_pyscf.py over molecules,
methods, models, permittivities and J/K schemes, a few cases at a time (one CPU slice each), and
appends one JSON line per case to the output file.

  .refenv/bin/python bench/pcm_validate.py --out bench/results/2026-11-07-pcm/validation_pyscf.jsonl \
      [--set energy-gradient|hessian|smd] [--slices ccd2,ccd7] [--threads 8] [--linus target/release/linus]
"""
import argparse, itertools, json, pathlib, queue, subprocess, sys, threading

ROOT = pathlib.Path(__file__).resolve().parents[1]
from bench_paths import REFENV  # noqa: E402  (env override LINUS_REFENV: bench/bench_paths.py)
PY = str(REFENV / "bin/python")
MOLS = {  # name: (xyz, charge, mult)
    "water": ("tests/molecules/water.xyz", 0, 1),
    "nh3": ("tests/molecules/nh3.xyz", 0, 1),
    "ch3sh": ("tests/molecules/ch3sh.xyz", 0, 1),
    "nh4+": ("tests/molecules/nh4.xyz", 1, 1),
    "acetate": ("tests/molecules/acetate.xyz", -1, 1),
    "ch3sh+": ("tests/molecules/ch3sh.xyz", 1, 2),
    "la18": ("bench/molecules/la18.xyz", 0, 1),
}


def cases(which):
    small = ["water", "nh3", "ch3sh", "nh4+", "acetate", "ch3sh+"]
    out = []
    if which == "energy-gradient":
        # every model, small molecules, HF and B3LYP, exact J/K, THF
        for m, meth, model in itertools.product(small, ["hf", "b3lyp"], ["cpcm", "cosmo", "iefpcm", "ssvpe"]):
            out.append(dict(mol=m, method=meth, model=model, eps=7.4257, df=False, grad=True))
        # water permittivity, C-PCM and IEF-PCM
        for m, meth, model in itertools.product(small, ["hf", "b3lyp"], ["cpcm", "iefpcm"]):
            out.append(dict(mol=m, method=meth, model=model, eps=78.36, df=False, grad=True))
        # wB97X-D3BJ (range-separated hybrid), exact and DF (DF: energies only; PySCF's DF gradient
        # is not the derivative of the wK energy compare_pcm_pyscf.py builds)
        for m, model, df in itertools.product(["water", "acetate", "ch3sh+"], ["cpcm", "iefpcm"], [False, True]):
            out.append(dict(mol=m, method="wb97x-d3bj", model=model, eps=7.4257, df=df, grad=not df))
        # DF J/K
        for m, meth, model in itertools.product(["nh3", "acetate", "ch3sh+"], ["hf", "b3lyp"], ["cpcm", "cosmo", "iefpcm", "ssvpe"]):
            out.append(dict(mol=m, method=meth, model=model, eps=7.4257, df=True, grad=True))
        # ISWIG and other Lebedev grids
        for m, model, leb, disc in [("ch3sh", "cpcm", 302, "iswig"), ("acetate", "iefpcm", 302, "iswig"), ("ch3sh", "cpcm", 110, "swig"), ("acetate", "iefpcm", 590, "swig")]:
            out.append(dict(mol=m, method="b3lyp", model=model, eps=7.4257, df=False, grad=True, lebedev=leb, discretization=disc))
        # L-lactide
        for meth, model, df in itertools.product(["hf", "b3lyp"], ["cpcm", "iefpcm"], [False, True]):
            out.append(dict(mol="la18", method=meth, model=model, eps=7.4257, df=df, grad=True))
        out.append(dict(mol="la18", method="wb97x-d3bj", model="cpcm", eps=7.4257, df=False, grad=True))
    elif which == "hessian":
        # PySCF's PCM Hessian is slow (water HF/def2-SVP: 336 s), so small molecules only
        for m, meth, model in [(m, "hf", x) for m in ["water", "nh3", "ch3sh", "nh4+", "acetate", "ch3sh+"] for x in ["cpcm", "cosmo", "iefpcm"]] + \
                [(m, "b3lyp", x) for m in ["water", "nh3", "ch3sh", "nh4+", "acetate", "ch3sh+"] for x in ["cpcm", "iefpcm"]]:
            out.append(dict(mol=m, method=meth, model=model, eps=7.4257, df=False, hess=True))
        for m, meth in itertools.product(["nh3", "acetate", "ch3sh+"], ["hf", "b3lyp"]):
            out.append(dict(mol=m, method=meth, model="cpcm", eps=78.36, df=True, hess=True))
        for m in ["water", "acetate"]:
            out.append(dict(mol=m, method="hf", model="smd", solvent="water", eps=0.0, df=False, hess=True))
    elif which == "smd":
        # SMD = IEF-PCM with SMD radii (590 points) + G_CDS, PySCF's conventions
        for m, meth, solv in itertools.product(["water", "nh3", "nh4+", "acetate", "ch3sh+"], ["hf", "b3lyp"], ["water", "tetrahydrofuran"]):
            out.append(dict(mol=m, method=meth, model="smd", solvent=solv, eps=0.0, df=False, grad=True))
        for meth in ["hf", "b3lyp"]:
            out.append(dict(mol="la18", method=meth, model="smd", solvent="tetrahydrofuran", eps=0.0, df=True, grad=True))
    return out


def run(case, slc, threads, linus):
    xyz, charge, mult = MOLS[case["mol"]]
    cmd = ["systemd-run", "--user", "--scope", "-q", "-p", "MemoryMax=24G", "-p", "MemorySwapMax=0",
           str(pathlib.Path.home() / ".local/bin/slice"), slc, PY, str(ROOT / "tests/python/compare_pcm_pyscf.py"),
           str(ROOT / xyz), case.get("basis", "def2-svp"), case["method"], "--model", case["model"], "--eps", str(case["eps"]),
           "--charge", str(charge), "--mult", str(mult), "--threads", str(threads), "--linus", linus,
           "--discretization", case.get("discretization", "swig")]
    if case.get("lebedev"):
        cmd += ["--lebedev", str(case["lebedev"])]
    if case.get("solvent"):
        cmd += ["--solvent", case["solvent"]]
    if case.get("df"):
        cmd.append("--df")
    if case.get("grad"):
        cmd.append("--grad")
    if case.get("hess"):
        cmd.append("--hess")
    p = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
    if p.returncode:
        return dict(case, error=p.stderr[-2000:])
    r = json.loads(p.stdout.strip().splitlines()[-1])
    r["mol"] = case["mol"]
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--set", default="energy-gradient")
    ap.add_argument("--slices", default="ccd2,ccd7")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--linus", default=str(ROOT / "target/release/linus"))
    ap.add_argument("--only", default="", help="comma-separated molecule names")
    ap.add_argument("--resume", action="store_true", help="skip the cases already in --out (without errors)")
    a = ap.parse_args()
    cs = cases(a.set)
    if a.only:
        keep = set(a.only.split(","))
        cs = [c for c in cs if c["mol"] in keep]
    slices = a.slices.split(",")
    outp = pathlib.Path(a.out)
    if a.resume and outp.exists():
        done = {(r["mol"], r["method"], r["model"], r["jk"], r.get("solvent") or float(r["eps"]))
                for r in map(json.loads, outp.read_text().splitlines()) if "error" not in r}
        cs = [c for c in cs if (c["mol"], c["method"], c["model"], "df" if c.get("df") else "exact",
                                c.get("solvent") or float(c["eps"])) not in done]
        print(len(cs), "cases left", flush=True)
    outp.parent.mkdir(parents=True, exist_ok=True)
    todo = queue.Queue()
    for c in cs:
        todo.put(c)
    lock = threading.Lock()
    keys = ["mol", "method", "model", "eps", "jk", "dE", "dE_solv", "max_abs_dG", "max_abs_dH", "solvent", "dE_cds", "error"]
    with outp.open("a") as f:
        def worker(slc):
            while True:
                try:
                    c = todo.get_nowait()
                except queue.Empty:
                    return
                r = run(c, slc, a.threads, a.linus)
                with lock:
                    f.write(json.dumps(r) + "\n")
                    f.flush()
                    print({k: r.get(k) for k in keys if k in r}, flush=True)
        ts = [threading.Thread(target=worker, args=(s,)) for s in slices]
        for t in ts:
            t.start()
        for t in ts:
            t.join()


if __name__ == "__main__":
    sys.exit(main())
