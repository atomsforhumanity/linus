"""Accuracy studies of linus DFT Hessians: grids and the RI approximation (B3LYP/def2-SVP).

HISTORICAL: the RI study hooks this script sets (LINUS_HESS_EXPL_JK, LINUS_HESS_RESP_JK,
LINUS_HESS_TWOLEVEL) were removed from linus after the study (results in
bench/results/2026-11-07-hess-accuracy); the `ri` runs no longer do what they did. The `grid`
runs still work.

Each calculation is one JSON request to examples/grid_study (any grid via a full GridSpec,
the RI study hooks via environment variables), run under a memory-capped systemd scope on a
CPU slice and cached as RUNS/<mol>.<tag>.json. Comparisons against a reference run: harmonic
frequencies (isotopic masses, translations and rotations projected; cm^-1: max, rms, lowest
six), max |gradient| difference, E_xc difference, max |Hessian| difference.

  python study.py grid  MOL[,MOL] TAG[,TAG|all]     run grid variants (grids.VARIANTS), RI-JK
  python study.py ri    MOL[,MOL] TAG[,TAG|all]     run RI variants (RI_VARIANTS) on PySCF level 5
  python study.py table TAG[,TAG] [MOL=REF ...]     grid table (default refs: la18 ri.pyscf7,
                                                    la17r and u1an39 ri.pyscf5)
  python study.py ritable MOL TAG[,TAG]             RI table against the exact-J/K run rc.pyscf5.ex

Environment: GS_BIN (grid_study binary; default target/release/examples/grid_study), RUNS
(cache directory; default ./runs), CPUS (slice name or cpus:LIST, default ccd2), MEM (GB, 14).
"""
import json, os, pathlib, subprocess, sys, time
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
BIN = pathlib.Path(os.environ.get("GS_BIN", ROOT / "target/release/examples/grid_study"))
RUNS = pathlib.Path(os.environ.get("RUNS", "runs"))
SLICE = os.path.expanduser("~/.local/bin/slice")
MOLS = {
    "la18": (ROOT / "bench/molecules/la18.xyz", 0, 1),
    "la17r": (ROOT / "bench/molecules/la17r.xyz", 0, 2),
    "u1an39": (ROOT / "bench/molecules/u1an39.xyz", -1, 1),
}
BOHR = 0.529177210544
MASS = {"H": 1.00782503223, "C": 12.0, "N": 14.00307400443, "O": 15.99491461957, "F": 18.99840316207, "S": 31.9720711744}
AMU_ME = 1822.888486209
HARTREE_CM = 219474.6313705
U = "def2-universal-jkfit"


def base_opts(**kw):
    o = {"method": "b3lyp", "basis": "def2-svp", "jk": {"kind": "df", "aux": U}, "threads": 8,
         "grid": {"kind": "pyscf", "level": 5}, "response": {"grid_response": True}, "grid_response": True}
    o.update(kw)
    return o


def run(mol, tag, opts, calc="hessian", env=None, force=False):
    RUNS.mkdir(exist_ok=True)
    f = RUNS / f"{mol}.{tag}.json"
    if f.exists() and not force:
        return json.loads(f.read_text())
    xyz, charge, mult = MOLS[mol]
    req = {"xyz": str(xyz), "charge": charge, "mult": mult, "calc": calc, "opts": opts}
    cmd = ["systemd-run", "--user", "--scope", "-q", "-p", f"MemoryMax={os.environ.get('MEM', '14')}G", "-p", "MemorySwapMax=0"]
    cpus = os.environ.get("CPUS", "ccd2")
    cmd += ["taskset", "-c", cpus[5:]] if cpus.startswith("cpus:") else [] if cpus == "none" else [SLICE, cpus]
    p = subprocess.run(cmd + [str(BIN)], input=json.dumps(req), capture_output=True, text=True, env=dict(os.environ, **(env or {})))
    if p.returncode != 0:
        raise RuntimeError(f"{mol} {tag}: {p.stderr[-2000:]}")
    d = json.loads(p.stdout)
    d.update(tag=tag, mol=mol, opts=opts, env=env or {}, date=time.strftime("%Y-%m-%dT%H:%M:%S"))
    if calc != "count":
        f.write_text(json.dumps(d))
    return d


def frequencies(h, mol):
    lines = pathlib.Path(MOLS[mol][0]).read_text().splitlines()
    at = [(l.split()[0], [float(x) for x in l.split()[1:4]]) for l in lines[2:2 + int(lines[0])]]
    n = len(at)
    h = np.asarray(h).reshape(3 * n, 3 * n)
    h = 0.5 * (h + h.T)
    m = np.array([MASS[a] for a, _ in at]) * AMU_ME
    x = np.array([c for _, c in at]) / BOHR
    x = x - (m[:, None] * x).sum(0) / m.sum()
    sm = np.sqrt(np.repeat(m, 3))
    vecs = []
    for k in range(3):
        v = np.zeros((n, 3)); v[:, k] = 1.0
        vecs.append((v * np.sqrt(m)[:, None]).ravel())
        e = np.zeros(3); e[k] = 1.0
        vecs.append((np.cross(e, x) * np.sqrt(m)[:, None]).ravel())
    q, s, _ = np.linalg.svd(np.array(vecs).T, full_matrices=False)
    q = q[:, s > 1e-6 * s.max()]
    p = np.eye(3 * n) - q @ q.T
    w = np.linalg.eigvalsh(p @ (h / np.outer(sm, sm)) @ p)
    w = w[np.argsort(np.abs(w))][q.shape[1]:]
    return np.sort(np.sign(w) * np.sqrt(np.abs(w)) * HARTREE_CM)


def compare(d, ref, nlow=6):
    out = {"npoints": d.get("npoints"), "wall": d.get("wall"), "cphf": d.get("cphf_iterations"),
           "dexc": d["e_xc"] - ref["e_xc"] if ref.get("e_xc") is not None else None, "de": d["energy"] - ref["energy"]}
    if "gradient" in d and "gradient" in ref:
        out["dg"] = float(np.abs(np.array(d["gradient"]) - np.array(ref["gradient"])).max())
    f, fr = frequencies(d["hessian"], d["mol"]), frequencies(ref["hessian"], ref["mol"])
    df = f - fr
    i = int(np.abs(df).argmax())
    out.update(dnu_max=float(np.abs(df).max()), dnu_rms=float(np.sqrt((df ** 2).mean())), dnu_low=float(np.abs(df[:nlow]).max()),
               nu_at_max=float(fr[i]), dh=float(np.abs(np.array(d["hessian"]) - np.array(ref["hessian"])).max()))
    return out


RI_VARIANTS = {
    # tag: (SCF/default J/K, study hooks): LINUS_HESS_EXPL_JK (explicit terms and F^x; also the
    # response J/K unless LINUS_HESS_RESP_JK is set), LINUS_HESS_RESP_JK (CPKS iterations),
    # LINUS_HESS_TWOLEVEL (right-hand sides and the final response build with the explicit-term
    # J/K: error quadratic in the iteration operator's error)
    "ex": ({"kind": "exact"}, {}),
    "ri": ({"kind": "df", "aux": U}, {}),
    "ri_tz": ({"kind": "df", "aux": "cc-pvtz-jkfit"}, {}),
    "ri_qz": ({"kind": "df", "aux": "cc-pvqz-jkfit"}, {}),
    "ex_respri": ({"kind": "exact"}, {"LINUS_HESS_RESP_JK": f"df:{U}"}),
    "ex_explri_respex": ({"kind": "exact"}, {"LINUS_HESS_EXPL_JK": f"df:{U}", "LINUS_HESS_RESP_JK": "exact"}),
    "ri_respex": ({"kind": "df", "aux": U}, {"LINUS_HESS_RESP_JK": "exact"}),
    "ri_explex_respri": ({"kind": "df", "aux": U}, {"LINUS_HESS_EXPL_JK": "exact", "LINUS_HESS_RESP_JK": f"df:{U}"}),
    "ri_allex": ({"kind": "df", "aux": U}, {"LINUS_HESS_EXPL_JK": "exact", "LINUS_HESS_RESP_JK": "exact"}),
    "ri_respexk": ({"kind": "df", "aux": U}, {"LINUS_HESS_RESP_JK": f"rij-exactk:{U}"}),
    "ri_respcosx3": ({"kind": "df", "aux": U}, {"LINUS_HESS_RESP_JK": f"rij-cosx:{U}:3"}),
    "ri_resp2l_ex": ({"kind": "df", "aux": U}, {"LINUS_HESS_EXPL_JK": "exact", "LINUS_HESS_RESP_JK": f"df:{U}", "LINUS_HESS_TWOLEVEL": "1"}),
    "ri_resp2l_qz": ({"kind": "df", "aux": U}, {"LINUS_HESS_EXPL_JK": "df:cc-pvqz-jkfit", "LINUS_HESS_RESP_JK": f"df:{U}", "LINUS_HESS_TWOLEVEL": "1"}),
    "ri_explqz": ({"kind": "df", "aux": U}, {"LINUS_HESS_EXPL_JK": "df:cc-pvqz-jkfit"}),
    "ri_tz_resp2l": ({"kind": "df", "aux": "cc-pvtz-jkfit"}, {"LINUS_HESS_RESP_JK": f"df:{U}", "LINUS_HESS_TWOLEVEL": "1"}),
    "ri_tz_respu": ({"kind": "df", "aux": "cc-pvtz-jkfit"}, {"LINUS_HESS_RESP_JK": f"df:{U}"}),
    "ri_respqz": ({"kind": "df", "aux": U}, {"LINUS_HESS_RESP_JK": "df:cc-pvqz-jkfit"}),
}


def fmt(name, c):
    s = f"{name:20s} {c.get('npoints') or 0:8d} "
    if c.get("dexc") is not None:
        s += f"dExc {c['dexc']:+.1e} "
    if "dg" in c:
        s += f"dG {c['dg']:.1e} "
    s += f"dnu max {c['dnu_max']:6.3f} rms {c['dnu_rms']:6.3f} low6 {c['dnu_low']:6.3f} (at {c['nu_at_max']:7.1f}) dH {c['dh']:.1e}"
    if c.get("wall") is not None:
        s += f" {c['wall']:6.1f}s"
    return s


def main():
    from grids import VARIANTS
    what = sys.argv[1]
    if what in ("grid", "ri"):
        mols, tags = sys.argv[2].split(","), sys.argv[3].split(",")
        table = VARIANTS if what == "grid" else RI_VARIANTS
        tags = list(table) if tags == ["all"] else tags
        for m in mols:
            for t in tags:
                if what == "grid":
                    d = run(m, f"ri.{t}", base_opts(grid=table[t]))
                else:
                    jk, env = table[t]
                    d = run(m, f"rc.pyscf5.{t}", base_opts(jk=jk), env=env)
                print(m, t, d.get("npoints"), f"{d['wall']:.1f}s", flush=True)
    elif what == "table":
        tags = sys.argv[2].split(",")
        refs = {"la18": "ri.pyscf7", "la17r": "ri.pyscf5", "u1an39": "ri.pyscf5"}
        refs.update(kv.split("=") for kv in sys.argv[3:])
        mols = [m for m in MOLS if (RUNS / f"{m}.{refs[m]}.json").exists()]
        R = {m: json.loads((RUNS / f"{m}.{refs[m]}.json").read_text()) for m in mols}
        for t in tags:
            worst, cells = 0.0, []
            for m in mols:
                f = RUNS / f"{m}.ri.{t}.json"
                if f.exists():
                    c = compare(json.loads(f.read_text()), R[m])
                    worst = max(worst, c["dnu_max"])
                    cells.append(f"{m} {c['npoints'] / 1000:5.0f}k max {c['dnu_max']:5.2f} rms {c['dnu_rms']:5.2f}")
            print(f"{t:12s} " + " | ".join(cells) + f" | worst {worst:5.2f}")
    elif what == "ritable":
        m, tags = sys.argv[2], sys.argv[3].split(",")
        ref = json.loads((RUNS / f"{m}.rc.pyscf5.ex.json").read_text())
        for t in tags:
            f = RUNS / f"{m}.rc.pyscf5.{t}.json"
            if f.exists():
                print(m, fmt(t, compare(json.loads(f.read_text()), ref)))


if __name__ == "__main__":
    main()
