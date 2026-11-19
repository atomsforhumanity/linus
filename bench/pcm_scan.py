"""Smoothness of PCM energies and gradients along a bond stretch and a torsion (linus CLI).

For each scan point the energy and the analytic gradient are computed; the script reports
- the largest jump in the energy's second difference (a discontinuity in E or dE/ds shows up as
  an outlier: compared with the median of its neighbours),
- how well the analytic gradient integrates to the energy: E(s) - E(s0) against the cumulative
  Simpson integral of dE/ds = g . dR/ds (max deviation over the scan),
- the number of surface points at each step (points switching on and off).

  .refenv/bin/python bench/pcm_scan.py --out bench/results/2026-11-07-pcm/scan_ch3sh.json \
      [--mol tests/molecules/ch3sh.xyz] [--method hf] [--basis def2-svp] [--model cpcm] \
      [--discretization swig] [--preset pyscf] [--threads 8]
"""
import argparse, json, pathlib, subprocess

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
BOHR = 0.529177210544


def read_xyz(path):
    lines = pathlib.Path(path).read_text().splitlines()
    n = int(lines[0])
    syms = [ln.split()[0] for ln in lines[2:2 + n]]
    pos = np.array([[float(v) / BOHR for v in ln.split()[1:4]] for ln in lines[2:2 + n]])
    return syms, pos


def rotate(pos, atoms, a, b, angle):
    """Rotate `atoms` about the axis through atoms a -> b by `angle` (rad)."""
    k = pos[b] - pos[a]
    k /= np.linalg.norm(k)
    out = pos.copy()
    for i in atoms:
        v = pos[i] - pos[b]
        out[i] = pos[b] + v * np.cos(angle) + np.cross(k, v) * np.sin(angle) + k * np.dot(k, v) * (1 - np.cos(angle))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--mol", default=str(ROOT / "tests/molecules/ch3sh.xyz"))
    ap.add_argument("--method", default="hf")
    ap.add_argument("--basis", default="def2-svp")
    ap.add_argument("--model", default="cpcm")
    ap.add_argument("--eps", type=float, default=7.4257)
    ap.add_argument("--preset", default="pyscf")
    ap.add_argument("--discretization", default=None)
    ap.add_argument("--lebedev", type=int, default=None)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--bond", default="0,1", help="atoms a,b: b and the atoms beyond it move along a->b")
    ap.add_argument("--moving", default="1,2", help="atoms moved by the stretch")
    ap.add_argument("--bond-range", default="-0.25,0.25,0.005", help="displacement start,stop,step (Å)")
    ap.add_argument("--torsion", default="1,0", help="axis atoms a,b of the torsion")
    ap.add_argument("--rotating", default="3,4,5", help="atoms rotated about the axis")
    ap.add_argument("--torsion-range", default="0,120,1.5", help="angle start,stop,step (deg)")
    ap.add_argument("--bin", default=str(ROOT / "target/release/linus"))
    a = ap.parse_args()
    syms, pos0 = read_xyz(a.mol)
    solvent = {"eps": a.eps, "model": a.model, "preset": a.preset}
    if a.discretization:
        solvent["discretization"] = a.discretization
    if a.lebedev:
        solvent["lebedev"] = a.lebedev

    def run(p):
        req = {"structure": {"symbols": syms, "geometry": p.tolist()}, "method": a.method, "basis": a.basis, "threads": a.threads,
               "calctype": "gradient", "scf": {"convergence": "extreme"}, "solvent": solvent}
        r = json.loads(subprocess.run([a.bin, "--json"], input=json.dumps(req), capture_output=True, text=True, check=True).stdout)
        return r["energy"], np.array(r["gradient"]), r["solvation"]["npoints"], r["solvation"]["electrostatic"]

    def scan(kind, values, geom):
        es, ds, npts, esol = [], [], [], []
        for v in values:
            p, dp = geom(v)
            e, g, n, es_ = run(p)
            es.append(e)
            ds.append(float(np.sum(g * dp)))  # dE/ds
            npts.append(n)
            esol.append(es_)
        es, ds = np.array(es), np.array(ds)
        h = values[1] - values[0]
        # cumulative Simpson (pairs of steps) of dE/ds against E differences
        dev = 0.0
        for k in range(2, len(values), 2):
            integ = sum(h / 3 * (ds[j] + 4 * ds[j + 1] + ds[j + 2]) for j in range(0, k, 2))
            dev = max(dev, abs(integ - (es[k] - es[0])))
        d2 = np.diff(es, 2)
        d3 = np.abs(np.diff(d2))
        jump = float(d3.max() / max(np.median(d3), 1e-300))
        return {"kind": kind, "values": list(values), "energy": es.tolist(), "dE_ds": ds.tolist(), "npoints": npts,
                "solvation_electrostatic": esol, "max_dev_integrated_gradient": dev,
                "max_abs_third_difference": float(d3.max()), "median_abs_third_difference": float(np.median(d3)),
                "jump_ratio": jump, "step": h}

    ia, ib = (int(x) for x in a.bond.split(","))
    moving = [int(x) for x in a.moving.split(",")]
    lo, hi, st = (float(x) for x in a.bond_range.split(","))
    u = (pos0[ib] - pos0[ia]) / np.linalg.norm(pos0[ib] - pos0[ia])

    def stretch(d):  # d in bohr
        p = pos0.copy()
        dp = np.zeros_like(p)
        for i in moving:
            p[i] += d * u
            dp[i] = u
        return p, dp

    vals = np.arange(lo / BOHR, hi / BOHR + 1e-12, st / BOHR)
    if len(vals) % 2 == 0:
        vals = vals[:-1]
    res = {"settings": {k: v for k, v in vars(a).items() if k != "out"}, "bond": scan("bond (bohr)", vals, stretch)}
    ta, tb = (int(x) for x in a.torsion.split(","))
    rot = [int(x) for x in a.rotating.split(",")]
    lo, hi, st = (float(x) for x in a.torsion_range.split(","))

    def torsion(t):  # t in rad; dR/dt by central difference of the rotation
        p = rotate(pos0, rot, ta, tb, t)
        eps = 1e-6
        dp = (rotate(pos0, rot, ta, tb, t + eps) - rotate(pos0, rot, ta, tb, t - eps)) / (2 * eps)
        return p, dp

    vals = np.radians(np.arange(lo, hi + 1e-9, st))
    if len(vals) % 2 == 0:
        vals = vals[:-1]
    res["torsion"] = scan("torsion (rad)", vals, torsion)
    pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(a.out).write_text(json.dumps(res, indent=1))
    for k in ("bond", "torsion"):
        r = res[k]
        print(k, {x: r[x] for x in ("max_dev_integrated_gradient", "max_abs_third_difference", "median_abs_third_difference", "jump_ratio")},
              "npoints", min(r["npoints"]), "-", max(r["npoints"]))


if __name__ == "__main__":
    main()
