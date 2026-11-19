#!/usr/bin/env python3
"""Response-tolerance sweep (decisions D28 and D35): errors of Hessians, dipole derivatives, IR
intensities and polarizabilities of `linus --hess --ir` against the CPHF/CPKS tolerances, each
against the run at the tightest tolerance.

    python3 bench/respcorr_sweep.py run --bin target/release/linus --out DIR [--cases ...]
        [--tols 1e-4,1e-5,1e-6,1e-7,1e-8,1e-9] [--ref 1e-10] [--fix-cphf 1e-6] [--prefix 'taskset -c 0-7']
    python3 bench/respcorr_sweep.py cmp --out DIR [--cases ...] [--tols ...]

`run` sets response.tol = response.field_tol = t (with --fix-cphf: response.tol fixed, only
response.field_tol = t; the reference run has both at --ref), B3LYP/def2-SVP, jk df,
precision.response fp64 (the residual's error alone; --default-precision keeps auto).
`cmp` prints per tolerance: the Hessian and polarizability CPHF iterations and final max-abs
residuals, max |dH|, max |d dmu/dX|, the IR intensities' max error (relative to the largest and
in km/mol), max |d alpha|, and the time of the Hessian's CPHF, of the polarizability's and of
the whole run. With LINUS_RESPCORR_EXP set, an experiment build (not on main) also wrote the
dipole-derivative variants of bench/results/2026-11-11-respcorr; `cmp --exp` reads them.
"""
import argparse, json, pathlib, shlex, subprocess, sys, time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
CASES = {
    # name: (xyz, charge, multiplicity, method)
    "water.b3lyp.rks": (ROOT / "tests/molecules/water.xyz", 0, 1, "b3lyp"),
    "water.b3lyp.uks": (ROOT / "tests/molecules/water.xyz", 1, 2, "b3lyp"),
    "water.hf.rks": (ROOT / "tests/molecules/water.xyz", 0, 1, "hf"),
    "la18.b3lyp.rks": (ROOT / "bench/molecules/la18.xyz", 0, 1, "b3lyp"),
    "la18.b3lyp.uks": (ROOT / "bench/molecules/la18.xyz", 1, 2, "b3lyp"),
    "u1an39.b3lyp.rks": (ROOT / "bench/molecules/u1an39.xyz", -1, 1, "b3lyp"),
}


def path(out, case, t):
    return pathlib.Path(out) / f"{case}.{t:.0e}.json"


def run(a):
    for c in a.cases.split(","):
        xyz, q, m, meth = CASES[c]
        for t in [float(x) for x in a.tols.split(",")] + [a.ref]:
            out = path(a.out, c, t)
            if out.exists():
                continue
            out.parent.mkdir(parents=True, exist_ok=True)
            cphf = a.fix_cphf if a.fix_cphf and t != a.ref else t
            cmd = shlex.split(a.prefix) + [a.bin, str(xyz), "--method", meth, "--basis", "def2-svp", "--jk", "df",
                                           "--charge", str(q), "--mult", str(m), "--threads", str(a.threads), "--json",
                                           "--hess", "--ir", "--set", f"response.tol={cphf:g}",
                                           "--set", f"response.field_tol={t:g}"]
            if not a.default_precision:
                cmd += ["--set", "precision.response=fp64"]
            t0 = time.perf_counter()
            p = subprocess.run(cmd, capture_output=True, text=True)
            try:
                d = json.loads(p.stdout)
            except json.JSONDecodeError:
                print(f"{c} {t:.0e}: exit {p.returncode}\n{p.stderr[-3000:]}", file=sys.stderr)
                continue
            d["_wall"] = time.perf_counter() - t0
            out.write_text(json.dumps(d))
            print(f"{c} t={t:.0e}: success={d.get('success', True)} wall {d['_wall']:.1f}s", flush=True)


def flat(x):
    return np.array(x, dtype=float).ravel()


def cmp(a):
    for c in a.cases.split(","):
        rp = path(a.out, c, a.ref)
        if not rp.exists():
            continue
        r = json.loads(rp.read_text())
        H0, D0, A0 = flat(r["hessian"]), flat(r["dipole_derivatives"]), flat(r["polarizability"])
        I0 = flat(r["vibrational_analysis"]["ir_intensities"])
        print(f"\n{c} (reference t={a.ref:.0e}: |H| {abs(H0).max():.2e}, |dmu| {abs(D0).max():.2e},"
              f" |alpha| {abs(A0).max():.2e}, Imax {I0.max():.2e} km/mol)")
        print(f"  {'tol':>6} {'it_H':>4} {'res_H':>8} {'it_a':>4} {'res_a':>8} {'dH':>8} {'ddmu':>8} {'dI(rel)':>8}"
              f" {'dI km/mol':>9} {'dalpha':>8} {'t_hcphf':>7} {'t_pol':>6} {'wall':>6}")
        for t in [float(x) for x in a.tols.split(",")] + [a.ref]:
            p = path(a.out, c, t)
            if not p.exists():
                continue
            d = json.loads(p.read_text())
            rr = d["resolved"]["response"]
            H, D, A = flat(d["hessian"]), flat(d["dipole_derivatives"]), flat(d["polarizability"])
            I = flat(d["vibrational_analysis"]["ir_intensities"])
            tm = d["timings"]
            th = sum(v for k, v in tm.items() if k.startswith("hess_cphf"))
            tp = sum(v for k, v in tm.items() if k.startswith("pol_cphf"))
            print(f"  {t:6.0e} {d['cphf_iterations']:4d} {rr['max_residual']:8.1e} {d['polarizability_cphf_iterations']:4d}"
                  f" {rr['pol_max_residual']:8.1e} {abs(H - H0).max():8.1e} {abs(D - D0).max():8.1e}"
                  f" {abs(I - I0).max() / I0.max():8.1e} {abs(I - I0).max():9.1e} {abs(A - A0).max():8.1e}"
                  f" {th:7.2f} {tp:6.2f} {d['_wall']:6.1f}")
            if a.exp:
                ep, er = p.with_suffix(".exp.json"), rp.with_suffix(".exp.json")
                if ep.exists() and er.exists():
                    e, eref = json.loads(ep.read_text()), json.loads(er.read_text())
                    print("         " + ", ".join(
                        f"{k} {abs(flat(v) - flat(eref[k])).max():.1e}" if isinstance(v, list)
                        else f"{k} {v:.3g}" if isinstance(v, float) else f"{k} {v}" for k, v in e.items()))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["run", "cmp"])
    ap.add_argument("--bin", default=str(ROOT / "target/release/linus"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--cases", default="water.b3lyp.rks,la18.b3lyp.rks,la18.b3lyp.uks")
    ap.add_argument("--tols", default="1e-4,1e-5,1e-6,1e-7,1e-8,1e-9")
    ap.add_argument("--ref", type=float, default=1e-10)
    ap.add_argument("--fix-cphf", type=float, default=None)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--prefix", default="", help="command prefix of every run (pinning, memory caps)")
    ap.add_argument("--default-precision", action="store_true")
    ap.add_argument("--exp", action="store_true")
    a = ap.parse_args()
    run(a) if a.what == "run" else cmp(a)


if __name__ == "__main__":
    main()
