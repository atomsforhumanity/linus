# Record of the 2026-11-07 study: runs on linus b36bf33 and earlier; its API (Calculator, QcoptEvaluator, propagate_wfn) is gone.
"""SCF guess reuse along a geometry optimization: SAD vs the previous step's wavefunction.

    taskset -c <8 cores> <venv>/bin/python bench/workflows/guess_reuse.py la18 [u1an39 ...]

1. geomeTRIC through qccompute with linus as the gradient child, from the perturbed start,
   with and without ``propagate_wfn=True`` (B3LYP-D3(BJ)/def2-SVP, RI-JK, 8 threads): SCF
   iterations and wall time per step, whole-run wall time.
2. Replay of run 1's geometries in one process, three ways on identical geometries: each SCF
   from SAD; from the previous step's density as it is (the driver's own reuse, what
   `linus.Calculator.set_positions` did before this measurement; the binding's raw
   `set_positions`); from the previous step's occupied orbitals made orthonormal at the new
   geometry (`Calculator.set_guess`, what propagate_wfn, `linus.evaluator.QcoptEvaluator` and now
   `Calculator.set_positions` use).

GUESS_REPLAY_STEPS=N replays only the first N geometries.

Writes bench/results/2026-11-07-workflows/guess_reuse_<mol>.json.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")  # numpy in geomeTRIC: small matrices (optimizer-quirks)
sys.path.insert(0, str(Path(__file__).resolve().parent))

import common as C  # noqa: E402
import numpy as np  # noqa: E402
from qccompute import compute  # noqa: E402
from qcdata import CalcType, ProgramInput, ProgramSpec  # noqa: E402

import linus  # noqa: E402

KEYWORDS = {"jk": "df", "threads": C.THREADS}


def gradient_spec() -> ProgramSpec:
    return ProgramSpec(program="linus", calctype=CalcType.gradient, model=C.SVP_LINUS, keywords=KEYWORDS)


def geometric_run(name: str, propagate: bool) -> dict:
    s = C.structure(name, perturbed=True)
    inp = ProgramInput(program="geometric", calctype=CalcType.optimization, structure=s,
                       keywords={"maxiter": 300, **C.GEOMETRIC_TIGHTOPT}, subprograms=[gradient_spec()])
    t = time.perf_counter()
    out = compute(inp, raise_exc=False, propagate_wfn=propagate)
    wall = time.perf_counter() - t
    frames = out.results.frames
    steps = []
    for f in frames:
        ex = f.results.extras or {}
        steps.append({"energy": f.results.energy, "scf_iterations": ex.get("scf_iterations"),
                      "scf_guess": ex.get("scf_guess"), "timings": ex.get("timings"),
                      "wall_s": f.execution.wall_time if f.execution else None})
    return {"success": out.success, "propagate_wfn": propagate, "n_steps": len(frames), "wall_s": wall,
            "final_energy": out.results.final_energy, "steps": steps,
            "geometries_bohr": [f.structure.geometry.tolist() for f in frames],
            "traceback": None if out.success else (out.traceback or "")[-2000:]}


def replay(name: str, geometries: list) -> dict:
    s = C.structure(name)
    args = dict(**C.SVP_LINUS_CALC, **KEYWORDS)
    out = {}
    for mode in ("sad", "density", "orbitals"):
        calc = linus.Calculator(s.model_copy(update={"geometry": np.asarray(geometries[0])}), **args)
        prev = None
        rows = []
        for k, x in enumerate(geometries):
            if mode == "density":  # the driver's reuse of the last density, no projection
                calc._calc.set_positions(np.ascontiguousarray(x, dtype=float))
            else:
                calc.set_positions(np.asarray(x))
            if mode == "sad" or k == 0:
                calc.reset_guess()
            elif mode == "orbitals":
                calc.reset_guess()
                calc.set_guess(prev)
            t = time.perf_counter()
            r = calc.gradient()
            rows.append({"iterations": r.iterations, "energy": r.energy, "wall_s": time.perf_counter() - t,
                         "timings": dict(r.timings)})
            prev = r
        out[mode] = rows
    return out


def summarize(rows: list[dict], skip_first: bool = True) -> dict:
    rs = rows[1:] if skip_first else rows
    it = [r["iterations"] if "iterations" in r else r["scf_iterations"] for r in rs]
    return {"steps": len(rs), "mean_iterations": float(np.mean(it)), "total_iterations": int(np.sum(it)),
            "total_wall_s": float(sum(r["wall_s"] for r in rs))}


def main() -> None:
    for name in sys.argv[1:] or ["la18"]:
        rec = {"molecule": name, "level": "linus b3lyp-d3bj/def2-svp RI-JK (def2-universal-jkfit), grid auto",
               "threads": C.THREADS, "affinity": C.affinity(), "ccd_busy_before": C.ccd_load(),
               "linus_version": linus.__version__}
        runs = {}
        for propagate in (False, True):
            r = geometric_run(name, propagate)
            runs["propagate" if propagate else "sad"] = r
            iters = [st["scf_iterations"] for st in r["steps"]]
            print(f"{name} geomeTRIC propagate_wfn={propagate}: {r['n_steps']} steps, {r['wall_s']:.1f} s, "
                  f"iterations {iters}", flush=True)
        rec["geometric"] = runs
        # the first GUESS_REPLAY_STEPS geometries (all by default; GUESS_REPLAY_STEPS bounds the cost)
        rep = replay(name, runs["sad"]["geometries_bohr"][: int(os.environ.get("GUESS_REPLAY_STEPS", "0")) or None])
        rec["replay"] = rep
        rec["replay_summary"] = {m: summarize(rows) for m, rows in rep.items()}
        for m, sm in rec["replay_summary"].items():
            print(f"{name} replay {m}: {sm}", flush=True)
        de = max(abs(a["energy"] - b["energy"]) for a, b in zip(rep["sad"], rep["orbitals"]))
        rec["replay_max_energy_difference"] = de
        rec["ccd_busy_after"] = C.ccd_load()
        C.save(f"guess_reuse_{name}", rec)


if __name__ == "__main__":
    main()
