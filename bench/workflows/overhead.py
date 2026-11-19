# Record of the 2026-11-07 study: runs on linus b36bf33 and earlier; its API (Calculator, QcoptEvaluator, propagate_wfn) is gone.
"""Per-call overhead of the Python and qc* layers around one linus gradient.

    taskset -c <8 cores> <venv>/bin/python bench/workflows/overhead.py la18 [repeats]

The same gradient (B3LYP-D3(BJ)/def2-SVP, RI-JK, 8 threads, SAD guess every time, so the SCF does
the same work) five ways, interleaved: a reused `linus.Calculator` (set_positions, reset_guess);
a new `linus.Calculator` per call; `linus.compute(ProgramInput)`; `qccompute.compute(ProgramInput)`;
the same with ``collect_wfn=True`` (what an optimizer step with propagate_wfn costs). Writes
bench/results/2026-11-07-workflows/overhead_<mol>.json.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import common as C  # noqa: E402
import numpy as np  # noqa: E402
from qccompute import compute  # noqa: E402
from qcdata import CalcType, ProgramInput  # noqa: E402

import linus  # noqa: E402


def main() -> None:
    name = sys.argv[1] if len(sys.argv) > 1 else "la18"
    repeats = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    s = C.structure(name)
    kw = {"jk": "df", "threads": C.THREADS}
    inp = ProgramInput(program="linus", calctype=CalcType.gradient, structure=s, model=C.SVP_LINUS, keywords=kw)
    args = dict(**C.SVP_LINUS_CALC, **kw)
    reused = linus.Calculator(s, **args)
    reused.gradient()  # warm up (basis files, thread pool, libxc)
    rows: dict[str, list[dict]] = {k: [] for k in ("reused", "new_calculator", "linus_compute", "qccompute",
                                                   "qccompute_collect_wfn")}
    construct = []
    for _ in range(repeats):
        t = time.perf_counter()
        reused.set_positions(s.geometry)
        reused.reset_guess()
        r = reused.gradient()
        rows["reused"].append({"wall_s": time.perf_counter() - t, "linus_total_s": r.timings["total"],
                               "iterations": r.iterations})

        t = time.perf_counter()
        calc = linus.Calculator(s, **args)
        construct.append(time.perf_counter() - t)
        r = calc.gradient()
        rows["new_calculator"].append({"wall_s": time.perf_counter() - t, "linus_total_s": r.timings["total"],
                                       "iterations": r.iterations})

        for key, fn in (("linus_compute", lambda: linus.compute(inp)),
                        ("qccompute", lambda: compute(inp)),
                        ("qccompute_collect_wfn", lambda: compute(inp, collect_wfn=True))):
            t = time.perf_counter()
            out = fn()
            wall = time.perf_counter() - t
            ex = out.results.extras
            rows[key].append({"wall_s": wall, "linus_total_s": ex["timings"]["total"],
                              "iterations": ex["scf_iterations"],
                              "execution_wall_s": out.execution.wall_time if out.execution else None})
    summary = {}
    for k, rs in rows.items():
        w = np.median([x["wall_s"] for x in rs])
        lt = np.median([x["linus_total_s"] for x in rs])
        summary[k] = {"median_wall_s": float(w), "median_linus_total_s": float(lt), "median_overhead_s": float(w - lt)}
        print(f"{name} {k:24s} wall {w:7.3f} s  linus total {lt:7.3f} s  outside {w - lt:6.3f} s", flush=True)
    rec = {"molecule": name, "repeats": repeats, "threads": C.THREADS, "affinity": C.affinity(),
           "ccd_busy": C.ccd_load(), "construct_s": construct, "rows": rows, "summary": summary,
           "linus_version": linus.__version__}
    print(f"{name} Calculator() construction: median {np.median(construct):.3f} s", flush=True)
    C.save(f"overhead_{name}", rec)


if __name__ == "__main__":
    main()
