#!/usr/bin/env python3
"""Python and pyo3 overhead per call on water (B3LYP/def2-SVP, jk df, CPU, 8 threads), at one geometry, warm:

  - the binding's call (`Session._engine.compute(...)`, a dict) against linus's own `timings.total`: the pyo3
    boundary and the report's conversion to Python objects (serde_json value -> dict, numpy arrays);
  - the package's call (`Session.compute(x)`, a `qcdata.Frame`) against the binding's: the Python wrapper (the
    request, the peak-memory probe, the frame's construction);
  - `linus.Session(...)` opening and `close()` (settings, basis set, thread pool; no device on the CPU).

  overhead.py OUT.json [N]           (run inside a job: run_all.py overhead)
"""
from __future__ import annotations

import gc
import statistics
import sys
import time

import numpy as np

import common as C


def main(out: str, n: int = 40) -> None:
    import json

    import linus
    import qcdata

    s = C.SYSTEMS["water-svp"]
    symbols, x = C.read_xyz(C.ROOT / s["xyz"])
    x = np.array(x)
    inp = qcdata.ProgramInput(program="linus", calctype="gradient",
                              structure=qcdata.Structure(symbols=symbols, geometry=x),
                              model={"method": C.METHOD, "basis": s["basis"]},
                              keywords={"jk": "df", "threads": C.THREADS, "backend": "cpu"})
    session = linus.Session(inp, calctypes=["gradient"])
    session.compute()
    request = json.dumps({"lineage": "null", "seed": [], "commit": True, "density": False, "fock": False})
    rows = {k: [] for k in ("raw_call", "raw_total", "pkg_call", "pkg_total", "construct", "drop")}
    gc.disable()
    for _ in range(n):
        gc.collect()
        t0 = time.perf_counter()
        d = session._engine.compute(x, "gradient", request)
        rows["raw_call"].append(time.perf_counter() - t0)
        rows["raw_total"].append(d["timings"]["total"])
        t0 = time.perf_counter()
        f = session.compute(x)
        rows["pkg_call"].append(time.perf_counter() - t0)
        rows["pkg_total"].append(f.execution.timings["total"])
        t0 = time.perf_counter()
        s2 = linus.Session(inp, calctypes=["gradient"])
        rows["construct"].append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        s2.close()
        rows["drop"].append(time.perf_counter() - t0)
    gc.enable()
    session.close()
    med = {k: statistics.median(v) for k, v in rows.items()}
    res = dict(
        n=n, median_s=med,
        binding_overhead_s=statistics.median(a - b for a, b in zip(rows["raw_call"], rows["raw_total"])),
        package_overhead_s=statistics.median(a - b for a, b in zip(rows["pkg_call"], rows["pkg_total"])),
        wrapper_s=med["pkg_call"] - med["raw_call"],
        raw=rows,
    )
    C.dump(out, res)
    C.log(f"overhead: binding {res['binding_overhead_s'] * 1e3:.2f} ms, package {res['package_overhead_s'] * 1e3:.2f} ms, "
          f"open {med['construct'] * 1e3:.2f} ms, close {med['drop'] * 1e3:.2f} ms, total {med['pkg_total'] * 1e3:.1f} ms")


if __name__ == "__main__":
    main(sys.argv[1], *(int(a) for a in sys.argv[2:3]))
