# Record of the 2026-11-07 study: runs on linus b36bf33 and earlier; its API (Calculator, QcoptEvaluator, propagate_wfn) is gone.
"""Repro: the RI (jk=df) Hessian's exchange cross terms run on 3 threads once the stored block
of LINUS_DF_HESS_CROSS_MB (default 3000 MB) does not hold every atom.

HISTORICAL: DfDerivJk's in-memory Hessian code and LINUS_DF_HESS_CROSS_MB were removed from
linus (crate::hess_df replaced them); kept as the record of the issue.

    taskset -c <8 cores> python bench/workflows/repros/df_hessian_cross_block_threads.py [threads]

Seen on ts61 (61 atoms, B3LYP-D3(BJ)/def2-SVP, nbf 666, nocc 153, aux cc-pVTZ-jkfit|
def2-universal-jkfit, 8 threads): the packed R^x rows take ~1 GB per atom (3 x naux x
nocc(nocc+1)/2 doubles), so `cross_block_atoms` stores 2-3 atoms, and for most of the Hessian
only 3 of 8 threads were busy (top -H), in DfDerivJk::rhf_hessian_2e (perf: faer GEMM
microkernels ~50%, closure #5). Cause, jk_df_deriv.rs `cross_gram`: every later atom is
recomputed one at a time (`recompute(c, c + 1)`, ~natom^2 / (2 blk) times: ~600 for ts61), and
its L^-1 solve `linv` is `ws.par_chunks_mut(r).for_each(|x| self.solve_l(x, r / naux, Par::Seq))`:
3 chunks (x, y, z) for one atom, so 3 tasks, each a sequential triangular solve over
naux x nocc(nocc+1)/2. Fix (RI Hessian owner): split each chunk's columns across threads (or
solve_l with Par::rayon when there are fewer chunks than threads), and recompute several atoms
per call.

On la18 (18 atoms, ~25 MB per atom) the default budget stores every atom; 60 MB forces blocks of
~2 atoms and the recompute path. Each run is a subprocess; CPU/wall shows how many cores
worked. Measured 2026-11-07 (4 threads, loaded node): default 39.6 s (CPU/wall 2.8), 60 MB
48.2 s (2.7): la18's recompute is small next to its other phases, so the starvation shows on
ts61-sized systems, where it dominates.
"""

import os
import resource
import subprocess
import sys
import time
from pathlib import Path

CHILD = """
import sys, numpy as np, linus
sys.path.insert(0, %r)
import common as C
s = C.structure("la18")
r = linus.Calculator(s, method="b3lyp", dispersion="d3bj", basis="def2-svp", jk="df", threads=%d).hessian()
print(r.energy, r.timings.get("total"))
"""

threads = int(sys.argv[1]) if len(sys.argv) > 1 else 8
wf = str(Path(__file__).resolve().parents[1])
for budget in (None, "60"):
    env = dict(os.environ)
    if budget:
        env["LINUS_DF_HESS_CROSS_MB"] = budget
    else:
        env.pop("LINUS_DF_HESS_CROSS_MB", None)
    r0 = resource.getrusage(resource.RUSAGE_CHILDREN)
    t = time.perf_counter()
    out = subprocess.run([sys.executable, "-c", CHILD % (wf, threads)], env=env, capture_output=True, text=True)
    wall = time.perf_counter() - t
    r1 = resource.getrusage(resource.RUSAGE_CHILDREN)
    cpu = (r1.ru_utime - r0.ru_utime) + (r1.ru_stime - r0.ru_stime)
    print(f"LINUS_DF_HESS_CROSS_MB={budget or 'default (3000)'}: wall {wall:.1f} s, CPU {cpu:.1f} s, "
          f"CPU/wall {cpu / wall:.1f} of {threads}; {out.stdout.strip() or out.stderr[-300:]}", flush=True)
