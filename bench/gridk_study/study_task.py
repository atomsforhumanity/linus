#!/usr/bin/env python3
"""One task of compensated grid-K's accuracy study in a CPU pod (make_study.py): gridk_study for one species and basis.
Task JSON: {id, xyz, charge, mult, basis, method, grids, grad}. Writes $OUT_DIR/result.json
(the study's JSON plus the task's id, wall time and failures as records)."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

t = json.load(open(os.environ["TASK_JSON"]))
out = Path(os.environ["OUT_DIR"])
out.mkdir(parents=True, exist_ok=True)
run = Path(os.environ["RUN_DIR"])
xyz = Path.cwd() / "mol.xyz"
xyz.write_text(t["xyz"])
env = dict(os.environ, THREADS=os.environ.get("NTHREADS", "16"), GRIDS=t["grids"],
           GRAD="1" if t.get("grad") else "0", LINUS_BASIS_DIR=str(run / "src/data/basis"))
t0 = time.monotonic()
p = subprocess.run([str(run / "bin/gridk_study"), str(xyz), t["basis"], str(t["charge"]), str(t["mult"]),
                    t["method"]], env=env, capture_output=True, text=True, timeout=14000)
rec = {"id": t["id"], "species": t["species"], "basis": t["basis"], "method": t["method"],
       "natoms": t["natoms"], "wall_s": round(time.monotonic() - t0, 2), "rc": p.returncode}
if p.returncode == 0:
    try:
        rec["result"] = json.loads(p.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError) as e:
        rec["rc"], rec["error"] = -2, f"unparsable output: {e}"
if rec["rc"] != 0:
    rec.setdefault("error", p.stderr[-3000:])
(out / "result.json").write_text(json.dumps(rec))
print(json.dumps({k: v for k, v in rec.items() if k != "result"}))
sys.exit(0 if rec["rc"] == 0 else 1)
