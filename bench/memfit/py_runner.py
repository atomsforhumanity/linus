"""py_runner.py REQ.json OUT.json FRONT: one linus calculation from Python.

FRONT: py (linus.compute in this process), qcc (qccompute.compute: the default runner, in this process),
qccw (a qccompute Runner with isolation "process": the calculation in a worker process). OUT.json gets
{"success", "results" (the output's resolved/extras), "error", "error_details", "peak_memory_gib"}.
"""
import json
import sys

req = json.load(open(sys.argv[1]))
front = sys.argv[3]
from qcdata import ProgramInput, Structure  # noqa: E402

st = Structure.from_xyz(req["structure"]["xyz"], charge=req["structure"]["charge"],
                        multiplicity=req["structure"]["multiplicity"])
kw = {k: v for k, v in req.items() if k not in ("structure", "method", "basis", "calctype")}
inp = ProgramInput(program="linus", structure=st, calctype=req["calctype"],
                   model={"method": req["method"], "basis": req["basis"]}, keywords=kw)


def dump(o):
    d = o.model_dump(mode="json")
    pr = d.get("provenance") or {}
    res = {"resolved": (d.get("results") or {}).get("resolved") if isinstance(d.get("results"), dict) else None}
    ex = d.get("extras") or {}
    return d, pr, res, ex


out = {}
if front == "plan":
    from linus.qc import program
    json.dump(program.plan(inp), open(sys.argv[2], "w"), default=str)
    sys.exit(0)
try:
    if front == "py":
        import linus
        o = linus.compute(inp)
    elif front == "qcc":
        import qccompute
        o = qccompute.compute(inp)
    else:
        import qccompute
        with qccompute.Runner(threads=req.get("threads", 1), gpus=(), isolation="process") as r:
            o = r.compute(inp)
    d = o.model_dump(mode="json")
    out = {"success": True, "output": d}
except Exception as e:  # noqa: BLE001
    po = getattr(e, "output", None)
    d = po.model_dump(mode="json") if po is not None and hasattr(po, "model_dump") else None
    det = None
    if d:
        det = (d.get("error") or {}).get("details")
    out = {"success": False, "error": str(e)[:3000], "error_details": det, "output": d}
json.dump(out, open(sys.argv[2], "w"), default=str)
