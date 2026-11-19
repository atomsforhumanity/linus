#!/usr/bin/env python3
"""Where a worker call's time goes, against the same call in process: timestamps on both sides of the pipe.

  pipe.py OUT.json [N] [--threads T]     (a timed job: CCD0 under the bench lock)
  pipe.py --worker STAMPS.json           (the instrumented worker, started by the above)

The case is SESSION2's: water, B3LYP/STO-3G gradients, 2 threads, N geometries 0.002 bohr apart (each SCF a few
ms), one session. Every stamp is CLOCK_MONOTONIC (one clock for every process of the machine), in microseconds:

- in process, per call: `Session.compute` entered, the binding's `compute` entered and returned (a proxy of the
  session's engine), the frame returned; and linus's own `execution.timings["total"]`;
- in process with a gap: the same with a sleep between calls as long as the worker's time between computes,
  so that what idling costs (threads parked, caches and clocks cooling) is measured apart from the pipe;
- through the worker: the client before its write and after its flush, after `readline` returned and after
  `json.loads`; the worker after its `readline` returned, after `json.loads`, after the Request was validated,
  around the binding's `compute` (the same proxy) and `Session.compute`, after the frame's `model_dump`, after
  `json.dumps`, and after its write and flush.

The worker runs qccompute's serve loop's `Host` (`qccompute.serve`) in a loop of its own that stamps each
stage, so the serve loop is measured as written, stage by stage.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time

import numpy as np

POS = np.array([[0.0, 0.0, 0.223], [0.038, 1.427, -0.890], [-0.019, -1.436, -0.879]])


def now() -> int:
    return time.clock_gettime_ns(time.CLOCK_MONOTONIC) // 1000


def program_input(threads: int):
    import qcdata

    return qcdata.ProgramInput(program="linus", calctype="gradient",
                               structure=qcdata.Structure(symbols=["O", "H", "H"], geometry=POS),
                               model={"method": "b3lyp", "basis": "sto-3g"}, keywords={"threads": threads})


class Engine:
    """The session's binding object, with its `compute` stamped (entered, returned)."""

    def __init__(self, inner, stamps: dict) -> None:
        object.__setattr__(self, "_inner", inner)
        object.__setattr__(self, "_stamps", stamps)

    def compute(self, *args, **kwargs):
        self._stamps["bind_in"] = now()
        try:
            return self._inner.compute(*args, **kwargs)
        finally:
            self._stamps["bind_out"] = now()

    def __getattr__(self, name):
        return getattr(self._inner, name)


def worker_main(stamps_path: str) -> None:
    """The serve loop's Host, its compute op unrolled into stamped stages; stdout for the protocol only."""
    from qccompute import serve

    proto = os.fdopen(os.dup(1), "w", buffering=1, encoding="utf-8")
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    from linus.qc import program

    worker = serve.Host(program)
    proto.write(serve._encode(worker.hello()) + "\n")
    proto.flush()
    rows = []
    for line in iter(sys.stdin.readline, ""):
        st = {"recv": now()}
        msg = json.loads(line)
        st["parsed"] = now()
        if msg["op"] != "compute":
            result = worker.handle(msg)
            if msg["op"] == "open":
                s = worker.sessions[result["session"]]
                s._engine = Engine(s._engine, st)
            proto.write(serve._encode({"id": msg["id"], "ok": True, "result": result}) + "\n")
            proto.flush()
            continue
        s = worker._session(msg)
        object.__setattr__(s._engine, "_stamps", st)
        r = serve._request({k: msg[k] for k in ("positions", "calctype", "lineage", "outputs", "seed", "commit")
                            if k in msg})
        st["validated"] = now()
        frame = s.compute(r.positions, r.calctype, lineage=r.lineage, outputs=r.outputs, seed=r.seed,
                          commit=r.commit)
        st["computed"] = now()
        d = serve._dump(frame)
        st["dumped"] = now()
        text = serve._encode({"id": msg["id"], "ok": True, "result": d})
        st["encoded"] = now()
        proto.write(text + "\n")
        proto.flush()
        st["written"] = now()
        st["id"], st["bytes"], st["linus_total_us"] = msg["id"], len(text), frame.execution.timings["total"] * 1e6
        rows.append(st)
    with open(stamps_path, "w") as f:
        json.dump(rows, f)


def in_process(threads: int, geoms, gap_s: float = 0.0) -> list[dict]:
    import linus

    rows = []
    with linus.Session(program_input(threads)) as s:
        st: dict = {}
        s._engine = Engine(s._engine, st)
        s.compute()
        for x in geoms:
            if gap_s:
                time.sleep(gap_s)
            st.clear()
            t0 = now()
            f = s.compute(x)
            t1 = now()
            rows.append(dict(call_in=t0, call_out=t1, bind_in=st["bind_in"], bind_out=st["bind_out"],
                             linus_total_us=f.execution.timings["total"] * 1e6))
    return rows


def through_worker(threads: int, geoms, stamps_path: str) -> tuple[list[dict], list[dict]]:
    p = subprocess.Popen([sys.executable, __file__, "--worker", stamps_path], stdin=subprocess.PIPE,
                         stdout=subprocess.PIPE, text=True)
    p.stdout.readline()

    def ask(msg: dict, row: dict | None = None) -> dict:
        line = json.dumps(msg) + "\n"
        if row is not None:
            row["send"] = now()
        p.stdin.write(line)
        p.stdin.flush()
        if row is not None:
            row["sent"] = now()
        text = p.stdout.readline()
        if row is not None:
            row["read"] = now()
        reply = json.loads(text)
        if row is not None:
            row["loaded"] = now()
        return reply

    sid = ask({"id": "open", "op": "open", "input": program_input(threads).model_dump(mode="json")})["result"]["session"]
    ask({"id": "first", "op": "compute", "session": sid})
    client = []
    for i, x in enumerate(geoms):
        row = {"id": i}
        ask({"id": i, "op": "compute", "session": sid, "positions": x.tolist()}, row)
        client.append(row)
    p.stdin.close()
    p.wait(timeout=60)
    with open(stamps_path) as f:
        worker = [r for r in json.load(f) if r["id"] != "first"]
    return client, worker


def med(xs) -> float:
    return statistics.median(xs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out", nargs="?")
    ap.add_argument("n", nargs="?", type=int, default=300)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--worker")
    a = ap.parse_args()
    if a.worker:
        worker_main(a.worker)
        return
    rng = np.random.default_rng(1)
    geoms = [POS + rng.normal(0, 0.002, POS.shape) for _ in range(a.n)]
    inproc = in_process(a.threads, geoms)
    client, worker = through_worker(a.threads, geoms, a.out + ".stamps.json")
    # the worker's time between the end of one compute and the start of the next, for the gapped run
    between = [w2["validated"] - w1["computed"] for w1, w2 in zip(worker, worker[1:])]
    gapped = in_process(a.threads, geoms, gap_s=med(between) / 1e6)

    def proc_rows(rows):
        return dict(
            call_us=med([r["call_out"] - r["call_in"] for r in rows]),
            python_before_binding_us=med([r["bind_in"] - r["call_in"] for r in rows]),
            binding_us=med([r["bind_out"] - r["bind_in"] for r in rows]),
            python_after_binding_us=med([r["call_out"] - r["bind_out"] for r in rows]),
            linus_total_us=med([r["linus_total_us"] for r in rows]),
            binding_beyond_linus_total_us=med([r["bind_out"] - r["bind_in"] - r["linus_total_us"] for r in rows]),
        )

    pairs = list(zip(client, worker))
    stages = dict(
        round_trip_us=med([c["loaded"] - c["send"] for c, _ in pairs]),
        client_write_flush_us=med([c["sent"] - c["send"] for c, _ in pairs]),
        pipe_to_worker_us=med([w["recv"] - c["sent"] for c, w in pairs]),
        worker_parse_us=med([w["parsed"] - w["recv"] for _, w in pairs]),
        worker_request_validate_us=med([w["validated"] - w["parsed"] for _, w in pairs]),
        session_compute_us=med([w["computed"] - w["validated"] for _, w in pairs]),
        of_it_python_before_binding_us=med([w["bind_in"] - w["validated"] for _, w in pairs]),
        of_it_binding_us=med([w["bind_out"] - w["bind_in"] for _, w in pairs]),
        of_it_python_after_binding_us=med([w["computed"] - w["bind_out"] for _, w in pairs]),
        linus_total_us=med([w["linus_total_us"] for _, w in pairs]),
        worker_model_dump_us=med([w["dumped"] - w["computed"] for _, w in pairs]),
        worker_json_dumps_us=med([w["encoded"] - w["dumped"] for _, w in pairs]),
        worker_write_flush_us=med([w["written"] - w["encoded"] for _, w in pairs]),
        pipe_to_client_us=med([c["read"] - w["written"] for c, w in pairs]),
        client_json_loads_us=med([c["loaded"] - c["read"] for c, _ in pairs]),
        worker_between_computes_us=med(between),
        reply_bytes=med([w["bytes"] for _, w in pairs]),
    )
    out = dict(case="water B3LYP/STO-3G gradient", threads=a.threads, calls=a.n, in_process=proc_rows(inproc),
               in_process_with_gap=dict(gap_us=med(between), **proc_rows(gapped)), worker=stages)
    out["worker_minus_in_process_us"] = stages["round_trip_us"] - out["in_process"]["call_us"]
    out["binding_worker_minus_in_process_us"] = stages["of_it_binding_us"] - out["in_process"]["binding_us"]
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
