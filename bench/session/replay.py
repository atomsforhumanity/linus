#!/usr/bin/env python3
"""Replay one trajectory (trajectory.py) in one mode, timing every gradient (bench/session/README.md).

  replay.py SYSTEM PLATFORM MODE OUT.json       (run inside a job: run_all.py replay)

Modes (B-D and W through linus.Session on a qcdata ProgramInput, calctypes ["gradient"]):
  A  the `linus` CLI per geometry (an xyz file, --grad, --json): process start, basis parsing, GPU context
     and kernel-cache load included in every step;
  B  a new `linus.Session` per geometry in this process (opened, compute(x), closed);
  C  one Session, `reset()` before each `compute(x)` (SAD every step);
  D  one Session, `compute(x)` in its default lineage (warm start from the last orbitals, Löwdin-projected);
  W  one Session in a worker process (qccompute's serve loop: JSON lines over a pipe), `compute` per step.

A step's wall time runs from before the CLI's start / the Session's opening / `reset` / the message's write to
the CLI's exit / the Frame returned (B: and the Session closed) / the reply parsed. The first step of B, C, D and
W includes the first Session's opening (in that process: the GPU context, kernel modules). The CLI's xyz files
are written before the steps are timed; the worker's start and imports before its first step.
"""
from __future__ import annotations

import gc
import json
import os
import pathlib
import resource
import subprocess
import sys
import time

import common as C


def usage(who) -> dict:
    r = resource.getrusage(who)
    return dict(utime=r.ru_utime, stime=r.ru_stime, minflt=r.ru_minflt, majflt=r.ru_majflt)


def delta(a, b) -> dict:
    return {k: round(b[k] - a[k], 6) for k in a}


def summary(d: dict) -> dict:
    """What a step keeps of the CLI's JSON."""
    res = d.get("resolved") or {}
    scf = d.get("scf") or {}
    rs = res.get("scf") or {}
    be = res.get("backend") or {}
    return dict(
        energy=d["energy"], gradient=d["gradient"], iterations=d["iterations"],
        fock_builds=scf.get("fock_builds"), timings=d.get("timings"), resources=d.get("resources"),
        closed_shell_probe=rs.get("closed_shell_probe"), stability_probe=rs.get("stability_probe"),
        probe_trigger=rs.get("probe_trigger"), df=res.get("df"),
        backend=dict(device=(be.get("device") or {}).get("name"), placement={k: be.get(k) for k in (
            "jk", "xc", "gradient_jk", "gradient_xc", "scf_loop")}, fallbacks=be.get("fallbacks"),
            precision=be.get("precision")),
        memory=res.get("memory"), s2=d.get("s2"), homo_lumo_gap=d.get("homo_lumo_gap"),
    )


def mode_a(traj, s, plat, work) -> list[dict]:
    work.mkdir(parents=True, exist_ok=True)
    files = []
    for i, st in enumerate(traj["steps"]):
        f = work / f"step{i:02d}.xyz"
        C.write_xyz(f, traj["symbols"], st["positions"], f"{traj['system']} step {i}")
        files.append(f)
    steps = []
    for i, f in enumerate(files):
        cmd = [str(C.LINUS_BIN), str(f), "--method", C.METHOD, "--basis", s["basis"], "--jk", "df", "--grad",
               "--threads", str(C.THREADS), "--backend", C.backend(plat), "--charge", str(s["charge"]),
               "--mult", str(s["mult"]), "--json"]
        u0 = usage(resource.RUSAGE_CHILDREN)
        t0 = time.perf_counter()
        p = subprocess.run(cmd, capture_output=True, text=True)
        wall = time.perf_counter() - t0
        u1 = usage(resource.RUSAGE_CHILDREN)
        if p.returncode != 0:
            raise RuntimeError(f"step {i}: linus exit {p.returncode}: {p.stderr[-2000:]}")
        d = json.loads(p.stdout)
        rec = dict(step=i, wall=wall, rusage=delta(u0, u1), **summary(d))
        rec["positions_error_bohr"] = max(abs(a - b) for x, y in zip(d["structure"]["geometry"],
                                                                     traj["steps"][i]["positions"])
                                          for a, b in zip(x, y))
        steps.append(rec)
        C.log(f"A step {i:2d} {wall:7.3f} s (linus total {d['timings']['total']:.3f}) it {d['iterations']}")
    return steps


def frame_summary(f: dict) -> dict:
    """What a step keeps of a frame (a qcdata Frame as JSON: in process or from the worker)."""
    res, rec, exe = f["results"], f.get("record") or {}, f.get("execution") or {}
    scf = rec.get("scf") or {}
    be = rec.get("backend") or {}
    return dict(
        energy=res["energy"], gradient=res["gradient"], iterations=res["scf_iterations"],
        fock_builds=scf.get("fock_builds"), timings=exe.get("timings"), resources=rec.get("sizes"),
        closed_shell_probe=scf.get("closed_shell_probe"), stability_probe=scf.get("stability_probe"),
        probe_trigger=scf.get("probe_trigger"), df=rec.get("df"),
        backend=dict(device=(be.get("device") or {}).get("name"), placement={k: be.get(k) for k in (
            "jk", "xc", "gradient_jk", "gradient_xc", "scf_loop")}, fallbacks=be.get("fallbacks"),
            precision=be.get("precision")),
        memory=rec.get("memory"), s2=res.get("scf_s_squared"), homo_lumo_gap=rec.get("homo_lumo_gap"),
        warm_start=rec.get("warm_start"), guess=scf.get("guess"), peak_memory_gib=exe.get("peak_memory_gib"),
    )


def program_input(traj, s, plat):
    import qcdata

    structure = qcdata.Structure(symbols=traj["symbols"], geometry=traj["steps"][0]["positions"],
                                 charge=s["charge"], multiplicity=s["mult"])
    return qcdata.ProgramInput(program="linus", calctype="gradient", structure=structure,
                               model={"method": C.METHOD, "basis": s["basis"]},
                               keywords={"jk": "df", "threads": C.THREADS, "backend": C.backend(plat)})


def mode_session(traj, s, plat, mode) -> tuple[list[dict], dict]:
    import numpy as np

    t0 = time.perf_counter()
    import linus
    t_import = time.perf_counter() - t0
    inp = program_input(traj, s, plat)
    geoms = [np.array(st["positions"]) for st in traj["steps"]]
    session = None
    steps = []
    gc.disable()  # no collector pauses inside a timed step; collected between steps
    for i, x in enumerate(geoms):
        gc.collect()
        u0 = usage(resource.RUSAGE_SELF)
        t0 = time.perf_counter()
        if session is None or mode == "B":
            session = linus.Session(inp, calctypes=["gradient"])
        elif mode == "C":
            session.reset()
        t1 = time.perf_counter()
        f = session.compute(x)
        t2 = time.perf_counter()
        if mode == "B":
            session.close()  # its device memory, thread pool and DF storage freed
            session = None
        t3 = time.perf_counter()
        u1 = usage(resource.RUSAGE_SELF)
        total = f.execution.timings["total"]
        rec = dict(step=i, wall=t3 - t0, prepare=t1 - t0, call=t2 - t1, drop=t3 - t2, python=t2 - t1 - total,
                   rusage=delta(u0, u1), **frame_summary(f.model_dump(mode="json")))
        steps.append(rec)
        C.log(f"{mode} step {i:2d} {t3 - t0:7.3f} s (linus total {total:.3f}) it {f.results.scf_iterations}")
    gc.enable()
    if session is not None:
        session.close()
    here = pathlib.Path(linus.__file__).resolve()
    where = str(here.relative_to(C.ROOT)) if here.is_relative_to(C.ROOT) else here.name
    return steps, dict(import_linus_s=t_import, linus_file=where, linus_version=linus.__version__)


def proc_usage(pid: int) -> dict:
    """A process's CPU times and page faults so far (/proc/PID/stat; seconds)."""
    f = pathlib.Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    tick = os.sysconf("SC_CLK_TCK")
    return dict(utime=int(f[11]) / tick, stime=int(f[12]) / tick, minflt=int(f[7]), majflt=int(f[9]))


def mode_worker(traj, s, plat) -> tuple[list[dict], dict]:
    """One session in a worker process: the serve loop's open, then one compute per step; a step's wall runs
    from before its message is written to after its reply is parsed (the first step includes the open)."""
    inp = program_input(traj, s, plat).model_dump(mode="json")
    p = subprocess.Popen(C.serve_command(), stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    hello = json.loads(p.stdout.readline())

    def ask(msg: dict) -> dict:
        p.stdin.write(json.dumps(msg) + "\n")
        p.stdin.flush()
        reply = json.loads(p.stdout.readline())
        if not reply.get("ok"):
            raise RuntimeError(f"worker: {reply.get('error')}")
        return reply["result"]

    steps, sid = [], None
    for i, st in enumerate(traj["steps"]):
        u0 = proc_usage(p.pid)
        t0 = time.perf_counter()
        if sid is None:
            sid = ask({"id": "open", "op": "open", "input": inp, "calctypes": ["gradient"]})["session"]
        t1 = time.perf_counter()
        f = ask({"id": i, "op": "compute", "session": sid, "positions": st["positions"]})
        t2 = time.perf_counter()
        u1 = proc_usage(p.pid)
        if not f["success"]:
            raise RuntimeError(f"step {i}: {f['error']}")
        total = f["execution"]["timings"]["total"]
        rec = dict(step=i, wall=t2 - t0, prepare=t1 - t0, call=t2 - t1, drop=0.0, rusage=delta(u0, u1),
                   **frame_summary(f))
        steps.append(rec)
        C.log(f"W step {i:2d} {t2 - t0:7.3f} s (linus total {total:.3f}) it {f['results']['scf_iterations']}")
    p.stdin.close()
    rc = p.wait(timeout=120)
    return steps, dict(worker=dict(hello=hello, exit=rc, command=[pathlib.Path(c).name for c in C.serve_command()]))


def main(system, plat, mode, out) -> None:
    import pathlib

    traj = json.loads(C.traj_path(system).read_text())
    s = C.SYSTEMS[system]
    t0 = time.time()
    extra = {}
    if mode == "A":
        steps = mode_a(traj, s, plat, C.SCRATCH / "work" / f"{system}-{plat}")
    elif mode == "W":
        steps, extra = mode_worker(traj, s, plat)
    else:
        steps, extra = mode_session(traj, s, plat, mode)
    C.dump(pathlib.Path(out), dict(system=system, platform=plat, mode=mode, nsteps=len(steps),
                                   job_wall=time.time() - t0, started=t0, **extra, steps=steps))


if __name__ == "__main__":
    main(*sys.argv[1:5])
