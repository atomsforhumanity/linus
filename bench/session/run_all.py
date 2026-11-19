#!/usr/bin/env python3
"""Driver of the session benchmark: trajectories, then interleaved replays (bench/session/README.md).

  run_all.py traj SYSTEM [SYSTEM ...] [--platform gpu]
      generate each system's trajectory (trajectory.py) as one locked job
  run_all.py replay [--systems S ...] [--platforms cpu gpu] [--repeats 0 1 ...]
      for each repeat r, each system and platform, the modes (A-D, W; --modes) in rotating order (repeat r starts at
      mode r mod the number of modes),
      each replay one locked job writing results/raw/<system>_<platform>_<mode>_r<r>.json; existing files are kept
      (rerun to resume)

Jobs: see common.py (bench lock; the exclusive GPU lock for GPU jobs; a systemd scope with the system's memory cap;
CCD0's physical cores). Each job's lock waits, wall time and the busy threads per CCD before it are appended to
results/jobs.jsonl.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time

import common as C


def record_job(kind, system, plat, mode, rep, res, load) -> None:
    C.RESULTS.mkdir(parents=True, exist_ok=True)
    with open(C.RESULTS / "jobs.jsonl", "a") as f:
        f.write(json.dumps(dict(kind=kind, system=system, platform=plat, mode=mode, repeat=rep,
                                at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), ccd_busy_before=load,
                                **res)) + "\n")


def traj(systems, plat) -> None:
    for system in systems:
        if C.traj_path(system).exists():
            C.log(f"{system}: trajectory exists, kept")
            continue
        load = C.ccd_load()
        cmd = [C.PYTHON, str(C.HERE / "trajectory.py"), system, plat]
        res = C.run_job(system, plat, cmd, C.SCRATCH / "logs" / f"traj_{system}.log")
        record_job("traj", system, plat, None, None, res, load)
        C.log(f"{system} trajectory: rc {res['rc']} {res['wall']:.0f} s (waits {res['waits']})")


def replay(systems, plats, repeats, modes=C.MODES) -> None:
    for rep in repeats:
        order = modes[rep % len(modes):] + modes[:rep % len(modes)]
        for system in systems:
            for plat in plats:
                if plat not in C.SYSTEMS[system]["platforms"]:
                    continue
                todo = [m for m in order
                        if not (C.RESULTS / "raw" / f"{system}_{plat}_{m}_r{rep}.json").exists()]
                # GPU: a system's modes of one repeat under one hold of the locks (each hold a few minutes), except
                # the TZVP case (about 4 min per mode): one mode per hold
                batch = plat == "gpu" and system not in BIG_GPU
                groups = [todo] if batch and todo else [[m] for m in todo]
                for group in groups:
                    load = C.ccd_load()
                    outs = [C.RESULTS / "raw" / f"{system}_{plat}_{m}_r{rep}.json" for m in group]
                    jobs = [([C.PYTHON, str(C.HERE / "replay.py"), system, plat, m, str(o)],
                             C.SCRATCH / "logs" / f"{o.stem}.log") for m, o in zip(group, outs)]
                    for m, o, res in zip(group, outs, C.run_jobs(system, plat, jobs)):
                        record_job("replay", system, plat, m, rep, res, load)
                        C.log(f"{o.stem}: rc {res['rc']} {res['wall']:.0f} s (waits "
                              + ", ".join(f"{k} {v:.0f} s" for k, v in res["waits"].items()) + ")")


BIG_GPU = ("u1an39-tzvp",)


NSYS = "/usr/local/cuda/bin/nsys"


def nsys_profiles(system, modes, cpus=None) -> None:
    """A GPU replay under Nsight Systems (CUDA API trace; untimed: the tracer's overhead is in its walls), and the
    CUDA API summary (results/nsys/<system>_gpu_<mode>_cuda_api.csv): what a step spends in allocations, pinned
    host memory, module loads and library handles. With `cpus`: untimed, under the shared GPU lock (through the
    turnstile), pinned to those CPUs instead of CCD0, without the bench lock."""
    out_dir = C.RESULTS / "nsys"
    out_dir.mkdir(parents=True, exist_ok=True)
    for mode in modes:
        stem = f"{system}_gpu_{mode}"
        rep = C.SCRATCH / "nsys" / stem
        rep.parent.mkdir(parents=True, exist_ok=True)
        cmd = [NSYS, "profile", "--trace=cuda", "--sample=none", "--cpuctxsw=none", "--force-overwrite=true",
               "-o", str(rep), C.PYTHON, str(C.HERE / "replay.py"), system, "gpu", mode,
               str(out_dir / f"{stem}_replay.json")]
        load = C.ccd_load()
        if cpus:
            res = C.run_untimed(system, cmd, cpus, C.SCRATCH / "logs" / f"nsys_{stem}.log")
        else:
            res = C.run_job(system, "gpu", cmd, C.SCRATCH / "logs" / f"nsys_{stem}.log")
        record_job("nsys", system, "gpu", mode, None, res, load)
        csv = subprocess.run([NSYS, "stats", "--report", "cuda_api_sum", "--format", "csv", "--force-export=true",
                              str(rep) + ".nsys-rep"], capture_output=True, text=True).stdout
        (out_dir / f"{stem}_cuda_api.csv").write_text(csv[csv.find("Time (%)"):] if "Time (%)" in csv else csv)  # no paths
        C.log(f"nsys {stem}: rc {res['rc']} {res['wall']:.0f} s")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("traj")
    t.add_argument("systems", nargs="+")
    t.add_argument("--platform", default="gpu")
    r = sub.add_parser("replay")
    r.add_argument("--systems", nargs="+", default=list(C.SYSTEMS))
    r.add_argument("--platforms", nargs="+", default=["cpu", "gpu"])
    r.add_argument("--repeats", nargs="+", type=int, default=[0, 1])
    r.add_argument("--modes", nargs="+", default=list(C.MODES), choices=C.MODES)
    sub.add_parser("overhead")
    sub.add_parser("pipe")
    n = sub.add_parser("nsys")
    n.add_argument("system")
    n.add_argument("modes", nargs="+")
    n.add_argument("--cpus", help="untimed: these CPUs, the shared GPU lock, no bench lock")
    a = ap.parse_args()
    meta = C.RESULTS / "meta.json"
    if not meta.exists():
        commit = subprocess.run(["git", "-C", str(C.ROOT), "rev-parse", "HEAD"], capture_output=True,
                                text=True).stdout.strip()
        linus_bin = C.LINUS_BIN.relative_to(C.ROOT) if C.LINUS_BIN.is_relative_to(C.ROOT) else C.LINUS_BIN.name
        C.dump(meta, dict(host=C.host_facts(), commit=commit, linus_bin=str(linus_bin),
                          threads=C.THREADS, cpus="CCD0 physical cores (slice bench)",
                          gpu_lock="exclusive (turnstile + lock) per GPU job",
                          systems=C.SYSTEMS, method=C.METHOD))
    if a.cmd == "traj":
        traj(a.systems, a.platform)
    elif a.cmd == "overhead":
        out = C.RESULTS / "overhead_water_cpu.json"
        load = C.ccd_load()
        res = C.run_job("water-svp", "cpu", [C.PYTHON, str(C.HERE / "overhead.py"), str(out)],
                        C.SCRATCH / "logs" / "overhead.log")
        record_job("overhead", "water-svp", "cpu", None, None, res, load)
    elif a.cmd == "pipe":
        out = C.RESULTS / "pipe_water_cpu.json"
        load = C.ccd_load()
        res = C.run_job("water-svp", "cpu", [C.PYTHON, str(C.HERE / "pipe.py"), str(out)],
                        C.SCRATCH / "logs" / "pipe.log")
        record_job("pipe", "water-svp", "cpu", None, None, res, load)
    elif a.cmd == "nsys":
        nsys_profiles(a.system, a.modes, a.cpus)
    else:
        replay(a.systems, a.platforms, a.repeats, tuple(a.modes))


if __name__ == "__main__":
    main()
