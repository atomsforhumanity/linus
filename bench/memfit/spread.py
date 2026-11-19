#!/usr/bin/env python3
"""spread.py CASES.json OUTDIR BIN [--jobs N] [--front cli|py|qcc] [--cg] [--only a,b] [--py PYTHON]

Run each case of CASES.json with linus at an explicit memory_gib, under /usr/bin/time -v (peak RSS) and, with
--cg, in a systemd scope whose MemoryMax equals memory_gib (MemorySwapMax=0: an overshoot is an OOM kill).
A case: {"name", "xyz", "method", "basis", "calctype", "charge", "mult", "threads", "memory_gib": float|"min",
"opts": {...extra request keys...}}. "min" probes the smallest memory_gib with a run at 0.05 GiB (its
host_memory failure's error_details.required_gib, rounded up to 0.1 GiB) and fails the case if the probe runs.

Fronts: cli (BIN --json), py (linus.compute in a fresh python, BIN is the python), qcc (qccompute's runner
in a fresh python: the worker process of qccompute.serve, BIN the python). One line per case to OUTDIR/table.tsv:
name, memory_gib, rc, peak_rss_gib, margin_gib (memory_gib - peak), wall_s, killed, the DF storage and budget.
"""
import json
import math
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_MOL = os.path.dirname(os.path.dirname(HERE))


def read_xyz(path):
    for base in ("", REPO_MOL + "/tests/molecules/", REPO_MOL + "/bench/molecules/"):
        p = base + path
        if os.path.exists(p):
            return open(p).read()
    raise FileNotFoundError(path)


def request(case, mem):
    xyz = read_xyz(case["xyz"])
    d = {
        "structure": {"xyz": xyz, "charge": case.get("charge", 0), "multiplicity": case.get("mult", 1)},
        "method": case["method"],
        "basis": case["basis"],
        "calctype": case["calctype"],
        "threads": case.get("threads", 1),
    }
    if mem is not None:
        d["memory_gib"] = mem
    for k, v in case.get("opts", {}).items():
        d[k] = v
    return d




def run_one(case, mem, outdir, binp, front, cg, cpus, tag):
    os.makedirs(outdir, exist_ok=True)
    name = case["name"] + tag
    inp = os.path.join(outdir, f"{name}.in.json")
    out = os.path.join(outdir, f"{name}.out.json")
    err = os.path.join(outdir, f"{name}.err")
    json.dump(request(case, mem), open(inp, "w"))
    if front == "cli":
        cmd = [binp, "--json"]
        stdin = open(inp)
        stdout = open(out, "w")
    else:
        cmd = [binp, os.path.join(HERE, "py_runner.py"), inp, out, front]
        stdin = subprocess.DEVNULL
        stdout = subprocess.DEVNULL
    cmd = ["taskset", "-c", cpus] + cmd
    cmd = ["/usr/bin/time", "-v"] + cmd
    if cg and mem is not None:
        cmd = ["systemd-run", "--user", "--scope", "-q", "-p", f"MemoryMax={int(mem * 1073741824)}",
               "-p", "MemorySwapMax=0", "-p", "OOMPolicy=continue"] + cmd
    env = dict(os.environ)
    if front != "cli" and "PYTHONPATH_LINUS" in env:
        env["PYTHONPATH"] = env["PYTHONPATH_LINUS"]
    t0 = time.time()
    rc = subprocess.call(cmd, stdin=stdin, stdout=stdout, stderr=open(err, "w"), env=env)
    wall = time.time() - t0
    etxt = open(err, errors="replace").read()
    m = re.search(r"Maximum resident set size \(kbytes\): (\d+)", etxt)
    rss = int(m.group(1)) / 1048576 if m else float("nan")
    killed = "signal 9" in etxt
    try:
        o = json.load(open(out))
    except Exception:  # noqa: BLE001
        o = {}
    return name, mem, rc, rss, wall, killed, o


def plan_required(case, outdir, binp, front):
    """The plan's memory.required_gib for the case (at memory_gib 64, so nothing fails), rounded up to 0.1 GiB."""
    inp = os.path.join(outdir, f"{case['name']}.plan.in.json")
    out = os.path.join(outdir, f"{case['name']}.plan.json")
    json.dump(request(case, 64.0), open(inp, "w"))
    env = dict(os.environ)
    if front == "cli":
        subprocess.run([binp, "--plan", "--json"], stdin=open(inp), stdout=open(out, "w"), check=True)
    else:
        if "PYTHONPATH_LINUS" in env:
            env["PYTHONPATH"] = env["PYTHONPATH_LINUS"]
        subprocess.run([binp, os.path.join(HERE, "py_runner.py"), inp, out, "plan"], check=True, env=env)
    p = json.load(open(out))
    req = (p.get("memory") or p["record"]["memory"])["required_gib"]
    return math.ceil(req * 10 - 1e-9) / 10


def find_resolved(x, depth=0):
    """The first dict under a key 'resolved' that has a 'record' (CLI JSON or a qcdata output dump)."""
    if depth > 6:
        return None
    if isinstance(x, dict):
        r = x.get("resolved")
        if isinstance(r, dict) and "record" in r:
            return r
        for v in x.values():
            f = find_resolved(v, depth + 1)
            if f:
                return f
    return None


def summarize(o):
    if not o:
        return "", "", ""
    try:
        res = find_resolved(o) or {}
        df = res.get("record", {}).get("df") or {}
        st = f"{df.get('storage', '')}/{df.get('long_range', '')}" if df else ""
        bud = df.get("budget_gib", "")
    except Exception:  # noqa: BLE001
        st, bud = "", ""
    err = o.get("error", "") if not o.get("success", True) else ""
    det = o.get("error_details") or {}
    if det:
        err = f"{o.get('error_code', '')} required {det.get('required_gib')}"
    return st, bud, err[:120]


def main():
    a = sys.argv[1:]
    cases = json.load(open(a[0]))
    outdir, binp = a[1], a[2]
    jobs, front, cg, only, cpuset = 4, "cli", False, None, "24-39,88-103"
    i = 3
    while i < len(a):
        if a[i] == "--jobs":
            jobs = int(a[i + 1]); i += 2
        elif a[i] == "--front":
            front = a[i + 1]; i += 2
        elif a[i] == "--cg":
            cg = True; i += 1
        elif a[i] == "--only":
            only = set(a[i + 1].split(",")); i += 2
        elif a[i] == "--cpus":
            cpuset = a[i + 1]; i += 2
        else:
            raise SystemExit(f"unknown {a[i]}")
    if only:
        cases = [c for c in cases if c["name"] in only]
    os.makedirs(outdir, exist_ok=True)
    # cpus for each job: the case's threads from the set, in order
    allc = []
    for part in cpuset.split(","):
        lo, _, hi = part.partition("-")
        allc.extend(range(int(lo), int(hi or lo) + 1))
    phys = [c for c in allc if c < 64]
    import threading
    lock = threading.Lock()
    free = list(phys)

    def take(n):
        while True:
            with lock:
                if len(free) >= n:
                    got = free[:n]
                    del free[:n]
                    return got
            time.sleep(0.5)

    def give(got):
        with lock:
            free.extend(got)
            free.sort()

    tsv = open(os.path.join(outdir, "table.tsv"), "a")

    def job(case):
        n = case.get("threads", 1)
        got = take(n)
        try:
            cpus = ",".join(map(str, got))
            mem = case["memory_gib"]
            if mem == "min":
                _, _, rc, _, _, _, o = run_one(case, 0.05, outdir, binp, front, False, cpus, ".probe")
                det = o.get("error_details") or {}
                req = det.get("required_gib")
                if rc == 0 or req is None:
                    line = f"{case['name']}\tmin\tprobe-ran\t\t\t\t\t\t\t{summarize(o)[2]}"
                    with lock:
                        tsv.write(line + "\n"); tsv.flush(); print(line, flush=True)
                    return
                mem = math.ceil(req * 10 - 1e-9) / 10
            if mem == "plan":
                mem = plan_required(case, outdir, binp, front)
            mem = None if mem == "auto" else mem
            for _ in range(4):
                name, mem, rc, rss, wall, killed, o = run_one(case, mem, outdir, binp, front, cg, cpus, "")
                det = o.get("error_details") or {}
                req = det.get("required_gib")
                # a later phase's host_memory failure (a Hessian's response tensors): retry at its minimum
                if rc != 0 and req is not None and mem is not None and req > mem and case["memory_gib"] == "min":
                    mem = math.ceil(req * 10 - 1e-9) / 10
                    continue
                break
            st, bud, err = summarize(o)
            margin = (mem - rss) if mem is not None else float("nan")
            line = (f"{name}\t{mem}\t{rc}\t{rss:.3f}\t{margin:+.3f}\t{wall:.0f}\t{'KILLED' if killed else ''}\t"
                    f"{st}\t{bud}\t{err}")
            with lock:
                tsv.write(line + "\n"); tsv.flush(); print(line, flush=True)
        finally:
            give(got)

    with ThreadPoolExecutor(jobs) as ex:
        list(ex.map(job, cases))


if __name__ == "__main__":
    main()
