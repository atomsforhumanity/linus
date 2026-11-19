#!/usr/bin/env python3
"""Run a command and sample its resident memory over time, with its stderr lines timestamped.

usage: rss_trace.py OUT.json [--stdin FILE] [--stdout FILE] [--dt 0.005] [--smaps] -- CMD ...

Samples /proc/PID/status (VmRSS, VmHWM, RssAnon, RssFile, RssShmem) of the process and its
descendants (summed) every dt seconds. With --smaps, also snapshots /proc/PID/smaps of the largest
process whenever its RSS grows by 4 MiB over the last snapshot (mappings with 256 kB or more
resident), and keeps the snapshot nearest the peak. Writes OUT.json with the samples, the stderr
lines (t, text), the peak and the exit status; prints a one-line summary.
"""
import json
import os
import subprocess
import sys
import threading
import time


def status(pid):
    out = {}
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                k, _, v = line.partition(":")
                if k in ("VmRSS", "VmHWM", "RssAnon", "RssFile", "RssShmem"):
                    out[k] = int(v.split()[0])
    except OSError:
        return None
    return out


def smaps(pid):
    maps = []
    cur = None
    try:
        with open(f"/proc/{pid}/smaps") as f:
            for line in f:
                parts = line.split()
                if "-" in parts[0] and len(parts) >= 5 and ":" not in parts[0]:
                    a, b = parts[0].split("-")
                    cur = [int(a, 16), int(b, 16), 0, parts[5] if len(parts) > 5 else ""]
                    maps.append(cur)
                elif parts[0] == "Rss:" and cur is not None:
                    cur[2] = int(parts[1])
    except OSError:
        return None
    return [m for m in maps if m[2] >= 256]


def children(pid):
    try:
        with open(f"/proc/{pid}/task/{pid}/children") as f:
            kids = [int(x) for x in f.read().split()]
    except OSError:
        return []
    out = []
    for k in kids:
        out.append(k)
        out.extend(children(k))
    return out


def main():
    a = sys.argv[1:]
    out = a.pop(0)
    stdin = stdout = None
    dt = 0.005
    want_smaps = False
    while a and a[0] != "--":
        o = a.pop(0)
        if o == "--stdin":
            stdin = a.pop(0)
        elif o == "--stdout":
            stdout = a.pop(0)
        elif o == "--dt":
            dt = float(a.pop(0))
        elif o == "--smaps":
            want_smaps = True
    a.pop(0)
    fin = open(stdin, "rb") if stdin else subprocess.DEVNULL
    fout = open(stdout, "wb") if stdout else subprocess.DEVNULL
    t0 = time.monotonic()
    p = subprocess.Popen(a, stdin=fin, stdout=fout, stderr=subprocess.PIPE)
    lines = []

    def reader():
        for raw in p.stderr:
            lines.append((time.monotonic() - t0, raw.decode(errors="replace").rstrip("\n")))

    th = threading.Thread(target=reader, daemon=True)
    th.start()
    samples = []
    snap = None
    snap_rss = 0
    while p.poll() is None:
        pids = [p.pid] + children(p.pid)
        tot = {}
        big = (0, None)
        for q in pids:
            s = status(q)
            if s:
                for k, v in s.items():
                    tot[k] = tot.get(k, 0) + v
                if s.get("VmRSS", 0) > big[0]:
                    big = (s["VmRSS"], q)
        if tot:
            t = time.monotonic() - t0
            samples.append((t, tot.get("VmRSS", 0), tot.get("RssAnon", 0),
                            tot.get("RssFile", 0), tot.get("RssShmem", 0), tot.get("VmHWM", 0)))
            if want_smaps and big[1] and big[0] > snap_rss + 4096:
                m = smaps(big[1])
                if m is not None:
                    snap = {"t": t, "pid": big[1], "rss_kb": big[0], "maps": m}
                    snap_rss = big[0]
        time.sleep(dt)
    rc = p.wait()
    th.join(timeout=5)
    wall = time.monotonic() - t0
    peak = max(samples, key=lambda s: s[1]) if samples else None
    hwm = max((s[5] for s in samples), default=0)
    res = {
        "cmd": a,
        "rc": rc,
        "wall": wall,
        "fields": ["t", "rss_kb", "anon_kb", "file_kb", "shmem_kb", "hwm_kb"],
        "samples": samples,
        "stderr": lines,
        "peak": peak,
        "hwm_kb": hwm,
        "smaps": snap,
    }
    with open(out, "w") as f:
        json.dump(res, f)
    G = 1048576
    if peak:
        print(f"{os.path.basename(out)} rc={rc} wall={wall:.1f}s hwm={hwm / G:.3f} GiB peak_rss={peak[1] / G:.3f} "
              f"(anon {peak[2] / G:.3f} file {peak[3] / G:.3f}) at t={peak[0]:.2f}s")
    else:
        print(f"{os.path.basename(out)} rc={rc} wall={wall:.1f}s (no samples)")


if __name__ == "__main__":
    main()
