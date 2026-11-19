#!/usr/bin/env python3
"""attrib.py RUN.json RUN.strace BIN [nframes]: attribute the resident mappings of the smaps snapshot nearest the
peak to the mmap call (with its stack) that last created each, from strace -f -k output (mmap/munmap/mremap).
Frames in BIN are symbolized with addr2line (file offset + 0x1000 - 1: the text segment's offset delta, the
return address's call); the first nframes frames in linus's crates are shown."""
import json
import re
import subprocess
import sys

run = json.load(open(sys.argv[1]))
binp = sys.argv[3]
nfr = int(sys.argv[4]) if len(sys.argv) > 4 else 4
snap = run["smaps"]
events = []  # (addr, len, [offsets])
cur = None
rx = re.compile(r"^(\d+)\s+mmap\((\S+), (\d+), [^)]*\) = (0x[0-9a-f]+)")
rf = re.compile(r"^ > " + re.escape(binp) + r"\(\S*\) \[(0x[0-9a-f]+)\]")
for line in open(sys.argv[2]):
    m = rx.match(line)
    if m:
        cur = [int(m.group(4), 16), int(m.group(3)), []]
        events.append(cur)
        continue
    if line.startswith(" > "):
        if cur is not None:
            mf = rf.match(line)
            if mf:
                cur[2].append(int(mf.group(1), 16))
    else:
        cur = None

G = 1048576
rows = []
need = set()
for a, b, rss, name in snap["maps"]:
    hit = None
    for ea, el, st in reversed(events):
        if ea <= a < ea + el:
            hit = st
            break
    rows.append([rss, a, b, name, hit or []])
    need.update(hit or [])
need = sorted(need)
sym = {}
if need:
    out = subprocess.run(["addr2line", "-f", "-C", "-i", "-e", binp] + [hex(x + 0x1000 - 1) for x in need],
                         capture_output=True, text=True).stdout.splitlines()
    # -i prints several (func, file:line) pairs per address; split by re-running per address is slow, so
    # use a sentinel: addr2line with -a prints the address first
    out = subprocess.run(["addr2line", "-a", "-f", "-C", "-i", "-e", binp] + [hex(x + 0x1000 - 1) for x in need],
                         capture_output=True, text=True).stdout.splitlines()
    curaddr = None
    for i, l in enumerate(out):
        if l.startswith("0x") and re.fullmatch(r"0x[0-9a-f]+", l):
            curaddr = int(l, 16) + 1 - 0x1000
            sym[curaddr] = []
        elif curaddr is not None:
            sym[curaddr].append(l)


def site(offs):
    out = []
    for o in offs:
        lines = sym.get(o, [])
        for k in range(0, len(lines) - 1, 2):
            fn, loc = lines[k], lines[k + 1]
            if "/crates/" in loc:
                short = re.sub(r".*/crates/", "", loc)
                f = re.sub(r"<([^<>]*)>::", r"\1::", fn)
                out.append(f"{f.split('::{closure')[0][-60:]} ({short})")
        if len(out) >= nfr:
            break
    return out[:nfr]


tot = sum(r[0] for r in rows)
rows.sort(reverse=True)
print(f"snapshot t={snap['t']:.2f}s rss={snap['rss_kb'] / G:.3f} GiB, mappings >= 256 kB: {tot / G:.3f} GiB")
for rss, a, b, name, offs in rows[:40]:
    s = site(offs) if offs else (["(no linus frame)"] if name == "" else [name])
    print(f"{rss / 1024:8.1f} MiB of {(b - a) / 1048576:8.1f} {name[-20:]:20} " + " < ".join(s))
