#!/usr/bin/env python3
"""mkin.py BASE.json OUT.json [key=value ...]: copy a request with keys set (JSON values; key=none drops it)."""
import json
import sys

d = json.load(open(sys.argv[1]))
for kv in sys.argv[3:]:
    k, _, v = kv.partition("=")
    ks = k.split(".")
    cur = d
    for kk in ks[:-1]:
        cur = cur.setdefault(kk, {})
    if v == "none":
        cur.pop(ks[-1], None)
        continue
    try:
        v = json.loads(v)
    except json.JSONDecodeError:
        pass
    cur[ks[-1]] = v
json.dump(d, open(sys.argv[2], "w"))
