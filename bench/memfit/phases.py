"""phases.py PLAN.json...: the plan's memory phases and parts (GiB)."""
import json
import sys

for f in sys.argv[1:]:
    m = json.load(open(f))["memory"]
    print(f"{f.split('/')[-1]}: required {m['required_gib']:.3f} limit {m['limit_gib']}")
    for p in m["phases"]:
        parts = ", ".join(f"{x['what'][:24]} {x['gib']:.3f}" for x in p["parts"])
        print(f"   {p['phase']:9} {p['required_gib']:.3f}: {parts}")
