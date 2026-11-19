"""Compare linus JSON results (from bench/scf_runs.py --out) bitwise: final energies and the
per-iteration SCF history, to locate where thread counts first make a difference.

usage: python bench/det_compare.py a.json b.json [c.json ...]
"""
import json, sys

runs = [(p, json.load(open(p))) for p in sys.argv[1:]]
ref_p, ref = runs[0]
for p, d in runs:
    print(f"{p.split('/')[-1]}: E {d['energy']!r} ({d['energy'].hex()}) builds {d['scf']['fock_builds']}")
for p, d in runs[1:]:
    first = None
    for a, b in zip(ref["scf"]["history"], d["scf"]["history"]):
        if a["energy"] != b["energy"] or a["gmax"] != b["gmax"]:
            first = (a["iter"], a["energy"] - b["energy"], a["gmax"], b["gmax"])
            break
    same = d["energy"] == ref["energy"]
    print(f"{p.split('/')[-1]} vs {ref_p.split('/')[-1]}: final E {'identical' if same else f'differs by {d[chr(101)+chr(110)+chr(101)+chr(114)+chr(103)+chr(121)] - ref[chr(101)+chr(110)+chr(101)+chr(114)+chr(103)+chr(121)]:+.3e}'}; "
          f"first differing iteration: {first}")
