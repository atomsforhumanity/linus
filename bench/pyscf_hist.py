"""Print the trace written by bench/scf_trace_pyscf.py."""
import json, sys

d = json.load(open(sys.argv[1]))
for r in d["trace"]:
    print(f"{r['cycle']:3d} {r['energy']:22.12f} {r['de']:10.2e} norm_gorb {r['norm_gorb']:9.2e} gmax {r['gmax']:9.2e} "
          f"grms {r['grms']:9.2e} {r['time']:7.3f}")
print({k: v for k, v in d.items() if k != "trace"})
