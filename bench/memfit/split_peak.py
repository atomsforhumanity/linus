"""split_peak.py TRACE.json...: the peak RSS (GiB) up to the last SCF iteration line and after it."""
import json
import sys

G = 1048576
for f in sys.argv[1:]:
    d = json.load(open(f))
    its = [t for t, l in d["stderr"] if l.lstrip().startswith("iter")]
    tb = its[-1] if its else 0.0
    a = max((s[1] for s in d["samples"] if s[0] <= tb + 0.05), default=0)
    b = max((s[1] for s in d["samples"] if s[0] > tb + 0.05), default=0)
    print(f"{f.split('/')[-1]:22} scf {a / G:.3f}  after {b / G:.3f}  (boundary {tb:.1f}s of {d['wall']:.1f}s)")
