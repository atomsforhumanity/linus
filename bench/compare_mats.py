"""Max abs difference between two square f64 matrices stored as raw little-endian files.

python bench/compare_mats.py a.bin b.bin    -> prints JSON {max_abs, max_ref, rel}
"""
import json
import sys

import numpy as np

a = np.fromfile(sys.argv[1], dtype="<f8")
b = np.fromfile(sys.argv[2], dtype="<f8")
n = int(round(len(a) ** 0.5))
a, b = a.reshape(n, n), b.reshape(n, n)
d = np.abs(a - b).max()
print(json.dumps({"n": n, "max_abs": float(d), "max_ref": float(np.abs(a).max()), "rel": float(d / np.abs(a).max())}))
