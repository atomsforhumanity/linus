#!/usr/bin/env python
"""Frequencies of X-H molecules with the default Hessian grids (auto: linus:3, CPKS on linus:1)
and with linus:2, against linus:4 with the CPKS on the same grid.

    LINUS_BIN=path/to/linus python grid3_freqs.py [METHOD] [NAMES]   (default b3lyp, the X-H set)

Same idealized geometry for every grid (not optimized: the comparison is between grids).
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from grid3_hydrides import MOLS, request, run  # noqa: E402


def main():
    method = sys.argv[1] if len(sys.argv) > 1 else "b3lyp"
    names = sys.argv[2].split(",") if len(sys.argv) > 2 else ["H2O", "PH3", "AsH3", "H2Se", "HBr", "GaH3", "HCl"]
    for name in names:
        sym, pos = MOLS[name]
        out = {}
        for label, extra in (("ref", {"grid": "linus:4", "response": {"grid": "linus:4"}}), ("default", {}),
                             ("linus:2", {"grid": "linus:2"})):
            o = run(request(sym, pos, method, "hessian", **extra))
            if o.get("error"):
                print(name, label, "error", str(o["error"])[:300])
                continue
            out[label] = np.array((o.get("vibrational_analysis") or {}).get("frequencies") or [], float)
        ref = out.get("ref")
        cells = []
        for label in ("default", "linus:2"):
            if label in out and ref is not None and ref.size and out[label].size == ref.size:
                cells.append(f"{label}: max |d nu| {np.abs(out[label] - ref).max():.2f} cm-1")
        print(f"{name:5s} " + " | ".join(cells) + f"  (linus:4: {np.round(ref, 1).tolist() if ref is not None else None})",
              flush=True)


if __name__ == "__main__":
    main()
