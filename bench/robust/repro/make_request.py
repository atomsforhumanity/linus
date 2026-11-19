#!/usr/bin/env python
"""Write a self-contained linus JSON request for a molecule of the ROBUST sets.

    python make_request.py <molecule id> <out.json> key=value ...   (values parsed as JSON)

    python make_request.py gmtkn55.W4-11.cn cn_uhf.json method='"hf"' basis='"def2-svp"' \
        jk='"df"' scf='{"convergence": "extreme", "max_iter": 300}'

Positions in bohr (CODATA 2022, as the sweep's task.py); run with `linus --json < out.json`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import sets as molecules  # noqa: E402


def main() -> int:
    mid, out = sys.argv[1], Path(sys.argv[2])
    rec = next(r for r in molecules.load() if r["id"] == mid)
    req = {"structure": {"symbols": rec["symbols"],
                         "geometry": [[c / 0.529177210544 for c in x] for x in rec["geometry_angstrom"]],
                         "charge": rec["charge"], "multiplicity": rec["multiplicity"]}}
    for kv in sys.argv[3:]:
        k, v = kv.split("=", 1)
        req[k] = json.loads(v)
    out.write_text(json.dumps(req, indent=1) + "\n")
    print(f"{mid}: {rec['natoms']} atoms, charge {rec['charge']}, multiplicity {rec['multiplicity']} -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
