#!/usr/bin/env python
"""A stratified sample of tmQM transition-metal complexes -> molecules/internal/tmqm.jsonl.

Source: tmQM (D. Balcells, B. B. Skjelstad, J. Chem. Inf. Model. 60, 6135 (2020),
doi:10.1021/acs.jcim.0c01041; 2024 release), https://github.com/uiocompcat/tmQM,
tmQM/tmQM_X1.xyz.gz. Mononuclear complexes from the Cambridge Structural Database, re-optimized
with GFN2-xTB, charges -1/0/+1, closed-shell singlets. Repository licence MIT; the structures
derive from the CSD, so they stay out of the public release until that is settled
(molecules/internal/).

Selection (deterministic, seed 20261110): 5 to 36 atoms; per metal up to 6 neutral and 2 charged
complexes. Derived open-shell variants at the same geometry, tagged "vertical": the
singly-ionized cation and the anion (doublets) of every third complex, and the triplet of every
fourth. These are not minima of their own state: they test the SCF and the open-shell code
paths on realistic TM geometries, not chemistry.

    python bench/robust/build/tmqm.py <tmQM_X1.xyz.gz>
"""

from __future__ import annotations

import gzip
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import MOLDIR, TM3D, TM4D, TM5D, Z, record, write_jsonl  # noqa: E402

TM = TM3D | TM4D | TM5D


def entries(path: Path):
    with gzip.open(path, "rt") as f:
        lines = f.read().splitlines()
    i = 0
    while i < len(lines):
        if not lines[i].strip():
            i += 1
            continue
        n = int(lines[i].split()[0])
        head = lines[i + 1]
        atoms = [ln.split() for ln in lines[i + 2:i + 2 + n]]
        i += 2 + n
        meta = dict(kv.split(" = ", 1) for kv in (p.strip() for p in head.split("|")) if " = " in kv)
        yield meta, [a[0] for a in atoms], [[float(v) for v in a[1:4]] for a in atoms]


def main(argv: list[str]) -> int:
    rng = random.Random(20261110)
    by_metal: dict[str, dict[str, list]] = defaultdict(lambda: {"neutral": [], "charged": []})
    for meta, syms, geom in entries(Path(argv[0])):
        if not 5 <= len(syms) <= 36:
            continue
        metals = [s for s in syms if Z[s] in TM]
        if len(metals) != 1:
            continue
        q = int(meta.get("q", "0"))
        s_ = int(meta.get("S", "0"))
        if s_ != 0:
            continue
        by_metal[metals[0]]["charged" if q else "neutral"].append((meta, syms, geom, q))
    recs = []
    for metal in sorted(by_metal, key=lambda m: Z[m]):
        pick = []
        for kind, k in (("neutral", 6), ("charged", 2)):
            pool = by_metal[metal][kind]
            pool.sort(key=lambda e: e[0]["CSD_code"])
            pick += rng.sample(pool, min(k, len(pool)))
        for meta, syms, geom, q in pick:
            code = re.sub(r"[^A-Za-z0-9]", "", meta["CSD_code"])
            src = f"tmQM 2024 (GFN2-xTB geometry of CSD entry {meta['CSD_code']}; Balcells & Skjelstad 2020)"
            recs.append(record("tmqm", metal, code, syms, geom, q, 1, source=src,
                               geometry_level="GFN2-xTB (tmQM)", tags=["tmqm"]))
    base = list(recs)
    for i, r in enumerate(base):
        if i % 3 == 0:
            for dq, tag in ((+1, "cation"), (-1, "anion")):
                if abs(r["charge"] + dq) <= 2:
                    recs.append(record("tmqm", r["subset"], f"{r['name']}_v{tag}", r["symbols"], r["geometry_angstrom"],
                                       r["charge"] + dq, 2, source=r["source"] + f"; vertical {tag}",
                                       geometry_level="GFN2-xTB of the neutral-state complex (vertical)",
                                       tags=["tmqm", "vertical"]))
        if i % 4 == 1:
            recs.append(record("tmqm", r["subset"], f"{r['name']}_vtriplet", r["symbols"], r["geometry_angstrom"],
                               r["charge"], 3, source=r["source"] + "; vertical triplet",
                               geometry_level="GFN2-xTB of the singlet (vertical)", tags=["tmqm", "vertical"]))
    write_jsonl(MOLDIR / "internal" / "tmqm.jsonl", recs)
    print("metals:", len({r["subset"] for r in recs}), " closed shell:", sum(r["multiplicity"] == 1 for r in recs),
          " open shell:", sum(r["multiplicity"] > 1 for r in recs))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
