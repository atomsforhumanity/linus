#!/usr/bin/env python
"""Collect the GFN2-xTB run (build/xtb_opt.py) into molecules/drugs.jsonl and molecules/inorganic.jsonl.

    python finish_xtb.py RUN_OUTPUT_DIR DRUG_INPUTS INORGANIC_INPUTS

Each optimized structure is kept with its GFN2 energy, cycle count and the bond check; a
structure whose bonding changed is kept but tagged "topology_changed" (GFN2 dissociating a
ligand of an aqua ion, a proton moving in a zwitterion: still a valid geometry for SCF tests).
Failed optimizations fall back to the starting geometry, tagged "unoptimized". Vertical states
(radical cations of drug-like molecules) take their parent's optimized geometry.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import MOLDIR, parse_xyz, record, write_jsonl  # noqa: E402

XTB = "GFN2-xTB (xtb 6.7.1, --opt tight)"


def main(argv: list[str]) -> int:
    out_dir, drug_in, inorg_in = (Path(x) for x in argv[:3])
    res = {}
    for f in out_dir.glob("*/result.json"):
        for r in json.loads(f.read_text())["results"]:
            res[r["id"]] = r
    drugs, inorg = [], []
    stats = Counter()
    for src, dest in ((drug_in, drugs), (inorg_in, inorg)):
        inputs = [json.loads(x) for x in (src / "inputs.jsonl").read_text().splitlines()]
        geoms = {}
        for x in inputs:
            vertical = x.get("vertical_of")
            r = res.get(vertical or x["id"])
            tags = []
            if r and r["status"] in ("ok", "not_optimized"):
                syms, pos, _ = parse_xyz(r["xyz"])
                level = XTB if r["status"] == "ok" else "as built (single atom)"
                if r.get("topology_changed"):
                    tags.append("topology_changed")
                stats["ok" if r["status"] == "ok" else "atom"] += 1
            else:
                syms, pos, _ = parse_xyz((src / x["xyz"]).read_text())
                level = "unoptimized (xtb failed: " + (r or {}).get("reason", "missing") + ")"
                tags.append("unoptimized")
                stats["failed"] += 1
            if vertical:
                tags.append("vertical")
                level += f", vertical from {vertical}"
            if dest is drugs:
                setname, subset, name = "drug", None, x["id"].split(".", 1)[1]
                source = f"PubChem CID {x['cid']} ({x['name']}), SMILES {x['smiles']}; variant {x['variant']}"
                extra = {"pubchem_cid": int(x["cid"]), "smiles": x["smiles"], "variant": x["variant"]}
                tags += ["drug_like", x["variant"]]
            else:
                setname, subset, name = x["set"], None, x["name"]
                source = "built (bench/robust/build/inorganic.py)"
                extra = {}
            if r and r.get("energy_gfn2") is not None:
                extra["energy_gfn2"] = r["energy_gfn2"]
            rec = record(setname, subset, name, syms, pos, x["charge"], x["multiplicity"], source=source,
                         geometry_level=level, tags=tags, extra=extra)
            geoms[x["id"]] = rec
            dest.append(rec)
    write_jsonl(MOLDIR / "drugs.jsonl", drugs)
    write_jsonl(MOLDIR / "inorganic.jsonl", inorg)
    print(dict(stats), "topology changed:", sum("topology_changed" in r["tags"] for r in drugs + inorg))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
