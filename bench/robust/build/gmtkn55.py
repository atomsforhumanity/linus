#!/usr/bin/env python
"""GMTKN55 geometries -> bench/robust/molecules/internal/gmtkn55.jsonl.

Source: the GMTKN55 database (L. Goerigk, A. Hansen, C. Bauer, S. Ehrlich, A. Najibi, S. Grimme,
Phys. Chem. Chem. Phys. 19, 32184 (2017), doi:10.1039/C7CP04913G), one tar per subset from
http://www.thch.uni-bonn.de/tc.old/downloads/GMTKN/GMTKN55/<SUBSET>.tar, each system a
directory with a Turbomole `coord` file (bohr) and `.CHRG` / `.UHF` (absent = 0).

Licence: the website states none; the geometries are for research use with citation. They are
not redistributed in the public release (bench/robust/molecules/internal/ is meant for the
release manifest's exclusions).

    python bench/robust/build/gmtkn55.py <dir with the 55 tar files>
"""

from __future__ import annotations

import sys
import tarfile
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import BOHR_TO_ANGSTROM, MOLDIR, record, write_jsonl  # noqa: E402

# the five GMTKN55 categories (Goerigk et al. 2017, Table 1)
CATEGORY = {
    "thermo_small": "W4-11 G21EA G21IP DIPCS10 PA26 SIE4x4 ALKBDE10 YBDE18 AL2X6 HEAVYSB11 NBPRC ALK8 RC21 G2RC "
                    "BH76RC FH51 TAUT15 DC13".split(),
    "thermo_large": "MB16-43 DARC RSE43 BSR36 CDIE20 ISO34 ISOL24 C60ISO PArel".split(),
    "barrier": "BH76 BHPERI BHDIV10 INV24 BHROT27 PX13 WCPT18".split(),
    "nci_inter": "RG18 ADIM6 S22 S66 HEAVY28 WATER27 CARBHB12 PNICO23 HAL59 AHB21 CHB6 IL16".split(),
    "nci_intra": "IDISP ICONF ACONF Amino20x4 PCONF21 MCONF SCONF UPU23 BUT14DIOL".split(),
}
CAT_OF = {s: c for c, subs in CATEGORY.items() for s in subs}


def parse_coord(text: str) -> tuple[list[str], list[list[float]]]:
    syms, xyz = [], []
    on = False
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("$coord"):
            on = True
            continue
        if s.startswith("$"):
            if on:
                break
            continue
        if on and s:
            p = s.split()
            xyz.append([float(v) * BOHR_TO_ANGSTROM for v in p[:3]])
            syms.append(p[3])
    return syms, xyz


def main(argv: list[str]) -> int:
    src = Path(argv[0])
    recs = []
    for tf in sorted(src.glob("*.tar")):
        subset = tf.stem
        with tarfile.open(tf) as t:
            files = {m.name: m for m in t.getmembers() if m.isfile()}
            systems = sorted({str(Path(n).parent) for n in files if n.endswith("/coord")})
            for sysdir in systems:
                def read(name: str) -> str | None:
                    m = files.get(f"{sysdir}/{name}")
                    return t.extractfile(m).read().decode() if m else None

                syms, geom = parse_coord(read("coord"))
                chrg = int((read(".CHRG") or "0").split()[0])
                uhf = int((read(".UHF") or "0").split()[0])
                name = Path(sysdir).name
                tags = [f"gmtkn55_{CAT_OF.get(subset, 'other')}"]
                from common import Z
                if (sum(Z[s.capitalize()] for s in syms) - chrg + uhf) % 2:
                    uhf += 1  # odd electron count without a .UHF file (e.g. SIE4x4/h): a doublet
                    tags.append("uhf_inferred")
                rec = record("gmtkn55", subset, name, syms, geom, chrg, uhf + 1,
                             source=f"GMTKN55 {subset}/{name} (coord, bohr; Goerigk et al. 2017)",
                             geometry_level="as published in GMTKN55", tags=tags)
                recs.append(rec)
    # the same structure appears in several subsets (BH76 and BH76RC, W4-11 and others): keep the
    # first, list the others as aliases
    seen: dict[tuple, dict] = {}
    uniq = []
    for r in recs:
        key = (tuple(r["symbols"]), r["charge"], r["multiplicity"],
               tuple(round(c, 4) for xyz in r["geometry_angstrom"] for c in xyz))
        if key in seen:
            seen[key].setdefault("aliases", []).append(f"{r['subset']}/{r['name']}")
            continue
        seen[key] = r
        uniq.append(r)
    print(f"{len(recs)} systems, {len(uniq)} unique structures")
    recs = uniq
    write_jsonl(MOLDIR / "internal" / "gmtkn55.jsonl", recs)
    c = Counter(r["subset"] for r in recs)
    print(len(c), "subsets;", sum(c.values()), "systems")
    print("open shell:", sum(r["multiplicity"] > 1 for r in recs), " charged:", sum(r["charge"] != 0 for r in recs),
          " ECP:", sum("ecp" in r["tags"] for r in recs), " <=30 atoms:", sum(r["natoms"] <= 30 for r in recs))
    print("elements:", sorted({e for r in recs for e in r["elements"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
