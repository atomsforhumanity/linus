"""Shared helpers for the ROBUST molecule sets (bench/robust/molecules/*.jsonl).

One JSON object per line and molecule:

    {"id": "gmtkn55.W4-11.cn", "set": "gmtkn55", "subset": "W4-11", "name": "cn",
     "symbols": ["C", "N"], "geometry_angstrom": [[...], [...]], "charge": 0, "multiplicity": 2,
     "natoms": 2, "nelectrons": 13, "elements": ["C", "N"], "tags": [...],
     "source": "...", "geometry_level": "..."}

Geometries are in angstrom with full float precision; sources in bohr are converted with the
CODATA 2022 factor (qcref), the same factor the pipeline uses back to bohr, so the round trip is
exact to ~1e-16. Tags are derived from the composition (`derive_tags`) plus whatever the builder
adds (subset category, "vertical", "stretched", ...).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable

SYMBOLS = [
    "X", "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar",
    "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr",
    "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I", "Xe",
    "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu",
    "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn",
]
Z = {s: i for i, s in enumerate(SYMBOLS)}
TM3D = set(range(21, 31))
TM4D = set(range(39, 49))
TM5D = {57} | set(range(72, 81))
LANTHANIDES = set(range(58, 72))
BOHR_TO_ANGSTROM = 0.529177210544  # CODATA 2022 (qcref.constants.BOHR_TO_ANGSTROM)

MOLDIR = Path(__file__).resolve().parents[1] / "molecules"


def norm_symbol(s: str) -> str:
    s = s.strip()
    return s[0].upper() + s[1:].lower()


def safe_id(text: str) -> str:
    """Filesystem- and run-id-safe: [A-Za-z0-9._-]."""
    t = text.replace("+", "p").replace("(", "_").replace(")", "_").replace(",", "_").replace("'", "q")
    t = re.sub(r"[^A-Za-z0-9._-]+", "_", t)
    return re.sub(r"_+", "_", t).strip("_")


def row_of(z: int) -> int:
    for r, hi in enumerate((2, 10, 18, 36, 54, 86), start=1):
        if z <= hi:
            return r
    return 7


def derive_tags(symbols: list[str], charge: int, mult: int) -> list[str]:
    zs = [Z[s] for s in symbols]
    t: set[str] = set()
    n = len(symbols)
    t.add("tiny" if n <= 4 else "small" if n <= 12 else "medium" if n <= 30 else "large" if n <= 60 else "xl")
    if mult == 1:
        t.add("closed_shell")
    else:
        t.add("open_shell")
        t.add({2: "doublet", 3: "triplet", 4: "quartet", 5: "quintet", 6: "sextet"}.get(mult, f"mult{mult}"))
    if charge > 0:
        t.add("cation")
    elif charge < 0:
        t.add("anion")
    if abs(charge) >= 2:
        t.add("multiply_charged")
    if any(z >= 37 for z in zs):
        t.add("ecp")  # def2 basis sets put ECPs on Rb-Rn
    if any(z in TM3D for z in zs):
        t.add("tm3d")
    if any(z in TM4D for z in zs):
        t.add("tm4d")
    if any(z in TM5D for z in zs):
        t.add("tm5d")
    if any(z in TM3D | TM4D | TM5D for z in zs):
        t.add("transition_metal")
    if any(z >= 37 and z not in TM4D | TM5D | LANTHANIDES for z in zs):
        t.add("heavy_main_group")
    t.add(f"row{max(row_of(z) for z in zs)}")
    return sorted(t)


def record(set_name: str, subset: str | None, name: str, symbols: list[str], geom_ang: list[list[float]],
           charge: int, mult: int, source: str, geometry_level: str, tags: Iterable[str] = (),
           extra: dict[str, Any] | None = None) -> dict[str, Any]:
    symbols = [norm_symbol(s) for s in symbols]
    nel = sum(Z[s] for s in symbols) - charge
    if (nel + mult - 1) % 2:
        raise ValueError(f"{set_name}/{subset}/{name}: {nel} electrons cannot have multiplicity {mult}")
    if mult - 1 > nel:
        raise ValueError(f"{set_name}/{subset}/{name}: multiplicity {mult} with {nel} electrons")
    rid = safe_id(".".join(x for x in (set_name, subset, name) if x))
    rec = {
        "id": rid, "set": set_name, "subset": subset, "name": name, "symbols": symbols,
        "geometry_angstrom": [[float(c) for c in xyz] for xyz in geom_ang], "charge": int(charge),
        "multiplicity": int(mult), "natoms": len(symbols), "nelectrons": nel,
        "elements": sorted(set(symbols), key=lambda s: Z[s]),
        "tags": sorted(set(derive_tags(symbols, charge, mult)) | set(tags)),
        "source": source, "geometry_level": geometry_level,
    }
    if extra:
        rec.update(extra)
    return rec


def write_jsonl(path: Path, recs: list[dict[str, Any]]) -> None:
    ids = [r["id"] for r in recs]
    dup = {i for i in ids if ids.count(i) > 1}
    if dup:
        raise ValueError(f"duplicate ids: {sorted(dup)[:10]}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in recs:
            f.write(json.dumps(r, separators=(",", ":")) + "\n")
    print(f"wrote {len(recs)} molecules to {path}")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def xyz_text(rec: dict[str, Any]) -> str:
    lines = [str(rec["natoms"]), f"{rec['id']} charge={rec['charge']} mult={rec['multiplicity']}"]
    for s, (x, y, z) in zip(rec["symbols"], rec["geometry_angstrom"]):
        lines.append(f"{s:2s} {x:20.12f} {y:20.12f} {z:20.12f}")
    return "\n".join(lines) + "\n"


def parse_xyz(text: str) -> tuple[list[str], list[list[float]], str]:
    lines = text.splitlines()
    n = int(lines[0].split()[0])
    syms, xyz = [], []
    for line in lines[2:2 + n]:
        p = line.split()
        syms.append(norm_symbol(p[0]))
        xyz.append([float(v) for v in p[1:4]])
    return syms, xyz, lines[1] if len(lines) > 1 else ""


def min_distance(geom: list[list[float]]) -> float:
    import itertools
    import math

    best = math.inf
    for a, b in itertools.combinations(geom, 2):
        best = min(best, math.dist(a, b))
    return best
