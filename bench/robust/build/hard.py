#!/usr/bin/env python
"""Known-hard SCF cases at fixed, hand-built geometries -> molecules/hard.jsonl (ours; public).

Stretched bonds (singlet and, where it matters, the broken-symmetry-prone triplet), small
HOMO-LUMO gaps (polyacenes, polyenes, C2, Cr2, ozone, carbenes), symmetric radical cations that
tend to break symmetry (He2+, Ne2+, N2+), highly charged anions, hypervalent molecules,
near-degenerate atoms and open-shell diatomics. Geometries are idealized (experimental or
textbook bond lengths, ideal angles), not minima of any method; the point is the SCF.

    python hard.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import MOLDIR, record, write_jsonl  # noqa: E402


def diatomic(a, b, r):
    return [a, b], [[0.0, 0.0, 0.0], [0.0, 0.0, r]]


def linear(syms, rs):
    z, pos = 0.0, [[0.0, 0.0, 0.0]]
    for r in rs:
        z += r
        pos.append([0.0, 0.0, z])
    return syms, pos


def bent(c, l, r, ang):
    h = math.radians(ang) / 2
    return [c, l, l], [[0, 0, 0], [r * math.sin(h), 0, r * math.cos(h)], [-r * math.sin(h), 0, r * math.cos(h)]]


def tetra(c, l, r):
    s = r / math.sqrt(3)
    return [c] + [l] * 4, [[0, 0, 0], [s, s, s], [s, -s, -s], [-s, s, -s], [-s, -s, s]]


def trigonal(c, l, r):
    return [c] + [l] * 3, [[0, 0, 0]] + [[r * math.cos(2 * math.pi * k / 3), r * math.sin(2 * math.pi * k / 3), 0] for k in range(3)]


def octa(c, l, r):
    return [c] + [l] * 6, [[0, 0, 0], [r, 0, 0], [-r, 0, 0], [0, r, 0], [0, -r, 0], [0, 0, r], [0, 0, -r]]


def acene(n):
    """Linear acene with n rings (C4n+2 H2n+4), ideal hexagons, C-C 1.40, C-H 1.08 A."""
    a = 1.40
    syms, pos = [], []
    cs = set()
    for k in range(n):
        cx = k * a * math.sqrt(3)
        for j in range(6):
            ang = math.radians(30 + 60 * j)
            p = (round(cx + a * math.cos(ang), 6), round(a * math.sin(ang), 6))
            cs.add(p)
    cs = sorted(cs)
    for p in cs:
        syms.append("C")
        pos.append([p[0], p[1], 0.0])
    for p in cs:
        nb = [q for q in cs if q != p and math.dist(p, q) < 1.5]
        if len(nb) == 2:
            v = [2 * p[0] - nb[0][0] - nb[1][0], 2 * p[1] - nb[0][1] - nb[1][1]]
            nv = math.hypot(*v)
            syms.append("H")
            pos.append([p[0] + 1.08 * v[0] / nv, p[1] + 1.08 * v[1] / nv, 0.0])
    return syms, pos


def polyene(n):
    """all-trans H-(CH=CH)n-H, C=C 1.35, C-C 1.45, 120 degree angles, planar."""
    syms, pos = [], []
    x, y = 0.0, 0.0
    up = True
    for i in range(2 * n):
        syms.append("C")
        pos.append([x, y, 0.0])
        r = 1.35 if i % 2 == 0 else 1.45
        x += r * math.cos(math.radians(30))
        y += r * math.sin(math.radians(30)) * (1 if up else -1)
        up = not up
    cpos = list(pos)
    for i, p in enumerate(cpos):
        nb = [cpos[j] for j in (i - 1, i + 1) if 0 <= j < len(cpos)]
        if len(nb) == 2:
            v = [2 * p[0] - nb[0][0] - nb[1][0], 2 * p[1] - nb[0][1] - nb[1][1]]
            nv = math.hypot(*v)
            syms.append("H")
            pos.append([p[0] + 1.08 * v[0] / nv, p[1] + 1.08 * v[1] / nv, 0.0])
        else:
            q = nb[0]
            v = [p[0] - q[0], p[1] - q[1]]
            for rot in (60, -60):
                c, s = math.cos(math.radians(rot)), math.sin(math.radians(rot))
                w = [c * v[0] - s * v[1], s * v[0] + c * v[1]]
                nw = math.hypot(*w)
                syms.append("H")
                pos.append([p[0] + 1.08 * w[0] / nw, p[1] + 1.08 * w[1] / nw, 0.0])
    return syms, pos


def ethane(rcc):
    h = 1.09
    t = math.radians(109.47 - 90)
    pos = [[0, 0, 0], [0, 0, rcc]]
    syms = ["C", "C"]
    for k in range(3):
        a = 2 * math.pi * k / 3
        syms.append("H")
        pos.append([h * math.cos(t) * math.cos(a), h * math.cos(t) * math.sin(a), -h * math.sin(t)])
    for k in range(3):
        a = 2 * math.pi * k / 3 + math.pi / 3
        syms.append("H")
        pos.append([h * math.cos(t) * math.cos(a), h * math.cos(t) * math.sin(a), rcc + h * math.sin(t)])
    return syms, pos


CASES = []


def add(name, geom, q, m, *tags):
    CASES.append((name, geom, q, m, tags))


for r in (0.74, 1.2, 1.6, 2.0, 3.0):
    add(f"H2_r{r}", diatomic("H", "H", r), 0, 1, "stretched" if r > 1 else "reference_geometry")
add("H2_r2.0_triplet", diatomic("H", "H", 2.0), 0, 3, "stretched")
for r in (1.10, 1.5, 2.0, 2.5):
    add(f"N2_r{r}", diatomic("N", "N", r), 0, 1, "stretched" if r > 1.2 else "reference_geometry")
for r in (1.41, 2.0, 2.8):
    add(f"F2_r{r}", diatomic("F", "F", r), 0, 1, "stretched" if r > 1.5 else "reference_geometry")
for r in (0.92, 1.5, 2.5):
    add(f"HF_r{r}", diatomic("H", "F", r), 0, 1, "stretched" if r > 1 else "reference_geometry")
for r in (1.56, 4.0, 7.0):
    add(f"LiF_r{r}", diatomic("Li", "F", r), 0, 1, "stretched" if r > 2 else "reference_geometry")
for r in (1.54, 2.2, 3.0):
    add(f"C2H6_rCC{r}", ethane(r), 0, 1, "stretched" if r > 1.6 else "reference_geometry")
add("C2H6_rCC3.0_triplet", ethane(3.0), 0, 3, "stretched")
add("H2O_rOH1.6", (["O", "H", "H"], [[0, 0, 0], [1.6, 0, 0], [0.96 * math.cos(math.radians(104.5)), 0.96 * math.sin(math.radians(104.5)), 0]]), 0, 1, "stretched")
add("H2O_sym_stretch_1.8", bent("O", "H", 1.8, 104.5), 0, 1, "stretched")
add("H4_square", (["H"] * 4, [[0, 0, 0], [1.2, 0, 0], [1.2, 1.2, 0], [0, 1.2, 0]]), 0, 1, "near_degenerate")
add("H4_square_triplet", (["H"] * 4, [[0, 0, 0], [1.2, 0, 0], [1.2, 1.2, 0], [0, 1.2, 0]]), 0, 3, "near_degenerate")
add("H6_ring_r1.5", (["H"] * 6, [[1.5 * math.cos(k * math.pi / 3), 1.5 * math.sin(k * math.pi / 3), 0] for k in range(6)]), 0, 1, "stretched")
add("C2_singlet", diatomic("C", "C", 1.243), 0, 1, "small_gap", "multireference")
add("C2_triplet", diatomic("C", "C", 1.31), 0, 3, "small_gap")
add("Be2", diatomic("Be", "Be", 2.45), 0, 1, "small_gap", "near_degenerate")
add("Cr2_1.68", diatomic("Cr", "Cr", 1.68), 0, 1, "small_gap", "multireference")
add("Cr2_2.5", diatomic("Cr", "Cr", 2.5), 0, 1, "stretched", "multireference")
add("Mn2_hs", diatomic("Mn", "Mn", 3.4), 0, 11, "high_spin")
add("O2_triplet", diatomic("O", "O", 1.208), 0, 3)
add("O2_singlet_cs", diatomic("O", "O", 1.216), 0, 1, "small_gap")
add("O3", bent("O", "O", 1.272, 116.8), 0, 1, "small_gap", "multireference")
add("O3_triplet", bent("O", "O", 1.30, 110), 0, 3, "small_gap")
add("NO2", bent("N", "O", 1.197, 134.1), 0, 2)
add("ClO2", bent("Cl", "O", 1.47, 117.4), 0, 2)
add("NO", diatomic("N", "O", 1.151), 0, 2)
add("CN_rad", diatomic("C", "N", 1.172), 0, 2)
add("CH2_triplet", bent("C", "H", 1.075, 134), 0, 3)
add("CH2_singlet", bent("C", "H", 1.11, 102), 0, 1, "small_gap")
add("SiH2_singlet", bent("Si", "H", 1.51, 92), 0, 1)
add("NH_triplet", diatomic("N", "H", 1.036), 0, 3)
add("CH_doublet", diatomic("C", "H", 1.12), 0, 2)
add("B_atom", (["B"], [[0, 0, 0]]), 0, 2, "near_degenerate")
add("C_atom_triplet", (["C"], [[0, 0, 0]]), 0, 3, "near_degenerate")
add("O_atom_triplet", (["O"], [[0, 0, 0]]), 0, 3, "near_degenerate")
add("N_atom_quartet", (["N"], [[0, 0, 0]]), 0, 4)
add("Si_atom_triplet", (["Si"], [[0, 0, 0]]), 0, 3, "near_degenerate")
add("Fe_atom_quintet", (["Fe"], [[0, 0, 0]]), 0, 5, "near_degenerate")
add("He2p", diatomic("He", "He", 1.08), 1, 2, "symmetry_breaking")
add("Ne2p", diatomic("Ne", "Ne", 1.75), 1, 2, "symmetry_breaking")
add("Ar2p", diatomic("Ar", "Ar", 2.43), 1, 2, "symmetry_breaking")
add("N2p", diatomic("N", "N", 1.116), 1, 2)
add("H2p_r3", diatomic("H", "H", 3.0), 1, 2, "symmetry_breaking", "stretched")
add("CO_p", diatomic("C", "O", 1.115), 1, 2)
add("SO4_2m", tetra("S", "O", 1.49), -2, 1, "multiply_charged")
add("PO4_3m", tetra("P", "O", 1.54), -3, 1, "multiply_charged")
add("CO3_2m", trigonal("C", "O", 1.29), -2, 1, "multiply_charged")
add("SiF6_2m", octa("Si", "F", 1.69), -2, 1, "multiply_charged")
add("AlF6_3m", octa("Al", "F", 1.80), -3, 1, "multiply_charged")
add("C2_2m", diatomic("C", "C", 1.21), -2, 1, "multiply_charged")
add("SF6", octa("S", "F", 1.56), 0, 1, "hypervalent")
add("PF5", (["P", "F", "F", "F", "F", "F"], [[0, 0, 0], [0, 0, 1.58], [0, 0, -1.58], [1.53, 0, 0],
                                             [-0.765, 1.325, 0], [-0.765, -1.325, 0]]), 0, 1, "hypervalent")
add("ClF3", (["Cl", "F", "F", "F"], [[0, 0, 0], [1.60, 0, 0], [0, 1.70, 0.0], [0, -1.70, 0.0]]), 0, 1, "hypervalent")
add("benzene", acene(1), 0, 1, "reference_geometry")
add("naphthalene", acene(2), 0, 1)
add("anthracene", acene(3), 0, 1, "small_gap")
add("tetracene", acene(4), 0, 1, "small_gap")
add("pentacene", acene(5), 0, 1, "small_gap")
add("pentacene_triplet", acene(5), 0, 3, "small_gap")
add("hexacene", acene(6), 0, 1, "small_gap", "multireference")
add("tetracene_radcat", acene(4), 1, 2, "small_gap")
add("tetracene_radanion", acene(4), -1, 2, "small_gap")
add("octatetraene", polyene(4), 0, 1, "small_gap")
add("C16H18_polyene", polyene(8), 0, 1, "small_gap")
add("C24H26_polyene", polyene(12), 0, 1, "small_gap")
add("C16H18_polyene_triplet", polyene(8), 0, 3, "small_gap")
# p-benzyne: benzene minus the two para H atoms (singlet biradical) and its triplet
bz = acene(1)
para = [i for i, s in enumerate(bz[0]) if s == "H"]
hs = sorted(para, key=lambda i: bz[1][i][0])
keep = [i for i in range(len(bz[0])) if i not in (hs[0], hs[-1])]
pb = ([bz[0][i] for i in keep], [bz[1][i] for i in keep])
add("p_benzyne_singlet", pb, 0, 1, "small_gap", "multireference")
add("p_benzyne_triplet", pb, 0, 3, "small_gap")


def main() -> int:
    recs = [record("hard", None, name, *geom, q, m, source="built (bench/robust/build/hard.py)",
                   geometry_level="idealized, fixed", tags=["hard", *tags]) for name, geom, q, m, tags in CASES]
    write_jsonl(MOLDIR / "hard.jsonl", recs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
