#!/usr/bin/env python
"""Build trans-PdCl2(PMe2Ph)2 (43 atoms) from standard bond lengths and angles (not
optimized; a benchmark structure): writes bench/molecules/pdcl2pme2ph2.xyz (Å).

Pd-Cl 2.30, Pd-P 2.33, P-C 1.83, C-H 1.09, aromatic C-C 1.395, C-H(ar) 1.085 Å; Pd-P-C
115 deg, tetrahedral methyls, phenyl ring planes perpendicular to the Pd-P-C plane.
"""
from pathlib import Path

import numpy as np


def unit(v):
    return v / np.linalg.norm(v)


def perp(d, hint):
    return unit(np.cross(d, hint) if np.linalg.norm(np.cross(d, hint)) > 1e-6 else np.cross(d, [1.0, 0, 0]))


def methyl(c, d):
    f1 = perp(d, [0.0, 0.0, 1.0])
    f2 = np.cross(d, f1)
    a = np.radians(180 - 109.47)
    return [c + 1.09 * (np.cos(a) * d + np.sin(a) * (np.cos(p) * f1 + np.sin(p) * f2)) for p in np.radians([0, 120, 240])]


def phenyl(c, d, plane):
    f1 = unit(plane - np.dot(plane, d) * d)
    rc = c + 1.395 * d
    ring = [rc + 1.395 * (-np.cos(t) * d + np.sin(t) * f1) for t in np.radians(np.arange(0, 360, 60))]
    hs = [rc + 2.48 * (-np.cos(t) * d + np.sin(t) * f1) for t in np.radians(np.arange(60, 360, 60))]
    return ring, hs


atoms = [("Pd", np.zeros(3)), ("Cl", np.array([2.30, 0, 0])), ("Cl", np.array([-2.30, 0, 0]))]
for s, off in ((1, 0.0), (-1, 60.0)):
    p = np.array([0, s * 2.33, 0.0])
    atoms.append(("P", p))
    u = np.array([0, s, 0.0])
    alpha = np.radians(180 - 115)
    for k, phi in enumerate(np.radians(np.array([0, 120, 240]) + off)):
        d = unit(np.cos(alpha) * u + np.sin(alpha) * (np.cos(phi) * np.array([1.0, 0, 0]) + np.sin(phi) * np.array([0, 0, 1.0])))
        c = p + 1.83 * d
        if k == 0:
            ring, hs = phenyl(c, d, np.cross(d, u))
            atoms += [("C", x) for x in ring] + [("H", x) for x in hs]
        else:
            atoms.append(("C", c))
            atoms += [("H", x) for x in methyl(c, d)]
xyz = np.array([x for _, x in atoms])
dist = np.linalg.norm(xyz[:, None] - xyz[None], axis=-1) + np.eye(len(atoms)) * 99
print(len(atoms), "atoms; shortest distance", dist.min().round(3), "; shortest H-H", min(dist[i, j] for i in range(len(atoms)) for j in range(i) if atoms[i][0] == atoms[j][0] == "H").round(3))
out = Path(__file__).resolve().parent / "pdcl2pme2ph2.xyz"
out.write_text(f"{len(atoms)}\ntrans-PdCl2(PMe2Ph)2, built from standard geometry (not optimized)\n"
               + "".join(f"{s:2s} {x[0]:12.6f} {x[1]:12.6f} {x[2]:12.6f}\n" for s, x in atoms))
print("wrote", out)
