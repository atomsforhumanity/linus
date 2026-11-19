#!/usr/bin/env python
"""Built inorganic and heavy-element molecules: initial geometries for GFN2-xTB.

    python inorganic.py OUTDIR    # OUTDIR/<id>.xyz + OUTDIR/inputs.jsonl (pre-xtb)

Two groups, all ours (no external data; public):
- transition-metal complexes (3d/4d/5d): octahedral aqua and ammine ions, tetrahedral and
  square-planar halides, carbonyls, oxo anions, metallocenes, cyanides, with the spin state of the
  common high- or low-spin configuration (multiplicity given per entry);
- heavy main-group molecules with def2 ECPs (Rb-Rn): halides, hydrides, methyl compounds,
  oxides, xenon fluorides, polyhalide ions.

Geometries start from ideal polyhedra (VSEPR directions) with ligand fragments placed along each
direction at a distance from covalent radii, then go through GFN2-xTB (build/xtb_opt.py).
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import Z, safe_id  # noqa: E402

# covalent radii (Pyykko & Atsumi 2009, single bond, angstrom)
RCOV = {"H": 0.32, "Li": 1.33, "Be": 1.02, "B": 0.85, "C": 0.75, "N": 0.71, "O": 0.63, "F": 0.64, "Na": 1.55,
        "Mg": 1.39, "Al": 1.26, "Si": 1.16, "P": 1.11, "S": 1.03, "Cl": 0.99, "K": 1.96, "Ca": 1.71, "Sc": 1.48,
        "Ti": 1.36, "V": 1.34, "Cr": 1.22, "Mn": 1.19, "Fe": 1.16, "Co": 1.11, "Ni": 1.10, "Cu": 1.12, "Zn": 1.18,
        "Ga": 1.24, "Ge": 1.21, "As": 1.21, "Se": 1.16, "Br": 1.14, "Kr": 1.17, "Rb": 2.10, "Sr": 1.85, "Y": 1.63,
        "Zr": 1.54, "Nb": 1.47, "Mo": 1.38, "Tc": 1.28, "Ru": 1.25, "Rh": 1.25, "Pd": 1.20, "Ag": 1.28, "Cd": 1.36,
        "In": 1.42, "Sn": 1.40, "Sb": 1.40, "Te": 1.36, "I": 1.33, "Xe": 1.31, "Cs": 2.32, "Ba": 1.96, "La": 1.80,
        "Hf": 1.52, "Ta": 1.46, "W": 1.37, "Re": 1.31, "Os": 1.29, "Ir": 1.22, "Pt": 1.23, "Au": 1.24, "Hg": 1.33,
        "Tl": 1.44, "Pb": 1.44, "Bi": 1.51, "Po": 1.45, "At": 1.47, "Rn": 1.42}

# ligand fragments: donor atom first at the origin, pointing along +z away from the metal
LIGANDS = {
    "F": [("F", 0, 0, 0)], "Cl": [("Cl", 0, 0, 0)], "Br": [("Br", 0, 0, 0)], "I": [("I", 0, 0, 0)],
    "H": [("H", 0, 0, 0)], "O": [("O", 0, 0, 0)], "S": [("S", 0, 0, 0)],
    "OH2": [("O", 0, 0, 0), ("H", 0.76, 0, 0.59), ("H", -0.76, 0, 0.59)],
    "NH3": [("N", 0, 0, 0), ("H", 0.94, 0, 0.38), ("H", -0.47, 0.81, 0.38), ("H", -0.47, -0.81, 0.38)],
    "CO": [("C", 0, 0, 0), ("O", 0, 0, 1.14)],
    "CN": [("C", 0, 0, 0), ("N", 0, 0, 1.16)],
    "NCS": [("N", 0, 0, 0), ("C", 0, 0, 1.18), ("S", 0, 0, 2.80)],
    "CH3": [("C", 0, 0, 0), ("H", 1.03, 0, 0.36), ("H", -0.51, 0.89, 0.36), ("H", -0.51, -0.89, 0.36)],
    "OH": [("O", 0, 0, 0), ("H", 0.90, 0, 0.45)],
    "PH3": [("P", 0, 0, 0), ("H", 1.19, 0, 0.55), ("H", -0.60, 1.03, 0.55), ("H", -0.60, -1.03, 0.55)],
}
# donor-metal distance offset (angstrom) added to the covalent-radius sum per ligand kind
DIST_EXTRA = {"OH2": 0.25, "NH3": 0.2, "CO": 0.05, "CN": 0.05, "PH3": 0.1}

SQ = 1 / math.sqrt(2)
SHAPES = {
    1: [(0, 0, 1)],
    "linear": [(0, 0, 1), (0, 0, -1)],
    "bent": [(math.sin(math.radians(52)), 0, math.cos(math.radians(52))),
             (-math.sin(math.radians(52)), 0, math.cos(math.radians(52)))],
    "trigonal": [(1, 0, 0), (-0.5, math.sqrt(3) / 2, 0), (-0.5, -math.sqrt(3) / 2, 0)],
    "pyramidal": [(0.94, 0, -0.34), (-0.47, 0.81, -0.34), (-0.47, -0.81, -0.34)],
    "tetrahedral": [(1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)],
    "seesaw": [(0, 0, 1), (0, 0, -1), (1, 0, 0), (-0.5, math.sqrt(3) / 2, 0)],
    "square": [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0)],
    "tbp": [(0, 0, 1), (0, 0, -1), (1, 0, 0), (-0.5, math.sqrt(3) / 2, 0), (-0.5, -math.sqrt(3) / 2, 0)],
    "sqpyr": [(0, 0, 1), (1, 0, -0.1), (-1, 0, -0.1), (0, 1, -0.1), (0, -1, -0.1)],
    "octahedral": [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)],
    "pbp": [(0, 0, 1), (0, 0, -1)] + [(math.cos(2 * math.pi * k / 5), math.sin(2 * math.pi * k / 5), 0) for k in range(5)],
}


def rot_to(v):
    """Rotation taking +z to the unit vector v."""
    v = np.asarray(v, float) / np.linalg.norm(v)
    z = np.array([0, 0, 1.0])
    c = float(z @ v)
    if c > 1 - 1e-12:
        return np.eye(3)
    if c < -1 + 1e-12:
        return np.diag([1, -1, -1.0])
    ax = np.cross(z, v)
    s = np.linalg.norm(ax)
    ax /= s
    K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
    return np.eye(3) + s * K + (1 - c) * K @ K


def complex_(metal: str, ligs: list[str], shape) -> tuple[list[str], np.ndarray]:
    dirs = SHAPES[shape]
    if len(dirs) != len(ligs):
        raise ValueError(f"{metal}: {len(ligs)} ligands for shape {shape}")
    syms, pos = [metal], [np.zeros(3)]
    for lig, d in zip(ligs, dirs):
        frag = LIGANDS[lig]
        r = RCOV[metal] + RCOV[frag[0][0]] + DIST_EXTRA.get(lig, 0.0)
        R = rot_to(d)
        u = np.asarray(d, float) / np.linalg.norm(d)
        for s, x, y, zc in frag:
            syms.append(s)
            pos.append(u * r + R @ np.array([x, y, zc]))
    return syms, np.array(pos)


def metallocene(metal: str, d: float = 1.65, ring: str = "Cp") -> tuple[list[str], np.ndarray]:
    syms, pos = [metal], [np.zeros(3)]
    n = 5 if ring == "Cp" else 6
    rc = 1.20 if n == 5 else 1.40
    for sgn in (1, -1):
        for k in range(n):
            a = 2 * math.pi * k / n + (math.pi / n if sgn < 0 and n == 5 else 0)
            syms.append("C")
            pos.append(np.array([rc * math.cos(a), rc * math.sin(a), sgn * d]))
            syms.append("H")
            pos.append(np.array([(rc + 1.08) * math.cos(a), (rc + 1.08) * math.sin(a), sgn * d]))
    return syms, np.array(pos)


def atom(sym: str) -> tuple[list[str], np.ndarray]:
    return [sym], np.zeros((1, 3))


def diatomic(a: str, b: str, r: float | None = None) -> tuple[list[str], np.ndarray]:
    r = r or RCOV[a] + RCOV[b]
    return [a, b], np.array([[0, 0, 0], [0, 0, r]], float)


# (id, builder, charge, multiplicity, tags)
TM = [
    # aqua ions: high-spin 3d (and Zn), low-spin 4d/5d
    ("Ti_aq6_3p", lambda: complex_("Ti", ["OH2"] * 6, "octahedral"), 3, 2),
    ("V_aq6_3p", lambda: complex_("V", ["OH2"] * 6, "octahedral"), 3, 3),
    ("Cr_aq6_3p", lambda: complex_("Cr", ["OH2"] * 6, "octahedral"), 3, 4),
    ("Mn_aq6_2p", lambda: complex_("Mn", ["OH2"] * 6, "octahedral"), 2, 6),
    ("Fe_aq6_2p", lambda: complex_("Fe", ["OH2"] * 6, "octahedral"), 2, 5),
    ("Fe_aq6_3p", lambda: complex_("Fe", ["OH2"] * 6, "octahedral"), 3, 6),
    ("Co_aq6_2p", lambda: complex_("Co", ["OH2"] * 6, "octahedral"), 2, 4),
    ("Ni_aq6_2p", lambda: complex_("Ni", ["OH2"] * 6, "octahedral"), 2, 3),
    ("Cu_aq6_2p", lambda: complex_("Cu", ["OH2"] * 6, "octahedral"), 2, 2),
    ("Zn_aq6_2p", lambda: complex_("Zn", ["OH2"] * 6, "octahedral"), 2, 1),
    ("Ru_aq6_2p", lambda: complex_("Ru", ["OH2"] * 6, "octahedral"), 2, 1),
    ("Ru_aq6_3p", lambda: complex_("Ru", ["OH2"] * 6, "octahedral"), 3, 2),
    ("Rh_aq6_3p", lambda: complex_("Rh", ["OH2"] * 6, "octahedral"), 3, 1),
    # ammines
    ("Co_nh3_6_3p", lambda: complex_("Co", ["NH3"] * 6, "octahedral"), 3, 1),
    ("Cr_nh3_6_3p", lambda: complex_("Cr", ["NH3"] * 6, "octahedral"), 3, 4),
    ("Ni_nh3_6_2p", lambda: complex_("Ni", ["NH3"] * 6, "octahedral"), 2, 3),
    ("Ru_nh3_6_2p", lambda: complex_("Ru", ["NH3"] * 6, "octahedral"), 2, 1),
    ("Ru_nh3_6_3p", lambda: complex_("Ru", ["NH3"] * 6, "octahedral"), 3, 2),
    ("Ir_nh3_6_3p", lambda: complex_("Ir", ["NH3"] * 6, "octahedral"), 3, 1),
    ("Cu_nh3_4_2p", lambda: complex_("Cu", ["NH3"] * 4, "square"), 2, 2),
    ("Ag_nh3_2_1p", lambda: complex_("Ag", ["NH3"] * 2, "linear"), 1, 1),
    # tetrahedral halides
    ("Fe_cl4_1m", lambda: complex_("Fe", ["Cl"] * 4, "tetrahedral"), -1, 6),
    ("Fe_cl4_2m", lambda: complex_("Fe", ["Cl"] * 4, "tetrahedral"), -2, 5),
    ("Co_cl4_2m", lambda: complex_("Co", ["Cl"] * 4, "tetrahedral"), -2, 4),
    ("Ni_cl4_2m", lambda: complex_("Ni", ["Cl"] * 4, "tetrahedral"), -2, 3),
    ("Cu_cl4_2m", lambda: complex_("Cu", ["Cl"] * 4, "tetrahedral"), -2, 2),
    ("Mn_cl4_2m", lambda: complex_("Mn", ["Cl"] * 4, "tetrahedral"), -2, 6),
    ("Zn_cl4_2m", lambda: complex_("Zn", ["Cl"] * 4, "tetrahedral"), -2, 1),
    ("Ti_cl4", lambda: complex_("Ti", ["Cl"] * 4, "tetrahedral"), 0, 1),
    ("V_cl4", lambda: complex_("V", ["Cl"] * 4, "tetrahedral"), 0, 2),
    ("Zr_cl4", lambda: complex_("Zr", ["Cl"] * 4, "tetrahedral"), 0, 1),
    ("Hf_cl4", lambda: complex_("Hf", ["Cl"] * 4, "tetrahedral"), 0, 1),
    ("Ti_cl3", lambda: complex_("Ti", ["Cl"] * 3, "trigonal"), 0, 2),
    ("Sc_f3", lambda: complex_("Sc", ["F"] * 3, "trigonal"), 0, 1),
    ("Y_cl3", lambda: complex_("Y", ["Cl"] * 3, "trigonal"), 0, 1),
    ("La_cl3", lambda: complex_("La", ["Cl"] * 3, "trigonal"), 0, 1),
    # square planar d8
    ("Pd_cl4_2m", lambda: complex_("Pd", ["Cl"] * 4, "square"), -2, 1),
    ("Pt_cl4_2m", lambda: complex_("Pt", ["Cl"] * 4, "square"), -2, 1),
    ("Au_cl4_1m", lambda: complex_("Au", ["Cl"] * 4, "square"), -1, 1),
    ("Ni_cn4_2m", lambda: complex_("Ni", ["CN"] * 4, "square"), -2, 1),
    ("Pt_cl2nh3_2_cis", lambda: complex_("Pt", ["Cl", "NH3", "Cl", "NH3"], "square"), 0, 1),
    ("Rh_co2cl2_1m", lambda: complex_("Rh", ["CO", "CO", "Cl", "Cl"], "square"), -1, 1),
    ("Ir_co_cl_ph3_2", lambda: complex_("Ir", ["CO", "Cl", "PH3", "PH3"], "square"), 0, 1),
    # octahedral halides and cyanides
    ("Pt_cl6_2m", lambda: complex_("Pt", ["Cl"] * 6, "octahedral"), -2, 1),
    ("Ir_cl6_2m", lambda: complex_("Ir", ["Cl"] * 6, "octahedral"), -2, 2),
    ("Os_cl6_2m", lambda: complex_("Os", ["Cl"] * 6, "octahedral"), -2, 3),
    ("Re_cl6_2m", lambda: complex_("Re", ["Cl"] * 6, "octahedral"), -2, 4),
    ("Mo_cl6_1m", lambda: complex_("Mo", ["Cl"] * 6, "octahedral"), -1, 2),
    ("W_f6", lambda: complex_("W", ["F"] * 6, "octahedral"), 0, 1),
    ("Mo_f6", lambda: complex_("Mo", ["F"] * 6, "octahedral"), 0, 1),
    ("Fe_cn6_4m", lambda: complex_("Fe", ["CN"] * 6, "octahedral"), -4, 1),
    ("Fe_cn6_3m", lambda: complex_("Fe", ["CN"] * 6, "octahedral"), -3, 2),
    ("Co_cn6_3m", lambda: complex_("Co", ["CN"] * 6, "octahedral"), -3, 1),
    ("Cr_ncs6_3m", lambda: complex_("Cr", ["NCS"] * 6, "octahedral"), -3, 4),
    # linear d10
    ("Cu_cl2_1m", lambda: complex_("Cu", ["Cl"] * 2, "linear"), -1, 1),
    ("Ag_cl2_1m", lambda: complex_("Ag", ["Cl"] * 2, "linear"), -1, 1),
    ("Au_cl2_1m", lambda: complex_("Au", ["Cl"] * 2, "linear"), -1, 1),
    ("Au_cn2_1m", lambda: complex_("Au", ["CN"] * 2, "linear"), -1, 1),
    ("Hg_cl2", lambda: complex_("Hg", ["Cl"] * 2, "linear"), 0, 1),
    ("Hg_me2", lambda: complex_("Hg", ["CH3"] * 2, "linear"), 0, 1),
    ("Cd_me2", lambda: complex_("Cd", ["CH3"] * 2, "linear"), 0, 1),
    ("Zn_me2", lambda: complex_("Zn", ["CH3"] * 2, "linear"), 0, 1),
    ("Au_me_ph3", lambda: complex_("Au", ["CH3", "PH3"], "linear"), 0, 1),
    # carbonyls
    ("Cr_co6", lambda: complex_("Cr", ["CO"] * 6, "octahedral"), 0, 1),
    ("Mo_co6", lambda: complex_("Mo", ["CO"] * 6, "octahedral"), 0, 1),
    ("W_co6", lambda: complex_("W", ["CO"] * 6, "octahedral"), 0, 1),
    ("V_co6", lambda: complex_("V", ["CO"] * 6, "octahedral"), 0, 2),
    ("V_co6_1m", lambda: complex_("V", ["CO"] * 6, "octahedral"), -1, 1),
    ("Fe_co5", lambda: complex_("Fe", ["CO"] * 5, "tbp"), 0, 1),
    ("Ru_co5", lambda: complex_("Ru", ["CO"] * 5, "tbp"), 0, 1),
    ("Os_co5", lambda: complex_("Os", ["CO"] * 5, "tbp"), 0, 1),
    ("Mn_co5", lambda: complex_("Mn", ["CO"] * 5, "sqpyr"), 0, 2),
    ("Mn_co5_1m", lambda: complex_("Mn", ["CO"] * 5, "tbp"), -1, 1),
    ("Re_co5_cl", lambda: complex_("Re", ["CO"] * 5 + ["Cl"], "octahedral"), 0, 1),
    ("Co_co4_1m", lambda: complex_("Co", ["CO"] * 4, "tetrahedral"), -1, 1),
    ("Ni_co4", lambda: complex_("Ni", ["CO"] * 4, "tetrahedral"), 0, 1),
    ("Pd_co4", lambda: complex_("Pd", ["CO"] * 4, "tetrahedral"), 0, 1),
    ("Ni_co3_ph3", lambda: complex_("Ni", ["CO", "CO", "CO", "PH3"], "tetrahedral"), 0, 1),
    # oxo species
    ("Os_o4", lambda: complex_("Os", ["O"] * 4, "tetrahedral"), 0, 1),
    ("Ru_o4", lambda: complex_("Ru", ["O"] * 4, "tetrahedral"), 0, 1),
    ("Mn_o4_1m", lambda: complex_("Mn", ["O"] * 4, "tetrahedral"), -1, 1),
    ("Mn_o4_2m", lambda: complex_("Mn", ["O"] * 4, "tetrahedral"), -2, 2),
    ("Tc_o4_1m", lambda: complex_("Tc", ["O"] * 4, "tetrahedral"), -1, 1),
    ("Re_o4_1m", lambda: complex_("Re", ["O"] * 4, "tetrahedral"), -1, 1),
    ("Cr_o4_2m", lambda: complex_("Cr", ["O"] * 4, "tetrahedral"), -2, 1),
    ("Mo_o4_2m", lambda: complex_("Mo", ["O"] * 4, "tetrahedral"), -2, 1),
    ("W_o4_2m", lambda: complex_("W", ["O"] * 4, "tetrahedral"), -2, 1),
    ("V_o_aq5_2p", lambda: complex_("V", ["O"] + ["OH2"] * 5, "octahedral"), 2, 2),
    ("Ti_o_cl4_2m", lambda: complex_("Ti", ["O", "Cl", "Cl", "Cl", "Cl"], "sqpyr"), -2, 1),
    ("Cr_o2cl2", lambda: complex_("Cr", ["O", "O", "Cl", "Cl"], "tetrahedral"), 0, 1),
    ("Fe_o_cl4_2m", lambda: complex_("Fe", ["O", "Cl", "Cl", "Cl", "Cl"], "sqpyr"), -2, 5),
    # metallocenes and arenes
    ("ferrocene", lambda: metallocene("Fe", 1.65), 0, 1),
    ("ferrocenium", lambda: metallocene("Fe", 1.70), 1, 2),
    ("cobaltocene", lambda: metallocene("Co", 1.72), 0, 2),
    ("cobaltocenium", lambda: metallocene("Co", 1.64), 1, 1),
    ("nickelocene", lambda: metallocene("Ni", 1.82), 0, 3),
    ("chromocene", lambda: metallocene("Cr", 1.80), 0, 3),
    ("vanadocene", lambda: metallocene("V", 1.92), 0, 4),
    ("manganocene_hs", lambda: metallocene("Mn", 2.05), 0, 6),
    ("ruthenocene", lambda: metallocene("Ru", 1.82), 0, 1),
    ("osmocene", lambda: metallocene("Os", 1.83), 0, 1),
    ("dibenzenechromium", lambda: metallocene("Cr", 1.61, ring="Bz"), 0, 1),
    # atoms and diatomics
    ("Sc_atom", lambda: atom("Sc"), 0, 2), ("Ti_atom", lambda: atom("Ti"), 0, 3), ("V_atom", lambda: atom("V"), 0, 4),
    ("Cr_atom", lambda: atom("Cr"), 0, 7), ("Mn_atom", lambda: atom("Mn"), 0, 6), ("Fe_atom", lambda: atom("Fe"), 0, 5),
    ("Co_atom", lambda: atom("Co"), 0, 4), ("Ni_atom", lambda: atom("Ni"), 0, 3), ("Cu_atom", lambda: atom("Cu"), 0, 2),
    ("Zn_atom", lambda: atom("Zn"), 0, 1), ("Pd_atom", lambda: atom("Pd"), 0, 1), ("Pt_atom", lambda: atom("Pt"), 0, 3),
    ("Ag_atom", lambda: atom("Ag"), 0, 2), ("Au_atom", lambda: atom("Au"), 0, 2), ("W_atom", lambda: atom("W"), 0, 5),
    ("Fe_2p_ion", lambda: atom("Fe"), 2, 5), ("Cu_1p_ion", lambda: atom("Cu"), 1, 1),
    ("FeO", lambda: diatomic("Fe", "O", 1.62), 0, 5), ("TiO", lambda: diatomic("Ti", "O", 1.62), 0, 3),
    ("VO", lambda: diatomic("V", "O", 1.59), 0, 4), ("CrO", lambda: diatomic("Cr", "O", 1.62), 0, 5),
    ("CuH", lambda: diatomic("Cu", "H", 1.46), 0, 1), ("FeH", lambda: diatomic("Fe", "H", 1.63), 0, 4),
    ("Cu2", lambda: diatomic("Cu", "Cu", 2.22), 0, 1), ("Ag2", lambda: diatomic("Ag", "Ag", 2.53), 0, 1),
    ("Ni2", lambda: diatomic("Ni", "Ni", 2.15), 0, 3), ("ZnH", lambda: diatomic("Zn", "H", 1.59), 0, 2),
    ("AuH", lambda: diatomic("Au", "H", 1.52), 0, 1), ("PtH", lambda: diatomic("Pt", "H", 1.53), 0, 2),
    ("CuF", lambda: diatomic("Cu", "F", 1.75), 0, 1), ("ScO", lambda: diatomic("Sc", "O", 1.67), 0, 2),
]

HEAVY = [
    # group 1/2 and 13
    ("RbF", lambda: diatomic("Rb", "F", 2.27), 0, 1), ("RbCl", lambda: diatomic("Rb", "Cl", 2.79), 0, 1),
    ("CsF", lambda: diatomic("Cs", "F", 2.35), 0, 1), ("CsOH", lambda: complex_("Cs", ["OH"], 1), 0, 1),
    ("SrCl2", lambda: complex_("Sr", ["Cl"] * 2, "bent"), 0, 1), ("SrO", lambda: diatomic("Sr", "O", 1.92), 0, 1),
    ("BaO", lambda: diatomic("Ba", "O", 1.94), 0, 1), ("BaCl2", lambda: complex_("Ba", ["Cl"] * 2, "bent"), 0, 1),
    ("BaH2", lambda: complex_("Ba", ["H"] * 2, "bent"), 0, 1),
    ("InCl3", lambda: complex_("In", ["Cl"] * 3, "trigonal"), 0, 1), ("InMe3", lambda: complex_("In", ["CH3"] * 3, "trigonal"), 0, 1),
    ("InH", lambda: diatomic("In", "H", 1.84), 0, 1), ("InCl4_1m", lambda: complex_("In", ["Cl"] * 4, "tetrahedral"), -1, 1),
    ("TlCl", lambda: diatomic("Tl", "Cl", 2.48), 0, 1), ("TlH", lambda: diatomic("Tl", "H", 1.87), 0, 1),
    ("TlMe3", lambda: complex_("Tl", ["CH3"] * 3, "trigonal"), 0, 1),
    # group 14
    ("SnH4", lambda: complex_("Sn", ["H"] * 4, "tetrahedral"), 0, 1), ("SnCl4", lambda: complex_("Sn", ["Cl"] * 4, "tetrahedral"), 0, 1),
    ("SnMe4", lambda: complex_("Sn", ["CH3"] * 4, "tetrahedral"), 0, 1), ("SnCl2", lambda: complex_("Sn", ["Cl"] * 2, "bent"), 0, 1),
    ("SnCl3_1m", lambda: complex_("Sn", ["Cl"] * 3, "pyramidal"), -1, 1), ("SnCl6_2m", lambda: complex_("Sn", ["Cl"] * 6, "octahedral"), -2, 1),
    ("PbH4", lambda: complex_("Pb", ["H"] * 4, "tetrahedral"), 0, 1), ("PbCl4", lambda: complex_("Pb", ["Cl"] * 4, "tetrahedral"), 0, 1),
    ("PbMe4", lambda: complex_("Pb", ["CH3"] * 4, "tetrahedral"), 0, 1), ("PbO", lambda: diatomic("Pb", "O", 1.92), 0, 1),
    ("PbCl2", lambda: complex_("Pb", ["Cl"] * 2, "bent"), 0, 1), ("PbMe3_rad", lambda: complex_("Pb", ["CH3"] * 3, "pyramidal"), 0, 2),
    ("SnMe3_rad", lambda: complex_("Sn", ["CH3"] * 3, "pyramidal"), 0, 2),
    # group 15
    ("SbH3", lambda: complex_("Sb", ["H"] * 3, "pyramidal"), 0, 1), ("SbCl3", lambda: complex_("Sb", ["Cl"] * 3, "pyramidal"), 0, 1),
    ("SbCl5", lambda: complex_("Sb", ["Cl"] * 5, "tbp"), 0, 1), ("SbMe3", lambda: complex_("Sb", ["CH3"] * 3, "pyramidal"), 0, 1),
    ("SbF6_1m", lambda: complex_("Sb", ["F"] * 6, "octahedral"), -1, 1),
    ("BiH3", lambda: complex_("Bi", ["H"] * 3, "pyramidal"), 0, 1), ("BiCl3", lambda: complex_("Bi", ["Cl"] * 3, "pyramidal"), 0, 1),
    ("BiMe3", lambda: complex_("Bi", ["CH3"] * 3, "pyramidal"), 0, 1), ("BiF5", lambda: complex_("Bi", ["F"] * 5, "tbp"), 0, 1),
    ("Bi2", lambda: diatomic("Bi", "Bi", 2.66), 0, 1), ("BiH2_rad", lambda: complex_("Bi", ["H"] * 2, "bent"), 0, 2),
    # group 16
    ("TeH2", lambda: complex_("Te", ["H"] * 2, "bent"), 0, 1), ("TeMe2", lambda: complex_("Te", ["CH3"] * 2, "bent"), 0, 1),
    ("TeF6", lambda: complex_("Te", ["F"] * 6, "octahedral"), 0, 1), ("TeCl4", lambda: complex_("Te", ["Cl"] * 4, "seesaw"), 0, 1),
    ("TeO2", lambda: complex_("Te", ["O"] * 2, "bent"), 0, 1), ("Te2", lambda: diatomic("Te", "Te", 2.56), 0, 3),
    ("PoH2", lambda: complex_("Po", ["H"] * 2, "bent"), 0, 1), ("PoCl2", lambda: complex_("Po", ["Cl"] * 2, "bent"), 0, 1),
    ("TeH_rad", lambda: diatomic("Te", "H", 1.66), 0, 2),
    # group 17
    ("I2", lambda: diatomic("I", "I", 2.67), 0, 1), ("ICl", lambda: diatomic("I", "Cl", 2.32), 0, 1),
    ("IBr", lambda: diatomic("I", "Br", 2.47), 0, 1), ("IF", lambda: diatomic("I", "F", 1.91), 0, 1),
    ("I3_1m", lambda: complex_("I", ["I"] * 2, "linear"), -1, 1), ("ICl2_1m", lambda: complex_("I", ["Cl"] * 2, "linear"), -1, 1),
    ("ICl4_1m", lambda: complex_("I", ["Cl"] * 4, "square"), -1, 1), ("IF5", lambda: complex_("I", ["F"] * 5, "sqpyr"), 0, 1),
    ("IF7", lambda: complex_("I", ["F"] * 7, "pbp"), 0, 1), ("IO3_1m", lambda: complex_("I", ["O"] * 3, "pyramidal"), -1, 1),
    ("I_1m", lambda: atom("I"), -1, 1), ("I_atom", lambda: atom("I"), 0, 2), ("I2_1m", lambda: diatomic("I", "I", 3.2), -1, 2),
    ("I2_1p", lambda: diatomic("I", "I", 2.58), 1, 2), ("HI_1p", lambda: diatomic("H", "I", 1.62), 1, 2),
    ("AtH", lambda: diatomic("At", "H", 1.72), 0, 1), ("At2", lambda: diatomic("At", "At", 3.0), 0, 1),
    # group 18
    ("XeF2", lambda: complex_("Xe", ["F"] * 2, "linear"), 0, 1), ("XeF4", lambda: complex_("Xe", ["F"] * 4, "square"), 0, 1),
    ("XeO3", lambda: complex_("Xe", ["O"] * 3, "pyramidal"), 0, 1), ("XeOF4", lambda: complex_("Xe", ["O", "F", "F", "F", "F"], "sqpyr"), 0, 1),
    ("XeF_1p", lambda: diatomic("Xe", "F", 1.89), 1, 1), ("Xe_atom", lambda: atom("Xe"), 0, 1),
    ("Rn_atom", lambda: atom("Rn"), 0, 1), ("RnF2", lambda: complex_("Rn", ["F"] * 2, "linear"), 0, 1),
    ("KrF2", lambda: complex_("Kr", ["F"] * 2, "linear"), 0, 1),
    # group 11/12 with organics
    ("AgMe", lambda: complex_("Ag", ["CH3"], 1), 0, 1), ("CdCl2", lambda: complex_("Cd", ["Cl"] * 2, "linear"), 0, 1),
    ("HgClMe", lambda: complex_("Hg", ["Cl", "CH3"], "linear"), 0, 1), ("Hg2Cl2", lambda: (["Cl", "Hg", "Hg", "Cl"],
                                                                                          np.array([[0, 0, -2.4], [0, 0, 0], [0, 0, 2.53], [0, 0, 4.93]])), 0, 1),
]


def main(argv: list[str]) -> int:
    out = Path(argv[0])
    out.mkdir(parents=True, exist_ok=True)
    inputs = []
    for group, entries in (("tm_built", TM), ("heavy_built", HEAVY)):
        for name, build, q, mult in entries:
            syms, pos = build()
            nel = sum(Z[s] for s in syms) - q
            if (nel + mult - 1) % 2:
                raise ValueError(f"{name}: {nel} electrons with multiplicity {mult}")
            mid = safe_id(f"{group}.{name}")
            xyz = [str(len(syms)), f"{mid} charge={q} mult={mult}"]
            xyz += [f"{s:2s} {x:16.10f} {y:16.10f} {z:16.10f}" for s, (x, y, z) in zip(syms, pos)]
            (out / f"{mid}.xyz").write_text("\n".join(xyz) + "\n")
            inputs.append({"id": mid, "name": name, "set": group, "charge": q, "multiplicity": mult,
                           "optimize": len(syms) > 1, "xyz": f"{mid}.xyz"})
    (out / "inputs.jsonl").write_text("".join(json.dumps(x) + "\n" for x in inputs))
    print(len(inputs), "inputs:", sum(x["set"] == "tm_built" for x in inputs), "TM,",
          sum(x["set"] == "heavy_built" for x in inputs), "heavy main-group")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
