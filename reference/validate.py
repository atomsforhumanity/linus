#!/usr/bin/env python
"""Validation matrix: linus against PySCF 2.15, Psi4 1.11 and ORCA 6.0.0.

Energies, dipoles, analytic gradients and analytic Hessians for the cases in ``CASES``
(molecules x methods x basis sets x J/K x dispersion). Run it with the reference
environment's Python, from the repository (or a worktree of it):

    .refenv/bin/python reference/validate.py              # the whole matrix
    .refenv/bin/python reference/validate.py --quick      # the subset for every merge (~4 min)
    .refenv/bin/python reference/validate.py --cases 'water.*' --refs all
    .refenv/bin/python reference/validation_summary.py    # print the matrix

Reference results are expensive and rarely change, so they are stored (committed) under
``reference/validation/runs/<case>/<run>.json`` and recomputed only when missing, when their
input specification changed, or with ``--refs all``. linus is rerun every time (``--linus
reuse`` skips it). Raw program outputs go to ``reference/out/validation/`` (git-ignored).
``reference/validation/results.json`` holds every comparison with its tolerance and verdict,
the linus commit and the program versions. Nothing written there names the host.

``--quick`` (the per-merge check) writes nothing that git tracks: its linus runs and its
results.json go to ``reference/out/validation/quick/`` (read back in place of the stored ones)
and the stored references are compared against. ``--quick --store`` writes into
``reference/validation/`` as a full run does.

Every program gets the same bohr coordinates (xyz in angstrom, CODATA 2022 via qcref); see
``run_ref.py`` for each reference program's settings and reference/validation/README.md for
the comparisons, tolerances and the explained differences.

Each calculation runs in its own memory-capped systemd scope (``systemd-run --user --scope
-p MemoryMax=...``) pinned to one CCD with ``~/.local/bin/slice`` (``--lanes``); one
calculation per lane at a time.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import datetime as _dt
import fnmatch
import hashlib
import json
import os
import queue
import re
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
import run_ref  # noqa: E402  (settings and paths of the reference runner)

VALDIR = HERE / "validation"
RUNS_DIR = VALDIR / "runs"  # stored (committed) runs
RESULTS = VALDIR / "results.json"
RAW_DIR = REPO / "reference" / "out" / "validation"
# --quick without --store: linus runs and results.json go here (git-ignored), not into VALDIR
QUICK_DIR = RAW_DIR / "quick"
WRITE_DIR = RUNS_DIR  # where this invocation stores runs (main() sets QUICK_DIR / "runs" for --quick)
LINUS_BIN = Path(os.environ.get("LINUS_BIN", REPO / "target" / "release" / "linus"))
SLICE = Path.home() / ".local" / "bin" / "slice"
PYTHON = run_ref.REFENV / "bin" / "python"

# Bump when the reference settings change in a way the stored spec does not capture
# (e.g. a tighter SCF threshold inside run_ref.py): every reference run is then redone.
REF_SETTINGS_VERSION = 2
AUX = "def2-universal-jkfit"  # the one auxiliary basis (ORCA def2/JK, the same functions)
# linus settings for every run: tightest SCF preset (max |F_ai| < 1e-10, dE < 1e-12),
# integral screening 1e-14 (the references use 1e-14), CPHF residual 1e-9
LINUS_SCF = {"scf": {"convergence": "extreme", "max_iter": 300}, "jk": {"screening": 1e-14}, "response": {"tol": 1e-9}}
LINUS_ENV: dict[str, str] = {}  # (the CPHF residual was the environment variable LINUS_CPHF_TOL before)
ORCA_NUMFREQ_STEP = 0.002  # bohr; ORCA's default 0.005 leaves 8e-6 Eh/bohr^2 truncation error (README)
ORCA_NUMGRAD_STEP = 0.001  # bohr, central differences of energies

# ------------------------------------------------------------------ the matrix


@dataclass(frozen=True)
class Mol:
    xyz: str  # relative to the repository
    charge: int
    mult: int
    label: str


MOLECULES = {
    "water": Mol("tests/molecules/water.xyz", 0, 1, "H2O"),
    "nh3": Mol("tests/molecules/nh3.xyz", 0, 1, "NH3 (distorted)"),
    "ch3sh": Mol("tests/molecules/ch3sh.xyz", 0, 1, "CH3SH (S, second row)"),
    # hcn_cl.xyz has 31 electrons: the closed-shell cation ClCNH+ (Cl; charge-dependent
    # dipole origin and D4 charges)
    "clcnh+": Mol("tests/molecules/hcn_cl.xyz", 1, 1, "ClCNH+ (Cl, cation)"),
    "nh3+": Mol("tests/molecules/nh3.xyz", 1, 2, "NH3+ doublet (open shell)"),
    "ch3sh+": Mol("tests/molecules/ch3sh.xyz", 1, 2, "CH3SH+ doublet (open shell, S)"),
    "la18": Mol("bench/molecules/la18.xyz", 0, 1, "lactide C6H8O4, 18 atoms"),
    # effective core potentials (def2-ECP, Rb-Rn)
    "hi": Mol("tests/molecules/hi.xyz", 0, 1, "HI (I: ECP28MWB)"),
    "ch3i": Mol("tests/molecules/ch3i.xyz", 0, 1, "CH3I (I: ECP28MWB, distorted)"),
    "ch2i": Mol("tests/molecules/ch2i.xyz", 0, 2, "CH2I radical doublet (I: ECP28MWB)"),
    "agcl": Mol("tests/molecules/agcl.xyz", 0, 1, "AgCl (Ag: ECP28MWB)"),
    "ptcl4--": Mol("tests/molecules/ptcl4.xyz", -2, 1, "PtCl4 2- (Pt: ECP60MWB)"),
    "au2": Mol("tests/molecules/au2.xyz", 0, 1, "Au2 (ECP60MWB)"),
    "au2+": Mol("tests/molecules/au2.xyz", 1, 2, "Au2+ doublet (ECP60MWB)"),
    "cscl": Mol("tests/molecules/cscl.xyz", 0, 1, "CsCl (Cs: ECP46MWB)"),
    "pdcl2ph3": Mol("tests/molecules/pdcl2ph3_2.xyz", 0, 1, "trans-PdCl2(PH3)2 (Pd: ECP28MWB), 11 atoms"),
}
# molecules with ECP atoms (def2 basis sets put ECPs on Rb-Rn)
ECP_MOLS = {"hi", "ch3i", "ch2i", "agcl", "ptcl4--", "au2", "au2+", "cscl", "pdcl2ph3"}

LDA = {"svwn-rpa"}


@dataclass(frozen=True)
class Case:
    mol: str
    method: str  # run_ref canonical name: hf, svwn-rpa, pbe, b3lyp, tpss, r2scan, wb97x
    basis: str
    calc: str = "hessian"  # highest derivative: energy, gradient, hessian
    jk: str = "exact"  # exact | df (RI-JK with def2-universal-jkfit in every program)
    cart: bool = False
    disp: str | None = None  # d3bj | d4
    level: int = 9  # PySCF grid level shared by linus and PySCF: energies, dipoles, gradients
    # PySCF grid level for DFT Hessians: PySCF's XC Hessian costs ~3 min for water and over
    # 15 min for CH3SH on level 9; level 5 (5x fewer points) keeps the matrix to hours
    hlevel: int = 5
    quick: bool = False
    psi4_fine: bool = True  # Psi4 on 150 x 974 too (reference triangle at large grids)
    note: str = ""
    # implicit solvent "<model>:<solvent>" (cpcm, iefpcm: linus's `pyscf` PCM preset; smd), with
    # the solvent's permittivity from SOLVENT_EPS given to both programs; PySCF only
    solvent: str | None = None
    # RI range-separated exchange: PySCF patched to Psi4's wK fit, the scheme linus uses
    # (energies only: PySCF's DF gradient fits the long-range exchange with the erf metric)
    wk: bool = False
    # reference programs: "all", or "pyscf" for features this harness runs in PySCF only
    # (implicit solvent, VV10, wK); the Psi4 and ORCA cells then read n/a
    refs: str = "all"

    @property
    def id(self) -> str:
        m = self.method + (f"-{self.disp}" if self.disp else "")
        extra = (f".{self.solvent.replace(':', '-')}" if self.solvent else "") + (".wk" if self.wk else "")
        return f"{self.mol}.{m}.{self.basis.replace('*', 's')}{'-cart' if self.cart else ''}.{self.jk}{extra}"

    @property
    def dft(self) -> bool:
        return self.method != "hf"

    @property
    def open_shell(self) -> bool:
        return MOLECULES[self.mol].mult != 1


def disp_of(case: Case) -> str | None:
    """The case's dispersion model: explicit, or implied by the method name (wB97X-D3BJ)."""
    return case.disp or run_ref.IMPLIED_DISPERSION.get(case.method)


RANGE_SEPARATED = {"wb97x", "wb97x-d3bj", "cam-b3lyp", "wb97x-v", "wb97m-v"}
# implicit-solvent cases: the permittivity both programs get (PySCF 2.15's SMD solvent table,
# solvent_db[name][5]; run_ref.py checks it for SMD) and the Lebedev points per sphere
SOLVENT_EPS = {"water": 78.3553}
PCM_LEBEDEV = {"cpcm": 302, "iefpcm": 302, "smd": 590}


def solvent_of(case: Case) -> tuple[str, str, float, int] | None:
    """(model, solvent, eps, Lebedev points per sphere) of an implicit-solvent case."""
    if not case.solvent:
        return None
    model, name = case.solvent.split(":")
    return model, name, SOLVENT_EPS[name], PCM_LEBEDEV[model]
ORCA_RSH_RIJK = ("ORCA 6.0.0 refuses range-separated RIJK gradients (\"the range sep RIJK gradient has an "
                 "unresolved bug\"): ORCA energy only, gradient by NumGrad, no Hessian")


def orca_rsh_rijk(case: Case) -> bool:
    return case.jk == "df" and case.method in RANGE_SEPARATED


def orca_numgrad(case: Case) -> bool:
    """ORCA gradients by NumGrad: its analytic D3(BJ) gradient is inconsistent with its D3
    energy, and it has no range-separated RIJK gradient."""
    return disp_of(case) == "d3bj" or orca_rsh_rijk(case)


def _cases() -> list[Case]:
    c: list[Case] = []
    # Hartree-Fock, exact integrals
    for mol in ("water", "nh3", "ch3sh", "clcnh+"):
        c.append(Case(mol, "hf", "def2-svp", quick=mol in ("water", "clcnh+")))
    for mol in ("water", "ch3sh", "clcnh+"):
        c.append(Case(mol, "hf", "def2-tzvp", quick=mol == "ch3sh"))
    c.append(Case("water", "hf", "6-31g*", cart=True, quick=True))
    c.append(Case("ch3sh", "hf", "6-31g*", cart=True))
    c.append(Case("nh3+", "hf", "def2-svp", quick=True))
    c.append(Case("nh3+", "hf", "def2-tzvp"))
    # Hartree-Fock, RI-JK
    c.append(Case("water", "hf", "def2-svp", jk="df", quick=True))
    c.append(Case("ch3sh", "hf", "def2-svp", jk="df"))
    c.append(Case("clcnh+", "hf", "def2-tzvp", jk="df"))
    c.append(Case("nh3+", "hf", "def2-svp", jk="df"))
    # DFT, exact J/K: LDA, GGA, hybrid GGA, meta-GGA on the four closed shells
    for method in ("svwn-rpa", "pbe", "b3lyp", "tpss"):
        for mol in ("water", "nh3", "ch3sh", "clcnh+"):
            c.append(Case(mol, method, "def2-svp", quick=(mol, method) in (("water", "b3lyp"), ("nh3", "pbe"), ("ch3sh", "tpss"))))
    # (def2-TZVP DFT Hessians cost the references most: one Hessian case, the rest gradients)
    c.append(Case("water", "r2scan", "def2-svp"))
    c.append(Case("ch3sh", "r2scan", "def2-svp", calc="gradient"))
    for method in ("pbe", "b3lyp", "tpss"):
        c.append(Case("ch3sh", method, "def2-tzvp", calc="hessian" if method == "b3lyp" else "gradient",
                      quick=method == "b3lyp"))
    c.append(Case("clcnh+", "b3lyp", "def2-tzvp", calc="gradient"))
    c.append(Case("water", "b3lyp", "6-31g*", cart=True, quick=True))
    c.append(Case("ch3sh", "b3lyp", "6-31g*", cart=True))
    # DFT, open shell (UKS energies, gradients and Hessians)
    for method in ("svwn-rpa", "pbe", "b3lyp", "tpss"):
        c.append(Case("nh3+", method, "def2-svp", quick=method == "b3lyp"))
    # DFT, RI-JK
    c.append(Case("water", "b3lyp", "def2-svp", jk="df", quick=True))
    c.append(Case("ch3sh", "pbe", "def2-svp", jk="df"))
    c.append(Case("ch3sh", "b3lyp", "def2-svp", jk="df"))
    c.append(Case("ch3sh", "b3lyp", "def2-tzvp", jk="df", calc="gradient"))
    c.append(Case("nh3+", "b3lyp", "def2-svp", jk="df"))
    # dispersion on B3LYP
    for disp in ("d3bj", "d4"):
        c.append(Case("ch3sh", "b3lyp", "def2-svp", disp=disp, quick=True))
        c.append(Case("clcnh+", "b3lyp", "def2-svp", disp=disp, calc="gradient"))
    # medium molecule
    c.append(Case("la18", "hf", "def2-svp", jk="df"))
    c.append(Case("la18", "hf", "def2-svp", calc="gradient"))
    c.append(Case("la18", "b3lyp", "def2-svp", jk="df", calc="gradient"))
    # exact J/K: level 7 (1.2M points); PySCF's grid-response gradient with exact J/K on level 9
    # (3M points) exceeds a 7 GB memory cap
    c.append(Case("la18", "b3lyp", "def2-svp", calc="gradient", level=7))
    # range-separated hybrids: wB97X, wB97X-D3BJ and CAM-B3LYP, exact and RI (linus's RI
    # long-range exchange is Psi4's wK fit; the other programs fit it their own ways, so the RI
    # rows carry the rsh-ri-fitting explanation; the exact wK comparison is ch3sh...df.wk below)
    c.append(Case("water", "wb97x", "def2-svp", quick=True))
    c.append(Case("ch3sh", "wb97x", "def2-svp", jk="df"))
    c.append(Case("water", "wb97x-d3bj", "def2-svp", calc="gradient", quick=True))
    c.append(Case("ch3sh", "wb97x-d3bj", "def2-svp", jk="df", calc="gradient"))
    c.append(Case("water", "cam-b3lyp", "def2-svp", calc="gradient", quick=True))
    c.append(Case("ch3sh", "cam-b3lyp", "def2-svp", jk="df", calc="gradient"))
    # effective core potentials (def2-ECP): HF, PBE, B3LYP; SVP and TZVP; closed and open
    # shells (open shells with a nondegenerate singly occupied orbital: CH2I, Au2+); exact and RI
    # (Hessians on non-linear molecules: bench/vib/refs.py's harmonic analysis of the
    # references does not take linear ones)
    c.append(Case("hi", "hf", "def2-svp", calc="gradient", quick=True))
    c.append(Case("hi", "b3lyp", "def2-svp", calc="gradient"))
    c.append(Case("hi", "pbe", "def2-tzvp", calc="gradient"))
    c.append(Case("ch3i", "hf", "def2-svp", quick=True))
    c.append(Case("ch3i", "b3lyp", "def2-svp", quick=True))
    c.append(Case("ch3i", "hf", "def2-tzvp", calc="gradient"))
    c.append(Case("ch3i", "b3lyp", "def2-svp", jk="df", calc="gradient"))
    c.append(Case("ch2i", "b3lyp", "def2-svp", calc="gradient"))
    c.append(Case("agcl", "b3lyp", "def2-tzvp", calc="gradient"))
    c.append(Case("ptcl4--", "hf", "def2-svp"))
    c.append(Case("ptcl4--", "pbe", "def2-svp", calc="gradient"))
    c.append(Case("au2", "hf", "def2-svp", calc="gradient"))
    c.append(Case("au2+", "hf", "def2-svp", calc="gradient"))
    c.append(Case("au2+", "b3lyp", "def2-svp", calc="gradient"))
    c.append(Case("cscl", "b3lyp", "def2-svp", calc="gradient"))
    c.append(Case("pdcl2ph3", "hf", "def2-svp", calc="gradient"))
    c.append(Case("pdcl2ph3", "b3lyp", "def2-svp", jk="df", calc="gradient", level=7))
    # PySCF-only cases (refs="pyscf"): features whose Psi4 and ORCA counterparts this harness does
    # not set up (implicit solvent, VV10, PySCF patched to the wK fit) and feature combinations;
    # every comparison is linus vs PySCF on the same grid, at the exact tolerances
    py = {"refs": "pyscf"}
    # implicit solvent (linus `pyscf` preset = PySCF's pcm module: C-PCM, IEF-PCM; SMD)
    c.append(Case("water", "b3lyp", "def2-svp", solvent="cpcm:water", quick=True, **py))  # KS PCM Hessian
    c.append(Case("water", "wb97x", "def2-svp", solvent="cpcm:water", **py))  # RSH PCM Hessian
    c.append(Case("ch3sh+", "b3lyp", "def2-svp", calc="gradient", solvent="iefpcm:water", **py))  # UKS
    c.append(Case("water", "b3lyp", "def2-svp", calc="gradient", solvent="smd:water", **py))
    c.append(Case("ch3i", "hf", "def2-svp", calc="gradient", solvent="cpcm:water", **py))  # ECP + PCM
    # VV10 nonlocal correlation (nlc grid PySCF level 3 in both), closed and open shell
    c.append(Case("nh3", "wb97m-v", "def2-svp", calc="gradient", **py))
    c.append(Case("nh3+", "wb97m-v", "def2-svp", calc="gradient", **py))
    # VV10 Hessians (RKS; PySCF 2.15 has no UKS NLC Hessian): PySCF's VV10 Fock derivative patched
    # to the fixed grid of its other terms (run_ref.pyscf_vv10_hessian_fixed_grid)
    c.append(Case("water", "wb97m-v", "def2-svp", quick=True, **py))
    c.append(Case("water", "b97m-v", "def2-svp", **py))
    # range-separated UKS Hessian; RI wK exchange against PySCF with the same fit
    c.append(Case("nh3+", "wb97x", "def2-svp", **py))
    c.append(Case("ch3sh", "wb97x", "def2-svp", jk="df", calc="energy", wk=True, **py))
    # ECP with an RI Hessian (PySCF's DF Hessian, full auxiliary-basis response)
    c.append(Case("ch3i", "hf", "def2-svp", jk="df", **py))
    return c


CASES = _cases()
# one representative case per feature, run first when CPU is short
CORE = {
    "water.hf.def2-svp.exact", "nh3+.hf.def2-svp.exact", "clcnh+.hf.def2-svp.exact", "water.hf.def2-svp.df",
    "water.hf.6-31gs-cart.exact", "water.svwn-rpa.def2-svp.exact", "water.pbe.def2-svp.exact",
    "water.b3lyp.def2-svp.exact", "water.tpss.def2-svp.exact", "nh3+.b3lyp.def2-svp.exact",
    "water.b3lyp.def2-svp.df", "ch3sh.b3lyp-d3bj.def2-svp.exact", "ch3sh.b3lyp-d4.def2-svp.exact",
    "water.wb97x.def2-svp.exact",
}

# ------------------------------------------------------------------ runs


@dataclass
class Run:
    """One program run for one case."""

    name: str  # unique within the case, e.g. "pyscf@pyscf:9", "linus@psi4", "orca-numfreq@fine"
    program: str  # linus | pyscf | psi4 | orca
    calc: str
    grid: str | None = None  # pyscf:N | psi4 | fine (DFT only)
    opts: dict[str, Any] = field(default_factory=dict)
    mem_gb: int = 5

    def is_ref(self) -> bool:
        return self.program != "linus"


def runs_for(case: Case) -> list[Run]:
    r = _runs_for(case)
    if case.refs == "pyscf":
        # linus and PySCF on PySCF's grids (or none): no Psi4 or ORCA runs, nor the linus runs that
        # only exist to meet them (psi4 grid, Psi4-style finite differences, ORCA's grid response)
        r = [x for x in r if x.program in ("linus", "pyscf") and (x.grid is None or x.grid.startswith("pyscf:"))
             and not x.name.startswith("linus-hgr")]
    return r


def _runs_for(case: Case) -> list[Run]:
    r: list[Run] = []
    mol = MOLECULES[case.mol]
    calc = case.calc
    big = case.mol == "la18"
    pmem = 7 if big else 5  # PySCF scope cap (GB); max_memory is set 1.5 GB below it
    if case.cart:
        # PySCF with linus's basis data: tells basis-data differences (6-31G* coefficients
        # have 7-8 digits in PySCF's and Psi4's libraries, 10 in BSE) from code differences
        grid = f"pyscf:{case.hlevel}" if case.dft else None
        r.append(Run(f"pyscf-bse@{grid}" if grid else "pyscf-bse", "pyscf", calc, grid, {"pyscf_linus_basis": True},
                     mem_gb=pmem))
    if not case.dft:
        r.append(Run("linus", "linus", calc, mem_gb=6 if big else 4))
        r.append(Run("pyscf", "pyscf", calc, mem_gb=pmem))
        r.append(Run("psi4", "psi4", calc))
        if not case.cart:
            r.append(Run("orca", "orca", calc))
        return r
    lv, hv = f"pyscf:{case.level}", f"pyscf:{case.hlevel}"
    # Psi4: analytic Hessians for HF and LDA only (fixed grid); GGA and meta-GGA Hessians
    # are finite differences of gradients without grid-weight derivatives (linus-fd@psi4)
    psi4_fd = case.method not in LDA
    lmem = 6 if big else 4
    r.append(Run(f"linus@{lv}", "linus", min_calc(calc, "gradient"), lv,
                 {"grid_response": True, "hess_grid_response": False}, mem_gb=lmem))
    r.append(Run(f"pyscf@{lv}", "pyscf", min_calc(calc, "gradient"), lv, mem_gb=pmem))
    if calc == "hessian":
        r.append(Run(f"linus@{hv}", "linus", "hessian", hv, {"grid_response": True, "hess_grid_response": False},
                     mem_gb=lmem))
        r.append(Run(f"pyscf@{hv}", "pyscf", "hessian", hv, mem_gb=pmem))
    r.append(Run("linus@psi4", "linus", calc, "psi4", {"grid_response": False, "hess_grid_response": psi4_fd},
                 mem_gb=lmem))
    r.append(Run("psi4@psi4", "psi4", calc, "psi4"))
    if calc == "hessian" and psi4_fd:
        # what Psi4 computes: finite differences of gradients without grid-weight
        # derivatives on its default grid (which turns with the molecule's standard
        # orientation), translations projected in mass-weighted coordinates; done with linus
        r.append(Run("linus-fd@psi4", "linus", "hessian", "psi4",
                     {"grid_response": False, "fd": "psi4"}, mem_gb=6 if big else 4))
    if case.psi4_fine:
        r.append(Run("psi4@fine", "psi4", min_calc(calc, "gradient"), "fine"))
    if not case.cart:
        if orca_rsh_rijk(case):
            # ORCA 6.0.0 stops range-separated RIJK gradients ("the range sep RIJK gradient
            # has an unresolved bug"): energy only, gradient by NumGrad, no Hessian
            r.append(Run("orca@fine", "orca", "energy", "fine", {"limit": ORCA_RSH_RIJK}))
        else:
            r.append(Run("orca@fine", "orca", calc, "fine",
                         {"orca_hessian": "analytic" if case.jk == "exact" else "numfreq"}))
        if calc == "hessian" and case.jk == "exact":
            r.append(Run("orca-numfreq@fine", "orca", "hessian", "fine", {"orca_hessian": "numfreq"}))
        if orca_numgrad(case) and calc in ("gradient", "hessian"):
            r.append(Run("orca-numgrad@fine", "orca", "gradient", "fine", {"orca_numgrad": True}))
    if calc == "hessian" and case.jk == "exact" and not case.cart:
        # ORCA's analytic DFT Hessian convention (first-order grid response, Bykov et al.)
        r.append(Run(f"linus-hgr@{hv}", "linus", "hessian", hv, {"grid_response": True, "hess_grid_response": True},
                     mem_gb=lmem))
    del mol
    return r


ORDER = ("energy", "gradient", "hessian")


def min_calc(a: str, b: str) -> str:
    return ORDER[min(ORDER.index(a), ORDER.index(b))]


# ------------------------------------------------------------------ geometry


def read_xyz(path: Path) -> tuple[list[str], list[list[float]]]:
    lines = path.read_text().splitlines()
    n = int(lines[0].split()[0])
    syms, xyz = [], []
    for line in lines[2 : 2 + n]:
        p = line.split()
        syms.append(p[0].capitalize())
        xyz.append([float(x) for x in p[1:4]])
    return syms, xyz


def geometry_bohr(mol: Mol) -> tuple[list[str], list[list[float]]]:
    """The same conversion as run_ref.py (qcref, CODATA 2022)."""
    from qcref import constants

    syms, xyz = read_xyz(REPO / mol.xyz)
    a2b = float(constants.ANGSTROM_TO_BOHR)
    return syms, (np.asarray(xyz) * a2b).tolist()


# ------------------------------------------------------------------ execution


def scoped(cmd: list[str], lane: str, mem_gb: int) -> list[str]:
    """Run in its own memory-capped scope (an OOM kill then stays inside it), pinned to a CPU
    set: a `slice` name (build, ccd2, ...), `cpus:LIST` (taskset, e.g. cpus:56-63,120-125 for
    CCD7 without hardware threads 126-127), or none."""
    pre = ["systemd-run", "--user", "--scope", "-q", "-p", f"MemoryMax={mem_gb}G", "-p", "MemorySwapMax=0"]
    if lane.startswith("cpus:"):
        pre += ["taskset", "-c", lane.split(":", 1)[1]]
    elif lane != "none":
        pre += [str(SLICE), lane]
    return pre + cmd


def spec_of(case: Case, run: Run) -> dict[str, Any]:
    mol = MOLECULES[case.mol]
    # only what determines the result (grid levels are in run.grid)
    model = {k: getattr(case, k) for k in ("mol", "method", "basis", "jk", "cart", "disp")}
    # newer model fields only when set, so the specs (and hashes) of the older cases stay as stored
    sol = solvent_of(case)
    if sol:
        model["solvent"] = dict(zip(("model", "solvent", "eps", "lebedev"), sol))
    if case.wk:
        model["wk"] = True
    if case.method in run_ref.VV10 and case.dft:
        model["nlc_grid"] = "pyscf:3"
    spec = {"case": model, "molecule": asdict(mol), "run": {k: v for k, v in asdict(run).items() if k != "mem_gb"},
            "aux": AUX if case.jk == "df" else None}
    if run.is_ref():
        spec["ref_settings_version"] = REF_SETTINGS_VERSION
        spec["orca_fd_step"] = (ORCA_NUMGRAD_STEP if run.opts.get("orca_numgrad") else ORCA_NUMFREQ_STEP) if run.program == "orca" else None
    else:
        spec["linus_settings"] = {**LINUS_SCF, **LINUS_ENV}
    return spec


def spec_hash(spec: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:16]


def sanitize(obj: Any) -> Any:
    """Drop host names and absolute paths from a result record."""
    home = str(Path.home())
    repo = str(REPO)
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if k in ("host", "raw_dir", "orca_dir", "hostname"):
                continue
            out[k] = sanitize(v)
        return out
    if isinstance(obj, list):
        return [sanitize(v) for v in obj]
    if isinstance(obj, str):
        s = obj.replace(repo + "/", "").replace(repo, ".")
        s = s.replace(str(run_ref.REFENV), "$REFENV").replace(str(run_ref.ORCA_DIR), "$ORCA_DIR")
        s = s.replace(str(RAW_DIR), "$RAW").replace(home, "~")
        s = re.sub(r"\bafh-[a-z0-9-]+\b", "<host>", s)
        return s
    return obj


def run_reference(case: Case, run: Run, lane: str, threads: int) -> dict[str, Any]:
    mol = MOLECULES[case.mol]
    rundir = RAW_DIR / case.id / run.name.replace(":", "")
    if rundir.exists():
        shutil.rmtree(rundir)
    rundir.mkdir(parents=True)
    mem = run.mem_gb
    prog_mem = max(1, mem - 2)  # memory the program is told (GB); the scope caps at mem_gb
    cmd = [str(PYTHON), str(HERE / "run_ref.py"), str(REPO / mol.xyz), "--charge", str(mol.charge),
           "--mult", str(mol.mult), "--method", case.method, "--basis", case.basis, "--calc", run.calc,
           "--programs", run.program, "--threads", str(threads), "--memory-gb", str(prog_mem),
           "--rundir", str(rundir)]
    if case.jk == "df":
        cmd += ["--approx", "df", "--aux", "def2/jk"]
    if case.cart:
        cmd += ["--cart"]
    if case.disp:
        cmd += ["--disp", case.disp]
    if case.dft:
        cmd += ["--grid", run.grid]
    sol = solvent_of(case)
    if sol:
        cmd += ["--pcm", sol[0], "--solvent", sol[1], "--eps", repr(sol[2]), "--pcm-lebedev", str(sol[3])]
    if case.wk:
        cmd += ["--wk"]
    if case.method in run_ref.VV10:
        cmd += ["--nlc-level", "3"]
    if run.program == "pyscf":
        cmd += ["--pyscf-max-memory-mb", str(int((mem - 1.5) * 1000))]
        if run.opts.get("pyscf_linus_basis"):
            cmd += ["--pyscf-linus-basis"]
    if run.program == "orca":
        cmd += ["--orca-fd-step", str(ORCA_NUMGRAD_STEP if run.opts.get("orca_numgrad") else ORCA_NUMFREQ_STEP)]
        if run.opts.get("orca_hessian"):
            cmd += ["--orca-hessian", run.opts["orca_hessian"]]
        if run.opts.get("orca_numgrad"):
            cmd += ["--orca-numgrad"]
    t0 = time.perf_counter()
    p = subprocess.run(scoped(cmd, lane, mem), capture_output=True, text=True)
    wall = time.perf_counter() - t0
    (rundir / "run_ref.log").write_text(p.stdout + p.stderr)
    f = rundir / f"{run.program}.json"
    if f.exists():
        rec = json.loads(f.read_text())
    else:
        rec = {"success": False, "error": f"run_ref.py exit {p.returncode} (no result; killed? see run_ref.log)",
               "log_tail": (p.stdout + p.stderr)[-2000:]}
    rec["scope_wall_time_s"] = wall
    return rec


def git_commit() -> dict[str, Any]:
    """The commit of the checkout the linus binary was built in (<checkout>/target/release/
    linus; else this one). dirty: uncommitted changes to linus's sources, not reference/ or docs.
    The binary's own build is not checked against the commit: rebuild after checking out."""
    repo = LINUS_BIN.parents[2] if LINUS_BIN.parent.name == "release" and LINUS_BIN.parents[1].name == "target" else REPO

    def g(*a):
        return subprocess.run(["git", "-C", str(repo), *a], capture_output=True, text=True).stdout.strip()

    dirty = bool(g("status", "--porcelain", "--untracked-files=no", "--", "crates", "Cargo.toml", "Cargo.lock"))
    return {"commit": g("rev-parse", "HEAD"), "describe": g("describe", "--always") + ("-dirty" if dirty else ""),
            "dirty": dirty, "binary_mtime": _dt.datetime.fromtimestamp(LINUS_BIN.stat().st_mtime, _dt.timezone.utc)
            .isoformat(timespec="seconds") if LINUS_BIN.exists() else None}


def vib_masses(symbols: list[str]) -> list[float]:
    """Most abundant isotope masses (u, qcelemental): given to linus and to every reference
    program's harmonic analysis, so frequencies differ only through the Hessians."""
    from qcelemental import periodictable

    return [float(periodictable.to_mass(s)) for s in symbols]


def linus_request(case: Case, run: Run, calc: str) -> dict[str, Any]:
    mol = MOLECULES[case.mol]
    syms, bohr = geometry_bohr(mol)
    # LINUS_SCF's settings that take effect here (D41: the four-center screening is jk exact's, the
    # response's tolerance a Hessian's)
    scf = {"scf": LINUS_SCF["scf"]}
    if case.jk not in ("df", "rij-snk", "rij_snk"):
        scf["jk"] = {"screening": LINUS_SCF["jk"]["screening"]}
    if calc == "hessian":
        scf["response"] = LINUS_SCF["response"]
    req: dict[str, Any] = {"structure": {"symbols": syms, "geometry": bohr, "charge": mol.charge,
                                         "multiplicity": mol.mult},
                           "calctype": calc, "method": case.method, "basis": case.basis,
                           "cartesian": bool(case.cart), **scf}
    if calc == "hessian":  # the Hessian plus linus's harmonic analysis (settings group freq)
        req["freq"] = {"masses": vib_masses(syms)}
    if case.jk == "df":
        req["jk"] = {**req.get("jk", {}), "kind": "df", "aux": AUX}
    elif case.jk == "exact":  # named: the default is df (decision 6)
        req["jk"] = {**req.get("jk", {}), "kind": "exact"}
    if case.dft:
        grid: dict[str, Any] = {"xc": run.grid}
        for k, v in run.opts.items():
            if k == "fd":
                continue
            # one key, grid.response, for the grid response of a gradient and of a Hessian (D41: none for
            # an energy); this script's opts keep two names: grid_response for a gradient and
            # hess_grid_response for a Hessian (a Hessian calculation's gradient then takes the Hessian's value)
            if k == "hess_grid_response":
                if calc == "hessian":
                    grid["response"] = v
            elif k == "grid_response":
                if calc == "gradient":
                    grid["response"] = v
            else:
                req[k] = v
        req["grid"] = grid
    if case.disp:
        req["dispersion"] = case.disp
    sol = solvent_of(case)
    if sol:
        model, name, eps, leb = sol
        req["solvent"] = {"model": model, "eps": eps, "preset": "pyscf", "lebedev": leb, "discretization": "swig"}
        if model == "smd":  # PySCF's CDS term uses the exact SASA
            req["solvent"].update(name=name, cds_contact_smoothing=0.0)
    if case.method in run_ref.VV10 and case.dft and (run.grid or "").startswith("pyscf:"):
        req["grid"]["nlc"] = "pyscf:3"  # PySCF's nlcgrids level 3 (run_ref --nlc-level)
    return req


def request_record(req: dict[str, Any]) -> dict[str, Any]:
    """The request as stored with a result: without the structure's symbols and geometry."""
    return {**req, "structure": {k: v for k, v in req["structure"].items() if k not in ("symbols", "geometry")}}


def linus_call(req: dict[str, Any], lane: str, mem_gb: int) -> tuple[int, dict[str, Any] | None, str, float]:
    env = {**os.environ, **LINUS_ENV}
    t0 = time.perf_counter()
    p = subprocess.run(scoped([str(LINUS_BIN), "--json"], lane, mem_gb), input=json.dumps(req),
                       capture_output=True, text=True, env=env)
    wall = time.perf_counter() - t0
    return p.returncode, (json.loads(p.stdout) if p.returncode == 0 else None), p.stderr, wall


PSI4_FD_STEP = 0.005  # bohr; Psi4's disp_size (Psi4 steps along mass-weighted SALCs)


def psi4_translation_projector(symbols: list[str]) -> np.ndarray:
    """Psi4 1.11 findif projects translations out of the Hessian in mass-weighted SALC
    coordinates even with fd_project false ("Translations projected? 1"): H = Q J Q^T with
    Q = M^1/2 (1 - T_m T_m^T) M^-1/2, T_m the mass-weighted translations (Psi4's masses)."""
    from qcelemental import periodictable

    m = np.repeat([periodictable.to_mass(s) for s in symbols], 3)
    n3 = m.size
    T = np.zeros((n3, 3))
    for k in range(3):
        T[k::3, k] = np.sqrt(m[k::3])
        T[:, k] /= np.linalg.norm(T[:, k])
    P = np.eye(n3) - T @ T.T
    return np.diag(np.sqrt(m)) @ P @ np.diag(1.0 / np.sqrt(m))


def run_linus_fd(case: Case, run: Run, lane: str, threads: int) -> dict[str, Any]:
    """5-point central differences of linus gradients (Cartesian steps), symmetrized and
    projected as Psi4 does (psi4_translation_projector): the Hessian Psi4 reports for GGAs."""
    rundir = RAW_DIR / case.id / run.name.replace(":", "")
    rundir.mkdir(parents=True, exist_ok=True)
    base = linus_request(case, run, "gradient")
    base["threads"] = threads
    base.pop("fd", None)
    pos0 = np.asarray(base["structure"]["geometry"])
    n3 = pos0.size
    rc, out0, err, wall = linus_call(base, lane, run.mem_gb)
    if rc:
        return {"program": "linus", "success": False, "error": err.strip()[-300:]}
    J = np.zeros((n3, n3))
    t0 = time.perf_counter()
    for k in range(n3):
        a, c = divmod(k, 3)
        gs = []
        for s in (-2, -1, 1, 2):
            p = pos0.copy()
            p[a, c] += s * PSI4_FD_STEP
            rc, out, err, _ = linus_call({**base, "structure": {**base["structure"], "geometry": p.tolist()}}, lane, run.mem_gb)
            if rc:
                return {"program": "linus", "success": False, "error": err.strip()[-300:]}
            gs.append(np.asarray(out["gradient"]).ravel())
        J[:, k] = (gs[0] - 8 * gs[1] + 8 * gs[2] - gs[3]) / (12 * PSI4_FD_STEP)
    Q = psi4_translation_projector(base["structure"]["symbols"])
    H = Q @ (0.5 * (J + J.T)) @ Q.T
    return {"program": "linus", "success": True, "calc_type": "hessian", "energy": out0["energy"],
            "dipole": out0["dipole"], "gradient": out0["gradient"], "hessian": H.tolist(),
            "hessian_method": (f"5-point central differences (step {PSI4_FD_STEP} bohr) of linus gradients without "
                               "grid response on the psi4 grid, symmetrized, translations projected as Psi4 does"),
            "jacobian_asymmetry": float(np.abs(J - J.T).max()), "version": out0.get("version"),
            "options": out0.get("options"), "wall_time_s": wall + time.perf_counter() - t0,
            "request": request_record(base)}


def run_linus(case: Case, run: Run, lane: str, threads: int) -> dict[str, Any]:
    """linus CLI, JSON request on stdin, JSON result on stdout. Open-shell Hessians are not
    implemented: then the gradient is run and the Hessian recorded as not implemented."""
    if run.opts.get("fd"):
        return run_linus_fd(case, run, lane, threads)
    rundir = RAW_DIR / case.id / run.name.replace(":", "")
    rundir.mkdir(parents=True, exist_ok=True)
    rec: dict[str, Any] = {"program": "linus"}
    # a hessian call carries linus's harmonic analysis (settings group freq)
    calc = run.calc
    while True:
        req = linus_request(case, run, calc)
        req["threads"] = threads
        (rundir / f"request_{calc}.json").write_text(json.dumps(req))
        rc, out, stderr, wall = linus_call(req, lane, run.mem_gb)
        (rundir / f"stderr_{calc}.txt").write_text(stderr)
        if rc == 0:
            rec.update({"success": True, "calc_type": calc, "wall_time_s": wall, "request": request_record(req)})
            for k in ("energy", "dipole", "gradient", "converged", "iterations", "cphf_iterations", "nbf",
                      "components", "version", "options"):
                if k in out:
                    rec[k] = out[k]
            if "hessian" in out:
                rec["hessian"] = np.asarray(out["hessian"]).reshape(3 * len(req["structure"]["symbols"]), -1).tolist()
            vib = out.get("vibrational_analysis")
            if vib:
                rec["frequencies"] = vib["frequencies"]
                rec["vib_analysis"] = {k: vib.get(k) for k in ("masses", "projection", "codata", "rotor_type", "n_imaginary")}
            rec["scf_convergence"] = out["scf"]["convergence"]
            # total: the whole call (before a0f30e4: the SCF's; "calculation" was the rest), energy: the SCF stage
            rec["timings_s"] = {k: v for k, v in out.get("timings", {}).items() if k in ("total", "energy", "calculation")}
            if out.get("resolved"):
                rec["resolved"] = out["resolved"]
            gr, hgr = run.opts.get("grid_response"), run.opts.get("hess_grid_response")
            if calc == "hessian" and case.dft and gr is not None and hgr is not None and gr != hgr:
                # grid.response is one key for the gradient and the Hessian: the Hessian call took the
                # Hessian's value, so the gradient with its own grid response (the one the references'
                # gradients have) is a call of its own
                greq = linus_request(case, run, "gradient")
                greq["threads"] = threads
                grc, gout, gstderr, gwall = linus_call(greq, lane, run.mem_gb)
                if grc:
                    rec.update({"success": False, "error": "gradient call: " + (gstderr.strip().splitlines() or [f"exit {grc}"])[-1],
                                "wall_time_s": wall})
                    return rec
                rec["gradient"] = gout["gradient"]
                rec["gradient_call"] = "separate call (grid.response differs from the Hessian's)"
                rec["wall_time_s"] += gwall
            return rec
        err = stderr.strip().splitlines()[-1] if stderr.strip() else f"exit {rc}"
        rec.setdefault("not_implemented", {})
        if "not implemented" in err.lower() or "only" in err.lower() and "closed shell" in err.lower():
            rec["not_implemented"][calc] = err
            if calc == "hessian":
                calc = "gradient"  # keep energy, dipole and gradient comparable
                continue
        rec.update({"success": False, "error": err, "wall_time_s": wall})
        return rec


# ------------------------------------------------------------------ comparisons

# Tolerances on max |difference| (energy Eh, dipole e bohr, gradient Eh/bohr, Hessian
# Eh/bohr^2), by kind of comparison:
#   exact  same model, integrals and grid (or no grid): what the SCF and CPHF thresholds allow
#   fd5    Psi4's 5-point finite-difference Hessian against the same finite differences of
#          linus gradients (step 0.005 bohr)
#   fd2    ORCA's central differences: NumFreq (0.002 bohr; its HF Hessian is 1.4e-6 from
#          ORCA's analytic one for water) and NumGrad (0.001 bohr)
#   grid   the same model on different grids (references among themselves for DFT)
# linus against ORCA for DFT (different grids) passes when linus is no further from ORCA
# than PySCF on linus's grid is: |linus - ORCA| <= tol + |PySCF - ORCA| (the "grid" column
# of the summary is that residual).
TOL = {
    "exact": {"energy": 1e-9, "dipole": 3e-8, "gradient": 3e-8, "hessian": 1e-7, "frequency": 1e-2},
    "fd5": {"energy": 1e-9, "dipole": 3e-8, "gradient": 3e-8, "hessian": 1e-6, "frequency": 5e-2},
    "fd2": {"energy": 1e-9, "dipole": 3e-8, "gradient": 3e-6, "hessian": 2e-5, "frequency": 0.5},
    "grid": {"energy": 1e-7, "dipole": 1e-6, "gradient": 3e-6, "hessian": 1e-4, "frequency": 2.0},
}
# grid errors grow with the number of atoms: "grid" tolerances are per 3 atoms (water)
# frequency: harmonic frequencies in cm^-1 (same masses everywhere, translations and
# rotations projected; each reference's Hessian through its own program's analysis)
QUANTITIES = ("energy", "dipole", "gradient", "hessian", "frequency")


def run_file(case: Case, run_name: str, suffix: str = ".json") -> Path:
    """A run's record: the one this invocation writes (WRITE_DIR) if it exists, else the stored
    one (--quick reads the stored references next to its own linus runs); the write path if
    neither exists."""
    name = run_name.replace(":", "") + suffix
    for d in dict.fromkeys((WRITE_DIR, RUNS_DIR)):
        if (d / case.id / name).exists():
            return d / case.id / name
    return WRITE_DIR / case.id / name


def freq_file(case: Case, run_name: str) -> Path:
    return run_file(case, run_name, ".freq.json")


def load_runs(case: Case) -> dict[str, dict[str, Any]]:
    """Stored runs of a case; reference runs get the frequencies of their own harmonic
    analysis (analyze_frequencies) attached."""
    out = {}
    for run in runs_for(case):
        f = run_file(case, run.name)
        if f.exists():
            out[run.name] = json.loads(f.read_text())
            ff = freq_file(case, run.name)
            if run.is_ref() and ff.exists():
                fr = json.loads(ff.read_text())
                if fr.get("hessian_hash") == hessian_hash(out[run.name]):
                    out[run.name]["frequencies"] = fr["frequencies"]
    return out


def hessian_hash(rec: dict[str, Any]) -> str | None:
    if not rec.get("success") or rec.get("hessian") is None:
        return None
    return hashlib.sha256(json.dumps(rec["hessian"]).encode()).hexdigest()[:16]


def analyze_frequencies(cases: list[Case]) -> None:
    """Harmonic frequencies of every stored reference Hessian, by the same program's own
    analysis (bench/vib/refs.py: PySCF thermo.harmonic_analysis, Psi4 qcdb vibanal, ORCA
    orca_vib), with the same masses as linus (vib_masses), translations and rotations
    projected. Cached in runs/<case>/<run>.freq.json by the Hessian's hash, with the program and
    its version (the run's: the analysis is that installation's own)."""
    sys.path.insert(0, str(REPO / "bench" / "vib"))
    import refs

    def orca_frequencies(z, pos, hess, masses, energy, mult):
        """orca_vib alone (refs.orca_analysis also runs PrintThermoChem, not needed here)."""
        wd = RAW_DIR / "orca_vib" / f"{c.id}.{run.name.replace(':', '')}"
        if wd.exists():
            shutil.rmtree(wd)
        wd.mkdir(parents=True)
        refs.write_orca_hess(wd / "mol.hess", z, pos, hess, masses, energy, mult)
        p = subprocess.run([str(refs.ORCA / "orca_vib"), "mol.hess"], cwd=wd, env=refs._orca_env(),
                           capture_output=True, text=True)
        (wd / "orca_vib.out").write_text(p.stdout + p.stderr)
        text = (wd / "mol.hess").read_text()
        sec = refs._hess_section(text, "vibrational_frequencies").splitlines()
        nf = int(sec[1])
        allf = np.array([float(x.split()[1]) for x in sec[2:2 + nf]])
        modes = refs._hess_matrix(refs._hess_section(text, "normal_modes")).T
        return {"frequencies": allf[np.linalg.norm(modes, axis=1) > 0]}

    fns = {"pyscf": refs.pyscf_analysis, "psi4": refs.psi4_analysis, "orca": orca_frequencies}
    for c in cases:
        if c.calc != "hessian":
            continue
        for run in runs_for(c):
            f = run_file(c, run.name)
            if not run.is_ref() or not f.exists():
                continue
            rec = json.loads(f.read_text())
            h = hessian_hash(rec)
            ff = freq_file(c, run.name)
            if h is None or (ff.exists() and json.loads(ff.read_text()).get("hessian_hash") == h):
                continue
            ff = WRITE_DIR / c.id / ff.name  # a new analysis goes where this invocation writes
            ff.parent.mkdir(parents=True, exist_ok=True)
            mol = rec["molecule"]
            masses = vib_masses(mol["symbols"])
            try:
                res = fns[rec["program"]](np.asarray(mol["atomic_numbers"]), np.asarray(mol["geometry_bohr"]),
                                          np.asarray(rec["hessian"]), masses, rec["energy"], mol["multiplicity"])
            except Exception as exc:  # keep going; the comparison shows n/a
                print(f"  frequency analysis {c.id} {run.name}: {type(exc).__name__}: {exc}", flush=True)
                continue
            ff.write_text(json.dumps({"case": c.id, "run": run.name, "program": rec["program"],
                                      "program_version": rec.get("program_version"), "hessian_hash": h, "masses": masses,
                                      "analysis": f"{rec['program']} harmonic analysis (bench/vib/refs.py), "
                                                  "translations and rotations projected",
                                      "frequencies": np.asarray(res["frequencies"]).tolist()}) + "\n")
            print(f"  frequencies {c.id} {run.name}: {len(res['frequencies'])} modes", flush=True)


def values(rec: dict[str, Any] | None, q: str):
    if not rec or not rec.get("success"):
        return None
    v = rec.get("frequencies" if q == "frequency" else q)
    return None if v is None else np.asarray(v, dtype=float)


def why_missing(rec: dict[str, Any] | None, q: str, prog: str) -> str:
    if rec is None:
        return "not run"
    ni = rec.get("not_implemented") or {}
    if q in ("hessian", "frequency") and "hessian" in ni:
        return "not implemented in linus: " + ni["hessian"]
    if not rec.get("success"):
        if prog == "linus" and "not implemented" in (rec.get("error") or ""):
            return "not implemented in linus: " + rec["error"]
        return "failed: " + (rec.get("error") or "")[:200]
    limit = ((rec.get("spec") or {}).get("run") or {}).get("opts", {}).get("limit")
    return f"not available: {limit}" if limit else "not computed"


def maxabs(a, b) -> float:
    return float(np.abs(a - b).max())


def linus_dispersion_energy(rec: dict[str, Any]) -> float | None:
    """The CLI's `components.dispersion` (since b3ca610); records from older builds have no
    dispersion entry, so it is energy - sum(components) for them."""
    comp = rec.get("components") or {}
    if not rec.get("success") or rec.get("energy") is None or not comp:
        return None
    if "dispersion" in comp:
        return comp["dispersion"] or 0.0
    return rec["energy"] - sum(v for v in comp.values() if v is not None)


# Known, explained differences: id -> text (reference/validation/README.md has the evidence)
EXPLAINED = {
    "orca-d3-gradient": "ORCA 6.0.0's analytic D3(BJ) gradient is not the derivative of its D3 energy; its NumFreq "
                        "(differences of those gradients) and analytic Hessians inherit the error. linus is compared "
                        "with ORCA NumGrad instead (matrix column), which agrees.",
    "orca-nori-nonhybrid-hessian": "ORCA 6.0.0's analytic Hessian of non-hybrid functionals (LDA, GGA, meta-GGA) "
                                   "with exact Coulomb (NoRI) is wrong by ~5e-2 Eh/bohr^2 (water PBE/def2-SVP bend "
                                   "1702 vs 1528 cm-1 from NumFreq), on every grid; with RI-J it agrees with NumFreq "
                                   "(6e-6), and hybrids are fine. The matrix uses ORCA NumFreq.",
}
EXPLAINED["basis-data-digits"] = (
    "PySCF's and Psi4's built-in 6-31G* coefficients have 7-8 significant digits, the Basis Set Exchange data "
    "linus reads have 10 (e.g. O 1s: 0.0018311 vs 0.00183107443); PySCF given linus's basis data agrees with linus "
    "to the exact tolerance (extra comparison in this case).")
EXPLAINED["wb97x-references"] = (
    "wB97X references among themselves (water, large grids: PySCF-Psi4 1.0e-6 Eh, PySCF-ORCA 3.0e-6, Psi4-ORCA "
    "4.0e-6; gradients up to 1e-5). PySCF-Psi4: grid. The B97-type functionals are far more sensitive to angular "
    "pruning than B3LYP: PySCF's pruned level-7 and level-9 grids agree with each other (4e-9) but sit 1.0e-6 from "
    "unpruned 150x974, on which PySCF and Psi4 agree to 6e-10 (B3LYP: 2e-9 on any of these). ORCA: its own wB97X "
    "is 4.0e-6 Eh above libxc's on unpruned grids, the same at IntAcc 8, 10 and 12 (B3LYP/G: 8e-9), so an "
    "implementation difference; its wB97X-D3BJ (the wB97X-V semilocal part) likewise sits 3.6e-6 Eh from "
    "PySCF and Psi4. No linus result enters these cells.")
EXPLAINED["pyscf-wb97x-hessian"] = (
    "wB97X Hessians on different grids or in different grid conventions: the B97-type functionals have large "
    "grid-response terms on pruned grids. PySCF's fixed-grid analytic Hessian vs central differences of its own "
    "gradients (water): 8.6e-4 Eh/bohr^2 on the pruned level-5 grid, 8.3e-5 on unpruned 100x590 (B3LYP: 2.8e-5 "
    "and 2.5e-5); Psi4 (75x302 unpruned, FD) and ORCA (NumFreq) differ from it by 8.5e-4 and from each other by "
    "2.4e-4. Not a program error as far as tested; linus vs PySCF on the same grid stays an exact comparison.")
EXPLAINED["orca-vib-frequency-factor"] = (
    "ORCA's orca_vib wavenumbers are a constant factor below those of PySCF, Psi4 and linus for the same Hessian and "
    "masses (ratio 1 - 1.8e-5, 0.07 cm^-1 at 4000 cm^-1; found by bench/vib); after removing the factor the "
    "difference is within tolerance (orca_frequency_ratio and max_abs_after_ratio in results.json).")
EXPLAINED["orca-tpss"] = (
    "ORCA 6.0.0's TPSS energy is 8.5e-7 Eh above libxc's (water/def2-SVP: ORCA vs Psi4 150x974 +8.51e-7 at IntAcc 8, "
    "+8.46e-7 at IntAcc 10; CBetaPBE/XMuePBE have no effect on it), while PySCF and Psi4 agree to 4e-9: an ORCA "
    "implementation difference. linus vs ORCA passes with the PySCF-ORCA difference as residual.")
EXPLAINED["psi4-fd-near-symmetric-top"] = (
    "NH3 is nearly a symmetric top: the standard orientation that Psi4's grid (and linus's psi4 grid) follows is "
    "ill-conditioned, so on that grid the gradient is not smooth in the geometry and finite-difference Hessians "
    "depend on the displacements: 5-point differences of linus gradients with steps 0.005 / 0.0025 / 0.01 bohr "
    "differ by 6.2e-4 / 2.4e-4 Eh/bohr^2 (water: 1e-8). linus and Psi4 gradients agree to 6e-11 at the reference "
    "and displaced geometries; Psi4 steps along mass-weighted SALCs, linus-fd along Cartesians.")
EXPLAINED["psi4-df-gradient"] = (
    "Psi4 1.11's density-fitted gradients differ from PySCF's, ORCA's and linus's (which agree with each other) by "
    "up to ~2e-8 Eh/bohr for CH3SH and ~3e-7 for la18, while the DF energies agree to 1e-11; its finite-difference "
    "Hessians (step 0.005 bohr) amplify that to ~1e-5 Eh/bohr^2.")
EXPLAINED["grid-response-convention"] = (
    "Fixed-grid DFT Hessian (PySCF, linus default) against ORCA's Hessian with grid response (NumFreq: differences "
    "of gradients with weight derivatives): the difference is the grid-response terms, large on the pruned level-5 "
    "grid for meta-GGAs (r2SCAN water 2.4e-2 Eh/bohr^2) and def2-TZVP. linus's Hessian with first-order grid "
    "response (hess_grid_response) agrees with ORCA NumFreq within the grid tolerance in this case (extra row).")
EXPLAINED["metagga-grid-sensitivity"] = (
    "Meta-GGA (TPSS, r2SCAN) references on different grids: these functionals are much more grid sensitive than "
    "GGAs, especially to PySCF's angular pruning (r2SCAN water gradients: PySCF level 9 vs Psi4 150x974 1.5e-5 "
    "Eh/bohr); linus agrees with each reference on its own grid.")
EXPLAINED["rsh-ri-fitting"] = (
    "Range-separated hybrids with RI-JK: PySCF, Psi4 and ORCA fit the long-range (erf) exchange differently with the "
    "same auxiliary basis (PySCF: erf-attenuated metric and 3-index integrals; Psi4 MemDF wK: mixed Coulomb/erf "
    "fit; ORCA: its own RIJK). CAM-B3LYP CH3SH/def2-SVP: PySCF-Psi4 1.9e-6 Eh on the same unpruned grid, where "
    "exact integrals agree to 6e-10; PySCF-ORCA 3.6e-6, Psi4-ORCA 1.7e-6 (exact: 2e-9 and 4e-9). A linus RI "
    "range-separated result can match at most one program's fit; the others differ by the fit.")
EXPLAINED["psi4-ecp-integrals"] = (
    "Psi4 1.11's ECP integrals (libecpint) are less accurate than libcint's (PySCF) and linus's, which agree with "
    "each other to ~1e-14 relative: the generalized eigenvalues of (U_ECP, S) of the same def2 basis (invariant "
    "to function order and normalization; tests/python/ecp_psi4_check.py) differ from both by up to 1e-5 (relative "
    "~4e-8; HI/def2-SVP 3.6e-7, Au2 9e-6), so Psi4 energies of ECP molecules sit 1e-8-1e-6 Eh from the others "
    "(HI HF/def2-SVP: +2.3e-8) and its gradients carry the same relative error. Same ECP data (Psi4's def2 gbs "
    "files match the BSE ones digit for digit).")
EXPLAINED["orca-ecp-integrals"] = (
    "ORCA 6.0.0 with ECP atoms differs from PySCF and linus, which agree with each other to ~1e-12 Eh (L-P), by the "
    "same amount in both columns: HF energies HI 5e-11, CH3I/def2-TZVP 2e-10, Au2 1.3e-8, PdCl2(PH3)2 1.3e-8, "
    "PtCl4 2- 3.6e-8 Eh; analytic HF Hessians up to 1.4e-6 Eh/bohr^2. Applied up to 1e-7 Eh (energy), 1e-6 (dipole, "
    "gradient), 1e-5 (Hessian), 0.1 cm^-1; beyond that a cell fails.")
EXPLAINED["orca-other-state"] = (
    "Au2+ (doublet): ORCA's SCF from its own guess converges to a state 0.077 Eh (UHF) / 0.021 Eh (UKS B3LYP) above "
    "the one linus, PySCF and Psi4 find (they agree with each other to 2e-12 Eh, L-P); not the same solution.")
EXPLAINED["ecp-heavy-atom-grids"] = (
    "DFT references on different grids for molecules with heavy (ECP) atoms: PySCF level 9, Psi4 150x974 and ORCA's "
    "finest grid differ by up to ~2e-6 Eh, 1.4e-5 e bohr and 6e-6 Eh/bohr (CsCl, PtCl4 2-, PdCl2(PH3)2), above the "
    "light-atom grid tolerance; linus matches each program on its own grid (L-P exact, L-S within Psi4's ECP "
    "integral error). Applied to reference pairs only, up to 30x the grid tolerance.")
NONHYBRID = {"svwn-rpa", "pbe", "tpss", "r2scan"}


def explain(case: Case, q: str, a: str, b: str, bse_ok: dict[str, bool] | None = None,
            hgr_ok: dict[str, bool] | None = None, d: float | None = None, tol: float | None = None) -> str | None:
    pair = {a, b}
    has_linus = any(x.startswith("linus") for x in pair)
    if case.cart and bse_ok and bse_ok.get(q) and any(x.startswith("linus") for x in pair) \
            and any(x.startswith(("pyscf", "psi4")) for x in pair):
        return "basis-data-digits"
    orca_runs = {"orca@fine", "orca-numfreq@fine", "orca"}
    if disp_of(case) == "d3bj" and q in ("gradient", "hessian") and pair & orca_runs:
        return "orca-d3-gradient"
    if case.method in NONHYBRID and case.jk == "exact" and q in ("hessian", "frequency") and pair & {"orca@fine"}:
        return "orca-nori-nonhybrid-hessian"
    if case.method == "tpss" and pair & orca_runs and not any(x.startswith("linus") for x in pair):
        return "orca-tpss"
    if case.method in RANGE_SEPARATED and case.jk == "df" and not case.wk \
            and q in ("energy", "dipole", "gradient", "hessian", "frequency"):
        return "rsh-ri-fitting"
    if case.method in ("wb97x", "wb97x-d3bj"):
        # PySCF's Hessian against other programs' grids and conventions; linus vs PySCF on the
        # same grid stays an exact comparison
        if q in ("hessian", "frequency") and any(x.startswith("pyscf") for x in pair) and not has_linus:
            return "pyscf-wb97x-hessian"
        if not has_linus:
            return "wb97x-references"
    if case.mol in ("nh3", "nh3+") and q in ("hessian", "frequency") and "psi4@psi4" in pair \
            and case.method not in LDA and case.dft:
        return "psi4-fd-near-symmetric-top"
    if case.jk == "df" and q in ("gradient", "hessian", "frequency") and any(x.startswith("psi4") for x in pair):
        return "psi4-df-gradient"
    if case.dft and q in ("hessian", "frequency") and hgr_ok and hgr_ok.get(q) \
            and pair & {"orca-numfreq@fine", "orca@fine"} and not any(x.startswith("linus-hgr") for x in pair):
        return "grid-response-convention"
    if case.method in ("tpss", "r2scan") and not has_linus:
        return "metagga-grid-sensitivity"
    if case.mol in ECP_MOLS and any(x.startswith("psi4") for x in pair):
        return "psi4-ecp-integrals"
    if case.mol == "au2+" and any(x.startswith("orca") for x in pair):
        return "orca-other-state"
    orca_ecp_max = {"energy": 1e-7, "dipole": 1e-6, "gradient": 1e-6, "hessian": 1e-5, "frequency": 0.1}
    if case.mol in ECP_MOLS and not case.dft and any(x.startswith("orca") for x in pair) and d is not None \
            and d <= orca_ecp_max[q]:
        return "orca-ecp-integrals"
    if case.mol in ECP_MOLS and case.dft and not has_linus and d is not None and tol is not None and d <= 30 * tol:
        return "ecp-heavy-atom-grids"
    return None


def compare_case(case: Case, recs: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """The matrix columns (L-P, L-S, L-O; P-S, P-O, S-O) and extra comparisons."""
    out: list[dict[str, Any]] = []
    ran = {r.name for r in runs_for(case)}
    dft = case.dft
    psi4_fd = dft and case.method not in LDA  # Psi4's Hessian = finite differences of gradients
    if dft:
        lv, hv = f"pyscf:{case.level}", f"pyscf:{case.hlevel}"
        L9, P9, LS, S, Sf = f"linus@{lv}", f"pyscf@{lv}", "linus@psi4", "psi4@psi4", "psi4@fine"
        L5, P5 = f"linus@{hv}", f"pyscf@{hv}"
        O, On, Og, Lh = "orca@fine", "orca-numfreq@fine", "orca-numgrad@fine", f"linus-hgr@{hv}"
        LSfd = "linus-fd@psi4" if psi4_fd else LS
        S_ref = Sf if case.psi4_fine else S
    else:
        L9, P9, LS, S, S_ref, O, On, Og, Lh, LSfd = "linus", "pyscf", "linus", "psi4", "psi4", "orca", None, None, None, "linus"
        L5, P5 = L9, P9
    # Cartesian cases: linus vs PySCF given linus's basis data (same grid as the Hessian runs)
    Pb = (f"pyscf-bse@pyscf:{case.hlevel}" if dft else "pyscf-bse") if case.cart else None
    bse_ok: dict[str, bool] = {}
    if Pb:
        for q in QUANTITIES:
            va, vb = values(recs.get(L5), q), values(recs.get(Pb), q)
            if va is not None and vb is not None:
                bse_ok[q] = maxabs(va, vb) <= TOL["exact"][q]
    natoms = max([len((r.get("molecule") or {}).get("symbols", [])) for r in recs.values()] + [3])
    # DFT Hessians: is linus's grid-response Hessian (Bykov scheme 1) within the grid tolerance
    # of ORCA's NumFreq? Then fixed-grid Hessians (PySCF, linus default) differ from NumFreq by
    # the grid-response terms, not by anything else.
    hgr_ok: dict[str, bool] = {}
    if dft and Lh and On:
        for q in ("hessian", "frequency"):
            va, vb = values(recs.get(Lh), q), values(recs.get(On), q)
            if va is not None and vb is not None and va.shape == vb.shape:
                d = maxabs(va, vb)
                if q == "frequency":  # without orca_vib's constant factor
                    sel = np.abs(va) > 100.0
                    r = float(np.median(vb[sel] / va[sel])) if sel.any() else 1.0
                    d = maxabs(r * va, vb)
                hgr_ok[q] = d <= TOL["grid"][q] * max(1.0, natoms / 3.0)

    def add(col: str, a: str | None, b: str | None, q: str, kind: str, residual_of: tuple[str, str] | None = None,
            label: str | None = None):
        if a is None or b is None:
            return
        item = {"column": col, "label": label or col, "quantity": q, "a": a, "b": b, "kind": kind}
        if case.refs != "all" and not {a, b} <= ran:
            if col != "x":  # a matrix cell of a program this case does not run
                item.update({"status": "n/a", "reason": "not available: PySCF-only case (this harness does not set "
                                                        "up Psi4 and ORCA for the feature)"})
                out.append(item)
            return
        ra, rb = recs.get(a), recs.get(b)
        va, vb = values(ra, q), values(rb, q)
        if va is None or vb is None:
            reason = why_missing(ra, q, "linus" if a.startswith("linus") else a) if va is None else \
                why_missing(rb, q, b)
            item.update({"status": "n/a", "reason": reason})
            out.append(item)
            return
        if va.shape != vb.shape:
            item.update({"status": "n/a", "reason": f"shapes differ: {va.shape} vs {vb.shape}"})
            out.append(item)
            return
        d = maxabs(va, vb)
        if kind == "info":  # shown for context, no verdict
            item.update({"max_abs": d, "status": "info"})
            out.append(item)
            return
        # per 3 atoms: grid errors, and integral-screening noise in energies, grow with size
        tol = TOL[kind][q] * (max(1.0, natoms / 3.0) if (kind == "grid" or q == "energy") else 1.0)
        item.update({"max_abs": d, "tol": tol})
        if residual_of is not None:
            r1, r2 = values(recs.get(residual_of[0]), q), values(recs.get(residual_of[1]), q)
            if r1 is not None and r2 is not None:
                res = maxabs(r1, r2)
                item.update({"residual": res, "residual_pair": list(residual_of), "tol": tol + res})
                tol += res
        status = "pass" if d <= tol else "FAIL"
        if q == "frequency" and (a.startswith("orca") != b.startswith("orca")):
            # orca_vib's wavenumbers are a constant factor below the others' for the same Hessian
            # and masses (bench/vib): report the factor and the difference after removing it
            o, other = (va, vb) if a.startswith("orca") else (vb, va)
            ok = np.abs(other) > 100.0
            if ok.any():
                ratio = float(np.median(o[ok] / other[ok]))
                item["orca_frequency_ratio"] = ratio
                item["max_abs_after_ratio"] = float(np.abs(o - ratio * other).max())
                if status == "FAIL" and item["max_abs_after_ratio"] <= tol:
                    status, item["explanation"] = "explained", "orca-vib-frequency-factor"
        if status == "FAIL":
            ex = explain(case, q, a, b, bse_ok, hgr_ok, d, tol)
            if ex:
                status = "explained"
                item["explanation"] = ex
        item["status"] = status
        out.append(item)

    def worst(*kinds: str) -> str:
        return max(kinds, key=lambda k: TOL[k][q])

    for q in QUANTITIES:
        need = {"dipole": "energy", "frequency": "hessian"}.get(q, q)
        if ORDER.index(case.calc) < ORDER.index(need):
            continue
        H = q in ("hessian", "frequency")
        L, P = (L5, P5) if H else (L9, P9)  # DFT Hessians on PySCF's level-5 grid
        # ORCA: DFT Hessians -> NumFreq in the matrix (analytic one in the extras); RI-JK
        # Hessians are always NumFreq (no analytic RIJK Hessian); D3(BJ) gradients -> NumGrad
        o_main = On if (H and dft and case.jk == "exact") else O
        o_kind = "fd2" if (H and (case.jk == "df" or o_main == On)) else "exact"
        if q == "gradient" and orca_numgrad(case) and Og:
            o_main, o_kind = Og, "fd2"
        add("L-P", L, P, q, "exact")
        if H and psi4_fd and q == "frequency" and case.refs == "all":
            out.append({"column": "L-S", "label": "L-S", "quantity": q, "a": LS, "b": S, "kind": "fd5", "status": "n/a",
                        "reason": "Psi4's GGA/meta-GGA Hessian is a finite-difference convention; the Hessian row "
                                  "compares it with the same differences of linus gradients"})
        elif H and psi4_fd:
            add("L-S", LSfd, S, q, "fd5")
        else:
            add("L-S", LS, S, q, "exact")
        if dft:
            add("L-O", L, o_main, q, o_kind, residual_of=(P, o_main))
            if not H:  # Psi4's Hessians are on its default grid only
                fine = S_ref == Sf
                add("P-S", P, S_ref, q, "grid" if fine else "info")
                add("S-O", S_ref, o_main, q, worst("grid", o_kind) if fine else "info")
            add("P-O", P, o_main, q, worst("grid", o_kind))
        else:
            add("L-O", L, O, q, o_kind)
            add("P-S", P, S, q, "exact")
            add("P-O", P, O, q, o_kind)
            add("S-O", S, O, q, o_kind)
        # extras
        if dft and H and case.jk == "exact" and not case.cart:
            add("x", Lh, O, q, "grid", label="linus(hess_grid_response) vs ORCA analytic")
            add("x", L, O, q, "info", label="linus(fixed grid) vs ORCA analytic")
            add("x", O, On, q, "info", label="ORCA analytic vs ORCA NumFreq")
            add("x", Lh, On, q, "fd2", residual_of=(P, On), label="linus(hess_grid_response) vs ORCA NumFreq")
        if H and psi4_fd and LSfd != LS:
            add("x", LS, S, q, "info", label="linus analytic (hess_grid_response, psi4 grid) vs Psi4 FD")
        if dft and not H and case.psi4_fine:
            add("x", L, Sf, q, "grid", residual_of=(P, Sf), label="linus vs Psi4 150x974")
        if dft and not H and case.calc == "hessian":
            add("x", L5, P5, q, "exact", label=f"linus vs PySCF, level-{case.hlevel} grid")
        if Pb:
            add("x", L5, Pb, q, "exact", label="linus vs PySCF given linus's basis data")
        if disp_of(case) == "d3bj" and q == "gradient" and Og:
            add("x", L, O, q, "exact", residual_of=(P, O), label="linus vs ORCA analytic D3(BJ) gradient")
        if disp_of(case) and q == "energy":
            for a, b in ((L, P), (LS, S), (L, O)):
                ea = linus_dispersion_energy(recs.get(a) or {})
                rb = recs.get(b) or {}
                eb = rb.get("dispersion_energy") if rb.get("success") else None
                if ea is not None and eb is not None:
                    d = abs(ea - eb)
                    out.append({"column": "x", "label": f"dispersion energy vs {b.split('@')[0]}", "quantity": "energy",
                                "a": a, "b": b, "kind": "exact", "max_abs": d, "tol": 1e-9,
                                "status": "pass" if d <= 1e-9 else "FAIL"})
    return out


# ------------------------------------------------------------------ driver


def select(cases: list[Case], patterns: list[str] | None, quick: bool, skip: list[str] | None = None) -> list[Case]:
    out = [c for c in cases if (not quick or c.quick)]
    if patterns:
        out = [c for c in out if any(fnmatch.fnmatch(c.id, p) for p in patterns)]
    if skip:
        out = [c for c in out if not any(fnmatch.fnmatch(c.id, p) for p in skip)]
    return out


def store(c: Case, r: Run, rec: dict[str, Any]) -> None:
    spec = spec_of(c, r)
    rec = sanitize(rec)
    rec.update({"case": c.id, "run": r.name, "spec": spec, "spec_hash": spec_hash(spec),
                "date": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")})
    if r.program == "linus":
        rec["linus_git"] = git_commit()
    f = WRITE_DIR / c.id / f"{r.name.replace(':', '')}.json"
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(rec, separators=(",", ":")) + "\n")


def adopt(spec: str) -> int:
    """Store a reference result that run_ref.py wrote into its raw directory although
    validate.py did not wait for it (an interrupted matrix run): CASE:RUN."""
    cid, rname = spec.split(":", 1)
    case = next((c for c in CASES if c.id == cid), None)
    run = next((r for r in runs_for(case) if r.name == rname), None) if case else None
    if run is None or not run.is_ref():
        print(f"validate: no reference run {spec}", file=sys.stderr)
        return 2
    f = RAW_DIR / case.id / run.name.replace(":", "") / f"{run.program}.json"
    if not f.exists():
        print(f"validate: {f} does not exist (still running?)", file=sys.stderr)
        return 2
    rec = json.loads(f.read_text())
    ok = (rec.get("method"), rec.get("basis"), rec.get("calc_type"), rec.get("dft_grid")) == \
        (case.method, case.basis, run.calc, run.grid)
    if not ok or not rec.get("success"):
        print(f"validate: {f} does not match {spec} or failed", file=sys.stderr)
        return 2
    rec["adopted"] = "stored by validate.py --adopt from the raw run directory"
    store(case, run, rec)
    print(f"adopted {spec}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true",
                    help="only the cases marked quick, against the stored references (~4 minutes); writes its linus "
                         "runs and results.json to reference/out/validation/quick/ (not tracked) unless --store")
    ap.add_argument("--store", action="store_true",
                    help="with --quick: store the runs and results.json in reference/validation/ as a full run does")
    ap.add_argument("--core", action="store_true",
                    help="only the core subset: every feature once (HF/LDA/GGA/hybrid/meta-GGA, R/U, exact/RI, "
                         "Cartesian, dispersion, frequencies, range-separated hybrids)")
    ap.add_argument("--cases", nargs="*", help="fnmatch patterns on case ids, e.g. 'water.*' '*.df'")
    ap.add_argument("--refs", choices=("missing", "all", "none"), default=None,
                    help="run references that are missing or whose spec changed (default; with --quick: none), "
                         "all of them, or none")
    ap.add_argument("--linus-bin", type=Path, default=None,
                    help="linus CLI to validate (default $LINUS_BIN or target/release/linus of this checkout)")
    ap.add_argument("--linus", choices=("run", "reuse"), default="run", help="rerun linus (default) or reuse stored results")
    ap.add_argument("--programs", default="linus,pyscf,psi4,orca", help="programs to (re)run")
    ap.add_argument("--lanes", default="build,ccd2,ccd3",
                    help="slice CPU sets, one calculation per lane at a time ('none': no pinning). The default "
                         "keeps one job on CCD1 (builds are bursty) and one each on CCD2 and CCD3")
    ap.add_argument("--threads", type=int, default=4, help="threads / MPI ranks per calculation")
    ap.add_argument("--list", action="store_true", help="list the cases and their runs, then exit")
    ap.add_argument("--dry-run", action="store_true", help="show what would run")
    ap.add_argument("--skip", nargs="*", help="fnmatch patterns of case ids to leave out")
    ap.add_argument("--adopt", nargs="*", help="CASE:RUN reference results left in reference/out/validation by an "
                                               "interrupted run: store them, then exit")
    ap.add_argument("--frequencies-only", action="store_true", help=argparse.SUPPRESS)
    a = ap.parse_args(argv)

    global WRITE_DIR
    scratch = a.quick and not a.store  # write nothing the repository tracks
    WRITE_DIR = QUICK_DIR / "runs" if scratch else RUNS_DIR
    results_out = QUICK_DIR / "results.json" if scratch else RESULTS
    if a.adopt:
        return max(adopt(x) for x in a.adopt)
    cases = select(CASES, a.cases, a.quick, a.skip)
    if a.core:
        cases = [c for c in cases if c.id in CORE]
    if a.frequencies_only:
        analyze_frequencies(cases)
        return 0
    if a.list:
        for c in cases:
            print(f"{c.id:45s} {c.calc:8s} {'quick' if c.quick else '':5s} " + ", ".join(r.name for r in runs_for(c)))
        print(f"{len(cases)} cases")
        return 0
    programs = set(a.programs.split(","))
    global LINUS_BIN
    if a.linus_bin is not None:
        LINUS_BIN = a.linus_bin.resolve()
    if a.refs is None:  # the per-merge check compares against the stored references
        a.refs = "none" if a.quick else "missing"
    if "linus" in programs and a.linus == "run" and not LINUS_BIN.exists():
        print(f"validate: build linus first (cargo build --release): {LINUS_BIN} missing", file=sys.stderr)
        return 2
    jobs: list[tuple[Case, Run, dict, str]] = []
    for c in cases:
        for r in runs_for(c):
            if r.program not in programs:
                continue
            spec = spec_of(c, r)
            h = spec_hash(spec)
            f = run_file(c, r.name)
            if r.is_ref():
                if a.refs == "none":
                    continue
                if a.refs == "missing" and f.exists():
                    old = json.loads(f.read_text())
                    if old.get("spec_hash") == h and old.get("success"):
                        continue
            elif a.linus == "reuse" and f.exists():
                continue
            jobs.append((c, r, spec, h))
    # expensive first: Hessians, references, larger molecules
    weight = {"frequencies": 3, "hessian": 3, "gradient": 2, "energy": 1}
    jobs.sort(key=lambda j: (-(j[1].is_ref()), -weight[j[1].calc], -(j[0].mol == "la18")))
    print(f"{len(cases)} cases, {len(jobs)} runs to do (lanes {a.lanes}, {a.threads} threads each)", flush=True)
    if a.dry_run:
        for c, r, _, h in jobs:
            print(f"  {c.id:45s} {r.name:22s} {r.calc}")
        return 0

    lanes: queue.Queue[str] = queue.Queue()
    for lane in a.lanes.split(","):
        lanes.put(lane)
    nlanes = lanes.qsize()
    done = [0]

    def work(job):
        c, r, spec, h = job
        lane = lanes.get()
        try:
            t0 = time.perf_counter()
            rec = run_linus(c, r, lane, a.threads) if r.program == "linus" else run_reference(c, r, lane, a.threads)
            store(c, r, rec)
            done[0] += 1
            ok = "ok" if rec.get("success") else ("NOT IMPLEMENTED" if "not implemented" in str(rec.get("error", "")).lower() else "FAILED " + str(rec.get("error"))[:200])
            print(f"[{done[0]}/{len(jobs)}] {c.id} {r.name} ({lane}) {time.perf_counter() - t0:.1f}s {ok}", flush=True)
        finally:
            lanes.put(lane)

    with cf.ThreadPoolExecutor(max_workers=nlanes) as ex:
        for fut in [ex.submit(work, j) for j in jobs]:
            fut.result()

    # reference frequencies (each program's own harmonic analysis of its Hessian), in a
    # subprocess run in the raw directory (importing Psi4 writes timer.dat into the cwd)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    subprocess.run([str(PYTHON), str(Path(__file__).resolve()), "--frequencies-only", *(["--quick"] if a.quick else []),
                    *(["--store"] if a.store else []), "--cases", *[c.id for c in cases]], cwd=RAW_DIR, check=False)
    # --quick (scratch): the comparisons of the cases run now; otherwise of every stored case
    write_results(cases if scratch else None, results_out)
    import validation_summary

    return validation_summary.main([str(results_out), "--cases"] + [c.id for c in cases])


def versions(all_recs: list[dict[str, Any]]) -> dict[str, Any]:
    v: dict[str, set] = {}
    for rec in all_recs:
        if not rec.get("success"):
            continue
        prog = rec.get("program")
        ver = rec.get("program_version") or rec.get("version")
        if prog and ver:
            v.setdefault(prog, set()).add(str(ver))
        if rec.get("libxc_version"):
            v.setdefault("pyscf libxc", set()).add(rec["libxc_version"])
    out = {k: sorted(s) for k, s in v.items()}
    try:
        import importlib.metadata as md

        for pkg in ("pyscf-dispersion", "dftd3", "dftd4"):
            pass
        out["qccompute"] = md.version("qccompute")
        out["qccodec"] = md.version("qccodec")
        out["qcref"] = md.version("qcref")
    except Exception:
        pass
    disp = {}
    for d in run_ref.DISP_OVERLAY.glob("*.dist-info") if run_ref.DISP_OVERLAY.is_dir() else []:
        m = re.match(r"(.+)-(\d[^-]*)\.dist-info", d.name)
        if m:
            disp[m.group(1)] = m.group(2)
    out["dispersion_overlay"] = disp
    return out


def write_results(cases: list[Case] | None = None, path: Path = RESULTS) -> None:
    """Recompute the comparisons from the runs and write results.json: of `cases`, else of every
    case with stored runs (so a partial rerun keeps the rest of the matrix)."""
    all_cases = CASES if cases is None else cases
    items, all_recs, linus_git = [], [], set()
    for c in all_cases:
        recs = load_runs(c)
        if not recs:
            continue
        all_recs += list(recs.values())
        for r in recs.values():
            if r.get("program") == "linus" and r.get("linus_git"):
                linus_git.add(r["linus_git"]["describe"])
        items.append({"id": c.id, "case": asdict(c), "molecule": asdict(MOLECULES[c.mol]),
                      "runs": {k: {"program": v.get("program"), "success": v.get("success"),
                                   "error": v.get("error"), "not_implemented": v.get("not_implemented"),
                                   "hessian_method": v.get("hessian_method"), "gradient_method": v.get("gradient_method"),
                                   "date": v.get("date"), "spec_hash": v.get("spec_hash"),
                                   "version": v.get("program_version") or v.get("version")}
                               for k, v in recs.items()},
                      "comparisons": compare_case(c, recs)})
    meta = {
        "generated": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "linus_git": sorted(linus_git),
        "linus_git_now": git_commit(),
        "versions": versions(all_recs),
        "tolerances": TOL,
        "explanations": EXPLAINED,
        "linus_settings": {**LINUS_SCF, **LINUS_ENV},
        "reference_settings": {"scf": run_ref.SCF, "pyscf_conv_tol_grad": run_ref.PYSCF_CONV_TOL_GRAD,
                               "pyscf_conv_tol_cpscf": run_ref.PYSCF_CONV_TOL_CPSCF, "cpscf_tol": run_ref.CPSCF_TOL,
                               "orca_numfreq_step_bohr": ORCA_NUMFREQ_STEP, "orca_numgrad_step_bohr": ORCA_NUMGRAD_STEP, "ref_settings_version": REF_SETTINGS_VERSION,
                               "runner_version": run_ref.RUNNER_VERSION},
        "units": {"energy": "Eh", "dipole": "e bohr (origin 0)", "gradient": "Eh/bohr", "hessian": "Eh/bohr^2"},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"meta": meta, "cases": items}, indent=1) + "\n")


if __name__ == "__main__":
    sys.exit(main())
