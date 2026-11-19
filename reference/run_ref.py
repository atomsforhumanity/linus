#!/usr/bin/env python
"""Reference runner for linus: one molecule, one model, PySCF / Psi4 / ORCA.

Run it with the reference environment's Python (it holds PySCF, Psi4, qc*):

    .refenv/bin/python reference/run_ref.py reference/molecules/water.xyz \
        --method hf --basis def2-svp --calc gradient --threads 8

Writes one JSON per program to
``reference/out/<molecule>/<method>_<basis>_<calc>_<approx>[_grid-<grid>][_<tag>]/<program>.json``
and keeps each program's raw files in ``<program>_run/`` beside it.

Units: energies Eh, gradients Eh/bohr (natoms x 3), Hessians Eh/bohr^2 (3N x 3N),
geometries bohr. The xyz file is read in angstrom and converted once, with CODATA
2022 (qcref, 0.529177210544 A/bohr); every program gets those exact bohr
coordinates (PySCF unit=Bohr; Psi4 via the qccodec driver, ``units bohr``,
``no_com``, ``no_reorient``; ORCA via ``! Bohrs``, because ORCA's own A->bohr factor
is CODATA 1998, 0.5291772083, which moves Vnn by ~4e-8 Eh for water). Nothing is
reoriented or recentred, so gradients and Hessians are in the input frame.

Settings are explicit and tight by default (see ``SCF`` below, and ``settings`` in
each JSON): SCF energy change 1e-12, orbital-gradient/density 1e-10, no integral
approximations unless ``--approx`` asks for one, spherical (pure) functions, no
symmetry. ORCA and Psi4 run through qccompute codecs; PySCF has no codec and runs
in-process (a qc-suite gap, noted in reference/README.md).
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import platform
import re
import shutil
import socket
import sys
import time
import traceback
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]


def _shared_dir(name: str) -> Path:
    """`name` (e.g. .refenv) in the repository, or in the main checkout when this is a
    git worktree (.claude/worktrees/<agent>/ has no environments of its own)."""
    for d in (REPO, *REPO.parents):
        if (d / name).exists():
            return d / name
    return REPO / name


REFENV = Path(os.environ.get("LINUS_REFENV", _shared_dir(".refenv")))
# pip --target overlay with pyscf-dispersion 1.7.0, dftd3 1.7.0, dftd4 4.3.1 (+ cffi) for
# PySCF's mf.disp and Psi4's -d3bj/-d4 (reference/validation/README.md has the commands)
DISP_OVERLAY = Path(os.environ.get("LINUS_REFENV_DISP", _shared_dir(".refenv-disp")))
ORCA_DIR = Path(os.environ.get("LINUS_ORCA_DIR", Path.home() / "opt" / "orca-6.0.0"))
RUNNER_VERSION = "2026-11-08.1"
PROGRAMS = ("pyscf", "psi4", "orca")

# ---------------------------------------------------------------- method maps

# canonical name -> (PySCF xc, Psi4 method, ORCA simple-input keyword)
# "b3lyp" is the VWN_RPA flavour (libxc HYB_GGA_XC_B3LYP, Gaussian's B3LYP), which
# is ORCA's B3LYP/G. ORCA's plain "B3LYP" is the VWN5 flavour = "b3lyp5" here.
# "svwn-rpa" is Slater + VWN_RPA (libxc id 8): Psi4's only SVWN, PySCF "slater,vwn3",
# ORCA "VWN3" (ORCA's VWN-3 is the RPA fit). PySCF's and ORCA's plain SVWN/LDA are VWN5.
FUNCTIONALS: dict[str, tuple[str | None, str, str]] = {
    "hf": (None, "hf", "HF"),
    "b3lyp": ("b3lyp", "b3lyp", "B3LYP/G"),
    "b3lyp5": ("b3lyp5", "b3lyp5", "B3LYP"),
    "svwn-rpa": ("slater,vwn3", "svwn", "VWN3"),
    "pbe": ("pbe", "pbe", "PBE"),
    "pbe0": ("pbe0", "pbe0", "PBE0"),
    "blyp": ("blyp", "blyp", "BLYP"),
    "tpss": ("tpss", "tpss", "TPSS"),
    "r2scan": ("r2scan", "r2scan", "r2SCAN"),
    "wb97x": ("wb97x", "wb97x", "wB97X"),
    # wB97X-V's semilocal part without VV10, plus two-body D3(BJ) (Najibi & Goerigk 2018)
    "wb97x-d3bj": ("wb97x-d3bj", "wb97x-d3bj", "wB97X-D3BJ"),
    "cam-b3lyp": ("camb3lyp", "cam-b3lyp", "CAM-B3LYP"),
    # VV10 nonlocal correlation on (PySCF: mf.nlc from the functional, nlcgrids level --nlc-level)
    "wb97m-v": ("wb97m-v", "wb97m-v", "wB97M-V"),
    "wb97x-v": ("wb97x-v", "wb97x-v", "wB97X-V"),
    "b97m-v": ("b97m-v", "b97m-v", "B97M-V"),
}
# functionals with VV10 nonlocal correlation
VV10 = {"wb97m-v", "wb97x-v", "b97m-v"}
# implicit solvent models (--pcm) -> PySCF's names; PySCF only (mf.PCM(), mf.SMD())
PCM_MODELS = {"cpcm": "C-PCM", "cosmo": "COSMO", "iefpcm": "IEF-PCM", "ssvpe": "SS(V)PE", "smd": "SMD"}
# methods whose names carry an empirical dispersion term (need the dispersion overlay)
IMPLIED_DISPERSION = {"wb97x-d3bj": "d3bj"}
# Mappings checked against each other in reference/README.md's cross-check table and in
# the validation matrix (reference/validation/README.md).
VERIFIED_FUNCTIONALS = {"hf", "b3lyp", "pbe", "svwn-rpa", "tpss"}  # pbe: ORCA needs ORCA_LIBXC_PBE
# empirical dispersion: canonical -> (PySCF mf.disp, Psi4 method suffix, ORCA keyword).
# All three are two-body D3(BJ) (PySCF "d3bj" has s9 = 0, Psi4 "-d3bj" is d3bj2b, ORCA
# D3BJ has no ATM term) and D4 with its default ATM term.
DISPERSION = {"d3bj": ("d3bj", "-d3bj", "D3BJ"), "d4": ("d4", "-d4", "D4")}

# auxiliary basis aliases -> per-program names (BSE names for PySCF/Psi4)
AUX_BASES: dict[str, dict[str, str]] = {
    "def2/j": {"pyscf": "def2-universal-jfit", "psi4": "def2-universal-jfit", "orca": "def2/J"},
    "def2/jk": {"pyscf": "def2-universal-jkfit", "psi4": "def2-universal-jkfit", "orca": "def2/JK"},
    "cc-pvdz/jk": {"pyscf": "cc-pvdz-jkfit", "psi4": "cc-pvdz-jkfit", "orca": "cc-pVDZ/JK"},
    "cc-pvtz/jk": {"pyscf": "cc-pvtz-jkfit", "psi4": "cc-pvtz-jkfit", "orca": "cc-pVTZ/JK"},
    "cc-pvqz/jk": {"pyscf": "cc-pvqz-jkfit", "psi4": "cc-pvqz-jkfit", "orca": "cc-pVQZ/JK"},
}

# approximation spec -> what it means
APPROX = {
    "exact": "exact 4-index J and K (no RI/DF/COSX)",
    "df": "density fitting (RI) for both J and K (ORCA RIJK, Psi4 MEM_DF, PySCF density_fit)",
    "rij": "RI/DF for J only, exact K (ORCA RIJONX or RI, Psi4 DFDIRJ+LINK, PySCF only_dfj)",
    "rijcosx": "RI-J + seminumerical K (ORCA RIJCOSX, Psi4 DFDIRJ+COSX, PySCF sgX); grids differ",
}

# Tight SCF settings (used for every correctness run)
SCF = {"e_conv": 1e-12, "d_conv": 1e-10, "maxiter": 300, "int_thresh": 1e-14}
# PySCF conv_tol_grad is the norm of the whole orbital gradient (Psi4/ORCA use RMS or
# max values), ~sqrt(nocc*nvirt) larger than the RMS; 1e-9 there is RMS < 1e-10 here.
# (Unset, PySCF uses sqrt(conv_tol).)
PYSCF_CONV_TOL_GRAD = 1e-9
# PySCF's CPHF tolerance is mf.conv_tol_cpscf (default 1e-8), multiplied by the number of
# atoms solved together (hessian/rhf.py); Hessian.conv_tol is not used by the CPHF solver.
# 1e-12 is too tight for PySCF's Krylov solver: it stalls near a true residual of 1e-7 and
# falls back to slow per-root solves (la18), or hits a singular subspace (water SVWN).
PYSCF_CONV_TOL_CPSCF = 1e-10
# PySCF (2.15) drops eigenvectors of the overlap matrix below 1e-6 (scf.hf.
# overlap_zero_eigenvalue_threshold, canonical orthogonalization), linus below 1e-8 (scf.lindep):
# set to linus's so a basis with an eigenvalue in between (cubane/def2-TZVP: 6.8e-7, 2.2e-6 Eh in
# HF; bench/robust/repro/pyscf_overlap_threshold) is the same space in both programs
PYSCF_OVERLAP_THRESHOLD = 1e-8
# CP-SCF / CPHF convergence for analytic Hessians in Psi4 (solver_convergence) and ORCA (Z_Tol)
CPSCF_TOL = 1e-10

# DFT grids: "fine" = the finest grid each program offers (ORCA has no Lebedev-974)
FINE_RAD, FINE_ANG = 150, 974
ORCA_FINE_GRID = {
    "intacc": 8.0,
    "angulargrid": 7,  # Lebedev-770, ORCA 6.0.0's largest (8 is rejected)
    "gridpruning": 0,  # no pruning
    "hgridreduced": False,  # do not drop H/He one angular level
    "bfcut": 1e-15,
}
# libxc's PBE constants (beta full precision, mu = beta*pi^2/3), set in ORCA's %method
ORCA_LIBXC_PBE = {"xmuepbe": 0.2195149727645171, "cbetapbe": 0.06672455060314922}


def die(msg: str) -> None:
    print(f"run_ref: error: {msg}", file=sys.stderr)
    sys.exit(2)


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9.+-]+", "-", text.lower()).strip("-")


def default_aux(basis: str, approx: str) -> str | None:
    b = basis.lower()
    if approx == "exact":
        return None
    if approx in ("rij", "rijcosx"):
        return "def2/j"
    m = re.match(r"(aug-)?cc-p[cw]?v([dtq])z", b)
    if m:
        return f"cc-pv{m.group(2)}z/jk"
    return "def2/jk"


# ---------------------------------------------------------------- molecule


def read_xyz(path: Path) -> tuple[list[str], list[list[float]], str]:
    lines = path.read_text().splitlines()
    n = int(lines[0].split()[0])
    symbols, coords = [], []
    for line in lines[2 : 2 + n]:
        parts = line.split()
        symbols.append(parts[0].capitalize())
        coords.append([float(x) for x in parts[1:4]])
    return symbols, coords, lines[1]


def ecp_core_electrons(symbols: list[str], basis: str) -> list[int]:
    """Core electrons an ECP of `basis` replaces per atom (the repository's BSE data; 0 without
    an ECP). The nuclear charges every program uses are Z minus these."""
    stem = basis.lower().replace("*", "_st_").replace("+", "_pl_").replace("/", "_sl_").replace("(", "_lp_").replace(")", "_rp_").replace(",", "_cm_")
    path = REPO / "data" / "basis" / f"{stem}.json"
    if not path.exists():
        return [0] * len(symbols)
    from qcref import periodic_table

    els = json.loads(path.read_text())["elements"]
    out = []
    for s in symbols:
        z = periodic_table.symbol_to_number(s) if hasattr(periodic_table, "symbol_to_number") else getattr(periodic_table, s).number
        out.append(int(els.get(str(z), {}).get("ecp_electrons", 0) or 0))
    return out


def nuclear_repulsion(Z: list[int], geom_bohr) -> float:
    import numpy as np

    g = np.asarray(geom_bohr)
    e = 0.0
    for i in range(len(Z)):
        for j in range(i):
            e += Z[i] * Z[j] / float(np.linalg.norm(g[i] - g[j]))
    return e


# ---------------------------------------------------------------- PySCF


def pyscf_vv10_hessian_fixed_grid() -> None:
    """PySCF 2.15's RKS Hessian of VV10 functionals takes the VV10 Fock-matrix derivative with
    grid response (`grid_response = True` hard-coded in hessian.rks._get_vnlc_deriv1) but the VV10
    energy second derivative (_get_enlc_deriv2) and every other XC term on a fixed grid. Patch the
    first to a fixed grid, so that the whole Hessian is the fixed-grid one (linus with
    hess_grid_response false); idempotent."""
    import inspect

    from pyscf.hessian import rks as rks_hess

    if getattr(rks_hess._get_vnlc_deriv1, "_fixed_grid", False):
        return
    src = inspect.getsource(rks_hess._get_vnlc_deriv1)
    assert "    grid_response = True\n" in src, "PySCF's _get_vnlc_deriv1 changed"
    exec(compile(src.replace("    grid_response = True\n", "    grid_response = False\n"), rks_hess.__file__, "exec"),
         rks_hess.__dict__)
    rks_hess._get_vnlc_deriv1._fixed_grid = True


def pyscf_ecp_grid_response_fix() -> None:
    """PySCF 2.15's grids_response_becke (DFT gradients with grid response) looks up the
    Bragg radii of the Becke-Treutler atomic-size adjustment with mol.atom_charges(), which are
    Z - ncore for ECP atoms, while the energy grid (radi.treutler_atomic_radii_adjust) uses the
    proton numbers, so the gradient is not the derivative of the energy (CsF B3LYP/def2-TZVP:
    up to 5e-5 Eh/bohr from central differences of PySCF's own energies). Patch it to call the
    original with the proton numbers in mol._atm (idempotent)."""
    import copy

    from pyscf import gto
    from pyscf.grad import rks as rks_grad

    orig = rks_grad.grids_response_becke
    if getattr(orig, "_ecp_fix", False):
        return

    def fixed(grids):
        m = grids.mol.copy()
        m._atm = m._atm.copy()
        m._atm[:, 0] = [gto.charge(s) for s in m.elements]
        g2 = copy.copy(grids)
        g2.mol = m
        return orig(g2)

    fixed._ecp_fix = True
    rks_grad.grids_response_becke = fixed


def run_pyscf(job: dict[str, Any], rundir: Path) -> dict[str, Any]:
    import numpy as np
    import pyscf
    from pyscf import dft, gto, lib, scf

    lib.num_threads(job["threads"])
    pxc, _, _ = FUNCTIONALS.get(job["method"], (job["method"], None, None))
    is_dft = job["method"] != "hf"
    open_shell = job["multiplicity"] != 1
    from pyscf.scf import hf as pyscf_hf

    pyscf_hf.overlap_zero_eigenvalue_threshold = PYSCF_OVERLAP_THRESHOLD

    mol = gto.Mole()
    mol.atom = [(s, tuple(x)) for s, x in zip(job["symbols"], job["geometry_bohr"])]
    mol.unit = "Bohr"
    mol.basis = job["basis"]
    if job.get("pyscf_linus_basis"):
        # linus's basis data (data/basis, BSE JSON) instead of PySCF's library: separates
        # basis-data differences (e.g. digits of 6-31G* coefficients) from code differences
        sys.path.insert(0, str(REPO / "tests" / "python"))
        from compare_ints_pyscf import pyscf_basis

        mol.basis = pyscf_basis(job["basis"], sorted(set(job["symbols"])))
    # effective core potentials: PySCF adds them only on request (mol.ecp); its def2 ECPs
    # (Rb-Rn) are the Basis Set Exchange ones that linus, Psi4 and ORCA use
    if any(gto.charge(s) >= 37 for s in job["symbols"]):
        mol.ecp = job["basis"]
    mol.charge = job["charge"]
    mol.spin = job["multiplicity"] - 1
    mol.cart = bool(job.get("cart"))
    mol.symmetry = False
    mol.verbose = 4
    mol.output = str(rundir / "pyscf.log")
    mol.max_memory = job.get("pyscf_max_memory_mb") or job["memory_gb"] * 1000
    mol.build()
    frame_err = float(np.abs(mol.atom_coords(unit="Bohr") - np.asarray(job["geometry_bohr"])).max())

    if is_dft:
        mf = (dft.UKS if open_shell else dft.RKS)(mol)
        mf.xc = pxc
    else:
        mf = (scf.UHF if open_shell else scf.RHF)(mol)
    mf.conv_tol = SCF["e_conv"]
    mf.conv_tol_grad = PYSCF_CONV_TOL_GRAD
    mf.conv_tol_cpscf = PYSCF_CONV_TOL_CPSCF
    mf.max_cycle = SCF["maxiter"]
    mf.direct_scf_tol = SCF["int_thresh"]
    if job.get("disp"):
        mf.disp = DISPERSION[job["disp"]][0]

    settings: dict[str, Any] = {
        "scf_class": type(mf).__name__,
        "xc": pxc,
        "conv_tol (energy)": mf.conv_tol,
        "conv_tol_grad (orbital gradient norm)": mf.conv_tol_grad,
        "conv_tol_cpscf (CPHF, per atom)": mf.conv_tol_cpscf,
        "max_cycle": mf.max_cycle,
        "direct_scf_tol": mf.direct_scf_tol,
        "init_guess": mf.init_guess,
        "cart": mol.cart,
        "basis_source": "linus data/basis (BSE JSON)" if job.get("pyscf_linus_basis") else "PySCF library",
        "symmetry": mol.symmetry,
        "unit": "Bohr",
        "max_memory_mb": mol.max_memory,
        "overlap_zero_eigenvalue_threshold": pyscf_hf.overlap_zero_eigenvalue_threshold,
    }
    if job.get("disp"):
        settings["disp"] = mf.disp
    aux = job["aux"]
    if job["approx"] == "df":
        mf = mf.density_fit(auxbasis=AUX_BASES[aux]["pyscf"])
        settings["density_fit"] = {"auxbasis": AUX_BASES[aux]["pyscf"], "only_dfj": False}
    elif job["approx"] == "rij":
        mf = mf.density_fit(auxbasis=AUX_BASES[aux]["pyscf"], only_dfj=True)
        settings["density_fit"] = {"auxbasis": AUX_BASES[aux]["pyscf"], "only_dfj": True}
    elif job["approx"] == "rijcosx":
        from pyscf import sgx

        mf = sgx.sgx_fit(mf, auxbasis=AUX_BASES[aux]["pyscf"])
        mf.with_df.dfj = True
        settings["sgx"] = {"auxbasis": AUX_BASES[aux]["pyscf"], "dfj": True,
                           "grids_level_i": mf.with_df.grids_level_i,
                           "grids_level_f": mf.with_df.grids_level_f}
    if job.get("wk"):
        # long-range (erf) exchange fitted as Psi4 and linus do (wK: Coulomb metric,
        # K_LR = sym[(ml|P) V^-1 (P|erf|ns)] D_ls) instead of PySCF's erf metric; only get_jk is
        # patched, so PySCF's DF gradient would not be the derivative of this energy
        if job["approx"] != "df":
            raise RuntimeError("--wk needs --approx df")
        if job["calc"] != "energy":
            raise RuntimeError("--wk: energies only (PySCF's DF gradient fits K_LR with the erf metric)")
        sys.path.insert(0, str(REPO / "tests" / "python"))
        from compare_rsh_pyscf import wk_get_jk

        mf = wk_get_jk(mf, AUX_BASES[aux]["pyscf"])
        settings["density_fit"]["long_range_exchange"] = (
            "wK (tests/python/compare_rsh_pyscf.wk_get_jk: Coulomb metric, (P|erf|ns) 3-index integrals)")

    if is_dft:
        grid = job["grid"]
        if grid == "fine":
            mf.grids.atom_grid = (FINE_RAD, FINE_ANG)
            mf.grids.prune = None
            mf.small_rho_cutoff = 0.0
        elif grid == "default":
            pass
        elif grid.startswith("pyscf:"):
            # PySCF's own level-N grid (what linus --grid pyscf:N reproduces point by point);
            # small_rho_cutoff 0: PySCF otherwise drops low-density points after the first
            # SCF cycle (default 1e-7), which linus does not do
            mf.grids.level = int(grid.split(":")[1])
            mf.small_rho_cutoff = 0.0
        elif grid == "psi4":
            raise RuntimeError("grid 'psi4' is Psi4's default grid; PySCF cannot build it")
        else:
            r, a = (int(x) for x in grid.split(","))
            mf.grids.atom_grid = (r, a)
            mf.grids.prune = None
            mf.small_rho_cutoff = 0.0
        settings["grid"] = {
            "atom_grid": mf.grids.atom_grid if mf.grids.atom_grid else None,
            "level": mf.grids.level,
            "prune": getattr(mf.grids.prune, "__name__", mf.grids.prune),
            "radi_method": mf.grids.radi_method.__name__,
            "becke_scheme": mf.grids.becke_scheme.__name__,
            "radii_adjust": getattr(mf.grids.radii_adjust, "__name__", mf.grids.radii_adjust),
            "small_rho_cutoff": mf.small_rho_cutoff,
        }
        if job["method"] in VV10:
            # VV10 on its own grid: PySCF level --nlc-level (PySCF 2.15's default nlcgrids is
            # level 3; linus grid.nlc pyscf:3), not pruned by density (small_rho_cutoff 0)
            mf.nlcgrids.level = job["nlc_level"]
            settings["nlc_grid"] = {"level": mf.nlcgrids.level,
                                    "prune": getattr(mf.nlcgrids.prune, "__name__", mf.nlcgrids.prune)}

    if job.get("pcm"):
        # implicit solvent as linus's `pyscf` PCM preset: PySCF's radii (modified Bondi x 1.2),
        # SWIG discretization, `pcm_lebedev` points per sphere; SMD with PySCF's solvent
        # descriptors (eps from its table) and the exact SASA (linus solvent.cds_contact_smoothing 0)
        from pyscf.dft import gen_grid
        from pyscf.solvent import pcm as _pcm  # noqa: F401  (registers mf.PCM)
        from pyscf.solvent import smd as _smd  # noqa: F401  (registers mf.SMD)

        leb = job["pcm_lebedev"]
        order = {v: k for k, v in gen_grid.LEBEDEV_ORDER.items()}[leb]
        if job["pcm"] == "smd":
            mf = mf.SMD()
            sol = mf.with_solvent
            sol.solvent = job["solvent"]
            sol.sasa_ng = leb
            eps = float(_smd.solvent_db[job["solvent"]][5])
            if job.get("eps") is not None and job["eps"] != eps:
                raise RuntimeError(f"SMD {job['solvent']}: eps {job['eps']} given, PySCF's table has {eps}")
        else:
            mf = mf.PCM()
            sol = mf.with_solvent
            sol.method = PCM_MODELS[job["pcm"]]
            sol.eps = eps = job["eps"]
        sol.lebedev_order = order
        sol.surface_discretization_method = "SWIG"
        settings["solvent"] = {"model": PCM_MODELS[job["pcm"]], "solvent": job.get("solvent"), "eps": eps,
                               "lebedev_order": order, "points_per_sphere": leb,
                               "surface_discretization_method": sol.surface_discretization_method}

    t0 = time.perf_counter()
    mf.kernel()
    if not mf.converged:
        raise RuntimeError("PySCF SCF did not converge")
    t_scf = time.perf_counter() - t0
    settings["eri_incore"] = getattr(mf, "_eri", None) is not None
    if is_dft:
        settings["grid"]["n_points"] = int(mf.grids.weights.size)
    res: dict[str, Any] = {
        "energy": float(mf.e_tot),
        "nuclear_repulsion": float(mol.energy_nuc()),
        "scf_iterations": int(getattr(mf, "cycles", -1)),
        "nbasis": int(mol.nao_nr()),
        "energy_components_native": {k: float(v) for k, v in getattr(mf, "scf_summary", {}).items()},
        "frame_max_abs_error_bohr": frame_err,
        # e bohr about the coordinate origin (PySCF's default origin is (0, 0, 0))
        "dipole": np.asarray(mf.dip_moment(unit="AU", origin=(0.0, 0.0, 0.0), verbose=0)).tolist(),
    }
    if job.get("disp") or job["method"] in IMPLIED_DISPERSION:
        res["dispersion_energy"] = float(mf.get_dispersion())
    if job.get("pcm"):
        res["solvation_energy"] = float(mf.with_solvent.e)
        res["pcm_points"] = int(mf.with_solvent.surface["grid_coords"].shape[0])
        if job["pcm"] == "smd":
            res["cds_energy"] = float(mf.with_solvent.e_cds)
    if is_dft and job["method"] in VV10:
        settings["nlc_grid"]["n_points"] = int(mf.nlcgrids.weights.size)
    if job["calc"] in ("gradient", "hessian"):
        g = mf.nuc_grad_method()
        if is_dft:
            g.grid_response = job["grid_response"]
            settings["gradient_grid_response"] = g.grid_response
            if g.grid_response and mol.has_ecp():
                pyscf_ecp_grid_response_fix()
                settings["gradient_grid_response_fix"] = (
                    "grids_response_becke given the proton numbers for the Becke-Treutler radii "
                    "(PySCF 2.15 uses mol.atom_charges() = Z - ncore there; reference/validation/README.md)")
        res["gradient"] = np.asarray(g.kernel()).tolist()
    if job["calc"] == "hessian":
        if is_dft and job["method"] in VV10:
            pyscf_vv10_hessian_fixed_grid()
        h = mf.Hessian()
        h.max_cycle = 200
        settings["hessian"] = {"cphf_tol": f"mf.conv_tol_cpscf {mf.conv_tol_cpscf:g} x atoms per CPHF block",
                               "max_cycle": h.max_cycle, "analytic": True,
                               "grid_response": False if is_dft else None,
                               "auxbasis_response": getattr(h, "auxbasis_response", None)}
        if is_dft and job["method"] in VV10:
            settings["hessian"]["vv10_fock_derivative"] = ("fixed grid (patched; PySCF 2.15 hard-codes grid "
                                                           "response in _get_vnlc_deriv1 only)")
        try:
            hh = np.asarray(h.kernel())  # (natm, natm, 3, 3)
        except np.linalg.LinAlgError:
            # PySCF's Krylov CPHF solver can hit a singular subspace matrix when asked to
            # converge this far (water SVWN on the level-5 grid); loosen 100x and retry
            mf.conv_tol_cpscf *= 100
            settings["hessian"]["cphf_tol"] = (f"mf.conv_tol_cpscf {mf.conv_tol_cpscf:g} x atoms per CPHF block "
                                               "(retry: LinAlgError at 100x tighter)")
            h = mf.Hessian()
            h.max_cycle = 200
            hh = np.asarray(h.kernel())
        n = hh.shape[0]
        res["hessian"] = hh.transpose(0, 2, 1, 3).reshape(3 * n, 3 * n).tolist()
        res["hessian_method"] = "analytic" + (" (fixed grid: no grid-weight or point-motion terms)" if is_dft else "")
        if job.get("disp"):
            res["hessian_method"] += "; D3 part: central differences of dftd3 gradients (PySCF)" if job["disp"] == "d3bj" else ""
    res["settings"] = settings
    res["program_version"] = pyscf.__version__
    res["scf_wall_time_s"] = t_scf
    try:
        from pyscf.dft import libxc

        res["libxc_version"] = libxc.__version__
    except Exception:
        pass
    return res


def pyscf_only(job: dict[str, Any], program: str) -> None:
    """Options this runner implements for PySCF only."""
    if job.get("pcm"):
        raise RuntimeError(f"implicit solvent (--pcm) is set up for PySCF only, not {program}")
    if job.get("wk"):
        raise RuntimeError(f"--wk is a PySCF patch; {program} uses its own range-separated RI")


# ---------------------------------------------------------------- Psi4


def run_psi4(job: dict[str, Any], rundir: Path) -> dict[str, Any]:
    pyscf_only(job, "Psi4")
    from qccompute import compute
    from qcdata import CalcType, Model, ProgramInput, Structure

    _, pmethod, _ = FUNCTIONALS.get(job["method"], (None, job["method"], None))
    if job.get("disp"):
        pmethod += DISPERSION[job["disp"]][1]
    is_dft = job["method"] != "hf"
    open_shell = job["multiplicity"] != 1
    kw: dict[str, Any] = {
        "reference": "uhf" if open_shell else "rhf",  # Psi4 maps rhf/uhf to rks/uks for DFT
        "e_convergence": SCF["e_conv"],
        "d_convergence": SCF["d_conv"],
        "maxiter": SCF["maxiter"],
        "ints_tolerance": SCF["int_thresh"],
        "puream": not job.get("cart"),
        "symmetry": "c1",  # molecule directive (codec); frame is fixed: no_com, no_reorient
        "fail_on_maxiter": True,
    }
    aux = job["aux"]
    if job["approx"] in ("rij", "rijcosx") and job["calc"] != "energy":
        # Psi4 1.11 computes DFDIRJ+LINK / DFDIRJ+COSX energies, but its gradient code
        # (DFJKGrad) density-fits K too: the gradient is not the energy's derivative
        # (1.7e-4 Eh/bohr off for water HF/def2-SVP). Refuse rather than mislead.
        raise RuntimeError(f"Psi4 1.11 {job['approx']} derivatives are inconsistent with its "
                           "energy (gradient uses DF-K); use approx=df or exact for Psi4")
    if job["approx"] == "exact":
        kw["scf_type"] = job["psi4_scf_type"]  # pk (default) or direct
        if kw["scf_type"] == "direct":
            kw["df_scf_guess"] = False
    elif job["approx"] == "df":
        kw["scf_type"] = "mem_df"  # in-core; never silently DiskDFJK
        kw["df_basis_scf"] = AUX_BASES[aux]["psi4"]
    elif job["approx"] == "rij":
        kw["scf_type"] = "dfdirj+link"
        kw["df_basis_scf"] = AUX_BASES[aux]["psi4"]
    elif job["approx"] == "rijcosx":
        kw["scf_type"] = "dfdirj+cosx"
        kw["df_basis_scf"] = AUX_BASES[aux]["psi4"]
    if is_dft:
        grid = job["grid"]
        if grid == "psi4":
            # Psi4's default grid (75 Treutler radial x 302 Lebedev, unpruned, Treutler/Becke
            # partitioning: what linus --grid psi4 builds), without Psi4's basis-extent point
            # removal, which linus does not do
            kw.update({"dft_remove_distant_points": False, "dft_basis_tolerance": 1e-15})
        elif grid.startswith("pyscf:"):
            raise RuntimeError(f"grid {grid!r} is PySCF's; Psi4 cannot build it")
        elif grid != "default":
            r, a = (FINE_RAD, FINE_ANG) if grid == "fine" else (int(x) for x in grid.split(","))
            kw.update({
                "dft_radial_points": r,
                "dft_spherical_points": a,
                "dft_pruning_scheme": "none",
                "dft_radial_scheme": "treutler",
                "dft_nuclear_scheme": "treutler",
                "dft_basis_tolerance": 1e-15,
            })
    hess_mode = job.get("psi4_hessian", "auto")
    if job["calc"] == "hessian":
        kw["solver_convergence"] = CPSCF_TOL  # CPHF (analytic RHF/UHF/LDA Hessians)
        kw["solver_maxiter"] = 200
        kw["points"] = 5  # finite differences (GGA/meta-GGA Hessians are FD of gradients)
        kw["fd_project"] = False  # keep the full Cartesian Hessian (non-stationary geometries)
        if hess_mode == "fd":
            kw["dertype"] = 1  # codec: driver keyword, finite differences of analytic gradients
        elif hess_mode == "analytic":
            kw["dertype"] = 2
    structure = Structure(
        symbols=job["symbols"], geometry=job["geometry_bohr"],
        charge=job["charge"], multiplicity=job["multiplicity"],
    )
    calctype = {"energy": CalcType.energy, "gradient": CalcType.gradient,
                "hessian": CalcType.hessian}[job["calc"]]
    env = {}
    if job.get("disp") or job["method"] in IMPLIED_DISPERSION:
        # Psi4's -d3bj / -d4 run through qcengine's s-dftd3 / dftd4 harnesses (python dftd3, dftd4)
        env["PYTHONPATH"] = str(DISP_OVERLAY)

    def run(calctype, keywords, sub: Path):
        sub.mkdir(parents=True, exist_ok=True)
        inp = ProgramInput(
            program="psi4", calctype=calctype, structure=structure,
            model=Model(method=pmethod, basis=job["basis"]), keywords=keywords,
            cmdline_args=["-n", str(job["threads"]), "--memory", f"{job['memory_gb']}GB"],
        )
        out = compute(
            inp, scratch_dir=sub, rm_scratch_dir=False, raise_exc=False,
            executables={"psi4": str(REFENV / "bin" / "psi4")}, environment=env,
        )
        out.save(sub / "qcoutput.json")
        stdout = out.logs or ""  # the codec runs "psi4 input.py -o stdout": output is the log
        (sub / "psi4.out").write_text(stdout)
        if not out.success:
            raise RuntimeError(f"Psi4 failed: {out.traceback}")
        return inp, out, stdout

    inp, out, stdout = run(calctype, kw, rundir)
    r = out.results
    variables = (r.extras or {}).get("psi4_variables", {})
    disk = bool(re.search(r"DiskDFJK|Algorithm:\s+Disk", stdout))
    res: dict[str, Any] = {
        "energy": float(r.energy),
        "nuclear_repulsion": float(r.nuclear_repulsion_energy),
        "scf_iterations": int(variables.get("SCF ITERATIONS", -1)),
        "nbasis": r.calcinfo_nbasis,
        "energy_components_native": {
            k: v for k, v in variables.items()
            if k in ("ONE-ELECTRON ENERGY", "TWO-ELECTRON ENERGY", "DFT XC ENERGY",
                     "NUCLEAR REPULSION ENERGY", "DFT VV10 ENERGY", "DISPERSION CORRECTION ENERGY")
        },
        "program_version": r.provenance.program_version,
        "native_input": json.loads((rundir / "job.json").read_text()),
        "codec_wall_time_s": out.execution.wall_time,
    }
    if "DISPERSION CORRECTION ENERGY" in variables:
        res["dispersion_energy"] = variables["DISPERSION CORRECTION ENERGY"]
    if r.gradient is not None and job["calc"] in ("gradient", "hessian"):
        res["gradient"] = r.gradient.tolist()
    fd_hessian = False
    if job["calc"] == "hessian":
        res["hessian"] = r.hessian.tolist()
        if "Using finite-differences of gradients" in stdout:
            fd_hessian = True
            res["hessian_method"] = (f"finite differences of analytic gradients ({kw['points']}-point, "
                                     f"disp_size 0.005, fd_project false)")
        elif "hessian() will perform analytic" in stdout:
            res["hessian_method"] = "analytic" + (" (fixed grid)" if is_dft else "")
        else:
            res["hessian_method"] = "unknown"
    # SCF DIPOLE is about the coordinate origin. After a finite-difference Hessian it may
    # belong to a displaced geometry, so take it from a separate single point then.
    if fd_hessian:
        ekw = {k: v for k, v in kw.items() if k not in ("dertype", "points", "fd_project",
                                                        "solver_convergence", "solver_maxiter")}
        _, eout, _ = run(CalcType.energy, ekw, rundir / "dipole_energy")
        res["dipole"] = list(eout.results.scf_dipole_moment)
        res["dipole_source"] = "separate single point (the FD Hessian run's SCF DIPOLE may be a displaced geometry's)"
        res["energy_single_point"] = float(eout.results.energy)
    elif r.scf_dipole_moment is not None:
        res["dipole"] = list(r.scf_dipole_moment)
    if disk:
        res["warnings"] = ["Psi4 used disk-based DF (DiskDFJK)"]
    settings = {k: v for k, v in kw.items()}
    settings["method_string"] = pmethod
    settings["molecule"] = "units bohr, no_com, no_reorient, symmetry c1 (qccodec psi4 driver)"
    settings["cmdline_args"] = inp.cmdline_args
    if job.get("disp"):
        m = re.search(r"(?i)engine\s*[:=]?\s*(s-dftd3|dftd3|dftd4)", stdout)
        settings["dispersion_engine"] = m.group(1) if m else None
    res["settings"] = settings
    return res


# ---------------------------------------------------------------- ORCA


def orca_environment() -> dict[str, str]:
    return {
        "PATH": f"{ORCA_DIR}:{REFENV / 'bin'}:{os.environ.get('PATH', '')}",
        "LD_LIBRARY_PATH": f"{ORCA_DIR / 'lib'}:{REFENV / 'lib'}",
        "OMP_NUM_THREADS": "1",
        # quiet, robust single-node Open MPI 4.1.6 (no openib, no CMA: ptrace_scope=1)
        "OMPI_MCA_btl": "self,vader",
        "OMPI_MCA_btl_vader_single_copy_mechanism": "none",
        "OMPI_MCA_hwloc_base_binding_policy": "none",
        "OMPI_MCA_rmaps_base_oversubscribe": "1",
    }


def orca_dipole(path: Path, job: dict[str, Any]) -> tuple[list[float], str] | None:
    """SCF dipole (e bohr) about the coordinate origin from ORCA's property file.

    ORCA puts the origin at the centre of mass (printed with 6 decimals only). The full-
    precision nuclear contribution about that origin gives it back exactly:
    mu_nuc(O) = sum_A Z_A R_A - Z_tot O, and mu(0) = mu(O) + Q O (Q = total charge)."""
    import numpy as np
    from qcref import periodic_table

    if not path.exists():
        return None
    text = path.read_text()
    blk = re.search(r"\$SCF_Dipole_Moment\s*\n\s*&GeometryIndex 1\b(.*?)\$End", text, re.S)
    if not blk:
        return None

    def array(tag: str):
        m = re.search(rf"&{tag} \[.*?\n(.*?)(?=\n\s*&|\Z)", blk.group(1), re.S)
        vals = re.findall(r"^\s*\d+\s+(-?\d\.\d+e[+-]\d+)\s*$", m.group(1), re.M) if m else []
        return np.array([float(v) for v in vals[:3]]) if len(vals) >= 3 else None

    tot, nuc = array("DIPOLETOTAL"), array("DIPOLENUCCONTRIB")
    if tot is None or nuc is None:
        return None
    Z = np.array([periodic_table.symbol_to_number(s) if hasattr(periodic_table, "symbol_to_number")
                  else getattr(periodic_table, s).number for s in job["symbols"]], dtype=float)
    Z -= np.array(ecp_core_electrons(job["symbols"], job["basis"]), dtype=float)  # ECP atoms: Z - ncore
    R = np.asarray(job["geometry_bohr"])
    origin = ((Z[:, None] * R).sum(axis=0) - nuc) / Z.sum()
    mu0 = tot + job["charge"] * origin
    return mu0.tolist(), (f"orca.property.txt $SCF_Dipole_Moment (origin {origin.round(6).tolist()} bohr "
                          "recovered from the nuclear contribution, shifted to (0, 0, 0))")


def run_orca(job: dict[str, Any], rundir: Path) -> dict[str, Any]:
    pyscf_only(job, "ORCA")
    import numpy as np
    from qccompute import compute
    from qcdata import CalcType, Model, ProgramInput, Structure
    from qcref import constants

    _, _, ofunc = FUNCTIONALS.get(job["method"], (None, None, job["method"]))
    is_dft = job["method"] != "hf"
    open_shell = job["multiplicity"] != 1
    if job.get("cart"):
        raise RuntimeError("ORCA uses spherical (pure) basis functions only; no Cartesian basis sets")
    if job["grid"] and (job["grid"].startswith("pyscf:") or job["grid"] == "psi4"):
        raise RuntimeError(f"grid {job['grid']!r} cannot be built in ORCA (use fine)")
    words = [ofunc]
    words.append(("UKS" if is_dft else "UHF") if open_shell else ("RKS" if is_dft else "RHF"))
    if job.get("disp"):
        words.append(DISPERSION[job["disp"]][2])
    aux = job["aux"]
    basis_block: dict[str, Any] = {}
    if job["approx"] == "exact":
        words += ["NoRI", "NoCOSX"]
    elif job["approx"] == "df":
        words += ["RIJK"]
        basis_block["auxjk"] = AUX_BASES[aux]["orca"]
        if job["method"] in ("svwn-rpa", "pbe", "blyp", "tpss", "r2scan"):
            # without exact exchange ORCA 6.0.0 fits J with AuxJ even under RIJK and stops
            # ("no AuxJ basis set in ORCA_GTOInt"): give it the same JK functions
            basis_block["auxj"] = AUX_BASES[aux]["orca"]
    elif job["approx"] == "rij":
        words += ["RIJONX" if job["method"] in ("hf", "b3lyp", "b3lyp5", "pbe0") else "RI"]
        basis_block["auxj"] = AUX_BASES[aux]["orca"]
    elif job["approx"] == "rijcosx":
        words += ["RIJCOSX"]
        basis_block["auxj"] = AUX_BASES[aux]["orca"]
    words += ["VeryTightSCF", "NoAutoStart", "Bohrs"]
    kw: dict[str, Any] = {
        "maxcore": max(500, int(job["memory_gb"] * 1000 / max(1, job["threads"]))),
        "scf": {
            "thresh": SCF["int_thresh"],
            "tcut": 1e-15,
            "tole": SCF["e_conv"],
            "tolrmsp": SCF["d_conv"],
            "tolmaxp": 1e-9,
            "tolerr": SCF["d_conv"],
            "tolg": 1e-9,
            "convcheckmode": 0,  # every criterion must be met
            "convforced": True,
            "directresetfreq": 1,  # rebuild the full Fock matrix every iteration
            "maxiter": SCF["maxiter"],
        },
    }
    if job["threads"] > 1:
        kw["pal"] = job["threads"]
    if basis_block:
        kw["basis"] = basis_block
    if is_dft:
        grid = job["grid"]
        if grid == "fine":
            kw["method"] = dict(ORCA_FINE_GRID)
        elif grid == "defgrid3":
            words.append("DefGrid3")
        elif grid != "default":
            kw["method"] = dict(ORCA_FINE_GRID)  # ORCA cannot take (R, A) directly
    if job["calc"] == "gradient" and job.get("orca_numgrad"):
        # central differences of energies (ORCA 6.0.0's analytic D3 gradient is not the
        # derivative of its D3 energy; the numerical one is)
        kw["numgrad"] = {"centraldiff": True, "dx": job.get("orca_fd_step", 0.005)}
    if job["calc"] == "hessian":
        kw["method"] = {**kw.get("method", {}), "z_tol": CPSCF_TOL}  # CP-SCF convergence
        mode = job.get("orca_hessian") or ("analytic" if job.get("orca_analytic_dft_hessian") or not is_dft else "numfreq")
        if job["approx"] == "df" and mode == "analytic":
            # ORCA 6.0.0: "Analytical Hessian not available with RIJK approximation"
            mode = "numfreq"
        if mode == "numfreq":
            # central differences of analytic gradients (with grid-weight derivatives)
            kw["numfreq"] = True
            kw["freq"] = {"centraldiff": True, "increment": job.get("orca_fd_step", 0.005)}
        else:
            # ORCA's analytic Freq writes no gradient unless asked
            kw["engrad"] = True
            if is_dft and job["grid"] == "fine":
                # ORCA 6.0.0 solves the CP-SCF equations on a separate grid (grid_cpscf) whose
                # size follows the DefGrid level and ignores IntAcc/AngularGrid: with the fine
                # grid settings alone (DefGrid2's CP-SCF grid) the analytic Hessian is 1.8e-4
                # Eh/bohr^2 from ORCA's own NumFreq (water B3LYP/def2-SVP), with DefGrid3 2.1e-6.
                # The %method block below still sets the main grid.
                words.append("DefGrid3")
        job = {**job, "orca_hessian_mode": mode}
    if job["method"] in ("pbe", "pbe0") and not job["orca_native_pbe"]:
        # ORCA 6.0.0's PBE uses the rounded constants of the 1996 paper (mu 0.21952,
        # beta 0.066725): 1.4e-5 Eh off libxc PBE for water. Use libxc's values.
        kw["method"] = {**kw.get("method", {}), **ORCA_LIBXC_PBE}
    # Feed ORCA the exact bohr geometry: the codec writes geometry.xyz in angstrom
    # (geometry * qcref BOHR_TO_ANGSTROM), so pre-divide by that factor and say "Bohrs".
    b2a = float(constants.BOHR_TO_ANGSTROM)
    geom = np.asarray(job["geometry_bohr"]) / b2a
    structure = Structure(symbols=job["symbols"], geometry=geom.tolist(),
                          charge=job["charge"], multiplicity=job["multiplicity"])
    calctype = {"energy": CalcType.energy, "gradient": CalcType.gradient,
                "hessian": CalcType.hessian}[job["calc"]]
    inp = ProgramInput(program="orca", calctype=calctype, structure=structure,
                       model=Model(method=" ".join(words), basis=job["basis"]), keywords=kw)
    out = compute(
        inp, scratch_dir=rundir, rm_scratch_dir=False, raise_exc=False,
        executables={"orca": str(ORCA_DIR / "orca")}, environment=orca_environment(),
    )
    out.save(rundir / "qcoutput.json")
    stdout = out.logs or ""
    (rundir / "orca.out").write_text(stdout)
    if "ORCA TERMINATED NORMALLY" not in stdout:
        # ORCA 6.0.0 can exit 0 after "ORCA finished by error termination"
        raise RuntimeError(f"ORCA did not terminate normally: {out.traceback}")
    recovered_hessian = None
    if not out.success:
        # Hessian runs: the codec's optional normal-mode parser compares ORCA's .hess
        # coordinates with the (pre-scaled, see above) input structure and refuses.
        # The calculation itself finished; read energy and Hessian with the codec's
        # own parsers instead.
        tb = out.traceback or ""
        if job["calc"] == "hessian" and "normal modes cannot be mapped" in tb:
            from qccodec.programs import orca as orca_codec

            recovered_hessian = orca_codec.parse_hessian((rundir / "orca.hess").read_text())
        else:
            raise RuntimeError(f"ORCA failed: {tb}")

    class _R:  # minimal stand-in for out.results when recovering
        energy = None
        gradient = None
        hessian = None
        provenance = None

    if recovered_hessian is None:
        r = out.results
    else:
        from qccodec.programs import orca as orca_codec

        r = _R()
        r.energy = orca_codec.parse_energy(stdout)
        r.hessian = np.asarray(recovered_hessian)
        r.provenance = type("P", (), {"program_version": orca_codec.parse_version(stdout)})()

    def grab(pattern: str, cast=float):
        m = re.search(pattern, stdout)
        return cast(m.group(1)) if m else None

    iters = [int(x) for x in re.findall(r"SCF CONVERGED AFTER\s+(\d+)\s+CYCLES", stdout)]
    trah = [int(x) for x in re.findall(r"TRAH.*?CONVERGED AFTER\s+(\d+)", stdout)]
    res: dict[str, Any] = {
        "energy": float(r.energy),
        "nuclear_repulsion": grab(r"Nuclear Repulsion\s+:\s+(-?\d+\.\d+)"),
        "scf_iterations": iters[-1] if iters else (trah[-1] if trah else -1),
        "nbasis": grab(r"Number of basis functions\s+\.\.\.\s+(\d+)", int),
        "energy_components_native": {
            k: v for k, v in {
                "One Electron Energy": grab(r"One Electron Energy:\s+(-?\d+\.\d+)"),
                "Two Electron Energy": grab(r"Two Electron Energy:\s+(-?\d+\.\d+)"),
                "E(XC)": grab(r"E\(XC\)\s+:\s+(-?\d+\.\d+)"),
                "Nuclear Repulsion": grab(r"Nuclear Repulsion\s+:\s+(-?\d+\.\d+)"),
            }.items() if v is not None
        },
        "program_version": r.provenance.program_version,
        "native_input": (rundir / "orca.inp").read_text(),
        "codec_wall_time_s": out.execution.wall_time,
        "orca_dir": str(ORCA_DIR),
    }
    lda = re.search(r"LDA part of GGA corr\.\s+LDAOpt\s+\.+\s+(\S+)", stdout)
    if lda:
        res["orca_ldaopt"] = lda.group(1)
    npts = re.search(r"Total number of grid points\s+\.\.\.\s+(\d+)", stdout)
    if job["calc"] in ("gradient", "hessian"):
        engrad = rundir / "orca.engrad"
        if engrad.exists():  # 12 decimals, vs 9 in stdout
            lines = [l for l in engrad.read_text().splitlines() if not l.startswith("#")]
            nat = int(lines[0])
            vals = [float(x) for x in lines[2 : 2 + 3 * nat]]
            res["gradient"] = np.asarray(vals).reshape(nat, 3).tolist()
            res["gradient_source"] = "orca.engrad"
        elif r.gradient is not None:
            res["gradient"] = r.gradient.tolist()
            res["gradient_source"] = "stdout"
    if job["calc"] == "hessian":
        res["hessian"] = r.hessian.tolist()
        res["hessian_method"] = (f"NumFreq: central differences of analytic gradients (increment {kw['freq']['increment']} bohr)"
                                 if kw.get("numfreq") else
                                 "analytic" + (" (first-order grid response, Bykov et al. 2015)" if is_dft else ""))
        res["hessian_source"] = "orca.hess ($hessian)" + (" via qccodec parse_hessian (normal-mode check skipped)" if recovered_hessian is not None else "")
    if kw.get("numgrad"):
        res["gradient_method"] = f"NumGrad: central differences of energies (dx {kw['numgrad']['dx']} bohr)"
    dip = orca_dipole(rundir / "orca.property.txt", job)
    if dip is not None:
        res["dipole"], res["dipole_source"] = dip
    m = re.search(r"Dispersion correction\s+(-?\d+\.\d+)", stdout)
    if m:
        res["dispersion_energy"] = float(m.group(1))
    settings: dict[str, Any] = {"simple_input": words, "blocks": kw,
                                "geometry": "bohr via ! Bohrs (exact, no A->bohr round trip)"}
    if is_dft and npts:
        settings["grid_points"] = int(npts.group(1))
    for label, rx in {
        "IntAcc": r"IntAcc\s+\.\.\.\s+(\S+)",
        "AngularGrid": r"AngularGrid\s+\.\.\.\s+(.+)",
        "GridPruning": r"GridPruning\s+\.\.\.\s+(.+)",
        "RadialGrid": r"RadialGrid\s+\.\.\.\s+(.+)",
        "WeightScheme": r"WeightScheme\.\.\.\s+(.+)",
        "BFCut": r"BFCut\s+\.\.\.\s+(\S+)",
        "WCut": r"WCut\s+\.\.\.\s+(\S+)",
        "Thresh": r"Integral Threshold\s+Thresh\s+\.+\s+(\S+)",
        "TCut": r"Primitive CutOff\s+TCut\s+\.+\s+(\S+)",
        "TolE": r"Energy Change\s+TolE\s+\.+\s+(\S+)",
        "TolRMSP": r"RMS-Density Change\s+TolRMSP\s+\.+\s+(\S+)",
        "TolMaxP": r"Max\. Density Change\s+TolMaxP\s+\.+\s+(\S+)",
        "TolErr": r"DIIS Error\s+TolErr\s+\.+\s+(\S+)",
        "TolG": r"Orbital Gradient\s+TolG\s+\.+\s+(\S+)",
        "ConvCheckMode": r"ConvCheckMode\s+\.+\s+(\S+)",
        "RI": r"(RI-approximation to the Coulomb term is turned \w+)",
        "COSX": r"(COSX.*?turned \w+|Chain-of-spheres.*?\w+)",
        "AuxJ": r"Auxiliary basis set for J\s*\.+\s*(.+)",
    }.items():
        m = re.search(rx, stdout)
        if m:
            settings.setdefault("as_printed", {})[label] = m.group(1).strip()
    res["settings"] = settings
    return res


# ---------------------------------------------------------------- driver

RUNNERS = {"pyscf": run_pyscf, "psi4": run_psi4, "orca": run_orca}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("xyz", type=Path, help="geometry file, angstrom")
    p.add_argument("--charge", type=int, default=0)
    p.add_argument("--mult", type=int, default=1, help="spin multiplicity 2S+1")
    p.add_argument("--method", default="hf", help=f"one of {sorted(FUNCTIONALS)} (others passed through unverified)")
    p.add_argument("--basis", default="def2-svp", help="orbital basis (program built-in library)")
    p.add_argument("--calc", choices=("energy", "gradient", "hessian"), default="energy")
    p.add_argument("--programs", default=",".join(PROGRAMS), help="comma list of pyscf,psi4,orca")
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--memory-gb", type=int, default=16, help="total memory per program")
    p.add_argument("--approx", choices=sorted(APPROX), default="exact")
    p.add_argument("--aux", default=None, help=f"aux basis alias {sorted(AUX_BASES)}; default from basis")
    p.add_argument("--grid", default="fine",
                   help="DFT grid: fine (finest per program), default (program defaults), "
                        "defgrid3 (ORCA DefGrid3; others default), RAD,ANG (PySCF/Psi4; ORCA uses fine), "
                        "pyscf:N (PySCF level N, = linus pyscf:N; PySCF only), psi4 (Psi4's default "
                        "grid without distant-point removal, = linus psi4; Psi4 only)")
    p.add_argument("--cart", action="store_true", help="Cartesian basis functions (PySCF, Psi4; ORCA has none)")
    p.add_argument("--disp", choices=sorted(DISPERSION), default=None, help="empirical dispersion")
    p.add_argument("--no-grid-response", action="store_true",
                   help="PySCF DFT gradients without grid-weight derivatives (Psi4 never has them)")
    p.add_argument("--orca-analytic-dft-hessian", action="store_true",
                   help="ORCA DFT Hessians analytic (default NumFreq: the analytic one is ~1e-4 off)")
    p.add_argument("--orca-hessian", choices=("analytic", "numfreq"), default=None,
                   help="ORCA Hessian method (default: analytic for HF, NumFreq for DFT; NumFreq with RIJK)")
    p.add_argument("--orca-fd-step", type=float, default=0.005,
                   help="ORCA NumFreq increment / NumGrad dx in bohr (ORCA's default 0.005)")
    p.add_argument("--orca-numgrad", action="store_true", help="ORCA gradient by NumGrad (central differences)")
    p.add_argument("--psi4-hessian", choices=("auto", "analytic", "fd"), default="auto",
                   help="Psi4 Hessian: auto (analytic for HF and LDA, else finite differences), analytic, fd")
    p.add_argument("--pyscf-max-memory-mb", type=int, default=None, help="PySCF max_memory (default memory-gb)")
    p.add_argument("--pyscf-linus-basis", action="store_true",
                   help="PySCF reads the orbital basis from linus's data/basis (BSE JSON) instead of its library")
    p.add_argument("--rundir", type=Path, default=None,
                   help="write <program>.json and <program>_run/ here instead of outdir/<molecule>/<name>/")
    p.add_argument("--orca-native-pbe", action="store_true",
                   help="keep ORCA's own PBE constants (mu 0.21952, beta 0.066725) instead of libxc's")
    p.add_argument("--psi4-scf-type", choices=("pk", "direct"), default="pk",
                   help="exact-integral algorithm in Psi4 (pk = in-core conventional)")
    p.add_argument("--pcm", choices=sorted(PCM_MODELS), default=None,
                   help="implicit solvent (PySCF only), set up as linus's `pyscf` PCM preset")
    p.add_argument("--solvent", default=None, help="solvent name (--pcm smd: PySCF's SMD descriptors, eps from its table)")
    p.add_argument("--eps", type=float, default=None, help="dielectric constant (--pcm other than smd)")
    p.add_argument("--pcm-lebedev", type=int, default=None, help="Lebedev points per sphere (default 302; smd 590)")
    p.add_argument("--nlc-level", type=int, default=3,
                   help="PySCF nlcgrids level of VV10 functionals (default 3, PySCF's own default)")
    p.add_argument("--wk", action="store_true",
                   help="PySCF with --approx df: range-separated exchange fitted by Psi4's wK scheme (linus's), "
                        "energies only")
    p.add_argument("--outdir", type=Path, default=REPO / "reference" / "out")
    p.add_argument("--tag", default="", help="extra label for the output directory")
    a = p.parse_args(argv)

    # thread env before numpy / PySCF are imported
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[var] = str(a.threads)
    if a.disp or a.method.lower() in IMPLIED_DISPERSION:
        if not DISP_OVERLAY.is_dir():
            die(f"--disp needs the dispersion overlay {DISP_OVERLAY} (reference/validation/README.md)")
        # PySCF finds pyscf.dispersion (and its libs-dftd3 / libdftd4) through PYSCF_EXT_PATH
        sys.path.insert(0, str(DISP_OVERLAY))
        os.environ["PYSCF_EXT_PATH"] = str(DISP_OVERLAY)

    import numpy as np
    from qcref import constants, periodic_table

    method = a.method.lower()
    basis = a.basis.lower()
    programs = [x.strip().lower() for x in a.programs.split(",") if x.strip()]
    for prog in programs:
        if prog not in RUNNERS:
            die(f"unknown program {prog!r}")
    aux = (a.aux or default_aux(basis, a.approx))
    if aux is not None:
        aux = aux.lower()
        if aux not in AUX_BASES:
            die(f"unknown aux basis {aux!r}; known: {sorted(AUX_BASES)}")
    symbols, coords, comment = read_xyz(a.xyz)
    a2b = float(constants.ANGSTROM_TO_BOHR)
    geom_bohr = (np.asarray(coords) * a2b).tolist()
    Z = [periodic_table.symbol_to_number(s) if hasattr(periodic_table, "symbol_to_number")
         else getattr(periodic_table, s).number for s in symbols]
    is_dft = method != "hf"
    grid = a.grid.lower() if is_dft else None

    name = f"{slug(method)}_{slug(basis)}{'-cart' if a.cart else ''}_{a.calc}_{a.approx}"
    if a.approx != "exact":
        name += f"-{slug(aux)}"
    if a.disp:
        name += f"_{a.disp}"
    if is_dft:
        name += f"_grid-{slug(grid)}"
    if a.pcm:
        if a.pcm == "smd" and not a.solvent:
            die("--pcm smd needs --solvent")
        if a.pcm != "smd" and a.eps is None:
            die(f"--pcm {a.pcm} needs --eps")
        name += f"_{a.pcm}-{slug(a.solvent or str(a.eps))}"
    if a.wk:
        name += "_wk"
    if a.tag:
        name += f"_{slug(a.tag)}"
    outdir = a.rundir if a.rundir is not None else a.outdir / a.xyz.stem / name
    outdir.mkdir(parents=True, exist_ok=True)

    job = {
        "symbols": symbols, "geometry_bohr": geom_bohr, "charge": a.charge,
        "multiplicity": a.mult, "method": method, "basis": basis, "calc": a.calc,
        "approx": a.approx, "aux": aux, "grid": grid, "threads": a.threads,
        "memory_gb": a.memory_gb, "grid_response": not a.no_grid_response,
        "psi4_scf_type": a.psi4_scf_type, "orca_native_pbe": a.orca_native_pbe,
        "orca_analytic_dft_hessian": a.orca_analytic_dft_hessian,
        "cart": a.cart, "disp": a.disp, "orca_hessian": a.orca_hessian,
        "orca_fd_step": a.orca_fd_step, "orca_numgrad": a.orca_numgrad,
        "psi4_hessian": a.psi4_hessian, "pyscf_max_memory_mb": a.pyscf_max_memory_mb,
        "pyscf_linus_basis": a.pyscf_linus_basis,
        "pcm": a.pcm, "solvent": a.solvent, "eps": a.eps,
        "pcm_lebedev": a.pcm_lebedev or (590 if a.pcm == "smd" else 302), "nlc_level": a.nlc_level, "wk": a.wk,
    }
    vnn_ref = nuclear_repulsion([z - c for z, c in zip(Z, ecp_core_electrons(symbols, basis))], geom_bohr)
    status = 0
    for prog in programs:
        rundir = outdir / f"{prog}_run"
        if rundir.exists():
            shutil.rmtree(rundir)
        rundir.mkdir(parents=True)
        record: dict[str, Any] = {
            "program": prog,
            "molecule": {
                "name": a.xyz.stem, "xyz_file": str(a.xyz.resolve()), "comment": comment,
                "symbols": symbols, "atomic_numbers": Z,
                "geometry_angstrom": coords, "geometry_bohr": geom_bohr,
                "angstrom_to_bohr": a2b, "conversion": "CODATA 2022 via qcref",
                "charge": a.charge, "multiplicity": a.mult,
            },
            "method": method, "basis": basis, "cartesian": a.cart, "dispersion": a.disp, "calc_type": a.calc,
            **({"solvation": {"model": a.pcm, "solvent": a.solvent, "eps": a.eps,
                              "lebedev": job["pcm_lebedev"]}} if a.pcm else {}),
            **({"long_range_fit": "wK"} if a.wk else {}),
            "approximation": {"spec": a.approx, "meaning": APPROX[a.approx], "aux_alias": aux,
                              "aux_basis": AUX_BASES[aux][prog] if aux else None},
            "dft_grid": grid,
            "functional_mapping": {"canonical": method,
                                   "native": dict(zip(PROGRAMS, FUNCTIONALS.get(method, (method,) * 3))).get(prog),
                                   "verified": method in VERIFIED_FUNCTIONALS},
            "threads": a.threads, "memory_gb": a.memory_gb,
            "units": {"energy": "Eh", "gradient": "Eh/bohr", "hessian": "Eh/bohr^2", "geometry": "bohr"},
            "nuclear_repulsion_runner": vnn_ref,
            "host": {"hostname": socket.gethostname(), "platform": platform.platform(),
                     "python": sys.version.split()[0]},
            "date": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
            "runner": {"file": str(Path(__file__).resolve()), "version": RUNNER_VERSION,
                       "argv": sys.argv},
            "raw_dir": str(rundir),
        }
        print(f"[{prog}] {a.xyz.stem} {method}/{basis} {a.calc} approx={a.approx} grid={grid} threads={a.threads} ...", flush=True)
        t0 = time.perf_counter()
        try:
            res = RUNNERS[prog](job, rundir)
            record.update(res)
            record["success"] = True
        except Exception as exc:  # keep going with the other programs
            record["success"] = False
            record["error"] = f"{type(exc).__name__}: {exc}"
            record["traceback"] = traceback.format_exc()
            status = 1
        record["wall_time_s"] = time.perf_counter() - t0
        if record.get("success") and record.get("nuclear_repulsion") is not None:
            record["nuclear_repulsion_minus_runner"] = record["nuclear_repulsion"] - vnn_ref
        if "gradient" in record:
            record["gradient_sum"] = np.asarray(record["gradient"]).sum(axis=0).tolist()
        (outdir / f"{prog}.json").write_text(json.dumps(record, indent=1) + "\n")
        if record["success"]:
            print(f"[{prog}] E = {record['energy']:.12f}  iters={record.get('scf_iterations')}  "
                  f"Vnn-Vnn(runner) = {record.get('nuclear_repulsion_minus_runner', float('nan')):.2e}  "
                  f"wall = {record['wall_time_s']:.2f} s", flush=True)
        else:
            print(f"[{prog}] FAILED: {record['error']}", flush=True)
    print(f"results in {outdir}")
    return status


if __name__ == "__main__":
    sys.exit(main())
