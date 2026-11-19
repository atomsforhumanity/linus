"""Shared helpers of the comparison scripts in tests/python (PySCF / Psi4 against linus).

- ROOT, BOHR (linus's constant, CODATA 2022), LINUS_BIN and EXAMPLES (this tree's binaries;
  LINUS_EXAMPLES=<dir> takes the example binaries from another build, e.g. target/test-fast/examples)
- read_xyz / read_xyz_bohr: symbols and positions of an XYZ file
- pyscf_basis / pyscf_ecp / pyscf_mol: PySCF given linus's basis data (data/basis, BSE JSON)
  and its def2 ECPs, the molecule in bohr with linus's constant
- PYSCF_XC / pyscf_xc: linus method name -> PySCF functional (and whether to switch VV10 off)
- wk_get_jk: PySCF density fitting with linus's (Psi4's) wK long-range exchange
- linus_json: run `linus --json` on a request
- mol_key / cached: the opt-in PySCF reference cache (pyscf_ref_cache.py, LINUS_PYSCF_CACHE)
- psi4_cwd: run Psi4 in a temporary working directory (timer.dat names the host)
- take_tolerance / verdict: a script's --tol-* flags and its pass/fail suffix and exit status (the merge gate's
  checks fail instead of printing digits)
"""
from __future__ import annotations

import atexit
import json
import os
import pathlib
import shutil
import subprocess
import tempfile
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
BOHR = 0.529177210544  # Å per bohr, linus's value (CODATA 2022); molecules go to PySCF in bohr
LINUS_BIN = ROOT / "target" / "release" / "linus"
EXAMPLES = pathlib.Path(os.environ.get("LINUS_EXAMPLES") or ROOT / "target" / "release" / "examples")
BASIS_DIR = ROOT / "data" / "basis"

# linus method -> (PySCF xc, switch VV10 off): the ORCA-style -D3BJ functionals are the -V
# functionals' semilocal parts without VV10 (plus D3(BJ), which linus reports separately as
# `components.dispersion` / `gradient_parts.dispersion`)
PYSCF_XC = {"wb97x-d3bj": ("wb97x-v", True), "wb97m-d3bj": ("wb97m-v", True)}


def pyscf_xc(method: str) -> tuple[str, bool]:
    return PYSCF_XC.get(method.lower(), (method, False))


# ------------------------------------------------------------------ molecules and basis sets


def read_xyz(path) -> tuple[list[str], list[list[float]]]:
    """Symbols (capitalized) and positions (Å) of the atoms of an XYZ file."""
    lines = pathlib.Path(path).read_text().splitlines()
    n = int(lines[0].split()[0])
    syms, pos = [], []
    for ln in lines[2 : 2 + n]:
        p = ln.split()
        syms.append(p[0].capitalize())
        pos.append([float(x) for x in p[1:4]])
    return syms, pos


def read_xyz_bohr(path) -> tuple[list[str], list[list[float]]]:
    """read_xyz with the positions in bohr (linus's constant)."""
    syms, pos = read_xyz(path)
    return syms, [[x / BOHR for x in p] for p in pos]


def _basis_file(name: str) -> pathlib.Path:
    stem = name.lower().replace("*", "_st_").replace("+", "_pl_").replace("/", "_sl_").replace("(", "_lp_").replace(")", "_rp_").replace(",", "_cm_")
    return BASIS_DIR / f"{stem}.json"


def _elements(name: str) -> dict:
    return json.loads(_basis_file(name).read_text())["elements"]


def pyscf_basis(name: str, elements) -> dict:
    """Basis `name` from the repository's BSE JSON as a PySCF basis dict for `elements`
    (symbols): the same contractions linus reads (zero coefficients dropped)."""
    from pyscf import gto

    data = _elements(name)
    out = {}
    for sym in elements:
        shells = []
        for sh in data[str(gto.charge(sym))]["electron_shells"]:
            exps = [float(e) for e in sh["exponents"]]
            ls = sh["angular_momentum"]
            rows = sh["coefficients"]
            if len(ls) == 1:
                ls = ls * len(rows)
            for l, row in zip(ls, rows):
                prims = [[e, float(c)] for e, c in zip(exps, row) if float(c) != 0.0]
                shells.append([l] + prims)
        out[sym] = shells
    return out


def ecp_entry(element: dict):
    """The ECP of one element of a BSE JSON basis (its "elements"[Z] dict) in PySCF's format,
    [n_core, [[l, [r^0 terms, r^1 terms, ...]], ...]] with the local channel as l = -1; None
    without an ECP."""
    if "ecp_potentials" not in element:
        return None
    pots = element["ecp_potentials"]
    lmax = max(p["angular_momentum"][0] for p in pots)
    chans = []
    for p in pots:
        l = p["angular_momentum"][0]
        rad = [[] for _ in range(8)]
        for n, e, c in zip(p["r_exponents"], p["gaussian_exponents"], p["coefficients"][0]):
            rad[n].append([float(e), float(c)])
        chans.append([-1 if l == lmax else l, rad])
    return [element["ecp_electrons"], chans]


def pyscf_ecp(name: str, elements) -> dict:
    """The ECPs basis `name` puts on `elements` (def2: Rb-Rn), PySCF's format; {} without."""
    from pyscf import gto

    data = _elements(name)
    out = {}
    for sym in elements:
        e = ecp_entry(data[str(gto.charge(sym))])
        if e is not None:
            out[sym] = e
    return out


def pyscf_mol(xyz, basis: str, charge: int = 0, mult: int = 1, cart: bool = False, max_memory=None, **kw):
    """PySCF Mole of an XYZ file (Å) in bohr (linus's constant) with linus's basis data and
    the basis set's ECPs; max_memory in MB (None: PySCF's default). Extra keywords go to gto.M."""
    from pyscf import gto

    syms, pos = read_xyz_bohr(xyz)
    els = sorted(set(syms))
    if max_memory is not None:
        kw["max_memory"] = max_memory
    return gto.M(atom=list(zip(syms, pos)), unit="Bohr", basis=pyscf_basis(basis, els), ecp=pyscf_ecp(basis, els),
                 charge=charge, spin=mult - 1, cart=cart, verbose=0, **kw)


def elements(mol) -> list[str]:
    """The distinct element symbols of a PySCF Mole (for auxiliary basis dicts)."""
    return sorted(set(mol.elements))


# ------------------------------------------------------------------ PySCF wK exchange


def wk_get_jk(mf, auxbasis):
    """Patch a density-fitted PySCF mean-field object so omega requests use the wK scheme of
    linus and Psi4 (Coulomb metric; K_LR = sym[(ml|P) V^-1 (P|erf|ns)] D_ls), built from
    PySCF integrals (int3c2e with and without mol.with_range_coulomb, int2c2e). PySCF's own DF
    fits K_LR with the erf metric (P|erf|Q). Only get_jk is patched: PySCF's DF gradient and
    Hessian still use the erf-metric scheme."""
    import scipy.linalg
    from pyscf import df, lib

    dfobj = mf.with_df
    orig = dfobj.get_jk
    mol = mf.mol
    auxmol = df.addons.make_auxmol(mol, auxbasis)
    nao, naux = mol.nao_nr(), auxmol.nao_nr()
    a3 = df.incore.aux_e2(mol, auxmol, "int3c2e", aosym="s1").reshape(nao * nao, naux)
    low = scipy.linalg.cholesky(auxmol.intor("int2c2e"), lower=True)
    b = scipy.linalg.solve_triangular(low, a3.T, lower=True).reshape(naux, nao, nao)
    cache = {}

    def get_jk(dm, hermi=1, with_j=True, with_k=True, direct_scf_tol=getattr(lib.param, "DIRECT_SCF_TOL", 1e-13), omega=None):
        if not omega:
            return orig(dm, hermi, with_j, with_k, direct_scf_tol, omega)
        assert not with_j and with_k and omega > 0
        if omega not in cache:
            with mol.with_range_coulomb(omega), auxmol.with_range_coulomb(omega):
                w3 = df.incore.aux_e2(mol, auxmol, "int3c2e", aosym="s1").reshape(nao * nao, naux)
            cache[omega] = scipy.linalg.solve_triangular(low, w3.T, lower=True).reshape(naux, nao, nao)
        bw = cache[omega]
        dms = np.asarray(dm)
        out = []
        for d in dms.reshape(-1, nao, nao):
            k = np.einsum("qml,ls,qns->mn", b, d, bw, optimize=True)
            out.append(0.5 * (k + k.T))
        return None, np.asarray(out).reshape(dms.shape)

    dfobj.get_jk = get_jk
    return mf


# ------------------------------------------------------------------ linus


def linus_json(req: dict, binary=None, env=None) -> dict:
    """Run `linus --json` (default: this tree's target/release/linus) on a request; the parsed
    result, with the wall time in `_wall` (s). Raises RuntimeError with linus's stderr."""
    t0 = time.perf_counter()
    p = subprocess.run([str(binary or LINUS_BIN), "--json"], input=json.dumps(req), capture_output=True, text=True, env=env)
    if p.returncode:
        raise RuntimeError(f"linus exit {p.returncode}: {p.stderr.strip()}")
    out = json.loads(p.stdout)
    out["_wall"] = time.perf_counter() - t0
    return out


# ------------------------------------------------------------------ tolerances (merge-gate checks)


def take_tolerance(args: list, flag: str) -> float | None:
    """Remove `flag VALUE` from args (in place); the value, or None without the flag."""
    if flag not in args:
        return None
    i = args.index(flag)
    v = float(args[i + 1])
    del args[i:i + 2]
    return v


def verdict(checks: dict) -> tuple[str, int]:
    """checks: name -> (difference, tolerance or None). The suffix of the script's output line ("" without
    tolerances, "  ok (...)" or "  FAIL: ...") and its exit status (1 when a difference is above its tolerance or
    not finite)."""
    given = {k: (d, t) for k, (d, t) in checks.items() if t is not None}
    if not given:
        return "", 0
    bad = [f"{k} {abs(d):.2e} > {t:g}" for k, (d, t) in given.items() if not abs(d) <= t]
    if bad:
        return "  FAIL: " + ", ".join(bad), 1
    return "  ok (" + ", ".join(f"{k} <= {t:g}" for k, (_, t) in given.items()) + ")", 0


# ------------------------------------------------------------------ PySCF reference cache


def mol_key(mol) -> dict:
    """What of a PySCF Mole determines a result: atoms and bohr positions, basis and ECP data
    (PySCF's normalized internal form), charge, spin, Cartesian functions."""
    return {"atoms": [[mol.atom_symbol(i), *map(float, mol.atom_coord(i))] for i in range(mol.natm)],
            "basis": mol._basis, "ecp": mol._ecp, "charge": int(mol.charge), "spin": int(mol.spin), "cart": bool(mol.cart)}


def cached(script, key: dict, compute):
    """compute() -> dict of numbers / arrays / strings (None values are dropped), stored in the
    PySCF reference cache under `key` (plus the calling script's source, this module's source and
    PySCF's version; pyscf_ref_cache.py) when LINUS_PYSCF_CACHE is set. Returns (result, hit):
    scalars come back as Python numbers, arrays as numpy arrays."""
    import pyscf_ref_cache

    path = pyscf_ref_cache.cache_path(script, key)
    hit = pyscf_ref_cache.load(path)
    if hit is not None:
        return hit, True
    res = {k: v for k, v in compute().items() if v is not None}
    pyscf_ref_cache.save(path, **res)
    return res, False


# ------------------------------------------------------------------ Psi4


def psi4_cwd(prefix: str = "linus_psi4_") -> pathlib.Path:
    """Move this process into a new temporary directory, removed at exit. Call it before
    importing psi4, with every path argument already made absolute: Psi4 writes timer.dat (which
    names the host) and scratch files into the working directory, timer.dat from its atexit
    handler (registered at import, so it runs before the removal registered here)."""
    d = tempfile.mkdtemp(prefix=prefix)
    os.chdir(d)
    atexit.register(shutil.rmtree, d, True)
    return pathlib.Path(d)
