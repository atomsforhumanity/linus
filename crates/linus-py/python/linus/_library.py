"""linus's data and settings, without a calculation: the basis sets on the search path and their
files, the functionals, the solvent tables, the settings' JSON Schema, element symbols.

Importing it points the library at the basis sets shipped with the package and, where qcref is
installed, at qcref's full CRC permittivity table.
"""

from __future__ import annotations

import importlib.util
import json
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from . import _linus, errors

__all__ = [
    "available_basis_sets",
    "available_functionals",
    "available_solvents",
    "basis_file",
    "basis_search_path",
    "settings_schema",
    "solvent_table",
]

# element symbols by atomic number (index 0: "X"), the library's
SYMBOLS: tuple[str, ...] = tuple(_linus.element_symbols())

# basis sets shipped with the package (in the source tree, symlinks to data/basis): the last
# directory of the library's basis set search path
PACKAGE_BASIS_DIR = Path(__file__).resolve().parent / "basis"
if PACKAGE_BASIS_DIR.is_dir():
    _linus.set_package_basis_dir(str(PACKAGE_BASIS_DIR))


def _qcref_solvent_table() -> Path | None:
    """qcref's full CRC permittivity table, found without importing qcref."""
    try:
        spec = importlib.util.find_spec("qcref")
    except (ImportError, ValueError):
        return None
    for d in (spec.submodule_search_locations or []) if spec is not None else []:
        p = Path(d) / "data" / "solvents" / "crc.csv"
        if p.is_file():
            return p
    return None


# linus ships the permittivities of the SMD solvents and about 30 other common liquids; the other
# CRC liquids resolve through qcref's full table where qcref is installed ($LINUS_SOLVENT_TABLE
# overrides it)
_QCREF_SOLVENT_TABLE = _qcref_solvent_table()
if _QCREF_SOLVENT_TABLE is not None:
    _linus.set_solvent_table(str(_QCREF_SOLVENT_TABLE))


def solvent_table() -> Path | None:
    """The full CRC permittivity table that solvent names outside the shipped table resolve
    through: ``$LINUS_SOLVENT_TABLE``, else qcref's where it is installed; None without one."""
    p = _linus.solvent_table()
    return None if p is None else Path(p)


def basis_search_path(basis_dir: str | os.PathLike[str] | Sequence[str | os.PathLike[str]] | None = None) -> list[Path]:
    """Where basis sets are read from, in order: `basis_dir` (a directory or a list; the
    ``basis_dir`` option), ``$LINUS_BASIS_DIR`` (directories separated as in ``PATH``), the
    source tree of a development build, and the sets shipped with the package. A basis set is
    read from the first directory that has its file."""
    if basis_dir is None:
        dirs: list[str] = []
    elif isinstance(basis_dir, (str, os.PathLike)):
        dirs = [os.fspath(basis_dir)]
    else:
        dirs = [os.fspath(d) for d in basis_dir]
    return [Path(d) for d in _linus.basis_search_path(dirs)]


def basis_file(name: str, basis_dir: str | os.PathLike[str] | Sequence[str | os.PathLike[str]] | None = None) -> Path | None:
    """The Basis Set Exchange JSON file of basis set `name` on the search path
    (`basis_search_path`), or None."""
    f = _linus.find_basis_file(name, [str(d) for d in basis_search_path(basis_dir)])
    return None if f is None else Path(f)


def available_basis_sets(basis_dir: str | os.PathLike[str] | Sequence[str | os.PathLike[str]] | None = None) -> list[str]:
    """Names of the basis sets (orbital and auxiliary) on the basis set search path
    (`basis_search_path`), sorted."""
    return list(_linus.available_basis_sets([str(d) for d in basis_search_path(basis_dir)]))


def available_functionals() -> list[str]:
    """Functional names linus defines (libxc names, such as ``"gga_x_pbe + gga_c_pbe"``, and
    mixtures, such as ``"0.5*b3lyp + 0.5*hf"``, work too). A dispersion suffix (``b3lyp-d3bj``)
    or the ``dispersion`` option adds D3(BJ) or D4."""
    return list(_linus.available_functionals())


def available_solvents() -> list[tuple[str, float, bool]]:
    """The shipped solvent database: (name, dielectric constant at 298.15 K, has SMD descriptors).
    The other liquids of the CRC table resolve by name through `solvent_table()`."""
    return [tuple(x) for x in _linus.available_solvents()]  # type: ignore[misc]


def settings_schema(calctype: str | None = None) -> dict[str, Any]:
    """The JSON Schema (draft 2020-12) of linus's settings (qcdata's ``keywords``, a JSON
    request's settings), with ``x-tier``, ``x-concern``, ``x-unit`` and ``x-applies-to`` on every
    key; with `calctype` (``"energy"``, ``"gradient"``, ``"hessian"``) only the settings that can
    apply to it. `linus.settings.Settings` types the same tree for editors."""
    return json.loads(_linus.settings_schema(calctype))


def atomic_numbers(atoms: Sequence[int | str]) -> list[int]:
    """Atomic numbers from numbers or element symbols."""
    out = []
    for a in atoms:
        if isinstance(a, str):
            sym = a.strip().capitalize()
            if sym not in SYMBOLS[1:]:
                raise errors.InputError(f"unknown element {a!r}")
            out.append(SYMBOLS.index(sym))
        else:
            z = int(a)
            if z != a or not 0 < z < len(SYMBOLS):
                raise errors.InputError(f"bad atomic number {a!r} (linus supports 1-86, H-Rn)")
            out.append(z)
    return out
