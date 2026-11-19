"""linus: Hartree-Fock and Kohn-Sham DFT in-process: energies, analytic gradients, analytic
Hessians with vibrational analysis and thermochemistry (closed and open shells; global, range-separated and
VV10 functionals; D3(BJ)/D4 dispersion; ECPs; implicit solvent), on qcdata objects.

All quantities are in atomic units (bohr, Hartree, e·bohr). A calculation runs on the
threads you give it (keyword ``threads``, default 1), on its own thread pool, with the GIL released.

>>> import qcdata, linus
>>> water = qcdata.Structure.open("water.xyz")                      # Å in the file, bohr in memory
>>> inp = qcdata.ProgramInput(program="linus", calctype="gradient", structure=water,
...                           model={"method": "b3lyp-d3bj", "basis": "def2-svp"}, keywords={"threads": 4})
>>> out = linus.compute(inp)                     # one calculation: a qcdata.ProgramOutput
>>> out.results.energy, out.results.gradient, out.results.energy_components, out.record["scf"]["history"]
>>> with linus.Session(inp) as s:               # one molecule and model at many geometries
...     f = s.compute()                          # the input's geometry and calctype: a qcdata.Frame
...     f = s.compute(x2)                        # new positions (bohr): starts from the last orbitals
...     h = s.compute(x2, calctype="hessian")    # with frequencies and thermochemistry
>>> h.results.vibrational_analysis.frequencies, h.results.thermochemistry[0].total.gibbs

A session keeps, per lineage (any JSON value naming a chain of geometries: an optimization, an
IRC branch, a path image), the last converged orbitals and the stability probes' state, so each
step starts from its predecessor; `qcdata.Request`'s ``seed`` and ``commit`` fan out finite
differences and path images from one state (docs/python.md, "Sessions").

Errors: a failed calculation raises a `LinusError` (a `qcdata.ProgramError`) carrying its partial
results (`results`; a session request's failed `frame`, `linus.compute`'s failed `output`):
`InputError` (a `ValueError`), `UnsupportedError` (a `NotImplementedError`),
`ResourceError` (a `MemoryError`), `BackendError` (the GPU backend failed), `ConvergenceError`
and `NumericalError` (an `ArithmeticError`); `kind` is the CLI's ``error_kind``, `code` the
error's code (``unknown_key``, ``not_applicable``, ``device_lost``, ...).

Builds: `linus.GPU_BUILD` is True in the GPU distribution (``afh-linus-cuda13``, NVIDIA GPUs with
CUDA 13; docs/gpu.md) and False in the CPU one (``afh-linus``); both have the same version.

For runners: `linus.qc.program` (qccompute's Program protocol; the ``qccompute.programs`` entry
point; under a runner, ``compute`` and ``Session`` take its ``ctx``). Any program's Hessian: `linus.vib`
(harmonic analysis and thermochemistry as qcdata's models). Orbitals to and from PySCF:
`linus.convert` (Molden files of any wavefunction: `qcinf.molden.write`).
"""

from __future__ import annotations

from typing import Any

from . import _linus
from ._library import (
    available_basis_sets,
    available_functionals,
    available_solvents,
    basis_file,
    basis_search_path,
    settings_schema,
    solvent_table,
)
from ._linus import GPU_BUILD, NUMERICS_EPOCH, __version__
from .errors import (
    BackendError,
    ConvergenceError,
    InputError,
    LinusError,
    NumericalError,
    ResourceError,
    UnsupportedError,
)
from .session import Session, compute
from . import vib

__all__ = [
    "BackendError",
    "ConvergenceError",
    "GPU_BUILD",
    "InputError",
    "LinusError",
    "NUMERICS_EPOCH",
    "NumericalError",
    "ResourceError",
    "Session",
    "UnsupportedError",
    "__version__",
    "available_basis_sets",
    "available_functionals",
    "available_solvents",
    "basis_file",
    "basis_search_path",
    "capabilities",
    "compute",
    "default_options",
    "normalize_options",
    "settings_schema",
    "solvent_table",
    "vib",
]


def capabilities() -> list[str]:
    """Calculation types this build can do (energy, gradient, hessian)."""
    return list(_linus.capabilities())


def default_options() -> dict[str, Any]:
    """Every setting at its default (``"auto"`` where the calculation decides), as a dict."""
    return dict(_linus.default_options())


def normalize_options(options: dict[str, Any]) -> dict[str, Any]:
    """The settings given, as the library reads them: canonical keys and values, shorthands
    expanded (``{"jk": "rijcosx"}`` -> ``{"jk": {"kind": "rij_snk"}}``, ``"b3lyp-d3bj"`` -> method
    ``b3lyp`` with dispersion ``d3bj``), checked; raises `InputError` for bad ones."""
    return _linus.normalize_options(options)


def __getattr__(name: str) -> Any:
    # `linus.convert` and `linus.qc` on first use
    if name in ("convert", "qc"):
        import importlib

        return importlib.import_module(f".{name}", __name__)
    raise AttributeError(f"module 'linus' has no attribute {name!r}")
