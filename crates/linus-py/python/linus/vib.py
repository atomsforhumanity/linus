"""Harmonic vibrational analysis and thermochemistry of any Cartesian Hessian, as qcdata's
`VibrationalAnalysis` and `Thermochemistry` (computed by the Rust `linus::vib`).

A ``hessian`` request carries its analysis (the settings group ``freq``; there is no separate
frequencies calculation): ``frame.results.vibrational_analysis`` and
``frame.results.thermochemistry``. These functions analyse a Hessian from anywhere:

>>> va = linus.vib.harmonic_analysis(numbers, positions, hessian)               # any program's Hessian
>>> va.frequencies, va.n_imaginary(), va.normal_modes_cartesian.shape          # cm^-1, (nmodes, natoms, 3)
>>> th = linus.vib.thermochemistry(va, energy, multiplicity=1, temperature=[298.15, 310.15])
>>> th[0].zero_point_energy, th[0].total.gibbs, th[0].gibbs_correction          # Eh

Units: Eh, bohr, u, cm^-1 (frequencies, rotational constants), mdyn/Å (force constants),
km/mol (IR intensities), K, Pa, Eh/K (entropies, heat capacities). Conventions: the Rust module
`linus::vib` (crates/linus-vib/src/lib.rs and thermo.rs).

The keyword arguments are the keys of the ``freq`` settings (``linus --list-settings freq``). Of
the analysis: ``mass_table`` (``"isotope"``: most abundant isotope, default; ``"average"``:
standard atomic weights as ORCA), ``masses`` (all atoms, u), ``isotopes`` (``[(atom, mass),
...]``), ``projection`` (``"translations_rotations"`` default, ``"translations"``, ``"none"``),
``symmetry_number`` (default: detected), ``symmetry_tol`` (bohr, 0.01), ``codata``
(``"codata2022"`` default, ``"codata2014"``). Of the thermochemistry: ``thermo`` (``"rrho"``
default, ``"qrrho"`` Grimme entropy, ``"qrrho-hg"`` plus Head-Gordon enthalpy,
``"quasiharmonic"`` Truhlar), ``temperature`` (K, 298.15; a list gives one thermochemistry per
temperature), ``pressure`` (Pa, 101325), ``concentration`` (mol/L, standard state instead),
``rotor_cutoff`` (cm^-1, qrrho models), ``raise_below`` (cm^-1, quasiharmonic), ``qrrho_alpha``
(4), ``qrrho_bav`` (kg m², 1e-44), ``imaginary_as_real`` (cm^-1), ``scale``,
``electronic_degeneracy``, ``codata``. A setting that cannot take effect (``rotor_cutoff`` with
``thermo="rrho"``) is an `InputError`.

linus's analysis has more than qcdata's model holds (the eigenvalues, the detected symmetry
number and its tolerance, the Hessian's asymmetry, the analysis's warnings, per-mode
thermochemistry): a frame keeps them in ``record["vibrational_analysis"]`` and
``record["thermochemistry"]``.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from qcdata import Thermochemistry, VibrationalAnalysis

from . import _linus

__all__ = ["atomic_mass", "harmonic_analysis", "thermochemistry"]

_VIB = frozenset(VibrationalAnalysis.model_fields)
_THERMO = frozenset(Thermochemistry.model_fields)
# linus's analysis keys that repeat the structure (not kept in a record)
_STRUCTURE_KEYS = frozenset({"natoms", "numbers"})


def _jsonable(x: Any) -> Any:
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, Mapping):
        return {k: _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    return x


def _freq_json(freq: Mapping[str, Any]) -> str:
    """The JSON object of ``freq`` settings (keys without the ``freq.`` prefix); the library
    checks names and values."""
    return json.dumps({k: _jsonable(v) for k, v in freq.items() if v is not None})


def analysis_model(d: Mapping[str, Any], dipole_derivatives: Any = None) -> VibrationalAnalysis:
    """linus's harmonic analysis (the library's dict; its field names are qcdata's where qcdata
    has the field) as qcdata's model, with the dipole derivatives (3N x 3) of an IR run."""
    n, nm = int(d["natoms"]), len(d["frequencies"])
    v = {k: d[k] for k in _VIB if d.get(k) is not None}
    v["normal_modes_cartesian"] = np.asarray(d["normal_modes_cartesian"], dtype=float).reshape(nm, n, 3)
    v["normal_modes_mass_weighted"] = np.asarray(d["normal_modes_mass_weighted"], dtype=float).reshape(nm, 3 * n)
    if dipole_derivatives is not None:
        v["dipole_derivatives"] = np.asarray(dipole_derivatives, dtype=float).reshape(3 * n, 3)
    return VibrationalAnalysis.model_validate(v)


def analysis_record(d: Mapping[str, Any]) -> dict[str, Any]:
    """What linus's analysis has beyond qcdata's model (``eigenvalues``, ``codata``,
    ``center_of_mass``, ``principal_axes``, ``rotational_constants_ghz``,
    ``detected_symmetry_number``, ``symmetry_tol``, ``max_deviation``, ``hessian_asymmetry``,
    ``warnings``, ``n_imaginary``): a frame's ``record["vibrational_analysis"]``."""
    return {k: v for k, v in d.items() if k not in _VIB and k not in _STRUCTURE_KEYS}


def thermochemistry_model(t: Mapping[str, Any]) -> Thermochemistry:
    """One of linus's thermochemistry entries (the library's dict) as qcdata's model."""
    return Thermochemistry.model_validate({k: t[k] for k in _THERMO if t.get(k) is not None})


def thermochemistry_record(t: Mapping[str, Any]) -> dict[str, Any]:
    """What one of linus's thermochemistry entries has beyond qcdata's model (``modes``,
    ``low_frequencies``, ``energy_0k``, ``mass``, ``warnings``, the corrections): an entry of a
    frame's ``record["thermochemistry"]``."""
    return {k: v for k, v in t.items() if k not in _THERMO}


def harmonic_analysis(numbers: Sequence[int | str], positions_bohr: Any, hessian: Any,
                      dipole_derivatives: Any = None, **freq: Any) -> VibrationalAnalysis:
    """Harmonic analysis of a Cartesian Hessian ((3N, 3N) or (N, N, 3, 3) as PySCF's, Eh/bohr²)
    of atoms `numbers` (atomic numbers or symbols) at `positions_bohr` (N x 3), as qcdata's
    `VibrationalAnalysis`. `dipole_derivatives` (3N x 3) add IR intensities. Keyword arguments:
    the analysis's ``freq`` settings (see the module docs)."""
    from ._library import atomic_numbers

    z = atomic_numbers(numbers)
    n3 = 3 * len(z)
    h = np.asarray(hessian, dtype=float)
    if h.ndim == 4:  # PySCF's (N, N, 3, 3)
        h = h.transpose(0, 2, 1, 3).reshape(n3, n3)
    h = np.ascontiguousarray(h.reshape(n3, n3))
    pos = np.ascontiguousarray(np.asarray(positions_bohr, dtype=float).reshape(-1, 3))
    dd = None if dipole_derivatives is None else np.ascontiguousarray(
        np.asarray(dipole_derivatives, dtype=float).reshape(n3, 3))
    return analysis_model(_linus.harmonic_analysis(z, pos, h, _freq_json(freq), dd), dd)


def thermochemistry(analysis: VibrationalAnalysis, energy: float, multiplicity: int = 1,
                    **freq: Any) -> list[Thermochemistry]:
    """Ideal-gas thermochemistry of a harmonic analysis (any program's `VibrationalAnalysis`
    with its masses, frequencies, external flags, rotor type, moments of inertia and symmetry
    number) at electronic energy `energy` (Eh): one `Thermochemistry` per temperature
    (``temperature`` may be a list). Keyword arguments: the thermochemistry's ``freq`` settings
    (see the module docs); ``codata`` names the constants (default CODATA 2022)."""
    codata = freq.pop("codata", "codata2022")
    missing = [k for k in ("rotor_type", "moments_of_inertia", "symmetry_number") if getattr(analysis, k) is None]
    if missing:
        from .errors import InputError

        raise InputError(f"thermochemistry needs the analysis's {', '.join(missing)}")
    nm, n = len(analysis.frequencies), len(analysis.masses)
    flags = list(analysis.external) or [False] * nm
    # the library's analysis: the fields thermochemistry reads from qcdata's, neutral values for
    # the ones it does not read (the modes, the principal axes, the symmetry detection)
    d = {
        "natoms": n, "numbers": [0] * n, "masses": _jsonable(analysis.masses), "projection": analysis.projection.value,
        "codata": codata, "rotor_type": analysis.rotor_type.value, "center_of_mass": [0.0, 0.0, 0.0],
        "moments_of_inertia": _jsonable(analysis.moments_of_inertia),
        "principal_axes": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        "rotational_constants": [None, None, None], "rotational_constants_ghz": [None, None, None],
        "point_group": analysis.point_group or "C1", "symmetry_number": int(analysis.symmetry_number),
        "detected_symmetry_number": int(analysis.symmetry_number),
        "symmetry_number_source": analysis.symmetry_number_source or "input", "symmetry_tol": 0.0,
        "max_deviation": 0.0, "frequencies": _jsonable(analysis.frequencies), "eigenvalues": [0.0] * nm,
        "reduced_masses": [0.0] * nm, "force_constants": [0.0] * nm, "external": flags,
        "n_imaginary": analysis.n_imaginary(), "normal_modes_cartesian": [[0.0] * (3 * n)] * nm,
        "normal_modes_mass_weighted": [[0.0] * (3 * n)] * nm, "ir_intensities": None, "hessian_asymmetry": 0.0,
    }
    out = _linus.thermochemistry(d, float(energy), int(multiplicity), _freq_json(freq))
    return [thermochemistry_model(t) for t in out]


def atomic_mass(z: int | str, table: str = "isotope") -> float:
    """Default mass (u) of an element: ``"isotope"`` (most abundant) or ``"average"``."""
    from ._library import atomic_numbers

    return _linus.atomic_mass(atomic_numbers([z])[0], table)
