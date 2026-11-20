"""linus's report (the binding's dict, the CLI's JSON) as qcdata data: the typed fields where
qcdata has them (the wavefunction: the orbital levels always; the orbitals' coefficients and the
AO matrices on request, with the basis set they share), the rest in ``extras`` under the names
agreed with qc* until qcdata adds them (``gradient_components``), the Molden file of
``output.molden`` in ``files``, the run record in ``record`` (linus's ``resolved.record``, the
SCF's history and stability analyses, the solvation, the sizes, the harmonic analysis's fields
beyond qcdata's model) and the timings, threads and peak memory in ``execution``."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
from qcdata import (
    BasisSet,
    ExecutionInfo,
    Frame,
    OrbitalKind,
    Orbitals,
    OrbitalSpin,
    ProgramWarning,
    Provenance,
    Resolved,
    SinglePointData,
    Wavefunction,
)

from . import _linus, _resources, vib


#: The report's run facts a frame keeps in its record (none has a field in qcdata's results):
#: linus's numerics epoch (what a result store keys on beside the settings), the HOMO-LUMO gap
#: (Eh: the lowest unoccupied minus the highest occupied orbital energy over both spins), the VV10 block of -V functionals (its
#: energy, included in ``energy_components["xc"]``; NLC grid; points), the response iterations of
#: a Hessian and of its electric-field response (``ir``), the solvation's surface statistics.
_REPORT_RECORD = ("numerics_epoch", "homo_lumo_gap", "vv10", "cphf_iterations", "polarizability_cphf_iterations",
                  "solvation")


def provenance() -> Provenance:
    """linus as the producer of results."""
    return Provenance(program="linus", program_version=_linus.__version__)


#: The wavefunction's AO matrices (qcdata's `Wavefunction` fields): the request outputs that add
#: them. Each comes with the basis set, which qcdata's AO-indexed arrays share
#: (`Wavefunction.basis`).
AO_OUTPUTS = frozenset({"hamiltonian", "overlap", "core_hamiltonian"})


def orbitals(r: Mapping[str, Any], coefficients: bool = False) -> Orbitals:
    """The canonical orbitals of a report as qcdata's `Orbitals`: the levels (energies and
    occupations per spin; restricted orbitals one set of energies) and the core electrons of
    the ECP atoms, and with `coefficients` the coefficients (in linus's AO order, which is
    qcdata's default convention; their basis set is the wavefunction's)."""
    restricted = bool(r["restricted"])
    ecp = (r.get("structure") or {}).get("ecp_core_electrons")
    fields: dict[str, Any] = {}
    if coefficients:
        fields = {"coefficients_a": r["mo_coefficients_alpha"],
                  "coefficients_b": None if restricted else r["mo_coefficients_beta"]}
    return Orbitals(spin=OrbitalSpin.restricted if restricted else OrbitalSpin.unrestricted,
                    kind=OrbitalKind.canonical,
                    energies_a=np.asarray(r["orbital_energies_alpha"], dtype=float),
                    energies_b=None if restricted else np.asarray(r["orbital_energies_beta"], dtype=float),
                    occupations_a=np.asarray(r["occupations_alpha"], dtype=float),
                    occupations_b=np.asarray(r["occupations_beta"], dtype=float),
                    core_electrons=[int(n) for n in ecp] if ecp and any(ecp) else None, **fields)


def wavefunction(r: Mapping[str, Any], outputs: set[str] = frozenset(),
                 basis: Callable[[], BasisSet] | None = None) -> Wavefunction:
    """The wavefunction of a report: the orbital levels always; with the request output
    ``orbitals`` the coefficients; the Fock matrices of the SCF's last build (``hamiltonian``:
    one per spin of unrestricted orbitals; the caller adds what it implies, ``overlap`` and
    ``orbitals``); the overlap; the core Hamiltonian. The basis set (`basis` makes it) comes
    with any of them and only then (qc* decision 25a): an overlap alone has no coefficients."""
    coefficients = basis is not None and "orbitals" in outputs
    matrices: dict[str, Any] = {}
    if basis is not None:
        if "hamiltonian" in outputs:
            matrices["hamiltonian_a"] = r["hamiltonian_alpha"]
            if not r["restricted"]:
                matrices["hamiltonian_b"] = r["hamiltonian_beta"]
        if "overlap" in outputs:
            matrices["overlap"] = r["overlap"]
        if "core_hamiltonian" in outputs:
            matrices["core_hamiltonian"] = r["core_hamiltonian"]
    if coefficients or matrices:
        matrices["basis"] = basis()
    return Wavefunction(orbitals=orbitals(r, coefficients), **matrices)


def single_point(r: Mapping[str, Any], outputs: set[str] = frozenset(),
                 basis: Callable[[], BasisSet] | None = None) -> SinglePointData:
    """The results of a report; `outputs` the request's optional outputs (the coefficients and
    AO matrices need `basis`, which makes their basis set; see `wavefunction`). ``extras`` holds
    plain JSON values (lists, not arrays: qcdata writes them as they are), so a frame always
    serializes."""
    extras: dict[str, Any] = {}
    if r.get("gradient_parts") is not None:
        extras["gradient_components"] = {k: np.asarray(v, dtype=float).tolist() for k, v in r["gradient_parts"].items()
                                         if v is not None}
    fields: dict[str, Any] = {
        "energy": r["energy"],
        "nuclear_repulsion_energy": r["components"]["nuclear_repulsion"],
        "energy_components": {k: v for k, v in r["components"].items() if v is not None},
        "scf_iterations": r["iterations"],
        "dipole_moment": [float(x) for x in r["dipole"]],
        "scf_s_squared": r.get("s2"),
        "wavefunction": wavefunction(r, outputs, basis),
        "calcinfo_natoms": r["natoms"],
        "calcinfo_nbasis": r["nbf"],
        "calcinfo_nmo": r["nmo"],
        "calcinfo_nalpha": r["nalpha"],
        "calcinfo_nbeta": r["nbeta"],
        "extras": extras,
    }
    if r.get("gradient") is not None:
        fields["gradient"] = np.asarray(r["gradient"], dtype=float)
    if r.get("hessian") is not None:
        fields["hessian"] = np.asarray(r["hessian"], dtype=float)
    if r.get("polarizability") is not None:
        fields["polarizability"] = np.asarray(r["polarizability"], dtype=float)
    if r.get("files"):
        fields["files"] = dict(r["files"])
    if r.get("vibrational_analysis") is not None:
        fields["vibrational_analysis"] = vib.analysis_model(r["vibrational_analysis"], r.get("dipole_derivatives"))
        fields["thermochemistry"] = [vib.thermochemistry_model(t) for t in r.get("thermochemistry") or ()]
    return SinglePointData(provenance=provenance(), **fields)


def record(r: Mapping[str, Any]) -> dict[str, Any]:
    """The run record of a report: linus's ``resolved.record`` (with qcdata's shared
    ``warm_start`` at its top), the SCF's thresholds, history and stability analyses, the
    report's run facts (`_REPORT_RECORD`), the sizes and, for a Hessian, what linus's harmonic
    analysis and thermochemistry have beyond qcdata's models (`linus.vib.analysis_record`)."""
    rec = dict((r.get("resolved") or {}).get("record") or {})
    scf = dict(rec.get("scf") or {})
    for k in ("fock_builds", "incremental_builds", "convergence", "history", "stability"):
        if k in (r.get("scf") or {}):
            scf[k] = r["scf"][k]
    rec["scf"] = scf
    for k in _REPORT_RECORD:
        if r.get(k) is not None:
            rec[k] = r[k]
    rec["sizes"] = {"nbf": r.get("nbf"), "nmo": r.get("nmo"), **dict(r.get("resources") or {})}
    if r.get("vibrational_analysis") is not None:
        rec["vibrational_analysis"] = vib.analysis_record(r["vibrational_analysis"])
        rec["thermochemistry"] = [vib.thermochemistry_record(t) for t in r.get("thermochemistry") or ()]
    return rec


def resolved(r: Mapping[str, Any]) -> Resolved:
    """The settings of a report that reproduce it, and why each ``auto`` took its value."""
    res = r.get("resolved") or {}
    return Resolved(settings=keywords(res.get("settings") or {}), reasons=res.get("reasons") or {})


def keywords(settings: Mapping[str, Any]) -> dict[str, Any]:
    """linus's resolved settings as qcdata keywords: without the method and basis set, which
    the model carries."""
    return {k: v for k, v in settings.items() if k not in ("method", "basis")}


def execution(r: Mapping[str, Any] | None, threads: int | None, wall: float,
              peak_gib: float | None = None, ran: Mapping[str, Any] | None = None) -> ExecutionInfo:
    """What the request ran with: wall time, threads, the process's peak resident memory
    during it (`linus._resources.PeakMemory`), linus's timings, and the part of the run's
    environment (decision 29) only linus knows: its peak device memory during the request, and
    the GPU its backend opened (``devices``; an explicit empty list when it ran on the CPU, which
    a runner keeps), from the record of what it reached (`ran`, default `r`; left unset
    when a failure reached nothing, for the runner to fill with its grant). The machine's part
    (host, CPU model, C library) is the runner's to record: one source for each."""
    r = r or {}
    record = ((r if ran is None else ran).get("resolved") or {}).get("record")
    where = {}
    if record is not None:
        device = (record.get("backend") or {}).get("device")
        where["devices"] = [device["name"]] if device else []
    return ExecutionInfo(wall_time=wall, threads=threads, peak_memory_gib=peak_gib,
                         peak_device_memory_gib=(r.get("resources") or {}).get("mem_device_peak_gib"),
                         timings=dict(r.get("timings") or {}), **where)


def warnings(r: Mapping[str, Any]) -> list[ProgramWarning]:
    """The report's warnings as qcdata's."""
    return [ProgramWarning(code=w["code"], message=w["message"], path=w.get("path") or None)
            for w in r.get("warnings") or ()]


def frame(structure: Any, r: Mapping[str, Any], outputs: set[str], threads: int | None, wall: float,
          extra: list[ProgramWarning], basis: Callable[[], BasisSet] | None = None,
          peak_gib: float | None = None) -> Frame:
    """The frame of a successful request; ``logs`` the SCF's progress lines of
    ``output.verbose`` (decision 24: in process they go there, never to stderr)."""
    return Frame(structure=structure, results=single_point(r, outputs, basis), success=True,
                 warnings=[*extra, *warnings(r)], record=record(r), logs=r.get("logs"),
                 execution=execution(r, threads, wall, peak_gib))
