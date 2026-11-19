"""Converters between qcdata's wavefunctions and PySCF's orbitals.

The models are qcdata's (`qcdata.Wavefunction` with its basis set, `qcdata.BasisSet` with its
`qcdata.Ecp`s, `qcdata.Orbitals`, `qcdata.AOConvention`); these are functions on them, for any
program's wavefunction with orbital coefficients (a linus frame's ``results.wavefunction`` with
the request output ``orbitals``, or one read from PySCF):

    from linus.convert import to_pyscf, from_pyscf
    wfn = frame.results.wavefunction                   # or qcdata.Wavefunction.model_validate_json(text)
    mol, mo_coeff, mo_energy, mo_occ = to_pyscf(frame.structure, wfn)
    structure, wfn = from_pyscf(mf.mol, mf.mo_coeff, mf.mo_energy, mf.mo_occ)

ECPs travel in the basis set (`BasisSet.ecp`: the local and semi-local channels, each term
coefficient r^power exp(-exponent r^2), and the core electrons each replaces,
`Ecp.core_electrons`, which the orbitals' `Orbitals.core_electrons` agree with).

Molden files of a wavefunction are qcinf's (`qcinf.molden.write(wfn, structure)`, and
`qcinf.molden.read`), for every program's orbitals, ECP cores included (``[core]``, afh-stack#67).
linus also writes the Molden file of its own orbitals itself with the setting ``output.molden``
(``results.files["orbitals.molden"]``; the command line's ``--molden``).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any, NamedTuple

import numpy as np
from qcdata import (
    PYSCF_CONVENTION,
    BasisSet,
    Ecp,
    EcpTerm,
    OrbitalKind,
    Orbitals,
    OrbitalSpin,
    Shell,
    Structure,
    Wavefunction,
)

__all__ = [
    "PyscfOrbitals",
    "from_pyscf",
    "from_pyscf_mol",
    "to_pyscf",
    "to_pyscf_mol",
]


def _dfact(n: int) -> float:
    """(n)!! with (-1)!! = 1."""
    r = 1.0
    while n > 1:
        r *= n
        n -= 2
    return r


def _normalize_raw(l: int, exps: np.ndarray, raw: np.ndarray) -> np.ndarray:
    """Scale coefficients of unnormalized primitives so that x^l of the shell has unit norm."""
    p = exps[:, None] + exps[None, :]
    s = float(raw @ ((math.pi / p) ** 1.5 * _dfact(2 * l - 1) / (2 * p) ** l) @ raw)
    return raw / math.sqrt(s)


def _check_centers(structure: Structure, basis: BasisSet) -> None:
    natoms = len(structure.symbols)
    if any(s.center >= natoms for s in basis.shells) or any(e.center >= natoms for e in basis.ecp):
        raise ValueError(f"the basis set has shells or ECPs on atoms beyond the structure's {natoms}")


def _spin_sets(wfn: Wavefunction) -> tuple[BasisSet, list[tuple[np.ndarray, np.ndarray | None, np.ndarray]]]:
    """The basis set and (coefficients, energies, occupations) per spin: two sets, the alpha and
    beta orbitals (restricted orbitals: the same coefficients and energies twice)."""
    orbitals = wfn.orbitals
    if wfn.basis is None or orbitals.coefficients_a is None:
        raise ValueError("this wavefunction has only orbital levels: converting it needs the orbitals' coefficients "
                         "and their basis set (the request output orbitals)")
    ca = np.asarray(orbitals.coefficients_a)
    cb = ca if orbitals.restricted else np.asarray(orbitals.coefficients_b)
    ea = orbitals.energies_a
    eb = ea if orbitals.restricted else orbitals.energies_b
    return wfn.basis, [(ca, ea, np.asarray(orbitals.occupations_a)), (cb, eb, np.asarray(orbitals.occupations_b))]


def _core(structure: Structure, basis: BasisSet) -> list[int]:
    """The core electrons per atom that the basis set's ECPs replace (`Ecp.core_electrons`;
    zeros without ECPs)."""
    core = [0] * len(structure.symbols)
    for e in basis.ecp:
        core[e.center] = int(e.core_electrons)
    return core


# --- PySCF ------------------------------------------------------------------------------


class PyscfOrbitals(NamedTuple):
    """Orbitals in PySCF's form: `mo_coeff` (nao x nmo, or 2 x nao x nmo unrestricted),
    `mo_energy` and `mo_occ` (restricted: total occupations 0..2)."""

    mol: Any
    mo_coeff: np.ndarray
    mo_energy: np.ndarray | None
    mo_occ: np.ndarray


def _pyscf_radial(terms: Sequence[EcpTerm]) -> list[list[list[float]]]:
    """A channel's terms in PySCF's radial form: one list per power of r, indexed by power + 2
    (Basis Set Exchange's n)."""
    rad: list[list[list[float]]] = [[] for _ in range(max((t.power for t in terms), default=-2) + 3)]
    for t in terms:
        rad[t.power + 2].append([float(t.exponent), float(t.coefficient)])
    return rad


def _pyscf_shell_order(structure: Structure, basis: BasisSet) -> tuple[Any, list[int]]:
    """A PySCF Mole for the structure, basis set and ECPs, and our shell index of each PySCF
    shell."""
    from pyscf import gto

    _check_centers(structure, basis)
    core = _core(structure, basis)
    pures = {s.pure for s in basis.shells if s.angular_momentum >= 2}
    if len(pures) > 1:
        raise ValueError("PySCF needs all d and higher shells spherical or all Cartesian")
    cart = pures == {False}
    labels = [f"{sym}{i + 1}" for i, sym in enumerate(structure.symbols)]
    pyscf_basis: dict[str, list[Any]] = {lab: [] for lab in labels}
    for s in basis.shells:
        l = s.angular_momentum
        # PySCF takes coefficients of radially normalized primitives and renormalizes
        d = s.coefficients / gto.gto_norm(l, s.exponents)
        pyscf_basis[labels[s.center]].append([l, *([float(e), float(c)] for e, c in zip(s.exponents, d))])
    ecp = {labels[e.center]: [core[e.center], [[-1, _pyscf_radial(e.local)],
                                               *([l, _pyscf_radial(t)] for l, t in enumerate(e.semilocal))]]
           for e in basis.ecp}
    mol = gto.M(
        atom=[(lab, tuple(map(float, xyz))) for lab, xyz in zip(labels, np.asarray(structure.geometry))],
        basis=pyscf_basis, ecp=ecp or None, unit="Bohr", charge=structure.charge, spin=structure.multiplicity - 1,
        cart=cart, verbose=0,
    )
    # match PySCF's shells (sorted by l per atom) to ours
    unused: dict[int, list[int]] = {}
    for i, s in enumerate(basis.shells):
        unused.setdefault(s.center, []).append(i)
    order = []
    for ib in range(mol.nbas):
        atom, l, exps = mol.bas_atom(ib), mol.bas_angular(ib), mol.bas_exp(ib)
        if mol.bas_nctr(ib) != 1:
            raise RuntimeError("unexpected general contraction in the PySCF basis")
        cand = [i for i in unused.get(atom, []) if basis.shells[i].angular_momentum == l
                and len(basis.shells[i].exponents) == len(exps)
                and np.allclose(basis.shells[i].exponents, exps, rtol=1e-14, atol=0)]
        if not cand:
            raise RuntimeError(f"cannot match PySCF shell {ib} (atom {atom}, l = {l})")
        order.append(cand[0])
        unused[atom].remove(cand[0])
    return mol, order


def to_pyscf_mol(structure: Structure, basis: BasisSet) -> Any:
    """A PySCF `Mole` with exactly this basis set and its ECPs (each with the core electrons it
    replaces, `Ecp.core_electrons`). The same functions; the AO order may differ for bases whose
    shells are not sorted by l per atom (see `to_pyscf`)."""
    return _pyscf_shell_order(structure, basis)[0]


def to_pyscf(structure: Structure, wavefunction: Wavefunction) -> PyscfOrbitals:
    """PySCF `Mole`, `mo_coeff`, `mo_energy` and `mo_occ` for a wavefunction's orbitals (with
    their coefficients), in PySCF's AO order and normalization, with their ECPs."""
    basis, ((ca, ea, oa), (cb, eb, ob)) = _spin_sets(wavefunction)
    orbitals = wavefunction.orbitals
    mol, order = _pyscf_shell_order(structure, basis)

    def conv(c: np.ndarray) -> np.ndarray:
        return basis.transform(c, PYSCF_CONVENTION, kind="coefficients", shell_order=order)

    if orbitals.restricted:
        return PyscfOrbitals(mol, conv(ca), ea, oa + ob)
    e = None if ea is None else np.stack([ea, eb])
    return PyscfOrbitals(mol, np.stack([conv(ca), conv(cb)]), e, np.stack([oa, ob]))


def from_pyscf_mol(mol: Any, name: str | None = None) -> tuple[Structure, BasisSet]:
    """The structure and basis set (AO order and convention: PySCF's; with its ECPs and the core
    electrons each replaces) of a PySCF `Mole`."""
    from pyscf import gto

    shells = []
    for ib in range(mol.nbas):
        l, exps = mol.bas_angular(ib), np.asarray(mol.bas_exp(ib), dtype=float)
        cs = np.asarray(mol.bas_ctr_coeff(ib), dtype=float).reshape(len(exps), -1)
        for ic in range(cs.shape[1]):
            raw = cs[:, ic] * gto.gto_norm(l, exps)
            shells.append(Shell(center=mol.bas_atom(ib), angular_momentum=l, pure=not mol.cart, exponents=exps,
                                coefficients=_normalize_raw(l, exps, raw)))
    ecps = []
    if mol.has_ecp():
        core = [int(mol.atom_nelec_core(i)) for i in range(mol.natm)]
        for i in range(mol.natm):
            if not core[i]:
                continue
            _, channels = mol._ecp[mol.atom_symbol(i)]
            by_l: dict[int, list[EcpTerm]] = {}
            for l, rad in channels:
                # a term [exponent, coefficient]; spin-orbit ECPs carry a third number
                if any(len(t) != 2 for terms in rad for t in terms):
                    raise ValueError("spin-orbit ECP terms cannot be represented (scalar ECPs only)")
                by_l[l] = [EcpTerm(power=n - 2, exponent=t[0], coefficient=t[1])
                           for n, terms in enumerate(rad) for t in terms]
            top = max((l for l in by_l if l >= 0), default=-1) + 1
            ecps.append(Ecp(center=i, core_electrons=core[i], local=by_l.get(-1, []),
                            semilocal=[by_l.get(l, []) for l in range(top)]))
    basis = BasisSet(name=name if name is not None else (mol.basis if isinstance(mol.basis, str) else None),
                     shells=shells, ecp=ecps, ao_convention=PYSCF_CONVENTION)
    structure = Structure(symbols=[mol.atom_pure_symbol(i) for i in range(mol.natm)],
                          geometry=mol.atom_coords(unit="Bohr"), charge=mol.charge, multiplicity=mol.spin + 1)
    return structure, basis


def from_pyscf(mol: Any, mo_coeff: Any, mo_energy: Any, mo_occ: Any,
               name: str | None = None) -> tuple[Structure, Wavefunction]:
    """Structure and wavefunction (canonical orbitals with their basis set) from PySCF's `mol`,
    `mo_coeff`, `mo_energy`, `mo_occ` (restricted: 2-D `mo_coeff`; unrestricted: a pair).
    Restricted total occupations n are split as alpha min(n, 1), beta n - alpha. The basis set
    keeps PySCF's AO convention; use `Wavefunction.with_convention` for another."""
    structure, basis = from_pyscf_mol(mol, name)
    core = _core(structure, basis) if basis.ecp else None
    c = np.asarray(mo_coeff, dtype=float)
    e = np.asarray(mo_energy, dtype=float)
    o = np.asarray(mo_occ, dtype=float)
    if c.ndim == 2:
        occ_a = np.minimum(o, 1.0)
        orbitals = Orbitals(spin=OrbitalSpin.restricted, kind=OrbitalKind.canonical, coefficients_a=c, energies_a=e,
                            occupations_a=occ_a, occupations_b=o - occ_a, core_electrons=core)
    else:
        orbitals = Orbitals(spin=OrbitalSpin.unrestricted, kind=OrbitalKind.canonical, coefficients_a=c[0],
                            coefficients_b=c[1], energies_a=e[0], energies_b=e[1], occupations_a=o[0],
                            occupations_b=o[1], core_electrons=core)
    return structure, Wavefunction(orbitals=orbitals, basis=basis)
