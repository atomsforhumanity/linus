"""Optional outputs change no other number (decision 25): a request with every wavefunction output
(the coefficients, the Fock matrices of the SCF's last build, the overlap, the core Hamiltonian)
returns the same results, bit for bit, as one without them, for energies, gradients and Hessians;
and the Fock matrices have the returned orbitals as their fresh eigenvectors.

On a GPU build the same runs on the device with ``LINUS_TEST_BACKEND=gpu``.
"""

import json
import os

import linus
import numpy as np
import pytest
from helpers import program_input

BACKEND = os.environ.get("LINUS_TEST_BACKEND", "cpu")
ALL = ("orbitals", "hamiltonian", "overlap", "core_hamiltonian")
# wall times, the machine's free memory (and the DF tensors' budget, which is within it when no
# memory_gib is given), and the lineage's name (each request runs in its own, both fresh): run
# facts, not results
VOLATILE = {"seconds", "time", "available_gib", "budget_gib", "df_tensor_budget_gib", "lineage"}

CASES = {
    # exact J/K builds incrementally: the SCF ends on a full build (numerics epoch 9)
    "uhf-exact": dict(name="water", charge=1, multiplicity=2, method="hf", jk="exact"),
    "rks-df": dict(name="water", charge=0, multiplicity=1, method="b3lyp", jk="df"),
    "rks-ecp": dict(name="ch3i", charge=0, multiplicity=1, method="b3lyp", jk="df"),
}


def _structure(molecules, case):
    from pathlib import Path

    from qcdata import Structure

    c = CASES[case]
    if c["name"] == "ch3i":
        path = Path(__file__).resolve().parents[3] / "tests" / "molecules" / "ch3i.xyz"
        return Structure.open(path)
    return molecules(c["name"], c["charge"], c["multiplicity"])


def _strip(x):
    """A frame's JSON without the outputs' own fields, the wall times and the free memory."""
    if isinstance(x, dict):
        # (t_*: an SCF iteration's times; mem_hwm_*: the process's peak memory at checkpoints)
        return {k: _strip(v) for k, v in x.items() if k not in VOLATILE and not k.startswith(("t_", "mem_hwm"))}
    if isinstance(x, list):
        return [_strip(v) for v in x]
    return x


def _diff(x, y, p=""):
    """The first leaf paths where two JSON values differ."""
    if isinstance(x, dict) and isinstance(y, dict):
        return [d for k in sorted(set(x) | set(y)) for d in _diff(x.get(k), y.get(k), f"{p}.{k}")][:5]
    if isinstance(x, list) and isinstance(y, list) and len(x) == len(y):
        return [d for i, (u, v) in enumerate(zip(x, y)) for d in _diff(u, v, f"{p}[{i}]")][:5]
    return [] if x == y else [f"{p}: {str(x)[:60]} != {str(y)[:60]}"]


@pytest.mark.parametrize("calctype", ["energy", "gradient", "hessian"])
@pytest.mark.parametrize("case", list(CASES))
def test_outputs_change_no_number(molecules, case, calctype):
    if case == "rks-ecp" and calctype == "hessian":
        pytest.skip("covered by the energy and gradient; the ECP Hessian takes long")
    c = CASES[case]
    inp = program_input(_structure(molecules, case), calctype, method=c["method"], jk=c["jk"], threads=4,
                        backend=BACKEND)
    with linus.Session(inp) as s:
        plain = s.compute(lineage="plain")
        full = s.compute(lineage="full", outputs=ALL)
    wa, wb = plain.results.wavefunction, full.results.wavefunction
    assert wa.orbitals.coefficients_a is None and wa.overlap is None and wa.hamiltonian_a is None
    a = json.loads(plain.model_dump_json(exclude={"execution"}))
    b = json.loads(full.model_dump_json(exclude={"execution"}))
    for d in (a, b):
        d["results"].pop("wavefunction")
    sa, sb = _strip(a), _strip(b)
    assert sa == sb, _diff(sa, sb)
    # the levels are the same; the orbitals are the Hamiltonian's eigenvectors in the overlap
    oa, ob = wa.orbitals, wb.orbitals
    for name in ("energies_a", "energies_b", "occupations_a", "occupations_b"):
        x, y = getattr(oa, name), getattr(ob, name)
        assert (x is None and y is None) or np.array_equal(x, y)
    s_ = np.asarray(wb.overlap)
    for c_, h, e in ((ob.coefficients_a, wb.hamiltonian_a, ob.energies_a),
                     (ob.coefficients_b, wb.hamiltonian_b, ob.energies_b)):
        if c_ is None:
            continue
        c_, h = np.asarray(c_), np.asarray(h)
        assert np.abs(h @ c_ - s_ @ c_ * e).max() < 1e-11
        assert np.abs(c_.T @ s_ @ c_ - np.eye(c_.shape[1])).max() < 1e-11
    assert ob.restricted == (wb.hamiltonian_b is None)


def test_each_output_brings_what_it_implies(water):
    """The basis set comes with any AO-indexed array and only then, and an AO matrix alone ships
    no coefficients (qc* decision 25a): ``overlap`` and ``core_hamiltonian`` alone have the basis
    set and no coefficients; ``hamiltonian`` brings the overlap and the orbitals that are its
    eigenvectors; ``orbitals`` the coefficients alone; nothing, only the levels."""
    inp = program_input(water, "energy", threads=4, backend=BACKEND)
    with linus.Session(inp) as s:
        def wfn(*outputs):
            return s.compute(lineage=list(outputs), outputs=outputs).results.wavefunction

        none = wfn()
        assert none.basis is None and none.orbitals.coefficients_a is None and none.overlap is None
        for name in ("overlap", "core_hamiltonian"):
            w = wfn(name)
            assert w.basis is not None and w.basis.nao == 24 and w.orbitals.coefficients_a is None
            assert getattr(w, name).shape == (24, 24) and w.hamiltonian_a is None
        w = wfn("orbitals")
        assert w.basis.nao == 24 and w.orbitals.coefficients_a.shape == (24, 24) and w.overlap is None
        w = wfn("hamiltonian")
        assert w.overlap is not None and w.orbitals.coefficients_a is not None and w.hamiltonian_a is not None
