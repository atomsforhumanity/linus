"""Accuracy targets (qc*'s shared keyword ``accuracy`` and a session request's ``accuracy``;
decisions 22 and 25b): linus's thresholds from the targets, the record key ``accuracy``, the
targets met against tightly converged references, and no number moved without targets."""

import linus
import numpy as np
import pytest
from helpers import program_input
from qcdata import Accuracy, ProgramError

TIGHT = {"scf": {"e_tol": 1e-13, "g_tol": 1e-10}}


def errors(out, ref):
    """The largest errors of the energy, gradient and Hessian against a reference output."""
    r, t = out.results, ref.results
    e = {"energy": abs(r.energy - t.energy)}
    if r.gradient is not None:
        e["gradient"] = float(np.abs(np.asarray(r.gradient) - np.asarray(t.gradient)).max())
    if r.hessian is not None:
        e["hessian"] = float(np.abs(np.asarray(r.hessian) - np.asarray(t.hessian)).max())
    return e


@pytest.mark.parametrize("case", ["rks-water", "uks-ch3", "uhf-nh3+"])
@pytest.mark.parametrize("calctype,target", [("energy", {"energy": 1e-6}), ("gradient", {"gradient": 1e-4}),
                                             ("hessian", {"hessian": 1e-4}), ("gradient", {"gradient": 1e-7})])
def test_targets_are_met(molecules, case, calctype, target):
    """Each target is met against a tight reference; the record says so with the thresholds it
    chose, and a loose target runs a looser SCF than the default."""
    name, q, m, method = {"rks-water": ("water", 0, 1, "b3lyp"), "uks-ch3": ("ch3", 0, 2, "b3lyp"),
                          "uhf-nh3+": ("nh3", 1, 2, "hf")}[case]
    s = molecules(name, q, m) if name != "ch3" else _ch3(q, m)
    extra = {"response": {"tol": 1e-10}} if calctype == "hessian" else {}
    ref = linus.compute(program_input(s, calctype, method=method, **TIGHT, **extra))
    out = linus.compute(program_input(s, calctype, method=method, accuracy=target))
    acc = out.record["accuracy"]
    assert acc["target"] == target and acc["met"] is True
    (q_name, t), = target.items()
    assert errors(out, ref)[q_name] <= t and acc["estimate"][q_name] <= t
    assert set(acc["thresholds"]) >= {"scf.e_tol", "scf.g_tol"}
    assert out.resolved.settings["scf"]["g_tol"] == acc["thresholds"]["scf.g_tol"]
    assert "accuracy" in out.resolved.reasons["scf.g_tol"]
    if calctype == "hessian":
        assert out.resolved.settings["response"]["tol"] == acc["thresholds"]["response.tol"]


def _ch3(q, m):
    from pathlib import Path

    from qcdata import Structure

    return Structure.open(Path(__file__).resolve().parents[3] / "tests" / "molecules" / "ch3.xyz", charge=q,
                          multiplicity=m)


def test_no_target_changes_no_number(water):
    """Without targets: no record key and the numbers of before (an energy target looser than the
    default's thresholds would run a cheaper SCF: another calculation)."""
    plain = linus.compute(program_input(water, "gradient", method="b3lyp"))
    assert "accuracy" not in plain.record
    again = linus.compute(program_input(water, "gradient", method="b3lyp"))
    assert plain.results.energy == again.results.energy
    np.testing.assert_array_equal(plain.results.gradient, again.results.gradient)
    loose = linus.compute(program_input(water, "gradient", method="b3lyp", accuracy={"gradient": 1e-3}))
    assert loose.results.scf_iterations < plain.results.scf_iterations


def test_given_thresholds_are_floors(water):
    """A target tightens a given threshold, never loosens it."""
    tight = linus.compute(program_input(water, "gradient", scf={"g_tol": 1e-9}, accuracy={"gradient": 1e-3}))
    assert tight.record["accuracy"]["thresholds"]["scf.g_tol"] == 1e-9
    looser = linus.compute(program_input(water, "gradient", scf={"g_tol": 1e-5}, accuracy={"gradient": 3.6e-6}))
    assert looser.record["accuracy"]["thresholds"]["scf.g_tol"] == pytest.approx(1e-7)


def test_keyword_is_read_as_qcdata_reads_it(water):
    """A bare number, an unknown key and an object without targets are refused (Accuracy.from_keywords);
    a target of a calctype the input does not run is not applicable (D41); an Accuracy model is
    stored as its JSON form."""
    for bad, code in ((1e-6, None), ({"energy": 1e-6, "force": 1e-4}, "unknown_key"), ({}, "no_effect"),
                      ({"gradient": 1e-4}, "not_applicable"), ({"energy": -1.0}, None)):
        with pytest.raises(ProgramError) as info:
            linus.compute(program_input(water, accuracy=bad))
        assert info.value.error.kind == "input" and (code is None or info.value.error.code == code), bad
    inp = program_input(water, accuracy=Accuracy(energy=1e-7))
    assert inp.keywords["accuracy"] == {"energy": 1e-7}
    assert linus.compute(inp).record["accuracy"]["met"] is True


def test_request_accuracy_replaces_the_inputs(water):
    """A session request's accuracy (an Accuracy, or a number for its calctype's highest derivative)
    replaces the input's for that request only."""
    inp = program_input(water, "gradient", method="b3lyp", accuracy={"gradient": 1e-6})
    with linus.Session(inp) as s:
        a = s.compute()
        b = s.compute(accuracy=1e-3, lineage="loose")
        c = s.compute(calctype="energy", accuracy=Accuracy(energy=1e-5), lineage="e")
        d = s.compute(lineage="again")
    assert a.record["accuracy"]["target"] == {"gradient": 1e-6} == d.record["accuracy"]["target"]
    assert b.record["accuracy"]["target"] == {"gradient": 1e-3}
    assert b.record["accuracy"]["thresholds"]["scf.g_tol"] > a.record["accuracy"]["thresholds"]["scf.g_tol"]
    assert c.record["accuracy"]["target"] == {"energy": 1e-5}
    assert a.results.energy == d.results.energy
