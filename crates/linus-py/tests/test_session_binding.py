"""The binding's session (`linus._linus.Session`): requests in lineages keyed by qcdata's
canonical text, committed on success, explicit close, declared calctypes."""

import json

import numpy as np
import pytest

import linus
from linus import _linus

qcdata = pytest.importorskip("qcdata")
lineage_key = getattr(qcdata, "lineage_key", None)

# water without symmetry (bohr)
POS = np.array([[0.0, 0.0, 0.223], [0.038, 1.427, -0.890], [-0.019, -1.436, -0.879]])
OPTS = json.dumps({"method": "hf", "basis": "sto-3g"})


def key(value):
    """A lineage's key text, as qcdata writes it (its own function when installed)."""
    if lineage_key is not None:
        return lineage_key(value)
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def moved(dz):
    x = POS.copy()
    x[0, 2] += dz
    return x


def session(**kw):
    return _linus.Session([8, 1, 1], POS, options=OPTS, **kw)


def test_requests_continue_their_lineage_and_commit_on_success():
    s = session()
    first = s.compute(None, "gradient")
    guess = first["resolved"]["record"]["scf"]["guess"]
    assert first["resolved"]["record"]["warm_start"] is False and guess["source"] == "sad"
    step = s.compute(moved(0.05), "gradient")
    rec = step["resolved"]["record"]
    assert rec["warm_start"] is True and rec["scf"]["guess"]["from_lineage"] is None
    assert step["iterations"] < first["iterations"]
    # a lineage of its own, keyed by qcdata's canonical text, seeded from the default one
    irc = ("irc", "forward")
    branch = s.compute(moved(0.08), "gradient", json.dumps({"lineage": key(irc), "seed": [key(None)]}))
    g = branch["resolved"]["record"]["scf"]["guess"]
    assert g["lineage"] == ["irc", "forward"] and g["from_lineage"] is None and g["source"] == "lineage"
    assert s.lineages() == sorted([key(None), key(irc)])
    # commit false stores nothing
    s.compute(moved(0.5), "gradient", json.dumps({"lineage": key("fd"), "commit": False}))
    assert key("fd") not in s.lineages()
    s.reset(key(irc))
    assert s.lineages() == [key(None)]
    s.reset(None)
    assert s.lineages() == []


def test_declared_calctypes_and_close():
    s = session(calctypes=["gradient"])
    s.compute(None, "gradient")
    with pytest.raises(linus.InputError):
        s.compute(None, "hessian")
    with pytest.raises(linus.InputError):
        s.compute(None, "energy", json.dumps({"lineage": "{"}))
    with pytest.raises(linus.InputError):
        s.compute(None, "energy", json.dumps({"bogus": 1}))
    assert not s.closed
    s.close()
    assert s.closed
    with pytest.raises(linus.InputError, match="closed"):
        s.compute(None, "energy")
    s.close()  # closing twice is fine
    assert repr(s) == "Session(<closed>)"


def test_errors_carry_codes():
    with pytest.raises(linus.InputError) as e:
        _linus.Session([8, 1, 1], POS, options=json.dumps({"method": "hf", "basis": "sto-3g", "scf": {"maxiter": 3}}))
    assert e.value.kind == "input" and e.value.code == "unknown_key"
    with pytest.raises(linus.InputError) as e:
        _linus.Session([8, 1, 1], POS, options=json.dumps({"method": "hf", "basis": "sto-3g",
                                                          "response": {"tol": 1e-6}}), calctypes=["gradient"])
    assert e.value.code == "not_applicable"
    assert _linus.device_lost() is False
