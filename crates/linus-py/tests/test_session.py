"""linus.Session and linus.compute on qcdata objects: frames, lineages (keys, seeds, commit,
reset), failures as qcdata.ProgramError with their frame, given orbitals, plans, the program
object, and qccompute's conformance check of the Session protocol."""

import numpy as np
import pytest

qcdata = pytest.importorskip("qcdata")
if not hasattr(qcdata, "lineage_key"):
    pytest.skip("needs qcdata's data-only branch (Request, ProgramError, lineage_key)", allow_module_level=True)

import linus  # noqa: E402
from helpers import failure  # noqa: E402

POS = np.array([[0.0, 0.0, 0.223], [0.038, 1.427, -0.890], [-0.019, -1.436, -0.879]])
WATER = qcdata.Structure(symbols=["O", "H", "H"], geometry=POS)


def inp(method="hf", calctype="gradient", **keywords):
    return qcdata.ProgramInput(program="linus", calctype=calctype, structure=WATER,
                               model={"method": method, "basis": "sto-3g"}, keywords=keywords)


def moved(dz, atom=0):
    x = POS.copy()
    x[atom, 2] += dz
    return x


def test_compute_returns_a_program_output_and_raises_failures():
    out = linus.compute(inp())
    assert isinstance(out, qcdata.ProgramOutput) and out.success
    assert out.results.gradient.shape == (3, 3) and out.results.scf_iterations > 0
    assert out.record["warm_start"] is False and out.record["scf"]["guess"]["source"] == "sad"
    assert "method" not in out.resolved.settings and out.execution.threads == 1
    assert sum(out.results.energy_components.values()) == pytest.approx(out.results.energy, abs=1e-10)
    # decision 9: a failure raises a qcdata.ProgramError with the failed output
    bad = failure(inp(scf={"maxiter": 3}))
    assert bad.error.kind == "input" and bad.error.code == "unknown_key"
    assert bad.traceback == bad.error.message and bad.results.provenance.program == "linus"
    # a calculation that ran and failed: its partial results, record and execution
    conv = failure(inp(scf={"max_iter": 2, "escalate": False}))
    assert conv.error.kind == "convergence" and conv.record["scf"]["history"]
    assert conv.results.provenance.program_version == linus.__version__ and conv.execution.threads == 1
    # D41: a one-off declares its calctype only
    hess_only = failure(inp(response={"tol": 1e-6}))
    assert hess_only.error.code == "not_applicable"
    with pytest.raises(TypeError):
        linus.compute({"program": "linus"})


def test_a_session_continues_its_lineages():
    opened = inp("b3lyp", grid={"xc": "pyscf:1"})
    with linus.Session(opened) as s:
        assert s.program == "linus" and s.program_version == linus.__version__ and s.input_data is opened
        first = s.compute()
        assert isinstance(first, qcdata.Frame) and first.success
        assert first.record["warm_start"] is False
        assert first.record["scf"]["closed_shell_probe"]["skipped"] == "gap_above_threshold"
        step = s.compute(moved(0.05))
        assert step.record["warm_start"] is True and step.record["scf"]["guess"]["orthonormalized"]
        assert step.record["scf"]["closed_shell_probe"]["skipped"] == "later_geometry"
        assert step.results.scf_iterations < first.results.scf_iterations
        np.testing.assert_array_equal(step.structure.geometry, moved(0.05))
        # an independent chain, keyed by any JSON value (a tuple is its list), seeded
        b = s.compute(moved(0.08), lineage=("irc", "forward"), seed=["none", None])
        assert b.record["scf"]["guess"]["source"] == "lineage" and b.record["scf"]["guess"]["lineage"] == ["irc", "forward"]
        assert sorted(map(str, s.lineages())) == sorted(map(str, [None, ["irc", "forward"]]))
        # reset(None) forgets only the default lineage; reset() every one
        s.reset(None)
        assert s.lineages() == [["irc", "forward"]]
        again = s.compute(moved(0.05))
        assert again.record["warm_start"] is False and again.record["scf"]["guess"]["reason"] == "no_history"
        s.reset()
        assert s.lineages() == []
        # a Hessian: energy, gradient and the analysis
        h = s.compute(calctype="hessian", outputs=["ir"])
        assert h.results.gradient is not None and h.results.vibrational_analysis is not None
        assert h.results.polarizability is not None and h.results.vibrational_analysis.ir_intensities is not None
    assert s.closed


def test_commit_false_reads_the_lineage_at_its_place():
    def chain(ks, commits):
        with linus.Session(inp()) as s:
            frames = s.compute_many([qcdata.Request(positions=moved(0.015 * (k + 1)), commit=c)
                                     for k, c in zip(ks, commits)])
            return [f.results.energy for f in frames], s.lineages()

    many, lineages = chain([0, 1, 2, 3], [True, False, True, False])
    assert many[:2] == chain([0, 1], [True, True])[0]
    assert many[2:] == chain([0, 2, 3], [True, True, True])[0][1:]
    assert lineages == [None]


def test_failures_are_program_errors_with_their_frame():
    with linus.Session(inp(calctype="gradient"), calctypes=["gradient"]) as s:
        with pytest.raises(qcdata.ProgramError) as e:
            s.compute(calctype="hessian")
        assert e.value.error.code == "not_applicable" and isinstance(e.value, linus.InputError)
        assert e.value.frame is not None and not e.value.frame.success
        with pytest.raises(linus.InputError):
            s.compute(np.array([POS[0], POS[0], POS[2]]))
        # every request runs, then the failures raise together, with every frame (decision 9)
        with pytest.raises(qcdata.ProgramErrors) as g:
            s.compute_many([{"positions": POS}, {"positions": np.full((3, 3), np.nan)}, {"positions": moved(0.02)}])
        frames = g.value.results
        assert [f.success for f in frames] == [True, False, True]
        assert frames[1].error.kind == "input" and len(g.value.exceptions) == 1
        assert isinstance(g.value.exceptions[0], linus.InputError) and g.value.exceptions[0].frame is frames[1]
        with pytest.raises(linus.UnsupportedError):
            s.compute(outputs=["bogus"])
    with linus.Session(inp(scf={"max_iter": 2, "escalate": False})) as s:
        with pytest.raises(linus.ConvergenceError) as e:
            s.compute()
        assert e.value.error.kind == "convergence" and e.value.frame.record["scf"]["history"]


def test_resources_and_given_orbitals():
    class Grant:
        threads, memory_gib = 2, None

    s = linus.Session(inp(threads=4), ctx=Grant())
    f = s.compute()
    assert f.execution.threads == 2 and [w.code for w in f.warnings] == ["resource_capped"]
    wfn = s.compute(outputs=["orbitals"]).results.wavefunction
    assert wfn.orbitals.coefficients_a.shape[0] == 7 and wfn.basis.nao == 7
    s.close()
    given = qcdata.SinglePointData(provenance=qcdata.Provenance(program="linus"), wavefunction=wfn)
    g = linus.compute(inp().model_copy(update={"given": given}))
    assert g.success and g.record["scf"]["guess"]["source"] == "given"
    assert g.results.scf_iterations < f.results.scf_iterations
    nothing = failure(inp().model_copy(update={"given": qcdata.SinglePointData(
        provenance=qcdata.Provenance(program="linus"), energy=-75.0)}))
    assert nothing.error.code == "not_applicable"


def test_threads_without_the_keyword_are_capped_at_8():
    """Under a grant of 32 threads (qc* decision 16) a calculation without the `threads` keyword
    runs on linus's 8 (the plan's `estimates.threads`), with the memory need of 8; an explicit
    keyword runs as given. A direct call keeps the default of 1."""

    class Grant:
        threads, memory_gib = 32, None

    def run(**kw):
        with linus.Session(inp(calctype="energy", **kw), ctx=Grant()) as s:
            p = s.plan()
            assert p["resolved"]["settings"]["threads"] == s.compute().execution.threads
            return p["record"]["memory"]["required_gib"], s.compute()

    req8, f8 = run()
    assert f8.execution.threads == 8 and not f8.warnings
    assert run(threads="auto")[1].execution.threads == 8
    req32, f32 = run(threads=32)
    assert f32.execution.threads == 32 and not f32.warnings
    assert req32 > req8
    with linus.Session(inp(calctype="energy", threads=8), ctx=Grant()) as s:
        assert s.plan()["record"]["memory"]["required_gib"] == req8
    assert linus.compute(inp(calctype="energy")).execution.threads == 1


def test_plan_and_program():
    from linus.qc import ENTRY_POINT_GROUP, program

    with linus.Session(inp(threads=2)) as s:
        p = s.plan()
        settings = p["resolved"]["settings"]
        # the DF exchange tensor's format is decided before the run (auto: f64), nothing deferred
        assert settings["jk"]["df_storage"] == "f64" and p["resolved"]["deferred"] == []
        assert p["record"]["calctype"] == "hessian" and p["record"]["sizes"]["nbf"] == 7
        r = p["resources"]
        assert "threads" not in r and r["gpus"] == 0 and 0 < r["memory_gib"] < 1
        est = p["record"]["estimates"]
        assert est["memory_gib"]["estimate"] and 0 < est["memory_gib"]["band"] < 1
        assert s.plan("energy")["record"]["calctype"] == "energy"
        f = s.compute()
        # the process's peak during the request (this test process holds the earlier tests' too)
        assert f.execution.threads == 2 and f.execution.peak_memory_gib > 0
    # the default calctypes are the method's: RIJCOSX has no analytic Hessian
    with linus.Session(inp(jk="rijcosx")) as s:
        assert s.calctypes == {"energy", "gradient"}
        with pytest.raises(linus.InputError):
            s.plan("hessian")
    caps = program.capabilities()
    assert caps["sessions"] and caps["methods"]["b3lyp"] == ["energy", "gradient", "hessian"]
    assert ENTRY_POINT_GROUP == "qccompute.programs"
    assert program.settings_schema("gradient")["type"] == "object"
    assert "warm_start" in program.record_schema("gradient")["properties"]
    assert program.compute(inp()).success
    assert program.numerics() == program.numerics(inp()) == str(linus.NUMERICS_EPOCH)


def test_qccompute_conformance():
    testing = pytest.importorskip("qccompute.testing")
    from linus.qc import program

    ran = testing.check_program(program, [inp(), inp(calctype="energy"), inp("b3lyp", calctype="hessian")])
    assert any("deferred" in c for c in ran) and any("session" in c for c in ran)
