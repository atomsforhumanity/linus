"""SCF starts from earlier orbitals: a session's lineages, given orbitals (`ProgramInput.given`),
and the drivers that open sessions (qccompute, qcopt)."""

import linus
import numpy as np
import pytest
from helpers import failure, moved, program_input
from qcdata import CalcType, Model, ProgramInput, ProgramSpec

ARGS = dict(method="b3lyp", jk="df")


@pytest.fixture(scope="module")
def displaced(water):
    rng = np.random.default_rng(3)
    return moved(water, water.geometry + rng.normal(0.0, 0.05, water.geometry.shape))


def gradient_input(structure, **keywords):
    return program_input(structure, "gradient", **{**ARGS, **keywords})


@pytest.fixture(scope="module")
def start(water):
    out = linus.compute(gradient_input(water, output={"orbitals": True}))
    assert out.success
    return out


def test_given_orbitals_of_a_nearby_geometry_converge_faster_to_the_same_energy(displaced, start):
    sad = linus.compute(gradient_input(displaced))
    warm = linus.compute(gradient_input(displaced).model_copy(update={"given": start.results}))
    assert warm.success and warm.record["scf"]["guess"]["source"] == "given"
    assert warm.results.energy == pytest.approx(sad.results.energy, abs=1e-8)
    np.testing.assert_allclose(warm.results.gradient, sad.results.gradient, atol=1e-6)
    assert warm.results.scf_iterations < sad.results.scf_iterations
    # the session's default lineage starts from them once, then continues itself
    with linus.Session(gradient_input(displaced).model_copy(update={"given": start.results})) as s:
        first = s.compute()
        assert first.record["scf"]["guess"]["source"] == "given" and first.results.scf_iterations == \
            warm.results.scf_iterations
        assert s.compute(displaced.geometry + 0.01).record["scf"]["guess"]["source"] == "lineage"


def test_a_lineage_starts_from_its_projected_last_orbitals(water, displaced):
    sad = linus.compute(gradient_input(displaced))
    with linus.Session(gradient_input(water)) as s:
        s.compute()
        r = s.compute(displaced.geometry)
        assert r.record["scf"]["guess"]["orthonormalized"] and r.record["warm_start"]
        assert r.results.energy == pytest.approx(sad.results.energy, abs=1e-8)
        assert r.results.scf_iterations < sad.results.scf_iterations
        # reset forgets the orbitals
        s.reset()
        assert not s.compute(water.geometry).record["warm_start"]


def test_given_orbitals_of_another_molecule_or_basis_are_refused(water, molecules, start):
    for inp in (gradient_input(water, basis="def2-tzvp"),
                gradient_input(water.model_copy(update={"charge": 1, "multiplicity": 2})),
                gradient_input(molecules("nh3"))):
        out = failure(inp.model_copy(update={"given": start.results}))
        assert out.error.kind == "input" and out.error.code == "not_applicable", inp


def test_open_shell_given_orbitals(water, displaced):
    cation = {"charge": 1, "multiplicity": 2}
    a = linus.compute(program_input(water.model_copy(update=cation), output={"orbitals": True}, **ARGS))
    sad = linus.compute(program_input(displaced.model_copy(update=cation), **ARGS))
    warm = linus.compute(program_input(displaced.model_copy(update=cation), **ARGS).model_copy(
        update={"given": a.results}))
    assert warm.results.energy == pytest.approx(sad.results.energy, abs=1e-8)
    assert warm.results.scf_iterations < sad.results.scf_iterations


def test_files_are_refused(water):
    """linus.compute takes no files: earlier orbitals come in `given`."""
    for files in ({"other.txt": "x"}, {"linus.orbitals.npz": b"x"}):
        out = failure(gradient_input(water).model_copy(update={"files": files}))
        assert out.error.kind == "input" and out.error.code == "not_applicable"


def test_qccompute_runs_linus_natively(water):
    """linus is a qccompute.programs entry point: qccompute.compute runs linus.compute in this
    process, with the record and the typed error, and qccompute.session opens a linus session."""
    qccompute = pytest.importorskip("qccompute")
    from qccompute.program import find_program

    program = find_program("linus")
    if program is None:
        pytest.skip("linus is not installed as a qccompute.programs entry point (an in-tree run without the wheel)")
    inp = program_input(water, "gradient", basis="sto-3g")
    out = qccompute.compute(inp)
    assert out.success and out.record["scf"]["guess"]["source"] == "sad"
    assert out.results.energy == linus.compute(inp).results.energy
    bad = qccompute.compute(program_input(water, basis="sto-3g", scf={"max_iter": 2, "escalate": False}),
                            raise_exc=False)
    assert not bad.success and bad.error.kind == "convergence"
    with qccompute.session(inp) as s:
        a = s.compute()
        b = s.compute(water.geometry + 0.01)
    assert b.record["warm_start"] and b.results.scf_iterations < a.results.scf_iterations


def test_geometric_optimization_through_the_codec(water):
    qccompute = pytest.importorskip("qccompute")
    pytest.importorskip("geometric")
    child = ProgramSpec(program="linus", calctype=CalcType.gradient, model=Model(method="hf", basis="sto-3g"))
    inp = ProgramInput(program="geometric", calctype=CalcType.optimization, structure=water, subprograms=[child])
    out = qccompute.compute(inp)
    assert out.success and len(out.results.frames) > 2
    assert out.results.frames[-1].results.energy < out.results.frames[0].results.energy


def test_qcopt_opens_a_session(water):
    """qcopt 0.3 opens a linus session per task: every step after the first starts from the
    previous step's orbitals (no evaluator of linus's own)."""
    qcopt = pytest.importorskip("qcopt")
    if not hasattr(qcopt, "compute"):
        pytest.skip("needs qcopt 0.3 (qcopt.compute)")
    sub = [{"program": "linus", "calctype": "gradient", "model": {"method": "hf", "basis": "sto-3g"}}]
    out = qcopt.compute(ProgramInput(program="qcopt", calctype="optimization", structure=water, subprograms=sub))
    assert out.success
    frames = out.results.frames
    assert frames[0].record["warm_start"] is False
    assert all(f.record["warm_start"] for f in frames[1:])
    ref = linus.compute(ProgramInput(program="linus", calctype=CalcType.energy, structure=frames[-1].structure,
                                     model=Model(method="hf", basis="sto-3g")))
    assert frames[-1].results.energy == pytest.approx(ref.results.energy, abs=1e-8)


def test_probes_run_once_per_lineage_and_on_a_gap_drop():
    """A stretched H2 (B3LYP) along one lineage: the default probes run at its first SCF only;
    the steps that continue it are skipped (`later_geometry`) until the gap falls more than
    0.02 Eh below the first probe's."""
    import qcdata

    bohr = 1 / 0.529177210544
    h2 = qcdata.Structure(symbols=["H", "H"], geometry=[[0.0, 0, 0], [0, 0, 2.0 * bohr]])
    rec = []
    with linus.Session(program_input(h2, "gradient", **ARGS)) as s:
        for r in (2.0, 2.02, 2.04, 2.45):
            f = s.compute(np.array([[0.0, 0, 0], [0, 0, r * bohr]]))
            scf = f.record["scf"]
            rec.append((scf["guess"]["source"], scf["closed_shell_probe"], scf.get("probe_trigger"),
                        [w.code for w in f.warnings]))
    (g1, p1, t1, w1), (g2, p2, t2, w2), (g3, p3, t3, w3), (g4, p4, t4, w4) = rec
    assert g1 == "sad" and p1["unstable"] and t1 is None
    for guess, p, t in ((g2, p2, t2), (g3, p3, t3)):
        assert guess == "lineage" and p["skipped"] == "later_geometry" and "eigenvalue" not in p
        assert t["last_gap"] == p1["gap"] and "trigger" not in t
    assert g4 == "lineage" and p4["unstable"] and p4["trigger"] == "gap_dropped"
    assert t4["trigger"] == "gap_dropped" and t4["gap"] < p1["gap"] - 0.02
    # the closed-shell probe's warning comes with each probe that finds the instability
    assert "closed_shell_unstable" in w1 and "closed_shell_unstable" in w4 and not (set(w2) | set(w3)) & set(w1)


def test_dispersion_named_after_the_functional(water):
    """qcdata/Psi4 write B3LYP-D3(BJ) as "b3lyp-d3bj"; linus's own wb97x-d3bj stays whole."""
    inp = ProgramInput(program="linus", calctype=CalcType.energy, structure=water,
                       model=Model(method="b3lyp-d3bj", basis="def2-svp"), keywords={"jk": "df"})
    out = linus.compute(inp)
    ref = linus.compute(program_input(water, method="b3lyp", dispersion="d3bj", jk="df"))
    assert out.results.energy == ref.results.energy
    assert out.results.energy_components["dispersion"] < 0
    assert out.resolved.settings["dispersion"] == "d3bj"
    assert linus.normalize_options({"method": "wb97x-d3bj"}) == {"method": "wb97x-d3bj"}
    assert linus.normalize_options({"method": "PBE0-D3(BJ)"})["dispersion"] == "d3bj"
    clash = failure(inp.model_copy(update={"keywords": {"jk": "df", "dispersion": "d4"}}))
    assert clash.error.kind == "input" and "dispersion" in clash.error.message
    # every setting of the tree is a keyword; a one-off is strict (D41): grid.response is a
    # gradient's and a Hessian's setting
    plain = Model(method="b3lyp", basis="def2-svp")
    kw = {"jk": "df", "dispersion": "d3bj", "grid": {"response": False}}
    out = linus.compute(inp.model_copy(update={"model": plain, "keywords": kw, "calctype": CalcType.gradient}))
    assert out.results.energy_components["dispersion"] < 0
    assert out.resolved.settings["grid"]["response"] is False
    energy = failure(inp.model_copy(update={"model": plain, "keywords": kw}))
    assert energy.error.code == "not_applicable"
