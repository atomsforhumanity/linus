"""The Python API on qcdata objects: results, settings, warm restarts, errors, threads."""

import threading
import time
from pathlib import Path

import linus
import numpy as np
import pytest
import qcdata
from helpers import moved, program_input, run


@pytest.fixture(scope="module")
def b3lyp(water):
    return run(water, method="b3lyp", threads=2, outputs=("hamiltonian", "core_hamiltonian"))


def test_result_is_consistent(b3lyp):
    f = b3lyp
    r = f.results
    orb = r.wavefunction.orbitals
    assert f.success and orb.spin == "restricted" and r.scf_iterations > 1
    assert (r.calcinfo_nalpha, r.calcinfo_nbeta, r.calcinfo_nbasis, r.calcinfo_nmo) == (5, 5, 24, 24)
    assert sum(r.energy_components.values()) == pytest.approx(r.energy, abs=1e-10)
    assert set(r.energy_components) == {"nuclear_repulsion", "one_electron", "coulomb", "exchange", "xc", "dispersion"}
    assert r.energy_components["dispersion"] == 0.0
    assert r.energy_components["exchange"] < 0 and r.energy_components["xc"] < 0
    # the wavefunction (decision 25): canonical restricted orbitals, occupations per spin, no
    # density stored; the Hamiltonian of the SCF's last (full) build has them as its fresh
    # generalized eigenvectors in the returned overlap, to rounding
    wf = r.wavefunction
    c, occ = orb.coefficients_a, orb.occupations_a
    assert c.shape == (24, 24) and orb.kind == "canonical" and orb.energies_b is None and orb.core_electrons is None
    assert orb.coefficients_b is None and wf.hamiltonian_b is None and wf.basis.nao == 24
    np.testing.assert_array_equal(occ, [1.0] * 5 + [0.0] * 19)
    np.testing.assert_array_equal(orb.occupations_b, occ)
    assert np.all(np.diff(orb.energies_a) >= 0)
    h, s = np.asarray(wf.hamiltonian_a), np.asarray(wf.overlap)
    assert np.abs(h @ c - s @ c * orb.energies_a).max() < 1e-12
    assert np.abs(c.T @ s @ c - np.eye(24)).max() < 1e-12
    # the core Hamiltonian's trace with C n C^T is the energy's one-electron term to the SCF's
    # residual: the energy is of the last build's density, one step behind the returned orbitals
    da, db = orb.density()
    assert np.sum((da + db) * np.asarray(wf.core_hamiltonian)) == pytest.approx(r.energy_components["one_electron"],
                                                                                 abs=1e-5)
    t = f.execution.timings
    assert t["total"] > 0 and "scf_total" in t and "xc" in t and f.execution.wall_time >= t["total"]
    assert np.linalg.norm(r.dipole_moment) == pytest.approx(0.78, abs=0.02)
    # a frame is JSON, and back
    back = qcdata.Frame.model_validate_json(f.model_dump_json())
    assert back.results.energy == r.energy and back.record == f.record


def test_settings_and_defaults(water):
    out = linus.compute(program_input(water, method="b3lyp", threads=2))
    assert out.success and out.warnings == [] and out.resolved.settings["grid"]["xc"] == "pyscf:3"
    assert out.resolved.settings["threads"] == 2 and "method" not in out.resolved.settings
    d = linus.default_options()
    # every setting at its default, "auto" where the calculation decides (tight thresholds for energies)
    assert d["scf"]["convergence"] == "auto" and d["scf"]["e_tol"] == "auto" and d["jk"]["kind"] == "df"
    assert d["scf"]["guess"] == "sad" and d["scf"]["max_iter"] == 288 and d["scf"]["soscf"] is False
    # DFT Hessians: first-order grid response by default (fixed grid only on request); one setting for it
    assert d["grid"] == {"xc": "auto", "nlc": "auto", "kernel": "auto", "response": True}
    assert "grid_response" not in d and "exact" not in d and "df" not in d
    assert d["jk"]["exact_storage"] == "auto" and d["jk"]["df_storage"] == "auto" and d["reference"] == "auto"
    assert d["backend"]["kind"] == "cpu" and d["precision"]["fp64"] == "auto" and d["backend"]["scf_loop"] == "auto"
    assert d["dispersion"] == "auto" and d["basis_dir"] == [] and d["memory_gib"] == "auto"
    assert d["output"] == {"core_hamiltonian": False, "hamiltonian": False, "orbitals": False, "overlap": False,
                           "verbose": False, "molden": "off"}
    assert linus.capabilities() == ["energy", "gradient", "hessian"]
    assert "def2-svp" in linus.available_basis_sets()
    # the basis set search path ends with the source tree (development builds) and the package's sets
    path = linus.basis_search_path("/nowhere")
    assert path[0] == Path("/nowhere") and linus.basis_file("def2-svp").is_file()
    assert linus.basis_file("def2-svp", "/nowhere") == linus.basis_file("def2-svp")
    assert linus.basis_file("no-such-basis") is None
    with linus.Session(program_input(water, method="b3lyp")) as s:
        assert repr(s) == "linus.Session(method='b3lyp', basis='def2-svp', atoms=3, calctypes=['energy', " \
                          "'gradient', 'hessian'])"
    assert repr(s).endswith(", closed)")


def test_basis_file_names():
    """Basis set names with ( ) , map to file names a wheel can hold, and back."""
    from linus._library import PACKAGE_BASIS_DIR

    f = linus.basis_file("6-311+G(2d,p)", PACKAGE_BASIS_DIR)
    assert f is not None and f.name == "6-311_pl_g_lp_2d_cm_p_rp_.json" and f.parent == PACKAGE_BASIS_DIR
    assert "6-311+g(2d,p)" in linus.available_basis_sets()
    assert "6-311+g(2d,p)" in linus.available_basis_sets(PACKAGE_BASIS_DIR)


def test_groups_shorthands_and_unknown_keys(water):
    n = linus.normalize_options
    assert n({"jk": "df"}) == {"jk": {"kind": "df"}}
    assert n({"jk": {"kind": "df", "aux": "cc-pvtz-jkfit"}}) == {"jk": {"kind": "df", "aux": "cc-pvtz-jkfit"}}
    assert n({"jk": {"kind": "rij_snk", "k_grid": "cosx:2"}})["jk"]["k_grid"] == "cosx:2"
    assert n({"grid": {"xc": "linus:3", "nlc": "linus:1"}}) == {"grid": {"xc": "linus:3", "nlc": "linus:1"}}
    assert n({"solvent": "water"}) == {"solvent": {"name": "water"}}
    assert n({"solvent": 7.4}) == {"solvent": {"eps": 7.4}}
    assert n({"solvent": {"name": "water", "model": "smd"}}) == {"solvent": {"name": "water", "model": "smd"}}
    assert n({"reference": "unrestricted", "cartesian": True, "memory_gib": 4}) == {
        "reference": "unrestricted", "cartesian": True, "memory_gib": 4}
    assert n({"method": "b3lyp-d3bj"}) == {"method": "b3lyp", "dispersion": "d3bj"}
    # the removed names are unknown keys, with a suggestion where there is a near one
    for old in ("aux", "k_grid", "pcm", "restricted", "pure", "memory_gb", "grid_response", "nlc_grid"):
        with pytest.raises(linus.InputError, match=old) as e:
            linus.Session(program_input(water, **{old: 1}))
        assert e.value.kind == "input" and e.value.code == "unknown_key"
    with pytest.raises(linus.InputError, match="Did you mean 'scf.e_tol'"):
        linus.Session(program_input(water, scf={"e_conv": 1e-6}))
    with pytest.raises(linus.InputError, match="Did you mean 'memory_gib'"):
        linus.Session(program_input(water, memory_gib_="4"))
    # the model carries the method and basis set
    with pytest.raises(linus.InputError, match="model"):
        linus.Session(program_input(water).model_copy(update={"keywords": {"method": "hf"}}))
    # a setting that cannot take effect is an error (D41)
    with pytest.raises(linus.InputError, match="grid"):
        linus.Session(program_input(water, grid="linus:3"))
    with pytest.raises(linus.InputError, match="screening"):
        linus.Session(program_input(water, method="pbe", jk={"kind": "df", "screening": 1e-10}))


def test_settings_schema_validates_requests():
    jsonschema = pytest.importorskip("jsonschema")
    schema = linus.settings_schema()
    jsonschema.Draft202012Validator.check_schema(schema)
    props = schema["properties"]
    for key in ("method", "basis", "jk", "grid", "scf", "response", "solvent", "reference", "cartesian", "memory_gib",
                "output", "threads"):
        assert key in props, key
    validator = jsonschema.Draft202012Validator(schema)
    for ok in ({"method": "b3lyp", "basis": "def2-svp", "jk": {"kind": "df", "aux": "def2-universal-jkfit"}},
               {"scf": {"e_tol": 1e-9, "stability": "always"}, "grid": {"xc": "linus:3"}, "reference": "unrestricted"},
               {"solvent": {"name": "water", "model": "smd"}, "output": {"verbose": True}}):
        assert not list(validator.iter_errors(ok)), ok
        linus.normalize_options(ok)  # the library agrees with the schema
    assert list(validator.iter_errors({"reference": "sideways"}))
    assert list(validator.iter_errors({"jk": {"kind": "df", "no_such_key": 1}}))
    gradient = linus.settings_schema("gradient")
    assert set(gradient["properties"]) <= set(props)


def test_settings_stubs_import():
    from linus.settings import Settings

    assert "jk" in Settings.__annotations__ and "scf" in Settings.__annotations__


@pytest.mark.parametrize("jk", ["incore", "df", "rij_snk"])
def test_jk_methods_close_to_exact(water, jk):
    exact = run(water, method="pbe0", jk="exact", threads=4).results.energy
    e = run(water, method="pbe0", jk=jk, threads=4).results.energy
    assert e == pytest.approx(exact, abs={"incore": 1e-9, "df": 1e-4, "rij_snk": 2e-3}[jk])


def test_warm_restart_and_reset(water):
    with linus.Session(program_input(water), calctypes=["energy"]) as s:
        first = s.compute()
        shifted = s.compute(water.geometry + np.array([0.0, 0.0, 0.02]))
        assert shifted.record["warm_start"] and shifted.results.scf_iterations < first.results.scf_iterations
        assert shifted.structure.geometry[0, 2] == pytest.approx(water.geometry[0, 2] + 0.02)
        s.reset()
        fresh = s.compute(water.geometry + np.array([0.0, 0.0, 0.02]))
        assert not fresh.record["warm_start"]
    assert fresh.results.energy == pytest.approx(shifted.results.energy, abs=1e-9)
    # a pure translation leaves the energy unchanged
    assert fresh.results.energy == pytest.approx(first.results.energy, abs=1e-9)


def test_unrestricted_closed_shell_equals_restricted(water):
    r = run(water, method="pbe", outputs=("orbitals",)).results
    u = run(water, method="pbe", reference="unrestricted", outputs=("orbitals",)).results
    assert u.wavefunction.orbitals.spin == "unrestricted" and u.energy == pytest.approx(r.energy, abs=1e-8)


def test_errors_are_clear_python_exceptions(water):
    with pytest.raises(ValueError, match="e_convv: not a setting"):
        linus.Session(program_input(water, e_convv=1e-6))
    with pytest.raises(ValueError, match="unknown basis set 'nope'"):
        linus.Session(program_input(water, basis="nope"))
    with pytest.raises(ValueError, match="cannot have multiplicity 2"):
        linus.Session(program_input(water.model_copy(update={"multiplicity": 2})))
    # RIJ + COSX: energies and gradients, but no analytic Hessians
    with pytest.raises(linus.UnsupportedError, match="rij_snk"):
        run(water, "hessian", basis="sto-3g", jk="rij_snk")
    with pytest.raises(ValueError, match="unknown dispersion"):
        linus.Session(program_input(water, method="b3lyp", dispersion="d5"))
    with pytest.raises(linus.UnsupportedError, match="ROHF"):
        linus.Session(program_input(water.model_copy(update={"charge": 1, "multiplicity": 2}), reference="restricted"))
    with pytest.raises(ValueError, match="unknown"):
        linus.Session(program_input(water, method="not-a-functional"))
    with pytest.raises(ValueError, match="guess"):
        linus.Session(program_input(water, scf={"guess": "huckel"}))
    radon = qcdata.Structure(symbols=["Rn"], geometry=np.zeros((1, 3)))
    with pytest.raises(ValueError, match="no functions for element"):
        linus.Session(program_input(radon, basis="cc-pvdz"))
    with linus.Session(program_input(water, scf={"max_iter": 3, "escalate": False})) as s:
        with pytest.raises(linus.ConvergenceError, match="did not converge in 3 iterations"):
            s.compute()
        with pytest.raises(ValueError, match="shape"):
            s.compute(np.zeros((2, 3)))
    assert issubclass(linus.ConvergenceError, linus.LinusError) and issubclass(linus.LinusError, RuntimeError)
    assert issubclass(linus.LinusError, qcdata.ProgramError)


def test_one_session_one_thread(water):
    """A second thread's call on a busy session is a clear linus error, not pyo3's RuntimeError."""
    s = linus.Session(program_input(water, "hessian", method="b3lyp", basis="def2-tzvp", threads=1))
    done = []
    t = threading.Thread(target=lambda: done.append(s.compute()))
    t.start()
    time.sleep(0.3)
    with pytest.raises(linus.InputError, match="another thread") as e:
        s.compute(calctype="energy")
    assert e.value.kind == "input"
    t.join()
    assert done and done[0].results.hessian.shape == (9, 9)
    s.close()


def test_error_kinds_and_front_validation(water):
    """One error vocabulary (the CLI's error_kind) on every exception; bad input fails before any work."""
    kinds = {}
    for name, make in [
        ("input", lambda: linus.Session(program_input(water, basis="nope"))),
        ("unsupported", lambda: run(water, "hessian", basis="sto-3g", jk="rij_snk")),
        ("convergence", lambda: run(water, basis="sto-3g", scf={"max_iter": 2, "escalate": False})),
        ("unsupported", lambda: linus.Session(program_input(water, backend="gpu"))),
    ]:
        with pytest.raises(linus.LinusError) as e:
            make()
        assert e.value.kind == name, (name, e.value)
        kinds[name] = e.value
    # a calculation's failure: the failed frame, with the SCF it reached
    conv = kinds["convergence"]
    assert conv.error.kind == "convergence" and not conv.frame.success
    assert conv.frame.record["scf"]["history"] and conv.frame.results.scf_iterations == 2
    assert issubclass(linus.NumericalError, linus.LinusError) and issubclass(linus.NumericalError, ArithmeticError)
    assert issubclass(linus.BackendError, linus.LinusError)
    # coincident atoms, at opening and when moving
    pos = np.asarray(water.geometry, dtype=float)
    with pytest.raises(linus.InputError, match="closer than"):
        linus.Session(program_input(moved(water, np.array([pos[0], pos[0], pos[2]])), basis="sto-3g"))
    with linus.Session(program_input(water, basis="sto-3g")) as s:
        with pytest.raises(linus.InputError, match="closer than"):
            s.compute(np.array([pos[0], pos[0] + 1e-3, pos[2]]))
    # more electrons per spin than basis functions
    hydride = qcdata.Structure(symbols=["H"], geometry=np.zeros((1, 3)), charge=-3, multiplicity=1)
    with pytest.raises(linus.InputError, match="orbitals"):
        linus.Session(program_input(hydride, basis="sto-3g"))
    with pytest.raises(linus.InputError, match="threads"):
        linus.Session(program_input(water, threads=0))
    # freq settings are checked before the Hessian
    with pytest.raises(linus.InputError, match="temperature") as e:
        run(water, "hessian", basis="sto-3g", freq={"temperature": -5})
    assert e.value.frame.results.hessian is None
    # harmonic analysis: panics contained, non-finite dipole derivatives refused
    with pytest.raises(linus.InputError, match="finite"):
        linus.vib.harmonic_analysis([1, 1], [[0, 0, 0], [0, 0, 1.4]], np.eye(6),
                                    dipole_derivatives=np.full((6, 3), np.nan))


def test_gradient_matches_finite_differences_and_hessian_is_wired(water):
    # one grid for the gradient and the Hessian (the default grid depends on the calculation), and
    # exact J/K (the default RI-JK's auxiliary basis depends on it too)
    kw = dict(method="b3lyp", basis="sto-3g", jk="exact", threads=4, scf={"e_tol": 1e-11, "g_tol": 1e-9})
    with linus.Session(program_input(water, "hessian", grid=3, **kw)) as s:
        g = s.compute(calctype="gradient").results.gradient
        hs = s.compute().results
        e_only = s.compute(calctype="energy").results
    assert g.shape == (3, 3)
    assert abs(g.sum(axis=0)).max() < 1e-6  # translational invariance (grid response on)
    x0 = np.asarray(water.geometry, dtype=float)
    h = 1e-4
    for atom, k in [(0, 2), (1, 1)]:
        e = []
        for sgn in (1.0, -1.0):
            x = x0.copy()
            x[atom, k] += sgn * h
            e.append(run(moved(water, x), **kw).results.energy)
        assert abs((e[0] - e[1]) / (2 * h) - g[atom, k]) < 2e-8
    assert hs.hessian.shape == (9, 9) and np.abs(hs.hessian - hs.hessian.T).max() < 1e-8
    np.testing.assert_allclose(hs.gradient, g, atol=1e-7)
    assert e_only.gradient is None and e_only.hessian is None


def test_rijcosx_gradient(water):
    """jk=rij_snk: with one unfitted COSX grid the energy is variational and the analytic
    gradient (grid response on) matches finite differences; the default COSX scheme is close
    to the exact-integral gradient and translation invariant."""
    kw = dict(method="b3lyp", threads=4, scf={"e_tol": 1e-11, "g_tol": 1e-9}, grid="linus:1")
    one_grid = {"kind": "rij_snk", "k_grid": "linus:1"}
    g = run(water, "gradient", jk=one_grid, **kw).results.gradient
    assert abs(g.sum(axis=0)).max() < 1e-8
    x0 = np.asarray(water.geometry, dtype=float)
    h = 1e-3
    for atom, k in [(0, 2), (1, 1)]:
        e = []
        for step in (h, -h, 2 * h, -2 * h):
            x = x0.copy()
            x[atom, k] += step
            e.append(run(moved(water, x), jk=one_grid, **kw).results.energy)
        fd = (8 * (e[0] - e[1]) - (e[2] - e[3])) / (12 * h)
        assert abs(fd - g[atom, k]) < 3e-8
    g_cosx = run(water, "gradient", jk="rij_snk", **kw).results.gradient
    g_exact = run(water, "gradient", jk="exact", **kw).results.gradient
    assert abs(g_cosx.sum(axis=0)).max() < 1e-8
    assert abs(g_cosx - g_exact).max() < 2e-4


def test_gil_is_released_during_compute(water):
    """Python threads keep running while an SCF computes."""
    s = linus.Session(program_input(water, method="b3lyp", basis="def2-tzvp", grid=5))
    done = threading.Event()
    out = {}

    def work():
        out["f"] = s.compute()
        done.set()

    t = threading.Thread(target=work)
    t0 = time.perf_counter()
    t.start()
    ticks = 0
    while not done.is_set():
        ticks += 1
        time.sleep(0.001)
    t.join()
    elapsed = time.perf_counter() - t0
    s.close()
    assert out["f"].success
    # with the GIL held, the loop could not run until the SCF returned (a tick or two). An absolute
    # floor, not a rate: on a loaded node sleep(0.001) oversleeps by up to tens of milliseconds
    assert elapsed > 0.05, f"the SCF took {elapsed:.3f} s, too short for this test"
    assert ticks >= 5, f"{ticks} loop ticks during a {elapsed:.2f} s SCF: the GIL was held"
