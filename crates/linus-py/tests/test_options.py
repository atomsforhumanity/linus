"""The settings grammar, typed errors and the result report shared with the CLI: options
(memory_gib, jk, response), the `resolved` block, error types and dispersion suffixes."""

import json
import os
import subprocess

import linus
import numpy as np
import pytest
from helpers import program_input, run


def test_error_types_keep_the_builtin_bases(water):
    assert issubclass(linus.InputError, linus.LinusError) and issubclass(linus.InputError, ValueError)
    assert issubclass(linus.UnsupportedError, NotImplementedError) and issubclass(linus.ResourceError, MemoryError)
    assert issubclass(linus.ConvergenceError, linus.LinusError) and issubclass(linus.LinusError, RuntimeError)
    def open_(**kw):
        return linus.Session(program_input(water, **kw))

    with pytest.raises(linus.InputError, match="Did you mean 'def2-svp'"):
        open_(basis="def2svp")
    with pytest.raises(linus.InputError, match="dispersion"):
        open_(method="b3lpy")
    # wb97x-d3 needs D3 with zero damping: rejected before any SCF, with the alternatives
    with pytest.raises(linus.UnsupportedError, match="wb97x-d3bj"):
        open_(method="wb97x-d3")
    with pytest.raises(linus.InputError, match="1-4"):
        open_(method="pbe", grid="linus:9")
    with pytest.raises(linus.InputError, match="COSX"):
        open_(method="pbe", grid="cosx:2")
    with pytest.raises(linus.InputError, match="df_storage"):
        open_(jk={"kind": "df", "df_storage": "f16"})
    with pytest.raises(linus.UnsupportedError, match="rij_snk"):
        run(water, "hessian", method="pbe", basis="sto-3g", jk="rij_snk")
    with pytest.raises(linus.InputError, match="Did you mean 'water'"):
        run(water, solvent="watr")


def test_unconverged_scf_carries_its_frame(water):
    with linus.Session(program_input(water, basis="sto-3g", scf={"max_iter": 3, "escalate": False})) as s:
        with pytest.raises(linus.ConvergenceError, match="scf.soscf") as e:
            s.compute()
        f = e.value.frame
        assert not f.success and f.results.scf_iterations == 3 and f.results.energy is not None
        assert len(f.record["scf"]["history"]) == 3 and "e_tol" in str(e.value)
        # a failed SCF leaves no state: the lineage starts from the configured guess again
        assert s.lineages() == []


def test_dispersion_suffixes_and_normalized_options(water):
    assert linus.normalize_options({"method": "B3LYP-D3(BJ)"})["dispersion"] == "d3bj"
    # bare d3 names two models (Psi4/qcdata: zero damping; ORCA: Becke-Johnson)
    for kw in ({"method": "b3lyp-d3"}, {"method": "b3lyp", "dispersion": "d3"}):
        with pytest.raises(linus.InputError, match="'d3' is ambiguous"):
            linus.normalize_options(kw)
    assert linus.normalize_options({"method": "wb97x-d3bj"})["method"] == "wb97x-d3bj"
    o = linus.normalize_options({"jk": {"kind": "rijcosx", "aux": "def2-universal-jkfit", "k_grid": "cosx:2"},
                                 "grid": {"xc": "pyscf:2"}, "solvent": {"name": "water", "model": "IEFPCM"}})
    assert o["jk"] == {"kind": "rij_snk", "aux": "def2-universal-jkfit", "k_grid": "cosx:2"}
    assert o["grid"] == {"xc": "pyscf:2"} and o["solvent"] == {"name": "water", "model": "iefpcm"}
    a = linus.compute(program_input(water, method="b3lyp-d3bj", basis="sto-3g"))
    b = linus.compute(program_input(water, method="b3lyp", dispersion="d3bj", basis="sto-3g"))
    assert a.results.energy == b.results.energy and a.results.energy_components["dispersion"] < 0
    assert a.resolved.settings["dispersion"] == "d3bj"
    assert "b3lyp" in linus.available_functionals() and any(s[0].lower() == "water" for s in linus.available_solvents())


def test_solvent_table_fallback(cli, water, tmp_path):
    """linus ships the SMD solvents and common liquids; the other CRC liquids (morpholine) resolve
    through qcref's full table, found at import, or the CLI's $LINUS_SOLVENT_TABLE."""
    import importlib.util

    if importlib.util.find_spec("qcref") is None:
        pytest.skip("qcref is not installed")
    table = linus.solvent_table()
    assert table is not None and table.name == "crc.csv" and table.parent.name == "solvents"
    assert not any(s[0].lower() == "morpholine" for s in linus.available_solvents())
    rec = run(water, basis="sto-3g", solvent="morpholine").record
    assert rec["solvent"]["eps"] == 7.42 and rec["solvent"]["eps_table"] == str(table)
    assert "eps_table" not in run(water, basis="sto-3g", solvent="water").record["solvent"]
    with pytest.raises(linus.InputError, match="not in the CRC table"):
        linus.Session(program_input(water, basis="sto-3g", solvent="no such liquid"))
    req = json.dumps({"structure": {"xyz": "3\n\nO 0 0 0.1173\nH 0 0.7572 -0.4692\nH 0 -0.7572 -0.4692\n"},
                      "basis": "sto-3g", "solvent": "morpholine"})
    env = {k: v for k, v in os.environ.items() if k != "LINUS_SOLVENT_TABLE"}
    p = subprocess.run([cli, "--json"], input=req, capture_output=True, text=True, env=env)
    assert p.returncode == 2 and "LINUS_SOLVENT_TABLE" in json.loads(p.stdout)["error"]
    p = subprocess.run([cli, "--json"], input=req, capture_output=True, text=True, env={**env, "LINUS_SOLVENT_TABLE": str(table)})
    res = json.loads(p.stdout)["resolved"]["record"]
    assert p.returncode == 0 and res["solvent"]["eps"] == 7.42 and res["solvent"]["eps_table"] == str(table)
    assert res["debug_env"]["LINUS_SOLVENT_TABLE"] == str(table)


def test_report_fields_resolved_and_resources(water):
    out = linus.compute(program_input(water, "gradient", method="pbe0", jk="df", memory_gib=4, threads=2))
    r, rec = out.results, out.record
    assert out.success and r.gradient.shape == (3, 3) and rec["purpose"] == "gradient"
    np.testing.assert_allclose(np.sum([np.asarray(g) for g in r.extras["gradient_components"].values()], axis=0),
                               r.gradient, atol=1e-12)
    assert rec["scf"]["fock_builds"] >= r.scf_iterations and len(rec["scf"]["history"]) == r.scf_iterations
    assert rec["scf"]["convergence"]["g_tol"] == 1e-7  # the gradient's default
    assert r.wavefunction.orbitals.core_electrons is None  # no ECP
    st = out.resolved.settings
    assert st["jk"]["aux"] == "def2-universal-jkfit" and st["grid"]["xc"] == "pyscf:3" and st["memory_gib"] == 4
    assert st["jk"]["df_storage"] == "f64" and rec["df"]["precision"] == "f64" and rec["memory"]["source"] == "memory_gib"
    assert rec["memory"]["df_tensor_budget_gib"] == pytest.approx(3.0)
    # timings are seconds only; total is the whole call; sizes in the record
    t = out.execution.timings
    assert "df_store_gib" not in t and rec["sizes"]["df_store_gib"] > 0 and rec["sizes"]["nbf"] == 24
    assert t["total"] >= t["energy"] > 0
    h = linus.compute(program_input(water, "hessian", method="b3lyp", basis="sto-3g", jk="df", response={"tol": 1e-8}))
    st = h.resolved.settings
    assert h.record["cphf_iterations"] > 0 and st["response"]["tol"] == 1e-8
    assert st["jk"]["aux"] == "cc-pvtz-jkfit|def2-universal-jkfit" and st["grid"]["kernel"] == "linus:1"
    assert h.record["response"]["single_precision"] is False and st["precision"]["response"] == "fp64"
    assert h.record["numerics_epoch"] == linus.NUMERICS_EPOCH and h.record["homo_lumo_gap"] > 0


def test_closed_shell_probe_record(water):
    """The closed-shell probe (D36): water's gap is above the gate; a stretched H2 (2 Å) is flagged
    with one warning (`closed_shell_unstable`), its result unchanged; Hartree-Fock is opt-in."""
    from qcdata import Structure

    p = run(water, method="b3lyp", jk="df").record["scf"]["closed_shell_probe"]
    assert p["skipped"] == "gap_above_threshold" and p["gap"] > 0.15 and p["threshold"] == 0.15
    h2 = Structure(symbols=["H", "H"], geometry=[[0, 0, 0], [0, 0, 2.0 / 0.529177210544]])
    on = run(h2, method="b3lyp", jk="df")
    flagged = [w for w in on.warnings if w.code == "closed_shell_unstable"]
    assert len(flagged) == 1 and "RKS is unstable towards a broken-symmetry UKS solution" in flagged[0].message
    p = on.record["scf"]["closed_shell_probe"]
    assert p["unstable"] and p["kind"] == "external" and p["eigenvalue"] < -1e-2 and p["products"] == 2
    off = run(h2, method="b3lyp", jk="df", scf={"stability_closed_probe": 0})
    assert off.results.energy == on.results.energy and off.record["scf"]["closed_shell_probe"]["skipped"] == "off"
    hf = run(h2, jk="df")
    assert hf.record["scf"]["closed_shell_probe"]["skipped"] == "hf_default"


def test_non_aufbau_state_after_the_escalation():
    """C2 singlet (1.243 Å) PBE: the escalation converges to a state with 3 sigma_g empty below the occupied pi
    pair (docs/internal/deggrad.md). The gradient is that state's (ORCA 6.0.0: -0.038282036; before the fix
    -0.1667, the aufbau determinant's), recorded as non-aufbau with one warning."""
    from qcdata import Structure

    c2 = Structure(symbols=["C", "C"], geometry=[[0, 0, 0], [0, 0, 1.243 / 0.529177210544]])
    f = run(c2, "gradient", method="pbe", jk="df", grid={"xc": "orca:3"}, scf={"convergence": "extreme", "max_iter": 300})
    assert f.results.gradient[1][2] == pytest.approx(-0.038282036, abs=1e-6)
    na = f.record["scf"]["non_aufbau"]
    assert na == [{"spin": "restricted", "holes": [4], "particles": [6], "inversion": pytest.approx(3.687e-4, abs=1e-6)}]
    assert [w.code for w in f.warnings].count("non_aufbau") == 1 and f.record["homo_lumo_gap"] < 0
    assert f.record["scf"]["closed_shell_probe"]["eigenvalue"] > -1


def test_response_and_df_options_keep_defaults(water):
    """Explicit default settings give the same bits as unset ones; single precision moves the
    Hessian by round-off only."""
    kw = dict(method="b3lyp", basis="sto-3g", threads=2)
    a = run(water, "hessian", jk="df", **kw).results
    b = run(water, "hessian", jk={"kind": "df", "df_storage": "f64"}, response={"iteration_aux": "auto"},
            precision={"response": "auto"}, **kw).results
    np.testing.assert_array_equal(a.hessian, b.hessian)
    c = run(water, "hessian", jk="df", precision={"response": "single"}, **kw)
    assert c.record["response"]["single_precision"] is True
    assert np.abs(c.results.hessian - a.hessian).max() < 1e-6


def test_cli_set_options_and_failures(cli, water, tmp_path):
    xyz = tmp_path / "w.xyz"
    sym = list(water.symbols)
    xyz.write_text(f"{len(sym)}\n\n" + "".join(f"{s} {x * 0.529177210544:.10f} {y * 0.529177210544:.10f} {z * 0.529177210544:.10f}\n"
                                              for s, (x, y, z) in zip(sym, water.geometry)))
    opts = tmp_path / "o.json"
    opts.write_text(json.dumps({"scf": {"level_shift": 0.3}, "grid": {"xc": "linus:2"}}))
    run = lambda *a: subprocess.run([cli, str(xyz), *a], capture_output=True, text=True)  # noqa: E731
    p = run("--method", "pbe", "--basis", "sto-3g", "--options", str(opts), "--set", "scf.soscf=true", "--grad", "--json")
    r = json.loads(p.stdout)
    assert p.returncode == 0 and r["options"]["scf"]["soscf"] and r["options"]["scf"]["level_shift"] == 0.3
    assert r["grid"] == "linus:2" and r["options"]["grid"] == {"xc": "linus:2"} and r["timings"]["total"] >= r["timings"]["energy"]
    assert "structure" in r and "molecule" not in r and "resolved" in r and "df_store_gib" not in r["timings"] and r["schema_version"] == 1
    p = run("--basis", "sto-3g", "--max-iter", "2", "--set", "scf.escalate=false", "--json")
    f = json.loads(p.stdout)
    assert p.returncode == 1 and f["success"] is False and f["error_kind"] == "convergence" and len(f["scf"]["history"]) == 2
    assert f["schema_version"] == 1
    p = run("--method", "pbe", "--grid", "linus:9")
    assert p.returncode == 2 and "1-4" in p.stderr
    # errors while setting up the calculation keep their kind (not "error")
    p = run("--method", "wb97x-d3", "--json")
    f = json.loads(p.stdout)
    assert p.returncode == 1 and f["error_kind"] == "unsupported" and f["schema_version"] == 1 and not f["success"]
    p = run("--basis", "def2svp", "--json")
    f = json.loads(p.stdout)
    assert p.returncode == 2 and f["error_kind"] == "input" and "def2-svp" in f["error"]
    p = run("--method", "pbe", "--basis", "sto-3g", "--grad")
    assert p.returncode == 0 and "gradient (Eh/bohr)" in p.stdout and "settings:" in p.stdout
    out = subprocess.run([cli, "--version"], capture_output=True, text=True).stdout
    assert out.startswith(f"linus {linus.__version__} (numerics epoch {linus.NUMERICS_EPOCH})"), out
    f = json.loads(run("--method", "pbe", "--basis", "sto-3g", "--json").stdout)
    assert f["version"] == linus.__version__ and f["numerics_epoch"] == linus.NUMERICS_EPOCH


def test_plan_before_the_run(water):
    with linus.Session(program_input(water, method="b3lyp", basis="sto-3g", jk="df")) as s:
        p = s.plan("hessian")
        rec = p["record"]
        assert rec["calctype"] == "hessian"
        assert p["resolved"]["settings"]["grid"]["xc"] == "linus:3" and rec["response"]["hessian"]["precision"] == "double"
        assert rec["placement"]["jk"]["on"] == "cpu" and any(d["what"].startswith("jk.df_long_range")
                                                             for d in rec["at_run_time"])
        assert s.lineages() == []  # nothing ran
        energy_plan = s.plan("energy")
    out = linus.compute(program_input(water, method="b3lyp", basis="sto-3g", jk="df"))
    assert energy_plan["resolved"]["settings"]["scf"]["e_tol"] == out.resolved.settings["scf"]["e_tol"]
    with linus.Session(program_input(water, basis="sto-3g")) as s:
        with pytest.raises(linus.UnsupportedError, match="calctype"):
            s.plan("optimization")
        with pytest.raises(linus.InputError, match="calctype"):
            s.plan("frequencies")  # no such calculation any more: the Hessian carries the analysis
