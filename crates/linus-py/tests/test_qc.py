"""qcdata and qccompute: linus.compute, qccompute running linus (the qccompute.programs entry
point) with its grants, failures that raise, the CLI's JSON mode."""

import json
import os
import subprocess
import threading
import time

from conftest import MOLECULES, ROOT
import linus
import numpy as np
import pytest
from helpers import run
from qcdata import CalcType, Model, ProgramInput, ProgramOutput

qccompute = pytest.importorskip("qccompute")


def program_input(structure, calctype=CalcType.energy, method="b3lyp", basis="def2-svp", orbitals=False,
                  hamiltonian=False, **keywords):
    # the optional outputs are the settings tree's `output` group
    if orbitals or hamiltonian:
        keywords["output"] = {**({"orbitals": True} if orbitals else {}),
                              **({"hamiltonian": True} if hamiltonian else {})}
    return ProgramInput(program="linus", calctype=calctype, structure=structure,
                        model=Model(method=method, basis=basis), keywords=keywords)


def check_results(out, ref):
    """The typed fields every route fills (linus.compute, qccompute's native route, the codec)."""
    assert out.success
    res, r = out.results, ref.results
    assert res.energy == r.energy and res.nuclear_repulsion_energy == r.nuclear_repulsion_energy
    np.testing.assert_array_equal(res.dipole_moment, r.dipole_moment)
    np.testing.assert_array_equal(res.wavefunction.orbitals.energies_a, r.wavefunction.orbitals.energies_a)
    np.testing.assert_array_equal(res.wavefunction.orbitals.occupations_b, r.wavefunction.orbitals.occupations_b)
    assert (res.calcinfo_natoms, res.calcinfo_nbasis, res.calcinfo_nmo) == (3, r.calcinfo_nbasis, r.calcinfo_nmo)
    assert (res.calcinfo_nalpha, res.calcinfo_nbeta) == (r.calcinfo_nalpha, r.calcinfo_nbeta)
    assert res.energy_components == r.energy_components and res.scf_iterations == r.scf_iterations
    assert res.provenance.program == "linus" and res.provenance.program_version == linus.__version__


@pytest.fixture(scope="module")
def reference(water):
    return run(water, method="b3lyp", jk="df", outputs=("orbitals",))


def test_linus_compute(water, reference):
    inp = ProgramInput(program="linus", calctype=CalcType.energy, structure=water,
                       model=Model(method="b3lyp", basis="def2-svp"),
                       keywords={"jk": "df", "threads": 2, "output": {"orbitals": True, "molden": "spherical"}})
    out = linus.compute(inp)
    assert isinstance(out, ProgramOutput) and out.input_data == inp
    check_results(out, reference)
    res = out.results
    assert sum(res.energy_components.values()) == pytest.approx(res.energy, abs=1e-10)
    assert res.scf_iterations > 1 and out.execution.threads == 2 and out.execution.wall_time > 0
    assert out.execution.timings["total"] > 0
    assert out.resolved.settings["jk"]["kind"] == "df" and out.resolved.settings["jk"]["df_storage"] != "auto"
    assert out.record["warm_start"] is False and out.record["sizes"]["nbf"] == reference.results.calcinfo_nbasis
    orb = res.wavefunction.orbitals
    assert res.wavefunction.basis.name == "def2-svp" and orb.spin == "restricted"
    np.testing.assert_array_equal(orb.coefficients_a, reference.results.wavefunction.orbitals.coefficients_a)
    assert res.files["orbitals.molden"].startswith("[Molden Format]")
    # the whole output survives JSON
    back = ProgramOutput.model_validate_json(out.model_dump_json())
    assert back.results.energy == res.energy
    np.testing.assert_array_equal(back.results.wavefunction.orbitals.coefficients_a, orb.coefficients_a)


@pytest.fixture(autouse=True)
def registered():
    """linus as qccompute finds it: the installed entry point, or registered here when the
    package is not installed (a source tree on PYTHONPATH)."""
    from qccompute.program import find_program, register_program, unregister_program

    from linus.qc import program

    if find_program("linus") is None:
        register_program(program)
        yield
        unregister_program("linus")
    else:
        yield


def test_qccompute_runs_linus_in_process(water, reference):
    """qccompute runs linus in process: the same results, record and resolved settings; its grant
    (threads, memory) is the smaller of the keyword and the share, and a keyword above it warns."""
    inp = program_input(water, jk="df", threads=2)
    out = qccompute.compute(inp)
    check_results(out, reference)
    assert out.record["scf"]["guess"]["source"] == "sad" and out.resolved.settings["jk"]["kind"] == "df"
    assert out.results.wavefunction.orbitals.coefficients_a is None and not out.results.files
    with qccompute.Runner(threads=1) as runner:
        capped = runner.compute(inp)
    assert capped.resolved.settings["threads"] == 1 and capped.execution.threads == 1
    assert [w.code for w in capped.warnings] == ["resource_capped"] and capped.warnings[0].path == "threads"
    assert capped.resolved.settings["memory_gib"] > 0  # the memory share, not linus's own default


def test_qccompute_open_shell_and_hamiltonian(molecules):
    s = molecules("nh3", 1, 2)
    out = qccompute.compute(program_input(s, method="hf", hamiltonian=True))
    ref = run(s, outputs=("hamiltonian",)).results
    assert out.results.energy == ref.energy
    assert (out.results.calcinfo_nalpha, out.results.calcinfo_nbeta) == (5, 4)
    assert ref.wavefunction.orbitals.spin == "unrestricted"
    np.testing.assert_array_equal(out.results.wavefunction.hamiltonian_b, ref.wavefunction.hamiltonian_b)
    np.testing.assert_allclose(out.results.wavefunction.orbitals.occupations_a,
                               [1.0] * 5 + [0.0] * (ref.calcinfo_nmo - 5))


def test_grant_rules(water, monkeypatch):
    """linus.compute under a grant (decisions 10 and 16): threads, memory and the device memory
    from the grant unless the keywords ask less; no GPU granted means the CPU; the granted GPU's
    device is its CUDA ordinal in this process (``RunContext.device_ordinals``, qc* #49)."""
    from linus.session import apply_grant
    from qccompute.program import RunContext

    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

    w = []
    s = apply_grant({"threads": 8, "memory_gib": "auto"}, RunContext(threads=4, memory_gib=12.5), w)
    assert s["threads"] == 4 and s["memory_gib"] == 12.5 and [x.path for x in w] == ["threads"]
    w = []
    s = apply_grant({"backend": "gpu"}, RunContext(threads=4), w)
    assert s["backend"] == {"kind": "cpu"} and w[0].code == "resource_capped" and w[0].path == "backend.kind"
    w = []
    s = apply_grant({"backend": {"kind": "auto"}, "device_memory_gib": 40}, RunContext(threads=4, gpus=("1",),
                                                                                       device_memory_gib=24.0), w)
    assert s["backend"] == {"kind": "auto", "devices": [1]} and s["device_memory_gib"] == 24.0
    assert apply_grant({"threads": 2}, None, w) == {"threads": 2}
    # a process that sees GPUs 3 and 1 opens GPU "1" as its CUDA ordinal 1; a GPU it cannot see
    # fails the input with kind environment
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3,1")
    s = apply_grant({"backend": "gpu"}, RunContext(threads=4, gpus=("1",)), w)
    assert s["backend"] == {"kind": "gpu", "devices": [1]}
    with pytest.raises(linus.LinusError) as e:
        apply_grant({"backend": "gpu"}, RunContext(threads=4, gpus=("5",)), w)
    assert e.value.error.kind == "environment"


def test_plan_resources_and_environment(water):
    """The plan asks for no threads (the runner gives a calculation its count, qc* decision 16);
    the machine's part of the environment (CPU model, C library) is the runner's to record (qc*
    #58), linus's the device it opened (none here); with output.verbose an output carries the
    SCF's progress in logs, never on stderr (decision 24)."""
    from linus.qc import program

    plan = program.plan(program_input(water))
    assert "threads" not in plan["resources"] and plan["resources"]["gpus"] == 0
    out = linus.compute(program_input(water, output={"verbose": True}))
    assert out.execution.cpu_model is None and out.execution.libc is None and out.execution.devices == []
    ran = qccompute.compute(program_input(water))
    assert ran.execution.cpu_model and ran.execution.libc.startswith("glibc ") and ran.execution.devices == []
    assert out.execution.peak_device_memory_gib is None
    assert "iter   1" in out.logs and "E = " in out.logs
    assert linus.compute(program_input(water)).logs is None
    # under the default runner: the runner's count (qc* decision 16), at most linus's 8 without the keyword
    assert 1 <= qccompute.compute(program_input(water)).execution.threads <= min(8, len(os.sched_getaffinity(0)))


def test_plan_is_qccomputes_plan(water):
    """`linus.qc.program.plan()` validates as qccompute's `Plan` (0.23.0: `Resources` holds
    memory and GPUs, no `threads`, and refuses unknown keys), as the runner reads it
    (`plan_of`), for every calctype, with and without the `threads` keyword."""
    from linus.qc import program
    from qccompute.program import Plan, plan_of

    assert "threads" not in Plan.model_fields["resources"].annotation.model_fields
    for calctype, kw in ((CalcType.energy, {}), (CalcType.gradient, {"threads": 4, "jk": "df"}),
                         (CalcType.hessian, {"method": "hf", "memory_gib": 2.0})):
        raw = program.plan(program_input(water, calctype, **kw))
        plan = Plan.model_validate(raw)
        assert plan == plan_of(program, program_input(water, calctype, **kw))
        assert plan.resources.gpus == 0 and plan.resources.memory_gib == raw["resources"]["memory_gib"] > 0
        assert plan.resolved.settings == raw["resolved"]["settings"]


def test_progress_events(water):
    """A runner's progress listener (`ctx.progress`, qc* 14a) gets linus's documented events with
    the runner's path: each stage's start and end, every SCF iteration, every response iteration;
    the results are bitwise those of a run nobody listened to."""
    events = []
    inp = program_input(water, CalcType.hessian, method="hf", basis="sto-3g", threads=2)
    out = qccompute.compute(inp, progress=events.append)
    assert out.results.hessian is not None
    assert {tuple(e.pop("path")) for e in events} == {("linus",)}
    names = [e["event"] for e in events]
    stages = [(e["event"], e["stage"]) for e in events if e["event"].startswith("stage")]
    assert stages == [("stage_start", "scf"), ("stage_end", "scf"), ("stage_start", "gradient"),
                      ("stage_end", "gradient"), ("stage_start", "hessian"), ("stage_end", "hessian")]
    assert all(e["success"] and e["seconds"] >= 0 for e in events if e["event"] == "stage_end")
    scf = [e for e in events if e["event"] == "scf_iteration"]
    history = out.record["scf"]["history"]
    assert [e["iteration"] for e in scf] == [h["iter"] for h in history]
    assert [e["energy"] for e in scf] == [h["energy"] for h in history]
    assert scf[-1]["gradient"] == history[-1]["gmax"] and set(scf[0]) == {"event", "iteration", "energy",
                                                                           "delta_energy", "gradient"}
    response = [e for e in events if e["event"] == "response_iteration"]
    assert [e["iteration"] for e in response] == list(range(1, out.record["cphf_iterations"] + 1))
    assert all(e["solve"] == "hessian" and e["total"] == 9 for e in response) and response[-1]["unconverged"] == 0
    assert names.index("response_iteration") > names.index("stage_start") and names[-1] == "stage_end"
    quiet = linus.compute(inp)
    np.testing.assert_array_equal(quiet.results.hessian, out.results.hessian)
    # a session's requests report theirs too; json round-trips every event
    seen = []
    with qccompute.session(program_input(water, method="hf", basis="sto-3g", threads=2), progress=seen.append) as s:
        s.compute()
    assert [e["event"] for e in seen][:2] == ["stage_start", "scf_iteration"]
    assert json.loads(json.dumps(seen)) == seen
    # the escalation within the budget is an event of its own (here at 2/3 of max_iter 6)
    seen = []
    failed = qccompute.compute(program_input(water, method="hf", basis="sto-3g", threads=2, scf={"max_iter": 6}),
                               progress=seen.append, raise_exc=False)
    esc = [e for e in seen if e["event"] == "scf_escalation"]
    record = failed.record["scf"]["escalation"]
    assert esc == [{"event": "scf_escalation", "iteration": record["at"], "trigger": "two_thirds",
                    "from_iteration": record["from_iter"], "path": ["linus"]}]
    assert [e for e in seen if e["event"] == "stage_end"][-1]["success"] is False


def test_version_is_the_distributions():
    """linus.__version__ and the program's version are the distribution's PEP 440 version (0.1.3
    for Cargo's 0.1.3), the one qccompute records in provenance. The distribution is afh-linus,
    or afh-linus-cuda13 for the GPU build (`linus.GPU_BUILD`), with the same version."""
    import importlib.metadata

    from linus.qc import program

    dist = "afh-linus-cuda13" if linus.GPU_BUILD else "afh-linus"
    assert linus.__version__ == importlib.metadata.version(dist) == program.version()
    assert "-" not in linus.__version__ and "+" not in linus.__version__


@pytest.mark.skipif(linus.GPU_BUILD, reason="the CPU build's answer")
def test_the_cpu_build_refuses_backend_gpu(water):
    """`linus.GPU_BUILD` is False exactly when `backend: "gpu"` is refused as unsupported."""
    e = _raises(program_input(water, backend="gpu"))
    assert isinstance(e, linus.UnsupportedError) and "without GPU support" in e.error.message


def test_verbose_never_prints_in_process(water, capfd):
    """output.verbose in process: the lines go to the output's logs (and a failure's), not to the
    process's stderr."""
    out = linus.compute(program_input(water, output={"verbose": True}))
    e = _raises(program_input(water, method="hf", output={"verbose": True}, scf={"max_iter": 2, "escalate": False}))
    captured = capfd.readouterr()
    assert captured.err == "" and captured.out == ""
    assert "iter   2" in e.output.logs and out.logs.count("iter ") >= 5


def _raises(inp):
    """The `qcdata.ProgramError` `linus.compute` raises for `inp` (decision 9)."""
    from qcdata import ProgramError

    with pytest.raises(ProgramError) as info:
        linus.compute(inp)
    e = info.value
    assert isinstance(e, linus.LinusError) and e.output is not None and not e.output.success
    assert e.output.error == e.error and e.results is e.output.results
    return e


def test_unsupported_inputs_fail_clearly(water):
    out = linus.compute(program_input(water, CalcType.gradient))
    assert out.success and np.asarray(out.results.gradient).shape == (3, 3)
    # linus.compute raises failures, with qcdata's error kinds and codes and what the run reached
    opt = _raises(program_input(water, CalcType.optimization))
    assert opt.error.kind == "unsupported" and "qcopt" in opt.error.message
    unknown = _raises(program_input(water, conv=1e-8))
    assert unknown.error.code == "unknown_key" and "conv" in unknown.error.message
    no_basis = _raises(ProgramInput(program="linus", calctype="energy", structure=water, model=Model(method="hf")))
    assert "basis set" in no_basis.error.message
    snk = _raises(program_input(water, CalcType.hessian, method="hf", basis="sto-3g", jk="rij_snk"))
    assert snk.error.kind == "unsupported" and "rij_snk" in snk.error.message
    failed = _raises(program_input(water, method="hf", scf={"max_iter": 2, "escalate": False}))
    assert failed.error.kind == "convergence" and "converge" in failed.output.traceback
    assert failed.error.code == "scf_not_converged"  # the code every qc* program gives it
    assert failed.results.provenance.program == "linus" and len(failed.output.record["scf"]["history"]) == 2
    assert failed.results.energy is not None  # the last iterate's
    # through qccompute: the same ProgramError (raise_exc=False: the failed output in place)
    from qcdata import ProgramError

    inp = program_input(water, method="hf", scf={"max_iter": 2, "escalate": False})
    with pytest.raises(ProgramError) as info:
        qccompute.compute(inp)
    assert info.value.error.kind == "convergence"
    out = qccompute.compute(inp, raise_exc=False)
    assert not out.success and "did not converge" in out.error.message


def test_cli_json_modes(cli, water, reference):
    ref = reference.results
    req = {"structure": {"symbols": list(water.symbols), "geometry": water.geometry.tolist()}, "method": "b3lyp",
           "basis": "def2-svp", "jk": "df", "output": {"orbitals": True}}
    r = json.loads(subprocess.run([cli, "--json"], input=json.dumps(req), capture_output=True, text=True,
                                  check=True).stdout)
    assert r["energy"] == pytest.approx(ref.energy, abs=1e-10)
    assert np.shape(r["mo_coefficients_alpha"]) == (ref.calcinfo_nbasis, ref.calcinfo_nmo)
    assert r["options"]["jk"] == {"kind": "df"}  # only the settings given; def2-universal-jkfit is in `resolved`
    assert r["resolved"]["settings"]["jk"]["aux"] == "def2-universal-jkfit" and "structure" in r
    bad = subprocess.run([cli, "--json"], input=json.dumps({**req, "metod": "hf"}), capture_output=True, text=True,
                         check=False)
    assert bad.returncode == 2 and "metod" in bad.stderr and "method" in bad.stderr  # unknown key, with the suggestion
    xyz = str(MOLECULES / "water.xyz")
    text = subprocess.run([cli, xyz, "--method", "b3lyp", "--jk", "df"], capture_output=True, text=True, check=True)
    assert "total energy" in text.stdout and f"{ref.energy:.8f}"[:-1] in text.stdout
    r = json.loads(subprocess.run([cli, xyz, "--method", "hf", "--charge", "1", "--mult", "2", "--json"],
                                  capture_output=True, text=True, check=True).stdout)
    assert r["nalpha"] == 5 and r["nbeta"] == 4 and not r["restricted"]


def test_memory_failures_give_their_numbers(cli, water):
    """A host_memory failure gives `required_gib` and `available_gib` in `ErrorInfo.details`
    (qc* #45; the CLI's `error_details`), both as memory_gib: rerun with memory_gib at
    `required_gib`, what failed fits (the in-core integrals, the DF exchange tensor)."""
    keywords = {"jk": {"kind": "exact", "exact_storage": "incore"}}
    out = _raises(program_input(water, method="hf", memory_gib=0.001, **keywords)).output
    assert (out.error.kind, out.error.code) == ("resource", "host_memory")
    need, have = out.error.details["required_gib"], out.error.details["available_gib"]
    assert have == pytest.approx(0.001, rel=1e-5) and need > have  # the budget in whole bytes
    assert linus.compute(program_input(water, method="hf", memory_gib=need * 1.001, **keywords)).success
    # the DF exchange tensor's budget likewise (75% of memory_gib, less the reservations)
    df = _raises(program_input(water, basis="def2-tzvp", jk="df", memory_gib=0.001)).output.error
    need_df, have_df = df.details["required_gib"], df.details["available_gib"]
    assert df.code == "host_memory" and have_df == pytest.approx(0.001, rel=1e-5) and need_df > have_df
    assert linus.compute(program_input(water, basis="def2-tzvp", jk="df", memory_gib=need_df * 1.01)).success
    req = {"structure": {"symbols": list(water.symbols), "geometry": water.geometry.tolist()}, "method": "hf",
           "basis": "def2-svp", "memory_gib": 0.001, **keywords}
    bad = subprocess.run([cli, "--json"], input=json.dumps(req), capture_output=True, text=True, check=False)
    j = json.loads(bad.stdout)
    assert bad.returncode == 1 and j["error_code"] == "host_memory"
    # memory_gib limits the whole process: the command line's own memory is 64 MiB less than Python's
    assert j["error_details"]["available_gib"] == have
    assert j["error_details"]["required_gib"] == pytest.approx(need - 0.0625, abs=1e-9)
    # a runner's memory grant reaches linus unrounded (qc* #69; 0.0004 GiB used to arrive as 0)
    tiny = qccompute.compute(program_input(water, method="hf", memory_gib=0.0004), raise_exc=False).error
    assert tiny.code == "host_memory" and tiny.details["available_gib"] == pytest.approx(0.0004, rel=1e-5)


def _stopped(inp, when, isolation="auto"):
    """The `qcdata.ProgramError` of `inp` submitted to a runner and asked to stop
    (`qccompute.stop`) at the first progress event `when(event)` accepts. In process the listener
    runs on the calculation's thread: it then waits out linus's shortest interval between asking
    in iterations (50 ms), so the stop lands at the next check."""
    ready, asked = threading.Event(), []

    def listen(e):
        if not asked and when(e):
            ready.wait(30)
            asked.append(qccompute.stop(fut))
            time.sleep(0.06)

    with qccompute.Runner(threads=2, isolation=isolation) as r:
        fut = r.submit(inp, progress=listen)
        ready.set()
        from qcdata import ProgramError

        with pytest.raises(ProgramError) as info:
            fut.result(timeout=300)
    assert asked == [True]
    e = info.value
    assert e.error.kind == "stopped" and e.output is not None and not e.output.success
    return e


def test_a_stop_request_ends_at_the_next_step(water):
    """`ctx.stop_requested()` (qccompute's `stop`, qc* #75): linus asks before each stage and
    between the SCF's and the responses' iterations, and ends with kind `stopped` and what it
    reached: in the SCF its last iterate, after it the converged SCF with the gradient it has."""
    ref = linus.compute(program_input(water, CalcType.hessian, method="hf", threads=2))
    # in the SCF: after the iteration it was asked in, with that iterate's energy
    e = _stopped(program_input(water, CalcType.hessian, method="hf", threads=2),
                 lambda e: e["event"] == "scf_iteration" and e["iteration"] == 3)
    history = e.output.record["scf"]["history"]
    assert e.error.message == "stopped on request in the SCF after iteration 3"
    assert [h["step"] for h in history][-1] == "stopped" and len(history) == 3
    assert e.results.energy == history[-1]["energy"] and e.results.gradient is None and e.results.hessian is None
    # between stages: the converged SCF, before the gradient
    e = _stopped(program_input(water, CalcType.gradient, method="hf", threads=2),
                 lambda e: e["event"] == "stage_end" and e["stage"] == "scf")
    assert e.error.message == "stopped on request before the gradient"
    g = linus.compute(program_input(water, CalcType.gradient, method="hf", threads=2))
    assert e.results.energy == g.results.energy and e.results.gradient is None
    # in the Hessian's response: the energy and the gradient it reached
    e = _stopped(program_input(water, CalcType.hessian, method="hf", threads=2),
                 lambda e: e["event"] == "response_iteration")
    assert e.error.message == "stopped on request in the Hessian after iteration 1"
    np.testing.assert_array_equal(e.results.gradient, ref.results.gradient)
    assert e.results.energy == ref.results.energy and e.results.hessian is None
    # in a worker the request crosses as a file: the calculation stops at a later step
    s = ROOT / "tests" / "molecules" / "caffeine.xyz"
    from qcdata import Structure

    caffeine = Structure.open(s)
    e = _stopped(program_input(caffeine, jk="df", threads=2), lambda e: e["event"] == "scf_iteration",
                 isolation="process")
    assert e.error.message.startswith("stopped on request in the SCF after iteration ")
    assert e.output.record["scf"]["history"][-1]["step"] == "stopped" and e.results.energy is not None
    # a run nobody stops is bitwise the same as a direct one (the checks change no number)
    out = qccompute.compute(program_input(water, CalcType.hessian, method="hf", threads=2))
    np.testing.assert_array_equal(out.results.hessian, ref.results.hessian)


def test_a_cpu_run_under_a_gpu_grant_says_no_device(water):
    """qc* #68: linus records the devices it ran on (`execution.devices`, set explicitly: empty
    on the CPU), and the runner keeps an explicit empty list under a GPU grant: linus planned with
    a GPU, run on the CPU."""
    from linus.qc import LinusProgram
    from qccompute.program import register_program, unregister_program

    class PlannedOnAGpu(LinusProgram):
        def plan(self, inp):
            p = super().plan(inp)
            return {**p, "resources": {**p["resources"], "gpus": 1}}

    out = linus.compute(program_input(water, method="hf"))
    assert "devices" in out.execution.model_fields_set and out.execution.devices == []
    register_program(PlannedOnAGpu())
    try:
        with qccompute.Runner(threads=2, gpus=["0"]) as r:
            ran = r.compute(program_input(water, method="hf"))
    finally:
        unregister_program("linus")
    assert ran.success and ran.execution.devices == [] and ran.results.energy == out.results.energy
