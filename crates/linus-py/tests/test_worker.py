"""linus in a worker process: qccompute's serve loop, `python -m qccompute.serve linus`, which finds
linus through the ``qccompute.programs`` entry point. Frames cross the pipe bit for bit, a session's
lineages live across calls (and across a socket's connections), and the worker exits with status 3
after answering a request that failed with ``device_lost``."""

import json
import os
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pytest

qcdata = pytest.importorskip("qcdata")
serve = pytest.importorskip("qccompute.serve", reason="needs qccompute's serve loop (branch execution-protocol)")

import linus  # noqa: E402
from qccompute.program import find_program  # noqa: E402

if find_program("linus") is None:
    pytest.skip("linus is not installed (no qccompute.programs entry point): install the wheel",
                allow_module_level=True)

POS = np.array([[0.0, 0.0, 0.223], [0.038, 1.427, -0.890], [-0.019, -1.436, -0.879]])
WATER = qcdata.Structure(symbols=["O", "H", "H"], geometry=POS)
INP = qcdata.ProgramInput(program="linus", calctype="gradient", structure=WATER,
                          model={"method": "b3lyp", "basis": "sto-3g"}, keywords={"threads": 2})
WIRE_INP = INP.model_dump(mode="json")


def moved(dz, atom=0):
    x = POS.copy()
    x[atom, 2] += dz
    return x


def _env():
    """This process's linus, qcdata and qccompute first on the worker's path."""
    import qccompute

    dirs = [str(Path(m.__file__).resolve().parents[1]) for m in (linus, qcdata, qccompute)]
    rest = [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p]
    return {**os.environ, "PYTHONPATH": os.pathsep.join(dict.fromkeys(dirs + rest))}


class Worker:
    """A worker on a pipe: its hello, and one reply per message."""

    def __init__(self, args, stderr):
        self.p = subprocess.Popen([sys.executable, *args], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=stderr, text=True, env=_env())
        self.hello = json.loads(self.p.stdout.readline())
        self.ids = 0

    def send(self, line):
        self.p.stdin.write(line + "\n")
        self.p.stdin.flush()
        return json.loads(self.p.stdout.readline())

    def ask(self, **msg):
        self.ids += 1
        reply = self.send(json.dumps({"id": self.ids, **msg}))
        assert reply["id"] == self.ids, reply
        return reply

    def ok(self, **msg):
        reply = self.ask(**msg)
        assert reply["ok"], reply
        return reply["result"]

    def end(self):
        """End of input: the worker closes its sessions and exits; its status, and what it wrote after."""
        self.p.stdin.close()
        status = self.p.wait(timeout=120)
        return status, self.p.stdout.read()


@pytest.fixture
def worker(tmp_path):
    with open(tmp_path / "stderr.log", "w") as err:
        w = Worker(["-m", "qccompute.serve", "linus"], err)
        yield w
        if w.p.poll() is None:
            w.p.kill()
            w.p.wait()


def test_frames_cross_bit_for_bit(worker):
    hello = worker.hello
    assert hello["protocol"] == 1 and hello["program"] == "linus" and hello["version"] == linus.__version__
    assert hello["capabilities"]["extras"] == {"numerics_epoch": linus.NUMERICS_EPOCH}
    assert sorted(hello["capabilities"]["methods"]["b3lyp"]) == ["energy", "gradient", "hessian"]  # a set
    sid = worker.ok(op="open", input=WIRE_INP, ctx={"threads": 2})["session"]
    steps = [None, moved(0.05), moved(-0.03, 1)]
    wire = [worker.ok(op="compute", session=sid, **({} if x is None else {"positions": x.tolist()})) for x in steps]
    wire_h = worker.ok(op="compute", session=sid, calctype="hessian")
    one = worker.ok(op="compute", input=WIRE_INP)
    with linus.Session(INP) as s:
        here = [s.compute(x) for x in steps]
        here_h = s.compute(calctype="hessian")
    for w, h in zip(wire + [wire_h], here + [here_h], strict=True):
        f = qcdata.Frame.model_validate(w)
        assert f.success and f.results.energy == h.results.energy
        assert np.array_equal(f.results.gradient, h.results.gradient)
        assert f.results.scf_iterations == h.results.scf_iterations
        assert np.array_equal(f.structure.geometry, h.structure.geometry)
        assert f.record["scf"]["guess"] == h.record["scf"]["guess"]
    assert np.array_equal(qcdata.Frame.model_validate(wire_h).results.hessian, here_h.results.hessian)
    # a one-off: linus.compute's ProgramOutput
    out = linus.compute(INP)
    assert one["success"] and one["input_data"]["program"] == "linus"
    assert one["results"]["energy"] == out.results.energy
    assert np.array_equal(np.array(one["results"]["gradient"]), out.results.gradient)
    # NaN never crosses: the message is refused; a request linus refuses is a failed frame
    nan = worker.ask(op="compute", session=sid, positions=np.full((3, 3), np.nan).tolist())
    assert not nan["ok"] and "NaN" in nan["error"]["message"]
    failed = worker.ok(op="compute", session=sid, outputs=["bogus"])
    assert not failed["success"] and failed["error"]["kind"] == "unsupported"
    assert not worker.ask(op="compute", session="nope")["ok"] and not worker.ask(op="fly")["ok"]
    assert worker.send("{not json")["ok"] is False
    assert worker.ask(op="close", session=sid)["ok"]
    assert worker.end() == (0, "")  # nothing but protocol lines


def test_each_call_runs_on_its_own_grant():
    """A runner's workers serve calls of different threads (qc* alpha 4: pooled by program and
    GPUs only), and linus applies each call's grant: the same worker runs one call on 1 thread and
    the next on 3, and each output says so."""
    import qccompute

    with qccompute.Runner(threads=4, isolation="process") as runner:
        outs = [runner.compute(INP.model_copy(update={"keywords": {"threads": t}})) for t in (1, 3, 1)]
    assert [o.execution.threads for o in outs] == [1, 3, 1]
    assert [o.resolved.settings["threads"] for o in outs] == [1, 3, 1]
    assert outs[0].results.energy == outs[2].results.energy


def test_progress_events_cross_from_the_worker():
    """linus's progress events reach the caller's listener from a worker as in process (qccompute
    streams them over the wire), and the results are bitwise those of a quiet run."""
    import qccompute

    events = []
    with qccompute.Runner(threads=2, isolation="process") as runner:
        out = runner.compute(INP, progress=events.append)
        quiet = runner.compute(INP)
    names = [e["event"] for e in events]
    assert names[0] == "stage_start" and names[-1] == "stage_end" and all(e["path"] == ["linus"] for e in events)
    assert names.count("scf_iteration") == len(out.record["scf"]["history"])
    np.testing.assert_array_equal(out.results.gradient, quiet.results.gradient)


def test_lineages_live_across_calls(worker):
    sid = worker.ok(op="open", input=WIRE_INP, ctx={"threads": 2})["session"]
    first = worker.ok(op="compute", session=sid)
    step = worker.ok(op="compute", session=sid, positions=moved(0.05).tolist())
    assert first["record"]["warm_start"] is False and step["record"]["warm_start"] is True
    assert step["record"]["scf"]["guess"]["source"] == "lineage"
    assert step["results"]["scf_iterations"] < first["results"]["scf_iterations"]
    # an IRC branch seeded from the default lineage, and a fan-out that commits nothing
    branch = worker.ok(op="compute", session=sid, positions=moved(0.08).tolist(), lineage=["irc", 1], seed=[None])
    guess = branch["record"]["scf"]["guess"]
    assert branch["record"]["warm_start"] is True and guess["source"] == "lineage"
    assert guess["lineage"] == ["irc", 1] and guess["from_lineage"] is None  # the default lineage's state
    many = worker.ok(op="compute_many", session=sid, requests=[
        {"positions": moved(0.06).tolist(), "commit": False},
        {"positions": [POS[0].tolist(), POS[0].tolist(), POS[2].tolist()]}])
    assert [f["success"] for f in many] == [True, False] and many[0]["record"]["warm_start"] is True
    assert sorted(map(json.dumps, worker.ok(op="lineages", session=sid))) == ['["irc", 1]', "null"]
    # the session's plan, and its input's at another calctype (the session's input_data)
    plan = worker.ok(op="plan", session=sid)
    assert plan["resolved"]["deferred"] == [] and plan["resources"]["memory_gib"] > 0
    assert worker.ok(op="plan", session=sid, calctype="energy")["record"]["calctype"] == "energy"
    # reset: null forgets the default lineage; absent, every one
    assert worker.ask(op="reset", session=sid, lineage=None)["ok"]
    assert worker.ok(op="lineages", session=sid) == [["irc", 1]]
    assert worker.ask(op="reset", session=sid)["ok"]
    assert worker.ok(op="lineages", session=sid) == []
    assert worker.ok(op="compute", session=sid)["record"]["warm_start"] is False
    assert worker.end()[0] == 0


def test_socket_clients_share_the_sessions(tmp_path):
    # a short path: a Unix socket's is at most 107 bytes
    path = Path(tempfile.mkdtemp(prefix="linus-")) / "w.sock"
    with open(tmp_path / "stderr.log", "w") as err:
        p = subprocess.Popen([sys.executable, "-m", "qccompute.serve", "linus", "--socket", str(path)],
                             stderr=err, env=_env())
        try:
            for _ in range(1200):
                if path.exists() or p.poll() is not None:
                    break
                time.sleep(0.05)

            def client(msgs):
                with socket.socket(socket.AF_UNIX) as c:
                    c.connect(str(path))
                    f = c.makefile("rw", encoding="utf-8")
                    replies = [json.loads(f.readline())]
                    for m in msgs:
                        f.write(json.dumps(m) + "\n")
                        f.flush()
                        replies.append(json.loads(f.readline()))
                    return replies

            a = client([{"id": 1, "op": "open", "input": WIRE_INP}, {"id": 2, "op": "compute", "session": "s1"}])
            assert a[0]["program"] == "linus" and a[2]["result"]["success"]
            assert a[2]["result"]["record"]["warm_start"] is False
            # the session outlives its client's connection
            b = client([{"id": 1, "op": "compute", "session": "s1", "positions": moved(0.05).tolist()}])
            assert b[1]["result"]["record"]["warm_start"] is True
        finally:
            p.terminate()
            p.wait(timeout=60)
            path.unlink(missing_ok=True)
            path.parent.rmdir()


# linus under the serve loop, its engine failing as after a GPU fault that poisoned the device context
POISONED = """
import sys
import linus.session
from linus.errors import BackendError
from qccompute import serve


class Poisoned:
    def __init__(self, engine):
        self._engine = engine

    def compute(self, *args):
        raise BackendError("a device error poisoned the CUDA context (injected)", code="device_lost")

    def __getattr__(self, name):
        return getattr(self._engine, name)


opened = linus.session.Session.__init__


def poisoned(self, *args, **kwargs):
    opened(self, *args, **kwargs)
    self._engine = Poisoned(self._engine)


linus.session.Session.__init__ = poisoned
sys.exit(serve.main(["linus"]))
"""


def test_device_lost_ends_the_worker_after_its_reply(tmp_path):
    with open(tmp_path / "stderr.log", "w") as err:
        w = Worker(["-c", POISONED], err)
        sid = w.ok(op="open", input=WIRE_INP)["session"]
        lost = w.ok(op="compute", session=sid)
        assert not lost["success"] and lost["error"]["kind"] == "backend" and lost["error"]["code"] == "device_lost"
        # nothing more runs in that process: the next message gets no reply
        try:
            w.p.stdin.write(json.dumps({"id": 9, "op": "compute", "session": sid}) + "\n")
            w.p.stdin.flush()
        except BrokenPipeError:  # it has exited already
            pass
        assert w.p.wait(timeout=60) == serve.EXIT_DEVICE_LOST == 3
        assert w.p.stdout.read() == ""
