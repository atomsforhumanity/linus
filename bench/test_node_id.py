#!/usr/bin/env python3
"""The node's opaque id in provenance (bench/node_id.py, D24): every writer records it, and none records a host name.

    python3 bench/test_node_id.py        (or pytest bench/test_node_id.py)

Writers: the linus-cpu store's records (reference/validate_gpu.py provenance), the GPU pods' provenance.json
(bench/gpu/h200/provenance.py), the gate's chunk facts (tools/internal/gate/refsuite_pod.py pod.json) and gatefast's
per-chunk meta.json (tools/internal/gate/gatefast.py). Each is run with $NODE_NAME set to a sentinel node name and
must hold this machine's id while neither the sentinel, this machine's host name nor the raw identifier the id hashes
(the root file system's UUID, the machine-id) appears anywhere in it.
"""
import importlib.util
import json
import os
import re
import socket
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))
import node_id as nid  # noqa: E402

SENTINEL = "sentinel-node-name"
ID = re.compile(r"^node-[0-9a-f]{12}$")


def load(rel: str, name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def secrets() -> list[str]:
    """What must never appear: the sentinel node name, this host's name, the raw identifiers behind the id."""
    out = [SENTINEL, socket.gethostname()]
    u = nid.root_uuid(nid._read("/proc/cmdline"))
    if u:
        out.append(u)
    mid = nid._read("/etc/machine-id")
    if mid:
        out.append(mid)
    return [s for s in out if s]


def test_sources():
    a = nid.node_id(cmdline="BOOT_IMAGE=/vmlinuz root=UUID=1111-aaaa ro quiet", environ={})
    b = nid.node_id(cmdline="root=UUID=2222-bbbb", environ={})
    assert ID.match(a) and ID.match(b) and a != b
    assert a == nid.node_id(cmdline="ro root=UUID=1111-aaaa", environ={"NODE_NAME": SENTINEL}), "root UUID first"
    assert "1111" not in a
    k = nid.node_id(cmdline="ro quiet", environ={"NODE_NAME": SENTINEL})
    assert ID.match(k) and k != a and SENTINEL not in k
    assert ID.match(nid.node_id(cmdline="", environ={}, machine_id="0123abcd"))
    # in a container the image's machine-id is not the host's: no id rather than a wrong one
    assert nid.node_id(cmdline="", environ={"KUBERNETES_SERVICE_HOST": "10.0.0.1"}) is None
    assert nid.node_id() is None or ID.match(nid.node_id())


def check(what: str, record, here: str):
    text = json.dumps(record, default=str)
    assert f'"node_id": "{here}"' in text, f"{what}: no node_id {here}: {text[:300]}"
    for s in secrets():
        assert s not in text, f"{what}: {s!r} appears"


def test_writers_record_the_id_and_no_host_name():
    old = os.environ.get("NODE_NAME")
    os.environ["NODE_NAME"] = SENTINEL
    try:
        here = nid.node_id()
        assert here and ID.match(here), "this machine has no id"
        vg = load("reference/validate_gpu.py", "validate_gpu")
        args = types.SimpleNamespace(commit="0" * 40, linus_bin="/usr/local/bin/linus", threads=1)
        check("validate_gpu provenance (linus-cpu record)", vg.provenance(args, 1.0), here)
        check("provenance.json", load("bench/gpu/h200/provenance.py", "provenance").collect(), here)
        pod = load("tools/internal/gate/refsuite_pod.py", "refsuite_pod").pod_facts()
        check("refsuite_pod pod.json", pod, here)
        gf = load("tools/internal/gate/gatefast.py", "gatefast")
        assert gf.node_id() == here
        check("gatefast meta.json chunks", gf.chunks_meta([("t8.00", dict(pod, start=0.0, end=1.0))]), here)
    finally:
        if old is None:
            os.environ.pop("NODE_NAME", None)
        else:
            os.environ["NODE_NAME"] = old


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"{name}: ok")
