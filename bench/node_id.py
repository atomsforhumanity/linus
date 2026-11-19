#!/usr/bin/env python3
"""The machine a result was computed on, as an opaque id for provenance (D24): two results that differ can be told
apart by machine (a node that silently corrupts arithmetic, 2026-11-13) without internal host names in anything that
is committed or published.

    node_id()  ->  "node-" + the first 12 hex digits of sha256("linus-node-id/1:<source>:<value>"), from the first of
      root   the host's root file system UUID on the kernel command line (/proc/cmdline: root=UUID=... or
             root=PARTUUID=...). The same on the host and in every container on it, the job runner's pods and the
             node-pinned debug pods alike, so one node has one id wherever the calculation ran.
      k8s    $NODE_NAME, the Kubernetes node (a command line without a root UUID).
      machine  /etc/machine-id, outside a container only (an image's machine-id is not the host's).
    None when none is available.

The ids' mapping to node names stays out of the repository, in the coordinator's state (refstore-meta/nodes.json).
Build it with the id of a known node:

    python3 bench/node_id.py                    this machine's id
    python3 bench/node_id.py --root-uuid UUID   the id of the node whose root file system has UUID
    python3 bench/node_id.py --name NODE        the id of a Kubernetes node name (the k8s source)
"""
import hashlib
import os
import sys
from pathlib import Path

SALT = "linus-node-id/1"


def _id(source: str, value: str) -> str:
    return "node-" + hashlib.sha256(f"{SALT}:{source}:{value}".encode()).hexdigest()[:12]


def _read(path: str) -> str:
    try:
        return Path(path).read_text().strip()
    except OSError:
        return ""


def root_uuid(cmdline: str) -> str | None:
    """The root file system's UUID (or PARTUUID) on a kernel command line, if it names one."""
    for tok in cmdline.split():
        for key in ("root=UUID=", "root=PARTUUID="):
            if tok.startswith(key) and len(tok) > len(key):
                return tok[len(key):]
    return None


def in_container(environ=None) -> bool:
    env = os.environ if environ is None else environ
    return "KUBERNETES_SERVICE_HOST" in env or Path("/.dockerenv").exists() or Path("/run/.containerenv").exists()


def node_id(cmdline: str | None = None, environ=None, machine_id: str | None = None) -> str | None:
    """This machine's opaque id (module docs); the arguments replace /proc/cmdline, the environment and
    /etc/machine-id (tests)."""
    env = os.environ if environ is None else environ
    u = root_uuid(_read("/proc/cmdline") if cmdline is None else cmdline)
    if u:
        return _id("root", u)
    if env.get("NODE_NAME"):
        return _id("k8s", env["NODE_NAME"])
    mid = (_read("/etc/machine-id") if not in_container(env) else "") if machine_id is None else machine_id
    return _id("machine", mid) if mid else None


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[:1] == ["--root-uuid"] and len(a) == 2:
        print(_id("root", a[1]))
    elif a[:1] == ["--name"] and len(a) == 2:
        print(_id("k8s", a[1]))
    elif not a:
        print(node_id())
    else:
        sys.exit(__doc__)
