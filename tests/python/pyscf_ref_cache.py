"""Opt-in cache of PySCF reference results for the compare scripts in tests/python.

Set LINUS_PYSCF_CACHE=<dir> to store each PySCF result (energy, gradient, Hessian, ...) under a
key built from everything that determines it: the payload the script passes (molecule, basis and
auxiliary-basis data, method and options), the script's name and a hash of its source and of
common.py (so editing a script, e.g. a tolerance or a PySCF setting, invalidates its entries),
and PySCF's version. A hit skips PySCF entirely, so repeated merge gates only rerun linus.
Unset: no caching. Scripts use it through common.cached(__file__, key, compute)."""
import hashlib, json, os, pathlib

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent


def source_hash(*files):
    h = hashlib.sha256()
    for f in files:
        h.update(pathlib.Path(f).read_bytes())
    return h.hexdigest()[:16]


def cache_path(script, payload):
    d = os.environ.get("LINUS_PYSCF_CACHE")
    if not d:
        return None
    import pyscf
    key = dict(payload, script=pathlib.Path(script).name, source=source_hash(script, HERE / "common.py"), pyscf=pyscf.__version__)
    h = hashlib.sha256(json.dumps(key, sort_keys=True, default=str).encode()).hexdigest()[:32]
    p = pathlib.Path(d)
    p.mkdir(parents=True, exist_ok=True)
    return p / f"{pathlib.Path(script).stem}-{h}.npz"


def load(path):
    """The stored arrays (0-d ones as Python scalars), or None."""
    if path is None or not path.exists():
        return None
    with np.load(path) as z:
        return {k: (z[k].item() if z[k].ndim == 0 else z[k]) for k in z.files}


def save(path, **arrays):
    if path is not None:
        tmp = path.with_suffix(".tmp.npz")
        np.savez(tmp, **arrays)
        tmp.replace(path)
