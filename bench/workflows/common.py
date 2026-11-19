# Record of the 2026-11-07 study: runs on linus b36bf33 and earlier; its API (Calculator, QcoptEvaluator, propagate_wfn) is gone.
"""Shared settings and helpers of the end-to-end workflow runs (linus vs ORCA 6.0.0).

Every calculation goes through qc* (qcdata ProgramInput, qccompute, qccodec); orchestration is
plain Python. Results land in bench/results/2026-11-07-workflows/ as JSON.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MOLECULES = ROOT / "bench" / "molecules"
INPUTS = Path(__file__).resolve().parent / "inputs"
RESULTS = ROOT / "bench" / "results" / "2026-11-07-workflows"

BOHR_TO_ANGSTROM = 0.529177210903
HARTREE_TO_KCAL = 627.509474

sys.path.insert(0, str(ROOT / "bench"))
from bench_paths import ORCA, REFENV  # noqa: E402  (env overrides LINUS_ORCA_DIR, LINUS_REFENV: bench/bench_paths.py)

# B3LYP-D3(BJ)/def2-SVP, both programs. ORCA's B3LYP/G is the VWN-RPA B3LYP of linus's "b3lyp".
SVP_ORCA = "B3LYP/G D3BJ def2/J RIJCOSX DefGrid2 TightSCF"
SVP_LINUS = {"method": "b3lyp-d3bj", "basis": "def2-svp"}  # qcdata Model
SVP_LINUS_CALC = {"method": "b3lyp", "dispersion": "d3bj", "basis": "def2-svp"}  # linus.Calculator
# Production single point
TZVP_ORCA = "wB97X-D3BJ def2/J RIJCOSX DefGrid2 TightSCF"
TZVP_LINUS = {"method": "wb97x-d3bj", "basis": "def2-tzvp"}
# ... in THF (C-PCM): linus keywords, ORCA simple-input word
PCM_LINUS = {"solvent": {"name": "thf", "model": "cpcm"}}
PCM_ORCA = "CPCM(THF)"
# TS workflow levels: name -> (linus qcdata Model, ORCA method string). "svp": the B3LYP-D3(BJ)
# workflow level; "wb97x-svp": wB97X-D3BJ/def2-SVP, for range-separated-hybrid frequencies.
LEVELS = {
    "svp": (SVP_LINUS, SVP_ORCA),
    "wb97x-svp": ({"method": "wb97x-d3bj", "basis": "def2-svp"}, "wB97X-D3BJ def2/J RIJCOSX DefGrid2 TightSCF"),
}

THREADS = 8
# ORCA ranks (WF_ORCA_PAL: fewer when the job must share a CCD; recorded with each result)
ORCA_PAL = int(os.environ.get("WF_ORCA_PAL", THREADS))
ORCA_MAXCORE = 2000  # MB per rank; 8 ranks

# Minimization convergence, ORCA's TightOpt everywhere: TolE 1e-6 Eh, TolRMSG 3e-5,
# TolMaxG 1e-4 Eh/bohr, TolRMSD 6e-4, TolMaxD 1e-3 bohr (geomeTRIC takes displacements in Å).
GEOMETRIC_TIGHTOPT = {
    "convergence_energy": 1e-6, "convergence_grms": 3e-5, "convergence_gmax": 1e-4,
    "convergence_drms": 6e-4 * BOHR_TO_ANGSTROM, "convergence_dmax": 1e-3 * BOHR_TO_ANGSTROM,
}
# Sella: max atomic force in eV/Å; 1e-4 Eh/bohr = 0.0051 eV/Å
SELLA_MIN_FMAX = 0.005
# AFH TS protocol (ts-optimization-irc skill): fmax 0.001 eV/Å for Sella TS searches
SELLA_TS_FMAX = 0.001

MOLS = {
    "la18": {"file": "la18.xyz", "charge": 0, "multiplicity": 1},
    "u1an39": {"file": "u1an39.xyz", "charge": -1, "multiplicity": 1},
    "ts61": {"file": "ts61.xyz", "charge": 0, "multiplicity": 1},
    # HCN to HNC 1,2-H shift, the AFH TS protocol's test reaction (a rough bent guess)
    "hcn_ts": {"file": "hcn_ts_guess.xyz", "dir": "inputs", "charge": 0, "multiplicity": 1},
}

# Perturbed starts: Gaussian displacements (sigma 0.05 Å per coordinate, fixed seed), net
# translation removed. Written once to inputs/ so both programs start from the same file.
PERTURB_SIGMA = 0.05
PERTURB_SEED = 20261107


def structure(name: str, *, perturbed: bool = False):
    """The molecule as a qcdata Structure (bohr), from bench/molecules or its perturbed copy."""
    from qcdata import Structure

    m = MOLS[name]
    if not perturbed:
        d = INPUTS if m.get("dir") == "inputs" else MOLECULES
        return Structure.open(d / m["file"], charge=m["charge"], multiplicity=m["multiplicity"])
    path = INPUTS / f"{name}_perturbed.xyz"
    if not path.exists():
        s = structure(name)
        rng = np.random.default_rng(PERTURB_SEED + sum(map(ord, name)))
        d = rng.normal(0.0, PERTURB_SIGMA, s.geometry.shape) / BOHR_TO_ANGSTROM
        d -= d.mean(axis=0)
        p = s.model_copy(update={"geometry": s.geometry + d})
        INPUTS.mkdir(parents=True, exist_ok=True)
        lines = [str(len(p.symbols)), f"{name} perturbed: sigma {PERTURB_SIGMA} A, seed {PERTURB_SEED}+"
                 f"{sum(map(ord, name))}, charge={m['charge']} mult={m['multiplicity']}"]
        for sym, xyz in zip(p.symbols, p.geometry * BOHR_TO_ANGSTROM):
            lines.append(f"{sym:2s} {xyz[0]:16.10f} {xyz[1]:16.10f} {xyz[2]:16.10f}")
        path.write_text("\n".join(lines) + "\n")
    return Structure.open(path, charge=m["charge"], multiplicity=m["multiplicity"])


def with_geometry(s, geometry):
    return s.model_copy(update={"geometry": np.asarray(geometry, dtype=float).reshape(-1, 3)})


MASSES = {"H": 1.008, "C": 12.011, "N": 14.007, "O": 15.999, "F": 18.998, "K": 39.098, "S": 32.06, "Cl": 35.45}


def rmsd(a, b, symbols=None) -> float:
    """Best-fit (Kabsch) RMSD in Å between two geometries in bohr (same atom order); mass
    weights when `symbols` are given."""
    a = np.asarray(a, dtype=float).reshape(-1, 3)
    b = np.asarray(b, dtype=float).reshape(-1, 3)
    w = np.ones(len(a)) if symbols is None else np.array([MASSES[s] for s in symbols])
    ca, cb = (w[:, None] * a).sum(0) / w.sum(), (w[:, None] * b).sum(0) / w.sum()
    pa, pb = a - ca, b - cb
    h = (w[:, None] * pa).T @ pb
    u, _, vt = np.linalg.svd(h)
    dmat = np.diag([1.0, 1.0, np.sign(np.linalg.det(vt.T @ u.T))])
    r = vt.T @ dmat @ u.T
    diff = pa @ r.T - pb
    return float(np.sqrt((w * (diff**2).sum(1)).sum() / w.sum()) * BOHR_TO_ANGSTROM)


def orca_env() -> dict[str, str]:
    return {
        "PATH": f"{ORCA}:{REFENV}/bin:" + os.environ.get("PATH", ""),
        "LD_LIBRARY_PATH": f"{ORCA}/lib:{REFENV}/lib",
        "OMPI_MCA_btl": "self,vader",
        "OMPI_MCA_btl_vader_single_copy_mechanism": "none",
        "OMPI_MCA_hwloc_base_binding_policy": "none",
    }


def ccd_load(seconds: float = 1.0) -> list[float]:
    """Busy physical cores per CCD (0-8) over `seconds` (/proc/stat), for the record."""

    def snap():
        d = {}
        for line in open("/proc/stat"):
            if line.startswith("cpu") and line[3].isdigit():
                p = line.split()
                v = list(map(int, p[1:]))
                d[int(p[0][3:])] = (sum(v), v[3] + v[4])
        return d

    a = snap()
    time.sleep(seconds)
    b = snap()
    busy = {c: 1 - (b[c][1] - a[c][1]) / max(1, b[c][0] - a[c][0]) for c in a}
    return [round(sum(busy.get(c, 0.0) for c in range(8 * k, 8 * k + 8)), 1) for k in range(8)]


def affinity() -> list[int]:
    return sorted(os.sched_getaffinity(0))


def save(name: str, data: dict[str, Any]) -> Path:
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / f"{name}.json"
    path.write_text(json.dumps(data, indent=1, default=_json_default) + "\n")
    return path


def load(name: str) -> dict[str, Any] | None:
    path = RESULTS / f"{name}.json"
    return json.loads(path.read_text()) if path.exists() else None


# Full Hessians stay out of git: `externalize_hessians` moves each "hessian" array of the result
# files into bench/scratch/hessians/<file>.<path>.npy and leaves {"npy": <that name>}.
HESSIANS = ROOT / "bench" / "scratch" / "hessians"


def externalize_hessians() -> None:
    HESSIANS.mkdir(parents=True, exist_ok=True)

    def walk(obj: Any, stem: str) -> Any:
        if isinstance(obj, dict):
            out = {}
            for k, v in obj.items():
                if k == "hessian" and isinstance(v, list):
                    fname = f"{stem}.npy"
                    np.save(HESSIANS / fname, np.asarray(v, dtype=float))
                    out[k] = {"npy": fname}
                else:
                    out[k] = walk(v, f"{stem}.{k}")
            return out
        if isinstance(obj, list):
            return [walk(v, f"{stem}.{i}") for i, v in enumerate(obj)]
        return obj

    for path in sorted(RESULTS.glob("*.json")):
        data = json.loads(path.read_text())
        new = walk(data, path.stem)
        if new != data:
            path.write_text(json.dumps(new, indent=1) + "\n")


def hessian(value: Any) -> np.ndarray | None:
    """A Hessian stored inline (list) or by `externalize_hessians` ({"npy": name})."""
    if value is None:
        return None
    if isinstance(value, dict):
        return np.load(HESSIANS / value["npy"])
    return np.asarray(value, dtype=float)


def _json_default(x: Any) -> Any:
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, Path):
        return str(x)
    raise TypeError(type(x).__name__)


def xyz_text(s, comment: str = "") -> str:
    lines = [str(len(s.symbols)), comment]
    for sym, xyz in zip(s.symbols, np.asarray(s.geometry) * BOHR_TO_ANGSTROM):
        lines.append(f"{sym:2s} {xyz[0]:16.10f} {xyz[1]:16.10f} {xyz[2]:16.10f}")
    return "\n".join(lines) + "\n"


class Clock:
    """Wall-clock sections of a workflow."""

    def __init__(self) -> None:
        self.t0 = time.perf_counter()
        self.sections: dict[str, float] = {}

    def lap(self, name: str, t_start: float) -> float:
        dt = time.perf_counter() - t_start
        self.sections[name] = self.sections.get(name, 0.0) + dt
        return dt

    @property
    def total(self) -> float:
        return time.perf_counter() - self.t0
