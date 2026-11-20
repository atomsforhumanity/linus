"""Shared fixtures: the reference molecules (Å files, read as bohr by qcdata)."""

import os
from pathlib import Path

import pytest

# PySCF on every core of a big node is slower than on a few for these small molecules; on a small
# runner more threads than cores oversubscribe it (2 CPUs: 8 threads 9% slower than 2)
_cpus = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
os.environ.setdefault("OMP_NUM_THREADS", str(min(8, _cpus)))


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "slow: minutes per test (PySCF's own VV10 runs); per-push CI deselects them with -m 'not slow' "
        "(run them with -m slow)",
    )

ROOT = Path(__file__).resolve().parents[3]
MOLECULES = ROOT / "reference" / "molecules"


@pytest.fixture(scope="session")
def molecules():
    """Open a reference molecule (water, nh3) with a charge and multiplicity."""
    from qcdata import Structure

    def get(name: str, charge: int = 0, multiplicity: int = 1) -> Structure:
        return Structure.open(MOLECULES / f"{name}.xyz", charge=charge, multiplicity=multiplicity)

    return get


@pytest.fixture(scope="session")
def water(molecules):
    return molecules("water")


@pytest.fixture(scope="session")
def cli():
    """The `linus` binary: $LINUS_CLI, else the release build of this source tree (skip without one)."""
    exe = Path(os.environ["LINUS_CLI"]) if os.environ.get("LINUS_CLI") else ROOT / "target" / "release" / "linus"
    if not exe.exists():
        pytest.skip(f"no {exe} (cargo build --release -p linus-cli, or set LINUS_CLI)")
    return exe
