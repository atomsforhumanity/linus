"""Shared paths of the benchmark scripts, each overridable by an environment variable.

  MAIN          the main checkout: LINUS_MAIN, else the parent of `git rev-parse --git-common-dir`,
                so every git worktree of the repository resolves to the same checkout
  LOCK          the shared bench lock (flock): BENCH_LOCK, else <MAIN>/bench/scratch/bench.lock
                (git-ignored); every timed run in every worktree takes this one lock
  REFENV        the reference environment (PySCF, Psi4, Open MPI): LINUS_REFENV, else <MAIN>/.refenv
  DISP_OVERLAY  dftd3/dftd4/pyscf-dispersion overlay: LINUS_REFENV_DISP, else <MAIN>/.refenv-disp
  ORCA          ORCA 6.0.0 binaries (licensed, never committed): LINUS_ORCA_DIR, else ~/opt/orca-6.0.0
  SCRATCH       work directories of runs: BENCH_SCRATCH, else <this checkout>/bench/scratch (git-ignored)

Scripts in bench/ import it directly (`from bench_paths import LOCK, ...`; their directory is on
sys.path when run as `python bench/<script>.py`); scripts in subdirectories add bench/ first.
"""
import os
import pathlib
import subprocess

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _main_checkout() -> pathlib.Path:
    if os.environ.get("LINUS_MAIN"):
        return pathlib.Path(os.environ["LINUS_MAIN"])
    try:
        common = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--path-format=absolute", "--git-common-dir"],
                                capture_output=True, text=True, check=True, timeout=30).stdout.strip()
        return pathlib.Path(common).parent
    except (OSError, subprocess.SubprocessError):
        return ROOT  # not a git checkout (an exported tree): use this one


MAIN = _main_checkout()
LOCK = os.environ.get("BENCH_LOCK", str(MAIN / "bench" / "scratch" / "bench.lock"))
REFENV = pathlib.Path(os.environ.get("LINUS_REFENV", MAIN / ".refenv"))
DISP_OVERLAY = pathlib.Path(os.environ.get("LINUS_REFENV_DISP", MAIN / ".refenv-disp"))
ORCA = pathlib.Path(os.environ.get("LINUS_ORCA_DIR", pathlib.Path.home() / "opt" / "orca-6.0.0"))
SCRATCH = pathlib.Path(os.environ.get("BENCH_SCRATCH", ROOT / "bench" / "scratch"))

try:  # so that open(LOCK, "a") works in a fresh checkout
    pathlib.Path(LOCK).parent.mkdir(parents=True, exist_ok=True)
except OSError:
    pass
