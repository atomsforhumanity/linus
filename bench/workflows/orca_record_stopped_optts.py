"""Record an ORCA OptTS that was stopped before ORCA finished, from its trajectory file.

    python bench/workflows/orca_record_stopped_optts.py <orca_trj.xyz> <wall_s> "<reason>"

Used once (2026-11-07): the ts61 OptTS (calc_hess true, TightOpt, maxiter 200) had crept down
~0.01 kcal/mol per step for 25 steps at step 138 after 4 h 15 min on a loaded CCD; per the AFH TS
protocol's failure table ("max steps, energy drifting down": a fresh exact Hessian at the current
point and a restart) it was stopped and continued (`orca_workflows.py ts`, stage optts2). This
writes the stopped stage as `optts` (converged False) so the chain resumes from its last frame.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common as C  # noqa: E402
from orca_workflows import env_record, read_xyz_frames  # noqa: E402


def main() -> None:
    path, wall, reason = sys.argv[1], float(sys.argv[2]), sys.argv[3]
    text = Path(path).read_text()
    frames = read_xyz_frames(text)
    energies = []
    for line in text.splitlines():
        if " E " in line:
            parts = line.split()
            energies.append(float(parts[parts.index("E") + 1]))
    key = "orca_ts_ts61"
    rec = C.load(key) or {"program": "orca 6.0.0", "molecule": "ts61", "level": C.SVP_ORCA + " / def2-SVP",
                          "threads": C.THREADS, "before": env_record(), "walls": {}}
    rec["optts"] = {"success": False, "converged": False, "not_converged_message": False, "n_steps": len(frames),
                    "energies": energies, "final_energy": energies[-1], "final_geometry_bohr": frames[-1],
                    "wall_s": wall, "stopped": reason, "traceback": None}
    C.save(key, rec)
    print(f"recorded optts: {len(frames)} frames, last E {energies[-1]}, wall {wall} s")


if __name__ == "__main__":
    main()
