"""Record a running Sella TS search (linus) that is about to be stopped: its log and last geometry.

    python bench/workflows/ts_checkpoint.py <sella_log> <linus.orbitals.npz or -> <wall_s> "<reason>"

The Sella codec runs in-process and returns its trajectory only at the end, so a stopped search
leaves nothing behind. Its log (Sella's step table, captured by qccompute in an unlinked temp
file, readable as /proc/<pid>/fd/N) gives energy and fmax per step. The latest geometry is only
available if an orbitals file of the latest step is at hand (the codec passes the guess orbitals
in the call's arguments, not as a staged file, so usually it is not: pass -). Writes
linus_ts_ts61.json `tsopt_partial`.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common as C  # noqa: E402
import numpy as np  # noqa: E402

EV_TO_HARTREE = 1 / 27.211386245988


def main() -> None:
    log, npz, wall, reason = sys.argv[1], sys.argv[2], float(sys.argv[3]), sys.argv[4]
    steps = []
    for line in Path(log).read_text().splitlines():
        p = line.split()
        if len(p) >= 8 and p[0] == "Sella" and p[1].isdigit():
            steps.append({"step": int(p[1]), "time": p[2], "energy_eh": float(p[3]) * EV_TO_HARTREE,
                          "fmax_ev_per_A": float(p[4]), "trust": float(p[6])})
    positions = None
    if npz != "-":
        with np.load(npz) as z:
            positions = z["positions"].tolist()
    key = "linus_ts_ts61"
    rec = C.load(key) or {"molecule": "ts61"}
    e0 = steps[0]["energy_eh"] if steps else None
    rec["tsopt_partial"] = {
        "stopped": reason, "wall_s": wall, "n_steps": len(steps) - 1, "steps": steps,
        "last_geometry_bohr": positions, "energy_drop_kcal": (e0 - steps[-1]["energy_eh"]) * C.HARTREE_TO_KCAL if steps else None,
        "last_fmax_ev_per_A": steps[-1]["fmax_ev_per_A"] if steps else None,
        "note": "step 0 is the starting gradient; the analytic Hessian ran between step 0 and step 1",
    }
    C.save(key, rec)
    print(f"recorded {len(steps) - 1} Sella steps, energy drop {rec['tsopt_partial']['energy_drop_kcal']:.2f} kcal/mol, "
          f"last fmax {rec['tsopt_partial']['last_fmax_ev_per_A']}")


if __name__ == "__main__":
    main()
