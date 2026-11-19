#!/usr/bin/env python3
"""A realistic trajectory for the session benchmark: one geometry optimization with a persistent
`linus.Session` (warm start), scipy BFGS on linus energies and gradients (initial Hessian
TRAJ_H0 times the identity, ASE BFGS's 70 eV/Å² guess, so that the first steps are a few hundredths
of a bohr as in a real optimizer rather than scipy's unit step), from the input
structure displaced by Gaussian noise (TRAJ_RMS_BOHR RMS, fixed seed, no net translation), until the
gradient RMS is below TRAJ_GRMS or after TRAJ_MAX_STEPS gradients. Every gradient evaluated (line-search
trials included) is a step of the trajectory; the replays (replay.py) compute exactly these geometries.

  trajectory.py SYSTEM PLATFORM      (run inside a job: run_all.py traj)
"""
from __future__ import annotations

import sys
import time

import numpy as np
import scipy
import scipy.optimize

import common as C


class Done(Exception):
    pass


def main(system: str, plat: str) -> None:
    s = C.SYSTEMS[system]
    import linus

    symbols, x0 = C.read_xyz(C.ROOT / s["xyz"])
    x0 = np.array(x0)
    rng = np.random.default_rng(s["seed"])
    d = rng.normal(size=x0.shape)
    d -= d.mean(axis=0)
    d *= C.TRAJ_RMS_BOHR / np.sqrt(np.mean(d ** 2))
    start = x0 + d
    import qcdata

    inp = qcdata.ProgramInput(program="linus", calctype="gradient",
                              structure=qcdata.Structure(symbols=symbols, geometry=start, charge=s["charge"],
                                                         multiplicity=s["mult"]),
                              model={"method": C.METHOD, "basis": s["basis"]},
                              keywords={"jk": "df", "threads": C.THREADS, "backend": C.backend(plat)})
    session = linus.Session(inp, calctypes=["gradient"])
    steps = []

    def fun(v):
        x = v.reshape(-1, 3)
        t0 = time.perf_counter()
        r = session.compute(x).results
        wall = time.perf_counter() - t0
        g = np.asarray(r.gradient)
        grms = float(np.sqrt(np.mean(g ** 2)))
        steps.append(dict(positions=x.tolist(), energy=r.energy, grms=grms, gmax=float(np.abs(g).max()),
                          iterations=r.scf_iterations, wall=wall))
        C.log(f"{system} {plat} step {len(steps):2d}: E {r.energy:.10f} grms {grms:.2e} it {r.scf_iterations} "
              f"{wall:.2f} s")
        if grms < C.TRAJ_GRMS or len(steps) >= C.TRAJ_MAX_STEPS:
            raise Done
        return r.energy, g.ravel()

    try:
        h0 = np.eye(start.size) / C.TRAJ_H0
        scipy.optimize.minimize(fun, start.ravel(), jac=True, method="BFGS",
                                options=dict(maxiter=10000, gtol=1e-12, hess_inv0=h0))
        stop = "optimizer stopped"
    except Done:
        stop = "converged" if steps[-1]["grms"] < C.TRAJ_GRMS else "max steps"
    disp = np.asarray(steps[-1]["positions"]) - start
    session.close()
    from linus._library import atomic_numbers

    out = dict(system=system, symbols=symbols, numbers=atomic_numbers(symbols), charge=s["charge"], multiplicity=s["mult"],
               method=C.METHOD, basis=s["basis"], jk="df", generated_on=plat, seed=s["seed"],
               start_rms_bohr=C.TRAJ_RMS_BOHR, stop=stop, grms_target=C.TRAJ_GRMS, max_steps=C.TRAJ_MAX_STEPS,
               optimizer=f"scipy {scipy.__version__} BFGS, hess_inv0 = I / {C.TRAJ_H0} (every gradient a step)",
               final_rms_displacement_from_start=float(np.sqrt(np.mean(disp ** 2))),
               nsteps=len(steps), steps=steps)
    C.dump(C.traj_path(system), out)
    C.log(f"{system}: {len(steps)} steps, {stop}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
