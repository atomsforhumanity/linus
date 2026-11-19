# Record of the 2026-11-07 study: runs on linus b36bf33 and earlier; its API (Calculator, QcoptEvaluator, propagate_wfn) is gone.
"""Repro: UKS DFT gradients of the OH radical never reach the gradient SCF threshold.

    python bench/workflows/repros/oh_radical_gradient_scf.py

OH (2Pi, 0.97 Å), B3LYP/def2-SVP, UKS. The energy (tight preset) converges in 10 iterations,
but a gradient, whose default asks max |F_ai| < 1e-7, raises ConvergenceError after 128
iterations (also 400; also PBE, jk exact, grid pyscf:5). HF converges; g_tol=1e-6 converges in
9 iterations at the same energy to 1e-10 Eh. PySCF 2.15 UKS (DF, conv_tol_grad 1e-7) also stops
unconverged after 50 cycles, so this is the degenerate pi shell with integer occupation (the
occupied/virtual pi_x/pi_y rotation is nearly flat), not a linus-only defect.

What a workflow sees: an optimizer step on a radical with a degenerate open shell fails with
ConvergenceError although the energy and the gradient are converged far beyond what the
optimizer needs. Possible fixes (SCF owner): accept the gradient threshold as met when the
energy change has been below ~1e-10 Eh for several iterations and |F_ai| has stopped
decreasing (report it in the result), or a second-order step along the flat rotation.
"""

import numpy as np

import linus

pos = np.array([[0, 0, 0], [0, 0, 0.97]]) / 0.529177210903
args = dict(method="b3lyp", basis="def2-svp", multiplicity=2, jk="df", threads=4)
r = linus.Calculator(["O", "H"], pos, **args).energy()
print(f"energy: {r.iterations} iterations, E = {r.energy:.10f}")
try:
    linus.Calculator(["O", "H"], pos, **args).gradient()
except linus.ConvergenceError as e:
    print("gradient (default g_tol 1e-7) FAILED:", e)
r = linus.Calculator(["O", "H"], pos, scf={"g_tol": 1e-6}, **args).gradient()
print(f"gradient with g_tol=1e-6: {r.iterations} iterations, E = {r.energy:.10f}, g_H,z = {r.gradient[1, 2]:.8f}")
