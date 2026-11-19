# Record of the 2026-11-07 study: runs on linus b36bf33 and earlier; its API (Calculator, QcoptEvaluator, propagate_wfn) is gone.
"""Repro: the RI (jk=df) Hessian of a one-electron system panics in jk_df_deriv.rs.

    python bench/workflows/repros/h_atom_hessian_panic.py

linus at 1921fd1 (+ this branch, which does not touch the Hessian):
    thread '<unnamed>' panicked at crates/linus/src/jk_df_deriv.rs:75:9: chunk_size must not be zero
    -> _linus.LinusError: internal error in linus: chunk_size must not be zero

The H atom (doublet, UKS B3LYP/def2-SVP, jk=df) energy and gradient work; the Hessian panics.
Probably the beta spin with no occupied orbitals (nbeta = 0) gives a zero-sized chunk. Any
system with an empty spin channel (H atom, H2+, ...) hits it; exact J/K is not affected.
"""

import numpy as np

import linus

calc = linus.Calculator(["H"], np.zeros((1, 3)), method="b3lyp", basis="def2-svp", multiplicity=2, jk="df")
print("energy", calc.energy().energy)
print("gradient", calc.gradient().gradient.tolist())
try:
    calc.hessian()
except linus.LinusError as e:
    print("hessian FAILED:", e)
calc = linus.Calculator(["H"], np.zeros((1, 3)), method="b3lyp", basis="def2-svp", multiplicity=2, jk="exact")
print("exact J/K Hessian ok:", calc.hessian().hessian.shape)
