# Record of the 2026-11-07 study: runs on linus b36bf33 and earlier; its API (Calculator, QcoptEvaluator, propagate_wfn) is gone.
"""Conformer screen: many single points through qccompute's batch path (a list of inputs).

    taskset -c <8 cores> <venv>/bin/python bench/workflows/screen.py [modes...]

u1an39 conformers (RDKit ETKDG + MMFF from the SMILES in bench/molecules/u1an39.xyz, written once
to inputs/u1an39_conformers.xyz), B3LYP-D3(BJ)/def2-SVP energies. Modes:

- inproc: one linus.Calculator in this process, moved from conformer to conformer (SAD each
  time): the time linus itself needs, no qc* layer.
- qc_t8: qccompute.compute(list), 8 threads per calculation (one at a time, worker processes).
- qc_t1, qc_t2, qc_t4: 1, 2 or 4 threads per calculation, as many side by side as 8 cores allow.
- orca_t8: ORCA, 8 ranks per calculation, through the same list path.

Writes bench/results/2026-11-07-workflows/screen_<mode>.json.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "1")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import common as C  # noqa: E402
import numpy as np  # noqa: E402

N_CONF = 16
SMILES = "FC(F)(F)c1cc(cc(c1)C(F)(F)F)[N-]C(=O)Nc1cc(cc(c1)C(F)(F)F)C(F)(F)F"


def conformers():
    from qcdata import Structure

    path = C.INPUTS / "u1an39_conformers.xyz"
    if not path.exists():
        from rdkit import Chem
        from rdkit.Chem import AllChem

        mol = Chem.AddHs(Chem.MolFromSmiles(SMILES))
        ids = AllChem.EmbedMultipleConfs(mol, numConfs=N_CONF, randomSeed=20261107)
        AllChem.MMFFOptimizeMoleculeConfs(mol, maxIters=2000)
        blocks = []
        for cid in ids:
            pos = mol.GetConformer(cid).GetPositions()
            lines = [str(mol.GetNumAtoms()), f"u1an39 conformer {cid} (RDKit ETKDG+MMFF, seed 20261107) charge=-1"]
            lines += [f"{a.GetSymbol():2s} {x:16.10f} {y:16.10f} {z:16.10f}" for a, (x, y, z) in zip(mol.GetAtoms(), pos)]
            blocks.append("\n".join(lines))
        C.INPUTS.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(blocks) + "\n")
    lines = path.read_text().splitlines()
    out, i = [], 0
    while i < len(lines):
        n = int(lines[i])
        rows = [r.split() for r in lines[i + 2:i + 2 + n]]
        geom = np.array([[float(x) for x in r[1:4]] for r in rows]) / C.BOHR_TO_ANGSTROM
        out.append(Structure(symbols=[r[0] for r in rows], geometry=geom, charge=-1, multiplicity=1))
        i += 2 + n
    return out


def run_inproc(structs) -> dict:
    import linus

    t = time.perf_counter()
    calc = linus.Calculator(structs[0], **C.SVP_LINUS_CALC, jk="df", threads=C.THREADS)
    setup = time.perf_counter() - t
    rows = []
    for s in structs:
        t1 = time.perf_counter()
        calc.set_positions(s.geometry)
        calc.reset_guess()
        r = calc.energy()
        rows.append({"energy": r.energy, "iterations": r.iterations, "wall_s": time.perf_counter() - t1,
                     "linus_total_s": r.timings.get("total")})
    return {"wall_s": time.perf_counter() - t, "setup_s": setup, "rows": rows}


def run_qc(structs, program: str, threads: int) -> dict:
    from qccompute import compute
    from qcdata import CalcType, Model, ProgramInput

    if program == "linus":
        inps = [ProgramInput(program="linus", calctype=CalcType.energy, structure=s, model=C.SVP_LINUS,
                             keywords={"jk": "df", "threads": threads}) for s in structs]
        kw = {}
    else:
        inps = [ProgramInput(program="orca", calctype=CalcType.energy, structure=s,
                             model=Model(method=C.SVP_ORCA, basis="def2-SVP"),
                             keywords={"pal": threads, "maxcore": C.ORCA_MAXCORE}) for s in structs]
        kw = {"environment": C.orca_env(), "executables": {"orca": str(C.ORCA / "orca")}}
    t = time.perf_counter()
    outs = compute(inps, raise_exc=False, max_workers=C.THREADS, **kw)
    wall = time.perf_counter() - t
    rows = []
    for o in outs:
        ex = (o.results.extras or {}) if o.success else {}
        rows.append({"success": o.success, "energy": getattr(o.results, "energy", None),
                     "iterations": ex.get("scf_iterations"), "wall_s": o.execution.wall_time if o.execution else None,
                     "linus_total_s": (ex.get("timings") or {}).get("total")})
    return {"wall_s": wall, "rows": rows}


def main() -> None:
    structs = conformers()
    modes = sys.argv[1:] or ["inproc", "qc_t8", "qc_t1", "qc_t2", "qc_t4"]
    for mode in modes:
        key = f"screen_{mode}"
        if C.load(key):
            continue
        before = C.ccd_load()
        if mode == "inproc":
            r = run_inproc(structs)
        elif mode.startswith("qc_t"):
            r = run_qc(structs, "linus", int(mode[4:]))
        elif mode.startswith("orca_t"):
            r = run_qc(structs, "orca", int(mode[6:]))
        else:
            raise ValueError(mode)
        r.update({"mode": mode, "n": len(structs), "affinity": C.affinity(), "ccd_busy_before": before,
                  "ccd_busy_after": C.ccd_load(), "per_structure_s": r["wall_s"] / len(structs)})
        lin = [x.get("linus_total_s") for x in r["rows"] if x.get("linus_total_s")]
        if lin:
            r["linus_sum_s"] = float(sum(lin))
        print(f"{mode}: {r['wall_s']:.1f} s for {len(structs)} ({r['per_structure_s']:.2f} s each)", flush=True)
        C.save(key, r)


if __name__ == "__main__":
    main()
