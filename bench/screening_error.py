"""First-order energy error of J/K screening: 1/2 Tr(D dJ) - 1/4 Tr(D dK) between a J/K build
at a threshold and one at 1e-16, for linus (FastJk) and PySCF, on the same density.

python bench/screening_error.py <mol> <basis> <thresh> [threads]
"""
import json
import os
import pathlib
import subprocess
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
S = ROOT / "bench/scratch"


def main():
    mol, basis, thresh = sys.argv[1], sys.argv[2], sys.argv[3]
    threads = sys.argv[4] if len(sys.argv) > 4 else "32"
    xyz = str(ROOT / f"tests/molecules/{mol}.xyz")
    dm = S / f"{mol}.{basis}.dm.bin"
    res = {}
    for tag, t in (("t", thresh), ("ref", "1e-16")):
        subprocess.run([str(ROOT / "target/release/examples/jk_bench"), xyz, basis, "--dens", str(dm), "--builder", "new",
                        "--threads", threads, "--thresh", t, "--out", str(S / f"se_linus_{tag}")], check=True, capture_output=True)
        subprocess.run([sys.executable, str(ROOT / "bench/pyscf_jk.py"), "jk", xyz, basis, str(dm), threads, t, "1",
                        str(S / f"se_pyscf_{tag}")], check=True, capture_output=True, env=dict(os.environ, OMP_NUM_THREADS=threads))
    d = np.fromfile(dm, dtype="<f8")
    n = int(round(len(d) ** 0.5))
    d = d.reshape(n, n)
    ld = lambda f: np.fromfile(f, dtype="<f8").reshape(n, n)
    for prog in ("linus", "pyscf"):
        dj = ld(S / f"se_{prog}_t.J.bin") - ld(S / f"se_{prog}_ref.J.bin")
        dk = ld(S / f"se_{prog}_t.K.bin") - ld(S / f"se_{prog}_ref.K.bin")
        res[prog] = {"dE_J": float(0.5 * np.einsum("ij,ij", d, dj)), "dE_K": float(-0.25 * np.einsum("ij,ij", d, dk)),
                     "maxabs_J": float(np.abs(dj).max()), "maxabs_K": float(np.abs(dk).max())}
    print(json.dumps({"molecule": mol, "basis": basis, "thresh": float(thresh), **res}))


if __name__ == "__main__":
    main()
