"""PySCF side of the J/K benchmarks (run in a subprocess so the thread count is fixed).

python bench/pyscf_jk.py density <xyz> <basis> <out.bin>
    converged RHF density (density-fitted SCF, only used as a realistic test density)
python bench/pyscf_jk.py jk <xyz> <basis> <dm.bin> <threads> <tol> <repeat> [out_prefix]
    time scf.hf.get_jk (libcint, integral-direct, 8-fold symmetry, direct_scf_tol=tol)
python bench/pyscf_jk.py rhf <xyz> <basis> <threads> <tol> [conv_tol [conv_tol_grad [full]]]
    full direct RHF energy (PySCF defaults otherwise, conv_tol 1e-10), timed; "full" rebuilds
    the whole Fock matrix every cycle (no incremental builds, for tight reference energies)

The basis comes from the same BSE JSON files linus reads (data/basis), so AO orders match.
Prints one JSON line.
"""
import json
import os
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]


def read_xyz(path):
    lines = pathlib.Path(path).read_text().splitlines()
    n = int(lines[0])
    return [(ln.split()[0], tuple(float(x) for x in ln.split()[1:4])) for ln in lines[2:2 + n]]


def pyscf_basis(name, elements):
    from pyscf import gto
    stem = name.lower().replace("*", "_st_").replace("+", "_pl_").replace("/", "_sl_").replace("(", "_lp_").replace(")", "_rp_").replace(",", "_cm_")
    data = json.loads((ROOT / "data/basis" / f"{stem}.json").read_text())
    out = {}
    for sym in elements:
        z = gto.charge(sym)
        shells = []
        for sh in data["elements"][str(z)]["electron_shells"]:
            exps = [float(e) for e in sh["exponents"]]
            ls = sh["angular_momentum"]
            rows = sh["coefficients"]
            if len(ls) == 1:
                ls = ls * len(rows)
            for l, row in zip(ls, rows):
                prims = [[e, float(c)] for e, c in zip(exps, row) if float(c) != 0.0]
                shells.append([l] + prims)
        out[sym] = sorted(shells, key=lambda s: s[0])
    return out


# linus converts Angstrom with CODATA 2022 (PySCF uses 2018: 0.529177210903); pass bohr so
# both programs see the same geometry (the difference moves energies by up to ~1e-9 Eh)
BOHR_ANGSTROM = 0.529177210544


def make_mol(xyz, basis):
    from pyscf import gto
    atoms = [(s, tuple(x / BOHR_ANGSTROM for x in c)) for s, c in read_xyz(xyz)]
    syms = sorted({a[0] for a in atoms})
    return gto.M(atom=atoms, unit="Bohr", basis=pyscf_basis(basis, syms), cart=False,
                 verbose=0, max_memory=4000)


def main():
    mode = sys.argv[1]
    import numpy as np
    from pyscf import scf, lib
    if mode == "density":
        mol = make_mol(sys.argv[2], sys.argv[3])
        mf = scf.RHF(mol).density_fit()
        mf.conv_tol = 1e-9
        mf.kernel()
        dm = np.ascontiguousarray(mf.make_rdm1())
        dm.astype("<f8").tofile(sys.argv[4])
        print(json.dumps({"nbf": mol.nao, "e_df": mf.e_tot, "converged": bool(mf.converged)}))
    elif mode == "jk":
        xyz, basis, dmf, threads, tol, repeat = sys.argv[2:8]
        mol = make_mol(xyz, basis)
        n = mol.nao
        dm = np.fromfile(dmf, dtype="<f8").reshape(n, n)
        mf = scf.RHF(mol)
        mf.direct_scf_tol = float(tol)
        t0 = time.perf_counter()
        opt = mf.init_direct_scf(mol)
        t_setup = time.perf_counter() - t0
        times = []
        for _ in range(int(repeat)):
            t0 = time.perf_counter()
            vj, vk = scf.hf.get_jk(mol, dm, 1, opt)
            times.append(time.perf_counter() - t0)
        if len(sys.argv) > 8:
            vj.astype("<f8").tofile(sys.argv[8] + ".J.bin")
            vk.astype("<f8").tofile(sys.argv[8] + ".K.bin")
        print(json.dumps({"program": "pyscf", "version": __import__("pyscf").__version__, "nbf": n,
                          "threads": lib.num_threads(), "tol": float(tol), "t_setup": t_setup,
                          "times": times, "median": sorted(times)[len(times) // 2]}))
    elif mode == "rhf":
        xyz, basis, threads, tol = sys.argv[2:6]
        mol = make_mol(xyz, basis)
        mf = scf.RHF(mol)
        mf.direct_scf_tol = float(tol)
        mf.conv_tol = float(sys.argv[6]) if len(sys.argv) > 6 else 1e-10
        if len(sys.argv) > 7:
            mf.conv_tol_grad = float(sys.argv[7])
        if len(sys.argv) > 8 and sys.argv[8] == "full":
            base = mf.get_veff
            mf.get_veff = lambda mol=None, dm=None, dm_last=0, vhf_last=0, hermi=1: base(mol, dm, hermi=hermi)
        mf.max_memory = 0  # never build the 8-fold ERI tensor in memory: integral-direct
        mf.init_guess = "minao"
        mf.max_cycle = 200
        t0 = time.perf_counter()
        e = mf.kernel()
        wall = time.perf_counter() - t0
        print(json.dumps({"program": "pyscf", "energy": e, "converged": bool(mf.converged),
                          "cycles": int(mf.cycles), "threads": lib.num_threads(), "wall": wall}))
    else:
        raise SystemExit(f"unknown mode {mode}")


if __name__ == "__main__":
    main()
