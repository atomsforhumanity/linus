"""Open-shell states of linus against PySCF 2.15 on the same basis functions (and ECPs), DFT grid
and auxiliary basis (bench/results/2026-11-07-openshell-scf).

For each case linus runs (CLI, JSON, with orbitals); PySCF then (1) evaluates its energy and orbital
gradient at linus's converged density (same state, so the energies must agree to the SCF
convergence), and (2) runs its own SCF from linus's orbitals (conv_tol 1e-12, conv_tol_grad 1e-9,
at most --from-cycles cycles, default 50) and (3), unless --no-own, its own SCF from its default guess (minao), so the
state PySCF reaches by itself is recorded too (energy, <S^2>, cycles, whether it converged).
Optionally (--stability) PySCF's internal stability analysis of the state it reached by itself.

Model: linus `pyscf:N` grids are PySCF's level-N grids (small_rho_cutoff 0); jk exact (PySCF
direct, 1e-14 screening) or df (def2-universal-jkfit for both); wB97X-D3BJ = libxc
HYB_GGA_XC_WB97X_V without VV10 (PySCF xc wb97x-v, nlc off) plus D3BJ, which is compared as the
SCF energy without dispersion. With jk df, linus fits the long-range exchange in Psi4's wK form and
PySCF with its own long-range fit, so range-separated DF energies differ by ~1e-7 Eh by
construction (crates/linus-py/tests/test_pyscf.py); compare those with exact J/K.

usage: .refenv/bin/python bench/openshell_pyscf.py --linus BIN --out FILE --cases mol:method:basis:jk:thresholds[,...]
       [--slice ccd5] [--no-own] [--stability]
(cases as in bench/openshell_scf.py)
"""
import argparse, json, pathlib, subprocess, sys, time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))
sys.path.insert(0, str(ROOT / "tests/python"))
from openshell_scf import MOLS, request  # noqa: E402
from ecp_pyscf_check import element_data, pyscf_mol  # noqa: E402
from pyscf import dft, gto, lib, scf  # noqa: E402

BOHR = 0.529177210544  # linus's CODATA 2022 value
XC = {"b3lyp": "b3lyp", "pbe": "pbe,pbe", "tpss": "tpss,tpss", "svwn-rpa": "slater,vwn_rpa", "wb97x-d3bj": "wb97x-v"}


def linus(binary, case, slc):
    req = request(case)
    req["orbitals"] = True
    req["verbose"] = False
    cmd = ["systemd-run", "--user", "--scope", "-q", "-p", "MemoryMax=24G", "-p", "MemorySwapMax=0", "-p", "OOMPolicy=continue",
           str(pathlib.Path.home() / ".local/bin/slice"), slc, binary, "--json"]
    t = time.perf_counter()
    p = subprocess.run(cmd, input=json.dumps(req), capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(p.stderr[-500:])
    d = json.loads(p.stdout)
    d["_wall"] = time.perf_counter() - t
    return req, d


def build(req, case):
    mol_name, method, basis, jk, *_ = case.split(":")
    lines = req["xyz"].splitlines()
    n = int(lines[0])
    atoms = [(l.split()[0], tuple(float(x) / BOHR for x in l.split()[1:4])) for l in lines[2:2 + n]]
    mol = pyscf_mol(atoms, basis, False)
    mol.charge, mol.spin = req["charge"], req["multiplicity"] - 1
    mol.max_memory = 16000
    mol.build()
    if method == "hf":
        mf = scf.UHF(mol)
    else:
        mf = dft.UKS(mol)
        mf.xc = XC[method]
        if method == "wb97x-d3bj":
            mf.nlc = 0
        mf.grids.level = int(req["grid"].split(":")[1])
        mf.small_rho_cutoff = 0.0
    if jk == "df":
        aux = {}
        for s, _ in atoms:
            sh, _e = element_data("def2-universal-jkfit", gto.charge(s))
            aux[s] = [[l] + [[x, c] for x, c in prims] for l, prims in sh]
        mf = mf.density_fit(auxbasis=aux)
    mf.direct_scf_tol = 1e-14
    mf.conv_tol = 1e-12
    mf.conv_tol_grad = 1e-9
    return mol, mf


def state(mf, dm=None):
    """Energy and max |F_ai| (canonical orthonormal basis, as linus) of the PySCF object's
    orbitals, or of the density dm."""
    if dm is None:
        dm = mf.make_rdm1()
    e = mf.energy_tot(dm=dm)
    f = mf.get_fock(dm=dm)
    s = mf.get_ovlp()
    g = 0.0
    for sp in range(2):
        # orbital gradient F D S - S D F in the orthonormal basis
        e_ = f[sp] @ dm[sp] @ s - s @ dm[sp] @ f[sp]
        w, v = np.linalg.eigh(s)
        xo = v / np.sqrt(w)
        g = max(g, np.abs(xo.T @ e_ @ xo).max())
    return e, g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--linus", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cases", required=True)
    ap.add_argument("--slice", default="ccd5")
    ap.add_argument("--no-own", action="store_true")
    ap.add_argument("--stability", action="store_true")
    ap.add_argument("--from-cycles", type=int, default=50, help="cycles of the SCF from linus's orbitals")
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    res = json.loads(out.read_text()) if out.exists() else {"conditions": __doc__.strip(), "pyscf": __import__("pyscf").__version__, "results": {}}
    for case in a.cases.split(","):
        req, d = linus(a.linus, case, a.slice)
        mol, mf = build(req, case)
        e_disp = d["components"]["dispersion"]
        e_lin = d["energy"] - e_disp
        na, nb = mol.nelec
        ca, cb = np.array(d["mo_coefficients_alpha"]), np.array(d["mo_coefficients_beta"])
        occ = np.array([[1.0 if i < na else 0.0 for i in range(ca.shape[1])], [1.0 if i < nb else 0.0 for i in range(cb.shape[1])]])
        mo = np.array([ca, cb])
        dm = mf.make_rdm1(mo, occ)
        e_at, g_at = state(mf, dm)
        r = {"linus": {"energy_scf": e_lin, "dispersion": e_disp, "s2": d["s2"], "converged": d["converged"], "fock_builds": d["scf"]["fock_builds"],
                       "final_gmax": d["scf"]["history"][-1]["gmax"], "stability": d["scf"].get("stability"), "wall": d["_wall"]},
             "pyscf_at_linus_density": {"energy": e_at, "max_fai": g_at, "s2": float(mf.spin_square(mo, mf.get_ovlp())[0]), "de_linus_minus_pyscf": e_lin - e_at}}
        mf.max_cycle = a.from_cycles
        cyc = []
        mf.callback = lambda env: cyc.append(1)
        t = time.perf_counter()
        e1 = mf.kernel(dm0=dm)
        r["pyscf_from_linus"] = {"energy": e1, "converged": bool(mf.converged), "cycles": len(cyc), "s2": float(mf.spin_square()[0]),
                                 "max_fai": state(mf)[1], "de_linus_minus_pyscf": e_lin - e1, "wall": time.perf_counter() - t}
        if not a.no_own:
            mol2, mf2 = build(req, case)
            mf2.max_cycle = 300
            cyc2 = []
            mf2.callback = lambda env: cyc2.append(1)
            t = time.perf_counter()
            e2 = mf2.kernel()
            r["pyscf_own"] = {"energy": e2, "converged": bool(mf2.converged), "cycles": len(cyc2), "s2": float(mf2.spin_square()[0]),
                              "max_fai": state(mf2)[1], "de_linus_minus_pyscf": e_lin - e2, "wall": time.perf_counter() - t}
            if a.stability:
                t = time.perf_counter()
                _, _, stable, _ = mf2.stability(return_status=True)
                r["pyscf_own"]["stable_internal"] = bool(stable)
                r["pyscf_own"]["stability_wall"] = time.perf_counter() - t
        res["results"][case] = r
        out.write_text(json.dumps(res, indent=1))
        own = r.get("pyscf_own", {})
        print(f"{case:42s} linus E {e_lin:.10f} s2 {d['s2']:.5f} builds {d['scf']['fock_builds']} | PySCF at linus D: dE {e_lin - e_at:+.1e} gmax {g_at:.1e}"
              f" | from linus: dE {e_lin - e1:+.1e} ({len(cyc)} cyc, conv {mf.converged})"
              + (f" | own: dE {own['de_linus_minus_pyscf']:+.1e} s2 {own['s2']:.5f} ({own['cycles']} cyc, conv {own['converged']})" if own else ""), flush=True)


if __name__ == "__main__":
    main()
