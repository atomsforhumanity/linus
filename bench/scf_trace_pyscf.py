"""PySCF SCF with a per-cycle convergence trace and per-phase wall times (JSON on stdout).

usage: python bench/scf_trace_pyscf.py <xyz> <method> <basis> [--charge Q] [--jk exact|df]
           [--aux A] [--grid L] [--conv-tol E] [--conv-tol-grad G] [--threads N] [--max-memory MB]

Trace per cycle: energy, dE, PySCF's norm_gorb (||g||_2 with g = 2 C_v^T F C_o for RHF), and
linus's measure max|X^T (F D_a S - S D_a F) X| (orthonormal basis, D_a = dm / 2), so the two
programs' criteria can be compared on the same states. Phase times wrap get_jk/get_j/get_k
(J/K), the numint XC call, eig (diagonalization) and get_fock (DIIS extrapolation + Fock
assembly). `fock_builds` counts J/K calls (PySCF's `cycles` excludes the guess build and
its final conv_check build).
"""
import argparse, json, time
import numpy as np
from pyscf import gto, scf, dft, lib

ap = argparse.ArgumentParser()
ap.add_argument("xyz")
ap.add_argument("method")
ap.add_argument("basis")
ap.add_argument("--charge", type=int, default=0)
ap.add_argument("--jk", default="exact")
ap.add_argument("--aux", default="def2-universal-jkfit")
ap.add_argument("--grid", type=int, default=3)
ap.add_argument("--conv-tol", type=float, default=1e-9)
ap.add_argument("--conv-tol-grad", type=float, default=None)
ap.add_argument("--threads", type=int, default=8)
ap.add_argument("--max-memory", type=int, default=8000)
ap.add_argument("--guess", default="minao")
a = ap.parse_args()
lib.num_threads(a.threads)
lines = open(a.xyz).read().splitlines()
n = int(lines[0])
atoms = [(l.split()[0], [float(x) for x in l.split()[1:4]]) for l in lines[2:2 + n]]
t0 = time.perf_counter()
mol = gto.M(atom=atoms, unit="Angstrom", basis=a.basis, charge=a.charge, verbose=0, max_memory=a.max_memory)
hf = a.method.lower() == "hf"
mf = scf.RHF(mol) if hf else dft.RKS(mol)
if not hf:
    mf.xc = a.method
    mf.grids.level = a.grid
if a.jk == "df":
    mf = mf.density_fit(auxbasis=a.aux)
mf.max_memory = a.max_memory
mf.conv_tol = a.conv_tol
if a.conv_tol_grad is not None:
    mf.conv_tol_grad = a.conv_tol_grad
mf.init_guess = a.guess
mf.max_cycle = 200

times = {"jk": 0.0, "xc": 0.0, "diag": 0.0, "fock_diis": 0.0}
counts = {"jk": 0}


def wrap(obj, name, key, count=False):
    f = getattr(obj, name)

    def g(*args, **kw):
        t = time.perf_counter()
        r = f(*args, **kw)
        times[key] += time.perf_counter() - t
        if count:
            counts["jk"] += 1
        return r
    setattr(obj, name, g)


for nm in ("get_jk", "get_j", "get_k"):
    wrap(mf, nm, "jk", count=True)
if not hf:
    wrap(mf._numint, "nr_rks", "xc")
wrap(mf, "eig", "diag")
wrap(mf, "get_fock", "fock_diis")

s = mol.intor("int1e_ovlp")
w, u = np.linalg.eigh(s)
keep = w > 1e-8
x = u[:, keep] / np.sqrt(w[keep])
trace = []
last = [None, time.perf_counter()]


def cb(env):
    f, dm = env["fock"], env["dm"] / 2
    e = x.T @ (f @ dm @ s - s @ dm @ f) @ x
    now = time.perf_counter()
    trace.append(dict(cycle=env["cycle"] + 1, energy=float(env["e_tot"]), de=float(env["e_tot"] - env["last_hf_e"]),
                      norm_gorb=float(env["norm_gorb"]), gmax=float(np.abs(e).max()), grms=float(np.sqrt((e * e).mean())),
                      time=now - last[1]))
    last[1] = now


mf.callback = cb
t1 = time.perf_counter()
e = mf.kernel()
t2 = time.perf_counter()
print(json.dumps(dict(program="pyscf", energy=e, converged=bool(mf.converged), cycles=mf.cycles, fock_builds=counts["jk"],
                      conv_tol=mf.conv_tol, conv_tol_grad=mf.conv_tol_grad, scf_time=t2 - t1, setup_time=t1 - t0,
                      phase_times=times, nbf=mol.nao, trace=trace)))
