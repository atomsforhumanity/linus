"""ORCA side of the J/K benchmarks: exact (NoRI NoCOSX) RHF through qccompute.

python bench/orca_jk.py <xyz> <basis> <threads> <thresh> <rundir> [maxiter]

Every SCF iteration rebuilds the full Fock matrix (DirectResetFreq 1), so ORCA's
"Fock matrix formation" time divided by the number of Fock builds is the cost of one
exact J+K build at Thresh = <thresh> (TCut = 0.01 * thresh, ORCA's own recommended ratio,
the same primitive cutoff linus uses). With [maxiter] (large cases), the SCF stops after that
many iterations; ORCA then aborts without its timing table, and the cost per build is the mean
wall time of iterations 2.. from the iteration table (Fock build plus DIIS and diagonalization,
1-decimal resolution; "fock_source": "iterations"). Run with the reference environment's Python
(.refenv: qccompute, Open MPI 4.1.6). Prints one JSON line.
"""
import json
import os
import pathlib
import re
import sys
import time

ORCA_DIR = pathlib.Path(os.environ.get("LINUS_ORCA_DIR", pathlib.Path.home() / "opt" / "orca-6.0.0"))
REFENV = pathlib.Path(sys.executable).resolve().parents[1]


def orca_environment():
    return {
        "PATH": f"{ORCA_DIR}:{REFENV / 'bin'}:{os.environ.get('PATH', '')}",
        "LD_LIBRARY_PATH": f"{ORCA_DIR / 'lib'}:{REFENV / 'lib'}",
        "OMP_NUM_THREADS": "1",
        "OMPI_MCA_btl": "self,vader",
        "OMPI_MCA_btl_vader_single_copy_mechanism": "none",
        "OMPI_MCA_hwloc_base_binding_policy": "none",
        "OMPI_MCA_rmaps_base_oversubscribe": "1",
    }


def main():
    xyz, basis, threads, thresh, rundir = sys.argv[1:6]
    threads, thresh = int(threads), float(thresh)
    maxiter = int(sys.argv[6]) if len(sys.argv) > 6 else 100
    from qccompute import compute
    from qcdata import CalcType, Model, ProgramInput, Structure

    lines = pathlib.Path(xyz).read_text().splitlines()
    n = int(lines[0])
    syms, geom = [], []
    for ln in lines[2:2 + n]:
        f = ln.split()
        syms.append(f[0])
        geom.append([float(x) for x in f[1:4]])
    from qcref import constants
    a2b = 1.0 / float(constants.BOHR_TO_ANGSTROM)
    structure = Structure(symbols=syms, geometry=[[x * a2b for x in r] for r in geom], charge=0, multiplicity=1)
    kw = {
        # MB per rank; 8 GB in total (the node is shared: keep well inside the job's memory cap)
        "maxcore": min(1000, 8000 // max(1, threads)),
        "scf": {"thresh": thresh, "tcut": 0.01 * thresh, "directresetfreq": 1, "maxiter": maxiter,
                "tole": 1e-10, "convforced": True},
    }
    if threads > 1:
        kw["pal"] = threads
    inp = ProgramInput(program="orca", calctype=CalcType.energy, structure=structure,
                       model=Model(method="HF RHF NoRI NoCOSX VeryTightSCF NoAutoStart", basis=basis),
                       keywords=kw)
    rd = pathlib.Path(rundir)
    rd.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    out = compute(inp, scratch_dir=rd, rm_scratch_dir=False, raise_exc=False,
                  executables={"orca": str(ORCA_DIR / "orca")}, environment=orca_environment())
    wall = time.perf_counter() - t0
    log = out.logs or ""
    (rd / "orca.out").write_text(log)
    ok = "ORCA TERMINATED NORMALLY" in log
    fock = re.search(r"Fock matrix formation\s+\.+\s+([\d.]+) sec", log)
    cyc = re.search(r"SCF CONVERGED AFTER\s+(\d+) CYCLES", log)
    e = re.search(r"FINAL SINGLE POINT ENERGY\s+(-?[\d.]+)", log)
    scf_t = re.search(r"Total SCF time: (\d+) days (\d+) hours (\d+) min (\d+) sec", log)
    res = {"program": "orca", "version": "6.0.0", "ok": ok, "threads": threads, "thresh": thresh, "wall": wall,
           "energy": float(e.group(1)) if e else None, "cycles": int(cyc.group(1)) if cyc else None,
           "fock_total": float(fock.group(1)) if fock else None}
    if fock and cyc:
        res["fock_per_build"] = res["fock_total"] / res["cycles"]
        res["fock_source"] = "timing table"
    elif maxiter < 100:
        # iteration table rows: iter, energy, dE, RMSDP, MaxDP, DIIS error, [damping,] time (s)
        rows = re.findall(r"^\s*(\d+)\s+-?\d+\.\d+\s+\S+\s+\S+\s+\S+\s+\S+\s+(?:\S+\s+)?(\d+\.\d+)\s*$", log, re.M)
        times = [float(t) for i, t in rows if int(i) >= 2]
        if times:
            res["ok"] = True
            res["fock_per_build"] = sum(times) / len(times)
            res["iterations_timed"] = len(times)
            res["fock_source"] = "iterations"

    if scf_t:
        d, h, m, s = (int(x) for x in scf_t.groups())
        res["scf_time"] = ((d * 24 + h) * 60 + m) * 60 + s
    print(json.dumps(res))


if __name__ == "__main__":
    main()
