"""Compare linus RI (DfJk) SCF energies with Psi4 1.11 DF-SCF, including range-separated exact
exchange (Psi4's wK: Coulomb 3-center integrals on one side, erf-attenuated on the other,
Coulomb metric).

Run with the reference environment's python (Psi4):
  .refenv/bin/python tests/python/compare_df_psi4.py <molecule.xyz> [basis] [aux] [threads] [charge] [mult]

Cases (no XC functional, so no grid differences):
  rhf       full-range exact exchange
  lr-hf     long-range only:       linus --exx 1,-1,w   <->  Psi4 x_hf alpha=0, beta=1
  sr-hf     short-range only:      linus --exx 0,1,w    <->  Psi4 x_hf alpha=1, beta=-1
  wb97x-x   wB97X exchange mix:    linus --exx 1,-0.842294,0.3 <-> Psi4 alpha=0.157706, beta=0.842294
Prints one JSON line per case with both energies and the difference. Psi4 runs in a temporary
working directory (common.psi4_cwd).
"""
import json, pathlib, subprocess, sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from common import EXAMPLES, psi4_cwd, read_xyz_bohr  # noqa: E402


def read_xyz(path):
    """(symbol, x, y, z) in bohr (linus's constant, CODATA 2022); Psi4 gets bohr."""
    syms, pos = read_xyz_bohr(path)
    return [(s, *p) for s, p in zip(syms, pos)]


def linus(xyz, basis, aux, threads, charge, mult, extra):
    exe = EXAMPLES / "scf"
    cmd = [str(exe), str(xyz), basis, "--jk", "ri", "--aux", aux, "--threads", str(threads), "--charge", str(charge), "--mult", str(mult),
           "--e-conv", "1e-12", "--g-conv", "1e-10"] + extra
    if mult != 1:
        cmd.append("--uhf")
    return json.loads(subprocess.check_output(cmd))


def psi4_energy(atoms, basis, aux, threads, charge, mult, x_hf):
    import psi4

    psi4.core.clean()
    psi4.core.clean_options()
    psi4.set_num_threads(threads)
    psi4.set_memory("16 GB")
    psi4.core.be_quiet()
    geom = f"{charge} {mult}\nunits bohr\nno_reorient\nno_com\nsymmetry c1\n" + "\n".join(f"{s} {x:.12f} {y:.12f} {z:.12f}" for s, x, y, z in atoms)
    psi4.geometry(geom)
    psi4.set_options({"basis": basis, "df_basis_scf": aux, "scf_type": "df", "puream": True, "e_convergence": 1e-12, "d_convergence": 1e-10,
                      "ints_tolerance": 0.0, "df_fitting_condition": 1e-15, "reference": "rhf" if mult == 1 else "uhf", "guess": "sad"})
    if x_hf is None:
        return psi4.energy("scf")
    func = {"name": "x_hf_only", "x_functionals": {}, "x_hf": x_hf, "c_functionals": {}}
    return psi4.energy("scf", dft_functional=func)


def main():
    xyz = pathlib.Path(sys.argv[1]).resolve()
    basis = sys.argv[2] if len(sys.argv) > 2 else "def2-svp"
    aux = sys.argv[3] if len(sys.argv) > 3 else "def2-universal-jkfit"
    threads = int(sys.argv[4]) if len(sys.argv) > 4 else 8
    charge = int(sys.argv[5]) if len(sys.argv) > 5 else 0
    mult = int(sys.argv[6]) if len(sys.argv) > 6 else 1
    atoms = read_xyz(xyz)
    psi4_cwd()  # before importing psi4 (timer.dat names the host)
    w = 0.3
    cases = [
        ("rhf" if mult == 1 else "uhf", [], None),
        ("lr-hf", ["--exx", f"1,-1,{w}"], {"alpha": 0.0, "beta": 1.0, "omega": w}),
        ("sr-hf", ["--exx", f"0,1,{w}"], {"alpha": 1.0, "beta": -1.0, "omega": w}),
        ("wb97x-x", ["--exx", f"1,-0.842294,{w}"], {"alpha": 0.157706, "beta": 0.842294, "omega": w}),
    ]
    for name, extra, xhf in cases:
        el = linus(xyz, basis, aux, threads, charge, mult, extra)
        ep = psi4_energy(atoms, basis, aux, threads, charge, mult, xhf)
        print(json.dumps({"xyz": pathlib.Path(xyz).name, "basis": basis, "aux": aux, "case": name, "linus": el["energy"], "psi4": ep,
                          "diff": el["energy"] - ep, "linus_iter": el["iterations"]}), flush=True)


if __name__ == "__main__":
    main()
