"""VV10 functionals: linus energies against Psi4 1.11 at model level (each program's default grids).

  .refenv/bin/python tests/python/compare_vv10_psi4.py <xyz> <basis> <method> [--charge Q] [--mult M] [--threads N]

Psi4: DF-SCF (def2-universal-jkfit), its default XC grid (75 x 302) and VV10 grid
(DFT_VV10_RADIAL_POINTS 50 x DFT_VV10_SPHERICAL_POINTS 146), self-consistent VV10.
linus: jk df with the same auxiliary basis, grid "psi4" (Psi4's XC grid in Psi4's standard
orientation, without its basis-extent point removal) and nlc_grid auto (GridSpec::psi4_vv10, the
same 50 x 146 construction). The grids are not point for point Psi4's, so agreement is at the
grid-error level (~1e-6 Eh); the VV10 energy itself is compared too (Psi4 "DFT VV10 ENERGY",
linus `vv10.energy`). Also runs linus on the converged-grid pair pyscf:7 / pyscf:4 for scale.
Prints one JSON line. Psi4 runs in a temporary working directory (common.psi4_cwd).
"""
import argparse, json, pathlib, sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from common import linus_json, psi4_cwd, read_xyz_bohr  # noqa: E402


def read_xyz(path):
    """(symbol, x, y, z) in bohr (linus's constant, CODATA 2022); Psi4 gets bohr."""
    syms, pos = read_xyz_bohr(path)
    return [(s, *p) for s, p in zip(syms, pos)]


def linus(atoms, a, grid, nlc_grid=None):
    req = {"structure": {"symbols": [s for s, *_ in atoms], "geometry": [list(p) for _, *p in atoms], "charge": a.charge,
                         "multiplicity": a.mult},
           "method": a.method, "basis": a.basis, "jk": {"kind": "df", "aux": "def2-universal-jkfit"}, "grid": {"xc": grid},
           "threads": a.threads, "scf": {"convergence": "verytight"}}
    if nlc_grid:
        req["grid"]["nlc"] = nlc_grid
    d = linus_json(req)
    return d["energy"], d["vv10"]["energy"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("xyz")
    ap.add_argument("basis")
    ap.add_argument("method")
    ap.add_argument("--charge", type=int, default=0)
    ap.add_argument("--mult", type=int, default=1)
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    atoms = read_xyz(a.xyz)
    psi4_cwd("psi4_vv10_")  # before importing psi4 (timer.dat names the host)
    import psi4

    psi4.set_num_threads(a.threads)
    psi4.set_memory("16 GB")
    psi4.core.be_quiet()
    geom = f"{a.charge} {a.mult}\nunits bohr\nno_reorient\nno_com\nsymmetry c1\n" + "\n".join(f"{s} {x:.12f} {y:.12f} {z:.12f}" for s, x, y, z in atoms)
    psi4.geometry(geom)
    psi4.set_options({"basis": a.basis, "df_basis_scf": "def2-universal-jkfit", "scf_type": "df", "puream": True, "e_convergence": 1e-11,
                      "d_convergence": 1e-9, "reference": "rks" if a.mult == 1 else "uks"})
    e_p = psi4.energy(a.method)
    nl_p = psi4.variable("DFT VV10 ENERGY")
    vv10_grid = [psi4.core.get_option("SCF", k) for k in ("DFT_VV10_RADIAL_POINTS", "DFT_VV10_SPHERICAL_POINTS", "DFT_VV10_RHO_CUTOFF", "DFT_VV10_POSTSCF")]
    e_l, nl_l = linus(atoms, a, "psi4")
    e_f, nl_f = linus(atoms, a, "pyscf:7", "pyscf:4")
    print(json.dumps({"xyz": pathlib.Path(a.xyz).name, "basis": a.basis, "method": a.method, "e_psi4": e_p, "e_vv10_psi4": nl_p,
                      "e_linus_psi4grid": e_l, "e_vv10_linus_psi4grid": nl_l, "dE": e_l - e_p, "dE_vv10": nl_l - nl_p,
                      "psi4_vv10_grid_rho_cutoff_postscf": vv10_grid, "e_linus_fine": e_f, "e_vv10_linus_fine": nl_f, "psi4_minus_fine": e_p - e_f}))


if __name__ == "__main__":
    main()
